"""Exercise environment setup without downloading wheels or importing PyTorch."""

import json
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples/framework_workloads/pytorch_smoke"
sys.path.insert(0, str(EXAMPLE))
SPEC = importlib.util.spec_from_file_location("pytorch_correctness", EXAMPLE / "pytorch_correctness.py")
CORRECTNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORRECTNESS)
MLP_SPEC = importlib.util.spec_from_file_location("pytorch_mlp_correctness", EXAMPLE / "pytorch_mlp_correctness.py")
MLP_CORRECTNESS = importlib.util.module_from_spec(MLP_SPEC)
MLP_SPEC.loader.exec_module(MLP_CORRECTNESS)

FAKE_PYTHON = '''#!{python}
import json
import os
import pathlib
import shutil
import sys

with open(os.environ["SETUP_TEST_LOG"], "a") as log:
    log.write(json.dumps({{"args": sys.argv[1:], "env": dict(os.environ)}}) + "\\n")
if sys.argv[1:3] == ["-m", "venv"]:
    if os.environ.get("SETUP_TEST_FAIL") == "venv":
        sys.exit(19)
    destination = pathlib.Path(sys.argv[3]) / "bin" / "python"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(__file__, destination)
    destination.chmod(0o755)
elif os.environ.get("SETUP_TEST_FAIL") == "pip":
    sys.exit(23)
elif os.environ.get("SETUP_TEST_FAIL") == "framework" and "--requirement" in sys.argv:
    sys.exit(29)
'''


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glimmer setup ")
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.script = self.root / "setup_pytorch_env.sh"
        shutil.copyfile(EXAMPLE / self.script.name, self.script)
        for profile in ("cu126", "cu132"):
            name = f"requirements-{profile}.txt"
            shutil.copyfile(EXAMPLE / name, self.root / name)
        self.python = self.root / "bootstrap python;literal"
        self.python.write_text(FAKE_PYTHON.format(python=sys.executable))
        self.python.chmod(0o755)
        self.log = self.root / "calls.jsonl"
        self.environment = dict(os.environ, SETUP_TEST_LOG=str(self.log))

    def run_setup(self, *arguments, python=None):
        return subprocess.run(
            ["bash", str(self.script), "--python", str(python or self.python), *arguments],
            env=self.environment, capture_output=True, text=True, timeout=10,
        )

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def check_profile(self, arguments, profile):
        result = self.run_setup(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        venv = self.root / ".venv"
        self.assertEqual(calls[0]["args"], ["-m", "venv", str(venv)])
        common = [
            "-m", "pip", "--cache-dir", str(venv / ".cache/pip"),
            "--timeout", "120", "--retries", "5", "--disable-pip-version-check",
        ]
        self.assertEqual(calls[1]["args"], common + [
            "install", "--index-url", "https://pypi.org/simple", "pip==26.2.1",
        ])
        self.assertEqual(calls[2]["args"], common + [
            "install", "--requirement", str(self.root / f"requirements-{profile}.txt"),
        ])
        self.assertIn(f"status=ok cuda={profile}", result.stdout)

    def test_default_preserves_cuda132_with_literal_space_and_semicolon_paths(self):
        self.check_profile([], "cu132")

    def test_explicit_cuda126_selects_pinned_volta_profile(self):
        self.check_profile(["--cuda", "cu126"], "cu126")
        requirements = (EXAMPLE / "requirements-cu126.txt").read_text().splitlines()
        self.assertEqual(requirements, [
            "--index-url https://pypi.org/simple",
            "--find-links https://download.pytorch.org/whl/cu126/torch/",
            "--find-links https://download.pytorch.org/whl/cu126/torchvision/",
            "torch==2.12.1+cu126", "torchvision==0.27.1+cu126",
        ])

    def test_pip_cannot_inherit_external_target_user_or_index_settings(self):
        for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_LOG",
                    "PYTHONPATH", "PYTHONHOME"):
            self.environment[key] = "untrusted-test-value"
        self.environment["PIP_USER"] = "1"
        self.environment["PIP_CONFIG_FILE"] = "/untrusted/config"
        result = self.run_setup("--cuda", "cu126")
        self.assertEqual(result.returncode, 0, result.stderr)
        environment = self.calls()[1]["env"]
        for key in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TARGET", "PIP_PREFIX", "PIP_ROOT", "PIP_LOG",
                    "PYTHONPATH", "PYTHONHOME"):
            self.assertNotIn(key, environment)
        self.assertEqual(environment["PIP_USER"], "0")
        self.assertEqual(environment["PIP_CONFIG_FILE"], "/dev/null")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_invalid_options_do_not_create_environment(self):
        for arguments in (["--cuda", "cu999"], ["--cuda"], ["--unknown"]):
            with self.subTest(arguments=arguments):
                result = self.run_setup(*arguments)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.log.exists())
                self.assertFalse((self.root / ".venv").exists())

    def test_missing_python_does_not_create_environment(self):
        result = self.run_setup(python=self.root / "missing python")
        self.assertEqual(result.returncode, 1)
        self.assertIn("python executable is not available", result.stderr)
        self.assertFalse(self.log.exists())

    def test_bootstrap_failure_does_not_install_or_report_success(self):
        self.environment["SETUP_TEST_FAIL"] = "venv"
        result = self.run_setup()
        self.assertEqual(result.returncode, 19)
        self.assertEqual(len(self.calls()), 1)
        self.assertNotIn("status=ok", result.stdout)

    def test_install_failure_is_not_reported_as_success(self):
        self.environment["SETUP_TEST_FAIL"] = "pip"
        result = self.run_setup()
        self.assertEqual(result.returncode, 23)
        self.assertEqual(len(self.calls()), 2)
        self.assertNotIn("status=ok", result.stdout)

    def test_framework_install_failure_is_not_reported_as_success(self):
        self.environment["SETUP_TEST_FAIL"] = "framework"
        result = self.run_setup()
        self.assertEqual(result.returncode, 29)
        self.assertEqual(len(self.calls()), 3)
        self.assertNotIn("status=ok", result.stdout)

    def test_help_does_not_create_environment(self):
        result = self.run_setup("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--cuda cu126|cu132", result.stdout)
        self.assertFalse(self.log.exists())


class CorrectnessArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="glimmer-correctness-")
        self.addCleanup(self.temporary.cleanup)
        self.output = pathlib.Path(self.temporary.name) / "nested" / "result.json"
        self.torch = SimpleNamespace(
            __version__="fixture-version", version=SimpleNamespace(cuda="fixture-cuda"),
            cuda=SimpleNamespace(
                is_available=lambda: True, set_device=lambda _: None,
                get_device_name=lambda: "fixture-device", get_device_capability=lambda: (7, 0),
                get_arch_list=lambda: ["sm_70"], synchronize=mock.Mock(),
            ),
            set_num_threads=lambda _: None, manual_seed=lambda _: None,
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace()),
                cudnn=SimpleNamespace(version=lambda: 91002),
            ),
        )

    def run_fixture(self, failure=None, cnn=False):
        names = ("matrix_gradients", "convolution_gradients",
                 "stream_event_and_allocator_order", "graph_replay", "cnn_workload_gradients")
        with mock.patch.dict(sys.modules, torch=self.torch), \
                mock.patch.object(CORRECTNESS.importlib.metadata, "distributions", return_value=[]):
            with mock.patch.multiple(CORRECTNESS, **{
                name: mock.Mock(side_effect=(failure if name == "convolution_gradients" and failure
                                             else lambda torch, checks, label=name: checks.append(label)))
                for name in names
            }):
                return CORRECTNESS.main(["--output", str(self.output)] + (["--cnn-workload"] if cnn else []))

    def test_optional_cnn_checks_require_live_tensor_allocation_recovery(self):
        for after in (4096, 8192):
            self.output = self.output.with_name(f"result-{after}.json")
            self.torch.cuda.empty_cache = mock.Mock()
            self.torch.cuda.memory_allocated = mock.Mock(side_effect=[4096, after])
            self.assertEqual(self.run_fixture(cnn=True), 0 if after == 4096 else 1)
            artifact = json.loads(self.output.read_text())
            self.assertEqual(artifact["cnn_allocator_bytes"], {"before": 4096, "after": after})
            self.assertIn("cnn_workload_gradients", artifact["checks"])
            self.assertEqual("cnn_workload_tensor_release" in artifact["checks"], after == 4096)
            self.assertEqual(self.torch.cuda.empty_cache.call_count, 2)

    def test_correctness_outputs_reject_existing_files_and_symlinks_before_torch(self):
        root = self.output.parent
        root.mkdir(parents=True)
        existing = root / "existing.json"
        existing.write_text("original\n")
        dangling = root / "dangling.json"
        dangling.symlink_to(root / "missing")
        loop = root / "loop.json"
        loop.symlink_to(loop)
        for target in (existing, dangling, loop):
            commands = (
                [str(EXAMPLE / "pytorch_correctness.py"), "--output", str(target)],
                [str(EXAMPLE / "pytorch_mlp_correctness.py"), "--output", str(target),
                 "--snapshot", str(root / "fresh.pt")],
                [str(EXAMPLE / "pytorch_mlp_correctness.py"), "--output", str(root / "fresh.json"),
                 "--snapshot", str(target)],
            )
            for command in commands:
                result = subprocess.run([sys.executable, "-S", *command], capture_output=True,
                                        text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("output already exists", result.stderr)
                self.assertNotIn("No module named 'torch'", result.stderr)
        result = subprocess.run(
            [sys.executable, "-S", str(EXAMPLE / "pytorch_mlp_correctness.py"),
             "--output", str(root / "same"), "--snapshot", str(root / "." / "same")],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("must be different paths", result.stderr)
        self.assertEqual(existing.read_text(), "original\n")
        self.assertTrue(dangling.is_symlink())
        self.assertTrue(loop.is_symlink())

    def test_correctness_writes_do_not_replace_a_racing_file(self):
        original_open = pathlib.Path.open

        def race_open(path, mode="r", *args, **kwargs):
            if path == self.output and mode == "x":
                with original_open(path, "w", encoding="utf-8") as stream:
                    stream.write("racing result\n")
            return original_open(path, mode, *args, **kwargs)

        with mock.patch.object(pathlib.Path, "open", race_open), self.assertRaises(FileExistsError):
            self.run_fixture()
        self.assertEqual(self.output.read_text(), "racing result\n")

        snapshot = self.output.parent / "snapshot.pt"
        fake_torch = mock.Mock()
        fake_torch.save.side_effect = lambda value, stream: stream.write(b"snapshot")
        MLP_CORRECTNESS.save_snapshot(fake_torch, snapshot, {"tensor": 1})
        self.assertEqual(snapshot.read_bytes(), b"snapshot")
        with self.assertRaises(FileExistsError):
            MLP_CORRECTNESS.save_snapshot(fake_torch, snapshot, {})
        self.assertEqual(snapshot.read_bytes(), b"snapshot")

    def test_help_does_not_require_framework_or_create_result(self):
        result = subprocess.run(
            [sys.executable, "-S", str(EXAMPLE / "pytorch_correctness.py"), "--help"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_missing_framework_preserves_failed_artifact(self):
        result = subprocess.run(
            [sys.executable, "-S", str(EXAMPLE / "pytorch_correctness.py"),
             "--output", str(self.output)], capture_output=True, text=True, timeout=10,
            env={key: value for key, value in os.environ.items()
                 if key not in ("PYTHONPATH", "PYTHONHOME")},
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        artifact = json.loads(self.output.read_text())
        self.assertEqual(artifact["status"], "failed")
        self.assertEqual(artifact["checks"], [])
        self.assertIn("ModuleNotFoundError", artifact["error"])

    def test_cuda_unavailable_cannot_be_reported_as_cpu_success(self):
        self.torch.cuda.is_available = lambda: False
        self.assertEqual(self.run_fixture(), 1)
        artifact = json.loads(self.output.read_text())
        self.assertEqual(artifact["status"], "failed")
        self.assertEqual(artifact["checks"], [])
        self.assertIn("CPU fallback", artifact["error"])
        self.torch.cuda.synchronize.assert_not_called()

    def test_partial_checks_are_not_a_passing_result(self):
        self.assertEqual(self.run_fixture(AssertionError("wrong GPU result")), 1)
        artifact = json.loads(self.output.read_text())
        self.assertEqual(artifact["status"], "failed")
        self.assertEqual(artifact["checks"], ["matrix_gradients"])
        self.assertIn("wrong GPU result", artifact["error"])
        self.torch.cuda.synchronize.assert_not_called()

    def test_success_records_environment_after_all_checks_and_device_drain(self):
        self.assertEqual(self.run_fixture(), 0)
        artifact = json.loads(self.output.read_text())
        self.assertEqual(artifact["status"], "ok")
        self.assertEqual(len(artifact["checks"]), 4)
        self.assertEqual(artifact["measurement_kind"], "correctness_only")
        self.assertEqual(artifact["torch_version"], "fixture-version")
        self.assertEqual(artifact["device_capability"], [7, 0])
        self.assertEqual(artifact["compiled_architectures"], ["sm_70"])
        self.torch.cuda.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
