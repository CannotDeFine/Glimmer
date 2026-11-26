"""Exercise CUDA compiler selection by configuring only; no GPU is used."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


PARSER = argparse.ArgumentParser(add_help=False)
PARSER.add_argument("--cmake", default="cmake")
PARSER.add_argument("--toolkit-root", required=True)
PARSER.add_argument("--expected-nvcc", required=True)
PARSER.add_argument("--source", choices=("root", "standalone"), default="root")
PARSER.add_argument("--cutlass-root")
OPTIONS, TEST_ARGUMENTS = PARSER.parse_known_args()
if OPTIONS.source == "standalone" and not OPTIONS.cutlass_root:
    PARSER.error("--source standalone requires --cutlass-root")
PROJECT = Path(__file__).resolve().parents[2]


class CudaCompilerSelectionTest(unittest.TestCase):
    def configure(self, build, environment, *options):
        source = PROJECT if OPTIONS.source == "root" else PROJECT / "examples/gemm_comparison"
        feature = (["-DGLIMMER_BUILD_CUDA_WORKLOAD=ON"] if OPTIONS.source == "root" else
                   [f"-DGLIMMER_EXPERIMENT_CUTLASS_ROOT={OPTIONS.cutlass_root}"])
        return subprocess.run(
            [OPTIONS.cmake, "-S", str(source), "-B", str(build), "-G", "Unix Makefiles",
             "-DBUILD_TESTING=OFF", *feature,
             "-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF", "-DCMAKE_CUDA_ARCHITECTURES=70",
             f"-DCUDAToolkit_ROOT={OPTIONS.toolkit_root}", *options],
            env=environment, capture_output=True, text=True, timeout=90)

    def test_default_matches_toolkit_and_explicit_choices_are_not_overridden(self):
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"CUDACXX", "LD_PRELOAD", "LD_AUDIT"}}
        with tempfile.TemporaryDirectory(prefix="glimmer-cuda-compiler-") as temporary:
            root = Path(temporary)
            build = root / "default"
            initial = ([f"-DCMAKE_CUDA_COMPILER={OPTIONS.expected_nvcc}"]
                       if OPTIONS.source == "standalone" else [])
            result = self.configure(build, environment, *initial)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cache = (build / "CMakeCache.txt").read_text()
            match = re.search(r"^CMAKE_CUDA_COMPILER:[^=]+=(.+)$", cache, re.MULTILINE)
            self.assertIsNotNone(match, "CUDA compiler missing from cache")
            self.assertEqual(Path(match.group(1)).resolve(), Path(OPTIONS.expected_nvcc).resolve())

            modules = root / "modules"
            modules.mkdir()
            for version, message in (("1.2.3", "Toolkit 1.2.3"),
                                     ("unknown", "Cannot compare NVIDIA CUDA compiler")):
                with self.subTest(toolkit_version=version):
                    (modules / "FindCUDAToolkit.cmake").write_text(
                        'include("${CMAKE_ROOT}/Modules/FindCUDAToolkit.cmake")\n'
                        f'set(CUDAToolkit_VERSION "{version}")\n')
                    result = self.configure(root / f"version-{version}", environment,
                                            f"-DCMAKE_MODULE_PATH={modules}",
                                            f"-DCMAKE_CUDA_COMPILER={OPTIONS.expected_nvcc}")
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn(message, result.stdout + result.stderr)

            missing = str(root / "nonexistent-nvcc")
            for name, options, overrides in (
                    ("cache", [f"-DCMAKE_CUDA_COMPILER={missing}"], {}),
                    ("environment", [], {"CUDACXX": missing})):
                with self.subTest(selection=name):
                    result = self.configure(root / name, {**environment, **overrides}, *options)
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("nonexistent-nvcc", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *TEST_ARGUMENTS])
