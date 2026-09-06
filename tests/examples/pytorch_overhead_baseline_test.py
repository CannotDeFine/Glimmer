"""Standard-library regression tests for optimized uncontended measurements."""

import contextlib
import csv
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples/framework_workloads/pytorch_smoke"
sys.path.insert(0, str(EXAMPLE))

from measurements import MEASUREMENT_VERSION
from pytorch_smoke import validate_memory_view
from run_overhead_baseline import (MODES, REPOSITORY, ROLES, VERSION, aggregate, check_remote_stats,
                                   collect_results, main, mode_settings, optimized_build, parse_args,
                                   remote_service, stop_service, workload_command)
from run_slo_validation import benchmark_environment


class OverheadBaselineTest(unittest.TestCase):
    def args(self):
        return parse_args([])

    def test_modes_keep_quota_and_launch_settings_matched(self):
        args = self.args()
        self.assertEqual(mode_settings("native", "training", args), {})
        preload = mode_settings("preload", "training", args)
        self.assertEqual(preload["GLIMMER_SCHEDULER_MODE"], "off")
        self.assertNotIn("GLIMMER_MEMORY_LIMIT_BYTES", preload)
        quota = mode_settings("quota", "training", args)
        self.assertEqual(quota["GLIMMER_SCHEDULER_MODE"], "off")
        local = mode_settings("local", "training", args)
        remote = mode_settings("remote", "training", args, "/tmp/test.sock")
        self.assertEqual(remote.pop("GLIMMER_SCHEDULER_CONTROL_SOCKET"), "/tmp/test.sock")
        self.assertEqual(remote, local)
        self.assertEqual(local["GLIMMER_MEMORY_LIMIT_BYTES"], quota["GLIMMER_MEMORY_LIMIT_BYTES"])
        self.assertEqual(local["GLIMMER_SCHEDULER_BATCH_SIZE"], "1")
        self.assertEqual(local["GLIMMER_MAX_CONCURRENT_KERNELS"], "2")
        with self.assertRaises(ValueError):
            mode_settings("remote", "training", args)
        for mode, role in (("bad", "training"), ("native", "bad")):
            with self.assertRaises(ValueError):
                mode_settings(mode, role, args)

    def test_commands_preserve_workload_and_verify_only_quota_modes(self):
        args = self.args()
        for role in ROLES:
            for mode in MODES:
                command = workload_command(mode, role, args, pathlib.Path("/tmp/results"), "/tmp/test.sock")
                self.assertEqual(command[command.index("--device") + 1], "cuda")
                self.assertEqual(command[command.index("--work-units") + 1], "2" if role == "training" else "1")
                self.assertEqual(command[command.index("--batch-size") + 1], str(args.batch_size))
                self.assertEqual(command[command.index("--duration-seconds") + 1], str(args.duration_seconds))
                self.assertEqual("--expected-memory-total-bytes" in command, mode in ("quota", "local", "remote"))
                self.assertFalse(any("TRACE" in item for item in command))

    def test_inherited_preload_and_glimmer_options_do_not_contaminate_native(self):
        with mock.patch.dict("os.environ", {"LD_PRELOAD": "other.so", "LD_DEBUG": "libs",
                             "LD_AUDIT": "audit.so", "GLIMMER_NEW_OPTION": "1",
                             "GLIMMER_SCHEDULER_BATCH_SIZE": "99", "CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            self.assertEqual(benchmark_environment(), {"CUDA_VISIBLE_DEVICES": "0"})

    def test_argument_validation(self):
        for options in (("--duration-seconds", "nan"), ("--duration-seconds", "inf"),
                        ("--duration-seconds", "0"), ("--duration-seconds", "3601"),
                        ("--repetitions", "0"), ("--warmup", "-1"), ("--batch-size", "0"),
                        ("--hidden-size", "0"), ("--training-work-units", "0"),
                        ("--quota-bytes", "0"), ("--quota-bytes", str(2**64))):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(options)

    def write_build(self, root, build_type="RelWithDebInfo", flags="-O2 -g -DNDEBUG", sanitizer="OFF"):
        (root / "CMakeCache.txt").write_text(
            f"CMAKE_BUILD_TYPE:STRING={build_type}\nGLIMMER_ENABLE_SANITIZERS:BOOL={sanitizer}\n")
        entries = [{"directory": str(root), "file": str(REPOSITORY / "src" / module / "test.cc"),
                    "command": f"c++ {flags} -c test.cc"} for module in ("core", "control", "interceptor")]
        (root / "compile_commands.json").write_text(json.dumps(entries))

    def test_build_requires_optimized_unsanitized_runtime_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for build_type in ("Release", "RelWithDebInfo"):
                self.write_build(root, build_type=build_type)
                self.assertEqual(optimized_build(root)["CMAKE_BUILD_TYPE"], build_type)
            for settings in ({"build_type": "Debug"}, {"sanitizer": "ON"},
                             {"flags": "-O2 -O0"}, {"flags": "-g"}, {"flags": "-O2 -fsanitize=address"}):
                self.write_build(root, **settings)
                with self.assertRaises(ValueError):
                    optimized_build(root)
            (root / "compile_commands.json").write_text("[]")
            with self.assertRaises(ValueError):
                optimized_build(root)

    def rows(self):
        return [{"mode": mode, "role": role, "repetition": repetition,
                 "measurement_version": MEASUREMENT_VERSION, "device_uuid": "gpu",
                 "device_index": "0", "torch_version": "torch", "cuda_version": "cuda",
                 "samples": 1000, "mean_ms": 1, "p50_ms": 1, "p95_ms": 1,
                 "p99_ms": 1 if mode == "native" else 2,
                 "steps_per_second": 100 if mode == "native" else 50}
                for mode in MODES for role in ROLES for repetition in (1, 2)]

    def test_aggregation_uses_role_local_native_and_keeps_ranges(self):
        result = aggregate(self.rows(), 2)
        self.assertEqual(len(result), 10)
        for row in result:
            self.assertEqual(row["p99_vs_native"], 1 if row["mode"] == "native" else 2)
            self.assertEqual(row["throughput_vs_native"], 1 if row["mode"] == "native" else 0.5)
            self.assertEqual(row["samples_min"], 1000)

    def test_invalid_or_incomplete_metrics_cannot_pass(self):
        rows = self.rows()
        for bad in (rows[:-1], rows + [rows[0]], [rows[0]] * len(rows)):
            with self.assertRaises(ValueError):
                aggregate(bad, 2)
        for key, value in (("p99_ms", 0), ("steps_per_second", float("nan")),
                           ("mean_ms", float("inf")), ("samples", -1), ("device_uuid", "other"),
                           ("torch_version", "different"), ("measurement_version", "old")):
            bad = self.rows()
            bad[0][key] = value
            with self.assertRaises(ValueError):
                aggregate(bad, 2)

    def test_remote_evidence_requires_completed_and_drained_admissions(self):
        valid = "GLIMMER_TASK_V1 STATS 10 0 0 10 0 0 4294967296 0 0 2 256\n"
        check_remote_stats(valid)
        for index, value in ((0, "BAD"), (2, "0"), (3, "1"), (4, "1"), (5, "9"),
                             (6, "1"), (7, "1"), (9, "1"), (10, "1"), (11, "1"), (5, "-1")):
            fields = valid.split()
            fields[index] = value
            with self.assertRaises(ValueError):
                check_remote_stats(" ".join(fields))
        with self.assertRaises(ValueError):
            check_remote_stats(valid + "extra")

    def test_memory_view_cannot_silently_accept_physical_capacity_or_cpu(self):
        validate_memory_view("cuda", 512, 512)
        for device, visible in (("cuda", 6144), ("cpu", 512)):
            with self.assertRaises(ValueError):
                validate_memory_view(device, visible, 512)

    def test_owned_service_cleanup_escalates_only_after_timeout(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("service", 5), 0]
        stop_service(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        process.reset_mock()
        process.poll.return_value = 0
        stop_service(process)
        process.terminate.assert_not_called()

    def test_service_start_failure_reaps_the_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch("run_overhead_baseline.subprocess.Popen") as popen:
                popen.return_value.poll.return_value = 1
                with self.assertRaises(RuntimeError):
                    with remote_service(self.args(), pathlib.Path(temporary)):
                        self.fail("failed service must not yield")

    def test_service_is_stopped_after_workload_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (mock.patch("run_overhead_baseline.subprocess.Popen") as popen,
                  mock.patch("run_overhead_baseline.subprocess.run"),
                  mock.patch("run_overhead_baseline.pathlib.Path.is_socket", return_value=True)):
                popen.return_value.poll.return_value = None
                with self.assertRaisesRegex(RuntimeError, "workload failure"):
                    with remote_service(self.args(), pathlib.Path(temporary)):
                        raise RuntimeError("workload failure")
                popen.return_value.terminate.assert_called_once()
                popen.return_value.wait.assert_called_once()

    def test_raw_collection_round_trip_and_quota_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            manifest = {"version": VERSION, "collection_complete": True,
                        "arguments": {"repetitions": 1, "duration_seconds": 0.01, "quota_bytes": 512}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            for mode in MODES:
                for role in ROLES:
                    directory = root / f"{mode}-{role}-r1"
                    directory.mkdir()
                    metadata = {"measurement_version": MEASUREMENT_VERSION, "pid": 1, "role": role,
                                "device": "cuda", "device_uuid": "gpu", "device_index": 0,
                                "torch_version": "torch", "cuda_version": "cuda", "validation": 1,
                                "start_ns": 1_000_000, "end_ns": 11_001_000,
                                "measurement_deadline_ns": 11_000_000, "iterations": 1,
                                "latency_target_ms": 0, "visible_memory_total_bytes": 512}
                    (directory / f"{role}.status").write_text("".join(f"{k}={v}\n" for k, v in metadata.items()))
                    sample = {"measurement_version": MEASUREMENT_VERSION, "role": role, "iteration": 1,
                              "iteration_start_ns": 1_000_001, "iteration_end_ns": 2_000_001}
                    with (directory / f"{role}.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=sample.keys())
                        writer.writeheader()
                        writer.writerow(sample)
                    if mode == "remote":
                        (directory / "service-stats.txt").write_text(
                            "GLIMMER_TASK_V1 STATS 10 0 0 10 0 0 4294967296 0 0 2 256\n")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(len(collect_results(root)), 10)
            self.assertTrue((root / "summary.csv").is_file())
            self.assertEqual(len(json.loads((root / "aggregate.json").read_text())), 10)
            metadata_path = root / "quota-training-r1/training.status"
            metadata_path.write_text(metadata_path.read_text().replace("visible_memory_total_bytes=512",
                                                                       "visible_memory_total_bytes=6144"))
            with self.assertRaisesRegex(ValueError, "memory view"):
                collect_results(root)

    def test_analysis_rejects_failed_or_incomplete_collection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "failure.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "failure"):
                collect_results(root)
            (root / "failure.json").unlink()
            (root / "manifest.json").write_text(json.dumps({"version": VERSION, "collection_complete": False}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                collect_results(root)

    def test_existing_results_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            original = root / "manifest.json"
            original.write_text("keep me")
            with mock.patch("run_overhead_baseline.optimized_build", return_value={}):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--output-dir", str(root)]), 1)
            self.assertEqual(original.read_text(), "keep me")
            self.assertFalse((root / "failure.json").exists())


if __name__ == "__main__":
    unittest.main()
