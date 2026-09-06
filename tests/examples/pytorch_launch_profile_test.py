"""Standard-library coverage for optional, warmed CUDA diagnostics."""

from contextlib import closing, redirect_stderr
import io
import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples/framework_workloads/pytorch_smoke"
sys.path.insert(0, str(EXAMPLE))

from analyze_cuda_profile import analyze_database, analyze_sqlite, distribution, interval_summary
from pytorch_smoke import cuda_profile_range
import run_launch_profile as runner


class LaunchProfileTest(unittest.TestCase):
    def test_smoke_trace_flag_combinations_keep_env_options_before_assignments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "lib").mkdir()
            # This fixture never calls CUDA; the loader ignores the empty library.
            (root / "lib/libglimmer_cuda_interceptor.so").touch()
            python = root / "fake-python"
            python.write_text("#!/bin/sh\n"
                              "printf 'trace=%s timing=%s mode=%s\\n' "
                              "\"$GLIMMER_TRACE_KERNEL_LAUNCHES\" "
                              "\"$GLIMMER_TRACE_LAUNCH_TIMINGS\" \"$GLIMMER_SCHEDULER_MODE\"\n")
            python.chmod(0o700)
            for mode in ("native", "observe", "enforce"):
                for trace, timing in ((False, False), (True, False), (False, True), (True, True)):
                    command = ["bash", str(EXAMPLE / "run_pytorch_smoke.sh"), "--mode", mode,
                               "--python", str(python), "--build-dir", str(root),
                               "--output", str(root / "result.csv")]
                    command += ["--trace"] if trace else []
                    command += ["--trace-timings"] if timing else []
                    environment = {key: value for key, value in os.environ.items()
                                   if key != "LD_PRELOAD" and not key.startswith("GLIMMER_")}
                    environment.update(GLIMMER_TRACE_KERNEL_LAUNCHES="stale", GLIMMER_TRACE_LAUNCH_TIMINGS="stale")
                    result = subprocess.run(command, capture_output=True, text=True, timeout=10, env=environment)
                    with self.subTest(mode=mode, trace=trace, timing=timing):
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(f"trace={'1' if trace and mode != 'native' else ''} "
                                      f"timing={'1' if timing else ''} mode={mode if mode != 'native' else ''}",
                                      result.stdout)

    def test_capture_is_disabled_by_default_and_stops_on_failure(self):
        torch = mock.Mock()
        with cuda_profile_range(torch, "cpu", False):
            pass
        torch.assert_not_called()
        self.assertEqual(torch.mock_calls, [])
        with self.assertRaisesRegex(RuntimeError, "workload"):
            with cuda_profile_range(torch, "cuda", True):
                raise RuntimeError("workload failed")
        self.assertEqual(torch.mock_calls, [mock.call.cuda.profiler.start(), mock.call.cuda.profiler.stop()])

    def test_invalid_device_and_failed_start_do_not_stop_an_unstarted_capture(self):
        torch = mock.Mock()
        with self.assertRaises(ValueError):
            with cuda_profile_range(torch, "cpu", True):
                self.fail("invalid capture entered")
        self.assertEqual(torch.mock_calls, [])
        torch.cuda.profiler.start.side_effect = RuntimeError("profiler unavailable")
        with self.assertRaises(RuntimeError):
            with cuda_profile_range(torch, "cuda", True):
                self.fail("failed capture entered")
        torch.cuda.profiler.stop.assert_not_called()

    def test_successful_capture_stops_once(self):
        torch = mock.Mock()
        with cuda_profile_range(torch, "cuda", True):
            torch.cuda.profiler.start.assert_called_once()
            torch.cuda.profiler.stop.assert_not_called()
        torch.cuda.profiler.stop.assert_called_once()

    def test_union_merges_nested_overlapping_and_adjacent_intervals(self):
        result = interval_summary([(40, 50), (0, 10), (5, 15), (15, 20), (6, 8)])
        self.assertEqual(result["span_ns"], 50)
        self.assertEqual(result["covered_ns"], 30)
        self.assertEqual(result["coverage_fraction"], 0.6)
        self.assertEqual(result["internal_gaps"]["total_ns"], 20)
        self.assertEqual(result["internal_gaps"]["count"], 1)
        self.assertEqual(interval_summary([(5, 10)])["internal_gaps"]["count"], 0)
        for intervals in ([], [(1, 1)], [(2, 1)], [(-1, 4)], [(None, 5)], [(None, 5), (1, 4)]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                interval_summary(intervals)

    def test_distribution_and_invalid_durations(self):
        self.assertEqual(distribution([])["count"], 0)
        self.assertEqual(distribution([0, 100])["p50_ns"], 50)
        self.assertEqual(distribution([0, 100])["p99_ns"], 99)
        with self.assertRaises(ValueError):
            distribution([-1])
        with self.assertRaises(ValueError):
            distribution([1, None])

    def database(self):
        database = sqlite3.connect(":memory:")
        self.addCleanup(database.close)
        database.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1, 'cuEventQuery'), (2, 'cudaLaunchKernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(0, 10, 0, 7), (30, 40, 0, 7);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER, deviceId INTEGER, globalPid INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(8, 35, 0, 7);
            CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER(start INTEGER, end INTEGER, nameId INTEGER, globalTid INTEGER, returnValue INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER VALUES(0, 25, 1, 8, 600), (0, 20, 1, 9, 0);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER, end INTEGER, nameId INTEGER, globalTid INTEGER, returnValue INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(0, 30, 2, 8, 0);
        """)
        return database

    def test_real_kernel_data_and_separate_api_sums(self):
        result = analyze_database(self.database())
        self.assertEqual(result["kernels"]["coverage_fraction"], 0.5)
        self.assertEqual(result["execution"]["coverage_fraction"], 1)
        self.assertEqual(result["kernel_duration"]["count"], 2)
        query = next(row for row in result["apis"] if row["name"] == "cuEventQuery")
        self.assertEqual(query["total_ns"], 45)  # Overlap must not be silently treated as elapsed time.
        self.assertEqual(query["threads"], 2)
        self.assertEqual(query["return_values"], {"600": 1, "0": 1})
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(result["trace_quality"], "no_reported_warnings")

    def test_missing_optional_memory_and_driver_tables_are_supported(self):
        database = self.database()
        database.executescript("DROP TABLE CUPTI_ACTIVITY_KIND_MEMCPY; DROP TABLE CUPTI_ACTIVITY_KIND_DRIVER;")
        result = analyze_database(database)
        self.assertEqual(result["activity_counts"]["MEMCPY"], 0)
        self.assertEqual(result["execution"], result["kernels"])

    def test_missing_empty_invalid_and_multi_process_data_fail(self):
        for statement in ("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL",
                          "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL",
                          "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=start",
                          "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=NULL",
                          "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=8 WHERE start=30",
                          "UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET deviceId=1",
                          "DELETE FROM StringIds",
                          "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME; DELETE FROM CUPTI_ACTIVITY_KIND_DRIVER"):
            with self.subTest(statement=statement), self.assertRaises(ValueError):
                database = self.database()
                database.executescript(statement)
                analyze_database(database)

    def test_profiler_errors_fail_and_warnings_are_preserved(self):
        database = self.database()
        database.executescript("""
            CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER, text TEXT);
            CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER, name TEXT);
            INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(2, 'Warning'), (3, 'Error');
            INSERT INTO DIAGNOSTIC_EVENT VALUES(2, 'Limited Unified Memory tracing');
        """)
        self.assertEqual(analyze_database(database)["profiler_diagnostics"][0]["severity"], "Warning")
        self.assertEqual(analyze_database(database)["trace_quality"], "warnings_require_review")
        database.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(3, 'Dropped data')")
        with self.assertRaisesRegex(ValueError, "Dropped data"):
            analyze_database(database)

    def test_sqlite_reader_never_creates_a_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "missing.sqlite"
            with self.assertRaises(sqlite3.Error):
                analyze_sqlite(path)
            self.assertFalse(path.exists())
            with closing(sqlite3.connect(path)) as destination:
                self.database().backup(destination)
            self.assertEqual(analyze_sqlite(path)["kernel_duration"]["count"], 2)

    def test_profile_command_keeps_preload_on_target_and_uses_warmed_range(self):
        for mode in runner.MODES:
            args = runner.parse_args(["--mode", mode, "--role", "training", "--warmup", "6"])
            command = runner.profile_command(args, pathlib.Path("/tmp/profile"), "/tmp/service.sock")
            self.assertIn("--capture-range=cudaProfilerApi", command)
            self.assertIn("--capture-range-end=stop", command)
            self.assertIn("--cuda-trace-all-apis=true", command)
            self.assertEqual(command[-1], "--profile-cuda")
            self.assertEqual(command[command.index("--warmup") + 1], "6")
            self.assertFalse(any("LD_PRELOAD" in item for item in command[:command.index("env")]))
            self.assertEqual(any(item.startswith("LD_PRELOAD=") for item in command), mode != "native")
            self.assertFalse(any("GLIMMER_TRACE" in item for item in command))

    def test_arguments_reject_baseline_analysis_and_unbounded_capture(self):
        for options in (["--analyze-only", "/tmp/results"], ["--repetitions", "2"],
                        ["--duration-seconds", "11"], ["--duration-seconds", "nan"],
                        ["--mode", "bad"], ["--warmup", "-1"]):
            with self.subTest(options=options), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(options)

    def test_output_preserved_on_failure_and_existing_results_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner, "optimized_build", return_value={}), \
                mock.patch.object(runner.shutil, "which", return_value="nsys"), \
                mock.patch.object(runner.subprocess, "run", return_value=mock.Mock(stdout="Nsight test")), \
                mock.patch.object(runner, "artifact_hashes", return_value={}), \
                mock.patch.object(runner, "run_bounded", side_effect=RuntimeError("capture failed")), \
                redirect_stderr(io.StringIO()):
            output = pathlib.Path(temporary) / "output"
            self.assertEqual(runner.main(["--mode", "native", "--output-dir", str(output)]), 1)
            manifest = (output / "manifest.json").read_text()
            self.assertFalse(json.loads(manifest)["collection_complete"])
            self.assertIn("capture failed", (output / "failure.json").read_text())
            self.assertEqual(runner.main(["--mode", "native", "--output-dir", str(output)]), 1)
            self.assertEqual((output / "manifest.json").read_text(), manifest)

    def test_profiled_workload_metadata_and_quota_are_required(self):
        args = runner.parse_args([])
        data = {"validation": "1", "cuda_profile": "1", "role": "training", "iterations": "2",
                "device_uuid": "gpu", "visible_memory_total_bytes": str(args.quota_bytes)}
        with mock.patch.object(runner, "read_workload", return_value=(data, [])):
            self.assertEqual(runner.validate_workload(args, pathlib.Path("/tmp/profile")), data)
        for field, value in (("validation", "0"), ("cuda_profile", "0"), ("role", "inference"),
                             ("iterations", "0"), ("device_uuid", "unknown"), ("visible_memory_total_bytes", "1")):
            with self.subTest(field=field), mock.patch.object(runner, "read_workload", return_value=({**data, field: value}, [])):
                with self.assertRaises(ValueError):
                    runner.validate_workload(args, pathlib.Path("/tmp/profile"))

    def test_success_requires_report_and_unchanged_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner, "optimized_build", return_value={}), \
                mock.patch.object(runner.shutil, "which", return_value="nsys"), \
                mock.patch.object(runner.subprocess, "run", return_value=mock.Mock(stdout="Nsight test")), \
                mock.patch.object(runner, "validate_workload", return_value={}), \
                redirect_stderr(io.StringIO()):
            for variant in ("success", "missing-report", "changed-artifacts"):
                output = pathlib.Path(temporary) / variant

                def collect(*_):
                    if variant != "missing-report":
                        (output / "timeline.nsys-rep").touch()
                    with closing(sqlite3.connect(output / "timeline.sqlite")) as destination:
                        self.database().backup(destination)

                hashes = [{}, {"changed": "hash"}] if variant == "changed-artifacts" else [{}, {}]
                with mock.patch.object(runner, "run_bounded", side_effect=collect), \
                        mock.patch.object(runner, "artifact_hashes", side_effect=hashes):
                    code = runner.main(["--mode", "native", "--output-dir", str(output)])
                success = variant == "success"
                self.assertEqual(code, 0 if success else 1)
                self.assertEqual(json.loads((output / "manifest.json").read_text())["collection_complete"], success)
                self.assertEqual((output / "failure.json").exists(), not success)

    def test_missing_profiler_fails_without_creating_results(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(runner, "optimized_build", return_value={}), \
                mock.patch.object(runner.shutil, "which", return_value=None), redirect_stderr(io.StringIO()):
            output = pathlib.Path(temporary) / "results"
            self.assertEqual(runner.main(["--output-dir", str(output)]), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
