"""CPU-only acceptance report/denominator and environment regressions."""

import argparse
import contextlib
import io
import json
import pathlib
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] /
                       "examples/framework_workloads/pytorch_smoke"))
import run_persistent_mlp_acceptance as runner


class AcceptanceTest(unittest.TestCase):
    def profile(self):
        return {"trace_quality": "no_reported_warnings",
                "kernel_duration": {"total_ns": 100, "count": 7},
                "kernel_duration_by_name": {"unknown_total_ns": 0, "total_ns": 100, "groups": [
                    {"name": "glimmer_persistent_sgemm", "total_ns": 70, "count": 2},
                    {"name": "vendor_gelu", "total_ns": 20, "count": 4},
                    {"name": "vendor_update", "total_ns": 10, "count": 1}]}}

    def test_denominator_is_all_actual_kernel_duration(self):
        result = runner.original_fraction(self.profile(), 2)
        self.assertEqual(result["remaining_original_fraction"], 0.3)
        self.assertEqual(result["remaining_original_launch_fraction"], 5 / 7)
        self.assertEqual(result["total_kernel_ns"], 100)

    def test_reject_incomplete_or_misclassified_profiles(self):
        cases = []
        value = self.profile()
        value["trace_quality"] = "warnings_require_review"
        cases.append(value)
        for field, replacement in (("unknown_total_ns", 1), ("total_ns", 99)):
            value = self.profile()
            value["kernel_duration_by_name"][field] = replacement
            cases.append(value)
        for field, replacement in (("name", None), ("name", "__glimmer_binary_slice"),
                                   ("total_ns", -1), ("count", 0)):
            value = self.profile()
            value["kernel_duration_by_name"]["groups"][1][field] = replacement
            cases.append(value)
        value = self.profile()
        value["kernel_duration"]["count"] = 8
        cases.append(value)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.original_fraction(value, 2)
        with self.assertRaises(ValueError):
            runner.original_fraction(self.profile(), 3)

    def test_timing_modes_disable_instrumentation(self):
        build = pathlib.Path("/test/build")
        native = runner.settings(build, False)
        self.assertFalse(any(key.startswith("GLIMMER_") for key in native))
        self.assertNotIn("LD_PRELOAD", native)
        preload = runner.settings(build, True)
        self.assertEqual(preload["GLIMMER_PERSISTENT_BLAS"], "1")
        self.assertNotIn("GLIMMER_PERSISTENT_AUDIT", preload)
        self.assertFalse(any("TRACE" in key for key in preload))
        self.assertEqual(runner.settings(build, True, True)["GLIMMER_TRACE_KERNEL_LAUNCHES"], "1")

    def test_overhead_is_matched_and_keeps_slowdown(self):
        native = {"samples": 20, "mean_ms": 2, "steps_per_second": 500,
                  "process_elapsed_seconds": 5,
                  "identity": dict(zip(runner.IDENTITY, ("gpu", "0", "torch", "cuda")))}
        preload = dict(native, mean_ms=20, steps_per_second=50, process_elapsed_seconds=10)
        result = runner.overhead(native, preload)
        self.assertEqual(result["latency_overhead_percent"], 900)
        self.assertEqual(result["throughput_retention"], 0.1)
        for field, value in (("samples", 19), ("mean_ms", float("nan")),
                             ("steps_per_second", 0), ("identity", {})):
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.overhead(native, dict(preload, **{field: value}))

    def test_native_and_preload_commands_keep_same_workload(self):
        args = argparse.Namespace(build=pathlib.Path("/build"), python=pathlib.Path("/python"))
        native = runner.workload(args, pathlib.Path("/output"), "training", 20, False)
        preload = runner.workload(args, pathlib.Path("/output"), "training", 20, True)
        self.assertEqual(native[native.index("/python"):],
                         preload[preload.index("/python"):-2])
        self.assertNotIn("--profile-cuda", preload)

    def test_failed_preflight_report_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            args = argparse.Namespace(build=root / "missing", output=root / "output")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run(args), 1)
            report = json.loads((args.output / "acceptance.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["error"])
            self.assertEqual(report["commands"], [])
            self.assertFalse(report["formal_performance_acceptance"])
            self.assertIn(
                "Short measurement; formal workload performance acceptance is not established.",
                report["limitations"])
            with self.assertRaises(FileExistsError):
                runner.run(args)

    def test_profile_preloads_only_target_and_supports_installed_nsight(self):
        args = argparse.Namespace(build=pathlib.Path("/build"), python=pathlib.Path("/python"))
        command = runner.profile_command(args, pathlib.Path("/output"), "training", "/nsys")
        target = command.index("/usr/bin/env")
        self.assertFalse(any("LD_PRELOAD" in item for item in command[:target]))
        self.assertTrue(any(item.startswith("LD_PRELOAD=") for item in command[target:]))
        self.assertFalse(any(item.startswith("--capture-range") for item in command))
        self.assertIn("--profile-cuda", command)
        self.assertNotIn("--cuda-trace-all-apis=true", command)
        self.assertNotIn("--cuda-event-trace=false", command)

    def warmed_database(self):
        database = sqlite3.connect(":memory:")
        self.addCleanup(database.close)
        database.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1,'cudaProfilerStart_v4000'),(2,'cudaProfilerStop_v4000'),
                (3,'cudaLaunchKernel'),(4,'glimmer_persistent_sgemm'),(5,'native');
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,
                globalTid INTEGER,returnValue INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(25,30,1,1,0),(80,90,2,1,0),(35,36,3,1,0);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,
                globalPid INTEGER,mangledName INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(10,20,0,1,5),(40,50,0,1,4),(60,70,0,1,5);
        """)
        return database

    def test_warmed_selection_preserves_source_and_complete_denominator(self):
        database = self.warmed_database()
        result = runner.analyze_warmed_database(database)
        self.assertEqual(result["capture"]["whole_process_kernel_count"], 3)
        self.assertEqual(runner.original_fraction(result, 1)["remaining_original_fraction"], 0.5)
        self.assertEqual(database.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0], 3)

    def test_warmed_selection_rejects_invalid_boundaries(self):
        for mutation in ("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId=1",
                         "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(25,30,1,1,0)",
                         "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=2 WHERE nameId=2",
                         "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=31 WHERE start=10",
                         "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET end=81 WHERE start=60"):
            database = self.warmed_database()
            database.execute(mutation)
            database.commit()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.analyze_warmed_database(database)


if __name__ == "__main__":
    unittest.main()
