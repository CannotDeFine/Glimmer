"""CPU-only raw-evidence, repetition and summary tests for GEMM analysis."""

import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

PROJECT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ANALYSIS = module(PROJECT / "examples/gemm_comparison/analyze_utilization.py", "analysis")
SUPPORTED_INPUTS = ANALYSIS.inputs
FIXTURES = module(Path(__file__).with_name("gemm_comparison_test.py"), "capture_fixtures")
RUNNER = FIXTURES.RUNNER


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.registry = {}
        self.batch = dict(status="captures_and_timings_require_review", ordinary_repetitions=3,
                          captures={}, timing={})
        self.vendor = self.folder / "vendor.so"
        self.vendor.write_bytes(b"synthetic vendor library, not loaded")
        self.fixtures = FIXTURES.EvidenceTest()
        for name in ANALYSIS.VARIANTS:
            self.create_variant(name)
        self.batch_path = self.folder / "batch.json"
        self.save_batch()
        patch = mock.patch.object(ANALYSIS, "inputs", side_effect=lambda name: self.registry[name])
        patch.start()
        self.addCleanup(patch.stop)
        original_load = ANALYSIS.load_runner

        def fixture_runner(path):
            runner = original_load(path)
            runner.vendor_identities = lambda: {str(self.vendor): RUNNER.digest(self.vendor)}
            return runner

        loader_patch = mock.patch.object(ANALYSIS, "load_runner", side_effect=fixture_runner)
        loader_patch.start()
        self.addCleanup(loader_patch.stop)

    def write_json(self, path, value):
        path.write_text(json.dumps(value))

    def save_batch(self):
        self.write_json(self.batch_path, self.batch)

    def output(self, records):
        return "\n".join(json.dumps(row) for row in records)

    def create_variant(self, name):
        folder = self.folder / name
        folder.mkdir()
        collector = folder / "run_comparison.py"
        shutil.copyfile(PROJECT / "examples/gemm_comparison/run_comparison.py", collector)
        source = folder / "gemm_comparison.cu"
        source.write_text("synthetic fixture source: " + name)
        binary = folder / "glimmer_gemm_comparison"
        binary.write_bytes(name.encode())  # Never executed.
        self.registry[name] = collector, binary
        identity = dict(binary_sha256=RUNNER.digest(binary), source_sha256=RUNNER.digest(source),
                        runner_sha256=RUNNER.digest(collector), cutlass_revision=RUNNER.PINNED_CUTLASS,
                        vendor_identities={str(self.vendor): RUNNER.digest(self.vendor)},
                        gpu_before="0, Tesla V100-PCIE-16GB, GPU-fixture, 560.35.05, 1, 0",
                        gpu_after="0, Tesla V100-PCIE-16GB, GPU-fixture, 560.35.05, 1, 0")
        capture_folder = folder / "capture"
        capture_folder.mkdir()
        profile_records = self.fixtures.records(True)
        profile_records[-1]["compiled_source_sha256"] = identity["source_sha256"]
        if name != "baseline":
            for row in profile_records[:-1]:
                row.update(persistent_audit_specialized=True, normal_kernel_checks=True)
        stdout = capture_folder / "stdout.log"
        stdout.write_text(self.output(profile_records))
        metrics = self.fixtures.compute_csv()
        if name == "baseline":
            rows = RUNNER.metric_inventory(metrics)
            rows = [rows[0]] + [row for operation in range(18)
                               for row in rows[1 + operation * 11:1 + operation * 11 + 7]]
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow(dict(row, ID=str(index)))
            metrics = buffer.getvalue()
        (capture_folder / "native-metrics.csv").write_text(metrics)
        (capture_folder / "native.ncu-rep").write_bytes(b"synthetic profiler bytes")
        capture = dict(identity, status="native_full_sm_coverage_verified", collect_native_sm=True,
                       check_only=True, native_full_sm_verified=True, cutlass_compute_full_sm_verified=True,
                       full_device_coverage_verified=True, coverage_policy="require_all_sms",
                       host_sanitizers=False, device_tool=None, returncode=0, timed_out=False,
                       performance_acceptance=False, application_acceptance=False, priority_acceptance=False,
                       finished_at_unix_s=time.time() - 60, application_replay_passes=1,
                       capture_sha256=RUNNER.digest(capture_folder / "native.ncu-rep"),
                       metrics_sha256=RUNNER.digest(capture_folder / "native-metrics.csv"),
                       stdout_sha256=RUNNER.digest(stdout), native_rows=RUNNER.native_metric_rows(metrics))
        if name != "baseline":
            capture.update(normal_compute_full_sm_verified=True,
                           compute_rows=RUNNER.persistent_metric_rows(metrics, profile_records[:-1]))
        capture_path = capture_folder / "report.json"
        self.write_json(capture_path, capture)
        self.batch["captures"][name] = str(capture_path)
        paths = []
        for repetition in range(3):
            receipt_folder = folder / str(repetition)
            receipt_folder.mkdir()
            records = self.fixtures.records()
            records[-1]["compiled_source_sha256"] = identity["source_sha256"]
            duration = 100.0 if name == "baseline" else 90.0 + repetition
            for row in records[:-1]:
                row["gpu_samples_us"][2] = [duration] * 21
                row["persistent_median_us"] = duration
                if name != "baseline":
                    row.update(persistent_audit_specialized=True, normal_kernel_checks=True)
            (receipt_folder / "stdout.log").write_text(self.output(records))
            report = dict(identity, status="standalone_timing_requires_review", check_only=False,
                          collect_native_sm=False, host_sanitizers=False, device_tool=None,
                          returncode=0, timed_out=False, coverage_policy="require_all_sms",
                          full_device_coverage_verified=True, native_full_sm_verified=True,
                          cutlass_compute_full_sm_verified=True, performance_acceptance=False,
                          application_acceptance=False, priority_acceptance=False, full_card_acceptance=False,
                          command=[str(binary)], environment=dict(ANALYSIS.TIMING_ENVIRONMENT),
                          native_sm_evidence=dict(path=str(capture_path),
                          sha256=RUNNER.digest(capture_path), native_launches=18),
                          wall_time_s=1.5, child_user_cpu_s=1.0, child_system_cpu_s=0.1,
                          finished_at_unix_s=time.time() - 30 + repetition, rows=records[:-1])
            if name != "baseline":
                report["normal_compute_full_sm_verified"] = True
            path = receipt_folder / "report.json"
            self.write_json(path, report)
            paths.append(str(path))
        self.batch["timing"][name] = paths

    def change_timing(self, field, value, raw=False):
        path = Path(self.batch["timing"]["split-audit"][0])
        report = json.loads(path.read_text())
        report[field] = value
        self.write_json(path, report)
        if raw:
            records = report["rows"] + [dict(compiled_source_sha256=report["source_sha256"],
                                            status="ok", cases=9, diagnostic_only=True,
                                            application_acceptance=False, priority_acceptance=False)]
            (path.parent / "stdout.log").write_text(self.output(records))

    def test_complete_batch_is_cpu_only_and_retains_ranges_and_counters(self):
        with mock.patch.object(subprocess, "run", side_effect=AssertionError("no subprocess allowed")), \
                mock.patch.object(subprocess, "Popen", side_effect=AssertionError("no GPU child allowed")):
            result = ANALYSIS.analyze_batch(self.batch_path)
        self.assertEqual(len(result["variants"]), 4)
        baseline, candidate = result["variants"][:2]
        self.assertEqual(len(baseline["full_compute_counters"]), 54)
        self.assertEqual(len(candidate["full_compute_counters"]), 72)
        self.assertIsNone(baseline["full_compute_counters"][0]["minimum_ffma_instructions"])
        self.assertEqual(candidate["cpu_seconds"]["median"], 1.1)
        self.assertEqual(candidate["cases"][0]["timing_us"]["persistent"]["per_run"], [90, 91, 92])
        self.assertEqual(candidate["comparison"][0]["observation"], "nonoverlapping_faster")
        self.assertAlmostEqual(candidate["comparison"][0]["persistent_speedup"], 100 / 91)
        self.assertEqual(candidate["comparison"][0]["native_control_ratio"], 1)
        self.assertAlmostEqual(candidate["cases"][0]["nominal_gemm_gflops_s"]["persistent"],
                               2 * 768 ** 3 / (91 * 1000))
        self.assertIn("not a statistical significance claim", ANALYSIS.render(result))

    def test_duplicate_or_missing_repetitions_reject(self):
        original = list(self.batch["timing"]["split-audit"])
        for paths in (original[:2], [original[0]] * 3):
            self.batch["timing"]["split-audit"] = paths
            self.save_batch()
            with self.assertRaises(ValueError):
                ANALYSIS.analyze_batch(self.batch_path)

    def test_explicit_qualified_pair_cannot_satisfy_full_batch(self):
        self.batch["status"] = "qualified_pair_requires_review"
        self.batch["captures"] = {name: self.batch["captures"][name] for name in ANALYSIS.VARIANTS[:2]}
        self.batch["timing"] = {name: self.batch["timing"][name] for name in ANALYSIS.VARIANTS[:2]}
        self.save_batch()
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)
        result = ANALYSIS.analyze_qualified_pair(self.batch_path)
        self.assertEqual(result["status"], "qualified_pair_comparison_requires_review")
        self.assertEqual(result["pending_variants"], ["64x32", "32x64"])
        self.assertEqual(len(result["variants"]), 2)

    def test_incomplete_or_unknown_batch_rejects(self):
        self.batch["status"] = "failed"
        self.save_batch()
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)
        with self.assertRaises(ValueError):
            SUPPORTED_INPUTS("unknown")

    def test_instrumented_failed_or_partial_timing_rejects(self):
        path = Path(self.batch["timing"]["split-audit"][0])
        original = path.read_text()
        for field, value in (("check_only", True), ("collect_native_sm", True),
                             ("host_sanitizers", True), ("device_tool", "memcheck"),
                             ("timed_out", True), ("returncode", 1), ("returncode", False),
                             ("full_device_coverage_verified", False), ("performance_acceptance", True),
                             ("normal_compute_full_sm_verified", False), ("command", ["sudo", "ncu"]),
                             ("environment", dict(ANALYSIS.TIMING_ENVIRONMENT, LD_PRELOAD="profiler.so"))):
            self.change_timing(field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                ANALYSIS.analyze_batch(self.batch_path)
            path.write_text(original)

    def test_identity_reference_and_cpu_corruption_reject(self):
        path = Path(self.batch["timing"]["split-audit"][0])
        original = path.read_text()
        for field, value in (("source_sha256", "b" * 64), ("binary_sha256", "b" * 64),
                             ("runner_sha256", "b" * 64), ("cutlass_revision", "unknown"),
                             ("native_sm_evidence", dict(path="missing", sha256="wrong", native_launches=18)),
                             ("gpu_after", "0, V100, GPU-other, 560.35.05, 1, 0"),
                             ("child_user_cpu_s", float("nan")), ("child_system_cpu_s", -1),
                             ("wall_time_s", 0), ("finished_at_unix_s", time.time() - 100000)):
            self.change_timing(field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                ANALYSIS.analyze_batch(self.batch_path)
            path.write_text(original)

    def test_summary_and_raw_median_changes_reject(self):
        path = Path(self.batch["timing"]["split-audit"][0])
        original = path.read_text()
        output = path.parent / "stdout.log"
        raw = output.read_text()
        report = json.loads(original)
        report["rows"][0]["persistent_median_us"] = 10
        self.change_timing("rows", report["rows"])
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)
        self.change_timing("rows", report["rows"], raw=True)
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)
        path.write_text(original)
        output.write_text(raw + "\n" + raw)
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)

    def test_missing_normal_checks_reject_even_if_stdout_matches_summary(self):
        path = Path(self.batch["timing"]["split-audit"][0])
        rows = json.loads(path.read_text())["rows"]
        for row in rows:
            del row["persistent_audit_specialized"]
            del row["normal_kernel_checks"]
        self.change_timing("rows", rows, raw=True)
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)

    def test_raw_counter_digest_and_staleness_reject(self):
        path = Path(self.batch["captures"]["split-audit"])
        metrics = path.parent / "native-metrics.csv"
        original = metrics.read_text()
        metrics.write_text(original + "corruption")
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)
        metrics.write_text(original)
        capture = json.loads(path.read_text())
        capture["finished_at_unix_s"] = time.time() - 100000
        self.write_json(path, capture)
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)

    def test_descriptive_classification_preserves_regressions_and_overlap(self):
        baseline = dict(timing_us={mode: dict(median=100, minimum=99, maximum=101) for mode in ANALYSIS.MODES})
        candidate = dict(case="probe", timing_us={mode: dict(median=110, minimum=109, maximum=111)
                                                 for mode in ANALYSIS.MODES})
        self.assertEqual(ANALYSIS.compare_case(baseline, candidate)["observation"], "nonoverlapping_slower")
        candidate["timing_us"]["persistent"].update(median=100, minimum=98, maximum=102)
        self.assertEqual(ANALYSIS.compare_case(baseline, candidate)["observation"], "overlapping_requires_review")

    def test_legacy_compute_summary_excludes_stop_resume_controls(self):
        path = Path(self.batch["captures"]["baseline"])
        collector = ANALYSIS.load_runner(self.registry["baseline"][0])
        rows = ANALYSIS.legacy_compute_counters(collector, path)
        self.assertEqual({int(row["launch_id"]) % 7 for row in rows}, {1, 2, 3})
        metrics_path = path.parent / "native-metrics.csv"
        inventory = RUNNER.metric_inventory(metrics_path.read_text())
        inventory[1]["Kernel Name"], inventory[5]["Kernel Name"] = \
            inventory[5]["Kernel Name"], inventory[1]["Kernel Name"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(inventory[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(inventory)
        metrics_path.write_text(buffer.getvalue())
        with self.assertRaises(ValueError):
            ANALYSIS.legacy_compute_counters(collector, path)

    def test_cross_variant_vendor_mismatch_rejects(self):
        result = ANALYSIS.analyze_batch(self.batch_path)
        result["variants"][-1]["identity"]["vendor_identities"] = {"different-vendor": "b" * 64}
        with mock.patch.object(ANALYSIS, "analyze_variant", side_effect=result["variants"]):
            with self.assertRaises(ValueError):
                ANALYSIS.analyze_batch(self.batch_path)

    def test_cli_render_failure_remains_failed(self):
        output = self.folder / "output"
        with mock.patch.object(sys, "argv", ["analyze", "--batch-report", str(self.batch_path),
                                              "--output-root", str(output)]), \
                mock.patch.object(ANALYSIS, "render", side_effect=OSError("disk failure")):
            self.assertEqual(ANALYSIS.main(), 1)
        report = json.loads(next(output.glob("*/report.json")).read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["full_goal_acceptance"])
        self.assertFalse(report["performance_acceptance"])

    def test_complete_batch_cli_keeps_all_acceptance_false(self):
        output = self.folder / "output"
        with mock.patch.object(sys, "argv", ["analyze", "--batch-report", str(self.batch_path),
                                              "--output-root", str(output)]):
            self.assertEqual(ANALYSIS.main(), 0)
        report_path = next(output.glob("*/report.json"))
        report = json.loads(report_path.read_text())
        self.assertEqual(report["status"], "bounded_comparison_requires_review")
        for flag in ("full_goal_acceptance", "performance_acceptance", "application_acceptance",
                     "priority_acceptance", "saturation_acceptance"):
            self.assertIs(report[flag], False)
        self.assertTrue(report_path.with_name("review.md").is_file())

    def test_cli_output_policy_uses_shared_helper(self):
        project = self.folder / "project"
        resolve = ANALYSIS.resolve_artifact_root

        def resolve_fixture(namespace, **options):
            return resolve(namespace, project_root=project, **options)

        with mock.patch.object(ANALYSIS, "resolve_artifact_root", side_effect=resolve_fixture):
            bad = project / "build/analysis-1"
            with mock.patch.object(sys, "argv", ["analyze", "--batch-report", str(self.batch_path),
                                                  "--output-root", str(bad)]), \
                    mock.patch.object(ANALYSIS, "analyze_batch") as analyze, \
                    mock.patch.object(sys, "stderr", io.StringIO()), \
                    self.assertRaises(SystemExit) as raised:
                ANALYSIS.main()
            self.assertEqual(raised.exception.code, 2)
            analyze.assert_not_called()
            self.assertFalse(project.exists())
            for output in (None, project / "artifacts/runs/explicit-analysis"):
                command = ["analyze", "--batch-report", str(self.batch_path)]
                if output is not None:
                    command += ["--output-root", str(output)]
                with self.subTest(output=output), mock.patch.object(sys, "argv", command):
                    self.assertEqual(ANALYSIS.main(), 0)
                root = output or project / "artifacts/runs/gemm-analysis"
                self.assertTrue(next(root.glob("attempt-*/run.json")).is_file())

    def test_boolean_gpu_samples_are_not_numeric_measurements(self):
        path = Path(self.batch["timing"]["split-audit"][0])
        rows = json.loads(path.read_text())["rows"]
        rows[0]["persistent_median_us"] = True
        rows[0]["gpu_samples_us"][2] = [True] * 21
        self.change_timing("rows", rows, raw=True)
        with self.assertRaises(ValueError):
            ANALYSIS.analyze_batch(self.batch_path)

    def test_baseline_only_cli_never_claims_candidate_comparison(self):
        output = self.folder / "output"
        with mock.patch.object(sys, "argv", ["analyze", "--baseline-timing", *self.batch["timing"]["baseline"],
                                              "--capture-report", self.batch["captures"]["baseline"],
                                              "--output-root", str(output)]):
            self.assertEqual(ANALYSIS.main(), 0)
        report = json.loads(next(output.glob("*/report.json")).read_text())
        self.assertEqual(report["status"], "baseline_only_no_candidate_comparison")
        self.assertFalse(report["performance_acceptance"])
        self.assertFalse(report["full_goal_acceptance"])


if __name__ == "__main__":
    unittest.main()
