"""Regression tests for benchmark accounting, without importing PyTorch."""

import argparse
import csv
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples/framework_workloads/pytorch_smoke"
sys.path.insert(0, str(EXAMPLE))

from measurements import (MEASUREMENT_VERSION, summarize_colocation, summarize_samples, summarize_solo,
                          validate_scheduler_metrics)
from run_slo_validation import (COMPARISON_VERSION, CONFIGURATIONS, SOLO_ROLES, add_solo_baselines,
                                aggregate, benchmark_environment, build_command, write_aggregate)


class MeasurementTest(unittest.TestCase):
    def test_scheduler_failures_cannot_pass_model_validation(self):
        valid = "GLIMMER_TASK_V1 METRICS 10 0 0 10 0 0 4096 0 0 2 256 100 10 200 20"
        validate_scheduler_metrics(valid)
        for index in (3, 4, 6, 7, 9, 10):
            fields = valid.split()
            fields[index] = "1"
            with self.assertRaisesRegex(ValueError, "drained"):
                validate_scheduler_metrics(" ".join(fields))
        for text in (valid.replace("10 0 0 10", "10 0 0 9"), valid + " extra",
                     valid.replace("METRICS", "STATS"), valid.replace("256", "-1"), ""):
            with self.assertRaises(ValueError):
                validate_scheduler_metrics(text)

    def test_wall_clock_throughput_includes_inter_step_gaps(self):
        rows = [{"iteration_start_ns": 1_000_000, "iteration_end_ns": 2_000_000},
                {"iteration_start_ns": 6_000_000, "iteration_end_ns": 7_000_000}]
        result = summarize_samples(rows, 1_000_000, 11_000_000, 0.5)
        self.assertEqual(result["steps_per_second"], 200)
        self.assertEqual(result["mean_ms"], 1)
        self.assertEqual(result["latency_target_miss_ratio"], 1)

    def test_shared_window_excludes_both_crossing_steps(self):
        rows = [{"iteration_start_ns": 1, "iteration_end_ns": 3},
                {"iteration_start_ns": 3, "iteration_end_ns": 5},
                {"iteration_start_ns": 5, "iteration_end_ns": 8}]
        result = summarize_samples(rows, 2, 7, 0)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["excluded_samples"], 2)

    def test_target_uses_full_precision_wall_timestamps(self):
        row = {"iteration_start_ns": 1_000_000, "iteration_end_ns": 2_000_001,
               "cuda_event_ms": 0.1, "elapsed_ms": "1.000"}
        self.assertEqual(summarize_samples([row], 1, 3_000_000, 1)["latency_target_miss_count"], 1)
        row["iteration_end_ns"] -= 1
        self.assertEqual(summarize_samples([row], 1, 3_000_000, 1)["latency_target_miss_count"], 0)

    def test_invalid_and_empty_windows_fail(self):
        for start, end, target in ((2, 1, 0), (1, 2, float("nan")), (1, 2, -1), (1, 2, 0)):
            with self.assertRaises(ValueError):
                summarize_samples([], start, end, target)
        with self.assertRaises(ValueError):
            summarize_samples([{"iteration_start_ns": 3, "iteration_end_ns": 2}], 1, 4, 0)

    def write_pair(self, root, training_start=1_000_000, inference_start=1_000_000,
                   training_end=11_000_000, inference_end=11_000_000, deadline=0):
        for role, begin, end, pid in (("training", training_start, training_end, 1),
                                      ("inference", inference_start, inference_end, 2)):
            record = {"measurement_version": MEASUREMENT_VERSION, "pid": pid, "role": role,
                      "torch_version": "test-torch", "cuda_version": "test-cuda",
                      "device": "cuda", "device_uuid": "test-device", "device_index": 0,
                      "validation": 1, "start_ns": begin, "end_ns": end,
                      "measurement_deadline_ns": deadline, "latency_target_ms": 1,
                      "iterations": 1}
            (root / f"{role}.status").write_text("".join(f"{k}={v}\n" for k,v in record.items()))
            with (root / f"{role}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("measurement_version", "role", "iteration",
                                                            "iteration_start_ns", "iteration_end_ns"))
                writer.writeheader()
                writer.writerow({"measurement_version": MEASUREMENT_VERSION, "role": role, "iteration": 1,
                                 "iteration_start_ns": begin + 1, "iteration_end_ns": begin + 1_000_000})

    def test_short_overlap_cannot_pass_duration_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write_pair(root, training_end=101_000_000)
            with self.assertRaisesRegex(ValueError, "coverage"):
                summarize_colocation(root, 0.95)

    def test_shared_deadline_is_used_instead_of_drain_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write_pair(root, training_end=11_100_000, inference_end=11_200_000, deadline=11_000_000)
            result = summarize_colocation(root, 0.95)
            self.assertEqual(result["window_end_ns"], 11_000_000)
            self.assertEqual(result["training"]["steps_per_second"], 100)
            self.assertEqual(result["inference"]["steps_per_second"], 100)

    def test_failed_identity_version_and_early_exit_are_rejected(self):
        for old, new in (("test-device", "other-device"), ("pid=2", "pid=1"),
                         (MEASUREMENT_VERSION, "legacy"), ("validation=1", "validation=0"),
                         ("device=cuda", "device=cpu"), ("end_ns=11000000", "end_ns=9000000")):
            with tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                self.write_pair(root, deadline=11_000_000)
                status = root / "inference.status"
                status.write_text(status.read_text().replace(old, new))
                with self.assertRaises(ValueError):
                    summarize_colocation(root)

    def test_truncated_csv_is_not_a_successful_measurement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write_pair(root)
            record = root / "training.status"
            record.write_text(record.read_text().replace("iterations=1", "iterations=2"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                summarize_colocation(root)

    def test_profiled_rows_are_not_accepted_as_performance_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write_pair(root, deadline=11_000_000)
            status = root / "training.status"
            status.write_text(status.read_text() + "cuda_profile=1\n")
            with self.assertRaisesRegex(ValueError, "profiled"):
                summarize_solo(root, "training")
            with self.assertRaisesRegex(ValueError, "profiled"):
                summarize_colocation(root)

    def test_solo_uses_deadline_and_the_same_wall_clock_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            self.write_pair(root, deadline=11_000_000, training_end=11_100_000)
            solo = summarize_solo(root, "training")
            paired = summarize_colocation(root, 0.95)["training"]
            for metric in ("mean_ms", "p99_ms", "steps_per_second", "samples", "excluded_samples"):
                self.assertEqual(solo[metric], paired[metric])
            self.assertEqual(solo["window_end_ns"], 11_000_000)
            self.assertEqual(solo["steps_per_second"], 100)

    def test_solo_rejects_missing_deadline_early_exit_and_long_drain(self):
        for deadline, end in ((0, 11_000_000), (11_000_000, 9_000_000), (11_000_000, 101_000_000)):
            with tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                self.write_pair(root, deadline=deadline, training_end=end)
                with self.assertRaises(ValueError):
                    summarize_solo(root, "training")

    def test_csv_role_step_identity_and_framework_mismatch_fail(self):
        for filename, old, new in (("inference.csv", "inference", "training"),
                                   ("inference.csv", ",1,", ",2,"),
                                   ("inference.status", "test-torch", "other-torch")):
            with tempfile.TemporaryDirectory() as temporary:
                root = pathlib.Path(temporary)
                self.write_pair(root)
                path = root / filename
                path.write_text(path.read_text().replace(old, new))
                with self.assertRaises(ValueError):
                    summarize_colocation(root)

    def test_repetition_summary_reports_per_run_quantiles(self):
        rows = [{"configuration": name, "inference_p50_ms": 1, "inference_p95_ms": 2,
                 "inference_p99_ms": p99, "inference_latency_target_miss_ratio": 0.1,
                 "training_steps_per_second": 100}
                for name in CONFIGURATIONS for p99 in (2, 4, 9)]
        result = aggregate(rows)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["inference_p99_ms_median"], 4)
        self.assertEqual(result[0]["inference_p99_ms_min"], 2)
        self.assertEqual(result[0]["inference_p99_ms_max"], 9)
        with self.assertRaises(ValueError):
            aggregate(rows[:3])
        with self.assertRaisesRegex(ValueError, "unequal"):
            aggregate(rows[:-1])

    def test_csv_round_trip_uses_emitted_column_names(self):
        rows = [{"configuration": name, "measurement_version": MEASUREMENT_VERSION,
                 "inference_p50_ms": 1, "inference_p95_ms": 2, "inference_p99_ms": 3,
                 "inference_latency_target_miss_ratio": 0.5, "training_steps_per_second": 10}
                for name in CONFIGURATIONS]
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            with (root / "summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            write_aggregate(root)
            self.assertTrue((root / "aggregate.json").exists())

    def matched_rows(self):
        common = {"measurement_version": MEASUREMENT_VERSION, "device_uuid": "test-device",
                  "device_index": 0, "torch_version": "test-torch", "cuda_version": "test-cuda",
                  "duration_seconds": 10, "tensor_batch_size": 32, "hidden_size": 1024,
                  "training_work_units": 2, "inference_target_ms": 2, "warmup": 5}
        rows = [{**common, "configuration": name, "repetition": repetition,
                 "inference_p50_ms": 1, "inference_p95_ms": 2, "inference_p99_ms": 4,
                 "inference_latency_target_miss_ratio": 0.5, "training_steps_per_second": 50}
                for name in CONFIGURATIONS for repetition in (1, 2)]
        solo = [{**common, "configuration": name, "role": role, "repetition": repetition,
                 "mean_ms": 1, "p50_ms": 1, "p95_ms": 2, "p99_ms": 2, "steps_per_second": 100}
                for name, role in SOLO_ROLES.items() for repetition in (1, 2)]
        return rows, solo

    def test_solo_and_native_ratios_use_different_denominators(self):
        rows, solo = self.matched_rows()
        result = aggregate(rows)
        baselines = add_solo_baselines(result, rows, solo, 2)
        self.assertEqual(len(baselines), 2)
        for row in result:
            self.assertEqual(row["inference_p99_vs_solo"], 2)
            self.assertEqual(row["training_throughput_vs_solo"], 0.5)
            self.assertEqual(row["inference_p99_vs_native"], 1)
            self.assertEqual(row["training_throughput_vs_native"], 1)

    def test_missing_duplicate_or_mismatched_baselines_are_rejected(self):
        for change in ("missing", "duplicate", "device", "batch", "role", "nan", "zero"):
            rows, solo = self.matched_rows()
            if change == "missing":
                solo.pop()
            elif change == "duplicate":
                solo[-1] = solo[0].copy()
            else:
                key, value = {"device": ("device_uuid", "other"), "batch": ("tensor_batch_size", 64),
                              "role": ("role", "other"), "nan": ("p99_ms", float("nan")),
                              "zero": ("steps_per_second", 0)}[change]
                solo[0][key] = value
            with self.subTest(change=change), self.assertRaises(ValueError):
                add_solo_baselines(aggregate(rows), rows, solo, 2)

    def test_complete_comparison_round_trip_and_fail_closed_analysis(self):
        rows, solo = self.matched_rows()
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for name, data in (("summary.csv", rows), ("solo_summary.csv", solo)):
                with (root / name).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
            manifest = {"comparison_version": COMPARISON_VERSION, "arguments": {"repetitions": 2}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            write_aggregate(root)
            self.assertEqual(json.loads((root / "aggregate.json").read_text())[0]["inference_p99_vs_solo"], 2)
            self.assertTrue((root / "solo_aggregate.json").is_file())
            (root / "failure.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "recorded failure"):
                write_aggregate(root)
            (root / "failure.json").unlink()
            (root / "solo_summary.csv").unlink()
            with self.assertRaises(OSError):
                write_aggregate(root)

    def test_commands_match_role_local_workload_and_environment_is_clean(self):
        args = argparse.Namespace(duration_seconds=10, warmup=5, batch_size=32, hidden_size=1024,
                                  training_work_units=2, inference_target_ms=2, python="python3",
                                  build_dir=pathlib.Path("build/cuda-gpu"),
                                  adaptive_slo_target_queue_us=500, adaptive_slo_window=8)
        for name in (*SOLO_ROLES, *CONFIGURATIONS):
            command = build_command(name, args, pathlib.Path("test-output"))
            for option, value in (("--duration-seconds", "10"), ("--warmup", "5"),
                                  ("--batch-size", "32"), ("--hidden-size", "1024")):
                self.assertEqual(command[command.index(option) + 1], value)
            if name in SOLO_ROLES:
                self.assertEqual(command[command.index("--mode") + 1], "native")
                self.assertEqual(command[command.index("--work-units") + 1],
                                 "2" if name == "training-alone" else "1")
            else:
                self.assertEqual(command[command.index("--training-work-units") + 1], "2")
        with mock.patch.dict("os.environ", {"GLIMMER_SCHEDULER_MODE": "enforce", "LD_PRELOAD": "hook.so",
                                            "LD_DEBUG": "all", "CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            self.assertEqual(benchmark_environment(), {"CUDA_VISIBLE_DEVICES": "0"})


if __name__ == "__main__":
    unittest.main()
