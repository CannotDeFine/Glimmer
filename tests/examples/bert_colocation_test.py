"""CPU contracts for paired BERT arrival accounting and process ownership."""

import copy
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "examples/framework_workloads/bert"))
import run_bert_colocation as runner
import bert_colocation_worker as worker
from bert_contract import PROFILE, SEED


class AccountingTest(unittest.TestCase):
    def setUp(self):
        self.schedule = dict(begin_ns=100, period_ns=10, requests=3)
        common = dict(status="ok", schema="bert_periodic_colocation_v1", instrumented=False,
                      dtype="float32", seed=SEED, warmup=1, worker_sha256="worker",
                      assets={"pinned": True}, environment={"device_uuid": "GPU-a"},
                      validation={"checked_steps": 3})
        self.low = dict(common, role="training", profile=PROFILE["training"],
                        measured_begin_ns=80, measured_end_ns=160,
                        records=[dict(start_ns=90, end_ns=110), dict(start_ns=111, end_ns=130),
                                 dict(start_ns=131, end_ns=160)])
        self.high = dict(common, role="inference", profile=PROFILE["inference"],
                         measured_begin_ns=100, measured_end_ns=145,
                         records=[dict(arrival_ns=100, start_ns=100, end_ns=115),
                                  dict(arrival_ns=110, start_ns=115, end_ns=130),
                                  dict(arrival_ns=120, start_ns=130, end_ns=145)])

    def analyze(self):
        return runner.analyze(self.low, self.high, self.schedule)

    def test_queue_inclusive_backlog_and_common_window(self):
        metrics = self.analyze()
        self.assertEqual(metrics["response"]["p99_ms"], 25 / 1e6)
        self.assertEqual(metrics["service"]["p99_ms"], 15 / 1e6)
        self.assertEqual(metrics["queue_and_wakeup"]["p99_ms"], 10 / 1e6)
        self.assertEqual(metrics["training_steps"], 1)
        self.assertEqual(metrics["training_excluded_steps"], 2)
        self.assertEqual(metrics["training_steps_per_second"], 1 / (45 / 1e9))

    def test_offer_window_does_not_end_at_early_last_completion(self):
        self.high["records"] = [dict(arrival_ns=n, start_ns=n, end_ns=n + 1) for n in (100, 110, 120)]
        self.high["measured_end_ns"] = 130
        self.assertEqual(self.analyze()["window_end_ns"], 130)

    def test_reject_coordinated_omission_and_missing_requests(self):
        self.high["records"][1]["arrival_ns"] = 115
        with self.assertRaises(ValueError):
            self.analyze()
        self.high["records"].pop()
        with self.assertRaises(ValueError):
            self.analyze()

    def test_reject_early_training_exit(self):
        self.low["records"][-1]["end_ns"] = 140
        with self.assertRaises(ValueError):
            self.analyze()

    def test_reject_late_training_start(self):
        self.low["records"][0]["start_ns"] = 101
        with self.assertRaises(ValueError):
            self.analyze()

    def test_zero_completed_training_is_retained_not_invented(self):
        self.low["records"] = [dict(start_ns=90, end_ns=160)]
        self.low["validation"] = {"checked_steps": 1}
        self.assertEqual(self.analyze()["training_steps_per_second"], 0)

    def test_completion_and_containment_are_distinct(self):
        result = self.analyze()
        self.assertEqual(result["training_completions_in_window"], 2)
        self.assertEqual(result["training_steps"], 1)
        self.assertFalse(result["training_sufficient_steps"])

    def test_exclusive_inference_uses_identical_request_accounting(self):
        result = runner.analyze_solo(self.high, "inference", self.schedule)
        paired = self.analyze()
        self.assertEqual(result["response"], paired["response"])
        self.assertEqual(result["service"], paired["service"])
        self.assertEqual(result["drain_seconds"], 15 / 1e9)
        self.assertEqual(result["completed_requests"], 3)
        self.assertEqual(result["pending_requests"], 0)
        self.assertNotIn("training_steps", result)

    def test_exclusive_training_excludes_drain_and_late_start(self):
        schedule = dict(begin_ns=89, end_ns=145)
        result = runner.analyze_solo(self.low, "training", schedule)
        self.assertEqual(result["window_begin_ns"], 90)
        self.assertEqual(result["window_end_ns"], 145)
        self.assertEqual(result["training_steps"], 2)
        self.assertEqual(result["training_excluded_steps"], 1)
        self.assertEqual(result["training_steps_per_second"], 2 / (55 / 1e9))
        self.assertNotIn("response", result)
        for schedule in (dict(begin_ns=10, end_ns=145), dict(begin_ns=91, end_ns=145),
                         dict(begin_ns=89, end_ns=200), dict(begin_ns=145, end_ns=89)):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                runner.analyze_solo(self.low, "training", schedule)

    def test_reject_identity_failure_instrumentation_and_overlap(self):
        saved = copy.deepcopy(self.high)
        for key, value in (("status", "failed"), ("instrumented", True), ("dtype", "float16"),
                           ("environment", {"device_uuid": "GPU-b"}), ("worker_sha256", "different"),
                           ("validation", {"checked_steps": 2}), ("warmup", 2)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.high = dict(saved, **{key: value})
                self.analyze()
        self.high = saved
        self.high["records"][1]["start_ns"] = 114
        with self.assertRaises(ValueError):
            self.analyze()

    def test_tail_sample_count_and_nearest_rank(self):
        self.assertFalse(runner.distribution([1] * 999)["p99_sufficient_samples"])
        result = runner.distribution(list(range(1000)))
        self.assertTrue(result["p99_sufficient_samples"])
        self.assertEqual(result["p99_ms"], 989 / 1e6)
        self.assertEqual(result["p95_ms"], 949 / 1e6)

    def test_slo_uses_queue_inclusive_latency_and_every_offered_request(self):
        metrics = runner.analyze(self.low, self.high, dict(self.schedule, slo_ns=20))
        self.assertEqual(metrics["slo"]["met"], 2)
        self.assertEqual(metrics["slo"]["completed_missed"], 1)
        self.assertEqual(metrics["slo"]["attainment_fraction_lower_bound"], 2 / 3)
        self.assertEqual(metrics["slo"]["attainment_fraction_upper_bound"], 2 / 3)
        self.assertTrue(metrics["slo"]["all_offered_resolved"])
        self.assertIsNone(self.analyze()["slo"])
        for threshold in (0, -1, True, 20.0):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                runner.analyze(self.low, self.high, dict(self.schedule, slo_ns=threshold))


class CpuCostTest(unittest.TestCase):
    def setUp(self):
        self.result = {"measured_begin_ns": 2_000_000_000, "measured_end_ns": 3_000_000_000,
                       "cpu": {"observed_begin_ns": 1_000_000_000, "observed_end_ns": 4_000_000_000,
                               "process_begin_ns": 500_000_000, "process_end_ns": 6_500_000_000}}
        self.before = dict(pid=17, start_ticks=99, ticks_per_second=100, user_ticks=10,
                           system_ticks=20, observed_ns=1_000_000_000)
        self.after = dict(self.before, user_ticks=160, system_ticks=70, observed_ns=2_000_000_000)

    def test_process_cpu_includes_threads_and_uses_its_actual_observation_interval(self):
        metrics = runner.worker_cpu_cost(self.result)
        self.assertEqual(metrics["process_cpu_seconds"], 6)
        self.assertEqual(metrics["observation_seconds"], 3)
        self.assertEqual(metrics["mean_logical_cpus"], 2)
        service = runner.service_cpu_cost(self.before, self.after)
        self.assertEqual(service["user_seconds"], 1.5)
        self.assertEqual(service["system_seconds"], .5)
        self.assertEqual(service["mean_logical_cpus"], 2)

    def test_invalid_worker_counter_or_uncovered_interval_rejects(self):
        for key, value in (("observed_begin_ns", True), ("observed_begin_ns", 2_000_000_001),
                           ("observed_end_ns", 2_999_999_999), ("process_begin_ns", -1),
                           ("process_end_ns", 499_999_999)):
            result = copy.deepcopy(self.result)
            result["cpu"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.worker_cpu_cost(result)

    def test_service_pid_reuse_clock_change_counter_rollback_or_bad_interval_rejects(self):
        for key, value in (("pid", 18), ("start_ticks", 100), ("ticks_per_second", 1000),
                           ("user_ticks", 9), ("system_ticks", True), ("observed_ns", 1_000_000_000)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.service_cpu_cost(self.before, dict(self.after, **{key: value}))

    def test_live_process_stat_parser_retains_pid_and_creation_identity(self):
        fields = ["S"] + ["0"] * 25
        fields[11], fields[12], fields[19] = "13", "29", "97"
        text = "17 (name with ) spaces) " + " ".join(fields)
        with mock.patch.object(runner.pathlib.Path, "read_text", return_value=text), \
                mock.patch.object(runner.os, "sysconf", return_value=100):
            sample = runner.process_cpu_snapshot(17)
            self.assertEqual((sample["pid"], sample["start_ticks"], sample["user_ticks"], sample["system_ticks"]),
                             (17, 97, 13, 29))


class RunnerTest(unittest.TestCase):
    def test_formal_comparison_is_blocked_before_gpu_or_process_start(self):
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(output=pathlib.Path(directory) / "evidence", role="both",
                modes=list(runner.MODES), requests=3, period_ms=1, burst_size=1, drain_seconds=1,
                completion_grace_seconds=1, timeout=600, training_seconds=1, warmup=1, repetitions=1)
            with mock.patch.object(runner, "optimized_build") as build, \
                    mock.patch.object(runner, "collect") as collect:
                self.assertEqual(runner.run(args), 1)
                build.assert_not_called()
                collect.assert_not_called()
            report = json.loads((args.output / "comparison.json").read_text())
            self.assertFalse(report["full_card_acceptance"])
            self.assertFalse(report["diagnostic_only"])
            self.assertIn("full-device qualification is pending", report["error"])

    def test_clean_modes_and_priorities(self):
        with mock.patch.dict(os.environ, {"GLIMMER_PERSISTENT_AUDIT": "1", "LD_PRELOAD": "bad",
                             "NSYS_INJECT": "bad", "CUBLAS_LOGINFO_DBG": "1", "TOKEN": "secret",
                             "GLIMMER_PERSISTENT_WORKER_BLOCKS": "3"}):
            native = runner.child_environment("native", pathlib.Path("/build"), "training", "/socket")
            low = runner.child_environment("priority-persistent", pathlib.Path("/build"), "training", "/socket")
            high = runner.child_environment("priority-persistent", pathlib.Path("/build"), "inference", "/socket")
        self.assertNotIn("LD_PRELOAD", native)
        self.assertFalse(any(key.startswith("GLIMMER_") for key in native))
        for env in (native, low, high):
            for key in ("GLIMMER_PERSISTENT_AUDIT", "NSYS_INJECT", "CUBLAS_LOGINFO_DBG", "TOKEN",
                        "GLIMMER_TRACE_KERNEL_LAUNCHES", "GLIMMER_TRACE_LAUNCH_TIMINGS"):
                self.assertNotIn(key, env)
        self.assertEqual(low["GLIMMER_SCHEDULER_PRIORITY"], "10")
        self.assertEqual(low["GLIMMER_PERSISTENT_WORKER_BLOCKS"], "320")
        self.assertEqual(high["GLIMMER_PERSISTENT_WORKER_BLOCKS"], "320")
        self.assertEqual(high["GLIMMER_SCHEDULER_PRIORITY"], "100")
        self.assertEqual(low["GLIMMER_SCHEDULER_CONTROL_SOCKET"], high["GLIMMER_SCHEDULER_CONTROL_SOCKET"])

    def test_foreign_gpu_process_is_rejected(self):
        with mock.patch.object(runner, "gpu_processes", return_value=[dict(pid=22, gpu_uuid="GPU-a")]):
            with self.assertRaises(RuntimeError):
                runner.check_gpu({11}, "GPU-a")
            runner.check_gpu({22}, "GPU-a")

    def test_pytorch_bare_uuid_cannot_hide_a_foreign_process(self):
        with mock.patch.object(runner, "gpu_processes", return_value=[dict(pid=22, gpu_uuid="GPU-a")]):
            with self.assertRaises(RuntimeError):
                runner.check_gpu({11}, "a")
            runner.check_gpu({22}, "a")
            runner.check_gpu({11}, "b")

    def test_equal_control_changes_only_inference_priority(self):
        for role in PROFILE:
            equal = runner.child_environment("equal-persistent", pathlib.Path("/build"), role, "/socket")
            priority = runner.child_environment("priority-persistent", pathlib.Path("/build"), role, "/socket")
            if role == "inference":
                self.assertEqual(priority.pop("GLIMMER_SCHEDULER_PRIORITY"), "100")
                self.assertEqual(equal.pop("GLIMMER_SCHEDULER_PRIORITY"), "10")
            self.assertEqual(equal, priority)
        for mode, role in (("typo", "inference"), ("native", "typo")):
            with self.assertRaises(ValueError):
                runner.child_environment(mode, pathlib.Path("/build"), role, "/socket")

    def test_exclusive_and_paired_process_ownership(self):
        for selection, mode in (("training", "native"), ("training", "priority-persistent"),
                                ("inference", "native"), ("inference", "priority-persistent"),
                                ("both", "native"), ("both", "equal-persistent"), ("both", "priority-persistent")):
            with self.subTest(role=selection, mode=mode), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                args = types.SimpleNamespace(build=root, python=root / "python", assets=root, export=root,
                    warmup=1, timeout=600, role=selection, requests=3, period_ms=1, training_seconds=1)
                roles = tuple(PROFILE) if selection == "both" else (selection,)
                children = []

                def spawn(command, **kwargs):
                    role = command[command.index("--role") + 1] if "--role" in command else "service"
                    process = mock.Mock(pid=100 + len(children), returncode=0)
                    process.poll.return_value = 0 if role == "inference" else None
                    process.wait.return_value = 0
                    children.append((role, process))
                    return process

                def ready(path, processes, pid, **kwargs):
                    role = path.name.split(".")[0]
                    target = root / "trial" / role
                    target.mkdir()
                    (target / "result.json").write_text(json.dumps({"status": "ok"}))
                    return dict(pid=pid, ready_ns=1, device_uuid="a")

                metric_text = "GLIMMER_TASK_V1 METRICS 1 0 0 1 0 0 12884901888 0 0 1 256 1 1 1 1"
                cpu = dict(pid=100, start_ticks=99, user_ticks=10, system_ticks=1,
                           ticks_per_second=100, observed_ns=1_000_000_000)
                with mock.patch.object(runner, "check_gpu", return_value={}), \
                        mock.patch.object(runner, "process_cpu_snapshot",
                            side_effect=[cpu, dict(cpu, observed_ns=2_000_000_000)]), \
                        mock.patch.object(runner, "read_ready", side_effect=ready), \
                        mock.patch.object(runner, "wait_until"), \
                        mock.patch.object(runner, "stop_group") as stop, \
                        mock.patch.object(runner, "analyze", return_value={}), \
                        mock.patch.object(runner, "analyze_solo", return_value={}), \
                        mock.patch.object(runner, "verify_progress", return_value={}), \
                        mock.patch.object(runner.subprocess, "Popen", side_effect=spawn), \
                        mock.patch.object(runner.subprocess, "run", return_value=types.SimpleNamespace(stdout=metric_text)), \
                        mock.patch.object(runner.time, "monotonic_ns", side_effect=[1_000_000_000, 8_000_000_000]):
                    report = runner.collect(args, mode, root / "trial")
                self.assertEqual(report["status"], "ok")
                expected = ([] if mode == "native" else ["service"]) + list(roles)
                self.assertEqual([role for role, _ in children], expected)
                self.assertEqual(stop.call_args_list, [mock.call(child) for _, child in reversed(children)])
                self.assertEqual(set(report["result_sha256"]), set(roles))
                self.assertEqual(report.get("scheduler_drained", False), mode != "native")
                self.assertEqual(report["schedule"]["begin_ns"], 1_000_000_000 + runner.START_LEAD_NS)
                if selection != "training":
                    self.assertIs(report["schedule"]["preparation_required"], True)

    def test_failed_partial_setup_preserves_evidence_and_stops_owned_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            args = types.SimpleNamespace(build=root, python=root / "python", assets=root, export=root,
                                         warmup=1, timeout=600)
            child = mock.Mock(pid=123)
            with mock.patch.object(runner, "check_gpu", return_value={}), \
                    mock.patch.object(runner.subprocess, "Popen", return_value=child), \
                    mock.patch.object(runner, "read_ready", side_effect=RuntimeError("setup failed")), \
                    mock.patch.object(runner, "stop_group") as stop:
                with self.assertRaisesRegex(RuntimeError, "setup failed"):
                    runner.collect(args, "native", root / "trial")
                stop.assert_called_once_with(child)
            report = json.loads((root / "trial/trial.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(report["commands"]), 1)

    def test_arrivals_are_fixed_even_when_service_is_slow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "start.json"
            worker.publish(path, dict(begin_ns=100, period_ns=10, requests=3))
            with mock.patch.object(worker, "sleep_until") as sleep, \
                    mock.patch.object(worker.time, "monotonic_ns", side_effect=[100, 115, 115, 130, 130, 145]):
                records, _, begin, end = worker.measure_inference(lambda: "result", path)
            self.assertEqual([row["arrival_ns"] for row in records], [100, 110, 120])
            self.assertEqual((begin, end), (100, 145))
            self.assertEqual(sleep.call_args_list, [mock.call(100), mock.call(110), mock.call(120), mock.call(145)])


class BoundedArrivalTest(unittest.TestCase):
    def setUp(self):
        self.trace = worker.make_trace(3, 10)
        self.schedule = dict(begin_ns=100, trace=self.trace, trace_sha256=worker.trace_digest(self.trace), drain_ns=20)

    def result(self, records, end):
        return dict(status="ok", schema="bert_bounded_arrival_v2", role="inference", instrumented=False,
                    profile=PROFILE["inference"], dtype="float32", seed=SEED, environment={"device_uuid": "a"},
                    records=records, validation={"checked_steps": len(records)},
                    measured_begin_ns=100, measured_end_ns=end)

    def test_burst_preserves_average_rate_and_window(self):
        trace = worker.make_trace(5, 10, 2)
        self.assertEqual(trace["offsets_ns"], [0, 0, 20, 20, 40])
        self.assertEqual(trace["duration_ns"], 50)
        self.assertEqual(worker.make_trace(5, 10)["duration_ns"], 50)

    def test_trace_parameters_and_tampering_are_rejected(self):
        for args in ((0, 10), (3, 0), (3, 10, 4), (True, 10), (1_000_001, 10)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                worker.make_trace(*args)
        for field, value in (("offsets_ns", [0, 10, 21]), ("offsets_ns", [0.0, 10, 20]),
                             ("duration_ns", 31), ("schema", "wrong")):
            trace = {**self.trace, field: value}
            schedule = {**self.schedule, "trace": trace, "trace_sha256": worker.trace_digest(trace)}
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                worker.schedule_spec(schedule)
        with self.assertRaises(ValueError):
            worker.schedule_spec({**self.schedule, "trace_sha256": "bad"})
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.schedule_spec({**self.schedule, "preparation_required": value})

    def test_partial_requests_do_not_become_full_p99(self):
        records = [dict(arrival_ns=100, start_ns=100, end_ns=160)]
        metrics = runner.inference_metrics(self.result(records, 160), self.schedule)
        self.assertIsNone(metrics["response"])
        self.assertEqual(metrics["completed_requests"], 1)
        self.assertEqual(metrics["pending_requests"], 2)
        self.assertEqual(metrics["at_drain_deadline"]["completed_requests"], 0)
        self.assertEqual(metrics["at_drain_deadline"]["inflight_requests"], 1)
        self.assertEqual(metrics["at_drain_deadline"]["queued_requests"], 2)
        self.assertEqual(metrics["completed_response"]["mean_ms"], 60 / 1e6)
        self.assertEqual(metrics["response_lower_bound"]["p99_ms"], 60 / 1e6)
        self.assertEqual(metrics["window_end_ns"], 130)
        self.assertEqual(metrics["observation_end_ns"], 160)

    def test_zero_completions_retains_every_arrival(self):
        metrics = runner.inference_metrics(self.result([], 151), self.schedule)
        self.assertIsNone(metrics["response"])
        self.assertIsNone(metrics["completed_response"])
        self.assertIsNone(metrics["service"])
        self.assertEqual(metrics["pending_requests"], 3)
        self.assertEqual(metrics["response_lower_bound"]["p99_ms"], 51 / 1e6)

    def test_incomplete_workload_slo_never_uses_completed_only_denominator(self):
        result = self.result([dict(arrival_ns=100, start_ns=100, end_ns=105)], 151)
        metrics = runner.inference_metrics(result, dict(self.schedule, slo_ns=100))
        self.assertIsNone(metrics["response"])
        self.assertEqual(metrics["slo"]["met"], 1)
        self.assertEqual(metrics["slo"]["offered_requests"], 3)
        self.assertEqual(metrics["slo"]["pending_unresolved"], 2)
        self.assertEqual(metrics["slo"]["attainment_fraction_lower_bound"], 1 / 3)
        self.assertEqual(metrics["slo"]["attainment_fraction_upper_bound"], 1)
        self.assertFalse(metrics["slo"]["all_offered_resolved"])
        expired = runner.inference_metrics(result, dict(self.schedule, slo_ns=20))["slo"]
        self.assertEqual(expired["pending_expired"], 2)
        self.assertTrue(expired["all_offered_resolved"])
        self.assertEqual(expired["attainment_fraction_upper_bound"], 1 / 3)

    def test_new_worker_schema_requires_cpu_evidence_even_with_no_completed_requests(self):
        result = self.result([], 151)
        result["schema"] = "bert_bounded_arrival_v3"
        with self.assertRaises(KeyError):
            runner.validate_worker(result, "inference")
        result["cpu"] = dict(observed_begin_ns=90, observed_end_ns=152,
                             process_begin_ns=0, process_end_ns=5)
        runner.validate_worker(result, "inference")

    def test_all_completed_after_cutoff_has_full_latency_and_missed_deadline(self):
        records = [dict(arrival_ns=100 + i * 10, start_ns=100 + i * 15, end_ns=end)
                   for i, end in enumerate((115, 130, 160))]
        metrics = runner.inference_metrics(self.result(records, 160), self.schedule)
        self.assertTrue(metrics["workload_completed"])
        self.assertTrue(metrics["deadline_exceeded"])
        self.assertEqual(metrics["response"]["samples"], 3)
        self.assertEqual(metrics["at_drain_deadline"]["pending_requests"], 1)

    def test_reject_missing_requests_before_deadline_and_late_start(self):
        for records, end in (([], 140), ([dict(arrival_ns=100, start_ns=150, end_ns=160)], 160),
                             ([dict(arrival_ns=101, start_ns=101, end_ns=160)], 160)):
            with self.subTest(records=records), self.assertRaises(ValueError):
                runner.inference_metrics(self.result(records, end), self.schedule)

    def test_worker_stops_at_request_boundary_and_journals_inflight_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "start.json"
            worker.publish(path, self.schedule)
            events = []
            execute = mock.Mock(return_value="valid")
            with mock.patch.object(worker, "sleep_until"), \
                    mock.patch.object(worker.time, "monotonic_ns", side_effect=[100, 160, 161]):
                records, values, begin, end = worker.measure_inference(execute, path, events.append)
            execute.assert_called_once()
            self.assertEqual((len(records), values, begin, end), (1, ["valid"], 100, 161))
            self.assertEqual([row["phase"] for row in events], ["begin", "start", "complete", "stop"])

    def test_burst_replay_does_not_reset_late_arrivals(self):
        trace = worker.make_trace(3, 10, 3)
        schedule = {**self.schedule, "trace": trace, "trace_sha256": worker.trace_digest(trace)}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "start.json"
            worker.publish(path, schedule)
            with mock.patch.object(worker, "sleep_until") as sleep, \
                    mock.patch.object(worker.time, "monotonic_ns", side_effect=[100, 110, 110, 120, 120, 135]):
                records, _, _, _ = worker.measure_inference(lambda: None, path)
            self.assertEqual([row["arrival_ns"] for row in records], [100, 100, 100])
            self.assertEqual(sleep.call_args_list, [mock.call(100), mock.call(100), mock.call(100), mock.call(135)])

    def test_preparation_must_finish_strictly_before_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "start.json"
            worker.publish(path, {**self.schedule, "preparation_required": True})
            for prepared in (100, 101):
                execute, emit = mock.Mock(), mock.Mock()
                with mock.patch.object(worker.time, "monotonic_ns", return_value=prepared), \
                        self.assertRaisesRegex(ValueError, "preparation missed"):
                    worker.measure_inference(execute, path, emit)
                execute.assert_not_called()
                emit.assert_not_called()
            events = []
            with mock.patch.object(worker, "sleep_until"), \
                    mock.patch.object(worker.time, "monotonic_ns", side_effect=[99, 100, 160, 161]):
                worker.measure_inference(lambda: None, path, events.append)
            self.assertEqual(events[0]["prepared_ns"], 99)
            self.assertEqual(events[0]["begin_ns"], 100)


class ProgressJournalTest(unittest.TestCase):
    def setUp(self):
        self.header = dict(phase="begin", schema="bert_progress_v1", role="inference", pid=123, begin_ns=100)
        self.events = [self.header, dict(phase="start", index=0, arrival_ns=100, start_ns=101),
                       dict(phase="complete", index=0, arrival_ns=100, start_ns=101, end_ns=110),
                       dict(phase="stop", end_ns=130, completed=1)]

    def read(self, events, tail=""):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "progress.jsonl"
            path.write_text("".join(json.dumps(event) + "\n" for event in events) + tail)
            return runner.read_progress(path)

    def test_complete_and_interrupted_prefixes(self):
        full = self.read(self.events)
        self.assertEqual(full["stopped_ns"], 130)
        self.assertEqual(full["records"], [dict(arrival_ns=100, start_ns=101, end_ns=110)])
        partial = self.read(self.events[:2], '{"phase":"complete"')
        self.assertTrue(partial["truncated_tail"])
        self.assertEqual(partial["records"], [])
        self.assertEqual(partial["inflight"], dict(arrival_ns=100, start_ns=101))

    def test_reject_duplicate_missing_or_overlapping_events(self):
        variants = [self.events + [self.events[-1]], [self.header, self.events[2]],
                    self.events[:2] + [self.events[1]], self.events[:2] + [self.events[-1]],
                    self.events[:2] + [dict(self.events[2], index=1)],
                    self.events[:2] + [dict(self.events[2], start_ns=102)],
                    self.events[:3] + [dict(self.events[-1], completed=2)],
                    [self.header, dict(self.events[1], start_ns=99)]]
        for events in variants:
            with self.subTest(events=events), self.assertRaises(ValueError):
                self.read(events)
        with self.assertRaises(ValueError):
            self.read(self.events[:2], '{"broken"\n')

    def test_binding_checks_and_failed_partial_are_not_promoted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            folder = root / "inference"
            folder.mkdir()
            trace = worker.make_trace(3, 10)
            schedule = dict(begin_ns=100, trace=trace, trace_sha256=worker.trace_digest(trace), drain_ns=20)
            worker.publish(root / "inference.start.json", schedule)
            setup = dict(status="prepared", pid=123, role="inference", environment={"device_uuid": "a"})
            worker.publish(folder / "setup.json", setup)
            header = {**self.header, "schedule_sha256": runner.sha256(root / "inference.start.json"),
                      "setup_sha256": runner.sha256(folder / "setup.json")}
            (folder / "progress.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [header, self.events[1]]))
            ready = dict(pid=123, device_uuid="a")
            runner.verify_progress(root, "inference", ready)
            report = dict(error="watchdog", schedule=schedule, inference_ready=ready)
            with mock.patch.object(runner.time, "monotonic_ns", return_value=160):
                runner.preserve_partial(root, report)
            partial = json.loads((root / "partial.json").read_text())
            self.assertFalse(partial["numerically_validated"])
            self.assertEqual(partial["status"], "failed")
            snapshot = partial["roles"]["inference"]["at_observation"]
            self.assertEqual(snapshot["pending_requests"], 3)
            self.assertIsNone(snapshot["response_lower_bound"])
            with self.assertRaises(ValueError):
                runner.verify_progress(root, "inference", dict(ready, pid=999))
            worker.publish(root / "inference.start.json", dict(schedule, begin_ns=101))
            with self.assertRaises(ValueError):
                runner.verify_progress(root, "inference", ready)

    def test_preparation_timestamp_is_required_and_bound_to_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            folder = root / "inference"
            folder.mkdir()
            worker.publish(root / "inference.start.json", dict(begin_ns=100, preparation_required=True))
            worker.publish(folder / "setup.json", dict(status="prepared", pid=123, role="inference",
                                                       environment={"device_uuid": "a"}))
            header = {**self.header, "schedule_sha256": runner.sha256(root / "inference.start.json"),
                      "setup_sha256": runner.sha256(folder / "setup.json")}
            for prepared in (None, True, 0, 100, 101, 99):
                event = {**header, "prepared_ns": prepared}
                (folder / "progress.jsonl").write_text(json.dumps(event) + "\n")
                if prepared == 99:
                    runner.verify_progress(root, "inference", dict(pid=123, device_uuid="a"))
                else:
                    with self.subTest(prepared=prepared), self.assertRaises(ValueError):
                        runner.verify_progress(root, "inference", dict(pid=123, device_uuid="a"))


if __name__ == "__main__":
    unittest.main()
