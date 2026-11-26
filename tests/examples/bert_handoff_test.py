"""CPU-only rejection tests for cross-engine physical handoff evidence."""

import copy
import hashlib
import io
import json
import os
import pathlib
import signal
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "examples/framework_workloads/bert"))
import verify_bert_handoff as verify
import run_bert_handoff as runner
import bert_colocation_worker as worker
from bert_contract import PROFILE, SEED, sha256, write_json
from run_persistent_mlp_handoff import AUDIT_PREFIX, CONTROL_PREFIX


def event(phase, timestamp, generation=1, launch=1, completed=0):
    return dict(phase=phase, pid=101, task=1, generation=generation, launch=launch,
                logical=8, completed=completed, exact=int(phase in {"suspended", "complete"}), time_ns=timestamp)


def audit(events):
    return "\n".join(AUDIT_PREFIX + " ".join(f"{key}={value}" for key, value in item.items())
                     for item in events) + "\n[glimmer] persistent_sgemm accepted=1\n"


def kernel(start, end, process, grid, api_start):
    return dict(start=start, end=end, device=0, context=1, stream=7, process=process,
                name=verify.OWNED if process == 101 else "native_gather", grid=grid,
                api_start=api_start, api_end=api_start + 1, api_name="cuLaunchKernel", api_thread=process + 1)


class PhysicalTest(unittest.TestCase):
    def setUp(self):
        self.events = [event("grant", 2, launch=0), event("prepare", 3), event("first_tile", 10, completed=1),
                       event("priority_retreat", 40), event("suspended", 50, completed=2),
                       event("grant", 80, generation=2, completed=2),
                       event("prepare", 90, generation=2, launch=2, completed=2),
                       event("first_tile", 100, generation=2, launch=2, completed=3),
                       event("complete", 110, generation=2, launch=2, completed=8)]
        self.low = [kernel(1000, 2000, 101, 1, 990), kernel(5000, 6000, 101, 2, 4900)]
        self.high = [kernel(3000, 4000, 102, 1, 2990)]
        self.control = CONTROL_PREFIX + ("task=2 generation=0 submit=1 status=0 enqueued_ns=21 "
                                        "started_ns=30 finished_ns=35 observed_ns=36\n")

    def check(self):
        return verify.physical_handoff(audit(self.events), self.control, self.low, self.high, 1, 20)

    def test_accept_independent_clocks_and_actual_native_gpu(self):
        self.assertEqual(self.check()["proof"]["low_partial_stop"]["completed"], 2)

    def test_low_progress_after_high_barrier_before_actual_submission_is_valid(self):
        self.events[2]["time_ns"] = 25
        self.control = CONTROL_PREFIX + ("task=2 generation=0 submit=1 status=0 enqueued_ns=26 "
                                        "started_ns=30 finished_ns=35 observed_ns=36\n")
        self.assertEqual(self.check()["proof"]["low_first_tile"]["time_ns"], 25)
        self.events[2]["time_ns"] = 27
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_early_issue_despite_late_gpu_start(self):
        self.low[1]["api_start"] = 3500
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_high_issued_before_physical_stop(self):
        self.high[0]["api_start"] = 1999
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_missing_or_extra_gpu_launch(self):
        self.low.pop()
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_different_stream_or_overlapping_launch(self):
        for field, value in (("stream", 8), ("start", 1999), ("api_start", 989), ("api_thread", 999)):
            saved = copy.deepcopy(self.low)
            self.low[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check()
            self.low = saved

    def test_reject_missing_high_execution(self):
        self.high = []
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_intent_only_and_false_progress(self):
        for index, field, value in ((2, "time_ns", 31), (4, "completed", 0), (4, "completed", 8),
                                    (4, "exact", 0), (5, "generation", 1), (8, "completed", 7)):
            saved = copy.deepcopy(self.events)
            self.events[index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                self.check()
            self.events = saved

    def test_reject_missing_submission_or_retreat(self):
        self.control = ""
        with self.assertRaises(ValueError):
            self.check()


class TimelineTest(unittest.TestCase):
    def setUp(self):
        self.database = sqlite3.connect(":memory:")
        self.addCleanup(self.database.close)
        self.database.executescript("""
            CREATE TABLE PROCESSES(globalPid INTEGER,pid INTEGER);
            CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,
                contextId INTEGER,streamId INTEGER,globalPid INTEGER,shortName INTEGER,gridId INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,nameId INTEGER,
                globalTid INTEGER,correlationId INTEGER);
            CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER,name TEXT);
            CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
            INSERT INTO PROCESSES VALUES(16777216,101),(33554432,102);
            INSERT INTO StringIds VALUES(1,'glimmer_persistent_sgemm'),(2,'native_gather'),
                (3,'cuLaunchKernel'),(4,'cudaProfilerStart_v10000'),(5,'cudaProfilerStop_v10000');
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                (1000,2000,0,1,7,16777216,1,1,1),(5000,6000,0,1,7,16777216,1,2,2),
                (3000,4000,0,1,8,33554432,2,1,3);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
                (990,991,3,16777217,1),(4900,4901,3,16777217,2),(2990,2991,3,33554433,3),
                (2500,2510,4,33554433,4),(4100,4110,5,33554433,5);
        """)

    def test_valid(self):
        low, high = verify.read_timeline(self.database, 101, 102)
        self.assertEqual((len(low), len(high)), (2, 1))
        self.assertEqual(low[1]["api_start"], 4900)

    def test_reject_corrupt_or_incomplete_capture(self):
        cases = [
            "DELETE FROM PROCESSES WHERE pid=102",
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET shortName=999 WHERE globalPid=33554432",
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET deviceId=1 WHERE globalPid=33554432",
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=123 WHERE globalPid=33554432",
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET gridId=1 WHERE globalPid=16777216",
            "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=2",
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=2",
            "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET start=5001 WHERE correlationId=2",
            "DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId=5",
            "UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=33554434 WHERE nameId=5",
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET start=2499 WHERE globalPid=33554432",
            "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET shortName=1 WHERE globalPid=33554432",
            "INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES(1,'Warning'); INSERT INTO DIAGNOSTIC_EVENT VALUES(1,'lost events')",
        ]
        for statement in cases:
            with self.subTest(statement=statement):
                clone = sqlite3.connect(":memory:")
                try:
                    self.database.backup(clone)
                    clone.executescript(statement)
                    with self.assertRaises(ValueError):
                        verify.read_timeline(clone, 101, 102)
                finally:
                    clone.close()

    def test_fixed_inventory_reads_all_geometry_and_requires_explicit_selection(self):
        self.database.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=33554432")
        self.database.execute("DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE globalTid=33554433")
        for name in ("gridX", "gridY", "gridZ", "blockX", "blockY", "blockZ"):
            self.database.execute(f"ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN {name} INTEGER DEFAULT 1")
        for index in range(4 * verify.HIGH_KERNELS_PER_REQUEST):
            start = 3000 + index * 10
            self.database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
                "(start,end,deviceId,contextId,streamId,globalPid,shortName,gridId,correlationId) "
                "VALUES(?,?,0,1,8,33554432,2,?,?)", (start, start + 5, index + 1, index + 1000))
            self.database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,3,33554433,?)",
                                  (start - 2, start - 1, index + 1000))
        signature = [["native_gather", 1, 1, 1, 1, 1, 1]] * verify.HIGH_KERNELS_PER_REQUEST
        expected = hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()
        with mock.patch.object(verify, "HIGH_REQUEST_SHA256", expected):
            with self.assertRaises(ValueError):
                verify.read_timeline(self.database, 101, 102)
            low, high = verify.read_timeline(self.database, 101, 102, "fixed-inventory")
            self.assertEqual((len(low), len(high)), (2, 646))
            self.assertEqual(high[0]["grid"], 647)
            self.assertEqual(high[0]["shape"], (1, 1, 1, 1, 1, 1))
        with self.assertRaises(ValueError):
            verify.read_timeline(self.database, 101, 102, "unknown")


class FixedInventoryTest(unittest.TestCase):
    def setUp(self):
        self.kernels = []
        for index in range(4 * verify.HIGH_KERNELS_PER_REQUEST):
            item = kernel(1000 + index * 10, 1005 + index * 10, 102, index + 1, 998 + index * 10)
            item.update(name=f"native_{index % 323}", shape=(index % 323 + 1, 1, 1, 256, 1, 1))
            self.kernels.append(item)
        signature = [[row["name"], *row["shape"]] for row in self.kernels[:323]]
        expected = hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()
        self.pin = mock.patch.object(verify, "HIGH_REQUEST_SHA256", expected)
        self.pin.start()
        self.addCleanup(self.pin.stop)

    def check(self, rows=None, markers=None):
        return verify.fixed_inference_work(self.kernels if rows is None else rows,
                                           [] if markers is None else markers)

    def test_only_final_two_requests_follow_four_complete_pinned_passes(self):
        result = self.check()
        self.assertEqual(len(result), 646)
        self.assertEqual(result[0]["grid"], 647)
        for rows in (self.kernels[-646:], self.kernels[:-323], self.kernels[:-1],
                     [*self.kernels, self.kernels[-1]]):
            with self.subTest(count=len(rows)), self.assertRaises(ValueError):
                self.check(rows=rows)

    def test_every_pass_checks_name_shape_and_order(self):
        for index in (0, 323, 646, 969, 1291):
            for key, value in (("name", "unknown_native"), ("name", verify.OWNED),
                               ("shape", (2, 1, 1, 256, 1, 1)),
                               ("shape", (True, 1, 1, 256, 1, 1)), ("shape", (0, 1, 1, 256, 1, 1))):
                rows = copy.deepcopy(self.kernels)
                rows[index][key] = value
                with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                    self.check(rows=rows)
        rows = copy.deepcopy(self.kernels)
        for key in ("name", "shape"):
            rows[0][key], rows[1][key] = rows[1][key], rows[0][key]
        with self.assertRaises(ValueError):
            self.check(rows=rows)

    def test_periodic_inventory_requires_all_declared_passes(self):
        rows = copy.deepcopy(self.kernels)
        for index, source in enumerate(self.kernels[:323], start=len(rows)):
            item = copy.deepcopy(source)
            item.update(start=1000 + index * 10, end=1005 + index * 10,
                        api_start=998 + index * 10, api_end=999 + index * 10, grid=index + 1)
            rows.append(item)
        selected = verify.fixed_inference_work(rows, [], requests=3)
        self.assertEqual(len(selected), 969)
        self.assertEqual(selected[0]["grid"], 647)
        for count in (True, 0, -1, 101, 2, 4):
            with self.subTest(count=count), self.assertRaises(ValueError):
                verify.fixed_inference_work(rows, [], requests=count)
        rows[-1]["name"] = "changed"
        with self.assertRaises(ValueError):
            verify.fixed_inference_work(rows, [], requests=3)

    def test_rejects_markers_and_nonserial_or_different_owner_work(self):
        with self.assertRaises(ValueError):
            self.check(markers=[(1, 2, "cudaProfilerStart", 103)])
        for key, value in (("device", 1), ("process", 103), ("context", 2), ("stream", 8),
                           ("api_thread", 999), ("start", self.kernels[0]["end"] - 1),
                           ("api_start", self.kernels[0]["api_end"] - 1)):
            rows = copy.deepcopy(self.kernels)
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check(rows=rows)


class RunnerTest(unittest.TestCase):
    def arguments(self, output):
        return types.SimpleNamespace(output=output, assets=pathlib.Path("/assets"), export=pathlib.Path("/export"),
                                     reference=pathlib.Path("/missing-native"), build=pathlib.Path("/build"),
                                     python=pathlib.Path("/python"), worker=False, control_transport="socket")

    def test_live_progress_ignores_warmup_small_and_already_ended_kernels(self):
        stream = io.StringIO()
        progress = runner.LiveLowProgress(stream, 100)

        def append(items):
            offset = stream.tell()
            stream.seek(0, io.SEEK_END)
            stream.write(audit(items))
            stream.seek(offset)

        append([dict(event("first_tile", 90, completed=1), logical=9216)])
        self.assertFalse(progress.ready())
        append([event("first_tile", 110, completed=1)])
        self.assertFalse(progress.ready())
        append([dict(event("first_tile", 120, launch=2, completed=1), logical=9216),
                dict(event("complete", 130, launch=2, completed=9216), logical=9216)])
        self.assertFalse(progress.ready())
        append([dict(event("first_tile", 140, launch=3, completed=1), logical=9216)])
        self.assertTrue(progress.ready())
        self.assertEqual(progress.observation["launch"], 3)
        append([dict(event("priority_retreat", 150, launch=3, completed=1), logical=9216)])
        self.assertFalse(progress.ready())

    def test_live_progress_waits_for_a_complete_published_audit_line(self):
        line = audit([dict(event("first_tile", 120, completed=1), logical=9216)]).splitlines(keepends=True)[0]
        stream = io.StringIO(line[:-1])
        progress = runner.LiveLowProgress(stream, 100)
        self.assertFalse(progress.ready())
        offset = stream.tell()
        stream.write("\n")
        stream.seek(offset)
        self.assertTrue(progress.ready())

    def test_early_prepare_trigger_is_explicit_intent_not_tile_progress(self):
        line = audit([dict(event("prepare", 120), logical=9216)])
        default = runner.LiveLowProgress(io.StringIO(line), 100)
        self.assertFalse(default.ready())
        early = runner.LiveLowProgress(io.StringIO(line), 100, "large-prepare")
        self.assertTrue(early.ready())
        self.assertEqual(early.observation["phase"], "prepare")
        self.assertEqual(early.observation["completed"], 0)
        with self.assertRaises(ValueError):
            runner.LiveLowProgress(io.StringIO(), 100, "unknown")

    def test_profiler_injection_and_credentials(self):
        args = self.arguments(pathlib.Path("/output"))
        inherited = {"AWS_SECRET_ACCESS_KEY": "secret", "LD_PRELOAD": "untrusted.so", "LD_AUDIT": "audit.so",
                     "CUDA_INJECTION64_PATH": "collector.so", "NSYS_PROFILING_SESSION_ID": "session",
                     "QUADD_CUDA_CONFIG": "config", "GLIMMER_UNRELATED": "1",
                     "GLIMMER_CONTROL_TRANSPORT": "shared_memory"}
        with mock.patch.dict(os.environ, inherited, clear=True):
            plain = runner.child_environment(args, 10, "low", "/socket")
            args.worker = True
            profiled = runner.child_environment(args, 10, "low", "/socket")
        for env in (plain, profiled):
            for key in ("AWS_SECRET_ACCESS_KEY", "LD_AUDIT", "GLIMMER_UNRELATED"):
                self.assertNotIn(key, env)
            self.assertEqual(env["LD_PRELOAD"], "/build/lib/libglimmer_cuda_interceptor.so")
            self.assertEqual(env["GLIMMER_CONTROL_TRANSPORT"], "socket")
            self.assertEqual(env["GLIMMER_PERSISTENT_WORKER_BLOCKS"], "320")
        self.assertNotIn("CUDA_INJECTION64_PATH", plain)
        self.assertEqual(profiled["CUDA_INJECTION64_PATH"], "collector.so")
        self.assertEqual(profiled["NSYS_PROFILING_SESSION_ID"], "session")

    def test_clear_environment_only_before_profiler(self):
        command = runner.profile_command(self.arguments(pathlib.Path("/output")), "/nsys", pathlib.Path("/profile"))
        self.assertEqual(command[:2], ["/usr/bin/env", "-i"])
        self.assertEqual(command.count("-i"), 1)
        self.assertFalse(any(value.startswith("LD_PRELOAD=") for value in command))
        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--control-transport") + 1], "socket")
        self.assertEqual(command[command.index("--high-timeline-boundary") + 1], "runtime-markers")

    def test_fixed_high_boundary_is_explicit_and_survives_outer_profiler(self):
        args = self.arguments(pathlib.Path("/output"))
        args.high_timeline_boundary = "fixed-inventory"
        command = runner.profile_command(args, "/nsys", pathlib.Path("/profile"))
        self.assertEqual(command[command.index("--high-timeline-boundary") + 1], "fixed-inventory")
        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--high-release-trigger") + 1], "live-first-tile")
        args.high_release_trigger = "large-prepare"
        command = runner.profile_command(args, "/nsys", pathlib.Path("/profile"))
        self.assertEqual(command[command.index("--high-release-trigger") + 1], "large-prepare")

    def test_periodic_high_is_explicit_bounded_and_keeps_original_low(self):
        args = self.arguments(pathlib.Path("/output"))
        self.assertEqual(runner.high_configuration(args)["high_workload"], "original")
        args.high_workload = "periodic"
        with self.assertRaises(ValueError):
            runner.high_configuration(args)
        args.high_timeline_boundary = "fixed-inventory"
        configuration = runner.high_configuration(args)
        self.assertEqual(configuration, dict(high_workload="periodic", high_requests=20, high_period_ms=100))
        args.worker = True
        command = runner.application_command(args, "high", args.output, configuration)
        self.assertIn(str(runner.EXAMPLE / "bert_colocation_worker.py"), command)
        self.assertNotIn("--profile-cuda", command)
        self.assertEqual(command[command.index("--start-file") + 1], "/output/inference.start.json")
        low = runner.application_command(args, "low", args.output, configuration)
        self.assertIn("--snapshot", low)
        self.assertIn("--reference", low)
        self.assertIn("--profile-cuda", low)
        self.assertEqual(low[low.index("--iterations") + 1], "2")
        command = runner.profile_command(args, "/nsys", pathlib.Path("/profile"))
        self.assertEqual(command[command.index("--high-workload") + 1], "periodic")
        self.assertEqual(command[command.index("--high-requests") + 1], "20")
        self.assertEqual(command[command.index("--high-period-ms") + 1], "100")
        for count, period in ((0, 100), (True, 100), (101, 1), (20, 0), (20, 501)):
            args.high_requests, args.high_period_ms = count, period
            with self.subTest(count=count, period=period), self.assertRaises(ValueError):
                runner.high_configuration(args)

    def test_explicit_shared_memory_selection_survives_profiler(self):
        args = self.arguments(pathlib.Path("/output"))
        args.control_transport = "shared_memory"
        for worker in (False, True):
            args.worker = worker
            for role, priority in (("low", 10), ("high", 100)):
                self.assertEqual(runner.child_environment(args, priority, role, "/socket")
                                 ["GLIMMER_CONTROL_TRANSPORT"], "shared_memory")
        command = runner.profile_command(args, "/nsys", pathlib.Path("/profile"))
        self.assertEqual(command[command.index("--control-transport") + 1], "shared_memory")
        args.control_transport = "invalid"
        with self.assertRaises(ValueError):
            runner.child_environment(args, 10, "low", "/socket")

    def test_missing_input_preserves_failure_without_spawning(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.arguments(pathlib.Path(directory) / "run")
            with mock.patch.object(runner.subprocess, "Popen") as spawn:
                with self.assertRaises(RuntimeError):
                    runner.collect(args)
                spawn.assert_not_called()
            result = json.loads((args.output / "probe.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("error", result)
            self.assertFalse(result["preemption_acceptance"])

    def test_ready_file_exists_before_json_is_published(self):
        process = types.SimpleNamespace(poll=lambda: None)
        value = {"pid": 77, "ready_ns": 1, "device_uuid": "gpu"}
        path = mock.Mock()
        path.read_text.side_effect = [FileNotFoundError(), "", '{"pid":', json.dumps(value)]
        self.assertEqual(runner.read_ready(path, [process], 77, timeout=1), value)
        self.assertEqual(path.read_text.call_count, 4)

    def test_ready_rejects_wrong_identity_and_dead_writer(self):
        for value in ({"pid": 78, "ready_ns": 1, "device_uuid": "gpu"},
                      {"pid": 77, "ready_ns": 0, "device_uuid": "gpu"},
                      {"pid": 77, "ready_ns": 1, "device_uuid": ""}):
            path = mock.Mock()
            path.read_text.return_value = json.dumps(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.read_ready(path, [], 77, timeout=1)
        path.read_text.return_value = ""
        with self.assertRaises(RuntimeError):
            runner.read_ready(path, [types.SimpleNamespace(poll=lambda: 1)], 77, timeout=1)
        with self.assertRaises(TimeoutError):
            runner.read_ready(path, [], 77, timeout=0)

    def test_timeout_kills_only_owned_group(self):
        process = mock.Mock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned", 10), -9]
        with mock.patch.object(runner.os, "killpg") as kill:
            runner.stop_group(process)
        self.assertEqual(kill.call_args_list, [mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL)])

    def test_startup_failure_cleans_owned_service_before_return(self):
        for worker in (False, True):
            with self.subTest(worker=worker), tempfile.TemporaryDirectory() as directory:
                args = self.arguments(pathlib.Path(directory) / "run")
                args.worker = worker
                process = mock.Mock(pid=1234)
                process.poll.return_value = None
                with mock.patch.object(runner, "sha256", return_value="digest"), \
                        mock.patch.object(runner.subprocess, "Popen", return_value=process) as spawn, \
                        mock.patch.object(runner, "wait_until", side_effect=RuntimeError("startup failed")), \
                        mock.patch.object(runner.os, "killpg") as kill:
                    with self.assertRaisesRegex(RuntimeError, "startup failed"):
                        runner.collect(args)
                self.assertEqual(spawn.call_args.kwargs["start_new_session"], not worker)
                if worker:
                    process.terminate.assert_called_once()
                    kill.assert_not_called()
                else:
                    kill.assert_called_once_with(1234, signal.SIGTERM)
                process.wait.assert_called_once()
                self.assertEqual(json.loads((args.output / "probe.json").read_text())["status"], "failed")


class PeriodicOracleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = pathlib.Path(temporary.name)
        folder = self.output / "inference"
        folder.mkdir()
        self.native = {"assets": {"pinned": True}}
        export = self.output / "export.json"
        write_json(export, {"inputs": {"input": "pinned"}})
        trace = worker.make_trace(2, 100_000_000)
        begin = 100_000_001
        self.schedule = dict(begin_ns=begin, trace=trace, trace_sha256=worker.trace_digest(trace),
                             preparation_required=True, drain_ns=10_000_000_000)
        write_json(self.output / "inference.start.json", self.schedule)
        self.manifest = dict(high_workload="periodic", high_requests=2, high_period_ms=100,
                             high_timeline_boundary="fixed-inventory", high_schedule=self.schedule,
                             high_release_ns=1, high_ready=dict(pid=102, device_uuid="GPU-a"),
                             fingerprints={str(export): sha256(export),
                                 str(runner.EXAMPLE / "bert_colocation_worker.py"): "worker"})
        setup = dict(status="prepared", pid=102, schema="bert_bounded_arrival_v2", role="inference",
                     profile=PROFILE["inference"], seed=SEED, dtype="float32", warmup=1, instrumented=False,
                     worker_sha256="worker", engine="onnxruntime-cuda", assets=self.native["assets"],
                     providers=["CUDAExecutionProvider"], inputs={"input": "pinned"},
                     export_sha256=sha256(export), environment=dict(device_uuid="GPU-a"))
        write_json(folder / "setup.json", setup)
        records = [dict(arrival_ns=begin + index * 100_000_000,
                        start_ns=begin + index * 100_000_000, end_ns=begin + index * 100_000_000 + 10)
                   for index in range(2)]
        end = begin + trace["duration_ns"]
        events = [dict(phase="begin", schema="bert_progress_v1", role="inference", pid=102,
                       begin_ns=begin, prepared_ns=begin - 1,
                       setup_sha256=sha256(folder / "setup.json"),
                       schedule_sha256=sha256(self.output / "inference.start.json"))]
        for index, record in enumerate(records):
            events += [dict(phase="start", index=index, **{key: value for key, value in record.items() if key != "end_ns"}),
                       dict(phase="complete", index=index, **record)]
        events.append(dict(phase="stop", end_ns=end, completed=2))
        (folder / "progress.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        self.high = dict(setup, status="ok", records=records, measured_begin_ns=begin, measured_end_ns=end,
                         validation=dict(checked_steps=2, kind="native PyTorch oracle, rtol=2e-4, atol=2e-5"),
                         journal_sha256=sha256(folder / "progress.jsonl"))
        self.high.pop("pid")

    def check(self):
        verify.validate_periodic_high(self.output, self.manifest, self.high, self.native)

    def test_complete_oracle_journal_schedule_and_setup(self):
        self.check()

    def test_reject_source_inputs_oracle_inventory_and_frozen_schedule_changes(self):
        original = copy.deepcopy(self.high)
        for key, value in (("worker_sha256", "changed"), ("inputs", {}), ("warmup", 2),
                           ("seed", 1), ("validation", dict(checked_steps=1)),
                           ("records", original["records"][:1]), ("journal_sha256", "changed")):
            self.high = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.check()
        self.high = original
        self.manifest["high_period_ms"] = 101
        with self.assertRaises(ValueError):
            self.check()
        self.manifest["high_period_ms"] = 100
        self.schedule["trace"]["offsets_ns"][1] += 1
        write_json(self.output / "inference.start.json", self.schedule)
        with self.assertRaises(ValueError):
            self.check()

    def test_reject_incomplete_journal_and_unprepared_arrival(self):
        journal = self.output / "inference/progress.jsonl"
        original = journal.read_text()
        for text in (original.rstrip("\n"), original.replace('"prepared_ns": 100000000', '"prepared_ns": 100000001')):
            journal.write_text(text)
            self.high["journal_sha256"] = sha256(journal)
            with self.assertRaises(ValueError):
                self.check()


if __name__ == "__main__":
    unittest.main()
