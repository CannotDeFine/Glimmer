"""No-GPU controls for the common Linux event-driven measurement barrier."""

import contextlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples/framework_workloads/bert"))
import bert_contract as contract


class StartBarrierTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(contract._retire_start_notifications)
        self.root = Path(self.temporary.name)
        self.ready, self.start = self.root / "ready.json", self.root / "start"

    def wait_until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("bounded fixture condition did not become true")
            threading.Event().wait(0.001)

    def worker(self, function):
        results = []
        done = threading.Event()

        def invoke():
            try:
                function()
                results.append(None)
            except BaseException as error:
                results.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=invoke)
        thread.start()
        self.addCleanup(thread.join, 4)
        return thread, done, results

    def assert_success(self, worker):
        thread, done, results = worker
        self.assertTrue(done.wait(3), "bounded event wakeup")
        thread.join()
        self.assertEqual(results, [None])

    def test_does_not_poll(self):
        with mock.patch.object(contract.time, "sleep", side_effect=AssertionError("polling forbidden")):
            worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
            self.wait_until(self.ready.exists)
            self.start.write_text("start\n")
            self.assert_success(worker)
        ready = json.loads(self.ready.read_text())
        self.assertEqual(ready["pid"], os.getpid())
        self.assertEqual(ready["device_uuid"], "gpu")

    def test_success_returns_before_any_potentially_blocking_listener_close(self):
        original = contract._StartNotification.close
        entered, released = threading.Event(), threading.Event()

        def held_close(notification):
            entered.set()
            if not released.wait(3):
                raise TimeoutError("bounded independently held listener cleanup")
            return original(notification)

        with mock.patch.object(contract._StartNotification, "close", new=held_close):
            worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
            self.wait_until(self.ready.exists)
            self.start.write_text("start\n")
            returned_without_close = worker[1].wait(0.1)
            closed_on_request_path = entered.is_set()
            released.set()
            self.assert_success(worker)
        # Assert only after releasing/settling the old cleanup path, so RED
        # cannot leave a blocked worker or unclosed descriptor behind.
        self.assertTrue(returned_without_close and not closed_on_request_path,
                        "successful request still waits for listener retirement")

    def test_completed_ownership_is_bounded_and_retirement_restores_fds(self):
        before = set(os.listdir("/proc/self/fd"))
        original = Path.open

        @contextlib.contextmanager
        def publish(path, *args, **kwargs):
            with original(path, *args, **kwargs) as stream:
                yield stream
            if path.name.startswith("ready-"):
                (self.root / path.name.replace("ready-", "start-", 1)).write_text("start\n")

        with mock.patch.object(Path, "open", new=publish):
            for index in range(contract._START_MAX_NOTIFICATIONS):
                contract.wait_for_start(self.root / f"ready-{index}", self.root / f"start-{index}", "gpu")
        self.assertEqual(len(contract._start_notifications), 8)
        self.assertEqual(len(os.listdir("/proc/self/fd")), len(before) + 16)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            contract.wait_for_start(self.ready, self.start, "gpu")
        self.assertFalse(self.ready.exists())
        contract._retire_start_notifications()
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)
        self.assertEqual(contract._start_notifications, [])

    def test_active_wait_cannot_be_retired_and_reuses_failed_slots(self):
        before = set(os.listdir("/proc/self/fd"))
        worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
        self.wait_until(self.ready.exists)
        with self.assertRaisesRegex(RuntimeError, "active"):
            contract._retire_start_notifications()
        self.assertEqual(len(contract._start_notifications), 1)
        self.start.write_text("start\n")
        self.assert_success(worker)
        contract._retire_start_notifications()
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)

    def test_inflight_native_setup_already_owns_a_capacity_slot(self):
        before = set(os.listdir("/proc/self/fd"))
        actual_open, nested, has_reentered = os.open, [], False

        def open_during_setup(*args, **kwargs):
            nonlocal has_reentered
            if not has_reentered:
                has_reentered = True
                for index in range(7):
                    nested.append(contract._StartNotification(self.root / f"nested-{index}"))
                self.assertEqual(len(contract._start_notifications), 8)
                with self.assertRaisesRegex(RuntimeError, "capacity"):
                    contract._StartNotification(self.root / "overflow")
            return actual_open(*args, **kwargs)

        with mock.patch.object(contract.os, "open", side_effect=open_during_setup):
            owner = contract._StartNotification(self.start)
        for notification in [owner, *nested]:
            notification.__exit__(None)
        contract._retire_start_notifications()
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)
        for index in range(12):
            with self.assertRaises(TimeoutError):
                contract.wait_for_start(self.root / f"timeout-{index}", self.root / "absent", "gpu", timeout=0)
        self.assertEqual(contract._start_notifications, [])
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)

    def test_close_error_still_attempts_every_descriptor_and_owner(self):
        before = set(os.listdir("/proc/self/fd"))
        owners = [contract._StartNotification(self.root / f"signal-{index}") for index in range(2)]
        for owner in owners:
            owner.__exit__(None)
        actual_close, calls = os.close, []

        def close_then_report_error(descriptor):
            actual_close(descriptor)
            calls.append(descriptor)
            if len(calls) == 1:
                raise OSError("reported close failure; never retry the same descriptor")

        with mock.patch.object(contract.os, "close", side_effect=close_then_report_error):
            with self.assertRaisesRegex(OSError, "reported close failure"):
                contract._retire_start_notifications()
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(set(calls)), 4)
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)
        self.assertEqual(contract._start_notifications, [])

    def test_retirement_keeps_capacity_and_closes_outside_exclusion(self):
        owner = contract._StartNotification(self.start)
        owner.__exit__(None)
        entered, release, failures = threading.Event(), threading.Event(), []
        original = os.close

        def held_close(descriptor):
            if descriptor == owner.descriptor or not entered.is_set():
                self.assertTrue(contract._start_notifications_lock.acquire(blocking=False))
                contract._start_notifications_lock.release()
                entered.set()
                if not release.wait(3):
                    raise TimeoutError("bounded native close hold")
            return original(descriptor)

        def retire():
            try:
                contract._retire_start_notifications()
            except BaseException as error:
                failures.append(error)

        with mock.patch.object(contract.os, "close", side_effect=held_close):
            thread = threading.Thread(target=retire)
            thread.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertIn(owner, contract._start_notifications)
                with self.assertRaisesRegex(RuntimeError, "retirement"):
                    contract._StartNotification(self.root / "late")
                with self.assertRaisesRegex(RuntimeError, "active"):
                    contract._retire_start_notifications()
            finally:
                release.set()
                thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(contract._start_notifications, [])
        with self.assertRaises(TimeoutError):
            contract.wait_for_start(self.ready, self.root / "absent", "gpu", timeout=0)

    def test_normal_process_exit_retires_completed_listener(self):
        source = """
import atexit, json, os, pathlib, sys, threading, time
def verify():
    if set(os.listdir('/proc/self/fd')) != before or contract._start_notifications:
        raise RuntimeError('normal-exit listener cleanup failed')
    print(json.dumps({'normal_exit_listener_cleanup': True}))
atexit.register(verify)
sys.path.insert(0, sys.argv[1])
import bert_contract as contract
root = pathlib.Path(sys.argv[2])
before = set(os.listdir('/proc/self/fd'))
def publish():
    deadline = time.monotonic() + 2
    while not (root / 'ready').exists():
        if time.monotonic() > deadline:
            return
        threading.Event().wait(0.001)
    (root / 'start').write_text('start')
writer = threading.Thread(target=publish)
writer.start()
contract.wait_for_start(root / 'ready', root / 'start', 'gpu', timeout=2)
writer.join()
if len(contract._start_notifications) != 1:
    raise RuntimeError('successful request lost its bounded cleanup owner')
"""
        result = subprocess.run([sys.executable, "-B", "-c", source, str(Path(contract.__file__).parent),
                                 str(self.root)], capture_output=True, text=True, timeout=5, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(result.stderr)
        self.assertEqual(json.loads(result.stdout), {"normal_exit_listener_cleanup": True})

    def test_exit_cleanup_failure_cannot_masquerade_as_exit_zero(self):
        source = """
import sys
sys.path.insert(0, sys.argv[1])
import bert_contract as contract
def fail():
    raise OSError('injected final cleanup failure')
contract._retire_start_notifications = fail
"""
        result = subprocess.run([sys.executable, "-B", "-c", source, str(Path(contract.__file__).parent)],
                                capture_output=True, text=True, timeout=5, check=False)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, "BERT start notification retirement failed\n")

    def test_immediate_release_after_ready_close_is_not_lost(self):
        original = Path.open

        @contextlib.contextmanager
        def publish(path, *args, **kwargs):
            with original(path, *args, **kwargs) as stream:
                yield stream
            if path == self.ready:
                self.start.write_text("start\n")

        with mock.patch.object(Path, "open", new=publish):
            contract.wait_for_start(self.ready, self.start, "gpu", timeout=2)

    def test_creation_is_not_a_complete_schedule(self):
        worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
        self.wait_until(self.ready.exists)
        with self.start.open("x") as stream:
            stream.write('{"release":')
            stream.flush()
            self.assertFalse(worker[1].wait(0.05), "must not consume an open partial schedule")
            stream.write("1}")
        self.assert_success(worker)
        self.assertEqual(json.loads(self.start.read_text()), {"release": 1})

    def test_atomic_publish_and_unrelated_notifications(self):
        worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
        self.wait_until(self.ready.exists)
        pending = self.root / "pending"
        pending.write_text("schedule\n")
        self.assertFalse(worker[1].wait(0.02))
        pending.rename(self.start)
        self.assert_success(worker)

    def test_stale_and_dangling_symlink_reject_before_ready(self):
        for symlink in (False, True):
            if symlink:
                self.start.symlink_to(self.root / "absent")
            else:
                self.start.write_text("stale")
            with self.assertRaises(ValueError):
                contract.wait_for_start(self.ready, self.start, "gpu")
            self.assertFalse(self.ready.exists())
            self.start.unlink()

    def test_timeout_keeps_ready_and_closes_descriptors(self):
        before = set(os.listdir("/proc/self/fd"))
        for index in range(8):
            ready = self.root / f"ready{index}"
            with self.assertRaises(TimeoutError):
                contract.wait_for_start(ready, self.start, "gpu", timeout=0)
            self.assertEqual(json.loads(ready.read_text())["device_uuid"], "gpu")
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)

    def test_existing_ready_and_setup_error_close_descriptors(self):
        self.ready.write_text("old")
        before = set(os.listdir("/proc/self/fd"))
        with self.assertRaises(FileExistsError):
            contract.wait_for_start(self.ready, self.start, "gpu", timeout=0)
        self.assertEqual(set(os.listdir("/proc/self/fd")), before)
        self.assertEqual(self.ready.read_text(), "old")
        with mock.patch.object(contract, "_StartNotification", side_effect=OSError("watch setup")):
            with self.assertRaisesRegex(OSError, "watch setup"):
                contract.wait_for_start(self.root / "fresh", self.start, "gpu")
        self.assertFalse((self.root / "fresh").exists())

    def test_watch_invalidation_is_not_success_or_polling(self):
        worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
        self.wait_until(self.ready.exists)
        self.root.rename(self.root.with_name(self.root.name + "-moved"))
        moved = self.root.with_name(self.root.name + "-moved")
        try:
            self.assertTrue(worker[1].wait(3))
            self.assertIsInstance(worker[2][0], ValueError)
        finally:
            moved.rename(self.root)

    def test_native_setup_errors_close_every_owned_descriptor(self):
        before = set(os.listdir("/proc/self/fd"))
        for failure in ("initialize", "watch"):
            library = mock.Mock()
            library.inotify_init1.side_effect = (
                (lambda _: -1) if failure == "initialize" else
                (lambda _: os.open("/dev/null", os.O_RDONLY | os.O_CLOEXEC)))
            library.inotify_add_watch.return_value = -1
            with mock.patch.object(contract.ctypes, "CDLL", return_value=library):
                with self.assertRaises(OSError):
                    contract.wait_for_start(self.ready, self.start, "gpu")
            self.assertFalse(self.ready.exists())
            self.assertEqual(set(os.listdir("/proc/self/fd")), before)

    def test_interrupted_wait_keeps_same_watch_and_deadline(self):
        original = contract.select.select
        interrupted = False

        def select_once(*args):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise InterruptedError()
            return original(*args)

        with mock.patch.object(contract.select, "select", side_effect=select_once):
            worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
            self.wait_until(self.ready.exists)
            self.start.write_text("start\n")
            self.assert_success(worker)
        self.assertTrue(interrupted)

    def test_renamed_symlink_cannot_publish_start(self):
        worker = self.worker(lambda: contract.wait_for_start(self.ready, self.start, "gpu", timeout=2))
        self.wait_until(self.ready.exists)
        target = self.root / "real"
        target.write_text("start\n")
        link = self.root / "link"
        link.symlink_to(target)
        link.rename(self.start)
        self.assertTrue(worker[1].wait(3))
        self.assertIsInstance(worker[2][0], ValueError)

    def test_event_parser_rejects_overflow_truncation_and_invalid_identity(self):
        event = struct.Struct("iIII")
        valid = event.pack(7, 8, 0, 8) + b"start\0\0\0"
        self.assertTrue(contract._start_event(valid, 7, b"start"))
        self.assertFalse(contract._start_event(valid, 7, b"other"))
        for frame in (valid[:-1], b"x", event.pack(-1, 0x4000, 0, 0),
                      event.pack(7, 0x8000, 0, 0), event.pack(8, 8, 0, 0),
                      event.pack(7, 8, 0, 3) + b"bad",
                      event.pack(7, 8, 0, 8) + b"start\0x\0"):
            with self.subTest(frame=frame):
                with self.assertRaises(ValueError):
                    contract._start_event(frame, 7, b"start")

    def test_omitted_and_unpaired_paths(self):
        contract.wait_for_start(None, None, "gpu")
        for paths in ((None, self.start), (self.ready, None)):
            with self.assertRaises(ValueError):
                contract.wait_for_start(*paths, "gpu")
        for invalid in (True, float("inf"), float("nan"), "120"):
            with self.assertRaises(ValueError):
                contract.wait_for_start(self.ready, self.start, "gpu", timeout=invalid)
        self.assertFalse(self.ready.exists())
        with self.assertRaises(ValueError):
            contract.wait_for_start(self.ready, self.ready, "gpu")


if __name__ == "__main__":
    unittest.main()
