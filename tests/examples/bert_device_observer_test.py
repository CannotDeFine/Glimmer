"""CPU-only sampler containment and cold preparation regressions."""

import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_device_observer as observer
import run_bert_device_observation as prepare
import aggregate_sm_audit as aggregate


def record(pid=42, uid=0, start=77, **extra):
    return dict(pid=pid, uid=uid, start_ticks=start, cgroup='/tasks/glimmer-bert-attempt-test-native.service',
                ppid=41, state='S', **extra)


def plan():
    return dict(version=1, mode='native', owner=record(pid=20, uid=1000),
                nonce='a' * 32, gpu_uuid=observer.GPU_UUID, fields=list(observer.FIELDS),
                period_us=200000, hard_timeout_s=700, observer_identities={'/sealed': 'b' * 64})


class DirectoryTests(unittest.TestCase):
    def test_journal_utf8_limit_is_enforced_before_write(self):
        stream = io.StringIO()
        writer = observer.BoundedWriter(stream, limit=4)
        writer.write('é')
        writer.write('ab')
        with self.assertRaisesRegex(ValueError, 'write bound'):
            writer.write('!')
        self.assertEqual(stream.getvalue(), 'éab')
        self.assertEqual(writer.written, 4)

    def test_directory_exclusive_files_and_zero_byte_owned_stop(self):
        with tempfile.TemporaryDirectory() as tmp, observer.PrivateDirectory(Path(tmp), os.getuid()) as directory:
            directory.publish('identity.json', dict(proof=True))
            self.assertEqual(directory.read('identity.json', os.getuid()), dict(proof=True))
            with self.assertRaises(FileExistsError):
                directory.publish('identity.json', dict(proof=False))
            self.assertFalse(directory.stopped())
            Path(tmp, 'stop').touch()
            self.assertTrue(directory.stopped())
            with (directory / 'samples.jsonl').open('x') as stream:
                stream.write('{}\n')
                stream.flush()
            with self.assertRaises(FileExistsError):
                (directory / 'samples.jsonl').open('x')
            with self.assertRaisesRegex(ValueError, 'undeclared'):
                directory.publish('../outside', {})

    def test_symlink_and_hardlink_stop_or_plan_never_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, 'observer')
            folder.mkdir(mode=0o700)
            outside = Path(tmp, 'outside')
            outside.write_text('{}')
            with observer.PrivateDirectory(folder, os.getuid()) as directory:
                (folder / 'stop').symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'stop request'):
                    directory.stopped()
                (folder / 'observer-plan.json').symlink_to(outside)
                with self.assertRaises(OSError):
                    directory.read('observer-plan.json', os.getuid())
                (folder / 'observer-plan.json').unlink()
                os.link(outside, folder / 'observer-plan.json')
                with self.assertRaisesRegex(ValueError, 'single-link'):
                    directory.read('observer-plan.json', os.getuid())
                self.assertEqual(outside.read_text(), '{}')

    def test_alias_permissions_and_directory_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, 'observer')
            folder.mkdir(mode=0o700)
            alias = Path(tmp, 'alias')
            alias.symlink_to(folder)
            with self.assertRaisesRegex(ValueError, 'canonical'):
                observer.PrivateDirectory(alias, os.getuid())
            with observer.PrivateDirectory(folder, os.getuid()) as directory:
                folder.chmod(0o755)
                with self.assertRaisesRegex(ValueError, 'private'):
                    directory.publish('ready.json', {})
                folder.chmod(0o700)
                folder.rename(Path(tmp, 'retired'))
                folder.mkdir(mode=0o700)
                with self.assertRaisesRegex(ValueError, 'changed'):
                    directory.publish('ready.json', {})
            self.assertFalse((folder / 'ready.json').exists())

    def test_malformed_duplicate_or_wrong_owner_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, observer.PrivateDirectory(Path(tmp), os.getuid()) as directory:
            path = Path(tmp, 'armed.json')
            path.write_text('{"a":1,"a":2}')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                directory.read('armed.json', os.getuid())
            path.write_text('{"a":NaN}')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                directory.read('armed.json', os.getuid())
            with self.assertRaisesRegex(ValueError, 'owned JSON'):
                directory.read('armed.json', os.getuid() + 1)


class IdentityTests(unittest.TestCase):
    def test_pidfd_open_recheck_catches_reuse_and_closes(self):
        own = record()
        with patch.object(observer, 'process_record', side_effect=[own, record(start=78)]), \
             patch.object(observer.os, 'pidfd_open', return_value=99), \
             patch.object(observer.select, 'poll') as poll, patch.object(observer.os, 'close') as close:
            poll.return_value.poll.return_value = []
            with self.assertRaisesRegex(ValueError, 'reused'):
                observer.PinnedProcess(own)
            close.assert_called_once_with(99)

    def test_pidfd_exit_even_with_same_numeric_pid_is_not_alive(self):
        own = record()
        with patch.object(observer, 'process_record', return_value=own), \
             patch.object(observer.os, 'pidfd_open', return_value=99), \
             patch.object(observer.select, 'poll') as poll, patch.object(observer.os, 'close') as close:
            poll.return_value.poll.return_value = []
            with observer.PinnedProcess(own) as proof:
                poll.return_value.poll.return_value = [(99, 1)]
                with self.assertRaisesRegex(ValueError, 'exited'):
                    proof.check()
            close.assert_called_once_with(99)

    def test_plan_uid_cgroup_limits_and_identity_before_sdk(self):
        value = plan()
        path = observer.OUTPUT_ROOT / 'attempt-test/native/observer'
        own = record(pid=42)
        with patch.object(observer, 'process_record', side_effect=lambda pid: value['owner'] if pid == 20 else own), \
             patch.object(observer.os, 'getpid', return_value=42), \
             patch.object(observer.os, 'getuid', return_value=0), \
             patch.object(observer.os, 'geteuid', return_value=0), \
             patch.object(observer, 'restrict_sampler_file_limit') as restrict, \
             patch.object(observer, 'capped_memory') as memory, patch.object(observer, 'verify_seals') as seals:
            self.assertEqual(observer.validate_sampler_plan(value, path), own)
            restrict.assert_called_once_with()
            memory.assert_called_once_with(own['cgroup'])
            seals.assert_called_once()
            for changes in (dict(fields=[1002]), dict(hard_timeout_s=True), dict(period_us=100000)):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    observer.validate_sampler_plan({**value, **changes}, path)
            with patch.object(observer.os, 'geteuid', return_value=1000), self.assertRaises(ValueError):
                observer.validate_sampler_plan(value, path)
            with patch.object(observer, 'capped_memory', side_effect=ValueError('memory limit')), \
                 self.assertRaisesRegex(ValueError, 'memory limit'):
                observer.validate_sampler_plan(value, path)
            restrict.reset_mock()
            with patch.object(observer, 'process_record', return_value={**own, 'cgroup': '/foreign'}), \
                 self.assertRaises(ValueError):
                observer.validate_sampler_plan(value, path)
            restrict.assert_not_called()

    def test_sudo_reset_file_limit_is_lowered_and_read_back(self):
        wanted = (observer.FILE_MAX,) * 2
        for previous in ((-1, -1), (observer.FILE_MAX, -1), (observer.FILE_MAX * 2,) * 2):
            with self.subTest(previous=previous), \
                 patch.object(observer.resource, 'getrlimit', side_effect=[previous, wanted]), \
                 patch.object(observer.resource, 'setrlimit') as setter:
                observer.restrict_sampler_file_limit()
                setter.assert_called_once_with(observer.resource.RLIMIT_FSIZE, wanted)

    def test_exact_file_limit_is_preserved_and_stricter_limits_never_raised(self):
        wanted = (observer.FILE_MAX,) * 2
        with patch.object(observer.resource, 'getrlimit', return_value=wanted), \
             patch.object(observer.resource, 'setrlimit') as setter:
            observer.restrict_sampler_file_limit()
            setter.assert_not_called()
        for previous in ((0, -1), (observer.FILE_MAX - 1,) * 2):
            with self.subTest(previous=previous), \
                 patch.object(observer.resource, 'getrlimit', return_value=previous), \
                 patch.object(observer.resource, 'setrlimit') as setter, \
                 self.assertRaisesRegex(ValueError, 'never raise'):
                observer.restrict_sampler_file_limit()
            setter.assert_not_called()

    def test_limit_setting_failure_or_wrong_readback_is_not_accepted(self):
        with patch.object(observer.resource, 'getrlimit', return_value=(-1, -1)), \
             patch.object(observer.resource, 'setrlimit', side_effect=OSError('denied')), \
             self.assertRaisesRegex(OSError, 'denied'):
            observer.restrict_sampler_file_limit()
        with patch.object(observer.resource, 'getrlimit', return_value=(-1, -1)), \
             patch.object(observer.resource, 'setrlimit'), \
             self.assertRaisesRegex(ValueError, 'restoration failed'):
            observer.restrict_sampler_file_limit()

    def test_wrong_file_limit_or_memory_event_fails(self):
        with patch.object(observer.resource, 'getrlimit', return_value=(-1, -1)), self.assertRaises(ValueError):
            observer.capped_memory('/group')
        with patch.object(observer.resource, 'getrlimit', return_value=(observer.FILE_MAX,) * 2), \
             patch.object(observer, 'memory_receipt', return_value={'memory.events': 'max 1'}), \
             self.assertRaisesRegex(ValueError, 'memory event'):
            observer.capped_memory('/group')

    def test_arm_requires_real_descendant_pidfd_before_publication(self):
        value, own, directory, guard = plan(), record(), Mock(), Mock()
        directory.read.return_value = dict(sampler=own, owner=value['owner'], nonce=value['nonce'], sdk_initialized=False)
        with patch.object(observer, 'process_record', return_value=own), \
             patch.object(observer, 'PinnedProcess') as pin:
            proof = observer.arm_sampler(directory, value, 41, guard)
            self.assertIs(proof, pin.return_value)
            guard.observe.assert_called_once_with(proof, 77)
            directory.publish.assert_called_once_with('armed.json', dict(sampler=own, owner=value['owner'], nonce=value['nonce']))
        for bad in (dict(uid=1000), dict(pid=41), dict(cgroup='/foreign')):
            directory.read.return_value = dict(sampler={**own, **bad}, owner=value['owner'], nonce=value['nonce'],
                                               sdk_initialized=False)
            with self.subTest(bad=bad), patch.object(observer, 'PinnedProcess') as pin, self.assertRaises(ValueError):
                observer.arm_sampler(directory, value, 41, guard)
            pin.assert_not_called()

    def test_arm_timeout_and_controller_death_never_enter_sdk(self):
        directory, controller, value, own = Mock(), Mock(), plan(), record()
        directory.stopped.return_value = False
        directory.exists.return_value = False
        ticks = iter([0., 0., 11.])
        with self.assertRaisesRegex(ValueError, 'within10s'):
            observer.wait_for_arm(directory, value, own, controller, clock=lambda: next(ticks), sleep=lambda _: None)
        controller.check.side_effect = ValueError('controller exited')
        with self.assertRaisesRegex(ValueError, 'controller exited'):
            observer.wait_for_arm(directory, value, own, controller, sleep=lambda _: None)

    def test_wrong_arm_does_not_reach_memory_recheck(self):
        directory, value, own = Mock(), plan(), record()
        directory.stopped.return_value = False
        directory.exists.return_value = True
        directory.read.return_value = dict(sampler=record(start=78), owner=value['owner'], nonce=value['nonce'])
        with patch.object(observer, 'capped_memory') as memory, self.assertRaisesRegex(ValueError, 'arm does not bind'):
            observer.wait_for_arm(directory, value, own, Mock())
        memory.assert_not_called()


class HookTests(unittest.TestCase):
    def fixture(self, raw=''):
        common = SimpleNamespace(verify_process=Mock(), check_pending=Mock())
        h = SimpleNamespace(query=Mock(return_value=raw), identity=Mock(return_value=77),
                            training=SimpleNamespace(require_idle=Mock()))
        guard = SimpleNamespace(group=record()['cgroup'], check=Mock(), observe=Mock())
        proof = SimpleNamespace(pid=42, expected=record(), check=Mock())
        return common, h, guard, proof

    def test_sampler_absent_is_valid_but_not_an_app_owner(self):
        common, h, guard, proof = self.fixture()
        with observer.scoped_helpers(common, h, guard, proof):
            h.training.require_idle()
            self.assertEqual(h.query('compute-apps', 'gpu_uuid,pid,process_name'), '')
        proof.check.assert_called()
        guard.observe.assert_not_called()

    def test_only_exact_sampler_is_filtered_real_apps_are_preserved(self):
        raw = f'{observer.GPU_UUID}, 42, sampler\n{observer.GPU_UUID}, 91, real-app'
        common, h, guard, proof = self.fixture(raw)
        with observer.scoped_helpers(common, h, guard, proof):
            self.assertEqual(h.query('compute-apps', 'gpu_uuid,pid,process_name'),
                             f'{observer.GPU_UUID}, 91, real-app')
            with self.assertRaisesRegex(ValueError, 'GPU work'):
                h.training.require_idle()

    def test_reused_root_wrong_uuid_uid_cgroup_never_allow_and_restore(self):
        for change in ('pid-reused', 'uuid', 'uid', 'cgroup'):
            with self.subTest(change=change):
                raw = f'{observer.GPU_UUID if change != "uuid" else "GPU-foreign"}, 42, sampler'
                common, h, guard, proof = self.fixture(raw)
                originals = h.query, h.identity, h.training, common.verify_process, common.check_pending
                if change == 'pid-reused':
                    proof.check.side_effect = ValueError('PID reused')
                elif change == 'uid':
                    proof.expected['uid'] = 1000
                elif change == 'cgroup':
                    proof.expected['cgroup'] = '/foreign'
                with self.assertRaises(ValueError), observer.scoped_helpers(common, h, guard, proof):
                    h.query('compute-apps', 'gpu_uuid,pid,process_name')
                self.assertEqual(originals, (h.query, h.identity, h.training, common.verify_process, common.check_pending))

    def test_escaped_app_rejected_before_release_and_resource_failure_before_pending(self):
        common, h, guard, proof = self.fixture()
        pending = common.check_pending
        with observer.scoped_helpers(common, h, guard, proof):
            with patch.object(observer, 'cgroup', return_value='/foreign'), self.assertRaisesRegex(ValueError, 'before release'):
                h.identity(91)
            guard.check.side_effect = ValueError('output bound')
            with self.assertRaisesRegex(ValueError, 'output bound'):
                common.check_pending()
        pending.assert_not_called()


class SamplerTests(unittest.TestCase):
    def test_sdk_initialization_failure_restores_retained_sampling_hooks(self):
        directory, value, own, controller = Mock(), plan(), record(), Mock()
        sampling = SimpleNamespace(guard=Mock(return_value=[]), publish=Mock(),
                                   catalogue=SimpleNamespace(validate_layout=Mock()))
        original_guard, original_publish = sampling.guard, sampling.publish
        structures = SimpleNamespace(_dcgmInit=Mock(side_effect=RuntimeError('SDK load failed')))
        with patch.object(observer, 'load_sampling', return_value=sampling), \
             patch.object(observer, 'capped_memory'), patch.object(observer, 'process_record', return_value=own), \
             patch.object(observer.importlib, 'import_module', side_effect=[structures, Mock()]), \
             patch.object(observer.Path, 'read_text', return_value='{}'), \
             self.assertRaisesRegex(RuntimeError, 'SDK load failed'):
            observer.sample_owned_watch(directory, value, own, controller)
        self.assertIs(sampling.guard, original_guard)
        self.assertIs(sampling.publish, original_publish)
        controller.check.assert_called_once()
        directory.check.assert_called_once()

    def test_pre_sdk_failure_and_partial_watch_failure_never_publish_success(self):
        for stage in ('validate', 'arm', 'watch', 'cleanup'):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp, \
                 observer.PrivateDirectory(Path(tmp), os.getuid()) as directory, \
                 patch.object(observer, 'validate_sampler_plan', return_value=record()) as validate, \
                 patch.object(observer, 'PinnedProcess') as pin, \
                 patch.object(observer, 'wait_for_arm') as arm, patch.object(observer, 'capped_memory'), \
                 patch.object(observer, 'verify_seals'):
                value = plan()
                value['owner']['uid'] = os.getuid()
                watch = Mock(return_value=dict(status='external_observation_collected_requires_window_review',
                    owned_watches_retired=True, embedded_retired=False, cleanup_errors=[]))
                if stage == 'validate':
                    validate.side_effect = ValueError('wrong limits')
                elif stage == 'arm':
                    arm.side_effect = ValueError('controller death')
                elif stage == 'watch':
                    watch.side_effect = RuntimeError('partial SDK initialization')
                result = observer.sampler(directory, value, run_watch=watch)
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(result['full_device_qualified'])
                if stage in ('validate', 'arm'):
                    watch.assert_not_called()
                    self.assertFalse(result['sdk_entry_attempted'])
                if stage != 'validate':
                    pin.return_value.__exit__.assert_called_once()
                self.assertEqual(directory.read('observer.json', os.getuid()), result)

    def test_normal_watch_still_does_not_qualify_hardware(self):
        with tempfile.TemporaryDirectory() as tmp, observer.PrivateDirectory(Path(tmp), os.getuid()) as directory, \
             patch.object(observer, 'validate_sampler_plan', return_value=record()), \
             patch.object(observer, 'PinnedProcess'), patch.object(observer, 'wait_for_arm'), \
             patch.object(observer, 'capped_memory'), patch.object(observer, 'verify_seals'), \
             patch.object(directory, 'file_identity', return_value=dict(bytes=3, sha256='a' * 64, uid=0)):
            value = plan()
            value['owner']['uid'] = os.getuid()
            watch = Mock(return_value=dict(status='external_observation_collected_requires_window_review',
                owned_watches_retired=True, embedded_retired=True, cleanup_errors=[]))
            result = observer.sampler(directory, value, run_watch=watch)
            self.assertNotEqual(result['status'], 'failed')
            self.assertTrue(result['sdk_entry_attempted'])
            for name in ('ordinary_timing', 'full_device_qualified', 'saturation_qualified',
                         'performance_qualified', 'full_goal_complete'):
                self.assertFalse(result[name])


class SamplingLoopTests(unittest.TestCase):
    def exercise(self, batches=None, *, cursor_backwards=False, stop=True,
                 hard_timeout=1, interrupt=False, split_callbacks=False):
        now = [1_000_000_000]
        def stamp():
            now[0] += 1
            return now[0]
        clock = SimpleNamespace(monotonic=lambda: now[0] / 1e9, monotonic_ns=stamp,
                                process_time_ns=lambda: 0,
                                sleep=lambda seconds: now.__setitem__(0, now[0] + int(seconds * 1e9)))
        original = {sig: object() for sig in (signal.SIGTERM, signal.SIGINT)}
        handlers = dict(original)
        def install(sig, handler):
            previous, handlers[sig] = handlers[sig], handler
            return previous
        signals = SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, signal=install)
        emitted, cursors = [], []
        helper = SimpleNamespace(validate_row=aggregate.validate_row, guard=lambda *_: [],
            clock_anchor=lambda: dict(kind='anchor', monotonic_ns=stamp(), realtime_ns=now[0] + 10**12,
                                      uncertainty_ns=0),
            publish=lambda path, value: emitted.append((path.name, value)),
            identity=lambda _: dict(start_ticks=77))
        def query(handle, group, fields, cursor, callback, user):
            index = len(cursors)
            cursors.append(cursor)
            batch = (batches[index] if batches is not None and index < len(batches)
                     else [self.raw(field, (index + 1) * 200000) for field in observer.FIELDS])
            if split_callbacks:
                for raw in batch:
                    callback(0, [raw], 1, user)
            else:
                callback(0, batch, len(batch), user)
            if interrupt:
                handlers[signal.SIGTERM](signal.SIGTERM, None)
            return -1 if cursor_backwards else (index + 1) * 200000
        agent = SimpleNamespace(dcgmFieldValueEnumeration_f=lambda callback: callback,
                                dcgmUpdateAllFields=Mock(), dcgmGetValuesSince=query)
        with tempfile.TemporaryDirectory() as tmp, patch.object(observer, 'time', clock), \
             patch.object(observer, 'signal', signals):
            folder = Path(tmp)
            if stop:
                (folder / 'stop').touch()
            error, result = None, None
            try:
                result = observer.collect_rows(helper, agent, None, None, None, folder, {}, hard_timeout)
            except Exception as caught:
                error = caught
            rows = aggregate.read_records(folder / 'samples.jsonl')
        self.assertEqual(handlers, original)
        return result, rows, emitted, cursors, error

    @staticmethod
    def raw(field=1001, stamp=200000, status=0, value=.5, field_type=100):
        return SimpleNamespace(fieldId=field, ts=stamp, status=status, fieldType=field_type,
                               value=SimpleNamespace(dbl=value))

    def test_valid_loop_records_events_readiness_stop_and_handler_restore(self):
        duplicate = [self.raw(field) for field in observer.FIELDS]
        for repeated, batches in ((False, None), (True, [duplicate, duplicate])):
            with self.subTest(repeated=repeated):
                result, rows, emitted, cursors, error = self.exercise(batches)
                self.assertIsNone(error)
                self.assertEqual(result, dict(samples=8 if repeated else 12, startup_invalid_samples=0,
                    cached_duplicates=4 if repeated else 0, no_data_events=0,
                    observation_seconds=0.4000000210000001, sampler_process_cpu_seconds=0.0))
                def anchor(stamp):
                    return dict(kind='anchor', monotonic_ns=stamp, realtime_ns=stamp + 10**12,
                                uncertainty_ns=0)
                expected = []
                for epoch, (before, after, observed, timestamp) in enumerate((
                        (1000000001, 1000000007, 1000000003, 200000),
                        (1200000009, 1200000014, 1200000010, 400000),
                        (1400000015, 1400000020, 1400000016, 600000))):
                    expected.append(anchor(before))
                    if epoch == 0:
                        expected.append(dict(kind='guard', monotonic_ns=1000000002, processes=[]))
                    expected.append(anchor(after))
                    if not (repeated and epoch == 1):
                        expected.extend(dict(kind='sample', gpu_id=0, field_id=field, field_type=100,
                            status=0, timestamp_us=timestamp, value=.5, raw_value_repr='0.5',
                            observed_monotonic_ns=observed + index)
                            for index, field in enumerate(observer.FIELDS))
                expected.append(anchor(1400000021))
                self.assertEqual(rows, expected)
                self.assertEqual(emitted, [('ready.json', dict(pid=os.getpid(), start_ticks=77,
                    owner={}, ready_ns=1000000008, fields=observer.FIELDS, period_us=200000))])
                self.assertEqual(cursors, [0, 200000, 400000])

    def test_post_ready_mixed_and_repeated_absence_has_no_sample_or_ready_credit(self):
        valid = [self.raw(field) for field in observer.FIELDS]
        absent = self.raw(stamp=0, status=-14, value=aggregate.NO_DATA_BLANK)
        batches = [valid, [absent, self.raw(1002, 400000), absent]]
        new = self.exercise(batches)
        self.assertIsNone(new[-1])
        self.assertEqual(new[0]['samples'], 9)
        self.assertEqual(new[0]['no_data_events'], 2)
        events = [row for row in new[1] if row['kind'] == 'no_data']
        self.assertEqual([row['row_index'] for row in events], [0, 2])
        for row in events:
            aggregate.validate_no_data(row)
            self.assertEqual((row['callback_count'], row['query_cursor_us'], row['next_query_cursor_us']),
                             (3, 200000, 400000))
        self.assertEqual(len(new[2]), 1)
        split = self.exercise(batches, split_callbacks=True)
        self.assertIsNone(split[-1])
        split_events = [row for row in split[1] if row['kind'] == 'no_data']
        self.assertEqual([(row['query_index'], row['callback_index'], row['row_index'])
                          for row in split_events], [(2, 1, 0), (2, 3, 0)])
        for row in split_events:
            aggregate.validate_no_data(row)
        startup = self.exercise([[absent]] * 3, stop=False, hard_timeout=.5)
        self.assertIsInstance(startup[-1], TimeoutError)
        self.assertEqual(startup[2], [])

    def test_changed_absence_real_error_conflict_and_backwards_cursor_still_fail(self):
        valid = [self.raw(field) for field in observer.FIELDS]
        for change in (dict(field=1007), dict(stamp=1), dict(status=-6), dict(status=0),
                       dict(value=aggregate.NO_DATA_BLANK + 1), dict(value=float('nan')),
                       dict(field_type=105)):
            args = dict(stamp=0, status=-14, value=aggregate.NO_DATA_BLANK)
            args.update(change)
            with self.subTest(change=change):
                result = self.exercise([valid, [self.raw(**args)]])
                self.assertIsInstance(result[-1], ValueError)
                self.assertFalse(any(row['kind'] == 'no_data' for row in result[1]))
        self.assertIsInstance(self.exercise(cursor_backwards=True)[-1], ValueError)
        self.assertIsInstance(self.exercise([valid, [self.raw(value=.6)]])[-1], ValueError)
        self.assertIsInstance(self.exercise(stop=False, hard_timeout=.3)[-1], TimeoutError)
        self.assertIsInstance(self.exercise(interrupt=True)[-1], InterruptedError)


class PreparationTests(unittest.TestCase):
    def test_worker_probe_only_changes_count_and_equal_mode(self):
        for workers in (320, 640, 1280):
            value = dict(worker_probe=True, recipe={**prepare.RECIPE, 'workers': workers, 'modes': ['equal']})
            self.assertEqual(prepare.expected_recipe(value), value['recipe'])
            for key, wrong in (('modes', ['priority']), ('gemm_tile_side', 64), ('priority_idle_us', 0)):
                changed = {**value, 'recipe': {**value['recipe'], key: wrong}}
                self.assertNotEqual(prepare.expected_recipe(changed), changed['recipe'])
        self.assertEqual(prepare.expected_recipe({}), prepare.RECIPE)
        for workers in (True, 0, 321, 2560):
            with self.assertRaises(ValueError):
                prepare.expected_recipe(dict(worker_probe=True, recipe=dict(workers=workers)))
        with self.assertRaises(ValueError):
            prepare.expected_recipe(dict(worker_probe=False, recipe=dict(workers=640)))
        pair = dict(native_equal=True, recipe={**prepare.RECIPE, 'modes': ['native', 'equal']})
        self.assertEqual(prepare.expected_recipe(pair), pair['recipe'])
        for key, wrong in (('modes', ['equal', 'native']), ('modes', ['native']),
                           ('modes', ['native', 'equal', 'priority']), ('workers', 640), ('period_us', 100000)):
            with self.subTest(key=key, wrong=wrong):
                changed = {**pair, 'recipe': {**pair['recipe'], key: wrong}}
                self.assertNotEqual(prepare.expected_recipe(changed), changed['recipe'])
        for bad in (dict(native_equal=False), dict(native_equal=1), dict(pair, worker_probe=True)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                prepare.expected_recipe(bad)
        with patch.object(prepare, 'observer_seals') as seals, self.assertRaises(ValueError):
            prepare.prepare(Path('/reference'), worker_probe=320, native_equal=True)
        seals.assert_not_called()

    def test_recipe_and_qualification_cannot_be_relaxed(self):
        value = dict(version=1, status='prepared_not_capture_ready', live_capture_enabled=False,
                     counter_collection=False, ordinary_timing=False,
                     recipe=dict(prepare.RECIPE), point=dict(prepare.FIXED_POINT),
                     gpu_uuid=observer.GPU_UUID, build=str(prepare.BUILD), observer_identities={'/sealed': 'b' * 64},
                     input_identities={'/data': dict(target='/data', bytes=1, sha256='c' * 64)},
                     full_device_qualified=False, saturation_qualified=False,
                     performance_qualified=False, full_goal_complete=False)
        for key, replacement in (('live_capture_enabled', True), ('full_device_qualified', True),
                                 ('ordinary_timing', True),
                                 ('point', {**prepare.FIXED_POINT, 'requests': 2}),
                                 ('recipe', {**prepare.RECIPE, 'workers': 8}), ('build', '/old/release')):
            with self.subTest(key=key), patch.object(observer, 'verify_seals') as seals, \
                 self.assertRaises(ValueError):
                prepare.validate_plan({**value, key: replacement})
            seals.assert_not_called()

    def test_omitting_any_mandatory_input_or_observer_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp, name) for name in ('candidate', 'ptx', 'adapter', 'binding')]
            for path in paths:
                path.write_text(path.name)
            inputs = {str(path): dict(target=str(path), bytes=path.stat().st_size, sha256=prepare.digest(path))
                      for path in paths[:2]}
            tools = {str(path): prepare.digest(path) for path in paths[2:]}
            value = dict(version=1, status='prepared_not_capture_ready', live_capture_enabled=False,
                         counter_collection=False, ordinary_timing=False, reference='/reference',
                         recipe=dict(prepare.RECIPE), point=dict(prepare.FIXED_POINT), gpu_uuid=observer.GPU_UUID,
                         build=str(prepare.BUILD), input_identities=inputs, observer_identities=tools,
                         full_device_qualified=False, saturation_qualified=False,
                         performance_qualified=False, full_goal_complete=False)
            with patch.object(prepare, 'observer_paths', return_value=(set(paths[3:]), {paths[2]})), \
                 patch.object(prepare, 'required_inputs', return_value=set(inputs)):
                prepare.validate_plan(value)
                for group in ('input_identities', 'observer_identities'):
                    for name in value[group]:
                        with self.subTest(group=group, name=name), self.assertRaisesRegex(ValueError, 'omitted'):
                            prepare.validate_plan({**value, group: {key: item for key, item in value[group].items()
                                                                  if key != name}})

    def test_changed_or_empty_helper_seals_cannot_be_prepared(self):
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            observer.verify_seals({})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, 'helper.py')
            path.write_text('original')
            seals = {str(path): prepare.digest(path)}
            observer.verify_seals(seals)
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                observer.verify_seals(seals)

    def test_removing_installed_tool_and_plan_entry_cannot_shrink_required_closure(self):
        tools = {observer.LIBRARIES / name for name in
                 ('libdcgm.so.4.0.0', 'libdcgmmoduleprofiling.so.4.0.0', 'libnvperf_dcgm_host.so')}
        tools.update(observer.BINDINGS / name for name in
                     ('dcgm_agent.py', 'dcgm_structs.py', 'dcgm_fields.py', 'dcgmvalue.py'))
        catalogue = dict(identities={str(path): 'a' * 64 for path in tools})
        with patch.object(prepare, 'read', return_value=catalogue), \
             patch.object(prepare.Path, 'is_file', return_value=False), \
             self.assertRaisesRegex(ValueError, 'runtime file is missing'):
            prepare.observer_paths()

    def test_preparation_rejects_incomplete_checkpoint_before_runtime_binding(self):
        for value in ({}, dict(status='failed'), dict(status='current_candidate_observed_controls_require_review',
                                                     input_identity_unchanged=False)):
            with self.subTest(value=value), patch.object(prepare, 'read', return_value=value), \
                 self.assertRaisesRegex(ValueError, 'checkpoint required'):
                prepare.qualify_checkpoint(Path('/reference'))

    def test_cold_import_and_help_have_no_capture_imports_or_processes(self):
        script = Path(prepare.__file__)
        result = subprocess.run([sys.executable, '-B', str(script), '--help'], check=True,
                                text=True, capture_output=True, timeout=10)
        self.assertIn('--check-plan', result.stdout)
        self.assertIn('--native-equal', result.stdout)
        for flags in (['--native-equal', '--worker-probe', '320'],
                      ['--native-equal', '--collect', '/missing-plan']):
            rejected = subprocess.run([sys.executable, '-B', str(script), *flags],
                                      text=True, capture_output=True, timeout=10)
            self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('--collect', result.stdout)
        self.assertNotIn('--sampler-folder', result.stdout)
        code = ('import sys; from unittest.mock import patch; '
                f'sys.path.insert(0,{str(script.parent)!r}); '
                'exec("with patch(\'subprocess.Popen\', side_effect=AssertionError(\'process\')):\\n '
                'import run_bert_device_observation\\n import bert_device_observer"); '
                'assert not any(n in sys.modules for n in '
                '["dcgm_sampling","dcgm_agent","dcgm_structs","torch",'
                '"collect_sustained_application","collect_application_block_colocation"])')
        subprocess.run([sys.executable, '-B', '-c', code], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
