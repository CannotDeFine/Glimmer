"""CPU controls for passive ordinary native identity/mapping preflight."""

from contextlib import ExitStack
import copy
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_bert_native_preflight as preflight


def status():
    return ('Uid:\t1003\t1003\t1003\t1003\nGid:\t1003\t1003\t1003\t1003\n'
            'Groups:\t1003\nNoNewPrivs:\t1\nCapEff:\t0\nCapPrm:\t0\nCapInh:\t0\n'
            'CapAmb:\t0\nCapBnd:\t000001ffffffffff\n')


def fake_guard():
    guard = preflight.MapGuard.__new__(preflight.MapGuard)
    guard.attempt, guard.group = Path('/ordinary'), '/task'
    guard.manifest = dict(uid=1003, gid=1003, application_environment={'LANG': 'C'})
    guard.wrapper, guard.commands = ['python', 'wrapper'], {'low': ['python', 'application']}
    guard.rows = {101: dict(role='low', pid=101, start=42, samples=[], map_sets={}, mappings={},
        terminal=None, first_sample_ns=None, last_sample_ns=None, max_sample_gap_ns=0, application_seen=False)}
    guard.handles = {101: 31}
    guard.maps = {101: Mock()}
    guard.maps[101].observe.return_value = dict(monotonic_ns=10, mappings={'/python': {'sha256': 'fixture'}})
    return guard, Mock(pid=101, poll=Mock(return_value=None))


def sample_context(command=None):
    stack = ExitStack()
    for target, value in (('identity', 42), ('cgroup', '/task'), ('native.relay.parent', os.getpid()),
                          ('os.getsid', 101), ('os.getpgid', 101),
                          ('native.relay.command_line', command or ['python', 'wrapper'])):
        stack.enter_context(patch('run_bert_native_preflight.' + target, return_value=value))
    stack.enter_context(patch.object(Path, 'samefile', return_value=True))
    stack.enter_context(patch.object(Path, 'read_bytes', return_value=b'LANG=C\0'))
    stack.enter_context(patch.object(Path, 'read_text', return_value=status()))
    stack.enter_context(patch.object(Path, 'exists', return_value=False))
    stack.enter_context(patch.object(preflight.time, 'monotonic_ns', return_value=5))
    return stack


def maps_fixture():
    names = ['/python']
    key = hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest()
    phases = ['wrapper', *(['application_startup'] * 3), 'prepared', 'prepared', 'released', 'released']
    frozen, observer = {'/python': {'sha256': 'fixture'}}, {'pid': 99, 'cgroup': '/task'}
    job = dict(application_identities={}, ready={}, application_launch_ns={})
    maps = {'roles': {}}
    for index, role in enumerate(('low', 'high')):
        identity = dict(pid=101 + index, start=42 + index)
        job['application_identities'][role] = identity
        job['ready'][role] = dict(ready_ns=45, released_ns=65)
        job['application_launch_ns'][role] = 15
        maps['roles'][role] = dict(identity, role=role, parent=99, cgroup='/task',
            terminal=dict(returncode=0, pidfd_terminal=True, monotonic_ns=90, wait_source='direct_parent_Popen.poll'),
            samples=[dict(begin_ns=(index + 1) * 10 - 1, monotonic_ns=(index + 1) * 10, phase=phase,
                          image='wrapper' if phase == 'wrapper' else 'application', map_set_sha256=key)
                     for index, phase in enumerate(phases)],
            map_sets={key: names}, mappings=frozen, first_sample_ns=10, last_sample_ns=80,
            max_sample_gap_ns=10, terminal_gap_ns=10)
    return maps, job, frozen, observer


class NativePreflightTests(unittest.TestCase):
    def test_real_cpu_child_maps_pidfd_caps_and_parent_wait(self):
        # Exercise real procfs/pidfd behavior without importing CUDA or Torch.
        paths = {Path(sys.executable).resolve()}
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith('/') and (
                    'x' in fields[1] or '.so' in fields[5]):
                paths.add(Path(fields[5]).resolve(strict=True))
        inputs = {str(path): dict(target=str(path), bytes=path.stat().st_size,
                                 sha256=preflight.digest(path)) for path in paths}
        frozen = preflight.native.mapping_inputs(inputs)
        argv = [sys.executable, '-I', '-S', '-B', '-c',
                'import sys; print("ready", flush=True); sys.stdin.buffer.read(1)']
        environment = {'LANG': 'C.UTF-8'}
        process = subprocess.Popen(['/usr/bin/setpriv', '--no-new-privs', *argv],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, env=environment)
        guard = preflight.MapGuard.__new__(preflight.MapGuard)
        guard.handles = {}
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0], 'CPU fixture startup timeout')
            self.assertEqual(process.stdout.readline(), b'ready\n')
            start = preflight.identity(process.pid)
            guard.group = preflight.cgroup(os.getpid())
            guard.manifest = dict(uid=os.getuid(), gid=os.getgid(), application_environment=environment)
            guard.wrapper, guard.commands = ['unused-wrapper'], {'low': argv}
            guard.handles[process.pid] = os.pidfd_open(process.pid)
            guard.maps = {process.pid: preflight.native.NativeMaps(inputs, frozen)}
            row = dict(role='low', pid=process.pid, start=start, samples=[], map_sets={}, mappings={},
                terminal=None, first_sample_ns=None, last_sample_ns=None, max_sample_gap_ns=0,
                application_seen=False, parent=os.getpid(), cgroup=guard.group)
            guard.rows = {process.pid: row}
            with patch.object(preflight.protocol, 'PYTHON', Path(sys.executable)):
                guard.sample(process, start)
                guard.sample(process, start)
                self.assertEqual(len(row['samples']), 2)
                self.assertIn(str(Path(sys.executable).resolve()), row['mappings'])
                process.stdin.write(b'x')
                process.stdin.flush()
                self.assertTrue(select.select([guard.handles[process.pid]], [], [], 5)[0],
                                'CPU fixture terminal timeout')
                guard.sample(process, start)
            self.assertEqual(row['terminal']['returncode'], 0)
            self.assertTrue(row['terminal']['pidfd_terminal'])
            self.assertEqual(row['terminal']['wait_source'], 'direct_parent_Popen.poll')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()
            guard.close()

    def test_caps0_keeps_bounding_set_distinct_from_effective_authority(self):
        self.assertNotEqual(int(preflight.caps0_status(status(), 1003, 1003)['CapBnd'], 16), 0)
        for field in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb'):
            with self.assertRaises(ValueError):
                preflight.caps0_status(status().replace(field + ':\t0', field + ':\t200000'), 1003, 1003)
        with self.assertRaises(ValueError):
            preflight.caps0_status(status().replace('NoNewPrivs:\t1', 'NoNewPrivs:\t0'), 1003, 1003)
        with self.assertRaises(ValueError):
            preflight.caps0_status(status(), 0, 1003)

    def test_sample_records_authenticated_map_union_and_gap(self):
        guard, process = fake_guard()
        with sample_context():
            guard.sample(process, 42)
            guard.maps[101].observe.return_value = dict(monotonic_ns=30, mappings={'/python': {'sha256': 'fixture'}})
            guard.sample(process, 42)
        row = guard.rows[101]
        self.assertEqual((row['first_sample_ns'], row['last_sample_ns'], row['max_sample_gap_ns']), (10, 30, 20))
        self.assertEqual(len(row['map_sets']), 1)
        self.assertEqual(len(row['samples']), 2)
        self.assertEqual(row['mappings'], {'/python': {'sha256': 'fixture'}})

    def test_live_identity_environment_and_unknown_map_fail_without_normalization(self):
        for target, value in (('identity', 43), ('cgroup', '/foreign'),
                              ('native.relay.command_line', ['other'])):
            guard, process = fake_guard()
            with sample_context(), patch('run_bert_native_preflight.' + target, return_value=value):
                with self.assertRaises(ValueError):
                    guard.sample(process, 42)
        guard, process = fake_guard()
        with sample_context(), patch.object(Path, 'read_bytes', return_value=b'LANG=C\0LANG=C\0'):
            with self.assertRaises(ValueError):
                guard.sample(process, 42)
        guard, process = fake_guard()
        guard.maps[101].observe.side_effect = ValueError('unfrozen mapping')
        with sample_context(), self.assertRaisesRegex(ValueError, 'unfrozen mapping'):
            guard.sample(process, 42)

    def test_missing_procfs_requires_terminal_pidfd_and_actual_parent_wait(self):
        guard, process = fake_guard()
        with sample_context(), patch('run_bert_native_preflight.native.relay.command_line', return_value=[]), \
                patch('select.select', return_value=([], [], [])), self.assertRaises(ValueError):
            guard.sample(process, 42)
        guard, process = fake_guard()
        process.poll.side_effect = [None, 0]
        with sample_context(), patch('run_bert_native_preflight.native.relay.command_line', return_value=[]), \
                patch('select.select', return_value=([31], [], [])), \
                patch.object(preflight.bridge, 'terminal', return_value=True):
            guard.sample(process, 42)
        self.assertEqual(guard.rows[101]['terminal']['returncode'], 0)
        self.assertEqual(guard.rows[101]['samples'], [])

    def test_application_may_not_return_to_wrapper(self):
        guard, process = fake_guard()
        with sample_context(['python', 'application']):
            guard.sample(process, 42)
        with sample_context(), self.assertRaises(ValueError):
            guard.sample(process, 42)

    def test_unit_command_is_unprivileged_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            with patch.object(preflight.native.ncu, 'unit_properties', return_value={'LoadState': 'not-found'}), \
                    patch.object(preflight.subprocess, 'run', return_value=Mock(returncode=0)) as run:
                result = preflight.launch_unit(attempt)
            argv = run.call_args.args[0]
            self.assertIn('--property=NoNewPrivileges=yes', argv)
            self.assertIn('--property=RuntimeMaxSec=800s', argv)
            self.assertEqual(run.call_args.kwargs['timeout'], 820)
            self.assertFalse(any('sudo' in value or 'root-supervisor' in value for value in argv))
            self.assertTrue(result['unit_removed'])

    def test_deadline_refuses_work_before_references_or_attempt_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'plan.json'
            path.write_text(json.dumps({'authorization_deadline_utc': '2000-01-01T00:00:00+00:00'}))
            with patch.object(preflight, 'references') as references, \
                    patch.object(preflight, 'create_attempt') as create, self.assertRaises(ValueError):
                preflight.collect(path)
            references.assert_not_called()
            create.assert_not_called()

    def test_sample_reanalysis_checks_phases_map_union_and_sampling_density(self):
        good = maps_fixture()
        preflight.audit_maps(*good)
        for defect in ('missing_phase', 'wrong_owner', 'wrong_set', 'unknown_map', 'tail_gap', 'live_gap'):
            maps, job, frozen, observer = copy.deepcopy(good)
            row = maps['roles']['low']
            if defect == 'missing_phase':
                row['samples'][0]['phase'] = 'application_startup'
            elif defect == 'wrong_owner':
                row['parent'] += 1
            elif defect == 'wrong_set':
                next(iter(row['map_sets'].values())).append('/unknown')
            elif defect == 'unknown_map':
                frozen.clear()
            elif defect == 'tail_gap':
                row['terminal']['monotonic_ns'] += 2000000000
                row['terminal_gap_ns'] += 2000000000
            else:
                for sample in row['samples'][3:]:
                    sample['monotonic_ns'] += 2000000000
                row['terminal']['monotonic_ns'] += 2000000000
                row['last_sample_ns'] += 2000000000
                row['max_sample_gap_ns'] += 2000000000
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                preflight.audit_maps(maps, job, frozen, observer)

    def test_boundary_crossing_sample_is_retained_but_does_not_witness_a_phase(self):
        ready = dict(ready_ns=45, released_ns=65)
        for begin, end in ((44, 46), (64, 66)):
            sample = dict(begin_ns=begin, monotonic_ns=end, image='application')
            self.assertEqual(preflight.sample_phase(sample, 15, ready, 90), 'transition')

    def test_resource_reanalysis_requires_actual_limits_and_zero_pressure(self):
        memory = {'memory.max': str(preflight.candidate.MEMORY_MAX), 'memory.swap.max': '0',
                  'memory.oom.group': '1', 'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0'}
        receipt = dict(per_file_limit=preflight.candidate.FILE_MAX, resource_checks=1,
            initial_memory=memory, last_memory=memory, pids_max='256', pids_events={'max': '0'},
            swap_current=0, last_output_bytes=100, disk_available_bytes=preflight.pair.DISK_FLOOR)
        preflight.audit_resources(receipt)
        for name, value in (('pids_max', '512'), ('pids_events', {'max': '1'}), ('swap_current', 1),
                            ('per_file_limit', 0), ('disk_available_bytes', 0)):
            with self.subTest(field=name), self.assertRaises(ValueError):
                preflight.audit_resources(dict(receipt, **{name: value}))

    def test_worker_rejects_copied_plan_mismatch_before_runtime_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            (attempt / 'plan.json').write_text('{}')
            (attempt / 'manifest.json').write_text(json.dumps({'plan_sha256': '0' * 64}))
            with patch.object(preflight.native.activity.numerical, 'authorization') as authorization:
                self.assertEqual(preflight.worker(attempt), 1)
            authorization.assert_not_called()
            report = json.loads((attempt / 'native-worker.json').read_text())
            self.assertIn('copied plan binding', report['error'])

    def test_analyzer_rejects_final_output_budget_before_input_use(self):
        for value in (-1, preflight.candidate.OUTPUT_MAX, '0', False):
            report = dict(status='native_environment_preflight_requires_independent_review',
                          input_identity_unchanged=True, output_bytes=value)
            with self.subTest(value=value), \
                    patch.object(preflight.protocol, 'PYTHON', Path(sys.prefix) / 'bin/python'), \
                    patch.object(preflight, 'read', side_effect=[report, {}, {}]), \
                    patch.object(preflight, 'digest') as digest, \
                    self.assertRaisesRegex(ValueError, 'final aggregate output budget'):
                preflight.analyze(Path('/unused'))
            digest.assert_not_called()


if __name__ == '__main__':
    unittest.main()
