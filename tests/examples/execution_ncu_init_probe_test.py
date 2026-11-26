"""CPU-only supervisor and rejection checks for the task-scoped NCU diagnostic."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_execution_ncu_init_probe as probe


class InitializationProbeTest(unittest.TestCase):
    def test_cold_help(self):
        with patch.object(sys, 'argv', ['probe', '--help']), \
                patch.object(probe.subprocess, 'Popen', side_effect=AssertionError('no process')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                probe.main()
            self.assertEqual(result.exception.code, 0)

    def test_nonroot_drop(self):
        command = probe.drop(dict(uid=1003, gid=1003))
        self.assertIn('--clear-groups', command)
        self.assertIn('--bounding-set=-all', command)
        self.assertIn('--no-new-privs', command)
        for uid in (0, -1, True, '1003'):
            with self.assertRaises(ValueError):
                probe.drop(dict(uid=uid, gid=1003))

    def test_debugger_is_bounded_and_exact(self):
        command = probe.debugger_command(123)
        self.assertEqual(command[command.index('-p') + 1], '123')
        self.assertIn("python gdb.execute('thread apply all bt 24') if gdb.selected_inferior().pid else None", command)
        self.assertIn('set debuginfod enabled off', command)
        self.assertNotIn('bt full', command)
        for pid in (0, -1, True, '123'):
            with self.assertRaises(ValueError):
                probe.debugger_command(pid)

    def test_wrong_incarnation_rejected_before_executable_access(self):
        with patch.object(probe, 'identity', return_value=7), \
                patch.object(probe.Path, 'stat', side_effect=AssertionError('too late')):
            with self.assertRaisesRegex(ValueError, 'start/cgroup'):
                probe.validate_target(dict(pid=123, start=8), {}, '/owned')

    def test_foreign_cgroup_rejected_before_executable_access(self):
        with patch.object(probe, 'identity', return_value=8), \
                patch.object(probe, 'cgroup', return_value='/foreign'), \
                patch.object(probe.Path, 'stat', side_effect=AssertionError('too late')):
            with self.assertRaisesRegex(ValueError, 'start/cgroup'):
                probe.validate_target(dict(pid=123, start=8), {}, '/owned')

    def test_root_collector_rejected_before_sudo(self):
        with patch.object(probe.os, 'geteuid', return_value=0), \
                patch.object(probe, 'command', side_effect=AssertionError('no command')):
            with self.assertRaisesRegex(ValueError, 'non-root'):
                probe.collect()

    def test_foreign_unit_ownership(self):
        state = dict(Description='expected', ControlGroup='/user/own.service')
        self.assertTrue(probe.owned_unit(state, 'own.service', 'expected'))
        self.assertFalse(probe.owned_unit(state, 'own.service', 'different'))
        self.assertFalse(probe.owned_unit(state, 'other.service', 'expected'))

    def test_atomic_handshake_rejects_reuse(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'ready.json'
            probe.publish_json(path, dict(pid=1, start=2))
            with self.assertRaises(FileExistsError):
                probe.publish_json(path, dict(pid=3, start=4))
            self.assertEqual(probe.read(path), dict(pid=1, start=2))

    def test_changed_profiler_baseline_rejected(self):
        previous = dict(ncu_sha256='old', ncu_exec_sha256='old', range_library='/range',
                        range_library_sha256='old')
        with patch.object(probe, 'digest', side_effect=['old', 'old', 'changed']):
            with self.assertRaisesRegex(ValueError, 'changed after failure'):
                probe.verify_profiler_baseline(previous)

    def manifest(self):
        return dict(uid=1003, gid=1003, service_cap_sys_admin=True, gpu_uuid='GPU-test',
                    range_library='/range', environment={key: 'sealed' for key in probe.SERVICE_ENVIRONMENT_KEYS})

    def service_status(self, cap=probe.CAP_SYS_ADMIN):
        return ('Uid:\t1003 1003 1003 1003\nGid:\t1003 1003 1003 1003\nGroups:\t\nNoNewPrivs:\t1\n' +
                ''.join(f'{name}:\t{cap:016x}\n' for name in
                        ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')))

    def test_capability_only_for_explicit_service(self):
        manifest = self.manifest()
        self.assertIn('--ambient-caps=-all', probe.drop(manifest))
        actual = probe.drop(manifest, service=True)
        for option in ('inh-caps', 'ambient-caps', 'bounding-set'):
            self.assertIn('--' + option + '=-all,+sys_admin', actual)
        self.assertIn('--no-new-privs', actual)
        manifest['service_cap_sys_admin'] = False
        self.assertIn('--ambient-caps=-all', probe.drop(manifest, service=True))

    def test_exact_actual_capabilities_and_identity(self):
        manifest = self.manifest()
        probe.validate_service_status(self.service_status(), manifest)
        for cap in (0, probe.CAP_SYS_ADMIN | 1, 1):
            with self.assertRaisesRegex(ValueError, 'capability'):
                probe.validate_service_status(self.service_status(cap), manifest)
        for field in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'):
            with self.assertRaisesRegex(ValueError, 'capability'):
                probe.validate_service_status(self.service_status().replace(
                    f'{field}:\t0000000000200000', f'{field}:\t0000000000000000'), manifest)
        for status in (self.service_status().replace('1003', '0'),
                       self.service_status().replace('NoNewPrivs:\t1', 'NoNewPrivs:\t0'),
                       self.service_status().replace('Groups:\t\n', 'Groups:\t1003\n')):
            with self.assertRaises(ValueError):
                probe.validate_service_status(status, manifest)
        manifest['service_cap_sys_admin'] = False
        probe.validate_service_status(self.service_status(0), manifest)
        with self.assertRaises(ValueError):
            probe.validate_service_status(self.service_status(), manifest)

    def test_direct_exec_command_preserves_clean_profiler(self):
        manifest = self.manifest()
        argv = probe.capable_service_command(Path('/attempt'), manifest, dict(pid=7, start=8))
        self.assertIn('/usr/bin/env', argv)
        self.assertNotIn('/usr/bin/python3', argv)
        self.assertNotIn('/bin/sh', argv)
        self.assertIn('--persistent-workers', argv)
        self.assertEqual(argv[argv.index('--persistent-workers') + 1], '320')
        self.assertIn('CUDA_VISIBLE_DEVICES=GPU-test', argv)
        self.assertNotIn('NVTX_INJECTION64_PATH=libcuda-injection.so', argv)
        for key in ('LD_PRELOAD', 'LD_AUDIT', 'NVTX_INJECTION64_PATH', 'ARBITRARY'):
            changed = self.manifest()
            changed['environment'][key] = 'bad'
            with self.assertRaisesRegex(ValueError, 'keys'):
                probe.capable_service_command(Path('/attempt'), changed, dict(pid=7, start=8))
        manifest['service_cap_sys_admin'] = False
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            probe.capable_service_command(Path('/attempt'), manifest, dict(pid=7, start=8))

    def test_actual_service_environment_is_verified(self):
        manifest = self.manifest()
        manifest['ncu_injection_seals'] = {'injection': 'sealed'}
        actual = {key.encode(): value.encode() for key, value in
                  probe.capable_service_environment(Path('/attempt'), manifest).items()}
        actual[b'NVTX_INJECTION64_PATH'] = b'libcuda-injection.so'
        with patch.object(probe, 'proc_environment', return_value=actual), \
                patch.object(probe, 'proc_mapped_paths', return_value={}), \
                patch.object(probe, 'verified_preload', return_value='verified') as verify:
            self.assertEqual(probe.validate_service_environment(dict(pid=7), Path('/attempt'), manifest), 'verified')
            verify.assert_called_once_with('', None, manifest['ncu_injection_seals'], {})
            actual[b'LD_AUDIT'] = b''
            with self.assertRaisesRegex(ValueError, 'LD_AUDIT'):
                probe.validate_service_environment(dict(pid=7), Path('/attempt'), manifest)
            del actual[b'LD_AUDIT']
            actual[b'LD_LIBRARY_PATH'] = b'changed'
            with self.assertRaisesRegex(ValueError, 'values'):
                probe.validate_service_environment(dict(pid=7), Path('/attempt'), manifest)

    def test_pipe_release_requires_exact_live_identity(self):
        record = dict(release='start', pid=7, start=8)
        with patch.object(probe, 'identity', return_value=8), patch.object(probe, 'cgroup', return_value='/unit'):
            self.assertEqual(probe.release_record(json.dumps(record), '/unit'), record)
            for changed in (dict(record, extra=1), dict(record, pid=True), dict(record, start=9),
                            dict(record, release='run')):
                with self.assertRaises(ValueError):
                    probe.release_record(json.dumps(changed), '/unit')
            for text in ('', 'start', '{', '{"release":"start","pid":7,"pid":7,"start":8}'):
                with self.assertRaises(ValueError):
                    probe.release_record(text, '/unit')

    def test_service_discovery_rejects_ambiguous_inode(self):
        with patch.object(probe, 'group_members', return_value={7, 8}), \
                patch.object(probe, 'identity', return_value=9), \
                patch.object(probe.Path, 'stat', return_value=Mock(st_dev=1, st_ino=2)), \
                patch.object(probe.os, 'pidfd_open', side_effect=AssertionError('no attach')):
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                probe.discover_service('/unit', ['service'], {})

    def test_discovery_closes_pidfd_on_validation_failure(self):
        with patch.object(probe, 'group_members', return_value={7}), \
                patch.object(probe, 'identity', return_value=9), \
                patch.object(probe.Path, 'stat', return_value=Mock(st_dev=1, st_ino=2)), \
                patch.object(probe.os, 'pidfd_open', return_value=42), \
                patch.object(probe.os, 'close') as close, \
                patch.object(probe, 'validate_target', side_effect=ValueError('identity')):
            with self.assertRaises(ValueError):
                probe.discover_service('/unit', ['service'], {})
            close.assert_called_once_with(42)

    def test_capability_flag_is_collect_only_and_explicit(self):
        with patch.object(probe, 'collect', return_value=0) as collect:
            with patch.object(sys, 'argv', ['probe', '--collect']):
                probe.main()
            collect.assert_called_once_with(False, counter_probe=False)
            collect.reset_mock()
            with patch.object(sys, 'argv', ['probe', '--collect', '--service-cap-sys-admin']):
                probe.main()
            collect.assert_called_once_with(True, counter_probe=False)
        with patch.object(sys, 'argv', ['probe', '--root', '/tmp', '--service-cap-sys-admin']), \
                contextlib.redirect_stderr(io.StringIO()), patch.object(probe, 'supervise') as supervise:
            with self.assertRaises(SystemExit):
                probe.main()
            supervise.assert_not_called()

    def test_compatibility_needs_clean_target_and_complete_oracle(self):
        report = dict(runtime_ready=True, service_terminal=True, service=dict(pid=7),
                      returncodes=dict(client=0, ncu=0, gdb=0))
        stack = '[Inferior 1 (process 7) exited normally]\n'
        client = 'Ordinary Runtime priority application verified: role=high; elements=81920'
        self.assertTrue(probe.compatibility_passed(report, stack, client))
        self.assertFalse(probe.compatibility_passed(report, stack.replace('7', '8'), client))
        self.assertFalse(probe.compatibility_passed(report, stack + 'received signal SIGSEGV', client))
        self.assertFalse(probe.compatibility_passed(report, stack, 'ready only'))
        for name in ('runtime_ready', 'service_terminal'):
            self.assertFalse(probe.compatibility_passed(dict(report, **{name: False}), stack, client))
        for name in ('client', 'ncu', 'gdb'):
            changed = dict(report, returncodes=dict(report['returncodes'], **{name: 1}))
            self.assertFalse(probe.compatibility_passed(changed, stack, client))

    def test_cleanup_requires_containment_for_remaining_descendants(self):
        report = dict(status='diagnostic', compatibility_passed=True)
        with patch.object(probe, 'remaining_descendants', return_value={7}):
            self.assertTrue(probe.retire_owned([], None, '/unit', [], report))
        self.assertFalse(report['compatibility_passed'])
        self.assertEqual(report['remaining_descendants'], [7])

    def test_clean_retirement_preserves_compatibility(self):
        process = Mock(pid=7, stdin=None, returncode=0)
        process.poll.return_value = 0
        report = dict(compatibility_passed=True)
        with patch.object(probe, 'remaining_descendants', return_value=set()):
            self.assertFalse(probe.retire_owned([process], None, '/unit', [], report))
        self.assertTrue(report['compatibility_passed'])
        process.terminate.assert_not_called()

    def test_cleanup_failure_cannot_pass(self):
        process = Mock(pid=7, stdin=None, returncode=None)
        process.poll.return_value = None
        process.wait.side_effect = probe.subprocess.TimeoutExpired('fixture', 2)
        report = dict(compatibility_passed=True)
        with patch.object(probe, 'remaining_descendants', return_value=set()):
            self.assertTrue(probe.retire_owned([process], None, '/unit', [], report))
        self.assertFalse(report['compatibility_passed'])
        self.assertEqual(report['child_cleanup'][0]['signals'], ['SIGTERM', 'SIGKILL'])

    def test_baseline_accepts_only_supervisor_and_exact_ancestors(self):
        with patch.object(probe.os, 'getpid', return_value=7), \
                patch.object(probe, 'identity', side_effect=lambda pid: pid + 10), \
                patch.object(probe, 'cgroup', return_value='/unit'), \
                patch.object(probe, 'task_ancestors', return_value=[dict(pid=6, start=16)]), \
                patch.object(probe, 'group_members', return_value={6, 7}) as members:
            self.assertEqual(probe.authenticated_baseline('/unit'),
                             [dict(pid=7, start=17), dict(pid=6, start=16)])
            members.return_value = {6, 7, 8}
            with self.assertRaisesRegex(ValueError, 'foreign initial'):
                probe.authenticated_baseline('/unit')

    def test_reused_baseline_pid_is_not_exempt_from_containment(self):
        with patch.object(probe, 'group_members', return_value={6, 7}), \
                patch.object(probe, 'identity', return_value=17), \
                patch.object(probe, 'cgroup', return_value='/unit'):
            self.assertEqual(probe.remaining_descendants('/unit',
                             [dict(pid=7, start=17), dict(pid=6, start=16)]), {6})

    def test_containment_fd_identity_mismatch_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            attempt = Path('/attempt-test')
            group = '/' + probe.unit_name(attempt)
            directory = root / group.lstrip('/')
            directory.mkdir()
            (directory / 'cgroup.kill').touch()
            with patch.object(probe, 'CGROUP_ROOT', root), \
                    patch.object(probe.os, 'open', return_value=42), \
                    patch.object(probe.os, 'fstat', return_value=Mock(st_dev=-1, st_ino=-1)), \
                    patch.object(probe.os, 'close') as close:
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    probe.open_containment(group, attempt)
                close.assert_called_once_with(42)

    def test_containment_cannot_target_other_unit(self):
        with patch.object(probe.os, 'open', side_effect=AssertionError('never open')):
            with self.assertRaisesRegex(ValueError, 'wrong containment unit'):
                probe.open_containment('/foreign.service', Path('/attempt-test'))

    def counter_csv(self, changed=None, *, extra_metric=None):
        fields = list(probe.CSV_IDENTITY + probe.COUNTER_METRICS)
        units = {key: '' for key in fields}
        units.update(zip(probe.COUNTER_METRICS, ('SM', '', '', 'cycle', 'inst')))
        row = dict(zip(probe.CSV_IDENTITY, ('0', '7', probe.SERVICE.name, '127.0.0.1', 'binary_high',
                                           '1', '13', '(32, 1, 1)', '(320, 1, 1)', '0', '7.0')))
        row.update(zip(probe.COUNTER_METRICS, ('80', '320', '32', '1,234', '567.0')))
        row.update(changed or {})
        if extra_metric:
            fields.append(extra_metric)
            row[extra_metric] = '1'
            units[extra_metric] = 'inst'
        output = io.StringIO()
        writer = csv.DictWriter(output, fields, quoting=csv.QUOTE_ALL, lineterminator='\n')
        writer.writeheader()
        writer.writerow(units)
        writer.writerow(row)
        return output.getvalue()

    def test_counter_access_does_not_qualify_sm_coverage(self):
        result = probe.audit_counter_csv(self.counter_csv(), 7)
        self.assertTrue(result['positive_hardware_counters'])
        self.assertEqual(result['metrics']['sm__cycles_active.sum'], 1234)
        for field in ('sm_coverage_verified', 'full_device_qualified', 'performance_qualified'):
            self.assertFalse(result[field])

    def test_counter_identity_and_geometry_rejections(self):
        for name, value in (('Process ID', '8'), ('Kernel Name', 'other'), ('Context', '0'), ('Stream', '0'),
                            ('Process Name', 'python3'), ('Device', '1'), ('CC', '8.0'),
                            ('Block Size', '(128, 1, 1)'), ('Grid Size', '(319, 1, 1)'),
                            ('launch__sm_count', '79'), ('launch__grid_size', '319'),
                            ('launch__block_size', '128'), ('ID', '-1')):
            with self.subTest(name=name), self.assertRaises(ValueError):
                probe.audit_counter_csv(self.counter_csv({name: value}), 7)

    def test_counter_numeric_errors_and_nonpositive_values_rejected(self):
        for value in ('nan', 'inf', 'N/A', 'ERR_NVGPUCTRPERM', '1e3', '1,23', '1,,000',
                      '-1', '0', '1.5', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.audit_counter_csv(self.counter_csv({'sm__cycles_active.sum': value}), 7)
        with self.assertRaises(ValueError):
            probe.audit_counter_csv(self.counter_csv({'sm__inst_executed.sum': '0'}), 7)

    def test_counter_csv_inventory_and_truncation_rejected(self):
        content = self.counter_csv()
        header, units, row = content.splitlines(keepends=True)
        bad_inputs = (content + row, content + header, header + units,
                      content.rstrip('\n'), content[:-3] + '\n', content.replace('"cycle"', '"ns"'),
                      content.replace('"ID",', '"ID","ID",'),
                      self.counter_csv(extra_metric='sm__other.sum'),
                      content.replace('sm__inst_executed.sum', 'sm__inst_executed.min'),
                      '==ERROR== permission denied\n' + content)
        for bad in bad_inputs:
            with self.subTest(bad=bad[:80]), self.assertRaises(ValueError):
                probe.audit_counter_csv(bad, 7)

    def test_counter_selection_is_one_nonwarmup_kernel_replay(self):
        argv = probe.ncu_probe_command(Path('/attempt'), ['target'], True)
        expected = {'--kernel-name-base': 'function', '--kernel-name': 'regex:^binary_high$',
                    '--launch-skip': '1', '--launch-count': '1', '--kill': 'no', '--replay-mode': 'kernel',
                    '--metrics': ','.join(probe.COUNTER_METRICS), '--export': '/attempt/capture'}
        for key, value in expected.items():
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertNotIn('--force-overwrite', argv)
        self.assertNotIn('--export', probe.ncu_probe_command(Path('/attempt'), ['target'], False))
        exported = probe.counter_export_command(Path('/attempt'))
        self.assertEqual(exported[exported.index('--metrics') + 1], ','.join(probe.COUNTER_METRICS))
        self.assertNotIn('--launch-count', exported)
        self.assertNotIn('--kernel-name', exported)

    def test_counter_opt_in_requires_capability_and_collect(self):
        with patch.object(probe, 'collect', return_value=0) as collect, \
                patch.object(sys, 'argv', ['probe', '--collect', '--service-cap-sys-admin', '--counter-probe']):
            probe.main()
            collect.assert_called_once_with(True, counter_probe=True)
        for argv in (['probe', '--collect', '--counter-probe'], ['probe', '--client', '/tmp', '--counter-probe']):
            with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                probe.main()
        for capable, counters in ((False, True), (True, 1), ('true', False)):
            with self.assertRaises(ValueError):
                probe.validate_probe_modes(capable, counters)

    def test_offline_metric_support_is_static_and_sealed(self):
        output = 'Chip gv100\nsm__cycles_active.sum Counter cycle description\nsm__inst_executed.sum Counter inst description\n'
        with patch.object(probe, 'command', return_value=output) as command, \
                patch.object(probe, 'digest', return_value='sealed'):
            result = probe.offline_metric_support()
            self.assertFalse(result['gpu_capture'])
            self.assertEqual(result['ncu_exec_sha256'], 'sealed')
            self.assertIn('--chips', command.call_args.args[0])
            self.assertEqual(command.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            for bad in (output.replace('Counter inst', 'Metric inst'), output.replace('gv100', 'ga100'),
                        output + 'sm__cycles_active.sum Counter cycle duplicate\n'):
                command.return_value = bad
                with self.assertRaises(ValueError):
                    probe.offline_metric_support()

    def test_counter_artifact_rejects_empty_and_symlink(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            report = directory / 'capture.ncu-rep'
            report.touch()
            with self.assertRaises(ValueError):
                probe.bounded_regular_digest(report)
            alias = directory / 'alias.ncu-rep'
            alias.symlink_to(report)
            with self.assertRaises(ValueError):
                probe.bounded_regular_digest(alias)

    def test_counter_report_must_not_change_during_export(self):
        with patch.object(probe, 'bounded_regular_digest', return_value='changed'), \
                patch.object(probe.Path, 'read_text', side_effect=AssertionError('no CSV read')):
            with self.assertRaisesRegex(ValueError, 'report changed'):
                probe.audit_counter_artifacts(Path('/attempt'), 7, 'original')

    def test_counter_csv_must_match_its_recorded_digest(self):
        with patch.object(probe, 'bounded_regular_digest', side_effect=['report', 'csv', 'changed']), \
                patch.object(probe.Path, 'read_text', return_value=self.counter_csv()):
            with self.assertRaisesRegex(ValueError, 'CSV changed'):
                probe.audit_counter_artifacts(Path('/attempt'), 7, 'report')
        with patch.object(probe, 'bounded_regular_digest', side_effect=['report', 'csv', 'csv']), \
                patch.object(probe.Path, 'read_text', return_value=self.counter_csv()):
            artifacts, audit = probe.audit_counter_artifacts(Path('/attempt'), 7, 'report')
            self.assertEqual(artifacts, dict(report_sha256='report', csv_sha256='csv'))
            self.assertTrue(audit['positive_hardware_counters'])

    def counter_diagnostic(self, directory):
        (directory / 'capture.ncu-rep').write_bytes(b'fixture report bytes')
        (directory / 'counter-metrics.csv').write_text(self.counter_csv())
        artifacts, audited = probe.audit_counter_artifacts(
            directory, 7, probe.bounded_regular_digest(directory / 'capture.ncu-rep'))
        return dict(compatibility_passed=True, counter_collection=True, counter_access_verified=True,
                    sm_coverage_verified=False, full_device_qualified=False, service=dict(pid=7),
                    counter_artifacts=artifacts, counter_audit=audited)

    def supervised_counter_fixture(self, *, counters=True, export_returncode=0, csv_content=None):
        """Exercise the real launch closure, artifact audit and retirement without processes."""
        with tempfile.TemporaryDirectory() as name, contextlib.ExitStack() as mocks:
            directory = Path(name)
            manifest = dict(self.manifest(), uid=probe.os.getuid(), counter_probe=counters, seals={})
            probe.save(directory / 'manifest.json', manifest)
            group = '/test/' + probe.unit_name(directory)
            service = dict(pid=7, start=107)
            processes, logs = [], []
            client_input = io.BytesIO()
            original_read_text = Path.read_text

            def read_text(path, *args, **kwargs):
                if path == Path('/proc/7/status'):
                    return self.service_status() + 'TracerPid:\t13\n'
                if path == Path('/proc/7/maps'):
                    return 'fixture mappings\n'
                return original_read_text(path, *args, **kwargs)

            def launch(argv, *, stdout, stdin, **kwargs):
                index = len(processes)
                self.assertLess(index, 4, 'unexpected subprocess launch')
                logs.append(stdout)
                terminal_code = export_returncode if index == 3 else 0
                process = Mock(pid=11 + index, stdin=client_input if index == 0 else None,
                               returncode=terminal_code)
                process.poll.side_effect = lambda: terminal_code if index == 3 or client_input.closed else None
                process.wait.return_value = terminal_code
                processes.append(process)
                if index == 0:
                    self.assertEqual(stdin, probe.subprocess.PIPE)
                    probe.publish_json(directory / 'client-ready.json', dict(pid=11, start=111))
                    stdout.write('Ordinary Runtime application ready: role=high\n'
                                 'Ordinary Runtime priority application verified: role=high; elements=10240\n')
                elif index == 1:
                    stdout.write('Execution service ready: fixture\n')
                    if counters:
                        (directory / 'capture.ncu-rep').write_bytes(b'fixture report bytes')
                elif index == 2:
                    stdout.write('GLIMMER_GDB_ATTACHED\n[Inferior 1 (process 7) exited normally]\n')
                else:
                    self.assertEqual(argv, probe.counter_export_command(directory))
                    self.assertTrue(client_input.closed, 'export must follow fixture retirement')
                    stdout.write(self.counter_csv() if csv_content is None else csv_content)
                stdout.flush()
                return process

            def mock(target, name, **kwargs):
                return mocks.enter_context(patch.object(target, name, **kwargs))

            mocks.enter_context(patch.dict(probe.os.environ, {}, clear=True))
            mock(probe.os, 'getresuid', return_value=(0, 0, 0))
            mock(probe.os, 'umask')
            mock(probe.resource, 'setrlimit')
            mock(probe, 'cgroup', return_value=group)
            mock(probe, 'identity', side_effect=lambda pid: pid + 100)
            mock(probe, 'memory_receipt', return_value={'memory.events': 'high 0\nmax 0\noom 0\n'})
            mock(probe, 'authenticated_baseline', return_value=[])
            mock(probe, 'open_containment', return_value=42)
            mock(probe, 'check_owned_gpu', return_value=[])
            mock(probe, 'proc_mapped_paths', return_value={})
            mock(probe, 'discover_service', return_value=(service, 43))
            mock(probe, 'validate_target')
            mock(probe, 'validate_service_environment', return_value={})
            mock(probe, 'pidfd_terminal', side_effect=lambda fd: client_input.closed)
            mock(probe, 'remaining_descendants', return_value=set())
            mock(probe.Path, 'read_text', autospec=True, side_effect=read_text)
            mock(probe.subprocess, 'Popen', side_effect=launch)
            mock(probe.subprocess, 'run', side_effect=AssertionError('no real subprocess'))
            mock(probe.time, 'sleep')
            close = mock(probe.os, 'close')
            write = mock(probe.os, 'write')
            signal = mock(probe.signal, 'pidfd_send_signal')
            audit = mock(probe, 'audit_counter_artifacts', wraps=probe.audit_counter_artifacts)
            result = probe.supervise(directory)
            report = probe.read(directory / 'diagnostic.json')
            self.assertEqual(len(processes), 4 if counters else 3, report.get('error'))
            self.assertTrue(all(log.closed for log in logs))
            self.assertEqual([call.args for call in close.call_args_list], [(43,), (42,)])
            write.assert_not_called()
            signal.assert_not_called()
            for process in processes:
                process.wait.assert_called_once_with(timeout=2)
                process.terminate.assert_not_called()
                process.kill.assert_not_called()
            self.assertEqual(report['remaining_descendants'], [])
            self.assertEqual(report['cleanup_errors'], [])
            self.assertEqual(len(report['child_cleanup']), len(processes))
            self.assertTrue(all(row['signals'] == [] for row in report['child_cleanup']))
            for field in ('sm_coverage_verified', 'full_device_qualified'):
                self.assertFalse(report[field])
            self.assertTrue(report['compatibility_passed'])
            self.assertEqual(audit.call_count, int(counters and export_returncode == 0))
            if report['counter_access_verified']:
                self.assertEqual(probe.audit_counter_postflight(
                    directory, report, unit_removed=True, inputs_unchanged=True), report['counter_audit'])
            elif counters:
                with self.assertRaisesRegex(ValueError, 'complete inner'):
                    probe.audit_counter_postflight(directory, report, unit_removed=True, inputs_unchanged=True)
            return result, report

    def test_supervisor_exports_after_reading_debugger_log(self):
        result, report = self.supervised_counter_fixture()
        self.assertEqual(result, 0)
        self.assertEqual(report['status'], 'diagnostic_requires_independent_review')
        self.assertEqual(report['counter_export_returncode'], 0)
        self.assertTrue(report['counter_collection'])
        self.assertTrue(report['counter_access_verified'])
        self.assertEqual(report['counter_audit']['metrics']['sm__cycles_active.sum'], 1234)

    def test_supervisor_export_failure_preserves_failure_and_retires_children(self):
        result, report = self.supervised_counter_fixture(export_returncode=1)
        self.assertEqual(result, 1)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['counter_export_returncode'], 1)
        self.assertIn('counter export must retire normally', report['error'])
        self.assertFalse(report['counter_collection'])
        self.assertFalse(report['counter_access_verified'])

    def test_supervisor_export_audit_failure_cannot_pass(self):
        result, report = self.supervised_counter_fixture(csv_content=self.counter_csv({'Process ID': '8'}))
        self.assertEqual(result, 1)
        self.assertEqual(report['status'], 'failed')
        self.assertIn('counter launch identity differs', report['error'])
        self.assertFalse(report['counter_collection'])
        self.assertFalse(report['counter_access_verified'])

    def test_supervisor_initialization_does_not_export_counters(self):
        result, report = self.supervised_counter_fixture(counters=False)
        self.assertEqual(result, 0)
        self.assertEqual(report['status'], 'diagnostic_requires_independent_review')
        self.assertNotIn('counter_export_command', report)
        self.assertFalse(report['counter_collection'])
        self.assertFalse(report['counter_access_verified'])

    def test_counter_postflight_rechecks_final_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            diagnostic = self.counter_diagnostic(directory)
            self.assertEqual(probe.audit_counter_postflight(directory, diagnostic, unit_removed=True,
                             inputs_unchanged=True), diagnostic['counter_audit'])
            for filename in ('capture.ncu-rep', 'counter-metrics.csv'):
                diagnostic = self.counter_diagnostic(directory)
                path = directory / filename
                path.write_bytes(path.read_bytes() + b'changed\n')
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    probe.audit_counter_postflight(directory, diagnostic, unit_removed=True,
                                                   inputs_unchanged=True)

    def test_counter_postflight_rejects_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            for filename in ('capture.ncu-rep', 'counter-metrics.csv'):
                diagnostic = self.counter_diagnostic(directory)
                (directory / filename).unlink()
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    probe.audit_counter_postflight(directory, diagnostic, unit_removed=True,
                                                   inputs_unchanged=True)

    def test_counter_postflight_rejects_mismatched_pid_audit_and_claims(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            diagnostic = self.counter_diagnostic(directory)
            changes = [('service', 'pid', 8), ('counter_audit', 'service_pid', 8),
                       ('counter_audit', 'sm_coverage_verified', True),
                       ('counter_audit', 'full_device_qualified', True),
                       ('counter_audit', 'performance_qualified', True),
                       ('counter_audit', 'performance_qualified', 0),
                       ('counter_artifacts', 'csv_sha256', '0' * 64)]
            for section, field, value in changes:
                changed = copy.deepcopy(diagnostic)
                changed[section][field] = value
                with self.subTest(section=section, field=field), self.assertRaises(ValueError):
                    probe.audit_counter_postflight(directory, changed, unit_removed=True, inputs_unchanged=True)
            for field in ('compatibility_passed', 'counter_collection', 'counter_access_verified',
                          'sm_coverage_verified', 'full_device_qualified'):
                changed = dict(diagnostic, **{field: not diagnostic[field]})
                with self.subTest(field=field), self.assertRaises(ValueError):
                    probe.audit_counter_postflight(directory, changed, unit_removed=True, inputs_unchanged=True)

    def test_counter_postflight_requires_outer_retirement_and_input_gates(self):
        with patch.object(probe, 'audit_counter_artifacts', side_effect=AssertionError('no artifact access')):
            for removed, unchanged in ((False, True), (True, False), (1, True), (True, 1)):
                with self.assertRaisesRegex(ValueError, 'removed unit and unchanged inputs'):
                    probe.audit_counter_postflight(Path('/attempt'), {}, unit_removed=removed,
                                                   inputs_unchanged=unchanged)
            with self.assertRaisesRegex(ValueError, 'complete inner'):
                probe.audit_counter_postflight(Path('/attempt'), dict(counter_access_verified=True),
                                               unit_removed=True, inputs_unchanged=True)

    def test_counter_baseline_binds_reviewed_evidence_bytes(self):
        with patch.object(probe, 'bounded_regular_digest', return_value='changed'), \
                patch.object(probe, 'read', side_effect=AssertionError('no baseline read')):
            with self.assertRaisesRegex(ValueError, 'reviewed initialization evidence changed'):
                probe.verify_initialized_candidate()
        baseline = dict(seals={str(path): 'sealed' for path in (probe.SERVICE, probe.CLIENT, probe.PRELOAD)})
        with patch.object(probe, 'bounded_regular_digest',
                          side_effect=lambda path: probe.INITIALIZED_SEALS[path.name]), \
                patch.object(probe, 'read', side_effect=[baseline, dict(compatibility_passed=True)]), \
                patch.object(probe, 'digest', return_value='sealed'):
            probe.verify_initialized_candidate()


if __name__ == '__main__':
    unittest.main()
