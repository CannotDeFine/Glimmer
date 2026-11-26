"""CPU state-machine and mapped-code rejection tests for the fixed root relay."""

from contextlib import ExitStack
from copy import deepcopy
import errno
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_ncu_relay as relay
import bert_ncu_replay_exec as bridge


def bare_relay():
    result = relay.ReplayRelay.__new__(relay.ReplayRelay)
    result.manifest = dict(uid=1003, gid=1003)
    result.service = bridge.SERVICE
    result.receive, result.reply = Mock(), Mock()
    result.current = dict(serial=1, helper=dict(pid=7, start=70), controller=dict(pid=8, start=80),
                          service=dict(pid=9, start=90), helper_connection=Mock(), controller_connection=Mock(),
                          phase='running', service_fd=123, receipt=None)
    return result


class RelayTest(unittest.TestCase):
    def test_fresh_claim_in_replay_plan_comes_only_from_root_authority(self):
        instance = bare_relay()
        instance.current['phase'] = 'await-launch'
        instance.manifest.update(service_environment={}, bridge_seals={})
        instance.fresh_authority = {'claim_sha256': 'a' * 64}
        instance.deadline, instance.template_sha = 100.0, 'b' * 64
        instance.outer, instance.group = Path('/attempt'), '/task'
        message = dict(op='launch', serial=1, directory='/attempt/passes/pass-0001-test',
                       socket='/private/service.sock', clients={})
        with patch.object(bridge, 'validate_launch', side_effect=ValueError('stop')) as validate:
            with self.assertRaisesRegex(ValueError, 'stop'):
                instance.launch(message, [7])
            self.assertEqual(validate.call_args.args[0]['fresh_claim_sha256'], 'a' * 64)
            with self.assertRaisesRegex(ValueError, 'protocol fields differ'):
                instance.launch(dict(message, fresh_claim_sha256='c' * 64), [7])

    def test_empty_service_command_requires_the_same_terminal_pidfd(self):
        for argv, dead, error in (([], True, ProcessLookupError), ([], False, ProcessLookupError),
                                  (['/wrong', '--fixed'], True, ValueError)):
            instance = bare_relay()
            instance.group = '/task'
            instance.current['plan'] = dict(clients=dict(low=dict(pid=1, start=1), high=dict(pid=2, start=2)),
                                            socket='/private/service.sock')
            with self.subTest(argv=argv, dead=dead), ExitStack() as stack:
                stack.enter_context(patch.object(relay, 'authenticate_process'))
                stack.enter_context(patch.object(relay, 'parent', return_value=7))
                stack.enter_context(patch.object(relay.os, 'getsid', return_value=9))
                stack.enter_context(patch.object(relay.os, 'getpgid', return_value=9))
                stack.enter_context(patch.object(relay, 'command_line', return_value=argv))
                terminal = stack.enter_context(patch.object(bridge, 'terminal', return_value=dead))
                with self.assertRaises(error):
                    instance.authenticate_service()
                terminal.assert_not_called()
                self.assertIsNone(instance.current['receipt'])
                if argv:
                    self.assertEqual(instance.current['command_line_failure']['argv'], argv)

    def test_service_baseline_and_profiler_additions_are_validated_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            library = outer / 'libcuda-injection.so'
            library.write_bytes(b'test injection')
            base = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                        CUDA_VISIBLE_DEVICES='GPU-test', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                        CUDA_CACHE_DISABLE='1', TMPDIR=str(outer / 'ncu-tmp'))
            service = dict(PATH=base['PATH'], LANG='C', LC_ALL='C', CUDA_VISIBLE_DEVICES='GPU-test',
                           LD_LIBRARY_PATH='/service/libraries', PYTHONNOUSERSITE='1', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            instance = bare_relay()
            instance.outer, instance.base = outer, base
            instance.manifest.update(service_environment=service, ncu_supervisor_environment=base,
                                     ncu_injection_seals={str(library): bridge.seal(library)})
            actual = dict(base, **service)
            actual.update(NVTX_INJECTION64_PATH='libcuda-injection.so',
                          NV_COMPUTE_PROFILER_PERFWORKS_DIR=str(outer),
                          NV_COMPUTE_PROFILER_DISABLE_FUNCTION_LAZY_LOADING='1',
                          **{'NVIDIA-PROCESS-TRACKING-CONFIGURATION':
                             'not-launcher-preload-prepend: ' + str(library)})
            original = dict(actual)
            with patch.object(bridge, 'NCU_DIRECTORY', outer), patch.object(bridge, 'protected_system_file'):
                instance.validate_service_environment(actual)
                self.assertEqual(actual, original)
                for key in service:
                    for value in (None, 'changed'):
                        changed = dict(actual)
                        if value is None:
                            del changed[key]
                        else:
                            changed[key] = value
                        with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                            instance.validate_service_environment(changed)
                for key, value in (('UNKNOWN', '1'), ('LD_AUDIT', '/bad'), ('PYTHONPATH', '/bad'),
                                   ('GLIMMER_UNKNOWN', '1'), ('CUDA_DEVICE_ORDER', 'FASTEST_FIRST'),
                                   ('NV_COMPUTE_PROFILER_DISABLE_FUNCTION_LAZY_LOADING', '0')):
                    with self.subTest(key=key), self.assertRaises(ValueError):
                        instance.validate_service_environment(dict(actual, **{key: value}))

    def test_map_disappearance_requires_the_same_terminal_pidfd_and_no_inferred_exit(self):
        for error in (FileNotFoundError(errno.ENOENT, 'gone', '/proc/9/exe'),
                      ProcessLookupError(errno.ESRCH, 'empty maps', '/proc/9/maps'),
                      ProcessLookupError(errno.ESRCH, 'empty command', '/proc/9/cmdline'),
                      ProcessLookupError(errno.ESRCH, 'gone'),
                      relay.NcuMappingUnavailable(['/libcuda-injection.so'], [])):
            instance = bare_relay()
            instance.current.update(initial_maps={'original': 1}, last_maps={'original': 2})
            instance.authenticate_service = Mock(side_effect=error)
            with patch.object(bridge, 'terminal', side_effect=[False, True]) as terminal:
                instance.sample_service_maps()
                self.assertEqual(terminal.call_args_list, [((123,),), ((123,),)])
            self.assertIsNone(instance.current['receipt'])
            self.assertEqual(instance.current['phase'], 'running')
            self.assertEqual(instance.current['last_maps'], {'original': 2})
            self.assertIn('terminal_map_read_race_ns', instance.current)
            failure = instance.current['map_read_failure']
            self.assertIs(failure['terminal_pidfd'], True)
            self.assertEqual(failure['error_type'], type(error).__name__)
            self.assertEqual(failure['monotonic_ns'], instance.current['terminal_map_read_race_ns'])
            self.assertIs(failure['immediate_terminal_pidfd'], True)
            self.assertEqual(failure['wait_timeout_ms'], 0)
            with patch.object(bridge, 'terminal', side_effect=[False, False, True]) as terminal, \
                    patch.object(relay.select, 'poll') as poll:
                poll.return_value.poll.return_value = [(123, relay.select.POLLIN)]
                instance.sample_service_maps()
                self.assertEqual(terminal.call_args_list, [((123,),), ((123,),), ((123,),)])
                poll.return_value.register.assert_called_once_with(123, relay.select.POLLIN)
                poll.return_value.poll.assert_called_once_with(250)
            failure = instance.current['map_read_failure']
            self.assertIs(failure['immediate_terminal_pidfd'], False)
            self.assertIs(failure['terminal_pidfd'], True)
            self.assertEqual(failure['wait_timeout_ms'], 250)
            self.assertEqual(failure['confirmation_elapsed_ns'],
                             failure['confirmation_finished_ns'] - failure['monotonic_ns'])
            self.assertIsNone(instance.current['receipt'])
            self.assertEqual(instance.current['last_maps'], {'original': 2})
            with patch.object(bridge, 'terminal', return_value=False), patch.object(relay.select, 'poll') as poll:
                poll.return_value.poll.return_value = []
                with self.assertRaisesRegex(ValueError, 'terminal pidfd'):
                    instance.sample_service_maps()
                poll.return_value.poll.assert_called_once_with(250)
            self.assertIs(instance.current['map_read_failure']['terminal_pidfd'], False)

        instance.current.pop('initial_maps')
        with patch.object(bridge, 'terminal', side_effect=[False, True]), self.assertRaises(ValueError):
            instance.sample_service_maps()

        for error in (FileNotFoundError(errno.ENOENT, 'gone', '/sealed/library.so'),
                      FileNotFoundError(errno.ENOENT, 'gone', '/proc/99/exe'),
                      FileNotFoundError(errno.ENOENT, 'gone', '/proc/9/../99/exe'),
                      FileNotFoundError(errno.ENOENT, 'gone', '/proc/9/root/library.so'),
                      FileNotFoundError(errno.EACCES, 'denied', '/proc/9/exe'),
                      ProcessLookupError('no explicit errno')):
            instance = bare_relay()
            instance.authenticate_service = Mock(side_effect=error)
            with self.subTest(error=error), patch.object(bridge, 'terminal', return_value=False) as terminal, \
                    patch.object(relay.select, 'poll') as poll, self.assertRaises(type(error)):
                instance.sample_service_maps()
            terminal.assert_called_once_with(123)
            poll.assert_not_called()

        for events in ([(123, relay.select.POLLNVAL)], [(124, relay.select.POLLIN)]):
            instance = bare_relay()
            instance.current.update(initial_maps={'original': 1}, last_maps={'original': 2})
            instance.authenticate_service = Mock(side_effect=FileNotFoundError(errno.ENOENT, 'gone', '/proc/9/exe'))
            with patch.object(bridge, 'terminal', return_value=False), patch.object(relay.select, 'poll') as poll:
                poll.return_value.poll.return_value = events
                with self.assertRaisesRegex(ValueError, 'invalid pidfd'):
                    instance.sample_service_maps()

    def test_terminal_service_skips_maps_but_validation_failures_are_never_suppressed(self):
        instance = bare_relay()
        for reason in ('unsealed mapping', 'duplicate NCU preload', 'NCU injection bytes changed',
                       'service baseline environment differs', 'unexpected or modified non-NCU preload'):
            instance.authenticate_service = Mock(side_effect=ValueError(reason))
            with patch.object(bridge, 'terminal', return_value=True):
                instance.sample_service_maps()
            instance.authenticate_service.assert_not_called()
            with patch.object(bridge, 'terminal', side_effect=[False, True]) as terminal:
                with self.assertRaisesRegex(ValueError, reason):
                    instance.sample_service_maps()
                terminal.assert_called_once_with(123)

    def test_postflight_maps_join_frozen_inputs_and_reject_missing_or_drifted_identity(self):
        service = '/memfd:glimmer-glimmer_execution_service (deleted)'
        inputs = {str(bridge.SERVICE): dict(target=str(bridge.SERVICE), sha256='service'),
                  '/vendor.so': dict(target='/vendor.so', sha256='vendor')}
        mappings = {service: dict(device=1, inode=11, sha256='service'),
                    '/vendor.so': dict(device=2, inode=22, sha256='vendor')}
        good = dict(initial_maps=dict(monotonic_ns=10, mappings=deepcopy(mappings)),
                    last_maps=dict(monotonic_ns=20, mappings=deepcopy(mappings)), observed_mappings=mappings)
        relay.validate_mapping_receipts(good, inputs)
        raced = dict(good, service=dict(pid=9, start=90), terminal_map_read_race_ns=21,
                     map_read_failure=dict(monotonic_ns=21, terminal_pidfd=True, error_type='ProcessLookupError',
                                           errno=errno.ESRCH, filename='/proc/9/maps', immediate_terminal_pidfd=False,
                                           confirmation_finished_ns=30, confirmation_elapsed_ns=9, wait_timeout_ms=250))
        relay.validate_mapping_receipts(raced, inputs)
        for changed in (dict(terminal_pidfd=False), dict(monotonic_ns=19), dict(error_type='ValueError'),
                        dict(filename='/sealed/library.so'), dict(errno=errno.ENOENT), dict(wait_timeout_ms=1000),
                        dict(confirmation_elapsed_ns=10), dict(confirmation_finished_ns=20)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                relay.validate_mapping_receipts(dict(raced, map_read_failure=dict(raced['map_read_failure'], **changed)), inputs)
        changes = [(['last_maps', 'monotonic_ns'], 9), (['initial_maps', 'monotonic_ns'], True),
                   (['last_maps', 'mappings'], {}),
                   (['observed_mappings', service, 'sha256'], 'wrong service'),
                   (['last_maps', 'mappings', '/vendor.so', 'sha256'], 'wrong library'),
                   (['last_maps', 'mappings', '/vendor.so', 'inode'], 23),
                   (['last_maps', 'mappings', '/vendor.so', 'device'], True),
                   (['observed_mappings'], {'/vendor.so': mappings['/vendor.so']})]
        for keys, value in changes:
            changed = deepcopy(good)
            target = changed
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                relay.validate_mapping_receipts(changed, inputs)
        changed = deepcopy(good)
        changed['observed_mappings']['/unknown.so'] = mappings['/vendor.so']
        with self.assertRaisesRegex(ValueError, 'frozen input'):
            relay.validate_mapping_receipts(changed, inputs)
        self.assertFalse(relay.complete_reparse(dict(relay.EXPECTED_REPARSE, full_tensor_recomparison=1)))

    def test_wrapper_settle_requires_known_sealed_recipe_and_final_service_authentication(self):
        for wrapper in (bridge.SETPRIV, bridge.ENV):
            instance = bare_relay()
            instance.group, instance.deadline = '/task', float('inf')
            instance.current['plan'] = dict(uid=1003, gid=1003, seals={str(wrapper): bridge.seal(wrapper)})
            instance.authenticate_service, instance.validate_service_environment, instance.sample = Mock(), Mock(), Mock()
            command = [str(bridge.ENV), '-i', 'KEY=value', '/proc/self/fd/77', '--fixed']
            if wrapper == bridge.SETPRIV:
                command = [*bridge.drop_command(1003, 1003, service=True), *command]
            original_open, original_stat = Path.open, Path.stat
            count = [0]

            def open_image(path, *args, **kwargs):
                return original_open(wrapper if str(path) == '/proc/9/exe' else path, *args, **kwargs)

            def stat_image(path, *args, **kwargs):
                return original_stat(wrapper if str(path) == '/proc/9/exe' else path, *args, **kwargs)

            def destination(path):
                count[0] += 1
                return str(wrapper) if count[0] == 1 else '/memfd:glimmer-glimmer_execution_service (deleted)'

            with ExitStack() as stack:
                stack.enter_context(patch.object(Path, 'open', autospec=True, side_effect=open_image))
                stack.enter_context(patch.object(Path, 'stat', autospec=True, side_effect=stat_image))
                stack.enter_context(patch.object(os, 'readlink', side_effect=destination))
                stack.enter_context(patch.object(bridge, 'terminal', return_value=False))
                stack.enter_context(patch.object(bridge, 'live_identity'))
                stack.enter_context(patch.object(relay, 'parent', return_value=7))
                stack.enter_context(patch.object(os, 'getsid', return_value=9))
                stack.enter_context(patch.object(os, 'getpgid', return_value=9))
                stack.enter_context(patch.object(relay, 'command_line', return_value=command))
                stack.enter_context(patch.object(relay, 'authenticate_process'))
                stack.enter_context(patch.object(bridge, 'service_command', return_value=['service', '--fixed']))
                stack.enter_context(patch.object(relay.time, 'sleep'))
                instance.settle_service_exec()
                instance.authenticate_service.assert_called_once()
                instance.validate_service_environment.assert_called_once_with({'KEY': 'value'})
                instance.reply.assert_not_called()
                count[0] = 0
                command[-1] = '--wrong'
                instance.authenticate_service.reset_mock()
                with self.assertRaisesRegex(ValueError, 'recipe differs'):
                    instance.settle_service_exec()
                instance.authenticate_service.assert_not_called()

    def test_settle_deadline_or_terminal_child_never_releases_clients(self):
        instance = bare_relay()
        instance.group, instance.deadline = '/task', 0
        instance.current['plan'] = {}
        with self.assertRaises(TimeoutError):
            instance.settle_service_exec()
        instance.deadline = float('inf')
        with patch.object(bridge, 'terminal', return_value=True), self.assertRaisesRegex(ValueError, 'before final exec'):
            instance.settle_service_exec()
        instance.reply.assert_not_called()

    def test_service_exit_requires_exact_actual_wait_receipt_and_terminal_pidfd(self):
        receipt = bridge.wait_receipt(9, 90, 0)
        for changed, terminal in ((receipt, True), (dict(receipt, pid=10), True),
                                   (dict(receipt, start=91), True), (dict(receipt, returncode=False), True),
                                   (dict(receipt, exit_code_observed=1), True),
                                   (bridge.wait_receipt(9, 90, int(signal.SIGTERM)), True), (receipt, False)):
            instance = bare_relay()
            instance.receive.return_value = (dict(op='exit', serial=1, child=changed), [])
            with patch.object(bridge, 'terminal', return_value=terminal):
                if changed is receipt and terminal:
                    instance.helper_message()
                    self.assertEqual(instance.current['phase'], 'oracle')
                    self.assertEqual(instance.current['receipt'], receipt)
                else:
                    with self.assertRaises(ValueError):
                        instance.helper_message()
                    self.assertEqual(instance.current['phase'], 'running')
            instance.reply.assert_not_called()

    def test_waiter_retains_terminal_map_read_receipt(self):
        instance = bare_relay()
        instance.sessions = []
        instance.controller = Mock()
        for key in ('helper_ancestors', 'controller_returncode', 'plan', 'initial_maps',
                    'last_maps', 'observed_mappings', 'complete'):
            instance.current[key] = {}
        failure = dict(monotonic_ns=100, terminal_pidfd=True, error_type='NcuMappingUnavailable',
                       expected=['/libcuda-injection.so'], missing=['/libcuda-injection.so'], mapped=[])
        instance.current.update(phase='waiter-exit', helper_fd=124,
                                map_read_failure=failure, terminal_map_read_race_ns=100)
        with patch.object(bridge, 'terminal', return_value=True):
            instance.retire_waiter()
        self.assertEqual(instance.sessions[0]['map_read_failure'], failure)
        self.assertEqual(instance.sessions[0]['terminal_map_read_race_ns'], 100)
        self.assertIsNone(instance.current)

    def test_poll_before_wait_receipt_stays_running_and_abort_never_selects_signal(self):
        instance = bare_relay()
        instance.receive.return_value = (dict(op='poll', serial=1), [])
        instance.controller_message()
        instance.reply.assert_called_once_with('poll', receipt=None)
        instance.receive.return_value = (dict(op='abort', serial=1), [])
        with self.assertRaisesRegex(ValueError, 'contain'):
            instance.controller_message()
        self.assertEqual(instance.current['phase'], 'running')

    def test_complete_is_provisional_until_real_controller_exit_and_exact_oracle(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / 'equal').mkdir()
            artifact = directory / 'equal/server.log'
            artifact.write_text('unique raw audit\n')
            clients = dict(low=dict(pid=10, start=100), high=dict(pid=11, start=110))
            instance = bare_relay()
            instance.controller = Mock()
            instance.controller.poll.return_value = None
            instance.current.update(phase='oracle', plan=dict(directory=str(directory), clients=clients),
                                    client_fds=dict(low=124, high=125), receipt=bridge.wait_receipt(9, 90, 0))
            job = dict(application_identities=clients, server_identity=instance.current['service'],
                       artifact_identities={'server.log': bridge.seal(artifact)})
            report = dict(status='pass_requires_profiler_review', reparse=relay.EXPECTED_REPARSE,
                          controller_identity=instance.current['controller'], serial=1,
                          topology='clean-ncu-sibling', pass_directory=str(directory), job=job)
            path = directory / 'pass.json'
            path.write_text(json.dumps(report))
            completion = dict(op='complete', serial=1, directory=str(directory), accepted=True,
                              report_sha256=bridge.seal(path))
            instance.receive.return_value = (completion, [])
            with patch.object(bridge, 'send_packet') as send, \
                    patch.object(bridge, 'terminal', return_value=True), \
                    patch.object(relay.candidate, 'normal_cleanup') as cleanup:
                instance.controller_message()
                self.assertEqual(instance.current['phase'], 'controller-exit')
                instance.finish_controller()
                send.assert_not_called()
                cleanup.assert_not_called()
                instance.controller.poll.return_value = 1
                with self.assertRaisesRegex(ValueError, 'controller exited'):
                    instance.finish_controller()
                send.assert_not_called()
                instance.controller.poll.return_value = 0
                instance.finish_controller()
                cleanup.assert_called_once_with(job)
                send.assert_called_once_with(instance.current['helper_connection'],
                                             dict(op='finish', serial=1, accepted=True))
                self.assertEqual(instance.current['phase'], 'waiter-exit')

    def test_failed_or_premature_complete_cannot_advance_replay(self):
        for phase, accepted in (('running', True), ('oracle', False), ('oracle', 1)):
            instance = bare_relay()
            instance.current.update(phase=phase, plan=dict(directory='/pass'))
            instance.receive.return_value = (dict(op='complete', serial=1, directory='/pass',
                                                  accepted=accepted, report_sha256='x'), [])
            with self.assertRaises(ValueError):
                instance.controller_message()
            instance.reply.assert_not_called()
            self.assertEqual(instance.current['phase'], phase)

    def test_foreign_packet_closes_transferred_fds(self):
        with tempfile.TemporaryFile() as source:
            descriptor = os.dup(source.fileno())
            instance = relay.ReplayRelay.__new__(relay.ReplayRelay)
            with patch.object(bridge, 'receive_packet', return_value=({}, [descriptor], (7, 1003, 1003))):
                with self.assertRaisesRegex(ValueError, 'sender credentials'):
                    instance.receive(Mock(), (8, 1003, 1003))
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_map_audit_binds_canonical_inode_and_bytes_and_rejects_unknown_code(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as mocks:
            library = Path(temporary) / 'vendor.so'
            library.write_bytes(b'sealed vendor bytes')
            metadata = library.stat()
            known = {str(library): dict(target=str(library), sha256=bridge.seal(library))}
            prefix = ('1000-2000 r-xp 00000000 00:01 11 '
                      '/memfd:glimmer-glimmer_execution_service (deleted)\n')
            line = (f'3000-4000 r-xp 00000000 {os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x} '
                    f'{metadata.st_ino} {library}\n')
            original_stat, original_read = Path.stat, Path.read_text
            text = [prefix + line]

            def fake_stat(path, *args, **kwargs):
                if str(path) == '/proc/9/exe':
                    return SimpleNamespace(st_dev=1, st_ino=11)
                return original_stat(path, *args, **kwargs)

            def fake_read(path, *args, **kwargs):
                return text[0] if str(path) == '/proc/9/maps' else original_read(path, *args, **kwargs)

            mocks.enter_context(patch.object(Path, 'stat', autospec=True, side_effect=fake_stat))
            mocks.enter_context(patch.object(Path, 'read_text', autospec=True, side_effect=fake_read))
            audit = relay.MappedCodeAudit(known)
            result = audit.observe(9, 'service digest')
            self.assertEqual(result['mappings'][str(library)]['sha256'], known[str(library)]['sha256'])
            for changed in (prefix + line.replace(str(metadata.st_ino), str(metadata.st_ino + 1)),
                            prefix + '3000-4000 rwxp 00000000 00:00 0\n',
                            prefix + line.rstrip() + ' (deleted)\n', line):
                text[0] = changed
                with self.subTest(mapping=changed), self.assertRaises(ValueError):
                    audit.observe(9, 'service digest')
            text[0] = prefix + line
            with self.assertRaisesRegex(ValueError, 'unsealed'):
                relay.MappedCodeAudit({}).observe(9, 'service digest')
            library.write_bytes(b'changed vendor bytes')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                audit.observe(9, 'service digest')

    def test_map_aliases_cannot_disagree_about_input_bytes(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            relay.MappedCodeAudit({'a': dict(target='/same.so', sha256='a'),
                                  'b': dict(target='/same.so', sha256='b')})


if __name__ == '__main__':
    unittest.main()
