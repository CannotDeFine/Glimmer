"""CPU-only service relay adapter tests; no sudo, CUDA or BERT launch."""

from contextlib import ExitStack
from copy import deepcopy
import io
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_ncu_replay_exec as bridge
import bert_ncu_service_proxy as proxy


class ServiceProxyTest(unittest.TestCase):
    def setUp(self):
        # Command construction resolves the range library strictly. Use a
        # hermetic stand-in, as launch fixtures do, instead of the host CUDA.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        provider = Path(temporary.name) / 'range.so'
        provider.touch()
        patcher = patch.object(bridge, 'RANGE_LIBRARY', provider)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ordinary_logical_release_requires_exact_issued_then_completed_chain(self):
        prefix = 'authenticated low binding\n'
        admission = 'Native admission audit: session=2, sequence=2842, accepted=true, operation=42\n'
        issued = ('Priority execution audit: event=low_issued, session=2, priority=0, sequence=2842, '
                  'completed_blocks=0, total_blocks=2304, physical_launches=1, status=0\n')
        done = ('Priority execution audit: event=low_logical_done, session=2, priority=0, sequence=2842, '
                'completed_blocks=2304, total_blocks=2304, physical_launches=1, status=0\n')
        parser, admissions, position = proxy.LogicalRelease(2, 7, 70, 2842), {}, len(prefix)
        for line in (admission, issued):
            self.assertIsNone(parser(line, position, 7, 70, admissions, 2842))
            position += len(line)
        receipt = parser(done, position, 7, 70, admissions, 2842)
        self.assertEqual((receipt['kind'], receipt['physical_completion_observed'],
                          receipt['pre_demand_compute_observed']), ('logical_done', False, False))
        job = dict(mode='equal', priority_audit_start=len(prefix),
                   application_identities=dict(low=dict(pid=7, start=70)), high_release_trigger=receipt)
        with patch.object(proxy, 'ordinary_release_session', return_value=2):
            self.assertEqual(proxy.validate_ordinary_release(prefix + admission + issued + done,
                                                             job, dict(session=2)), receipt)
            for altered in (prefix + admission + done, prefix + admission + issued + done + done,
                            prefix + admission + issued + issued + done,
                            prefix + admission + issued + issued.replace('low_issued', 'nonpreemptible_wait') + done,
                            prefix + admission + issued + done.replace('status=0', 'status=oops'),
                            prefix + admission.replace('operation=42', 'operation=30') + issued + done,
                            prefix + admission + issued + done.replace('completed_blocks=2304',
                                                                       'completed_blocks=1'),
                            prefix + admission + issued + done.replace('status=0', 'status=1')):
                with self.subTest(content=altered), self.assertRaises(ValueError):
                    proxy.validate_ordinary_release(altered, job, dict(session=2))
            with self.assertRaisesRegex(ValueError, 'owners differ'):
                proxy.validate_ordinary_release(prefix + admission + issued + done, job, dict(session=3))

    def test_ordinary_release_hook_is_scoped_and_restored_after_exception(self):
        original_parser = Mock()
        common = SimpleNamespace(subprocess=Mock(), verify_process=Mock(), read_ready=Mock(),
                                 check_pending=Mock(), _arrival_record=original_parser)
        helpers = SimpleNamespace(stop_owned=Mock())

        def release(*args, **kwargs):
            self.assertIsInstance(common._arrival_record, proxy.LogicalRelease)
            raise RuntimeError('injected release failure')

        common.release_on_publication = release
        ordinary = dict(schema='fresh_runtime_v3', variant='scalar',
                        contract=dict(schema='fresh_runtime_sgemm_v3', specialization='ordinary'))
        observed = dict(schema='fresh_runtime_v2', variant='scalar')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'log'
            path.write_text('prefix\n')
            with proxy.service_launches(common, helpers, Mock(), Path(temporary), {}, signal_binding=observed):
                self.assertIs(common.release_on_publication, release)
                self.assertIs(common._arrival_record, original_parser)
            with proxy.service_launches(common, helpers, Mock(), Path(temporary), {}, signal_binding=ordinary), \
                    patch.object(proxy, 'ordinary_release_session', return_value=2):
                with self.assertRaisesRegex(RuntimeError, 'injected release failure'):
                    common.release_on_publication(path, 7, 7, 70, selected_sequence=2842)
                self.assertIs(common._arrival_record, original_parser)
            self.assertIs(common.release_on_publication, release)
            self.assertIs(common._arrival_record, original_parser)

    def test_deferred_launch_publishes_once_and_never_on_close(self):
        message = json.dumps(dict(command=['app'], environment={})) + '\n'
        stream = io.StringIO()
        deferred = proxy.DeferredLaunch(stream)
        deferred.write(message)
        deferred.flush()
        self.assertEqual(stream.getvalue(), '')
        self.assertEqual(deferred.state, 'never_released')
        deferred.release()
        self.assertEqual(stream.getvalue(), message)
        self.assertEqual(deferred.state, 'released_once')
        with self.assertRaises(ValueError):
            deferred.release()
        other = Mock()
        pending = proxy.DeferredLaunch(other)
        pending.write(message)
        pending.flush()
        pending.close()
        other.write.assert_not_called()
        other.close.assert_called_once()

    def test_deferred_launch_is_fail_closed_for_bad_messages_and_uncertain_writes(self):
        message = json.dumps(dict(command=['app'], environment={})) + '\n'
        for invalid in ('fragment', message + message, '{}\n'):
            with self.subTest(message=invalid), self.assertRaises((ValueError, json.JSONDecodeError)):
                proxy.DeferredLaunch(io.StringIO()).write(invalid)
        for failure in ('write', 'partial', 'flush'):
            stream = Mock()
            stream.write.return_value = len(message) if failure != 'partial' else 1
            if failure == 'write':
                stream.write.side_effect = OSError('write failed')
            if failure == 'flush':
                stream.flush.side_effect = OSError('flush failed')
            deferred = proxy.DeferredLaunch(stream)
            deferred.write(message)
            deferred.flush()
            with self.subTest(failure=failure), self.assertRaises((OSError, ValueError)):
                deferred.release()
            self.assertEqual(deferred.state, 'release_uncertain')
            with self.assertRaises(ValueError):
                deferred.release()
            stream.write.assert_called_once()

    def test_fresh_order_waits_for_authenticated_low_and_restores_hooks(self):
        environment = dict(os.environ)
        wrapper = 'import sys; sys.stdin.readline()'
        helpers = SimpleNamespace(stop_owned=Mock(), wrapper=wrapper)
        original_ready = Mock(return_value=None)
        common = SimpleNamespace(subprocess=subprocess, verify_process=Mock(), read_ready=original_ready,
                                 check_pending=Mock())
        processes = []
        try:
            with proxy.service_launches(common, helpers, Mock(), Path('/attempt'), environment,
                                        signal_binding=dict(schema='fresh_runtime_v2', variant='scalar')) as ordering:
                for _ in range(2):
                    processes.append(common.subprocess.Popen([sys.executable, '-B', '-c', wrapper],
                                     stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, text=True, env=environment,
                                     start_new_session=True))
                low, high = processes
                low_start = bridge.process_identity(low.pid)['start']
                message = json.dumps(dict(command=['app'], environment={})) + '\n'
                high.stdin.write(message)
                high.stdin.flush()
                self.assertIsNone(common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                                     ['app'], {}, time.monotonic_ns()))
                self.assertIsNone(high.poll())
                self.assertEqual(ordering, {})
                with patch.object(proxy.time, 'monotonic_ns', return_value=high.stdin.queued_ns + 100 * 10**9):
                    with self.assertRaisesRegex(ValueError, 'timed out'):
                        common.check_pending({'low': low, 'high': high}, None, {}, {}, None)
                self.assertIsNone(high.poll())
                original_ready.side_effect = RuntimeError('low readiness failed')
                with self.assertRaisesRegex(RuntimeError, 'low readiness failed'):
                    common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                      ['app'], {}, time.monotonic_ns())
                self.assertIsNone(high.poll())
                self.assertEqual(ordering, {})
                original_ready.side_effect = None
                original_ready.return_value = {'authenticated_ns': time.monotonic_ns()}
                identity = bridge.process_identity(high.pid)
                with patch.object(proxy.bridge, 'process_identity', return_value=dict(identity, start=identity['start'] + 1)):
                    with self.assertRaisesRegex(ValueError, 'no longer waits'):
                        common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                          ['app'], {}, time.monotonic_ns())
                with patch.object(proxy, 'proc_environment', return_value={}):
                    with self.assertRaisesRegex(ValueError, 'environment changed'):
                        common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                          ['app'], {}, time.monotonic_ns())
                with patch.object(proxy.os, 'readlink', return_value='socket:[123]'):
                    with self.assertRaisesRegex(ValueError, 'owns a socket'):
                        common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                          ['app'], {}, time.monotonic_ns())
                original_read_text = Path.read_text
                children = Path(f'/proc/{high.pid}/task/{high.pid}/children')
                def fake_read_text(path, *args, **kwargs):
                    return '123' if path == children else original_read_text(path, *args, **kwargs)
                with patch.object(Path, 'read_text', fake_read_text):
                    with self.assertRaisesRegex(ValueError, 'owns a child'):
                        common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                          ['app'], {}, time.monotonic_ns())
                self.assertIsNone(high.poll())
                self.assertEqual(ordering, {})
                common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                  ['app'], {}, time.monotonic_ns())
                self.assertEqual(ordering['state'], 'released_once')
                self.assertEqual(high.wait(timeout=5), 0)
                with self.assertRaises(ValueError):
                    common.read_ready(helpers, Path('/attempt'), 'low', low, low_start,
                                      ['app'], {}, time.monotonic_ns())
            self.assertIs(common.read_ready, original_ready)
            self.assertIs(common.subprocess, subprocess)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
                process.stdin.close()

    def test_order_receipt_rejects_changed_message_or_start_gate(self):
        clients = dict(low=dict(pid=10, start=20), high=dict(pid=11, start=21))
        command, environment = ['app'], {'LANG': 'C'}
        message = (json.dumps(dict(command=command, environment=environment)) + '\n').encode()
        job = dict(application_identities=clients, application_commands={'high': command},
                   application_environments={'high': environment}, application_launch_ns={'high': 1},
                   ready=dict(low=dict(authenticated_ns=3, released_ns=7),
                              high=dict(ready_ns=6, authenticated_ns=6, released_ns=8)))
        receipt = dict(low=clients['low'], high=clients['high'], low_authenticated_ns=3, high_queued_ns=2,
                       high_verified_ns=4, high_release_ns=5, message_bytes=len(message),
                       message_sha256=hashlib.sha256(message).hexdigest(), state='released_once',
                       no_children=True, no_sockets=True)
        proxy.validate_launch_order(job, receipt)
        for changed in (dict(receipt, high_release_ns=2), dict(receipt, message_sha256='0' * 64),
                        dict(receipt, state='release_uncertain'), dict(receipt, no_children=False)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                proxy.validate_launch_order(job, changed)

    def make_process(self):
        relay = Mock(serial=1, deadline_ns=time.monotonic_ns() + 10 ** 9)
        return relay, proxy.ServiceProcessProxy(relay, dict(pid=7, start=70), ['service'], {'PATH': '/usr/bin:/bin'})

    def test_poll_requires_actual_raw_wait_status_and_caches_it(self):
        for raw in (0, 17 << 8, int(signal.SIGTERM)):
            relay, process = self.make_process()
            receipt = bridge.wait_receipt(7, 70, raw)
            relay.request.side_effect = [dict(op='poll', serial=1, receipt=None),
                                         dict(op='poll', serial=1, receipt=receipt)]
            self.assertIsNone(process.poll())
            self.assertIsNone(process.returncode)
            self.assertEqual(process.poll(), receipt['returncode'])
            self.assertEqual(process.poll(), receipt['returncode'])
            self.assertEqual(relay.request.call_count, 2)

    def test_pidfd_or_profiler_exit_never_supplies_success(self):
        changed = [dict(pid=7, start=70, raw_status=None, returncode=0, exit_code_observed=True),
                   dict(bridge.wait_receipt(7, 70, 0), pid=8),
                   dict(bridge.wait_receipt(7, 70, 0), start=71),
                   dict(bridge.wait_receipt(7, 70, 0), returncode=False),
                   dict(bridge.wait_receipt(7, 70, 0), exit_code_observed=1),
                   dict(bridge.wait_receipt(7, 70, 17 << 8), returncode=0),
                   dict(bridge.wait_receipt(7, 70, 0), pidfd_terminal=True)]
        for receipt in changed:
            relay, process = self.make_process()
            relay.request.return_value = dict(op='poll', serial=1, receipt=receipt)
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                process.poll()
            self.assertIsNone(process.returncode)
        relay, process = self.make_process()
        relay.request.return_value = dict(op='poll', serial=1, ncu_returncode=0)
        with self.assertRaises(ValueError):
            process.poll()

    def test_wait_timeout_is_not_an_exit_receipt(self):
        relay, process = self.make_process()
        relay.request.return_value = dict(op='poll', serial=1, receipt=None)
        with self.assertRaises(subprocess.TimeoutExpired):
            process.wait(timeout=0)
        self.assertIsNone(process.receipt)

    def test_normal_cleanup_does_not_request_a_signal(self):
        relay, process = self.make_process()
        relay.request.return_value = dict(op='poll', serial=1, receipt=bridge.wait_receipt(7, 70, 0))
        process.stop(70)
        relay.request.assert_called_once_with('poll')

    def test_live_cleanup_aborts_owned_replay_without_arbitrary_signal_or_pid(self):
        relay, process = self.make_process()
        relay.request.side_effect = [dict(op='poll', serial=1, receipt=None), dict(op='abort', serial=1)]
        with self.assertRaisesRegex(RuntimeError, 'invalidates'):
            process.stop(70)
        self.assertEqual([entry.args for entry in relay.request.call_args_list], [('poll',), ('abort',)])
        self.assertIsNone(process.returncode)

    def test_verification_binds_original_identity_recipe_and_root_receipt(self):
        relay, process = self.make_process()
        relay.request.return_value = dict(op='verify', serial=1, service=dict(pid=7, start=70), verified=True)
        process.verify(70, ['service'], {'PATH': '/usr/bin:/bin'})
        for start, command, environment in ((71, ['service'], process.environment),
                                            (70, ['other'], process.environment),
                                            (70, ['service'], {'PATH': '/bad'})):
            with self.assertRaises(ValueError):
                process.verify(start, command, environment)
        relay.request.return_value['service']['start'] = 71
        with self.assertRaises(ValueError):
            process.verify(70, ['service'], process.environment)

    def test_service_request_sends_only_fixed_dynamic_identities(self):
        request = dict(directory='/attempt/passes/pass-0001', socket='/attempt/ncu-tmp/glimmer-bert-co-x/service.sock',
                       clients=dict(low=dict(pid=7, start=70), high=dict(pid=8, start=80)))
        command = bridge.service_command(request)
        self.assertEqual(proxy.service_request(command, Path(request['directory'])), request)
        for changed in (['/bin/sh'], [*command, '--unexpected'],
                        [value.replace('7:70:0:configured', '7:70:100:configured') for value in command],
                        [value.replace('7:70:3127', '7:70:3128') for value in command],
                        [value.replace('7:70:3127', '8:80:3127') for value in command],
                        [value.replace('--audit-sgemm-target-client', '--audit-sgemm-target') for value in command],
                        [value.replace('320', '80') for value in command]):
            with self.subTest(command=changed), self.assertRaises(ValueError):
                proxy.service_request(changed, Path(request['directory']))

    def test_scoped_adapter_preserves_client_launch_and_restores_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as resources:
            directory = Path(temporary)
            (directory / 'equal').mkdir()
            log = resources.enter_context((directory / 'equal/server.log').open('x'))
            original_subprocess, original_verify, original_stop = Mock(), Mock(), Mock()
            common = SimpleNamespace(ROOT=directory, subprocess=original_subprocess, verify_process=original_verify)
            helpers = SimpleNamespace(stop_owned=original_stop)
            relay = Mock(serial=1, deadline_ns=time.monotonic_ns() + 10 ** 9)
            relay.request.return_value = dict(op='launch', serial=1, service=dict(pid=9, start=90))
            command = bridge.service_command(dict(socket='/tmp/glimmer-bert-co-x/service.sock',
                                  clients=dict(low=dict(pid=7, start=70), high=dict(pid=8, start=80))))
            environment = dict(PATH='/usr/bin:/bin')
            launch = dict(cwd=directory, env=environment, stdout=log,
                          stderr=subprocess.STDOUT, start_new_session=True)
            with self.assertRaisesRegex(RuntimeError, 'test restoration'):
                with proxy.service_launches(common, helpers, relay, directory, environment):
                    client = common.subprocess.Popen(['client'], stdin=subprocess.PIPE)
                    self.assertIs(client, original_subprocess.Popen.return_value)
                    original_subprocess.Popen.assert_called_once_with(['client'], stdin=subprocess.PIPE)
                    service = common.subprocess.Popen(command, **launch)
                    self.assertIsInstance(service, proxy.ServiceProcessProxy)
                    self.assertEqual((service.pid, service.start), (9, 90))
                    self.assertEqual(relay.request.call_args.kwargs['descriptors'], (log.fileno(),))
                    with self.assertRaisesRegex(ValueError, 'reused'):
                        common.subprocess.Popen(command, **launch)
                    helpers.stop_owned(client, 70)
                    original_stop.assert_called_once_with(client, 70)
                    raise RuntimeError('test restoration')
            self.assertIs(common.subprocess, original_subprocess)
            self.assertIs(common.verify_process, original_verify)
            self.assertIs(helpers.stop_owned, original_stop)

    def test_failed_helper_reason_is_saved_before_mandatory_abort(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as resources:
            directory = Path(temporary)
            (directory / 'equal').mkdir()
            log = resources.enter_context((directory / 'equal/server.log').open('x'))
            helpers = SimpleNamespace(stop_owned=Mock())
            common = SimpleNamespace(ROOT=directory, subprocess=Mock(), verify_process=Mock())
            relay = Mock(serial=1, deadline_ns=time.monotonic_ns() + 10 ** 9)
            relay.request.side_effect = [dict(op='launch', serial=1, service=dict(pid=9, start=90)),
                                         dict(op='poll', serial=1, receipt=None), dict(op='abort', serial=1)]
            command = bridge.service_command(dict(socket='/tmp/glimmer-bert-co-x/service.sock',
                                  clients=dict(low=dict(pid=7, start=70), high=dict(pid=8, start=80))))
            environment = dict(PATH='/usr/bin:/bin')

            def run_control(service):
                record = dict(status='failed', error='ValueError: diagnostic reason')
                helpers.stop_owned(service, 90)
                return record

            common.run_control = run_control
            stderr = io.StringIO()
            with proxy.service_launches(common, helpers, relay, directory, environment), \
                    patch.object(proxy.sys, 'stderr', stderr):
                service = common.subprocess.Popen(command, cwd=directory, env=environment, stdout=log,
                                                  stderr=subprocess.STDOUT, start_new_session=True)
                with self.assertRaisesRegex(RuntimeError, 'invalidates'):
                    common.run_control(service)
            self.assertEqual(json.loads(stderr.getvalue()), {'application_error': 'ValueError: diagnostic reason'})
            self.assertEqual([call.args[0] for call in relay.request.call_args_list], ['launch', 'poll', 'abort'])

    def test_client_injection_checks_precede_original_verifier(self):
        original = Mock()
        common = SimpleNamespace(subprocess=Mock(), verify_process=original)
        helpers = SimpleNamespace(stop_owned=Mock())
        client = Mock(pid=7)
        with proxy.service_launches(common, helpers, Mock(), Path('/attempt'), {}), \
                patch.object(proxy, 'proc_mapped_paths', return_value={}), \
                patch.object(proxy, 'proc_environment', return_value={}) as environment:
            common.verify_process(helpers, client, 70, ['client'], {})
            original.assert_called_once_with(helpers, client, 70, ['client'], {})
            environment.return_value = {b'NVTX_INJECTION64_PATH': b'libcuda-injection.so'}
            with self.assertRaisesRegex(ValueError, 'NCU environment'):
                common.verify_process(helpers, client, 70)
            self.assertEqual(original.call_count, 1)

    def test_frontend_ready_mappings_require_exact_clients_environment_and_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'libglimmer_execution_interceptor.so'
            path.write_bytes(b'frontend fixture')
            info = path.stat()
            product = dict(path=str(path), device=info.st_dev, inode=info.st_ino, size=info.st_size,
                           mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns, mode=info.st_mode & 0o7777,
                           uid=info.st_uid, gid=info.st_gid, sha256=bridge.seal(path))
            base = {'PATH': '/bin'}
            environment = dict(base, LD_PRELOAD=str(path), GLIMMER_EXECUTION_HOST_RANGE_SEEK='0',
                               GLIMMER_EXECUTION_SOCKET='/private/socket', GLIMMER_EXECUTION_SERVER_PID='42',
                               GLIMMER_EXECUTION_SERVER_START='420', GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='240000')
            actual = {os.fsencode(key): os.fsencode(value) for key, value in environment.items()}
            device = f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}'
            rows = [f'{index:08x}-{index + 1:08x} {permission} 0 {device} {info.st_ino} {path}'
                    for index, permission in enumerate(('r--p', 'r-xp', 'r--p', 'r--p', 'rw-p'), 1)]
            clients = {'low': dict(pid=7, start=70), 'high': dict(pid=8, start=80)}
            job = dict(application_identities=clients, application_environments=dict(low=environment, high=environment),
                       application_environment=environment, server_environment=base,
                       server_identity=dict(pid=42, start=420), socket_path='/private/socket')
            observer = proxy.FrontendMappings(product, base)
            with patch.object(proxy, 'proc_environment', return_value=actual), \
                    patch.object(Path, 'read_text', return_value='\n'.join(rows)), \
                    patch.object(bridge, 'process_identity', side_effect=lambda pid: dict(pid=pid, start=pid * 10)):
                for role, identity in clients.items():
                    observer.observe(SimpleNamespace(pid=identity['pid']), identity['start'], environment)
                    observer.observe(SimpleNamespace(pid=identity['pid']), identity['start'], environment)
                    if role == 'low':
                        with self.assertRaises(ValueError):
                            observer.for_job(job)
                receipts = observer.for_job(job)
                self.assertEqual(set(receipts), {'low', 'high'})
                for field, value in (('pid', 9), ('device', 0), ('maps', 4), ('seek', '1'),
                                     ('permissions', ['rwxp'] * 5), ('environment', {})):
                    changed = deepcopy(receipts)
                    changed['low'][field] = value
                    with self.subTest(receipt=field), self.assertRaises(ValueError):
                        proxy.validate_frontend_receipts(product, base, job, changed)
                altered_job, altered_receipts = deepcopy(job), deepcopy(receipts)
                altered_env = dict(environment, PATH='/other')
                altered_job['application_environment'] = altered_env
                altered_job['application_environments'] = dict(low=altered_env, high=altered_env)
                for item in altered_receipts.values():
                    item['environment'] = altered_env
                with self.assertRaises(ValueError):
                    proxy.validate_frontend_receipts(product, base, altered_job, altered_receipts)
                with self.assertRaises(ValueError):
                    observer.for_job({'application_identities': {'low': clients['low'], 'high': clients['low']}})
                with self.assertRaises(ValueError):
                    observer.observe(SimpleNamespace(pid=9), 90, dict(environment, PATH='/other'))
                observer.observe(SimpleNamespace(pid=9), 90, environment)
                with self.assertRaises(ValueError):
                    observer.for_job(job)

            variants = [rows[:-1], rows + [rows[-1]],
                        [rows[0] + ' (deleted)', *rows[1:]],
                        [rows[0].replace(str(path), str(path.parent / 'alias.so')), *rows[1:]],
                        [rows[0].replace('r--p', 'rwxp'), *rows[1:]],
                        [rows[0].replace(f' {info.st_ino} ', ' 1 '), *rows[1:]],
                        [rows[0].replace(device, 'ff:ff'), *rows[1:]]]
            for changed in variants:
                with self.subTest(mapping=changed), patch.object(proxy, 'proc_environment', return_value=actual), \
                        patch.object(Path, 'read_text', return_value='\n'.join(changed)), \
                        patch.object(bridge, 'process_identity', return_value=clients['low']):
                    with self.assertRaises(ValueError):
                        proxy.FrontendMappings(product, base).observe(SimpleNamespace(pid=7), 70, environment)
            with patch.object(proxy, 'proc_environment', return_value=actual), \
                    patch.object(Path, 'read_text', return_value='\n'.join(rows)), \
                    patch.object(bridge, 'process_identity', side_effect=[clients['low'], dict(pid=7, start=71)]):
                with self.assertRaises(ValueError):
                    proxy.FrontendMappings(product, base).observe(SimpleNamespace(pid=7), 70, environment)
            for key, value in (('GLIMMER_EXECUTION_HOST_RANGE_SEEK', '1'), ('LD_AUDIT', '/audit.so'),
                               ('NVTX_INJECTION64_PATH', '/inject.so'), ('LD_PRELOAD', '/other.so')):
                changed = dict(environment, **{key: value})
                actual_changed = {os.fsencode(name): os.fsencode(item) for name, item in changed.items()}
                with self.subTest(environment=key), patch.object(proxy, 'proc_environment', return_value=actual_changed), \
                        patch.object(bridge, 'process_identity', return_value=clients['low']):
                    with self.assertRaises(ValueError):
                        proxy.FrontendMappings(product, base).observe(SimpleNamespace(pid=7), 70, changed)
            with patch.object(proxy, 'proc_environment', side_effect=FileNotFoundError('process vanished')), \
                    patch.object(bridge, 'process_identity', return_value=clients['low']):
                with self.assertRaises(FileNotFoundError):
                    proxy.FrontendMappings(product, base).observe(SimpleNamespace(pid=7), 70, environment)

    def test_frontend_observer_runs_only_after_ready_authentication(self):
        original, observer = Mock(), Mock()
        common = SimpleNamespace(subprocess=Mock(), verify_process=original)
        helpers, client = SimpleNamespace(stop_owned=Mock()), Mock(pid=7)
        with proxy.service_launches(common, helpers, Mock(), Path('/attempt'), {}, frontend_observer=observer), \
                patch.object(proxy, 'proc_mapped_paths', return_value={}), \
                patch.object(proxy, 'proc_environment', return_value={}):
            common.verify_process(helpers, client, 70)
            observer.observe.assert_not_called()
            common.verify_process(helpers, client, 70, ['client'], {'PATH': '/bin'})
            observer.observe.assert_called_once_with(client, 70, {'PATH': '/bin'})
            self.assertEqual(original.call_count, 3)
            observer.observe.side_effect = ValueError('changed on second ready verification')
            with self.assertRaisesRegex(ValueError, 'second ready verification'):
                common.verify_process(helpers, client, 70, ['client'], {'PATH': '/bin'})
            self.assertEqual(original.call_count, 4)


if __name__ == '__main__':
    unittest.main()
