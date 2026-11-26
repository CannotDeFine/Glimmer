"""CPU process and socket regression tests; no CUDA, sudo or systemd."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from bert_ncu_client_broker import (BrokerConnection, ClientBroker, ReapedChild,
                                    broker_client_launches, clean_client_environment)
from bert_sgemm_ncu_process import authenticated_profile_processes


WRAPPER = ('import json,os,sys; x=json.loads(sys.stdin.buffer.readline()); '
           'os.execve(x["command"][0],x["command"],x["environment"])')
ENVIRONMENT = dict(PATH='/usr/bin:/bin', LANG='C')


@contextmanager
def running_broker():
    with tempfile.TemporaryDirectory(prefix='gb-') as tmp:
        outer = Path(tmp)
        (outer / 'passes').mkdir()
        read_fd, write_fd = os.pipe()
        broker = ClientBroker(outer, [sys.executable, '-B', '-c', WRAPPER], ENVIRONMENT,
                              outer, [os.fsdecode(value) for value in
                              Path('/proc/self/cmdline').read_bytes().split(b'\0')[:-1]],
                              time.monotonic() + 20)
        published, errors, ready = threading.Event(), [], {}

        def publish(value):
            ready.update(value)
            published.set()

        def serve():
            try:
                broker.serve(read_fd, publish)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            if not published.wait(3):
                raise RuntimeError(f'broker did not start: {errors}')
            yield outer, ready, broker, errors
        finally:
            os.close(write_fd)
            thread.join(timeout=8)
            os.close(read_fd)
            if thread.is_alive():
                raise RuntimeError('broker thread did not retire')
            if (outer / 'broker.sock').exists():
                raise RuntimeError('broker socket did not retire')


def replay(outer, ready, name):
    directory = outer / 'passes' / ('pass-' + name)
    (directory / 'equal').mkdir(parents=True)
    return directory, BrokerConnection(outer, directory, ready)


def launch(connection, directory, role, code):
    with (directory / 'equal' / (role + '.log')).open('x') as log:
        process = connection.spawn(role, log)
    process.stdin.write(json.dumps(dict(command=[sys.executable, '-B', '-c', code],
                                        environment=ENVIRONMENT)) + '\n')
    process.stdin.flush()
    return process


class BrokerTests(unittest.TestCase):
    def test_real_exit_statuses_and_fresh_replay(self):
        with running_broker() as (outer, ready, server, errors):
            observed = []
            for name, codes in [('first', [0, 17]), ('second', [3, -signal.SIGTERM])]:
                directory, connection = replay(outer, ready, name)
                try:
                    processes = [launch(connection, directory, role,
                                 'import os,signal; ' + (f'os._exit({code})' if code >= 0 else
                                                       f'os.kill(os.getpid(), {-code})'))
                                 for role, code in zip(('low', 'high'), codes)]
                    self.assertEqual([process.wait(3) for process in processes], codes)
                    receipt = connection.finish()
                    self.assertEqual([receipt[role]['returncode'] for role in ('low', 'high')], codes)
                    observed += [(p.pid, p.start) for p in processes]
                finally:
                    connection.close()
            self.assertEqual(len(set(observed)), 4)
        self.assertFalse(errors)
        self.assertEqual(len(server.sessions), 2)
        self.assertTrue(all(session['status'] == 'complete' for session in server.sessions))

    def test_controller_disconnect_reaps_both_live_children_and_fails(self):
        with running_broker() as (outer, ready, server, errors):
            directory, connection = replay(outer, ready, 'disconnect')
            processes = [launch(connection, directory, role, 'import time; time.sleep(15)')
                         for role in ('low', 'high')]
            connection.close()
            until = time.monotonic() + 5
            while not server.sessions and time.monotonic() < until:
                time.sleep(.01)
        self.assertTrue(errors)
        self.assertEqual(server.sessions[0]['status'], 'failed')
        self.assertEqual(set(server.sessions[0]['retired_clients']), {'low', 'high'})
        for process in processes:
            self.assertFalse(Path(f'/proc/{process.pid}').exists())

    def test_partial_second_start_failure_reaps_first_child(self):
        with running_broker() as (outer, ready, server, errors):
            directory, connection = replay(outer, ready, 'partial')
            process = launch(connection, directory, 'low', 'import time; time.sleep(15)')
            try:
                with (directory / 'equal/high.log').open('x') as log:
                    log.write('not fresh')
                    log.flush()
                    with self.assertRaises((EOFError, ConnectionError)):
                        connection.spawn('high', log)
            finally:
                connection.close()
        self.assertTrue(errors)
        self.assertFalse(Path(f'/proc/{process.pid}').exists())
        self.assertEqual(set(server.sessions[0]['retired_clients']), {'low'})

    def test_transport_loss_cannot_return_success_and_wait_times_out(self):
        with running_broker() as (outer, ready, _, errors):
            directory, connection = replay(outer, ready, 'timeout')
            process = launch(connection, directory, 'low', 'import time; time.sleep(15)')
            with self.assertRaises(subprocess.TimeoutExpired):
                process.wait(.02)
            connection.close()
            with self.assertRaises(OSError):
                process.poll()
        self.assertTrue(errors)

    def test_missing_wait_status_is_not_success(self):
        with subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(15)'],
                              start_new_session=True) as process:
            child = ReapedChild(process)
            try:
                with patch('bert_ncu_client_broker.os.waitpid', side_effect=ChildProcessError):
                    with self.assertRaises(ChildProcessError):
                        child.poll()
                self.assertIsNone(child.returncode)
            finally:
                child.close()

    def test_reused_directory_rejected(self):
        with running_broker() as (outer, ready, server, errors):
            directory, connection = replay(outer, ready, 'reused')
            try:
                for role in ('low', 'high'):
                    launch(connection, directory, role, 'pass').wait(3)
                connection.finish()
            finally:
                connection.close()
            with self.assertRaises(EOFError):
                BrokerConnection(outer, directory, ready)
        self.assertTrue(errors)
        self.assertEqual(server.sessions[0]['status'], 'complete')

    def test_scoped_adapter_keeps_global_popen_and_restores_after_failure(self):
        original = subprocess.Popen
        original_env = lambda *a: {'NVTX_INJECTION64_PATH': 'libcuda-injection.so', 'LD_PRELOAD': 'owned'}
        common = SimpleNamespace(subprocess=subprocess, owned_environment=original_env,
                                 ROOT=Path.cwd())
        helpers = SimpleNamespace(training=SimpleNamespace(PYTHON=sys.executable), wrapper=WRAPPER)
        fake = SimpleNamespace(spawn=lambda role, log: (role, log))
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with broker_client_launches(common, helpers, fake, ENVIRONMENT):
                self.assertIs(subprocess.Popen, original)
                self.assertIsNot(common.subprocess, subprocess)
                self.assertIs(common.subprocess.check_output, subprocess.check_output)
                self.assertEqual(common.owned_environment(), {'LD_PRELOAD': 'owned'})
                with tempfile.NamedTemporaryFile() as log:
                    with self.assertRaises(ValueError):
                        common.subprocess.Popen([sys.executable, '-B', '-c', WRAPPER], stdout=log)
                raise RuntimeError('injected')
        self.assertIs(common.subprocess, subprocess)
        self.assertIs(common.owned_environment, original_env)

    def test_adapter_success_preserves_real_pids_service_launch_and_client_exec(self):
        with running_broker() as (outer, ready, server, errors):
            directory, connection = replay(outer, ready, 'adapter')
            common = SimpleNamespace(subprocess=subprocess, ROOT=outer,
                owned_environment=lambda: dict(ENVIRONMENT, NVTX_INJECTION64_PATH='libcuda-injection.so'))
            helpers = SimpleNamespace(training=SimpleNamespace(PYTHON=sys.executable), wrapper=WRAPPER)
            try:
                with broker_client_launches(common, helpers, connection, ENVIRONMENT):
                    # The service remains a normal direct child of this controller.
                    with common.subprocess.Popen([sys.executable, '-B', '-c', 'pass']) as service:
                        self.assertEqual(service.wait(timeout=3), 0)
                    processes = []
                    for role in ('low', 'high'):
                        with (directory / 'equal' / (role + '.log')).open('x') as log:
                            process = common.subprocess.Popen(
                                [sys.executable, '-B', '-c', WRAPPER], cwd=outer,
                                stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                text=True, env=dict(ENVIRONMENT, NVTX_INJECTION64_PATH='libcuda-injection.so'),
                                start_new_session=True)
                        processes.append(process)
                        process.stdin.write(json.dumps(dict(command=[sys.executable, '-B', '-c',
                            'import os; assert "NVTX_INJECTION64_PATH" not in os.environ'],
                            environment=common.owned_environment())) + '\n')
                        process.stdin.flush()
                    self.assertEqual([p.wait(3) for p in processes], [0, 0])
                    receipt = connection.finish()
                    self.assertEqual({p.pid for p in processes}, {r['pid'] for r in receipt.values()})
            finally:
                connection.close()
        self.assertFalse(errors)
        self.assertEqual(server.sessions[0]['status'], 'complete')

    def test_only_registered_client_is_exempt_from_injection_requirement(self):
        process = SimpleNamespace(pid=45, start=67)
        calls = []
        verify = lambda *args: calls.append(args)
        common = SimpleNamespace(verify_process=verify)
        with patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value={}), \
                patch('bert_sgemm_ncu_process.proc_environment', return_value={}):
            with authenticated_profile_processes(common, {}, {'low': process}):
                common.verify_process(None, process, 67, ['client'], {})
                with self.assertRaises(ValueError):
                    common.verify_process(None, process, 68, ['service'], {})
            self.assertIs(common.verify_process, verify)
        self.assertEqual(len(calls), 1)
        with patch('bert_sgemm_ncu_process.proc_mapped_paths', return_value={'injection': (1, 2)}):
            with authenticated_profile_processes(common, {}, {'low': process}):
                with self.assertRaisesRegex(ValueError, 'inherited'):
                    common.verify_process(None, process, 67)

    def test_client_environment_preserves_every_non_profiler_setting(self):
        original = dict(ENVIRONMENT, NVTX_INJECTION64_PATH='libcuda-injection.so', LD_PRELOAD='owned')
        self.assertEqual(clean_client_environment(original), dict(ENVIRONMENT, LD_PRELOAD='owned'))
        self.assertIn('NVTX_INJECTION64_PATH', original)


if __name__ == '__main__':
    unittest.main()
