"""CPU-only replay bridge tests: no sudo, CUDA, service or privileged process."""

from contextlib import contextmanager, ExitStack
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import bert_ncu_replay_exec as bridge
import run_bert_native_ncu_capture as native_capture


def status(uid, gid, capability=False):
    cap = bridge.CAP_SYS_ADMIN if capability else 0
    return (f'Uid:\t{uid} {uid} {uid} {uid}\nGid:\t{gid} {gid} {gid} {gid}\n'
            'Groups:\t\nNoNewPrivs:\t1\n' +
            ''.join(f'{key}:\t{cap:016x}\n' for key in bridge.CAP_FIELDS))


@contextmanager
def launch_fixture():
    with tempfile.TemporaryDirectory(prefix='gr-') as temporary, ExitStack() as resources:
        outer = Path(temporary)
        directory = outer / 'passes/pass-one'
        (directory / 'equal').mkdir(parents=True)
        socket_directory = outer / 'ncu-tmp/glimmer-bert-co-test'
        socket_directory.mkdir(parents=True, mode=0o700)
        log = resources.enter_context((directory / 'equal/server.log').open('x'))
        provider = outer / 'range.so'
        provider.touch()
        resources.enter_context(patch.object(bridge, 'RANGE_LIBRARY', provider))
        plan = dict(op='launch', serial=1, uid=os.getuid(), gid=os.getgid(), directory=str(directory),
                    socket=str(socket_directory / 'service.sock'),
                    clients=dict(low=dict(pid=7, start=70), high=dict(pid=8, start=80)),
                    environment={key: 'sealed' for key in bridge.ENV_KEYS},
                    seals={str(path): 'a' * 64 for path in
                           (bridge.SCRIPT, bridge.SERVICE, bridge.PYTHON, bridge.SETPRIV, bridge.ENV, provider)},
                    deadline_ns=time.monotonic_ns() + 10_000_000_000)
        plan['environment'].update(PATH='/usr/bin:/bin',
                                   CUDA_VISIBLE_DEVICES='GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2')
        template = dict(service_environment=plan['environment'], uid=plan['uid'], gid=plan['gid'])
        (outer / 'manifest.json').write_bytes(bridge.encode(template))
        plan['template_sha256'] = bridge.hashlib.sha256((outer / 'manifest.json').read_bytes()).hexdigest()
        original_read = Path.read_text

        def read(path, *args, **kwargs):
            if path in (Path('/proc/7/status'), Path('/proc/8/status')):
                return status(plan['uid'], plan['gid'])
            return original_read(path, *args, **kwargs)

        resources.enter_context(patch.object(bridge, 'live_identity'))
        resources.enter_context(patch.object(bridge.os, 'getpgid', side_effect=lambda pid: pid))
        resources.enter_context(patch.object(bridge.os, 'getsid', side_effect=lambda pid: pid))
        resources.enter_context(patch.object(bridge.Path, 'read_text', autospec=True, side_effect=read))
        resources.enter_context(patch.object(bridge, 'seal', return_value='a' * 64))
        resources.enter_context(patch.object(bridge, 'protected_system_file'))
        yield outer, plan, log


@contextmanager
def tracker_fixture():
    with tempfile.TemporaryDirectory() as temporary, ExitStack() as resources:
        root = Path(temporary) / 'NVIDIA-treetracker-Test12'
        root.mkdir(mode=0o700)
        descriptors = []
        for role in ('Writing', 'Reading'):
            path = root / ('pipe-0-owner' + role + 'Stream')
            os.mkfifo(path, 0o600)
            path.chmod(0o664)
            descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            descriptors.append(descriptor)
            resources.callback(os.close, descriptor)
        duplicates = [os.dup(descriptor) for descriptor in descriptors]
        for descriptor in duplicates:
            resources.callback(os.close, descriptor)
        tracking = (f'tree-tracker-id: {root}\nconnection-name: {root}/pipe-1\n'
                    f'pipe-descriptor:\n  - {descriptors[0]}\n  - {descriptors[1]}\n'
                    '  - ""\n  - 0\n  - false')
        environment = {'NVIDIA-PROCESS-TRACKING-CONFIGURATION': tracking}
        yield Path(temporary), root, environment, descriptors + duplicates


class ReplayExecTest(unittest.TestCase):
    def test_tracker_retains_advertised_inode_duplicates_only(self):
        with tracker_fixture() as (temporary, root, environment, expected), tempfile.TemporaryFile() as sentinel:
            self.assertEqual(bridge.tracker_descriptors(environment, temporary, os.getuid()), set(expected))
            actions = bridge.child_file_actions(sentinel.fileno(), 200, set(expected))
            closed = {action[1] for action in actions if action[0] == os.POSIX_SPAWN_CLOSE}
            self.assertFalse(closed & set(expected))
            self.assertIn(sentinel.fileno(), closed)

    def test_tracker_retains_other_opens_of_advertised_fifo_without_changing_flags(self):
        with tracker_fixture() as (temporary, root, environment, expected):
            descriptor = os.open(root / 'pipe-0-ownerReadingStream', os.O_RDONLY | os.O_NONBLOCK)
            try:
                before = bridge.fcntl.fcntl(descriptor, bridge.fcntl.F_GETFL)
                self.assertEqual(bridge.tracker_descriptors(environment, temporary, os.getuid()),
                                 {*expected, descriptor})
                self.assertEqual(bridge.fcntl.fcntl(descriptor, bridge.fcntl.F_GETFL), before)
            finally:
                os.close(descriptor)

    def test_tracker_rejects_bad_configuration_roles_owner_and_directory_permissions(self):
        with tracker_fixture() as (temporary, root, environment, descriptors):
            tracking = environment['NVIDIA-PROCESS-TRACKING-CONFIGURATION']
            for changed in (tracking + '\ntree-tracker-id: ' + str(root),
                            tracking + '\npipe-descriptor:\n  - invalid',
                            tracking.replace('  - ""', '  - "unexpected"'),
                            tracking.replace('  - false', '  - true'),
                            tracking.replace(f'  - {descriptors[0]}\n', f'  - {descriptors[1]}\n'),
                            tracking.replace(f'  - {descriptors[1]}\n', '  - 0\n')):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    bridge.tracker_descriptors({'NVIDIA-PROCESS-TRACKING-CONFIGURATION': changed},
                                               temporary, os.getuid())
            with self.assertRaisesRegex(ValueError, 'ownership'):
                bridge.tracker_descriptors(environment, temporary, os.getuid() + 1)
            root.chmod(0o750)
            with self.assertRaisesRegex(ValueError, 'permissions'):
                bridge.tracker_descriptors(environment, temporary, os.getuid())

    def test_tracker_rejects_unadvertised_stream_and_replaced_path(self):
        with tracker_fixture() as (temporary, root, environment, _):
            path = root / 'pipe-2-ownerReadingStream'
            os.mkfifo(path, 0o600)
            descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC)
            try:
                with self.assertRaisesRegex(ValueError, 'unadvertised'):
                    bridge.tracker_descriptors(environment, temporary, os.getuid())
            finally:
                os.close(descriptor)
            original = root / 'pipe-0-ownerWritingStream'
            original.unlink()
            os.mkfifo(original, 0o600)
            with self.assertRaisesRegex(ValueError, 'destination|protected FIFO'):
                bridge.tracker_descriptors(environment, temporary, os.getuid())

    def test_sealed_snapshot_survives_source_replacement_and_cannot_be_modified(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'helper.py'
            original = b'print("original snapshot")\n'
            path.write_bytes(original)
            expected = bridge.hashlib.sha256(original).hexdigest()
            descriptor = bridge.sealed_snapshot(path, expected)
            try:
                path.write_bytes(b'raise RuntimeError("changed pathname")\n')
                bridge.validate_snapshot(descriptor, expected)
                for mutate in (lambda: os.write(descriptor, b'bad'), lambda: os.ftruncate(descriptor, 0)):
                    with self.assertRaises(OSError):
                        mutate()
                command = bridge.sealed_python_command(f'/proc/self/fd/{descriptor}', expected)
                result = subprocess.run(command, pass_fds=(descriptor,), capture_output=True,
                                        text=True, check=True, timeout=5,
                                        env=dict(PATH='/usr/bin:/bin', LANG='C'))
                self.assertEqual(result.stdout, 'original snapshot\n')
                with self.assertRaisesRegex(ValueError, 'snapshot bytes'):
                    bridge.sealed_snapshot(path, expected)
            finally:
                os.close(descriptor)

    def test_sealed_elf_can_execute_without_reopening_source(self):
        descriptor = bridge.sealed_snapshot(Path('/usr/bin/true'), bridge.seal(Path('/usr/bin/true')))
        try:
            subprocess.run([f'/proc/self/fd/{descriptor}'], pass_fds=(descriptor,), check=True,
                           timeout=5, env=dict(PATH='/usr/bin:/bin', LANG='C'))
        finally:
            os.close(descriptor)

    @unittest.skipUnless(Path('/usr/bin/cc').exists(), 'C compiler needed for sealed preload fixture')
    # The child runs bridge.PYTHON, so the mapping audit must require that
    # interpreter rather than the BERT workload environment.
    @patch.object(native_capture.protocol, 'PYTHON', bridge.PYTHON)
    def test_sealed_preload_mapping_is_bound_to_live_child_and_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'libnative_sgemm_range.so'
            subprocess.run(['/usr/bin/cc', '-shared', '-fPIC', '-x', 'c', '-o', str(path), '-'],
                           input='int glimmer_preload_test(void) { return 17; }\n', text=True,
                           check=True, timeout=10, capture_output=True)
            expected = bridge.seal(path)
            descriptor = bridge.sealed_snapshot(path, expected)
            other = bridge.sealed_snapshot(path, expected)
            code = ('import ctypes,sys; '
                    'assert ctypes.CDLL(None).glimmer_preload_test()==17; '
                    'print("ready",flush=True); sys.stdin.readline(); '
                    'ctypes.CDLL(sys.argv[1]); print("source-loaded",flush=True); sys.stdin.readline()')
            command = [str(bridge.SETPRIV), '--no-new-privs', '--inh-caps=-all',
                       '--ambient-caps=-all', str(bridge.ENV), '-i',
                       'PATH=/usr/bin:/bin', f'LD_PRELOAD=/proc/self/fd/{descriptor}',
                       str(bridge.PYTHON), '-I', '-S', '-B', '-c', code, str(path)]
            process = None
            try:
                process = subprocess.Popen(command, pass_fds=(descriptor, other), stdin=subprocess.PIPE,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                self.assertTrue(select.select([process.stdout], [], [], 5)[0], 'preload child did not start')
                self.assertEqual(process.stdout.readline(), 'ready\n')
                identity = bridge.process_identity(process.pid)
                group = bridge.process_group(os.getpid())
                receipt = bridge.validate_target_preload_mapping(descriptor, expected, identity, group, descriptor)
                self.assertEqual(receipt['pid'], process.pid)
                self.assertEqual(receipt['inode'], os.fstat(descriptor).st_ino)
                self.assertEqual(receipt['sha256'], expected)
                self.assertEqual(receipt['mode'], 0o555)
                self.assertEqual(receipt['seals'], bridge.SNAPSHOT_SEALS)
                self.assertTrue(any('x' in permissions for permissions in receipt['mapping_permissions']))
                inputs = {}
                for line in Path(f'/proc/{process.pid}/maps').read_text().splitlines():
                    fields = line.split(maxsplit=5)
                    name = fields[5] if len(fields) == 6 else ''
                    if (('x' not in fields[1] and '.so' not in name) or
                            name in ('[vdso]', '[vsyscall]', bridge.TARGET_PRELOAD_MAP_NAME)):
                        continue
                    mapped = Path(name).resolve(strict=True)
                    inputs[str(mapped)] = dict(target=str(mapped), bytes=mapped.stat().st_size,
                                               sha256=bridge.seal(mapped))
                inputs[str(path.resolve())] = dict(target=str(path.resolve()), bytes=path.stat().st_size,
                                                   sha256=expected)
                audit = native_capture.NativeMaps(inputs, native_capture.mapping_inputs(inputs))
                with self.assertRaisesRegex(ValueError, 'unknown native executable mapping'):
                    audit.observe(process.pid)
                source_status = path.stat()
                forbidden_source = dict(path=str(path.resolve()), device=source_status.st_dev,
                                        inode=source_status.st_ino)
                observation = audit.observe(process.pid, sealed_preload=dict(
                    descriptor=descriptor, expected=expected, identity=identity, group=group,
                    target_descriptor=descriptor), forbidden_source=forbidden_source)
                self.assertEqual(observation['sealed_preload'], receipt)
                for arguments in ((descriptor, '0' * 64, identity, group, descriptor),
                                  (descriptor, expected, dict(identity, start=identity['start'] + 1),
                                   group, descriptor),
                                  (descriptor, expected, identity, '/wrong-group', descriptor),
                                  (descriptor, expected, identity, group, other),
                                  (other, expected, identity, group, other),
                                  (descriptor, expected, identity, group, 2)):
                    with self.subTest(arguments=arguments), self.assertRaises((ValueError, OSError)):
                        bridge.validate_target_preload_mapping(*arguments)
                original_readlink = os.readlink
                target_path = Path(f'/proc/{process.pid}/fd/{descriptor}')
                observations = []

                def replaced_descriptor(path):
                    if path == target_path:
                        observations.append(path)
                        if len(observations) == 2:
                            return '/unexpected'
                    return original_readlink(path)

                with patch.object(bridge.os, 'readlink', side_effect=replaced_descriptor):
                    with self.assertRaisesRegex(ValueError, 'descriptor name differs'):
                        bridge.validate_target_preload_mapping(descriptor, expected, identity, group, descriptor)
                self.assertEqual(len(observations), 2)
                os.fchmod(other, 0o700)
                with self.assertRaisesRegex(ValueError, 'mode or seal set differs'):
                    bridge.validate_target_preload_mapping(other, expected, identity, group, other)
                process.stdin.write('\n')
                process.stdin.flush()
                self.assertTrue(select.select([process.stdout], [], [], 5)[0], 'source mapping did not start')
                self.assertEqual(process.stdout.readline(), 'source-loaded\n')
                with self.assertRaisesRegex(ValueError, 'diagnostic source mapping forbidden'):
                    audit.observe(process.pid, sealed_preload=dict(
                        descriptor=descriptor, expected=expected, identity=identity, group=group,
                        target_descriptor=descriptor), forbidden_source=forbidden_source)
                process.stdin.write('\n')
                process.stdin.flush()
                self.assertEqual(process.wait(timeout=5), 0)
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None:
                            stream.close()
                os.close(other)
                os.close(descriptor)

    def test_real_nonroot_spawn_setpriv_env_sealed_elf_chain(self):
        # CPU loader compatibility only: dropping bounding caps/groups needs root.
        descriptor = bridge.sealed_snapshot(bridge.PYTHON, bridge.seal(bridge.PYTHON))
        with tempfile.TemporaryFile(mode='w+') as log:
            try:
                os.set_inheritable(descriptor, True)
                code = ('import json,os; from pathlib import Path; '
                        'print(json.dumps(dict(uid=os.getuid(),env=dict(os.environ),'
                        'status=Path("/proc/self/status").read_text())))')
                command = [str(bridge.SETPRIV), '--no-new-privs', '--inh-caps=-all',
                           '--ambient-caps=-all', str(bridge.ENV), '-i', 'PATH=/usr/bin:/bin',
                           'LANG=C.UTF-8', f'/proc/self/fd/{descriptor}', '-I', '-S', '-B', '-c', code]
                pid = os.posix_spawn(command[0], command, dict(PATH='/usr/bin:/bin'),
                                     file_actions=bridge.child_file_actions(log.fileno(), descriptor),
                                     setsid=True)
                child = bridge.process_identity(pid)
                pidfd = os.pidfd_open(pid)
                reaped = False
                try:
                    receipt = bridge.reap_child(child, time.monotonic_ns() + 3 * 10 ** 9)
                    reaped = True
                    self.assertEqual(receipt['returncode'], 0)
                    log.seek(0)
                    result = bridge.json.loads(log.read())
                    self.assertNotEqual(result['uid'], 0)
                    self.assertEqual(result['env'], dict(PATH='/usr/bin:/bin', LANG='C.UTF-8'))
                    fields = dict(line.split(':', 1) for line in result['status'].splitlines())
                    self.assertEqual(fields['NoNewPrivs'].strip(), '1')
                    for key in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb'):
                        self.assertEqual(int(fields[key], 16), 0)
                finally:
                    if not reaped:
                        bridge.retire_child(child, pidfd)
                    os.close(pidfd)
            finally:
                os.close(descriptor)

    @unittest.skipUnless(Path('/usr/bin/cc').exists(), 'C compiler needed for sealed preload fixture')
    def test_target_mode_real_capability_free_low_high_lifecycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / 'pass'
            (directory / 'native').mkdir(parents=True)
            source, provider = root / 'libnative_sgemm_range.so', root / 'libcublas.so'
            subprocess.run(['/usr/bin/cc', '-shared', '-fPIC', '-x', 'c', '-o', str(source), '-'],
                           input='int glimmer_preload_test(void) { return 17; }\n', text=True,
                           check=True, timeout=10, capture_output=True)
            provider.write_bytes(b'provider')
            expected = bridge.seal(source)
            descriptor = bridge.sealed_snapshot(source, expected)
            gate = root / 'finish'
            # The while loop exits naturally when the test creates the gate.
            script = ('import pathlib,sys,time\nprint("ready",flush=True)\n'
                      'gate=pathlib.Path(sys.argv[1])\nwhile not gate.exists(): time.sleep(.02)')
            command = [str(bridge.PYTHON), '-I', '-S', '-B', '-c', script, str(gate)]
            inputs = {str(path): dict(target=str(path), bytes=path.stat().st_size, sha256=bridge.seal(path))
                      for path in (source, provider)}
            frozen = native_capture.mapping_inputs(inputs)
            metadata = source.stat()
            shim = dict(descriptor=descriptor, expected=expected, provider=str(provider),
                        forbidden_source=dict(path=str(source), device=metadata.st_dev, inode=metadata.st_ino))
            manifest = dict(uid=os.getuid(), gid=os.getgid(), ncu_injection_seals={})
            targets = []

            def verify_capability_free(target):
                deadline = time.monotonic() + 5
                log = directory / 'native' / (target.role + '.log')
                while time.monotonic() < deadline and 'ready' not in log.read_text():
                    self.assertIsNone(target.poll())
                    time.sleep(.01)
                self.assertIn('ready', log.read_text())
                bridge.live_identity(target.identity, target.group)
                fields = dict(line.split(':', 1) for line in
                              Path(f'/proc/{target.identity["pid"]}/status').read_text().splitlines())
                self.assertEqual(fields['Uid'].split(), [str(os.getuid())] * 4)
                self.assertEqual(fields['NoNewPrivs'].strip(), '1')
                self.assertTrue(all(int(fields[name], 16) == 0 for name in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb')))
                proc = Path(f'/proc/{target.identity["pid"]}')
                environment = proc.joinpath('environ').read_bytes().split(b'\0')
                mappings = proc.joinpath('maps').read_text()
                if target.role == 'low':
                    self.assertIn(f'LD_PRELOAD=/proc/self/fd/{descriptor}'.encode(), environment)
                    self.assertIn(f'GLIMMER_TRACE_CUBLAS_PROVIDER={provider}'.encode(), environment)
                    self.assertEqual(os.readlink(proc / 'fd' / str(descriptor)), bridge.TARGET_PRELOAD_MAP_NAME)
                    self.assertIn(bridge.TARGET_PRELOAD_MAP_NAME, mappings)
                    self.assertNotIn(str(source), mappings)
                else:
                    self.assertFalse(any(item.startswith((b'LD_PRELOAD=', b'GLIMMER_TRACE_')) for item in environment))
                    self.assertNotIn(bridge.TARGET_PRELOAD_MAP_NAME, mappings)
                    fd = proc / 'fd' / str(descriptor)
                    try:
                        inherited_name = os.readlink(fd)
                    except FileNotFoundError:
                        inherited_name = None
                    self.assertNotEqual(inherited_name, bridge.TARGET_PRELOAD_MAP_NAME)

            try:
                with ExitStack() as stack:
                    for name, value in (('target_environment', {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}),
                                        ('bridge.drop_command', [str(bridge.SETPRIV), '--no-new-privs',
                                         '--inh-caps=-all', '--ambient-caps=-all']),
                                        ('bridge.tracker_descriptors', set()),
                                        ('protocol.target_application_commands', {'low': command, 'high': command}),
                                        ('NativeTarget.verify_bootstrap', None)):
                        stack.enter_context(patch('run_bert_native_ncu_capture.' + name, return_value=value))
                    stack.enter_context(patch.object(native_capture.NativeTarget, 'verify', verify_capability_free))
                    for role in ('high', 'low'):
                        log = stack.enter_context((directory / 'native' / (role + '.log')).open('w+'))
                        targets.append(native_capture.NativeTarget(
                            root, manifest, directory, role, log.fileno(), dict(PATH='/usr/bin:/bin'),
                            inputs, frozen, target_mode=True, sealed_shim=shim if role == 'low' else None))
                        targets[-1].release()
                    os.fstat(descriptor)
                    gate.touch()
                    while targets:
                        target = targets[0]
                        deadline = time.monotonic() + 5
                        while target.poll() is None and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertEqual(target.receipt['returncode'], 0)
                        target.close()
                        targets.pop(0)
                    os.fstat(descriptor)
            finally:
                for target in targets:
                    target.close()
                os.close(descriptor)

    def test_profiler_environment_rejects_unsealed_inherited_loader_and_control_values(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as mocks:
            outer = Path(temporary)
            library = outer / 'libcuda-injection.so'
            library.touch()
            mocks.enter_context(patch.object(bridge, 'NCU_DIRECTORY', outer))
            mocks.enter_context(patch.object(bridge, 'protected_system_file'))
            mocks.enter_context(patch.object(bridge, 'seal', return_value='a' * 64))
            gpu = 'GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2'
            base = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(outer / 'ncu-tmp'),
                        CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_CACHE_DISABLE='1')
            template = dict(ncu_supervisor_environment=base, service_environment=dict(CUDA_VISIBLE_DEVICES=gpu),
                            ncu_injection_seals={str(library): 'a' * 64})
            inherited = dict(base, LD_LIBRARY_PATH=str(outer), NV_COMPUTE_PROFILER_PERFWORKS_DIR=str(outer),
                             NVTX_INJECTION64_PATH='libcuda-injection.so', LD_PRELOAD=str(library),
                             **{'NVIDIA-PROCESS-TRACKING-CONFIGURATION':
                                'not-launcher-preload-prepend: ' + str(library)})
            self.assertEqual(bridge.profiler_environment(inherited, template, outer), inherited)
            replay = dict(inherited, NV_COMPUTE_PROFILER_DISABLE_FUNCTION_LAZY_LOADING='1')
            self.assertEqual(bridge.profiler_environment(replay, template, outer), replay)
            for key, value in (('LD_AUDIT', '/bad'), ('PYTHONPATH', '/bad'), ('LD_PRELOAD', '/usr/bin/true'),
                                ('LD_LIBRARY_PATH', '/tmp'), ('PATH', '/bad'),
                                ('NVTX_INJECTION64_PATH', '/bad'), ('NV_CUDA_START_SUSPENDED', '0'),
                                ('NV_COMPUTE_PROFILER_DISABLE_FUNCTION_LAZY_LOADING', '0'),
                                ('NV_COMPUTE_PROFILER_DISABLE_FUNCTION_LAZY_LOADING', ''),
                                ('NV_UNEXPECTED_CONTROL', 'private-value-not-for-errors'),
                                ('NVIDIA-PROCESS-TRACKING-CONFIGURATION',
                                 'not-launcher-preload-prepend: /usr/bin/true')):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    bridge.profiler_environment(dict(inherited, **{key: value}), template, outer)
            for key, value in (('NV_UNEXPECTED_CONTROL', 'private-value-not-for-errors'),
                               ('NV_COMPUTE_PROFILER_DISABLE_FUNCTION_LAZY_LOADING', 'private-wrong-value')):
                with self.subTest(key=key), self.assertRaises(ValueError) as raised:
                    bridge.profiler_environment(dict(inherited, **{key: value}), template, outer)
                self.assertIn(key, str(raised.exception))
                self.assertNotIn(value, str(raised.exception))

    def test_launch_requires_exact_frozen_service_values(self):
        with launch_fixture() as (outer, plan, log):
            for key in ('LD_LIBRARY_PATH', 'LANG', 'MKL_NUM_THREADS'):
                changed = dict(plan, environment=dict(plan['environment'], **{key: 'different'}))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'frozen template'):
                    bridge.validate_launch(changed, outer, '/task', log.fileno())
            (outer / 'manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'frozen manifest differs'):
                bridge.validate_launch(plan, outer, '/task', log.fileno())

    def test_fresh_replay_requires_claim_and_stays_blocked_after_authentication(self):
        with launch_fixture() as (outer, plan, log):
            template = bridge.frozen_template(outer, plan['template_sha256'])
            with self.assertRaisesRegex(ValueError, 'protocol fields differ'):
                bridge.validate_launch(dict(plan, fresh_claim_sha256='a' * 64), outer, '/task', log.fileno())
            template['fresh_runtime'] = {}
            with patch.object(bridge, 'frozen_template', return_value=template), \
                    patch.object(bridge, 'authenticate_fresh_replay', return_value=None) as authenticate:
                with self.assertRaisesRegex(ValueError, 'protocol fields differ'):
                    bridge.validate_launch(plan, outer, '/task', log.fileno())
                selected = dict(plan, fresh_claim_sha256='a' * 64)
                with self.assertRaisesRegex(ValueError, 'fresh runtime service authentication is not implemented'):
                    bridge.validate_launch(selected, outer, '/task', log.fileno())
                authenticate.assert_called_once_with(selected, outer, template)

    def test_fresh_replay_threads_authenticated_context_to_service(self):
        with launch_fixture() as (outer, plan, log):
            service = outer / 'service'
            service.write_bytes(b'service fixture')
            service.chmod(0o775)
            info = service.stat()
            product = dict(path=str(service), device=info.st_dev, inode=info.st_ino, size=info.st_size,
                           mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns, mode=0o775,
                           uid=info.st_uid, gid=info.st_gid, sha256='a' * 64)
            binding = dict(schema='fresh_runtime_v2', variant='sharded',
                           contract=dict(schema='fresh_runtime_sgemm_v2', target='2:3127', runtime=dict(service=product),
                                         service_environment=plan['environment']))
            template = dict(uid=plan['uid'], gid=plan['gid'], fresh_runtime=binding,
                            service_environment=plan['environment'], sgemm_target='2:3127',
                            service_cap_sys_admin=True, bridge_seals={str(service): 'a' * 64})
            plan['seals'].pop(str(bridge.SERVICE))
            plan['seals'][str(service)] = 'a' * 64
            plan['fresh_claim_sha256'] = 'c' * 64
            authority = dict(claim_sha256='c' * 64, manifest_sha256=plan['template_sha256'],
                             binding_sha256='d' * 64, binding=binding, template=template)
            with patch.object(bridge, 'frozen_template', return_value=template), \
                    patch.object(bridge, 'authenticate_fresh_replay', return_value=authority), \
                    patch.object(bridge.os, 'geteuid', return_value=0), \
                    patch.object(bridge.os, 'getegid', return_value=0), \
                    patch.object(bridge, 'profiler_environment', return_value={}):
                directory, verified = bridge.validate_launch(plan, outer, '/task', log.fileno())
                self.assertEqual(directory, Path(plan['directory']))
                self.assertEqual(verified, authority)
                command = bridge.service_exec_command(plan, outer, {}, 123, fresh_authority=verified)
                self.assertIn('/proc/self/fd/123', command)
                self.assertIn('--experimental-sharded-gemm-signal', command)
                with self.assertRaisesRegex(ValueError, 'context differs'):
                    bridge.selected_service(template, fresh_authority=dict(authority, binding={}))

    def test_cold_isolated_help(self):
        result = subprocess.run(['/usr/bin/python3', '-I', '-S', '-B', str(bridge.SCRIPT), '--help'],
                                capture_output=True, text=True, check=True, timeout=5,
                                env=dict(PATH='/usr/bin:/bin', LANG='C'))
        self.assertIn('Private NCU replay launch bridge', result.stdout)

    def test_protocol_rejects_duplicates_nonfinite_and_nonobjects(self):
        for data in (b'', b'[]', b'{"op":1,"op":2}', b'{"nested":{"x":1,"x":2}}',
                     b'{"x":NaN}', b'{"x":Infinity}', b'x' * (bridge.MESSAGE_MAX + 1)):
            with self.subTest(data=data[:80]), self.assertRaises(ValueError):
                bridge.decode(data)
        self.assertEqual(bridge.decode(bridge.encode(dict(op='hello'))), dict(op='hello'))

    def test_privilege_drop_is_service_only(self):
        for service in (False, True):
            command = bridge.drop_command(1003, 1003, service=service)
            expected = '-all,+sys_admin' if service else '-all'
            for option in ('inh-caps', 'ambient-caps', 'bounding-set'):
                self.assertIn('--' + option + '=' + expected, command)
            self.assertIn('--no-new-privs', command)
            self.assertIn('--clear-groups', command)
        for uid, gid, service in ((0, 1, True), (True, 1, True), (1, True, True),
                                  (1, -1, True), (1, 2 ** 32, True), (1, 1, 1)):
            with self.assertRaises(ValueError):
                bridge.drop_command(uid, gid, service=service)

    def test_actual_credentials_reject_extra_capabilities(self):
        for capability in (False, True):
            expected = status(1003, 1003, capability)
            bridge.validate_status(expected, 1003, 1003, capability)
            for key in bridge.CAP_FIELDS:
                cap = bridge.CAP_SYS_ADMIN if capability else 0
                changed = expected.replace(f'{key}:\t{cap:016x}', f'{key}:\t{(cap | 1):016x}')
                with self.assertRaisesRegex(ValueError, 'capability'):
                    bridge.validate_status(changed, 1003, 1003, capability)
            for changed in (expected.replace('NoNewPrivs:\t1', 'NoNewPrivs:\t0'),
                            expected.replace('Groups:\t\n', 'Groups:\t1003\n'),
                            expected.replace('1003', '0'), expected + 'Uid:\t1003\n'):
                with self.assertRaises(ValueError):
                    bridge.validate_status(changed, 1003, 1003, capability)

    def test_exact_template_constructs_only_current_service(self):
        with launch_fixture() as (outer, plan, log):
            self.assertEqual(bridge.validate_launch(plan, outer, '/task', log.fileno()), Path(plan['directory']))
            with patch.object(bridge, 'profiler_environment', return_value={}):
                command = bridge.service_exec_command(plan, outer, {}, 123)
                with self.assertRaisesRegex(ValueError, 'fresh authority mode differ'):
                    bridge.service_exec_command(plan, outer, {}, 123, fresh_authority={})
            self.assertEqual(command[0], str(bridge.SETPRIV))
            self.assertNotIn(str(bridge.PYTHON), command)
            self.assertIn('7:70:0:configured', command)
            self.assertIn('8:80:0:original', command)
            self.assertIn('-i', command)
            self.assertIn('/proc/self/fd/123', command)
            self.assertNotIn(str(bridge.SERVICE), command)
            service = bridge.service_command(plan)
            self.assertEqual(service[0], str(bridge.SERVICE))
            self.assertEqual(service[service.index('--persistent-workers') + 1], '320')
            self.assertEqual(service[service.index('--audit-sgemm-target-client') + 1], '7:70:3127')
            self.assertNotIn('--audit-sgemm-target', service)
            self.assertEqual(service[service.index('--gemm-tile-side') + 1], '32')

    def test_launch_rejects_command_environment_and_shape_injection(self):
        with launch_fixture() as (outer, plan, log):
            changed_plans = [dict(plan, argv=['/bin/sh']), dict(plan, op='exec'), dict(plan, serial=True),
                             dict(plan, deadline_ns=1), dict(plan, clients=dict(low=plan['clients']['low'])),
                             dict(plan, clients=dict.fromkeys(('low', 'high'), plan['clients']['low']))]
            for key in ('LD_PRELOAD', 'LD_AUDIT', 'PYTHONPATH', 'NVTX_INJECTION64_PATH', 'BASH_ENV'):
                changed_plans.append(dict(plan, environment=dict(plan['environment'], **{key: '/bad'})))
            changed_plans.append(dict(plan, environment=dict(plan['environment'], PATH='/bad')))
            changed_plans.append(dict(plan, environment=dict(plan['environment'], LANG='C\0bad')))
            changed_plans.append(dict(plan, seals=dict(plan['seals'], **{'/bin/sh': 'a' * 64})))
            for changed in changed_plans:
                with self.subTest(changed=changed.keys()), self.assertRaises(ValueError):
                    bridge.validate_launch(changed, outer, '/task', log.fileno())

    def test_launch_rejects_foreign_or_reused_output_paths(self):
        with launch_fixture() as (outer, plan, log):
            for path in (str(outer / 'service.sock'), '/tmp/foreign/service.sock',
                         str(Path(plan['socket']).with_name('other.sock'))):
                with self.assertRaises((ValueError, OSError)):
                    bridge.validate_launch(dict(plan, socket=path), outer, '/task', log.fileno())
            Path(plan['socket']).touch()
            with self.assertRaisesRegex(ValueError, 'socket'):
                bridge.validate_launch(plan, outer, '/task', log.fileno())
        with launch_fixture() as (outer, plan, log):
            log.write('previous output')
            log.flush()
            with self.assertRaisesRegex(ValueError, 'log descriptor'):
                bridge.validate_launch(plan, outer, '/task', log.fileno())

    def test_launch_rejects_readonly_alias_and_changed_seal(self):
        with launch_fixture() as (outer, plan, log):
            with Path(log.name).open('r') as reader:
                with self.assertRaisesRegex(ValueError, 'log descriptor'):
                    bridge.validate_launch(plan, outer, '/task', reader.fileno())
            with patch.object(bridge, 'seal', return_value='b' * 64):
                with self.assertRaisesRegex(ValueError, 'seal'):
                    bridge.validate_launch(plan, outer, '/task', log.fileno())

    def test_wait_status_never_infers_success(self):
        for raw, expected in ((0, 0), (17 << 8, 17), (int(signal.SIGTERM), -signal.SIGTERM),
                              (signal.SIGSEGV | 0x80, -signal.SIGSEGV)):
            receipt = bridge.wait_receipt(7, 70, raw)
            self.assertEqual(receipt['returncode'], expected)
            self.assertTrue(receipt['exit_code_observed'])
            self.assertEqual(receipt['raw_status'], raw)
        for raw in (None, True, -1, 65536, 0x7f, 0xffff):
            with self.assertRaises(ValueError):
                bridge.wait_receipt(7, 70, raw)

    def test_wait_requires_real_parenthood_and_terminal_status(self):
        with patch.object(bridge.os, 'waitpid', side_effect=ChildProcessError('not a child')):
            with self.assertRaises(ChildProcessError):
                bridge.reap_child(dict(pid=7, start=70), time.monotonic_ns() + 10 ** 9)
        with patch.object(bridge.os, 'waitpid', return_value=(8, 0)):
            with self.assertRaisesRegex(ValueError, 'another child'):
                bridge.reap_child(dict(pid=7, start=70), time.monotonic_ns() + 10 ** 9)
        with patch.object(bridge.os, 'waitpid') as wait:
            with self.assertRaises(TimeoutError):
                bridge.reap_child(dict(pid=7, start=70), 1)
            wait.assert_not_called()

    def test_actual_waitpid_exit_and_signal(self):
        for code in (0, 17, -signal.SIGTERM):
            process = subprocess.Popen(['/usr/bin/python3', '-I', '-S', '-B', '-c',
                                        'import os,signal; ' + (f'os._exit({code})' if code >= 0 else
                                        f'os.kill(os.getpid(), {-code})')],
                                       env=dict(PATH='/usr/bin:/bin', LANG='C'))
            try:
                child = bridge.process_identity(process.pid)
                receipt = bridge.reap_child(child, time.monotonic_ns() + 3_000_000_000)
                process.returncode = receipt['returncode']
                self.assertEqual(receipt['returncode'], code)
            finally:
                if process.returncode is None:
                    process.kill()
                    process.wait(timeout=3)

    def test_waiter_requires_full_pass_ack(self):
        for accepted in (True, False):
            with ExitStack() as mocks:
                def mock(target, name, **kwargs):
                    return mocks.enter_context(patch.object(target, name, **kwargs))
                mock(bridge, 'validate_status')
                mock(bridge.Path, 'read_text', return_value='status')
                mock(bridge.os, 'getresuid', return_value=(1003,) * 3)
                mock(bridge.os, 'getresgid', return_value=(1003,) * 3)
                mock(bridge.os, 'getgroups', return_value=[])
                mock(bridge, 'authenticate_supervisor', return_value=('/task', 42))
                mock(bridge, 'process_group', return_value='/task')
                mock(bridge, 'validate_child_fd')
                mock(bridge, 'process_identity', return_value=dict(pid=6, start=60))
                send = mock(bridge, 'send_packet')
                receive = mock(bridge, 'receive_no_fds', side_effect=[dict(op='wait', serial=1),
                                   dict(op='finish', serial=1, accepted=accepted)])
                mock(bridge, 'terminal', return_value=False)
                mock(bridge, 'reap_child', return_value=bridge.wait_receipt(7, 70, 0))
                close = mock(bridge.os, 'close')
                self.assertEqual(bridge.wait_only(Mock(), dict(pid=5, start=50), Path('/attempt'),
                                 dict(pid=7, start=70), 43, 1003, 1003, 1,
                                 time.monotonic_ns() + 10 ** 9), 0 if accepted else 1)
                self.assertEqual(receive.call_count, 2)
                self.assertEqual([call.args[1]['op'] for call in send.call_args_list], ['waiter_ready', 'exit'])
                self.assertEqual([call.args for call in close.call_args_list], [(43,), (42,)])

    def test_waiter_supervisor_loss_and_deadline_kill_and_reap_real_child(self):
        for failure in ('supervisor', 'deadline'):
            with self.subTest(failure=failure), ExitStack() as resources:
                process = subprocess.Popen(['/usr/bin/python3', '-I', '-S', '-B', '-c',
                                            'import time; time.sleep(10)'],
                                           env=dict(PATH='/usr/bin:/bin', LANG='C'))
                child = bridge.process_identity(process.pid)
                child_fd = os.pidfd_open(process.pid)
                root_read, root_write = os.pipe()
                os.close(root_write)
                left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
                resources.enter_context(left)
                resources.enter_context(right)
                resources.enter_context(patch.object(bridge, 'validate_status'))
                resources.enter_context(patch.object(bridge.os, 'getgroups', return_value=[]))
                resources.enter_context(patch.object(bridge, 'authenticate_supervisor',
                                                      return_value=('/task', root_read)))
                resources.enter_context(patch.object(bridge, 'send_packet'))
                resources.enter_context(patch.object(bridge, 'receive_no_fds',
                                                      return_value=dict(op='wait', serial=1)))
                # The real inherited pidfd, actual parenthood and exact process start are checked.
                until = time.monotonic_ns() + 10 ** 9 if failure == 'supervisor' else 1
                try:
                    with self.assertRaises((ValueError, TimeoutError)):
                        bridge.wait_only(left, dict(pid=5, start=50), Path('/attempt'), child,
                                         child_fd, os.getuid(), os.getgid(), 1, until)
                    process.returncode = -signal.SIGKILL
                    self.assertFalse(Path(f'/proc/{process.pid}').exists())
                    with self.assertRaises(ChildProcessError):
                        os.waitpid(process.pid, os.WNOHANG)
                finally:
                    if process.returncode is None:
                        process.kill()
                        process.wait(timeout=3)

    def test_real_packets_credentials_and_fd_ownership(self):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        with left, right, tempfile.TemporaryFile() as log:
            right.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            bridge.send_packet(left, dict(op='launch'), [log.fileno()])
            value, descriptors, credentials = bridge.receive_packet(right)
            try:
                self.assertEqual(value, dict(op='launch'))
                self.assertEqual(credentials, (os.getpid(), os.getuid(), os.getgid()))
                self.assertEqual(len(descriptors), 1)
                self.assertEqual(os.fstat(descriptors[0]).st_ino, os.fstat(log.fileno()).st_ino)
                self.assertFalse(os.get_inheritable(descriptors[0]))
            finally:
                for descriptor in descriptors:
                    os.close(descriptor)

    def test_invalid_packet_closes_received_fds(self):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        with left, right, tempfile.TemporaryFile() as log:
            for payload in (b'{"op":1,"op":2}', b'x' * (bridge.MESSAGE_MAX + 1)):
                left.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                         bridge.array.array('i', [log.fileno()]))])
                with patch.object(bridge.os, 'close', wraps=os.close) as close:
                    with self.assertRaises(ValueError):
                        bridge.receive_packet(right)
                    close.assert_called_once()
                self.assertFalse(log.closed)

    def test_launch_exec_failure_kills_only_owned_child_and_closes_fds(self):
        with ExitStack() as mocks:
            def mock(target, name, **kwargs):
                return mocks.enter_context(patch.object(target, name, **kwargs))
            mock(bridge.os, 'getresuid', return_value=(0, 0, 0))
            mock(bridge, 'authenticate_supervisor', return_value=('/task', 42))
            mock(bridge, 'process_identity', return_value=dict(pid=7, start=70))
            mock(bridge, 'live_identity')
            mock(bridge, 'send_packet')
            mock(bridge, 'receive_packet', return_value=(dict(serial=1, uid=1003, gid=1003,
                 deadline_ns=10 ** 18, template_sha256='sealed',
                 seals={str(bridge.SCRIPT): 'code', str(bridge.SERVICE): 'binary'}), [44], (5, 0, 0)))
            mock(bridge, 'validate_launch')
            mock(bridge, 'frozen_template', return_value={})
            mock(bridge, 'profiler_environment', return_value={'PATH': '/usr/bin:/bin'})
            mock(bridge, 'validate_snapshot')
            mock(bridge, 'sealed_snapshot', return_value=46)
            mock(bridge, 'CURRENT_CODE_FD', new=47)
            mock(bridge, 'terminal', return_value=False)
            mock(bridge, 'service_exec_command', return_value=['/usr/bin/setpriv', 'service'])
            mock(bridge, 'tracker_descriptors', return_value={48, 49})
            actions = [(os.POSIX_SPAWN_CLOSE, 45)]
            mock(bridge, 'child_file_actions', return_value=actions)
            spawn = mock(bridge.os, 'posix_spawn', return_value=7)
            mock(bridge.os, 'pidfd_open', return_value=43)
            mock(bridge.os, 'set_inheritable')
            close_except = mock(bridge, 'close_except')
            execute = mock(bridge.os, 'execve', side_effect=OSError('exec failed'))
            kill = mock(bridge.signal, 'pidfd_send_signal')
            wait = mock(bridge.os, 'waitpid', return_value=(7, int(signal.SIGKILL)))
            close = mock(bridge.os, 'close')
            connection = Mock()
            connection.fileno.return_value = 45
            with self.assertRaisesRegex(OSError, 'exec failed'):
                bridge.launch_once(connection, dict(pid=5, start=50), Path('/attempt'))
            self.assertEqual(spawn.call_args.kwargs['file_actions'], actions)
            self.assertTrue(spawn.call_args.kwargs['setsid'])
            command = execute.call_args.args[1]
            self.assertIn('--bounding-set=-all', command)
            self.assertNotIn('--bounding-set=-all,+sys_admin', command)
            self.assertIn('--wait-fd', command)
            self.assertIn('-I', command)
            self.assertIn('-S', command)
            kill.assert_called_once_with(43, signal.SIGKILL)
            wait.assert_called_once_with(7, os.WNOHANG)
            close_except.assert_called_once_with({43, 45, 47, 48, 49})
            self.assertEqual(sorted(call.args[0] for call in close.call_args_list), [42, 43, 44, 46])

    def test_fresh_runtime_remains_blocked_before_service_snapshot_or_spawn(self):
        with ExitStack() as mocks:
            def mock(target, name, **kwargs):
                return mocks.enter_context(patch.object(target, name, **kwargs))
            mock(bridge.os, 'getresuid', return_value=(0, 0, 0))
            mock(bridge, 'authenticate_supervisor', return_value=('/task', 42))
            mock(bridge, 'process_identity', return_value=dict(pid=7, start=70))
            mock(bridge, 'send_packet')
            mock(bridge, 'receive_packet', return_value=(dict(serial=1, uid=1003, gid=1003,
                 deadline_ns=10 ** 18, template_sha256='sealed', seals={str(bridge.SCRIPT): 'code'}),
                 [44], (5, 0, 0)))
            mock(bridge, 'validate_launch')
            mock(bridge, 'frozen_template', return_value={'fresh_runtime': {}})
            mock(bridge, 'profiler_environment', return_value={'PATH': '/usr/bin:/bin'})
            mock(bridge, 'validate_snapshot')
            snapshot = mock(bridge, 'service_executable_snapshot')
            spawn = mock(bridge.os, 'posix_spawn')
            close = mock(bridge.os, 'close')
            with self.assertRaisesRegex(ValueError, 'fresh authority mode differ'):
                bridge.launch_once(Mock(), dict(pid=5, start=50), Path('/attempt'))
            snapshot.assert_not_called()
            spawn.assert_not_called()
            self.assertEqual(sorted(call.args[0] for call in close.call_args_list), [42, 44])


if __name__ == '__main__':
    unittest.main()
