"""Fixed native BERT counter recipes and caps0 controller adapters.

Only two predeclared application roles cross this protocol. It never forwards
arbitrary commands, environments, signals or a privileged process's stdin.
The surrounding collector must authenticate peers, bound resources and retain
actual parent wait statuses. This module alone does not grant qualification.
"""

from contextlib import contextmanager
from collections import Counter
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import threading
import time
from types import ModuleType

import bert_ncu_replay_exec as bridge

PROJECT = Path(__file__).resolve().parents[1]
ENTRY = PROJECT / 'scripts/run_bert_native_ncu_capture.py'
PYTHON = PROJECT / 'examples/framework_workloads/pytorch_smoke/.venv/bin/python'
EXAMPLE = PROJECT / 'examples/framework_workloads/bert'
ROLES = ('low', 'high')
METRICS = ('launch__sm_count', 'launch__grid_size', 'launch__block_size',
           'sm__cycles_active.min', 'sm__sass_thread_inst_executed_op_ffma_pred_on.min')
KERNEL_FILTER = 'regex:^volta_sgemm_.*$'
TARGET_KERNEL = 'volta_sgemm_128x32_nn'
TARGET_DESCRIPTOR = (0, 0, 3072, 768, 768, 3072, 768, 3072)
TARGET_CALLS_PER_ITERATION = 146
TARGET_ORDINAL = 149
TARGET_NVTX_FILTER = r'regex:glimmer\.native\.sgemm\.pid-[0-9]+@GlimmerNativeSgemm149-pid-[0-9]+/'
NCU_BINARY = '/opt/nvidia/nsight-compute/2024.3.2/ncu'
ONNX_VALIDATOR = PROJECT / 'build/tally-full-implementation-1/owned_onnx_result.py'
require = bridge.require


def application_commands(manifest, directory):
    """Construct commands; no executable, argv or environment comes from RPC."""
    directory = Path(directory)
    require(directory.is_absolute() and manifest['python'] == str(PYTHON), 'fixed native Python required')
    result = {
        'low': [str(PYTHON), '-B', str(EXAMPLE / 'bert_workload.py'),
                '--assets', manifest['assets'], '--output', str(directory / 'training'),
                '--role', 'training', '--warmup', '1', '--iterations', '2', '--snapshot',
                '--reference', manifest['startup_reference']],
        'high': [str(PYTHON), '-B', str(EXAMPLE / 'onnx_bert_workload.py'),
                 '--assets', manifest['assets'], '--export', manifest['export'],
                 '--output', str(directory / 'inference'), '--warmup', '1', '--iterations', '2']}
    for role, command in result.items():
        command += ['--ready-file', str(directory / (role + '.ready.json')),
                    '--start-file', str(directory / (role + '.start'))]
    return result


def target_application_commands(manifest, directory):
    """Measured-phase target recipe; this does not change the native default."""
    commands = application_commands(manifest, directory)
    for command in commands.values():
        require('--profile-cuda' not in command, 'duplicate profiler phase marker')
        command.append('--profile-cuda')
    return commands


def target_command_binding(manifest, directory, job, targets):
    """Join logical releases to root-recorded argv; caller authenticates both receipts."""
    logical = application_commands(manifest, directory)
    actual = target_application_commands(manifest, directory)
    require(type(job) is dict and job.get('mode') == 'native' and job.get('application_commands') == logical and
            type(job.get('application_identities')) is dict and
            set(job['application_identities']) == set(ROLES) and type(targets) is dict and
            set(targets) == set(ROLES), 'complete logical and target role records required')
    for role in ROLES:
        identity = job['application_identities'][role]
        bridge.validate_identity(identity)
        record = targets[role]
        require(actual[role] == [*logical[role], '--profile-cuda'] and
                type(record) is dict and record.get('role') == role and record.get('identity') == identity and
                record.get('command') == actual[role] and record.get('released') is True and
                type(record.get('samples')) is int and record['samples'] > 0,
                'root-recorded target command or role differs')
    require(job['application_identities']['low']['pid'] != job['application_identities']['high']['pid'],
            'distinct target role identities required')
    return dict(logical_release_commands=logical, root_recorded_target_commands=actual,
                identities={role: dict(job['application_identities'][role]) for role in ROLES},
                status='target_commands_require_authenticated_root_receipts')


def _instrumented_onnx_adapter(module, source_sha256, event_start):
    """Use an isolated original validator, changing only the required profiler flag."""
    require(type(event_start) is bool and type(source_sha256) is str and
            re.fullmatch(r'[0-9a-f]{64}', source_sha256) is not None and
            callable(module._exact) and callable(module.validate_onnx_result),
            'frozen inference validator and explicit barrier mode required')
    original_exact = module._exact
    lock = threading.Lock()

    def validate(directory, gpu_uuid, export_root, *, expected_warmup=1, expected_iterations=2, reference=None):
        require(reference is None, 'instrumented native inference requires no fresh reference')
        require(lock.acquire(blocking=False), 'instrumented inference validator already in use')
        installed = False
        try:
            require(module._exact is original_exact, 'instrumented inference validator is not isolated')
            matched = 0

            def checked_exact(actual, expected, label):
                nonlocal matched
                if label == 'result instrumented':
                    require(expected is False and actual is True and matched == 0,
                            'one explicitly instrumented inference result required')
                    matched += 1
                    return original_exact(actual, True, label)
                return original_exact(actual, expected, label)

            module._exact = checked_exact
            installed = True
            receipt = module.validate_onnx_result(directory, gpu_uuid, export_root,
                expected_warmup=expected_warmup, expected_iterations=expected_iterations,
                reference=None, event_start=event_start)
            require(matched == 1 and type(receipt) is dict and receipt.get('diagnostic_only') is True and
                    receipt.get('performance_qualified') is False and not
                    {'application_instrumented', 'profiler_diagnostic_only', 'base_validator_sha256',
                     'instrumented_adapter'} & set(receipt),
                    'complete instrumented inference receipt required')
            return dict(receipt, application_instrumented=True, profiler_diagnostic_only=True,
                        base_validator_sha256=source_sha256, instrumented_adapter='flag-v1')
        finally:
            if installed:
                module._exact = original_exact
            lock.release()

    return validate


def instrumented_onnx_validator(expected_sha256, *, event_start):
    """Load the frozen original ONNX oracle privately; caller must be caps0."""
    require(type(expected_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', expected_sha256) is not None and
            ONNX_VALIDATOR == ONNX_VALIDATOR.resolve(strict=True), 'fixed inference validator identity required')
    with ONNX_VALIDATOR.open('rb') as stream:
        before = os.fstat(stream.fileno())
        source = stream.read(1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
        current = ONNX_VALIDATOR.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    require(stat.S_ISREG(before.st_mode) and 0 < len(source) <= 1024 * 1024 and
            identity(before) == identity(after) == identity(current) and
            hashlib.sha256(source).hexdigest() == expected_sha256, 'frozen inference validator bytes changed')
    module = ModuleType('glimmer_private_instrumented_onnx_oracle')
    module.__file__ = str(ONNX_VALIDATOR)
    exec(compile(source, str(ONNX_VALIDATOR), 'exec'), module.__dict__)
    return _instrumented_onnx_adapter(module, expected_sha256, event_start)


def validate_directory(outer, directory, uid, serial):
    directory = Path(directory)
    require(type(serial) is int and serial > 0 and directory == directory.resolve(strict=True) and
            directory.parent == outer / 'passes' and directory.name.startswith(f'pass-{serial:04d}-') and
            directory.is_dir() and directory.stat().st_uid == uid and
            stat.S_IMODE(directory.stat().st_mode) == 0o700, 'fresh owned serial replay directory required')
    return directory


def validate_log(descriptor, directory, role, uid):
    require(role in ROLES and type(descriptor) is int and descriptor > 2, 'one fixed role log required')
    path = directory / 'native' / (role + '.log')
    actual, expected = os.fstat(descriptor), path.lstat()
    require(path.parent.resolve(strict=True) == path.parent and stat.S_ISREG(actual.st_mode) and
            stat.S_ISREG(expected.st_mode) and actual.st_uid == uid and actual.st_nlink == 1 and
            actual.st_size == 0 and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino) and
            fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_WRONLY,
            'exact fresh role log descriptor required')


def validate_release(value, command, environment):
    require(type(value) is str and value.endswith('\n') and value.count('\n') == 1,
            'one complete fixed application release required')
    request = bridge.decode(value.encode())
    require(request == dict(command=command, environment=environment), 'native application recipe changed')


class FixedInput:
    def __init__(self, process):
        self.process, self.released, self.closed = process, False, False

    def write(self, value):
        require(not self.closed and not self.released, 'native release is single-use')
        validate_release(value, self.process.args, self.process.environment)
        self.process.connection.call('release', role=self.process.role)
        self.released = True
        return len(value)

    def flush(self):
        require(not self.closed and self.released, 'release must precede flush')

    def close(self):
        self.closed = True


class NativeProcess:
    """Popen subset: only authenticated parent waitpid can report an exit."""
    def __init__(self, connection, role, identity, command, environment):
        bridge.validate_identity(identity)
        self.connection, self.role = connection, role
        self.pid, self.start = identity['pid'], identity['start']
        self.args, self.environment = command, dict(environment)
        self.returncode, self.receipt = None, None
        self.stdin = FixedInput(self)

    def poll(self):
        if self.receipt is None:
            reply = self.connection.call('poll', role=self.role)
            receipt = reply['receipt']
            if receipt is not None:
                expected = bridge.wait_receipt(self.pid, self.start, receipt['raw_status'])
                require(receipt == expected, 'actual exact parent wait receipt required')
                self.receipt, self.returncode = receipt, receipt['returncode']
        return self.returncode

    def wait(self, timeout=None):
        until = self.connection.deadline if timeout is None else min(
            self.connection.deadline, time.monotonic() + timeout)
        while self.poll() is None:
            require(time.monotonic() < until, 'native process wait exceeded bound')
            time.sleep(.02)
        return self.returncode

    def verify(self, start, command, environment):
        require(start == self.start and (command is None or command == self.args) and
                (environment is None or environment == self.environment), 'native verification recipe differs')
        result = self.connection.call('verify', role=self.role, released=command is not None)
        require(result['identity'] == dict(pid=self.pid, start=self.start) and result['verified'] is True,
                'root native authentication failed')


def receive(connection, credentials):
    value, descriptors, actual = bridge.receive_packet(connection)
    if actual != credentials:
        for fd in descriptors:
            os.close(fd)
        raise ValueError('native protocol sender credentials differ')
    return value, descriptors


class Connection:
    """A pinned root supervisor, not an editable readiness file, supplies authority."""
    def __init__(self, outer, endpoint, supervisor, serial, deadline, kind):
        require(re.fullmatch(r'glimmer-native-ncu-[0-9a-f]{32}', endpoint) is not None and
                kind in ('controller', 'bridge') and type(serial) is int and serial >= 0,
                'fixed private native endpoint required')
        bridge.validate_identity(supervisor)
        self.deadline, self.serial, self.supervisor = deadline, serial, supervisor
        self.socket, self.pidfd = None, None
        try:
            self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            self.socket.settimeout(30)
            self.socket.connect('\0' + endpoint)
            peer = struct.unpack('3i', self.socket.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            require(peer == (supervisor['pid'], 0, 0), 'exact root supervisor peer required')
            bridge.live_identity(supervisor, bridge.process_group(os.getpid()))
            command = [os.fsdecode(value) for value in
                       Path(f"/proc/{supervisor['pid']}/cmdline").read_bytes().split(b'\0')[:-1]]
            require(command == ['/usr/bin/python3', '-B', str(ENTRY), '--root-supervisor', str(outer)],
                    'unexpected root supervisor command')
            self.pidfd = os.pidfd_open(supervisor['pid'])
            reply = self.call('hello', kind=kind, identity=bridge.process_identity(os.getpid()))
            if kind == 'bridge':
                require(type(reply['assigned_serial']) is int and reply['assigned_serial'] > 0,
                        'one positive assigned replay serial required')
                self.serial = reply['assigned_serial']
        except BaseException:
            self.close()
            raise

    def call(self, operation, *, descriptors=(), **fields):
        require(time.monotonic() < self.deadline and not bridge.terminal(self.pidfd),
                'native supervisor deadline or lifecycle ended')
        self.socket.settimeout(min(30., self.deadline - time.monotonic()))
        bridge.send_packet(self.socket, dict(op=operation, serial=self.serial, **fields), descriptors)
        reply, fds = receive(self.socket, (self.supervisor['pid'], 0, 0))
        for fd in fds:
            os.close(fd)
        require(not fds and reply.get('op') == operation and reply.get('serial') == self.serial and
                not bridge.terminal(self.pidfd), 'native reply operation or identity differs')
        return reply

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.pidfd is not None:
            os.close(self.pidfd)
            self.pidfd = None


@contextmanager
def native_launches(common, h, connection, directory, manifest):
    """Replace this helper instance only; retain ready/start/numerical semantics."""
    original_subprocess, original_verify, original_stop = common.subprocess, common.verify_process, h.stop_owned
    processes = {}
    commands = application_commands(manifest, directory / 'native')
    environment = manifest['application_environment']
    wrapper = [str(PYTHON), '-B', '-c', h.wrapper]
    connection.call('begin', directory=str(directory))

    def launch(command, **kwargs):
        require(command == wrapper and set(kwargs) ==
                {'cwd', 'stdin', 'stdout', 'stderr', 'text', 'env', 'start_new_session'} and
                kwargs['cwd'] == common.ROOT and kwargs['stdin'] == subprocess.PIPE and
                kwargs['stderr'] == subprocess.STDOUT and kwargs['text'] is True and
                kwargs['env'] == environment and kwargs['start_new_session'] is True,
                'only the exact native wrapper request is allowed')
        role = Path(kwargs['stdout'].name).stem
        require(role in ROLES and role not in processes and
                Path(kwargs['stdout'].name) == directory / 'native' / (role + '.log'), 'native role/log differs')
        reply = connection.call('spawn', role=role, descriptors=(kwargs['stdout'].fileno(),))
        process = NativeProcess(connection, role, reply['identity'], commands[role], environment)
        processes[role] = process
        return process

    def verify(helpers, process, start, command=None, environment=None):
        require(any(process is item for item in processes.values()), 'unknown native process')
        return process.verify(start, command, environment)

    def stop(process, start):
        require(any(process is item for item in processes.values()) and start == process.start,
                'unknown cleanup process')
        if process.poll() is None:
            connection.call('abort')
            raise RuntimeError('live native target at cleanup invalidates replay')

    class SubprocessAdapter:
        Popen = staticmethod(launch)

        def __getattr__(self, name):
            return getattr(original_subprocess, name)

    common.subprocess, common.verify_process, h.stop_owned = SubprocessAdapter(), verify, stop
    try:
        yield processes
    finally:
        common.subprocess, common.verify_process, h.stop_owned = original_subprocess, original_verify, original_stop


def metric_inventory(content, roles, *, selected_roles=None):
    """All selected-family launches, retaining zeros; no cross-launch SM union."""
    require(type(content) is str and len(content) <= 64 * 1024 ** 2 and
            set(roles.values()) == set(ROLES) and len(roles) == 2, 'bounded complete role inventory required')
    if selected_roles is None:
        selected_roles = set(ROLES)
    require(selected_roles in (set(ROLES), {'low'}), 'unsupported selected native roles')
    lines = content.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith('"ID",')]
    require(len(starts) == 1, 'one NCU raw CSV header required')
    reader = csv.DictReader(io.StringIO('\n'.join(lines[starts[0]:])), strict=True)
    identity = ('ID', 'Process ID', 'Kernel Name', 'Context', 'Stream')
    fields = reader.fieldnames or []
    require(len(fields) == len(set(fields)) and set(identity + METRICS) <= set(fields),
            'complete unique native metric fields required')
    units = next(reader, {})
    require(all(units.get(key) == '' for key in identity) and
            [units.get(key) for key in METRICS] == ['SM', '', '', 'cycle', 'inst'], 'native metric units differ')
    result, observed = [], set()
    for row in reader:
        require(None not in row and None not in row.values() and
                all(row[key].isdecimal() for key in ('ID', 'Process ID', 'Context', 'Stream')) and
                re.fullmatch(r'volta_sgemm_[A-Za-z0-9_]+', row['Kernel Name']) is not None,
                'invalid or unexpected native launch')
        pid = int(row['Process ID'])
        key = tuple(int(row[name]) for name in ('Process ID', 'Context', 'Stream', 'ID'))
        require(pid in roles and key not in observed, 'unknown role or duplicate native launch')
        observed.add(key)
        values = {}
        for name in METRICS:
            require(re.fullmatch(r'(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?',
                                 row[name]) is not None, 'invalid native counter spelling')
            whole, _, fraction = row[name].replace(',', '').partition('.')
            require(not fraction or set(fraction) == {'0'}, 'integer native counter required')
            values[name] = int(whole)
        require(values['launch__sm_count'] == 80 and values['launch__grid_size'] > 0 and
                values['launch__block_size'] > 0, 'native SM domain or launch geometry differs')
        result.append(dict(role=roles[pid], process_id=pid, context=key[1], stream=key[2], launch_id=key[3],
                           kernel=row['Kernel Name'], metrics=values,
                           all_80_useful=values[METRICS[3]] > 0 and values[METRICS[4]] > 0))
    require(result and {row['role'] for row in result} == selected_roles,
            'selected native role inventories differ')
    return result


def replay_inventory(content, log, roles_by_pass, *, target_only=False):
    """Bind export to a unique actual replay; do not invent log PID attribution.

    Concurrent Profiling lines carry no PID. Vendor strict/all replay supplies
    per-process/per-device matching; this log check validates whole-pass counts
    and every authenticated process lifetime, not a per-role log inventory.
    """
    require(type(target_only) is bool and '==ERROR==' not in log and roles_by_pass,
            'complete error-free native replay required')
    all_pids = [pid for roles in roles_by_pass for pid in roles]
    require(len(all_pids) == len(set(all_pids)), 'replay PID attribution is ambiguous')
    matches = []
    for serial, roles in enumerate(roles_by_pass, 1):
        try:
            rows = metric_inventory(content, roles, selected_roles={'low'} if target_only else None)
            if target_only:
                require(len(rows) == 1 and rows[0]['kernel'] == TARGET_KERNEL and
                        rows[0]['metrics']['launch__grid_size'] == 576 and
                        rows[0]['metrics']['launch__block_size'] == 256 and rows[0]['all_80_useful'],
                        'one useful full-device native target launch required')
            matches.append((serial, rows))
        except ValueError:
            continue
    require(len(matches) == 1, 'CSV must identify exactly one complete authenticated replay')
    exported, rows = matches[0]
    connected, disconnected, profiles, active = [], [], {}, set()
    by_pid = {pid: serial for serial, roles in enumerate(roles_by_pass, 1) for pid in roles}
    for line in log.splitlines():
        match = re.fullmatch(r'==PROF== Connected to process ([1-9][0-9]*) \(.+\)', line)
        if match:
            pid = int(match[1])
            require(pid in by_pid and pid not in connected, 'extra or duplicate native NCU process')
            require(not active or all(by_pid[item] == by_pid[pid] for item in active),
                    'native replay process lifetimes overlap sessions')
            connected.append(pid)
            active.add(pid)
            continue
        match = re.fullmatch(r'==PROF== Disconnected from process ([1-9][0-9]*)', line)
        if match:
            pid = int(match[1])
            require(pid in active and pid not in disconnected, 'unknown or duplicate native disconnect')
            active.remove(pid)
            disconnected.append(pid)
            continue
        match = re.fullmatch(r'==PROF== Profiling "([^"]+)": Application replay pass ([1-9][0-9]*)', line)
        if match:
            serial = int(match[2])
            require(active and all(by_pid[pid] == serial for pid in active), 'native profile outside its replay')
            if target_only:
                require(1 <= serial <= len(roles_by_pass), 'unknown target replay serial')
                lows = [pid for pid, role in roles_by_pass[serial - 1].items() if role == 'low']
                require(len(lows) == 1 and lows[0] in active,
                        'native target profile without its authenticated low process')
            profiles.setdefault(serial, Counter()).update([match[1]])
        elif line.startswith(('==PROF== Connected', '==PROF== Disconnected', '==PROF== Profiling')):
            raise ValueError('malformed native profiler lifecycle line')
    require(not active and Counter(connected) == Counter(disconnected) == Counter(all_pids),
            'native profiler process lifecycle incomplete')
    expected = Counter(row['kernel'] for row in rows)
    require(set(profiles) == set(range(1, len(roles_by_pass) + 1)) and
            all(value == expected for value in profiles.values()), 'complete native strict replay inventories differ')
    return exported, rows


def target_call_receipt(content, pid, provider):
    """Validate the complete fixed shim receipt; this does not prove phase or counters."""
    require(type(content) is str and len(content) <= 8 * 1024 ** 2 and
            type(pid) is int and pid > 0 and type(provider) is str and provider.startswith('/'),
            'bounded target receipt identity required')
    begin_pattern = re.compile(r'GLIMMER_NATIVE_SGEMM_BEGIN pid=(\d+) tid=(\d+) ordinal=(\d+) '
        r'ta=(\d+) tb=(\d+) m=(\d+) n=(\d+) k=(\d+) lda=(\d+) ldb=(\d+) ldc=(\d+) selected=(\d+)')
    end_pattern = re.compile(r'GLIMMER_NATIVE_SGEMM_END pid=(\d+) ordinal=(\d+) status=(\d+)')
    range_pattern = re.compile(r'GLIMMER_NATIVE_SGEMM_RANGE pid=(\d+) ordinal=(\d+) domain=(\d+) '
        r'push=(-?\d+) pop=(-?\d+) name=(\S+)')
    records, endings, ranges, resolved = [], [], [], []
    pending = None
    for line in content.splitlines():
        if not line.startswith('GLIMMER_NATIVE_SGEMM_'):
            continue
        if line.startswith('GLIMMER_NATIVE_SGEMM_RESOLVE '):
            require(not records and not resolved and pending is None,
                    'native provider resolution is not first and unique')
            resolved.append(line.removeprefix('GLIMMER_NATIVE_SGEMM_RESOLVE path='))
            continue
        match = begin_pattern.fullmatch(line)
        if match:
            require(resolved and pending is None, 'overlapping native SGEMM calls')
            pending = tuple(map(int, match.groups()))
            records.append(pending)
            continue
        match = end_pattern.fullmatch(line)
        if match:
            end = tuple(map(int, match.groups()))
            require(pending is not None and end[:2] == (pending[0], pending[2]),
                    'native SGEMM end does not match active call')
            endings.append(end)
            pending = None
            continue
        match = range_pattern.fullmatch(line)
        if match:
            require(pending is not None and pending[2] == TARGET_ORDINAL and pending[-1] == 1,
                    'native range outside selected call')
            ranges.append(match.groups())
            continue
        raise ValueError('malformed native SGEMM receipt line')
    count = 3 * TARGET_CALLS_PER_ITERATION
    require(pending is None and resolved == [provider] and len(records) == len(endings) == count and
            [row[2] for row in records] == list(range(1, count + 1)) and
            all(row[0] == pid and row[1] > 0 for row in records) and
            len({row[1] for row in records}) == 1 and
            endings == [(pid, ordinal, 0) for ordinal in range(1, count + 1)],
            'incomplete, reordered or unsuccessful native SGEMM receipt')
    descriptors = [row[3:11] for row in records]
    selected = [row for row in records if row[-1] == 1]
    require(all(row[-1] in (0, 1) for row in records) and
            all(row[3] in (0, 1, 2) and row[4] in (0, 1, 2) and
                all(0 < value <= 0x7fffffff for value in row[5:11]) for row in records) and
            descriptors[:TARGET_CALLS_PER_ITERATION] ==
                descriptors[TARGET_CALLS_PER_ITERATION:2 * TARGET_CALLS_PER_ITERATION] ==
                descriptors[2 * TARGET_CALLS_PER_ITERATION:] and
            descriptors.count(TARGET_DESCRIPTOR) == 36 and
            len(selected) == 1 and selected[0][2] == TARGET_ORDINAL and
            selected[0][3:11] == TARGET_DESCRIPTOR and len(ranges) == 1 and
            ranges[0][:3] == (str(pid), str(TARGET_ORDINAL), '1') and
            ranges[0][5] == f'GlimmerNativeSgemm149-pid-{pid}',
            'native target descriptor, sequence or range differs')
    return dict(pid=pid, caller_tid=records[0][1], ordinal=TARGET_ORDINAL,
                descriptor=list(TARGET_DESCRIPTOR), calls=count, provider=provider,
                descriptor_sequence=[list(value) for value in descriptors],
                range_name=ranges[0][5], phase_verified=False, counter_verified=False)


def target_profiler_command(outer, bridge_command):
    """Construct fixed profiler selection; caller authenticates the bridge and paths."""
    require(isinstance(outer, Path) and outer.is_absolute() and type(bridge_command) is list and
            bridge_command and all(type(value) is str and value for value in bridge_command),
            'fixed absolute profiler output and bridge command required')
    return [NCU_BINARY, '--config-file', '0', '--target-processes', 'all', '--nvtx',
            '--profile-from-start', 'off', '--nvtx-include', TARGET_NVTX_FILTER,
            '--kernel-name-base', 'function', '--kernel-name', KERNEL_FILTER,
            '--replay-mode', 'application', '--app-replay-mode', 'strict', '--app-replay-match', 'all',
            '--kill', 'no', '--clock-control', 'none', '--cache-control', 'none',
            '--metrics', ','.join(METRICS), '--log-file', str(outer / 'ncu.log'),
            '--export', str(outer / 'capture'), *bridge_command]


def target_evidence(content, log, roles_by_pass, *, shim_logs, provider, reference_descriptors,
                    oracles, actual_command, outer, bridge_command, manifest,
                    native_directories, application_commands_by_pass):
    """Join narrow observations; caller still authenticates all inputs and processes."""
    expected_oracle = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                           source_bound_complete_inference=True, first_command_handoff=False)
    require(type(roles_by_pass) is list and len(roles_by_pass) == 3 and
            type(shim_logs) is list and len(shim_logs) == len(roles_by_pass) and
            type(oracles) is list and oracles == [expected_oracle] * len(roles_by_pass) and
            type(reference_descriptors) is list and len(reference_descriptors) == 438 and
            type(native_directories) is list and len(native_directories) == len(roles_by_pass) and
            type(application_commands_by_pass) is list and
            application_commands_by_pass == [target_application_commands(manifest, directory)
                                             for directory in native_directories] and
            actual_command == target_profiler_command(outer, bridge_command),
            'target replay, oracle, descriptor baseline or profiler command differs')
    receipts = []
    for roles, text in zip(roles_by_pass, shim_logs):
        require(type(roles) is dict and len(roles) == 2 and set(roles.values()) == set(ROLES),
                'two authenticated native roles required in every pass')
        low = next(pid for pid, role in roles.items() if role == 'low')
        receipt = target_call_receipt(text, low, provider)
        require(receipt['descriptor_sequence'] == reference_descriptors,
                'native replay descriptor sequence differs from reviewed reference')
        receipts.append(receipt)
    serial, rows = replay_inventory(content, log, roles_by_pass, target_only=True)
    require(rows[0]['process_id'] == receipts[serial - 1]['pid'], 'exported target PID differs')
    return dict(status='target_observation_requires_independent_review', exported_replay_serial=serial,
                target=rows[0], calls=receipts, phase_verified=False, counter_verified=False,
                application_phase_verified=False, native_spatial_participation_verified=False,
                full_device_qualified=False, performance_qualified=False, priority_qualified=False,
                saturation_qualified=False)
