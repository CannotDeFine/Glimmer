#!/usr/bin/env python3
"""Private NCU replay launch bridge; standard library only, never a workload runner.

The privileged phase accepts one fixed service recipe from an authenticated
supervisor. The service child and the parent both immediately exec setpriv.
The unprivileged parent owns waitpid and waits for the supervisor's full-pass
acceptance before allowing NCU to replay. There is no general command endpoint.
"""

import argparse
import array
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import socket
import stat
import struct
import sys
import time


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parent.parent
SERVICE = PROJECT / 'build/debug/bin/glimmer_execution_service'
SUPERVISOR = PROJECT / 'scripts/run_bert_sgemm_ncu_capture.py'
PYTHON = Path('/usr/bin/python3')
SETPRIV = Path('/usr/bin/setpriv')
ENV = Path('/usr/bin/env')
RANGE_LIBRARY = Path('/usr/local/cuda/lib64/libnvToolsExt.so.1')
NCU_DIRECTORY = Path('/opt/nvidia/nsight-compute/2024.3.2/target/linux-desktop-glibc_2_11_3-x64')
MESSAGE_MAX = 65536
SNAPSHOT_SEALS = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
TARGET_PRELOAD_MAP_NAME = '/memfd:glimmer-libnative_sgemm_range.so (deleted)'
CURRENT_CODE_FD = globals().get('_SEALED_SELF_FD')
SEALED_LOADER = '''import fcntl,hashlib,os,sys
origin,source,expected=sys.argv[1:4]
fd=os.open(source,os.O_RDONLY)
required=fcntl.F_SEAL_SEAL|fcntl.F_SEAL_SHRINK|fcntl.F_SEAL_GROW|fcntl.F_SEAL_WRITE
if fcntl.fcntl(fd,fcntl.F_GET_SEALS)&required!=required: raise ValueError("unsealed helper")
with os.fdopen(os.dup(fd),"rb") as stream: data=stream.read(262145)
if len(data)>262144 or hashlib.sha256(data).hexdigest()!=expected: raise ValueError("helper identity")
os.set_inheritable(fd,True)
sys.argv=[origin,*sys.argv[4:]]
exec(compile(data,origin,"exec"),{"__name__":"__main__","__file__":origin,"_SEALED_SELF_FD":fd})
'''
CAP_SYS_ADMIN = 1 << 21
CAP_FIELDS = ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')
ENV_KEYS = frozenset(('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'LANG', 'LC_ALL',
                      'LD_LIBRARY_PATH', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                      'OPENBLAS_NUM_THREADS', 'PATH', 'PYTHONNOUSERSITE',
                      'TOKENIZERS_PARALLELISM', 'TRANSFORMERS_OFFLINE'))


def require(value, message):
    if not value:
        raise ValueError(message)


def positive(value):
    return type(value) is int and 0 < value <= (1 << 63) - 1


def process_identity(pid):
    require(positive(pid), 'positive exact PID required')
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    require(len(fields) >= 20 and fields[19].isdecimal(), 'invalid process start identity')
    return dict(pid=pid, start=int(fields[19]))


def process_group(pid):
    lines = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    groups = [line[3:] for line in lines if line.startswith('0::')]
    require(len(groups) == 1 and groups[0].startswith('/'), 'one unified task cgroup required')
    return groups[0]


def validate_identity(record):
    require(type(record) is dict and set(record) == {'pid', 'start'} and
            all(positive(value) for value in record.values()), 'exact positive PID/start required')
    return record


def live_identity(record, group):
    validate_identity(record)
    require(process_identity(record['pid']) == record and process_group(record['pid']) == group,
            'process incarnation or task cgroup differs')


def terminal(descriptor):
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    return bool(poller.poll(0))


def seal(path):
    """Hash one opened regular file without trusting a replaced path during hashing."""
    path = Path(path)
    require(path.is_absolute(), 'absolute sealed path required')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), 'sealed input must be regular')
        digest = hashlib.sha256()
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
        after = os.fstat(stream.fileno())
        current = path.stat()
    fields = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    require(fields(before) == fields(after) == fields(current), 'sealed file changed while hashing')
    return digest.hexdigest()


def sealed_snapshot(path, expected):
    """Copy verified bytes into a kernel-sealed object, immune to later path writes."""
    descriptor = os.memfd_create('glimmer-' + Path(path).name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        digest = hashlib.sha256()
        with Path(path).open('rb') as source, os.fdopen(os.dup(descriptor), 'wb') as destination:
            metadata = os.fstat(source.fileno())
            require(stat.S_ISREG(metadata.st_mode) and 0 < metadata.st_size <= 1024 ** 3,
                    'bounded regular snapshot source required')
            for data in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(data)
                destination.write(data)
        require(digest.hexdigest() == expected, 'snapshot bytes differ from sealed input')
        os.fchmod(descriptor, 0o555)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, SNAPSHOT_SEALS)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def validate_snapshot(descriptor, expected):
    require(type(descriptor) is int and descriptor > 2 and
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & SNAPSHOT_SEALS == SNAPSHOT_SEALS and
            seal(Path(f'/proc/self/fd/{descriptor}')) == expected, 'exact immutable snapshot required')


def validate_target_preload_mapping(descriptor, expected, identity, group, target_descriptor):
    """Check one sealed diagnostic preload; caller authenticates target ownership and privilege."""
    validate_identity(identity)
    require(type(group) is str and group.startswith('/') and type(target_descriptor) is int and
            target_descriptor > 2, 'exact target group and inherited descriptor required')
    live_identity(identity, group)
    validate_snapshot(descriptor, expected)
    source = os.fstat(descriptor)
    require(stat.S_ISREG(source.st_mode) and stat.S_IMODE(source.st_mode) == 0o555 and
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == SNAPSHOT_SEALS,
            'sealed preload mode or seal set differs')
    name = TARGET_PRELOAD_MAP_NAME
    target_path = Path(f"/proc/{identity['pid']}/fd/{target_descriptor}")

    def check_target_descriptor():
        require(os.readlink(target_path) == name, 'target preload descriptor name differs')
        duplicate = os.open(target_path, os.O_RDONLY | os.O_CLOEXEC)
        try:
            target = os.fstat(duplicate)
            require((target.st_dev, target.st_ino, target.st_size) ==
                    (source.st_dev, source.st_ino, source.st_size) and
                    stat.S_IMODE(target.st_mode) == 0o555 and
                    fcntl.fcntl(duplicate, fcntl.F_GET_SEALS) == SNAPSHOT_SEALS,
                    'target preload descriptor is not the sealed parent snapshot')
        finally:
            os.close(duplicate)

    check_target_descriptor()
    matches = []
    for line in Path(f"/proc/{identity['pid']}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        require(len(fields) >= 5, 'malformed target memory map')
        device = fields[3].split(':')
        require(len(device) == 2 and fields[4].isdecimal(), 'malformed target mapping identity')
        if ((int(device[0], 16), int(device[1], 16), int(fields[4])) !=
                (os.major(source.st_dev), os.minor(source.st_dev), source.st_ino)):
            continue
        require(len(fields) == 6 and fields[5] == name and not
                ('w' in fields[1] and 'x' in fields[1]), 'unexpected sealed mapping name or permissions')
        matches.append(fields[1])
    require(matches and any('x' in permissions for permissions in matches),
            'executable sealed preload mapping missing')
    check_target_descriptor()
    live_identity(identity, group)
    validate_snapshot(descriptor, expected)
    final = os.fstat(descriptor)
    require((final.st_dev, final.st_ino, final.st_size, stat.S_IMODE(final.st_mode)) ==
            (source.st_dev, source.st_ino, source.st_size, 0o555) and
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == SNAPSHOT_SEALS,
            'sealed parent preload changed during inspection')
    return dict(pid=identity['pid'], start=identity['start'], target_descriptor=target_descriptor,
                device=source.st_dev, inode=source.st_ino, bytes=source.st_size,
                sha256=expected, mode=stat.S_IMODE(final.st_mode), seals=SNAPSHOT_SEALS,
                mapping_count=len(matches), mapping_permissions=matches)


def sealed_python_command(source, expected, *arguments):
    return [str(PYTHON), '-I', '-S', '-B', '-c', SEALED_LOADER,
            str(SCRIPT), str(source), expected, *arguments]


def protected_system_file(path):
    """Installed privileged loader inputs must not be replaceable by the task user."""
    resolved = Path(path).resolve(strict=True)
    for entry in (resolved, *resolved.parents):
        metadata = entry.stat()
        require(metadata.st_uid == 0 and metadata.st_mode & 0o022 == 0,
                'system loader path is not root-owned and protected')
    require(resolved.is_file(), 'system loader input is not regular')


def frozen_template(outer, expected):
    require(type(expected) is str and re.fullmatch(r'[0-9a-f]{64}', expected) is not None,
            'frozen manifest digest required')
    path = outer / 'manifest.json'
    owner = path.lstat()
    data = _owned_bounded_file(path, owner.st_uid, owner.st_gid, MESSAGE_MAX)
    require(hashlib.sha256(data).hexdigest() == expected, 'frozen manifest differs')
    template = decode(data)
    exact_fields(template['service_environment'], ENV_KEYS)
    return template


def profiler_environment(inherited, template, outer):
    """Only the frozen launcher base and observed, pinned NCU additions may survive."""
    base = template['ncu_supervisor_environment']
    expected_base = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                         TMPDIR=str(outer / 'ncu-tmp'), CUDA_CACHE_DISABLE='1',
                         CUDA_DEVICE_ORDER='PCI_BUS_ID',
                         CUDA_VISIBLE_DEVICES=template['service_environment']['CUDA_VISIBLE_DEVICES'])
    require(base == expected_base and all(inherited.get(key) == value for key, value in base.items()),
            'frozen profiler base environment differs')
    static = {'NVIDIA-PROCESS-TRACKING-FILTER': 'exclude-tree:nvidia-modprobe',
              'NVIDIA_PROCESS_INJECTION_CRASH_REPORTING': '0',
              'NVIDIA_PROCESS_INJECTION_XML_TARGET_SETTINGS': '0',
              'NV_CUDA_START_SUSPENDED': '1', 'NV_NSIGHT_INJECTION_PORT_BASE': '49152',
              'NV_NSIGHT_INJECTION_PORT_RANGE_BEGIN': '0', 'NV_NSIGHT_INJECTION_PORT_RANGE_END': '63',
              'NV_NSIGHT_INJECTION_TRANSPORT_TYPE': 'uds', 'NV_TPS_LAUNCH_TOKEN': 'Interactive Profile',
              'NV_COMPUTE_PROFILER_DISABLE_FUNCTION_LAZY_LOADING': '1',
              'NVTX_INJECTION64_PATH': 'libcuda-injection.so', 'PWD': str(outer)}
    dynamic = {'LD_PRELOAD', 'LD_LIBRARY_PATH', 'NV_COMPUTE_PROFILER_PERFWORKS_DIR',
               'NVIDIA-PROCESS-TRACKING-CONFIGURATION'}
    unexpected = sorted(set(inherited) - (set(base) | set(static) | dynamic))
    mismatched = sorted(key for key, value in static.items() if inherited.get(key, value) != value)
    require(not unexpected and not mismatched,
            'unexpected inherited profiler variable or value: unexpected=' +
            repr([key[:96] for key in unexpected[:8]]) + '; mismatched=' + repr(mismatched))
    require(inherited.get('NVTX_INJECTION64_PATH') == 'libcuda-injection.so', 'missing NVTX injection')
    for key in ('LD_LIBRARY_PATH', 'NV_COMPUTE_PROFILER_PERFWORKS_DIR'):
        require(type(inherited.get(key)) is str and Path(inherited[key]).resolve(strict=True) ==
                NCU_DIRECTORY.resolve(strict=True), 'unexpected profiler library search path')
    seals = template['ncu_injection_seals']
    require(type(seals) is dict and seals, 'pinned injection libraries required')
    for path, expected in seals.items():
        require(Path(path).parent == NCU_DIRECTORY.resolve(strict=True) and
                Path(path).name in ('libcuda-injection.so', 'libInterceptorInjectionTarget.so',
                                   'libTreeLauncherTargetInjection.so',
                                   'libTreeLauncherTargetUpdatePreloadInjection.so'),
                'unexpected injection library')
        protected_system_file(path)
        require(seal(path) == expected, 'injection library seal differs')
    preloads = [value for value in re.split(r'[ :]+', inherited.get('LD_PRELOAD', '')) if value]
    require(all(str(Path(path).resolve(strict=True)) in seals for path in preloads),
            'unsealed inherited loader injection')
    tracking = inherited.get('NVIDIA-PROCESS-TRACKING-CONFIGURATION', '')
    require(type(tracking) is str and 0 < len(tracking) <= 16384 and '\0' not in tracking,
            'bounded NCU process tracking required')
    # Loader paths embedded in the authenticated NCU tracker must name only pinned tools.
    for line in tracking.splitlines():
        if '-preload-prepend:' in line:
            paths = line.split(': ', 1)[1].split(':')
            require(all(str(Path(path).resolve(strict=True)) in seals for path in paths),
                    'tracker contains an unsealed preload')
    return dict(inherited)


def validate_status(content, uid, gid, capability):
    require(positive(uid) and type(gid) is int and gid >= 0 and
            type(capability) is bool, 'non-root explicit capability policy required')
    fields = {}
    for line in content.splitlines():
        key, separator, value = line.partition(':')
        if separator:
            require(key not in fields, 'duplicate process status field')
            fields[key] = value.strip()
    require(fields.get('Uid', '').split() == [str(uid)] * 4 and
            fields.get('Gid', '').split() == [str(gid)] * 4 and
            fields.get('Groups') == '' and fields.get('NoNewPrivs') == '1',
            'actual UID/GID/groups/no-new-privs differ')
    expected = CAP_SYS_ADMIN if capability else 0
    for key in CAP_FIELDS:
        require(re.fullmatch(r'[0-9a-fA-F]{16}', fields.get(key, '')) is not None and
                int(fields[key], 16) == expected, 'actual capability set differs: ' + key)


def drop_command(uid, gid, *, service):
    require(positive(uid) and uid < 2 ** 32 and type(gid) is int and 0 <= gid < 2 ** 32 and
            type(service) is bool, 'explicit non-root privilege drop required')
    caps = '-all,+sys_admin' if service else '-all'
    return [str(SETPRIV), '--reuid=' + str(uid), '--regid=' + str(gid), '--clear-groups',
            '--inh-caps=' + caps, '--ambient-caps=' + caps, '--bounding-set=' + caps,
            '--no-new-privs']


def supervisor_command(outer):
    return [str(PYTHON), '-B', str(SUPERVISOR), '--root-supervisor', str(outer)]


def authenticate_supervisor(connection, expected, outer):
    """The root-created channel, not a file or a nonce alone, supplies authority."""
    validate_identity(expected)
    pid, uid, gid = struct.unpack('3i', connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i')))
    require((pid, uid, gid) == (expected['pid'], 0, 0), 'unexpected supervisor credentials')
    group = process_group(os.getpid())
    require(group.endswith('/glimmer-bert-' + outer.name + '-equal.service'), 'wrong task unit')
    live_identity(expected, group)
    actual = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]
    require(actual == [os.fsencode(value) for value in supervisor_command(outer)],
            'unexpected privileged supervisor command')
    if os.geteuid() == 0:
        executable = Path(f'/proc/{pid}/exe')
        require(executable.samefile(PYTHON) and seal(executable) == seal(PYTHON),
                'supervisor Python executable differs')
    # The post-exec waiter retains the same connected FD and exact root peer.
    # Its unprivileged identity cannot dereference the root peer's /proc/exe.
    descriptor = os.pidfd_open(pid)
    try:
        live_identity(expected, group)
        require(not terminal(descriptor), 'supervisor has terminated')
    except BaseException:
        os.close(descriptor)
        raise
    return group, descriptor


def encode(value):
    payload = json.dumps(value, allow_nan=False, separators=(',', ':')).encode()
    require(0 < len(payload) <= MESSAGE_MAX, 'bounded protocol object required')
    return payload


def decode(payload):
    def unique(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), 'duplicate protocol key')
        return result

    def reject_constant(value):
        raise ValueError('non-finite JSON: ' + value)

    require(0 < len(payload) <= MESSAGE_MAX, 'bounded protocol object required')
    value = json.loads(payload, object_pairs_hook=unique, parse_constant=reject_constant)
    require(type(value) is dict, 'protocol object required')
    return value


def send_packet(connection, value, descriptors=()):
    payload = encode(value)
    ancillary = ([(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', descriptors))]
                 if descriptors else [])
    require(connection.sendmsg([payload], ancillary) == len(payload), 'short protocol packet')


def receive_packet(connection):
    """Return the message, transferred FDs and kernel-supplied sender credentials."""
    descriptors, credentials, invalid_ancillary = [], None, False
    try:
        payload, ancillary, flags, _ = connection.recvmsg(
            MESSAGE_MAX, socket.CMSG_SPACE(4 * array.array('i').itemsize) +
            socket.CMSG_SPACE(struct.calcsize('3i')), socket.MSG_CMSG_CLOEXEC)
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array('i')
                values.frombytes(data)
                descriptors.extend(values)
            elif (level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS and
                  credentials is None and len(data) == struct.calcsize('3i')):
                credentials = struct.unpack('3i', data)
            else:
                invalid_ancillary = True
        require(not invalid_ancillary, 'unexpected ancillary message')
        require(not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC), 'truncated protocol packet')
        require(payload, 'protocol peer closed before completion')
        return decode(payload), descriptors, credentials
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def exact_fields(value, fields):
    require(type(value) is dict and set(value) == set(fields), 'protocol fields differ')


def validate_launch(plan, outer, group, log_fd):
    """Validate an exact sealed template plus minimal dynamic per-pass identities."""
    require(type(plan) is dict and 'template_sha256' in plan, 'frozen launch template required')
    template = frozen_template(outer, plan['template_sha256'])
    fresh = 'fresh_runtime' in template
    fields = ('op', 'serial', 'uid', 'gid', 'directory', 'socket', 'clients',
              'environment', 'seals', 'deadline_ns', 'template_sha256')
    exact_fields(plan, (*fields, 'fresh_claim_sha256') if fresh else fields)
    require(plan['op'] == 'launch' and positive(plan['serial']) and
            positive(plan['deadline_ns']) and
            time.monotonic_ns() < plan['deadline_ns'] <= time.monotonic_ns() + 800_000_000_000,
            'invalid or expired launch request')
    uid, gid = plan['uid'], plan['gid']
    drop_command(uid, gid, service=True)
    require(outer == outer.resolve(strict=True) and outer.is_dir() and outer.stat().st_uid == uid,
            'owned canonical attempt required')
    directory = Path(plan['directory'])
    require(directory == directory.resolve(strict=True) and directory.is_dir() and
            directory.parent == outer / 'passes' and directory.name.startswith('pass-') and
            directory.stat().st_uid == uid, 'fresh owned replay directory required')
    path = Path(plan['socket'])
    parent = path.parent
    require(path.name == 'service.sock' and not path.exists() and not path.is_symlink() and
            parent == parent.resolve(strict=True) and parent.is_dir() and
            parent.parent == outer / 'ncu-tmp' and parent.name.startswith('glimmer-bert-co-') and
            parent.stat().st_uid == uid and parent.stat().st_mode & 0o777 == 0o700 and
            len(os.fsencode(path)) < 108, 'invalid private service socket destination')
    log = directory / 'equal/server.log'
    actual, expected = os.fstat(log_fd), log.lstat()
    require(log.parent == log.parent.resolve(strict=True) and stat.S_ISREG(actual.st_mode) and
            stat.S_ISREG(expected.st_mode) and actual.st_uid == uid and actual.st_nlink == 1 and
            actual.st_size == 0 and (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino) and
            fcntl.fcntl(log_fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_WRONLY,
            'fresh O_EXCL service log descriptor required')
    clients = plan['clients']
    exact_fields(clients, ('low', 'high'))
    require(clients['low'] != clients['high'], 'distinct fresh application identities required')
    for record in clients.values():
        live_identity(record, group)
        validate_status(Path(f"/proc/{record['pid']}/status").read_text(), uid, gid, False)
        require(os.getpgid(record['pid']) == os.getsid(record['pid']) == record['pid'],
                'application must have its own process session')
    environment = plan['environment']
    require(environment == template['service_environment'] and
            (uid, gid) == (template['uid'], template['gid']), 'service differs from frozen template')
    exact_fields(environment, ENV_KEYS)
    require(all(type(value) is str and '\0' not in value for value in environment.values()) and
            re.fullmatch(r'GPU-[0-9a-f-]{36}', environment['CUDA_VISIBLE_DEVICES']) is not None and
            environment['PATH'] == '/usr/bin:/bin', 'invalid fixed service environment')
    authority = authenticate_fresh_replay(plan, outer, template) if fresh else None
    service = selected_service(template, fresh_authority=authority) if fresh else selected_service(template)
    paths = (SCRIPT, service, PYTHON, SETPRIV, ENV, RANGE_LIBRARY.resolve(strict=True))
    expected_seals = plan['seals']
    exact_fields(expected_seals, (str(path) for path in paths))
    require(all(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None
                for value in expected_seals.values()) and
            all(seal(path) == expected_seals[str(path)] for path in paths), 'launch input seal differs')
    for path in (PYTHON, SETPRIV, ENV, RANGE_LIBRARY):
        protected_system_file(path)
    return (directory, authority) if fresh else directory


def selected_service(template, *, fresh_authority=None):
    """Allow only a manifest-bound immutable compiler variant or the legacy service."""
    if 'fresh_runtime' in template:
        require(fresh_authority is not None, 'fresh runtime service authentication is not implemented')
        require(os.geteuid() == 0 and os.getegid() == 0, 'root-only authenticated fresh service selection required')
        exact_fields(fresh_authority, ('claim_sha256', 'manifest_sha256', 'binding_sha256', 'binding', 'template'))
        binding = selected_signal_binding(template)
        require(fresh_authority['template'] == template and fresh_authority['binding'] == binding and
                binding['contract']['target'] == template['sgemm_target'] == '2:3127' and
                binding['contract']['service_environment'] == template['service_environment'] and
                template['service_cap_sys_admin'] is True, 'authenticated fresh service context differs')
        product = binding['contract']['runtime']['service']
        exact_fields(product, ('path', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'mode', 'uid', 'gid', 'sha256'))
        path = Path(product['path'])
        require(path.is_absolute() and path == path.resolve(strict=True) and
                type(product['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', product['sha256']) is not None and
                template['bridge_seals'].get(str(path)) == product['sha256'] and
                (product['uid'], product['gid']) == (template['uid'], template['gid']) and
                type(product['uid']) is int and product['uid'] > 0 and
                type(product['gid']) is int and product['gid'] >= 0 and
                type(product['mode']) is int and product['mode'] == 0o775,
                'authenticated fresh service product differs')
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                 stat.S_IMODE(info.st_mode), info.st_uid, info.st_gid) ==
                (product['device'], product['inode'], product['size'], product['mtime_ns'], product['ctime_ns'],
                 product['mode'], product['uid'], product['gid']), 'fresh service metadata changed')
        return path
    require(fresh_authority is None, 'legacy service cannot accept fresh authority')
    if 'current_sdk' in template:
        require('host_pair' not in template, 'only one runtime selection is allowed')
        binding = template['current_sdk']
        manifest = Path(binding['runtime_manifest'])
        require(manifest == manifest.resolve(strict=True) and
                manifest.parent.parent == PROJECT / 'artifacts/archives/execution-current-sdk' and
                re.fullmatch(r'attempt-[a-z0-9_]+', manifest.parent.name) is not None and
                manifest.name == 'runtime.json' and seal(manifest) == binding['runtime_manifest_sha256'],
                'exact immutable current SDK runtime manifest required')
        runtime = decode(manifest.read_bytes())
        variant = binding['variant']
        require(variant in ('scalar', 'sharded'), 'known current SDK signal variant required')
        product = runtime['variants'][variant]['glimmer_execution_service']
        service = manifest.parent / variant / 'glimmer_execution_service'
        metadata = service.stat()
        require(service == service.resolve(strict=True) and str(service) == binding['server'] == product['path'] and
                stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == product['mode'] == 0o555 and
                metadata.st_uid == template['uid'] and
                (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_ctime_ns) ==
                (product['device'], product['inode'], product['size'], product['ctime_ns']) and
                seal(service) == product['sha256'], 'immutable selected current SDK service identity changed')
        return service
    if 'host_pair' not in template:
        return SERVICE
    binding = template['host_pair']
    manifest = Path(binding['runtime_manifest'])
    require(manifest == manifest.resolve(strict=True) and
            manifest.parent.parent == PROJECT / 'artifacts/archives/execution-host-pair' and
            re.fullmatch(r'attempt-[a-z0-9_]+', manifest.parent.name) is not None and
            manifest.name == 'runtime.json' and seal(manifest) == binding['runtime_manifest_sha256'],
            'exact immutable host runtime manifest required')
    runtime = decode(manifest.read_bytes())
    variant = binding['variant']
    require(variant in ('o0', 'o2'), 'known host compiler variant required')
    product = runtime['variants'][variant]['glimmer_execution_service']
    service = manifest.parent / variant / 'glimmer_execution_service'
    metadata = service.stat()
    require(service == service.resolve(strict=True) and str(service) == binding['server'] == product['path'] and
            stat.S_ISREG(metadata.st_mode) and stat.S_IMODE(metadata.st_mode) == product['mode'] == 0o555 and
            metadata.st_uid == template['uid'] and
            (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_ctime_ns) ==
            (product['device'], product['inode'], product['size'], product['ctime_ns']) and
            seal(service) == product['sha256'], 'immutable selected service identity changed')
    return service


def _owned_bounded_file(path, uid, gid, limit, *, mode=None):
    """Read a private task input through one no-follow FD and check its identity."""
    path = Path(path)
    require(path.is_absolute() and path == path.resolve(strict=True), 'canonical task input required')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and (before.st_uid, before.st_gid, before.st_nlink) == (uid, gid, 1) and
                0 < before.st_size <= limit and (mode is None or stat.S_IMODE(before.st_mode) == mode),
                'owned bounded task input required')
        chunks, remaining = [], before.st_size
        while remaining:
            data = os.read(descriptor, min(remaining, 1024 * 1024))
            require(data, 'task input ended early')
            chunks.append(data)
            remaining -= len(data)
        after, current = os.fstat(descriptor), path.lstat()
        def identity(info):
            return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
                    info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        require(not os.read(descriptor, 1) and identity(before) == identity(after) == identity(current),
                'task input changed during reading')
        return b''.join(chunks)
    finally:
        os.close(descriptor)


def verify_fresh_claim(outer, claim_sha256, uid, gid, *, minimum_remaining=0):
    """Validate a reviewed one-shot claim, without granting launch authority.

    The caller must authenticate uid/gid and the claim digest independently.
    A later root launch must still bind this result to each replay and its
    sealed service snapshot; this function does not open the V2 launch guard.
    """
    outer = Path(outer)
    require(type(uid) is int and uid > 0 and type(gid) is int and gid >= 0 and
            type(claim_sha256) is str and re.fullmatch(r'[0-9a-f]{64}', claim_sha256) is not None and
            type(minimum_remaining) is int and minimum_remaining >= 0 and
            outer.is_absolute() and outer == outer.resolve(strict=True) and
            outer.parent == PROJECT / 'artifacts/profiles/ncu' and
            re.fullmatch(r'attempt-[A-Za-z0-9_]+', outer.name) is not None,
            'exact fresh claim authority and attempt required')
    metadata = outer.lstat()
    require(stat.S_ISDIR(metadata.st_mode) and (metadata.st_uid, metadata.st_gid) == (uid, gid) and
            stat.S_IMODE(metadata.st_mode) == 0o700, 'private fresh attempt required')
    manifest_bytes = _owned_bounded_file(outer / 'manifest.json', uid, gid, MESSAGE_MAX)
    template = decode(manifest_bytes)
    binding = selected_signal_binding(template)
    require(binding is not None and (template.get('uid'), template.get('gid')) == (uid, gid) and
            template.get('service_cap_sys_admin') is True, 'fresh non-root service template required')
    exact_fields(binding, ('schema', 'variant', 'contract', 'plan', 'plan_sha256', 'review', 'review_sha256'))
    contract = binding['contract']
    specialization = capture_specialization(binding)
    fields = ('schema', 'variant', 'target', 'max_seconds', 'deadline_utc', 'runtime',
              'service_environment', 'numerical', 'reference')
    exact_fields(contract, (*fields, 'specialization') if specialization == 'ordinary' else fields)
    require(contract['schema'] == ('fresh_runtime_sgemm_v3' if specialization == 'ordinary'
                                   else 'fresh_runtime_sgemm_v2') and contract['variant'] == binding['variant'] and
            contract['target'] == template.get('sgemm_target') == '2:3127' and
            type(contract['max_seconds']) is int and contract['max_seconds'] == 800 and
            contract['service_environment'] == template.get('service_environment'),
            'fresh claim contract differs from template')
    deadline = datetime.fromisoformat(contract['deadline_utc'])
    require(deadline.tzinfo is not None and
            (deadline - datetime.now(timezone.utc)).total_seconds() >= minimum_remaining and
            deadline > datetime.now(timezone.utc), 'fresh claim deadline exhausted')
    plan_path, review_path = Path(binding['plan']), Path(binding['review'])
    require(plan_path.is_absolute() and plan_path == plan_path.resolve(strict=True) and
            plan_path.is_relative_to(PROJECT / 'artifacts/maintenance') and
            review_path.parent == plan_path.parent and review_path == review_path.resolve(strict=True),
            'canonical reviewed claim plan required')
    plan_directory = plan_path.parent.lstat()
    require(stat.S_ISDIR(plan_directory.st_mode) and plan_directory.st_uid == uid and
            plan_directory.st_mode & 0o022 == 0, 'private reviewed claim directory required')
    plan_bytes = _owned_bounded_file(plan_path, uid, gid, MESSAGE_MAX)
    review_bytes = _owned_bounded_file(review_path, uid, gid, MESSAGE_MAX)
    require(hashlib.sha256(plan_bytes).hexdigest() == binding['plan_sha256'] and
            hashlib.sha256(review_bytes).hexdigest() == binding['review_sha256'],
            'fresh reviewed plan or review differs')
    plan, review = decode(plan_bytes), decode(review_bytes)
    exact_fields(review, ('status', 'contract', 'source_sha256'))
    require(plan == dict(contract, review=dict(path=str(review_path), sha256=binding['review_sha256'])) and
            review.get('status') == ('fresh_runtime_v3_ready' if specialization == 'ordinary'
                                     else 'fresh_runtime_v2_ready') and review.get('contract') == contract,
            'fresh independent review or contract differs')
    sources = review['source_sha256']
    required_sources = {str(PROJECT / 'scripts' / name) for name in (
        'bert_ncu_current_sdk.py', 'bert_ncu_replay_exec.py', 'run_bert_candidate_validation.py',
        'run_bert_sgemm_ncu_capture.py', 'bert_ncu_sibling.py', 'bert_ncu_relay.py',
        'bert_ncu_service_proxy.py', 'run_bert_sgemm_descriptor_probe.py',
        'bert_sgemm_coverage_candidates.py', 'bert_sgemm_ncu_process.py')}
    require(type(sources) is dict and required_sources <= set(sources) and len(sources) <= 64,
            'reviewed fresh source inventory required')
    for name, expected in sources.items():
        require(type(name) is str and type(expected) is str and re.fullmatch(r'[0-9a-f]{64}', expected),
                'invalid reviewed fresh source')
        source = Path(name)
        require(source.is_absolute() and source.is_relative_to(PROJECT) and
                hashlib.sha256(_owned_bounded_file(source, uid, gid, 4 * 1024 ** 2)).hexdigest() == expected,
                'reviewed fresh source differs')
    claim_path = plan_path.parent / (plan_path.name + '.launched')
    claim_bytes = _owned_bounded_file(claim_path, uid, gid, 4096, mode=0o600)
    require(hashlib.sha256(claim_bytes).hexdigest() == claim_sha256, 'fresh claim digest differs')
    claim = decode(claim_bytes)
    claim_fields = ('schema', 'attempt', 'variant', 'target', 'plan', 'plan_sha256', 'review',
                    'review_sha256', 'binding_sha256', 'manifest_sha256', 'inputs_sha256')
    exact_fields(claim, (*claim_fields, 'specialization') if specialization == 'ordinary' else claim_fields)
    binding_bytes = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    inputs = _owned_bounded_file(outer / 'inputs-before.json', uid, gid, 64 * 1024 ** 2)
    expected_claim = dict(schema='fresh_runtime_claim_v2' if specialization == 'ordinary' else 'fresh_runtime_claim_v1',
        attempt=str(outer), variant=binding['variant'],
        target=contract['target'], plan=str(plan_path), plan_sha256=binding['plan_sha256'],
        review=str(review_path), review_sha256=binding['review_sha256'],
        binding_sha256=hashlib.sha256(binding_bytes).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), inputs_sha256=hashlib.sha256(inputs).hexdigest())
    if specialization == 'ordinary':
        expected_claim['specialization'] = specialization
    require(claim == expected_claim,
        'fresh claim is not bound to this attempt and inputs')
    application = template.get('application_input_contract')
    exact_fields(application, ('schema', 'recorded_files', 'historical_sha256', 'historical_stat_only',
        'recorded_digest', 'historical_sha256_digest', 'historical_stat_only_digest', 'loader_sha256',
        'complete_python_import_closure'))
    counts = [application[key] for key in ('recorded_files', 'historical_sha256', 'historical_stat_only')]
    require(application['schema'] == 'recorded_application_inputs_v1' and
            all(type(value) is int and value >= 0 for value in counts) and
            counts[0] > 0 and counts[0] == counts[1] + counts[2] and
            application['complete_python_import_closure'] is False and
            all(type(application[key]) is str and re.fullmatch(r'[0-9a-f]{64}', application[key])
                for key in ('recorded_digest', 'historical_sha256_digest', 'historical_stat_only_digest')) and
            type(application['loader_sha256']) is dict and len(application['loader_sha256']) == 5 and
            all(type(name) is str and Path(name).is_absolute() and type(value) is str and
                re.fullmatch(r'[0-9a-f]{64}', value) for name, value in application['loader_sha256'].items()),
            'fresh recorded application input contract required')
    return dict(claim_sha256=claim_sha256, manifest_sha256=claim['manifest_sha256'],
                binding_sha256=claim['binding_sha256'], binding=binding, template=template)


def consume_fresh_claim(outer, authority):
    """Publish one root-owned receipt; any failure leaves the claim consumed."""
    require(os.geteuid() == 0 and os.getegid() == 0, 'root-only fresh claim consumption required')
    exact_fields(authority, ('claim_sha256', 'manifest_sha256', 'binding_sha256', 'binding', 'template'))
    outer = Path(outer)
    template = authority['template']
    require(outer == outer.resolve(strict=True) and template['fresh_runtime'] == authority['binding'],
            'authenticated fresh claim context required')
    record = dict(schema='fresh_root_claim_v1', attempt=str(outer), claim_sha256=authority['claim_sha256'],
                  manifest_sha256=authority['manifest_sha256'], binding_sha256=authority['binding_sha256'])
    payload = encode(record)
    directory_fd = os.open(outer, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    descriptor, created = None, False
    try:
        directory, current = os.fstat(directory_fd), outer.lstat()
        require(stat.S_ISDIR(directory.st_mode) and
                (directory.st_dev, directory.st_ino, directory.st_uid, directory.st_gid) ==
                (current.st_dev, current.st_ino, template['uid'], template['gid']) and
                stat.S_IMODE(directory.st_mode) == 0o700, 'fresh claim attempt directory changed')
        descriptor = os.open('root-fresh-claim.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                             os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory_fd)
        created = True
        os.fchmod(descriptor, 0o600)
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600 and
                (info.st_uid, info.st_gid, info.st_nlink) == (0, 0, 1), 'root claim receipt identity differs')
        remaining = memoryview(payload)
        while remaining:
            count = os.write(descriptor, remaining)
            require(count > 0, 'short root claim receipt write')
            remaining = remaining[count:]
        os.fsync(descriptor)
        require(os.fstat(descriptor).st_size == len(payload), 'root claim receipt size differs')
    finally:
        try:
            if descriptor is not None:
                os.close(descriptor)
        finally:
            try:
                if created:
                    os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    return record


def verify_root_claim_receipt(outer, authority):
    """Recheck the consumed claim in a root replay helper before service spawn."""
    require(os.geteuid() == 0 and os.getegid() == 0, 'root-only fresh claim receipt required')
    exact_fields(authority, ('claim_sha256', 'manifest_sha256', 'binding_sha256', 'binding', 'template'))
    path = Path(outer) / 'root-fresh-claim.json'
    receipt = decode(_owned_bounded_file(path, 0, 0, 4096, mode=0o600))
    require(receipt == dict(schema='fresh_root_claim_v1', attempt=str(outer),
        claim_sha256=authority['claim_sha256'], manifest_sha256=authority['manifest_sha256'],
        binding_sha256=authority['binding_sha256']), 'root fresh claim receipt differs')
    return receipt


def authenticate_fresh_replay(plan, outer, template):
    """Independently bind one root replay to the consumed fresh claim."""
    require(type(plan) is dict and type(template) is dict and 'fresh_runtime' in template and
            type(plan.get('fresh_claim_sha256')) is str, 'fresh replay claim required')
    authority = verify_fresh_claim(outer, plan['fresh_claim_sha256'], plan['uid'], plan['gid'])
    require(authority['manifest_sha256'] == plan['template_sha256'] and authority['template'] == template,
            'fresh replay differs from authenticated template')
    verify_root_claim_receipt(outer, authority)
    return authority


def _read_runtime_product(product, uid, gid, destination=None):
    """Verify and optionally copy the same opened user-owned runtime file."""
    exact_fields(product, ('path', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'mode', 'uid', 'gid', 'sha256'))
    require(type(uid) is int and uid > 0 and type(gid) is int and gid >= 0 and
            type(product['path']) is str and
            0 < len(product['path']) <= 4096 and '\0' not in product['path'], 'invalid runtime product owner or path')
    path = Path(product['path'])
    require(path.is_absolute() and path == path.resolve(strict=True), 'canonical runtime product path required')
    require(all(type(product[key]) is int and product[key] >= 0 for key in
                ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'mode', 'uid', 'gid')) and
            product['inode'] > 0 and 0 < product['size'] <= 1024 ** 3 and product['mode'] <= 0o7777 and
            type(product['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', product['sha256']) is not None,
            'invalid runtime product identity')
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)

        def metadata(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns,
                    stat.S_IMODE(value.st_mode), value.st_uid, value.st_gid)

        expected = (product['device'], product['inode'], product['size'], product['mtime_ns'],
                    product['ctime_ns'], product['mode'], product['uid'], product['gid'])
        require((product['uid'], product['gid']) == (uid, gid) and
                stat.S_ISREG(before.st_mode) and metadata(before) == expected,
                'runtime product metadata differs')
        digest = hashlib.sha256()
        remaining = product['size']
        while remaining:
            data = os.read(descriptor, min(remaining, 1024 * 1024))
            require(data, 'runtime product ended during verification')
            digest.update(data)
            if destination is not None:
                require(destination.write(data) == len(data), 'short runtime snapshot write')
            remaining -= len(data)
        require(not os.read(descriptor, 1), 'runtime product grew during verification')
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        require(stat.S_ISREG(current.st_mode) and metadata(after) == metadata(current) == expected and
                digest.hexdigest() == product['sha256'], 'runtime product changed or bytes differ')
    finally:
        os.close(descriptor)
    return path


def validate_runtime_product(product, uid, gid):
    """Verify one runtime product and return its canonical path."""
    return _read_runtime_product(product, uid, gid)


def snapshot_runtime_product(product, uid, gid):
    """Seal bytes copied from the same FD whose runtime identity was checked."""
    require(type(product) is dict and type(product.get('path')) is str, 'runtime product path required')
    descriptor = os.memfd_create('glimmer-' + Path(product['path']).name,
                                 os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        with os.fdopen(os.dup(descriptor), 'wb') as destination:
            _read_runtime_product(product, uid, gid, destination)
        require(os.fstat(descriptor).st_size == product['size'] and
                seal(Path(f'/proc/self/fd/{descriptor}')) == product['sha256'],
                'runtime snapshot bytes differ')
        os.fchmod(descriptor, 0o555)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, SNAPSHOT_SEALS)
        require(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) == SNAPSHOT_SEALS,
                'runtime snapshot seals differ')
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def service_executable_snapshot(template, plan, service, *, fresh_authority=None):
    """Snapshot an already selected service; caller authenticates the frozen template."""
    if 'fresh_runtime' not in template:
        require(fresh_authority is None, 'legacy snapshot cannot accept fresh authority')
        return sealed_snapshot(service, plan['seals'][str(service)])
    require(selected_service(template, fresh_authority=fresh_authority) == service,
            'fresh service snapshot selection differs')
    binding = selected_signal_binding(template)
    exact_fields(binding, ('schema', 'variant', 'contract', 'plan', 'plan_sha256', 'review', 'review_sha256'))
    contract = binding['contract']
    specialization = capture_specialization(binding)
    fields = ('schema', 'variant', 'target', 'max_seconds', 'deadline_utc', 'runtime',
              'service_environment', 'numerical', 'reference')
    exact_fields(contract, (*fields, 'specialization') if specialization == 'ordinary' else fields)
    exact_fields(contract['runtime'], ('service', 'frontend'))
    product = contract['runtime']['service']
    exact_fields(product, ('path', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'mode', 'uid', 'gid', 'sha256'))
    require(contract['schema'] == ('fresh_runtime_sgemm_v3' if specialization == 'ordinary'
                                   else 'fresh_runtime_sgemm_v2') and contract['variant'] == binding['variant'] and
            contract['target'] == template['sgemm_target'] == '2:3127' and
            type(contract['max_seconds']) is int and contract['max_seconds'] == 800 and
            template['service_cap_sys_admin'] is True and
            contract['service_environment'] == template['service_environment'] == plan['environment'] and
            (product['uid'], product['gid']) == (plan['uid'], plan['gid']) == (template['uid'], template['gid']) and
            type(product['mode']) is int and product['mode'] == 0o775 and product['mode'] & 0o100 != 0 and
            product['mode'] & 0o7000 == 0 and str(service) == product['path'] and
            plan['seals'][str(service)] == product['sha256'], 'fresh service snapshot binding differs')
    return snapshot_runtime_product(product, plan['uid'], plan['gid'])


def validate_numerical_runtime(runtime, numerical_inputs, uid, gid):
    """Match exact runtime products to an independently authenticated numerical manifest."""
    exact_fields(runtime, ('service', 'frontend'))
    require(type(numerical_inputs) is dict, 'numerical input inventory required')
    result = {}
    for name in ('service', 'frontend'):
        product = runtime[name]
        exact_fields(product, ('path', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'mode', 'uid', 'gid', 'sha256'))
        require(type(product['path']) is str, 'runtime product path required')
        item = numerical_inputs.get(product['path'])
        exact_fields(item, ('identity', 'resolved', 'sha256'))
        require(item['resolved'] == product['path'] and item['sha256'] == product['sha256'] and
                item['identity'] == [product['device'], product['inode'], product['size'],
                                     product['mtime_ns'], product['ctime_ns']],
                'runtime differs from numerical manifest')
        result[name] = validate_runtime_product(product, uid, gid)
    require(result['service'] != result['frontend'] and
            (runtime['service']['device'], runtime['service']['inode']) !=
            (runtime['frontend']['device'], runtime['frontend']['inode']),
            'service and frontend must be distinct products')
    return result


def capture_specialization(binding):
    """Derive the only allowed kernel specialization from an authenticated binding."""
    if binding is None:
        return 'observed'
    require(type(binding) is dict and binding.get('variant') in ('scalar', 'sharded'),
            'invalid fresh runtime signal binding')
    if binding.get('schema') == 'fresh_runtime_v2':
        require('contract' not in binding or
                (type(binding['contract']) is dict and
                 binding['contract'].get('schema') == 'fresh_runtime_sgemm_v2' and
                 'specialization' not in binding['contract']), 'invalid observed capture binding')
        return 'observed'
    require(binding.get('schema') == 'fresh_runtime_v3' and type(binding.get('contract')) is dict and
            binding['contract'].get('schema') == 'fresh_runtime_sgemm_v3' and
            binding['contract'].get('specialization') == 'ordinary', 'invalid ordinary capture binding')
    return 'ordinary'


def capture_kernel(binding):
    return ('glimmer_owned_sgemm_nn' if capture_specialization(binding) == 'ordinary'
            else 'glimmer_owned_sgemm_nn_observed')


def signal_arguments(binding):
    """Derive the optional signal switch from the caller's authenticated binding.

    Callers must authenticate the binding against the frozen task template;
    this function does not authenticate a dictionary on its own.
    """
    if binding is None:
        return []
    capture_specialization(binding)
    return ['--experimental-sharded-gemm-signal'] if binding['variant'] == 'sharded' else []


def selected_signal_binding(template):
    """Select the signal binding only from the authenticated task template."""
    require(type(template) is dict, 'task template required')
    if 'fresh_runtime' not in template:
        return None
    require('current_sdk' not in template and 'host_pair' not in template,
            'only one runtime selection is allowed')
    binding = template['fresh_runtime']
    require(type(binding) is dict, 'fresh runtime binding required')
    signal_arguments(binding)
    return binding


def service_command(plan, *, service=SERVICE, signal_binding=None):
    low = plan['clients']['low']
    pins = [f"{plan['clients'][role]['pid']}:{plan['clients'][role]['start']}:0:{selection}"
            for role, selection in (('low', 'configured'), ('high', 'original'))]
    observation = ['--observe-block-completion'] if capture_specialization(signal_binding) == 'observed' else []
    return [str(service), '--socket', plan['socket'], '--exit-when-idle',
            '--persistent-workers', '320', '--kernel-mode', 'preemptive', '--scheduler', 'priority',
            '--replace-gemm', '--audit-native', '--audit-priority', *observation,
            '--memory-per-client', '12884901888', '--host-memory-per-client', '536870912',
            '--client-priority', pins[0], '--client-priority', pins[1], '--priority-idle-us', '2000',
            '--gemm-tile-side', '32', '--audit-sgemm-target-client', f"{low['pid']}:{low['start']}:3127",
            '--audit-sgemm-range-library', str(RANGE_LIBRARY.resolve(strict=True)),
            *signal_arguments(signal_binding)]


def service_exec_command(plan, outer, inherited, executable_fd, *, fresh_authority=None):
    template = frozen_template(outer, plan['template_sha256'])
    require(('fresh_runtime' in template) == (fresh_authority is not None),
            'service command and fresh authority mode differ')
    environment = dict(profiler_environment(inherited, template, outer), **plan['environment'])
    environment.update(CUDA_CACHE_DISABLE='1', TMPDIR=str(outer / 'ncu-tmp'))
    service = (selected_service(template, fresh_authority=fresh_authority) if 'fresh_runtime' in template
               else selected_service(template))
    command = service_command(plan, service=service,
                              signal_binding=selected_signal_binding(template))
    command[0] = f'/proc/self/fd/{executable_fd}'
    return [*drop_command(plan['uid'], plan['gid'], service=True), str(ENV), '-i',
            *[key + '=' + value for key, value in sorted(environment.items())], *command]


def tracker_descriptors(environment, temporary, owner):
    """Retain only authenticated NCU FIFOs, including its private duplicate FDs.

    NCU reserves descriptors that its close interposer rejects. The advertised
    pair is not the whole set: exec also needs the internal duplicate streams.
    Call only after authenticating the profiler ancestry and loader environment.
    """
    tracking = environment.get('NVIDIA-PROCESS-TRACKING-CONFIGURATION', '')
    require(type(tracking) is str and 0 < len(tracking) <= 16384, 'bounded tracker configuration required')
    roots = re.findall(r'^tree-tracker-id: (.+)$', tracking, re.MULTILINE)
    names = re.findall(r'^connection-name: (.+)$', tracking, re.MULTILINE)
    pairs = re.findall(r'^pipe-descriptor:\n  - ([1-9][0-9]*)\n  - ([1-9][0-9]*)\n'
                       r'  - ""\n  - ([0-9]+)\n  - false(?:\n(?=\S)|\Z)', tracking, re.MULTILINE)
    require(len(roots) == len(names) == len(pairs) == 1, 'exact tracker paths and descriptor pair required')
    require(len(re.findall(r'^pipe-descriptor:', tracking, re.MULTILINE)) == 1,
            'duplicate tracker descriptor field')
    temporary = Path(temporary).resolve(strict=True)
    root = Path(roots[0])
    require(root == root.resolve(strict=True) and root.parent == temporary and
            re.fullmatch(r'NVIDIA-treetracker-[A-Za-z0-9]+', root.name) is not None,
            'tracker directory is not in the private task temporary directory')
    metadata = root.stat()
    require(stat.S_ISDIR(metadata.st_mode) and metadata.st_uid == owner and
            metadata.st_mode & 0o7777 == 0o700, 'tracker directory ownership or permissions differ')
    connection = Path(names[0])
    require(connection.parent == root and re.fullmatch(r'pipe-[0-9]+', connection.name) is not None,
            'tracker connection is not private')
    advertised = [int(value) for value in pairs[0][:2]]
    require(len(set(advertised)) == 2 and all(2 < descriptor < 2 ** 31 for descriptor in advertised),
            'two distinct live tracker descriptors required')

    def verified_fifo(descriptor, role=None):
        path = Path(os.readlink(f'/proc/self/fd/{descriptor}'))
        match = re.fullmatch(r'(pipe-[0-9]+)-owner(Reading|Writing)Stream', path.name)
        require(path.parent == root and match is not None and path == path.resolve(strict=True) and
                (role is None or match[2] == role), 'unexpected tracker descriptor destination or role')
        actual, expected = os.fstat(descriptor), path.lstat()
        # NCU creates mode 0664 FIFOs inside its private mode 0700 directory.
        # The directory controls access; group write on the FIFO is not exposure.
        require(stat.S_ISFIFO(actual.st_mode) and stat.S_ISFIFO(expected.st_mode) and
                actual.st_uid == expected.st_uid == owner and actual.st_mode & 0o7111 == 0 and
                actual.st_mode & 0o600 == 0o600 and
                (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino),
                'tracker descriptor does not identify its protected FIFO')
        access = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        # The advertised pair describes the target's ends. NCU also retains
        # different opens of those same two FIFO inodes with other access modes.
        if role is not None:
            require(access in ((os.O_WRONLY, os.O_RDWR) if role == 'Reading' else (os.O_RDONLY, os.O_RDWR)),
                    'tracker stream access mode differs')
        return (actual.st_dev, actual.st_ino), path, match[1]

    declared = [verified_fifo(descriptor, role) for descriptor, role in zip(advertised, ('Writing', 'Reading'))]
    identities = {entry[0] for entry in declared}
    require(len(identities) == 2 and declared[0][2] == declared[1][2], 'tracker streams are not one distinct pair')
    retained = set()
    for name in os.listdir('/proc/self/fd'):
        descriptor = int(name)
        if descriptor <= 2:
            continue
        try:
            path = Path(os.readlink(f'/proc/self/fd/{descriptor}'))
            metadata = os.fstat(descriptor)
        except FileNotFoundError:
            continue
        matched = (metadata.st_dev, metadata.st_ino) in identities
        if path.parent != root and not matched:
            continue
        require(matched and verified_fifo(descriptor)[0] in identities, 'unadvertised tracker descriptor')
        retained.add(descriptor)
    require(set(advertised) <= retained and len(retained) <= 16,
            'advertised NCU descriptors are not authenticated live FIFOs')
    return retained


def child_file_actions(log_fd, executable_fd, retained=()):
    actions = [(os.POSIX_SPAWN_OPEN, 0, '/dev/null', os.O_RDONLY, 0),
               (os.POSIX_SPAWN_DUP2, log_fd, 1), (os.POSIX_SPAWN_DUP2, log_fd, 2)]
    # The service receives only its executable and authenticated profiler streams.
    for name in os.listdir('/proc/self/fd'):
        descriptor = int(name)
        if descriptor > 2 and descriptor != executable_fd and descriptor not in retained:
            try:
                os.fstat(descriptor)
            except OSError:
                continue
            actions.append((os.POSIX_SPAWN_CLOSE, descriptor))
    return actions


def close_except(retained):
    for name in os.listdir('/proc/self/fd'):
        descriptor = int(name)
        if descriptor > 2 and descriptor not in retained:
            try:
                os.close(descriptor)
            except OSError:
                # /proc/self/fd's directory descriptor closes after enumeration.
                require(not Path(f'/proc/self/fd/{descriptor}').exists(), 'descriptor close failed')


def wait_receipt(pid, start, raw_status):
    require(positive(pid) and positive(start) and type(raw_status) is int and 0 <= raw_status <= 65535,
            'valid child identity and raw wait status required')
    require(os.WIFEXITED(raw_status) or os.WIFSIGNALED(raw_status), 'terminal child wait status required')
    return dict(pid=pid, start=start, raw_status=raw_status,
                returncode=os.waitstatus_to_exitcode(raw_status), exit_code_observed=True)


def reap_child(child, deadline_ns, guard=lambda: None):
    """Only the actual parent can produce this receipt; ECHILD is never success."""
    validate_identity(child)
    while time.monotonic_ns() < deadline_ns:
        guard()
        pid, status = os.waitpid(child['pid'], os.WNOHANG)
        require(pid in (0, child['pid']), 'waitpid returned another child')
        if pid:
            return wait_receipt(pid, child['start'], status)
        time.sleep(.02)
    raise TimeoutError('service wait deadline reached')


def validate_child_fd(child, descriptor, group):
    require(type(descriptor) is int and descriptor > 2 and
            os.readlink(f'/proc/self/fd/{descriptor}') == 'anon_inode:[pidfd]',
            'inherited service pidfd required')
    fields = dict(line.split(':', 1) for line in Path(f'/proc/self/fdinfo/{descriptor}').read_text().splitlines())
    require(fields.get('Pid', '').strip() == str(child['pid']), 'service pidfd identifies another process')
    live_identity(child, group)
    status = Path(f"/proc/{child['pid']}/stat").read_text().rsplit(')', 1)[1].split()
    require(status[1] == str(os.getpid()), 'service is not the waiter direct child')


def retire_child(child, descriptor):
    """Failure cleanup owns one authenticated pidfd and must observe real waitpid."""
    if not terminal(descriptor):
        try:
            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return reap_child(child, time.monotonic_ns() + 2_000_000_000)


def receive_no_fds(connection, expected_credentials):
    message, descriptors, credentials = receive_packet(connection)
    for descriptor in descriptors:
        os.close(descriptor)
    require(not descriptors and credentials == expected_credentials,
            'unexpected descriptor or sender credentials')
    return message


def wait_only(connection, supervisor, outer, child, child_fd, uid, gid, serial, deadline_ns):
    validate_status(Path('/proc/self/status').read_text(), uid, gid, False)
    require(os.getresuid() == (uid,) * 3 and os.getresgid() == (gid,) * 3 and os.getgroups() == [],
            'waiter actual credentials differ')
    group = process_group(os.getpid())
    validate_child_fd(child, child_fd, group)
    supervisor_fd, reaped = None, False
    try:
        group, supervisor_fd = authenticate_supervisor(connection, supervisor, outer)
        # The root supervisor validates this process after exec and authenticates the service.
        send_packet(connection, dict(op='waiter_ready', serial=serial,
                                     waiter=process_identity(os.getpid()), child=child))
        reply = receive_no_fds(connection, (supervisor['pid'], 0, 0))
        require(reply == dict(op='wait', serial=serial), 'waiter release differs')
        require(not terminal(supervisor_fd), 'supervisor terminated before wait')
        def guard():
            require(not terminal(supervisor_fd), 'supervisor terminated during wait')
            readable, _, _ = select.select([connection], [], [], 0)
            require(not readable, 'supervisor disconnected or sent premature control')

        receipt = reap_child(child, deadline_ns, guard)
        reaped = True
        send_packet(connection, dict(op='exit', serial=serial, child=receipt))
        # Keep the replay target alive until the full CPU oracle and cleanup finish.
        reply = receive_no_fds(connection, (supervisor['pid'], 0, 0))
        exact_fields(reply, ('op', 'serial', 'accepted'))
        require(reply['op'] == 'finish' and reply['serial'] == serial and
                type(reply['accepted']) is bool, 'invalid full-pass acknowledgement')
        if not reply['accepted']:
            return 1
        require(not terminal(supervisor_fd) and time.monotonic_ns() < deadline_ns,
                'supervisor lost or replay deadline exhausted')
        return receipt['returncode']
    finally:
        try:
            if not reaped:
                retire_child(child, child_fd)
        finally:
            os.close(child_fd)
            if supervisor_fd is not None:
                os.close(supervisor_fd)


def launch_once(connection, supervisor, outer):
    require(os.getresuid() == (0, 0, 0), 'root launch phase required')
    group, supervisor_fd = authenticate_supervisor(connection, supervisor, outer)
    child_fd, child_pid = None, None
    with ExitStack() as resources:
        resources.callback(os.close, supervisor_fd)
        send_packet(connection, dict(op='hello', helper=process_identity(os.getpid())))
        plan, descriptors, credentials = receive_packet(connection)
        for descriptor in descriptors:
            resources.callback(os.close, descriptor)
        require(credentials == (supervisor['pid'], 0, 0) and len(descriptors) == 1,
                'one log descriptor from the exact supervisor required')
        log_fd = descriptors[0]
        validated = validate_launch(plan, outer, group, log_fd)
        require(not isinstance(validated, tuple) or len(validated) == 2, 'invalid validated launch context')
        fresh_authority = validated[1] if isinstance(validated, tuple) else None
        template = frozen_template(outer, plan['template_sha256'])
        require(('fresh_runtime' in template) == (fresh_authority is not None),
                'launch template and fresh authority mode differ')
        environment = profiler_environment(dict(os.environ), template, outer)
        code_digest = plan['seals'][str(SCRIPT)]
        validate_snapshot(CURRENT_CODE_FD, code_digest)
        service = (selected_service(template, fresh_authority=fresh_authority) if fresh_authority is not None
                   else selected_service(template))
        executable_fd = (service_executable_snapshot(template, plan, service, fresh_authority=fresh_authority)
                         if fresh_authority is not None else service_executable_snapshot(template, plan, service))
        resources.callback(os.close, executable_fd)
        os.set_inheritable(executable_fd, True)
        require(not terminal(supervisor_fd), 'supervisor terminated before spawn')
        command = (service_exec_command(plan, outer, environment, executable_fd, fresh_authority=fresh_authority)
                   if fresh_authority is not None else service_exec_command(plan, outer, environment, executable_fd))
        tracking_fds = tracker_descriptors(environment, outer / 'ncu-tmp', 0)
        try:
            child_pid = os.posix_spawn(command[0], command, environment,
                                       file_actions=child_file_actions(log_fd, executable_fd, tracking_fds),
                                       setsid=True)
            child_fd = os.pidfd_open(child_pid)
            child = process_identity(child_pid)
            live_identity(child, group)
            send_packet(connection, dict(op='spawned', serial=plan['serial'], child=child))
            # exec preserves parenthood. The replacement waiter has no capabilities.
            wait_arguments = ['--wait-fd', str(connection.fileno()),
                         '--supervisor', f"{supervisor['pid']}:{supervisor['start']}",
                         '--attempt', str(outer), '--child', f"{child['pid']}:{child['start']}",
                         '--child-fd', str(child_fd),
                         '--uid', str(plan['uid']), '--gid', str(plan['gid']),
                         '--serial', str(plan['serial']), '--deadline-ns', str(plan['deadline_ns'])]
            wait_argv = [*drop_command(plan['uid'], plan['gid'], service=False),
                         *sealed_python_command(f'/proc/self/fd/{CURRENT_CODE_FD}', code_digest, *wait_arguments)]
            os.set_inheritable(connection.fileno(), True)
            os.set_inheritable(child_fd, True)
            os.set_inheritable(CURRENT_CODE_FD, True)
            # Close only after ExitStack releases its explicitly owned descriptors.
            resources.close()
            environment = profiler_environment(dict(os.environ), template, outer)
            tracking_fds = tracker_descriptors(environment, outer / 'ncu-tmp', 0)
            close_except({connection.fileno(), child_fd, CURRENT_CODE_FD} | tracking_fds)
            os.execve(SETPRIV, wait_argv, environment)
        except BaseException:
            if child_fd is not None:
                # A failed privilege drop never leaves the launched child unsupervised.
                try:
                    retire_child(process_identity(child_pid), child_fd)
                finally:
                    os.close(child_fd)
            # If pidfd acquisition itself failed, the supervisor must contain
            # its exact private cgroup. Never signal an unauthenticated bare PID.
            raise


def parse_identity(value):
    require(re.fullmatch(r'[1-9][0-9]*:[1-9][0-9]*', value) is not None, 'PID:START required')
    pid, start = map(int, value.split(':'))
    return validate_identity(dict(pid=pid, start=start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--endpoint')
    modes.add_argument('--wait-fd', type=int)
    parser.add_argument('--supervisor', required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    parser.add_argument('--child')
    parser.add_argument('--child-fd', type=int)
    parser.add_argument('--uid', type=int)
    parser.add_argument('--gid', type=int)
    parser.add_argument('--serial', type=int)
    parser.add_argument('--deadline-ns', type=int)
    args = parser.parse_args()
    outer = args.attempt.resolve(strict=True)
    supervisor = parse_identity(args.supervisor)
    if args.endpoint is not None:
        require(re.fullmatch(r'glimmer-ncu-[0-9a-f]{32}', args.endpoint) is not None,
                'private replay endpoint required')
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            connection.settimeout(30)
            connection.connect('\0' + args.endpoint)
            launch_once(connection, supervisor, outer)
            raise RuntimeError('service launch unexpectedly returned')
    require(type(args.wait_fd) is int and args.wait_fd > 2 and positive(args.serial) and
            positive(args.deadline_ns) and
            time.monotonic_ns() < args.deadline_ns <= time.monotonic_ns() + 800_000_000_000,
            'bounded wait-only arguments required')
    with socket.socket(fileno=args.wait_fd) as connection:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        connection.settimeout(max(.001, (args.deadline_ns - time.monotonic_ns()) / 1e9))
        code = wait_only(connection, supervisor, outer, parse_identity(args.child),
                         args.child_fd, args.uid, args.gid, args.serial, args.deadline_ns)
    if code < 0:
        signum = -code
        signal.signal(signum, signal.SIG_DFL) if signum not in (signal.SIGKILL, signal.SIGSTOP) else None
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signum})
        os.kill(os.getpid(), signum)
        return 1
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'replay bridge failed: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
