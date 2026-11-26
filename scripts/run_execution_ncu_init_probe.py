#!/usr/bin/env python3
"""Bounded NCU/service initialization or opt-in single-kernel counter diagnostic.

The authorized user unit contains the profiler, debugger, production service
and one clean sibling Runtime fixture. This never qualifies SM coverage.
"""

import argparse
from contextlib import ExitStack
import csv
from decimal import Decimal
import io
import json
import os
from pathlib import Path
import re
import resource
import select
import signal
import subprocess
import time

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
from bert_sgemm_ncu_process import proc_environment, proc_mapped_paths, verified_preload
from run_bert_sgemm_ncu_capture import (check_owned_gpu, publish_json, task_ancestors,
                                       validate_unprivileged_process)
from run_rnn_v8_validation import cgroup, command, digest, identity, memory_receipt, save

SCRIPT = Path(__file__).resolve()
FAILED = PROJECT_ROOT / 'artifacts/profiles/ncu/attempt-5s4yhnnc'
INITIALIZED = PROJECT_ROOT / 'artifacts/profiles/ncu-init/attempt-y6uss9gj'
INITIALIZED_SEALS = {
    'report.json': '296eb8ef3d8920c0c155e661dcfbf93fe00ddf7c3bf32e92b242dcc57161dbd9',
    'manifest.json': '585ee05ccc5493b5452f4bc6292826e82bce9d3b66af199c16c32286f7ae6c3d',
    'diagnostic.json': '018e1238bd4bc5a82678d2b0403352a0332f1744e84ea5916dc7ccc6fce1cfb6',
}
SERVICE = PROJECT_ROOT / 'build/debug/bin/glimmer_execution_service'
CLIENT = PROJECT_ROOT / 'build/debug/src/interceptor/execution/glimmer_execution_runtime_gpu_test'
PRELOAD = PROJECT_ROOT / 'build/debug/lib/libglimmer_execution_interceptor.so'
NCU = Path('/opt/nvidia/nsight-compute/2024.3.2/ncu')
NCU_EXEC = NCU.parent / 'target/linux-desktop-glibc_2_11_3-x64/ncu'
FILE_MAX = 8 * 1024 ** 2
OUTPUT_MAX = 64 * 1024 ** 2
CAP_SYS_ADMIN = 1 << 21
COUNTER_METRICS = ('launch__sm_count', 'launch__grid_size', 'launch__block_size',
                   'sm__cycles_active.sum', 'sm__inst_executed.sum')
CSV_IDENTITY = ('ID', 'Process ID', 'Process Name', 'Host Name', 'Kernel Name', 'Context',
                'Stream', 'Block Size', 'Grid Size', 'Device', 'CC')
CGROUP_ROOT = Path('/sys/fs/cgroup')
SERVICE_ENVIRONMENT_KEYS = frozenset(('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'LANG', 'LC_ALL',
                                     'LD_LIBRARY_PATH', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                                     'OPENBLAS_NUM_THREADS', 'PATH', 'PYTHONNOUSERSITE',
                                     'TOKENIZERS_PARALLELISM', 'TRANSFORMERS_OFFLINE'))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def unit_name(directory):
    return 'glimmer-ncu-init-' + directory.name + '.service'


def unit_state(unit):
    return dict(line.split('=', 1) for line in command([
        'systemctl', '--user', 'show', unit,
        '--property=LoadState,Description,ControlGroup']).splitlines())


def owned_unit(state, unit, description):
    return state.get('Description') == description and (
        not state.get('ControlGroup') or state['ControlGroup'].endswith('/' + unit))


def verify_profiler_baseline(previous):
    for path, key in ((NCU, 'ncu_sha256'), (NCU_EXEC, 'ncu_exec_sha256'),
                      (Path(previous['range_library']), 'range_library_sha256')):
        require(digest(path) == previous[key], 'profiler/range provider changed after failure')


def validate_probe_modes(capable, counters):
    require(type(capable) is bool and type(counters) is bool, 'boolean probe mode flags required')
    require(not counters or capable, 'counter probe requires explicit service capability opt-in')


def offline_metric_support():
    argv = [str(NCU), '--config-file', '0', '--query-metrics', '--chips', 'gv100',
            '--query-metrics-mode', 'suffix', '--metrics', 'sm__cycles_active,sm__inst_executed']
    output = command(argv, timeout=10, env=dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
                                               CUDA_VISIBLE_DEVICES=''))
    require(len(output.encode()) <= 256 * 1024 and 'Chip gv100' in output,
            'bounded offline GV100 metric query required')
    for name, unit in zip(COUNTER_METRICS[-2:], ('cycle', 'inst')):
        require(len(re.findall(r'^' + re.escape(name) + r'\s+Counter\s+' + unit + r'\s',
                               output, re.MULTILINE)) == 1, 'requested hardware counter unavailable')
    return dict(command=argv, ncu_exec_sha256=digest(NCU_EXEC), output=output, gpu_capture=False)


def ncu_probe_command(directory, target, counters):
    require(type(counters) is bool, 'boolean counter mode required')
    selection = (['--kernel-name-base', 'function', '--kernel-name', 'regex:^binary_high$',
                  '--launch-skip', '1', '--launch-count', '1', '--kill', 'no',
                  '--replay-mode', 'kernel', '--disable-extra-suffixes',
                  '--metrics', ','.join(COUNTER_METRICS), '--export', str(directory / 'capture')]
                 if counters else ['--kernel-name', 'regex:^__glimmer_no_selected_kernel__$',
                                   '--metrics', 'launch__sm_count'])
    return [str(NCU), '--config-file', '0', '--target-processes', 'all', '--nvtx',
            '--check-exit-code', 'yes', '--clock-control', 'none', '--cache-control', 'none',
            '--log-file', str(directory / 'ncu.log'), *selection, *target]


def counter_export_command(directory):
    return [str(NCU), '--config-file', '0', '--import', str(directory / 'capture.ncu-rep'),
            '--csv', '--page', 'raw', '--print-units', 'base', '--print-kernel-base', 'function',
            '--metrics', ','.join(COUNTER_METRICS)]


def audit_counter_csv(content, service_pid):
    """Audit one permission-probe result, never infer per-SM participation from sums."""
    require(type(service_pid) is int and service_pid > 0, 'authenticated service PID required')
    require(type(content) is str and len(content.encode()) <= FILE_MAX and content.endswith('\n'),
            'bounded complete counter CSV required')
    require('==ERROR==' not in content, 'NCU export reported an error')
    lines = content.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith('"ID",')]
    require(len(starts) == 1, 'exactly one raw counter CSV header required')
    try:
        reader = csv.DictReader(io.StringIO('\n'.join(lines[starts[0]:])), strict=True)
        fields = reader.fieldnames or []
        require(len(fields) == len(set(fields)) and set(fields) == set(CSV_IDENTITY + COUNTER_METRICS),
                'counter CSV fields or metric inventory differ')
        rows = list(reader)
    except csv.Error as error:
        raise ValueError('malformed counter CSV') from error
    require(len(rows) == 2 and all(None not in row and None not in row.values() for row in rows),
            'one units row and one complete counter launch required')
    units, row = rows
    require(all(units[name] == '' for name in CSV_IDENTITY) and
            [units[name] for name in COUNTER_METRICS] == ['SM', '', '', 'cycle', 'inst'],
            'counter CSV units differ')
    require(all(row[name].isdecimal() for name in ('ID', 'Process ID', 'Context', 'Stream')) and
            int(row['Process ID']) == service_pid and int(row['Context']) > 0 and int(row['Stream']) > 0 and
            row['Kernel Name'] == 'binary_high', 'counter launch identity differs')
    require(row['Process Name'] == SERVICE.name and row['Device'] == '0' and row['CC'] == '7.0' and
            row['Block Size'] == '(32, 1, 1)' and row['Grid Size'] == '(320, 1, 1)',
            'counter process/device/geometry metadata differs')
    values = {}
    for name in COUNTER_METRICS:
        require(re.fullmatch(r'(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9]+)?',
                             row[name]) is not None, 'invalid counter metric spelling')
        value = Decimal(row[name].replace(',', ''))
        require(value.is_finite() and value > 0 and value == value.to_integral_value(),
                'positive integral counter metric required')
        values[name] = int(value)
    require([values[name] for name in COUNTER_METRICS[:3]] == [80, 320, 32],
            'counter launch geometry differs')
    return dict(service_pid=service_pid, action_id=int(row['ID']), context=int(row['Context']),
                stream=int(row['Stream']), kernel='binary_high', metrics=values,
                positive_hardware_counters=True, sm_coverage_verified=False,
                full_device_qualified=False, performance_qualified=False)


def bounded_regular_digest(path):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= FILE_MAX,
            'nonempty bounded regular counter artifact required')
    return digest(path)


def audit_counter_artifacts(directory, service_pid, before):
    require(bounded_regular_digest(directory / 'capture.ncu-rep') == before,
            'counter report changed during export')
    table = directory / 'counter-metrics.csv'
    csv_digest = bounded_regular_digest(table)
    audited = audit_counter_csv(table.read_text(), service_pid)
    require(bounded_regular_digest(table) == csv_digest, 'counter CSV changed during audit')
    return dict(report_sha256=before, csv_sha256=csv_digest), audited


def audit_counter_postflight(directory, diagnostic, *, unit_removed, inputs_unchanged):
    """Revalidate final bytes after unit retirement; inner success alone is insufficient."""
    require(unit_removed is True and inputs_unchanged is True,
            'counter postflight requires removed unit and unchanged inputs')
    require(all(diagnostic.get(key) is True for key in
                ('compatibility_passed', 'counter_collection', 'counter_access_verified')),
            'complete inner counter collection required')
    require(all(diagnostic.get(key) is False for key in
                ('sm_coverage_verified', 'full_device_qualified')),
            'counter probe cannot qualify full-device participation')
    expected = diagnostic['counter_artifacts']
    require(type(expected) is dict and set(expected) == {'report_sha256', 'csv_sha256'} and
            all(type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) for value in expected.values()),
            'exact counter artifact hashes required')
    artifacts, audited = audit_counter_artifacts(directory, diagnostic['service']['pid'],
                                                 expected['report_sha256'])
    require(artifacts == expected, 'final counter artifact hashes differ')
    require(json.dumps(audited, sort_keys=True) == json.dumps(diagnostic['counter_audit'], sort_keys=True),
            'final counter audit differs')
    return audited


def verify_initialized_candidate():
    require(all(bounded_regular_digest(INITIALIZED / name) == value
                for name, value in INITIALIZED_SEALS.items()), 'reviewed initialization evidence changed')
    baseline = read(INITIALIZED / 'manifest.json')
    require(read(INITIALIZED / 'report.json').get('compatibility_passed') is True and
            all(digest(path) == baseline['seals'][str(path)] for path in (SERVICE, CLIENT, PRELOAD)),
            'counter probe requires the initialized production candidate')


def drop(manifest, *, service=False):
    require(type(manifest['uid']) is int and manifest['uid'] > 0 and
            type(manifest['gid']) is int and manifest['gid'] >= 0, 'non-root identity required')
    caps = '-all,+sys_admin' if service and manifest.get('service_cap_sys_admin') is True else '-all'
    return ['/usr/bin/setpriv', '--reuid=' + str(manifest['uid']),
            '--regid=' + str(manifest['gid']), '--clear-groups', '--inh-caps=' + caps,
            '--ambient-caps=' + caps, '--bounding-set=' + caps, '--no-new-privs']


def validate_service_status(status, manifest):
    fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
    require(fields['Uid'].split() == [str(manifest['uid'])] * 4 and
            fields['Gid'].split() == [str(manifest['gid'])] * 4 and not fields['Groups'].split(),
            'service actual credentials differ')
    expected = CAP_SYS_ADMIN if manifest.get('service_cap_sys_admin') is True else 0
    for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'):
        require(int(fields[name], 16) == expected, 'service capability set differs: ' + name)
    require(fields['NoNewPrivs'].strip() == '1', 'service must prevent new privileges')


def validate_target(record, manifest, group):
    """Authenticate the sole service before a debugger may attach to it."""
    pid = record['pid']
    require(type(pid) is int and pid > 0 and type(record['start']) is int and
            record['start'] > 0, 'positive exact service identity required')
    require(identity(pid) == record['start'] and cgroup(pid) == group,
            'service start/cgroup differs')
    expected, actual = SERVICE.stat(), Path(f'/proc/{pid}/exe').stat()
    require((expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino) and
            digest(SERVICE) == manifest['seals'][str(SERVICE)], 'service executable differs')
    status = Path(f'/proc/{pid}/status').read_text()
    validate_service_status(status, manifest)
    if 'argv' in record:
        require(Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1] ==
                [os.fsencode(value) for value in record['argv']], 'service arguments differ')
    require(identity(pid) == record['start'] and cgroup(pid) == group,
            'service identity changed during validation')


def group_members(group):
    return {int(value) for value in (Path('/sys/fs/cgroup') / group.lstrip('/') /
                                    'cgroup.procs').read_text().split()}


def authenticated_baseline(group):
    records = [dict(pid=os.getpid(), start=identity(os.getpid())), *task_ancestors(group)]
    require(all(type(row['pid']) is int and row['pid'] > 0 and type(row['start']) is int and
                row['start'] > 0 for row in records), 'invalid baseline identity')
    require(len({row['pid'] for row in records}) == len(records), 'duplicate baseline identity')
    require(group_members(group) == {row['pid'] for row in records}, 'foreign initial unit member')
    require(not remaining_descendants(group, records), 'baseline changed during authentication')
    return records


def remaining_descendants(group, baseline):
    expected = {row['pid']: row['start'] for row in baseline}
    remaining = set()
    for pid in group_members(group):
        try:
            if pid not in expected or identity(pid) != expected[pid] or cgroup(pid) != group:
                remaining.add(pid)
        except FileNotFoundError:
            continue
    return remaining


def open_containment(group, attempt):
    require(group.endswith('/' + unit_name(attempt)), 'wrong containment unit')
    root = CGROUP_ROOT.resolve(strict=True)
    directory = (root / group.lstrip('/')).resolve(strict=True)
    require(directory.is_relative_to(root) and directory != root, 'invalid containment directory')
    kill = directory / 'cgroup.kill'
    descriptor = os.open(kill, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        expected, actual = kill.stat(), os.fstat(descriptor)
        require((expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino),
                'containment control identity differs')
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def discover_service(group, argv, manifest):
    """Return the sole authenticated service and its live pidfd, never a name match."""
    expected, candidates = SERVICE.stat(), []
    for pid in group_members(group):
        try:
            actual = Path(f'/proc/{pid}/exe').stat()
            if (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino):
                candidates.append(dict(pid=pid, start=identity(pid), argv=argv))
        except FileNotFoundError:
            continue
    require(len(candidates) == 1, 'exactly one service executable required in private unit')
    record = candidates[0]
    descriptor = os.pidfd_open(record['pid'])
    try:
        validate_target(record, manifest, group)
        require(not pidfd_terminal(descriptor), 'service already terminated')
    except BaseException:
        os.close(descriptor)
        raise
    return record, descriptor


def pidfd_terminal(descriptor):
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    return bool(poller.poll(0))


def debugger_command(pid):
    require(type(pid) is int and pid > 0, 'positive debugger target required')
    expressions = ['set pagination off', 'set confirm off', 'set debuginfod enabled off',
                   'set print frame-arguments none', 'handle SIGSEGV stop print pass',
                   'echo GLIMMER_GDB_ATTACHED\\n', 'continue']
    # A normally exited inferior has no registers/maps; do not turn success into a GDB error.
    expressions.extend('python gdb.execute(' + repr(value) +
                       ') if gdb.selected_inferior().pid else None' for value in
                       ('info registers', 'thread apply all bt 24', 'info sharedlibrary',
                        'info proc mappings'))
    expressions.append('quit')
    return ['/usr/bin/gdb', '-q', '-nx', '-batch', '-p', str(pid),
            *[item for expression in expressions for item in ('-ex', expression)]]


def release_record(line, group):
    """Read only the supervisor-created pipe, not an attempt-directory authorization file."""
    def unique_pairs(pairs):
        result = dict(pairs)
        require(len(result) == len(pairs), 'duplicate parent release key')
        return result
    record = json.loads(line, object_pairs_hook=unique_pairs)
    require(set(record) == {'release', 'pid', 'start'} and record['release'] == 'start',
            'invalid parent release')
    require(type(record['pid']) is int and record['pid'] > 0 and
            type(record['start']) is int and record['start'] > 0, 'invalid parent target identity')
    require(identity(record['pid']) == record['start'] and cgroup(record['pid']) == group,
            'parent target identity differs')
    return record


def service_command(directory, manifest, client):
    return [str(SERVICE), '--socket', str(directory / 'service.sock'), '--exit-when-idle',
            '--persistent-workers', '320', '--kernel-mode', 'preemptive', '--scheduler', 'priority',
            '--replace-gemm', '--audit-native', '--audit-priority', '--observe-block-completion',
            '--memory-per-client', '12884901888', '--host-memory-per-client', '536870912',
            '--client-priority', f"{client['pid']}:{client['start']}:0:configured",
            '--priority-idle-us', '2000', '--gemm-tile-side', '32',
            '--audit-sgemm-target', '2:3127', '--audit-sgemm-range-library', manifest['range_library']]


def capable_service_environment(directory, manifest):
    require(set(manifest['environment']) == SERVICE_ENVIRONMENT_KEYS, 'service environment keys differ')
    environment = dict(manifest['environment'], CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'],
                       CUDA_CACHE_DISABLE='1', TMPDIR=str(directory))
    require(all(re.fullmatch(r'[A-Z][A-Z0-9_]*', key) and isinstance(value, str) and
                '\0' not in value for key, value in environment.items()), 'invalid environment assignment')
    return environment


def capable_service_command(directory, manifest, client):
    require(manifest.get('service_cap_sys_admin') is True, 'explicit service capability opt-in required')
    environment = capable_service_environment(directory, manifest)
    # env immediately execs the service; root NCU does not inherit the workload library search path.
    return [*drop(manifest, service=True), '/usr/bin/env',
            *[key + '=' + value for key, value in sorted(environment.items())],
            *service_command(directory, manifest, client)]


def validate_service_environment(record, directory, manifest):
    actual = proc_environment(record['pid'])
    require(b'LD_AUDIT' not in actual, 'service must not have LD_AUDIT')
    require(actual.get(b'NVTX_INJECTION64_PATH') == b'libcuda-injection.so',
            'service NVTX injection differs')
    if manifest.get('service_cap_sys_admin') is True:
        expected = capable_service_environment(directory, manifest)
        require(all(actual.get(os.fsencode(key)) == os.fsencode(value)
                    for key, value in expected.items()), 'service environment values differ')
    return verified_preload(os.fsdecode(actual.get(b'LD_PRELOAD', b'')), None,
                            manifest['ncu_injection_seals'], proc_mapped_paths(record['pid']))


def compatibility_passed(report, stack, client_log):
    return (report.get('runtime_ready') is True and
            report.get('service_terminal') is True and
            all(report['returncodes'].get(name) == 0 for name in ('client', 'ncu', 'gdb')) and
            'received signal' not in stack and
            re.search(r'^\[Inferior \d+ \(process ' + str(report['service']['pid']) +
                      r'\) exited normally\]$', stack, re.MULTILINE) is not None and
            'Ordinary Runtime priority application verified: role=high;' in client_log)


def retire_owned(children, service_fd, group, baseline, report):
    """Reap direct children; request exact-cgroup containment on any cleanup uncertainty."""
    errors = []
    if service_fd is not None:
        try:
            if not pidfd_terminal(service_fd):
                signal.pidfd_send_signal(service_fd, signal.SIGTERM)
                report['service_cleanup_signal'] = 'SIGTERM'
                report['compatibility_passed'] = False
        except ProcessLookupError:
            pass
        except BaseException as error:
            errors.append('service: ' + str(error))
        finally:
            os.close(service_fd)
    report['child_cleanup'] = []
    for process in reversed(children):
        signals = []
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.terminate()
                signals.append('SIGTERM')
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                signals.append('SIGKILL')
                process.wait(timeout=2)
        except BaseException as error:
            errors.append(str(process.pid) + ': ' + str(error))
        report['child_cleanup'].append(dict(pid=process.pid, returncode=process.returncode,
                                           signals=signals))
        if signals:
            report['compatibility_passed'] = False
    remaining = remaining_descendants(group, baseline)
    report['remaining_descendants'] = sorted(remaining)
    report['cleanup_errors'] = errors
    if remaining or errors:
        report.update(status='failed', compatibility_passed=False,
                      containment='kill_exact_private_cgroup')
    return bool(remaining or errors)


def child(directory, role):
    manifest = read(directory / 'manifest.json')
    validate_probe_modes(manifest.get('service_cap_sys_admin', False), manifest.get('counter_probe', False))
    validate_unprivileged_process(manifest, Path('/proc/self/status').read_text(),
                                  os.getresuid(), os.getresgid(), os.getgroups())
    require(cgroup(os.getpid()).endswith('/' + unit_name(directory)), 'wrong child task unit')
    if role == 'client':
        require(not proc_mapped_paths(os.getpid()) and 'NVTX_INJECTION64_PATH' not in os.environ,
                'client must be outside the NCU process tree')
        publish_json(directory / 'client-ready.json', dict(pid=os.getpid(), start=identity(os.getpid())))
        record = release_record(input(), cgroup(os.getpid()))
        environment = dict(manifest['environment'], LD_PRELOAD=str(PRELOAD),
                           GLIMMER_EXECUTION_SOCKET=str(directory / 'service.sock'),
                           GLIMMER_EXECUTION_SERVER_PID=str(record['pid']),
                           GLIMMER_EXECUTION_SERVER_START=str(record['start']),
                           GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='30000')
        environment.pop('NVTX_INJECTION64_PATH', None)
        os.execve(CLIENT, [str(CLIENT), '--high'], environment)
    require(os.environ.get('NVTX_INJECTION64_PATH') == 'libcuda-injection.so',
            'service controller lost NCU injection')
    client = read(directory / 'client-ready.json')
    environment = dict(manifest['environment'], NVTX_INJECTION64_PATH='libcuda-injection.so')
    argv = service_command(directory, manifest, client)
    with (directory / 'service.log').open('x') as log:
        process = subprocess.Popen(argv, env=environment, stdout=log, stderr=subprocess.STDOUT)
        record = dict(pid=process.pid, start=identity(process.pid), argv=argv)
        publish_json(directory / 'service-ready.json', record)
        record['returncode'] = process.wait(timeout=100)
        save(directory / 'service-exit.json', record)
        return int(record['returncode'] != 0)


def supervise(directory):
    require(os.getresuid() == (0, 0, 0), 'root diagnostic supervisor required')
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
            'clean supervisor required')
    os.umask(0o022)
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_MAX, FILE_MAX))
    manifest = read(directory / 'manifest.json')
    validate_probe_modes(manifest.get('service_cap_sys_admin', False), manifest.get('counter_probe', False))
    group = cgroup(os.getpid())
    require(group.endswith('/' + unit_name(directory)), 'exact private unit required')
    require(directory.stat().st_uid == manifest['uid'], 'attempt owner differs')
    capable = manifest.get('service_cap_sys_admin') is True
    counters = manifest.get('counter_probe') is True
    report = dict(status='started', counter_collection=False, sm_coverage_verified=False,
                  counter_collection_requested=counters, counter_access_verified=False,
                  full_device_qualified=False, service_cap_sys_admin=capable, compatibility_passed=False)
    children, service_fd = [], None
    # Root owns containment of privileged descendants, not the unprivileged user manager.
    memory_receipt(group)
    baseline = authenticated_baseline(group)
    report['baseline_processes'] = baseline
    kill_fd = open_containment(group, directory)
    deadline = time.monotonic() + 105
    environment = dict(PATH='/usr/bin:/bin', LANG='C', LC_ALL='C',
                       CUDA_VISIBLE_DEVICES=manifest['gpu_uuid'], CUDA_CACHE_DISABLE='1',
                       TMPDIR=str(directory))

    def guard():
        require(time.monotonic() < deadline, 'diagnostic deadline reached')
        memory = memory_receipt(group)
        require(all(int(line.split()[1]) == 0 for line in memory['memory.events'].splitlines()),
                'memory pressure/OOM')
        require(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
                < OUTPUT_MAX - 1024 ** 2, 'diagnostic output bound reached')
        report['last_memory'] = memory
        check_owned_gpu(group, manifest['gpu_uuid'])

    def wait_until(predicate, process, bound=20):
        until = min(deadline, time.monotonic() + bound)
        while not predicate():
            guard()
            require(process.poll() is None and time.monotonic() < until, 'bounded startup failed')
            time.sleep(.05)

    try:
        guard()
        require(not check_owned_gpu(group, manifest['gpu_uuid']), 'idle GPU required')
        require(all(digest(Path(path)) == value for path, value in manifest['seals'].items()),
                'frozen probe input changed')
        with ExitStack() as stack:
            def launch(argv, filename, stdin=None):
                log = stack.enter_context((directory / filename).open('x'))
                process = subprocess.Popen(argv, env=environment, cwd=PROJECT_ROOT, stdin=stdin,
                                           stdout=log, stderr=subprocess.STDOUT)
                children.append(process)
                require(cgroup(process.pid) == group, 'child escaped task unit')
                return process
            client = launch([*drop(manifest), '/usr/bin/python3', '-B', str(SCRIPT),
                             '--client', str(directory)], 'client.log', subprocess.PIPE)
            wait_until(lambda: (directory / 'client-ready.json').is_file(), client)
            client_record = read(directory / 'client-ready.json')
            require(client_record == dict(pid=client.pid, start=identity(client.pid)) and
                    not proc_mapped_paths(client.pid), 'client identity or injection differs')
            target = (capable_service_command(directory, manifest, client_record) if capable else
                      [*drop(manifest), '/usr/bin/python3', '-B', str(SCRIPT), '--service', str(directory)])
            ncu_argv = ncu_probe_command(directory, target, counters)
            report['ncu_command'] = ncu_argv
            ncu = launch(ncu_argv, 'service.log' if capable else 'ncu-parent.log')
            wait_until(lambda: (directory / 'service.log').is_file() and
                       'Execution service ready:' in (directory / 'service.log').read_text(), ncu)
            if capable:
                service, service_fd = discover_service(
                    group, service_command(directory, manifest, client_record), manifest)
                publish_json(directory / 'service-ready.json', service)
            else:
                service = read(directory / 'service-ready.json')
                validate_target(service, manifest, group)
                service_fd = os.pidfd_open(service['pid'])
            validate_target(service, manifest, group)
            report['service'] = service
            report['service_status'] = Path(f"/proc/{service['pid']}/status").read_text()
            report['injection'] = validate_service_environment(service, directory, manifest)
            report['service_maps'] = Path(f"/proc/{service['pid']}/maps").read_text()
            require(not pidfd_terminal(service_fd), 'service died before GDB attach')
            validate_target(service, manifest, group)
            debugger = launch(debugger_command(service['pid']), 'gdb.log')
            wait_until(lambda: 'GLIMMER_GDB_ATTACHED' in (directory / 'gdb.log').read_text(), debugger)
            status = Path(f"/proc/{service['pid']}/status").read_text().splitlines()
            require([line.split()[1] for line in status if line.startswith('TracerPid:')] ==
                    [str(debugger.pid)], 'GDB did not attach to the authenticated service')
            validate_target(service, manifest, group)
            validate_service_environment(service, directory, manifest)
            require(not pidfd_terminal(service_fd), 'service died before client release')
            client.stdin.write((json.dumps(dict(release='start', pid=service['pid'],
                                               start=service['start'])) + '\n').encode())
            client.stdin.flush()
            report['client_released'] = True
            run_sent = False
            while client.poll() is None and ncu.poll() is None:
                guard()
                if not run_sent and 'Ordinary Runtime application ready:' in (
                        directory / 'client.log').read_text():
                    client.stdin.write(b'run\n')
                    client.stdin.flush()
                    client.stdin.close()
                    run_sent = True
                    report['runtime_ready'] = True
                time.sleep(.1)
            # A clean fixture closes its session; --exit-when-idle retires the service.
            for process in (client, ncu, debugger):
                until = min(deadline, time.monotonic() + 8)
                while process.poll() is None and time.monotonic() < until:
                    guard()
                    time.sleep(.1)
            report['returncodes'] = dict(client=client.poll(), ncu=ncu.poll(), gdb=debugger.poll())
            guard()
            report['service_exit'] = read(directory / 'service-exit.json') if (
                directory / 'service-exit.json').exists() else None
            debugger_output = (directory / 'gdb.log').read_text()
            report['segv_observed'] = 'received signal SIGSEGV' in debugger_output
            report['stack_observed'] = re.search(r'^#0\s', debugger_output, re.MULTILINE) is not None
            report['service_terminal'] = pidfd_terminal(service_fd)
            report['compatibility_passed'] = compatibility_passed(
                report, debugger_output, (directory / 'client.log').read_text())
            require((report['segv_observed'] and report['stack_observed']) or
                    report['compatibility_passed'],
                    'neither a captured SIGSEGV nor a complete passing fixture')
            if counters:
                require(report['compatibility_passed'] and not remaining_descendants(group, baseline),
                        'clean fixture retirement required before counter export')
                raw = directory / 'capture.ncu-rep'
                before = bounded_regular_digest(raw)
                report['counter_export_command'] = counter_export_command(directory)
                exporter = launch(report['counter_export_command'], 'counter-metrics.csv')
                while exporter.poll() is None:
                    guard()
                    time.sleep(.05)
                report['counter_export_returncode'] = exporter.returncode
                require(exporter.returncode == 0 and not remaining_descendants(group, baseline),
                        'counter export must retire normally without descendants')
                guard()
                report['counter_artifacts'], report['counter_audit'] = audit_counter_artifacts(
                    directory, service['pid'], before)
                report['counter_collection'] = True
                report['counter_access_verified'] = True
            report['status'] = 'diagnostic_requires_independent_review'
    except BaseException as error:
        report.update(status='failed', counter_access_verified=False,
                      error=f'{type(error).__name__}: {error}')
    finally:
        containment_required = True
        try:
            containment_required = retire_owned(children, service_fd, group, baseline, report)
            if not report['compatibility_passed'] or report['status'] == 'failed':
                report['counter_access_verified'] = False
            save(directory / 'diagnostic.json', report)
        except BaseException:
            containment_required = True
            raise
        finally:
            if containment_required:
                os.write(kill_fd, b'1')
            os.close(kill_fd)
    return int(report['status'] == 'failed' or (capable and not report['compatibility_passed']) or
               (counters and not report['counter_access_verified']))


def collect(service_cap_sys_admin=False, *, counter_probe=False):
    validate_probe_modes(service_cap_sys_admin, counter_probe)
    require(os.geteuid() != 0, 'start as the authorized non-root user')
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector required')
    command(['/usr/bin/sudo', '-n', '/usr/bin/true'])
    gpu = command(['/usr/bin/nvidia-smi', '--query-gpu=uuid,name,driver_version,compute_mode',
                   '--format=csv,noheader,nounits']).strip()
    previous = read(FAILED / 'manifest.json')
    verify_profiler_baseline(previous)
    require(gpu == previous['gpu'], 'failed-candidate GPU identity differs')
    require(not command(['/usr/bin/nvidia-smi', '--query-compute-apps=pid',
                         '--format=csv,noheader,nounits']).strip(), 'idle GPU required')
    frozen = read(FAILED / 'inputs-after.json')
    require(all(digest(path) == frozen[str(path)]['sha256'] for path in (SERVICE, PRELOAD)),
            'production candidate changed after the failure')
    metric_support = None
    if counter_probe:
        verify_initialized_candidate()
        metric_support = offline_metric_support()
    old_job = read(FAILED / 'passes/pass-mfgahnze/equal/attempt.json')
    environment = dict(old_job['server_environment'])
    environment.pop('NVTX_INJECTION64_PATH', None)
    sources = {SCRIPT, SERVICE, CLIENT, PRELOAD, NCU, NCU_EXEC, Path('/usr/bin/gdb'),
               Path('/usr/bin/python3'), Path('/usr/bin/setpriv'), Path('/usr/bin/env'),
               Path(previous['range_library']),
               PROJECT_ROOT / 'tests/examples/execution_ncu_init_probe_test.py',
               PROJECT_ROOT / 'tests/interceptor/execution_binary_kernel_fixture.cu',
               PROJECT_ROOT / 'tests/interceptor/execution_cached_runtime_fixture.h',
               PROJECT_ROOT / 'tests/interceptor/execution_runtime_gpu_test.cu'}
    sources.update(SCRIPT.parent.glob('*.py'))
    sources.update(Path(name) for name in previous['ncu_injection_seals'])
    seals = {str(path): digest(path) for path in sources}
    directory = create_attempt(resolve_artifact_root('ncu-init', category='profiles'))
    print('Initialization diagnostic: ' + str(directory), flush=True)
    manifest = dict(uid=os.getuid(), gid=os.getgid(), groups=[], gpu_uuid=gpu.split(',')[0],
                    service_cap_sys_admin=service_cap_sys_admin,
                    counter_probe=counter_probe,
                    environment=environment, seals=seals, range_library=previous['range_library'],
                    ncu_injection_seals=previous['ncu_injection_seals'])
    save(directory / 'manifest.json', manifest)
    if metric_support is not None:
        save(directory / 'metric-support.json', metric_support)
    unit = unit_name(directory)
    description = 'Glimmer NCU initialization ' + directory.name
    require(unit_state(unit)['LoadState'] == 'not-found', 'never reuse an existing unit')
    argv = ['systemd-run', '--user', '--unit=' + unit, '--description=' + description,
            '--wait', '--pipe', '--collect',
            '--property=RuntimeMaxSec=120s', '--property=TimeoutStopSec=5s',
            '--property=MemoryMax=17179869184', '--property=MemorySwapMax=0',
            '--property=OOMPolicy=kill', '--property=TasksMax=256', '--property=KillMode=control-group',
            '--property=LimitFSIZE=' + str(FILE_MAX), '/usr/bin/sudo', '-n', '/usr/bin/python3',
            '-B', str(SCRIPT), '--root', str(directory)]
    result = dict(status='started', command=argv, compatibility_passed=False, counter_access_verified=False,
                  sm_coverage_verified=False, full_device_qualified=False)
    try:
        with (directory / 'unit.log').open('x') as log:
            process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=140, check=False)
        result['returncode'] = process.returncode
        result['status'] = 'diagnostic_requires_independent_review' if process.returncode == 0 else 'failed'
    except BaseException as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        state = unit_state(unit)
        if state['LoadState'] != 'not-found':
            require(owned_unit(state, unit, description), 'refuse to stop an unowned unit')
            command(['systemctl', '--user', 'stop', unit], timeout=10)
        result['unit_removed'] = unit_state(unit)['LoadState'] == 'not-found'
        result['input_identity_unchanged'] = all(digest(Path(path)) == value for path, value in seals.items())
        if not result['unit_removed'] or not result['input_identity_unchanged']:
            result['status'] = 'failed'
        diagnostic = directory / 'diagnostic.json'
        result['compatibility_passed'] = (result['status'] != 'failed' and diagnostic.is_file() and
                                          read(diagnostic).get('compatibility_passed') is True)
        if counter_probe and result['compatibility_passed']:
            try:
                result['counter_postflight'] = audit_counter_postflight(
                    directory, read(diagnostic), unit_removed=result['unit_removed'],
                    inputs_unchanged=result['input_identity_unchanged'])
                result['counter_access_verified'] = True
            except (OSError, ValueError, KeyError, TypeError) as error:
                result.update(status='failed', counter_access_verified=False,
                              counter_postflight_error=f'{type(error).__name__}: {error}')
        if counter_probe and not result['counter_access_verified']:
            result['status'] = 'failed'
        save(directory / 'report.json', result)
    print(json.dumps(dict(status=result['status'], report=str(directory / 'report.json'))), flush=True)
    return int(result['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--service-cap-sys-admin', action='store_true',
                        help='explicitly authorize only the non-root service to retain CAP_SYS_ADMIN')
    parser.add_argument('--counter-probe', action='store_true',
                        help='collect one non-warmup binary_high launch; requires service capability opt-in')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', action='store_true')
    for role in ('root', 'service', 'client'):
        group.add_argument('--' + role, type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.service_cap_sys_admin and not args.collect:
        parser.error('--service-cap-sys-admin requires --collect')
    if args.counter_probe and (not args.collect or not args.service_cap_sys_admin):
        parser.error('--counter-probe requires --collect --service-cap-sys-admin')
    if args.collect:
        return collect(args.service_cap_sys_admin, counter_probe=args.counter_probe)
    if args.root:
        return supervise(args.root.resolve(strict=True))
    return child((args.service or args.client).resolve(strict=True), 'service' if args.service else 'client')


if __name__ == '__main__':
    raise SystemExit(main())
