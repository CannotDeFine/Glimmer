#!/usr/bin/env python3
"""Bounded native-first fixed cuDNN RNN jobs, each in one capped user unit.

This is not a timing, priority, persistent-RNN or full-device qualification.
The caller must have authorized temporary task-only 16 GiB / zero-swap units.
No sudo, permanent configuration, foreign signals or automatic retry is used.
"""

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
from cudnn8_legacy_audit import audit_job as audit_legacy
from rnn_v8_audit import audit, open_receipt, require

MEMORY_MAX = 16 * 1024 ** 3
JOB_OUTPUT_MAX = 80 * 1024 ** 2
JOB_STREAM_MAX = 512 * 1024 ** 2
CAPTURE_OUTPUT_MAX = 512 * 1024 ** 2
ROUTES = ('native-direct', 'native-cached', 'owned-direct', 'owned-cached')
PROFILES = {
    'sdk9-rnn-v8': dict(namespace='rnn-v8-training', build='build/debug', soname='libcudnn.so.9',
                        target='glimmer_execution_dnn_rnn_weight_params_gpu_test',
                        source='execution_dnn_rnn_weight_params_gpu_test.cc', operations={112: 36}),
    'sdk8-legacy': dict(namespace='cudnn8-arena-requalification', build='build/cudnn8-profile',
                       soname='libcudnn.so.8', target='glimmer_execution_dnn_legacy_sdk_test',
                       source='execution_dnn_legacy_sdk_test.cc', operations={96: 1, 150: 6}),
}


def save(path, data):
    with Path(path).open('w') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def command(args, timeout=5, **kwargs):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout, **kwargs).stdout


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def output_budget(attempt, addition=0):
    # Leave8MiB for final failure/identity/unit receipts even after capture stops.
    used = sum(p.stat().st_size for p in attempt.rglob('*') if p.is_file())
    require(used + addition <= CAPTURE_OUTPUT_MAX - 8 * 1024 ** 2, 'cumulative capture output bound')


def identity(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19])


def cgroup(pid):
    lines = Path(f'/proc/{pid}/cgroup').read_text().splitlines()
    require(len(lines) == 1 and lines[0].startswith('0::/'), 'unified cgroup v2 membership required')
    return lines[0][3:]


def memory_receipt(group):
    root = Path('/sys/fs/cgroup') / group.lstrip('/')
    result = {name: (root / name).read_text().strip() for name in
              ('memory.max', 'memory.swap.max', 'memory.oom.group', 'memory.current', 'memory.events')}
    require(result['memory.max'] == str(MEMORY_MAX) and result['memory.swap.max'] == '0' and
            result['memory.oom.group'] == '1', 'actual aggregate kernel limits are required')
    return result


def query(kind, fields):
    return command(['nvidia-smi', '--id=0', f'--query-{kind}={fields}', '--format=csv,noheader']).strip()


def wait_for_idle(attempt, clock=time.monotonic, sleep=time.sleep):
    """One bounded environmental check:60 consecutive sampled idle seconds, <=10min."""
    started, idle_since = clock(), None
    samples = 0
    with (attempt / 'idle-window.jsonl').open('x') as output:
        while clock() - started < 595:  # Leave the query's5second timeout inside the total bound.
            observed = query('compute-apps', 'gpu_uuid,pid,process_name')
            now = clock()
            idle_since = None if observed else now if idle_since is None else idle_since
            elapsed_idle = 0. if idle_since is None else now - idle_since
            samples += 1
            output.write(json.dumps(dict(monotonic=now, observed=observed, idle_seconds=elapsed_idle)) + '\n')
            output.flush()
            if elapsed_idle >= 60:
                return dict(achieved=True, sampled_idle_seconds=elapsed_idle,
                            waited_seconds=now - started, samples=samples)
            sleep(min(1., max(0., started + 595 - clock())))
    return dict(achieved=False, waited_seconds=clock() - started, samples=samples)


def guard(allowed):
    observed = query('compute-apps', 'gpu_uuid,pid,process_name')
    for line in observed.splitlines():
        fields = line.split(',', 2)
        require(len(fields) == 3, 'parseable GPU ownership')
        pid = int(fields[1])
        require(pid in allowed and identity(pid) == allowed[pid],
                'foreign/caller GPU process observed; never signal it: ' + line)
    return dict(monotonic_ns=time.monotonic_ns(), observed=observed)


def stop_child(process, start):
    if process.poll() is not None:
        return
    require(identity(process.pid) == start, 'owned direct child identity before cleanup')
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        require(identity(process.pid) == start, 'owned direct child identity before kill')
        process.kill()
        process.wait(timeout=3)


def mapped_providers(text, sdk, seals):
    found = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        name = Path(fields[5]).name
        if not name.startswith(('libcudnn', 'libcuda.', 'libcublas', 'libcurand', 'libcudart',
                                'libnvrtc', 'libnccl', 'libglimmer_execution_interceptor')):
            continue
        path = Path(fields[5]).resolve(strict=True)
        require(str(path) in seals, 'unsealed actual GPU/provider mapping: ' + str(path))
        if name.startswith('libcudnn'):
            require(path.parent == sdk / 'lib', 'mixed or wrong-root cuDNN provider')
        found.add(str(path))
    return found


def physical_receipt(log, profile='sdk9-rnn-v8'):
    admissions = re.findall(r'Native admission audit: session=(\d+), sequence=(\d+), '
                            r'accepted=true, operation=(\d+)', log)
    issues = re.findall(r'Native library issue audit: session=(\d+), sequence=(\d+), status=(-?\d+)', log)
    tails = re.findall(r'Native execution audit: session=(\d+), issued=(\d+), '
                       r'retained=(\d+), retired=(\d+), status=(-?\d+)', log)
    accepted = {(s, n): int(op) for s, n, op in admissions}
    require(len(accepted) == len(admissions), 'unique admission identities')
    expected = PROFILES[profile]['operations']
    selected = {key for key, op in accepted.items() if op in expected}
    require(all(sum(op == wanted for op in accepted.values()) == count
                for wanted, count in expected.items()), 'exact profile-specific computation inventory')
    require(len(selected) == sum(expected.values()) and len(issues) == len(selected) and
            {(s, n) for s, n, _ in issues} == selected and all(status == '0' for _, _, status in issues),
            'exact successful issues, no replay or unexplained library issue')
    for s, n in selected:
        found = [r for r in tails if r[:2] == (s, n)]
        require(len(found) == 1 and found[0][2:] == ('0', '1', '0'), 'exact physical retirement')
        require(log.index(f'Native admission audit: session={s}, sequence={n}, ') <
                log.index(f'Native library issue audit: session={s}, sequence={n}, ') <
                log.index(f'Native execution audit: session={s}, issued={n}, '), 'physical record order')
    return dict(rnn_computations=expected.get(112, expected.get(150, 0)),
                dropout_initializations=expected.get(96, 0), identities=sorted(selected))


def validate_application_providers(records, sdk, seals, profile):
    if profile == 'sdk8-legacy':
        observed = [r for r in records if r['kind'] == 'providers']
        require(len(observed) == 1, 'one original SDK8 provider observation')
        paths = {Path(name).resolve(strict=True) for name in observed[0]['paths']}
        require(paths and all(p.parent == sdk / 'lib' and str(p) in seals for p in paths),
                'actual sealed SDK8 application providers')
        return sorted(map(str, paths))
    application_maps = '\n'.join(r['line'] for r in records if r['kind'] == 'mapping')
    paths = mapped_providers(application_maps, sdk, seals)
    require(any(Path(p).name.startswith('libcudnn') for p in paths), 'actual cuDNN mappings saved')
    providers = [r for r in records if r['kind'] == 'provider']
    require(len(providers) == 31 and len({r['symbol'] for r in providers}) == 31,
            'all actual SDK/Driver function providers')
    for provider in providers:
        require(str(Path(provider['path']).resolve(strict=True)) in seals, 'actual function provider is sealed')
    return sorted(paths)


def validate_service_providers(paths, profile):
    if profile == 'sdk8-legacy':
        require(len({p for p in paths if Path(p).name.startswith('libcudnn')}) == 7,
                'all seven selected SDK8 service providers must be captured')
    else:
        require(any(Path(p).name.startswith('libcudnn_adv.') for p in paths),
                'service native RNN provider observed')


def clean_environment(sdk, gpu):
    env = {k: v for k, v in os.environ.items() if k not in ('LD_PRELOAD', 'LD_LIBRARY_PATH') and
           not k.startswith(('GLIMMER_', 'CUDNN_LOG', 'CUDA_INJECTION', 'NSYS_'))}
    env.update(CUDA_VISIBLE_DEVICES=gpu.split(',')[0].strip(),
               LD_LIBRARY_PATH=str(sdk / 'lib') + ':/usr/local/cuda-12.6/lib64')
    return env


def worker(attempt, route):
    """Observer, service and application are all descendants of the same unit."""
    manifest = json.loads((attempt / 'manifest.json').read_text())
    profile = manifest.get('profile', 'sdk9-rnn-v8')
    spec = PROFILES[profile]
    build, sdk = Path(manifest['build']), Path(manifest['sdk'])
    directory = attempt / route
    directory.mkdir()
    report = dict(status='started', profile=profile, normal_unsignalled_cleanup=False, children=[])
    processes = []
    group = cgroup(os.getpid())
    expected_unit = f'glimmer-rnn-{attempt.name}-{route}.service'
    require(group.endswith('/' + expected_unit), 'worker runs inside its exact task unit')
    report['observer'] = dict(pid=os.getpid(), start_ticks=identity(os.getpid()), cgroup=group)
    report['initial_memory'] = memory_receipt(group)
    save(directory / 'job.json', report)
    deadline = time.monotonic() + 215
    output_bytes = 0
    next_observation = 0.
    service = None
    env = clean_environment(sdk, manifest['gpu'])
    cached, owned = route.endswith('cached'), route.startswith('owned')
    socket_directory = Path(tempfile.mkdtemp(prefix='glimmer-rnn-v8-'))
    socket = socket_directory / 'service.sock'
    try:
        guard({})
        require(all(digest(p) == v for p, v in manifest['seals'].items()), 'frozen inputs before job')
        with ExitStack() as stack, selectors.DefaultSelector() as selector:
            samples = stack.enter_context((directory / 'observations.jsonl').open('x'))
            maps_seen = set()

            def launch(args, selected, label):
                # The single-threaded observer checks inheritance in the child
                # before exec, so no GPU API precedes actual limit verification.
                def before_exec():
                    require(cgroup(os.getpid()) == group, 'child inherits exact task cgroup')
                    memory_receipt(group)

                process = subprocess.Popen(args, cwd=PROJECT_ROOT, env=selected, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT if label == 'service' else subprocess.PIPE,
                                           preexec_fn=before_exec)
                start = identity(process.pid)
                processes.append((process, start))
                membership = cgroup(process.pid)
                require(membership == group, 'application/service is inside aggregate cap')
                report['children'].append(dict(label=label, pid=process.pid, start_ticks=start,
                                               cgroup=membership, command=args))
                save(directory / 'job.json', report)
                for stream, suffix in ((process.stdout, 'out'), (process.stderr, 'err')):
                    path = directory / f'{label}.{suffix}'
                    output = stack.enter_context(gzip.open(str(path) + '.gz', 'xb', compresslevel=1)
                                                 if label == 'application' and suffix == 'out'
                                                 else path.open('xb'))
                    if stream is not None:
                        selector.register(stream, selectors.EVENT_READ, (output, label))
                return process

            def observe_mappings():
                if service is None or service.poll() is not None:
                    return
                try:
                    require(identity(service.pid) == processes[0][1], 'live service identity')
                    maps = Path(f'/proc/{service.pid}/maps').read_text()
                except (FileNotFoundError, ProcessLookupError):
                    require(service.poll() is not None, 'live service mapping read failure')
                    return
                observed = mapped_providers(maps, sdk, manifest['seals'])
                if observed - maps_seen:
                    maps_seen.update(observed)
                    require(len(maps.encode()) < 2 * 1024 ** 2, 'bounded provider mappings')
                    output_budget(attempt, len(maps.encode()))
                    name = f'service-{len(maps_seen)}.maps'
                    (directory / name).write_text(maps)
                    with (directory / 'service-provider-observations.jsonl').open('a') as output:
                        output.write(json.dumps(dict(monotonic_ns=time.monotonic_ns(),
                                                     pid=service.pid, start_ticks=processes[0][1],
                                                     maps=name, providers=sorted(observed))) + '\n')

            def observe(allowed):
                nonlocal output_bytes, next_observation
                require(time.monotonic() < deadline, 'whole-job work deadline including normal cleanup')
                output_budget(attempt, 65536 + 32768)
                require(os.statvfs(attempt).f_bavail * os.statvfs(attempt).f_frsize >= 6 * 1024 ** 3,
                        '6GiB free-storage floor')
                for key, _ in selector.select(.004 if profile == 'sdk8-legacy' else .01):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        key.data[0].close()
                        continue
                    require(output_bytes + len(data) <= JOB_STREAM_MAX, 'bounded raw job stream')
                    key.data[0].write(data)
                    key.data[0].flush()
                    output_bytes += len(data)
                stored = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
                require(stored <= JOB_OUTPUT_MAX, 'bounded losslessly stored job output')
                # Observe mappings around potentially blocking ownership queries.
                # Actual timestamps, not a requested interval, establish evidence.
                observe_mappings()
                if time.monotonic() < next_observation:
                    return
                next_observation = time.monotonic() + .2
                sample = guard(allowed)
                sample['memory'] = memory_receipt(group)
                encoded = json.dumps(sample) + '\n'
                require(len(encoded.encode()) <= 32768, 'bounded ownership/memory sample')
                samples.write(encoded)
                samples.flush()
                observe_mappings()

            if owned:
                service = launch([str(build / 'bin/glimmer_execution_service'), '--socket', str(socket),
                                  '--exit-when-idle', '--audit-native', '--persistent-workers', '320'],
                                 env, 'service')
                allowed = {service.pid: processes[0][1]}
                ready_deadline = time.monotonic() + 20
                while True:
                    observe(allowed)
                    log = (directory / 'service.out').read_text() + (directory / 'service.err').read_text()
                    ready = re.search(r'Execution service ready: pid=(\d+), start=(\d+);', log)
                    if ready:
                        require(tuple(map(int, ready.groups())) == (service.pid, processes[0][1]),
                                'authenticated service readiness identity')
                        break
                    require(service.poll() is None and time.monotonic() < ready_deadline, 'service readiness')
                env.update(LD_PRELOAD=str(build / 'lib/libglimmer_execution_interceptor.so'),
                           GLIMMER_EXECUTION_SOCKET=str(socket), GLIMMER_EXECUTION_SERVER_PID=str(service.pid),
                           GLIMMER_EXECUTION_SERVER_START=str(processes[0][1]))
            application = launch([str(build / 'src/interceptor/execution' / spec['target']),
                                  str(sdk / 'lib' / spec['soname']), *(['--cached'] if cached else [])],
                                 env, 'application')
            allowed = ({service.pid: processes[0][1]} if owned else {application.pid: processes[-1][1]})
            while application.poll() is None:
                observe(allowed)
            report['application_returncode'] = application.returncode
            # Drain buffered records before reporting a failure; preserve raw pre-assertion output.
            while any(key.data[1] == 'application' for key in selector.get_map().values()):
                observe(allowed if owned else {})
            require(application.returncode == 0, 'application failed; preserve raw output and stop batch')
            if service is not None:
                while service.poll() is None:
                    observe(allowed)
                report['service_returncode'] = service.returncode
                require(service.returncode == 0 and not socket.exists(), 'normal service/socket cleanup')
            while selector.get_map():
                observe({})
            with open_receipt(directory / 'application.out.gz') as stream:
                records = [json.loads(line) for line in stream]
            report['application_providers'] = validate_application_providers(records, sdk, manifest['seals'], profile)
            report['independent_audit'] = (
                audit_legacy(directory / 'application.out.gz', cached, owned, sdk, manifest['seals'])
                if profile == 'sdk8-legacy' else audit(directory / 'application.out.gz', cached))
            if owned:
                validate_service_providers(maps_seen, profile)
                report['service_providers'] = sorted(maps_seen)
                report['physical_receipt'] = physical_receipt(
                    (directory / 'service.out').read_text() + (directory / 'service.err').read_text(), profile)
            require(query('gpu', 'uuid,name,driver_version,compute_mode') == manifest['gpu'], 'stable GPU identity')
            guard({})
            report.update(status='passed_requires_review', normal_unsignalled_cleanup=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        for process, start in reversed(processes):
            stop_child(process, start)
        report['final_memory'] = memory_receipt(group)
        events = dict(line.split() for line in report['final_memory']['memory.events'].splitlines())
        if any(int(events.get(name, 0)) for name in ('oom', 'oom_kill', 'oom_group_kill')):
            report.update(status='failed', error='task cgroup reports memory exhaustion')
        report['raw_stream_bytes'] = output_bytes
        report['inputs_unchanged'] = all(digest(p) == v for p, v in manifest['seals'].items())
        if not report['inputs_unchanged']:
            report.update(status='failed', error='frozen inputs changed')
        save(directory / 'job.json', report)
        # Only this exact freshly created empty directory is removed. A leftover
        # socket or other artifact is preserved as failure evidence.
        if not any(socket_directory.iterdir()):
            socket_directory.rmdir()
    print(json.dumps(dict(route=route, status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] != 'passed_requires_review')


def seal_inputs(build, sdk, profile='sdk9-rnn-v8'):
    spec = PROFILES[profile]
    paths = [Path(__file__), PROJECT_ROOT / 'scripts/rnn_v8_audit.py',
             PROJECT_ROOT / 'scripts/cudnn8_legacy_audit.py', PROJECT_ROOT / 'scripts/artifact_paths.py',
             PROJECT_ROOT / 'tests/interceptor' / spec['source'],
             PROJECT_ROOT / 'tests/interceptor/rnn_lstm_oracle.h', PROJECT_ROOT / 'CMakeLists.txt',
             PROJECT_ROOT / 'src/backend/execution_submission_queue.cc']
    binaries = [build / 'bin/glimmer_execution_service', build / 'lib/libglimmer_execution_interceptor.so',
                build / 'src/interceptor/execution' / spec['target']]
    paths += binaries
    for name in ('src/backend/cuda', 'src/backend/internal', 'src/control', 'src/core', 'src/execution_service',
                 'src/interceptor/execution', 'src/internal', 'src/transform', 'include/glimmer', 'cmake'):
        paths.extend(p for p in (PROJECT_ROOT / name).rglob('*') if p.is_file())
    paths.extend(p for p in sdk.rglob('*') if p.is_file())
    for prefix in ('libcublas', 'libcudart', 'libcurand', 'libnvrtc', 'libnvJitLink'):
        paths.extend(p for p in Path('/usr/local/cuda-12.6/lib64').glob(prefix + '*.so*') if p.is_file())
    cache = (build / 'CMakeCache.txt').read_text()
    nccl = re.search(r'^GLIMMER_NCCL_ROOT:PATH=(.+)$', cache, re.M)
    if nccl:
        paths.extend(p for p in (Path(nccl[1]) / 'lib').glob('*.so*') if p.is_file())
    for binary in binaries:
        paths.extend(Path(p) for p in re.findall(r'(?:=> )?(/\S+) \(', command(['ldd', str(binary)], 10)))
    return {str(p): digest(p) for p in sorted({p.resolve(strict=True) for p in paths})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), default='sdk9-rnn-v8')
    parser.add_argument('--build', type=Path, help='compatible existing build; default follows profile')
    parser.add_argument('--sdk', type=Path)
    parser.add_argument('--wait-for-idle', action='store_true',
                        help='wait at most10min for60 consecutive sampled idle seconds; one batch only')
    parser.add_argument('--worker-attempt', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-route', choices=ROUTES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_route is not None, 'internal worker route')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_route)
    require(args.sdk is not None, 'explicit selected SDK root is required')
    spec = PROFILES[args.profile]
    build = (args.build or PROJECT_ROOT / spec['build']).resolve(strict=True)
    sdk = args.sdk.resolve(strict=True)
    attempt = create_attempt(resolve_artifact_root(spec['namespace']))
    print(f'Evidence directory: {attempt}', flush=True)
    report = dict(status='started', profile=args.profile, jobs=[], full_device_qualified=False, saturation_qualified=False,
                  persistent_qualified=False, priority_qualified=False, performance_qualified=False,
                  full_goal_qualified=False)
    manifest = None
    try:
        if not args.wait_for_idle:
            guard({})
        gpu = query('gpu', 'uuid,name,driver_version,compute_mode')
        require(len(gpu.splitlines()) == 1 and 'V100' in gpu, 'one selected V100')
        manifest = dict(profile=args.profile, build=str(build), sdk=str(sdk), gpu=gpu,
                        seals=seal_inputs(build, sdk, args.profile))
        save(attempt / 'manifest.json', manifest)
        if args.wait_for_idle:
            report['idle_window'] = wait_for_idle(attempt)
            save(attempt / 'report.json', report)
            require(report['idle_window']['achieved'], '60second idle window unavailable; no experiment started')
        for route in ROUTES:
            guard({})
            unit = f'glimmer-rnn-{attempt.name}-{route}.service'
            state = command(['systemctl', '--user', 'show', unit, '--property=LoadState', '--value']).strip()
            require(state == 'not-found', 'never reuse or replace an existing unit')
            run = ['systemd-run', '--user', '--unit=' + unit, '--wait', '--pipe',
                   '--property=MemoryMax=' + str(MEMORY_MAX), '--property=MemorySwapMax=0',
                   '--property=OOMPolicy=kill', '--property=RuntimeMaxSec=225s',
                   '--property=TimeoutStopSec=5s', '--property=KillMode=control-group',
                   '--property=TasksMax=256', '/usr/bin/env', '-u', 'LD_PRELOAD', '-u', 'LD_LIBRARY_PATH',
                   '/usr/bin/python3', '-B', str(Path(__file__).resolve()), '--worker-attempt', str(attempt),
                   '--worker-route', route]
            with (attempt / (route + '-unit.log')).open('x') as log:
                try:
                    result = subprocess.run(run, stdout=log, stderr=subprocess.STDOUT, timeout=240, check=False)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    command(['systemctl', '--user', 'stop', unit], 10)
                    code = -1
            properties = command(['systemctl', '--user', 'show', unit, '--property=LoadState,ActiveState,'
                                  'SubState,Result,ExecMainCode,ExecMainStatus,MemoryCurrent,MemoryPeak,ControlGroup'])
            record = dict(route=route, returncode=code, unit_properties=properties)
            found_group = re.search(r'^ControlGroup=(/.+)$', properties, re.M)
            if found_group:
                cgroot = Path('/sys/fs/cgroup') / found_group[1].lstrip('/')
                record['terminal_cgroup'] = {name: (cgroot / name).read_text().strip() for name in
                                             ('memory.events', 'memory.peak') if (cgroot / name).exists()}
            report['jobs'].append(record)
            save(attempt / 'report.json', report)
            if 'LoadState=not-found' not in properties:
                command(['systemctl', '--user', 'stop', unit], 10)
                # Failed transient units otherwise remain loaded. Only our unique
                # already-finished task unit is eligible for this cleanup.
                remaining = command(['systemctl', '--user', 'show', unit,
                                     '--property=LoadState,ActiveState'])
                if 'ActiveState=failed' in remaining:
                    command(['systemctl', '--user', 'reset-failed', unit], 5)
            record['unit_removed'] = command(['systemctl', '--user', 'show', unit,
                                               '--property=LoadState', '--value']).strip() == 'not-found'
            require(record['unit_removed'], 'temporary task unit must be gone after cleanup')
            require(code == 0, route + ' failed; no later jobs launched')
            job = json.loads((attempt / route / 'job.json').read_text())
            require(job['status'] == 'passed_requires_review' and job['normal_unsignalled_cleanup'],
                    'complete normal job receipt')
            require(sum(p.stat().st_size for p in attempt.rglob('*') if p.is_file()) <= 512 * 1024 ** 2,
                    'cumulative512MiB output bound')
        report['status'] = 'four_routes_collected_requires_review'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if manifest:
            report['inputs_unchanged'] = all(digest(p) == v for p, v in manifest['seals'].items())
            if not report['inputs_unchanged']:
                report.update(status='failed', error='frozen inputs changed')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), report=str(attempt / 'report.json'))))
    return int(report['status'] == 'failed')


if __name__ == '__main__':
    raise SystemExit(main())
