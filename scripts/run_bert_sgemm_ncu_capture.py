#!/usr/bin/env python3
"""Bounded whole-application NCU capture for one fixed BERT SGEMM.

This is diagnostic evidence only until independent review and native
operation matching. Never interpret the capture as a timing comparison.
"""

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

from bert_sgemm_coverage_candidates import (authenticated_training_sessions, targeted_training_binding, targeted_issue_range,
                                             targeted_ncu_metrics)
from artifact_paths import create_attempt, resolve_artifact_root
from bert_sgemm_ncu_process import (NCU_INJECTION_NAMES, authenticated_profile_processes,
                                    proc_mapped_paths, verified_preload)
from bert_ncu_client_broker import (BrokerConnection, ClientBroker, broker_client_launches,
                                    clean_client_environment)
from execution_semantic_audit import parse_semantic_audit
import run_bert_candidate_validation as candidate
from run_bert_sgemm_descriptor_probe import (FLAGS, ncu_environment, parse_target,
                                               selected_service_target)
from run_rnn_v8_validation import cgroup, command, digest, identity, memory_receipt, save


NCU = Path('/opt/nvidia/nsight-compute/2024.3.2/ncu')
NCU_EXEC = NCU.parent / 'target/linux-desktop-glibc_2_11_3-x64/ncu'
TARGET_RANGE = 'glimmer.sgemm.target.issue.v1/'
METRICS = ('launch__sm_count', 'launch__grid_size', 'launch__block_size',
           'sm__cycles_active.min', 'sm__sass_thread_inst_executed_op_ffma_pred_on.min')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def publish_json(path, data):
    """Atomically publish a new handshake without exposing partial JSON."""
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            os.fchmod(stream.fileno(), 0o644)
            json.dump(data, stream)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.unlink(temporary)


def validate_unprivileged_process(manifest, status, uids, gids, groups):
    """Reject privilege or group drift before any application can start."""
    require(type(manifest['uid']) is int and type(manifest['gid']) is int and
            manifest['uid'] > 0 and uids == (manifest['uid'],) * 3 and
            gids == (manifest['gid'],) * 3,
            'profile worker must run as the sealed non-root user')
    require(type(manifest['groups']) is list and sorted(groups) == manifest['groups'],
            'profile worker supplemental groups differ')
    fields = {}
    for line in status.splitlines():
        name, separator, value = line.partition(':')
        if separator:
            require(name not in fields, 'duplicate process status field')
            fields[name] = value.strip()
    for name in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb'):
        require(name in fields and re_full_zero_hex(fields[name]), 'profile worker has capabilities')
    require(fields.get('NoNewPrivs') == '1', 'profile worker must prevent new privileges')


def re_full_zero_hex(value):
    return bool(value) and all(char == '0' for char in value)


def task_ancestors(group):
    """Record PID/start ancestry without needing to read root-owned exe links."""
    chain, pid = [], os.getppid()
    for _ in range(32):
        if pid <= 1 or cgroup(pid) != group:
            break
        data = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        require(len(data) >= 20 and data[1].isdecimal() and data[19].isdecimal(),
                'invalid ancestor process identity')
        chain.append(dict(pid=pid, start=int(data[19])))
        pid = int(data[1])
    require(chain, 'profile worker has no task-local ancestor')
    return chain


def observed_ncu_processes(group):
    """Privileged sampler identifies the real NCU executable in this unit."""
    result = []
    members = Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.procs'
    expected = NCU_EXEC.stat()
    for item in members.read_text().splitlines():
        require(item.isdecimal(), 'invalid task cgroup member')
        pid = int(item)
        try:
            executable = Path(f'/proc/{pid}/exe')
            stat = executable.stat()
            if (stat.st_dev, stat.st_ino) != (expected.st_dev, expected.st_ino):
                continue
            fields = Path(f'/proc/{pid}/status').read_text().splitlines()
            uid = [line.split()[1:5] for line in fields if line.startswith('Uid:')]
            require(uid == [['0', '0', '0', '0']] and cgroup(pid) == group,
                    'actual NCU process must remain privileged in the task unit')
            result.append(dict(pid=pid, start=identity(pid)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return sorted(result, key=lambda item: item['pid'])


def task_unit_paths(group, outer):
    """Resolve only this unique unit's containment controls."""
    require(group.endswith(f'/glimmer-bert-{outer.name}-equal.service'),
            'profiler must be inside its exact task-only unit')
    root = Path('/sys/fs/cgroup').resolve(strict=True)
    directory = (root / group.lstrip('/')).resolve(strict=True)
    require(directory.is_relative_to(root) and directory != root,
            'task cgroup cannot be the cgroup root')
    kill = directory / 'cgroup.kill'
    descriptor = os.open(kill, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        expected, actual = kill.stat(), os.fstat(descriptor)
        require((expected.st_dev, expected.st_ino) == (actual.st_dev, actual.st_ino),
                'task containment control changed while opening')
    except BaseException:
        os.close(descriptor)
        raise
    return directory, descriptor


def await_outer_handshake(outer, group, directory, *, fresh=False):
    """Require the unprivileged parent to independently identify our unit."""
    metadata = directory.stat()
    ready = dict(pid=os.getpid(), start=identity(os.getpid()), group=group,
                 cgroup_device=metadata.st_dev, cgroup_inode=metadata.st_ino)
    publish_json(outer / 'ncu-ready.json', ready)
    until = time.monotonic() + 30
    path = outer / 'ncu-go.json'
    while not path.exists():
        require(time.monotonic() < until, 'outer unit authentication timed out')
        time.sleep(.05)
    go = json.loads(path.read_text())
    fields = set(ready) | {'unit_started_us'} | ({'fresh_claim_sha256'} if fresh else set())
    require(type(go) is dict and set(go) == fields and
            all(go[name] == value for name, value in ready.items()) and
            type(go['unit_started_us']) is int and go['unit_started_us'] > 0,
            'outer unit identity handshake differs')
    if fresh:
        require(type(go['fresh_claim_sha256']) is str and
                re.fullmatch(r'[0-9a-f]{64}', go['fresh_claim_sha256']) is not None,
                'fresh one-shot claim digest required')
    started = go['unit_started_us'] / 1_000_000
    require(started <= time.monotonic() < started + 780, 'invalid or exhausted unit deadline')
    return (started + 780, go['fresh_claim_sha256']) if fresh else started + 780


def require_retired_descendants(group):
    """Do not accept a phase that left any of its own processes running."""
    allowed = {os.getpid()} | {entry['pid'] for entry in task_ancestors(group)}
    members = Path('/sys/fs/cgroup') / group.lstrip('/') / 'cgroup.procs'
    observed = members.read_text().splitlines()
    require(all(item.isdecimal() and int(item) in allowed for item in observed),
            'profiler phase left task-owned descendant processes alive')


def profiled_pass(outer):
    """Run one complete fresh replay; callers must authenticate the NCU parent."""
    outer = outer.resolve(strict=True)
    report, guard, directory, broker = dict(status='started', **FLAGS), None, None, None
    try:
        manifest = json.loads((outer / 'manifest.json').read_text())
        require(not os.environ.get('LD_AUDIT'), 'LD_AUDIT is forbidden')
        validate_unprivileged_process(manifest, Path('/proc/self/status').read_text(),
                                      os.getresuid(), os.getresgid(), os.getgroups())
        guard = candidate.ResourceGuard(outer, 'equal')
        report['controller_identity'] = dict(pid=os.getpid(), start=identity(os.getpid()))
        report['ancestors'] = task_ancestors(guard.group)
        passes = outer / 'passes'
        require(passes.is_dir() and not passes.is_symlink(), 'owned replay output directory required')
        directory = Path(tempfile.mkdtemp(prefix='pass-', dir=passes))
        report['pass_directory'] = str(directory)
        seals = manifest['ncu_injection_seals']
        mapped = proc_mapped_paths(os.getpid())
        report['injection_observation'] = dict(
            preload_present='LD_PRELOAD' in os.environ,
            nvtx_injection_expected=os.environ.get('NVTX_INJECTION64_PATH') == 'libcuda-injection.so',
            known_mappings=mapped)
        report['worker_ncu_injection'] = verified_preload(
            os.environ.get('LD_PRELOAD'), None, seals, mapped)
        require(os.environ.get('NVTX_INJECTION64_PATH') == 'libcuda-injection.so',
                'actual profiler NVTX injection is missing')
        provider = Path(manifest['range_library'])
        require(provider.is_absolute() and provider == provider.resolve(strict=True) and
                provider.is_file() and hashlib.sha256(provider.read_bytes()).hexdigest() ==
                manifest['range_library_sha256'], 'sealed range provider differs')
        session, sequence = parse_target(manifest['sgemm_target'])
        common, h = candidate.load_helpers()
        h.startup_reference, h.export = common.application_references(h, manifest)
        gpu = manifest['gpu'].split(',')[0].strip()
        require(h.query('gpu', h.training.GPU_FIELDS) == manifest['gpu'], 'GPU identity changed')
        environment = ncu_environment(h.training.clean_environment(gpu), os.environ)
        native = candidate.EXISTING_NATIVE / 'native'
        ready = json.loads((outer / 'broker-ready.json').read_text())
        broker = BrokerConnection(outer, directory, ready)
        with broker_client_launches(common, h, broker, clean_client_environment(environment)), \
                authenticated_profile_processes(common, seals, broker.processes), \
                selected_service_target(common, manifest['sgemm_target'], provider):
            job = candidate.guarded_control(common, h, directory, 'equal', environment, guard, native)
        report['job'] = job
        report['broker_clients'] = broker.finish()
        broker.close()
        broker = None
        report['reparse'] = candidate.reparse(common, h, directory, 'equal', job, gpu, native)
        argv = job['server_command']
        require(argv.count('--audit-sgemm-target') == 1 and
                argv[argv.index('--audit-sgemm-target') + 1] == manifest['sgemm_target'] and
                argv.count('--audit-sgemm-range-library') == 1 and
                argv[argv.index('--audit-sgemm-range-library') + 1] == str(provider),
                'authenticated service must use the exact target and provider')
        log = directory / 'equal/server.log'
        require(log.stat().st_size <= 64 * 1024 * 1024, 'service audit exceeds parser capacity')
        data = log.read_bytes()
        require(len(data) <= 64 * 1024 * 1024 and
                hashlib.sha256(data).hexdigest() == job['artifact_identities']['server.log'],
                'service audit identity differs')
        content = data.decode('utf-8')
        semantic = parse_semantic_audit(content, require_complete=True)
        require(session in authenticated_training_sessions(semantic, job),
                'selected command must belong to the authenticated training process')
        report['range'] = targeted_issue_range(content, session, sequence)
        guard.check(force=True)
        report['status'] = 'pass_requires_profiler_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if broker is not None:
            broker.close()
        if guard is not None:
            report['resources'] = guard.receipt
        if directory is not None:
            save(directory / 'pass.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'),
                          directory=str(directory) if directory else None)), flush=True)
    return int(report['status'] != 'pass_requires_profiler_review')


def ncu_command(outer, manifest):
    """Strict whole-application replay, with BERT explicitly dropped to cdf."""
    import bert_ncu_replay_exec as bridge

    require(type(manifest['uid']) is int and manifest['uid'] > 0 and type(manifest['gid']) is int and
            manifest['gid'] >= 0 and manifest['groups'] == [],
            'profile worker must be non-root without supplemental groups')
    python = Path(manifest['python'])
    require(python.is_absolute() and python.is_file(), 'sealed application Python required')
    return [str(NCU), '--config-file', '0', '--target-processes', 'all', '--nvtx',
            '--nvtx-include', TARGET_RANGE, '--kernel-name-base', 'function', '--kernel-name',
            bridge.capture_kernel(bridge.selected_signal_binding(manifest)),
            '--replay-mode', 'application', '--app-replay-mode', 'strict',
            '--app-replay-match', 'all',
            '--kill', 'no', '--clock-control', 'none', '--cache-control', 'none',
            '--metrics', ','.join(METRICS), '--log-file', str(outer / 'ncu.log'),
            '--export', str(outer / 'capture'), '/usr/bin/setpriv',
            '--reuid=' + str(manifest['uid']), '--regid=' + str(manifest['gid']),
            '--clear-groups', '--inh-caps=-all', '--ambient-caps=-all', '--bounding-set=-all',
            '--no-new-privs', str(python), '-B', str(Path(__file__).resolve()), '--pass', str(outer)]


def check_owned_gpu(group, gpu_uuid):
    """Reject foreign GPU contexts; never signal their processes."""
    result = subprocess.run(['/usr/bin/nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=10, check=True)
    observed = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(',', 2)]
        require(len(parts) == 3 and parts[1].isdecimal() and parts[0] == gpu_uuid,
                'unexpected GPU inventory')
        pid = int(parts[1])
        require(pid > 0 and cgroup(pid) == group, 'foreign GPU process; never signal it')
        observed.append(dict(pid=pid, name=parts[2]))
    return observed


def monitored_command(command, environment, outer, group, deadline, log_path, gpu_uuid, broker=None):
    """Bound every profiler phase within the same cgroup and output budget."""
    def sample():
        require(time.monotonic() < deadline, 'whole NCU batch exceeded its deadline')
        if broker is not None:
            process, ready = broker
            require(process.poll() is None and identity(process.pid) == ready['start'] and
                    cgroup(process.pid) == group and not proc_mapped_paths(process.pid),
                    'authenticated clean client broker stopped or changed during capture')
        memory = memory_receipt(group)
        events = dict(line.split() for line in memory['memory.events'].splitlines())
        require(events and all(int(value) == 0 for value in events.values()),
                'task memory pressure or OOM')
        return dict(monotonic_ns=time.monotonic_ns(), gpu=check_owned_gpu(group, gpu_uuid),
                    ncu=observed_ncu_processes(group), output_bytes=candidate.check_output(outer))

    samples = [sample()]
    with log_path.open('x') as log:
        process = subprocess.Popen(command, cwd=str(outer), env=environment,
                                   stdout=log, stderr=subprocess.STDOUT)
        require(cgroup(process.pid) == group, 'profiler escaped task cgroup')
        while True:
            observed = sample()
            if samples[-1]['gpu'] != observed['gpu'] or samples[-1]['ncu'] != observed['ncu']:
                samples.append(observed)
            status = process.poll()
            if status is not None:
                require(status == 0, 'NCU phase exited unsuccessfully')
                return dict(returncode=status, samples=samples)
            time.sleep(.2)


def broker_worker(outer):
    """Unprivileged sibling of NCU, contained by the same experiment unit."""
    outer = outer.resolve(strict=True)
    report, broker = dict(status='started'), None
    try:
        manifest = json.loads((outer / 'manifest.json').read_text())
        validate_unprivileged_process(manifest, Path('/proc/self/status').read_text(),
                                      os.getresuid(), os.getresgid(), os.getgroups())
        report['resources'] = memory_receipt(cgroup(os.getpid()))
        require(resource.getrlimit(resource.RLIMIT_FSIZE) == (candidate.FILE_MAX, candidate.FILE_MAX),
                'broker file size limit differs')
        report['file_limit'] = candidate.FILE_MAX
        require(cgroup(os.getpid()).endswith(f'/glimmer-bert-{outer.name}-equal.service'),
                'broker must remain in its exact task unit')
        go = json.loads((outer / 'ncu-go.json').read_text())
        deadline = go['unit_started_us'] / 1_000_000 + 780
        common, h = candidate.load_helpers()
        environment = h.training.clean_environment(manifest['gpu'].split(',')[0].strip())
        script = str(Path(__file__).resolve())
        broker = ClientBroker(outer, [manifest['python'], '-B', '-c', h.wrapper],
                              environment, common.ROOT,
                              [manifest['python'], '-B', script, '--pass', str(outer)], deadline)
        broker.serve(0, lambda ready: publish_json(outer / 'broker-ready.json', ready))
        report['status'] = 'retired'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if broker is not None:
            report['sessions'] = broker.sessions
        save(outer / 'broker.json', report)
    return int(report['status'] != 'retired')


@contextmanager
def managed_client_broker(outer, manifest, environment, group, deadline):
    """Root owns the broker's true wait status; its clients never enter NCU ancestry."""
    # Reuse the same privilege drop as the NCU application controller.
    recipe = ncu_command(outer, manifest)
    drop = recipe[recipe.index('/usr/bin/setpriv'):]
    drop[-2] = '--broker'
    process = None
    with (outer / 'broker.log').open('x') as log:
        try:
            process = subprocess.Popen(drop, env=environment, cwd=outer, stdin=subprocess.PIPE,
                                       stdout=log, stderr=subprocess.STDOUT)
            require(cgroup(process.pid) == group, 'broker escaped the task unit')
            path = outer / 'broker-ready.json'
            until = min(deadline, time.monotonic() + 20)
            while not path.exists():
                require(process.poll() is None and time.monotonic() < until,
                        'broker failed bounded startup')
                time.sleep(.02)
            ready = json.loads(path.read_text())
            require(set(ready) == {'pid', 'start', 'group', 'socket_device', 'socket_inode'} and
                    ready['pid'] == process.pid and ready['start'] == identity(process.pid) and
                    ready['group'] == group and not proc_mapped_paths(process.pid),
                    'broker readiness identity or injection differs')
            validate_unprivileged_process(manifest, Path(f'/proc/{process.pid}/status').read_text(),
                                          (manifest['uid'],) * 3, (manifest['gid'],) * 3, [])
            status = Path(f'/proc/{process.pid}/status').read_text().splitlines()
            require([line.split()[1:5] for line in status if line.startswith('Uid:')] ==
                    [[str(manifest['uid'])] * 4] and
                    [line.split()[1:5] for line in status if line.startswith('Gid:')] ==
                    [[str(manifest['gid'])] * 4] and
                    [line.split()[1:] for line in status if line.startswith('Groups:')] == [[]],
                    'broker actual UID/GID/groups differ')
            yield process, ready
        finally:
            if process is not None:
                process.stdin.close()
                status = process.wait(timeout=10)
                require(status == 0, 'client broker failed; inspect broker.json')


def root_supervisor(outer):
    """Run NCU only inside the pre-authorized user unit; no system changes."""
    outer = outer.resolve(strict=True)
    report, kill_fd, started, previous_umask = dict(status='started', **FLAGS), None, False, None
    try:
        require(os.getresuid() == (0, 0, 0) and not os.environ.get('LD_PRELOAD') and
                not os.environ.get('LD_AUDIT'), 'clean privileged profiler supervisor required')
        # The attempt directory is private to the invoking user (0700), while
        # root-owned evidence inside it must remain readable by that user.
        previous_umask = os.umask(0o022)
        resource.setrlimit(resource.RLIMIT_FSIZE, (candidate.FILE_MAX, candidate.FILE_MAX))
        require(resource.getrlimit(resource.RLIMIT_FSIZE) == (candidate.FILE_MAX, candidate.FILE_MAX),
                'sudo-reset file size limit was not restored')
        group = cgroup(os.getpid())
        directory, kill_fd = task_unit_paths(group, outer)
        memory_receipt(group)
        require_retired_descendants(group)
        import bert_ncu_replay_exec as bridge
        manifest_path = outer / 'manifest.json'
        manifest_owner = manifest_path.lstat()
        manifest_bytes = bridge._owned_bounded_file(manifest_path, manifest_owner.st_uid, manifest_owner.st_gid,
                                                    bridge.MESSAGE_MAX)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = bridge.decode(manifest_bytes)
        require(manifest['uid'] > 0 and manifest['gid'] >= 0, 'root BERT is forbidden')
        work = outer / 'ncu-tmp'
        require(work.is_dir() and not work.is_symlink(), 'bounded profiler temporary directory required')
        if 'fresh_runtime' in manifest:
            deadline, claim_sha256 = await_outer_handshake(outer, group, directory, fresh=True)
            sudo_uid, sudo_gid = os.environ.get('SUDO_UID', ''), os.environ.get('SUDO_GID', '')
            require(re.fullmatch(r'[0-9]+', sudo_uid) is not None and
                    re.fullmatch(r'[0-9]+', sudo_gid) is not None, 'sudo caller identity required')
            authority = bridge.verify_fresh_claim(outer, claim_sha256, int(sudo_uid), int(sudo_gid),
                                                   minimum_remaining=780)
            require(authority['manifest_sha256'] == manifest_sha256,
                    'fresh manifest changed during supervisor authentication')
            manifest = authority['template']
            bridge.consume_fresh_claim(outer, authority)
            bridge.verify_root_claim_receipt(outer, authority)
            report['fresh_claim_sha256'] = authority['claim_sha256']
            report['fresh_manifest_sha256'] = authority['manifest_sha256']
            report['fresh_binding_sha256'] = authority['binding_sha256']
            report['fresh_claim_receipt_verified'] = True
        else:
            deadline = await_outer_handshake(outer, group, directory)
        require(digest(NCU) == manifest['ncu_sha256'] and
                digest(NCU_EXEC) == manifest['ncu_exec_sha256'],
                'NCU launcher or executable differs from the frozen manifest')
        environment = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                           TMPDIR=str(work), CUDA_VISIBLE_DEVICES=manifest['gpu'].split(',')[0].strip(),
                           CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_CACHE_DISABLE='1')
        gpu_uuid = manifest['gpu'].split(',')[0].strip()
        started = True
        if manifest.get('service_cap_sys_admin') is True:
            import bert_ncu_relay
            if 'fresh_runtime' in manifest:
                report['capture'] = bert_ncu_relay.capture(sys.modules[__name__], outer, manifest, environment,
                                                           group, deadline, fresh_authority=authority)
            else:
                report['capture'] = bert_ncu_relay.capture(sys.modules[__name__], outer, manifest, environment,
                                                           group, deadline)
            report['topology'] = 'clean-ncu-sibling'
        else:
            with managed_client_broker(outer, manifest, environment, group, deadline) as broker:
                report['capture'] = monitored_command(ncu_command(outer, manifest), environment, outer, group,
                                                       deadline, outer / 'ncu-supervisor.log', gpu_uuid, broker)
            broker_receipt = json.loads((outer / 'broker.json').read_text())
            require(broker_receipt['status'] == 'retired' and not (outer / 'broker.sock').exists(),
                    'client broker must retire before exports')
            report['broker'] = broker_receipt
        require_retired_descendants(group)
        raw = outer / 'capture.ncu-rep'
        require(raw.is_file() and not raw.is_symlink(), 'NCU report is missing')
        export = [str(NCU), '--config-file', '0', '--import', str(raw), '--csv', '--page', 'raw',
                  '--print-units', 'base', '--print-kernel-base', 'function']
        report['export'] = monitored_command(export, environment, outer, group, deadline,
                                              outer / 'ncu-metrics.csv', gpu_uuid)
        require_retired_descendants(group)
        report['range_export'] = monitored_command([*export, '--print-nvtx-rename', 'kernel'],
                                                   environment, outer, group, deadline,
                                                   outer / 'ncu-range-metrics.csv', gpu_uuid)
        require_retired_descendants(group)
        report['status'] = 'capture_requires_replay_and_counter_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        if started and kill_fd is not None:
            # This exact private unit contains only our experiment processes.
            # Writing cgroup.kill terminates this supervisor as well, so persist
            # the reason first. The outer unit manager independently verifies
            # retirement and never signals an unrelated GPU process.
            try:
                save(outer / 'ncu-supervisor.json', report)
            except BaseException as save_error:
                print(f'failed to save profiler failure: {save_error}', flush=True)
            finally:
                try:
                    os.write(kill_fd, b'1')
                except BaseException as kill_error:
                    report['containment_error'] = f'{type(kill_error).__name__}: {kill_error}'
                    print(f'task containment failed: {kill_error}', flush=True)
    try:
        if kill_fd is not None:
            os.close(kill_fd)
        save(outer / 'ncu-supervisor.json', report)
    finally:
        if previous_umask is not None:
            os.umask(previous_umask)
    return int(report['status'] == 'failed')


def fresh_capture_service(manifest, supervisor, replay, inputs, binding, claim, manifest_sha256):
    """Check the completed replay against the collector's authenticated inputs."""
    require(type(binding) is dict and type(claim) is dict and
            type(claim.get('sha256')) is str and re.fullmatch(r'[0-9a-f]{64}', claim['sha256']) and
            manifest.get('fresh_runtime') == binding and
            supervisor.get('fresh_claim_receipt_verified') is True and
            supervisor.get('fresh_claim_sha256') == claim.get('sha256') and
            supervisor.get('fresh_manifest_sha256') == manifest_sha256 and
            replay['plan'].get('fresh_claim_sha256') == claim.get('sha256') and
            replay['plan'].get('template_sha256') == manifest_sha256 and
            (replay['plan'].get('uid'), replay['plan'].get('gid')) == (manifest['uid'], manifest['gid']) and
            replay['plan'].get('environment') == manifest['service_environment'] and
            replay['plan'].get('seals') == manifest['bridge_seals'],
            'completed fresh replay authority differs')
    encoded = json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    require(supervisor.get('fresh_binding_sha256') == hashlib.sha256(encoded).hexdigest(),
            'completed fresh replay binding differs')
    product = binding['contract']['runtime']['service']
    service = Path(product['path'])
    require(service.is_absolute() and str(service) in inputs and
            inputs[str(service)]['target'] == str(service) and
            inputs[str(service)]['sha256'] == product['sha256'] ==
            manifest['bridge_seals'].get(str(service)), 'completed fresh replay service seal differs')
    return service


def review_capture(outer, target, *, fresh_binding=None, fresh_claim=None):
    """Join strict replay receipts to one NCU launch; never accept comparisons."""
    session, sequence = parse_target(target)
    manifest = json.loads((outer / 'manifest.json').read_text())
    import bert_ncu_replay_exec as bridge
    signal_binding = bridge.selected_signal_binding(manifest)
    target_kernel = bridge.capture_kernel(signal_binding)
    supervisor = json.loads((outer / 'ncu-supervisor.json').read_text())
    require(supervisor['status'] == 'capture_requires_replay_and_counter_review' and
            supervisor['capture']['returncode'] == 0 and supervisor['export']['returncode'] == 0 and
            supervisor['range_export']['returncode'] == 0,
            'privileged capture and export must complete normally')
    ncu_processes = {(entry['pid'], entry['start']) for sample in supervisor['capture']['samples']
                     for entry in sample['ncu']}
    require(ncu_processes, 'actual NCU executable was not observed by the privileged supervisor')
    sibling = supervisor.get('topology') == 'clean-ncu-sibling'
    require('fresh_runtime' not in manifest or sibling, 'fresh replay requires sibling topology')
    sessions = supervisor['capture']['sessions'] if sibling else supervisor['broker']['sessions']
    broker = None if sibling else supervisor['broker']
    require(sessions and (sibling or (broker['status'] == 'retired' and
            all(session['status'] == 'complete' and
                all(value['returncode'] == 0 and value['exit_code_observed'] is True
                    for value in session['clients'].values()) for session in sessions))),
            'every broker replay must have real successful client wait statuses')
    log = outer / 'ncu.log'
    require(log.stat().st_size <= 64 * 1024 * 1024, 'NCU log exceeds bound')
    text = log.read_text()
    matches = re.findall(r'^==PROF== Profiling "([^"]+)": Application replay pass ([1-9][0-9]*)$',
                         text, re.MULTILINE)
    require(all(name == target_kernel for name, _ in matches), 'replay log names a different kernel')
    passes = [int(value) for _, value in matches]
    require(passes and passes == list(range(1, len(passes) + 1)),
            'one ordered profiled launch per complete application replay required')
    entries = sorted((outer / 'passes').iterdir())
    require(len(entries) == len(passes) and all(path.is_dir() and not path.is_symlink() and
            path.name.startswith('pass-') for path in entries), 'replay output inventory differs')
    broker_sessions = {(session['plan']['directory'] if sibling else session['directory']): session for session in sessions}
    require(len(broker_sessions) == len(entries) == len(sessions) and
            set(broker_sessions) == {str(directory) for directory in entries},
            'broker and NCU replay directory inventories differ')
    identities, descriptors, client_identities, controllers, bindings, connection_orders = [], [], [], [], [], []
    expected_reparse = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                            source_bound_complete_inference=True, first_command_handoff=False)
    for directory in entries:
        receipt = json.loads((directory / 'pass.json').read_text())
        require(receipt['status'] == 'pass_requires_profiler_review' and
                receipt['pass_directory'] == str(directory) and receipt['reparse'] == expected_reparse,
                'every replay needs its own full numerical receipt')
        job = receipt['job']
        controller = receipt['controller_identity']
        replay = broker_sessions[str(directory)]
        if sibling:
            from bert_ncu_relay import complete_reparse, validate_mapping_receipts
            import bert_ncu_replay_exec as bridge
            inputs = json.loads((outer / 'inputs-before.json').read_text())
            if 'fresh_runtime' in manifest:
                service = fresh_capture_service(manifest, supervisor, replay, inputs, fresh_binding, fresh_claim,
                                                digest(outer / 'manifest.json'))
            else:
                require(fresh_binding is None and fresh_claim is None, 'legacy review cannot accept fresh authority')
                service = bridge.selected_service(manifest)
            validate_mapping_receipts(replay, inputs, service=service)
            require(receipt['topology'] == 'clean-ncu-sibling' and replay['controller_returncode'] == 0 and
                    type(replay['controller_returncode']) is int and complete_reparse(receipt['reparse']) and
                    controller == replay['controller'] and job['server_identity'] == replay['service'] and
                    job['application_identities'] == replay['plan']['clients'] and
                    replay['receipt']['returncode'] == 0 and replay['receipt']['exit_code_observed'] is True and
                    digest(directory / 'pass.json') == replay['complete']['report_sha256'] and
                    any((entry['pid'], entry['start']) in ncu_processes for entry in replay['helper_ancestors']) and
                    not any((entry['pid'], entry['start']) in ncu_processes for entry in receipt['ancestors']),
                    'sibling/controller/helper ancestry or complete replay receipt differs')
            clients = job['application_identities']
            if 'fresh_runtime' in manifest:
                from bert_ncu_service_proxy import validate_frontend_receipts, validate_launch_order
                contract = manifest['fresh_runtime']['contract']
                validate_frontend_receipts(contract['runtime']['frontend'], contract['service_environment'],
                                           job, receipt['frontend_mappings'])
                validate_launch_order(job, receipt['launch_order'])
        else:
            require(any((entry['pid'], entry['start']) in ncu_processes
                        for entry in receipt['ancestors']), 'replay is not a child of the observed NCU')
            clients = receipt['broker_clients']
            require(controller == replay['controller'] and clients == replay['clients'] == replay['retired_clients'] and
                    set(clients) == {'low', 'high'} and
                    {role: dict(pid=value['pid'], start=value['start']) for role, value in clients.items()}
                    == job['application_identities'] and
                    {role: value['returncode'] for role, value in clients.items()} == job['application_returncodes'],
                    'broker clients are not the actual successful replay applications')
        controllers.append((controller['pid'], controller['start']))
        client_identities.extend((value['pid'], value['start']) for value in clients.values())
        candidate.normal_cleanup(job)
        if 'host_pair' in manifest:
            from bert_ncu_host_pair import validate_job
            validate_job(manifest['host_pair'], job)
        elif 'current_sdk' in manifest:
            from bert_ncu_current_sdk import validate_job
            validate_job(manifest['current_sdk'], job)
        require(job['mode'] == 'equal' and
                all(digest(directory / 'equal' / name) == value
                    for name, value in job['artifact_identities'].items()),
                'replay artifact identities changed')
        service_log = directory / 'equal/server.log'
        require(service_log.stat().st_size <= 64 * 1024 * 1024, 'service log exceeds bound')
        content = service_log.read_text()
        if 'current_sdk' in manifest:
            require(content.count('GEMM signal diagnostic: sharded=true') ==
                    (manifest['current_sdk']['variant'] == 'sharded'), 'replay signal selector differs')
        semantic = parse_semantic_audit(content, require_complete=True)
        if 'fresh_runtime' in manifest:
            roles = {identity['pid']: role for role, identity in job['application_identities'].items()}
            connections = [(roles[record.value('pid')], record.value('connection'))
                           for record in semantic.family(4) if record.value('event') == 1 and
                           record.value('role') == 0 and record.value('pid') in roles]
            require(connections and connections[0][0] == 'low' and
                    any(role == 'high' for role, _ in connections), 'replay connection order differs')
            connection_orders.append(tuple(connections))
        if sibling:
            binding = targeted_training_binding(content, semantic, job, sequence, replay['plan']['uid'])
            require(binding == receipt['target_binding'], 'replay target binding changed')
            if bridge.capture_specialization(signal_binding) == 'ordinary':
                from bert_ncu_service_proxy import validate_ordinary_release
                require(validate_ordinary_release(content, job, binding) == receipt.get('logical_release'),
                        'ordinary release evidence changed')
            bindings.append(binding)
            session = binding['session']
        require(session in authenticated_training_sessions(semantic, job),
                'target must belong to the replay training process')
        selected = targeted_issue_range(content, session, sequence)
        require(selected == receipt['range'], 'replay range metadata changed')
        descriptor = selected['descriptor']
        descriptors.append(tuple(descriptor[key] for key in
                                 ('transa', 'transb', 'm', 'n', 'k', 'lda', 'ldb', 'ldc',
                                  'device_scalars', 'logical_blocks', 'workers')))
        identities.append(job['server_identity'])
    require(len(set(descriptors)) == 1 and
            descriptors[0] == (0, 0, 3072, 768, 768, 3072, 768, 3072, 0, 2304, 320),
            'replay SGEMM descriptors differ from the fixed target')
    if 'fresh_runtime' in manifest:
        require(len(set(connection_orders)) == 1 and len({value['session'] for value in bindings}) == 1,
                'replay target session or client connection order changed')
    require(len(set(controllers)) == len(entries) and
            len(set(client_identities)) == 2 * len(entries),
            'application replay reused a controller or client process identity')
    require(all(type(entry['pid']) is int and type(entry['start']) is int and
                entry['pid'] > 0 and entry['start'] > 0 for entry in identities) and
            len({entry['pid'] for entry in identities}) == len(identities),
            'replay service process identities are not unique')
    metrics_path = outer / 'ncu-metrics.csv'
    require(metrics_path.stat().st_size <= 4 * 1024 * 1024, 'NCU metrics exceed bound')
    metrics = targeted_ncu_metrics(metrics_path.read_text(), target_kernel)
    require(sum(entry['pid'] == metrics['process_id'] for entry in identities) == 1 and
            metrics['metrics']['launch__block_size'] == 128, 'NCU launch is not one replay service')
    range_path = outer / 'ncu-range-metrics.csv'
    require(range_path.stat().st_size <= 4 * 1024 * 1024, 'NVTX-renamed metrics exceed bound')
    ranged = targeted_ncu_metrics(range_path.read_text(), TARGET_RANGE + target_kernel)
    require(all(metrics[key] == ranged[key] for key in
                ('process_id', 'launch_id', 'context', 'stream', 'metrics')),
            'NVTX range and physical kernel exports do not identify the same launch')
    return dict(status='counter_observation_requires_independent_review', replay_count=len(entries),
                specialization=bridge.capture_specialization(signal_binding), target_kernel=target_kernel,
                services=identities, descriptor=descriptors[0], metrics=metrics,
                source_target=target, target_bindings=bindings,
                full_device_qualified=False, sm_coverage_verified=False,
                native_operation_matched=False, saturation_verified=False,
                performance_qualified=False)


def loader_dependencies(output):
    """Include direct ELF-interpreter rows, not just ldd's arrow-form DSOs."""
    require(type(output) is str and output.strip() and 'not found' not in output,
            'unresolved or empty loader dependency inventory')
    paths = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or re.fullmatch(r'linux-vdso\.so\.\d+ \(0x[0-9a-fA-F]+\)', line):
            continue
        match = re.fullmatch(r'(?:\S+ => )?(/\S+) \(0x[0-9a-fA-F]+\)', line)
        require(match is not None, 'unrecognized loader dependency row: ' + line)
        paths.add(Path(match[1]))
    require(paths, 'file-backed loader dependencies required')
    return paths


def prepare_inputs(outer, target, *, service_cap_sys_admin=False, host_selection=None, current_selection=None,
                   fresh_selection=None):
    """Seal the fixed BERT inputs and toolchain before any GPU launch."""
    parse_target(target)
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
            'clean unprivileged collector required')
    require(shutil.disk_usage(outer).free >= 8 * 1024 ** 3, 'at least 8 GiB free required')
    example_socket = outer / 'ncu-tmp/glimmer-bert-co-xxxxxxxx/service.sock'
    require(len(os.fsencode(example_socket)) < 108, 'NCU temporary root cannot fit service socket')
    require(sum(item is not None for item in (host_selection, current_selection, fresh_selection)) <= 1,
            'one runtime selection required')
    if fresh_selection is not None:
        from bert_ncu_current_sdk import configure_fresh
        binding, _, _, pin, _ = fresh_selection
        require(service_cap_sys_admin is True and target == binding['contract']['target'] == '2:3127',
                'fresh runtime requires the fixed clean-NCU sibling topology')
        require(shutil.disk_usage(outer).free >= candidate.OUTPUT_MAX + 8 * 1024 ** 3,
                'counter output allowance plus 8GiB reserve required')
        pin.verify()
        common, h = configure_fresh(binding)
    elif current_selection is not None:
        from bert_ncu_current_sdk import configure
        require(service_cap_sys_admin is True, 'current SDK requires the clean-NCU sibling topology')
        require(shutil.disk_usage(outer).free >= candidate.OUTPUT_MAX + 8 * 1024 ** 3,
                'counter output allowance plus 8GiB reserve required')
        common, h = configure(current_selection[0])
    elif host_selection is None:
        common, h = candidate.load_helpers()
    else:
        from bert_ncu_host_pair import configure
        require(service_cap_sys_admin is True, 'host pair requires the clean-NCU sibling topology')
        require(shutil.disk_usage(outer).free >= candidate.OUTPUT_MAX + 8 * 1024 ** 3,
                'counter output allowance plus 8GiB reserve required')
        common, h = configure(host_selection[0])
    gpu = h.query('gpu', h.training.GPU_FIELDS)
    fields = [value.strip() for value in gpu.split(',')]
    require(len(fields) == 4 and '\n' not in gpu and fields[0].startswith('GPU-') and 'V100' in fields[1],
            'one identified V100 required')
    require(fields[0].removeprefix('GPU-') == h.training.read_json(
        h.training.ORIGINAL / 'result.json')['environment']['device_uuid'], 'original GPU identity required')
    h.training.require_idle()
    provider = Path('/usr/local/cuda/lib64/libnvToolsExt.so.1').resolve(strict=True)
    library_dir = NCU_EXEC.parent
    libraries = {(library_dir / name).resolve(strict=True) for name in NCU_INJECTION_NAMES}
    require(len(libraries) == len(NCU_INJECTION_NAMES) and all(path.is_file() for path in libraries),
            'complete pinned NCU injection libraries required')
    seals = {str(path): digest(path) for path in libraries}
    manifest = dict(gpu=gpu, startup_reference_bundle=str(common.STARTUP_INPUTS),
                    startup_reference_report_sha256=common.STARTUP_INPUTS_SHA, sgemm_target=target,
                    uid=os.getuid(), gid=os.getgid(), groups=[], python=str(h.training.PYTHON),
                    ncu_sha256=digest(NCU), ncu_exec_sha256=digest(NCU_EXEC),
                    ncu_injection_seals=seals, range_library=str(provider),
                    range_library_sha256=digest(provider))
    require(manifest['uid'] > 0 and os.geteuid() == manifest['uid'], 'non-root collector required')
    h.startup_reference, h.export = common.application_references(h, manifest)
    if fresh_selection is not None:
        manifest['fresh_runtime'], native_paths, native_job, _, _ = fresh_selection
    elif current_selection is not None:
        manifest['current_sdk'], native_paths, native_job = current_selection
    elif host_selection is None:
        native_paths, native_job = candidate.existing_native_reference(
            candidate.EXISTING_NATIVE, gpu, common.STARTUP_INPUTS, common.STARTUP_INPUTS_SHA)
    else:
        manifest['host_pair'], native_paths, native_job = host_selection
    inference = native_job['inference_numerical']
    require(digest(h.export / 'export.json') == inference['export_sha256'] and
            digest(h.export / 'bert.onnx') == inference['model_sha256'], 'native export identity differs')
    example = Path(__file__).resolve().parents[1] / 'examples/framework_workloads/bert'
    sources = {'bert_contract.py', 'bert_workload.py', 'onnx_bert_workload.py', 'export_bert.py'}
    require(set(inference['source_sha256']) == sources and
            all(digest(example / name) == value for name, value in inference['source_sha256'].items()),
            'native workload sources differ')
    project = Path(__file__).resolve().parents[1]
    application_candidates = common.input_paths(h)
    paths = application_candidates | native_paths | libraries | {
        Path(__file__).resolve(), Path(candidate.__file__).resolve(),
        project / 'scripts/bert_sgemm_coverage_candidates.py',
        project / 'scripts/bert_sgemm_ncu_process.py',
        project / 'scripts/bert_ncu_client_broker.py',
        project / 'scripts/run_bert_sgemm_descriptor_probe.py',
        project / 'scripts/artifact_paths.py', project / 'scripts/run_rnn_v8_validation.py',
        project / 'scripts/rnn_v8_audit.py', project / 'scripts/cudnn8_legacy_audit.py',
        candidate.BUILD / 'CMakeCache.txt', candidate.BUILD / 'compile_commands.json',
        Path('/usr/bin/setpriv'), NCU, NCU_EXEC, provider, outer / 'manifest.json'}
    if service_cap_sys_admin:
        import bert_ncu_replay_exec as bridge
        require(target == '2:3127', 'the bounded service bridge supports only the fixed reviewed target')
        service_environment = h.training.clean_environment(fields[0])
        if fresh_selection is not None:
            require(service_environment == fresh_selection[0]['contract']['service_environment'] and
                    h.training.SERVER == Path(fresh_selection[0]['contract']['runtime']['service']['path']) and
                    h.training.PRELOAD == Path(fresh_selection[0]['contract']['runtime']['frontend']['path']),
                    'fresh replay environment or runtime differs')
            from bert_ncu_current_sdk import fresh_application_inventory
            application = fresh_selection[4]
            application_paths, application['expected'], application['contract'] = fresh_application_inventory(
                common, h, application_candidates, application['manifests'], service_environment)
            application.update(common=common, environment=service_environment)
            paths.update(application_paths)
            manifest['application_input_contract'] = application['contract']
        if fresh_selection is None:
            require(bridge.selected_service(manifest) == h.training.SERVER, 'selected bridge/service differ')
        else:
            require(h.training.SERVER == Path(fresh_selection[0]['contract']['runtime']['service']['path']),
                    'fresh collector and reviewed service differ')
        manifest.update(service_cap_sys_admin=True, service_environment=service_environment,
                        sgemm_selector=dict(kind='client', role='low', sequence=3127),
                        ncu_supervisor_environment=dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8',
                            TMPDIR=str(outer / 'ncu-tmp'), CUDA_CACHE_DISABLE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                            CUDA_VISIBLE_DEVICES=fields[0]),
                        client_wrapper_command=[manifest['python'], '-B', '-c', h.wrapper],
                        bridge_seals={str(path): digest(path) for path in
                            (bridge.SCRIPT, h.training.SERVER, bridge.PYTHON, bridge.SETPRIV, bridge.ENV, provider)})
        paths.update(project / 'scripts' / name for name in
                     ('bert_ncu_replay_exec.py', 'bert_ncu_service_proxy.py', 'bert_ncu_sibling.py', 'bert_ncu_relay.py'))
        paths.update((bridge.PYTHON, bridge.ENV))
        # These pinned NCU components are loaded dynamically by the injection
        # and launcher. They are not additional permitted LD_PRELOAD entries.
        runtime_libraries = set(libraries)
        for name in ('libnvperf_target.so', 'libnvperf_host.so', 'libTreeLauncherPlaceholder.so'):
            path = (library_dir / name).resolve(strict=True)
            require(path.parent == library_dir.resolve(strict=True), 'NCU runtime escaped its pinned directory')
            bridge.protected_system_file(path)
            runtime_libraries.add(path)
            paths.add(path)
        # Freeze lazily loaded installed driver code, plus NCU's loader closure.
        for pattern in ('libnvidia*.so*', 'libnvoptix.so*', 'libcuda.so*'):
            paths.update(Path('/usr/lib/x86_64-linux-gnu').glob(pattern))
        # The profiled service lazily loads glibc's UTF-32 conversion module.
        # Seal the observed codec and selector inputs before launch; unknown
        # executable mappings remain rejected by the runtime audit.
        gconv = Path('/usr/lib/x86_64-linux-gnu/gconv')
        codec = (gconv / 'UTF-32.so').resolve(strict=True)
        for path in (codec, gconv / 'gconv-modules', gconv / 'gconv-modules.cache'):
            bridge.protected_system_file(path)
            paths.add(path)
        for binary in sorted(runtime_libraries | {codec}):
            output = subprocess.check_output(['/usr/bin/ldd', str(binary)], text=True,
                env=dict(PATH='/usr/bin:/bin', LANG='C', LD_LIBRARY_PATH=str(library_dir)), timeout=30)
            paths.update(loader_dependencies(output))
    save(outer / 'manifest.json', manifest)
    before = h.training.manifest(paths)
    if fresh_selection is not None:
        from bert_ncu_current_sdk import validate_fresh_application_inputs
        application['summary'] = validate_fresh_application_inputs(application['expected'], before)
    if current_selection is not None:
        from bert_ncu_current_sdk import validate_numerical_inputs
        validate_numerical_inputs(current_selection[0], before)
    save(outer / 'inputs-before.json', before)
    h.training.require_idle()
    require(before == h.training.manifest(paths), 'frozen input changed before capture')
    if fresh_selection is not None:
        require(validate_fresh_application_inputs(application['expected'], before) == application['summary'],
                'recorded application input changed before capture')
    return h, gpu, paths, before


def unit_properties(unit):
    output = command(['systemctl', '--user', 'show', unit,
                      '--property=LoadState,ActiveState,Description,ControlGroup,ExecMainStartTimestampMonotonic,MainPID'])
    result = {}
    for line in output.splitlines():
        key, separator, value = line.partition('=')
        require(separator and key not in result, 'invalid or duplicate unit property')
        result[key] = value
    require(set(result) == {'LoadState', 'ActiveState', 'Description', 'ControlGroup',
                            'ExecMainStartTimestampMonotonic', 'MainPID'}, 'incomplete unit identity')
    return result


def task_cgroup_empty(handshake, unit):
    group = handshake['group']
    require(type(group) is str and group.startswith('/') and group.endswith('/' + unit) and
            '..' not in Path(group).parts, 'exact task cgroup required for retirement')
    path = Path('/sys/fs/cgroup') / group.lstrip('/')
    try:
        before = path.stat()
    except FileNotFoundError:
        return True
    require(not path.is_symlink() and (before.st_dev, before.st_ino) ==
            (handshake['cgroup_device'], handshake['cgroup_inode']), 'task cgroup identity changed')
    try:
        entries = dict(line.split() for line in (path / 'cgroup.events').read_text().splitlines())
        after = path.stat()
    except FileNotFoundError:
        require(not path.exists(), 'task cgroup state disappeared unexpectedly')
        return True
    require((after.st_dev, after.st_ino) == (before.st_dev, before.st_ino) and entries.get('populated') == '0',
            'task cgroup is still populated or changed')
    return True


def require_owned_unit(properties, unit, description, owned, handshake):
    require(owned and properties['Description'] == description, 'refuse to modify an unowned unit')
    if handshake is not None and properties['ActiveState'] not in ('inactive', 'failed'):
        require(properties['ControlGroup'] == handshake['group'] and
                properties['ControlGroup'].endswith('/' + unit), 'active task cgroup changed')
        directory = Path('/sys/fs/cgroup') / properties['ControlGroup'].lstrip('/')
        metadata = directory.stat()
        require(not directory.is_symlink() and (metadata.st_dev, metadata.st_ino) ==
                (handshake['cgroup_device'], handshake['cgroup_inode']), 'active task cgroup identity changed')
    elif properties['ActiveState'] in ('inactive', 'failed'):
        require(properties['MainPID'] == '0', 'inactive unit still has a main process')


def retire_unit(unit, description, process, owned, handshake):
    """Independently attempt stop, process reap and final unit/cgroup verification."""
    errors = []
    try:
        properties = unit_properties(unit)
        if properties['LoadState'] != 'not-found':
            require_owned_unit(properties, unit, description, owned, handshake)
            if properties['ActiveState'] not in ('inactive', 'failed'):
                command(['systemctl', '--user', 'stop', unit], timeout=15)
    except BaseException as error:
        errors.append(f'stop {type(error).__name__}: {error}')
    try:
        properties = unit_properties(unit)
        if properties['LoadState'] != 'not-found' and properties['ActiveState'] == 'failed':
            require_owned_unit(properties, unit, description, owned, handshake)
            command(['systemctl', '--user', 'reset-failed', unit], timeout=5)
    except BaseException as error:
        errors.append(f'reset {type(error).__name__}: {error}')
    if process is not None:
        try:
            process.wait(timeout=10)
        except BaseException as error:
            errors.append(f'reap {type(error).__name__}: {error}')
    removed, empty = False, False
    try:
        properties = unit_properties(unit)
        removed = (properties['LoadState'] == 'not-found' and properties['ActiveState'] == 'inactive' and
                   properties['MainPID'] == '0' and properties['ControlGroup'] == '')
        require(removed, 'task unit was not removed')
        empty = handshake is not None and task_cgroup_empty(handshake, unit)
        require(handshake is None or empty, 'task cgroup retirement was not established')
    except BaseException as error:
        errors.append(f'verify {type(error).__name__}: {error}')
    result = dict(unit_removed=removed, cgroup_empty=empty)
    if errors:
        result['cleanup_error'] = '; '.join(errors)
    return result


def launch_unit(outer, gpu, *, fresh_claim=None):
    """Start one unique task unit and authenticate it before root NCU runs."""
    unit = f'glimmer-bert-{outer.name}-equal.service'
    description = 'Glimmer NCU capture ' + outer.name
    require(unit_properties(unit)['LoadState'] == 'not-found', 'never reuse an existing unit')
    run = ['systemd-run', '--user', '--unit=' + unit, '--description=' + description, '--wait', '--pipe',
           '--property=MemoryMax=' + str(candidate.MEMORY_MAX), '--property=MemorySwapMax=0',
           '--property=OOMPolicy=kill', '--property=RuntimeMaxSec=800s', '--property=TimeoutStopSec=5s',
           '--property=KillMode=control-group', '--property=TasksMax=256',
           '--property=LimitFSIZE=' + str(candidate.FILE_MAX), '/usr/bin/env', '-u', 'LD_PRELOAD',
           '-u', 'LD_LIBRARY_PATH', '/usr/bin/sudo', '-n', '/usr/bin/python3', '-B',
           str(Path(__file__).resolve()), '--root-supervisor', str(outer)]
    result = dict(unit=unit, returncode=None, unit_removed=False)
    process, owned = None, False
    try:
        with (outer / 'unit.log').open('x') as log:
            process = subprocess.Popen(run, stdout=log, stderr=subprocess.STDOUT)
            startup_deadline = time.monotonic() + 35
            ready_path = outer / 'ncu-ready.json'
            while not ready_path.exists():
                properties = unit_properties(unit)
                owned = owned or (properties['LoadState'] != 'not-found' and
                                  properties['Description'] == description)
                require(process.poll() is None and time.monotonic() < startup_deadline,
                        'bounded task unit did not reach the NCU handshake')
                time.sleep(.05)
            ready = json.loads(ready_path.read_text())
            require(type(ready) is dict and set(ready) ==
                    {'pid', 'start', 'group', 'cgroup_device', 'cgroup_inode'} and
                    all(type(ready[key]) is int and ready[key] > 0
                        for key in ('pid', 'start', 'cgroup_device', 'cgroup_inode')),
                    'malformed privileged supervisor handshake')
            properties = unit_properties(unit)
            owned = properties['LoadState'] != 'not-found' and properties['Description'] == description
            require(owned and properties['ActiveState'] == 'active' and
                    properties['ControlGroup'] == ready['group'] and
                    properties['ExecMainStartTimestampMonotonic'].isdecimal(),
                    'systemd unit identity differs from the supervisor')
            require(identity(ready['pid']) == ready['start'] and cgroup(ready['pid']) == ready['group'],
                    'supervisor process identity or cgroup differs')
            status = Path(f"/proc/{ready['pid']}/status").read_text().splitlines()
            require([line.split()[1:5] for line in status if line.startswith('Uid:')] ==
                    [['0', '0', '0', '0']], 'supervisor is not root')
            task = Path('/sys/fs/cgroup') / ready['group'].lstrip('/')
            metadata = task.stat()
            require((metadata.st_dev, metadata.st_ino) ==
                    (ready['cgroup_device'], ready['cgroup_inode']), 'unit cgroup identity differs')
            memory_receipt(ready['group'])
            require(not check_owned_gpu(ready['group'], gpu.split(',')[0].strip()),
                    'GPU must be idle before launching the profiler')
            start_us = int(properties['ExecMainStartTimestampMonotonic'])
            require(start_us > 0, 'actual unit start time required')
            go = dict(ready, unit_started_us=start_us)
            if fresh_claim is not None:
                require(type(fresh_claim) is dict and set(fresh_claim) == {'path', 'sha256'} and
                        type(fresh_claim['sha256']) is str and
                        re.fullmatch(r'[0-9a-f]{64}', fresh_claim['sha256']) is not None,
                        'one authenticated fresh claim required')
                go['fresh_claim_sha256'] = fresh_claim['sha256']
            publish_json(outer / 'ncu-go.json', go)
            result['handshake'] = ready
            result['unit_started_us'] = start_us
            timeout = max(1., start_us / 1_000_000 + 820 - time.monotonic())
            result['returncode'] = process.wait(timeout=timeout)
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        result.update(retire_unit(unit, description, process, owned, result.get('handshake')))
        save(outer / 'unit.json', result)
    require(result['returncode'] == 0 and result['unit_removed'] and result['cgroup_empty'] and 'error' not in result and
            'cleanup_error' not in result, 'bounded task unit did not complete and retire normally')
    return result


def supervisor_failure(outer):
    """Expose a bounded inner diagnostic without changing unit acceptance."""
    path = outer / 'ncu-supervisor.json'
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (type(value) is dict and value.get('status') == 'failed' and
            type(value.get('error')) is str and 0 < len(value['error']) <= 4096):
        return value['error']
    return None


def collect(target, *, service_cap_sys_admin=False, host_pair_plan=None, current_sdk_plan=None,
            fresh_runtime_plan=None, variant=None):
    """One bounded batch, no automatic retry or unqualified publication."""
    parse_target(target)
    require(subprocess.run(['/usr/bin/sudo', '-n', '/usr/bin/true'], capture_output=True,
                           timeout=10, check=False).returncode == 0,
            'sudo authorization unavailable; no GPU experiment started')
    plans = (host_pair_plan, current_sdk_plan, fresh_runtime_plan)
    require(sum(item is not None for item in plans) <= 1, 'only one runtime plan is permitted')
    require(all(item is None for item in plans) == (variant is None),
            'runtime plan and variant must be selected together')
    if fresh_runtime_plan is not None:
        require(service_cap_sys_admin is True and target == '2:3127' and variant in ('scalar', 'sharded'),
                'fresh runtime requires the fixed sibling target and signal variant')
    host_selection, current_selection = None, None
    if host_pair_plan is not None:
        from bert_ncu_host_pair import prepare
        require(service_cap_sys_admin is True and target == '2:3127', 'fixed sibling target required')
        host_selection = prepare(host_pair_plan, variant)
    elif current_sdk_plan is not None:
        from bert_ncu_current_sdk import prepare
        require(service_cap_sys_admin is True and target == '2:3127', 'fixed sibling target required')
        current_selection = prepare(current_sdk_plan, variant)
    outer = create_attempt(resolve_artifact_root('ncu', category='profiles'))
    print('NCU capture evidence: ' + str(outer), flush=True)
    report, h, gpu, paths, before = dict(status='started', target=target, **FLAGS), None, None, None, None
    try:
        with ExitStack() as preparations:
            fresh_selection = None
            if fresh_runtime_plan is not None:
                from bert_ncu_current_sdk import prepare_fresh
                fresh_selection = preparations.enter_context(prepare_fresh(fresh_runtime_plan, variant))
                report['fresh_runtime'] = fresh_selection[0]
            try:
                selections = dict(host_selection=host_selection, current_selection=current_selection)
                if fresh_selection is not None:
                    selections['fresh_selection'] = fresh_selection
                h, gpu, paths, before = prepare_inputs(outer, target, service_cap_sys_admin=service_cap_sys_admin,
                                                       **selections)
                (outer / 'passes').mkdir()
                (outer / 'ncu-tmp').mkdir()
                h.training.require_idle()
                if host_pair_plan is not None:
                    from bert_ncu_host_pair import claim_launch
                    claim_launch(host_pair_plan, variant)
                    report['host_pair'] = host_selection[0]
                elif current_sdk_plan is not None:
                    from bert_ncu_current_sdk import claim_launch
                    claim_launch(current_selection[0])
                    report['current_sdk'] = current_selection[0]
                elif fresh_selection is not None:
                    from bert_ncu_current_sdk import claim_fresh
                    fresh_selection[3].verify()
                    report['fresh_claim'] = claim_fresh(fresh_selection[0], outer, fresh_selection[4]['contract'])
                if fresh_selection is not None:
                    report['unit'] = launch_unit(outer, gpu, fresh_claim=report['fresh_claim'])
                else:
                    report['unit'] = launch_unit(outer, gpu)
                if fresh_selection is not None:
                    report['observation'] = review_capture(outer, target, fresh_binding=fresh_selection[0],
                                                           fresh_claim=report['fresh_claim'])
                else:
                    report['observation'] = review_capture(outer, target)
                if any(item is not None for item in plans):
                    require(report['observation']['metrics']['minimums_positive'] is True,
                            'current target failed all-SM active-cycle or useful-compute coverage')
                h.training.require_idle()
                report['status'] = 'counter_observation_requires_independent_review'
            except BaseException as error:
                report.update(status='failed', error=f'{type(error).__name__}: {error}')
                reason = supervisor_failure(outer)
                if reason is not None:
                    report['supervisor_error'] = reason
            finally:
                if before is not None:
                    try:
                        after = h.training.manifest(paths)
                        save(outer / 'inputs-after.json', after)
                        report['input_identity_unchanged'] = before == after
                        require(before == after and h.query('gpu', h.training.GPU_FIELDS) == gpu,
                                'frozen input or GPU changed during capture')
                        if fresh_selection is not None:
                            from bert_ncu_current_sdk import fresh_application_inventory, validate_fresh_application_inputs
                            application = fresh_selection[4]
                            _, expected, contract = fresh_application_inventory(
                                application['common'], h, application['common'].input_paths(h), application['manifests'],
                                application['environment'])
                            require(expected == application['expected'] and contract == application['contract'],
                                    'application loader inventory changed')
                            postflight = validate_fresh_application_inputs(application['expected'], after)
                            require(postflight == application['summary'], 'application input summary changed')
                            report['application_input_summary'] = application['contract']
                    except BaseException as error:
                        report.update(status='failed', identity_error=f'{type(error).__name__}: {error}')
    except BaseException as error:
        report.update(status='failed', preparation_error=f'{type(error).__name__}: {error}')
    finally:
        save(outer / 'report.json', report)
    summary = dict(status=report['status'], error=report.get('error'),
                   supervisor_error=report.get('supervisor_error'), report=str(outer / 'report.json'))
    if fresh_runtime_plan is not None:
        summary.update(preparation_error=report.get('preparation_error'), identity_error=report.get('identity_error'))
    print(json.dumps(summary), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--collect', action='store_true')
    group.add_argument('--pass', dest='pass_attempt', type=Path, help=argparse.SUPPRESS)
    group.add_argument('--root-supervisor', type=Path, help=argparse.SUPPRESS)
    group.add_argument('--broker', type=Path, help=argparse.SUPPRESS)
    group.add_argument('--sibling-pass', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--service-cap-sys-admin', action='store_true',
                        help='opt in to the reviewed non-root service-only counter capability')
    parser.add_argument('--endpoint', help=argparse.SUPPRESS)
    parser.add_argument('--supervisor', help=argparse.SUPPRESS)
    parser.add_argument('--serial', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--target', default='2:3127')
    parser.add_argument('--host-pair-plan', type=Path,
                        help='reviewed immutable host-pair counter plan; requires sibling mode')
    parser.add_argument('--current-sdk-plan', type=Path,
                        help='reviewed current SDK counter plan; requires sibling mode')
    parser.add_argument('--fresh-runtime-plan', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--variant', choices=('o0', 'o2', 'scalar', 'sharded'),
                        help='explicit immutable runtime variant')
    args = parser.parse_args()
    if args.pass_attempt:
        return profiled_pass(args.pass_attempt)
    if args.root_supervisor:
        return root_supervisor(args.root_supervisor)
    if args.broker:
        return broker_worker(args.broker)
    if args.sibling_pass:
        import bert_ncu_replay_exec as bridge
        import bert_ncu_sibling
        return bert_ncu_sibling.run(args.sibling_pass, args.endpoint, bridge.parse_identity(args.supervisor), args.serial)
    if args.collect:
        try:
            return collect(args.target, service_cap_sys_admin=args.service_cap_sys_admin,
                           host_pair_plan=args.host_pair_plan, current_sdk_plan=args.current_sdk_plan,
                           fresh_runtime_plan=args.fresh_runtime_plan, variant=args.variant)
        except BaseException as error:
            print(f'capture not started: {type(error).__name__}: {error}', flush=True)
            return 1
    parser.print_help()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
