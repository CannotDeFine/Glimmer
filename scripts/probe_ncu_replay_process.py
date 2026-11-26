#!/usr/bin/env python3
"""Bounded, unprivileged, CPU-only NCU spawn/exec/FD compatibility diagnostic.

No CUDA API, application workload, capability elevation, or GPU qualification.
The two policies compare inherited NCU descriptors with the bridge's close policy.
An external acknowledgement holds the real parent alive after waitpid.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bert_ncu_replay_exec as bridge
import bert_sgemm_ncu_process as process_checks


SCRIPT = Path(__file__).resolve()
NCU = Path('/opt/nvidia/nsight-compute/2024.3.2/ncu')
ARTIFACTS = SCRIPT.parent.parent / 'artifacts/profiles/ncu-process'


def snapshot():
    fields = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    seals = {str((bridge.NCU_DIRECTORY / name).resolve()): bridge.seal(bridge.NCU_DIRECTORY / name)
             for name in process_checks.NCU_INJECTION_NAMES}
    parsed = process_checks.proc_environment(os.getpid(), ncu_seals=seals)
    normalized = {os.fsdecode(key): os.fsdecode(value) for key, value in parsed.items()}
    bridge.require(normalized == dict(os.environ), 'procfs parser differs from the live environment')
    raw_environment = Path('/proc/self/environ').read_bytes()
    descriptors, descriptor_metadata = {}, {}
    for name in os.listdir('/proc/self/fd'):
        try:
            descriptors[name] = os.readlink('/proc/self/fd/' + name)
            metadata = os.fstat(int(name))
            record = dict(device=metadata.st_dev, inode=metadata.st_ino,
                          mode=oct(metadata.st_mode), uid=metadata.st_uid,
                          flags=bridge.fcntl.fcntl(int(name), bridge.fcntl.F_GETFL))
            if descriptors[name].startswith('/') and Path(descriptors[name]).exists():
                entry = Path(descriptors[name]).lstat()
                parent = Path(descriptors[name]).parent.stat()
                record['path'] = dict(device=entry.st_dev, inode=entry.st_ino,
                                     mode=oct(entry.st_mode), uid=entry.st_uid)
                record['parent'] = dict(mode=oct(parent.st_mode), uid=parent.st_uid)
            descriptor_metadata[name] = record
        except FileNotFoundError:
            pass
    return dict(identity=bridge.process_identity(os.getpid()), parent=os.getppid(),
                uid=os.getuid(), status={key: fields[key].strip() for key in
                    (*bridge.CAP_FIELDS, 'NoNewPrivs', 'Groups')},
                maps=Path('/proc/self/maps').read_text(), descriptors=descriptors,
                descriptor_metadata=descriptor_metadata,
                environment=normalized, proc_environment_matches_live=True,
                raw_environment_sha256=bridge.hashlib.sha256(raw_environment).hexdigest(),
                raw_environment_residues=sum(bool(item) and b'=' not in item
                                             for item in raw_environment.split(b'\0')))


def cpu_drop():
    # A non-root process cannot clear its bounding set or supplementary groups.
    # This tests the real loader chain, not the root-only credential transition.
    return [str(bridge.SETPRIV), '--no-new-privs', '--inh-caps=-all', '--ambient-caps=-all']


def file_identity(record):
    return record['device'], record['inode'], stat.S_IFMT(int(record['mode'], 8))


def sentinels(directory):
    """Inheritable unrelated descriptors must disappear only in selective mode."""
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    descriptors = [left.detach(), right.detach(), *os.pipe(),
                   os.open(directory / 'sentinel', os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)]
    for descriptor in descriptors:
        os.set_inheritable(descriptor, True)
    return descriptors


def worker(args):
    bridge.require(os.geteuid() != 0, 'CPU probe must never run as root')
    if args.stage == 'child':
        print(json.dumps(snapshot(), sort_keys=True), flush=True)
        return 0
    if args.stage == 'parent':
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
            connection.connect(args.endpoint)
            with (args.directory / 'child.json').open('x') as log:
                descriptor = bridge.sealed_snapshot(bridge.PYTHON, bridge.seal(bridge.PYTHON))
                try:
                    os.set_inheritable(descriptor, True)
                    extra = sentinels(args.directory)
                    observation = snapshot()
                    evidence = dict(before_spawn=observation, executable_fd=descriptor,
                                    sentinels={str(fd): observation['descriptor_metadata'][str(fd)] for fd in extra})
                    (args.directory / 'parent.json').write_text(json.dumps(evidence, indent=2) + '\n')
                    child_environment = dict(os.environ)
                    if args.service_template:
                        service = json.loads(args.service_template.read_text())['service_environment']
                        bridge.exact_fields(service, bridge.ENV_KEYS)
                        child_environment.update(service)
                    command = [*cpu_drop(), str(bridge.ENV), '-i',
                               *[key + '=' + value for key, value in sorted(child_environment.items())],
                               f'/proc/self/fd/{descriptor}', '-I', '-S', '-B', str(SCRIPT),
                               '--stage', 'child']
                    if args.policy in ('close', 'tracker'):
                        retained = (bridge.tracker_descriptors(dict(os.environ), Path(os.environ['TMPDIR']),
                                                               os.getuid()) if args.policy == 'tracker' else set())
                        actions = bridge.child_file_actions(log.fileno(), descriptor, retained)
                    else:
                        actions = [(os.POSIX_SPAWN_OPEN, 0, '/dev/null', os.O_RDONLY, 0),
                                   (os.POSIX_SPAWN_DUP2, log.fileno(), 1),
                                   (os.POSIX_SPAWN_DUP2, log.fileno(), 2)]
                    pid = os.posix_spawn(command[0], command, dict(os.environ),
                                         file_actions=actions, setsid=True)
                    child = bridge.process_identity(pid)
                    child_fd = os.pidfd_open(pid)
                finally:
                    os.close(descriptor)
            os.set_inheritable(connection.fileno(), True)
            os.set_inheritable(child_fd, True)
            (args.directory / 'handoff.json').write_text(json.dumps(dict(child=child,
                    control_fd=connection.fileno(), child_fd=child_fd, before_exec=snapshot()), indent=2) + '\n')
            command = [*cpu_drop(), str(bridge.PYTHON), '-I', '-S', '-B', str(SCRIPT),
                       '--stage', 'waiter', '--directory', str(args.directory),
                       '--control-fd', str(connection.fileno()), '--child-fd', str(child_fd),
                       '--child', f"{child['pid']}:{child['start']}"]
            if args.policy in ('close', 'tracker'):
                retained = (bridge.tracker_descriptors(dict(os.environ), Path(os.environ['TMPDIR']),
                                                       os.getuid()) if args.policy == 'tracker' else set())
                bridge.close_except({connection.fileno(), child_fd} | retained)
            os.execve(bridge.SETPRIV, command, dict(os.environ))
    with socket.socket(fileno=args.control_fd) as connection:
        connection.settimeout(10)
        try:
            receipt = bridge.reap_child(bridge.parse_identity(args.child), time.monotonic_ns() + 8 * 10 ** 9)
            bridge.send_packet(connection, dict(op='reaped', receipt=receipt, waiter=snapshot()))
            message, descriptors, _ = bridge.receive_packet(connection)
            for descriptor in descriptors:
                os.close(descriptor)
            bridge.require(not descriptors and message == dict(op='accept'), 'external barrier missing')
            return receipt['returncode']
        finally:
            os.close(args.child_fd)


def validate_snapshot(record):
    bridge.require(record['uid'] != 0 and record['status']['NoNewPrivs'] == '1', 'non-root NNP required')
    for key in ('CapInh', 'CapPrm', 'CapEff', 'CapAmb'):
        bridge.require(int(record['status'][key], 16) == 0, 'unexpected active capability')
    bridge.require('libTreeLauncherTargetInjection.so' in record['maps'], 'NCU tracker not mapped')
    bridge.require('NVIDIA-PROCESS-TRACKING-CONFIGURATION' in record['environment'], 'NCU tracking lost')


def validate_inventory(record, explicit, sentinel_records, policy):
    observed = {file_identity(value) for value in record['descriptor_metadata'].values()}
    sent = {file_identity(value) for value in sentinel_records.values()}
    if policy == 'inherit':
        bridge.require(sent <= observed, 'inherit control did not retain all sentinel types')
        return
    bridge.require(not observed & sent, 'unrelated sentinel descriptor leaked')
    tracking = {}
    for name, target in record['descriptors'].items():
        if int(name) <= 2 or int(name) in explicit:
            continue
        metadata = record['descriptor_metadata'][name]
        bridge.require('/NVIDIA-treetracker-' in target and stat.S_ISFIFO(int(metadata['mode'], 8)),
                       'unexpected inherited descriptor')
        tracking[name] = metadata
    bridge.require(len({file_identity(value) for value in tracking.values()}) == 2,
                   'exactly two tracker FIFO inode families required')
    bridge.require(len(tracking) >= 2, 'tracker descriptors missing')


def collect_one(directory, policy, *, capture=None, injection_seals=None, service_template=None):
    directory.mkdir()
    with tempfile.TemporaryDirectory(prefix='gncp-') as temporary, ExitStack() as resources:
        endpoint = str(Path(temporary) / 'control')
        listener = resources.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET))
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        listener.bind(endpoint)
        listener.listen(1)
        listener.settimeout(15)
        log = resources.enter_context((directory / ('ncu-supervisor.log' if capture else 'ncu.log')).open('x'))
        launch_outer = Path(temporary) if capture else directory
        tmpdir = launch_outer / 'ncu-tmp' if capture else Path(temporary)
        if capture:
            tmpdir.mkdir()
        env = dict(PATH='/usr/bin:/bin', LANG='C.UTF-8', LC_ALL='C.UTF-8', TMPDIR=str(tmpdir),
                   CUDA_CACHE_DISABLE='1', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES='GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2')
        prefix = [str(NCU), '--config-file', '0', '--target-processes', 'all', '--nvtx']
        if capture:
            recipe = capture.ncu_command(directory, dict(uid=os.getuid(), gid=os.getgid(), groups=[],
                                                         python=str(bridge.PYTHON)))
            prefix = recipe[:recipe.index('/usr/bin/setpriv')]
        command = [*prefix, str(bridge.PYTHON), '-I', '-S', '-B', str(SCRIPT), '--stage', 'parent',
                   '--policy', policy, '--directory', str(directory), '--endpoint', endpoint]
        if service_template:
            command += ['--service-template', str(service_template)]
        process = subprocess.Popen(command, env=env, cwd=launch_outer, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        result = dict(policy=policy, command=command, ncu_identity=bridge.process_identity(process.pid),
                      capture_options=capture is not None, launcher_environment=env,
                      launcher_directory=str(launch_outer))
        pidfd, waiter_fd = os.pidfd_open(process.pid), None
        try:
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(15)
                message, descriptors, credentials = bridge.receive_packet(connection)
                for descriptor in descriptors:
                    os.close(descriptor)
                bridge.require(not descriptors and message['op'] == 'reaped', 'invalid wait receipt')
                waiter = message['waiter']['identity']
                peer = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                bridge.require(credentials == peer == (waiter['pid'], os.getuid(), os.getgid()) and
                               bridge.process_identity(waiter['pid']) == waiter,
                               'post-exec waiter credentials or incarnation differ')
                waiter_fd = os.pidfd_open(waiter['pid'])
                cmdline = Path(f"/proc/{waiter['pid']}/cmdline").read_bytes().split(b'\0')[:-1]
                bridge.require(os.fsencode(SCRIPT) in cmdline and b'--stage' in cmdline and b'waiter' in cmdline,
                               'actual parent did not exec the waiter')
                result.update(message)
                child = json.loads((directory / 'child.json').read_text())
                result['child'] = child
                validate_snapshot(child)
                validate_snapshot(message['waiter'])
                parent = json.loads((directory / 'parent.json').read_text())
                handoff = json.loads((directory / 'handoff.json').read_text())
                if capture:
                    template = dict(ncu_supervisor_environment=env,
                                    service_environment=dict(CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES']),
                                    ncu_injection_seals=injection_seals)
                    result['profiler_environment_checks'] = {}
                    for stage, observation in (('parent', parent['before_spawn']), ('child', child),
                                               ('before_exec', handoff['before_exec']), ('waiter', message['waiter'])):
                        if stage == 'child' and service_template:
                            import bert_ncu_relay as relay
                            service = json.loads(service_template.read_text())['service_environment']
                            instance = relay.ReplayRelay.__new__(relay.ReplayRelay)
                            instance.outer, instance.base = launch_outer, env
                            instance.manifest = dict(template, service_environment=service)
                            instance.validate_service_environment(observation['environment'])
                            result['service_environment_check'] = True
                        else:
                            bridge.profiler_environment(observation['environment'], template, launch_outer)
                        result['profiler_environment_checks'][stage] = True
                validate_inventory(child, {parent['executable_fd']}, parent['sentinels'], policy)
                validate_inventory(message['waiter'], {handoff['control_fd'], handoff['child_fd']},
                                   parent['sentinels'], policy)
                bridge.require(child['identity'] == {key: message['receipt'][key] for key in ('pid', 'start')}
                               and child['parent'] == message['waiter']['identity']['pid']
                               and message['receipt']['returncode'] == 0,
                               'child identity, parenthood, or exit differs')
                # NCU must not return while the application parent awaits acceptance.
                for _ in range(5):
                    bridge.require(process.poll() is None and not bridge.terminal(waiter_fd) and
                                   bridge.process_identity(waiter['pid']) == waiter,
                                   'NCU or actual waiter exited before the full-pass barrier')
                    time.sleep(.04)
                result['barrier_held'] = True
                bridge.send_packet(connection, dict(op='accept'))
            result['returncode'] = process.wait(timeout=8)
            bridge.require(result['returncode'] == 0 and bridge.terminal(waiter_fd),
                           'NCU exit differs or waiter did not retire')
            result['status'] = 'cpu_process_compatibility_passed'
        except Exception as error:
            result.update(status='failed', error=repr(error))
        finally:
            if process.poll() is None:
                # Let a failing target flush its diagnostic and NCU reap it.
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                # The unreaped session leader and live pidfd pin this probe's exact group.
                if not bridge.terminal(pidfd) and os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            os.close(pidfd)
            if waiter_fd is not None:
                os.close(waiter_fd)
            result['final_returncode'] = process.returncode
        (directory / 'report.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        return {key: result[key] for key in ('policy', 'status', 'final_returncode')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    parser.add_argument('--capture-options', action='store_true',
                        help='Use the production NCU replay options and full profiler environment checks; CPU only')
    parser.add_argument('--service-template', type=Path,
                        help='With capture options, give the CPU child the sealed manifest service environment')
    parser.add_argument('--policies', nargs='+', choices=('inherit', 'close', 'tracker'),
                        default=('inherit', 'tracker'))
    parser.add_argument('--stage', choices=('parent', 'child', 'waiter'))
    parser.add_argument('--policy', choices=('inherit', 'close', 'tracker'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--control-fd', type=int)
    parser.add_argument('--child-fd', type=int)
    parser.add_argument('--child')
    args = parser.parse_args()
    bridge.require(os.geteuid() != 0, 'CPU probe must never run as root')
    if args.stage:
        try:
            return worker(args)
        except Exception as error:
            if args.directory is not None:
                (args.directory / (args.stage + '-error.json')).write_text(
                    json.dumps(dict(error=repr(error), observation=snapshot()), indent=2) + '\n')
            raise
    bridge.require(args.collect, 'explicit --collect required')
    bridge.require(not args.service_template or args.capture_options,
                   'service environment probe requires the production capture options')
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    outer = Path(tempfile.mkdtemp(prefix='attempt-', dir=ARTIFACTS))
    print('CPU-only NCU process diagnostic: ' + str(outer), flush=True)
    capture = None
    if args.capture_options:
        import run_bert_sgemm_ncu_capture as capture
    if args.service_template:
        import bert_ncu_relay
        args.service_template = args.service_template.resolve(strict=True)
    sources = sorted({Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                      if getattr(module, '__file__', None) and
                      Path(module.__file__).resolve().is_relative_to(SCRIPT.parent.parent) and
                      Path(module.__file__).suffix == '.py'})
    paths = (*sources, NCU, bridge.NCU_DIRECTORY / 'ncu', bridge.PYTHON,
             bridge.SETPRIV, bridge.ENV, *sorted(bridge.NCU_DIRECTORY.glob('*Injection*.so')),
             bridge.NCU_DIRECTORY / 'libcuda-injection.so')
    if args.service_template:
        paths += (args.service_template,)
    before = {str(path): bridge.seal(path) for path in paths}
    (outer / 'inputs-before.json').write_text(json.dumps(before, indent=2) + '\n')
    service_template = None
    if args.service_template:
        data = args.service_template.read_bytes()
        bridge.require(bridge.hashlib.sha256(data).hexdigest() == before[str(args.service_template)],
                       'service template snapshot changed')
        service = json.loads(data)['service_environment']
        bridge.exact_fields(service, bridge.ENV_KEYS)
        bridge.require(all(type(value) is str and '\0' not in value for value in service.values()),
                       'invalid service environment value')
        service_template = outer / 'service-template.json'
        service_template.write_bytes(data)
        paths += (service_template,)
        before[str(service_template)] = before[str(args.service_template)]
        (outer / 'inputs-before.json').write_text(json.dumps(before, indent=2) + '\n')
    (outer / 'sources').mkdir()
    for path in sources:
        data = path.read_bytes()
        bridge.require(bridge.hashlib.sha256(data).hexdigest() == before[str(path)], 'source snapshot changed')
        destination = outer / 'sources' / path.relative_to(SCRIPT.parent.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    injection_seals = {str((bridge.NCU_DIRECTORY / name).resolve()): before[str(bridge.NCU_DIRECTORY / name)]
                       for name in process_checks.NCU_INJECTION_NAMES}
    results = [collect_one(outer / policy, policy, capture=capture, injection_seals=injection_seals,
                           service_template=service_template)
               for policy in args.policies]
    after = {str(path): bridge.seal(path) for path in paths}
    (outer / 'inputs-after.json').write_text(json.dumps(after, indent=2) + '\n')
    report = dict(scope='CPU only: no CUDA, root transition, or SM qualification', results=results,
                  inputs_unchanged=before == after, capture_options=args.capture_options,
                  service_environment_tested=service_template is not None)
    (outer / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(results), flush=True)
    return 0 if before == after and all(result['status'] == 'cpu_process_compatibility_passed'
                                       for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
