"""Root-owned, fixed-recipe relay for one bounded BERT NCU application replay.

The surrounding collector owns resource limits, complete input seals and exact
cgroup containment. This module never accepts a general executable or signal.
The workspace/operator is trusted; maps/input audits do not isolate a malicious
same-UID writer. Only the fixed service retains CAP_SYS_ADMIN.
"""

from contextlib import ExitStack
import errno
import json
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import socket
import stat
import struct
import subprocess
import time

import bert_ncu_replay_exec as bridge
from bert_sgemm_ncu_process import (NCU_INJECTION_NAMES, NcuMappingUnavailable, proc_environment,
                                   proc_mapped_paths, verified_preload)
import run_bert_candidate_validation as candidate


EXPECTED_REPARSE = dict(tensors=401, elements=217787909, full_tensor_recomparison=True,
                        source_bound_complete_inference=True, first_command_handoff=False)
TERMINAL_CONFIRMATION_MS = 250


def service_disappearance(error_type, error_errno, filename, pid):
    proc_file = (type(filename) is str and re.fullmatch(
        rf'/proc/{pid}/(?:stat|status|cmdline|exe|environ|maps|cgroup|fd/[1-9][0-9]*)', filename) is not None)
    return ((error_type == 'FileNotFoundError' and error_errno == errno.ENOENT and proc_file) or
            (error_type == 'ProcessLookupError' and error_errno == errno.ESRCH and
             (filename is None or proc_file)))


def complete_reparse(value):
    return (type(value) is dict and value == EXPECTED_REPARSE and
            all(type(value[key]) is type(expected) for key, expected in EXPECTED_REPARSE.items()))


def validate_mapping_receipts(session, inputs, *, service=bridge.SERVICE):
    seals = MappedCodeAudit(inputs).seals
    executable = inputs[str(service)]['sha256']
    snapshots = [session[key] for key in ('initial_maps', 'last_maps')]
    bridge.require(all(type(record) is dict and set(record) == {'monotonic_ns', 'mappings'} and
                       bridge.positive(record['monotonic_ns']) for record in snapshots) and
                   snapshots[0]['monotonic_ns'] <= snapshots[1]['monotonic_ns'], 'map audit timestamps differ')
    all_maps = session['observed_mappings']
    for mappings in [all_maps, *[record['mappings'] for record in snapshots]]:
        bridge.require(type(mappings) is dict and mappings, 'nonempty mapped-code evidence required')
        memfd = '/memfd:glimmer-glimmer_execution_service (deleted)'
        bridge.require(memfd in mappings, 'sealed service mapping missing')
        for name, record in mappings.items():
            bridge.exact_fields(record, ('device', 'inode', 'sha256'))
            bridge.require(type(record['device']) is int and record['device'] >= 0 and bridge.positive(record['inode']),
                           'mapped file identity is not exact')
            expected = executable if name == memfd else seals.get(name)
            bridge.require(expected is not None and record['sha256'] == expected, 'mapped code is not a frozen input')
            bridge.require(all_maps.get(name) == record, 'mapped code identity drifted during the replay')
    if 'map_read_failure' in session:
        failure = session['map_read_failure']
        bridge.require(type(failure) is dict and failure.get('terminal_pidfd') is True and
                       type(failure.get('monotonic_ns')) is int and
                       failure['monotonic_ns'] >= snapshots[-1]['monotonic_ns'] and
                       failure['monotonic_ns'] == session.get('terminal_map_read_race_ns') and
                       failure.get('error_type') in ('FileNotFoundError', 'ProcessLookupError',
                                                     'NcuMappingUnavailable'),
                       'terminal map-read race receipt differs')
        bridge.require(type(failure.get('immediate_terminal_pidfd')) is bool and
                       type(failure.get('confirmation_finished_ns')) is int and
                       type(failure.get('confirmation_elapsed_ns')) is int and
                       failure['confirmation_finished_ns'] >= failure['monotonic_ns'] and
                       failure['confirmation_elapsed_ns'] == failure['confirmation_finished_ns'] - failure['monotonic_ns'] and
                       type(failure.get('wait_timeout_ms')) is int and failure['wait_timeout_ms'] ==
                       (0 if failure['immediate_terminal_pidfd'] else TERMINAL_CONFIRMATION_MS),
                       'terminal confirmation interval differs')
        if failure['error_type'] == 'NcuMappingUnavailable':
            for key in ('expected', 'missing', 'mapped'):
                paths = failure.get(key)
                bridge.require(type(paths) is list and all(type(path) is str for path in paths) and
                               paths == sorted(set(paths)) and
                               all(path in seals and Path(path).name in NCU_INJECTION_NAMES for path in paths),
                               'terminal injection inventory is not sealed')
            bridge.require(failure['missing'] and set(failure['missing']) ==
                           set(failure['expected']) - set(failure['mapped']),
                           'terminal injection missing set differs')
        else:
            bridge.require(service_disappearance(failure['error_type'], failure.get('errno'),
                                                 failure.get('filename'), session['service']['pid']),
                           'map-read failure was not service disappearance')


def command_line(pid):
    return [os.fsdecode(value) for value in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[:-1]]


def environment(pid, *, ncu_seals=None):
    return {os.fsdecode(key): os.fsdecode(value)
            for key, value in proc_environment(pid, ncu_seals=ncu_seals).items()}


def parent(pid):
    return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])


def ncu_ancestry(record, group, allowed):
    bridge.live_identity(record, group)
    ancestors, pid = [], parent(record['pid'])
    for _ in range(32):
        if pid <= 1 or bridge.process_group(pid) != group:
            break
        entry = bridge.process_identity(pid)
        ancestors.append(entry)
        if (entry['pid'], entry['start']) in allowed:
            return ancestors
        pid = parent(pid)
    raise ValueError('replay target is not a descendant of the authenticated NCU executable')


def authenticate_process(record, group, uid, gid, *, capability, command=None, executable=None):
    bridge.live_identity(record, group)
    pid = record['pid']
    bridge.validate_status(Path(f'/proc/{pid}/status').read_text(), uid, gid, capability)
    if command is not None:
        bridge.require(command_line(pid) == command, 'actual process command differs')
    if executable is not None:
        bridge.require(Path(f'/proc/{pid}/exe').samefile(executable), 'actual process executable differs')
    bridge.live_identity(record, group)


class MappedCodeAudit:
    def __init__(self, inputs):
        self.seals, self.cache, self.observed = {}, {}, {}
        for record in inputs.values():
            target, digest = record['target'], record['sha256']
            bridge.require(target not in self.seals or self.seals[target] == digest, 'ambiguous input target seal')
            self.seals[target] = digest

    def observe(self, pid, executable):
        """Bind every observed executable file/DSO to a frozen canonical input."""
        paths = {}
        executable_metadata = Path(f'/proc/{pid}/exe').stat()
        content = Path(f'/proc/{pid}/maps').read_text()
        if not content:
            raise ProcessLookupError(errno.ESRCH, 'service mappings disappeared', f'/proc/{pid}/maps')
        for line in content.splitlines():
            fields = line.split(maxsplit=5)
            bridge.require(len(fields) >= 5, 'malformed process mapping')
            name = fields[5] if len(fields) == 6 else ''
            code = 'x' in fields[1]
            if not code and '.so' not in name:
                continue
            if name in ('[vdso]', '[vsyscall]'):
                continue
            device = fields[3].split(':')
            mapped = (os.makedev(int(device[0], 16), int(device[1], 16)), int(fields[4]))
            if name == '/memfd:glimmer-glimmer_execution_service (deleted)':
                bridge.require(mapped == (executable_metadata.st_dev, executable_metadata.st_ino),
                               'mapped service differs from actual sealed executable')
                paths[name] = dict(device=mapped[0], inode=mapped[1], sha256=executable)
                continue
            bridge.require(name.startswith('/') and not name.endswith(' (deleted)'),
                           'unknown anonymous or deleted executable mapping')
            path = Path(name).resolve(strict=True)
            canonical = str(path)
            bridge.require(canonical in self.seals, 'unsealed executable mapping: ' + canonical)
            metadata = path.stat()
            bridge.require(stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == mapped,
                           'actual mapped inode differs from frozen path')
            fingerprint = mapped + (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns)
            if self.cache.get(canonical) != fingerprint:
                bridge.require(bridge.seal(path) == self.seals[canonical], 'mapped library bytes changed')
                self.cache[canonical] = fingerprint
            paths[canonical] = dict(device=mapped[0], inode=mapped[1], sha256=self.seals[canonical])
        bridge.require(any(name.startswith('/memfd:glimmer-glimmer_execution_service') for name in paths),
                       'sealed service code was not mapped')
        self.observed.update(paths)
        return dict(monotonic_ns=time.monotonic_ns(), mappings=paths)


class ReplayRelay:
    def __init__(self, collector, outer, manifest, base, group, deadline, resources, *, fresh_authority=None):
        self.c, self.outer, self.manifest, self.base = collector, outer, manifest, base
        if 'fresh_runtime' in manifest:
            bridge.require(type(fresh_authority) is dict and fresh_authority.get('template') == manifest,
                           'authenticated fresh relay context required')
            bridge.verify_root_claim_receipt(outer, fresh_authority)
            bridge.require(bridge.frozen_template(outer, fresh_authority['manifest_sha256']) == manifest,
                           'fresh relay manifest changed after authentication')
        else:
            bridge.require(fresh_authority is None, 'legacy relay cannot accept fresh authority')
        self.fresh_authority = fresh_authority
        self.service = (bridge.selected_service(manifest, fresh_authority=fresh_authority)
                        if fresh_authority is not None else bridge.selected_service(manifest))
        self.group, self.deadline, self.resources = group, deadline, resources
        self.supervisor = bridge.process_identity(os.getpid())
        self.endpoint = 'glimmer-ncu-' + secrets.token_hex(16)
        self.listener = resources.enter_context(socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET))
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        self.listener.bind('\0' + self.endpoint)
        self.listener.listen(2)
        self.listener.settimeout(1)
        self.code_digest = manifest['bridge_seals'][str(bridge.SCRIPT)]
        self.code_fd = bridge.sealed_snapshot(bridge.SCRIPT, self.code_digest)
        resources.callback(os.close, self.code_fd)
        self.template_sha = bridge.seal(outer / 'manifest.json')
        if fresh_authority is not None:
            bridge.require(self.template_sha == fresh_authority['manifest_sha256'], 'fresh relay manifest differs')
        inputs = json.loads((outer / 'inputs-before.json').read_text())
        self.audit = MappedCodeAudit(inputs)
        self.sessions, self.samples, self.current = [], [], None
        self.last_resource, self.last_maps = 0., 0.
        self.controller = self.ncu = None
        self.executable_identities = set()
        self.ncu_identities = set()

    def target(self):
        return bridge.sealed_python_command(f'/proc/{os.getpid()}/fd/{self.code_fd}', self.code_digest,
            '--endpoint', self.endpoint, '--supervisor', f"{self.supervisor['pid']}:{self.supervisor['start']}",
            '--attempt', str(self.outer))

    def controller_target(self, serial):
        return [self.manifest['python'], '-B', str(bridge.SUPERVISOR), '--sibling-pass', str(self.outer),
                '--endpoint', self.endpoint, '--supervisor',
                f"{self.supervisor['pid']}:{self.supervisor['start']}", '--serial', str(serial)]

    def sample(self, force=False):
        now = time.monotonic()
        bridge.require(now < self.deadline, 'bounded relay deadline exhausted')
        if not force and now - self.last_resource < .2:
            return
        self.last_resource = now
        if 'host_pair' in self.manifest or 'current_sdk' in self.manifest:
            bridge.require(shutil.disk_usage(self.outer).free >= 8 * 1024 ** 3,
                           'counter capture exhausted disk reserve')
        memory = self.c.memory_receipt(self.group)
        events = dict(line.split() for line in memory['memory.events'].splitlines())
        bridge.require(events and all(int(value) == 0 for value in events.values()), 'task memory pressure/OOM')
        sample = dict(monotonic_ns=time.monotonic_ns(),
                      gpu=self.c.check_owned_gpu(self.group, self.manifest['service_environment']['CUDA_VISIBLE_DEVICES']),
                      ncu=self.c.observed_ncu_processes(self.group), output_bytes=candidate.check_output(self.outer))
        self.ncu_identities.update((entry['pid'], entry['start']) for entry in sample['ncu'])
        if not self.samples or any(sample[key] != self.samples[-1][key] for key in ('gpu', 'ncu')):
            self.samples.append(sample)

    def receive(self, connection, expected_credentials):
        message, descriptors, credentials = bridge.receive_packet(connection)
        if credentials != expected_credentials:
            for descriptor in descriptors:
                os.close(descriptor)
            raise ValueError('relay packet sender credentials differ')
        return message, descriptors

    def accept(self):
        connection, _ = self.listener.accept()
        self.resources.enter_context(connection)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        connection.settimeout(min(10., self.deadline - time.monotonic()))
        peer = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        message, descriptors = self.receive(connection, peer)
        for descriptor in descriptors:
            os.close(descriptor)
        bridge.require(not descriptors, 'initial relay packet cannot transfer descriptors')
        if message.get('op') == 'hello':
            bridge.require(self.current is None and peer[1:] == (0, 0), 'unexpected or overlapping root helper')
            bridge.exact_fields(message, ('op', 'helper'))
            helper = message['helper']
            bridge.require(helper['pid'] == peer[0] and command_line(peer[0]) == self.target() and
                           Path(f'/proc/{peer[0]}/exe').samefile(bridge.PYTHON), 'root helper executable/argv differs')
            self.sample(force=True)
            ancestors = ncu_ancestry(helper, self.group, self.ncu_identities)
            bridge.profiler_environment(environment(peer[0], ncu_seals=self.manifest['ncu_injection_seals']),
                                        self.manifest, self.outer)
            descriptor = os.pidfd_open(peer[0])
            self.resources.callback(os.close, descriptor)
            serial = len(self.sessions) + 1
            self.current = dict(serial=serial, helper=helper, helper_fd=descriptor, helper_connection=connection,
                                phase='controller-starting', controller_connection=None, receipt=None,
                                helper_ancestors=ancestors, service=None, plan=None, complete=None)
            env = dict(self.manifest['service_environment'], TMPDIR=str(self.outer / 'ncu-tmp'), CUDA_CACHE_DISABLE='1')
            log = self.resources.enter_context((self.outer / f'controller-{serial:04d}.log').open('x'))
            command = [*bridge.drop_command(self.manifest['uid'], self.manifest['gid'], service=False),
                       *self.controller_target(serial)]
            self.controller = subprocess.Popen(command, env=env, cwd=self.outer, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.current['controller'] = bridge.process_identity(self.controller.pid)
            self.current['controller_environment'] = env
        else:
            current = self.current
            bridge.require(current is not None and current['phase'] == 'controller-starting' and
                           peer == (current['controller']['pid'], self.manifest['uid'], self.manifest['gid']) and
                           message == dict(op='controller', serial=current['serial'], controller=current['controller']),
                           'unexpected sibling controller connection')
            authenticate_process(current['controller'], self.group, self.manifest['uid'], self.manifest['gid'],
                                 capability=False, command=self.controller_target(current['serial']),
                                 executable=Path(self.manifest['python']))
            bridge.require(parent(peer[0]) == os.getpid() and not proc_mapped_paths(peer[0]) and
                           environment(peer[0]) == current['controller_environment'],
                           'controller is not a clean direct sibling')
            current.update(controller_connection=connection, phase='await-launch')
            self.reply('controller')

    def reply(self, operation, **values):
        bridge.send_packet(self.current['controller_connection'], dict(op=operation, serial=self.current['serial'], **values))

    def launch(self, message, descriptors):
        current = self.current
        bridge.exact_fields(message, ('op', 'serial', 'directory', 'socket', 'clients'))
        bridge.require(current['phase'] == 'await-launch' and len(descriptors) == 1, 'one fresh service launch required')
        plan = dict(message, op='launch', uid=self.manifest['uid'], gid=self.manifest['gid'],
                    environment=self.manifest['service_environment'], seals=self.manifest['bridge_seals'],
                    deadline_ns=int(self.deadline * 10 ** 9), template_sha256=self.template_sha)
        if self.fresh_authority is not None:
            plan['fresh_claim_sha256'] = self.fresh_authority['claim_sha256']
        checked = bridge.validate_launch(plan, self.outer, self.group, descriptors[0])
        if self.fresh_authority is not None:
            bridge.require(isinstance(checked, tuple) and len(checked) == 2 and
                           checked[1] == self.fresh_authority, 'fresh relay replay authority changed')
        bridge.require(Path(plan['directory']).name.startswith(f"pass-{current['serial']:04d}-"),
                       'replay directory serial differs')
        current['client_fds'] = {}
        for role, identity in plan['clients'].items():
            authenticate_process(identity, self.group, plan['uid'], plan['gid'], capability=False,
                                 command=self.manifest['client_wrapper_command'], executable=Path(self.manifest['python']))
            bridge.require(parent(identity['pid']) == current['controller']['pid'] and
                           environment(identity['pid']) == self.manifest['service_environment'] and
                           not proc_mapped_paths(identity['pid']), 'client is not an unchanged clean local wrapper')
            descriptor = os.pidfd_open(identity['pid'])
            self.resources.callback(os.close, descriptor)
            current['client_fds'][role] = descriptor
        current.update(plan=plan, phase='spawning')
        bridge.send_packet(current['helper_connection'], plan, descriptors)

    def authenticate_service(self):
        current, manifest = self.current, self.manifest
        service, plan = current['service'], current['plan']
        authenticate_process(service, self.group, manifest['uid'], manifest['gid'], capability=True)
        pid = service['pid']
        bridge.require(parent(pid) == current['helper']['pid'] and os.getsid(pid) == os.getpgid(pid) == pid,
                       'service parenthood or private session differs')
        argv = command_line(pid)
        if not argv:
            # procfs can drop the mm/argv before the existing pidfd signals exit.
            # The sampling caller must still confirm that same pidfd and obtain
            # the real wait receipt; an empty command is never authentication.
            raise ProcessLookupError(errno.ESRCH, 'service command line unavailable during authentication',
                                     f'/proc/{pid}/cmdline')
        expected = bridge.service_command(plan, service=self.service,
                                          signal_binding=bridge.selected_signal_binding(manifest))
        valid_command = (re.fullmatch(r'/proc/self/fd/[1-9][0-9]*', argv[0]) is not None and
                         argv[1:] == expected[1:])
        if not valid_command:
            current['command_line_failure'] = dict(monotonic_ns=time.monotonic_ns(),
                                                  service=service, argv=argv, expected_tail=expected[1:])
        bridge.require(valid_command, 'fixed sealed service command differs')
        with Path(f'/proc/{pid}/exe').open('rb') as executable:
            metadata = os.fstat(executable.fileno())
            fingerprint = (pid, service['start'], metadata.st_dev, metadata.st_ino, metadata.st_size)
            bridge.require(bridge.fcntl.fcntl(executable.fileno(), bridge.fcntl.F_GET_SEALS) &
                           bridge.SNAPSHOT_SEALS == bridge.SNAPSHOT_SEALS, 'service executable lost its kernel seals')
            if fingerprint not in self.executable_identities:
                bridge.validate_snapshot(executable.fileno(), plan['seals'][str(self.service)])
                self.executable_identities.add(fingerprint)
            bridge.require(Path(f"/proc/{pid}/fd/{argv[0].rsplit('/', 1)[1]}").samefile(f'/proc/{pid}/exe'),
                           'service inherited executable descriptor differs')
        actual = environment(pid, ncu_seals=manifest['ncu_injection_seals'])
        self.validate_service_environment(actual)
        verified_preload(actual.get('LD_PRELOAD'), None, manifest['ncu_injection_seals'], proc_mapped_paths(pid))
        sample = self.audit.observe(pid, plan['seals'][str(self.service)])
        current.setdefault('initial_maps', sample)
        current['last_maps'] = sample
        observed = current.setdefault('observed_mappings', {})
        for name, record in sample['mappings'].items():
            bridge.require(name not in observed or observed[name] == record, 'mapped code identity changed within replay')
            observed[name] = record
        bridge.live_identity(service, self.group)

    def validate_service_environment(self, actual):
        manifest = self.manifest
        for key, value in manifest['service_environment'].items():
            bridge.require(actual.get(key) == value, 'service baseline environment differs: ' + key)
        bridge.require(actual.get('TMPDIR') == str(self.outer / 'ncu-tmp') and
                       actual.get('CUDA_CACHE_DISABLE') == '1' and actual.get('NVTX_INJECTION64_PATH') == 'libcuda-injection.so' and
                       'LD_AUDIT' not in actual and not any(key.startswith(('GLIMMER_', 'PYTHONPATH')) for key in actual),
                       'service profiler environment differs')
        # The service baseline was checked above; only NCU additions belong in
        # the profiler validator. Do not overwrite unrelated profiler values.
        service_keys = manifest['service_environment'].keys()
        normalized = {key: value for key, value in actual.items() if key not in service_keys}
        normalized.update({key: value for key, value in self.base.items() if key in service_keys})
        normalized['LD_LIBRARY_PATH'] = str(bridge.NCU_DIRECTORY)
        bridge.profiler_environment(normalized, manifest, self.outer)

    def settle_service_exec(self):
        """Wait only for the authenticated setpriv/env chain, never for unknown code."""
        current = self.current
        service, plan = current['service'], current['plan']
        until = min(self.deadline, time.monotonic() + 10)
        while time.monotonic() < until:
            bridge.require(not bridge.terminal(current['service_fd']), 'service exited before final exec')
            bridge.live_identity(service, self.group)
            pid = service['pid']
            bridge.require(parent(pid) == current['helper']['pid'] and os.getsid(pid) == os.getpgid(pid) == pid,
                           'launch-chain identity or session differs')
            path = Path(f'/proc/{pid}/exe')
            with path.open('rb') as executable:
                # Pin the opened image across exec; /proc/PID/exe can change
                # between any two reads of the short-lived wrapper chain.
                pinned = Path(f'/proc/self/fd/{executable.fileno()}')
                before = os.fstat(executable.fileno())
                destination = os.readlink(pinned)
                argv = command_line(pid)
                after = path.stat()
                if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                    continue
                if destination == '/memfd:glimmer-glimmer_execution_service (deleted)':
                    self.authenticate_service()
                    return
                wrapper = next((entry for entry in (bridge.SETPRIV, bridge.ENV) if pinned.samefile(entry)), None)
                bridge.require(wrapper is not None and bridge.seal(pinned) == plan['seals'][str(wrapper)],
                               'unexpected executable during fixed service launch')
            if wrapper == bridge.SETPRIV:
                drop = bridge.drop_command(plan['uid'], plan['gid'], service=True)
                bridge.require(argv[:len(drop)] == drop, 'setpriv launch recipe differs')
                argv = argv[len(drop):]
            else:
                authenticate_process(service, self.group, plan['uid'], plan['gid'], capability=True)
            bridge.require(argv[:2] == [str(bridge.ENV), '-i'], 'fixed env wrapper required')
            assignments, index = {}, 2
            while index < len(argv) and '=' in argv[index]:
                key, value = argv[index].split('=', 1)
                bridge.require(key not in assignments, 'duplicate wrapper environment key')
                assignments[key] = value
                index += 1
            expected = bridge.service_command(plan, service=self.service,
                                              signal_binding=bridge.selected_signal_binding(self.manifest))
            bridge.require(index < len(argv) and re.fullmatch(r'/proc/self/fd/[1-9][0-9]*', argv[index]) is not None and
                           argv[index + 1:] == expected[1:], 'wrapper final service recipe differs')
            self.validate_service_environment(assignments)
            self.sample()
            time.sleep(.01)
        raise TimeoutError('fixed service did not complete its bounded final exec')

    def sample_service_maps(self):
        current = self.current
        descriptor = current['service_fd']
        if bridge.terminal(descriptor):
            return
        try:
            self.authenticate_service()
        except (FileNotFoundError, ProcessLookupError, NcuMappingUnavailable) as error:
            # Only disappearance may race exit. Duplicate/unknown mappings,
            # changed bytes and credentials remain fatal even during shutdown.
            failure = dict(monotonic_ns=time.monotonic_ns(), error_type=type(error).__name__)
            current['map_read_failure'] = failure
            if isinstance(error, NcuMappingUnavailable):
                failure.update(expected=error.expected, missing=error.missing, mapped=error.mapped)
            else:
                failure.update(errno=error.errno, filename=error.filename)
                if not service_disappearance(failure['error_type'], error.errno, error.filename,
                                             current['service']['pid']):
                    raise
            immediate = bridge.terminal(descriptor)
            failure.update(immediate_terminal_pidfd=immediate, terminal_pidfd=immediate,
                           wait_timeout_ms=0 if immediate else TERMINAL_CONFIRMATION_MS)
            bridge.require(current.get('initial_maps') and current.get('last_maps'),
                           'service disappeared without prior mapped-code evidence')
            if not immediate:
                # procfs teardown can precede pidfd readiness. Wait on the
                # already authenticated pidfd, without inferring an exit code.
                poller = select.poll()
                poller.register(descriptor, select.POLLIN)
                events = poller.poll(TERMINAL_CONFIRMATION_MS)
                bridge.require(all(fd == descriptor and flags & select.POLLIN and
                                   not flags & ~(select.POLLIN | select.POLLHUP) for fd, flags in events),
                               'terminal confirmation returned an invalid pidfd event')
                bridge.require(current['service_fd'] == descriptor, 'service pidfd changed during confirmation')
                failure['terminal_pidfd'] = bridge.terminal(descriptor)
            failure['confirmation_finished_ns'] = time.monotonic_ns()
            failure['confirmation_elapsed_ns'] = failure['confirmation_finished_ns'] - failure['monotonic_ns']
            bridge.require(failure['terminal_pidfd'] and current.get('initial_maps') and current.get('last_maps'),
                           'service disappeared without terminal pidfd or prior mapped-code evidence')
            current['terminal_map_read_race_ns'] = failure['monotonic_ns']

    def helper_message(self):
        current = self.current
        root_phase = current['phase'] == 'spawning'
        credentials = (current['helper']['pid'], 0, 0) if root_phase else (
            current['helper']['pid'], self.manifest['uid'], self.manifest['gid'])
        message, descriptors = self.receive(current['helper_connection'], credentials)
        for descriptor in descriptors:
            os.close(descriptor)
        bridge.require(not descriptors and type(message.get('serial')) is int and
                       message['serial'] == current['serial'], 'helper replay identity differs')
        if message.get('op') == 'spawned':
            bridge.exact_fields(message, ('op', 'serial', 'child'))
            bridge.require(root_phase, 'unexpected spawned receipt')
            service = message['child']
            bridge.live_identity(service, self.group)
            descriptor = os.pidfd_open(service['pid'])
            self.resources.callback(os.close, descriptor)
            current.update(service=service, service_fd=descriptor, phase='await-waiter')
        elif message.get('op') == 'waiter_ready':
            bridge.exact_fields(message, ('op', 'serial', 'waiter', 'child'))
            bridge.require(current['phase'] == 'await-waiter' and message['waiter'] == current['helper'] and
                           message['child'] == current['service'], 'waiter/service identities differ')
            authenticate_process(current['helper'], self.group, self.manifest['uid'], self.manifest['gid'],
                                 capability=False, executable=bridge.PYTHON)
            argv = command_line(current['helper']['pid'])
            bridge.require(len(argv) == 27 and re.fullmatch(r'/proc/self/fd/[1-9][0-9]*', argv[7]) is not None and
                           argv[:9] == bridge.sealed_python_command(argv[7], self.code_digest), 'waiter sealed loader differs')
            arguments = dict(zip(argv[9::2], argv[10::2]))
            for option in ('--wait-fd', '--child-fd'):
                bridge.require(re.fullmatch(r'[1-9][0-9]*', arguments.get(option, '')) is not None and
                               int(arguments[option]) > 2, 'invalid inherited waiter descriptor')
            expected = ['--wait-fd', arguments['--wait-fd'], '--supervisor',
                        f"{self.supervisor['pid']}:{self.supervisor['start']}", '--attempt', str(self.outer),
                        '--child', f"{current['service']['pid']}:{current['service']['start']}",
                        '--child-fd', arguments['--child-fd'], '--uid', str(self.manifest['uid']),
                        '--gid', str(self.manifest['gid']), '--serial', str(current['serial']),
                        '--deadline-ns', str(current['plan']['deadline_ns'])]
            bridge.require(argv[9:] == expected, 'waiter exact command differs')
            with Path(f"/proc/{current['helper']['pid']}/fd/{argv[7].rsplit('/', 1)[1]}").open('rb') as source:
                bridge.validate_snapshot(source.fileno(), self.code_digest)
            child_info = Path(f"/proc/{current['helper']['pid']}/fdinfo/{arguments['--child-fd']}").read_text()
            bridge.require(re.findall(r'^Pid:\s+([0-9]+)$', child_info, re.MULTILINE) == [str(current['service']['pid'])],
                           'waiter does not hold the actual child pidfd')
            self.settle_service_exec()
            current['phase'] = 'running'
            bridge.send_packet(current['helper_connection'], dict(op='wait', serial=current['serial']))
            self.reply('launch', service=current['service'])
        else:
            bridge.exact_fields(message, ('op', 'serial', 'child'))
            bridge.require(message['op'] == 'exit' and current['phase'] == 'running', 'unexpected service exit receipt')
            receipt = message['child']
            expected = bridge.wait_receipt(current['service']['pid'], current['service']['start'], receipt['raw_status'])
            bridge.require(receipt == expected and type(receipt['returncode']) is int and
                           receipt['exit_code_observed'] is True and receipt['returncode'] == 0 and
                           bridge.terminal(current['service_fd']), 'service did not exit normally with a real wait receipt')
            current.update(receipt=receipt, phase='oracle')

    def controller_message(self):
        current = self.current
        message, descriptors = self.receive(current['controller_connection'],
            (current['controller']['pid'], self.manifest['uid'], self.manifest['gid']))
        try:
            bridge.require(type(message.get('serial')) is int and message['serial'] == current['serial'],
                           'controller replay serial differs')
            operation = message.get('op')
            if operation == 'launch':
                self.launch(message, descriptors)
                return
            bridge.require(not descriptors, 'only launch may transfer a descriptor')
            if operation in ('poll', 'verify', 'abort'):
                bridge.exact_fields(message, ('op', 'serial'))
                bridge.require(current['phase'] in ('running', 'oracle'), 'service operation before authenticated launch')
                bridge.require(operation != 'abort', 'application aborted; contain the exact replay unit')
                if operation == 'verify':
                    self.authenticate_service()
                    self.reply('verify', service=current['service'], verified=True)
                else:
                    self.reply('poll', receipt=current['receipt'])
            else:
                bridge.exact_fields(message, ('op', 'serial', 'directory', 'accepted', 'report_sha256'))
                bridge.require(operation == 'complete' and message['accepted'] is True and current['phase'] == 'oracle' and
                               message['directory'] == current['plan']['directory'], 'full application did not complete')
                current['complete'] = message
                self.reply('complete')
                current['phase'] = 'controller-exit'
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def finish_controller(self):
        current = self.current
        status = self.controller.poll()
        if status is None:
            return
        bridge.require(status == 0 and current['phase'] == 'controller-exit', 'controller exited before full oracle')
        directory = Path(current['plan']['directory'])
        bridge.require(bridge.seal(directory / 'pass.json') == current['complete']['report_sha256'], 'final replay report changed')
        report = json.loads((directory / 'pass.json').read_text())
        bridge.require(report['status'] == 'pass_requires_profiler_review' and complete_reparse(report['reparse']) and
                       report['controller_identity'] == current['controller'] and report['serial'] == current['serial'] and
                       report['topology'] == 'clean-ncu-sibling' and report['pass_directory'] == str(directory),
                       'complete numerical replay receipt differs')
        job = report['job']
        candidate.normal_cleanup(job)
        bridge.require(job['application_identities'] == current['plan']['clients'] and
                       job['server_identity'] == current['service'] and
                       all(bridge.terminal(descriptor) for descriptor in current['client_fds'].values()) and
                       bridge.terminal(current['service_fd']) and
                       all(bridge.seal(directory / 'equal' / name) == digest
                           for name, digest in job['artifact_identities'].items()),
                       'replay participants did not retire or artifact identities changed')
        bridge.send_packet(current['helper_connection'], dict(op='finish', serial=current['serial'], accepted=True))
        current['phase'] = 'waiter-exit'
        current['controller_returncode'] = status

    def retire_waiter(self):
        current = self.current
        if not bridge.terminal(current['helper_fd']):
            return
        bridge.require(current['phase'] == 'waiter-exit', 'waiter exited before full replay acknowledgement')
        self.sessions.append({key: current[key] for key in ('serial', 'helper', 'helper_ancestors', 'controller',
            'controller_returncode', 'service', 'receipt', 'plan', 'initial_maps', 'last_maps', 'observed_mappings', 'complete')})
        for key in ('map_read_failure', 'terminal_map_read_race_ns'):
            if key in current:
                self.sessions[-1][key] = current[key]
        current['helper_connection'].close()
        current['controller_connection'].close()
        self.current, self.controller = None, None

    def run(self):
        command = self.c.ncu_command(self.outer, self.manifest)
        command = [*command[:command.index('/usr/bin/setpriv')], *self.target()]
        log = self.resources.enter_context((self.outer / 'ncu-supervisor.log').open('x'))
        self.sample(force=True)
        self.ncu = subprocess.Popen(command, cwd=self.outer, env=self.base, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        while True:
            self.sample()
            if self.current is not None:
                if self.current['phase'] != 'waiter-exit':
                    self.finish_controller()
                self.retire_waiter()
            if self.ncu.poll() is not None:
                bridge.require(self.ncu.returncode == 0 and self.current is None and self.sessions,
                               'NCU exited before every complete replay retired')
                return dict(returncode=0, samples=self.samples, sessions=self.sessions, command=command)
            watched = [self.listener]
            if self.current is not None:
                if self.current['phase'] not in ('waiter-exit',):
                    watched.append(self.current['helper_connection'])
                if self.current['controller_connection'] is not None and self.current['phase'] != 'controller-exit' and \
                        self.current['phase'] != 'waiter-exit':
                    watched.append(self.current['controller_connection'])
            readable, _, _ = select.select(watched, [], [], .02)
            for connection in readable:
                if connection is self.listener:
                    # NCU can enqueue the successor just before old pidfd polling.
                    if self.current is not None and self.current['phase'] == 'waiter-exit':
                        self.retire_waiter()
                    self.accept()
                elif connection is self.current['helper_connection']:
                    self.helper_message()
                else:
                    self.controller_message()
            if self.current is not None and self.current['phase'] == 'running' and time.monotonic() - self.last_maps > .2:
                self.last_maps = time.monotonic()
                self.sample_service_maps()


def capture(collector, outer, manifest, environment, group, deadline, *, fresh_authority=None):
    with ExitStack() as resources:
        relay = ReplayRelay(collector, outer, manifest, environment, group, deadline, resources,
                            fresh_authority=fresh_authority)
        try:
            return relay.run()
        finally:
            # The collector persists any exception and kills its exact cgroup.
            # Do not discard completed passes or the failing pass's last maps.
            snapshot = dict(completed=relay.sessions, samples=relay.samples,
                            current=None if relay.current is None else {key: value for key, value in relay.current.items()
                                if key not in ('helper_connection', 'controller_connection', 'helper_fd', 'service_fd', 'client_fds')})
            collector.save(outer / 'replay-relay.json', snapshot)
