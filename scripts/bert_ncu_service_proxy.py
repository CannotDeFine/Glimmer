"""Unprivileged sibling-controller adapter for the fixed BERT NCU replay.

Only service launch/verification/cleanup crosses the authenticated root relay.
Clients retain their original local Popen, waitpid, numerical and cleanup path.
No pidfd or profiler status is converted into a successful service exit code.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time

import bert_ncu_replay_exec as bridge
from bert_sgemm_ncu_process import proc_environment, proc_mapped_paths

FRONTEND_PERMISSIONS = ['r--p', 'r--p', 'r--p', 'r-xp', 'rw-p']


class LogicalRelease:
    """Recognize one ordinary-mode logical completion, never a block observation."""

    def __init__(self, session, pid, start, sequence):
        bridge.require(sequence == 2842 and all(bridge.positive(value) for value in (session, pid, start)),
                       'fixed logical release identity required')
        self.session, self.pid, self.start, self.sequence = session, pid, start, sequence
        self.issued, self.done = False, False

    def __call__(self, original, position, pid, start, admissions, selected_sequence=None):
        from execution_priority_audit import BLOCK_MARKER, RECORDS

        bridge.require((pid, start, selected_sequence) == (self.pid, self.start, self.sequence),
                       'logical release identity changed')
        line = original.rstrip('\r\n')
        bridge.require(BLOCK_MARKER not in line, 'ordinary release cannot use a block observation')
        if 'Native admission audit' in line:
            admission = RECORDS['Native admission audit'].fullmatch(line)
            bridge.require(admission is not None, 'malformed logical release admission')
            session, sequence, operation = map(int, admission.groups()[1:])
            bridge.require((session, sequence) not in admissions, 'duplicate logical release admission')
            admissions[session, sequence] = operation
            return None
        if 'Priority execution audit' not in line:
            return None
        match = RECORDS['Priority execution audit'].fullmatch(line)
        bridge.require(match is not None, 'malformed logical release event')
        event, session, priority, sequence, completed, total, launches, status = match.groups()[1:]
        session, priority, sequence, completed, total, launches, status = map(
            int, (session, priority, sequence, completed, total, launches, status))
        if (session, sequence) != (self.session, self.sequence):
            return None
        bridge.require(admissions.get((session, sequence)) == 42 and priority == 0 and total == 2304 and
                       launches == 1 and status == 0, 'logical release admission or geometry differs')
        if event == 'low_issued':
            bridge.require(not self.issued and not self.done and completed == 0,
                           'duplicate or invalid logical release issue')
            self.issued = True
            return None
        if event == 'low_logical_done':
            bridge.require(self.issued and not self.done and completed == total,
                           'duplicate or invalid logical release completion')
            self.done = True
            return dict(kind='logical_done', session=session, sequence=sequence, operation=42, priority=priority,
                        pid=pid, start=start, completed=completed, total=total, issued_physical_launches=launches,
                        physical_completion_observed=False, useful_compute_proof=False,
                        pre_demand_compute_observed=False, provisional=True,
                        position=position + match.start('record'), end_position=position + len(original))
        raise ValueError('unexpected logical release event')


def ordinary_release_session(content, pid, start):
    from execution_semantic_audit import parse_semantic_audit

    audit = parse_semantic_audit(content)
    bindings = [entry for entry in audit.family(4) if entry.value('event') == 1 and
                entry.value('role') == 0 and entry.value('effective') == 1 and
                entry.value('retained') == 1 and entry.value('kernel_mode') == 2 and
                (entry.value('pid'), entry.value('start'), entry.value('uid'), entry.value('priority')) ==
                (pid, start, os.getuid(), 0)]
    bridge.require(len(bindings) == 1, 'one authenticated configured low release session required')
    return bindings[0].value('connection')


def validate_ordinary_release(content, job, target_binding):
    """Reparse the full log and join the release to the authenticated target owner."""
    offset = job['priority_audit_start']
    low = job['application_identities']['low']
    bridge.require(type(offset) is int and 0 <= offset <= len(content) and
                   job.get('mode') == 'equal', 'ordinary release requires the equal-mode log boundary')
    session = ordinary_release_session(content[:offset], low['pid'], low['start'])
    bridge.require(session == target_binding['session'], 'release and target owners differ')
    parser = LogicalRelease(session, low['pid'], low['start'], 2842)
    admissions, matches, position = {}, [], offset
    for line in content[offset:].splitlines(keepends=True):
        bridge.require(line.endswith('\n'), 'incomplete ordinary release audit line')
        result = parser(line, position, low['pid'], low['start'], admissions, 2842)
        if result is not None:
            matches.append(result)
        position += len(line)
    bridge.require(len(matches) == 1 and matches[0] == job.get('high_release_trigger') and
                   job.get('pre_demand_compute_observed') is not True,
                   'ordinary logical release receipt differs from the complete audit')
    return matches[0]


class DeferredLaunch:
    """Hold one wrapper's existing launch message until the low client is ready."""

    def __init__(self, stream):
        self.stream, self.message, self.queued_ns, self.released_ns = stream, None, None, None
        self.state = 'never_released'

    def write(self, message):
        bridge.require(self.message is None and type(message) is str and len(message.encode()) <= 1024 * 1024 and
                       message.endswith('\n') and '\n' not in message[:-1], 'one bounded launch message required')
        value = json.loads(message)
        bridge.require(type(value) is dict and set(value) == {'command', 'environment'},
                       'unexpected deferred launch message')
        self.message = message
        return len(message)

    def flush(self):
        bridge.require(self.message is not None and self.queued_ns is None, 'invalid deferred launch flush')
        self.queued_ns = time.monotonic_ns()

    def release(self):
        bridge.require(self.queued_ns is not None and self.state == 'never_released', 'invalid deferred release')
        self.state = 'release_uncertain'
        bridge.require(self.stream.write(self.message) == len(self.message), 'partial deferred launch write')
        self.stream.flush()
        self.released_ns = time.monotonic_ns()
        self.state = 'released_once'

    def close(self):
        self.stream.close()


def validate_launch_order(job, receipt):
    bridge.exact_fields(receipt, ('low', 'high', 'low_authenticated_ns', 'high_queued_ns', 'high_verified_ns',
                                  'high_release_ns', 'message_bytes', 'message_sha256', 'state', 'no_children',
                                  'no_sockets'))
    clients, ready, launched = job['application_identities'], job['ready'], job['application_launch_ns']
    message = (json.dumps(dict(command=job['application_commands']['high'],
                               environment=job['application_environments']['high'])) + '\n').encode()
    bridge.require(receipt['low'] == clients['low'] and receipt['high'] == clients['high'] and
                   receipt['low_authenticated_ns'] == ready['low']['authenticated_ns'] and
                   receipt['state'] == 'released_once' and receipt['no_children'] is True and
                   receipt['no_sockets'] is True and type(receipt['message_bytes']) is int and
                   receipt['message_bytes'] == len(message) and
                   receipt['message_sha256'] == hashlib.sha256(message).hexdigest() and
                   all(type(receipt[key]) is int for key in ('low_authenticated_ns', 'high_queued_ns',
                                                              'high_verified_ns', 'high_release_ns')) and
                   launched['high'] <= receipt['high_queued_ns'] <= receipt['low_authenticated_ns'] <=
                   receipt['high_verified_ns'] <= receipt['high_release_ns'] <= ready['high']['ready_ns'] <=
                   ready['high']['authenticated_ns'] and
                   ready['low']['authenticated_ns'] <= ready['low']['released_ns'] and
                   ready['high']['authenticated_ns'] <= ready['low']['released_ns'] <= ready['high']['released_ns'],
                   'authenticated replay launch and start-gate order differs')


def validate_frontend_receipts(product, service_environment, job, receipts):
    """Check the recorded ready observations against the exact replay job."""
    clients, environments = job['application_identities'], job['application_environments']
    bridge.require(type(clients) is dict and type(environments) is dict and type(receipts) is dict and
                   set(clients) == set(environments) == set(receipts) == {'low', 'high'},
                   'two frontend application roles required')
    for identity in clients.values():
        bridge.validate_identity(identity)
    bridge.require(len({item['pid'] for item in clients.values()}) == 2, 'distinct application PIDs required')
    server = bridge.validate_identity(job['server_identity'])
    bridge.require(type(service_environment) is dict and job['server_environment'] == service_environment and
                   type(job['socket_path']) is str and job['socket_path'].startswith('/'),
                   'reviewed service environment and socket required')
    expected_environment = dict(service_environment, LD_PRELOAD=product['path'],
                                GLIMMER_EXECUTION_SOCKET=job['socket_path'],
                                GLIMMER_EXECUTION_SERVER_PID=str(server['pid']),
                                GLIMMER_EXECUTION_SERVER_START=str(server['start']),
                                GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='240000',
                                GLIMMER_EXECUTION_HOST_RANGE_SEEK='0')
    bridge.require(job['application_environment'] == expected_environment and
                   all(value == expected_environment for value in environments.values()),
                   'frontend environment differs from reviewed service and replay identities')
    for role, identity in clients.items():
        receipt = receipts[role]
        bridge.exact_fields(receipt, ('pid', 'start', 'path', 'device', 'inode', 'maps', 'seek',
                                      'permissions', 'environment'))
        permissions = receipt['permissions']
        bridge.require(type(permissions) is list and permissions == FRONTEND_PERMISSIONS and
                       receipt == dict(identity, path=product['path'], device=product['device'],
                                       inode=product['inode'], maps=5, seek='0', permissions=permissions,
                                       environment=environments[role]) and
                       environments[role].get('LD_PRELOAD') == product['path'] and
                       environments[role].get('GLIMMER_EXECUTION_HOST_RANGE_SEEK') == '0' and
                       'LD_AUDIT' not in environments[role] and
                       not any(key.startswith(('NV_', 'NVTX_', 'NVIDIA-PROCESS-', 'NVIDIA_PROCESS_'))
                               for key in environments[role]), 'frontend observation differs from replay job')


class FrontendMappings:
    """Record the two authenticated ready-time frontend mappings, not whole-run coverage."""

    def __init__(self, product, service_environment):
        bridge.exact_fields(product, ('path', 'device', 'inode', 'size', 'mtime_ns', 'ctime_ns',
                                      'mode', 'uid', 'gid', 'sha256'))
        bridge.require(type(product['path']) is str and type(product['sha256']) is str and
                       re.fullmatch(r'[0-9a-f]{64}', product['sha256']) is not None and
                       all(type(product[key]) is int and product[key] >= 0 for key in
                           ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns', 'mode', 'uid', 'gid')) and
                       product['inode'] > 0 and product['size'] > 0, 'valid frontend product required')
        self.product = dict(product)
        self.service_environment = dict(service_environment)
        self.observed = {}

    def observe(self, process, start, environment):
        product, path = self.product, Path(self.product['path'])
        identity = dict(pid=process.pid, start=start)
        bridge.validate_identity(identity)
        bridge.require(bridge.process_identity(process.pid) == identity and path.is_absolute() and
                       path == path.resolve(strict=True), 'frontend client or path changed')
        def check_product():
            metadata = path.lstat()
            bridge.require(stat.S_ISREG(metadata.st_mode) and
                           (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns,
                            metadata.st_ctime_ns, stat.S_IMODE(metadata.st_mode), metadata.st_uid, metadata.st_gid) ==
                           (product['device'], product['inode'], product['size'], product['mtime_ns'],
                            product['ctime_ns'], product['mode'], product['uid'], product['gid']),
                           'frontend product identity changed')

        check_product()
        actual = proc_environment(process.pid)
        bridge.require(type(environment) is dict and
                       actual == {os.fsencode(key): os.fsencode(value) for key, value in environment.items()} and
                       actual.get(b'LD_PRELOAD') == os.fsencode(path) and
                       actual.get(b'GLIMMER_EXECUTION_HOST_RANGE_SEEK') == b'0' and b'LD_AUDIT' not in actual and
                       not any(key.startswith((b'NV_', b'NVTX_', b'NVIDIA-PROCESS-', b'NVIDIA_PROCESS_'))
                               for key in actual), 'frontend client environment differs')
        selected = []
        for line in Path(f'/proc/{process.pid}/maps').read_text().splitlines():
            row = line.split(maxsplit=5)
            bridge.require(len(row) >= 5 and row[4].isdecimal(), 'malformed client mapping')
            device = row[3].split(':')
            bridge.require(len(device) == 2 and all(re.fullmatch(r'[0-9a-fA-F]+', item) for item in device),
                           'malformed client mapping device')
            mapped_device = os.makedev(*(int(item, 16) for item in device))
            name = row[5] if len(row) == 6 else ''
            same_file = (mapped_device, int(row[4])) == (product['device'], product['inode'])
            if same_file or name == str(path) or Path(name.removesuffix(' (deleted)')).name == path.name:
                bridge.require(same_file and name == str(path) and
                               re.fullmatch(r'[r-][w-][x-][ps]', row[1]) is not None and
                               not ('w' in row[1] and 'x' in row[1]),
                               'unexpected frontend mapping identity or permissions')
                selected.append(row[1])
        check_product()
        bridge.require(sorted(selected) == FRONTEND_PERMISSIONS and
                       bridge.process_identity(process.pid) == identity,
                       'incomplete frontend mappings or client incarnation changed')
        receipt = dict(identity, path=str(path), device=product['device'], inode=product['inode'],
                       maps=5, seek='0', permissions=sorted(selected), environment=dict(environment))
        key = (process.pid, start)
        bridge.require(not any(pid == process.pid and previous != start for pid, previous in self.observed) and
                       (key not in self.observed or self.observed[key] == receipt),
                       'frontend mapping changed between ready checks')
        self.observed[key] = receipt

    def for_job(self, job):
        clients = job['application_identities']
        bridge.require(type(clients) is dict and set(clients) == {'low', 'high'}, 'two application roles required')
        for identity in clients.values():
            bridge.validate_identity(identity)
        keys = {(item['pid'], item['start']) for item in clients.values()}
        bridge.require(len(keys) == len({item['pid'] for item in clients.values()}) == 2 and
                       keys == set(self.observed), 'two exact frontend mappings required')
        receipts = {role: self.observed[(item['pid'], item['start'])] for role, item in clients.items()}
        validate_frontend_receipts(self.product, self.service_environment, job, receipts)
        return receipts


class RelayConnection:
    def __init__(self, outer, endpoint, supervisor, serial, deadline_ns):
        bridge.require(re.fullmatch(r'glimmer-ncu-[0-9a-f]{32}', endpoint) is not None and
                       bridge.positive(serial), 'private relay endpoint and replay serial required')
        self.outer, self.supervisor, self.serial = outer, supervisor, serial
        self.deadline_ns, self.supervisor_fd, self.connection = deadline_ns, None, None
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
            connection.settimeout(30)
            connection.connect('\0' + endpoint)
            _, self.supervisor_fd = bridge.authenticate_supervisor(connection, supervisor, outer)
            self.connection = connection
            self.request('controller', controller=bridge.process_identity(os.getpid()))
        except BaseException:
            connection.close()
            if self.supervisor_fd is not None:
                os.close(self.supervisor_fd)
            raise

    def request(self, operation, *, descriptors=(), **fields):
        remaining = (self.deadline_ns - time.monotonic_ns()) / 10 ** 9
        bridge.require(remaining > 0 and not bridge.terminal(self.supervisor_fd),
                       'relay deadline exhausted or supervisor terminated')
        self.connection.settimeout(min(30., remaining))
        bridge.send_packet(self.connection, dict(op=operation, serial=self.serial, **fields), descriptors)
        reply = bridge.receive_no_fds(self.connection, (self.supervisor['pid'], 0, 0))
        bridge.require(reply.get('op') == operation and reply.get('serial') == self.serial and
                       not bridge.terminal(self.supervisor_fd), 'reply operation, serial or supervisor differs')
        return reply

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.supervisor_fd is not None:
            os.close(self.supervisor_fd)
            self.supervisor_fd = None


class ServiceProcessProxy:
    def __init__(self, relay, identity, command, environment):
        bridge.validate_identity(identity)
        self.relay, self.pid, self.start, self.args = relay, identity['pid'], identity['start'], command
        self.returncode, self.receipt = None, None
        self.environment = dict(environment)

    def poll(self):
        if self.receipt is not None:
            return self.returncode
        reply = self.relay.request('poll')
        bridge.exact_fields(reply, ('op', 'serial', 'receipt'))
        receipt = reply['receipt']
        if receipt is None:
            return None
        bridge.exact_fields(receipt, ('pid', 'start', 'raw_status', 'returncode', 'exit_code_observed'))
        bridge.require(type(receipt['returncode']) is int and receipt['exit_code_observed'] is True,
                       'exact observed integer exit code required')
        expected = bridge.wait_receipt(self.pid, self.start, receipt['raw_status'])
        bridge.require(receipt == expected, 'service exit is not the exact real waitpid receipt')
        self.receipt, self.returncode = receipt, receipt['returncode']
        return self.returncode

    def wait(self, timeout=None):
        until = self.relay.deadline_ns if timeout is None else min(
            self.relay.deadline_ns, time.monotonic_ns() + int(timeout * 10 ** 9))
        while self.poll() is None:
            if time.monotonic_ns() >= until:
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(.02)
        return self.returncode

    def verify(self, start, command, environment):
        bridge.require(start == self.start and command == self.args and environment == self.environment,
                       'service proxy identity/command/environment differs')
        reply = self.relay.request('verify')
        bridge.exact_fields(reply, ('op', 'serial', 'service', 'verified'))
        bridge.require(reply['verified'] is True and reply['service'] == dict(pid=self.pid, start=self.start),
                       'root service authentication failed')

    def stop(self, start):
        bridge.require(start == self.start, 'service cleanup start identity differs')
        if self.poll() is None:
            # This is an unsuccessful application, not an ordinary exit request.
            # Root contains the exact task cgroup; clients cannot select a signal/PID.
            self.relay.request('abort')
            raise RuntimeError('live service at cleanup invalidates this replay')


def service_request(command, directory, *, service=bridge.SERVICE, signal_binding=None):
    bridge.require(type(command) is list and command and command[0] == str(service) and
                   all(type(value) is str for value in command), 'fixed service command required')
    pins = [command[index + 1] for index, value in enumerate(command[:-1]) if value == '--client-priority']
    bridge.require(len(pins) == 2 and command.count('--socket') == 1, 'two service pins and one socket required')
    clients = {}
    for role, pin, selection in zip(('low', 'high'), pins, ('configured', 'original')):
        match = re.fullmatch(r'([1-9][0-9]*):([1-9][0-9]*):0:' + selection, pin)
        bridge.require(match is not None, 'fixed role/priority/selection pin required')
        clients[role] = dict(pid=int(match[1]), start=int(match[2]))
    request = dict(directory=str(directory), socket=command[command.index('--socket') + 1], clients=clients)
    bridge.require(command == bridge.service_command(request, service=service, signal_binding=signal_binding),
                   'unexpected service option or ordering')
    return request


@contextmanager
def service_signal(common, signal_binding):
    """Decorate the targeted service command using an authenticated binding."""
    if signal_binding is None:
        yield
        return
    options = bridge.signal_arguments(signal_binding)
    original = common.service_command

    def selected(*args, **kwargs):
        command, pins = original(*args, **kwargs)
        bridge.require('--experimental-sharded-gemm-signal' not in command,
                       'signal option must come only from the runtime binding')
        bridge.require(command.count('--observe-block-completion') == 1,
                       'exact original block-observation selector required')
        if bridge.capture_specialization(signal_binding) == 'ordinary':
            command = [value for value in command if value != '--observe-block-completion']
        return [*command, *options], pins

    common.service_command = selected
    try:
        yield
    finally:
        common.service_command = original


@contextmanager
def service_launches(common, helpers, relay, directory, environment, *, service=bridge.SERVICE,
                     signal_binding=None, frontend_observer=None):
    """Modify one loaded helper module only, never the global subprocess API."""
    original_subprocess, original_verify, original_stop = common.subprocess, common.verify_process, helpers.stop_owned
    proxy = None
    ordered = signal_binding is not None
    ordinary = ordered and bridge.capture_specialization(signal_binding) == 'ordinary'
    original_ready = common.read_ready if ordered else None
    original_pending = common.check_pending if ordered else None
    original_release = common.release_on_publication if ordinary else None
    if ordered:
        bridge.require(not getattr(common, '_glimmer_ordered_replay', False), 'ordered replay already active')
    wrappers, deferred, ordering = [], None, {}

    def launch(command, **kwargs):
        nonlocal proxy, deferred
        bridge.require(not command or Path(command[0]).name != 'glimmer_execution_service' or
                       command[0] == str(service), 'unexpected service variant; no local fallback')
        if not command or command[0] != str(service):
            if ordered:
                bridge.require(command == [sys.executable, '-B', '-c', helpers.wrapper] and len(wrappers) < 2 and
                               kwargs.get('stdin') == subprocess.PIPE and kwargs.get('text') is True and
                               kwargs.get('env') == environment and kwargs.get('start_new_session') is True,
                               'unexpected fresh replay application wrapper')
            process = original_subprocess.Popen(command, **kwargs)
            if ordered:
                try:
                    wrappers.append((process, bridge.process_identity(process.pid)))
                    if len(wrappers) == 2:
                        deferred = DeferredLaunch(process.stdin)
                        process.stdin = deferred
                except BaseException:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)
                    if process.stdin:
                        process.stdin.close()
                    raise
            return process
        bridge.require(proxy is None and set(kwargs) == {'cwd', 'env', 'stdout', 'stderr', 'start_new_session'} and
                       Path(kwargs['cwd']) == Path(common.ROOT) and kwargs['env'] == environment and
                       kwargs['stderr'] == subprocess.STDOUT and kwargs['start_new_session'] is True and
                       Path(kwargs['stdout'].name) == directory / 'equal/server.log',
                       'unexpected service launch or reused service proxy')
        request = service_request(command, directory, service=service, signal_binding=signal_binding)
        reply = relay.request('launch', descriptors=(kwargs['stdout'].fileno(),), **request)
        bridge.exact_fields(reply, ('op', 'serial', 'service'))
        proxy = ServiceProcessProxy(relay, reply['service'], command, environment)
        return proxy

    def verify(helpers_arg, process, start, command=None, environment=None):
        if process is proxy:
            return proxy.verify(start, command, environment)
        bridge.require(not proc_mapped_paths(process.pid), 'BERT client inherited NCU code')
        actual = proc_environment(process.pid)
        bridge.require(not any(key.startswith((b'NV_', b'NVTX_', b'NVIDIA-PROCESS-', b'NVIDIA_PROCESS_'))
                               for key in actual), 'BERT client inherited NCU environment')
        result = original_verify(helpers_arg, process, start, command, environment)
        if frontend_observer is not None and command is not None and environment is not None:
            frontend_observer.observe(process, start, environment)
            original_verify(helpers_arg, process, start, command, environment)
        return result

    def stop(process, start):
        if process is proxy:
            # The frozen helper catches the application error before cleanup,
            # but aborting the service also terminates the controller. Preserve
            # that diagnostic before the relay contains the failed unit.
            try:
                caller = sys._getframe(1)
                if caller.f_code is common.run_control.__code__:
                    record = caller.f_locals.get('record')
                    if type(record) is dict and record.get('status') == 'failed':
                        reason = record.get('error')
                        if type(reason) is str:
                            print(json.dumps(dict(application_error=reason[:4096])), file=sys.stderr, flush=True)
            except (AttributeError, OSError, ValueError):
                pass  # Diagnostic publication must never prevent containment.
            return proxy.stop(start)
        return original_stop(process, start)

    def read_ready(helpers_arg, ready_directory, role, process, start, command, role_environment, launched_ns):
        result = original_ready(helpers_arg, ready_directory, role, process, start, command,
                                role_environment, launched_ns)
        if role != 'low' or result is None:
            return result
        bridge.require(len(wrappers) == 2 and deferred is not None and not ordering and
                       wrappers[0][0] is process and wrappers[0][1] == dict(pid=process.pid, start=start),
                       'unexpected low readiness or wrapper identities')
        high, high_identity = wrappers[1]
        bridge.require(high.poll() is None and bridge.process_identity(high.pid) == high_identity and
                       deferred.queued_ns is not None and deferred.released_ns is None,
                       'high wrapper no longer waits for its launch')
        original_verify(helpers_arg, high, high_identity['start'],
                        [sys.executable, '-B', '-c', helpers.wrapper], environment)
        bridge.require(proc_environment(high.pid) == {os.fsencode(k): os.fsencode(v)
                                                       for k, v in environment.items()} and
                       not proc_mapped_paths(high.pid), 'high wrapper environment changed')
        descriptors = Path(f'/proc/{high.pid}/fd')
        bridge.require(not any(os.readlink(path).startswith('socket:') for path in descriptors.iterdir()),
                       'high wrapper unexpectedly owns a socket')
        bridge.require(not Path(f'/proc/{high.pid}/task/{high.pid}/children').read_text().strip(),
                       'high wrapper unexpectedly owns a child')
        verified_ns = time.monotonic_ns()
        ordering.update(low=wrappers[0][1], high=high_identity, low_authenticated_ns=result['authenticated_ns'],
                        high_queued_ns=deferred.queued_ns, high_verified_ns=verified_ns,
                        message_bytes=len(deferred.message.encode()),
                        message_sha256=hashlib.sha256(deferred.message.encode()).hexdigest(),
                        no_children=True, no_sockets=True)
        try:
            deferred.release()
        finally:
            ordering.update(state=deferred.state, high_release_ns=deferred.released_ns)
        return result

    def check_pending(processes, monitor, launched, ready, service_process, allow_exits=False):
        original_pending(processes, monitor, launched, ready, service_process, allow_exits=allow_exits)
        if deferred is not None and deferred.queued_ns is not None and 'high' not in ready:
            stage_ns = deferred.released_ns or deferred.queued_ns
            bridge.require(time.monotonic_ns() - stage_ns < 100 * 10**9,
                           'bounded ordered replay initialization timed out')

    def logical_release(path, offset, pid, start, **kwargs):
        bridge.require(kwargs.get('selected_sequence') == 2842, 'fixed equal-mode release sequence required')
        with path.open(encoding='utf-8') as stream:
            prefix = stream.read(offset)
        bridge.require(len(prefix) == offset, 'logical release service-log prefix was truncated')
        session = ordinary_release_session(prefix, pid, start)
        original = common._arrival_record
        common._arrival_record = LogicalRelease(session, pid, start, 2842)
        try:
            return original_release(path, offset, pid, start, **kwargs)
        finally:
            common._arrival_record = original

    class SubprocessAdapter:
        Popen = staticmethod(launch)

        def __getattr__(self, name):
            return getattr(original_subprocess, name)

    common.subprocess, common.verify_process, helpers.stop_owned = SubprocessAdapter(), verify, stop
    if ordered:
        common._glimmer_ordered_replay = True
        common.read_ready, common.check_pending = read_ready, check_pending
    if ordinary:
        common.release_on_publication = logical_release
    try:
        yield ordering
    finally:
        common.subprocess, common.verify_process, helpers.stop_owned = original_subprocess, original_verify, original_stop
        if ordered:
            ordering.setdefault('state', deferred.state if deferred is not None else 'never_released')
            common.read_ready, common.check_pending = original_ready, original_pending
            del common._glimmer_ordered_replay
        if ordinary:
            common.release_on_publication = original_release
