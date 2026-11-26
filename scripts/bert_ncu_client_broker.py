"""Task-local client launcher outside NCU ancestry; no CUDA calls.

One broker serves sequential complete application replay sessions. Only the
fixed client wrapper can be launched. A connected controller receives real
PID/start identities, stdin descriptors and the broker's actual waitpid status.
The containing task unit remains the authority for aggregate resource limits.
"""

import array
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import select
import signal
import socket
import stat
import struct
import subprocess
import time

from bert_sgemm_ncu_process import proc_mapped_paths
from run_rnn_v8_validation import cgroup, identity


MESSAGE_MAX = 65536
ROLES = ('low', 'high')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def clean_client_environment(environment):
    result = dict(environment)
    result.pop('NVTX_INJECTION64_PATH', None)
    return result


def send_message(connection, value, descriptors=()):
    payload = json.dumps(value, allow_nan=False).encode()
    require(len(payload) <= MESSAGE_MAX, 'broker message exceeds bound')
    ancillary = ([(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', descriptors))]
                 if descriptors else [])
    require(connection.sendmsg([payload], ancillary) == len(payload), 'short broker packet')


def receive_message(connection):
    descriptors = []
    try:
        data, ancillary, flags, _ = connection.recvmsg(
            MESSAGE_MAX, socket.CMSG_SPACE(4 * array.array('i').itemsize), socket.MSG_CMSG_CLOEXEC)
        for level, kind, payload in ancillary:
            require(level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS,
                    'unexpected broker ancillary message')
            values = array.array('i')
            values.frombytes(payload)
            descriptors.extend(values)
        require(not flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC), 'truncated broker message')
        if not data:
            raise EOFError('broker connection closed before completion')
        value = json.loads(data)
        require(type(value) is dict, 'broker object required')
        return value, descriptors
    except BaseException:
        for descriptor in descriptors:
            os.close(descriptor)
        raise


def peer_identity(connection, expected_uid, group):
    pid, uid, _ = struct.unpack('3i', connection.getsockopt(
        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize('3i')))
    require(uid == expected_uid and uid > 0 and cgroup(pid) == group,
            'broker peer must be the same unprivileged task owner')
    return dict(pid=pid, start=identity(pid))


class ReapedChild:
    """Exclusive waitpid ownership; ECHILD is an error, never a zero exit."""

    def __init__(self, process):
        self.process = process
        self.pid = process.pid
        self.start = identity(self.pid)
        self.returncode = None
        require(os.getpgid(self.pid) == self.pid and os.getsid(self.pid) == self.pid,
                'broker child needs its own session')
        self.pidfd = os.pidfd_open(self.pid)

    def poll(self):
        if self.returncode is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            require(pid in (0, self.pid), 'unexpected waitpid result')
            if pid:
                self.returncode = os.waitstatus_to_exitcode(status)
                self.process.returncode = self.returncode
        return self.returncode

    def close(self):
        try:
            if self.process.stdin:
                self.process.stdin.close()
            for signum in (signal.SIGTERM, signal.SIGKILL):
                if self.poll() is not None:
                    break
                require(identity(self.pid) == self.start, 'child identity changed before cleanup')
                try:
                    signal.pidfd_send_signal(self.pidfd, signum)
                except ProcessLookupError:
                    # The following waitpid must still report its real status.
                    pass
                deadline = time.monotonic() + 2
                while self.poll() is None and time.monotonic() < deadline:
                    time.sleep(.01)
            require(self.poll() is not None, 'broker child did not retire')
        finally:
            os.close(self.pidfd)

    def receipt(self):
        return dict(pid=self.pid, start=self.start, returncode=self.poll(),
                    exit_code_observed=self.returncode is not None)


class ClientBroker:
    """Single-threaded server; immutable command, environment and task scope."""

    def __init__(self, outer, wrapper_command, environment, cwd, peer_command, deadline):
        self.outer = Path(outer).resolve(strict=True)
        self.wrapper_command, self.environment = list(wrapper_command), dict(environment)
        self.cwd, self.peer_command = Path(cwd).resolve(strict=True), list(peer_command)
        self.deadline, self.group = deadline, cgroup(os.getpid())
        self.sessions, self.used_directories = [], set()

    def validate_peer(self, connection):
        peer = peer_identity(connection, os.getuid(), self.group)
        argv = Path(f"/proc/{peer['pid']}/cmdline").read_bytes().split(b'\0')[:-1]
        require(argv == [os.fsencode(value) for value in self.peer_command] and
                Path(f"/proc/{peer['pid']}/exe").resolve(strict=True) ==
                Path(self.wrapper_command[0]).resolve(strict=True), 'unexpected broker controller command')
        return peer

    def begin(self, request):
        require(set(request) == {'op', 'directory'} and request['op'] == 'begin' and
                type(request['directory']) is str, 'invalid replay registration')
        directory = Path(request['directory'])
        require(directory == directory.resolve(strict=True) and directory.is_dir() and
                directory.parent == self.outer / 'passes' and directory.name.startswith('pass-') and
                directory not in self.used_directories and directory.stat().st_uid == os.getuid(),
                'replay requires a fresh owned pass directory')
        self.used_directories.add(directory)
        return directory

    def launch(self, request, descriptors, directory, children):
        require(set(request) == {'op', 'role'} and request['role'] in ROLES and
                request['role'] not in children and len(descriptors) == 1,
                'one new client role and one log descriptor required')
        role, descriptor = request['role'], descriptors[0]
        log = directory / 'equal' / (role + '.log')
        actual, expected = os.fstat(descriptor), log.lstat()
        require(log.parent == log.parent.resolve(strict=True) and stat.S_ISREG(actual.st_mode) and
                stat.S_ISREG(expected.st_mode) and actual.st_uid == os.getuid() and
                actual.st_nlink == 1 and actual.st_size == 0 and
                (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino) and
                fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE == os.O_WRONLY,
                'client log descriptor must identify the fresh owned role log')
        process = subprocess.Popen(self.wrapper_command, cwd=self.cwd, env=self.environment,
                                   stdin=subprocess.PIPE, stdout=descriptor,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        # Register immediately so partial startup is always cleaned by session().
        try:
            child = ReapedChild(process)
        except BaseException:
            process.kill()
            process.wait(timeout=3)
            if process.stdin:
                process.stdin.close()
            raise
        children[role] = child
        require(cgroup(child.pid) == self.group and not proc_mapped_paths(child.pid),
                'client escaped task containment or inherited NCU injection')
        return child

    def session(self, connection, stop_fd):
        children, record = {}, dict(status='started')
        try:
            record['controller'] = self.validate_peer(connection)
            request, fds = receive_message(connection)
            try:
                require(not fds, 'registration cannot transfer descriptors')
                directory = self.begin(request)
            finally:
                for fd in fds:
                    os.close(fd)
            record['directory'] = str(directory)
            send_message(connection, dict(ok=True))
            while True:
                require(time.monotonic() < self.deadline, 'broker task deadline exhausted')
                readable, _, _ = select.select([connection, stop_fd], [], [], .2)
                require(stop_fd not in readable, 'supervisor stopped during replay')
                if connection not in readable:
                    continue
                require(identity(record['controller']['pid']) == record['controller']['start'],
                        'broker controller identity changed')
                request, fds = receive_message(connection)
                try:
                    operation = request.get('op')
                    if operation == 'spawn':
                        child = self.launch(request, fds, directory, children)
                        send_message(connection, dict(ok=True, **child.receipt()),
                                     [child.process.stdin.fileno()])
                    elif operation == 'poll':
                        require(not fds and set(request) == {'op', 'role'} and
                                request['role'] in children, 'unknown broker child')
                        send_message(connection, dict(ok=True, **children[request['role']].receipt()))
                    elif operation == 'finish':
                        require(not fds and set(request) == {'op'} and set(children) == set(ROLES),
                                'complete replay requires both fresh clients')
                        receipts = {role: child.receipt() for role, child in children.items()}
                        require(all(value['exit_code_observed'] for value in receipts.values()),
                                'cannot finish a replay with live clients')
                        record.update(status='complete', clients=receipts)
                        send_message(connection, dict(ok=True, clients=receipts))
                        break
                    else:
                        raise ValueError('unknown broker operation')
                finally:
                    for fd in fds:
                        os.close(fd)
        except BaseException as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            errors = []
            for role, child in children.items():
                try:
                    child.close()
                    record.setdefault('retired_clients', {})[role] = child.receipt()
                except BaseException as error:
                    errors.append(f'{role}: {type(error).__name__}: {error}')
            if errors:
                record.update(status='failed', cleanup_errors=errors)
            self.sessions.append(record)
            require(not errors, 'broker child cleanup failed')

    def serve(self, stop_fd, ready):
        path = self.outer / 'broker.sock'
        require(len(os.fsencode(path)) < 108, 'broker socket path exceeds Linux bound')
        require(not proc_mapped_paths(os.getpid()), 'broker inherited NCU injection')
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as server:
            server.bind(str(path))
            metadata = path.stat()
            try:
                os.chmod(path, 0o600)
                server.listen(1)
                ready(dict(pid=os.getpid(), start=identity(os.getpid()), group=self.group,
                           socket_device=metadata.st_dev, socket_inode=metadata.st_ino))
                while True:
                    require(time.monotonic() < self.deadline, 'broker task deadline exhausted')
                    readable, _, _ = select.select([server, stop_fd], [], [], .2)
                    if stop_fd in readable:
                        require(os.read(stop_fd, 1) == b'', 'invalid broker stop request')
                        return
                    if server in readable:
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(3)
                            self.session(connection, stop_fd)
            finally:
                current = path.lstat()
                require((current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino),
                        'broker socket identity changed before removal')
                path.unlink()


class BrokerProcess:
    """Popen subset for historical harness; all exit codes come from broker."""

    def __init__(self, broker, role, receipt, descriptor):
        self.broker, self.role = broker, role
        self.pid, self.start = receipt['pid'], receipt['start']
        self.returncode = receipt['returncode']
        self.stdin = os.fdopen(descriptor, 'w')

    def poll(self):
        if self.returncode is None:
            receipt, fds = self.broker.request(dict(op='poll', role=self.role))
            self.broker.no_descriptors(fds)
            require(receipt['pid'] == self.pid and receipt['start'] == self.start,
                    'broker returned a different child')
            code = receipt['returncode']
            require(code is None or type(code) is int, 'invalid broker return code')
            require(receipt['exit_code_observed'] == (code is not None), 'ambiguous broker status')
            self.returncode = code
        return self.returncode

    def wait(self, timeout=None):
        deadline = float('inf') if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('broker client ' + self.role, timeout)
            time.sleep(.01)
        return self.returncode


class BrokerConnection:
    def __init__(self, outer, directory, ready):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.processes = {}
        try:
            self.connection.settimeout(5)
            path = Path(outer) / 'broker.sock'
            metadata = path.lstat()
            require(stat.S_ISSOCK(metadata.st_mode) and metadata.st_uid == os.getuid() and
                    metadata.st_mode & 0o777 == 0o600 and
                    (metadata.st_dev, metadata.st_ino) ==
                    (ready['socket_device'], ready['socket_inode']), 'broker socket identity differs')
            self.connection.connect(str(path))
            peer = peer_identity(self.connection, os.getuid(), cgroup(os.getpid()))
            require(peer == {key: ready[key] for key in ('pid', 'start')} and
                    ready['group'] == cgroup(os.getpid()) and not proc_mapped_paths(peer['pid']),
                    'broker process identity or injection differs')
            _, fds = self.request(dict(op='begin', directory=str(directory)))
            self.no_descriptors(fds)
        except BaseException:
            self.connection.close()
            raise

    @staticmethod
    def no_descriptors(descriptors):
        for descriptor in descriptors:
            os.close(descriptor)
        require(not descriptors, 'unexpected broker descriptors')

    def request(self, value, descriptors=()):
        send_message(self.connection, value, descriptors)
        reply, fds = receive_message(self.connection)
        try:
            require(reply.get('ok') is True, 'broker request did not succeed')
            return reply, fds
        except BaseException:
            for descriptor in fds:
                os.close(descriptor)
            raise

    def spawn(self, role, log):
        reply, fds = self.request(dict(op='spawn', role=role), [log.fileno()])
        try:
            require(role not in self.processes and len(fds) == 1 and
                    type(reply['pid']) is int and type(reply['start']) is int and
                    reply['pid'] > 0 and reply['start'] > 0 and
                    reply['returncode'] is None and reply['exit_code_observed'] is False and
                    identity(reply['pid']) == reply['start'] and
                    cgroup(reply['pid']) == cgroup(os.getpid()) and
                    os.getpgid(reply['pid']) == reply['pid'] and os.getsid(reply['pid']) == reply['pid'],
                    'invalid broker child identity')
            process = BrokerProcess(self, role, reply, fds[0])
        except BaseException:
            for descriptor in fds:
                os.close(descriptor)
            raise
        self.processes[role] = process
        return process

    def finish(self):
        require(set(self.processes) == set(ROLES) and
                all(process.poll() is not None for process in self.processes.values()),
                'broker clients must retire before replay finishes')
        reply, fds = self.request(dict(op='finish'))
        self.no_descriptors(fds)
        expected = {role: dict(pid=p.pid, start=p.start, returncode=p.returncode,
                               exit_code_observed=True) for role, p in self.processes.items()}
        require(reply['clients'] == expected, 'final broker statuses differ')
        return expected

    def close(self):
        self.connection.close()
        for process in self.processes.values():
            process.stdin.close()


@contextmanager
def broker_client_launches(common, helpers, broker, environment):
    """Replace only one loaded helper's module attribute; never global Popen."""
    original_subprocess, original_environment = common.subprocess, common.owned_environment
    wrapper = [str(helpers.training.PYTHON), '-B', '-c', helpers.wrapper]

    def launch(command, **kwargs):
        if command != wrapper:
            require(kwargs.get('stdin') is None, 'unexpected non-client stdin launch')
            return original_subprocess.Popen(command, **kwargs)
        require(set(kwargs) == {'cwd', 'stdin', 'stdout', 'stderr', 'text', 'env', 'start_new_session'} and
                Path(kwargs['cwd']).resolve(strict=True) == Path(common.ROOT).resolve(strict=True) and
                kwargs['stdin'] == subprocess.PIPE and kwargs['stderr'] == subprocess.STDOUT and
                kwargs['text'] is True and kwargs['start_new_session'] is True and
                clean_client_environment(kwargs['env']) == environment, 'unexpected client launch recipe')
        role = Path(kwargs['stdout'].name).stem
        require(role in ROLES, 'unexpected client log role')
        return broker.spawn(role, kwargs['stdout'])

    def owned_environment(*args, **kwargs):
        return clean_client_environment(original_environment(*args, **kwargs))

    class SubprocessAdapter:
        Popen = staticmethod(launch)

        def __getattr__(self, name):
            return getattr(original_subprocess, name)

    common.subprocess = SubprocessAdapter()
    common.owned_environment = owned_environment
    try:
        yield
    finally:
        common.subprocess, common.owned_environment = original_subprocess, original_environment
