"""Bounded, owned-process CPU observations without GPU counter access."""

from contextlib import contextmanager, ExitStack
import json
import os
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import bert_device_observer as observer
from run_rnn_v8_validation import save
from run_bert_device_observation import require

FREQUENCY = 500
INTERVAL_S = .05


def profile_environment(clean, directory, library):
    require(not clean.get('LD_PRELOAD') and not clean.get('LD_AUDIT')
            and not any(key.startswith('CPUPROFILE') for key in clean), 'clean service environment required')
    return dict(clean, LD_PRELOAD=str(library), CPUPROFILE=str(directory / 'service.cpu.prof'),
                CPUPROFILESIGNAL=str(int(signal.SIGUSR2)), CPUPROFILE_FREQUENCY=str(FREQUENCY))


def parse_stat(text):
    fields = text.rsplit(')', 1)[1].split()
    return dict(state=fields[0], user_ticks=int(fields[11]), system_ticks=int(fields[12]),
                start_ticks=int(fields[19]))


def optional_text(path):
    try:
        return dict(available=True, value=path.read_text().strip())
    except (PermissionError, FileNotFoundError) as error:
        return dict(available=False, reason=type(error).__name__)


def sample_process(proof):
    proof.check()
    root = Path(f'/proc/{proof.pid}')
    row = dict(pid=proof.pid, low_ns=time.monotonic_ns(), process=parse_stat((root / 'stat').read_text()),
               threads=[])
    tasks = sorted((root / 'task').iterdir(), key=lambda p: int(p.name))
    require(len(tasks) <= 128, 'thread observation bound exceeded')
    for task in tasks:
        thread = dict(tid=int(task.name))
        try:
            thread['stat'] = parse_stat((task / 'stat').read_text())
            thread['comm'] = optional_text(task / 'comm')
            thread['wchan'] = optional_text(task / 'wchan')
            thread['schedstat'] = optional_text(task / 'schedstat')
            status = optional_text(task / 'status')
            thread['status'] = (dict(available=True, value={line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                for line in status['value'].splitlines() if line.startswith(
                    ('State:', 'voluntary_ctxt_switches:', 'nonvoluntary_ctxt_switches:'))})
                if status['available'] else status)
            require(parse_stat((task / 'stat').read_text())['start_ticks'] == thread['stat']['start_ticks'],
                    'thread incarnation changed during sample')
        except FileNotFoundError:
            thread = dict(tid=int(task.name), retired_during_sample=True)
        row['threads'].append(thread)
    row['high_ns'] = time.monotonic_ns()
    proof.check()
    return row


def authenticate_profiler(proof, command, environment, library):
    proof.check()
    root = Path(f'/proc/{proof.pid}')
    actual = dict(item.split('=', 1) for item in (root / 'environ').read_text().split('\0') if item)
    argv = (root / 'cmdline').read_text().split('\0')[:-1]
    mappings = {line.split(maxsplit=5)[5] for line in (root / 'maps').read_text().splitlines()
                if len(line.split(maxsplit=5)) == 6}
    status = dict(line.split(':', 1) for line in (root / 'status').read_text().splitlines() if ':' in line)
    mask = 1 << (int(signal.SIGUSR2) - 1)
    require(argv == command and actual == environment
            and (root / 'exe').samefile(command[0]) and str(library) in mappings
            and os.getpgid(proof.pid) == proof.pid and os.getsid(proof.pid) == proof.pid
            and int(status['SigCgt'], 16) & mask and not int(status['SigBlk'], 16) & mask
            and not int(status['SigIgn'], 16) & mask, 'authenticated profiler toggle unavailable; no signal')
    proof.check()


def validate_toggles(rows, begin, end):
    require([row['action'] for row in rows] == ['start', 'stop'], 'one start/stop toggle pair required')
    require(all((row['pid'], row['start_ticks'], row['pidfd_bound'], row['signal']) ==
                (rows[0]['pid'], rows[0]['start_ticks'], True, int(signal.SIGUSR2)) for row in rows),
            'toggle identity changed')
    require(all(target <= row['low_ns'] <= row['high_ns'] < target + 100_000_000
                for row, target in zip(rows, (begin, end))), 'CPU toggle missed its bounded window')


class HostWindow:
    def __init__(self, directory, cpu, guard):
        self.directory, self.cpu, self.guard = directory, cpu, guard
        self.stop = threading.Event()
        self.thread = self.error = None
        self.receipt = dict(status='not_started', snapshots=[], toggles=[],
                            interval_ms=50, ticks_per_second=os.sysconf('SC_CLK_TCK'))

    def start(self, schedule, owners):
        require(self.thread is None, 'one host window required')
        begin, end = schedule['begin_ns'] + 5_000_000_000, schedule['begin_ns'] + 10_000_000_000
        self.receipt.update(begin_ns=begin, end_ns=end,
            owners={role: row['identity'] for role, row in owners.items()})

        def run():
            try:
                with ExitStack() as stack:
                    proofs = {role: stack.enter_context(observer.PinnedProcess(row['identity']))
                              for role, row in owners.items()}
                    require(not self.stop.wait(max(0, (begin - time.monotonic_ns()) / 1e9)),
                            'host window cancelled before sampling')
                    if 'service' in owners:
                        row, proof = owners['service'], proofs['service']
                        authenticate_profiler(proof, row['command'], row['environment'], self.cpu.LIBRARY)
                        require(not list(self.directory.glob('service.cpu.prof*')), 'fresh CPU profile required')
                        self.toggle(proof, 'start')
                    require(time.monotonic_ns() - begin < 100_000_000, 'host window start missed')
                    while True:
                        self.receipt['snapshots'].append(dict(
                            monotonic_ns=time.monotonic_ns(),
                            processes={role: sample_process(proof) for role, proof in proofs.items()}))
                        require(len(self.receipt['snapshots']) <= 120, 'host snapshot count bound exceeded')
                        remaining = (end - time.monotonic_ns()) / 1e9
                        if remaining <= 0:
                            break
                        require(not self.stop.wait(min(INTERVAL_S, remaining)), 'host window cancelled')
                    require(time.monotonic_ns() - end < 100_000_000, 'host window stop missed')
                    if 'service' in owners:
                        row, proof = owners['service'], proofs['service']
                        authenticate_profiler(proof, row['command'], row['environment'], self.cpu.LIBRARY)
                        self.toggle(proof, 'stop')
                        validate_toggles(self.receipt['toggles'], begin, end)
                        deadline = time.monotonic() + 2
                        while True:
                            proof.check()
                            try:
                                self.receipt['profile'] = self.cpu.parse_profile(self.directory / 'service.cpu.prof.0')
                                break
                            except (ValueError, FileNotFoundError):
                                require(time.monotonic() < deadline, 'CPU stop did not produce a complete profile')
                                time.sleep(.01)
                        self.receipt['profile']['scope'] = 'Service CPU samples between authenticated window start/stop toggles; not blocked time'
                        self.receipt['profile_complete_observed_ns'] = time.monotonic_ns()
                        require(sorted(p.name for p in self.directory.glob('service.cpu.prof*')) == ['service.cpu.prof.0'],
                                'unexpected additional CPU profile')
                    self.receipt['status'] = 'complete_requires_review'
            except BaseException as error:
                self.error = error
                self.receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            finally:
                self.receipt['observer_finished_ns'] = time.monotonic_ns()

        self.thread = threading.Thread(target=run, name='owned-host-window', daemon=True)
        self.thread.start()

    def toggle(self, proof, action):
        proof.check()
        row = dict(action=action, pid=proof.pid, start_ticks=proof.expected['start_ticks'],
                   signal=int(signal.SIGUSR2), low_ns=time.monotonic_ns(), pidfd_bound=True)
        signal.pidfd_send_signal(proof.fd, signal.SIGUSR2)
        row['high_ns'] = time.monotonic_ns()
        self.receipt['toggles'].append(row)

    def check(self):
        if self.error is not None:
            raise ValueError('host observer failed: ' + str(self.error)) from self.error

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            require(not self.thread.is_alive(), 'host observer did not retire')
        require(len(json.dumps(self.receipt, indent=2).encode()) <= 16 * 1024 ** 2,
                'host observer JSON exceeds16MiB')
        save(self.directory.parent / (self.directory.name + '-host-window.json'), self.receipt)
        self.check()
        require(self.receipt['status'] == 'complete_requires_review', 'host window incomplete')


@contextmanager
def host_hooks(common, h, sustained, guard, mode, clean, window):
    original = dict(subprocess=sustained.subprocess, owner=sustained.owner_command,
                    verify=common.verify_process, pending=common.check_pending, schedule=sustained.schedule)
    service_command, owners = [], {}
    profiled = profile_environment(clean, window.directory, window.cpu.LIBRARY)

    def owner(*args, **kwargs):
        argv, pins = original['owner'](*args, **kwargs)
        require(mode == 'equal' and not service_command, 'one equal service required')
        service_command.extend(argv)
        return argv, pins

    def popen(argv, **kwargs):
        require(kwargs.get('env') == clean and kwargs.get('cwd') == sustained.ROOT
                and kwargs.get('start_new_session') is True, 'unchanged child launch required')
        if service_command and argv == service_command:
            kwargs = dict(kwargs, env=profiled)
        else:
            require(argv == [sustained.sys.executable, '-B', '-c', h.wrapper], 'unexpected application launch')
        return original['subprocess'].Popen(argv, **kwargs)

    def verify(helper, process, start, command=None, environment=None):
        is_service = bool(service_command) and command == service_command
        if is_service:
            require(environment == clean, 'original service environment changed')
            environment = profiled
        original['verify'](helper, process, start, command, environment)
        guard.observe(process, start)
        if command is not None and environment is not None:
            role = 'service' if is_service else command[command.index('--role') + 1]
            require(role in ('training', 'inference', 'service'), 'unknown participant role')
            identity = observer.process_record(process.pid)
            require(identity['cgroup'] == guard.group and identity['start_ticks'] == start,
                    'participant escaped task unit')
            owners[role] = dict(identity=identity, command=list(command), environment=dict(environment))

    def schedule(*args, **kwargs):
        value = original['schedule'](*args, **kwargs)
        require(set(owners) == {'training', 'inference'} | ({'service'} if mode == 'equal' else set()),
                'complete host participant inventory required')
        window.start(value, owners)
        return value

    def pending(*args, **kwargs):
        guard.check()
        window.check()
        return original['pending'](*args, **kwargs)

    sustained.subprocess = SimpleNamespace(Popen=popen, PIPE=original['subprocess'].PIPE,
                                           STDOUT=original['subprocess'].STDOUT)
    sustained.owner_command, common.verify_process = owner, verify
    common.check_pending, sustained.schedule = pending, schedule
    try:
        yield owners
    finally:
        sustained.subprocess, sustained.owner_command = original['subprocess'], original['owner']
        common.verify_process, common.check_pending = original['verify'], original['pending']
        sustained.schedule = original['schedule']
