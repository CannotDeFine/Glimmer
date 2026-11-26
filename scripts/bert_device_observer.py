"""Contain a private BERT counter sampler; no process or SDK on import.

The source-managed controller owns supervision. The GPU query allowance never
grants application ownership, ordinary timing or hardware qualification.
"""

from contextlib import contextmanager
import importlib
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import select
import signal
import stat
import sys
import threading
import time
from types import SimpleNamespace
import uuid

from aggregate_sm_audit import (FIELDS, PERIOD_US, is_no_data, reject_constant, require,
                               unique_object, validate_no_data)
from artifact_paths import PROJECT_ROOT
from run_bert_candidate_validation import FILE_MAX
from run_rnn_v8_validation import cgroup, digest, memory_receipt

COMPUTE = PROJECT_ROOT / 'build/bert-application-validation-1/compute-observation-1'
TOOLS = COMPUTE / 'tools/dcgm-4.0.0'
LIBRARIES = TOOLS / 'root/usr/lib/x86_64-linux-gnu'
BINDINGS = TOOLS / 'root/usr/share/datacenter-gpu-manager-4/bindings/python3'
OUTPUT_ROOT = PROJECT_ROOT / 'artifacts/profiles/bert-current-device'
GPU_UUID = 'GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2'
MODES = ('exclusive', 'native', 'equal', 'priority')
JOURNAL_MAX = 16 * 1024 ** 2
JSON_MAX = 2 * 1024 ** 2
WORK_SECONDS = 700
IDENTITY_KEYS = ('pid', 'start_ticks', 'uid', 'cgroup')


def process_record(pid):
    require(type(pid) is int and pid > 1, 'positive non-init process identity required')
    root = Path(f'/proc/{pid}')
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    uid_line = next(line for line in (root / 'status').read_text().splitlines()
                    if line.startswith('Uid:'))
    uids = list(map(int, uid_line.split()[1:]))
    require(len(uids) == 4 and len(set(uids)) == 1, 'uniform process UIDs required')
    return dict(pid=pid, start_ticks=int(fields[19]), uid=uids[0], ppid=int(fields[1]),
                state=fields[0], cgroup=cgroup(pid))


def exact_identity(expected, actual):
    return (type(expected) is dict and type(actual) is dict and actual.get('state') in ('R', 'S', 'D', 'T', 't', 'I', 'W', 'P')
            and all(type(expected.get(key)) is int and type(actual.get(key)) is int and expected[key] == actual[key]
                    for key in IDENTITY_KEYS[:3])
            and expected['pid'] > 1 and expected['start_ticks'] > 0 and expected['uid'] >= 0
            and type(expected.get('cgroup')) is str and expected['cgroup'].startswith('/')
            and expected['cgroup'] == actual.get('cgroup'))


class PinnedProcess:
    """Read-only pidfd with identity rechecks. It never sends a signal."""

    def __init__(self, expected):
        self.expected, self.pid, self.fd = dict(expected), expected['pid'], None
        require(exact_identity(expected, process_record(self.pid)), 'initial process identity differs')
        try:
            self.fd = os.pidfd_open(self.pid, 0)
            self.check()
        except BaseException:
            self.close()
            raise

    def poll(self):
        require(self.fd is not None, 'closed process proof')
        poller = select.poll()
        poller.register(self.fd, select.POLLIN)
        return 0 if poller.poll(0) else None

    def check(self):
        require(self.poll() is None and exact_identity(self.expected, process_record(self.pid)),
                'process exited, changed UID/cgroup or reused its PID')

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class BoundedWriter:
    """Reject before writing any byte beyond the journal bound."""

    def __init__(self, stream, limit=JOURNAL_MAX):
        self.stream, self.limit, self.written = stream, limit, 0

    def write(self, value):
        size = len(value.encode('utf-8'))
        require(self.written + size <= self.limit, '16MiB sampler journal write bound exceeded')
        result = self.stream.write(value)
        require(result == len(value), 'short sampler journal write')
        self.written += size
        return result

    def flush(self):
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()


class PrivateDirectory:
    """Anchored exclusive file operations in a canonical user-owned0700 folder.

    This is task containment, not a security sandbox against the authorized
    account replacing its own project code. No writes follow file symlinks.
    """

    NAMES = {'observer-plan.json', 'identity.json', 'armed.json', 'ready.json',
             'observer.json', 'samples.jsonl', 'stop'}

    def __init__(self, path, owner_uid):
        self.path, self.owner_uid, self.fd = Path(path), owner_uid, None
        require(type(owner_uid) is int and owner_uid > 0, 'nonroot output owner required')
        require(self.path.is_absolute() and self.path.resolve(strict=True) == self.path,
                'canonical observer directory required')
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            initial = os.fstat(self.fd)
            self.directory_identity = initial.st_dev, initial.st_ino
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        current = self.path.lstat()
        require(self.path.resolve(strict=True) == self.path and stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == self.directory_identity
                and current.st_uid == self.owner_uid and stat.S_IMODE(current.st_mode) == 0o700,
                'observer directory changed, aliased or is not private')
        require((os.fstat(self.fd).st_dev, os.fstat(self.fd).st_ino) == self.directory_identity,
                'anchored directory identity differs')

    def _name(self, name):
        require(name in self.NAMES, 'undeclared sampler file')
        self.check()

    def read(self, name, uid):
        self._name(name)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=self.fd)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == uid
                    and 0 < info.st_size <= JSON_MAX, 'bounded single-link owned JSON required')
            content = stream.read(JSON_MAX + 1)
            require(len(content) <= JSON_MAX, 'JSON grew beyond its bound')
        return json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)

    def exists(self, name):
        self._name(name)
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            return True
        except FileNotFoundError:
            return False

    def file_identity(self, name, uid, limit):
        self._name(name)
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=self.fd)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_uid == uid and info.st_nlink == 1
                    and 0 < info.st_size <= limit, 'bounded single-link owned sample file required')
            checksum, size = hashlib.sha256(), 0
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                require(size <= limit, 'sample file grew beyond bound')
                checksum.update(chunk)
            require(size == info.st_size, 'sample file size changed during identity read')
        return dict(bytes=size, sha256=checksum.hexdigest(), uid=uid)

    def stopped(self):
        if not self.exists('stop'):
            return False
        info = os.stat('stop', dir_fd=self.fd, follow_symlinks=False)
        require(stat.S_ISREG(info.st_mode) and info.st_uid == self.owner_uid
                and info.st_nlink == 1 and info.st_size == 0, 'invalid stop request identity/type')
        return True

    def publish(self, name, value):
        self._name(name)
        content = (json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n').encode()
        require(len(content) <= JSON_MAX, 'sampler receipt bound exceeded')
        temporary = '.' + name + '.' + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o644, dir_fd=self.fd)
        try:
            with os.fdopen(fd, 'wb') as stream:
                os.fchmod(stream.fileno(), 0o644)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
        finally:
            os.unlink(temporary, dir_fd=self.fd)

    def __truediv__(self, name):
        self._name(name)
        return SamplerFile(self, name)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class SamplerFile:
    """Only the three operations used by the retained sample loop."""

    def __init__(self, directory, name):
        self.directory, self.name = directory, name

    def open(self, mode):
        require(self.name == 'samples.jsonl' and mode == 'x', 'exclusive journal operation required')
        self.directory.check()
        fd = os.open(self.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o644, dir_fd=self.directory.fd)
        try:
            os.fchmod(fd, 0o644)
            stream = os.fdopen(fd, 'w', encoding='utf-8', newline='\n')
        except BaseException:
            os.close(fd)
            raise
        return BoundedWriter(stream)

    def exists(self):
        require(self.name == 'stop', 'only stop is probed by the sample loop')
        return self.directory.stopped()


def restrict_sampler_file_limit():
    """Restore the task's process-local bound after sudo/PAM resets it.

    Called only after authenticating the root sampler's controller and cgroup.
    Never increase a stricter inherited limit or alter system policy.
    """
    limits = resource.getrlimit(resource.RLIMIT_FSIZE)
    require(all(value == resource.RLIM_INFINITY or value >= FILE_MAX for value in limits),
            'sampler file limit is stricter than the fixed recipe; never raise it')
    if limits != (FILE_MAX, FILE_MAX):
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_MAX, FILE_MAX))
    require(resource.getrlimit(resource.RLIMIT_FSIZE) == (FILE_MAX, FILE_MAX),
            'sampler file limit restoration failed')


def capped_memory(group):
    require(resource.getrlimit(resource.RLIMIT_FSIZE) == (FILE_MAX, FILE_MAX),
            'actual inherited1GiB file limit required before SDK entry')
    memory = memory_receipt(group)
    events = dict(line.split() for line in memory['memory.events'].splitlines())
    require(events and all(int(value) == 0 for value in events.values()),
            'task memory event invalidates the observation')
    return memory


def verify_seals(seals):
    require(type(seals) is dict and 0 < len(seals) <= 1024, 'bounded nonempty observer seals required')
    for name, value in seals.items():
        require(type(name) is str and Path(name).is_absolute()
                and type(value) is str and len(value) == 64 and digest(name) == value,
                'observer/tool identity changed: ' + str(name))


def validate_sampler_plan(plan, path):
    require(type(plan) is dict and type(plan.get('version')) is int and plan['version'] == 1
            and plan.get('mode') in MODES,
            'explicit sampler plan/version/mode required')
    require(path.parent.name == plan['mode'] and path.name == 'observer'
            and path.parent.parent.parent == OUTPUT_ROOT
            and path.parent.parent.name.startswith('attempt-'), 'sampler outside its dedicated attempt')
    owner = plan.get('owner', {})
    require(exact_identity(owner, process_record(owner.get('pid'))) and owner['uid'] > 0,
            'live exact nonroot controller required')
    expected = f"/glimmer-bert-{path.parent.parent.name}-{plan['mode']}.service"
    require(owner['cgroup'].endswith(expected), 'controller is not in the unique capped task unit')
    require(plan.get('gpu_uuid') == GPU_UUID and plan.get('fields') == list(FIELDS)
            and type(plan.get('period_us')) is int and plan['period_us'] == PERIOD_US
            and type(plan.get('hard_timeout_s')) is int and plan['hard_timeout_s'] == WORK_SECONDS,
            'frozen device/fields/sampling bounds required')
    require(type(plan.get('nonce')) is str and len(plan['nonce']) == 32
            and all(char in '0123456789abcdef' for char in plan['nonce']), 'fresh handshake nonce required')
    require(os.getuid() == os.geteuid() == 0, 'sampler must have actual root identity')
    own = process_record(os.getpid())
    require(own['uid'] == 0 and own['cgroup'] == owner['cgroup'], 'root sampler escaped task cgroup')
    restrict_sampler_file_limit()
    capped_memory(own['cgroup'])
    verify_seals(plan.get('observer_identities'))
    return own


def wait_for_arm(directory, plan, own, controller, *, clock=time.monotonic, sleep=time.sleep):
    """Publish real sampler identity; enter no SDK until controller holds pidfd."""
    require(not directory.stopped(), 'stop requested before sampler startup')
    directory.publish('identity.json', dict(sampler=own, owner=plan['owner'], nonce=plan['nonce'],
                                            sdk_initialized=False))
    deadline = clock() + 10
    while not directory.exists('armed.json'):
        controller.check()
        directory.check()
        require(not directory.stopped(), 'sampler stopped before arm')
        require(clock() < deadline, 'controller did not arm exact sampler within10s')
        sleep(.02)
    arm = directory.read('armed.json', plan['owner']['uid'])
    require(arm == dict(sampler=own, owner=plan['owner'], nonce=plan['nonce']),
            'arm does not bind the actual sampler/controller identities')
    controller.check()
    require(exact_identity(own, process_record(os.getpid())), 'sampler identity changed before SDK')
    capped_memory(own['cgroup'])


def arm_sampler(directory, plan, wrapper_pid, guard):
    """Controller half: distinguish actual root PID from sudo/timeout wrappers.

    Caller retains the returned pidfd until normal sampler exit/retirement is
    proved. This grants only an observer allowance, never GPU-owner authority.
    """
    bootstrap = directory.read('identity.json', 0)
    own = bootstrap.get('sampler', {})
    require(bootstrap == dict(sampler=own, owner=plan['owner'], nonce=plan['nonce'], sdk_initialized=False)
            and own.get('uid') == 0 and own.get('cgroup') == plan['owner']['cgroup']
            and own.get('pid') != wrapper_pid, 'actual pre-SDK root sampler identity required')
    pid, seen = own.get('pid'), set()
    while pid != wrapper_pid:
        require(type(pid) is int and pid > 1 and pid not in seen and len(seen) < 16,
                'actual sampler is not a descendant of the owned wrapper')
        seen.add(pid)
        pid = process_record(pid)['ppid']
    proof = PinnedProcess(own)
    try:
        guard.observe(proof, own['start_ticks'])
        directory.publish('armed.json', dict(sampler=own, owner=plan['owner'], nonce=plan['nonce']))
        return proof
    except BaseException:
        proof.close()
        raise


@contextmanager
def scoped_helpers(common, helpers, guard, sampler):
    """One application control, one monitor thread; restore even after failure.

    Only identity bookkeeping is shared with the monitor (under its own lock).
    ResourceGuard remains on the collector thread. Original GPU-owner sets,
    process-group checks and mandatory app/service observations stay unchanged.
    """
    original = helpers.query, helpers.identity, helpers.training, common.verify_process, common.check_pending
    query, identify, training, verify, pending = original
    observed, lock = {}, threading.Lock()

    def checked_identity(pid):
        start = identify(pid)
        require(cgroup(pid) == guard.group, 'application/service escaped the capped unit before release')
        with lock:
            require(pid not in observed or observed[pid] == start, 'application PID reused')
            require(pid in observed or len(observed) < 8, 'bounded exact participant inventory exceeded')
            observed[pid] = start
        return start

    def checked_query(kind, fields):
        raw = query(kind, fields)
        if kind != 'compute-apps':
            return raw
        require(fields == 'gpu_uuid,pid,process_name', 'explicit GPU ownership query required')
        sampler.check()
        kept = []
        for line in raw.splitlines():
            parts = [part.strip() for part in line.split(',', 2)]
            require(len(parts) == 3, 'unparseable GPU process identity')
            if int(parts[1]) == sampler.pid:
                require(parts[0] == GPU_UUID and sampler.expected['uid'] == 0
                        and sampler.expected['cgroup'] == guard.group, 'sampler GPU allowance mismatch')
            else:
                kept.append(line)
        return '\n'.join(kept)

    def require_idle():
        require(not checked_query('compute-apps', 'gpu_uuid,pid,process_name'), 'foreign/app GPU work present')

    def checked_verify(h, process, start, *args, **kwargs):
        verify(h, process, start, *args, **kwargs)
        guard.observe(process, start)

    def checked_pending(*args, **kwargs):
        guard.check()
        sampler.check()
        return pending(*args, **kwargs)

    helpers.query, helpers.identity = checked_query, checked_identity
    helpers.training = SimpleNamespace(**{**vars(training), 'require_idle': require_idle})
    common.verify_process, common.check_pending = checked_verify, checked_pending
    try:
        yield
    finally:
        helpers.query, helpers.identity, helpers.training, common.verify_process, common.check_pending = original


def load_sampling():
    # Only after the controller and real root sampler have exchanged pidfd proof.
    sys.path.insert(0, str(COMPUTE))
    return importlib.import_module('dcgm_sampling')


def mapped_tools(seals):
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        name = fields[5]
        if any(token in Path(name).name for token in ('libdcgm', 'libnvperf', 'libnvidia-ml')):
            path = Path(name)
            require(name in seals and digest(path) == seals[name], 'unsealed mapped counter provider')
            if 'libnvidia-ml' not in path.name:
                require(path.is_relative_to(LIBRARIES), 'counter provider outside pinned tool installation')
            paths.add(name)
    require(any(Path(name).name.startswith('libdcgm.so') for name in paths), 'actual DCGM mapping missing')
    return {name: seals[name] for name in sorted(paths)}


def collect_rows(sampling, agent, handle, group, fields, folder, owner, hard_timeout_s):
    """Preserve the bounded watch protocol while separating empty-query events.

    DCGM 4's GetValuesSince can return an OK API call with a per-field NO_DATA
    callback for an empty cache range. That response is never a ratio sample.
    The API cursor still follows DCGM; field timestamps and coverage require
    actual samples. The retained implementation remains immutable.
    """
    started, cpu_started = time.monotonic(), time.process_time_ns()
    latest, seen = {}, {}
    ready, cursor = False, 0
    query_index = 0
    samples, startup_invalid, duplicates, no_data = 0, 0, 0, 0
    stop_at, next_guard = None, 0.0
    interrupted = []
    previous_handlers = {sig: signal.signal(sig, lambda signum, _: interrupted.append(signum))
                         for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with (folder / 'samples.jsonl').open('x') as stream:
            def append(value):
                stream.write(json.dumps(value, allow_nan=False, separators=(',', ':')) + '\n')
                stream.flush()

            while True:
                now = time.monotonic()
                if interrupted:
                    raise InterruptedError('external sampler received termination')
                if now - started >= hard_timeout_s:
                    raise TimeoutError('private counter watch reached its declared hard bound')
                append(sampling.clock_anchor())
                if now >= next_guard:
                    append(dict(kind='guard', monotonic_ns=time.monotonic_ns(),
                                processes=sampling.guard(owner, os.getpid())))
                    next_guard = now + 1.0
                rows, callback_errors = [], []
                query_index += 1
                callback_index = 0

                def receive(gpu_id, values, count, _):
                    nonlocal callback_index
                    callback_index += 1
                    try:
                        require(0 <= count <= 4096, 'oversized or negative DCGM callback count')
                        for index in range(count):
                            raw = values[index]
                            value = float(raw.value.dbl)
                            row = dict(kind='sample', gpu_id=int(gpu_id), field_id=int(raw.fieldId),
                                field_type=int(raw.fieldType), status=int(raw.status), timestamp_us=int(raw.ts),
                                value=value if math.isfinite(value) else None, raw_value_repr=repr(value),
                                observed_monotonic_ns=time.monotonic_ns())
                            if is_no_data(row):
                                row.update(kind='no_data', callback_count=count, row_index=index,
                                           query_index=query_index, callback_index=callback_index)
                            rows.append(row)
                        return 0
                    except Exception as error:
                        callback_errors.append(f'{type(error).__name__}: {error}')
                        return 1

                callback = agent.dcgmFieldValueEnumeration_f(receive)
                agent.dcgmUpdateAllFields(handle, 1)
                updated = agent.dcgmGetValuesSince(handle, group, fields, cursor, callback, None)
                require(not callback_errors and type(updated) is int and updated >= cursor,
                        f'callback failure or backwards history cursor: {callback_errors}')
                append(sampling.clock_anchor())
                for row in rows:
                    if row['kind'] == 'no_data':
                        row.update(query_cursor_us=cursor, next_query_cursor_us=updated)
                        validate_no_data(row)
                        append(row)
                        no_data += 1
                        continue
                    key = (row['field_id'], row['timestamp_us'])
                    payload = {name: value for name, value in row.items() if name != 'observed_monotonic_ns'}
                    if key in seen:
                        require(seen[key] == payload, 'conflicting repeated cached sample')
                        duplicates += 1
                        continue
                    seen[key] = payload
                    append(row)
                    try:
                        sampling.validate_row(row)
                    except ValueError:
                        if ready:
                            raise
                        startup_invalid += 1
                        continue
                    require(row['timestamp_us'] > latest.get(row['field_id'], 0),
                            'profiling field timestamps moved backwards')
                    latest[row['field_id']] = row['timestamp_us']
                    samples += 1
                cursor = updated
                if not ready and set(latest) == set(FIELDS):
                    ready = True
                    sampling.publish(folder / 'ready.json', dict(pid=os.getpid(),
                        start_ticks=sampling.identity(os.getpid())['start_ticks'], owner=owner,
                        ready_ns=time.monotonic_ns(), fields=FIELDS, period_us=PERIOD_US))
                if not ready and time.monotonic() - started > 10:
                    raise RuntimeError('no complete valid counter epoch within startup allowance')
                if (folder / 'stop').exists() and stop_at is None:
                    stop_at = time.monotonic() + 2 * PERIOD_US / 1e6
                if stop_at is not None and time.monotonic() >= stop_at:
                    if not ready:
                        raise RuntimeError('stop requested before a complete valid epoch')
                    break
                time.sleep(PERIOD_US / 1e6)
            append(sampling.clock_anchor())
    finally:
        for sig, previous in previous_handlers.items():
            signal.signal(sig, previous)
    return dict(samples=samples, startup_invalid_samples=startup_invalid, cached_duplicates=duplicates,
                no_data_events=no_data, observation_seconds=time.monotonic() - started,
                sampler_process_cpu_seconds=(time.process_time_ns() - cpu_started) / 1e9)


def sample_owned_watch(directory, plan, own, controller):
    sampling = load_sampling()
    sampling.catalogue.validate_layout(json.loads((TOOLS / 'layout.json').read_text()))
    original_guard, original_publish = sampling.guard, sampling.publish

    def checked_guard(owner, own_pid, inventory=None):
        controller.check()
        directory.check()
        require(own_pid == own['pid'] and exact_identity(own, process_record(own_pid)),
                'actual root sampler identity changed')
        capped_memory(own['cgroup'])
        rows = original_guard(owner, own_pid, inventory)
        # Keep raw, unfiltered GPU rows and contemporaneous process incarnations.
        # Descendant permission alone is not the later exact-app-owner proof.
        return [{**row, 'identity': process_record(row['pid'])} for row in rows]

    def publish_ready(path, value):
        require(path.name == 'ready.json', 'only owned readiness publication is permitted')
        directory.publish(path.name, value)

    sampling.guard, sampling.publish = checked_guard, publish_ready
    try:
        checked_guard(plan['owner'], own['pid'])
        sys.path.insert(0, str(BINDINGS))
        structures = importlib.import_module('dcgm_structs')
        agent = importlib.import_module('dcgm_agent')
        structures._dcgmInit(str(LIBRARIES))

        def body(handle, group, fields):
            providers = mapped_tools(plan['observer_identities'])
            result = collect_rows(sampling, agent, handle, group, fields, directory, plan['owner'], WORK_SECONDS)
            result['mapped_providers'] = mapped_tools(plan['observer_identities'])
            require(providers.items() <= result['mapped_providers'].items(), 'mapped providers disappeared')
            return result

        return sampling.owned_watch(agent, structures, lambda: sampling.start_embedded(structures), body)
    finally:
        sampling.guard, sampling.publish = original_guard, original_publish


def sampler(directory, plan, *, run_watch=sample_owned_watch):
    """Internal adapter only. Future caller must provide the privileged timeout.

    Before SDK entry a controller pidfd, capped group and two-way identity proof
    are mandatory. Native partial initialization/cleanup is owned_watch's RAII.
    """
    report = dict(status='failed', sdk_entry_attempted=False, counter_collection=True,
                  ordinary_timing=False, full_device_qualified=False, saturation_qualified=False,
                  performance_qualified=False, full_goal_complete=False)
    try:
        report['file_limit_before'] = list(resource.getrlimit(resource.RLIMIT_FSIZE))
        own = validate_sampler_plan(plan, directory.path)
        require(directory.owner_uid == plan['owner']['uid'], 'output/controller owner differs')
        with PinnedProcess(plan['owner']) as controller:
            wait_for_arm(directory, plan, own, controller)
            report['sdk_entry_attempted'] = True
            report.update(run_watch(directory, plan, own, controller))
            controller.check()
            capped_memory(own['cgroup'])
            verify_seals(plan['observer_identities'])
            require(report.get('owned_watches_retired') is True and report.get('embedded_retired') is True
                    and report.get('cleanup_errors') == [], 'private watch/engine retirement unproved')
            report.update(sampler=own, owner=plan['owner'])
            report['sample_identity'] = directory.file_identity('samples.jsonl', 0, JOURNAL_MAX)
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    report['file_limit_after'] = list(resource.getrlimit(resource.RLIMIT_FSIZE))
    # Native helper output cannot promote a component capture into qualification.
    report.update(ordinary_timing=False, full_device_qualified=False, saturation_qualified=False,
                  performance_qualified=False, full_goal_complete=False)
    directory.publish('observer.json', report)
    return report
