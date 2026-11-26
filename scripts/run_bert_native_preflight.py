#!/usr/bin/env python3
"""Ordinary native BERT correctness and sampled dependency-map preflight.

No sudo, profiler, service, capability grant or automatic retry. Sampling cannot
prove completeness of transient dlopen; a future profiler must still reject
unknown mappings. A prepared, independently reviewed plan is required to run.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import run_bert_native_ncu_capture as native
from run_rnn_v8_validation import cgroup, command, digest, identity, save, wait_for_idle

ENTRY = Path(__file__).resolve()
candidate, pair, protocol, bridge = native.candidate, native.pair, native.protocol, native.bridge
require, read = native.require, native.read
FLAGS = dict(native.FLAGS, counter_collection=False, native_environment_preflight_verified=False,
             sampled_maps_only=True, transient_dlopen_completeness=False)
PHASES = ('wrapper', 'application_startup', 'prepared', 'released')
MAP_POLICY = dict(period_seconds=.02, minimum_samples_per_role=8,
                  maximum_live_gap_ns=1000000000, maximum_terminal_gap_ns=1000000000)


def sample_phase(sample, launch, ready, terminal):
    """Only a sample wholly inside a phase can witness that phase's mappings."""
    begin, end = sample['begin_ns'], sample['monotonic_ns']
    if sample['image'] == 'wrapper':
        return 'wrapper' if end <= launch else 'transition'
    require(sample['image'] == 'application', 'unknown ordinary sampled image')
    if launch <= begin <= end < ready['ready_ns']:
        return 'application_startup'
    if ready['ready_ns'] <= begin <= end < ready['released_ns']:
        return 'prepared'
    if ready['released_ns'] <= begin <= end <= terminal:
        return 'released'
    return 'transition'


def caps0_status(content, uid, gid):
    fields = dict(line.split(':', 1) for line in content.splitlines())
    require(uid > 0 and fields['Uid'].split() == [str(uid)] * 4 and
            fields['Gid'].split() == [str(gid)] * 4 and fields['NoNewPrivs'].strip() == '1' and
            all(int(fields[key], 16) == 0 for key in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb')),
            'ordinary target must remain nonroot NNP with no effective/permitted/inheritable/ambient caps')
    return {name: fields[name].strip() for name in
            ('Uid', 'Gid', 'Groups', 'NoNewPrivs', 'CapEff', 'CapPrm', 'CapInh', 'CapAmb', 'CapBnd')}


def reviewed(plan):
    require(plan['map_policy'] == MAP_POLICY, 'fixed ordinary map sampling policy required')
    require(digest(plan['launch_review']) == plan['launch_review_sha256'], 'ordinary launch review changed')
    review = read(plan['launch_review'])
    require(review['status'] == 'native_environment_preflight_ready' and
            str(ENTRY) in review['source_sha256'], 'ordinary preflight launch review required')
    for path, sha in review['source_sha256'].items():
        require(digest(path) == sha, 'ordinary reviewed source changed: ' + path)
    require(digest(plan['brief']) == plan['brief_sha256'], 'ordinary brief changed')
    return review['source_sha256']


def references(plan):
    activity = Path(plan['activity']).resolve(strict=True)
    require(digest(activity / 'report.json') == plan['activity_report_sha256'] and
            digest(activity / 'review.json') == plan['activity_review_sha256'] and
            read(activity / 'review.json')['status'] == 'host_pair_aggregate_activity_increase_accepted',
            'accepted diagnostic reference binding changed')
    old_plan = read(activity / 'plan.json')
    _, reference, original, paths = native.activity.references(old_plan, artifacts=True)
    repair = Path(plan['repair_review']).resolve(strict=True)
    require(digest(repair) == plan['repair_review_sha256'], 'CPU environment repair review changed')
    record = read(repair)
    require(record['status'] == 'fixed_python_environment_inventory_cpu_repair_accepted' and
            record['ordinary_application_preflight_required'] is True and
            record['historical_mapped_bytes_proven'] is False, 'accepted static-only repair required')
    require(digest(record['audit']) == record['audit_sha256'], 'CPU inventory audit changed')
    for path, sha in record['source_sha256'].items():
        require(digest(path) == sha, 'accepted inventory source changed')
    # Retain existing installed system/vendor/profiler inputs, not its launch
    # authority or output paths. New package files get fresh identities below.
    counter = Path(old_plan['targets']['o0']['attempt'])
    paths.update(Path(path) for path in read(counter / 'inputs-before.json')
                 if not Path(path).is_relative_to(PROJECT_ROOT / 'artifacts'))
    paths.update((repair, Path(record['audit']), activity / 'report.json', activity / 'review.json'))
    own = Path(reference['variant_attempts']['o0']) / 'native'
    job = read(own / 'attempt.json')
    pair.validate_job({}, job)
    return reference, original, paths, own, job


def audit_maps(maps, job, frozen, observer):
    require(set(maps['roles']) == set(protocol.ROLES), 'both original sampled roles required')
    for role, row in maps['roles'].items():
        require(job['application_identities'][role] == dict(pid=row['pid'], start=row['start']) and
                row['role'] == role and row['parent'] == observer['pid'] and row['cgroup'] == observer['cgroup'],
                'sampled role identity or ownership differs')
        terminal, samples = row['terminal'], row['samples']
        require(terminal is not None and type(terminal['returncode']) is int and terminal['returncode'] == 0
                and terminal['pidfd_terminal'] is True and
                terminal['wait_source'] == 'direct_parent_Popen.poll', 'exact ordinary normal wait required')
        require(len(samples) >= MAP_POLICY['minimum_samples_per_role'] and len(samples) <= 50000,
                'ordinary role has insufficient or excessive samples')
        stamps = [sample['monotonic_ns'] for sample in samples]
        require(all(type(value) is int and value > 0 for value in stamps) and
                all(first < second for first, second in zip(stamps, stamps[1:])), 'invalid ordinary sample clock')
        gap = max(second - first for first, second in zip(stamps, stamps[1:]))
        tail = terminal['monotonic_ns'] - stamps[-1]
        require((row['first_sample_ns'], row['last_sample_ns'], row['max_sample_gap_ns'], row['terminal_gap_ns']) ==
                (stamps[0], stamps[-1], gap, tail) and gap <= MAP_POLICY['maximum_live_gap_ns'] and
                0 <= tail <= MAP_POLICY['maximum_terminal_gap_ns'], 'ordinary sample density/gap gate failed')
        ready, launch = job['ready'][role], job['application_launch_ns'][role]
        phases = [sample_phase(sample, launch, ready, terminal['monotonic_ns']) for sample in samples]
        stable = [phase for phase in phases if phase != 'transition']
        require(set(stable) == set(PHASES) and
                list(map(PHASES.index, stable)) == sorted(map(PHASES.index, stable)), 'incomplete lifecycle phases')
        seen, used_sets = set(), set()
        for sample, expected_phase in zip(samples, phases):
            stamp, phase, key = sample['monotonic_ns'], sample['phase'], sample['map_set_sha256']
            require(type(sample['begin_ns']) is int and sample['begin_ns'] > 0 and
                    0 <= stamp - sample['begin_ns'] <= MAP_POLICY['maximum_live_gap_ns'] and
                    phase == expected_phase, 'ordinary phase or sample interval differs')
            names = row['map_sets'][key]
            require(names and names == sorted(set(names)) and
                    hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest() == key,
                    'ordinary sample map-set digest differs')
            seen.update(names)
            used_sets.add(key)
            if phase == 'wrapper':
                require(stamp <= launch, 'wrapper sample followed application release')
            elif phase == 'application_startup':
                require(launch <= stamp < ready['ready_ns'], 'startup sample is outside its original phase')
            elif phase == 'prepared':
                require(ready['ready_ns'] <= stamp < ready['released_ns'], 'prepared sample is outside its barrier')
            elif phase == 'released':
                require(ready['released_ns'] <= stamp <= terminal['monotonic_ns'], 'released sample is outside its phase')
        require(used_sets == set(row['map_sets']) and seen == set(row['mappings']) and
                all(name in frozen and row['mappings'][name] == frozen[name] for name in seen),
                'sampled maps do not equal the frozen observed union')


def audit_resources(resources):
    require(resources['per_file_limit'] == candidate.FILE_MAX and resources['resource_checks'] > 0,
            'ordinary actual file/resource limits required')
    for key in ('initial_memory', 'last_memory'):
        memory = resources[key]
        events = dict(line.split() for line in memory['memory.events'].splitlines())
        require(memory['memory.max'] == str(candidate.MEMORY_MAX) and memory['memory.swap.max'] == '0' and
                memory['memory.oom.group'] == '1' and events and all(int(value) == 0 for value in events.values()),
                'ordinary actual memory limits or pressure gate failed')
    require(resources['pids_max'] == '256' and resources['pids_events'] and
            all(int(value) == 0 for value in resources['pids_events'].values()) and
            resources['swap_current'] == 0 and resources['last_output_bytes'] <= candidate.OUTPUT_MAX and
            resources['disk_available_bytes'] >= pair.DISK_FLOOR, 'ordinary actual task/swap/storage gate failed')


class MapGuard(candidate.ResourceGuard):
    """Passive sampled role mappings; original parent owns children and waits."""

    def __init__(self, attempt, manifest, helpers, inputs, frozen):
        self.manifest, self.helpers = manifest, helpers
        self.commands = protocol.application_commands(manifest, attempt / 'native')
        self.wrapper = [str(protocol.PYTHON), '-B', '-c', helpers.wrapper]
        self.maps, self.rows, self.handles = {}, {}, {}
        self.inputs, self.frozen, self.last_map_check = inputs, frozen, None
        super().__init__(attempt, 'native')

    def observe(self, process, start):
        super().observe(process, start)
        if process.pid not in self.rows:
            output = Path(f'/proc/{process.pid}/fd/1').resolve(strict=True)
            roles = [role for role in protocol.ROLES if output == self.attempt / 'native' / (role + '.log')]
            require(len(roles) == 1 and roles[0] not in {row['role'] for row in self.rows.values()},
                    'one unique original log-bound role required')
            fd = os.pidfd_open(process.pid)
            self.handles[process.pid] = fd
            self.maps[process.pid] = native.NativeMaps(self.inputs, self.frozen)
            self.rows[process.pid] = dict(role=roles[0], pid=process.pid, start=start, samples=[],
                map_sets={}, mappings={}, terminal=None, first_sample_ns=None, last_sample_ns=None,
                max_sample_gap_ns=0, application_seen=False, parent=os.getpid(), cgroup=self.group)
        self.sample(process, start)

    def retired(self, process, row, *, grace=False):
        if grace:
            import select
            select.select([self.handles[process.pid]], [], [], .25)
        code = process.poll()  # Actual direct parent's Popen wait, never synthesized.
        if code is None:
            return False
        require(bridge.terminal(self.handles[process.pid]), 'parent wait lacks matching terminal pidfd')
        row['terminal'] = dict(returncode=code, pidfd_terminal=True, monotonic_ns=time.monotonic_ns(),
                               wait_source='direct_parent_Popen.poll')
        return True

    def sample(self, process, start):
        row = self.rows[process.pid]
        if row['terminal'] is not None or self.retired(process, row):
            return
        begin = time.monotonic_ns()
        try:
            require(identity(process.pid) == start == row['start'] and cgroup(process.pid) == self.group and
                    native.relay.parent(process.pid) == os.getpid() and
                    os.getsid(process.pid) == os.getpgid(process.pid) == process.pid,
                    'ordinary role process ownership changed')
            actual = native.relay.command_line(process.pid)
            if not actual:
                raise ProcessLookupError('ordinary command line disappeared')
            require(actual in (self.wrapper, self.commands[row['role']]) and
                    Path(f'/proc/{process.pid}/exe').samefile(protocol.PYTHON),
                    'ordinary role executable or argv changed')
            raw = Path(f'/proc/{process.pid}/environ').read_bytes()
            if not raw:
                raise ProcessLookupError('ordinary environment disappeared')
            values = [entry.split(b'=', 1) for entry in raw.split(b'\0') if entry]
            require(all(len(entry) == 2 for entry in values) and len(dict(values)) == len(values) and
                    dict(values) == {os.fsencode(key): os.fsencode(value)
                                    for key, value in self.manifest['application_environment'].items()},
                    'ordinary role environment changed')
            row['status_fields'] = caps0_status(Path(f'/proc/{process.pid}/status').read_text(),
                                                self.manifest['uid'], self.manifest['gid'])
            image = 'wrapper'
            if actual == self.commands[row['role']]:
                row['application_seen'] = True
                image = 'application'
            else:
                require(not row['application_seen'], 'ordinary role reverted to its wrapper')
            sample = self.maps[process.pid].observe(process.pid)
            require(identity(process.pid) == start and cgroup(process.pid) == self.group,
                    'ordinary role identity changed during map sampling')
        except (FileNotFoundError, ProcessLookupError):
            require(self.retired(process, row, grace=True), 'live ordinary procfs identity became unavailable')
            return
        stamp = sample['monotonic_ns']
        require(len(row['samples']) < 50000, 'ordinary map sample count bound')
        names = sorted(sample['mappings'])
        key = hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest()
        row['map_sets'].setdefault(key, names)
        row['mappings'].update(sample['mappings'])
        row['samples'].append(dict(begin_ns=begin, monotonic_ns=stamp, image=image, map_set_sha256=key))
        if row['first_sample_ns'] is None:
            row['first_sample_ns'] = stamp
        if row['last_sample_ns'] is not None:
            require(stamp > row['last_sample_ns'], 'ordinary sample clock regressed')
            row['max_sample_gap_ns'] = max(row['max_sample_gap_ns'], stamp - row['last_sample_ns'])
        row['last_sample_ns'] = stamp

    def check(self, force=False):
        previous = self.receipt['resource_checks'] if hasattr(self, 'receipt') else 0
        super().check(force)
        if self.receipt['resource_checks'] != previous:
            root = Path('/sys/fs/cgroup') / self.group.lstrip('/')
            self.receipt.update(pids_max=(root / 'pids.max').read_text().strip(),
                pids_events=dict(line.split() for line in (root / 'pids.events').read_text().splitlines()),
                swap_current=int((root / 'memory.swap.current').read_text()),
                disk_available_bytes=shutil.disk_usage(self.attempt).free)
            # ResourceGuard assigns initial_memory after its first virtual check.
            initial = self.receipt.get('initial_memory', self.receipt['last_memory'])
            audit_resources(dict(self.receipt, initial_memory=initial))
        for pid, (process, _) in self.participants.items():
            if pid in self.rows and self.rows[pid]['terminal'] is None:
                self.retired(process, self.rows[pid])
        now = time.monotonic()
        if force or self.last_map_check is None or now - self.last_map_check >= MAP_POLICY['period_seconds']:
            self.last_map_check = now
            for pid, (process, start) in self.participants.items():
                if pid in self.rows:
                    self.sample(process, start)

    def finish(self, job):
        self.check(force=True)
        require(len(self.rows) == 2, 'both ordinary native roles required')
        for row in self.rows.values():
            expected = job['application_identities'][row['role']]
            require(expected == dict(pid=row['pid'], start=row['start']) and
                    row['terminal'] is not None and row['terminal']['returncode'] == 0,
                    'ordinary role lacks its exact normal parent wait')
            ready = job['ready'][row['role']]
            for sample in row['samples']:
                sample['phase'] = sample_phase(sample, job['application_launch_ns'][row['role']],
                                               ready, row['terminal']['monotonic_ns'])
            row['terminal_gap_ns'] = row['terminal']['monotonic_ns'] - row['last_sample_ns']
            require(row['terminal_gap_ns'] >= 0, 'ordinary terminal clock regressed')
        result = dict(scope='sampled lifecycle map union, not transient dlopen completeness',
                      roles={row['role']: row for row in self.rows.values()})
        audit_maps(result, job, self.frozen, self.receipt['observer'])
        return result

    def close(self):
        for fd in self.handles.values():
            os.close(fd)
        self.handles.clear()


def worker(attempt):
    report, guard = dict(status='failed', **FLAGS), None
    try:
        plan, manifest = read(attempt / 'plan.json'), read(attempt / 'manifest.json')
        require(digest(attempt / 'plan.json') == manifest['plan_sha256'] == digest(manifest['plan_source']) and
                read(manifest['plan_source']) == plan, 'ordinary copied plan binding changed')
        native.activity.numerical.authorization(plan, 800)
        require(reviewed(plan) == manifest['source_sha256'], 'ordinary worker source binding changed')
        report['controller_status'] = caps0_status(Path('/proc/self/status').read_text(),
                                                  manifest['uid'], manifest['gid'])
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean ordinary worker')
        inputs, frozen = read(attempt / 'inputs-before.json'), read(attempt / 'mapping-inputs.json')
        require(digest(attempt / 'mapping-inputs.json') == manifest['mapping_inputs_sha256'] and
                digest(attempt / 'native-loader.json') == manifest['loader_closure_sha256'],
                'ordinary frozen input receipt changed')
        common, helpers = candidate.load_helpers()
        helpers.startup_reference, helpers.export = common.application_references(helpers, manifest)
        gpu = manifest['gpu'].split(',')[0].strip()
        require(helpers.training.clean_environment(gpu) == manifest['application_environment'] and
                helpers.query('gpu', helpers.training.GPU_FIELDS) == manifest['gpu'],
                'ordinary original environment or GPU changed')
        guard = MapGuard(attempt, manifest, helpers, inputs, frozen)
        job = candidate.guarded_control(common, helpers, attempt, 'native', manifest['application_environment'], guard)
        report['job'] = job
        pair.validate_job({}, job)
        require(job['application_commands'] == guard.commands, 'ordinary native commands changed')
        report['maps'] = guard.finish(job)
        report['reparse'] = candidate.reparse(common, helpers, attempt, 'native', job, gpu)
        require(report['reparse'] == pair.ORACLE, 'complete original numerical oracle required')
        guard.check(force=True)
        report['status'] = 'native_environment_preflight_worker_requires_review'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        if guard is not None:
            report['resources'] = guard.receipt
            report['sampled_roles'] = list(guard.rows.values())
            guard.close()
        save(attempt / 'native-worker.json', report)
    return int(report['status'] == 'failed')


def launch_unit(attempt):
    unit = f'glimmer-bert-{attempt.name}-native.service'
    description = 'Glimmer ordinary native preflight ' + attempt.name
    require(native.ncu.unit_properties(unit)['LoadState'] == 'not-found', 'fresh ordinary unit required')
    argv = ['systemd-run', '--user', '--unit=' + unit, '--description=' + description, '--wait', '--pipe',
        '--property=MemoryMax=' + str(candidate.MEMORY_MAX), '--property=MemorySwapMax=0',
        '--property=OOMPolicy=kill', '--property=RuntimeMaxSec=800s', '--property=TimeoutStopSec=5s',
        '--property=KillMode=control-group', '--property=TasksMax=256', '--property=NoNewPrivileges=yes',
        '--property=LimitFSIZE=' + str(candidate.FILE_MAX), '/usr/bin/env', '-u', 'LD_PRELOAD', '-u', 'LD_AUDIT',
        '-u', 'LD_LIBRARY_PATH', str(protocol.PYTHON), '-B', str(ENTRY), '--worker-attempt', str(attempt)]
    result = dict(unit=unit, command=argv, returncode=None, unit_removed=False)
    try:
        with (attempt / 'unit.log').open('x') as stream:
            result['returncode'] = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                                 timeout=820, check=False).returncode
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            state = native.ncu.unit_properties(unit)
            if state['LoadState'] != 'not-found':
                require(state['Description'] == description, 'refuse to stop an unowned ordinary unit')
                command(['systemctl', '--user', 'stop', unit], timeout=15)
                if native.ncu.unit_properties(unit)['ActiveState'] == 'failed':
                    command(['systemctl', '--user', 'reset-failed', unit], timeout=5)
            result['unit_removed'] = native.ncu.unit_properties(unit)['LoadState'] == 'not-found'
        except BaseException as error:
            result['cleanup_error'] = f'{type(error).__name__}: {error}'
        save(attempt / 'unit.json', result)
    require(result['returncode'] == 0 and result['unit_removed'] and
            'error' not in result and 'cleanup_error' not in result, 'ordinary bounded unit failed')
    return result


def collect(plan_path):
    plan = read(plan_path)
    native.activity.numerical.authorization(plan, 1800)
    require(not plan_path.with_name('launch.json').exists(), 'ordinary one-shot claim already exists')
    sources = reviewed(plan)
    reference, original, paths, own, job = references(plan)
    require(os.getuid() > 0 and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
            'clean nonroot ordinary collector required')
    require(shutil.disk_usage(PROJECT_ROOT).free >= candidate.OUTPUT_MAX + pair.DISK_FLOOR,
            'ordinary output budget plus disk reserve required')
    common, helpers = candidate.load_helpers()
    runtime = {path: value for path, value in original.items() if native.activity.recipe.production_input(Path(path))}
    require(helpers.training.manifest({Path(path) for path in runtime}) == runtime,
            'accepted runtime input bytes changed')
    gpu = helpers.query('gpu', helpers.training.GPU_FIELDS)
    require(gpu == reference['gpu_identity_after'], 'ordinary accepted GPU identity changed')
    attempt = create_attempt(resolve_artifact_root('bert-native-preflight', category='runs'))
    print('Ordinary native preflight: ' + str(attempt), flush=True)
    manifest = dict(gpu=gpu, uid=os.getuid(), gid=os.getgid(), python=str(protocol.PYTHON),
        startup_reference_bundle=str(common.STARTUP_INPUTS), startup_reference_report_sha256=common.STARTUP_INPUTS_SHA,
        application_environment=job['application_environment'], source_sha256=sources, plan_sha256=digest(plan_path),
        plan_source=str(plan_path), map_policy=MAP_POLICY)
    helpers.startup_reference, helpers.export = common.application_references(helpers, manifest)
    manifest.update(assets=str(helpers.training.ASSET_ROOT), startup_reference=str(helpers.startup_reference),
                    export=str(helpers.export))
    require(protocol.application_commands(manifest, own) == job['application_commands'], 'original native recipe changed')
    loader_paths, receipt = native.native_loader_closure(manifest['application_environment'])
    save(attempt / 'native-loader.json', receipt)
    manifest['loader_closure_sha256'] = digest(attempt / 'native-loader.json')
    shutil.copyfile(plan_path, attempt / 'plan.json')
    paths.update(loader_paths | {attempt / 'native-loader.json', attempt / 'plan.json', plan_path,
                                Path(plan['brief']), Path(plan['launch_review'])})
    paths.update(map(Path, sources))
    paths.update(Path(module.__file__).resolve() for module in tuple(sys.modules.values())
                 if getattr(module, '__file__', None) and Path(module.__file__).is_relative_to(PROJECT_ROOT))
    before = helpers.training.manifest(paths)
    frozen = native.mapping_inputs(before)
    save(attempt / 'inputs-before.json', before)
    save(attempt / 'mapping-inputs.json', frozen)
    manifest['mapping_inputs_sha256'] = digest(attempt / 'mapping-inputs.json')
    save(attempt / 'manifest.json', manifest)
    report = dict(status='failed', manifest_sha256=digest(attempt / 'manifest.json'), **FLAGS)
    try:
        native.activity.numerical.authorization(plan, 1600)
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'ordinary idle window unavailable')
        require(helpers.training.manifest(paths) == before and native.mapping_inputs(before) == frozen,
                'ordinary preflight input identity changed')
        native.check_loader_state(receipt)
        native.activity.numerical.authorization(plan, 900)
        helpers.training.require_idle()
        with plan_path.with_name('launch.json').open('x') as stream:
            json.dump(dict(plan_sha256=digest(plan_path), attempt=str(attempt)), stream)
        report['unit'] = launch_unit(attempt)
        worker_report = read(attempt / 'native-worker.json')
        require(worker_report['status'] == 'native_environment_preflight_worker_requires_review' and
                worker_report['reparse'] == pair.ORACLE, 'ordinary worker failed its complete preflight')
        report['worker'] = worker_report
        report['status'] = 'native_environment_preflight_requires_independent_review'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            after = helpers.training.manifest(paths)
            save(attempt / 'inputs-after.json', after)
            report['input_identity_unchanged'] = before == after
            require(before == after and native.mapping_inputs(after) == frozen and
                    read(attempt / 'manifest.json') == manifest and
                    digest(attempt / 'plan.json') == manifest['plan_sha256'] == digest(plan_path),
                    'ordinary final input or copied plan identity changed')
            native.check_loader_state(receipt)
            require(helpers.query('gpu', helpers.training.GPU_FIELDS) == gpu, 'ordinary final GPU identity changed')
            helpers.training.require_idle()
            report['output_bytes'] = candidate.check_output(attempt)
        except BaseException as error:
            report.update(status='failed', final_error=f'{type(error).__name__}: {error}')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), attempt=str(attempt))), flush=True)
    return int(report['status'] == 'failed')


def analyze(attempt):
    """Read-only reparse of actual outputs and sampled map/resource evidence."""
    require(Path(sys.prefix).resolve() == protocol.PYTHON.parent.parent.resolve(),
            'CPU numerical reanalysis requires the fixed venv Python')
    report, manifest, plan = (read(attempt / name) for name in ('report.json', 'manifest.json', 'plan.json'))
    require(report['status'] == 'native_environment_preflight_requires_independent_review' and
            report['input_identity_unchanged'] is True, 'complete ordinary preflight required')
    require(type(report['output_bytes']) is int and
            0 <= report['output_bytes'] <= candidate.OUTPUT_MAX - 8 * 1024 ** 2,
            'ordinary final aggregate output budget failed')
    require(digest(attempt / 'manifest.json') == report['manifest_sha256'] and
            digest(attempt / 'plan.json') == manifest['plan_sha256'] == digest(manifest['plan_source']) and
            read(manifest['plan_source']) == plan and manifest['map_policy'] == MAP_POLICY,
            'ordinary source plan, manifest or sampling policy changed')
    require(reviewed(plan) == manifest['source_sha256'], 'ordinary source binding changed')
    references(plan)
    before, after = read(attempt / 'inputs-before.json'), read(attempt / 'inputs-after.json')
    common, helpers = candidate.load_helpers()
    require(before == after == helpers.training.manifest({Path(name) for name in before}),
            'ordinary frozen input closure changed')
    frozen = read(attempt / 'mapping-inputs.json')
    require(digest(attempt / 'mapping-inputs.json') == manifest['mapping_inputs_sha256'] and
            native.mapping_inputs(before) == frozen and
            digest(attempt / 'native-loader.json') == manifest['loader_closure_sha256'],
            'ordinary mapping or loader identity changed')
    native.check_loader_state(read(attempt / 'native-loader.json'))
    unit, worker_report = read(attempt / 'unit.json'), read(attempt / 'native-worker.json')
    require(unit == report['unit'] and unit['returncode'] == 0 and unit['unit_removed'] is True and
            'error' not in unit and 'cleanup_error' not in unit and
            worker_report == report['worker'] and worker_report['status'] ==
            'native_environment_preflight_worker_requires_review', 'ordinary worker/unit closure differs')
    resources = worker_report['resources']
    observer = resources['observer']
    require(observer['cgroup'].endswith('/glimmer-bert-' + attempt.name + '-native.service'),
            'ordinary resource receipt belongs to another unit')
    audit_resources(resources)
    for fields in [worker_report['controller_status'],
                   *[row['status_fields'] for row in worker_report['maps']['roles'].values()]]:
        caps0_status('\n'.join(key + ':' + value for key, value in fields.items()), manifest['uid'], manifest['gid'])
    job = worker_report['job']
    pair.validate_job({}, job)
    require(job['application_commands'] == protocol.application_commands(manifest, attempt / 'native') and
            job['application_environment'] == manifest['application_environment'] and
            all(value == manifest['application_environment'] for value in job['application_environments'].values()),
            'ordinary original recipe changed')
    audit_maps(worker_report['maps'], job, frozen, observer)
    require(worker_report['sampled_roles'] == list(worker_report['maps']['roles'].values()),
            'ordinary raw sampled role records differ')
    require({(item['pid'], item['start'], item['cgroup']) for item in resources['participants']} ==
            {(item['pid'], item['start'], observer['cgroup']) for item in job['application_identities'].values()},
            'ordinary resource participants differ from authenticated roles')
    helpers.startup_reference, helpers.export = common.application_references(helpers, manifest)
    gpu = manifest['gpu'].split(',')[0].strip()
    require(helpers.query('gpu', helpers.training.GPU_FIELDS) == manifest['gpu'], 'ordinary GPU identity changed')
    oracle = candidate.reparse(common, helpers, attempt, 'native', job, gpu)
    require(oracle == worker_report['reparse'] == pair.ORACLE, 'ordinary complete numerical reparse differs')
    return dict(status='native_environment_preflight_reanalysis_passed', numerical=oracle,
                role_samples={role: len(row['samples']) for role, row in worker_report['maps']['roles'].items()}, **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--collect', type=Path)
    modes.add_argument('--worker-attempt', type=Path)
    modes.add_argument('--analyze', type=Path)
    args = parser.parse_args()
    if args.analyze:
        print(json.dumps(analyze(args.analyze.resolve(strict=True))))
        return 0
    if args.worker_attempt:
        return worker(args.worker_attempt.resolve(strict=True))
    return collect(args.collect.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
