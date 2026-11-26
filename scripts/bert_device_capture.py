"""Current-owner sustained BERT observation integration, with bounded ownership.

Cold import does no work. Counter captures never qualify ordinary timing,
spatial coverage, sustained saturation, performance or whole-application qualification.
"""

import copy
from bisect import bisect_left, bisect_right
from collections import Counter
from contextlib import contextmanager
import importlib
import json
import os
import re
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

import aggregate_sm_audit as aggregate
from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_device_observer as observer
import run_bert_candidate_validation as candidate
import run_bert_device_observation as recipe
from run_rnn_v8_validation import command, digest, identity, save, wait_for_idle

FLAGS = dict(ordinary_timing=False, ordinary_timing_evidence=False, full_device_qualified=False,
             physical_priority_qualified=False, saturation_qualified=False,
             performance_qualified=False, full_goal_complete=False, full_card_acceptance=False,
             saturation_acceptance=False, performance_acceptance=False, preemption_acceptance=False,
             ordinary_timing_acceptance=False)
SUCCESS = 'sustained_controls_require_independent_review'
PROFILED = 'current_candidate_profiled_control_requires_review'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def helpers():
    common, h = candidate.load_helpers()
    sustained = importlib.import_module('collect_sustained_application')
    require(sustained.ordinary.common is common, 'one explicit current-owner helper binding required')
    require(h.training.SERVER == candidate.BUILD / 'bin/glimmer_execution_service'
            and h.training.PRELOAD == candidate.BUILD / 'lib/libglimmer_execution_interceptor.so',
            'historical Release/default interceptor cannot substitute')
    return common, h, sustained


@contextmanager
def helper_configuration(common, h, variant, depth_output=None):
    """Select only the reviewed runtime; restore all helper bindings on failure."""
    if variant is None:
        yield
        return
    owner, environment = common.service_command, common.owned_environment
    server, preload = h.training.SERVER, h.training.PRELOAD

    def service_command(*args, **kwargs):
        argv, pins = owner(*args, **kwargs)
        require('--skip-idle-helper-plans' not in argv, 'duplicate helper runtime selection')
        argv = [*argv, '--skip-idle-helper-plans'] if variant['helper_plans'] else argv
        if variant.get('issue_depth', False):
            require(depth_output is not None and depth_output.is_absolute()
                    and '--issue-depth-output' not in argv, 'explicit unique hardware depth path required')
            argv = [*argv, '--issue-depth-output', str(depth_output)]
        require('--issue-depth-events' not in argv, 'duplicate owner-event selection')
        if variant.get('issue_events', False):
            require(variant.get('issue_depth') is True, 'owner events require depth output')
            argv = [*argv, '--issue-depth-events']
        require('--issue-depth-selections' not in argv, 'duplicate selected-command observation')
        if variant.get('selected_commands', False):
            require(variant.get('issue_events') is True, 'selected commands require owner events')
            argv = [*argv, '--issue-depth-selections']
        return argv, pins

    def owned_environment(*args, **kwargs):
        selected = environment(*args, **kwargs)
        require(all(key not in selected for key in variant['runtime_environment']), 'inherited helper runtime variant')
        return dict(selected, **variant['runtime_environment'])

    h.training.SERVER = Path(variant['build']) / 'bin/glimmer_execution_service'
    h.training.PRELOAD = Path(variant['build']) / 'lib/libglimmer_execution_interceptor.so'
    common.service_command, common.owned_environment = service_command, owned_environment
    try:
        yield
    finally:
        common.service_command, common.owned_environment = owner, environment
        h.training.SERVER, h.training.PRELOAD = server, preload


@contextmanager
def worker_configuration(sustained, workers):
    """Scope the diagnostic count to one worker process; preserve raw receipts."""
    require(type(workers) is int and workers in (320, 640, 1280), 'bounded worker count required')
    if workers == 320:
        yield
        return
    original_command = sustained.owner_command
    original_log = sustained.ordinary.ordinary_log

    def owner_command(*args, **kwargs):
        argv, pins = original_command(*args, **kwargs)
        require(argv.count('--persistent-workers') == 1, 'one worker count required')
        index = argv.index('--persistent-workers') + 1
        require(argv[index] == '320', 'original worker policy changed')
        argv = list(argv)
        argv[index] = str(workers)
        return argv, pins

    def configured_log(content):
        require(not re.search(r'\[(?:error|critical)\]', content, re.I), 'owner reported an error')
        require(not re.search(r'(?: audit:|Block completion|Copy RPC|==PROF==)', content),
                'counter control contains per-operation observation')
        lines = [line for line in content.splitlines() if 'Execution service configuration' in line]
        pattern = (sustained.ordinary.common.PREFIX + r'Execution service configuration: scheduler=priority, '
                   r'kernel_mode=preemptive, configured_workers=' + str(workers)
                   + r', nccl_enabled=false; full_device_qualified=false')
        require(len(lines) == 1 and re.fullmatch(pattern, lines[0]) is not None,
                'exact diagnostic worker configuration missing or changed')
        return dict(scheduler='priority', kernel_mode='preemptive', configured_workers=workers,
                    nccl_enabled=False, full_device_qualified=False)

    sustained.owner_command, sustained.ordinary.ordinary_log = owner_command, configured_log
    try:
        yield
    finally:
        sustained.owner_command, sustained.ordinary.ordinary_log = original_command, original_log


def startup_budget(attempt, mode, *, now_ns=time.monotonic_ns):
    unit = f'glimmer-bert-{attempt.name}-{mode}.service'
    properties = command(['systemctl', '--user', 'show', unit,
                          '--property=ActiveEnterTimestampMonotonic,RuntimeMaxUSec'])
    values = dict(line.split('=', 1) for line in properties.splitlines())
    require(values.get('RuntimeMaxUSec') in ('13min 20s', '800s', '800000000'),
            'actual unit runtime bound differs')
    entered_us = int(values['ActiveEnterTimestampMonotonic'])
    launch = recipe.read(attempt / (mode + '-launch.json'))
    require(type(launch['outer_deadline_ns']) is int and type(launch['started_ns']) is int
            and launch['outer_deadline_ns'] - launch['started_ns'] == 820_000_000_000,
            'exact parent820s deadline required')
    now = now_ns()
    deadline = min(entered_us * 1000 + 800_000_000_000, launch['outer_deadline_ns'])
    # Includes timeout710s + root kill5s + controller cleanup15s + unit stop5s.
    require(0 < entered_us * 1000 <= now and deadline - now >= 735_000_000_000,
            'insufficient actual remaining unit/outer lifetime; no privileged launch')
    return dict(unit_started_ns=entered_us * 1000, outer_deadline_ns=launch['outer_deadline_ns'],
                observed_ns=now, deadline_ns=deadline, minimum_remaining_ns=735_000_000_000)


def worker_binding(attempt, mode, plan, *, now_ns=time.monotonic_ns):
    """Retain full outer seal brackets without charging multi-GiB I/O to unit.

    Never use this as a full input verification or a standalone capture gate.
    Each unit's collector must fully verify all inputs immediately before launch
    and after removal; it accepts no control until that postflight succeeds.
    """
    recipe.validate_structure(plan)
    launch = recipe.read(attempt / (mode + '-launch.json'))
    require(launch.get('mode') == mode and launch.get('attempt') == str(attempt)
            and launch.get('full_preflight_status') == 'complete_input_seals_unchanged'
            and launch['plan_sha256'] == digest(attempt / 'plan.json')
            and type(launch['preflight_verified_ns']) is int
            and launch['preflight_verified_ns'] == launch['started_ns']
            and 0 <= now_ns() - launch['preflight_verified_ns'] <= 30_000_000_000,
            'fresh exact parent full-seal preflight required')
    owner = launch['collector']
    require(owner.get('uid') == os.getuid() > 0 and owner['pid'] != os.getpid(), 'exact live nonroot collector required')
    with observer.PinnedProcess(owner) as parent:
        parent.check()
        observer.verify_seals(plan['observer_identities'])
        local = {candidate.BUILD / name for name in ('bin/glimmer_execution_service',
            'lib/libglimmer_execution_interceptor.so', 'src/backend/cuda/cuda_sgemm.ptx',
            'src/backend/cuda/cuda_sgemm_tile32.ptx', 'CMakeCache.txt', 'compile_commands.json')}
        if 'helper_variant' in plan:
            local.update(recipe.helper_runtime_paths(plan['helper_variant']))
        for name, value in plan['input_identities'].items():
            path = Path(name)
            require(str(path.resolve(strict=True)) == value['target'] and path.stat().st_size == value['bytes'],
                    'worker input target/size changed: ' + name)
            if path in local or any(path.is_relative_to(PROJECT_ROOT / prefix) for prefix in ('src', 'include')):
                require(digest(path) == value['sha256'], 'worker source/candidate identity changed: ' + name)
        parent.check()
    return dict(preflight_verified_ns=launch['preflight_verified_ns'], plan_sha256=launch['plan_sha256'],
                collector=owner, scope='local source/tool/candidate plus required outer full-input brackets')


class SamplerSupervisor:
    """Own sudo/timeout Popen + actual root pidfd, never signal any root PID.

    A failed stop has bounded observation but cannot claim normal retirement.
    The independently privileged timeout retains its710+5s hard containment.
    """

    def __init__(self, attempt, mode, plan, guard):
        self.attempt, self.mode, self.plan, self.guard = attempt, mode, plan, guard
        self.directory = self.process = self.proof = self.log = None
        self.receipt = dict(status='not_started', normal_retirement=False)

    def check(self):
        self.guard.check()
        require(self.process.poll() is None and identity(self.process.pid) == self.wrapper_start
                and observer.cgroup(self.process.pid) == self.guard.group, 'owned sampler wrapper exited/changed')

    def wait_file(self, name, seconds):
        deadline = time.monotonic() + seconds
        try:
            while not self.directory.exists(name):
                self.check()
                require(time.monotonic() < deadline, 'bounded sampler startup failed: ' + name)
                time.sleep(.02)
            self.check()
        except BaseException as error:
            self.receipt['startup_error'] = f'{type(error).__name__}: {error}'
            try:
                if self.directory.exists('observer.json'):
                    self.receipt['startup_observer'] = self.directory.read('observer.json', 0)
            except Exception as detail_error:
                self.receipt['startup_receipt_error'] = f'{type(detail_error).__name__}: {detail_error}'
            raise

    def start(self):
        folder = self.attempt / self.mode / 'observer'
        folder.mkdir(mode=0o700)
        self.directory = observer.PrivateDirectory(folder, os.getuid())
        owner = observer.process_record(os.getpid())
        require(owner['uid'] > 0 and owner['cgroup'] == self.guard.group, 'exact capped nonroot controller required')
        self.watch_plan = dict(version=1, mode=self.mode, owner=owner, nonce=uuid.uuid4().hex,
            gpu_uuid=observer.GPU_UUID, fields=list(aggregate.FIELDS), period_us=aggregate.PERIOD_US,
            hard_timeout_s=700, observer_identities=self.plan['observer_identities'])
        self.directory.publish('observer-plan.json', self.watch_plan)
        self.receipt.update(status='starting', budget=startup_budget(self.attempt, self.mode))
        argv = [str(folder) if value == '{observer_folder}' else value
                for value in recipe.PRIVILEGED_COMMAND]
        self.log = (folder.parent / 'sampler-wrapper.log').open('x')
        self.process = subprocess.Popen(argv, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, env={key: value for key, value in os.environ.items()
                if key not in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT')})
        self.wrapper_start = identity(self.process.pid)
        self.guard.observe(self.process, self.wrapper_start)
        self.receipt.update(command=argv, wrapper=dict(pid=self.process.pid, start=self.wrapper_start))
        self.wait_file('identity.json', 20)
        self.proof = observer.arm_sampler(self.directory, self.watch_plan, self.process.pid, self.guard)
        self.receipt['sampler'] = self.proof.expected
        self.wait_file('ready.json', 20)
        ready = self.directory.read('ready.json', 0)
        require(ready['pid'] == self.proof.pid and ready['start_ticks'] == self.proof.expected['start_ticks']
                and ready['owner'] == owner and ready['fields'] == list(aggregate.FIELDS)
                and ready['period_us'] == aggregate.PERIOD_US and type(ready['ready_ns']) is int
                and 0 < ready['ready_ns'] <= time.monotonic_ns(), 'exact live root readiness required')
        self.proof.check()
        self.receipt.update(status='ready', ready=ready)
        return self.proof

    def stop(self):
        if self.process is None:
            return
        try:
            if not self.directory.exists('stop'):
                # Exclusive, zero-byte request owned by the nonroot controller.
                fd = os.open('stop', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=self.directory.fd)
                os.close(fd)
            try:
                returned = self.process.wait(timeout=15)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError('root retirement unproved; independent710+5s timeout remains active') from error
            self.receipt['wrapper_returncode'] = returned
            if self.directory.exists('observer.json'):
                # Preserve the root cause even when the wrapper exits nonzero.
                self.receipt['terminal_observer'] = self.directory.read('observer.json', 0)
            require(type(returned) is int and returned == 0 and self.proof is not None
                    and self.proof.poll() is not None, 'wrapper status and actual sampler exit both required')
            report = self.receipt.get('terminal_observer')
            require(type(report) is dict, 'exact root normal-watch retirement receipt required')
            sample = self.directory.file_identity('samples.jsonl', 0, observer.JOURNAL_MAX)
            require(report.get('status') == 'external_observation_collected_requires_window_review'
                    and report.get('sampler') == self.proof.expected
                    and report.get('owner') == self.watch_plan['owner']
                    and report.get('owned_watches_retired') is True and report.get('embedded_retired') is True
                    and report.get('cleanup_errors') == [] and report.get('sample_identity') == sample
                    and report.get('counter_collection') is True and report.get('ordinary_timing') is False,
                    'exact root source/sample/normal-watch retirement receipt required')
            self.receipt.update(status='retired_requires_review', normal_retirement=True, observer=report,
                                root_pidfd_exited=True, sample_identity=sample)
        except BaseException as error:
            self.receipt.update(status='failed', normal_retirement=False, error=f'{type(error).__name__}: {error}')
            raise

    def close(self):
        errors = []
        for owned in (self.proof, self.log, self.directory):
            if owned is not None:
                try:
                    owned.close()
                except Exception as error:
                    errors.append(str(error))
        require(not errors, 'sampler proof/log/directory cleanup failed: ' + '; '.join(errors))


def normal_application(job):
    require(job.get('status') == SUCCESS, 'original sustained application failed: ' + str(job.get('error')))
    mode = job['mode']
    roles = ('inference',) if mode == 'exclusive' else ('training', 'inference')
    identities = dict(job['application_identities'])
    if mode in ('equal', 'priority'):
        identities['service'] = job['server_identity']
    require(set(identities) == set(roles) | ({'service'} if mode in ('equal', 'priority') else set()),
            'exact actual application/service inventory required')
    for role, expected in identities.items():
        cleanup = job['authenticated_child_cleanup'][role]
        require(type(job['final_returncodes'][role]) is int and job['final_returncodes'][role] == 0
                and cleanup['exit_confirmed'] is True and cleanup['signals'] == []
                and (cleanup['pid'], cleanup['start']) == (expected['pid'], expected['start']),
                'normal unsignalled exact sustained child cleanup required')
    require(job['socket_retained'] is False, 'owned service socket retained')
    expected_gpu = ({identities['service']['pid']} if mode in ('equal', 'priority') else
                    {identities[role]['pid'] for role in roles})
    require(set(job['observed_gpu_pids']) == expected_gpu, 'actual application/service GPU owners absent or changed')
    return expected_gpu


def audit_guards(records, job, sampler, owner, begin, end):
    expected = normal_application(job)
    known = expected | {sampler['pid']}
    starts = {row['pid']: row['start'] for row in job['application_identities'].values()}
    if job.get('server_identity'):
        starts[job['server_identity']['pid']] = job['server_identity']['start']
    guards = [row for row in records if row.get('kind') == 'guard']
    require(len(guards) >= 2 and guards[0]['monotonic_ns'] <= begin
            and guards[-1]['monotonic_ns'] >= end, 'raw owner guards do not bracket offered window')
    previous, inside = None, 0
    for row in guards:
        stamp = row['monotonic_ns']
        require(type(stamp) is int and stamp > 0 and (previous is None or stamp > previous),
                'invalid/reversed raw owner guard timestamp')
        if previous is not None and stamp >= begin and previous <= end:
            require(stamp - previous <= 2_500_000_000, 'raw owner guard gap exceeds2.5s')
        previous = stamp
        require(type(row['processes']) is list and len(row['processes']) <= 3, 'bounded raw GPU inventory required')
        seen = set()
        for process in row['processes']:
            require(process.get('gpu_uuid') == observer.GPU_UUID and type(process.get('pid')) is int
                    and process['pid'] in known and process['pid'] not in seen, 'foreign/duplicate raw GPU owner')
            pid = process['pid']
            expected_identity = (sampler if pid == sampler['pid'] else
                dict(pid=pid, start_ticks=starts[pid], uid=owner['uid'], cgroup=owner['cgroup']))
            require(observer.exact_identity(expected_identity, process.get('identity')),
                    'raw guard process incarnation/UID/cgroup differs')
            seen.add(process['pid'])
        if begin <= stamp <= end:
            require(expected <= seen, 'root sampler cannot substitute for actual application/service GPU owners')
            inside += 1
    require(inside > 0, 'no actual concurrent raw owner observations')
    return dict(samples=len(guards), inside_window=inside, required_gpu_owners=sorted(expected),
                observer_allowance=sampler, exact_application_identities=job['application_identities'],
                server_identity=job.get('server_identity'), ownership_observation_verified=True)


def audit_application(sustained, measurement, h, directory, job, point):
    normal_application(job)
    require(recipe.read(directory / 'attempt.json') == job, 'raw sustained envelope differs')
    for name, checksum in job['artifact_identities'].items():
        path = directory / name
        require(path.resolve(strict=True).is_relative_to(directory) and digest(path) == checksum,
                'raw sustained output changed: ' + name)
    roles = ('inference',) if job['mode'] == 'exclusive' else ('training', 'inference')
    require(job['application_commands'] == sustained.commands(h, directory, roles), 'application recipe differs')
    if job['mode'] in ('equal', 'priority'):
        processes = {role: SimpleNamespace(pid=row['pid']) for role, row in job['application_identities'].items()}
        starts = {role: row['start'] for role, row in job['application_identities'].items()}
        argv, pins = sustained.owner_command(h, Path(job['socket_path']), processes, starts, job['mode'])
        require(argv == job['server_command'] and pins == job['client_pins']
                and sustained.ordinary.ordinary_log((directory / 'server.log').read_text()) == job['service_configuration'],
                'current service command/pins/worker policy differs')
    expected = sustained.schedule(measurement, point, job['schedule']['begin_ns'])
    require(job['schedule'] == recipe.read(directory / 'inference.start.json') == expected, 'frozen request trace differs')
    reparsed = copy.deepcopy(job)
    sustained.assess(measurement, directory, reparsed)
    require(all(job[key] == reparsed[key] for key in ('metrics', 'workers_cpu', 'completion', 'numerical_scope')),
            'independent finite/full-inference/arrival accounting differs')
    begin, end = job['metrics']['window_begin_ns'], job['metrics']['window_end_ns']
    require(begin == expected['begin_ns'] and end - begin == 30_002_000_000,
            'fixed sustained offered window required')
    return begin, end


def profiled_envelope(job, observation, raw_digest):
    result = copy.deepcopy(job)
    result.update(FLAGS, status=PROFILED, counter_collection=True, diagnostic_only=True,
                  raw_helper_receipt=dict(path='application/helper-attempt.json', sha256=raw_digest,
                    warning='historical helper flags are not canonical externally-profiled provenance'),
                  counter_observation=observation)
    return result


def validate_hardware_depth(document, job, path, begin, end, *, expected_events=False,
                            expected_generations=False, expected_selections=False):
    """Authenticate application identities independently; connection IDs are producer-sourced."""
    from run_bert_delivery_boundary import DEPTH_OPCODES, DEPTH_CATEGORIES

    owners = job['application_identities']
    require(job['mode'] == 'equal' and set(owners) == {'training', 'inference'}
            and len({(row['pid'], row['start']) for row in owners.values()}) == 2,
            'two distinct authenticated applications required')
    pins = [f"{owners[role]['pid']}:{owners[role]['start']}:0:{selection}"
            for role, selection in (('training', 'configured'), ('inference', 'original'))]
    argv = job['server_command']
    require(job['client_pins'] == pins
            and [argv[index+1] for index, value in enumerate(argv) if value == '--client-priority'] == pins
            and argv.count('--issue-depth-output') == 1
            and argv[argv.index('--issue-depth-output')+1] == str(path), 'actual depth pins/path differ')
    require(type(expected_selections) is bool and (not expected_selections or expected_generations)
            and type(expected_generations) is bool and (not expected_generations or expected_events)
            and type(document['schema']) is int
            and type(expected_events) is bool
            and document['schema'] == (5 if expected_selections else 3 if expected_generations else 2 if expected_events else 1)
            and ('events' in document) == expected_events
            and ('selections' in document) == expected_selections
            and argv.count('--issue-depth-events') == int(expected_events)
            and argv.count('--issue-depth-selections') == int(expected_selections)
            and document['diagnostic_only'] is True and document['valid'] is True
            and document['failure'] == document['unclosed'] == 0
            and document['pid'] == job['server_identity']['pid']
            and document['readiness_is_instantaneous'] is True and document['outstanding_is_gpu_busy'] is False,
            'invalid hardware depth publication')
    slots, connections, roles, total = document['slots'], set(), {}, 0
    require(len(slots) == 2, 'exactly two positive ordinary compute slots required')
    for slot in slots:
        matched = [role for role, owner in owners.items()
                   if (owner['pid'], owner['start']) == (slot['pid'], slot['start_ticks'])]
        require(all(type(value) is int and value >= 0 for value in slot.values())
                and slot['connection'] > 0 and slot['connection'] not in connections
                and len(matched) == 1 and matched[0] not in roles
                and slot['admitted'] == slot['submitted'] == slot['retired'] > 0,
                'foreign, repeated or incomplete compute slot')
        connections.add(slot['connection'])
        roles[matched[0]] = dict(slot)
        total += slot['admitted']
    census = {}
    for row in document['opcodes']:
        require(type(row['opcode']) is int and row['opcode'] in DEPTH_OPCODES
                and row['opcode'] not in census and type(row['admitted']) is int and row['admitted'] > 0,
                'unsupported or repeated depth opcode')
        census[row['opcode']] = row['admitted']
    require(sum(census.values()) == total, 'depth opcode census does not close')
    origin, first, last, width = (document[key] for key in ('origin_ns', 'begin_ns', 'end_ns', 'bucket_ns'))
    require(all(type(value) is int for value in (origin, first, last, width, begin, end))
            and width == 200_000_000 and origin % width == 0 and origin <= first < origin+width
            and first < last < origin+4096*width and end-begin == 30_002_000_000
            and first <= begin < end <= last, 'depth/common-window clock differs')
    buckets, selected = document['buckets'], []
    require(len(buckets) == (last-origin)//width+1, 'depth bucket inventory incomplete')
    for index, row in enumerate(buckets):
        left = origin+index*width
        require(all(type(value) is int and value >= 0 for value in row.values())
                and row['begin_ns'] == left
                and row['covered_ns'] == max(0, min(left+width, last)-max(left, first))
                and row['issue_call_ns'] <= row['admitted_not_submitted_ns']
                and 0 <= row['no_admitted_compute_ns'] <= row['no_inflight_compute_ns'] <= row['covered_ns']
                and sum(row[key] for key in DEPTH_CATEGORIES) == row['observations']
                and row['ready_without_inflight'] <= row['ready_compute'] and row['unknown'] == 0,
                'depth accounting is inconsistent')
        if left >= begin and left+width <= end and row['covered_ns'] == width:
            selected.append(row)
    require(sum(row['covered_ns'] for row in buckets) == last-first
            and all(sum(row[key] for row in buckets) == total for key in ('admitted', 'submitted', 'retired')),
            'complete hardware depth ledger does not close')
    duration = len(selected)*width
    require(duration*100 >= (end-begin)*95, 'complete interior depth coverage below95%')
    sums = {key: sum(row[key] for row in selected) for key in selected[0]
            if key not in ('begin_ns', 'maximum_observation_gap_ns')}
    states = dict(no_admitted=sums['no_admitted_compute_ns'],
                  admitted_but_no_inflight=sums['no_inflight_compute_ns']-sums['no_admitted_compute_ns'],
                  issuing_or_submitted=duration-sums['no_inflight_compute_ns'])
    require(sum(states.values()) == duration, 'host state partition lost coverage')
    dominant = [key for key, value in states.items() if value*100 >= duration*60]
    result = dict(roles=roles, opcodes=census, admitted=total,
        connection_binding_source='service_ledger', connection_ids_independently_observed=False,
        pid_start_independently_authenticated=True, offered_begin_ns=begin, offered_end_ns=end,
        included_begin_ns=selected[0]['begin_ns'], included_end_ns=selected[-1]['begin_ns']+width,
        effective_duration_ns=duration, omitted_edge_ns=end-begin-duration,
        complete_bucket_coverage=duration/(end-begin), host_state_ns=states,
        host_state_fraction={key: value/duration for key, value in states.items()},
        next_direction=dominant[0] if dominant else 'mixed', bucket_totals=sums,
        maximum_observation_gap_ns=max(row['maximum_observation_gap_ns'] for row in selected),
        mean_admitted_not_submitted_depth=sums['admitted_not_submitted_ns']/duration,
        mean_submitted_not_retired_depth=sums['submitted_not_retired_ns']/duration,
        readiness_is_instantaneous=True, outstanding_is_gpu_busy=False,
        diagnostic_only=True, full_device_qualified=False, performance_qualified=False,
        gpu_gap_causality_qualified=False)
    if expected_events:
        producer_connection_contract = {slot['connection']: slot['retired'] for slot in roles.values()}
        validate_hardware_events(document['events'], producer_connection_contract, census, first, last,
                                 expected_generations=expected_generations, expected_selections=expected_selections)
        result['events'] = hardware_event_phases(document['events']['records'], begin, end,
                                                expected_generations=expected_generations,
                                                expected_selections=expected_selections)
        result['events'].update(completed=total, same_plan_coverage=1.0,
            command_census_independently_observed=False, connection_sequence_and_opcode_source='service_ledger')
        if expected_selections:
            from run_bert_delivery_boundary import validate_issue_selections
            result['events']['selections'] = validate_issue_selections(document)
            result['events']['earlier_selection_combinations'] = hardware_selection_combinations(
                document, begin, end, result['events']['observation_generations'])
    return result


def validate_hardware_events(events, producer_connection_contract, census, first, last, *,
                             expected_generations=False, expected_selections=False):
    """Close producer event rows against slots with independently authenticated PID/start."""
    from run_bert_delivery_boundary import EVENT_FIELDS, GENERATION_FIELDS, PREPARATION_FIELDS

    require(type(expected_generations) is bool and type(expected_selections) is bool
            and (not expected_selections or expected_generations), 'explicit generation/selection flags required')
    fields = PREPARATION_FIELDS if expected_selections else GENERATION_FIELDS if expected_generations else EVENT_FIELDS
    require(set(events) == {'capacity', 'complete', 'ready_duration', 'gpu_timestamps', 'fields', 'records'}
            and type(events['capacity']) is int and events['capacity'] == 524288
            and events['complete'] is True and events['ready_duration'] is False
            and events['gpu_timestamps'] is False and events['fields'] == list(fields)
            and type(events['records']) is list
            and len(events['records']) == sum(producer_connection_contract.values()) <= events['capacity'],
            'incomplete hardware owner-event schema')
    keys, serials, counts, operations = set(), set(), Counter(), Counter()
    for row in events['records']:
        require(type(row) is list and len(row) == len(fields)
                and all(type(value) is int and (0 if (expected_generations and index == 13)
                                                or (expected_selections and index in (15, 16)) else 1)
                        <= value <= 0xffffffffffffffff for index, value in enumerate(row)),
                'invalid hardware owner-event row')
        require(not expected_generations or row[14] > row[13], 'invalid observation-generation order')
        if expected_selections:
            require(row[15] in (0, 1)
                    and (row[2] == 11 and row[3] <= row[16] <= row[4] if row[15] else row[16] == 0),
                    'hardware preparation provenance or timestamp differs')
        connection, sequence, opcode = row[:3]
        key, serial = (connection, sequence), row[10]
        require(connection in producer_connection_contract and key not in keys and serial not in serials
                and opcode in census and first <= row[3] <= row[9] <= last
                and all(left <= right for left, right in zip(row[3:9], row[4:10]))
                and row[11] >= row[12] > 0, 'hardware owner-event identity, chronology or plan differs')
        keys.add(key)
        serials.add(serial)
        counts[connection] += 1
        operations[opcode] += 1
    require(counts == Counter(producer_connection_contract) and operations == Counter(census),
            'hardware owner-event slot or opcode census differs')


def hardware_event_phases(records, begin, end, *, expected_generations=False, expected_selections=False):
    """Partition command latency mass, never wall time or GPU-idle duration."""
    require(type(expected_generations) is bool and type(expected_selections) is bool
            and (not expected_selections or expected_generations), 'explicit generation phase selection required')
    cohort = [row for row in records if begin <= row[7] < end]
    selected = [row for row in cohort if begin <= row[3] and row[8] <= end]
    require(cohort and len(selected)*100 >= len(cohort)*95,
            'complete command timestamp coverage below95% of issue-begin cohort')
    require(all(all(left <= right for left, right in zip(row[3:9], row[4:10])) for row in selected),
            'nonnegative command latency partition required')
    names = ('admitted_to_first_observed', 'first_to_matched_observed', 'matched_to_authorized',
             'authorized_to_issue_begin', 'issue_begin_to_end')
    phases = {name: sum(row[index+1]-row[index] for row in selected)
              for name, index in zip(names, range(3, 8))}
    total = sum(phases.values())
    require(total > 0, 'positive complete command latency mass required')
    dominant = [name for name, duration in phases.items() if duration*100 >= total*60]
    retirement = sorted(row[9]-row[8] for row in selected)
    retirement_summary = dict(count=len(retirement), mean_ns=sum(retirement)/len(retirement),
        minimum_ns=retirement[0], maximum_ns=retirement[-1],
        **{f'p{percentile}_ns': retirement[(len(retirement)*percentile+99)//100-1]
           for percentile in (50, 95, 99)})
    result = dict(issue_begin_cohort=len(cohort), included_commands=len(selected),
                omitted_edge_commands=len(cohort)-len(selected),
                complete_command_coverage=len(selected)/len(cohort), phase_latency_ns=phases,
                total_latency_ns=total, phase_latency_fraction={name: value/total for name, value in phases.items()},
                next_boundary=dominant[0] if len(dominant) == 1 else 'mixed',
                host_retirement_latency=retirement_summary,
                weighting='workload_weighted_command_latency_mass',
                latency_mass_is_wall_time=False, continuous_readiness=False, gpu_timestamps=False,
                admission_starvation_excluded=False, first_phase_separates_dependencies_from_owner_revisit=False)
    if expected_generations:
        result['observation_generations'] = hardware_generation_groups(
            selected, phases[names[0]], first_phase_dominant=result['next_boundary'] == names[0],
            expected_selections=expected_selections)
    return result


def hardware_generation_groups(records, first_phase_ns, *, first_phase_dominant, expected_selections=False):
    """Subdivide first-phase latency mass; slot observations are not owner-loop iterations."""
    names = ('first_observation_positive', 'earlier_observations_without_positive')
    counts, masses = dict.fromkeys(names, 0), dict.fromkeys(names, 0)
    for row in records:
        require(len(row) == (17 if expected_selections else 15) and all(type(value) is int for value in row[13:15])
                and 0 <= row[13] < row[14] <= 0xffffffffffffffff,
                'ordered same-row observation generations required')
        name = names[int(row[14]-row[13] > 1)]
        counts[name] += 1
        masses[name] += row[4]-row[3]
    require(sum(masses.values()) == first_phase_ns and all(value >= 0 for value in masses.values()),
            'generation latency masses must close to first phase')
    dominant = [name for name, value in masses.items() if first_phase_ns > 0 and value*100 >= first_phase_ns*60]
    return dict(command_counts=counts, command_count_fraction={name: count/len(records) for name, count in counts.items()},
        first_phase_latency_ns=masses,
        first_phase_latency_fraction={name: value/first_phase_ns if first_phase_ns else None for name, value in masses.items()},
        first_phase_dominant=first_phase_dominant,
        next_boundary=(dominant[0] if len(dominant) == 1 else 'mixed') if first_phase_dominant else 'not_selected',
        weighting='first_phase_command_latency_mass', counts_select_direction=False,
        owner_loop_iterations_inferred=False, same_step_inferred=False, dependency_cause_identified=False,
        continuous_readiness=False, gpu_gap_causality_qualified=False)


def hardware_selection_combinations(document, begin, end, generations):
    """Partition whole first-phase mass by instantaneous observation category sets.

    Call after the complete schema5 ledger validator. Prefix counts and binary
    searches keep overlapping admitted intervals from multiplying journal scans.
    Missing generations identify zero pending only, never empty queues or duration.
    """
    slots, targets = {}, {}
    for connection, generation, stamp, sequence, classification in document['selections']['records']:
        require(classification in (0, 1, 2), 'unknown earlier selection')
        slot = slots.setdefault(connection, dict(generations=[], stamps=[], prefix=[(0, 0, 0)]))
        require(not slot['generations'] or generation > slot['generations'][-1],
                'earlier selections must preserve same-slot generation order')
        require(not slot['stamps'] or stamp >= slot['stamps'][-1], 'same-slot timestamp reversal')
        slot['generations'].append(generation)
        slot['stamps'].append(stamp)
        slot['prefix'].append(tuple(count + int(index == classification)
                                    for index, count in enumerate(slot['prefix'][-1])))
        if classification < 2:
            targets.setdefault((connection, sequence), ([], []))[classification].append(generation)
    names = ('zero_pending', 'target_not_ready', 'other_tracked', 'scope_external')
    combinations, count, mass = {}, 0, 0
    for event in document['events']['records']:
        if not (begin <= event[7] < end and begin <= event[3] and event[8] <= end
                and event[14]-event[13] > 1):
            continue
        slot = slots[event[0]]
        lo = bisect_right(slot['generations'], event[13])
        hi = bisect_left(slot['generations'], event[14])
        earlier = event[14]-event[13]-1
        require(0 <= hi-lo <= earlier and (lo == hi or
                event[3] <= slot['stamps'][lo] <= slot['stamps'][hi-1] <= event[4]),
                'earlier selection generation or timestamp interval differs')
        ready, not_ready, external = (right-left for left, right in
                                      zip(slot['prefix'][lo], slot['prefix'][hi]))
        own = targets.get(tuple(event[:2]), ([], []))
        own_ready, own_not_ready = (bisect_left(values, event[14])-bisect_right(values, event[13])
                                    for values in own)
        require(own_ready == 0 and 0 <= own_not_ready <= not_ready,
                'target ready precedes its first positive or not-ready census differs')
        observations = dict(zip(names, (earlier-(hi-lo), own_not_ready,
                                        ready+not_ready-own_not_ready, external)))
        require(sum(observations.values()) == earlier and all(value >= 0 for value in observations.values()),
                'earlier observation categories do not close')
        present = [name for name in names if observations[name]]
        require(present, 'waiting event requires a nonempty observation combination')
        group = combinations.setdefault('+'.join(present), dict(categories=present, commands=0,
            first_phase_latency_ns=0, earlier_observations=0, observation_counts=dict.fromkeys(names, 0),
            other_tracked_ready=0, other_tracked_not_ready=0))
        latency = event[4]-event[3]
        require(latency >= 0, 'negative first-phase latency')
        group['commands'] += 1
        group['first_phase_latency_ns'] += latency
        group['earlier_observations'] += earlier
        group['other_tracked_ready'] += ready
        group['other_tracked_not_ready'] += not_ready-own_not_ready
        for name, value in observations.items():
            group['observation_counts'][name] += value
        count += 1
        mass += latency
    parent = 'earlier_observations_without_positive'
    require(count == generations['command_counts'][parent]
            and mass == generations['first_phase_latency_ns'][parent]
            and len(combinations) <= 15
            and sum(row['commands'] for row in combinations.values()) == count
            and sum(row['first_phase_latency_ns'] for row in combinations.values()) == mass,
            'selection combination partition differs from generation cohort')
    active = generations['first_phase_dominant'] and generations['next_boundary'] == parent
    dominant = [name for name, row in combinations.items()
                if mass > 0 and row['first_phase_latency_ns']*100 >= mass*60]
    for row in combinations.values():
        row['first_phase_latency_fraction'] = row['first_phase_latency_ns']/mass if mass else None
    return dict(commands=count, first_phase_latency_ns=mass, combinations=combinations,
        hierarchy_active=active,
        next_boundary=('not_exercised' if not count else 'not_selected' if not active
                       else dominant[0] if len(dominant) == 1 else 'mixed'),
        weighting='whole_first_phase_mass_per_mutually_exclusive_observation_category_set',
        counts_select_direction=False, observations_are_durations=False, continuous_readiness=False,
        policy_winner_inferred=False, empty_queue_inferred=False, dependency_cause_identified=False,
        gpu_gap_causality_qualified=False)


def hardware_depth_receipt(path, job, begin, end, *, expected_events=False,
                           expected_generations=False, expected_selections=False):
    info = path.lstat()
    maximum = (512 if expected_events else 8)*1024**2
    require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
            and stat.S_IMODE(info.st_mode) == 0o600 and 0 < info.st_size <= maximum,
            'owned bounded hardware depth output required')
    checksum = digest(path)
    document = json.loads(path.read_text(), object_pairs_hook=recipe.unique_object, parse_constant=recipe.reject_constant)
    summary = validate_hardware_depth(document, job, path, begin, end, expected_events=expected_events,
                                      expected_generations=expected_generations, expected_selections=expected_selections)
    require(digest(path) == checksum, 'hardware depth changed during analysis')
    return dict(path=str(path), sha256=checksum, summary=summary)


def worker(attempt, mode):
    report, guard, supervisor = dict(status='started', mode=mode, counter_collection=True, **FLAGS), None, None
    try:
        require(mode in observer.MODES and not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'),
                'explicit mode and clean worker loader required')
        guard = candidate.ResourceGuard(attempt, mode)
        plan = recipe.read(attempt / 'plan.json')
        recipe.authorization_budget(plan, 800)
        report['parent_preflight_binding'] = worker_binding(attempt, mode, plan)
        require(mode in plan['recipe']['modes'], 'mode outside frozen capture recipe')
        report['configured_workers'] = plan['recipe']['workers']
        report['worker_count_diagnostic'] = plan.get('worker_probe', False)
        common, h, sustained = helpers()
        metadata = recipe.read(Path(plan['reference']) / 'manifest.json')
        h.startup_reference, h.export = common.application_references(h, metadata)
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['checkpoint_gpu'], 'actual GPU/driver differs')
        h.training.require_idle()
        resources = sustained.load_resources()
        report['resources_before'] = resources.resource_snapshot()
        require(report['resources_before']['available_device_sms'] == 80, 'full physical device unavailable')
        directory = attempt / mode
        directory.mkdir()
        depth_path = directory / 'issue-depth.json'
        depth_enabled = plan.get('helper_variant', {}).get('issue_depth', False) and mode == 'equal'
        require(not depth_path.exists() and not depth_path.is_symlink(), 'hardware depth output must be fresh')
        supervisor = SamplerSupervisor(attempt, mode, plan, guard)
        try:
            proof = supervisor.start()
            with helper_configuration(common, h, plan.get('helper_variant'), depth_path if depth_enabled else None), \
                 worker_configuration(sustained, plan['recipe']['workers']), \
                 observer.scoped_helpers(common, h, guard, proof):
                job = sustained.run_control(h, sustained.load_measurement(), directory / 'application', mode,
                    h.training.clean_environment(observer.GPU_UUID), plan['point'])
        except BaseException as error:
            report['operation_error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            try:
                supervisor.stop()
            except BaseException as error:
                report['sampler_cleanup_error'] = f'{type(error).__name__}: {error}'
                if 'operation_error' not in report:
                    raise
        guard.check(force=True)
        with helper_configuration(common, h, plan.get('helper_variant'), depth_path if depth_enabled else None), \
             worker_configuration(sustained, plan['recipe']['workers']):
            begin, end = audit_application(sustained, sustained.load_measurement(), h, directory / 'application', job, plan['point'])
        if 'helper_variant' in plan:
            recipe.validate_helper_job(job, plan['helper_variant'])
        if depth_enabled:
            report['issue_depth'] = hardware_depth_receipt(depth_path, job, begin, end,
                expected_events=plan.get('helper_variant', {}).get('issue_events', False),
                expected_generations=plan.get('helper_variant', {}).get('observation_generations', False),
                expected_selections=plan.get('helper_variant', {}).get('selected_commands', False))
            save(directory / 'issue-depth-receipt.json', report['issue_depth'])
        rows = aggregate.read_records(directory / 'observer/samples.jsonl')
        ready = supervisor.receipt['ready']
        report['activity'] = aggregate.analyze_records(rows, begin, end, ready['ready_ns'])
        report['guard_audit'] = audit_guards(rows, job, proof.expected, supervisor.watch_plan['owner'], begin, end)
        raw = directory / 'application/attempt.json'
        raw_digest = digest(raw)
        raw.rename(raw.with_name('helper-attempt.json'))
        report['job'] = profiled_envelope(job, supervisor.receipt, raw_digest)
        save(directory / 'attempt.json', report['job'])
        report['resources_after'] = resources.resource_snapshot()
        require(report['resources_before'] == report['resources_after'], 'physical resource policy changed')
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if supervisor is not None:
            report['sampler_supervision'] = supervisor.receipt
            try:
                supervisor.close()
            except BaseException as error:
                report.update(status='failed', cleanup_error=f'{type(error).__name__}: {error}')
        if guard is not None:
            report['resources'] = guard.receipt
        if report['status'] == 'failed' and 'job' in report:
            report['job'].update(FLAGS, status='failed', outer_qualification_invalidated=True)
            save(attempt / mode / 'attempt.json', report['job'])
        save(attempt / (mode + '-worker.json'), report)
    print(json.dumps(dict(mode=mode, status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] != 'passed_requires_review')


def sampler_entry(folder):
    require(folder.resolve(strict=True) == folder, 'canonical sampler folder required')
    with observer.PrivateDirectory(folder, folder.stat().st_uid) as directory:
        plan = directory.read('observer-plan.json', directory.owner_uid)
        result = observer.sampler(directory, plan)
    return int(result['status'] == 'failed')


def collect(plan_path):
    plan = recipe.read(plan_path)
    recipe.authorization_budget(plan, 2300)
    recipe.validate_plan(plan)
    attempt = create_attempt(resolve_artifact_root('bert-current-device', category='profiles'))
    save(attempt / 'plan.json', plan)
    report = dict(status='started', jobs=[], counter_collection=True, plan_source=str(plan_path),
                  plan_sha256=digest(plan_path), launched_modes=[], **FLAGS)
    print('Current-owner counter evidence: ' + str(attempt), flush=True)
    try:
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector loader')
        require(shutil.disk_usage(attempt).free >= 8 * 1024 ** 3, '8GiB free required; no deletion')
        # Read-only permission probe in the actual launch context, not a claim
        # that a user's other terminal or the subsequent user unit is authorized.
        probe = subprocess.run(['/usr/bin/sudo', '-n', '/opt/nvidia/nsight-compute/2024.3.2/ncu', '--version'],
                               text=True, capture_output=True, timeout=10, check=False)
        save(attempt / 'authentication.json', dict(returncode=probe.returncode, stdout=probe.stdout, stderr=probe.stderr))
        require(probe.returncode == 0, 'actual-context sudo unavailable; no experiment started')
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'sampled idle window unavailable; no experiment started')
        recipe.validate_plan(plan)
        for mode in plan['recipe']['modes']:
            recipe.authorization_budget(plan, 840)
            # Full hashes bracket every actual unit. Their known multi-GiB cost
            # is outside both unit runtime and its parent's820s launch deadline.
            recipe.validate_plan(plan)
            started = time.monotonic_ns()
            save(attempt / (mode + '-launch.json'), dict(started_ns=started, mode=mode, attempt=str(attempt),
                full_preflight_status='complete_input_seals_unchanged',
                outer_deadline_ns=started + 820_000_000_000, preflight_verified_ns=started,
                plan_sha256=digest(attempt / 'plan.json'), collector=observer.process_record(os.getpid())))
            report['launched_modes'].append(mode)
            save(attempt / 'report.json', report)
            try:
                unit, result = candidate.launch_unit(attempt, mode, worker_script=Path(recipe.__file__))
            finally:
                postflight = dict(status='failed', started_ns=time.monotonic_ns())
                try:
                    recipe.validate_plan(plan)
                    postflight['status'] = 'complete_input_seals_unchanged'
                finally:
                    postflight['finished_ns'] = time.monotonic_ns()
                    save(attempt / (mode + '-postflight.json'), postflight)
                    report['input_identity_unchanged'] = postflight['status'] == 'complete_input_seals_unchanged'
            report['jobs'].append(dict(mode=mode, unit=unit, worker=result))
            candidate.check_output(attempt)
            save(attempt / 'report.json', report)
        report['status'] = 'current_candidate_counter_controls_require_independent_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'))), flush=True)
    return int(report['status'] == 'failed')
