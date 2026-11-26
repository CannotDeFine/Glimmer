#!/usr/bin/env python3
"""Collect one bounded aggregate-activity batch for an accepted BERT host pair."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import aggregate_sm_audit as aggregate
import bert_device_capture as capture
import bert_device_observer as observer
import bert_ncu_host_pair as host
import run_bert_device_observation as recipe
import run_bert_host_pair_timing as timing
from run_rnn_v8_validation import digest, save, wait_for_idle

pair, numerical, candidate = timing.pair, timing.numerical, timing.candidate
read, require = pair.read, pair.require
ENTRY = Path(__file__).resolve()
ORDER = [('o0', 'native'), ('o0', 'equal'), ('o2', 'equal'), ('o0', 'native')]
FLAGS = dict(capture.FLAGS, diagnostic_only=True, counter_collection=True,
             native_operation_matched=False, gpu_gap_causality_qualified=False)
LIMIT = 6 * 1024 ** 3


def accepted(binding, status):
    root = Path(binding['attempt']).resolve(strict=True)
    report, review = read(root / 'report.json'), read(root / 'review.json')
    require(digest(root / 'report.json') == binding['report_sha256'] and
            digest(root / 'review.json') == binding['review_sha256'] and
            review['report_sha256'] == binding['report_sha256'] and review['status'] == status,
            'accepted evidence binding changed')
    return root, report, review


def references(plan, *, artifacts=False):
    """Reuse the actual accepted timing prerequisite, not a synthesized six-run plan."""
    require(plan['order'] == [list(row) for row in ORDER] and plan['point'] == timing.POINT and
            plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'fixed four-control activity contract required')
    root, ordinary, _ = accepted(plan['ordinary'], 'host_pair_ordinary_diagnostic_accepted')
    old_plan = read(root / 'plan.json')
    require(digest(root / 'plan.json') == plan['ordinary']['plan_sha256'] and
            ordinary['status'] == 'host_pair_ordinary_batch_requires_review' and
            ordinary['input_identity_unchanged'] is True, 'accepted ordinary prerequisite changed')
    timing.reviewed_entry(old_plan)
    numerical_plan, reference, original, paths = timing.validate_reference(old_plan, artifacts=artifacts)
    paths.update(root / name for name in ('plan.json', 'report.json', 'review.json'))
    require(set(plan['targets']) == set(pair.VARIANTS), 'both target acceptances required')
    for variant in pair.VARIANTS:
        target, report, review = accepted(plan['targets'][variant], 'host_pair_replacement_target_80sm_accepted')
        require(review['variant'] == report['host_pair']['variant'] == variant and
                review['replacement_target_sm_participation_verified'] is True and
                review['participating_sms'] == 80 and report['input_identity_unchanged'] is True and
                report['unit']['returncode'] == 0 and report['unit']['unit_removed'] is True and
                report['host_pair']['runtime_manifest_sha256'] == old_plan['runtime_manifest_sha256'] and
                report['host_pair']['numerical_report_sha256'] == old_plan['numerical_report_sha256'],
                'current own-variant numerical/runtime/target binding required')
        host.check_binding(report['host_pair'])
        paths.update(target / name for name in ('report.json', 'review.json'))
    closure = Path(plan['no_data']['closure'])
    require(digest(closure) == plan['no_data']['closure_sha256'] and
            read(closure)['status'] == 'reviewed_hardware_response_supported_goal_incomplete',
            'accepted sampler correction required')
    capsule = Path(read(closure)['collector_source_capsule'])
    require(digest(capsule) == read(closure)['collector_source_sha256'], 'sampler source capsule changed')
    paths.update((closure, capsule))
    installed, helpers = recipe.observer_paths()
    require({str(path) for path in installed | helpers} <= plan['observer_identities'].keys(),
            'sampler dependency identity omitted')
    observer.verify_seals(plan['observer_identities'])
    paths.update(Path(name) for name in plan['observer_identities'])
    return numerical_plan, reference, original, paths


def reviewed(plan):
    require(digest(ENTRY) == plan['entrypoint_sha256'] and
            digest(plan['launch_review']) == plan['launch_review_sha256'], 'activity launch review changed')
    review = read(plan['launch_review'])
    require(review['status'] == 'host_pair_activity_ready' and
            review['entrypoint_sha256'] == plan['entrypoint_sha256'], 'activity launch is not reviewed')
    for name, expected in review['source_sha256'].items():
        require(digest(name) == expected, 'reviewed activity source changed: ' + name)


def prepare(brief_path):
    brief = read(brief_path)
    evidence = brief['current_evidence']

    def binding(name):
        root = (PROJECT_ROOT / evidence[name]).resolve(strict=True)
        return dict(attempt=str(root), report_sha256=digest(root / 'report.json'),
                    review_sha256=digest(root / 'review.json'))

    ordinary = binding('ordinary')
    ordinary['plan_sha256'] = digest(Path(ordinary['attempt']) / 'plan.json')
    plan = dict(status='prepared_requires_launch_review', entrypoint_sha256=digest(ENTRY),
        authorization_deadline_utc=brief['authorization_deadline_utc'],
        order=[list(row) for row in ORDER], point=timing.POINT,
        runtime_environment={'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'}, ordinary=ordinary,
        targets={variant: binding(variant + '_target_80sm') for variant in pair.VARIANTS},
        no_data={key: brief['no_data_provenance'][key] for key in ('closure', 'closure_sha256')},
        observer_identities=recipe.observer_seals(), **FLAGS)
    references(plan, artifacts=True)
    path = brief_path.with_name('plan.json')
    with path.open('x') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(str(path), flush=True)


def children(plan):
    paths = [Path(name) for name in plan['children']]
    require(len(paths) == len(set(paths)) == len(ORDER) and all(
        path.resolve(strict=True) == path and path.parent == observer.OUTPUT_ROOT and
        path.name.startswith('attempt-') for path in paths), 'four unique dedicated sampler attempts required')
    return paths


def budget(batch, plan):
    require(shutil.disk_usage(batch).free >= pair.DISK_FLOOR, '8GiB disk reserve required')
    total = sum(candidate.check_output(root) for root in [batch, *children(plan)])
    require(total <= LIMIT - 16 * 1024 ** 2, 'whole batch6GiB output limit')
    return total


class ActivityGuard(candidate.ResourceGuard):
    def __init__(self, attempt, mode, batch, plan):
        self.batch, self.plan, self.last_batch_check = batch, plan, None
        super().__init__(attempt, mode)

    def check(self, force=False):
        super().check(force)
        now = time.monotonic()
        if not force and self.last_batch_check is not None and now - self.last_batch_check < .2:
            return
        self.last_batch_check = now
        root = Path('/sys/fs/cgroup' + self.group)
        events = dict(line.split() for line in (root / 'pids.events').read_text().splitlines())
        require((root / 'pids.max').read_text().strip() == '256' and events and
                all(int(value) == 0 for value in events.values()) and
                (root / 'memory.swap.current').read_text().strip() == '0', 'task pressure or swap usage')
        self.receipt.update(batch_output_bytes=budget(self.batch, self.plan), pids_events=events, swap_current=0)


def preflight_binding(attempt, mode, *, observed_ns=None):
    """Persist the exact generated launch receipt, independently replayable after exit."""
    path = attempt / (mode + '-launch.json')
    launch = read(path)
    observed = time.monotonic_ns() if observed_ns is None else observed_ns
    require(launch['mode'] == mode and launch['attempt'] == str(attempt) and
            launch['plan_sha256'] == digest(attempt / 'plan.json') and
            launch['full_preflight_status'] == 'complete_input_seals_unchanged' and
            type(launch['started_ns']) is int and type(launch['preflight_verified_ns']) is int and
            type(launch['outer_deadline_ns']) is int and type(observed) is int and
            launch['preflight_verified_ns'] == launch['started_ns'] and
            launch['outer_deadline_ns'] - launch['started_ns'] == 820_000_000_000 and
            0 <= observed - launch['started_ns'] <= 30_000_000_000,
            'fresh exact parent input preflight required')
    return dict(launch_path=str(path), launch_sha256=digest(path), launch=launch, observed_ns=observed)


def validate_preflight(attempt, mode, binding):
    require(preflight_binding(attempt, mode, observed_ns=binding['observed_ns']) == binding,
            'persisted parent launch/preflight binding changed')


def worker_inputs(attempt, mode):
    child = read(attempt / 'plan.json')
    batch = Path(child['batch']).resolve(strict=True)
    require(batch.parent == PROJECT_ROOT / 'artifacts/profiles/bert-host-pair-activity' and
            batch.name.startswith('attempt-'), 'dedicated activity batch required')
    plan = read(batch / 'plan.json')
    require(digest(batch / 'plan.json') == child['batch_plan_sha256'] and
            type(child['run_index']) is int and 0 <= child['run_index'] < len(ORDER) and
            children(plan)[child['run_index']] == attempt and ORDER[child['run_index']][1] == mode,
            'worker batch/index/recipe binding differs')
    reviewed(plan)
    numerical.authorization(plan, 800)
    preflight = preflight_binding(attempt, mode)
    owner = preflight['launch']['collector']
    require(owner['uid'] == os.getuid() > 0 and owner['pid'] != os.getpid(), 'nonroot collector required')
    with observer.PinnedProcess(owner):
        observer.verify_seals(plan['observer_identities'])
        old = read(Path(plan['ordinary']['attempt']) / 'plan.json')
        numerical_plan = read(Path(old['numerical_reference']) / 'plan.json')
        variant = ORDER[child['run_index']][0]
        selected = pair.selected_plan(numerical_plan, variant)
        _, target_report, _ = accepted(plan['targets'][variant], 'host_pair_replacement_target_80sm_accepted')
        target = target_report['host_pair']
        require(target['variant'] == variant, 'worker target variant differs')
        host.check_binding(target)
        require(all(selected[key] == target[key] for key in ('server', 'frontend')),
                'own-variant immutable runtime differs')
    return batch, plan, child['run_index'], numerical_plan, read(target['numerical_report']), target, preflight


def summarize(activity, job):
    fields = activity['fields']
    require(set(fields) == {str(value) for value in aggregate.FIELDS}, 'complete activity fields required')
    for row in fields.values():
        values = [row[key] for key in ('interior_coverage_fraction', 'whole_window_ratio_lower_bound',
                                      'whole_window_ratio_upper_bound', 'interior_duration_weighted_ratio')]
        require(all(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                    for value in values) and values[0] >= .95 and values[1] <= values[2],
                'insufficient or invalid activity coverage/bounds')
    sm = fields['1002']
    return dict(sm_active=sm['interior_duration_weighted_ratio'],
                sm_lower=sm['whole_window_ratio_lower_bound'], sm_upper=sm['whole_window_ratio_upper_bound'],
                minimum_coverage=min(row['interior_coverage_fraction'] for row in fields.values()),
                **timing.counts(job))


def decision(rows):
    require([(row['variant'], row['mode']) for row in rows] == ORDER, 'exact four-control order required')
    require(all(type(row[key]) is int and row[key] >= 0 for row in rows
                for key in ('inference', 'training')), 'integer progress counts required')
    require(all(all(type(row[key]) in (int, float) and math.isfinite(row[key]) for key in
                    ('sm_lower', 'sm_upper', 'minimum_coverage')) and
                0 <= row['sm_lower'] <= row['sm_upper'] <= 1 and .95 <= row['minimum_coverage'] <= 1
                for row in rows), 'valid covered activity bounds required')
    first, o0, o2, last = rows
    overlap = max(first['sm_lower'], last['sm_lower']) <= min(first['sm_upper'], last['sm_upper'])
    drift = {key: min(first[key], last[key]) > 0 and max(first[key], last[key]) * 5 <=
             min(first[key], last[key]) * 6 for key in ('inference', 'training')}
    progress = all(row[key] > 0 for row in (o0, o2) for key in ('inference', 'training'))
    eligible = overlap and all(drift.values()) and progress
    verdict = 'inconclusive'
    if eligible and o2['sm_lower'] > o0['sm_upper']:
        verdict = 'supports_increased_aggregate_activity'
    elif eligible and o2['sm_upper'] < o0['sm_lower']:
        verdict = 'aggregate_activity_decreased'
    return dict(eligible=eligible, native_bounds_overlap=overlap, native_completion_drift=drift,
                both_equal_paths_progress=progress,
                verdict=verdict, **FLAGS)


def worker(attempt, mode):
    report, guard, supervisor = dict(status='failed', mode=mode, **FLAGS), None, None
    try:
        batch, plan, index, numerical_plan, reference, target, preflight = worker_inputs(attempt, mode)
        report['parent_preflight_binding'] = preflight
        selected, common, h, sustained, measurement, resources = timing.helpers(numerical_plan, ORDER[index][0])
        h.startup_reference, h.export = common.application_references(
            h, read(Path(target['numerical_report']).parent / 'manifest.json'))
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['gpu'], 'GPU/driver changed')
        h.training.require_idle()
        guard = ActivityGuard(attempt, mode, batch, plan)
        report.update(run=index + 1, variant=ORDER[index][0], native_reference=target['native_reference'],
                      resources_before=resources.resource_snapshot())
        require(report['resources_before']['available_device_sms'] == 80, 'whole device unavailable')
        directory = attempt / mode
        directory.mkdir()
        supervisor = capture.SamplerSupervisor(attempt, mode, plan, guard)
        try:
            proof = supervisor.start()
            with observer.scoped_helpers(common, h, guard, proof):
                job = sustained.run_control(h, measurement, directory / 'application', mode,
                                            h.training.clean_environment(observer.GPU_UUID), timing.POINT)
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
        begin, end = capture.audit_application(sustained, measurement, h, directory / 'application', job, timing.POINT)
        timing.validate_job(selected, reference, job)
        rows = aggregate.read_records(directory / 'observer/samples.jsonl')
        report['activity'] = aggregate.analyze_records(rows, begin, end, supervisor.receipt['ready']['ready_ns'])
        report['guard_audit'] = capture.audit_guards(rows, job, proof.expected, supervisor.watch_plan['owner'], begin, end)
        report['summary'] = summarize(report['activity'], job)
        raw = directory / 'application/attempt.json'
        raw_digest = digest(raw)
        report['job'] = capture.profiled_envelope(job, supervisor.receipt, raw_digest)
        report['job']['raw_helper_receipt']['path'] = 'application/attempt.json'
        save(directory / 'attempt.json', report['job'])
        report['resources_after'] = resources.resource_snapshot()
        require(report['resources_after'] == report['resources_before'], 'device resource policy changed')
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
                report.update(status='failed', cleanup_error=str(error))
        if guard is not None:
            report['resources'] = guard.receipt
        if report['status'] == 'failed' and 'job' in report:
            report['job'].update(status='failed', outer_qualification_invalidated=True)
            save(attempt / mode / 'attempt.json', report['job'])
        save(attempt / (mode + '-worker.json'), report)
    return int(report['status'] != 'passed_requires_review')


def collect(plan_path):
    plan = read(plan_path)
    numerical.authorization(plan, 4100)
    reviewed(plan)
    numerical_plan, reference, original, paths = references(plan, artifacts=True)
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector loader')
    require(shutil.disk_usage(PROJECT_ROOT).free >= LIMIT + pair.DISK_FLOOR, 'output plus disk reserve required')
    require(subprocess.run(['/usr/bin/sudo', '-n', '/usr/bin/true'], timeout=10, check=False).returncode == 0,
            'actual-context sampler authorization unavailable')
    _, common, h, _, _, _ = timing.helpers(numerical_plan, 'o0')
    runtime = {name: value for name, value in original.items() if recipe.production_input(Path(name))}
    require(h.training.manifest({Path(name) for name in runtime}) == runtime,
            'accepted numerical runtime/workload identities changed')
    gpu = h.query('gpu', h.training.GPU_FIELDS)
    require(gpu == reference['gpu_identity_after'], 'numerical GPU identity changed')
    with plan_path.with_name('launch.json').open('x') as stream:
        json.dump(dict(entry=str(ENTRY), plan_sha256=digest(plan_path)), stream)
    batch = create_attempt(resolve_artifact_root('bert-host-pair-activity', category='profiles'))
    runs = [create_attempt(resolve_artifact_root('bert-current-device', category='profiles')) for _ in ORDER]
    plan = dict(plan, children=[str(path) for path in runs], gpu=gpu)
    save(batch / 'plan.json', plan)
    for index, child in enumerate(runs):
        save(child / 'plan.json', dict(batch=str(batch), batch_plan_sha256=digest(batch / 'plan.json'), run_index=index))
    paths.update({ENTRY, plan_path, Path(plan['launch_review']), batch / 'plan.json'})
    paths.update(child / 'plan.json' for child in runs)
    paths.update(Path(module.__file__).resolve(strict=True) for module in tuple(sys.modules.values())
                 if getattr(module, '__file__', None) and Path(module.__file__).is_relative_to(PROJECT_ROOT))
    before = h.training.manifest(paths)
    save(batch / 'inputs-before.json', before)
    report = dict(status='started', jobs=[], **FLAGS)
    print('Host pair aggregate activity: ' + str(batch), flush=True)
    try:
        report['idle_window'] = wait_for_idle(batch)
        require(report['idle_window']['achieved'], 'idle window unavailable')
        for index, ((variant, mode), child) in enumerate(zip(ORDER, runs)):
            numerical.authorization(plan, 840)
            require(h.training.manifest(paths) == before, 'frozen inputs changed before unit')
            h.training.require_idle()
            started = time.monotonic_ns()
            save(child / (mode + '-launch.json'), dict(started_ns=started, mode=mode, attempt=str(child),
                full_preflight_status='complete_input_seals_unchanged', preflight_verified_ns=started,
                outer_deadline_ns=started + 820_000_000_000, plan_sha256=digest(child / 'plan.json'),
                collector=observer.process_record(os.getpid())))
            report['launched_run'] = index + 1
            save(batch / 'report.json', report)
            try:
                unit, result = candidate.launch_unit(child, mode, worker_script=ENTRY)
            finally:
                unchanged = h.training.manifest(paths) == before
                save(child / (mode + '-postflight.json'), dict(input_identity_unchanged=unchanged))
                require(unchanged, 'frozen inputs changed during unit')
            report['jobs'].append(dict(run=index + 1, variant=variant, mode=mode,
                                      attempt=str(child), unit=unit, worker=result))
            budget(batch, plan)
            save(batch / 'report.json', report)
            print(json.dumps(dict(run=index + 1, variant=variant, mode=mode, **result['summary'])), flush=True)
        report['decision'] = decision([dict(variant=row['variant'], mode=row['mode'], **row['worker']['summary'])
                                       for row in report['jobs']])
        report['status'] = 'host_pair_activity_requires_independent_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            after = h.training.manifest(paths)
            save(batch / 'inputs-after.json', after)
            report['input_identity_unchanged'] = before == after
            require(before == after and h.query('gpu', h.training.GPU_FIELDS) == gpu, 'final input/GPU changed')
            h.training.require_idle()
            report['total_output_bytes'] = budget(batch, plan)
        except BaseException as error:
            report.update(status='failed', identity_error=str(error))
        save(batch / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), batch=str(batch))), flush=True)
    return int(report['status'] == 'failed')


def analyze(batch):
    """Reparse original application journals and raw DCGM samples without hardware."""
    plan, report = read(batch / 'plan.json'), read(batch / 'report.json')
    reviewed(plan)
    require(report['status'] == 'host_pair_activity_requires_independent_review' and
            report['input_identity_unchanged'] is True and len(report['jobs']) == len(ORDER),
            'complete activity batch required')
    numerical_plan, reference, _, _ = references(plan, artifacts=True)
    _, common, h, sustained, measurement, _ = timing.helpers(numerical_plan, 'o0')
    old_plan = read(Path(plan['ordinary']['attempt']) / 'plan.json')
    h.startup_reference, h.export = common.application_references(
        h, read(Path(old_plan['numerical_reference']) / 'manifest.json'))
    before = read(batch / 'inputs-before.json')
    require(before == read(batch / 'inputs-after.json') == h.training.manifest({Path(name) for name in before}),
            'activity input identity changed')
    controls = []
    for index, (child, row) in enumerate(zip(children(plan), report['jobs'])):
        variant, mode = ORDER[index]
        require((row['variant'], row['mode'], row['attempt'], row['run']) ==
                (variant, mode, str(child), index + 1), 'raw activity order differs')
        worker, unit = row['worker'], row['unit']
        validate_preflight(child, mode, worker['parent_preflight_binding'])
        require(worker == read(child / (mode + '-worker.json')) and unit == read(child / (mode + '-unit.json')) and
                worker['status'] == 'passed_requires_review' and unit['returncode'] == 0 and
                unit['unit_removed'] is True and
                read(child / (mode + '-postflight.json'))['input_identity_unchanged'] is True,
                'normal raw unit/worker/postflight required')
        selected = pair.selected_plan(numerical_plan, variant)
        h.training.SERVER, h.training.PRELOAD = Path(selected['server']), Path(selected['frontend'])
        directory = child / mode
        raw = directory / 'application/attempt.json'
        job = read(raw)
        begin, end = capture.audit_application(sustained, measurement, h, raw.parent, job, timing.POINT)
        timing.validate_job(selected, reference, job)
        receipt = worker['sampler_supervision']
        require(receipt['status'] == 'retired_requires_review' and receipt['normal_retirement'] is True and
                receipt['wrapper_returncode'] == 0 and receipt['root_pidfd_exited'] is True,
                'actual sampler retirement required')
        with observer.PrivateDirectory(directory / 'observer', os.getuid()) as owned:
            require(owned.read('observer.json', 0) == receipt['observer'] == receipt['terminal_observer'] and
                    owned.file_identity('samples.jsonl', 0, observer.JOURNAL_MAX) == receipt['sample_identity'],
                    'raw root sampler/sample identities differ')
        envelope = capture.profiled_envelope(job, receipt, digest(raw))
        envelope['raw_helper_receipt']['path'] = 'application/attempt.json'
        require(envelope == worker['job'] == read(directory / 'attempt.json'), 'counter provenance differs')
        rows = aggregate.read_records(directory / 'observer/samples.jsonl')
        activity = aggregate.analyze_records(rows, begin, end, receipt['ready']['ready_ns'])
        guards = capture.audit_guards(rows, job, receipt['sampler'], receipt['ready']['owner'], begin, end)
        summary = summarize(activity, job)
        require(activity == worker['activity'] and guards == worker['guard_audit'] and
                summary == worker['summary'] and worker['resources_before'] == worker['resources_after'],
                'raw activity/accounting/resource reparse differs')
        controls.append(dict(run=index + 1, variant=variant, mode=mode, **summary))
    result = decision(controls)
    require(result == report['decision'], 'recomputed activity decision differs')
    return dict(status='host_pair_activity_analysis_requires_review', controls=controls, decision=result, **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', type=Path)
    group.add_argument('--collect', type=Path)
    group.add_argument('--analyze', type=Path)
    group.add_argument('--worker-attempt', type=Path)
    parser.add_argument('--worker-mode', choices=pair.MODES)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode is not None, 'worker mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if args.prepare:
        prepare(args.prepare.resolve(strict=True))
        return 0
    if args.analyze:
        result = analyze(args.analyze.resolve(strict=True))
        with (args.analyze / 'analysis.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(result['decision']))
        return 0
    return collect(args.collect.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
