#!/usr/bin/env python3
"""Run one bounded ordinary BERT timing batch for a qualified host compiler pair."""

import argparse
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import bert_device_capture as capture
import run_bert_host_pair_numerical as pair
from run_rnn_v8_validation import digest, save, wait_for_idle

candidate, numerical = pair.candidate, pair.numerical
read, require = pair.read, pair.require
ENTRY = Path(__file__).resolve()
ORDER = [('o0', 'native'), ('o0', 'equal'), ('o2', 'equal'),
         ('o2', 'equal'), ('o0', 'equal'), ('o2', 'native')]
POINT = dict(period_ms=7, requests=4286, slo_ms=12, burst_size=1, repetitions=1, warmup=1)
FLAGS = dict(numerical.FLAGS, ordinary_timing=True, counter_collection=False)


def validate_reference(plan, *, artifacts=False):
    """Check the accepted four-path prerequisite without running an application."""
    require(plan['order'] == [list(row) for row in ORDER] and plan['point'] == POINT and
            plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'fixed ordinary batch contract required')
    reference = Path(plan['numerical_reference']).resolve(strict=True)
    require(digest(reference / 'report.json') == plan['numerical_report_sha256'] and
            digest(reference / 'review.json') == plan['numerical_review_sha256'],
            'accepted numerical receipts changed')
    review, report = read(reference / 'review.json'), read(reference / 'report.json')
    require(review['status'] == 'fresh_host_pair_numerical_accepted' and
            review['report_sha256'] == plan['numerical_report_sha256'] and
            report['status'] == 'fresh_o0_o2_native_equal_numerical_requires_review' and
            report['input_identity_unchanged'] is True, 'complete numerical acceptance required')
    numerical_plan = read(reference / 'plan.json')
    pair.require_launch_review(numerical_plan)
    paths = pair.validate_plan(numerical_plan)
    require(plan['runtime_manifest'] == numerical_plan['runtime_manifest'] and
            plan['runtime_manifest_sha256'] == numerical_plan['runtime_manifest_sha256'],
            'same immutable host pair required')
    before = read(reference / 'inputs-before.json')
    require(before == read(reference / 'inputs-after.json') and
            digest(reference / 'inputs-before.json') == review['inputs_manifest_sha256'],
            'numerical input manifests differ')
    paths.update(Path(name) for name in before)
    paths.update(reference / name for name in
                 ('plan.json', 'manifest.json', 'report.json', 'review.json',
                  'inputs-before.json', 'inputs-after.json'))
    require([(row['variant'], row['mode']) for row in report['jobs']] ==
            [(variant, mode) for variant in pair.VARIANTS for mode in pair.MODES],
            'exact four-path numerical inventory required')
    for row in report['jobs']:
        child, mode = Path(report['variant_attempts'][row['variant']]), row['mode']
        worker, unit = row['worker'], row['unit']
        require(worker['status'] == 'passed_requires_review' and worker['reparse'] == pair.ORACLE
                and unit['returncode'] == 0 and unit['unit_removed'] is True,
                'normal complete numerical prerequisite required')
        for name, value in ((mode + '-worker.json', worker), (mode + '-unit.json', unit),
                            (mode + '/attempt.json', worker['job'])):
            path = child / name
            require(read(path) == value, 'raw numerical receipt differs')
            paths.add(path)
        pair.validate_job(read(child / 'plan.json'), worker['job'])
        for name, expected in worker['job']['artifact_identities'].items():
            path = child / mode / name
            require(path.resolve(strict=True).is_relative_to(child / mode), 'numerical artifact escape')
            if artifacts:
                require(digest(path) == expected, 'numerical artifact changed')
            paths.add(path)
    return numerical_plan, report, before, paths


def reviewed_entry(plan):
    require(digest(ENTRY) == plan['entrypoint_sha256'] and
            digest(plan['launch_review']) == plan['launch_review_sha256'], 'timing review changed')
    review = read(plan['launch_review'])
    require(review['status'] == 'host_pair_timing_ready' and
            review['entrypoint_sha256'] == plan['entrypoint_sha256'], 'timing launch not reviewed')


def helpers(numerical_plan, variant):
    selected = pair.selected_plan(numerical_plan, variant)
    common, h = numerical.configure(selected)
    sustained = importlib.import_module('collect_sustained_application')
    require(sustained.ordinary.common is common, 'one helper instance required')
    return selected, common, h, sustained, sustained.load_measurement(), sustained.load_resources()


@contextmanager
def guard_hooks(common, guard):
    verify, pending = common.verify_process, common.check_pending

    def checked_verify(h, process, start, *args, **kwargs):
        verify(h, process, start, *args, **kwargs)
        guard.observe(process, start)

    def checked_pending(*args, **kwargs):
        guard.check()
        return pending(*args, **kwargs)

    common.verify_process, common.check_pending = checked_verify, checked_pending
    try:
        yield
    finally:
        common.verify_process, common.check_pending = verify, pending


def validate_job(selected, reference_report, job):
    capture.normal_application(job)
    mode = job['mode']
    require(mode in pair.MODES, 'native/equal only')
    baseline = reference_report['jobs'][0]['worker']['job']['application_environment']
    require(set(baseline) == pair.BASE_ENVIRONMENT_KEYS, 'clean fixed reference environment required')
    environment = dict(baseline)
    if mode == 'equal':
        owner = job['server_identity']
        environment.update(LD_PRELOAD=selected['frontend'],
            GLIMMER_EXECUTION_SOCKET=job['socket_path'],
            GLIMMER_EXECUTION_SERVER_PID=str(owner['pid']),
            GLIMMER_EXECUTION_SERVER_START=str(owner['start']),
            GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS='240000',
            GLIMMER_EXECUTION_HOST_RANGE_SEEK='1')
        require(job['server_environment'] == baseline, 'extra service environment option')
        command, pins, normalized = job['server_command'], [], []
        index = 0
        while index < len(command):
            if command[index] == '--client-priority':
                pins.append(command[index + 1].split(':'))
                index += 2
            else:
                normalized.append(command[index])
                index += 1
        require(normalized == [selected['server'], '--socket', job['socket_path'],
            '--exit-when-idle', '--persistent-workers', '320', '--kernel-mode', 'preemptive',
            '--scheduler', 'priority', '--replace-gemm', '--memory-per-client', '12884901888',
            '--host-memory-per-client', '536870912', '--priority-idle-us', '2000',
            '--gemm-tile-side', '32'], 'extra or changed ordinary service option')
        require(len(pins) == 2 and all(len(pin) == 4 and pin[2] == '0' for pin in pins)
                and {pin[3] for pin in pins} == {'configured', 'original'} and
                {(int(pin[0]), int(pin[1])) for pin in pins} ==
                {(value['pid'], value['start']) for value in job['application_identities'].values()},
                'exact two equal-priority clients required')
    else:
        require(job.get('server_environment') is None and 'server_command' not in job,
                'native service contamination')
    require(job['application_environment'] == environment, 'actual application environment differs')


def counts(job):
    metrics = job['metrics']
    require(metrics['window_seconds'] == 30.002 and metrics['offered_requests'] == 4286,
            'whole fixed offered window required')
    result = dict(inference=metrics['at_offer_end']['completed_requests'],
                  training=metrics['training_completions_in_window'],
                  contained_training=metrics['training_steps'])
    require(all(type(value) is int and value >= 0 for value in result.values()), 'invalid progress count')
    return result


def decision(controls):
    require([(row['variant'], row['mode']) for row in controls] == ORDER, 'exact six-path order required')
    require(all(type(row[key]) is int and row[key] >= 0 for row in controls
                for key in ('inference', 'training')), 'integer completed counts required')
    eligible = all(row[key] >= 10 for row in controls for key in ('inference', 'training'))
    drift = []
    for left, right in ((0, 5), (1, 4), (2, 3)):
        for key in ('inference', 'training'):
            low, high = sorted((controls[left][key], controls[right][key]))
            passed = low > 0 and high * 5 <= low * 6
            drift.append(dict(runs=[left + 1, right + 1], metric=key,
                              maximum_over_minimum=high / low if low else None, passed=passed))
            eligible = eligible and passed
    comparisons = []
    for optimized, baseline in ((2, 1), (3, 4)):
        a, b = controls[optimized], controls[baseline]
        comparisons.append(dict(optimized_run=optimized + 1, baseline_run=baseline + 1,
            inference_ratio=a['inference'] / b['inference'] if b['inference'] else None,
            training_ratio=a['training'] / b['training'] if b['training'] else None,
            inference_supported=a['inference'] * 5 >= b['inference'] * 6,
            training_preserved=a['training'] * 10 >= b['training'] * 9,
            inference_direction=(a['inference'] > b['inference']) - (a['inference'] < b['inference'])))
    verdict = 'inconclusive'
    if eligible:
        if all(row['inference_supported'] and row['training_preserved'] for row in comparisons):
            verdict = 'supports_retaining_whole_host_optimization_candidate'
        elif all(not row['inference_supported'] for row in comparisons) and not (
                comparisons[0]['inference_direction'] * comparisons[1]['inference_direction'] < 0):
            verdict = 'against_whole_host_compiler_sufficiency'
    return dict(eligible=eligible, drift=drift, comparisons=comparisons, verdict=verdict, **FLAGS)


def worker(attempt, mode):
    report, guard = dict(status='failed', mode=mode, **FLAGS), None
    try:
        plan = read(attempt / 'plan.json')
        numerical.authorization(plan, 800)
        reviewed_entry(plan)
        numerical_plan, reference, _, _ = validate_reference(plan)
        require((plan['variant'], mode) == ORDER[plan['run_index']], 'worker order binding differs')
        selected, common, h, sustained, measurement, resources = helpers(numerical_plan, plan['variant'])
        require(selected['server'] == plan['server'] and selected['frontend'] == plan['frontend'],
                'worker immutable pair differs')
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean worker loader')
        guard = pair.PairGuard(attempt, mode)
        manifest = read(Path(plan['numerical_reference']) / 'manifest.json')
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['gpu'], 'GPU/driver changed')
        report['resources_before'] = resources.resource_snapshot()
        require(report['resources_before']['available_device_sms'] == 80, 'whole device unavailable')
        with guard_hooks(common, guard):
            job = sustained.run_control(h, measurement, attempt / mode, mode,
                h.training.clean_environment(plan['gpu'].split(',')[0].strip()), POINT)
            report['job'] = job
            capture.audit_application(sustained, measurement, h, attempt / mode, job, POINT)
            validate_job(selected, reference, job)
        report['counts'] = counts(job)
        report['resources_after'] = resources.resource_snapshot()
        require(report['resources_after'] == report['resources_before'], 'device resource policy changed')
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / (mode + '-worker.json'), report)
    return int(report['status'] != 'passed_requires_review')


def analyze(attempt):
    plan, report = read(attempt / 'plan.json'), read(attempt / 'report.json')
    reviewed_entry(plan)
    require(report['status'] == 'host_pair_ordinary_batch_requires_review' and
            report['input_identity_unchanged'] is True and len(report['jobs']) == len(ORDER),
            'complete identity-verified ordinary batch required')
    numerical_plan, reference, _, _ = validate_reference(plan, artifacts=True)
    selected, _, h, sustained, measurement, _ = helpers(numerical_plan, 'o0')
    manifest = read(Path(plan['numerical_reference']) / 'manifest.json')
    h.startup_reference, h.export = sustained.ordinary.common.application_references(h, manifest)
    frozen = read(attempt / 'inputs-before.json')
    require(frozen == read(attempt / 'inputs-after.json') ==
            h.training.manifest({Path(name) for name in frozen}), 'timing inputs changed')
    controls = []
    for index, row in enumerate(report['jobs']):
        child, mode = Path(row['attempt']), row['mode']
        require((row['variant'], mode) == ORDER[index], 'actual batch order differs')
        worker, unit = row['worker'], row['unit']
        require(worker == read(child / (mode + '-worker.json')) and
                unit == read(child / (mode + '-unit.json')) and
                worker['status'] == 'passed_requires_review' and unit['returncode'] == 0 and
                unit['unit_removed'] is True, 'normal raw timing worker/unit required')
        selected = pair.selected_plan(numerical_plan, row['variant'])
        h.training.SERVER, h.training.PRELOAD = Path(selected['server']), Path(selected['frontend'])
        capture.audit_application(sustained, measurement, h, child / mode, worker['job'], POINT)
        validate_job(selected, reference, worker['job'])
        require(worker['resources_before'] == worker['resources_after'] and
                worker['counts'] == counts(worker['job']), 'resource policy or progress differs')
        controls.append(dict(run=index + 1, variant=row['variant'], mode=mode, **worker['counts']))
    return dict(status='host_pair_ordinary_analysis_requires_review', controls=controls,
                decision=decision(controls), **FLAGS)


def collect(plan_path):
    plan = read(plan_path)
    numerical.authorization(plan, 5500)
    reviewed_entry(plan)
    numerical_plan, reference, original, paths = validate_reference(plan, artifacts=True)
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector loader')
    require(shutil.disk_usage(PROJECT_ROOT).free >= pair.TOTAL_LIMIT + pair.DISK_FLOOR,
            'aggregate output plus disk reserve required')
    _, common, h, sustained, _, _ = helpers(numerical_plan, 'o0')
    require(h.training.manifest({Path(name) for name in original}) == original,
            'accepted numerical inputs changed before collection')
    gpu = h.query('gpu', h.training.GPU_FIELDS)
    require(gpu == reference['gpu_identity_after'], 'accepted GPU/driver changed')
    with plan_path.with_name('launch.json').open('x') as stream:
        json.dump(dict(entry=str(ENTRY), sha256=digest(ENTRY), plan_sha256=digest(plan_path)), stream)
    attempt = create_attempt(resolve_artifact_root('bert-host-pair-timing'))
    print('Host pair ordinary evidence: ' + str(attempt), flush=True)
    save(attempt / 'plan.json', plan)
    report = dict(status='started', jobs=[], **FLAGS)
    paths.update({ENTRY, plan_path, Path(plan['launch_review'])})
    paths.update(Path(module.__file__).resolve(strict=True) for module in tuple(sys.modules.values())
                 if getattr(module, '__file__', None) and Path(module.__file__).is_relative_to(PROJECT_ROOT))
    children = []
    for index, (variant, mode) in enumerate(ORDER):
        child = create_attempt(resolve_artifact_root('host-pair-timing-run',
                    output_root=attempt / ('run-' + str(index + 1))))
        selected = pair.selected_plan(numerical_plan, variant)
        save(child / 'plan.json', dict(plan, run_index=index, variant=variant, gpu=gpu,
            server=selected['server'], frontend=selected['frontend'], aggregate_attempt=str(attempt)))
        paths.add(child / 'plan.json')
        children.append(child)
    before = h.training.manifest(paths)
    save(attempt / 'inputs-before.json', before)
    try:
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'idle window unavailable')
        for index, ((variant, mode), child) in enumerate(zip(ORDER, children)):
            numerical.authorization(plan, 840)
            require(h.training.manifest(paths) == before, 'frozen inputs changed before launch')
            h.training.require_idle()
            unit, result = candidate.launch_unit(child, mode, worker_script=ENTRY)
            report['jobs'].append(dict(run=index + 1, variant=variant, mode=mode,
                                      attempt=str(child), unit=unit, worker=result))
            pair.aggregate_budget(attempt)
            save(attempt / 'report.json', report)
            print(json.dumps(dict(run=index + 1, variant=variant, mode=mode,
                                  counts=result['counts'])), flush=True)
        report['status'] = 'host_pair_ordinary_batch_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        try:
            after = h.training.manifest(paths)
            save(attempt / 'inputs-after.json', after)
            report['input_identity_unchanged'] = before == after
            require(before == after and h.query('gpu', h.training.GPU_FIELDS) == gpu,
                    'final input/GPU identity changed')
            pair.validate_plan(numerical_plan)
            report['total_output_bytes'] = pair.aggregate_budget(attempt)
        except BaseException as error:
            report.update(status='failed', identity_error=f'{type(error).__name__}: {error}')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), attempt=str(attempt))), flush=True)
    return int(report['status'] == 'failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', type=Path)
    group.add_argument('--analyze', type=Path)
    group.add_argument('--check', type=Path)
    group.add_argument('--worker-attempt', type=Path)
    parser.add_argument('--worker-mode', choices=pair.MODES)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode is not None, 'worker mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if args.check:
        _, _, _, paths = validate_reference(read(args.check), artifacts=True)
        print(json.dumps(dict(status='numerical_prerequisite_checked', identities=len(paths))))
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
