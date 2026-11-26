#!/usr/bin/env python3
"""Collect existing owner-stage ledgers for a qualified immutable BERT host pair."""

import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import sys

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import analyze_bert_service_stages as stage_parser
import run_bert_host_pair_timing as timing
from run_rnn_v8_validation import digest, save, wait_for_idle

pair, numerical, candidate = timing.pair, timing.numerical, timing.candidate
read, require, capture = pair.read, pair.require, timing.capture
ENTRY = Path(__file__).resolve()
ORDER = ['o0', 'o2']
FLAGS = dict(capture.FLAGS, diagnostic_only=True, counter_collection=False,
             host_stage_instrumented=True, gpu_gap_causality_qualified=False,
             priority_qualified=False, service_frontend_causality_qualified=False)
GROUPS = dict(transport=['transport', 'receive'], progression=['observe', 'advance'],
    planning=['plan_receive', 'plan_slot', 'plan_issue', 'plan_other',
              'observations', 'snapshot', 'pending', 'policy'],
    issue=['issue_attempt', 'issue_native'], other=['step'])


def references(plan, *, artifacts=False):
    require(plan['order'] == ORDER and plan['point'] == timing.POINT,
            'one fixed O0/O2 equal stage pair required')
    root = Path(plan['ordinary']['attempt']).resolve(strict=True)
    for name in ('plan', 'report', 'review'):
        require(digest(root / (name + '.json')) == plan['ordinary'][name + '_sha256'],
                'accepted ordinary binding changed')
    ordinary, review = read(root / 'report.json'), read(root / 'review.json')
    require(review['status'] == 'host_pair_ordinary_diagnostic_accepted' and
            review['report_sha256'] == plan['ordinary']['report_sha256'] and
            ordinary['status'] == 'host_pair_ordinary_batch_requires_review' and
            ordinary['input_identity_unchanged'] is True, 'accepted ordinary prerequisite required')
    old = read(root / 'plan.json')
    timing.reviewed_entry(old)
    qualified, reference, original, paths = timing.validate_reference(old, artifacts=artifacts)
    paths.update(root / (name + '.json') for name in ('plan', 'report', 'review'))
    return qualified, reference, original, paths, ordinary


def reviewed(plan):
    require(digest(ENTRY) == plan['entrypoint_sha256'] and
            digest(plan['launch_review']) == plan['launch_review_sha256'], 'stage launch review changed')
    review = read(plan['launch_review'])
    require(review['status'] == 'host_pair_stages_ready' and
            review['entrypoint_sha256'] == plan['entrypoint_sha256'], 'stage launch not reviewed')
    for name, expected in review['source_sha256'].items():
        require(digest(name) == expected, 'reviewed stage source changed: ' + name)


def verify_runtime_inputs(original, manifest):
    # Historical test/document receipts are not runtime prerequisites. Keep their
    # original evidence intact and seal current helper/test bytes for this batch.
    runtime = {name: value for name, value in original.items()
               if capture.recipe.production_input(Path(name))}
    require(runtime and manifest({Path(name) for name in runtime}) == runtime,
            'accepted runtime inputs changed')


@contextmanager
def stage_option(sustained, directory):
    owner = sustained.owner_command

    def command(*args, **kwargs):
        argv, pins = owner(*args, **kwargs)
        require('--host-stage-output' not in argv, 'duplicate stage output option')
        return [*argv, '--host-stage-output', str(directory / 'host-stages.json')], pins

    sustained.owner_command = command
    try:
        yield
    finally:
        sustained.owner_command = owner


def validate_job(selected, reference, job, directory):
    argv = job['server_command']
    require(job['mode'] == 'equal' and argv.count('--host-stage-output') == 1 and
            argv[-2:] == ['--host-stage-output', str(directory / 'host-stages.json')],
            'only the exact appended stage output is permitted')
    normalized = copy.deepcopy(job)
    normalized['server_command'] = argv[:-2]
    timing.validate_job(selected, reference, normalized)


def canonical_job(directory, job):
    """Preserve the raw helper receipt without publishing its ordinary-timing flag."""
    raw = directory / 'attempt.json'
    raw_sha = digest(raw)
    raw.rename(directory / 'helper-attempt.json')
    result = copy.deepcopy(job)
    result.update(FLAGS, raw_helper_receipt=dict(path='helper-attempt.json', sha256=raw_sha,
        warning='Historical helper timing flags do not qualify this instrumented run.'))
    save(raw, result)
    return result


def worker(attempt, mode):
    report, guard = dict(status='failed', mode=mode, **FLAGS), None
    try:
        plan = read(attempt / 'plan.json')
        numerical.authorization(plan, 800)
        reviewed(plan)
        require(mode == 'equal' and plan['variant'] == ORDER[plan['run_index']], 'worker order differs')
        qualified, reference, _, _, _ = references(plan)
        selected, common, h, sustained, measurement, resources = timing.helpers(qualified, plan['variant'])
        require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean worker loader')
        guard = pair.PairGuard(attempt, mode)
        manifest = read(Path(read(Path(plan['ordinary']['attempt']) / 'plan.json')['numerical_reference'])
                        / 'manifest.json')
        h.startup_reference, h.export = common.application_references(h, manifest)
        require(h.query('gpu', h.training.GPU_FIELDS) == plan['gpu'], 'GPU/driver changed')
        report['resources_before'] = resources.resource_snapshot()
        require(report['resources_before']['available_device_sms'] == 80, 'whole device unavailable')
        directory = attempt / mode
        with timing.guard_hooks(common, guard), stage_option(sustained, directory):
            job = sustained.run_control(h, measurement, directory, mode,
                h.training.clean_environment(plan['gpu'].split(',')[0].strip()), timing.POINT)
            report['job'] = canonical_job(directory, job)
            job = report['job']
            capture.audit_application(sustained, measurement, h, directory, job, timing.POINT)
            validate_job(selected, reference, job, directory)
        output = directory / 'host-stages.json'
        require(output.resolve(strict=True) == output and output.stat().st_size <= 512 * 1024 ** 2,
                'bounded canonical stage output required')
        document = read(output)
        begin = job['schedule']['begin_ns'] + 5_000_000_000
        report['service_stages'] = stage_parser.stages(document, job, begin, begin + 5_000_000_000)
        report['stage_output'] = dict(path=str(output), sha256=digest(output),
                                     clock_reads=document['clock_reads'])
        report['counts'] = timing.counts(job)
        report['resources_after'] = resources.resource_snapshot()
        require(report['resources_before'] == report['resources_after'], 'resource policy changed')
        guard.check(force=True)
        report['status'] = 'passed_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        if guard is not None:
            report['resources'] = guard.receipt
        save(attempt / (mode + '-worker.json'), report)
    return int(report['status'] != 'passed_requires_review')


def groups(stages, completions):
    rows = {row['stage']: row for row in stages['stages']}
    require(set(rows) == {name for names in GROUPS.values() for name in names}, 'exact disjoint groups required')
    cpu, duration = stages['exclusive_observed_cpu_ns'], stages['requested_duration_ns']
    steps = rows['step']['calls']
    result = []
    for group, names in GROUPS.items():
        total = sum(rows[name]['exclusive_cpu_ns'] for name in names)
        result.append(dict(group=group, stages=names, exclusive_cpu_ns=total,
            cpu_window_fraction=total / duration, observed_cpu_fraction=total / cpu if cpu else None,
            cpu_ns_per_step=total / steps if steps else None,
            cpu_ns_per_completion={role: total / count if count else None for role, count in completions.items()},
            calls_by_stage={name: rows[name]['calls'] for name in names}))
    return result


def decision(controls, ordinary):
    require([row['variant'] for row in controls] == ORDER, 'exact diagnostic pair required')
    checks = []
    for row in controls:
        baseline = [entry['worker']['counts'] for entry in ordinary['jobs']
                    if entry['mode'] == 'equal' and entry['variant'] == row['variant']]
        require(len(baseline) == 2, 'both accepted ordinary repetitions required')
        for role in ('inference', 'training'):
            actual, minimum = row['counts'][role], min(item[role] for item in baseline)
            require(type(actual) is int and actual >= 0 and type(minimum) is int and minimum > 0,
                    'valid actual and baseline completion counts required')
            checks.append(dict(variant=row['variant'], role=role, actual=actual,
                ordinary_minimum=minimum, ratio=actual / minimum, passed=actual > 0 and actual * 5 >= minimum * 4))
    coverage = all(row['service_stages']['attribution_coverage_gate'] for row in controls)
    eligible = coverage and all(row['passed'] for row in checks)
    selected = [row['group'] for row in controls[1]['groups'] if
                row['observed_cpu_fraction'] is not None and row['observed_cpu_fraction'] >= .6
                and row['cpu_window_fraction'] >= .2]
    return dict(eligible=eligible, coverage_passed=coverage, perturbation_checks=checks,
        verdict=('select_read_only_source_boundary' if len(selected) == 1 else 'mixed_stop')
            if eligible else 'coverage_or_perturbation_inconclusive',
        selected_group=selected[0] if eligible and len(selected) == 1 else None, **FLAGS)


def collect(plan_path):
    plan = read(plan_path)
    numerical.authorization(plan, 2300)
    reviewed(plan)
    print('Checking accepted numerical and ordinary inputs before the one-shot batch.', flush=True)
    qualified, reference, original, paths, _ = references(plan, artifacts=True)
    require(not os.environ.get('LD_PRELOAD') and not os.environ.get('LD_AUDIT'), 'clean collector loader')
    require(shutil.disk_usage(PROJECT_ROOT).free >= pair.TOTAL_LIMIT + pair.DISK_FLOOR,
            'output plus disk reserve required')
    _, common, h, _, _, _ = timing.helpers(qualified, 'o0')
    verify_runtime_inputs(original, h.training.manifest)
    gpu = h.query('gpu', h.training.GPU_FIELDS)
    require(gpu == reference['gpu_identity_after'], 'accepted GPU changed')
    with plan_path.with_name('launch.json').open('x') as stream:
        json.dump(dict(entry=str(ENTRY), plan_sha256=digest(plan_path)), stream)
    attempt = create_attempt(resolve_artifact_root('bert-host-pair-stages', category='profiles'))
    print('Host-stage evidence: ' + str(attempt), flush=True)
    save(attempt / 'plan.json', plan)
    report = dict(status='started', jobs=[], **FLAGS)
    paths.update({ENTRY, plan_path, Path(plan['launch_review'])})
    paths.update(Path(module.__file__).resolve(strict=True) for module in tuple(sys.modules.values())
                 if getattr(module, '__file__', None) and Path(module.__file__).is_relative_to(PROJECT_ROOT))
    children = []
    for index, variant in enumerate(ORDER):
        child = create_attempt(resolve_artifact_root('host-stage-run', category='profiles',
                    output_root=attempt / ('run-' + str(index + 1))))
        save(child / 'plan.json', dict(plan, run_index=index, variant=variant, gpu=gpu,
                                     aggregate_attempt=str(attempt)))
        paths.add(child / 'plan.json')
        children.append(child)
    print('Freezing input identities, then waiting for 60 sampled idle seconds.', flush=True)
    before = h.training.manifest(paths)
    save(attempt / 'inputs-before.json', before)
    try:
        report['idle_window'] = wait_for_idle(attempt)
        require(report['idle_window']['achieved'], 'idle window unavailable')
        for index, (variant, child) in enumerate(zip(ORDER, children)):
            numerical.authorization(plan, 840)
            print('Checking frozen inputs for ' + variant + '; GPU launch follows.', flush=True)
            require(h.training.manifest(paths) == before, 'frozen inputs changed before launch')
            h.training.require_idle()
            print('Launching ' + variant + ' equal, capped at 800 seconds.', flush=True)
            unit, result = candidate.launch_unit(child, 'equal', worker_script=ENTRY)
            report['jobs'].append(dict(variant=variant, mode='equal', attempt=str(child), unit=unit, worker=result))
            pair.aggregate_budget(attempt)
            save(attempt / 'report.json', report)
            print(json.dumps(dict(variant=variant, counts=result['counts'])), flush=True)
        report['status'] = 'host_pair_stages_requires_review'
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        print('Application phase ended; checking final input identities and cleanup.', flush=True)
        try:
            after = h.training.manifest(paths)
            save(attempt / 'inputs-after.json', after)
            report['input_identity_unchanged'] = before == after
            require(before == after and h.query('gpu', h.training.GPU_FIELDS) == gpu, 'final input/GPU changed')
            report['total_output_bytes'] = pair.aggregate_budget(attempt)
        except BaseException as error:
            report.update(status='failed', identity_error=f'{type(error).__name__}: {error}')
        save(attempt / 'report.json', report)
    print(json.dumps(dict(status=report['status'], error=report.get('error'), attempt=str(attempt))), flush=True)
    return int(report['status'] == 'failed')


def analyze(attempt):
    plan, report = read(attempt / 'plan.json'), read(attempt / 'report.json')
    reviewed(plan)
    require(report['status'] == 'host_pair_stages_requires_review' and
            report['input_identity_unchanged'] is True and len(report['jobs']) == 2, 'complete stage pair required')
    qualified, reference, _, _, ordinary = references(plan, artifacts=True)
    _, common, h, sustained, measurement, _ = timing.helpers(qualified, 'o0')
    old = read(Path(plan['ordinary']['attempt']) / 'plan.json')
    h.startup_reference, h.export = common.application_references(h, read(Path(old['numerical_reference']) / 'manifest.json'))
    frozen = read(attempt / 'inputs-before.json')
    require(frozen == read(attempt / 'inputs-after.json') == h.training.manifest({Path(name) for name in frozen}),
            'frozen inputs changed')
    controls = []
    for index, entry in enumerate(report['jobs']):
        require(entry['variant'] == ORDER[index] and entry['mode'] == 'equal', 'actual order differs')
        child, worker, unit = Path(entry['attempt']), entry['worker'], entry['unit']
        require(child.resolve(strict=True) == child and child.parent.parent == attempt and
                worker == read(child / 'equal-worker.json') and unit == read(child / 'equal-unit.json') and
                worker['status'] == 'passed_requires_review' and unit['returncode'] == 0 and
                unit['unit_removed'] is True, 'normal raw worker/unit required')
        selected = pair.selected_plan(qualified, entry['variant'])
        h.training.SERVER, h.training.PRELOAD = Path(selected['server']), Path(selected['frontend'])
        directory, job = child / 'equal', worker['job']
        require(digest(directory / 'helper-attempt.json') == job['raw_helper_receipt']['sha256'] and
                worker['resources_before'] == worker['resources_after'], 'raw helper or resources changed')
        expected = dict(read(directory / 'helper-attempt.json'), **FLAGS)
        expected['raw_helper_receipt'] = job['raw_helper_receipt']
        require(job == expected, 'canonical receipt differs beyond diagnostic provenance')
        with stage_option(sustained, directory):
            capture.audit_application(sustained, measurement, h, directory, job, timing.POINT)
            validate_job(selected, reference, job, directory)
        output = directory / 'host-stages.json'
        require(output.stat().st_size <= 512 * 1024 ** 2 and str(output) == worker['stage_output']['path']
                and digest(output) == worker['stage_output']['sha256'], 'stage payload changed')
        begin = job['schedule']['begin_ns'] + 5_000_000_000
        end = begin + 5_000_000_000
        parsed = stage_parser.stages(read(output), job, begin, end)
        require(parsed == worker['service_stages'] and timing.counts(job) == worker['counts'], 'reparse differs')
        completions = {}
        for role in ('inference', 'training'):
            result = read(directory / role / 'result.json')
            progress = measurement.verify_progress(directory, role, job['ready'][role], result)
            completions[role] = sum(begin < row['end_ns'] <= end for row in progress['records'])
        controls.append(dict(variant=entry['variant'], counts=worker['counts'], window_completions=completions,
            service_stages=parsed, groups=groups(parsed, completions), stage_output=worker['stage_output']))
    return dict(status='host_pair_stage_analysis_requires_review', controls=controls,
        decision=decision(controls, ordinary), normalization_scope='Service CPU per role completion is not role attribution.',
        attribution_scope='Both frontend and service change between variants; no isolated compiler causality.', **FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument('--collect', type=Path)
    options.add_argument('--analyze', type=Path)
    options.add_argument('--worker-attempt', type=Path)
    parser.add_argument('--worker-mode', choices=['equal'])
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode == 'equal', 'equal worker mode required')
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if args.analyze:
        result = analyze(args.analyze.resolve(strict=True))
        with (args.analyze / 'analysis.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(result['decision']), flush=True)
        return 0
    return collect(args.collect.resolve(strict=True))


if __name__ == '__main__':
    raise SystemExit(main())
