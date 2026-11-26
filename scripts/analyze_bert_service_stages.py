#!/usr/bin/env python3
"""Analyze disjoint completed service-stage spans inside one fixed BERT window.

End-bucket attribution omits some boundary work. Fractions use the entire
requested window as denominator, never renormalize the selected stages to 100%.
Thus a stage's wall/window fraction is a conservative observed coverage, not
an exact owner-CPU fraction or a GPU-gap measurement.
"""

import argparse
import json
from pathlib import Path

import run_bert_stage_diagnostic as collector
from run_rnn_v8_validation import digest, save

require = collector.require
STAGES = ['step', 'transport', 'receive', 'observe', 'advance', 'plan_receive',
          'plan_slot', 'plan_issue', 'plan_other', 'observations', 'snapshot',
          'pending', 'policy', 'issue_attempt', 'issue_native']
COUNTERS = ['calls', 'wall_ns', 'cpu_ns', 'exclusive_wall_ns', 'exclusive_cpu_ns']
QUEUES = ['samples', 'requests_sum', 'queued_sum', 'issuing_sum', 'paused_sum',
          'awaiting_stop_sum', 'ready_samples', 'physical_samples', 'low_epochs_sum',
          'preparation_samples', 'requests_max', 'queued_max']


def natural(value):
    return type(value) is int and value >= 0


def stages(document, job, begin, end):
    require(document['schema'] == 1 and document['valid'] is True
            and document['diagnostic_only'] is True and document['idle_enabled'] is False
            and document['pid'] == job['server_identity']['pid']
            and document['attribution'] == 'span_end_bucket'
            and document['stages'] == STAGES and document['counter_fields'] == COUNTERS
            and document['queue_fields'] == QUEUES, 'exact valid stage schema required')
    width, origin = document['bucket_ns'], document['origin_ns']
    require(width == 50_000_000 and natural(origin) and origin % width == 0
            and 0 < len(document['buckets']) <= 16384 and len(document['slots']) <= 128,
            'fixed capacity and monotonic bucket geometry required')
    owners = {(row['pid'], row['start']) for row in job['application_identities'].values()}
    roles = {(row['pid'], row['start']): role for role, row in job['application_identities'].items()}
    connections = set()
    for slot in document['slots']:
        require(len(slot) == 3 and all(natural(value) and value > 0 for value in slot)
                and tuple(slot[1:]) in owners and slot[0] not in connections,
                'authenticated unique client connection required')
        connections.add(slot[0])
    sums = [[0] * 5 for _ in STAGES]
    queue_sums, outcomes, selected, span_bounds = {}, [0] * 4, [], []
    for index, bucket in enumerate(document['buckets']):
        stamp, first, last = (bucket[name] for name in
            ('begin_ns', 'minimum_span_start_ns', 'maximum_span_end_ns'))
        require(stamp == origin + index * width and all(natural(v) for v in (first, last))
                and len(bucket['stages']) == len(STAGES)
                and len(bucket['issue_outcomes']) == 4
                and all(natural(v) for v in bucket['issue_outcomes']), 'invalid stage bucket')
        require((first == last == 0) or 0 < first <= last and stamp <= last < stamp + width,
                'span endpoint outside its end bucket')
        for values in bucket['stages']:
            require(len(values) == 5 and all(natural(v) for v in values)
                    and values[3] <= values[1] and values[4] <= values[2], 'invalid stage counters')
        seen = set()
        for values in bucket['queues']:
            require(len(values) == 13 and all(natural(v) for v in values)
                    and values[0] < len(document['slots']) and values[0] not in seen
                    and values[1] > 0 and values[7] <= values[1]
                    and values[8] <= values[1] and values[10] <= values[1], 'invalid copied queue counters')
            seen.add(values[0])
        if not (begin <= stamp and stamp + width <= end and begin <= first <= last <= end):
            continue
        selected.append(stamp)
        span_bounds.append((first, last))
        for total, values in zip(sums, bucket['stages']):
            for i, value in enumerate(values):
                total[i] += value
        for values in bucket['queues']:
            total = queue_sums.setdefault(values[0], [0] * 12)
            for i, value in enumerate(values[1:]):
                total[i] = max(total[i], value) if i >= 10 else total[i] + value
        outcomes = [a + b for a, b in zip(outcomes, bucket['issue_outcomes'])]
    require(len(selected) >= 90, 'less than90 whole valid50ms buckets inside requested5s window')
    duration = end - begin
    rows = []
    for name, values in zip(STAGES, sums):
        row = dict(stage=name, **dict(zip(COUNTERS, values)))
        row.update(exclusive_wall_window_fraction=values[3] / duration,
                   exclusive_cpu_window_fraction=values[4] / duration,
                   inclusive_wall_us_per_call=values[1] / values[0] / 1000 if values[0] else None)
        rows.append(row)
    # Exclusive spans are disjoint, including nested parents. Selection cannot
    # recover work from open/cross-boundary parents or gaps between service steps.
    require(sum(row['exclusive_wall_ns'] for row in rows) <= duration,
            'exclusive spans overcount the requested wall window')
    planning = {'plan_receive', 'plan_slot', 'plan_issue', 'plan_other',
                'observations', 'snapshot', 'pending', 'policy'}
    plan_wall = sum(row['exclusive_wall_ns'] for row in rows if row['stage'] in planning)
    plan_cpu = sum(row['exclusive_cpu_ns'] for row in rows if row['stage'] in planning)
    root_wall = sums[0][1]
    effective_begin = min(first for first, _ in span_bounds)
    effective_end = max(last for _, last in span_bounds)
    require(0 < root_wall <= effective_end - effective_begin <= duration,
            'root-step coverage exceeds the accepted span envelope')
    coverage = root_wall / duration
    return dict(requested_begin_ns=begin, requested_end_ns=end, requested_duration_ns=duration,
        selected_buckets=len(selected), selected_bucket_capacity_ns=len(selected) * width,
        first_selected_bucket_ns=selected[0], last_selected_bucket_ns=selected[-1],
        boundary_work_omitted=True, denominator='entire requested window; no renormalization',
        accepted_span_envelope_begin_ns=effective_begin, accepted_span_envelope_end_ns=effective_end,
        accepted_span_envelope_duration_ns=effective_end - effective_begin,
        root_step_wall_ns=root_wall, root_step_requested_window_coverage=coverage,
        root_step_span_envelope_coverage=root_wall / (effective_end - effective_begin),
        minimum_root_coverage=.95, attribution_coverage_gate=coverage >= .95,
        exclusive_observed_wall_ns=sum(row['exclusive_wall_ns'] for row in rows),
        exclusive_observed_cpu_ns=sum(row['exclusive_cpu_ns'] for row in rows),
        stages=sorted(rows, key=lambda row: row['exclusive_wall_ns'], reverse=True),
        planning_group=dict(stages=sorted(planning), exclusive_wall_ns=plan_wall,
            exclusive_cpu_ns=plan_cpu, wall_window_fraction=plan_wall / duration,
            cpu_window_fraction=plan_cpu / duration,
            conservative_60percent_gate=coverage >= .95 and plan_wall >= .6 * duration),
        issue_outcomes=dict(zip(('issued', 'idle', 'not_ready', 'other_error'), outcomes)),
        queues=[dict(slot=document['slots'][index], role=roles[tuple(document['slots'][index][1:])],
                     **dict(zip(QUEUES, values)))
                for index, values in sorted(queue_sums.items())],
        queue_scope='observation-weighted copied snapshots, not wall-time queue averages',
        issue_scope='standard issue_scheduled_command path only; excludes pair/NCCL paths')


def analyze(attempt):
    report = collector.read(attempt / 'report.json')
    plan = collector.read(attempt / 'plan.json')
    frozen = collector.read(attempt / 'inputs-before.json')
    require(frozen == collector.read(attempt / 'inputs-after.json'), 'full input manifests differ')
    for name in ('analyzer', 'stage_plan'):
        bound = plan[name]
        require(digest(Path(bound['path'])) == bound['sha256'] == frozen[bound['path']]['sha256'],
                'pre-registered analyzer or stage plan changed')
    scope = plan.get('qualification_scope')
    if scope is not None:
        require(digest(Path(scope['path'])) == scope['sha256'] == frozen[scope['path']]['sha256']
                and scope['modes'] == ['native', 'equal']
                and scope['priority_qualified'] is False and scope['triplet_qualified'] is False,
                'explicit numerical qualification scope changed')
        reference = Path(plan['reference'])
        collector.selected_checkpoint_jobs(reference, collector.read(reference / 'report.json'),
                                           Path(scope['path']))
    require(report['status'] == 'service_stage_pair_requires_independent_review'
            and report['input_identity_unchanged'] is True, 'complete diagnostic pair required')
    common, h, sustained = collector.helpers()
    require(h.training.manifest({Path(name) for name in frozen}) == frozen,
            'frozen source, qualification or raw evidence bytes changed')
    measurement = sustained.load_measurement()
    controls = []
    for entry in report['jobs']:
        mode, worker, unit = entry['mode'], entry['worker'], entry['unit']
        require(frozen == collector.read(attempt / (mode + '-inputs-after.json')),
                'per-unit full input manifest differs')
        require(worker == collector.read(attempt / (mode + '-worker.json'))
                and unit == collector.read(attempt / (mode + '-unit.json'))
                and worker['status'] == 'passed_requires_review'
                and unit['returncode'] == 0 and unit['unit_removed'] is True, 'normal raw worker/unit required')
        directory, job = attempt / mode, worker['job']
        raw = collector.read(directory / 'helper-attempt.json')
        collector.capture.normal_application(raw)
        require(job == collector.read(directory / 'attempt.json')
                and digest(directory / 'helper-attempt.json') == job['raw_helper_receipt']['sha256'],
                'canonical/helper receipt differs')
        for name, checksum in raw['artifact_identities'].items():
            require((directory / name).resolve(strict=True).is_relative_to(directory)
                    and digest(directory / name) == checksum, 'raw application evidence changed')
        begin = raw['schedule']['begin_ns'] + 5_000_000_000
        end = begin + 5_000_000_000
        progress = measurement.verify_progress(directory, 'inference', raw['ready']['inference'],
                                              collector.read(directory / 'inference/result.json'))
        snapshots = [measurement.request_snapshot(progress['records'], progress['inflight'],
            raw['schedule']['begin_ns'], raw['schedule']['trace'], stamp) for stamp in (begin, end)]
        control = dict(mode=mode, application_window=dict(begin_ns=begin, end_ns=end,
            begin=snapshots[0], end=snapshots[1]), completion=raw['completion'])
        if mode == 'equal':
            path = directory / 'host-stages.json'
            require(path.stat().st_size <= 512 * 1024 ** 2
                    and digest(path) == worker['stage_output']['sha256'], 'bounded stage output identity changed')
            control['service_stages'] = stages(json.loads(path.read_text()), raw, begin, end)
        controls.append(control)
    require([row['mode'] for row in controls] == ['native', 'equal'], 'one matched native/equal pair required')
    return dict(status='stage_analysis_requires_independent_review', controls=controls,
                qualification_scope=scope, **collector.FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    output = attempt / 'stage-analysis.json'
    require(not output.exists(), 'fresh analysis output required; never overwrite evidence')
    result = analyze(attempt)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
