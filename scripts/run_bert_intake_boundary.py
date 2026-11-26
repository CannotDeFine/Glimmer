#!/usr/bin/env python3
"""Qualify and collect a plan-bound native/equal intake diagnostic pair."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys

import run_bert_frontend_lifecycle as lifecycle
from analyze_bert_host_range import union, duration
from artifact_paths import PROJECT_ROOT
from run_rnn_v8_validation import digest

require = lifecycle.require
ENTRY = Path(__file__).resolve()
original_worker = lifecycle.worker
original_analyze = lifecycle.analyze
POST_PHASES = ('pre_frame', 'frame_to_owner', 'owner_to_retained')


def split_outbound(sent, returned, frame, owner, retained):
    require(0 < sent <= frame <= owner <= retained and sent <= returned,
            'invalid same-wire intake ordering')
    post = min(returned, retained)
    points = (sent, post, max(post, frame), max(post, owner), retained)
    return [(left, right, phase) for left, right, phase in
            zip(points, points[1:], ('during_submit', *POST_PHASES)) if left < right]


def partition(gaps, intervals):
    """Half-open occupancy by exact active multisets, including multiplicity."""
    events = defaultdict(list)
    for left, right in gaps:
        require(left < right, 'positive gap required')
        events[left].append((None, 1))
        events[right].append((None, -1))
    for left, right, key in intervals:
        require(left < right, 'positive interval required')
        events[left].append((key, 1))
        events[right].append((key, -1))
    active, totals = Counter(), Counter()
    inside, previous, uncovered = 0, None, 0
    for stamp in sorted(events):
        if inside and previous is not None:
            if active:
                totals[tuple(sorted(active.items()))] += stamp - previous
            else:
                uncovered += stamp - previous
        changes = Counter(key for key, delta in events[stamp] if delta == 1)
        changes.subtract(key for key, delta in events[stamp] if delta == -1)
        inside += changes.pop(None, 0)
        require(inside in (0, 1), 'overlapping or reversed gaps')
        for key, delta in changes.items():
            active[key] += delta
            require(active[key] >= 0, 'negative active count')
            if not active[key]:
                del active[key]
        previous = stamp
    require(inside == 0 and not active, 'unclosed intervals')
    require(sum(totals.values()) + uncovered == duration(gaps), 'partition lost gap time')
    return totals, uncovered


def summarize(buckets):
    post, during, mixed, exclusive = 0, 0, 0, Counter()
    for key, elapsed in buckets.items():
        phases = {signature[-1] for signature, count in key}
        require(phases <= {'during_submit', *POST_PHASES}, 'unknown phase')
        if phases == {'during_submit'}:
            during += elapsed
        else:
            post += elapsed
            if len(phases) == 1:
                exclusive[next(iter(phases))] += elapsed
            else:
                mixed += elapsed
    require(sum(exclusive.values()) + mixed == post, 'post-submit partition lost time')
    dominant = [phase for phase in POST_PHASES if post and exclusive[phase] >= .6 * post]
    return dict(post_submit_union_ns=post, during_submit_only_ns=during,
        exclusive_post_phase_ns={phase: exclusive[phase] for phase in POST_PHASES},
        mixed_post_overlap_ns=mixed,
        exclusive_fraction_of_post={phase: exclusive[phase] / post if post else None
                                    for phase in POST_PHASES},
        next_boundary=dominant[0] if len(dominant) == 1 else 'inconclusive')


def analyze(attempt, scope):
    original_analyze(attempt, scope)
    coarse = lifecycle.collector.read(attempt / 'frontend-analysis.json')
    request = lifecycle.collector.read(attempt / 'request-analysis.json')
    worker = lifecycle.collector.read(attempt / 'equal-worker.json')
    job = lifecycle.collector.read(attempt / 'equal/helper-attempt.json')
    identity = job['application_identities']['inference']
    inputs = {}

    def bound(path, checksum):
        require(digest(path) == checksum, 'intake analysis input changed')
        inputs[str(path)] = checksum
        return json.loads(path.read_text())

    receipt = worker['frontend_outputs']['inference']
    require(receipt['application_identity'] == identity, 'inference frontend owner differs')
    base = bound(Path(receipt['path']), receipt['sha256'])
    service = bound(Path(worker['stage_output']['path']), worker['stage_output']['sha256'])
    validate_intake(service)
    begin, end = coarse['begin_ns'], coarse['end_ns']
    require(end - begin == 5_000_000_000, 'fixed five-second window required')
    _, _, wires = lifecycle.frontend.frontend(base, identity, begin, end)
    states = lifecycle.frontend.service(service, begin, end)
    owned = [(index, slot, state) for index, (slot, state) in enumerate(zip(service['slots'], states))
             if tuple(slot[1:]) == (identity['pid'], identity['start']) and state['kind'] == 0]
    require(len(owned) == 1, 'one authenticated inference compute slot required')
    index, slot, state = owned[0]
    coverage = [row for row in request['request_lifecycle']['slots']
                if row['role'] == 'inference' and row['compute_connection']]
    require(len(coverage) == 1 and coverage[0]['slot'] == slot and coverage[0]['coverage_gate']
            and .95 <= coverage[0]['lifecycle_coverage'] <= 1, 'complete service window coverage required')
    intakes = {row[4]: row for row in service['intake']['records'] if row[3] == index}
    expected = {wire for wire, call in state['calls'].items()
                if call['receive'] < end and call['reply'] >= begin}
    matched, intervals, original, post_intervals = set(), [], [], []
    for (session, wire), row in wires.items():
        if session != slot[0] or row['enter'] >= end or row['finish'] < begin:
            continue
        require(wire in state['calls'] and wire in intakes, 'missing same-slot wire')
        call = state['calls'][wire]
        frame, owner, retained, _, _, opcode = intakes[wire]
        require(opcode == row['opcode'] == call['opcode'] and retained == call['receive']
                and retained <= call['reply'] <= row['reply'], 'three-way wire identity differs')
        if wire in expected:
            matched.add(wire)
        signature = (opcode, row['thread'], int(row['boundary'] != 0), row['flags'])
        left, right = max(begin, row['send_begin']), min(end, retained)
        if left < right:
            original.append((left, right, signature))
        left = max(begin, min(row['send_end'], retained))
        if left < right:
            post_intervals.append((left, right, signature))
        for left, right, phase in split_outbound(row['send_begin'], row['send_end'], frame, owner, retained):
            left, right = max(begin, left), min(end, right)
            if left < right:
                intervals.append((left, right, (*signature, phase)))
    require(matched == expected and matched, 'complete window compute correlation required')
    gaps = union(state['gaps'], begin, end)
    buckets, uncovered = partition(gaps, intervals)
    parent, original_uncovered = partition(gaps, original)
    collapsed = Counter()
    for key, elapsed in buckets.items():
        unsplit = Counter()
        for signature, count in key:
            unsplit[signature[:-1]] += count
        collapsed[tuple(sorted(unsplit.items()))] += elapsed
    require(collapsed == parent and uncovered == original_uncovered,
            'intake split changed original outbound multiset')
    prior = next(row for row in coarse['roles'] if row['role'] == 'inference')
    require(duration(gaps) == prior['service_supply_gap_ns']
            == coverage[0]['zero_wire_and_zero_unissued_ns']
            and sum(parent.values()) == prior['phase_ns']['outbound_submission'],
            'original supply-gap or outbound union differs')
    result = summarize(buckets)
    post_buckets, _ = partition(gaps, post_intervals)
    require(result['post_submit_union_ns'] == sum(post_buckets.values()),
            'independent post-submit union differs')
    result.update(status='intake_boundary_requires_independent_review',
        begin_ns=begin, end_ns=end, application_identity=identity, compute_slot=slot,
        correlated_compute_wires=len(matched), supply_gap_ns=duration(gaps),
        outbound_ns=sum(parent.values()), other_gap_ns=uncovered,
        buckets=[dict(active_multiset=[dict(signature=key, count=count) for key, count in active],
                      duration_ns=elapsed) for active, elapsed in sorted(buckets.items())],
        raw_inputs=inputs, lifecycle_scope=scope,
        diagnostic_only=True, full_device_qualified=False, performance_qualified=False,
        gpu_gap_causality_qualified=False, priority_qualified=False)
    require(all(digest(Path(path)) == checksum for path, checksum in inputs.items()),
            'raw intake inputs changed during analysis')
    encoded = json.dumps(result, indent=2, allow_nan=False) + '\n'
    require(len(encoded.encode()) < 32 * 1024 ** 2, 'bounded analysis output required')
    with (attempt / 'intake-analysis.json').open('x') as stream:
        stream.write(encoded)
    print(json.dumps({key: result[key] for key in ('next_boundary', 'exclusive_fraction_of_post')}))
    return 0


def validate_intake(document):
    intake = document['intake']
    events = document['lifecycle']['events']
    require(intake['schema'] == 1
            and intake['fields'] == ['frame_received_ns', 'owner_enter_ns', 'retained_ns',
                                     'slot', 'wire', 'opcode']
            and intake['capacity'] == document['lifecycle']['capacity']
            and 0 < len(intake['records']) <= intake['capacity'],
            'complete bounded intake schema required')
    receives = {}
    for stamp, slot, kind, wire, command, detail in events:
        if kind == 0:
            require((slot, wire) not in receives, 'duplicate service receive')
            receives[slot, wire] = (stamp, detail & 0xffffffff)
    require(len(receives) == len(intake['records']), 'one intake per retained receive required')
    for row in intake['records']:
        require(len(row) == 6 and all(type(value) is int and value >= 0 for value in row),
                'invalid intake values')
        frame, owner, retained, slot, wire, opcode = row
        require(0 < frame <= owner <= retained and slot < len(document['slots']) and wire > 0
                and receives.pop((slot, wire), None) == (retained, opcode),
                'intake clock or exact receive correlation differs')
    require(not receives, 'unmatched service receive')
    return len(intake['records'])


def worker(attempt, mode):
    result = original_worker(attempt, mode)
    if result or mode != 'equal':
        return result
    path = attempt / 'equal-worker.json'
    report = lifecycle.collector.read(path)
    try:
        output = attempt / 'equal/host-stages.json'
        require(digest(output) == report['stage_output']['sha256'], 'service output changed')
        report['intake_records'] = validate_intake(json.loads(output.read_text()))
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        result = 1
    lifecycle.candidate.save(path, report)
    return result


def configure_plan(path):
    plan = lifecycle.collector.read(path)
    require(plan['status'] == 'frozen_intake_boundary_application'
            and plan['modes'] == ['native', 'equal']
            and plan['build'] == str(PROJECT_ROOT / 'build/candidate')
            and plan['runtime_environment'] == {'GLIMMER_EXECUTION_HOST_RANGE_SEEK': '1'},
            'explicit frozen native/equal intake plan required')
    seal_path = path.with_name('build-seal.json')
    seal = lifecycle.collector.read(seal_path)
    require(seal['plan_sha256'] == digest(path), 'application plan changed')
    for name, checksum in seal['identities'].items():
        require(digest(Path(name)) == checksum, 'frozen intake input changed: ' + name)
    overhead_path = Path(plan['overhead_report'])
    overhead = lifecycle.collector.read(overhead_path)
    require(digest(overhead_path) == plan['overhead_sha256']
            and overhead['status'] == 'passed_requires_independent_review'
            and set(overhead['medians']) == {'wall', 'cpu'}
            and all(0 < value <= 1.20 for value in overhead['medians'].values()),
            'accepted incremental recorder overhead required')
    lifecycle.ENTRY, lifecycle.PLAN = ENTRY, path
    lifecycle.BUILD = Path(plan['build'])
    lifecycle.SERVER = lifecycle.BUILD / 'bin/glimmer_execution_service'
    lifecycle.PRELOAD = lifecycle.BUILD / 'lib/libglimmer_execution_interceptor.so'
    require(digest(lifecycle.SERVER) == plan['server_sha256']
            and digest(lifecycle.PRELOAD) == plan['frontend_sha256'], 'candidate binary changed')
    lifecycle.OUTPUT_SCHEMA = 6
    lifecycle.RUNTIME_ENV = plan['runtime_environment']
    lifecycle.EXTRA_ENV = {}
    lifecycle.collector.STAGE_PLAN = path
    lifecycle.stage.DEADLINE = datetime.fromisoformat(plan['ordinary_run_deadline_utc'])
    lifecycle.worker = worker
    lifecycle.analyze = analyze
    original_identity = lifecycle.frontend_identity

    def identity():
        result = original_identity()
        result['identities'].update(seal['identities'])
        result['identities'][str(seal_path)] = digest(seal_path)
        return result

    lifecycle.frontend_identity = identity


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--check', action='store_true')
    args, remaining = parser.parse_known_args()
    if args.plan is None:
        option = '--worker-attempt' if '--worker-attempt' in remaining else '--analyze'
        require(option in remaining, 'explicit --plan required')
        attempt = Path(remaining[remaining.index(option) + 1])
        document = attempt / 'plan.json'
        scope = (lifecycle.collector.read(document)['lifecycle_scope'] if document.is_file()
                 else lifecycle.collector.read(attempt / 'numerical-scope.json'))
        args.plan = Path(scope['plan']['path'])
        require(digest(args.plan) == scope['plan']['sha256'], 'worker plan changed')
    configure_plan(args.plan.resolve(strict=True))
    if args.check:
        require(not remaining, 'configuration check is CPU-only')
        print(json.dumps(lifecycle.configure(), indent=2))
        return 0
    sys.argv = [str(ENTRY), *remaining]
    return lifecycle.main()


if __name__ == '__main__':
    raise SystemExit(main())
