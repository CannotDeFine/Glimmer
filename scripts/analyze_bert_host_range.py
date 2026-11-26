"""Measure local host-range validation within correlated host supply gaps."""

from collections import Counter, defaultdict

import analyze_bert_frontend_wait as waits

require = waits.require
STAGES = ['open', 'read', 'parse', 'close_other']
DETAIL_FIELDS = ['serial', 'begin', 'end', 'cpu_begin', 'cpu_end',
                 *(name + '_wall' for name in STAGES), *(name + '_cpu' for name in STAGES),
                 'reads', 'bytes', 'closing']
SEEK_FIELDS = DETAIL_FIELDS + ['hint_attempted', 'hint_hit']


def details(base, phases, *, enabled=True):
    """Validate every detail against its exact coarse helper, including failures."""
    fields = SEEK_FIELDS if base['schema'] == 6 else DETAIL_FIELDS
    require(base['schema'] == phases['schema'] and base['schema'] in (5, 6)
            and all(base[key] == phases[key] for key in (*waits.COMMON, 'host_detail', 'host_count'))
            and base['host_detail'] is enabled and phases['host_fields'] == fields
            and phases['host_count'] == len(phases['host_details'])
            and 0 <= phases['host_count'] <= phases['host_capacity'] <= 131072,
            'complete matching schema5/6 detail publications required')
    helpers = {serial: (kind, left, right) for serial, kind, left, right in phases['phase_events'] if kind >= 5}
    require(len(helpers) == sum(kind >= 5 for _, kind, _, _ in phases['phase_events']),
            'duplicate helper serial')
    output = {}
    for values in phases['host_details']:
        require(len(values) == len(fields) and all(type(v) is int and v >= 0 for v in values),
                'invalid host detail encoding')
        row = dict(zip(fields, values))
        serial = row['serial']
        require(serial in helpers and serial not in output, 'unmatched or duplicate detail serial')
        kind, left, right = helpers[serial]
        require(5 <= kind <= 8 and 0 < left <= row['begin'] <= row['end'] <= right
                and 0 < row['cpu_begin'] <= row['cpu_end']
                and sum(row[name + '_wall'] for name in STAGES) == row['end'] - row['begin']
                and sum(row[name + '_cpu'] for name in STAGES) == row['cpu_end'] - row['cpu_begin']
                and row['closing'] in (0, 1)
                and (row['reads'] > 0 or row['bytes'] == 0),
                'detail clock/partition/correspondence mismatch')
        if kind % 2:
            require(row['closing'] == 1 and row['reads'] > 0 and row['bytes'] > 0,
                    'successful getter must observe the real maps read and cleanup')
        if base['schema'] == 6:
            require(0 <= row['hint_hit'] <= row['hint_attempted'] <= 1
                    and (not row['hint_attempted'] or row['reads'] >= 1)
                    and (not row['hint_hit'] or (kind == 5 and row['reads'] == 1
                                               and 1 < row['bytes'] <= 4096)),
                    'hint flags require one actual successful bounded getter read')
        output[serial] = dict(row, coarse_begin=left, coarse_end=right, success=kind % 2 == 1)
    require(set(output) == (set(helpers) if enabled else set())
            and (enabled or phases['host_capacity'] == 0), 'complete bidirectional detail/helper match required')
    return output


def stage_summary(rows, main_serials, begin, end):
    rows = [row for serial, row in rows.items() if serial in main_serials]
    clipped = duration(union(((row['coarse_begin'], row['coarse_end']) for row in rows), begin, end))
    selected = [row for row in rows if begin <= row['coarse_begin'] <= row['coarse_end'] <= end]
    contained = sum(row['coarse_end'] - row['coarse_begin'] for row in selected)
    wall = {name: sum(row[name + '_wall'] for row in selected) for name in STAGES}
    cpu = {name: sum(row[name + '_cpu'] for row in selected) for name in STAGES}
    elapsed = sum(wall.values())
    require(elapsed == sum(row['end'] - row['begin'] for row in selected), 'incomplete stage partition')
    coverage = contained / clipped if clipped else 0
    dominant = [name for name in STAGES if elapsed and wall[name] >= .6 * elapsed]
    return dict(count=len(selected), clipped_all_helper_ns=clipped, contained_helper_ns=contained,
        cohort_coverage=coverage, trace_elapsed_ns=elapsed, wall_ns=wall, thread_cpu_ns=cpu,
        read_calls=sum(row['reads'] for row in selected), read_bytes=sum(row['bytes'] for row in selected),
        dominant_stage=dominant[0] if coverage >= .95 and len(dominant) == 1 else 'inconclusive',
        interpretation='Exclusive complete-helper stages; read duration is syscall-bound, not a kernel-stack profile. No phase fraction is attributed to the supply-gap intersection.')


def union(spans, begin, end):
    result = []
    for left, right in sorted(spans):
        left, right = max(begin, left), min(end, right)
        if left >= right:
            continue
        if result and left <= result[-1][1]:
            result[-1] = result[-1][0], max(right, result[-1][1])
        else:
            result.append((left, right))
    return result


def intersection(left, right):
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        start, stop = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if start < stop:
            result.append((start, stop))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def duration(spans):
    return sum(right - left for left, right in spans)


def helpers(base, phases, identity, begin, end):
    """Require every getter's exact boundary and successful local validation."""
    boundaries, calls, _ = waits.lifecycle.frontend(base, identity, begin, end)
    require(base['schema'] in (3, 5, 6), 'host-range analysis requires schema3/5/6')
    rows = {}
    outcomes = Counter()
    for serial, kind, left, right in phases['phase_events']:
        if kind < 5:
            continue
        require(kind < 9 and serial in boundaries and serial not in rows,
                'one exact host-range span per API boundary required')
        boundary = boundaries[serial]
        require(boundary['locked'] <= left <= right <= boundary['finish'],
                'host-range span escaped its boundary')
        outcomes[waits.HOST_NAMES[kind - 5]] += 1
        rows[serial] = dict(caller='get_blas_value' if kind < 7 else 'cublasGetStream',
                            success=kind % 2 == 1, begin=left, end=right,
                            thread=boundary['thread'])
    require(all(row['success'] for row in rows.values()),
            'all complete helper checks must succeed before dominance interpretation')
    children = defaultdict(list)
    getters = set()
    for call in calls:
        if call['opcode'] == 29:
            getters.add(call['boundary'])
        if call['boundary'] in rows:
            children[call['boundary']].append(call)
    require(getters == set(rows), 'getter/helper coverage must be complete in both directions')
    for serial, row in rows.items():
        nested = children[serial]
        require(len(nested) == 1 and nested[0]['opcode'] == 29 and nested[0]['wire'] > 0
                and nested[0]['flags'] == 3 and row['end'] <= nested[0]['enter']
                and row['thread'] == nested[0]['thread'],
                'helper must precede its exact same-thread getter RPC')
    return list(rows.values()), dict(outcomes)


def analyze(document, bases, phases, job, begin, end, request, wait_result):
    require(end - begin == 5_000_000_000, 'predeclared fixed5s window required')
    states = waits.lifecycle.service(document, begin, end)
    output = []
    for role, identity in job['application_identities'].items():
        wires, events = waits.ledger(bases[role], phases[role], identity, begin, end)
        spans, outcomes = helpers(bases[role], phases[role], identity, begin, end)
        require(role != 'inference' or spans,
                'inference decision role requires observed helper spans')
        prior = next(row for row in wait_result['roles'] if row['role'] == role)
        coarse = next(row for row in wait_result['coarse_lifecycle']['roles'] if row['role'] == role)
        require(coarse['compute_correlation_fraction'] == coarse['service_lifecycle_coverage']
                == coarse['frontend_recorder_window_coverage'] == 1.0,
                'complete exact frontend/service coverage required')
        owned = [(slot, state) for slot, state in zip(document['slots'], states)
                 if tuple(slot[1:]) == (identity['pid'], identity['start'])]
        compute = [(slot, state) for slot, state in owned if state['kind'] == 0]
        require(len(compute) == 1, 'one authenticated compute connection required')
        service = {(slot[0], wire): call for slot, state in owned
                   for wire, call in state['calls'].items()}
        mutex, threads = [], set()
        for key, row in wires.items():
            if row['opcode'] not in (137, 138) or row['boundary']:
                continue
            if row['enter'] >= end or row['finish'] < begin:
                continue
            require(key in service, 'background RPC lacks exact service owner')
            call = service[key]
            require(row['opcode'] == call['opcode'] and row['send_begin'] <= call['receive']
                    <= call['reply'] <= row['reply'], 'background same-wire order differs')
            threads.add(row['thread'])
            mutex.extend((max(left, call['reply']), min(right, row['reply']))
                         for kind, left, right in events[row['serial']] if kind == 3)
        require(len(threads) == 1, 'one observed background progress thread required')
        mutex = union(mutex, begin, end)
        require(duration(mutex) == prior['phase_ns']['mutex_reacquire'],
                'mutex union differs from same-thread latency decomposition')
        gaps = union(compute[0][1]['gaps'], begin, end)
        require(duration(gaps) == coarse['service_supply_gap_ns'], 'supply gap join differs')
        selected = intersection(mutex, gaps)
        denominator = duration(selected)
        require(denominator > 0, 'no host critical-section denominator observed')
        callers = []
        for caller in ('get_blas_value', 'cublasGetStream'):
            intervals = union(((row['begin'], row['end']) for row in spans
                               if row['caller'] == caller and row['thread'] == identity['pid']), begin, end)
            overlap = duration(intersection(selected, intervals))
            callers.append(dict(caller=caller, window_union_ns=duration(intervals),
                                intersection_ns=overlap, fraction=overlap / denominator))
        all_helpers = union(((row['begin'], row['end']) for row in spans
                             if row['thread'] == identity['pid']), begin, end)
        overlap = duration(intersection(selected, all_helpers))
        require(overlap == sum(row['intersection_ns'] for row in callers),
                'helper caller attribution duplicates wall time')
        fraction = overlap / denominator
        decision = ('dominant_host_range_validation' if fraction >= .6 else
                    'host_range_validation_not_dominant' if fraction < .2 else 'inconclusive')
        output.append(dict(role=role, application_identity=identity, helper_outcomes=outcomes,
            mutex_reacquire_ns=duration(mutex), supply_gap_ns=duration(gaps),
            denominator_ns=denominator, helper_intersection_ns=overlap, fraction=fraction,
            callers=callers, decision=decision))
        if bases[role]['schema'] in (5, 6):
            boundaries, _, _ = waits.lifecycle.frontend(bases[role], identity, begin, end)
            summary = stage_summary(details(bases[role], phases[role]),
                {serial for serial, row in boundaries.items() if row['thread'] == identity['pid']}, begin, end)
            output[-1]['stages'] = summary
            output[-1]['stage_decision'] = (summary['dominant_stage']
                if decision == 'dominant_host_range_validation' else 'inconclusive_coarse_gate')
    inference = [row for row in output if row['role'] == 'inference']
    require(len(inference) == 1, 'one inference decision role required')
    result = dict(diagnostic_only=True, begin_ns=begin, end_ns=end, roles=output,
        decision_role='inference', decision=inference[0]['decision'],
        denominator='Fresh same-window inference background mutex-reacquire union intersect zero-wire/zero-unissued supply gaps.',
        scope='Main-thread helper spans under frontend exclusion; not GPU idle, full-device coverage or whole-application causality.',
        dominant_threshold=.6, not_dominant_threshold=.2, priority_qualified=False,
        full_device_qualified=False, gpu_gap_causality_qualified=False, performance_qualified=False)
    if 'stages' in inference[0]:
        result['coarse_helper_decision'] = result['decision']
        result['stage_decision'] = inference[0]['stage_decision']
        result['decision'] = result['stage_decision']
    return result
