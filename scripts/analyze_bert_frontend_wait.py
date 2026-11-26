"""Attribute exact same-wire async reply latency to disjoint CPU wait phases."""

from collections import Counter, defaultdict

import analyze_bert_frontend_lifecycle as lifecycle

require = lifecycle.require
NAMES = ['quick_yield', 'wait_call', 'fallback_sleep', 'mutex_reacquire', 'poll_take']
HOST_NAMES = ['blas_value_range_success', 'blas_value_range_failure',
              'blas_stream_range_success', 'blas_stream_range_failure']
COMMON = ('schema', 'diagnostic_only', 'valid', 'pid', 'begin_ns', 'end_ns',
          'base_count', 'phase_count')


def ledger(base, phases, identity, begin, end):
    """Validate all completed records, including those outside the decision window."""
    boundaries, calls, wires = lifecycle.frontend(base, identity, begin, end)
    require(base['schema'] in (2, 3, 5, 6) and all(base[key] == phases[key] for key in COMMON)
            and base['base_count'] == len(base['records'])
            and phases['phase_count'] == len(phases['phase_events'])
            and phases['phase_fields'] == ['serial', 'phase', 'begin', 'end']
            and phases['phase_names'] == NAMES + (HOST_NAMES if base['schema'] in (3, 5, 6) else [])
            and 0 < len(phases['phase_events']) <= phases['phase_capacity'] <= 8388608,
            'complete matching base/phase publications required')
    target = {row['serial']: row for row in calls if row['wire'] and row['opcode'] in (137, 138)}
    by_serial, last, seen = defaultdict(list), {}, set()
    helpers = []
    for event in phases['phase_events']:
        require(len(event) == 4 and all(type(value) is int and value >= 0 for value in event),
                'invalid phase event encoding')
        serial, kind, left, right = event
        if kind >= len(NAMES):
            require(base['schema'] in (3, 5, 6) and kind < len(NAMES) + len(HOST_NAMES)
                    and serial in boundaries, 'helper phase lacks exact API boundary')
            row = boundaries[serial]
            require(row['locked'] <= left <= right <= row['finish']
                    and serial not in last, 'helper boundary clock/order/uniqueness differs')
            last[serial] = right
            helpers.append((left, right))
            continue
        require(serial in target and kind < len(NAMES), 'phase lacks exact accepted target RPC')
        row = target[serial]
        require(row['send_end'] <= left <= right <= row['reply']
                and left >= last.get(serial, row['send_end']) and tuple(event) not in seen,
                'phase clock/order/uniqueness differs from its RPC')
        last[serial] = right
        seen.add(tuple(event))
        by_serial[serial].append((kind, left, right))
    require(set(by_serial) == set(target)
            and all(any(kind == 4 for kind, _, _ in events) for events in by_serial.values()),
            'every completed target RPC must retain its poll/take phase')
    helpers.sort()
    require(all(first[1] <= second[0] for first, second in zip(helpers, helpers[1:])),
            'host-range spans overlap despite shared exclusion')
    return wires, by_serial


def decomposition(row, reply, phases, begin, end):
    """Clip latency and phase intervals to the same fixed observation window."""
    left, right = max(begin, reply), min(end, row['reply'])
    if left >= right:
        return 0, Counter()
    counts = Counter({name: 0 for name in NAMES})
    for kind, start, stop in phases:
        counts[NAMES[kind]] += max(0, min(right, stop) - max(left, start))
    unresolved = right - left - sum(counts.values())
    require(unresolved >= 0, 'phase attribution duplicates latency')
    counts['unresolved'] = unresolved
    return right - left, counts


def analyze(document, bases, phase_documents, job, begin, end, request_lifecycle):
    coarse = lifecycle.analyze(document, bases, job, begin, end, request_lifecycle)
    states = lifecycle.service(document, begin, end)
    output = []
    for role, identity in job['application_identities'].items():
        wires, events = ledger(bases[role], phase_documents[role], identity, begin, end)
        owned = [(slot, state) for slot, state in zip(document['slots'], states)
                 if tuple(slot[1:]) == (identity['pid'], identity['start'])]
        service_calls = {(slot[0], wire): call for slot, state in owned
                         for wire, call in state['calls'].items()}
        groups = {}
        for key, row in wires.items():
            if row['opcode'] not in (137, 138) or row['enter'] >= end or row['finish'] < begin:
                continue
            require(key in service_calls, 'target phase RPC lacks exact service identity')
            call = service_calls[key]
            require(call['opcode'] == row['opcode'] and row['send_begin'] <= call['receive']
                    <= call['reply'] <= row['reply'], 'target phase RPC service order differs')
            duration, counts = decomposition(row, call['reply'], events[row['serial']], begin, end)
            name = (row['thread'], row['opcode'], row['boundary'] == 0)
            group = groups.setdefault(name, dict(thread=row['thread'], opcode=row['opcode'],
                background=row['boundary'] == 0, calls=0, latency_ns=0, phase_ns=Counter()))
            group['calls'] += 1
            group['latency_ns'] += duration
            group['phase_ns'].update(counts)
        selected = [group for group in groups.values() if group['background']]
        total = sum(group['latency_ns'] for group in selected)
        require(total > 0 and len({group['thread'] for group in selected}) == 1,
                'one observed background async progress thread per role required')
        counts = Counter()
        for group in selected:
            counts.update(group['phase_ns'])
        require(sum(counts.values()) == total, 'latency partition is incomplete')
        dominant = [name for name, value in counts.items() if value >= .6 * total]
        output.append(dict(role=role, application_identity=identity, groups=list(groups.values()),
            background_async_reply_latency_ns=total, phase_ns=counts,
            phase_fraction={name: value / total for name, value in counts.items()},
            dominant_phase=dominant[0] if len(dominant) == 1 else 'inconclusive',
            base_records=bases[role]['base_count'], phase_events=phase_documents[role]['phase_count']))
    inference = [row for row in output if row['role'] == 'inference']
    require(len(inference) == 1, 'one predeclared inference decision role required')
    return dict(diagnostic_only=True, begin_ns=begin, end_ns=end, roles=output,
        decision_role='inference', dominant_phase=inference[0]['dominant_phase'],
        individual_phase_threshold=.6, training_role='descriptive consistency evidence only',
        denominator='Same-wire background137/138 reply-to-read latency sum clipped to fixed window; not wall or GPU time.',
        coarse_lifecycle=coarse, full_device_qualified=False, priority_qualified=False,
        gpu_gap_causality_qualified=False, performance_qualified=False)
