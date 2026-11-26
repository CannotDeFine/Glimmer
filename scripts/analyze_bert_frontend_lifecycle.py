"""Join exact frontend/service IDs and partition host-visible supply gaps."""

from collections import Counter

from run_bert_candidate_validation import require

FIELDS = ['kind', 'serial', 'thread', 'boundary', 'session', 'wire', 'opcode', 'enter',
          'locked', 'send_begin', 'send_end', 'reply', 'finish', 'cpu_enter', 'cpu_finish', 'flags']
PHASES = ['outbound_submission', 'reply_delivery_and_rpc_return', 'helper_or_background_rpc',
          'frontend_boundary_preparation_or_lock', 'outside_observed_boundary']


def frontend(document, identity, begin, end):
    require(document['schema'] in (1, 2, 3, 5, 6) and document['diagnostic_only'] is True
            and document['valid'] is True and document['pid'] == identity['pid']
            and document['fields'] == FIELDS and document['begin_ns'] <= begin < end <= document['end_ns']
            and 0 < len(document['records']) <= document['capacity']
            <= {1: 1048576, 2: 2097152, 3: 2097152, 5: 2097152, 6: 2097152}.get(document['schema'], 0),
            'complete bounded authenticated frontend observation required')
    boundaries, calls, serials = {}, [], set()
    for values in document['records']:
        require(len(values) == len(FIELDS) and all(type(v) is int and v >= 0 for v in values),
                'invalid frontend record')
        row = dict(zip(FIELDS, values))
        require(row['serial'] > 0 and row['serial'] not in serials and row['thread'] > 0
                and row['kind'] in (0, 1) and row['cpu_enter'] <= row['cpu_finish']
                and document['begin_ns'] <= row['enter'] <= row['locked'] <= row['finish'] <= document['end_ns'],
                'invalid frontend record identity/clock/order')
        serials.add(row['serial'])
        if row['kind'] == 0:
            require(row['session'] == row['wire'] == row['opcode'] == row['boundary'] == row['flags'] == 0,
                    'nested or malformed API boundary')
            boundaries[row['serial']] = row
        else:
            if row['wire']:
                require(row['session'] > 0 and row['flags'] == 3
                        and row['locked'] <= row['send_begin'] <= row['send_end']
                        <= row['reply'] <= row['finish'], 'incomplete frontend wire lifecycle')
            else:
                require(row['flags'] == 0, 'unsubmitted RPC has a partial reply')
            calls.append(row)
    by_wire = {}
    for row in calls:
        if row['boundary']:
            require(row['boundary'] in boundaries, 'RPC lacks its API boundary')
            boundary = boundaries[row['boundary']]
            require(boundary['thread'] == row['thread']
                    and boundary['locked'] <= row['enter'] <= row['finish'] <= boundary['finish'],
                    'RPC escaped its actual caller boundary')
        if row['wire']:
            key = row['session'], row['wire']
            require(key not in by_wire, 'duplicate frontend session/wire')
            by_wire[key] = row
    return boundaries, calls, by_wire


def service(document, begin, end):
    """Exact owner-visible zero-wire/zero-unissued intervals; no GPU state."""
    states = [dict(wires=set(), unissued=set(), last=None, gaps=[], calls={}, kind=None)
              for _ in document['slots']]
    for stamp, index, kind, wire, command, detail in document['lifecycle']['events']:
        state = states[index]
        if state['last'] is not None and not state['wires'] and not state['unissued']:
            left, right = max(begin, state['last']), min(end, stamp)
            if left < right:
                state['gaps'].append((left, right))
        state['last'] = stamp
        if kind == 0:
            state['kind'] = detail >> 32
            state['wires'].add(wire)
            state['calls'][wire] = dict(receive=stamp, opcode=detail & 0xffffffff)
        elif kind == 1:
            state['unissued'].add(command)
        elif kind == 2:
            state['unissued'].remove(command)
        elif kind == 4:
            state['wires'].remove(wire)
            state['calls'][wire]['reply'] = stamp
            if state['kind'] != 0:
                state['unissued'].discard(command)
    return states


def partition(gaps, intervals, begin, end):
    """Use ordered precedence and interval unions, never add thread durations."""
    events = []
    for left, right in gaps:
        events.extend(((left, -1, 1), (right, -1, -1)))
    for phase, spans in enumerate(intervals):
        for left, right in spans:
            left, right = max(begin, left), min(end, right)
            if left < right:
                events.extend(((left, phase, 1), (right, phase, -1)))
    events.sort()
    active = [0] * (len(PHASES) - 1)
    inside, previous = 0, begin
    totals, overlaps = Counter(), Counter()
    for stamp, phase, delta in events:
        if inside and previous < stamp:
            present = tuple(i for i, count in enumerate(active) if count)
            selected = present[0] if present else len(PHASES) - 1
            totals[PHASES[selected]] += stamp - previous
            overlaps[','.join(PHASES[i] for i in present) or 'none'] += stamp - previous
        previous = stamp
        if phase == -1:
            inside += delta
            require(inside in (0, 1), 'overlapping service supply gaps')
        else:
            active[phase] += delta
            require(active[phase] >= 0, 'reversed classification interval')
    require(inside == 0 and not any(active), 'unclosed classification intervals')
    duration = sum(right - left for left, right in gaps)
    require(sum(totals.values()) == duration, 'supply-gap partition lost or duplicated time')
    return duration, dict(totals), dict(overlaps)


def analyze(document, frontend_documents, job, begin, end, request_lifecycle):
    require(request_lifecycle['begin_ns'] == begin and request_lifecycle['end_ns'] == end,
            'request lifecycle window differs')
    states = service(document, begin, end)
    output = []
    for role, identity in job['application_identities'].items():
        boundaries, calls, frontend_wires = frontend(frontend_documents[role], identity, begin, end)
        owned = [(slot, state) for slot, state in zip(document['slots'], states)
                 if tuple(slot[1:]) == (identity['pid'], identity['start'])]
        compute = [(slot, state) for slot, state in owned if state['kind'] == 0]
        require(len(compute) == 1, 'one authenticated ordinary compute connection per role required')
        slot, state = compute[0]
        session = slot[0]
        coverage_rows = [row for row in request_lifecycle['slots']
                         if row['role'] == role and row['compute_connection']]
        require(len(coverage_rows) == 1 and coverage_rows[0]['slot'] == slot
                and coverage_rows[0]['coverage_gate'] is True,
                'exact qualified service lifecycle coverage required')
        service_coverage = coverage_rows[0]['lifecycle_coverage']
        require(.95 <= service_coverage <= 1.0,
                'less than95percent combined frontend/service coverage')
        service_calls = {(owner[0], wire): call for owner, value in owned
                         for wire, call in value['calls'].items()}
        intervals = [[] for _ in range(4)]
        matched, service_count = 0, 0
        details = Counter()
        for wire, call in state['calls'].items():
            if call['receive'] < end and call['reply'] >= begin:
                service_count += 1
                require((session, wire) in frontend_wires, 'window compute wire has no frontend receipt')
        for key, row in frontend_wires.items():
            if row['enter'] >= end or row['finish'] < begin:
                continue
            require(key in service_calls, 'window frontend wire has no exact service owner')
            call = service_calls[key]
            require(row['opcode'] == call['opcode']
                    and row['send_begin'] <= call['receive'] <= call['reply'] <= row['reply'],
                    'frontend/service opcode or preflush monotonic ordering differs')
            if key[0] == session:
                if call['receive'] < end and call['reply'] >= begin:
                    matched += 1
                intervals[0].append((row['send_begin'], call['receive']))
                intervals[1].append((call['reply'], row['finish']))
                details['send_call_wall_ns'] += row['send_end'] - row['send_begin']
                details['rpc_wall_ns'] += row['finish'] - row['enter']
                details['rpc_cpu_ns'] += row['cpu_finish'] - row['cpu_enter']
                details['reply_to_read_ns'] += row['reply'] - call['reply']
                details['read_to_rpc_return_ns'] += row['finish'] - row['reply']
                details['rpc_calls_touching_window'] += 1
            if key[0] != session or row['boundary'] == 0:
                intervals[2].append((row['enter'], row['finish']))
        require(matched == service_count and matched > 0, 'less than100percent correlated compute wires')
        # Local no-wire RPCs still identify frontend work, but cannot invent a
        # service request or a transport delay. Boundary intervals contain them.
        for row in calls:
            if not row['wire'] and not row['boundary']:
                intervals[2].append((row['enter'], row['finish']))
        intervals[3] = [(row['enter'], row['finish']) for row in boundaries.values()]
        duration, phases, overlaps = partition(state['gaps'], intervals, begin, end)
        require(duration > 0 and duration == coverage_rows[0]['zero_wire_and_zero_unissued_ns'],
                'service supply-gap duration differs from validated request lifecycle')
        dominant = [name for name, value in phases.items() if duration and value >= .6 * duration]
        output.append(dict(role=role, session=session, correlated_compute_wires=matched,
            compute_correlation_fraction=1.0, frontend_recorder_window_coverage=1.0,
            service_lifecycle_coverage=service_coverage,
            observed_window_coverage=min(1.0, service_coverage),
            service_supply_gap_ns=duration, phase_ns=phases,
            phase_fraction_of_supply_gap={name: value / duration for name, value in phases.items()},
            raw_interval_overlap_ns=overlaps, dominance_denominator='service supply-gap duration only',
            dominant_supply_boundary=dominant[0] if len(dominant) == 1 else 'inconclusive',
            rpc_span_sums=dict(details),
            rpc_span_warning='Touching-window whole RPC spans may cross boundaries and overlap threads; not wall fractions.'))
    return dict(diagnostic_only=True, begin_ns=begin, end_ns=end, precedence=PHASES,
        scope='Host-visible supply gaps, not GPU-idle time. Outside observed boundary means upstream or unobserved work.',
        helper_scope='Helper/background coverage is concurrent activity, not proof that it blocks compute supply.',
        roles=output, gpu_gap_causality_qualified=False, performance_qualified=False)
