"""Reconstruct bounded service-visible lifecycle intervals, never GPU idle time."""

from collections import defaultdict
import re

from artifact_paths import PROJECT_ROOT
from run_bert_candidate_validation import require

WIDTH = 50_000_000
FIELDS = ['wall_ns', 'slot', 'kind', 'wire', 'command', 'detail']
KINDS = ['receive', 'admit', 'issue_begin', 'issue_end', 'reply', 'retire']


def opcode_names():
    text = (PROJECT_ROOT / 'include/glimmer/control/execution_protocol.h').read_text()
    body = text.split('enum class ExecutionOpcode : std::uint32_t {', 1)[1].split('};', 1)[0]
    names, number = {}, 0
    for name, explicit in re.findall(r'\b(k\w+)\s*(?:=\s*(\d+))?\s*,', body):
        number = int(explicit) if explicit else number + 1
        names[number] = name
    return names


def reconstruct(ledger, slots, begin, end):
    require(ledger['fields'] == FIELDS and ledger['kinds'] == KINDS
            and 0 < ledger['capacity'] <= 4194304
            and 0 < len(ledger['events']) <= ledger['capacity']
            and 0 < end - begin <= 5_000_000_000 and (end - begin) % WIDTH == 0,
            'bounded exact lifecycle schema and window required')
    states = [dict(wires={}, commands={}, last_wire=0, last_command=0, first=None, last=None, kind=None,
                   retired=False, receives=0, admissions=0, issues=0, replacements=0,
                   max_depth=0, max_unissued=0, spans=defaultdict(lambda: [0, 0, 0]),
                   depth=[[0, 0, 0] for _ in range((end - begin) // WIDTH)],
                   unissued_depth=[0, 0, 0], supply_gap_ns=0,
                   phases=defaultdict(int), op_count=defaultdict(int)) for _ in slots]
    last_stamp, issuing = 0, None

    def span(state, record, name, left, right):
        if left is None or right is None:
            return
        require(left <= right, 'reversed lifecycle span')
        if begin <= left <= right <= end:
            value = state['spans'][(record['op'], name)]
            value[0] += 1
            value[1] += right - left
            value[2] = max(value[2], right - left)

    for event in ledger['events']:
        require(len(event) == 6 and all(type(v) is int and v >= 0 for v in event),
                'invalid lifecycle event')
        stamp, slot, kind, wire, command, detail = event
        require(last_stamp <= stamp and slot < len(states) and kind < len(KINDS),
                'event clock/order/slot invalid')
        last_stamp = stamp
        state = states[slot]
        wires, commands = state['wires'], state['commands']
        if state['last'] is not None:
            left, right = max(begin, state['last']), min(end, stamp)
            depth = min(len(wires), 2)
            unissued_depth = min(sum(r['issue_begin'] is None for r in commands.values()), 2)
            # Mutually exclusive service-visible state, not an execution grant.
            phase = ('no_request_with_unissued_command' if not wires and unissued_depth else
                     'no_request' if not wires else
                     'unadmitted_request' if any(r['admit'] is None for r in wires.values()) else
                     'admitted_waiting_issue' if any(r['issue_begin'] is None for r in wires.values()) else
                     'inside_native_issue' if any(r['issue_end'] is None for r in wires.values()) else
                     'issued_waiting_reply')
            while left < right:
                index = (left - begin) // WIDTH
                stop = min(right, begin + (index + 1) * WIDTH)
                state['depth'][index][depth] += stop - left
                state['unissued_depth'][unissued_depth] += stop - left
                if not depth and not unissued_depth:
                    state['supply_gap_ns'] += stop - left
                state['phases'][phase] += stop - left
                left = stop
        state['last'] = stamp
        if kind == 0:
            opcode, slot_kind = detail & 0xffffffff, detail >> 32
            require(slot_kind in (0, 1, 2, 3) and state['kind'] in (None, slot_kind),
                    'unknown or changed immutable slot kind')
            state['kind'] = slot_kind
            require(not state['retired'] and wire == state['last_wire'] + 1
                    and wire not in wires and command == 0 and len(wires) < 8,
                    'duplicate, unordered, retired or excessive pending wire')
            if state['first'] is None:
                state['first'] = stamp
            state['last_wire'] = wire
            state['receives'] += 1
            wires[wire] = dict(op=opcode, receive=stamp, admit=None, issue_begin=None,
                              issue_end=None, reply=None, command=0)
            if begin <= stamp < end:
                state['op_count'][opcode] += 1
            state['max_depth'] = max(state['max_depth'], len(wires))
        elif kind == 1:
            require(wire in wires and command > state['last_command'] and detail in (0, 1),
                    'admission lacks a unique pending wire/command')
            record = wires[wire]
            require(record['admit'] is None, 'duplicate admission')
            record.update(admit=stamp, command=command)
            commands[command] = record
            state['last_command'] = command
            state['admissions'] += 1
            state['replacements'] += detail
            unissued = sum(r['issue_begin'] is None for r in commands.values())
            require(unissued <= 64 and len(commands) <= 72, 'command correlation capacity exceeded')
            state['max_unissued'] = max(state['max_unissued'], unissued)
            span(state, record, 'receive_to_admit', record['receive'], stamp)
        elif kind == 2:
            require(command in commands and wire == 0 and detail in (0, 1) and issuing is None,
                    'unmatched, resumed or concurrent ordinary issue')
            record = commands[command]
            require(record['issue_begin'] is None, 'duplicate first issue')
            record['issue_begin'] = stamp
            issuing = (slot, command)
            span(state, record, 'admit_to_issue', record['admit'], stamp)
        elif kind == 3:
            require(issuing == (slot, command) and wire == command and detail == 0,
                    'missing or unsuccessful actual native issue')
            record = commands[command]
            record['issue_end'] = stamp
            state['issues'] += 1
            span(state, record, 'native_issue', record['issue_begin'], stamp)
            issuing = None
            if record['reply'] is not None:
                del commands[command]
        elif kind == 4:
            require(wire in wires, 'reply lacks its pending wire')
            record = wires[wire]
            require(command == record['command'], 'wire-to-command reply association changed')
            disposition, status = detail & 0xffffffff, detail >> 32
            if command:
                require(status == 0 and (state['kind'] != 0 or disposition == 4
                                        or record['issue_end'] is not None),
                        'admitted command lacks issue or admission-only reply')
            else:
                require(disposition != 4, 'admission-only reply lacks command')
            record['reply'] = stamp
            span(state, record, 'receive_to_reply', record['receive'], stamp)
            if record['issue_end'] is not None:
                span(state, record, 'issue_to_reply', record['issue_end'], stamp)
                del commands[command]
            elif command and state['kind'] != 0:
                # Helper admissions are not ordinary scheduler issue events.
                # Their reply closes correlation only, never physical work.
                del commands[command]
            del wires[wire]
        else:
            require(not state['retired'] and wire == command == detail == 0,
                    'duplicate or invalid slot retirement marker')
            state['retired'] = True
    require(issuing is None, 'unterminated native issue')
    output = []
    names = opcode_names()
    for slot, state in enumerate(states):
        require(state['first'] is not None and state['retired'] and not state['wires']
                and not state['commands'], 'incomplete lifecycle or unmatched associations')
        totals = [sum(row[i] for row in state['depth']) for i in range(3)]
        coverage = sum(totals) / (end - begin)
        gate = state['first'] <= begin and state['last'] >= end and coverage >= .95
        pending = (totals[1] + totals[2]) / (end - begin)
        boundary = ('inconclusive' if not gate else 'service_owned_wait' if pending >= .6 else
                    'frontend_transport_supply' if state['supply_gap_ns'] >= .6 * (end - begin)
                    else 'inconclusive')
        output.append(dict(slot=slots[slot], slot_kind=state['kind'], compute_connection=state['kind'] == 0,
            lifecycle_coverage=coverage, coverage_gate=gate, pending_window_fraction=pending,
            next_boundary=boundary, depth_fields=['zero_ns', 'one_ns', 'multiple_ns'],
            depth_ns=totals, depth_buckets=state['depth'], phase_ns=dict(state['phases']),
            admitted_unissued_depth_ns=state['unissued_depth'],
            zero_wire_and_zero_unissued_ns=state['supply_gap_ns'],
            lifetime_receives=state['receives'], lifetime_admissions=state['admissions'],
            lifetime_issues=state['issues'], lifetime_replacements=state['replacements'],
            max_pending_wires=state['max_depth'], max_admitted_unissued=state['max_unissued'],
            received_in_window=[dict(opcode=op, name=names.get(op, str(op)), count=count)
                                for op, count in sorted(state['op_count'].items())],
            spans=[dict(opcode=op, name=names.get(op, str(op)), phase=phase,
                        count=values[0], total_ns=values[1], max_ns=values[2])
                   for (op, phase), values in sorted(state['spans'].items())]))
    return output


def analyze(document, job, begin, end):
    require(document['valid'] is True and document['pid'] == job['server_identity']['pid'],
            'valid exact service owner required')
    slots = document['slots']
    roles = {(row['pid'], row['start']): role for role, row in job['application_identities'].items()}
    require(len(slots) <= 128 and len({slot[0] for slot in slots}) == len(slots)
            and all(tuple(slot[1:]) in roles for slot in slots), 'authenticated unique slots required')
    rows = reconstruct(document['lifecycle'], slots, begin, end)
    for row in rows:
        row['role'] = roles[tuple(row['slot'][1:])]
    compute = [row for row in rows if row['compute_connection']]
    require(len(compute) == 2 and {row['role'] for row in compute} == {'training', 'inference'}
            and all(row['coverage_gate'] for row in compute), 'two complete compute lifecycles required')
    return dict(begin_ns=begin, end_ns=end, diagnostic_only=True,
        meaning='Service-visible intervals only; no GPU idle, physical completion or frontend cause proof.',
        phase_precedence='no request; unadmitted wire; admitted waiting issue; inside issue; issued waiting reply',
        latency_scope='Both endpoints inside fixed window; concurrent request durations may overlap.',
        slots=rows, summary=[{key: row[key] for key in ('role', 'pending_window_fraction',
            'next_boundary', 'phase_ns', 'max_pending_wires', 'max_admitted_unissued')}
            for row in compute])
