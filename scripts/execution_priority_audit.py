"""Pure offline typed native-command/physical-priority evidence validation."""
import re
from bisect import bisect_right

from execution_semantic_audit import parse_semantic_audit

LAUNCH = 11
PERSISTENT_COMPUTE_OPERATIONS = {LAUNCH, 30, 31, 32, 42, 56}
ORIGINAL_ONLY_COMPUTE_OPERATIONS = {129, 139, 140}
COMPUTE_OPERATIONS = PERSISTENT_COMPUTE_OPERATIONS | ORIGINAL_ONLY_COMPUTE_OPERATIONS
UNIFIED_COPY = 141
MEMORY_OPERATIONS = {21, 23, 25, UNIFIED_COPY}
ORDERING_OPERATIONS = {48, 52, 54}
ORDERING_EVENTS = {
    'event_accepted': 1, 'event_issued': 2, 'event_retired': 3,
    'foreign_event_accepted': 4,
}
FIELDS = ('session', 'priority', 'sequence', 'completed', 'total', 'launches', 'status')
EVENT = (r'Priority execution audit: event=([a-z_]+), session=([0-9]+), priority=([0-9]+), '
         r'sequence=([0-9]+), completed_blocks=([0-9]+), total_blocks=([0-9]+), '
         r'physical_launches=([0-9]+), status=(-?[0-9]+)')
ADMISSION = (r'Native admission audit: session=([0-9]+), sequence=([0-9]+), '
             r'accepted=true, operation=([0-9]+)')
PIPELINE = (r'Native kernel pipeline audit: session=([0-9]+), first=([0-9]+), '
            r'second=([0-9]+), both_pending=true')
# Permit naked fixture records and the service's bracketed spdlog prefix, but
# never a substring match that accepts a numeric prefix or drops trailing data.
LOG_PREFIX = r'[ \t]*(?:\[[^\]\r\n]*\][ \t]*)*'
RECORDS = {
    marker: re.compile(LOG_PREFIX + r'(?P<record>' + pattern + r')')
    for marker, pattern in (
        ('Priority execution audit', EVENT),
        ('Native admission audit', ADMISSION),
        ('Native kernel pipeline audit', PIPELINE),
    )
}

BLOCK_MARKER = 'Block completion audit'
BLOCK_FIELDS = ('session', 'sequence', 'priority', 'pid', 'start', 'uid', 'total', 'workers')
BLOCK_RECORD = re.compile(
    LOG_PREFIX + r'(?P<record>Block completion audit: v=1, session=(\d+), sequence=(\d+), '
    r'priority=(\d+), pid=(\d+), start=(\d+), uid=(\d+), total_blocks=(\d+), '
    r'physical_launches=1, physical_stops=0, stopped_completed_blocks=0, '
    r'workers=(\d+), has_completed_block=true)')


def _validate_block_completions(content, admissions, events, semantic):
    """An owner-observed mapped word is neither a claim counter nor a stop."""
    bindings = {record.value('connection'): record for record in semantic.family(4)
                if record.value('event') == 1}
    publications, seen = [], set()
    offset = 0
    for original in content.splitlines(keepends=True):
        line = original.rstrip('\r\n')
        if BLOCK_MARKER in line:
            match = BLOCK_RECORD.fullmatch(line)
            if line.count(BLOCK_MARKER) != 1 or match is None:
                raise ValueError('malformed block-completion publication')
            value = dict(zip(BLOCK_FIELDS, map(int, match.groups()[1:])))
            position = offset + match.start('record')
            key = value['session'], value['sequence']
            if (key in seen or not all(value[field] for field in
                                      ('session', 'sequence', 'pid', 'start', 'total', 'workers')) or
                    any(number < 0 or number > 0xffffffffffffffff for number in value.values()) or
                    any(value[field] > 0xffffffff for field in ('pid', 'uid', 'priority')) or
                    value['workers'] > 0x7fffffff):
                raise ValueError('duplicate or invalid block-completion identity/geometry')
            binding = bindings.get(value['session'])
            if (binding is None or binding.position >= position or binding.value('role') != 0 or
                    binding.value('effective') != 1 or
                    any(value[field] != binding.value(field) for field in
                        ('pid', 'start', 'uid', 'priority')) or
                    any(record.value('connection') == value['session'] and
                        record.value('event') != 1 and record.position < position
                        for record in semantic.family(4))):
                raise ValueError('block publication lacks its exact active authenticated configured source')
            admission = admissions.get(key)
            issues = [(point, event) for point, event in events
                      if (event['session'], event['sequence']) == key and event['event'] == 'low_issued']
            if (admission is None or admission[0] not in PERSISTENT_COMPUTE_OPERATIONS or
                    len(issues) != 1 or not admission[1] < issues[0][0] < position):
                raise ValueError('block publication lacks its exact typed persistent first issue')
            issue = issues[0][1]
            if (issue['launches'] != 1 or issue['completed'] != 0 or
                    issue['total'] != value['total'] or issue['priority'] != value['priority'] or
                    any((event['session'], event['sequence']) == key and
                        issues[0][0] < point < position and event['event'] in
                        ('retreat_requested', 'low_physical_stop', 'low_resumed', 'low_logical_done',
                         'nonpreemptible_wait') for point, event in events)):
                raise ValueError('block publication is not an active unstopped original generation')
            seen.add(key)
            publications.append(dict(value, position=position, end_position=offset + len(original),
                                     operation=admission[0], physical_launches=1, physical_stops=0,
                                     stopped_completed_blocks=0, has_completed_block=True))
        offset += len(original)
    return publications


def parse_block_completions(content):
    admissions, events, _, semantic = _parse_audit_model(content)
    return _validate_block_completions(content, admissions, events, semantic)


def compute_operation_classes():
    """Recognized compute is not a promise of a useful or preemptible launch."""
    return dict(persistent_capable=sorted(PERSISTENT_COMPUTE_OPERATIONS),
                original_only=sorted(ORIGINAL_ONLY_COMPUTE_OPERATIONS))


def _validate_unified_copy_progress(admissions, events):
    """Generic SDK copies cannot publish persistent generations or block work.

    The runtime emits native wait/final completion for these commands. Its
    low_physical_stop requires positive persistent launch/stop counters, not a
    native copy's trailing event; high_physical_done is a session-level record.
    """
    ordinary = {'low_issued', 'high_issued', 'nonpreemptible_wait', 'low_logical_done'}
    for position, event in events:
        admission = admissions.get((event['session'], event['sequence']))
        if admission is None or admission[0] != UNIFIED_COPY:
            continue
        if (admission[1] >= position or event['event'] not in ordinary or
                event['completed'] or event['total'] or event['launches']):
            raise ValueError('ordinary unified copy fabricated persistent compute progress')


def _validate_original_only(admissions, events):
    """Keep native-only frontiers visible even in a filtered SDK fixture.

    Application analysis additionally checks every command's full lifecycle.
    Here the appended class must never hide behind the standalone verifier's
    selected low/high types or measurement boundary. A high native completion
    may remain unproven only if no subsequent lower-priority work crosses that
    frontier. Low/high audit cohorts do not forbid equal-priority pipelining.
    """
    if not any(operation in ORIGINAL_ONLY_COMPUTE_OPERATIONS
               for operation, _ in admissions.values()):
        return
    # Fix the contract from actual native-only issues, then check the entire
    # session history, including earlier setup and filtered command kinds.
    # Merely admitting a GEAM does not impose a role on an unused session.
    session_roles = {}
    role_events = {
        'low': {'low_bound', 'low_issued', 'nonpreemptible_wait', 'low_logical_done',
                'retreat_requested', 'low_physical_stop', 'low_resumed'},
        'high': {'native_bound', 'high_issued', 'high_physical_done'},
    }
    for names in role_events.values():
        names.update(ORDERING_EVENTS)
    for _, event in events:
        admitted = admissions.get((event['session'], event['sequence']))
        if (admitted is not None and admitted[0] in ORIGINAL_ONLY_COMPUTE_OPERATIONS and
                event['event'] in ('low_issued', 'high_issued')):
            contract = ('low' if event['event'] == 'low_issued' else 'high', event['priority'])
            if session_roles.setdefault(event['session'], contract) != contract:
                raise ValueError('original-only session changed priority or role')
    for _, event in events:
        contract = session_roles.get(event['session'])
        if contract is None:
            continue
        role, priority = contract
        if event['priority'] != priority or event['event'] not in role_events[role]:
            raise ValueError('original-only session changed priority or role')
        if event['event'] in ('low_bound', 'native_bound', 'high_physical_done'):
            if event['sequence'] or event['completed'] or event['total'] or event['launches']:
                raise ValueError('original-only session record has command progress')
        if event['event'] in ORDERING_EVENTS and (
                event['completed'] or event['total'] or event['launches']):
            raise ValueError('ordering event fabricated compute progress')
    originals, active_lows, pending_high, active_ordering = {}, {}, {}, {}
    for position, event in events:
        name = event['event']
        key = event['session'], event['sequence']
        admitted = admissions.get(key)
        operation = admitted[0] if admitted is not None else None
        is_original = operation in ORIGINAL_ONLY_COMPUTE_OPERATIONS
        if is_original:
            if admitted[1] >= position:
                raise ValueError('missing or late original-only command admission')
            if event['completed'] or event['total'] or event['launches']:
                raise ValueError('original-only compute fabricated persistent progress')
            if name in ('low_issued', 'high_issued'):
                if key in originals:
                    raise ValueError('original-only compute issued more than once')
                originals[key] = dict(role=name, done=False, waited=False)
            elif name in ('nonpreemptible_wait', 'low_logical_done'):
                state = originals.get(key)
                if state is None or state['role'] != 'low_issued' or state['done']:
                    raise ValueError('original-only compute lacks its active native issue')
                if name == 'nonpreemptible_wait':
                    if state['waited']:
                        raise ValueError('original-only compute repeated its native wait')
                    state['waited'] = True
                else:
                    state['done'] = True
            else:
                raise ValueError('original-only compute fabricated a persistent transition')

        if name == 'event_issued':
            if any(priority > event['priority'] and kind in ORIGINAL_ONLY_COMPUTE_OPERATIONS
                   for pending in pending_high.values() for kind, priority in pending):
                raise ValueError('ordering issue while higher original completion unproven')
            if any(priority < event['priority'] and kind in ORIGINAL_ONLY_COMPUTE_OPERATIONS
                   for kind, priority in active_lows.values()):
                raise ValueError('ordering issue before lower original physical stop')
            active_ordering[key] = event['priority']
            continue
        if name == 'event_retired':
            active_ordering.pop(key, None)
            continue
        if name in ('low_issued', 'low_resumed'):
            if (any(priority > event['priority'] and
                   (is_original or kind in ORIGINAL_ONLY_COMPUTE_OPERATIONS)
                   for pending in pending_high.values() for kind, priority in pending) or
                    (is_original and any(priority > event['priority']
                                         for priority in active_ordering.values()))):
                action = 'issue' if name == 'low_issued' else 'resume'
                raise ValueError(f'low {action} while high completion unproven')
            active_lows[key] = operation, event['priority']
        elif name in ('low_physical_stop', 'low_logical_done'):
            active_lows.pop(key, None)
        elif name == 'high_issued':
            if (any(priority < event['priority'] and
                   (is_original or kind in ORIGINAL_ONLY_COMPUTE_OPERATIONS)
                   for kind, priority in active_lows.values()) or
                    (is_original and any(priority < event['priority']
                                         for priority in active_ordering.values()))):
                raise ValueError('high issue before proven low physical stop')
            pending_high.setdefault(event['session'], []).append((operation, event['priority']))
        elif name == 'high_physical_done':
            pending = pending_high.get(event['session'], [])
            if any(kind in ORIGINAL_ONLY_COMPUTE_OPERATIONS for kind, _ in pending):
                if event['sequence'] or event['completed'] or event['total'] or event['launches']:
                    raise ValueError('original-only high completion has command progress')
            if not event['sequence']:
                pending_high.pop(event['session'], None)
    if any(state['role'] == 'low_issued' and not state['done'] for state in originals.values()):
        raise ValueError('missing final exactly-once original-only completion')


def _parse_audit_model(content):
    """Validate every known marker, including setup and filtered memory work.

    A ready boundary limits the physical-order proof, not failure detection:
    unexplained nonzero statuses anywhere make the entire receipt ineligible.
    Unrelated service messages are not evidence and remain ignored.
    """
    semantic = parse_semantic_audit(
        content, require_complete='Execution semantic audit' in content)
    admissions, events, pipelines = {}, [], []
    connections = {record.value('connection'): record for record in semantic.family(4)
                   if record.value('event') == 1}

    def bound_connection(session, position, priority=None, command=False):
        if not semantic.records:
            return
        binding = connections.get(session)
        if (binding is None or binding.position >= position or
                (priority is not None and binding.value('priority') != priority) or
                (command and binding.value('role') != 0)):
            raise ValueError('legacy audit lacks its exact authenticated source binding')

    ordering = {}
    for record in semantic.family(3):
        key = record.value('connection'), record.value('sequence')
        name = next(name for name, code in ORDERING_EVENTS.items() if code == record.value('event'))
        ordering[key, name] = record
        if record.value('event') == 1:
            admissions[key] = record.value('opcode'), record.position
    seen_ordering = set()
    line_offset = 0
    for number, original in enumerate(content.splitlines(keepends=True), 1):
        line = original.removesuffix('\n').removesuffix('\r')
        markers = [marker for marker in RECORDS if marker in line]
        if markers:
            marker = markers[0]
            match = RECORDS[marker].fullmatch(line)
            if sum(line.count(marker) for marker in markers) != 1 or match is None:
                raise ValueError(f'malformed {marker} at line {number}')
            position = line_offset + match.start('record')
            values = match.groups()[1:]
            if marker == 'Native admission audit':
                session, sequence, operation = map(int, values)
                key = session, sequence
                if (not session or not sequence or key in admissions or
                        operation not in COMPUTE_OPERATIONS | MEMORY_OPERATIONS):
                    raise ValueError('invalid/duplicate typed admission')
                bound_connection(session, position, command=True)
                admissions[key] = operation, position
            elif marker == 'Priority execution audit':
                name, *numbers = values
                event = dict(event=name, **dict(zip(FIELDS, map(int, numbers))))
                if event['status']:
                    raise ValueError('actual priority SDK/policy failure')
                bound_connection(event['session'], position, event['priority'])
                if name in ORDERING_EVENTS:
                    identity = ((event['session'], event['sequence']), name)
                    record = ordering.get(identity)
                    if (record is None or identity in seen_ordering or
                            event['priority'] != record.value('priority') or
                            event['status'] != record.value('status') or
                            event['completed'] or event['total'] or event['launches']):
                        raise ValueError('ordering priority annotation lacks its exact typed lifecycle')
                    seen_ordering.add(identity)
                events.append((position, event))
            else:
                pipeline = tuple(map(int, values))
                bound_connection(pipeline[0], position, command=True)
                pipelines.append((position, pipeline))
        line_offset += len(original)
    for (key, name), record in ordering.items():
        if (key, name) not in seen_ordering:
            events.append((record.position, dict(
                event=name, session=key[0], sequence=key[1],
                priority=record.value('priority'), status=record.value('status'),
                completed=0, total=0, launches=0)))
    events.sort(key=lambda item: item[0])
    if semantic.records:
        _join_source_issues(admissions, events, semantic)
    _validate_unified_copy_progress(admissions, events)
    _validate_original_only(admissions, events)
    _validate_block_completions(content, admissions, events, semantic)
    return admissions, events, pipelines, semantic


def _join_source_issues(admissions, events, semantic):
    """Reconcile every actual leaf result with its next issue, never peer credit."""
    issue_names = {'low_issued', 'high_issued', 'low_resumed', 'event_issued'}
    issues = [(position, event) for position, event in events if event['event'] in issue_names]
    positions = [position for position, _ in issues]
    grants = {record.serial: record for record in semantic.family(2)
              if record.value('event') == 10}
    grant_positions = sorted(record.position for record in grants.values())
    consumed, initial = set(), set()
    for _, event in issues:
        if event['event'] in ('low_issued', 'high_issued'):
            key = event['session'], event['sequence']
            if key in initial:
                raise ValueError('semantic command repeated its original issue annotation')
            initial.add(key)
    for record in semantic.family(2):
        if record.value('event') != 12 or record.value('issue_state') != 1:
            continue
        key = record.value('source'), record.value('command_sequence')
        admission = admissions.get(key)
        if admission is None or admission[1] >= record.position:
            raise ValueError('source result lacks its prior exact typed admission')
        index = bisect_right(positions, record.position)
        if index == len(issues):
            raise ValueError('source result lacks its actual issue or resume annotation')
        position, event = issues[index]
        next_grant = bisect_right(grant_positions, record.position)
        if (position in consumed or (event['session'], event['sequence']) != key or
                event['status'] != record.value('native_status') or
                event['priority'] != grants[record.value('grant')].value('priority') or
                (next_grant < len(grant_positions) and position >= grant_positions[next_grant])):
            raise ValueError('source result changed its exact actual issue or resume')
        if ((admission[0] in ORDERING_OPERATIONS) != (event['event'] == 'event_issued')):
            raise ValueError('ordering and compute source issue types were interchanged')
        consumed.add(position)


def _parse_audit(content):
    """Retain the legacy tuple while validating every new semantic family."""
    admissions, events, pipelines, _ = _parse_audit_model(content)
    return admissions, events, pipelines


def _priority_phase(content, offset, low_operation):
    if type(offset) is not int or not 0 < offset < len(content):
        raise ValueError('missing post-ready phase boundary')
    admissions, events, pipelines = _parse_audit(content)
    raw, kernels = [], []
    for position, event in events:
        if position < offset:
            continue
        raw.append(event)
        if event['sequence']:
            kind = admissions.get((event['session'], event['sequence']))
            if kind is not None and kind[0] in ORDERING_OPERATIONS:
                # One typed acceptance record is both admission and an ordering
                # transition. The complete semantic model already validates it;
                # it must remain visible without acquiring kernel credit.
                continue
            if kind is None or kind[1] >= position:
                raise ValueError('missing or late typed command admission')
            if kind[0] not in {LAUNCH, low_operation}: continue
        kernels.append((position, event))
    return raw, kernels, admissions, pipelines


def kernel_phase(content, offset, low_operation=LAUNCH):
    """Return raw and kernel-only phases; never discard errors or invent types."""
    raw, kernels, _, _ = _priority_phase(content, offset, low_operation)
    return raw, kernels

def verify_priority(content, offset, preemptible=True, low_count=1, require_pipeline=True,
                    low_operation=LAUNCH, total_blocks=65536):
    if low_operation not in COMPUTE_OPERATIONS or type(total_blocks) is not int or total_blocks <= 0:
        raise ValueError('unsupported explicit compute operation/geometry')
    if preemptible and low_operation not in PERSISTENT_COMPUTE_OPERATIONS:
        raise ValueError('original-only compute cannot qualify preemption')
    raw, phase, admissions, pipelines = _priority_phase(content, offset, low_operation)
    events = [event for _,event in phase]
    lows = [i for i,e in enumerate(events) if e['event']=='low_issued']
    highs = [i for i,e in enumerate(events) if e['event']=='high_issued']
    if len(lows)!=low_count or len(highs)!=2:
        raise ValueError('wrong number of actual low/high kernel SDK issues')
    low_ids = [(events[i]['session'],events[i]['sequence']) for i in lows]
    high_ids = [(events[i]['session'],events[i]['sequence']) for i in highs]
    typed = {key: operation for key, (operation, _) in admissions.items()}
    if any(typed[key] != low_operation for key in low_ids) or any(typed[key] != LAUNCH for key in high_ids):
        raise ValueError('actual compute command types do not match the selected contract')
    if len(set(low_ids))!=low_count or len(set(high_ids))!=2 or len({s for s,_ in low_ids})!=1 or len({s for s,_ in high_ids})!=1 or low_ids[0][0]==high_ids[0][0]:
        raise ValueError('invalid kernel role/session identities')
    if any(events[i]['priority'] >= events[highs[0]]['priority'] for i in lows):
        raise ValueError('low/high role priorities are not ordered')
    high_done = next((i for i,e in enumerate(events) if i>highs[1] and e['event']=='high_physical_done' and e['session']==high_ids[0][0]), None)
    if high_done is None: raise ValueError('missing actual high stop after both kernel issues')
    if require_pipeline:
        witness = False
        for position, (session, first, second) in pipelines:
            if [(session,first),(session,second)]==high_ids and phase[highs[1]][0]<position<phase[high_done][0]: witness=True
        if not witness: raise ValueError('missing exact two-kernel pending-event pipeline')
    for identity in low_ids:
        own = [(i,e) for i,e in enumerate(events) if (e['session'],e['sequence'])==identity]
        names = ('low_issued','retreat_requested','low_physical_stop','low_resumed','low_logical_done') if preemptible else ('low_issued','nonpreemptible_wait','low_logical_done')
        positions = {name:[i for i,e in own if e['event']==name] for name in names}
        if any(not v for v in positions.values()) or len(positions['low_issued'])!=1 or len(positions['low_logical_done'])!=1:
            raise ValueError('missing or repeated low kernel lifecycle')
        first = {name:v[0] for name,v in positions.items()}
        if not preemptible:
            if not first['low_issued']<first['nonpreemptible_wait']<first['low_logical_done']<highs[0]<highs[1]:
                raise ValueError('native kernel did not stop before higher SDK issue')
            if any(e['event'] in ('retreat_requested','low_physical_stop','low_resumed') for _,e in own):
                raise ValueError('native-only kernel fabricated preemption')
            continue
        if not first['low_issued']<first['retreat_requested']<first['low_physical_stop']<highs[0]<highs[1]<high_done<first['low_resumed']<first['low_logical_done']:
            raise ValueError('missing incomplete stop/high completion/exact continuation order')
        partial, done = events[first['low_physical_stop']],events[first['low_logical_done']]
        if not (0<partial['completed']<partial['total']==total_blocks and partial['launches']==1 and done['completed']==done['total']==total_blocks and done['launches']>=2):
            raise ValueError('positive incomplete compute or final exactly-once completion unproven')
        generation, completed, active, paused, requested = 0,0,False,False,False
        for _,e in own:
            name = e['event']
            if name=='low_issued':
                if generation or e['launches']!=1: raise ValueError('repeated original low kernel')
                generation,active = 1,True
            elif name=='retreat_requested':
                if not active or requested or e['launches']!=generation: raise ValueError('invalid retreat generation')
                requested=True
            elif name=='low_physical_stop':
                if not active or not requested or e['launches']!=generation or not completed<=e['completed']<e['total']==total_blocks: raise ValueError('unproven physical stop')
                completed,active,paused=e['completed'],False,True
            elif name=='low_resumed':
                if active or not paused or e['launches']!=generation+1 or e['completed']!=completed: raise ValueError('reset/replay/early continuation')
                generation,active,paused,requested=e['launches'],True,False,False
            elif name=='low_logical_done':
                if not active or e['launches']!=generation or e['completed']!=total_blocks: raise ValueError('unproven logical completion')
                active=False
    original_commands = [dict(session=event['session'], sequence=event['sequence'],
                              operation=admissions[(event['session'], event['sequence'])][0])
                         for event in raw if event['event'] in ('low_issued', 'high_issued') and
                         admissions[(event['session'], event['sequence'])][0]
                         in ORIGINAL_ONLY_COMPUTE_OPERATIONS]
    return dict(transitions=raw,kernel_transitions=events,low_kernel_commands=low_ids,high_kernel_commands=high_ids,
                compute_operation_classes=compute_operation_classes(),
                original_only_compute_commands=original_commands)
