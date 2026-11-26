"""Strict offline application handoff components, never full qualification.

Claim counters are not pre-demand compute witnesses. An optional independent
mapped block-exit publication may join an explicit external-demand boundary,
but never supplies physical stop/cleanup authority. Legacy logs cannot borrow
it. No CUDA, collector or application module is imported and no missing
physical completion is manufactured.
"""

from dataclasses import dataclass, field

from execution_priority_audit import (
    COMPUTE_OPERATIONS, MEMORY_OPERATIONS, ORIGINAL_ONLY_COMPUTE_OPERATIONS,
    ORDERING_EVENTS, ORDERING_OPERATIONS, PERSISTENT_COMPUTE_OPERATIONS,
    _parse_audit_model, compute_operation_classes,
    _validate_block_completions,
)


MISSING_PRE_DEMAND = (
    "Independent useful-compute-before-high-demand evidence is missing: the "
    "current completed counter is observed only after physical stop/readback."
)
LOW_EVENTS = {
    "low_issued", "retreat_requested", "low_physical_stop", "low_resumed",
    "low_logical_done", "nonpreemptible_wait", "low_bound",
}
HIGH_EVENTS = {"high_issued", "high_physical_done", "native_bound"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(event, operation):
    return dict(session=event["session"], sequence=event["sequence"], operation=operation)


def _zero_progress(event):
    return not (event["completed"] or event["total"] or event["launches"])


def _validate_first_issue(event, operation):
    if event["total"]:
        _require(operation in PERSISTENT_COMPUTE_OPERATIONS and event["launches"] == 1 and
                 event["completed"] == 0, "invalid first persistent compute generation")
    else:
        _require(_zero_progress(event), "native issue has inconsistent progress")


@dataclass
class _LowCommand:
    command: dict
    issue_position: int
    total: int
    generation: int
    completed: int = 0
    state: str = "active"
    retreat_position: int | None = None
    has_native_wait: bool = False
    final_position: int | None = None
    gaps: list = field(default_factory=list)

    def observe(self, position, event):
        name = event["event"]
        _require(self.state != "done", "low command repeated work after final completion")
        _require(event["total"] == self.total, "low command changed logical tile geometry")
        if name == "low_resumed":
            _require(self.state == "paused" and event["launches"] == self.generation + 1 and
                     event["completed"] == self.completed,
                     "low continuation reset/replayed its counter or generation")
            self.gaps[-1]["resume_position"] = position
            self.gaps[-1]["resumed_generation"] = event["launches"]
            self.generation = event["launches"]
            self.state, self.retreat_position = "active", None
            return
        _require(event["launches"] == self.generation, "low transition changed generation")
        if name in ("retreat_requested", "nonpreemptible_wait"):
            _require(self.state == "active" and self.retreat_position is None and
                     not self.has_native_wait and event["completed"] == self.completed,
                     "repeated/inactive low retreat or unobserved live counter change")
            if name == "retreat_requested":
                _require(self.total > 0, "native-only command fabricated a retreat")
                self.retreat_position = position
            else:
                self.has_native_wait = True
        elif name == "low_physical_stop":
            _require(self.state == "active" and self.retreat_position is not None and
                     self.completed <= event["completed"] < self.total,
                     "low physical stop lacks a request or has invalid progress")
            self.completed, self.state = event["completed"], "paused"
            self.gaps.append(dict(
                low=self.command.copy(), issue_position=self.issue_position,
                retreat_position=self.retreat_position, stop_position=position,
                generation=self.generation, completed_blocks=self.completed,
                total_blocks=self.total,
            ))
        elif name == "low_logical_done":
            _require(self.state == "active" and event["completed"] == self.total,
                     "low final completion is paused or incomplete")
            self.completed, self.state, self.final_position = self.total, "done", position
        else:
            raise ValueError("low transition has no matching original issue")


class _PhysicalOwnership:
    """Ordering and opaque native reservations, never useful-compute witnesses.

    The central parser has already authenticated every complete lifecycle and
    grant/path. A returned source receipt is nevertheless single-use here: only
    the next exact issue/resume may exempt its own root batch, not another batch
    or an ordinary frontier. Reply publication/release does not retire a batch.
    """

    def __init__(self, low_priority, high_priority):
        self.priorities = (low_priority, high_priority)
        self.ordering = {}
        self.ordering_commands = []
        self.batches = {}
        self.native_intervals = []
        self.grants = {}
        self.paths = {}
        self.pending_source = None
        self.source_issues = []

    def semantic(self, position, record):
        if record.family == 2:
            if record.value("event") == 10:
                self.grants[record.serial] = record
                self.paths[record.serial] = []
            elif record.value("event") == 11:
                self.paths[record.value("grant")].append({
                    field: record.value(field) for field in (
                        "index", "source", "target", "event_object", "record_sequence",
                        "source_capture", "target_capture", "target_sequence")})
            elif record.value("event") == 12 and record.value("issue_state") == 1:
                _require(self.pending_source is None,
                         "required source receipt lacks its exact immediate issue/resume")
                grant = self.grants[record.value("grant")]
                self.pending_source = dict(
                    session=record.value("source"), sequence=record.value("command_sequence"),
                    grant_serial=grant.serial, result_position=position,
                    reservation=grant.value("reservation"), generation=grant.value("generation"),
                    cutoff=grant.value("cutoff"), root_source=grant.value("root_source"),
                    leaf_source=grant.value("leaf_source"), capture=grant.value("capture"),
                    priority=grant.value("priority"), path=self.paths[grant.serial].copy(),
                )
            return
        key = record.value("reservation"), record.value("generation")
        priority = record.value("priority")
        _require(priority in self.priorities, "native batch has an unexpected application priority")
        previous = self.batches.get(key)
        is_active = bool(record.value("has_physical_work") or record.value("open_group") or
                         record.value("unknown"))
        if previous is not None and previous["active"] and (
                not is_active or record.value("action") == 12):
            self.native_intervals[previous["interval"]]["stop_position"] = position
        if record.value("action") == 12:
            self.batches.pop(key, None)
            return
        interval = previous["interval"] if previous is not None else None
        if is_active and (previous is None or not previous["active"]):
            interval = len(self.native_intervals)
            self.native_intervals.append(dict(
                peer=record.value("peer"), priority=priority, reservation=key[0], generation=key[1],
                start_position=position, stop_position=None,
            ))
        self.batches[key] = dict(active=is_active, priority=priority, interval=interval)

    def before_issue(self, position, event):
        key = event["session"], event["sequence"]
        allowed_batch = None
        if self.pending_source is not None:
            receipt = self.pending_source
            _require(key == (receipt["session"], receipt["sequence"]),
                     "required source receipt does not name the next exact issue/resume")
            _require(receipt["result_position"] < position,
                     "required source issue precedes its actual returned receipt")
            _require(event["priority"] == receipt["priority"],
                     "required source issue changed its authenticated grant priority")
            allowed_batch = receipt["reservation"], receipt["generation"]
            self.source_issues.append(dict(receipt, issue_position=position, event=event["event"]))
            self.pending_source = None
        priority = event["priority"]
        _require(not any(other != priority for other in self.ordering.values()),
                 "ordinary issue/resume crosses an unretired unequal-priority ordering frontier")
        _require(not any(batch["active"] and batch["priority"] != priority and batch_key != allowed_batch
                         for batch_key, batch in self.batches.items()),
                 "ordinary issue/resume crosses an active unequal-priority native reservation")

    def ordering_event(self, position, event, operation):
        _require(operation in ORDERING_OPERATIONS and _zero_progress(event),
                 "ordering lifecycle fabricated useful-compute progress")
        key = event["session"], event["sequence"]
        if event["event"] == "event_issued":
            self.before_issue(position, event)
            self.ordering[key] = event["priority"]
            self.ordering_commands.append(dict(
                **_identity(event, operation), priority=event["priority"],
                issue_position=position, retirement_position=None,
            ))
        elif event["event"] == "event_retired":
            self.ordering.pop(key, None)
            for command in reversed(self.ordering_commands):
                if (command["session"], command["sequence"]) == key:
                    command["retirement_position"] = position
                    break

    def finish(self):
        _require(self.pending_source is None,
                 "required source receipt lacks its exact immediate issue/resume")
        _require(not self.ordering and not any(batch["active"] for batch in self.batches.values()),
                 "ordering or native physical ownership has no final healthy stop")


def _application_timeline(events, semantic):
    """Use typed ordering positions, not an earlier legacy completion annotation."""
    timeline = [(position, "ordinary", event) for position, event in events
                if event["event"] not in ORDERING_EVENTS]
    for record in semantic.records:
        if record.family == 3:
            name = next(name for name, value in ORDERING_EVENTS.items()
                        if value == record.value("event"))
            timeline.append((record.position, "ordinary", dict(
                event=name, session=record.value("connection"), sequence=record.value("sequence"),
                priority=record.value("priority"), status=record.value("status"),
                completed=0, total=0, launches=0)))
        elif (record.family == 1 and record.value("event") == 16 and
              record.value("subject") == 2) or (
                record.family == 2 and record.value("event") in (10, 11, 12)):
            timeline.append((record.position, "semantic", record))
    return sorted(timeline, key=lambda item: item[0])


def analyze_application_priority(content, offset, low_priority=0, high_priority=100,
                                 *, demand_position=None):
    """Return JSON-compatible partial handoff evidence for a complete service log.

    All known records, setup statuses and low lifecycles are validated. Only
    chains whose original low issue is at/after the explicit measurement offset
    can qualify. A session-wide high physical-stop record proves completion of
    that session's preceding issues, not any future command or other session.
    All actual high issues require ordered physical-stop proof for every active
    low frontier, including native and memory work. Low issue/resume requires
    matching completion of every preceding high frontier. Admissions and context
    bindings are not physical issues. Missing ordered proof fails closed; it
    does not itself establish a runtime scheduling violation. Older logs can
    omit short high completions; those remain diagnostic only if no subsequent
    low work crosses their unproven frontier. Zero-progress pauses and memory-
    only high gaps cannot qualify. Original-only compute has no persistent
    counters and cannot supply useful high computation from its opcode alone.
    Typed ordering and native batch ownership participate in physical ordering,
    but never compute/handoff counts. Only the exact returned source issue may
    exempt its validated root batch; ordinary high frontiers and other native
    batches remain blocking. Diagnostic interval lists cover the complete log.
    Invalid transitions or incomplete low
    lifecycles raise ValueError.
    Handoff counts are per low-command stopped interval, not unique global
    scheduler switches when several low epochs share one high-work interval.
    """
    _require(isinstance(content, str), "audit content must be text")
    _require(type(offset) is int and 0 < offset < len(content),
             "missing post-ready application phase boundary")
    _require(type(low_priority) is int and type(high_priority) is int and
             0 <= low_priority < high_priority, "distinct ordered application priorities required")
    admissions, events, pipelines, semantic = _parse_audit_model(content)
    publications = _validate_block_completions(content, admissions, events, semantic)
    if demand_position is not None:
        _require(type(demand_position) is int and offset < demand_position < len(content) and
                 content[demand_position - 1] == '\n',
                 'external demand requires a complete post-measurement log boundary')
    lows, highs, pending_high, priorities = {}, {}, {}, {}
    active_lows = set()
    issued_positions = {}
    unmatched_high_stops = []
    ownership = _PhysicalOwnership(low_priority, high_priority)

    for position, kind, event in _application_timeline(events, semantic):
        if kind == "semantic":
            ownership.semantic(position, event)
            continue
        name, session, sequence = event["event"], event["session"], event["sequence"]
        _require(session > 0, "audit event has no session identity")
        if name in LOW_EVENTS:
            expected_priority = low_priority
        elif name in HIGH_EVENTS:
            expected_priority = high_priority
        elif name in ORDERING_EVENTS:
            _require(event["priority"] in (low_priority, high_priority),
                     "ordering event has an unexpected application priority")
            expected_priority = event["priority"]
        else:
            raise ValueError(f"unsupported application priority event: {name}")
        _require(event["priority"] == expected_priority,
                 "event role does not match its trusted application priority")
        _require(priorities.setdefault(session, expected_priority) == expected_priority,
                 "one session changed application priority/role")
        if name in ("low_bound", "native_bound", "high_physical_done"):
            _require(sequence == 0 and _zero_progress(event),
                     "session-level audit record has command progress")
            if name == "high_physical_done":
                pending = pending_high.pop(session, [])
                if not pending:
                    unmatched_high_stops.append(dict(session=session, position=position))
                for key in pending:
                    highs[key]["physical_done_position"] = position
            continue

        key = session, sequence
        admitted = admissions.get(key)
        _require(sequence > 0 and admitted is not None and
                 (admitted[1] <= position if name == "event_accepted" else admitted[1] < position),
                 "missing or late typed application command admission")
        operation = admitted[0]
        _require(operation in COMPUTE_OPERATIONS | MEMORY_OPERATIONS | ORDERING_OPERATIONS,
                 "unsupported application command type")
        if name in ORDERING_EVENTS:
            if name == "event_issued":
                if event["priority"] == low_priority:
                    _require(not pending_high, "low ordering issue while high completion unproven")
                else:
                    _require(not active_lows, "high ordering issue before proven low physical stop")
            ownership.ordering_event(position, event, operation)
            continue
        _require(operation not in ORDERING_OPERATIONS,
                 "ordering command fabricated a compute lifecycle")
        if name in ("low_issued", "high_issued"):
            _require(key not in issued_positions, "command issued more than once")
            _validate_first_issue(event, operation)
            ownership.before_issue(position, event)
            issued_positions[key] = position
            if name == "low_issued":
                _require(not pending_high, "low issue while high completion unproven")
                lows[key] = _LowCommand(_identity(event, operation), position,
                                        event["total"], event["launches"])
                active_lows.add(key)
            else:
                _require(not active_lows, "high issue before proven low physical stop")
                highs[key] = dict(**_identity(event, operation), issue_position=position,
                                  physical_done_position=None)
                pending_high.setdefault(session, []).append(key)
        else:
            _require(key in lows, "low transition has no original low issue")
            if name == "low_resumed":
                _require(not pending_high, "low resume while high completion unproven")
                ownership.before_issue(position, event)
            lows[key].observe(position, event)
            if lows[key].state == "active":
                active_lows.add(key)
            else:
                active_lows.discard(key)

    ownership.finish()
    for low in lows.values():
        _require(low.state == "done", "missing final exactly-once low completion")
    # An optional pair witness is never required for an application, but any
    # witness present must name two actually issued original kernel commands.
    for position, (session, first, second) in pipelines:
        _require(first != second, "pipeline repeats one command")
        for sequence in (first, second):
            key = session, sequence
            _require(key in admissions and admissions[key][0] == 11 and
                     key in issued_positions and issued_positions[key] < position,
                     "pipeline does not name two preceding typed kernel issues")
            terminal = (lows[key].final_position if key in lows else
                        highs[key]["physical_done_position"])
            _require(terminal is None or terminal > position,
                     "pipeline claims an already physically completed command")

    handoffs, unqualified = [], []
    for low in lows.values():
        if low.issue_position < offset:
            continue
        for gap in low.gaps:
            evidence = dict(gap, final_position=low.final_position)
            inside = [high.copy() for high in highs.values()
                      if gap["stop_position"] < high["issue_position"] < gap["resume_position"]]
            compute = [high for high in inside
                       if high["operation"] in PERSISTENT_COMPUTE_OPERATIONS]
            reasons = []
            if low.command["operation"] not in PERSISTENT_COMPUTE_OPERATIONS:
                reasons.append("low command is not compute")
            if gap["completed_blocks"] == 0:
                reasons.append("no positive stopped compute progress")
            if not compute:
                reasons.append("no typed high compute eligible for handoff evidence inside the stopped interval")
                if any(high["operation"] in ORIGINAL_ONLY_COMPUTE_OPERATIONS for high in inside):
                    reasons.append("original-only compute opcode does not prove useful GPU work")
            if any(high["physical_done_position"] is None or
                   high["physical_done_position"] >= gap["resume_position"] for high in inside):
                reasons.append("high work inside the gap lacks matching physical completion before resume")
            evidence["high_commands"] = inside
            if reasons:
                unqualified.append(dict(evidence, reasons=reasons))
            else:
                handoffs.append(evidence)

    low_commands = [dict(low.command, issue_position=low.issue_position,
                         final_position=low.final_position, total_blocks=low.total,
                         final_generation=low.generation)
                    for low in lows.values() if low.issue_position >= offset and
                    low.command["operation"] in COMPUTE_OPERATIONS]
    high_commands = [high.copy() for high in highs.values() if high["issue_position"] >= offset and
                     high["operation"] in COMPUTE_OPERATIONS]
    result = dict(
        handoff_count=len(handoffs), handoffs=handoffs, unqualified_gaps=unqualified,
        low_session_ids=sorted(session for session, priority in priorities.items()
                               if priority == low_priority),
        high_session_ids=sorted(session for session, priority in priorities.items()
                                if priority == high_priority),
        low_compute_commands=low_commands, high_compute_commands=high_commands,
        compute_operation_classes=compute_operation_classes(),
        low_original_only_compute_commands=[command.copy() for command in low_commands
                                            if command["operation"] in ORIGINAL_ONLY_COMPUTE_OPERATIONS],
        high_original_only_compute_commands=[command.copy() for command in high_commands
                                             if command["operation"] in ORIGINAL_ONLY_COMPUTE_OPERATIONS],
        unproven_high_commands=[high.copy() for high in highs.values()
                                if high["physical_done_position"] is None],
        unmatched_high_physical_stops=unmatched_high_stops,
        ordering_commands=ownership.ordering_commands,
        native_physical_intervals=ownership.native_intervals,
        required_source_issues=ownership.source_issues,
        partial_handoff_component_verified=bool(handoffs),
        full_application_physical_priority_qualified=False,
        pre_demand_compute_observed=False,
        missing_evidence=[MISSING_PRE_DEMAND],
    )
    if publications or demand_position is not None:
        result['block_completion_observations'] = publications
    if demand_position is not None:
        witnessed = []
        high_sessions = set(result['high_session_ids'])
        for gap in handoffs:
            key = gap['low']['session'], gap['low']['sequence']
            witnesses = [publication for publication in publications
                         if (publication['session'], publication['sequence']) == key and
                         offset <= gap['issue_position'] < publication['position'] and
                         publication['end_position'] <= demand_position <= gap['retreat_position'] and
                         gap['generation'] == 1 and publication['total'] == gap['total_blocks']]
            if len(witnesses) != 1:
                continue
            publication = witnesses[0]
            # A high SDK admission or issue before this external release would
            # make the claimed pre-demand boundary ineligible, even if a later
            # complete handoff exists. Keep setup outside measurement separate.
            if (any(session in high_sessions and offset <= point < demand_position
                    for (session, _), (_, point) in admissions.items()) or
                    any(event['session'] in high_sessions and offset <= point < demand_position
                        and event['event'] in ('high_issued', 'event_issued')
                        for point, event in events)):
                continue
            witnessed.append(dict(low=gap['low'].copy(), publication=publication,
                                   demand_position=demand_position,
                                   retreat_position=gap['retreat_position'],
                                   stop_position=gap['stop_position'], resume_position=gap['resume_position']))
        result['pre_demand_handoff_components'] = witnessed
        result['pre_demand_compute_observed'] = bool(witnessed)
        if witnessed:
            result['missing_evidence'] = []
    return result
