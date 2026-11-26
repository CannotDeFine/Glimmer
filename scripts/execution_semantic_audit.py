"""Pure, bounded decoding of typed execution evidence; never GPU qualification.

Positions use Python text offsets, matching the existing priority-audit APIs.
Unrelated log lines are not evidence. A line containing our marker must decode
in full, so malformed or appended records cannot disappear during filtering.
"""

from dataclasses import dataclass
import json
import re


MAX_CONTENT_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 64 * 1024
MAX_RECORDS = 1_000_000
MAX_U64 = (1 << 64) - 1
MIN_I64 = -(1 << 63)
MARKER = "Execution semantic audit"
_FRAME = re.compile(
    r"[ \t]*(?:\[[^\]\r\n]*\][ \t]*)*"
    r"(?P<record>Execution semantic audit: (?P<payload>\{[^\r\n]*\}))"
)
_COMMON_FIELDS = frozenset(("v", "service", "serial", "family"))
_FAMILIES = frozenset((1, 2, 3, 4))
_ACTOR_FIELDS = frozenset("""
actor_serial event subject action binding id peer priority caller0 caller1 group
sequence ordinary_sequence cleanup_parent origin_source reservation generation
cutoff input_reservation input_generation input_cutoff native_sequence
native_operation native_kind ordinary_opcode connection wire_sequence attempt
source stream capture command_sequence related index count allocation ticket depth
stage previous_stage phase outcome disposition host_origin status carrier
original_status async_status native_status completion_status library_status
accepted sequence_consumed has_original has_async has_library has_host
has_output_written output_written reply_ready reply_released enqueue_complete
physical_complete has_physical_work open_group unknown retired bound consumed active_call
ready_continuation
caller_primary_domain has_context before_kind before_identity before_context before_device
after_kind after_identity after_context after_device
communicator_domain communicator_context communicator_object output_contract
""".split())
_ACTOR_FLAGS = frozenset("""
accepted sequence_consumed has_original has_async has_library has_host
has_output_written output_written reply_ready reply_released enqueue_complete
physical_complete has_physical_work open_group unknown retired bound consumed active_call
ready_continuation
has_context
""".split())
_ACTOR_STATUS_FIELDS = frozenset("""
status carrier original_status async_status native_status completion_status library_status
""".split())
_ACTOR_ENUM_MAXIMA = {
    "event": 20, "subject": 6, "action": 14, "binding": 6,
    "native_kind": 11, "phase": 6, "outcome": 5, "disposition": 4,
    "host_origin": 3, "stage": 11, "previous_stage": 11,
    "output_contract": 2,
}
_COMMUNICATOR_FIELDS = ("communicator_domain", "communicator_context", "communicator_object")
_GROUP_MEMBER_KINDS = frozenset((2, 3, 5, 6, 9, 10, 11))
_RUNTIME_FIELDS = {
    1: frozenset(("pid", "incarnation0", "incarnation1")),
    2: frozenset(),
    3: frozenset(("reason",)),
    10: frozenset("""
        policy peer priority reservation generation cutoff target_kind target action
        mode root_source leaf_source capture edges path_kind
        """.split()),
    11: frozenset("""
        grant index source target event_object record_sequence source_capture
        target_capture target_sequence
        """.split()),
    12: frozenset("""
        grant status issue_state command_sequence native_status completion_status
        has_library library_status source
        """.split()),
    14: frozenset("""
        grant continuation attempt status outcome has_wire wire_status has_original
        original_status receipt_sequence mailbox_sequence capture source
        """.split()),
}
_ORDERING_FIELDS = frozenset("""
event connection priority sequence opcode capture event_source event_object
record_sequence event_capture status
""".split())
_PEER_FIELDS = frozenset("""
event connection peer pid start uid priority role policy kernel_mode effective retained
""".split())
_STATUS_FIELDS = _ACTOR_STATUS_FIELDS | frozenset(("wire_status",))


@dataclass(frozen=True)
class SemanticRecord:
    """An immutable record; numeric identities never denote native addresses."""

    position: int
    end_position: int
    line: int
    service: int
    serial: int
    family: int
    fields: tuple[tuple[str, int | str], ...]

    def value(self, name):
        for key, value in self.fields:
            if key == name:
                return value
        raise KeyError(name)

    @property
    def actor_statuses(self):
        """Keep native/async/carrier domains distinct for typed actor consumers."""
        _require(self.family == 1, "status projection requires an actor record")
        return _actor_statuses(self)


@dataclass(frozen=True)
class SemanticAudit:
    """Validated semantic evidence, separate from useful-compute qualification."""

    records: tuple[SemanticRecord, ...]
    service: int | None
    is_complete: bool
    unresolved: tuple[str, ...]

    def family(self, number):
        return tuple(record for record in self.records if record.family == number)


@dataclass(frozen=True)
class SemanticStatuses:
    """Different result domains; an original seven is never carrier success."""

    status: int
    carrier: int
    original: int | None
    asynchronous: int | None
    library: int | None
    native: int
    completion: int


def _actor_statuses(record):
    return SemanticStatuses(
        status=record.value("status"), carrier=record.value("carrier"),
        original=record.value("original_status") if record.value("has_original") else None,
        asynchronous=record.value("async_status") if record.value("has_async") else None,
        library=record.value("library_status") if record.value("has_library") else None,
        native=record.value("native_status"), completion=record.value("completion_status"),
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(token):
    digits = token[1:] if token.startswith("-") else token
    _require(len(digits) <= 20, "semantic integer exceeds its bounded representation")
    value = int(token)
    _require(MIN_I64 <= value <= MAX_U64, "semantic integer is outside 64-bit bounds")
    return value


def _not_integer(_token):
    raise ValueError("semantic numbers must be finite integers")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate semantic JSON key")
        result[key] = value
    return result


def _decode_records(content):
    """Decode framing and common identities; family schemas validate next."""
    _require(type(content) is str, "semantic audit content must be text")
    _require(len(content) <= MAX_CONTENT_BYTES, "semantic audit exceeds input bound")
    try:
        byte_count = len(content.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise ValueError("semantic audit contains invalid Unicode") from error
    _require(byte_count <= MAX_CONTENT_BYTES, "semantic audit exceeds input bound")
    records = []
    service = None
    offset = 0
    for number, original in enumerate(content.splitlines(keepends=True), 1):
        line = original.removesuffix("\n").removesuffix("\r")
        if MARKER in line:
            match = _FRAME.fullmatch(line)
            _require(line.count(MARKER) == 1 and match is not None,
                     f"malformed semantic audit framing at line {number}")
            payload = match.group("payload")
            _require(len(payload.encode("utf-8")) <= MAX_RECORD_BYTES,
                     "semantic record exceeds input bound")
            _require(len(records) < MAX_RECORDS, "too many semantic records")
            try:
                values = json.loads(
                    payload, object_pairs_hook=_unique_object,
                    parse_int=_integer, parse_float=_not_integer,
                    parse_constant=_not_integer,
                )
            except (ValueError, RecursionError) as error:
                raise ValueError(f"invalid semantic JSON at line {number}: {error}") from error
            _require(type(values) is dict and _COMMON_FIELDS <= values.keys(),
                     "missing semantic common fields")
            for key, value in values.items():
                _require(type(key) is str and bool(re.fullmatch(r"[a-z][a-z0-9_]*", key)),
                         "invalid semantic field name")
                if key in ("event", "subject") and type(value) is str:
                    _require(type(value) is str and
                             bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value)),
                             "invalid semantic enum tag")
                else:
                    _require(type(value) is int, "semantic fields must contain integers")
            _require(values["v"] == 1, "unsupported semantic schema version")
            _require(values["family"] in _FAMILIES, "unknown semantic record family")
            _require(0 < values["service"] <= MAX_U64, "invalid semantic service identity")
            _require(values["serial"] == len(records) + 1,
                     "semantic service serial is not contiguous from one")
            if service is None:
                service = values["service"]
            _require(values["service"] == service, "mixed semantic service incarnations")
            records.append(SemanticRecord(
                position=offset + match.start("record"),
                end_position=offset + match.end("record"),
                line=number, service=service, serial=values["serial"],
                family=values["family"], fields=tuple(values.items()),
            ))
        offset += len(original)
    return tuple(records)


def _validate_actor_record(record):
    values = dict(record.fields)
    _require(values.keys() == _COMMON_FIELDS | _ACTOR_FIELDS,
             "missing or unknown actor semantic field")
    for name in _ACTOR_FIELDS:
        value = values[name]
        _require(type(value) is int, "actor semantic fields must be numeric")
        if name in _ACTOR_STATUS_FIELDS:
            _require(-(1 << 31) <= value < (1 << 31), "actor status exceeds signed32")
        else:
            _require(0 <= value <= MAX_U64, "actor identity exceeds unsigned64")
        if name in _ACTOR_FLAGS:
            _require(value in (0, 1), "actor flag is not zero or one")
        if name in _ACTOR_ENUM_MAXIMA:
            _require(value <= _ACTOR_ENUM_MAXIMA[name], "unknown actor enum value")
    _require(values["event"] > 0 and values["subject"] > 0,
             "actor event and subject must be active enum values")
    if values["subject"] == 1:
        allowed = {"actor_serial", "event", "subject"}
        if values["event"] == 17:
            allowed.add("peer")
        if values["event"] == 19:
            allowed.add("retired")
        _require(values["event"] in (1, 17, 18, 19, 20) and
                 all(not values[name] for name in _ACTOR_FIELDS - allowed),
                 "actor lifecycle contains inactive payload or wrong event")
    _require(values["actor_serial"] > 0, "actor serial must be positive")
    _require(values["priority"] < (1 << 32) and values["ordinary_opcode"] < (1 << 32),
             "actor unsigned32 field exceeds bounds")
    for presence, field in (("has_original", "original_status"),
                            ("has_async", "async_status"),
                            ("has_library", "library_status")):
        _require(values[presence] or values[field] == 0,
                 "inactive actor status is not canonical zero")
    for side in ("before", "after"):
        kind = values[f"{side}_kind"]
        identity, context, device = (values[f"{side}_{name}"]
                                     for name in ("identity", "context", "device"))
        if not values["has_context"]:
            _require(not (kind or identity or context or device),
                     "inactive context transition is not canonical zero")
        elif kind == 1:
            _require(not (identity or context or device), "NULL context has nonnull identity")
        else:
            _require(kind in (2, 3) and identity > 0 and context > 0 and 0 < device <= (1 << 31),
                     "context transition lacks a valid symbolic owner/device")
    _require(values["has_output_written"] or not values["output_written"],
             "inactive output-write observation is not canonical zero")
    communicator = tuple(values[name] for name in _COMMUNICATOR_FIELDS)
    _require(not any(communicator) or all(communicator), "partial communicator identity")
    if values["subject"] == 3:
        _require(values["output_contract"] in (1, 2), "native output contract is absent")
        _require(values["native_kind"] not in (1, 7, 8) or not any(communicator),
                 "noncommunicator call contains communicator identity")
    else:
        _require(not any(communicator) and not values["output_contract"],
                 "nonnative subject contains native communicator/output contract")
    if not values["has_host"] and not (values["event"] == 7 or
                                       (values["event"] == 4 and values["action"] == 8)):
        _require(not values["host_origin"] and not values["completion_status"],
                 "inactive host result is not canonical zero")
    if values["subject"] == 4:
        _require(values["stage"] <= 9 and values["previous_stage"] <= 9,
                 "unknown ordinary continuation stage")


def _validate_record(record):
    if record.family == 1:
        _validate_actor_record(record)
        return
    values = dict(record.fields)
    event = values.get("event")
    _require(type(event) is int, "semantic event must be numeric")
    if record.family == 2:
        _require(event in _RUNTIME_FIELDS, "unknown runtime semantic event")
        expected = _COMMON_FIELDS | {"event"} | _RUNTIME_FIELDS[event]
        if event == 1 and "native_audit_schema" in values:
            _require(values["native_audit_schema"] == 2, "unknown native audit schema")
            expected = expected | {"native_audit_schema"}
    elif record.family == 3:
        _require(1 <= event <= 4, "unknown ordering semantic event")
        expected = _COMMON_FIELDS | _ORDERING_FIELDS
    else:
        _require(1 <= event <= 3, "unknown peer semantic event")
        expected = _COMMON_FIELDS | _PEER_FIELDS
    _require(values.keys() == expected, "missing or unknown semantic event field")
    for key, value in values.items():
        _require(type(value) is int, "semantic payload must be numeric")
        if key in _STATUS_FIELDS:
            _require(-(1 << 31) <= value < (1 << 31), "semantic status exceeds signed32")
        else:
            _require(0 <= value <= MAX_U64, "semantic identity exceeds unsigned64")
    for key in ("priority", "uid"):
        if key in values:
            _require(values[key] < (1 << 32), "semantic unsigned32 exceeds bounds")
    if record.family == 2:
        if event == 10:
            _require(values["target_kind"] in (1, 2) and values["mode"] in (1, 2, 3, 4),
                     "unknown grant target or mode")
            _require(values["action"] in (3, 5), "runtime grant is not an execution decision")
            _require(values["path_kind"] in (0, 1), "unknown runtime dependency path kind")
            _require(not values["path_kind"] or (values["mode"] == 2 and values["edges"] > 0),
                     "parent rejection lacks its actual foreign edge path")
        if event == 12:
            _require(values["issue_state"] in (0, 1, 2, 3), "unknown source issue state")
            _require(values["has_library"] in (0, 1), "invalid library presence")
            _require(values["has_library"] or values["library_status"] == 0,
                     "inactive library status is nonzero")
        if event == 14:
            _require(1 <= values["outcome"] <= 5, "unknown handler outcome")
            for flag, status in (("has_wire", "wire_status"),
                                 ("has_original", "original_status")):
                _require(values[flag] in (0, 1), "invalid handler status presence")
                _require(values[flag] or values[status] == 0,
                         "inactive handler status is nonzero")
    elif record.family == 3:
        _require(values["opcode"] in (48, 52, 54), "unknown ordering opcode")
        for key in ("connection", "sequence", "capture", "event_source", "event_object"):
            _require(values[key] > 0, "ordering identity is zero")
    else:
        _require(values["role"] in (0, 1, 2, 3) and values["kernel_mode"] in (0, 1, 2),
                 "unknown peer role or kernel mode")
        _require(values["effective"] in (0, 1) and values["retained"] in (0, 1),
                 "invalid peer flag")
        for key in ("connection", "pid", "start"):
            _require(values[key] > 0, "peer identity is zero")


class _Validation:
    """Owner-order state machines; no callbacks, native imports or inference by PID."""

    def __init__(self):
        self.begun = False
        self.ended = False
        self.actor_begun = False
        self.actor_stopped = False
        self.actor_serial = 0
        self.connections = {}
        self.peers = {}
        self.ordering = {}
        self.grants = {}
        self.entities = {}
        self.removed = set()
        self.batches = {}
        self.parents = {}
        self.tickets = {}
        self.sequences = {}
        self.native_sequences = {}
        self.depths = {}
        self.actor_grants = {}
        self.attachments = {}
        self.bindings = set()
        self.native_operations = {}
        self.problems = set()
        self.closed = False
        self.disconnected = set()
        self.parent_history = {}
        self.async_pending = set()
        self.process_priorities = {}
        self.caller_batches = {}
        self.caller_groups = {}
        self.last_groups = {}
        self.open_runtime_grants = {}
        self.source_groups = {}
        self.communicators = {}
        self.group_members = {}

    def apply(self, record):
        _validate_record(record)
        values = dict(record.fields)
        _require(not self.ended, "semantic record appears after service end")
        if not self.begun:
            _require(record.family == 2 and values["event"] == 1 and record.serial == 1,
                     "semantic capture lacks its initial service begin")
        if record.family == 1:
            self.actor(record, values)
        elif record.family == 2:
            self.runtime(record, values)
        elif record.family == 3:
            self.event(record, values)
        else:
            self.peer(values)

    def peer(self, values):
        event, connection = values["event"], values["connection"]
        identity = tuple(values[key] for key in
                         ("pid", "start", "uid", "priority", "role", "policy",
                          "kernel_mode", "effective"))
        priority = self.process_priorities.setdefault(identity[:3], values["priority"])
        _require(priority == values["priority"], "authenticated process priority changed")
        _require(values["role"] == 0 or values["effective"] == 0,
                 "nonsource peer claims configured compute mode")
        if event == 1:
            _require(connection not in self.connections, "repeated peer connection binding")
            self.connections[connection] = dict(identity=identity, peer=values["peer"],
                                                disconnected=False)
        else:
            _require(connection in self.connections, "orphan peer transition")
            bound = self.connections[connection]
            _require(bound["identity"] == identity, "peer binding changed identity or policy")
            _require(not bound["disconnected"], "repeated transition after peer disconnect")
            if event == 2:
                _require(values["peer"] == bound["peer"], "disconnect changed native peer")
                bound["disconnected"] = True
            else:
                _require(values["peer"] > 0 and bound["peer"] in (0, values["peer"]),
                         "native peer association changed ownership")
                _require(bound["peer"] == 0, "repeated native peer association")
                bound["peer"] = values["peer"]
        if values["peer"]:
            peer_identity = identity[:4]
            old = self.peers.setdefault(values["peer"], peer_identity)
            _require(old == peer_identity, "native peer reused by another authenticated identity")

    def connection(self, connection, priority=None, peer=None):
        _require(connection in self.connections, "unbound source connection")
        bound = self.connections[connection]
        if priority is not None:
            _require(bound["identity"][3] == priority, "source priority changed")
        if peer is not None:
            _require(peer in self.peers and bound["identity"][:4] == self.peers[peer],
                     "source belongs to a different authenticated peer")
        return bound

    def source_connection(self, connection, priority=None, peer=None):
        bound = self.connection(connection, priority, peer)
        _require(bound["identity"][4] == 0, "dependency owner is not an actual source connection")
        return bound

    def event(self, record, values):
        self.source_connection(values["connection"], values["priority"])
        source = self.source_connection(values["event_source"])
        target = self.connections[values["connection"]]
        _require(source["identity"][:3] == target["identity"][:3],
                 "ordering edge crosses authenticated peers")
        key = values["connection"], values["sequence"]
        identity = tuple(values[name] for name in
                         ("priority", "opcode", "capture", "event_source", "event_object",
                          "record_sequence", "event_capture"))
        event = values["event"]
        if event == 1:
            _require(key not in self.ordering, "duplicate ordering admission")
            if values["opcode"] == 48:
                _require(values["event_source"] == values["connection"] and
                         values["record_sequence"] == values["sequence"] and
                         values["event_capture"] == values["capture"],
                         "record operation does not identify its actual generation")
            elif values["record_sequence"]:
                source_key = values["event_source"], values["record_sequence"]
                _require(source_key in self.ordering, "wait lacks its original accepted event record")
                original = self.ordering[source_key]["identity"]
                _require(original[1] == 48 and
                         original[4] == values["event_object"] and
                         original[2] == values["event_capture"] > 0,
                         "wait substituted a different event object/generation/capture")
            else:
                _require(values["event_capture"] == 0,
                         "initial unrecorded event claims a recorded capture")
            _require(values["opcode"] != 52 or values["event_source"] == values["connection"],
                     "local wait names a foreign source")
            self.ordering[key] = dict(identity=identity, issued=False, retired=False,
                                      foreign=False, position=record.position)
        else:
            _require(key in self.ordering, "orphan ordering transition")
            state = self.ordering[key]
            _require(state["identity"] == identity, "ordering changed immutable generation")
            _require(not state["retired"], "ordering transition follows retirement")
            if event == 2:
                _require(not state["issued"], "repeated ordering issue")
                state["issued"] = True
            elif event == 3:
                _require(state["issued"] or values["status"] != 0,
                         "successful ordering retirement lacks original issue")
                state["retired"] = True
            else:
                _require(values["opcode"] == 54 and not state["foreign"] and
                         not state["issued"], "late or repeated foreign ordering annotation")
                state["foreign"] = True
        if values["status"]:
            self.problems.add(f"ordering status {values['status']} for {key}")

    def runtime(self, record, values):
        event = values["event"]
        if event == 1:
            _require(not self.begun and values["pid"] > 0 and
                     (values["incarnation0"] or values["incarnation1"]),
                     "invalid or duplicate service begin")
            self.begun = True
            return
        if event == 2:
            _require(not self.pending(), "service end does not prove unresolved ownership stopped")
            self.ended = True
            return
        if event == 3:
            raise ValueError("semantic audit reported lost evidence")
        if event == 10:
            for name in ("policy", "peer", "reservation", "generation", "cutoff", "target"):
                _require(values[name] > 0, "runtime grant identity is zero")
            _require(values["peer"] in self.peers and
                     self.peers[values["peer"]][3] == values["priority"],
                     "runtime grant lacks authenticated peer priority")
            key = (3 if values["target_kind"] == 1 else 4, values["target"])
            _require(key in self.entities, "runtime grant names an unadmitted actor target")
            _require(key not in self.open_runtime_grants, "target has an unconsumed runtime grant")
            entity = self.entities[key]
            _require(tuple(values[name] for name in ("peer", "reservation", "generation", "cutoff"))
                     == tuple(entity["last"][name] for name in
                              ("peer", "reservation", "generation", "cutoff")),
                     "runtime grant is stale or belongs to another batch")
            if values["mode"] in (2, 4):
                self.source_connection(values["root_source"], peer=values["peer"])
                self.source_connection(values["leaf_source"], peer=values["peer"])
                _require(values["edges"] or values["leaf_source"] == values["root_source"],
                         "zero-edge source grant changed its exact root into another leaf")
                known_source = any(binding[1] == values["root_source"]
                                   for binding in entity["bindings"]) or any(
                    attachment[0] == key and attachment[2] == values["root_source"]
                    for attachment in self.attachments)
                _require(known_source, "source grant lacks its exact admitted root attachment")
                if not values["edges"]:
                    if key[0] == 3:
                        _require(values["capture"] == 0,
                                 "direct native request invented a frontier capture")
                    else:
                        stage = entity["last"]["stage"]
                        binding_kind = 1 if stage in (1, 3) else 2 if stage in (4, 6) else 0
                        captures = [binding[2] for binding in entity["bindings"]
                                    if binding[0] == binding_kind and
                                    binding[1] == values["root_source"]]
                        _require(binding_kind and captures == [values["capture"]] and
                                 values["capture"] > 0,
                                 "continuation source grant changed its active stage-specific cut")
            else:
                _require(not any(values[name] for name in
                                 ("root_source", "leaf_source", "capture", "edges")),
                         "native/handler grant invented a source dependency path")
            self.grants[record.serial] = dict(values=values, edges=[], result=None, handler=None,
                                              position=record.position, actor_seen=False,
                                              actor_finished=False)
            self.open_runtime_grants[key] = record.serial
            return
        _require(values["grant"] in self.grants, "runtime result/path lacks its original grant")
        grant = self.grants[values["grant"]]
        root = grant["values"]
        _require(grant["result"] is None and grant["handler"] is None,
                 "runtime grant was already consumed")
        if event == 11:
            _require(values["index"] == len(grant["edges"]) and
                     len(grant["edges"]) < root["edges"], "runtime path order/count mismatch")
            previous = root["root_source"] if not grant["edges"] else grant["edges"][-1]["source"]
            _require(values["target"] == previous, "runtime path is not a contiguous dependency")
            for name in ("source", "target"):
                self.source_connection(values[name], peer=root["peer"])
            key = values["target"], values["target_sequence"]
            _require(key in self.ordering, "runtime path lacks accepted exact ordering target")
            ordering = self.ordering[key]
            identity = ordering["identity"]
            _require(identity[1] == 54 and ordering["foreign"] and
                     (identity[2], identity[3], identity[4], identity[5], identity[6]) ==
                     (values["target_capture"], values["source"], values["event_object"],
                      values["record_sequence"], values["source_capture"]),
                     "runtime path changed the captured foreign generation")
            _require(not any(edge["source"] == values["source"] and
                             edge["source_capture"] == values["source_capture"]
                             for edge in grant["edges"]), "cyclic runtime dependency path")
            grant["edges"].append(values)
        elif event == 12:
            _require(len(grant["edges"]) == root["edges"], "runtime grant path is truncated")
            _require(values["source"] == root["leaf_source"], "runtime result changed exact leaf")
            if grant["edges"]:
                leaf_field = "target" if root["path_kind"] else "source"
                _require(grant["edges"][-1][leaf_field] == root["leaf_source"],
                         "runtime path does not end at the selected leaf")
                _require(root["path_kind"] or
                         grant["edges"][-1]["source_capture"] == root["capture"],
                         "runtime grant changed the exact selected leaf capture")
            if values["issue_state"] in (1, 2):
                _require(root["mode"] == 2 and values["command_sequence"] > 0,
                         "actual source issue lacks its exact progress grant")
            elif values["issue_state"] == 0:
                _require(values["command_sequence"] == 0,
                         "nonissue runtime result fabricated a command")
            if root["mode"] != 2:
                _require(values["issue_state"] == 0 and not values["has_library"],
                         "native/handler grant fabricated ordinary SDK issue")
            actor_result = grant.get("actor_result")
            if (root["target_kind"] == 1 and root["mode"] in (2, 4) and not root["edges"] and
                    (values["status"] == 0 or values["command_sequence"] or
                     values["issue_state"] in (1, 2))):
                expected_action = 8 if root["mode"] == 2 else 6
                _require(grant["actor_seen"] and grant["actor_finished"] and
                         actor_result is not None and actor_result["action"] == expected_action,
                         "direct request source result lacks its exact actor Turn proof")
            if actor_result is not None and actor_result["action"] == 8:
                _require(actor_result["outcome"] == values["issue_state"] + 1 and
                         all(actor_result[name] == values[name] for name in
                             ("command_sequence", "native_status", "completion_status",
                              "has_library", "library_status")),
                         "runtime result changed the actual actor source issue receipt")
            if root["mode"] == 1 and values["status"] == 0:
                _require(grant["actor_seen"] and grant["actor_finished"],
                         "successful native grant lacks actual actor execution")
            if root["path_kind"]:
                if values["issue_state"] == 2:
                    _require(values["native_status"] != 0 and values["command_sequence"] ==
                             grant["edges"][-1]["target_sequence"],
                             "parent rejection changed the actual rejected wait")
                else:
                    self.problems.add(f"parent rejection not actually issued for {values['grant']}")
            grant["result"] = values
            del self.open_runtime_grants[(3 if root["target_kind"] == 1 else 4, root["target"])]
            if (values["status"] not in (0, 600) or values["completion_status"] or
                    values["native_status"] or
                    (values["has_library"] and values["library_status"])):
                self.problems.add(f"failed runtime grant {values['grant']}")
        else:
            _require(root["mode"] == 3 and grant["handler"] is None and
                     values["continuation"] == root["target"] and values["attempt"] > 0,
                     "orphan or repeated handler outcome")
            key = 4, values["continuation"]
            _require(key in self.actor_grants and self.actor_grants[key]["grant"] == values["grant"]
                     and self.entities[key]["last"]["attempt"] == values["attempt"],
                     "handler outcome lacks its exact armed actor attempt")
            grant["handler"] = values
            del self.open_runtime_grants[(4, root["target"])]
            if values["outcome"] == 1:
                _require(values["status"] == 600 and not values["has_wire"] and
                         not values["receipt_sequence"] and not values["mailbox_sequence"],
                         "Deferred outcome claims accepted work or reply")
            if values["outcome"] == 5:
                self.problems.add("uncertain ordinary handler outcome")
            if ((values["has_wire"] and values["wire_status"] != 0) or
                    (values["has_original"] and values["original_status"] != 0)):
                self.problems.add(f"ordinary requested error for grant {values['grant']}")

    @staticmethod
    def batch_key(values):
        return values["reservation"], values["generation"]

    @staticmethod
    def caller_key(values):
        return values["peer"], values["caller0"], values["caller1"]

    @staticmethod
    def parent_key(values):
        return (values["peer"], values["caller0"], values["caller1"],
                values["group"], values["ordinary_sequence"], values["origin_source"])

    def batch(self, values, *, create=False):
        key = self.batch_key(values)
        _require(all(key) and values["peer"] in self.peers, "unbound actor batch identity")
        identity = values["peer"], values["priority"]
        _require(self.peers[identity[0]][3] == identity[1], "actor batch priority changed")
        if key not in self.batches:
            _require(create, "actor transition names an unknown batch")
            _require(not any(old[0] == key[0] for old in self.batches),
                     "live reservation changed generation")
            self.batches[key] = dict(identity=identity, unknown=False, open=False)
        state = self.batches[key]
        _require(state["identity"] == identity, "actor batch changed authenticated owner")
        if values["unknown"]:
            state["unknown"] = True
            self.problems.add(f"unknown batch {key}")
        return state

    def entity(self, values):
        key = values["subject"], values["id"]
        _require(key in self.entities, "orphan or removed actor entity")
        state = self.entities[key]
        identity = state["identity"]
        _require(all(values[name] == old for name, old in identity),
                 "actor entity changed immutable identity")
        if key[0] == 3 and values["event"] != 8:
            _require(tuple(values[name] for name in _COMMUNICATOR_FIELDS) == state["communicator"],
                     "communicator identity changed or appeared before native acceptance")
        self.batch(values)
        return key, state

    def actor(self, record, values):
        _require(not self.actor_stopped, "actor record follows stopped proof")
        _require(values["actor_serial"] == self.actor_serial + 1,
                 "actor serial is not contiguous from one")
        self.actor_serial += 1
        event, subject = values["event"], values["subject"]
        if not self.actor_begun:
            _require(event == 1 and subject == 1, "actor capture lacks begin")
            self.actor_begun = True
            return
        _require(event != 1, "duplicate actor begin")
        if event == 20:
            raise ValueError("actor reported lost evidence")
        if event in (17, 18, 19):
            _require(subject == 1, "actor lifecycle event has wrong subject")
            if event == 17:
                _require(values["peer"] in self.peers and
                         values["peer"] not in self.disconnected, "orphan/repeated disconnect")
                self.disconnected.add(values["peer"])
            elif event == 18:
                _require(not self.closed, "repeated actor close")
                self.closed = True
            else:
                _require(self.closed and not self.entities and not self.batches and
                         not self.parents and not self.tickets and not self.actor_grants and
                         not self.async_pending and not any(self.depths.values()),
                         "actor stopped with retained or open ownership")
                _require(values["status"] == 0 and not values["unknown"],
                         "actor stopped without healthy proof")
                self.actor_stopped = True
            return
        if event == 2:
            self.admission(values)
            return
        if subject == 2:
            self.batch_event(values)
            return
        if subject == 5:
            self.parent_event(values)
            return
        if subject == 6:
            self.cleanup_event(values)
            return
        _require(subject in (3, 4), "invalid actor event/subject combination")
        key, state = self.entity(values)
        if event in (3, 4):
            self.actor_grant(record, values, key, state)
        elif event == 5:
            self.entity_state(values, state)
        elif event == 6:
            self.binding(values, key, state)
        elif event == 7:
            self.source_action(values, key)
        elif event == 8:
            _require(subject == 3 and key in self.actor_grants,
                     "native submission lacks its exact actor grant")
            _require(values["native_operation"] > 0 and
                     not state["submitted"] and
                     values["native_operation"] not in self.native_operations,
                     "repeated or invalid native operation")
            if values["native_kind"] in (9, 10, 11):
                sources = [phase for attachment, phase in self.attachments.items()
                           if attachment[0] == key and attachment[1] == 3]
                _require(sources and all(phase == 2 for phase in sources),
                         "collective submission lacks every accepted source turn begin")
            caller = self.caller_key(values)
            _require(values["native_sequence"] == self.native_sequences.get(caller, 0) + 1,
                     "native lane sequence is not contiguous")
            self.native_sequences[caller] = values["native_sequence"]
            self.native_operations[values["native_operation"]] = key
            state["submitted"] = values["native_operation"]
            self.bind_communicator(record, values, key, state)
        elif event == 9:
            self.observation(values, key, state)
        elif event == 10:
            self.group(values, state)
        elif event == 11:
            _require(values["reply_ready"] and not state["published"],
                     "duplicate or unavailable actor reply")
            if subject == 3 and values["action"] != 6:
                _require(state["observation"] is not None and values["has_original"],
                         "native reply lacks actual requested status")
            if values["action"] == 6:
                _require(subject == 3 and not state["submitted"] and values["status"] != 0,
                         "requested native work was rewritten as a policy rejection")
                self.problems.add(f"known preissue rejection for {key}")
            state["published"] = True
        elif event == 12:
            _require(values["action"] == 10 and values["reply_released"] and
                     state["published"] and not state["released"],
                     "orphan or repeated reply release")
            state["released"] = True
        elif event == 13:
            self.retirement(values, key, state)
        else:
            raise ValueError("unknown actor entity transition")
        if values["unknown"] or (subject == 3 and values["stage"] == 10) or (
                subject == 4 and values["stage"] == 9):
            self.problems.add(f"uncertain actor entity {key}")
        if subject == 4 and values["reply_ready"] and (
                values["native_status"] != 0 or
                (values["has_original"] and values["original_status"] != 0) or
                (values["has_host"] and
                 (values["completion_status"] != 0 or values["host_origin"] == 3))):
            self.problems.add(f"ordinary reply contains requested/host error for {key}")

    def admission(self, values):
        subject = values["subject"]
        _require(subject in (3, 4), "unknown admission namespace")
        _require(not any(values[name] for name in _COMMUNICATOR_FIELDS),
                 "admission claims an already accepted native communicator")
        caller = self.caller_key(values)
        _require(caller[0] in self.peers and self.peers[caller[0]][3] == values["priority"],
                 "admission lacks authenticated peer")
        sequence = values["sequence"] if subject == 3 else values["ordinary_sequence"]
        namespace = (subject, caller, values["cleanup_parent"],
                     values["group"] if values["cleanup_parent"] else 0)
        if values["sequence_consumed"]:
            _require(caller[1] and caller[2] and
                     sequence == self.sequences.get(namespace, 0) + 1,
                     "actor public sequence is not contiguous in its namespace")
            self.sequences[namespace] = sequence
        if not values["accepted"]:
            _require(values["id"] == 0 and values["status"] != 0 and values["action"] == 6,
                     "rejected admission claims an accepted identity or success")
            self.problems.add(f"rejected admission {caller}/{sequence}: {values['status']}")
            return
        _require(not self.closed and caller[0] not in self.disconnected,
                 "admission follows owner closure")
        _require(values["sequence_consumed"] and values["status"] == 0 and values["action"] == 1 and
                 values["id"] > 0 and values["cutoff"] > 0 and values["stage"] == 1,
                 "accepted actor admission has invalid initial state")
        key = subject, values["id"]
        _require(key not in self.entities and key not in self.removed,
                 "duplicate or recycled actor identity")
        self.batch(values, create=True)
        names = ["peer", "priority", "caller0", "caller1", "group", "reservation", "generation"]
        if subject == 3:
            _require(1 <= values["native_kind"] <= 11 and not values["ordinary_sequence"],
                     "invalid native request kind/namespace")
            names += ["sequence", "origin_source", "native_kind", "caller_primary_domain",
                      "output_contract"]
            if self.depths.get(caller, 0):
                _require(values["group"] == self.caller_groups[caller] and
                         self.batch_key(values) == self.caller_batches[caller],
                         "native admission changed the caller's active group or batch")
            elif values["native_kind"] == 7:
                _require(values["group"] > self.last_groups.get(caller, 0),
                         "outer GroupStart reused a stale or zero group")
            else:
                _require(values["group"] == 0, "native admission retained a closed group")
        else:
            # Callback enqueue, exact original BLAS and generic copies belong to
            # SCN1. Callback and ASY1 observations remain unwrapped metadata.
            _require((0 < values["ordinary_opcode"] < 131 or
                      values["ordinary_opcode"] in (134, 139, 140, 141)) and
                     values["wire_sequence"] > 0,
                     "ordinary continuation is an opaque or invalid command")
            self.connection(values["connection"], values["priority"], values["peer"])
            if not values["cleanup_parent"]:
                _require(self.depths.get(caller, 0) > 0 and
                         values["group"] == self.caller_groups.get(caller) and
                         self.batch_key(values) == self.caller_batches.get(caller),
                         "ordinary continuation lacks its exact open caller group")
            else:
                parent_key = (values["peer"], values["caller0"], values["caller1"],
                              values["group"], values["cleanup_parent"], values["origin_source"])
                _require(parent_key in self.parents and self.parents[parent_key]["bound"] and
                         not self.parents[parent_key]["consumed"],
                         "cleanup continuation lacks its exact durable import")
            names += ["ordinary_sequence", "cleanup_parent", "origin_source", "connection",
                      "wire_sequence", "ordinary_opcode"]
        self.entities[key] = dict(identity=tuple((name, values[name]) for name in names),
                                  last=values, submitted=0, observation=None, published=False,
                                  released=False, retired=False, cancelled=False, group_seen=False,
                                  async_seen=False, async_resolved=False, members=set(),
                                  bindings=set(), outcome=None, communicator=(0, 0, 0),
                                  submit_serial=0, async_required=set(), async_query_consumed=False,
                                  outer_end=False, group_member=False, member_count=None)

    def bind_communicator(self, record, values, key, state):
        """Only accepted native ownership, never a caller context or input claim."""
        communicator = tuple(values[name] for name in _COMMUNICATOR_FIELDS)
        if values["native_kind"] in (2, 3):
            _require(communicator[2] == values["allocation"],
                     "init communicator differs from its actual reservation")
        if any(communicator):
            domain, context, object_id = communicator
            owner = values["peer"], context
            old = self.communicators.setdefault((domain, object_id), owner)
            _require(old == owner, "communicator changed authenticated peer or context")
        state["communicator"] = communicator
        state["submit_serial"] = record.serial
        caller = self.caller_key(values)
        depth = self.depths.get(caller, 0)
        if depth and values["native_kind"] in _GROUP_MEMBER_KINDS:
            state["group_member"] = True
            self.group_members.setdefault(caller, set()).add(key)
        if depth == 1 and values["native_kind"] == 8:
            state["outer_end"] = True
            _require(state["members"] == self.group_members.get(caller, set()),
                     "outer End omitted or invented actual group members")
            _require(state["member_count"] in (None, len(state["members"])),
                     "outer End member relation is incomplete")

    def entity_state(self, values, state):
        previous = state["last"]
        _require(values["stage"] > 0 and values["cutoff"] >= previous["cutoff"],
                 "actor stage/cutoff moved backwards")
        if values["previous_stage"]:
            _require(values["previous_stage"] == previous["stage"] and
                     values["stage"] != previous["stage"], "inconsistent explicit stage transition")
        if previous["stage"] != values["stage"]:
            _require(values["cutoff"] > previous["cutoff"] or
                     values["stage"] in ((10,) if values["subject"] == 3 else (8, 9)),
                     "changed actor stage reused its old cutoff")
        terminal = (10, 11) if values["subject"] == 3 else (7, 8, 9)
        _require(previous["stage"] not in terminal or values["stage"] == previous["stage"],
                 "actor terminal stage resumed")
        if values["reply_ready"]:
            _require(state["published"], "state invented reply publication")
        if values["reply_released"]:
            _require(state["released"] or state["cancelled"], "state invented reply release")
        # A state-change notification can precede its explicit retirement event.
        # Only retirement(), never this summary flag, authorizes removal.
        state["last"] = values

    def actor_grant(self, record, values, key, state):
        inputs = tuple(values[name] for name in ("input_reservation", "input_generation", "input_cutoff"))
        if values["event"] == 3:
            _require(key not in self.actor_grants and all(inputs), "duplicate or empty actor grant")
            serial = self.open_runtime_grants.get(key)
            _require(serial is not None and
                     tuple(self.grants[serial]["values"][name] for name in
                           ("reservation", "generation", "cutoff")) == inputs,
                     "actor grant lacks unique fresh Core authority")
            source_cut = None
            if key[0] == 3 and values["action"] in (6, 8):
                root = self.grants[serial]["values"]
                expected_mode = 2 if values["action"] == 8 else 4
                _require(root["mode"] == expected_mode and not root["capture"] and
                         root["root_source"] == root["leaf_source"] == values["source"] and
                         state["last"]["stage"] == values["stage"] == 2,
                         "actor source action changed its runtime root or active stage")
                members = [phase for attachment, phase in self.attachments.items()
                           if attachment[0] == key and attachment[1] == 3 and
                           attachment[2] == values["source"] and
                           attachment[3] == values["capture"]]
                _require(values["capture"] > 0 and members and all(phase == 1 for phase in members),
                         "actor source action lacks its exact accepted unbegun Turn")
                source_cut = values["source"], values["capture"]
            self.actor_grants[key] = dict(inputs=inputs, grant=serial,
                                          action=values["action"], source_cut=source_cut)
            self.grants[serial]["actor_seen"] = True
            if key[0] == 4 and values["stage"] == 2:
                _require(values["attempt"] > state["last"]["attempt"], "repeated ordinary attempt")
                state["last"] = values
        else:
            _require(key in self.actor_grants, "orphan actor grant return")
            started = self.actor_grants.pop(key)
            expected_action = 3 if key[0] == 4 and started["action"] == 0 else started["action"]
            _require(started["inputs"] == inputs and expected_action == values["action"],
                     "actor grant return changed original authority")
            if started["source_cut"] is not None:
                _require(started["source_cut"] == (values["source"], values["capture"]),
                         "actor grant return changed its exact source Turn")
            self.grants[started["grant"]]["actor_finished"] = True
            self.grants[started["grant"]]["actor_result"] = values
            # The saved input snapshot can legitimately predate a stage change.
            if values["status"] not in (0, 600):
                self.problems.add(f"actor grant failed at serial {record.serial}")

    def binding(self, values, key, state):
        binding = values["binding"]
        _require(binding in (1, 2, 3, 4, 5, 6), "empty actor binding")
        if binding in (5, 6):
            related = (3, values["related"])
            _require(related in self.entities and
                     self.batch_key(self.entities[related]["last"]) == self.batch_key(values),
                     "group/publication relation names foreign or erased member")
            if binding == 6:
                _require(values["native_kind"] == 8 and
                         self.entities[related]["last"]["native_kind"] in (2, 3),
                         "publication relation is not an outer End/init pair")
            elif related != key:
                _require(values["native_kind"] == 8 and not state["submitted"] and
                         self.caller_key(self.entities[related]["last"]) == self.caller_key(values),
                         "group member lacks its original caller or precedes admission")
                if state["member_count"] is None:
                    state["member_count"] = values["count"]
                _require(state["member_count"] == values["count"] and
                         values["index"] == len(state["members"]) and
                         related not in state["members"], "group members are incomplete or repeated")
                state["members"].add(related)
            else:
                _require(state["group_member"], "member annotation lacks actual grouped submission")
            relation = (key, binding, related, values["index"])
            _require(relation not in self.bindings and values["index"] < values["count"],
                     "duplicate or out-of-range group relation")
            self.bindings.add(relation)
            return
        self.source_connection(values["source"], peer=values["peer"])
        _require(values["capture"] > 0, "binding lacks exact frontier capture")
        relation = binding, values["source"], values["capture"], values["command_sequence"]
        if values["action"] == 12:
            _require(binding in (1, 2) and relation in state["bindings"],
                     "orphan removed precall/target binding")
            state["bindings"].remove(relation)
            return
        _require(relation not in state["bindings"], "duplicate source/frontier binding")
        if binding == 1:
            _require(key[0] == 4 and not state["last"]["attempt"],
                     "precall dependency was attached after execution")
        state["bindings"].add(relation)

    def source_action(self, values, key):
        self.source_connection(values["source"], peer=values["peer"])
        _require(values["capture"] > 0 and 0 <= values["index"] < values["count"],
                 "source action lacks exact cohort/capture identity")
        _require(values["binding"] in (3, 4), "source transition has wrong binding kind")
        attachment = key, values["binding"], values["source"], values["capture"], values["stream"]
        action, status = values["action"], values["status"]
        previous = self.attachments.get(attachment)
        _require(key in self.actor_grants, "source native action lacks exact actor grant")
        group_key = key, values["binding"], values["source"], values["capture"], action, status
        group = self.source_groups.get(group_key)
        if group is None or (status == 600 and group["next"] == group["count"]):
            group = dict(count=values["count"], next=0, streams=set())
            self.source_groups[group_key] = group
        _require(values["count"] == group["count"] and values["index"] == group["next"] and
                 values["stream"] not in group["streams"], "source cohort is reordered or duplicated")
        group["next"] += 1
        group["streams"].add(values["stream"])
        if status != 0:
            _require(action in (2, 3, 4, 5), "failed source admission is not a published attachment")
            if status != 600:
                self.problems.add(f"failed source boundary {attachment}")
            return
        if action == 1:
            _require(previous is None and values["binding"] == 3, "repeated source acceptance")
        elif action == 2:
            _require(previous == 1 or (previous is None and values["binding"] == 4),
                     "source began before acceptance or began twice")
        elif action in (3, 5):
            _require(previous == 2 and ((action == 3) == (values["binding"] == 3)),
                     "source finish/record lacks its matching begin")
        elif action == 4:
            _require(previous in (1, 2) and not self.entities[key]["submitted"],
                     "submitted source boundary was cancelled without known unentered proof")
        else:
            raise ValueError("unknown source boundary action")
        self.attachments[attachment] = action

    def observation(self, values, key, state):
        if key[0] == 4:
            _require(1 <= values["outcome"] <= 5, "ordinary observation lacks outcome kind")
            state["outcome"] = values
            if values["outcome"] == 1:
                _require(values["status"] == 600 and not values["command_sequence"],
                         "Deferred ordinary result claims accepted work")
            if values["outcome"] == 5:
                self.problems.add(f"uncertain continuation {key}")
            return
        _require(state["submitted"] == values["native_operation"] and state["submitted"] > 0,
                 "native observation lacks its actual submission")
        _require(values["phase"] > 0, "native observation lacks coordinator phase")
        previous = state["observation"]
        if previous is not None:
            if previous["has_original"]:
                _require(values["has_original"] and
                         values["original_status"] == previous["original_status"],
                         "requested native status was erased or changed")
            _require(not previous["physical_complete"] or values["physical_complete"],
                     "native physical completion regressed")
            _require(not previous["enqueue_complete"] or values["enqueue_complete"],
                     "native enqueue proof regressed")
            if previous["has_context"]:
                _require(values["has_context"] and all(
                    values[f"{side}_{field}"] == previous[f"{side}_{field}"]
                    for side in ("before", "after")
                    for field in ("kind", "identity", "context", "device")),
                    "actual native context pair was changed or erased")
            if previous["has_async"]:
                _require(values["has_async"] and values["async_status"] == previous["async_status"],
                         "same native query changed or erased its observed async value")
            if previous["has_output_written"]:
                _require(values["has_output_written"] and
                         values["output_written"] == previous["output_written"],
                         "same native operation changed or erased its observed output-write proof")
        if values["has_original"] and values["carrier"] == 0 and not values["has_context"]:
            self.problems.add(f"native context observation missing for {key}")
        if values["has_original"]:
            native = values["original_status"]
            if native == 7 and not state["async_seen"]:
                state["async_seen"] = True
                state["async_started"] = values["serial"]
                if state["outer_end"]:
                    state["async_required"] = {
                        self.communicator_key(self.entities[member])
                        for member in state["members"]
                    }
                else:
                    state["async_required"] = {self.communicator_key(state)}
                # Empty/NULL native paths are diagnostic observations, not
                # evidence that any query can resolve an unknown participant.
                if not state["async_required"]:
                    state["async_required"].add(None)
                self.async_pending.add(key)
            elif native not in (0, 7):
                self.problems.add(f"native status {native} for {key}")
        if values["has_async"]:
            _require(values["native_kind"] == 4 and values["has_original"],
                     "async cell lacks requested GetAsyncError provenance")
            if values["async_status"] not in (0, 7):
                self.problems.add(f"async status {values['async_status']} for {key}")
            if values["original_status"] == 0 and values["async_status"] == 0:
                qualified = (values["carrier"] == 0 and values["has_context"] and
                             values["output_contract"] == 2 and values["has_output_written"] and
                             values["output_written"] and self.communicator_key(state) is not None)
                if not qualified:
                    self.problems.add(f"async readiness lacks pinned write/context/owner proof for {key}")
                elif not state["async_query_consumed"]:
                    state["async_query_consumed"] = True
                    communicator = self.communicator_key(state)
                    for pending in tuple(self.async_pending):
                        candidate = self.entities[pending]
                        if (self.batch_key(candidate["last"]) == self.batch_key(values) and
                                state["submit_serial"] > candidate["async_started"]):
                            candidate["async_required"].discard(communicator)
                            if not candidate["async_required"]:
                                candidate["async_resolved"] = True
                                self.async_pending.remove(pending)
        if values["physical_complete"]:
            _require(values["has_original"] and values["carrier"] == 0,
                     "physical completion lacks healthy observed requested status")
            # refresh() may publish an older operation's changed observation
            # before the newer Query record in the same owner pass. Retain the
            # unresolved async obligation until that actual query is observed;
            # neither this flag nor a reply can authorize erasure meanwhile.
        if values["carrier"] not in (0, 600) or values["phase"] == 6:
            self.problems.add(f"native carrier/proof failure for {key}")
        state["observation"] = values

    @staticmethod
    def communicator_key(state):
        if not all(state["communicator"]):
            return None
        return state["last"]["peer"], *state["communicator"]

    def group(self, values, state):
        _require(values["subject"] == 3 and not state["group_seen"],
                 "repeated or nonnative group transition")
        observed = state["observation"]
        _require(observed is not None and observed["has_original"],
                 "group transition lacks actual returned native status")
        caller = self.caller_key(values)
        old = self.depths.get(caller, 0)
        _require(values["index"] == old, "group transition changed previous TLS depth")
        if old:
            _require(values["group"] == self.caller_groups[caller],
                     "group transition changed the caller's actual group")
        if values["native_kind"] == 7:
            _require(observed["original_status"] == 0 and values["depth"] == old + 1,
                     "GroupStart depth lacks actual success")
            if old == 0:
                self.caller_batches[caller] = self.batch_key(values)
                self.caller_groups[caller] = values["group"]
                self.last_groups[caller] = values["group"]
            _require(self.caller_batches.get(caller) == self.batch_key(values),
                     "nested group changed batch")
        else:
            _require(values["native_kind"] == 8 and old > 0 and
                     observed["original_status"] in (0, 7) and values["depth"] == old - 1,
                     "GroupEnd depth lacks actual return boundary")
            _require(self.caller_batches.get(caller) == self.batch_key(values),
                     "GroupEnd changed original batch")
            if values["depth"] == 0:
                _require(state["outer_end"], "outer group transition lacks accepted End membership")
                if (observed["original_status"] == 0 and observed["carrier"] == 0 and
                        observed["has_context"]):
                    # Actual outer success reconciles inner InProgress; this
                    # grants neither outgoing enqueue nor physical-tail proof.
                    for member in state["members"]:
                        candidate = self.entities[member]
                        if member in self.async_pending:
                            candidate["async_required"].clear()
                            candidate["async_resolved"] = True
                            self.async_pending.remove(member)
                self.group_members.pop(caller, None)
                del self.caller_groups[caller]
                del self.caller_batches[caller]
        self.depths[caller] = values["depth"]
        state["group_seen"] = True

    def retirement(self, values, key, state):
        action = values["action"]
        if action == 4:
            _require(key[0] == 4 and not state["submitted"] and not state["retired"] and
                     values["stage"] == 8 and values["retired"] and values["reply_released"],
                     "invalid cancellation proof")
            state["cancelled"] = state["released"] = state["retired"] = True
        elif action == 11:
            _require(not state["retired"], "repeated physical retirement")
            if key[0] == 3:
                _require(state["observation"] is not None and
                         state["observation"]["physical_complete"] and state["released"],
                         "native retirement replaced physical proof with reply release")
            else:
                _require(values["retired"], "ordinary retirement lacks exact frontier proof")
            state["retired"] = True
        elif action == 12:
            rejected = key[0] == 3 and values["stage"] == 11 and not state["submitted"]
            _require(state["released"] and (state["retired"] or rejected),
                     "entity removed before physical stop or known preissue rejection")
            _require(key not in self.actor_grants and key not in self.async_pending,
                     "entity removed with active authority or async ownership")
            _require(all(phase in (3, 4, 5) for attachment, phase in self.attachments.items()
                         if attachment[0] == key), "entity removed with an unfinished source boundary")
            self.removed.add(key)
            del self.entities[key]
        else:
            raise ValueError("unknown actor retirement action")

    def batch_event(self, values):
        _require(values["event"] == 16, "batch subject has wrong event")
        key = self.batch_key(values)
        batch = self.batch(values)
        if values["action"] == 12:
            _require(values["retired"] and not values["open_group"] and not batch["unknown"],
                     "batch removal lacks healthy closed state")
            _require(not any(self.batch_key(state["last"]) == key for state in self.entities.values())
                     and not any(self.batch_key(parent) == key for parent in self.parents.values()),
                     "batch removed with retained members or cleanup parents")
            del self.batches[key]
        else:
            _require(values["action"] == 0, "unknown batch summary action")
            opened = key in self.caller_batches.values()
            _require(bool(values["open_group"]) == opened, "batch summary changed actual group openness")
            physical = opened
            for entity_key, state in self.entities.items():
                if self.batch_key(state["last"]) != key:
                    continue
                if state["submitted"] and (state["observation"] is None or
                                           not state["observation"]["physical_complete"]):
                    physical = True
                if any(attachment[0] == entity_key and phase not in (3, 4, 5)
                       for attachment, phase in self.attachments.items()):
                    physical = True
            _require(not physical or values["has_physical_work"],
                     "batch summary erased actual retained physical ownership")
            batch["open"] = bool(values["open_group"])
            if values["open_group"]:
                _require(values["has_physical_work"], "open group lost conservative reservation")

    def parent_event(self, values):
        _require(values["event"] == 14, "parent subject has wrong event")
        key = self.parent_key(values)
        self.batch(values)
        self.connection(values["connection"], peer=values["peer"])
        previous = self.parents.get(key)
        if previous is None:
            _require(values["action"] != 12 and
                     any(state["last"]["subject"] == 4 and
                         state["last"]["ordinary_opcode"] == 22 and
                         self.parent_key(state["last"]) == key for state in self.entities.values()),
                     "cleanup parent lacks its admitted ordinary import")
        else:
            _require(previous["connection"] == values["connection"] and
                     self.batch_key(previous) == self.batch_key(values), "cleanup parent changed owner")
            _require(not previous["bound"] or values["bound"], "cleanup binding regressed")
            _require(not previous["consumed"] or values["consumed"], "cleanup consumption regressed")
            if previous["bound"]:
                _require(previous["allocation"] == values["allocation"], "cleanup allocation changed")
        if values["unknown"]:
            self.problems.add(f"unknown cleanup parent {key}")
        if values["action"] == 13:
            _require(values["ticket"] in self.tickets and values["index"] < values["count"],
                     "sealed parent lacks its exact cleanup ticket")
            ticket = self.tickets[values["ticket"]]
            _require(key not in ticket["parents"] and values["connection"] == ticket["connection"] and
                     values["count"] == ticket["count"] and values["index"] == len(ticket["parents"]),
                     "duplicate or wrong-owner sealed parent")
            ticket["parents"].add(key)
        if values["action"] == 12:
            _require(previous is not None and values["consumed"] and not values["related"] and
                     not values["unknown"], "cleanup parent removed without consumed healthy proof")
            self.parent_history[key] = values
            del self.parents[key]
        else:
            self.parents[key] = values

    def cleanup_event(self, values):
        _require(values["event"] == 15 and values["id"] == values["ticket"] > 0,
                 "cleanup ticket identity mismatch")
        self.connection(values["connection"], peer=values["peer"])
        key = values["ticket"]
        if values["action"] == 13:
            _require(key not in self.tickets and values["count"] > 0, "repeated or empty cleanup seal")
            self.tickets[key] = dict(connection=values["connection"], peer=values["peer"],
                                     count=values["count"], parents=set())
        elif values["action"] == 14:
            _require(key in self.tickets, "cleanup proof lacks its seal")
            ticket = self.tickets[key]
            _require(ticket["connection"] == values["connection"] and
                     ticket["peer"] == values["peer"] and len(ticket["parents"]) == ticket["count"],
                     "cleanup proof changed or omitted sealed owners")
            if values["status"] == 0:
                _require(values["retired"] and not values["unknown"] and
                         all(parent in self.parent_history or
                             (parent in self.parents and self.parents[parent]["consumed"])
                             for parent in ticket["parents"]), "cleanup proof precedes parent retirement")
                del self.tickets[key]
            elif values["status"] != 600:
                self.problems.add(f"cleanup proof failed for {key}")
        else:
            raise ValueError("unknown cleanup ticket action")

    def pending(self):
        result = []
        if self.actor_begun and not self.actor_stopped:
            result.append("actor not stopped")
        result.extend(f"ordering {key} unretired" for key, state in self.ordering.items()
                      if not state["retired"])
        result.extend(f"grant {key} incomplete" for key, grant in self.grants.items()
                      if grant["result"] is None and grant["handler"] is None)
        result.extend(f"entity {key} retained" for key in self.entities)
        result.extend(f"batch {key} retained" for key in self.batches)
        result.extend(f"parent {key} retained" for key in self.parents)
        result.extend(f"cleanup ticket {key} retained" for key in self.tickets)
        result.extend(f"actor grant {key} unreturned" for key in self.actor_grants)
        result.extend(f"async operation {key} unresolved" for key in self.async_pending)
        result.extend(f"source cohort {key} truncated" for key, group in self.source_groups.items()
                      if group["next"] != group["count"])
        return result


def parse_semantic_audit(content, *, require_complete=False):
    """Validate all semantic records before any caller-selected measurement window.

    Diagnostic mode retains incomplete/failed evidence with explicit unresolved
    reasons. Complete mode additionally requires actual healthy lifecycle closure.
    Neither mode confers useful-compute, physical-priority or full-device credit.
    """
    _require(type(require_complete) is bool, "require_complete must be a boolean")
    records = _decode_records(content)
    state = _Validation()
    for record in records:
        try:
            state.apply(record)
        except (ValueError, KeyError) as error:
            raise ValueError(f"semantic line {record.line}, serial {record.serial}: {error}") from error
    unresolved = state.pending() + sorted(state.problems)
    if not state.begun:
        unresolved.append("no semantic evidence")
    elif not state.ended:
        unresolved.append("service end not observed")
    complete = bool(records) and state.ended and not unresolved
    _require(not require_complete or complete,
             "semantic audit is incomplete or failed: " + "; ".join(unresolved))
    return SemanticAudit(records, records[0].service if records else None,
                         complete, tuple(unresolved))
