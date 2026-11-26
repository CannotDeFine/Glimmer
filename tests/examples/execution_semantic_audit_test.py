"""Independent hostile semantic-log fixtures; no native or collector imports."""

from dataclasses import FrozenInstanceError
import json
import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import execution_semantic_audit as audit
from execution_semantic_service_test import check_host_function_disconnect


def record(serial=1, **fields):
    values = dict(v=1, service=37, serial=serial, family=4)
    values.update(fields)
    return "Execution semantic audit: " + json.dumps(values, separators=(",", ":")) + "\n"


class SemanticFraming(unittest.TestCase):
    def test_exact_offsets_prefix_crlf_and_immutable_records(self):
        before = "unrelated service line\n"
        prefix = "[2026-10-22 12:00:00.000] [info] "
        first = record().removesuffix("\n")
        text = before + prefix + first + "\r\n" + record(2)
        decoded = audit._decode_records(text)
        self.assertEqual(len(decoded), 2)
        self.assertEqual(decoded[0].position, len(before + prefix))
        self.assertEqual(decoded[0].end_position, len(before + prefix + first))
        self.assertEqual(decoded[0].line, 2)
        self.assertEqual(decoded[1].position, len(before + prefix + first + "\r\n"))
        self.assertEqual(decoded[0].value("service"), 37)
        with self.assertRaises(FrozenInstanceError):
            decoded[0].serial = 9
        with self.assertRaises(KeyError):
            decoded[0].value("missing")

    def test_marker_cannot_hide_in_prefix_substring_or_trailing_data(self):
        valid = record().strip()
        for text in (
                "prefix " + valid, valid + " trailing", valid + valid,
                "[" + valid + "] " + valid, valid.replace("audit:", "audit :"),
                valid.replace("audit:", "audit_extra:"),
                "Execution semantic audit:", "Execution semantic audit: []",
                "Execution semantic audit: {\n}\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit._decode_records(text)

    def test_duplicate_keys_include_escaped_spelling(self):
        for extra in ('"v":1', '"\\u0076":1', '"serial":1', '"family":4'):
            text = record().replace("}", "," + extra + "}")
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                audit._decode_records(text)

    def test_scalar_types_finite_and_bounded_integer_values(self):
        for value in (True, False, None, 1.0, [], {}, "1", -(1 << 63) - 1, 1 << 64):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit._decode_records(record(extra=value))
        for value in ("NaN", "Infinity", "-Infinity", "1e0", "1.0", "9" * 100):
            text = record().replace('"v":1', '"v":' + value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit._decode_records(text)

    def test_common_schema_and_serial_cannot_be_forged(self):
        for fields in (dict(v=0), dict(v=2), dict(family=0), dict(family=5),
                       dict(service=0), dict(service=-1), dict(serial=0), dict(serial=2)):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                audit._decode_records(record(**fields))
        for text in (record() + record(), record() + record(3),
                     record() + record(2, service=38)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit._decode_records(text)
        for key in ("v", "service", "serial", "family"):
            values = dict(v=1, service=37, serial=1, family=4)
            del values[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit._decode_records("Execution semantic audit: " + json.dumps(values))

    def test_bounds_include_unrelated_input_and_utf8_bytes(self):
        text = record()
        with patch.object(audit, "MAX_CONTENT_BYTES", len(text.encode("utf-8"))):
            self.assertEqual(len(audit._decode_records(text)), 1)
            with self.assertRaises(ValueError):
                audit._decode_records(text + "x")
            with self.assertRaises(ValueError):
                audit._decode_records("é" * len(text))
        payload = text.removeprefix("Execution semantic audit: ").strip()
        with patch.object(audit, "MAX_RECORD_BYTES", len(payload) - 1):
            with self.assertRaises(ValueError):
                audit._decode_records(text)
        with patch.object(audit, "MAX_RECORDS", 1):
            with self.assertRaises(ValueError):
                audit._decode_records(text + record(2))
        with self.assertRaises(ValueError):
            audit._decode_records("\ud800")

    def test_no_evidence_is_not_a_synthetic_record(self):
        self.assertEqual(audit._decode_records("unrelated service output\n"), ())
        for content in (None, b"", 1):
            with self.subTest(content=content), self.assertRaises(ValueError):
                audit._decode_records(content)


# Deliberately independent of the parser's schema constants. These are invented
# CPU diagnostic identities, not hardware receipts or qualification claims.
ACTOR_NAMES = """
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
""".split()


class Trace:
    def __init__(self, *, actor=True):
        self.rows = []
        self.actor_serial = 0
        self.public_sequence = 0
        self.native_sequence = 0
        self.cutoff = 0
        self.depth = 0
        self.pending_requests = set()
        self.group_members = []
        self.runtime(1, pid=700, incarnation0=3, incarnation1=4)
        self.peer(1)
        if actor:
            self.actor(1)

    def add(self, family, **fields):
        row = dict(v=1, service=37, serial=len(self.rows) + 1, family=family)
        row.update(fields)
        self.rows.append(row)
        return row["serial"]

    def runtime(self, event, **fields):
        return self.add(2, event=event, **fields)

    def peer(self, connection, **fields):
        row = dict(event=1, connection=connection, peer=1, pid=800, start=900,
                   uid=1000, priority=7, role=0, policy=11, kernel_mode=2,
                   effective=1, retained=1)
        row.update(fields)
        return self.add(4, **row)

    def actor(self, event, **fields):
        row = dict.fromkeys(ACTOR_NAMES, 0)
        self.actor_serial += 1
        row.update(subject=1, event=event, actor_serial=self.actor_serial)
        row.update(fields)
        return self.add(1, **row)

    def ordering(self, event, **fields):
        row = dict(connection=1, priority=7, sequence=1, opcode=48, capture=101,
                   event_source=1, event_object=51, record_sequence=1,
                   event_capture=101, status=0)
        row.update(fields)
        return self.add(3, event=event, **row)

    def admit(self, kind=1, *, group=0, reservation=9, output_contract=2):
        self.public_sequence += 1
        self.cutoff += 1
        key = dict(subject=3, id=self.public_sequence, peer=1, priority=7,
                   caller0=7, caller1=8, group=group, sequence=self.public_sequence,
                   reservation=reservation, generation=1, cutoff=self.cutoff,
                   native_kind=kind, stage=1, output_contract=output_contract)
        self.actor(2, **key, action=1, accepted=1, sequence_consumed=1)
        return key

    def grant(self, key, *, mode=1, source=0, capture=0, edges=0, leaf=None):
        return self.runtime(10, policy=11, peer=key["peer"], priority=key["priority"],
                            reservation=key["reservation"], generation=key["generation"],
                            cutoff=key["cutoff"], target_kind=1 if key["subject"] == 3 else 2,
                            target=key["id"], action=5, mode=mode, root_source=source,
                            leaf_source=source if leaf is None else leaf, capture=capture, edges=edges,
                            path_kind=0)

    def attach(self, key, source=1, stream=0):
        grant = self.grant(key)
        inputs = dict(input_reservation=key["reservation"], input_generation=key["generation"],
                      input_cutoff=key["cutoff"])
        self.actor(3, **key, **inputs, action=2)
        self.actor(7, **key, action=1, binding=3, source=source, capture=301,
                   stream=stream, index=0, count=1)
        self.cutoff += 1
        previous = key["stage"]
        key.update(stage=2, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=previous)
        self.actor(4, **key, **inputs, action=2, status=600)
        self.result(grant, status=600)

    def source_progress(self, key, grant, *, source=1, capture=301, mode=2, **result):
        inputs = dict(input_reservation=key["reservation"], input_generation=key["generation"],
                      input_cutoff=key["cutoff"])
        action = 8 if mode == 2 else 6
        self.actor(3, **key, **inputs, action=action, source=source, capture=capture)
        issue = dict(status=0, issue_state=1 if mode == 2 else 0,
                     command_sequence=1 if mode == 2 else 0, native_status=0,
                     completion_status=0, has_library=0, library_status=0)
        issue.update(result)
        fields = dict(status=issue["status"])
        if action == 8:
            fields.update(outcome=issue["issue_state"] + 1,
                          **{name: issue[name] for name in ("command_sequence", "native_status",
                             "completion_status", "has_library", "library_status")})
        self.actor(4, **key, **inputs, action=action, source=source, capture=capture, **fields)
        self.result(grant, source=source, **issue)

    def result(self, grant, **fields):
        row = dict(status=0, issue_state=0, command_sequence=0, native_status=0,
                   completion_status=0, has_library=0, library_status=0, source=0)
        row.update(fields)
        return self.runtime(12, grant=grant, **row)

    def request(self, kind=1, *, group=0, reservation=9, native=0,
                physical=True, async_value=None, communicator=None, output_contract=2,
                write_proof=True, context_proof=True, carrier=0, members=None):
        key = self.admit(kind, group=group, reservation=reservation,
                         output_contract=output_contract)
        outer_end = kind == 8 and self.depth == 1
        grouped = self.depth > 0 and kind in (2, 3, 5, 6, 9, 10, 11)
        if outer_end:
            joined = self.group_members if members is None else members
            for index, member in enumerate(joined):
                self.actor(6, **key, binding=5, action=7, related=member["id"],
                           index=index, count=len(joined))
        if kind in (9, 10, 11):
            self.attach(key)
        grant = self.grant(key)
        inputs = dict(input_reservation=reservation, input_generation=1,
                      input_cutoff=key["cutoff"])
        self.actor(3, **key, **inputs, action=2)
        if kind in (9, 10, 11):
            self.actor(7, **key, action=2, binding=3, source=1, capture=301,
                       stream=0, index=0, count=1)
        self.native_sequence += 1
        key.update(native_sequence=self.native_sequence, native_operation=self.native_sequence)
        if communicator is None:
            communicator = ((51, 61, 100 + key["id"]) if kind in (2, 3) else
                            (51, 61, 71) if kind in (4, 5, 6, 9, 10, 11) else (0, 0, 0))
        key.update(communicator_domain=communicator[0], communicator_context=communicator[1],
                   communicator_object=communicator[2])
        self.actor(8, **key, allocation=communicator[2] if kind in (2, 3) else 0)
        self.cutoff += 1
        previous_stage = key["stage"]
        key.update(stage=3, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=previous_stage)
        self.actor(4, **key, **inputs, action=2)
        self.result(grant)
        key.update(has_original=1, original_status=native, phase=5 if physical else 3,
                   physical_complete=int(physical), enqueue_complete=int(physical),
                   has_context=int(context_proof), before_kind=int(context_proof),
                   after_kind=int(context_proof), carrier=carrier)
        if async_value is not None:
            key.update(has_async=1, async_status=async_value,
                       has_output_written=int(write_proof is not None), output_written=int(bool(write_proof)))
        observed = dict(key, stage=0, cutoff=0)
        self.actor(9, **observed)
        if grouped:
            self.group_members.append(key)
            self.actor(6, **key, binding=5, related=key["id"],
                       index=len(self.group_members) - 1, count=len(self.group_members))
        if kind in (9, 10, 11):
            finish_grant = self.grant(key)
            finish_inputs = dict(input_reservation=reservation, input_generation=1,
                                 input_cutoff=key["cutoff"])
            self.actor(3, **key, **finish_inputs, action=2)
            self.actor(7, **key, action=3, binding=3, source=1, capture=301,
                       stream=0, index=0, count=1)
            self.actor(4, **key, **finish_inputs, action=2)
            self.result(finish_grant)
        if kind in (7, 8):
            previous = self.depth
            self.depth += 1 if kind == 7 else -1
            self.actor(10, **key, index=previous, depth=self.depth)
            if outer_end:
                self.group_members = []
        self.cutoff += 1
        key.update(stage=9 if physical else 5, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=3)
        key["reply_ready"] = 1
        self.actor(11, **key, action=9)
        key["reply_released"] = 1
        self.actor(12, **key, action=10)
        if physical:
            self.actor(13, **key, action=11)
            self.actor(13, **key, action=12)
        else:
            self.pending_requests.add(key["id"])
        self.batch(reservation)
        return key

    def batch(self, reservation=9, *, remove=False, **fields):
        row = dict(subject=2, id=reservation, peer=1, priority=7,
                   reservation=reservation, generation=1, open_group=int(self.depth > 0),
                   has_physical_work=int(self.depth > 0 or bool(self.pending_requests)),
                   count=len(self.pending_requests))
        if remove:
            row.update(action=12, retired=1)
        row.update(fields)
        self.actor(16, **row)

    def finish_pending(self, key):
        key.update(physical_complete=1, enqueue_complete=1, phase=5)
        self.actor(9, **dict(key, stage=0, cutoff=0, reply_released=0))
        self.cutoff += 1
        previous = key["stage"]
        key.update(stage=9, cutoff=self.cutoff)
        self.actor(5, **key, previous_stage=previous)
        self.actor(13, **key, action=11)
        self.actor(13, **key, action=12)
        self.pending_requests.remove(key["id"])

    def finish(self, *, actor=True):
        if actor:
            self.actor(18)
            self.actor(19, retired=1)
        self.runtime(2)

    def continuation(self, *, deferred=False, held=False, opcode=8, has_original=True):
        self.cutoff += 1
        key = dict(subject=4, id=1, peer=1, priority=7, caller0=7, caller1=8,
                   group=10, ordinary_sequence=1, reservation=9, generation=1,
                   cutoff=self.cutoff, stage=1, connection=1, wire_sequence=41,
                   ordinary_opcode=opcode, origin_source=1)
        self.actor(2, **key, action=1, accepted=1, sequence_consumed=1)
        self.actor(6, **key, binding=1, action=7, source=1, capture=101)
        for attempt in range(1, 3 if deferred else 2):
            grant = self.grant(key, mode=3)
            inputs = dict(input_reservation=9, input_generation=1, input_cutoff=key["cutoff"])
            self.cutoff += 1
            key.update(attempt=attempt, stage=2, cutoff=self.cutoff)
            self.actor(3, **key, **inputs)
            retry = deferred and attempt == 1
            self.runtime(14, grant=grant, continuation=1, attempt=attempt,
                         status=600 if retry else 0, outcome=1 if retry else 4,
                         has_wire=int(not retry), wire_status=0,
                         has_original=int(not retry and has_original), original_status=0,
                         receipt_sequence=0, mailbox_sequence=0,
                         capture=0 if retry else 202, source=0 if retry else 1)
            self.actor(9, **dict(key, status=600 if retry else 0), outcome=1 if retry else 4)
            if not retry:
                key.update(reply_ready=1, disposition=2, has_original=int(has_original),
                           original_status=0)
                self.actor(11, **key, action=9)
                self.actor(6, **key, binding=2, action=7, source=1, capture=202)
            self.cutoff += 1
            key.update(stage=3 if retry else 6, cutoff=self.cutoff, status=600 if retry else 0)
            self.actor(5, **key, previous_stage=2)
            self.actor(4, **dict(key, status=0), **inputs, action=3)
        if not held:
            self.retire_continuation(key)
        return key

    def retire_continuation(self, key):
        self.cutoff += 1
        key.update(stage=7, cutoff=self.cutoff, retired=1)
        self.actor(5, **key, previous_stage=6)
        self.actor(13, **key, action=11)
        key["reply_released"] = 1
        self.actor(12, **key, action=10)
        self.actor(6, **key, binding=1, action=12, source=1, capture=101)
        self.actor(6, **key, binding=2, action=12, source=1, capture=202)
        self.actor(13, **key, action=12)

    def text(self):
        return "".join("Execution semantic audit: " + json.dumps(row, separators=(",", ":")) + "\n"
                       for row in self.rows)

    def renumber(self):
        actor_serial = 0
        for serial, row in enumerate(self.rows, 1):
            row["serial"] = serial
            if row["family"] == 1:
                actor_serial += 1
                row["actor_serial"] = actor_serial


class SemanticLifecycle(unittest.TestCase):
    def complete(self, trace):
        result = audit.parse_semantic_audit(trace.text(), require_complete=True)
        self.assertTrue(result.is_complete)
        self.assertEqual(result.unresolved, ())
        return result

    def test_empty_service_and_actor_lifetimes(self):
        for actor in (False, True):
            trace = Trace(actor=actor)
            trace.finish(actor=actor)
            result = self.complete(trace)
            self.assertEqual(len(result.family(4)), 1)
            self.assertEqual(result.service, 37)

    def test_native_request_complete_is_not_compute_credit(self):
        trace = Trace()
        trace.request()
        trace.batch(remove=True)
        trace.finish()
        result = self.complete(trace)
        observed = next(row for row in result.family(1) if row.value("event") == 9)
        self.assertEqual(observed.actor_statuses.original, 0)
        self.assertIsNone(observed.actor_statuses.asynchronous)
        self.assertFalse(hasattr(result, "useful_compute"))

    def test_nested_group_depth_and_complete_members(self):
        trace = Trace()
        trace.request(7, group=10)
        trace.request(7, group=10)
        trace.request(8, group=10)
        trace.request(8, group=10)
        trace.batch(remove=True)
        trace.finish()
        self.complete(trace)

    def test_live_group_identity_applies_to_nested_start_inner_calls_and_end(self):
        for kind in (1, 4, 7, 8, 9):
            trace = Trace()
            trace.request(7, group=10)
            trace.admit(kind, group=11)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "active group"):
                audit.parse_semantic_audit(trace.text())
        trace = Trace()
        trace.request(7, group=10)
        trace.admit(4, group=10, reservation=10)
        with self.assertRaisesRegex(ValueError, "active group or batch"):
            audit.parse_semantic_audit(trace.text())

    def test_closed_group_query_uses_zero_and_next_outer_start_is_fresh(self):
        trace = Trace()
        trace.request(7, group=10)
        trace.request(8, group=10)
        trace.request(4, async_value=0)
        trace.batch(remove=True)
        trace.request(7, group=11, reservation=10)
        trace.request(8, group=11, reservation=10)
        trace.batch(10, remove=True)
        trace.finish()
        self.complete(trace)
        for kind, group in ((4, 10), (7, 0), (7, 10), (7, 9)):
            trace = Trace()
            trace.request(7, group=10)
            trace.request(8, group=10)
            trace.admit(kind, group=group)
            with self.subTest(kind=kind, group=group), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())

    def test_normal_continuation_cannot_change_current_group(self):
        trace = Trace()
        trace.request(7, group=10)
        trace.continuation(held=True)
        first = next(row for row in trace.rows if row["family"] == 1 and
                     row["event"] == 2 and row["subject"] == 4)
        first["group"] = 11
        with self.assertRaisesRegex(ValueError, "exact open caller group"):
            audit.parse_semantic_audit(trace.text())

    def test_deferred_retry_has_one_terminal_per_grant_and_exact_later_tail(self):
        trace = Trace()
        trace.request(7, group=10)
        ordinary = trace.continuation(deferred=True, held=True)
        result = audit.parse_semantic_audit(trace.text())
        self.assertTrue(any("entity (4, 1)" in reason for reason in result.unresolved))
        trace.request(8, group=10)
        trace.retire_continuation(ordinary)
        trace.batch(remove=True)
        trace.finish()
        result = self.complete(trace)
        outcomes = [row for row in result.family(2) if row.value("event") == 14]
        self.assertEqual([row.value("outcome") for row in outcomes], [1, 4])
        self.assertEqual([row.value("attempt") for row in outcomes], [1, 2])

    def test_handler_cannot_return_twice_or_launder_deferred_acceptance(self):
        for change in (dict(receipt_sequence=8), dict(mailbox_sequence=9), dict(has_wire=1),
                       dict(status=0)):
            trace = Trace()
            trace.request(7, group=10)
            trace.continuation(deferred=True)
            first = next(row for row in trace.rows if row["family"] == 2 and row["event"] == 14)
            first.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        trace = Trace()
        trace.request(7, group=10)
        trace.continuation()
        outcome = next(row for row in trace.rows if row["family"] == 2 and row["event"] == 14)
        trace.result(outcome["grant"])
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_reserved_batch_cannot_hide_released_but_unretired_native_work(self):
        trace = Trace()
        trace.request(2, physical=False)
        trace.rows[-1]["has_physical_work"] = 0
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_symbolic_context_transitions_survive_later_carrier_failure(self):
        trace = Trace()
        key = trace.request(2, physical=False)
        # Retain an already observed actual pair through the later carrier fault.
        for row in trace.rows:
            if row["family"] == 1 and row["event"] == 9:
                row.update(before_kind=2, before_identity=1, before_context=44, before_device=1,
                           after_kind=3, after_identity=3, after_context=55, after_device=1)
        fields = dict(key, stage=0, cutoff=0, reply_released=0, has_context=1,
                      before_kind=2, before_identity=1, before_context=44, before_device=1,
                      after_kind=3, after_identity=3, after_context=55, after_device=1,
                      carrier=999, phase=6)
        trace.actor(9, **fields)
        result = audit.parse_semantic_audit(trace.text())
        self.assertEqual(result.family(1)[-1].value("before_context"), 44)
        self.assertEqual(result.family(1)[-1].actor_statuses.original, 0)
        self.assertFalse(result.is_complete)
        for change in (dict(has_context=0), dict(before_kind=4), dict(before_context=0),
                       dict(after_device=0), dict(before_kind=1), dict(before_device=-1)):
            malformed = dict(fields)
            malformed.update(change)
            trace.rows.pop()
            trace.actor_serial -= 1
            trace.actor(9, **malformed)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())

    def test_absent_native_context_proof_retains_status_but_never_positive_evidence(self):
        trace = Trace()
        trace.request()
        trace.batch(remove=True)
        trace.finish()
        observed = next(row for row in trace.rows if row["family"] == 1 and row["event"] == 9)
        observed.update(has_context=0, before_kind=0, after_kind=0)
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertFalse(parsed.is_complete)
        self.assertTrue(any("context observation missing" in reason for reason in parsed.unresolved))
        self.assertEqual(next(row for row in parsed.family(1) if row.value("event") == 9)
                         .actor_statuses.original, 0)

    def test_end_seven_query_seven_retains_until_actual_zero_and_physical_proof(self):
        trace = Trace()
        trace.request(7, group=10)
        member = trace.request(2, group=10, physical=False, communicator=(51, 61, 71))
        end = trace.request(8, group=10, native=7, physical=False)
        trace.request(4, async_value=7)
        incomplete = audit.parse_semantic_audit(trace.text())
        self.assertFalse(incomplete.is_complete)
        self.assertTrue(any("async operation" in item for item in incomplete.unresolved))
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text(), require_complete=True)
        trace.request(4, async_value=0)
        trace.finish_pending(member)
        trace.finish_pending(end)
        trace.batch(remove=True)
        trace.finish()
        self.complete(trace)

    def test_end_ack_release_disconnect_never_prove_stop(self):
        trace = Trace()
        trace.request(7, group=10)
        trace.request(8, group=10, native=7, physical=False)
        trace.actor(17, peer=1)
        trace.actor(18)
        partial = audit.parse_semantic_audit(trace.text())
        self.assertTrue(any("entity" in reason for reason in partial.unresolved))
        trace.actor(19, retired=1)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_native_success_later_carrier_failure_is_retained_not_rewritten(self):
        trace = Trace()
        key = trace.request(2, physical=False)
        trace.actor(9, **dict(key, stage=0, cutoff=0, reply_released=0, carrier=999, phase=6))
        result = audit.parse_semantic_audit(trace.text())
        observed = [row for row in result.family(1) if row.value("event") == 9][-1]
        self.assertEqual(observed.actor_statuses.original, 0)
        self.assertEqual(observed.actor_statuses.carrier, 999)
        self.assertTrue(any("carrier" in reason for reason in result.unresolved))
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text(), require_complete=True)

    def test_no_evidence_partial_and_strict_completion(self):
        result = audit.parse_semantic_audit("unrelated log\n")
        self.assertFalse(result.is_complete)
        self.assertEqual(result.records, ())
        for text in ("", Trace().text()):
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.parse_semantic_audit(text, require_complete=True)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit("", require_complete=1)

    def test_unknown_actor_fields_enums_flags_and_presence(self):
        for change in (dict(extra=0), dict(event=21), dict(subject=7), dict(action=15),
                       dict(unknown=2), dict(carrier=1 << 31), dict(original_status=7),
                       dict(async_status=7), dict(output_written=1), dict(stage=12)):
            trace = Trace()
            trace.rows[-1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        trace = Trace()
        del trace.rows[-1]["has_physical_work"]
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_orphan_duplicate_public_sequence_and_stale_grant(self):
        for change in (dict(id=0), dict(sequence=2), dict(accepted=0),
                       dict(stage=2), dict(generation=0), dict(peer=8)):
            trace = Trace()
            trace.admit()
            trace.rows[-1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        trace = Trace()
        key = trace.admit()
        trace.grant(dict(key, cutoff=key["cutoff"] + 1))
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_repeated_or_missing_lifecycle_events(self):
        for event in (8, 11, 12, 13):
            trace = Trace()
            trace.request()
            index = next(i for i, row in enumerate(trace.rows)
                         if row["family"] == 1 and row["event"] == event)
            trace.rows.insert(index + 1, dict(trace.rows[index]))
            trace.renumber()
            # Insertion also invalidates later runtime grant serial references;
            # each selected duplicate is rejected before reaching such a reference.
            with self.subTest(event=event), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        trace = Trace()
        key = trace.admit()
        trace.actor(13, **key, action=12, reply_released=1)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_lost_unknown_cleanup_and_after_end_cannot_be_laundered(self):
        trace = Trace()
        trace.actor(20)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())
        trace = Trace(actor=False)
        trace.runtime(3, reason=1)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())
        trace = Trace(actor=False)
        trace.finish(actor=False)
        trace.peer(2)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())


class CommunicatorReadinessTests(unittest.TestCase):
    @staticmethod
    def two_communicators():
        trace = Trace()
        trace.request(7, group=10)
        first = trace.request(2, group=10, physical=False, communicator=(51, 61, 71))
        second = trace.request(3, group=10, physical=False, communicator=(51, 61, 72))
        end = trace.request(8, group=10, native=7, physical=False)
        return trace, first, second, end

    def assert_async_pending(self, trace, key):
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertIn(f"async operation (3, {key['id']}) unresolved", parsed.unresolved)
        self.assertFalse(parsed.is_complete)
        return parsed

    def finish(self, trace, *keys):
        for key in keys:
            trace.finish_pending(key)
        trace.batch(remove=True)
        trace.finish()
        self.assertTrue(audit.parse_semantic_audit(trace.text(), require_complete=True).is_complete)

    def test_two_distinct_participants_and_queries_never_replace_held_tails(self):
        trace, first, second, end = self.two_communicators()
        trace.request(4, communicator=(51, 61, 71), async_value=0)
        trace.request(4, communicator=(51, 61, 72), async_value=7)
        self.assert_async_pending(trace, end)
        trace.request(4, communicator=(51, 61, 72), async_value=0)
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertFalse(any("async operation" in reason for reason in parsed.unresolved))
        self.assertIn(f"entity (3, {end['id']}) retained", parsed.unresolved)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text(), require_complete=True)
        self.finish(trace, first, second, end)

    def test_duplicate_communicator_members_need_one_query_but_every_actual_tail(self):
        trace = Trace()
        trace.request(7, group=10)
        first = trace.request(9, group=10, physical=False)
        trace.request(7, group=10)
        second = trace.request(10, group=10, physical=False)
        trace.request(8, group=10)
        end = trace.request(8, group=10, native=7, physical=False)
        trace.request(4, async_value=0)
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertFalse(any("async operation" in reason for reason in parsed.unresolved))
        self.finish(trace, first, second, end)

    def test_outer_success_reconciles_inner_in_progress_without_query_or_tail_inference(self):
        trace = Trace()
        trace.request(7, group=10)
        first = trace.request(2, group=10, native=7, physical=False)
        self.assert_async_pending(trace, first)
        end = trace.request(8, group=10, physical=False)
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertFalse(any("async operation" in reason for reason in parsed.unresolved))
        self.assertIn(f"entity (3, {first['id']}) retained", parsed.unresolved)
        self.finish(trace, first, end)

    def test_earlier_query_cannot_pay_a_later_end_obligation(self):
        trace = Trace()
        trace.request(7, group=10)
        member = trace.request(2, group=10, physical=False, communicator=(51, 61, 71))
        trace.request(4, group=10, async_value=0)
        end = trace.request(8, group=10, native=7, physical=False)
        self.assert_async_pending(trace, end)
        trace.request(4, async_value=0)
        self.finish(trace, member, end)

    def test_query_wrong_domain_or_object_cannot_pay_another_participant(self):
        for communicator in ((52, 61, 71), (51, 61, 73), (0, 0, 0)):
            trace, _, _, end = self.two_communicators()
            trace.request(4, communicator=communicator, async_value=0)
            with self.subTest(communicator=communicator):
                self.assert_async_pending(trace, end)

    def test_query_in_another_batch_cannot_pay_the_original_generation(self):
        trace, _, _, end = self.two_communicators()
        trace.request(4, reservation=10, communicator=(51, 61, 71), async_value=0)
        self.assert_async_pending(trace, end)

    def test_accepted_null_communicator_and_omitted_init_output_remain_diagnostic(self):
        for kind in (2, 3, 4, 5, 6):
            trace = Trace()
            trace.request(kind, communicator=(0, 0, 0), native=4)
            parsed = audit.parse_semantic_audit(trace.text())
            observed = next(row for row in parsed.family(1) if row.value("event") == 9)
            self.assertEqual(observed.value("communicator_object"), 0)
            self.assertEqual(observed.actor_statuses.original, 4)
            self.assertFalse(parsed.is_complete)

    def test_accepted_communicator_context_and_peer_cannot_rebind(self):
        trace, _, _, _ = self.two_communicators()
        trace.request(4, communicator=(51, 62, 71), async_value=0)
        with self.assertRaisesRegex(ValueError, "communicator changed authenticated"):
            audit.parse_semantic_audit(trace.text())
        trace, _, _, _ = self.two_communicators()
        trace.peer(2, peer=2, pid=801, start=901)
        first_row = len(trace.rows)
        trace.request(4, reservation=10, communicator=(51, 61, 71), async_value=0)
        for row in trace.rows[first_row:]:
            if row["family"] == 1:
                row["peer"] = 2
                if row["caller0"]:
                    row.update(caller0=17, caller1=18, sequence=1)
                if row["native_sequence"]:
                    row["native_sequence"] = 1
            elif row["event"] == 10:
                row["peer"] = 2
        with self.assertRaisesRegex(ValueError, "communicator changed authenticated"):
            audit.parse_semantic_audit(trace.text())

    def test_unwritten_deferred_or_missing_context_query_is_diagnostic_not_readiness(self):
        for fields in (dict(write_proof=None), dict(write_proof=False), dict(output_contract=1),
                       dict(context_proof=False), dict(carrier=999, physical=False)):
            trace, _, _, end = self.two_communicators()
            trace.request(4, communicator=(51, 61, 71), async_value=0, **fields)
            trace.request(4, communicator=(51, 61, 72), async_value=0)
            with self.subTest(fields=fields):
                parsed = self.assert_async_pending(trace, end)
                self.assertTrue(any("async readiness lacks" in reason for reason in parsed.unresolved))
                observed = [row for row in parsed.family(1) if row.value("event") == 9 and
                            row.value("native_kind") == 4][0]
                self.assertEqual(observed.actor_statuses.original, 0)
                self.assertEqual(observed.actor_statuses.asynchronous, 0)

    def test_later_carrier_failure_preserves_tuple_output_and_prevents_qualification(self):
        trace, _, _, end = self.two_communicators()
        query = trace.request(4, communicator=(51, 61, 71), async_value=0,
                              physical=False)
        trace.actor(9, **dict(query, stage=0, cutoff=0, reply_released=0, carrier=999, phase=6))
        parsed = self.assert_async_pending(trace, end)
        last = parsed.family(1)[-1]
        self.assertEqual(last.value("communicator_object"), 71)
        self.assertEqual(last.actor_statuses.original, 0)
        self.assertEqual(last.actor_statuses.asynchronous, 0)
        self.assertTrue(any("carrier/proof failure" in item for item in parsed.unresolved))

    def test_same_query_cannot_change_seven_to_zero_or_erase_observed_cell(self):
        for change in (dict(async_status=0), dict(has_async=0, async_status=0)):
            trace, _, _, _ = self.two_communicators()
            query = trace.request(4, communicator=(51, 61, 71), async_value=7,
                                  physical=False)
            trace.actor(9, **dict(query, stage=0, cutoff=0, reply_released=0, **change))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "same native query"):
                audit.parse_semantic_audit(trace.text())

    def test_same_operation_cannot_change_or_erase_true_or_false_write_evidence(self):
        for initial in (False, True):
            for change in (dict(output_written=int(not initial)),
                           dict(has_output_written=0, output_written=0)):
                trace = Trace()
                query = trace.request(4, async_value=7, physical=False, write_proof=initial)
                trace.actor(9, **dict(query, stage=0, cutoff=0, reply_released=0, **change))
                with self.subTest(initial=initial, change=change), self.assertRaisesRegex(
                        ValueError, "observed output-write proof"):
                    audit.parse_semantic_audit(trace.text())

    def test_tuple_schema_native_acceptance_and_output_contract_are_immutable(self):
        for change in (dict(communicator_domain=51), dict(output_contract=3),
                       dict(output_contract=0), dict(communicator_domain=51,
                                                    communicator_context=61, communicator_object=71)):
            trace = Trace()
            trace.admit(4)
            trace.rows[-1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        for change in (dict(communicator_object=72), dict(communicator_context=62),
                       dict(output_contract=1)):
            trace = Trace()
            query = trace.request(4, physical=False, async_value=0)
            trace.actor(9, **dict(query, stage=0, cutoff=0, reply_released=0, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())

    def test_missing_or_repeated_end_members_cannot_shrink_participant_debt(self):
        for duplicate in (False, True):
            trace = Trace()
            trace.request(7, group=10)
            first = trace.request(2, group=10, physical=False)
            trace.request(3, group=10, physical=False)
            trace.request(8, group=10, native=7, physical=False,
                          members=[first, first] if duplicate else [first])
            with self.subTest(duplicate=duplicate), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())


class OrderingAndGrantTests(unittest.TestCase):
    def continuation_cut(self, stage):
        trace = Trace()
        trace.request(7, group=10)
        trace.cutoff += 1
        key = dict(subject=4, id=1, peer=1, priority=7, caller0=7, caller1=8,
                   group=10, ordinary_sequence=1, reservation=9, generation=1,
                   cutoff=trace.cutoff, stage=1, connection=1, wire_sequence=41,
                   ordinary_opcode=8, origin_source=1)
        trace.actor(2, **key, action=1, accepted=1, sequence_consumed=1)
        trace.actor(6, **key, binding=1, action=7, source=1, capture=101)
        if stage != 1:
            handler = trace.grant(key, mode=3)
            inputs = dict(input_reservation=9, input_generation=1, input_cutoff=key["cutoff"])
            trace.cutoff += 1
            key.update(stage=2, attempt=1, cutoff=trace.cutoff)
            trace.actor(3, **key, **inputs)
            deferred = stage == 3
            outcome = 1 if deferred else 2 if stage == 4 else 4
            trace.runtime(14, grant=handler, continuation=1, attempt=1,
                          status=600 if deferred else 0, outcome=outcome,
                          has_wire=int(stage == 6), wire_status=0,
                          has_original=int(stage == 6), original_status=0,
                          receipt_sequence=17 if stage == 4 else 0, mailbox_sequence=0,
                          capture=0 if deferred else 202, source=0 if deferred else 1)
            trace.actor(9, **dict(key, status=600 if deferred else 0), outcome=outcome)
            if not deferred:
                trace.actor(6, **key, binding=2, action=7, source=1, capture=202)
            if stage == 6:
                key.update(reply_ready=1, disposition=2, has_original=1)
                trace.actor(11, **key, action=9)
            trace.cutoff += 1
            key.update(stage=stage, cutoff=trace.cutoff)
            trace.actor(5, **key, previous_stage=2)
            trace.actor(4, **key, **inputs, action=3)
        return trace, key, 101 if stage in (1, 3) else 202

    def foreign_grant(self):
        trace = Trace()
        trace.peer(2)
        trace.ordering(1)
        trace.ordering(1, connection=2, opcode=54, capture=201)
        trace.ordering(4, connection=2, opcode=54, capture=201)
        key = trace.admit()
        trace.attach(key, source=2)
        grant = trace.grant(key, mode=2, source=2, leaf=1, capture=101, edges=1)
        trace.runtime(11, grant=grant, index=0, source=1, target=2, event_object=51,
                      record_sequence=1, source_capture=101, target_capture=201, target_sequence=1)
        return trace, grant

    def test_exact_authenticated_foreign_path_and_actual_leaf_sequence(self):
        trace, grant = self.foreign_grant()
        trace.result(grant, source=1, issue_state=1, command_sequence=1)
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertEqual(parsed.family(2)[-1].value("command_sequence"), 1)
        self.assertFalse(any("failed runtime grant" in reason for reason in parsed.unresolved))

    def test_zero_edge_grant_cannot_issue_or_reject_an_unrelated_same_peer_source(self):
        for mode in (2, 4):
            trace = Trace()
            trace.peer(2)
            key = trace.admit()
            trace.attach(key, source=1)
            grant = trace.grant(key, mode=mode, source=1, leaf=1, capture=0, edges=0)
            trace.source_progress(key, grant, mode=mode)
            audit.parse_semantic_audit(trace.text())
            # Both connections have the same authenticated peer, but no captured
            # foreign edge authorizes replacing attached source A with source B.
            trace.rows[grant - 1]["leaf_source"] = 2
            trace.rows[-1]["source"] = 2
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "zero-edge source grant"):
                audit.parse_semantic_audit(trace.text())

    def test_request_cut_requires_actual_exact_actor_turn_begin_and_end(self):
        for mode in (2, 4):
            for mutation in ("runtime_capture", "begin_capture", "end_capture", "begin_source", "missing"):
                trace = Trace()
                trace.peer(2)
                key = trace.admit()
                trace.attach(key)
                grant = trace.grant(key, mode=mode, source=1)
                trace.source_progress(key, grant, mode=mode)
                audit.parse_semantic_audit(trace.text())
                if mutation == "runtime_capture":
                    trace.rows[grant - 1]["capture"] = 301
                elif mutation == "begin_capture":
                    trace.rows[-3]["capture"] = 302
                elif mutation == "end_capture":
                    trace.rows[-2]["capture"] = 302
                elif mutation == "begin_source":
                    trace.rows[-3]["source"] = 2
                else:
                    del trace.rows[-3:-1]
                    trace.renumber()
                with self.subTest(mode=mode, mutation=mutation), self.assertRaises(ValueError):
                    audit.parse_semantic_audit(trace.text())

    def test_request_preentry_errors_do_not_invent_actor_calls_or_issued_work(self):
        for status in (600, 1, 999):
            trace = Trace()
            key = trace.admit()
            trace.attach(key)
            grant = trace.grant(key, mode=2, source=1)
            trace.result(grant, status=status, source=1)
            parsed = audit.parse_semantic_audit(trace.text())
            self.assertFalse(parsed.is_complete)
            self.assertEqual(parsed.family(2)[-1].value("command_sequence"), 0)
        trace = Trace()
        key = trace.admit()
        trace.attach(key)
        begun = trace.grant(key)
        inputs = dict(input_reservation=9, input_generation=1, input_cutoff=key["cutoff"])
        trace.actor(3, **key, **inputs, action=2)
        trace.actor(7, **key, action=2, binding=3, source=1, capture=301,
                    index=0, count=1)
        trace.actor(4, **key, **inputs, action=2)
        trace.result(begun)
        grant = trace.grant(key, mode=2, source=1)
        trace.source_progress(key, grant)
        with self.assertRaisesRegex(ValueError, "accepted unbegun Turn"):
            audit.parse_semantic_audit(trace.text())

    def test_continuation_direct_grant_uses_current_stage_specific_capture(self):
        for stage in (1, 3, 4, 6):
            for mode in (2, 4):
                trace, key, capture = self.continuation_cut(stage)
                grant = trace.grant(key, mode=mode, source=1, capture=capture)
                trace.result(grant, source=1, status=600)
                audit.parse_semantic_audit(trace.text())
                trace.rows[grant - 1]["capture"] = 202 if capture == 101 else 101
                with self.subTest(stage=stage, mode=mode), self.assertRaisesRegex(
                        ValueError, "stage-specific cut"):
                    audit.parse_semantic_audit(trace.text())

    def test_removed_or_wrong_stage_continuation_binding_cannot_authorize_progress(self):
        for stage in (1, 3, 4, 6):
            trace, key, capture = self.continuation_cut(stage)
            trace.actor(6, **key, action=12, binding=1 if stage in (1, 3) else 2,
                        source=1, capture=capture)
            grant = trace.grant(key, mode=2, source=1, capture=capture)
            trace.result(grant, source=1, status=600)
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())

    def test_only_source_role_can_own_ordering_turns_and_frontier_bindings(self):
        for role in (1, 2, 3):
            trace = Trace(actor=False)
            trace.rows[1].update(role=role, effective=0)
            for event in (1, 2, 3):
                trace.ordering(event)
            trace.finish(actor=False)
            with self.subTest(role=role, owner="ordering"), self.assertRaisesRegex(
                    ValueError, "actual source connection"):
                audit.parse_semantic_audit(trace.text())
            trace = Trace()
            trace.peer(2, role=role, effective=0)
            key = trace.admit()
            trace.attach(key, source=2)
            with self.subTest(role=role, owner="turn"), self.assertRaisesRegex(
                    ValueError, "actual source connection"):
                audit.parse_semantic_audit(trace.text())
            trace, key, _ = self.continuation_cut(1)
            trace.peer(2, role=role, effective=0)
            trace.actor(6, **key, binding=1, action=7, source=2, capture=303)
            with self.subTest(role=role, owner="frontier"), self.assertRaisesRegex(
                    ValueError, "actual source connection"):
                audit.parse_semantic_audit(trace.text())
            trace = Trace()
            trace.peer(2, role=role, effective=0)
            key = trace.admit()
            trace.attach(key)
            grant = trace.grant(key, mode=2, source=1, leaf=2)
            trace.result(grant, source=2, status=600)
            with self.subTest(role=role, owner="grant"), self.assertRaisesRegex(
                    ValueError, "actual source connection"):
                audit.parse_semantic_audit(trace.text())

    def test_foreign_edge_stale_wrong_owner_capture_and_truncation(self):
        for change in (dict(source_capture=102), dict(target_capture=202), dict(source=2),
                       dict(target=1), dict(event_object=52), dict(record_sequence=2),
                       dict(index=1), dict(target_sequence=2)):
            trace, _ = self.foreign_grant()
            trace.rows[-1].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        trace, grant = self.foreign_grant()
        trace.rows.pop()
        trace.result(grant, source=1, issue_state=1, command_sequence=1)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_parent_rejection_requires_explicit_shape_and_exact_actual_wait(self):
        trace, grant = self.foreign_grant()
        root = next(row for row in trace.rows if row["serial"] == grant)
        root.update(path_kind=1, leaf_source=2, capture=0)
        trace.result(grant, source=2, issue_state=2, command_sequence=1, native_status=1, status=1)
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertTrue(any("failed runtime grant" in reason for reason in parsed.unresolved))
        for changes in (dict(path_kind=0), dict(path_kind=2), dict(mode=1), dict(edges=0)):
            saved = dict(root)
            root.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
            root.clear()
            root.update(saved)
        trace.rows[-1]["command_sequence"] = 2
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_ordering_record_local_and_foreign_exact_old_generation(self):
        trace = Trace(actor=False)
        trace.peer(2)
        trace.ordering(1)
        trace.ordering(2)
        # A later re-record cannot replace the immutable generation selected below.
        later = dict(sequence=2, capture=102, record_sequence=2, event_capture=102)
        trace.ordering(1, **later)
        trace.ordering(2, **later)
        local = dict(sequence=3, opcode=52, capture=103)
        foreign = dict(connection=2, sequence=1, opcode=54, capture=201)
        for fields in (local, foreign):
            trace.ordering(1, **fields)
            if fields["opcode"] == 54:
                trace.ordering(4, **fields)
            trace.ordering(2, **fields)
            trace.ordering(3, **fields)
        trace.ordering(3)
        trace.ordering(3, **later)
        trace.finish(actor=False)
        self.assertTrue(audit.parse_semantic_audit(trace.text(), require_complete=True).is_complete)

    def test_missing_or_wrong_event_generation_never_aliases_current(self):
        for fields in (dict(event_capture=0), dict(event_capture=102), dict(event_object=52),
                       dict(record_sequence=2), dict(event_source=3)):
            trace = Trace(actor=False)
            trace.peer(2)
            trace.ordering(1)
            row = dict(connection=2, opcode=54, capture=201)
            row.update(fields)
            trace.ordering(1, **row)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        trace = Trace(actor=False)
        trace.ordering(1, opcode=52, record_sequence=0, event_capture=101)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())

    def test_initial_event_and_known_rejection_are_not_compute_or_success(self):
        trace = Trace(actor=False)
        fields = dict(opcode=52, record_sequence=0, event_capture=0)
        trace.ordering(1, **fields)
        trace.ordering(3, **fields, status=1)
        trace.finish(actor=False)
        result = audit.parse_semantic_audit(trace.text())
        self.assertFalse(result.is_complete)
        self.assertTrue(any("ordering status 1" in reason for reason in result.unresolved))
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text(), require_complete=True)

    def test_ordering_orphans_repeats_and_opcode_whitelist(self):
        for events in ((2,), (3,), (1, 1), (1, 2, 2), (1, 2, 3, 3), (1, 4)):
            trace = Trace(actor=False)
            for event in events:
                trace.ordering(event)
            with self.subTest(events=events), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        for opcode in (4, 129, 131, 132):
            trace = Trace(actor=False)
            trace.ordering(1, opcode=opcode)
            with self.subTest(opcode=opcode), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())

    def test_peer_association_and_cross_process_priority(self):
        trace = Trace(actor=False)
        trace.peer(2, peer=0)
        trace.peer(2, event=3)
        trace.peer(2, event=2)
        trace.finish(actor=False)
        self.assertTrue(audit.parse_semantic_audit(trace.text(), require_complete=True).is_complete)
        for change in (dict(pid=801), dict(start=901), dict(uid=1001), dict(priority=9)):
            trace = Trace(actor=False)
            trace.peer(2, **change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())

    def test_source_error_keeps_attempt_sequence_and_separate_library_error(self):
        for fields in (dict(issue_state=3, command_sequence=7, native_status=1),
                       dict(issue_state=1, command_sequence=7, has_library=1, library_status=13)):
            trace = Trace()
            key = trace.admit()
            trace.attach(key)
            grant = trace.grant(key, mode=2, source=1, capture=0)
            trace.source_progress(key, grant, **fields)
            result = audit.parse_semantic_audit(trace.text())
            self.assertTrue(any("failed runtime grant" in reason for reason in result.unresolved))
            self.assertEqual(result.family(2)[-1].value("command_sequence"), 7)

    def test_source_cohort_actions_require_one_exact_grant_and_all_members(self):
        trace = Trace()
        key = trace.admit()
        grant = trace.grant(key)
        inputs = dict(input_reservation=9, input_generation=1, input_cutoff=key["cutoff"])
        trace.actor(3, **key, **inputs, action=2)
        for action in (1, 2, 3):
            for index, stream in enumerate((4, 5)):
                trace.actor(7, **key, action=action, binding=3, source=1,
                            capture=301, stream=stream, index=index, count=2)
        trace.actor(4, **key, **inputs, action=2)
        trace.result(grant)
        result = audit.parse_semantic_audit(trace.text())
        self.assertFalse(any("cohort" in reason for reason in result.unresolved))
        # Merely removing one member is not hidden by successful events for the other.
        rows = [dict(row) for row in trace.rows]
        target = next(index for index, row in enumerate(trace.rows)
                      if row["family"] == 1 and row["event"] == 7 and
                      row["action"] == 3 and row["index"] == 1)
        trace.rows.pop(target)
        trace.renumber()
        result = audit.parse_semantic_audit(trace.text())
        self.assertTrue(any("cohort" in reason for reason in result.unresolved))
        trace.rows = rows
        target = next(row for row in trace.rows if row["family"] == 1 and row["event"] == 7)
        target["capture"] = 0
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text())


class CleanupOwnerTests(unittest.TestCase):
    def prepared_parent(self, *, helper_role=None):
        trace = Trace()
        connection = 1
        if helper_role is not None:
            connection = 2
            trace.peer(connection, role=helper_role, effective=0)
        trace.request(7, group=10)
        trace.cutoff += 1
        continuation = dict(subject=4, id=1, peer=1, priority=7, caller0=7, caller1=8,
                            group=10, ordinary_sequence=1, reservation=9, generation=1,
                            cutoff=trace.cutoff, stage=1, connection=connection, wire_sequence=41,
                            ordinary_opcode=22, origin_source=0)
        trace.actor(2, **continuation, action=1, accepted=1, sequence_consumed=1)
        parent = dict(subject=5, peer=1, priority=7, caller0=7, caller1=8,
                      group=10, ordinary_sequence=1, reservation=9, generation=1,
                      connection=connection, allocation=72, bound=1)
        trace.actor(14, **parent)
        return trace, parent

    def test_historical_cleanup_group_and_helper_connection_are_not_source_authority(self):
        trace, _ = self.prepared_parent(helper_role=1)
        trace.request(8, group=10)
        trace.cutoff += 1
        cleanup = dict(subject=4, id=2, peer=1, priority=7, caller0=7, caller1=8,
                       group=10, ordinary_sequence=1, cleanup_parent=1, origin_source=0,
                       reservation=9, generation=1, cutoff=trace.cutoff, stage=1,
                       connection=2, wire_sequence=42, ordinary_opcode=23)
        trace.actor(2, **cleanup, action=1, accepted=1, sequence_consumed=1)
        # The helper is a legitimate continuation/parent owner, but the captured
        # CUDA frontier still belongs to the independent source connection.
        trace.actor(6, **cleanup, action=7, binding=1, source=1, capture=101)
        audit.parse_semantic_audit(trace.text())
        trace.rows[-1]["source"] = 2
        with self.assertRaisesRegex(ValueError, "actual source connection"):
            audit.parse_semantic_audit(trace.text())

    def test_exact_cleanup_seal_membership_and_actual_retirement(self):
        trace, parent = self.prepared_parent()
        ticket = dict(subject=6, id=88, ticket=88, peer=1, priority=7, connection=1, count=1)
        trace.actor(15, **ticket, action=13)
        trace.actor(14, **parent, action=13, ticket=88, index=0, count=1)
        trace.actor(15, **ticket, action=14, status=600)
        result = audit.parse_semantic_audit(trace.text())
        self.assertTrue(any("cleanup ticket" in reason for reason in result.unresolved))
        trace.actor(14, **parent, action=11, ticket=88, consumed=1)
        trace.actor(14, **parent, action=12, consumed=1)
        trace.actor(15, **ticket, action=14, retired=1)
        result = audit.parse_semantic_audit(trace.text())
        self.assertFalse(any("cleanup ticket" in reason for reason in result.unresolved))
        self.assertFalse(result.is_complete)  # Other actual group ownership is still open.

    def test_cleanup_status_or_disappearance_cannot_substitute_for_retirement(self):
        for transition in (dict(event=14, action=12),
                           dict(event=15, action=14, subject=6, id=88, ticket=88,
                                count=1, retired=1)):
            trace, parent = self.prepared_parent()
            fields = dict(parent)
            fields.update(transition)
            event = fields.pop("event")
            trace.actor(event, **fields)
            with self.subTest(transition=transition), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
        trace, parent = self.prepared_parent()
        trace.actor(14, **parent, unknown=1)
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertTrue(any("unknown cleanup" in reason for reason in parsed.unresolved))


class HostFunctionSemanticControls(unittest.TestCase):
    @staticmethod
    def disconnected(*, submit_after_disconnect=False):
        trace = Trace()
        trace.peer(2, role=3, effective=0, retained=0, policy=12)
        trace.request(7, group=10)
        trace.continuation(opcode=134, has_original=False)
        if submit_after_disconnect:
            dependent = trace.request(9, group=10, physical=False)
            for row in trace.rows:
                if (row["family"] == 1 and row["subject"] == 3 and
                        row["id"] == dependent["id"]):
                    row["origin_source"] = 0
                    row["caller_primary_domain"] = 1
                    if row["event"] == 7:
                        row["stream"] = 17
            blocked = next(index for index, row in enumerate(trace.rows)
                           if row["family"] == 1 and row["subject"] == 3 and
                           row["id"] == dependent["id"] and row["event"] == 5 and
                           row["stage"] == 2)
            trace.peer(1, event=2)
            # Preserve the accepted turn and complete returned setup grant, then
            # disconnect BEFORE the later valid native grant begins that turn.
            trace.rows.insert(blocked + 3, trace.rows.pop())
        else:
            dependent = trace.admit(9, group=10)
            dependent["caller_primary_domain"] = trace.rows[-1]["caller_primary_domain"] = 1
            trace.attach(dependent, stream=17)
            trace.peer(1, event=2)
        trace.peer(2, event=2, role=3, effective=0, retained=0, policy=12)
        trace.actor(17, peer=1)
        trace.actor(18)
        serials = {row["serial"]: index for index, row in enumerate(trace.rows, 1)}
        for row in trace.rows:
            if row["family"] == 2 and "grant" in row:
                row["grant"] = serials[row["grant"]]
        trace.renumber()
        return trace

    def test_enqueue_is_valid_but_observation_and_reserved_opcodes_remain_rejected(self):
        trace = Trace()
        trace.request(7, group=10)
        trace.continuation(opcode=134, has_original=False)
        trace.request(8, group=10)
        trace.batch(remove=True)
        trace.finish()
        parsed = audit.parse_semantic_audit(trace.text(), require_complete=True)
        self.assertTrue(parsed.is_complete)
        self.assertEqual(parsed.unresolved, ())
        for opcode in (0, 131, 132, 133, 135, 136, 137, 138, 142, (1 << 32) - 1):
            malformed = Trace()
            malformed.request(7, group=10)
            malformed.continuation(opcode=opcode, has_original=False)
            with self.subTest(opcode=opcode), self.assertRaisesRegex(ValueError, "ordinary continuation"):
                audit.parse_semantic_audit(malformed.text())

    def test_exact_original_blas_extensions_preserve_scoped_continuation(self):
        for opcode in (139, 140):
            trace = Trace()
            trace.request(7, group=10)
            trace.continuation(opcode=opcode, has_original=True)
            trace.request(8, group=10)
            trace.batch(remove=True)
            trace.finish()
            with self.subTest(opcode=opcode):
                parsed = audit.parse_semantic_audit(trace.text(), require_complete=True)
                self.assertTrue(parsed.is_complete)
                self.assertEqual(parsed.unresolved, ())
                admission = next(row for row in trace.rows if row["family"] == 1 and
                                 row["subject"] == 4 and row["event"] == 2)
                admission["ordinary_opcode"] = 138
                with self.assertRaisesRegex(ValueError, "ordinary continuation"):
                    audit.parse_semantic_audit(trace.text())

    def test_generic_copy_continuation_keeps_existing_host_owner_role(self):
        trace = Trace()
        # PAG1 uses the same HostOwner role1 as HST1/STG1. A helper is not
        # promoted to an ordinary source or assigned a new semantic role.
        trace.peer(2, role=1, effective=0)
        helper = trace.rows[-1]
        trace.request(7, group=10)
        trace.continuation(opcode=141, has_original=True)
        trace.request(8, group=10)
        trace.batch(remove=True)
        trace.finish()
        parsed = audit.parse_semantic_audit(trace.text(), require_complete=True)
        self.assertTrue(parsed.is_complete)
        self.assertEqual(parsed.unresolved, ())
        for changed in (dict(role=4), dict(effective=1)):
            original = dict(helper)
            helper.update(changed)
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                audit.parse_semantic_audit(trace.text())
            helper.clear()
            helper.update(original)

    def test_exact_blocked_disconnect_is_diagnostic_not_healthy_completion(self):
        trace = self.disconnected()
        parsed = check_host_function_disconnect(trace.text(), (7, 8))
        dependent = next(record for record in parsed.family(1)
                         if record.value("event") == 2 and record.value("native_kind") == 9)
        self.assertEqual(dependent.value("origin_source"), 0)
        self.assertEqual(dependent.value("caller_primary_domain"), 1)
        self.assertFalse(parsed.is_complete)
        self.assertTrue(any("entity (3," in reason for reason in parsed.unresolved))
        self.assertIn("service end not observed", parsed.unresolved)
        with self.assertRaises(ValueError):
            audit.parse_semantic_audit(trace.text(), require_complete=True)

    def test_missing_preparation_or_retirement_witness_cannot_use_pending_count(self):
        for family, event, subject, action in ((1, 2, 4, 1), (1, 7, 3, 1),
                                                (1, 5, 3, 0), (4, 2, None, None)):
            trace = self.disconnected()
            selected = next(index for index, row in enumerate(trace.rows)
                            if row["family"] == family and row["event"] == event and
                            (subject is None or row["subject"] == subject) and
                            (action is None or row["action"] == action) and
                            (family != 1 or subject != 3 or row["native_kind"] == 9))
            del trace.rows[selected]
            trace.renumber()
            with self.subTest(family=family, event=event), self.assertRaises(ValueError):
                check_host_function_disconnect(trace.text(), (7, 8))

    def test_wrong_source_capture_caller_and_fabricated_native_reply_are_rejected(self):
        for change in (dict(source=2), dict(stream=0), dict(capture=0), dict(count=2)):
            trace = self.disconnected()
            turn = next(row for row in trace.rows if row["family"] == 1 and
                        row["event"] == 7 and row["action"] == 1)
            turn.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_host_function_disconnect(trace.text(), (7, 8))
        trace = self.disconnected()
        with self.assertRaises(ValueError):
            check_host_function_disconnect(trace.text(), (8, 7))
        for row in trace.rows:
            if row["family"] == 1 and row["subject"] == 4 and row["event"] == 11:
                row["has_original"] = 1
        with self.assertRaises(ValueError):
            check_host_function_disconnect(trace.text(), (7, 8))

    def test_source_must_retire_after_blocked_turn_while_native_carrier_is_alive(self):
        for retire_before_blocked in (True, False):
            trace = self.disconnected()
            source = next(index for index, row in enumerate(trace.rows)
                          if row["family"] == 4 and row["event"] == 2 and row["connection"] == 1)
            if retire_before_blocked:
                target = next(index for index, row in enumerate(trace.rows)
                              if row["family"] == 1 and row["event"] == 5 and
                              row["subject"] == 3 and row["stage"] == 2)
                trace.rows.insert(target, trace.rows.pop(source))
            else:
                carrier = next(index for index, row in enumerate(trace.rows)
                               if row["family"] == 4 and row["event"] == 2 and
                               row["connection"] == 2)
                trace.rows[source], trace.rows[carrier] = trace.rows[carrier], trace.rows[source]
            trace.renumber()
            with self.subTest(early_source=retire_before_blocked), self.assertRaises(ValueError):
                check_host_function_disconnect(trace.text(), (7, 8))

    def test_generic_accepted_work_after_disconnect_cannot_launder_missing_callback(self):
        trace = self.disconnected(submit_after_disconnect=True)
        # Generic accepted native work may outlive a socket. This scenario's
        # absent callback is a stronger, independently enforced prohibition.
        parsed = audit.parse_semantic_audit(trace.text())
        self.assertFalse(parsed.is_complete)
        with self.assertRaisesRegex(ValueError, "cannot begin, submit"):
            check_host_function_disconnect(trace.text(), (7, 8))


if __name__ == "__main__":
    unittest.main()
