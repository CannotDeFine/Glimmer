"""Compose real authenticated CPU service transitions with the offline decoder."""

import pathlib
import subprocess
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from execution_semantic_audit import parse_semantic_audit
from execution_priority_audit import _parse_audit
from execution_application_receipt import execution_application_receipt


SERVICE_FIXTURE = None


def host_function_admission(audit, caller):
    """Select the exact fixture callback, never an unrelated ordinary request."""
    accepted = [record for record in audit.family(1)
                if record.value("event") == 2 and record.value("subject") == 4 and
                record.value("accepted") and record.value("ordinary_opcode") == 134 and
                (record.value("caller0"), record.value("caller1")) == caller]
    if len(accepted) != 1:
        raise ValueError("expected one exact accepted host-function continuation")
    return accepted[0]


def check_host_function_disconnect(content, caller):
    """Require exact retained-owner evidence, not healthy or GPU qualification.

    The fixture's pending-count wait is only pacing. Missing SourceBlocked or
    source-turn evidence before disconnection fails this independent audit.
    """
    audit = parse_semantic_audit(content, require_complete=False)
    callback = host_function_admission(audit, caller)
    actor = audit.family(1)

    def one(records, description):
        if len(records) != 1:
            raise ValueError("expected one " + description)
        return records[0]

    def require(condition, description):
        if not condition:
            raise ValueError(description)

    scope = ("peer", "caller0", "caller1", "group", "reservation", "generation")
    dependent = one([record for record in actor
                     if record.value("event") == 2 and record.value("subject") == 3 and
                     record.value("accepted") and record.value("native_kind") == 9 and
                     all(record.value(name) == callback.value(name) for name in scope)],
                    "accepted native dependent in the callback's exact group")
    source = callback.value("connection")
    require(source > 0 and callback.value("origin_source") == source,
            "callback continuation must name its exact source")
    # Native origin_source is the caller's entry context, which is zero after
    # observed-primary selection. It is not the collective's source owner:
    # only this request's accepted source turn below supplies that exact join.
    callback_reply = one([record for record in actor
                          if record.value("subject") == 4 and
                          record.value("id") == callback.value("id") and
                          record.value("event") == 11], "host-function admission reply")
    require(callback_reply.value("reply_ready") == 1 and
            callback_reply.value("disposition") == 2 and
            callback_reply.value("native_status") == 0 and
            not callback_reply.value("has_original"),
            "host-function enqueue requires a successful synthetic reply, not native execution")
    records = [record for record in actor if record.value("subject") == 3 and
               record.value("id") == dependent.value("id")]
    accepted_turn = one([record for record in records if record.value("event") == 7 and
                         record.value("action") == 1 and record.value("binding") == 3],
                        "accepted source turn for the exact native dependent")
    require(accepted_turn.value("source") == source and accepted_turn.value("stream") > 0 and
            accepted_turn.value("capture") > 0 and accepted_turn.value("index") == 0 and
            accepted_turn.value("count") == 1 and accepted_turn.value("status") == 0,
            "native dependent requires its exact owned-stream source-turn capture")
    blocked = one([record for record in records if record.value("event") == 5 and
                   record.value("previous_stage") == 1 and record.value("stage") == 2],
                  "Prepared-to-SourceBlocked transition")
    retired = one([record for record in audit.family(4) if record.value("event") == 2 and
                   record.value("connection") == source and record.value("role") == 0],
                  "exact source retirement")
    require(retired.value("peer") == callback.value("peer") and
            callback.serial < callback_reply.serial < dependent.serial < accepted_turn.serial <
            blocked.serial < retired.serial,
            "accepted callback, source turn and SourceBlocked must precede exact source retirement")
    carrier = one([record for record in audit.family(4) if record.value("event") == 1 and
                   record.value("role") == 3 and record.value("peer") == callback.value("peer")],
                  "authenticated live native carrier")
    carrier_retired = one([record for record in audit.family(4) if record.value("event") == 2 and
                           record.value("connection") == carrier.value("connection")],
                          "later native-carrier retirement")
    require(carrier.serial < dependent.serial and retired.serial < carrier_retired.serial,
            "native carrier must remain alive through exact source retirement")
    require(not any(record.value("event") in (8, 9, 11, 12, 13) or
                    record.value("native_operation") or record.value("native_sequence") or
                    record.value("has_original") or
                    (record.value("event") == 7 and record.value("action") != 1)
                    for record in records),
            "disconnected unrun callback cannot begin, submit, reply or retire its native suffix")
    states = [record for record in records if record.value("event") == 5]
    require(states[-1].value("stage") == 2 and not audit.is_complete and audit.unresolved and
            f"entity (3, {dependent.value('id')}) retained" in audit.unresolved and
            not any(record.value("event") == 2 for record in audit.family(2)) and
            not any(record.value("event") == 19 for record in actor),
            "retained SourceBlocked owner must remain explicitly incomplete and unqualified")
    return audit


class SemanticService(unittest.TestCase):
    def test_actual_service_lifecycles(self):
        for scenario in ("ordinary", "deferred", "foreign", "host", "host-high", "async", "async-two",
                         "host-function"):
            with self.subTest(scenario=scenario):
                result = subprocess.run(
                    [SERVICE_FIXTURE, "--semantic", scenario],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, timeout=45, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout[-12000:])
                self.assertIn("Actual semantic service audit scenario passed", result.stdout)
                audit = parse_semantic_audit(result.stdout, require_complete=True)
                self.assertTrue(audit.is_complete)
                self.assertFalse(audit.unresolved)
                receipt = execution_application_receipt(result.stdout, expected_workers=320)
                self.assertTrue(receipt["semantic_complete"])
                self.assertGreater(receipt["native_submission_count"], 0)
                self.assertFalse(receipt["physical_priority_qualified"])
                if scenario == "host-high":
                    self.assertGreater(receipt["source_granted_issue_attempts"], 0,
                                       "actual retained-batch source progress must be exercised")
                admissions, events, _ = _parse_audit(result.stdout)
                ordering_commands = {(record.value("connection"), record.value("sequence"))
                                     for record in audit.family(3) if record.value("event") == 1}
                self.assertTrue(ordering_commands <= admissions.keys())
                self.assertEqual(sum(event["event"] == "event_issued" for _, event in events),
                                 sum(record.value("event") == 2 for record in audit.family(3)))
                self.assertTrue(audit.family(1), "actual actor transitions must be visible")
                self.assertTrue(audit.family(4), "actual process bindings must be visible")
                if scenario == "host-function":
                    host_function_admission(audit, (337, 347))
                grants = [record for record in audit.family(2) if record.value("event") == 10]
                self.assertTrue(grants, "Core authorization cannot be inferred from host replies")
                self.assertTrue(any(record.value("has_original") for record in audit.family(1)))
                if scenario == "foreign":
                    self.assertTrue(any(record.value("opcode") == 54 for record in audit.family(3)))
                if scenario == "host":
                    self.assertTrue(any(record.value("subject") == 4 for record in audit.family(1)))
                # The same genuine receipt must reject truncation and record
                # loss; a final service message cannot repair missing ownership.
                lines = result.stdout.splitlines(keepends=True)
                semantic = [index for index, line in enumerate(lines)
                            if "Execution semantic audit: " in line]
                self.assertGreater(len(semantic), 3)
                for omitted in (semantic[1], semantic[-1]):
                    broken = "".join(line for index, line in enumerate(lines) if index != omitted)
                    with self.assertRaises(ValueError):
                        parse_semantic_audit(broken, require_complete=True)

    def test_disconnected_host_function_retains_exact_blocked_native_owner(self):
        result = subprocess.run(
            [SERVICE_FIXTURE, "--semantic", "host-function-disconnect"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=45, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-12000:])
        self.assertIn("Actual semantic service audit scenario passed", result.stdout)
        check_host_function_disconnect(result.stdout, (349, 353))
        with self.assertRaises(ValueError):
            parse_semantic_audit(result.stdout, require_complete=True)
        with self.assertRaises(ValueError):
            execution_application_receipt(result.stdout, expected_workers=320)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: execution_semantic_service_test.py SERVICE_FIXTURE")
    SERVICE_FIXTURE = str(pathlib.Path(sys.argv.pop()).resolve(strict=True))
    unittest.main()
