"""Independent completion/public demand joins; no GPU, SDK or collector imports."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from execution_priority_audit import parse_block_completions
from application_priority_audit import analyze_application_priority
from application_priority_audit_test import admission, event, setup, high_work
from semantic_ordering_fixture import SemanticTrace


def block():
    return ("Block completion audit: v=1, session=1, sequence=2, priority=0, pid=811, "
            "start=911, uid=1000, total_blocks=12, physical_launches=1, physical_stops=0, "
            "stopped_completed_blocks=0, workers=320, has_completed_block=true\n")


def fixture(extra_before="", completed=3):
    trace = SemanticTrace(actor=False)
    trace.peer(2, peer=12, priority=100, kernel_mode=1, effective=0, retained=0)
    trace.append(setup())
    offset = trace.position
    trace.append(admission(1, 2, 42) + event("low_issued", 1, 2, 0, 12, 1))
    trace.append(extra_before + block() + "External release boundary\n")
    demand = trace.position
    trace.append(event("retreat_requested", 1, 2, 0, 12, 1) +
                 event("low_physical_stop", 1, 2, completed, 12, 1) + high_work() +
                 event("low_resumed", 1, 2, completed, 12, 2) +
                 event("low_logical_done", 1, 2, 12, 12, 2))
    return trace.finish(), offset, demand


class BlockCompletionAudit(unittest.TestCase):
    def test_complete_same_command_before_external_demand(self):
        text, offset, demand = fixture()
        self.assertEqual(len(parse_block_completions(text)), 1)
        proof = analyze_application_priority(text, offset, demand_position=demand)
        self.assertTrue(proof["pre_demand_compute_observed"])
        self.assertFalse(proof["full_application_physical_priority_qualified"])
        self.assertEqual(len(proof["pre_demand_handoff_components"]), 1)
        self.assertEqual(proof["pre_demand_handoff_components"][0]["low"],
                         dict(session=1, sequence=2, operation=42))
        self.assertEqual(proof["missing_evidence"], [])

    def test_no_external_boundary_or_no_publication(self):
        text, offset, demand = fixture()
        self.assertFalse(analyze_application_priority(text, offset)["pre_demand_compute_observed"])
        proof = analyze_application_priority(text.replace(block(), ""), offset,
                                             demand_position=demand - len(block()))
        self.assertFalse(proof["pre_demand_compute_observed"])

    def test_strict_identity_and_native_state(self):
        for old, new in (("v=1", "v=2"), ("sequence=2", "sequence=3"),
                         ("session=1", "session=3"), ("priority=0", "priority=100"),
                         ("pid=811", "pid=812"), ("start=911", "start=912"),
                         ("uid=1000", "uid=1001"), ("total_blocks=12", "total_blocks=13"),
                         ("physical_launches=1", "physical_launches=2"),
                         ("physical_stops=0", "physical_stops=1"),
                         ("stopped_completed_blocks=0", "stopped_completed_blocks=1"),
                         ("workers=320", "workers=0"),
                         ("has_completed_block=true", "has_completed_block=false")):
            text, _, _ = fixture()
            with self.subTest(old=old), self.assertRaises(ValueError):
                parse_block_completions(text.replace(block(), block().replace(old, new)))

    def test_duplicate_and_unparsed_trailing_fields(self):
        text, _, _ = fixture()
        for replacement in (block() * 2, block().rstrip() + ", extra=true\n", block().rstrip() + "junk\n"):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                parse_block_completions(text.replace(block(), replacement))

    def test_first_issue_and_authenticated_configuration_required(self):
        text, _, _ = fixture()
        changes = [text.replace(event("low_issued", 1, 2, 0, 12, 1), ""),
                   text.replace(admission(1, 2, 42), admission(1, 2, 25)),
                   text.replace('"effective":1', '"effective":0'),
                   '\n'.join(line for line in text.splitlines() if "Execution semantic audit" not in line)]
        for changed in changes:
            with self.subTest(changed=changed[:60]), self.assertRaises(ValueError):
                parse_block_completions(changed)

    def test_after_retreat_stop_resume_or_done_is_not_first_generation(self):
        text, _, _ = fixture()
        for transition in (event("retreat_requested", 1, 2, 0, 12, 1),
                           event("low_physical_stop", 1, 2, 3, 12, 1),
                           event("low_resumed", 1, 2, 3, 12, 2),
                           event("low_logical_done", 1, 2, 12, 12, 2)):
            changed = text.replace(block(), "").replace(transition, transition + block())
            with self.subTest(transition=transition), self.assertRaises(ValueError):
                parse_block_completions(changed)

    def test_demand_precedes_publication_or_boundary_is_malformed(self):
        text, offset, _ = fixture()
        publication = parse_block_completions(text)[0]
        for boundary in (offset, True, publication["position"] + 1, len(text), -1):
            with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                analyze_application_priority(text, offset, demand_position=boundary)
        proof = analyze_application_priority(text, offset, demand_position=publication["position"])
        self.assertFalse(proof["pre_demand_compute_observed"])

    def test_earlier_high_admission_is_not_hidden_by_later_handoff(self):
        text, offset, demand = fixture(admission(2, 90, 11))
        proof = analyze_application_priority(text, offset, demand_position=demand)
        self.assertFalse(proof["pre_demand_compute_observed"])

    def test_warmup_or_zero_progress_is_not_a_measured_useful_handoff(self):
        text, _, demand = fixture()
        publication = parse_block_completions(text)[0]
        proof = analyze_application_priority(text, publication["end_position"], demand_position=demand)
        self.assertFalse(proof["pre_demand_compute_observed"])
        text, offset, demand = fixture(completed=0)
        proof = analyze_application_priority(text, offset, demand_position=demand)
        self.assertFalse(proof["pre_demand_compute_observed"])


if __name__ == "__main__":
    unittest.main()
