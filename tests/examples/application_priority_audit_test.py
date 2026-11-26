"""Independent deterministic application priority proofs; no GPU or collectors."""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from application_priority_audit import analyze_application_priority
from semantic_ordering_fixture import SemanticTrace


def admission(session, sequence, operation):
    return (f"Native admission audit: session={session}, sequence={sequence}, "
            f"accepted=true, operation={operation}\n")


def event(name, session, sequence=0, completed=0, total=0, launches=0,
          priority=None, status=0):
    if priority is None:
        priority = 0 if session in (1, 3) else 100
    return (f"Priority execution audit: event={name}, session={session}, priority={priority}, "
            f"sequence={sequence}, completed_blocks={completed}, total_blocks={total}, "
            f"physical_launches={launches}, status={status}\n")


def setup():
    return ("Service ready\n" + event("low_bound", 1) + event("native_bound", 2) +
            admission(1, 1, 25) + event("low_issued", 1, 1) +
            event("low_logical_done", 1, 1) + admission(2, 1, 21) +
            event("high_issued", 2, 1) + event("high_physical_done", 2))


def high_work(session=2, sequence=2, operation=11, has_completion=True):
    return (admission(session, sequence, operation) + event("high_issued", session, sequence) +
            (event("high_physical_done", session) if has_completion else ""))


def low_epoch(high=None, completed=3):
    if high is None:
        high = high_work()
    return (admission(1, 2, 42) + event("low_issued", 1, 2, 0, 12, 1) +
            event("retreat_requested", 1, 2, 0, 12, 1) +
            event("low_physical_stop", 1, 2, completed, 12, 1) + high +
            event("low_resumed", 1, 2, completed, 12, 2) +
            event("low_logical_done", 1, 2, 12, 12, 2))


def fixture(high=None, completed=3):
    prefix = setup()
    return prefix + low_epoch(high, completed), len(prefix)


class ApplicationPriorityAudit(unittest.TestCase):
    def check_unqualified(self, text, offset):
        result = analyze_application_priority(text, offset)
        self.assertEqual(result["handoff_count"], 0)
        self.assertFalse(result["partial_handoff_component_verified"])
        self.assertFalse(result["full_application_physical_priority_qualified"])
        self.assertFalse(result["pre_demand_compute_observed"])
        return result

    def test_one_actual_high_command_is_sufficient_but_never_full_qualification(self):
        text, offset = fixture()
        result = analyze_application_priority(text, offset)
        self.assertEqual(result["handoff_count"], 1)
        self.assertTrue(result["partial_handoff_component_verified"])
        self.assertFalse(result["full_application_physical_priority_qualified"])
        self.assertFalse(result["pre_demand_compute_observed"])
        self.assertIn("only after physical stop", result["missing_evidence"][0])
        gap = result["handoffs"][0]
        self.assertEqual(gap["low"], dict(session=1, sequence=2, operation=42))
        self.assertEqual((gap["generation"], gap["resumed_generation"]), (1, 2))
        self.assertEqual((gap["completed_blocks"], gap["total_blocks"]), (3, 12))
        high = gap["high_commands"][0]
        positions = [gap["issue_position"], gap["retreat_position"], gap["stop_position"],
                     high["issue_position"], high["physical_done_position"],
                     gap["resume_position"], gap["final_position"]]
        self.assertEqual(positions, sorted(set(positions)))
        self.assertEqual(result["low_session_ids"], [1])
        self.assertEqual(result["high_session_ids"], [2])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_all_persistent_capable_compute_kinds_not_only_launch_or_lt(self):
        for low in (11, 30, 31, 32, 42, 56):
            for high in (11, 30, 31, 32, 42, 56):
                text, offset = fixture(high_work(operation=high))
                text = text.replace(admission(1, 2, 42), admission(1, 2, low))
                with self.subTest(low=low, high=high):
                    result = analyze_application_priority(text, offset)
                    self.assertEqual(result["handoff_count"], 1)
                    self.assertEqual(result["handoffs"][0]["high_commands"][0]["operation"], high)

    def test_strided_sgemm_keeps_exact_continuation_and_classification(self):
        text, offset = fixture(high_work(operation=56))
        text = text.replace(admission(1, 2, 42), admission(1, 2, 56))
        result = analyze_application_priority(text, offset)
        self.assertEqual(result["handoff_count"], 1)
        self.assertEqual(result["compute_operation_classes"],
                         dict(persistent_capable=[11, 30, 31, 32, 42, 56], original_only=[129,139,140]))
        self.assertEqual(result["low_original_only_compute_commands"], [])
        self.assertEqual(result["high_original_only_compute_commands"], [])
        self.assertFalse(result["full_application_physical_priority_qualified"])
        self.assertFalse(result["pre_demand_compute_observed"])
        resume = event("low_resumed", 1, 2, 3, 12, 2)
        for replacement in (event("low_resumed", 1, 2, 0, 12, 2),
                            event("low_resumed", 1, 2, 3, 12, 1), ""):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                analyze_application_priority(text.replace(resume, replacement), offset)

    def test_sgemv_sgemmex_original_lifetimes_cannot_supply_persistent_proof(self):
        for operation in (139, 140):
            prefix = setup()
            body = (admission(1, 2, operation) + event("low_issued", 1, 2) +
                    event("nonpreemptible_wait", 1, 2) + event("low_logical_done", 1, 2) +
                    high_work(operation=56))
            with self.subTest(operation=operation):
                result = self.check_unqualified(prefix + body, len(prefix))
                self.assertEqual(result["low_original_only_compute_commands"][0]["operation"],
                                 operation)
                text, offset = fixture(high_work(operation=operation))
                result = self.check_unqualified(text, offset)
                self.assertEqual(result["high_original_only_compute_commands"][0]["operation"],
                                 operation)
                for name in ("low_issued", "nonpreemptible_wait", "low_logical_done"):
                    invalid = body.replace(event(name, 1, 2), event(name, 1, 2, total=1))
                    with self.assertRaises(ValueError):
                        analyze_application_priority(prefix + invalid, len(prefix))
                with self.assertRaises(ValueError):
                    analyze_application_priority(
                        prefix + body.replace(event("low_logical_done", 1, 2), ""), len(prefix))

    def test_geam_native_low_wait_finishes_once_without_a_handoff(self):
        prefix = setup()
        for has_wait in (False, True):
            body = (admission(1, 2, 129) + event("low_issued", 1, 2) +
                    (event("nonpreemptible_wait", 1, 2) if has_wait else "") +
                    event("low_logical_done", 1, 2) + high_work(operation=56))
            with self.subTest(has_wait=has_wait):
                result = self.check_unqualified(prefix + body, len(prefix))
                commands = result["low_original_only_compute_commands"]
                self.assertEqual(commands, result["low_compute_commands"])
                self.assertEqual(len(commands), 1)
                self.assertEqual((commands[0]["operation"], commands[0]["total_blocks"],
                                  commands[0]["final_generation"]), (129, 0, 0))
                self.assertEqual(result["unqualified_gaps"], [])

    def test_geam_only_or_geam_and_memory_high_gap_has_no_useful_compute_proof(self):
        for extra in ("", high_work(sequence=3, operation=25)):
            text, offset = fixture(high_work(operation=129) + extra)
            with self.subTest(extra=extra):
                result = self.check_unqualified(text, offset)
                self.assertEqual(len(result["high_original_only_compute_commands"]), 1)
                self.assertEqual(result["high_compute_commands"],
                                 result["high_original_only_compute_commands"])
                self.assertEqual(result["high_compute_commands"][0]["operation"], 129)
                self.assertIn("original-only compute opcode does not prove useful GPU work",
                              result["unqualified_gaps"][0]["reasons"])

    def test_mixed_high_geam_and_qualified_compute_require_both_physical_completions(self):
        high = high_work(4, 1, 129) + high_work(operation=56)
        text, offset = fixture(high)
        result = analyze_application_priority(text, offset)
        self.assertEqual(result["handoff_count"], 1)
        self.assertEqual([item["operation"] for item in result["handoffs"][0]["high_commands"]],
                         [129, 56])
        self.assertEqual(len(result["high_original_only_compute_commands"]), 1)
        self.assertFalse(result["full_application_physical_priority_qualified"])
        self.assertFalse(result["pre_demand_compute_observed"])
        for completion in ("", event("high_physical_done", 6)):
            invalid = text.replace(event("high_physical_done", 4), completion)
            with self.subTest(completion=completion), self.assertRaisesRegex(
                    ValueError, "while high completion unproven"):
                analyze_application_priority(invalid, offset)
        invalid = text.replace(event("high_physical_done", 4), "") + event("high_physical_done", 4)
        with self.assertRaisesRegex(ValueError, "while high completion unproven"):
            analyze_application_priority(invalid, offset)

    def test_geam_never_accepts_persistent_counters_on_low_or_high_events(self):
        prefix = setup()
        low = (admission(1, 2, 129) + event("low_issued", 1, 2) +
               event("nonpreemptible_wait", 1, 2) + event("low_logical_done", 1, 2))
        for name in ("low_issued", "nonpreemptible_wait", "low_logical_done"):
            for progress in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 12, 2)):
                invalid = low.replace(event(name, 1, 2), event(name, 1, 2, *progress))
                with self.subTest(name=name, progress=progress), self.assertRaises(ValueError):
                    analyze_application_priority(prefix + invalid, len(prefix))
        text, offset = fixture(high_work(operation=129))
        for progress in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 12, 1)):
            invalid = text.replace(event("high_issued", 2, 2), event("high_issued", 2, 2, *progress))
            with self.subTest(progress=progress), self.assertRaises(ValueError):
                analyze_application_priority(invalid, offset)
        persistent, offset = fixture()
        with self.assertRaises(ValueError):
            analyze_application_priority(persistent.replace(admission(1, 2, 42),
                                                            admission(1, 2, 129)), offset)

    def test_geam_zero_counter_retreat_stop_and_resume_are_also_fabricated(self):
        prefix = setup()
        for role, session, issued in (("low", 1, "low_issued"), ("high", 2, "high_issued")):
            for name in ("retreat_requested", "low_physical_stop", "low_resumed"):
                body = (admission(session, 2, 129) + event(issued, session, 2) +
                        event(name, session, 2))
                with self.subTest(role=role, name=name), self.assertRaises(ValueError):
                    analyze_application_priority(prefix + body, len(prefix))

    def test_geam_native_lifecycle_and_setup_frontiers_remain_strict(self):
        prefix = setup()
        body = (admission(1, 2, 129) + event("low_issued", 1, 2) +
                event("nonpreemptible_wait", 1, 2) + event("low_logical_done", 1, 2))
        variants = [body.replace(event("low_logical_done", 1, 2), ""),
                    body + event("low_logical_done", 1, 2),
                    body.replace(event("low_issued", 1, 2), event("low_issued", 1, 2) * 2),
                    body.replace(event("nonpreemptible_wait", 1, 2),
                                 event("nonpreemptible_wait", 1, 2) * 2),
                    body.replace(admission(1, 2, 129), "") + admission(1, 2, 129)]
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                analyze_application_priority(prefix + invalid, len(prefix))
        # A setup-only bad GEAM frontier cannot disappear at the measured boundary.
        bad_setup = (prefix + admission(1, 9, 129) + event("low_issued", 1, 9) +
                     high_work(4, 1) + event("low_logical_done", 1, 9))
        with self.assertRaisesRegex(ValueError, "before proven low physical stop"):
            analyze_application_priority(bad_setup + low_epoch(), len(bad_setup))
        text = prefix + high_work(operation=129, has_completion=False)
        result = self.check_unqualified(text, len(prefix))
        self.assertEqual(result["unproven_high_commands"],
                         result["high_original_only_compute_commands"])

    def test_appended_compute_types_do_not_weaken_malformed_or_unknown_checks(self):
        text, offset = fixture()
        for operation in (55, 57, 128, 130, "129junk", "56.0"):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                analyze_application_priority(text + admission(2, 9, operation), offset)
        for status in (1, -1, "0junk"):
            invalid = text + admission(2, 9, 129) + event("high_issued", 2, 9, status=status)
            with self.subTest(status=status), self.assertRaises(ValueError):
                analyze_application_priority(invalid, offset)

    def test_geam_low_session_rejects_role_or_priority_changes_before_and_after_issue(self):
        prefix = setup()
        body = (admission(1, 2, 129) + event("low_issued", 1, 2) +
                event("nonpreemptible_wait", 1, 2) + event("low_logical_done", 1, 2))
        text = prefix + body
        for name, sequence in (("low_bound", 0), ("low_issued", 1), ("low_logical_done", 1),
                               ("low_issued", 2), ("nonpreemptible_wait", 2), ("low_logical_done", 2)):
            invalid = text.replace(event(name, 1, sequence),
                                   event(name, 1, sequence, priority=100))
            with self.subTest(name=name, sequence=sequence), self.assertRaisesRegex(
                    ValueError, "original-only session changed priority or role"):
                analyze_application_priority(invalid, len(prefix))
        for name in ("native_bound", "high_physical_done"):
            bad_setup = prefix + event(name, 1)
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "original-only session changed priority or role"):
                analyze_application_priority(bad_setup + body, len(bad_setup))

    def test_geam_high_session_rejects_wrong_completion_role_even_with_valid_kernel_work(self):
        high = high_work(operation=129, has_completion=False) + high_work(sequence=3)
        text, offset = fixture(high)
        for name, sequence in (("native_bound", 0), ("high_issued", 1), ("high_issued", 2),
                               ("high_issued", 3), ("high_physical_done", 0)):
            invalid = text.replace(event(name, 2, sequence),
                                   event(name, 2, sequence, priority=0))
            with self.subTest(name=name, sequence=sequence), self.assertRaisesRegex(
                    ValueError, "original-only session changed priority or role"):
                analyze_application_priority(invalid, offset)
        for name in ("low_bound", "low_issued", "nonpreemptible_wait", "low_logical_done"):
            extra = event(name, 2)
            with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "original-only session changed priority or role"):
                analyze_application_priority(extra + text, len(extra) + offset)

    def test_many_high_commands_multiple_sessions_and_two_low_generations(self):
        high = (high_work(sequence=2, operation=30, has_completion=False) +
                high_work(sequence=3, operation=25, has_completion=False) +
                high_work(session=4, sequence=1, operation=42) +
                high_work(sequence=4, operation=11))
        text, offset = fixture(high)
        final = event("low_logical_done", 1, 2, 12, 12, 2)
        second = (event("retreat_requested", 1, 2, 3, 12, 2) +
                  event("low_physical_stop", 1, 2, 8, 12, 2) + high_work(sequence=5) +
                  event("low_resumed", 1, 2, 8, 12, 3) +
                  event("low_logical_done", 1, 2, 12, 12, 3))
        result = analyze_application_priority(text.replace(final, second), offset)
        self.assertEqual(result["handoff_count"], 2)
        self.assertEqual(len(result["handoffs"][0]["high_commands"]), 4)
        self.assertEqual(result["handoffs"][1]["generation"], 2)
        self.assertEqual(result["high_session_ids"], [2, 4])
        self.assertEqual(result["unproven_high_commands"], [])

    def test_multiple_low_commands_and_contexts_each_complete(self):
        text, offset = fixture()
        later = low_epoch(high_work(4, 2, 31)).replace("session=1", "session=3")
        result = analyze_application_priority(text + later, offset)
        self.assertEqual(result["handoff_count"], 2)
        self.assertEqual(result["low_session_ids"], [1, 3])
        self.assertEqual(len(result["low_compute_commands"]), 2)

    def test_memory_only_high_gap_cannot_qualify(self):
        for operation in (21, 23, 25, 141):
            text, offset = fixture(high_work(operation=operation))
            with self.subTest(operation=operation):
                result = self.check_unqualified(text, offset)
                self.assertIn("no typed high compute", result["unqualified_gaps"][0]["reasons"][0])
                self.assertEqual(result["high_compute_commands"], [])
                self.assertEqual(result["high_original_only_compute_commands"], [])

    def test_generic_copy_wait_and_completion_remain_noncompute(self):
        prefix = setup()
        body = (admission(1, 2, 141) + event("low_issued", 1, 2) +
                event("nonpreemptible_wait", 1, 2) + event("low_logical_done", 1, 2) +
                high_work(operation=141))
        result = self.check_unqualified(prefix + body, len(prefix))
        self.assertEqual(result["low_compute_commands"], [])
        self.assertEqual(result["high_compute_commands"], [])
        for progress in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (3, 12, 2)):
            for name in ("low_issued", "nonpreemptible_wait", "low_logical_done"):
                changed = body.replace(event(name, 1, 2), event(name, 1, 2, *progress))
                with self.subTest(progress=progress, name=name), self.assertRaises(ValueError):
                    analyze_application_priority(prefix + changed, len(prefix))
        with self.assertRaises(ValueError):
            analyze_application_priority(prefix + body.replace("operation=141", "operation=142"),
                                         len(prefix))

    def test_native_memory_wait_and_compute_without_a_stop_are_diagnostic(self):
        prefix = setup()
        for operation in (25, 11, 42):
            body = (admission(1, 2, operation) + event("low_issued", 1, 2) +
                    event("nonpreemptible_wait", 1, 2) + event("low_logical_done", 1, 2) +
                    high_work())
            with self.subTest(operation=operation):
                self.check_unqualified(prefix + body, len(prefix))
        # A retreat request can race natural full completion without a partial stop.
        body = (admission(1, 2, 42) + event("low_issued", 1, 2, 0, 12, 1) +
                event("retreat_requested", 1, 2, 0, 12, 1) +
                event("low_logical_done", 1, 2, 12, 12, 1) + high_work())
        self.check_unqualified(prefix + body, len(prefix))

    def test_zero_stop_is_valid_but_full_or_overflow_stop_is_invalid(self):
        text, offset = fixture(completed=0)
        result = self.check_unqualified(text, offset)
        self.assertIn("no positive", result["unqualified_gaps"][0]["reasons"][0])
        for completed in (12, 13):
            text, offset = fixture(completed=completed)
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                analyze_application_priority(text, offset)

    def test_missing_wrong_session_and_earlier_high_completion_never_supply_proof(self):
        cases = [high_work(has_completion=False),
                 event("high_physical_done", 2) + high_work(has_completion=False),
                 high_work(has_completion=False) + event("high_physical_done", 4)]
        for index, high in enumerate(cases):
            text, offset = fixture(high)
            with self.subTest(case=index), self.assertRaisesRegex(
                    ValueError, "low resume while high completion unproven"):
                analyze_application_priority(text, offset)
        text, offset = fixture(high_work(has_completion=False))
        with self.assertRaisesRegex(ValueError, "low resume while high completion unproven"):
            analyze_application_priority(text + event("high_physical_done", 2), offset)

    def test_all_high_work_inside_gap_must_be_proven_not_one_happy_subsequence(self):
        high = high_work() + high_work(4, 1, 25, has_completion=False)
        text, offset = fixture(high)
        with self.assertRaisesRegex(ValueError, "low resume while high completion unproven"):
            analyze_application_priority(text, offset)

    def test_high_compute_issued_before_stop_cannot_be_replaced_by_later_memory(self):
        text, offset = fixture(high_work(sequence=3, operation=25))
        issued = high_work(has_completion=False)
        stop = event("low_physical_stop", 1, 2, 3, 12, 1)
        text = text.replace(stop, issued + stop)
        with self.assertRaisesRegex(ValueError, "before proven low physical stop"):
            analyze_application_priority(text, offset)

    def test_early_high_issue_cannot_hide_behind_a_later_valid_handoff(self):
        for operation in (11, 30, 31, 32, 42, 56, 129, 21, 23, 25, 141):
            for after_request in (False, True):
                text, offset = fixture(high_work(sequence=3))
                marker = (event("retreat_requested", 1, 2, 0, 12, 1) if after_request else
                          event("low_issued", 1, 2, 0, 12, 1))
                invalid = text.replace(marker, marker + high_work(operation=operation))
                with self.subTest(operation=operation, after_request=after_request), \
                        self.assertRaisesRegex(ValueError, "before proven low physical stop"):
                    analyze_application_priority(invalid, offset)

    def test_all_active_low_epochs_need_physical_stop_not_just_one(self):
        for session, sequence in ((1, 3), (3, 1)):
            for native in (False, True):
                text, offset = fixture()
                progress = (0, 0, 0) if native else (0, 20, 1)
                issued = (admission(session, sequence, 25 if native else 30) +
                          event("low_issued", session, sequence, *progress))
                marker = event("low_issued", 1, 2, 0, 12, 1)
                invalid = text.replace(marker, marker + issued)
                invalid += event("low_logical_done", session, sequence,
                                 0 if native else 20, 0 if native else 20, 0 if native else 1)
                with self.subTest(session=session, native=native), self.assertRaisesRegex(
                        ValueError, "before proven low physical stop"):
                    analyze_application_priority(invalid, offset)

    def test_native_and_memory_low_frontiers_need_completion_even_after_wait(self):
        prefix = setup()
        for operation in (11, 30, 31, 32, 42, 56, 129, 21, 23, 25, 141):
            for has_wait in (False, True):
                body = (admission(1, 2, operation) + event("low_issued", 1, 2) +
                        (event("nonpreemptible_wait", 1, 2) if has_wait else "") +
                        high_work() + event("low_logical_done", 1, 2))
                with self.subTest(operation=operation, has_wait=has_wait), self.assertRaisesRegex(
                        ValueError, "before proven low physical stop"):
                    analyze_application_priority(prefix + body, len(prefix))

    def test_new_low_issue_requires_all_high_sessions_to_complete(self):
        prefix = setup()
        for operation in (11, 30, 31, 32, 42, 56, 129, 21, 23, 25, 141):
            for low_operation in (42, 25, 129, 141):
                high = (high_work(4, 1, operation, has_completion=False) + high_work())
                low = (admission(1, 2, low_operation) + event("low_issued", 1, 2) +
                       event("low_logical_done", 1, 2))
                text = prefix + high + low + event("high_physical_done", 4)
                with self.subTest(operation=operation, low_operation=low_operation), \
                        self.assertRaisesRegex(ValueError, "low issue while high completion unproven"):
                    analyze_application_priority(text, len(prefix))

    def test_setup_exclusion_is_checked_before_measurement_filtering(self):
        invalid_setup = (setup() + admission(1, 3, 25) + event("low_issued", 1, 3) +
                         high_work(4, 1) + event("low_logical_done", 1, 3))
        with self.assertRaisesRegex(ValueError, "before proven low physical stop"):
            analyze_application_priority(invalid_setup + low_epoch(), len(invalid_setup))
        invalid_setup = (setup() + high_work(4, 1, has_completion=False) +
                         admission(1, 3, 25) + event("low_issued", 1, 3) +
                         event("high_physical_done", 4) + event("low_logical_done", 1, 3))
        with self.assertRaisesRegex(ValueError, "low issue while high completion unproven"):
            analyze_application_priority(invalid_setup + low_epoch(), len(invalid_setup))

    def test_simultaneously_paused_low_epochs_can_share_one_high_frontier(self):
        prefix = setup()
        first = (admission(1, 2, 42) + event("low_issued", 1, 2, 0, 12, 1) +
                 admission(3, 1, 30) + event("low_issued", 3, 1, 0, 20, 1))
        stops = (event("retreat_requested", 1, 2, 0, 12, 1) +
                 event("low_physical_stop", 1, 2, 3, 12, 1) +
                 event("retreat_requested", 3, 1, 0, 20, 1) +
                 event("low_physical_stop", 3, 1, 5, 20, 1))
        resumes = (event("low_resumed", 1, 2, 3, 12, 2) +
                   event("low_resumed", 3, 1, 5, 20, 2) +
                   event("low_logical_done", 1, 2, 12, 12, 2) +
                   event("low_logical_done", 3, 1, 20, 20, 2))
        result = analyze_application_priority(prefix + first + stops + high_work() + resumes,
                                              len(prefix))
        self.assertEqual(result["handoff_count"], 2)
        self.assertTrue(result["partial_handoff_component_verified"])
        self.assertFalse(result["full_application_physical_priority_qualified"])

    def test_admissions_and_bootstrap_bindings_are_not_physical_issues(self):
        text, offset = fixture()
        high_admission = admission(2, 2, 11)
        text = text.replace(high_admission, "")
        low_issue = event("low_issued", 1, 2, 0, 12, 1)
        text = text.replace(low_issue, low_issue + high_admission + event("native_bound", 4))
        result = analyze_application_priority(text, offset)
        self.assertEqual(result["handoff_count"], 1)
        prefix = setup()
        body = (high_work(has_completion=False) + admission(1, 2, 25) + event("low_bound", 3) +
                event("high_physical_done", 2) + event("low_issued", 1, 2) +
                event("low_logical_done", 1, 2))
        self.check_unqualified(prefix + body, len(prefix))

    def test_missing_old_high_completion_without_later_low_work_stays_diagnostic(self):
        prefix = setup()
        body = high_work(has_completion=False)
        result = self.check_unqualified(prefix + body, len(prefix))
        self.assertEqual(len(result["unproven_high_commands"]), 1)
        self.assertIsNone(result["unproven_high_commands"][0]["physical_done_position"])

    def test_missing_low_final_completion_fails_even_after_an_otherwise_valid_chain(self):
        text, offset = fixture()
        for invalid in (text.replace(event("low_logical_done", 1, 2, 12, 12, 2), ""),
                        text + admission(3, 1, 11) + event("low_issued", 3, 1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                analyze_application_priority(invalid, offset)

    def test_retreat_stop_resume_final_transition_negatives(self):
        text, offset = fixture()
        retreat = event("retreat_requested", 1, 2, 0, 12, 1)
        stop = event("low_physical_stop", 1, 2, 3, 12, 1)
        resume = event("low_resumed", 1, 2, 3, 12, 2)
        done = event("low_logical_done", 1, 2, 12, 12, 2)
        variants = [
            text.replace(retreat, ""), text.replace(retreat, retreat + retreat),
            text.replace(stop, ""), text.replace(stop, stop + stop),
            text.replace(resume, ""), text.replace(resume, resume + resume),
            text.replace(done, done + done),
            text.replace(stop, event("low_physical_stop", 1, 2, 3, 12, 2)),
            text.replace(resume, event("low_resumed", 1, 2, 3, 12, 1)),
            text.replace(resume, event("low_resumed", 1, 2, 3, 12, 3)),
            text.replace(resume, event("low_resumed", 1, 2, 2, 12, 2)),
            text.replace(resume, event("low_resumed", 1, 2, 4, 12, 2)),
            text.replace(resume, event("low_resumed", 1, 2, 3, 13, 2)),
            text.replace(done, event("low_logical_done", 1, 2, 11, 12, 2)),
            text.replace(done, event("low_logical_done", 1, 2, 12, 12, 1)),
            text.replace(retreat, event("retreat_requested", 1, 2, 1, 12, 1)),
            text.replace(stop, stop + done).replace(resume + done, ""),
        ]
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                analyze_application_priority(invalid, offset)

    def test_second_generation_counter_cannot_regress_or_hide_an_invalid_epoch(self):
        text, offset = fixture()
        final = event("low_logical_done", 1, 2, 12, 12, 2)
        invalid = (event("retreat_requested", 1, 2, 3, 12, 2) +
                   event("low_physical_stop", 1, 2, 2, 12, 2) +
                   event("low_resumed", 1, 2, 2, 12, 3) +
                   event("low_logical_done", 1, 2, 12, 12, 3))
        with self.assertRaises(ValueError):
            analyze_application_priority(text.replace(final, invalid), offset)

    def test_issued_identity_admission_and_role_negatives(self):
        text, offset = fixture()
        variants = [text.replace(admission(1, 2, 42), ""),
                    text.replace(admission(2, 2, 11), "") + admission(2, 2, 11),
                    text + admission(1, 2, 42),
                    text + event("high_issued", 2, 2),
                    text.replace(admission(1, 2, 42), admission(1, 2, 25)),
                    text.replace(admission(2, 2, 11), admission(2, 2, 43)),
                    text.replace(event("high_issued", 2, 2),
                                 event("high_issued", 2, 2, priority=0)),
                    text + event("native_bound", 1, priority=100)]
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                analyze_application_priority(invalid, offset)

    def test_malformed_signed_and_setup_errors_are_never_filtered(self):
        text, offset = fixture()
        lines = [event("high_issued", 2, 1, status=-1),
                 event("high_issued", 2, 1, status="0junk"),
                 "Priority execution audit: event=high_issued, session=2\n",
                 "Native admission audit: session=2, sequence=5, accepted=true, operation=11junk\n",
                 "Native kernel pipeline audit: session=2, first=1\n"]
        for line in lines:
            for before in (False, True):
                candidate = line + text if before else text + line
                boundary = offset + len(line) if before else offset
                with self.subTest(line=line, before=before), self.assertRaises(ValueError):
                    analyze_application_priority(candidate, boundary)
        for status in (1, -2147483648):
            line = event("policy_failure", 1, status=status)
            with self.subTest(status=status), self.assertRaises(ValueError):
                analyze_application_priority(line + text, offset + len(line))

    def test_measurement_boundary_never_promotes_a_warmup_chain(self):
        text, _ = fixture()
        prefix = text
        text += admission(1, 3, 25) + event("low_issued", 1, 3) + event("low_logical_done", 1, 3)
        self.check_unqualified(text, len(prefix))
        for offset in (0, -1, True, len(text), len(text) + 1):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                analyze_application_priority(text, offset)
        for priorities in ((0, 0), (100, 0), (True, 100), (-1, 100)):
            with self.subTest(priorities=priorities), self.assertRaises(ValueError):
                analyze_application_priority(text, len(prefix), *priorities)

    def test_spdlog_prefix_full_json_and_optional_pipeline(self):
        high = (high_work(has_completion=False) + high_work(sequence=3, has_completion=False) +
                "Native kernel pipeline audit: session=2, first=2, second=3, both_pending=true\n" +
                event("high_physical_done", 2))
        text, offset = fixture(high)
        prefix = "[2026-10-21 12:00:00.000] [info] "
        before = "".join(prefix + line + "\r\n" for line in text[:offset].splitlines())
        after = "".join(prefix + line + "\r\n" for line in text[offset:].splitlines())
        result = analyze_application_priority(before + after, len(before))
        self.assertEqual(result["handoff_count"], 1)
        for first in (1, 3, 99):
            with self.subTest(first=first), self.assertRaises(ValueError):
                analyze_application_priority(text.replace("first=2", f"first={first}"), offset)


class ApplicationSemanticPriority(unittest.TestCase):
    def assert_no_full_qualification(self, result):
        self.assertFalse(result["full_application_physical_priority_qualified"])
        self.assertFalse(result["pre_demand_compute_observed"])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_ordering_and_geam_share_sessions_without_compute_or_progress_credit(self):
        trace = SemanticTrace(actor=False)
        trace.peer(2, peer=22, priority=100)
        offset = trace.position
        low_record = trace.ordering(1)
        trace.ordering(2, low_record)
        trace.ordering(3, low_record)
        trace.append(admission(1, 1, 129) + event("low_issued", 1, 1) +
                     event("low_logical_done", 1, 1))
        high_record = trace.ordering(1, connection=2, priority=100, event_source=2,
                                     capture=201, event_capture=201)
        trace.ordering(2, high_record)
        trace.ordering(3, high_record)
        trace.append(high_work(sequence=1, operation=129) + low_epoch(high_work(sequence=2)))
        result = analyze_application_priority(trace.finish(), offset)
        self.assertEqual(result["handoff_count"], 1)
        self.assertEqual([command["operation"] for command in result["ordering_commands"]], [48, 48])
        self.assertEqual(len(result["low_original_only_compute_commands"]), 1)
        self.assertEqual(len(result["high_original_only_compute_commands"]), 1)
        self.assertTrue(all(command["retirement_position"] > command["issue_position"]
                            for command in result["ordering_commands"]))
        self.assert_no_full_qualification(result)

    def test_ordering_only_high_gap_is_not_a_compute_handoff(self):
        trace = SemanticTrace(actor=False, source=2, peer=22, priority=100)
        trace.peer(1)
        offset = trace.position
        trace.append(admission(1, 2, 56) + event("low_issued", 1, 2, 0, 12, 1) +
                     event("retreat_requested", 1, 2, 0, 12, 1) +
                     event("low_physical_stop", 1, 2, 3, 12, 1))
        command = trace.ordering(1, connection=2, priority=100, event_source=2,
                                 capture=201, event_capture=201)
        trace.ordering(2, command)
        trace.ordering(3, command)
        trace.append(event("low_resumed", 1, 2, 3, 12, 2) +
                     event("low_logical_done", 1, 2, 12, 12, 2))
        result = analyze_application_priority(trace.finish(), offset)
        self.assertEqual(result["handoff_count"], 0)
        self.assertEqual(result["high_compute_commands"], [])
        self.assertEqual(len(result["ordering_commands"]), 1)
        self.assertIn("no typed high compute", result["unqualified_gaps"][0]["reasons"][0])
        self.assert_no_full_qualification(result)

    def test_ordering_physical_frontier_survives_session_completion_annotation(self):
        for ordering_priority, ordinary_priority in ((0, 100), (100, 0)):
            trace = SemanticTrace(actor=False, source=1, peer=11, priority=ordering_priority)
            trace.peer(2, peer=22, priority=ordinary_priority)
            offset = trace.position
            command = trace.ordering(1, priority=ordering_priority)
            trace.ordering(2, command)
            # A session-level ordinary completion cannot retire a typed event.
            trace.append(event("high_physical_done", 1, priority=ordering_priority)
                         if ordering_priority == 100 else "")
            name = "low_issued" if ordinary_priority == 0 else "high_issued"
            trace.append(admission(2, 1, 56) + event(name, 2, 1, priority=ordinary_priority))
            if ordinary_priority == 0:
                trace.append(event("low_logical_done", 2, 1, priority=0))
            else:
                trace.append(event("high_physical_done", 2, priority=100))
            trace.ordering(3, command)
            with self.subTest(ordering=ordering_priority), self.assertRaisesRegex(
                    ValueError, "ordering frontier"):
                analyze_application_priority(trace.finish(), offset)

    def test_ordering_issue_cannot_cross_ordinary_low_or_high_physical_ownership(self):
        for ordering_priority in (0, 100):
            trace = SemanticTrace(actor=False, source=1, priority=ordering_priority)
            ordinary_priority = 100 - ordering_priority
            trace.peer(2, peer=22, priority=ordinary_priority)
            offset = trace.position
            role = "high" if ordinary_priority == 100 else "low"
            trace.append(admission(2, 1, 56) + event(role + "_issued", 2, 1,
                                                   priority=ordinary_priority))
            command = trace.ordering(1, priority=ordering_priority)
            trace.ordering(2, command)
            trace.ordering(3, command)
            trace.append(event("high_physical_done", 2, priority=100) if role == "high" else
                         event("low_logical_done", 2, 1, priority=0))
            with self.subTest(ordering=ordering_priority), self.assertRaisesRegex(
                    ValueError, "ordering issue"):
                analyze_application_priority(trace.finish(), offset)

    def test_only_typed_retirement_not_early_legacy_annotation_unblocks_issue(self):
        trace = SemanticTrace(actor=False)
        trace.peer(2, peer=22, priority=100)
        offset = trace.position
        command = trace.ordering(1)
        trace.ordering(2, command)
        trace.append(event("event_retired", 1, 10))
        trace.append(high_work(sequence=1, operation=56))
        trace.ordering(3, command)
        with self.assertRaisesRegex(ValueError, "ordering frontier"):
            analyze_application_priority(trace.finish(), offset)

    def test_matching_legacy_ordering_annotations_are_not_a_second_lifecycle(self):
        trace = SemanticTrace(actor=False)
        offset = trace.position
        command = trace.ordering(1)
        trace.append(event("event_accepted", 1, 10))
        trace.append(event("event_issued", 1, 10))
        trace.ordering(2, command)
        trace.append(event("event_retired", 1, 10))
        trace.ordering(3, command)
        text = trace.finish()
        result = analyze_application_priority(text, offset)
        self.assertEqual(len(result["ordering_commands"]), 1)
        self.assertEqual(result["low_compute_commands"], [])
        self.assert_no_full_qualification(result)
        for old, replacement in (
                (event("event_issued", 1, 10), event("event_issued", 1, 10, completed=1)),
                (event("event_retired", 1, 10), event("event_retired", 1, 11)),
                (event("event_accepted", 1, 10), event("event_accepted", 1, 10) * 2)):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                analyze_application_priority(text.replace(old, replacement), offset)

    def test_native_reply_release_and_ordinary_stop_do_not_retire_lower_batch(self):
        for stop_first in (False, True):
            trace = SemanticTrace()
            trace.peer(2, peer=22, priority=100)
            offset = trace.position
            native = trace.native()
            trace.append(event("high_physical_done", 2))
            if stop_first:
                trace.finish_native(native)
            trace.append(high_work(sequence=1, operation=56))
            if not stop_first:
                trace.finish_native(native)
            text = trace.finish()
            if not stop_first:
                with self.assertRaisesRegex(ValueError, "native reservation"):
                    analyze_application_priority(text, offset)
            else:
                result = analyze_application_priority(text, offset)
                self.assertEqual(len(result["native_physical_intervals"]), 1)
                interval = result["native_physical_intervals"][0]
                self.assertLess(interval["stop_position"], result["high_compute_commands"][0]["issue_position"])
                self.assertEqual(result["handoff_count"], 0)
                self.assert_no_full_qualification(result)

    def test_higher_native_reservation_blocks_low_first_issue_and_resume(self):
        for resume in (False, True):
            trace = SemanticTrace(source=2, peer=22, priority=100)
            trace.peer(1)
            offset = trace.position
            if resume:
                trace.append(admission(1, 1, 56) + event("low_issued", 1, 1, 0, 12, 1) +
                             event("retreat_requested", 1, 1, 0, 12, 1) +
                             event("low_physical_stop", 1, 1, 3, 12, 1))
            native = trace.native()
            trace.append(event("high_physical_done", 2))
            trace.append(event("low_resumed", 1, 1, 3, 12, 2) if resume else
                         admission(1, 1, 56) + event("low_issued", 1, 1, 0, 12, 1))
            trace.append(event("low_logical_done", 1, 1, 12, 12, 2 if resume else 1))
            trace.finish_native(native)
            with self.subTest(resume=resume), self.assertRaisesRegex(ValueError, "native reservation"):
                analyze_application_priority(trace.finish(), offset)

    def test_open_group_retains_conservative_reservation_after_member_physical_stop(self):
        for close_first in (False, True):
            trace = SemanticTrace()
            trace.peer(2, peer=22, priority=100)
            offset = trace.position
            trace.native(7, group=77, physical=True)
            if close_first:
                trace.native(8, group=77, physical=True)
            trace.append(high_work(sequence=1, operation=56))
            if not close_first:
                trace.native(8, group=77, physical=True)
            text = trace.finish()
            if close_first:
                result = analyze_application_priority(text, offset)
                self.assertEqual(len(result["native_physical_intervals"]), 1)
                self.assertEqual(result["handoff_count"], 0)
                self.assert_no_full_qualification(result)
            else:
                with self.assertRaisesRegex(ValueError, "native reservation"):
                    analyze_application_priority(text, offset)

    def test_opaque_only_high_native_interval_never_earns_handoff_credit(self):
        trace = SemanticTrace(source=2, peer=22, priority=100)
        trace.peer(1)
        offset = trace.position
        trace.append(admission(1, 1, 56) + event("low_issued", 1, 1, 0, 12, 1) +
                     event("retreat_requested", 1, 1, 0, 12, 1) +
                     event("low_physical_stop", 1, 1, 3, 12, 1))
        native = trace.native()
        trace.finish_native(native)
        trace.append(event("low_resumed", 1, 1, 3, 12, 2) +
                     event("low_logical_done", 1, 1, 12, 12, 2))
        result = analyze_application_priority(trace.finish(), offset)
        self.assertEqual(result["handoff_count"], 0)
        self.assertEqual(result["high_compute_commands"], [])
        self.assertEqual(len(result["native_physical_intervals"]), 1)
        self.assertIn("no typed high compute", result["unqualified_gaps"][0]["reasons"][0])
        self.assert_no_full_qualification(result)

    def test_required_source_is_exact_once_not_any_other_same_peer_command(self):
        for actual_session in (1, 3):
            trace = SemanticTrace()
            trace.peer(3)
            offset = trace.position
            native = trace.admit_native()
            trace.append(admission(1, 20, 56) + admission(3, 20, 56))
            grant = trace.source_issue(native, 1, 20)
            trace.append(event("low_issued", actual_session, 20) +
                         event("low_logical_done", actual_session, 20))
            trace.run_native(native, physical=True)
            text = trace.finish()
            if actual_session == 3:
                with self.assertRaisesRegex(ValueError, "exact actual issue or resume"):
                    analyze_application_priority(text, offset)
            else:
                result = analyze_application_priority(text, offset)
                self.assertEqual(len(result["required_source_issues"]), 1)
                receipt = result["required_source_issues"][0]
                self.assertEqual((receipt["session"], receipt["sequence"], receipt["grant_serial"],
                                 receipt["reservation"], receipt["generation"], receipt["cutoff"]),
                                 (1, 20, grant, 9, 1, 2))
                self.assertEqual((receipt["capture"], receipt["root_source"], receipt["leaf_source"]),
                                 (0, 1, 1))
                rows = [json.loads(line.removeprefix("Execution semantic audit: "))
                        for line in text.splitlines() if line.startswith("Execution semantic audit: ")]
                observe = [row for row in rows if row["family"] == 1 and
                           row["event"] in (3, 4) and row["action"] == 8]
                self.assertEqual([(row["source"], row["capture"]) for row in observe],
                                 [(1, 10001), (1, 10001)])
                self.assertEqual(result["handoff_count"], 0)
                self.assert_no_full_qualification(result)

    def test_required_source_resume_preserves_progress_without_reusable_authority(self):
        trace = SemanticTrace()
        offset = trace.position
        trace.append(admission(1, 20, 56) + event("low_issued", 1, 20, 0, 12, 1) +
                     event("retreat_requested", 1, 20, 0, 12, 1) +
                     event("low_physical_stop", 1, 20, 3, 12, 1))
        native = trace.admit_native()
        trace.source_issue(native, 1, 20)
        trace.append(event("low_resumed", 1, 20, 3, 12, 2) +
                     event("low_logical_done", 1, 20, 12, 12, 2))
        trace.append(admission(1, 21, 56) + event("low_issued", 1, 21) +
                     event("low_logical_done", 1, 21))
        trace.run_native(native, physical=True)
        result = analyze_application_priority(trace.finish(), offset)
        self.assertEqual(len(result["required_source_issues"]), 1)
        self.assertEqual(result["required_source_issues"][0]["event"], "low_resumed")
        self.assertEqual(len(result["low_compute_commands"]), 2)
        self.assertEqual(result["handoff_count"], 0)
        self.assert_no_full_qualification(result)

    def test_required_foreign_leaf_joins_exact_older_generation_and_ordering_issue(self):
        trace = SemanticTrace()
        trace.peer(3)
        offset = trace.position
        original = trace.ordering(1, connection=3, event_source=3, capture=301, event_capture=301)
        later = trace.ordering(1, connection=3, event_source=3, sequence=11,
                               record_sequence=11, capture=302, event_capture=302)
        wait = trace.ordering(1, opcode=54, event_source=3, event_capture=301)
        trace.ordering(4, wait)
        native = trace.admit_native()
        edge = dict(source=3, target=1, event_object=51, record_sequence=10,
                    source_capture=301, target_capture=101, target_sequence=10)
        trace.source_issue(native, 3, 10, root=1, capture=301, path=(edge,))
        trace.ordering(2, original)
        trace.ordering(3, original)
        trace.ordering(2, wait)
        trace.ordering(3, wait)
        trace.ordering(2, later)
        trace.ordering(3, later)
        trace.run_native(native, physical=True)
        text = trace.finish()
        result = analyze_application_priority(text, offset)
        self.assertEqual(len(result["ordering_commands"]), 3)
        self.assertEqual(len(result["required_source_issues"]), 1)
        receipt = result["required_source_issues"][0]
        self.assertEqual((receipt["root_source"], receipt["leaf_source"], receipt["capture"],
                          receipt["event"]), (1, 3, 301, "event_issued"))
        self.assertEqual(receipt["path"], [dict(index=0, **edge)])
        rows = [json.loads(line.removeprefix("Execution semantic audit: "))
                for line in text.splitlines() if line.startswith("Execution semantic audit: ")]
        self.assertFalse(any(row["family"] == 1 and row["event"] in (3, 4) and
                             row["action"] == 8 for row in rows))
        self.assertEqual(result["low_compute_commands"], [])
        self.assert_no_full_qualification(result)
        with self.assertRaises(ValueError):
            analyze_application_priority(text.replace('"source_capture":301', '"source_capture":302'), offset)

    def test_exact_source_receipt_never_bypasses_unrelated_high_ordinary_frontier(self):
        trace = SemanticTrace()
        trace.peer(2, peer=22, priority=100)
        offset = trace.position
        native = trace.admit_native()
        trace.append(high_work(sequence=1, operation=56, has_completion=False))
        trace.append(admission(1, 20, 56))
        trace.source_issue(native, 1, 20)
        trace.append(event("low_issued", 1, 20) + event("low_logical_done", 1, 20) +
                     event("high_physical_done", 2))
        trace.run_native(native, physical=True)
        with self.assertRaisesRegex(ValueError, "while high completion unproven"):
            analyze_application_priority(trace.finish(), offset)

    def test_exact_source_receipt_never_bypasses_another_native_batch(self):
        trace = SemanticTrace()
        trace.peer(2, peer=22, priority=100)
        offset = trace.position
        native = trace.admit_native(reservation=9)
        trace.peer_id, trace.priority = 22, 100
        other = trace.native(reservation=19)
        trace.append(admission(1, 20, 56))
        trace.source_issue(native, 1, 20)
        trace.append(event("low_issued", 1, 20) + event("low_logical_done", 1, 20))
        trace.finish_native(other)
        trace.run_native(native, physical=True)
        with self.assertRaisesRegex(ValueError, "native reservation"):
            analyze_application_priority(trace.finish(), offset)

    def test_required_source_missing_issue_and_stale_root_cutoff_fail_closed(self):
        for missing, stale in ((True, False), (False, True)):
            trace = SemanticTrace()
            offset = trace.position
            native = trace.admit_native()
            trace.append(admission(1, 20, 56))
            if stale:
                native["cutoff"] += 1
            trace.source_issue(native, 1, 20)
            if not missing:
                trace.append(event("low_issued", 1, 20) + event("low_logical_done", 1, 20))
            trace.run_native(native, physical=True)
            with self.subTest(missing=missing, stale=stale), self.assertRaises(ValueError):
                analyze_application_priority(trace.finish(), offset)

    def test_required_source_cannot_change_priority_or_accumulate_unused_receipts(self):
        for duplicate in (False, True):
            trace = SemanticTrace()
            offset = trace.position
            native = trace.admit_native()
            trace.append(admission(1, 20, 56))
            trace.source_issue(native, 1, 20)
            if duplicate:
                trace.source_issue(native, 1, 20)
                trace.append(event("low_issued", 1, 20) + event("low_logical_done", 1, 20))
            else:
                trace.append(event("high_issued", 1, 20, priority=100) +
                             event("high_physical_done", 1, priority=100))
            trace.run_native(native, physical=True)
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(
                    ValueError, "exact actual issue or resume" if duplicate else
                    "authenticated source binding"):
                analyze_application_priority(trace.finish(), offset)

    def test_idle_source_grant_does_not_manufacture_an_issue_or_exemption(self):
        trace = SemanticTrace()
        trace.peer(2, peer=22, priority=100)
        offset = trace.position
        native = trace.admit_native()
        trace.peer_id, trace.priority = 22, 100
        other = trace.native(reservation=19)
        trace.append(admission(1, 20, 56))
        grant = trace.grant(native, mode=2, source=1)
        trace.result(grant, source=1)
        trace.append(event("low_issued", 1, 20) + event("low_logical_done", 1, 20))
        trace.finish_native(other)
        trace.run_native(native, physical=True)
        with self.assertRaisesRegex(ValueError, "native reservation"):
            analyze_application_priority(trace.finish(), offset)

    def test_semantic_errors_and_unknown_ownership_cannot_hide_before_measurement(self):
        trace = SemanticTrace()
        trace.peer(2, peer=22, priority=100)
        native = trace.native()
        trace.finish_native(native)
        offset = trace.position
        trace.append(low_epoch())
        text = trace.finish()
        invalid = (
            text.replace('"has_physical_work":1,"open_group":0,"unknown":0',
                         '"has_physical_work":1,"open_group":0,"unknown":1'),
            text.replace('"family":1', '"family":9', 1),
            text.replace('"original_status":0', '"original_status":1', 1),
            text.replace('"has_physical_work":1', '"has_physical_work":0'),
        )
        for index, candidate in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                analyze_application_priority(candidate, offset)

    def test_native_and_continuation_envelopes_never_become_compute_admissions(self):
        text, offset = fixture()
        for operation in (48, 52, 54, 131, 132):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                analyze_application_priority(text + admission(2, 99, operation), offset)

    def test_native_communicator_identity_appears_only_at_submit_and_stays_bound(self):
        trace = SemanticTrace()
        offset = trace.position
        native = trace.native()
        trace.finish_native(native)
        text = trace.finish()
        submitted = False
        for line in text.splitlines():
            row = json.loads(line.removeprefix("Execution semantic audit: "))
            if row["family"] != 1:
                continue
            communicator = tuple(row[field] for field in
                                 ("communicator_domain", "communicator_context", "communicator_object"))
            if row["subject"] != 3:
                self.assertEqual((communicator, row["output_contract"]), ((0, 0, 0), 0))
                continue
            self.assertEqual(row["output_contract"], 2)
            submitted = submitted or row["event"] == 8
            self.assertEqual(communicator, (1, 1, 1011) if submitted else (0, 0, 0))
        result = analyze_application_priority(text, offset)
        self.assertTrue(submitted)
        self.assertEqual(result["handoff_count"], 0)
        self.assert_no_full_qualification(result)
        for old, new in ((1011, 1012), (1011, 0)):
            invalid = text.replace(f'"communicator_object":{old}',
                                   f'"communicator_object":{new}', 1)
            with self.subTest(communicator=new), self.assertRaises(ValueError):
                analyze_application_priority(invalid, offset)
        invalid = text.replace('"output_contract":2', '"output_contract":1', 1)
        with self.assertRaises(ValueError):
            analyze_application_priority(invalid, offset)


if __name__ == "__main__":
    unittest.main()
