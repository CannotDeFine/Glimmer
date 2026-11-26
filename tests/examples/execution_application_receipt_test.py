"""Current collector composition controls, without CUDA or process launch."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from execution_application_receipt import execution_application_receipt
from semantic_ordering_fixture import SemanticTrace


ADMISSION = "Native admission audit: session=1, sequence=1, accepted=true, operation=56\n"
PERSISTENT = ("Native persistent audit: session=1, sequence=1, total_blocks=1024, "
              "physical_launches=1, workers=320\n")
ISSUE = "Native execution audit: session=1, issued=1, retained=1, retired=0, status=0\n"
LIBRARY = "Native library issue audit: session=1, sequence=1, status=0\n"
DISPATCH = "Native issue audit: session=1, sequence=1, status=0\n"
RETIRE = "Native execution audit: session=1, issued=1, retained=0, retired=1, status=0\n"


def ordinary(*lines):
    trace = SemanticTrace(actor=False)
    for line in lines or (ADMISSION, PERSISTENT, ISSUE):
        trace.append(line)
    return trace.finish()


def receipt(content, **fields):
    return execution_application_receipt(content, expected_workers=320, **fields)


class ApplicationReceipt(unittest.TestCase):
    def test_complete_typed_service_and_actual_persistent_selection(self):
        result = receipt(ordinary())
        self.assertEqual(result["unqualified_actual_issue_count"], 1)
        self.assertEqual(result["unqualified_persistent_issue_count"], 1)
        self.assertEqual(result["admitted_opcode_values"], [56])
        self.assertTrue(result["semantic_complete"])
        for field in ("compute_family_inventory_qualified", "useful_sm_coverage_qualified",
                      "physical_priority_qualified", "saturation_qualified", "performance_qualified"):
            self.assertFalse(result[field])

    def test_semantic_capture_is_required_even_for_successful_legacy_text(self):
        for content in (ADMISSION + PERSISTENT + ISSUE, ordinary().rsplit("Execution semantic audit:", 1)[0]):
            with self.subTest(content=content), self.assertRaises(ValueError):
                receipt(content)

    def test_explicit_positive_worker_geometry_is_not_a_small_default(self):
        for count in (None, True, 0, -1, 1 << 32, "320"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                execution_application_receipt(ordinary(), expected_workers=count)
        with self.assertRaises(ValueError):
            receipt(ordinary().replace("workers=320", "workers=32"))

    def test_exact_admission_and_single_actual_issue_are_required(self):
        for lines in ((ISSUE,), (ADMISSION,), (ADMISSION, ISSUE, ISSUE),
                      (ADMISSION, ADMISSION, ISSUE), (ADMISSION, PERSISTENT, ISSUE.replace("issued=1", "issued=2"))):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                receipt(ordinary(*lines))

    def test_actual_errors_and_unknown_or_malformed_markers_fail(self):
        for extra in ("[error] unexpected failure\n", "status=7\n", "completion_status=-1\n",
                      "status=0junk\n", "Native invented audit: status=0\n", ISSUE + "trailing\n"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                receipt(ordinary(ADMISSION, ISSUE, extra))
        with self.assertRaises(ValueError):
            receipt(ordinary(ADMISSION, ISSUE.replace("status=0", "status=7")))

    def test_noncommand_envelopes_and_duplicate_ordering_admissions_are_not_compute(self):
        for operation in (0, 48, 52, 54, 131, 132):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                receipt(ordinary(ADMISSION.replace("operation=56", f"operation={operation}"), ISSUE))

    def test_original_geam_route_is_retained_without_fake_persistent_credit(self):
        content = ordinary(ADMISSION.replace("operation=56", "operation=129"), LIBRARY, ISSUE)
        result = receipt(content, require_geam=True)
        self.assertEqual(result["unqualified_geam_issue_count"], 1)
        self.assertEqual(result["unqualified_persistent_issue_count"], 0)
        with self.assertRaises(ValueError):
            receipt(content.replace(LIBRARY, PERSISTENT), require_geam=True)
        with self.assertRaises(ValueError):
            receipt(ordinary(), require_geam=True)

    def test_library_selection_and_pipeline_must_precede_and_follow_exact_issue(self):
        with self.assertRaises(ValueError):
            receipt(ordinary(ADMISSION, ISSUE, LIBRARY))
        second = ADMISSION.replace("sequence=1", "sequence=2")
        issued = ISSUE.replace("issued=1", "issued=2")
        pipeline = "Native kernel pipeline audit: session=1, first=1, second=2, both_pending=true\n"
        receipt(ordinary(ADMISSION, ISSUE, second, issued, pipeline))
        for lines in ((ADMISSION, ISSUE, second, pipeline, issued),
                      (ADMISSION, ISSUE, pipeline),
                      (ADMISSION, ISSUE, pipeline.replace("second=2", "second=1"))):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                receipt(ordinary(*lines))

    def test_passive_schema_keeps_issue_and_physical_retirement_distinct(self):
        second_admission = ADMISSION.replace("sequence=1", "sequence=2")
        second_dispatch = DISPATCH.replace("sequence=1", "sequence=2")
        second_retire = RETIRE.replace("issued=1", "issued=2")
        pipeline = "Native kernel pipeline audit: session=1, first=1, second=2, both_pending=true\n"

        def passive(*lines):
            return ordinary(*lines).replace('"incarnation1":4',
                                            '"incarnation1":4,"native_audit_schema":2')

        receipt(passive(ADMISSION, PERSISTENT, DISPATCH, second_admission, second_dispatch,
                        pipeline, RETIRE, second_retire))
        for lines in ((ADMISSION, RETIRE), (ADMISSION, RETIRE, DISPATCH),
                      (ADMISSION, DISPATCH, DISPATCH, RETIRE),
                      (ADMISSION, DISPATCH, second_admission, pipeline, second_dispatch,
                       RETIRE, second_retire)):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                receipt(passive(*lines))
        with self.assertRaises(ValueError):
            receipt(ordinary(ADMISSION, DISPATCH, RETIRE))
        with self.assertRaises(ValueError):
            receipt(passive(ADMISSION, DISPATCH, RETIRE).replace('"native_audit_schema":2',
                                                                  '"native_audit_schema":3'))

    def test_ordering_is_an_independent_lifecycle_not_a_fake_native_issue(self):
        trace = SemanticTrace(actor=False)
        command = trace.ordering(1)
        trace.ordering(2, command)
        trace.ordering(3, command)
        result = receipt(trace.finish())
        self.assertEqual(result["ordering_issue_count"], 1)
        self.assertEqual(result["unqualified_actual_issue_count"], 0)
        self.assertEqual(result["unqualified_persistent_issue_count"], 0)

    def test_grant_bound_source_result_is_an_actual_issue_not_an_extra_command(self):
        trace = SemanticTrace()
        native = trace.admit_native(1)
        trace.append(ADMISSION)
        trace.source_issue(native, 1, 1)
        trace.run_native(native, physical=True)
        result = receipt(trace.finish())
        self.assertEqual(result["unqualified_actual_issue_count"], 1)
        self.assertEqual(result["source_granted_issue_attempts"], 1)
        self.assertEqual(result["native_submission_count"], 1)
        self.assertEqual(result["unqualified_persistent_issue_count"], 0)

    def test_unknown_or_missing_source_command_cannot_hide_behind_native_batch(self):
        trace = SemanticTrace()
        native = trace.admit_native(1)
        trace.source_issue(native, 1, 1)
        trace.run_native(native, physical=True)
        with self.assertRaises(ValueError):
            receipt(trace.finish())

    def test_authenticated_source_cannot_be_replaced_by_metadata_or_another_connection(self):
        content = ordinary()
        for changed in (content.replace("session=1", "session=2"),
                        content.replace('"role":0', '"role":2')):
            with self.subTest(content=changed), self.assertRaises(ValueError):
                receipt(changed)

    def test_priority_annotations_also_pass_the_typed_priority_adapter(self):
        content = ordinary()
        text = ("Priority execution audit: event=low_issued, session=1, priority=0, "
                "sequence=1, completed_blocks=0, total_blocks=1024, physical_launches=1, status=0\n")
        # Insert inside the complete service, not after its end proof.
        content = content.replace(ISSUE, text + ISSUE)
        receipt(content)
        with self.assertRaises(ValueError):
            receipt(content.replace("priority=0", "priority=7"))

    def test_empty_service_and_lost_record_cannot_qualify(self):
        with self.assertRaises(ValueError):
            receipt(ordinary("unrelated service output\n"))
        lines = ordinary().splitlines(keepends=True)
        damaged = [line for line in lines if '"family":4' not in line]
        with self.assertRaises(ValueError):
            receipt("".join(damaged))

    def test_disconnect_retains_exact_admitted_work_but_cannot_admit_more(self):
        trace = SemanticTrace(actor=False)
        trace.append(ADMISSION)
        trace.peer(1, event=2, retained=1)
        trace.append(PERSISTENT + ISSUE)
        result = receipt(trace.finish())
        self.assertEqual(result["unqualified_actual_issue_count"], 1)
        trace = SemanticTrace(actor=False)
        trace.peer(1, event=2, retained=1)
        trace.append(ADMISSION + PERSISTENT + ISSUE)
        with self.assertRaises(ValueError):
            receipt(trace.finish())

    def test_native_records_before_begin_or_after_end_cannot_reuse_old_source_binding(self):
        trace = SemanticTrace(actor=False)
        complete = trace.finish()
        for content in (complete + ADMISSION + ISSUE, ADMISSION + ISSUE + complete):
            with self.subTest(content=content), self.assertRaises(ValueError):
                receipt(content)

    def test_ordering_source_result_requires_prior_admission_and_later_actual_issue(self):
        def build(*, late_admission=False, early_issue=False):
            trace = SemanticTrace()
            native = trace.admit_native(1)
            if not late_admission:
                command = trace.ordering(1, sequence=1, record_sequence=1)
            if early_issue:
                trace.ordering(2, command)
            trace.source_issue(native, 1, 1)
            if late_admission:
                command = trace.ordering(1, sequence=1, record_sequence=1)
            if not early_issue:
                trace.ordering(2, command)
            trace.ordering(3, command)
            trace.run_native(native, physical=True)
            return trace.finish()
        self.assertEqual(receipt(build())["ordering_issue_count"], 1)
        for content in (build(late_admission=True), build(early_issue=True)):
            with self.subTest(content=content), self.assertRaises(ValueError):
                receipt(content)

    def test_granted_geam_library_receipt_does_not_require_an_omitted_legacy_annotation(self):
        def build(*, has_library=1, library_status=0, operation=129, borrow=False):
            trace = SemanticTrace()
            native = trace.admit_native(1)
            trace.append(ADMISSION.replace("operation=56", f"operation={operation}"))
            grant = trace.grant(native, mode=2, source=1)
            trace.result(grant, issue_state=1, command_sequence=1, source=1,
                         has_library=has_library, library_status=library_status)
            if borrow:
                trace.append(ADMISSION.replace("sequence=1", "sequence=2").replace("operation=56", "operation=129"))
                trace.append(ISSUE.replace("issued=1", "issued=2"))
            trace.run_native(native, physical=True)
            return trace.finish()
        result = receipt(build(), require_geam=True)
        self.assertEqual(result["unqualified_geam_issue_count"], 1)
        self.assertEqual(result["unqualified_original_library_issue_count"], 1)
        self.assertEqual(result["unqualified_persistent_issue_count"], 0)
        for content in (build(has_library=0), build(library_status=7),
                        build(operation=56), build(operation=56, borrow=True)):
            with self.subTest(content=content), self.assertRaises(ValueError):
                receipt(content, require_geam=True)


if __name__ == "__main__":
    unittest.main()
