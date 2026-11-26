"""Pure adversarial controls for the exact original generic-copy issue ledger."""

import unittest

from unified_copy_execution_receipt import MAX_BYTES, validate_execution


def ledger():
    result = ["[info] Execution service ready: fixture only"]
    for sequence in range(1, 201):
        result += [f"[info] Native admission audit: session=7, sequence={sequence}, "
                   "accepted=true, operation=141",
                   f"[info] Native execution audit: session=7, issued={sequence}, "
                   "retained=1, retired=0, status=0"]
    return result


class LedgerTest(unittest.TestCase):
    def test_complete(self):
        result = validate_execution("\n".join(ledger()).encode())
        self.assertEqual(result, {
            "operation": 141, "admitted_commands": 200, "issued_commands": 200,
            "actual_generic_copy_selection": True, "physical_retirement_qualified": False,
            "full_device_qualified": False, "priority_qualified": False,
            "performance_qualified": False,
        })
        # Independently ordered admissions/issues need not be adjacent. Neither
        # a zero retained count nor a nonzero retired count qualifies retirement.
        lines = ledger()
        batched = lines[:1] + lines[1::2] + lines[2::2]
        self.assertEqual(validate_execution("\n".join(batched)), result)
        observed = "\n".join(lines).replace("retained=1, retired=0", "retained=0, retired=1")
        self.assertEqual(validate_execution(observed), result)

    def test_missing_every_record(self):
        original = ledger()
        for index in range(1, len(original)):
            with self.subTest(index=index, mutation="missing"):
                with self.assertRaises(ValueError):
                    validate_execution("\n".join(original[:index] + original[index + 1:]))

    def test_actual_duplicates(self):
        original = ledger()
        for index in range(1, len(original)):
            with self.subTest(index=index), self.assertRaises(ValueError):
                validate_execution("\n".join(original[:index] + [original[index]] + original[index:]))

    def test_reject_mutations(self):
        original = "\n".join(ledger())
        for before, after in (("operation=141", "operation=21"),
                              ("status=0", "status=700"),
                              ("status=0", "status=false"),
                              ("sequence=1,", "sequence=0,"),
                              ("issued=1,", "issued=201,"),
                              ("session=7", "session=0"),
                              ("session=7", f"session={1 << 64}"),
                              ("retained=1", f"retained={1 << 64}"),
                              ("accepted=true", "accepted=false"),
                              ("operation=141", "operation=141 trailing")):
            with self.subTest(after=after), self.assertRaises(ValueError):
                validate_execution(original.replace(before, after, 1))
        for extra in ("[error] failure", "[info] Native invented audit: session=7",
                      "[info] Native persistent audit: session=7, sequence=1, "
                      "total_blocks=1, physical_launches=1, workers=320",
                      "[info] preparation_status=1"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_execution(original + "\n" + extra)

    def test_reject_issue_before_admission(self):
        lines = ledger()
        lines[1], lines[2] = lines[2], lines[1]
        with self.assertRaises(ValueError):
            validate_execution("\n".join(lines))

    def test_reject_reordered_identities(self):
        original = ledger()
        # Whole pairs used to pass set-only identity checks despite reversed
        # source-command order. Also independently test admission/issue order.
        pairs = original[:1] + original[3:5] + original[1:3] + original[5:]
        admitted = original[:1] + original[1::2]
        issued = original[2::2]
        wrong_admissions = admitted[:]
        wrong_admissions[1], wrong_admissions[2] = wrong_admissions[2], wrong_admissions[1]
        wrong_issues = issued[:]
        wrong_issues[0], wrong_issues[1] = wrong_issues[1], wrong_issues[0]
        for lines in (pairs, wrong_admissions + issued, admitted + wrong_issues):
            with self.subTest(first=lines[1]), self.assertRaises(ValueError):
                validate_execution("\n".join(lines))

    def test_reject_foreign_session_and_unmatched_issue(self):
        original = ledger()
        for positions in ((1,), (2,), (1, 2), tuple(range(201, 401))):
            lines = original[:]
            for position in positions:
                lines[position] = lines[position].replace("session=7", "session=8")
            with self.subTest(positions=positions[:2]), self.assertRaises(ValueError):
                validate_execution("\n".join(lines))
        for index in range(1, 201):
            lines = original[:]
            lines[index * 2] = lines[index * 2].replace(f"issued={index},", "issued=0,")
            with self.subTest(issue=index), self.assertRaises(ValueError):
                validate_execution("\n".join(lines))

    def test_reject_noncanonical_numbers(self):
        original = "\n".join(ledger())
        for before, after in (("session=7", "session=07"),
                              ("session=7", "session=٧"),
                              ("sequence=1,", "sequence=01,"),
                              ("sequence=1,", "sequence=١,"),
                              ("issued=1,", "issued=01,"),
                              ("retained=1", "retained=01"),
                              ("retained=1", "retained=-1"),
                              ("retired=0", "retired=٠"),
                              ("retired=0", f"retired={1 << 64}"),
                              ("status=0", "status=-0"),
                              ("status=0", "status=00"),
                              ("status=0", "status=٠"),
                              ("status=0", "status=0.0")):
            with self.subTest(after=after), self.assertRaises(ValueError):
                validate_execution(original.replace(before, after, 1))

    def test_malformed_extra_audit_cannot_hide_behind_marker_typo(self):
        original = "\n".join(ledger())
        for extra in ("[info] Native execution audit : session=7, issued=1",
                      "[info] Native admission audi: session=7, sequence=201",
                      "[info] native execution audit: session=7, issued=1",
                      "[info] Native invented audit : session=7",
                      "[info] Native library issue audit: session=7, sequence=1, status=0",
                      "[info] Native kernel pipeline audit: session=7, first=1, second=2, "
                      "both_pending=true"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_execution(original + "\n" + extra)

    def test_admissions_alone_are_not_actual_issue(self):
        original = ledger()
        for lines in (original[:1] + original[1::2], original[:1] + original[2::2]):
            with self.assertRaises(ValueError):
                validate_execution("\n".join(lines))

    def test_timestamp_prefix_and_service_error_controls(self):
        original = "\n".join(ledger())
        self.assertEqual(validate_execution(original.replace("[info]", "[2026-10-24 10:11:12] "
                                                            "[info]"))["issued_commands"], 200)
        for extra in ("[CRITICAL] uncertain owner", "[info] preparation_status=700",
                      "[info] original_status=NaN", "[info] original_status=-0"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_execution(original + "\n" + extra)

    def test_bounds(self):
        for raw in (b"\xff", "x" * (MAX_BYTES + 1), "x" * (32 * 1024 + 1),
                    "\ud800", None, 1, b"", "", bytearray(b"log")):
            with self.subTest(kind=type(raw)), self.assertRaises((ValueError, UnicodeError)):
                validate_execution(raw)


if __name__ == "__main__":
    unittest.main()
