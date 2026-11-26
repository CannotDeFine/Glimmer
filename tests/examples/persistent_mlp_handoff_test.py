"""CPU-only rejection tests for physical MLP handoff evidence."""

import copy
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] /
                       "examples/framework_workloads/pytorch_smoke"))
import run_persistent_mlp_handoff as handoff


def event(phase, timestamp, task=1, generation=1, launch=1, completed=0):
    return dict(phase=phase, time_ns=timestamp, pid=100 + task, task=task,
                generation=generation, launch=launch, logical=8, completed=completed,
                exact=int(phase in {"suspended", "complete"}))


def text(events, control=False):
    lines = [handoff.AUDIT_PREFIX + " ".join(f"{key}={value}" for key, value in item.items())
             for item in events]
    lines += ["[glimmer] persistent_sgemm accepted=1", "status=ok validation=1"]
    if control:
        lines += [handoff.CONTROL_PREFIX + "task=2 generation=0 submit=1 status=0 "
                  "enqueued_ns=21 started_ns=30 finished_ns=35 observed_ns=36"]
    return "\n".join(lines) + "\n"


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.low = [event("grant", 2, launch=0), event("prepare", 3),
                    event("first_tile", 10, completed=1), event("priority_retreat", 40),
                    event("retreat", 41), event("suspended", 50, completed=2),
                    event("grant", 80, generation=2, completed=2),
                    event("prepare", 90, generation=2, launch=2, completed=2),
                    event("first_tile", 100, generation=2, launch=2, completed=3),
                    event("complete", 110, generation=2, launch=2, completed=8)]
        self.high = [event("grant", 60, task=2, launch=0), event("prepare", 61, task=2),
                     event("first_tile", 70, task=2, completed=1),
                     event("complete", 75, task=2, completed=8)]

    def verify(self, low=None, high=None):
        return handoff.verify_handoff(text(self.low if low is None else low),
                                       text(self.high if high is None else high, True), 1, 20)

    def test_real_order(self):
        self.assertEqual(self.verify()["low_partial_stop"]["completed"], 2)

    def test_compute_observation_is_not_a_completed_tile(self):
        low = copy.deepcopy(self.low)
        low.insert(2, event("first_compute", 9))
        low.insert(9, event("first_compute", 95, generation=2, launch=2, completed=2))
        self.assertEqual(self.verify(low=low)["low_partial_stop"]["completed"], 2)
        for index, field, value in ((2, "completed", 1), (2, "exact", 1),
                                    (2, "generation", 2), (9, "completed", 3)):
            bad = copy.deepcopy(low)
            bad[index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                handoff.check_physical_audits(text(bad))
        duplicate = low[:3] + [copy.deepcopy(low[2])] + low[3:]
        with self.assertRaises(ValueError):
            handoff.check_physical_audits(text(duplicate))
        # A computation marker cannot replace the existing full-tile gate.
        without_tile = [row for row in low if row['phase'] != 'first_tile']
        handoff.check_physical_audits(text(without_tile))
        with self.assertRaises(ValueError):
            self.verify(low=without_tile)

    def test_reject_weaker_low_evidence(self):
        cases = [(2, "time_ns", 31), (5, "completed", 0), (5, "completed", 8),
                 (5, "exact", 0), (6, "generation", 1), (7, "time_ns", 72),
                 (9, "completed", 7), (9, "exact", 0), (9, "generation", 3)]
        for index, field, value in cases:
            with self.subTest(index=index, field=field):
                low = copy.deepcopy(self.low)
                low[index][field] = value
                with self.assertRaises(ValueError):
                    self.verify(low=low)

    def test_no_priority_request_or_no_completion(self):
        for index in (2, 3, 5, 7, 9):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.verify(low=self.low[:index] + self.low[index + 1:])

    def test_high_cannot_issue_before_low_stop(self):
        high = copy.deepcopy(self.high)
        high[0]["time_ns"] = 45
        high[1]["time_ns"] = 49
        with self.assertRaises(ValueError):
            self.verify(high=high)

    def test_late_observation_does_not_prove_late_resume(self):
        low = copy.deepcopy(self.low)
        low[6]["time_ns"] = 71
        low[7]["time_ns"] = 72
        with self.assertRaises(ValueError):
            self.verify(low=low)

    def test_reject_malformed_or_missing_inventory(self):
        for bad in (text(self.low).replace("task=1 ", "task=1 task=1 ", 1),
                    text(self.low).replace("persistent_sgemm accepted=1", "declined"),
                    handoff.AUDIT_PREFIX + "task=1\n"):
            with self.assertRaises(ValueError):
                handoff.check_audits(bad)

    def test_exact_stop_proves_short_resume_without_pending_observation(self):
        low = self.low[:8] + [event("suspended", 100, generation=2, launch=2, completed=3),
                             event("grant", 120, generation=3, launch=2, completed=3),
                             event("prepare", 130, generation=3, launch=3, completed=3),
                             event("complete", 140, generation=3, launch=3, completed=8)]
        result = self.verify(low=low)
        self.assertEqual(result["low_resumed"]["phase"], "suspended")
        self.assertEqual(result["low_resumed"]["completed"], 3)
        low[8]["completed"] = 2
        with self.assertRaises(ValueError):
            self.verify(low=low)
        low[8]["completed"] = 3
        low[6]["time_ns"], low[7]["time_ns"] = 71, 72
        with self.assertRaises(ValueError):
            self.verify(low=low)

    def test_partial_live_line_is_not_physical_proof(self):
        self.assertEqual(handoff.records(handoff.AUDIT_PREFIX + "task=1", handoff.AUDIT_PREFIX,
                                          allow_partial=True), [])

    def test_numerical_requires_actual_native_comparison(self):
        low = text(self.low).replace("status=ok validation=1",
                                    "MLP numerical checks passed: 22; native_compared=True")
        handoff.verify_handoff(low, text(self.high, True), 1, 20, numerical=True)
        for bad in (text(self.low), low.replace("22;", "21;"),
                    low.replace("compared=True", "compared=False")):
            with self.assertRaises(ValueError):
                handoff.verify_handoff(bad, text(self.high, True), 1, 20, numerical=True)

    def test_numerical_report_inventory_and_provenance(self):
        checks = []
        for step in range(2):
            checks.extend([f"loss_{step}", f"forward_{step}_0", f"forward_{step}_1"])
            for parameter in ("0.weight", "0.bias", "2.weight", "2.bias"):
                checks.extend([f"gradient_{step}_{parameter}", f"update_{step}_{parameter}"])
        result = dict(status="ok", checks=checks, native_compared=True,
                      application_sha256="source", rtol=2e-4, atol=2e-5)
        handoff.validate_numerical(result, "source")
        for field, value in (("status", "failed"), ("native_compared", False),
                             ("application_sha256", "other"), ("rtol", 1.0),
                             ("checks", checks[:-1]), ("checks", checks[:-1] + [checks[0]])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                handoff.validate_numerical(dict(result, **{field: value}), "source")

    def test_missing_artifact_preserves_failed_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            args = types.SimpleNamespace(output=directory / "result", build=directory / "missing",
                                         python=pathlib.Path(sys.executable), numerical_reference=None)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(handoff.run(args), 1)
            report = json.loads((args.output / "handoff.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["error"])
            self.assertEqual(report["commands"], [])


if __name__ == "__main__":
    unittest.main()
