"""Independent CPU controls for exact equal-priority Runtime arrival evidence."""

import copy
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
from execution_arrival_audit import (ARRIVAL, NATIVE, PHASE, native_trace, verify_arrival,
                                     verify_service_arrival, verify_service_pair_profile)
from semantic_ordering_fixture import SemanticTrace


def admission(session):
    return f"Native admission audit: session={session}, sequence=3, accepted=true, operation=11\n"


def event(name, session, completed=0, total=65536, generation=1):
    return (f"Priority execution audit: event={name}, session={session}, priority=0, sequence=3, "
            f"completed_blocks={completed}, total_blocks={total}, physical_launches={generation}, status=0\n")


def launch(serial, client, *, workers=160, original=False):
    args = [client * 1000, 65536 * 32, 0, 1, 8192, 0, 0,
            *([0] * 6 if original else [65536, 1, 1, 0, 99 + client, 199 + client])]
    return dict(event="launch", serial=serial, kind="original" if original else "persistent",
                context=1, module=client, function=client * 11, stream=client,
                workers=65536 if original else workers, arguments=args)


def fixture(learning=False):
    trace = SemanticTrace(actor=False)
    trace.peer(2, peer=22)
    if not learning:
        trace.peer(3, peer=33)
    trace.append(PHASE)
    if learning:
        trace.append(ARRIVAL)
        for client in (1, 2):
            trace.append(admission(client) + event("low_issued", client))
        for client in (1, 2):
            trace.append(event("low_logical_done", client, 65536))
        rows = [dict(event="counter_initialize", serial=1, counter=100, value=0),
                dict(event="counter_initialize", serial=2, counter=101, value=0),
                launch(3, 1), launch(4, 2),
                dict(event="counter_free", serial=5, counter=100),
                dict(event="counter_free", serial=6, counter=101)]
    else:
        trace.append(admission(1) + event("low_issued", 1) + ARRIVAL)
        trace.append(event("retreat_requested", 1) + event("low_physical_stop", 1, 160) +
                     event("low_resumed", 1, 160, generation=2) +
                     admission(2) + event("low_issued", 2) +
                     event("low_logical_done", 1, 65536, generation=2) +
                     event("low_logical_done", 2, 65536) + admission(3) +
                     event("low_issued", 3, generation=0) +
                     event("low_logical_done", 3, 65536, generation=0))
        rows = [dict(event="counter_initialize", serial=1, counter=100, value=0),
                launch(2, 1, workers=320), launch(3, 1),
                dict(event="counter_initialize", serial=4, counter=101, value=0),
                launch(5, 2), dict(event="counter_free", serial=6, counter=100),
                dict(event="counter_free", serial=7, counter=101), launch(8, 3, original=True)]
    count = 2 if learning else 3
    summary = dict(status="passed", stage="pair-profile" if learning else "arrival", cached=False,
                   ordinary_public_children=count, common_contexts=1, common_destroys=1,
                   native_contexts=count + 1, native_destroys=count + 1,
                   phase_native_launches=2 if learning else 4, shared_worker_launches=2,
                   trained_pairs=2, actual_measurements=True, cleanup=True,
                   hardware_co_residency_qualified=False, full_device_qualified=False,
                   priority_qualified=False, performance_qualified=False, full_goal_qualified=False)
    return trace.finish(), rows, summary


def render(parts):
    text, rows, summary = parts
    return text + "".join(NATIVE + json.dumps(row) + "\n" for row in rows) + json.dumps(summary) + "\n"


def official_fixture(original_total=65536):
    text, _, _ = fixture()
    configuration = ("Execution service configuration: scheduler=workload-aware, kernel_mode=preemptive, "
                     "configured_workers=320, nccl_enabled=false; full_device_qualified=false\n")
    native = "Native persistent audit: session=1, sequence=3, total_blocks=65536, physical_launches=1, workers=320\n"
    text = configuration + text.replace(event("low_issued", 1), event("low_issued", 1) + native)
    text = text.replace(event("low_issued", 3, generation=0),
                        event("low_issued", 3, total=original_total, generation=0))
    text = text.replace(event("low_logical_done", 3, 65536, generation=0),
                        event("low_logical_done", 3, original_total, total=original_total, generation=0))
    return text, text.index(PHASE) + len(PHASE), text.index(ARRIVAL), native


def official_pair_fixture():
    trace = SemanticTrace(actor=False)
    trace.peer(2, peer=22)
    trace.append("Execution service configuration: scheduler=workload-aware, kernel_mode=preemptive, "
                 "configured_workers=320, nccl_enabled=false; full_device_qualified=false\n")
    for total, sequence in ((1, 2), (65536, 3)):
        if total == 65536:
            trace.append(PHASE)
        for client in (1, 2):
            trace.append((admission(client) + event("low_issued", client, total=total)).replace(
                "sequence=3", f"sequence={sequence}"))
            trace.append(f"Native persistent audit: session={client}, sequence={sequence}, "
                         f"total_blocks={total}, physical_launches=1, workers=160\n")
        for client in (1, 2):
            trace.append(event("low_logical_done", client, total, total=total).replace(
                "sequence=3", f"sequence={sequence}"))
    text = trace.finish()
    return text, text.index(PHASE) + len(PHASE)


class ArrivalAudit(unittest.TestCase):
    def test_official_pair_profile_joins_both_stages_without_co_residency_credit(self):
        text, phase = official_pair_fixture()
        proof = verify_service_pair_profile(text, phase, [811, 822])
        self.assertEqual(len(proof["command_keys"]), 4)
        self.assertEqual(proof["explicit_workers"], 160)
        self.assertTrue(proof["complete_typed_native_issue_joins"])
        self.assertFalse(proof["hardware_co_residency_qualified"])
        self.assertFalse(proof["native_pointer_trace_qualified"])

    def test_official_pair_profile_rejects_partial_stale_or_wrong_geometry(self):
        text, phase = official_pair_fixture()
        for changed, pids in (
            (text, [811, 823]), (text.replace("workers=160", "workers=320"), [811, 822]),
            (text.replace("scheduler=workload-aware", "scheduler=naive"), [811, 822]),
            (text.replace("full_device_qualified=false", "full_device_qualified=falsejunk"), [811, 822]),
            (text + "[error] failed preparation\n", [811, 822]),
            (text.replace("workers=160", "workers=160junk"), [811, 822]),
            (text.replace("physical_launches=1, workers", "physical_launches=0, workers"), [811, 822]),
            (text.replace("total_blocks=65536, physical_launches=1, workers",
                          "total_blocks=65535, physical_launches=1, workers"), [811, 822]),
            (text.replace(event("low_logical_done", 1, 65536), event("low_logical_done", 1, 160)), [811, 822]),
            (text.replace(event("low_logical_done", 1, 65536), ""), [811, 822]),
            (text.replace("sequence=3, total_blocks=65536, physical_launches=1, workers=160",
                          "sequence=4, total_blocks=65536, physical_launches=1, workers=160"), [811, 822]),
        ):
            with self.subTest(pids=pids, changed=changed[-30:]), self.assertRaises(ValueError):
                verify_service_pair_profile(changed, phase, pids)

    def test_official_service_has_typed_not_native_pointer_authority(self):
        text, phase, arrival, native = official_fixture()
        proof = verify_service_arrival(text, phase, arrival, [811, 822, 833])
        self.assertTrue(proof["typed_same_command_handoff"])
        self.assertFalse(proof["native_pointer_trace_qualified"])
        for changed, pids in ((text, [811, 822, 834]), (text.replace("workers=320", "workers=8"), [811, 822, 833]),
                              (text.replace(native, ""), [811, 822, 833]),
                              (text.replace("scheduler=workload-aware", "scheduler=naive"), [811, 822, 833]),
                              (text.replace(event("low_physical_stop", 1, 160), event("low_physical_stop", 1, 0)), [811, 822, 833])):
            with self.subTest(pids=pids, changed=changed[-30:]), self.assertRaises(ValueError):
                verify_service_arrival(changed, phase, arrival, pids)

    def test_official_original_without_progress_cannot_earn_persistent_credit(self):
        text, phase, arrival, _ = official_fixture(original_total=0)
        proof = verify_service_arrival(text, phase, arrival, [811, 822, 833])
        self.assertEqual(proof["completions"][-1]["physical_generations"], 0)
        self.assertEqual(proof["first_incomplete_stop"]["session"], 1)
        self.assertFalse(proof["native_pointer_trace_qualified"])
        for old, new in (
            (event("low_issued", 3, total=0, generation=0),
             event("low_issued", 3, total=0, generation=1)),
            (event("low_logical_done", 3, total=0, generation=0),
             event("low_logical_done", 3, 65536, generation=0)),
            (event("low_logical_done", 3, total=0, generation=0),
             event("retreat_requested", 3, total=0, generation=0) +
             event("low_logical_done", 3, total=0, generation=0)),
            (event("low_logical_done", 3, total=0, generation=0),
             event("low_logical_done", 3, 1, total=0, generation=0)),
        ):
            changed = text.replace(old, new)
            with self.subTest(old=old), self.assertRaises(ValueError):
                verify_service_arrival(changed, phase, arrival, [811, 822, 833])
        for total in (1, 65535, 65537):
            changed, phase, arrival, _ = official_fixture(original_total=total)
            with self.subTest(total=total), self.assertRaises(ValueError):
                verify_service_arrival(changed, phase, arrival, [811, 822, 833])

    def test_retained_checkpoint_and_original_fallback(self):
        proof = verify_arrival(render(fixture()))
        self.assertEqual(proof["command_keys"], [(1, 3), (2, 3), (3, 3)])
        self.assertEqual(proof["first_incomplete_stop"]["completed"], 160)
        self.assertEqual(proof["checked_values"], 3 * 65536 * 32)

    def test_isolated_pair_is_not_an_arrival_or_priority_claim(self):
        text = render(fixture(True))
        self.assertTrue(verify_arrival(text, learning=True)["complete_native_joins"])
        with self.assertRaises(ValueError):
            verify_arrival(text)

    def test_original_fallback_keeps_shape_not_persistent_epoch_credit(self):
        text = render(fixture())
        for old, new in (
            (event("low_issued", 3, generation=0), event("low_issued", 3, total=0, generation=0)),
            (event("low_issued", 3, generation=0), event("low_issued", 3, generation=1)),
            (event("low_logical_done", 3, 65536, generation=0), event("low_logical_done", 3, 0, generation=0)),
        ):
            with self.subTest(old=old), self.assertRaises(ValueError):
                verify_arrival(text.replace(old, new))

    def test_cached_entry_keeps_the_same_complete_proof(self):
        text, rows, summary = fixture()
        summary["cached"] = True
        self.assertTrue(verify_arrival(render((text, rows, summary)), cached=True)["complete_native_joins"])

    def test_every_bad_lifecycle_or_boundary_rejects(self):
        text = render(fixture())
        replacements = [
            (event("retreat_requested", 1), ""),
            (event("low_physical_stop", 1, 160), event("low_physical_stop", 1, 0)),
            (event("low_physical_stop", 1, 160), event("low_physical_stop", 1, 65536)),
            (event("low_resumed", 1, 160, generation=2), event("low_resumed", 1, 0, generation=2)),
            (event("low_resumed", 1, 160, generation=2), event("low_resumed", 1, 160, generation=3)),
            (event("low_logical_done", 1, 65536, generation=2), event("low_logical_done", 1, 160, generation=2)),
            ("priority=0, sequence=3", "priority=100, sequence=3"),
            (ARRIVAL, ""),
            (PHASE, PHASE + PHASE),
            (admission(2), ""),
            ("\"effective\":1", "\"effective\":0"),
        ]
        for old, new in replacements:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                verify_arrival(text.replace(old, new))

    def test_native_abi_counter_and_namespace_mutations_reject(self):
        for scenario in range(13):
            text, original, summary = fixture()
            rows = copy.deepcopy(original)
            if scenario == 0:
                rows[2]["arguments"][11] = 999
            if scenario == 1:
                rows[2]["arguments"][3] = 99
            if scenario == 2:
                rows[2]["module"] = 99
            if scenario == 3:
                rows[4]["context"] = 99
            if scenario == 4:
                rows[0]["value"] = 3
            if scenario == 5:
                rows[5]["serial"] = 2
            if scenario == 6:
                rows.pop(5)
            if scenario == 7:
                rows[2]["workers"] = 8
            if scenario == 8:
                rows[2]["kind"] = "original"
            if scenario == 9:
                rows[7]["arguments"][11] = 100
            if scenario == 10:
                rows[4]["stream"] = 1
            if scenario == 11:
                rows[0]["value"] = False
            if scenario == 12:
                rows[1]["arguments"][2] = 1 << 32
            with self.subTest(scenario=scenario), self.assertRaises(ValueError):
                verify_arrival(render((text, rows, summary)))

    def test_exact_summary_and_json_types(self):
        for field in ("cached", "cleanup", "trained_pairs", "phase_native_launches", "shared_worker_launches",
                      "actual_measurements", "hardware_co_residency_qualified"):
            text, rows, summary = fixture()
            summary[field] = not summary[field] if type(summary[field]) is bool else summary[field] + 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_arrival(render((text, rows, summary)))
        text = render(fixture())
        for changed in (text.replace('"cached": false', '"cached": false, "cached": false'),
                        text.replace('"native_contexts": 4', '"native_contexts": 4.0'), text.rstrip()):
            with self.assertRaises(ValueError):
                verify_arrival(changed)

    def test_counter_address_reuse_requires_distinct_retired_lifetimes(self):
        rows = [dict(event="counter_initialize", serial=1, counter=100, value=0), launch(2, 1),
                dict(event="counter_free", serial=3, counter=100),
                dict(event="counter_initialize", serial=4, counter=100, value=0), launch(5, 2),
                dict(event="counter_free", serial=6, counter=100)]
        rows[4]["arguments"][11] = 100
        self.assertEqual(len(native_trace(rows, 2)[0]), 2)
        rows[2]["serial"] = 5
        with self.assertRaises(ValueError):
            native_trace(rows, 2)


if __name__ == "__main__":
    unittest.main()
