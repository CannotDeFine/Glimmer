"""Reject false byte, progress, identity and exactly-once H2H claims."""

import copy
import json
import unittest

import h2h_copy_receipt as oracle

UUID = "GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2"


def receipt():
    return dict(record="h2h_copy_result", schema=1, cached=False, gpu_uuid=UUID,
                driver_version=12060, case_count=32, progress_count=4, original_copy_calls=36,
                normal_cleanup=True, full_device_qualified=False, priority_qualified=False,
                formal_comparison_qualified=False, progress=oracle.expected_progress(),
                cases=oracle.expected_cases())


def ledger():
    result = []
    serial = 0
    def semantic(sequence, event):
        nonlocal serial
        serial += 1
        return "Execution semantic audit: " + json.dumps(dict(
            v=1, service=9, serial=serial, family=3, event=event, connection=9, priority=0,
            sequence=sequence, opcode=48, capture=sequence + 20, event_source=9, event_object=11,
            record_sequence=sequence, event_capture=sequence + 20, status=0))
    for sequence in range(1, 45):
        if sequence in (2, 5, 8, 11):
            result.append(semantic(sequence, 1))
            continue
        gate = sequence in (1, 4, 7, 10)
        result.append(f"Native admission audit: session=9, sequence={sequence}, accepted=true, operation={11 if gate else 141}")
        if gate:
            result.append(f"Native persistent audit: session=9, sequence={sequence}, total_blocks=1, physical_launches=1, workers=320")
        result.append(f"Native execution audit: session=9, issued={sequence}, retained=1, retired=0, status=0")
        if sequence in (3, 6, 9, 12):
            result.extend((semantic(sequence - 1, 2), semantic(sequence - 1, 3)))
    return result


class ReceiptTests(unittest.TestCase):
    def test_complete(self):
        oracle.parse_receipt(json.dumps(receipt()), expected_cached=False, expected_gpu_uuid=UUID)
        self.assertEqual(oracle.validate_execution("\n".join(ledger()))["copy_commands"], 36)

    def test_each_image_byte(self):
        valid = receipt()
        for case_index, image_index, byte_index in ((0, 0, 0), (0, 1, 43), (17, 2, 17), (31, 3, 255)):
            value = copy.deepcopy(valid)
            image = bytearray.fromhex(value["cases"][case_index]["images"][image_index])
            image[byte_index] ^= 1
            value["cases"][case_index]["images"][image_index] = image.hex()
            with self.assertRaises(ValueError):
                oracle.parse_receipt(json.dumps(value), expected_cached=False)

    def test_progress_corruption(self):
        for field, wrong in (("watchdog_released", True), ("status", 1), ("physical_event_status", 600),
                             ("initial_event_status", 0), ("source_image", "00" * 256),
                             ("control_image", "00" * 256), ("async", 0), ("pinned_source", 0)):
            value = receipt()
            value["progress"][0][field] = wrong
            with self.assertRaises(ValueError):
                oracle.parse_receipt(json.dumps(value), expected_cached=False)

    def test_false_cleanup_types_inventory(self):
        for field, wrong in (("normal_cleanup", False), ("normal_cleanup", 1), ("cases", []),
                             ("progress", []), ("cached", True), ("case_count", 31),
                             ("original_copy_calls", 35), ("driver_version", True)):
            value = receipt()
            value[field] = wrong
            with self.assertRaises(ValueError):
                oracle.parse_receipt(json.dumps(value), expected_cached=False)

    def test_duplicate_trailing_oversized(self):
        valid = json.dumps(receipt())
        for raw in (valid[:-1] + ',"normal_cleanup":true}', valid + "{}", " " * (oracle.MAX_RECEIPT_BYTES + 1)):
            with self.assertRaises(ValueError):
                oracle.parse_receipt(raw, expected_cached=False)
        with self.assertRaises(ValueError):
            oracle.parse_receipt(valid, expected_cached=False, expected_gpu_uuid="GPU-wrong")

    def test_missing_duplicate_issued_or_gate(self):
        valid = ledger()
        for index in (0, 1, 2, 3, len(valid) - 1):
            for rows in (valid[:index] + valid[index + 1:], valid[:index] + [valid[index]] + valid[index:]):
                with self.assertRaises(ValueError):
                    oracle.validate_execution("\n".join(rows))

    def test_reordered_noncanonical_unknown(self):
        valid = "\n".join(ledger())
        for raw in (valid.replace("workers=320", "workers=32", 1),
                    valid.replace("sequence=3", "sequence=03", 1),
                    valid.replace("operation=141", "operation=25", 1),
                    valid.replace("status=0", "status=1", 1),
                    valid + "\nNative unknown audit: session=9, status=0",
                    "\n".join(reversed(ledger()))):
            with self.assertRaises(ValueError):
                oracle.validate_execution(raw)

    def test_event_capability_corruption(self):
        valid = "\n".join(ledger())
        for before, after in (('"record_sequence": 2', '"record_sequence": 3'),
                              ('"event_source": 9', '"event_source": 10'),
                              ('"opcode": 48', '"opcode": 49'),
                              ('"event_capture": 22', '"event_capture": 23')):
            with self.assertRaises(ValueError):
                oracle.validate_execution(valid.replace(before, after, 1))

    def test_event_enqueue_can_precede_kernel_issue(self):
        rows = ledger()
        event = rows.pop(3)
        rows.insert(2, event)
        self.assertEqual(oracle.validate_execution("\n".join(rows))["event_generations"], 4)


if __name__ == "__main__":
    unittest.main()
