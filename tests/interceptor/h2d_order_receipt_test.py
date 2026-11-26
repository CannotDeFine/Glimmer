"""Reject forged upload images, phases, return causality and mixed API ledgers."""

import json
import unittest

import h2d_order_receipt as oracle

UUID = "GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2"


def receipt():
    cases = []
    for ordinal in range(36):
        pinned, old = bool(ordinal % 2), bool(ordinal % 3)
        source = 2 if pinned else 0
        images = [bytearray([0x80 + index] * 256) for index in range(4)]
        images[source][17:145] = bytes([0x5d]) * 128
        images[1][43:171] = bytes([0x11 if old else 0x5d]) * 128
        cases.append(dict(ordinal=ordinal, **{"async": bool(ordinal % 12 // 6)},
                          stream_kind=ordinal % 6 // 2, api=ordinal // 12, pinned_source=pinned,
                          status=0, callback_calls=1, callback_complete_at_return=not old,
                          watchdog_before_return=not old, copied_before_callback=old,
                          images=[image.hex() for image in images]))
    return dict(record="h2d_order_result", schema=1, cached=False, gpu_uuid=UUID,
                driver_version=12060, case_count=36, normal_cleanup=True,
                full_device_qualified=False, priority_qualified=False,
                formal_comparison_qualified=False, cases=cases)


def ledger():
    rows = []
    for sequence in range(1, 109):
        case, phase = divmod(sequence - 1, 3)
        middle = 141 if case < 12 else 23 if case % 2 else 25
        operation = middle if phase == 1 else 141
        rows.extend((f"Native admission audit: session=7, sequence={sequence}, accepted=true, operation={operation}",
                     f"Native execution audit: session=7, issued={sequence}, retained=0, retired=1, status=0"))
    return rows


class UploadTests(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(ValueError):
            oracle.parse_receipt(json.dumps(value), expected_cached=False)

    def test_complete_images_and_exact_ledger(self):
        oracle.parse_receipt(json.dumps(receipt()), expected_cached=False, expected_gpu_uuid=UUID)
        self.assertEqual(oracle.validate_execution("\n".join(ledger()))["issued_commands"], 108)

    def test_every_image_and_span_guard(self):
        for image, offset in ((0, 0), (0, 17), (1, 42), (1, 43), (1, 170), (1, 171), (2, 255), (3, 64)):
            value = receipt()
            raw = bytearray.fromhex(value["cases"][0]["images"][image])
            raw[offset] ^= 1
            value["cases"][0]["images"][image] = raw.hex()
            self.reject(value)

    def test_phase_metadata_and_callback_types(self):
        for key, wrong in (("copied_before_callback", True), ("callback_calls", 2),
                            ("callback_complete_at_return", 1), ("watchdog_before_return", False),
                            ("api", 2), ("pinned_source", 0), ("status", 1), ("ordinal", 1)):
            value = receipt()
            value["cases"][0][key] = wrong
            self.reject(value)

    def test_missing_identity_cleanup_and_trailing(self):
        for key, wrong in (("normal_cleanup", False), ("cases", []), ("case_count", 35),
                            ("driver_version", True), ("cached", True), ("gpu_uuid", "wrong")):
            value = receipt()
            value[key] = wrong
            self.reject(value)
        for raw in (json.dumps(receipt()) + "{}", json.dumps(receipt())[:-1] + ',"case_count":36}'):
            with self.assertRaises(ValueError):
                oracle.parse_receipt(raw, expected_cached=False)

    def test_wrong_middle_api_is_not_generic(self):
        valid = "\n".join(ledger())
        for raw in (valid.replace("operation=25", "operation=141", 1),
                    valid.replace("operation=23", "operation=25", 1),
                    valid.replace("operation=141", "operation=21", 1),
                    valid.replace("status=0", "status=1", 1)):
            with self.assertRaises(ValueError):
                oracle.validate_execution(raw)

    def test_incomplete_duplicate_and_reordered_issues(self):
        rows = ledger()
        for wrong in (rows[:-1], rows + [rows[-1]], list(reversed(rows))):
            with self.assertRaises(ValueError):
                oracle.validate_execution("\n".join(wrong))

    def test_operation_inventory_cannot_come_from_wire(self):
        from unified_copy_execution_receipt import validate_operations
        for wrong in ([141], (), (True,), (21,), (141,) * 257):
            with self.assertRaises(ValueError):
                validate_operations("", wrong)


if __name__ == "__main__":
    unittest.main()
