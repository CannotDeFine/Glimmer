"""Reject forged source order, callback completion, images and copy inventories."""

import json
import unittest

import h2h_order_receipt as oracle

UUID = "GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2"


def receipt():
    cases = []
    for ordinal in range(24):
        route = ordinal % 4
        source, destination = (2 if route & 1 else 0), (3 if route & 2 else 1)
        old = bool(ordinal % 3)
        images = [bytearray([0x80 + image] * 256) for image in range(4)]
        images[source][17:145] = bytes([0x5d]) * 128
        images[destination][43:171] = bytes([0x11 if old else 0x5d]) * 128
        cases.append(dict(ordinal=ordinal, **{"async": bool(ordinal // 12)}, stream_kind=ordinal // 4 % 3,
                          route=route, status=0, callback_calls=1, copied_before_callback=old,
                          callback_complete_at_return=not old, watchdog_before_return=not old,
                          images=[image.hex() for image in images]))
    return dict(record="h2h_order_result", schema=1, cached=False, gpu_uuid=UUID, driver_version=12060,
                case_count=24, normal_cleanup=True, full_device_qualified=False,
                priority_qualified=False, formal_comparison_qualified=False, cases=cases)


class OrderTests(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(ValueError):
            oracle.parse_receipt(json.dumps(value), expected_cached=False)

    def test_complete_both_observed_alternatives(self):
        oracle.parse_receipt(json.dumps(receipt()), expected_cached=False, expected_gpu_uuid=UUID)

    def test_source_and_destination_every_guard(self):
        for image, offset in ((0, 0), (0, 17), (1, 43), (1, 171), (2, 255), (3, 100)):
            value = receipt()
            image_bytes = bytearray.fromhex(value["cases"][0]["images"][image])
            image_bytes[offset] ^= 1
            value["cases"][0]["images"][image] = image_bytes.hex()
            self.reject(value)

    def test_false_phase_cannot_choose_its_own_bytes(self):
        value = receipt()
        value["cases"][0]["copied_before_callback"] = True
        self.reject(value)

    def test_callbacks_and_types(self):
        for field, wrong in (("callback_calls", 0), ("callback_calls", 2), ("callback_calls", True),
                             ("callback_complete_at_return", 1), ("watchdog_before_return", 0),
                             ("status", 1), ("route", 4), ("ordinal", 1)):
            value = receipt()
            value["cases"][0][field] = wrong
            self.reject(value)
        value = receipt()
        value["cases"][0]["watchdog_before_return"] = False
        self.reject(value)

    def test_inventory_cleanup_identity(self):
        for field, wrong in (("cases", []), ("normal_cleanup", False), ("case_count", 23),
                             ("cached", True), ("gpu_uuid", "wrong"), ("driver_version", True)):
            value = receipt()
            value[field] = wrong
            self.reject(value)

    def test_duplicate_and_trailing(self):
        valid = json.dumps(receipt())
        for raw in (valid + "{}", valid[:-1] + ',"case_count":24}', " " * (oracle.MAX_RECEIPT_BYTES + 1)):
            with self.assertRaises(ValueError):
                oracle.parse_receipt(raw, expected_cached=False)

    def test_exact_fixed_issue_ledger(self):
        lines = []
        for sequence in range(1, 25):
            lines.extend((f"Native admission audit: session=1, sequence={sequence}, accepted=true, operation=141",
                          f"Native execution audit: session=1, issued={sequence}, retained=0, retired=1, status=0"))
        self.assertEqual(oracle.validate_execution("\n".join(lines))["issued_commands"], 24)
        for rows in (lines[:-1], lines + [lines[-1]], list(reversed(lines))):
            with self.assertRaises(ValueError):
                oracle.validate_execution("\n".join(rows))

    def test_fixed_count_type_bounds(self):
        from unified_copy_execution_receipt import validate_execution
        for count in (False, 0, 257, "24", 24.0):
            with self.assertRaises(ValueError):
                validate_execution("", expected_commands=count)


if __name__ == "__main__":
    unittest.main()
