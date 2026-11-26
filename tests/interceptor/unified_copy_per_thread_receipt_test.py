"""Special-handle selection cannot replace the fixed numerical or cleanup oracle."""

import json
import unittest

import unified_copy_per_thread_receipt as oracle
from unified_copy_gpu_receipt_test import UUID, fixture as baseline_fixture


def fixture(cached=False):
    value = baseline_fixture(cached)
    value["record"] = "unified_copy_per_thread_result"
    for ordinal in (*range(15, 20), *range(35, 40)):
        value["cases"][ordinal]["mode"] = "async_per_thread"
    return value


class PerThreadTests(unittest.TestCase):
    def valid(self, value, cached=False):
        return oracle.parse_receipt(json.dumps(value), expected_cached=cached,
                                    expected_gpu_uuid=UUID)

    def reject(self, value):
        with self.assertRaises(ValueError):
            self.valid(value)

    def test_complete_original_inventory_and_bytes(self):
        for cached in (False, True):
            value = fixture(cached)
            self.assertEqual(self.valid(value, cached), value)

    def test_exact_special_handle_selection(self):
        for ordinal in range(40):
            value = fixture()
            value["cases"][ordinal]["mode"] = "async_nonblocking"
            self.reject(value)
        self.reject(baseline_fixture())

    def test_every_full_image_guard(self):
        for ordinal in range(40):
            for name in fixture()["cases"][ordinal]["images"]:
                value = fixture()
                image = bytearray.fromhex(value["cases"][ordinal]["images"][name])
                image[(ordinal * 13) % 256] ^= 1
                value["cases"][ordinal]["images"][name] = image.hex()
                self.reject(value)

    def test_identity_cleanup_types_and_inventory(self):
        for key, wrong in (("cached", True), ("gpu_uuid", "wrong"),
                            ("normal_cleanup", False), ("original_copy_calls", 199),
                            ("full_device_qualified", True), ("case_count", True),
                            ("cases", [])):
            value = fixture()
            value[key] = wrong
            self.reject(value)

    def test_trailing_duplicate_and_nonfinite_json(self):
        raw = json.dumps(fixture())
        for wrong in (raw + "{}", raw[:-1] + ',"schema":1}',
                      raw.replace('"status": 0', '"status": NaN', 1), b"\xff"):
            with self.assertRaises((ValueError, UnicodeError)):
                oracle.parse_receipt(wrong, expected_cached=False)


if __name__ == "__main__":
    unittest.main()
