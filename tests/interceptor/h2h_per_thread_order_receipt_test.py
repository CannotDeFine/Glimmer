"""Reject phase, complete-image, identity, selection and issue forgeries."""

import json
import unittest

import h2h_per_thread_order_receipt as oracle

UUID = "GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2"


def fixture(cached=False):
    cases = []
    for ordinal in range(8):
        route, old = ordinal % 4, ordinal % 2 != 0
        source, destination = (2 if route & 1 else 0), (3 if route & 2 else 1)
        images = [bytearray([0x80 + image] * 256) for image in range(4)]
        images[source][17:145] = bytes([0x5d]) * 128
        images[destination][43:171] = bytes([0x11 if old else 0x5d]) * 128
        cases.append(dict(ordinal=ordinal, **{"async": bool(ordinal // 4)}, stream_kind=3,
                          route=route, status=0, callback_calls=1,
                          callback_complete_at_return=not old, watchdog_before_return=not old,
                          copied_before_callback=old, images=[image.hex() for image in images]))
    return dict(record="h2h_per_thread_order_result", schema=1, cached=cached, gpu_uuid=UUID,
                driver_version=12060, case_count=8, normal_cleanup=True,
                full_device_qualified=False, priority_qualified=False,
                formal_comparison_qualified=False, cases=cases)


class CallbackTests(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(ValueError):
            oracle.parse_receipt(json.dumps(value), expected_cached=False)

    def test_complete_direct_cached(self):
        for cached in (False, True):
            value = fixture(cached)
            self.assertEqual(oracle.parse_receipt(json.dumps(value), expected_cached=cached,
                                                 expected_gpu_uuid=UUID), value)

    def test_each_complete_image_guard_and_phase(self):
        for ordinal in range(8):
            for image in range(4):
                for index in (0, 16, 17, 42, 43, 144, 145, 170, 171, 255):
                    value = fixture()
                    raw = bytearray.fromhex(value["cases"][ordinal]["images"][image])
                    raw[index] ^= 1
                    value["cases"][ordinal]["images"][image] = raw.hex()
                    self.reject(value)

    def test_causality_exact_types_inventory_and_callback_count(self):
        for key, wrong in (("stream_kind", 0), ("route", 1), ("async", 0), ("callback_calls", 2),
                            ("status", 1), ("callback_complete_at_return", 1),
                            ("watchdog_before_return", False), ("copied_before_callback", True)):
            value = fixture()
            value["cases"][0][key] = wrong
            self.reject(value)
        for key, wrong in (("normal_cleanup", False), ("cached", True), ("gpu_uuid", "wrong"),
                            ("driver_version", True), ("case_count", 7), ("cases", [])):
            value = fixture()
            value[key] = wrong
            self.reject(value)

    def test_strict_json(self):
        raw = json.dumps(fixture())
        for wrong in (raw + "{}", raw[:-1] + ',"schema":1}',
                      raw.replace('"status": 0', '"status": NaN', 1), b"\xff"):
            with self.assertRaises((ValueError, UnicodeError)):
                oracle.parse_receipt(wrong, expected_cached=False)

    def test_exact_eight_generic_issues(self):
        rows = []
        for sequence in range(1, 9):
            rows.extend((f"Native admission audit: session=7, sequence={sequence}, accepted=true, operation=141",
                         f"Native execution audit: session=7, issued={sequence}, retained=0, retired=1, status=0"))
        raw = "\n".join(rows)
        self.assertEqual(oracle.validate_execution(raw)["issued_commands"], 8)
        for wrong in ("\n".join(rows[:-1]), raw + "\n" + rows[-1],
                      raw.replace("operation=141", "operation=23", 1),
                      raw.replace("status=0", "status=1", 1), "\n".join(reversed(rows))):
            with self.assertRaises(ValueError):
                oracle.validate_execution(wrong)


if __name__ == "__main__":
    unittest.main()
