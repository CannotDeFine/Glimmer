"""Pure CPU hostile-input controls; never import CUDA or collector helpers."""

import copy
import json
import unittest

import unified_copy_gpu_receipt as oracle

UUID = "GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2"
DRIVER = 12060


def fixture(cached=False):
    """Independent test builder, not the validator's private ledger/generator."""
    names = ["device_source", "device_destination", "pageable_source",
             "pageable_destination", "pinned_source", "pinned_destination"]
    routes = ["pageable_h2d", "pinned_h2d", "pageable_d2h", "pinned_d2h", "d2d"]
    modes = ["sync_legacy", "async_legacy", "async_blocking", "async_nonblocking"]
    source = [2, 4, 0, 0, 0]
    destination = [1, 1, 3, 5, 1]
    cases = []
    for ordinal in range(40):
        size = 32 if ordinal < 20 else 65
        mode = ordinal // 5 % 4
        route = ordinal % 5
        images = []
        for image in range(6):
            value = bytearray([128 + image] * 256)
            for index in range(16, 240):
                value[index] = (17 * index + 31 * image + 7 * ordinal) & 255
            images.append(value)
        for offset in range(size):
            images[destination[route]][43 + offset] = images[source[route]][17 + offset]
        cases.append({
            "ordinal": ordinal, "route": routes[route], "mode": modes[mode],
            "api": "cuMemcpy" if mode == 0 else "cuMemcpyAsync", "bytes": size,
            "source_image": names[source[route]], "destination_image": names[destination[route]],
            "source_offset": 17, "destination_offset": 43, "status": 0,
            "stream_complete": True, "full_bytes_verified": True, "checked_bytes": 1536,
            "images": dict(zip(names, (image.hex() for image in images), strict=True)),
        })
    return {
        "record": "unified_copy_result", "schema": 1, "cached": cached,
        "gpu_uuid": UUID, "driver_version": DRIVER, "case_count": 40,
        "storage_bytes": 256, "guard_bytes": 16, "image_count": 6,
        "checked_bytes": 61440, "original_copy_calls": 200,
        "sync_copy_calls": 170, "async_copy_calls": 30, "normal_cleanup": True,
        "full_device_qualified": False, "priority_qualified": False,
        "formal_comparison_qualified": False, "cases": cases,
    }


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = fixture()
        cls.encoded = json.dumps(cls.original, separators=(",", ":"))

    def valid(self, value, cached=False):
        return oracle.validate_receipt(value, expected_cached=cached,
                                       expected_gpu_uuid=UUID, expected_driver_version=DRIVER)

    def invalid(self, mutate):
        value = copy.deepcopy(self.original)
        mutate(value)
        with self.assertRaises(oracle.ReceiptError):
            self.valid(value)

    def test_valid_direct_and_cached_complete_images(self):
        self.assertIs(self.valid(self.original), self.original)
        self.assertEqual(self.valid(fixture(True), cached=True), fixture(True))
        self.assertLess(len(self.encoded), oracle.MAX_RECEIPT_BYTES)
        for raw in (self.encoded, (self.encoded + "\n").encode()):
            self.assertEqual(oracle.parse_receipt(raw, expected_cached=False,
                                                expected_gpu_uuid=UUID,
                                                expected_driver_version=DRIVER), self.original)

    def test_every_image_every_case_and_guard_boundary(self):
        for ordinal in range(40):
            for image_index, image in enumerate(self.original["cases"][ordinal]["images"]):
                # Guards, unchanged payload, first/last copied bytes, adjacent
                # untouched byte, and end guard are covered for both lengths.
                # Every other case/image receives an independent mutation too.
                size = self.original["cases"][ordinal]["bytes"]
                boundaries = (0, 15, 16, 17, 42, 43, 43 + size - 1, 43 + size, 239, 240, 255)
                offsets = boundaries if ordinal in (0, 20) else (
                    boundaries[(ordinal + image_index) % len(boundaries)],)
                for offset in offsets:
                    with self.subTest(case=ordinal, image=image, offset=offset):
                        value = copy.deepcopy(self.original)
                        raw = bytearray.fromhex(value["cases"][ordinal]["images"][image])
                        raw[offset] ^= 1
                        value["cases"][ordinal]["images"][image] = raw.hex()
                        with self.assertRaises(oracle.ReceiptError):
                            self.valid(value)

    def test_order_missing_extra_duplicate_and_wrong_case(self):
        mutations = [
            lambda v: v["cases"].pop(),
            lambda v: v["cases"].append(copy.deepcopy(v["cases"][-1])),
            lambda v: v["cases"].reverse(),
            lambda v: v["cases"].__setitem__(1, copy.deepcopy(v["cases"][0])),
            lambda v: v.__setitem__("cases", {}),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.invalid(mutation)

    def test_every_descriptor_field_exact_and_typed(self):
        for ordinal in (0, 4, 5, 19, 20, 39):
            for key, original in self.original["cases"][ordinal].items():
                if key == "images":
                    continue
                changed = not original if type(original) is bool else (
                    original + 1 if type(original) is int else original + "_changed")
                for replacement in (changed, None, float(original) if type(original) is int else 0):
                    with self.subTest(case=ordinal, key=key, replacement=replacement):
                        self.invalid(lambda v, i=ordinal, k=key, r=replacement:
                                     v["cases"][i].__setitem__(k, r))

    def test_summary_counts_cleanup_flags_and_exact_types(self):
        for key, original in self.original.items():
            if key in ("cases", "gpu_uuid", "driver_version"):
                continue
            wrong = not original if type(original) is bool else (
                original + 1 if type(original) is int else original + "_changed")
            with self.subTest(key=key):
                self.invalid(lambda v, k=key, r=wrong: v.__setitem__(k, r))
                self.invalid(lambda v, k=key: v.__setitem__(k, None))
                if type(original) is int:
                    self.invalid(lambda v, k=key, r=float(original): v.__setitem__(k, r))
                if type(original) is bool:
                    self.invalid(lambda v, k=key, r=int(original): v.__setitem__(k, r))

    def test_authenticated_identity_and_lookup_mode(self):
        for key, values in {
            "gpu_uuid": [UUID.upper(), UUID[:-1], "GPU-00000000-0000-0000-0000-000000000000", 1],
            "driver_version": [True, 12060.0, 3999, 1000000, DRIVER + 1],
            "cached": [True, 0, "false"],
        }.items():
            for replacement in values:
                with self.subTest(key=key, value=replacement):
                    self.invalid(lambda v, k=key, r=replacement: v.__setitem__(k, r))
        for kwargs in ({"expected_cached": 0},
                       {"expected_cached": False, "expected_gpu_uuid": "wrong"},
                       {"expected_cached": False, "expected_driver_version": True}):
            with self.assertRaises(oracle.ReceiptError):
                oracle.validate_receipt(self.original, **kwargs)

    def test_missing_extra_fields_at_every_level(self):
        for level in ("summary", "case", "images"):
            for extra in (False, True):
                def mutate(value):
                    target = value if level == "summary" else value["cases"][0]
                    if level == "images":
                        target = target["images"]
                    if extra:
                        target["untrusted_extra"] = True
                    else:
                        target.pop(next(iter(target)))
                with self.subTest(level=level, extra=extra):
                    self.invalid(mutate)

    def test_image_encoding_and_complete_length(self):
        image = self.original["cases"][0]["images"]["device_source"]
        for replacement in (image[:-2], image + "00", image.upper(), "gg" + image[2:],
                            [0] * 256, bytes.fromhex(image), None):
            with self.subTest(replacement=type(replacement)):
                self.invalid(lambda v, r=replacement:
                             v["cases"][0]["images"].__setitem__("device_source", r))

    def test_failed_status_and_false_no_write_claim(self):
        for status in (1, 600, 701, 999, -1):
            self.invalid(lambda v, s=status: v["cases"][0].__setitem__("status", s))
        value = copy.deepcopy(self.original)
        destination = bytearray.fromhex(value["cases"][0]["images"]["device_destination"])
        for index in range(43, 75):
            destination[index] = (index * 17 + 31) % 256
        value["cases"][0]["images"]["device_destination"] = destination.hex()
        with self.assertRaises(oracle.ReceiptError):
            self.valid(value)

    def test_duplicate_json_keys_at_every_depth(self):
        alterations = [
            ('"schema":1', '"schema":1,"schema":1'),
            ('"ordinal":0', '"ordinal":0,"ordinal":0'),
            ('"images":{', '"images":{"device_source":"00",'),
        ]
        for old, new in alterations:
            with self.subTest(field=old), self.assertRaises(oracle.ReceiptError):
                oracle.parse_receipt(self.encoded.replace(old, new, 1), expected_cached=False)

    def test_nonfinite_truncation_extra_text_and_bounds(self):
        malformed = ["", "{}", "[]", "null", self.encoded[:-1],
                     self.encoded + self.encoded, self.encoded + "native diagnostic",
                     "x" * (oracle.MAX_RECEIPT_BYTES + 1), b"\xff", "\ud800",
                     "[" * 2000 + "]" * 2000]
        for constant in ("NaN", "Infinity", "-Infinity"):
            malformed.append(self.encoded.replace('"schema":1', f'"schema":{constant}', 1))
        for raw in malformed:
            with self.subTest(prefix=str(raw)[:20]), self.assertRaises(oracle.ReceiptError):
                oracle.parse_receipt(raw, expected_cached=False)
        with self.assertRaises(oracle.ReceiptError):
            oracle.parse_receipt({}, expected_cached=False)

    def test_failure_record_cannot_be_a_success_receipt(self):
        failure = {"record": "unified_copy_failure", "schema": 1, "error": "SDK status 1",
                   "cleanup_error": "", "completed_cases": 39, "normal_cleanup": True}
        with self.assertRaises(oracle.ReceiptError):
            oracle.parse_receipt(json.dumps(failure), expected_cached=False)


if __name__ == "__main__":
    unittest.main()
