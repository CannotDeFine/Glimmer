import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest

import lt_scalar_bias_receipt as receipt


def case(index):
    format_, device, bias, alias, zero = receipt.planned_cases()[index]
    return dict(id=index, format=format_, device_scalars=device, bias=bias, in_place=alias,
                zero_beta=zero, algorithm=3, pointer_mask=3, epilogue_mask=5,
                mismatches=0, guard_mismatches=0, inputs_unchanged=True,
                output_bytes=receipt.ELEMENTS["c" if alias else "d"] * (2 if format_ == 1 else 4))


def write_case(directory, value):
    initial = receipt.initial_images(value["format"], value["zero_beta"])
    output = "c" if value["in_place"] else "d"
    expected = receipt.expected_image(value)
    for name, before in initial.items():
        (directory / f'{value["id"]}-{name}-before.bin').write_bytes(before)
        (directory / f'{value["id"]}-{name}-after.bin').write_bytes(expected if name == output else before)
    (directory / f'{value["id"]}-expected.bin').write_bytes(expected)


class ReceiptTests(unittest.TestCase):
    def test_complete_numerical_failure_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            images = directory / "images"
            images.mkdir()
            values = [case(index) for index in range(48)]
            for value in values:
                write_case(images, value)
            first = values[0]
            first["mismatches"] = 4845
            result = bytearray(receipt.initial_images(1, False)["d"])
            for column in range(96):
                for row in range(64):
                    base = 4 * ((row + column * 72) % 9 - 4) + 8 * (row % 9 - 4)
                    struct.pack_into("<e", result, 2 * (row + column * 80), base / 64)
            (images / "0-d-after.bin").write_bytes(result)
            value = dict(status="numerical_failure", cleanup=True, diagnostic_only=True,
                         lt_version=120604, blas_version=[12, 6, 4], cases=values,
                         **{key: False for key in receipt.FLAGS})
            log = directory / "application.log"
            log.write_text(json.dumps(value) + "\n" + receipt.FAILURE + "\n")
            audited = receipt.validate(log, images)
            self.assertFalse(audited["numerical_passed"])
            self.assertFalse(audited["full_goal_complete"])
            self.assertTrue(audited["cleanup"])
            log.write_text(json.dumps(value) + "\n")
            with self.assertRaises(ValueError):
                receipt.validate(log, images)

    def test_original_reference_and_all_exact_half_values(self):
        value = case(0)
        self.assertEqual(struct.unpack_from("<e", receipt.expected_image(value))[0], .5)
        self.assertEqual(len(set(receipt.planned_cases())), 48)
        for bias in (False, True):
            for zero in (False, True):
                value.update(bias=bias, zero_beta=zero, format=0)
                full = receipt.expected_image(value)
                for index in range(7712):
                    scalar = struct.unpack_from("<f", full, index * 4)[0]
                    self.assertEqual(struct.unpack("<e", struct.pack("<e", scalar))[0], scalar)

    def test_complete_raw_matrix_and_hostile_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            images = directory / "images"
            images.mkdir()
            values = [case(index) for index in range(48)]
            for value in values:
                write_case(images, value)
            original = dict(status="passed", cleanup=True, diagnostic_only=True,
                            lt_version=120604, blas_version=[12, 6, 4], cases=values,
                            **{name: False for name in receipt.FLAGS})
            log = directory / "application.log"
            log.write_text(json.dumps(original) + "\n")
            audited = receipt.validate(log, images)
            self.assertTrue(audited["numerical_passed"])
            self.assertTrue(audited["saved_images_independently_reloaded"])
            changed = []
            for key, replacement in (("cleanup", False), ("diagnostic_only", False),
                                     ("full_goal_complete", True), ("cases", values[:-1])):
                changed.append(dict(original, **{key: replacement}))
            for key, replacement in (("mismatches", 1), ("device_scalars", 1),
                                     ("format", True), ("id", 1), ("output_bytes", 4)):
                value = copy.deepcopy(original)
                value["cases"][0][key] = replacement
                changed.append(value)
            for value in changed:
                log.write_text(json.dumps(value) + "\n")
                with self.assertRaises(ValueError):
                    receipt.validate(log, images)
            for content in (json.dumps(original) + "\nextra\n", '{"status":NaN}',
                            json.dumps(original)[:-1] + ',"cleanup":true}'):
                log.write_text(content)
                with self.assertRaises(ValueError):
                    receipt.validate(log, images)
            log.write_text(json.dumps(original) + "\n")
            extra = images / "unexpected.bin"
            extra.write_bytes(b"")
            with self.assertRaises(ValueError):
                receipt.validate(log, images)

    def test_real_failure_hypothesis_is_not_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            images = Path(temporary)
            value = case(0)
            write_case(images, value)
            output = bytearray(receipt.initial_images(1, False)["d"])
            for column in range(96):
                for row in range(64):
                    base = 4 * ((row + column * 72) % 9 - 4) + 8 * (row % 9 - 4)
                    struct.pack_into("<e", output, 2 * (row + column * 80), base / 64)
            (images / "0-d-after.bin").write_bytes(output)
            value["mismatches"] = 4845
            audited = receipt.audit_case(value, images)
            self.assertEqual(audited["mismatches"], 4845)
            self.assertEqual(audited["missing_product_matches"], 4845)
            value["mismatches"] = 0
            with self.assertRaises(ValueError):
                receipt.audit_case(value, images)

    def test_input_guard_expected_truncation_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            images = Path(temporary)
            value = case(0)
            for name in ("0-a-after.bin", "0-d-after.bin", "0-expected.bin"):
                write_case(images, value)
                path = images / name
                original = path.read_bytes()
                mutated = bytearray(original)
                mutated[-1] ^= 1
                path.write_bytes(mutated)
                with self.assertRaises(ValueError):
                    receipt.audit_case(value, images)
            write_case(images, value)
            path = images / "0-d-after.bin"
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                receipt.audit_case(value, images)
            path.unlink()
            path.symlink_to(images / "0-expected.bin")
            with self.assertRaises(ValueError):
                receipt.audit_case(value, images)


if __name__ == "__main__":
    unittest.main()
