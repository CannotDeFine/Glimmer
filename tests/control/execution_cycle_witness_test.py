"""Independent complete synthetic GCW1 frames; no CUDA or GPU calls."""

from contextlib import contextmanager
import mmap
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import execution_cycle_witness as witness


def frame(path, kind, cycle_shift=0):
    # Literal schema and linear recurrence composition, not parser-produced rows.
    header = (0x31574347, 1, 64, 48, 320, 128, 1024, (4096, 2048, 64)[kind], kind,
              192, 1310720, 40960, 62914560, 163840, 0xA5A5A5A5, 0)
    multiplier, increment = 1, 0
    for _ in range(header[7]):
        multiplier = multiplier * 1664525 & 0xFFFFFFFF
        increment = (increment * 1664525 + 1013904223) & 0xFFFFFFFF
    with path.open("xb") as stream:
        stream.write(struct.pack("<16I", *header))
        stream.write(b"\xA5" * 192)
        block = bytearray(48 * 1024)
        for identity in range(1280):
            group = identity % 2 if kind == 2 else kind
            state = (0x12345678 + group * 0x10203 + (identity * 32 + 1) * 17) & 0xFFFFFFFF
            for index in range(1024):
                state = (multiplier * state + increment) & 0xFFFFFFFF
                cycle = 10000 + identity // 4 * 8192 + index * 8
                cycle += (identity % 4 // 2 * 2 if kind == 2 else identity % 4 * 2)
                before = cycle * 10
                struct.pack_into("<3Q6I", block, index * 48, before, cycle + cycle_shift, before + 4,
                                 identity // 4 % 80, identity // 4 % 80, identity, state, group, 0)
            stream.write(block)
        stream.write(b"\xA5" * 384)
        final_multiplier, final_increment = 1, 0
        for _ in range(1024):
            final_multiplier = multiplier * final_multiplier & 0xFFFFFFFF
            final_increment = (multiplier * final_increment + increment) & 0xFFFFFFFF
        values = bytearray(40960 * 4)
        for index in range(40960):
            group = index // 32 % 2 if kind == 2 else kind
            seed = 0x12345678 + group * 0x10203 + (index + 1) * 17
            struct.pack_into("<I", values, index * 4,
                             (final_multiplier * seed + final_increment) & 0xFFFFFFFF)
        stream.write(values)
        stream.write(b"\xA5" * 192)


@contextmanager
def patched(path, offset, data):
    with path.open("r+b") as stream:
        stream.seek(offset)
        original = stream.read(len(data))
        try:
            stream.seek(offset)
            stream.write(data)
            stream.flush()
            yield
        finally:
            stream.seek(offset)
            stream.write(original)


class CycleWitnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="glimmer-cycle-unit-")
        cls.root = Path(cls.temporary.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def fixture(self, name, kind, shift=0):
        path = self.root / name
        if not path.exists():
            frame(path, kind, shift)
        return path

    @property
    def calibration(self):
        return self.fixture("calibration.gcw", 2)

    @property
    def low(self):
        return self.fixture("low.gcw", 0)

    @property
    def high(self):
        return self.fixture("high.gcw", 1)

    @property
    def shifted(self):
        return self.fixture("shifted.gcw", 1, 1)

    def test_full_calibration_positive_without_promoted_flags(self):
        result = witness.coissue(self.calibration, calibration=True)
        self.assertEqual(result["matching_cycle_keys"], 655360)
        self.assertEqual(result["matching_sample_pairs"], 655360)
        self.assertEqual(len(result["coissue_sm_ids"]), 80)
        self.assertEqual(len(result["first16_pair_identities"]), 16)
        self.assertEqual(result["source_summaries"][0]["records"], 1310720)
        self.assertEqual(result["source_summaries"][0]["checked_guard_bytes"], 768)
        self.assertFalse(result["hardware_co_residency_qualified"])

    def test_cross_source_exact_cycle_and_interval_only_negative(self):
        result = witness.coissue(self.low, self.high)
        self.assertEqual(result["matching_cycle_keys"], 1310720)
        self.assertEqual(result["matching_sample_pairs"], 1310720)
        negative = witness.coissue(self.low, self.shifted)
        self.assertEqual(negative["matching_cycle_keys"], 0)
        self.assertFalse(negative["interval_overlap_used_as_witness"])

    def test_header_and_file_bound_rejections(self):
        for offset, value in ((0, 0), (4, 2), (12, 40), (16, 80), (24, 0),
                              (28, 1), (32, 3), (36, 0), (40, 0), (60, 1)):
            with self.subTest(offset=offset), patched(self.low, offset, struct.pack("<I", value)):
                with self.assertRaises(ValueError):
                    witness.validate_dataset(self.low)
        short = self.root / "short.gcw"
        with self.low.open("rb") as stream:
            short.write_bytes(stream.read(1024))
        with self.assertRaises(ValueError):
            witness.validate_dataset(short)
        link = self.root / "link.gcw"
        link.symlink_to(self.low)
        with self.assertRaises(OSError):
            witness.validate_dataset(link)

    def test_complete_guards_values_and_record_identity(self):
        body = 64 + 192
        corruptions = [(64, b"\0"), (64 + 192 + 62914560, b"\0"),
                       (witness.FILE_BYTES - 1, b"\0"),
                       (body + 24, struct.pack("<I", 81)),
                       (body + 28, struct.pack("<I", 81)),
                       (body + 32, struct.pack("<I", 1)),
                       (body + 36, struct.pack("<I", 0)),
                       (body + 40, struct.pack("<I", 1)),
                       (body + 44, struct.pack("<I", 1)),
                       (body + 8, struct.pack("<Q", 0)),
                       (body + 16, struct.pack("<Q", 1)),
                       (body + 48 + 24, struct.pack("<II", 1, 1)),
                       (64 + 2 * 192 + 62914560 + 192, struct.pack("<I", 0))]
        for offset, value in corruptions:
            with self.subTest(offset=offset), patched(self.low, offset, value):
                with self.assertRaises(ValueError):
                    witness.validate_dataset(self.low)

    def test_alias_and_disjoint_time_cannot_fake_coissue(self):
        with self.assertRaises(ValueError):
            witness.coissue(self.low, self.low)
        with self.assertRaises(ValueError):
            witness.coissue(self.low, calibration=True)
        # A reused cycle with disjoint time cannot be silently skipped while
        # other positive rows are accepted. Change one complete valid bracket.
        body = 64 + 192
        with patched(self.high, body, struct.pack("<Q", 99990)), \
                patched(self.high, body + 16, struct.pack("<Q", 99994)):
            with self.assertRaises(ValueError):
                witness.coissue(self.low, self.high)

    def test_changed_mapped_source_is_not_stable(self):
        with witness.Dataset(self.low) as source:
            with patched(self.low, 60, struct.pack("<I", 1)):
                # Simulate a write hidden by coarse or restored file timestamps.
                source.initial = self.low.lstat()
                self.assertFalse(source.stable())


if __name__ == "__main__":
    unittest.main()
