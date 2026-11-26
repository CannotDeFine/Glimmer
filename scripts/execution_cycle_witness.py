"""Pure full-image SM-cycle validation; elapsed overlap is never a witness.

GCW1 is a fixed Linux little-endian diagnostic, not execution authority.
The k-way merge retains at most one sample per warp, not millions of dict rows.
All raw records, mathematical values and guards are checked before joining.
"""

from contextlib import ExitStack
import hashlib
import heapq
import mmap
import os
from pathlib import Path
import stat
import struct

HEADER = struct.Struct("<16I")
SAMPLE = struct.Struct("<3Q6I")
BLOCKS, THREADS, WARP, SAMPLES = 320, 128, 32, 1024
WARPS = BLOCKS * THREADS // WARP
RECORDS, VALUES = WARPS * SAMPLES, BLOCKS * THREADS
GUARDS, MARKER = 192, 0xA5A5A5A5
TRACE_BYTES, VALUE_BYTES = RECORDS * SAMPLE.size, VALUES * 4
FILE_BYTES = HEADER.size + TRACE_BYTES + VALUE_BYTES + 4 * GUARDS
MASK, A, B, SEED, GROUP_SEED = (1 << 32) - 1, 1664525, 1013904223, 0x12345678, 0x10203
ROUNDS = (4096, 2048, 64)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def affine(count):
    """Independent repeated-squaring oracle, modulo2**32."""
    if not count:
        return 1, 0
    multiplier, increment = affine(count // 2)
    squared = multiplier * multiplier & MASK
    shifted = increment * (multiplier + 1) & MASK
    return ((A * squared & MASK, (A * shifted + B) & MASK)
            if count % 2 else (squared, shifted))


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self.descriptor = None
        self.mapping = None
        self.summary = None
        self.sms = []

    def __enter__(self):
        try:
            self.descriptor = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            self.initial = os.fstat(self.descriptor)
            require(stat.S_ISREG(self.initial.st_mode) and self.initial.st_size == FILE_BYTES,
                    "one complete bounded regular GCW1 image required")
            self.mapping = mmap.mmap(self.descriptor, 0, access=mmap.ACCESS_READ)
            header = HEADER.unpack_from(self.mapping)
            kind = header[8]
            require(kind in (0, 1, 2), "unknown original sample group kind")
            expected = (0x31574347, 1, 64, 48, BLOCKS, THREADS, SAMPLES, ROUNDS[kind], kind,
                        GUARDS, RECORDS, VALUES, TRACE_BYTES, VALUE_BYTES, MARKER, 0)
            require(header == expected, "noncanonical/malformed fixed GCW1 header")
            self.kind = kind
            # Metadata timestamps can remain unchanged across a same-size write.
            self.initial_digest = hashlib.sha256(self.mapping).digest()
            require(self._metadata_stable() and HEADER.unpack_from(self.mapping) == header,
                    "raw sample file changed during initial read")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None

    def sample(self, identity, index):
        return SAMPLE.unpack_from(self.mapping, HEADER.size + GUARDS +
                                  (identity * SAMPLES + index) * SAMPLE.size)

    def _metadata_stable(self):
        # mmap/read may update atime; it is not a content-identity field.
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode")
        def identity(value):
            return tuple(getattr(value, name) for name in fields)
        return identity(os.fstat(self.descriptor)) == identity(self.path.lstat()) == identity(self.initial)

    def stable(self):
        # Two snapshots detect retained changes, not writes restored between them.
        return (self._metadata_stable() and
                hashlib.sha256(self.mapping).digest() == self.initial_digest and
                self._metadata_stable())

    def validate(self):
        require(self.summary is None, "dataset validation cannot replay a saved result")
        payload = self.mapping
        guard = b"\xA5" * GUARDS
        values_at = HEADER.size + 2 * GUARDS + TRACE_BYTES
        for offset in (HEADER.size, HEADER.size + GUARDS + TRACE_BYTES,
                       values_at, values_at + GUARDS + VALUE_BYTES):
            require(payload[offset:offset + GUARDS] == guard, "changed original allocation guard")
        jump, increment = affine(ROUNDS[self.kind])
        states = hashlib.sha256()
        groups, sms = [0, 0], set()
        for identity in range(WARPS):
            expected_group = identity % 2 if self.kind == 2 else self.kind
            state = (SEED + expected_group * GROUP_SEED + (identity * WARP + 1) * 17) & MASK
            previous_cycle = previous_time = 0
            sm = None
            for index in range(SAMPLES):
                before, cycle, after, first, last, actual, value, group, reserved = self.sample(identity, index)
                state = (jump * state + increment) & MASK
                if sm is None:
                    sm = first
                require(actual == identity and group == expected_group and reserved == 0,
                        "duplicate/aliased/foreign sample identity or group")
                require(value == state, "full non-idempotent sample state mismatch")
                require(first == last == sm and first < 1 << 20, "migrated or malformed SM identity")
                require(0 < before <= after < 1 << 63 and cycle > previous_cycle and
                        before >= previous_time, "nonmonotonic/reset/wrapped device observations")
                require(after - before <= 1_000_000, "uncertain oversized clock bracket")
                previous_cycle, previous_time = cycle, after
                states.update(struct.pack("<I", value))
                groups[group] += 1
            self.sms.append(sm)
            sms.add(sm)
        require(len(sms) <= 80, "more observed SMs than the fixed V100 contract")
        final_jump, final_increment = affine(ROUNDS[self.kind] * SAMPLES)
        final_at = values_at + GUARDS
        for index in range(VALUES):
            group = (index // WARP) % 2 if self.kind == 2 else self.kind
            seed = (SEED + group * GROUP_SEED + (index + 1) * 17) & MASK
            actual = struct.unpack_from("<I", payload, final_at + index * 4)[0]
            require(actual == (final_jump * seed + final_increment) & MASK,
                    "full final thread value mismatch")
        require(self.stable(), "raw sample file changed during validation")
        self.summary = dict(kind=self.kind, records=RECORDS, final_values=VALUES,
                            checked_guard_bytes=4 * GUARDS, sample_groups=groups,
                            sm_ids=sorted(sms), state_sha256=states.hexdigest(),
                            final_values_sha256=hashlib.sha256(payload[final_at:final_at + VALUE_BYTES]).hexdigest())
        return self.summary


def validate_dataset(path):
    with Dataset(path) as dataset:
        return dataset.validate()


def coissue(first, second=None, *, calibration=False):
    """Count every exact same-SM/cycle key; retain only fixed first16 examples.

    Any same cycle reused at disjoint device-time brackets invalidates the
    counter model, rather than being silently dropped to find favorable rows.
    Caller must calibrate positive and serialized negative controls separately.
    """
    require(type(calibration) is bool and (second is None) is calibration,
            "explicit one-file calibration or two independent role images required")
    with ExitStack() as resources:
        datasets = [resources.enter_context(Dataset(first))]
        if second is not None:
            datasets.append(resources.enter_context(Dataset(second)))
            require((datasets[0].initial.st_dev, datasets[0].initial.st_ino) !=
                    (datasets[1].initial.st_dev, datasets[1].initial.st_ino),
                    "aliased source images cannot provide cross-kernel evidence")
        summaries = [dataset.validate() for dataset in datasets]
        require(([dataset.kind for dataset in datasets] == [2] if calibration else
                 [dataset.kind for dataset in datasets] == [0, 1]), "exact original group roles required")
        require(len(set().union(*(summary["sm_ids"] for summary in summaries))) <= 80,
                "cross-input SM identities exceed the one-device contract")
        heap = []
        for source, dataset in enumerate(datasets):
            for identity, sm in enumerate(dataset.sms):
                sample = dataset.sample(identity, 0)
                heap.append((sm, sample[1], source, identity, 0))
        heapq.heapify(heap)
        keys = pairs = 0
        witness_sms, examples = set(), []
        digest = hashlib.sha256()
        while heap:
            key = heap[0][:2]
            rows = []
            while heap and heap[0][:2] == key:
                sm, cycle, source, identity, index = heapq.heappop(heap)
                sample = datasets[source].sample(identity, index)
                rows.append((source, identity, index, sample))
                require(len(rows) <= 64, "impossible simultaneous warp/sample multiplicity")
                if index + 1 < SAMPLES:
                    following = datasets[source].sample(identity, index + 1)
                    heapq.heappush(heap, (sm, following[1], source, identity, index + 1))
            matched = False
            for left_index, left in enumerate(rows):
                for right in rows[left_index + 1:]:
                    a, b = left[3], right[3]
                    require(max(a[0], b[0]) <= min(a[2], b[2]),
                            "same SM cycle reused at disjoint device time: ambiguous counter model")
                    if a[7] == b[7] or (calibration and left[1] // 4 != right[1] // 4):
                        continue
                    require(calibration or left[0] != right[0], "one source cannot impersonate two kernels")
                    matched = True
                    pairs += 1
                    witness_sms.add(key[0])
                    # Fixed order from the complete k-way merge, not chosen evidence.
                    fields = (key[0], key[1], *left[:3], *right[:3], a[0], a[2], b[0], b[2])
                    digest.update(struct.pack("<12Q", *fields))
                    if len(examples) < 16:
                        examples.append(list(fields))
            keys += int(matched)
        for dataset in datasets:
            require(dataset.stable(),
                    "raw source image changed during complete cycle join")
        return dict(calibration=calibration, source_summaries=summaries,
                    matching_cycle_keys=keys, matching_sample_pairs=pairs,
                    coissue_sm_ids=sorted(witness_sms), all_pair_identities_sha256=digest.hexdigest(),
                    first16_pair_identities=examples, inputs_fully_validated=True,
                    interval_overlap_used_as_witness=False,
                    hardware_co_residency_qualified=False, full_device_qualified=False,
                    physical_priority_qualified=False, saturation_qualified=False,
                    performance_qualified=False, full_goal_complete=False)
