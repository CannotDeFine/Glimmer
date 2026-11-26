"""Fixed special-handle matrix; reuse the unchanged independent full-byte oracle."""

import copy
import json

import unified_copy_gpu_receipt as baseline

MAX_RECEIPT_BYTES = baseline.MAX_RECEIPT_BYTES


def parse_receipt(raw, *, expected_cached, expected_gpu_uuid=None):
    baseline._require(type(raw) in (str, bytes), "bounded per-thread receipt required")
    baseline._require(0 < len(raw) <= MAX_RECEIPT_BYTES, "per-thread receipt size")
    if type(raw) is bytes:
        raw = raw.decode("utf-8", errors="strict")
    baseline._require(len(raw.encode("utf-8")) <= MAX_RECEIPT_BYTES, "per-thread UTF-8 size")
    value = json.loads(raw, object_pairs_hook=baseline._unique_object,
                       parse_constant=baseline._nonfinite)
    baseline._require(type(value) is dict, "per-thread receipt object")
    baseline._exact(value.get("record"), "unified_copy_per_thread_result", "special-handle record")
    normalized = copy.deepcopy(value)
    normalized["record"] = "unified_copy_result"
    cases = normalized.get("cases")
    baseline._require(type(cases) is list and len(cases) == 40, "fixed special-handle inventory")
    for ordinal in (*range(15, 20), *range(35, 40)):
        case = cases[ordinal]
        baseline._require(type(case) is dict, "special-handle case object")
        baseline._exact(case.get("mode"), "async_per_thread", "exact per-thread case selection")
        case["mode"] = "async_nonblocking"
    baseline.validate_receipt(normalized, expected_cached=expected_cached,
                              expected_gpu_uuid=expected_gpu_uuid)
    return value
