"""Independent upload byte/causality and exact108-operation issue oracle."""

import json
import re

import h2h_copy_receipt as shared
from unified_copy_execution_receipt import MAX_BYTES, validate_operations

MAX_RECEIPT_BYTES = 256 * 1024


def operations():
    result = []
    for ordinal in range(36):
        api, pinned = ordinal // 12, ordinal % 2 != 0
        result.extend((141, 141 if api == 0 else 23 if pinned else 25, 141))
    return tuple(result)


def validate_execution(raw):
    return validate_operations(raw, operations())


def parse_receipt(raw, *, expected_cached, expected_gpu_uuid=None):
    if type(raw) is bytes:
        raw = raw.decode("utf-8", errors="strict")
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise ValueError("oversized upload source-order receipt")
    value = json.loads(raw, object_pairs_hook=shared._unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if type(value) is not dict:
        raise ValueError("missing upload source-order object")
    uuid, version, cases = value.get("gpu_uuid"), value.get("driver_version"), value.get("cases")
    if (type(uuid) is not str or re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid) is None or
            (expected_gpu_uuid is not None and uuid != expected_gpu_uuid) or
            type(version) is not int or not 1000 <= version < 1000000 or
            type(cases) is not list or len(cases) != 36):
        raise ValueError("incorrect upload identity or inventory")
    expected_cases = []
    for ordinal, case in enumerate(cases):
        if type(case) is not dict:
            raise ValueError("invalid upload case")
        flags = {key: case.get(key) for key in ("callback_complete_at_return", "watchdog_before_return",
                                               "copied_before_callback")}
        if (any(type(flag) is not bool for flag in flags.values()) or
                (flags["callback_complete_at_return"] and not flags["watchdog_before_return"])):
            raise ValueError("invalid held producer causality")
        pinned = ordinal % 2 != 0
        source = 2 if pinned else 0
        images = [bytearray([0x80 + index] * 256) for index in range(4)]
        images[source][17:145] = bytes([0x5d]) * 128
        images[1][43:171] = bytes([0x11 if flags["copied_before_callback"] else 0x5d]) * 128
        expected_cases.append(dict(ordinal=ordinal, **{"async": bool(ordinal % 12 // 6)},
                                   stream_kind=ordinal % 6 // 2, api=ordinal // 12,
                                   pinned_source=pinned, status=0, callback_calls=1, **flags,
                                   images=[image.hex() for image in images]))
    expected = dict(record="h2d_order_result", schema=1, cached=expected_cached, gpu_uuid=uuid,
                    driver_version=version, case_count=36, normal_cleanup=True,
                    full_device_qualified=False, priority_qualified=False,
                    formal_comparison_qualified=False, cases=expected_cases)
    shared._same(value, expected)
    return value
