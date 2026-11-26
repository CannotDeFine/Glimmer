"""Independent fixed eight-case per-thread callback/source-order oracle."""

import json
import re

import h2h_copy_receipt as shared
from unified_copy_execution_receipt import MAX_BYTES, validate_execution as validate_copies

MAX_RECEIPT_BYTES = 128 * 1024


def validate_execution(raw):
    return validate_copies(raw, expected_commands=8)


def parse_receipt(raw, *, expected_cached, expected_gpu_uuid=None):
    if type(raw) is bytes:
        raw = raw.decode("utf-8", errors="strict")
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise ValueError("oversized per-thread callback receipt")
    value = json.loads(raw, object_pairs_hook=shared._unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if type(value) is not dict:
        raise ValueError("missing per-thread callback receipt")
    uuid, version, cases = value.get("gpu_uuid"), value.get("driver_version"), value.get("cases")
    if (type(uuid) is not str or re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid) is None or
            (expected_gpu_uuid is not None and uuid != expected_gpu_uuid) or
            type(version) is not int or not 1000 <= version < 1000000 or
            type(cases) is not list or len(cases) != 8):
        raise ValueError("incorrect per-thread callback identity or inventory")
    expected_cases = []
    for ordinal, case in enumerate(cases):
        if type(case) is not dict:
            raise ValueError("invalid per-thread callback case")
        flags = {key: case.get(key) for key in ("callback_complete_at_return", "watchdog_before_return",
                                               "copied_before_callback")}
        if (any(type(flag) is not bool for flag in flags.values()) or
                (flags["callback_complete_at_return"] and not flags["watchdog_before_return"])):
            raise ValueError("invalid held callback causality")
        route = ordinal % 4
        source, destination = (2 if route & 1 else 0), (3 if route & 2 else 1)
        images = [bytearray([0x80 + image] * 256) for image in range(4)]
        images[source][17:145] = bytes([0x5d]) * 128
        images[destination][43:171] = bytes([0x11 if flags["copied_before_callback"] else 0x5d]) * 128
        expected_cases.append(dict(ordinal=ordinal, **{"async": bool(ordinal // 4)}, stream_kind=3,
                                   route=route, status=0, callback_calls=1, **flags,
                                   images=[image.hex() for image in images]))
    expected = dict(record="h2h_per_thread_order_result", schema=1, cached=expected_cached,
                    gpu_uuid=uuid, driver_version=version, case_count=8, normal_cleanup=True,
                    full_device_qualified=False, priority_qualified=False,
                    formal_comparison_qualified=False, cases=expected_cases)
    shared._same(value, expected)
    return value
