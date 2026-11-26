"""Independent full-byte oracle for the fixed ordinary SDK UVA copy input.

This module imports no GPU helpers, fixture source, or generated case inventory.
Validated receipts establish byte conformance and claimed normal cleanup only;
the collector must separately authenticate processes, exit, and input identities.
"""

import json
import re

MAX_RECEIPT_BYTES = 256 * 1024

_IMAGE_NAMES = (
    "device_source", "device_destination", "pageable_source",
    "pageable_destination", "pinned_source", "pinned_destination",
)
_ROUTES = (
    ("pageable_h2d", 2, 1),
    ("pinned_h2d", 4, 1),
    ("pageable_d2h", 0, 3),
    ("pinned_d2h", 0, 5),
    ("d2d", 0, 1),
)
_MODES = ("sync_legacy", "async_legacy", "async_blocking", "async_nonblocking")
_SUMMARY_CONSTANTS = {
    "record": "unified_copy_result",
    "schema": 1,
    "case_count": 40,
    "storage_bytes": 256,
    "guard_bytes": 16,
    "image_count": 6,
    "checked_bytes": 61440,
    "original_copy_calls": 200,
    "sync_copy_calls": 170,
    "async_copy_calls": 30,
    "normal_cleanup": True,
    "full_device_qualified": False,
    "priority_qualified": False,
    "formal_comparison_qualified": False,
}
_SUMMARY_KEYS = set(_SUMMARY_CONSTANTS) | {"cached", "gpu_uuid", "driver_version", "cases"}
_UUID = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HEX_IMAGE = re.compile(r"[0-9a-f]{512}")


class ReceiptError(ValueError):
    """The input is not the complete fixed conformance receipt."""


def _require(condition, message):
    if not condition:
        raise ReceiptError(message)


def _exact(actual, expected, location):
    _require(type(actual) is type(expected) and actual == expected,
             f"{location}: incorrect value or exact type")


def _object(value, keys, location):
    _require(type(value) is dict and set(value) == keys,
             f"{location}: missing, extra, or invalid object fields")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ReceiptError(f"nonfinite JSON number: {value}")


def _case_expectations():
    ordinal = 0
    for byte_count in (32, 65):
        for mode_index, mode in enumerate(_MODES):
            for route, source, destination in _ROUTES:
                descriptor = {
                    "ordinal": ordinal,
                    "route": route,
                    "mode": mode,
                    "api": "cuMemcpyAsync" if mode_index else "cuMemcpy",
                    "bytes": byte_count,
                    "source_image": _IMAGE_NAMES[source],
                    "destination_image": _IMAGE_NAMES[destination],
                    "source_offset": 17,
                    "destination_offset": 43,
                    "status": 0,
                    "stream_complete": True,
                    "full_bytes_verified": True,
                    "checked_bytes": 1536,
                }
                images = [bytearray(
                    0x80 + image if index < 16 or index >= 240
                    else (index * 17 + image * 31 + ordinal * 7) % 256
                    for index in range(256)
                ) for image in range(6)]
                images[destination][43:43 + byte_count] = images[source][17:17 + byte_count]
                yield descriptor, images
                ordinal += 1


def validate_receipt(receipt, *, expected_cached, expected_gpu_uuid=None,
                     expected_driver_version=None):
    """Validate exact descriptors and every byte; return the original dict.

    Supply externally authenticated UUID/version when available. Omitting them
    checks syntax/range, not the identity of the process that produced JSON.
    Compare native/owned returned dicts exactly for each matched cached setting.
    """
    _require(type(expected_cached) is bool, "expected_cached must be bool")
    if expected_gpu_uuid is not None:
        _require(type(expected_gpu_uuid) is str and _UUID.fullmatch(expected_gpu_uuid),
                 "invalid expected GPU UUID")
    if expected_driver_version is not None:
        _require(type(expected_driver_version) is int and
                 4000 <= expected_driver_version <= 999999,
                 "invalid expected Driver version")
    _object(receipt, _SUMMARY_KEYS, "summary")
    for key, value in _SUMMARY_CONSTANTS.items():
        _exact(receipt[key], value, f"summary.{key}")
    _exact(receipt["cached"], expected_cached, "summary.cached")
    _require(type(receipt["gpu_uuid"]) is str and _UUID.fullmatch(receipt["gpu_uuid"]),
             "summary.gpu_uuid: invalid canonical UUID")
    _require(type(receipt["driver_version"]) is int and
             4000 <= receipt["driver_version"] <= 999999,
             "summary.driver_version: invalid exact integer")
    if expected_gpu_uuid is not None:
        _exact(receipt["gpu_uuid"], expected_gpu_uuid, "authenticated GPU UUID")
    if expected_driver_version is not None:
        _exact(receipt["driver_version"], expected_driver_version, "authenticated Driver version")
    _require(type(receipt["cases"]) is list and len(receipt["cases"]) == 40,
             "exact ordered forty-case inventory required")
    for ordinal, (case, (descriptor, images)) in enumerate(
            zip(receipt["cases"], _case_expectations(), strict=True)):
        location = f"case[{ordinal}]"
        _object(case, set(descriptor) | {"images"}, location)
        for key, value in descriptor.items():
            _exact(case[key], value, f"{location}.{key}")
        _object(case["images"], set(_IMAGE_NAMES), f"{location}.images")
        for name, expected_image in zip(_IMAGE_NAMES, images, strict=True):
            image = case["images"][name]
            _require(type(image) is str and _HEX_IMAGE.fullmatch(image),
                     f"{location}.{name}: exact 256-byte lower-case hex image required")
            _require(image == expected_image.hex(),
                     f"{location}.{name}: full source/destination/guard image mismatch")
    return receipt


def parse_receipt(raw, *, expected_cached, expected_gpu_uuid=None,
                  expected_driver_version=None):
    """Parse one bounded strict JSON value, rejecting duplicate keys at any depth."""
    _require(type(raw) in (str, bytes), "receipt input must be str or bytes")
    _require(0 < len(raw) <= MAX_RECEIPT_BYTES, "receipt input exceeds size bound or is empty")
    try:
        if type(raw) is bytes:
            raw = raw.decode("utf-8", errors="strict")
        _require(len(raw.encode("utf-8", errors="strict")) <= MAX_RECEIPT_BYTES,
                 "receipt UTF-8 input exceeds size bound")
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    except ReceiptError:
        raise
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ReceiptError(f"invalid strict JSON: {error}") from error
    return validate_receipt(value, expected_cached=expected_cached,
                            expected_gpu_uuid=expected_gpu_uuid,
                            expected_driver_version=expected_driver_version)
