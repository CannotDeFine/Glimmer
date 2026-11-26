"""Independent full-byte and actual-issue oracle for the fixed H2H SDK input."""

import json
import re
import struct

from unified_copy_execution_receipt import (
    ADMISSION, AUDIT_CANDIDATE, ISSUE, MAX_BYTES, PREFIX, UINT,
)

MAX_RECEIPT_BYTES = 256 * 1024
PERSISTENT = re.compile(PREFIX + r"Native persistent audit: session=" + UINT + r", sequence=" +
                        UINT + r", total_blocks=(1), physical_launches=(1), workers=(320)")


def _same(actual, expected):
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError("incomplete, mistyped or incorrect H2H receipt")


def expected_cases():
    result = []
    for size in (32, 65):
        for mode in range(4):
            for route in range(4):
                ordinal = len(result)
                images = [bytearray(0x80 + image if index < 16 or index >= 240 else
                                    (index * 17 + image * 31 + ordinal * 7) & 255
                                    for index in range(256)) for image in range(4)]
                source, destination = (2 if route & 1 else 0), (3 if route & 2 else 1)
                images[destination][43:43 + size] = images[source][17:17 + size]
                result.append(dict(ordinal=ordinal, bytes=size, mode=mode, route=route,
                                   status=0, images=[image.hex() for image in images]))
    return result


def expected_progress():
    source, control = [0xc35a91e7] * 64, [0xc35a91e7] * 64
    source[16] = 1
    for index in (0, 16, 32):
        control[index] = 1
    return [dict(ordinal=index, **{"async": bool(index // 2)}, pinned_source=bool(index % 2),
                 status=0, initial_event_status=600, physical_event_status=0,
                 watchdog_released=False, control_image=struct.pack("<64I", *control).hex(),
                 source_image=struct.pack("<64I", *source).hex()) for index in range(4)]


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate H2H field")
        result[key] = value
    return result


def parse_receipt(raw, *, expected_cached, expected_gpu_uuid=None):
    if type(raw) is bytes:
        raw = raw.decode("utf-8", errors="strict")
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise ValueError("oversized H2H receipt")
    value = json.loads(raw, object_pairs_hook=_unique,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict):
        raise ValueError("H2H receipt is not an object")
    uuid, version = value.get("gpu_uuid"), value.get("driver_version")
    if (type(uuid) is not str or re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid) is None
            or (expected_gpu_uuid is not None and uuid != expected_gpu_uuid)
            or type(version) is not int or not 1000 <= version < 1000000):
        raise ValueError("invalid or unmatched actual GPU/Driver identity")
    expected = dict(record="h2h_copy_result", schema=1, cached=expected_cached,
                    gpu_uuid=uuid, driver_version=version, case_count=32, progress_count=4,
                    original_copy_calls=36, normal_cleanup=True, full_device_qualified=False,
                    priority_qualified=False, formal_comparison_qualified=False,
                    progress=expected_progress(), cases=expected_cases())
    _same(value, expected)
    return value


def validate_execution(raw):
    """Prove36 copies,4 gates and4 separate event generations; not retirement."""
    if type(raw) is bytes:
        raw = raw.decode("utf-8", errors="strict")
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_BYTES:
        raise ValueError("oversized H2H execution ledger")
    admissions, issues, gates, events = {}, {}, {}, {}
    kernel_ids, event_ids = {1, 4, 7, 10}, {2, 5, 8, 11}
    native_ids = [sequence for sequence in range(1, 45) if sequence not in event_ids]
    semantic_service, last_serial = None, 0
    for position, line in enumerate(raw.splitlines()):
        if len(line.encode("utf-8")) > 32768 or re.search(r"\[(?:error|critical)\]", line, re.I):
            raise ValueError("invalid H2H service diagnostics")
        if any(status != "0" for status in re.findall(r"\b[A-Za-z_]*status\s*=\s*([^,;\s]+)", line)):
            raise ValueError("nonzero or malformed SDK status")
        if "Execution semantic audit:" in line:
            match = re.fullmatch(PREFIX + r"Execution semantic audit: (\{.*\})", line)
            if not match:
                raise ValueError("malformed event semantic record")
            body = json.loads(match[1], object_pairs_hook=_unique)
            service, serial = body.get("service"), body.get("serial")
            if (type(service) is not int or not 0 < service < 1 << 64 or
                    type(serial) is not int or not last_serial < serial < 1 << 64 or
                    (semantic_service is not None and service != semantic_service) or
                    body.get("v") != 1):
                raise ValueError("invalid semantic producer identity")
            semantic_service, last_serial = service, serial
            if body.get("family") == 3:
                sequence, event = body.get("sequence"), body.get("event")
                expected_fields = {"v", "service", "serial", "family", "event", "connection",
                                   "priority", "sequence", "opcode", "capture", "event_source",
                                   "event_object", "record_sequence", "event_capture", "status"}
                if (set(body) != expected_fields or any(type(value) is not int for value in body.values()) or
                        sequence not in event_ids or event not in (1, 2, 3) or
                        body["opcode"] != 48 or body["record_sequence"] != sequence or
                        body["event_source"] != body["connection"] or body["priority"] != 0 or
                        body["status"] != 0 or not 0 < body["event_object"] < 1 << 64 or
                        not 0 < body["capture"] < 1 << 64 or body["event_capture"] != body["capture"]):
                    raise ValueError("incorrect exact event generation")
                key = (body["connection"], sequence)
                prior = events.setdefault(key, [])
                if event != len(prior) + 1:
                    raise ValueError("missing, duplicated or reordered event lifecycle")
                identity = (body["event_object"], body["capture"])
                if prior and identity != prior[0][1]:
                    raise ValueError("event generation identity changed")
                prior.append((position, identity))
            continue
        if not AUDIT_CANDIDATE.search(line):
            continue
        admission, issue, persistent = ADMISSION.fullmatch(line), ISSUE.fullmatch(line), PERSISTENT.fullmatch(line)
        if admission:
            session, sequence, operation = map(int, admission.groups())
            key = (session, sequence)
            if (not 0 < session < 1 << 64 or len(admissions) >= len(native_ids) or
                    sequence != native_ids[len(admissions)] or
                    operation != (11 if sequence in kernel_ids else 141) or key in admissions):
                raise ValueError("incorrect typed H2H admission inventory")
            admissions[key] = position
        elif persistent:
            session, sequence, _, _, _ = map(int, persistent.groups())
            key = (session, sequence)
            if key not in admissions or sequence not in kernel_ids or key in gates:
                raise ValueError("unexpected or duplicate real gate selection")
            gates[key] = position
        elif issue:
            session, sequence, retained, retired, _ = map(int, issue.groups())
            key = (session, sequence)
            if (key not in admissions or key in issues or admissions[key] >= position or
                    len(issues) >= len(native_ids) or sequence != native_ids[len(issues)] or
                    retained >= 1 << 64 or retired >= 1 << 64 or
                    (sequence in kernel_ids and (key not in gates or gates[key] >= position))):
                raise ValueError("missing, duplicate or unordered actual issue")
            issues[key] = position
        else:
            raise ValueError("unknown or malformed H2H audit record")
    if (len(admissions) != 40 or set(admissions) != set(issues) or len(gates) != 4 or
            len({session for session, _ in admissions}) != 1):
        raise ValueError("incomplete H2H actual issue ledger")
    session = next(iter(admissions))[0]
    if set(events) != {(session, sequence) for sequence in event_ids}:
        raise ValueError("missing independently recorded gate events")
    for (_, sequence), rows in events.items():
        if (len(rows) != 3 or not admissions[(session, sequence - 1)] < rows[0][0] <
                admissions[(session, sequence + 1)] or
                issues[(session, sequence - 1)] >= admissions[(session, sequence + 1)] or
                not issues[(session, sequence + 1)] < rows[1][0] < rows[2][0]):
            raise ValueError("event lifecycle is not bound to its actual gate and release copy")
    return dict(actual_generic_copy_selection=True, copy_commands=36, kernel_commands=4,
                event_generations=4,
                selected_persistent_gates=4, workers=320, physical_retirement_qualified=False,
                full_device_qualified=False, priority_qualified=False, performance_qualified=False)
