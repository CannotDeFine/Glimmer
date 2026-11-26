"""Independent, bounded operation141 issue ledger; never GPU retirement proof."""

import re

MAX_BYTES = 2 * 1024 * 1024
PREFIX = r"[ \t]*(?:\[[^\]\r\n]*\][ \t]*)*"
UINT = r"(0|[1-9][0-9]{0,19})"
ADMISSION = re.compile(PREFIX + r"Native admission audit: session=" + UINT + r", sequence=" +
                       UINT + r", accepted=true, operation=" + UINT)
ISSUE = re.compile(PREFIX + r"Native execution audit: session=" + UINT + r", issued=" + UINT +
                   r", retained=" + UINT + r", retired=" + UINT + r", status=(0)")
AUDIT_CANDIDATE = re.compile(
    r"\bNative\s+(?:admission|execution|library|persistent|kernel)\b|\bNative\b.*\baudit\b",
    re.IGNORECASE)


def validate_execution(raw, *, expected_commands=200):
    """Require the exact fixed input's admitted and actually issued copies."""
    if type(expected_commands) is not int or not 1 <= expected_commands <= 256:
        raise ValueError("invalid fixed generic-copy inventory")
    return validate_operations(raw, (141,) * expected_commands)


def validate_operations(raw, expected_operations):
    """Typed fixed test inventory; callers never supply this through the wire."""
    if (type(expected_operations) is not tuple or not 1 <= len(expected_operations) <= 256 or
            any(type(operation) is not int or operation not in (23, 25, 141)
                for operation in expected_operations)):
        raise ValueError("invalid fixed copy-operation inventory")
    expected_commands = len(expected_operations)
    if type(raw) is bytes:
        if len(raw) > MAX_BYTES:
            raise ValueError("oversized native ledger")
        raw = raw.decode("utf-8", errors="strict")
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_BYTES:
        raise ValueError("native ledger must be bounded UTF-8 text")
    admitted, issued = {}, {}
    for position, line in enumerate(raw.splitlines()):
        if len(line.encode("utf-8")) > 32 * 1024:
            raise ValueError("oversized native ledger line")
        if re.search(r"\[(?:error|critical)\]", line, re.IGNORECASE):
            raise ValueError("service error in successful generic-copy matrix")
        for status in re.findall(r"\b[A-Za-z_]*status\s*=\s*([^,;\s]+)", line):
            if status != "0":
                raise ValueError("failed or malformed service status")
        # A malformed/truncated known audit cannot disappear merely because its
        # exact marker was changed. Ordinary unrelated service text is allowed.
        if AUDIT_CANDIDATE.search(line) is None:
            continue
        admission, issue = ADMISSION.fullmatch(line), ISSUE.fullmatch(line)
        if admission is not None:
            session, sequence, operation = map(int, admission.groups())
            key = (session, sequence)
            if (not 0 < session < 1 << 64 or not 1 <= sequence <= expected_commands or
                    sequence != len(admitted) + 1):
                raise ValueError("invalid or unexpected command identity")
            if operation != expected_operations[sequence - 1] or key in admitted:
                raise ValueError("duplicate or non-generic admission")
            admitted[key] = position
        elif issue is not None:
            session, sequence, retained, retired, status = map(int, issue.groups())
            key = (session, sequence)
            if (key not in admitted or key in issued or admitted[key] >= position or status or
                    sequence != len(issued) + 1 or
                    not 0 <= retained < 1 << 64 or not 0 <= retired < 1 << 64):
                raise ValueError("untyped, failed, duplicate or malformed actual issue")
            issued[key] = position
        else:
            raise ValueError("unexpected or malformed native audit")
    if (len(admitted) != expected_commands or set(admitted) != set(issued) or
            len({session for session, _ in admitted}) != 1 or
            {sequence for _, sequence in admitted} != set(range(1, expected_commands + 1))):
        raise ValueError("incomplete exact generic-copy command ledger")
    generic_only = all(operation == 141 for operation in expected_operations)
    result = dict(operation=141 if generic_only else None,
                  admitted_commands=expected_commands, issued_commands=expected_commands,
                  actual_generic_copy_selection=generic_only, physical_retirement_qualified=False,
                  full_device_qualified=False, priority_qualified=False, performance_qualified=False)
    if not generic_only:
        result["actual_copy_selection"] = True
    return result
