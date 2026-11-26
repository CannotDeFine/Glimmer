"""Strict current-candidate numerical intake; never hardware or priority proof.

Historical collectors keep their byte-bound parsers. New collectors use this
composition to retain native-batch/ordering ownership and independent statuses.
Only the standard library and the existing pure audit parsers are imported.
"""

import re

from execution_semantic_audit import parse_semantic_audit


_PREFIX = r"[ \t]*(?:\[[^\]\r\n]*\][ \t]*)*"
_PATTERNS = {
    "Native admission audit": r"session=(\d+), sequence=(\d+), accepted=true, operation=(\d+)",
    "Native issue audit": r"session=(\d+), sequence=(\d+), status=(-?\d+)",
    "Native execution audit": r"session=(\d+), issued=(\d+), retained=(\d+), retired=(\d+), status=(-?\d+)",
    "Native persistent audit": r"session=(\d+), sequence=(\d+), total_blocks=(\d+), physical_launches=(\d+), workers=(\d+)",
    "Native library issue audit": r"session=(\d+), sequence=(\d+), status=(-?\d+)",
    "Native kernel pipeline audit": r"session=(\d+), first=(\d+), second=(\d+), both_pending=true",
}
_RECORDS = {name: re.compile(_PREFIX + name + ": " + pattern)
            for name, pattern in _PATTERNS.items()}
_ORDERING = {48, 52, 54}


def execution_application_receipt(content, *, expected_workers, require_geam=False):
    """Validate the whole log before returning only unqualified issue counts.

    Actual filtered source progress can omit the legacy issue log, so its exact
    grant-bound result supplies a separate issue receipt. Repeated progress of
    one admitted command is counted as attempts, never extra logical commands.
    Enqueue, physical completion and useful compute are not inferred here.
    """
    if type(expected_workers) is not int or not 1 <= expected_workers <= 0xffffffff:
        raise ValueError("explicit positive worker geometry is required")
    if type(require_geam) is not bool:
        raise ValueError("GEAM selection must be explicit")
    semantic = parse_semantic_audit(content, require_complete=True)
    begin = next(record for record in semantic.family(2) if record.value("event") == 1)
    service_begin = begin.position
    passive_schema = dict(begin.fields).get("native_audit_schema") == 2
    service_end = next(record.position for record in semantic.family(2) if record.value("event") == 2)
    connections, disconnections = {}, {}
    for record in semantic.family(4):
        if record.value("event") == 1:
            connections[record.value("connection")] = record
        elif record.value("event") == 2:
            disconnections[record.value("connection")] = record.position

    def authenticated(session, position):
        bound = connections.get(session)
        if bound is None or bound.position >= position or bound.value("role") != 0:
            raise ValueError("ordinary command lacks its prior authenticated source")

    admissions, issues, dispatches, persistent, libraries, pipelines = {}, {}, {}, {}, {}, []
    position = 0
    for original in content.splitlines(keepends=True):
        line = original.rstrip("\r\n")
        if re.search(r"\[(?:error|critical)\]", line, re.I):
            raise ValueError("service logged an unexplained error")
        for status in re.findall(r"\b[A-Za-z_]*status\s*=\s*([^,;\s]+)", line):
            if re.fullmatch(r"-?\d+", status) is None or int(status):
                raise ValueError("nonzero or malformed untyped execution status")
        markers = [name for name in _RECORDS if name in line]
        if not markers:
            if "Native " in line and " audit:" in line:
                raise ValueError("unknown native audit record")
            position += len(original)
            continue
        marker = markers[0]
        match = _RECORDS[marker].fullmatch(line)
        if len(markers) != 1 or line.count(marker) != 1 or match is None:
            raise ValueError("malformed native audit record")
        if not service_begin < position < service_end:
            raise ValueError("native execution evidence is outside the actual service lifecycle")
        values = tuple(map(int, match.groups()))
        if any(value < 0 or value > 0xffffffffffffffff for value in values):
            raise ValueError("native audit integer outside its unsigned domain")
        key = values[:2]
        if not all(key):
            raise ValueError("zero native command identity")
        authenticated(key[0], position + match.start())
        if marker == "Native admission audit":
            if key[0] in disconnections and position >= disconnections[key[0]]:
                raise ValueError("ordinary admission follows its authenticated disconnect")
            if key in admissions or not 1 <= values[2] <= 130 or values[2] in _ORDERING:
                raise ValueError("duplicate or invalid ordinary admission")
            admissions[key] = values[2], position
        elif marker == "Native kernel pipeline audit":
            pipelines.append((values, position))
        else:
            if key not in admissions or admissions[key][1] >= position:
                raise ValueError("actual issue lacks a preceding exact ordinary admission")
            selected = (issues if marker == "Native execution audit" else
                        dispatches if marker == "Native issue audit" else
                        persistent if marker == "Native persistent audit" else libraries)
            if key in selected:
                raise ValueError("repeated ordinary issue or selection record")
            if marker == "Native persistent audit":
                if not values[2] or not values[3] or values[4] != expected_workers:
                    raise ValueError("invalid actual persistent issue geometry")
            elif values[-1]:
                raise ValueError("actual native or library issue failed")
            selected[key] = position
        position += len(original)

    source_receipts, source_libraries = {}, set()
    ordering_issues = [record for record in semantic.family(3)
                       if record.value("event") == 2]
    ordering_keys = {(row.value("connection"), row.value("sequence"))
                     for row in ordering_issues}
    ordering_admissions = {(row.value("connection"), row.value("sequence")): row
                           for row in semantic.family(3) if row.value("event") == 1}
    ordering_issue_positions = {(row.value("connection"), row.value("sequence")): row.position
                               for row in ordering_issues}
    source_ordering = set()
    for record in semantic.family(2):
        if record.value("event") != 12 or record.value("issue_state") != 1:
            continue
        key = record.value("source"), record.value("command_sequence")
        if key in admissions:
            if admissions[key][1] >= record.position:
                raise ValueError("source result precedes its exact ordinary admission")
            source_receipts.setdefault(key, []).append(record.position)
            if record.value("has_library"):
                source_libraries.add(key)
        elif key not in ordering_keys:
            raise ValueError("source result lacks an exact ordinary or ordering admission")
        else:
            if (key in source_ordering or key not in ordering_admissions or
                    not ordering_admissions[key].position < record.position < ordering_issue_positions[key]):
                raise ValueError("source result lacks a prior ordering admission and its next actual issue")
            source_ordering.add(key)
    # Current records distinguish issue from physical retirement; legacy logs
    # have only the older execution observation. Never promote a retirement
    # record into issue proof if the current issue schema is present.
    if passive_schema and (set(issues) - set(dispatches) or
                           any(dispatches[key] >= position for key, position in issues.items())):
        raise ValueError("physical retirement lacks its earlier exact issue")
    if not passive_schema and dispatches:
        raise ValueError("native issue records require their declared schema")
    issue_positions = dispatches if passive_schema else issues
    actual = set(issue_positions) | set(source_receipts)
    if actual != set(admissions):
        raise ValueError("missing or incomplete actual ordinary command issues")
    native_submissions = [record for record in semantic.family(1)
                          if record.value("event") == 8]
    if not actual and not native_submissions and not ordering_issues:
        raise ValueError("empty execution evidence")
    for selections in (persistent, libraries):
        for key, selected_position in selections.items():
            positions = sorted(([issue_positions[key]] if key in issue_positions else []) +
                               source_receipts.get(key, []))
            if not positions or selected_position >= positions[0]:
                raise ValueError("private/library selection does not precede actual issue")
    for (session, first, second), pipeline_position in pipelines:
        if first == second or any((session, sequence) not in actual for sequence in (first, second)):
            raise ValueError("pipeline lacks two distinct actual commands")
        for sequence in (first, second):
            key = session, sequence
            positions = sorted(([issue_positions[key]] if key in issue_positions else []) + source_receipts.get(key, []))
            if positions[0] >= pipeline_position:
                raise ValueError("pipeline precedes one of its actual issues")
    # An ordering lifecycle never enters the ordinary admission map or these
    # persistent/library counts. Native batches remain opaque owned work.
    geam = {key for key, (operation, _) in admissions.items() if operation == 129}
    library_keys = set(libraries) | source_libraries
    if ((require_geam and not geam) or not geam <= library_keys or geam & persistent.keys()):
        raise ValueError("GEAM requires its original owned library route")
    if "Priority execution audit" in content:
        from execution_priority_audit import _parse_audit_model
        _parse_audit_model(content)
    return dict(unqualified_actual_issue_count=len(actual),
                unqualified_persistent_issue_count=len(persistent),
                unqualified_original_library_issue_count=len(library_keys),
                unqualified_geam_issue_count=len(geam),
                ordering_issue_count=len(ordering_issues),
                native_submission_count=len(native_submissions),
                source_granted_issue_attempts=sum(map(len, source_receipts.values())),
                admitted_opcode_values=sorted({operation for operation, _ in admissions.values()}),
                semantic_service=semantic.service, semantic_records=len(semantic.records),
                semantic_complete=semantic.is_complete,
                compute_family_inventory_qualified=False, useful_sm_coverage_qualified=False,
                physical_priority_qualified=False, saturation_qualified=False,
                performance_qualified=False)
