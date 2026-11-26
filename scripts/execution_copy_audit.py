"""Strict CPU-only source-copy/RPC and authenticated owner timing validation.

Only the existing source API kinds1..4 and owner operations22..25 carry these
clocks. Generic opcode141 admission/issue is ordinary memory evidence, not an
instrumented source-copy API or owner timing record.
"""

import re

from execution_priority_audit import LOG_PREFIX, _parse_audit
from execution_semantic_audit import parse_semantic_audit

SOURCE = "Glimmer execution copy audit: "
RPC = "Glimmer execution copy rpc: "
OWNER = "Copy request audit: "
HEADER_FIELDS = ("pid", "serial", "api", "bytes", "entry_ns", "locked_ns", "finish_ns",
                 "cpu_entry_ns", "cpu_finish_ns", "has_result", "status", "calls", "dropped")
RPC_FIELDS = ("pid", "serial", "call", "operation", "session", "request", "begin_ns",
              "submitted_ns", "done_ns", "has_reply", "status")
OWNER_FIELDS = ("connection", "session", "request", "command", "pid", "operation", "frame_ns", "owner_ns",
                "reply_queued_ns", "status")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pattern(prefix, fields, suffix=""):
    return re.compile(prefix + "v=1, " + ", ".join(
        name + (r"=(-?\d+)" if name == "status" else r"=(\d+)") for name in fields) + suffix)


def numeric(match, fields):
    result = dict(zip(fields, map(int, match.groups())))
    require(all(-(1 << 31) <= value < 1 << 31 if name == "status" else 0 <= value < 1 << 64
                for name, value in result.items()), "copy audit numeric range changed")
    return result


def parse_source_copies(content, expected_pid):
    require(type(expected_pid) is int and 0 < expected_pid < 1 << 32 and content.endswith("\n"),
            "exact source PID and complete log required")
    patterns = {SOURCE: pattern(re.escape(SOURCE), HEADER_FIELDS),
                RPC: pattern(re.escape(RPC), RPC_FIELDS)}
    groups = {}
    for line in content.splitlines():
        present = [marker for marker in patterns if marker in line]
        if not present:
            continue
        require(len(present) == 1 and line.count(present[0]) == 1, "ambiguous copy audit line")
        marker = present[0]
        match = patterns[marker].fullmatch(line)
        require(match is not None, "malformed source copy audit")
        row = numeric(match, HEADER_FIELDS if marker == SOURCE else RPC_FIELDS)
        require(row["pid"] == expected_pid and row["serial"] > 0, "foreign source copy identity")
        key = row["pid"], row["serial"]
        if marker == SOURCE:
            require(key not in groups and row["api"] in (1, 2, 3, 4) and row["calls"] <= 8 and
                    row["has_result"] in (0, 1) and row["entry_ns"] > 0 and
                    (row["finish_ns"] == 0 or row["finish_ns"] >= row["entry_ns"]) and
                    (row["locked_ns"] == 0 or row["entry_ns"] <= row["locked_ns"] <= row["finish_ns"]) and
                    ((row["cpu_entry_ns"] == row["cpu_finish_ns"] == 0) or
                     0 < row["cpu_entry_ns"] <= row["cpu_finish_ns"]), "invalid copy API lifetime/clocks")
            groups[key] = dict(row, rpcs=[])
        else:
            require(key in groups, "RPC has no original source API")
            require(row["operation"] != 141,
                    "generic copy has no instrumented source API timing scope")
            parent = groups[key]
            require(row["call"] == len(parent["rpcs"]) + 1 <= parent["calls"] and
                    row["has_reply"] in (0, 1) and row["session"] > 0 and
                    parent["entry_ns"] <= row["begin_ns"] <= row["done_ns"] <= parent["finish_ns"] and
                    (row["begin_ns"] <= row["submitted_ns"] <= row["done_ns"] if row["request"] else
                     row["submitted_ns"] == 0 and row["has_reply"] == 0), "invalid exact source RPC lifecycle")
            parent["rpcs"].append(row)
    require(groups and all(len(row["rpcs"]) == row["calls"] for row in groups.values()),
            "complete source API/RPC observations required")
    return sorted(groups.values(), key=lambda row: row["entry_ns"])


def parse_owner_copies(content):
    semantic = parse_semantic_audit(content)
    bindings = {row.value("connection"): row for row in semantic.family(4) if row.value("event") == 1}
    expression = pattern(LOG_PREFIX + re.escape(OWNER), OWNER_FIELDS,
                         r"; gpu_completion_qualified=false")
    rows, position = {}, 0
    for original in content.splitlines(keepends=True):
        if OWNER in original:
            match = expression.fullmatch(original.rstrip("\r\n"))
            require(match is not None and original.endswith("\n") and original.count(OWNER) == 1,
                    "malformed owner copy audit")
            row = numeric(match, OWNER_FIELDS)
            key = row["session"], row["request"]
            binding = bindings.get(row["connection"])
            require(key not in rows and row["connection"] == row["session"] > 0 and row["request"] > 0 and
                    row["operation"] in (22, 23, 24, 25) and
                    0 < row["frame_ns"] <= row["owner_ns"] <= row["reply_queued_ns"] and
                    binding is not None and binding.position < position and binding.value("pid") == row["pid"],
                    "owner copy lacks exact authenticated request/frame provenance")
            rows[key] = row
        position += len(original)
    require(rows, "no actual owner copy timing")
    return rows


def first_post_release_copy(source, owner, expected_pid, release_begin_ns, release_end_ns):
    require(type(release_begin_ns) is type(release_end_ns) is int and
            0 < release_begin_ns <= release_end_ns, "actual exclusive-release timestamp bracket required")
    apis = parse_source_copies(source, expected_pid)
    owners = parse_owner_copies(owner)
    admissions, events, _ = _parse_audit(owner)
    candidates = []
    for row in apis:
        if row["entry_ns"] < release_begin_ns:
            continue
        for call in row["rpcs"]:
            if call["operation"] not in (23, 25):
                continue
            server = owners.get((call["session"], call["request"]))
            require(server is not None and server["pid"] == expected_pid and
                    server["operation"] == call["operation"],
                    "post-release source copy lacks its original authenticated owner result")
            key = server["connection"], server["command"]
            require(key in admissions and admissions[key][0] == call["operation"],
                    "owner copy request lacks its exact typed GPU command")
            issued = [position for position, event in events if
                      (event["session"], event["sequence"]) == key and
                      event["event"] in ("low_issued", "high_issued")]
            require(len(issued) == 1 and admissions[key][1] < issued[0],
                    "one admitted actual GPU copy issue required")
            candidates.append((issued[0], row, key))
    first = min(candidates, key=lambda value: value[0]) if candidates else None
    selected = first[1] if first else None
    require(selected is not None and selected["has_result"] == 1 and selected["status"] == 0 and
            selected["dropped"] == 0 and selected["locked_ns"] > 0, "first actual post-release copy is incomplete")
    joined = []
    for call in selected["rpcs"]:
        if call["operation"] not in (22, 23, 24, 25):
            continue
        server = owners.get((call["session"], call["request"]))
        require(server is not None and server["pid"] == expected_pid and server["operation"] == call["operation"] and
                server["status"] == call["status"] == 0 and call["has_reply"] == 1 and
                call["begin_ns"] <= call["submitted_ns"] <= server["frame_ns"] <= server["owner_ns"] <=
                server["reply_queued_ns"] <= call["done_ns"], "source/owner copy clocks or exact RPC join differ")
        joined.append(dict(operation=call["operation"], session=call["session"], request=call["request"],
            command=server["command"],
            source_before_submit_ns=call["submitted_ns"] - call["begin_ns"],
            submit_to_frame_ns=server["frame_ns"] - call["submitted_ns"],
            frame_to_owner_ns=server["owner_ns"] - server["frame_ns"],
            owner_to_reply_queue_ns=server["reply_queued_ns"] - server["owner_ns"],
            reply_queue_to_source_ns=call["done_ns"] - server["reply_queued_ns"]))
    return dict(api=selected, first_copy_issue_position=first[0], first_copy_command=list(first[2]),
        source_api_count=len(apis), joined_copy_rpcs=joined,
        release_to_api_entry_lower_ns=max(0, selected["entry_ns"] - release_end_ns),
        release_to_api_entry_upper_ns=selected["entry_ns"] - release_begin_ns,
        source_lock_wait_ns=selected["locked_ns"] - selected["entry_ns"],
        source_api_wall_ns=selected["finish_ns"] - selected["entry_ns"],
        source_api_thread_cpu_ns=(selected["cpu_finish_ns"] - selected["cpu_entry_ns"]
                                 if selected["cpu_entry_ns"] else None),
        diagnostic_only=True, gpu_completion_qualified=False, ordinary_timing_qualified=False,
        full_application_physical_priority_qualified=False, performance_qualified=False)
