"""Offline exact-command audit for the ordinary Runtime arrival GPU input."""

import json
import re

from execution_priority_audit import LOG_PREFIX, kernel_phase
from execution_semantic_audit import parse_semantic_audit

PHASE = "PUBLIC_ARRIVAL_PHASE\n"
ARRIVAL = "PUBLIC_ARRIVAL_NEW_INPUT\n"
NATIVE = "PUBLIC_ARRIVAL_NATIVE "
TOTAL = 65536


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def load_json(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


def positive(value):
    return type(value) is int and 0 < value < 1 << 64


def native_trace(rows, count):
    require(rows and all(type(row) is dict and positive(row.get("serial")) for row in rows),
            "bounded native records required")
    require(len(rows) <= 96 and all(left["serial"] < right["serial"]
                                   for left, right in zip(rows, rows[1:])),
            "native callback order must be exact")
    roots = {}
    for row in rows:
        if row.get("event") == "launch":
            fields = {"event", "serial", "kind", "context", "module", "function", "stream",
                      "workers", "arguments"}
            require(set(row) == fields and row["kind"] in ("persistent", "original") and
                    all(positive(row[key]) for key in fields - {"event", "kind", "arguments"}),
                    "exact native launch fields required")
            args = row["arguments"]
            require(type(args) is list and len(args) == 13 and
                    all(type(value) is int and 0 <= value < 1 << 64 for value in args) and
                    all(value < 1 << 32 for value in args[1:11]) and
                    args[0] > 0 and args[1] == TOTAL * 32 and args[3:7] == [1, 8192, 0, 0],
                    "original non-idempotent ABI changed")
            if row["kind"] == "persistent":
                require(row["workers"] in (160, 320) and args[7:10] == [TOTAL, 1, 1] and
                        args[11] > 0 and args[12] > 0 and args[11] != args[0],
                        "explicit persistent geometry and private counter required")
            else:
                require(row["workers"] == TOTAL and args[7:] == [0] * 6,
                        "original fallback cannot acquire a continuation counter")
            previous = roots.setdefault(args[0], [])
            if previous:
                require(all(row[key] == previous[0][key] for key in
                            fields - {"serial", "workers"}),
                        "partial original replay or changed continuation ABI/identity")
            previous.append(row)
        elif row.get("event") == "counter_initialize":
            require(set(row) == {"event", "serial", "counter", "value"} and
                    positive(row["counter"]) and type(row["value"]) is int and not row["value"],
                    "private counter must initialize exactly to zero")
        else:
            require(row.get("event") == "counter_free" and
                    set(row) == {"event", "serial", "counter"} and positive(row["counter"]),
                    "unknown native callback")
    require(len(roots) == count, "every independent command needs its actual output")
    launches = [row for row in rows if row["event"] == "launch"]
    require(len({row["context"] for row in launches}) == 1 and
            len({group[0]["module"] for group in roots.values()}) == count and
            len({group[0]["stream"] for group in roots.values()}) == count,
            "one common compute owner and independent client namespaces required")
    consumed = set()
    for group in roots.values():
        initial, final = group[0], group[-1]
        if initial["kind"] == "original":
            require(len(group) == 1, "an original may not be replayed")
            continue
        counter = initial["arguments"][11]
        related = [row for row in rows if row.get("counter") == counter]
        before = [row for row in related if row["serial"] < initial["serial"]]
        after = [row for row in related if row["serial"] > final["serial"]]
        require(before and before[-1]["event"] == "counter_initialize" and after and
                after[0]["event"] == "counter_free" and not any(
                    initial["serial"] <= row["serial"] <= final["serial"] for row in related),
                "counter reset, early free or missing retained lifetime")
        pair = {before[-1]["serial"], after[0]["serial"]}
        require(not pair & consumed, "one initialization/free cannot serve two live commands")
        consumed.update(pair)
    require(consumed == {row["serial"] for row in rows if row["event"] != "launch"},
            "unattributed or duplicate counter initialization/free")
    return roots, launches


def verify_arrival(content, cached=False, learning=False):
    require(type(cached) is type(learning) is bool, "explicit entry/stage required")
    require(content.endswith("\n") and content.count(PHASE) == content.count(ARRIVAL) == 1,
            "unique complete ordinary phase and arrival records required")
    offset, arrival = content.index(PHASE) + len(PHASE), content.index(ARRIVAL)
    require(offset <= arrival if learning else offset < arrival,
            "new input cannot precede the ordinary phase")
    lines = content.splitlines()
    require(lines.count(PHASE.strip()) == lines.count(ARRIVAL.strip()) == 1 and
            all(NATIVE not in line or line.startswith(NATIVE) for line in lines),
            "markers must occupy complete records")
    for line in lines:
        require(not re.search(r"\[(error|critical)\]", line, re.IGNORECASE), "service error")
        for value in re.findall(r"\b[A-Za-z_]*status\s*=\s*([^,;\s]+)", line):
            require(re.fullmatch(r"-?\d+", value) is not None and int(value) == 0,
                    "unexpected or malformed SDK status")
    require(lines[-1].startswith("{") and sum(line.startswith("{") for line in lines) == 1,
            "one complete terminal receipt required")
    actual = load_json(lines[-1])
    count = 2 if learning else 3
    expected = dict(status="passed", stage="pair-profile" if learning else "arrival", cached=cached,
                    ordinary_public_children=count, common_contexts=1, common_destroys=1,
                    native_contexts=count + 1, native_destroys=count + 1, trained_pairs=2,
                    actual_measurements=True, cleanup=True, hardware_co_residency_qualified=False,
                    full_device_qualified=False, priority_qualified=False,
                    performance_qualified=False, full_goal_qualified=False)
    require(type(actual) is dict and set(actual) == set(expected) | {
        "phase_native_launches", "shared_worker_launches"}, "exact assembly receipt fields")
    for key, value in expected.items():
        require(type(actual[key]) is type(value) and actual[key] == value, f"changed {key}")
    rows = [load_json(line[len(NATIVE):]) for line in lines if line.startswith(NATIVE)]
    roots, launches = native_trace(rows, count)
    require(type(actual["phase_native_launches"]) is type(actual["shared_worker_launches"]) is int and
            actual["phase_native_launches"] == len(launches) and
            actual["shared_worker_launches"] == sum(row["workers"] == 160 for row in launches),
            "native callback and actual geometry counts differ")
    _, phase = kernel_phase(content, offset)
    require(phase and all(row["priority"] == 0 for _, row in phase), "exact equal priorities")
    require(not any(row["event"].startswith("high_") or row["event"] == "nonpreemptible_wait"
                    for _, row in phase), "arrival input cannot borrow high-priority evidence")
    initial = [(position, row) for position, row in phase if row["event"] == "low_issued"]
    keys = [(row["session"], row["sequence"]) for _, row in initial]
    require(len(keys) == count and len(set(keys)) == count and
            len({session for session, _ in keys}) == count, "one original command per client")
    semantic = parse_semantic_audit(content)
    bindings = {row.value("connection"): row for row in semantic.family(4)
                if row.value("event") == 1}
    require(all(session in bindings and bindings[session].value("role") == 0 and
                bindings[session].value("effective") == 1 and
                bindings[session].value("priority") == 0 for session, _ in keys) and
            len({bindings[session].value("pid") for session, _ in keys}) == count,
            "independent authenticated configured ordinary clients required")
    require(len(roots) == len(keys), "native roots must join every typed command")
    root_groups = list(roots.values())
    first_stop = None
    for index, (key, group) in enumerate(zip(keys, root_groups)):
        transitions = [(position, row) for position, row in phase
                       if (row["session"], row["sequence"]) == key]
        names = [row["event"] for _, row in transitions]
        require(names[0] == "low_issued" and names[-1] == "low_logical_done" and
                names.count("low_issued") == names.count("low_logical_done") == 1,
                "complete once-only logical lifecycle required")
        persistent = group[0]["kind"] == "persistent"
        completed, generation, stopped, requested = 0, int(persistent), False, False
        for position, row in transitions:
            # All three inputs were accepted as configured persistent commands.
            # A measured original fallback retains that logical progress record,
            # but has ZERO persistent physical generations/native counters.
            require(row["total"] == TOTAL, "changed original logical geometry")
            name = row["event"]
            if name == "low_issued":
                require(row["launches"] == generation and not row["completed"], "initial epoch")
            elif name == "retreat_requested":
                require(persistent and not stopped and not requested and row["launches"] == generation,
                        "retreat must bind one actual active epoch")
                requested = True
            elif name == "low_physical_stop":
                require(persistent and requested and not stopped and row["launches"] == generation and
                        completed <= row["completed"] < TOTAL, "invalid actual incomplete stop")
                completed, stopped = row["completed"], True
                if index == 0 and first_stop is None:
                    require(position > arrival and 0 < completed < TOTAL, "positive first A stop after input")
                    first_stop = dict(row, position=position)
            elif name == "low_resumed":
                require(stopped and row["completed"] == completed and row["launches"] == generation + 1,
                        "resume must preserve the exact stopped prefix")
                generation, stopped, requested = generation + 1, False, False
            elif name == "low_logical_done":
                require(not stopped and row["launches"] == generation and
                        row["completed"] == TOTAL, "complete logical retirement")
            else:
                require(False, "unknown kernel transition")
        require(generation == (len(group) if persistent else 0), "all native continuations must join")
    if learning:
        require(len(launches) == count and all(row["workers"] == 160 for row in launches) and
                all(position > arrival for position, _ in initial), "actual isolated160/160 sample")
    else:
        require(len([1 for position, _ in initial if position < arrival]) == 1 and first_stop and
                root_groups[0][0]["workers"] == 320 and len(root_groups[0]) >= 2 and
                all(position > first_stop["position"] for position, _ in initial[1:]),
                "A must actually stop before new ordinary command issue")
        require(all(position > arrival for position, row in phase if row["event"] in
                    ("retreat_requested", "low_physical_stop", "low_resumed", "low_logical_done")),
                "warmup or earlier natural completion cannot supply an arrival checkpoint")
    return dict(assembly=actual, native_callbacks=rows, command_keys=keys,
                first_incomplete_stop=first_stop, complete_native_joins=True,
                checked_values=count * TOTAL * 32, ordinary_timing_qualified=False)


def require_official_configuration(content):
    marker = "Execution service configuration: "
    pattern = re.compile(LOG_PREFIX + re.escape(marker) +
        r"scheduler=workload-aware, kernel_mode=(preemptive|sliced), configured_workers=320, "
        r"nccl_enabled=false; full_device_qualified=false")
    rows = [pattern.fullmatch(line) for line in content.splitlines() if marker in line]
    require(len(rows) == 1 and rows[0] is not None and NATIVE not in content and
            not re.search(r"\[(error|critical)\]", content, re.IGNORECASE),
            "complete healthy official workload-aware configuration required")


def verify_service_arrival(content, phase_offset, arrival_offset, expected_pids):
    """Official service proof; deliberately confers NO native-pointer trace credit."""
    require(type(phase_offset) is type(arrival_offset) is int and
            0 < phase_offset < arrival_offset < len(content) and content.endswith("\n"),
            "exact complete official service phase/input boundaries required")
    require(type(expected_pids) in (tuple, list) and len(expected_pids) == 3 and
            all(type(pid) is int and 0 < pid < 1 << 32 for pid in expected_pids) and
            len(set(expected_pids)) == 3, "three exact ordinary client PIDs required")
    require_official_configuration(content)
    _, phase = kernel_phase(content, phase_offset)
    initial = [(position, row) for position, row in phase if row["event"] == "low_issued"]
    keys = [(row["session"], row["sequence"]) for _, row in initial]
    require(len(keys) == len(set(keys)) == 3 and len({key[0] for key in keys}) == 3 and
            all(row["priority"] == 0 for _, row in phase), "one equal-priority command per client")
    semantic = parse_semantic_audit(content)
    bindings = {row.value("connection"): row for row in semantic.family(4) if row.value("event") == 1}
    require(all(session in bindings and bindings[session].value("role") == 0 and
                bindings[session].value("effective") == 1 and bindings[session].value("retained") == 1 and
                bindings[session].value("priority") == 0 for session, _ in keys) and
            {bindings[session].value("pid") for session, _ in keys} == set(expected_pids),
            "official commands lack exact authenticated configured application sources")
    require(sum(position < arrival_offset for position, _ in initial) == 1 and
            initial[0][1]["launches"] == 1 and initial[0][1]["total"] == TOTAL,
            "ordinary new input must follow exactly one actual A issue")
    source, sequence = keys[0]
    native = list(re.finditer(
        rf"Native persistent audit: session={source}, sequence={sequence}, total_blocks={TOTAL}, "
        r"physical_launches=1, workers=320(?=\r?\n)", content))
    require(len(native) == 1 and initial[0][0] < native[0].start() < arrival_offset,
            "actual initial explicit320-worker persistent issue required")
    require(not any(row["event"].startswith("high_") or row["event"] == "nonpreemptible_wait"
                    for _, row in phase), "equal-priority input cannot borrow high/native-only credit")
    first_stop, completions = None, []
    for index, key in enumerate(keys):
        transitions = [(position, row) for position, row in phase
                       if (row["session"], row["sequence"]) == key]
        names = [row["event"] for _, row in transitions]
        require(names[0] == "low_issued" and names[-1] == "low_logical_done" and
                names.count("low_issued") == names.count("low_logical_done") == 1,
                "complete once-only official command lifecycle required")
        generation = transitions[0][1]["launches"]
        require(generation in (0, 1), "initial original/private generation required")
        persistent = generation == 1
        # A single-profile choice can admit the original without ever creating
        # a progress lease. A later paired original choice can retain the
        # prepared logical total, but neither path has persistent generations.
        # This is progress metadata, not the original kernel's logical grid.
        progress_total = transitions[0][1]["total"]
        require(progress_total == TOTAL if persistent else progress_total in (0, TOTAL),
                "official original/private progress shape changed")
        completed, stopped, requested = 0, False, False
        for position, row in transitions:
            name = row["event"]
            require(row["total"] == progress_total, "official progress shape changed within command")
            if name == "low_issued":
                require(not row["completed"], "official initial work cannot borrow stopped progress")
            elif name == "retreat_requested":
                require(persistent and not stopped and not requested and row["launches"] == generation and
                        row["completed"] == completed, "retreat must bind the exact active official epoch")
                requested = True
            elif name == "low_physical_stop":
                require(persistent and requested and not stopped and row["launches"] == generation and
                        completed <= row["completed"] < TOTAL, "official incomplete stop changed")
                completed, stopped = row["completed"], True
                if index == 0 and first_stop is None:
                    require(position > arrival_offset and 0 < completed < TOTAL,
                            "positive same-A incomplete stop must follow ordinary new input")
                    first_stop = dict(row, position=position)
            elif name == "low_resumed":
                require(stopped and row["completed"] == completed and row["launches"] == generation + 1,
                        "official continuation must retain its exact stopped prefix")
                generation, stopped, requested = generation + 1, False, False
            elif name == "low_logical_done":
                require(not stopped and row["launches"] == generation and row["completed"] == progress_total,
                        "official command must fully retire, not an early ACK")
            else:
                require(False, "unknown official kernel lifecycle")
        completions.append(dict(session=key[0], sequence=key[1], physical_generations=generation,
                                progress_total=progress_total))
    require(first_stop and all(position > first_stop["position"] for position, _ in initial[1:]) and
            completions[0]["physical_generations"] >= 2 and
            all(position > arrival_offset for position, row in phase if row["event"] in
                ("retreat_requested", "low_physical_stop", "low_resumed", "low_logical_done")),
            "actual same-command stop must precede every new ordinary command and retained resume")
    return dict(command_keys=keys, expected_pids=list(expected_pids), first_incomplete_stop=first_stop,
                completions=completions, typed_same_command_handoff=True,
                native_pointer_trace_qualified=False, independent_pre_demand_useful_compute=False,
                full_device_qualified=False, ordinary_timing_qualified=False)


def verify_service_pair_profile(content, phase_offset, expected_pids):
    """Join official warmup/phase160-worker issues; never claim co-residency."""
    require(type(phase_offset) is int and 0 < phase_offset < len(content) and content.endswith("\n") and
            NATIVE not in content, "complete official pair-profile boundary required")
    require_official_configuration(content)
    require(type(expected_pids) in (tuple, list) and len(expected_pids) == 2 and
            all(type(pid) is int and 0 < pid < 1 << 32 for pid in expected_pids) and
            len(set(expected_pids)) == 2, "two exact ordinary profiling PIDs required")
    marker = "Native persistent audit: "
    pattern = re.compile(LOG_PREFIX + re.escape(marker) +
                         r"session=(\d+), sequence=(\d+), total_blocks=(\d+), physical_launches=1, workers=160")
    native, position = {}, 0
    for original in content.splitlines(keepends=True):
        if marker in original:
            match = pattern.fullmatch(original.rstrip("\r\n"))
            require(match is not None and original.count(marker) == 1,
                    "exact native160-worker pair-profile record required")
            session, sequence, total = map(int, match.groups())
            key = session, sequence
            require(positive(session) and positive(sequence) and key not in native and total in (1, TOTAL),
                    "unique original profiling geometry required")
            native[key] = position, total
        position += len(original)
    # Official startup precedes every command. Keep the shared validator's
    # positive boundary while including all warmup and phase admissions.
    _, events = kernel_phase(content, 1)
    issues = [(position, row) for position, row in events if row["event"] == "low_issued"]
    keys = [(row["session"], row["sequence"]) for _, row in issues]
    require(len(keys) == len(set(keys)) == len(native) == 4 and set(keys) == set(native),
            "every warmup/phase profiling command must join its actual native issue")
    semantic = parse_semantic_audit(content)
    bindings = {row.value("connection"): row for row in semantic.family(4) if row.value("event") == 1}
    sessions = {session for session, _ in keys}
    require(len(sessions) == 2 and all(session in bindings and bindings[session].value("role") == 0 and
            bindings[session].value("effective") == bindings[session].value("retained") == 1 and
            bindings[session].value("priority") == 0 for session in sessions) and
            {bindings[session].value("pid") for session in sessions} == set(expected_pids),
            "pair-profile commands need exact authenticated configured sources")
    stages = {True: [], False: []}
    for issue_position, issue in issues:
        key = issue["session"], issue["sequence"]
        transitions = [(position, row) for position, row in events
                       if (row["session"], row["sequence"]) == key]
        require([row["event"] for _, row in transitions] == ["low_issued", "low_logical_done"],
                "isolated profiling cannot borrow retreat/resume or duplicate retirement")
        native_position, total = native[key]
        done_position, done = transitions[-1]
        is_warmup = issue_position < phase_offset
        require(issue_position < native_position < done_position and
                total == (1 if is_warmup else TOTAL) and
                all(row["priority"] == 0 and row["total"] == total and row["launches"] == 1
                    for _, row in transitions) and issue["completed"] == 0 and done["completed"] == total and
                (done_position < phase_offset if is_warmup else done_position > phase_offset),
                "complete original geometry/epoch and phase-specific native issue required")
        stages[is_warmup].append((key, native_position, done_position))
    for stage in stages.values():
        require(len(stage) == 2 and {key[0] for key, _, _ in stage} == sessions and
                max(position for _, position, _ in stage) < min(position for _, _, position in stage),
                "both actual profiling legs must issue before either retirement")
    return dict(command_keys=keys, explicit_workers=160, complete_typed_native_issue_joins=True,
                native_pointer_trace_qualified=False, hardware_co_residency_qualified=False,
                full_device_qualified=False, ordinary_timing_qualified=False)
