#!/usr/bin/env python3
"""Verify physical BERT/ONNX handoff using audits and one immutable GPU trace."""

import argparse
from contextlib import closing
import hashlib
import json
import pathlib
import sqlite3
import sys

from bert_contract import sha256, write_json
from run_bert_baseline import EXAMPLE, SMOKE

sys.path.insert(0, str(SMOKE))
from run_persistent_mlp_handoff import check_physical_audits, records, CONTROL_PREFIX
from measurements import validate_scheduler_metrics

OWNED = "glimmer_persistent_sgemm"
HIGH_BOUNDARIES = ("runtime-markers", "fixed-inventory")
HIGH_KERNELS_PER_REQUEST = 323
HIGH_REQUEST_SHA256 = "545ccb3c163199adf65ae60e7644205f3a42149fc3b72bedec658e1173ab6501"


def fixed_inference_work(kernels, markers, requests=2):
    """Select measured requests only after validating every fixed application pass."""
    if (type(requests) is not int or not 1 <= requests <= 100 or markers
            or len(kernels) != (requests + 2) * HIGH_KERNELS_PER_REQUEST):
        raise ValueError("fixed inference boundary needs all declared full passes and no high markers")
    if (len({(row["device"], row["process"], row["context"], row["stream"], row["api_thread"])
             for row in kernels}) != 1
            or any(left["end"] > right["start"] or left["api_end"] > right["api_start"]
                   for left, right in zip(kernels, kernels[1:]))):
        raise ValueError("fixed inference inventory is not one serial issuing stream")
    for begin in range(0, len(kernels), HIGH_KERNELS_PER_REQUEST):
        signature = []
        for row in kernels[begin:begin + HIGH_KERNELS_PER_REQUEST]:
            shape = row.get("shape", ())
            if (len(shape) != 6 or any(type(value) is not int or value <= 0 for value in shape)
                    or not isinstance(row["name"], str) or not row["name"]
                    or "glimmer" in row["name"].lower()):
                raise ValueError("fixed inference inventory has invalid native work or geometry")
            signature.append([row["name"], *shape])
        actual = hashlib.sha256(json.dumps(signature, separators=(",", ":")).encode()).hexdigest()
        if actual != HIGH_REQUEST_SHA256:
            raise ValueError("fixed inference request name/geometry/order fingerprint differs")
    return kernels[2 * HIGH_KERNELS_PER_REQUEST:]


def read_timeline(database, low_pid, high_pid, high_boundary="runtime-markers", high_requests=2):
    """Use Nsight's serialized process identities; never align host/GPU clocks."""
    if high_boundary not in HIGH_BOUNDARIES:
        raise ValueError("unknown high inference boundary")
    if low_pid == high_pid or min(low_pid, high_pid) <= 0:
        raise ValueError("require two distinct process identities")
    tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME", "StringIds", "PROCESSES"}
    if not required <= tables:
        raise ValueError("missing CUDA timeline tables")
    if "DIAGNOSTIC_EVENT" in tables:
        for severity, message in database.execute(
                "SELECT s.name,d.text FROM DIAGNOSTIC_EVENT d "
                "LEFT JOIN ENUM_DIAGNOSTIC_SEVERITY_LEVEL s ON s.id=d.severity"):
            if severity not in {"Info", "Information"}:
                raise ValueError(f"incomplete or uncertain profiler capture: {severity}: {message}")
    identities = {}
    for pid in (low_pid, high_pid):
        matches = {row[0] for row in database.execute("SELECT globalPid FROM PROCESSES WHERE pid=?", (pid,))}
        if len(matches) != 1 or None in matches:
            raise ValueError("missing or ambiguous profiler process identity")
        identities[pid] = matches.pop()
    rows = database.execute(
        "SELECT k.start,k.end,k.deviceId,k.contextId,k.streamId,k.globalPid,s.value,k.gridId,k.correlationId "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds s ON s.id=k.shortName ORDER BY k.start").fetchall()
    if not rows or {row[5] for row in rows} != set(identities.values()) or len({row[2] for row in rows}) != 1:
        raise ValueError("require exactly the two CUDA processes on one device")
    kernels = []
    seen = set()
    apis = {}
    for start, end, name, thread, correlation in database.execute(
            "SELECT a.start,a.end,s.value,a.globalTid,a.correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME a "
            "JOIN StringIds s ON a.nameId=s.id WHERE s.value LIKE '%LaunchKernel%'"):
        apis.setdefault((thread & 0xFFFFFFFFFF000000, correlation), []).append((start, end, name, thread))
    for start, end, device, context, stream, process, name, grid, correlation in rows:
        identity = (process, context, grid)
        if (type(start) is not int or type(end) is not int or not 0 < start < end
                or not isinstance(name, str) or not name or identity in seen
                or ("glimmer" in name.lower() and name != OWNED)):
            raise ValueError("invalid, duplicated or unknown GPU kernel record")
        seen.add(identity)
        api = apis.get((process, correlation), [])
        if (len(api) != 1 or not 0 < api[0][0] <= api[0][1] or api[0][0] > start
                or "LaunchKernel" not in api[0][2]):
            raise ValueError("kernel has no unique, valid launch API correlation")
        kernels.append(dict(start=start, end=end, device=device, context=context, stream=stream,
                            process=process, name=name, grid=grid, api_start=api[0][0],
                            api_end=api[0][1], api_name=api[0][2], api_thread=api[0][3]))
    high_id = identities[high_pid]
    # Same process extraction used by NVIDIA's installed cuda_kern_exec_sum.py.
    markers = database.execute(
        "SELECT a.start,a.end,s.value,a.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME a "
        "JOIN StringIds s ON a.nameId=s.id WHERE (a.globalTid & 0xFFFFFFFFFF000000)=? "
        "AND (s.value LIKE 'cudaProfilerStart%' OR s.value LIKE 'cudaProfilerStop%')", (high_id,)).fetchall()
    starts = [item for item in markers if item[2].startswith("cudaProfilerStart")]
    stops = [item for item in markers if item[2].startswith("cudaProfilerStop")]
    high = [item for item in kernels if item["process"] == high_id]
    low = [item for item in kernels if item["process"] == identities[low_pid] and item["name"] == OWNED]
    if high_boundary == "fixed-inventory":
        shapes = {(context, grid): (gx, gy, gz, bx, by, bz)
                  for context, grid, gx, gy, gz, bx, by, bz in database.execute(
                      "SELECT contextId,gridId,gridX,gridY,gridZ,blockX,blockY,blockZ "
                      "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=?", (high_id,))}
        for item in high:
            item["shape"] = shapes[(item["context"], item["grid"])]
        high = fixed_inference_work(high, markers, high_requests)
        if not low:
            raise ValueError("missing owned low GPU execution")
        return low, high
    if (len(starts) != 1 or len(stops) != 1 or starts[0][3] != stops[0][3]
            or not 0 < starts[0][0] <= starts[0][1] < stops[0][0] <= stops[0][1]):
        raise ValueError("missing or contradictory high measured CUDA range")
    begin, end = starts[0][1], stops[0][0]
    if any(item["name"] == OWNED or (item["start"] < begin < item["end"])
           or (item["start"] < end < item["end"]) for item in high):
        raise ValueError("high engine is not native or crosses its measured boundary")
    high = [item for item in high if begin <= item["start"] < item["end"] <= end]
    if not high or not low:
        raise ValueError("missing measured high or owned low GPU execution")
    return low, high


def physical_handoff(low_text, high_text, low_kernels, high_kernels, low_release_ns, high_release_ns):
    events = check_physical_audits(low_text)
    prepares = [event for event in events if event["phase"] == "prepare"]
    if (len(prepares) != len(low_kernels) or len({(row["process"], row["context"], row["stream"])
                                               for row in low_kernels}) != 1
            or any(left["time_ns"] >= right["time_ns"] for left, right in zip(prepares, prepares[1:]))
            or any(left["end"] > right["start"] or left["api_end"] > right["api_start"]
                   for left, right in zip(low_kernels, low_kernels[1:]))
            or len({row["api_thread"] for row in low_kernels}) != 1):
        raise ValueError("owned preparations cannot map one-to-one to a complete serial GPU stream")
    grants = [(event["task"], event["generation"]) for event in events if event["phase"] == "grant"]
    if len(grants) != len(set(grants)) or not 0 < low_release_ns < high_release_ns:
        raise ValueError("reused generation or invalid release ordering")

    def key(event):
        return event["task"], event["generation"], event["launch"]

    actual = {key(prepare): gpu for prepare, gpu in zip(prepares, low_kernels)}
    submissions = [event for event in records(high_text, CONTROL_PREFIX)
                   if event["submit"] == 1 and event["status"] == 0
                   and high_release_ns <= event["enqueued_ns"] <= event["started_ns"]
                   <= event["finished_ns"] <= event["observed_ns"]]
    candidates = []
    for first in events:
        if first["phase"] != "first_tile" or first["time_ns"] < low_release_ns:
            continue
        launch = [event for event in events if key(event) == key(first)]
        stops = [event for event in launch if event["phase"] == "suspended"
                 and 0 < event["completed"] < event["logical"]]
        for stop in stops:
            arrivals = [item for item in submissions
                        if first["time_ns"] < item["enqueued_ns"] <= item["finished_ns"] < stop["time_ns"]
                        and any(event["phase"] == "priority_retreat"
                                and item["started_ns"] <= event["time_ns"] <= stop["time_ns"] for event in launch)]
            resumes = [event for event in prepares if event["task"] == stop["task"]
                       and event["launch"] == stop["launch"] + 1 and event["generation"] != stop["generation"]
                       and event["time_ns"] > stop["time_ns"]]
            for resume in resumes:
                progress = [event for event in events if key(event) == key(resume)
                            and event["phase"] in {"first_tile", "suspended", "complete"}
                            and event["completed"] > stop["completed"]]
                before, after = actual[key(stop)], actual[key(resume)]
                native = [item for item in high_kernels
                          if before["end"] < item["api_start"] <= item["start"] < item["end"]
                          < after["api_start"] <= after["start"]]
                if arrivals and progress and native:
                    candidates.append(dict(low_first_tile=first, high_submission=arrivals[0], low_partial_stop=stop,
                        low_resume_prepare=resume, low_resumed=progress[0], low_stopped_gpu=before,
                        high_native_gpu=native[0], low_resumed_gpu=after))
    if not candidates:
        raise ValueError("no independent physical low/native-high/low handoff")
    return {"proof": candidates[0], "owned_physical_launches": len(prepares),
            "owned_logical_tasks": len({event["task"] for event in events}),
            "measured_high_kernel_count": len(high_kernels)}


def declared_high_requests(manifest):
    workload = manifest.get("high_workload", "original")
    count = manifest.get("high_requests", 2)
    if workload == "original" and type(count) is int and count == 2:
        return count
    period = manifest.get("high_period_ms")
    if (workload != "periodic" or manifest.get("high_timeline_boundary") != "fixed-inventory"
            or any(type(value) is not int or value <= 0 for value in (count, period))
            or count > 100 or count * period > 10_000):
        raise ValueError("invalid declared high workload or bounded periodic inventory")
    return count


def validate_periodic_high(directory, manifest, high, native):
    # Import only during offline validation: the measurement runner imports the
    # handoff process owner, so a module-level import would create a cycle.
    from run_bert_colocation import inference_metrics, verify_progress
    from bert_colocation_worker import make_trace

    count = declared_high_requests(manifest)
    exports = [path for path in manifest["fingerprints"] if pathlib.Path(path).name == "export.json"]
    worker = str(EXAMPLE / "bert_colocation_worker.py")
    if (len(exports) != 1 or high.get("worker_sha256") != manifest["fingerprints"].get(worker)
            or high.get("assets") != native.get("assets") or high.get("warmup") != 1
            or high.get("engine") != "onnxruntime-cuda" or high.get("instrumented") is not False
            or high.get("providers", [None])[0] != "CUDAExecutionProvider"
            or high.get("export_sha256") != manifest["fingerprints"][exports[0]]
            or high.get("inputs") != json.loads(pathlib.Path(exports[0]).read_text())["inputs"]
            or high.get("validation") != {"checked_steps": count,
                                          "kind": "native PyTorch oracle, rtol=2e-4, atol=2e-5"}):
        raise ValueError("periodic high source, inputs, engine or complete oracle differs")
    schedule = json.loads((directory / "inference.start.json").read_text())
    if (schedule != manifest.get("high_schedule") or schedule.get("preparation_required") is not True
            or schedule.get("begin_ns") != manifest["high_release_ns"] + 100_000_000
            or schedule.get("drain_ns") != 10_000_000_000
            or schedule.get("trace") != make_trace(count, manifest["high_period_ms"] * 1_000_000)):
        raise ValueError("periodic high frozen schedule differs")
    verify_progress(directory, "inference", manifest["high_ready"], high)
    metrics = inference_metrics(high, schedule)
    if metrics["completed_requests"] != count or metrics["pending_requests"] != 0:
        raise ValueError("periodic high did not complete its full declared inventory")


def validate_colocation(directory, reference):
    manifest = json.loads((directory / "probe.json").read_text())
    if manifest.get("status") != "ok" or manifest.get("scheduler_drained") is not True:
        raise ValueError("co-location runner did not complete")
    for path, digest in manifest["fingerprints"].items():
        if sha256(path) != digest:
            raise ValueError("source, runtime or native reference identity changed")
    validate_scheduler_metrics((directory / "scheduler.metrics").read_text())
    high_requests = declared_high_requests(manifest)
    periodic = manifest.get("high_workload") == "periodic"
    folders = {"low": "low", "high": "inference" if periodic else "high"}
    results = {role: json.loads((directory / folder / "result.json").read_text()) for role, folder in folders.items()}
    for role, result in results.items():
        if result.get("status") != "ok" or sha256(directory / folders[role] / "result.json") != manifest[f"{role}_result_sha256"]:
            raise ValueError("missing or changed numerical result")
    low, high = results["low"], results["high"]
    native = json.loads((reference / "result.json").read_text())
    identity = ("role", "profile", "seed", "dtype", "engine", "attention", "warmup", "iterations",
                "application_sha256", "contract_sha256", "inputs", "assets", "environment", "parameter_count")
    if (native.get("status") != "ok" or any(low.get(key) != native.get(key) for key in identity)
            or low.get("parameter_count") != 108893186 or low.get("iterations") != 2 or low.get("warmup") != 1
            or low.get("numerical_comparison", {}).get("rtol") != 2e-4
            or low.get("numerical_comparison", {}).get("atol") != 2e-5
            or (not periodic and (high.get("providers", [None])[0] != "CUDAExecutionProvider"
                or high.get("validation") != {"reference": "native PyTorch logits", "rtol": 2e-4,
                                              "atol": 2e-5, "checked_requests": 2}))):
        raise ValueError("wrong numerical identity, workload or oracle coverage")
    if periodic:
        validate_periodic_high(directory, manifest, high, native)
    for path, result in ((reference, native), (directory / "low", low)):
        if sha256(path / "snapshot.pt") != result.get("snapshot_sha256"):
            raise ValueError("missing or changed training snapshot")
    import torch
    expected = torch.load(reference / "snapshot.pt", map_location="cpu", weights_only=True)
    actual = torch.load(directory / "low/snapshot.pt", map_location="cpu", weights_only=True)
    checks = low["numerical_comparison"]["checks"]
    if len(expected) != 401 or expected.keys() != actual.keys() or len(checks) != 401 or set(checks) != set(expected):
        raise ValueError("incomplete training tensor inventory")
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-4, atol=2e-5, msg=name)
    ready = {role: manifest[f"{role}_ready"] for role in ("low", "high")}
    if (not ready["low"]["device_uuid"] or ready["low"]["device_uuid"] != ready["high"]["device_uuid"]
            or any(results[role]["environment"]["device_uuid"] != ready[role]["device_uuid"] for role in ready)):
        raise ValueError("inconsistent CUDA device identities")
    low_text, high_text = ((directory / f"{role}.log").read_text() for role in ("low", "high"))
    if (low_text.count("persistent_lt_bias accepted=1") != 216
            or low_text.count("persistent_sgemm accepted=1") != 432):
        raise ValueError("missing fixed BERT forward/backward selection")
    if {event["pid"] for event in check_physical_audits(low_text)} != {ready["low"]["pid"]}:
        raise ValueError("audit PID does not match the low application")
    summary = dict(training_tensor_checks=401,
                   bitwise_equal=sum(torch.equal(actual[name], expected[name]) for name in expected),
                   inference_checked_requests=high_requests, scheduler_drained=True)
    return manifest, low_text, high_text, summary


def verify(directory, timeline, reference):
    manifest, low_text, high_text, summary = validate_colocation(directory, reference)
    boundary = manifest.get("high_timeline_boundary", "runtime-markers")
    high_requests = declared_high_requests(manifest)
    if boundary == "fixed-inventory" and manifest.get("high_workload", "original") == "original":
        high_result = json.loads((directory / "high/result.json").read_text())
        if (high_result.get("warmup") != 1 or high_result.get("iterations") != 2
                or high_result.get("instrumented") is not False):
            raise ValueError("fixed high inventory needs the unchanged unmarked one-warmup/two-request CLI")
    with closing(sqlite3.connect(timeline.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        low_gpu, high_gpu = read_timeline(database, manifest["low_ready"]["pid"], manifest["high_ready"]["pid"],
                                         boundary, high_requests)
    if len(high_gpu) != high_requests * HIGH_KERNELS_PER_REQUEST:
        raise ValueError("declared measured ONNX kernel inventory differs")
    proof = physical_handoff(low_text, high_text, low_gpu, high_gpu,
                             manifest["low_release_ns"], manifest["high_release_ns"])
    return dict(status="ok", diagnostic_only=True, performance_acceptance=False, physical_handoff=proof,
                high_timeline_boundary=boundary,
                high_workload=manifest.get("high_workload", "original"),
                **summary, timeline_sha256=sha256(timeline))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--timeline", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    report = {"status": "failed", "performance_acceptance": False}
    try:
        report = verify(args.run, args.timeline, args.reference)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    write_json(args.output, report)
    print(json.dumps(report))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
