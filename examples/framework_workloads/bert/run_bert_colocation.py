#!/usr/bin/env python3
"""Measure matched exclusive and co-located BERT controls without kernel traces."""

import argparse
import bisect
import contextlib
import csv
import json
import math
import os
import pathlib
import statistics
import subprocess
import tempfile
import time

from bert_contract import PROFILE, SEED, sha256, write_json
from bert_colocation_worker import make_trace, publish, schedule_spec, trace_digest
from run_bert_baseline import EXAMPLE, ROOT, SMOKE, environment, optimized_build
from run_bert_handoff import read_ready, stop_group
from run_persistent_mlp_handoff import wait_until
from measurements import validate_scheduler_metrics

MODES = ("native", "equal-persistent", "priority-persistent")
START_LEAD_NS = 5_000_000_000


def child_environment(mode, build, role, socket):
    if mode not in MODES or role not in PROFILE:
        raise ValueError("unknown mode or workload role")
    env = environment("native" if mode == "native" else "cooperative", build)
    if mode != "native":
        priority = "100" if mode == "priority-persistent" and role == "inference" else "10"
        env.update(GLIMMER_SCHEDULER_PRIORITY=priority,
                   GLIMMER_PERSISTENT_WORKER_BLOCKS="320",
                   GLIMMER_SCHEDULER_TENANT_ID=role, GLIMMER_SCHEDULER_CONTROL_SOCKET=str(socket))
    return env


def distribution(values):
    ordered = sorted(values)
    if not ordered or any(type(value) is not int or value < 0 for value in ordered):
        raise ValueError("invalid latency samples")
    return {"samples": len(ordered), "mean_ms": statistics.mean(ordered) / 1e6,
            "p50_ms": ordered[math.ceil(len(ordered) * .50) - 1] / 1e6,
            "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1] / 1e6,
            "p99_ms": ordered[math.ceil(len(ordered) * .99) - 1] / 1e6,
            "max_ms": ordered[-1] / 1e6, "p99_method": "nearest rank",
            "p99_sufficient_samples": len(ordered) >= 1000}


def validate_worker(result, role):
    if (result.get("status") != "ok" or result.get("role") != role
            or result.get("schema") not in {"bert_periodic_colocation_v1", "bert_bounded_arrival_v2",
                                            "bert_bounded_arrival_v3"}
            or result.get("instrumented") is not False or result.get("profile") != PROFILE[role]
            or result.get("dtype") != "float32" or result.get("seed") != SEED
            or not isinstance(result.get("records"), list)
            or (not result["records"] and (role != "inference" or result["schema"] == "bert_periodic_colocation_v1"))
            or result.get("validation", {}).get("checked_steps") != len(result["records"])):
        raise ValueError("incomplete or incompatible worker evidence")
    previous = result["measured_begin_ns"]
    if (type(previous) is not int or previous <= 0
            or type(result["measured_end_ns"]) is not int or not result["environment"].get("device_uuid")):
        raise ValueError("invalid worker interval or GPU identity")
    for record in result["records"]:
        start, end = record["start_ns"], record["end_ns"]
        if (type(start) is not int or type(end) is not int
                or not previous <= start < end <= result["measured_end_ns"]):
            raise ValueError("invalid or overlapping step interval")
        previous = end
    if result["schema"] == "bert_bounded_arrival_v3":
        worker_cpu_cost(result)


def worker_cpu_cost(result):
    cpu = result["cpu"]
    fields = ("observed_begin_ns", "observed_end_ns", "process_begin_ns", "process_end_ns")
    if (any(type(cpu.get(key)) is not int for key in fields)
            or not 0 < cpu["observed_begin_ns"] <= result["measured_begin_ns"]
            <= result["measured_end_ns"] <= cpu["observed_end_ns"]
            or cpu["observed_begin_ns"] == cpu["observed_end_ns"]
            or not 0 <= cpu["process_begin_ns"] <= cpu["process_end_ns"]):
        raise ValueError("invalid worker CPU counters or observation interval")
    seconds = (cpu["process_end_ns"] - cpu["process_begin_ns"]) / 1e9
    wall_seconds = (cpu["observed_end_ns"] - cpu["observed_begin_ns"]) / 1e9
    return {"process_cpu_seconds": seconds, "observation_seconds": wall_seconds,
            "mean_logical_cpus": seconds / wall_seconds,
            "scope": "worker measurement routine including lead-in and drain; not isolated framework CPU"}


def process_cpu_snapshot(pid):
    text = pathlib.Path(f"/proc/{pid}/stat").read_text()
    identity, fields = text.rsplit(") ", 1)
    values = fields.split()
    ticks = os.sysconf("SC_CLK_TCK")
    if int(identity.split(" (", 1)[0]) != pid or type(ticks) is not int or ticks <= 0:
        raise ValueError("invalid service CPU identity or clock rate")
    return {"pid": pid, "start_ticks": int(values[19]), "user_ticks": int(values[11]),
            "system_ticks": int(values[12]), "ticks_per_second": ticks,
            "observed_ns": time.monotonic_ns()}


def service_cpu_cost(before, after):
    fields = ("pid", "start_ticks", "user_ticks", "system_ticks", "ticks_per_second", "observed_ns")
    if (any(type(row.get(key)) is not int or row[key] < 0 for row in (before, after) for key in fields)
            or any(before[key] != after[key] for key in ("pid", "start_ticks", "ticks_per_second"))
            or not before["pid"] or not before["start_ticks"] or not before["ticks_per_second"]
            or not 0 < before["observed_ns"] < after["observed_ns"]
            or any(before[key] > after[key] for key in ("user_ticks", "system_ticks"))):
        raise ValueError("service CPU counter or process identity changed")
    ticks = before["ticks_per_second"]
    user = (after["user_ticks"] - before["user_ticks"]) / ticks
    system = (after["system_ticks"] - before["system_ticks"]) / ticks
    wall = (after["observed_ns"] - before["observed_ns"]) / 1e9
    return {"user_seconds": user, "system_seconds": system, "process_cpu_seconds": user + system,
            "observation_seconds": wall, "mean_logical_cpus": (user + system) / wall,
            "scope": "service from prepared workers through their exit; includes lead-in, drain and validation"}


def request_slo(inference, schedule):
    if "slo_ns" not in schedule:
        return None
    threshold = schedule["slo_ns"]
    if type(threshold) is not int or threshold <= 0:
        raise ValueError("SLO threshold must be a positive integer nanosecond value")
    records = inference["records"]
    begin = schedule["begin_ns"]
    offsets = (schedule["trace"]["offsets_ns"] if "trace" in schedule
               else [index * schedule["period_ns"] for index in range(schedule["requests"])])
    met = sum(row["end_ns"] - row["arrival_ns"] <= threshold for row in records)
    expired = sum(inference["measured_end_ns"] - (begin + offset) > threshold
                  for offset in offsets[len(records):])
    unresolved = len(offsets) - len(records) - expired
    return {"threshold_ms": threshold / 1e6, "offered_requests": len(offsets), "met": met,
            "completed_missed": len(records) - met, "pending_expired": expired,
            "pending_unresolved": unresolved, "all_offered_resolved": unresolved == 0,
            "attainment_fraction_lower_bound": met / len(offsets),
            "attainment_fraction_upper_bound": (met + unresolved) / len(offsets),
            "scope": "queue-inclusive responses over every offered request; pending responses remain censored"}


def analyze(training, inference, schedule):
    validate_worker(training, "training")
    metrics = inference_metrics(inference, schedule)
    if (training["environment"] != inference["environment"]
            or not training["environment"].get("device_uuid")
            or training["assets"] != inference["assets"]
            or training["worker_sha256"] != inference["worker_sha256"]
            or training["warmup"] != inference["warmup"]):
        raise ValueError("worker identities differ")
    begin, end = metrics["window_begin_ns"], metrics["window_end_ns"]
    if (training["records"][0]["start_ns"] > begin
            or training["records"][-1]["end_ns"] < metrics.get("observation_end_ns", end)):
        raise ValueError("training does not cover the entire inference window")
    return {**metrics, **training_metrics(training, begin, end)}


def inference_metrics(inference, schedule):
    validate_worker(inference, "inference")
    if "trace" in schedule:
        return {**bounded_inference_metrics(inference, schedule), "slo": request_slo(inference, schedule)}
    begin, period, count = (schedule[key] for key in ("begin_ns", "period_ns", "requests"))
    if any(type(value) is not int or value <= 0 for value in (begin, period, count)):
        raise ValueError("invalid schedule")
    requests = inference["records"]
    if len(requests) != count or inference["measured_begin_ns"] != begin:
        raise ValueError("missing requests or mismatched epoch")
    for index, record in enumerate(requests):
        if record.get("arrival_ns") != begin + index * period or record["start_ns"] < record["arrival_ns"]:
            raise ValueError("arrival trace shifted, dropped or executed early")
    end = max(begin + count * period, requests[-1]["end_ns"])
    if inference["measured_end_ns"] != end:
        raise ValueError("inference interval does not match offered window and drain")
    seconds = (end - begin) / 1e9
    return {"window_begin_ns": begin, "window_end_ns": end, "window_seconds": seconds,
            "arrival_period_ms": period / 1e6, "offered_requests_per_second": 1e9 / period,
            "response": distribution([row["end_ns"] - row["arrival_ns"] for row in requests]),
            "service": distribution([row["end_ns"] - row["start_ns"] for row in requests]),
            "queue_and_wakeup": distribution([row["start_ns"] - row["arrival_ns"] for row in requests]),
            "offered_requests": count, "completed_requests": count, "pending_requests": 0,
            "drain_seconds": max(0, end - (begin + count * period)) / 1e9,
            "inference_completed_requests_per_second": count / seconds,
            "slo": request_slo(inference, schedule)}


def request_snapshot(records, inflight, begin, trace, timestamp):
    offsets = trace["offsets_ns"]
    offered = bisect.bisect_right(offsets, timestamp - begin)
    completed = sum(row["end_ns"] <= timestamp for row in records)
    started = sum(row["start_ns"] <= timestamp for row in records)
    started += int(inflight is not None and inflight["start_ns"] <= timestamp)
    if not 0 <= completed <= started <= offered or started - completed > 1:
        raise ValueError("inconsistent request snapshot")
    bounds = [(records[index]["end_ns"] if index < len(records) and records[index]["end_ns"] <= timestamp
               else timestamp) - (begin + offset) for index, offset in enumerate(offsets[:offered])]
    return {"planned_requests": len(offsets), "offered_requests": offered,
            "completed_requests": completed, "pending_requests": offered - completed,
            "queued_requests": offered - started, "inflight_requests": started - completed,
            "not_yet_offered_requests": len(offsets) - offered,
            "response_lower_bound": distribution(bounds) if bounds else None}


def validate_request_prefix(records, inflight, begin, trace, observed):
    previous = begin
    for index, row in enumerate([*records, *([inflight] if inflight else [])]):
        if index >= len(trace["offsets_ns"]):
            raise ValueError("too many requests")
        arrival = begin + trace["offsets_ns"][index]
        if (row.get("arrival_ns") != arrival or type(row.get("start_ns")) is not int
                or not max(arrival, previous) <= row["start_ns"] <= observed):
            raise ValueError("invalid request sequence or shifted arrival")
        if index < len(records):
            if type(row.get("end_ns")) is not int or not row["start_ns"] < row["end_ns"] <= observed:
                raise ValueError("invalid completed request")
            previous = row["end_ns"]


def bounded_inference_metrics(inference, schedule):
    begin, trace, deadline = schedule_spec(schedule)
    records, observed = inference["records"], inference["measured_end_ns"]
    offered_end = begin + trace["duration_ns"]
    if (deadline is None or inference["measured_begin_ns"] != begin
            or observed < offered_end or (len(records) < len(trace["offsets_ns"]) and observed < deadline)):
        raise ValueError("inference ended before offered window or drain deadline")
    validate_request_prefix(records, None, begin, trace, observed)
    if any(row["start_ns"] >= deadline for row in records):
        raise ValueError("request started after the drain cutoff")
    expected_end = max(offered_end, records[-1]["end_ns"] if records else begin)
    if len(records) == len(trace["offsets_ns"]) and observed != expected_end:
        raise ValueError("complete inference interval differs")
    current = request_snapshot(records, None, begin, trace, observed)
    at_offer_end = request_snapshot(records, None, begin, trace, offered_end)
    at_deadline = request_snapshot(records, None, begin, trace, deadline) if observed >= deadline else None
    responses = distribution([row["end_ns"] - row["arrival_ns"] for row in records]) if records else None
    return {"window_begin_ns": begin, "window_end_ns": offered_end, "window_seconds": trace["duration_ns"] / 1e9,
            "observation_end_ns": observed, "observation_seconds": (observed - begin) / 1e9,
            "arrival_period_ms": trace["period_ns"] / 1e6, "burst_size": trace["burst_size"],
            "trace_sha256": trace_digest(trace), "offered_requests_per_second": 1e9 / trace["period_ns"],
            **current, "response": responses if current["pending_requests"] == 0 else None,
            "completed_response": responses,
            "service": distribution([row["end_ns"] - row["start_ns"] for row in records]) if records else None,
            "queue_and_wakeup": distribution([row["start_ns"] - row["arrival_ns"] for row in records]) if records else None,
            "at_offer_end": at_offer_end, "at_drain_deadline": at_deadline,
            "drain_seconds": (observed - offered_end) / 1e9,
            "workload_completed": current["pending_requests"] == 0,
            "deadline_exceeded": at_deadline is not None and at_deadline["pending_requests"] > 0,
            "inference_completed_requests_per_second": len(records) * 1e9 / (observed - begin)}


def read_progress(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    truncated = bool(lines and not lines[-1].endswith("\n"))
    if truncated:
        lines.pop()
    events = [json.loads(line) for line in lines]
    if not events or not isinstance(events[0], dict):
        raise ValueError("missing progress header")
    header = events[0]
    if (header.get("phase") != "begin" or header.get("schema") != "bert_progress_v1"
            or header.get("role") not in PROFILE or type(header.get("pid")) is not int or header["pid"] <= 0
            or type(header.get("begin_ns")) is not int or header["begin_ns"] <= 0):
        raise ValueError("invalid progress header")
    records, inflight, stopped = [], None, None
    previous = header["begin_ns"]
    for event in events[1:]:
        if not isinstance(event, dict) or stopped is not None:
            raise ValueError("invalid progress event after stop")
        phase = event.get("phase")
        fields = {"start_ns", "arrival_ns"} if header["role"] == "inference" else {"start_ns"}
        if phase in {"start", "complete"}:
            expected = fields | {"phase", "index"} | ({"end_ns"} if phase == "complete" else set())
            if (event.keys() != expected or type(event["index"]) is not int
                    or event["index"] != len(records) or type(event["start_ns"]) is not int):
                raise ValueError("invalid progress inventory or index")
        if phase == "start":
            if (inflight is not None or event["start_ns"] < previous
                    or (header["role"] == "inference" and
                        (type(event["arrival_ns"]) is not int or not header["begin_ns"] <= event["arrival_ns"] <= event["start_ns"]))):
                raise ValueError("overlapping or early progress start")
            inflight = {key: event[key] for key in fields}
        elif phase == "complete":
            if (inflight is None or any(event[key] != value for key, value in inflight.items())
                    or type(event["end_ns"]) is not int or event["end_ns"] <= event["start_ns"]):
                raise ValueError("unmatched progress completion")
            records.append({**inflight, "end_ns": event["end_ns"]})
            previous, inflight = event["end_ns"], None
        elif phase == "stop":
            if (event.keys() != {"phase", "end_ns", "completed"} or inflight is not None
                    or type(event["completed"]) is not int or event["completed"] != len(records)
                    or type(event["end_ns"]) is not int or event["end_ns"] < previous):
                raise ValueError("invalid progress stop")
            stopped = event["end_ns"]
        else:
            raise ValueError("unknown progress phase")
    return {"header": header, "records": records, "inflight": inflight,
            "stopped_ns": stopped, "truncated_tail": truncated}


def verify_progress(output, role, ready, result=None):
    folder = output / role
    progress = read_progress(folder / "progress.jsonl")
    header = progress["header"]
    setup = json.loads((folder / "setup.json").read_text())
    schedule = json.loads((output / f"{role}.start.json").read_text())
    if role == "inference" and schedule.get("preparation_required") is True:
        prepared = header.get("prepared_ns")
        if (type(prepared) is not int or not 0 < prepared < schedule["begin_ns"]
                or header["begin_ns"] != schedule["begin_ns"]):
            raise ValueError("arrival trace was not prepared before the epoch")
    if (header["pid"] != ready["pid"] or header["role"] != role
            or header.get("schedule_sha256") != sha256(output / f"{role}.start.json")
            or header.get("setup_sha256") != sha256(folder / "setup.json")
            or setup.get("pid") != ready["pid"] or setup.get("status") != "prepared"
            or setup.get("role") != role or setup.get("environment", {}).get("device_uuid") != ready["device_uuid"]):
        raise ValueError("progress identity or schedule binding differs")
    if result is not None and (progress["truncated_tail"] or progress["inflight"] is not None
            or progress["records"] != result["records"] or progress["stopped_ns"] != result["measured_end_ns"]
            or header["begin_ns"] != result["measured_begin_ns"]
            or any(result.get(key) != value for key, value in setup.items() if key not in {"status", "pid"})
            or result.get("journal_sha256") != sha256(folder / "progress.jsonl")):
        raise ValueError("successful result and progress journal differ")
    return progress


def preserve_partial(output, report):
    """Retain authenticated progress after cleanup; never promote it to validated timing."""
    if "schedule" not in report:
        return
    observed = time.monotonic_ns()
    partial = {"status": "failed", "numerically_validated": False, "observed_ns": observed,
               "error": report["error"], "schedule": report["schedule"], "roles": {}}
    for role in PROFILE:
        if f"{role}_ready" not in report:
            continue
        try:
            progress = verify_progress(output, role, report[f"{role}_ready"])
            item = {"journal_sha256": sha256(output / role / "progress.jsonl"),
                    "completed_steps": len(progress["records"]), "inflight": progress["inflight"],
                    "truncated_tail": progress["truncated_tail"]}
            if role == "inference":
                begin, trace, deadline = schedule_spec(report["schedule"])
                if progress["header"]["begin_ns"] != begin:
                    raise ValueError("progress epoch differs from the recorded schedule")
                validate_request_prefix(progress["records"], progress["inflight"], begin, trace, observed)
                item["at_observation"] = request_snapshot(progress["records"], progress["inflight"], begin, trace, observed)
                item["at_drain_deadline"] = (request_snapshot(progress["records"], progress["inflight"], begin, trace, deadline)
                    if deadline is not None and deadline <= observed else None)
                # A kill between GPU completion and journal publication leaves
                # physical completion unknown. Do not infer latency lower bounds
                # from absence of a committed completion record in a failed run.
                item["count_scope"] = "journal-confirmed completions; unrecorded physical completion may be unknown"
                for snapshot in (item["at_observation"], item["at_drain_deadline"]):
                    if snapshot is not None:
                        snapshot["response_lower_bound"] = None
            partial["roles"][role] = item
        except Exception as error:
            partial["roles"][role] = {"progress_unavailable": f"{type(error).__name__}: {error}"}
    write_json(output / "partial.json", partial)
    report["partial_sha256"] = sha256(output / "partial.json")


def training_metrics(training, begin, end):
    if type(begin) is not int or type(end) is not int or not 0 < begin < end:
        raise ValueError("invalid training analysis window")
    contained = [row for row in training["records"] if begin <= row["start_ns"] and row["end_ns"] <= end]
    completions = sum(begin < row["end_ns"] <= end for row in training["records"])
    seconds = (end - begin) / 1e9
    return {"window_begin_ns": begin, "window_end_ns": end, "window_seconds": seconds,
            "training_steps": len(contained), "training_steps_per_second": len(contained) / seconds,
            "training_samples_per_second": len(contained) * PROFILE["training"]["batch_size"] / seconds,
            "training_excluded_steps": len(training["records"]) - len(contained),
            "training_total_steps": len(training["records"]),
            "training_completions_in_window": completions,
            "training_sufficient_steps": len(contained) >= 10,
            "training_step_service": distribution([row["end_ns"] - row["start_ns"]
                                                     for row in training["records"]])}


def analyze_solo(result, role, schedule):
    if role == "inference":
        return inference_metrics(result, schedule)
    if role != "training":
        raise ValueError("unknown solo role")
    validate_worker(result, role)
    begin, end = schedule["begin_ns"], schedule["end_ns"]
    if type(begin) is not int or type(end) is not int or not 0 < begin < end:
        raise ValueError("invalid solo training schedule")
    actual_begin = result["records"][0]["start_ns"]
    if (actual_begin < begin or end - actual_begin < .95 * (end - begin)
            or result["records"][-1]["end_ns"] < end):
        raise ValueError("solo training does not cover the declared window")
    return training_metrics(result, actual_begin, end)


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=10)
    return [{"gpu_uuid": uuid.strip(), "pid": int(pid.strip()), "name": name.strip()}
            for uuid, pid, name in csv.reader(result.stdout.splitlines())]


def check_gpu(allowed, device_uuid=None):
    processes = gpu_processes()
    # PyTorch reports a bare UUID; NVML/nvidia-smi prefix it with GPU-.
    if any(row["pid"] not in allowed and (device_uuid is None
           or row["gpu_uuid"].removeprefix("GPU-") == device_uuid.removeprefix("GPU-"))
           for row in processes):
        raise RuntimeError(f"foreign GPU processes observed; do not interfere: {processes}")
    return {"time_ns": time.monotonic_ns(), "processes": processes}


def collect(args, mode, output):
    output.mkdir(parents=True, exist_ok=False)
    role_selection = getattr(args, "role", "both")
    roles = tuple(PROFILE) if role_selection == "both" else (role_selection,)
    report = {"status": "failed", "mode": mode, "role": role_selection,
              "commands": [], "gpu_observations": []}
    try:
        report["gpu_observations"].append(check_gpu(set()))
        with contextlib.ExitStack() as resources:
            temporary = resources.enter_context(tempfile.TemporaryDirectory(prefix="glimmer-bert-timing-"))
            socket = pathlib.Path(temporary) / "control.sock"

            def spawn(name, command, env):
                stream = resources.enter_context((output / f"{name}.log").open("w", encoding="utf-8"))
                report["commands"].append({"role": name, "argv": command, "environment": env})
                write_json(output / "trial.json", report)
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                resources.callback(stop_group, process)
                return process

            base = environment("native", args.build)
            services = []
            if mode != "native":
                services.append(spawn("service", [str(args.build / "bin/glimmer_control_service"),
                    "--socket", str(socket), "--quota-bytes", "12884901888", "--execution-mode", "remote",
                    "--lease-timeout-ms", "5000", "--max-queued-tasks", "256", "--max-concurrent-tasks", "1",
                    "--scheduler-policy", "priority", "--priority-reserved-slots", "0",
                    "--priority-threshold", "50", "--bind-leases-to-process"], base))
                wait_until(socket.exists, services, timeout=30)
            workers = {}
            for role in roles:
                command = [str(args.python), "-B", str(EXAMPLE / "bert_colocation_worker.py"),
                    "--assets", str(args.assets), "--export", str(args.export), "--role", role,
                    "--output", str(output / role), "--warmup", str(args.warmup),
                    "--timeout", str(args.timeout), "--ready-file", str(output / f"{role}.ready.json"),
                    "--start-file", str(output / f"{role}.start.json"),
                    "--stop-file", str(output / "training.stop"), "--active-file", str(output / "training.active.json")]
                workers[role] = spawn(role, command, child_environment(mode, args.build, role, socket))
                report[f"{role}_ready"] = read_ready(output / f"{role}.ready.json",
                    [*services, *workers.values()], workers[role].pid, timeout=240)
            uuid = report[f"{roles[0]}_ready"]["device_uuid"]
            if any(uuid != report[f"{role}_ready"]["device_uuid"] for role in roles):
                raise ValueError("workers must share one GPU")
            allowed = {process.pid for process in workers.values()}
            report["gpu_observations"].append(check_gpu(allowed, uuid))
            if services:
                report["service_cpu_before"] = process_cpu_snapshot(services[0].pid)
            if role_selection == "both":
                publish(output / "training.start.json", {})
                wait_until((output / "training.active.json").exists, [*services, *workers.values()], timeout=30)
            trace = None
            if role_selection != "training":
                trace = make_trace(args.requests, args.period_ms * 1_000_000, getattr(args, "burst_size", 1))
                digest = trace_digest(trace)
            schedule = {"begin_ns": time.monotonic_ns() + START_LEAD_NS}
            if role_selection == "training":
                schedule["end_ns"] = schedule["begin_ns"] + args.training_seconds * 1_000_000_000
                publish(output / "training.start.json", schedule)
            else:
                schedule.update(trace=trace, trace_sha256=digest, preparation_required=True,
                                drain_ns=getattr(args, "drain_seconds", 60) * 1_000_000_000)
                if getattr(args, "slo_ms", None) is not None:
                    schedule["slo_ns"] = args.slo_ms * 1_000_000
                publish(output / "inference.start.json", schedule)
            report["schedule"] = schedule
            write_json(output / "trial.json", report)
            deadline, next_check = time.monotonic() + args.timeout, 0

            def pending():
                return (time.monotonic_ns() < schedule["end_ns"] if role_selection == "training"
                        else workers["inference"].poll() is None)

            while pending():
                if (("training" in workers and workers["training"].poll() is not None)
                        or any(p.poll() is not None for p in services)):
                    raise RuntimeError("training/service exited during measurement")
                if time.monotonic() >= deadline:
                    raise TimeoutError("co-location timed out")
                if role_selection != "training":
                    request_deadline = schedule["begin_ns"] + schedule["trace"]["duration_ns"] + schedule["drain_ns"]
                    if time.monotonic_ns() >= request_deadline + getattr(args, "completion_grace_seconds", 120) * 1_000_000_000:
                        raise TimeoutError("in-flight inference exceeded completion grace")
                if time.monotonic() >= next_check:
                    report["gpu_observations"].append(check_gpu(allowed, uuid))
                    next_check = time.monotonic() + 5
                time.sleep(.05)
            if "inference" in workers and workers["inference"].returncode != 0:
                raise RuntimeError("inference failed")
            if "training" in workers:
                (output / "training.stop").touch(exist_ok=False)
                drain = min(getattr(args, "completion_grace_seconds", 120), max(1, deadline - time.monotonic()))
                if workers["training"].wait(timeout=drain) != 0:
                    raise RuntimeError("training failed")
            if any(p.poll() is not None for p in services):
                raise RuntimeError("service exited")
            results = {role: json.loads((output / role / "result.json").read_text()) for role in roles}
            for role, result in results.items():
                verify_progress(output, role, report[f"{role}_ready"], result)
            report["metrics"] = (analyze(results["training"], results["inference"], schedule)
                                 if role_selection == "both" else analyze_solo(results[role_selection], role_selection, schedule))
            report["identity"] = {role: {key: result.get(key) for key in
                ("profile", "seed", "dtype", "warmup", "assets", "inputs", "environment", "worker_sha256",
                 "engine", "attention", "parameter_count", "onnxruntime", "providers", "provider_options", "export_sha256")}
                for role, result in results.items()}
            report["result_sha256"] = {role: sha256(output / role / "result.json") for role in roles}
            report["cpu_cost"] = {"workers": {role: worker_cpu_cost(result) if "cpu" in result else None
                                              for role, result in results.items()}, "service": None}
            report["completion"] = "bounded-incomplete" if report["metrics"].get("pending_requests", 0) else "complete"
            if services:
                report["service_cpu_after"] = process_cpu_snapshot(services[0].pid)
                report["cpu_cost"]["service"] = service_cpu_cost(report["service_cpu_before"], report["service_cpu_after"])
                command = [str(args.build / "bin/glimmer_control_client"), "--socket", str(socket), "metrics"]
                metrics = subprocess.run(command, env=base, capture_output=True, text=True, check=True, timeout=10)
                (output / "scheduler.metrics").write_text(metrics.stdout, encoding="utf-8")
                validate_scheduler_metrics(metrics.stdout)
                report["scheduler_drained"] = True
            report["gpu_observations"].append(check_gpu(set(), uuid))
        report["status"] = "ok"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        try:
            preserve_partial(output, report)
        except Exception as partial_error:
            report["partial_error"] = f"{type(partial_error).__name__}: {partial_error}"
    finally:
        write_json(output / "trial.json", report)
    if report["status"] != "ok":
        raise RuntimeError(report["error"])
    return report


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "schema": "bert_controlled_measurement_v4", "trials": [],
              "performance_acceptance": False, "preemption_acceptance": False,
              "full_card_acceptance": False,
              "diagnostic_only": getattr(args, "diagnostic_only", False),
              "start_lead_ns": START_LEAD_NS,
              "scope": "bounded periodic/burst inference and continuous training; request/step journal only",
              "configuration": {key: getattr(args, key) for key in
                                ("role", "modes", "requests", "period_ms", "burst_size", "drain_seconds",
                                 "completion_grace_seconds", "timeout", "training_seconds", "warmup", "repetitions")}}
    try:
        report["configuration"]["slo_ms"] = getattr(args, "slo_ms", None)
        if not getattr(args, "diagnostic_only", False):
            raise ValueError("BERT full-device qualification is pending; use --diagnostic-only "
                             "only for unqualified measurements, not a full-card comparison")
        optimized_build(args.build)
        paths = [*EXAMPLE.glob("*.py"),
                 *(SMOKE / name for name in ("measurements.py", "run_persistent_mlp_handoff.py",
                                            "run_overhead_baseline.py", "run_slo_validation.py")),
                 args.build / "lib/libglimmer_cuda_interceptor.so",
                 args.build / "bin/glimmer_control_service", args.build / "bin/glimmer_control_client",
                 args.export / "export.json"]
        report["fingerprints"] = {str(path): sha256(path) for path in paths}
        if args.role != "training":
            trace = make_trace(args.requests, args.period_ms * 1_000_000, args.burst_size)
            write_json(args.output / "arrival-trace.json", trace)
            report["arrival_trace_sha256"] = trace_digest(trace)
            report["arrival_trace_file_sha256"] = sha256(args.output / "arrival-trace.json")
        device = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,pstate,temperature.gpu",
                                 "--format=csv"], capture_output=True, text=True, check=True, timeout=10)
        report["gpu_before"] = device.stdout
        for repetition in range(1, args.repetitions + 1):
            offset = (repetition - 1) % len(args.modes)
            modes = args.modes[offset:] + args.modes[:offset]
            for mode in modes:
                name = f"{repetition}-{mode}"
                print(f"Starting {name}", flush=True)
                trial = collect(args, mode, args.output / name)
                if args.role != "training" and trial["metrics"].get("trace_sha256") != report["arrival_trace_sha256"]:
                    raise ValueError("trial did not replay the frozen arrival trace")
                if report["trials"] and trial["identity"] != report["identity"]:
                    raise ValueError("trial workload/environment identity changed")
                report["identity"] = trial["identity"]
                report["trials"].append({"name": name, "mode": mode, "repetition": repetition,
                    "completion": trial["completion"], "metrics": trial["metrics"],
                    "cpu_cost": trial["cpu_cost"],
                    "trial_sha256": sha256(args.output / name / "trial.json")})
                write_json(args.output / "comparison.json", report)
                print(json.dumps({"trial": name, "metrics": trial["metrics"]}), flush=True)
        if any(sha256(path) != digest for path, digest in report["fingerprints"].items()):
            raise ValueError("source/runtime changed during experiment")
        if args.role != "training" and sha256(args.output / "arrival-trace.json") != report["arrival_trace_file_sha256"]:
            raise ValueError("frozen arrival trace changed during experiment")
        report["status"] = "ok"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(args.output / "comparison.json", report)
    print(json.dumps({"status": report["status"], "error": report.get("error"), "output": str(args.output)}), flush=True)
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "export", "output"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--python", type=pathlib.Path, default=SMOKE / ".venv/bin/python")
    parser.add_argument("--build", type=pathlib.Path, default=ROOT / "build/cuda-perf")
    parser.add_argument("--role", choices=("both", *PROFILE), default="both")
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--training-seconds", type=int, default=100, help="window for training-only trials")
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--period-ms", type=int, default=100)
    parser.add_argument("--slo-ms", type=int, help="freeze a queue-inclusive response threshold before the run")
    parser.add_argument("--burst-size", type=int, default=1, help="simultaneous requests, preserving average arrival rate")
    parser.add_argument("--drain-seconds", type=int, default=60, help="stop starting inference after offered window plus this allowance")
    parser.add_argument("--completion-grace-seconds", type=int, default=120, help="watchdog allowance for an in-flight request or final training step")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="allow unqualified BERT measurements; full-card acceptance remains false")
    args = parser.parse_args()
    if args.modes is None:
        args.modes = list(MODES) if args.role == "both" else ["native", "priority-persistent"]
    duration_ms = args.training_seconds * 1000 if args.role == "training" else args.requests * args.period_ms
    reserve = START_LEAD_NS / 1e9 + (args.completion_grace_seconds if args.role == "training"
                                   else args.drain_seconds + 2 * args.completion_grace_seconds)
    if (min(args.requests, args.period_ms, args.warmup, args.repetitions, args.timeout,
            args.training_seconds, args.burst_size, args.completion_grace_seconds) <= 0
            or args.drain_seconds < 0 or args.requests > 1_000_000 or args.burst_size > args.requests
            or (args.slo_ms is not None and args.slo_ms <= 0)
            or duration_ms >= (args.timeout - reserve) * 1000 or len(set(args.modes)) != len(args.modes)):
        parser.error("invalid counts, duplicate modes, or timeout cannot cover offered window, drain and completion grace")
    for name in ("assets", "export", "output", "build"):
        setattr(args, name, getattr(args, name).resolve())
    args.python = args.python.absolute()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
