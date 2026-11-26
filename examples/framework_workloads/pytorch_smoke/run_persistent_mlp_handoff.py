#!/usr/bin/env python3
"""Prove an unchanged MLP's audited persistent-kernel priority handoff.

This diagnostic is deliberately separate from performance measurement. It uses
the application's existing ready/start files, never a private application API.
"""

import argparse
import contextlib
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import time

from measurements import read_metadata, validate_scheduler_metrics


ROOT = pathlib.Path(__file__).resolve().parents[3]
APPLICATION = pathlib.Path(__file__).with_name("pytorch_smoke.py")
NUMERICAL = APPLICATION.with_name("pytorch_mlp_correctness.py")
AUDIT_PREFIX = "[glimmer] persistent_audit "
CONTROL_PREFIX = "[glimmer] cooperative control timing "


def records(text, prefix, allow_partial=False):
    result = []
    if allow_partial and not text.endswith("\n"):
        text = text.rpartition("\n")[0]
    for line in text.splitlines():
        if prefix not in line:
            continue
        if not line.startswith(prefix):
            raise ValueError("interleaved diagnostic record")
        fields = {}
        for item in line[len(prefix):].split():
            key, value = item.split("=", 1)
            if key in fields:
                raise ValueError("duplicate diagnostic field")
            fields[key] = value if key == "phase" else int(value)
        required = ({"phase", "pid", "task", "generation", "launch", "logical", "completed", "exact", "time_ns"}
                    if prefix == AUDIT_PREFIX else
                    {"task", "generation", "submit", "status", "enqueued_ns", "started_ns", "finished_ns", "observed_ns"})
        if fields.keys() != required:
            raise ValueError("incomplete or unknown diagnostic fields")
        result.append(fields)
    return result


def check_physical_audits(text):
    """Validate complete owned-GEMM progress independently of application output."""
    events = records(text, AUDIT_PREFIX)
    if not events or len({event["pid"] for event in events}) != 1:
        raise ValueError("missing or mixed-process persistent audit")
    tasks = {}
    for event in events:
        if (event["pid"] <= 0 or event["task"] <= 0 or event["generation"] <= 0 or event["time_ns"] <= 0
                or event["logical"] <= 0 or event["exact"] not in {0, 1}
                or not 0 <= event["completed"] <= event["logical"]):
            raise ValueError("invalid persistent audit identity or progress")
        if event["phase"] not in {"grant", "prepare", "first_tile", "first_compute", "retreat",
                                   "priority_retreat", "suspended", "complete"}:
            raise ValueError("unknown persistent audit phase")
        if event["phase"] in {"suspended", "complete"} and event["exact"] != 1:
            raise ValueError("logical tile audit failed")
        tasks.setdefault(event["task"], []).append(event)
    for task_events in tasks.values():
        generation, launch, completed, lower_bound, last_time = 0, 0, 0, 0, 0
        state = "new"
        has_compute_witness = False
        for event in task_events:
            if event["time_ns"] < last_time:
                raise ValueError("audit time moved backwards")
            last_time = event["time_ns"]
            phase = event["phase"]
            if phase == "grant":
                if state not in {"new", "suspended"} or event["generation"] == generation:
                    raise ValueError("invalid or reused execution grant")
                generation = event["generation"]
                state = "granted"
            elif phase == "prepare":
                if state != "granted" or event["launch"] != launch + 1 or event["generation"] != generation:
                    raise ValueError("physical launch without its fresh grant")
                launch = event["launch"]
                state = "running"
                has_compute_witness = False
            else:
                if state != "running" or event["generation"] != generation or event["launch"] != launch:
                    raise ValueError("observation without its matching physical launch")
                if phase == "first_compute":
                    if has_compute_witness or event["completed"] != completed or event["exact"] != 0:
                        raise ValueError("invalid or duplicate compute observation")
                    has_compute_witness = True
                elif phase == "first_tile":
                    if event["completed"] != completed + 1:
                        raise ValueError("invalid first-tile progress lower bound")
                    lower_bound = max(lower_bound, completed + 1)
                elif phase in {"suspended", "complete"}:
                    if event["completed"] < max(completed, lower_bound):
                        raise ValueError("physical progress was lost across a stop")
                    completed = event["completed"]
                    state = phase
                    if phase == "suspended" and completed >= event["logical"]:
                        raise ValueError("completed work mislabeled as suspended")
        completions = [event for event in task_events if event["phase"] == "complete"]
        if (len(completions) != 1 or completions[0]["completed"] != completions[0]["logical"]
                or task_events[-1] != completions[0]
                or len({event["logical"] for event in task_events}) != 1):
            raise ValueError("incomplete, duplicated or inconsistent logical completion")
    accepted = text.count("persistent_sgemm accepted=1") + text.count("persistent_lt_bias accepted=1")
    if accepted != len(tasks):
        raise ValueError("accepted GEMM and audited logical task inventories differ")
    return events


def check_audits(text, numerical=False):
    events = check_physical_audits(text)
    validation = ("MLP numerical checks passed: 22; native_compared=True" if numerical
                  else "status=ok validation=1")
    if (validation not in text.splitlines() if numerical else validation not in text):
        raise ValueError("application or numerical companion did not validate")
    return events


def verify_handoff(low_text, high_text, low_release_ns, high_release_ns, numerical=False):
    low = check_audits(low_text, numerical=numerical)
    high = check_audits(high_text)
    submissions = [event for event in records(high_text, CONTROL_PREFIX)
                   if event["submit"] == 1 and event["status"] == 0
                   and high_release_ns <= event["enqueued_ns"] <= event["started_ns"]
                   <= event["finished_ns"] <= event["observed_ns"]]
    for first in low:
        if first["phase"] != "first_tile" or not low_release_ns <= first["time_ns"] < high_release_ns:
            continue
        same_launch = [event for event in low if event["task"] == first["task"]
                       and event["generation"] == first["generation"]
                       and event["launch"] == first["launch"]]
        for submit in submissions:
            # The entire control round trip must be after observed low execution
            # and before the low stop. This bounds actual high arrival, rather
            # than calling a start-file write a scheduler arrival.
            if not first["time_ns"] < submit["enqueued_ns"]:
                continue
            stops = [event for event in same_launch if event["phase"] == "suspended"
                     and 0 < event["completed"] < event["logical"]
                     and event["time_ns"] > submit["finished_ns"]]
            for stop in stops:
                retreats = [event for event in same_launch if event["phase"] == "priority_retreat"
                            and submit["started_ns"] <= event["time_ns"] <= stop["time_ns"]]
                if not retreats:
                    continue
                high_starts = [event for event in high if event["task"] == submit["task"]
                               and event["phase"] == "first_tile" and event["time_ns"] > stop["time_ns"]]
                high_ends = [event for event in high if event["task"] == submit["task"]
                             and event["phase"] == "complete" and event["time_ns"] > stop["time_ns"]]
                if not high_starts or not high_ends or high_starts[0]["time_ns"] > high_ends[0]["time_ns"]:
                    continue
                high_prepares = [event for event in high if event["task"] == submit["task"]
                                 and event["phase"] == "prepare"]
                if not high_prepares or min(event["time_ns"] for event in high_prepares) <= stop["time_ns"]:
                    continue
                resumed = [event for event in low if event["task"] == first["task"]
                           and (event["phase"] == "first_tile"
                                or (event["phase"] in {"suspended", "complete"} and event["exact"] == 1))
                           and event["completed"] > stop["completed"]
                           and event["generation"] != first["generation"]
                           and event["launch"] == first["launch"] + 1
                           and event["time_ns"] > high_ends[0]["time_ns"]]
                resume_prepares = [event for event in low if event["task"] == first["task"]
                                   and event["phase"] == "prepare" and event["launch"] == first["launch"] + 1
                                   and event["time_ns"] > high_ends[0]["time_ns"]]
                if resumed and resume_prepares:
                    return {"low_first_tile": first, "high_submission": submit,
                            "low_priority_retreat": retreats[0], "low_partial_stop": stop,
                            "high_prepare": high_prepares[0], "high_first_tile": high_starts[0], "high_complete": high_ends[0],
                            "low_resume_prepare": resume_prepares[0], "low_resumed": resumed[0],
                            "low_logical_tasks": sum(event["phase"] == "complete" for event in low),
                            "high_logical_tasks": sum(event["phase"] == "complete" for event in high)}
    raise ValueError("no physically proven incomplete low/high/low handoff")


def wait_until(predicate, processes, timeout=120):
    deadline = time.monotonic() + timeout
    while not predicate():
        if any(process.poll() is not None for process in processes):
            raise RuntimeError("a required process exited before readiness/progress")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for actual readiness/progress")
        time.sleep(0.001)


def clean_environment():
    return {key: value for key, value in os.environ.items()
            if not key.startswith("GLIMMER_") and key not in {"LD_PRELOAD", "LD_AUDIT"}}


def validate_numerical(result, application_hash):
    expected = set()
    for step in range(2):
        expected.add(f"loss_{step}")
        expected.update(f"forward_{step}_{unit}" for unit in range(2))
        expected.update(f"{kind}_{step}_{layer}.{parameter}" for kind in ("gradient", "update")
                        for layer in ("0", "2") for parameter in ("weight", "bias"))
    if (result.get("status") != "ok" or result.get("native_compared") is not True
            or result.get("application_sha256") != application_hash
            or result.get("rtol") != 2e-4 or result.get("atol") != 2e-5
            or len(result.get("checks", [])) != 22 or set(result["checks"]) != expected):
        raise ValueError("missing or mismatched concurrent numerical evidence")


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run(arguments):
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    interceptor = arguments.build / "lib/libglimmer_cuda_interceptor.so"
    children = []
    logs = []
    report = {"status": "failed", "diagnostic_only": True,
              "scope": "persistent kernel tasks inside unchanged training/inference MLPs",
              "commands": []}
    try:
        app_hash = hashlib.sha256(APPLICATION.read_bytes()).hexdigest()
        interceptor_hash = hashlib.sha256(interceptor.read_bytes()).hexdigest()
        numerical_hash = hashlib.sha256(NUMERICAL.read_bytes()).hexdigest()
        report.update(application_sha256=app_hash, interceptor_sha256=interceptor_hash)
        if arguments.numerical_reference:
            report.update(scope="numerical companion importing unchanged MLP with unchanged high inference",
                          numerical_companion_sha256=numerical_hash,
                          native_snapshot_sha256=hashlib.sha256(arguments.numerical_reference.read_bytes()).hexdigest())
        with contextlib.ExitStack() as resources:
            temporary = resources.enter_context(tempfile.TemporaryDirectory(prefix="glimmer-mlp-"))
            socket = pathlib.Path(temporary) / "control.sock"
            def spawn(command, name, environment):
                stream = (output / f"{name}.log").open("w", encoding="utf-8")
                logs.append(stream)
                report["commands"].append({"role": name, "argv": [str(value) for value in command],
                                           "environment": {key: value for key, value in environment.items()
                                                           if key.startswith("GLIMMER_") or key == "LD_PRELOAD"}})
                process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                children.append(process)
                # Stop application children before their control service and
                # remove its socket only after all owned processes have exited.
                resources.callback(stop_process, process)
                return process
            base = clean_environment()
            base.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            service = spawn([str(arguments.build / "bin/glimmer_control_service"), "--socket", str(socket),
                             "--quota-bytes", "4294967296", "--execution-mode", "remote",
                             "--lease-timeout-ms", "5000", "--max-queued-tasks", "256",
                             "--max-concurrent-tasks", "1", "--scheduler-policy", "priority",
                             "--priority-reserved-slots", "0", "--priority-threshold", "50",
                             "--bind-leases-to-process", "--trace-scheduler"], "service", base)
            wait_until(socket.exists, [service])
            processes = {}
            for name, role, priority, work in (("low", "training", 10, 2), ("high", "inference", 100, 1)):
                environment = dict(base, LD_PRELOAD=str(arguments.build / "lib/libglimmer_cuda_interceptor.so"),
                                   GLIMMER_PERSISTENT_BLAS="1", GLIMMER_PERSISTENT_AUDIT="1",
                                   GLIMMER_COOPERATIVE_DRIVER_PREVIEW="1", GLIMMER_COOPERATIVE_RUNTIME_PREVIEW="1",
                                   GLIMMER_MEMORY_LIMIT_BYTES="4294967296", GLIMMER_SCHEDULER_MODE="enforce",
                                   GLIMMER_SCHEDULER_POLICY="priority", GLIMMER_SCHEDULER_PRIORITY=str(priority),
                                   GLIMMER_SCHEDULER_TENANT_ID=name, GLIMMER_SCHEDULER_CONTROL_SOCKET=str(socket),
                                   GLIMMER_MAX_CONCURRENT_KERNELS="1", GLIMMER_TRACE_KERNEL_LAUNCHES="1",
                                   GLIMMER_TRACE_LAUNCH_TIMINGS="1")
                command = [str(arguments.python), str(APPLICATION), "--role", role,
                                         "--warmup", "1", "--iterations", "2", "--work-units", str(work),
                                         "--ready-file", str(output / f"{name}.ready.json"),
                                         "--start-file", str(output / f"{name}.start"),
                                         "--status-file", str(output / f"{name}.status.json"),
                                         "--output", str(output / f"{name}.csv")]
                if name == "low" and arguments.numerical_reference:
                    command = [str(arguments.python), str(NUMERICAL),
                               "--reference", str(arguments.numerical_reference),
                               "--snapshot", str(output / "numerical.pt"),
                               "--output", str(output / "numerical.json"),
                               "--ready-file", str(output / "low.ready.json"),
                               "--start-file", str(output / "low.start")]
                processes[name] = spawn(command, name, environment)
                wait_until((output / f"{name}.ready.json").exists, [service, processes[name]])
                report[f"{name}_ready"] = read_metadata(output / f"{name}.ready.json")
            if (report["low_ready"].get("device_uuid") != report["high_ready"].get("device_uuid")
                    or not report["low_ready"].get("device_uuid")
                    or int(report["low_ready"]["pid"]) == int(report["high_ready"]["pid"])):
                raise ValueError("handoff requires distinct processes on the same observed GPU")
            low_release_ns = time.monotonic_ns()
            (output / "low.start").touch()
            def low_is_physical():
                return any(event["phase"] == "first_tile" and event["time_ns"] >= low_release_ns
                           for event in records((output / "low.log").read_text(), AUDIT_PREFIX, allow_partial=True))
            wait_until(low_is_physical, [service, *processes.values()])
            high_release_ns = time.monotonic_ns()
            (output / "high.start").touch()
            report.update(low_release_ns=low_release_ns, high_release_ns=high_release_ns)
            for name, process in processes.items():
                if process.wait(timeout=120) != 0:
                    raise RuntimeError(f"{name} application failed")
            if service.poll() is not None:
                raise RuntimeError("control service exited prematurely")
            metrics_command = [str(arguments.build / "bin/glimmer_control_client"),
                               "--socket", str(socket), "metrics"]
            report["commands"].append({"role": "metrics", "argv": metrics_command, "environment": {}})
            metrics = subprocess.run(metrics_command, env=base, capture_output=True, text=True,
                                     check=True, timeout=10)
            (output / "scheduler.metrics").write_text(metrics.stdout, encoding="utf-8")
            validate_scheduler_metrics(metrics.stdout)
            report["scheduler_drained"] = True
            low_text = (output / "low.log").read_text()
            high_text = (output / "high.log").read_text()
            report["handoff"] = verify_handoff(low_text, high_text,
                                                low_release_ns, high_release_ns,
                                                numerical=bool(arguments.numerical_reference))
            if arguments.numerical_reference:
                result = json.loads((output / "numerical.json").read_text())
                validate_numerical(result, app_hash)
                report["numerical"] = result
                if (hashlib.sha256(NUMERICAL.read_bytes()).hexdigest() != numerical_hash
                        or hashlib.sha256(arguments.numerical_reference.read_bytes()).hexdigest()
                        != report["native_snapshot_sha256"]):
                    raise RuntimeError("numerical companion or reference changed during run")
            if (low_text.count("persistent_sgemm accepted=1") != (12 if arguments.numerical_reference else 18)
                    or low_text.count("persistent_lt_bias accepted=1") != (8 if arguments.numerical_reference else 12)
                    or high_text.count("persistent_lt_bias accepted=1") != 6
                    or high_text.count("persistent_sgemm accepted=1") != 0):
                raise ValueError("selected MLP forward/backward coverage inventory differs")
            if hashlib.sha256(APPLICATION.read_bytes()).hexdigest() != app_hash:
                raise RuntimeError("application source changed")
            if hashlib.sha256(interceptor.read_bytes()).hexdigest() != interceptor_hash:
                raise RuntimeError("interceptor binary changed during the run")
            report["status"] = "ok"
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    finally:
        for process in reversed(children):
            stop_process(process)
        for stream in logs:
            stream.close()
        (output / "handoff.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output / "handoff.json"),
                      "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--build", type=pathlib.Path, default=ROOT / "build/cuda-perf")
    parser.add_argument("--python", type=pathlib.Path, default=APPLICATION.parent / ".venv/bin/python")
    parser.add_argument("--numerical-reference", type=pathlib.Path,
                        help="also compare low numerical-companion results with this native snapshot")
    arguments = parser.parse_args()
    arguments.build = arguments.build.resolve()
    arguments.python = arguments.python.absolute()
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
