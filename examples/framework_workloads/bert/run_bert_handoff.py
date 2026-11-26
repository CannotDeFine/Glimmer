#!/usr/bin/env python3
"""Check unchanged BERT/ONNX numerics and independent physical priority handoff."""

import argparse
import contextlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import time

from bert_contract import sha256, write_json
from bert_colocation_worker import make_trace, publish, trace_digest
from run_bert_baseline import EXAMPLE, ROOT, SMOKE, environment, optimized_build, run_bounded
from run_persistent_mlp_handoff import AUDIT_PREFIX, records, stop_process, wait_until
from measurements import validate_scheduler_metrics
from verify_bert_handoff import HIGH_BOUNDARIES, validate_colocation, verify

HIGH_RELEASE_TRIGGERS = ("live-first-tile", "large-prepare")
HIGH_WORKLOADS = ("original", "periodic")


def high_configuration(args):
    workload = getattr(args, "high_workload", "original")
    count, period = getattr(args, "high_requests", 20), getattr(args, "high_period_ms", 100)
    if workload not in HIGH_WORKLOADS:
        raise ValueError("unknown high workload")
    if workload == "periodic" and (
            getattr(args, "high_timeline_boundary", "runtime-markers") != "fixed-inventory"
            or any(type(value) is not int or value <= 0 for value in (count, period))
            or count > 100 or count * period > 10_000):
        raise ValueError("periodic high needs fixed inventory and at most 100 requests / 10 offered seconds")
    return {"high_workload": workload, "high_requests": count if workload == "periodic" else 2,
            "high_period_ms": period if workload == "periodic" else None}


def application_command(args, role, output, configuration):
    if role == "high" and configuration["high_workload"] == "periodic":
        return [str(args.python), "-B", str(EXAMPLE / "bert_colocation_worker.py"),
                "--assets", str(args.assets), "--export", str(args.export), "--role", "inference",
                "--output", str(output / "inference"), "--warmup", "1", "--timeout", "600",
                "--ready-file", str(output / "high.ready.json"),
                "--start-file", str(output / "inference.start.json"),
                "--stop-file", str(output / "training.stop"),
                "--active-file", str(output / "training.active.json")]
    app = "bert_workload.py" if role == "low" else "onnx_bert_workload.py"
    command = [str(args.python), "-B", str(EXAMPLE / app), "--assets", str(args.assets),
               "--output", str(output / role), "--warmup", "1", "--iterations", "2",
               "--ready-file", str(output / f"{role}.ready.json"), "--start-file", str(output / f"{role}.start")]
    command += (["--role", "training", "--snapshot", "--reference", str(args.reference)] if role == "low"
                else ["--export", str(args.export)])
    if args.worker and (role != "high" or getattr(args, "high_timeline_boundary", "runtime-markers") == "runtime-markers"):
        command.append("--profile-cuda")
    return command


def child_environment(args, priority, name, socket):
    if args.control_transport not in {"socket", "shared_memory", "auto"}:
        raise ValueError("invalid control transport")
    env = environment("cooperative", args.build)
    if args.worker:
        # Preserve the installed profiler's injection, not arbitrary inherited
        # credentials, loader overrides or application instrumentation.
        injection = {"CUDA_INJECTION64_PATH", "CUPTI_PROFILE_MODE", "TRACE_START_IMMEDIATELY",
                     "USE_AGENT_API", "__GL_CONSTANT_FRAME_RATE_HINT"}
        env.update({key: value for key, value in os.environ.items()
                    if key in injection or key.startswith(("NSYS_", "QUADD_"))})
    env.update(GLIMMER_PERSISTENT_AUDIT="1", GLIMMER_SCHEDULER_PRIORITY=str(priority),
               GLIMMER_SCHEDULER_TENANT_ID=name, GLIMMER_SCHEDULER_CONTROL_SOCKET=str(socket),
               GLIMMER_TRACE_KERNEL_LAUNCHES="1", GLIMMER_TRACE_LAUNCH_TIMINGS="1")
    env["GLIMMER_CONTROL_TRANSPORT"] = args.control_transport
    return env


def stop_group(process):
    """Only processes started by this runner with their own session qualify."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def read_ready(path, processes, expected_pid, timeout=240):
    """File creation is not publication of a complete JSON ready record."""
    record = None

    def published():
        nonlocal record
        try:
            candidate = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        if (not isinstance(candidate, dict) or candidate.keys() != {"pid", "ready_ns", "device_uuid"}
                or type(candidate["pid"]) is not int or candidate["pid"] != expected_pid
                or type(candidate["ready_ns"]) is not int or not 0 < candidate["ready_ns"] <= time.monotonic_ns()
                or not isinstance(candidate["device_uuid"], str) or not candidate["device_uuid"]):
            raise ValueError("invalid readiness identity")
        record = candidate
        return True

    wait_until(published, processes, timeout=timeout)
    return record


class LiveLowProgress:
    """Observe only newly published, still-running large BERT GEMM progress."""

    def __init__(self, stream, release_ns, trigger="live-first-tile"):
        if trigger not in HIGH_RELEASE_TRIGGERS:
            raise ValueError("unknown high release trigger")
        self.stream = stream
        self.release_ns = release_ns
        self.trigger = trigger
        self.pending = ""
        self.running = {}
        self.observation = None

    def ready(self):
        self.pending += self.stream.read()
        end = self.pending.rfind("\n") + 1
        complete, self.pending = self.pending[:end], self.pending[end:]
        for item in records(complete, AUDIT_PREFIX):
            if item["time_ns"] < self.release_ns:
                continue
            key = (item["task"], item["generation"], item["launch"])
            if item["phase"] == "first_tile" or (self.trigger == "large-prepare" and item["phase"] == "prepare"):
                self.running[key] = item
            elif item["phase"] in {"complete", "suspended", "retreat", "priority_retreat"}:
                self.running.pop(key, None)
        # The fixed 3072x768 output has 9216 16x16 logical tiles.
        # The optional prepare trigger is intent only, giving the unchanged high
        # CLI's measured host lead-in an earlier stimulus. It supplies no GPU
        # progress proof: the independent verifier still requires first-tile
        # observation before actual queued high demand in the stopped launch.
        self.observation = next((item for item in self.running.values()
                                 if item["logical"] >= 9216 and item["completed"] < item["logical"]
                                 and (item["completed"] > 0 or self.trigger == "large-prepare")), None)
        return self.observation is not None


def collect(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "diagnostic_only": True, "preemption_acceptance": False,
              "performance_acceptance": False, "commands": [],
              "high_timeline_boundary": getattr(args, "high_timeline_boundary", "runtime-markers"),
              "high_release_trigger": getattr(args, "high_release_trigger", "live-first-tile")}
    try:
        report.update(high_configuration(args))
        if report["high_timeline_boundary"] not in HIGH_BOUNDARIES:
            raise ValueError("unknown high inference boundary")
        if report["high_release_trigger"] not in HIGH_RELEASE_TRIGGERS:
            raise ValueError("unknown high release trigger")
        paths = [EXAMPLE / name for name in ("bert_workload.py", "onnx_bert_workload.py", "bert_contract.py",
                 "run_bert_baseline.py", "run_bert_handoff.py", "verify_bert_handoff.py",
                 "bert_colocation_worker.py", "run_bert_colocation.py")]
        paths += [SMOKE / "run_persistent_mlp_handoff.py", args.build / "lib/libglimmer_cuda_interceptor.so",
                  args.reference / "result.json", args.reference / "snapshot.pt", args.export / "export.json"]
        report["fingerprints"] = {str(path): sha256(path) for path in paths}
        with contextlib.ExitStack() as resources:
            temporary = resources.enter_context(tempfile.TemporaryDirectory(prefix="glimmer-bert-"))
            socket = pathlib.Path(temporary) / "control.sock"

            def spawn(name, command, env):
                stream = resources.enter_context((output / f"{name}.log").open("w", encoding="utf-8"))
                report["commands"].append({"role": name, "argv": command, "environment": env})
                write_json(output / "probe.json", report)
                # A profiled worker's descendants stay in the outer bounded
                # capture group. Direct collection owns individual groups.
                process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=not args.worker)
                resources.callback(stop_process if args.worker else stop_group, process)
                return process

            base = environment("native", args.build)
            service = spawn("service", [str(args.build / "bin/glimmer_control_service"), "--socket", str(socket),
                "--quota-bytes", "12884901888", "--execution-mode", "remote", "--lease-timeout-ms", "5000",
                "--max-queued-tasks", "256", "--max-concurrent-tasks", "1", "--scheduler-policy", "priority",
                "--priority-reserved-slots", "0", "--priority-threshold", "50", "--bind-leases-to-process",
                "--trace-scheduler"], base)
            wait_until(socket.exists, [service], timeout=30)
            processes = {}
            for name, priority in (("low", 10), ("high", 100)):
                env = child_environment(args, priority, name, socket)
                command = application_command(args, name, output, report)
                processes[name] = spawn(name, command, env)
                report[f"{name}_ready"] = read_ready(output / f"{name}.ready.json",
                                                     [service, processes[name]], processes[name].pid)
            if (not report["low_ready"]["device_uuid"]
                    or report["low_ready"]["device_uuid"] != report["high_ready"]["device_uuid"]
                    or report["low_ready"]["pid"] == report["high_ready"]["pid"]):
                raise ValueError("require distinct processes on the same GPU")
            low_stream = resources.enter_context((output / "low.log").open("r", encoding="utf-8"))
            low_stream.seek(0, os.SEEK_END)
            report["low_audit_read_begin_offset"] = low_stream.tell()
            report["low_release_ns"] = time.monotonic_ns()
            progress = LiveLowProgress(low_stream, report["low_release_ns"], report["high_release_trigger"])
            (output / "low.start").touch()
            wait_until(progress.ready, [service, *processes.values()], timeout=120)
            report["low_progress_for_high_release"] = progress.observation
            report["low_audit_read_end_offset"] = low_stream.tell()
            report["high_release_ns"] = time.monotonic_ns()
            if report["high_workload"] == "periodic":
                trace = make_trace(report["high_requests"], report["high_period_ms"] * 1_000_000)
                report["high_schedule"] = {"begin_ns": report["high_release_ns"] + 100_000_000,
                    "trace": trace, "trace_sha256": trace_digest(trace), "preparation_required": True,
                    "drain_ns": 10_000_000_000}
                publish(output / "inference.start.json", report["high_schedule"])
            else:
                (output / "high.start").touch()
            write_json(output / "probe.json", report)
            deadline = time.monotonic() + 600
            for name, process in processes.items():
                if process.wait(timeout=max(0, deadline - time.monotonic())) != 0:
                    raise RuntimeError(f"{name} application failed")
            if service.poll() is not None:
                raise RuntimeError("control service exited")
            for name in processes:
                folder = "inference" if name == "high" and report["high_workload"] == "periodic" else name
                result = json.loads((output / folder / "result.json").read_text())
                if result.get("status") != "ok":
                    raise ValueError(f"{name} numerical checks failed")
                report[f"{name}_result_sha256"] = sha256(output / folder / "result.json")
            command = [str(args.build / "bin/glimmer_control_client"), "--socket", str(socket), "metrics"]
            report["commands"].append({"role": "metrics", "argv": command, "environment": base})
            metrics = subprocess.run(command, env=base, capture_output=True, text=True, check=True, timeout=10)
            (output / "scheduler.metrics").write_text(metrics.stdout, encoding="utf-8")
            validate_scheduler_metrics(metrics.stdout)
            report["scheduler_drained"] = True
            if any(sha256(path) != digest for path, digest in report["fingerprints"].items()):
                raise ValueError("source, runtime or reference changed")
        report["status"] = "ok"
    except Exception as error:
        report["error"] = str(error)
    finally:
        write_json(output / "probe.json", report)
    if report["status"] != "ok":
        raise RuntimeError(report.get("error", "collection failed"))


def profile_command(args, nsys, output):
    base = environment("native", args.build)
    return ["/usr/bin/env", "-i", *(f"{key}={value}" for key, value in sorted(base.items())),
            nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--export=sqlite",
            "--force-overwrite=false", f"--output={output / 'timeline'}", str(args.python), "-B", __file__,
            "--worker", "--assets", str(args.assets), "--export", str(args.export),
            "--reference", str(args.reference), "--build", str(args.build), "--python", str(args.python),
            "--control-transport", args.control_transport,
            "--high-timeline-boundary", getattr(args, "high_timeline_boundary", "runtime-markers"),
            "--high-release-trigger", getattr(args, "high_release_trigger", "live-first-tile"),
            "--high-workload", getattr(args, "high_workload", "original"),
            "--high-requests", str(getattr(args, "high_requests", 20)),
            "--high-period-ms", str(getattr(args, "high_period_ms", 100)),
            "--output", str(output / "application")]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "diagnostic_only": True, "performance_acceptance": False}
    try:
        optimized_build(args.build)
        nsys = shutil.which(args.nsys)
        if not nsys:
            raise RuntimeError("Nsight Systems is required for independent physical proof")
        unprofiled = argparse.Namespace(**vars(args))
        unprofiled.output = args.output / "unprofiled"
        collect(unprofiled)
        report["unprofiled_numerics"] = validate_colocation(unprofiled.output, args.reference)[3]
        profiled = args.output / "profiled"
        profiled.mkdir()
        run_bounded([nsys, "--version"], profiled / "nsys-version.log", 30)
        command = profile_command(args, nsys, profiled)
        report["profile_command"] = command
        write_json(args.output / "handoff.json", report)
        run_bounded(command, profiled / "profile.log", 900)
        report["profiled_correctness"] = verify(profiled / "application", profiled / "timeline.sqlite", args.reference)
        report["status"] = "ok"
        report["scope"] = "unchanged BERT training / native ONNX inference kernel-task handoff, not whole-request isolation"
    except Exception as error:
        report["error"] = str(error)
    finally:
        write_json(args.output / "handoff.json", report)
    print(json.dumps({"status": report["status"], "error": report.get("error"), "report": str(args.output / "handoff.json")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "export", "reference", "output"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--build", type=pathlib.Path, default=ROOT / "build/cuda-perf")
    parser.add_argument("--python", type=pathlib.Path, default=SMOKE / ".venv/bin/python")
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--control-transport", choices=("socket", "shared_memory", "auto"), default="socket")
    parser.add_argument("--high-timeline-boundary", choices=HIGH_BOUNDARIES, default="runtime-markers",
                        help="diagnostic high CUDA boundary; fixed-inventory omits high Runtime profiler markers")
    parser.add_argument("--high-release-trigger", choices=HIGH_RELEASE_TRIGGERS, default="live-first-tile",
                        help="diagnostic host stimulus only; prepare intent cannot replace physical GPU proof")
    parser.add_argument("--high-workload", choices=HIGH_WORKLOADS, default="original")
    parser.add_argument("--high-requests", type=int, default=20, help="periodic diagnostic requests, at most 100")
    parser.add_argument("--high-period-ms", type=int, default=100, help="periodic diagnostic arrival period")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("assets", "export", "reference", "output", "build"):
        setattr(args, name, getattr(args, name).resolve())
    args.python = args.python.absolute()
    if args.worker:
        collect(args)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
