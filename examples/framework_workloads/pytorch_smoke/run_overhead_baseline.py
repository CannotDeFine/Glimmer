#!/usr/bin/env python3
"""Measure optimized, uncontended preload, quota, and launch-admission overhead.

Reuses the existing CUDA MLP and wall-clock accounting. No performance threshold
is implied by successful collection. This is not a co-location/SLO experiment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

from measurements import MEASUREMENT_VERSION, read_metadata, summarize_solo
from run_slo_validation import EXAMPLE, REPOSITORY, benchmark_environment, run_bounded


VERSION = "uncontended_overhead_v1"
MODES = ("native", "preload", "quota", "local", "remote")
ROLES = ("inference", "training")
IDENTITY = ("device_uuid", "device_index", "torch_version", "cuda_version")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, default=REPOSITORY / "build/cuda-perf")
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--analyze-only", type=pathlib.Path)
    parser.add_argument("--python", default=str(EXAMPLE / ".venv/bin/python"))
    parser.add_argument("--duration-seconds", type=float, default=10)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--training-work-units", type=int, default=2)
    parser.add_argument("--quota-bytes", type=int, default=536870912)
    args = parser.parse_args(argv)
    if args.analyze_only is not None:
        return args
    for name in ("repetitions", "warmup", "batch_size", "hidden_size", "training_work_units", "quota_bytes"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if args.quota_bytes > 2**64 - 1:
        parser.error("quota exceeds the unsigned 64-bit range")
    if not math.isfinite(args.duration_seconds) or not 0 < args.duration_seconds <= 3600:
        parser.error("duration must be finite and in (0, 3600]")
    args.build_dir = args.build_dir.resolve()
    return args


def optimized_build(directory: pathlib.Path) -> dict:
    cache = {}
    for line in (directory / "CMakeCache.txt").read_text().splitlines():
        if line and not line.startswith(("#", "//")) and "=" in line:
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    if cache.get("CMAKE_BUILD_TYPE") not in ("Release", "RelWithDebInfo"):
        raise ValueError("performance baseline requires Release or RelWithDebInfo; use cuda-perf")
    for option in ("GLIMMER_ENABLE_SANITIZERS", "GLIMMER_ENABLE_CLANG_TIDY"):
        if cache.get(option, "OFF").upper() not in ("OFF", "FALSE", "0", "NO", ""):
            raise ValueError(f"performance baseline requires {option}=OFF")
    entries = json.loads((directory / "compile_commands.json").read_text())
    checked = set()
    for entry in entries:
        source = pathlib.Path(entry["file"])
        if not source.is_absolute():
            source = pathlib.Path(entry["directory"]) / source
        for module in ("core", "control", "interceptor"):
            if source.resolve().is_relative_to(REPOSITORY / "src" / module):
                flags = entry.get("arguments") or shlex.split(entry["command"])
                optimization = [flag for flag in flags if flag.startswith("-O")]
                if (not optimization or optimization[-1] not in ("-O1", "-O2", "-O3", "-Os", "-Oz")
                        or any(flag.startswith("-fsanitize") for flag in flags)):
                    raise ValueError(f"unoptimized or sanitized compile command: {source}")
                checked.add(module)
    if checked != {"core", "control", "interceptor"}:
        raise ValueError("compilation database is missing runtime modules")
    return {key: value for key, value in cache.items()
            if key.startswith(("CMAKE_CXX_", "CMAKE_CUDA_", "GLIMMER_")) or key == "CMAKE_BUILD_TYPE"}


def mode_settings(mode: str, role: str, args: argparse.Namespace, socket_path: str = "") -> dict[str, str]:
    if mode not in MODES or role not in ROLES:
        raise ValueError("unknown mode or role")
    if mode == "native":
        return {}
    settings = {"LD_PRELOAD": str(args.build_dir / "lib/libglimmer_cuda_interceptor.so"),
                "GLIMMER_SCHEDULER_MODE": "off"}
    if mode in ("quota", "local", "remote"):
        settings["GLIMMER_MEMORY_LIMIT_BYTES"] = str(args.quota_bytes)
    if mode in ("local", "remote"):
        settings.update(GLIMMER_SCHEDULER_MODE="enforce", GLIMMER_SCHEDULER_POLICY="priority",
                        GLIMMER_SCHEDULER_PRIORITY="100" if role == "inference" else "10",
                        GLIMMER_SCHEDULER_TENANT_ID=f"overhead-{role}",
                        GLIMMER_MAX_CONCURRENT_KERNELS="2", GLIMMER_SCHEDULER_BATCH_SIZE="1")
    if mode == "remote":
        if not socket_path:
            raise ValueError("remote baseline requires a control socket")
        settings["GLIMMER_SCHEDULER_CONTROL_SOCKET"] = socket_path
    return settings


def workload_command(mode: str, role: str, args: argparse.Namespace, directory: pathlib.Path,
                     socket_path: str = "") -> list[str]:
    settings = mode_settings(mode, role, args, socket_path)
    command = ["env", *(f"{key}={value}" for key, value in settings.items()), args.python,
               str(EXAMPLE / "pytorch_smoke.py"), "--role", role, "--device", "cuda",
               "--output", str(directory / f"{role}.csv"),
               "--status-file", str(directory / f"{role}.status"),
               "--work-units", str(args.training_work_units if role == "training" else 1)]
    for option in ("duration_seconds", "warmup", "batch_size", "hidden_size"):
        command += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    if "GLIMMER_MEMORY_LIMIT_BYTES" in settings:
        command += ["--expected-memory-total-bytes", str(args.quota_bytes)]
    return command


def check_remote_stats(response: str) -> None:
    fields = response.split()
    if (len(fields) != 13 or fields[:2] != ["GLIMMER_TASK_V1", "STATS"]
            or any(not field.isascii() or not field.isdecimal() for field in fields[2:])):
        raise ValueError("invalid remote scheduler stats")
    values = list(map(int, fields[2:]))
    total, queued, running, completed, cancelled, failed, _, reserved, allocated, slots, _ = values
    if (total <= 0 or completed != total or any((queued, running, cancelled, failed, reserved, allocated))
            or slots != 2):
        raise ValueError("remote baseline has missing admissions or undrained/failed tasks")


def stop_service(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@contextmanager
def remote_service(args: argparse.Namespace, directory: pathlib.Path):
    # Only this temporary directory and child process are owned by the runner.
    with tempfile.TemporaryDirectory(prefix="glimmer-overhead-", dir="/tmp") as temporary:
        socket_path = str(pathlib.Path(temporary) / "control.sock")
        command = [str(args.build_dir / "bin/glimmer_control_service"), "--socket", socket_path,
                   "--execution-mode", "remote", "--quota-bytes", "4294967296",
                   "--max-concurrent-tasks", "2", "--max-queued-tasks", "256",
                   "--scheduler-policy", "priority", "--bind-leases-to-process"]
        (directory / "service-command.json").write_text(json.dumps(command, indent=2) + "\n")
        client = [str(args.build_dir / "bin/glimmer_control_client"), "--socket", socket_path, "stats"]
        with (directory / "service.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=benchmark_environment())
            try:
                deadline = time.monotonic() + 15
                while not pathlib.Path(socket_path).is_socket():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("control service failed to start; inspect service.log")
                    time.sleep(0.01)
                subprocess.run(client, check=True, capture_output=True, timeout=5, env=benchmark_environment())
                yield socket_path
                deadline = time.monotonic() + 10
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("control service exited during the workload")
                    result = subprocess.run(client, check=True, capture_output=True, text=True,
                                            timeout=5, env=benchmark_environment())
                    (directory / "service-stats.txt").write_text(result.stdout)
                    try:
                        check_remote_stats(result.stdout)
                        break
                    except ValueError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.01)
            finally:
                stop_service(process)


def aggregate(rows: list[dict], repetitions: int) -> list[dict]:
    expected = {(mode, role, repetition) for mode in MODES for role in ROLES
                for repetition in range(1, repetitions + 1)}
    observed = [(row["mode"], row["role"], int(row["repetition"])) for row in rows]
    if repetitions <= 0 or len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("missing, duplicate, or unexpected overhead repetitions")
    identity = tuple(rows[0][key] for key in IDENTITY)
    for row in rows:
        if tuple(row[key] for key in IDENTITY) != identity or row["measurement_version"] != MEASUREMENT_VERSION:
            raise ValueError("GPU identity, framework, or measurement versions differ")
    results = []
    for role in ROLES:
        for mode in MODES:
            group = [row for row in rows if row["role"] == role and row["mode"] == mode]
            result = {"role": role, "mode": mode, "runs": repetitions}
            for metric in ("samples", "mean_ms", "p50_ms", "p95_ms", "p99_ms", "steps_per_second"):
                values = [float(row[metric]) for row in group]
                if any(not math.isfinite(value) or value <= 0 for value in values):
                    raise ValueError(f"invalid metric: {metric}")
                result[metric + "_median"] = statistics.median(values)
                result[metric + "_min"] = min(values)
                result[metric + "_max"] = max(values)
            results.append(result)
        native = next(row for row in results if row["role"] == role and row["mode"] == "native")
        for row in results:
            if row["role"] == role:
                row["p99_vs_native"] = row["p99_ms_median"] / native["p99_ms_median"]
                row["throughput_vs_native"] = row["steps_per_second_median"] / native["steps_per_second_median"]
    return results


def collect_results(output: pathlib.Path) -> list[dict]:
    if (output / "failure.json").exists():
        raise ValueError("experiment has a recorded failure")
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["version"] != VERSION or not manifest.get("collection_complete"):
        raise ValueError("incompatible or incomplete overhead collection")
    args = manifest["arguments"]
    rows = []
    for repetition in range(1, args["repetitions"] + 1):
        for mode in MODES:
            for role in ROLES:
                directory = output / f"{mode}-{role}-r{repetition}"
                framework = read_metadata(directory / f"{role}.status")
                measurement = summarize_solo(directory, role)
                if measurement["window_end_ns"] - measurement["window_start_ns"] < args["duration_seconds"] * 0.95e9:
                    raise ValueError("measurement window is shorter than requested")
                if mode in ("quota", "local", "remote"):
                    if int(framework["visible_memory_total_bytes"]) != args["quota_bytes"]:
                        raise ValueError("quota-enabled run did not expose the requested memory view")
                if mode == "remote":
                    check_remote_stats((directory / "service-stats.txt").read_text())
                rows.append({"mode": mode, "role": role, "repetition": repetition,
                             "measurement_version": MEASUREMENT_VERSION,
                             **{key: framework[key] for key in IDENTITY}, **measurement})
    result = aggregate(rows, args["repetitions"])
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    print("role mode p99_ms steps_per_second p99_vs_native throughput_vs_native")
    for row in result:
        print(f"{row['role']} {row['mode']} {row['p99_ms_median']:.3f} "
              f"{row['steps_per_second_median']:.3f} {row['p99_vs_native']:.3f} "
              f"{row['throughput_vs_native']:.3f}")
    return result


def artifact_hashes(args: argparse.Namespace) -> dict:
    paths = [args.build_dir / name for name in ("CMakeCache.txt", "compile_commands.json",
             "lib/libglimmer_cuda_interceptor.so", "bin/glimmer_control_service", "bin/glimmer_control_client")]
    paths += list(EXAMPLE.glob("*.py"))
    paths += [REPOSITORY / "CMakeLists.txt", REPOSITORY / "CMakePresets.json"]
    for root in ("src", "include", "cmake"):
        paths += [path for path in (REPOSITORY / root).rglob("*") if path.is_file()]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main(argv=None) -> int:
    args = parse_args(argv)
    output = None
    run_id = "preflight"
    try:
        if args.analyze_only is not None:
            collect_results(args.analyze_only)
            return 0
        build = optimized_build(args.build_dir)
        if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
            raise ValueError("disable CUDA_LAUNCH_BLOCKING for performance measurement")
        if args.output_dir is None:
            (EXAMPLE / "output").mkdir(exist_ok=True)
            output = pathlib.Path(tempfile.mkdtemp(prefix="overhead-baseline-", dir=EXAMPLE / "output"))
        else:
            destination = args.output_dir.resolve()
            if destination.exists() and any(destination.iterdir()):
                raise ValueError(f"output directory must be empty: {destination}")
            destination.mkdir(parents=True, exist_ok=True)
            output = destination
        manifest = {"version": VERSION, "collection_complete": False, "build": build,
                    "arguments": {key: str(value) if isinstance(value, pathlib.Path) else value
                                  for key, value in vars(args).items()},
                    "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES",
                        "CUDA_LAUNCH_BLOCKING", "CUDA_MODULE_LOADING", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
                        "NVIDIA_TF32_OVERRIDE")},
                    "started_unix_ns": time.time_ns(), "artifacts_sha256": artifact_hashes(args)}
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"overhead_baseline output={output}", flush=True)
        pairs = [(mode, role) for mode in MODES for role in ROLES]
        for repetition in range(1, args.repetitions + 1):
            offset = (repetition - 1) % len(pairs)
            for mode, role in pairs[offset:] + pairs[:offset]:
                run_id = f"{mode}-{role}-r{repetition}"
                directory = output / run_id
                directory.mkdir()
                print(f"overhead_baseline run={run_id}", flush=True)

                def run(socket_path=""):
                    command = workload_command(mode, role, args, directory, socket_path)
                    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                    run_bounded(command, directory / "workload.log", args.duration_seconds + 180)

                if mode == "remote":
                    with remote_service(args, directory) as socket_path:
                        run(socket_path)
                else:
                    run()
        if artifact_hashes(args) != manifest["artifacts_sha256"]:
            raise ValueError("runtime artifacts or measurement sources changed during the experiment")
        manifest["collection_complete"] = True
        manifest["finished_unix_ns"] = time.time_ns()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        collect_results(output)
        print(f"overhead_baseline status=ok performance_pass=not_asserted output={output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        if output is not None:
            (output / "failure.json").write_text(json.dumps(
                {"run_id": run_id, "error": str(error)}, indent=2) + "\n")
        print(f"overhead_baseline status=failed run={run_id} error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
