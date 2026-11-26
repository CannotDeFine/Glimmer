#!/usr/bin/env python3
"""Compare native solo baselines with native and Glimmer co-location.

All runs use the same workload and shared measurement duration. Detailed
tracing is disabled. Results describe only the explicitly selected workload.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import time

from measurements import (MEASUREMENT_VERSION, WORKLOAD_VERSIONS, read_metadata,
                          summarize_colocation, summarize_solo, workload_identity)


EXAMPLE = pathlib.Path(__file__).resolve().parent
REPOSITORY = EXAMPLE.parents[2]
sys.path.insert(0, str(REPOSITORY))
try:
    from scripts import artifact_paths
finally:
    sys.path.pop(0)
if pathlib.Path(artifact_paths.__file__).resolve() != REPOSITORY / "scripts/artifact_paths.py":
    raise ImportError("artifact_paths must come from this repository")

COMPARISON_VERSION = "solo_colocation_v1"
SOLO_ROLES = {"training-alone": "training", "inference-alone": "inference"}
CONFIGURATIONS = {
    "native": ["--mode", "native"],
    "priority": ["--mode", "priority"],
    "static": ["--mode", "priority", "--priority-reserved-slots", "1"],
    "adaptive": ["--mode", "priority", "--adaptive-slo-max-reserved-slots", "1"],
}
MATCHED_FIELDS = ("measurement_version", "device_uuid", "device_index", "torch_version",
                  "cuda_version", "duration_seconds", "tensor_batch_size", "hidden_size",
                  "training_work_units", "inference_target_ms", "warmup")


def prepare_output(namespace: str, explicit: pathlib.Path | None, *, category: str = "runs") -> pathlib.Path:
    if explicit is not None and explicit.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {explicit}")
    root = artifact_paths.resolve_artifact_root(namespace, category=category, output_root=explicit)
    if explicit is None:
        return artifact_paths.create_attempt(root)
    output = root.path
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    marker = output / ".glimmer-collection"
    try:
        marker.mkdir()
    except FileExistsError as error:
        raise ValueError(f"output directory is already claimed: {output}") from error
    if any(path != marker for path in output.iterdir()):
        raise ValueError(f"output directory changed during claim: {output}")
    return output


def benchmark_environment() -> dict[str, str]:
    excluded = {"LD_PRELOAD", "LD_DEBUG", "LD_DEBUG_OUTPUT", "LD_AUDIT"}
    return {key: value for key, value in os.environ.items()
            if key not in excluded and not key.startswith(("GLIMMER_", "CUDNN_LOG", "CUDNN_FRONTEND_LOG"))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path,
                        help="exact empty result directory in artifacts/ or outside the repository")
    parser.add_argument("--analyze-only", type=pathlib.Path,
                        help="rebuild aggregates from an existing summary.csv directory")
    parser.add_argument("--build-dir", type=pathlib.Path, default=REPOSITORY / "build/cuda-gpu")
    parser.add_argument("--python", default=str(EXAMPLE / ".venv/bin/python"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--inference-target-ms", type=float)
    parser.add_argument("--adaptive-slo-target-queue-us", type=int, default=500)
    parser.add_argument("--adaptive-slo-window", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--workload", choices=tuple(WORKLOAD_VERSIONS), default="mlp")
    parser.add_argument("--training-work-units", type=int, default=2)
    parser.add_argument("--cooperative-preview", action="store_true",
                        help="explicitly enable cooperative routing for Glimmer co-runs")
    parser.add_argument("--persistent-worker-blocks", type=int, default=3)
    for option in ("max-calls", "prefix-calls", "pending-prefixes", "acceptance-us"):
        parser.add_argument("--native-epoch-" + option, type=int)
    args = parser.parse_args()
    if args.analyze_only is not None:
        return args
    for name in ("repetitions", "warmup", "batch_size", "hidden_size", "training_work_units",
                 "adaptive_slo_target_queue_us", "adaptive_slo_window"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not math.isfinite(args.duration_seconds) or not 0 < args.duration_seconds <= 3600:
        parser.error("duration must be finite and in (0, 3600]")
    if (args.inference_target_ms is None or not math.isfinite(args.inference_target_ms)
            or args.inference_target_ms <= 0):
        parser.error("inference target must be finite and positive")
    if args.adaptive_slo_window > 1_000_000:
        parser.error("adaptive window cannot exceed 1000000")
    if not 1 <= args.persistent_worker_blocks <= 65536:
        parser.error("persistent worker blocks must be in [1, 65536]")
    if args.workload == "cnn" and args.hidden_size != 1024:
        parser.error("CNN requires the default unused MLP hidden size")
    extent = (args.native_epoch_max_calls, args.native_epoch_prefix_calls,
              args.native_epoch_pending_prefixes, args.native_epoch_acceptance_us)
    if any(value is not None for value in extent):
        if not args.cooperative_preview or any(value is None for value in extent):
            parser.error("native epochs require cooperative preview and all four bounds")
        calls, width, pending, lifetime = extent
        if not (1 <= calls <= 65536 and 1 <= width <= calls and
                1 <= pending <= 16 and 1 <= lifetime <= 1000000):
            parser.error("native epoch extent exceeds protocol bounds")
    return args


def run_bounded(command: list[str], log: pathlib.Path, timeout: float) -> None:
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=benchmark_environment())
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            # This process group belongs only to the runner launched above.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f"workload failed with status {code}; inspect {log}")


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    if rows and any(workload_identity(row) != workload_identity(rows[0]) for row in rows):
        raise ValueError("comparison workload identities differ")
    if any(row.get("configuration") not in CONFIGURATIONS for row in rows):
        raise ValueError("unknown comparison configuration")
    expected_runs = None
    for configuration in CONFIGURATIONS:
        group = [row for row in rows if row["configuration"] == configuration]
        if not group:
            raise ValueError(f"no completed runs for {configuration}")
        if expected_runs is not None and len(group) != expected_runs:
            raise ValueError("comparison has unequal repetition counts")
        expected_runs = len(group)
        result = {"configuration": configuration, "runs": len(group),
                  **dict(zip(("workload", "workload_version", "cudnn_version"), workload_identity(group[0])))}
        for field in ("inference_p50_ms", "inference_p95_ms", "inference_p99_ms",
                      "inference_latency_target_miss_ratio", "training_steps_per_second"):
            values = [float(row[field]) for row in group]
            if any(not math.isfinite(value) or value < 0 or
                   (field.endswith("miss_ratio") and value > 1) for value in values):
                raise ValueError(f"invalid metric: {field}")
            result[field + "_median"] = statistics.median(values)
            result[field + "_min"] = min(values)
            result[field + "_max"] = max(values)
        output.append(result)
    native = output[0]
    if native["inference_p99_ms_median"] <= 0 or native["training_steps_per_second_median"] <= 0:
        raise ValueError("native baseline must have positive latency and throughput")
    for result in output:
        result["inference_p99_vs_native"] = (
            result["inference_p99_ms_median"] / native["inference_p99_ms_median"])
        result["training_throughput_vs_native"] = (
            result["training_steps_per_second_median"] / native["training_steps_per_second_median"])
    return output


def validate_run_set(rows: list[dict], names, repetitions: int) -> None:
    expected = {(name, repetition) for name in names for repetition in range(1, repetitions + 1)}
    observed = [(row["configuration"], int(row["repetition"])) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("missing, duplicate, or unexpected comparison repetitions")


def add_solo_baselines(result: list[dict], rows: list[dict], solo_rows: list[dict],
                       repetitions: int) -> list[dict]:
    if repetitions <= 0:
        raise ValueError("comparison repetitions must be positive")
    validate_run_set(rows, CONFIGURATIONS, repetitions)
    validate_run_set(solo_rows, SOLO_ROLES, repetitions)
    reference = rows[0]
    for row in [*rows, *solo_rows]:
        if workload_identity(row) != workload_identity(reference):
            raise ValueError("solo and colocated workload identities differ")
        if any(str(row[field]) != str(reference[field]) for field in MATCHED_FIELDS):
            raise ValueError("solo and colocated settings, identity, or versions differ")
    baselines = []
    for name, role in SOLO_ROLES.items():
        group = [row for row in solo_rows if row["configuration"] == name]
        if any(row["role"] != role for row in group):
            raise ValueError("solo baseline has the wrong role")
        baseline = {"role": role, "runs": len(group),
                    **dict(zip(("workload", "workload_version", "cudnn_version"), workload_identity(group[0])))}
        for field in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "steps_per_second"):
            values = [float(row[field]) for row in group]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"invalid solo metric: {field}")
            baseline[field + "_median"] = statistics.median(values)
            baseline[field + "_min"] = min(values)
            baseline[field + "_max"] = max(values)
        baselines.append(baseline)
    training, inference = baselines
    for comparison in result:
        comparison["inference_p99_vs_solo"] = (
            comparison["inference_p99_ms_median"] / inference["p99_ms_median"])
        comparison["training_throughput_vs_solo"] = (
            comparison["training_steps_per_second_median"] / training["steps_per_second_median"])
    return baselines


def write_aggregate(directory: pathlib.Path) -> None:
    if (directory / "failure.json").exists():
        raise ValueError("comparison has a recorded failure; inspect raw results")
    with (directory / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row.get("measurement_version") != MEASUREMENT_VERSION for row in rows):
        raise ValueError("summary must contain wall_clock_v2 runs")
    result = aggregate(rows)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    requested_workload = manifest.get("arguments", {}).get("workload", "mlp")
    if requested_workload not in WORKLOAD_VERSIONS or any(
            workload_identity(row)[0] != requested_workload for row in rows):
        raise ValueError("manifest and measured workload identities differ")
    solo_path = directory / "solo_summary.csv"
    if manifest.get("comparison_version") == COMPARISON_VERSION:
        with solo_path.open(newline="") as stream:
            solo_rows = list(csv.DictReader(stream))
        baselines = add_solo_baselines(result, rows, solo_rows,
                                       int(manifest["arguments"]["repetitions"]))
        (directory / "solo_aggregate.json").write_text(json.dumps(baselines, indent=2) + "\n")
        print("native_solo role p99_ms steps_per_second")
        for baseline in baselines:
            print(f"native_solo {baseline['role']} {baseline['p99_ms_median']:.3f} "
                  f"{baseline['steps_per_second_median']:.3f}")
    elif "comparison_version" in manifest or solo_path.exists():
        raise ValueError("solo comparison manifest is missing or incompatible")
    else:
        print("solo_baselines=unavailable legacy_comparison=1")
    (directory / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"workload={requested_workload}")
    print("configuration inference_p99_ms miss_ratio training_steps_per_second")
    for row in result:
        print(f"{row['configuration']:13} {row['inference_p99_ms_median']:.3f} "
              f"{row['inference_latency_target_miss_ratio_median']:.4f} "
              f"{row['training_steps_per_second_median']:.3f}")


def build_command(name: str, args: argparse.Namespace, directory: pathlib.Path) -> list[str]:
    workload = ["--duration-seconds", str(args.duration_seconds),
                "--workload", getattr(args, "workload", "mlp")]
    for option in ("warmup", "batch_size", "hidden_size"):
        workload += ["--" + option.replace("_", "-"), str(getattr(args, option))]
    if name in SOLO_ROLES:
        role = SOLO_ROLES[name]
        return [str(EXAMPLE / "run_pytorch_smoke.sh"), "--mode", "native", "--role", role,
                "--python", args.python, "--output", str(directory / f"{role}.csv"),
                "--", *workload, "--work-units", str(args.training_work_units if role == "training" else 1),
                "--latency-target-ms", str(args.inference_target_ms if role == "inference" else 0),
                "--status-file", str(directory / f"{role}.status")]
    command = [str(EXAMPLE / "run_pytorch_priority.sh"), *CONFIGURATIONS[name],
               "--output-dir", str(directory), "--build-dir", str(args.build_dir.resolve()),
               "--python", args.python, *workload,
               "--training-work-units", str(args.training_work_units),
               "--inference-target-ms", str(args.inference_target_ms),
               "--max-concurrent-kernels", "2", "--priority-threshold", "100",
               "--training-launch-batch-size", "1", "--inference-launch-batch-size", "1"]
    if name == "adaptive":
        command += ["--adaptive-slo-target-queue-us", str(args.adaptive_slo_target_queue_us),
                    "--adaptive-slo-window", str(args.adaptive_slo_window)]
    if name != "native" and getattr(args, "cooperative_preview", False):
        command += ["--cooperative-preview", "--persistent-worker-blocks",
                    str(args.persistent_worker_blocks)]
        if getattr(args, "native_epoch_max_calls", None) is not None:
            for option in ("max_calls", "prefix_calls", "pending_prefixes", "acceptance_us"):
                command += ["--native-epoch-" + option.replace("_", "-"),
                            str(getattr(args, "native_epoch_" + option))]
    return command


def main() -> int:
    args = parse_args()
    if args.analyze_only is not None:
        try:
            write_aggregate(args.analyze_only)
        except (OSError, ValueError, KeyError) as error:
            print(f"slo_validation status=failed error={error}", file=sys.stderr)
            return 1
        return 0
    try:
        output = prepare_output("pytorch-slo-validation", args.output_dir)
    except (OSError, ValueError) as error:
        print(f"slo_validation status=failed error={error}", file=sys.stderr)
        return 2
    print(f"slo_validation output={output}", flush=True)
    run_id = "preflight"
    try:
        artifacts = [REPOSITORY / "scripts/artifact_paths.py",
                     args.build_dir / "lib/libglimmer_cuda_interceptor.so",
                     args.build_dir / "bin/glimmer_control_service",
                     *(EXAMPLE / name for name in ("pytorch_smoke.py", "measurements.py",
                        "run_pytorch_smoke.sh", "run_pytorch_priority.sh", "run_slo_validation.py"))]
        manifest = {"measurement_version": MEASUREMENT_VERSION, "comparison_version": COMPARISON_VERSION,
                    "arguments": {key: str(value) if isinstance(value, pathlib.Path) else value
                                  for key, value in vars(args).items()},
                    "trace_timings": False, "trace_scheduler": False, "trace_cudnn": False,
                    "configurations": CONFIGURATIONS,
                    "solo_roles": SOLO_ROLES,
                    "environment": {key: os.environ.get(key) for key in (
                        "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING", "CUDA_MODULE_LOADING",
                        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NVIDIA_TF32_OVERRIDE", "PYTORCH_CUDA_ALLOC_CONF")},
                    "artifacts_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in artifacts},
                    "started_unix_ns": time.time_ns()}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        expected_identity = None
        with ((output / "summary.csv").open("w", newline="") as stream,
              (output / "solo_summary.csv").open("w", newline="") as solo_stream):
            writer, solo_writer = None, None
            names = [*SOLO_ROLES, *CONFIGURATIONS]
            for repetition in range(1, args.repetitions + 1):
                # Rotate ordering to reduce systematic warmup/thermal order bias.
                offset = (repetition - 1) % len(names)
                for name in names[offset:] + names[:offset]:
                    run_id = f"{name}-r{repetition}"
                    directory = output / run_id
                    directory.mkdir()
                    command = build_command(name, args, directory)
                    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
                    print(f"slo_validation run={run_id}", flush=True)
                    run_bounded(command, directory / "runner.log", args.duration_seconds + 180)
                    role = SOLO_ROLES.get(name, "inference")
                    framework = read_metadata(directory / f"{role}.status")
                    actual_workload = workload_identity(framework)
                    if actual_workload[:2] != (args.workload, WORKLOAD_VERSIONS[args.workload]):
                        raise ValueError("worker did not execute the requested workload")
                    measurement = (summarize_solo(directory, role) if name in SOLO_ROLES else
                                   summarize_colocation(directory, minimum_coverage=0.95))
                    start_ns = measurement["window_start_ns"]
                    end_ns = measurement["window_end_ns"]
                    if end_ns - start_ns < args.duration_seconds * 0.95 * 1_000_000_000:
                        raise ValueError("measurement window is shorter than the configured duration")
                    identity = tuple(framework[key] for key in
                                     ("device_uuid", "device_index", "torch_version", "cuda_version"))
                    if expected_identity is not None and identity != expected_identity:
                        raise ValueError("GPU identity or framework versions changed between runs")
                    expected_identity = identity
                    row = {"run_id": run_id, "configuration": name, "repetition": repetition,
                           "measurement_version": MEASUREMENT_VERSION, "device_uuid": framework["device_uuid"],
                           "device_index": framework["device_index"], "warmup": args.warmup,
                           "torch_version": framework["torch_version"], "cuda_version": framework["cuda_version"],
                           "duration_seconds": args.duration_seconds, "tensor_batch_size": args.batch_size,
                           "hidden_size": args.hidden_size, "training_work_units": args.training_work_units,
                           "inference_target_ms": args.inference_target_ms}
                    row.update(zip(("workload", "workload_version", "cudnn_version"), actual_workload))
                    if name in SOLO_ROLES:
                        row.update(role=role, **measurement)
                        if solo_writer is None:
                            solo_writer = csv.DictWriter(solo_stream, fieldnames=row.keys())
                            solo_writer.writeheader()
                        solo_writer.writerow(row)
                        solo_stream.flush()
                        continue
                    row["overlap_ms"] = measurement["overlap_ms"]
                    for role in ("training", "inference"):
                        for key, value in measurement[role].items():
                            row[f"{role}_{key}"] = value
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=row.keys())
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
        if any(hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest() != digest
               for path, digest in manifest["artifacts_sha256"].items()):
            raise ValueError("a runtime artifact or measurement source changed during the experiment")
        write_aggregate(output)
        print(f"slo_validation status=ok measurements_complete=1 scheduling_benefit=not_asserted output={output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        (output / "failure.json").write_text(json.dumps(
            {"status": "failed", "run_id": run_id, "error": str(error)}, indent=2) + "\n")
        print(f"slo_validation status=failed error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
