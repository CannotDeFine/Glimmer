#!/usr/bin/env python3
"""Capture a short, warmed solo CUDA timeline, separate from performance runs."""

import argparse
from contextlib import nullcontext
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys

from analyze_cuda_profile import analyze_sqlite
from measurements import read_workload
from run_overhead_baseline import (MODES, ROLES, artifact_hashes, optimized_build,
                                   parse_args as baseline_args, remote_service, workload_command)
from run_slo_validation import benchmark_environment, prepare_output, run_bounded


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Also accepts baseline options: --build-dir, --python, --duration-seconds, "
        "--warmup, --batch-size, --hidden-size, --training-work-units, --quota-bytes, --output-dir."))
    parser.add_argument("--mode", choices=MODES, default="local")
    parser.add_argument("--role", choices=ROLES, default="training")
    parser.add_argument("--nsys", default="nsys")
    parser.add_argument("--cooperative-preview", action="store_true")
    parser.add_argument("--persistent-worker-blocks", type=int)
    parser.add_argument("--cuda-api-trace", choices=("full", "default"), default="full",
                        help="explicitly allow selected default APIs on older Nsight versions")
    options, remaining = parser.parse_known_args(argv)
    args = baseline_args(["--duration-seconds", "1", "--repetitions", "1", *remaining])
    if args.analyze_only is not None or args.repetitions != 1:
        parser.error("capture one diagnostic at a time; baseline analysis/repetitions do not apply")
    if args.duration_seconds > 10:
        parser.error("diagnostic duration is limited to 10 seconds")
    args.mode, args.role, args.nsys = options.mode, options.role, options.nsys
    args.cuda_api_trace = options.cuda_api_trace
    if options.cooperative_preview and options.mode not in ("local", "remote"):
        parser.error("cooperative preview requires local or remote mode")
    if options.persistent_worker_blocks is not None and not options.cooperative_preview:
        parser.error("persistent workers require explicit cooperative preview")
    workers = 3 if options.persistent_worker_blocks is None else options.persistent_worker_blocks
    if not 1 <= workers <= 65536:
        parser.error("persistent workers must be in [1, 65536]")
    args.cooperative_preview = options.cooperative_preview
    args.persistent_worker_blocks = workers
    return args


def profile_command(args, output, socket_path=""):
    # LD_PRELOAD belongs to the target's env command, not the Nsight process.
    target = workload_command(args.mode, args.role, args, output, socket_path)
    if args.cooperative_preview:
        target[1:1] = ["GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1",
                       "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1",
                       f"GLIMMER_PERSISTENT_WORKER_BLOCKS={args.persistent_worker_blocks}"]
    executable = shutil.which(target[0])
    if executable is None:
        raise ValueError(f"profile target executable is unavailable: {target[0]}")
    target[0] = str(pathlib.Path(executable).resolve())
    flags = (["--cuda-trace-all-apis=true", "--cuda-event-trace=false"]
             if args.cuda_api_trace == "full" else [])
    return [args.nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", *flags,
            "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--export=sqlite",
            "--force-overwrite=false", f"--output={output / 'timeline'}",
            *target, "--profile-cuda"]


def validate_workload(args, output):
    metadata, _ = read_workload(output, args.role, allow_profile=True)
    if (metadata.get("validation") != "1" or metadata.get("cuda_profile") != "1"
            or metadata.get("role") != args.role or int(metadata.get("iterations", "0")) <= 0
            or metadata.get("device_uuid", "unknown") in ("unknown", "cpu", "")):
        raise ValueError("missing or invalid profiled workload result")
    if args.mode in ("quota", "local", "remote"):
        if int(metadata["visible_memory_total_bytes"]) != args.quota_bytes:
            raise ValueError("profiled workload did not expose the configured quota")
    return metadata


def main(argv=None):
    args = parse_args(argv)
    output = None
    try:
        build = optimized_build(args.build_dir)
        if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
            raise ValueError("disable CUDA_LAUNCH_BLOCKING before profiling")
        executable = shutil.which(args.nsys)
        if executable is None:
            raise ValueError("Nsight Systems is unavailable; install it separately before capture")
        args.nsys = executable
        version = subprocess.run([args.nsys, "--version"], check=True, capture_output=True,
                                 text=True, timeout=10, env=benchmark_environment()).stdout.strip()
        output = prepare_output("pytorch-launch-profile", args.output_dir, category="profiles")
        manifest = {"version": "warmed_cuda_profile_v1", "diagnostic_only": True,
                    "collection_complete": False, "nsys_version": version, "build": build,
                    "arguments": {key: str(value) if isinstance(value, pathlib.Path) else value
                                  for key, value in vars(args).items()},
                    "environment": {key: value for key, value in benchmark_environment().items()
                                    if key.startswith(("CUDA_", "PYTORCH_", "OMP_", "MKL_", "NVIDIA_"))},
                    "artifacts_sha256": artifact_hashes(args)}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"launch_profile diagnostic_only=1 output={output}", flush=True)
        with remote_service(args, output) if args.mode == "remote" else nullcontext("") as socket_path:
            command = profile_command(args, output, socket_path)
            (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            run_bounded(command, output / "profile.log", args.duration_seconds + 180)
        workload = validate_workload(args, output)
        result = analyze_sqlite(output / "timeline.sqlite")
        if not (output / "timeline.nsys-rep").is_file():
            raise ValueError("Nsight report is missing")
        if artifact_hashes(args) != manifest["artifacts_sha256"]:
            raise ValueError("runtime artifacts or diagnostic sources changed during capture")
        result.update(mode=args.mode, role=args.role, workload=workload)
        (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
        manifest["collection_complete"] = True
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"launch_profile status=captured diagnostic_only=1 trace_quality={result['trace_quality']} "
              f"kernels={result['kernel_duration']['count']} "
              f"execution_coverage={result['execution']['coverage_fraction']:.4f} output={output}")
        for diagnostic in result["profiler_diagnostics"]:
            if diagnostic["severity"] == "Warning":
                print(f"launch_profile warning={diagnostic['message']}", file=sys.stderr)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error,
            subprocess.SubprocessError, KeyboardInterrupt) as error:
        if output is not None:
            (output / "failure.json").write_text(json.dumps({"error": str(error)}, indent=2) + "\n")
        print(f"launch_profile status=failed error={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
