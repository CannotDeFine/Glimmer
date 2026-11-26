#!/usr/bin/env python3
"""Collect complete warmed BERT GPU kernel inventories separately from timing."""

import argparse
import json
import pathlib
import shutil
import subprocess

from bert_contract import sha256, write_json
from run_bert_baseline import EXAMPLE, ROOT, SMOKE, environment, optimized_build, run_bounded, validate_result
from run_persistent_mlp_acceptance import analyze_warmed
from run_bert_qualification import check_gpu, validate_control


def coverage(profile, mode):
    if profile.get("trace_quality") != "no_reported_warnings":
        raise ValueError("collection warnings invalidate the kernel denominator")
    data = profile["kernel_duration_by_name"]
    if data["unknown_total_ns"] or not data["groups"]:
        raise ValueError("all actual kernel names must be known")
    owned_ns = owned_count = original_ns = original_count = 0
    for group in data["groups"]:
        name, duration, count = group["name"], group["total_ns"], group["count"]
        if (not isinstance(name, str) or not name or type(duration) is not int or duration <= 0
                or type(count) is not int or count <= 0):
            raise ValueError("invalid kernel duration or launch count")
        if name == "glimmer_persistent_sgemm":
            owned_ns += duration
            owned_count += count
        else:
            if "glimmer" in name.lower():
                raise ValueError("unexpected transformed kernel cannot be counted as original")
            original_ns += duration
            original_count += count
    total_ns, total_count = owned_ns + original_ns, owned_count + original_count
    if (total_ns != data["total_ns"] or total_ns != profile["kernel_duration"]["total_ns"]
            or total_count != profile["kernel_duration"]["count"]):
        raise ValueError("kernel inventory or duration denominator does not reconcile")
    if mode != "cooperative" and owned_count:
        raise ValueError("unexpected transformation in a native/default-off run")
    return {"denominator": "sum of all captured warmed GPU kernel durations, overlap counted",
            "total_kernel_ns": total_ns, "total_kernel_launches": total_count,
            "owned_persistent_ns": owned_ns, "owned_persistent_launches": owned_count,
            "original_ns": original_ns, "original_launches": original_count,
            "original_time_fraction": original_ns / total_ns,
            "original_launch_fraction": original_count / total_count,
            "scope": "diagnostic solo kernel inventory, not busy time, wall time, preemption or acceleration"}


def command(args, nsys):
    # Clear inherited state before the profiler starts, not after it injects
    # CUDA tracing into the target. A nested `env -i` erases that injection.
    profiler_env = environment("native", args.build)
    child_env = environment(args.mode, args.build)
    if getattr(args, "reference", None) is not None and args.mode == "cooperative":
        child_env["GLIMMER_TRACE_KERNEL_LAUNCHES"] = "1"
    application = EXAMPLE / ("bert_workload.py" if args.engine == "pytorch" else "onnx_bert_workload.py")
    target = [str(args.python), "-B", str(application), "--assets", str(args.assets),
              "--output", str(args.output / "application"), "--warmup", str(getattr(args, "warmup", 3)),
              "--iterations", "2", "--profile-cuda"]
    if args.engine == "pytorch":
        target += ["--role", args.role]
    else:
        target += ["--export", str(args.export)]
    if getattr(args, "reference", None) is not None:
        target += ["--reference", str(args.reference)]
    return ["/usr/bin/env", "-i", *(f"{key}={value}" for key, value in sorted(profiler_env.items())),
            nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--export=sqlite",
            "--force-overwrite=false", f"--output={args.output / 'timeline'}", "/usr/bin/env",
            *(f"{key}={value}" for key, value in sorted(child_env.items())), *target]


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "diagnostic_only": True, "preemption_acceptance": False,
              "performance_acceptance": False, "full_card_acceptance": False,
              "mode": args.mode, "engine": args.engine, "role": args.role}
    try:
        nsys = shutil.which(args.nsys)
        if not nsys:
            raise RuntimeError("Nsight Systems is required")
        report["gpu_before"] = check_gpu(set())
        if args.mode != "native":
            optimized_build(args.build)
            report["interceptor_sha256"] = sha256(args.build / "lib/libglimmer_cuda_interceptor.so")
        paths = list(EXAMPLE.glob("*.py"))
        paths += [SMOKE / name for name in ("run_persistent_mlp_acceptance.py", "analyze_cuda_profile.py")]
        if getattr(args, "reference", None) is not None:
            paths += [args.reference / "result.json", args.reference / "snapshot.pt"]
        identities = {path: sha256(path) for path in paths}
        report["source_sha256"] = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): digest
                                   for path, digest in identities.items()}
        run_bounded([nsys, "--version"], args.output / "nsys-version.log", 30)
        report["command"] = command(args, nsys)
        run_bounded(report["command"], args.output / "profile.log", 600)
        report["gpu_after"] = check_gpu(set())
        result = json.loads((args.output / "application/result.json").read_text())
        validate_result(result, args.role, 2)
        if result.get("instrumented") is not True:
            raise ValueError("application did not enable the declared profiler range")
        if getattr(args, "reference", None) is not None:
            report["actual_blas_selections"] = validate_control(
                result, args.mode, args.role, (args.output / "profile.log").read_text())
            report["numerical_checks"] = len(result["numerical_comparison"]["checks"])
        report["profile"] = analyze_warmed(args.output / "timeline.sqlite")
        report["coverage"] = coverage(report["profile"], args.mode)
        report["application_result_sha256"] = sha256(args.output / "application/result.json")
        report["sqlite_sha256"] = sha256(args.output / "timeline.sqlite")
        if any(sha256(path) != digest for path, digest in identities.items()):
            raise RuntimeError("profile sources changed during capture")
        if (args.mode != "native" and report["interceptor_sha256"]
                != sha256(args.build / "lib/libglimmer_cuda_interceptor.so")):
            raise RuntimeError("interceptor changed during capture")
        report["status"] = "ok"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    finally:
        write_json(args.output / "profile.json", report)
    print(json.dumps({"status": report["status"], "coverage": report.get("coverage"), "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("native", "preload", "cooperative"), default="native")
    parser.add_argument("--engine", choices=("pytorch", "onnx"), default="pytorch")
    parser.add_argument("--role", choices=("training", "inference"), default="training")
    parser.add_argument("--export", type=pathlib.Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--reference", type=pathlib.Path,
                        help="require the complete native training oracle in the captured application")
    parser.add_argument("--python", type=pathlib.Path, default=SMOKE / ".venv/bin/python")
    parser.add_argument("--build", type=pathlib.Path, default=ROOT / "build/cuda-perf")
    parser.add_argument("--nsys", default="nsys")
    args = parser.parse_args()
    if args.engine == "onnx" and (args.role != "inference" or args.export is None):
        parser.error("ONNX requires inference role and a pinned export directory")
    if args.warmup < 0 or (args.reference is not None and (args.engine != "pytorch" or args.role != "training")):
        parser.error("require nonnegative warmup and a training-only native reference")
    if args.reference is not None and args.warmup != 1:
        parser.error("the fixed 401-tensor qualification reference requires --warmup 1")
    for name in ("assets", "output", "build", "export", "reference"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.python = args.python.absolute()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
