#!/usr/bin/env python3
"""Run bounded, clean BERT solo intake trials; never claim transparent preemption."""

import argparse
import json
import os
import pathlib
import subprocess
import sys

from bert_contract import PROFILE, sha256, summarize, write_json

EXAMPLE = pathlib.Path(__file__).resolve().parent
ROOT = EXAMPLE.parents[2]
SMOKE = EXAMPLE.parent / "pytorch_smoke"
sys.path.insert(0, str(SMOKE))
from run_overhead_baseline import optimized_build
from run_slo_validation import run_bounded


def environment(mode, build):
    # Explicit passthrough also prevents credentials or inherited instrumentation
    # from becoming part of a subprocess command/timeout diagnostic.
    allowed = {"PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LANG", "LC_ALL"}
    result = {key: value for key, value in os.environ.items() if key in allowed}
    result.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                  TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  PYTHONNOUSERSITE="1")
    if mode != "native":
        result["LD_PRELOAD"] = str(build / "lib/libglimmer_cuda_interceptor.so")
    if mode == "cooperative":
        result.update(GLIMMER_PERSISTENT_BLAS="1", GLIMMER_COOPERATIVE_DRIVER_PREVIEW="1",
                      GLIMMER_PERSISTENT_WORKER_BLOCKS="320",
                      GLIMMER_COOPERATIVE_RUNTIME_PREVIEW="1", GLIMMER_MEMORY_LIMIT_BYTES="12884901888",
                      GLIMMER_SCHEDULER_MODE="enforce", GLIMMER_SCHEDULER_POLICY="priority",
                      GLIMMER_MAX_CONCURRENT_KERNELS="1")
    return result


def validate_result(result, role, iterations):
    if (result.get("status") != "ok" or result.get("role") != role
            or result.get("profile") != PROFILE[role] or result.get("iterations") != iterations
            or len(result.get("samples_ns", [])) != iterations
            or result.get("timing", {}).get("samples") != iterations
            or result.get("preemption_acceptance") is not False
            or result.get("performance_acceptance") is not False):
        raise ValueError("incomplete or mismatched BERT baseline evidence")
    expected = summarize(result["samples_ns"], result["measured_end_ns"] - result["measured_begin_ns"],
                         PROFILE[role]["batch_size"])
    if result["timing"] != expected:
        raise ValueError("timing summary does not match raw samples and interval")


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "mode": args.mode, "commands": [], "runs": [],
              "preemption_acceptance": False, "performance_acceptance": False,
              "diagnostic": args.vendor_log or args.snapshot or bool(args.reference_root)}
    try:
        if args.mode != "native":
            optimized_build(args.build)
            report["interceptor_sha256"] = sha256(args.build / "lib/libglimmer_cuda_interceptor.so")
        report["source_sha256"] = {name: sha256(EXAMPLE / name) for name in
                                   ("bert_workload.py", "bert_contract.py", "run_bert_baseline.py")}
        base = environment(args.mode, args.build)
        roles = tuple(PROFILE) if args.role == "both" else (args.role,)
        for repetition in range(1, args.repetitions + 1):
            for role in roles:
                name = f"{role}-{repetition}"
                output = args.output / name
                child_env = dict(base)
                if args.vendor_log:
                    child_env.update(CUBLAS_LOGINFO_DBG="1", CUBLAS_LOGDEST_DBG=str(args.output / f"{name}.blas.log"),
                                     CUBLASLT_LOG_LEVEL="5", CUBLASLT_LOG_FILE=str(args.output / f"{name}.lt.log"))
                command = [str(args.python), "-B", str(EXAMPLE / "bert_workload.py"),
                           "--assets", str(args.assets), "--output", str(output), "--role", role,
                           "--warmup", str(args.warmup), "--iterations", str(args.iterations)]
                if args.snapshot:
                    command.append("--snapshot")
                if args.reference_root:
                    command += ["--reference", str(args.reference_root / name)]
                observed_env = {key: value for key, value in child_env.items()
                                if key.startswith(("GLIMMER_", "CUBLAS", "OMP_", "MKL_", "OPENBLAS_", "HF_"))
                                or key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "TOKENIZERS_PARALLELISM", "TRANSFORMERS_OFFLINE"}}
                report["commands"].append({"argv": command, "environment": observed_env})
                # Reuse the existing process-group timeout/cleanup owner. env -i
                # ensures inherited instrumentation cannot leak into the child.
                wrapped = ["/usr/bin/env", "-i", *(f"{key}={value}" for key, value in child_env.items()), *command]
                run_bounded(wrapped, args.output / f"{name}.log", 600)
                result = json.loads((output / "result.json").read_text())
                validate_result(result, role, args.iterations)
                report["runs"].append({"name": name, "timing": result["timing"],
                                       "result_sha256": sha256(output / "result.json")})
        for name, expected in report["source_sha256"].items():
            if sha256(EXAMPLE / name) != expected:
                raise RuntimeError("BERT source changed during baseline run")
        if args.mode != "native" and sha256(args.build / "lib/libglimmer_cuda_interceptor.so") != report["interceptor_sha256"]:
            raise RuntimeError("interceptor binary changed during baseline run")
        report["status"] = "ok"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    finally:
        write_json(args.output / "baseline.json", report)
    print(json.dumps({"status": report["status"], "report": str(args.output / "baseline.json"),
                      "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--mode", choices=("native", "preload", "cooperative"), default="native")
    parser.add_argument("--role", choices=("both", *PROFILE), default="both")
    parser.add_argument("--python", type=pathlib.Path, default=SMOKE / ".venv/bin/python")
    parser.add_argument("--build", type=pathlib.Path, default=ROOT / "build/cuda-perf")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--reference-root", type=pathlib.Path)
    parser.add_argument("--vendor-log", action="store_true")
    args = parser.parse_args()
    if min(args.iterations, args.repetitions) <= 0 or args.warmup < 0:
        parser.error("iterations/repetitions must be positive; warmup must be nonnegative")
    for name in ("assets", "output", "build", "reference_root"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    args.python = args.python.absolute()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
