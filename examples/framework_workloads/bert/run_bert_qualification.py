#!/usr/bin/env python3
"""Check actual native/persistent BERT application paths before device qualification."""

import argparse
import json
import pathlib
import subprocess
import time

from bert_contract import PROFILE, SEED, sha256, validate_assets, write_json
from onnx_bert_workload import validate_export
from run_bert_baseline import EXAMPLE, ROOT, SMOKE, environment, optimized_build, validate_result
from run_bert_colocation import check_gpu
from run_bert_handoff import stop_group

CONTROLS = (("native", "training"), ("cooperative", "training"),
            ("native", "inference"), ("cooperative", "inference"))
SELECTIONS = {"persistent_sgemm": 432, "persistent_lt_bias": 216}


def validate_control(result, mode, role, log):
    validate_result(result, role, 2)
    if result.get("warmup") != 1 or result.get("seed") != SEED or result.get("dtype") != "float32":
        raise ValueError("the fixed application configuration changed")
    counts = {name: log.count(f"[glimmer] {name} accepted=1") for name in SELECTIONS}
    if role == "training":
        comparison = result.get("numerical_comparison", {})
        checks = comparison.get("checks", [])
        if (not isinstance(checks, list) or any(not isinstance(name, str) for name in checks)
                or len(checks) != 401 or len(set(checks)) != 401
                or not {"loss", "start_logits", "end_logits"} <= set(checks)
                or comparison.get("rtol") != 2e-4 or comparison.get("atol") != 2e-5):
            raise ValueError("complete unchanged 401-tensor training comparison is required")
        gradients = {name.removeprefix("gradient/") for name in checks if name.startswith("gradient/")}
        parameters = {name.removeprefix("parameter/") for name in checks if name.startswith("parameter/")}
        if len(gradients) != 199 or gradients != parameters:
            raise ValueError("training gradient/updated-parameter inventories differ")
        if mode == "cooperative" and counts != SELECTIONS:
            raise ValueError("the pinned training BLAS replacement inventory changed")
    elif (result.get("providers", [None])[0] != "CUDAExecutionProvider"
          or result.get("validation") != {"reference": "native PyTorch logits", "rtol": 2e-4,
                                          "atol": 2e-5, "checked_requests": 2}):
        raise ValueError("complete native-PyTorch ONNX output validation is required")
    if (mode == "native" or role == "inference") and any(counts.values()):
        raise ValueError("unexpected replacement in a native application path")
    return counts


def command(args, role, output):
    application = EXAMPLE / ("bert_workload.py" if role == "training" else "onnx_bert_workload.py")
    argv = [str(args.python), "-B", str(application), "--assets", str(args.assets),
            "--output", str(output), "--warmup", "1", "--iterations", "2"]
    return argv + (["--role", role, "--reference", str(args.reference)] if role == "training"
                   else ["--export", str(args.export)])


def mapped_vendor_libraries(pid):
    try:
        lines = pathlib.Path(f"/proc/{pid}/maps").read_text().splitlines()
    except FileNotFoundError:
        return set()
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[-1].startswith("/"):
            path = pathlib.Path(fields[-1])
            if path.name.startswith(("libcublas", "libcudart", "libcuda.so", "libonnxruntime")):
                paths.add(path.resolve(strict=True))
    return paths


def collect(args, mode, role, folder):
    folder.mkdir()
    env = environment(mode, args.build)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if mode == "cooperative":
        env["GLIMMER_TRACE_KERNEL_LAUNCHES"] = "1"
    report = {"status": "failed", "mode": mode, "role": role,
              "command": command(args, role, folder / "application"), "environment": env,
              "diagnostic_only": True, "full_card_acceptance": False,
              "performance_acceptance": False, "preemption_acceptance": False}
    vendor_paths = set()
    try:
        report["gpu_before"] = check_gpu(set())
        write_json(folder / "control.json", report)
        with (folder / "application.log").open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(report["command"], env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline, next_check = time.monotonic() + args.timeout, 0
                report["pid"] = process.pid
                write_json(folder / "control.json", report)
                while process.poll() is None:
                    vendor_paths.update(mapped_vendor_libraries(process.pid))
                    if time.monotonic() >= deadline:
                        raise TimeoutError("application correctness control timed out")
                    if time.monotonic() >= next_check:
                        check_gpu({process.pid})
                        next_check = time.monotonic() + 1
                    time.sleep(.1)
                report["returncode"] = process.returncode
            finally:
                stop_group(process)
        report["gpu_after"] = check_gpu(set())
        if report["returncode"]:
            raise RuntimeError("application failed; inspect application.log and result.json")
        result_path = folder / "application/result.json"
        result = json.loads(result_path.read_text())
        report["actual_blas_selections"] = validate_control(
            result, mode, role, (folder / "application.log").read_text())
        if (result.get("environment", {}).get("device_uuid", "").removeprefix("GPU-")
                != args.device_uuid.removeprefix("GPU-")):
            raise ValueError("application used a different GPU")
        if not any(path.name.startswith("libcublas.so") for path in vendor_paths):
            raise ValueError("actual application cuBLAS library identity was not observed")
        report["vendor_libraries"] = {str(path): sha256(path) for path in sorted(vendor_paths)}
        report["result_sha256"] = sha256(result_path)
        report["log_sha256"] = sha256(folder / "application.log")
        report["status"] = "ok"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(folder / "control.json", report)
    return report


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "controls": [], "diagnostic_only": True,
              "provider": "retained application-owned 16x16 PTX SGEMM; not standalone CUTLASS",
              "full_card_acceptance": False, "performance_acceptance": False,
              "preemption_acceptance": False, "expected_useful_compute_sms": 80,
              "persistent_worker_blocks": 320}
    try:
        report["build_configuration"] = optimized_build(args.build)
        assets = validate_assets(args.assets)
        validate_export(args.export, assets)
        reference = json.loads((args.reference / "result.json").read_text())
        if (reference.get("status") != "ok" or reference.get("role") != "training"
                or reference.get("profile") != PROFILE["training"]
                or reference.get("warmup") != 1 or reference.get("iterations") != 2
                or reference.get("snapshot_sha256") != sha256(args.reference / "snapshot.pt")):
            raise ValueError("a complete identity-bound native training reference is required")
        names = ("bert_contract.py", "bert_workload.py", "onnx_bert_workload.py", "export_bert.py",
                 "bert_colocation_worker.py", "run_bert_baseline.py", "run_bert_colocation.py",
                 "run_bert_handoff.py", "verify_bert_handoff.py", "run_bert_qualification.py")
        helpers = ("run_overhead_baseline.py", "run_slo_validation.py", "measurements.py",
                   "run_persistent_mlp_handoff.py")
        paths = [*(EXAMPLE / name for name in names), *(SMOKE / name for name in helpers),
                 args.python.resolve(strict=True),
                 args.reference / "result.json", args.reference / "snapshot.pt",
                 args.export / "export.json", args.export / "bert.onnx",
                 args.assets / "manifest.json", args.build / "lib/libglimmer_cuda_interceptor.so"]
        report["fingerprints"] = {str(path): sha256(path) for path in paths}
        report["gpu_before"] = check_gpu(set())
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version",
                              "--format=csv,noheader"], check=True, capture_output=True,
                             text=True, timeout=10).stdout.strip()
        if len(gpu.splitlines()) != 1 or "Tesla V100" not in gpu:
            raise ValueError("this qualification requires the single fixed V100")
        args.device_uuid = gpu.split(",")[0].strip()
        report["gpu_identity"] = gpu
        for mode, role in CONTROLS:
            name = f"{mode}-{role}"
            report["active_control"] = name
            write_json(args.output / "qualification.json", report)
            print(f"Checking {name}; bounded to {args.timeout} seconds", flush=True)
            control = collect(args, mode, role, args.output / name)
            report["controls"].append({"name": name, "status": control["status"],
                "report_sha256": sha256(args.output / name / "control.json"),
                "actual_blas_selections": control["actual_blas_selections"]})
        if any(sha256(path) != digest for path, digest in report["fingerprints"].items()):
            raise ValueError("application, runtime or reference changed during qualification")
        report.pop("active_control")
        report["status"] = "correctness_and_selection_verified_full_device_pending"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(args.output / "qualification.json", report)
    print(json.dumps({key: report.get(key) for key in ("status", "error")}), flush=True)
    return 0 if report["status"] != "failed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "export", "reference", "output"):
        parser.add_argument(f"--{name}", type=pathlib.Path, required=True)
    parser.add_argument("--build", type=pathlib.Path, default=ROOT / "build/cuda-perf")
    parser.add_argument("--python", type=pathlib.Path, default=SMOKE / ".venv/bin/python")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 600:
        parser.error("timeout must be in [1, 600] seconds per control")
    for name in ("assets", "export", "reference", "output", "build"):
        setattr(args, name, getattr(args, name).resolve())
    args.python = args.python.absolute()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
