#!/usr/bin/env python3
"""Fresh two-path numerical prerequisite for a separately reviewed CPU profile."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from artifact_paths import PROJECT_ROOT, create_attempt, resolve_artifact_root
import run_bert_candidate_validation as candidate
from run_rnn_v8_validation import digest, save, wait_for_idle

require = candidate.require
ENTRY = Path(__file__).resolve()
MODES = ("native", "equal")
FLAGS = dict(diagnostic_only=True, full_device_qualified=False, performance_qualified=False,
             saturation_qualified=False, priority_qualified=False, gpu_gap_causality_qualified=False)


def read(path):
    return json.loads(Path(path).read_text())


def authorization(plan, seconds):
    deadline = datetime.fromisoformat(plan["authorization_deadline_utc"])
    require(deadline.tzinfo is not None and
            (deadline - datetime.now(timezone.utc)).total_seconds() >= seconds,
            "insufficient remaining application authorization")


def validate_plan(plan):
    require(plan["modes"] == list(MODES) and plan["runtime_environment"] ==
            {"GLIMMER_EXECUTION_HOST_RANGE_SEEK": "1"}, "unchanged two-path runtime required")
    for key, relative in (("server", "bin/glimmer_execution_service"),
                          ("frontend", "lib/libglimmer_execution_interceptor.so")):
        path = Path(plan[key])
        require(path == PROJECT_ROOT / "build/candidate" / relative and
                digest(path) == plan[key + "_sha256"], "selected binary changed")
    require(digest(Path(plan["observer"])) == plan["observer_sha256"], "qualified observer changed")
    review = read(Path(plan["protocol_qualification"]) / "review.json")
    require(review["status"] == "standalone_controller_qualified" and
            review["application_ready"] is False, "standalone protocol qualification required")
    seal = read(plan["prior_build_seal"])["identities"]
    selected = {name: sha for name, sha in seal.items()
                if any(Path(name).is_relative_to(PROJECT_ROOT / prefix)
                       for prefix in ("src", "include"))}
    selected.update({plan[key]: plan[key + "_sha256"] for key in ("server", "frontend")})
    require(selected and all(seal.get(name) == sha == digest(Path(name))
                            for name, sha in selected.items()), "sealed current source/binary changed")
    return selected


def configure(plan):
    """One explicit helper instance; no changes to historical helper files."""
    common, h = candidate.load_helpers()
    h.training.SERVER = Path(plan["server"])
    h.training.PRELOAD = Path(plan["frontend"])
    original_environment = common.owned_environment

    def owned_environment(*args, **kwargs):
        result = original_environment(*args, **kwargs)
        require(not any(key in result for key in plan["runtime_environment"]), "inherited runtime override")
        return dict(result, **plan["runtime_environment"])

    common.owned_environment = owned_environment
    candidate.load_helpers = lambda: (common, h)
    return common, h


def validate_job(plan, job):
    candidate.normal_cleanup(job)
    environments = [job["application_environment"], *job.get("application_environments", {}).values()]
    for environment in environments:
        require(not any(key.startswith("CPUPROFILE") or key.startswith("GLIMMER_CPU_")
                        for key in environment), "numerical sampler must be absent")
        if job["mode"] == "native":
            require("LD_PRELOAD" not in environment and not any(
                key.startswith("GLIMMER_EXECUTION_") for key in environment), "native contamination")
        else:
            require(environment.get("LD_PRELOAD") == plan["frontend"] and
                    environment.get("GLIMMER_EXECUTION_HOST_RANGE_SEEK") == "1" and
                    "GLIMMER_EXECUTION_RPC_OUTPUT_DIR" not in environment, "equal frontend differs")
    if job["mode"] == "equal":
        command = job["server_command"]
        server_environment = job["server_environment"]
        require(command[0] == plan["server"] and not any(
            key.startswith("CPUPROFILE") or key.startswith("GLIMMER_CPU_") or
            key.startswith("LD_") and key != "LD_LIBRARY_PATH" for key in server_environment),
                "numerical service sampler must be absent")
        require(server_environment.get("LD_LIBRARY_PATH") ==
                str(PROJECT_ROOT / "examples/framework_workloads/pytorch_smoke/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib")
                + ":/usr/local/cuda-12.6/lib64", "service loader search path differs")
        for option, value in (("--persistent-workers", "320"), ("--gemm-tile-side", "32")):
            require(command.count(option) == 1 and command[command.index(option) + 1] == value,
                    "explicit worker geometry differs")
        require(not any(option in command for option in ("--host-stage-output", "--poll-journal-output",
                "--issue-depth-output", "--skip-idle-helper-plans", "--async-intake-burst")),
                "extra service observer or optimization")


def worker(attempt, mode):
    plan = read(attempt / "plan.json")
    authorization(plan, 800)
    validate_plan(plan)
    configure(plan)
    status = candidate.worker(attempt, mode)
    if status == 0:
        report = read(attempt / (mode + "-worker.json"))
        try:
            validate_job(plan, report["job"])
            require(report["reparse"] == dict(tensors=401, elements=217787909,
                full_tensor_recomparison=True, source_bound_complete_inference=True,
                first_command_handoff=False), "complete numerical scope required")
        except Exception as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}",
                          scoped_publication_valid=False)
            candidate.invalidate_publication(report["job"])
            save(attempt / (mode + "-worker.json"), report)
            return 1
    return status


def collect(plan_path):
    plan = read(plan_path)
    authorization(plan, 2300)
    selected = validate_plan(plan)
    with plan_path.with_name("numerical-launch.json").open("x") as stream:
        json.dump(dict(entry=str(ENTRY), plan=str(plan_path), plan_sha256=digest(plan_path)), stream)
    attempt = create_attempt(resolve_artifact_root("bert-direct-cpu-numerical"))
    print("Numerical evidence: " + str(attempt), flush=True)
    save(attempt / "plan.json", plan)
    report = dict(status="started", jobs=[], plan_source=str(plan_path), **FLAGS)
    paths = before = None
    try:
        require(not os.environ.get("LD_PRELOAD") and not os.environ.get("LD_AUDIT"), "clean collector")
        require(shutil.disk_usage(attempt).free >= 16 * 1024**3, "8GiB evidence plus8GiB reserve required")
        common, h = configure(plan)
        gpu = h.query("gpu", h.training.GPU_FIELDS)
        require("GPU-0db675c8-4623-fc5a-9e7b-1c237f625ce2" in gpu and "V100" in gpu,
                "identified V100 required")
        manifest = dict(gpu=gpu, startup_reference_bundle=str(common.STARTUP_INPUTS),
                        startup_reference_report_sha256=common.STARTUP_INPUTS_SHA)
        h.startup_reference, h.export = common.application_references(h, manifest)
        save(attempt / "manifest.json", manifest)
        paths = common.input_paths(h) | {ENTRY, Path(candidate.__file__).resolve(), plan_path,
            Path(plan["prior_build_seal"]), Path(plan["observer"]),
            Path(plan["protocol_qualification"]) / "review.json",
            PROJECT_ROOT / "scripts/artifact_paths.py", PROJECT_ROOT / "scripts/run_rnn_v8_validation.py",
            PROJECT_ROOT / "build/candidate/CMakeCache.txt", PROJECT_ROOT / "build/candidate/compile_commands.json"}
        paths.update(Path(name) for name in selected)
        before = h.training.manifest(paths)
        save(attempt / "inputs-before.json", before)
        report["idle_window"] = wait_for_idle(attempt)
        require(report["idle_window"]["achieved"], "idle window unavailable; no numerical run")
        for mode in MODES:
            authorization(plan, 840)
            h.training.require_idle()
            unit, result = candidate.launch_unit(attempt, mode, worker_script=ENTRY)
            validate_job(plan, result["job"])
            report["jobs"].append(dict(mode=mode, unit=unit, worker=result))
            candidate.check_output(attempt)
            save(attempt / "report.json", report)
        report["status"] = "fresh_native_equal_numerical_pair_requires_review"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if before is not None:
            try:
                after = h.training.manifest(paths)
                save(attempt / "inputs-after.json", after)
                report["input_identity_unchanged"] = before == after
                report["gpu_identity_after"] = h.query("gpu", h.training.GPU_FIELDS)
                require(before == after and report["gpu_identity_after"] == gpu, "frozen identity changed")
            except BaseException as error:
                report.update(status="failed", identity_error=str(error))
        save(attempt / "report.json", report)
    print(json.dumps(dict(status=report["status"], error=report.get("error"), attempt=str(attempt))), flush=True)
    return int(report["status"] == "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualify", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--worker-attempt", type=Path)
    parser.add_argument("--worker-mode", choices=MODES)
    args = parser.parse_args()
    if args.worker_attempt:
        require(args.worker_mode is not None, "worker mode required")
        return worker(args.worker_attempt.resolve(strict=True), args.worker_mode)
    if args.check:
        plan = read(args.check)
        selected = validate_plan(plan)
        common, h = configure(plan)
        require(h.training.SERVER == Path(plan["server"]) and h.training.PRELOAD == Path(plan["frontend"]),
                "actual helper binding differs")
        print(json.dumps(dict(status="source_binary_helper_binding_checked", identities=len(selected))))
        return 0
    if args.qualify:
        return collect(args.qualify.resolve(strict=True))
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
