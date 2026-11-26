#!/usr/bin/env python3
"""Reproduce bounded persistent MLP correctness, handoff, overhead and coverage.

Short solo measurements are not the later formal co-location performance gate.
Run on an otherwise idle GPU, without concurrent builds or other benchmarks.
"""

import argparse
from contextlib import closing
import hashlib
import json
import math
import os
import pathlib
import shutil
import sqlite3
import subprocess
import time

from analyze_cuda_profile import analyze_database
from measurements import read_workload, summarize_samples
from run_overhead_baseline import optimized_build
import run_persistent_mlp_handoff as handoff
from run_slo_validation import run_bounded


ROOT = handoff.ROOT
EXAMPLE = handoff.APPLICATION.parent
IDENTITY = ("device_uuid", "device_index", "torch_version", "cuda_version")
THREADS = {"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}


def settings(build, preload, trace=False):
    result = dict(THREADS)
    if preload:
        result.update(LD_PRELOAD=str(build / "lib/libglimmer_cuda_interceptor.so"),
                      GLIMMER_PERSISTENT_BLAS="1", GLIMMER_COOPERATIVE_DRIVER_PREVIEW="1",
                      GLIMMER_COOPERATIVE_RUNTIME_PREVIEW="1", GLIMMER_MEMORY_LIMIT_BYTES="4294967296",
                      GLIMMER_SCHEDULER_MODE="enforce", GLIMMER_SCHEDULER_POLICY="priority",
                      GLIMMER_MAX_CONCURRENT_KERNELS="1")
        if trace:
            result["GLIMMER_TRACE_KERNEL_LAUNCHES"] = "1"
    return result


def command_with_environment(command, environment):
    return ["/usr/bin/env", *(f"{key}={value}" for key, value in sorted(environment.items())),
            *(str(value) for value in command)]


def workload(args, output, role, iterations, preload, profile=False):
    command = [args.python, handoff.APPLICATION, "--role", role, "--workload", "mlp",
               "--device", "cuda", "--dtype", "float32", "--batch-size", "32",
               "--hidden-size", "1024", "--work-units", "2" if role == "training" else "1",
               "--warmup", "3", "--iterations", str(iterations),
               "--output", output / f"{role}.csv", "--status-file", output / f"{role}.status"]
    if preload:
        command += ["--expected-memory-total-bytes", "4294967296"]
    if profile:
        command.append("--profile-cuda")
    return command_with_environment(command, settings(args.build, preload, trace=profile))


def original_fraction(profile, expected_owned):
    if profile.get("trace_quality") != "no_reported_warnings":
        raise ValueError("profile collection warnings prevent a complete denominator")
    data = profile["kernel_duration_by_name"]
    if data["unknown_total_ns"] or not data["groups"]:
        raise ValueError("all actual kernel names must be known")
    owned_ns = owned_count = original_ns = original_count = 0
    for group in data["groups"]:
        name, duration, count = group["name"], group["total_ns"], group["count"]
        if (not isinstance(name, str) or not name or type(duration) is not int or duration <= 0
                or type(count) is not int or count <= 0):
            raise ValueError("invalid kernel duration or count")
        if name == "glimmer_persistent_sgemm":
            owned_ns += duration
            owned_count += count
        else:
            if "glimmer" in name.lower():
                raise ValueError("unexpected transformed kernel cannot be classified as original")
            original_ns += duration
            original_count += count
    total_ns, total_count = owned_ns + original_ns, owned_count + original_count
    if (owned_count != expected_owned or not original_count or not owned_ns
            or total_ns != data["total_ns"] or total_ns != profile["kernel_duration"]["total_ns"]
            or total_count != profile["kernel_duration"]["count"]):
        raise ValueError("kernel inventory or duration denominator does not reconcile")
    return {"denominator": "sum of all captured GPU kernel durations (overlap counted)",
            "total_kernel_ns": total_ns, "owned_persistent_ns": owned_ns,
            "remaining_original_ns": original_ns, "remaining_original_fraction": original_ns / total_ns,
            "total_kernel_launches": total_count, "owned_persistent_launches": owned_count,
            "remaining_original_launches": original_count,
            "remaining_original_launch_fraction": original_count / total_count,
            "scope": "warmed solo profile, not GPU busy time, wall time, or co-location coverage"}


def profile_command(args, directory, role, nsys):
    # Nsight 2024.5 captures GPU activities with its default API subset. Newer
    # full-API/event switches are not supported by the installed CUDA 12.6 tool.
    # Capture through process exit, then select the recorded warmed API interval.
    # Stopping collection at cudaProfilerStop loses host events on this tool.
    return [nsys, "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--export=sqlite",
            "--force-overwrite=false", f"--output={directory / 'timeline'}",
            *workload(args, directory, role, 5, True, profile=True)]


def analyze_warmed_database(database):
    full = analyze_database(database)
    if full["trace_quality"] != "no_reported_warnings":
        raise ValueError("whole-process collection has warnings; cannot assert complete warmed coverage")
    markers = database.execute(
        "SELECT a.start, a.end, s.value, a.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME a "
        "JOIN StringIds s ON s.id=a.nameId WHERE s.value LIKE 'cudaProfilerStart%' "
        "OR s.value LIKE 'cudaProfilerStop%'").fetchall()
    starts = [row for row in markers if row[2].startswith("cudaProfilerStart")]
    stops = [row for row in markers if row[2].startswith("cudaProfilerStop")]
    if (len(starts) != 1 or len(stops) != 1 or starts[0][3] != stops[0][3]
            or not 0 < starts[0][0] <= starts[0][1] < stops[0][0] <= stops[0][1]):
        raise ValueError("missing, duplicate or contradictory warmed profiler API boundaries")
    begin, end = starts[0][1], stops[0][0]
    # Never modify the retained Nsight artifact. Reuse the common analyzer on a
    # private in-memory copy; keep every diagnostic, including out-of-range ones.
    with closing(sqlite3.connect(":memory:")) as selected:
        database.backup(selected)
        tables = {row[0] for row in selected.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for kind in ("KERNEL", "MEMCPY", "MEMSET", "RUNTIME", "DRIVER"):
            table = f"CUPTI_ACTIVITY_KIND_{kind}"
            if table not in tables:
                continue
            if kind in {"KERNEL", "MEMCPY", "MEMSET"} and selected.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE (start < ? AND end > ?) OR (start < ? AND end > ?)",
                    (begin, begin, end, end)).fetchone()[0]:
                raise ValueError("GPU activity straddles warmed boundary; do not silently clip it")
            selected.execute(f"DELETE FROM {table} WHERE start < ? OR end > ?", (begin, end))
        result = analyze_database(selected)
    result["capture"] = {"mode": "whole process, then exact warmed Runtime profiler boundaries",
                         "begin_ns": begin, "end_ns": end, "whole_process_kernel_count": full["kernel_duration"]["count"],
                         "whole_process_diagnostics": full["profiler_diagnostics"],
                         "cuda_api_trace": "installed Nsight default API subset"}
    return result


def analyze_warmed(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        return analyze_warmed_database(database)


def overhead(native, preload):
    if any(native["identity"].get(key) != preload["identity"].get(key) for key in IDENTITY):
        raise ValueError("native/preload hardware or framework identities differ")
    if native["samples"] != preload["samples"] or native["samples"] <= 0:
        raise ValueError("native/preload measured work counts differ")
    fields = ("mean_ms", "steps_per_second", "process_elapsed_seconds")
    if any(not math.isfinite(value[field]) or value[field] <= 0
           for value in (native, preload) for field in fields):
        raise ValueError("invalid overhead measurement")
    return {"native": native, "preload": preload,
            "latency_overhead_percent": (preload["mean_ms"] / native["mean_ms"] - 1) * 100,
            "throughput_retention": preload["steps_per_second"] / native["steps_per_second"],
            "process_elapsed_ratio": preload["process_elapsed_seconds"] / native["process_elapsed_seconds"]}


def fingerprints(args):
    paths = [handoff.APPLICATION, handoff.NUMERICAL, pathlib.Path(handoff.__file__),
             pathlib.Path(__file__), EXAMPLE / "measurements.py", EXAMPLE / "analyze_cuda_profile.py",
             args.build / "lib/libglimmer_cuda_interceptor.so",
             args.build / "bin/glimmer_control_service", args.build / "bin/glimmer_control_client"]
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "formal_performance_acceptance": False, "commands": [],
              "overhead": {}, "remaining_original": {},
              "measurement_scope": "solo local-enforcement warmed full-step wall time; initialization/warmup excluded",
              "limitations": ["Short measurement; formal workload performance acceptance is not established.",
                              "Handoff concerns kernel tasks, not whole inference exclusivity.",
                              "Overhead uses local admission; remote transport and co-location costs are not measured here.",
                              "Primary-context process exit is not proof of explicit device frees; public fixtures check cleanup."]}
    try:
        if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "0":
            raise ValueError("CUDA_LAUNCH_BLOCKING must be disabled")
        report["build"] = optimized_build(args.build)
        report["fingerprints"] = fingerprints(args)
        nsys = shutil.which(args.nsys)
        if not nsys:
            raise ValueError("installed Nsight Systems is required; no dependency is installed automatically")
        report["nsys_version"] = subprocess.run([nsys, "--version"], check=True, capture_output=True,
                                                text=True, timeout=10).stdout.strip()

        def execute(command, directory, name="run", timeout=180):
            directory.mkdir(parents=True, exist_ok=True)
            report["commands"].append({"argv": [str(value) for value in command],
                                       "log": str(directory / f"{name}.log")})
            started = time.monotonic()
            run_bounded([str(value) for value in command], directory / f"{name}.log", timeout)
            return time.monotonic() - started

        # This suite checks native/public selection, errors and explicit cleanup.
        execute([args.ctest, "--test-dir", args.build, "--output-on-failure", "-V", "-R",
                 "^glimmer_persistent_(lt|sgemm)_(native|public)_gpu_test$"], output, "public")
        if "100% tests passed, 0 tests failed out of 4" not in (output / "public.log").read_text():
            raise ValueError("public cleanup test inventory is incomplete")
        report["public_cleanup_tests"] = 4
        native_dir = output / "native-numerical"
        execute(command_with_environment([args.python, handoff.NUMERICAL,
                "--output", native_dir / "numerical.json", "--snapshot", native_dir / "numerical.pt"],
                THREADS), native_dir)
        native_result = json.loads((native_dir / "numerical.json").read_text())
        if native_result.get("native_compared") is not False:
            raise ValueError("fresh reference is not a native-only result")
        handoff.validate_numerical(dict(native_result, native_compared=True),
                                   report["fingerprints"][str(handoff.APPLICATION.resolve())])
        report["native_numerical"] = native_result
        for name, reference in (("unchanged-handoff", None),
                                ("numerical-handoff", native_dir / "numerical.pt")):
            options = argparse.Namespace(output=output / name, build=args.build, python=args.python,
                                         numerical_reference=reference)
            if handoff.run(options) != 0:
                raise ValueError(f"{name} failed; inspect retained handoff.json")
            report[name] = json.loads((output / name / "handoff.json").read_text())

        observed_identity = None

        def check_identity(metadata):
            nonlocal observed_identity
            identity = {key: metadata[key] for key in IDENTITY}
            if observed_identity is not None and identity != observed_identity:
                raise ValueError("acceptance workloads changed device or framework identity")
            observed_identity = identity
            if metadata["device_uuid"] != report["unchanged-handoff"]["low_ready"]["device_uuid"]:
                raise ValueError("measurement GPU differs from actual handoff GPU")
            return identity

        for role in ("training", "inference"):
            report["overhead"][role] = []
            for repeat in range(3):
                pair = {}
                for mode in (("native", "preload") if repeat % 2 == 0 else ("preload", "native")):
                    directory = output / f"timing-{role}-{repeat + 1}-{mode}"
                    elapsed = execute(workload(args, directory, role, args.iterations, mode == "preload"), directory)
                    metadata, rows = read_workload(directory, role)
                    if len(rows) != args.iterations:
                        raise ValueError("timing iteration inventory differs")
                    if mode == "preload" and int(metadata["visible_memory_total_bytes"]) != 4294967296:
                        raise ValueError("preload quota view was not active")
                    pair[mode] = {**summarize_samples(rows, int(metadata["start_ns"]), int(metadata["end_ns"]), 0),
                                  "identity": check_identity(metadata), "process_elapsed_seconds": elapsed}
                report["overhead"][role].append(overhead(pair["native"], pair["preload"]))

        # Profiling is a separate diagnostic and never enters the overhead data.
        for role in ("training", "inference"):
            directory = output / f"profile-{role}"
            measured = 5
            command = profile_command(args, directory, role, nsys)
            execute(command, directory)
            metadata, rows = read_workload(directory, role, allow_profile=True)
            check_identity(metadata)
            if metadata.get("cuda_profile") != "1" or len(rows) != measured:
                raise ValueError("missing warmed profile range")
            profile = analyze_warmed(directory / "timeline.sqlite")
            (directory / "analysis.json").write_text(json.dumps(profile, indent=2) + "\n")
            text = (directory / "run.log").read_text()
            lt_per_step, sgemm_per_step = (4, 6) if role == "training" else (2, 0)
            if (text.count("persistent_lt_bias accepted=1") != (measured + 3) * lt_per_step
                    or text.count("persistent_sgemm accepted=1") != (measured + 3) * sgemm_per_step):
                raise ValueError("profiled public selection inventory differs")
            report["remaining_original"][role] = original_fraction(profile, measured * (lt_per_step + sgemm_per_step))
        if fingerprints(args) != report["fingerprints"]:
            raise ValueError("source or runtime artifacts changed during acceptance")
        report["performance_insufficient"] = any(pair["throughput_retention"] < 0.9
                                                  for pairs in report["overhead"].values() for pair in pairs)
        report["status"] = "ok"
    except (OSError, ValueError, KeyError, RuntimeError, sqlite3.Error, subprocess.SubprocessError) as error:
        report["error"] = str(error)
    finally:
        (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output / "acceptance.json"),
                      "error": report.get("error")}))
    return 0 if report["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--build", type=pathlib.Path, default=ROOT / "build/cuda-perf")
    parser.add_argument("--python", type=pathlib.Path, default=EXAMPLE / ".venv/bin/python")
    parser.add_argument("--nsys", default="/usr/local/cuda-12.6/bin/nsys")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if not 5 <= args.iterations <= 100:
        parser.error("short acceptance timing requires 5 to 100 measured steps")
    args.build = args.build.resolve()
    args.python = args.python.absolute()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
