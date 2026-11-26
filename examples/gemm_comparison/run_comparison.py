"""Retain bounded, idle-device GEMM evidence without production preload."""

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import re
import signal
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.artifact_paths import create_attempt, resolve_artifact_root

PINNED_CUTLASS = "f7b19de32c5d1f3cedfc735c2849f12b537522ee"
NCU = Path("/opt/nvidia/nsight-compute/2024.3.2/ncu")
# A separate group without setsid preserves the invoking terminal's sudo cache.
# Executing a tiny isolated helper avoids unsafe preexec_fn callbacks after fork.
SESSION_GROUP_EXEC = "import os,sys; os.setpgid(0,0); os.execv(sys.argv[1],sys.argv[1:])"
NATIVE_METRICS = ("launch__sm_count", "sm__cycles_active.min",
                  "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
                  "sm__warps_active.avg.pct_of_peak_sustained_active",
                  "sm__throughput.avg.pct_of_peak_sustained_elapsed")
COMPUTE_METRIC = "sm__sass_thread_inst_executed_op_ffma_pred_on.min"
CASES = {
    "forward_qkv": (768, 768, 768), "forward_expand": (3072, 768, 768),
    "forward_contract": (768, 768, 3072), "input_qkv": (768, 768, 768),
    "input_expand": (768, 768, 3072), "input_contract": (3072, 768, 768),
    "weight_qkv": (768, 768, 768), "weight_expand": (3072, 768, 768),
    "weight_contract": (768, 3072, 768),
}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def validate_output(output, check_only, diagnostic_only=False):
    records = [json.loads(line) for line in output.splitlines() if line.strip()]
    if len(records) != 10:
        raise ValueError("missing complete nine-case success receipt")
    scope = dict(records[-1])
    compiled_source = scope.pop("compiled_source_sha256", None)
    if (not isinstance(compiled_source, str) or len(compiled_source) != 64
            or any(character not in "0123456789abcdef" for character in compiled_source)):
        raise ValueError("missing embedded build source identity")
    if scope != {
        "status": "ok", "cases": 9, "diagnostic_only": True,
        "application_acceptance": False, "priority_acceptance": False,
    }:
        raise ValueError("missing complete nine-case success receipt")
    rows = records[:-1]
    if [row["case"] for row in rows] != list(CASES):
        raise ValueError("missing, reordered or duplicate GEMM case")
    for row in rows:
        if "persistent_audit_specialized" in row and (
                row["persistent_audit_specialized"] is not True or row.get("normal_kernel_checks") is not True):
            raise ValueError("normal persistent specialization lacks complete correctness checks")
        if (tuple(row[key] for key in ("m", "n", "k")) != CASES[row["case"]]
                or row["workers"] != 320 or row["threads"] != 128
                or row["sm_count"] != 80 or row["logical_tiles"] < 80
                or row["resident_blocks_per_sm"] * 80 < 320
                or row["samples_per_mode"] != (0 if check_only else 21)
                or row["ordinary_persistent_bitwise_equal"] is not True
                or row["retreat_resume"] is not True):
            raise ValueError("GEMM execution contract mismatch")
        error = row["max_native_abs_error"]
        if not math.isfinite(error) or error < 0:
            raise ValueError("invalid numerical error")
        for mode in ("native", "ordinary", "persistent"):
            duration = row[f"{mode}_median_us"]
            if not math.isfinite(duration) or duration < 0 or (not check_only and duration == 0):
                raise ValueError("invalid GPU timing")
        for mode in ("ordinary", "persistent"):
            count = row[f"{mode}_compute_sms"]
            if type(count) is not int or not 0 < count <= 80:
                raise ValueError("invalid compute SM participation")
            if not diagnostic_only and count != row["sm_count"]:
                raise ValueError("full-device gate failed: " + mode + " has partial compute SM coverage")
        samples = row["gpu_samples_us"]
        if len(samples) != 3:
            raise ValueError("missing timing modes")
        for mode in samples:
            if len(mode) != row["samples_per_mode"] or any(
                    not math.isfinite(value) or value <= 0 for value in mode):
                raise ValueError("incomplete or invalid raw GPU samples")
    return rows


def validate_application_output(output, check_only, diagnostic_only=False, application_replay=False):
    """Validate every replay, never select only the last successful receipt."""
    if application_replay:
        # NCU and the child share stdout: a complete progress line can interrupt
        # a buffered JSON write, even inside a key or number. Remove only this
        # exact profiler grammar, including its newline, before framing records.
        # Unknown/truncated messages remain and must not repair malformed JSON.
        output = re.sub(
            r'==PROF== Profiling "[^"\r\n]+" - [0-9]+: Application replay pass [1-9][0-9]*\r?\n',
            "", output)
    lines = [line for line in output.splitlines() if line.startswith("{")]
    if not application_replay:
        rows = validate_output("\n".join(lines), check_only, diagnostic_only)
    else:
        if not check_only:
            raise ValueError("application replay cannot qualify ordinary timing")
        if not lines or len(lines) % 10:
            raise ValueError("incomplete application replay success receipt")
        reference = None
        for start in range(0, len(lines), 10):
            group = lines[start:start + 10]
            rows = validate_output("\n".join(group), True, diagnostic_only)
            records = [json.loads(line) for line in group]
            if reference is not None and records != reference:
                raise ValueError("application replay results or source identity changed")
            reference = records
    return dict(rows=rows, application_runs=len(lines) // 10,
                compiled_source_sha256=json.loads(lines[-1])["compiled_source_sha256"])


def device_identity(inventory):
    fields = next(csv.reader([inventory], skipinitialspace=True))
    if len(fields) != 6 or not fields[2].startswith("GPU-"):
        raise ValueError("invalid single-device identity")
    return fields[1:4]  # Device name, UUID, and driver, not fluctuating utilization.


def vendor_identities():
    paths = [Path("/usr/local/cuda-12.6/lib64") / name
             for name in ("libcublas.so.12", "libcublasLt.so.12")]
    return {str(path.resolve(strict=True)): digest(path) for path in paths}


def metric_number(value):
    # Raw NCU export may append instance values after the aggregate.
    number = float(value.split(" (")[0].replace(",", ""))
    if not math.isfinite(number):
        raise ValueError("nonfinite SM counter")
    return number


def metric_inventory(output):
    lines = output.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith('"ID",')), None)
    if start is None:
        raise ValueError("missing raw native NCU table")
    inventory = [row for row in csv.DictReader(io.StringIO("\n".join(lines[start:])))
                 if row.get("ID", "").isdigit()]
    if len({row["ID"] for row in inventory}) != len(inventory):
        raise ValueError("duplicate launch in native capture")
    if [int(row["ID"]) for row in inventory] != sorted(int(row["ID"]) for row in inventory):
        raise ValueError("native capture launch order changed")
    return inventory


def native_metric_rows(output):
    inventory = metric_inventory(output)
    # Capture the entire program, not a name-filtered subset: an unrecognized
    # reduction/helper kernel might belong to a multi-kernel cuBLAS operation.
    # Only source-owned CUTLASS/audit helpers and SDK memset setup are excluded.
    support = re.compile(r"^(?:void )?(?:(?:\(anonymous namespace\)|<unnamed>)::)?"
                         r"(?:read_sm_id_extent|persistent_gemm|audited_ordinary_gemm|"
                         r"cutlass::Kernel|memset_kernel)(?:<|\(|$)")
    rows = []
    for row in inventory:
        name = row["Kernel Name"]
        if name.startswith("volta_sgemm"):
            rows.append(row)
        elif not support.match(name):
            raise ValueError("unclassified kernel in complete capture; native operation grouping requires review: " + name)
    # The unchanged check-only program makes exactly two native calls per case,
    # in CASES order (beta zero then one). Split-K/multiple-kernel algorithms need
    # explicit operation grouping; never silently accept a representative subset.
    if len(rows) != 2 * len(CASES) or len({row["ID"] for row in rows}) != len(rows):
        raise ValueError("expected all 18 native GEMM launches; subset/multi-kernel capture is unqualified")
    result = []
    for index, row in enumerate(rows):
        if (not row["Kernel Name"].startswith("volta_sgemm")
                or metric_number(row["launch__sm_count"]) != 80):
            raise ValueError("unmatched native kernel or SM domain")
        minimum = metric_number(row["sm__cycles_active.min"])
        if minimum <= 0:
            raise ValueError("native full-device gate failed: at least one SM was idle for an entire GEMM")
        result.append(dict(case=list(CASES)[index // 2], beta=index % 2,
                           kernel=row["Kernel Name"], sm_count=80,
                           minimum_active_cycles=minimum,
                           utilization={name: metric_number(row[name]) for name in NATIVE_METRICS[2:]}))
        if any(value < 0 for value in result[-1]["utilization"].values()):
            raise ValueError("negative native utilization counter")
    return result


def persistent_metric_rows(output, cases):
    """Bind full normal compute to its exact position, not retreat controls."""
    inventory = metric_inventory(output)
    prefix = r"^(?:void )?(?:(?:\(anonymous namespace\)|<unnamed>)::)?"
    setup = re.compile(prefix + r"memset_kernel(?:<|\(|$)")
    inventory = [row for row in inventory if not setup.match(row["Kernel Name"])]
    if len(inventory) != 1 + 22 * len(CASES):
        raise ValueError("incomplete split-audit operation/control capture")
    if not re.match(prefix + r"read_sm_id_extent\(", inventory[0]["Kernel Name"]):
        raise ValueError("missing SM identifier extent launch")
    patterns = {
        "native": r"^volta_sgemm",
        "ordinary": r"^(?:void )?cutlass::Kernel<",
        "ordinary_audit": prefix + r"audited_ordinary_gemm<",
        "audit": prefix + r"persistent_gemm<.*[, ](?:true|\(bool\)1|1)>\(",
        "normal": prefix + r"persistent_gemm<.*[, ](?:false|\(bool\)0|0)>\(",
    }
    modes = ("native", "ordinary", "audit", "ordinary_audit", "audit", "audit", "audit",
             "normal", "normal", "audit", "normal")
    result = []
    for operation in range(2 * len(CASES)):
        case = cases[operation // 2]
        group = inventory[1 + operation * 11:1 + (operation + 1) * 11]
        for position, (row, mode) in enumerate(zip(group, modes)):
            if not re.match(patterns[mode], row["Kernel Name"]):
                raise ValueError("split-audit launch sequence or specialization mismatch")
            if mode != "native":
                expected_grid = (case["logical_tiles"] if mode.startswith("ordinary")
                                 else 1 if position in (5, 9) else case["workers"])
                if (metric_number(row["launch__grid_size"]) != expected_grid
                        or metric_number(row["launch__block_size"]) != case["threads"]):
                    raise ValueError("split-audit launch geometry mismatch")
        # Only complete fresh-work launches qualify spatial coverage; pre-stop,
        # single-worker partial-stop and resumed launches are explicit controls.
        for position in (0, 1, 2, 7):
            row = group[position]
            minimum_compute = metric_number(row[COMPUTE_METRIC])
            minimum_cycles = metric_number(row["sm__cycles_active.min"])
            if (metric_number(row["launch__sm_count"]) != 80
                    or minimum_compute <= 0 or minimum_cycles <= 0):
                raise ValueError("full useful-compute SM gate failed for " + modes[position])
            utilization = {name: metric_number(row[name]) for name in NATIVE_METRICS[2:]}
            if any(value < 0 for value in utilization.values()):
                raise ValueError("negative compute utilization counter")
            result.append(dict(case=case["case"], beta=operation % 2, mode=modes[position],
                               kernel=row["Kernel Name"], launch_id=row["ID"],
                               minimum_ffma_instructions=minimum_compute,
                               minimum_active_cycles=minimum_cycles, utilization=utilization,
                               registers_per_thread=metric_number(row["launch__registers_per_thread"])))
    return result


def native_profile_command(binary, evidence, environment, sudo_ncu):
    command = [str(NCU), "--replay-mode", "application", "--app-replay-mode", "strict",
               "--clock-control", "none", "--cache-control", "none",
               "--section-folder", str(NCU.parent / "sections"),
               "--metrics", ",".join((*NATIVE_METRICS, COMPUTE_METRIC)), "--export", str(evidence / "native"),
               str(binary), "--check-only"]
    if sudo_ncu:
        # Elevate only the bounded profiling child with a clean environment.
        command = ["/usr/bin/sudo", "-n", "/usr/bin/timeout", "--signal=TERM",
                   "--kill-after=5s", "240s", "/usr/bin/env", "-i",
                   *(f"{key}={value}" for key, value in sorted(environment.items())), *command]
    return command


def verify_native_evidence(path, current):
    path = path.resolve(strict=True)
    report = json.loads(path.read_text())
    if (report.get("status") != "native_full_sm_coverage_verified"
            or report.get("collect_native_sm") is not True
            or report.get("check_only") is not True
            or report.get("native_full_sm_verified") is not True
            or report.get("cutlass_compute_full_sm_verified") is not True
            or report.get("coverage_policy") != "require_all_sms"
            or report.get("full_device_coverage_verified") is not True
            or report.get("host_sanitizers") is not False or report.get("device_tool") is not None
            or report.get("performance_acceptance") is not False
            or report.get("application_acceptance") is not False
            or report.get("priority_acceptance") is not False
            or report.get("returncode") != 0 or report.get("timed_out") is not False):
        raise ValueError("native full-device evidence did not pass its complete capture")
    for key in ("binary_sha256", "source_sha256", "cutlass_revision", "vendor_identities"):
        if report.get(key) != current[key]:
            raise ValueError("native full-device evidence identity mismatch: " + key)
    if device_identity(report["gpu_before"]) != device_identity(current["gpu_before"]):
        raise ValueError("native full-device evidence GPU/driver mismatch")
    if device_identity(report["gpu_after"]) != device_identity(current["gpu_before"]):
        raise ValueError("native full-device evidence post-capture GPU/driver mismatch")
    age = time.time() - report["finished_at_unix_s"]
    if not 0 <= age <= 86400:
        raise ValueError("native full-device evidence must be fresh (within 24 hours)")
    for name, key in (("native.ncu-rep", "capture_sha256"), ("native-metrics.csv", "metrics_sha256"),
                      ("stdout.log", "stdout_sha256")):
        if digest(path.parent / name) != report[key]:
            raise ValueError("native full-device raw evidence changed: " + name)
    output = (path.parent / "stdout.log").read_text()
    application = validate_application_output(output, True, application_replay=True)
    if application["application_runs"] != report.get("application_replay_passes", 1):
        raise ValueError("native application replay count differs from retained summary")
    if application["compiled_source_sha256"] != current["source_sha256"]:
        raise ValueError("native profile executable source differs")
    rows = native_metric_rows((path.parent / "native-metrics.csv").read_text())
    if rows != report["native_rows"]:
        raise ValueError("native metrics differ from the retained summary")
    if any(row.get("persistent_audit_specialized") for row in application["rows"]):
        compute = persistent_metric_rows((path.parent / "native-metrics.csv").read_text(), application["rows"])
        if report.get("normal_compute_full_sm_verified") is not True or compute != report.get("compute_rows"):
            raise ValueError("normal persistent useful-compute qualification differs or is missing")
    return dict(path=str(path), sha256=digest(path), native_launches=len(rows))


def run_bounded(command, environment, timeout_s, preserve_session=False):
    launched = (["/usr/bin/python3", "-I", "-c", SESSION_GROUP_EXEC, *command]
                if preserve_session else command)
    process = subprocess.Popen(
        launched, env=environment, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=not preserve_session,
    )
    try:
        output, errors = process.communicate(timeout=timeout_s)
        return process.returncode, output, errors, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            if not preserve_session:
                raise
            # The sudo command retains its own privileged timeout owner. An
            # ordinary collector cannot guarantee signals to a root-only group.
        try:
            output, errors = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError:
                if not preserve_session:
                    raise
            try:
                output, errors = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("owned child did not retire after timeout; GPU containment "
                                   "is unverified, do not start another experiment") from error
        return process.returncode, output, errors, True


def gpu_inventory(environment):
    def query(arguments):
        return subprocess.run(
            ["/usr/bin/nvidia-smi", *arguments], env=environment,
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    inventory = query([
        "--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if len(inventory.splitlines()) != 1 or "V100" not in inventory:
        raise ValueError("this initial experiment requires one V100")
    processes = query(["--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    if processes:
        raise ValueError(f"GPU compute processes present; no experiment started: {processes}")
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path,
                        help="default: artifacts/runs/gemm-comparison; build/ output is not supported")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--device-tool", choices=("memcheck", "synccheck"))
    parser.add_argument("--host-sanitizers", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="allow unqualified partial/missing SM coverage; no full-device comparison")
    parser.add_argument("--collect-native-sm", action="store_true",
                        help="untimed NCU coverage capture of every native GEMM")
    parser.add_argument("--sudo-ncu", action="store_true",
                        help="use pre-authenticated sudo only for the NCU capture")
    parser.add_argument("--native-sm-evidence", type=Path,
                        help="fresh matched complete native capture report required for default timing")
    arguments = parser.parse_args()
    if arguments.collect_native_sm:
        if arguments.device_tool or arguments.host_sanitizers or arguments.diagnostic_only or arguments.native_sm_evidence:
            parser.error("native coverage collection cannot mix diagnostic, sanitizer or imported-evidence modes")
        arguments.check_only = True
    if arguments.sudo_ncu and not arguments.collect_native_sm:
        parser.error("--sudo-ncu is only valid with --collect-native-sm")
    if arguments.device_tool and not arguments.check_only:
        parser.error("device instrumentation requires --check-only")
    if arguments.host_sanitizers and (not arguments.check_only or arguments.device_tool):
        parser.error("host instrumentation requires --check-only without a device tool")
    try:
        artifact_root = resolve_artifact_root(
            "gemm-comparison", output_root=arguments.output_root)
    except ValueError as error:
        parser.error(str(error))
    environment = {
        "PATH": "/usr/local/cuda-12.6/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.6/lib64",
        "CUDA_VISIBLE_DEVICES": "0", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    }
    if arguments.host_sanitizers:
        environment["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1:protect_shadow_gap=0"
        environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    binary = arguments.binary.resolve(strict=True)
    upstream = arguments.cutlass_root.resolve(strict=True)
    evidence = create_attempt(artifact_root)
    report = {
        "status": "failed", "diagnostic_only": True,
        "application_acceptance": False, "priority_acceptance": False,
        "performance_acceptance": False, "full_card_acceptance": False,
        "native_full_sm_verified": False,
        "normal_compute_full_sm_verified": False,
        "check_only": arguments.check_only, "device_tool": arguments.device_tool,
        "host_sanitizers": arguments.host_sanitizers,
        "collect_native_sm": arguments.collect_native_sm,
        "coverage_policy": "diagnostic_only" if arguments.diagnostic_only else "require_all_sms",
        "full_device_coverage_verified": False, "saturation_verified": False,
        "binary": str(binary), "binary_sha256": digest(binary),
        "source_sha256": digest(Path(__file__).with_name("gemm_comparison.cu")),
        "runner_sha256": digest(__file__), "environment": environment,
        "artifact_layout_version": 1,
        "artifact_helper_sha256": digest(PROJECT_ROOT / "scripts/artifact_paths.py"),
        "legacy_build_output": False,
    }
    print(f"Evidence directory: {evidence}", flush=True)
    try:
        revision = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(upstream), "status", "--porcelain", "--", "include"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if revision != PINNED_CUTLASS or dirty:
            raise ValueError("CUTLASS revision or clean include tree mismatch")
        report["cutlass_revision"] = revision
        build_inputs = json.loads(binary.with_name("build-inputs.json").read_text())
        if (build_inputs["source_sha256"] != report["source_sha256"]
                or build_inputs["cutlass_revision"] != revision
                or build_inputs["host_sanitizers"] is not arguments.host_sanitizers):
            raise ValueError("binary build-input identity differs from current source")
        report["build_inputs"] = build_inputs
        if not arguments.check_only and not arguments.diagnostic_only and not arguments.native_sm_evidence:
            raise ValueError("full-device timing requires --native-sm-evidence; use --diagnostic-only "
                             "only to retain unqualified operation diagnostics")
        report["gpu_before"] = gpu_inventory(environment)
        report["vendor_identities"] = vendor_identities()
        if arguments.native_sm_evidence:
            report["native_sm_evidence"] = verify_native_evidence(arguments.native_sm_evidence, report)
            report["native_full_sm_verified"] = True
            report["normal_compute_full_sm_verified"] = True
        command = [str(binary)] + (["--check-only"] if arguments.check_only else [])
        if arguments.collect_native_sm:
            report["ncu_sha256"] = digest(NCU)
            command = native_profile_command(binary, evidence, environment, arguments.sudo_ncu)
        if arguments.device_tool:
            command = [
                "/usr/local/cuda-12.6/bin/compute-sanitizer", "--tool", arguments.device_tool,
                "--report-api-errors", "all", "--error-exitcode", "1",
            ] + (["--leak-check", "full"] if arguments.device_tool == "memcheck" else []) + command
        report["command"] = command
        report["process_isolation"] = ("separate_group_same_terminal_session" if arguments.sudo_ncu
                                       else "separate_group_and_session")
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        started = time.monotonic()
        # The privileged child owns the 240s timeout and 5s kill grace. Allow its
        # monitor to settle before the ordinary controller's deadline expires.
        returncode, output, errors, timed_out = run_bounded(
            command, environment, 250 if arguments.sudo_ncu else 240,
            preserve_session=arguments.sudo_ncu)
        report["wall_time_s"] = time.monotonic() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        report["child_user_cpu_s"] = after.ru_utime - before.ru_utime
        report["child_system_cpu_s"] = after.ru_stime - before.ru_stime
        report["returncode"] = returncode
        report["timed_out"] = timed_out
        (evidence / "stdout.log").write_text(output)
        (evidence / "stderr.log").write_text(errors)
        report["gpu_after"] = gpu_inventory(environment)
        if device_identity(report["gpu_after"]) != device_identity(report["gpu_before"]):
            raise ValueError("GPU/driver changed during experiment")
        if returncode or timed_out:
            if arguments.sudo_ncu and "sudo:" in errors and "password is required" in errors:
                raise ValueError("sudo authentication was unavailable in the invoking terminal; "
                                 "run sudo -v and this collector in the same terminal; NCU did not start")
            raise ValueError("bounded GEMM process failed; logs retained")
        # Device-tool banners are not application JSON; require the tool's exit code too.
        application = validate_application_output(
            output, arguments.check_only, arguments.diagnostic_only, arguments.collect_native_sm)
        rows = application["rows"]
        if application["compiled_source_sha256"] != report["source_sha256"]:
            raise ValueError("executable embedded source identity differs from current source")
        report["rows"] = rows
        if not all(row.get("persistent_audit_specialized") is True
                   and row.get("normal_kernel_checks") is True for row in rows):
            raise ValueError("current candidate lacks normal-specialization correctness receipts")
        report["cutlass_compute_full_sm_verified"] = all(
            row["ordinary_compute_sms"] == row["persistent_compute_sms"] == 80 for row in rows
        )
        if arguments.collect_native_sm:
            report["application_replay_passes"] = application["application_runs"]
            capture = evidence / "native.ncu-rep"
            report["capture_sha256"] = digest(capture)
            import_command = [str(NCU), "--import", str(capture), "--csv", "--page", "raw",
                              "--print-units", "base", "--print-kernel-base", "demangled"]
            report["import_command"] = import_command
            imported = subprocess.run(import_command, env=environment, check=True,
                                      text=True, capture_output=True, timeout=30)
            (evidence / "native-metrics.csv").write_text(imported.stdout)
            report["metrics_sha256"] = digest(evidence / "native-metrics.csv")
            report["stdout_sha256"] = digest(evidence / "stdout.log")
            report["native_rows"] = native_metric_rows(imported.stdout)
            report["compute_rows"] = persistent_metric_rows(imported.stdout, rows)
            report["normal_compute_full_sm_verified"] = True
            report["native_full_sm_verified"] = True
        if digest(binary) != report["binary_sha256"]:
            raise ValueError("binary changed during experiment")
        if digest(Path(__file__).with_name("gemm_comparison.cu")) != report["source_sha256"]:
            raise ValueError("source changed during experiment")
        if vendor_identities() != report["vendor_identities"]:
            raise ValueError("vendor libraries changed during experiment")
        if arguments.collect_native_sm and digest(NCU) != report["ncu_sha256"]:
            raise ValueError("NCU changed during counter collection")
        if arguments.native_sm_evidence:
            if verify_native_evidence(arguments.native_sm_evidence, report) != report["native_sm_evidence"]:
                raise ValueError("native evidence changed during experiment")
        report["full_device_coverage_verified"] = (
            not arguments.diagnostic_only and report["cutlass_compute_full_sm_verified"]
            and report["native_full_sm_verified"] and report["normal_compute_full_sm_verified"])
        report["status"] = "standalone_correctness_passed"
        if arguments.collect_native_sm:
            report["status"] = "native_full_sm_coverage_verified"
        if not arguments.check_only:
            report["status"] = "standalone_timing_requires_review"
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        report["full_device_coverage_verified"] = False
    report["finished_at_unix_s"] = time.time()
    (evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["status"] + (": " + report["error"] if report["status"] == "failed" else ""))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
