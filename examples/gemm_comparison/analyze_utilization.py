"""Offline, fail-closed review of the bounded GEMM utilization experiment."""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.artifact_paths import create_attempt, resolve_artifact_root

EXPERIMENT = PROJECT / "build/cutlass-adoption-1"
MODES = ("native", "ordinary", "persistent")
VARIANTS = ("baseline", "split-audit", "64x32", "32x64")
TIMING_ENVIRONMENT = {"PATH": "/usr/local/cuda-12.6/bin:/usr/bin:/bin",
                      "LD_LIBRARY_PATH": "/usr/local/cuda-12.6/lib64", "CUDA_VISIBLE_DEVICES": "0",
                      "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


def inputs(name):
    """Never import or execute a path/command obtained from an evidence report."""
    if name == "baseline":
        return (EXPERIMENT / "utilization-baseline/source/run_comparison.py",
                EXPERIMENT / "utilization-baseline/binary/glimmer_gemm_comparison")
    if name == "split-audit":
        return (Path(__file__).with_name("run_comparison.py"),
                EXPERIMENT / "root/examples/gemm_comparison/glimmer_gemm_comparison")
    if name in VARIANTS[2:]:
        folder = EXPERIMENT / ("geometry-" + name)
        return folder / "run_comparison.py", folder / "glimmer_gemm_comparison"
    raise ValueError("unknown experiment variant")


def load_runner(path):
    spec = importlib.util.spec_from_file_location("offline_gemm_collector", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def finite_number(value, label, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError("invalid " + label)
    return value


def spread(values):
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                per_run=list(values))


def validate_timing(path, runner, binary, capture_path, capture, require_normal=False):
    report = json.loads(path.read_text())
    flags = {"status": "standalone_timing_requires_review", "check_only": False,
             "collect_native_sm": False, "host_sanitizers": False, "device_tool": None,
             "timed_out": False, "returncode": 0, "coverage_policy": "require_all_sms",
             "full_device_coverage_verified": True, "native_full_sm_verified": True,
             "cutlass_compute_full_sm_verified": True,
             "performance_acceptance": False, "application_acceptance": False,
             "priority_acceptance": False, "full_card_acceptance": False}
    if any(type(report.get(key)) is not type(value) or report.get(key) != value
           for key, value in flags.items()):
        raise ValueError("incomplete, instrumented or unqualified timing receipt")
    if report.get("command") != [str(binary.resolve())]:
        raise ValueError("timing command is not the ordinary standalone executable")
    if report.get("environment") != TIMING_ENVIRONMENT:
        raise ValueError("timing environment differs from the clean ordinary collector")
    if report["binary_sha256"] != runner.digest(binary):
        raise ValueError("timing binary identity mismatch")
    collector = Path(runner.__file__)
    expected_source = runner.digest(collector.with_name("gemm_comparison.cu"))
    if report["source_sha256"] != expected_source:
        raise ValueError("timing source identity mismatch")
    if report["runner_sha256"] != runner.digest(collector):
        raise ValueError("timing collector identity mismatch")
    for key in ("binary_sha256", "source_sha256", "cutlass_revision", "vendor_identities"):
        if report[key] != capture[key]:
            raise ValueError("timing/capture identity mismatch: " + key)
    reference = report["native_sm_evidence"]
    if (Path(reference["path"]).resolve() != capture_path.resolve()
            or reference["sha256"] != runner.digest(capture_path)
            or reference["native_launches"] != 18):
        raise ValueError("timing capture reference changed")
    finished = finite_number(report["finished_at_unix_s"], "timing timestamp", True)
    wall = finite_number(report["wall_time_s"], "wall time", True)
    capture_finished = finite_number(capture["finished_at_unix_s"], "capture timestamp", True)
    if not 0 <= finished - capture_finished <= 86400 or capture_finished > finished - wall:
        raise ValueError("capture is stale or postdates timing")
    for inventory in (report["gpu_before"], report["gpu_after"]):
        if runner.device_identity(inventory) != runner.device_identity(capture["gpu_before"]):
            raise ValueError("timing GPU/driver mismatch")
    output = (path.parent / "stdout.log").read_text()
    application = runner.validate_application_output(output, False)
    if (application["compiled_source_sha256"] != expected_source
            or application["rows"] != report["rows"]):
        raise ValueError("timing raw output/source differs from retained summary")
    if require_normal and not all(row.get("persistent_audit_specialized") is True
                                  and row.get("normal_kernel_checks") is True
                                  for row in application["rows"]):
        raise ValueError("candidate normal-specialization correctness checks missing")
    if any(row.get("persistent_audit_specialized") for row in application["rows"]):
        if report.get("normal_compute_full_sm_verified") is not True:
            raise ValueError("normal compute qualification missing")
    for row in application["rows"]:
        for mode, samples in zip(MODES, row["gpu_samples_us"]):
            finite_number(row[mode + "_median_us"], "GPU median", True)
            for sample in samples:
                finite_number(sample, "GPU sample", True)
            # JSON serialization rounds floats; accept formatting error only.
            if not math.isclose(row[mode + "_median_us"], statistics.median(samples),
                                rel_tol=1e-7, abs_tol=1e-6):
                raise ValueError("reported median disagrees with raw samples")
    cpu = sum(finite_number(report[key], key) for key in
              ("child_user_cpu_s", "child_system_cpu_s"))
    return report, cpu


def legacy_compute_counters(runner, capture_path):
    """Report old full-work launches; never average retreat/resume controls."""
    application = runner.validate_application_output(
        (capture_path.parent / "stdout.log").read_text(), True, application_replay=True)
    common = load_runner(Path(__file__).with_name("run_comparison.py"))
    rows = common.metric_inventory((capture_path.parent / "native-metrics.csv").read_text())
    prefix = r"^(?:void )?(?:(?:\(anonymous namespace\)|<unnamed>)::)?"
    rows = [row for row in rows if not re.match(prefix + r"memset_kernel(?:<|\(|$)", row["Kernel Name"])]
    if len(rows) != 127 or not re.match(prefix + r"read_sm_id_extent\(", rows[0]["Kernel Name"]):
        raise ValueError("incomplete legacy operation/control inventory")
    patterns = (r"^volta_sgemm", r"^(?:void )?cutlass::Kernel<",
                prefix + r"persistent_gemm<", prefix + r"audited_ordinary_gemm<")
    positions = (0, 1, 2, 3, 2, 2, 2)
    result = []
    for operation in range(18):
        case = application["rows"][operation // 2]
        group = rows[1 + operation * 7:1 + (operation + 1) * 7]
        for position, (row, pattern_index) in enumerate(zip(group, positions)):
            if not re.match(patterns[pattern_index], row["Kernel Name"]):
                raise ValueError("legacy operation/control launch order changed")
            if position != 0:
                grid = case["logical_tiles"] if position in (1, 3) else 1 if position == 5 else case["workers"]
                if (runner.metric_number(row["launch__grid_size"]) != grid
                        or runner.metric_number(row["launch__block_size"]) != case["threads"]):
                    raise ValueError("legacy launch geometry changed")
        for position, mode in enumerate(("native", "ordinary", "legacy_persistent_audit")):
            row = group[position]
            cycles = runner.metric_number(row["sm__cycles_active.min"])
            if runner.metric_number(row["launch__sm_count"]) != 80 or cycles <= 0:
                raise ValueError("legacy complete launch has an idle SM")
            result.append(dict(case=case["case"], beta=operation % 2, mode=mode,
                               launch_id=row["ID"], kernel=row["Kernel Name"],
                               minimum_active_cycles=cycles, minimum_ffma_instructions=None,
                               registers_per_thread=runner.metric_number(row["launch__registers_per_thread"]),
                               utilization={key: runner.metric_number(row[key]) for key in runner.NATIVE_METRICS[2:]}))
    return result


def analyze_variant(name, paths, capture_path):
    paths = [Path(path).resolve(strict=True) for path in paths]
    if len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError("three distinct repetition receipts are required")
    collector, binary = inputs(name)
    runner = load_runner(collector)
    capture_path = Path(capture_path).resolve(strict=True)
    capture = json.loads(capture_path.read_text())
    current = dict(binary_sha256=runner.digest(binary),
                   source_sha256=runner.digest(collector.with_name("gemm_comparison.cu")),
                   cutlass_revision=runner.PINNED_CUTLASS,
                   vendor_identities=runner.vendor_identities(),
                   gpu_before=capture["gpu_before"])
    # Revalidates raw profiler/stdout/CSV digests and all full-compute launches.
    # This is offline: no GPU inventory or subprocess operation is invoked.
    runner.verify_native_evidence(capture_path, current)
    runs = [validate_timing(path, runner, binary, capture_path, capture, name != "baseline") for path in paths]
    counters = (legacy_compute_counters(runner, capture_path) if name == "baseline" else capture["compute_rows"])
    for row in counters:
        registers = finite_number(row["registers_per_thread"], "registers per thread", True)
        if registers != int(registers):
            raise ValueError("fractional register count")
        for value in row["utilization"].values():
            finite_number(value, "utilization counter")
    cases = []
    for index, case in enumerate(runner.CASES):
        m, n, k = runner.CASES[case]
        times = {mode: spread([run[0]["rows"][index][mode + "_median_us"] for run in runs])
                 for mode in MODES}
        cases.append(dict(case=case, dimensions=[m, n, k], timing_us=times,
                          nominal_gemm_gflops_s={mode: 2 * m * n * k / (times[mode]["median"] * 1000)
                                                for mode in MODES}))
    return dict(name=name, receipts=[dict(path=str(path), sha256=runner.digest(path),
                                         stdout_sha256=runner.digest(path.parent / "stdout.log"))
                                   for path in paths],
                capture=dict(path=str(capture_path), sha256=runner.digest(capture_path)),
                identity=current, cases=cases, cpu_seconds=spread([run[1] for run in runs]),
                wall_seconds=spread([run[0]["wall_time_s"] for run in runs]),
                native_counters=capture["native_rows"],
                full_compute_counters=counters,
                counter_contract=("legacy_native_and_audited_masks" if name == "baseline"
                                  else "independent_normal_and_audit_ffma"))


def compare_case(baseline, candidate):
    base = baseline["timing_us"]["persistent"]
    trial = candidate["timing_us"]["persistent"]
    if trial["maximum"] < base["minimum"]:
        observation = "nonoverlapping_faster"
    elif trial["minimum"] > base["maximum"]:
        observation = "nonoverlapping_slower"
    else:
        observation = "overlapping_requires_review"
    return dict(case=candidate["case"], observation=observation,
                persistent_speedup=base["median"] / trial["median"],
                native_control_ratio=baseline["timing_us"]["native"]["median"] /
                                     candidate["timing_us"]["native"]["median"])


def analyze_reports(path, names, required_status):
    batch = json.loads(path.read_text())
    if (batch.get("status") != required_status
            or batch.get("ordinary_repetitions") != 3
            or set(batch.get("captures", {})) != set(names)
            or set(batch.get("timing", {})) != set(names)):
        raise ValueError("incomplete requested comparison; no comparison accepted")
    receipts = [Path(item).resolve(strict=True) for paths in batch["timing"].values() for item in paths]
    if len(set(receipts)) != 3 * len(names):
        raise ValueError("duplicate/missing timing receipts across variants")
    result = [analyze_variant(name, batch["timing"][name], batch["captures"][name]) for name in names]
    reference = result[0]["identity"]
    runner = load_runner(inputs("baseline")[0])
    for candidate in result[1:]:
        identity = candidate["identity"]
        if (identity["vendor_identities"] != reference["vendor_identities"]
                or identity["cutlass_revision"] != reference["cutlass_revision"]
                or runner.device_identity(identity["gpu_before"]) != runner.device_identity(reference["gpu_before"])):
            raise ValueError("unmatched GPU/driver/vendor across variants")
        candidate["comparison"] = [compare_case(base, trial)
                                   for base, trial in zip(result[0]["cases"], candidate["cases"])]
    return dict(status=("bounded_comparison_requires_review" if names == VARIANTS
                        else "qualified_pair_comparison_requires_review"), variants=result,
                pending_variants=[name for name in VARIANTS if name not in names],
                batch=dict(path=str(path.resolve()), sha256=runner.digest(path)))


def analyze_batch(path):
    return analyze_reports(path, VARIANTS, "captures_and_timings_require_review")


def analyze_qualified_pair(path):
    # An explicit intermediate view, never a substitute for the four-variant gate.
    return analyze_reports(path, VARIANTS[:2], "qualified_pair_requires_review")


def render(result):
    lines = ["# Offline GEMM utilization review", "",
             "No automatic performance, saturation, priority or goal acceptance.", "",
             "Timing is beta=0. Counters include beta=0/1 and are separate from timing.",
             "CPU cost covers the entire child, including initialization and validation.",
             "Three repetitions are summarized as per-run medians/ranges, not 63 independent trials.",
             "Nonoverlapping ranges are descriptive, not a statistical significance claim.", ""]
    for variant in result["variants"]:
        lines.extend(["## " + variant["name"], "",
                      "Counter contract: " + variant["counter_contract"] + ".", "",
                      "Whole-child CPU seconds: " + str(variant["cpu_seconds"]) + ".", "",
                      "| Case | Native us | Ordinary us | Persistent us (min--max) | Speedup vs baseline | Observation |",
                      "| --- | ---: | ---: | ---: | ---: | --- |"])
        comparisons = {row["case"]: row for row in variant.get("comparison", [])}
        for row in variant["cases"]:
            timing = row["timing_us"]
            trial = timing["persistent"]
            comparison = comparisons.get(row["case"], {})
            speed = comparison.get("persistent_speedup")
            lines.append(f"| {row['case']} | {timing['native']['median']:.3f} | "
                         f"{timing['ordinary']['median']:.3f} | {trial['median']:.3f} "
                         f"({trial['minimum']:.3f}--{trial['maximum']:.3f}) | "
                         f"{format(speed, '.4f') if speed is not None else 'n/a'} | "
                         f"{comparison.get('observation', 'baseline only')} |")
        lines.extend(["", "Per-case native controls, nominal GEMM FLOP rates and every counter record",
                      "are retained in report.json; missing legacy normal FFMA evidence stays missing.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-report", type=Path)
    group.add_argument("--qualified-pair-report", type=Path)
    group.add_argument("--baseline-timing", type=Path, nargs=3)
    parser.add_argument("--capture-report", type=Path)
    parser.add_argument("--output-root", type=Path,
                        help="dedicated output root; defaults to artifacts/runs/gemm-analysis")
    args = parser.parse_args()
    if bool(args.baseline_timing) != bool(args.capture_report):
        parser.error("--capture-report is required only with --baseline-timing")
    try:
        artifact_root = resolve_artifact_root("gemm-analysis", output_root=args.output_root)
    except ValueError as error:
        parser.error(str(error))
    folder = create_attempt(artifact_root)
    result = dict(status="failed", full_goal_acceptance=False, performance_acceptance=False,
                  saturation_acceptance=False, application_acceptance=False, priority_acceptance=False)
    try:
        if args.batch_report:
            result.update(analyze_batch(args.batch_report))
        elif args.qualified_pair_report:
            result.update(analyze_qualified_pair(args.qualified_pair_report))
        else:
            result.update(status="baseline_only_no_candidate_comparison",
                          variants=[analyze_variant("baseline", args.baseline_timing, args.capture_report)])
        (folder / "review.md").write_text(render(result))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result["status"] = "failed"
        result["error"] = str(error)
    (folder / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Analysis evidence:", folder)
    print(result["status"] + (": " + result["error"] if "error" in result else ""))
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
