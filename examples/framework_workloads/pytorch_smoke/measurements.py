"""Wall-clock measurement and common-window analysis; no PyTorch dependency."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import time


MEASUREMENT_VERSION = "wall_clock_v2"


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("measurement window contains no samples")
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def summarize_samples(rows: list[dict], start_ns: int, end_ns: int,
                      target_ms: float) -> dict:
    if end_ns <= start_ns or not math.isfinite(target_ms) or target_ms < 0:
        raise ValueError("invalid measurement window or latency target")
    # Boundary-crossing steps remain in the raw CSV, but only fully contained
    # steps contribute to common-window throughput and latency percentiles.
    selected = []
    for row in rows:
        begin, finish = int(row["iteration_start_ns"]), int(row["iteration_end_ns"])
        if begin <= 0 or finish <= begin:
            raise ValueError("invalid iteration timestamps")
        if begin >= start_ns and finish <= end_ns:
            selected.append((finish - begin) / 1_000_000)
    if not selected:
        raise ValueError("measurement window contains no complete steps")
    misses = sum(value > target_ms for value in selected) if target_ms else 0
    return {
        "samples": len(selected),
        "excluded_samples": len(rows) - len(selected),
        "mean_ms": statistics.fmean(selected),
        "p50_ms": percentile(selected, 50),
        "p95_ms": percentile(selected, 95),
        "p99_ms": percentile(selected, 99),
        "steps_per_second": len(selected) * 1_000_000_000 / (end_ns - start_ns),
        "latency_target_ms": target_ms,
        "latency_target_miss_count": misses,
        "latency_target_miss_ratio": misses / len(selected),
    }


def read_metadata(path: pathlib.Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def validate_scheduler_metrics(text: str) -> None:
    fields = text.split()
    if (len(fields) != 17 or fields[:2] != ["GLIMMER_TASK_V1", "METRICS"]
            or any(not value.isascii() or not value.isdecimal() for value in fields[2:])):
        raise ValueError("malformed scheduler metrics")
    total, queued, running, completed, cancelled, failed, _, reserved, allocated, *_ = map(int, fields[2:])
    if total == 0 or queued or running or cancelled or failed or reserved or allocated or completed != total:
        raise ValueError("scheduler tasks did not all complete cleanly with quota drained")


def read_workload(directory: pathlib.Path, role: str, *, allow_profile: bool = False) -> tuple[dict, list[dict]]:
    """Read validated CUDA measurements for either solo or paired analysis."""
    if role not in ("training", "inference"):
        raise ValueError("unknown workload role")
    record = read_metadata(directory / f"{role}.status")
    if not allow_profile and record.get("cuda_profile", "0") != "0":
        raise ValueError("profiled diagnostics must not be analyzed as benchmark measurements")
    if record.get("measurement_version") != MEASUREMENT_VERSION:
        raise ValueError("incompatible measurement version; rerun the workload")
    if (record.get("role") != role or record["validation"] != "1" or
            not (record["device"] == "cuda" or record["device"].startswith("cuda:"))):
        raise ValueError("workload must pass validation on CUDA with the expected role")
    if (record["device_uuid"] in ("", "unknown", "cpu") or int(record["pid"]) <= 0 or
            int(record["device_index"]) < 0):
        raise ValueError("workload must have a known CUDA device and process identity")
    if not record.get("torch_version") or not record.get("cuda_version"):
        raise ValueError("workload must identify framework and runtime versions")
    if int(record["start_ns"]) <= 0 or int(record["end_ns"]) <= int(record["start_ns"]):
        raise ValueError("invalid process interval")
    with (directory / f"{role}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(record["iterations"]):
        raise ValueError(f"{role} CSV is incomplete")
    previous_end = int(record["start_ns"])
    for index, row in enumerate(rows, 1):
        if (row.get("measurement_version") != MEASUREMENT_VERSION or
                row.get("role") != role or int(row["iteration"]) != index):
            raise ValueError(f"{role} CSV has an incompatible schema or step identity")
        begin, finish = int(row["iteration_start_ns"]), int(row["iteration_end_ns"])
        if begin < previous_end or finish <= begin or finish > int(record["end_ns"]):
            raise ValueError(f"{role} CSV has inconsistent iteration intervals")
        previous_end = finish
    return record, rows


def summarize_solo(directory: pathlib.Path, role: str, minimum_coverage: float = 0.95) -> dict:
    if not math.isfinite(minimum_coverage) or not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum coverage must be in [0, 1]")
    record, rows = read_workload(directory, role)
    start_ns = int(record["start_ns"])
    end_ns = int(record["measurement_deadline_ns"])
    if end_ns <= start_ns or int(record["end_ns"]) < end_ns:
        raise ValueError("solo workload must finish its duration window")
    coverage = (end_ns - start_ns) / (int(record["end_ns"]) - start_ns)
    if coverage < minimum_coverage:
        raise ValueError("solo window coverage is below the required minimum")
    return {**summarize_samples(rows, start_ns, end_ns, float(record["latency_target_ms"])),
            "coverage": coverage, "window_start_ns": start_ns, "window_end_ns": end_ns}


def summarize_colocation(directory: pathlib.Path, minimum_coverage: float = 0.0) -> dict:
    if not math.isfinite(minimum_coverage) or not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum coverage must be in [0, 1]")
    workloads = {role: read_workload(directory, role) for role in ("training", "inference")}
    metadata = {role: workload[0] for role, workload in workloads.items()}
    training, inference = metadata["training"], metadata["inference"]
    if training["pid"] == inference["pid"]:
        raise ValueError("co-location requires different process IDs")
    for record in metadata.values():
        if any(record[field] != training[field] for field in
               ("device_uuid", "device_index", "torch_version", "cuda_version")):
            raise ValueError("workloads must use the same CUDA device and framework versions")
    start_ns = max(int(record["start_ns"]) for record in metadata.values())
    end_ns = min(int(record["end_ns"]) for record in metadata.values())
    deadlines = [int(record["measurement_deadline_ns"]) for record in metadata.values()]
    if any(deadlines):
        if deadlines[0] == 0 or deadlines[0] != deadlines[1]:
            raise ValueError("workloads must use the same measurement deadline")
        if any(int(record["end_ns"]) < deadlines[0] for record in metadata.values()):
            raise ValueError("a workload stopped before the shared deadline")
        end_ns = min(end_ns, deadlines[0])
    if end_ns <= start_ns:
        raise ValueError("workloads have no common measurement window")
    result = {"measurement_version": MEASUREMENT_VERSION,
              "window_start_ns": start_ns, "window_end_ns": end_ns,
              "overlap_ms": (end_ns - start_ns) / 1_000_000,
              "device_uuid": training["device_uuid"]}
    for role, record in metadata.items():
        coverage = (end_ns - start_ns) / (int(record["end_ns"]) - int(record["start_ns"]))
        if coverage < minimum_coverage:
            raise ValueError(f"{role} common-window coverage {coverage:.3f} below {minimum_coverage}")
        rows = workloads[role][1]
        summary = summarize_samples(rows, start_ns, end_ns, float(record["latency_target_ms"]))
        result[role] = {**summary, "coverage": coverage}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--minimum-coverage", type=float, default=0.0)
    parser.add_argument("--start-duration-seconds", type=float)
    parser.add_argument("--check-scheduler", action="store_true")
    args = parser.parse_args()
    try:
        if args.check_scheduler:
            validate_scheduler_metrics((args.directory / "control-metrics.txt").read_text())
            return
        if args.start_duration_seconds is not None:
            if not math.isfinite(args.start_duration_seconds) or not 0 < args.start_duration_seconds <= 3600:
                raise ValueError("duration must be finite and in (0, 3600]")
            start_ns = time.monotonic_ns() + 500_000_000
            data = {"start_ns": start_ns,
                    "end_ns": start_ns + int(args.start_duration_seconds * 1_000_000_000)}
            destination = args.directory / "start.signal"
        else:
            data = summarize_colocation(args.directory, args.minimum_coverage)
            destination = args.directory / "colocation.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(destination)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"measurement error: {error}\n")


if __name__ == "__main__":
    main()
