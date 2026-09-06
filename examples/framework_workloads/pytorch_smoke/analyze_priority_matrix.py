#!/usr/bin/env python3
"""Aggregate repeatable PyTorch co-location measurements.

The input is the ``summary.csv`` emitted by ``run_priority_matrix.sh``. The
aggregates are descriptive (means of per-run values); they are not a claim of
an application-wide performance guarantee.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
import sys
from collections import defaultdict


NUMERIC_FIELDS = (
    "training_mean_ms",
    "training_p50_ms",
    "training_p95_ms",
    "training_p99_ms",
    "training_steps_per_second",
    "inference_mean_ms",
    "inference_p50_ms",
    "inference_p95_ms",
    "inference_p99_ms",
    "inference_steps_per_second",
    "inference_target_miss_ratio",
    "scheduler_total_queue_wait_us",
    "scheduler_max_queue_wait_us",
    "scheduler_total_service_time_us",
    "scheduler_max_service_time_us",
    "overlap_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=pathlib.Path, help="matrix summary.csv")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="aggregate CSV output (default: <summary>.aggregate.csv)",
    )
    return parser.parse_args()


def parse_nonnegative(row: dict[str, str], field: str, line_number: int) -> float:
    value = row.get(field, "")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"line {line_number}: {field} is not numeric") from error
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"line {line_number}: {field} must be finite and non-negative")
    return parsed


def configuration_name(row: dict[str, str]) -> str:
    measurement = row.get("measurement_version", "cuda_event_v1")
    if row["mode"] == "native":
        return f"{measurement}-native"
    return (
        f"{measurement}-priority-c{row['max_concurrent_kernels']}"
        f"-reserve-s{row['priority_reserved_slots']}"
        f"-threshold-{row['priority_threshold']}"
        f"-train-b{row['training_launch_batch_size']}"
        f"-infer-b{row['inference_launch_batch_size']}"
    )


def aggregate(summary: pathlib.Path) -> list[dict[str, object]]:
    with summary.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required_fields = {
            "mode",
            "tensor_batch_size",
            "hidden_size",
            "max_concurrent_kernels",
            "priority_reserved_slots",
            "priority_threshold",
            "training_launch_batch_size",
            "inference_launch_batch_size",
            "inference_target_ms",
        }
        required_fields.update(NUMERIC_FIELDS)
        missing_fields = sorted(required_fields - set(reader.fieldnames or []))
        if missing_fields:
            raise ValueError(f"summary is missing fields: {', '.join(missing_fields)}")

        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        group_rows: dict[str, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            if row.get("mode") not in ("native", "priority"):
                raise ValueError(f"line {line_number}: mode must be native or priority")
            for field in (
                "tensor_batch_size",
                "hidden_size",
                "max_concurrent_kernels",
                "priority_reserved_slots",
                "priority_threshold",
                "training_launch_batch_size",
                "inference_launch_batch_size",
            ):
                parse_nonnegative(row, field, line_number)
            name = configuration_name(row)
            parsed = {
                field: parse_nonnegative(row, field, line_number) for field in NUMERIC_FIELDS
            }
            parse_nonnegative(row, "inference_target_ms", line_number)
            groups[name].append(parsed)
            group_rows.setdefault(name, row)

    results: list[dict[str, object]] = []
    for name, rows in groups.items():
        first = group_rows[name]
        result: dict[str, object] = {
            "configuration": name,
            "measurement_version": first.get("measurement_version", "cuda_event_v1"),
            "runs": len(rows),
            "tensor_batch_size": first.get("tensor_batch_size", ""),
            "hidden_size": first.get("hidden_size", ""),
            "priority_reserved_slots": first.get("priority_reserved_slots", "0"),
            "priority_threshold": first.get("priority_threshold", "0"),
            "inference_target_ms": first.get("inference_target_ms", "0.000"),
        }
        for field in NUMERIC_FIELDS:
            result[f"{field}_mean"] = f"{statistics.fmean(row[field] for row in rows):.6f}"
        result["scheduler_max_queue_wait_us_max"] = (
            f"{max(row['scheduler_max_queue_wait_us'] for row in rows):.6f}"
        )
        result["scheduler_max_service_time_us_max"] = (
            f"{max(row['scheduler_max_service_time_us'] for row in rows):.6f}"
        )
        results.append(result)
    return results


def write_results(output: pathlib.Path, rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["configuration", "runs"]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    arguments = parse_args()
    output = arguments.output or arguments.summary.with_suffix(".aggregate.csv")
    try:
        rows = aggregate(arguments.summary)
        if not rows:
            raise ValueError("summary contains no runs")
        write_results(output, rows)
    except (OSError, ValueError) as error:
        print(f"priority_matrix_analysis error={error}", file=sys.stderr)
        return 1
    print(f"priority_matrix_analysis status=ok rows={len(rows)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
