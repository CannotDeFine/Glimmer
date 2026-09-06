#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite export; never report GPU utilization."""

from collections import Counter, defaultdict
from contextlib import closing
import math
import pathlib
import sqlite3


def distribution(values):
    values = list(values)
    if not values:
        return {"count": 0, "total_ns": 0, "p50_ns": 0, "p99_ns": 0, "max_ns": 0}
    if any(not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("invalid duration")
    values.sort()

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        return values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower)

    return {"count": len(values), "total_ns": sum(values), "p50_ns": percentile(0.5),
            "p99_ns": percentile(0.99), "max_ns": values[-1]}


def interval_summary(intervals):
    """Union intervals across streams; count only internal gaps, not edge time."""
    if not intervals:
        raise ValueError("no GPU execution intervals")
    for start, end in intervals:
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
            raise ValueError("invalid GPU interval")
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    gaps = [right[0] - left[1] for left, right in zip(merged, merged[1:])]
    span = merged[-1][1] - merged[0][0]
    covered = sum(end - start for start, end in merged)
    return {"first_start_ns": merged[0][0], "last_end_ns": merged[-1][1],
            "span_ns": span, "covered_ns": covered, "coverage_fraction": covered / span,
            "internal_gaps": distribution(gaps)}


def analyze_database(database):
    tables = {row[0] for row in database.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"
    if kernel_table not in tables or "StringIds" not in tables:
        raise ValueError("Nsight export is missing CUDA kernels or string data")
    kernel_rows = database.execute(
        f"SELECT start, end, deviceId, globalPid FROM {kernel_table}").fetchall()
    if not kernel_rows:
        raise ValueError("no actual CUDA kernels captured")
    identities = {(row[2], row[3]) for row in kernel_rows}
    if len(identities) != 1 or any(value is None for value in next(iter(identities))):
        raise ValueError("diagnostic requires one CUDA process on one device")
    intervals = [(row[0], row[1]) for row in kernel_rows]
    result = {"version": "cuda_timeline_v1", "diagnostic_only": True,
              "device_id": kernel_rows[0][2], "global_pid": kernel_rows[0][3],
              "kernels": interval_summary(intervals),
              "kernel_duration": distribution([end - start for start, end in intervals])}
    activity_counts = {"KERNEL": len(intervals)}
    for kind in ("MEMCPY", "MEMSET"):
        table = f"CUPTI_ACTIVITY_KIND_{kind}"
        rows = database.execute(f"SELECT start, end, deviceId, globalPid FROM {table}").fetchall() if table in tables else []
        if any((row[2], row[3]) not in identities for row in rows):
            raise ValueError("GPU activity belongs to another process or device")
        intervals += [(row[0], row[1]) for row in rows]
        activity_counts[kind] = len(rows)
    result["execution"] = interval_summary(intervals)
    result["activity_counts"] = activity_counts
    apis = []
    for kind in ("RUNTIME", "DRIVER"):
        table = f"CUPTI_ACTIVITY_KIND_{kind}"
        if table not in tables:
            continue
        grouped = defaultdict(list)
        for start, end, name, thread, code in database.execute(
                f"SELECT a.start, a.end, s.value, a.globalTid, a.returnValue FROM {table} a "
                "LEFT JOIN StringIds s ON a.nameId=s.id"):
            if name is None or not isinstance(start, int) or not isinstance(end, int) or end < start:
                raise ValueError("invalid CUDA API record")
            grouped[name].append((end - start, thread, code))
        for name, calls in grouped.items():
            apis.append({"kind": kind, "name": name, **distribution([call[0] for call in calls]),
                         "threads": len({call[1] for call in calls}),
                         "return_values": dict(Counter(str(call[2]) for call in calls))})
    if not apis:
        raise ValueError("no CUDA API trace captured")
    result["apis"] = sorted(apis, key=lambda row: row["total_ns"], reverse=True)
    result["extended_launch_apis"] = [row for row in result["apis"] if "LaunchKernelEx" in row["name"]]
    result["profiler_diagnostics"] = []
    if "DIAGNOSTIC_EVENT" in tables:
        for severity, message in database.execute(
                "SELECT s.name, d.text FROM DIAGNOSTIC_EVENT d "
                "LEFT JOIN ENUM_DIAGNOSTIC_SEVERITY_LEVEL s ON s.id=d.severity"):
            if severity in (None, "Error", "UnknownLevel"):
                raise ValueError(f"profiler diagnostic: {message}")
            result["profiler_diagnostics"].append({"severity": severity, "message": message})
    result["trace_quality"] = ("warnings_require_review" if any(
        item["severity"] == "Warning" for item in result["profiler_diagnostics"]) else "no_reported_warnings")
    result["limitations"] = [
        "Instrumented diagnostic, not an untraced performance measurement.",
        "Coverage is interval union/span, not device-wide idle time or SM utilization.",
        "Internal gaps exclude leading/trailing capture time and activities of other processes.",
        "API sums overlap across threads and nested Runtime/Driver calls; do not add them as latency.",
        "API duration is host time, not GPU kernel execution time.",
        "Return values are raw profiler fields, not an independent CUDA status validation.",
        "Profiler injection can change symbol routing; audit unprofiled interception separately.",
        "Warnings about missing events prevent completeness claims and reliable gap comparisons.",
    ]
    return result


def analyze_sqlite(path: pathlib.Path):
    # Read-only mode also prevents a typo from creating an empty database.
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as database:
        return analyze_database(database)
