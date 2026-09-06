#!/usr/bin/env python3
"""Run a small PyTorch CUDA inference or training workload.

This program intentionally uses only public PyTorch APIs. It does not import
Glimmer code; the interceptor is supplied by the caller through LD_PRELOAD.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import math
import os
import pathlib
import sys
import time

from measurements import MEASUREMENT_VERSION, summarize_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("inference", "training"), default="inference")
    parser.add_argument("--device", default="cuda", help="Torch device, for example cuda or cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=0)
    parser.add_argument("--work-units", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--duration-seconds", type=float, default=0.0,
                        help="run until a wall-clock deadline instead of an iteration count")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--latency-target-ms",
        type=float,
        default=0.0,
        help="optional per-iteration latency target; zero disables SLO accounting",
    )
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--ready-file", type=pathlib.Path)
    parser.add_argument("--start-file", type=pathlib.Path)
    parser.add_argument("--status-file", type=pathlib.Path)
    parser.add_argument("--expected-memory-total-bytes", type=int,
                        help="verify the CUDA memory view before warmup and measurement")
    parser.add_argument("--profile-cuda", action="store_true",
                        help="diagnostic only: capture the warmed loop with a CUDA profiler")
    return parser.parse_args()


def validate_positive(arguments: argparse.Namespace) -> None:
    fields = ("batch_size", "hidden_size", "work_units", "iterations")
    for field in fields:
        if getattr(arguments, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if arguments.warmup < 0:
        raise ValueError("--warmup must not be negative")
    if arguments.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if not math.isfinite(arguments.latency_target_ms) or arguments.latency_target_ms < 0.0:
        raise ValueError("--latency-target-ms must be a finite non-negative value")
    if not math.isfinite(arguments.duration_seconds) or not 0 <= arguments.duration_seconds <= 3600:
        raise ValueError("--duration-seconds must be finite and in [0, 3600]")
    if arguments.expected_memory_total_bytes is not None and arguments.expected_memory_total_bytes <= 0:
        raise ValueError("--expected-memory-total-bytes must be positive")


def validate_memory_view(device_type: str, visible_total: int, expected_total: int) -> None:
    if device_type != "cuda" or visible_total != expected_total:
        raise ValueError(f"unexpected CUDA memory view: expected={expected_total} actual={visible_total}")


@contextmanager
def cuda_profile_range(torch, device_type: str, enabled: bool):
    if not enabled:
        yield
        return
    if device_type != "cuda":
        raise ValueError("--profile-cuda requires a CUDA device")
    torch.cuda.profiler.start()
    try:
        yield
    finally:
        torch.cuda.profiler.stop()


def import_torch():
    try:
        import torch
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(
            "pytorch_smoke error=import_failed "
            f"detail={error}. Install a CUDA-enabled PyTorch build in the environment.",
            file=sys.stderr,
        )
        return None
    return torch


def build_model(torch, hidden_size: int, intermediate_size: int, device, dtype):
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, intermediate_size),
        torch.nn.GELU(),
        torch.nn.Linear(intermediate_size, hidden_size),
    )
    return model.to(device=device, dtype=dtype)


def run_step(torch, model, inputs, targets, optimizer, role: str, work_units: int):
    if role == "inference":
        output = None
        with torch.inference_mode():
            for _ in range(work_units):
                output = model(inputs)
        return output

    optimizer.zero_grad(set_to_none=True)
    total_loss = None
    for _ in range(work_units):
        output = model(inputs)
        loss = torch.nn.functional.mse_loss(output, targets)
        total_loss = loss if total_loss is None else total_loss + loss
    total_loss = total_loss / work_units
    total_loss.backward()
    optimizer.step()
    return total_loss.detach()


def write_measurements(output: pathlib.Path, rows: list[dict[str, object]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(output: pathlib.Path, fields: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "".join(f"{key}={value}\n" for key, value in fields.items()), encoding="utf-8"
    )
    temporary.replace(output)


def wait_for_start(start_file: pathlib.Path, duration_seconds: float = 0.0,
                   timeout_seconds: float = 120.0) -> int:
    deadline = time.monotonic() + timeout_seconds
    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"start barrier was not created: {start_file}")
        time.sleep(0.001)
    if duration_seconds == 0:
        return 0
    window = json.loads(start_file.read_text())
    start_ns, end_ns = int(window["start_ns"]), int(window["end_ns"])
    if end_ns - start_ns != int(duration_seconds * 1_000_000_000):
        raise ValueError("shared measurement duration does not match the workload")
    while True:
        remaining_ns = start_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            break
        time.sleep(min(remaining_ns / 1_000_000_000, 0.001))
    if time.monotonic_ns() >= end_ns:
        raise TimeoutError("shared measurement window expired before workload start")
    return end_ns


def main() -> int:
    arguments = parse_args()
    try:
        validate_positive(arguments)
        if arguments.ready_file is not None and arguments.start_file is None:
            raise ValueError("--ready-file requires --start-file")
    except ValueError as error:
        print(f"pytorch_smoke error=invalid_argument detail={error}", file=sys.stderr)
        return 2

    torch = import_torch()
    if torch is None:
        return 2

    device = torch.device(arguments.device)
    visible_memory_total = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("pytorch_smoke error=cuda_unavailable", file=sys.stderr)
            return 3
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(device_index)
    elif device.type != "cpu":
        print(f"pytorch_smoke error=unsupported_device device={device}", file=sys.stderr)
        return 2
    if arguments.profile_cuda and (device.type != "cuda" or arguments.start_file is not None):
        print("pytorch_smoke error=profile_requires_solo_cuda", file=sys.stderr)
        return 2

    if arguments.expected_memory_total_bytes is not None:
        try:
            visible_memory_total = int(torch.cuda.mem_get_info(device)[1]) if device.type == "cuda" else 0
            validate_memory_view(device.type, visible_memory_total, arguments.expected_memory_total_bytes)
        except (ValueError, RuntimeError) as error:
            print(f"pytorch_smoke error=memory_view_failed detail={error}", file=sys.stderr)
            return 1

    dtype = getattr(torch, arguments.dtype)
    intermediate_size = arguments.intermediate_size or arguments.hidden_size * 4
    if intermediate_size <= 0:
        print("pytorch_smoke error=intermediate_size_must_be_positive", file=sys.stderr)
        return 2

    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    if device.type == "cuda":
        device_properties = torch.cuda.get_device_properties(device)
        device_index = torch.cuda.current_device()
        device_name = device_properties.name
        device_uuid = str(getattr(device_properties, "uuid", "unknown"))
    else:
        device_index = -1
        device_name = "cpu"
        device_uuid = "cpu"

    model = build_model(torch, arguments.hidden_size, intermediate_size, device, dtype)
    model.train(arguments.role == "training")
    inputs = torch.randn(arguments.batch_size, arguments.hidden_size, device=device, dtype=dtype)
    targets = torch.randn(arguments.batch_size, arguments.hidden_size, device=device, dtype=dtype)
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=arguments.learning_rate)
        if arguments.role == "training"
        else None
    )

    for _ in range(arguments.warmup):
        run_step(torch, model, inputs, targets, optimizer, arguments.role, arguments.work_units)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # These timestamps are compared as elapsed intervals by the co-location
    # runner. A monotonic clock prevents wall-clock corrections from making a
    # valid interval appear to run backwards.
    ready_ns = time.monotonic_ns()
    measurement_deadline_ns = 0
    if arguments.ready_file is not None:
        try:
            write_metadata(
                arguments.ready_file,
                {
                    "pid": os.getpid(),
                    "role": arguments.role,
                    "device": device,
                    "device_index": device_index,
                    "device_name": device_name,
                    "device_uuid": device_uuid,
                    "ready_ns": ready_ns,
                },
            )
            measurement_deadline_ns = wait_for_start(arguments.start_file, arguments.duration_seconds)
        except (OSError, TimeoutError, ValueError, KeyError) as error:
            print(f"pytorch_smoke error=barrier_failed detail={error}", file=sys.stderr)
            return 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    with cuda_profile_range(torch, device.type, arguments.profile_cuda):
        start_ns = time.monotonic_ns()
        if arguments.duration_seconds and not measurement_deadline_ns:
            measurement_deadline_ns = start_ns + int(arguments.duration_seconds * 1_000_000_000)

        start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        rows: list[dict[str, object]] = []
        last_result = None
        iteration = 0
        while (time.monotonic_ns() < measurement_deadline_ns if measurement_deadline_ns
               else iteration < arguments.iterations):
            iteration += 1
            iteration_start_ns = time.monotonic_ns()
            if start_event is not None:
                start_event.record()
            last_result = run_step(
                torch, model, inputs, targets, optimizer, arguments.role, arguments.work_units
            )
            if end_event is not None:
                end_event.record()
                end_event.synchronize()
            iteration_end_ns = time.monotonic_ns()
            elapsed_ms = (iteration_end_ns - iteration_start_ns) / 1_000_000
            cuda_event_ms = float(start_event.elapsed_time(end_event)) if end_event is not None else ""
            allocated_bytes = (
                int(torch.cuda.memory_allocated(device)) if device.type == "cuda" else 0
            )
            peak_bytes = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )
            rows.append(
                {
                    "role": arguments.role,
                    "device": str(device),
                    "iteration": iteration,
                    "measurement_version": MEASUREMENT_VERSION,
                    "iteration_start_ns": iteration_start_ns,
                    "iteration_end_ns": iteration_end_ns,
                    "elapsed_ms": f"{elapsed_ms:.9f}",
                    "cuda_event_ms": cuda_event_ms,
                    "latency_target_ms": f"{arguments.latency_target_ms:.9f}",
                    "latency_target_met": (int(elapsed_ms <= arguments.latency_target_ms)
                                           if arguments.latency_target_ms else ""),
                    "memory_allocated_bytes": allocated_bytes,
                    "peak_memory_allocated_bytes": peak_bytes,
                }
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end_ns = time.monotonic_ns()
    if not rows:
        print("pytorch_smoke error=no_measured_steps", file=sys.stderr)
        return 1
    if arguments.role == "inference":
        validation = bool(torch.isfinite(last_result).all().item())
    else:
        validation = bool(torch.isfinite(last_result).item())
    if not validation:
        print("pytorch_smoke error=non_finite_result", file=sys.stderr)
        return 1

    summary = summarize_samples(rows, start_ns, end_ns, arguments.latency_target_ms)
    mean_ms, p50_ms, p95_ms, p99_ms = (summary[key] for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"))
    throughput = summary["steps_per_second"]
    target_miss_count = summary["latency_target_miss_count"]
    target_miss_ratio = summary["latency_target_miss_ratio"]

    if arguments.output is not None:
        write_measurements(arguments.output, rows)
    if arguments.status_file is not None:
        try:
            write_metadata(
                arguments.status_file,
                {
                    "pid": os.getpid(),
                    "role": arguments.role,
                    "device": device,
                    "device_index": device_index,
                    "device_name": device_name,
                    "device_uuid": device_uuid,
                    "ready_ns": ready_ns,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "iterations": len(rows),
                    "measurement_version": MEASUREMENT_VERSION,
                    "measurement_deadline_ns": measurement_deadline_ns,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "validation": int(validation),
                    "visible_memory_total_bytes": visible_memory_total,
                    "cuda_profile": int(arguments.profile_cuda),
                    "mean_ms": f"{mean_ms:.3f}",
                    "p50_ms": f"{p50_ms:.3f}",
                    "p95_ms": f"{p95_ms:.3f}",
                    "p99_ms": f"{p99_ms:.3f}",
                    "steps_per_second": f"{throughput:.3f}",
                    "latency_target_ms": f"{arguments.latency_target_ms:.9f}",
                    "latency_target_miss_count": target_miss_count,
                    "latency_target_miss_ratio": f"{target_miss_ratio:.6f}",
                },
            )
        except OSError as error:
            print(f"pytorch_smoke error=status_write_failed detail={error}", file=sys.stderr)
            return 1

    peak_bytes = rows[-1]["peak_memory_allocated_bytes"]
    print(
        "pytorch_smoke "
        f"role={arguments.role} status=ok validation=1 device={device} "
        f"device_name={device_name!r} torch={torch.__version__} "
        f"cuda={torch.version.cuda!r} iterations={len(rows)} measurement_version={MEASUREMENT_VERSION} "
        f"cuda_profile={int(arguments.profile_cuda)} "
        f"work_units={arguments.work_units} mean_ms={mean_ms:.3f} "
        f"p50_ms={p50_ms:.3f} p95_ms={p95_ms:.3f} p99_ms={p99_ms:.3f} "
        f"steps_per_second={throughput:.3f} peak_memory_allocated_bytes={peak_bytes} "
        f"latency_target_ms={arguments.latency_target_ms:.9f} "
        f"latency_target_miss_count={target_miss_count} "
        f"latency_target_miss_ratio={target_miss_ratio:.6f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
