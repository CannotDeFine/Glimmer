#!/usr/bin/env python3
"""Run a small PyTorch CUDA inference or training workload.

This program intentionally uses only public PyTorch APIs. It does not import
Glimmer code; the interceptor is supplied by the caller through LD_PRELOAD.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import statistics
import sys
import time


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
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--ready-file", type=pathlib.Path)
    parser.add_argument("--start-file", type=pathlib.Path)
    parser.add_argument("--status-file", type=pathlib.Path)
    return parser.parse_args()


def percentile(values: list[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def validate_positive(arguments: argparse.Namespace) -> None:
    fields = ("batch_size", "hidden_size", "work_units", "iterations")
    for field in fields:
        if getattr(arguments, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive")
    if arguments.warmup < 0:
        raise ValueError("--warmup must not be negative")
    if arguments.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")


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


def wait_for_start(start_file: pathlib.Path, timeout_seconds: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"start barrier was not created: {start_file}")
        time.sleep(0.001)


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
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("pytorch_smoke error=cuda_unavailable", file=sys.stderr)
            return 3
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_device(device_index)
    elif device.type != "cpu":
        print(f"pytorch_smoke error=unsupported_device device={device}", file=sys.stderr)
        return 2

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
            wait_for_start(arguments.start_file)
        except (OSError, TimeoutError) as error:
            print(f"pytorch_smoke error=barrier_failed detail={error}", file=sys.stderr)
            return 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_ns = time.monotonic_ns()

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    rows: list[dict[str, object]] = []
    elapsed_values: list[float] = []
    last_result = None
    for iteration in range(1, arguments.iterations + 1):
        if start_event is not None:
            start_event.record()
        else:
            cpu_start = time.perf_counter()
        last_result = run_step(
            torch, model, inputs, targets, optimizer, arguments.role, arguments.work_units
        )
        if end_event is not None:
            end_event.record()
            end_event.synchronize()
            elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            elapsed_ms = (time.perf_counter() - cpu_start) * 1000.0
        elapsed_values.append(elapsed_ms)
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
                "elapsed_ms": f"{elapsed_ms:.3f}",
                "memory_allocated_bytes": allocated_bytes,
                "peak_memory_allocated_bytes": peak_bytes,
            }
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end_ns = time.monotonic_ns()
    if arguments.role == "inference":
        validation = bool(torch.isfinite(last_result).all().item())
    else:
        validation = bool(torch.isfinite(last_result).item())
    if not validation:
        print("pytorch_smoke error=non_finite_result", file=sys.stderr)
        return 1

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
                    "iterations": arguments.iterations,
                    "validation": int(validation),
                },
            )
        except OSError as error:
            print(f"pytorch_smoke error=status_write_failed detail={error}", file=sys.stderr)
            return 1

    mean_ms = statistics.fmean(elapsed_values)
    throughput = 1000.0 / mean_ms if mean_ms > 0.0 else 0.0
    peak_bytes = rows[-1]["peak_memory_allocated_bytes"]
    print(
        "pytorch_smoke "
        f"role={arguments.role} status=ok validation=1 device={device} "
        f"device_name={device_name!r} torch={torch.__version__} "
        f"cuda={torch.version.cuda!r} iterations={arguments.iterations} "
        f"work_units={arguments.work_units} mean_ms={mean_ms:.3f} "
        f"p50_ms={percentile(elapsed_values, 50.0):.3f} "
        f"p95_ms={percentile(elapsed_values, 95.0):.3f} "
        f"p99_ms={percentile(elapsed_values, 99.0):.3f} "
        f"steps_per_second={throughput:.3f} peak_memory_allocated_bytes={peak_bytes}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
