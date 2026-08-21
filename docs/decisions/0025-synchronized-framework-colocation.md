# ADR 0025: Prove framework co-location with explicit barriers and GPU identity

- Status: Accepted
- Date: 2026-08-21
- Decision owners: Glimmer maintainers

## Context

A priority experiment is not meaningful if inference and training are merely
started near one another or accidentally use different devices. The framework
workload must prove that both independent processes were ready, started from a
common barrier, used one physical GPU, and executed concurrently.

## Decision

The PyTorch co-location runner uses a file-based local synchronization protocol:

1. Each process initializes CUDA, creates its model, completes warmup, and
   atomically writes a readiness record.
2. The parent waits for both readiness records and creates one shared start
   marker.
3. Each process records its PID, CUDA device index, device name, device UUID,
   and monotonic start/end timestamps in a status record. Monotonic time is
   required because the parent compares interval durations and wall-clock
   corrections must not make a valid run appear to go backwards.
4. The parent requires distinct PIDs, equal non-unknown GPU UUIDs, successful
   validation, and a positive interval intersection before printing
   `colocated=1`.

Priority mode additionally routes both processes through the same authenticated
remote control service with one concurrent slot and distinct priorities.

## Consequences

The experiment has auditable evidence for physical co-location and real
contention. The file barrier is intentionally limited to the example runner;
it is not a scheduler transport or a production synchronization primitive.
Monotonic interval overlap demonstrates concurrent process execution, while scheduler
logs and per-process CSV files provide the performance measurements.
