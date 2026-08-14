# 0016: Make scheduler concurrency an explicit capacity

Status: Accepted

## Context

The first task scheduler allowed only one running task. That was useful for
deterministic bootstrap tests, but a remote control service with multiple CUDA
workers needs to overlap independent task boundaries when the configured GPU
quota and workload permit it. A worker lease already keeps CUDA resources
process-local, so increasing scheduler capacity does not require serializing
CUDA handles or adding preemption.

## Decision

Add a `max_running_tasks` capacity to `core::Scheduler`:

- the default remains `1` for compatibility;
- a positive configured value allows that many running task IDs at once;
- `dispatch_next()` remains serialized by the scheduler mutex and never
  returns the same queued task to two claimers;
- each running task retains its own committed quota charge until completion,
  failure, or cancellation; and
- `TaskAdmissionService` tracks one lease deadline per running task so
  heartbeats and expiry recovery remain independent.

The control service exposes this as `--max-concurrent-tasks N`. The setting
controls remote worker leases; it does not claim that CUDA kernels are
preemptible or that a single GPU can safely run every workload concurrently.
Quota reservations remain the memory safety boundary, and task fairness still
applies only at explicit dispatch boundaries.

## Consequences

Multiple worker processes can claim tasks concurrently while the service keeps
task identity, quota accounting, and lease recovery centralized. The default
single-slot behavior remains deterministic for existing simulated and unit
tests. Operators must choose a capacity compatible with the GPU, streams, and
workload; increasing the slot count does not increase the physical GPU memory
limit.
