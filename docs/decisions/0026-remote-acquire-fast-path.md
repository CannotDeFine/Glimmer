# ADR 0026: Combine uncontended remote launch admission

- Status: Accepted
- Date: 2026-08-21
- Decision owners: Glimmer maintainers

## Context

The transparent remote launch gate previously submitted a one-unit task and
then issued a separate task-specific `CLAIM` request. Every protocol request
created a new Unix socket connection. This added a fixed control-plane cost to
every CUDA kernel launch, even when no other process was waiting.

## Decision

Add the versioned `ACQUIRE` operation. It carries the same admission metadata
as `SUBMIT` and lets the endpoint submit the task and attempt its immediate
task-specific claim in one request. The endpoint returns:

- `LEASE` when the task is immediately running; or
- `ACCEPTED` with the task id when the task remains queued.

The caller polls `CLAIM <task_id>` only for the queued case, then uses the
existing `COMPLETE`, `FAIL`, `HEARTBEAT`, and `CANCEL` operations. Existing
`SUBMIT` and `CLAIM` operations remain available for explicit clients and
protocol compatibility.

## Consequences

The uncontended transparent launch path removes one request/response. The
Unix transport now reuses the authenticated connection for subsequent requests
from the same calling thread, so uncontended admission and completion avoid
repeated socket setup as well. The operation does not batch leases or preempt
running CUDA kernels; those are separate optimizations. A queued task still
incurs claim polling, so contention latency remains an explicit metric.
