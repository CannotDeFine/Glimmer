# 0015: Use explicit remote task leases for worker execution

Status: Accepted

## Context

The control service can admit logical tasks and a local simulated executor can
advance them, but real CUDA work must run in a process that owns its CUDA
context, streams, events, pointers, and kernel arguments. Passing those handles
through an IPC protocol would be invalid because CUDA objects are process-local.
The scheduler also cannot safely preempt an arbitrary running kernel.

## Decision

Add an explicit worker lease flow to the versioned control protocol:

```text
SUBMIT tenant memory weight work_units -> OK task_id
CLAIM                              -> LEASE task_id tenant memory weight work_units
HEARTBEAT task_id                 -> STATE task_id RUNNING
COMPLETE task_id                   -> STATE task_id COMPLETED
FAIL task_id                       -> STATE task_id FAILED
```

`CLAIM` is the only operation that transitions a queued task to `RUNNING` and
commits its quota reservation. If no task is queued, it returns `EMPTY`. The
worker uses the returned metadata to construct and execute local CUDA work; no
CUDA handle, pointer, stream, event, or argument address is serialized. The
worker must report exactly one terminal result. `CANCEL` remains valid only for
queued tasks, and an invalid terminal transition returns `INVALID_REQUEST` (or
`UNKNOWN_TASK` when the task does not exist).

Lease timeout is opt-in. When configured, `HEARTBEAT task_id` renews a running
lease and the service marks an unrenewed lease as `FAILED` after the timeout,
releasing its reservation. The default timeout is disabled for compatibility
with workers that do not yet send heartbeats.

The Unix socket remains the local transport and authenticates the peer UID.
Remote lease mode disables the simulated executor loop, while simulated mode
remains the default compatibility path for deterministic control-plane tests.
The service may expose multiple running leases when its explicit concurrency
capacity is greater than one; see [ADR 0016](0016-configurable-concurrent-task-slots.md).

## Consequences

The service and worker have a clear ownership boundary: the service owns task
state, fairness, and quota; the worker owns CUDA resources and execution. This
supports transparent CUDA execution in a dedicated worker without pretending
that process-local CUDA objects are transferable. With timeout enabled, a
worker crash is eventually recovered as a failed task; timeout disabled still
leaves a running lease until service restart.
