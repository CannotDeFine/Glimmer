# 0018: Expose a read-only control-plane stats snapshot

Status: Accepted

## Context

The scheduler and remote lease service now enforce quota, running-slot, and
waiting-queue limits, but an operator can only inspect one task at a time. A
service that is otherwise working correctly can therefore be difficult to
diagnose without attaching a debugger or parsing application-specific logs.

## Decision

Add a read-only `STATS` operation to the versioned control protocol. The
response reports:

- total, queued, running, completed, cancelled, and failed task counts;
- quota limit, reserved bytes, and allocated bytes; and
- configured running-task and queued-task limits.

The snapshot is assembled under the scheduler's synchronization boundary and
does not mutate task state, renew leases, or perform CUDA work. The Unix socket
client exposes it as `stats`. The response is intentionally a bounded
diagnostic record, not a durable time-series metrics format; a future metrics
exporter may consume equivalent observations without changing scheduling or
wire semantics.

## Consequences

Operators can observe queue pressure, active work, terminal outcomes, and quota
recovery with one request. The protocol remains process-local and carries no
CUDA handles, pointers, tenant payloads, or credentials. Terminal task records
remain subject to the existing retention policy; `STATS` does not persist or
prune them.
