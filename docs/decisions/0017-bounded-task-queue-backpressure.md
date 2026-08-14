# 0017: Bound queued work with explicit backpressure

Status: Accepted

## Context

The scheduler's memory quota limits bytes, but it does not limit the number of
small logical tasks waiting in the queue. A client can therefore submit many
tiny tasks and grow scheduler metadata and tenant queues without approaching
the GPU memory limit. A general workload service needs a deterministic way to
push back before this becomes an unbounded resource.

## Decision

Add an optional `max_queued_tasks` scheduler capacity:

- `0` means unlimited for compatibility;
- a positive value bounds tasks in `QUEUED` state only;
- running tasks do not consume waiting-queue capacity;
- a rejected submission returns `SubmitStatus::kQueueFull` without reserving
  quota or creating a task ID; and
- the control protocol exposes the rejection as `ERROR QUEUE_FULL`.

The control service exposes the setting as `--max-queued-tasks N`. This is
backpressure, not admission priority: weighted tenant scheduling remains
responsible for ordering tasks that have already entered the queue. Terminal
task retention and persistence are separate lifecycle concerns and remain
outside this capacity setting.

## Consequences

Operators can bound waiting-work metadata independently from GPU memory quota,
and clients can distinguish queue pressure from a memory rejection and retry or
apply their own backoff policy. Existing callers retain the unlimited default.
