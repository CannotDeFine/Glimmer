# 0032: Reserve capacity for latency-sensitive priority work

Status: Accepted

## Context

Strict priority ordering only affects queued work. If all GPU launch slots are
occupied by throughput-oriented work before an inference request arrives, the
inference request must still wait for a completion boundary. Transparent CUDA
interception cannot safely preempt an arbitrary running kernel, so ordering
alone cannot provide a useful latency-isolation boundary.

## Decision

Add an optional priority-capacity reservation to the scheduler:

- `SchedulerOptions::priority_reserved_slots` reserves running slots for tasks
  at or above `priority_reservation_threshold` when the `priority` policy is
  selected;
- lower-priority tasks may use only the unreserved slots, even when the
  reserved slots are currently idle;
- the lower-priority allowance counts running lower-priority tasks, not all
  running tasks: inference occupying a reserved slot does not block training
  from filling a free unreserved slot;
- high-priority tasks may use any available slot and retain the existing strict
  priority ordering and submission-order tie breaking;
- a running lower-priority task is never preempted when a high-priority task is
  submitted;
- the reservation is disabled by default and is clamped to the configured
  running capacity;
- the control service exposes `--priority-reserved-slots` and
  `--priority-threshold`; and
- the transparent local gate accepts the corresponding trusted environment
  settings `GLIMMER_SCHEDULER_PRIORITY_RESERVED_SLOTS` and
  `GLIMMER_SCHEDULER_PRIORITY_THRESHOLD`.

The reservation is an explicit capacity trade-off: unused reserved slots are
not borrowed by lower-priority work. Operators should configure at least two
running slots and a workload that can safely overlap before enabling it.

## Consequences

Inference can receive a scheduling slot immediately while training occupies the
unreserved capacity. This makes priority useful for co-located workloads
without changing CUDA ABIs or pretending to provide kernel preemption. The
trade-off is possible under-utilization when no high-priority work is present,
and a high-priority request can still wait for already-running lower-priority
kernels. The setting is therefore a best-effort latency-isolation mechanism,
not a hard real-time guarantee.

## Verification

- Core scheduler tests verify that lower-priority work cannot borrow a reserved
  slot, high-priority work uses the slot, and the lower-priority queue drains
  afterward.
- Existing launch-gate and control-service tests retain their default behavior
  with zero reserved slots.
- CUDA interceptor builds verify configuration parsing and preserve the
  existing no-GPU and GPU launch tests.
- Real framework co-location runs compare p95/p99 inference latency, target
  misses, queue wait, and training throughput with and without the reservation.
