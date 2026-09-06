# 0034: Adapt priority reservation from queue-wait SLO feedback

Status: Accepted

## Context

The static priority reservation from [ADR 0032](0032-priority-reserved-capacity.md)
protects a latency class, but one reservation value cannot fit every workload.
Too little reserved capacity lets inference queue behind throughput work; too
much leaves the GPU idle when the latency class is quiet. Glimmer already
observes task dispatch and completion boundaries, so it can adjust the
reservation from measured high-priority queue wait without changing a CUDA ABI.

## Decision

Add an opt-in `core::AdaptiveSloController` and connect it to the control
service's task-admission completion observations:

- only tasks at or above `priority_threshold` are sampled;
- a sample is a violation when queue wait is greater than
  `target_queue_wait_microseconds`;
- after `observation_window` samples, a violation ratio of at least 25 percent
  increases the reservation by one slot, while a lower violation ratio
  decreases it by one slot;
- reservations are clamped to an operator-provided minimum and maximum;
- the controller is disabled when the target is zero and the default static
  reservation remains unchanged;
- updates are applied through the scheduler's thread-safe
  `set_priority_reserved_slots` method, with recommendation and application
  serialized together across concurrent completion callbacks; and
- the controller emits an optional diagnostic record when a recommendation is
  applied.

The control service enables this policy with
`--adaptive-slo-target-queue-us N`, optionally setting
`--adaptive-slo-window N` and `--adaptive-slo-max-reserved-slots N`. Adaptive
operation requires the `priority` scheduler policy and at least one available
reserved slot. In simulated mode, the service explicitly reports dispatch and
terminal transitions performed by its deterministic executor so the same
feedback path is exercised. In remote mode, worker `COMPLETE`/`FAIL` reports
provide the completion boundary.

Lease state and scheduler transitions are serialized by the admission service.
Completion callbacks run after releasing that lock, before notifying blocked
claimers. This preserves lease ownership and ensures a capacity change made by
the callback wakes eligible work without holding admission's lock during
feedback or diagnostics.

## Non-goals and limitations

This is a bounded feedback controller, not a hard real-time guarantee. CUDA
kernels remain non-preemptive, and a high-priority task can still wait for a
lower-priority kernel that is already running. The controller does not inspect
kernel duration, GPU occupancy, memory pressure, or graph contents. It also
does not alter the versioned task protocol; remote clients continue to use the
existing lease operations.

Operators should choose a target from measured queue-wait distributions and
keep the observation window large enough to avoid reacting to one outlier. A
maximum below the scheduler's running capacity is recommended so unreserved
throughput work can continue.

## Consequences

The service can converge toward a reservation that matches observed inference
pressure while retaining deterministic scheduler behavior inside each window.
The one-slot step and window boundary limit oscillation and make changes easy
to diagnose. Low-violation high-priority completion windows gradually return
capacity to throughput work; absence of completions alone does not trigger
decay. Repeated SLO violations consume additional reserved
slots up to the configured cap. Feedback is best effort: callback failures do
not fail or roll back a task transition.

## Verification

- Core unit tests cover disabled feedback, priority filtering, window ratio
  thresholds, increase/decrease steps, bounds, and option normalization.
- Admission tests cover completion observations for service-owned and
  externally advanced scheduler transitions.
- A concurrent admission regression interleaves process-bound submit, claim,
  heartbeat, completion, and expiry scans, checking terminal observations and
  quota cleanup. A feedback regression checks independent lease progress while
  a callback is paused and wakes a blocked claimant after capacity changes.
- The control service argument validation rejects adaptive settings with a
  non-priority policy or an unusable reservation range.
- Existing scheduler, control, and CUDA tests remain unchanged when adaptive
  feedback is disabled.
