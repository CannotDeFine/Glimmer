# 0030: Add opt-in scheduler dispatch-order diagnostics

Status: Accepted

## Context

The priority workload runner already proves that independent processes use the
same physical GPU and overlap in time. Per-process launch timings show waiting
and transport overhead, but they cannot establish the order in which the
control service selected queued tasks. A diagnostic is needed to audit strict
priority behavior without making logging part of the normal scheduling path.

## Decision

Add the disabled-by-default `--trace-scheduler` option to
`glimmer_control_service`. When enabled, `TaskAdmissionService` emits a
best-effort observation after each successful admission-service dispatch. The
service writes one stderr record containing:

- a process-local monotonic dispatch sequence;
- task ID, tenant ID, priority, memory bytes, and work units; and
- a monotonic dispatch timestamp.

The sequence is assigned by `core::Scheduler` at the running-state transition,
so concurrent admission callers can be ordered without relying on callback
timing.

The callback is outside lease locks, never changes the scheduler result, and
swallows observer allocation or callback failures. The trace covers dispatches
made through `TaskAdmissionService`; components that mutate a `Scheduler`
directly remain responsible for their own diagnostics. The PyTorch priority
runner exposes `--trace-scheduler` and stores the service output in
`control-service.log`.

## Consequences

Operators can compare the authoritative service dispatch sequence with process
CSV timings when validating priority or fairness experiments. The trace adds
copying, formatting, and I/O only when explicitly enabled, so it is not a
performance metric and must remain disabled for benchmark measurements. It does
not provide kernel preemption, a latency guarantee, or durable observability.

## Verification

- Admission unit tests verify that the observer sees each successful dispatch
  once, in strict-priority order, with monotonic sequence values.
- The priority runner accepts the option and retains the service log in its
  output directory.
- Debug, sanitizer, lint, formatting, and CUDA suites remain the applicable
  regression gates; no fixed timestamp or duration is asserted.
