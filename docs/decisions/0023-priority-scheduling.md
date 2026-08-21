# 0023: Add explicit priority scheduling

Status: Accepted

## Context

Glimmer already supports FIFO, weighted round-robin, and deficit round-robin
dispatch. Those policies express ordering and tenant fairness, but they do not
let a workload identify latency-sensitive work that should run before ordinary
queued work. Adding an implicit priority layer to every existing policy would
silently change their fairness guarantees and make deployments difficult to
reason about.

## Decision

Add a separate `priority` scheduling policy:

- `TaskSpec` and task-admission requests carry an unsigned 32-bit priority;
- a larger priority value is more important, and an omitted value defaults to
  zero;
- the priority policy selects the highest-priority queued task globally;
- tasks with equal priority are selected by ascending task ID, which is their
  submission order;
- FIFO, weighted round-robin, and deficit round-robin ignore the priority field
  and retain their existing ordering and fairness semantics;
- the versioned submit request accepts an optional priority field after
  `work_units`, while the old six-token form remains valid;
- the transparent launch gate accepts the priority from trusted configuration
  and forwards it in remote launch submissions; and
- priority applies only at task-boundary admission. It does not preempt a
  running CUDA kernel or provide an aging guarantee for lower-priority work.

The standalone control client exposes the optional value after `work_units`.
The default remains compatible with existing clients and workloads.

## Consequences

Priority-sensitive work can bypass lower-priority queued work when the
`priority` policy is explicitly selected. Strict priority can starve lower
priority tasks under a sustained high-priority workload; operators that need
tenant fairness should use weighted or deficit round-robin instead. The
scheduler, control protocol, transparent launch path, tests, and user-facing
configuration all use the same field and policy name.

## Verification

- Core tests verify descending priority order, FIFO tie-breaking, and the
  policy parser/name, including that task-specific dispatch cannot bypass the
  priority head.
- Protocol tests verify optional priority round trips, legacy submit parsing,
  explicit zero compatibility, and overflow rejection.
- Control endpoint tests verify that a real `SUBMIT`/`CLAIM` flow dispatches
  higher-priority work before lower-priority work.
- Admission and protocol plumbing preserve the priority field, while launch
  gate configuration keeps the default zero-priority behavior unchanged.
- Debug, sanitizer, lint, CUDA lint, formatting, and GPU suites remain
  required according to the repository testing matrix.
