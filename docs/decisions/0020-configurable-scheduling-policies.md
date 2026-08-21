# 0020: Keep task scheduling policy configurable

Status: Accepted

## Context

Different GPU workloads need different task-boundary trade-offs. Weighted
tenant fairness is useful for shared service capacity, while a simple FIFO
order is easier to reason about for latency-sensitive or single-tenant work.
Embedding one policy directly in `core::Scheduler` would make later policies
change the accounting and lifecycle code as well as the selection order.

CUDA kernels cannot be safely preempted through the ordinary Driver or Runtime
submission APIs. A policy therefore controls which queued task is dispatched
next; it does not interrupt a kernel that is already running.

## Decision

Keep policy selection inside the scheduler core and separate it from quota
admission and task state transitions. The supported policies are:

- `weighted_rr` (the default): weighted round-robin ordering between tenants;
- `drr`: deficit round-robin using `work_units` as task cost and `weight` as
  the tenant quantum;
- `fifo`: oldest accepted queued task first; and
- `priority`: highest-priority queued task first, with submission order as the
  tie-breaker.

The control service selects the policy at startup with
`--scheduler-policy fifo|weighted_rr|drr|priority`. A scheduler's policy is immutable for
its lifetime; changing it requires restarting the service. Unknown policy
values are rejected at the configuration boundary, while an invalid enum
received directly by the C++ API falls back to the safe default.

## Consequences

Policy experiments can be added and tested without changing CUDA interception,
quota accounting, or terminal transition logic. Existing deployments retain
weighted fairness by default. Priority values are ignored by the other
policies, and strict priority may starve lower-priority work under sustained
load. Kernel preemption remains outside this interface.
