# 0012: Use transactional task admission for explicit execution clients

Status: Accepted

## Context

The explicit CUDA backend can now execute a trusted task descriptor and the
core scheduler can apply tenant admission and weighted dispatch. An external
launcher or control service still needs a stable way to submit a logical task
without depending on CUDA headers or embedding scheduler policy.

Admitting a task and registering its backend resources are one logical
operation. If scheduler admission succeeds but resource registration fails,
leaving the task queued would retain a reservation that cannot ever execute.
The control layer therefore needs a small transaction boundary while remaining
independent of CUDA and transport details.

## Decision

Add `control::TaskAdmissionService` as an in-process request adapter. It:

- accepts a `TaskAdmissionRequest` containing tenant identity, declared memory,
  weight, and work units;
- delegates scheduler validation, quota reservation, and task identity to
  `core::Scheduler`;
- invokes a caller-owned, non-throwing registration callback with the accepted
  `TaskId` so the caller can bind backend resources;
- cancels the scheduler task and releases its reservation when registration
  fails;
- exposes cancellation and task lookup without owning scheduling policy,
  CUDA handles, or a transport connection.

The callback is a local composition seam, not a wire protocol. A future Unix
socket, RPC, or launcher implementation must authenticate the request and keep
the task descriptor and referenced resources alive until backend terminal
status. The transparent preload interceptor does not use this service.

## Consequences

Explicit clients get a single admission path with rollback semantics and a
stable logical `TaskId`. The control module remains CUDA-independent, and the
backend remains responsible for execution-specific resource registration.

The current service is process-local and synchronous; it does not provide
cross-process transport, persistence, authentication, or recovery. Those
concerns require a future control-plane design and must not be inferred from
this callback API.
