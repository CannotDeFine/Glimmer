# 0019: Optionally bind remote leases to the peer process

Status: Accepted

## Context

The remote task protocol keeps CUDA handles and pointers process-local, but a
same-UID client that learns a task ID could otherwise renew or complete another
worker's running lease. The Unix socket transport already authenticates the
peer UID through `SO_PEERCRED`; it can also provide a PID and a Linux process
start-time value without putting identity data on the task wire format.

The existing unbound mode remains useful for simple command-line smoke tests,
where `claim` and `complete` may be separate client processes. Process-bound
leases therefore need to be explicit and backwards compatible.

## Decision

Add an opt-in `--bind-leases-to-process` control-service option:

- the Unix socket server derives a peer identity from `SO_PEERCRED` and
  `/proc/<pid>/stat`;
- the pending lease records the peer PID, UID, and process start-time ticks at
  submission, and the active lease preserves that identity when it is claimed;
- submission, claim, `HEARTBEAT`, `COMPLETE`, and `FAIL` must come from the
  same identity;
- a recycled PID does not match because its start-time value changes;
- no identity or CUDA handle is serialized in the protocol; and
- the default remains unbound for compatibility with the existing CLI smoke
  workflow.

When process binding is enabled, a submission or claim without a trusted local
peer identity is rejected. A worker must keep the same process for the
complete lease lifecycle. Lease timeout and expiry recovery remain unchanged.

## Consequences

The control service can enforce process ownership of a running lease without
changing the CUDA ABI or trusting a client-supplied PID. The feature is Linux
specific because it relies on `SO_PEERCRED` and `/proc` process metadata. It is
an ownership check, not a kernel-preemption mechanism: a running CUDA kernel
still ends through backend completion, failure, or lease expiry.
