# 0005: Use a versioned shared quota control plane for multi-process accounting

Status: Accepted

## Context

The current quota implementation is process-local. That is a useful and
deterministic first milestone, but it cannot enforce one limit for several
CUDA processes that belong to the same tenant. A preload library must also
remain safe when it is entered concurrently by multiple threads and when one
of the participating processes terminates unexpectedly.

The shared state must therefore provide a small, explicit control plane rather
than expose CUDA handles or allocation pointers. It must preserve the existing
reservation, commit, cancellation, and release contract while making the
admission decision across processes atomic.

## Decision

The quota implementation uses a Linux POSIX shared-memory region per tenant
when shared mode is explicitly selected. The region is owned and accessed
through the `control` contract; the `core` ledger remains independent of
shared memory and CUDA. A process-local allocation registry continues to own
pointer and CUDA-context metadata.
Quota counters are selected by CUDA device identity; the region must not
silently combine memory from different devices into one quota.

The shared region is an accounting control plane, not a CUDA-memory allocator,
scheduler, or security boundary.

### Tenant and process identity

- A tenant is the quota ownership boundary. Its identity is an opaque,
  canonical control-plane key supplied by configuration or a future IPC
  controller; it is not inferred from a PID.
- The shared-memory object name is derived from the tenant key and protocol
  version. The raw tenant key is not used as an unchecked filesystem name.
- A process registration contains at least the PID, a PID-reuse-resistant
  process-start identity, a registration generation/token, and process state.
  PID alone is never sufficient to identify an owner.
- A registration is leased while the process is active. The implementation may
  use Linux `pidfd` where available and must otherwise validate the process
  start identity before treating a stale PID as the same process.

The exact configuration variable or IPC message that supplies the tenant key
is now defined for the preload MVP as `GLIMMER_QUOTA_MODE=shared` together
with `GLIMMER_QUOTA_TENANT_ID`, `GLIMMER_MEMORY_LIMIT_BYTES`, and the optional
`GLIMMER_QUOTA_DEVICE_ID`. Without the explicit shared mode, the existing
`GLIMMER_MEMORY_LIMIT_BYTES` process-local mode remains the default.

### Shared-memory layout

The first implementation uses a fixed, versioned layout so that no C++ object
layout or allocator ABI crosses a process boundary. The region contains:

1. A header with a magic value, protocol version, header size, region size,
   tenant-key bytes, generation, and initialization state.
2. A bounded array of device-quota records. Each record contains the device
   identity, configured limit, and the aggregate `reserved_bytes` and
   `committed_bytes` counters for that device.
3. A process-shared robust POSIX mutex.
4. A bounded array of process slots. Each slot contains the process identity,
   lifecycle state, heartbeat/last-seen data, and per-device reserved and
   committed byte totals.

Only fixed-width integers, byte arrays, and explicitly initialized POSIX
synchronization objects may be stored in the region. It must not contain raw
pointers, CUDA handles, `std::mutex`, C++ containers, or other process-local
addresses. Region size and slot limits are validated before use.

The shared object is created with an explicit create-or-open protocol. A
process must observe the header's ready state and compatible version before it
may lock or update the counters. A partially initialized or incompatible
region is rejected rather than interpreted.

### Reservation and accounting protocol

All shared counter transitions occur while holding the region mutex. The lock
is never held while calling CUDA, loading a symbol, invoking a callback, or
performing potentially blocking I/O.

The protocol is:

```text
reserve(device, size)
  if device has no available region record: reject
  if a new device record is required, initialize it with the configured
  tenant-default limit
  if device.committed_bytes + device.reserved_bytes + size > device.limit_bytes:
    reject
  else increase the caller's and device's reserved_bytes

real CUDA allocation

commit(reservation)
  move the reservation from reserved_bytes to committed_bytes for its device

cancel(reservation)
  remove the reservation from reserved_bytes

real CUDA release

release(allocation)
  decrease committed_bytes for its device only after a successful real release
```

Each reservation carries an opaque generation/token that identifies the
registering process and device, and prevents a second commit or cancellation
from changing the counters. A real allocation failure cancels its reservation.
If the real allocation succeeds but commit or local metadata recording fails,
the interceptor attempts to release the real allocation. It cancels or rolls
back the reservation only after cleanup succeeds; if cleanup cannot prove that
the allocation was released, it abandons the reservation so the charge remains
and enters its existing conservative degraded state. For stream-ordered
cleanup, cancellation is performed only after stream synchronization confirms
completion; if completion cannot be proven, the reservation remains charged
and later allocations are rejected.

The shared region stores aggregate and per-process byte totals only. The local
allocation registry remains the authority for pointer lookup, allocation kind,
CUDA device/context identity, and deferred stream completion. Unknown,
duplicate, or failed releases never reduce shared usage.

### Synchronization and lock ownership

- The first version uses one `PTHREAD_PROCESS_SHARED` and
  `PTHREAD_MUTEX_ROBUST` mutex per region. The initialization attributes are
  checked during creation. Compatibility is recorded through the protocol
  version and fixed layout fields; the POSIX mutex attributes are not copied
  into the shared header.
- The shared quota mutex is the only lock in the shared-memory protocol.
  Operations acquire it for a short state transition and release it before
  entering CUDA or other project subsystems.
- No code may acquire the shared quota mutex while holding the local
  allocation-registry lock, and no cross-process lock ordering may be added
  implicitly. This avoids lock cycles between process-local and shared state.
- If a lock acquisition reports owner death, the caller validates the region
  header and counters, calls `pthread_mutex_consistent`, and continues only if
  the state is internally valid. A separate PID/start-time recovery scan
  reclaims registrations proven dead; corrupt or ambiguous state fails closed.

### Crash recovery and stale registrations

Crash recovery distinguishes reservations from committed allocations:

- Reservations belonging to a process that is proven dead are reclaimed. They
  represent work that has not been committed and must not permanently consume
  capacity.
- Committed bytes are not reclaimed solely because a heartbeat is old or a
  robust mutex reports owner death. CUDA allocations may still exist until the
  driver tears down the process, so premature reclamation could allow a real
  quota violation.
- Committed-byte reclamation requires PID-reuse-safe death evidence plus a
  generation fence and a conservative quiescence policy. The first
  implementation may fail closed and require explicit region cleanup when that
  evidence is unavailable.
- The implementation uses a five-second committed-byte recovery grace after
  dead-process evidence. Reservations are reclaimed immediately, while
  committed counters remain charged during the grace window; this deliberately
  favors temporary under-admission over a quota violation while CUDA teardown
  completes.
- A process exit path performs best-effort deregistration, but correctness
  cannot depend on destructors or `atexit` handlers running.

Recovery is idempotent. Repeating recovery for the same generation must not
double-decrement counters, and a newly registered process must never inherit a
stale slot or stale bytes.

After `fork()`, inherited shared-quota state is not reused as the child’s
identity. The first child operation replaces process-local reservation and
allocation metadata, then registers a fresh process slot. A child that
performs no quota operation cannot mutate the parent slot during destruction.
The child must not use an inherited CUDA driver handle or enter a CUDA call
while the parent was inside a CUDA call; those loader and driver states are not
fork-safe. Applications that need CUDA in a child must use the normal
`fork()`-then-`exec()` boundary, or create the child before CUDA initialization.

Administrative `remove_region` takes the region and file locks, refuses to
remove a region with tracked usage, marks an idle compatible region unusable,
and unlinks it while the file lock is held. This prevents an opener that raced
with cleanup from registering into an old valid inode; incompatible or stale
region layouts may be removed explicitly.

### Migration and compatibility

The existing `ProcessMemoryQuota` contract remains the interceptor's stable
admission interface. `SharedMemoryQuota` is implemented behind the same
control-layer contract rather than teaching CUDA wrappers about shared-memory
layout.

- If no shared tenant control is configured, behavior remains process-local and
  compatible with ADR 0004.
- Shared mode is enabled explicitly with `GLIMMER_QUOTA_MODE=shared`,
  `GLIMMER_QUOTA_TENANT_ID`, and `GLIMMER_MEMORY_LIMIT_BYTES`; the interceptor
  never silently combines a process-local limit with a shared limit.
- `GLIMMER_QUOTA_DEVICE_ID` selects the device record initialized at attach
  time. The MVP uses the configured limit for additional device records in the
  same tenant region and keeps their counters separate.
- The protocol version, region size, and feature flags are validated at attach
  time. Incompatible regions fail closed with a documented configuration error.
- The shared implementation must preserve the existing wrapper semantics for
  rejection, real CUDA failures, rollback, deferred stream release, and visible
  memory queries.

Changing the region layout, identity rules, recovery guarantees, or selection
semantics requires a superseding ADR and a protocol-version migration plan.

## Scope and non-goals

This decision covers one Linux host and one tenant control plane. It does not
provide:

- physical GPU partitioning, kernel preemption, or scheduler fairness;
- a security boundary against a process that bypasses the preload library;
- cross-host accounting or a durable database;
- CUDA VMM, memory-pool, NVML, or other API coverage not already accepted in
  the CUDA coverage matrix;
- an unbounded process table or a dynamically resized shared-memory ABI.

An external controller or daemon may replace the shared-memory transport in a
future decision while preserving the control contract.

## Alternatives considered

- **Keep one ledger per process:** preserves the current simplicity but cannot
  enforce a tenant-wide limit.
- **Use a lock file with ad-hoc counters:** makes the data format and crash
  recovery harder to validate and does not provide a versioned in-memory
  contract.
- **Require an external quota daemon immediately:** can provide stronger
  ownership and observability, but adds a service lifecycle and IPC dependency
  before the local control contract is stable.
- **Use CUDA or MIG for isolation:** provides hardware-level partitioning, not
  the software scheduling and shared-quota behavior targeted by Glimmer.

## Consequences

The quota admission decision becomes atomic across participating processes and
can enforce one tenant limit without putting CUDA pointers in shared state.
No-GPU tests can exercise the protocol using ordinary Linux processes, while
GPU tests remain focused on CUDA ABI and driver behavior.

The implementation must handle shared-memory bootstrap, robust mutex recovery,
PID reuse, bounded process slots, protocol validation, and conservative
degraded behavior. The shared region adds operational cleanup and observability
requirements, and a stale or corrupt region must never be treated as valid
accounting state.

## Verification status

The deterministic no-GPU suite currently covers:

- two or more processes sharing one quota boundary and per-device counters;
- reservation commit, cancellation, real-allocation rollback, and release;
- process registration, PID-reuse-resistant identity, and stale-region,
  version, and size rejection;
- idempotent reservation recovery after an unclean process exit and robust
  mutex owner death;
- conservative handling of committed bytes after an unclean process exit;
- unchanged process-local behavior when shared mode is not configured.

These tests run without a GPU. CUDA integration tests then verify that the
existing Driver and Runtime wrappers use the shared control contract exactly
once for representative synchronous and stream-ordered workloads.
