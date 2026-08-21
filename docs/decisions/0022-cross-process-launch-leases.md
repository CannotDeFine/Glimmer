# 0022: Coordinate transparent launch admission through the control plane

Status: Accepted

## Context

The first transparent launch gate was process-local. That bounds concurrent
work inside one process, but two CUDA processes could each observe an empty
local gate and exceed the intended GPU-wide concurrency policy. A transparent
client must also be bound to the task it submitted; a generic `CLAIM` operation
cannot safely return another tenant's task to a CUDA process.

## Decision

Glimmer adds an opt-in control-plane launch-lease mode:

- The existing Linux Unix-socket control service remains the single scheduler
  owner. No scheduler state or CUDA handles cross the process boundary.
- A preload client submits a one-unit launch task with its configured tenant
  and weight, then polls `CLAIM <task_id>`. The scheduler dispatches that task
  only when it is next under the configured policy and a running slot exists.
- The client forwards the CUDA launch only after receiving its lease. A
  completion event then causes `COMPLETE <task_id>`; while the event is
  pending, the client renews the lease with `HEARTBEAT`. Launch or event
  failures use `FAIL <task_id>`.
- Lease ownership can be bound to the peer PID, UID, and process start-time
  identity with `--bind-leases-to-process`; in that mode submission and every
  lease operation require a valid authenticated peer, and only the submitting
  process may claim, renew, or finish its task. A configured lease timeout
  remains the crash-recovery fallback.
- The original no-socket process-local gate remains the default. The remote
  path is enabled only by the trusted launcher with
  `GLIMMER_SCHEDULER_CONTROL_SOCKET`.
- The versioned protocol keeps `CLAIM` compatibility and adds an optional task
  id. `METRICS` is a separate diagnostic operation so existing `STATS` clients
  retain their response shape.

The scheduler exposes FIFO, weighted round-robin, deficit round-robin, and
strict priority selection. Priority selects the largest task priority and
breaks ties by submission order; the other policies ignore that field. DRR
treats `work_units` as a scheduling cost and `weight` as the tenant quantum.
Scheduler snapshots aggregate queue-wait and service-time
microseconds; they are diagnostic measurements, not hard latency guarantees.

## Consequences

Two or more preload processes can share one global launch-slot policy without
linking an SDK or moving CUDA pointers through IPC. Admission fairness is
enforced at task boundaries and can be tested with ordinary Linux processes.
The remote path adds Unix-socket latency to every admitted launch and depends
on a running control service; transport failure therefore fails closed for the
enforced launch. It still does not preempt an already running CUDA kernel.

The central service is deliberately not a durable database or a security
boundary against a process that bypasses `LD_PRELOAD`. A future low-latency
shared-memory or batched-lease transport may preserve this control contract
without changing CUDA wrappers.

## Verification

- Core tests verify policy-preserving task-specific dispatch, DRR quantum
  behavior, and latency aggregate invariants.
- Protocol and endpoint tests verify optional task-bound claims and the
  separate metrics response.
- The remote process test starts the real control-service executable and runs
  two independent worker processes against a single global slot. It is skipped
  only when the host cannot bind Linux Unix sockets.
