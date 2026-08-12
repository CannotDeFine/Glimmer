# 0008: Use process-scoped task limits for transparent CUDA memory isolation

Status: Accepted

## Context

Glimmer already has two related but different forms of accounting:

- `core::Scheduler` reserves the declared `TaskSpec::memory_bytes` during task
  admission and releases that reservation at a terminal task transition;
- the CUDA interceptor records actual device allocations by process, device,
  context, pointer, and allocation kind, then charges a process-local or
  tenant-shared quota through the `control` contract.

The scheduler's `TaskId` is not currently available at a transparent CUDA ABI
boundary. A CUDA application can launch work without linking a Glimmer SDK,
and the interceptor cannot infer which logical scheduler task owns an
arbitrary pointer or kernel launch. Treating an unverified environment value as
the scheduler task identity would make isolation and accounting claims
unreliable.

## Decision

The first task-memory-isolation increment defines an execution task as one
CUDA process attached to the Glimmer preload library. The process is the
smallest identity that the interceptor can establish safely and consistently
for all covered Driver and Runtime allocation paths.

The implementation will add an optional process-scoped task limit:

- process-local quota mode keeps the existing `GLIMMER_MEMORY_LIMIT_BYTES`
  behavior; that limit is the task limit for the process;
- shared tenant mode keeps `GLIMMER_MEMORY_LIMIT_BYTES` as the tenant-wide
  aggregate limit and may additionally configure
  `GLIMMER_TASK_MEMORY_LIMIT_BYTES` as the per-process task limit;
- when `GLIMMER_TASK_MEMORY_LIMIT_BYTES` is absent in shared mode, the existing
  tenant-only behavior remains unchanged;
- task and tenant configuration is trusted launcher/control-plane input, not a
  value that an untrusted workload may raise after initialization.

For every covered allocation, the interceptor must satisfy:

```text
task-limit admission
    -> tenant-limit admission (when shared mode is enabled)
    -> real CUDA allocation
    -> commit both reservations
    -> record the process-local allocation identity
```

The allocation is rejected before the real CUDA call if either limit would be
exceeded. A real allocation failure cancels both reservations. If cleanup or a
commit cannot be proven complete, the implementation retains the conservative
charge and enters the existing degraded state rather than under-reporting
usage. Successful release, including deferred stream-ordered release, must
release both the task and tenant charges exactly once.

Virtualized CUDA and NVML memory queries report the most restrictive visible
capacity from the task limit, tenant limit, and physical device availability.
The task limit must therefore be visible to the process that owns the task,
while the tenant view continues to aggregate all participating processes.

Physical capacity is resolved lazily through the interceptor's Driver dispatch
for each device that is queried or allocated. The dispatch supports Linux
installations where `libcuda.so.1` is a loader shim and the vendor Driver is a
separate already-loaded object; this is required for transparent operation on
WSL and similar split-driver layouts. The effective limit is:

```text
min(configured task/tenant limit, physical device total memory)
```

Before a real allocation is submitted, the interceptor also checks the latest
physical free-memory value when the current context makes that query
available. This check is advisory against allocations outside Glimmer and
the CUDA driver remains authoritative for races; a driver allocation failure
still follows the normal rollback path.

The core scheduler remains the owner of logical `TaskId`, queue policy, and
task-boundary transitions. It does not receive CUDA pointers or dynamic-loader
state. A future controller may bind a logical `TaskId` to a process execution
lease, but that requires an explicit control-plane contract and is outside
this increment.

## Consequences

Ordinary CUDA applications receive a real per-process memory ceiling without
source changes or a Glimmer SDK. Shared tenants can enforce both an aggregate
quota and an individual task ceiling, preventing one process from consuming
the entire tenant quota.

The first increment does not provide independent limits for multiple logical
tasks inside one process. It also does not provide kernel preemption, kernel
time slicing, or a security boundary against a process that bypasses the
preload library. Those guarantees require a trusted task context and a
backend/control-plane design that can identify ownership at every allocation
and completion boundary.

The task-limit composition must be tested for synchronous, asynchronous,
Runtime, Driver, memory-pool, and VMM allocation paths, including rejection,
rollback, deferred release, process exit, and cross-process shared accounting.
