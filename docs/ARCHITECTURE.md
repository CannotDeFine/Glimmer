# Architecture

## Purpose

Glimmer is a Linux C++20 project for scheduler-based GPU sharing. This document
defines the responsibilities and allowed dependencies of its modules. It
distinguishes the current bootstrap from the planned runtime architecture.

## Current bootstrap

| Location | Responsibility |
| --- | --- |
| `CMakeLists.txt` and `cmake/` | Configure the Linux build, quality options, and project-wide targets. |
| `src/` | Contains the `glimmer` entry point plus the opt-in `glimmer_control_service` and `glimmer_control_client` binaries for the transport validation path. |
| `3rdparty/` | Contains pinned source dependencies. It currently contains `spdlog`. |
| `scripts/` | Provides convenience commands for local development. |
| `examples/` | Provides independent real-GPU CUDA workload harnesses and an explicit task-boundary demonstration; it does not define runtime interfaces. |
| `tests/` | Contains core, simulated-backend, control, interceptor, and optional CUDA GPU tests. CTest verifies startup and module behavior. |
| `docs/` | Defines engineering rules, architecture, dependencies, tests, and decisions. |

## Planned runtime modules

The following modules are planned. Do not create a module until it has a
concrete responsibility and a testable interface.

| Module | Responsibility | Must not own |
| --- | --- | --- |
| `core` | Domain types, task lifecycle, quota accounting, and scheduler policy. | CUDA calls, dynamic-linker logic, transport, or process-global state. |
| `backend` | Defines GPU execution and monitoring contracts, then implements simulated and CUDA/NVML backends. | Tenant fairness policy or public transport protocol. |
| `control` | Maps local API or IPC requests to core operations and authenticates request identity. | Direct GPU calls or scheduling policy. |
| `interceptor` | Provides a preloadable shared library that observes or gates CUDA API calls. | Scheduler policy, durable accounting, or blocking work while resolving symbols. |
| `metrics` | Exposes measurements and health information gathered from other modules. | Ownership of scheduler or backend state. |
| `app` | Parses process configuration, initializes modules, and manages process lifetime. | Domain policy or resource-accounting decisions. |

## Implemented modules

| Module | Location | Current responsibility |
| --- | --- | --- |
| `core` | `include/glimmer/core/`, `src/core/` | Provides the thread-safe quota ledger and task-boundary scheduler with explicit admission, configurable FIFO, weighted-round-robin, deficit-round-robin, or strict-priority dispatch policies, weighted tenant queues, configurable concurrent dispatch slots, optional queued-work backpressure, completion, cancellation, and failure transitions. It has no CUDA, dynamic-linker, transport, or process-global dependencies. |
| `backend` | `include/glimmer/backend/`, `src/backend/` | Defines the internal task execution contract, provides a deterministic simulated backend, provides the single-threaded executor that translates backend progress into scheduler terminal transitions, and optionally provides the explicit CUDA task backend under `src/backend/cuda/`. It does not own tenant fairness, quota policy, or transparent CUDA interception. |
| `control` | `include/glimmer/control/`, `src/control/` | Defines the quota-store contract, adapts process-local quota requests to `core`, provides transactional explicit-task admission with backend-registration rollback, composes aggregate, task, and physical-device capacity quotas, implements the Linux shared-memory tenant accounting store, provides authenticated Unix-socket server/client transport adapters, coordinates optional cross-process transparent launch leases, and exposes bounded read-only scheduler/quota stats and latency snapshots. It computes tenant/task/device-visible memory information and has no CUDA or dynamic-linker dependencies. |
| `app` | `src/control_service_main.cc`, `src/control_client_main.cc` | Provides standalone control-service/client entry points. The service supports deterministic simulated execution and remote worker leases; neither binary serializes or owns CUDA resources. |
| `interceptor` | `src/interceptor/` and `src/interceptor/internal/` | Provides ABI-compatible wrappers for covered CUDA Driver, PTDS stream-ordered Driver, CUDA kernel-launch, Runtime, memory-pool, IPC, external-memory, array, graphics, and graph-memory APIs, routes supported symbol lookups, and owns process-local allocation metadata while using `control` for quota decisions. Accounted allocations update the ledger; APIs whose ownership or byte lifetime cannot be reconstructed are explicitly fail-closed under quota and delegated when quota is disabled. Observe mode forwards launches and emits a sampled boundary diagnostic; enforce mode admits covered launches through a process-local `control::LaunchGate`, or an opt-in authenticated Unix-socket launch lease, and releases the slot after an internal CUDA event completes. Neither path provides kernel preemption. The `internal/` headers are private implementation interfaces and are not public project headers. |
| `examples` | `examples/` | Contains independent real-GPU CUDA workload harnesses, optional framework workload programs, and the opt-in `cuda_task_backend/` demonstration. Each directory owns its source and generated artifacts; examples do not modify transparent interception behavior or define scheduler policy. |

`control::CompositeQuota` is the isolation boundary used when a shared tenant
has a narrower per-process task limit. It admits, commits, cancels, releases,
and virtualizes memory through both quota stores while keeping shared-memory
layout details out of the interceptor. The interceptor owns only the
process-local allocation identity and delegates durable aggregate accounting to
`control`.

The interceptor is split into focused implementation units:

| Unit | Responsibility |
| --- | --- |
| `driver_api_interceptor.cc` | CUDA Driver admission, quota accounting, context/stream completion, launch-gate admission, and symbol-resolution policy. |
| `driver_api_wrappers.cc` | Exported C/CUDA ABI entry points. These wrappers only contain boundary exception handling and delegate to the interceptor implementation. |
| `runtime_api_interceptor.cc` | CUDA Runtime symbol resolution, Runtime-call reentrancy, compiler-generated and public kernel-launch admission, independent Runtime async/pool accounting entry points, and memory-pool import policy. |
| `driver_dispatch.cc` | Dynamic loading and guarded invocation of real CUDA Driver functions, including PTDS variants, kernel launch and event entry points, and Linux loader/vendor Driver discovery. |
| `nvml_dispatch.cc` | Dynamic loading and guarded invocation of the real NVML library and its optional v1/v2 entry points. |
| `nvml_api_wrappers.cc` | Exported NVML ABI entry points that route to the interceptor's NVML presentation policy. |
| `symbol_interceptor.cc` | `dlsym` interception, caller classification, and safe delegation to the real loader. |
| `symbol_registry.cc` | The single registry of exported aliases used by `dlsym` and `cuGetProcAddress`. |
| `internal/allocation_registry.cc` | Process-local allocation metadata, release state, and deferred stream completion. |
| `launch_completion_tracker.cc` | Records a Driver event after each admitted launch, drains batched events, and completes or fails the corresponding process-local launch lease from a monitor thread. |
| `internal/diagnostics.cc` | Allocation-free diagnostics for loader and accounting failure paths, plus opt-in structured kernel-launch, virtualized memory-view, and launch-path timing observations. |

## Dependency direction

```text
app
 ├── control ──> core <── backend contract
 ├── metrics  ──> core / backend observations
 └── interceptor ──> control contract

simulated backend ──> backend contract
CUDA/NVML backend ──> backend contract
```

Dependencies point toward stable abstractions. `core` remains independent of
CUDA, NVML, `dlopen`, `dlsym`, transport libraries, and metrics exporters.
CUDA-specific code stays in a backend or interceptor implementation. The
interceptor communicates through a narrow control contract and does not call
into scheduler internals directly.

The small public header tree is intentional: `include/glimmer/core/` and
`include/glimmer/control/` contain the stable project API, while
`include/glimmer/backend/` contains the internal execution contract. CUDA
interceptor headers stay under `src/interceptor/internal/`, and explicit CUDA
backend headers stay under `src/backend/cuda/internal/`, because they are
private implementation contracts and expose CUDA ABI details only to their
owning targets and tests.

## Runtime boundaries

- A tenant is the unit of resource accounting and scheduling fairness.
- The first transparent task-memory isolation unit is a CUDA process attached
  to the preload library. A future control-plane lease may bind a logical
  scheduler task to that process, but a CUDA wrapper must never infer a task
  identity from an untrusted pointer or launch argument.
- Multi-process tenant accounting is governed by
  [ADR 0005](decisions/0005-shared-quota-control-plane.md); shared state must
  remain behind the `control` contract and must not leak into CUDA wrappers or
  the `core` ledger.
- The scheduler operates at explicit task boundaries. It does not claim to
  preempt arbitrary running CUDA kernels.
- Scheduling policy is selected when the scheduler is constructed. The
  current policies are FIFO, weighted round-robin, deficit round-robin, and
  strict priority; all control dispatch order only and keep quota admission
  and terminal state transitions in the scheduler core. Priority selects the
  largest task priority and preserves submission order for ties. DRR uses
  `work_units` as task cost and `weight` as tenant quantum; other policies
  ignore priority.
- Explicit clients use `control::TaskAdmissionService` to bind a logical task
  admission to backend-resource registration. A failed registration cancels the
  queued task and releases its scheduler reservation.
- If another component advances the same scheduler directly, it must call
  `TaskAdmissionService::notify_scheduler_change()` so blocked task-specific
  waits observe that progress; the simulated control service does this after
  each externally executed state transition.
- External lease cancellation is limited to queued tasks. A running CUDA task
  remains owned by its backend until a completion or failure event, because the
  scheduler cannot preempt an arbitrary kernel safely.
- The transport-neutral `control::task_protocol` codec carries only versioned
  task metadata and lease state. It never serializes CUDA handles, device
  pointers, or kernel argument addresses; a future transport adapter must
  authenticate peers and bind the decoded request to the admission service.
- `control::UnixSocketControlServer` is the first Linux transport adapter. It
  authenticates `SO_PEERCRED`, bounds each request line and I/O operation, and
  serves sequential request/response pairs on persistent client connections;
  each connection is handled independently so one idle client cannot block
  other tenants. It delegates all semantics to the endpoint and does not own
  CUDA resources. `UnixSocketControlClient` reuses one connection per calling
  thread and discards a connection after any transport failure without
  retrying a side-effecting request.
- In remote execution mode, `ACQUIRE` combines submission with an immediate
  task-specific claim when a scheduler slot is available. If the task is
  queued, the client uses a bounded task-specific `WAIT` request so the
  service blocks the connection until the task is dispatchable or the wait
  expires; it falls back to `CLAIM` polling when talking to an older service.
  This removes busy polling from the contended launch path while preserving
  the versioned `SUBMIT`/`CLAIM` operations for explicit clients. Up to the
  configured concurrency limit may run at once. A worker must report
  `COMPLETE` or `FAIL`; the service retains the reservation while the lease is
  running.
- When a lease timeout is configured, workers renew running leases with
  `HEARTBEAT`; the service reaps an unrenewed lease as `FAILED` and releases its
  reservation. With timeout disabled, a running lease remains visible until an
  explicit terminal report or service restart.
- The control service can opt into process-bound leases. The Unix transport
  derives a PID/UID/start-time identity from `SO_PEERCRED` and `/proc`; submit,
  claim, heartbeat, completion, and failure must then use a valid peer, and
  only the submitting process may claim it while only the claiming process may
  renew or complete that lease. Unscoped `CLAIM` is rejected because it cannot
  prove pending-task ownership. This is an ownership check, not CUDA kernel
  preemption, and the default remains unbound for CLI compatibility.
- The optional queued-task capacity rejects new work with explicit
  `QUEUE_FULL` backpressure before quota reservation; it does not limit running
  tasks or change weighted ordering.
- The control service accepts `--trace-scheduler` as a disabled-by-default
  diagnostic. It emits one stderr record for each successful admission-service
  dispatch, including a monotonic sequence, task identity, tenant, priority,
  and dispatch timestamp. The trace is observational only; it does not change
  policy, admission, lease ownership, or CUDA execution.
- The versioned `STATS` control operation is read-only and reports task-state
  counts, quota bytes, and configured scheduler limits. It is a diagnostic
  snapshot, not durable metrics storage or a CUDA resource API.
- A backend owns only the resources it creates and reports failures through its
  contract; the core owns admission and accounting decisions.
- An interceptor runs inside an application process and must treat all CUDA
  handles and device pointers as process-local.
- In scheduler enforce mode, the interceptor blocks a covered launch caller
  before forwarding until the process-local launch gate admits it. A Driver
  event recorded on the same stream releases the slot after the work reaches
  that event. Event support is required; an unavailable event path fails
  closed rather than forwarding an unenforced launch.
- With `GLIMMER_SCHEDULER_CONTROL_SOCKET`, the same gate uses a one-unit
  `ACQUIRE` request and sends bounded task-specific `WAIT` requests only when
  the first request is queued. The service applies the global policy and may
  bind the lease to the requesting process identity. Completion and failure
  are reported over the same protocol. A missing or failed remote lease fails
  the launch closed.
- `GLIMMER_SCHEDULER_BATCH_SIZE` may group consecutive covered launches within a
  process under one lease, including launches from multiple host threads. The
  completion tracker records one event per launch and closes the lease only
  after every acquired call has tracked its event; any launch or event error
  fails the entire batch. This reduces remote transport
  overhead without moving CUDA handles across processes. The default batch
  size is one, and batching must not be presented as kernel preemption or a
  memory-isolation boundary.
- `GLIMMER_TRACE_LAUNCH_TIMINGS=1` enables an allocation-free diagnostic path
  that measures launch admission, remote request transport, wait requests and
  legacy claim polling, CUDA forwarding, event tracking, and completion
  reporting. It is disabled by default and does not alter admission,
  scheduling, or completion behavior.
- Transparent launch admission remains a task-boundary mechanism: it does not
  preempt a running kernel or capture CUDA graph/cooperative-launch state.

## Change rules

- Add a module responsibility here before implementing a new top-level module.
- Record an ADR before reversing a dependency direction or exposing a new
  cross-module contract.
- Add or update tests whenever a module's observable behavior changes.
