# Architecture

## Purpose

Glimmer is a Linux C++20 project for scheduler-based GPU sharing. This document
defines the responsibilities and allowed dependencies of its modules. It
distinguishes the current bootstrap from the planned runtime architecture.

## Current bootstrap

| Location | Responsibility |
| --- | --- |
| `CMakeLists.txt` and `cmake/` | Configure the Linux build, quality options, and project-wide targets. |
| `src/` | Contains the `glimmer` executable entry point. It currently initializes the process and emits a startup log. |
| `3rdparty/` | Contains pinned source dependencies. It currently contains `spdlog`. |
| `scripts/` | Provides convenience commands for local development. |
| `examples/` | Provides independent real-GPU CUDA workload harnesses; it does not define runtime interfaces. |
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
| `core` | `include/glimmer/core/`, `src/core/` | Provides the thread-safe quota ledger and task-boundary scheduler with explicit admission, weighted tenant queues, dispatch, completion, cancellation, and failure transitions. It has no CUDA, dynamic-linker, transport, or process-global dependencies. |
| `backend` | `include/glimmer/backend/`, `src/backend/` | Defines the internal task execution contract and provides a deterministic simulated backend. It does not own tenant fairness, quota policy, or CUDA interception. |
| `control` | `include/glimmer/control/`, `src/control/` | Defines the quota-store contract, adapts process-local quota requests to `core`, composes aggregate, task, and physical-device capacity quotas, and implements the Linux shared-memory tenant accounting store. It computes tenant/task/device-visible memory information and has no CUDA or dynamic-linker dependencies. |
| `interceptor` | `src/interceptor/` and `src/interceptor/internal/` | Provides ABI-compatible wrappers for covered CUDA Driver, PTDS stream-ordered Driver, CUDA kernel-launch, Runtime, and memory-pool APIs, routes supported symbol lookups, and owns process-local allocation metadata while using `control` for quota decisions. Kernel launches are forwarded unchanged; `GLIMMER_SCHEDULER_MODE=observe` emits a sampled boundary diagnostic. The `internal/` headers are private implementation interfaces and are not public project headers. |
| `examples` | `examples/` | Contains independent real-GPU CUDA workload harnesses. Each workload directory owns its source and generated artifacts; examples do not participate in runtime scheduling or CUDA interception. |

`control::CompositeQuota` is the isolation boundary used when a shared tenant
has a narrower per-process task limit. It admits, commits, cancels, releases,
and virtualizes memory through both quota stores while keeping shared-memory
layout details out of the interceptor. The interceptor owns only the
process-local allocation identity and delegates durable aggregate accounting to
`control`.

The interceptor is split into focused implementation units:

| Unit | Responsibility |
| --- | --- |
| `driver_api_interceptor.cc` | CUDA Driver admission, quota accounting, context/stream completion, and symbol-resolution policy. |
| `driver_api_wrappers.cc` | Exported C/CUDA ABI entry points. These wrappers only contain boundary exception handling and delegate to the interceptor implementation. |
| `runtime_api_interceptor.cc` | CUDA Runtime symbol resolution, Runtime-call reentrancy, compiler-generated and public kernel-launch forwarding, independent Runtime async/pool accounting entry points, and memory-pool import policy. |
| `driver_dispatch.cc` | Dynamic loading and guarded invocation of real CUDA Driver functions, including PTDS variants, kernel launch entry points, and Linux loader/vendor Driver discovery. |
| `nvml_dispatch.cc` | Dynamic loading and guarded invocation of the real NVML library and its optional v1/v2 entry points. |
| `nvml_api_wrappers.cc` | Exported NVML ABI entry points that route to the interceptor's NVML presentation policy. |
| `symbol_interceptor.cc` | `dlsym` interception, caller classification, and safe delegation to the real loader. |
| `symbol_registry.cc` | The single registry of exported aliases used by `dlsym` and `cuGetProcAddress`. |
| `internal/allocation_registry.cc` | Process-local allocation metadata, release state, and deferred stream completion. |
| `internal/diagnostics.cc` | Allocation-free diagnostics for loader and accounting failure paths, plus opt-in structured kernel-launch and virtualized memory-view observations. |

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
`include/glimmer/control/` contain the stable project API. CUDA interceptor
headers stay under `src/interceptor/internal/` because they are private
implementation contracts and expose CUDA ABI details only to the preload
library and its tests.

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
- A backend owns only the resources it creates and reports failures through its
  contract; the core owns admission and accounting decisions.
- An interceptor runs inside an application process and must treat all CUDA
  handles and device pointers as process-local.

## Change rules

- Add a module responsibility here before implementing a new top-level module.
- Record an ADR before reversing a dependency direction or exposing a new
  cross-module contract.
- Add or update tests whenever a module's observable behavior changes.
