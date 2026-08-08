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
| `tests/` | Will contain project tests. CTest currently verifies that the executable starts. |
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
| `core` | `include/glimmer/core/`, `src/core/` | Provides a thread-safe in-process quota ledger with explicit reservation, commit, cancellation, and release transitions. It has no CUDA, dynamic-linker, transport, or process-global dependencies. |
| `control` | `include/glimmer/control/`, `src/control/` | Adapts quota requests to `core` and computes tenant-visible memory information. It has no CUDA or dynamic-linker dependencies. |

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

## Runtime boundaries

- A tenant is the unit of resource accounting and scheduling fairness.
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
