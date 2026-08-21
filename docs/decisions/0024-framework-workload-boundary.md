# ADR 0024: Keep framework workloads optional and outside the core build

- Status: Accepted
- Date: 2026-08-21
- Decision owners: Glimmer maintainers

## Context

The project needs evidence from real framework workloads, not only standalone
CUDA and cuBLAS programs. Frameworks such as PyTorch are large, independently
versioned runtime dependencies. Their CUDA, driver, and allocator behavior is
part of the compatibility surface we want to measure, but not part of the
Glimmer scheduler or interceptor ABI.

## Decision

Framework workloads live under `examples/framework_workloads/` and run as
user-supplied programs. They may depend on a framework installed in the host
environment, but they must not:

- add the framework to CMake or the default build;
- vendor framework binaries, Python environments, or model data;
- include Glimmer headers or call private Glimmer interfaces;
- become mandatory CTest cases.

The workload must report the framework version, CUDA version, device, timing
distribution, memory peak, and validation result. Native, observe, and enforce
runs use the same workload arguments so scheduler overhead can be measured
separately from framework overhead.

## Consequences

This keeps the core build reproducible and Linux/CUDA-focused while allowing
compatibility experiments against a user-selected framework version. A
framework smoke result is evidence for the tested version and workload, not a
claim of universal framework support. Framework-specific API gaps are tracked
in the CUDA API coverage documentation before expanding interceptor claims.
