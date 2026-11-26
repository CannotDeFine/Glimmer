# CUDA compatibility

[Project home](../README.md) / [Documentation](README.md) / CUDA compatibility

Compatibility is specific to the execution path and application operations.
Glimmer does not implement the complete CUDA stack, and its two preload
libraries are not interchangeable.

## Reference environment

The CUDA and framework examples have been exercised on a Linux NVIDIA V100
(compute capability 7.0) with CUDA 12.6. Recorded PyTorch smoke runs used driver
560.35.05, Python 3.10.12, PyTorch 2.12.1+cu126 and cuDNN 9.10.2.
This is a reference configuration, not a minimum-driver requirement or a
certification of every operation in those packages. It does not certify an
untested checkout or a different environment.

The PyTorch example keeps its package selections in
[requirements-cu126.txt](../examples/framework_workloads/pytorch_smoke/requirements-cu126.txt)
and [requirements-cu132.txt](../examples/framework_workloads/pytorch_smoke/requirements-cu132.txt).
An available setup profile is not evidence of GPU or framework coverage.
Host-only scheduling and control tests do not require a CUDA device.

## Launch-admission preload

Library: `libglimmer_cuda_interceptor.so`. Service: `glimmer_control_service`.

- Covered Runtime/Driver kernel launches can acquire shared scheduling leases;
  CUDA contexts and execution remain in the application process.
- Allocation examples exercise ordinary, stream-ordered/pool, managed and
  pitched allocations, memory copies, events and multiple streams. Accounting
  is limited to the intercepted paths and their supported lifetimes.
- The priority demo uses CUDA Runtime and cuBLAS operations. The PyTorch example
  exercises MLP inference/training, with optional CNN workloads. These are
  bounded workloads, not blanket cuBLAS, cuDNN or PyTorch compatibility claims.
- Admission of a graph replay does not provide scheduling between its nodes.
  Running ordinary kernels cannot be interrupted by launch priority.

Use the [CUDA workload matrix](../examples/cuda_workloads/README.md) and
[PyTorch example](../examples/framework_workloads/pytorch_smoke/README.md) for
the actual programs and configuration. A dynamically loaded wrapper does not
prove that every framework lookup or launch path reaches it. Static linkage,
different symbol aliases and different library versions need separate checks.

## Centralized execution frontend

Library: `libglimmer_execution_interceptor.so`. Service: `glimmer_execution_service`.
Both are explicit build options; the service owns GPU resources.

Implemented routes include bounded Driver/Runtime memory, copy, stream, event,
module and kernel operations, plus selected cuBLAS/cuBLASLt operations.
cuDNN and NCCL adapters depend on separately selected SDKs; NCCL execution also
requires explicit build-time and runtime activation. API signatures, descriptor
forms, data types, memory provenance and asynchronous lifetimes restrict these
routes. Their presence does not imply full library or framework support.

Use matching frontend/service builds and the authenticated connection described
in [Usage](USAGE.md#centralized-execution). Do not reuse launch-admission settings
or assume an unsupported request falls back to execution in the application.
Cross-process resources, multi-GPU operation and arbitrary third-party CUDA
extensions are not generally qualified by the single-device examples.

## Cooperative execution and kernel replacements

Cooperative control applies to eligible transformed kernels and selected
replacements, not arbitrary native vendor kernels. Binary-mapped variants are
restricted to exact qualified images, ABIs, shapes and devices. The SM70
CUTLASS provider targets compute capability 7.0; it is not a generic provider
for other GPU architectures.

Correct output, actual replacement selection, pause/resume behavior and
whole-device participation are different properties. A successful numerical
test or a large worker count does not establish the others. No general speedup,
latency bound or full-device saturation is promised.

## Deployment limits

- Linux is the supported platform. GPU execution requires compatible NVIDIA
  hardware, driver and libraries for the selected application.
- Preload interception and software quotas are not hardware isolation for
  mutually untrusted tenants.
- Unknown API/ABI paths must not be assumed to fall back safely. A successful
  build or startup is not complete application validation.
- Tests for one frontend, SDK or GPU do not qualify another. Before deployment,
  check complete application results, error paths and cleanup in the intended
  configuration; see [Testing](TESTING.md).
