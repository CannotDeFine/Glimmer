# Examples

[Project home](../README.md) / [Documentation](../docs/README.md) / Examples

Choose an example by the execution path you want to use. For your own
application, start with [Usage](../docs/USAGE.md).

## Launch admission

These programs run ordinary CUDA or framework code without Glimmer API calls.
The launch-admission preload schedules covered work while CUDA execution stays
in the application process.

- [Priority demo](priority_demo/README.md): two synthetic, model-shaped CUDA
  workloads with different scheduling priorities.
- [CUDA workload matrix](cuda_workloads/README.md): allocation, accounting,
  streams and memory-lifetime examples.
- [PyTorch workloads](framework_workloads/pytorch_smoke/README.md): MLP and CNN
  workloads, including training/inference co-location.

The `cuda-gpu` preset builds the CUDA workload executables and enables hardware
checks. `cuda-lint` and `cuda-perf` also enable workload builds. PyTorch uses its
own optional Python environment, outside the CMake build.

## Explicit tasks

The [CUDA task backend](cuda_task_backend/README.md) deliberately submits task
descriptors through Glimmer. It demonstrates application-owned resource
lifetimes and scheduler-controlled dispatch, not transparent interception.

## Isolated GEMM comparison

The [GEMM comparison](gemm_comparison/README.md) compares native cuBLAS, CUTLASS
and persistent CUTLASS kernels. It is a separate diagnostic target with strict
correctness and full-device coverage gates, not an application launcher.

## Interpreting results

Examples demonstrate bounded behavior, not complete framework compatibility or
performance guarantees. Correct output, overlapping process lifetimes, actual
GPU concurrency and full-device saturation are different observations.
The launch-admission examples do not enable the centralized-execution frontend;
see [compatibility](../docs/CUDA_API_COVERAGE.md) before changing paths.
