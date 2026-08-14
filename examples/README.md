# Examples

Each workload is self-contained: its directory owns the executable source and
any generated artifacts it produces. These programs are ordinary CUDA
applications for hardware integration validation, except for the explicit task
backend demonstration, which intentionally links the scheduler-controlled
backend. They demonstrate API coverage, accounting, and task-boundary behavior,
not throughput benchmarks.

## Directory layout

| Directory | Purpose |
| --- | --- |
| `cuda_workload/` | Baseline real-GPU CUDA Runtime workload for synchronous allocation and kernel-path validation. |
| `cuda_workloads/` | Real-GPU workload matrix for asynchronous/pool, managed, pitched, and multi-stream Runtime paths. |
| `cuda_task_backend/` | Explicit Driver-API task-boundary demo that submits PTX kernels through the scheduler-controlled CUDA backend. |

## CUDA workload

The CUDA workload is built only by the `cuda-gpu` preset because it requires a
CUDA compiler and a compatible GPU runtime. It is intentionally a normal CUDA
application: it does not include Glimmer headers or call Glimmer APIs. Use it
to compare the same allocation and kernel path with and without `LD_PRELOAD`.

See [`cuda_workload/README.md`](cuda_workload/README.md) for the baseline build
and run commands, and [`cuda_workloads/README.md`](cuda_workloads/README.md)
for the workload matrix. The matrix covers stream-ordered and pool
allocation, managed memory, pitched memory, and concurrent streams. The
shared-tenant multi-process boundary remains a dedicated GPU integration test
because it must coordinate process lifetimes and quota ownership explicitly.

See [`cuda_task_backend/README.md`](cuda_task_backend/README.md) for the
explicit scheduler-controlled Driver-API task examples. They are intentionally
separate from the transparent preload workloads: the caller owns the kernel
descriptor and keeps its referenced CUDA resources alive while the controller
drives scheduler admission, weighted tenant dispatch, and CUDA event
completion.

The workload programs are built by the `cuda-gpu` preset and are registered as
optional GPU tests when CUDA hardware testing is enabled. Do not commit build
directories or temporary CSV files.
