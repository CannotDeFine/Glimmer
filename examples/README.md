# Examples

Each workload is self-contained: its directory owns the executable source and
any generated artifacts it produces. These programs are ordinary CUDA
applications for hardware integration validation. They demonstrate API
coverage and accounting behavior, not throughput benchmarks.

## Directory layout

| Directory | Purpose |
| --- | --- |
| `cuda_workload/` | Baseline real-GPU CUDA Runtime workload for synchronous allocation and kernel-path validation. |
| `cuda_workloads/` | Real-GPU workload matrix for asynchronous/pool, managed, pitched, and multi-stream Runtime paths. |

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

The workload programs are built by the `cuda-gpu` preset and are registered as
optional GPU tests when CUDA hardware testing is enabled. Do not commit build
directories or temporary CSV files.
