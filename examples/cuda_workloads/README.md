# CUDA workload matrix

[Project home](../../README.md) / [Documentation](../../docs/README.md) / [Examples](../../examples/README.md) / CUDA workloads

These programs are ordinary CUDA Runtime applications. They do not include
Glimmer headers or call Glimmer APIs; the same executable can be run with no
preload, with the interceptor, or with different quota settings.

The directory contains the baseline synchronous workload and a matrix of
specialized allocation and stream behaviors:

| Workload | CUDA behavior exercised |
| --- | --- |
| `glimmer_cuda_workload` | Synchronous allocation, full-buffer kernel access, CUDA events, and visible-memory reporting. |
| `glimmer_cuda_async_workload` | Stream-ordered allocation, explicit memory-pool allocation, asynchronous kernel work, and deferred free. |
| `glimmer_cuda_managed_workload` | Managed-memory allocation, device kernel access, synchronization, host validation, and release. |
| `glimmer_cuda_pitched_workload` | Pitched allocation, pitch-aware kernel indexing, 2D copy, validation, and release. |
| `glimmer_cuda_multistream_workload` | Two independent streams, concurrent kernel submissions, events, asynchronous copies, and independent buffers. |

Build all workload examples with:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu -j2
```

Run one workload with an 8 MiB virtual quota:

```sh
env -u LD_AUDIT GLIMMER_MEMORY_LIMIT_BYTES=8388608 \
    GLIMMER_TRACE_MEMORY_INFO=1 \
    LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" \
    "$PWD/build/cuda-gpu/examples/cuda_workloads/glimmer_cuda_async_workload"
```

The specialized workloads print a key-value summary with requested bytes,
visible memory snapshots and a validation flag. The baseline instead prints a
CSV header and row with CUDA status codes; it does not validate the buffer's
contents. A non-zero exit status indicates a reported CUDA, cleanup or
workload-validation failure.

The `cuda-gpu` CTest preset registers one test for each matrix entry when the
interceptor and GPU tests are enabled. These tests are optional because they
require a compatible Linux CUDA driver and GPU.

## Baseline workload

The baseline is a small CUDA Runtime application for exercising memory
accounting on a real Linux GPU. It deliberately does not use Glimmer headers,
so the same executable can be run with and without `LD_PRELOAD`.

Build it through the GPU preset:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu --target glimmer_cuda_workload -j2
```

Run a 512 MiB workload without interception:

```sh
env -u LD_PRELOAD -u LD_AUDIT \
    ./build/cuda-gpu/examples/cuda_workloads/glimmer_cuda_workload \
    --bytes 536870912 \
    --iterations 10
```

Run a smaller 1 MiB allocation with an 8 MiB Glimmer quota and memory diagnostics:

```sh
env -u LD_AUDIT GLIMMER_MEMORY_LIMIT_BYTES=8388608 \
    GLIMMER_TRACE_MEMORY_INFO=1 \
    LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" \
    ./build/cuda-gpu/examples/cuda_workloads/glimmer_cuda_workload \
    --bytes 1048576 \
    --iterations 10
```

The CSV is written to standard output and contains visible total/free bytes and
numeric CUDA status codes. To save it, choose a new path with `--output PATH`;
the program overwrites an existing file at that path. A request larger than the
active quota should return a non-zero process status and report a failed
allocation; that is an expected rejection, not a workload crash.
