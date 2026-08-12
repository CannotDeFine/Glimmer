# CUDA workload matrix

These programs are ordinary CUDA Runtime applications. They do not include
Glimmer headers or call Glimmer APIs; the same executable can be run with no
preload, with the interceptor, or with different quota settings.

The matrix complements the baseline workload in
[`../cuda_workload/`](../cuda_workload/):

| Workload | CUDA behavior exercised |
| --- | --- |
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
env GLIMMER_MEMORY_LIMIT_BYTES=8388608 \
    GLIMMER_TRACE_MEMORY_INFO=1 \
    LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" \
    "$PWD/build/cuda-gpu/examples/cuda_workloads/glimmer_cuda_async_workload"
```

Each program prints one machine-readable summary line containing its workload
name, requested bytes, visible memory snapshots, and a validation flag. A
non-zero exit status means that CUDA execution, cleanup, or accounting
validation failed.

The `cuda-gpu` CTest preset registers one test for each matrix entry when the
interceptor and GPU tests are enabled. These tests are optional because they
require a compatible Linux CUDA driver and GPU.
