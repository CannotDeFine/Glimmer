# CUDA workload

This workload is a small, reproducible CUDA Runtime application for validating
Glimmer on a real Linux GPU. It allocates a configurable device buffer,
touches the complete buffer in a CUDA kernel, synchronizes through CUDA
events, releases the buffer, and records the visible CUDA memory view in CSV.

It deliberately does not use Glimmer headers. The same executable can therefore
be run with no preload, with the Glimmer interceptor, or with different quota
settings.

Build it through the GPU preset:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu --target glimmer_cuda_workload -j2
```

Run a 512 MiB workload without interception:

```sh
./build/cuda-gpu/examples/cuda_workload/glimmer_cuda_workload \
    --bytes 536870912 \
    --iterations 10 \
    --output /tmp/glimmer-cuda-baseline.csv
```

Run the same workload with an 8 MiB Glimmer quota and memory diagnostics:

```sh
env GLIMMER_MEMORY_LIMIT_BYTES=8388608 \
    GLIMMER_TRACE_MEMORY_INFO=1 \
    LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" \
    ./build/cuda-gpu/examples/cuda_workload/glimmer_cuda_workload \
    --bytes 1048576 \
    --iterations 10 \
    --output /tmp/glimmer-cuda-quota.csv
```

The CSV contains the visible total/free bytes and numeric CUDA status codes.
The interceptor's diagnostic line additionally reports the physical total and
free bytes used for the capacity boundary. A request larger than the active
quota should return a non-zero process status and report a failed allocation;
that is an expected rejection, not a workload crash.
