# PyTorch CUDA smoke workload

This directory contains an optional framework-level workload. The Python
program uses public PyTorch APIs only: a two-layer MLP, GELU activation, CUDA
events, and (for training) autograd plus SGD. It does not import Glimmer
headers or call Glimmer APIs. The interceptor is supplied externally through
`LD_PRELOAD`.

PyTorch is intentionally not a Glimmer build dependency. The example is not
registered as a default CMake target or CTest case because PyTorch is an
environment-specific framework installation. This keeps the core project
buildable without Python or PyTorch.

## Prerequisites

- Linux
- A CUDA-enabled PyTorch installation compatible with the host driver
- A CUDA build of Glimmer for `observe` or `enforce` mode

The environment is kept inside this example directory and is ignored by Git.
Create it with:

```sh
./examples/framework_workloads/pytorch_smoke/setup_pytorch_env.sh
```

This installs the pinned CUDA 13.2 requirements from
`requirements-cu132.txt` into
`examples/framework_workloads/pytorch_smoke/.venv/`. It does not write to your
home directory. The host must provide `python3-venv`; on Debian/Ubuntu install
it with `sudo apt install python3-venv` if necessary.

Check the framework and CUDA runtime before running:

```sh
examples/framework_workloads/pytorch_smoke/.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
PY
```

The workload reports a clear error instead of silently falling back to a
non-CUDA framework path.

## Build

Build the interceptor when testing transparent modes:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu --target glimmer_cuda_interceptor -j2
```

## Run

Run the same workload in native, observe, and enforce modes. The runner writes
CSV measurements under this directory's ignored `output/` directory by
default.

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_smoke.sh \
    --mode native --role inference -- \
    --iterations 10 --warmup 2 --batch-size 32 --hidden-size 1024

./examples/framework_workloads/pytorch_smoke/run_pytorch_smoke.sh \
    --mode observe --role inference --trace -- \
    --iterations 10 --warmup 2 --batch-size 32 --hidden-size 1024

./examples/framework_workloads/pytorch_smoke/run_pytorch_smoke.sh \
    --mode enforce --role training -- \
    --iterations 10 --warmup 2 --batch-size 32 --hidden-size 1024 --work-units 2
```

For a workload closer to a transformer feed-forward block, increase the
matrix dimensions and work units after the smoke test succeeds:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_smoke.sh \
    --mode enforce --role training -- \
    --iterations 20 --warmup 3 --batch-size 32 --hidden-size 2048 --work-units 2
```

The summary includes mean, p50, p95, and p99 CUDA step latency, throughput,
framework/CUDA versions, validation status, and peak allocated memory. The
CSV contains one row per measured iteration. Compare native and observe first
to separate framework overhead from scheduler overhead; then compare enforce
under the same dimensions. This workload does not claim kernel preemption or
framework-wide compatibility with CUDA Graphs, cuDNN, NCCL, or every allocator
path. Pass `--python PATH` to the runner if you intentionally want to use a
different Python environment.

## Synchronized co-location experiment

To demonstrate that priority scheduling is tested with two real framework
processes on one physical GPU, run:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode native --iterations 20 --warmup 3

./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --iterations 20 --warmup 3
```

Each process performs CUDA warmup, writes a readiness record, and waits for a
shared start barrier. The runner then verifies that the processes have
different PIDs, the same CUDA device UUID, valid results, and a positive
overlap between their wall-clock execution intervals. A successful run prints
`colocated=1 same_gpu_uuid=1 overlap_ms=...`. Priority mode also uses one
remote control service and one concurrent scheduler slot, with inference at
priority 100 and training at priority 10. The generated ready/status files,
logs, CSV files, and local virtual environment are ignored by Git.

The runner configures a five-second process-bound lease timeout. Active
workloads renew leases through interceptor heartbeats; if a workload exits
without closing a batched lease, the service reclaims the slot instead of
leaving a permanent scheduler stall.

The strict one-slot setting is useful for measuring scheduler ordering, but it
serializes all admitted kernels. To measure a larger concurrency window, pass
`--max-concurrent-kernels N` to the priority runner. The value is applied to
both the control service and the interceptor; increasing it improves overlap
and throughput at the cost of a weaker latency-isolation boundary. For
example:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --max-concurrent-kernels 2 \
    --iterations 20 --warmup 3
```

The runner can also amortize admission for throughput-oriented training with a
per-process launch batch. This is a scheduler batch, not the PyTorch tensor
batch size: it groups consecutive covered kernel launches in the process under
one lease, including launches from multiple framework host threads.
Inference remains fine-grained by default. Keep its launch batch at `1` when
tail latency matters, and compare larger training values only with repeated
runs:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority \
    --max-concurrent-kernels 2 \
    --training-launch-batch-size 4 \
    --inference-launch-batch-size 1 \
    --iterations 20 --warmup 3
```

Larger launch batches reduce control-plane overhead but delay priority and
fairness decisions until the batch closes. A launch or completion failure
fails the whole batch; batching does not preempt a running CUDA kernel.
The co-location runner rejects a training batch larger than one when the
concurrency window is one, because that combination can hold the only slot
across several training launches and starve the latency-sensitive peer.

To inspect the per-launch cost breakdown during the priority run, add
`--trace-timings`. The child logs then include admission, socket transport,
claim polling, CUDA launch, event tracking, completion, and batched-lease reuse
timing records:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --iterations 20 --warmup 3 --trace-timings
```

For a repeatable comparison, use the matrix runner. It keeps the PyTorch
tensor batch fixed and compares native execution with priority configurations
using one or two concurrent scheduler slots and training launch batches of one
or four. The output CSV contains one row per repetition and separates tensor
batch size from scheduler launch batch size:

```sh
./examples/framework_workloads/pytorch_smoke/run_priority_matrix.sh \
    --repetitions 3 --iterations 20 --warmup 3 \
    --batch-size 32 --hidden-size 1024 \
    --output-dir /tmp/glimmer-priority-matrix
```

The matrix is a measurement aid, not a performance guarantee. Repeat it on an
otherwise idle GPU and compare inference p95/p99 latency with training
throughput. Enabling `--trace-timings` adds diagnostic logging and should be
used for path analysis rather than latency baselines.
