# Transparent priority scheduling demo

[Project home](../../README.md) / [Documentation](../../docs/README.md) / [Examples](../../examples/README.md) / Priority demo

This synthetic example demonstrates a mixed-load scenario: a
throughput-oriented training-shaped process and a latency-sensitive inference-shaped process
run as ordinary CUDA applications while Glimmer transparently orders their
kernel launches through a shared control-plane launch gate.

The demo uses ordinary CUDA Runtime and cuBLAS APIs. It does not include
Glimmer headers or link against Glimmer libraries. By default it runs a small
model-shaped execution graph: inference performs projection, activation, and
output projection; training performs forward projection, activation, two
backward projections, and a parameter-update kernel. The training process uses
a `2048 x 2048` matrix and two model steps per launch, while inference uses a
`1024 x 1024` matrix and one step. These profiles are still synthetic, but
their operation mix is closer to a compute-bound model than a single GEMM or
pointwise micro-kernel. Both processes remain ordinary CUDA applications; the
role profiles and trusted launcher configuration are explicit:

- training uses priority `10`;
- inference uses priority `100`;
- the control service admits up to two launches concurrently with the
  `priority` policy, preserving normal CUDA overlap while ordering queued work.

Priority applies when launches are queued. It does not preempt a kernel that is
already running. The generated CSV files contain launch-request, launch-return,
completion, admission-wait, and device-kernel timings. The SVG visualizes the
two process traces on a shared monotonic timeline.

## Build

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu -j2 --target \
    glimmer_cuda_interceptor glimmer_control_service glimmer_cuda_priority_workload
```

## Run

```sh
./examples/priority_demo/run_priority_demo.sh \
    --mode priority \
    --build-dir build/cuda-gpu
```

Run the same workload without interception or the control service to produce
a native CUDA baseline:

```sh
./examples/priority_demo/run_priority_demo.sh \
    --mode native \
    --build-dir build/cuda-gpu
```

Each run creates a fresh `attempt-*` directory under
`artifacts/runs/priority-demo-<mode>/`. To choose a different parent directory,
pass `--output-dir DIR`; the runner still creates a new attempt inside it and
does not overwrite existing results. Project-local output must be under
`artifacts/`; a dedicated directory outside the project is also supported.

The native run allows the CUDA driver to schedule both processes normally. Its
timings are useful as a baseline, but they are not an apples-to-apples
throughput comparison: priority mode adds a task-boundary admission path and
allows two concurrent launch leases. Inspect both role traces, but do not use
these short synthetic runs as evidence of a general performance benefit or
full-device saturation. Memory quota behavior is covered by the dedicated CUDA
workload examples and is intentionally not included in this scheduling
comparison.

The runner removes inherited `LD_PRELOAD` and `LD_AUDIT` from its service and
workload processes; priority mode then adds its own interceptor. Do not use it
unchanged for an application that requires another loader instrumentation.

The command prints the output directory. It contains:

```text
training.csv
inference.csv
run.json
training.ready
inference.ready
start.signal
training.log
inference.log
control-service.log
priority_trace.svg
```

`control-service.log` is present only in priority mode.

Open the SVG or inspect the CSV files to compare the two launch streams. Use
`--no-plot` when only the CSV traces are needed. Use `--trace` to enable
per-launch diagnostic logging; it is disabled by default so the comparison
does not include avoidable logging overhead. Run the workload binary directly
with `--kernel pointwise` for a low-cost control case, or change
`--matrix-size`, `--work-units`, and `--iterations` to match the target GPU.
Use `--kernel gemm` for the tiled-GEMM control profile.
