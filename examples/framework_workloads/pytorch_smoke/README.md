# PyTorch CUDA workloads

[Project home](../../../README.md) / [Documentation](../../../docs/README.md) / [Examples](../../README.md) / PyTorch

Run ordinary PyTorch inference and training with the launch-admission preload.
The default model is a two-layer MLP with GELU; it uses public PyTorch APIs and
does not import Glimmer. The same program can run natively, observe covered
launches, or enforce launch admission.

These commands do not use centralized execution or enable cooperative kernel
replacements. For your own model, use the [application guide](../../../docs/USAGE.md).

## Prerequisites and build

Use Linux, Python 3.10 or newer, a compatible NVIDIA GPU/driver and a CUDA-enabled
PyTorch installation. See the [reference environment](../../../docs/CUDA_API_COVERAGE.md#reference-environment).
Run all commands from the repository root, without inherited preload or
`GLIMMER_*` settings.

For the CUDA 12.6 profile, including the V100 reference configuration:

```sh
./examples/framework_workloads/pytorch_smoke/setup_pytorch_env.sh --cuda cu126
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu -j2 --target \
  glimmer_cuda_interceptor glimmer_control_service glimmer_control_client
```

The setup script creates `.venv` beside the workload and installs the pinned
[cu126 requirements](requirements-cu126.txt), including PyTorch and torchvision.
It requires package-download access. The alternative `--cuda cu132` profile
uses [requirements-cu132.txt](requirements-cu132.txt) and is the setup script's
default; choose it only with a compatible GPU and driver. A package profile is
not a claim that every framework operation is supported. Use `--python PATH`
on the runners to select an existing environment instead of installing one.

## Run a single application

Run the same MLP natively and with process-local launch admission:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_smoke.sh \
  --mode native --role inference -- \
  --iterations 10 --warmup 2 --batch-size 32 --hidden-size 1024

./examples/framework_workloads/pytorch_smoke/run_pytorch_smoke.sh \
  --mode enforce --role inference -- \
  --iterations 10 --warmup 2 --batch-size 32 --hidden-size 1024
```

Use `--role training` for forward/backward/update steps. `--mode observe`
loads the interceptor without enforcing launch scheduling. The single-process
`enforce` runner uses a local weighted-round-robin gate with two slots; it does
not start the shared service.

Each run writes its CSV to a new directory under
`artifacts/runs/pytorch-smoke/`. Use `--output PATH` before `--` to select an
exact output file; existing files and symbolic links are not overwritten.
Summaries include wall-clock latency, throughput, validation status, peak
allocated memory and framework/device identity. A missing CUDA device is an
error, not a silent CPU fallback.

## Share the GPU between training and inference

The co-location runner owns the shared control service and two framework
processes, with training priority `10` and inference priority `100`:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
  --mode priority --max-concurrent-kernels 2 \
  --iterations 20 --warmup 3
```

The runner warms both applications, coordinates a common start, validates
outputs and service drain, and stops its owned processes on exit. It checks
that the processes have distinct PIDs and the same CUDA device UUID. A successful
run includes `colocated=1 same_gpu_uuid=1 overlap_ms=...`.

By default, each run uses a new directory under
`artifacts/runs/pytorch-colocation-<mode>/`. An explicit `--output-dir` is the
exact result directory; it must not contain output from an earlier run. Failed
runs retain their results and cannot reuse the same directory.

Repeat with `--mode native` for execution without Glimmer. Positive process-time
overlap does not prove simultaneous kernel execution. Priority applies to queued
launches and does not interrupt running kernels.

The runner defaults to one concurrent slot if the option is omitted, which
serializes admitted launches. With two slots, `--priority-reserved-slots 1
--priority-threshold 100` reserves one slot from training; it may remain idle
when inference is absent. This trades training capacity for inference admission
opportunities, not a latency guarantee.

## Validate numerical behavior

The correctness workload checks matrix/convolution forward and gradient results
against CPU references, stream/event ordering, allocator lifetimes and CUDA
Graph replay:

```sh
examples/framework_workloads/pytorch_smoke/.venv/bin/python \
  examples/framework_workloads/pytorch_smoke/pytorch_correctness.py \
  --output artifacts/runs/correctness-native.json
```

Choose a new output path for each run; an existing result is never overwritten.

Run unchanged inputs with the intended preload configuration separately before
relying on an application comparison. Native correctness does not establish
interception, cooperative selection or GPU scheduling behavior.

The workload also accepts `--workload cnn` after the smoke runner's `--`
separator. This selects a small FP32 convolutional model with 32x32 RGB inputs,
GELU, pooling and a linear output, using cuDNN. Keep MLP dimension options at
their defaults for this mode. Add `--cnn-workload` to `pytorch_correctness.py`
to check this exact model's gradients, updates and tensor-allocation cleanup;
MLP success does not qualify the CNN path.

## Optional measurements

Use optimized `cuda-perf` builds for timings, not the Debug quick-start build.
Keep dimensions, inputs, binaries and runtime identities matched. Do not run
other GPU jobs, builds or tests concurrently with a measurement.

- `run_overhead_baseline.py` compares native, preload-only, quota, local and
  remote solo execution. Its results do not measure interference from another
  application.
- `run_slo_validation.py` compares native solo/co-location and priority
  configurations over common warmed windows. For example:

```sh
python3 -B examples/framework_workloads/pytorch_smoke/run_slo_validation.py \
  --build-dir build/cuda-perf \
  --duration-seconds 10 --repetitions 3 --inference-target-ms 2
```

The 2 ms target is an example measurement criterion, not a promised service
level. Inspect sample counts and every repetition before interpreting p99.
Completed collection is not a performance or full-device qualification pass.
Keep cooperative-preview options off unless qualifying that separate path.

`run_launch_profile.py` captures optional warmed CUDA timelines with an already
installed Nsight Systems CLI. Use its `--help` for selection options. Trace
durations are not ordinary benchmark timings, and execution-interval coverage
is not SM utilization. Incomplete traces cannot establish whole-workload costs.

Each measurement runner's `--help` describes output selection. Retain the raw
evidence and failed-run diagnostics needed to support the results, together with
their identities. Review and remove superseded large temporary outputs under the
[storage rules](../../../docs/STORAGE_LAYOUT.md#retain-results-not-every-large-output).
The SLO and overhead runners create fresh directories under `artifacts/runs/`
by default; timeline captures use `artifacts/profiles/`. An explicit
`--output-dir` is an exact, empty directory inside `artifacts/` or outside the
repository; a claimed directory cannot be reused.
See [Testing](../../../docs/TESTING.md) for comparison requirements.
