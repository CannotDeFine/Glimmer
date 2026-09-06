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

The `wall_clock_v2` summary includes mean, p50, p95, and p99 host step latency
(before submission through CUDA synchronization), wall-clock throughput,
framework/CUDA versions, validation status, and peak allocated memory. The
CSV contains one row per measured iteration. Compare native and observe first
to separate framework overhead from scheduler overhead; then compare enforce
under the same dimensions. This workload does not claim kernel preemption or
framework-wide compatibility with CUDA Graphs, cuDNN, NCCL, or every allocator
path. Pass `--python PATH` to the runner if you intentionally want to use a
different Python environment.

To record an inference latency target, pass it after the `--` separator. The
target is a measurement SLO; it does not alter CUDA scheduling:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_smoke.sh \
    --mode enforce --role inference -- \
    --iterations 50 --warmup 5 --latency-target-ms 1.0
```

The per-iteration CSV and summary include the target, miss count, and miss
ratio. A target of zero (the default) disables miss accounting.
Each CSV also records `iteration_start_ns`, `iteration_end_ns`, and the separate
`cuda_event_ms` diagnostic. `elapsed_ms` uses the host timestamps; earlier
event-based CSVs are not directly comparable. Throughput is completed steps
divided by wall-clock interval, including inter-step overhead. A disabled
target leaves `latency_target_met` empty instead of labeling the sample a pass.

## Optimized uncontended overhead baseline

Use this experiment before changing scheduling policy. It measures one process
at a time, not training/inference co-location. The existing `cuda-gpu` preset
is Debug; keep its results separate from optimized-build results.

```sh
cmake --preset cuda-perf
cmake --build --preset cuda-perf -j2
ctest --preset cuda-perf --output-on-failure
python3 examples/framework_workloads/pytorch_smoke/run_overhead_baseline.py \
    --duration-seconds 10 --repetitions 3
```

`cuda-perf` uses RelWithDebInfo with sanitizers and clang-tidy disabled. Its
build refreshes the root `compile_commands.json`, including CUDA examples.
The runner checks the build type and actual runtime compile commands, then
uses the example-local `.venv` unless `--python` specifies another interpreter.
It does not install packages or fall back to CPU.

| Mode | Preload | Memory quota | Launch admission |
| --- | --- | --- | --- |
| `native` | No | No | Native CUDA |
| `preload` | Yes | No | Off |
| `quota` | Yes | Yes | Off |
| `local` | Yes | Yes | Process-local priority gate |
| `remote` | Yes | Yes | Remote priority service, without a competing worker |

Each mode runs inference and training separately. Defaults are 512 MiB quota,
tensor batch 32, hidden size 1024, training work units two, inference work units
one, two launch slots, and launch batch one. `--quota-bytes` changes the same
quota for all quota-enabled modes; it is an experimental setting, not a GPU
capacity estimate. Increase it if a larger model does not fit. The optional
workload `--expected-memory-total-bytes` check verifies the visible CUDA
capacity before model setup/warmup, outside the timed interval. A mismatch or
OOM is a failed experiment, not a silently weakened quota.

Three repetitions produce 30 sequential runs with rotating order. Do not build,
run tests, or start another GPU experiment concurrently. Detailed traces and
inherited Glimmer/loader instrumentation are removed. Nonzero
`CUDA_LAUNCH_BLOCKING` is rejected. A private control service is started only
for each remote run; its final stats must show completed, drained admissions.
Those counters include initialization, warmup, and drain, not only timed steps.

Results default to a new ignored `output/overhead-baseline-*` directory.
`--output-dir` accepts only an empty directory. Files include:

- `manifest.json`: build options, selected environment, source/binary hashes,
  arguments, and collection-completion state.
- Per-run commands, logs, raw role CSV/status files, and remote service stats.
- `summary.csv`: validated per-run wall-clock metrics and sample counts.
- `aggregate.json`: medians/ranges of run-level metrics, with role-local native
  latency and throughput ratios. Ratios are not additive component costs.
- `failure.json`: explicit failure context; failed or incomplete collections
  cannot be analyzed as successful experiments.

Rebuild aggregates from saved raw data with:

```sh
python3 examples/framework_workloads/pytorch_smoke/run_overhead_baseline.py \
    --analyze-only /path/to/overhead-baseline-results
```

The runner does not assert a performance gain. Short traced diagnostics and
GPU timelines must remain separate from the untraced baseline. For sharing
interference and net scheduling benefit, run the following common-window
experiment separately, adding `--build-dir build/cuda-perf`. These closed-loop
MLP experiments do not establish serving deadlines or general workload support.

## Warmed CUDA timeline diagnostics

Use the separately installed NVIDIA Nsight Systems CLI (`nsys`) to investigate
the optimized solo overhead. This is optional development tooling, not a
Glimmer/PyTorch build dependency. The helper does not install it or require
CPU sampling privileges. Actual CUDA tracing still requires host GPU access.

```sh
python3 examples/framework_workloads/pytorch_smoke/run_launch_profile.py \
    --mode local --role training --duration-seconds 1
```

Repeat with `--mode native`, `quota`, and `remote`, and with `--role inference`.
`preload` is also accepted. Model, quota, slot, and launch-batch defaults match
the uncontended baseline above. Each command captures only one solo process;
remote mode owns a private service and checks that its admissions drain.
Use the same tensor parameters for all modes. Do not run captures concurrently
with each other, tests, builds, or benchmarks.

The workload's opt-in `--profile-cuda` starts CUDA profiling after warmup and
stops before result validation/output. Nsight uses
`--capture-range=cudaProfilerApi --capture-range-end=stop`; the application
continues to validate and clean up after capture. Profiling is rejected for
CPU or start-barrier runs. Without this flag, the workload makes no profiler
calls. Its status labels diagnostic data so normal benchmark analysis rejects
profiled results.

Results go into a fresh ignored `output/launch-profile-<mode>-<role>-*`
directory; a nonempty `--output-dir` is rejected. Files include the exact
command, `profile.log`, framework CSV/status, source/build hashes and Nsight
version in `manifest.json`, plus:

- `timeline.nsys-rep`: open with the Nsight Systems UI to inspect the timeline.
- `timeline.sqlite`: raw CUDA activities and API calls.
- `analysis.json`: kernel-duration distribution, merged execution intervals,
  internal gaps, API counts/durations, raw return fields, and profiler warnings.
- `failure.json`: missing kernels, failed workload/quota checks, incompatible
  trace schema, or other capture failures. These are not successful captures.

The analyzer currently expects one CUDA process and one device. Kernel
intervals are unioned across streams; execution coverage additionally includes
recorded memcpy/memset intervals. The denominator runs from the first recorded
execution activity to the last, not the whole measurement window. Coverage is
**not SM utilization**, and a gap does not prove device-wide idleness. Other
processes and unsupported activity types are outside that view. CUDA API sums
can overlap across threads and nested Runtime/Driver calls; do not sum them as
an additive latency breakdown. Raw return fields are not independent evidence
that an event completed.

All-API tracing is enabled to expose short event queries. It adds overhead;
do not use the profiled p99/throughput as benchmark results. Capture completion
is not a guarantee of complete collection: the runner prints warnings and
labels `trace_quality=warnings_require_review` when any are reported. In
particular, missing-event warnings prevent reliable gap/coverage comparisons;
retain such captures as diagnostic evidence, not complete timelines.
Profiler injection can also affect symbol routing. The analyzer lists `LaunchKernelEx` API
observations separately. Driver/Runtime extended launches and their PTDS aliases
now have explicit wrappers, but seeing kernels is not proof that all launches were admitted. Audit
the corresponding unprofiled lookup/launch path before claiming coverage.
See [ADR 0037](../../../docs/decisions/0037-warmed-cuda-timeline-diagnostics.md).
The coverage and completion fixes are recorded in
[ADR 0038](../../../docs/decisions/0038-extended-launch-and-completion-correctness.md).

## Common-window SLO validation

Use this command for native training alone, native inference alone, and four
co-location configurations: native, unreserved priority, static one-slot
reservation, and adaptive reservation. It fixes each role's workload, tensor
and scheduler batch sizes, and uses two slots in every priority configuration:

```sh
python3 examples/framework_workloads/pytorch_smoke/run_slo_validation.py \
    --duration-seconds 10 --repetitions 3 --inference-target-ms 2
```

The 2 ms target is an example experiment criterion, not a promised service
level. Choose it before comparing results. The adaptive queue-wait target
defaults to 500 microseconds over eight lease completions and is a separate
setting. Detailed tracing is disabled. Configuration order rotates between
repetitions, for six configurations per repetition. Native solo runs also use
duration windows and exclude final boundary-crossing steps. Every process from
one run exits before the next run starts. Each invocation creates a fresh directory under this example's
ignored `output/slo-validation-*`; `--output-dir` accepts an empty directory.

- `manifest.json`: experiment settings, relevant environment values, and hashes
  of the runtime artifacts and measurement sources.
- `summary.csv`: one row per co-location run, including sample counts and window coverage.
- `solo_summary.csv` and `solo_aggregate.json`: native solo measurements and
  per-role medians/ranges.
- `aggregate.json`: medians/ranges of per-run percentiles, miss ratios, and
  training throughput. `*_vs_native` divides by the native co-location baseline;
  `*_vs_solo` divides by the corresponding role's native solo baseline. Lower
  latency ratios are better; higher throughput ratios are better.
- Each run directory: raw per-step CSVs, worker identity/status, logs, and
  the exact command. Paired runs also include `colocation.json`.

Recompute aggregates without running GPU work with
`run_slo_validation.py --analyze-only PATH_TO_RESULT_DIRECTORY`. An incomplete
comparison fails analysis instead of silently omitting a configuration. New
`solo_colocation_v1` comparisons require every repetition and both baselines,
with matching device, framework, workload settings, and measurement version.
A recorded run failure prevents successful aggregation. Older comparisons can
still report co-location ratios but explicitly report unavailable solo
baselines. Sources or binaries changing during a run also fail validation.

Native solo versus native co-location quantifies interference; native versus
Glimmer co-location quantifies the scheduler's net effect. A completed
experiment does not assert a scheduling benefit or an inference SLO pass. Do
not require slowdown to validate co-location: complementary workloads may
show little degradation. These measurements also do not isolate host and GPU
causes or prove simultaneous physical kernel execution.

Both workers warm up, wait for an atomic shared start record, and keep running
until the same deadline. The analyzer requires at least 95 percent shared
window coverage for both workers. Fully contained steps determine percentiles,
target misses, and completed steps per window second; boundary-crossing steps
remain in the raw CSV and are counted as exclusions. Fixed-iteration runs
remain available for smoke testing but can have very low training coverage.
The co-location runner's summary uses common-window metrics; each individual
worker's summary covers its entire measured interval including the final drain.

This measures a closed-loop MLP workload, not an open-loop inference service.
Inspect sample counts before interpreting p99, compare every repetition, and
do not infer physical kernel concurrency solely from overlapping process
intervals. Scheduler counters include warmup and cleanup, unlike the common
measurement window. Pure measurement tests need only standard-library Python:

```sh
python3 -B tests/examples/pytorch_measurements_test.py
```

## Synchronized co-location experiment

To demonstrate that priority scheduling is tested with two real framework
processes on one physical GPU, run:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode native --iterations 20 --warmup 3

./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --iterations 20 --warmup 3
```

An explicit miss-ratio bound can be used as a repeatable experiment gate. It
fails the runner when the observed inference miss ratio is above the selected
bound:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --iterations 50 --warmup 5 \
    --inference-target-ms 2.0 \
    --max-inference-target-miss-ratio 0.10
```

Each process performs CUDA warmup, writes a readiness record, and waits for a
shared start barrier. The runner then verifies that the processes have
different PIDs, the same CUDA device UUID, valid results, and a positive
overlap between their wall-clock execution intervals. A successful run prints
`colocated=1 same_gpu_uuid=1 overlap_ms=...`. Priority mode also uses one
remote control service, with inference at priority 100 and training at
priority 10. The scheduler has one concurrent slot by default and can be
expanded with `--max-concurrent-kernels`; the generated ready/status files,
logs, CSV files, and local virtual environment are ignored by Git.

Before reporting success, priority mode also requires service metrics to show
nonzero admitted work, every task completed, no queued/running/cancelled/failed
tasks, and no outstanding quota. Valid model output alone cannot hide a failed
lease or an incomplete scheduler drain. The final response is retained in
`control-metrics.txt` and malformed responses fail validation.

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

To keep one of two slots available for inference, enable priority capacity
reservation. The runner passes these settings to both the remote service and
the interceptor:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --max-concurrent-kernels 2 \
    --priority-reserved-slots 1 --priority-threshold 100 \
    --inference-target-ms 2.0 \
    --iterations 50 --warmup 5
```

Training (priority 10) can use only the unreserved slot; inference (priority
100) can use either slot. Reserved capacity remains idle when inference is not
running, so this setting trades peak training throughput for more predictable
inference admission. It does not preempt a kernel that is already running.
Because this runner assigns training priority 10, keep at least one slot
unreserved (or set the threshold at or below 10) so the training process can
make progress.

The co-location runner can enable the control service's adaptive reservation
with a queue-wait target. It starts with the static reservation and then adds
or removes one reserved slot at each completed feedback window. The target is
measured for inference-priority leases; it does not change CUDA kernel
preemption semantics:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --max-concurrent-kernels 2 \
    --adaptive-slo-target-queue-us 500 \
    --adaptive-slo-window 32 \
    --adaptive-slo-max-reserved-slots 1 \
    --priority-threshold 100 \
    --iterations 50 --warmup 5 --trace-scheduler
```

Adaptive feedback is available only in priority mode. A zero target (the
default) keeps the static reservation path, and the optional cap defaults to
one fewer than the configured concurrency window. Inspect
`control-service.log` for applied reservation changes and the final summary
for queue/service counters. Use repeated runs on an otherwise idle GPU when
comparing convergence; a single run is not a real-time guarantee.

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

The matrix also records the control service's aggregate queue-wait and
service-time counters. Aggregate repeated runs with the standard-library-only
helper:

```sh
python3 examples/framework_workloads/pytorch_smoke/analyze_priority_matrix.py \
    /tmp/glimmer-priority-matrix/summary.csv
```

The aggregate output keeps scheduler launch-batch size separate from the
framework tensor batch size. It reports descriptive means of p50/p95/p99
latency, training and inference throughput, target misses, queue wait, service
time, and co-location overlap; it does not turn those measurements into a
general performance guarantee.
