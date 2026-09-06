# Glimmer

Glimmer is a Linux scheduler-based shared GPU project. It schedules work from
multiple tenants or workloads on a single physical GPU to provide fair,
controllable, and observable GPU sharing without hardware-level vGPU
partitioning.

## Build

Glimmer is a self-contained Linux C++20 project built with CMake 3.20 or
newer. Its CMake configuration uses only paths relative to the source tree, so
the repository can be moved or cloned to any location without reconfiguration.

## Prerequisites

- CMake 3.20+
- A C++20-capable compiler: GCC or Clang
- Git, including submodule support
- CUDA Toolkit for CUDA interceptor or workload builds
- An NVIDIA driver for CUDA runtime execution and GPU tests

Clone the repository with its third-party dependencies:

```sh
git clone --recurse-submodules <repository-url>
```

For an existing clone, initialize them with:

```sh
git submodule update --init --recursive
```

### Build

Use the `debug` preset for day-to-day development:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

The build script runs the configure and build steps:

```sh
./scripts/build.sh
```

Additional presets are available for optimized builds, runtime sanitizers, and
static analysis:

```sh
cmake --preset release
cmake --preset asan-ubsan
cmake --preset lint
cmake --preset cuda-lint
```

The `lint` and `cuda-lint` presets require `clang-tidy`. The executable is
written to `bin/` inside the selected build directory.

`./scripts/check.sh` also runs ShellCheck for shell scripts under `scripts/`
and `examples/` when `shellcheck` is installed.

Run the complete pre-commit verification with:

```sh
./scripts/check.sh
```

This runs all available build, test, sanitizer, formatting, and static-analysis
checks. The CUDA lint checks are included when the CUDA Toolkit is installed.

When developing the CUDA interceptor, use its dedicated presets. They use
separate build directories so CUDA-enabled and non-CUDA artifacts cannot share
stale CMake cache state:

```sh
cmake --preset cuda-debug
cmake --build --preset cuda-debug
ctest --preset cuda-debug --output-on-failure
```

For clang-tidy coverage of both the interceptor and its tests, use
`cuda-lint` or run `./scripts/check.sh`.

The CUDA GPU test preset requires a compatible NVIDIA driver and GPU:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu
ctest --preset cuda-gpu --output-on-failure
```

For performance experiments, use the separate optimized build with profiling
symbols; `cuda-gpu` remains a Debug configuration:

```sh
cmake --preset cuda-perf
cmake --build --preset cuda-perf -j2
ctest --preset cuda-perf --output-on-failure
```

Pass `--build-dir build/cuda-perf` to example runners. The PyTorch example
also provides an [uncontended overhead baseline](examples/framework_workloads/pytorch_smoke/README.md#optimized-uncontended-overhead-baseline).
Completing a benchmark is not evidence of a scheduling benefit.
`scripts/check.sh` includes the optimized no-GPU regressions when `nvcc` is
available; the full GPU suite above remains opt-in.

The GPU preset includes an embedded Driver-PTX workload, a Runtime kernel
compiled with `nvcc`, the synchronous `glimmer_cuda_workload` baseline, and a
Runtime workload matrix covering async/pool, managed, pitched, and
multi-stream execution. It also builds the opt-in
`glimmer_cuda_task_backend_demo`, which submits explicit Driver-API PTX tasks
through the scheduler-controlled backend. To run only the end-to-end kernel
checks:

```sh
ctest --preset cuda-gpu -R 'glimmer_cuda_interceptor_(kernel_gpu_test|runtime_kernel_gpu_test)' --output-on-failure
```

For a reproducible mixed-load comparison between native CUDA execution and
transparent priority scheduling, see
[`examples/priority_demo/README.md`](examples/priority_demo/README.md).

For an optional real-framework smoke test using public PyTorch CUDA APIs, see
[`examples/framework_workloads/pytorch_smoke/README.md`](examples/framework_workloads/pytorch_smoke/README.md).
It remains outside the CMake build because PyTorch is supplied by the host
environment. The co-location runner can record an inference latency target,
target misses, and control-service queue/service metrics for repeatable
priority experiments.

Run the real Runtime workload matrix with the interceptor:

```sh
ctest --preset cuda-gpu -R 'glimmer_cuda_(workload_gpu_test|async_workload_gpu_test|managed_workload_gpu_test|pitched_workload_gpu_test|multistream_workload_gpu_test)' --output-on-failure
```

To verify a real cross-process shared quota, run:

```sh
ctest --preset cuda-gpu -R glimmer_cuda_interceptor_runtime_shared_quota_gpu_test --output-on-failure
```

For structured launch observations, enable tracing in observe mode:

```sh
env GLIMMER_MEMORY_LIMIT_BYTES=8388608 GLIMMER_SCHEDULER_MODE=observe GLIMMER_TRACE_KERNEL_LAUNCHES=1 LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" "$PWD/build/cuda-gpu/src/interceptor/glimmer_cuda_interceptor_runtime_kernel_gpu_test"
```

Add `GLIMMER_TRACE_MEMORY_INFO=1` to also print the virtualized total, used,
and free memory together with the physical total and free bytes after each
successful memory-information query.

For enforce-mode launch-path timing, add `GLIMMER_TRACE_LAUNCH_TIMINGS=1`.
The interceptor then prints one allocation-free diagnostic for each admitted
or forwarded launch and one when its completion lease is released. The fields
separate admission time, remote control transport time, bounded wait-request
count and legacy claim-poll count, CUDA launch time, event tracking time,
completion transport time, and whether the launch reused a batched lease. This
is an opt-in diagnostic stream, not a stable metrics export.

### Transparent launch scheduling

The preload library can also admit covered CUDA kernel and graph launches through a
launch gate. In enforce mode, a launch call waits for an
available slot, forwards the original CUDA call, and records a non-timing
event on the same stream. The slot is released when that event completes. A
launch failure or event-tracking failure releases the lease; if event support
is unavailable, the launch is rejected instead of silently bypassing the
policy.

```sh
env GLIMMER_SCHEDULER_MODE=enforce \
    GLIMMER_MAX_CONCURRENT_KERNELS=1 \
    GLIMMER_SCHEDULER_BATCH_SIZE=1 \
    GLIMMER_SCHEDULER_POLICY=weighted_rr \
    GLIMMER_MEMORY_LIMIT_BYTES=8388608 \
    LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" \
    ./your_cuda_application
```

`GLIMMER_SCHEDULER_POLICY` accepts `weighted_rr` (the default), `drr`, `fifo`,
or `priority`. `GLIMMER_SCHEDULER_PRIORITY` sets the unsigned priority of each
transparent launch when the `priority` policy is selected; larger values run
first and the default is `0`.
`GLIMMER_SCHEDULER_TENANT_ID` and `GLIMMER_SCHEDULER_WEIGHT` identify and
weight the queue; the tenant falls back to
`GLIMMER_QUOTA_TENANT_ID`. This path does not preempt running kernels or inspect
graph nodes. Graph launches are admitted and tracked as one task on their
supplied stream. To coordinate multiple
processes, set `GLIMMER_SCHEDULER_CONTROL_SOCKET` to a running remote control
service socket. The gate then submits a task-specific lease before each
covered launch, renews long-running leases while their events are pending, and
reports completion after the final CUDA event for that lease. Set
`GLIMMER_SCHEDULER_BATCH_SIZE` to
reuse one admitted lease for that many launches within the same process. When a
remote lease is queued, the client uses a bounded `WAIT` request instead of
busy-polling `CLAIM`; older services are supported through the legacy polling
fallback. The
default is `1`, which gives the finest scheduling granularity. A larger value
reduces control-plane overhead for throughput-oriented work but delays priority
or fairness decisions by up to the batch size; keep latency-sensitive inference
at `1`. A batch closes on its final successfully tracked launch, and any launch
or event failure fails the whole batch. Transport or lease failures fail the
launch closed; leaving the variable unset preserves the process-local default.
`GLIMMER_SCHEDULER_PRIORITY_RESERVED_SLOTS` reserves launch slots for
priorities at or above `GLIMMER_SCHEDULER_PRIORITY_THRESHOLD` (default `1`).
The reservation applies only to the `priority` policy and is disabled when the
slot count is `0`; lower-priority launches cannot borrow reserved capacity.

### Task-scoped memory isolation

In process-local mode, `GLIMMER_MEMORY_LIMIT_BYTES` is the memory ceiling for
the attached CUDA process. In shared tenant mode, it remains the aggregate
tenant ceiling. Set `GLIMMER_TASK_MEMORY_LIMIT_BYTES` in shared mode to add a
narrower per-process task ceiling:

```sh
env GLIMMER_QUOTA_MODE=shared \
    GLIMMER_QUOTA_TENANT_ID=demo-tenant \
    GLIMMER_MEMORY_LIMIT_BYTES=8589934592 \
    GLIMMER_TASK_MEMORY_LIMIT_BYTES=2147483648 \
    LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" \
    ./your_cuda_application
```

Every covered Driver and Runtime allocation must pass both the tenant and
task limits. The task limit applies to the CUDA process attached to the
preload library; multiple logical tasks inside one process are not yet
independently identifiable. Kernel preemption and time slicing are outside
this memory-isolation feature. Glimmer also queries the CUDA device capacity
automatically: the effective visible and enforceable limit is the minimum of
the configured quota and the device's physical total memory. Allocation
admission additionally checks the current physical free memory whenever the
current CUDA context identifies the requested device, so a quota larger than
the GPU (for example, 8 GiB configured on a 6 GiB device) is clamped to the
real device capacity.

The root `compile_commands.json` link follows the most recently built preset.
Use `cuda-lint` last when editor diagnostics must include CUDA interceptor files
and Toolkit include paths.

### Local control service

The debug build provides a Linux-only control service. The default `simulated`
mode advances a deterministic backend; `remote` mode exposes task leases for a
separate worker process. In both modes the service owns scheduling and quota
state, while CUDA handles and pointers remain inside the worker process.

```sh
cmake --build --preset debug
./build/debug/bin/glimmer_control_service \
    --socket /tmp/glimmer-control.sock \
    --quota-bytes 8388608
```

To run the explicit worker-lease mode, add `--execution-mode remote`:

```sh
./build/debug/bin/glimmer_control_service \
    --socket /tmp/glimmer-control.sock \
    --quota-bytes 8388608 \
    --execution-mode remote \
    --lease-timeout-ms 5000 \
    --max-concurrent-tasks 2 \
    --max-queued-tasks 64 \
    --scheduler-policy weighted_rr
```

Add `--trace-scheduler` when you need to audit the service-side dispatch
order. It writes one structured line per successful dispatch to stderr (or the
service log when stderr is redirected), including sequence, task, tenant, and
priority. The option is disabled by default and does not change scheduling.

Submit and inspect tasks from another shell:

```sh
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock submit tenant-a 1048576 2 1
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock query 1
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock stats
```

The optional priority argument sets priority when the service uses
`--scheduler-policy priority`, for example:

```sh
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock submit tenant-a 1048576 2 1 10
```

The framework co-location example exposes the same diagnostic for its remote
priority service:

```sh
./examples/framework_workloads/pytorch_smoke/run_pytorch_priority.sh \
    --mode priority --trace-scheduler --iterations 20 --warmup 3
```

Inspect `control-service.log` in the reported output directory to verify that
higher-priority queued tasks were dispatched first. The process CSV files and
the reported `overlap_ms` remain the evidence for workload latency and
co-location; scheduler trace timestamps are diagnostic rather than a
benchmark metric.

A worker claims the next queued task, executes its local CUDA work, and reports
the terminal result:

```sh
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock claim
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock complete 1
```

This command-line smoke workflow intentionally uses the default unbound mode:
submission, claim, and completion are separate client processes. With
`--bind-leases-to-process`, the submitting process must also claim the same
task by id and keep the lease lifecycle in that process; an unrelated worker
cannot claim it with unscoped `CLAIM`.

When `--lease-timeout-ms` is enabled, the CUDA lease worker renews its lease
automatically. The standalone client can renew a lease explicitly with
`heartbeat TASK_ID`. An unrenewed lease is marked `FAILED` and its quota is
released after the configured timeout.
Set `--lease-timeout-ms 0` to disable lease expiry; in that mode a running
lease remains until an explicit terminal report or service restart.

`--max-concurrent-tasks` controls the number of remote workers that may hold a
running lease simultaneously. It defaults to `1`; quota reservations still
bound the total memory admitted by the service.

`--scheduler-policy` selects the task dispatch order. `weighted_rr` is the
default and preserves weighted tenant fairness; `drr` uses `work_units` as a
task cost and `weight` as its tenant quantum; `fifo` dispatches the oldest
queued task first; and `priority` dispatches the largest priority first while
preserving submission order for ties. Priority is strict and can starve lower
priority work under sustained load. Policies control task-boundary submission
order and do not preempt a kernel that is already running.

For latency-sensitive co-location, reserve capacity explicitly:

```sh
./build/debug/bin/glimmer_control_service \
    --socket /tmp/glimmer-control-priority.sock \
    --quota-bytes 8388608 \
    --execution-mode remote \
    --max-concurrent-tasks 2 \
    --scheduler-policy priority \
    --priority-reserved-slots 1 \
    --priority-threshold 100
```

`--priority-reserved-slots N` keeps N running slots available for tasks whose
priority is at least `--priority-threshold`. Lower-priority tasks use only the
remaining slots, even when the reserved capacity is idle. This setting is
disabled by default and is a best-effort latency-isolation mechanism; it does
not preempt a kernel that is already running.

For workloads whose latency changes over time, the service can adapt that
reservation from measured high-priority queue wait:

```sh
./build/debug/bin/glimmer_control_service \
    --socket /tmp/glimmer-control-adaptive.sock \
    --quota-bytes 8388608 \
    --execution-mode remote \
    --max-concurrent-tasks 2 \
    --scheduler-policy priority \
    --priority-threshold 100 \
    --adaptive-slo-target-queue-us 500 \
    --adaptive-slo-window 32 \
    --adaptive-slo-max-reserved-slots 1 \
    --trace-scheduler
```

`--adaptive-slo-target-queue-us N` enables feedback. After each observation
window, at least 25 percent of high-priority tasks exceeding the target adds
one reserved slot; a lower violation ratio removes one slot. The configured initial
reservation is retained until the first complete window, and capacity is
clamped by `--adaptive-slo-max-reserved-slots`. Set the target to `0` to leave
the static scheduler unchanged. Adaptive feedback is a bounded task-boundary
controller, not a hard real-time guarantee: running CUDA kernels are never
preempted, and remote workers must still report `COMPLETE` or `FAIL`. Without
new high-priority completions, the reservation does not decay on its own.

`--bind-leases-to-process` binds submission, task-specific claim,
`HEARTBEAT`, `COMPLETE`, and `FAIL` to an authenticated Linux Unix-socket peer
identity. Unscoped `CLAIM` is rejected because it cannot prove ownership of a
pending task. Keep submission, claim, and execution in the same worker process
when this option is enabled. The option is disabled by default so separate
command-line smoke-test invocations remain compatible.

`--max-queued-tasks` optionally bounds waiting tasks. A value of `0` (the
default) leaves the queue unlimited; when the bound is reached, submission
returns `ERROR QUEUE_FULL` without consuming quota.

`stats` returns a read-only snapshot of task-state counts, quota bytes, and the
configured running/queued limits. `metrics` additionally reports aggregate and
maximum queue-wait and service-time microseconds. Both are intended for
diagnostics and smoke checks, not as a durable time-series metrics export.

Use `fail TASK_ID` when execution cannot complete. `CANCEL` is intentionally
limited to queued tasks; the service does not pretend to preempt a running CUDA
kernel.

The service uses `SO_PEERCRED` and accepts the server process UID by default.
This first service is a control-plane validation harness: it does not accept
CUDA pointers or kernel descriptors and does not execute arbitrary application
workloads. CUDA resource binding remains process-local.

The automated multi-process check starts the service and client as separate
processes and verifies quota rejection, an empty claim, lease metadata,
concurrent running-state observation, heartbeat renewal, and explicit terminal
transitions:

```sh
ctest --preset debug -R glimmer_control_service_process_test --output-on-failure
```

## Contributing

Follow the project's [development lifecycle and review rules](docs/DEVELOPMENT_LIFECYCLE.md)
and [Git guidelines](docs/GIT_GUIDELINES.md) when contributing.
