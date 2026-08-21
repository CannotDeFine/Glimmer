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

`./scripts/check.sh` also runs ShellCheck for the repository scripts when
`shellcheck` is installed.

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

The GPU preset includes an embedded Driver-PTX workload, a Runtime kernel
compiled with `nvcc`, the standalone `glimmer_cuda_workload` baseline, and a
Runtime workload matrix covering async/pool, managed, pitched, and
multi-stream execution. It also builds the opt-in
`glimmer_cuda_task_backend_demo`, which submits explicit Driver-API PTX tasks
through the scheduler-controlled backend. To run only the end-to-end kernel
checks:

```sh
ctest --preset cuda-gpu -R 'glimmer_cuda_interceptor_(kernel_gpu_test|runtime_kernel_gpu_test)' --output-on-failure
```

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

### Transparent launch scheduling

The preload library can also admit covered CUDA kernel launches through a
launch gate. In enforce mode, a launch call waits for an
available slot, forwards the original CUDA call, and records a non-timing
event on the same stream. The slot is released when that event completes. A
launch failure or event-tracking failure releases the lease; if event support
is unavailable, the launch is rejected instead of silently bypassing the
policy.

```sh
env GLIMMER_SCHEDULER_MODE=enforce \
    GLIMMER_MAX_CONCURRENT_KERNELS=1 \
    GLIMMER_SCHEDULER_POLICY=weighted_rr \
    GLIMMER_MEMORY_LIMIT_BYTES=8388608 \
    LD_PRELOAD="$PWD/build/cuda-gpu/lib/libglimmer_cuda_interceptor.so" \
    ./your_cuda_application
```

`GLIMMER_SCHEDULER_POLICY` accepts `weighted_rr` (the default), `drr`, or `fifo`.
`GLIMMER_SCHEDULER_TENANT_ID` and `GLIMMER_SCHEDULER_WEIGHT` identify and
weight the queue; the tenant falls back to
`GLIMMER_QUOTA_TENANT_ID`. This path does not preempt running kernels or capture
CUDA graph launches. To coordinate multiple
processes, set `GLIMMER_SCHEDULER_CONTROL_SOCKET` to a running remote control
service socket. The gate then submits a task-specific lease before each
covered launch, renews long-running leases while their events are pending, and
reports completion after each CUDA event. Transport or lease failures fail the
launch closed; leaving the variable unset preserves the process-local default.

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
    --scheduler-policy weighted_rr \
    --bind-leases-to-process
```

Submit and inspect tasks from another shell:

```sh
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock submit tenant-a 1048576 2 1
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock query 1
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock stats
```

A worker claims the next queued task, executes its local CUDA work, and reports
the terminal result:

```sh
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock claim
./build/debug/bin/glimmer_control_client \
    --socket /tmp/glimmer-control.sock complete 1
```

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
queued task first. Policies control task-boundary submission order and do not
preempt a kernel that is already running.

`--bind-leases-to-process` binds submission, claim, `HEARTBEAT`, `COMPLETE`, and
`FAIL` to an authenticated Linux Unix-socket peer identity. Keep submission,
claim, and execution in the same worker process when this option is enabled.
The option is disabled by default so separate command-line smoke-test
invocations remain compatible.

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
