# Testing Guidelines

Tests are a required part of a behavior change. A change is not complete until
its expected behavior and important failure paths are verified.

## Required coverage

- Add or update a test for every new behavior and every bug fix.
- Test public behavior and observable outcomes rather than private
  implementation details.
- Cover success, rejection, timeout, cancellation, and cleanup paths when they
  apply.
- Add a regression test that fails before a bug fix and passes after it.
- Keep tests deterministic, isolated, and independent of execution order.

## Test levels

- **Unit tests** verify scheduling policy, quota accounting, task state
  transitions, and error handling without CUDA hardware.
- **Integration tests** verify interactions between the scheduler, API layer,
  and backend using a simulated backend by default.
- **GPU tests** verify CUDA- or NVML-specific integration only. They must be
  optional so that the standard test suite can run on a Linux machine without a
  GPU.

The scheduler core must remain testable with a simulated backend and a
controllable clock. Real hardware tests complement this coverage; they do not
replace it.

## Current module coverage

| Module | Current tests | Hardware required |
| --- | --- | --- |
| `core` | Quota reservation admission, rejection, commit, cancellation, release failures, concurrent reservations, lifetime safety, counter overflow protection, task admission, FIFO, weighted, deficit-round-robin, and strict-priority ordering, policy-preserving task-specific dispatch, configurable concurrent dispatch slots, priority-reserved capacity, adaptive SLO reservation feedback, queued-work backpressure, scheduler state and latency snapshots, high-priority inference dispatch ahead of a training backlog, completion, cancellation, failure, and input validation. | No |
| `backend` | Simulated task submission, deterministic progress, completion, cancellation, duplicate rejection, unknown-task handling, executor-to-scheduler submission/progress/terminal transitions, and fake-Driver CUDA launch/event state mapping. | No |
| `examples` | Optional CUDA workload harnesses cover synchronous, stream-ordered/pool, managed, pitched, and multi-stream Runtime paths with device-side validation and visible-memory reporting under `LD_PRELOAD`; the priority demo compares native scheduling with transparent priority enforcement using configurable cuBLAS model-shaped, tiled-GEMM, or pointwise workloads; the optional framework workload directory provides a public-API PyTorch CUDA MLP smoke test for native/observe/enforce comparisons, configurable inference latency targets and miss ratios, synchronized co-location, and repeatable queue/service metric aggregation; explicit task demos cover Driver-API PTX launch, scheduler admission, event polling, terminal quota release, and remote Unix-socket lease workers with process-local CUDA execution. The remote lease GPU integration test starts the endpoint with process-bound leases, forks two workers, and verifies both completed states. | CUDA and optional framework |
| `control` | Quota-visible memory information, physical-memory bounds, physical-capacity admission and free-memory rejection, rejected reservations, shared-memory accounting, aggregate-plus-task quota composition, multi-process quota boundaries, per-device counters, fork re-registration, continuous stale-process recovery, committed-byte recovery grace, stale-process reservation/commit recovery, safe region cleanup, reservation lifecycle checks, robust-mutex owner-death recovery, transactional explicit-task admission with registration rollback and dispatch serialization, endpoint-to-executor lifecycle transitions, priority-ordered endpoint claims, combined remote `ACQUIRE` fast-path and queued fallback, bounded task-specific `WAIT` admission, remote lease claim/heartbeat/complete/fail transitions, task-specific process-bound claims with cross-peer rejection, optional PID/UID/start-time lease ownership binding, lease expiry and quota recovery, queue-full backpressure, read-only stats and latency metrics snapshots, opt-in dispatch-order and completion-timing observations, adaptive SLO reservation updates, versioned task-protocol request/response validation, authenticated Unix-socket request/response handling, persistent request/response pairs, and request-count-aware service shutdown. | No |
| `app` | Service and client argument validation and usage smoke tests, plus a multi-process service/client check covering quota rejection, queued-work backpressure, stats snapshots, empty claims, lease metadata, running-slot exclusion, heartbeat renewal, explicit completion, and explicit failure. | No |
| `interceptor` | Driver/Runtime preload coverage through fake CUDA Driver, Runtime, and NVML libraries, `cuInit`, kernel and graph launch forwarding (including PTDS aliases), process-local enforce-mode launch admission and event completion, batched lease reuse and terminal event draining, `dlsym` including explicit CUDA Driver/Runtime/NVML handles, both `cuGetProcAddress` forms, invalid-argument rejection, legacy/versioned and PTDS allocation/query aliases, exact PTDS availability checks, context-aware allocation records, duplicate-pointer degraded-state handling, ambiguous successful-null allocation rollback, context-bound cleanup that preserves context-independent allocations, stream cleanup, device-grouped asynchronous completion, fork reinitialization of local allocation metadata, `cuDeviceTotalMem_v2`, `cuMemAllocManaged`, `cuMemAllocPitch_v2`, Runtime `cudaMalloc3D`, physical-capacity clamping and free-memory rejection, stream-ordered Driver/Runtime allocation/free and completion accounting, memory-pool lifecycle and import policy, device-resident `cuMemCreate`/`cuMemRelease` VMM handle accounting with retain/release references, VMM address reserve/map/access/unmap/free and query/import policy, CUDA IPC export/close forwarding and quota-enabled import rejection for Driver and Runtime APIs, NVML initialization/device lookup and passthrough plus v1/v2 memory-view virtualization, process-scoped task-limit enforcement, deterministic allocation-registry tests, and injectable Driver dispatch tests. | GPU tests for real CUDA/NVML routing, Driver and Runtime kernel/graph launches, and cross-process shared quota; no GPU for symbol, fake preload, registry, dispatch, and core tests |

The current interceptor milestone covers the Driver stream-ordered allocation
path, its PTDS aliases, independently accounted Runtime async and pool
allocation paths, explicit completion boundaries, memory-pool lifecycle
forwarding with quota-enabled import rejection, CUDA IPC export/close
forwarding with quota-enabled import rejection, external-memory and
array/mipmapped-array routing with quota-enabled rejection, sparse array
mapping rejection, graphics interop mapping rejection, and graph memory node
rejection, the device-resident VMM
physical handle path with retain/release references, VMM address and query
policy, NVML passthrough and memory-view virtualization, and the shared-memory
quota control path. Remaining API families and their exact guarantees remain
governed by [CUDA_API_COVERAGE.md](CUDA_API_COVERAGE.md). The fake preload
suite also runs dedicated quota-disabled passthrough checks for POSIX-handle
imports and NVML physical fields.

The CUDA GPU preset also builds `glimmer_cuda_interceptor_kernel_gpu_test`. It
loads an embedded PTX module through the Driver API, allocates its output with
the intercepted allocator, launches the kernel through the intercepted
`cuLaunchKernel` and extended-launch paths, synchronizes, verifies the device result, and cleans up
under `LD_PRELOAD`. This is the first end-to-end real CUDA workload test. The
public and compiler-generated Runtime launch symbols are also forwarded and
covered by the fake Runtime forwarding test. `GLIMMER_TRACE_KERNEL_LAUNCHES=1`
prints one structured, allocation-free line per successful launch, including
the API, grid, block, shared-memory size, stream, and process-local sequence
number. A hardware regression test for a Runtime-compiled kernel is provided by
`glimmer_cuda_interceptor_runtime_kernel_gpu_test` when the CUDA GPU preset is
enabled. Its batched enforce variant launches the same kernel twice with
`GLIMMER_SCHEDULER_BATCH_SIZE=2` to exercise Runtime lease reuse and final-event
release on real CUDA. `glimmer_cuda_interceptor_runtime_shared_quota_gpu_test` forks two
Runtime processes with the same tenant identity: one holds 6 MiB and the other
must be rejected when it requests 4 MiB from the shared 8 MiB quota. The holder
then releases its allocation and verifies that its visible quota is restored.
The same Driver and Runtime kernel binaries have enforce-mode CTest entries;
those entries exercise the real Driver event completion path in addition to
the observe-mode forwarding checks.

`GLIMMER_TRACE_MEMORY_INFO=1` reports the virtualized CUDA memory view after
successful `cuMemGetInfo_v2` or `cudaMemGetInfo` calls, including visible total,
used, and free bytes plus the physical total and free bytes used for the
capacity boundary.

`GLIMMER_TRACE_LAUNCH_TIMINGS=1` reports the opt-in launch-path timing records.
The records are intended for workload analysis only; tests must not assert
fixed duration values. Control tests verify that timing snapshots distinguish
local and remote requests and account for remote transport calls.

The same preset builds `glimmer_cuda_workload`, an ordinary CUDA Runtime
application under `examples/cuda_workloads/baseline/`. Its CTest entry runs a
1 MiB allocation and kernel workload with an 8 MiB preload quota and requires
a CSV row showing the virtualized capacity. The same directory also builds the
workload matrix for asynchronous and explicit pool allocation, managed memory,
pitched memory, and two-stream execution. Each matrix entry uses a distinct
executable and emits a summary line with memory snapshots and a validation
flag. Run the executables directly to vary the baseline allocation size or to
compare the matrix paths; an allocation larger than the configured quota is
expected to return a non-zero status.

The priority demo also has a GPU smoke test for its cuBLAS model-shaped
inference path. The smoke test uses a small matrix and observe-mode preload to
verify that the ordinary cuBLAS application remains transparent to the
interceptor; the larger native-versus-priority comparison remains an explicit
example run.

The optional `examples/framework_workloads/pytorch_smoke/` workload is not a
CMake or CTest target because PyTorch is supplied by the host environment. It
uses public PyTorch APIs for an MLP inference or training step and reports
framework/CUDA versions, validation, latency percentiles, throughput, and peak
allocated memory. Run it with the same arguments in `native`, `observe`, and
`enforce` modes; a missing PyTorch installation is an explicit unavailable
environment rather than a test failure in the core suite. Its co-location
runner adds a readiness barrier, shared start barrier, process IDs, CUDA device
UUIDs, and wall-clock start/end records; it reports success only when both
processes use the same physical GPU and their execution intervals overlap.
For performance validation, use `run_slo_validation.py --duration-seconds 10
--repetitions 3 --inference-target-ms 2` in that example directory. Its
`wall_clock_v2` metrics measure host submission through synchronization and
completed steps per common wall-clock second. Both workers run to a shared
deadline, with at least 95 percent window coverage and explicit boundary
exclusions. The four configurations compare native, priority without
reservation, static reservation, and adaptive reservation with detailed
tracing disabled. Every repetition also runs native training and inference
alone, using matched role-local parameters and duration-window accounting.
Separate ratios quantify sharing interference relative to solo and the net
scheduler effect relative to native co-location. Missing, duplicate, or
mismatched baselines fail analysis; experiment completion is not a performance
pass. `glimmer_pytorch_measurements_test` validates the metric and
window accounting through CTest when a Python interpreter is available; it
uses only the standard library and does not need PyTorch or a GPU.
The inference process accepts an optional latency target and records target
misses without changing scheduler behavior. The runner can apply an explicit
maximum miss-ratio gate for a chosen experiment. The
`analyze_priority_matrix.py` helper validates and aggregates repeated matrix
rows, including control-service queue-wait and service-time counters.
The `run_priority_matrix.sh` helper repeats a fixed tensor-batch workload across
native, one-slot, two-slot, and batched-training configurations, writing a CSV
that separates framework batch size from scheduler launch batch size. It is a
manual GPU benchmark rather than a CTest target; do not use its timing output
as a guarantee of application-wide performance.

The same CUDA presets build `glimmer_cuda_task_backend_demo` under
`examples/cuda_task_backend/`. Its optional GPU test submits two explicit PTX
tasks through `CudaTaskController` and requires both tasks to complete with
the scheduler quota released. This path is not enabled by `LD_PRELOAD` and
does not queue arbitrary application launches.

The directory also provides `glimmer_cuda_multi_tenant_demo`, which submits
three weight-2 tasks for one tenant and two weight-1 tasks for another. Its GPU
test checks the weighted dispatch order and verifies that all task quota is
released after CUDA event completion.

The priority co-location runner can pass `--trace-scheduler` to the remote
service. The resulting `control-service.log` contains the authoritative
admission order, including task priority and sequence, while the process CSV
files retain latency and co-location measurements. The trace is for diagnosis
and is disabled for benchmark timing unless explicitly requested.

`glimmer_control_task_admission_concurrency_test` runs four process-bound
clients through 4,000 submit/claim/heartbeat/complete cycles while scanning for
expired leases. It checks progress, exactly-once completion observations, and
zero remaining quota. Competing claims and terminal reports also check
idempotence, cross-peer rejection, and exactly-once terminal accounting.
A coordinated feedback test verifies that other leases
can progress while a completion callback is paused and that changing reserved
capacity wakes a blocked claimant. Run these regressions under `asan-ubsan`
as well as the regular control tests; sanitizers supplement, rather than prove,
the shared-lock and lock-order review.

The control service also supports opt-in adaptive priority reservation. Enable
it with `--scheduler-policy priority` and
`--adaptive-slo-target-queue-us N`; use
`--adaptive-slo-window N` to choose the feedback window and
`--adaptive-slo-max-reserved-slots N` to cap reserved capacity. A non-zero
`--trace-scheduler` setting prints each applied reservation change. Tests
should assert the controller's window and bound behavior, not fixed wall-clock
durations or a particular convergence speed.

The socket framing and client tests cover fragmented and coalesced lines,
read-ahead across receive boundaries, newline-inclusive size limits, bounded
EOF framing, partial-line timeouts, shutdown, and connection reset after path,
timeout, PID, or transport failure changes. The server regression checks that
an oversized request's suffix cannot submit a task after the error response.
Run `glimmer_control_socket_line_reader_test`,
`glimmer_control_unix_socket_client_test`, and
`glimmer_control_unix_socket_server_test` under both Debug and ASan/UBSan.
Sandbox restrictions on socket creation, binding, or I/O are not successful
transport validation: rerun socket tests and the two control process tests on
an unrestricted Linux host. No test asserts a fixed performance improvement.

## Test design

- Use descriptive test names that state the condition and expected result.
- Keep setup explicit. Prefer small test fixtures and builders over shared,
  mutable global state.
- Inject time, randomness, and backend behavior when they affect a test.
- Assert both the result and the resulting resource/accounting state.
- Verify that failed launches, rejected admissions, and cancellations release
  reservations exactly once.
- The launch-gate unit test covers a full slot, a blocked waiter, timeout
  cancellation, and terminal release. The fake enforce test verifies that
  event-backed admission initializes while observe mode still forwards. Its
  batch-size-two variant verifies lease reuse and final-event draining.
- The remote launch-gate process test starts the real Linux control service and
  runs two independent processes through one global launch slot. It verifies
  task-specific claims, bounded `WAIT` admission, persistent per-process
  request reuse, and serialization; environments that cannot bind Unix sockets
  report an explicit CTest skip.
- If Driver event creation, recording, querying, or destruction fails after
  enforce-mode admission, verify that the launch lease is failed and no slot
  remains occupied. If the event API is unavailable, verify the fail-closed
  `CUDA_ERROR_NOT_SUPPORTED` result.
- Exercise the control protocol codec with canonical round trips plus bounded
  `WAIT` requests, optional
  priority and legacy submit forms, empty,
  oversized, unsupported-version, malformed, unsafe-tenant, zero, and
  overflowing input. Test response states and error codes as well as request
  operations.
- Do not use arbitrary sleeps to wait for asynchronous behavior. Use explicit
  synchronization, events, or a controllable test executor.

For CUDA quota tests, assert both the CUDA result and the accounting-visible
state. A successful allocation must reduce visible free memory, a successful
free must restore it, and rejected or failed operations must leave usage
unchanged. Unknown or duplicate releases must never reduce usage.

Tests that exercise the preload library must use the exact ABI symbol and
signature under test. The real Driver library is loaded dynamically; tests must
not silently replace it with a different implementation.

## Build configuration matrix

Each preset has its own build directory. Do not switch CUDA options inside an
existing build directory; reconfigure with the intended preset instead.

| Preset | Purpose | GPU required |
| --- | --- | --- |
| `debug` | Standard build and no-GPU tests. | No |
| `asan-ubsan` | Ownership, lifetime, and undefined-behavior checks. | No |
| `cuda-debug` | Builds `libglimmer_cuda_interceptor.so` and runs no-GPU tests. | CUDA Toolkit, no GPU |
| `cuda-lint` | Runs clang-tidy for the interceptor and CUDA test targets. | CUDA Toolkit, no GPU |
| `cuda-gpu` | Builds the interceptor and enables CUDA integration tests. | CUDA Toolkit and GPU |
| `cuda-perf` | Same CUDA features/tests with RelWithDebInfo; optimized performance baseline. | CUDA Toolkit and GPU |
| `lint` | Runs clang-tidy when installed. | No |

## Running tests

Build and run the standard suite with:

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug --output-on-failure
```

Run formatting verification alongside tests:

```sh
cmake --build --preset debug --target format-check
```

For the complete pre-commit verification, use the project check script:

```sh
./scripts/check.sh
```

It runs all available no-GPU, sanitizer, clang-tidy, formatting, and CUDA lint
checks, plus ShellCheck for repository and example scripts. ShellCheck is
skipped with an explicit warning when it is not installed. CUDA hardware tests
remain opt-in and are not run by this script.
When `nvcc` is available, it also builds `cuda-perf` and runs its no-GPU tests,
excluding the `*_gpu_test` entries. This catches optimization-dependent
regressions without requiring GPU access; the full optimized GPU suite remains
an explicit command.

Run the sanitizer suite when changing memory ownership, lifetime, or
asynchronous behavior:

```sh
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan
ctest --preset asan-ubsan --output-on-failure
```

Build and test the CUDA interceptor without a GPU:

```sh
cmake --preset cuda-debug
cmake --build --preset cuda-debug
cmake --build --preset cuda-debug --target format-check
ctest --preset cuda-debug --output-on-failure
```

Run the optional GPU suite only on a compatible Linux CUDA host:

```sh
cmake --preset cuda-gpu
cmake --build --preset cuda-gpu
ctest --preset cuda-gpu --output-on-failure
```

Run only the ordinary Runtime workload matrix with the interceptor:

```sh
ctest --preset cuda-gpu \
  -R 'glimmer_cuda_(workload_gpu_test|async_workload_gpu_test|managed_workload_gpu_test|pitched_workload_gpu_test|multistream_workload_gpu_test)' \
  --output-on-failure
```

On a split-driver Linux installation, record the loader and vendor Driver
objects visible to the process (for example with `LD_DEBUG=libs`) alongside
the GPU result. The interceptor must resolve the vendor object when the
system exposes a loader shim, as in WSL; a successful no-GPU test alone does
not verify that deployment path.

The root `compile_commands.json` is updated by the build and points to the
most recently built preset. Use `cuda-debug` last when editor diagnostics need
the CUDA interceptor's Toolkit include paths.

For optimized performance measurements, build and test `cuda-perf` instead of
overriding the Debug cache in place. Run
`python3 examples/framework_workloads/pytorch_smoke/run_overhead_baseline.py`
to compare native, preload-only, quota-only, local admission, and remote
admission for each role separately. The runner rejects Debug/sanitizer builds,
checks actual runtime compile flags, keeps tracing disabled, verifies the
quota-visible capacity before timing, and requires completed remote admissions.
Results and build/source fingerprints stay in the example's ignored output
directory. Inspect p99 sample counts and repeated-run ranges; success means
measurement completion, not a performance pass. Run the common-window
co-location experiment separately with `--build-dir build/cuda-perf`.

`glimmer_pytorch_overhead_baseline_test` covers mode construction, inherited
environment cleanup, optimized-build checks, matched workload parameters,
quota verification, complete repetition sets, identity/metric validation,
service evidence and cleanup, and failure/output preservation. It uses only
the Python standard library and runs without PyTorch or a GPU. See
[ADR 0036](decisions/0036-optimized-uncontended-baselines.md) for scope and
limitations. Do not run builds, tests, or other benchmarks alongside timed GPU
experiments.

`glimmer_pytorch_launch_profile_test` checks the optional warmed CUDA profiler
range, exception cleanup, disabled behavior, trace command/environment
construction, read-only SQLite analysis, interval union/gaps, missing or
malformed GPU data, diagnostics, quota checks, and preservation of failed
outputs. It requires neither Nsight nor PyTorch. The measurement suite rejects
profiled data as ordinary benchmark results. Run
`run_launch_profile.py --mode local --role training --duration-seconds 1`
from the PyTorch example path for optional real Nsight validation; repeat
native/quota/remote and inference separately. See
[ADR 0037](decisions/0037-warmed-cuda-timeline-diagnostics.md). Profiling success
is not a performance or complete-interception-coverage pass.

The following additional interceptor regressions require CUDA headers but not
GPU hardware:

- `glimmer_cuda_launch_completion_tracker_test`: deterministic close-during-destroy
  ownership, destruction/query/recording/creation failures, multi-event batches,
  pending-event shutdown, and NOT_READY progress before completion.
- `glimmer_cuda_extended_launch_test`: default and PTDS Driver/Runtime extended
  symbols through explicit handles, default lookup and Driver proc-address
  flags, unchanged configuration/attribute/argument pointers, and native errors.
- `glimmer_cuda_interceptor_dispatch_test`: optional extended entry points and
  real-call reentrancy guards, including missing-symbol failure.

`glimmer_cuda_extended_launch_gpu_test` additionally executes both Runtime
extended aliases with a real kernel and a launch attribute under one-slot
enforcement. The Driver PTX GPU test exercises both proc-address stream flags.
The PyTorch measurement test rejects failed, cancelled, undrained, and malformed
scheduler metrics even when application output is valid. See
[ADR 0038](decisions/0038-extended-launch-and-completion-correctness.md).

When clang-tidy is unavailable, report it explicitly rather than treating the
lint preset as passed. When GPU hardware or the CUDA Toolkit is unavailable,
run the no-GPU suite and report the skipped GPU checks.

If a required test cannot run, state the missing dependency or hardware
requirement and describe the alternative verification performed.
