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
| `core` | Reservation admission, rejection, commit, cancellation, release failures, concurrent reservations, lifetime safety, and counter overflow protection. | No |
| `control` | Quota-visible memory information, physical-memory bounds, rejected reservations, shared-memory accounting, multi-process quota boundaries, per-device counters, fork re-registration, continuous stale-process recovery, committed-byte recovery grace, stale-process reservation/commit recovery, safe region cleanup, reservation lifecycle checks, and robust-mutex owner-death recovery. | No |
| `interceptor` | Driver/Runtime preload coverage through fake CUDA Driver, Runtime, and NVML libraries, `cuInit`, `dlsym` including explicit CUDA Driver/Runtime/NVML handles, both `cuGetProcAddress` forms, invalid-argument rejection, legacy/versioned and PTDS allocation/query aliases, exact PTDS availability checks, context-aware allocation records, duplicate-pointer degraded-state handling, ambiguous successful-null allocation rollback, context-bound cleanup that preserves context-independent allocations, stream cleanup, device-grouped asynchronous completion, fork reinitialization of local allocation metadata, `cuDeviceTotalMem_v2`, `cuMemAllocManaged`, `cuMemAllocPitch_v2`, stream-ordered Driver/Runtime allocation/free and completion accounting, memory-pool lifecycle and import policy, device-resident `cuMemCreate`/`cuMemRelease` VMM handle accounting with retain/release references, VMM address reserve/map/access/unmap/free and query/import policy, NVML initialization/device lookup and passthrough plus v1/v2 memory-view virtualization, deterministic allocation-registry tests, and injectable Driver dispatch tests. | GPU test for real CUDA/NVML routing; no GPU for symbol, fake preload, registry, dispatch, and core tests |

The current interceptor milestone covers the Driver stream-ordered allocation
path, its PTDS aliases, independently accounted Runtime async and pool
allocation paths, explicit completion boundaries, memory-pool lifecycle
forwarding with quota-enabled import rejection, the device-resident VMM
physical handle path with retain/release references, VMM address and query
policy, NVML passthrough and memory-view virtualization, and the shared-memory
quota control path. Remaining API families and their exact guarantees remain
governed by [CUDA_API_COVERAGE.md](CUDA_API_COVERAGE.md). The fake preload
suite also runs dedicated quota-disabled passthrough checks for POSIX-handle
imports and NVML physical fields.

## Test design

- Use descriptive test names that state the condition and expected result.
- Keep setup explicit. Prefer small test fixtures and builders over shared,
  mutable global state.
- Inject time, randomness, and backend behavior when they affect a test.
- Assert both the result and the resulting resource/accounting state.
- Verify that failed launches, rejected admissions, and cancellations release
  reservations exactly once.
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
checks. CUDA hardware tests remain opt-in and are not run by this script.

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

The root `compile_commands.json` is updated by the build and points to the
most recently built preset. Use `cuda-debug` last when editor diagnostics need
the CUDA interceptor's Toolkit include paths.

When clang-tidy is unavailable, report it explicitly rather than treating the
lint preset as passed. When GPU hardware or the CUDA Toolkit is unavailable,
run the no-GPU suite and report the skipped GPU checks.

If a required test cannot run, state the missing dependency or hardware
requirement and describe the alternative verification performed.
