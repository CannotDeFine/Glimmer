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
| `control` | Quota-visible memory information, physical-memory bounds, and rejected reservations. | No |
| `interceptor` | M1 Driver API integration for `cuMemAlloc_v2`, `cuMemFree_v2`, and `cuMemGetInfo_v2`, including successful allocation, release, quota rejection, invalid arguments, and unknown release. | GPU test: yes |

The current interceptor milestone does not claim coverage for `dlsym`,
`cuGetProcAddress`, async allocation, memory pools, VMM, NVML, or multi-process
shared accounting. Those remain governed by
[CUDA_API_COVERAGE.md](CUDA_API_COVERAGE.md).

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
