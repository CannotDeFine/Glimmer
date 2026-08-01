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

## Running tests

Build and run the standard suite with:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Run formatting verification alongside tests:

```sh
cmake --build build --target format-check
```

If a required test cannot run, state the missing dependency or hardware
requirement and describe the alternative verification performed.
