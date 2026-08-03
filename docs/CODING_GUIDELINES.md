# Coding Guidelines

These guidelines keep Glimmer small, predictable, and maintainable as it grows
from a prototype into a shared-GPU scheduler.

## Formatting and language

- Write source code, comments, documentation, and identifiers in English.
- Target Linux and C++20 only.
- Format C++ and CUDA source with the repository `.clang-format` file. Run
  `cmake --build build --target format-check` before committing.
- Prefer clear, conventional C++ over clever or overly generic abstractions.
- Keep functions short enough that their control flow and ownership rules are
  obvious. Extract a named helper when a function combines unrelated steps.

## Naming

- Use `PascalCase` for types, `snake_case` for functions and variables, and
  `kPascalCase` for constants.
- Name booleans with `is_`, `has_`, `can_`, or `should_` prefixes.
- Name units explicitly when ambiguity is possible: `timeout_ms`,
  `memory_bytes`, and `queue_depth`.
- Avoid unexplained abbreviations and names that encode implementation details
  instead of domain meaning.

## Architecture and dependencies

- Give each component one clear responsibility. Keep scheduling policy, CUDA
  execution, API interception, accounting, and metrics in separate modules.
- Keep the scheduling core independent of CUDA, NVML, and dynamic-linker
  details. Access GPU-specific behavior through a narrow backend interface.
- Keep public headers minimal. Do not expose CUDA or third-party types from a
  public interface unless that dependency is intentionally part of the API.
- Add a dependency only when the standard library or existing project code is
  insufficient. Do not modify `3rdparty/` except for an explicit dependency
  update.
- Follow [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) before adding,
  updating, or removing a dependency.

## Build quality

- Resolve all compiler warnings in Glimmer-owned code. CI enables warnings as
  errors for the project targets.
- Run the appropriate CMake preset for the requested configuration. Use the
  `asan-ubsan` preset when changing ownership, lifetime, or asynchronous code.
- Run the `lint` preset when clang-tidy is available and resolve findings in
  Glimmer-owned code before merging.

## Resource ownership and errors

- Use RAII for every owned resource, including memory, locks, file descriptors,
  dynamic-library handles, CUDA resources, and background threads.
- Express ownership in types. Prefer values, references, and smart pointers;
  do not use owning raw pointers.
- Check every CUDA, NVML, POSIX, and dynamic-linker result. Propagate errors
  with context or convert them into a project error type; never silently ignore
  a failed call.
- Keep cleanup safe to call after partial initialization. Destructors must not
  throw.

## Logging and assertions

- Use the project logging facility (`spdlog`) instead of `std::cout`,
  `std::cerr`, or ad-hoc printing in production code.
- Use `trace` for fine-grained diagnostics, `debug` for developer diagnostics,
  `info` for meaningful lifecycle events, `warn` for recoverable anomalies,
  `error` for failed operations, and `critical` only when the process cannot
  safely continue.
- Log an error at the layer that has the most useful context. Include the
  operation, tenant, task or request identifier, GPU/device identifier, and
  the original error code or message when available.
- Do not log secrets, credentials, complete request payloads, or data that a
  tenant has submitted for GPU processing.
- Do not emit per-operation `info` logs on high-frequency paths. Use `trace`,
  sampling, metrics, or aggregation to avoid making logging a performance
  bottleneck.
- Logging must not change program behavior. In particular, an interceptor must
  not log through a path that can invoke CUDA again or introduce recursive
  dynamic-linker calls.
- Use assertions only for programmer errors and internal invariants that must
  hold if the code is correct. Assertions must be side-effect free.
- Do not use assertions for invalid user input, CUDA/NVML/POSIX failures,
  allocation failures, or any condition that can occur in normal operation.
  Handle those conditions explicitly and return a contextual error.
- Do not rely on an assertion for required control flow: release builds may
  disable them. Use `static_assert` for compile-time invariants.

## Concurrency and interception

- Document the thread-safety contract of every shared component and public
  interface.
- Avoid global mutable state. If unavoidable, make its owner, initialization,
  synchronization, and teardown explicit.
- Define lock ordering when a component can hold more than one lock. Do not
  call external or user-controlled code while holding an internal lock.
- Keep CUDA API interceptors minimal and reentrant. They must not recursively
  invoke an intercepted CUDA API, and should avoid allocations, blocking I/O,
  and scheduler calls on hot paths unless required.
- Resolve dynamically loaded symbols once in a thread-safe manner, then retain
  the resolved function pointer for later calls.

## Scheduling semantics

- Attribute every task, reservation, and metric to a tenant before it enters a
  shared queue.
- Make admission, dispatch, completion, cancellation, and failure explicit
  state transitions. Release reservations on every terminal path.
- Do not claim kernel-level preemption that the backend cannot provide. Apply
  fairness at defined task boundaries.
- Treat quota and accounting updates as consistency-sensitive operations; their
  success and rollback behavior must be testable.

## Testing and documentation

- Add or update tests for every behavior change and every corrected defect.
- Test scheduling logic with a simulated backend and deterministic time source;
  real GPU tests complement but do not replace these tests.
- Cover failure and rollback paths, not only successful execution.
- Update the relevant architecture or decision documentation when changing
  public APIs, module boundaries, scheduling semantics, or resource isolation.
