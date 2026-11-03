# Coding Guidelines

[Project home](../README.md) / [Documentation](../docs/README.md) / Coding guidelines

These guidelines keep Glimmer small, predictable, and maintainable as it grows
from a prototype into a shared-GPU scheduler.

## Formatting and language

- Write source code, comments, documentation, and identifiers in English.
- Target Linux and C++20 only.
- Format C++ and CUDA source with the repository `.clang-format` file. Run
  `cmake --build --preset debug --target format-check` before committing. The
  target checks headers in `include/` and C++/CUDA sources and headers in
  `src/`, `examples/` and `tests/`; it excludes `build/`, `output/` and `.venv/`
  directories.
- The checked-in `.clang-format` and `.clang-tidy` settings are authoritative
  for Glimmer. They use four-space indentation and other local conventions;
  do not reformat unrelated code to match an external guide.
- Prefer clear, conventional C++ over clever or overly generic abstractions.
- Keep functions short enough that their control flow and ownership rules are
  obvious. Extract a named helper when a function combines unrelated steps.

## Naming

- In C++, use `PascalCase` for types and `snake_case` for functions and
  variables. Namespace-scope constants use `kPascalCase`; local and class
  constants and data members use `snake_case`, as configured in `.clang-tidy`.
- Preserve names required by CUDA and other external APIs at their ABI boundary.
- Name booleans with `is_`, `has_`, `can_`, or `should_` prefixes.
- Name units explicitly when ambiguity is possible: `timeout_ms`,
  `memory_bytes`, and `queue_depth`.
- Avoid unexplained abbreviations and names that encode implementation details
  instead of domain meaning.

## Python and shell

- Use descriptive Python names, small functions, standard-library facilities
  when sufficient, and a `main()` guard for executable modules. Importing a
  module should not start workloads or modify the environment.
- Validate arguments and external failures explicitly; do not use `assert` for
  required runtime checks. Use context managers or `finally` for owned resources.
  Catch broad exceptions only at a boundary that records or propagates them.
- Keep shell scripts focused on command orchestration, quote expansions, check
  failures, and clean up resources they create. Prefer Python for complex data
  processing. Run ShellCheck on changed shell scripts when available.
- Preserve existing local conventions when editing; avoid unrelated rewrites or
  introducing a new formatter or dependency just to enforce style.

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

- Resolve compiler warnings in Glimmer-owned code. CI enables warnings as
  errors in its host configurations; optional CUDA targets need separate checks.
- Run the appropriate CMake preset for the requested configuration. Use the
  `asan-ubsan` preset when changing ownership, lifetime, or asynchronous code.
- Run the `lint` preset when clang-tidy is available and resolve findings in
  Glimmer-owned code before merging.
- The CMake lint configuration selects header paths beginning with this
  checkout's `include/` or `src/` directory. A relative include that traverses
  out of those directories can also select an external header. When invoking
  clang-tidy directly, supply a `--header-filter` if its default is too broad.

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
- The preload interceptor's loader and accounting-failure boundaries may use
  its fixed-string `report_diagnostic` write path instead of `spdlog`; its
  opt-in kernel trace uses a fixed buffer for the same reason. Keep those
  messages rare or explicitly enabled, allocation-free, and free of CUDA or
  loader calls.
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

## Failure containment

- Validate externally supplied pointers, sizes, handles, environment values,
  and symbol-resolution results before dereferencing or using them.
- Do not allow C++ exceptions to cross a C, CUDA, POSIX, or dynamic-loader ABI
  boundary. Convert them to a documented error result at that boundary.
- Return the original external error unchanged whenever Glimmer has not made a
  policy decision. Return a Glimmer-specific error only for a documented
  rejection or an unrecoverable internal consistency failure.
- If allocation accounting or rollback cannot be completed after a real GPU
  allocation succeeds, retain the reservation and reject later allocations.
  Do not release or undercount quota merely to keep the process running.
- Treat unknown or duplicate release requests as non-accounting events. A
  release must never reduce usage unless it matches a successful, recorded
  allocation.
- Guard all arithmetic used for memory sizes, quotas, and counters against
  overflow and underflow. Saturate diagnostic values or reject requests rather
  than wrapping.

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

- Follow [TESTING.md](TESTING.md) for behavior changes, deterministic scheduler
  tests, and applicable failure and rollback coverage.
- Update the relevant architecture and API documentation when changing
  public APIs, module boundaries, scheduling semantics, or resource isolation.
- Keep public guides focused on implemented behavior, usage, prerequisites and
  limitations. Use relative links, keep examples runnable, and avoid credentials
  or machine-specific paths. Update existing guidance instead of duplicating it.

## References

These are references, not a claim that every Google rule is adopted. Glimmer's
local settings and the rules above take precedence:

- [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Google Shell Style Guide](https://google.github.io/styleguide/shellguide.html)
