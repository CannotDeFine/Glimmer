# Testing

[Project home](../README.md) / [Documentation](../docs/README.md) / Testing

This guide contains testing rules and entry points. Test registration in CMake
and documented API contracts define feature-specific requirements. Results from
an earlier source or binary version do not establish that a changed candidate
passes.

## Choosing tests

- Test observable behavior and important failure cases, not one test per
  function or implementation detail. Extend an existing fixture when it covers
  the same contract; add a new test when it provides distinct confidence.
- Prefer fast, deterministic CPU tests for logic that does not require a GPU.
  Use integration, real-GPU and application tests for boundaries that smaller
  tests cannot establish. Keep each test independent and give long-running
  tests explicit timeouts.
- Review test maintenance cost and repeated coverage as well as coverage gaps.
  Consolidate or retire a test only after verifying that its important behavior
  remains covered; do not remove tests merely to reduce their count.
- Register maintained checks used by a standard configuration with CTest.
  Feature-specific tests may be registered only when their dependencies are
  enabled. A standalone script is not run by CTest or CI unless explicitly
  registered or invoked; run it directly when its change requires it.

This follows the principles in Google's guidance on
[testing behaviors](https://testing.googleblog.com/2014/04/testing-on-toilet-test-behaviors-not.html)
and [test sizes](https://testing.googleblog.com/2010/12/test-sizes.html).

The repository CI builds and tests the default host Debug and sanitizer
configurations, checks formatting, and builds the host clang-tidy configuration.
It runs ShellCheck on repository shell scripts when available and warns when it
is skipped. It does not enable optional CUDA targets or run GPU tests. Those paths require
separate checks in a compatible environment. CI uses `ubuntu-latest`, whose
image and tool versions can change; record them when investigating compatibility.

The CI sanitizer job excludes tests labeled `no-instrumented-code`, which run
in the Debug job. Use this label only for tests registered in the default Debug
configuration that neither load nor invoke instrumented Glimmer code; remove
it if either condition changes. The check-script fixture compares the default
host Debug and sanitizer registrations; the filter itself does not enforce the
rule for optional configurations or verify whether a test invokes instrumented
code. Running CTest directly without a label filter still selects those tests.

Configuring the root project with `BUILD_TESTING=ON` requires a Python 3.10 or
newer interpreter: CMake fails at configuration if it is unavailable, rather
than silently omitting the Python tests. A build with `BUILD_TESTING=OFF` does
not require Python for the core C++ targets.

## Required checks

Follow the [development lifecycle](DEVELOPMENT_LIFECYCLE.md#stage-4-verify-with-evidence)
for the checks required by each change class. In particular:

- Every behavior change or defect fix needs a regression test and relevant
  success, rejection, failure and cleanup coverage.
- Keep scheduler policy/accounting tests deterministic and hardware-independent,
  with a simulated backend and controllable time. GPU tests complement them.
- Cover ownership, partial initialization, rollback, cancellation, timeouts,
  stale/foreign identities and exactly-once terminal transitions where relevant.
- Exercise ABI/symbol lookup, reentrancy, original error propagation and complete
  numerical output for intercepted paths; exported symbols alone are not support.
- Run ASan/UBSan for ownership, lifetime and asynchronous changes, and applicable
  clang-tidy/CUDA lint and hardware checks for CUDA changes.
- Presets cover only enabled targets: `cuda-lint` does not enable the optional
  execution service or frontend. Analyze those targets with an appropriately
  configured build when a change affects them, and report the SDK profile used.
- The lint presets apply clang-tidy to C++ sources, not CUDA `.cu` sources.
  Compiling a CUDA target does not by itself provide static analysis of its
  device code or validate its behavior on a GPU.
- Do not weaken an oracle, tolerance or resource gate to obtain a pass. Keep
  native and intercepted/replacement inputs and runtime identities matched.

## Standard commands

For a compatible default host-side build:

```sh
cmake --preset debug
cmake --build --preset debug -j2
ctest --preset debug --output-on-failure --no-tests=error
cmake --build --preset debug --target format-check
```

CTest runs the tests registered in the selected build configuration. Use
`ctest --preset debug -N` to list them without executing them, or select an
affected test by name with `ctest --preset debug --output-on-failure --no-tests=error -R
'^glimmer_core_execution_priority_test$'`. Run a standalone Python test with
`python3 -B path/to/test_file.py`, substituting its actual path. A preset does
not reset every previously cached option: inspect the cache and test list before
assuming an existing build excludes GPU tests. Registration does not guarantee
that a test executable still exists after build cleanup; rebuild the affected
targets before running those tests. The `--no-tests=error` option makes an empty
selection fail instead of reporting success. A skipped test is still selected:
CTest can exit successfully even if every selected test is skipped. Inspect
the summary and report missing hardware or other skipped coverage separately.
Each registration is counted separately, even when cases share a test executable;
different arguments or environments can exercise distinct behavior.

New root-project and standalone GEMM configurations use a 180-second default
per test. A test's explicit `TIMEOUT` takes precedence; set
`DART_TESTING_TIMEOUT` when configuring or pass `ctest --timeout` to choose a
different fallback. Existing caches retain their configured value.

For a focused change, select relevant tests instead of repeating the entire
suite after every edit. A clang-tidy build checks source during compilation; if
its runtime configuration matches the Debug build, rerunning the same tests
solely because clang-tidy ran adds no distinct runtime configuration. Check the
actual cache and test list first: sanitizer, optimized, CUDA-enabled or other
feature-specific builds can exercise different behavior and still need their
applicable tests. Run the full relevant gates before review.

By default, building updates the root `compile_commands.json` link to that
build. It refuses to replace an existing non-symlink file or directory. Configure
with `-DCMAKE_EXPORT_COMPILE_COMMANDS=OFF` when an isolated check must leave the
existing link unchanged.

If the preset slot contains evidence or an incompatible cache, a temporary
external build can run the host checks without changing that slot or the editor
link. Check available space first and retain the directory path for cleanup:

```sh
glimmer_check_dir="$(mktemp -d /tmp/glimmer-check.XXXXXX)"
cmake -S . -B "$glimmer_check_dir" -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF -DGLIMMER_ENABLE_CUDA_GPU_TESTS=OFF
cmake --build "$glimmer_check_dir" -j2
CUDA_VISIBLE_DEVICES='' ctest --test-dir "$glimmer_check_dir" --output-on-failure --no-tests=error
```

This uses the default host Debug options, not the cached options from an existing
preset slot. Remove disposable outputs after review under the
[storage rules](STORAGE_LAYOUT.md).

For host ownership and lifetime checks:

```sh
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan -j2
ctest --preset asan-ubsan --output-on-failure --no-tests=error
```

The instrumented loaded-image fixture skips its separate dynamic-loader
namespace case because that fixture cannot load a second ASan runtime. Run the
ordinary Debug test as well when changing that behavior.

`cuda-debug` builds the optional launch-admission path with hardware tests off;
`cuda-gpu` enables actual device tests. Optional execution-service/frontend
targets need their own explicit build options. Use the [build guide](BUILDING.md)
and inspect the selected cache; a preset name does not erase prior settings.

`./scripts/check.sh` runs the repository's available host, sanitizer, formatting,
static-analysis and script checks. It configures several build slots, so inspect
their caches and the [storage rules](STORAGE_LAYOUT.md) first. It refuses to
reconfigure a symlinked or frozen slot, a symlinked cache, a nonempty slot
without a regular cache, or a slot with GPU tests enabled. It also refuses an
existing non-symlink compilation database. Preserve any needed evidence before
selecting a compatible build. It checks `build/` against the 10 GB limit before
and after successful configure, build and test steps; it cannot predict or
interrupt peak usage within a step. It configures
each selected preset with GPU tests off, including its `cuda-perf` check, sets
`CUDA_VISIBLE_DEVICES` to empty for CTest and filters `_gpu_test` names in the
optimized CUDA check. Optional lint and CUDA checks are selected using
`clang-tidy` and `nvcc` on `PATH`; a CMake Toolkit setting alone does not select
them, and the `nvcc` found on `PATH` may differ from the Toolkit CMake selects. The
filter is a naming convention, not a security
boundary for programs that access hardware outside CUDA visibility controls;
some GPU-gated tests have other names, so the CMake setting is also necessary.
It does not replace the explicitly requested GPU suite. Unless `ASAN_OPTIONS`
is set, the script uses `detect_leaks=0`; its sanitizer pass therefore does not
establish leak freedom. Supply appropriate options and run leak checks separately
when needed. Its non-Debug passes exclude `no-instrumented-code` tests after
running them in Debug; this does not verify those tests under different Python
interpreters or fixture toolchains that may differ between presets. Record
unavailable tools or hardware and the alternative checks; an unavailable gate
is not a passing result.

## Hardware and measurement evidence

Coordinate device use and do not run competing tests or builds alongside timed
experiments. Preserve failed attempts, exact commands, source/binary/library and
GPU identities, full correctness results and cleanup evidence. Reuse compatible
working builds and preserve frozen baselines under the
[storage policy](STORAGE_LAYOUT.md).

Keep ordinary correctness/timing, instrumented counters and physical scheduling
observations separate. Passing a component fixture does not qualify an entire
framework or application. Worker count, occupancy capacity and GPU busy time
alone do not establish useful compute on every SM or sustained saturation.

Formal comparisons must meet the
[full-device qualification rules](DEVELOPMENT_LIFECYCLE.md#full-device-comparison-qualification)
for each compared path, with complete matched correctness. Missing evidence
fails qualification; diagnostic-only runs must not become performance claims.
State application-specific inputs, supported operations and acceptance criteria
before collection. Do not substitute a narrower fixture for that application.
