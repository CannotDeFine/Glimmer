# Development Lifecycle and Code Review

[Project home](../README.md) / [Documentation](../docs/README.md) / Development lifecycle

This document defines the minimum lifecycle for every Glimmer change. It is
intended to keep the single `main` branch buildable while making correctness,
test evidence, and review decisions explicit. The rules apply to code, tests,
build configuration, dependencies, documentation, and operational changes.

## Principles

- Prefer small, complete, reversible changes over large batches of work.
- Review behavior and risk, not just formatting or line count.
- Treat tests and documentation as part of the change, not follow-up work.
- Keep rules useful, consistent and practical to check; avoid duplicate policy.
- Do not claim support for a platform, CUDA path, or workload without evidence.
- Keep the repository Linux-only and preserve the module boundaries in
  [ARCHITECTURE.md](ARCHITECTURE.md).
- Record an exception when a required gate cannot run; an unreported skipped
  check is not a passing check.

## Lifecycle

Scope the change, implement it with its checks and documentation, verify it,
then review the result. Scale the written plan and evidence to the risk: a
small documentation fix does not need the same record as a CUDA ABI change.
Keep unavailable checks and unresolved risks visible; do not call them passed.

## Stage 1: Propose the change

Before editing, identify the problem, scope, acceptance criteria and affected
interfaces. For changes involving concurrency, resource accounting, CUDA,
hardware measurements or compatibility, also record risks, failure conditions,
resource budget and rollback in the issue or task notes. Keep the record brief;
a small change does not require a separate document.

Classify the change before implementation:

| Class | Minimum plan |
| --- | --- |
| Documentation-only | Identify the affected rule or user workflow and verify links and wording. |
| Internal code change | Identify ownership, error paths, concurrency behavior, and tests. |
| Behavior or API change | Define compatibility, rejection behavior, regression tests, and documentation updates. |
| Module or dependency change | Read [ARCHITECTURE.md](ARCHITECTURE.md), affected API contracts, and [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) before editing. |
| CUDA/interceptor change | Define the intercepted symbol paths, ABI signatures, reentrancy behavior, quota effects, and GPU/no-GPU test plan. |

Resolve unclear acceptance criteria or ownership before implementing them.

## Stage 2: Design before implementation

Use the relevant [coding](CODING_GUIDELINES.md), [testing](TESTING.md),
[architecture](ARCHITECTURE.md), [dependency](DEPENDENCY_MANAGEMENT.md) and
[CUDA compatibility](CUDA_API_COVERAGE.md) guides. For affected interfaces,
identify ownership, input validation, error mapping, cleanup, concurrency and
reentrancy boundaries, and the checks that demonstrate the intended behavior.
Update the architecture guide when a module responsibility, cross-module
contract or dependency direction changes.

## Stage 3: Implement the smallest complete change

- Keep one logical behavior change together: implementation, headers, tests,
  CMake wiring, and documentation belong in the same change when they are
  required for correctness.
- Do not split work by file or mix unrelated cleanup, generated files, or
  dependency updates into the change.
- Preserve public ABI signatures and check every external result.
- Add regression tests before or with a bug fix; a test must fail on the old
  behavior and pass on the new behavior when practical.
- Keep high-frequency interceptor paths reentrant and free of accidental CUDA,
  loader, allocation, or blocking-I/O recursion.
- Keep diagnostics actionable and safe. Do not commit temporary tracing,
  credentials, developer-specific paths, or generated build output. Document
  the scope of any fixed toolchain path required by a validation profile.

## Stage 4: Verify with evidence

Run the smallest applicable checks while developing, then run the full relevant
set before review. Retain the commands and results in the change notes when the
change is risky or hardware-dependent.

| Change | Required evidence |
| --- | --- |
| Documentation-only | `git diff --check` and, when staged, `git diff --cached --check`; inspect changed links and rendered sections. |
| Core or control behavior | `debug` build, CTest, format check, and `asan-ubsan` for ownership, lifetime, or concurrency changes. |
| CMake, build, or dependency change | Affected preset build and CTest; `./scripts/check.sh` when available. |
| CUDA interceptor change | `cuda-lint` or equivalent clang-tidy coverage, no-GPU interceptor tests, and `cuda-gpu` tests when a compatible GPU is available. |
| Public behavior or failure-path change | A regression test plus success, rejection, cleanup, and rollback assertions that apply to the behavior. |

Common commands include:

```sh
./scripts/check.sh
cmake --build --preset debug --target format-check
ctest --preset debug --output-on-failure --no-tests=error
```

For sanitizer or CUDA work, follow [TESTING.md](TESTING.md). Run the GPU suite
when compatible hardware is safely available and its use is authorized;
otherwise report the skipped check and the no-GPU alternative. The aggregate
script configures multiple build slots, so inspect their caches and storage
first. If using the root editor database, use `readlink compile_commands.json`
to verify that it refers to the intended build.

## Full-device comparison qualification

Before formal GPU comparisons, qualify actual SM participation for every matched
dominant operation and compared path by default. Every SM of the compared device
is required, using the count queried from that device: 80/80 on a V100 and
108/108 on an A100.
Explicit worker geometry and sufficient occupancy capacity are prerequisites,
not coverage evidence. Small correctness-test worker defaults must not leak into
measurement launchers. Preserve actual selection and numerical evidence too;
an unused replacement kernel does not qualify the application path.

Record native and replacement coverage separately. Missing, stale, partial,
unmatched or representative-only captures fail qualification, not merely add a
warning. Explicit diagnostic-only runs can inspect unqualified configurations,
but must retain false full-card/performance/preemption acceptance. Never promote
previous partial-card timings to a full-device baseline.

A user-approved fixed-workload application contract may instead require equal
whole-device resource availability and separately measured whole-path SM
participation and aggregate co-location saturation. Record that choice explicitly
before measurement; do not silently weaken an operation-level gate. Small native
operations need not individually span all SMs under this alternative, but remain
in the workload and measured cost. Available SM count is not participation;
NVML GPU busy time is not SM activity or saturation. Aggregate co-location
activity must come from the concurrent workload, not an average of serialized
counter replays. Missing aggregate observations remain an open qualification
gate. Such a contract must state the workload, resource bounds, measured scope
and acceptance criteria in the comparison's reproducibility record. It does not
change standalone GEMM qualification or historical acceptance flags.

Counter collection is separate from ordinary timing. Bind it to GPU/driver,
binary/source/vendor-library and workload identities. Use unchanged correctness
oracles. Report active-cycle utilization, achieved occupancy and throughput
separately from spatial SM coverage; participation does not prove continuous
saturation. Profiling replay cannot qualify latency or physical priority overlap.
Do not change MPS, clocks, driver permissions or GPU resource partitioning merely
to make a gate pass. Reuse compatible builds and fresh small evidence folders.

## Generated artifact retention and cleanup

Follow [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md) before building and when finishing
experiments. Reuse compatible build slots, budget storage, and preserve frozen
baselines and unique evidence. A new run does not require another full build.
Declare large-output retention before collection; record cleanup and any lost
reanalysis capability after review. Ignoring a directory does not authorize
deleting its contents.

## Stage 5: Review protocol

The author reviews the complete change even when working alone. Check:

- Correctness: intended behavior, input validation, error mapping, ownership,
  cleanup, concurrency, accounting and ABI boundaries affected by the change.
- Maintainability: module boundaries, complexity, naming, tests, build wiring
  and documentation. Tests should verify behavior and meaningful failure paths.
- Evidence: the appropriate checks passed; skipped checks and remaining risks
  are described accurately.

For each finding, identify its location, impact and a concrete resolution.
Resolve a `Blocker` (such as a crash, security issue, broken ABI or incorrect
accounting) or `Major` issue (such as missing required behavior or a likely
race) before integration. Fix a `Minor` issue when practical or record a
follow-up; a `Nit` is optional. Recheck fixes and their affected tests.
Automated formatters and analyzers own mechanical style checks. Prefer an
improvement that meets these gates over delaying it for optional polish.

For more review guidance, see Google's
[code review practices](https://google.github.io/eng-practices/review/reviewer/).

## Stage 6: Integrate on the single `main` branch

Follow the [Git guidelines](GIT_GUIDELINES.md) for logical changes, staging and
commit messages. Preserve unrelated work and avoid rewriting shared history.

## Definition of Done

A change is ready when its acceptance criteria, applicable checks and review
are satisfied, documentation reflects actual behavior, and no blocking finding
remains. Record unavailable checks, their risks and any follow-up; do not treat
an unrun check as a pass. Exclude credentials and generated artifacts.
