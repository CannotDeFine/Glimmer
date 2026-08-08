# Development Lifecycle and Code Review

This document defines the minimum lifecycle for every Glimmer change. It is
intended to keep the single `main` branch buildable while making correctness,
test evidence, and review decisions explicit. The rules apply to code, tests,
build configuration, dependencies, documentation, and operational changes.

## Principles

- Prefer small, complete, reversible changes over large batches of work.
- Review behavior and risk, not just formatting or line count.
- Treat tests and documentation as part of the change, not follow-up work.
- Do not claim support for a platform, CUDA path, or workload without evidence.
- Keep the repository Linux-only and preserve the module boundaries in
  [ARCHITECTURE.md](ARCHITECTURE.md).
- Record an exception when a required gate cannot run; an unreported skipped
  check is not a passing check.

## Lifecycle

Every change moves through the following states. `Blocked` means that progress
requires an external decision or dependency; `Deferred` means the work is
intentionally postponed with a documented reason.

```text
Proposed -> Designed -> Implemented -> Verified -> Reviewed -> Integrated -> Released
    |           |            |            |           |            |
    +--------- Blocked or Deferred at any gate when its exit criteria are unmet
```

| Stage | Required output | Exit criteria |
| --- | --- | --- |
| Proposed | Scope, non-goals, acceptance criteria, risk, and rollback plan. | The problem and observable result are unambiguous. |
| Designed | Affected modules, interfaces, data flow, and decisions. | Architecture and dependency impacts are understood; required ADRs are identified. |
| Implemented | Source, tests, build wiring, and documentation for the logical change. | The change is complete enough to build and review as one unit. |
| Verified | Reproducible commands and their results. | Required build, test, lint, sanitizer, format, and hardware checks pass or are explicitly recorded as unavailable. |
| Reviewed | Review findings with severity and disposition. | No unresolved Blocker or Major findings remain; every accepted risk is documented. |
| Integrated | One intentional commit on `main`. | The staged diff contains only the reviewed change and the commit is independently identifiable. |
| Released | User-facing notes, migration details, and smoke-test evidence when applicable. | The delivered behavior and its operational limits are documented. |

## Stage 1: Propose the change

Before editing files, write a short change brief in the issue, task notes, or
working document. It must answer:

- What problem or user-visible behavior is being changed?
- What is explicitly out of scope?
- What are the acceptance criteria and failure conditions?
- Which modules, public interfaces, ABI boundaries, or resource-accounting
  rules may be affected?
- What is the risk of regression and how can the change be rolled back?

Classify the change before implementation:

| Class | Minimum plan |
| --- | --- |
| Documentation-only | Identify the affected rule or user workflow and verify links and wording. |
| Internal code change | Identify ownership, error paths, concurrency behavior, and tests. |
| Behavior or API change | Define compatibility, rejection behavior, regression tests, and documentation updates. |
| Module or dependency change | Read [ARCHITECTURE.md](ARCHITECTURE.md), relevant ADRs, and [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) before editing. |
| CUDA/interceptor change | Define the intercepted symbol paths, ABI signatures, reentrancy behavior, quota effects, and GPU/no-GPU test plan. |

Do not begin implementation when the acceptance criteria or ownership model is
still ambiguous. Resolve the ambiguity or mark the work `Blocked`.

## Stage 2: Design before implementation

Read the project rules relevant to the change:

- [CODING_GUIDELINES.md](CODING_GUIDELINES.md) for source-level behavior,
  ownership, logging, assertions, and concurrency.
- [TESTING.md](TESTING.md) for the required test level and failure-path
  coverage.
- [ARCHITECTURE.md](ARCHITECTURE.md) and relevant ADRs for module boundaries
  and dependency direction.
- [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) before changing a
  dependency.
- [CUDA_API_COVERAGE.md](CUDA_API_COVERAGE.md) before changing interceptor
  coverage or compatibility claims.

The design must identify the affected module and its contract. For a new
cross-module interface, a changed dependency direction, or a changed accepted
architecture decision, add or update an ADR before implementation. The design
must also state:

- input validation and external error mapping;
- ownership and cleanup on success, rejection, cancellation, and partial
  initialization;
- synchronization, lock ordering, and reentrancy constraints;
- logging and assertion behavior;
- the tests that will demonstrate the acceptance criteria.

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
- Keep diagnostics actionable and safe. Never leave temporary tracing,
  credentials, machine-specific paths, or generated build output in the source
  tree.

## Stage 4: Verify with evidence

Run the smallest applicable checks while developing, then run the full relevant
set before review. Use the exact commands below and retain the result in the
change notes when the change is risky or hardware-dependent.

| Change | Required evidence |
| --- | --- |
| Documentation-only | `git diff --check`; inspect every changed link and rendered section. |
| Core or control behavior | `debug` build, CTest, format check, and `asan-ubsan` for ownership, lifetime, or concurrency changes. |
| CMake, build, or dependency change | Affected preset build and CTest; `./scripts/check.sh` when available. |
| CUDA interceptor change | `cuda-lint` or equivalent clang-tidy coverage, no-GPU interceptor tests, and `cuda-gpu` tests when a compatible GPU is available. |
| Public behavior or failure-path change | A regression test plus success, rejection, cleanup, and rollback assertions that apply to the behavior. |

The standard commands are:

```sh
./scripts/check.sh
cmake --build --preset debug --target format-check
ctest --preset debug --output-on-failure
```

For sanitizer or CUDA work, also run the applicable commands in
[TESTING.md](TESTING.md). The CUDA GPU suite is optional only when the required
Toolkit, driver, or hardware is unavailable; report the skipped check and the
no-GPU alternative. Verify that the editor database was refreshed when the
build configuration changed:

```sh
readlink compile_commands.json
```

The link should point to the most recently built preset required by the change.
Do not commit the generated database or build directories.

## Stage 5: Review protocol

Review the diff in two passes. First verify correctness and safety; then verify
maintainability and project consistency. The author performs this review even
when no second reviewer is available.

### Correctness pass

- Does the implementation satisfy every acceptance criterion and preserve
  behavior outside the stated scope?
- Are all success, rejection, timeout, cancellation, partial-initialization,
  and cleanup paths handled?
- Are ownership, lifetime, arithmetic overflow/underflow, and error mapping
  correct?
- Are shared-state synchronization, lock ordering, and thread safety clear?
- For an interceptor, are ABI signatures, symbol lookup paths, initialization,
  reentrancy guards, and fallback behavior correct?
- Do tests exercise observable behavior and important failure paths rather than
  only implementation details?

### Maintainability pass

- Does the change preserve module responsibilities and dependency direction?
- Are names, logging levels, assertions, comments, and public documentation
  consistent with the project rules?
- Is the change small enough to understand and revert?
- Are build presets, lint configuration, test registration, and generated-file
  handling correct?
- Do the documentation and API coverage tables describe the actual behavior,
  including limitations and planned work?

### Finding severity

Use one of these labels for every finding:

| Severity | Meaning | Required action |
| --- | --- | --- |
| `Blocker` | Build failure, crash, security issue, data corruption, undefined behavior, broken ABI, or incorrect quota/accounting result. | Must fix before integration. |
| `Major` | Required behavior or test coverage is missing, a race/leak is likely, or documentation materially contradicts the implementation. | Must fix before integration. |
| `Minor` | Local correctness risk or maintainability problem that does not invalidate the change. | Fix before integration when practical; otherwise record a follow-up. |
| `Nit` | Wording or style preference with no behavior impact. | Optional; do not block integration. |

Every finding should state the location, the risk, and a concrete resolution.
Do not mark a finding resolved without rechecking the changed code and tests.

### Review comment rules

- Prefix a blocking comment with its severity, for example
  `[Blocker]` or `[Major]`.
- Report one independent issue per comment and reference the affected symbol,
  path, or behavior.
- Explain why the issue matters and, when useful, suggest a concrete fix or a
  test that would demonstrate the fix.
- Ask for clarification when intent is unclear; do not turn a preference into
  a blocking finding.
- Let automated tools own mechanical style findings. Do not spend review time
  debating a rule that `.clang-format`, clang-tidy, or the compiler can decide.
- The author must either fix the finding, provide a documented rationale, or
  create a follow-up task before marking it resolved.
- After a fix, re-run the affected checks and review the complete diff again;
  do not review only the changed line in isolation.

Use this compact record for a significant change or a risky self-review:

```text
Change:
Scope and acceptance criteria:
Risk and rollback:

Checks:
- [ ] debug build and CTest
- [ ] format check
- [ ] clang-tidy / CUDA lint (when applicable)
- [ ] ASan/UBSan (when applicable)
- [ ] GPU tests (when available and applicable)

Findings:
- [ ] Blocker: none / resolved
- [ ] Major: none / resolved
- [ ] Minor and Nit: resolved or recorded as follow-ups

Review decision: approved / blocked / deferred
Notes and follow-ups:
```

## Stage 6: Integrate on the single `main` branch

Glimmer currently has one shared branch, so the commit is the durable change
boundary. Follow [GIT_GUIDELINES.md](GIT_GUIDELINES.md):

- inspect `git status`, the unstaged diff, and the staged diff;
- stage only files belonging to the reviewed logical change;
- keep implementation, tests, build wiring, and required documentation
  together, rather than creating one commit per file;
- use `[<type>]: <short imperative summary>` in English;
- do not amend, reset, force-push, or rewrite shared history;
- confirm the final commit contains the intended files and checks.

When working alone, use the review checklist above as a mandatory self-review
record. A second person or external review is encouraged for risky changes, but
it does not replace the verification gates.

## Definition of Done

A change is ready to integrate only when all applicable statements are true:

- the scope and acceptance criteria are met;
- architecture, dependency, API-coverage, and user documentation are updated;
- tests cover the new behavior and relevant failure/rollback paths;
- required builds, tests, format checks, lint, and sanitizers pass;
- unavailable hardware or tools are explicitly reported with an alternative
  check;
- the diff has no unresolved `Blocker` or `Major` findings;
- no temporary logs, generated artifacts, credentials, or machine-specific
  files are included;
- `compile_commands.json` points to the intended last-built configuration;
- the working tree and staged diff contain only the intended logical change;
- the commit is made only after explicit authorization.

If any statement is false, keep the change `Implemented`, `Verified`, or
`Blocked` as appropriate instead of presenting it as integrated.

## Exceptions and follow-ups

If a rule cannot be followed, record the reason, affected risk, alternative
verification, owner, and follow-up issue or task. An exception expires when the
change is integrated unless it is intentionally converted into a documented
project rule or ADR.
