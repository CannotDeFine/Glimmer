# Development storage layout

[Project home](../README.md) / [Documentation](../docs/README.md) / Storage layout

Keep reusable builds separate from experiment output. A new run needs a fresh
evidence directory, not another full CMake build or copied model/environment.

| Location | Purpose | Reuse and retention |
| --- | --- | --- |
| `build/<preset>/`, `build/candidate/` | Reusable CMake products | Use only the fixed slots below; never rebuild a qualified frozen baseline in place. |
| `artifacts/runs/<name>/attempt-*/` | Ordinary experiment evidence | One fresh attempt per run; preserve failed results and identity bindings. |
| `artifacts/profiles/<name>/attempt-*/` | Instrumented counter/trace evidence | Keep separate from ordinary timing; raw evidence needs a retention decision. |
| `artifacts/maintenance/` | Development checks and cleanup manifests | Keep small receipts, commands and exact removal records. |
| `artifacts/archives/` | Deliberately sealed historical inputs | Preserve unique source/binary capsules; do not copy a full tree by default. |
| `assets/` | Reusable input models, data and reference tensors | Share immutable inputs across runs; never silently prune. |
| `.deps/` | Optional isolated installed SDKs/environments | Version-pin separately; do not mix ABIs, install globally or copy per run. |
| `.cache/` | Reproducible caches | Not authoritative evidence or reference data. |
| `scripts/`, `examples/` | Maintained collectors and experiment entry points | Source belongs here, not in generated evidence. |

These generated locations are ignored by Git, not automatically disposable.
Large per-run outputs are temporary by default: declare what must survive
analysis, then remove superseded payloads under the retention procedure below.
See [the retention rules](DEVELOPMENT_LIFECYCLE.md#generated-artifact-retention-and-cleanup).
This is the policy for new work, not permission to move/delete historical inputs.
Existing SDK/provider paths, models and sealed evidence under `build/` are
retained historical inputs, not permission to write new results there.
Relocation can invalidate loader identities and byte-bound audits. Migrate
them only with consumer checks and an explicit migration plan.

## Fixed build slots

Repository-local builds use one directory per non-hidden configure preset in
`CMakePresets.json`: currently `debug`, `release`, `asan-ubsan`, `lint`,
`cuda-debug`, `cuda-lint`, `cuda-gpu` and `cuda-perf`. Create only the
configurations needed for the task; these are not eight mandatory build copies.
Reuse the same directory for compatible incremental builds.

There is one additional slot, `build/candidate/`, for a temporary root or overlay
configuration. Do not create numbered, dated or task-named build directories,
or add a preset just to retain another run. A new durable preset must represent
a recurring, incompatible compiler, SDK or instrumentation configuration.
Before repurposing a slot, preserve the required evidence and remove only its
verified disposable products. Never reuse ABI-incompatible objects or silently
switch a cache between root and overlay source trees.

The root project and all diagnostic overlays check this policy at configure
time. They reject other repository-local build paths, in-source/ancestor paths,
symlink aliases and directories containing `.glimmer-frozen`, including a
dangling symlink marker. Canonical external directories remain supported for
CI and temporary checks; they follow
the same retention rules and are not an alternative place to accumulate builds.
This check is a workflow safeguard, not a filesystem lock: CMake itself may
create or update its cache before the check runs, including values passed with
`-D`. Do not invoke CMake directly on a frozen directory. The aggregate check
script inspects its selected slots before invoking CMake, rejecting linked
caches and nonempty slots without a regular cache; other programs are not
restricted by this guard. These checks do not prevent concurrent filesystem
changes.

Keep at most one active comparison baseline and one active candidate. A new
baseline is a minimal immutable capsule under `artifacts/archives/`, containing
the required source, binaries, runtime dependencies, configuration, commands
and identity/evidence references, not a copy of the entire build tree. Do not
edit historical inputs in place or overwrite hardlinked files when reusing a
build slot. Retiring a baseline does not authorize deleting its unique evidence.

Historical build directories remain readable, but are not supported for new
configuration. This rule does not certify them as disposable or automatically
migrate them. Preserve active loader paths, imported libraries, reference
tensors and the current compilation database until their consumers have been
explicitly migrated.

Before building, record the slot, expected peak storage and available space.
Keep the allocated size of the repository's `build/` below 10 GB
(10,000,000,000 bytes), including retained historical inputs. From the
repository root, check an existing directory and its filesystem with:

```sh
du -s -B1 build
df -B1 .
```

`du` counts hardlinked files once within its traversal; it is not a measurement
of bytes that deletion would release. If the expected peak exceeds the budget,
do not start the build until enough budget is available under the retention and
cleanup rules below. CMake and the aggregate check script do not enforce this
size limit during a running command. `scripts/build.sh` and `scripts/check.sh`
measure `build/` before and after their successful configure, build and test
steps, and stop further steps when usage is at or above the limit or cannot be
measured.
These checks do not predict peak usage, prevent concurrent changes, interrupt a
command in progress, or apply to direct CMake invocations. They do not delete
anything; estimate peak usage and leave enough space before starting.

At handoff, record actual usage, removed intermediates and why remaining
temporary products are retained. Clean idle reproducible intermediates from
completed temporary work; keep compatible working products for incremental
reuse. Preserve the configuration, commands, logs and source/binary identities
needed to interpret retained results. Do not run a full rebuild solely to check
a storage cleanup. Exact removal manifests and reclaimed bytes belong in
`artifacts/maintenance/`.

Before removal, resolve the exact targets, check for live consumers, and exclude
source, dependencies, archived inputs and protected baselines. Record paths,
types, sizes and file identities appropriate to the cleanup; recheck those
identities immediately before removal and verify protected artifacts afterward.
Metadata can identify an idle reproducible intermediate for cleanup, but does
not prove its bytes are unchanged. Use content hashes when a byte-bound audit or
integrity check requires them, not automatically for every disposable object.
Reject ambiguous targets and never remove the workspace or build root wholesale.
Report reclaimed space and whether removal requires recompilation. Do not claim
a binary-bound audit remains replayable after its required files are removed.

## New evidence directories

Use the standard-library helper (Python 3.9+, no third-party packages) from
source-managed collectors:

```python
from scripts.artifact_paths import create_attempt, resolve_artifact_root

root = resolve_artifact_root("my-experiment", category="runs")
# Validate the experiment's other arguments before creating output.
evidence = create_attempt(root)
```

Or allocate a directory for an external tool:

```sh
python3 -B scripts/artifact_paths.py --name my-experiment --category profiles
```

Resolution/help/import creates nothing. Creation uses an atomic unique
`attempt-*` directory with `run.json` recording the layout and canonical root.
Simple namespace names cannot contain separators. Project-local output must
be specified under and resolve inside `artifacts/`; source/dependency/build
aliases are rejected even when they point outside the project.
Explicit dedicated roots outside the project are supported. The artifact
boundary cannot itself be a symlink, and an internal alias cannot escape it.
This developer policy is not a hostile-concurrent-filesystem security boundary.

The GEMM runner defaults to `artifacts/runs/gemm-comparison/` and its offline
analyzer to `artifacts/runs/gemm-analysis/` when `--output-root` is omitted.
New output under `build/` is rejected, including
when the deprecated `--allow-legacy-build-root` helper option is supplied.
Historical run metadata remains readable. Other collectors that do not use
this helper must follow the same output policy; the helper cannot enforce
arbitrary filesystem writes. No correctness, SM coverage, GPU-idle,
instrumentation or timing gate is relaxed by this change.

## Retain results, not every large output

Before a run, record its peak output budget and classify its outputs as active
inputs, required evidence, or temporary analysis payloads. A successful analysis
must not implicitly turn every tensor dump, replay snapshot, or derived trace
into a permanent archive.

At experiment handoff:

- Keep the compact result: commands, configuration, input/source identities,
  numerical scope and tolerances, measured values, failures, and review outcome.
- Keep the full inputs and outputs still required by the active baseline,
  candidate, correctness oracle, or an unresolved investigation. Preserve unique
  source/runtime capsules and the raw counter evidence supporting active claims.
- After the required analysis and review, remove superseded tensor outputs,
  replay copies, and redundant derived data that have no remaining consumer.
  Retain failed-run diagnostics needed to explain the failure; failure alone
  does not require keeping every large output indefinitely.
- Record exact paths, file identities, retained evidence, and any lost reanalysis
  capability before removal. Check live consumers and recheck identities before
  unlinking only the selected payloads. Ambiguous or unique required evidence
  needs an explicit retention decision, not an age-based deletion rule.
- Report allocated bytes actually released. Moving data into another directory
  on the same filesystem is not cleanup. Shared inodes require separate
  accounting; removing one hardlink may release no payload blocks.

Do not promise that a hash or summary can reproduce a deleted tensor or support
a fresh full-output comparison. Mark retired attempts as unavailable for that
reanalysis, while preserving their original reports unchanged. New comparisons
still require fresh complete correctness and the applicable full-device gates.

## Test runtime libraries

Some execution tests share `lib/test-support/libglimmer_execution_test_runtime.so`
within a build. Keep that library with its matching test executables; it is not
an installed or production ABI. `GLIMMER_SHARE_EXECUTION_TEST_RUNTIME=OFF`
selects static fixture linkage.

Do not share ABI-specific objects or libraries across compiler, Debug/Release,
sanitizer or SDK configurations. Reusing a working build must preserve the
matching runtime libraries, not just the test executable.
