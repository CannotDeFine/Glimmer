# Experiment output directories

Glimmer's source-managed collectors can use the standard-library helper to keep
new run and profiling output separate from reusable CMake build directories.

```python
from scripts.artifact_paths import create_attempt, resolve_artifact_root

root = resolve_artifact_root("my-experiment", category="runs")
evidence = create_attempt(root)
```

Alternatively, allocate a fresh directory for an external tool:

```sh
python3 -B scripts/artifact_paths.py --name my-experiment --category profiles
```

Each call creates a unique `attempt-*` directory and a `run.json` record.
Project-local output must resolve under `artifacts/`; new output under
`build/` is rejected. Explicit dedicated directories outside the project are
supported. Existing build contents are not moved or deleted. Treat this as a
developer workflow guard, not a security boundary against concurrent changes.

## Reusable build directories

For builds inside the repository, the root CMake project accepts the configured
preset slots under `build/` and the additional `build/candidate/` slot. It
rejects numbered or other unlisted directories, in-source and ancestor builds,
symlink aliases, frozen markers, and caches owned by a different source tree.
Canonical external build directories remain supported.

This is a workflow safeguard, not a filesystem lock. CMake may create or update
its cache before the check runs, so do not invoke it on a frozen directory.
The guard does not measure or enforce a build storage budget, remove files, or
make historical build contents disposable.

The `scripts/build.sh` and `scripts/check.sh` entry points inspect selected
build slots before invoking CMake. They reject symlinked or frozen slots,
nonempty slots without a regular cache, and a non-symlink root compilation
database. The check script also refuses to reconfigure a selected slot with GPU
tests enabled. Both scripts stop further steps when allocated `build/` usage is
at or above 10,000,000,000 bytes or cannot be measured. These checks cannot
predict peak usage, interrupt a running build, or protect against concurrent
filesystem changes; they do not delete files. Direct CMake invocations do not
enforce this storage budget.
