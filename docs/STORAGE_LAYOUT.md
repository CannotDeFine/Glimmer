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
