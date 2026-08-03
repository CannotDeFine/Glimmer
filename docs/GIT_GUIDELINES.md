# Git Guidelines

Glimmer currently uses a single shared `main` branch. Commit messages are the
primary way to identify the kind, area, and intent of each change.

## Core rules

- Keep `main` buildable and formatted after every commit.
- Make each commit atomic: it must contain one complete logical change that
  can be reviewed, built, tested, and reverted independently.
- Commit all files required for that change together. A feature commit may
  include implementation, headers, tests, build wiring, and documentation.
- Do not split commits by file. Split only independent logical changes.
- Do not mix unrelated feature work, refactoring, formatting, generated files,
  or dependency updates in the same commit.
- Write commit messages in English.
- Do not rewrite or force-push commits that have already been shared with
  others. Add a corrective commit instead.

Before committing, run the applicable checks:

```sh
cmake --preset debug
cmake --build --preset debug
cmake --build --preset debug --target format-check
ctest --preset debug --output-on-failure
```

## Commit message format

Use a bracketed change type:

```text
[<type>]: <short imperative summary>
```

Keep the summary to 72 characters or fewer, start it with a lowercase verb,
use the imperative mood, and do not end it with a period.

Good examples:

```text
[feat]: add weighted tenant queue
[fix]: prevent recursive symbol resolution
[test]: cover rejected memory reservation
[build]: add CUDA toolkit discovery
[docs]: describe task admission flow
```

Avoid vague messages:

```text
[feat]: update code
[fix]: bug fix
[chore]: changes
[wip]: work in progress
```

## Types

| Type | Use for |
| --- | --- |
| `feat` | A new user-visible capability. |
| `fix` | A correction to existing behavior. |
| `refactor` | Internal restructuring without an intended behavior change. |
| `perf` | A measured performance improvement. |
| `test` | Adding or correcting tests only. |
| `docs` | Documentation-only changes. |
| `build` | CMake, toolchain, packaging, or dependency changes. |
| `ci` | Continuous-integration configuration changes. |
| `style` | Formatting-only changes. |
| `chore` | Maintenance that fits none of the above. |

## Commit body and footers

Add a body when the summary cannot explain the reason for a change. Wrap body
lines at 72 characters and describe the motivation and important constraints,
not a line-by-line implementation narrative.

```text
[fix]: release memory reservation after launch failure

The reservation previously remained charged to a tenant when the backend
rejected a launch, eventually blocking later valid work.

Fixes: #42
```

For an incompatible public behavior change, include a `BREAKING CHANGE:`
footer:

```text
[feat]: replace task priority with scheduling class

BREAKING CHANGE: clients must send a scheduling class instead of an integer
priority.
```

## Change grouping

Group files by behavior, not by file type. For example, a new admission rule
normally belongs in one `[feat]` commit with its scheduler implementation,
public declarations, tests, CMake wiring, and user-facing documentation. Split
it into multiple commits only when each part remains valid and reviewable by
itself.

Use a separate commit for an independent cleanup, dependency update, or
formatting-only change.

Never commit generated build directories, `compile_commands.json`, local CUDA
artifacts, credentials, or machine-specific configuration.
