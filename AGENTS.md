# Agent Instructions

## Git commits

Do not create a commit unless the user explicitly requests one.

Before creating a commit:

1. Read [docs/GIT_GUIDELINES.md](docs/GIT_GUIDELINES.md).
2. Inspect `git status` and the staged and unstaged diffs.
3. Stage only files that belong to the requested logical change. Do not stage
   unrelated user changes.
4. Run the applicable build, test, and formatting checks described in the Git
   guidelines. Report any check that cannot be run and why.
5. Use the required commit-message format:

   ```text
   [<type>]: <short imperative summary>
   ```

6. Confirm the resulting commit contains the intended files and report its
   hash and summary to the user.

Never amend, reset, force-push, or otherwise rewrite shared history unless the
user explicitly asks for that exact operation.

## Repository conventions

- Read and follow [docs/DEVELOPMENT_LIFECYCLE.md](docs/DEVELOPMENT_LIFECYCLE.md)
  before starting a change and complete its applicable review gates before
  integration.
- Read and follow [docs/CODING_GUIDELINES.md](docs/CODING_GUIDELINES.md) before
  implementing or modifying project code.
- Read and follow [docs/TESTING.md](docs/TESTING.md) when changing behavior or
  correcting a defect.
- Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before adding a module or
  changing module responsibilities or dependencies. Review relevant accepted
  design constraints before changing them.
- Read [docs/DEPENDENCY_MANAGEMENT.md](docs/DEPENDENCY_MANAGEMENT.md) before
  adding, updating, or removing a dependency.
- Keep source code, documentation, comments, and commit messages in English.
- Target Linux only.
- Do not modify files under `3rdparty/` unless the user explicitly requests a
  dependency change.

## Documentation

- Keep the root README and public guides focused on implemented Glimmer
  capabilities, usage and limitations. Preserve third-party attribution.

## Independent direction review

- For multi-step execution implementation and application-validation work, use
  one read-only direction-review agent at milestone planning and evidence
  review. Keep implementation ownership with the primary agent; do not expand
  into unrelated completeness work or larger teams without a task-specific
  reason and authorization.
- Direction approval does not replace technical review or authorize a commit,
  relaxed acceptance criteria, or external changes.

## Build storage hygiene

- Check allocated `build/` usage and available disk space before building; keep
  `build/` below 10,000,000,000 bytes. Follow the development lifecycle's
  applicable storage guidance when creating builds and finishing experiments.
- Reuse compatible working builds; fresh evidence does not require a full
  duplicate build. Preserve immutable qualified baselines.
- Clean verified, idle, reproducible temporary intermediates at task handoff.
  Preserve current working builds, reference tensors, source snapshots and
  unique evidence; request a retention decision for ambiguous artifacts.
- Record exact removal manifests and report reclaimed space. Do not claim that
  deleted historical binaries or objects can still satisfy byte-bound audits.

## Full-device experiment qualification

- Treat full-device SM participation as a prerequisite for formal comparisons.
  On the current V100 require evidence for all 80 SMs in every compared path.
- Set benchmark worker geometry explicitly; never inherit the interceptor's
  small correctness-test worker default. Worker counts and occupancy capacity
  alone are not proof of useful compute coverage or sustained saturation.
- Validate native and replacement paths separately with matched operations,
  GPU/runtime/binary/input identities, and fresh complete correctness checks.
- Keep counter collection separate from ordinary timing and priority evidence.
  Missing or partial coverage must fail qualification. Explicit diagnostic-only
  runs may investigate it but must not be presented as full-device comparisons.
