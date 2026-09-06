# 0037: Diagnose launch overhead with warmed CUDA timelines

Status: Accepted

## Context

The optimized solo baselines in ADR 0036 show substantial local and remote
admission overhead. Host API duration alone cannot distinguish GPU execution
from submission gaps, event polling, or control transport. The completion
monitor's condition-variable predicate is immediately satisfied while events
are pending; its one-millisecond idle wait is not a polling interval.

## Decision and change brief

Add an opt-in CUDA profiler range around the existing PyTorch example's warmed
measurement loop. Keep model mathematics, synchronization, and ordinary
benchmark defaults unchanged. A separate diagnostic runner reuses the baseline
mode configuration and remote-service lifecycle for short, solo captures.
Nsight Systems collects CUDA activities and all CUDA API calls without CPU
sampling. Its report and SQLite export belong to a fresh example output
directory, separate from untraced performance measurements.

Analyze API counts/durations and the union of recorded GPU execution intervals.
Validate that actual kernels were captured. Label gaps as gaps in this captured
process's activities, not device-wide idle time or SM utilization. API durations
can overlap across threads and nested Runtime/Driver calls; they are not an
additive latency decomposition. Profiler overhead, especially on short event
queries, prevents treating diagnostic throughput as a benchmark result.
Profiler injection can also affect symbol routing. Record extended
`LaunchKernelEx` calls as a separate coverage-audit signal: at the time of this
decision they had no explicit Glimmer wrappers, and a profiled trace does not establish the
unprofiled application's admission coverage.

## Non-goals, risk, and rollback

Do not change scheduling policy, admission, event polling, or runtime APIs in
this step. Do not promise kernel preemption or infer a causal speedup. Nsight
Systems is an optional, user-installed NVIDIA development tool, not a build or
runtime dependency. Do not fetch, vendor, or link it. Record its actual version;
reject unavailable/incompatible trace data instead of silently falling back to
host-only timing. It requires a supported host/Driver and can perturb execution.

The disabled profiler range makes no profiler calls. Start only after warmup,
stop before validation/output, and stop on exceptions after a successful start.
Rollback removes only example diagnostics, tests, and this decision; the
interceptor and scheduler remain unchanged.

## Acceptance and verification

- Standard-library tests cover capture cleanup, disabled behavior, command
  construction, trace schema validation, interval union/gaps, overlapping API
  durations, missing GPU data, and output/error handling.
- Exercise actual Nsight capture with native and quota/local/remote PyTorch
  modes on the available GPU; retain logs, commands, build fingerprints, quota
  checks, and drained remote-admission evidence.
- Do not run builds, tests, or other Glimmer experiments during capture. Use
  the untraced runner for subsequent performance comparisons.

The unprofiled routing check also exposed an existing smoke-runner defect:
enabled trace assignments could precede a later `env -u` option, making `env`
try to execute `-u`. Keep all unset options before assignments and test all
trace/timing flag combinations in native, observe, and enforce modes without
requiring CUDA. This is a diagnostic-runner fix, not a CUDA behavior change.

## Follow-up gates

Initial real-framework diagnostics exposed dense event polling, extended
launches without explicit wrappers, a remote terminal-report failure, and
profiler completeness warnings. These are follow-up findings, not fixed
runtime behavior or proof of a performance gain. Audit the exact extended
launch and completion/teardown paths before tuning policies. Treat warning-
bearing timelines as diagnostic observations; use untraced experiments for
subsequent performance acceptance.

[ADR 0038](0038-extended-launch-and-completion-correctness.md) records the
subsequent wrapper and completion-race fixes, and the deferred polling
optimization, with unprofiled checks.
The profiler completeness warning remains an explicit diagnostic limitation.

## Reference

[NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
documents CUDA profiler capture ranges and API tracing overhead.
