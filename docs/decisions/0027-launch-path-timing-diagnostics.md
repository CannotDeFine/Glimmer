# 0027: Add opt-in launch-path timing diagnostics

Status: Accepted

## Context

Transparent launch enforcement currently combines several costs: scheduler
admission, Unix-socket transport, CUDA forwarding, Driver-event tracking, and
completion reporting. Aggregate scheduler queue and service counters cannot
identify which boundary dominates a real framework workload. Timing must be
safe inside the preload library and must not add allocations, logging
recursion, or scheduling behavior when disabled.

## Decision

Add the opt-in `GLIMMER_TRACE_LAUNCH_TIMINGS=1` diagnostic setting. The
interceptor reports allocation-free records for launch admission and completion
using fixed buffers and the existing low-level diagnostic write path. The
records include:

- admission duration and, for remote leases, summed request transport time;
- whether the launch reused a batched lease;
- request and task-specific claim-poll counts;
- real CUDA launch and Driver-event tracking duration;
- completion transition duration and remote completion transport time.

The control client and launch gate expose caller-owned timing snapshots. They
do not persist metrics, change protocol messages, or change scheduler policy.
Timing is captured only when the setting is enabled. The existing scheduler
queue-wait and service-time aggregates remain the source for policy-level
statistics.

## Consequences

Real workload runs can separate transport, admission, and CUDA overhead before
we choose a persistent socket, blocking lease, admission window, or different
task boundary. The diagnostic stream is intentionally not a stable API or a
time-series exporter. It may perturb measurements slightly because formatting
and writes occur when explicitly enabled; performance comparisons must use the
default-disabled path and repeated runs.

## Verification

- Local launch-gate tests verify that timing snapshots do not report remote
  requests.
- The Linux remote launch-gate process test verifies that request counts and
  transport durations are populated for acquire and completion.
- Interceptor builds and existing fake, sanitizer, lint, and GPU suites remain
  the behavioral regression gates; no fixed timing threshold is asserted.
