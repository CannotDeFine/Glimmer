# 0035: Validate framework scheduling in a common wall-clock window

Status: Accepted

## Context

The fixed-iteration co-location test finishes inference before training. Its
aggregate training throughput therefore includes a long uncontended tail.
CUDA-event elapsed time also omits some host work and cannot replace the time
an application waits from submission through synchronization. Neither a
positive overlap nor a zero miss counter with no configured target proves an
inference SLO benefit.

## Decision

The PyTorch harness records monotonic host timestamps immediately before each
step and after synchronization. `elapsed_ms` and latency-target comparisons use
that interval. CUDA-event time remains a separate `cuda_event_ms` diagnostic.
The result schema is identified as `wall_clock_v2`; older event-based results
must not be compared directly. This updates the metric in [ADR 0031](0031-framework-latency-slo-observations.md).

An optional duration mode publishes an atomic shared start/deadline record
after both workers have warmed up. Both keep submitting steps until the same
deadline and drain their final step. The common analysis window starts at the
later actual start and ends at the shared deadline. Only fully contained steps
contribute to the common-window latency distribution and completed steps per
wall-clock second. Boundary-crossing samples remain in each raw CSV and are
reported as excluded. Validation rejects different GPU UUIDs, failed workload
results, incompatible measurement schemas, early exits, empty samples, or less
than 95 percent common-window coverage of either worker's measured interval.

The example's standard-library validation driver compares native, unreserved
priority, static reservation, and adaptive reservation using the same model,
tensor batch, launch batch, two-slot concurrency, and duration. Detailed launch
and scheduler tracing stays disabled. Repetitions rotate configuration order,
retain per-run results, and report medians and ranges of per-run percentiles
and throughput. An explicit inference latency target is required. Queue-wait
targets and inference step-latency targets remain distinct.

The validation driver also runs native training and inference separately in
every repetition, with the same role-local workload and duration-window
accounting. All six configurations share one manifest and rotated execution
order. Native co-location is compared with solo baselines to quantify sharing
interference; Glimmer co-location is compared with native co-location to
quantify its net effect. These denominators must not be conflated.

Comparison schema `solo_colocation_v1` adds `solo_summary.csv`,
`solo_aggregate.json`, and explicit `*_vs_solo` ratios alongside the existing
`*_vs_native` ratios. It requires complete, unique repetition sets and matching
device identity, framework/runtime versions, duration, and workload settings.
Analysis rejects recorded run failures and missing baselines. Older comparison
directories remain readable but report solo baselines as unavailable. Runtime
and measurement-source hashes detect changes during an experiment; inherited
preload, loader tracing, and Glimmer configuration are removed from benchmark
children before the runner supplies the requested mode.

## Limitations

These are closed-loop MLP steps with one outstanding step per process, not
open-loop request arrivals, serving latency, or a hard deadline guarantee.
Shared process intervals demonstrate concurrent offered work on the same GPU;
they do not prove simultaneous physical kernel execution. Tail percentiles
require adequate samples, and boundary exclusions can bias very short runs.
The harness reports counts rather than treating a few samples as strong tail
evidence. CUDA-event collection and per-step synchronization remain overheads
shared by all configurations. Scheduler counters cover the service lifetime,
including warmup and drain, and must not be labeled common-window GPU times.

## Verification

Standard-library tests verify wall-clock throughput with inter-step gaps,
precision at the SLO threshold, boundary exclusion, common-window coverage,
deadline enforcement, identity/schema checks, and per-run aggregation. They
run through CTest when Python is available without importing PyTorch. Real GPU
validation remains an explicit example command and reports unavailable CUDA
access as a failure, never a CPU fallback.
Solo tests reuse the same measurement validation and verify deadline/drain
handling, role/iteration identity, matched denominators, incomplete or duplicate
baselines, mismatched settings, environment cleanup, and CSV round trips.
