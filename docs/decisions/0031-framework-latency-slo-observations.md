# 0031: Measure framework latency targets without claiming hard deadlines

Status: Accepted

## Context

The representative target workload is a latency-sensitive inference process
co-located with throughput-oriented training. The PyTorch example already
reports latency percentiles, but it does not record an explicit target or make
target misses easy to compare across repeated runs. The scheduler can order
queued launch leases, but CUDA Driver and Runtime APIs do not provide kernel
preemption or a hard deadline guarantee.

## Decision

Add an optional `--latency-target-ms` measurement setting to the framework
workload. When it is non-zero, each measured iteration records whether its
wall-clock step latency met the target. This replaces the original CUDA-event
metric as specified in [ADR 0035](0035-common-window-framework-measurements.md).
The workload and co-location runner report
the target, miss count, and miss ratio in both status metadata and summaries.
The runner may enforce an operator-selected maximum miss ratio with
`--max-inference-target-miss-ratio`; the default is diagnostic-only and never
changes scheduling behavior.

The priority matrix also records the control service's aggregate queue-wait and
service-time metrics. A standard-library-only analysis helper aggregates repeat
runs while preserving the distinction between framework tensor batch size and
scheduler launch batch size.

## Consequences

Runs can state an observable inference SLO and fail explicitly when the chosen
miss-ratio bound is exceeded. Percentiles and queue metrics make scheduler
overhead visible instead of conflating it with model compute. The measurements
are workload- and environment-specific; they are not a universal latency
guarantee. A running CUDA kernel may still occupy a slot until its completion
boundary, and a large training lease batch can delay a later inference lease.

## Verification

- The PyTorch workload validates target argument handling and writes target
  fields alongside its per-iteration CSV measurements.
- The synchronized runner validates and reports target misses and rejects a
  configured miss-ratio bound when it is exceeded.
- The priority matrix carries scheduler queue/service aggregates into its CSV,
  and the analysis helper rejects malformed or incomplete input.
- Core scheduler tests verify that a high-priority inference task is dispatched
  ahead of a queued training backlog and that the backlog still drains.
