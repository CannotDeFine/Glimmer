# 0028: Batch consecutive transparent launch leases

Status: Accepted

## Context

The transparent scheduler currently performs one control-plane admission and
one completion report for every covered kernel launch. Timing diagnostics show
that Unix-socket transport can dominate short framework kernels. Reusing a
socket connection was tested and rejected because it did not reduce the
per-request protocol work enough and made failure handling less clear.

Throughput-oriented training can tolerate a small scheduling granularity while
latency-sensitive inference needs a decision at each launch. The optimization
must preserve CUDA stream ordering, completion tracking, process isolation, and
fail-closed behavior.

## Decision

Add the opt-in `GLIMMER_SCHEDULER_BATCH_SIZE` setting to the preload
interceptor. Its default is `1`. For a value greater than one, the process
associates the next N covered launches with one scheduler lease; launches from
multiple framework host threads may contribute to that batch. The interceptor
records a separate non-timing CUDA event for every launch. The lease is closed
after the final launch has been forwarded and every acquired call has recorded
its event; the completion tracker releases it only after every event in the
batch reaches a terminal state.

Any launch failure, event-record failure, event-query failure, or event-destroy
failure marks the batch failed. Already-recorded events are still drained
before the terminal `FAIL` transition, preventing an early release while CUDA
work is running. Per-call thread-local state is reset on fork, launch failure,
and tracking failure, while the batch accounting itself is process-local.
Batches never cross processes.

The setting is independent from `GLIMMER_MAX_CONCURRENT_KERNELS`: the latter
controls the number of active leases, while the batch size controls the number
of launches represented by one lease. The priority framework example exposes
separate training and inference launch-batch settings and keeps inference at
one by default.

## Consequences

Remote scheduling can amortize admission and completion transport over several
training launches without changing the control protocol or moving CUDA state
between processes. The tradeoff is coarser priority/fairness decisions: a
high-priority tenant may wait for an admitted lower-priority batch to close.
The default remains fine-grained and existing workloads are behaviorally
unchanged. Batching does not provide kernel preemption and cannot guarantee a
latency bound.

## Verification

- The fake CUDA interceptor suite runs an enforce-mode batch-size-two variant
  and checks that reused-lease timing records are emitted.
- Existing Driver and Runtime launch tests continue to verify event tracking,
  launch failure rollback, and process-local scheduling.
- Framework co-location runs can compare separate training and inference batch
  sizes with the priority runner; timing diagnostics remain disabled for
  performance measurements.
