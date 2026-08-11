# 0007: Preserve application transparency with trusted scheduler configuration

Status: Accepted

## Context

Glimmer must work with ordinary CUDA applications without requiring source
changes, a Glimmer-specific task API, or an application-owned backend. The
preload library is the compatibility boundary, while scheduling policy and
execution backends are internal implementation details.

At the same time, arbitrary CUDA kernel launches cannot be safely delayed after
the API call returns: launch argument storage may belong to the caller's stack,
and an already-running kernel cannot be transparently preempted by a user-space
interceptor. A scheduler that promises transparent kernel time slicing would
therefore claim guarantees it cannot provide.

## Decision

Glimmer preserves application transparency at the CUDA ABI boundary:

- Applications continue to use normal CUDA Driver and Runtime APIs.
- The Backend contract, scheduler task descriptors, and simulated backend are
  internal project interfaces; applications never construct or submit them.
- The interceptor observes covered launch, stream, event, and completion
  boundaries and performs admission decisions before forwarding a real CUDA
  call when enforcement is enabled.
- The first scheduler does not queue arbitrary launch calls after the caller
  returns, and it does not claim kernel-level preemption or cancellation.
- Scheduler configuration is supplied by a trusted launcher or control plane.
  Environment variables are the MVP transport; a future control service may
  replace them without changing the interceptor contract.

The configuration model has three modes:

- `off`: preserve forwarding behavior and do not enforce scheduling policy;
- `observe`: record task-boundary observations and decisions without delaying
  or rejecting application CUDA calls;
- `enforce`: apply the supported admission and quota policy before forwarding
  a call.

The existing memory and tenant settings remain the resource configuration:
`GLIMMER_MEMORY_LIMIT_BYTES`, `GLIMMER_QUOTA_MODE`,
`GLIMMER_QUOTA_TENANT_ID`, and `GLIMMER_QUOTA_DEVICE_ID`. A future scheduler
weight setting must come from the trusted launcher or control plane in shared
mode; an untrusted workload must not be able to raise its own weight.

`GLIMMER_TRACE_KERNEL_LAUNCHES=1` is an opt-in diagnostic setting. It reports
structured launch observations without changing admission, ordering, or CUDA
execution behavior.

`GLIMMER_TRACE_MEMORY_INFO=1` is an opt-in diagnostic setting for the
virtualized memory view. It reports values returned by CUDA memory-information
queries without changing quota decisions or external CUDA results.

Configuration is read once at process initialization. Invalid values are
reported and fail closed for the affected policy rather than silently falling
back to a more permissive mode. Dynamic policy changes require a future
control-plane decision.

## Consequences

Existing CUDA workloads remain source-compatible and can use `LD_PRELOAD`
without linking a Glimmer SDK. The simulated backend can exercise scheduler
ordering and failure handling without pretending to be a CUDA execution path.

The first enforcement guarantee is admission at observable boundaries and
memory accounting, not transparent preemption. Workloads with long-running
kernels may still occupy the device until a completion boundary is observed.
Implementing delayed launch queues, cooperative cancellation, or stronger
latency guarantees requires a new design that safely captures launch state and
documents its CUDA-specific limitations.
