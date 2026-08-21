# 0021: Admit transparent CUDA launches with a process-local gate

Status: Accepted

## Context

The explicit control service can schedule logical tasks, but an ordinary CUDA
application does not submit those tasks to Glimmer. The preload interceptor is
therefore the only transparent boundary at which a normal Driver or Runtime
kernel launch can be delayed before it reaches the NVIDIA Driver.

The interceptor cannot safely copy arbitrary kernel arguments for later
execution, preempt a kernel that is already running, or infer a trustworthy
logical task identity from CUDA pointers and launch arguments. A first
transparent implementation must keep the CUDA ABI unchanged and release
admission from a completion boundary that the Driver can observe.

## Decision

In `GLIMMER_SCHEDULER_MODE=enforce`, the covered Driver and Runtime launch
wrappers use a process-local `control::LaunchGate`:

- the caller blocks before forwarding while the configured concurrent-launch
  capacity is full;
- the gate uses the existing `core::Scheduler` contract, with a synthetic
  one-byte task reservation so policy and terminal transitions remain outside
  the interceptor;
- after a successful real launch, the interceptor records a non-timing CUDA
  Driver event on the same stream;
- a monitor thread queries the event and completes the gate lease when the
  event reaches the completed state;
- a real launch error, event creation/recording failure, or event destruction
  failure fails the lease and releases the slot; if the event API is
  unavailable, enforce mode rejects the launch with `CUDA_ERROR_NOT_SUPPORTED`;
- `observe` and `off` preserve forwarding behavior, with observe retaining its
  existing sampled diagnostics.

`GLIMMER_MAX_CONCURRENT_KERNELS` defaults to one. The gate accepts the same
`weighted_rr`, `drr`, `fifo`, and `priority` policy names as the explicit
control service, plus a process-local tenant identifier, optional weight, and
optional unsigned priority from trusted launcher configuration. Priority is
used only when the `priority` policy is selected; larger values run first and
ties retain submission order. The state is reset after `fork()` so a child
does not inherit a monitor thread it cannot join; the preload state is
intentionally kept alive until process exit.

This is a process-local admission mechanism, not a replacement for the
multi-process control-plane lease path. It does not preempt running kernels,
provide cross-process fairness, or capture CUDA graph and cooperative-launch
state. Those capabilities require a future shared task identity and backend
contract.

## Consequences

Normal CUDA applications can opt into bounded in-flight kernel admission
without linking a Glimmer SDK or changing launch arguments. Completion is tied
to the caller's stream, so a long-running kernel can still occupy a slot until
that work completes. Requiring Driver event support and failing closed after a
tracking error avoids silently claiming enforcement while losing the release
boundary.

The control contract remains CUDA-independent, and fake Driver tests can cover
admission, event completion, failure, and cleanup without a GPU. The monitor
adds one process-local thread and one Driver event per admitted launch; future
work may replace polling with a shared control-plane completion interface or a
backend-specific event mechanism after measuring the overhead.
