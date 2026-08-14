# 0011: Use an explicit CUDA task backend for scheduled launches

Status: Accepted

## Context

The core scheduler now has a generic execution loop, but a transparent CUDA
preload boundary cannot safely capture arbitrary kernel arguments, retain their
storage, or queue a launch after the application call returns. The existing
interceptor therefore remains an observation and memory-quota boundary rather
than an owner of scheduler policy.

Glimmer still needs a real CUDA execution path for workloads that opt into an
explicit scheduler-controlled task boundary. That path must translate CUDA
launch and completion results into the existing backend contract without
coupling `core` to CUDA headers or the dynamic loader.

## Decision

Add an optional CUDA backend under `src/backend/cuda/`. It will:

- accept an explicit, trusted kernel-launch descriptor associated with a
  scheduler `TaskId`;
- submit the descriptor through an injected CUDA Driver function table;
- create and record a CUDA event after a successful launch;
- report `CUDA_SUCCESS`, `CUDA_ERROR_NOT_READY`, and other Driver results as
  backend completion, pending, or failure states;
- destroy owned events on terminal completion and during backend teardown;
- report cancellation as unsupported because arbitrary running CUDA kernels
  cannot be safely preempted by this backend;
- remain separate from the preload interceptor and from the scheduler core.

The explicit controller must keep kernel argument storage and referenced CUDA
resources alive until the backend reports a terminal state. The transparent
interceptor does not use this controller and keeps its current forwarding and
observation behavior.

## Consequences

The scheduler can drive a real CUDA stream/event execution path without
embedding CUDA types in `core` or adding scheduler policy to the interceptor.
The fake Driver function table makes launch, event, pending, completion, and
failure behavior testable without a GPU.

This backend schedules only explicitly registered task descriptors. It does
not provide transparent kernel queuing, kernel preemption, or independent
task identity for arbitrary CUDA applications. A future control-plane or
launcher integration must define descriptor ownership and admission before
exposing this path to users.
