# 0006: Schedule GPU work at explicit task boundaries

Status: Accepted

## Context

Glimmer targets shared GPU execution, but a preload library cannot safely
preempt an arbitrary CUDA kernel that is already running. CUDA streams and
events provide observable submission and completion boundaries, while kernel
runtime and device capabilities vary across workloads. Claiming transparent
kernel-level preemption would therefore make fairness and cancellation
semantics unreliable.

The scheduler also needs to remain deterministic and testable without CUDA.
Embedding queue policy in the interceptor would couple scheduling decisions to
dynamic-linker recursion, process-local CUDA handles, and driver-specific
failure behavior.

## Decision

The first scheduler uses explicit task-boundary scheduling. A task is admitted,
dispatched, completed, cancelled, or failed through a scheduler contract; the
backend reports execution progress and capabilities through a separate
interface. The scheduler owns tenant identity, queue policy, fairness, task
state transitions, and scheduler-side reservations. The backend owns CUDA
streams, events, execution submission, and device-specific errors.

The initial implementation will:

- keep the scheduler core independent of CUDA, NVML, and the dynamic loader;
- implement a deterministic simulated backend before a CUDA backend, as
  required by ADR 0002;
- apply fairness and admission at task submission and completion boundaries;
- make cancellation and failure explicit terminal transitions that release
  scheduler reservations exactly once;
- report backend capabilities instead of assuming kernel preemption, stream
  priorities, or a fixed completion latency;
- treat a running task as non-preemptible unless a future backend explicitly
  provides a safe preemption capability.

The CUDA interceptor remains a resource-observation and quota-enforcement
boundary. It does not call scheduler policy directly or claim that every CUDA
kernel is an independently preemptible task.

## Consequences

The scheduler can be unit-tested with a fake clock and simulated backend on a
machine without a GPU. Queue ordering, weighted fairness, admission, failure,
and cancellation become deterministic contracts before CUDA integration.

Fairness is only guaranteed between observable task boundaries. A long-running
kernel may continue to occupy the device until it completes, so the first
version does not promise latency bounds or kernel-level time slicing. Workloads
that require tighter sharing must submit bounded work units or use a backend
with an explicitly supported preemption mechanism.

The CUDA backend must translate real stream/event completion and driver errors
into the scheduler contract without moving scheduler policy into CUDA-specific
code. Any later decision to support kernel preemption, cooperative workload
control, or a stronger latency guarantee requires a superseding ADR and
backend capability evidence.
