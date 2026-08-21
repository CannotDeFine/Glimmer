# 0013: Define a versioned task-lease wire format

Status: Accepted

## Context

The control layer now has an in-process transactional admission service, but
an external launcher or control service needs a stable boundary for submitting
logical work and observing its lease state. A protocol that carries CUDA
handles, device pointers, or kernel argument addresses would be invalid: those
objects belong to one process and one CUDA context.

## Decision

Add a transport-neutral, line-oriented codec in
`glimmer/control/task_protocol.h`. The versioned wire format is:

```text
GLIMMER_TASK_V1 SUBMIT <tenant> <memory_bytes> <weight> <work_units> [<priority>]
GLIMMER_TASK_V1 CANCEL <task_id>
GLIMMER_TASK_V1 QUERY <task_id>
GLIMMER_TASK_V1 CLAIM
GLIMMER_TASK_V1 HEARTBEAT <task_id>
GLIMMER_TASK_V1 COMPLETE <task_id>
GLIMMER_TASK_V1 FAIL <task_id>
GLIMMER_TASK_V1 STATS

GLIMMER_TASK_V1 OK <task_id>
GLIMMER_TASK_V1 EMPTY
GLIMMER_TASK_V1 LEASE <task_id> <tenant> <memory_bytes> <weight> <work_units>
GLIMMER_TASK_V1 STATE <task_id> <QUEUED|RUNNING|COMPLETED|CANCELLED|FAILED>
GLIMMER_TASK_V1 STATS <total_tasks> <queued_tasks> <running_tasks> <completed_tasks> <cancelled_tasks> <failed_tasks> <quota_limit_bytes> <quota_reserved_bytes> <quota_allocated_bytes> <max_running_tasks> <max_queued_tasks>
GLIMMER_TASK_V1 ERROR <INVALID_REQUEST|QUOTA_EXCEEDED|QUEUE_FULL|UNKNOWN_TASK|INTERNAL_ERROR|UNSUPPORTED_VERSION>
```

Requests and responses are bounded to 256 bytes, use decimal unsigned
integers, reject zero and overflow for required positive fields, and reject
unknown fields. Priority is an optional unsigned 32-bit value; it defaults to
zero and may therefore be explicitly set to zero. Tenant IDs are
non-empty tokens of at most 96 ASCII bytes containing only letters, digits,
`.`, `_`, or `-`. The codec accepts an optional line terminator and emits one
newline for canonical framing.

The codec does not open sockets, authenticate peers, persist leases, or own
CUDA resources. The Unix-domain-socket adapter authenticates the caller and
applies the admission service, while process-local task resources remain alive
until a terminal state is reported. CUDA descriptors and pointers remain inside
the execution process and are never serialized.

`STATS` is a read-only diagnostic snapshot. It reports scheduler task-state
counts, quota bytes, and configured running/queued limits. It is not a durable
metrics export format and does not expose tenant payloads or CUDA handles.

`CANCEL` applies only while a task is queued. A running task is controlled by
the backend's completion or failure event; the scheduler does not claim to
preempt an arbitrary CUDA kernel.

## Consequences

The protocol can be tested without CUDA or a transport implementation and can
evolve through an explicit version token. Malformed, oversized, unsupported,
and overflowing input is rejected before it reaches scheduling code. A future
transport remains responsible for framing, peer identity, backpressure,
timeouts, reconnect behavior, and recovery.
