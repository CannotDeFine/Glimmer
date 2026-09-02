# 0029: Replace remote claim polling with bounded task waits

Status: Accepted

## Context

The remote transparent launch gate already combines submission and an
uncontended claim in `ACQUIRE`. Under contention, a queued launch previously
polled task-specific `CLAIM` every few milliseconds. That generated repeated
Unix-socket requests for short framework kernels and made control-plane work a
large part of the launch path. Increasing the polling interval would reduce
overhead but add avoidable admission latency.

## Decision

Add a versioned `WAIT <task_id> <timeout_ms>` operation to the Linux control
protocol.

- The endpoint waits for the named task to become dispatchable under the
  configured scheduler policy, or for the bounded timeout to expire.
- The admission service uses a condition variable and a generation counter to
  wake waiters after submit, dispatch, cancellation, completion, failure, or
  lease recovery. If an external executor advances the referenced scheduler
  directly, it calls `notify_scheduler_change()` to publish that progress. It
  does not hold its lease mutex while waiting or while notifying another
  waiter.
- A successful wait returns the same task-specific `LEASE` response as
  `CLAIM`. A timeout returns `EMPTY`, a terminal task returns its `STATE`, and
  an unknown task returns `UNKNOWN_TASK`.
- The transparent client uses bounded wait slices so the server never holds a
  connection indefinitely. A client-side acquire deadline remains authoritative
  and causes the queued task to be cancelled when it expires.
- If the server reports an invalid or unsupported `WAIT` operation, the client
  falls back to the legacy task-specific `CLAIM` polling path. This preserves
  mixed-version compatibility and keeps `SUBMIT`/`CLAIM` available to explicit
  clients.
- Launch timing diagnostics distinguish `wait_requests` from legacy
  `claim_polls`.

The operation changes only the control-plane wait strategy. Scheduler policy,
lease ownership, CUDA event completion, quota accounting, and the no-socket
local path are unchanged. It does not provide kernel preemption or a hard
latency guarantee.

## Consequences

Contended remote launches no longer busy-poll the endpoint and normally use one
blocking request per wait slice. The endpoint may have one worker thread blocked
per waiting connection, so waits have a maximum protocol timeout and the server
continues to isolate clients by connection. Older services continue to work
with higher request overhead through the fallback.

## Verification

- Protocol tests cover canonical wait round trips, zero and oversized timeout
  rejection, and invalid formatting.
- Admission tests cover timeout while the slot is occupied, wake-up after
  completion, and terminal quota release.
- Endpoint tests exercise a queued `WAIT` followed by completion.
- The remote launch-gate process test exercises the real service and records
  the request timing for the bounded wait path when Unix sockets are available.
- Existing debug, sanitizer, lint, fake-interceptor, and CUDA GPU suites remain
  required regression gates for the affected targets.
