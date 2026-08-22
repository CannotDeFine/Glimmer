# 0014: Use an authenticated Unix socket for local task control

Status: Accepted

## Context

The task protocol and endpoint now provide a process-local control contract.
External launchers still need to submit metadata and observe task leases
without linking against CUDA or entering the interceptor's hot path. Glimmer
targets Linux only, so a local Unix domain socket provides the required IPC
boundary without adding a network RPC dependency.

## Decision

Add `control::UnixSocketControlServer` as a small transport adapter. It:

- binds a caller-selected filesystem socket path and refuses to remove an
  existing filesystem entry;
- accepts a client connection and emits one response per complete request
  line, allowing sequential request/response pairs on the same connection;
- bounds line reads to the task protocol limit and applies an I/O timeout;
- authenticates the peer with Linux `SO_PEERCRED`, accepting only the
  configured UID (the server's effective UID by default);
- removes only the socket inode created by that server instance during
  shutdown; and
- delegates request semantics to `TaskControlEndpoint` rather than owning
  scheduling, quota, or CUDA resources.

The transport uses one worker thread per accepted client connection. This keeps
an idle client from blocking other tenants while allowing the transparent
launch path to reuse its authenticated connection. Clients keep one
connection per calling thread and never retry a request after a transport
failure because requests may have side effects. A future daemon may add
service supervision, authorization policy, rate limiting, backpressure, and
recovery without changing the wire format.

## Consequences

Local clients get a narrow, authenticated control channel with no new runtime
dependency. CUDA handles, device pointers, and kernel arguments remain inside
the execution process. The current adapter is not a general network service,
does not persist leases, and does not cancel running CUDA work.
