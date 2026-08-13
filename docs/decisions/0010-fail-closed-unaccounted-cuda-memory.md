# 0010: Fail closed for CUDA memory outside the allocation ledger

Status: Accepted

## Context

External-memory mappings, CUDA arrays, mipmapped arrays, sparse/deferred array
mapping, graphics interop mapping, and CUDA Graph memory nodes consume or expose
device-backed memory through lifetimes that are not represented by the ordinary
pointer allocation ledger. In particular, graph memory can be created during
graph construction and released by graph launch, auto-free, update, or process
teardown. External handles, arrays, tile-pool mappings, and graphics resources
also do not provide a uniform authenticated byte size at the interception
boundary.

Charging an estimate would risk both false rejection and under-accounting.
Forwarding these paths while claiming quota isolation would allow workloads to
consume memory outside the configured limit.

## Decision

When quota mode is enabled, Glimmer rejects external-memory imports/mappings,
CUDA array and mipmapped-array creation, sparse/deferred array mapping, graphics
resource mapping and mapped resource acquisition, and CUDA Graph
memory-allocation nodes with the CUDA `NotSupported` result. Destruction and
cleanup calls are forwarded. When quota mode is disabled, all covered calls
delegate unchanged.

The wrappers, symbol registry, fake Driver/Runtime tests, and coverage matrix
make this limitation explicit. Future support must first define exact byte
size, ownership, lifetime, rollback, and process-exit accounting semantics.

## Consequences

The interceptor cannot silently bypass a quota through these APIs, at the cost
of rejecting applications that rely on them while quota enforcement is active.
This is a compatibility limitation, not a claim that these APIs are fully
accounted. The policy can be relaxed only after a tested accounting design is
available.
