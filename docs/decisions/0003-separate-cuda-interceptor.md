# 0003: Keep the CUDA API interceptor separate from the scheduler core

Status: Accepted

## Context

CUDA API interception uses dynamic-linker behavior, process-local handles, and
strict reentrancy constraints. These concerns differ from scheduling policy and
resource accounting.

## Decision

Implement any preloadable CUDA interceptor as a separate shared-library module.
It will use a narrow control contract to communicate with the scheduler and
will not embed scheduler policy.

## Consequences

The scheduler remains testable without `LD_PRELOAD` or CUDA libraries. The
interceptor must handle symbol resolution, recursion prevention, and
process-local resource semantics explicitly.
