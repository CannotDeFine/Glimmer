# 0002: Implement a simulated backend before a CUDA backend

Status: Accepted

## Context

Scheduler policy, quota accounting, and failure handling must be deterministic
and testable without a GPU. Coupling them to CUDA hardware would make early
development slow and difficult to validate in CI.

## Decision

Define a GPU backend contract and implement a deterministic simulated backend
before integrating CUDA Runtime, Driver API, or NVML calls.

## Consequences

The scheduler core can be tested on any supported Linux development machine.
The CUDA backend must conform to the established contract rather than define
scheduler semantics itself.
