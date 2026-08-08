# 0004: Configure the M1 memory quota per process

Status: Accepted

## Context

The first CUDA Driver API interceptor needs a deterministic quota before a
container-level control plane and shared accounting are available. The setting
must be available inside a preloaded application process without adding an IPC
or service dependency to the first milestone.

## Decision

M1 reads `GLIMMER_MEMORY_LIMIT_BYTES` when the interceptor initializes.

- If the variable is unset, the interceptor forwards the supported Driver APIs
  without applying a quota.
- If the variable contains a valid unsigned byte count, the interceptor enables
  a process-local quota of that size.
- If the variable is set but invalid, allocation wrappers fail closed with
  `CUDA_ERROR_INVALID_VALUE`; release and query wrappers still delegate to the
  real Driver.

The environment variable is an M1 process-level configuration mechanism, not
the future tenant accounting protocol. A later control-plane decision may
replace its quota source while preserving the interceptor's admission contract.

## Consequences

M1 can be deployed and tested with only the preload library and one explicit
environment variable. The quota applies independently to each process, so it
does not yet enforce a shared container or Pod quota. Operators must treat an
invalid configured value as a launch configuration error.
