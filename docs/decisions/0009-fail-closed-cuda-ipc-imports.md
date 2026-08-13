# 0009: Fail closed for CUDA IPC imports under quota enforcement

Status: Accepted

## Context

CUDA IPC lets one process export a device allocation and another process open
it through an opaque handle. The importing process receives a pointer, but the
IPC boundary does not provide Glimmer with an authenticated allocation record
containing the exporting process, device, charged size, or tenant identity.

Accepting an IPC import while a quota is enabled would therefore create a
device-memory path that is not admitted or charged by the interceptor. A
process could consume memory owned by another process without reducing its
visible quota, and local release accounting would not identify the original
allocation safely.

## Decision

Glimmer applies the following policy to Driver and Runtime CUDA IPC APIs:

- `cuIpcGetMemHandle`/`cudaIpcGetMemHandle` are forwarded after argument
  validation because exporting an allocation does not create a new physical
  allocation in the importing process;
- `cuIpcCloseMemHandle`/`cudaIpcCloseMemHandle` are forwarded after argument
  validation because they release an imported reference whose ownership is
  outside the local allocation ledger;
- `cuIpcOpenMemHandle` (including `_v2`) and `cudaIpcOpenMemHandle` return
  `CUDA_ERROR_NOT_SUPPORTED` or `cudaErrorNotSupported` when quota mode is
  enabled;
- when quota mode is disabled, all IPC calls delegate to the native CUDA
  implementation so application transparency is preserved.

The policy is enforced before the native import call and is shared by direct
exports, `dlsym`, and the supported Driver procedure-address lookup paths.
Fake Driver and Runtime preload tests cover both rejection and quota-disabled
passthrough behavior.

## Consequences

Quota-enabled processes cannot use CUDA IPC imports to bypass tenant or task
limits. This is a conservative compatibility boundary: applications that rely
on IPC sharing must either run without Glimmer quota enforcement or wait for a
future trusted control-plane contract.

Implementing trusted IPC accounting later will require authenticated
cross-process metadata for the exporting allocation, a lifetime protocol for
imported references, and recovery semantics for exporter failure. Those
requirements are intentionally not approximated in the current process-local
ledger.
