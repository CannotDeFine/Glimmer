# CUDA API Coverage Matrix

## Purpose

This document defines the planned CUDA interception coverage required for
Glimmer to enforce a memory quota across general CUDA workloads. It is a
design and test contract, not a statement that every listed API is already
implemented.

The initial integration target is the CUDA Driver API. CUDA Runtime API
workloads have explicit wrappers as well; their allocation and completion
paths must be verified by integration tests rather than inferred from a
particular `libcudart` implementation.

## Scope and guarantee

Glimmer targets Linux CUDA processes launched with its preload library in the
dynamic-loader environment. It provides software quota enforcement at covered
CUDA API boundaries; it does not physically partition GPU memory or provide a
security boundary against a process that can deliberately bypass the preload
library.

For a covered allocation path, the required behavior is:

```text
allocation request
  -> quota admission through the control contract
  -> real CUDA Driver allocation
  -> record successful process-local allocation metadata
  -> update tenant usage through the control contract
```

The inverse applies to a successful release. A failed allocation or release
must not change quota accounting. A rejected allocation returns
`CUDA_ERROR_OUT_OF_MEMORY` without calling the real allocation API.

For stream-ordered Driver allocation, Glimmer charges a successful enqueue
immediately. A successful `cuMemFreeAsync` only marks the allocation pending;
the quota is released after a successful `cuStreamQuery`,
`cuStreamSynchronize`, or `cuCtxSynchronize` confirms completion. If
completion cannot be observed, the charge is retained; if the interceptor
cannot safely preserve the corresponding metadata, it enters its conservative
degraded state rather than under-reporting usage.

When an asynchronous allocation must be rolled back after the real Driver call
has succeeded, Glimmer synchronizes the cleanup stream before cancelling the
reservation. If that completion cannot be proven, the reservation is retained
and the process enters degraded mode rather than releasing quota early.

Destroying a stream detaches pending records from that stream handle. This
prevents a recycled handle from completing an old record; a later successful
stream or context synchronization can finalize the saved device/context
record.
Destroying a CUDA context removes context-bound records for that context after
the real Driver destroy succeeds and releases their charged bytes by device.
Context-independent allocations, including stream-ordered allocations from
CUDA memory pools, remain charged until an explicit completion or release is
observed.

The interceptor owns ABI-compatible wrappers, real-symbol resolution,
reentrancy protection, and process-local handle metadata. It must not embed
scheduler policy or become the durable source of tenant accounting; those
belong to the `control` and `core` contracts defined by the architecture.

Quota accounting has two explicit modes. The default process-local mode is
configured with `GLIMMER_MEMORY_LIMIT_BYTES` and preserves the M1 behavior.
Cross-process accounting requires `GLIMMER_QUOTA_MODE=shared`,
`GLIMMER_QUOTA_TENANT_ID`, and `GLIMMER_MEMORY_LIMIT_BYTES`; the optional
`GLIMMER_QUOTA_DEVICE_ID` selects the initial device record. Shared state and
its recovery guarantees are defined by
[ADR 0005](decisions/0005-shared-quota-control-plane.md).

## Symbol acquisition coverage

General workloads may obtain Driver API entry points in more than one way.
The interceptor must cover each supported path before claiming compatibility.

| Path | Required behavior | Priority | Test |
| --- | --- | --- | --- |
| Direct dynamic import | Export ABI-compatible `cu*` wrapper symbols, including legacy, PTDS, allocation, stream identity, and query aliases, from the preload library. | M1/M2/M3 | A fixture directly calls versioned, PTDS, and legacy allocation/query symbols. |
| `dlsym` | Return a supported wrapper when a CUDA Driver allocation, release, stream identity, or query symbol is requested. Delegate all other symbols to the real resolver. | M2/M3 (implemented) | A preload fixture resolves supported PTDS symbols and verifies delegation for an unsupported symbol. |
| `cuGetProcAddress` and `cuGetProcAddress_v2` | Return a supported wrapper for requested CUDA Driver APIs and versions. Delegate unsupported requests unchanged. | M2 (implemented) | A CUDA integration test resolves and calls `cuMemGetInfo` through the versioned API and verifies an unsupported request. |
| CUDA Runtime interception | Wrap `cudaMalloc`, `cudaFree`, `cudaMemGetInfo`, `cudaMallocAsync`/`cudaMallocAsync_ptsz`, `cudaFreeAsync`/`cudaFreeAsync_ptsz`, `cudaDeviceSynchronize`, `cudaStreamSynchronize`/`cudaStreamSynchronize_ptsz`, `cudaStreamQuery`/`cudaStreamQuery_ptsz`, and `cudaStreamDestroy`. Runtime calls use a reentrancy guard and independently account stream-ordered allocations through the shared registry. | M2/M3 (implemented) | Fake Runtime and CUDA integration tests for synchronous, stream-ordered, and PTDS Runtime calls. |

`cuInit` is also wrapped as an initialization safety boundary. It does not
make an accounting decision; it establishes the Driver-call guard so that
Driver-internal `dlsym` requests are delegated to the real resolver.

The real Driver implementation is resolved from `libcuda.so.1` and cached
outside the public wrapper path. Resolution must not recursively call an
intercepted resolver. Every wrapper must match the exact exported symbol name,
version, calling convention, and function signature of its target CUDA
version.

The current `dlsym` implementation covers `RTLD_DEFAULT` and explicit
`libcuda.so`/`libcudart.so` handles for supported symbols while preserving real
resolution for `RTLD_NEXT`, non-CUDA handles, CUDA Runtime-internal calls, and
Driver-internal calls. This prevents CUDA Driver internals from resolving their
own private symbols back to the interceptor. The interceptor's internal
real-symbol lookup uses the resolved loader function directly, so it cannot
accidentally resolve back to its own wrappers.

Runtime names are exposed through `RTLD_DEFAULT` and explicit `libcudart.so`
handles. An explicit `libcuda.so` handle is restricted to Driver-style `cu*`
names so a lookup on the wrong library cannot return a Runtime wrapper.
Versioned `dlvsym` lookups are delegated unchanged and remain outside the
current interception guarantee.

The preload library exports CUDA Runtime boundaries for `cudaMalloc`,
`cudaFree`, `cudaMemGetInfo`, `cudaMallocAsync`/`cudaMallocAsync_ptsz`,
`cudaFreeAsync`/`cudaFreeAsync_ptsz`, `cudaDeviceSynchronize`,
`cudaStreamSynchronize`/`cudaStreamSynchronize_ptsz`,
`cudaStreamQuery`/`cudaStreamQuery_ptsz`, and `cudaStreamDestroy`. Wrappers
resolve the real `libcudart` functions once,
guard reentrant Runtime calls, and reuse the same process-local allocation
ledger and quota contract as Driver API wrappers. Runtime stream-ordered
allocation and release are accounted by a dedicated Runtime adapter, so a
`libcudart` implementation that bypasses Driver allocation symbols remains
covered. Runtime completion boundaries use the same context-aware stream
identity rules as Driver wrappers. Because the Runtime ABI does not expose
whether a null stream uses legacy or per-thread default-stream semantics, null
Runtime streams are conservatively isolated by calling thread; device or
context synchronization remains the authoritative completion boundary for
legacy-default-stream workloads.

The current `cuGetProcAddress` wrappers reject null or empty symbols and null
output pointers before entering the real Driver. Version and flag validation
is still delegated to the real Driver. The v2 wrapper prefers the real v2
resolver and uses a
thread-local reentrancy boundary for drivers that call the legacy resolver
during lookup. It falls back to the legacy resolver only when the v2 entry point
is unavailable and translates its not-found result to the v2 status contract.
When the Driver returns a valid supported function, the wrapper replaces only
the returned address with the corresponding Glimmer wrapper; unsupported
symbols and other Driver errors remain unchanged.

The runtime symbol registry in `src/interceptor/symbol_registry.cc` is the
single source of truth for supported names, aliases, and wrapper addresses.
Both `dlsym` and `cuGetProcAddress` use this registry;
allocation behavior and
accounting remain implemented in the API wrappers themselves.

## Memory operation coverage

| Family | APIs | Accounting event | Planned milestone | Required test |
| --- | --- | --- | --- | --- |
| Ordinary device allocation | `cuMemAlloc_v2`, `cuMemFree_v2` | Charge the successful allocation size; release the recorded size after a successful free. | M1 | Quota admit, quota reject, real allocation failure, double/unknown free handling. |
| Memory information query | `cuMemGetInfo_v2` | Return a virtual total and virtual free value consistent with the tenant quota and current usage, without claiming more free memory than the physical device reports. | M1 | Query results before allocation, after allocation, and after free. |
| Capacity query | `cuDeviceTotalMem_v2` | Return the tenant-visible capacity. | M2 (implemented) | A CUDA integration test observes the configured quota. |
| Pitched allocation | `cuMemAllocPitch_v2`, `cuMemFree_v2` | Charge actual reserved bytes using returned pitch and requested height. | M2 (implemented) | A CUDA integration test verifies pitch-based accounting and release. |
| Managed allocation | `cuMemAllocManaged`, `cuMemFree_v2` | Charge a successful managed allocation and release it by recorded pointer. | M2 (implemented) | A CUDA integration test verifies managed allocation and release. |
| Stream-ordered allocation | `cuMemAllocAsync`, `cuMemAllocAsync_ptsz`, `cuMemAllocFromPoolAsync`, `cuMemAllocFromPoolAsync_ptsz`, `cuMemFreeAsync`, `cuMemFreeAsync_ptsz`, `cuStreamGetDevice[_ptsz]`, `cuStreamGetCtx[_ptsz]`, `cuStreamQuery[_ptsz]`, `cuStreamSynchronize[_ptsz]`, `cuStreamDestroy[_v2]`, `cuCtxSynchronize`, `cudaMallocAsync`, `cudaFreeAsync`, `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamDestroy`, `cudaDeviceSynchronize` | Charge successful allocation submission; retain an enqueued free until an explicit successful completion boundary. Context teardown does not release context-independent allocations. | M3 (implemented) | No-GPU fake Driver and CUDA tests verify Driver/PTDS and Runtime dispatch, deferred release, stream destruction, and context completion. |
| Memory pools | `cuMemAllocFromPoolAsync` through the default/explicit pool allocation path | Charge requested stream-ordered allocations. Pool creation, reuse policy, release thresholds, and trim remain outside this milestone. | M3 partial | Pool allocation/free path and deferred completion; dedicated pool lifecycle tests remain future work. |
| CUDA VMM | `cuMemCreate`, `cuMemRelease` | Charge physical allocation handles; do not charge address reservation or mapping alone. | M3 | Reserve, create, map, unmap, and release lifecycle. |
| Context lifecycle | `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxDestroy_v2` (plus legacy alias) | Associate process-local metadata with the correct device and clean it up on intercepted context teardown. | M2 partial / M3 | Multiple contexts, primary-context lifecycle, and cleanup after process/context exit. |
| NVML presentation | `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetMemoryInfo_v2` | Present tenant-visible total, used, and free values where NVML compatibility is enabled. | M4 | NVML query agrees with the CUDA query contract. |

`M1` is the first implementation milestone. A workload is not considered
generally supported merely because it succeeds through one M1 path; it must
use only the covered API families and symbol-acquisition paths.

## Allocation record requirements

Each successful allocation record must include at least:

| Field | Reason |
| --- | --- |
| Tenant identity | Selects the quota to charge. |
| Process identity | Supports shared tenant accounting and stale-process recovery. |
| CUDA device and context identity | Device pointers and allocations are process-local and context-sensitive. |
| Allocation kind | Determines which release API and lifetime rules apply. |
| Pointer or allocation handle | Correlates release with allocation. |
| Charged size | Allows exact release accounting. |
| Stream, when applicable | Required for deferred async allocation/free semantics. |

The allocation map is always local metadata. The authoritative tenant usage
and admission decision remain behind the control contract, so process-local
and shared-memory quota stores can be selected without changing wrapper
semantics. The shared-accounting control-plane contract is defined in [ADR
0005](decisions/0005-shared-quota-control-plane.md).

The process-local registry serializes pointer release and pointer reuse and
attaches the CUDA device and context identity to each record. Context-bound
records are removed and released by device when an intercepted context is
destroyed after the real Driver teardown succeeds. Context-independent
stream-ordered records remain charged across context teardown and are removed
only after an explicit completion boundary or release, so stream-handle reuse
cannot silently under-report usage.
Context creation/reset paths that are not represented by the covered Driver
context APIs remain outside the current guarantee. Cross-process quota
aggregation is available only through the explicit shared-memory control-plane
mode described in ADR 0005.

## Concurrency and failure rules

- Admission, real allocation, and accounting commit require a concurrency
  protocol that prevents concurrent requests from permanently exceeding a
  tenant quota.
- The protocol must define how an allocation is rolled back if the real Driver
  call succeeds but accounting commit fails.
- Real CUDA errors are returned unchanged unless Glimmer rejects the request
  for quota; only that rejection returns `CUDA_ERROR_OUT_OF_MEMORY`.
- Unknown, duplicate, or failed releases must never reduce usage.
- Wrapper logging must not allocate GPU memory, invoke CUDA APIs, or depend on
  a lock held across a callback into the CUDA Driver.

## Test strategy

Every covered API requires both a no-GPU unit test and, where CUDA is
available, an integration test.

| Level | Purpose |
| --- | --- |
| Unit | Use a fake Driver dispatch table to test admission, accounting, rollback, and pointer lookup deterministically. |
| Preload integration | Build a small dynamically linked Driver API fixture and verify direct import, `dlsym`, and `cuGetProcAddress` routing. |
| CUDA integration | Run on a CUDA-capable Linux runner to verify real Driver behavior, query virtualization, and representative Runtime workloads. |
| Framework regression | Run a small allocation workload for supported framework versions and record the covered API path. |

No API is marked supported until its ABI wrapper, accounting behavior, and
required tests are present.

## M1 exit criteria

M1 is complete only when all of the following are true:

1. `cuMemAlloc_v2`, `cuMemFree_v2`, and `cuMemGetInfo_v2` have ABI-compatible
   wrappers and use the real Driver dispatch table.
2. A configured quota rejects an otherwise valid allocation before calling the
   real allocation function.
3. Successful allocation and free update usage exactly once, including failure
   and unknown-pointer cases.
4. Virtual memory query results are internally consistent and never exceed
   physical availability.
5. Unit and preload integration tests pass without a physical GPU; CUDA
   integration tests run when a compatible Linux CUDA environment is present.
6. Unsupported API paths are documented as unsupported rather than silently
   advertised as isolated.

## M2 exit criteria

M2 is complete only when all of the following are true:

1. Driver wrappers are registered in one table and the table is used by both
   `dlsym` and `cuGetProcAddress` routing.
2. `cuInit` and Driver-call guards prevent symbol-resolution recursion during
   Driver loading and Driver-internal calls.
3. Capacity, managed-memory, and pitched-allocation paths charge and release
   the correct bytes, including rejection and rollback behavior.
4. Runtime `cudaMalloc`, `cudaFree`, and `cudaMemGetInfo` share the
   process-local ledger and have an independent Runtime-call reentrancy guard;
   Runtime stream-ordered allocation, release, and completion boundaries use
   an independent adapter backed by the same context-aware registry.
5. No-GPU preload tests, including a fake Driver/Runtime fixture, static
   analysis, formatting checks, and compatible CUDA GPU integration tests pass
   with warnings treated as errors. GPU integration remains an environment
   requirement and is reported separately when the host blocks GPU access.
6. Memory-pool management and trim, VMM, and NVML remain explicitly documented
   as future milestones; the covered stream-ordered Driver allocation path and
   shared-memory multi-process accounting have their own completion tests.
