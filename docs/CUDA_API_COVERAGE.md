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
  -> task-limit admission and aggregate quota admission through the control contract
  -> real CUDA Driver allocation
  -> record successful process-local allocation metadata
  -> update quota-store usage through the control contract
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
Shared mode may additionally set `GLIMMER_TASK_MEMORY_LIMIT_BYTES` to apply a
narrower limit to the CUDA process attached to the preload library. Covered
allocations must pass both the tenant aggregate and task limit. If the task
limit is absent, shared mode retains tenant-only behavior. The process is the
first trustworthy task identity at a transparent CUDA ABI boundary; multiple
logical tasks inside one process are outside this guarantee.
When quota mode is enabled, Glimmer lazily queries each used CUDA device and
uses `min(configured_quota, physical_total_memory)` as the effective capacity.
Every allocation checks the current physical free-memory query before calling
the real allocator whenever the current CUDA context identifies the requested
device. A configured value larger than the device is therefore never exposed
through the virtualized CUDA or NVML view.
The optional CUDA GPU preset includes a two-process Runtime test that verifies
one shared tenant cannot exceed its combined quota and that the quota is
restored after the holder releases its allocation.

## Symbol acquisition coverage

General workloads may obtain Driver API entry points in more than one way.
The interceptor must cover each supported path before claiming compatibility.

| Path | Required behavior | Priority | Test |
| --- | --- | --- | --- |
| Direct dynamic import | Export ABI-compatible `cu*` wrapper symbols, including legacy, PTDS, allocation, kernel-launch, stream identity, and query aliases, from the preload library. | M1/M2/M3 | A fixture directly calls versioned, PTDS, legacy allocation/query, and kernel-launch symbols. |
| `dlsym` | Return a supported wrapper when a CUDA Driver allocation, release, VMM handle, kernel launch, stream identity, or query symbol is requested. Delegate all other symbols to the real resolver. | M2/M3 (implemented) | A preload fixture resolves supported PTDS, kernel-launch, and VMM symbols and verifies delegation for an unsupported symbol. |
| `cuGetProcAddress` and `cuGetProcAddress_v2` | Return a supported wrapper for requested CUDA Driver APIs and versions. Delegate unsupported requests unchanged. | M2 (implemented) | A CUDA integration test resolves and calls `cuMemGetInfo` through the versioned API and verifies an unsupported request. |
| CUDA Runtime interception | Wrap synchronous allocation/free (`cudaMalloc`, `cudaMallocManaged`, `cudaMallocPitch`, `cudaMalloc3D`) and stream-ordered allocation/free APIs, including `cudaMallocFromPoolAsync`/`cudaMallocFromPoolAsync_ptsz`, completion boundaries, memory-pool lifecycle/query APIs, and memory-pool import/export APIs. Runtime calls use a reentrancy guard; allocation bytes are independently accounted through the shared registry, while imported pool handles/pointers are rejected when quota mode is enabled. | M2/M3 (implemented) | Fake Runtime and CUDA integration tests plus the real workload matrix for synchronous, stream-ordered, managed, pitched, multi-stream, and memory-pool Runtime calls. |
| CUDA kernel launch observation | Forward `cuLaunchKernel`, `cuLaunchKernel_ptsz`, `cudaLaunchKernel`, `cudaLaunchKernel_ptsz`, `__cudaLaunchKernel`, and `__cudaLaunchKernel_ptsz` with their exact ABIs and preserve the caller's launch arguments. In `observe` mode, emit one allocation-free boundary diagnostic after a successful launch; no queueing, delay, rejection, or kernel preemption is performed. | M4 partial | Dispatch, symbol-registry, and fake Runtime forwarding tests plus real GPU Driver-PTX and Runtime-compiled workloads that allocate memory, launch a kernel, synchronize, and verify the result through `LD_PRELOAD`. |

`cuInit` is also wrapped as an initialization safety boundary. It does not
make an accounting decision; it establishes the Driver-call guard so that
Driver-internal `dlsym` requests are delegated to the real resolver.

The real Driver implementation is resolved from `libcuda.so.1` and cached
outside the public wrapper path. On Linux installations that split the Driver
into a loader and a vendor object (for example, WSL's
`/usr/lib/wsl/lib/libcuda.so.1` plus a vendor `libcuda.so.1.1`), dispatch opens
the loader with eager, local, deep-bound relocations, scans the loaded link map
for the vendor object, and prefers that handle for symbol resolution. If no
secondary object is present, it retains the primary handle. Resolution must
not recursively call an intercepted resolver, and the Driver handles remain
loaded for the process lifetime so late CUDA Runtime teardown cannot call an
unloaded function. Every wrapper must match the exact exported symbol name,
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

The preload library exports CUDA Runtime boundaries for synchronous
(`cudaMalloc`, `cudaMallocManaged`, `cudaMallocPitch`, and `cudaMalloc3D`) and
stream-ordered allocation/free, including `cudaMallocFromPoolAsync` and its
PTDS alias, completion boundaries, memory-pool lifecycle/query operations, and
memory-pool export/import operations. Wrappers resolve the real `libcudart`
functions once, guard reentrant Runtime calls, and reuse the same process-local
allocation ledger and quota contract as Driver API wrappers. Runtime
stream-ordered allocation and release are accounted by a dedicated Runtime
adapter, so a `libcudart` implementation that bypasses Driver allocation
symbols remains covered. Imported pool handles and pointers are rejected in
quota mode because their physical ownership cannot be safely reconstructed at
the wrapper boundary; quota-disabled mode delegates them unchanged. Runtime
completion boundaries use the same context-aware stream identity rules as
Driver wrappers. Because the Runtime ABI does not expose whether a null stream
uses legacy or per-thread default-stream semantics, null Runtime streams are
conservatively isolated by calling thread; device or context synchronization
remains the authoritative completion boundary for legacy-default-stream
workloads.

CUDA IPC exports and closes are transparent, but IPC imports are deliberately
fail-closed while quota mode is enabled. An imported IPC pointer is backed by
physical memory allocated and charged in another process, and the current
control contract has no authenticated record that supplies its device and
allocation size. Accepting the import would create an unaccounted path around
the tenant or task quota. Quota-disabled mode preserves native CUDA behavior.
The same policy applies to Driver and Runtime IPC entry points and is covered
by fake-library tests; trusted cross-process IPC accounting remains a future
control-plane increment. The decision is recorded in
[ADR 0009](decisions/0009-fail-closed-cuda-ipc-imports.md).

The current `cuGetProcAddress` wrappers reject null or empty symbols and null
output pointers before entering the real Driver. Version and flag validation
is still delegated to the real Driver. The v2 wrapper prefers the real v2
resolver and uses a thread-local reentrancy boundary for drivers that call the
legacy resolver during lookup. It falls back to the legacy resolver only when
the v2 entry point is unavailable and translates its not-found result to the v2
status contract. When the Driver returns a valid supported function, the
wrapper replaces only the returned address with the corresponding Glimmer
wrapper; unsupported symbols and other Driver errors remain unchanged.

The runtime symbol registry in `src/interceptor/symbol_registry.cc` is the
single source of truth for supported names, aliases, and wrapper addresses.
Both `dlsym` and `cuGetProcAddress` use this registry; allocation behavior and
accounting remain implemented in the API wrappers themselves.

## Memory operation coverage

| Family | APIs | Accounting event | Planned milestone | Required test |
| --- | --- | --- | --- | --- |
| Ordinary device allocation | `cuMemAlloc_v2`, `cuMemFree_v2` | Charge the successful allocation size; release the recorded size after a successful free. | M1 | Quota admit, quota reject, real allocation failure, double/unknown free handling. |
| Memory information query | `cuMemGetInfo_v2` | Return a virtual total and virtual free value consistent with the configured quota, physical device capacity, current usage, and current physical free memory. | M1 | Query results before allocation, after allocation, and after free. |
| Capacity query | `cuDeviceTotalMem_v2` | Return the tenant-visible capacity clamped to physical device total memory. | M2 (implemented) | Fake and CUDA integration tests observe the effective capacity. |
| Pitched allocation | `cuMemAllocPitch_v2`, `cudaMallocPitch`, `cudaMalloc3D`, `cuMemFree_v2`, `cudaFree` | Charge actual reserved bytes using returned pitch and row count. `cudaMalloc3D` multiplies height and depth with checked arithmetic before admission. | M2 (implemented) | Fake Driver/Runtime and CUDA integration tests verify pitch-based accounting, 3D row-count accounting, rejection, and release. |
| Managed allocation | `cuMemAllocManaged`, `cudaMallocManaged`, `cuMemFree_v2`, `cudaFree` | Charge a successful managed allocation and release it by recorded pointer. | M2 (implemented) | Fake Driver/Runtime and CUDA integration tests verify managed allocation and release. |
| Stream-ordered allocation | `cuMemAllocAsync`, `cuMemAllocAsync_ptsz`, `cuMemAllocFromPoolAsync`, `cuMemAllocFromPoolAsync_ptsz`, `cuMemFreeAsync`, `cuMemFreeAsync_ptsz`, `cuStreamGetDevice[_ptsz]`, `cuStreamGetCtx[_ptsz]`, `cuStreamQuery[_ptsz]`, `cuStreamSynchronize[_ptsz]`, `cuStreamDestroy[_v2]`, `cuCtxSynchronize`, `cudaMallocAsync`, `cudaFreeAsync`, `cudaStreamQuery`, `cudaStreamSynchronize`, `cudaStreamDestroy`, `cudaDeviceSynchronize` | Charge successful allocation submission; retain an enqueued free until an explicit successful completion boundary. Context teardown does not release context-independent allocations. | M3 (implemented) | No-GPU fake Driver and CUDA tests verify Driver/PTDS and Runtime dispatch, deferred release, stream destruction, and context completion. |
| Memory pools | `cuMemAllocFromPoolAsync`, `cuMemPoolTrimTo`, `cuMemPoolSetAttribute`, `cuMemPoolGetAttribute`, `cuMemPoolSetAccess`, `cuMemPoolGetAccess`, `cuMemPoolCreate`, `cuMemPoolDestroy`, `cuDeviceGetMemPool`, `cuDeviceSetMemPool`, `cuDeviceGetDefaultMemPool`, `cuMemGetDefaultMemPool`, `cuMemGetMemPool`, `cuMemSetMemPool`, `cuMemPoolExportToShareableHandle`, `cuMemPoolImportFromShareableHandle`, `cuMemPoolExportPointer`, `cuMemPoolImportPointer` and corresponding Runtime pool APIs | Charge requested stream-ordered allocations. Pool lifecycle, attributes, access, and selection are forwarded without an additional quota charge. Imported pool handles and pointers are rejected in quota mode and delegated when quota mode is disabled, because their physical ownership cannot be reconstructed safely. Pool allocation bytes remain tracked by the existing async allocation path. | M3 (implemented) | Fake Driver/Runtime tests cover lifecycle, attribute/access/query, handle/pointer policy, and deferred allocation completion; GPU tests cover supported pool lifecycle. |
| CUDA VMM | `cuMemAddressReserve`, `cuMemAddressFree`, `cuMemCreate`, `cuMemRelease`, `cuMemMap`, `cuMemUnmap`, `cuMemSetAccess`, `cuMemGetAddressRange_v2`, `cuMemGetAccess`, `cuMemExportToShareableHandle`, `cuMemImportFromShareableHandle`, `cuMemGetAllocationGranularity`, `cuMemGetAllocationPropertiesFromHandle`, `cuMemRetainAllocationHandle` | Charge device-resident physical allocation handles. Retained handles increment a local reference count and release quota only on the final successful `cuMemRelease`. Virtual-address, mapping, access, and query operations are forwarded without a second charge. Imported VMM handles are rejected in quota mode and delegated when quota mode is disabled. Host allocations are delegated because they do not consume the device quota. | M3 (implemented) | Fake Driver tests verify query, retain/release reference counting, import policy, invalid arguments, full address lifecycle, quota admission/rejection, and symbol routing. CUDA integration tests cover query and lifecycle APIs when VMM support is advertised. |
| CUDA IPC memory | `cuIpcGetMemHandle`, `cuIpcOpenMemHandle`, `cuIpcOpenMemHandle_v2`, `cuIpcCloseMemHandle`, `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, `cudaIpcCloseMemHandle` | Export and close are forwarded. IPC imports are rejected with `CUDA_ERROR_NOT_SUPPORTED`/`cudaErrorNotSupported` when quota mode is enabled because the importing process cannot reconstruct the exporting process's physical ownership and charged size. Quota-disabled mode delegates imports unchanged. | M3 partial | Fake Driver/Runtime tests verify ABI aliases, symbol routing, quota-enabled rejection, and quota-disabled passthrough. A trusted cross-process IPC accounting contract is not implemented yet. |
| External memory | `cuImportExternalMemory`, `cuExternalMemoryGetMappedBuffer`, `cuExternalMemoryGetMappedMipmappedArray`, `cuDestroyExternalMemory`, and Runtime equivalents | External objects and mappings are rejected when quota mode is enabled because ownership and physical size are outside the CUDA allocation ledger; destruction is forwarded. Quota-disabled mode delegates. | M3 partial | Fake Driver/Runtime routing, rejection, and passthrough tests. |
| CUDA arrays and mipmapped arrays | `cuArrayCreate[_v2]`, `cuArray3DCreate[_v2]`, `cuMipmappedArrayCreate`, `cuArrayDestroy`, `cuMipmappedArrayDestroy`, `cudaMallocArray`, `cudaMalloc3DArray`, `cudaMallocMipmappedArray`, `cudaFreeArray`, `cudaFreeMipmappedArray` | Array allocation is rejected while quota mode is enabled until a driver-reported memory-requirement accounting path is available; destruction is forwarded. | M3 partial | Fake Driver/Runtime tests verify ABI aliases, rejection, and passthrough. |
| Sparse/deferred CUDA array mapping | `cuMemMapArrayAsync` | Rejected while quota mode is enabled because tile-pool mappings can consume physical memory without a stable size and lifetime at this ABI boundary. Quota-disabled mode delegates unchanged. | M3 partial | Fake Driver routing, `cuGetProcAddress`, quota rejection, and quota-disabled passthrough tests. |
| CUDA graphics interop mappings | `cuGraphicsMapResources`, `cuGraphicsUnmapResources`, `cuGraphicsResourceGetMappedPointer[_v2]`, `cuGraphicsSubResourceGetMappedArray`, `cuGraphicsResourceGetMappedMipmappedArray`, map-flag, and Runtime equivalents | Resource mapping and mapped pointer/array acquisition are rejected while quota mode is enabled because imported graphics ownership and byte size cannot be reconstructed safely. Unmapping, unregistering, and map-flag updates are forwarded. Quota-disabled mode delegates. | M3 partial | Fake Driver/Runtime routing, `cuGetProcAddress` aliases, rejection, and passthrough tests. |
| CUDA Graph memory nodes | `cudaGraphAddMemAllocNode` | Rejected while quota mode is enabled because graph allocation lifetime spans graph construction, launch, auto-free, updates, and process exit beyond the ordinary pointer ledger. Quota-disabled mode delegates. | M3 partial | Fake Runtime routing and quota rejection test. |
| Context lifecycle | `cuCtxGetCurrent`, `cuCtxGetDevice`, `cuCtxDestroy_v2` (plus legacy alias) | Associate process-local metadata with the correct device and clean it up on intercepted context teardown. | M2 partial / M3 | Multiple contexts, primary-context lifecycle, and cleanup after process/context exit. |
| NVML presentation | `nvmlInit`, `nvmlInit_v2`, `nvmlInitWithFlags`, `nvmlShutdown`, `nvmlDeviceGetCount[_v2]`, `nvmlDeviceGetHandleByIndex[_v2]`, `nvmlDeviceGetIndex`, `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetMemoryInfo_v2` | Dynamically resolve NVML, preserve the complete physical structure when quota mode is disabled, and present tenant-visible total/used/free values when quota mode is enabled. Device identity is mapped through NVML index; v2 reserved bytes are set to zero only for the virtualized quota view. | M3 (implemented) | Fake NVML preload tests verify passthrough and virtualized v1/v2 memory views, explicit-handle `dlsym`, and usage changes; GPU tests query real NVML through the preload boundary. |

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
cannot silently under-report usage. Context creation/reset paths that are not
represented by the covered Driver context APIs remain outside the current
guarantee. Cross-process quota aggregation is available only through the
explicit shared-memory control-plane mode described in ADR 0005.

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
4. Runtime `cudaMalloc`, `cudaMallocManaged`, `cudaMallocPitch`, `cudaMalloc3D`, `cudaFree`, and
   `cudaMemGetInfo` share the
   process-local ledger and have an independent Runtime-call reentrancy guard;
   Runtime stream-ordered and memory-pool allocation, release, and completion
   boundaries use an independent adapter backed by the same context-aware
   registry, while imported pool handles/pointers are rejected in quota mode.
5. No-GPU preload tests, including a fake Driver/Runtime fixture, static
   analysis, formatting checks, and compatible CUDA GPU integration tests pass
   with warnings treated as errors. GPU integration remains an environment
   requirement and is reported separately when the host blocks GPU access.
6. Memory-pool lifecycle forwarding, quota-enabled import rejection, VMM query
   forwarding and retain/release reference accounting, NVML passthrough and
   virtualization, the covered stream-ordered Driver/Runtime allocation paths,
   and shared-memory multi-process accounting have their own completion tests.

## M3 VMM increment exit criteria

The VMM increment is complete only when all of the following are true:

1. Device-resident `cuMemCreate` and `cuMemRelease` enforce quota admission
   and exact release accounting, including real-driver failure and cleanup
   failures.
2. `cuMemAddressReserve`, `cuMemAddressFree`, `cuMemMap`, `cuMemUnmap`, and
   `cuMemSetAccess` have ABI-compatible wrappers, dispatch-table entries, and
   `dlsym`/`cuGetProcAddress` routing.
3. Address lifecycle operations are forwarded without charging or releasing
   physical quota a second time; the physical handle remains the accounting
   identity.
4. Fake-Driver tests cover invalid arguments, complete reserve/create/map/
   access/unmap/free/release sequencing, explicit-handle lookup, and both
   procedure-address resolvers. CUDA integration tests exercise the same
   lifecycle when the device advertises VMM support.
5. Host VMM allocations are delegated conservatively; allocation-granularity
   and access-query APIs are forwarded without accounting. Imported VMM and
   memory-pool handles/pointers are rejected in quota mode, delegated when
   quota mode is disabled, and covered by dedicated fake tests alongside NVML
   presentation. CUDA IPC imports follow the same fail-closed policy, while
   IPC exports and closes remain transparent in both quota modes.

## Kernel launch observation increment

The current launch increment proves transparent execution for a real CUDA
Driver workload. `cuLaunchKernel` and its PTDS entry point are dynamically
resolved, exported through the preload library, and forwarded without changing
the launch configuration, argument storage, stream, or return code. The
optional `GLIMMER_SCHEDULER_MODE=observe` setting reports the first successful
launch through the interceptor's allocation-free diagnostic path. Set
`GLIMMER_TRACE_KERNEL_LAUNCHES=1` to report every successful launch with its
API, dimensions, shared-memory size, stream, and process-local sequence number.
Set `GLIMMER_TRACE_MEMORY_INFO=1` to report the virtualized memory view returned
by the CUDA memory-information APIs.
`enforce`
is intentionally not an admission queue: it currently forwards CUDA calls and
reports that enforcement is not implemented. Cooperative and graph launch
families remain separate follow-up coverage. The Runtime compiler-generated
path is covered by the optional CUDA GPU test, while its public and `__cuda`
ABI forwarding wrappers remain transparent and non-enforcing.
