# CUDA API Coverage Matrix

## Purpose

This document defines the planned CUDA interception coverage required for
Glimmer to enforce a memory quota across general CUDA workloads. It is a
design and test contract, not a statement that every listed API is already
implemented.

The initial integration target is the CUDA Driver API. CUDA Runtime API
workloads are expected to reach the Driver API through `libcudart`, but that
path must be verified by integration tests rather than assumed.

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

The interceptor owns ABI-compatible wrappers, real-symbol resolution,
reentrancy protection, and process-local handle metadata. It must not embed
scheduler policy or become the durable source of tenant accounting; those
belong to the `control` and `core` contracts defined by the architecture.

## Symbol acquisition coverage

General workloads may obtain Driver API entry points in more than one way.
The interceptor must cover each supported path before claiming compatibility.

| Path | Required behavior | Priority | Test |
| --- | --- | --- | --- |
| Direct dynamic import | Export ABI-compatible `cu*` wrapper symbols from the preload library. | M1 | A fixture directly calls `cuMemAlloc_v2`. |
| `dlsym` | Return a supported wrapper when a CUDA Driver allocation, release, or query symbol is requested. Delegate all other symbols to the real resolver. | M2 | A fixture resolves and calls `cuMemAlloc_v2` with `dlsym`. |
| `cuGetProcAddress` and `cuGetProcAddress_v2` | Return a supported wrapper for requested CUDA Driver APIs and versions. Delegate unsupported requests unchanged. | M2 | A fixture resolves and calls `cuMemAlloc_v2` through each API. |
| CUDA Runtime forwarding | Verify that representative Runtime API calls reach a covered Driver allocation path, or add the needed Runtime wrapper. | M3 | CUDA integration tests for Runtime allocation and free. |

The real Driver implementation is resolved from `libcuda.so.1` and cached
outside the public wrapper path. Resolution must not recursively call an
intercepted resolver. Every wrapper must match the exact exported symbol name,
version, calling convention, and function signature of its target CUDA
version.

## Memory operation coverage

| Family | APIs | Accounting event | Planned milestone | Required test |
| --- | --- | --- | --- | --- |
| Ordinary device allocation | `cuMemAlloc_v2`, `cuMemFree_v2` | Charge the successful allocation size; release the recorded size after a successful free. | M1 | Quota admit, quota reject, real allocation failure, double/unknown free handling. |
| Memory information query | `cuMemGetInfo_v2` | Return a virtual total and virtual free value consistent with the tenant quota and current usage, without claiming more free memory than the physical device reports. | M1 | Query results before allocation, after allocation, and after free. |
| Capacity query | `cuDeviceTotalMem_v2` | Return the tenant-visible capacity. | M2 | A framework-style capacity query observes the configured quota. |
| Pitched allocation | `cuMemAllocPitch_v2`, `cuMemFree_v2` | Charge actual reserved bytes using returned pitch and requested height. | M2 | Alignment causes a charge different from requested width times height. |
| Managed allocation | `cuMemAllocManaged`, `cuMemFree_v2` | Charge a successful managed allocation and release it by recorded pointer. | M2 | Managed allocation obeys the same quota. |
| Stream-ordered allocation | `cuMemAllocAsync`, `cuMemAllocFromPoolAsync`, `cuMemFreeAsync` | Apply a documented submission and completion policy; do not treat an enqueued free as immediately reusable unless the policy permits it. | M3 | Stream ordering and deferred-free tests. |
| Memory pools | `cuMemPool*` allocation and trim APIs | Account physical pool reservation separately from virtual suballocation where required. | M3 | Pool growth, reuse, trim, and quota behavior. |
| CUDA VMM | `cuMemCreate`, `cuMemRelease` | Charge physical allocation handles; do not charge address reservation or mapping alone. | M3 | Reserve, create, map, unmap, and release lifecycle. |
| Context lifecycle | `cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxRelease_v2`, context destroy APIs | Associate process-local metadata with the correct device and clean it up on context teardown. | M3 | Multiple contexts and cleanup after process/context exit. |
| NVML presentation | `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetMemoryInfo_v2` | Present tenant-visible total, used, and free values where NVML compatibility is enabled. | M4 | NVML query agrees with the CUDA query contract. |

`M1` is the first implementation milestone. A workload is not considered
generally supported merely because it succeeds through one M1 path; it must
use only the covered API families and symbol-acquisition paths.

## Allocation record requirements

Each successful allocation record must include at least:

| Field | Reason |
| --- | --- |
| Tenant identity | Selects the quota to charge. |
| Process identity | Supports later container-level multi-process aggregation. |
| CUDA device and context identity | Device pointers and allocations are process-local and context-sensitive. |
| Allocation kind | Determines which release API and lifetime rules apply. |
| Pointer or allocation handle | Correlates release with allocation. |
| Charged size | Allows exact release accounting. |
| Stream, when applicable | Required for deferred async allocation/free semantics. |

For the initial process-local milestone, the allocation map is local metadata.
The authoritative tenant usage and admission decision must remain behind the
control contract so that a later shared-memory implementation can aggregate
multiple processes without changing wrapper semantics.

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
