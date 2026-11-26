#pragma once

#include "cuda_operation_queue.h"
#include "qualified_cubin.h"

#include <cstdint>

namespace glimmer::interceptor {

class AllocationRegistry;
struct CudaCopyRangeDispatch;

struct CudaBinarySliceProgress {
    std::uint64_t logical_blocks{};
    std::uint64_t completed_blocks{};
    std::uint64_t physical_launches{};
};

// Externally serialized by the queue owner. All state is CPU metadata; only
// inspect_stopped, following actual queue event proof, advances completed work.
class CudaBinarySliceContinuation : public CudaQueueContinuation {
   public:
    [[nodiscard]] virtual CudaBinarySliceProgress progress() const noexcept = 0;
};

// TRUSTED internal preparation, not binary discovery or public CUDA routing.
// Caller must prove the function's pinned single-CTA/range mapping, ABI and context,
// and retain its module and referenced allocations through kernel_resources.
// Copies the original 4/4/24-byte parameters; only the private aggregate grows.
// Qualified shape: 1D, 128 threads, aligned pointer aggregate. Single-CTA mapping
// accepts <=32769 elements. Range mapping retains that granularity for small
// inputs and uses <=1024-CTA ranges above it, through 4194305 elements.
// Output/input ranges must be disjoint or exact aliases; wrapped ranges reject.
[[nodiscard]] std::optional<CudaQueuedKernel> prepare_mapped_add_slices(
    CUfunction qualified_function, std::array<unsigned int, 3> grid,
    std::array<unsigned int, 3> block, void* const* arguments,
    std::shared_ptr<void> kernel_resources, bool should_trace = false,
    QualifiedAddMapping mapping = QualifiedAddMapping::kSingleCta) noexcept;

// TRUSTED pinned GEMM function/context and allocation ownership prerequisite as
// above; no image qualification, native issue or public selection occurs here.
// Copies the original 176-byte mode-one, immediate-scalar parameter and appends
// one private eight-byte batch base. Preserves X/Y tiles, issues Z=1 for the
// plane mapping or at most eight Z planes for the range mapping. Advances
// X*Y*issued-Z logical CTAs only after trailing-event proof. Bounded domain:
// M<=129, N<=33, K<=2048, batches<=36, 256/1/1 threads; leading/stride padding
// 0..5. Output is disjoint from A/B; input C is disjoint or an exact layout alias.
[[nodiscard]] std::optional<CudaQueuedKernel> prepare_mapped_gemm_slices(
    CUfunction qualified_function, std::array<unsigned int, 3> grid,
    std::array<unsigned int, 3> block, void* const* arguments,
    std::shared_ptr<void> kernel_resources, bool should_trace = false,
    QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) noexcept;

// Additionally prove every complete access span against native
// metadata and observed ordinary allocation provenance, using the same owned
// parameter snapshot as execution. Missing/untracked/releasing storage rejects.
// Caller still owes exact code/current-context validation, a native reentrancy
// scope and externally ordered allocation lifetime through logical completion.
// This is a preparation check, not a lease or public launch/fallback operation.
[[nodiscard]] std::optional<CudaQueuedKernel> prepare_checked_mapped_gemm_slices(
    CUfunction qualified_function, std::array<unsigned int, 3> grid,
    std::array<unsigned int, 3> block, void* const* arguments,
    std::shared_ptr<void> kernel_resources, const CudaCopyRangeDispatch& dispatch,
    const AllocationRegistry& allocations, CUcontext context, CUdevice device,
    bool should_trace = false,
    QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) noexcept;

}  // namespace glimmer::interceptor
