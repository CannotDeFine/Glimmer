#pragma once

#include "allocation_registry.h"
#include "cuda_copy_range.h"

namespace glimmer::interceptor {

// Read-only native metadata plus tracked ordinary allocation provenance.
// Caller supplies the native reentrancy scope and externally ordered lifetime.
[[nodiscard]] bool has_independent_gemm_range(const CudaCopyRangeDispatch& dispatch,
                                              const AllocationRegistry& allocations,
                                              AllocationIdentity range,
                                              std::uint64_t bytes) noexcept;

}  // namespace glimmer::interceptor
