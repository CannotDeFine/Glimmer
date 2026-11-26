#pragma once

#include <cuda.h>

#include <cstddef>

namespace glimmer::interceptor {

struct CudaCopyRangeDispatch {
    decltype(&cuPointerGetAttribute) pointer_attribute{};
    decltype(&cuMemGetAddressRange) address_range{};
    decltype(&cuMemHostGetFlags) host_flags{};
};

// Native metadata only: no copy, allocation, context mutation, locks or owned
// resource. The caller supplies original pointers and the native reentrancy scope.
// False requires original native validation, never deferred acceptance.
[[nodiscard]] bool can_defer_cuda_copy_range(const CudaCopyRangeDispatch& dispatch,
                                             CUcontext current_context, CUdeviceptr pointer,
                                             std::size_t memory_bytes,
                                             CUmemorytype expected_type) noexcept;

}  // namespace glimmer::interceptor
