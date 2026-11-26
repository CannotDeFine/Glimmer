#include "cuda_copy_range.h"

#include <limits>

namespace glimmer::interceptor {

bool can_defer_cuda_copy_range(const CudaCopyRangeDispatch& dispatch, CUcontext current_context,
                               CUdeviceptr pointer, std::size_t memory_bytes,
                               CUmemorytype expected_type) noexcept {
    if (!dispatch.pointer_attribute || !dispatch.address_range || !current_context || !pointer ||
        !memory_bytes || memory_bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - pointer) {
        return false;
    }
    CUmemorytype type{};
    if (dispatch.pointer_attribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, pointer) !=
            CUDA_SUCCESS ||
        type != expected_type) {
        return false;
    }
    if (type == CU_MEMORYTYPE_DEVICE) {
        CUcontext owner{};
        unsigned int is_managed{};
        if (dispatch.pointer_attribute(&owner, CU_POINTER_ATTRIBUTE_CONTEXT, pointer) !=
                CUDA_SUCCESS ||
            owner != current_context ||
            dispatch.pointer_attribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, pointer) !=
                CUDA_SUCCESS ||
            is_managed != 0) {
            return false;
        }
    } else if (type == CU_MEMORYTYPE_HOST) {
        void* host_alias{};
        unsigned int flags{};
        constexpr unsigned int k_supported_flags =
            CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED;
        if (!dispatch.host_flags ||
            dispatch.pointer_attribute(&host_alias, CU_POINTER_ATTRIBUTE_HOST_POINTER, pointer) !=
                CUDA_SUCCESS ||
            reinterpret_cast<CUdeviceptr>(host_alias) != pointer ||
            dispatch.host_flags(&flags, host_alias) != CUDA_SUCCESS ||
            (flags & ~k_supported_flags) != 0) {
            return false;
        }
    } else {
        return false;
    }
    unsigned int is_mapped{};
    CUdeviceptr base{};
    std::size_t extent{};
    if (dispatch.pointer_attribute(&is_mapped, CU_POINTER_ATTRIBUTE_MAPPED, pointer) !=
            CUDA_SUCCESS ||
        is_mapped != 1 || dispatch.address_range(&base, &extent, pointer) != CUDA_SUCCESS ||
        !base || !extent || extent - 1 > std::numeric_limits<CUdeviceptr>::max() - base ||
        pointer < base) {
        return false;
    }
    const auto offset = pointer - base;
    return offset < extent && memory_bytes <= extent - offset;
}

}  // namespace glimmer::interceptor
