#include "cuda_binary_slice.h"
#include "cuda_gemm_allocation.h"
#include "allocation_registry.h"
#include "cuda_copy_range.h"

#include <cstring>
#include <limits>

namespace glimmer::interceptor {
namespace {

template <typename T>
T parameter(const std::byte* bytes, std::size_t offset) noexcept {
    T value{};
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}
}  // namespace

bool has_independent_gemm_range(const CudaCopyRangeDispatch& dispatch,
                                const AllocationRegistry& allocations, AllocationIdentity range,
                                std::uint64_t bytes) noexcept {
    const auto pointer = range.device_pointer;
    if (!can_defer_cuda_copy_range(dispatch, range.context, pointer, bytes, CU_MEMORYTYPE_DEVICE))
        return false;
    CUdeviceptr base{};
    std::size_t size{};
    if (dispatch.address_range(&base, &size, pointer) != CUDA_SUCCESS || !base || !size ||
        pointer < base || size > std::numeric_limits<CUdeviceptr>::max() - base)
        return false;
    const auto offset = pointer - base;
    if (offset > size || bytes > size - offset) return false;
    range.device_pointer = base;
    return allocations.has_live_independent_allocation(range, offset + bytes);
}

std::optional<CudaQueuedKernel> prepare_checked_mapped_gemm_slices(
    CUfunction qualified_function, std::array<unsigned int, 3> grid,
    std::array<unsigned int, 3> block, void* const* arguments,
    std::shared_ptr<void> kernel_resources, const CudaCopyRangeDispatch& dispatch,
    const AllocationRegistry& allocations, CUcontext context, CUdevice device, bool should_trace,
    QualifiedGemmMapping mapping) noexcept {
    if (!context || device < 0) return std::nullopt;
    auto prepared = prepare_mapped_gemm_slices(qualified_function, grid, block, arguments,
                                               std::move(kernel_resources), should_trace, mapping);
    if (!prepared) return std::nullopt;
    const auto* bytes = static_cast<const std::byte*>(prepared->arguments.addresses()[0]);
    constexpr std::array<std::size_t, 4> pointer_offsets{0, 8, 16, 168};
    for (std::size_t index = 0; index != pointer_offsets.size(); ++index) {
        const auto pointer = parameter<std::uint64_t>(bytes, pointer_offsets[index]);
        const auto stride = parameter<std::uint64_t>(bytes, 56 + index * 8);
        const auto leading = parameter<std::uint32_t>(bytes, 88 + index * 4);
        const auto columns = parameter<std::uint32_t>(bytes, index < 2 ? 112 : 108);
        // All factors and pointer endpoints were bounded by trusted preparation
        // before any native query. Never reread application parameter storage.
        const auto extent = ((grid[2] - 1) * stride + std::uint64_t{leading} * columns) * 4;
        if (!has_independent_gemm_range(dispatch, allocations, {pointer, context, device}, extent))
            return std::nullopt;
    }
    return prepared;
}

}  // namespace glimmer::interceptor
