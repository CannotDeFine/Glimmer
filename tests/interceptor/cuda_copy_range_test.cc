#include "internal/cuda_copy_range.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using glimmer::interceptor::can_defer_cuda_copy_range;
using glimmer::interceptor::CudaCopyRangeDispatch;

int current_storage{};
int foreign_storage{};
const auto kCurrentContext = reinterpret_cast<CUcontext>(&current_storage);
const auto kForeignContext = reinterpret_cast<CUcontext>(&foreign_storage);
std::array<std::byte, 256> host_storage;
std::array<std::byte, 256> foreign_host_storage;

struct NativeMetadata {
    CUmemorytype type{CU_MEMORYTYPE_DEVICE};
    CUcontext context{kCurrentContext};
    CUdeviceptr base{0x1000};
    std::size_t extent{256};
    unsigned int is_mapped{1};
    unsigned int is_managed{};
    unsigned int host_flags{};
    bool has_disjoint_alias{};
    CUpointer_attribute failed_attribute{};
    bool fails_range{};
    bool fails_flags{};
    unsigned int attribute_calls{};
    unsigned int range_calls{};
    unsigned int flag_calls{};
};
NativeMetadata metadata;

CUresult CUDAAPI attribute(void* output, CUpointer_attribute selected, CUdeviceptr pointer) {
    ++metadata.attribute_calls;
    if (metadata.failed_attribute == selected) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    switch (selected) {
        case CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
            *static_cast<CUmemorytype*>(output) = metadata.type;
            break;
        case CU_POINTER_ATTRIBUTE_CONTEXT:
            *static_cast<CUcontext*>(output) = metadata.context;
            break;
        case CU_POINTER_ATTRIBUTE_IS_MANAGED:
            *static_cast<unsigned int*>(output) = metadata.is_managed;
            break;
        case CU_POINTER_ATTRIBUTE_MAPPED:
            *static_cast<unsigned int*>(output) = metadata.is_mapped;
            break;
        case CU_POINTER_ATTRIBUTE_HOST_POINTER:
            if (pointer < metadata.base || pointer - metadata.base >= host_storage.size()) {
                return CUDA_ERROR_INVALID_VALUE;
            }
            *static_cast<void**>(output) =
                (metadata.has_disjoint_alias ? foreign_host_storage.data() : host_storage.data()) +
                (pointer - metadata.base);
            break;
        default:
            return CUDA_ERROR_NOT_SUPPORTED;
    }
    return CUDA_SUCCESS;
}
CUresult CUDAAPI address_range(CUdeviceptr* base, std::size_t* extent, CUdeviceptr) {
    ++metadata.range_calls;
    if (metadata.fails_range) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    *base = metadata.base;
    *extent = metadata.extent;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_flags(unsigned int* flags, void*) {
    ++metadata.flag_calls;
    if (metadata.fails_flags) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = metadata.host_flags;
    return CUDA_SUCCESS;
}
const CudaCopyRangeDispatch kDispatch{attribute, address_range, host_flags};

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
bool device_range(CUdeviceptr pointer, std::size_t memory_bytes,
                  const CudaCopyRangeDispatch& selected = kDispatch) {
    return can_defer_cuda_copy_range(selected, kCurrentContext, pointer, memory_bytes,
                                     CU_MEMORYTYPE_DEVICE);
}
void mapped_extent_and_overflow() {
    metadata = {};
    require(device_range(0x1000, 256) && device_range(0x1001, 255) && device_range(0x10ff, 1),
            "complete mapped allocation and interior/tail copies are admissible");
    require(!device_range(0x1000, 257) && !device_range(0x1001, 256) && !device_range(0x10ff, 2) &&
                !device_range(0x1100, 1) && !device_range(0x0fff, 1),
            "oversized, one-past and preceding ranges require native validation");
    metadata.extent = 0;
    require(!device_range(0x1000, 1), "successful empty native range is not proof");
    metadata.extent = 256;
    metadata.is_mapped = 0;
    require(!device_range(0x1000, 1), "reserved but unmapped address space is not proof");
    metadata.is_mapped = 2;
    require(!device_range(0x1000, 1), "unknown native mapped state is not proof");
    metadata = {};
    metadata.base = std::numeric_limits<CUdeviceptr>::max() - 3;
    metadata.extent = 4;
    require(device_range(metadata.base, 4) && device_range(metadata.base + 3, 1),
            "subtraction bounds do not need a representable one-past endpoint");
    require(!device_range(metadata.base + 3, 2), "copy endpoint overflow rejects before queries");
    metadata.extent = 5;
    require(!device_range(metadata.base, 1), "native reported extent overflow is not proof");
    metadata.base = 0;
    metadata.extent = 256;
    require(!device_range(1, 1), "successful native null range base is not proof");
}
void type_context_and_native_failures() {
    metadata = {};
    require(device_range(0x1000, 1) && metadata.attribute_calls == 4 && metadata.range_calls == 1 &&
                metadata.flag_calls == 0,
            "device metadata is queried once per required field without host queries");
    metadata.context = kForeignContext;
    require(!device_range(0x1000, 1), "foreign device owner cannot select this queue");
    metadata.context = nullptr;
    require(!device_range(0x1000, 1), "unknown device owner cannot select this queue");
    metadata = {};
    metadata.is_managed = 1;
    require(!device_range(0x1000, 1), "managed memory needs native routing");
    metadata = {};
    metadata.type = CU_MEMORYTYPE_HOST;
    require(!device_range(0x1000, 1), "payload type cannot disagree with the actual pointer");
    for (const auto selected : {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_CONTEXT,
                                CU_POINTER_ATTRIBUTE_IS_MANAGED, CU_POINTER_ATTRIBUTE_MAPPED}) {
        metadata = {};
        metadata.failed_attribute = selected;
        require(!device_range(0x1000, 1), "failed native attribute query cannot authorize issue");
    }
    metadata = {};
    metadata.fails_range = true;
    require(!device_range(0x1000, 1), "missing mapped range proof needs native routing");
}
void host_alias_flags_and_extent() {
    metadata = {};
    metadata.type = CU_MEMORYTYPE_HOST;
    metadata.base = reinterpret_cast<CUdeviceptr>(host_storage.data());
    const auto copy = [&](std::size_t offset, std::size_t bytes) {
        return can_defer_cuda_copy_range(kDispatch, kCurrentContext, metadata.base + offset, bytes,
                                         CU_MEMORYTYPE_HOST);
    };
    require(copy(0, 256) && copy(128, 128) && copy(255, 1) && !copy(128, 129),
            "actual pinned host alias uses the complete mapped extent");
    metadata.host_flags =
        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED;
    require(copy(0, 1), "known pinned allocation flags remain admissible");
    metadata.host_flags = CU_MEMHOSTREGISTER_READ_ONLY;
    require(!copy(0, 1), "readonly registration needs original native validation");
    metadata.host_flags = 0x80000000U;
    require(!copy(0, 1), "unknown host flags cannot select a typed queue payload");
    metadata.host_flags = 0;
    metadata.has_disjoint_alias = true;
    require(!copy(0, 1), "device alias cannot be cast to a CPU host copy pointer");
    metadata.has_disjoint_alias = false;
    metadata.failed_attribute = CU_POINTER_ATTRIBUTE_HOST_POINTER;
    require(!copy(0, 1), "missing actual host alias needs native routing");
    metadata.failed_attribute = {};
    metadata.fails_flags = true;
    require(!copy(0, 1), "failed pinned flag query needs native routing");
}
void missing_dispatch_and_invalid_inputs() {
    metadata = {};
    const CudaCopyRangeDispatch absent;
    auto missing_range = kDispatch;
    missing_range.address_range = nullptr;
    require(!device_range(0x1000, 1, absent) && !device_range(0x1000, 1, missing_range) &&
                !device_range(0, 1) && !device_range(0x1000, 0) &&
                !device_range(0x1000, std::numeric_limits<std::size_t>::max()) &&
                !can_defer_cuda_copy_range(kDispatch, nullptr, 0x1000, 1, CU_MEMORYTYPE_DEVICE) &&
                metadata.attribute_calls == 0 && metadata.range_calls == 0,
            "absent proof and invalid arithmetic/context make no native query");
    auto missing_flags = kDispatch;
    missing_flags.host_flags = nullptr;
    require(device_range(0x1000, 1, missing_flags),
            "device validation does not require a host kDispatch function");
    metadata.type = CU_MEMORYTYPE_HOST;
    require(
        !can_defer_cuda_copy_range(missing_flags, kCurrentContext, 0x1000, 1, CU_MEMORYTYPE_HOST),
        "pinned host validation requires native flag proof");
    metadata.type = CU_MEMORYTYPE_ARRAY;
    require(!can_defer_cuda_copy_range(kDispatch, kCurrentContext, 0x1000, 1, CU_MEMORYTYPE_ARRAY),
            "nonlinear memory never becomes a one-dimensional queued pointer");
}
}  // namespace

int main() {
    try {
        mapped_extent_and_overflow();
        type_context_and_native_failures();
        host_alias_flags_and_extent();
        missing_dispatch_and_invalid_inputs();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
