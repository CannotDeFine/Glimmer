#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_map>

namespace {

constexpr std::size_t kPhysicalMemoryBytes = std::size_t{64} * 1024 * 1024;
constexpr std::size_t kForcedAllocationFailureBytes = 1536;
std::mutex g_mutex;
std::unordered_map<CUdeviceptr, std::size_t> g_allocations;
std::unordered_map<CUdeviceptr, CUstream> g_pending_frees;
struct FakeVmmAllocation {
    std::size_t memory_bytes;
    std::size_t reference_count;
};
std::unordered_map<CUmemGenericAllocationHandle, FakeVmmAllocation> g_vmm_allocations;
std::unordered_map<CUdeviceptr, CUmemGenericAllocationHandle> g_vmm_mappings;
std::size_t g_used_bytes = 0;
CUdeviceptr g_next_pointer = 0x100000U;
CUmemGenericAllocationHandle g_next_vmm_handle = 1;
CUdeviceptr g_next_virtual_address = 0x40000000U;
bool g_context_alive = true;
std::uint8_t g_context_token = 0;
std::uint8_t g_default_pool_token = 0;
std::uint8_t g_custom_pool_token = 0;
std::uint64_t g_pool_release_threshold = 0;

CUcontext fake_context() {
    return reinterpret_cast<CUcontext>(&g_context_token);
}

CUmemoryPool fake_default_pool() {
    return reinterpret_cast<CUmemoryPool>(&g_default_pool_token);
}

CUmemoryPool fake_custom_pool() {
    return reinterpret_cast<CUmemoryPool>(&g_custom_pool_token);
}

CUresult reserve_memory(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::scoped_lock lock(g_mutex);
    if (memory_bytes == kForcedAllocationFailureBytes) {
        *device_pointer = 0;
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!g_context_alive || memory_bytes > kPhysicalMemoryBytes - g_used_bytes) {
        *device_pointer = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    // Fault injection for the interceptor's ambiguous-success rollback path:
    // emulate a broken external allocator that reports success without a
    // usable pointer. No allocation is inserted into the fake driver's map.
    if (std::getenv("GLIMMER_FAKE_NULL_SUCCESS_POINTER") != nullptr) {
        *device_pointer = 0;
        return CUDA_SUCCESS;
    }

    // The dedicated preload regression test uses this switch to emulate a
    // driver returning a pointer identity that is already present in the
    // interceptor registry. The wrapper must fail closed in that case.
    if (std::getenv("GLIMMER_FAKE_DUPLICATE_POINTER") != nullptr && !g_allocations.empty()) {
        *device_pointer = g_allocations.begin()->first;
        return CUDA_SUCCESS;
    }

    const CUdeviceptr pointer = g_next_pointer;
    g_next_pointer += static_cast<CUdeviceptr>(memory_bytes + 0x1000U);
    g_allocations.emplace(pointer, memory_bytes);
    g_used_bytes += memory_bytes;
    *device_pointer = pointer;
    return CUDA_SUCCESS;
}

CUresult fake_mem_free(CUdeviceptr device_pointer) {
    std::scoped_lock lock(g_mutex);
    if (g_pending_frees.contains(device_pointer)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto allocation = g_allocations.find(device_pointer);
    if (allocation == g_allocations.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    g_used_bytes -= allocation->second;
    g_allocations.erase(allocation);
    return CUDA_SUCCESS;
}

CUresult fake_mem_free_async(CUdeviceptr device_pointer, CUstream stream) {
    std::scoped_lock lock(g_mutex);
    if (!g_allocations.contains(device_pointer) || g_pending_frees.contains(device_pointer)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    g_pending_frees.emplace(device_pointer, stream);
    return CUDA_SUCCESS;
}

void complete_pending_frees(CUstream stream, bool match_stream) {
    std::scoped_lock lock(g_mutex);
    for (auto pending = g_pending_frees.begin(); pending != g_pending_frees.end();) {
        if (match_stream && pending->second != stream) {
            ++pending;
            continue;
        }
        const auto allocation = g_allocations.find(pending->first);
        if (allocation != g_allocations.end()) {
            g_used_bytes -= allocation->second;
            g_allocations.erase(allocation);
        }
        pending = g_pending_frees.erase(pending);
    }
}

void* lookup_symbol(const char* symbol);

}  // namespace

extern "C" CUresult CUDAAPI cuInit(unsigned int) {
    std::scoped_lock lock(g_mutex);
    g_context_alive = true;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return reserve_memory(device_pointer, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              unsigned int) {
    return reserve_memory(device_pointer, memory_bytes);
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters)
extern "C" CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr* device_pointer, std::size_t* pitch,
                                               std::size_t width_bytes, std::size_t height,
                                               unsigned int) {
    if (pitch == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    constexpr std::size_t k_pitch_alignment = 256;
    if (width_bytes > std::numeric_limits<std::size_t>::max() - (k_pitch_alignment - 1U)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pitch = (width_bytes + k_pitch_alignment - 1U) / k_pitch_alignment * k_pitch_alignment;
    if (height != 0 && *pitch > std::numeric_limits<std::size_t>::max() / height) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return reserve_memory(device_pointer, *pitch * height);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr device_pointer) {
    return fake_mem_free(device_pointer);
}

extern "C" CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                            CUstream) {
    return reserve_memory(device_pointer, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemAllocAsync_ptsz(CUdeviceptr* device_pointer,
                                                 std::size_t memory_bytes, CUstream stream) {
    return cuMemAllocAsync(device_pointer, memory_bytes, stream);
}

extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* device_pointer,
                                                    std::size_t memory_bytes, CUmemoryPool,
                                                    CUstream) {
    return reserve_memory(device_pointer, memory_bytes);
}

extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* device_pointer,
                                                         std::size_t memory_bytes,
                                                         CUmemoryPool pool, CUstream stream) {
    return cuMemAllocFromPoolAsync(device_pointer, memory_bytes, pool, stream);
}

extern "C" CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle,
                                        std::size_t memory_bytes, const CUmemAllocationProp* prop,
                                        unsigned long long) {
    if (handle == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    if (memory_bytes == kForcedAllocationFailureBytes) {
        *handle = 0;
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!g_context_alive || memory_bytes > kPhysicalMemoryBytes - g_used_bytes) {
        *handle = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    const CUmemGenericAllocationHandle allocated_handle = g_next_vmm_handle++;
    g_vmm_allocations.emplace(
        allocated_handle, FakeVmmAllocation{.memory_bytes = memory_bytes, .reference_count = 1});
    g_used_bytes += memory_bytes;
    *handle = allocated_handle;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    std::scoped_lock lock(g_mutex);
    const auto allocation = g_vmm_allocations.find(handle);
    if (allocation == g_vmm_allocations.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (allocation->second.reference_count > 1) {
        --allocation->second.reference_count;
        return CUDA_SUCCESS;
    }
    g_used_bytes -= allocation->second.memory_bytes;
    g_vmm_allocations.erase(allocation);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, std::size_t, CUdeviceptr,
                                                unsigned long long) {
    if (device_pointer == nullptr || memory_bytes == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    *device_pointer = g_next_virtual_address;
    g_next_virtual_address += static_cast<CUdeviceptr>(memory_bytes);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemAddressFree(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    return device_pointer == 0 || memory_bytes == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemMap(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                     std::size_t, CUmemGenericAllocationHandle handle,
                                     unsigned long long) {
    if (device_pointer == 0 || memory_bytes == 0 || handle == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    const auto allocation = g_vmm_allocations.find(handle);
    if (allocation == g_vmm_allocations.end() || memory_bytes > allocation->second.memory_bytes) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    g_vmm_mappings[device_pointer] = handle;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    if (device_pointer == 0 || memory_bytes == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    return g_vmm_mappings.erase(device_pointer) == 1 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

extern "C" CUresult CUDAAPI cuMemSetAccess(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                           const CUmemAccessDesc* access_descriptors,
                                           std::size_t descriptor_count) {
    return device_pointer == 0 || memory_bytes == 0 ||
                   (descriptor_count != 0 && access_descriptors == nullptr)
               ? CUDA_ERROR_INVALID_VALUE
               : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* base_pointer,
                                                    std::size_t* memory_bytes,
                                                    CUdeviceptr device_pointer) {
    if (base_pointer == nullptr || memory_bytes == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *base_pointer = device_pointer;
    *memory_bytes = 4096;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location,
                                           CUdeviceptr device_pointer) {
    if (flags == nullptr || location == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemExportToShareableHandle(void* shareable_handle,
                                                         CUmemGenericAllocationHandle handle,
                                                         CUmemAllocationHandleType,
                                                         unsigned long long) {
    if (shareable_handle == nullptr || handle == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    return g_vmm_allocations.contains(handle) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}

extern "C" CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                                                           void* os_handle,
                                                           CUmemAllocationHandleType handle_type) {
    if (handle == nullptr ||
        (os_handle == nullptr && handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    *handle = g_next_vmm_handle++;
    g_vmm_allocations.emplace(*handle, FakeVmmAllocation{.memory_bytes = 0, .reference_count = 1});
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemGetAllocationGranularity(std::size_t* granularity,
                                                          const CUmemAllocationProp* prop,
                                                          CUmemAllocationGranularity_flags) {
    if (granularity == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *granularity = 65536;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) {
    if (prop == nullptr || handle == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    if (!g_vmm_allocations.contains(handle)) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    *prop = {};
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = 0;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle,
                                                        void* device_pointer) {
    if (handle == nullptr || device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    const auto mapping = g_vmm_mappings.find(reinterpret_cast<CUdeviceptr>(device_pointer));
    if (mapping == g_vmm_mappings.end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto allocation = g_vmm_allocations.find(mapping->second);
    if (allocation == g_vmm_allocations.end()) {
        return CUDA_ERROR_INVALID_HANDLE;
    }
    if (allocation->second.reference_count == std::numeric_limits<std::size_t>::max()) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    ++allocation->second.reference_count;
    *handle = mapping->second;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, std::size_t) {
    return pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) {
    if (pool == nullptr || value == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (attribute == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) {
        g_pool_release_threshold = *static_cast<std::uint64_t*>(value);
    }
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) {
    if (pool == nullptr || value == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (attribute == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) {
        *static_cast<std::uint64_t*>(value) = g_pool_release_threshold;
    }
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool,
                                               const CUmemAccessDesc* access_descriptors,
                                               std::size_t descriptor_count) {
    return pool == nullptr || (descriptor_count != 0 && access_descriptors == nullptr)
               ? CUDA_ERROR_INVALID_VALUE
               : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool pool,
                                               CUmemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = fake_custom_pool();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool) {
    return pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice device) {
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = fake_default_pool();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuDeviceSetMemPool(CUdevice device, CUmemoryPool pool) {
    return device < 0 || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* pool, CUdevice device) {
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = fake_default_pool();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemGetDefaultMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                                   CUmemAllocationType) {
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = fake_default_pool();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType) {
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool = fake_default_pool();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemSetMemPool(CUmemLocation* location, CUmemAllocationType,
                                            CUmemoryPool pool) {
    return location == nullptr || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool,
                                                             CUmemAllocationHandleType,
                                                             unsigned long long) {
    return handle_out == nullptr || pool == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI
cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle,
                                   CUmemAllocationHandleType handle_type, unsigned long long) {
    if (pool_out == nullptr ||
        (handle == nullptr && handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pool_out = fake_custom_pool();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolExportPointer(CUmemPoolPtrExportData* share_data_out,
                                                   CUdeviceptr device_pointer) {
    return share_data_out == nullptr || device_pointer == 0 ? CUDA_ERROR_INVALID_VALUE
                                                            : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemPoolImportPointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                                   CUmemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer_out = 0x70000000U;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr device_pointer, CUstream stream) {
    return fake_mem_free_async(device_pointer, stream);
}

extern "C" CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr device_pointer, CUstream stream) {
    return cuMemFreeAsync(device_pointer, stream);
}

extern "C" CUresult CUDAAPI cuStreamGetDevice(CUstream, CUdevice* device) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamGetDevice_ptsz(CUstream stream, CUdevice* device) {
    return cuStreamGetDevice(stream, device);
}

extern "C" CUresult CUDAAPI cuStreamGetCtx(CUstream, CUcontext* context) {
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    if (!g_context_alive) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *context = fake_context();
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream stream, CUcontext* context) {
    return cuStreamGetCtx(stream, context);
}

extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream) {
    complete_pending_frees(stream, true);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream stream) {
    return cuStreamQuery(stream);
}

extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    complete_pending_frees(stream, true);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream stream) {
    return cuStreamSynchronize(stream);
}

extern "C" CUresult CUDAAPI cuStreamDestroy_v2(CUstream) {
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuStreamDestroy(CUstream stream) {
    return cuStreamDestroy_v2(stream);
}

extern "C" CUresult CUDAAPI cuCtxSynchronize() {
    complete_pending_frees(nullptr, false);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    *free_bytes = kPhysicalMemoryBytes - g_used_bytes;
    *total_bytes = kPhysicalMemoryBytes;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem_v2(std::size_t* total_bytes, CUdevice) {
    if (total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *total_bytes = kPhysicalMemoryBytes;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    *context = g_context_alive ? fake_context() : nullptr;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    if (!g_context_alive) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context) {
    if (context != fake_context()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    std::scoped_lock lock(g_mutex);
    g_context_alive = false;
    g_pending_frees.clear();
    g_allocations.clear();
    g_used_bytes = 0;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** function_pointer, int,
                                             cuuint64_t) {
    if (symbol == nullptr || function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *function_pointer = lookup_symbol(symbol);
    return *function_pointer == nullptr ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** function_pointer,
                                                int version, cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status) {
    const CUresult result = cuGetProcAddress(symbol, function_pointer, version, flags);
    if (symbol_status != nullptr) {
        *symbol_status = result == CUDA_SUCCESS ? CU_GET_PROC_ADDRESS_SUCCESS
                                                : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    return result;
}

namespace {

void* lookup_symbol(const char* symbol) {
    if (symbol == nullptr) {
        return nullptr;
    }
    if (std::string_view(symbol) == "cuInit") {
        return reinterpret_cast<void*>(&cuInit);
    }
    if (std::string_view(symbol) == "cuMemAlloc" || std::string_view(symbol) == "cuMemAlloc_v2") {
        return reinterpret_cast<void*>(&cuMemAlloc_v2);
    }
    if (std::string_view(symbol) == "cuMemAllocManaged") {
        return reinterpret_cast<void*>(&cuMemAllocManaged);
    }
    if (std::string_view(symbol) == "cuMemAllocPitch" ||
        std::string_view(symbol) == "cuMemAllocPitch_v2") {
        return reinterpret_cast<void*>(&cuMemAllocPitch_v2);
    }
    if (std::string_view(symbol) == "cuMemAllocAsync") {
        return reinterpret_cast<void*>(&cuMemAllocAsync);
    }
    if (std::string_view(symbol) == "cuMemAllocAsync_ptsz") {
        return reinterpret_cast<void*>(&cuMemAllocAsync_ptsz);
    }
    if (std::string_view(symbol) == "cuMemAllocFromPoolAsync") {
        return reinterpret_cast<void*>(&cuMemAllocFromPoolAsync);
    }
    if (std::string_view(symbol) == "cuMemAllocFromPoolAsync_ptsz") {
        return reinterpret_cast<void*>(&cuMemAllocFromPoolAsync_ptsz);
    }
    if (std::string_view(symbol) == "cuMemCreate") {
        return reinterpret_cast<void*>(&cuMemCreate);
    }
    if (std::string_view(symbol) == "cuMemRelease") {
        return reinterpret_cast<void*>(&cuMemRelease);
    }
    if (std::string_view(symbol) == "cuMemAddressReserve") {
        return reinterpret_cast<void*>(&cuMemAddressReserve);
    }
    if (std::string_view(symbol) == "cuMemAddressFree") {
        return reinterpret_cast<void*>(&cuMemAddressFree);
    }
    if (std::string_view(symbol) == "cuMemMap") {
        return reinterpret_cast<void*>(&cuMemMap);
    }
    if (std::string_view(symbol) == "cuMemUnmap") {
        return reinterpret_cast<void*>(&cuMemUnmap);
    }
    if (std::string_view(symbol) == "cuMemSetAccess") {
        return reinterpret_cast<void*>(&cuMemSetAccess);
    }
    if (std::string_view(symbol) == "cuMemGetAddressRange" ||
        std::string_view(symbol) == "cuMemGetAddressRange_v2") {
        return reinterpret_cast<void*>(&cuMemGetAddressRange_v2);
    }
    if (std::string_view(symbol) == "cuMemGetAccess") {
        return reinterpret_cast<void*>(&cuMemGetAccess);
    }
    if (std::string_view(symbol) == "cuMemExportToShareableHandle") {
        return reinterpret_cast<void*>(&cuMemExportToShareableHandle);
    }
    if (std::string_view(symbol) == "cuMemImportFromShareableHandle") {
        return reinterpret_cast<void*>(&cuMemImportFromShareableHandle);
    }
    if (std::string_view(symbol) == "cuMemGetAllocationGranularity") {
        return reinterpret_cast<void*>(&cuMemGetAllocationGranularity);
    }
    if (std::string_view(symbol) == "cuMemGetAllocationPropertiesFromHandle") {
        return reinterpret_cast<void*>(&cuMemGetAllocationPropertiesFromHandle);
    }
    if (std::string_view(symbol) == "cuMemRetainAllocationHandle") {
        return reinterpret_cast<void*>(&cuMemRetainAllocationHandle);
    }
    if (std::string_view(symbol) == "cuMemPoolTrimTo") {
        return reinterpret_cast<void*>(&cuMemPoolTrimTo);
    }
    if (std::string_view(symbol) == "cuMemPoolSetAttribute") {
        return reinterpret_cast<void*>(&cuMemPoolSetAttribute);
    }
    if (std::string_view(symbol) == "cuMemPoolGetAttribute") {
        return reinterpret_cast<void*>(&cuMemPoolGetAttribute);
    }
    if (std::string_view(symbol) == "cuMemPoolSetAccess") {
        return reinterpret_cast<void*>(&cuMemPoolSetAccess);
    }
    if (std::string_view(symbol) == "cuMemPoolGetAccess") {
        return reinterpret_cast<void*>(&cuMemPoolGetAccess);
    }
    if (std::string_view(symbol) == "cuMemPoolCreate") {
        return reinterpret_cast<void*>(&cuMemPoolCreate);
    }
    if (std::string_view(symbol) == "cuMemPoolDestroy") {
        return reinterpret_cast<void*>(&cuMemPoolDestroy);
    }
    if (std::string_view(symbol) == "cuDeviceGetMemPool") {
        return reinterpret_cast<void*>(&cuDeviceGetMemPool);
    }
    if (std::string_view(symbol) == "cuDeviceSetMemPool") {
        return reinterpret_cast<void*>(&cuDeviceSetMemPool);
    }
    if (std::string_view(symbol) == "cuDeviceGetDefaultMemPool") {
        return reinterpret_cast<void*>(&cuDeviceGetDefaultMemPool);
    }
    if (std::string_view(symbol) == "cuMemGetDefaultMemPool") {
        return reinterpret_cast<void*>(&cuMemGetDefaultMemPool);
    }
    if (std::string_view(symbol) == "cuMemGetMemPool") {
        return reinterpret_cast<void*>(&cuMemGetMemPool);
    }
    if (std::string_view(symbol) == "cuMemSetMemPool") {
        return reinterpret_cast<void*>(&cuMemSetMemPool);
    }
    if (std::string_view(symbol) == "cuMemPoolExportToShareableHandle") {
        return reinterpret_cast<void*>(&cuMemPoolExportToShareableHandle);
    }
    if (std::string_view(symbol) == "cuMemPoolImportFromShareableHandle") {
        return reinterpret_cast<void*>(&cuMemPoolImportFromShareableHandle);
    }
    if (std::string_view(symbol) == "cuMemPoolExportPointer") {
        return reinterpret_cast<void*>(&cuMemPoolExportPointer);
    }
    if (std::string_view(symbol) == "cuMemPoolImportPointer") {
        return reinterpret_cast<void*>(&cuMemPoolImportPointer);
    }
    if (std::string_view(symbol) == "cuMemFree" || std::string_view(symbol) == "cuMemFree_v2") {
        return reinterpret_cast<void*>(&cuMemFree_v2);
    }
    if (std::string_view(symbol) == "cuMemFreeAsync") {
        return reinterpret_cast<void*>(&cuMemFreeAsync);
    }
    if (std::string_view(symbol) == "cuMemFreeAsync_ptsz") {
        return reinterpret_cast<void*>(&cuMemFreeAsync_ptsz);
    }
    if (std::string_view(symbol) == "cuMemGetInfo" ||
        std::string_view(symbol) == "cuMemGetInfo_v2") {
        return reinterpret_cast<void*>(&cuMemGetInfo_v2);
    }
    if (std::string_view(symbol) == "cuDeviceTotalMem" ||
        std::string_view(symbol) == "cuDeviceTotalMem_v2") {
        return reinterpret_cast<void*>(&cuDeviceTotalMem_v2);
    }
    if (std::string_view(symbol) == "cuCtxGetCurrent") {
        return reinterpret_cast<void*>(&cuCtxGetCurrent);
    }
    if (std::string_view(symbol) == "cuCtxGetDevice") {
        return reinterpret_cast<void*>(&cuCtxGetDevice);
    }
    if (std::string_view(symbol) == "cuCtxDestroy" ||
        std::string_view(symbol) == "cuCtxDestroy_v2") {
        return reinterpret_cast<void*>(&cuCtxDestroy_v2);
    }
    if (std::string_view(symbol) == "cuStreamGetDevice") {
        return reinterpret_cast<void*>(&cuStreamGetDevice);
    }
    if (std::string_view(symbol) == "cuStreamGetDevice_ptsz") {
        return reinterpret_cast<void*>(&cuStreamGetDevice_ptsz);
    }
    if (std::string_view(symbol) == "cuStreamGetCtx") {
        return reinterpret_cast<void*>(&cuStreamGetCtx);
    }
    if (std::string_view(symbol) == "cuStreamGetCtx_ptsz") {
        return reinterpret_cast<void*>(&cuStreamGetCtx_ptsz);
    }
    if (std::string_view(symbol) == "cuStreamQuery") {
        return reinterpret_cast<void*>(&cuStreamQuery);
    }
    if (std::string_view(symbol) == "cuStreamQuery_ptsz") {
        return reinterpret_cast<void*>(&cuStreamQuery_ptsz);
    }
    if (std::string_view(symbol) == "cuStreamSynchronize") {
        return reinterpret_cast<void*>(&cuStreamSynchronize);
    }
    if (std::string_view(symbol) == "cuStreamSynchronize_ptsz") {
        return reinterpret_cast<void*>(&cuStreamSynchronize_ptsz);
    }
    if (std::string_view(symbol) == "cuStreamDestroy" ||
        std::string_view(symbol) == "cuStreamDestroy_v2") {
        return reinterpret_cast<void*>(&cuStreamDestroy_v2);
    }
    if (std::string_view(symbol) == "cuCtxSynchronize") {
        return reinterpret_cast<void*>(&cuCtxSynchronize);
    }
    if (std::string_view(symbol) == "cuGetProcAddress") {
        return reinterpret_cast<void*>(&cuGetProcAddress);
    }
    if (std::string_view(symbol) == "cuGetProcAddress_v2") {
        return reinterpret_cast<void*>(&cuGetProcAddress_v2);
    }
    return nullptr;
}

}  // namespace
