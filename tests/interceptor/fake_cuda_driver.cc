#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>
#include "event_audit_fixture.h"

#ifdef cuEventDestroy
#undef cuEventDestroy
#endif

#ifdef cuEventRecord
#undef cuEventRecord
#endif

#ifdef cuEventRecordWithFlags
#undef cuEventRecordWithFlags
#endif

#ifdef cuStreamWaitEvent
#undef cuStreamWaitEvent
#endif

#ifdef cudaEventRecord
#undef cudaEventRecord
#endif

#ifdef cudaEventRecordWithFlags
#undef cudaEventRecordWithFlags
#endif

#ifdef cudaStreamWaitEvent
#undef cudaStreamWaitEvent
#endif

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
#ifdef cuCtxDestroy
#undef cuCtxDestroy
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
#ifdef cuGraphLaunch
#undef cuGraphLaunch
#endif
#ifdef cuLaunchKernelEx
#undef cuLaunchKernelEx
#endif
#ifdef cuIpcOpenMemHandle
#undef cuIpcOpenMemHandle
#endif
#ifdef cuIpcOpenMemHandle_v2
#undef cuIpcOpenMemHandle_v2
#endif
#ifdef cuArrayCreate
#undef cuArrayCreate
#endif
#ifdef cuArrayCreate_v2
#undef cuArrayCreate_v2
#endif
#ifdef cuArray3DCreate
#undef cuArray3DCreate
#endif
#ifdef cuArray3DCreate_v2
#undef cuArray3DCreate_v2
#endif
#ifdef cuGraphicsResourceGetMappedPointer
#undef cuGraphicsResourceGetMappedPointer
#endif
#ifdef cuGraphicsResourceGetMappedPointer_v2
#undef cuGraphicsResourceGetMappedPointer_v2
#endif
#ifdef cuGraphicsResourceSetMapFlags
#undef cuGraphicsResourceSetMapFlags
#endif
#ifdef cuGraphicsResourceSetMapFlags_v2
#undef cuGraphicsResourceSetMapFlags_v2
#endif

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_map>

extern "C" void* glimmer_test_memory_symbol(const char* name, cuuint64_t flags);

namespace {
glimmer::tests::EventAudit g_event_audit;

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
std::uint8_t g_external_memory_token = 0;
std::uint8_t g_array_token = 0;
std::uint8_t g_graphics_resource_token = 0;
std::uint8_t g_event_token = 0;
std::uint8_t g_module_token = 0;
std::uint8_t g_kernel_token = 0;
std::uint8_t g_persistent_kernel_token = 0;
bool g_has_persistent_variant = false;
std::atomic<int> g_capture_status{CU_STREAM_CAPTURE_STATUS_NONE};
std::array<std::atomic<unsigned int>, 2> g_capture_queries{};
std::atomic<unsigned int> g_internal_events{0};
std::atomic<unsigned int> g_context_device_queries{0};
bool g_is_cold_primary_initialized = false;
unsigned int g_capacity_allocation_calls = 0;
unsigned int g_capacity_memory_queries = 0;
unsigned int g_persistent_worker_count = 0;
unsigned int g_persistent_logical_count = 0;
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
    if (std::getenv("GLIMMER_FAKE_COLD_RUNTIME_CAPACITY") != nullptr) {
        ++g_capacity_allocation_calls;
    }
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
    g_is_cold_primary_initialized = true;
    *device_pointer = pointer;
    if (std::getenv("GLIMMER_FAKE_PROGRESS_ALLOCATION_ERROR") != nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
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

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned int grid_x, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int,
                                           unsigned int, CUstream, void** arguments, void**) {
    std::scoped_lock lock(g_mutex);
    if (function == reinterpret_cast<CUfunction>(&g_persistent_kernel_token)) {
        if (arguments == nullptr || arguments[2] == nullptr) {
            return CUDA_ERROR_INVALID_VALUE;
        }
        g_persistent_worker_count = grid_x;
        g_persistent_logical_count = *static_cast<unsigned int*>(arguments[2]);
    } else {
        g_persistent_worker_count = grid_x;
        g_persistent_logical_count = grid_x;
    }
    return CUDA_SUCCESS;
}

extern "C" unsigned int glimmer_fake_persistent_worker_count() {
    std::scoped_lock lock(g_mutex);
    return g_persistent_worker_count;
}

extern "C" unsigned int glimmer_fake_persistent_logical_count() {
    std::scoped_lock lock(g_mutex);
    return g_persistent_logical_count;
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
    std::scoped_lock lock(g_mutex);
    if (module == nullptr || image == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto* text = static_cast<const char*>(image);
    g_has_persistent_variant =
        (text[0] == '.' || text[0] == '\n') &&
        std::string_view(text).find("write_indices__glimmer_ptb") != std::string_view::npos;
    *module = reinterpret_cast<CUmodule>(&g_module_token);
    return CUDA_SUCCESS;
}

namespace {
unsigned int g_library_calls{};
std::uint8_t g_library_token{};
}  // namespace
extern "C" unsigned int glimmer_fake_library_calls() {
    std::scoped_lock lock(g_mutex);
    return g_library_calls;
}
extern "C" CUresult CUDAAPI cuLibraryLoadData(CUlibrary* library, const void* image,
                                              CUjit_option* options, void** values,
                                              unsigned int count, CUlibraryOption* library_options,
                                              void** library_values, unsigned int library_count) {
    std::scoped_lock lock(g_mutex);
    ++g_library_calls;
    if (!library || !image) return CUDA_ERROR_INVALID_VALUE;
    if (count || library_count) {
        return count == 1 && library_count == 1 && options && values && library_options &&
                       library_values && options[0] == CU_JIT_MAX_REGISTERS &&
                       library_options[0] == CU_LIBRARY_BINARY_IS_PRESERVED &&
                       values[0] == library && library_values[0] == library
                   ? CUDA_ERROR_INVALID_IMAGE
                   : CUDA_ERROR_INVALID_VALUE;
    }
    *library = reinterpret_cast<CUlibrary>(&g_library_token);
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuLibraryLoadFromFile(
    CUlibrary* library, const char* path, CUjit_option* options, void** values, unsigned int count,
    CUlibraryOption* library_options, void** library_values, unsigned int library_count) {
    return cuLibraryLoadData(library, path, options, values, count, library_options, library_values,
                             library_count);
}
extern "C" CUresult CUDAAPI cuLibraryGetModule(CUmodule* module, CUlibrary library) {
    std::scoped_lock lock(g_mutex);
    ++g_library_calls;
    if (!module || library != reinterpret_cast<CUlibrary>(&g_library_token))
        return CUDA_ERROR_INVALID_HANDLE;
    *module = reinterpret_cast<CUmodule>(&g_module_token);
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuLibraryUnload(CUlibrary library) {
    std::scoped_lock lock(g_mutex);
    ++g_library_calls;
    return library == reinterpret_cast<CUlibrary>(&g_library_token) ? CUDA_SUCCESS
                                                                    : CUDA_ERROR_INVALID_HANDLE;
}

extern "C" CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int,
                                               CUjit_option*, void**) {
    return cuModuleLoadData(module, image);
}

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule module,
                                                const char* name) {
    std::scoped_lock lock(g_mutex);
    if (function == nullptr || name == nullptr ||
        module != reinterpret_cast<CUmodule>(&g_module_token)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (std::string_view(name) == "write_indices__glimmer_ptb") {
        if (!g_has_persistent_variant) {
            return CUDA_ERROR_NOT_FOUND;
        }
        *function = reinterpret_cast<CUfunction>(&g_persistent_kernel_token);
    } else {
        *function = reinterpret_cast<CUfunction>(&g_kernel_token);
    }
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module) {
    return module == reinterpret_cast<CUmodule>(&g_module_token) ? CUDA_SUCCESS
                                                                 : CUDA_ERROR_INVALID_HANDLE;
}

extern "C" void glimmer_fake_set_capture_status(int status) {
    g_capture_status.store(status);
}

extern "C" unsigned int glimmer_fake_capture_query_count(unsigned int alias) {
    return alias < g_capture_queries.size() ? g_capture_queries[alias].load() : 0;
}

extern "C" unsigned int glimmer_fake_internal_event_count() {
    return g_internal_events.load();
}

namespace {

CUresult fake_capture_status(CUstreamCaptureStatus* status, std::size_t alias) {
    ++g_capture_queries[alias];
    if (status == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const int configured = g_capture_status.load();
    if (configured < 0) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    *status = static_cast<CUstreamCaptureStatus>(configured);
    return CUDA_SUCCESS;
}

}  // namespace

extern "C" CUresult CUDAAPI cuStreamIsCapturing(CUstream, CUstreamCaptureStatus* status) {
    return fake_capture_status(status, 0);
}

extern "C" CUresult CUDAAPI cuStreamIsCapturing_ptsz(CUstream, CUstreamCaptureStatus* status) {
    return fake_capture_status(status, 1);
}

extern "C" CUresult CUDAAPI cuOccupancyMaxActiveBlocksPerMultiprocessor(int* blocks, CUfunction,
                                                                        int, std::size_t) {
    if (blocks == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *blocks = 1;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction function, unsigned int grid_dim_x,
                                                unsigned int grid_dim_y, unsigned int grid_dim_z,
                                                unsigned int block_dim_x, unsigned int block_dim_y,
                                                unsigned int block_dim_z,
                                                unsigned int shared_memory_bytes, CUstream stream,
                                                void** kernel_parameters, void** extra) {
    return cuLaunchKernel(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                          block_dim_z, shared_memory_bytes, stream, kernel_parameters, extra);
}

extern "C" CUresult CUDAAPI cuGraphLaunch(CUgraphExec graph_exec, CUstream) {
    return graph_exec == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction function,
                                             void** arguments, void** extra) {
    if (config == nullptr || function == nullptr || arguments == nullptr ||
        extra != arguments + 3 || arguments[0] != config || arguments[1] != config->attrs ||
        config->numAttrs != 1 || config->gridDimX != 3 || config->blockDimX != 32 ||
        config->sharedMemBytes != 64) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return *static_cast<CUresult*>(arguments[2]);
}

extern "C" CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction function,
                                                  void** arguments, void** extra) {
    return cuLaunchKernelEx(config, function, arguments, extra);
}

extern "C" CUresult CUDAAPI cuGraphLaunch_ptsz(CUgraphExec graph_exec, CUstream stream) {
    return cuGraphLaunch(graph_exec, stream);
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

extern "C" CUresult CUDAAPI cuMemMapArrayAsync(CUarrayMapInfo* map_info_list, unsigned int count,
                                               CUstream) {
    return count != 0 && map_info_list == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
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

extern "C" CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle* handle, CUdeviceptr device_pointer) {
    if (handle == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = {};
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuIpcOpenMemHandle_v2(CUdeviceptr* device_pointer, CUipcMemHandle,
                                                  unsigned int) {
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x60000000U;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                               unsigned int flags) {
    return cuIpcOpenMemHandle_v2(device_pointer, handle, flags);
}

extern "C" CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr device_pointer) {
    return device_pointer == 0 ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuImportExternalMemory(
    CUexternalMemory* external_memory, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc) {
    if (external_memory == nullptr || handle_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *external_memory = reinterpret_cast<CUexternalMemory>(&g_external_memory_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI
cuExternalMemoryGetMappedBuffer(CUdeviceptr* device_pointer, CUexternalMemory external_memory,
                                const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc) {
    if (device_pointer == nullptr || external_memory == nullptr || buffer_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x61000000U;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray* mipmap, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc) {
    if (mipmap == nullptr || external_memory == nullptr || mipmap_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *mipmap = reinterpret_cast<CUmipmappedArray>(&g_external_memory_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory external_memory) {
    return external_memory == nullptr ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuArrayCreate_v2(CUarray* array,
                                             const CUDA_ARRAY_DESCRIPTOR* descriptor) {
    if (array == nullptr || descriptor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *array = reinterpret_cast<CUarray>(&g_array_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuArrayCreate(CUarray* array, const CUDA_ARRAY_DESCRIPTOR* descriptor) {
    return cuArrayCreate_v2(array, descriptor);
}

extern "C" CUresult CUDAAPI cuArray3DCreate_v2(CUarray* array,
                                               const CUDA_ARRAY3D_DESCRIPTOR* descriptor) {
    if (array == nullptr || descriptor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *array = reinterpret_cast<CUarray>(&g_array_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuArray3DCreate(CUarray* array,
                                            const CUDA_ARRAY3D_DESCRIPTOR* descriptor) {
    return cuArray3DCreate_v2(array, descriptor);
}

extern "C" CUresult CUDAAPI cuArrayDestroy(CUarray array) {
    return array == nullptr ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray* mipmap,
                                                   const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                                   unsigned int level_count) {
    if (mipmap == nullptr || descriptor == nullptr || level_count == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *mipmap = reinterpret_cast<CUmipmappedArray>(&g_array_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray mipmap) {
    return mipmap == nullptr ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
    return resource == nullptr ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(CUarray* array,
                                                                CUgraphicsResource resource,
                                                                unsigned int, unsigned int) {
    if (array == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *array = reinterpret_cast<CUarray>(&g_graphics_resource_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* mipmap,
                                                                      CUgraphicsResource resource) {
    if (mipmap == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *mipmap = reinterpret_cast<CUmipmappedArray>(&g_graphics_resource_token);
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* device_pointer,
                                                                  std::size_t* size,
                                                                  CUgraphicsResource resource) {
    if (device_pointer == nullptr || size == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device_pointer = 0x63000000U;
    *size = 4096;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr* device_pointer,
                                                               std::size_t* size,
                                                               CUgraphicsResource resource) {
    return cuGraphicsResourceGetMappedPointer_v2(device_pointer, size, resource);
}

extern "C" CUresult CUDAAPI cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                                             unsigned int) {
    return resource == nullptr ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                                          unsigned int flags) {
    return cuGraphicsResourceSetMapFlags_v2(resource, flags);
}

extern "C" CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                                   CUgraphicsResource* resources, CUstream) {
    return count != 0 && resources == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                                     CUgraphicsResource* resources, CUstream) {
    return count != 0 && resources == nullptr ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
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

namespace {
CUresult CUDAAPI native_stream_get_context(CUstream, CUcontext* context) {
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
CUresult CUDAAPI native_stream_get_context_ptsz(CUstream stream, CUcontext* context) {
    return native_stream_get_context(stream, context);
}

#if CUDA_VERSION >= 12050
CUresult CUDAAPI native_stream_get_context_v2(CUstream stream, CUcontext* context,
                                              CUgreenCtx* green_context) {
    if (!green_context) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto result = native_stream_get_context(stream, context);
    if (result == CUDA_SUCCESS) {
        *green_context = nullptr;
    }
    return result;
}
CUresult CUDAAPI native_stream_get_context_v2_ptsz(CUstream stream, CUcontext* context,
                                                   CUgreenCtx* green_context) {
    return native_stream_get_context_v2(stream, context, green_context);
}
#endif
}  // namespace

extern "C" CUresult CUDAAPI cuStreamGetCtx(CUstream stream, CUcontext* context) {
    return native_stream_get_context(stream, context);
}
extern "C" CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream stream, CUcontext* context) {
    return native_stream_get_context_ptsz(stream, context);
}
#if CUDA_VERSION >= 12050
extern "C" CUresult CUDAAPI cuStreamGetCtx_v2(CUstream stream, CUcontext* context,
                                              CUgreenCtx* green_context) {
    return native_stream_get_context_v2(stream, context, green_context);
}
extern "C" CUresult CUDAAPI cuStreamGetCtx_v2_ptsz(CUstream stream, CUcontext* context,
                                                   CUgreenCtx* green_context) {
    return native_stream_get_context_v2_ptsz(stream, context, green_context);
}
#endif

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

extern "C" CUresult CUDAAPI cuEventCreate(CUevent* event, unsigned int flags) {
    if (std::getenv("GLIMMER_FAKE_EVENT_AUDIT") != nullptr) {
        glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kCreate, event,
                                          nullptr, nullptr, nullptr, flags);
        return static_cast<CUresult>(g_event_audit.result);
    }
    if (event == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *event = reinterpret_cast<CUevent>(&g_event_token);
    ++g_internal_events;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream) {
    if (std::getenv("GLIMMER_FAKE_EVENT_AUDIT") != nullptr) {
        glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecord,
                                          nullptr, event, stream, nullptr, 0);
        return static_cast<CUresult>(g_event_audit.result);
    }
    return event == reinterpret_cast<CUevent>(&g_event_token) ? CUDA_SUCCESS
                                                              : CUDA_ERROR_INVALID_HANDLE;
}

extern "C" CUresult CUDAAPI cuEventQuery(CUevent event) {
    if (std::getenv("GLIMMER_FAKE_EVENT_AUDIT") != nullptr) {
        glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kQuery, nullptr,
                                          event, nullptr, nullptr, 0);
        return static_cast<CUresult>(g_event_audit.result);
    }
    return event == reinterpret_cast<CUevent>(&g_event_token) ? CUDA_SUCCESS
                                                              : CUDA_ERROR_INVALID_HANDLE;
}

extern "C" CUresult CUDAAPI cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode* mode) {
    thread_local CUstreamCaptureMode current = CU_STREAM_CAPTURE_MODE_GLOBAL;
    if (mode == nullptr ||
        (*mode != CU_STREAM_CAPTURE_MODE_GLOBAL && *mode != CU_STREAM_CAPTURE_MODE_THREAD_LOCAL &&
         *mode != CU_STREAM_CAPTURE_MODE_RELAXED)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto previous = current;
    current = *mode;
    *mode = previous;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuEventDestroy_v2(CUevent event) {
    if (std::getenv("GLIMMER_FAKE_EVENT_AUDIT") != nullptr) {
        glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kDestroy,
                                          nullptr, event, nullptr, nullptr, 0);
        return static_cast<CUresult>(g_event_audit.result);
    }
    return event == reinterpret_cast<CUevent>(&g_event_token) ? CUDA_SUCCESS
                                                              : CUDA_ERROR_INVALID_HANDLE;
}

extern "C" glimmer::tests::EventAudit* glimmer_fake_event_audit() {
    return &g_event_audit;
}

extern "C" CUresult CUDAAPI cuEventRecord_ptsz(CUevent event, CUstream stream) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecordPtds,
                                      nullptr, event, stream, nullptr, 0);
    return static_cast<CUresult>(g_event_audit.result);
}

extern "C" CUresult CUDAAPI cuEventRecordWithFlags(CUevent event, CUstream stream,
                                                   unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecordFlags,
                                      nullptr, event, stream, nullptr, flags);
    return static_cast<CUresult>(g_event_audit.result);
}

extern "C" CUresult CUDAAPI cuEventRecordWithFlags_ptsz(CUevent event, CUstream stream,
                                                        unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kRecordFlagsPtds,
                                      nullptr, event, stream, nullptr, flags);
    return static_cast<CUresult>(g_event_audit.result);
}

extern "C" CUresult CUDAAPI cuEventSynchronize(CUevent event) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kSynchronize,
                                      nullptr, event, nullptr, nullptr, 0);
    return static_cast<CUresult>(g_event_audit.result);
}

extern "C" CUresult CUDAAPI cuEventElapsedTime(float* milliseconds, CUevent start, CUevent end) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kElapsed,
                                      milliseconds, start, nullptr, end, 0);
    return static_cast<CUresult>(g_event_audit.result);
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kWait, nullptr,
                                      event, stream, nullptr, flags);
    return static_cast<CUresult>(g_event_audit.result);
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream stream, CUevent event,
                                                   unsigned int flags) {
    glimmer::tests::record_event_call(g_event_audit, glimmer::tests::EventCall::kWaitPtds, nullptr,
                                      event, stream, nullptr, flags);
    return static_cast<CUresult>(g_event_audit.result);
}

extern "C" CUresult CUDAAPI cuEventDestroy(CUevent event) {
    return cuEventDestroy_v2(event);
}

extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    if (std::getenv("GLIMMER_FAKE_COLD_RUNTIME_CAPACITY") != nullptr) {
        ++g_capacity_memory_queries;
        if (!g_is_cold_primary_initialized) {
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        }
    }
    *free_bytes = kPhysicalMemoryBytes - g_used_bytes;
    *total_bytes = kPhysicalMemoryBytes;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice device, unsigned int* flags,
                                                       int* active) {
    if (device != 0 || !flags || !active) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    *flags = 0;
    *active = g_context_alive ? 1 : 0;
    if (std::getenv("GLIMMER_FAKE_COLD_RUNTIME_CAPACITY") != nullptr &&
        !g_is_cold_primary_initialized) {
        *active = 0;
        if (std::getenv("GLIMMER_FAKE_COLD_PRIMARY_UNKNOWN") != nullptr) {
            *active = 2;
        }
        if (std::getenv("GLIMMER_FAKE_COLD_PRIMARY_FAILURE") != nullptr) {
            return CUDA_ERROR_INVALID_DEVICE;
        }
    }
    return CUDA_SUCCESS;
}

extern "C" unsigned int glimmer_test_capacity_allocation_calls() {
    std::scoped_lock lock(g_mutex);
    return g_capacity_allocation_calls;
}

extern "C" unsigned int glimmer_test_capacity_memory_queries() {
    std::scoped_lock lock(g_mutex);
    return g_capacity_memory_queries;
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
    if (std::getenv("GLIMMER_FAKE_PROGRESS_CONTEXT_ERROR") != nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    if (std::getenv("GLIMMER_FAKE_UNBOUND_CAPACITY") != nullptr) {
        *context = nullptr;
        return CUDA_SUCCESS;
    }
    std::scoped_lock lock(g_mutex);
    *context = g_context_alive ? fake_context() : nullptr;
    return CUDA_SUCCESS;
}

extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    g_context_device_queries.fetch_add(1, std::memory_order_relaxed);
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    std::scoped_lock lock(g_mutex);
    if (!g_context_alive || std::getenv("GLIMMER_FAKE_UNBOUND_CAPACITY") != nullptr) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    *device = 0;
    return CUDA_SUCCESS;
}

extern "C" unsigned int glimmer_test_context_device_queries() {
    return g_context_device_queries.load(std::memory_order_relaxed);
}

namespace {
using DestroyQuery = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t);
using DestroyQueryV2 = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t,
                                          CUdriverProcAddressQueryResult*);
thread_local DestroyQuery g_destroy_query{};
thread_local DestroyQueryV2 g_destroy_query_v2{};

CUresult CUDAAPI native_context_destroy_legacy(CUcontext) {
    return CUDA_ERROR_INVALID_HANDLE;
}

CUresult CUDAAPI native_context_destroy_modern(CUcontext context) {
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
}  // namespace

extern "C" void glimmer_test_set_destroy_query_callbacks(DestroyQuery query,
                                                         DestroyQueryV2 query_v2) {
    g_destroy_query = query;
    g_destroy_query_v2 = query_v2;
}

extern "C" CUresult CUDAAPI cuCtxDestroy(CUcontext context) {
    return native_context_destroy_legacy(context);
}

extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context) {
    return native_context_destroy_modern(context);
}

extern "C" void* glimmer_test_graph_symbol(const char*, int, cuuint64_t);
namespace {
CUresult CUDAAPI native_primary_reset_legacy(CUdevice device) {
    return device == 0 ? CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE : CUDA_ERROR_INVALID_DEVICE;
}

CUresult CUDAAPI native_primary_reset_modern(CUdevice device) {
    return device == 0 ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_INVALID_DEVICE;
}

const char* versioned_memory_symbol(const char* symbol, int version) {
    if (version >= 10010 && symbol) {
        if (std::string_view{symbol} == "cuStreamBeginCapture") {
            return "cuStreamBeginCapture_v2";
        }
        if (std::string_view{symbol} == "cuStreamBeginCapture_ptsz") {
            return "cuStreamBeginCapture_v2_ptsz";
        }
    }
    return symbol;
}
CUresult CUDAAPI native_primary_release_legacy(CUdevice) {
    return CUDA_ERROR_INVALID_DEVICE;
}
CUresult CUDAAPI native_primary_release_modern(CUdevice) {
    return CUDA_ERROR_INVALID_CONTEXT;
}
CUresult CUDAAPI native_get_proc_address(const char* symbol, void** function_pointer, int version,
                                         cuuint64_t flags) {
    if (symbol == nullptr || function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (g_destroy_query && (std::string_view(symbol) == "cuCtxDestroy" ||
                            std::string_view(symbol) == "cuCtxDestroy_v2")) {
        return g_destroy_query(symbol, function_pointer, version, flags);
    }
    if (std::string_view(symbol) == "cuCtxDestroy") {
        if (version < 2000) {
            return CUDA_ERROR_NOT_FOUND;
        }
        *function_pointer = version < 4000
                                ? reinterpret_cast<void*>(&native_context_destroy_legacy)
                                : reinterpret_cast<void*>(&native_context_destroy_modern);
        return CUDA_SUCCESS;
    }
    if (std::string_view(symbol) == "cuCtxDestroy_v2") {
        return CUDA_ERROR_NOT_FOUND;
    }
    if (std::string_view(symbol) == "cuDevicePrimaryCtxRelease" && version >= 7000) {
        *function_pointer = version < 11000
                                ? reinterpret_cast<void*>(&native_primary_release_legacy)
                                : reinterpret_cast<void*>(&native_primary_release_modern);
        return CUDA_SUCCESS;
    }
    if ((std::string_view(symbol) == "cuDevicePrimaryCtxRelease" && version < 7000) ||
        (std::string_view(symbol) == "cuDevicePrimaryCtxRelease_v2" && version < 11000)) {
        *function_pointer = nullptr;
        return CUDA_ERROR_NOT_FOUND;
    }
    if (std::string_view(symbol) == "cuDevicePrimaryCtxReset" && version >= 7000) {
        *function_pointer = version < 11000 ? reinterpret_cast<void*>(&native_primary_reset_legacy)
                                            : reinterpret_cast<void*>(&native_primary_reset_modern);
        return CUDA_SUCCESS;
    }
    if ((std::string_view(symbol) == "cuDevicePrimaryCtxReset" && version < 7000) ||
        (std::string_view(symbol) == "cuDevicePrimaryCtxReset_v2" && version < 11000)) {
        *function_pointer = nullptr;
        return CUDA_ERROR_NOT_FOUND;
    }
#if CUDA_VERSION >= 12050
    if ((std::string_view(symbol) == "cuStreamGetCtx" ||
         std::string_view(symbol) == "cuStreamGetCtx_ptsz") &&
        version >= 12050) {
        const bool is_ptds = (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0 ||
                             std::string_view(symbol) == "cuStreamGetCtx_ptsz";
        *function_pointer = is_ptds ? reinterpret_cast<void*>(&native_stream_get_context_v2_ptsz)
                                    : reinterpret_cast<void*>(&native_stream_get_context_v2);
        return CUDA_SUCCESS;
    }
#endif
    if (std::string_view(symbol) == "cuStreamIsCapturing" ||
        std::string_view(symbol) == "cuStreamIsCapturing_ptsz") {
        const bool is_ptds = (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0 ||
                             std::string_view(symbol) == "cuStreamIsCapturing_ptsz";
        *function_pointer = is_ptds ? reinterpret_cast<void*>(&cuStreamIsCapturing_ptsz)
                                    : reinterpret_cast<void*>(&cuStreamIsCapturing);
        return CUDA_SUCCESS;
    }
    if (std::getenv("GLIMMER_FAKE_SELF_COPY_LOOKUP") != nullptr &&
        std::string_view{symbol} == "cuMemcpyHtoDAsync") {
        *function_pointer = dlsym(RTLD_DEFAULT, "cuMemcpyHtoDAsync_v2");
        return *function_pointer ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
    }
    if (std::string_view(symbol) == "glimmer_test_progress_accounting_callback") {
        *function_pointer = dlsym(RTLD_DEFAULT, symbol);
        return *function_pointer == nullptr ? CUDA_ERROR_NOT_FOUND
                                            : reinterpret_cast<CUresult (*)()>(*function_pointer)();
    }
    *function_pointer = glimmer_test_memory_symbol(versioned_memory_symbol(symbol, version), flags);
    if (*function_pointer == nullptr) {
        *function_pointer = glimmer_test_graph_symbol(symbol, version, flags);
    }
    if (*function_pointer == nullptr) {
        *function_pointer = lookup_symbol(symbol);
    }
    return *function_pointer == nullptr ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
}
}  // namespace

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice device) {
    return native_primary_reset_legacy(device);
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice device) {
    return native_primary_reset_modern(device);
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice device) {
    return native_primary_release_legacy(device);
}
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice device) {
    return native_primary_release_modern(device);
}

extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** function_pointer,
                                             int version, cuuint64_t flags) {
    return native_get_proc_address(symbol, function_pointer, version, flags);
}

extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** function_pointer,
                                                int version, cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status) {
    if (symbol != nullptr && function_pointer != nullptr &&
        (std::string_view(symbol) == "cuCtxDestroy" ||
         std::string_view(symbol) == "cuCtxDestroy_v2")) {
        if (g_destroy_query_v2) {
            return g_destroy_query_v2(symbol, function_pointer, version, flags, symbol_status);
        }
        const bool is_base = std::string_view(symbol) == "cuCtxDestroy";
        *function_pointer = !is_base || version < 2000 ? nullptr
                            : version < 4000
                                ? reinterpret_cast<void*>(&native_context_destroy_legacy)
                                : reinterpret_cast<void*>(&native_context_destroy_modern);
        if (symbol_status != nullptr) {
            *symbol_status = !is_base         ? CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
                             : version < 2000 ? CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT
                                              : CU_GET_PROC_ADDRESS_SUCCESS;
        }
        return CUDA_SUCCESS;
    }
    if (symbol != nullptr && function_pointer != nullptr) {
        *function_pointer =
            glimmer_test_memory_symbol(versioned_memory_symbol(symbol, version), flags);
        if (*function_pointer == nullptr) {
            *function_pointer = glimmer_test_graph_symbol(symbol, version, flags);
        }
        if (*function_pointer != nullptr) {
            if (symbol_status != nullptr) {
                *symbol_status = CU_GET_PROC_ADDRESS_SUCCESS;
            }
            return CUDA_SUCCESS;
        }
    }
    const CUresult result = native_get_proc_address(symbol, function_pointer, version, flags);
    if (symbol_status != nullptr) {
        *symbol_status = result == CUDA_SUCCESS ? CU_GET_PROC_ADDRESS_SUCCESS
                                                : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    }
    return result;
}

namespace {

// Constructor-routing tests own no GPU frontend. These opt-in dispatch entries
// allow CPU router initialization, but reject every unmodeled resource effect.
CUresult CUDAAPI unavailable_context_push(CUcontext) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_context_pop(CUcontext*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_stream_create(CUstream*, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_stream_flags(CUstream, unsigned int*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_stream_priority(CUstream, int*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_host_allocate(void**, std::size_t, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_host_free(void*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_host_map(CUdeviceptr*, void*, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_pointer_attribute(void*, CUpointer_attribute, CUdeviceptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI unavailable_host_flags(unsigned int*, void*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI empty_graph_nodes(CUgraph graph, CUgraphNode*, std::size_t* count) {
    if (graph != reinterpret_cast<CUgraph>(0xfa020) || !count) return CUDA_ERROR_INVALID_VALUE;
    *count = 0;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI empty_graph_node_type(CUgraphNode, CUgraphNodeType*) {
    return CUDA_ERROR_INVALID_VALUE;
}

void* lookup_symbol(const char* symbol) {
    if (symbol == nullptr) {
        return nullptr;
    }
    if (std::getenv("GLIMMER_FAKE_GRAPH_OWNER_DISPATCH") != nullptr) {
        const std::string_view name{symbol};
        if (name == "cuGraphGetNodes") return reinterpret_cast<void*>(&empty_graph_nodes);
        if (name == "cuGraphNodeGetType") return reinterpret_cast<void*>(&empty_graph_node_type);
        if (name == "cuCtxPushCurrent") return reinterpret_cast<void*>(&unavailable_context_push);
        if (name == "cuCtxPopCurrent") return reinterpret_cast<void*>(&unavailable_context_pop);
        if (name == "cuStreamCreate") return reinterpret_cast<void*>(&unavailable_stream_create);
        if (name == "cuStreamGetFlags") return reinterpret_cast<void*>(&unavailable_stream_flags);
        if (name == "cuStreamGetPriority")
            return reinterpret_cast<void*>(&unavailable_stream_priority);
        if (name == "cuMemHostAlloc") return reinterpret_cast<void*>(&unavailable_host_allocate);
        if (name == "cuMemFreeHost") return reinterpret_cast<void*>(&unavailable_host_free);
        if (name == "cuMemHostGetDevicePointer")
            return reinterpret_cast<void*>(&unavailable_host_map);
        if (name == "cuPointerGetAttribute")
            return reinterpret_cast<void*>(&unavailable_pointer_attribute);
        if (name == "cuMemHostGetFlags") return reinterpret_cast<void*>(&unavailable_host_flags);
    }
    if (std::string_view(symbol) == "cuInit") {
        return reinterpret_cast<void*>(&cuInit);
    }
    if (std::string_view(symbol) == "cuDevicePrimaryCtxReset") {
        return reinterpret_cast<void*>(&native_primary_reset_legacy);
    }
    if (std::string_view(symbol) == "cuDevicePrimaryCtxReset_v2") {
        return reinterpret_cast<void*>(&native_primary_reset_modern);
    }
    if (std::string_view(symbol) == "cuDevicePrimaryCtxRelease") {
        return reinterpret_cast<void*>(&native_primary_release_legacy);
    }
    if (std::string_view(symbol) == "cuDevicePrimaryCtxRelease_v2") {
        return reinterpret_cast<void*>(&native_primary_release_modern);
    }
    if (std::string_view(symbol) == "cuModuleLoadData") {
        return reinterpret_cast<void*>(&cuModuleLoadData);
    }
    if (std::string_view(symbol) == "cuModuleLoadDataEx") {
        return reinterpret_cast<void*>(&cuModuleLoadDataEx);
    }
    if (std::string_view(symbol) == "cuModuleGetFunction") {
        return reinterpret_cast<void*>(&cuModuleGetFunction);
    }
    if (std::string_view(symbol) == "cuLibraryLoadData")
        return reinterpret_cast<void*>(&cuLibraryLoadData);
    if (std::string_view(symbol) == "cuLibraryLoadFromFile")
        return reinterpret_cast<void*>(&cuLibraryLoadFromFile);
    if (std::string_view(symbol) == "cuLibraryGetModule")
        return reinterpret_cast<void*>(&cuLibraryGetModule);
    if (std::string_view(symbol) == "cuLibraryUnload")
        return reinterpret_cast<void*>(&cuLibraryUnload);
    if (std::string_view(symbol) == "cuModuleUnload") {
        return reinterpret_cast<void*>(&cuModuleUnload);
    }
    if (std::string_view(symbol) == "cuLaunchKernel") {
        return reinterpret_cast<void*>(&cuLaunchKernel);
    }
    if (std::string_view(symbol) == "cuLaunchKernel_ptsz") {
        return reinterpret_cast<void*>(&cuLaunchKernel_ptsz);
    }
    if (std::string_view(symbol) == "cuGraphLaunch") {
        return reinterpret_cast<void*>(&cuGraphLaunch);
    }
    if (std::string_view(symbol) == "cuLaunchKernelEx") {
        return reinterpret_cast<void*>(&cuLaunchKernelEx);
    }
    if (std::string_view(symbol) == "cuLaunchKernelEx_ptsz") {
        return reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz);
    }
    if (std::string_view(symbol) == "cuGraphLaunch_ptsz") {
        return reinterpret_cast<void*>(&cuGraphLaunch_ptsz);
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
    if (std::string_view(symbol) == "cuMemMapArrayAsync") {
        return reinterpret_cast<void*>(&cuMemMapArrayAsync);
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
    if (std::string_view(symbol) == "cuIpcGetMemHandle") {
        return reinterpret_cast<void*>(&cuIpcGetMemHandle);
    }
    if (std::string_view(symbol) == "cuIpcOpenMemHandle") {
        return reinterpret_cast<void*>(&cuIpcOpenMemHandle);
    }
    if (std::string_view(symbol) == "cuIpcOpenMemHandle_v2") {
        return reinterpret_cast<void*>(&cuIpcOpenMemHandle_v2);
    }
    if (std::string_view(symbol) == "cuIpcCloseMemHandle") {
        return reinterpret_cast<void*>(&cuIpcCloseMemHandle);
    }
    if (std::string_view(symbol) == "cuImportExternalMemory") {
        return reinterpret_cast<void*>(&cuImportExternalMemory);
    }
    if (std::string_view(symbol) == "cuExternalMemoryGetMappedBuffer") {
        return reinterpret_cast<void*>(&cuExternalMemoryGetMappedBuffer);
    }
    if (std::string_view(symbol) == "cuExternalMemoryGetMappedMipmappedArray") {
        return reinterpret_cast<void*>(&cuExternalMemoryGetMappedMipmappedArray);
    }
    if (std::string_view(symbol) == "cuDestroyExternalMemory") {
        return reinterpret_cast<void*>(&cuDestroyExternalMemory);
    }
    if (std::string_view(symbol) == "cuArrayCreate" ||
        std::string_view(symbol) == "cuArrayCreate_v2") {
        return reinterpret_cast<void*>(&cuArrayCreate);
    }
    if (std::string_view(symbol) == "cuArray3DCreate" ||
        std::string_view(symbol) == "cuArray3DCreate_v2") {
        return reinterpret_cast<void*>(&cuArray3DCreate);
    }
    if (std::string_view(symbol) == "cuArrayDestroy") {
        return reinterpret_cast<void*>(&cuArrayDestroy);
    }
    if (std::string_view(symbol) == "cuMipmappedArrayCreate") {
        return reinterpret_cast<void*>(&cuMipmappedArrayCreate);
    }
    if (std::string_view(symbol) == "cuMipmappedArrayDestroy") {
        return reinterpret_cast<void*>(&cuMipmappedArrayDestroy);
    }
    if (std::string_view(symbol) == "cuGraphicsUnregisterResource") {
        return reinterpret_cast<void*>(&cuGraphicsUnregisterResource);
    }
    if (std::string_view(symbol) == "cuGraphicsSubResourceGetMappedArray") {
        return reinterpret_cast<void*>(&cuGraphicsSubResourceGetMappedArray);
    }
    if (std::string_view(symbol) == "cuGraphicsResourceGetMappedMipmappedArray") {
        return reinterpret_cast<void*>(&cuGraphicsResourceGetMappedMipmappedArray);
    }
    if (std::string_view(symbol) == "cuGraphicsResourceGetMappedPointer" ||
        std::string_view(symbol) == "cuGraphicsResourceGetMappedPointer_v2") {
        return reinterpret_cast<void*>(&cuGraphicsResourceGetMappedPointer_v2);
    }
    if (std::string_view(symbol) == "cuGraphicsResourceSetMapFlags" ||
        std::string_view(symbol) == "cuGraphicsResourceSetMapFlags_v2") {
        return reinterpret_cast<void*>(&cuGraphicsResourceSetMapFlags_v2);
    }
    if (std::string_view(symbol) == "cuGraphicsMapResources") {
        return reinterpret_cast<void*>(&cuGraphicsMapResources);
    }
    if (std::string_view(symbol) == "cuGraphicsUnmapResources") {
        return reinterpret_cast<void*>(&cuGraphicsUnmapResources);
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
    if (std::string_view(symbol) == "cuCtxDestroy") {
        return reinterpret_cast<void*>(&native_context_destroy_legacy);
    }
    if (std::string_view(symbol) == "cuCtxDestroy_v2") {
        return reinterpret_cast<void*>(&native_context_destroy_modern);
    }
    if (std::string_view(symbol) == "cuStreamGetDevice") {
        return reinterpret_cast<void*>(&cuStreamGetDevice);
    }
    if (std::string_view(symbol) == "cuStreamGetDevice_ptsz") {
        return reinterpret_cast<void*>(&cuStreamGetDevice_ptsz);
    }
    if (std::string_view(symbol) == "cuStreamGetCtx") {
        return reinterpret_cast<void*>(&native_stream_get_context);
    }
    if (std::string_view(symbol) == "cuStreamGetCtx_ptsz") {
        return reinterpret_cast<void*>(&native_stream_get_context_ptsz);
    }
    if (std::string_view(symbol) == "cuStreamIsCapturing") {
        return reinterpret_cast<void*>(&cuStreamIsCapturing);
    }
    if (std::string_view(symbol) == "cuStreamIsCapturing_ptsz") {
        return reinterpret_cast<void*>(&cuStreamIsCapturing_ptsz);
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
    if (std::string_view(symbol) == "cuEventCreate") {
        return reinterpret_cast<void*>(&cuEventCreate);
    }
    if (std::string_view(symbol) == "cuEventRecord") {
        return reinterpret_cast<void*>(&cuEventRecord);
    }
    if (std::string_view(symbol) == "cuEventQuery") {
        return reinterpret_cast<void*>(&cuEventQuery);
    }
    if (std::string_view(symbol) == "cuThreadExchangeStreamCaptureMode") {
        return reinterpret_cast<void*>(&cuThreadExchangeStreamCaptureMode);
    }
    if (std::string_view(symbol) == "cuEventDestroy_v2") {
        return reinterpret_cast<void*>(&cuEventDestroy_v2);
    }
    if (std::string_view(symbol) == "cuEventRecord_ptsz") {
        return reinterpret_cast<void*>(&cuEventRecord_ptsz);
    }
    if (std::string_view(symbol) == "cuEventRecordWithFlags") {
        return reinterpret_cast<void*>(&cuEventRecordWithFlags);
    }
    if (std::string_view(symbol) == "cuEventRecordWithFlags_ptsz") {
        return reinterpret_cast<void*>(&cuEventRecordWithFlags_ptsz);
    }
    if (std::string_view(symbol) == "cuEventSynchronize") {
        return reinterpret_cast<void*>(&cuEventSynchronize);
    }
    if (std::string_view(symbol) == "cuEventElapsedTime") {
        return reinterpret_cast<void*>(&cuEventElapsedTime);
    }
    if (std::string_view(symbol) == "cuStreamWaitEvent") {
        return reinterpret_cast<void*>(&cuStreamWaitEvent);
    }
    if (std::string_view(symbol) == "cuStreamWaitEvent_ptsz") {
        return reinterpret_cast<void*>(&cuStreamWaitEvent_ptsz);
    }
    if (std::string_view(symbol) == "cuEventDestroy") {
        return reinterpret_cast<void*>(&cuEventDestroy);
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
