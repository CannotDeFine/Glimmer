#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
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
std::size_t g_used_bytes = 0;
CUdeviceptr g_next_pointer = 0x100000U;
bool g_context_alive = true;
std::uint8_t g_context_token = 0;

CUcontext fake_context() {
    return reinterpret_cast<CUcontext>(&g_context_token);
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
