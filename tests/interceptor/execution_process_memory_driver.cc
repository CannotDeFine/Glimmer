#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

namespace {
std::atomic<unsigned int> native_calls{};
CUresult unexpected() {
    ++native_calls;
    return CUDA_ERROR_INVALID_CONTEXT;
}
}  // namespace

extern "C" unsigned int glimmer_process_memory_native_calls() {
    return native_calls.load();
}
extern "C" CUresult CUDAAPI cuInit(unsigned int flags) {
    return flags ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* output) {
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    *output = nullptr;
    return CUDA_SUCCESS;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact external SDK ABI.
extern "C" CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr*, std::size_t, CUstream) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr*, std::size_t, CUmemoryPool,
                                                    CUstream) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool*, CUdevice) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool*, CUdevice) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuDeviceSetMemPool(CUdevice, CUmemoryPool) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuPointerGetAttribute(void*, CUpointer_attribute, CUdeviceptr) {
    return unexpected();
}
extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t, std::size_t) {
    ++native_calls;
    return cudaErrorInvalidResourceHandle;
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    if (!name || !output) return CUDA_ERROR_INVALID_VALUE;
    if (flags) return CUDA_ERROR_NOT_SUPPORTED;
    if (version < 11020) return CUDA_ERROR_NOT_FOUND;
    const std::string_view symbol{name};
#define PROCESS_MEMORY_ENTRY(NAME, FUNCTION)            \
    if (symbol == (NAME)) {                             \
        *output = reinterpret_cast<void*>(&(FUNCTION)); \
        return CUDA_SUCCESS;                            \
    }
    PROCESS_MEMORY_ENTRY("cuMemAllocAsync", cuMemAllocAsync)
    PROCESS_MEMORY_ENTRY("cuMemAllocFromPoolAsync", cuMemAllocFromPoolAsync)
    PROCESS_MEMORY_ENTRY("cuDeviceGetDefaultMemPool", cuDeviceGetDefaultMemPool)
    PROCESS_MEMORY_ENTRY("cuDeviceGetMemPool", cuDeviceGetMemPool)
    PROCESS_MEMORY_ENTRY("cuDeviceSetMemPool", cuDeviceSetMemPool)
    PROCESS_MEMORY_ENTRY("cuMemFree", cuMemFree_v2)
    PROCESS_MEMORY_ENTRY("cuPointerGetAttribute", cuPointerGetAttribute)
#undef PROCESS_MEMORY_ENTRY
    return CUDA_ERROR_NOT_FOUND;
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int version,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* query_status) {
    const auto result = cuGetProcAddress(name, output, version, flags);
    if (query_status)
        *query_status = result == CUDA_SUCCESS ? CU_GET_PROC_ADDRESS_SUCCESS
                                               : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return result;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
