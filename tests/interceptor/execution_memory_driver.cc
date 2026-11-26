#include <cuda.h>

#include <atomic>
#include <cstring>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
namespace {
std::atomic<unsigned int> native_memory_calls{};
CUresult unexpected_memory() {
    ++native_memory_calls;
    return CUDA_ERROR_INVALID_CONTEXT;
}
}  // namespace
extern "C" unsigned int glimmer_memory_native_calls() {
    return native_memory_calls.load();
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native CUDA ABI.
extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, std::size_t) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemHostAlloc(void**, std::size_t, unsigned int) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemAllocHost_v2(void**, std::size_t) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemFreeHost(void*) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemHostGetFlags(unsigned int*, void*) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr*, void*, unsigned int) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemHostRegister_v2(void*, std::size_t, unsigned int) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemHostUnregister(void*) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr, const void*, std::size_t, CUstream) {
    return unexpected_memory();
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void*, CUdeviceptr, std::size_t, CUstream) {
    return unexpected_memory();
}
#define MEMORY_FILL(BITS, TYPE)                                                                    \
    extern "C" CUresult CUDAAPI cuMemsetD##BITS##_v2(CUdeviceptr, TYPE, std::size_t) {             \
        return unexpected_memory();                                                                \
    }                                                                                              \
    extern "C" CUresult CUDAAPI cuMemsetD##BITS##Async(CUdeviceptr, TYPE, std::size_t, CUstream) { \
        return unexpected_memory();                                                                \
    }
MEMORY_FILL(8, unsigned char)
MEMORY_FILL(16, unsigned short)
MEMORY_FILL(32, unsigned int)
#undef MEMORY_FILL
// NOLINTEND(bugprone-easily-swappable-parameters)
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact native CUDA ABI.
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    if (!name || !output) return CUDA_ERROR_INVALID_VALUE;
    if (flags) return CUDA_ERROR_NOT_SUPPORTED;
    if (version < 3020) return CUDA_ERROR_NOT_FOUND;
    const std::string_view symbol{name};
#define MEMORY_ENTRY(NAME, FUNCTION)                    \
    if (symbol == (NAME)) {                             \
        *output = reinterpret_cast<void*>(&(FUNCTION)); \
        return CUDA_SUCCESS;                            \
    }
    MEMORY_ENTRY("cuMemcpyDtoD", cuMemcpyDtoD_v2)
    MEMORY_ENTRY("cuMemHostAlloc", cuMemHostAlloc)
    MEMORY_ENTRY("cuMemAllocHost", cuMemAllocHost_v2)
    MEMORY_ENTRY("cuMemFreeHost", cuMemFreeHost)
    MEMORY_ENTRY("cuMemHostGetFlags", cuMemHostGetFlags)
    MEMORY_ENTRY("cuMemHostGetDevicePointer", cuMemHostGetDevicePointer_v2)
    MEMORY_ENTRY("cuMemHostRegister", cuMemHostRegister_v2)
    MEMORY_ENTRY("cuMemHostUnregister", cuMemHostUnregister)
    MEMORY_ENTRY("cuMemcpyHtoDAsync", cuMemcpyHtoDAsync_v2)
    MEMORY_ENTRY("cuMemcpyDtoHAsync", cuMemcpyDtoHAsync_v2)
    MEMORY_ENTRY("cuMemcpyDtoDAsync", cuMemcpyDtoDAsync_v2)
    MEMORY_ENTRY("cuMemsetD8", cuMemsetD8_v2)
    MEMORY_ENTRY("cuMemsetD8Async", cuMemsetD8Async)
    MEMORY_ENTRY("cuMemsetD16", cuMemsetD16_v2)
    MEMORY_ENTRY("cuMemsetD16Async", cuMemsetD16Async)
    MEMORY_ENTRY("cuMemsetD32", cuMemsetD32_v2)
    MEMORY_ENTRY("cuMemsetD32Async", cuMemsetD32Async)
#undef MEMORY_ENTRY
    return CUDA_ERROR_NOT_FOUND;
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int version,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* status) {
    const auto result = cuGetProcAddress(name, output, version, flags);
    if (status)
        *status = result == CUDA_SUCCESS ? CU_GET_PROC_ADDRESS_SUCCESS
                                         : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return result;
}
