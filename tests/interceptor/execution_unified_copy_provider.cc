#include <cuda.h>

#include <atomic>
#include <cstddef>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif

namespace {
std::atomic<unsigned int> calls{};
CUresult unexpected() {
    ++calls;
    return CUDA_ERROR_NOT_SUPPORTED;
}
}  // namespace

extern "C" unsigned int glimmer_unified_copy_native_calls() {
    return calls.load();
}
extern "C" CUresult CUDAAPI cuMemcpy(CUdeviceptr, CUdeviceptr, std::size_t) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr, CUdeviceptr, std::size_t) {
    return unexpected();
}
extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr, CUdeviceptr, std::size_t, CUstream) {
    return unexpected();
}
// This parameter order is fixed by the original CUDA Driver ABI.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int version,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* status) {
    if (!name || !output || flags > CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
        return CUDA_ERROR_INVALID_VALUE;
    *output = nullptr;
    if (status) *status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    if (version < 4000) {
        if (status) *status = CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;
        return CUDA_SUCCESS;
    }
    const bool is_per_thread = flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
    if (std::string_view{name} == "cuMemcpy")
        *output = is_per_thread ? reinterpret_cast<void*>(&cuMemcpy_ptds)
                                : reinterpret_cast<void*>(&cuMemcpy);
    else if (std::string_view{name} == "cuMemcpyAsync")
        *output = is_per_thread ? reinterpret_cast<void*>(&cuMemcpyAsync_ptsz)
                                : reinterpret_cast<void*>(&cuMemcpyAsync);
    if (status && *output) *status = CU_GET_PROC_ADDRESS_SUCCESS;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    return cuGetProcAddress_v2(name, output, version, flags, nullptr);
}
