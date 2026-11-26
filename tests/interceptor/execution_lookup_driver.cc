#include <cuda.h>

#include <atomic>
#include <string_view>
#include <dlfcn.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuLaunchKernel
#undef cuLaunchKernel
#endif
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);
namespace {
std::atomic<unsigned int> calls[4];
CUresult CUDAAPI old_allocate(unsigned int*, unsigned int) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI other_entry() {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI threaded_launch(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int, CUstream, void**,
                                 void**) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
}  // namespace
extern "C" {
unsigned int glimmer_execution_lookup_count(unsigned int index) {
    return index < 4 ? calls[index].load(std::memory_order_relaxed) : 0;
}
CUresult CUDAAPI cuInit(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDeviceGetCount(int* value) {
    ++calls[3];
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    *value = 2;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDriverGetVersion(int* value) {
    ++calls[3];
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    *value = 12080;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetCurrent(CUcontext* value) {
    ++calls[3];  // Read-only initialization metadata, not a GPU resource call.
    if (value) *value = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxCreate_v2(CUcontext*, unsigned int, CUdevice) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuCtxGetLimit(std::size_t*, CUlimit) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuCtxCreate_v3(CUcontext*, CUexecAffinityParam*, int, unsigned int, CUdevice) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuCtxCreate_v4(CUcontext*, CUctxCreateParams*, unsigned int, CUdevice) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr*, std::size_t) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuModuleGetGlobal_v2(CUdeviceptr*, std::size_t*, CUmodule, const char*) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                                unsigned int, unsigned int, unsigned int, CUstream, void**,
                                void**) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuLookupUnsupported() {
    ++calls[2];
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuDevicePrimaryCtxRetain(CUcontext*, CUdevice) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice, unsigned int*, int*) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuDevicePrimaryCtxSetFlags_v2(CUdevice, unsigned int) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuCtxGetFlags(unsigned int*) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuCtxSetFlags(unsigned int) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuCtxGetDevice(CUdevice*) {
    ++calls[2];
    return CUDA_ERROR_NOT_SUPPORTED;
}
}
namespace {
struct LookupSelection {
    int version{};
    cuuint64_t flags{};
};
CUresult lookup(const char* name, void** output, LookupSelection selection,
                CUdriverProcAddressQueryResult* status) {
    const auto version = selection.version;
    const auto flags = selection.flags;
    if (!name || !output) return CUDA_ERROR_INVALID_VALUE;
    if (std::string_view(name) == "runtime_sdk_error") {
        *output = reinterpret_cast<void*>(&other_entry);
        return CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;
    }
    if (status) *status = CU_GET_PROC_ADDRESS_SUCCESS;
    *output = nullptr;
    const std::string_view symbol{name};
    if (symbol == "sdk_error") {
        *output = reinterpret_cast<void*>(&other_entry);
        return CUDA_ERROR_DEINITIALIZED;
    }
    if (symbol == "malformed" || symbol == "runtime_malformed") {
        *output = reinterpret_cast<void*>(&cuCtxGetCurrent);
        if (status) *status = CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT;
    } else if (symbol == "cuCtxGetCurrent")
        *output = reinterpret_cast<void*>(&cuCtxGetCurrent);
    else if (symbol == "cuMemAlloc")
        *output = version < 3020 ? reinterpret_cast<void*>(&old_allocate)
                                 : reinterpret_cast<void*>(&cuMemAlloc_v2);
    else if (symbol == "cuCtxCreate")
        *output = version >= 12050   ? reinterpret_cast<void*>(&cuCtxCreate_v4)
                  : version >= 11040 ? reinterpret_cast<void*>(&cuCtxCreate_v3)
                                     : reinterpret_cast<void*>(&cuCtxCreate_v2);
    else if (symbol == "cuLaunchKernel")
        *output = flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                      ? reinterpret_cast<void*>(&threaded_launch)
                      : reinterpret_cast<void*>(&cuLaunchKernel);
    else if (symbol == "cuGetProcAddress")
        *output = version >= 12000 ? reinterpret_cast<void*>(&cuGetProcAddress_v2)
                                   : reinterpret_cast<void*>(&cuGetProcAddress);
    else if (symbol == "cuLookupUnsupported")
        *output = reinterpret_cast<void*>(&cuLookupUnsupported);
    else if (symbol == "cuDevicePrimaryCtxRetain")
        *output = reinterpret_cast<void*>(&cuDevicePrimaryCtxRetain);
    else if (symbol == "cuDevicePrimaryCtxRelease" && version >= 11000)
        *output = reinterpret_cast<void*>(&cuDevicePrimaryCtxRelease_v2);
    else if (symbol == "cuDevicePrimaryCtxReset" && version >= 11000)
        *output = reinterpret_cast<void*>(&cuDevicePrimaryCtxReset_v2);
    else if (symbol == "cuDevicePrimaryCtxSetFlags" && version >= 11000)
        *output = reinterpret_cast<void*>(&cuDevicePrimaryCtxSetFlags_v2);
    else if (symbol == "cuDevicePrimaryCtxGetState")
        *output = reinterpret_cast<void*>(&cuDevicePrimaryCtxGetState);
    else if (symbol == "cuCtxGetFlags")
        *output = reinterpret_cast<void*>(&cuCtxGetFlags);
    else if (symbol == "cuCtxSetFlags")
        *output = reinterpret_cast<void*>(&cuCtxSetFlags);
    else if (symbol == "cuCtxGetDevice")
        *output = reinterpret_cast<void*>(&cuCtxGetDevice);
    else if (symbol == "cuCtxGetLimit")
        *output = reinterpret_cast<void*>(&cuCtxGetLimit);
    if (!*output && status)
        *status = symbol == "old_version" ? CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT
                                          : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_SUCCESS;
}
}  // namespace
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    ++calls[0];
    return lookup(name, output, {version, flags}, nullptr);
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int version,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* status) {
    ++calls[1];
    return lookup(name, output, {version, flags}, status);
}
extern "C" void* glimmer_execution_driver_internal_lookup(void* handle, const char* name) {
    return ::dlsym(handle, name);
}
extern "C" CUresult glimmer_execution_native_driver_entry(const char* name, void** output,
                                                          int version, cuuint64_t flags,
                                                          CUdriverProcAddressQueryResult* status) {
    return lookup(name, output, {version, flags}, status);
}
