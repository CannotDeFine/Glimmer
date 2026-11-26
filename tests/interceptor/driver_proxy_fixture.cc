#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int,
                                           CUstream, void**, void**) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char*, void**, int, cuuint64_t,
                                                CUdriverProcAddressQueryResult*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}

extern "C" cudaError_t CUDARTAPI cudaGetLastError() {
    return cudaErrorNotSupported;
}

extern "C" int cuGlimmerProxyValue() {
    return 42;
}
