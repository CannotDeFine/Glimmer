#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

using DriverAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes);
using DriverFreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using DriverMemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);

template <typename Function>
Function resolve_driver_function(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

void* to_runtime_pointer(CUdeviceptr device_pointer) noexcept {
    void* runtime_pointer = nullptr;
    static_assert(sizeof(runtime_pointer) == sizeof(device_pointer));
    std::memcpy(reinterpret_cast<void*>(&runtime_pointer),
                reinterpret_cast<const void*>(&device_pointer), sizeof(runtime_pointer));
    return runtime_pointer;
}

CUdeviceptr to_device_pointer(void* runtime_pointer) noexcept {
    CUdeviceptr device_pointer = 0;
    static_assert(sizeof(runtime_pointer) == sizeof(device_pointer));
    std::memcpy(reinterpret_cast<void*>(&device_pointer),
                reinterpret_cast<const void*>(&runtime_pointer), sizeof(device_pointer));
    return device_pointer;
}

}  // namespace

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverAllocFunction allocate =
        resolve_driver_function<DriverAllocFunction>("cuMemAlloc_v2");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    CUdeviceptr driver_pointer = 0;
    const CUresult result = allocate(&driver_pointer, memory_bytes);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer) {
    const DriverFreeFunction release = resolve_driver_function<DriverFreeFunction>("cuMemFree_v2");
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const CUresult result = release(to_device_pointer(device_pointer));
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    const DriverMemGetInfoFunction query =
        resolve_driver_function<DriverMemGetInfoFunction>("cuMemGetInfo_v2");
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    const CUresult result = query(free_bytes, total_bytes);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}
