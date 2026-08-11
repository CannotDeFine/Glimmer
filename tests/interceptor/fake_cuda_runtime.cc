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
using DriverAllocAsyncFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              CUstream stream);
using DriverPoolAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                             CUmemoryPool pool, CUstream stream);
using DriverFreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using DriverFreeAsyncFunction = CUresult (*)(CUdeviceptr device_pointer, CUstream stream);
using DriverContextSynchronizeFunction = CUresult (*)();
using DriverStreamSynchronizeFunction = CUresult (*)(CUstream stream);
using DriverStreamQueryFunction = CUresult (*)(CUstream stream);
using DriverStreamDestroyFunction = CUresult (*)(CUstream stream);
using DriverMemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);

std::uint8_t g_pool_token = 0;

cudaMemPool_t fake_pool() noexcept {
    return reinterpret_cast<cudaMemPool_t>(&g_pool_token);
}

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

extern "C" cudaError_t CUDARTAPI cudaGetDevice(int* device) {
    if (device == nullptr) {
        return cudaErrorInvalidValue;
    }
    *device = 0;
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

extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverAllocAsyncFunction allocate =
        resolve_driver_function<DriverAllocAsyncFunction>("cuMemAllocAsync");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    CUdeviceptr driver_pointer = 0;
    const CUresult result = allocate(&driver_pointer, memory_bytes, stream);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** device_pointer,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream) {
    return cudaMallocAsync(device_pointer, memory_bytes, stream);
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** device_pointer,
                                                         std::size_t memory_bytes,
                                                         cudaMemPool_t pool, cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DriverPoolAllocFunction allocate =
        resolve_driver_function<DriverPoolAllocFunction>("cuMemAllocFromPoolAsync");
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    CUdeviceptr driver_pointer = 0;
    const CUresult result =
        allocate(&driver_pointer, memory_bytes, reinterpret_cast<CUmemoryPool>(pool), stream);
    if (result != CUDA_SUCCESS) {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = to_runtime_pointer(driver_pointer);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync_ptsz(void** device_pointer,
                                                              std::size_t memory_bytes,
                                                              cudaMemPool_t pool,
                                                              cudaStream_t stream) {
    return cudaMallocFromPoolAsync(device_pointer, memory_bytes, pool, stream);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr || device < 0) {
        return cudaErrorInvalidValue;
    }
    *pool = fake_pool();
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
    return device < 0 || pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t* pool, int device) {
    return cudaDeviceGetDefaultMemPool(pool, device);
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool, std::size_t) {
    return pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool, cudaMemPoolAttr,
                                                         void* value) {
    return pool == nullptr || value == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool, cudaMemPoolAttr,
                                                         void* value) {
    return pool == nullptr || value == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t pool,
                                                      const cudaMemAccessDesc* descriptors,
                                                      std::size_t descriptor_count) {
    return pool == nullptr || (descriptor_count != 0 && descriptors == nullptr)
               ? cudaErrorInvalidValue
               : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t pool,
                                                      cudaMemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    *flags = cudaMemAccessFlagsProtReadWrite;
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                                   const cudaMemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return cudaErrorInvalidValue;
    }
    *pool = fake_pool();
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
    return pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemGetDefaultMemPool(cudaMemPool_t* pool,
                                                          cudaMemLocation* location,
                                                          cudaMemAllocationType) {
    return pool == nullptr || location == nullptr ? cudaErrorInvalidValue
                                                  : (*pool = fake_pool(), cudaSuccess);
}

extern "C" cudaError_t CUDARTAPI cudaMemGetMemPool(cudaMemPool_t* pool, cudaMemLocation* location,
                                                   cudaMemAllocationType type) {
    return cudaMemGetDefaultMemPool(pool, location, type);
}

extern "C" cudaError_t CUDARTAPI cudaMemSetMemPool(cudaMemLocation* location, cudaMemAllocationType,
                                                   cudaMemPool_t pool) {
    return location == nullptr || pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void* handle_out, cudaMemPool_t pool, enum cudaMemAllocationHandleType, unsigned int) {
    return handle_out == nullptr || pool == nullptr ? cudaErrorInvalidValue : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t* pool_out, void*, enum cudaMemAllocationHandleType, unsigned int) {
    return pool_out == nullptr ? cudaErrorInvalidValue : (*pool_out = fake_pool(), cudaSuccess);
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportPointer(cudaMemPoolPtrExportData* share_data_out,
                                                          void* device_pointer) {
    return share_data_out == nullptr || device_pointer == nullptr ? cudaErrorInvalidValue
                                                                  : cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportPointer(void** pointer_out, cudaMemPool_t pool,
                                                          cudaMemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return cudaErrorInvalidValue;
    }
    *pointer_out = reinterpret_cast<void*>(0x70000000U);
    return cudaSuccess;
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream) {
    const DriverFreeAsyncFunction release =
        resolve_driver_function<DriverFreeAsyncFunction>("cuMemFreeAsync");
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const CUresult result = release(to_device_pointer(device_pointer), stream);
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* device_pointer, cudaStream_t stream) {
    return cudaFreeAsync(device_pointer, stream);
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    const DriverContextSynchronizeFunction synchronize =
        resolve_driver_function<DriverContextSynchronizeFunction>("cuCtxSynchronize");
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    return synchronize() == CUDA_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    const DriverStreamSynchronizeFunction synchronize =
        resolve_driver_function<DriverStreamSynchronizeFunction>("cuStreamSynchronize");
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    return synchronize(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? cudaSuccess
                                                                           : cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    const DriverStreamQueryFunction query =
        resolve_driver_function<DriverStreamQueryFunction>("cuStreamQuery");
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    return query(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? cudaSuccess
                                                                     : cudaErrorUnknown;
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream) {
    return cudaStreamQuery(stream);
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    const DriverStreamDestroyFunction destroy =
        resolve_driver_function<DriverStreamDestroyFunction>("cuStreamDestroy");
    if (destroy == nullptr) {
        return cudaErrorNotSupported;
    }
    return destroy(reinterpret_cast<CUstream>(stream)) == CUDA_SUCCESS ? cudaSuccess
                                                                       : cudaErrorUnknown;
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
