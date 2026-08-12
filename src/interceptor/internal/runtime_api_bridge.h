#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace glimmer::interceptor {

using RuntimeMallocFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes);
using RuntimeMallocManagedFunction = cudaError_t (*)(void** device_pointer,
                                                     std::size_t memory_bytes, unsigned int flags);
using RuntimeMallocPitchFunction = cudaError_t (*)(void** device_pointer, std::size_t* pitch,
                                                   std::size_t width_bytes, std::size_t height);
using RuntimeGetDeviceFunction = cudaError_t (*)(int* device);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
using RuntimeMemGetInfoFunction = cudaError_t (*)(std::size_t* free_bytes,
                                                  std::size_t* total_bytes);
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeMallocFromPoolAsyncFunction = cudaError_t (*)(void** device_pointer,
                                                           std::size_t memory_bytes,
                                                           cudaMemPool_t pool, cudaStream_t stream);
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();
using RuntimeStreamSynchronizeFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamQueryFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamDestroyFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeDeviceGetDefaultMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, int device);
using RuntimeDeviceSetMemPoolFunction = cudaError_t (*)(int device, cudaMemPool_t pool);
using RuntimeDeviceGetMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, int device);
using RuntimeMemPoolTrimToFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                     std::size_t min_bytes_to_keep);
using RuntimeMemPoolSetAttributeFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                           cudaMemPoolAttr attribute, void* value);
using RuntimeMemPoolGetAttributeFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                           cudaMemPoolAttr attribute, void* value);
using RuntimeMemPoolSetAccessFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                        const cudaMemAccessDesc* descriptors,
                                                        std::size_t descriptor_count);
using RuntimeMemPoolGetAccessFunction = cudaError_t (*)(cudaMemAccessFlags* flags,
                                                        cudaMemPool_t pool,
                                                        cudaMemLocation* location);
using RuntimeMemPoolCreateFunction = cudaError_t (*)(cudaMemPool_t* pool,
                                                     const cudaMemPoolProps* properties);
using RuntimeMemPoolDestroyFunction = cudaError_t (*)(cudaMemPool_t pool);
using RuntimeMemGetDefaultMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool,
                                                            cudaMemLocation* location,
                                                            cudaMemAllocationType allocation_type);
using RuntimeMemGetMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, cudaMemLocation* location,
                                                     cudaMemAllocationType allocation_type);
using RuntimeMemSetMemPoolFunction = cudaError_t (*)(cudaMemLocation* location,
                                                     cudaMemAllocationType allocation_type,
                                                     cudaMemPool_t pool);
using RuntimeMemPoolExportToShareableHandleFunction =
    cudaError_t (*)(void* handle_out, cudaMemPool_t pool,
                    enum cudaMemAllocationHandleType handle_type, unsigned int flags);
using RuntimeMemPoolImportFromShareableHandleFunction =
    cudaError_t (*)(cudaMemPool_t* pool_out, void* handle,
                    enum cudaMemAllocationHandleType handle_type, unsigned int flags);
using RuntimeMemPoolExportPointerFunction =
    cudaError_t (*)(cudaMemPoolPtrExportData* share_data_out, void* device_pointer);
using RuntimeMemPoolImportPointerFunction = cudaError_t (*)(void** pointer_out, cudaMemPool_t pool,
                                                            cudaMemPoolPtrExportData* share_data);

[[nodiscard]] bool is_inside_runtime_call() noexcept;

[[nodiscard]] cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                                   RuntimeMallocFunction allocate,
                                                   RuntimeFreeFunction release,
                                                   RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_managed(void** device_pointer,
                                                           std::size_t memory_bytes,
                                                           unsigned int flags,
                                                           RuntimeMallocManagedFunction allocate,
                                                           RuntimeFreeFunction release,
                                                           RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_pitch(void** device_pointer, std::size_t* pitch,
                                                         std::size_t width_bytes,
                                                         std::size_t height,
                                                         RuntimeMallocPitchFunction allocate,
                                                         RuntimeFreeFunction release,
                                                         RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_free(void* device_pointer, RuntimeFreeFunction release);
[[nodiscard]] cudaError_t intercept_runtime_mem_get_info(std::size_t* free_bytes,
                                                         std::size_t* total_bytes,
                                                         RuntimeMemGetInfoFunction query);
[[nodiscard]] cudaError_t intercept_runtime_malloc_async(
    void** device_pointer, std::size_t memory_bytes, cudaStream_t stream,
    RuntimeMallocAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_from_pool_async(
    void** device_pointer, std::size_t memory_bytes, cudaMemPool_t pool, cudaStream_t stream,
    RuntimeMallocFromPoolAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_mem_pool_import_from_shareable_handle(
    cudaMemPool_t* pool_out, void* handle, cudaMemAllocationHandleType handle_type,
    unsigned long long flags, RuntimeMemPoolImportFromShareableHandleFunction import_pool);
[[nodiscard]] cudaError_t intercept_runtime_mem_pool_import_pointer(
    void** pointer_out, cudaMemPool_t pool, cudaMemPoolPtrExportData* share_data,
    RuntimeMemPoolImportPointerFunction import_pointer);
[[nodiscard]] cudaError_t intercept_runtime_free_async(void* device_pointer, cudaStream_t stream,
                                                       RuntimeFreeAsyncFunction release);
[[nodiscard]] cudaError_t intercept_runtime_device_synchronize(
    RuntimeDeviceSynchronizeFunction synchronize);
[[nodiscard]] cudaError_t intercept_runtime_stream_synchronize(
    cudaStream_t stream, RuntimeStreamSynchronizeFunction synchronize);
[[nodiscard]] cudaError_t intercept_runtime_stream_query(cudaStream_t stream,
                                                         RuntimeStreamQueryFunction query);
[[nodiscard]] cudaError_t intercept_runtime_stream_destroy(cudaStream_t stream,
                                                           RuntimeStreamDestroyFunction destroy);

}  // namespace glimmer::interceptor
