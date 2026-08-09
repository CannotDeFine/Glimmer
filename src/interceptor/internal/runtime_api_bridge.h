#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace glimmer::interceptor {

using RuntimeMallocFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes);
using RuntimeGetDeviceFunction = cudaError_t (*)(int* device);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
using RuntimeMemGetInfoFunction = cudaError_t (*)(std::size_t* free_bytes,
                                                  std::size_t* total_bytes);
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();
using RuntimeStreamSynchronizeFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamQueryFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamDestroyFunction = cudaError_t (*)(cudaStream_t stream);

[[nodiscard]] bool is_inside_runtime_call() noexcept;

[[nodiscard]] cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                                   RuntimeMallocFunction allocate,
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
