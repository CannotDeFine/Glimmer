#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace glimmer::interceptor {

using RuntimeMallocFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
using RuntimeMemGetInfoFunction = cudaError_t (*)(std::size_t* free_bytes,
                                                  std::size_t* total_bytes);
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();

[[nodiscard]] bool is_inside_runtime_call() noexcept;

[[nodiscard]] cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                                   RuntimeMallocFunction allocate,
                                                   RuntimeFreeFunction release);
[[nodiscard]] cudaError_t intercept_runtime_free(void* device_pointer, RuntimeFreeFunction release);
[[nodiscard]] cudaError_t intercept_runtime_mem_get_info(std::size_t* free_bytes,
                                                         std::size_t* total_bytes,
                                                         RuntimeMemGetInfoFunction query);

}  // namespace glimmer::interceptor
