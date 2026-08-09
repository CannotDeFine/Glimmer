#pragma once

#include <cuda.h>

#include <cstddef>

namespace glimmer::interceptor {

[[nodiscard]] CUresult intercept_init(unsigned int flags);
[[nodiscard]] CUresult intercept_mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes);
[[nodiscard]] CUresult intercept_mem_alloc_managed(CUdeviceptr* device_pointer,
                                                   std::size_t memory_bytes, unsigned int flags);
[[nodiscard]] CUresult intercept_mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                                 std::size_t width_bytes, std::size_t height,
                                                 unsigned int element_size_bytes);
[[nodiscard]] CUresult intercept_mem_alloc_async(CUdeviceptr* device_pointer,
                                                 std::size_t memory_bytes, CUstream stream,
                                                 bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_mem_alloc_from_pool_async(CUdeviceptr* device_pointer,
                                                           std::size_t memory_bytes,
                                                           CUmemoryPool pool, CUstream stream,
                                                           bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_mem_free(CUdeviceptr device_pointer);
[[nodiscard]] CUresult intercept_mem_free_async(CUdeviceptr device_pointer, CUstream stream,
                                                bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes);
[[nodiscard]] CUresult intercept_device_total_mem(std::size_t* total_bytes, CUdevice device);
[[nodiscard]] CUresult intercept_context_get_current(CUcontext* context);
[[nodiscard]] CUresult intercept_context_get_device(CUdevice* device);
[[nodiscard]] CUresult intercept_context_destroy(CUcontext context);
[[nodiscard]] CUresult intercept_stream_get_device(CUstream stream, CUdevice* device,
                                                   bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_get_context(CUstream stream, CUcontext* context,
                                                    bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_query(CUstream stream, bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_synchronize(CUstream stream,
                                                    bool per_thread_default_stream);
[[nodiscard]] CUresult intercept_stream_destroy(CUstream stream);
[[nodiscard]] CUresult intercept_context_synchronize();
[[nodiscard]] CUresult intercept_get_proc_address(const char* symbol, void** function_pointer,
                                                  int cuda_version, cuuint64_t flags);
[[nodiscard]] CUresult intercept_get_proc_address_v2(const char* symbol, void** function_pointer,
                                                     int cuda_version, cuuint64_t flags,
                                                     CUdriverProcAddressQueryResult* symbol_status);

}  // namespace glimmer::interceptor
