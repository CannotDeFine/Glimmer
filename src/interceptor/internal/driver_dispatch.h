#pragma once

#include <cuda.h>

#include <cstddef>

namespace glimmer::interceptor {

using MemAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes);
using InitFunction = CUresult (*)(unsigned int flags);
using MemAllocManagedFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                             unsigned int flags);
using MemAllocPitchFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t* pitch,
                                           std::size_t width_bytes, std::size_t height,
                                           unsigned int element_size_bytes);
using MemAllocAsyncFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                           CUstream stream);
using MemAllocFromPoolAsyncFunction = CUresult (*)(CUdeviceptr* device_pointer,
                                                   std::size_t memory_bytes, CUmemoryPool pool,
                                                   CUstream stream);
using MemFreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using MemFreeAsyncFunction = CUresult (*)(CUdeviceptr device_pointer, CUstream stream);
using MemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);
using DeviceTotalMemFunction = CUresult (*)(std::size_t* total_bytes, CUdevice device);
using ContextSynchronizeFunction = CUresult (*)();
using ContextGetCurrentFunction = CUresult (*)(CUcontext* context);
using ContextGetDeviceFunction = CUresult (*)(CUdevice* device);
using ContextDestroyFunction = CUresult (*)(CUcontext context);
using StreamGetDeviceFunction = CUresult (*)(CUstream stream, CUdevice* device);
using StreamGetContextFunction = CUresult (*)(CUstream stream, CUcontext* context);
using StreamQueryFunction = CUresult (*)(CUstream stream);
using StreamSynchronizeFunction = CUresult (*)(CUstream stream);
using DlsymFunction = void* (*)(void* handle, const char* name);
using GetProcAddressFunction = CUresult (*)(const char* symbol, void** function_pointer,
                                            int cuda_version, cuuint64_t flags);
using GetProcAddressV2Function = CUresult (*)(const char* symbol, void** function_pointer,
                                              int cuda_version, cuuint64_t flags,
                                              CUdriverProcAddressQueryResult* symbol_status);

struct DriverFunctionTable {
    InitFunction init = nullptr;
    MemAllocFunction mem_alloc = nullptr;
    MemAllocManagedFunction mem_alloc_managed = nullptr;
    MemAllocPitchFunction mem_alloc_pitch = nullptr;
    MemAllocAsyncFunction mem_alloc_async = nullptr;
    MemAllocAsyncFunction mem_alloc_async_ptsz = nullptr;
    MemAllocFromPoolAsyncFunction mem_alloc_from_pool_async = nullptr;
    MemAllocFromPoolAsyncFunction mem_alloc_from_pool_async_ptsz = nullptr;
    MemFreeFunction mem_free = nullptr;
    MemFreeAsyncFunction mem_free_async = nullptr;
    MemFreeAsyncFunction mem_free_async_ptsz = nullptr;
    MemGetInfoFunction mem_get_info = nullptr;
    DeviceTotalMemFunction device_total_mem = nullptr;
    ContextSynchronizeFunction context_synchronize = nullptr;
    ContextGetCurrentFunction context_get_current = nullptr;
    ContextGetDeviceFunction context_get_device = nullptr;
    ContextDestroyFunction context_destroy = nullptr;
    StreamGetDeviceFunction stream_get_device = nullptr;
    StreamGetDeviceFunction stream_get_device_ptsz = nullptr;
    StreamGetContextFunction stream_get_context = nullptr;
    StreamGetContextFunction stream_get_context_ptsz = nullptr;
    StreamQueryFunction stream_query = nullptr;
    StreamQueryFunction stream_query_ptsz = nullptr;
    StreamSynchronizeFunction stream_synchronize = nullptr;
    StreamSynchronizeFunction stream_synchronize_ptsz = nullptr;
    GetProcAddressFunction get_proc_address = nullptr;
    GetProcAddressV2Function get_proc_address_v2 = nullptr;
};

[[nodiscard]] DlsymFunction resolve_real_dlsym() noexcept;
[[nodiscard]] bool is_inside_driver_call() noexcept;

class DriverDispatch {
   public:
    DriverDispatch() = default;
    explicit DriverDispatch(DriverFunctionTable functions) noexcept;

    DriverDispatch(const DriverDispatch&) = delete;
    DriverDispatch& operator=(const DriverDispatch&) = delete;

    ~DriverDispatch();

    [[nodiscard]] bool initialize();

    [[nodiscard]] CUresult mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) const;
    [[nodiscard]] CUresult init(unsigned int flags) const;
    [[nodiscard]] CUresult mem_alloc_managed(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                             unsigned int flags) const;
    [[nodiscard]] CUresult mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                           std::size_t width_bytes, std::size_t height,
                                           unsigned int element_size_bytes) const;
    [[nodiscard]] CUresult mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                           CUstream stream) const;
    [[nodiscard]] CUresult mem_alloc_async_ptsz(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, CUstream stream) const;
    [[nodiscard]] CUresult mem_alloc_from_pool_async(CUdeviceptr* device_pointer,
                                                     std::size_t memory_bytes, CUmemoryPool pool,
                                                     CUstream stream) const;
    [[nodiscard]] CUresult mem_alloc_from_pool_async_ptsz(CUdeviceptr* device_pointer,
                                                          std::size_t memory_bytes,
                                                          CUmemoryPool pool, CUstream stream) const;
    [[nodiscard]] CUresult mem_free(CUdeviceptr device_pointer) const;
    [[nodiscard]] CUresult mem_free_async(CUdeviceptr device_pointer, CUstream stream) const;
    [[nodiscard]] CUresult mem_free_async_ptsz(CUdeviceptr device_pointer, CUstream stream) const;
    [[nodiscard]] CUresult mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) const;
    [[nodiscard]] CUresult device_total_mem(std::size_t* total_bytes, CUdevice device) const;
    [[nodiscard]] CUresult context_synchronize() const;
    [[nodiscard]] CUresult context_get_current(CUcontext* context) const;
    [[nodiscard]] CUresult context_get_device(CUdevice* device) const;
    [[nodiscard]] CUresult context_destroy(CUcontext context) const;
    [[nodiscard]] CUresult stream_get_device(CUstream stream, CUdevice* device) const;
    [[nodiscard]] CUresult stream_get_device_ptsz(CUstream stream, CUdevice* device) const;
    [[nodiscard]] CUresult stream_get_context(CUstream stream, CUcontext* context) const;
    [[nodiscard]] CUresult stream_get_context_ptsz(CUstream stream, CUcontext* context) const;
    [[nodiscard]] CUresult stream_query(CUstream stream) const;
    [[nodiscard]] CUresult stream_query_ptsz(CUstream stream) const;
    [[nodiscard]] CUresult stream_synchronize(CUstream stream) const;
    [[nodiscard]] CUresult stream_synchronize_ptsz(CUstream stream) const;

    [[nodiscard]] CUresult get_proc_address(const char* symbol, void** function_pointer,
                                            int cuda_version, cuuint64_t flags) const;
    [[nodiscard]] CUresult get_proc_address_v2(const char* symbol, void** function_pointer,
                                               int cuda_version, cuuint64_t flags,
                                               CUdriverProcAddressQueryResult* symbol_status) const;

    [[nodiscard]] bool has_get_proc_address() const;
    [[nodiscard]] bool has_get_proc_address_v2() const;
    [[nodiscard]] bool has_mem_alloc_managed() const;
    [[nodiscard]] bool has_mem_alloc_pitch() const;
    [[nodiscard]] bool has_mem_alloc_async() const;
    [[nodiscard]] bool has_mem_alloc_async_ptsz() const;
    [[nodiscard]] bool has_mem_alloc_from_pool_async() const;
    [[nodiscard]] bool has_mem_alloc_from_pool_async_ptsz() const;
    [[nodiscard]] bool has_mem_free_async() const;
    [[nodiscard]] bool has_mem_free_async_ptsz() const;
    [[nodiscard]] bool has_device_total_mem() const;
    [[nodiscard]] bool has_context_synchronize() const;
    [[nodiscard]] bool has_context_queries() const;
    [[nodiscard]] bool has_context_destroy() const;
    [[nodiscard]] bool has_stream_identity() const;
    [[nodiscard]] bool has_stream_identity_ptsz() const;
    [[nodiscard]] bool has_stream_query() const;
    [[nodiscard]] bool has_stream_query_ptsz() const;
    [[nodiscard]] bool has_stream_synchronize() const;
    [[nodiscard]] bool has_stream_synchronize_ptsz() const;

   private:
    [[nodiscard]] void* load_symbol(const char* name) const;

    void* library_handle_ = nullptr;
    InitFunction init_ = nullptr;
    MemAllocFunction mem_alloc_ = nullptr;
    MemAllocManagedFunction mem_alloc_managed_ = nullptr;
    MemAllocPitchFunction mem_alloc_pitch_ = nullptr;
    MemAllocAsyncFunction mem_alloc_async_ = nullptr;
    MemAllocAsyncFunction mem_alloc_async_ptsz_ = nullptr;
    MemAllocFromPoolAsyncFunction mem_alloc_from_pool_async_ = nullptr;
    MemAllocFromPoolAsyncFunction mem_alloc_from_pool_async_ptsz_ = nullptr;
    MemFreeFunction mem_free_ = nullptr;
    MemFreeAsyncFunction mem_free_async_ = nullptr;
    MemFreeAsyncFunction mem_free_async_ptsz_ = nullptr;
    MemGetInfoFunction mem_get_info_ = nullptr;
    DeviceTotalMemFunction device_total_mem_ = nullptr;
    ContextSynchronizeFunction context_synchronize_ = nullptr;
    ContextGetCurrentFunction context_get_current_ = nullptr;
    ContextGetDeviceFunction context_get_device_ = nullptr;
    ContextDestroyFunction context_destroy_ = nullptr;
    StreamGetDeviceFunction stream_get_device_ = nullptr;
    StreamGetDeviceFunction stream_get_device_ptsz_ = nullptr;
    StreamGetContextFunction stream_get_context_ = nullptr;
    StreamGetContextFunction stream_get_context_ptsz_ = nullptr;
    StreamQueryFunction stream_query_ = nullptr;
    StreamQueryFunction stream_query_ptsz_ = nullptr;
    StreamSynchronizeFunction stream_synchronize_ = nullptr;
    StreamSynchronizeFunction stream_synchronize_ptsz_ = nullptr;
    GetProcAddressFunction get_proc_address_ = nullptr;
    GetProcAddressV2Function get_proc_address_v2_ = nullptr;
};

}  // namespace glimmer::interceptor
