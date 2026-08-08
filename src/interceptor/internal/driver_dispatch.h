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
using MemFreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using MemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);
using DeviceTotalMemFunction = CUresult (*)(std::size_t* total_bytes, CUdevice device);
using ContextGetCurrentFunction = CUresult (*)(CUcontext* context);
using ContextGetDeviceFunction = CUresult (*)(CUdevice* device);
using ContextDestroyFunction = CUresult (*)(CUcontext context);
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
    MemFreeFunction mem_free = nullptr;
    MemGetInfoFunction mem_get_info = nullptr;
    DeviceTotalMemFunction device_total_mem = nullptr;
    ContextGetCurrentFunction context_get_current = nullptr;
    ContextGetDeviceFunction context_get_device = nullptr;
    ContextDestroyFunction context_destroy = nullptr;
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
    [[nodiscard]] CUresult mem_free(CUdeviceptr device_pointer) const;
    [[nodiscard]] CUresult mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) const;
    [[nodiscard]] CUresult device_total_mem(std::size_t* total_bytes, CUdevice device) const;
    [[nodiscard]] CUresult context_get_current(CUcontext* context) const;
    [[nodiscard]] CUresult context_get_device(CUdevice* device) const;
    [[nodiscard]] CUresult context_destroy(CUcontext context) const;

    [[nodiscard]] CUresult get_proc_address(const char* symbol, void** function_pointer,
                                            int cuda_version, cuuint64_t flags) const;
    [[nodiscard]] CUresult get_proc_address_v2(const char* symbol, void** function_pointer,
                                               int cuda_version, cuuint64_t flags,
                                               CUdriverProcAddressQueryResult* symbol_status) const;

    [[nodiscard]] bool has_get_proc_address() const;
    [[nodiscard]] bool has_get_proc_address_v2() const;
    [[nodiscard]] bool has_mem_alloc_managed() const;
    [[nodiscard]] bool has_mem_alloc_pitch() const;
    [[nodiscard]] bool has_device_total_mem() const;
    [[nodiscard]] bool has_context_queries() const;
    [[nodiscard]] bool has_context_destroy() const;

   private:
    [[nodiscard]] void* load_symbol(const char* name) const;

    void* library_handle_ = nullptr;
    InitFunction init_ = nullptr;
    MemAllocFunction mem_alloc_ = nullptr;
    MemAllocManagedFunction mem_alloc_managed_ = nullptr;
    MemAllocPitchFunction mem_alloc_pitch_ = nullptr;
    MemFreeFunction mem_free_ = nullptr;
    MemGetInfoFunction mem_get_info_ = nullptr;
    DeviceTotalMemFunction device_total_mem_ = nullptr;
    ContextGetCurrentFunction context_get_current_ = nullptr;
    ContextGetDeviceFunction context_get_device_ = nullptr;
    ContextDestroyFunction context_destroy_ = nullptr;
    GetProcAddressFunction get_proc_address_ = nullptr;
    GetProcAddressV2Function get_proc_address_v2_ = nullptr;
};

}  // namespace glimmer::interceptor
