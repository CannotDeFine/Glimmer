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
using MemCreateFunction = CUresult (*)(CUmemGenericAllocationHandle* handle,
                                       std::size_t memory_bytes, const CUmemAllocationProp* prop,
                                       unsigned long long flags);
using MemReleaseFunction = CUresult (*)(CUmemGenericAllocationHandle handle);
using MemAddressReserveFunction = CUresult (*)(CUdeviceptr* device_pointer,
                                               std::size_t memory_bytes, std::size_t alignment,
                                               CUdeviceptr requested_address,
                                               unsigned long long flags);
using MemAddressFreeFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes);
using MemMapFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                    std::size_t offset, CUmemGenericAllocationHandle handle,
                                    unsigned long long flags);
using MemUnmapFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes);
using MemSetAccessFunction = CUresult (*)(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                          const CUmemAccessDesc* access_descriptors,
                                          std::size_t descriptor_count);
using MemGetAddressRangeFunction = CUresult (*)(CUdeviceptr* base_pointer,
                                                std::size_t* memory_bytes,
                                                CUdeviceptr device_pointer);
using MemGetAccessFunction = CUresult (*)(unsigned long long* flags, const CUmemLocation* location,
                                          CUdeviceptr device_pointer);
using MemExportToShareableHandleFunction = CUresult (*)(void* shareable_handle,
                                                        CUmemGenericAllocationHandle handle,
                                                        CUmemAllocationHandleType handle_type,
                                                        unsigned long long flags);
using MemImportFromShareableHandleFunction = CUresult (*)(CUmemGenericAllocationHandle* handle,
                                                          void* os_handle,
                                                          CUmemAllocationHandleType handle_type);
using MemGetAllocationGranularityFunction = CUresult (*)(std::size_t* granularity,
                                                         const CUmemAllocationProp* prop,
                                                         CUmemAllocationGranularity_flags option);
using MemGetAllocationPropertiesFunction = CUresult (*)(CUmemAllocationProp* prop,
                                                        CUmemGenericAllocationHandle handle);
using MemRetainAllocationHandleFunction = CUresult (*)(CUmemGenericAllocationHandle* handle,
                                                       void* device_pointer);
using MemPoolTrimToFunction = CUresult (*)(CUmemoryPool pool, std::size_t min_bytes_to_keep);
using MemPoolSetAttributeFunction = CUresult (*)(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                 void* value);
using MemPoolGetAttributeFunction = CUresult (*)(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                 void* value);
using MemPoolSetAccessFunction = CUresult (*)(CUmemoryPool pool,
                                              const CUmemAccessDesc* access_descriptors,
                                              std::size_t descriptor_count);
using MemPoolGetAccessFunction = CUresult (*)(CUmemAccess_flags* flags, CUmemoryPool pool,
                                              CUmemLocation* location);
using MemPoolCreateFunction = CUresult (*)(CUmemoryPool* pool, const CUmemPoolProps* properties);
using MemPoolDestroyFunction = CUresult (*)(CUmemoryPool pool);
using DeviceGetMemPoolFunction = CUresult (*)(CUmemoryPool* pool, CUdevice device);
using DeviceSetMemPoolFunction = CUresult (*)(CUdevice device, CUmemoryPool pool);
using DeviceGetDefaultMemPoolFunction = CUresult (*)(CUmemoryPool* pool, CUdevice device);
using MemGetDefaultMemPoolFunction = CUresult (*)(CUmemoryPool* pool, CUmemLocation* location,
                                                  CUmemAllocationType allocation_type);
using MemGetMemPoolFunction = CUresult (*)(CUmemoryPool* pool, CUmemLocation* location,
                                           CUmemAllocationType allocation_type);
using MemSetMemPoolFunction = CUresult (*)(CUmemLocation* location,
                                           CUmemAllocationType allocation_type, CUmemoryPool pool);
using MemPoolExportToShareableHandleFunction = CUresult (*)(void* handle_out, CUmemoryPool pool,
                                                            CUmemAllocationHandleType handle_type,
                                                            unsigned long long flags);
using MemPoolImportFromShareableHandleFunction = CUresult (*)(CUmemoryPool* pool_out, void* handle,
                                                              CUmemAllocationHandleType handle_type,
                                                              unsigned long long flags);
using MemPoolExportPointerFunction = CUresult (*)(CUmemPoolPtrExportData* share_data_out,
                                                  CUdeviceptr device_pointer);
using MemPoolImportPointerFunction = CUresult (*)(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                                  CUmemPoolPtrExportData* share_data);
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
using StreamDestroyFunction = CUresult (*)(CUstream stream);
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
    MemCreateFunction mem_create = nullptr;
    MemReleaseFunction mem_release = nullptr;
    MemAddressReserveFunction mem_address_reserve = nullptr;
    MemAddressFreeFunction mem_address_free = nullptr;
    MemMapFunction mem_map = nullptr;
    MemUnmapFunction mem_unmap = nullptr;
    MemSetAccessFunction mem_set_access = nullptr;
    MemGetAddressRangeFunction mem_get_address_range = nullptr;
    MemGetAccessFunction mem_get_access = nullptr;
    MemExportToShareableHandleFunction mem_export_to_shareable_handle = nullptr;
    MemImportFromShareableHandleFunction mem_import_from_shareable_handle = nullptr;
    MemGetAllocationGranularityFunction mem_get_allocation_granularity = nullptr;
    MemGetAllocationPropertiesFunction mem_get_allocation_properties = nullptr;
    MemRetainAllocationHandleFunction mem_retain_allocation_handle = nullptr;
    MemPoolTrimToFunction mem_pool_trim_to = nullptr;
    MemPoolSetAttributeFunction mem_pool_set_attribute = nullptr;
    MemPoolGetAttributeFunction mem_pool_get_attribute = nullptr;
    MemPoolSetAccessFunction mem_pool_set_access = nullptr;
    MemPoolGetAccessFunction mem_pool_get_access = nullptr;
    MemPoolCreateFunction mem_pool_create = nullptr;
    MemPoolDestroyFunction mem_pool_destroy = nullptr;
    DeviceGetMemPoolFunction device_get_mem_pool = nullptr;
    DeviceSetMemPoolFunction device_set_mem_pool = nullptr;
    DeviceGetDefaultMemPoolFunction device_get_default_mem_pool = nullptr;
    MemGetDefaultMemPoolFunction mem_get_default_mem_pool = nullptr;
    MemGetMemPoolFunction mem_get_mem_pool = nullptr;
    MemSetMemPoolFunction mem_set_mem_pool = nullptr;
    MemPoolExportToShareableHandleFunction mem_pool_export_to_shareable_handle = nullptr;
    MemPoolImportFromShareableHandleFunction mem_pool_import_from_shareable_handle = nullptr;
    MemPoolExportPointerFunction mem_pool_export_pointer = nullptr;
    MemPoolImportPointerFunction mem_pool_import_pointer = nullptr;
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
    StreamDestroyFunction stream_destroy = nullptr;
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
    [[nodiscard]] CUresult mem_create(CUmemGenericAllocationHandle* handle,
                                      std::size_t memory_bytes, const CUmemAllocationProp* prop,
                                      unsigned long long flags) const;
    [[nodiscard]] CUresult mem_release(CUmemGenericAllocationHandle handle) const;
    [[nodiscard]] CUresult mem_address_reserve(CUdeviceptr* device_pointer,
                                               std::size_t memory_bytes, std::size_t alignment,
                                               CUdeviceptr requested_address,
                                               unsigned long long flags) const;
    [[nodiscard]] CUresult mem_address_free(CUdeviceptr device_pointer,
                                            std::size_t memory_bytes) const;
    [[nodiscard]] CUresult mem_map(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                   std::size_t offset, CUmemGenericAllocationHandle handle,
                                   unsigned long long flags) const;
    [[nodiscard]] CUresult mem_unmap(CUdeviceptr device_pointer, std::size_t memory_bytes) const;
    [[nodiscard]] CUresult mem_set_access(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                          const CUmemAccessDesc* access_descriptors,
                                          std::size_t descriptor_count) const;
    [[nodiscard]] CUresult mem_get_address_range(CUdeviceptr* base_pointer,
                                                 std::size_t* memory_bytes,
                                                 CUdeviceptr device_pointer) const;
    [[nodiscard]] CUresult mem_get_access(unsigned long long* flags, const CUmemLocation* location,
                                          CUdeviceptr device_pointer) const;
    [[nodiscard]] CUresult mem_export_to_shareable_handle(void* shareable_handle,
                                                          CUmemGenericAllocationHandle handle,
                                                          CUmemAllocationHandleType handle_type,
                                                          unsigned long long flags) const;
    [[nodiscard]] CUresult mem_import_from_shareable_handle(
        CUmemGenericAllocationHandle* handle, void* os_handle,
        CUmemAllocationHandleType handle_type) const;
    [[nodiscard]] CUresult mem_get_allocation_granularity(
        std::size_t* granularity, const CUmemAllocationProp* prop,
        CUmemAllocationGranularity_flags option) const;
    [[nodiscard]] CUresult mem_get_allocation_properties(CUmemAllocationProp* prop,
                                                         CUmemGenericAllocationHandle handle) const;
    [[nodiscard]] CUresult mem_retain_allocation_handle(CUmemGenericAllocationHandle* handle,
                                                        void* device_pointer) const;
    [[nodiscard]] CUresult mem_pool_trim_to(CUmemoryPool pool, std::size_t min_bytes_to_keep) const;
    [[nodiscard]] CUresult mem_pool_set_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) const;
    [[nodiscard]] CUresult mem_pool_get_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) const;
    [[nodiscard]] CUresult mem_pool_set_access(CUmemoryPool pool,
                                               const CUmemAccessDesc* access_descriptors,
                                               std::size_t descriptor_count) const;
    [[nodiscard]] CUresult mem_pool_get_access(CUmemAccess_flags* flags, CUmemoryPool pool,
                                               CUmemLocation* location) const;
    [[nodiscard]] CUresult mem_pool_create(CUmemoryPool* pool,
                                           const CUmemPoolProps* properties) const;
    [[nodiscard]] CUresult mem_pool_destroy(CUmemoryPool pool) const;
    [[nodiscard]] CUresult device_get_mem_pool(CUmemoryPool* pool, CUdevice device) const;
    [[nodiscard]] CUresult device_set_mem_pool(CUdevice device, CUmemoryPool pool) const;
    [[nodiscard]] CUresult device_get_default_mem_pool(CUmemoryPool* pool, CUdevice device) const;
    [[nodiscard]] CUresult mem_get_default_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                                    CUmemAllocationType allocation_type) const;
    [[nodiscard]] CUresult mem_get_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType allocation_type) const;
    [[nodiscard]] CUresult mem_set_mem_pool(CUmemLocation* location,
                                            CUmemAllocationType allocation_type,
                                            CUmemoryPool pool) const;
    [[nodiscard]] CUresult mem_pool_export_to_shareable_handle(
        void* handle_out, CUmemoryPool pool, CUmemAllocationHandleType handle_type,
        unsigned long long flags) const;
    [[nodiscard]] CUresult mem_pool_import_from_shareable_handle(
        CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handle_type,
        unsigned long long flags) const;
    [[nodiscard]] CUresult mem_pool_export_pointer(CUmemPoolPtrExportData* share_data_out,
                                                   CUdeviceptr device_pointer) const;
    [[nodiscard]] CUresult mem_pool_import_pointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                                   CUmemPoolPtrExportData* share_data) const;
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
    [[nodiscard]] CUresult stream_destroy(CUstream stream) const;

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
    [[nodiscard]] bool has_mem_create() const;
    [[nodiscard]] bool has_mem_release() const;
    [[nodiscard]] bool has_mem_address_reserve() const;
    [[nodiscard]] bool has_mem_address_free() const;
    [[nodiscard]] bool has_mem_map() const;
    [[nodiscard]] bool has_mem_unmap() const;
    [[nodiscard]] bool has_mem_set_access() const;
    [[nodiscard]] bool has_mem_get_address_range() const;
    [[nodiscard]] bool has_mem_get_access() const;
    [[nodiscard]] bool has_mem_export_to_shareable_handle() const;
    [[nodiscard]] bool has_mem_import_from_shareable_handle() const;
    [[nodiscard]] bool has_mem_get_allocation_granularity() const;
    [[nodiscard]] bool has_mem_get_allocation_properties() const;
    [[nodiscard]] bool has_mem_retain_allocation_handle() const;
    [[nodiscard]] bool has_mem_pool_trim_to() const;
    [[nodiscard]] bool has_mem_pool_set_attribute() const;
    [[nodiscard]] bool has_mem_pool_get_attribute() const;
    [[nodiscard]] bool has_mem_pool_set_access() const;
    [[nodiscard]] bool has_mem_pool_get_access() const;
    [[nodiscard]] bool has_mem_pool_create() const;
    [[nodiscard]] bool has_mem_pool_destroy() const;
    [[nodiscard]] bool has_device_get_mem_pool() const;
    [[nodiscard]] bool has_device_set_mem_pool() const;
    [[nodiscard]] bool has_device_get_default_mem_pool() const;
    [[nodiscard]] bool has_mem_get_default_mem_pool() const;
    [[nodiscard]] bool has_mem_get_mem_pool() const;
    [[nodiscard]] bool has_mem_set_mem_pool() const;
    [[nodiscard]] bool has_mem_pool_export_to_shareable_handle() const;
    [[nodiscard]] bool has_mem_pool_import_from_shareable_handle() const;
    [[nodiscard]] bool has_mem_pool_export_pointer() const;
    [[nodiscard]] bool has_mem_pool_import_pointer() const;
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
    [[nodiscard]] bool has_stream_destroy() const;

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
    MemCreateFunction mem_create_ = nullptr;
    MemReleaseFunction mem_release_ = nullptr;
    MemAddressReserveFunction mem_address_reserve_ = nullptr;
    MemAddressFreeFunction mem_address_free_ = nullptr;
    MemMapFunction mem_map_ = nullptr;
    MemUnmapFunction mem_unmap_ = nullptr;
    MemSetAccessFunction mem_set_access_ = nullptr;
    MemGetAddressRangeFunction mem_get_address_range_ = nullptr;
    MemGetAccessFunction mem_get_access_ = nullptr;
    MemExportToShareableHandleFunction mem_export_to_shareable_handle_ = nullptr;
    MemImportFromShareableHandleFunction mem_import_from_shareable_handle_ = nullptr;
    MemGetAllocationGranularityFunction mem_get_allocation_granularity_ = nullptr;
    MemGetAllocationPropertiesFunction mem_get_allocation_properties_ = nullptr;
    MemRetainAllocationHandleFunction mem_retain_allocation_handle_ = nullptr;
    MemPoolTrimToFunction mem_pool_trim_to_ = nullptr;
    MemPoolSetAttributeFunction mem_pool_set_attribute_ = nullptr;
    MemPoolGetAttributeFunction mem_pool_get_attribute_ = nullptr;
    MemPoolSetAccessFunction mem_pool_set_access_ = nullptr;
    MemPoolGetAccessFunction mem_pool_get_access_ = nullptr;
    MemPoolCreateFunction mem_pool_create_ = nullptr;
    MemPoolDestroyFunction mem_pool_destroy_ = nullptr;
    DeviceGetMemPoolFunction device_get_mem_pool_ = nullptr;
    DeviceSetMemPoolFunction device_set_mem_pool_ = nullptr;
    DeviceGetDefaultMemPoolFunction device_get_default_mem_pool_ = nullptr;
    MemGetDefaultMemPoolFunction mem_get_default_mem_pool_ = nullptr;
    MemGetMemPoolFunction mem_get_mem_pool_ = nullptr;
    MemSetMemPoolFunction mem_set_mem_pool_ = nullptr;
    MemPoolExportToShareableHandleFunction mem_pool_export_to_shareable_handle_ = nullptr;
    MemPoolImportFromShareableHandleFunction mem_pool_import_from_shareable_handle_ = nullptr;
    MemPoolExportPointerFunction mem_pool_export_pointer_ = nullptr;
    MemPoolImportPointerFunction mem_pool_import_pointer_ = nullptr;
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
    StreamDestroyFunction stream_destroy_ = nullptr;
    GetProcAddressFunction get_proc_address_ = nullptr;
    GetProcAddressV2Function get_proc_address_v2_ = nullptr;
};

}  // namespace glimmer::interceptor
