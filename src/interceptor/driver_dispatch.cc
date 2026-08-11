#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_dispatch.h"

#include <dlfcn.h>

namespace glimmer::interceptor {

namespace {

thread_local bool g_is_inside_driver_call = false;

class DriverCallScope {
   public:
    DriverCallScope() : previous_state_(g_is_inside_driver_call) {
        g_is_inside_driver_call = true;
    }

    ~DriverCallScope() {
        g_is_inside_driver_call = previous_state_;
    }

   private:
    bool previous_state_;
};

}  // namespace

bool is_inside_driver_call() noexcept {
    return g_is_inside_driver_call;
}

DlsymFunction resolve_real_dlsym() noexcept {
    static DlsymFunction real_dlsym = []() noexcept {
        void* symbol = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        return reinterpret_cast<DlsymFunction>(symbol);
    }();
    return real_dlsym;
}

DriverDispatch::~DriverDispatch() {
    if (library_handle_ != nullptr) {
        dlclose(library_handle_);
    }
}

DriverDispatch::DriverDispatch(DriverFunctionTable functions) noexcept
    : init_(functions.init),
      launch_kernel_(functions.launch_kernel),
      launch_kernel_ptsz_(functions.launch_kernel_ptsz),
      mem_alloc_(functions.mem_alloc),
      mem_alloc_managed_(functions.mem_alloc_managed),
      mem_alloc_pitch_(functions.mem_alloc_pitch),
      mem_alloc_async_(functions.mem_alloc_async),
      mem_alloc_async_ptsz_(functions.mem_alloc_async_ptsz),
      mem_alloc_from_pool_async_(functions.mem_alloc_from_pool_async),
      mem_alloc_from_pool_async_ptsz_(functions.mem_alloc_from_pool_async_ptsz),
      mem_create_(functions.mem_create),
      mem_release_(functions.mem_release),
      mem_address_reserve_(functions.mem_address_reserve),
      mem_address_free_(functions.mem_address_free),
      mem_map_(functions.mem_map),
      mem_unmap_(functions.mem_unmap),
      mem_set_access_(functions.mem_set_access),
      mem_get_address_range_(functions.mem_get_address_range),
      mem_get_access_(functions.mem_get_access),
      mem_export_to_shareable_handle_(functions.mem_export_to_shareable_handle),
      mem_import_from_shareable_handle_(functions.mem_import_from_shareable_handle),
      mem_get_allocation_granularity_(functions.mem_get_allocation_granularity),
      mem_get_allocation_properties_(functions.mem_get_allocation_properties),
      mem_retain_allocation_handle_(functions.mem_retain_allocation_handle),
      mem_pool_trim_to_(functions.mem_pool_trim_to),
      mem_pool_set_attribute_(functions.mem_pool_set_attribute),
      mem_pool_get_attribute_(functions.mem_pool_get_attribute),
      mem_pool_set_access_(functions.mem_pool_set_access),
      mem_pool_get_access_(functions.mem_pool_get_access),
      mem_pool_create_(functions.mem_pool_create),
      mem_pool_destroy_(functions.mem_pool_destroy),
      device_get_mem_pool_(functions.device_get_mem_pool),
      device_set_mem_pool_(functions.device_set_mem_pool),
      device_get_default_mem_pool_(functions.device_get_default_mem_pool),
      mem_get_default_mem_pool_(functions.mem_get_default_mem_pool),
      mem_get_mem_pool_(functions.mem_get_mem_pool),
      mem_set_mem_pool_(functions.mem_set_mem_pool),
      mem_pool_export_to_shareable_handle_(functions.mem_pool_export_to_shareable_handle),
      mem_pool_import_from_shareable_handle_(functions.mem_pool_import_from_shareable_handle),
      mem_pool_export_pointer_(functions.mem_pool_export_pointer),
      mem_pool_import_pointer_(functions.mem_pool_import_pointer),
      mem_free_(functions.mem_free),
      mem_free_async_(functions.mem_free_async),
      mem_free_async_ptsz_(functions.mem_free_async_ptsz),
      mem_get_info_(functions.mem_get_info),
      device_total_mem_(functions.device_total_mem),
      context_synchronize_(functions.context_synchronize),
      context_get_current_(functions.context_get_current),
      context_get_device_(functions.context_get_device),
      context_destroy_(functions.context_destroy),
      stream_get_device_(functions.stream_get_device),
      stream_get_device_ptsz_(functions.stream_get_device_ptsz),
      stream_get_context_(functions.stream_get_context),
      stream_get_context_ptsz_(functions.stream_get_context_ptsz),
      stream_query_(functions.stream_query),
      stream_query_ptsz_(functions.stream_query_ptsz),
      stream_synchronize_(functions.stream_synchronize),
      stream_synchronize_ptsz_(functions.stream_synchronize_ptsz),
      stream_destroy_(functions.stream_destroy),
      get_proc_address_(functions.get_proc_address),
      get_proc_address_v2_(functions.get_proc_address_v2) {}

bool DriverDispatch::initialize() {
    // The CUDA driver's loader invokes dlsym while initializing its own
    // shared object.  Keep those lookups on the real path; returning Glimmer
    // wrappers from inside the driver's constructor can corrupt its setup.
    DriverCallScope scope;
    library_handle_ = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (library_handle_ == nullptr) {
        return false;
    }

    init_ = reinterpret_cast<InitFunction>(load_symbol("cuInit"));
    launch_kernel_ = reinterpret_cast<LaunchKernelFunction>(load_symbol("cuLaunchKernel"));
    launch_kernel_ptsz_ =
        reinterpret_cast<LaunchKernelFunction>(load_symbol("cuLaunchKernel_ptsz"));
    mem_alloc_ = reinterpret_cast<MemAllocFunction>(load_symbol("cuMemAlloc_v2"));
    mem_alloc_managed_ =
        reinterpret_cast<MemAllocManagedFunction>(load_symbol("cuMemAllocManaged"));
    mem_alloc_pitch_ = reinterpret_cast<MemAllocPitchFunction>(load_symbol("cuMemAllocPitch_v2"));
    mem_alloc_async_ = reinterpret_cast<MemAllocAsyncFunction>(load_symbol("cuMemAllocAsync"));
    mem_alloc_async_ptsz_ =
        reinterpret_cast<MemAllocAsyncFunction>(load_symbol("cuMemAllocAsync_ptsz"));
    mem_alloc_from_pool_async_ =
        reinterpret_cast<MemAllocFromPoolAsyncFunction>(load_symbol("cuMemAllocFromPoolAsync"));
    mem_alloc_from_pool_async_ptsz_ = reinterpret_cast<MemAllocFromPoolAsyncFunction>(
        load_symbol("cuMemAllocFromPoolAsync_ptsz"));
    mem_create_ = reinterpret_cast<MemCreateFunction>(load_symbol("cuMemCreate"));
    mem_release_ = reinterpret_cast<MemReleaseFunction>(load_symbol("cuMemRelease"));
    mem_address_reserve_ =
        reinterpret_cast<MemAddressReserveFunction>(load_symbol("cuMemAddressReserve"));
    mem_address_free_ = reinterpret_cast<MemAddressFreeFunction>(load_symbol("cuMemAddressFree"));
    mem_map_ = reinterpret_cast<MemMapFunction>(load_symbol("cuMemMap"));
    mem_unmap_ = reinterpret_cast<MemUnmapFunction>(load_symbol("cuMemUnmap"));
    mem_set_access_ = reinterpret_cast<MemSetAccessFunction>(load_symbol("cuMemSetAccess"));
    mem_get_address_range_ =
        reinterpret_cast<MemGetAddressRangeFunction>(load_symbol("cuMemGetAddressRange_v2"));
    if (mem_get_address_range_ == nullptr) {
        mem_get_address_range_ =
            reinterpret_cast<MemGetAddressRangeFunction>(load_symbol("cuMemGetAddressRange"));
    }
    mem_get_access_ = reinterpret_cast<MemGetAccessFunction>(load_symbol("cuMemGetAccess"));
    mem_export_to_shareable_handle_ = reinterpret_cast<MemExportToShareableHandleFunction>(
        load_symbol("cuMemExportToShareableHandle"));
    mem_import_from_shareable_handle_ = reinterpret_cast<MemImportFromShareableHandleFunction>(
        load_symbol("cuMemImportFromShareableHandle"));
    mem_get_allocation_granularity_ = reinterpret_cast<MemGetAllocationGranularityFunction>(
        load_symbol("cuMemGetAllocationGranularity"));
    mem_get_allocation_properties_ = reinterpret_cast<MemGetAllocationPropertiesFunction>(
        load_symbol("cuMemGetAllocationPropertiesFromHandle"));
    mem_retain_allocation_handle_ = reinterpret_cast<MemRetainAllocationHandleFunction>(
        load_symbol("cuMemRetainAllocationHandle"));
    mem_pool_trim_to_ = reinterpret_cast<MemPoolTrimToFunction>(load_symbol("cuMemPoolTrimTo"));
    mem_pool_set_attribute_ =
        reinterpret_cast<MemPoolSetAttributeFunction>(load_symbol("cuMemPoolSetAttribute"));
    mem_pool_get_attribute_ =
        reinterpret_cast<MemPoolGetAttributeFunction>(load_symbol("cuMemPoolGetAttribute"));
    mem_pool_set_access_ =
        reinterpret_cast<MemPoolSetAccessFunction>(load_symbol("cuMemPoolSetAccess"));
    mem_pool_get_access_ =
        reinterpret_cast<MemPoolGetAccessFunction>(load_symbol("cuMemPoolGetAccess"));
    mem_pool_create_ = reinterpret_cast<MemPoolCreateFunction>(load_symbol("cuMemPoolCreate"));
    mem_pool_destroy_ = reinterpret_cast<MemPoolDestroyFunction>(load_symbol("cuMemPoolDestroy"));
    device_get_mem_pool_ =
        reinterpret_cast<DeviceGetMemPoolFunction>(load_symbol("cuDeviceGetMemPool"));
    device_set_mem_pool_ =
        reinterpret_cast<DeviceSetMemPoolFunction>(load_symbol("cuDeviceSetMemPool"));
    device_get_default_mem_pool_ =
        reinterpret_cast<DeviceGetDefaultMemPoolFunction>(load_symbol("cuDeviceGetDefaultMemPool"));
    mem_get_default_mem_pool_ =
        reinterpret_cast<MemGetDefaultMemPoolFunction>(load_symbol("cuMemGetDefaultMemPool"));
    mem_get_mem_pool_ = reinterpret_cast<MemGetMemPoolFunction>(load_symbol("cuMemGetMemPool"));
    mem_set_mem_pool_ = reinterpret_cast<MemSetMemPoolFunction>(load_symbol("cuMemSetMemPool"));
    mem_pool_export_to_shareable_handle_ = reinterpret_cast<MemPoolExportToShareableHandleFunction>(
        load_symbol("cuMemPoolExportToShareableHandle"));
    mem_pool_import_from_shareable_handle_ =
        reinterpret_cast<MemPoolImportFromShareableHandleFunction>(
            load_symbol("cuMemPoolImportFromShareableHandle"));
    mem_pool_export_pointer_ =
        reinterpret_cast<MemPoolExportPointerFunction>(load_symbol("cuMemPoolExportPointer"));
    mem_pool_import_pointer_ =
        reinterpret_cast<MemPoolImportPointerFunction>(load_symbol("cuMemPoolImportPointer"));
    mem_free_ = reinterpret_cast<MemFreeFunction>(load_symbol("cuMemFree_v2"));
    mem_free_async_ = reinterpret_cast<MemFreeAsyncFunction>(load_symbol("cuMemFreeAsync"));
    mem_free_async_ptsz_ =
        reinterpret_cast<MemFreeAsyncFunction>(load_symbol("cuMemFreeAsync_ptsz"));
    mem_get_info_ = reinterpret_cast<MemGetInfoFunction>(load_symbol("cuMemGetInfo_v2"));
    device_total_mem_ =
        reinterpret_cast<DeviceTotalMemFunction>(load_symbol("cuDeviceTotalMem_v2"));
    context_synchronize_ =
        reinterpret_cast<ContextSynchronizeFunction>(load_symbol("cuCtxSynchronize"));
    context_get_current_ =
        reinterpret_cast<ContextGetCurrentFunction>(load_symbol("cuCtxGetCurrent"));
    context_get_device_ = reinterpret_cast<ContextGetDeviceFunction>(load_symbol("cuCtxGetDevice"));
    context_destroy_ = reinterpret_cast<ContextDestroyFunction>(load_symbol("cuCtxDestroy_v2"));
    stream_get_device_ =
        reinterpret_cast<StreamGetDeviceFunction>(load_symbol("cuStreamGetDevice"));
    stream_get_device_ptsz_ =
        reinterpret_cast<StreamGetDeviceFunction>(load_symbol("cuStreamGetDevice_ptsz"));
    stream_get_context_ = reinterpret_cast<StreamGetContextFunction>(load_symbol("cuStreamGetCtx"));
    stream_get_context_ptsz_ =
        reinterpret_cast<StreamGetContextFunction>(load_symbol("cuStreamGetCtx_ptsz"));
    stream_query_ = reinterpret_cast<StreamQueryFunction>(load_symbol("cuStreamQuery"));
    stream_query_ptsz_ = reinterpret_cast<StreamQueryFunction>(load_symbol("cuStreamQuery_ptsz"));
    stream_synchronize_ =
        reinterpret_cast<StreamSynchronizeFunction>(load_symbol("cuStreamSynchronize"));
    stream_synchronize_ptsz_ =
        reinterpret_cast<StreamSynchronizeFunction>(load_symbol("cuStreamSynchronize_ptsz"));
    stream_destroy_ = reinterpret_cast<StreamDestroyFunction>(load_symbol("cuStreamDestroy_v2"));
    if (stream_destroy_ == nullptr) {
        stream_destroy_ = reinterpret_cast<StreamDestroyFunction>(load_symbol("cuStreamDestroy"));
    }
    get_proc_address_ = reinterpret_cast<GetProcAddressFunction>(load_symbol("cuGetProcAddress"));
    get_proc_address_v2_ =
        reinterpret_cast<GetProcAddressV2Function>(load_symbol("cuGetProcAddress_v2"));
    if (init_ != nullptr && mem_alloc_ != nullptr && mem_free_ != nullptr &&
        mem_get_info_ != nullptr) {
        return true;
    }

    dlclose(library_handle_);
    library_handle_ = nullptr;
    return false;
}

CUresult DriverDispatch::mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) const {
    DriverCallScope scope;
    if (mem_alloc_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_(device_pointer, memory_bytes);
}

CUresult DriverDispatch::init(unsigned int flags) const {
    DriverCallScope scope;
    if (init_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return init_(flags);
}

CUresult DriverDispatch::launch_kernel(CUfunction function, unsigned int grid_dim_x,
                                       unsigned int grid_dim_y, unsigned int grid_dim_z,
                                       unsigned int block_dim_x, unsigned int block_dim_y,
                                       unsigned int block_dim_z, unsigned int shared_memory_bytes,
                                       CUstream stream, void** kernel_parameters,
                                       void** extra) const {
    DriverCallScope scope;
    if (launch_kernel_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return launch_kernel_(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                          block_dim_z, shared_memory_bytes, stream, kernel_parameters, extra);
}

CUresult DriverDispatch::launch_kernel_ptsz(CUfunction function, unsigned int grid_dim_x,
                                            unsigned int grid_dim_y, unsigned int grid_dim_z,
                                            unsigned int block_dim_x, unsigned int block_dim_y,
                                            unsigned int block_dim_z,
                                            unsigned int shared_memory_bytes, CUstream stream,
                                            void** kernel_parameters, void** extra) const {
    DriverCallScope scope;
    if (launch_kernel_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return launch_kernel_ptsz_(function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                               block_dim_y, block_dim_z, shared_memory_bytes, stream,
                               kernel_parameters, extra);
}

CUresult DriverDispatch::mem_alloc_managed(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                           unsigned int flags) const {
    DriverCallScope scope;
    if (mem_alloc_managed_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_managed_(device_pointer, memory_bytes, flags);
}

CUresult DriverDispatch::mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                         std::size_t width_bytes, std::size_t height,
                                         unsigned int element_size_bytes) const {
    DriverCallScope scope;
    if (mem_alloc_pitch_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_pitch_(device_pointer, pitch, width_bytes, height, element_size_bytes);
}

CUresult DriverDispatch::mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                         CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_async_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_async_(device_pointer, memory_bytes, stream);
}

CUresult DriverDispatch::mem_alloc_async_ptsz(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_async_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_async_ptsz_(device_pointer, memory_bytes, stream);
}

CUresult DriverDispatch::mem_alloc_from_pool_async(CUdeviceptr* device_pointer,
                                                   std::size_t memory_bytes, CUmemoryPool pool,
                                                   CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_from_pool_async_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_from_pool_async_(device_pointer, memory_bytes, pool, stream);
}

CUresult DriverDispatch::mem_alloc_from_pool_async_ptsz(CUdeviceptr* device_pointer,
                                                        std::size_t memory_bytes, CUmemoryPool pool,
                                                        CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_from_pool_async_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_from_pool_async_ptsz_(device_pointer, memory_bytes, pool, stream);
}

CUresult DriverDispatch::mem_create(CUmemGenericAllocationHandle* handle, std::size_t memory_bytes,
                                    const CUmemAllocationProp* prop,
                                    unsigned long long flags) const {
    DriverCallScope scope;
    if (mem_create_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_create_(handle, memory_bytes, prop, flags);
}

CUresult DriverDispatch::mem_release(CUmemGenericAllocationHandle handle) const {
    DriverCallScope scope;
    if (mem_release_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_release_(handle);
}

CUresult DriverDispatch::mem_address_reserve(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                             std::size_t alignment, CUdeviceptr requested_address,
                                             unsigned long long flags) const {
    DriverCallScope scope;
    if (mem_address_reserve_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_address_reserve_(device_pointer, memory_bytes, alignment, requested_address, flags);
}

CUresult DriverDispatch::mem_address_free(CUdeviceptr device_pointer,
                                          std::size_t memory_bytes) const {
    DriverCallScope scope;
    if (mem_address_free_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_address_free_(device_pointer, memory_bytes);
}

CUresult DriverDispatch::mem_map(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                 std::size_t offset, CUmemGenericAllocationHandle handle,
                                 unsigned long long flags) const {
    DriverCallScope scope;
    if (mem_map_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_map_(device_pointer, memory_bytes, offset, handle, flags);
}

CUresult DriverDispatch::mem_unmap(CUdeviceptr device_pointer, std::size_t memory_bytes) const {
    DriverCallScope scope;
    if (mem_unmap_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_unmap_(device_pointer, memory_bytes);
}

CUresult DriverDispatch::mem_set_access(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                        const CUmemAccessDesc* access_descriptors,
                                        std::size_t descriptor_count) const {
    DriverCallScope scope;
    if (mem_set_access_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_set_access_(device_pointer, memory_bytes, access_descriptors, descriptor_count);
}

CUresult DriverDispatch::mem_get_address_range(CUdeviceptr* base_pointer, std::size_t* memory_bytes,
                                               CUdeviceptr device_pointer) const {
    DriverCallScope scope;
    if (mem_get_address_range_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_address_range_(base_pointer, memory_bytes, device_pointer);
}

CUresult DriverDispatch::mem_get_access(unsigned long long* flags, const CUmemLocation* location,
                                        CUdeviceptr device_pointer) const {
    DriverCallScope scope;
    if (mem_get_access_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_access_(flags, location, device_pointer);
}

CUresult DriverDispatch::mem_export_to_shareable_handle(void* shareable_handle,
                                                        CUmemGenericAllocationHandle handle,
                                                        CUmemAllocationHandleType handle_type,
                                                        unsigned long long flags) const {
    DriverCallScope scope;
    if (mem_export_to_shareable_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_export_to_shareable_handle_(shareable_handle, handle, handle_type, flags);
}

CUresult DriverDispatch::mem_import_from_shareable_handle(
    CUmemGenericAllocationHandle* handle, void* os_handle,
    CUmemAllocationHandleType handle_type) const {
    DriverCallScope scope;
    if (mem_import_from_shareable_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_import_from_shareable_handle_(handle, os_handle, handle_type);
}

CUresult DriverDispatch::mem_get_allocation_granularity(
    std::size_t* granularity, const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) const {
    DriverCallScope scope;
    if (mem_get_allocation_granularity_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_allocation_granularity_(granularity, prop, option);
}

CUresult DriverDispatch::mem_get_allocation_properties(CUmemAllocationProp* prop,
                                                       CUmemGenericAllocationHandle handle) const {
    DriverCallScope scope;
    if (mem_get_allocation_properties_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_allocation_properties_(prop, handle);
}

CUresult DriverDispatch::mem_retain_allocation_handle(CUmemGenericAllocationHandle* handle,
                                                      void* device_pointer) const {
    DriverCallScope scope;
    if (mem_retain_allocation_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_retain_allocation_handle_(handle, device_pointer);
}

CUresult DriverDispatch::mem_pool_trim_to(CUmemoryPool pool, std::size_t min_bytes_to_keep) const {
    DriverCallScope scope;
    if (mem_pool_trim_to_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_trim_to_(pool, min_bytes_to_keep);
}

CUresult DriverDispatch::mem_pool_set_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                void* value) const {
    DriverCallScope scope;
    if (mem_pool_set_attribute_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_set_attribute_(pool, attribute, value);
}

CUresult DriverDispatch::mem_pool_get_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                void* value) const {
    DriverCallScope scope;
    if (mem_pool_get_attribute_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_get_attribute_(pool, attribute, value);
}

CUresult DriverDispatch::mem_pool_set_access(CUmemoryPool pool,
                                             const CUmemAccessDesc* access_descriptors,
                                             std::size_t descriptor_count) const {
    DriverCallScope scope;
    if (mem_pool_set_access_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_set_access_(pool, access_descriptors, descriptor_count);
}

CUresult DriverDispatch::mem_pool_get_access(CUmemAccess_flags* flags, CUmemoryPool pool,
                                             CUmemLocation* location) const {
    DriverCallScope scope;
    if (mem_pool_get_access_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_get_access_(flags, pool, location);
}

CUresult DriverDispatch::mem_pool_create(CUmemoryPool* pool,
                                         const CUmemPoolProps* properties) const {
    DriverCallScope scope;
    if (mem_pool_create_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_create_(pool, properties);
}

CUresult DriverDispatch::mem_pool_destroy(CUmemoryPool pool) const {
    DriverCallScope scope;
    if (mem_pool_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_destroy_(pool);
}

CUresult DriverDispatch::device_get_mem_pool(CUmemoryPool* pool, CUdevice device) const {
    DriverCallScope scope;
    if (device_get_mem_pool_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return device_get_mem_pool_(pool, device);
}

CUresult DriverDispatch::device_set_mem_pool(CUdevice device, CUmemoryPool pool) const {
    DriverCallScope scope;
    if (device_set_mem_pool_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return device_set_mem_pool_(device, pool);
}

CUresult DriverDispatch::device_get_default_mem_pool(CUmemoryPool* pool, CUdevice device) const {
    DriverCallScope scope;
    if (device_get_default_mem_pool_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return device_get_default_mem_pool_(pool, device);
}

CUresult DriverDispatch::mem_get_default_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                                  CUmemAllocationType allocation_type) const {
    DriverCallScope scope;
    if (mem_get_default_mem_pool_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_default_mem_pool_(pool, location, allocation_type);
}

CUresult DriverDispatch::mem_get_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                          CUmemAllocationType allocation_type) const {
    DriverCallScope scope;
    if (mem_get_mem_pool_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_mem_pool_(pool, location, allocation_type);
}

CUresult DriverDispatch::mem_set_mem_pool(CUmemLocation* location,
                                          CUmemAllocationType allocation_type,
                                          CUmemoryPool pool) const {
    DriverCallScope scope;
    if (mem_set_mem_pool_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_set_mem_pool_(location, allocation_type, pool);
}

CUresult DriverDispatch::mem_pool_export_to_shareable_handle(void* handle_out, CUmemoryPool pool,
                                                             CUmemAllocationHandleType handle_type,
                                                             unsigned long long flags) const {
    DriverCallScope scope;
    if (mem_pool_export_to_shareable_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_export_to_shareable_handle_(handle_out, pool, handle_type, flags);
}

CUresult DriverDispatch::mem_pool_import_from_shareable_handle(
    CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handle_type,
    unsigned long long flags) const {
    DriverCallScope scope;
    if (mem_pool_import_from_shareable_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_import_from_shareable_handle_(pool_out, handle, handle_type, flags);
}

CUresult DriverDispatch::mem_pool_export_pointer(CUmemPoolPtrExportData* share_data_out,
                                                 CUdeviceptr device_pointer) const {
    DriverCallScope scope;
    if (mem_pool_export_pointer_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_export_pointer_(share_data_out, device_pointer);
}

CUresult DriverDispatch::mem_pool_import_pointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                                 CUmemPoolPtrExportData* share_data) const {
    DriverCallScope scope;
    if (mem_pool_import_pointer_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_pool_import_pointer_(pointer_out, pool, share_data);
}

CUresult DriverDispatch::mem_free(CUdeviceptr device_pointer) const {
    DriverCallScope scope;
    if (mem_free_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_free_(device_pointer);
}

CUresult DriverDispatch::mem_free_async(CUdeviceptr device_pointer, CUstream stream) const {
    DriverCallScope scope;
    if (mem_free_async_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_free_async_(device_pointer, stream);
}

CUresult DriverDispatch::mem_free_async_ptsz(CUdeviceptr device_pointer, CUstream stream) const {
    DriverCallScope scope;
    if (mem_free_async_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_free_async_ptsz_(device_pointer, stream);
}

CUresult DriverDispatch::mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) const {
    DriverCallScope scope;
    if (mem_get_info_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_info_(free_bytes, total_bytes);
}

CUresult DriverDispatch::device_total_mem(std::size_t* total_bytes, CUdevice device) const {
    DriverCallScope scope;
    if (device_total_mem_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return device_total_mem_(total_bytes, device);
}

CUresult DriverDispatch::context_synchronize() const {
    DriverCallScope scope;
    if (context_synchronize_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_synchronize_();
}

CUresult DriverDispatch::context_get_current(CUcontext* context) const {
    DriverCallScope scope;
    if (context_get_current_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_get_current_(context);
}

CUresult DriverDispatch::context_get_device(CUdevice* device) const {
    DriverCallScope scope;
    if (context_get_device_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_get_device_(device);
}

CUresult DriverDispatch::context_destroy(CUcontext context) const {
    DriverCallScope scope;
    if (context_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_destroy_(context);
}

CUresult DriverDispatch::stream_get_device(CUstream stream, CUdevice* device) const {
    DriverCallScope scope;
    if (stream_get_device_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_device_(stream, device);
}

CUresult DriverDispatch::stream_get_device_ptsz(CUstream stream, CUdevice* device) const {
    DriverCallScope scope;
    if (stream_get_device_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_device_ptsz_(stream, device);
}

CUresult DriverDispatch::stream_get_context(CUstream stream, CUcontext* context) const {
    DriverCallScope scope;
    if (stream_get_context_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_context_(stream, context);
}

CUresult DriverDispatch::stream_get_context_ptsz(CUstream stream, CUcontext* context) const {
    DriverCallScope scope;
    if (stream_get_context_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_context_ptsz_(stream, context);
}

CUresult DriverDispatch::stream_query(CUstream stream) const {
    DriverCallScope scope;
    if (stream_query_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_query_(stream);
}

CUresult DriverDispatch::stream_query_ptsz(CUstream stream) const {
    DriverCallScope scope;
    if (stream_query_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_query_ptsz_(stream);
}

CUresult DriverDispatch::stream_synchronize(CUstream stream) const {
    DriverCallScope scope;
    if (stream_synchronize_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_synchronize_(stream);
}

CUresult DriverDispatch::stream_synchronize_ptsz(CUstream stream) const {
    DriverCallScope scope;
    if (stream_synchronize_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_synchronize_ptsz_(stream);
}

CUresult DriverDispatch::stream_destroy(CUstream stream) const {
    DriverCallScope scope;
    if (stream_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_destroy_(stream);
}

CUresult DriverDispatch::get_proc_address(const char* symbol, void** function_pointer,
                                          int cuda_version, cuuint64_t flags) const {
    DriverCallScope scope;
    if (get_proc_address_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return get_proc_address_(symbol, function_pointer, cuda_version, flags);
}

CUresult DriverDispatch::get_proc_address_v2(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags,
                                             CUdriverProcAddressQueryResult* symbol_status) const {
    DriverCallScope scope;
    if (get_proc_address_v2_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return get_proc_address_v2_(symbol, function_pointer, cuda_version, flags, symbol_status);
}

bool DriverDispatch::has_get_proc_address() const {
    return get_proc_address_ != nullptr;
}

bool DriverDispatch::has_get_proc_address_v2() const {
    return get_proc_address_v2_ != nullptr;
}

bool DriverDispatch::has_launch_kernel() const {
    return launch_kernel_ != nullptr;
}

bool DriverDispatch::has_launch_kernel_ptsz() const {
    return launch_kernel_ptsz_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_managed() const {
    return mem_alloc_managed_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_pitch() const {
    return mem_alloc_pitch_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_async() const {
    return mem_alloc_async_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_async_ptsz() const {
    return mem_alloc_async_ptsz_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_from_pool_async() const {
    return mem_alloc_from_pool_async_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_from_pool_async_ptsz() const {
    return mem_alloc_from_pool_async_ptsz_ != nullptr;
}

bool DriverDispatch::has_mem_create() const {
    return mem_create_ != nullptr;
}

bool DriverDispatch::has_mem_release() const {
    return mem_release_ != nullptr;
}

bool DriverDispatch::has_mem_address_reserve() const {
    return mem_address_reserve_ != nullptr;
}

bool DriverDispatch::has_mem_address_free() const {
    return mem_address_free_ != nullptr;
}

bool DriverDispatch::has_mem_map() const {
    return mem_map_ != nullptr;
}

bool DriverDispatch::has_mem_unmap() const {
    return mem_unmap_ != nullptr;
}

bool DriverDispatch::has_mem_set_access() const {
    return mem_set_access_ != nullptr;
}

bool DriverDispatch::has_mem_get_address_range() const {
    return mem_get_address_range_ != nullptr;
}

bool DriverDispatch::has_mem_get_access() const {
    return mem_get_access_ != nullptr;
}

bool DriverDispatch::has_mem_export_to_shareable_handle() const {
    return mem_export_to_shareable_handle_ != nullptr;
}

bool DriverDispatch::has_mem_import_from_shareable_handle() const {
    return mem_import_from_shareable_handle_ != nullptr;
}

bool DriverDispatch::has_mem_get_allocation_granularity() const {
    return mem_get_allocation_granularity_ != nullptr;
}

bool DriverDispatch::has_mem_get_allocation_properties() const {
    return mem_get_allocation_properties_ != nullptr;
}

bool DriverDispatch::has_mem_retain_allocation_handle() const {
    return mem_retain_allocation_handle_ != nullptr;
}

bool DriverDispatch::has_mem_pool_trim_to() const {
    return mem_pool_trim_to_ != nullptr;
}

bool DriverDispatch::has_mem_pool_set_attribute() const {
    return mem_pool_set_attribute_ != nullptr;
}

bool DriverDispatch::has_mem_pool_get_attribute() const {
    return mem_pool_get_attribute_ != nullptr;
}

bool DriverDispatch::has_mem_pool_set_access() const {
    return mem_pool_set_access_ != nullptr;
}

bool DriverDispatch::has_mem_pool_get_access() const {
    return mem_pool_get_access_ != nullptr;
}

bool DriverDispatch::has_mem_pool_create() const {
    return mem_pool_create_ != nullptr;
}

bool DriverDispatch::has_mem_pool_destroy() const {
    return mem_pool_destroy_ != nullptr;
}

bool DriverDispatch::has_device_get_mem_pool() const {
    return device_get_mem_pool_ != nullptr;
}

bool DriverDispatch::has_device_set_mem_pool() const {
    return device_set_mem_pool_ != nullptr;
}

bool DriverDispatch::has_device_get_default_mem_pool() const {
    return device_get_default_mem_pool_ != nullptr;
}

bool DriverDispatch::has_mem_get_default_mem_pool() const {
    return mem_get_default_mem_pool_ != nullptr;
}

bool DriverDispatch::has_mem_get_mem_pool() const {
    return mem_get_mem_pool_ != nullptr;
}

bool DriverDispatch::has_mem_set_mem_pool() const {
    return mem_set_mem_pool_ != nullptr;
}

bool DriverDispatch::has_mem_pool_export_to_shareable_handle() const {
    return mem_pool_export_to_shareable_handle_ != nullptr;
}

bool DriverDispatch::has_mem_pool_import_from_shareable_handle() const {
    return mem_pool_import_from_shareable_handle_ != nullptr;
}

bool DriverDispatch::has_mem_pool_export_pointer() const {
    return mem_pool_export_pointer_ != nullptr;
}

bool DriverDispatch::has_mem_pool_import_pointer() const {
    return mem_pool_import_pointer_ != nullptr;
}

bool DriverDispatch::has_mem_free_async() const {
    return mem_free_async_ != nullptr;
}

bool DriverDispatch::has_mem_free_async_ptsz() const {
    return mem_free_async_ptsz_ != nullptr;
}

bool DriverDispatch::has_device_total_mem() const {
    return device_total_mem_ != nullptr;
}

bool DriverDispatch::has_context_synchronize() const {
    return context_synchronize_ != nullptr;
}

bool DriverDispatch::has_context_queries() const {
    return context_get_current_ != nullptr && context_get_device_ != nullptr;
}

bool DriverDispatch::has_context_destroy() const {
    return context_destroy_ != nullptr;
}

bool DriverDispatch::has_stream_identity() const {
    return stream_get_device_ != nullptr && stream_get_context_ != nullptr;
}

bool DriverDispatch::has_stream_identity_ptsz() const {
    return stream_get_device_ptsz_ != nullptr && stream_get_context_ptsz_ != nullptr;
}

bool DriverDispatch::has_stream_query() const {
    return stream_query_ != nullptr;
}

bool DriverDispatch::has_stream_query_ptsz() const {
    return stream_query_ptsz_ != nullptr;
}

bool DriverDispatch::has_stream_synchronize() const {
    return stream_synchronize_ != nullptr;
}

bool DriverDispatch::has_stream_synchronize_ptsz() const {
    return stream_synchronize_ptsz_ != nullptr;
}

bool DriverDispatch::has_stream_destroy() const {
    return stream_destroy_ != nullptr;
}

void* DriverDispatch::load_symbol(const char* name) const {
    const DlsymFunction real_dlsym = resolve_real_dlsym();
    if (real_dlsym == nullptr) {
        return nullptr;
    }

    dlerror();
    void* symbol = real_dlsym(library_handle_, name);
    if (dlerror() != nullptr) {
        return nullptr;
    }
    return symbol;
}

}  // namespace glimmer::interceptor
