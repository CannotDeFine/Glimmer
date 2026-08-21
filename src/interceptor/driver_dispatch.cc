#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_dispatch.h"

#include <dlfcn.h>
#include <link.h>

#include <string_view>

namespace glimmer::interceptor {

namespace {

thread_local bool g_is_inside_driver_call = false;
thread_local bool g_is_inside_proc_address_dispatch = false;

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

class ProcAddressDispatchScope {
   public:
    ProcAddressDispatchScope() : previous_state_(g_is_inside_proc_address_dispatch) {
        g_is_inside_proc_address_dispatch = true;
    }

    ~ProcAddressDispatchScope() {
        g_is_inside_proc_address_dispatch = previous_state_;
    }

   private:
    bool previous_state_;
};

[[nodiscard]] bool belongs_to_cuda_driver(void* symbol) noexcept {
    if (symbol == nullptr) {
        return false;
    }

    Dl_info symbol_info{};
    if (dladdr(symbol, &symbol_info) == 0 || symbol_info.dli_fname == nullptr) {
        return false;
    }
    return std::string_view(symbol_info.dli_fname).find("libcuda.so") != std::string_view::npos;
}

[[nodiscard]] const char* library_path(void* handle) noexcept {
    if (handle == nullptr) {
        return nullptr;
    }

    void* link_map_storage = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, reinterpret_cast<void*>(&link_map_storage)) != 0 ||
        link_map_storage == nullptr) {
        return nullptr;
    }
    const auto* link_map = static_cast<const struct link_map*>(link_map_storage);
    return link_map->l_name;
}

struct SecondaryCudaLibrarySearch {
    std::string_view primary_path;
    const char* secondary_path = nullptr;
};

int find_secondary_cuda_library(struct dl_phdr_info* info, std::size_t, void* data) noexcept {
    if (info == nullptr || data == nullptr || info->dlpi_name == nullptr ||
        info->dlpi_name[0] == '\0') {
        return 0;
    }

    auto* search = static_cast<SecondaryCudaLibrarySearch*>(data);
    const std::string_view path{info->dlpi_name};
    if (path.find("libcuda.so") == std::string_view::npos || path == search->primary_path) {
        return 0;
    }

    // WSL keeps a small libcuda loader in /usr/lib/wsl/lib and maps the actual
    // vendor Driver from /usr/lib/wsl/drivers/.../libcuda.so.1.1. The latter
    // must be preferred because the loader's exported stubs can resolve back
    // through the preload scope.
    if (path.find("libcuda.so.1.1") != std::string_view::npos) {
        search->secondary_path = info->dlpi_name;
        return 1;
    }
    if (search->secondary_path == nullptr) {
        search->secondary_path = info->dlpi_name;
    }
    return 0;
}

void prime_cuda_driver_symbols(void* primary_handle) noexcept {
    if (primary_handle == nullptr) {
        return;
    }

    // A split Driver may not map its vendor object until the first lookup.
    // Use the libc resolver directly so that lookup can trigger that mapping
    // without entering Glimmer's dlsym wrapper.
    const DlsymFunction real_dlsym = resolve_real_dlsym();
    if (real_dlsym == nullptr) {
        return;
    }

    (void)dlerror();
    void* symbol = real_dlsym(primary_handle, "cuInit");
    const char* error = dlerror();
    if (symbol == nullptr || error != nullptr) {
        return;
    }
}

[[nodiscard]] void* open_cuda_driver() noexcept {
    constexpr int k_flags = RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND;
    void* primary_handle = dlopen("libcuda.so.1", k_flags);
    if (primary_handle == nullptr) {
        return nullptr;
    }

    prime_cuda_driver_symbols(primary_handle);
    const char* primary_path = library_path(primary_handle);
    SecondaryCudaLibrarySearch search{
        .primary_path = primary_path == nullptr ? std::string_view{} : primary_path,
    };
    dl_iterate_phdr(&find_secondary_cuda_library, &search);
    if (search.secondary_path != nullptr) {
        if (void* secondary_handle = dlopen(search.secondary_path, k_flags);
            secondary_handle != nullptr) {
            return secondary_handle;
        }
    }
    return primary_handle;
}

}  // namespace

bool is_inside_driver_call() noexcept {
    return g_is_inside_driver_call;
}

DlsymFunction resolve_real_dlsym() noexcept {
    static DlsymFunction real_dlsym = []() noexcept {
        // Resolve dlsym from libc itself. RTLD_NEXT depends on the current
        // link-map position and can resolve back through another interposer
        // when CUDA and Glimmer are loaded in a different order.
        void* libc_handle = dlopen("libc.so.6", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (libc_handle == nullptr) {
            libc_handle = dlopen("libc.so.6", RTLD_NOW | RTLD_LOCAL);
        }
        void* symbol =
            libc_handle == nullptr ? nullptr : dlvsym(libc_handle, "dlsym", "GLIBC_2.2.5");
        if (symbol == nullptr) {
            symbol = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        }
        return reinterpret_cast<DlsymFunction>(symbol);
    }();
    return real_dlsym;
}

DriverDispatch::~DriverDispatch() {
    // Keep libcuda loaded until the process exits. CUDA Runtime may register
    // later atexit handlers that still call into the Driver after Glimmer's
    // state is destroyed; unloading it here can turn normal teardown into a
    // use-after-unload crash. The operating system reclaims this handle at
    // process exit, which is the lifetime expected for a preload interceptor.
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
      mem_map_array_async_(functions.mem_map_array_async),
      mem_unmap_(functions.mem_unmap),
      mem_set_access_(functions.mem_set_access),
      mem_get_address_range_(functions.mem_get_address_range),
      mem_get_access_(functions.mem_get_access),
      mem_export_to_shareable_handle_(functions.mem_export_to_shareable_handle),
      mem_import_from_shareable_handle_(functions.mem_import_from_shareable_handle),
      ipc_get_mem_handle_(functions.ipc_get_mem_handle),
      ipc_open_mem_handle_(functions.ipc_open_mem_handle),
      ipc_close_mem_handle_(functions.ipc_close_mem_handle),
      import_external_memory_(functions.import_external_memory),
      external_memory_get_mapped_buffer_(functions.external_memory_get_mapped_buffer),
      external_memory_get_mapped_mipmapped_array_(
          functions.external_memory_get_mapped_mipmapped_array),
      destroy_external_memory_(functions.destroy_external_memory),
      array_create_(functions.array_create),
      array_3d_create_(functions.array_3d_create),
      array_destroy_(functions.array_destroy),
      mipmapped_array_create_(functions.mipmapped_array_create),
      mipmapped_array_destroy_(functions.mipmapped_array_destroy),
      graphics_unregister_resource_(functions.graphics_unregister_resource),
      graphics_subresource_get_mapped_array_(functions.graphics_subresource_get_mapped_array),
      graphics_resource_get_mapped_mipmapped_array_(
          functions.graphics_resource_get_mapped_mipmapped_array),
      graphics_resource_get_mapped_pointer_(functions.graphics_resource_get_mapped_pointer),
      graphics_resource_set_map_flags_(functions.graphics_resource_set_map_flags),
      graphics_map_resources_(functions.graphics_map_resources),
      graphics_unmap_resources_(functions.graphics_unmap_resources),
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
      event_create_(functions.event_create),
      event_record_(functions.event_record),
      event_query_(functions.event_query),
      event_destroy_(functions.event_destroy),
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
    // libcuda calls other Driver entry points internally. With LD_PRELOAD,
    // ordinary global symbol lookup can bind those internal calls back to our
    // wrappers (notably cuGetProcAddress), creating recursion when the quota
    // path queries physical device capacity. DEEPBIND keeps the driver's own
    // entry points ahead of the preload scope while preserving our explicit
    // wrapper calls at the application boundary.
    // Resolve the driver's own relocations before exposing any entry point.
    // Lazy binding can defer an internal CUDA call until after the preload
    // scope is active and bind that call back to a Glimmer wrapper.
    library_handle_ = open_cuda_driver();
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
    mem_map_array_async_ =
        reinterpret_cast<MemMapArrayAsyncFunction>(load_symbol("cuMemMapArrayAsync"));
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
    ipc_get_mem_handle_ =
        reinterpret_cast<IpcGetMemHandleFunction>(load_symbol("cuIpcGetMemHandle"));
    ipc_open_mem_handle_ =
        reinterpret_cast<IpcOpenMemHandleFunction>(load_symbol("cuIpcOpenMemHandle_v2"));
    if (ipc_open_mem_handle_ == nullptr) {
        ipc_open_mem_handle_ =
            reinterpret_cast<IpcOpenMemHandleFunction>(load_symbol("cuIpcOpenMemHandle"));
    }
    ipc_close_mem_handle_ =
        reinterpret_cast<IpcCloseMemHandleFunction>(load_symbol("cuIpcCloseMemHandle"));
    import_external_memory_ =
        reinterpret_cast<ImportExternalMemoryFunction>(load_symbol("cuImportExternalMemory"));
    external_memory_get_mapped_buffer_ = reinterpret_cast<ExternalMemoryGetMappedBufferFunction>(
        load_symbol("cuExternalMemoryGetMappedBuffer"));
    external_memory_get_mapped_mipmapped_array_ =
        reinterpret_cast<ExternalMemoryGetMappedMipmappedArrayFunction>(
            load_symbol("cuExternalMemoryGetMappedMipmappedArray"));
    destroy_external_memory_ =
        reinterpret_cast<DestroyExternalMemoryFunction>(load_symbol("cuDestroyExternalMemory"));
    array_create_ = reinterpret_cast<ArrayCreateFunction>(load_symbol("cuArrayCreate_v2"));
    if (array_create_ == nullptr) {
        array_create_ = reinterpret_cast<ArrayCreateFunction>(load_symbol("cuArrayCreate"));
    }
    array_3d_create_ = reinterpret_cast<Array3DCreateFunction>(load_symbol("cuArray3DCreate_v2"));
    if (array_3d_create_ == nullptr) {
        array_3d_create_ = reinterpret_cast<Array3DCreateFunction>(load_symbol("cuArray3DCreate"));
    }
    array_destroy_ = reinterpret_cast<ArrayDestroyFunction>(load_symbol("cuArrayDestroy"));
    mipmapped_array_create_ =
        reinterpret_cast<MipmappedArrayCreateFunction>(load_symbol("cuMipmappedArrayCreate"));
    mipmapped_array_destroy_ =
        reinterpret_cast<MipmappedArrayDestroyFunction>(load_symbol("cuMipmappedArrayDestroy"));
    graphics_unregister_resource_ = reinterpret_cast<GraphicsUnregisterResourceFunction>(
        load_symbol("cuGraphicsUnregisterResource"));
    graphics_subresource_get_mapped_array_ =
        reinterpret_cast<GraphicsSubResourceGetMappedArrayFunction>(
            load_symbol("cuGraphicsSubResourceGetMappedArray"));
    graphics_resource_get_mapped_mipmapped_array_ =
        reinterpret_cast<GraphicsResourceGetMappedMipmappedArrayFunction>(
            load_symbol("cuGraphicsResourceGetMappedMipmappedArray"));
    graphics_resource_get_mapped_pointer_ =
        reinterpret_cast<GraphicsResourceGetMappedPointerFunction>(
            load_symbol("cuGraphicsResourceGetMappedPointer_v2"));
    if (graphics_resource_get_mapped_pointer_ == nullptr) {
        graphics_resource_get_mapped_pointer_ =
            reinterpret_cast<GraphicsResourceGetMappedPointerFunction>(
                load_symbol("cuGraphicsResourceGetMappedPointer"));
    }
    graphics_resource_set_map_flags_ = reinterpret_cast<GraphicsResourceSetMapFlagsFunction>(
        load_symbol("cuGraphicsResourceSetMapFlags_v2"));
    if (graphics_resource_set_map_flags_ == nullptr) {
        graphics_resource_set_map_flags_ = reinterpret_cast<GraphicsResourceSetMapFlagsFunction>(
            load_symbol("cuGraphicsResourceSetMapFlags"));
    }
    graphics_map_resources_ =
        reinterpret_cast<GraphicsMapResourcesFunction>(load_symbol("cuGraphicsMapResources"));
    graphics_unmap_resources_ =
        reinterpret_cast<GraphicsUnmapResourcesFunction>(load_symbol("cuGraphicsUnmapResources"));
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
    event_create_ = reinterpret_cast<EventCreateFunction>(load_symbol("cuEventCreate"));
    event_record_ = reinterpret_cast<EventRecordFunction>(load_symbol("cuEventRecord"));
    event_query_ = reinterpret_cast<EventQueryFunction>(load_symbol("cuEventQuery"));
    event_destroy_ = reinterpret_cast<EventDestroyFunction>(load_symbol("cuEventDestroy_v2"));
    if (event_destroy_ == nullptr) {
        event_destroy_ = reinterpret_cast<EventDestroyFunction>(load_symbol("cuEventDestroy"));
    }
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

CUresult DriverDispatch::mem_map_array_async(CUarrayMapInfo* map_info_list, unsigned int count,
                                             CUstream stream) const {
    DriverCallScope scope;
    if (mem_map_array_async_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_map_array_async_(map_info_list, count, stream);
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

CUresult DriverDispatch::ipc_get_mem_handle(CUipcMemHandle* handle,
                                            CUdeviceptr device_pointer) const {
    DriverCallScope scope;
    if (ipc_get_mem_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return ipc_get_mem_handle_(handle, device_pointer);
}

CUresult DriverDispatch::ipc_open_mem_handle(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                             unsigned int flags) const {
    DriverCallScope scope;
    if (ipc_open_mem_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return ipc_open_mem_handle_(device_pointer, handle, flags);
}

CUresult DriverDispatch::ipc_close_mem_handle(CUdeviceptr device_pointer) const {
    DriverCallScope scope;
    if (ipc_close_mem_handle_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return ipc_close_mem_handle_(device_pointer);
}

CUresult DriverDispatch::import_external_memory(
    CUexternalMemory* external_memory, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc) const {
    DriverCallScope scope;
    if (import_external_memory_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return import_external_memory_(external_memory, handle_desc);
}

CUresult DriverDispatch::external_memory_get_mapped_buffer(
    CUdeviceptr* device_pointer, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc) const {
    DriverCallScope scope;
    if (external_memory_get_mapped_buffer_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return external_memory_get_mapped_buffer_(device_pointer, external_memory, buffer_desc);
}

CUresult DriverDispatch::external_memory_get_mapped_mipmapped_array(
    CUmipmappedArray* mipmap, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc) const {
    DriverCallScope scope;
    if (external_memory_get_mapped_mipmapped_array_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return external_memory_get_mapped_mipmapped_array_(mipmap, external_memory, mipmap_desc);
}

CUresult DriverDispatch::destroy_external_memory(CUexternalMemory external_memory) const {
    DriverCallScope scope;
    if (destroy_external_memory_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return destroy_external_memory_(external_memory);
}

CUresult DriverDispatch::array_create(CUarray* array,
                                      const CUDA_ARRAY_DESCRIPTOR* descriptor) const {
    DriverCallScope scope;
    if (array_create_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return array_create_(array, descriptor);
}

CUresult DriverDispatch::array_3d_create(CUarray* array,
                                         const CUDA_ARRAY3D_DESCRIPTOR* descriptor) const {
    DriverCallScope scope;
    if (array_3d_create_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return array_3d_create_(array, descriptor);
}

CUresult DriverDispatch::array_destroy(CUarray array) const {
    DriverCallScope scope;
    if (array_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return array_destroy_(array);
}

CUresult DriverDispatch::mipmapped_array_create(CUmipmappedArray* mipmap,
                                                const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                                unsigned int level_count) const {
    DriverCallScope scope;
    if (mipmapped_array_create_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mipmapped_array_create_(mipmap, descriptor, level_count);
}

CUresult DriverDispatch::mipmapped_array_destroy(CUmipmappedArray mipmap) const {
    DriverCallScope scope;
    if (mipmapped_array_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mipmapped_array_destroy_(mipmap);
}

CUresult DriverDispatch::graphics_unregister_resource(CUgraphicsResource resource) const {
    DriverCallScope scope;
    if (graphics_unregister_resource_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return graphics_unregister_resource_(resource);
}

CUresult DriverDispatch::graphics_subresource_get_mapped_array(CUarray* array,
                                                               CUgraphicsResource resource,
                                                               unsigned int array_index,
                                                               unsigned int mip_level) const {
    DriverCallScope scope;
    if (graphics_subresource_get_mapped_array_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return graphics_subresource_get_mapped_array_(array, resource, array_index, mip_level);
}

CUresult DriverDispatch::graphics_resource_get_mapped_mipmapped_array(
    CUmipmappedArray* mipmap, CUgraphicsResource resource) const {
    DriverCallScope scope;
    if (graphics_resource_get_mapped_mipmapped_array_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return graphics_resource_get_mapped_mipmapped_array_(mipmap, resource);
}

CUresult DriverDispatch::graphics_resource_get_mapped_pointer(CUdeviceptr* device_pointer,
                                                              std::size_t* size,
                                                              CUgraphicsResource resource) const {
    DriverCallScope scope;
    if (graphics_resource_get_mapped_pointer_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return graphics_resource_get_mapped_pointer_(device_pointer, size, resource);
}

CUresult DriverDispatch::graphics_resource_set_map_flags(CUgraphicsResource resource,
                                                         unsigned int flags) const {
    DriverCallScope scope;
    if (graphics_resource_set_map_flags_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return graphics_resource_set_map_flags_(resource, flags);
}

CUresult DriverDispatch::graphics_map_resources(unsigned int count, CUgraphicsResource* resources,
                                                CUstream stream) const {
    DriverCallScope scope;
    if (graphics_map_resources_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return graphics_map_resources_(count, resources, stream);
}

CUresult DriverDispatch::graphics_unmap_resources(unsigned int count, CUgraphicsResource* resources,
                                                  CUstream stream) const {
    DriverCallScope scope;
    if (graphics_unmap_resources_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return graphics_unmap_resources_(count, resources, stream);
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

CUresult DriverDispatch::event_create(CUevent* event, unsigned int flags) const {
    DriverCallScope scope;
    if (event_create_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return event_create_(event, flags);
}

CUresult DriverDispatch::event_record(CUevent event, CUstream stream) const {
    DriverCallScope scope;
    if (event_record_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return event_record_(event, stream);
}

CUresult DriverDispatch::event_query(CUevent event) const {
    DriverCallScope scope;
    if (event_query_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return event_query_(event);
}

CUresult DriverDispatch::event_destroy(CUevent event) const {
    DriverCallScope scope;
    if (event_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return event_destroy_(event);
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
    if (g_is_inside_proc_address_dispatch) {
        if (library_handle_ != nullptr && symbol != nullptr && function_pointer != nullptr) {
            *function_pointer = resolve_direct_symbol(symbol);
            return *function_pointer == nullptr ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    ProcAddressDispatchScope proc_scope;
    DriverCallScope scope;
    if (get_proc_address_ == nullptr ||
        (library_handle_ != nullptr &&
         !belongs_to_cuda_driver(reinterpret_cast<void*>(get_proc_address_)))) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return get_proc_address_(symbol, function_pointer, cuda_version, flags);
}

CUresult DriverDispatch::get_proc_address_v2(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags,
                                             CUdriverProcAddressQueryResult* symbol_status) const {
    if (g_is_inside_proc_address_dispatch) {
        if (library_handle_ != nullptr && symbol != nullptr && function_pointer != nullptr) {
            *function_pointer = resolve_direct_symbol(symbol);
            if (symbol_status != nullptr) {
                *symbol_status = *function_pointer == nullptr ? CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
                                                              : CU_GET_PROC_ADDRESS_SUCCESS;
            }
            return *function_pointer == nullptr ? CUDA_ERROR_NOT_FOUND : CUDA_SUCCESS;
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    ProcAddressDispatchScope proc_scope;
    DriverCallScope scope;
    if (get_proc_address_v2_ == nullptr ||
        (library_handle_ != nullptr &&
         !belongs_to_cuda_driver(reinterpret_cast<void*>(get_proc_address_v2_)))) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return get_proc_address_v2_(symbol, function_pointer, cuda_version, flags, symbol_status);
}

bool DriverDispatch::has_get_proc_address() const {
    return get_proc_address_ != nullptr &&
           (library_handle_ == nullptr ||
            belongs_to_cuda_driver(reinterpret_cast<void*>(get_proc_address_)));
}

bool DriverDispatch::has_get_proc_address_v2() const {
    return get_proc_address_v2_ != nullptr &&
           (library_handle_ == nullptr ||
            belongs_to_cuda_driver(reinterpret_cast<void*>(get_proc_address_v2_)));
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

bool DriverDispatch::has_mem_map_array_async() const {
    return mem_map_array_async_ != nullptr;
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

bool DriverDispatch::has_ipc_get_mem_handle() const {
    return ipc_get_mem_handle_ != nullptr;
}

bool DriverDispatch::has_ipc_open_mem_handle() const {
    return ipc_open_mem_handle_ != nullptr;
}

bool DriverDispatch::has_ipc_close_mem_handle() const {
    return ipc_close_mem_handle_ != nullptr;
}

bool DriverDispatch::has_import_external_memory() const {
    return import_external_memory_ != nullptr;
}

bool DriverDispatch::has_external_memory_get_mapped_buffer() const {
    return external_memory_get_mapped_buffer_ != nullptr;
}

bool DriverDispatch::has_external_memory_get_mapped_mipmapped_array() const {
    return external_memory_get_mapped_mipmapped_array_ != nullptr;
}

bool DriverDispatch::has_destroy_external_memory() const {
    return destroy_external_memory_ != nullptr;
}

bool DriverDispatch::has_array_create() const {
    return array_create_ != nullptr;
}

bool DriverDispatch::has_array_3d_create() const {
    return array_3d_create_ != nullptr;
}

bool DriverDispatch::has_array_destroy() const {
    return array_destroy_ != nullptr;
}

bool DriverDispatch::has_mipmapped_array_create() const {
    return mipmapped_array_create_ != nullptr;
}

bool DriverDispatch::has_mipmapped_array_destroy() const {
    return mipmapped_array_destroy_ != nullptr;
}

bool DriverDispatch::has_graphics_unregister_resource() const {
    return graphics_unregister_resource_ != nullptr;
}

bool DriverDispatch::has_graphics_subresource_get_mapped_array() const {
    return graphics_subresource_get_mapped_array_ != nullptr;
}

bool DriverDispatch::has_graphics_resource_get_mapped_mipmapped_array() const {
    return graphics_resource_get_mapped_mipmapped_array_ != nullptr;
}

bool DriverDispatch::has_graphics_resource_get_mapped_pointer() const {
    return graphics_resource_get_mapped_pointer_ != nullptr;
}

bool DriverDispatch::has_graphics_resource_set_map_flags() const {
    return graphics_resource_set_map_flags_ != nullptr;
}

bool DriverDispatch::has_graphics_map_resources() const {
    return graphics_map_resources_ != nullptr;
}

bool DriverDispatch::has_graphics_unmap_resources() const {
    return graphics_unmap_resources_ != nullptr;
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

bool DriverDispatch::has_event_api() const {
    return event_create_ != nullptr && event_record_ != nullptr && event_query_ != nullptr &&
           event_destroy_ != nullptr;
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

void* DriverDispatch::resolve_direct_symbol(const char* name) const {
    if (name == nullptr) {
        return nullptr;
    }

    // This path is used only while the real Driver resolver is already on the
    // stack. Calling dlsym again here can re-enter the loader and recreate the
    // recursion we are trying to break. Return the already-resolved Driver
    // entry points instead.
    const std::string_view symbol{name};
    if (symbol == "cuInit") {
        return reinterpret_cast<void*>(init_);
    }
    if (symbol == "cuLaunchKernel") {
        return reinterpret_cast<void*>(launch_kernel_);
    }
    if (symbol == "cuLaunchKernel_ptsz") {
        return reinterpret_cast<void*>(launch_kernel_ptsz_);
    }
    if (symbol == "cuMemAlloc_v2" || symbol == "cuMemAlloc") {
        return reinterpret_cast<void*>(mem_alloc_);
    }
    if (symbol == "cuMemFree_v2" || symbol == "cuMemFree") {
        return reinterpret_cast<void*>(mem_free_);
    }
    if (symbol == "cuMemGetInfo_v2" || symbol == "cuMemGetInfo") {
        return reinterpret_cast<void*>(mem_get_info_);
    }
    if (symbol == "cuMemMapArrayAsync") {
        return reinterpret_cast<void*>(mem_map_array_async_);
    }
    if (symbol == "cuIpcGetMemHandle") {
        return reinterpret_cast<void*>(ipc_get_mem_handle_);
    }
    if (symbol == "cuIpcOpenMemHandle") {
        return reinterpret_cast<void*>(ipc_open_mem_handle_);
    }
    if (symbol == "cuIpcOpenMemHandle_v2") {
        return reinterpret_cast<void*>(ipc_open_mem_handle_);
    }
    if (symbol == "cuIpcCloseMemHandle") {
        return reinterpret_cast<void*>(ipc_close_mem_handle_);
    }
    if (symbol == "cuImportExternalMemory") {
        return reinterpret_cast<void*>(import_external_memory_);
    }
    if (symbol == "cuExternalMemoryGetMappedBuffer") {
        return reinterpret_cast<void*>(external_memory_get_mapped_buffer_);
    }
    if (symbol == "cuExternalMemoryGetMappedMipmappedArray") {
        return reinterpret_cast<void*>(external_memory_get_mapped_mipmapped_array_);
    }
    if (symbol == "cuDestroyExternalMemory") {
        return reinterpret_cast<void*>(destroy_external_memory_);
    }
    if (symbol == "cuArrayCreate" || symbol == "cuArrayCreate_v2") {
        return reinterpret_cast<void*>(array_create_);
    }
    if (symbol == "cuArray3DCreate" || symbol == "cuArray3DCreate_v2") {
        return reinterpret_cast<void*>(array_3d_create_);
    }
    if (symbol == "cuArrayDestroy") {
        return reinterpret_cast<void*>(array_destroy_);
    }
    if (symbol == "cuMipmappedArrayCreate") {
        return reinterpret_cast<void*>(mipmapped_array_create_);
    }
    if (symbol == "cuMipmappedArrayDestroy") {
        return reinterpret_cast<void*>(mipmapped_array_destroy_);
    }
    if (symbol == "cuGraphicsUnregisterResource") {
        return reinterpret_cast<void*>(graphics_unregister_resource_);
    }
    if (symbol == "cuGraphicsSubResourceGetMappedArray") {
        return reinterpret_cast<void*>(graphics_subresource_get_mapped_array_);
    }
    if (symbol == "cuGraphicsResourceGetMappedMipmappedArray") {
        return reinterpret_cast<void*>(graphics_resource_get_mapped_mipmapped_array_);
    }
    if (symbol == "cuGraphicsResourceGetMappedPointer" ||
        symbol == "cuGraphicsResourceGetMappedPointer_v2") {
        return reinterpret_cast<void*>(graphics_resource_get_mapped_pointer_);
    }
    if (symbol == "cuGraphicsResourceSetMapFlags" || symbol == "cuGraphicsResourceSetMapFlags_v2") {
        return reinterpret_cast<void*>(graphics_resource_set_map_flags_);
    }
    if (symbol == "cuGraphicsMapResources") {
        return reinterpret_cast<void*>(graphics_map_resources_);
    }
    if (symbol == "cuGraphicsUnmapResources") {
        return reinterpret_cast<void*>(graphics_unmap_resources_);
    }
    if (symbol == "cuDeviceTotalMem_v2" || symbol == "cuDeviceTotalMem") {
        return reinterpret_cast<void*>(device_total_mem_);
    }
    if (symbol == "cuCtxSynchronize") {
        return reinterpret_cast<void*>(context_synchronize_);
    }
    if (symbol == "cuCtxGetCurrent") {
        return reinterpret_cast<void*>(context_get_current_);
    }
    if (symbol == "cuCtxGetDevice") {
        return reinterpret_cast<void*>(context_get_device_);
    }
    if (symbol == "cuCtxDestroy_v2" || symbol == "cuCtxDestroy") {
        return reinterpret_cast<void*>(context_destroy_);
    }
    if (symbol == "cuStreamGetDevice") {
        return reinterpret_cast<void*>(stream_get_device_);
    }
    if (symbol == "cuStreamGetDevice_ptsz") {
        return reinterpret_cast<void*>(stream_get_device_ptsz_);
    }
    if (symbol == "cuStreamGetCtx") {
        return reinterpret_cast<void*>(stream_get_context_);
    }
    if (symbol == "cuStreamGetCtx_ptsz") {
        return reinterpret_cast<void*>(stream_get_context_ptsz_);
    }
    if (symbol == "cuStreamQuery") {
        return reinterpret_cast<void*>(stream_query_);
    }
    if (symbol == "cuStreamQuery_ptsz") {
        return reinterpret_cast<void*>(stream_query_ptsz_);
    }
    if (symbol == "cuStreamSynchronize") {
        return reinterpret_cast<void*>(stream_synchronize_);
    }
    if (symbol == "cuStreamSynchronize_ptsz") {
        return reinterpret_cast<void*>(stream_synchronize_ptsz_);
    }
    if (symbol == "cuStreamDestroy_v2" || symbol == "cuStreamDestroy") {
        return reinterpret_cast<void*>(stream_destroy_);
    }
    return nullptr;
}

void* DriverDispatch::load_symbol(const char* name) const {
    const DlsymFunction real_dlsym = resolve_real_dlsym();
    if (real_dlsym == nullptr || library_handle_ == nullptr || name == nullptr) {
        return nullptr;
    }

    dlerror();
    void* symbol = real_dlsym(library_handle_, name);
    return dlerror() == nullptr && belongs_to_cuda_driver(symbol) ? symbol : nullptr;
}

}  // namespace glimmer::interceptor
