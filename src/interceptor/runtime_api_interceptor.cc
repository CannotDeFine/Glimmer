#include "internal/runtime_api_bridge.h"

#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"

#include <cstddef>
#include <dlfcn.h>

#ifdef cudaMallocFromPoolAsync
#undef cudaMallocFromPoolAsync
#endif
#ifdef cudaLaunchKernel
#undef cudaLaunchKernel
#endif

namespace {

thread_local bool g_is_inside_runtime_call = false;

class RuntimeCallScope {
   public:
    RuntimeCallScope() : previous_state_(g_is_inside_runtime_call) {
        g_is_inside_runtime_call = true;
    }

    ~RuntimeCallScope() {
        g_is_inside_runtime_call = previous_state_;
    }

   private:
    bool previous_state_;
};

using glimmer::interceptor::DlsymFunction;
using glimmer::interceptor::RuntimeDestroyExternalMemoryFunction;
using glimmer::interceptor::RuntimeDeviceGetDefaultMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceGetMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceSetMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceSynchronizeFunction;
using glimmer::interceptor::RuntimeExternalMemoryGetMappedBufferFunction;
using glimmer::interceptor::RuntimeExternalMemoryGetMappedMipmappedArrayFunction;
using glimmer::interceptor::RuntimeFreeArrayFunction;
using glimmer::interceptor::RuntimeFreeAsyncFunction;
using glimmer::interceptor::RuntimeFreeFunction;
using glimmer::interceptor::RuntimeFreeMipmappedArrayFunction;
using glimmer::interceptor::RuntimeGetDeviceFunction;
using glimmer::interceptor::RuntimeGraphAddMemAllocNodeFunction;
using glimmer::interceptor::RuntimeGraphicsMapResourcesFunction;
using glimmer::interceptor::RuntimeGraphicsResourceGetMappedMipmappedArrayFunction;
using glimmer::interceptor::RuntimeGraphicsResourceGetMappedPointerFunction;
using glimmer::interceptor::RuntimeGraphicsResourceSetMapFlagsFunction;
using glimmer::interceptor::RuntimeGraphicsSubResourceGetMappedArrayFunction;
using glimmer::interceptor::RuntimeGraphicsUnmapResourcesFunction;
using glimmer::interceptor::RuntimeGraphicsUnregisterResourceFunction;
using glimmer::interceptor::RuntimeImportExternalMemoryFunction;
using glimmer::interceptor::RuntimeIpcCloseMemHandleFunction;
using glimmer::interceptor::RuntimeIpcGetMemHandleFunction;
using glimmer::interceptor::RuntimeIpcOpenMemHandleFunction;
using glimmer::interceptor::RuntimeMalloc3DArrayFunction;
using glimmer::interceptor::RuntimeMalloc3DFunction;
using glimmer::interceptor::RuntimeMallocArrayFunction;
using glimmer::interceptor::RuntimeMallocAsyncFunction;
using glimmer::interceptor::RuntimeMallocFromPoolAsyncFunction;
using glimmer::interceptor::RuntimeMallocFunction;
using glimmer::interceptor::RuntimeMallocManagedFunction;
using glimmer::interceptor::RuntimeMallocMipmappedArrayFunction;
using glimmer::interceptor::RuntimeMallocPitchFunction;
using RuntimeLaunchKernelFunction = cudaError_t (*)(const void* function, dim3 grid_dim,
                                                    dim3 block_dim, void** arguments,
                                                    std::size_t shared_memory_bytes,
                                                    cudaStream_t stream);
using RuntimeInternalLaunchKernelFunction = cudaError_t (*)(cudaKernel_t kernel, dim3 grid_dim,
                                                            dim3 block_dim, void** arguments,
                                                            std::size_t shared_memory_bytes,
                                                            cudaStream_t stream);
using glimmer::interceptor::RuntimeMemGetDefaultMemPoolFunction;
using glimmer::interceptor::RuntimeMemGetInfoFunction;
using glimmer::interceptor::RuntimeMemGetMemPoolFunction;
using glimmer::interceptor::RuntimeMemPoolCreateFunction;
using glimmer::interceptor::RuntimeMemPoolDestroyFunction;
using glimmer::interceptor::RuntimeMemPoolExportPointerFunction;
using glimmer::interceptor::RuntimeMemPoolExportToShareableHandleFunction;
using glimmer::interceptor::RuntimeMemPoolGetAccessFunction;
using glimmer::interceptor::RuntimeMemPoolGetAttributeFunction;
using glimmer::interceptor::RuntimeMemPoolImportFromShareableHandleFunction;
using glimmer::interceptor::RuntimeMemPoolImportPointerFunction;
using glimmer::interceptor::RuntimeMemPoolSetAccessFunction;
using glimmer::interceptor::RuntimeMemPoolSetAttributeFunction;
using glimmer::interceptor::RuntimeMemPoolTrimToFunction;
using glimmer::interceptor::RuntimeMemSetMemPoolFunction;
using glimmer::interceptor::RuntimeStreamDestroyFunction;
using glimmer::interceptor::RuntimeStreamQueryFunction;
using glimmer::interceptor::RuntimeStreamSynchronizeFunction;

[[nodiscard]] RuntimeMallocFunction resolve_runtime_malloc() noexcept {
    static RuntimeMallocFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMallocFunction>(real_dlsym(RTLD_NEXT, "cudaMalloc"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocManagedFunction resolve_runtime_malloc_managed() noexcept {
    static RuntimeMallocManagedFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocManagedFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocManaged"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocPitchFunction resolve_runtime_malloc_pitch() noexcept {
    static RuntimeMallocPitchFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocPitchFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocPitch"));
    }();
    return function;
}

[[nodiscard]] RuntimeMalloc3DFunction resolve_runtime_malloc_3d() noexcept {
    static RuntimeMalloc3DFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMalloc3DFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMalloc3D"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeFunction resolve_runtime_free() noexcept {
    static RuntimeFreeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeFreeFunction>(real_dlsym(RTLD_NEXT, "cudaFree"));
    }();
    return function;
}

[[nodiscard]] RuntimeIpcGetMemHandleFunction resolve_runtime_ipc_get_mem_handle() noexcept {
    static RuntimeIpcGetMemHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeIpcGetMemHandleFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaIpcGetMemHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeIpcOpenMemHandleFunction resolve_runtime_ipc_open_mem_handle() noexcept {
    static RuntimeIpcOpenMemHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeIpcOpenMemHandleFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaIpcOpenMemHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeIpcCloseMemHandleFunction resolve_runtime_ipc_close_mem_handle() noexcept {
    static RuntimeIpcCloseMemHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeIpcCloseMemHandleFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaIpcCloseMemHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeImportExternalMemoryFunction
resolve_runtime_import_external_memory() noexcept {
    static RuntimeImportExternalMemoryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeImportExternalMemoryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaImportExternalMemory"));
    }();
    return function;
}

[[nodiscard]] RuntimeExternalMemoryGetMappedBufferFunction
resolve_runtime_external_memory_get_mapped_buffer() noexcept {
    static RuntimeExternalMemoryGetMappedBufferFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeExternalMemoryGetMappedBufferFunction>(
                         real_dlsym(RTLD_NEXT, "cudaExternalMemoryGetMappedBuffer"));
    }();
    return function;
}

[[nodiscard]] RuntimeExternalMemoryGetMappedMipmappedArrayFunction
resolve_runtime_external_memory_get_mapped_mipmapped_array() noexcept {
    static RuntimeExternalMemoryGetMappedMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeExternalMemoryGetMappedMipmappedArrayFunction>(
                         real_dlsym(RTLD_NEXT, "cudaExternalMemoryGetMappedMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeDestroyExternalMemoryFunction
resolve_runtime_destroy_external_memory() noexcept {
    static RuntimeDestroyExternalMemoryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDestroyExternalMemoryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDestroyExternalMemory"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocArrayFunction resolve_runtime_malloc_array() noexcept {
    static RuntimeMallocArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeMalloc3DArrayFunction resolve_runtime_malloc_3d_array() noexcept {
    static RuntimeMalloc3DArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMalloc3DArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMalloc3DArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocMipmappedArrayFunction
resolve_runtime_malloc_mipmapped_array() noexcept {
    static RuntimeMallocMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocMipmappedArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeArrayFunction resolve_runtime_free_array() noexcept {
    static RuntimeFreeArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeMipmappedArrayFunction resolve_runtime_free_mipmapped_array() noexcept {
    static RuntimeFreeMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeMipmappedArrayFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsUnregisterResourceFunction
resolve_runtime_graphics_unregister_resource() noexcept {
    static RuntimeGraphicsUnregisterResourceFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphicsUnregisterResourceFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphicsUnregisterResource"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsResourceSetMapFlagsFunction
resolve_runtime_graphics_resource_set_map_flags() noexcept {
    static RuntimeGraphicsResourceSetMapFlagsFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsResourceSetMapFlagsFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsResourceSetMapFlags"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsMapResourcesFunction
resolve_runtime_graphics_map_resources() noexcept {
    static RuntimeGraphicsMapResourcesFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphicsMapResourcesFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphicsMapResources"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsUnmapResourcesFunction
resolve_runtime_graphics_unmap_resources() noexcept {
    static RuntimeGraphicsUnmapResourcesFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphicsUnmapResourcesFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphicsUnmapResources"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsResourceGetMappedPointerFunction
resolve_runtime_graphics_resource_get_mapped_pointer() noexcept {
    static RuntimeGraphicsResourceGetMappedPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsResourceGetMappedPointerFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedPointer"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsSubResourceGetMappedArrayFunction
resolve_runtime_graphics_subresource_get_mapped_array() noexcept {
    static RuntimeGraphicsSubResourceGetMappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsSubResourceGetMappedArrayFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsSubResourceGetMappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphicsResourceGetMappedMipmappedArrayFunction
resolve_runtime_graphics_resource_get_mapped_mipmapped_array() noexcept {
    static RuntimeGraphicsResourceGetMappedMipmappedArrayFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeGraphicsResourceGetMappedMipmappedArrayFunction>(
                         real_dlsym(RTLD_NEXT, "cudaGraphicsResourceGetMappedMipmappedArray"));
    }();
    return function;
}

[[nodiscard]] RuntimeGraphAddMemAllocNodeFunction
resolve_runtime_graph_add_mem_alloc_node() noexcept {
    static RuntimeGraphAddMemAllocNodeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGraphAddMemAllocNodeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGraphAddMemAllocNode"));
    }();
    return function;
}

[[nodiscard]] RuntimeGetDeviceFunction resolve_runtime_get_device() noexcept {
    static RuntimeGetDeviceFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGetDeviceFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGetDevice"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetInfoFunction resolve_runtime_mem_get_info() noexcept {
    static RuntimeMemGetInfoFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetInfoFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetInfo"));
    }();
    return function;
}

[[nodiscard]] RuntimeLaunchKernelFunction resolve_runtime_launch_kernel() noexcept {
    static RuntimeLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaLaunchKernel"));
    }();
    return function;
}

[[nodiscard]] RuntimeLaunchKernelFunction resolve_runtime_launch_kernel_ptsz() noexcept {
    static RuntimeLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaLaunchKernel_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeInternalLaunchKernelFunction
resolve_runtime_internal_launch_kernel() noexcept {
    static RuntimeInternalLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeInternalLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "__cudaLaunchKernel"));
    }();
    return function;
}

[[nodiscard]] RuntimeInternalLaunchKernelFunction
resolve_runtime_internal_launch_kernel_ptsz() noexcept {
    static RuntimeInternalLaunchKernelFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeInternalLaunchKernelFunction>(
                                           real_dlsym(RTLD_NEXT, "__cudaLaunchKernel_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocAsyncFunction resolve_runtime_malloc_async() noexcept {
    static RuntimeMallocAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocAsyncFunction resolve_runtime_malloc_async_ptsz() noexcept {
    static RuntimeMallocAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocFromPoolAsyncFunction resolve_runtime_malloc_from_pool_async() noexcept {
    static RuntimeMallocFromPoolAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocFromPoolAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocFromPoolAsyncFunction
resolve_runtime_malloc_from_pool_async_ptsz() noexcept {
    static RuntimeMallocFromPoolAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocFromPoolAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeAsyncFunction resolve_runtime_free_async() noexcept {
    static RuntimeFreeAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeAsyncFunction resolve_runtime_free_async_ptsz() noexcept {
    static RuntimeFreeAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceSynchronizeFunction resolve_runtime_device_synchronize() noexcept {
    static RuntimeDeviceSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamSynchronizeFunction resolve_runtime_stream_synchronize() noexcept {
    static RuntimeStreamSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamSynchronizeFunction resolve_runtime_stream_synchronize_ptsz() noexcept {
    static RuntimeStreamSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamSynchronize_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamQueryFunction resolve_runtime_stream_query() noexcept {
    static RuntimeStreamQueryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamQueryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamQuery"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamQueryFunction resolve_runtime_stream_query_ptsz() noexcept {
    static RuntimeStreamQueryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamQueryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamQuery_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamDestroyFunction resolve_runtime_stream_destroy() noexcept {
    static RuntimeStreamDestroyFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamDestroyFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamDestroy"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceGetDefaultMemPoolFunction
resolve_runtime_device_get_default_mem_pool() noexcept {
    static RuntimeDeviceGetDefaultMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceGetDefaultMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceGetDefaultMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceSetMemPoolFunction resolve_runtime_device_set_mem_pool() noexcept {
    static RuntimeDeviceSetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceSetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceSetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceGetMemPoolFunction resolve_runtime_device_get_mem_pool() noexcept {
    static RuntimeDeviceGetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceGetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceGetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolTrimToFunction resolve_runtime_mem_pool_trim_to() noexcept {
    static RuntimeMemPoolTrimToFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolTrimToFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolTrimTo"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolSetAttributeFunction resolve_runtime_mem_pool_set_attribute() noexcept {
    static RuntimeMemPoolSetAttributeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolSetAttributeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolSetAttribute"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolGetAttributeFunction resolve_runtime_mem_pool_get_attribute() noexcept {
    static RuntimeMemPoolGetAttributeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolGetAttributeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolGetAttribute"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolSetAccessFunction resolve_runtime_mem_pool_set_access() noexcept {
    static RuntimeMemPoolSetAccessFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolSetAccessFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolSetAccess"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolGetAccessFunction resolve_runtime_mem_pool_get_access() noexcept {
    static RuntimeMemPoolGetAccessFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolGetAccessFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolGetAccess"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolCreateFunction resolve_runtime_mem_pool_create() noexcept {
    static RuntimeMemPoolCreateFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolCreateFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolCreate"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolDestroyFunction resolve_runtime_mem_pool_destroy() noexcept {
    static RuntimeMemPoolDestroyFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolDestroyFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolDestroy"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetDefaultMemPoolFunction
resolve_runtime_mem_get_default_mem_pool() noexcept {
    static RuntimeMemGetDefaultMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetDefaultMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetDefaultMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetMemPoolFunction resolve_runtime_mem_get_mem_pool() noexcept {
    static RuntimeMemGetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemSetMemPoolFunction resolve_runtime_mem_set_mem_pool() noexcept {
    static RuntimeMemSetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemSetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemSetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolExportToShareableHandleFunction
resolve_runtime_mem_pool_export_to_shareable_handle() noexcept {
    static RuntimeMemPoolExportToShareableHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMemPoolExportToShareableHandleFunction>(
                         real_dlsym(RTLD_NEXT, "cudaMemPoolExportToShareableHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolImportFromShareableHandleFunction
resolve_runtime_mem_pool_import_from_shareable_handle() noexcept {
    static RuntimeMemPoolImportFromShareableHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMemPoolImportFromShareableHandleFunction>(
                         real_dlsym(RTLD_NEXT, "cudaMemPoolImportFromShareableHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolExportPointerFunction
resolve_runtime_mem_pool_export_pointer() noexcept {
    static RuntimeMemPoolExportPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolExportPointerFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolExportPointer"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolImportPointerFunction
resolve_runtime_mem_pool_import_pointer() noexcept {
    static RuntimeMemPoolImportPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolImportPointerFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolImportPointer"));
    }();
    return function;
}

}  // namespace

namespace glimmer::interceptor {

bool is_inside_runtime_call() noexcept {
    return g_is_inside_runtime_call;
}

}  // namespace glimmer::interceptor

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocFunction real_allocate = resolve_runtime_malloc();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes);
        }
        return glimmer::interceptor::intercept_runtime_malloc(device_pointer, memory_bytes,
                                                              real_allocate, resolve_runtime_free(),
                                                              resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocManaged(void** device_pointer, std::size_t memory_bytes,
                                                   unsigned int flags) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocManagedFunction real_allocate = resolve_runtime_malloc_managed();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, flags);
        }
        return glimmer::interceptor::intercept_runtime_malloc_managed(
            device_pointer, memory_bytes, flags, real_allocate, resolve_runtime_free(),
            resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocPitch(void** device_pointer, std::size_t* pitch,
                                                 std::size_t width_bytes, std::size_t height) {
    if (device_pointer == nullptr || pitch == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocPitchFunction real_allocate = resolve_runtime_malloc_pitch();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, pitch, width_bytes, height);
        }
        return glimmer::interceptor::intercept_runtime_malloc_pitch(
            device_pointer, pitch, width_bytes, height, real_allocate, resolve_runtime_free(),
            resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitched_device_pointer,
                                              struct cudaExtent extent) {
    if (pitched_device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMalloc3DFunction real_allocate = resolve_runtime_malloc_3d();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(pitched_device_pointer, extent);
        }
        return glimmer::interceptor::intercept_runtime_malloc_3d(
            pitched_device_pointer, extent, real_allocate, resolve_runtime_free(),
            resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeFunction real_release = resolve_runtime_free();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported : real_release(device_pointer);
        }
        return glimmer::interceptor::intercept_runtime_free(device_pointer, real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle,
                                                     void* device_pointer) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeIpcGetMemHandleFunction real_get_handle = resolve_runtime_ipc_get_mem_handle();
        if (is_reentrant) {
            return real_get_handle == nullptr ? cudaErrorNotSupported
                                              : real_get_handle(handle, device_pointer);
        }
        return glimmer::interceptor::intercept_runtime_ipc_get_mem_handle(handle, device_pointer,
                                                                          real_get_handle);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void** device_pointer,
                                                      cudaIpcMemHandle_t handle,
                                                      unsigned int flags) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeIpcOpenMemHandleFunction real_open_handle =
            resolve_runtime_ipc_open_mem_handle();
        if (is_reentrant) {
            return real_open_handle == nullptr ? cudaErrorNotSupported
                                               : real_open_handle(device_pointer, handle, flags);
        }
        return glimmer::interceptor::intercept_runtime_ipc_open_mem_handle(device_pointer, handle,
                                                                           flags, real_open_handle);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void* device_pointer) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeIpcCloseMemHandleFunction real_close_handle =
            resolve_runtime_ipc_close_mem_handle();
        if (is_reentrant) {
            return real_close_handle == nullptr ? cudaErrorNotSupported
                                                : real_close_handle(device_pointer);
        }
        return glimmer::interceptor::intercept_runtime_ipc_close_mem_handle(device_pointer,
                                                                            real_close_handle);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaImportExternalMemory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeImportExternalMemoryFunction real_import =
            resolve_runtime_import_external_memory();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(external_memory, handle_desc);
        }
        return glimmer::interceptor::intercept_runtime_import_external_memory(
            external_memory, handle_desc, real_import);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI
cudaExternalMemoryGetMappedBuffer(void** device_pointer, cudaExternalMemory_t external_memory,
                                  const struct cudaExternalMemoryBufferDesc* buffer_desc) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeExternalMemoryGetMappedBufferFunction real_get_buffer =
            resolve_runtime_external_memory_get_mapped_buffer();
        if (is_reentrant) {
            return real_get_buffer == nullptr
                       ? cudaErrorNotSupported
                       : real_get_buffer(device_pointer, external_memory, buffer_desc);
        }
        return glimmer::interceptor::intercept_runtime_external_memory_get_mapped_buffer(
            device_pointer, external_memory, buffer_desc, real_get_buffer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaExternalMemoryGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeExternalMemoryGetMappedMipmappedArrayFunction real_get_mipmap =
            resolve_runtime_external_memory_get_mapped_mipmapped_array();
        if (is_reentrant) {
            return real_get_mipmap == nullptr
                       ? cudaErrorNotSupported
                       : real_get_mipmap(mipmap, external_memory, mipmap_desc);
        }
        return glimmer::interceptor::intercept_runtime_external_memory_get_mapped_mipmapped_array(
            mipmap, external_memory, mipmap_desc, real_get_mipmap);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t external_memory) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeDestroyExternalMemoryFunction real_destroy =
            resolve_runtime_destroy_external_memory();
        if (is_reentrant) {
            return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(external_memory);
        }
        return glimmer::interceptor::intercept_runtime_destroy_external_memory(external_memory,
                                                                               real_destroy);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t* array,
                                                 const struct cudaChannelFormatDesc* descriptor,
                                                 std::size_t width, std::size_t height,
                                                 unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocArrayFunction real_allocate = resolve_runtime_malloc_array();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(array, descriptor, width, height, flags);
        }
        return glimmer::interceptor::intercept_runtime_malloc_array(array, descriptor, width,
                                                                    height, flags, real_allocate);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t* array,
                                                   const struct cudaChannelFormatDesc* descriptor,
                                                   struct cudaExtent extent, unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMalloc3DArrayFunction real_allocate = resolve_runtime_malloc_3d_array();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(array, descriptor, extent, flags);
        }
        return glimmer::interceptor::intercept_runtime_malloc_3d_array(array, descriptor, extent,
                                                                       flags, real_allocate);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocMipmappedArray(
    cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
    struct cudaExtent extent, unsigned int level_count, unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocMipmappedArrayFunction real_allocate =
            resolve_runtime_malloc_mipmapped_array();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(mipmap, descriptor, extent, level_count, flags);
        }
        return glimmer::interceptor::intercept_runtime_malloc_mipmapped_array(
            mipmap, descriptor, extent, level_count, flags, real_allocate);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeArrayFunction real_release = resolve_runtime_free_array();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported : real_release(array);
        }
        return glimmer::interceptor::intercept_runtime_free_array(array, real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmap) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeMipmappedArrayFunction real_release =
            resolve_runtime_free_mipmapped_array();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported : real_release(mipmap);
        }
        return glimmer::interceptor::intercept_runtime_free_mipmapped_array(mipmap, real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphicsUnregisterResourceFunction real_unregister =
            resolve_runtime_graphics_unregister_resource();
        if (is_reentrant) {
            return real_unregister == nullptr ? cudaErrorNotSupported : real_unregister(resource);
        }
        return glimmer::interceptor::intercept_runtime_graphics_unregister_resource(
            resource, real_unregister);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
                                                                 unsigned int flags) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphicsResourceSetMapFlagsFunction real_set_flags =
            resolve_runtime_graphics_resource_set_map_flags();
        if (is_reentrant) {
            return real_set_flags == nullptr ? cudaErrorNotSupported
                                             : real_set_flags(resource, flags);
        }
        return glimmer::interceptor::intercept_runtime_graphics_resource_set_map_flags(
            resource, flags, real_set_flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsMapResources(int count,
                                                          cudaGraphicsResource_t* resources,
                                                          cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphicsMapResourcesFunction real_map =
            resolve_runtime_graphics_map_resources();
        if (is_reentrant) {
            return real_map == nullptr ? cudaErrorNotSupported : real_map(count, resources, stream);
        }
        return glimmer::interceptor::intercept_runtime_graphics_map_resources(count, resources,
                                                                              stream, real_map);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count,
                                                            cudaGraphicsResource_t* resources,
                                                            cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphicsUnmapResourcesFunction real_unmap =
            resolve_runtime_graphics_unmap_resources();
        if (is_reentrant) {
            return real_unmap == nullptr ? cudaErrorNotSupported
                                         : real_unmap(count, resources, stream);
        }
        return glimmer::interceptor::intercept_runtime_graphics_unmap_resources(count, resources,
                                                                                stream, real_unmap);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource) {
    if (device_pointer == nullptr || size == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphicsResourceGetMappedPointerFunction real_get_pointer =
            resolve_runtime_graphics_resource_get_mapped_pointer();
        if (is_reentrant) {
            return real_get_pointer == nullptr ? cudaErrorNotSupported
                                               : real_get_pointer(device_pointer, size, resource);
        }
        return glimmer::interceptor::intercept_runtime_graphics_resource_get_mapped_pointer(
            device_pointer, size, resource, real_get_pointer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI
cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource,
                                      unsigned int array_index, unsigned int mip_level) {
    if (array == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphicsSubResourceGetMappedArrayFunction real_get_array =
            resolve_runtime_graphics_subresource_get_mapped_array();
        if (is_reentrant) {
            return real_get_array == nullptr
                       ? cudaErrorNotSupported
                       : real_get_array(array, resource, array_index, mip_level);
        }
        return glimmer::interceptor::intercept_runtime_graphics_subresource_get_mapped_array(
            array, resource, array_index, mip_level, real_get_array);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource) {
    if (mipmap == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphicsResourceGetMappedMipmappedArrayFunction real_get_mipmap =
            resolve_runtime_graphics_resource_get_mapped_mipmapped_array();
        if (is_reentrant) {
            return real_get_mipmap == nullptr ? cudaErrorNotSupported
                                              : real_get_mipmap(mipmap, resource);
        }
        return glimmer::interceptor::intercept_runtime_graphics_resource_get_mapped_mipmapped_array(
            mipmap, resource, real_get_mipmap);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeGraphAddMemAllocNodeFunction real_add_node =
            resolve_runtime_graph_add_mem_alloc_node();
        if (is_reentrant) {
            return real_add_node == nullptr ? cudaErrorNotSupported
                                            : real_add_node(graph_node, graph, dependencies,
                                                            dependency_count, parameters);
        }
        return glimmer::interceptor::intercept_runtime_graph_add_mem_alloc_node(
            graph_node, graph, dependencies, dependency_count, parameters, real_add_node);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMemGetInfoFunction real_query = resolve_runtime_mem_get_info();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported
                                         : real_query(free_bytes, total_bytes);
        }
        return glimmer::interceptor::intercept_runtime_mem_get_info(free_bytes, total_bytes,
                                                                    real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocAsyncFunction real_allocate = resolve_runtime_malloc_async();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_async(
            device_pointer, memory_bytes, stream, real_allocate, resolve_runtime_free_async(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** device_pointer,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocAsyncFunction real_allocate = resolve_runtime_malloc_async_ptsz();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_async(
            device_pointer, memory_bytes, stream, real_allocate, resolve_runtime_free_async_ptsz(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** device_pointer,
                                                         std::size_t memory_bytes,
                                                         cudaMemPool_t pool, cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocFromPoolAsyncFunction real_allocate =
            resolve_runtime_malloc_from_pool_async();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, memory_bytes, pool, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, real_allocate, resolve_runtime_free_async(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync_ptsz(void** device_pointer,
                                                              std::size_t memory_bytes,
                                                              cudaMemPool_t pool,
                                                              cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocFromPoolAsyncFunction real_allocate =
            resolve_runtime_malloc_from_pool_async_ptsz();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, memory_bytes, pool, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, real_allocate,
            resolve_runtime_free_async_ptsz(), resolve_runtime_device_synchronize(),
            resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceGetDefaultMemPoolFunction real_query =
            resolve_runtime_device_get_default_mem_pool();
        return real_query == nullptr ? cudaErrorNotSupported : real_query(pool, device);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceSetMemPoolFunction real_set = resolve_runtime_device_set_mem_pool();
        return real_set == nullptr ? cudaErrorNotSupported : real_set(device, pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceGetMemPoolFunction real_query = resolve_runtime_device_get_mem_pool();
        return real_query == nullptr ? cudaErrorNotSupported : real_query(pool, device);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool,
                                                   std::size_t min_bytes_to_keep) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolTrimToFunction real_trim = resolve_runtime_mem_pool_trim_to();
        return real_trim == nullptr ? cudaErrorNotSupported : real_trim(pool, min_bytes_to_keep);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value) {
    if (pool == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolSetAttributeFunction real_set =
            resolve_runtime_mem_pool_set_attribute();
        return real_set == nullptr ? cudaErrorNotSupported : real_set(pool, attribute, value);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value) {
    if (pool == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolGetAttributeFunction real_get =
            resolve_runtime_mem_pool_get_attribute();
        return real_get == nullptr ? cudaErrorNotSupported : real_get(pool, attribute, value);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t pool,
                                                      const cudaMemAccessDesc* descriptors,
                                                      std::size_t descriptor_count) {
    if (pool == nullptr || (descriptor_count != 0 && descriptors == nullptr)) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolSetAccessFunction real_set = resolve_runtime_mem_pool_set_access();
        return real_set == nullptr ? cudaErrorNotSupported
                                   : real_set(pool, descriptors, descriptor_count);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t pool,
                                                      cudaMemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolGetAccessFunction real_get = resolve_runtime_mem_pool_get_access();
        return real_get == nullptr ? cudaErrorNotSupported : real_get(flags, pool, location);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                                   const cudaMemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolCreateFunction real_create = resolve_runtime_mem_pool_create();
        return real_create == nullptr ? cudaErrorNotSupported : real_create(pool, properties);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolDestroyFunction real_destroy = resolve_runtime_mem_pool_destroy();
        return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetDefaultMemPool(cudaMemPool_t* pool,
                                                          cudaMemLocation* location,
                                                          cudaMemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemGetDefaultMemPoolFunction real_get =
            resolve_runtime_mem_get_default_mem_pool();
        return real_get == nullptr ? cudaErrorNotSupported
                                   : real_get(pool, location, allocation_type);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetMemPool(cudaMemPool_t* pool, cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemGetMemPoolFunction real_get = resolve_runtime_mem_get_mem_pool();
        return real_get == nullptr ? cudaErrorNotSupported
                                   : real_get(pool, location, allocation_type);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemSetMemPool(cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type,
                                                   cudaMemPool_t pool) {
    if (location == nullptr || pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemSetMemPoolFunction real_set = resolve_runtime_mem_set_mem_pool();
        return real_set == nullptr ? cudaErrorNotSupported
                                   : real_set(location, allocation_type, pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void* handle_out, cudaMemPool_t pool, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags) {
    if (handle_out == nullptr || pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolExportToShareableHandleFunction real_export =
            resolve_runtime_mem_pool_export_to_shareable_handle();
        return real_export == nullptr ? cudaErrorNotSupported
                                      : real_export(handle_out, pool, handle_type, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t* pool_out, void* handle, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags) {
    if (pool_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolImportFromShareableHandleFunction real_import =
            resolve_runtime_mem_pool_import_from_shareable_handle();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(pool_out, handle, handle_type, flags);
        }
        return glimmer::interceptor::intercept_runtime_mem_pool_import_from_shareable_handle(
            pool_out, handle, handle_type, flags, real_import);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportPointer(cudaMemPoolPtrExportData* share_data_out,
                                                          void* device_pointer) {
    if (share_data_out == nullptr || device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolExportPointerFunction real_export =
            resolve_runtime_mem_pool_export_pointer();
        return real_export == nullptr ? cudaErrorNotSupported
                                      : real_export(share_data_out, device_pointer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportPointer(void** pointer_out, cudaMemPool_t pool,
                                                          cudaMemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolImportPointerFunction real_import =
            resolve_runtime_mem_pool_import_pointer();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(pointer_out, pool, share_data);
        }
        return glimmer::interceptor::intercept_runtime_mem_pool_import_pointer(
            pointer_out, pool, share_data, real_import);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeAsyncFunction real_release = resolve_runtime_free_async();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported
                                           : real_release(device_pointer, stream);
        }
        return glimmer::interceptor::intercept_runtime_free_async(device_pointer, stream,
                                                                  real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* device_pointer, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeAsyncFunction real_release = resolve_runtime_free_async_ptsz();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported
                                           : real_release(device_pointer, stream);
        }
        return glimmer::interceptor::intercept_runtime_free_async(device_pointer, stream,
                                                                  real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceSynchronizeFunction real_synchronize =
            resolve_runtime_device_synchronize();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize();
        }
        return glimmer::interceptor::intercept_runtime_device_synchronize(real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamSynchronizeFunction real_synchronize =
            resolve_runtime_stream_synchronize();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_synchronize(stream, real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamSynchronizeFunction real_synchronize =
            resolve_runtime_stream_synchronize_ptsz();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_synchronize(stream, real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamQueryFunction real_query = resolve_runtime_stream_query();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_query(stream, real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamQueryFunction real_query = resolve_runtime_stream_query_ptsz();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_query(stream, real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamDestroyFunction real_destroy = resolve_runtime_stream_destroy();
        if (is_reentrant) {
            return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_destroy(stream, real_destroy);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* function, dim3 grid_dim,
                                                  dim3 block_dim, void** arguments,
                                                  std::size_t shared_memory_bytes,
                                                  cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeLaunchKernelFunction real_launch = resolve_runtime_launch_kernel();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        const cudaError_t result =
            real_launch(function, grid_dim, block_dim, arguments, shared_memory_bytes, stream);
        if (!is_reentrant && result == cudaSuccess) {
            glimmer::interceptor::report_kernel_launch_observed({
                .api_name = "cudaLaunchKernel",
                .grid_dim_x = grid_dim.x,
                .grid_dim_y = grid_dim.y,
                .grid_dim_z = grid_dim.z,
                .block_dim_x = block_dim.x,
                .block_dim_y = block_dim.y,
                .block_dim_z = block_dim.z,
                .shared_memory_bytes = shared_memory_bytes,
                .stream = reinterpret_cast<const void*>(stream),
            });
        }
        return result;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void* function, dim3 grid_dim,
                                                       dim3 block_dim, void** arguments,
                                                       std::size_t shared_memory_bytes,
                                                       cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeLaunchKernelFunction real_launch = resolve_runtime_launch_kernel_ptsz();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        const cudaError_t result =
            real_launch(function, grid_dim, block_dim, arguments, shared_memory_bytes, stream);
        if (!is_reentrant && result == cudaSuccess) {
            glimmer::interceptor::report_kernel_launch_observed({
                .api_name = "cudaLaunchKernel_ptsz",
                .grid_dim_x = grid_dim.x,
                .grid_dim_y = grid_dim.y,
                .grid_dim_z = grid_dim.z,
                .block_dim_x = block_dim.x,
                .block_dim_y = block_dim.y,
                .block_dim_z = block_dim.z,
                .shared_memory_bytes = shared_memory_bytes,
                .stream = reinterpret_cast<const void*>(stream),
            });
        }
        return result;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming): preserve CUDA compiler
// ABI names.
extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid_dim,
                                                    dim3 block_dim, void** arguments,
                                                    std::size_t shared_memory_bytes,
                                                    cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeInternalLaunchKernelFunction real_launch =
            resolve_runtime_internal_launch_kernel();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        const cudaError_t result =
            real_launch(kernel, grid_dim, block_dim, arguments, shared_memory_bytes, stream);
        if (!is_reentrant && result == cudaSuccess) {
            glimmer::interceptor::report_kernel_launch_observed({
                .api_name = "__cudaLaunchKernel",
                .grid_dim_x = grid_dim.x,
                .grid_dim_y = grid_dim.y,
                .grid_dim_z = grid_dim.z,
                .block_dim_x = block_dim.x,
                .block_dim_y = block_dim.y,
                .block_dim_z = block_dim.z,
                .shared_memory_bytes = shared_memory_bytes,
                .stream = reinterpret_cast<const void*>(stream),
            });
        }
        return result;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid_dim,
                                                         dim3 block_dim, void** arguments,
                                                         std::size_t shared_memory_bytes,
                                                         cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeInternalLaunchKernelFunction real_launch =
            resolve_runtime_internal_launch_kernel_ptsz();
        if (real_launch == nullptr) {
            return cudaErrorNotSupported;
        }
        const cudaError_t result =
            real_launch(kernel, grid_dim, block_dim, arguments, shared_memory_bytes, stream);
        if (!is_reentrant && result == cudaSuccess) {
            glimmer::interceptor::report_kernel_launch_observed({
                .api_name = "__cudaLaunchKernel_ptsz",
                .grid_dim_x = grid_dim.x,
                .grid_dim_y = grid_dim.y,
                .grid_dim_z = grid_dim.z,
                .block_dim_x = block_dim.x,
                .block_dim_y = block_dim.y,
                .block_dim_z = block_dim.z,
                .shared_memory_bytes = shared_memory_bytes,
                .stream = reinterpret_cast<const void*>(stream),
            });
        }
        return result;
    } catch (...) {
        return cudaErrorUnknown;
    }
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
