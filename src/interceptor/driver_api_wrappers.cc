#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_api_interceptor.h"

#include <cuda.h>

#ifdef cuEventDestroy
#undef cuEventDestroy
#endif

#ifdef cuEventRecord
#undef cuEventRecord
#endif

#ifdef cuEventRecordWithFlags
#undef cuEventRecordWithFlags
#endif

#ifdef cuStreamWaitEvent
#undef cuStreamWaitEvent
#endif

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuLaunchKernel
#undef cuLaunchKernel
#endif
#ifdef cuGraphLaunch
#undef cuGraphLaunch
#endif
#ifdef cuLaunchKernelEx
#undef cuLaunchKernelEx
#endif
#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
#ifdef cuMemAlloc
#undef cuMemAlloc
#endif
#ifdef cuMemAllocPitch
#undef cuMemAllocPitch
#endif
#ifdef cuMemFree
#undef cuMemFree
#endif
#ifdef cuMemGetInfo
#undef cuMemGetInfo
#endif
#ifdef cuIpcGetMemHandle
#undef cuIpcGetMemHandle
#endif
#ifdef cuIpcOpenMemHandle
#undef cuIpcOpenMemHandle
#endif
#ifdef cuIpcOpenMemHandle_v2
#undef cuIpcOpenMemHandle_v2
#endif
#ifdef cuIpcCloseMemHandle
#undef cuIpcCloseMemHandle
#endif
#ifdef cuArrayCreate
#undef cuArrayCreate
#endif
#ifdef cuArrayCreate_v2
#undef cuArrayCreate_v2
#endif
#ifdef cuArray3DCreate
#undef cuArray3DCreate
#endif
#ifdef cuArray3DCreate_v2
#undef cuArray3DCreate_v2
#endif
#ifdef cuGraphicsResourceGetMappedPointer
#undef cuGraphicsResourceGetMappedPointer
#endif
#ifdef cuGraphicsResourceGetMappedPointer_v2
#undef cuGraphicsResourceGetMappedPointer_v2
#endif
#ifdef cuGraphicsResourceSetMapFlags
#undef cuGraphicsResourceSetMapFlags
#endif
#ifdef cuGraphicsResourceSetMapFlags_v2
#undef cuGraphicsResourceSetMapFlags_v2
#endif
#ifdef cuCtxDestroy
#undef cuCtxDestroy
#endif
#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
#ifdef cuCtxSynchronize
#undef cuCtxSynchronize
#endif
#ifdef cuMemAllocAsync
#undef cuMemAllocAsync
#endif
#ifdef cuMemAllocFromPoolAsync
#undef cuMemAllocFromPoolAsync
#endif
#ifdef cuMemCreate
#undef cuMemCreate
#endif
#ifdef cuMemRelease
#undef cuMemRelease
#endif
#ifdef cuMemAddressReserve
#undef cuMemAddressReserve
#endif
#ifdef cuMemAddressFree
#undef cuMemAddressFree
#endif
#ifdef cuMemMap
#undef cuMemMap
#endif
#ifdef cuMemUnmap
#undef cuMemUnmap
#endif
#ifdef cuMemSetAccess
#undef cuMemSetAccess
#endif
#ifdef cuMemGetAddressRange
#undef cuMemGetAddressRange
#endif
#ifdef cuMemFreeAsync
#undef cuMemFreeAsync
#endif
#ifdef cuStreamGetDevice
#undef cuStreamGetDevice
#endif
#ifdef cuStreamGetCtx
#undef cuStreamGetCtx
#endif
#ifdef cuStreamGetCtx_v2
#undef cuStreamGetCtx_v2
#endif
#ifdef cuStreamQuery
#undef cuStreamQuery
#endif
#ifdef cuStreamSynchronize
#undef cuStreamSynchronize
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif

namespace {

template <typename Function>
CUresult guard_modeled_cuda_boundary(Function&& function) noexcept {
    try {
        const glimmer::interceptor::CooperativeCallScope scope;
        return function();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

template <typename Function>
CUresult guard_cuda_boundary(Function&& function) noexcept {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_cooperative_native_boundary(function); });
}

template <typename Function>
CUresult guard_memory_cuda_boundary(Function&& function) noexcept {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_cooperative_memory_boundary(function); });
}

}  // namespace

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return guard_memory_cuda_boundary([device_pointer, memory_bytes] {
        return glimmer::interceptor::intercept_mem_alloc(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return cuMemAlloc_v2(device_pointer, memory_bytes);
}

extern "C" CUresult CUDAAPI cuInit(unsigned int flags) {
    return guard_modeled_cuda_boundary(
        [flags] { return glimmer::interceptor::intercept_init(flags); });
}

extern "C" CUresult CUDAAPI cuModuleLoadData(CUmodule* module, const void* image) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_module_load_data(module, image, false, 0, nullptr,
                                                                nullptr);
    });
}

extern "C" CUresult CUDAAPI cuModuleLoadDataEx(CUmodule* module, const void* image,
                                               unsigned int option_count, CUjit_option* options,
                                               void** option_values) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_module_load_data(module, image, true, option_count,
                                                                options, option_values);
    });
}

extern "C" CUresult CUDAAPI cuModuleGetFunction(CUfunction* function, CUmodule module,
                                                const char* name) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_module_get_function(function, module, name);
    });
}

extern "C" CUresult CUDAAPI cuModuleUnload(CUmodule module) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_module_unload(module); });
}

extern "C" CUresult CUDAAPI cuLibraryLoadData(CUlibrary* library, const void* image,
                                              CUjit_option* options, void** option_values,
                                              unsigned int option_count,
                                              CUlibraryOption* library_options,
                                              void** library_option_values,
                                              unsigned int library_option_count) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_library_load_data(
            library, image, options, option_values, option_count, library_options,
            library_option_values, library_option_count);
    });
}

extern "C" CUresult CUDAAPI cuLibraryLoadFromFile(CUlibrary* library, const char* path,
                                                  CUjit_option* options, void** option_values,
                                                  unsigned int option_count,
                                                  CUlibraryOption* library_options,
                                                  void** library_option_values,
                                                  unsigned int library_option_count) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_library_load_from_file(
            library, path, options, option_values, option_count, library_options,
            library_option_values, library_option_count);
    });
}

extern "C" CUresult CUDAAPI cuLibraryGetModule(CUmodule* module, CUlibrary library) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_library_get_module(module, library); });
}

extern "C" CUresult CUDAAPI cuLibraryUnload(CUlibrary library) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_library_unload(library); });
}

extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned int grid_dim_x,
                                           unsigned int grid_dim_y, unsigned int grid_dim_z,
                                           unsigned int block_dim_x, unsigned int block_dim_y,
                                           unsigned int block_dim_z,
                                           unsigned int shared_memory_bytes, CUstream stream,
                                           void** kernel_parameters, void** extra) {
    return guard_modeled_cuda_boundary([function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                                        block_dim_y, block_dim_z, shared_memory_bytes, stream,
                                        kernel_parameters, extra] {
        return glimmer::interceptor::intercept_launch_kernel(
            function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
            shared_memory_bytes, stream, kernel_parameters, extra, false);
    });
}

extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction function, unsigned int grid_dim_x,
                                                unsigned int grid_dim_y, unsigned int grid_dim_z,
                                                unsigned int block_dim_x, unsigned int block_dim_y,
                                                unsigned int block_dim_z,
                                                unsigned int shared_memory_bytes, CUstream stream,
                                                void** kernel_parameters, void** extra) {
    return guard_modeled_cuda_boundary([function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                                        block_dim_y, block_dim_z, shared_memory_bytes, stream,
                                        kernel_parameters, extra] {
        return glimmer::interceptor::intercept_launch_kernel(
            function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y, block_dim_z,
            shared_memory_bytes, stream, kernel_parameters, extra, true);
    });
}

extern "C" CUresult CUDAAPI cuGraphLaunch(CUgraphExec graph_exec, CUstream stream) {
    return guard_modeled_cuda_boundary([graph_exec, stream] {
        return glimmer::interceptor::intercept_graph_launch(graph_exec, stream, false);
    });
}

extern "C" CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig* config, CUfunction function,
                                             void** kernel_parameters, void** extra) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_launch_kernel_ex(config, function, kernel_parameters,
                                                                extra, false);
    });
}

extern "C" CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig* config, CUfunction function,
                                                  void** kernel_parameters, void** extra) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_launch_kernel_ex(config, function, kernel_parameters,
                                                                extra, true);
    });
}

extern "C" CUresult CUDAAPI cuGraphLaunch_ptsz(CUgraphExec graph_exec, CUstream stream) {
    return guard_modeled_cuda_boundary([graph_exec, stream] {
        return glimmer::interceptor::intercept_graph_launch(graph_exec, stream, true);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              unsigned int flags) {
    return guard_memory_cuda_boundary([device_pointer, memory_bytes, flags] {
        return glimmer::interceptor::intercept_mem_alloc_managed(device_pointer, memory_bytes,
                                                                 flags);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr* device_pointer, std::size_t* pitch,
                                               std::size_t width_bytes, std::size_t height,
                                               unsigned int element_size_bytes) {
    return guard_memory_cuda_boundary(
        [device_pointer, pitch, width_bytes, height, element_size_bytes] {
            return glimmer::interceptor::intercept_mem_alloc_pitch(
                device_pointer, pitch, width_bytes, height, element_size_bytes);
        });
}

extern "C" CUresult CUDAAPI cuMemAllocPitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                            std::size_t width_bytes, std::size_t height,
                                            unsigned int element_size_bytes) {
    return cuMemAllocPitch_v2(device_pointer, pitch, width_bytes, height, element_size_bytes);
}

extern "C" CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                            CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, stream] {
        return glimmer::interceptor::intercept_mem_alloc_async(device_pointer, memory_bytes, stream,
                                                               false);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocAsync_ptsz(CUdeviceptr* device_pointer,
                                                 std::size_t memory_bytes, CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, stream] {
        return glimmer::interceptor::intercept_mem_alloc_async(device_pointer, memory_bytes, stream,
                                                               true);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* device_pointer,
                                                    std::size_t memory_bytes, CUmemoryPool pool,
                                                    CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, pool, stream] {
        return glimmer::interceptor::intercept_mem_alloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, false);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* device_pointer,
                                                         std::size_t memory_bytes,
                                                         CUmemoryPool pool, CUstream stream) {
    return guard_cuda_boundary([device_pointer, memory_bytes, pool, stream] {
        return glimmer::interceptor::intercept_mem_alloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, true);
    });
}

extern "C" CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle,
                                        std::size_t memory_bytes, const CUmemAllocationProp* prop,
                                        unsigned long long flags) {
    return guard_cuda_boundary([handle, memory_bytes, prop, flags] {
        return glimmer::interceptor::intercept_mem_create(handle, memory_bytes, prop, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle) {
    return guard_cuda_boundary(
        [handle] { return glimmer::interceptor::intercept_mem_release(handle); });
}

extern "C" CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, std::size_t alignment,
                                                CUdeviceptr requested_address,
                                                unsigned long long flags) {
    return guard_cuda_boundary([device_pointer, memory_bytes, alignment, requested_address, flags] {
        return glimmer::interceptor::intercept_mem_address_reserve(
            device_pointer, memory_bytes, alignment, requested_address, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemAddressFree(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    return guard_cuda_boundary([device_pointer, memory_bytes] {
        return glimmer::interceptor::intercept_mem_address_free(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemMap(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                     std::size_t offset, CUmemGenericAllocationHandle handle,
                                     unsigned long long flags) {
    return guard_cuda_boundary([device_pointer, memory_bytes, offset, handle, flags] {
        return glimmer::interceptor::intercept_mem_map(device_pointer, memory_bytes, offset, handle,
                                                       flags);
    });
}

extern "C" CUresult CUDAAPI cuMemMapArrayAsync(CUarrayMapInfo* map_info_list, unsigned int count,
                                               CUstream stream) {
    return guard_cuda_boundary([map_info_list, count, stream] {
        return glimmer::interceptor::intercept_mem_map_array_async(map_info_list, count, stream);
    });
}

extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    return guard_cuda_boundary([device_pointer, memory_bytes] {
        return glimmer::interceptor::intercept_mem_unmap(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemSetAccess(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                           const CUmemAccessDesc* access_descriptors,
                                           std::size_t descriptor_count) {
    return guard_cuda_boundary(
        [device_pointer, memory_bytes, access_descriptors, descriptor_count] {
            return glimmer::interceptor::intercept_mem_set_access(
                device_pointer, memory_bytes, access_descriptors, descriptor_count);
        });
}

extern "C" CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* base_pointer,
                                                    std::size_t* memory_bytes,
                                                    CUdeviceptr device_pointer) {
    return guard_cuda_boundary([base_pointer, memory_bytes, device_pointer] {
        return glimmer::interceptor::intercept_mem_get_address_range(base_pointer, memory_bytes,
                                                                     device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* base_pointer,
                                                 std::size_t* memory_bytes,
                                                 CUdeviceptr device_pointer) {
    return cuMemGetAddressRange_v2(base_pointer, memory_bytes, device_pointer);
}

extern "C" CUresult CUDAAPI cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location,
                                           CUdeviceptr device_pointer) {
    return guard_cuda_boundary([flags, location, device_pointer] {
        return glimmer::interceptor::intercept_mem_get_access(flags, location, device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemExportToShareableHandle(void* shareable_handle,
                                                         CUmemGenericAllocationHandle handle,
                                                         CUmemAllocationHandleType handle_type,
                                                         unsigned long long flags) {
    return guard_cuda_boundary([shareable_handle, handle, handle_type, flags] {
        return glimmer::interceptor::intercept_mem_export_to_shareable_handle(
            shareable_handle, handle, handle_type, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                                                           void* os_handle,
                                                           CUmemAllocationHandleType handle_type) {
    return guard_cuda_boundary([handle, os_handle, handle_type] {
        return glimmer::interceptor::intercept_mem_import_from_shareable_handle(handle, os_handle,
                                                                                handle_type);
    });
}

extern "C" CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle* handle, CUdeviceptr device_pointer) {
    return guard_cuda_boundary([handle, device_pointer] {
        return glimmer::interceptor::intercept_ipc_get_mem_handle(handle, device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuIpcOpenMemHandle_v2(CUdeviceptr* device_pointer,
                                                  CUipcMemHandle handle, unsigned int flags) {
    return guard_cuda_boundary([device_pointer, handle, flags] {
        return glimmer::interceptor::intercept_ipc_open_mem_handle(device_pointer, handle, flags);
    });
}

extern "C" CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                               unsigned int flags) {
    return cuIpcOpenMemHandle_v2(device_pointer, handle, flags);
}

extern "C" CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr device_pointer) {
    return guard_cuda_boundary([device_pointer] {
        return glimmer::interceptor::intercept_ipc_close_mem_handle(device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuImportExternalMemory(
    CUexternalMemory* external_memory, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc) {
    return guard_cuda_boundary([external_memory, handle_desc] {
        return glimmer::interceptor::intercept_import_external_memory(external_memory, handle_desc);
    });
}

extern "C" CUresult CUDAAPI
cuExternalMemoryGetMappedBuffer(CUdeviceptr* device_pointer, CUexternalMemory external_memory,
                                const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc) {
    return guard_cuda_boundary([device_pointer, external_memory, buffer_desc] {
        return glimmer::interceptor::intercept_external_memory_get_mapped_buffer(
            device_pointer, external_memory, buffer_desc);
    });
}

extern "C" CUresult CUDAAPI cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray* mipmap, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc) {
    return guard_cuda_boundary([mipmap, external_memory, mipmap_desc] {
        return glimmer::interceptor::intercept_external_memory_get_mapped_mipmapped_array(
            mipmap, external_memory, mipmap_desc);
    });
}

extern "C" CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory external_memory) {
    return guard_cuda_boundary([external_memory] {
        return glimmer::interceptor::intercept_destroy_external_memory(external_memory);
    });
}

extern "C" CUresult CUDAAPI cuArrayCreate_v2(CUarray* array,
                                             const CUDA_ARRAY_DESCRIPTOR* descriptor) {
    return guard_cuda_boundary([array, descriptor] {
        return glimmer::interceptor::intercept_array_create(array, descriptor);
    });
}

extern "C" CUresult CUDAAPI cuArrayCreate(CUarray* array, const CUDA_ARRAY_DESCRIPTOR* descriptor) {
    return cuArrayCreate_v2(array, descriptor);
}

extern "C" CUresult CUDAAPI cuArray3DCreate_v2(CUarray* array,
                                               const CUDA_ARRAY3D_DESCRIPTOR* descriptor) {
    return guard_cuda_boundary([array, descriptor] {
        return glimmer::interceptor::intercept_array_3d_create(array, descriptor);
    });
}

extern "C" CUresult CUDAAPI cuArray3DCreate(CUarray* array,
                                            const CUDA_ARRAY3D_DESCRIPTOR* descriptor) {
    return cuArray3DCreate_v2(array, descriptor);
}

extern "C" CUresult CUDAAPI cuArrayDestroy(CUarray array) {
    return guard_cuda_boundary(
        [array] { return glimmer::interceptor::intercept_array_destroy(array); });
}

extern "C" CUresult CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray* mipmap,
                                                   const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                                   unsigned int level_count) {
    return guard_cuda_boundary([mipmap, descriptor, level_count] {
        return glimmer::interceptor::intercept_mipmapped_array_create(mipmap, descriptor,
                                                                      level_count);
    });
}

extern "C" CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray mipmap) {
    return guard_cuda_boundary(
        [mipmap] { return glimmer::interceptor::intercept_mipmapped_array_destroy(mipmap); });
}

extern "C" CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource) {
    return guard_cuda_boundary([resource] {
        return glimmer::interceptor::intercept_graphics_unregister_resource(resource);
    });
}

extern "C" CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(CUarray* array,
                                                                CUgraphicsResource resource,
                                                                unsigned int array_index,
                                                                unsigned int mip_level) {
    return guard_cuda_boundary([array, resource, array_index, mip_level] {
        return glimmer::interceptor::intercept_graphics_subresource_get_mapped_array(
            array, resource, array_index, mip_level);
    });
}

extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* mipmap,
                                                                      CUgraphicsResource resource) {
    return guard_cuda_boundary([mipmap, resource] {
        return glimmer::interceptor::intercept_graphics_resource_get_mapped_mipmapped_array(
            mipmap, resource);
    });
}

extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* device_pointer,
                                                                  std::size_t* size,
                                                                  CUgraphicsResource resource) {
    return guard_cuda_boundary([device_pointer, size, resource] {
        return glimmer::interceptor::intercept_graphics_resource_get_mapped_pointer(device_pointer,
                                                                                    size, resource);
    });
}

extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr* device_pointer,
                                                               std::size_t* size,
                                                               CUgraphicsResource resource) {
    return cuGraphicsResourceGetMappedPointer_v2(device_pointer, size, resource);
}

extern "C" CUresult CUDAAPI cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                                             unsigned int flags) {
    return guard_cuda_boundary([resource, flags] {
        return glimmer::interceptor::intercept_graphics_resource_set_map_flags(resource, flags);
    });
}

extern "C" CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                                          unsigned int flags) {
    return cuGraphicsResourceSetMapFlags_v2(resource, flags);
}

extern "C" CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                                   CUgraphicsResource* resources, CUstream stream) {
    return guard_cuda_boundary([count, resources, stream] {
        return glimmer::interceptor::intercept_graphics_map_resources(count, resources, stream);
    });
}

extern "C" CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                                     CUgraphicsResource* resources,
                                                     CUstream stream) {
    return guard_cuda_boundary([count, resources, stream] {
        return glimmer::interceptor::intercept_graphics_unmap_resources(count, resources, stream);
    });
}

extern "C" CUresult CUDAAPI cuMemGetAllocationGranularity(std::size_t* granularity,
                                                          const CUmemAllocationProp* prop,
                                                          CUmemAllocationGranularity_flags option) {
    return guard_cuda_boundary([granularity, prop, option] {
        return glimmer::interceptor::intercept_mem_get_allocation_granularity(granularity, prop,
                                                                              option);
    });
}

extern "C" CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle) {
    return guard_cuda_boundary([prop, handle] {
        return glimmer::interceptor::intercept_mem_get_allocation_properties(prop, handle);
    });
}

extern "C" CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle,
                                                        void* device_pointer) {
    return guard_cuda_boundary([handle, device_pointer] {
        return glimmer::interceptor::intercept_mem_retain_allocation_handle(handle, device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, std::size_t min_bytes_to_keep) {
    return guard_cuda_boundary([pool, min_bytes_to_keep] {
        return glimmer::interceptor::intercept_mem_pool_trim_to(pool, min_bytes_to_keep);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) {
    return guard_cuda_boundary([pool, attribute, value] {
        return glimmer::interceptor::intercept_mem_pool_set_attribute(pool, attribute, value);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value) {
    return guard_cuda_boundary([pool, attribute, value] {
        return glimmer::interceptor::intercept_mem_pool_get_attribute(pool, attribute, value);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool,
                                               const CUmemAccessDesc* access_descriptors,
                                               std::size_t descriptor_count) {
    return guard_cuda_boundary([pool, access_descriptors, descriptor_count] {
        return glimmer::interceptor::intercept_mem_pool_set_access(pool, access_descriptors,
                                                                   descriptor_count);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool pool,
                                               CUmemLocation* location) {
    return guard_cuda_boundary([flags, pool, location] {
        return glimmer::interceptor::intercept_mem_pool_get_access(flags, pool, location);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* properties) {
    return guard_cuda_boundary([pool, properties] {
        return glimmer::interceptor::intercept_mem_pool_create(pool, properties);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool) {
    return guard_cuda_boundary(
        [pool] { return glimmer::interceptor::intercept_mem_pool_destroy(pool); });
}

extern "C" CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice device) {
    return guard_cuda_boundary([pool, device] {
        return glimmer::interceptor::intercept_device_get_mem_pool(pool, device);
    });
}

extern "C" CUresult CUDAAPI cuDeviceSetMemPool(CUdevice device, CUmemoryPool pool) {
    return guard_cuda_boundary([device, pool] {
        return glimmer::interceptor::intercept_device_set_mem_pool(device, pool);
    });
}

extern "C" CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* pool, CUdevice device) {
    return guard_cuda_boundary([pool, device] {
        return glimmer::interceptor::intercept_device_get_default_mem_pool(pool, device);
    });
}

extern "C" CUresult CUDAAPI cuMemGetDefaultMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                                   CUmemAllocationType allocation_type) {
    return guard_cuda_boundary([pool, location, allocation_type] {
        return glimmer::interceptor::intercept_mem_get_default_mem_pool(pool, location,
                                                                        allocation_type);
    });
}

extern "C" CUresult CUDAAPI cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType allocation_type) {
    return guard_cuda_boundary([pool, location, allocation_type] {
        return glimmer::interceptor::intercept_mem_get_mem_pool(pool, location, allocation_type);
    });
}

extern "C" CUresult CUDAAPI cuMemSetMemPool(CUmemLocation* location,
                                            CUmemAllocationType allocation_type,
                                            CUmemoryPool pool) {
    return guard_cuda_boundary([location, allocation_type, pool] {
        return glimmer::interceptor::intercept_mem_set_mem_pool(location, allocation_type, pool);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool,
                                                             CUmemAllocationHandleType handle_type,
                                                             unsigned long long flags) {
    return guard_cuda_boundary([handle_out, pool, handle_type, flags] {
        return glimmer::interceptor::intercept_mem_pool_export_to_shareable_handle(
            handle_out, pool, handle_type, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolImportFromShareableHandle(
    CUmemoryPool* pool_out, void* handle, CUmemAllocationHandleType handle_type,
    unsigned long long flags) {
    return guard_cuda_boundary([pool_out, handle, handle_type, flags] {
        return glimmer::interceptor::intercept_mem_pool_import_from_shareable_handle(
            pool_out, handle, handle_type, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolExportPointer(CUmemPoolPtrExportData* share_data_out,
                                                   CUdeviceptr device_pointer) {
    return guard_cuda_boundary([share_data_out, device_pointer] {
        return glimmer::interceptor::intercept_mem_pool_export_pointer(share_data_out,
                                                                       device_pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemPoolImportPointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                                   CUmemPoolPtrExportData* share_data) {
    return guard_cuda_boundary([pointer_out, pool, share_data] {
        return glimmer::interceptor::intercept_mem_pool_import_pointer(pointer_out, pool,
                                                                       share_data);
    });
}

extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr device_pointer) {
    return guard_memory_cuda_boundary(
        [device_pointer] { return glimmer::interceptor::intercept_mem_free(device_pointer); });
}

extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr device_pointer) {
    return cuMemFree_v2(device_pointer);
}

extern "C" CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr device_pointer, CUstream stream) {
    return guard_modeled_cuda_boundary([device_pointer, stream] {
        return glimmer::interceptor::intercept_mem_free_async(device_pointer, stream, false);
    });
}

extern "C" CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr device_pointer, CUstream stream) {
    return guard_modeled_cuda_boundary([device_pointer, stream] {
        return glimmer::interceptor::intercept_mem_free_async(device_pointer, stream, true);
    });
}

extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
    return guard_modeled_cuda_boundary([free_bytes, total_bytes] {
        return glimmer::interceptor::intercept_mem_get_info(free_bytes, total_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    return cuMemGetInfo_v2(free_bytes, total_bytes);
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem_v2(std::size_t* total_bytes, CUdevice device) {
    return guard_modeled_cuda_boundary([total_bytes, device] {
        return glimmer::interceptor::intercept_device_total_mem(total_bytes, device);
    });
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem(std::size_t* total_bytes, CUdevice device) {
    return cuDeviceTotalMem_v2(total_bytes, device);
}

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
    return guard_modeled_cuda_boundary(
        [context] { return glimmer::interceptor::intercept_context_get_current(context); });
}

extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    return guard_modeled_cuda_boundary(
        [device] { return glimmer::interceptor::intercept_context_get_device(device); });
}

extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context) {
    return guard_modeled_cuda_boundary(
        [context] { return glimmer::interceptor::intercept_context_destroy(context); });
}

extern "C" CUresult CUDAAPI cuCtxDestroy(CUcontext context) {
    return guard_modeled_cuda_boundary(
        [context] { return glimmer::interceptor::intercept_context_destroy(context, false); });
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice device) {
    return guard_modeled_cuda_boundary(
        [device] { return glimmer::interceptor::intercept_primary_context_reset(device, false); });
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice device) {
    return guard_modeled_cuda_boundary(
        [device] { return glimmer::interceptor::intercept_primary_context_reset(device, true); });
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice device) {
    return guard_modeled_cuda_boundary([device] {
        return glimmer::interceptor::intercept_primary_context_release(device, false);
    });
}

extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice device) {
    return guard_modeled_cuda_boundary(
        [device] { return glimmer::interceptor::intercept_primary_context_release(device, true); });
}

extern "C" CUresult CUDAAPI cuStreamGetDevice(CUstream stream, CUdevice* device) {
    return guard_modeled_cuda_boundary([stream, device] {
        return glimmer::interceptor::intercept_stream_get_device(stream, device, false);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetDevice_ptsz(CUstream stream, CUdevice* device) {
    return guard_modeled_cuda_boundary([stream, device] {
        return glimmer::interceptor::intercept_stream_get_device(stream, device, true);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetCtx(CUstream stream, CUcontext* context) {
    return guard_modeled_cuda_boundary([stream, context] {
        return glimmer::interceptor::intercept_stream_get_context(stream, context, false);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream stream, CUcontext* context) {
    return guard_modeled_cuda_boundary([stream, context] {
        return glimmer::interceptor::intercept_stream_get_context(stream, context, true);
    });
}

extern "C" CUresult CUDAAPI cuEventCreate(CUevent* event, unsigned int flags) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_event_create(event, flags); });
}

extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_event_record(event, stream); });
}

extern "C" CUresult CUDAAPI cuEventQuery(CUevent event) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_event_query(event); });
}

extern "C" CUresult CUDAAPI cuEventDestroy_v2(CUevent event) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_event_destroy(event); });
}

extern "C" CUresult CUDAAPI cuEventRecord_ptsz(CUevent event, CUstream stream) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_event_record_ptsz(event, stream); });
}

extern "C" CUresult CUDAAPI cuEventRecordWithFlags(CUevent event, CUstream stream,
                                                   unsigned int flags) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_event_record_with_flags(event, stream, flags);
    });
}

extern "C" CUresult CUDAAPI cuEventRecordWithFlags_ptsz(CUevent event, CUstream stream,
                                                        unsigned int flags) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_event_record_with_flags_ptsz(event, stream, flags);
    });
}

extern "C" CUresult CUDAAPI cuEventSynchronize(CUevent event) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_event_synchronize(event); });
}

extern "C" CUresult CUDAAPI cuEventElapsedTime(float* milliseconds, CUevent start, CUevent end) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_event_elapsed_time(milliseconds, start, end);
    });
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_stream_wait_event(stream, event, flags); });
}

extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream stream, CUevent event,
                                                   unsigned int flags) {
    return guard_modeled_cuda_boundary([&] {
        return glimmer::interceptor::intercept_stream_wait_event_ptsz(stream, event, flags);
    });
}

extern "C" CUresult CUDAAPI cuEventDestroy(CUevent event) {
    return guard_modeled_cuda_boundary(
        [&] { return glimmer::interceptor::intercept_event_destroy(event); });
}

extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream) {
    return guard_modeled_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_query(stream, false); });
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream stream) {
    return guard_modeled_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_query(stream, true); });
}

extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    return guard_modeled_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_synchronize(stream, false); });
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream stream) {
    return guard_modeled_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_synchronize(stream, true); });
}

extern "C" CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream) {
    return guard_modeled_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_destroy(stream); });
}

extern "C" CUresult CUDAAPI cuStreamDestroy(CUstream stream) {
    return cuStreamDestroy_v2(stream);
}

extern "C" CUresult CUDAAPI cuCtxSynchronize() {
    return guard_modeled_cuda_boundary(
        [] { return glimmer::interceptor::intercept_context_synchronize(); });
}

extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags) {
    return guard_modeled_cuda_boundary([symbol, function_pointer, cuda_version, flags] {
        return glimmer::interceptor::intercept_get_proc_address(symbol, function_pointer,
                                                                cuda_version, flags);
    });
}

extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** function_pointer,
                                                int cuda_version, cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status) {
    return guard_modeled_cuda_boundary(
        [symbol, function_pointer, cuda_version, flags, symbol_status] {
            return glimmer::interceptor::intercept_get_proc_address_v2(
                symbol, function_pointer, cuda_version, flags, symbol_status);
        });
}
