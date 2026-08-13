#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>

namespace {

using InitFunction = CUresult (*)(unsigned int flags);
using IpcGetMemHandleFunction = CUresult (*)(CUipcMemHandle* handle, CUdeviceptr device_pointer);
using IpcOpenMemHandleFunction = CUresult (*)(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                              unsigned int flags);
using IpcCloseMemHandleFunction = CUresult (*)(CUdeviceptr device_pointer);
using ImportExternalMemoryFunction = CUresult (*)(
    CUexternalMemory* external_memory, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc);
using ExternalMemoryGetMappedBufferFunction =
    CUresult (*)(CUdeviceptr* device_pointer, CUexternalMemory external_memory,
                 const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc);
using ExternalMemoryGetMappedMipmappedArrayFunction =
    CUresult (*)(CUmipmappedArray* mipmap, CUexternalMemory external_memory,
                 const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc);
using DestroyExternalMemoryFunction = CUresult (*)(CUexternalMemory external_memory);
using ArrayCreateFunction = CUresult (*)(CUarray* array, const CUDA_ARRAY_DESCRIPTOR* descriptor);
using Array3DCreateFunction = CUresult (*)(CUarray* array,
                                           const CUDA_ARRAY3D_DESCRIPTOR* descriptor);
using ArrayDestroyFunction = CUresult (*)(CUarray array);
using MipmappedArrayCreateFunction = CUresult (*)(CUmipmappedArray* mipmap,
                                                  const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                                  unsigned int level_count);
using MipmappedArrayDestroyFunction = CUresult (*)(CUmipmappedArray mipmap);
using GraphicsUnregisterResourceFunction = CUresult (*)(CUgraphicsResource resource);
using GraphicsSubResourceGetMappedArrayFunction = CUresult (*)(CUarray* array,
                                                               CUgraphicsResource resource,
                                                               unsigned int array_index,
                                                               unsigned int mip_level);
using GraphicsResourceGetMappedMipmappedArrayFunction = CUresult (*)(CUmipmappedArray* mipmap,
                                                                     CUgraphicsResource resource);
using GraphicsResourceGetMappedPointerFunction = CUresult (*)(CUdeviceptr* device_pointer,
                                                              std::size_t* size,
                                                              CUgraphicsResource resource);
using GraphicsResourceSetMapFlagsFunction = CUresult (*)(CUgraphicsResource resource,
                                                         unsigned int flags);
using GraphicsMapResourcesFunction = CUresult (*)(unsigned int count, CUgraphicsResource* resources,
                                                  CUstream stream);
using GraphicsUnmapResourcesFunction = CUresult (*)(unsigned int count,
                                                    CUgraphicsResource* resources, CUstream stream);
using MapArrayAsyncFunction = CUresult (*)(CUarrayMapInfo* map_info_list, unsigned int count,
                                           CUstream stream);
using ImportHandleFunction = CUresult (*)(CUmemGenericAllocationHandle* handle, void* os_handle,
                                          CUmemAllocationHandleType handle_type);
using ReleaseFunction = CUresult (*)(CUmemGenericAllocationHandle handle);
using PoolImportHandleFunction = CUresult (*)(CUmemoryPool* pool_out, void* handle,
                                              CUmemAllocationHandleType handle_type,
                                              unsigned long long flags);
using PoolImportPointerFunction = CUresult (*)(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                               CUmemPoolPtrExportData* share_data);

template <typename Function>
Function resolve(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

}  // namespace

int main() {
    unsetenv("GLIMMER_MEMORY_LIMIT_BYTES");
    unsetenv("GLIMMER_QUOTA_MODE");
    unsetenv("GLIMMER_QUOTA_TENANT_ID");

    const InitFunction init = resolve<InitFunction>("cuInit");
    const ImportHandleFunction import_handle =
        resolve<ImportHandleFunction>("cuMemImportFromShareableHandle");
    const IpcGetMemHandleFunction ipc_get_handle =
        resolve<IpcGetMemHandleFunction>("cuIpcGetMemHandle");
    const IpcOpenMemHandleFunction ipc_open_handle =
        resolve<IpcOpenMemHandleFunction>("cuIpcOpenMemHandle");
    const IpcCloseMemHandleFunction ipc_close_handle =
        resolve<IpcCloseMemHandleFunction>("cuIpcCloseMemHandle");
    const ImportExternalMemoryFunction import_external_memory =
        resolve<ImportExternalMemoryFunction>("cuImportExternalMemory");
    const ExternalMemoryGetMappedBufferFunction get_external_buffer =
        resolve<ExternalMemoryGetMappedBufferFunction>("cuExternalMemoryGetMappedBuffer");
    const ExternalMemoryGetMappedMipmappedArrayFunction get_external_mipmap =
        resolve<ExternalMemoryGetMappedMipmappedArrayFunction>(
            "cuExternalMemoryGetMappedMipmappedArray");
    const DestroyExternalMemoryFunction destroy_external_memory =
        resolve<DestroyExternalMemoryFunction>("cuDestroyExternalMemory");
    const ArrayCreateFunction array_create = resolve<ArrayCreateFunction>("cuArrayCreate");
    const Array3DCreateFunction array_3d_create = resolve<Array3DCreateFunction>("cuArray3DCreate");
    const ArrayDestroyFunction array_destroy = resolve<ArrayDestroyFunction>("cuArrayDestroy");
    const MipmappedArrayCreateFunction mipmapped_array_create =
        resolve<MipmappedArrayCreateFunction>("cuMipmappedArrayCreate");
    const MipmappedArrayDestroyFunction mipmapped_array_destroy =
        resolve<MipmappedArrayDestroyFunction>("cuMipmappedArrayDestroy");
    const GraphicsUnregisterResourceFunction graphics_unregister_resource =
        resolve<GraphicsUnregisterResourceFunction>("cuGraphicsUnregisterResource");
    const GraphicsSubResourceGetMappedArrayFunction graphics_get_mapped_array =
        resolve<GraphicsSubResourceGetMappedArrayFunction>("cuGraphicsSubResourceGetMappedArray");
    const GraphicsResourceGetMappedMipmappedArrayFunction graphics_get_mapped_mipmap =
        resolve<GraphicsResourceGetMappedMipmappedArrayFunction>(
            "cuGraphicsResourceGetMappedMipmappedArray");
    const GraphicsResourceGetMappedPointerFunction graphics_get_mapped_pointer =
        resolve<GraphicsResourceGetMappedPointerFunction>("cuGraphicsResourceGetMappedPointer_v2");
    const GraphicsResourceSetMapFlagsFunction graphics_set_map_flags =
        resolve<GraphicsResourceSetMapFlagsFunction>("cuGraphicsResourceSetMapFlags_v2");
    const GraphicsMapResourcesFunction graphics_map_resources =
        resolve<GraphicsMapResourcesFunction>("cuGraphicsMapResources");
    const GraphicsUnmapResourcesFunction graphics_unmap_resources =
        resolve<GraphicsUnmapResourcesFunction>("cuGraphicsUnmapResources");
    const MapArrayAsyncFunction map_array_async =
        resolve<MapArrayAsyncFunction>("cuMemMapArrayAsync");
    const ReleaseFunction release = resolve<ReleaseFunction>("cuMemRelease");
    const PoolImportHandleFunction import_pool =
        resolve<PoolImportHandleFunction>("cuMemPoolImportFromShareableHandle");
    const PoolImportPointerFunction import_pointer =
        resolve<PoolImportPointerFunction>("cuMemPoolImportPointer");
    if (init == nullptr || import_handle == nullptr || ipc_get_handle == nullptr ||
        ipc_open_handle == nullptr || ipc_close_handle == nullptr ||
        import_external_memory == nullptr || get_external_buffer == nullptr ||
        get_external_mipmap == nullptr || destroy_external_memory == nullptr ||
        array_create == nullptr || array_3d_create == nullptr || array_destroy == nullptr ||
        mipmapped_array_create == nullptr || mipmapped_array_destroy == nullptr ||
        graphics_unregister_resource == nullptr || graphics_get_mapped_array == nullptr ||
        graphics_get_mapped_mipmap == nullptr || graphics_get_mapped_pointer == nullptr ||
        graphics_set_map_flags == nullptr || graphics_map_resources == nullptr ||
        graphics_unmap_resources == nullptr || map_array_async == nullptr || release == nullptr ||
        import_pool == nullptr || import_pointer == nullptr) {
        std::cerr << "CUDA import passthrough symbols were not exported\n";
        return EXIT_FAILURE;
    }
    if (init(0) != CUDA_SUCCESS) {
        std::cerr << "CUDA initialization failed\n";
        return EXIT_FAILURE;
    }

    CUmemGenericAllocationHandle imported_handle = 0;
    const bool imported = import_handle(&imported_handle, nullptr,
                                        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS &&
                          imported_handle != 0;
    const bool released = imported && release(imported_handle) == CUDA_SUCCESS;
    CUipcMemHandle ipc_handle{};
    const bool ipc_exported = ipc_get_handle(&ipc_handle, 0x100000U) == CUDA_SUCCESS;
    CUdeviceptr ipc_pointer = 0;
    const bool ipc_opened =
        ipc_open_handle(&ipc_pointer, ipc_handle, 0) == CUDA_SUCCESS && ipc_pointer != 0;
    const bool ipc_closed = ipc_opened && ipc_close_handle(ipc_pointer) == CUDA_SUCCESS;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC external_handle_desc{};
    CUexternalMemory external_memory = nullptr;
    const bool external_imported =
        import_external_memory(&external_memory, &external_handle_desc) == CUDA_SUCCESS &&
        external_memory != nullptr;
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC external_buffer_desc{};
    CUdeviceptr external_device_pointer = 0;
    const bool external_buffer_mapped =
        external_imported &&
        get_external_buffer(&external_device_pointer, external_memory, &external_buffer_desc) ==
            CUDA_SUCCESS &&
        external_device_pointer != 0;
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC external_mipmap_desc{};
    CUmipmappedArray external_mipmap = nullptr;
    const bool external_mipmap_mapped =
        external_imported &&
        get_external_mipmap(&external_mipmap, external_memory, &external_mipmap_desc) ==
            CUDA_SUCCESS &&
        external_mipmap != nullptr;
    const bool external_destroyed =
        external_imported && destroy_external_memory(external_memory) == CUDA_SUCCESS;
    CUDA_ARRAY_DESCRIPTOR array_descriptor{};
    CUarray array = nullptr;
    const bool array_created =
        array_create(&array, &array_descriptor) == CUDA_SUCCESS && array != nullptr;
    CUDA_ARRAY3D_DESCRIPTOR array_3d_descriptor{};
    CUarray array_3d = nullptr;
    const bool array_3d_created =
        array_3d_create(&array_3d, &array_3d_descriptor) == CUDA_SUCCESS && array_3d != nullptr;
    CUmipmappedArray mipmapped_array = nullptr;
    const bool mipmapped_array_created =
        mipmapped_array_create(&mipmapped_array, &array_3d_descriptor, 1) == CUDA_SUCCESS &&
        mipmapped_array != nullptr;
    const bool arrays_destroyed = array_created && array_destroy(array) == CUDA_SUCCESS &&
                                  mipmapped_array_created &&
                                  mipmapped_array_destroy(mipmapped_array) == CUDA_SUCCESS;
    CUgraphicsResource graphics_resource = reinterpret_cast<CUgraphicsResource>(0x64000000U);
    CUdeviceptr graphics_pointer = 0;
    std::size_t graphics_size = 0;
    CUarray graphics_array = nullptr;
    CUmipmappedArray graphics_mipmap = nullptr;
    const bool graphics_mapped =
        graphics_map_resources(1, &graphics_resource, nullptr) == CUDA_SUCCESS &&
        graphics_get_mapped_pointer(&graphics_pointer, &graphics_size, graphics_resource) ==
            CUDA_SUCCESS &&
        graphics_pointer != 0 && graphics_size != 0 &&
        graphics_get_mapped_array(&graphics_array, graphics_resource, 0, 0) == CUDA_SUCCESS &&
        graphics_array != nullptr &&
        graphics_get_mapped_mipmap(&graphics_mipmap, graphics_resource) == CUDA_SUCCESS &&
        graphics_mipmap != nullptr;
    const bool graphics_destroyed =
        graphics_mapped && graphics_set_map_flags(graphics_resource, 0) == CUDA_SUCCESS &&
        graphics_unmap_resources(1, &graphics_resource, nullptr) == CUDA_SUCCESS &&
        graphics_unregister_resource(graphics_resource) == CUDA_SUCCESS;
    const bool sparse_mapping_forwarded = map_array_async(nullptr, 0, nullptr) == CUDA_SUCCESS;
    CUmemoryPool imported_pool = nullptr;
    const bool pool_imported =
        import_pool(&imported_pool, nullptr, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
            CUDA_SUCCESS &&
        imported_pool != nullptr;
    CUmemPoolPtrExportData share_data{};
    CUdeviceptr imported_pointer = 0;
    const bool pointer_imported =
        import_pointer(&imported_pointer, imported_pool, &share_data) == CUDA_SUCCESS &&
        imported_pointer != 0;
    if (!imported || !released || !ipc_exported || !ipc_closed || !external_imported ||
        !external_buffer_mapped || !external_mipmap_mapped || !external_destroyed ||
        !array_created || !array_3d_created || !mipmapped_array_created || !arrays_destroyed ||
        !graphics_mapped || !graphics_destroyed || !sparse_mapping_forwarded || !pool_imported ||
        !pointer_imported) {
        std::cerr << "CUDA imports were not delegated without a quota\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
