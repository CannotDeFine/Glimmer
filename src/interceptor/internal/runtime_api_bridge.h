#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

namespace glimmer::interceptor {

using RuntimeMallocFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes);
using RuntimeMallocManagedFunction = cudaError_t (*)(void** device_pointer,
                                                     std::size_t memory_bytes, unsigned int flags);
using RuntimeMallocPitchFunction = cudaError_t (*)(void** device_pointer, std::size_t* pitch,
                                                   std::size_t width_bytes, std::size_t height);
using RuntimeMalloc3DFunction = cudaError_t (*)(struct cudaPitchedPtr* pitched_device_pointer,
                                                struct cudaExtent extent);
using RuntimeGetDeviceFunction = cudaError_t (*)(int* device);
using RuntimeFreeFunction = cudaError_t (*)(void* device_pointer);
using RuntimeIpcGetMemHandleFunction = cudaError_t (*)(cudaIpcMemHandle_t* handle,
                                                       void* device_pointer);
using RuntimeIpcOpenMemHandleFunction = cudaError_t (*)(void** device_pointer,
                                                        cudaIpcMemHandle_t handle,
                                                        unsigned int flags);
using RuntimeIpcCloseMemHandleFunction = cudaError_t (*)(void* device_pointer);
using RuntimeImportExternalMemoryFunction = cudaError_t (*)(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc);
using RuntimeExternalMemoryGetMappedBufferFunction =
    cudaError_t (*)(void** device_pointer, cudaExternalMemory_t external_memory,
                    const struct cudaExternalMemoryBufferDesc* buffer_desc);
using RuntimeExternalMemoryGetMappedMipmappedArrayFunction =
    cudaError_t (*)(cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
                    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc);
using RuntimeDestroyExternalMemoryFunction = cudaError_t (*)(cudaExternalMemory_t external_memory);
using RuntimeMallocArrayFunction = cudaError_t (*)(cudaArray_t* array,
                                                   const struct cudaChannelFormatDesc* descriptor,
                                                   std::size_t width, std::size_t height,
                                                   unsigned int flags);
using RuntimeMalloc3DArrayFunction = cudaError_t (*)(cudaArray_t* array,
                                                     const struct cudaChannelFormatDesc* descriptor,
                                                     struct cudaExtent extent, unsigned int flags);
using RuntimeMallocMipmappedArrayFunction =
    cudaError_t (*)(cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
                    struct cudaExtent extent, unsigned int level_count, unsigned int flags);
using RuntimeFreeArrayFunction = cudaError_t (*)(cudaArray_t array);
using RuntimeFreeMipmappedArrayFunction = cudaError_t (*)(cudaMipmappedArray_t mipmap);
using RuntimeGraphicsUnregisterResourceFunction = cudaError_t (*)(cudaGraphicsResource_t resource);
using RuntimeGraphicsResourceSetMapFlagsFunction = cudaError_t (*)(cudaGraphicsResource_t resource,
                                                                   unsigned int flags);
using RuntimeGraphicsMapResourcesFunction = cudaError_t (*)(int count,
                                                            cudaGraphicsResource_t* resources,
                                                            cudaStream_t stream);
using RuntimeGraphicsUnmapResourcesFunction = cudaError_t (*)(int count,
                                                              cudaGraphicsResource_t* resources,
                                                              cudaStream_t stream);
using RuntimeGraphicsResourceGetMappedPointerFunction =
    cudaError_t (*)(void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource);
using RuntimeGraphicsSubResourceGetMappedArrayFunction =
    cudaError_t (*)(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int array_index,
                    unsigned int mip_level);
using RuntimeGraphicsResourceGetMappedMipmappedArrayFunction =
    cudaError_t (*)(cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource);
using RuntimeGraphAddMemAllocNodeFunction = cudaError_t (*)(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters);
using RuntimeGraphLaunchFunction = cudaError_t (*)(cudaGraphExec_t graph_exec, cudaStream_t stream);
using RuntimeMemGetInfoFunction = cudaError_t (*)(std::size_t* free_bytes,
                                                  std::size_t* total_bytes);
using RuntimeMallocAsyncFunction = cudaError_t (*)(void** device_pointer, std::size_t memory_bytes,
                                                   cudaStream_t stream);
using RuntimeMallocFromPoolAsyncFunction = cudaError_t (*)(void** device_pointer,
                                                           std::size_t memory_bytes,
                                                           cudaMemPool_t pool, cudaStream_t stream);
using RuntimeFreeAsyncFunction = cudaError_t (*)(void* device_pointer, cudaStream_t stream);
using RuntimeDeviceSynchronizeFunction = cudaError_t (*)();
using RuntimeStreamSynchronizeFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamQueryFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeStreamDestroyFunction = cudaError_t (*)(cudaStream_t stream);
using RuntimeDeviceGetDefaultMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, int device);
using RuntimeDeviceSetMemPoolFunction = cudaError_t (*)(int device, cudaMemPool_t pool);
using RuntimeDeviceGetMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, int device);
using RuntimeMemPoolTrimToFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                     std::size_t min_bytes_to_keep);
using RuntimeMemPoolSetAttributeFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                           cudaMemPoolAttr attribute, void* value);
using RuntimeMemPoolGetAttributeFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                           cudaMemPoolAttr attribute, void* value);
using RuntimeMemPoolSetAccessFunction = cudaError_t (*)(cudaMemPool_t pool,
                                                        const cudaMemAccessDesc* descriptors,
                                                        std::size_t descriptor_count);
using RuntimeMemPoolGetAccessFunction = cudaError_t (*)(cudaMemAccessFlags* flags,
                                                        cudaMemPool_t pool,
                                                        cudaMemLocation* location);
using RuntimeMemPoolCreateFunction = cudaError_t (*)(cudaMemPool_t* pool,
                                                     const cudaMemPoolProps* properties);
using RuntimeMemPoolDestroyFunction = cudaError_t (*)(cudaMemPool_t pool);
using RuntimeMemGetDefaultMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool,
                                                            cudaMemLocation* location,
                                                            cudaMemAllocationType allocation_type);
using RuntimeMemGetMemPoolFunction = cudaError_t (*)(cudaMemPool_t* pool, cudaMemLocation* location,
                                                     cudaMemAllocationType allocation_type);
using RuntimeMemSetMemPoolFunction = cudaError_t (*)(cudaMemLocation* location,
                                                     cudaMemAllocationType allocation_type,
                                                     cudaMemPool_t pool);
using RuntimeMemPoolExportToShareableHandleFunction =
    cudaError_t (*)(void* handle_out, cudaMemPool_t pool,
                    enum cudaMemAllocationHandleType handle_type, unsigned int flags);
using RuntimeMemPoolImportFromShareableHandleFunction =
    cudaError_t (*)(cudaMemPool_t* pool_out, void* handle,
                    enum cudaMemAllocationHandleType handle_type, unsigned int flags);
using RuntimeMemPoolExportPointerFunction =
    cudaError_t (*)(cudaMemPoolPtrExportData* share_data_out, void* device_pointer);
using RuntimeMemPoolImportPointerFunction = cudaError_t (*)(void** pointer_out, cudaMemPool_t pool,
                                                            cudaMemPoolPtrExportData* share_data);

[[nodiscard]] bool is_inside_runtime_call() noexcept;

[[nodiscard]] cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                                   RuntimeMallocFunction allocate,
                                                   RuntimeFreeFunction release,
                                                   RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_managed(void** device_pointer,
                                                           std::size_t memory_bytes,
                                                           unsigned int flags,
                                                           RuntimeMallocManagedFunction allocate,
                                                           RuntimeFreeFunction release,
                                                           RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_pitch(void** device_pointer, std::size_t* pitch,
                                                         std::size_t width_bytes,
                                                         std::size_t height,
                                                         RuntimeMallocPitchFunction allocate,
                                                         RuntimeFreeFunction release,
                                                         RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_3d(struct cudaPitchedPtr* pitched_device_pointer,
                                                      struct cudaExtent extent,
                                                      RuntimeMalloc3DFunction allocate,
                                                      RuntimeFreeFunction release,
                                                      RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_free(void* device_pointer, RuntimeFreeFunction release);
[[nodiscard]] cudaError_t intercept_runtime_ipc_get_mem_handle(
    cudaIpcMemHandle_t* handle, void* device_pointer, RuntimeIpcGetMemHandleFunction get_handle);
[[nodiscard]] cudaError_t intercept_runtime_ipc_open_mem_handle(
    void** device_pointer, cudaIpcMemHandle_t handle, unsigned int flags,
    RuntimeIpcOpenMemHandleFunction open_handle);
[[nodiscard]] cudaError_t intercept_runtime_ipc_close_mem_handle(
    void* device_pointer, RuntimeIpcCloseMemHandleFunction close_handle);
[[nodiscard]] cudaError_t intercept_runtime_import_external_memory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc,
    RuntimeImportExternalMemoryFunction import_memory);
[[nodiscard]] cudaError_t intercept_runtime_external_memory_get_mapped_buffer(
    void** device_pointer, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryBufferDesc* buffer_desc,
    RuntimeExternalMemoryGetMappedBufferFunction get_buffer);
[[nodiscard]] cudaError_t intercept_runtime_external_memory_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc,
    RuntimeExternalMemoryGetMappedMipmappedArrayFunction get_mipmap);
[[nodiscard]] cudaError_t intercept_runtime_destroy_external_memory(
    cudaExternalMemory_t external_memory, RuntimeDestroyExternalMemoryFunction destroy_memory);
[[nodiscard]] cudaError_t intercept_runtime_malloc_array(
    cudaArray_t* array, const struct cudaChannelFormatDesc* descriptor, std::size_t width,
    std::size_t height, unsigned int flags, RuntimeMallocArrayFunction allocate);
[[nodiscard]] cudaError_t intercept_runtime_malloc_3d_array(
    cudaArray_t* array, const struct cudaChannelFormatDesc* descriptor, struct cudaExtent extent,
    unsigned int flags, RuntimeMalloc3DArrayFunction allocate);
[[nodiscard]] cudaError_t intercept_runtime_malloc_mipmapped_array(
    cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
    struct cudaExtent extent, unsigned int level_count, unsigned int flags,
    RuntimeMallocMipmappedArrayFunction allocate);
[[nodiscard]] cudaError_t intercept_runtime_free_array(cudaArray_t array,
                                                       RuntimeFreeArrayFunction release);
[[nodiscard]] cudaError_t intercept_runtime_free_mipmapped_array(
    cudaMipmappedArray_t mipmap, RuntimeFreeMipmappedArrayFunction release);
[[nodiscard]] cudaError_t intercept_runtime_graphics_unregister_resource(
    cudaGraphicsResource_t resource, RuntimeGraphicsUnregisterResourceFunction unregister);
[[nodiscard]] cudaError_t intercept_runtime_graphics_resource_set_map_flags(
    cudaGraphicsResource_t resource, unsigned int flags,
    RuntimeGraphicsResourceSetMapFlagsFunction set_flags);
[[nodiscard]] cudaError_t intercept_runtime_graphics_map_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsMapResourcesFunction map_resources);
[[nodiscard]] cudaError_t intercept_runtime_graphics_unmap_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsUnmapResourcesFunction unmap_resources);
[[nodiscard]] cudaError_t intercept_runtime_graphics_resource_get_mapped_pointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedPointerFunction get_pointer);
[[nodiscard]] cudaError_t intercept_runtime_graphics_subresource_get_mapped_array(
    cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int array_index,
    unsigned int mip_level, RuntimeGraphicsSubResourceGetMappedArrayFunction get_array);
[[nodiscard]] cudaError_t intercept_runtime_graphics_resource_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedMipmappedArrayFunction get_mipmap);
[[nodiscard]] cudaError_t intercept_runtime_graph_add_mem_alloc_node(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters,
    RuntimeGraphAddMemAllocNodeFunction add_node);
[[nodiscard]] cudaError_t intercept_runtime_graph_launch(const char* api_name,
                                                         cudaGraphExec_t graph_exec,
                                                         cudaStream_t stream,
                                                         RuntimeGraphLaunchFunction launch,
                                                         bool per_thread_default_stream = false);
[[nodiscard]] cudaError_t intercept_runtime_mem_get_info(std::size_t* free_bytes,
                                                         std::size_t* total_bytes,
                                                         RuntimeMemGetInfoFunction query);
[[nodiscard]] cudaError_t intercept_runtime_malloc_async(
    void** device_pointer, std::size_t memory_bytes, cudaStream_t stream,
    RuntimeMallocAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_malloc_from_pool_async(
    void** device_pointer, std::size_t memory_bytes, cudaMemPool_t pool, cudaStream_t stream,
    RuntimeMallocFromPoolAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device);
[[nodiscard]] cudaError_t intercept_runtime_mem_pool_import_from_shareable_handle(
    cudaMemPool_t* pool_out, void* handle, cudaMemAllocationHandleType handle_type,
    unsigned long long flags, RuntimeMemPoolImportFromShareableHandleFunction import_pool);
[[nodiscard]] cudaError_t intercept_runtime_mem_pool_import_pointer(
    void** pointer_out, cudaMemPool_t pool, cudaMemPoolPtrExportData* share_data,
    RuntimeMemPoolImportPointerFunction import_pointer);
[[nodiscard]] cudaError_t intercept_runtime_free_async(void* device_pointer, cudaStream_t stream,
                                                       RuntimeFreeAsyncFunction release);
[[nodiscard]] cudaError_t intercept_runtime_device_synchronize(
    RuntimeDeviceSynchronizeFunction synchronize);
[[nodiscard]] cudaError_t intercept_runtime_stream_synchronize(
    cudaStream_t stream, RuntimeStreamSynchronizeFunction synchronize);
[[nodiscard]] cudaError_t intercept_runtime_stream_query(cudaStream_t stream,
                                                         RuntimeStreamQueryFunction query);
[[nodiscard]] cudaError_t intercept_runtime_stream_destroy(cudaStream_t stream,
                                                           RuntimeStreamDestroyFunction destroy);

}  // namespace glimmer::interceptor
