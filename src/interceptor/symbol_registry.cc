#include "internal/symbol_registry.h"
#include "internal/driver_copy_wrappers.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#ifdef cudaStreamBeginCapture
#undef cudaStreamBeginCapture
#endif
#ifdef cudaStreamEndCapture
#undef cudaStreamEndCapture
#endif

#ifdef cudaGraphInstantiateWithParams
#undef cudaGraphInstantiateWithParams
#endif
#if CUDART_VERSION >= 12000
extern "C" cudaError_t CUDARTAPI cudaGraphInstantiateWithParams_ptsz(cudaGraphExec_t*, cudaGraph_t,
                                                                     cudaGraphInstantiateParams*);
#endif

#ifdef cudaMemcpy
#undef cudaMemcpy
#endif
#ifdef cudaMemcpy_ptds
#undef cudaMemcpy_ptds
#endif
#ifdef cudaMemcpyAsync
#undef cudaMemcpyAsync
#endif
#ifdef cudaMemcpyAsync_ptsz
#undef cudaMemcpyAsync_ptsz
#endif
#ifdef cudaMemset
#undef cudaMemset
#endif
#ifdef cudaMemset_ptds
#undef cudaMemset_ptds
#endif
#ifdef cudaMemsetAsync
#undef cudaMemsetAsync
#endif
#ifdef cudaMemsetAsync_ptsz
#undef cudaMemsetAsync_ptsz
#endif
#ifdef cudaMallocHost
#undef cudaMallocHost
#endif
#ifdef cudaHostAlloc
#undef cudaHostAlloc
#endif
#ifdef cudaFreeHost
#undef cudaFreeHost
#endif
#ifdef cudaHostRegister
#undef cudaHostRegister
#endif
#ifdef cudaHostUnregister
#undef cudaHostUnregister
#endif
#ifdef cudaHostGetDevicePointer
#undef cudaHostGetDevicePointer
#endif
#ifdef cudaStreamCreate
#undef cudaStreamCreate
#endif
#ifdef cudaStreamCreateWithFlags
#undef cudaStreamCreateWithFlags
#endif
#ifdef cudaStreamCreateWithPriority
#undef cudaStreamCreateWithPriority
#endif
#ifdef cudaStreamGetFlags
#undef cudaStreamGetFlags
#endif
#ifdef cudaStreamGetPriority
#undef cudaStreamGetPriority
#endif

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

#ifdef cudaEventRecord
#undef cudaEventRecord
#endif

#ifdef cudaEventRecordWithFlags
#undef cudaEventRecordWithFlags
#endif

#ifdef cudaStreamWaitEvent
#undef cudaStreamWaitEvent
#endif
#include <nvml.h>

#include <array>
#include <string_view>
#include "internal/driver_graph_wrappers.h"

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
#ifdef cudaLaunchKernelExC
#undef cudaLaunchKernelExC
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
#ifdef cuStreamGetDevice
#undef cuStreamGetDevice
#endif
#ifdef cuStreamGetCtx
#undef cuStreamGetCtx
#endif
#ifdef cuStreamGetCtx_v2
#undef cuStreamGetCtx_v2
#endif
#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
#ifdef cuIpcOpenMemHandle
#undef cuIpcOpenMemHandle
#endif
#ifdef cuArrayCreate
#undef cuArrayCreate
#endif
#ifdef cuArray3DCreate
#undef cuArray3DCreate
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
#ifdef cuMemGetAddressRange
#undef cuMemGetAddressRange
#endif
#ifdef nvmlInit
#undef nvmlInit
#endif
#ifdef nvmlDeviceGetCount
#undef nvmlDeviceGetCount
#endif
#ifdef nvmlDeviceGetHandleByIndex
#undef nvmlDeviceGetHandleByIndex
#endif
#ifdef cudaMallocFromPoolAsync
#undef cudaMallocFromPoolAsync
#endif
#ifdef cudaLaunchKernel
#undef cudaLaunchKernel
#endif
#ifdef cudaGraphLaunch
#undef cudaGraphLaunch
#endif

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* device_pointer, std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuInit(unsigned int flags);
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned int grid_dim_x,
                                           unsigned int grid_dim_y, unsigned int grid_dim_z,
                                           unsigned int block_dim_x, unsigned int block_dim_y,
                                           unsigned int block_dim_z,
                                           unsigned int shared_memory_bytes, CUstream stream,
                                           void** kernel_parameters, void** extra);
extern "C" CUresult CUDAAPI cuLaunchKernel_ptsz(CUfunction function, unsigned int grid_dim_x,
                                                unsigned int grid_dim_y, unsigned int grid_dim_z,
                                                unsigned int block_dim_x, unsigned int block_dim_y,
                                                unsigned int block_dim_z,
                                                unsigned int shared_memory_bytes, CUstream stream,
                                                void** kernel_parameters, void** extra);
extern "C" CUresult CUDAAPI cuGraphLaunch(CUgraphExec graph_exec, CUstream stream);
extern "C" CUresult CUDAAPI cuGraphLaunch_ptsz(CUgraphExec graph_exec, CUstream stream);
extern "C" CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              unsigned int flags);
extern "C" CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr* device_pointer, std::size_t* pitch,
                                               std::size_t width_bytes, std::size_t height,
                                               unsigned int element_size_bytes);
extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr device_pointer);
extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes);
extern "C" CUresult CUDAAPI cuDeviceTotalMem_v2(std::size_t* total_bytes, CUdevice device);
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context);
extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device);
extern "C" CUresult CUDAAPI cuCtxDestroy(CUcontext context);
extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context);
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice device);
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset_v2(CUdevice device);
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice device);
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease_v2(CUdevice device);
extern "C" CUresult CUDAAPI cuMemAllocAsync(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                            CUstream stream);
extern "C" CUresult CUDAAPI cuMemAllocAsync_ptsz(CUdeviceptr* device_pointer,
                                                 std::size_t memory_bytes, CUstream stream);
extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync(CUdeviceptr* device_pointer,
                                                    std::size_t memory_bytes, CUmemoryPool pool,
                                                    CUstream stream);
extern "C" CUresult CUDAAPI cuMemAllocFromPoolAsync_ptsz(CUdeviceptr* device_pointer,
                                                         std::size_t memory_bytes,
                                                         CUmemoryPool pool, CUstream stream);
extern "C" CUresult CUDAAPI cuMemCreate(CUmemGenericAllocationHandle* handle,
                                        std::size_t memory_bytes, const CUmemAllocationProp* prop,
                                        unsigned long long flags);
extern "C" CUresult CUDAAPI cuMemRelease(CUmemGenericAllocationHandle handle);
extern "C" CUresult CUDAAPI cuMemAddressReserve(CUdeviceptr* device_pointer,
                                                std::size_t memory_bytes, std::size_t alignment,
                                                CUdeviceptr requested_address,
                                                unsigned long long flags);
extern "C" CUresult CUDAAPI cuMemAddressFree(CUdeviceptr device_pointer, std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemMap(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                     std::size_t offset, CUmemGenericAllocationHandle handle,
                                     unsigned long long flags);
extern "C" CUresult CUDAAPI cuMemMapArrayAsync(CUarrayMapInfo* map_info_list, unsigned int count,
                                               CUstream stream);
extern "C" CUresult CUDAAPI cuMemUnmap(CUdeviceptr device_pointer, std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuMemSetAccess(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                           const CUmemAccessDesc* access_descriptors,
                                           std::size_t descriptor_count);
extern "C" CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr* base_pointer,
                                                    std::size_t* memory_bytes,
                                                    CUdeviceptr device_pointer);
extern "C" CUresult CUDAAPI cuMemGetAddressRange(CUdeviceptr* base_pointer,
                                                 std::size_t* memory_bytes,
                                                 CUdeviceptr device_pointer);
extern "C" CUresult CUDAAPI cuMemGetAccess(unsigned long long* flags, const CUmemLocation* location,
                                           CUdeviceptr device_pointer);
extern "C" CUresult CUDAAPI cuMemExportToShareableHandle(void* shareable_handle,
                                                         CUmemGenericAllocationHandle handle,
                                                         CUmemAllocationHandleType handle_type,
                                                         unsigned long long flags);
extern "C" CUresult CUDAAPI cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                                                           void* os_handle,
                                                           CUmemAllocationHandleType handle_type);
extern "C" CUresult CUDAAPI cuIpcGetMemHandle(CUipcMemHandle* handle, CUdeviceptr device_pointer);
extern "C" CUresult CUDAAPI cuIpcOpenMemHandle(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                               unsigned int flags);
extern "C" CUresult CUDAAPI cuIpcOpenMemHandle_v2(CUdeviceptr* device_pointer,
                                                  CUipcMemHandle handle, unsigned int flags);
extern "C" CUresult CUDAAPI cuIpcCloseMemHandle(CUdeviceptr device_pointer);
extern "C" CUresult CUDAAPI cuImportExternalMemory(
    CUexternalMemory* external_memory, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc);
extern "C" CUresult CUDAAPI
cuExternalMemoryGetMappedBuffer(CUdeviceptr* device_pointer, CUexternalMemory external_memory,
                                const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc);
extern "C" CUresult CUDAAPI cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray* mipmap, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc);
extern "C" CUresult CUDAAPI cuDestroyExternalMemory(CUexternalMemory external_memory);
extern "C" CUresult CUDAAPI cuArrayCreate(CUarray* array, const CUDA_ARRAY_DESCRIPTOR* descriptor);
extern "C" CUresult CUDAAPI cuArrayCreate_v2(CUarray* array,
                                             const CUDA_ARRAY_DESCRIPTOR* descriptor);
extern "C" CUresult CUDAAPI cuArray3DCreate(CUarray* array,
                                            const CUDA_ARRAY3D_DESCRIPTOR* descriptor);
extern "C" CUresult CUDAAPI cuArray3DCreate_v2(CUarray* array,
                                               const CUDA_ARRAY3D_DESCRIPTOR* descriptor);
extern "C" CUresult CUDAAPI cuArrayDestroy(CUarray array);
extern "C" CUresult CUDAAPI cuMipmappedArrayCreate(CUmipmappedArray* mipmap,
                                                   const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                                   unsigned int level_count);
extern "C" CUresult CUDAAPI cuMipmappedArrayDestroy(CUmipmappedArray mipmap);
extern "C" CUresult CUDAAPI cuGraphicsUnregisterResource(CUgraphicsResource resource);
extern "C" CUresult CUDAAPI cuGraphicsSubResourceGetMappedArray(CUarray* array,
                                                                CUgraphicsResource resource,
                                                                unsigned int array_index,
                                                                unsigned int mip_level);
extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray* mipmap,
                                                                      CUgraphicsResource resource);
extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr* device_pointer,
                                                                  std::size_t* size,
                                                                  CUgraphicsResource resource);
extern "C" CUresult CUDAAPI cuGraphicsResourceGetMappedPointer(CUdeviceptr* device_pointer,
                                                               std::size_t* size,
                                                               CUgraphicsResource resource);
extern "C" CUresult CUDAAPI cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                                             unsigned int flags);
extern "C" CUresult CUDAAPI cuGraphicsResourceSetMapFlags(CUgraphicsResource resource,
                                                          unsigned int flags);
extern "C" CUresult CUDAAPI cuGraphicsMapResources(unsigned int count,
                                                   CUgraphicsResource* resources, CUstream stream);
extern "C" CUresult CUDAAPI cuGraphicsUnmapResources(unsigned int count,
                                                     CUgraphicsResource* resources,
                                                     CUstream stream);
extern "C" CUresult CUDAAPI cuMemGetAllocationGranularity(std::size_t* granularity,
                                                          const CUmemAllocationProp* prop,
                                                          CUmemAllocationGranularity_flags option);
extern "C" CUresult CUDAAPI cuMemGetAllocationPropertiesFromHandle(
    CUmemAllocationProp* prop, CUmemGenericAllocationHandle handle);
extern "C" CUresult CUDAAPI cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle,
                                                        void* device_pointer);
extern "C" CUresult CUDAAPI cuMemPoolTrimTo(CUmemoryPool pool, std::size_t min_bytes_to_keep);
extern "C" CUresult CUDAAPI cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value);
extern "C" CUresult CUDAAPI cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                                  void* value);
extern "C" CUresult CUDAAPI cuMemPoolSetAccess(CUmemoryPool pool,
                                               const CUmemAccessDesc* access_descriptors,
                                               std::size_t descriptor_count);
extern "C" CUresult CUDAAPI cuMemPoolGetAccess(CUmemAccess_flags* flags, CUmemoryPool pool,
                                               CUmemLocation* location);
extern "C" CUresult CUDAAPI cuMemPoolCreate(CUmemoryPool* pool, const CUmemPoolProps* properties);
extern "C" CUresult CUDAAPI cuMemPoolDestroy(CUmemoryPool pool);
extern "C" CUresult CUDAAPI cuDeviceGetMemPool(CUmemoryPool* pool, CUdevice device);
extern "C" CUresult CUDAAPI cuDeviceSetMemPool(CUdevice device, CUmemoryPool pool);
extern "C" CUresult CUDAAPI cuDeviceGetDefaultMemPool(CUmemoryPool* pool, CUdevice device);
extern "C" CUresult CUDAAPI cuMemGetDefaultMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                                   CUmemAllocationType allocation_type);
extern "C" CUresult CUDAAPI cuMemGetMemPool(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType allocation_type);
extern "C" CUresult CUDAAPI cuMemSetMemPool(CUmemLocation* location,
                                            CUmemAllocationType allocation_type, CUmemoryPool pool);
extern "C" CUresult CUDAAPI cuMemPoolExportToShareableHandle(void* handle_out, CUmemoryPool pool,
                                                             CUmemAllocationHandleType handle_type,
                                                             unsigned long long flags);
extern "C" CUresult CUDAAPI
cuMemPoolImportFromShareableHandle(CUmemoryPool* pool_out, void* handle,
                                   CUmemAllocationHandleType handle_type, unsigned long long flags);
extern "C" CUresult CUDAAPI cuMemPoolExportPointer(CUmemPoolPtrExportData* share_data_out,
                                                   CUdeviceptr device_pointer);
extern "C" CUresult CUDAAPI cuMemPoolImportPointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                                   CUmemPoolPtrExportData* share_data);
extern "C" CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr device_pointer, CUstream stream);
extern "C" CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr device_pointer, CUstream stream);
extern "C" CUresult CUDAAPI cuStreamGetDevice(CUstream stream, CUdevice* device);
extern "C" CUresult CUDAAPI cuStreamGetDevice_ptsz(CUstream stream, CUdevice* device);
extern "C" CUresult CUDAAPI cuStreamGetCtx(CUstream stream, CUcontext* context);
extern "C" CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream stream, CUcontext* context);
extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream);
extern "C" CUresult CUDAAPI cuEventCreate(CUevent* event, unsigned int flags);
extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream stream);
extern "C" CUresult CUDAAPI cuEventQuery(CUevent event);
extern "C" CUresult CUDAAPI cuEventDestroy_v2(CUevent event);
extern "C" CUresult CUDAAPI cuEventRecord_ptsz(CUevent event, CUstream stream);
extern "C" CUresult CUDAAPI cuEventRecordWithFlags(CUevent event, CUstream stream,
                                                   unsigned int flags);
extern "C" CUresult CUDAAPI cuEventRecordWithFlags_ptsz(CUevent event, CUstream stream,
                                                        unsigned int flags);
extern "C" CUresult CUDAAPI cuEventSynchronize(CUevent event);
extern "C" CUresult CUDAAPI cuEventElapsedTime(float* milliseconds, CUevent start, CUevent end);
extern "C" CUresult CUDAAPI cuStreamWaitEvent(CUstream stream, CUevent event, unsigned int flags);
extern "C" CUresult CUDAAPI cuStreamWaitEvent_ptsz(CUstream stream, CUevent event,
                                                   unsigned int flags);
extern "C" CUresult CUDAAPI cuEventDestroy(CUevent event);
extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t* event);
extern "C" cudaError_t CUDARTAPI cudaMemcpy(void* destination, const void* source,
                                            std::size_t memory_bytes, cudaMemcpyKind kind);
extern "C" cudaError_t CUDARTAPI cudaMemcpy_ptds(void* destination, const void* source,
                                                 std::size_t memory_bytes, cudaMemcpyKind kind);
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync(void* destination, const void* source,
                                                 std::size_t memory_bytes, cudaMemcpyKind kind,
                                                 cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMemcpyAsync_ptsz(void* destination, const void* source,
                                                      std::size_t memory_bytes, cudaMemcpyKind kind,
                                                      cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMemset(void* pointer, int value, std::size_t memory_bytes);
extern "C" cudaError_t CUDARTAPI cudaMemset_ptds(void* pointer, int value,
                                                 std::size_t memory_bytes);
extern "C" cudaError_t CUDARTAPI cudaMemsetAsync(void* pointer, int value, std::size_t memory_bytes,
                                                 cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMemsetAsync_ptsz(void* pointer, int value,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMallocHost(void** pointer, std::size_t memory_bytes);
extern "C" cudaError_t CUDARTAPI cudaHostAlloc(void** pointer, std::size_t memory_bytes,
                                               unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaFreeHost(void* pointer);
extern "C" cudaError_t CUDARTAPI cudaHostRegister(void* pointer, std::size_t memory_bytes,
                                                  unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaHostUnregister(void* pointer);
extern "C" cudaError_t CUDARTAPI cudaHostGetDevicePointer(void** device_pointer, void* host_pointer,
                                                          unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaStreamCreate(cudaStream_t* stream);
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t* stream,
                                                           unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t* stream,
                                                              unsigned int flags, int priority);
extern "C" cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags);
extern "C" cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t stream, int* priority);
extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaEventRecord_ptsz(cudaEvent_t event, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream,
                                                          unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaEventRecordWithFlags_ptsz(cudaEvent_t event,
                                                               cudaStream_t stream,
                                                               unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event);
extern "C" cudaError_t CUDARTAPI cudaEventSynchronize(cudaEvent_t event);
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event);
extern "C" cudaError_t CUDARTAPI cudaEventElapsedTime(float* milliseconds, cudaEvent_t start,
                                                      cudaEvent_t end);
extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                                     unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaStreamWaitEvent_ptsz(cudaStream_t stream, cudaEvent_t event,
                                                          unsigned int flags);
extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream stream);
extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream stream);
extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream stream);
extern "C" CUresult CUDAAPI cuStreamDestroy(CUstream stream);
extern "C" CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream);
extern "C" CUresult CUDAAPI cuCtxSynchronize();
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags);
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** function_pointer,
                                                int cuda_version, cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status);
extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes);
extern "C" cudaError_t CUDARTAPI cudaMallocManaged(void** device_pointer, std::size_t memory_bytes,
                                                   unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaMallocPitch(void** device_pointer, std::size_t* pitch,
                                                 std::size_t width_bytes, std::size_t height);
extern "C" cudaError_t CUDARTAPI cudaMalloc3D(struct cudaPitchedPtr* pitched_device_pointer,
                                              struct cudaExtent extent);
extern "C" cudaError_t CUDARTAPI cudaLaunchKernel(const void* function, dim3 grid_dim,
                                                  dim3 block_dim, void** arguments,
                                                  std::size_t shared_memory_bytes,
                                                  cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaLaunchKernel_ptsz(const void* function, dim3 grid_dim,
                                                       dim3 block_dim, void** arguments,
                                                       std::size_t shared_memory_bytes,
                                                       cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch(cudaGraphExec_t graph_exec, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaGraphLaunch_ptsz(cudaGraphExec_t graph_exec,
                                                      cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamBeginCapture_ptsz(cudaStream_t, cudaStreamCaptureMode);
extern "C" cudaError_t CUDARTAPI cudaStreamEndCapture_ptsz(cudaStream_t, cudaGraph_t*);
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming): preserve CUDA compiler
// ABI names.
extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel(cudaKernel_t kernel, dim3 grid_dim,
                                                    dim3 block_dim, void** arguments,
                                                    std::size_t shared_memory_bytes,
                                                    cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 grid_dim,
                                                         dim3 block_dim, void** arguments,
                                                         std::size_t shared_memory_bytes,
                                                         cudaStream_t stream);
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** device_pointer,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** device_pointer,
                                                         std::size_t memory_bytes,
                                                         cudaMemPool_t pool, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync_ptsz(void** device_pointer,
                                                              std::size_t memory_bytes,
                                                              cudaMemPool_t pool,
                                                              cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer);
extern "C" cudaError_t CUDARTAPI cudaIpcGetMemHandle(cudaIpcMemHandle_t* handle,
                                                     void* device_pointer);
extern "C" cudaError_t CUDARTAPI cudaIpcOpenMemHandle(void** device_pointer,
                                                      cudaIpcMemHandle_t handle,
                                                      unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaIpcCloseMemHandle(void* device_pointer);
extern "C" cudaError_t CUDARTAPI cudaImportExternalMemory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc);
extern "C" cudaError_t CUDARTAPI
cudaExternalMemoryGetMappedBuffer(void** device_pointer, cudaExternalMemory_t external_memory,
                                  const struct cudaExternalMemoryBufferDesc* buffer_desc);
extern "C" cudaError_t CUDARTAPI cudaExternalMemoryGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc);
extern "C" cudaError_t CUDARTAPI cudaDestroyExternalMemory(cudaExternalMemory_t external_memory);
extern "C" cudaError_t CUDARTAPI cudaMallocArray(cudaArray_t* array,
                                                 const struct cudaChannelFormatDesc* descriptor,
                                                 std::size_t width, std::size_t height,
                                                 unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaMalloc3DArray(cudaArray_t* array,
                                                   const struct cudaChannelFormatDesc* descriptor,
                                                   struct cudaExtent extent, unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaMallocMipmappedArray(
    cudaMipmappedArray_t* mipmap, const struct cudaChannelFormatDesc* descriptor,
    struct cudaExtent extent, unsigned int level_count, unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaFreeArray(cudaArray_t array);
extern "C" cudaError_t CUDARTAPI cudaFreeMipmappedArray(cudaMipmappedArray_t mipmap);
extern "C" cudaError_t CUDARTAPI cudaGraphicsUnregisterResource(cudaGraphicsResource_t resource);
extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceSetMapFlags(cudaGraphicsResource_t resource,
                                                                 unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaGraphicsMapResources(int count,
                                                          cudaGraphicsResource_t* resources,
                                                          cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaGraphicsUnmapResources(int count,
                                                            cudaGraphicsResource_t* resources,
                                                            cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedPointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource);
extern "C" cudaError_t CUDARTAPI
cudaGraphicsSubResourceGetMappedArray(cudaArray_t* array, cudaGraphicsResource_t resource,
                                      unsigned int array_index, unsigned int mip_level);
extern "C" cudaError_t CUDARTAPI cudaGraphicsResourceGetMappedMipmappedArray(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource);
extern "C" cudaError_t CUDARTAPI cudaGraphAddMemAllocNode(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters);
extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* device_pointer, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes);
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize();
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device);
extern "C" cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t pool);
extern "C" cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t* pool, int device);
extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool,
                                                   std::size_t min_bytes_to_keep);
extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value);
extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value);
extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t pool,
                                                      const cudaMemAccessDesc* descriptors,
                                                      std::size_t descriptor_count);
extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t pool,
                                                      cudaMemLocation* location);
extern "C" cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                                   const cudaMemPoolProps* properties);
extern "C" cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool);
extern "C" cudaError_t CUDARTAPI cudaMemGetDefaultMemPool(cudaMemPool_t* pool,
                                                          cudaMemLocation* location,
                                                          cudaMemAllocationType allocation_type);
extern "C" cudaError_t CUDARTAPI cudaMemGetMemPool(cudaMemPool_t* pool, cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type);
extern "C" cudaError_t CUDARTAPI cudaMemSetMemPool(cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type,
                                                   cudaMemPool_t pool);
extern "C" cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void* handle_out, cudaMemPool_t pool, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t* pool_out, void* handle, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags);
extern "C" cudaError_t CUDARTAPI cudaMemPoolExportPointer(cudaMemPoolPtrExportData* share_data_out,
                                                          void* device_pointer);
extern "C" cudaError_t CUDARTAPI cudaMemPoolImportPointer(void** pointer_out, cudaMemPool_t pool,
                                                          cudaMemPoolPtrExportData* share_data);
// NOLINTBEGIN(readability-identifier-naming): preserve the NVML ABI names.
extern "C" nvmlReturn_t nvmlInit();
extern "C" nvmlReturn_t nvmlInit_v2();
extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags);
extern "C" nvmlReturn_t nvmlShutdown();
extern "C" nvmlReturn_t nvmlDeviceGetCount(unsigned int* device_count);
extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* device_count);
extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device);
extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device);
extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index);
extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory);
extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory);
// NOLINTEND(readability-identifier-naming)

namespace glimmer::interceptor {

extern "C" CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig*, CUfunction, void**, void**);
extern "C" CUresult CUDAAPI cuLaunchKernelEx_ptsz(const CUlaunchConfig*, CUfunction, void**,
                                                  void**);
extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC(const cudaLaunchConfig_t*, const void*,
                                                     void**);
extern "C" cudaError_t CUDARTAPI cudaLaunchKernelExC_ptsz(const cudaLaunchConfig_t*, const void*,
                                                          void**);

namespace {

struct InterceptorSymbol {
    std::string_view name;
    void* wrapper;
};

const auto kInterceptorSymbols = std::to_array<InterceptorSymbol>({
    {"cublasLtCreate", reinterpret_cast<void*>(&cublasLtCreate)},
    {"cublasLtDestroy", reinterpret_cast<void*>(&cublasLtDestroy)},
    {"cublasLtMatmul", reinterpret_cast<void*>(&cublasLtMatmul)},
    {"cublasSgemm_v2", reinterpret_cast<void*>(&cublasSgemm_v2)},
    {"cublasCreate_v2", reinterpret_cast<void*>(&cublasCreate_v2)},
    {"cublasDestroy_v2", reinterpret_cast<void*>(&cublasDestroy_v2)},
    {"cuMemcpyHtoDAsync_v2", reinterpret_cast<void*>(&cuMemcpyHtoDAsync_v2)},
    {"cuMemcpy", reinterpret_cast<void*>(&cuMemcpy)},
    {"cuMemcpy_ptds", reinterpret_cast<void*>(&cuMemcpy_ptds)},
    {"cuMemcpyAsync", reinterpret_cast<void*>(&cuMemcpyAsync)},
    {"cuMemcpyAsync_ptsz", reinterpret_cast<void*>(&cuMemcpyAsync_ptsz)},
    {"cuMemcpyHtoDAsync", reinterpret_cast<void*>(&cuMemcpyHtoDAsync_v2)},
    {"cuMemcpyHtoDAsync_v2_ptsz", reinterpret_cast<void*>(&cuMemcpyHtoDAsync_v2_ptsz)},
    {"cuMemcpyHtoDAsync_ptsz", reinterpret_cast<void*>(&cuMemcpyHtoDAsync_v2_ptsz)},
    {"cuMemcpyHtoD_v2", reinterpret_cast<void*>(&cuMemcpyHtoD_v2)},
    {"cuMemcpyHtoD", reinterpret_cast<void*>(&cuMemcpyHtoD_v2)},
    {"cuMemcpyHtoD_v2_ptds", reinterpret_cast<void*>(&cuMemcpyHtoD_v2_ptds)},
    {"cuMemcpyHtoD_ptds", reinterpret_cast<void*>(&cuMemcpyHtoD_v2_ptds)},
    {"cuMemcpyDtoHAsync_v2", reinterpret_cast<void*>(&cuMemcpyDtoHAsync_v2)},
    {"cuMemcpyDtoHAsync", reinterpret_cast<void*>(&cuMemcpyDtoHAsync_v2)},
    {"cuMemcpyDtoHAsync_v2_ptsz", reinterpret_cast<void*>(&cuMemcpyDtoHAsync_v2_ptsz)},
    {"cuMemcpyDtoHAsync_ptsz", reinterpret_cast<void*>(&cuMemcpyDtoHAsync_v2_ptsz)},
    {"cuMemcpyDtoH_v2", reinterpret_cast<void*>(&cuMemcpyDtoH_v2)},
    {"cuMemcpyDtoH", reinterpret_cast<void*>(&cuMemcpyDtoH_v2)},
    {"cuMemcpyDtoH_v2_ptds", reinterpret_cast<void*>(&cuMemcpyDtoH_v2_ptds)},
    {"cuMemcpyDtoH_ptds", reinterpret_cast<void*>(&cuMemcpyDtoH_v2_ptds)},
    {"cuMemcpyDtoDAsync_v2", reinterpret_cast<void*>(&cuMemcpyDtoDAsync_v2)},
    {"cuMemcpyDtoDAsync", reinterpret_cast<void*>(&cuMemcpyDtoDAsync_v2)},
    {"cuMemcpyDtoDAsync_v2_ptsz", reinterpret_cast<void*>(&cuMemcpyDtoDAsync_v2_ptsz)},
    {"cuMemcpyDtoDAsync_ptsz", reinterpret_cast<void*>(&cuMemcpyDtoDAsync_v2_ptsz)},
    {"cuMemcpyDtoD_v2", reinterpret_cast<void*>(&cuMemcpyDtoD_v2)},
    {"cuMemcpyDtoD", reinterpret_cast<void*>(&cuMemcpyDtoD_v2)},
    {"cuMemcpyDtoD_v2_ptds", reinterpret_cast<void*>(&cuMemcpyDtoD_v2_ptds)},
    {"cuMemcpyDtoD_ptds", reinterpret_cast<void*>(&cuMemcpyDtoD_v2_ptds)},
    {"cuMemHostAlloc", reinterpret_cast<void*>(&cuMemHostAlloc)},
    {"cuMemAllocHost_v2", reinterpret_cast<void*>(&cuMemAllocHost_v2)},
    {"cuMemAllocHost", reinterpret_cast<void*>(&cuMemAllocHost_v2)},
    {"cuMemFreeHost", reinterpret_cast<void*>(&cuMemFreeHost)},
    {"cuMemHostRegister_v2", reinterpret_cast<void*>(&cuMemHostRegister_v2)},
    {"cuMemHostRegister", reinterpret_cast<void*>(&cuMemHostRegister_v2)},
    {"cuMemHostUnregister", reinterpret_cast<void*>(&cuMemHostUnregister)},
    {"cuMemHostGetDevicePointer_v2", reinterpret_cast<void*>(&cuMemHostGetDevicePointer_v2)},
    {"cuMemHostGetDevicePointer", reinterpret_cast<void*>(&cuMemHostGetDevicePointer_v2)},
    {"cuStreamCreate", reinterpret_cast<void*>(&cuStreamCreate)},
    {"cuStreamCreateWithPriority", reinterpret_cast<void*>(&cuStreamCreateWithPriority)},
    {"cuStreamGetFlags", reinterpret_cast<void*>(&cuStreamGetFlags)},
    {"cuStreamGetPriority", reinterpret_cast<void*>(&cuStreamGetPriority)},
    {"cuMemsetD8_v2", reinterpret_cast<void*>(&cuMemsetD8_v2)},
    {"cuMemsetD8", reinterpret_cast<void*>(&cuMemsetD8_v2)},
    {"cuMemsetD8_v2_ptds", reinterpret_cast<void*>(&cuMemsetD8_v2_ptds)},
    {"cuMemsetD8_ptds", reinterpret_cast<void*>(&cuMemsetD8_v2_ptds)},
    {"cuMemsetD8Async", reinterpret_cast<void*>(&cuMemsetD8Async)},
    {"cuMemsetD8Async_ptsz", reinterpret_cast<void*>(&cuMemsetD8Async_ptsz)},
    {"cuMemsetD16_v2", reinterpret_cast<void*>(&cuMemsetD16_v2)},
    {"cuMemsetD16", reinterpret_cast<void*>(&cuMemsetD16_v2)},
    {"cuMemsetD16_v2_ptds", reinterpret_cast<void*>(&cuMemsetD16_v2_ptds)},
    {"cuMemsetD16_ptds", reinterpret_cast<void*>(&cuMemsetD16_v2_ptds)},
    {"cuMemsetD16Async", reinterpret_cast<void*>(&cuMemsetD16Async)},
    {"cuMemsetD16Async_ptsz", reinterpret_cast<void*>(&cuMemsetD16Async_ptsz)},
    {"cuMemsetD32_v2", reinterpret_cast<void*>(&cuMemsetD32_v2)},
    {"cuMemsetD32", reinterpret_cast<void*>(&cuMemsetD32_v2)},
    {"cuMemsetD32_v2_ptds", reinterpret_cast<void*>(&cuMemsetD32_v2_ptds)},
    {"cuMemsetD32_ptds", reinterpret_cast<void*>(&cuMemsetD32_v2_ptds)},
    {"cuMemsetD32Async", reinterpret_cast<void*>(&cuMemsetD32Async)},
    {"cuMemsetD32Async_ptsz", reinterpret_cast<void*>(&cuMemsetD32Async_ptsz)},
    {"cuStreamBeginCapture", reinterpret_cast<void*>(&cuStreamBeginCapture)},
    {"cuStreamBeginCapture_v2", reinterpret_cast<void*>(&cuStreamBeginCapture_v2)},
    {"cuStreamBeginCapture_ptsz", reinterpret_cast<void*>(&cuStreamBeginCapture_ptsz)},
    {"cuStreamBeginCapture_v2_ptsz", reinterpret_cast<void*>(&cuStreamBeginCapture_v2_ptsz)},
    {"cuStreamEndCapture", reinterpret_cast<void*>(&cuStreamEndCapture)},
    {"cuStreamEndCapture_ptsz", reinterpret_cast<void*>(&cuStreamEndCapture_ptsz)},
    {"cuInit", reinterpret_cast<void*>(&cuInit)},
    {"cuModuleLoadData", reinterpret_cast<void*>(&cuModuleLoadData)},
    {"cuModuleLoadDataEx", reinterpret_cast<void*>(&cuModuleLoadDataEx)},
    {"cuModuleGetFunction", reinterpret_cast<void*>(&cuModuleGetFunction)},
    {"cuLibraryLoadData", reinterpret_cast<void*>(&cuLibraryLoadData)},
    {"cuLibraryLoadFromFile", reinterpret_cast<void*>(&cuLibraryLoadFromFile)},
    {"cuLibraryGetModule", reinterpret_cast<void*>(&cuLibraryGetModule)},
    {"cuLibraryUnload", reinterpret_cast<void*>(&cuLibraryUnload)},
    {"cuModuleUnload", reinterpret_cast<void*>(&cuModuleUnload)},
    {"cuLaunchKernel", reinterpret_cast<void*>(&cuLaunchKernel)},
    {"cuLaunchKernel_ptsz", reinterpret_cast<void*>(&cuLaunchKernel_ptsz)},
    {"cuLaunchKernelEx", reinterpret_cast<void*>(&cuLaunchKernelEx)},
    {"cuLaunchKernelEx_ptsz", reinterpret_cast<void*>(&cuLaunchKernelEx_ptsz)},
    {"cuGraphLaunch", reinterpret_cast<void*>(&cuGraphLaunch)},
    {"cuGraphLaunch_ptsz", reinterpret_cast<void*>(&cuGraphLaunch_ptsz)},
    {"cuGraphInstantiate", reinterpret_cast<void*>(&cuGraphInstantiate)},
    {"cuGraphInstantiate_v2", reinterpret_cast<void*>(&cuGraphInstantiate_v2)},
    {"cuGraphInstantiateWithFlags", reinterpret_cast<void*>(&cuGraphInstantiateWithFlags)},
    {"cuGraphInstantiateWithParams", reinterpret_cast<void*>(&cuGraphInstantiateWithParams)},
    {"cuGraphInstantiateWithParams_ptsz",
     reinterpret_cast<void*>(&cuGraphInstantiateWithParams_ptsz)},
    {"cuGraphExecDestroy", reinterpret_cast<void*>(&cuGraphExecDestroy)},
    {"cuGraphExecEventRecordNodeSetEvent",
     reinterpret_cast<void*>(&cuGraphExecEventRecordNodeSetEvent)},
    {"cuGraphExecEventWaitNodeSetEvent",
     reinterpret_cast<void*>(&cuGraphExecEventWaitNodeSetEvent)},
    {"cuGraphExecChildGraphNodeSetParams",
     reinterpret_cast<void*>(&cuGraphExecChildGraphNodeSetParams)},
    {"cuGraphNodeSetEnabled", reinterpret_cast<void*>(&cuGraphNodeSetEnabled)},
#if CUDA_VERSION >= 12020
    {"cuGraphExecNodeSetParams", reinterpret_cast<void*>(&cuGraphExecNodeSetParams)},
#endif
    {"cuGraphExecUpdate", reinterpret_cast<void*>(&cuGraphExecUpdate)},
    {"cuGraphExecUpdate_v2", reinterpret_cast<void*>(&cuGraphExecUpdate_v2)},
    {"cuMemAlloc", reinterpret_cast<void*>(&cuMemAlloc_v2)},
    {"cuMemAlloc_v2", reinterpret_cast<void*>(&cuMemAlloc_v2)},
    {"cuMemAllocManaged", reinterpret_cast<void*>(&cuMemAllocManaged)},
    {"cuMemAllocPitch", reinterpret_cast<void*>(&cuMemAllocPitch_v2)},
    {"cuMemAllocPitch_v2", reinterpret_cast<void*>(&cuMemAllocPitch_v2)},
    {"cuMemFree", reinterpret_cast<void*>(&cuMemFree_v2)},
    {"cuMemFree_v2", reinterpret_cast<void*>(&cuMemFree_v2)},
    {"cuMemGetInfo", reinterpret_cast<void*>(&cuMemGetInfo_v2)},
    {"cuMemGetInfo_v2", reinterpret_cast<void*>(&cuMemGetInfo_v2)},
    {"cuDeviceTotalMem", reinterpret_cast<void*>(&cuDeviceTotalMem_v2)},
    {"cuDeviceTotalMem_v2", reinterpret_cast<void*>(&cuDeviceTotalMem_v2)},
    {"cuCtxGetCurrent", reinterpret_cast<void*>(&cuCtxGetCurrent)},
    {"cuCtxGetDevice", reinterpret_cast<void*>(&cuCtxGetDevice)},
    {"cuCtxDestroy", reinterpret_cast<void*>(&cuCtxDestroy)},
    {"cuCtxDestroy_v2", reinterpret_cast<void*>(&cuCtxDestroy_v2)},
    {"cuDevicePrimaryCtxReset", reinterpret_cast<void*>(&cuDevicePrimaryCtxReset)},
    {"cuDevicePrimaryCtxReset_v2", reinterpret_cast<void*>(&cuDevicePrimaryCtxReset_v2)},
    {"cuDevicePrimaryCtxRelease", reinterpret_cast<void*>(&cuDevicePrimaryCtxRelease)},
    {"cuDevicePrimaryCtxRelease_v2", reinterpret_cast<void*>(&cuDevicePrimaryCtxRelease_v2)},
    {"cuMemAllocAsync", reinterpret_cast<void*>(&cuMemAllocAsync)},
    {"cuMemAllocAsync_ptsz", reinterpret_cast<void*>(&cuMemAllocAsync_ptsz)},
    {"cuMemAllocFromPoolAsync", reinterpret_cast<void*>(&cuMemAllocFromPoolAsync)},
    {"cuMemAllocFromPoolAsync_ptsz", reinterpret_cast<void*>(&cuMemAllocFromPoolAsync_ptsz)},
    {"cuMemCreate", reinterpret_cast<void*>(&cuMemCreate)},
    {"cuMemRelease", reinterpret_cast<void*>(&cuMemRelease)},
    {"cuMemAddressReserve", reinterpret_cast<void*>(&cuMemAddressReserve)},
    {"cuMemAddressFree", reinterpret_cast<void*>(&cuMemAddressFree)},
    {"cuMemMap", reinterpret_cast<void*>(&cuMemMap)},
    {"cuMemMapArrayAsync", reinterpret_cast<void*>(&cuMemMapArrayAsync)},
    {"cuMemUnmap", reinterpret_cast<void*>(&cuMemUnmap)},
    {"cuMemSetAccess", reinterpret_cast<void*>(&cuMemSetAccess)},
    {"cuMemGetAddressRange", reinterpret_cast<void*>(&cuMemGetAddressRange_v2)},
    {"cuMemGetAddressRange_v2", reinterpret_cast<void*>(&cuMemGetAddressRange_v2)},
    {"cuMemGetAccess", reinterpret_cast<void*>(&cuMemGetAccess)},
    {"cuMemExportToShareableHandle", reinterpret_cast<void*>(&cuMemExportToShareableHandle)},
    {"cuMemImportFromShareableHandle", reinterpret_cast<void*>(&cuMemImportFromShareableHandle)},
    {"cuIpcGetMemHandle", reinterpret_cast<void*>(&cuIpcGetMemHandle)},
    {"cuIpcOpenMemHandle", reinterpret_cast<void*>(&cuIpcOpenMemHandle)},
    {"cuIpcOpenMemHandle_v2", reinterpret_cast<void*>(&cuIpcOpenMemHandle_v2)},
    {"cuIpcCloseMemHandle", reinterpret_cast<void*>(&cuIpcCloseMemHandle)},
    {"cuImportExternalMemory", reinterpret_cast<void*>(&cuImportExternalMemory)},
    {"cuExternalMemoryGetMappedBuffer", reinterpret_cast<void*>(&cuExternalMemoryGetMappedBuffer)},
    {"cuExternalMemoryGetMappedMipmappedArray",
     reinterpret_cast<void*>(&cuExternalMemoryGetMappedMipmappedArray)},
    {"cuDestroyExternalMemory", reinterpret_cast<void*>(&cuDestroyExternalMemory)},
    {"cuArrayCreate", reinterpret_cast<void*>(&cuArrayCreate)},
    {"cuArrayCreate_v2", reinterpret_cast<void*>(&cuArrayCreate_v2)},
    {"cuArray3DCreate", reinterpret_cast<void*>(&cuArray3DCreate)},
    {"cuArray3DCreate_v2", reinterpret_cast<void*>(&cuArray3DCreate_v2)},
    {"cuArrayDestroy", reinterpret_cast<void*>(&cuArrayDestroy)},
    {"cuMipmappedArrayCreate", reinterpret_cast<void*>(&cuMipmappedArrayCreate)},
    {"cuMipmappedArrayDestroy", reinterpret_cast<void*>(&cuMipmappedArrayDestroy)},
    {"cuGraphicsUnregisterResource", reinterpret_cast<void*>(&cuGraphicsUnregisterResource)},
    {"cuGraphicsSubResourceGetMappedArray",
     reinterpret_cast<void*>(&cuGraphicsSubResourceGetMappedArray)},
    {"cuGraphicsResourceGetMappedMipmappedArray",
     reinterpret_cast<void*>(&cuGraphicsResourceGetMappedMipmappedArray)},
    {"cuGraphicsResourceGetMappedPointer",
     reinterpret_cast<void*>(&cuGraphicsResourceGetMappedPointer)},
    {"cuGraphicsResourceGetMappedPointer_v2",
     reinterpret_cast<void*>(&cuGraphicsResourceGetMappedPointer_v2)},
    {"cuGraphicsResourceSetMapFlags", reinterpret_cast<void*>(&cuGraphicsResourceSetMapFlags)},
    {"cuGraphicsResourceSetMapFlags_v2",
     reinterpret_cast<void*>(&cuGraphicsResourceSetMapFlags_v2)},
    {"cuGraphicsMapResources", reinterpret_cast<void*>(&cuGraphicsMapResources)},
    {"cuGraphicsUnmapResources", reinterpret_cast<void*>(&cuGraphicsUnmapResources)},
    {"cuMemGetAllocationGranularity", reinterpret_cast<void*>(&cuMemGetAllocationGranularity)},
    {"cuMemGetAllocationPropertiesFromHandle",
     reinterpret_cast<void*>(&cuMemGetAllocationPropertiesFromHandle)},
    {"cuMemRetainAllocationHandle", reinterpret_cast<void*>(&cuMemRetainAllocationHandle)},
    {"cuMemPoolTrimTo", reinterpret_cast<void*>(&cuMemPoolTrimTo)},
    {"cuMemPoolSetAttribute", reinterpret_cast<void*>(&cuMemPoolSetAttribute)},
    {"cuMemPoolGetAttribute", reinterpret_cast<void*>(&cuMemPoolGetAttribute)},
    {"cuMemPoolSetAccess", reinterpret_cast<void*>(&cuMemPoolSetAccess)},
    {"cuMemPoolGetAccess", reinterpret_cast<void*>(&cuMemPoolGetAccess)},
    {"cuMemPoolCreate", reinterpret_cast<void*>(&cuMemPoolCreate)},
    {"cuMemPoolDestroy", reinterpret_cast<void*>(&cuMemPoolDestroy)},
    {"cuDeviceGetMemPool", reinterpret_cast<void*>(&cuDeviceGetMemPool)},
    {"cuDeviceSetMemPool", reinterpret_cast<void*>(&cuDeviceSetMemPool)},
    {"cuDeviceGetDefaultMemPool", reinterpret_cast<void*>(&cuDeviceGetDefaultMemPool)},
    {"cuMemGetDefaultMemPool", reinterpret_cast<void*>(&cuMemGetDefaultMemPool)},
    {"cuMemGetMemPool", reinterpret_cast<void*>(&cuMemGetMemPool)},
    {"cuMemSetMemPool", reinterpret_cast<void*>(&cuMemSetMemPool)},
    {"cuMemPoolExportToShareableHandle",
     reinterpret_cast<void*>(&cuMemPoolExportToShareableHandle)},
    {"cuMemPoolImportFromShareableHandle",
     reinterpret_cast<void*>(&cuMemPoolImportFromShareableHandle)},
    {"cuMemPoolExportPointer", reinterpret_cast<void*>(&cuMemPoolExportPointer)},
    {"cuMemPoolImportPointer", reinterpret_cast<void*>(&cuMemPoolImportPointer)},
    {"cuMemFreeAsync", reinterpret_cast<void*>(&cuMemFreeAsync)},
    {"cuMemFreeAsync_ptsz", reinterpret_cast<void*>(&cuMemFreeAsync_ptsz)},
    {"cuStreamGetDevice", reinterpret_cast<void*>(&cuStreamGetDevice)},
    {"cuStreamGetDevice_ptsz", reinterpret_cast<void*>(&cuStreamGetDevice_ptsz)},
    {"cuStreamGetCtx", reinterpret_cast<void*>(&cuStreamGetCtx)},
    {"cuStreamGetCtx_ptsz", reinterpret_cast<void*>(&cuStreamGetCtx_ptsz)},
    {"cuStreamQuery", reinterpret_cast<void*>(&cuStreamQuery)},
    {"cuEventCreate", reinterpret_cast<void*>(&cuEventCreate)},
    {"cuEventRecord", reinterpret_cast<void*>(&cuEventRecord)},
    {"cuEventQuery", reinterpret_cast<void*>(&cuEventQuery)},
    {"cuEventDestroy_v2", reinterpret_cast<void*>(&cuEventDestroy_v2)},
    {"cuEventRecord_ptsz", reinterpret_cast<void*>(&cuEventRecord_ptsz)},
    {"cuEventRecordWithFlags", reinterpret_cast<void*>(&cuEventRecordWithFlags)},
    {"cuEventRecordWithFlags_ptsz", reinterpret_cast<void*>(&cuEventRecordWithFlags_ptsz)},
    {"cuEventSynchronize", reinterpret_cast<void*>(&cuEventSynchronize)},
    {"cuEventElapsedTime", reinterpret_cast<void*>(&cuEventElapsedTime)},
    {"cuStreamWaitEvent", reinterpret_cast<void*>(&cuStreamWaitEvent)},
    {"cuStreamWaitEvent_ptsz", reinterpret_cast<void*>(&cuStreamWaitEvent_ptsz)},
    {"cuEventDestroy", reinterpret_cast<void*>(&cuEventDestroy)},
    {"cudaEventCreate", reinterpret_cast<void*>(&cudaEventCreate)},
    {"cudaMemcpy", reinterpret_cast<void*>(&cudaMemcpy)},
    {"cudaMemcpy_ptds", reinterpret_cast<void*>(&cudaMemcpy_ptds)},
    {"cudaMemcpyAsync", reinterpret_cast<void*>(&cudaMemcpyAsync)},
    {"cudaMemcpyAsync_ptsz", reinterpret_cast<void*>(&cudaMemcpyAsync_ptsz)},
    {"cudaMemset", reinterpret_cast<void*>(&cudaMemset)},
    {"cudaMemset_ptds", reinterpret_cast<void*>(&cudaMemset_ptds)},
    {"cudaMemsetAsync", reinterpret_cast<void*>(&cudaMemsetAsync)},
    {"cudaMemsetAsync_ptsz", reinterpret_cast<void*>(&cudaMemsetAsync_ptsz)},
    {"cudaMallocHost", reinterpret_cast<void*>(&cudaMallocHost)},
    {"cudaHostAlloc", reinterpret_cast<void*>(&cudaHostAlloc)},
    {"cudaFreeHost", reinterpret_cast<void*>(&cudaFreeHost)},
    {"cudaHostRegister", reinterpret_cast<void*>(&cudaHostRegister)},
    {"cudaHostUnregister", reinterpret_cast<void*>(&cudaHostUnregister)},
    {"cudaHostGetDevicePointer", reinterpret_cast<void*>(&cudaHostGetDevicePointer)},
    {"cudaStreamCreate", reinterpret_cast<void*>(&cudaStreamCreate)},
    {"cudaStreamCreateWithFlags", reinterpret_cast<void*>(&cudaStreamCreateWithFlags)},
    {"cudaStreamCreateWithPriority", reinterpret_cast<void*>(&cudaStreamCreateWithPriority)},
    {"cudaStreamGetFlags", reinterpret_cast<void*>(&cudaStreamGetFlags)},
    {"cudaStreamGetPriority", reinterpret_cast<void*>(&cudaStreamGetPriority)},
    {"cudaEventCreateWithFlags", reinterpret_cast<void*>(&cudaEventCreateWithFlags)},
    {"cudaEventRecord", reinterpret_cast<void*>(&cudaEventRecord)},
    {"cudaEventRecord_ptsz", reinterpret_cast<void*>(&cudaEventRecord_ptsz)},
    {"cudaEventRecordWithFlags", reinterpret_cast<void*>(&cudaEventRecordWithFlags)},
    {"cudaEventRecordWithFlags_ptsz", reinterpret_cast<void*>(&cudaEventRecordWithFlags_ptsz)},
    {"cudaEventQuery", reinterpret_cast<void*>(&cudaEventQuery)},
    {"cudaEventSynchronize", reinterpret_cast<void*>(&cudaEventSynchronize)},
    {"cudaEventDestroy", reinterpret_cast<void*>(&cudaEventDestroy)},
    {"cudaEventElapsedTime", reinterpret_cast<void*>(&cudaEventElapsedTime)},
    {"cudaStreamWaitEvent", reinterpret_cast<void*>(&cudaStreamWaitEvent)},
    {"cudaStreamWaitEvent_ptsz", reinterpret_cast<void*>(&cudaStreamWaitEvent_ptsz)},
    {"cuStreamQuery_ptsz", reinterpret_cast<void*>(&cuStreamQuery_ptsz)},
    {"cuStreamSynchronize", reinterpret_cast<void*>(&cuStreamSynchronize)},
    {"cuStreamSynchronize_ptsz", reinterpret_cast<void*>(&cuStreamSynchronize_ptsz)},
    {"cuStreamDestroy", reinterpret_cast<void*>(&cuStreamDestroy_v2)},
    {"cuStreamDestroy_v2", reinterpret_cast<void*>(&cuStreamDestroy_v2)},
    {"cuCtxSynchronize", reinterpret_cast<void*>(&cuCtxSynchronize)},
    {"cuGetProcAddress", reinterpret_cast<void*>(&cuGetProcAddress)},
    {"cuGetProcAddress_v2", reinterpret_cast<void*>(&cuGetProcAddress_v2)},
    {"cudaMalloc", reinterpret_cast<void*>(&cudaMalloc)},
    {"cudaMallocManaged", reinterpret_cast<void*>(&cudaMallocManaged)},
    {"cudaMallocPitch", reinterpret_cast<void*>(&cudaMallocPitch)},
    {"cudaMalloc3D", reinterpret_cast<void*>(&cudaMalloc3D)},
    {"cudaLaunchKernel", reinterpret_cast<void*>(&cudaLaunchKernel)},
    {"cudaLaunchKernel_ptsz", reinterpret_cast<void*>(&cudaLaunchKernel_ptsz)},
    {"cudaLaunchKernelExC", reinterpret_cast<void*>(&cudaLaunchKernelExC)},
    {"cudaLaunchKernelExC_ptsz", reinterpret_cast<void*>(&cudaLaunchKernelExC_ptsz)},
    {"cudaGraphLaunch", reinterpret_cast<void*>(&cudaGraphLaunch)},
    {"cudaGraphLaunch_ptsz", reinterpret_cast<void*>(&cudaGraphLaunch_ptsz)},
    {"cudaStreamBeginCapture", reinterpret_cast<void*>(&cudaStreamBeginCapture)},
    {"cudaStreamBeginCapture_ptsz", reinterpret_cast<void*>(&cudaStreamBeginCapture_ptsz)},
    {"cudaStreamEndCapture", reinterpret_cast<void*>(&cudaStreamEndCapture)},
    {"cudaStreamEndCapture_ptsz", reinterpret_cast<void*>(&cudaStreamEndCapture_ptsz)},
    {"cudaGraphInstantiateWithFlags", reinterpret_cast<void*>(&cudaGraphInstantiateWithFlags)},
#if CUDART_VERSION >= 12000
    {"cudaGraphInstantiateWithParams", reinterpret_cast<void*>(&cudaGraphInstantiateWithParams)},
    {"cudaGraphInstantiateWithParams_ptsz",
     reinterpret_cast<void*>(&cudaGraphInstantiateWithParams_ptsz)},
#endif
    {"cudaGraphExecDestroy", reinterpret_cast<void*>(&cudaGraphExecDestroy)},
    {"cudaGraphExecUpdate", reinterpret_cast<void*>(&cudaGraphExecUpdate)},
    {"cudaGraphExecChildGraphNodeSetParams",
     reinterpret_cast<void*>(&cudaGraphExecChildGraphNodeSetParams)},
    {"cudaGraphExecEventRecordNodeSetEvent",
     reinterpret_cast<void*>(&cudaGraphExecEventRecordNodeSetEvent)},
    {"cudaGraphExecEventWaitNodeSetEvent",
     reinterpret_cast<void*>(&cudaGraphExecEventWaitNodeSetEvent)},
    {"cudaGraphNodeSetEnabled", reinterpret_cast<void*>(&cudaGraphNodeSetEnabled)},
#if CUDART_VERSION >= 12020
    {"cudaGraphExecNodeSetParams", reinterpret_cast<void*>(&cudaGraphExecNodeSetParams)},
#endif
    {"__cudaLaunchKernel", reinterpret_cast<void*>(&__cudaLaunchKernel)},
    {"__cudaLaunchKernel_ptsz", reinterpret_cast<void*>(&__cudaLaunchKernel_ptsz)},
    {"cudaMallocAsync", reinterpret_cast<void*>(&cudaMallocAsync)},
    {"cudaMallocAsync_ptsz", reinterpret_cast<void*>(&cudaMallocAsync_ptsz)},
    {"cudaMallocFromPoolAsync", reinterpret_cast<void*>(&cudaMallocFromPoolAsync)},
    {"cudaMallocFromPoolAsync_ptsz", reinterpret_cast<void*>(&cudaMallocFromPoolAsync_ptsz)},
    {"cudaFree", reinterpret_cast<void*>(&cudaFree)},
    {"cudaIpcGetMemHandle", reinterpret_cast<void*>(&cudaIpcGetMemHandle)},
    {"cudaIpcOpenMemHandle", reinterpret_cast<void*>(&cudaIpcOpenMemHandle)},
    {"cudaIpcCloseMemHandle", reinterpret_cast<void*>(&cudaIpcCloseMemHandle)},
    {"cudaImportExternalMemory", reinterpret_cast<void*>(&cudaImportExternalMemory)},
    {"cudaExternalMemoryGetMappedBuffer",
     reinterpret_cast<void*>(&cudaExternalMemoryGetMappedBuffer)},
    {"cudaExternalMemoryGetMappedMipmappedArray",
     reinterpret_cast<void*>(&cudaExternalMemoryGetMappedMipmappedArray)},
    {"cudaDestroyExternalMemory", reinterpret_cast<void*>(&cudaDestroyExternalMemory)},
    {"cudaMallocArray", reinterpret_cast<void*>(&cudaMallocArray)},
    {"cudaMalloc3DArray", reinterpret_cast<void*>(&cudaMalloc3DArray)},
    {"cudaMallocMipmappedArray", reinterpret_cast<void*>(&cudaMallocMipmappedArray)},
    {"cudaFreeArray", reinterpret_cast<void*>(&cudaFreeArray)},
    {"cudaFreeMipmappedArray", reinterpret_cast<void*>(&cudaFreeMipmappedArray)},
    {"cudaGraphicsUnregisterResource", reinterpret_cast<void*>(&cudaGraphicsUnregisterResource)},
    {"cudaGraphicsResourceSetMapFlags", reinterpret_cast<void*>(&cudaGraphicsResourceSetMapFlags)},
    {"cudaGraphicsMapResources", reinterpret_cast<void*>(&cudaGraphicsMapResources)},
    {"cudaGraphicsUnmapResources", reinterpret_cast<void*>(&cudaGraphicsUnmapResources)},
    {"cudaGraphicsResourceGetMappedPointer",
     reinterpret_cast<void*>(&cudaGraphicsResourceGetMappedPointer)},
    {"cudaGraphicsSubResourceGetMappedArray",
     reinterpret_cast<void*>(&cudaGraphicsSubResourceGetMappedArray)},
    {"cudaGraphicsResourceGetMappedMipmappedArray",
     reinterpret_cast<void*>(&cudaGraphicsResourceGetMappedMipmappedArray)},
    {"cudaGraphAddMemAllocNode", reinterpret_cast<void*>(&cudaGraphAddMemAllocNode)},
    {"cudaFreeAsync", reinterpret_cast<void*>(&cudaFreeAsync)},
    {"cudaFreeAsync_ptsz", reinterpret_cast<void*>(&cudaFreeAsync_ptsz)},
    {"cudaMemGetInfo", reinterpret_cast<void*>(&cudaMemGetInfo)},
    {"cudaDeviceSynchronize", reinterpret_cast<void*>(&cudaDeviceSynchronize)},
    {"cudaDeviceReset", reinterpret_cast<void*>(&cudaDeviceReset)},
    {"cudaStreamSynchronize", reinterpret_cast<void*>(&cudaStreamSynchronize)},
    {"cudaStreamSynchronize_ptsz", reinterpret_cast<void*>(&cudaStreamSynchronize_ptsz)},
    {"cudaStreamQuery", reinterpret_cast<void*>(&cudaStreamQuery)},
    {"cudaStreamQuery_ptsz", reinterpret_cast<void*>(&cudaStreamQuery_ptsz)},
    {"cudaStreamDestroy", reinterpret_cast<void*>(&cudaStreamDestroy)},
    {"cudaDeviceGetDefaultMemPool", reinterpret_cast<void*>(&cudaDeviceGetDefaultMemPool)},
    {"cudaDeviceSetMemPool", reinterpret_cast<void*>(&cudaDeviceSetMemPool)},
    {"cudaDeviceGetMemPool", reinterpret_cast<void*>(&cudaDeviceGetMemPool)},
    {"cudaMemPoolTrimTo", reinterpret_cast<void*>(&cudaMemPoolTrimTo)},
    {"cudaMemPoolSetAttribute", reinterpret_cast<void*>(&cudaMemPoolSetAttribute)},
    {"cudaMemPoolGetAttribute", reinterpret_cast<void*>(&cudaMemPoolGetAttribute)},
    {"cudaMemPoolSetAccess", reinterpret_cast<void*>(&cudaMemPoolSetAccess)},
    {"cudaMemPoolGetAccess", reinterpret_cast<void*>(&cudaMemPoolGetAccess)},
    {"cudaMemPoolCreate", reinterpret_cast<void*>(&cudaMemPoolCreate)},
    {"cudaMemPoolDestroy", reinterpret_cast<void*>(&cudaMemPoolDestroy)},
    {"cudaMemGetDefaultMemPool", reinterpret_cast<void*>(&cudaMemGetDefaultMemPool)},
    {"cudaMemGetMemPool", reinterpret_cast<void*>(&cudaMemGetMemPool)},
    {"cudaMemSetMemPool", reinterpret_cast<void*>(&cudaMemSetMemPool)},
    {"cudaMemPoolExportToShareableHandle",
     reinterpret_cast<void*>(&cudaMemPoolExportToShareableHandle)},
    {"cudaMemPoolImportFromShareableHandle",
     reinterpret_cast<void*>(&cudaMemPoolImportFromShareableHandle)},
    {"cudaMemPoolExportPointer", reinterpret_cast<void*>(&cudaMemPoolExportPointer)},
    {"cudaMemPoolImportPointer", reinterpret_cast<void*>(&cudaMemPoolImportPointer)},
    {"nvmlInit", reinterpret_cast<void*>(&nvmlInit)},
    {"nvmlInit_v2", reinterpret_cast<void*>(&nvmlInit_v2)},
    {"nvmlInitWithFlags", reinterpret_cast<void*>(&nvmlInitWithFlags)},
    {"nvmlShutdown", reinterpret_cast<void*>(&nvmlShutdown)},
    {"nvmlDeviceGetCount", reinterpret_cast<void*>(&nvmlDeviceGetCount_v2)},
    {"nvmlDeviceGetCount_v2", reinterpret_cast<void*>(&nvmlDeviceGetCount_v2)},
    {"nvmlDeviceGetHandleByIndex", reinterpret_cast<void*>(&nvmlDeviceGetHandleByIndex_v2)},
    {"nvmlDeviceGetHandleByIndex_v2", reinterpret_cast<void*>(&nvmlDeviceGetHandleByIndex_v2)},
    {"nvmlDeviceGetIndex", reinterpret_cast<void*>(&nvmlDeviceGetIndex)},
    {"nvmlDeviceGetMemoryInfo", reinterpret_cast<void*>(&nvmlDeviceGetMemoryInfo)},
    {"nvmlDeviceGetMemoryInfo_v2", reinterpret_cast<void*>(&nvmlDeviceGetMemoryInfo_v2)},
});

}  // namespace

void* find_interceptor_symbol(const char* name, bool per_thread_default_stream) noexcept {
    if (name == nullptr) {
        return nullptr;
    }

    const std::string_view requested_name{name};
    if (per_thread_default_stream) {
        for (const InterceptorSymbol& symbol : kInterceptorSymbols) {
            if (symbol.name.size() == requested_name.size() + 5 &&
                symbol.name.starts_with(requested_name) &&
                (symbol.name.ends_with("_ptsz") || symbol.name.ends_with("_ptds"))) {
                return symbol.wrapper;
            }
        }
    }
    for (const InterceptorSymbol& symbol : kInterceptorSymbols) {
        if (symbol.name == requested_name) {
            return symbol.wrapper;
        }
    }
    return nullptr;
}

}  // namespace glimmer::interceptor
