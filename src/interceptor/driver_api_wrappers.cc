#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_api_interceptor.h"

#include <cuda.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
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
#ifdef cuCtxDestroy
#undef cuCtxDestroy
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

namespace {

template <typename Function>
CUresult guard_cuda_boundary(Function&& function) noexcept {
    try {
        return function();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

}  // namespace

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return guard_cuda_boundary([device_pointer, memory_bytes] {
        return glimmer::interceptor::intercept_mem_alloc(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemAlloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return cuMemAlloc_v2(device_pointer, memory_bytes);
}

extern "C" CUresult CUDAAPI cuInit(unsigned int flags) {
    return guard_cuda_boundary([flags] { return glimmer::interceptor::intercept_init(flags); });
}

extern "C" CUresult CUDAAPI cuMemAllocManaged(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              unsigned int flags) {
    return guard_cuda_boundary([device_pointer, memory_bytes, flags] {
        return glimmer::interceptor::intercept_mem_alloc_managed(device_pointer, memory_bytes,
                                                                 flags);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocPitch_v2(CUdeviceptr* device_pointer, std::size_t* pitch,
                                               std::size_t width_bytes, std::size_t height,
                                               unsigned int element_size_bytes) {
    return guard_cuda_boundary([device_pointer, pitch, width_bytes, height, element_size_bytes] {
        return glimmer::interceptor::intercept_mem_alloc_pitch(device_pointer, pitch, width_bytes,
                                                               height, element_size_bytes);
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

extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr device_pointer) {
    return guard_cuda_boundary(
        [device_pointer] { return glimmer::interceptor::intercept_mem_free(device_pointer); });
}

extern "C" CUresult CUDAAPI cuMemFree(CUdeviceptr device_pointer) {
    return cuMemFree_v2(device_pointer);
}

extern "C" CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr device_pointer, CUstream stream) {
    return guard_cuda_boundary([device_pointer, stream] {
        return glimmer::interceptor::intercept_mem_free_async(device_pointer, stream, false);
    });
}

extern "C" CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr device_pointer, CUstream stream) {
    return guard_cuda_boundary([device_pointer, stream] {
        return glimmer::interceptor::intercept_mem_free_async(device_pointer, stream, true);
    });
}

extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
    return guard_cuda_boundary([free_bytes, total_bytes] {
        return glimmer::interceptor::intercept_mem_get_info(free_bytes, total_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    return cuMemGetInfo_v2(free_bytes, total_bytes);
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem_v2(std::size_t* total_bytes, CUdevice device) {
    return guard_cuda_boundary([total_bytes, device] {
        return glimmer::interceptor::intercept_device_total_mem(total_bytes, device);
    });
}

extern "C" CUresult CUDAAPI cuDeviceTotalMem(std::size_t* total_bytes, CUdevice device) {
    return cuDeviceTotalMem_v2(total_bytes, device);
}

extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
    return guard_cuda_boundary(
        [context] { return glimmer::interceptor::intercept_context_get_current(context); });
}

extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    return guard_cuda_boundary(
        [device] { return glimmer::interceptor::intercept_context_get_device(device); });
}

extern "C" CUresult CUDAAPI cuCtxDestroy_v2(CUcontext context) {
    return guard_cuda_boundary(
        [context] { return glimmer::interceptor::intercept_context_destroy(context); });
}

extern "C" CUresult CUDAAPI cuCtxDestroy(CUcontext context) {
    return cuCtxDestroy_v2(context);
}

extern "C" CUresult CUDAAPI cuStreamGetDevice(CUstream stream, CUdevice* device) {
    return guard_cuda_boundary([stream, device] {
        return glimmer::interceptor::intercept_stream_get_device(stream, device, false);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetDevice_ptsz(CUstream stream, CUdevice* device) {
    return guard_cuda_boundary([stream, device] {
        return glimmer::interceptor::intercept_stream_get_device(stream, device, true);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetCtx(CUstream stream, CUcontext* context) {
    return guard_cuda_boundary([stream, context] {
        return glimmer::interceptor::intercept_stream_get_context(stream, context, false);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream stream, CUcontext* context) {
    return guard_cuda_boundary([stream, context] {
        return glimmer::interceptor::intercept_stream_get_context(stream, context, true);
    });
}

extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_query(stream, false); });
}

extern "C" CUresult CUDAAPI cuStreamQuery_ptsz(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_query(stream, true); });
}

extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_synchronize(stream, false); });
}

extern "C" CUresult CUDAAPI cuStreamSynchronize_ptsz(CUstream stream) {
    return guard_cuda_boundary(
        [stream] { return glimmer::interceptor::intercept_stream_synchronize(stream, true); });
}

extern "C" CUresult CUDAAPI cuCtxSynchronize() {
    return guard_cuda_boundary(
        [] { return glimmer::interceptor::intercept_context_synchronize(); });
}

extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags) {
    return guard_cuda_boundary([symbol, function_pointer, cuda_version, flags] {
        return glimmer::interceptor::intercept_get_proc_address(symbol, function_pointer,
                                                                cuda_version, flags);
    });
}

extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** function_pointer,
                                                int cuda_version, cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status) {
    return guard_cuda_boundary([symbol, function_pointer, cuda_version, flags, symbol_status] {
        return glimmer::interceptor::intercept_get_proc_address_v2(
            symbol, function_pointer, cuda_version, flags, symbol_status);
    });
}
