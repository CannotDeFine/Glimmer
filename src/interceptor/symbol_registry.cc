#include "internal/symbol_registry.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuCtxDestroy
#undef cuCtxDestroy
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

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* device_pointer, std::size_t memory_bytes);
extern "C" CUresult CUDAAPI cuInit(unsigned int flags);
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
extern "C" CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr device_pointer, CUstream stream);
extern "C" CUresult CUDAAPI cuMemFreeAsync_ptsz(CUdeviceptr device_pointer, CUstream stream);
extern "C" CUresult CUDAAPI cuStreamGetDevice(CUstream stream, CUdevice* device);
extern "C" CUresult CUDAAPI cuStreamGetDevice_ptsz(CUstream stream, CUdevice* device);
extern "C" CUresult CUDAAPI cuStreamGetCtx(CUstream stream, CUcontext* context);
extern "C" CUresult CUDAAPI cuStreamGetCtx_ptsz(CUstream stream, CUcontext* context);
extern "C" CUresult CUDAAPI cuStreamQuery(CUstream stream);
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
extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** device_pointer,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer);
extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* device_pointer, cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes);
extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize();
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream);
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream);

namespace glimmer::interceptor {

namespace {

struct InterceptorSymbol {
    std::string_view name;
    void* wrapper;
};

const std::array<InterceptorSymbol, 48> kInterceptorSymbols{{
    {"cuInit", reinterpret_cast<void*>(&cuInit)},
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
    {"cuCtxDestroy", reinterpret_cast<void*>(&cuCtxDestroy_v2)},
    {"cuCtxDestroy_v2", reinterpret_cast<void*>(&cuCtxDestroy_v2)},
    {"cuMemAllocAsync", reinterpret_cast<void*>(&cuMemAllocAsync)},
    {"cuMemAllocAsync_ptsz", reinterpret_cast<void*>(&cuMemAllocAsync_ptsz)},
    {"cuMemAllocFromPoolAsync", reinterpret_cast<void*>(&cuMemAllocFromPoolAsync)},
    {"cuMemAllocFromPoolAsync_ptsz", reinterpret_cast<void*>(&cuMemAllocFromPoolAsync_ptsz)},
    {"cuMemFreeAsync", reinterpret_cast<void*>(&cuMemFreeAsync)},
    {"cuMemFreeAsync_ptsz", reinterpret_cast<void*>(&cuMemFreeAsync_ptsz)},
    {"cuStreamGetDevice", reinterpret_cast<void*>(&cuStreamGetDevice)},
    {"cuStreamGetDevice_ptsz", reinterpret_cast<void*>(&cuStreamGetDevice_ptsz)},
    {"cuStreamGetCtx", reinterpret_cast<void*>(&cuStreamGetCtx)},
    {"cuStreamGetCtx_ptsz", reinterpret_cast<void*>(&cuStreamGetCtx_ptsz)},
    {"cuStreamQuery", reinterpret_cast<void*>(&cuStreamQuery)},
    {"cuStreamQuery_ptsz", reinterpret_cast<void*>(&cuStreamQuery_ptsz)},
    {"cuStreamSynchronize", reinterpret_cast<void*>(&cuStreamSynchronize)},
    {"cuStreamSynchronize_ptsz", reinterpret_cast<void*>(&cuStreamSynchronize_ptsz)},
    {"cuStreamDestroy", reinterpret_cast<void*>(&cuStreamDestroy_v2)},
    {"cuStreamDestroy_v2", reinterpret_cast<void*>(&cuStreamDestroy_v2)},
    {"cuCtxSynchronize", reinterpret_cast<void*>(&cuCtxSynchronize)},
    {"cuGetProcAddress", reinterpret_cast<void*>(&cuGetProcAddress)},
    {"cuGetProcAddress_v2", reinterpret_cast<void*>(&cuGetProcAddress_v2)},
    {"cudaMalloc", reinterpret_cast<void*>(&cudaMalloc)},
    {"cudaMallocAsync", reinterpret_cast<void*>(&cudaMallocAsync)},
    {"cudaMallocAsync_ptsz", reinterpret_cast<void*>(&cudaMallocAsync_ptsz)},
    {"cudaFree", reinterpret_cast<void*>(&cudaFree)},
    {"cudaFreeAsync", reinterpret_cast<void*>(&cudaFreeAsync)},
    {"cudaFreeAsync_ptsz", reinterpret_cast<void*>(&cudaFreeAsync_ptsz)},
    {"cudaMemGetInfo", reinterpret_cast<void*>(&cudaMemGetInfo)},
    {"cudaDeviceSynchronize", reinterpret_cast<void*>(&cudaDeviceSynchronize)},
    {"cudaStreamSynchronize", reinterpret_cast<void*>(&cudaStreamSynchronize)},
    {"cudaStreamSynchronize_ptsz", reinterpret_cast<void*>(&cudaStreamSynchronize_ptsz)},
    {"cudaStreamQuery", reinterpret_cast<void*>(&cudaStreamQuery)},
    {"cudaStreamQuery_ptsz", reinterpret_cast<void*>(&cudaStreamQuery_ptsz)},
    {"cudaStreamDestroy", reinterpret_cast<void*>(&cudaStreamDestroy)},
}};

}  // namespace

void* find_interceptor_symbol(const char* name) noexcept {
    if (name == nullptr) {
        return nullptr;
    }

    const std::string_view requested_name{name};
    for (const InterceptorSymbol& symbol : kInterceptorSymbols) {
        if (symbol.name == requested_name) {
            return symbol.wrapper;
        }
    }
    return nullptr;
}

}  // namespace glimmer::interceptor
