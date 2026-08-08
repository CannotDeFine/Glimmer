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
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags);
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* symbol, void** function_pointer,
                                                int cuda_version, cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* symbol_status);
extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes);
extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer);
extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes);

namespace glimmer::interceptor {

namespace {

struct InterceptorSymbol {
    std::string_view name;
    void* wrapper;
};

const std::array<InterceptorSymbol, 21> kInterceptorSymbols{{
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
    {"cuGetProcAddress", reinterpret_cast<void*>(&cuGetProcAddress)},
    {"cuGetProcAddress_v2", reinterpret_cast<void*>(&cuGetProcAddress_v2)},
    {"cudaMalloc", reinterpret_cast<void*>(&cudaMalloc)},
    {"cudaFree", reinterpret_cast<void*>(&cudaFree)},
    {"cudaMemGetInfo", reinterpret_cast<void*>(&cudaMemGetInfo)},
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
