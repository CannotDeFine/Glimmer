#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstring>
#include <string_view>
#include <dlfcn.h>

extern "C" CUresult glimmer_execution_native_driver_entry(const char*, void**, int, cuuint64_t,
                                                          CUdriverProcAddressQueryResult*);
namespace {
std::atomic<unsigned int> calls[6];
cudaError_t lookup(const char* name, void** output, unsigned int version, unsigned long long flags,
                   cudaDriverEntryPointQueryResult* status) {
    if (!name || !output) return cudaErrorInvalidValue;
    if (std::string_view(name) == "runtime_sdk_error") {
        *output = reinterpret_cast<void*>(&cudaMalloc);
        return cudaErrorInsufficientDriver;  // Status output deliberately untouched.
    }
    const bool malformed = std::string_view(name) == "runtime_malformed";
    CUdriverProcAddressQueryResult selected{};
    const auto result = glimmer_execution_native_driver_entry(malformed ? "cuCtxGetCurrent" : name,
                                                              output, static_cast<int>(version),
                                                              flags, status ? &selected : nullptr);
    if (status) {
        const int raw =
            malformed ? cudaDriverEntryPointVersionNotSufficent : static_cast<int>(selected);
        std::memcpy(status, &raw, sizeof(raw));
    }
    return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorCudartUnloading;
}
}  // namespace
extern "C" {
unsigned int glimmer_execution_runtime_lookup_count(unsigned int index) {
    return index < 6 ? calls[index].load(std::memory_order_relaxed) : 0;
}
cudaError_t CUDARTAPI cudaGetDeviceCount(int*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaDeviceGetLimit(std::size_t*, cudaLimit) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp*, int) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaDeviceGetAttribute(int*, cudaDeviceAttr, int) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaDeviceGetP2PAttribute(int*, cudaDeviceP2PAttr, int, int) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaDeviceCanAccessPeer(int*, int, int) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaDeviceGetStreamPriorityRange(int*, int*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t*, std::size_t*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaDriverGetVersion(int*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaStreamCreateWithPriority(cudaStream_t*, unsigned int, int) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaStreamGetPriority(cudaStream_t, int*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaStreamGetFlags(cudaStream_t, unsigned int*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaStreamGetCaptureInfo(cudaStream_t, cudaStreamCaptureStatus*,
                                               unsigned long long*, cudaGraph_t*,
                                               const cudaGraphNode_t**, std::size_t*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaThreadExchangeStreamCaptureMode(cudaStreamCaptureMode*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaMalloc(void**, std::size_t) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaGetSymbolAddress(void**, const void*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaGetSymbolSize(std::size_t*, const void*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaMemcpyToSymbol(const void*, const void*, std::size_t, std::size_t,
                                         cudaMemcpyKind) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaMemcpyFromSymbol(void*, const void*, std::size_t, std::size_t,
                                           cudaMemcpyKind) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaMemcpyToSymbolAsync(const void*, const void*, std::size_t, std::size_t,
                                              cudaMemcpyKind, cudaStream_t) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaMemcpyFromSymbolAsync(void*, const void*, std::size_t, std::size_t,
                                                cudaMemcpyKind, cudaStream_t) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaFree(void*) {
    ++calls[2];
    return cudaErrorNotSupported;
}
cudaError_t CUDARTAPI cudaLookupUnsupported() {
    ++calls[2];
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* output) {
    if (!output) return cudaErrorInvalidValue;
    *output = CUDART_VERSION;
    return cudaSuccess;
}
cudaError_t CUDARTAPI cudaGetDriverEntryPoint(const char* name, void** output,
                                              unsigned long long flags,
                                              cudaDriverEntryPointQueryResult* status) {
    ++calls[0];
    return lookup(name, output, CUDART_VERSION, flags, status);
}
cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion(const char* name, void** output,
                                                       unsigned int version,
                                                       unsigned long long flags,
                                                       cudaDriverEntryPointQueryResult* status) {
    ++calls[1];
    return lookup(name, output, version, flags, status);
}
cudaError_t CUDARTAPI cudaGetDriverEntryPoint_ptsz(const char* name, void** output,
                                                   unsigned long long flags,
                                                   cudaDriverEntryPointQueryResult* status) {
    ++calls[3];
    return lookup(name, output, CUDART_VERSION,
                  flags ? flags : static_cast<unsigned long long>(cudaEnablePerThreadDefaultStream),
                  status);
}
cudaError_t CUDARTAPI cudaGetDriverEntryPointByVersion_ptsz(
    const char* name, void** output, unsigned int version, unsigned long long flags,
    cudaDriverEntryPointQueryResult* status) {
    ++calls[4];
    return lookup(name, output, version,
                  flags ? flags : static_cast<unsigned long long>(cudaEnablePerThreadDefaultStream),
                  status);
}
cudaError_t CUDARTAPI cudaGetExportTable(const void**, const cudaUUID_t*) {
    ++calls[5];
    return cudaSuccess;
}
void* glimmer_execution_runtime_internal_lookup(void* handle, const char* name) {
    return ::dlsym(handle, name);
}
}
