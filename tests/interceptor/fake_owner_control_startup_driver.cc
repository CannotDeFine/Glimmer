// A startup-only SDK, not a GPU execution or numerical oracle. Keep original
// symbols locally bound, including the addresses returned by cuGetProcAddress.
#include <cuda.h>
#include <array>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <string_view>

#undef cuMemAlloc
#undef cuMemFree
#undef cuMemGetInfo
#undef cuMemcpyHtoDAsync
#undef cuMemcpyDtoHAsync
#undef cuMemcpyDtoDAsync
#undef cuCtxPushCurrent
#undef cuCtxPopCurrent
#undef cuStreamDestroy
#undef cuEventDestroy
#undef cuMemHostGetDevicePointer
#undef cuMemGetAddressRange
#undef cuGetProcAddress

namespace {
const auto kContext = reinterpret_cast<CUcontext>(0x4200);
thread_local CUcontext current = reinterpret_cast<CUcontext>(0x4200);
thread_local std::array<CUcontext, 4> context_stack{};
thread_local std::size_t context_depth{};
std::atomic<unsigned int> launches{};
std::atomic<unsigned int> last_grid{};
CUresult allocate(void** pointer, std::size_t bytes) {
    if (!pointer || !bytes) return CUDA_ERROR_INVALID_VALUE;
    *pointer = nullptr;
    return ::posix_memalign(pointer, 64, bytes) == 0 ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult copy(void* target, const void* source, std::size_t bytes) {
    if ((!target || !source) && bytes) return CUDA_ERROR_INVALID_VALUE;
    if (bytes) std::memcpy(target, source, bytes);
    return CUDA_SUCCESS;
}
}  // namespace
extern "C" unsigned int glimmer_owner_startup_launch_count() {
    return launches.load();
}
extern "C" unsigned int glimmer_owner_startup_last_grid() {
    return last_grid.load();
}
extern "C" CUresult CUDAAPI cuInit(unsigned int flags) {
    return flags == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* pointer, std::size_t bytes) {
    if (!pointer) return CUDA_ERROR_INVALID_VALUE;
    void* allocation{};
    const auto result = allocate(&allocation, bytes);
    *pointer = reinterpret_cast<CUdeviceptr>(allocation);
    return result;
}
extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr pointer) {
    std::free(std::bit_cast<void*>(pointer));
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (!free_bytes || !total_bytes) return CUDA_ERROR_INVALID_VALUE;
    *free_bytes = *total_bytes = 1ULL << 30;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetCurrent(CUcontext* context) {
    if (!context) return CUDA_ERROR_INVALID_VALUE;
    *context = current;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetDevice(CUdevice* device) {
    if (!device || !current) return CUDA_ERROR_INVALID_CONTEXT;
    *device = 0;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxGetApiVersion(CUcontext context, unsigned int* version) {
    if (context != kContext || !version) return CUDA_ERROR_INVALID_CONTEXT;
    *version = 3020;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxPushCurrent_v2(CUcontext context) {
    if (context != kContext) return CUDA_ERROR_INVALID_CONTEXT;
    if (context_depth == context_stack.size()) return CUDA_ERROR_OUT_OF_MEMORY;
    context_stack[context_depth++] = current;
    current = context;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxPopCurrent_v2(CUcontext* context) {
    if (!context || !context_depth) return CUDA_ERROR_INVALID_CONTEXT;
    *context = current;
    current = context_stack[--context_depth];
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuCtxSynchronize() {
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuLaunchKernel(CUfunction function, unsigned int grid_x, unsigned int,
                                           unsigned int, unsigned int, unsigned int, unsigned int,
                                           unsigned int, CUstream, void**, void**) {
    if (!function || !current) return CUDA_ERROR_INVALID_VALUE;
    last_grid.store(grid_x);
    ++launches;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemHostAlloc(void** pointer, std::size_t bytes, unsigned int) {
    return allocate(pointer, bytes);
}
extern "C" CUresult CUDAAPI cuMemFreeHost(void* pointer) {
    std::free(pointer);
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* device, void* host,
                                                         unsigned int) {
    if (!device || !host) return CUDA_ERROR_INVALID_VALUE;
    *device = reinterpret_cast<CUdeviceptr>(host);
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr target, const void* source,
                                                 std::size_t bytes, CUstream) {
    return copy(std::bit_cast<void*>(target), source, bytes);
}
extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* target, CUdeviceptr source,
                                                 std::size_t bytes, CUstream) {
    return copy(target, std::bit_cast<void*>(source), bytes);
}
extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr target, CUdeviceptr source,
                                                 std::size_t bytes, CUstream) {
    return copy(std::bit_cast<void*>(target), std::bit_cast<void*>(source), bytes);
}
extern "C" CUresult CUDAAPI cuMemFreeAsync(CUdeviceptr pointer, CUstream) {
    return cuMemFree_v2(pointer);
}
extern "C" CUresult CUDAAPI cuEventCreate(CUevent* event, unsigned int) {
    if (!event) return CUDA_ERROR_INVALID_VALUE;
    void* handle{};
    const auto result = allocate(&handle, sizeof(unsigned int));
    *event = reinterpret_cast<CUevent>(handle);
    return result;
}
extern "C" CUresult CUDAAPI cuEventRecord(CUevent event, CUstream) {
    if (!event) return CUDA_ERROR_INVALID_HANDLE;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventQuery(CUevent event) {
    return event ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}
extern "C" CUresult CUDAAPI cuEventDestroy_v2(CUevent event) {
    std::free(reinterpret_cast<void*>(event));
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuEventSynchronize(CUevent event) {
    return cuEventQuery(event);
}
extern "C" CUresult CUDAAPI cuStreamCreate(CUstream* stream, unsigned int) {
    if (!stream) return CUDA_ERROR_INVALID_VALUE;
    void* handle{};
    const auto result = allocate(&handle, sizeof(unsigned int));
    *stream = reinterpret_cast<CUstream>(handle);
    return result;
}
extern "C" CUresult CUDAAPI cuStreamDestroy_v2(CUstream stream) {
    std::free(reinterpret_cast<void*>(stream));
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetFlags(CUstream, unsigned int* flags) {
    if (!flags) return CUDA_ERROR_INVALID_VALUE;
    *flags = CU_STREAM_NON_BLOCKING;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamGetPriority(CUstream, int* priority) {
    if (!priority) return CUDA_ERROR_INVALID_VALUE;
    *priority = 0;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamSynchronize(CUstream) {
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuStreamWaitEvent(CUstream, CUevent event, unsigned int) {
    return cuEventQuery(event);
}
extern "C" CUresult CUDAAPI cuStreamIsCapturing(CUstream, CUstreamCaptureStatus* status) {
    if (!status) return CUDA_ERROR_INVALID_VALUE;
    *status = CU_STREAM_CAPTURE_STATUS_NONE;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuPointerGetAttribute(void*, CUpointer_attribute, CUdeviceptr) {
    return CUDA_ERROR_INVALID_VALUE;
}
extern "C" CUresult CUDAAPI cuMemGetAddressRange_v2(CUdeviceptr*, std::size_t*, CUdeviceptr) {
    return CUDA_ERROR_INVALID_VALUE;
}
extern "C" CUresult CUDAAPI cuMemHostGetFlags(unsigned int*, void*) {
    return CUDA_ERROR_INVALID_VALUE;
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* symbol, void** pointer, int,
                                             cuuint64_t flags) {
    if (!symbol || !pointer) return CUDA_ERROR_INVALID_VALUE;
    *pointer = nullptr;
    const std::string_view name{symbol};
    if (flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) return CUDA_ERROR_NOT_SUPPORTED;
#define GLIMMER_STARTUP_SYMBOL(base, function)           \
    if (name == (base)) {                                \
        *pointer = reinterpret_cast<void*>(&(function)); \
        return CUDA_SUCCESS;                             \
    }
    GLIMMER_STARTUP_SYMBOL("cuCtxGetCurrent", cuCtxGetCurrent)
    GLIMMER_STARTUP_SYMBOL("cuCtxPushCurrent", cuCtxPushCurrent_v2)
    GLIMMER_STARTUP_SYMBOL("cuCtxPopCurrent", cuCtxPopCurrent_v2)
    GLIMMER_STARTUP_SYMBOL("cuLaunchKernel", cuLaunchKernel)
    GLIMMER_STARTUP_SYMBOL("cuMemcpyHtoDAsync", cuMemcpyHtoDAsync_v2)
    GLIMMER_STARTUP_SYMBOL("cuMemcpyDtoHAsync", cuMemcpyDtoHAsync_v2)
    GLIMMER_STARTUP_SYMBOL("cuMemcpyDtoDAsync", cuMemcpyDtoDAsync_v2)
    GLIMMER_STARTUP_SYMBOL("cuMemFreeAsync", cuMemFreeAsync)
    GLIMMER_STARTUP_SYMBOL("cuEventCreate", cuEventCreate)
    GLIMMER_STARTUP_SYMBOL("cuEventRecord", cuEventRecord)
    GLIMMER_STARTUP_SYMBOL("cuEventQuery", cuEventQuery)
    GLIMMER_STARTUP_SYMBOL("cuEventDestroy", cuEventDestroy_v2)
    GLIMMER_STARTUP_SYMBOL("cuEventSynchronize", cuEventSynchronize)
    GLIMMER_STARTUP_SYMBOL("cuStreamSynchronize", cuStreamSynchronize)
    GLIMMER_STARTUP_SYMBOL("cuStreamWaitEvent", cuStreamWaitEvent)
    GLIMMER_STARTUP_SYMBOL("cuStreamCreate", cuStreamCreate)
    GLIMMER_STARTUP_SYMBOL("cuStreamDestroy", cuStreamDestroy_v2)
    GLIMMER_STARTUP_SYMBOL("cuStreamGetFlags", cuStreamGetFlags)
    GLIMMER_STARTUP_SYMBOL("cuStreamGetPriority", cuStreamGetPriority)
    GLIMMER_STARTUP_SYMBOL("cuMemHostAlloc", cuMemHostAlloc)
    GLIMMER_STARTUP_SYMBOL("cuMemFreeHost", cuMemFreeHost)
    GLIMMER_STARTUP_SYMBOL("cuMemHostGetDevicePointer", cuMemHostGetDevicePointer_v2)
    GLIMMER_STARTUP_SYMBOL("cuPointerGetAttribute", cuPointerGetAttribute)
    GLIMMER_STARTUP_SYMBOL("cuMemGetAddressRange", cuMemGetAddressRange_v2)
    GLIMMER_STARTUP_SYMBOL("cuMemHostGetFlags", cuMemHostGetFlags)
#undef GLIMMER_STARTUP_SYMBOL
    return CUDA_ERROR_NOT_FOUND;
}
