#include "internal/driver_copy_wrappers.h"
#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/runtime_api_bridge.h"

namespace {
template <typename Operation>
CUresult guard_copy(Operation&& operation) noexcept {
    try {
        const glimmer::interceptor::CooperativeCallScope scope;
        return operation();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}
template <typename Function, typename... Arguments>
CUresult forward_metadata(Function function, Arguments... arguments) noexcept {
    return guard_copy([&] {
        if (function == nullptr) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        const glimmer::interceptor::NativeDriverCallScope scope;
        return function(arguments...);
    });
}
template <typename Function, typename... Arguments>
CUresult forward_native(Function function, Arguments... arguments) noexcept {
    return guard_copy([&] {
        return glimmer::interceptor::intercept_cooperative_native_boundary(
            [&] { return forward_metadata(function, arguments...); });
    });
}
}  // namespace

extern "C" CUresult CUDAAPI cuMemcpy(CUdeviceptr destination, CUdeviceptr source,
                                     std::size_t memory_bytes) {
    return guard_copy([&] {
        static const auto native = reinterpret_cast<decltype(&cuMemcpy)>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpy", false));
        return forward_native(native, destination, source, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t memory_bytes) {
    return guard_copy([&] {
        static const auto native = reinterpret_cast<decltype(&cuMemcpy)>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpy", true));
        return forward_native(native, destination, source, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyAsync(CUdeviceptr destination, CUdeviceptr source,
                                          std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        static const auto native = reinterpret_cast<decltype(&cuMemcpyAsync)>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyAsync", false));
        return glimmer::interceptor::intercept_driver_uva_copy(
            destination, source, memory_bytes, stream, false,
            [&] { return forward_metadata(native, destination, source, memory_bytes, stream); });
    });
}

extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr destination, CUdeviceptr source,
                                               std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        static const auto native = reinterpret_cast<decltype(&cuMemcpyAsync)>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyAsync", true));
        return glimmer::interceptor::intercept_driver_uva_copy(
            destination, source, memory_bytes, stream, true,
            [&] { return forward_metadata(native, destination, source, memory_bytes, stream); });
    });
}

extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr destination, const void* host,
                                                 std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        return glimmer::interceptor::intercept_driver_copy(destination, 0, const_cast<void*>(host),
                                                           memory_bytes, stream, 0, false);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyHtoDAsync_v2_ptsz(CUdeviceptr destination, const void* host,
                                                      std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        return glimmer::interceptor::intercept_driver_copy(destination, 0, const_cast<void*>(host),
                                                           memory_bytes, stream, 0, true);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr destination, const void* host,
                                            std::size_t memory_bytes) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, const void*, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyHtoD", false));
        return forward_native(native, destination, host, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyHtoD_v2_ptds(CUdeviceptr destination, const void* host,
                                                 std::size_t memory_bytes) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, const void*, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyHtoD", true));
        return forward_native(native, destination, host, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void* host, CUdeviceptr source,
                                                 std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        return glimmer::interceptor::intercept_driver_copy(0, source, host, memory_bytes, stream, 1,
                                                           false);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoHAsync_v2_ptsz(void* host, CUdeviceptr source,
                                                      std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        return glimmer::interceptor::intercept_driver_copy(0, source, host, memory_bytes, stream, 1,
                                                           true);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2(void* host, CUdeviceptr source,
                                            std::size_t memory_bytes) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(void*, CUdeviceptr, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyDtoH", false));
        return forward_native(native, host, source, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoH_v2_ptds(void* host, CUdeviceptr source,
                                                 std::size_t memory_bytes) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(void*, CUdeviceptr, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyDtoH", true));
        return forward_native(native, host, source, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        return glimmer::interceptor::intercept_driver_copy(destination, source, nullptr,
                                                           memory_bytes, stream, 2, false);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoDAsync_v2_ptsz(CUdeviceptr destination, CUdeviceptr source,
                                                      std::size_t memory_bytes, CUstream stream) {
    return guard_copy([&] {
        return glimmer::interceptor::intercept_driver_copy(destination, source, nullptr,
                                                           memory_bytes, stream, 2, true);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2(CUdeviceptr destination, CUdeviceptr source,
                                            std::size_t memory_bytes) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, CUdeviceptr, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyDtoD", false));
        return forward_native(native, destination, source, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemcpyDtoD_v2_ptds(CUdeviceptr destination, CUdeviceptr source,
                                                 std::size_t memory_bytes) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, CUdeviceptr, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemcpyDtoD", true));
        return forward_native(native, destination, source, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemHostAlloc(void** pointer, std::size_t memory_bytes,
                                           unsigned int flags) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(void**, std::size_t, unsigned int);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemHostAlloc"));
        return forward_native(native, pointer, memory_bytes, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemAllocHost_v2(void** pointer, std::size_t memory_bytes) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(void**, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemAllocHost"));
        return forward_native(native, pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemFreeHost(void* pointer) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(void*);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemFreeHost"));
        return forward_native(native, pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemHostRegister_v2(void* pointer, std::size_t memory_bytes,
                                                 unsigned int flags) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(void*, std::size_t, unsigned int);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemHostRegister"));
        return forward_native(native, pointer, memory_bytes, flags);
    });
}

extern "C" CUresult CUDAAPI cuMemHostUnregister(void* pointer) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(void*);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemHostUnregister"));
        return forward_native(native, pointer);
    });
}

extern "C" CUresult CUDAAPI cuMemHostGetDevicePointer_v2(CUdeviceptr* device_pointer, void* pointer,
                                                         unsigned int flags) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr*, void*, unsigned int);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemHostGetDevicePointer"));
        return forward_metadata(native, device_pointer, pointer, flags);
    });
}

extern "C" CUresult CUDAAPI cuStreamCreate(CUstream* stream, unsigned int flags) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream*, unsigned int);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuStreamCreate"));
        return glimmer::interceptor::intercept_driver_stream_create(
            [&] { return forward_metadata(native, stream, flags); });
    });
}

extern "C" CUresult CUDAAPI cuStreamCreateWithPriority(CUstream* stream, unsigned int flags,
                                                       int priority) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream*, unsigned int, int);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuStreamCreateWithPriority"));
        return glimmer::interceptor::intercept_driver_stream_create(
            [&] { return forward_metadata(native, stream, flags, priority); });
    });
}

extern "C" CUresult CUDAAPI cuStreamGetFlags(CUstream stream, unsigned int* flags) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream, unsigned int*);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuStreamGetFlags"));
        return forward_metadata(native, stream, flags);
    });
}

extern "C" CUresult CUDAAPI cuStreamGetPriority(CUstream stream, int* priority) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream, int*);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuStreamGetPriority"));
        return forward_metadata(native, stream, priority);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD8_v2(CUdeviceptr pointer, unsigned char value,
                                          std::size_t count) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned char, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD8", false));
        return forward_native(native, pointer, value, count);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD8_v2_ptds(CUdeviceptr pointer, unsigned char value,
                                               std::size_t count) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned char, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD8", true));
        return forward_native(native, pointer, value, count);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD8Async(CUdeviceptr pointer, unsigned char value,
                                            std::size_t count, CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned char, std::size_t, CUstream);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD8Async", false));
        return forward_native(native, pointer, value, count, stream);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD8Async_ptsz(CUdeviceptr pointer, unsigned char value,
                                                 std::size_t count, CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned char, std::size_t, CUstream);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD8Async", true));
        return forward_native(native, pointer, value, count, stream);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD16_v2(CUdeviceptr pointer, unsigned short value,
                                           std::size_t count) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned short, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD16", false));
        return forward_native(native, pointer, value, count);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD16_v2_ptds(CUdeviceptr pointer, unsigned short value,
                                                std::size_t count) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned short, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD16", true));
        return forward_native(native, pointer, value, count);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD16Async(CUdeviceptr pointer, unsigned short value,
                                             std::size_t count, CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned short, std::size_t, CUstream);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD16Async", false));
        return forward_native(native, pointer, value, count, stream);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD16Async_ptsz(CUdeviceptr pointer, unsigned short value,
                                                  std::size_t count, CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned short, std::size_t, CUstream);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD16Async", true));
        return forward_native(native, pointer, value, count, stream);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD32_v2(CUdeviceptr pointer, unsigned int value,
                                           std::size_t count) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned int, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD32", false));
        return forward_native(native, pointer, value, count);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD32_v2_ptds(CUdeviceptr pointer, unsigned int value,
                                                std::size_t count) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned int, std::size_t);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD32", true));
        return forward_native(native, pointer, value, count);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD32Async(CUdeviceptr pointer, unsigned int value,
                                             std::size_t count, CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned int, std::size_t, CUstream);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD32Async", false));
        return forward_native(native, pointer, value, count, stream);
    });
}

extern "C" CUresult CUDAAPI cuMemsetD32Async_ptsz(CUdeviceptr pointer, unsigned int value,
                                                  std::size_t count, CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUdeviceptr, unsigned int, std::size_t, CUstream);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuMemsetD32Async", true));
        return forward_native(native, pointer, value, count, stream);
    });
}

extern "C" CUresult CUDAAPI cuStreamBeginCapture(CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream);
        static const auto native =
            reinterpret_cast<Function>(glimmer::interceptor::resolve_native_driver_function(
                "cuStreamBeginCapture", false, 10000));
        if (const auto dispatched = glimmer::interceptor::dispatch_runtime_capture_origin(
                stream, false, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                [&] { return forward_metadata(native, stream); })) {
            return *dispatched;
        }
        return glimmer::interceptor::intercept_driver_capture(
            stream, false, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
            [&] { return forward_metadata(native, stream); });
    });
}

extern "C" CUresult CUDAAPI cuStreamBeginCapture_ptsz(CUstream stream) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream);
        static const auto native =
            reinterpret_cast<Function>(glimmer::interceptor::resolve_native_driver_function(
                "cuStreamBeginCapture", true, 10000));
        if (const auto dispatched = glimmer::interceptor::dispatch_runtime_capture_origin(
                stream, true, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                [&] { return forward_metadata(native, stream); })) {
            return *dispatched;
        }
        return glimmer::interceptor::intercept_driver_capture(
            stream, true, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
            [&] { return forward_metadata(native, stream); });
    });
}

extern "C" CUresult CUDAAPI cuStreamBeginCapture_v2(CUstream stream, CUstreamCaptureMode mode) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream, CUstreamCaptureMode);
        static const auto native =
            reinterpret_cast<Function>(glimmer::interceptor::resolve_native_driver_function(
                "cuStreamBeginCapture", false, 10010));
        return guard_copy([&] {
            if (const auto dispatched = glimmer::interceptor::dispatch_runtime_capture_origin(
                    stream, false, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                    [&] { return forward_metadata(native, stream, mode); })) {
                return *dispatched;
            }
            return glimmer::interceptor::intercept_driver_capture(
                stream, false, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                [&] { return forward_metadata(native, stream, mode); });
        });
    });
}

extern "C" CUresult CUDAAPI cuStreamBeginCapture_v2_ptsz(CUstream stream,
                                                         CUstreamCaptureMode mode) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream, CUstreamCaptureMode);
        static const auto native =
            reinterpret_cast<Function>(glimmer::interceptor::resolve_native_driver_function(
                "cuStreamBeginCapture", true, 10010));
        return guard_copy([&] {
            if (const auto dispatched = glimmer::interceptor::dispatch_runtime_capture_origin(
                    stream, true, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                    [&] { return forward_metadata(native, stream, mode); })) {
                return *dispatched;
            }
            return glimmer::interceptor::intercept_driver_capture(
                stream, true, glimmer::interceptor::CudaSourceCaptureKind::kBegin,
                [&] { return forward_metadata(native, stream, mode); });
        });
    });
}

extern "C" CUresult CUDAAPI cuStreamEndCapture(CUstream stream, CUgraph* graph) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream, CUgraph*);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuStreamEndCapture", false));
        return guard_copy([&] {
            if (const auto dispatched = glimmer::interceptor::dispatch_runtime_capture_origin(
                    stream, false, glimmer::interceptor::CudaSourceCaptureKind::kEnd,
                    [&] { return forward_metadata(native, stream, graph); }, graph)) {
                return *dispatched;
            }
            return glimmer::interceptor::intercept_driver_capture(
                stream, false, glimmer::interceptor::CudaSourceCaptureKind::kEnd,
                [&] { return forward_metadata(native, stream, graph); });
        });
    });
}

extern "C" CUresult CUDAAPI cuStreamEndCapture_ptsz(CUstream stream, CUgraph* graph) {
    return guard_copy([&]() -> CUresult {
        using Function = CUresult(CUDAAPI*)(CUstream, CUgraph*);
        static const auto native = reinterpret_cast<Function>(
            glimmer::interceptor::resolve_native_driver_function("cuStreamEndCapture", true));
        return guard_copy([&] {
            if (const auto dispatched = glimmer::interceptor::dispatch_runtime_capture_origin(
                    stream, true, glimmer::interceptor::CudaSourceCaptureKind::kEnd,
                    [&] { return forward_metadata(native, stream, graph); }, graph)) {
                return *dispatched;
            }
            return glimmer::interceptor::intercept_driver_capture(
                stream, true, glimmer::interceptor::CudaSourceCaptureKind::kEnd,
                [&] { return forward_metadata(native, stream, graph); });
        });
    });
}
