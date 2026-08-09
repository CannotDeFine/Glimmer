#include "internal/runtime_api_bridge.h"

#include "internal/driver_dispatch.h"

#include <cstddef>
#include <dlfcn.h>

namespace {

thread_local bool g_is_inside_runtime_call = false;

class RuntimeCallScope {
   public:
    RuntimeCallScope() : previous_state_(g_is_inside_runtime_call) {
        g_is_inside_runtime_call = true;
    }

    ~RuntimeCallScope() {
        g_is_inside_runtime_call = previous_state_;
    }

   private:
    bool previous_state_;
};

using glimmer::interceptor::DlsymFunction;
using glimmer::interceptor::RuntimeDeviceSynchronizeFunction;
using glimmer::interceptor::RuntimeFreeAsyncFunction;
using glimmer::interceptor::RuntimeFreeFunction;
using glimmer::interceptor::RuntimeGetDeviceFunction;
using glimmer::interceptor::RuntimeMallocAsyncFunction;
using glimmer::interceptor::RuntimeMallocFunction;
using glimmer::interceptor::RuntimeMemGetInfoFunction;
using glimmer::interceptor::RuntimeStreamDestroyFunction;
using glimmer::interceptor::RuntimeStreamQueryFunction;
using glimmer::interceptor::RuntimeStreamSynchronizeFunction;

[[nodiscard]] RuntimeMallocFunction resolve_runtime_malloc() noexcept {
    static RuntimeMallocFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMallocFunction>(real_dlsym(RTLD_NEXT, "cudaMalloc"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeFunction resolve_runtime_free() noexcept {
    static RuntimeFreeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeFreeFunction>(real_dlsym(RTLD_NEXT, "cudaFree"));
    }();
    return function;
}

[[nodiscard]] RuntimeGetDeviceFunction resolve_runtime_get_device() noexcept {
    static RuntimeGetDeviceFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeGetDeviceFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaGetDevice"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetInfoFunction resolve_runtime_mem_get_info() noexcept {
    static RuntimeMemGetInfoFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetInfoFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetInfo"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocAsyncFunction resolve_runtime_malloc_async() noexcept {
    static RuntimeMallocAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocAsyncFunction resolve_runtime_malloc_async_ptsz() noexcept {
    static RuntimeMallocAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeAsyncFunction resolve_runtime_free_async() noexcept {
    static RuntimeFreeAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeFreeAsyncFunction resolve_runtime_free_async_ptsz() noexcept {
    static RuntimeFreeAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeAsync_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceSynchronizeFunction resolve_runtime_device_synchronize() noexcept {
    static RuntimeDeviceSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceSynchronize"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamSynchronizeFunction resolve_runtime_stream_synchronize() noexcept {
    static RuntimeStreamSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamSynchronize"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamSynchronizeFunction resolve_runtime_stream_synchronize_ptsz() noexcept {
    static RuntimeStreamSynchronizeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamSynchronizeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamSynchronize_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamQueryFunction resolve_runtime_stream_query() noexcept {
    static RuntimeStreamQueryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamQueryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamQuery"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamQueryFunction resolve_runtime_stream_query_ptsz() noexcept {
    static RuntimeStreamQueryFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamQueryFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamQuery_ptsz"));
    }();
    return function;
}

[[nodiscard]] RuntimeStreamDestroyFunction resolve_runtime_stream_destroy() noexcept {
    static RuntimeStreamDestroyFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeStreamDestroyFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaStreamDestroy"));
    }();
    return function;
}

}  // namespace

namespace glimmer::interceptor {

bool is_inside_runtime_call() noexcept {
    return g_is_inside_runtime_call;
}

}  // namespace glimmer::interceptor

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocFunction real_allocate = resolve_runtime_malloc();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes);
        }
        return glimmer::interceptor::intercept_runtime_malloc(device_pointer, memory_bytes,
                                                              real_allocate, resolve_runtime_free(),
                                                              resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFree(void* device_pointer) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeFunction real_release = resolve_runtime_free();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported : real_release(device_pointer);
        }
        return glimmer::interceptor::intercept_runtime_free(device_pointer, real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetInfo(std::size_t* free_bytes, std::size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMemGetInfoFunction real_query = resolve_runtime_mem_get_info();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported
                                         : real_query(free_bytes, total_bytes);
        }
        return glimmer::interceptor::intercept_runtime_mem_get_info(free_bytes, total_bytes,
                                                                    real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocAsyncFunction real_allocate = resolve_runtime_malloc_async();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_async(
            device_pointer, memory_bytes, stream, real_allocate, resolve_runtime_free_async(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocAsync_ptsz(void** device_pointer,
                                                      std::size_t memory_bytes,
                                                      cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocAsyncFunction real_allocate = resolve_runtime_malloc_async_ptsz();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_async(
            device_pointer, memory_bytes, stream, real_allocate, resolve_runtime_free_async_ptsz(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeAsyncFunction real_release = resolve_runtime_free_async();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported
                                           : real_release(device_pointer, stream);
        }
        return glimmer::interceptor::intercept_runtime_free_async(device_pointer, stream,
                                                                  real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync_ptsz(void* device_pointer, cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeFreeAsyncFunction real_release = resolve_runtime_free_async_ptsz();
        if (is_reentrant) {
            return real_release == nullptr ? cudaErrorNotSupported
                                           : real_release(device_pointer, stream);
        }
        return glimmer::interceptor::intercept_runtime_free_async(device_pointer, stream,
                                                                  real_release);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceSynchronizeFunction real_synchronize =
            resolve_runtime_device_synchronize();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize();
        }
        return glimmer::interceptor::intercept_runtime_device_synchronize(real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamSynchronizeFunction real_synchronize =
            resolve_runtime_stream_synchronize();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_synchronize(stream, real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize_ptsz(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamSynchronizeFunction real_synchronize =
            resolve_runtime_stream_synchronize_ptsz();
        if (is_reentrant) {
            return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_synchronize(stream, real_synchronize);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamQueryFunction real_query = resolve_runtime_stream_query();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_query(stream, real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamQuery_ptsz(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamQueryFunction real_query = resolve_runtime_stream_query_ptsz();
        if (is_reentrant) {
            return real_query == nullptr ? cudaErrorNotSupported : real_query(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_query(stream, real_query);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeStreamDestroyFunction real_destroy = resolve_runtime_stream_destroy();
        if (is_reentrant) {
            return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(stream);
        }
        return glimmer::interceptor::intercept_runtime_stream_destroy(stream, real_destroy);
    } catch (...) {
        return cudaErrorUnknown;
    }
}
