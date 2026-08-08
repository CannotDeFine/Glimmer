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
using glimmer::interceptor::RuntimeMallocAsyncFunction;
using glimmer::interceptor::RuntimeMallocFunction;
using glimmer::interceptor::RuntimeMemGetInfoFunction;

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

[[nodiscard]] RuntimeFreeAsyncFunction resolve_runtime_free_async() noexcept {
    static RuntimeFreeAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeFreeAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaFreeAsync"));
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

}  // namespace

namespace glimmer::interceptor {

bool is_inside_runtime_call() noexcept {
    return g_is_inside_runtime_call;
}

}  // namespace glimmer::interceptor

extern "C" cudaError_t CUDARTAPI cudaMalloc(void** device_pointer, std::size_t memory_bytes) {
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocFunction real_allocate = resolve_runtime_malloc();
        if (is_reentrant) {
            return real_allocate == nullptr ? cudaErrorNotSupported
                                            : real_allocate(device_pointer, memory_bytes);
        }
        return glimmer::interceptor::intercept_runtime_malloc(
            device_pointer, memory_bytes, real_allocate, resolve_runtime_free());
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

// Runtime stream-ordered allocation is deliberately forwarded without the
// synchronous Runtime-call guard.  libcudart normally lowers these calls to
// the covered Driver stream-ordered APIs; leaving the Driver guard available
// lets the Driver interceptor perform the single authoritative accounting.
extern "C" cudaError_t CUDARTAPI cudaMallocAsync(void** device_pointer, std::size_t memory_bytes,
                                                 cudaStream_t stream) {
    try {
        const RuntimeMallocAsyncFunction real_allocate = resolve_runtime_malloc_async();
        return real_allocate == nullptr ? cudaErrorNotSupported
                                        : real_allocate(device_pointer, memory_bytes, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaFreeAsync(void* device_pointer, cudaStream_t stream) {
    try {
        const RuntimeFreeAsyncFunction real_release = resolve_runtime_free_async();
        return real_release == nullptr ? cudaErrorNotSupported
                                       : real_release(device_pointer, stream);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSynchronize() {
    try {
        const RuntimeDeviceSynchronizeFunction real_synchronize =
            resolve_runtime_device_synchronize();
        return real_synchronize == nullptr ? cudaErrorNotSupported : real_synchronize();
    } catch (...) {
        return cudaErrorUnknown;
    }
}
