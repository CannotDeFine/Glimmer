#include "internal/runtime_api_bridge.h"

#include "internal/driver_dispatch.h"

#include <cstddef>
#include <dlfcn.h>

#ifdef cudaMallocFromPoolAsync
#undef cudaMallocFromPoolAsync
#endif

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
using glimmer::interceptor::RuntimeDeviceGetDefaultMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceGetMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceSetMemPoolFunction;
using glimmer::interceptor::RuntimeDeviceSynchronizeFunction;
using glimmer::interceptor::RuntimeFreeAsyncFunction;
using glimmer::interceptor::RuntimeFreeFunction;
using glimmer::interceptor::RuntimeGetDeviceFunction;
using glimmer::interceptor::RuntimeMallocAsyncFunction;
using glimmer::interceptor::RuntimeMallocFromPoolAsyncFunction;
using glimmer::interceptor::RuntimeMallocFunction;
using glimmer::interceptor::RuntimeMemGetDefaultMemPoolFunction;
using glimmer::interceptor::RuntimeMemGetInfoFunction;
using glimmer::interceptor::RuntimeMemGetMemPoolFunction;
using glimmer::interceptor::RuntimeMemPoolCreateFunction;
using glimmer::interceptor::RuntimeMemPoolDestroyFunction;
using glimmer::interceptor::RuntimeMemPoolExportPointerFunction;
using glimmer::interceptor::RuntimeMemPoolExportToShareableHandleFunction;
using glimmer::interceptor::RuntimeMemPoolGetAccessFunction;
using glimmer::interceptor::RuntimeMemPoolGetAttributeFunction;
using glimmer::interceptor::RuntimeMemPoolImportFromShareableHandleFunction;
using glimmer::interceptor::RuntimeMemPoolImportPointerFunction;
using glimmer::interceptor::RuntimeMemPoolSetAccessFunction;
using glimmer::interceptor::RuntimeMemPoolSetAttributeFunction;
using glimmer::interceptor::RuntimeMemPoolTrimToFunction;
using glimmer::interceptor::RuntimeMemSetMemPoolFunction;
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

[[nodiscard]] RuntimeMallocFromPoolAsyncFunction resolve_runtime_malloc_from_pool_async() noexcept {
    static RuntimeMallocFromPoolAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocFromPoolAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync"));
    }();
    return function;
}

[[nodiscard]] RuntimeMallocFromPoolAsyncFunction
resolve_runtime_malloc_from_pool_async_ptsz() noexcept {
    static RuntimeMallocFromPoolAsyncFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMallocFromPoolAsyncFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMallocFromPoolAsync_ptsz"));
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

[[nodiscard]] RuntimeDeviceGetDefaultMemPoolFunction
resolve_runtime_device_get_default_mem_pool() noexcept {
    static RuntimeDeviceGetDefaultMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceGetDefaultMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceGetDefaultMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceSetMemPoolFunction resolve_runtime_device_set_mem_pool() noexcept {
    static RuntimeDeviceSetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceSetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceSetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeDeviceGetMemPoolFunction resolve_runtime_device_get_mem_pool() noexcept {
    static RuntimeDeviceGetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeDeviceGetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaDeviceGetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolTrimToFunction resolve_runtime_mem_pool_trim_to() noexcept {
    static RuntimeMemPoolTrimToFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolTrimToFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolTrimTo"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolSetAttributeFunction resolve_runtime_mem_pool_set_attribute() noexcept {
    static RuntimeMemPoolSetAttributeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolSetAttributeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolSetAttribute"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolGetAttributeFunction resolve_runtime_mem_pool_get_attribute() noexcept {
    static RuntimeMemPoolGetAttributeFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolGetAttributeFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolGetAttribute"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolSetAccessFunction resolve_runtime_mem_pool_set_access() noexcept {
    static RuntimeMemPoolSetAccessFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolSetAccessFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolSetAccess"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolGetAccessFunction resolve_runtime_mem_pool_get_access() noexcept {
    static RuntimeMemPoolGetAccessFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolGetAccessFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolGetAccess"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolCreateFunction resolve_runtime_mem_pool_create() noexcept {
    static RuntimeMemPoolCreateFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolCreateFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolCreate"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolDestroyFunction resolve_runtime_mem_pool_destroy() noexcept {
    static RuntimeMemPoolDestroyFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolDestroyFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolDestroy"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetDefaultMemPoolFunction
resolve_runtime_mem_get_default_mem_pool() noexcept {
    static RuntimeMemGetDefaultMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetDefaultMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetDefaultMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemGetMemPoolFunction resolve_runtime_mem_get_mem_pool() noexcept {
    static RuntimeMemGetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemGetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemGetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemSetMemPoolFunction resolve_runtime_mem_set_mem_pool() noexcept {
    static RuntimeMemSetMemPoolFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemSetMemPoolFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemSetMemPool"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolExportToShareableHandleFunction
resolve_runtime_mem_pool_export_to_shareable_handle() noexcept {
    static RuntimeMemPoolExportToShareableHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMemPoolExportToShareableHandleFunction>(
                         real_dlsym(RTLD_NEXT, "cudaMemPoolExportToShareableHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolImportFromShareableHandleFunction
resolve_runtime_mem_pool_import_from_shareable_handle() noexcept {
    static RuntimeMemPoolImportFromShareableHandleFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr
                   ? nullptr
                   : reinterpret_cast<RuntimeMemPoolImportFromShareableHandleFunction>(
                         real_dlsym(RTLD_NEXT, "cudaMemPoolImportFromShareableHandle"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolExportPointerFunction
resolve_runtime_mem_pool_export_pointer() noexcept {
    static RuntimeMemPoolExportPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolExportPointerFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolExportPointer"));
    }();
    return function;
}

[[nodiscard]] RuntimeMemPoolImportPointerFunction
resolve_runtime_mem_pool_import_pointer() noexcept {
    static RuntimeMemPoolImportPointerFunction function = []() noexcept {
        const DlsymFunction real_dlsym = glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr
                                     : reinterpret_cast<RuntimeMemPoolImportPointerFunction>(
                                           real_dlsym(RTLD_NEXT, "cudaMemPoolImportPointer"));
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

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync(void** device_pointer,
                                                         std::size_t memory_bytes,
                                                         cudaMemPool_t pool, cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocFromPoolAsyncFunction real_allocate =
            resolve_runtime_malloc_from_pool_async();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, memory_bytes, pool, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, real_allocate, resolve_runtime_free_async(),
            resolve_runtime_device_synchronize(), resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMallocFromPoolAsync_ptsz(void** device_pointer,
                                                              std::size_t memory_bytes,
                                                              cudaMemPool_t pool,
                                                              cudaStream_t stream) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMallocFromPoolAsyncFunction real_allocate =
            resolve_runtime_malloc_from_pool_async_ptsz();
        if (is_reentrant) {
            return real_allocate == nullptr
                       ? cudaErrorNotSupported
                       : real_allocate(device_pointer, memory_bytes, pool, stream);
        }
        return glimmer::interceptor::intercept_runtime_malloc_from_pool_async(
            device_pointer, memory_bytes, pool, stream, real_allocate,
            resolve_runtime_free_async_ptsz(), resolve_runtime_device_synchronize(),
            resolve_runtime_get_device());
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceGetDefaultMemPoolFunction real_query =
            resolve_runtime_device_get_default_mem_pool();
        return real_query == nullptr ? cudaErrorNotSupported : real_query(pool, device);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceSetMemPool(int device, cudaMemPool_t pool) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceSetMemPoolFunction real_set = resolve_runtime_device_set_mem_pool();
        return real_set == nullptr ? cudaErrorNotSupported : real_set(device, pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaDeviceGetMemPool(cudaMemPool_t* pool, int device) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeDeviceGetMemPoolFunction real_query = resolve_runtime_device_get_mem_pool();
        return real_query == nullptr ? cudaErrorNotSupported : real_query(pool, device);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolTrimTo(cudaMemPool_t pool,
                                                   std::size_t min_bytes_to_keep) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolTrimToFunction real_trim = resolve_runtime_mem_pool_trim_to();
        return real_trim == nullptr ? cudaErrorNotSupported : real_trim(pool, min_bytes_to_keep);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value) {
    if (pool == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolSetAttributeFunction real_set =
            resolve_runtime_mem_pool_set_attribute();
        return real_set == nullptr ? cudaErrorNotSupported : real_set(pool, attribute, value);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAttribute(cudaMemPool_t pool,
                                                         cudaMemPoolAttr attribute, void* value) {
    if (pool == nullptr || value == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolGetAttributeFunction real_get =
            resolve_runtime_mem_pool_get_attribute();
        return real_get == nullptr ? cudaErrorNotSupported : real_get(pool, attribute, value);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolSetAccess(cudaMemPool_t pool,
                                                      const cudaMemAccessDesc* descriptors,
                                                      std::size_t descriptor_count) {
    if (pool == nullptr || (descriptor_count != 0 && descriptors == nullptr)) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolSetAccessFunction real_set = resolve_runtime_mem_pool_set_access();
        return real_set == nullptr ? cudaErrorNotSupported
                                   : real_set(pool, descriptors, descriptor_count);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolGetAccess(cudaMemAccessFlags* flags, cudaMemPool_t pool,
                                                      cudaMemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolGetAccessFunction real_get = resolve_runtime_mem_pool_get_access();
        return real_get == nullptr ? cudaErrorNotSupported : real_get(flags, pool, location);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolCreate(cudaMemPool_t* pool,
                                                   const cudaMemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolCreateFunction real_create = resolve_runtime_mem_pool_create();
        return real_create == nullptr ? cudaErrorNotSupported : real_create(pool, properties);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolDestroy(cudaMemPool_t pool) {
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolDestroyFunction real_destroy = resolve_runtime_mem_pool_destroy();
        return real_destroy == nullptr ? cudaErrorNotSupported : real_destroy(pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetDefaultMemPool(cudaMemPool_t* pool,
                                                          cudaMemLocation* location,
                                                          cudaMemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemGetDefaultMemPoolFunction real_get =
            resolve_runtime_mem_get_default_mem_pool();
        return real_get == nullptr ? cudaErrorNotSupported
                                   : real_get(pool, location, allocation_type);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemGetMemPool(cudaMemPool_t* pool, cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemGetMemPoolFunction real_get = resolve_runtime_mem_get_mem_pool();
        return real_get == nullptr ? cudaErrorNotSupported
                                   : real_get(pool, location, allocation_type);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemSetMemPool(cudaMemLocation* location,
                                                   cudaMemAllocationType allocation_type,
                                                   cudaMemPool_t pool) {
    if (location == nullptr || pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemSetMemPoolFunction real_set = resolve_runtime_mem_set_mem_pool();
        return real_set == nullptr ? cudaErrorNotSupported
                                   : real_set(location, allocation_type, pool);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportToShareableHandle(
    void* handle_out, cudaMemPool_t pool, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags) {
    if (handle_out == nullptr || pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolExportToShareableHandleFunction real_export =
            resolve_runtime_mem_pool_export_to_shareable_handle();
        return real_export == nullptr ? cudaErrorNotSupported
                                      : real_export(handle_out, pool, handle_type, flags);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportFromShareableHandle(
    cudaMemPool_t* pool_out, void* handle, enum cudaMemAllocationHandleType handle_type,
    unsigned int flags) {
    if (pool_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolImportFromShareableHandleFunction real_import =
            resolve_runtime_mem_pool_import_from_shareable_handle();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(pool_out, handle, handle_type, flags);
        }
        return glimmer::interceptor::intercept_runtime_mem_pool_import_from_shareable_handle(
            pool_out, handle, handle_type, flags, real_import);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolExportPointer(cudaMemPoolPtrExportData* share_data_out,
                                                          void* device_pointer) {
    if (share_data_out == nullptr || device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolExportPointerFunction real_export =
            resolve_runtime_mem_pool_export_pointer();
        return real_export == nullptr ? cudaErrorNotSupported
                                      : real_export(share_data_out, device_pointer);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

extern "C" cudaError_t CUDARTAPI cudaMemPoolImportPointer(void** pointer_out, cudaMemPool_t pool,
                                                          cudaMemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return cudaErrorInvalidValue;
    }
    const bool is_reentrant = glimmer::interceptor::is_inside_runtime_call();
    RuntimeCallScope scope;
    try {
        const RuntimeMemPoolImportPointerFunction real_import =
            resolve_runtime_mem_pool_import_pointer();
        if (is_reentrant) {
            return real_import == nullptr ? cudaErrorNotSupported
                                          : real_import(pointer_out, pool, share_data);
        }
        return glimmer::interceptor::intercept_runtime_mem_pool_import_pointer(
            pointer_out, pool, share_data, real_import);
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
