#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_dispatch.h"

#include <dlfcn.h>

namespace glimmer::interceptor {

namespace {

thread_local bool g_is_inside_driver_call = false;

class DriverCallScope {
   public:
    DriverCallScope() : previous_state_(g_is_inside_driver_call) {
        g_is_inside_driver_call = true;
    }

    ~DriverCallScope() {
        g_is_inside_driver_call = previous_state_;
    }

   private:
    bool previous_state_;
};

}  // namespace

bool is_inside_driver_call() noexcept {
    return g_is_inside_driver_call;
}

DlsymFunction resolve_real_dlsym() noexcept {
    static DlsymFunction real_dlsym = []() noexcept {
        void* symbol = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        return reinterpret_cast<DlsymFunction>(symbol);
    }();
    return real_dlsym;
}

DriverDispatch::~DriverDispatch() {
    if (library_handle_ != nullptr) {
        dlclose(library_handle_);
    }
}

DriverDispatch::DriverDispatch(DriverFunctionTable functions) noexcept
    : init_(functions.init),
      mem_alloc_(functions.mem_alloc),
      mem_alloc_managed_(functions.mem_alloc_managed),
      mem_alloc_pitch_(functions.mem_alloc_pitch),
      mem_alloc_async_(functions.mem_alloc_async),
      mem_alloc_async_ptsz_(functions.mem_alloc_async_ptsz),
      mem_alloc_from_pool_async_(functions.mem_alloc_from_pool_async),
      mem_alloc_from_pool_async_ptsz_(functions.mem_alloc_from_pool_async_ptsz),
      mem_free_(functions.mem_free),
      mem_free_async_(functions.mem_free_async),
      mem_free_async_ptsz_(functions.mem_free_async_ptsz),
      mem_get_info_(functions.mem_get_info),
      device_total_mem_(functions.device_total_mem),
      context_synchronize_(functions.context_synchronize),
      context_get_current_(functions.context_get_current),
      context_get_device_(functions.context_get_device),
      context_destroy_(functions.context_destroy),
      stream_get_device_(functions.stream_get_device),
      stream_get_device_ptsz_(functions.stream_get_device_ptsz),
      stream_get_context_(functions.stream_get_context),
      stream_get_context_ptsz_(functions.stream_get_context_ptsz),
      stream_query_(functions.stream_query),
      stream_query_ptsz_(functions.stream_query_ptsz),
      stream_synchronize_(functions.stream_synchronize),
      stream_synchronize_ptsz_(functions.stream_synchronize_ptsz),
      stream_destroy_(functions.stream_destroy),
      get_proc_address_(functions.get_proc_address),
      get_proc_address_v2_(functions.get_proc_address_v2) {}

bool DriverDispatch::initialize() {
    // The CUDA driver's loader invokes dlsym while initializing its own
    // shared object.  Keep those lookups on the real path; returning Glimmer
    // wrappers from inside the driver's constructor can corrupt its setup.
    DriverCallScope scope;
    library_handle_ = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (library_handle_ == nullptr) {
        return false;
    }

    init_ = reinterpret_cast<InitFunction>(load_symbol("cuInit"));
    mem_alloc_ = reinterpret_cast<MemAllocFunction>(load_symbol("cuMemAlloc_v2"));
    mem_alloc_managed_ =
        reinterpret_cast<MemAllocManagedFunction>(load_symbol("cuMemAllocManaged"));
    mem_alloc_pitch_ = reinterpret_cast<MemAllocPitchFunction>(load_symbol("cuMemAllocPitch_v2"));
    mem_alloc_async_ = reinterpret_cast<MemAllocAsyncFunction>(load_symbol("cuMemAllocAsync"));
    mem_alloc_async_ptsz_ =
        reinterpret_cast<MemAllocAsyncFunction>(load_symbol("cuMemAllocAsync_ptsz"));
    mem_alloc_from_pool_async_ =
        reinterpret_cast<MemAllocFromPoolAsyncFunction>(load_symbol("cuMemAllocFromPoolAsync"));
    mem_alloc_from_pool_async_ptsz_ = reinterpret_cast<MemAllocFromPoolAsyncFunction>(
        load_symbol("cuMemAllocFromPoolAsync_ptsz"));
    mem_free_ = reinterpret_cast<MemFreeFunction>(load_symbol("cuMemFree_v2"));
    mem_free_async_ = reinterpret_cast<MemFreeAsyncFunction>(load_symbol("cuMemFreeAsync"));
    mem_free_async_ptsz_ =
        reinterpret_cast<MemFreeAsyncFunction>(load_symbol("cuMemFreeAsync_ptsz"));
    mem_get_info_ = reinterpret_cast<MemGetInfoFunction>(load_symbol("cuMemGetInfo_v2"));
    device_total_mem_ =
        reinterpret_cast<DeviceTotalMemFunction>(load_symbol("cuDeviceTotalMem_v2"));
    context_synchronize_ =
        reinterpret_cast<ContextSynchronizeFunction>(load_symbol("cuCtxSynchronize"));
    context_get_current_ =
        reinterpret_cast<ContextGetCurrentFunction>(load_symbol("cuCtxGetCurrent"));
    context_get_device_ = reinterpret_cast<ContextGetDeviceFunction>(load_symbol("cuCtxGetDevice"));
    context_destroy_ = reinterpret_cast<ContextDestroyFunction>(load_symbol("cuCtxDestroy_v2"));
    stream_get_device_ =
        reinterpret_cast<StreamGetDeviceFunction>(load_symbol("cuStreamGetDevice"));
    stream_get_device_ptsz_ =
        reinterpret_cast<StreamGetDeviceFunction>(load_symbol("cuStreamGetDevice_ptsz"));
    stream_get_context_ = reinterpret_cast<StreamGetContextFunction>(load_symbol("cuStreamGetCtx"));
    stream_get_context_ptsz_ =
        reinterpret_cast<StreamGetContextFunction>(load_symbol("cuStreamGetCtx_ptsz"));
    stream_query_ = reinterpret_cast<StreamQueryFunction>(load_symbol("cuStreamQuery"));
    stream_query_ptsz_ = reinterpret_cast<StreamQueryFunction>(load_symbol("cuStreamQuery_ptsz"));
    stream_synchronize_ =
        reinterpret_cast<StreamSynchronizeFunction>(load_symbol("cuStreamSynchronize"));
    stream_synchronize_ptsz_ =
        reinterpret_cast<StreamSynchronizeFunction>(load_symbol("cuStreamSynchronize_ptsz"));
    stream_destroy_ = reinterpret_cast<StreamDestroyFunction>(load_symbol("cuStreamDestroy_v2"));
    if (stream_destroy_ == nullptr) {
        stream_destroy_ = reinterpret_cast<StreamDestroyFunction>(load_symbol("cuStreamDestroy"));
    }
    get_proc_address_ = reinterpret_cast<GetProcAddressFunction>(load_symbol("cuGetProcAddress"));
    get_proc_address_v2_ =
        reinterpret_cast<GetProcAddressV2Function>(load_symbol("cuGetProcAddress_v2"));
    if (init_ != nullptr && mem_alloc_ != nullptr && mem_free_ != nullptr &&
        mem_get_info_ != nullptr) {
        return true;
    }

    dlclose(library_handle_);
    library_handle_ = nullptr;
    return false;
}

CUresult DriverDispatch::mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) const {
    DriverCallScope scope;
    if (mem_alloc_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_(device_pointer, memory_bytes);
}

CUresult DriverDispatch::init(unsigned int flags) const {
    DriverCallScope scope;
    if (init_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return init_(flags);
}

CUresult DriverDispatch::mem_alloc_managed(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                           unsigned int flags) const {
    DriverCallScope scope;
    if (mem_alloc_managed_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_managed_(device_pointer, memory_bytes, flags);
}

CUresult DriverDispatch::mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                         std::size_t width_bytes, std::size_t height,
                                         unsigned int element_size_bytes) const {
    DriverCallScope scope;
    if (mem_alloc_pitch_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_pitch_(device_pointer, pitch, width_bytes, height, element_size_bytes);
}

CUresult DriverDispatch::mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                         CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_async_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_async_(device_pointer, memory_bytes, stream);
}

CUresult DriverDispatch::mem_alloc_async_ptsz(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                              CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_async_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_async_ptsz_(device_pointer, memory_bytes, stream);
}

CUresult DriverDispatch::mem_alloc_from_pool_async(CUdeviceptr* device_pointer,
                                                   std::size_t memory_bytes, CUmemoryPool pool,
                                                   CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_from_pool_async_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_from_pool_async_(device_pointer, memory_bytes, pool, stream);
}

CUresult DriverDispatch::mem_alloc_from_pool_async_ptsz(CUdeviceptr* device_pointer,
                                                        std::size_t memory_bytes, CUmemoryPool pool,
                                                        CUstream stream) const {
    DriverCallScope scope;
    if (mem_alloc_from_pool_async_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_alloc_from_pool_async_ptsz_(device_pointer, memory_bytes, pool, stream);
}

CUresult DriverDispatch::mem_free(CUdeviceptr device_pointer) const {
    DriverCallScope scope;
    if (mem_free_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_free_(device_pointer);
}

CUresult DriverDispatch::mem_free_async(CUdeviceptr device_pointer, CUstream stream) const {
    DriverCallScope scope;
    if (mem_free_async_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_free_async_(device_pointer, stream);
}

CUresult DriverDispatch::mem_free_async_ptsz(CUdeviceptr device_pointer, CUstream stream) const {
    DriverCallScope scope;
    if (mem_free_async_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_free_async_ptsz_(device_pointer, stream);
}

CUresult DriverDispatch::mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) const {
    DriverCallScope scope;
    if (mem_get_info_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return mem_get_info_(free_bytes, total_bytes);
}

CUresult DriverDispatch::device_total_mem(std::size_t* total_bytes, CUdevice device) const {
    DriverCallScope scope;
    if (device_total_mem_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return device_total_mem_(total_bytes, device);
}

CUresult DriverDispatch::context_synchronize() const {
    DriverCallScope scope;
    if (context_synchronize_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_synchronize_();
}

CUresult DriverDispatch::context_get_current(CUcontext* context) const {
    DriverCallScope scope;
    if (context_get_current_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_get_current_(context);
}

CUresult DriverDispatch::context_get_device(CUdevice* device) const {
    DriverCallScope scope;
    if (context_get_device_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_get_device_(device);
}

CUresult DriverDispatch::context_destroy(CUcontext context) const {
    DriverCallScope scope;
    if (context_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return context_destroy_(context);
}

CUresult DriverDispatch::stream_get_device(CUstream stream, CUdevice* device) const {
    DriverCallScope scope;
    if (stream_get_device_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_device_(stream, device);
}

CUresult DriverDispatch::stream_get_device_ptsz(CUstream stream, CUdevice* device) const {
    DriverCallScope scope;
    if (stream_get_device_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_device_ptsz_(stream, device);
}

CUresult DriverDispatch::stream_get_context(CUstream stream, CUcontext* context) const {
    DriverCallScope scope;
    if (stream_get_context_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_context_(stream, context);
}

CUresult DriverDispatch::stream_get_context_ptsz(CUstream stream, CUcontext* context) const {
    DriverCallScope scope;
    if (stream_get_context_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_get_context_ptsz_(stream, context);
}

CUresult DriverDispatch::stream_query(CUstream stream) const {
    DriverCallScope scope;
    if (stream_query_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_query_(stream);
}

CUresult DriverDispatch::stream_query_ptsz(CUstream stream) const {
    DriverCallScope scope;
    if (stream_query_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_query_ptsz_(stream);
}

CUresult DriverDispatch::stream_synchronize(CUstream stream) const {
    DriverCallScope scope;
    if (stream_synchronize_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_synchronize_(stream);
}

CUresult DriverDispatch::stream_synchronize_ptsz(CUstream stream) const {
    DriverCallScope scope;
    if (stream_synchronize_ptsz_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_synchronize_ptsz_(stream);
}

CUresult DriverDispatch::stream_destroy(CUstream stream) const {
    DriverCallScope scope;
    if (stream_destroy_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return stream_destroy_(stream);
}

CUresult DriverDispatch::get_proc_address(const char* symbol, void** function_pointer,
                                          int cuda_version, cuuint64_t flags) const {
    DriverCallScope scope;
    if (get_proc_address_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return get_proc_address_(symbol, function_pointer, cuda_version, flags);
}

CUresult DriverDispatch::get_proc_address_v2(const char* symbol, void** function_pointer,
                                             int cuda_version, cuuint64_t flags,
                                             CUdriverProcAddressQueryResult* symbol_status) const {
    DriverCallScope scope;
    if (get_proc_address_v2_ == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return get_proc_address_v2_(symbol, function_pointer, cuda_version, flags, symbol_status);
}

bool DriverDispatch::has_get_proc_address() const {
    return get_proc_address_ != nullptr;
}

bool DriverDispatch::has_get_proc_address_v2() const {
    return get_proc_address_v2_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_managed() const {
    return mem_alloc_managed_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_pitch() const {
    return mem_alloc_pitch_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_async() const {
    return mem_alloc_async_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_async_ptsz() const {
    return mem_alloc_async_ptsz_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_from_pool_async() const {
    return mem_alloc_from_pool_async_ != nullptr;
}

bool DriverDispatch::has_mem_alloc_from_pool_async_ptsz() const {
    return mem_alloc_from_pool_async_ptsz_ != nullptr;
}

bool DriverDispatch::has_mem_free_async() const {
    return mem_free_async_ != nullptr;
}

bool DriverDispatch::has_mem_free_async_ptsz() const {
    return mem_free_async_ptsz_ != nullptr;
}

bool DriverDispatch::has_device_total_mem() const {
    return device_total_mem_ != nullptr;
}

bool DriverDispatch::has_context_synchronize() const {
    return context_synchronize_ != nullptr;
}

bool DriverDispatch::has_context_queries() const {
    return context_get_current_ != nullptr && context_get_device_ != nullptr;
}

bool DriverDispatch::has_context_destroy() const {
    return context_destroy_ != nullptr;
}

bool DriverDispatch::has_stream_identity() const {
    return stream_get_device_ != nullptr && stream_get_context_ != nullptr;
}

bool DriverDispatch::has_stream_identity_ptsz() const {
    return stream_get_device_ptsz_ != nullptr && stream_get_context_ptsz_ != nullptr;
}

bool DriverDispatch::has_stream_query() const {
    return stream_query_ != nullptr;
}

bool DriverDispatch::has_stream_query_ptsz() const {
    return stream_query_ptsz_ != nullptr;
}

bool DriverDispatch::has_stream_synchronize() const {
    return stream_synchronize_ != nullptr;
}

bool DriverDispatch::has_stream_synchronize_ptsz() const {
    return stream_synchronize_ptsz_ != nullptr;
}

bool DriverDispatch::has_stream_destroy() const {
    return stream_destroy_ != nullptr;
}

void* DriverDispatch::load_symbol(const char* name) const {
    const DlsymFunction real_dlsym = resolve_real_dlsym();
    if (real_dlsym == nullptr) {
        return nullptr;
    }

    dlerror();
    void* symbol = real_dlsym(library_handle_, name);
    if (dlerror() != nullptr) {
        return nullptr;
    }
    return symbol;
}

}  // namespace glimmer::interceptor
