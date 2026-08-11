#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/nvml_dispatch.h"

#include "internal/driver_dispatch.h"

#include <dlfcn.h>

namespace glimmer::interceptor {

namespace {

thread_local bool g_is_inside_nvml_call = false;

class NvmlCallScope {
   public:
    NvmlCallScope() : previous_state_(g_is_inside_nvml_call) {
        g_is_inside_nvml_call = true;
    }

    ~NvmlCallScope() {
        g_is_inside_nvml_call = previous_state_;
    }

   private:
    bool previous_state_;
};

}  // namespace

bool is_inside_nvml_call() noexcept {
    return g_is_inside_nvml_call;
}

NvmlDispatch::NvmlDispatch(NvmlFunctionTable functions) noexcept
    : init_(functions.init),
      init_with_flags_(functions.init_with_flags),
      shutdown_(functions.shutdown),
      device_get_count_(functions.device_get_count),
      device_get_handle_by_index_(functions.device_get_handle_by_index),
      device_get_index_(functions.device_get_index),
      device_get_memory_info_(functions.device_get_memory_info),
      device_get_memory_info_v2_(functions.device_get_memory_info_v2) {}

NvmlDispatch::~NvmlDispatch() {
    if (library_handle_ != nullptr) {
        dlclose(library_handle_);
    }
}

bool NvmlDispatch::initialize() {
    std::lock_guard lock(mutex_);
    if (init_ != nullptr && shutdown_ != nullptr && device_get_count_ != nullptr &&
        device_get_handle_by_index_ != nullptr && device_get_index_ != nullptr &&
        device_get_memory_info_ != nullptr) {
        return true;
    }

    NvmlCallScope scope;
    library_handle_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (library_handle_ == nullptr) {
        return false;
    }

    init_ = reinterpret_cast<NvmlInitFunction>(load_symbol("nvmlInit_v2"));
    if (init_ == nullptr) {
        init_ = reinterpret_cast<NvmlInitFunction>(load_symbol("nvmlInit"));
    }
    init_with_flags_ =
        reinterpret_cast<NvmlInitWithFlagsFunction>(load_symbol("nvmlInitWithFlags"));
    shutdown_ = reinterpret_cast<NvmlShutdownFunction>(load_symbol("nvmlShutdown"));
    device_get_count_ =
        reinterpret_cast<NvmlDeviceGetCountFunction>(load_symbol("nvmlDeviceGetCount_v2"));
    if (device_get_count_ == nullptr) {
        device_get_count_ =
            reinterpret_cast<NvmlDeviceGetCountFunction>(load_symbol("nvmlDeviceGetCount"));
    }
    device_get_handle_by_index_ = reinterpret_cast<NvmlDeviceGetHandleByIndexFunction>(
        load_symbol("nvmlDeviceGetHandleByIndex_v2"));
    if (device_get_handle_by_index_ == nullptr) {
        device_get_handle_by_index_ = reinterpret_cast<NvmlDeviceGetHandleByIndexFunction>(
            load_symbol("nvmlDeviceGetHandleByIndex"));
    }
    device_get_index_ =
        reinterpret_cast<NvmlDeviceGetIndexFunction>(load_symbol("nvmlDeviceGetIndex"));
    device_get_memory_info_ =
        reinterpret_cast<NvmlDeviceGetMemoryInfoFunction>(load_symbol("nvmlDeviceGetMemoryInfo"));
    device_get_memory_info_v2_ = reinterpret_cast<NvmlDeviceGetMemoryInfoV2Function>(
        load_symbol("nvmlDeviceGetMemoryInfo_v2"));

    if (init_ != nullptr && shutdown_ != nullptr && device_get_count_ != nullptr &&
        device_get_handle_by_index_ != nullptr && device_get_index_ != nullptr &&
        device_get_memory_info_ != nullptr) {
        return true;
    }

    dlclose(library_handle_);
    library_handle_ = nullptr;
    init_ = nullptr;
    init_with_flags_ = nullptr;
    shutdown_ = nullptr;
    device_get_count_ = nullptr;
    device_get_handle_by_index_ = nullptr;
    device_get_index_ = nullptr;
    device_get_memory_info_ = nullptr;
    device_get_memory_info_v2_ = nullptr;
    return false;
}

nvmlReturn_t NvmlDispatch::init() const {
    NvmlCallScope scope;
    return init_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND : init_();
}

nvmlReturn_t NvmlDispatch::init_with_flags(unsigned int flags) const {
    NvmlCallScope scope;
    return init_with_flags_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND : init_with_flags_(flags);
}

nvmlReturn_t NvmlDispatch::shutdown() const {
    NvmlCallScope scope;
    return shutdown_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND : shutdown_();
}

nvmlReturn_t NvmlDispatch::device_get_count(unsigned int* device_count) const {
    NvmlCallScope scope;
    return device_get_count_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                                        : device_get_count_(device_count);
}

nvmlReturn_t NvmlDispatch::device_get_handle_by_index(unsigned int index,
                                                      nvmlDevice_t* device) const {
    NvmlCallScope scope;
    return device_get_handle_by_index_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                                                  : device_get_handle_by_index_(index, device);
}

nvmlReturn_t NvmlDispatch::device_get_index(nvmlDevice_t device, unsigned int* index) const {
    NvmlCallScope scope;
    return device_get_index_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                                        : device_get_index_(device, index);
}

nvmlReturn_t NvmlDispatch::device_get_memory_info(nvmlDevice_t device, nvmlMemory_t* memory) const {
    NvmlCallScope scope;
    return device_get_memory_info_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                                              : device_get_memory_info_(device, memory);
}

nvmlReturn_t NvmlDispatch::device_get_memory_info_v2(nvmlDevice_t device,
                                                     nvmlMemory_v2_t* memory) const {
    NvmlCallScope scope;
    return device_get_memory_info_v2_ == nullptr ? NVML_ERROR_FUNCTION_NOT_FOUND
                                                 : device_get_memory_info_v2_(device, memory);
}

bool NvmlDispatch::has_init() const {
    return init_ != nullptr;
}

bool NvmlDispatch::has_init_with_flags() const {
    return init_with_flags_ != nullptr;
}

bool NvmlDispatch::has_shutdown() const {
    return shutdown_ != nullptr;
}

bool NvmlDispatch::has_device_get_count() const {
    return device_get_count_ != nullptr;
}

bool NvmlDispatch::has_device_get_handle_by_index() const {
    return device_get_handle_by_index_ != nullptr;
}

bool NvmlDispatch::has_device_get_index() const {
    return device_get_index_ != nullptr;
}

bool NvmlDispatch::has_device_get_memory_info() const {
    return device_get_memory_info_ != nullptr;
}

bool NvmlDispatch::has_device_get_memory_info_v2() const {
    return device_get_memory_info_v2_ != nullptr;
}

void* NvmlDispatch::load_symbol(const char* name) const {
    const DlsymFunction real_dlsym = resolve_real_dlsym();
    if (real_dlsym == nullptr) {
        return nullptr;
    }

    dlerror();
    void* symbol = real_dlsym(library_handle_, name);
    return dlerror() == nullptr ? symbol : nullptr;
}

}  // namespace glimmer::interceptor
