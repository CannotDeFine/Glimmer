#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_api_interceptor.h"

#include <nvml.h>

#ifdef nvmlInit
#undef nvmlInit
#endif
#ifdef nvmlDeviceGetCount
#undef nvmlDeviceGetCount
#endif
#ifdef nvmlDeviceGetHandleByIndex
#undef nvmlDeviceGetHandleByIndex
#endif

namespace {

template <typename Function>
nvmlReturn_t guard_nvml_boundary(Function&& function) noexcept {
    try {
        return function();
    } catch (...) {
        return NVML_ERROR_UNKNOWN;
    }
}

}  // namespace

// NOLINTBEGIN(readability-identifier-naming): preserve the NVML ABI names.
extern "C" nvmlReturn_t nvmlInit() {
    return guard_nvml_boundary([] { return glimmer::interceptor::intercept_nvml_init(); });
}

extern "C" nvmlReturn_t nvmlInit_v2() {
    return nvmlInit();
}

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
    return guard_nvml_boundary(
        [flags] { return glimmer::interceptor::intercept_nvml_init_with_flags(flags); });
}

extern "C" nvmlReturn_t nvmlShutdown() {
    return guard_nvml_boundary([] { return glimmer::interceptor::intercept_nvml_shutdown(); });
}

extern "C" nvmlReturn_t nvmlDeviceGetCount(unsigned int* device_count) {
    return guard_nvml_boundary([device_count] {
        return glimmer::interceptor::intercept_nvml_device_get_count(device_count);
    });
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* device_count) {
    return nvmlDeviceGetCount(device_count);
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
    return guard_nvml_boundary([index, device] {
        return glimmer::interceptor::intercept_nvml_device_get_handle_by_index(index, device);
    });
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    return nvmlDeviceGetHandleByIndex(index, device);
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) {
    return guard_nvml_boundary([device, index] {
        return glimmer::interceptor::intercept_nvml_device_get_index(device, index);
    });
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    return guard_nvml_boundary([device, memory] {
        return glimmer::interceptor::intercept_nvml_device_get_memory_info(device, memory);
    });
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
    return guard_nvml_boundary([device, memory] {
        return glimmer::interceptor::intercept_nvml_device_get_memory_info_v2(device, memory);
    });
}
// NOLINTEND(readability-identifier-naming)
