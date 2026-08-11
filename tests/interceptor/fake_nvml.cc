#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <nvml.h>

#include <cstddef>

#ifdef nvmlInit
#undef nvmlInit
#endif
#ifdef nvmlDeviceGetCount
#undef nvmlDeviceGetCount
#endif
#ifdef nvmlDeviceGetHandleByIndex
#undef nvmlDeviceGetHandleByIndex
#endif

struct nvmlDevice_st {
    unsigned int index;
};

namespace {

nvmlDevice_st g_devices[] = {{.index = 0}, {.index = 1}};

bool is_known_device(nvmlDevice_t device) noexcept {
    return device == &g_devices[0] || device == &g_devices[1];
}

}  // namespace

// NOLINTBEGIN(readability-identifier-naming): preserve the NVML ABI names.
extern "C" nvmlReturn_t nvmlInit() {
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlInit_v2() {
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlInitWithFlags(unsigned int) {
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlShutdown() {
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount(unsigned int* device_count) {
    if (device_count == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *device_count = static_cast<unsigned int>(sizeof(g_devices) / sizeof(g_devices[0]));
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* device_count) {
    if (device_count == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *device_count = static_cast<unsigned int>(sizeof(g_devices) / sizeof(g_devices[0]));
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
    if (device == nullptr || index >= sizeof(g_devices) / sizeof(g_devices[0])) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *device = &g_devices[index];
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    if (device == nullptr || index >= sizeof(g_devices) / sizeof(g_devices[0])) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *device = &g_devices[index];
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) {
    if (device == nullptr || index == nullptr || !is_known_device(device)) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    *index = device->index;
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (device == nullptr || memory == nullptr || !is_known_device(device)) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    memory->total = 16384;
    memory->free = 12288;
    memory->used = 4096;
    return NVML_SUCCESS;
}

extern "C" nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
    if (device == nullptr || memory == nullptr || !is_known_device(device) ||
        memory->version != nvmlMemory_v2) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    memory->total = 16384;
    memory->reserved = 256;
    memory->free = 12288;
    memory->used = 4096;
    return NVML_SUCCESS;
}
// NOLINTEND(readability-identifier-naming)
