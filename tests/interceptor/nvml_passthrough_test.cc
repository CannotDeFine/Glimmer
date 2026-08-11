#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <nvml.h>

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

namespace {

using NvmlInitFunction = nvmlReturn_t (*)();
using NvmlShutdownFunction = nvmlReturn_t (*)();
using NvmlDeviceGetHandleByIndexFunction = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using NvmlDeviceGetMemoryInfoFunction = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
using NvmlDeviceGetMemoryInfoV2Function = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_v2_t*);

template <typename Function>
Function resolve(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

}  // namespace

int main() {
    unsetenv("GLIMMER_MEMORY_LIMIT_BYTES");
    unsetenv("GLIMMER_QUOTA_MODE");
    unsetenv("GLIMMER_QUOTA_TENANT_ID");

    const NvmlInitFunction init = resolve<NvmlInitFunction>("nvmlInit_v2");
    const NvmlShutdownFunction shutdown = resolve<NvmlShutdownFunction>("nvmlShutdown");
    const NvmlDeviceGetHandleByIndexFunction get_handle =
        resolve<NvmlDeviceGetHandleByIndexFunction>("nvmlDeviceGetHandleByIndex_v2");
    const NvmlDeviceGetMemoryInfoFunction get_memory_info =
        resolve<NvmlDeviceGetMemoryInfoFunction>("nvmlDeviceGetMemoryInfo");
    const NvmlDeviceGetMemoryInfoV2Function get_memory_info_v2 =
        resolve<NvmlDeviceGetMemoryInfoV2Function>("nvmlDeviceGetMemoryInfo_v2");
    if (init == nullptr || shutdown == nullptr || get_handle == nullptr ||
        get_memory_info == nullptr || get_memory_info_v2 == nullptr) {
        std::cerr << "NVML passthrough symbols were not exported\n";
        return EXIT_FAILURE;
    }
    if (init() != NVML_SUCCESS) {
        std::cerr << "NVML initialization failed\n";
        return EXIT_FAILURE;
    }

    nvmlDevice_t device = nullptr;
    nvmlMemory_t memory{};
    nvmlMemory_v2_t memory_v2{};
    memory_v2.version = nvmlMemory_v2;
    const bool passed = get_handle(0, &device) == NVML_SUCCESS &&
                        get_memory_info(device, &memory) == NVML_SUCCESS &&
                        get_memory_info_v2(device, &memory_v2) == NVML_SUCCESS &&
                        memory.total == 16384 && memory.free == 12288 && memory.used == 4096 &&
                        memory_v2.total == 16384 && memory_v2.free == 12288 &&
                        memory_v2.used == 4096 && memory_v2.reserved == 256;
    const bool shutdown_passed = shutdown() == NVML_SUCCESS;
    if (!passed || !shutdown_passed) {
        std::cerr << "NVML physical memory values were modified without a quota\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
