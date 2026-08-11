#include <nvml.h>

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

namespace {

using InitFunction = nvmlReturn_t (*)();
using ShutdownFunction = nvmlReturn_t (*)();
using GetCountFunction = nvmlReturn_t (*)(unsigned int*);
using GetHandleFunction = nvmlReturn_t (*)(unsigned int, nvmlDevice_t*);
using GetMemoryInfoFunction = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*);
using GetMemoryInfoV2Function = nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_v2_t*);

template <typename Function>
Function resolve(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

bool check(nvmlReturn_t result, const char* operation) {
    if (result == NVML_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with NVML status " << static_cast<int>(result) << '\n';
    return false;
}

}  // namespace

int main() {
    void* library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        std::cerr << "libnvidia-ml.so.1 is unavailable\n";
        return EXIT_FAILURE;
    }

    const InitFunction init = resolve<InitFunction>("nvmlInit_v2");
    const ShutdownFunction shutdown = resolve<ShutdownFunction>("nvmlShutdown");
    const GetCountFunction get_count = resolve<GetCountFunction>("nvmlDeviceGetCount_v2");
    const GetHandleFunction get_handle =
        resolve<GetHandleFunction>("nvmlDeviceGetHandleByIndex_v2");
    const GetMemoryInfoFunction get_memory_info =
        resolve<GetMemoryInfoFunction>("nvmlDeviceGetMemoryInfo");
    const GetMemoryInfoV2Function get_memory_info_v2 =
        resolve<GetMemoryInfoV2Function>("nvmlDeviceGetMemoryInfo_v2");
    bool passed = init != nullptr && shutdown != nullptr && get_count != nullptr &&
                  get_handle != nullptr && get_memory_info != nullptr &&
                  get_memory_info_v2 != nullptr;
    passed &= check(init == nullptr ? NVML_ERROR_UNKNOWN : init(), "nvmlInit_v2");
    unsigned int device_count = 0;
    nvmlDevice_t device = nullptr;
    nvmlMemory_t memory{};
    nvmlMemory_v2_t memory_v2{};
    memory_v2.version = nvmlMemory_v2;
    passed &= check(get_count == nullptr ? NVML_ERROR_UNKNOWN : get_count(&device_count),
                    "nvmlDeviceGetCount_v2");
    passed &= device_count > 0;
    passed &= check(get_handle == nullptr ? NVML_ERROR_UNKNOWN : get_handle(0, &device),
                    "nvmlDeviceGetHandleByIndex_v2");
    passed &=
        check(get_memory_info == nullptr ? NVML_ERROR_UNKNOWN : get_memory_info(device, &memory),
              "nvmlDeviceGetMemoryInfo");
    passed &= check(
        get_memory_info_v2 == nullptr ? NVML_ERROR_UNKNOWN : get_memory_info_v2(device, &memory_v2),
        "nvmlDeviceGetMemoryInfo_v2");
    passed &= memory.total == 8388608ULL && memory.free <= memory.total &&
              memory.used == memory.total - memory.free;
    passed &= memory_v2.total == memory.total && memory_v2.free == memory.free &&
              memory_v2.used == memory.used && memory_v2.reserved == 0;
    passed &= check(shutdown == nullptr ? NVML_ERROR_UNKNOWN : shutdown(), "nvmlShutdown");
    dlclose(library);
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
