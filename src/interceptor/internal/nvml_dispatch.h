#pragma once

#include <nvml.h>

#include <dlfcn.h>

#include <mutex>

namespace glimmer::interceptor {

using NvmlInitFunction = nvmlReturn_t (*)();
using NvmlInitWithFlagsFunction = nvmlReturn_t (*)(unsigned int flags);
using NvmlShutdownFunction = nvmlReturn_t (*)();
using NvmlDeviceGetCountFunction = nvmlReturn_t (*)(unsigned int* device_count);
using NvmlDeviceGetHandleByIndexFunction = nvmlReturn_t (*)(unsigned int index,
                                                            nvmlDevice_t* device);
using NvmlDeviceGetIndexFunction = nvmlReturn_t (*)(nvmlDevice_t device, unsigned int* index);
using NvmlDeviceGetMemoryInfoFunction = nvmlReturn_t (*)(nvmlDevice_t device, nvmlMemory_t* memory);
using NvmlDeviceGetMemoryInfoV2Function = nvmlReturn_t (*)(nvmlDevice_t device,
                                                           nvmlMemory_v2_t* memory);

struct NvmlFunctionTable {
    NvmlInitFunction init = nullptr;
    NvmlInitWithFlagsFunction init_with_flags = nullptr;
    NvmlShutdownFunction shutdown = nullptr;
    NvmlDeviceGetCountFunction device_get_count = nullptr;
    NvmlDeviceGetHandleByIndexFunction device_get_handle_by_index = nullptr;
    NvmlDeviceGetIndexFunction device_get_index = nullptr;
    NvmlDeviceGetMemoryInfoFunction device_get_memory_info = nullptr;
    NvmlDeviceGetMemoryInfoV2Function device_get_memory_info_v2 = nullptr;
};

[[nodiscard]] bool is_inside_nvml_call() noexcept;

class NvmlDispatch {
   public:
    NvmlDispatch() = default;
    explicit NvmlDispatch(NvmlFunctionTable functions) noexcept;

    NvmlDispatch(const NvmlDispatch&) = delete;
    NvmlDispatch& operator=(const NvmlDispatch&) = delete;

    ~NvmlDispatch();

    [[nodiscard]] bool initialize();
    [[nodiscard]] nvmlReturn_t init() const;
    [[nodiscard]] nvmlReturn_t init_with_flags(unsigned int flags) const;
    [[nodiscard]] nvmlReturn_t shutdown() const;
    [[nodiscard]] nvmlReturn_t device_get_count(unsigned int* device_count) const;
    [[nodiscard]] nvmlReturn_t device_get_handle_by_index(unsigned int index,
                                                          nvmlDevice_t* device) const;
    [[nodiscard]] nvmlReturn_t device_get_index(nvmlDevice_t device, unsigned int* index) const;
    [[nodiscard]] nvmlReturn_t device_get_memory_info(nvmlDevice_t device,
                                                      nvmlMemory_t* memory) const;
    [[nodiscard]] nvmlReturn_t device_get_memory_info_v2(nvmlDevice_t device,
                                                         nvmlMemory_v2_t* memory) const;

    [[nodiscard]] bool has_init() const;
    [[nodiscard]] bool has_init_with_flags() const;
    [[nodiscard]] bool has_shutdown() const;
    [[nodiscard]] bool has_device_get_count() const;
    [[nodiscard]] bool has_device_get_handle_by_index() const;
    [[nodiscard]] bool has_device_get_index() const;
    [[nodiscard]] bool has_device_get_memory_info() const;
    [[nodiscard]] bool has_device_get_memory_info_v2() const;

   private:
    [[nodiscard]] void* load_symbol(const char* name) const;

    mutable std::mutex mutex_;
    void* library_handle_ = nullptr;
    NvmlInitFunction init_ = nullptr;
    NvmlInitWithFlagsFunction init_with_flags_ = nullptr;
    NvmlShutdownFunction shutdown_ = nullptr;
    NvmlDeviceGetCountFunction device_get_count_ = nullptr;
    NvmlDeviceGetHandleByIndexFunction device_get_handle_by_index_ = nullptr;
    NvmlDeviceGetIndexFunction device_get_index_ = nullptr;
    NvmlDeviceGetMemoryInfoFunction device_get_memory_info_ = nullptr;
    NvmlDeviceGetMemoryInfoV2Function device_get_memory_info_v2_ = nullptr;
};

}  // namespace glimmer::interceptor
