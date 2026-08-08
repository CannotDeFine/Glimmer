#include "glimmer/control/process_memory_quota.h"

#include <cuda.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <unordered_map>

namespace {

using glimmer::control::ProcessMemoryQuota;
using glimmer::core::MemoryBytes;

using MemAllocFunction = CUresult (*)(CUdeviceptr* device_pointer, std::size_t memory_bytes);
using MemFreeFunction = CUresult (*)(CUdeviceptr device_pointer);
using MemGetInfoFunction = CUresult (*)(std::size_t* free_bytes, std::size_t* total_bytes);

enum class QuotaMode : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalidConfiguration,
};

class DriverDispatch {
   public:
    DriverDispatch() = default;

    DriverDispatch(const DriverDispatch&) = delete;
    DriverDispatch& operator=(const DriverDispatch&) = delete;

    ~DriverDispatch() {
        if (library_handle_ != nullptr) {
            dlclose(library_handle_);
        }
    }

    [[nodiscard]] bool initialize() {
        library_handle_ = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (library_handle_ == nullptr) {
            return false;
        }

        mem_alloc_ = load_symbol<MemAllocFunction>("cuMemAlloc_v2");
        mem_free_ = load_symbol<MemFreeFunction>("cuMemFree_v2");
        mem_get_info_ = load_symbol<MemGetInfoFunction>("cuMemGetInfo_v2");
        if (mem_alloc_ != nullptr && mem_free_ != nullptr && mem_get_info_ != nullptr) {
            return true;
        }

        dlclose(library_handle_);
        library_handle_ = nullptr;
        return false;
    }

    [[nodiscard]] CUresult mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) const {
        return mem_alloc_(device_pointer, memory_bytes);
    }

    [[nodiscard]] CUresult mem_free(CUdeviceptr device_pointer) const {
        return mem_free_(device_pointer);
    }

    [[nodiscard]] CUresult mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) const {
        return mem_get_info_(free_bytes, total_bytes);
    }

   private:
    template <typename Function>
    [[nodiscard]] Function load_symbol(const char* name) const {
        dlerror();
        void* symbol = dlsym(library_handle_, name);
        if (dlerror() != nullptr) {
            return nullptr;
        }
        return reinterpret_cast<Function>(symbol);
    }

    void* library_handle_ = nullptr;
    MemAllocFunction mem_alloc_ = nullptr;
    MemFreeFunction mem_free_ = nullptr;
    MemGetInfoFunction mem_get_info_ = nullptr;
};

struct InterceptorState {
    std::once_flag initialization_once;
    DriverDispatch driver;
    QuotaMode quota_mode = QuotaMode::kDisabled;
    std::unique_ptr<ProcessMemoryQuota> quota;
    std::mutex allocation_mutex;
    std::unordered_map<CUdeviceptr, MemoryBytes> allocations;
    bool is_accounting_degraded = false;
    bool is_driver_ready = false;
};

[[nodiscard]] InterceptorState& get_state() {
    static InterceptorState state;
    return state;
}

[[nodiscard]] std::optional<MemoryBytes> read_quota_limit() {
    const char* value = std::getenv("GLIMMER_MEMORY_LIMIT_BYTES");
    if (value == nullptr) {
        return std::nullopt;
    }

    const char* end = value + std::strlen(value);
    MemoryBytes limit_bytes = 0;
    const auto [parsed_end, error] = std::from_chars(value, end, limit_bytes);
    if (error != std::errc{} || parsed_end != end) {
        return std::nullopt;
    }
    return limit_bytes;
}

void initialize_state(InterceptorState& state) noexcept {
    state.is_driver_ready = state.driver.initialize();

    const char* configured_limit = std::getenv("GLIMMER_MEMORY_LIMIT_BYTES");
    if (configured_limit == nullptr) {
        state.quota_mode = QuotaMode::kDisabled;
        return;
    }

    const std::optional<MemoryBytes> limit_bytes = read_quota_limit();
    if (!limit_bytes.has_value()) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        return;
    }

    try {
        state.quota = std::make_unique<ProcessMemoryQuota>(*limit_bytes);
        state.quota_mode = QuotaMode::kEnabled;
    } catch (...) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
    }
}

[[nodiscard]] bool ensure_initialized(InterceptorState& state) {
    std::call_once(state.initialization_once, [&state] { initialize_state(state); });
    return state.is_driver_ready;
}

template <typename Function>
CUresult guard_cuda_boundary(Function&& function) noexcept {
    try {
        return function();
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

}  // namespace

namespace {

CUresult intercept_mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    InterceptorState& state = get_state();
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (device_pointer == nullptr || state.quota_mode == QuotaMode::kDisabled) {
        return state.driver.mem_alloc(device_pointer, memory_bytes);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    {
        std::scoped_lock lock(state.allocation_mutex);
        if (state.is_accounting_degraded) {
            return CUDA_ERROR_UNKNOWN;
        }
    }

    if (state.quota == nullptr) {
        return CUDA_ERROR_UNKNOWN;
    }

    auto reservation = state.quota->try_reserve(memory_bytes);
    if (!reservation.has_value()) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    const CUresult allocation_result = state.driver.mem_alloc(device_pointer, memory_bytes);
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }

    bool is_recorded = false;
    try {
        std::scoped_lock lock(state.allocation_mutex);
        is_recorded = state.allocations.emplace(*device_pointer, memory_bytes).second;
    } catch (...) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            const bool committed = reservation->commit();
            std::scoped_lock lock(state.allocation_mutex);
            state.is_accounting_degraded = true;
            return committed ? cleanup_result : CUDA_ERROR_UNKNOWN;
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    if (!is_recorded) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            const bool committed = reservation->commit();
            std::scoped_lock lock(state.allocation_mutex);
            state.is_accounting_degraded = true;
            return committed ? cleanup_result : CUDA_ERROR_UNKNOWN;
        }
        return CUDA_ERROR_UNKNOWN;
    }

    if (!reservation->commit()) {
        std::scoped_lock lock(state.allocation_mutex);
        state.is_accounting_degraded = true;
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_mem_free(CUdeviceptr device_pointer) {
    InterceptorState& state = get_state();
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    const CUresult free_result = state.driver.mem_free(device_pointer);
    if (free_result != CUDA_SUCCESS || state.quota_mode != QuotaMode::kEnabled) {
        return free_result;
    }

    std::optional<MemoryBytes> allocation_size;
    {
        std::scoped_lock lock(state.allocation_mutex);
        const auto allocation = state.allocations.find(device_pointer);
        if (allocation != state.allocations.end()) {
            allocation_size = allocation->second;
            state.allocations.erase(allocation);
        }
    }

    if (allocation_size.has_value() &&
        (state.quota == nullptr || !state.quota->release(*allocation_size))) {
        std::scoped_lock lock(state.allocation_mutex);
        state.is_accounting_degraded = true;
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) {
    InterceptorState& state = get_state();
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    const CUresult info_result = state.driver.mem_get_info(free_bytes, total_bytes);
    if (info_result != CUDA_SUCCESS || state.quota_mode != QuotaMode::kEnabled) {
        return info_result;
    }

    if (state.quota == nullptr) {
        return CUDA_ERROR_UNKNOWN;
    }

    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(*total_bytes, *free_bytes);
    *total_bytes = static_cast<std::size_t>(memory_info.total_bytes);
    *free_bytes = static_cast<std::size_t>(memory_info.free_bytes);
    return CUDA_SUCCESS;
}

}  // namespace

extern "C" CUresult CUDAAPI cuMemAlloc_v2(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    return guard_cuda_boundary([device_pointer, memory_bytes] {
        return intercept_mem_alloc(device_pointer, memory_bytes);
    });
}

extern "C" CUresult CUDAAPI cuMemFree_v2(CUdeviceptr device_pointer) {
    return guard_cuda_boundary([device_pointer] { return intercept_mem_free(device_pointer); });
}

extern "C" CUresult CUDAAPI cuMemGetInfo_v2(std::size_t* free_bytes, std::size_t* total_bytes) {
    return guard_cuda_boundary(
        [free_bytes, total_bytes] { return intercept_mem_get_info(free_bytes, total_bytes); });
}
