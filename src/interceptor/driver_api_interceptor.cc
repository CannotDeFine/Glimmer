#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "glimmer/control/process_memory_quota.h"
#include "glimmer/control/shared_memory_quota.h"

#include "internal/allocation_registry.h"
#include "internal/diagnostics.h"
#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/runtime_api_bridge.h"
#include "internal/symbol_registry.h"
#include "internal/nvml_dispatch.h"

#include <cuda.h>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace glimmer::interceptor {

namespace {

using glimmer::control::DeviceId;
using glimmer::control::ProcessMemoryQuota;
using glimmer::control::QuotaStore;
using glimmer::control::SharedMemoryQuota;
using glimmer::control::SharedMemoryQuotaConfig;
using glimmer::control::SharedMemoryQuotaError;
using glimmer::core::MemoryBytes;
using glimmer::interceptor::AllocationRegistry;
using glimmer::interceptor::AllocationScope;
using glimmer::interceptor::AsyncStreamIdentity;
using glimmer::interceptor::DriverDispatch;

enum class QuotaMode : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalidConfiguration,
};

struct InterceptorState {
    std::once_flag initialization_once;
    DriverDispatch driver;
    NvmlDispatch nvml;
    QuotaMode quota_mode = QuotaMode::kDisabled;
    std::unique_ptr<QuotaStore> quota;
    glimmer::interceptor::AllocationRegistry allocations;
    bool is_driver_ready = false;
    bool uses_shared_quota = false;
};

thread_local bool g_is_inside_proc_address_v2 = false;

class ProcAddressV2Scope {
   public:
    ProcAddressV2Scope() : previous_state_(g_is_inside_proc_address_v2) {
        g_is_inside_proc_address_v2 = true;
    }

    ~ProcAddressV2Scope() {
        g_is_inside_proc_address_v2 = previous_state_;
    }

   private:
    bool previous_state_;
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

[[nodiscard]] std::optional<glimmer::control::DeviceId> read_quota_device() {
    const char* value = std::getenv("GLIMMER_QUOTA_DEVICE_ID");
    if (value == nullptr) {
        return glimmer::control::QuotaStore::default_device;
    }

    const char* end = value + std::strlen(value);
    std::int64_t device_id = 0;
    const auto [parsed_end, error] = std::from_chars(value, end, device_id);
    if (error != std::errc{} || parsed_end != end || device_id < 0 ||
        device_id > std::numeric_limits<glimmer::control::DeviceId>::max()) {
        return std::nullopt;
    }
    return static_cast<glimmer::control::DeviceId>(device_id);
}

[[nodiscard]] std::optional<MemoryBytes> checked_multiply(MemoryBytes left, MemoryBytes right) {
    if (right != 0 && left > std::numeric_limits<MemoryBytes>::max() / right) {
        return std::nullopt;
    }
    return left * right;
}

void initialize_state(InterceptorState& state) noexcept {
    state.is_driver_ready = state.driver.initialize();
    if (!state.is_driver_ready) {
        glimmer::interceptor::report_diagnostic(
            "[glimmer] CUDA Driver symbol initialization failed\n");
    }

    const char* configured_mode = std::getenv("GLIMMER_QUOTA_MODE");
    if (configured_mode != nullptr && std::strcmp(configured_mode, "process") != 0 &&
        std::strcmp(configured_mode, "shared") != 0) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        glimmer::interceptor::report_diagnostic("[glimmer] GLIMMER_QUOTA_MODE is invalid\n");
        return;
    }

    const char* configured_limit = std::getenv("GLIMMER_MEMORY_LIMIT_BYTES");
    if (configured_limit == nullptr &&
        (configured_mode == nullptr || std::strcmp(configured_mode, "process") == 0)) {
        state.quota_mode = QuotaMode::kDisabled;
        return;
    }

    const std::optional<MemoryBytes> limit_bytes = read_quota_limit();
    if (!limit_bytes.has_value()) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        glimmer::interceptor::report_diagnostic(
            "[glimmer] GLIMMER_MEMORY_LIMIT_BYTES is invalid\n");
        return;
    }

    try {
        if (configured_mode != nullptr && std::strcmp(configured_mode, "shared") == 0) {
            const char* tenant_id = std::getenv("GLIMMER_QUOTA_TENANT_ID");
            if (tenant_id == nullptr || *tenant_id == '\0') {
                state.quota_mode = QuotaMode::kInvalidConfiguration;
                glimmer::interceptor::report_diagnostic(
                    "[glimmer] GLIMMER_QUOTA_TENANT_ID is required in shared mode\n");
                return;
            }

            const std::optional<glimmer::control::DeviceId> device_id = read_quota_device();
            if (!device_id.has_value()) {
                state.quota_mode = QuotaMode::kInvalidConfiguration;
                glimmer::interceptor::report_diagnostic(
                    "[glimmer] GLIMMER_QUOTA_DEVICE_ID is invalid\n");
                return;
            }

            SharedMemoryQuotaConfig config{
                .tenant_id = tenant_id,
                .device = *device_id,
                .limit_bytes = *limit_bytes,
            };
            SharedMemoryQuotaError error = SharedMemoryQuotaError::kNone;
            state.quota = SharedMemoryQuota::open(config, &error);
            if (state.quota == nullptr) {
                state.quota_mode = QuotaMode::kInvalidConfiguration;
                glimmer::interceptor::report_diagnostic(
                    "[glimmer] failed to initialize the shared memory quota\n");
                return;
            }
            state.uses_shared_quota = true;
        } else {
            state.quota = std::make_unique<ProcessMemoryQuota>(*limit_bytes);
        }
        state.quota_mode = QuotaMode::kEnabled;
    } catch (...) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        glimmer::interceptor::report_diagnostic(
            "[glimmer] failed to initialize the memory quota\n");
    }
}

void ensure_state_initialized(InterceptorState& state) {
    std::call_once(state.initialization_once, [&state] { initialize_state(state); });
}

[[nodiscard]] bool ensure_initialized(InterceptorState& state) {
    ensure_state_initialized(state);
    return state.is_driver_ready;
}

struct ContextIdentity {
    CUcontext context = nullptr;
    CUdevice device = 0;
};

[[nodiscard]] std::optional<ContextIdentity> capture_context_identity(
    InterceptorState& state) noexcept {
    if (!state.driver.has_context_queries()) {
        return std::nullopt;
    }

    CUcontext context = nullptr;
    if (state.driver.context_get_current(&context) != CUDA_SUCCESS || context == nullptr) {
        return std::nullopt;
    }

    CUdevice device = 0;
    if (state.driver.context_get_device(&device) != CUDA_SUCCESS || device < 0) {
        return std::nullopt;
    }
    return ContextIdentity{.context = context, .device = device};
}

[[nodiscard]] std::optional<ContextIdentity> capture_stream_identity(
    InterceptorState& state, CUstream stream, bool per_thread_default_stream) noexcept {
    if (stream == nullptr) {
        return capture_context_identity(state);
    }
    if (per_thread_default_stream ? !state.driver.has_stream_identity_ptsz()
                                  : !state.driver.has_stream_identity()) {
        return std::nullopt;
    }

    CUcontext context = nullptr;
    const CUresult context_result = per_thread_default_stream
                                        ? state.driver.stream_get_context_ptsz(stream, &context)
                                        : state.driver.stream_get_context(stream, &context);
    if (context_result != CUDA_SUCCESS || context == nullptr) {
        return std::nullopt;
    }

    CUdevice device = 0;
    const CUresult device_result = per_thread_default_stream
                                       ? state.driver.stream_get_device_ptsz(stream, &device)
                                       : state.driver.stream_get_device(stream, &device);
    if (device_result != CUDA_SUCCESS || device < 0) {
        return std::nullopt;
    }
    return ContextIdentity{.context = context, .device = device};
}

[[nodiscard]] glimmer::interceptor::AllocationIdentity make_allocation_identity(
    CUdeviceptr device_pointer, const ContextIdentity& context) noexcept {
    return glimmer::interceptor::AllocationIdentity{
        .device_pointer = device_pointer, .context = context.context, .device = context.device};
}

[[nodiscard]] glimmer::interceptor::AllocationIdentity make_allocation_identity(
    const void* device_pointer, const ContextIdentity& context) noexcept {
    return make_allocation_identity(
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_pointer)), context);
}

[[nodiscard]] AsyncStreamIdentity make_async_stream_identity(CUstream stream,
                                                             const ContextIdentity& context,
                                                             bool per_thread_default_stream) {
    return AsyncStreamIdentity{
        .stream = stream,
        .context = context.context,
        .per_thread_default_stream = stream == nullptr && per_thread_default_stream,
        .thread_id = std::this_thread::get_id(),
    };
}

template <typename Availability, typename Invocation>
[[nodiscard]] CUresult forward_optional_driver_call(InterceptorState& state,
                                                    Availability&& is_available,
                                                    Invocation&& invoke) {
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return std::forward<Invocation>(invoke)();
    }
    if (!ensure_initialized(state) || !std::forward<Availability>(is_available)()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return std::forward<Invocation>(invoke)();
}

}  // namespace

}  // namespace glimmer::interceptor

namespace glimmer::interceptor {

cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                     RuntimeMallocFunction allocate, RuntimeFreeFunction release,
                                     RuntimeGetDeviceFunction get_device) {
    InterceptorState& state = get_state();
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? allocate(device_pointer, memory_bytes)
                   : cudaErrorUnknown;
    }

    if (state.quota_mode == QuotaMode::kDisabled) {
        return allocate(device_pointer, memory_bytes);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }

    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }

    if (state.quota == nullptr) {
        return cudaErrorUnknown;
    }
    if (!state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }

    DeviceId device = QuotaStore::default_device;
    bool device_resolved = false;
    if (get_device != nullptr) {
        int runtime_device = 0;
        if (get_device(&runtime_device) == cudaSuccess && runtime_device >= 0) {
            device = static_cast<DeviceId>(runtime_device);
            device_resolved = true;
        }
    }

    const std::optional<ContextIdentity> context_before_allocation =
        capture_context_identity(state);
    if (context_before_allocation.has_value()) {
        device = static_cast<DeviceId>(context_before_allocation->device);
        device_resolved = true;
    } else if (state.uses_shared_quota && !device_resolved) {
        return cudaErrorInvalidValue;
    }

    auto reservation = state.quota->try_reserve(device, memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? cudaErrorMemoryAllocation : cudaErrorUnknown;
    }

    const cudaError_t allocation_result = allocate(device_pointer, memory_bytes);
    if (allocation_result != cudaSuccess) {
        return allocation_result;
    }

    if (*device_pointer == nullptr) {
        // A successful driver result with no pointer is an ambiguous external
        // state. Keep the reservation charged because the allocation cannot
        // be safely released without a pointer.
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value() || static_cast<DeviceId>(context->device) != device) {
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return cudaErrorUnknown;
    }
    if (!reservation->commit()) {
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  memory_bytes)) {
        // A failed record is an accounting invariant violation, even when
        // external cleanup succeeds. Stop admitting work before attempting
        // the compensating cleanup.
        state.allocations.mark_accounting_degraded();
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(context->device), memory_bytes)) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_free(void* device_pointer, RuntimeFreeFunction release) {
    InterceptorState& state = get_state();
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? release(device_pointer)
                                                                    : cudaErrorUnknown;
    }

    if (state.quota_mode != QuotaMode::kEnabled) {
        return release(device_pointer);
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        const cudaError_t free_result = release(device_pointer);
        if (free_result == cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return free_result;
    }

    auto release_state =
        state.allocations.begin_release(make_allocation_identity(device_pointer, *context));
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        release_state = state.allocations.begin_release_by_pointer(
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_pointer)),
            context->device);
    }
    const auto [release_status, release_ticket] = release_state;
    if (release_status == AllocationRegistry::ReleaseStatus::kInProgress) {
        return cudaErrorInvalidValue;
    }
    if (release_status == AllocationRegistry::ReleaseStatus::kUnknown ||
        !release_ticket.has_value()) {
        return release(device_pointer);
    }

    const cudaError_t free_result = release(device_pointer);
    if (free_result != cudaSuccess) {
        state.allocations.cancel_release(*release_ticket);
        return free_result;
    }

    const bool is_released =
        state.quota != nullptr &&
        state.quota->release(static_cast<DeviceId>(release_ticket->identity.device),
                             release_ticket->memory_bytes);
    const bool is_completed = state.allocations.complete_release(*release_ticket);
    if (!is_released || !is_completed) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes,
                                           RuntimeMemGetInfoFunction query) {
    InterceptorState& state = get_state();
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? query(free_bytes, total_bytes)
                                                                    : cudaErrorUnknown;
    }

    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }

    const cudaError_t info_result = query(free_bytes, total_bytes);
    if (info_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return info_result;
    }

    if (state.quota == nullptr) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value() && state.uses_shared_quota) {
        return cudaErrorInvalidValue;
    }
    const DeviceId device =
        context.has_value() ? static_cast<DeviceId>(context->device) : QuotaStore::default_device;
    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(device, *total_bytes, *free_bytes);
    if (!state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }
    *total_bytes = static_cast<std::size_t>(memory_info.total_bytes);
    *free_bytes = static_cast<std::size_t>(memory_info.free_bytes);
    return cudaSuccess;
}

}  // namespace glimmer::interceptor

namespace glimmer::interceptor {

template <typename AllocateFunction>
CUresult intercept_tracked_allocation(CUdeviceptr* device_pointer, MemoryBytes memory_bytes,
                                      AllocateFunction&& allocate) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (state.quota_mode == QuotaMode::kDisabled) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (state.quota == nullptr) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    auto reservation =
        state.quota->try_reserve(static_cast<DeviceId>(context->device), memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    }

    const CUresult allocation_result = std::forward<AllocateFunction>(allocate)();
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }
    if (*device_pointer == 0) {
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  memory_bytes)) {
        state.allocations.mark_accounting_degraded();
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(context->device), memory_bytes)) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

CUresult cleanup_async_allocation(InterceptorState& state, CUdeviceptr device_pointer,
                                  CUstream stream, bool per_thread_default_stream) {
    const CUresult free_result = per_thread_default_stream
                                     ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                     : state.driver.mem_free_async(device_pointer, stream);
    if (free_result != CUDA_SUCCESS) {
        return free_result;
    }
    return per_thread_default_stream ? state.driver.stream_synchronize_ptsz(stream)
                                     : state.driver.stream_synchronize(stream);
}

template <typename AllocateFunction>
CUresult intercept_async_tracked_allocation(CUdeviceptr* device_pointer, MemoryBytes memory_bytes,
                                            CUstream stream, bool per_thread_default_stream,
                                            AllocateFunction&& allocate) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (state.quota_mode == QuotaMode::kDisabled) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (per_thread_default_stream ? !state.driver.has_mem_free_async_ptsz()
                                  : !state.driver.has_mem_free_async()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, stream, per_thread_default_stream);
    if (!stream_identity.has_value()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    auto reservation =
        state.quota->try_reserve(static_cast<DeviceId>(stream_identity->device), memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    }

    const CUresult allocation_result = std::forward<AllocateFunction>(allocate)();
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }
    if (device_pointer == nullptr || *device_pointer == 0) {
        // A successful external allocation with no pointer is ambiguous. The
        // reservation must remain charged because there is no safe pointer to
        // pass to the asynchronous free path.
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result =
            cleanup_async_allocation(state, *device_pointer, stream, per_thread_default_stream);
        if (cleanup_result != CUDA_SUCCESS) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    const auto identity = make_allocation_identity(*device_pointer, *stream_identity);
    if (!state.allocations.record(identity, memory_bytes, AllocationScope::kContextIndependent)) {
        const CUresult cleanup_result =
            cleanup_async_allocation(state, *device_pointer, stream, per_thread_default_stream);
        state.allocations.mark_accounting_degraded();
        if (cleanup_result != CUDA_SUCCESS) {
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(stream_identity->device), memory_bytes)) {
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

void finalize_async_release(InterceptorState& state, DeviceId device, MemoryBytes released_bytes) {
    if (released_bytes == 0 || state.allocations.is_accounting_degraded()) {
        return;
    }
    if (state.quota == nullptr || !state.quota->release(device, released_bytes)) {
        state.allocations.mark_accounting_degraded();
    }
}

void finalize_async_releases(
    InterceptorState& state,
    const std::vector<AllocationRegistry::DeviceReleaseSummary>& summaries) {
    for (const auto& summary : summaries) {
        finalize_async_release(state, static_cast<DeviceId>(summary.device), summary.memory_bytes);
    }
}

[[nodiscard]] AsyncStreamIdentity make_runtime_async_stream_identity(
    cudaStream_t stream, const ContextIdentity& context) {
    const CUstream driver_stream = reinterpret_cast<CUstream>(stream);
    // Runtime does not expose whether a null stream is legacy or per-thread.
    // Treat it as per-thread to keep records from different calling threads
    // isolated; explicit streams are identified by their owning context.
    return make_async_stream_identity(driver_stream, context, stream == nullptr);
}

[[nodiscard]] cudaError_t cleanup_runtime_async_allocation(
    void* device_pointer, cudaStream_t stream, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize) {
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const cudaError_t free_result = release(device_pointer, stream);
    if (free_result != cudaSuccess) {
        return free_result;
    }
    return synchronize == nullptr ? cudaErrorUnknown : synchronize();
}

template <typename Allocate>
cudaError_t intercept_runtime_malloc_async_impl(void** device_pointer, std::size_t memory_bytes,
                                                cudaStream_t stream, Allocate allocate,
                                                RuntimeFreeAsyncFunction release,
                                                RuntimeDeviceSynchronizeFunction synchronize,
                                                RuntimeGetDeviceFunction get_device) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? allocate() : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return allocate();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }

    DeviceId device = QuotaStore::default_device;
    bool device_resolved = false;
    if (get_device != nullptr) {
        int runtime_device = 0;
        if (get_device(&runtime_device) == cudaSuccess && runtime_device >= 0) {
            device = static_cast<DeviceId>(runtime_device);
            device_resolved = true;
        }
    }
    const CUstream driver_stream = reinterpret_cast<CUstream>(stream);
    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, driver_stream, false);
    if (stream_identity.has_value()) {
        device = static_cast<DeviceId>(stream_identity->device);
        device_resolved = true;
    } else if (state.uses_shared_quota && !device_resolved) {
        return cudaErrorInvalidValue;
    }

    auto reservation = state.quota->try_reserve(device, memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? cudaErrorMemoryAllocation : cudaErrorUnknown;
    }
    const cudaError_t allocation_result = allocate();
    if (allocation_result != cudaSuccess) {
        return allocation_result;
    }
    if (*device_pointer == nullptr) {
        // A successful external allocation with no pointer is ambiguous. The
        // reservation must remain charged because there is no safe pointer to
        // pass to the asynchronous free path.
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }

    const std::optional<ContextIdentity> final_identity =
        capture_stream_identity(state, driver_stream, false);
    if (!final_identity.has_value() || static_cast<DeviceId>(final_identity->device) != device) {
        const cudaError_t cleanup_result =
            cleanup_runtime_async_allocation(*device_pointer, stream, release, synchronize);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    if (!reservation->commit()) {
        const cudaError_t cleanup_result =
            cleanup_runtime_async_allocation(*device_pointer, stream, release, synchronize);
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *final_identity),
                                  memory_bytes, AllocationScope::kContextIndependent)) {
        const cudaError_t cleanup_result =
            cleanup_runtime_async_allocation(*device_pointer, stream, release, synchronize);
        state.allocations.mark_accounting_degraded();
        if (cleanup_result != cudaSuccess) {
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(final_identity->device), memory_bytes)) {
            return cudaErrorUnknown;
        }
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_malloc_async(void** device_pointer, std::size_t memory_bytes,
                                           cudaStream_t stream, RuntimeMallocAsyncFunction allocate,
                                           RuntimeFreeAsyncFunction release,
                                           RuntimeDeviceSynchronizeFunction synchronize,
                                           RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_async_impl(
        device_pointer, memory_bytes, stream,
        [device_pointer, memory_bytes, stream, allocate] {
            return allocate(device_pointer, memory_bytes, stream);
        },
        release, synchronize, get_device);
}

cudaError_t intercept_runtime_malloc_from_pool_async(
    void** device_pointer, std::size_t memory_bytes, cudaMemPool_t pool, cudaStream_t stream,
    RuntimeMallocFromPoolAsyncFunction allocate, RuntimeFreeAsyncFunction release,
    RuntimeDeviceSynchronizeFunction synchronize, RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_async_impl(
        device_pointer, memory_bytes, stream,
        [device_pointer, memory_bytes, pool, stream, allocate] {
            return allocate(device_pointer, memory_bytes, pool, stream);
        },
        release, synchronize, get_device);
}

cudaError_t intercept_runtime_mem_pool_import_from_shareable_handle(
    cudaMemPool_t* pool_out, void* handle, cudaMemAllocationHandleType handle_type,
    unsigned long long flags, RuntimeMemPoolImportFromShareableHandleFunction import_pool) {
    InterceptorState& state = get_state();
    if (pool_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (import_pool == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? import_pool(pool_out, handle, handle_type, flags)
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return cudaErrorNotSupported;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    return import_pool(pool_out, handle, handle_type, flags);
}

cudaError_t intercept_runtime_mem_pool_import_pointer(
    void** pointer_out, cudaMemPool_t pool, cudaMemPoolPtrExportData* share_data,
    RuntimeMemPoolImportPointerFunction import_pointer) {
    InterceptorState& state = get_state();
    if (pointer_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (import_pointer == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? import_pointer(pointer_out, pool, share_data)
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return cudaErrorNotSupported;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    return import_pointer(pointer_out, pool, share_data);
}

cudaError_t intercept_runtime_free_async(void* device_pointer, cudaStream_t stream,
                                         RuntimeFreeAsyncFunction release) {
    InterceptorState& state = get_state();
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? release(device_pointer, stream)
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return release(device_pointer, stream);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return release(device_pointer, stream);
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }

    const CUstream driver_stream = reinterpret_cast<CUstream>(stream);
    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, driver_stream, false);
    if (!stream_identity.has_value()) {
        const cudaError_t free_result = release(device_pointer, stream);
        if (free_result == cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return free_result;
    }
    const AsyncStreamIdentity async_stream =
        make_runtime_async_stream_identity(stream, *stream_identity);
    auto release_state = state.allocations.begin_async_release(
        make_allocation_identity(device_pointer, *stream_identity), async_stream);
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        release_state = state.allocations.begin_async_release_by_pointer(
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_pointer)),
            stream_identity->device, async_stream);
    }
    const auto [release_status, release_ticket] = release_state;
    if (release_status == AllocationRegistry::ReleaseStatus::kInProgress) {
        return cudaErrorInvalidValue;
    }
    if (release_status == AllocationRegistry::ReleaseStatus::kUnknown ||
        !release_ticket.has_value()) {
        return release(device_pointer, stream);
    }
    const cudaError_t free_result = release(device_pointer, stream);
    if (free_result != cudaSuccess) {
        state.allocations.cancel_release(*release_ticket);
        return free_result;
    }
    if (!state.allocations.commit_async_release(*release_ticket)) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_device_synchronize(RuntimeDeviceSynchronizeFunction synchronize) {
    InterceptorState& state = get_state();
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? synchronize()
                                                                    : cudaErrorUnknown;
    }
    const cudaError_t sync_result = synchronize();
    if (sync_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return sync_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto summaries =
        state.allocations.complete_async_releases_for_context_by_device(context->context);
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : sync_result;
}

cudaError_t intercept_runtime_stream_synchronize(cudaStream_t stream,
                                                 RuntimeStreamSynchronizeFunction synchronize) {
    InterceptorState& state = get_state();
    if (synchronize == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? synchronize(stream)
                                                                    : cudaErrorUnknown;
    }
    const cudaError_t sync_result = synchronize(stream);
    if (sync_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return sync_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context =
        capture_stream_identity(state, reinterpret_cast<CUstream>(stream), false);
    if (!context.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto summaries = state.allocations.complete_async_releases_for_stream_by_device(
        make_runtime_async_stream_identity(stream, *context));
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : sync_result;
}

cudaError_t intercept_runtime_stream_query(cudaStream_t stream, RuntimeStreamQueryFunction query) {
    InterceptorState& state = get_state();
    if (query == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? query(stream)
                                                                    : cudaErrorUnknown;
    }
    const cudaError_t query_result = query(stream);
    if (query_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return query_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context =
        capture_stream_identity(state, reinterpret_cast<CUstream>(stream), false);
    if (!context.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto summaries = state.allocations.complete_async_releases_for_stream_by_device(
        make_runtime_async_stream_identity(stream, *context));
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : query_result;
}

cudaError_t intercept_runtime_stream_destroy(cudaStream_t stream,
                                             RuntimeStreamDestroyFunction destroy) {
    InterceptorState& state = get_state();
    if (destroy == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? destroy(stream)
                                                                    : cudaErrorUnknown;
    }
    std::optional<AsyncStreamIdentity> async_stream;
    if (state.quota_mode == QuotaMode::kEnabled && !state.allocations.is_accounting_degraded()) {
        const std::optional<ContextIdentity> context =
            capture_stream_identity(state, reinterpret_cast<CUstream>(stream), false);
        if (!context.has_value()) {
            state.allocations.mark_accounting_degraded();
        } else {
            async_stream = make_runtime_async_stream_identity(stream, *context);
        }
    }
    const cudaError_t destroy_result = destroy(stream);
    if (destroy_result != cudaSuccess || state.quota_mode != QuotaMode::kEnabled) {
        return destroy_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    if (async_stream.has_value() &&
        !state.allocations.detach_async_releases_for_stream(*async_stream)) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

CUresult intercept_mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                   CUstream stream, bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream
                   ? state.driver.mem_alloc_async_ptsz(device_pointer, memory_bytes, stream)
                   : state.driver.mem_alloc_async(device_pointer, memory_bytes, stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_mem_alloc_async_ptsz()
                                   : !state.driver.has_mem_alloc_async())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return intercept_async_tracked_allocation(
        device_pointer, memory_bytes, stream, per_thread_default_stream,
        [&state, device_pointer, memory_bytes, stream, per_thread_default_stream] {
            return per_thread_default_stream
                       ? state.driver.mem_alloc_async_ptsz(device_pointer, memory_bytes, stream)
                       : state.driver.mem_alloc_async(device_pointer, memory_bytes, stream);
        });
}

CUresult intercept_mem_alloc_from_pool_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                             CUmemoryPool pool, CUstream stream,
                                             bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.mem_alloc_from_pool_async_ptsz(
                                               device_pointer, memory_bytes, pool, stream)
                                         : state.driver.mem_alloc_from_pool_async(
                                               device_pointer, memory_bytes, pool, stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_mem_alloc_from_pool_async_ptsz()
                                   : !state.driver.has_mem_alloc_from_pool_async())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return intercept_async_tracked_allocation(
        device_pointer, memory_bytes, stream, per_thread_default_stream,
        [&state, device_pointer, memory_bytes, pool, stream, per_thread_default_stream] {
            return per_thread_default_stream ? state.driver.mem_alloc_from_pool_async_ptsz(
                                                   device_pointer, memory_bytes, pool, stream)
                                             : state.driver.mem_alloc_from_pool_async(
                                                   device_pointer, memory_bytes, pool, stream);
        });
}

CUresult intercept_mem_create(CUmemGenericAllocationHandle* handle, std::size_t memory_bytes,
                              const CUmemAllocationProp* prop, unsigned long long flags) {
    InterceptorState& state = get_state();
    if (handle == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_create(handle, memory_bytes, prop, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    // VMM can also create host allocations. Only device-resident physical
    // allocations consume the CUDA device quota; address reservation and
    // non-device VMM paths remain delegated to the real driver.
    if (prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
        return state.driver.mem_create(handle, memory_bytes, prop, flags);
    }
    if (prop->location.id < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return state.driver.mem_create(handle, memory_bytes, prop, flags);
    }
    if (!state.driver.has_mem_release()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration ||
        state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return state.quota_mode == QuotaMode::kInvalidConfiguration ? CUDA_ERROR_INVALID_VALUE
                                                                    : CUDA_ERROR_UNKNOWN;
    }

    const DeviceId device = static_cast<DeviceId>(prop->location.id);
    auto reservation = state.quota->try_reserve(device, memory_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    }

    const CUresult create_result = state.driver.mem_create(handle, memory_bytes, prop, flags);
    if (create_result != CUDA_SUCCESS) {
        return create_result;
    }
    if (*handle == 0) {
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result = state.driver.mem_release(*handle);
        if (cleanup_result != CUDA_SUCCESS) {
            reservation->abandon();
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    const VmmAllocationIdentity identity{.handle = *handle, .device = prop->location.id};
    if (state.allocations.record_vmm(identity, memory_bytes)) {
        return CUDA_SUCCESS;
    }

    state.allocations.mark_accounting_degraded();
    const CUresult cleanup_result = state.driver.mem_release(*handle);
    if (cleanup_result != CUDA_SUCCESS) {
        return cleanup_result;
    }
    if (!state.quota->release(device, memory_bytes)) {
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}

CUresult intercept_mem_release(CUmemGenericAllocationHandle handle) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_release(handle);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_release()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return state.driver.mem_release(handle);
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }

    // Handles created outside this interceptor (for example imported handles)
    // are delegated because their owning allocation is not locally trackable.
    const auto release_state = state.allocations.begin_vmm_release_by_handle(handle);
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        return state.driver.mem_release(handle);
    }
    if (release_state.first == AllocationRegistry::ReleaseStatus::kInProgress ||
        !release_state.second.has_value()) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const AllocationRegistry::VmmReleaseTicket& ticket = *release_state.second;
    const CUresult release_result = state.driver.mem_release(handle);
    if (release_result != CUDA_SUCCESS) {
        state.allocations.cancel_vmm_release(ticket);
        return release_result;
    }

    const bool is_released =
        !ticket.is_last_reference ||
        state.quota->release(static_cast<DeviceId>(ticket.identity.device), ticket.memory_bytes);
    const bool is_completed = state.allocations.complete_vmm_release(ticket);
    if (!is_released || !is_completed) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_mem_address_reserve(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                       std::size_t alignment, CUdeviceptr requested_address,
                                       unsigned long long flags) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_address_reserve(device_pointer, memory_bytes, alignment,
                                                requested_address, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_address_reserve()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_address_reserve(device_pointer, memory_bytes, alignment,
                                            requested_address, flags);
}

CUresult intercept_mem_address_free(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_address_free(device_pointer, memory_bytes);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_address_free()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_address_free(device_pointer, memory_bytes);
}

CUresult intercept_mem_map(CUdeviceptr device_pointer, std::size_t memory_bytes, std::size_t offset,
                           CUmemGenericAllocationHandle handle, unsigned long long flags) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_map(device_pointer, memory_bytes, offset, handle, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_map()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_map(device_pointer, memory_bytes, offset, handle, flags);
}

CUresult intercept_mem_unmap(CUdeviceptr device_pointer, std::size_t memory_bytes) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_unmap(device_pointer, memory_bytes);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_unmap()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_unmap(device_pointer, memory_bytes);
}

CUresult intercept_mem_set_access(CUdeviceptr device_pointer, std::size_t memory_bytes,
                                  const CUmemAccessDesc* access_descriptors,
                                  std::size_t descriptor_count) {
    InterceptorState& state = get_state();
    if (descriptor_count != 0 && access_descriptors == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_set_access(device_pointer, memory_bytes, access_descriptors,
                                           descriptor_count);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_set_access()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.driver.mem_set_access(device_pointer, memory_bytes, access_descriptors,
                                       descriptor_count);
}

CUresult intercept_mem_get_address_range(CUdeviceptr* base_pointer, std::size_t* memory_bytes,
                                         CUdeviceptr device_pointer) {
    if (base_pointer == nullptr || memory_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_address_range(); },
        [&state, base_pointer, memory_bytes, device_pointer] {
            return state.driver.mem_get_address_range(base_pointer, memory_bytes, device_pointer);
        });
}

CUresult intercept_mem_get_access(unsigned long long* flags, const CUmemLocation* location,
                                  CUdeviceptr device_pointer) {
    if (flags == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_access(); },
        [&state, flags, location, device_pointer] {
            return state.driver.mem_get_access(flags, location, device_pointer);
        });
}

CUresult intercept_mem_export_to_shareable_handle(void* shareable_handle,
                                                  CUmemGenericAllocationHandle handle,
                                                  CUmemAllocationHandleType handle_type,
                                                  unsigned long long flags) {
    if (shareable_handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_export_to_shareable_handle(); },
        [&state, shareable_handle, handle, handle_type, flags] {
            return state.driver.mem_export_to_shareable_handle(shareable_handle, handle,
                                                               handle_type, flags);
        });
}

CUresult intercept_mem_import_from_shareable_handle(CUmemGenericAllocationHandle* handle,
                                                    void* os_handle,
                                                    CUmemAllocationHandleType handle_type) {
    if (handle == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_import_from_shareable_handle(handle, os_handle, handle_type);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_import_from_shareable_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_import_from_shareable_handle(handle, os_handle, handle_type);
}

CUresult intercept_mem_get_allocation_granularity(std::size_t* granularity,
                                                  const CUmemAllocationProp* prop,
                                                  CUmemAllocationGranularity_flags option) {
    if (granularity == nullptr || prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_allocation_granularity(); },
        [&state, granularity, prop, option] {
            return state.driver.mem_get_allocation_granularity(granularity, prop, option);
        });
}

CUresult intercept_mem_get_allocation_properties(CUmemAllocationProp* prop,
                                                 CUmemGenericAllocationHandle handle) {
    if (prop == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_allocation_properties(); },
        [&state, prop, handle] {
            return state.driver.mem_get_allocation_properties(prop, handle);
        });
}

CUresult intercept_mem_retain_allocation_handle(CUmemGenericAllocationHandle* handle,
                                                void* device_pointer) {
    if (handle == nullptr || device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_retain_allocation_handle(handle, device_pointer);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_retain_allocation_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return state.driver.mem_retain_allocation_handle(handle, device_pointer);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration ||
        state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return state.quota_mode == QuotaMode::kInvalidConfiguration ? CUDA_ERROR_INVALID_VALUE
                                                                    : CUDA_ERROR_UNKNOWN;
    }

    const CUresult retain_result =
        state.driver.mem_retain_allocation_handle(handle, device_pointer);
    if (retain_result != CUDA_SUCCESS) {
        return retain_result;
    }
    if (*handle == 0) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    if (state.allocations.retain_vmm_handle(*handle)) {
        return CUDA_SUCCESS;
    }

    const CUresult cleanup_result = state.driver.mem_release(*handle);
    if (cleanup_result != CUDA_SUCCESS) {
        state.allocations.mark_accounting_degraded();
        return cleanup_result;
    }
    state.allocations.mark_accounting_degraded();
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult intercept_mem_pool_trim_to(CUmemoryPool pool, std::size_t min_bytes_to_keep) {
    if (pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_trim_to(); },
        [&state, pool, min_bytes_to_keep] {
            return state.driver.mem_pool_trim_to(pool, min_bytes_to_keep);
        });
}

CUresult intercept_mem_pool_set_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                          void* value) {
    if (pool == nullptr || value == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_set_attribute(); },
        [&state, pool, attribute, value] {
            return state.driver.mem_pool_set_attribute(pool, attribute, value);
        });
}

CUresult intercept_mem_pool_get_attribute(CUmemoryPool pool, CUmemPool_attribute attribute,
                                          void* value) {
    if (pool == nullptr || value == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_get_attribute(); },
        [&state, pool, attribute, value] {
            return state.driver.mem_pool_get_attribute(pool, attribute, value);
        });
}

CUresult intercept_mem_pool_set_access(CUmemoryPool pool, const CUmemAccessDesc* access_descriptors,
                                       std::size_t descriptor_count) {
    if (pool == nullptr || (descriptor_count != 0 && access_descriptors == nullptr)) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_set_access(); },
        [&state, pool, access_descriptors, descriptor_count] {
            return state.driver.mem_pool_set_access(pool, access_descriptors, descriptor_count);
        });
}

CUresult intercept_mem_pool_get_access(CUmemAccess_flags* flags, CUmemoryPool pool,
                                       CUmemLocation* location) {
    if (flags == nullptr || pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_get_access(); },
        [&state, flags, pool, location] {
            return state.driver.mem_pool_get_access(flags, pool, location);
        });
}

CUresult intercept_mem_pool_create(CUmemoryPool* pool, const CUmemPoolProps* properties) {
    if (pool == nullptr || properties == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_create(); },
        [&state, pool, properties] { return state.driver.mem_pool_create(pool, properties); });
}

CUresult intercept_mem_pool_destroy(CUmemoryPool pool) {
    if (pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_destroy(); },
        [&state, pool] { return state.driver.mem_pool_destroy(pool); });
}

CUresult intercept_device_get_mem_pool(CUmemoryPool* pool, CUdevice device) {
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_device_get_mem_pool(); },
        [&state, pool, device] { return state.driver.device_get_mem_pool(pool, device); });
}

CUresult intercept_device_set_mem_pool(CUdevice device, CUmemoryPool pool) {
    if (device < 0 || pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_device_set_mem_pool(); },
        [&state, device, pool] { return state.driver.device_set_mem_pool(device, pool); });
}

CUresult intercept_device_get_default_mem_pool(CUmemoryPool* pool, CUdevice device) {
    if (pool == nullptr || device < 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_device_get_default_mem_pool(); },
        [&state, pool, device] { return state.driver.device_get_default_mem_pool(pool, device); });
}

CUresult intercept_mem_get_default_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                            CUmemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_default_mem_pool(); },
        [&state, pool, location, allocation_type] {
            return state.driver.mem_get_default_mem_pool(pool, location, allocation_type);
        });
}

CUresult intercept_mem_get_mem_pool(CUmemoryPool* pool, CUmemLocation* location,
                                    CUmemAllocationType allocation_type) {
    if (pool == nullptr || location == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_get_mem_pool(); },
        [&state, pool, location, allocation_type] {
            return state.driver.mem_get_mem_pool(pool, location, allocation_type);
        });
}

CUresult intercept_mem_set_mem_pool(CUmemLocation* location, CUmemAllocationType allocation_type,
                                    CUmemoryPool pool) {
    if (location == nullptr || pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_set_mem_pool(); },
        [&state, location, allocation_type, pool] {
            return state.driver.mem_set_mem_pool(location, allocation_type, pool);
        });
}

CUresult intercept_mem_pool_export_to_shareable_handle(void* handle_out, CUmemoryPool pool,
                                                       CUmemAllocationHandleType handle_type,
                                                       unsigned long long flags) {
    if (handle_out == nullptr || pool == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_export_to_shareable_handle(); },
        [&state, handle_out, pool, handle_type, flags] {
            return state.driver.mem_pool_export_to_shareable_handle(handle_out, pool, handle_type,
                                                                    flags);
        });
}

CUresult intercept_mem_pool_import_from_shareable_handle(CUmemoryPool* pool_out, void* handle,
                                                         CUmemAllocationHandleType handle_type,
                                                         unsigned long long flags) {
    if (pool_out == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_pool_import_from_shareable_handle(pool_out, handle, handle_type,
                                                                  flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_pool_import_from_shareable_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_pool_import_from_shareable_handle(pool_out, handle, handle_type, flags);
}

CUresult intercept_mem_pool_export_pointer(CUmemPoolPtrExportData* share_data_out,
                                           CUdeviceptr device_pointer) {
    if (share_data_out == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mem_pool_export_pointer(); },
        [&state, share_data_out, device_pointer] {
            return state.driver.mem_pool_export_pointer(share_data_out, device_pointer);
        });
}

CUresult intercept_mem_pool_import_pointer(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                           CUmemPoolPtrExportData* share_data) {
    if (pointer_out == nullptr || pool == nullptr || share_data == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_pool_import_pointer(pointer_out, pool, share_data);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_pool_import_pointer()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_pool_import_pointer(pointer_out, pool, share_data);
}

namespace {

[[nodiscard]] nvmlReturn_t initialize_nvml(InterceptorState& state) {
    return state.nvml.initialize() ? NVML_SUCCESS : NVML_ERROR_LIBRARY_NOT_FOUND;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): output pointers are paired by name.
[[nodiscard]] nvmlReturn_t virtualize_nvml_memory(InterceptorState& state, nvmlDevice_t device,
                                                  unsigned long long physical_total,
                                                  unsigned long long physical_free,
                                                  unsigned long long* visible_total,
                                                  unsigned long long* visible_free) {
    ensure_state_initialized(state);
    if (state.quota_mode == QuotaMode::kDisabled) {
        *visible_total = physical_total;
        *visible_free = physical_free;
        return NVML_SUCCESS;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration ||
        state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return state.quota_mode == QuotaMode::kInvalidConfiguration ? NVML_ERROR_INVALID_ARGUMENT
                                                                    : NVML_ERROR_UNKNOWN;
    }

    unsigned int device_index = 0;
    if (!state.nvml.has_device_get_index() ||
        state.nvml.device_get_index(device, &device_index) != NVML_SUCCESS) {
        return NVML_ERROR_UNKNOWN;
    }
    const glimmer::control::MemoryInfo memory_info = state.quota->get_memory_info(
        static_cast<DeviceId>(device_index), static_cast<MemoryBytes>(physical_total),
        static_cast<MemoryBytes>(physical_free));
    if (!state.quota->is_healthy()) {
        return NVML_ERROR_UNKNOWN;
    }
    *visible_total = static_cast<unsigned long long>(memory_info.total_bytes);
    *visible_free = static_cast<unsigned long long>(memory_info.free_bytes);
    return NVML_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace

nvmlReturn_t intercept_nvml_init() {
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.init() : initialize_result;
}

nvmlReturn_t intercept_nvml_init_with_flags(unsigned int flags) {
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    if (initialize_result != NVML_SUCCESS) {
        return initialize_result;
    }
    return state.nvml.init_with_flags(flags);
}

nvmlReturn_t intercept_nvml_shutdown() {
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.shutdown() : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_count(unsigned int* device_count) {
    if (device_count == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.device_get_count(device_count)
                                             : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_handle_by_index(unsigned int index, nvmlDevice_t* device) {
    if (device == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.device_get_handle_by_index(index, device)
                                             : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_index(nvmlDevice_t device, unsigned int* index) {
    if (device == nullptr || index == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    return initialize_result == NVML_SUCCESS ? state.nvml.device_get_index(device, index)
                                             : initialize_result;
}

nvmlReturn_t intercept_nvml_device_get_memory_info(nvmlDevice_t device, nvmlMemory_t* memory) {
    if (device == nullptr || memory == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    if (initialize_result != NVML_SUCCESS) {
        return initialize_result;
    }
    const nvmlReturn_t query_result = state.nvml.device_get_memory_info(device, memory);
    if (query_result != NVML_SUCCESS) {
        return query_result;
    }
    unsigned long long visible_total = memory->total;
    unsigned long long visible_free = memory->free;
    const nvmlReturn_t virtualize_result = virtualize_nvml_memory(
        state, device, memory->total, memory->free, &visible_total, &visible_free);
    if (virtualize_result != NVML_SUCCESS) {
        return virtualize_result;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return NVML_SUCCESS;
    }
    memory->total = visible_total;
    memory->free = visible_free;
    memory->used = visible_total >= visible_free ? visible_total - visible_free : 0;
    return NVML_SUCCESS;
}

nvmlReturn_t intercept_nvml_device_get_memory_info_v2(nvmlDevice_t device,
                                                      nvmlMemory_v2_t* memory) {
    if (device == nullptr || memory == nullptr) {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    InterceptorState& state = get_state();
    const nvmlReturn_t initialize_result = initialize_nvml(state);
    if (initialize_result != NVML_SUCCESS) {
        return initialize_result;
    }
    if (!state.nvml.has_device_get_memory_info_v2()) {
        return NVML_ERROR_FUNCTION_NOT_FOUND;
    }
    const nvmlReturn_t query_result = state.nvml.device_get_memory_info_v2(device, memory);
    if (query_result != NVML_SUCCESS) {
        return query_result;
    }
    unsigned long long visible_total = memory->total;
    unsigned long long visible_free = memory->free;
    const nvmlReturn_t virtualize_result = virtualize_nvml_memory(
        state, device, memory->total, memory->free, &visible_total, &visible_free);
    if (virtualize_result != NVML_SUCCESS) {
        return virtualize_result;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return NVML_SUCCESS;
    }
    memory->total = visible_total;
    memory->free = visible_free;
    memory->reserved = 0;
    memory->used = visible_total >= visible_free ? visible_total - visible_free : 0;
    return NVML_SUCCESS;
}

CUresult intercept_mem_free_async(CUdeviceptr device_pointer, CUstream stream,
                                  bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                         : state.driver.mem_free_async(device_pointer, stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_mem_free_async_ptsz()
                                   : !state.driver.has_mem_free_async())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return per_thread_default_stream ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                         : state.driver.mem_free_async(device_pointer, stream);
    }
    if (state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }

    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, stream, per_thread_default_stream);
    if (!stream_identity.has_value()) {
        const CUresult free_result = per_thread_default_stream
                                         ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                         : state.driver.mem_free_async(device_pointer, stream);
        if (free_result == CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        return free_result;
    }

    const AsyncStreamIdentity async_stream =
        make_async_stream_identity(stream, *stream_identity, per_thread_default_stream);
    auto release_state = state.allocations.begin_async_release(
        make_allocation_identity(device_pointer, *stream_identity), async_stream);
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        release_state = state.allocations.begin_async_release_by_pointer(
            device_pointer, stream_identity->device, async_stream);
    }
    const auto [release_status, release_ticket] = release_state;
    if (release_status == AllocationRegistry::ReleaseStatus::kInProgress) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (release_status == AllocationRegistry::ReleaseStatus::kUnknown ||
        !release_ticket.has_value()) {
        return per_thread_default_stream ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                         : state.driver.mem_free_async(device_pointer, stream);
    }

    const CUresult free_result = per_thread_default_stream
                                     ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                     : state.driver.mem_free_async(device_pointer, stream);
    if (free_result != CUDA_SUCCESS) {
        state.allocations.cancel_release(*release_ticket);
        return free_result;
    }
    if (!state.allocations.commit_async_release(*release_ticket)) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_stream_query(CUstream stream, bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.stream_query_ptsz(stream)
                                         : state.driver.stream_query(stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_stream_query_ptsz()
                                   : !state.driver.has_stream_query())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const CUresult query_result = per_thread_default_stream ? state.driver.stream_query_ptsz(stream)
                                                            : state.driver.stream_query(stream);
    if (query_result == CUDA_SUCCESS && state.quota_mode == QuotaMode::kEnabled) {
        const std::optional<ContextIdentity> context =
            capture_stream_identity(state, stream, per_thread_default_stream);
        if (!context.has_value()) {
            state.allocations.mark_accounting_degraded();
        } else {
            const AsyncStreamIdentity async_stream =
                make_async_stream_identity(stream, *context, per_thread_default_stream);
            const auto summaries =
                state.allocations.complete_async_releases_for_stream_by_device(async_stream);
            if (!summaries.has_value()) {
                state.allocations.mark_accounting_degraded();
            } else {
                finalize_async_releases(state, *summaries);
            }
        }
    }
    return state.allocations.is_accounting_degraded() ? CUDA_ERROR_UNKNOWN : query_result;
}

CUresult intercept_stream_synchronize(CUstream stream, bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.stream_synchronize_ptsz(stream)
                                         : state.driver.stream_synchronize(stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_stream_synchronize_ptsz()
                                   : !state.driver.has_stream_synchronize())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const CUresult sync_result = per_thread_default_stream
                                     ? state.driver.stream_synchronize_ptsz(stream)
                                     : state.driver.stream_synchronize(stream);
    if (sync_result == CUDA_SUCCESS && state.quota_mode == QuotaMode::kEnabled) {
        const std::optional<ContextIdentity> context =
            capture_stream_identity(state, stream, per_thread_default_stream);
        if (!context.has_value()) {
            state.allocations.mark_accounting_degraded();
        } else {
            const AsyncStreamIdentity async_stream =
                make_async_stream_identity(stream, *context, per_thread_default_stream);
            const auto summaries =
                state.allocations.complete_async_releases_for_stream_by_device(async_stream);
            if (!summaries.has_value()) {
                state.allocations.mark_accounting_degraded();
            } else {
                finalize_async_releases(state, *summaries);
            }
        }
    }
    return state.allocations.is_accounting_degraded() ? CUDA_ERROR_UNKNOWN : sync_result;
}

CUresult intercept_stream_destroy(CUstream stream) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.stream_destroy(stream);
    }
    if (!ensure_initialized(state) || !state.driver.has_stream_destroy()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    std::optional<AsyncStreamIdentity> async_stream;
    if (state.quota_mode == QuotaMode::kEnabled && !state.allocations.is_accounting_degraded()) {
        const std::optional<ContextIdentity> context =
            capture_stream_identity(state, stream, false);
        if (!context.has_value()) {
            state.allocations.mark_accounting_degraded();
        } else {
            async_stream = make_async_stream_identity(stream, *context, false);
        }
    }

    const CUresult destroy_result = state.driver.stream_destroy(stream);
    if (destroy_result != CUDA_SUCCESS || state.quota_mode != QuotaMode::kEnabled) {
        return destroy_result;
    }
    if (state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }

    // Stream destruction does not guarantee that queued work has completed. Detach
    // the handle so a recycled stream cannot complete the old record; a later
    // context synchronization can still finalize it using its saved context.
    if (async_stream.has_value() &&
        !state.allocations.detach_async_releases_for_stream(*async_stream)) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_stream_get_device(CUstream stream, CUdevice* device,
                                     bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.stream_get_device_ptsz(stream, device)
                                         : state.driver.stream_get_device(stream, device);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_stream_identity_ptsz()
                                   : !state.driver.has_stream_identity())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return per_thread_default_stream ? state.driver.stream_get_device_ptsz(stream, device)
                                     : state.driver.stream_get_device(stream, device);
}

CUresult intercept_stream_get_context(CUstream stream, CUcontext* context,
                                      bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.stream_get_context_ptsz(stream, context)
                                         : state.driver.stream_get_context(stream, context);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_stream_identity_ptsz()
                                   : !state.driver.has_stream_identity())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return per_thread_default_stream ? state.driver.stream_get_context_ptsz(stream, context)
                                     : state.driver.stream_get_context(stream, context);
}

CUresult intercept_context_synchronize() {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.context_synchronize();
    }
    if (!ensure_initialized(state) || !state.driver.has_context_synchronize()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const CUresult sync_result = state.driver.context_synchronize();
    if (sync_result != CUDA_SUCCESS || state.quota_mode != QuotaMode::kEnabled) {
        return sync_result;
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    const auto summaries =
        state.allocations.complete_async_releases_for_context_by_device(context->context);
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
    } else {
        finalize_async_releases(state, *summaries);
    }
    return state.allocations.is_accounting_degraded() ? CUDA_ERROR_UNKNOWN : sync_result;
}

CUresult intercept_init(unsigned int flags) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.init(flags);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }
    return state.driver.init(flags);
}

CUresult intercept_mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    InterceptorState& state = get_state();
    return intercept_tracked_allocation(
        device_pointer, memory_bytes, [&state, device_pointer, memory_bytes] {
            return state.driver.mem_alloc(device_pointer, memory_bytes);
        });
}

CUresult intercept_mem_alloc_managed(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                     unsigned int flags) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        if (!state.driver.has_mem_alloc_managed()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return state.driver.mem_alloc_managed(device_pointer, memory_bytes, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_alloc_managed()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    return intercept_tracked_allocation(
        device_pointer, memory_bytes, [&state, device_pointer, memory_bytes, flags] {
            return state.driver.mem_alloc_managed(device_pointer, memory_bytes, flags);
        });
}

CUresult intercept_mem_alloc_pitch(CUdeviceptr* device_pointer, std::size_t* pitch,
                                   std::size_t width_bytes, std::size_t height,
                                   unsigned int element_size_bytes) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr || pitch == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        if (!state.driver.has_mem_alloc_pitch()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return state.driver.mem_alloc_pitch(device_pointer, pitch, width_bytes, height,
                                            element_size_bytes);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_alloc_pitch()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (state.quota_mode == QuotaMode::kDisabled) {
        return state.driver.mem_alloc_pitch(device_pointer, pitch, width_bytes, height,
                                            element_size_bytes);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (state.quota == nullptr) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    const std::optional<MemoryBytes> requested_bytes =
        checked_multiply(static_cast<MemoryBytes>(width_bytes), static_cast<MemoryBytes>(height));
    if (!requested_bytes.has_value()) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    auto reservation =
        state.quota->try_reserve(static_cast<DeviceId>(context->device), *requested_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
    }

    const CUresult allocation_result = state.driver.mem_alloc_pitch(
        device_pointer, pitch, width_bytes, height, element_size_bytes);
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }

    if (*device_pointer == 0) {
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    const std::optional<MemoryBytes> actual_bytes =
        checked_multiply(static_cast<MemoryBytes>(*pitch), static_cast<MemoryBytes>(height));
    if (!actual_bytes.has_value()) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::optional<glimmer::control::MemoryReservation> adjustment;
    if (*actual_bytes > *requested_bytes) {
        adjustment = state.quota->try_reserve(static_cast<DeviceId>(context->device),
                                              *actual_bytes - *requested_bytes);
        if (!adjustment.has_value()) {
            const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
            if (cleanup_result != CUDA_SUCCESS) {
                reservation->abandon();
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            return state.quota->is_healthy() ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_UNKNOWN;
        }
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            reservation->abandon();
            if (adjustment.has_value()) {
                adjustment->abandon();
            }
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    MemoryBytes committed_bytes = *requested_bytes;
    if (adjustment.has_value()) {
        if (!adjustment->commit()) {
            const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
            if (cleanup_result != CUDA_SUCCESS) {
                adjustment->abandon();
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            if (state.quota == nullptr ||
                !state.quota->release(static_cast<DeviceId>(context->device), committed_bytes)) {
                state.allocations.mark_accounting_degraded();
            }
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        committed_bytes = *actual_bytes;
    } else if (*actual_bytes < *requested_bytes) {
        if (state.quota == nullptr || !state.quota->release(static_cast<DeviceId>(context->device),
                                                            *requested_bytes - *actual_bytes)) {
            const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
            state.allocations.mark_accounting_degraded();
            return cleanup_result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : cleanup_result;
        }
        committed_bytes = *actual_bytes;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  *actual_bytes)) {
        state.allocations.mark_accounting_degraded();
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (state.quota == nullptr ||
            !state.quota->release(static_cast<DeviceId>(context->device), committed_bytes)) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_mem_free(CUdeviceptr device_pointer) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_free(device_pointer);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (state.quota_mode != QuotaMode::kEnabled) {
        return state.driver.mem_free(device_pointer);
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        const CUresult free_result = state.driver.mem_free(device_pointer);
        if (free_result == CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        return free_result;
    }

    auto release_state =
        state.allocations.begin_release(make_allocation_identity(device_pointer, *context));
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        release_state = state.allocations.begin_release_by_pointer(device_pointer, context->device);
    }
    const auto [release_status, release_ticket] = release_state;
    if (release_status == AllocationRegistry::ReleaseStatus::kInProgress) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (release_status == AllocationRegistry::ReleaseStatus::kUnknown ||
        !release_ticket.has_value()) {
        return state.driver.mem_free(device_pointer);
    }

    const CUresult free_result = state.driver.mem_free(device_pointer);
    if (free_result != CUDA_SUCCESS) {
        state.allocations.cancel_release(*release_ticket);
        return free_result;
    }

    const bool is_released =
        state.quota != nullptr &&
        state.quota->release(static_cast<DeviceId>(release_ticket->identity.device),
                             release_ticket->memory_bytes);
    const bool is_completed = state.allocations.complete_release(*release_ticket);
    if (!is_released || !is_completed) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_context_get_current(CUcontext* context) {
    InterceptorState& state = get_state();
    if (context == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.context_get_current(context);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }
    return state.driver.context_get_current(context);
}

CUresult intercept_context_get_device(CUdevice* device) {
    InterceptorState& state = get_state();
    if (device == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.context_get_device(device);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }
    return state.driver.context_get_device(device);
}

CUresult intercept_context_destroy(CUcontext context) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.context_destroy(context);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!state.driver.has_context_destroy()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return state.driver.context_destroy(context);
    }

    const CUresult destroy_result = state.driver.context_destroy(context);
    if (destroy_result != CUDA_SUCCESS) {
        return destroy_result;
    }

    const auto summaries = state.allocations.erase_context_by_device(context);
    if (!summaries.has_value()) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    finalize_async_releases(state, *summaries);
    if (state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) {
    InterceptorState& state = get_state();
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_get_info(free_bytes, total_bytes);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }

    const CUresult info_result = state.driver.mem_get_info(free_bytes, total_bytes);
    if (info_result != CUDA_SUCCESS || state.quota_mode != QuotaMode::kEnabled) {
        return info_result;
    }

    if (state.quota == nullptr) {
        return CUDA_ERROR_UNKNOWN;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value() && state.uses_shared_quota) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    const DeviceId device =
        context.has_value() ? static_cast<DeviceId>(context->device) : QuotaStore::default_device;
    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(device, *total_bytes, *free_bytes);
    if (!state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }
    *total_bytes = static_cast<std::size_t>(memory_info.total_bytes);
    *free_bytes = static_cast<std::size_t>(memory_info.free_bytes);
    return CUDA_SUCCESS;
}

CUresult intercept_device_total_mem(std::size_t* total_bytes, CUdevice device) {
    InterceptorState& state = get_state();
    if (total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        if (!state.driver.has_device_total_mem()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return state.driver.device_total_mem(total_bytes, device);
    }
    if (!ensure_initialized(state) || !state.driver.has_device_total_mem()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }

    const CUresult total_result = state.driver.device_total_mem(total_bytes, device);
    if (total_result != CUDA_SUCCESS || state.quota_mode != QuotaMode::kEnabled) {
        return total_result;
    }

    if (state.quota == nullptr) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }
    const glimmer::core::QuotaUsage usage = state.quota->usage(static_cast<DeviceId>(device));
    if (!state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }
    *total_bytes = static_cast<std::size_t>(std::min(usage.limit_bytes, *total_bytes));
    return CUDA_SUCCESS;
}

CUresult intercept_get_proc_address(const char* symbol, void** function_pointer, int cuda_version,
                                    cuuint64_t flags) {
    InterceptorState& state = get_state();
    if (symbol == nullptr || symbol[0] == '\0' || function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_get_proc_address()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (g_is_inside_proc_address_v2) {
        return state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
    }

    const CUresult result =
        state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
    if (result != CUDA_SUCCESS || function_pointer == nullptr || *function_pointer == nullptr) {
        return result;
    }

    if (void* intercepted_symbol = glimmer::interceptor::find_interceptor_symbol(symbol);
        intercepted_symbol != nullptr) {
        *function_pointer = intercepted_symbol;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_get_proc_address_v2(const char* symbol, void** function_pointer,
                                       int cuda_version, cuuint64_t flags,
                                       CUdriverProcAddressQueryResult* symbol_status) {
    InterceptorState& state = get_state();
    if (symbol == nullptr || symbol[0] == '\0' || function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        if (state.driver.has_get_proc_address_v2()) {
            return state.driver.get_proc_address_v2(symbol, function_pointer, cuda_version, flags,
                                                    symbol_status);
        }
        if (!state.driver.has_get_proc_address()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
    }
    if (!ensure_initialized(state) ||
        (!state.driver.has_get_proc_address() && !state.driver.has_get_proc_address_v2())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (g_is_inside_proc_address_v2) {
        if (state.driver.has_get_proc_address_v2()) {
            return state.driver.get_proc_address_v2(symbol, function_pointer, cuda_version, flags,
                                                    symbol_status);
        }
        if (!state.driver.has_get_proc_address()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        return state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
    }

    ProcAddressV2Scope scope;
    CUresult result = CUDA_ERROR_NOT_SUPPORTED;
    if (state.driver.has_get_proc_address_v2()) {
        *function_pointer = nullptr;
        result = state.driver.get_proc_address_v2(symbol, function_pointer, cuda_version, flags,
                                                  symbol_status);
        if (result == CUDA_ERROR_NOT_FOUND) {
            result = CUDA_SUCCESS;
            if (symbol_status != nullptr) {
                *symbol_status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
            }
        }
    } else if (state.driver.has_get_proc_address()) {
        *function_pointer = nullptr;
        result = state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
        const bool symbol_not_found = result == CUDA_ERROR_NOT_FOUND;
        if (symbol_not_found) {
            result = CUDA_SUCCESS;
        }
        if (result == CUDA_SUCCESS && symbol_status != nullptr) {
            *symbol_status = symbol_not_found || *function_pointer == nullptr
                                 ? CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
                                 : CU_GET_PROC_ADDRESS_SUCCESS;
        }
    } else {
        result = state.driver.get_proc_address_v2(symbol, function_pointer, cuda_version, flags,
                                                  symbol_status);
    }
    if (result != CUDA_SUCCESS || function_pointer == nullptr || *function_pointer == nullptr) {
        return result;
    }

    if (void* intercepted_symbol = glimmer::interceptor::find_interceptor_symbol(symbol);
        intercepted_symbol != nullptr) {
        *function_pointer = intercepted_symbol;
    }
    return CUDA_SUCCESS;
}

}  // namespace glimmer::interceptor
