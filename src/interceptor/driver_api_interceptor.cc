#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "glimmer/control/process_memory_quota.h"

#include "internal/allocation_registry.h"
#include "internal/diagnostics.h"
#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/runtime_api_bridge.h"
#include "internal/symbol_registry.h"

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
#include <utility>

namespace glimmer::interceptor {

namespace {

using glimmer::control::ProcessMemoryQuota;
using glimmer::core::MemoryBytes;
using glimmer::interceptor::AllocationRegistry;
using glimmer::interceptor::AllocationScope;
using glimmer::interceptor::DriverDispatch;

enum class QuotaMode : std::uint8_t {
    kDisabled,
    kEnabled,
    kInvalidConfiguration,
};

struct InterceptorState {
    std::once_flag initialization_once;
    DriverDispatch driver;
    QuotaMode quota_mode = QuotaMode::kDisabled;
    std::unique_ptr<ProcessMemoryQuota> quota;
    glimmer::interceptor::AllocationRegistry allocations;
    bool is_driver_ready = false;
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

    const char* configured_limit = std::getenv("GLIMMER_MEMORY_LIMIT_BYTES");
    if (configured_limit == nullptr) {
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
        state.quota = std::make_unique<ProcessMemoryQuota>(*limit_bytes);
        state.quota_mode = QuotaMode::kEnabled;
    } catch (...) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        glimmer::interceptor::report_diagnostic(
            "[glimmer] failed to initialize the memory quota\n");
    }
}

[[nodiscard]] bool ensure_initialized(InterceptorState& state) {
    std::call_once(state.initialization_once, [&state] { initialize_state(state); });
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
    if (state.driver.context_get_device(&device) != CUDA_SUCCESS) {
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
    if (device_result != CUDA_SUCCESS) {
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

}  // namespace

}  // namespace glimmer::interceptor

namespace glimmer::interceptor {

cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                     RuntimeMallocFunction allocate, RuntimeFreeFunction release) {
    InterceptorState& state = get_state();
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? allocate(device_pointer, memory_bytes)
                   : cudaErrorUnknown;
    }

    if (device_pointer == nullptr || state.quota_mode == QuotaMode::kDisabled) {
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

    auto reservation = state.quota->try_reserve(memory_bytes);
    if (!reservation.has_value()) {
        return cudaErrorMemoryAllocation;
    }

    const cudaError_t allocation_result = allocate(device_pointer, memory_bytes);
    if (allocation_result != cudaSuccess) {
        return allocation_result;
    }

    if (device_pointer == nullptr) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return cudaErrorUnknown;
    }

    if (!reservation->commit()) {
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  memory_bytes)) {
        const cudaError_t cleanup_result =
            release == nullptr ? cudaErrorUnknown : release(*device_pointer);
        if (cleanup_result != cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (state.quota == nullptr || !state.quota->release(memory_bytes)) {
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
        state.quota != nullptr && state.quota->release(release_ticket->memory_bytes);
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
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return cudaErrorInvalidValue;
    }

    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(*total_bytes, *free_bytes);
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
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (device_pointer == nullptr || state.quota_mode == QuotaMode::kDisabled) {
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

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    auto reservation = state.quota->try_reserve(memory_bytes);
    if (!reservation.has_value()) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    const CUresult allocation_result = std::forward<AllocateFunction>(allocate)();
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  memory_bytes)) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (state.quota == nullptr || !state.quota->release(memory_bytes)) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return CUDA_SUCCESS;
}

template <typename AllocateFunction>
CUresult intercept_async_tracked_allocation(CUdeviceptr* device_pointer, MemoryBytes memory_bytes,
                                            CUstream stream, bool per_thread_default_stream,
                                            AllocateFunction&& allocate) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }

    if (device_pointer == nullptr || state.quota_mode == QuotaMode::kDisabled) {
        return std::forward<AllocateFunction>(allocate)();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr) {
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

    auto reservation = state.quota->try_reserve(memory_bytes);
    if (!reservation.has_value()) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    const CUresult allocation_result = std::forward<AllocateFunction>(allocate)();
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }
    if (device_pointer == nullptr || *device_pointer == 0) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result =
            per_thread_default_stream ? state.driver.mem_free_async_ptsz(*device_pointer, stream)
                                      : state.driver.mem_free_async(*device_pointer, stream);
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    const auto identity = make_allocation_identity(*device_pointer, *stream_identity);
    if (!state.allocations.record(identity, memory_bytes, AllocationScope::kContextIndependent)) {
        const CUresult cleanup_result =
            per_thread_default_stream ? state.driver.mem_free_async_ptsz(*device_pointer, stream)
                                      : state.driver.mem_free_async(*device_pointer, stream);
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

void finalize_async_release(InterceptorState& state, MemoryBytes released_bytes) {
    if (released_bytes == 0 || state.allocations.is_accounting_degraded()) {
        return;
    }
    if (state.quota == nullptr || !state.quota->release(released_bytes)) {
        state.allocations.mark_accounting_degraded();
    }
}

CUresult intercept_mem_alloc_async(CUdeviceptr* device_pointer, std::size_t memory_bytes,
                                   CUstream stream, bool per_thread_default_stream) {
    InterceptorState& state = get_state();
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

    auto release_state = state.allocations.begin_async_release(
        make_allocation_identity(device_pointer, *stream_identity), stream);
    if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
        release_state = state.allocations.begin_async_release_by_pointer(
            device_pointer, stream_identity->device, stream);
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
        finalize_async_release(state, state.allocations.complete_async_releases_for_stream(stream));
    }
    return query_result;
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
        finalize_async_release(state, state.allocations.complete_async_releases_for_stream(stream));
    }
    return sync_result;
}

CUresult intercept_stream_get_device(CUstream stream, CUdevice* device,
                                     bool per_thread_default_stream) {
    InterceptorState& state = get_state();
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
    if (context.has_value()) {
        finalize_async_release(state, state.allocations.complete_async_releases_for_context(
                                          context->context, context->device));
    }
    return sync_result;
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

    if (device_pointer == nullptr || pitch == nullptr || state.quota_mode == QuotaMode::kDisabled) {
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

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        return CUDA_ERROR_INVALID_CONTEXT;
    }

    const std::optional<MemoryBytes> requested_bytes =
        checked_multiply(static_cast<MemoryBytes>(width_bytes), static_cast<MemoryBytes>(height));
    if (!requested_bytes.has_value()) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    auto reservation = state.quota->try_reserve(*requested_bytes);
    if (!reservation.has_value()) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    const CUresult allocation_result = state.driver.mem_alloc_pitch(
        device_pointer, pitch, width_bytes, height, element_size_bytes);
    if (allocation_result != CUDA_SUCCESS) {
        return allocation_result;
    }

    const std::optional<MemoryBytes> actual_bytes =
        checked_multiply(static_cast<MemoryBytes>(*pitch), static_cast<MemoryBytes>(height));
    if (!actual_bytes.has_value()) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::optional<glimmer::control::MemoryReservation> adjustment;
    if (*actual_bytes > *requested_bytes) {
        adjustment = state.quota->try_reserve(*actual_bytes - *requested_bytes);
        if (!adjustment.has_value()) {
            const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
            if (cleanup_result != CUDA_SUCCESS) {
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }

    if (!reservation->commit()) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    MemoryBytes committed_bytes = *requested_bytes;
    if (adjustment.has_value()) {
        if (!adjustment->commit()) {
            const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
            if (cleanup_result != CUDA_SUCCESS) {
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            if (state.quota == nullptr || !state.quota->release(committed_bytes)) {
                state.allocations.mark_accounting_degraded();
            }
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        committed_bytes = *actual_bytes;
    } else if (*actual_bytes < *requested_bytes) {
        if (state.quota == nullptr || !state.quota->release(*requested_bytes - *actual_bytes)) {
            const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
            state.allocations.mark_accounting_degraded();
            return cleanup_result == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : cleanup_result;
        }
        committed_bytes = *actual_bytes;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  *actual_bytes)) {
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (state.quota == nullptr || !state.quota->release(committed_bytes)) {
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
        state.quota != nullptr && state.quota->release(release_ticket->memory_bytes);
    const bool is_completed = state.allocations.complete_release(*release_ticket);
    if (!is_released || !is_completed) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_context_get_current(CUcontext* context) {
    InterceptorState& state = get_state();
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

    const MemoryBytes released_bytes = state.allocations.erase_context(context);
    if (released_bytes == 0 || state.allocations.is_accounting_degraded()) {
        return CUDA_SUCCESS;
    }
    if (state.quota == nullptr || !state.quota->release(released_bytes)) {
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult intercept_mem_get_info(std::size_t* free_bytes, std::size_t* total_bytes) {
    InterceptorState& state = get_state();
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
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(*total_bytes, *free_bytes);
    *total_bytes = static_cast<std::size_t>(memory_info.total_bytes);
    *free_bytes = static_cast<std::size_t>(memory_info.free_bytes);
    return CUDA_SUCCESS;
}

CUresult intercept_device_total_mem(std::size_t* total_bytes, CUdevice device) {
    InterceptorState& state = get_state();
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
    if (total_bytes == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    const glimmer::core::QuotaUsage usage = state.quota->usage();
    *total_bytes = static_cast<std::size_t>(std::min(usage.limit_bytes, *total_bytes));
    return CUDA_SUCCESS;
}

CUresult intercept_get_proc_address(const char* symbol, void** function_pointer, int cuda_version,
                                    cuuint64_t flags) {
    InterceptorState& state = get_state();
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

    if (symbol == nullptr || symbol[0] == '\0' || function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
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
