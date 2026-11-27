#pragma once

// Private state shared by the driver API interceptor translation units.
// Only driver_api_*.cc may include this header.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "glimmer/control/process_memory_quota.h"
#include "glimmer/control/composite_quota.h"
#include "glimmer/control/device_capacity_quota.h"
#include "glimmer/control/launch_gate.h"
#include "glimmer/control/shared_memory_quota.h"
#include "glimmer/core/scheduler_mode.h"

#include "internal/allocation_registry.h"
#include "internal/diagnostics.h"
#include "internal/driver_api_interceptor.h"
#include "internal/driver_dispatch.h"
#include "internal/launch_completion_tracker.h"
#include "internal/loaded_image_path.h"
#include "internal/persistent_kernel.h"
#include "internal/persistent_sgemm.h"
#include "internal/cuda_gemm_allocation.h"
#include "internal/cuda_cooperative_pool.h"
#include "internal/cuda_binary_slice.h"
#include "internal/qualified_cubin_source.h"
#include "internal/driver_cooperative_router.h"
#include "internal/runtime_api_bridge.h"
#include <filesystem>
#include "internal/symbol_registry.h"
#include "internal/nvml_dispatch.h"

#include <cuda.h>

#include <charconv>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <exception>
#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <system_error>
#include <type_traits>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include <utility>
#include <vector>

namespace glimmer::interceptor {

// Hidden like the former anonymous namespace, so the export table is unchanged.
#pragma GCC visibility push(hidden)
namespace interceptor_detail {

using glimmer::control::CompositeQuota;
using glimmer::control::DeviceCapacityQuota;
using glimmer::control::DeviceId;
using glimmer::control::DeviceMemoryCapacity;
using glimmer::control::DeviceMemoryCapacityResolver;
using glimmer::control::LaunchGate;
using glimmer::control::LaunchGateOptions;
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
    std::mutex construction_mutex;
    bool has_started_initialization{};
    std::optional<CudaNativeEpochConfiguration> native_epoch_configuration;
    bool should_owner_pump_control{};
    std::once_flag driver_initialization_once;
    DriverDispatch driver;
    NvmlDispatch nvml;
    std::unique_ptr<LaunchGate> launch_gate;
    std::unique_ptr<LaunchCompletionTracker> launch_tracker;
    std::unique_ptr<PersistentKernelRuntime> persistent_kernels;
    std::unique_ptr<DriverCooperativeRouter> cooperative_router;
    bool is_cooperative_requested{};
    bool is_runtime_persistent_enabled = false;
    std::uint32_t persistent_worker_blocks = 0;
    QuotaMode quota_mode = QuotaMode::kDisabled;
    glimmer::core::SchedulerMode scheduler_mode = glimmer::core::SchedulerMode::kOff;
    std::atomic<bool> launch_observation_reported = false;
    std::atomic<std::uint64_t> launch_observation_count = 0;
    bool trace_kernel_launches = false;
    bool persistent_audit{};
    bool trace_memory_info = false;
    bool trace_launch_timings = false;
    std::size_t launch_batch_size = 1;
    struct LaunchBatchState {
        std::size_t remaining_launches = 0;
        std::size_t active_calls = 0;
        bool close_requested = false;
    };
    std::mutex launch_batch_mutex;
    std::unordered_map<glimmer::core::TaskId, LaunchBatchState> launch_batches;
    std::unique_ptr<QuotaStore> quota;
    glimmer::interceptor::AllocationRegistry allocations;
    bool is_driver_ready = false;
    std::atomic<bool> is_native_dispatch_ready{false};
    bool launch_scheduler_ready = false;
    bool uses_shared_quota = false;
};

inline thread_local bool g_is_inside_proc_address_v2 = false;
inline thread_local bool g_is_inside_cooperative_boundary = false;

struct ThreadLaunchBatch final {
    glimmer::core::TaskId task_id = 0;
    bool final_launch = false;
};

inline thread_local ThreadLaunchBatch g_thread_launch_batch;

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

[[nodiscard]] inline InterceptorState& get_state() {
    // The state intentionally outlives normal C++ teardown. A forked child
    // cannot join threads inherited from the parent, so it receives a fresh
    // state on its first intercepted call instead of destroying the parent's
    // thread objects during child exit.
    static InterceptorState* state = new InterceptorState();
    static pid_t state_pid = ::getpid();
    const pid_t current_pid = ::getpid();
    if (state_pid != current_pid) {
        state = new InterceptorState();
        state_pid = current_pid;
        g_thread_launch_batch = {};
    }
    return *state;
}

[[nodiscard]] inline std::optional<MemoryBytes> read_memory_limit(const char* variable_name) {
    if (variable_name == nullptr) {
        return std::nullopt;
    }
    const char* value = std::getenv(variable_name);
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

[[nodiscard]] inline std::optional<MemoryBytes> read_quota_limit() {
    return read_memory_limit("GLIMMER_MEMORY_LIMIT_BYTES");
}

[[nodiscard]] inline std::optional<std::size_t> read_positive_size(const char* variable_name) {
    const std::optional<MemoryBytes> value = read_memory_limit(variable_name);
    if (!value.has_value() || value.value() == 0 ||
        value.value() > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(value.value());
}

[[nodiscard]] inline bool read_public_native_epoch_configuration(InterceptorState& state) {
    constexpr std::array names{
        "GLIMMER_NATIVE_EPOCH_MAX_CALLS", "GLIMMER_NATIVE_EPOCH_CALLS_PER_PREFIX",
        "GLIMMER_NATIVE_EPOCH_PENDING_PREFIXES", "GLIMMER_NATIVE_EPOCH_ACCEPTANCE_US"};
    if (std::ranges::none_of(names, [](const char* name) { return std::getenv(name) != nullptr; }))
        return true;
    if (state.native_epoch_configuration || !cooperative_runtime_preview_requested() ||
        state.scheduler_mode != glimmer::core::SchedulerMode::kEnforce)
        return false;
    const auto limit = read_quota_limit();
    const char* quota_mode = std::getenv("GLIMMER_QUOTA_MODE");
    if (!limit || *limit == 0 ||
        (quota_mode && std::strcmp(quota_mode, "process") != 0 &&
         std::strcmp(quota_mode, "shared") != 0))
        return false;
    const char* batch = std::getenv("GLIMMER_SCHEDULER_BATCH_SIZE");
    if (batch && read_positive_size("GLIMMER_SCHEDULER_BATCH_SIZE") != 1) return false;
    control::NativeEpochExtent extent;
    const std::array fields{&extent.max_calls, &extent.calls_per_prefix,
                            &extent.max_pending_prefixes, &extent.acceptance_microseconds};
    for (std::size_t index = 0; index < names.size(); ++index) {
        const auto value = read_memory_limit(names[index]);
        if (!value || *value > std::numeric_limits<std::uint32_t>::max()) return false;
        *fields[index] = static_cast<std::uint32_t>(*value);
    }
    if (!control::valid_native_epoch_extent(extent)) return false;
    state.native_epoch_configuration = CudaNativeEpochConfiguration{extent, {}};
    return true;
}

[[nodiscard]] inline bool read_experimental_control_executor(InterceptorState& state) {
    const char* selection = std::getenv("GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR");
    if (!selection || std::strcmp(selection, "thread") == 0) return true;
    const char* socket = std::getenv("GLIMMER_SCHEDULER_CONTROL_SOCKET");
    if (std::strcmp(selection, "owner") != 0 ||
        state.scheduler_mode != glimmer::core::SchedulerMode::kEnforce ||
        !cooperative_runtime_preview_requested() || !socket || *socket == '\0')
        return false;
    state.should_owner_pump_control = true;
    return true;
}

[[nodiscard]] inline std::optional<std::size_t> read_nonnegative_size(const char* variable_name) {
    const std::optional<MemoryBytes> value = read_memory_limit(variable_name);
    if (!value.has_value() || value.value() > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(value.value());
}

[[nodiscard]] inline std::optional<std::uint32_t> read_positive_weight() {
    const std::optional<MemoryBytes> value = read_memory_limit("GLIMMER_SCHEDULER_WEIGHT");
    if (!value.has_value() || value.value() == 0 ||
        value.value() > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value.value());
}

[[nodiscard]] inline std::optional<std::uint32_t> read_scheduler_priority() {
    const std::optional<MemoryBytes> value = read_memory_limit("GLIMMER_SCHEDULER_PRIORITY");
    if (!value.has_value() || value.value() > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value.value());
}

[[nodiscard]] inline std::optional<std::uint32_t> read_scheduler_priority_threshold() {
    const std::optional<MemoryBytes> value =
        read_memory_limit("GLIMMER_SCHEDULER_PRIORITY_THRESHOLD");
    if (!value.has_value() || value.value() > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value.value());
}

[[nodiscard]] inline std::optional<glimmer::core::SchedulingPolicy> read_scheduling_policy() {
    const char* value = std::getenv("GLIMMER_SCHEDULER_POLICY");
    if (value == nullptr) {
        return glimmer::core::SchedulingPolicy::kWeightedRoundRobin;
    }
    return glimmer::core::parse_scheduling_policy(value);
}

[[nodiscard]] inline std::optional<glimmer::control::DeviceId> read_quota_device() {
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

[[nodiscard]] inline std::optional<glimmer::core::SchedulerMode> read_scheduler_mode() {
    const char* value = std::getenv("GLIMMER_SCHEDULER_MODE");
    if (value == nullptr) {
        return glimmer::core::SchedulerMode::kOff;
    }
    return glimmer::core::parse_scheduler_mode(value);
}

[[nodiscard]] inline bool read_trace_setting(const char* variable_name) noexcept {
    const char* value = std::getenv(variable_name);
    return value != nullptr && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0);
}

[[nodiscard]] inline bool read_kernel_launch_trace_setting() noexcept {
    return read_trace_setting("GLIMMER_TRACE_KERNEL_LAUNCHES");
}

[[nodiscard]] inline bool read_memory_info_trace_setting() noexcept {
    return read_trace_setting("GLIMMER_TRACE_MEMORY_INFO");
}

[[nodiscard]] inline bool read_launch_timing_trace_setting() noexcept {
    return read_trace_setting("GLIMMER_TRACE_LAUNCH_TIMINGS");
}

inline bool resolve_device_capacity(void* context, DeviceId device,
                                    DeviceMemoryCapacity* capacity) noexcept {
    if (context == nullptr || capacity == nullptr || device < 0) {
        return false;
    }

    auto* driver = static_cast<DriverDispatch*>(context);
    CUcontext current_context = nullptr;
    CUdevice current_device = -1;
    const bool has_matching_context =
        driver->has_context_queries() &&
        driver->context_get_current(&current_context) == CUDA_SUCCESS &&
        current_context != nullptr && driver->context_get_device(&current_device) == CUDA_SUCCESS &&
        current_device == device;
    bool is_primary_active = false;
    const bool can_query_free_capacity =
        has_matching_context &&
        (!is_inside_runtime_call() ||
         (driver->primary_context_is_active(current_device, is_primary_active) == CUDA_SUCCESS &&
          is_primary_active));
    if (can_query_free_capacity) {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (driver->mem_get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS && total_bytes != 0) {
            *capacity = DeviceMemoryCapacity{.total_bytes = total_bytes, .free_bytes = free_bytes};
            return true;
        }
    }

    // A cold Runtime may report a current handle before its primary is active.
    // Do not force SDK initialization or infer that handle's primary identity.
    // Non-primary Runtime callers with an inactive primary conservatively use
    // this fallback too. Ready callers and direct Driver calls still query free
    // capacity above; the original allocator always checks physical availability.
    std::size_t total_bytes = 0;
    if (!driver->has_device_total_mem() ||
        driver->device_total_mem(&total_bytes, static_cast<CUdevice>(device)) != CUDA_SUCCESS ||
        total_bytes == 0) {
        return false;
    }
    // Device total is valid without Runtime context readiness. Retain the total
    // boundary and quota ledger; failed original allocation cancels reservation.
    *capacity = DeviceMemoryCapacity{.total_bytes = total_bytes, .free_bytes = total_bytes};
    return true;
}

inline std::uint64_t next_launch_observation_count(std::atomic<std::uint64_t>& count) noexcept {
    std::uint64_t current = count.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<std::uint64_t>::max()) {
        if (count.compare_exchange_weak(current, current + 1, std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
            return current + 1;
        }
    }
    return current;
}

[[nodiscard]] inline std::optional<MemoryBytes> checked_multiply(MemoryBytes left,
                                                                 MemoryBytes right) {
    if (right != 0 && left > std::numeric_limits<MemoryBytes>::max() / right) {
        return std::nullopt;
    }
    return left * right;
}

[[nodiscard]] inline std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point finished) noexcept {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started);
    return elapsed.count() < 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(elapsed.count());
}

// The setup owns allocator-backed scheduler state but converts every setup
// failure into a diagnostic and a disabled enforcement path.
// NOLINTNEXTLINE(bugprone-exception-escape)
inline void initialize_launch_scheduler(InterceptorState& state) noexcept {
    if (state.scheduler_mode != glimmer::core::SchedulerMode::kEnforce) {
        return;
    }

    const char* configured_max_concurrent = std::getenv("GLIMMER_MAX_CONCURRENT_KERNELS");
    const std::optional<std::size_t> max_concurrent =
        configured_max_concurrent == nullptr ? std::optional<std::size_t>{1}
                                             : read_positive_size("GLIMMER_MAX_CONCURRENT_KERNELS");
    const char* configured_batch_size = std::getenv("GLIMMER_SCHEDULER_BATCH_SIZE");
    const std::optional<std::size_t> batch_size =
        configured_batch_size == nullptr ? std::optional<std::size_t>{1}
                                         : read_positive_size("GLIMMER_SCHEDULER_BATCH_SIZE");
    const char* configured_reserved_slots =
        std::getenv("GLIMMER_SCHEDULER_PRIORITY_RESERVED_SLOTS");
    const std::optional<std::size_t> reserved_slots =
        configured_reserved_slots == nullptr
            ? std::optional<std::size_t>{0}
            : read_nonnegative_size("GLIMMER_SCHEDULER_PRIORITY_RESERVED_SLOTS");
    const char* configured_priority_threshold = std::getenv("GLIMMER_SCHEDULER_PRIORITY_THRESHOLD");
    const std::optional<std::uint32_t> priority_threshold =
        configured_priority_threshold == nullptr ? std::optional<std::uint32_t>{1}
                                                 : read_scheduler_priority_threshold();
    const std::optional<glimmer::core::SchedulingPolicy> policy = read_scheduling_policy();
    const std::optional<std::uint32_t> weight = read_positive_weight();
    const char* configured_priority = std::getenv("GLIMMER_SCHEDULER_PRIORITY");
    const std::optional<std::uint32_t> priority = configured_priority == nullptr
                                                      ? std::optional<std::uint32_t>{0}
                                                      : read_scheduler_priority();
    const char* configured_tenant = std::getenv("GLIMMER_SCHEDULER_TENANT_ID");
    const char* quota_tenant = std::getenv("GLIMMER_QUOTA_TENANT_ID");
    const char* tenant = configured_tenant != nullptr ? configured_tenant : quota_tenant;
    const char* configured_socket = std::getenv("GLIMMER_SCHEDULER_CONTROL_SOCKET");
    if (!max_concurrent.has_value() || !batch_size.has_value() || !reserved_slots.has_value() ||
        !priority_threshold.has_value() || !policy.has_value() || !priority.has_value() ||
        reserved_slots.value() > max_concurrent.value() ||
        (configured_tenant != nullptr && *configured_tenant == '\0') ||
        (configured_socket != nullptr && *configured_socket == '\0') ||
        (weight.has_value() == false && std::getenv("GLIMMER_SCHEDULER_WEIGHT") != nullptr)) {
        glimmer::interceptor::report_diagnostic(
            "[glimmer] launch scheduler configuration is invalid\n");
        return;
    }
    state.launch_batch_size = batch_size.value();

    try {
        LaunchGateOptions options{
            .max_concurrent_launches = max_concurrent.value(),
            .priority_reserved_slots = reserved_slots.value(),
            .priority_reservation_threshold = priority_threshold.value(),
            .scheduling_policy = policy.value(),
            .tenant_id = tenant == nullptr ? "default" : tenant,
            .tenant_weight = weight.value_or(1),
            .task_priority = priority.value(),
            .control_socket =
                configured_socket == nullptr ? std::string{} : std::string(configured_socket),
        };
        if (state.is_cooperative_requested) {
            if (state.launch_batch_size != 1) {
                report_diagnostic("[glimmer] cooperative preview requires launch batch size one\n");
                return;
            }
            if (state.native_epoch_configuration) {
                const auto& configuration = *state.native_epoch_configuration;
                state.cooperative_router = std::make_unique<DriverCooperativeRouter>(
                    state.driver, options, state.trace_kernel_launches, state.trace_launch_timings,
                    configuration.extent, configuration.projections,
                    state.persistent_audit ? 16384 : 0, state.should_owner_pump_control);
            } else {
                state.cooperative_router = std::make_unique<DriverCooperativeRouter>(
                    state.driver, options, state.trace_kernel_launches, state.trace_launch_timings,
                    std::optional<control::NativeEpochExtent>{},
                    CudaNativeEpochProjectionOverrides{}, state.persistent_audit ? 16384 : 0,
                    state.should_owner_pump_control);
            }
            state.launch_scheduler_ready = true;
            return;
        }
        state.launch_gate = std::make_unique<LaunchGate>(std::move(options));
        state.launch_tracker = std::make_unique<LaunchCompletionTracker>(
            state.driver, *state.launch_gate, state.trace_launch_timings);
        state.launch_scheduler_ready =
            state.driver.has_event_api() && state.launch_tracker->start();
        if (!state.launch_scheduler_ready) {
            state.launch_tracker.reset();
            state.launch_gate.reset();
            glimmer::interceptor::report_diagnostic(
                "[glimmer] launch scheduler event tracking is unavailable\n");
        }
    } catch (...) {
        state.launch_tracker.reset();
        state.launch_gate.reset();
        glimmer::interceptor::report_diagnostic(
            "[glimmer] launch scheduler initialization failed\n");
    }
}

inline void initialize_persistent_kernels(InterceptorState& state) noexcept {
    try {
        if (std::getenv("GLIMMER_PERSISTENT_WORKER_BLOCKS") == nullptr &&
            !state.is_cooperative_requested) {
            return;
        }
        const auto workers = std::getenv("GLIMMER_PERSISTENT_WORKER_BLOCKS") == nullptr
                                 ? std::optional<MemoryBytes>{3}
                                 : read_memory_limit("GLIMMER_PERSISTENT_WORKER_BLOCKS");
        if (!workers.has_value() || *workers == 0 || *workers > 65536) {
            report_diagnostic(
                "[glimmer] GLIMMER_PERSISTENT_WORKER_BLOCKS is invalid; "
                "PTX transformation disabled\n");
            return;
        }
        state.persistent_kernels = std::make_unique<PersistentKernelRuntime>(
            state.driver, static_cast<std::uint32_t>(*workers), state.trace_kernel_launches,
            state.is_cooperative_requested);
        state.persistent_worker_blocks = static_cast<std::uint32_t>(*workers);
        const auto* runtime_setting = std::getenv("GLIMMER_PERSISTENT_RUNTIME");
        state.is_runtime_persistent_enabled =
            runtime_setting != nullptr && std::strcmp(runtime_setting, "1") == 0;
        if (runtime_setting != nullptr && std::strcmp(runtime_setting, "0") != 0 &&
            std::strcmp(runtime_setting, "1") != 0) {
            report_diagnostic(
                "[glimmer] GLIMMER_PERSISTENT_RUNTIME is invalid; Runtime PTB disabled\n");
        }
    } catch (...) {
        report_diagnostic("[glimmer] persistent kernel initialization failed\n");
    }
}

inline bool initialize_native_driver_dispatch(InterceptorState& state) {
    std::call_once(state.driver_initialization_once, [&state] {
        state.is_native_dispatch_ready.store(state.driver.initialize(), std::memory_order_release);
    });
    return state.is_native_dispatch_ready.load(std::memory_order_acquire);
}

inline void initialize_state_impl(InterceptorState& state) {
    state.trace_kernel_launches = read_kernel_launch_trace_setting();
    state.trace_memory_info = read_memory_info_trace_setting();
    state.trace_launch_timings = read_launch_timing_trace_setting();
    state.is_cooperative_requested = cooperative_driver_preview_requested();
    const auto* persistent_audit = std::getenv("GLIMMER_PERSISTENT_AUDIT");
    state.persistent_audit = persistent_audit && std::strcmp(persistent_audit, "1") == 0;
    const std::optional<glimmer::core::SchedulerMode> configured_scheduler_mode =
        read_scheduler_mode();
    if (!configured_scheduler_mode.has_value()) {
        glimmer::interceptor::report_diagnostic(
            "[glimmer] GLIMMER_SCHEDULER_MODE is invalid; scheduler integration disabled\n");
    } else {
        state.scheduler_mode = *configured_scheduler_mode;
    }

    if (!read_public_native_epoch_configuration(state)) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        report_diagnostic("[glimmer] native epoch startup configuration is invalid\n");
        return;
    }
    if (!read_experimental_control_executor(state)) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        report_diagnostic("[glimmer] experimental control executor configuration is invalid\n");
        return;
    }
    state.is_driver_ready = initialize_native_driver_dispatch(state);
    if (!state.is_driver_ready) {
        glimmer::interceptor::report_diagnostic(
            "[glimmer] CUDA Driver symbol initialization failed\n");
    }

    initialize_launch_scheduler(state);

    initialize_persistent_kernels(state);

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
            const char* task_limit_value = std::getenv("GLIMMER_TASK_MEMORY_LIMIT_BYTES");
            if (task_limit_value != nullptr) {
                const std::optional<MemoryBytes> task_limit =
                    read_memory_limit("GLIMMER_TASK_MEMORY_LIMIT_BYTES");
                if (!task_limit.has_value()) {
                    state.quota_mode = QuotaMode::kInvalidConfiguration;
                    glimmer::interceptor::report_diagnostic(
                        "[glimmer] GLIMMER_TASK_MEMORY_LIMIT_BYTES is invalid\n");
                    return;
                }
                auto task_quota = std::make_unique<ProcessMemoryQuota>(*task_limit);
                state.quota =
                    std::make_unique<CompositeQuota>(std::move(state.quota), std::move(task_quota));
            }
            state.uses_shared_quota = true;
        } else {
            state.quota = std::make_unique<ProcessMemoryQuota>(*limit_bytes);
        }

        auto physical_quota = std::make_unique<DeviceCapacityQuota>(
            static_cast<DeviceMemoryCapacityResolver>(&resolve_device_capacity), &state.driver);
        state.quota =
            std::make_unique<CompositeQuota>(std::move(state.quota), std::move(physical_quota));
        state.quota_mode = QuotaMode::kEnabled;
    } catch (...) {
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        glimmer::interceptor::report_diagnostic(
            "[glimmer] failed to initialize the memory quota\n");
    }
}

inline void initialize_state(InterceptorState& state) noexcept {
    try {
        initialize_state_impl(state);
    } catch (...) {
        state.is_driver_ready = false;
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        report_diagnostic("[glimmer] interceptor initialization failed\n");
    }
}

inline void ensure_state_initialized(InterceptorState& state) {
    std::call_once(state.initialization_once, [&state] {
        {
            const std::lock_guard lock(state.construction_mutex);
            state.has_started_initialization = true;
        }
        initialize_state(state);  // no metadata lock spans loader/SDK/control effects
    });
}

[[nodiscard]] inline bool ensure_initialized(InterceptorState& state) {
    ensure_state_initialized(state);
    return state.is_driver_ready;
}

[[nodiscard]] inline bool should_route_cooperative(const InterceptorState& state) noexcept {
    return state.is_cooperative_requested && !g_is_inside_cooperative_boundary &&
           !is_inside_driver_call() &&
           (!is_inside_runtime_call() || runtime_cooperative_driver_routing_is_enabled());
}

[[nodiscard]] inline CUresult source_native_stream_execution(
    InterceptorState& state, CUstream stream, bool is_ptds,
    const std::function<CUresult()>& original,
    const CudaSourceEventOperation* event_operation = nullptr,
    const CudaNativeGraphExecutable* graph_operation = nullptr) {
    if (!state.is_driver_ready || !state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    auto& gate = cooperative_process_call_gate();
    CudaProcessNativeToken token;
    const auto owned = state.cooperative_router->source_native_token(gate, token);
    if (owned != CUDA_SUCCESS) {
        return owned;
    }
    CUcontext context{};
    CUresult queried{};
    {
        const NativeDriverCallScope native;
        queried = is_ptds ? state.driver.stream_get_context_ptsz(stream, &context)
                          : state.driver.stream_get_context(stream, &context);
    }
    if (queried != CUDA_SUCCESS || !context) {
        return queried == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : queried;
    }
    return state.cooperative_router->native_session_execution(
        token, context, stream, is_ptds, original,
        [&](const auto& issue) {
            const ProcessCallGate::Transition source(gate);
            return issue();
        },
        &gate, event_operation, graph_operation);
}

[[nodiscard]] inline CUresult source_native_current_context(InterceptorState& state,
                                                            CudaProcessNativeToken& token,
                                                            CUcontext& context) {
    if (!state.is_driver_ready || !state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const auto owned =
        state.cooperative_router->source_native_token(cooperative_process_call_gate(), token);
    if (owned != CUDA_SUCCESS) {
        return owned;
    }
    const auto queried = state.driver.context_get_current(&context);
    if (queried != CUDA_SUCCESS || !context) {
        return queried == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : queried;
    }
    return CUDA_SUCCESS;
}

[[nodiscard]] inline CUresult source_native_event_create(InterceptorState& state, CUevent* event,
                                                         unsigned int flags) {
    CudaProcessNativeToken token;
    CUcontext context{};
    const auto ready = source_native_current_context(state, token, context);
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    if (!event || (flags & CU_EVENT_INTERPROCESS) != 0) {
        const auto observed = state.cooperative_router->observe_native_session_context(context);
        return observed == CUDA_SUCCESS ? state.driver.event_create(event, flags) : observed;
    }
    CudaEventRegistrationReservation reservation;
    const auto prepared =
        state.cooperative_router->prepare_native_event(token, context, reservation);
    if (prepared != CUDA_SUCCESS) {
        return prepared;
    }
    CUevent created{};
    const auto native_status = state.driver.event_create(&created, flags);
    if (native_status == CUDA_SUCCESS || created) {
        *event = created;
    }
    return state.cooperative_router->publish_native_event(reservation, created, native_status);
}

[[nodiscard]] inline CUresult cooperative_frontend_current(
    InterceptorState& state, std::shared_ptr<CudaCooperativeFrontend>& frontend) {
    if (!ensure_initialized(state) || !state.cooperative_router || !state.persistent_kernels) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    CUcontext context = nullptr;
    const auto result = state.driver.context_get_current(&context);
    return result == CUDA_SUCCESS ? state.cooperative_router->frontend(context, frontend) : result;
}

class CooperativeBoundaryScope final {
   public:
    CooperativeBoundaryScope() : previous_(g_is_inside_cooperative_boundary) {
        g_is_inside_cooperative_boundary = true;
    }
    ~CooperativeBoundaryScope() {
        g_is_inside_cooperative_boundary = previous_;
    }
    CooperativeBoundaryScope(const CooperativeBoundaryScope&) = delete;
    CooperativeBoundaryScope& operator=(const CooperativeBoundaryScope&) = delete;

   private:
    bool previous_;
};

// Once the SDK accepts a metadata effect, every exceptional exit must fail
// the router as well as the actor. A temporary callback may allocate before
// the noexcept private command is entered. No native cleanup or retry here.
class RuntimeMetadataEffectGuard final {
   public:
    RuntimeMetadataEffectGuard(DriverCooperativeRouter& router, CUcontext context_value)
        : router_(router), context(context_value) {}
    RuntimeMetadataEffectGuard(const RuntimeMetadataEffectGuard&) = delete;
    RuntimeMetadataEffectGuard& operator=(const RuntimeMetadataEffectGuard&) = delete;
    ~RuntimeMetadataEffectGuard() {
        if (!is_complete_) {
            router_.fail_context_owner(context, CUDA_ERROR_UNKNOWN);
        }
    }
    void complete() noexcept {
        is_complete_ = true;
    }

   private:
    DriverCooperativeRouter& router_;
    const CUcontext context;
    bool is_complete_{};
};

struct RuntimeQueuedRelease {
    AllocationRegistry::ReleaseTicket ticket;
    bool was_consumed{};
    bool has_owned_release{};
};
class ReleaseExceptionGuard final {
   public:
    ReleaseExceptionGuard(AllocationRegistry& registry,
                          const AllocationRegistry::ReleaseTicket& ticket,
                          const bool& has_owned_release)
        : registry_(registry),
          ticket_(ticket),
          has_owned_release_(has_owned_release),
          exception_count_(std::uncaught_exceptions()) {}
    ~ReleaseExceptionGuard() {
        if (std::uncaught_exceptions() > exception_count_) {
            if (has_owned_release_) {
                registry_.mark_accounting_degraded();
            } else {
                registry_.cancel_release(ticket_);
            }
        }
    }
    ReleaseExceptionGuard(const ReleaseExceptionGuard&) = delete;
    ReleaseExceptionGuard& operator=(const ReleaseExceptionGuard&) = delete;

   private:
    AllocationRegistry& registry_;
    const AllocationRegistry::ReleaseTicket& ticket_;
    const bool& has_owned_release_;
    int exception_count_;
};
inline thread_local RuntimeQueuedRelease* g_runtime_queued_release{};
class RuntimeQueuedReleaseScope final {
   public:
    explicit RuntimeQueuedReleaseScope(RuntimeQueuedRelease& pending)
        : previous_(g_runtime_queued_release) {
        g_runtime_queued_release = &pending;
    }
    ~RuntimeQueuedReleaseScope() {
        g_runtime_queued_release = previous_;
    }
    RuntimeQueuedReleaseScope(const RuntimeQueuedReleaseScope&) = delete;
    RuntimeQueuedReleaseScope& operator=(const RuntimeQueuedReleaseScope&) = delete;

   private:
    RuntimeQueuedRelease* previous_;
};

class CooperativeReleaseProof final : public CudaQueueReleaseObserver {
   public:
    CooperativeReleaseProof(AllocationRegistry& registry, AllocationRegistry::ReleaseTicket ticket)
        : registry_(registry), k_ticket(ticket) {}
    void confirmed_release(CUdeviceptr pointer) noexcept override {
        if (pointer != k_ticket.identity.device_pointer ||
            !registry_.confirm_cooperative_release(k_ticket)) {
            registry_.mark_accounting_degraded();
            report_diagnostic("[glimmer] cooperative free completion ticket mismatch\n");
        }
    }
    void context_destroyed(CUdeviceptr) noexcept override {
        // Context destruction cannot prove an ambiguous native free or release
        // a context-independent allocation. Preserve accounting ownership.
        registry_.mark_accounting_degraded();
    }

   private:
    AllocationRegistry& registry_;
    const AllocationRegistry::ReleaseTicket k_ticket;
};

[[nodiscard]] inline bool can_queue_cooperative_release(
    InterceptorState& state, const AllocationRegistry::ReleaseTicket& ticket) {
    if (!state.cooperative_router || !ticket.memory_bytes ||
        ticket.memory_bytes > std::numeric_limits<std::size_t>::max()) {
        return false;
    }
    const NativeDriverCallScope scope;
    CUcontext current{};
    const auto& dispatch = state.cooperative_router->copy_range_dispatch();
    CUdeviceptr base{};
    std::size_t extent{};
    const auto pointer = ticket.identity.device_pointer;
    return state.driver.context_get_current(&current) == CUDA_SUCCESS &&
           current == ticket.identity.context &&
           can_defer_cuda_copy_range(dispatch, current, pointer,
                                     static_cast<std::size_t>(ticket.memory_bytes),
                                     CU_MEMORYTYPE_DEVICE) &&
           dispatch.address_range(&base, &extent, pointer) == CUDA_SUCCESS && base == pointer;
}

[[nodiscard]] inline CudaQueuedReleaseAdmission queue_cooperative_release(
    InterceptorState& state, CudaCooperativeFrontend& frontend, CUstream stream,
    bool per_thread_default_stream, const AllocationRegistry::ReleaseTicket& ticket) noexcept {
    try {
        auto observer = std::make_shared<CooperativeReleaseProof>(state.allocations, ticket);
        if (!state.allocations.prepare_cooperative_release(ticket)) {
            return {CUDA_ERROR_UNKNOWN, false};
        }
        return frontend.release(stream, per_thread_default_stream,
                                {ticket.identity.device_pointer,
                                 state.cooperative_router->resources(), std::move(observer)});
    } catch (...) {
        return {CUDA_ERROR_OUT_OF_MEMORY, false};
    }
}

inline CUresult cooperative_event_completion(InterceptorState& state, CUevent event,
                                             bool should_wait, bool& needs_native_boundary) {
    needs_native_boundary = false;
    if (!state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.cooperative_router->logical_event_completion(event, should_wait,
                                                              needs_native_boundary);
}

inline CUresult cooperative_record(InterceptorState& state, CUevent event, CUstream stream,
                                   bool per_thread_default_stream,
                                   const std::function<CUresult()>& original) {
    if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
        const CudaSourceEventOperation descriptor{CudaSourceEventOperation::Kind::kRecord, event,
                                                  0};
        return source_native_stream_execution(state, stream, per_thread_default_stream, original,
                                              &descriptor);
    }
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    const auto ready = cooperative_frontend_current(state, frontend);
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    CUcontext event_context = nullptr;
    std::shared_ptr<CudaCooperativeFrontend> event_owner;
    const auto found = state.cooperative_router->event_frontend(event, event_context, event_owner);
    if (found == CUDA_SUCCESS && event_owner == frontend) {
        CudaEventProjection projection;
        auto projected = state.cooperative_router->event_projection(event, projection);
        if (projected != CUDA_SUCCESS) {
            return projected;
        }
        if (!projection.is_logical() && !projection.has_native_completion() &&
            !projection.issued_native_prefix()) {
            // Retain the binding and historical waits. Prove all native use
            // complete before queue ownership resumes against this handle.
            const auto completed =
                intercept_cooperative_native_boundary([] { return CUDA_SUCCESS; });
            if (completed != CUDA_SUCCESS) {
                return completed;
            }
            projection = {};
            projected = state.cooperative_router->event_projection(event, projection);
            if (projected != CUDA_SUCCESS) {
                return projected;
            }
        }
        std::shared_ptr<CudaQueueEventRecordGuard> guard;
        const auto prepared =
            state.cooperative_router->prepare_event_record_guard(projection, guard);
        if (prepared != CUDA_SUCCESS) {
            return prepared;
        }
        return frontend->record_event(event, stream, per_thread_default_stream, std::move(guard));
    }
    if (found != CUDA_SUCCESS && found != CUDA_ERROR_INVALID_HANDLE) {
        return found;
    }
    return intercept_cooperative_native_boundary(original);
}

inline CUresult cooperative_wait(InterceptorState& state, CUstream stream, CUevent event,
                                 bool per_thread_default_stream,
                                 const std::function<CUresult()>& original) {
    if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return source_native_stream_execution(state, stream, per_thread_default_stream, original);
    }
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    const auto ready = cooperative_frontend_current(state, frontend);
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    CUcontext event_context = nullptr;
    std::shared_ptr<CudaCooperativeFrontend> event_owner;
    const auto found = state.cooperative_router->event_frontend(event, event_context, event_owner);
    if (found == CUDA_SUCCESS && event_owner == frontend) {
        CudaEventProjection projection;
        const auto projected = state.cooperative_router->event_projection(event, projection);
        if (projected != CUDA_SUCCESS) {
            return projected;
        }
        if (!projection.is_logical()) {
            return intercept_cooperative_native_boundary(original);
        }
        return frontend->wait_event(stream, per_thread_default_stream, event);
    }
    if (found != CUDA_SUCCESS && found != CUDA_ERROR_INVALID_HANDLE) {
        return found;
    }
    return intercept_cooperative_native_boundary(original);
}

[[nodiscard]] inline void* find_proc_address_interceptor(const char* symbol, cuuint64_t flags,
                                                         int cuda_version = 12600) noexcept {
    const std::string_view name{symbol};
    if ((name == "cuLibraryLoadData" || name == "cuLibraryLoadFromFile" ||
         name == "cuLibraryGetModule" || name == "cuLibraryUnload") &&
        cuda_version < 12000)
        return nullptr;
    if (name == "cuCtxDestroy") {
        return cuda_version < 2000
                   ? nullptr
                   : find_interceptor_symbol(
                         cuda_version < 4000 ? "cuCtxDestroy" : "cuCtxDestroy_v2", false);
    }
    if (name == "cuCtxDestroy_v2") {
        return nullptr;
    }
    if (name == "cuDevicePrimaryCtxRelease") {
        if (cuda_version < 7000) {
            return nullptr;
        }
        return find_interceptor_symbol(
            cuda_version < 11000 ? "cuDevicePrimaryCtxRelease" : "cuDevicePrimaryCtxRelease_v2",
            false);
    }
    if (name == "cuDevicePrimaryCtxRelease_v2" && cuda_version < 11000) {
        return nullptr;
    }
    if (name == "cuDevicePrimaryCtxReset") {
        if (cuda_version < 7000) {
            return nullptr;
        }
        return find_interceptor_symbol(
            cuda_version < 11000 ? "cuDevicePrimaryCtxReset" : "cuDevicePrimaryCtxReset_v2", false);
    }
    if (name == "cuDevicePrimaryCtxReset_v2" && cuda_version < 11000) {
        return nullptr;
    }
    if (name == "cuGraphInstantiate") {
        return find_interceptor_symbol(cuda_version < 11000   ? "cuGraphInstantiate"
                                       : cuda_version < 11040 ? "cuGraphInstantiate_v2"
                                                              : "cuGraphInstantiateWithFlags",
                                       false);
    }
    if (name == "cuGraphExecUpdate") {
        return find_interceptor_symbol(
            cuda_version < 12000 ? "cuGraphExecUpdate" : "cuGraphExecUpdate_v2", false);
    }
    if (name == "cuStreamBeginCapture" || name == "cuStreamBeginCapture_ptsz") {
        if (cuda_version < 10000) {
            return nullptr;
        }
        const bool is_ptds = name == "cuStreamBeginCapture_ptsz" ||
                             (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0;
        return find_interceptor_symbol(
            cuda_version < 10010 ? "cuStreamBeginCapture" : "cuStreamBeginCapture_v2", is_ptds);
    }
    const bool is_copy = name.starts_with("cuMemcpyHtoD") || name.starts_with("cuMemcpyDtoH") ||
                         name.starts_with("cuMemcpyDtoD") || name == "cuMemcpy" ||
                         name == "cuMemcpyAsync";
    const bool is_memset = name.starts_with("cuMemsetD8") || name.starts_with("cuMemsetD16") ||
                           name.starts_with("cuMemsetD32");
    // Never substitute a modern pointer/size ABI for an old resolver result.
    // CUDA's cudaTypedefs.h identifies these versioned signatures.
    if (((is_copy || is_memset || name == "cuMemAllocHost" ||
          name == "cuMemHostGetDevicePointer") &&
         cuda_version < 3020) ||
        ((name == "cuMemcpy" || name == "cuMemcpyAsync") && cuda_version < 4000) ||
        (name == "cuMemHostRegister" && cuda_version < 6050) ||
        ((name == "cuStreamGetCtx" || name == "cuStreamGetCtx_ptsz") &&
         (cuda_version < 9020 || cuda_version >= 12050)) ||
        ((is_copy || is_memset) && (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0 &&
         cuda_version < 7000)) {
        return nullptr;
    }
    return glimmer::interceptor::find_interceptor_symbol(
        symbol, (flags & CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) != 0);
}

inline void replace_runtime_proc_address(const DriverDispatch& driver, const char* symbol,
                                         void** address, cuuint64_t flags, int cuda_version) {
    // Runtime cache bindings outlive a trusted private initialization call.
    // Keep them controlled; their later invocation still honors native guards.
    // Driver-only private lookups have no Runtime frame and remain original.
    if (*address == nullptr || (is_inside_driver_call() && !is_inside_runtime_call()) ||
        !runtime_persistent_lookup_is_enabled()) {
        return;
    }
    if (std::strcmp(symbol, "cuGetProcAddress") == 0) {
        // libcudart caches both resolver ABIs before querying its launch entry.
        // Match the actual native pointer rather than guessing from a version.
        if (driver.is_proc_address_function(*address, true)) {
            *address = reinterpret_cast<void*>(&runtime_driver_get_proc_address_v2);
        } else if (driver.is_proc_address_function(*address, false)) {
            *address = reinterpret_cast<void*>(&runtime_driver_get_proc_address);
        }
    } else if (std::strcmp(symbol, "cuLaunchKernel") == 0 ||
               cooperative_runtime_preview_requested()) {
        if (auto* replacement = find_proc_address_interceptor(symbol, flags, cuda_version);
            replacement != nullptr) {
            *address = replacement;
        }
    }
}

struct ContextIdentity {
    CUcontext context = nullptr;
    CUdevice device = 0;
};

[[nodiscard]] inline std::optional<ContextIdentity> capture_context_identity(
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

[[nodiscard]] inline std::optional<ContextIdentity> capture_stream_identity(
    InterceptorState& state, CUstream stream, bool per_thread_default_stream) noexcept {
    if (stream == nullptr) {
        return capture_context_identity(state);
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
    if (device_result == CUDA_ERROR_NOT_SUPPORTED) {
        const std::optional<ContextIdentity> current = capture_context_identity(state);
        if (!current.has_value() || current->context != context) {
            return std::nullopt;
        }
        device = current->device;
    } else if (device_result != CUDA_SUCCESS) {
        return std::nullopt;
    }
    if (device < 0) {
        return std::nullopt;
    }
    return ContextIdentity{.context = context, .device = device};
}

[[nodiscard]] inline glimmer::interceptor::AllocationIdentity make_allocation_identity(
    CUdeviceptr device_pointer, const ContextIdentity& context) noexcept {
    return glimmer::interceptor::AllocationIdentity{
        .device_pointer = device_pointer, .context = context.context, .device = context.device};
}

[[nodiscard]] inline glimmer::interceptor::AllocationIdentity make_allocation_identity(
    const void* device_pointer, const ContextIdentity& context) noexcept {
    return make_allocation_identity(
        static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(device_pointer)), context);
}

[[nodiscard]] inline AsyncStreamIdentity make_async_stream_identity(
    CUstream stream, const ContextIdentity& context, bool per_thread_default_stream) {
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

template <typename Invocation>
[[nodiscard]] cudaError_t forward_untracked_runtime_call(InterceptorState& state,
                                                         Invocation&& invoke) {
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? std::forward<Invocation>(invoke)()
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return cudaErrorNotSupported;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    return std::forward<Invocation>(invoke)();
}

inline CUresult execute_cooperative_graph(InterceptorState& state, CUgraphExec executable,
                                          CUstream stream, bool is_ptds,
                                          const std::function<CUresult()>& original) {
    auto& router = *state.cooperative_router;
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_native_only_on_current_thread())
        return router.native_graph_execution(executable, stream, is_ptds, original);
    CudaNativeGraphExecutable owned;
    const auto found = router.graphs().lookup(executable, owned);
    if (found != CUDA_SUCCESS) return found;
    // Source admission drained logical producers. Native session/executor proof,
    // not a normal queue prefix, retains the original graph's physical effects.
    return source_native_stream_execution(
        state, stream, is_ptds,
        [&] {
            CUcontext context{};
            const auto current = state.driver.context_get_current(&context);
            if (current != CUDA_SUCCESS || context != owned.context())
                return current == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : current;
            const auto valid = router.graphs().is_current(owned);
            return valid == CUDA_SUCCESS ? original() : valid;
        },
        nullptr, &owned);
}
}  // namespace interceptor_detail
#pragma GCC visibility pop
using namespace interceptor_detail;

// Defined in driver_api_memory.cc.
void finalize_async_release(InterceptorState& state, DeviceId device, MemoryBytes released_bytes);
void finalize_async_releases(
    InterceptorState& state,
    const std::vector<AllocationRegistry::DeviceReleaseSummary>& summaries);

template <typename AllocateFunction>
CUresult intercept_tracked_allocation(
    CUdeviceptr* device_pointer, MemoryBytes memory_bytes, AllocateFunction&& allocate,
    bool is_private_progress = false,
    AllocationProvenance provenance = AllocationProvenance::kUnknown) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (!is_private_progress && (glimmer::interceptor::is_inside_driver_call() ||
                                 glimmer::interceptor::is_inside_runtime_call())) {
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

    CUresult allocation_result = CUDA_ERROR_UNKNOWN;
    try {
        allocation_result = std::forward<AllocateFunction>(allocate)();
    } catch (...) {
        if (is_private_progress) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
        }
        throw;
    }
    if (allocation_result != CUDA_SUCCESS) {
        if (is_private_progress && *device_pointer != 0) {
            // Native output ownership is ambiguous. Do not release its quota
            // or automatically replay allocation/free through the pool.
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
        }
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
        } else if (is_private_progress) {
            *device_pointer = 0;
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != CUDA_SUCCESS ? cleanup_result : CUDA_ERROR_UNKNOWN;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context), memory_bytes,
                                  AllocationScope::kContextBound, provenance)) {
        state.allocations.mark_accounting_degraded();
        const CUresult cleanup_result = state.driver.mem_free(*device_pointer);
        if (cleanup_result != CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (is_private_progress) {
            *device_pointer = 0;
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

}  // namespace glimmer::interceptor
