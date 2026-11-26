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

namespace {

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

thread_local bool g_is_inside_proc_address_v2 = false;
thread_local bool g_is_inside_cooperative_boundary = false;

struct ThreadLaunchBatch final {
    glimmer::core::TaskId task_id = 0;
    bool final_launch = false;
};

thread_local ThreadLaunchBatch g_thread_launch_batch;

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

[[nodiscard]] std::optional<MemoryBytes> read_memory_limit(const char* variable_name) {
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

[[nodiscard]] std::optional<MemoryBytes> read_quota_limit() {
    return read_memory_limit("GLIMMER_MEMORY_LIMIT_BYTES");
}

[[nodiscard]] std::optional<std::size_t> read_positive_size(const char* variable_name) {
    const std::optional<MemoryBytes> value = read_memory_limit(variable_name);
    if (!value.has_value() || value.value() == 0 ||
        value.value() > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(value.value());
}

[[nodiscard]] bool read_public_native_epoch_configuration(InterceptorState& state) {
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

[[nodiscard]] bool read_experimental_control_executor(InterceptorState& state) {
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

[[nodiscard]] std::optional<std::size_t> read_nonnegative_size(const char* variable_name) {
    const std::optional<MemoryBytes> value = read_memory_limit(variable_name);
    if (!value.has_value() || value.value() > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(value.value());
}

[[nodiscard]] std::optional<std::uint32_t> read_positive_weight() {
    const std::optional<MemoryBytes> value = read_memory_limit("GLIMMER_SCHEDULER_WEIGHT");
    if (!value.has_value() || value.value() == 0 ||
        value.value() > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value.value());
}

[[nodiscard]] std::optional<std::uint32_t> read_scheduler_priority() {
    const std::optional<MemoryBytes> value = read_memory_limit("GLIMMER_SCHEDULER_PRIORITY");
    if (!value.has_value() || value.value() > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value.value());
}

[[nodiscard]] std::optional<std::uint32_t> read_scheduler_priority_threshold() {
    const std::optional<MemoryBytes> value =
        read_memory_limit("GLIMMER_SCHEDULER_PRIORITY_THRESHOLD");
    if (!value.has_value() || value.value() > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(value.value());
}

[[nodiscard]] std::optional<glimmer::core::SchedulingPolicy> read_scheduling_policy() {
    const char* value = std::getenv("GLIMMER_SCHEDULER_POLICY");
    if (value == nullptr) {
        return glimmer::core::SchedulingPolicy::kWeightedRoundRobin;
    }
    return glimmer::core::parse_scheduling_policy(value);
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

[[nodiscard]] std::optional<glimmer::core::SchedulerMode> read_scheduler_mode() {
    const char* value = std::getenv("GLIMMER_SCHEDULER_MODE");
    if (value == nullptr) {
        return glimmer::core::SchedulerMode::kOff;
    }
    return glimmer::core::parse_scheduler_mode(value);
}

[[nodiscard]] bool read_trace_setting(const char* variable_name) noexcept {
    const char* value = std::getenv(variable_name);
    return value != nullptr && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0);
}

[[nodiscard]] bool read_kernel_launch_trace_setting() noexcept {
    return read_trace_setting("GLIMMER_TRACE_KERNEL_LAUNCHES");
}

[[nodiscard]] bool read_memory_info_trace_setting() noexcept {
    return read_trace_setting("GLIMMER_TRACE_MEMORY_INFO");
}

[[nodiscard]] bool read_launch_timing_trace_setting() noexcept {
    return read_trace_setting("GLIMMER_TRACE_LAUNCH_TIMINGS");
}

bool resolve_device_capacity(void* context, DeviceId device,
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

std::uint64_t next_launch_observation_count(std::atomic<std::uint64_t>& count) noexcept {
    std::uint64_t current = count.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<std::uint64_t>::max()) {
        if (count.compare_exchange_weak(current, current + 1, std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
            return current + 1;
        }
    }
    return current;
}

[[nodiscard]] std::optional<MemoryBytes> checked_multiply(MemoryBytes left, MemoryBytes right) {
    if (right != 0 && left > std::numeric_limits<MemoryBytes>::max() / right) {
        return std::nullopt;
    }
    return left * right;
}

[[nodiscard]] std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point started,
    std::chrono::steady_clock::time_point finished) noexcept {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started);
    return elapsed.count() < 0 ? std::uint64_t{0} : static_cast<std::uint64_t>(elapsed.count());
}

// The setup owns allocator-backed scheduler state but converts every setup
// failure into a diagnostic and a disabled enforcement path.
// NOLINTNEXTLINE(bugprone-exception-escape)
void initialize_launch_scheduler(InterceptorState& state) noexcept {
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

void initialize_persistent_kernels(InterceptorState& state) noexcept {
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

bool initialize_native_driver_dispatch(InterceptorState& state) {
    std::call_once(state.driver_initialization_once, [&state] {
        state.is_native_dispatch_ready.store(state.driver.initialize(), std::memory_order_release);
    });
    return state.is_native_dispatch_ready.load(std::memory_order_acquire);
}

void initialize_state_impl(InterceptorState& state) {
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

void initialize_state(InterceptorState& state) noexcept {
    try {
        initialize_state_impl(state);
    } catch (...) {
        state.is_driver_ready = false;
        state.quota_mode = QuotaMode::kInvalidConfiguration;
        report_diagnostic("[glimmer] interceptor initialization failed\n");
    }
}

void ensure_state_initialized(InterceptorState& state) {
    std::call_once(state.initialization_once, [&state] {
        {
            const std::lock_guard lock(state.construction_mutex);
            state.has_started_initialization = true;
        }
        initialize_state(state);  // no metadata lock spans loader/SDK/control effects
    });
}

[[nodiscard]] bool ensure_initialized(InterceptorState& state) {
    ensure_state_initialized(state);
    return state.is_driver_ready;
}

[[nodiscard]] bool should_route_cooperative(const InterceptorState& state) noexcept {
    return state.is_cooperative_requested && !g_is_inside_cooperative_boundary &&
           !is_inside_driver_call() &&
           (!is_inside_runtime_call() || runtime_cooperative_driver_routing_is_enabled());
}

[[nodiscard]] CUresult source_native_stream_execution(
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

[[nodiscard]] CUresult source_native_current_context(InterceptorState& state,
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

[[nodiscard]] CUresult source_native_event_create(InterceptorState& state, CUevent* event,
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

[[nodiscard]] CUresult cooperative_frontend_current(
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
thread_local RuntimeQueuedRelease* g_runtime_queued_release{};
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

[[nodiscard]] bool can_queue_cooperative_release(InterceptorState& state,
                                                 const AllocationRegistry::ReleaseTicket& ticket) {
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

[[nodiscard]] CudaQueuedReleaseAdmission queue_cooperative_release(
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

CUresult cooperative_event_completion(InterceptorState& state, CUevent event, bool should_wait,
                                      bool& needs_native_boundary) {
    needs_native_boundary = false;
    if (!state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return state.cooperative_router->logical_event_completion(event, should_wait,
                                                              needs_native_boundary);
}

CUresult cooperative_record(InterceptorState& state, CUevent event, CUstream stream,
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

CUresult cooperative_wait(InterceptorState& state, CUstream stream, CUevent event,
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

[[nodiscard]] void* find_proc_address_interceptor(const char* symbol, cuuint64_t flags,
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

void replace_runtime_proc_address(const DriverDispatch& driver, const char* symbol, void** address,
                                  cuuint64_t flags, int cuda_version) {
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

}  // namespace

bool cooperative_driver_preview_requested() noexcept {
    const char* value = std::getenv("GLIMMER_COOPERATIVE_DRIVER_PREVIEW");
    return (value != nullptr && std::strcmp(value, "1") == 0) ||
           cooperative_runtime_preview_requested();
}

ProcessCallGate& cooperative_process_call_gate() noexcept {
    static ProcessCallGate gate;
    return gate;
}

DriverCooperativeRouter* cooperative_source_router() noexcept {
    try {
        if (!cooperative_process_call_gate().is_entered_by_current_thread()) {
            return nullptr;
        }
        auto& state = get_state();
        return ensure_initialized(state) && state.is_cooperative_requested
                   ? state.cooperative_router.get()
                   : nullptr;
    } catch (...) {
        return nullptr;
    }
}

std::optional<PersistentBlasContext> persistent_blas_current_context() noexcept {
    try {
        auto& gate = cooperative_process_call_gate();
        if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread() ||
            is_inside_driver_call() || is_inside_runtime_call())
            return std::nullopt;
        auto& state = get_state();
        PersistentBlasContext result;
        if (!ensure_initialized(state) || !should_route_cooperative(state) ||
            state.driver.context_get_current(&result.context) != CUDA_SUCCESS || !result.context)
            return std::nullopt;
        static const auto get_id =
            reinterpret_cast<decltype(&cuCtxGetId)>(resolve_native_driver_function("cuCtxGetId"));
        const NativeDriverCallScope native;
        if (!get_id || get_id(result.context, &result.id) != CUDA_SUCCESS) return std::nullopt;
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<CUresult> try_intercept_persistent_sgemm(CUcontext handle_context,
                                                       PersistentSgemmProblem problem,
                                                       CUdeviceptr a, CUdeviceptr b, CUdeviceptr c,
                                                       CUdeviceptr bias, CUdeviceptr workspace,
                                                       std::size_t workspace_bytes) {
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread() ||
        is_inside_driver_call() || is_inside_runtime_call())
        return std::nullopt;
    const auto layout = persistent_sgemm_layout(problem);
    if (!handle_context || !layout || !a || !b || !c || a % 4 || b % 4 || c % 4)
        return std::nullopt;
    if ((layout->bias_bytes != 0) != (bias != 0) || bias % 4 || (workspace_bytes && !workspace))
        return std::nullopt;
    auto& state = get_state();
    if (!ensure_initialized(state) || !should_route_cooperative(state) ||
        !state.persistent_kernels || !state.cooperative_router)
        return std::nullopt;
    CUcontext context{};
    CUdevice device{};
    CUstreamCaptureStatus capture{};
    if (state.driver.context_get_current(&context) != CUDA_SUCCESS || context != handle_context ||
        state.driver.context_get_device(&device) != CUDA_SUCCESS ||
        state.driver.stream_capture_status(nullptr, &capture, false) != CUDA_SUCCESS ||
        capture != CU_STREAM_CAPTURE_STATUS_NONE)
        return std::nullopt;
    {
        const NativeDriverCallScope native;
        const auto& dispatch = state.cooperative_router->copy_range_dispatch();
        if (!has_independent_gemm_range(dispatch, state.allocations, {a, context, device},
                                        layout->a_bytes) ||
            !has_independent_gemm_range(dispatch, state.allocations, {b, context, device},
                                        layout->b_bytes) ||
            !has_independent_gemm_range(dispatch, state.allocations, {c, context, device},
                                        layout->c_bytes) ||
            (bias && !has_independent_gemm_range(dispatch, state.allocations,
                                                 {bias, context, device}, layout->bias_bytes)) ||
            (workspace_bytes &&
             !has_independent_gemm_range(dispatch, state.allocations, {workspace, context, device},
                                         workspace_bytes)))
            return std::nullopt;
    }
    // The range qualification above bounds these half-open endpoints.
    if ((a < c + layout->c_bytes && c < a + layout->a_bytes) ||
        (b < c + layout->c_bytes && c < b + layout->b_bytes) ||
        (bias && bias < c + layout->c_bytes && c < bias + layout->bias_bytes))
        return std::nullopt;
    if (workspace_bytes) {
        const auto overlaps_workspace = [&](CUdeviceptr pointer, std::size_t bytes) {
            return bytes && workspace < pointer + bytes && pointer < workspace + workspace_bytes;
        };
        if (overlaps_workspace(a, layout->a_bytes) || overlaps_workspace(b, layout->b_bytes) ||
            overlaps_workspace(c, layout->c_bytes) || overlaps_workspace(bias, layout->bias_bytes))
            return std::nullopt;
    }
    auto code = state.persistent_kernels->owned_sgemm(context);
    if (!code) {
        const auto prepared = intercept_cooperative_native_boundary(
            [&] { return state.persistent_kernels->prepare_owned_sgemm(context); });
        if (prepared == CUDA_ERROR_NOT_SUPPORTED) return std::nullopt;
        if (prepared != CUDA_SUCCESS) return prepared;
        code = state.persistent_kernels->owned_sgemm(context);
        if (!code) return CUDA_ERROR_UNKNOWN;
    }
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    const auto ready = state.cooperative_router->frontend(context, frontend);
    if (ready != CUDA_SUCCESS) return ready;
    struct Owners {
        std::shared_ptr<void> code;
        std::shared_ptr<void> router;
    };
    const auto owners =
        std::make_shared<Owners>(Owners{code->resources, state.cooperative_router->resources()});
    if (!owners->code || !owners->router) return CUDA_ERROR_UNKNOWN;
    CooperativeKernelVariant variant{
        context,
        code->function,
        {kPersistentSgemmParameters.begin(), kPersistentSgemmParameters.end()},
        {layout->grid[0] * layout->grid[1], 256},
        code->worker_blocks};
    if (state.persistent_audit)
        variant.audit = CooperativeKernelVariant::AuditParameters{14, 15, true};
    CUdeviceptr first_tile_observer{};
    CUdeviceptr tile_counts{};
    auto reduction_depth = layout->reduction_depth;
    std::array<void*, 17> arguments{&a,
                                    &b,
                                    &c,
                                    &problem.rows,
                                    &problem.columns,
                                    &problem.depth,
                                    &problem.leading_a,
                                    &problem.leading_b,
                                    &problem.leading_c,
                                    &problem.transpose_b,
                                    &problem.alpha,
                                    &problem.beta,
                                    &problem.transpose_a,
                                    &bias,
                                    &first_tile_observer,
                                    &tile_counts,
                                    &reduction_depth};
    const auto result = frontend->launch(variant, nullptr, false, layout->grid, layout->block,
                                         arguments.data(), owners);
    if (result == CUDA_SUCCESS && state.trace_kernel_launches)
        report_diagnostic(bias ? "[glimmer] persistent_lt_bias accepted=1\n"
                               : "[glimmer] persistent_sgemm accepted=1\n");
    return result;
}

CUresult configure_cooperative_native_epochs(
    const CudaNativeEpochConfiguration& configuration) noexcept {
    try {
        auto& gate = cooperative_process_call_gate();
        if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread() ||
            is_inside_driver_call() || is_inside_runtime_call())
            return CUDA_ERROR_NOT_PERMITTED;
        if (!control::valid_native_epoch_extent(configuration.extent))
            return CUDA_ERROR_INVALID_VALUE;
        const ProcessCallGate::Transition transition(gate);
        auto& state = get_state();
        const std::lock_guard lock(state.construction_mutex);
        if (state.has_started_initialization || state.native_epoch_configuration)
            return CUDA_ERROR_NOT_PERMITTED;
        state.native_epoch_configuration = configuration;
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

std::optional<std::size_t> cooperative_submitted_release_count(CUcontext context) noexcept {
    try {
        if (!context || !cooperative_process_call_gate().is_entered_by_current_thread())
            return std::nullopt;
        const auto submitted = get_state().allocations.submitted_async_releases(true);
        if (!submitted) return std::nullopt;
        return static_cast<std::size_t>(std::ranges::count_if(
            *submitted, [&](const auto& ticket) { return ticket.identity.context == context; }));
    } catch (...) {
        return std::nullopt;
    }
}

CooperativeCallScope::CooperativeCallScope() {
    if (cooperative_driver_preview_requested() && !is_inside_driver_call() &&
        !g_is_inside_cooperative_boundary) {
        auto& gate = cooperative_process_call_gate();
        const bool is_outer = !gate.is_entered_by_current_thread();
        call_.emplace(gate);
        if (is_outer && gate.is_native_only_on_current_thread()) {
            auto& state = get_state();
            if (!state.cooperative_router)
                throw std::runtime_error("missing retained source owner");
            const CooperativeBoundaryScope boundary;
            const auto finished = state.cooperative_router->try_finish_source_capture(gate, true);
            if (finished != CUDA_SUCCESS && finished != CUDA_ERROR_NOT_READY)
                throw std::runtime_error("failed retained source ownership");
        }
    }
}

CUresult intercept_cooperative_native_boundary(const std::function<CUresult()>& operation) {
    if (!cooperative_driver_preview_requested() || is_inside_driver_call() ||
        (is_inside_runtime_call() && !runtime_cooperative_driver_routing_is_enabled()) ||
        g_is_inside_cooperative_boundary) {
        return operation();
    }
    InterceptorState& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router || !state.persistent_kernels) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const CooperativeBoundaryScope scope;
    return state.cooperative_router->native_boundary(
        operation, [] { return intercept_context_synchronize(); });
}

CUresult intercept_cooperative_memory_boundary(const std::function<CUresult()>& operation) {
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_native_only_on_current_thread()) {
        return intercept_cooperative_native_boundary(operation);
    }
    auto& state = get_state();
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!should_route_cooperative(state)) {
        return operation();
    }
    if (!state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const CooperativeBoundaryScope scope;
    return state.cooperative_router->source_memory_metadata(gate, operation);
}

CUresult intercept_cooperative_module_boundary(const std::function<CUresult()>& operation) {
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_native_only_on_current_thread()) {
        return intercept_cooperative_native_boundary(operation);
    }
    auto& state = get_state();
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!should_route_cooperative(state)) {
        return operation();
    }
    if (!state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const CooperativeBoundaryScope scope;
    return state.cooperative_router->source_module_metadata(gate, operation);
}

CUresult intercept_graph_instantiation(CUgraphExec* output, CUgraph graph,
                                       bool should_prepare_effects,
                                       const std::function<CUresult(CUgraphExec*)>& original) {
    auto& state = get_state();
    ensure_state_initialized(state);
    if (!should_route_cooperative(state)) {
        return original(output);
    }
    if (!output) {
        return intercept_cooperative_native_boundary([&] { return original(output); });
    }
    if (!state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    CUcontext context{};
    const auto current = state.driver.context_get_current(&context);
    if (current != CUDA_SUCCESS || !context) {
        return intercept_cooperative_native_boundary([&] { return original(output); });
    }
    CudaNativeGraphReservation reservation;
    const auto prepared = state.cooperative_router->prepare_graph(context, graph, reservation,
                                                                  should_prepare_effects);
    if (prepared != CUDA_SUCCESS) {
        return prepared;
    }
    return intercept_cooperative_native_boundary([&] {
        // Native failure may leave the application's output uninitialized.
        // Never inspect it: stage the exact native output in initialized storage.
        CUgraphExec native_output{};
        const auto result = original(&native_output);
        const auto published =
            state.cooperative_router->graphs().publish(reservation, native_output, result);
        if (native_output || result == CUDA_SUCCESS) {
            *output = native_output;
        }
        return published;
    });
}

CUresult intercept_graph_mutation(CUgraphExec executable, bool is_destroy,
                                  const std::function<CUresult()>& original) {
    auto& state = get_state();
    ensure_state_initialized(state);
    if (!should_route_cooperative(state)) {
        return original();
    }
    if (!state.cooperative_router) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    state.cooperative_router->graphs().invalidate(executable);
    // Executable mutation is CPU/native graph metadata. Do not drain unrelated
    // low work or treat successful native destruction as GPU completion proof.
    const auto result = original();
    if (is_destroy && result == CUDA_SUCCESS) {
        state.cooperative_router->graphs().forget(executable);
    }
    return result;
}

CUresult cooperative_graph_executable(CUgraphExec executable,
                                      CudaNativeGraphExecutable& output) noexcept {
    try {
        auto& state = get_state();
        ensure_state_initialized(state);
        return state.cooperative_router
                   ? state.cooperative_router->graphs().lookup(executable, output)
                   : CUDA_ERROR_NOT_SUPPORTED;
    } catch (...) {
        return CUDA_ERROR_UNKNOWN;
    }
}

CUresult intercept_graph_event_update(CUgraphExec executable, CUgraphNode node, CUevent event,
                                      bool is_record, const std::function<CUresult()>& original) {
    auto& state = get_state();
    ensure_state_initialized(state);
    if (!should_route_cooperative(state)) return original();
    if (!state.cooperative_router) return CUDA_ERROR_NOT_SUPPORTED;
    auto& router = *state.cooperative_router;
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            router.prepare_graph_event_update(executable, node, event, is_record, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE) return prepared;
    }
    const auto status = original();
    return replacement ? router.graphs().publish_event_update(replacement, status) : status;
}

CUresult intercept_graph_node_update(CUgraphExec executable, CUgraphNode node,
                                     std::optional<CUgraphNodeType> expected_type,
                                     const std::function<CUresult()>& original) {
    auto& state = get_state();
    ensure_state_initialized(state);
    if (!should_route_cooperative(state)) return original();
    if (!state.cooperative_router) return CUDA_ERROR_NOT_SUPPORTED;
    auto& graphs = state.cooperative_router->graphs();
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            graphs.prepare_node_update(executable, node, expected_type, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE) return prepared;
    }
    const auto status = original();
    return replacement ? graphs.publish_event_update(replacement, status) : status;
}

CUresult intercept_graph_update(CUgraphExec executable, CUgraph graph,
                                std::optional<CUgraphNode> child_node,
                                const std::function<CUresult()>& original) {
    auto& state = get_state();
    ensure_state_initialized(state);
    if (!should_route_cooperative(state)) return original();
    if (!state.cooperative_router) return CUDA_ERROR_NOT_SUPPORTED;
    auto& router = *state.cooperative_router;
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            router.prepare_graph_update(executable, graph, child_node, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE) return prepared;
    }
    const auto status = original();
    return replacement ? router.graphs().publish_event_update(replacement, status) : status;
}

cudaError_t intercept_cooperative_runtime_stream_destroy(
    cudaStream_t stream, const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    auto& gate = cooperative_process_call_gate();
    if (!original) {
        return cudaErrorInvalidValue;
    }
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary) {
        return original();
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope scope;
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{}, has_native_effect{};
    const auto invoke = [&] {
        was_invoked = true;
        native_status = original();
        has_native_effect = native_status == cudaSuccess;
        return has_native_effect ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    };
    const auto detach = [&](CUcontext context) {
        if (state.quota_mode != QuotaMode::kEnabled) {
            return CUDA_SUCCESS;
        }
        const AsyncStreamIdentity saved{.stream = stream, .context = context};
        if (state.allocations.is_accounting_degraded() ||
            !state.allocations.detach_async_releases_for_stream(saved)) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    };
    const auto status =
        gate.is_native_only_on_current_thread()
            ? state.cooperative_router->source_stream_destroy(gate, stream, invoke, detach,
                                                              CudaSourceMemoryOrigin::kRuntime,
                                                              &has_native_effect)
            : state.cooperative_router->native_stream_destroy(
                  stream, invoke, detach, [] { return intercept_context_synchronize(); },
                  &has_native_effect);
    return was_invoked && native_status != cudaSuccess ? native_status
           : status == CUDA_SUCCESS                    ? native_status
                                                       : cudaErrorUnknown;
}

cudaError_t intercept_cooperative_runtime_event_record(
    cudaEvent_t event, cudaStream_t stream, bool is_ptds, unsigned int flags,
    const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    auto& gate = cooperative_process_call_gate();
    if (!original) {
        return cudaErrorInvalidValue;
    }
    if (!cooperative_runtime_preview_requested() || !gate.is_native_only_on_current_thread() ||
        is_inside_driver_call() || g_is_inside_cooperative_boundary) {
        return original();
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope scope;
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    bool has_native_effect{};
    const auto status = state.cooperative_router->source_runtime_event_record(
        gate, event, stream, is_ptds, flags,
        [&] {
            was_invoked = true;
            native_status = original();
            has_native_effect = native_status == cudaSuccess;
            return has_native_effect ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        },
        &has_native_effect);
    return was_invoked && native_status != cudaSuccess ? native_status
           : status == CUDA_SUCCESS                    ? native_status
                                                       : cudaErrorUnknown;
}

std::optional<cudaError_t> intercept_normal_runtime_event_destroy(
    cudaEvent_t event, const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!state.cooperative_router) {
        return std::nullopt;
    }
    CudaEventProjection owned;
    const auto found = state.cooperative_router->event_projection(event, owned);
    const CooperativeBoundaryScope scope;
    if (found == CUDA_SUCCESS && !owned.allows_logical_binding()) {
        const auto status = original();
        return status == cudaSuccess &&
                       state.cooperative_router->retire_metadata_event(owned) != CUDA_SUCCESS
                   ? cudaErrorUnknown
                   : status;
    }
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{}, has_native_effect{};
    const auto status = state.cooperative_router->native_event_destroy(
        event,
        [&] {
            was_invoked = true;
            native_status = original();
            has_native_effect = native_status == cudaSuccess;
            return has_native_effect ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        },
        [] { return intercept_context_synchronize(); }, &has_native_effect);
    return was_invoked && native_status != cudaSuccess ? native_status
           : status == CUDA_SUCCESS                    ? native_status
                                                       : cudaErrorUnknown;
}

std::optional<cudaError_t> intercept_normal_runtime_event_completion(
    cudaEvent_t event, bool should_wait, const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    // Original SDK validates this exact handle and retains native last-error /
    // lazy binding. Its readiness alone does not complete a deferred record.
    const auto native = original();
    if (native != cudaSuccess) {
        // Native pending status is already conservative. In particular, do
        // not turn an unknown pending query into a blocking context drain.
        return native;
    }
    bool needs_native_boundary{};
    const auto logical =
        cooperative_event_completion(state, event, should_wait, needs_native_boundary);
    if (logical == CUDA_ERROR_NOT_READY) {
        return cudaErrorNotReady;  // Explicit logical status mapping, not an enum cast.
    }
    if (logical != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    if (needs_native_boundary &&
        state.cooperative_router->native_boundary([] { return CUDA_SUCCESS; },
                                                  [] { return intercept_context_synchronize(); }) !=
            CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    return native;
}

std::optional<cudaError_t> intercept_normal_runtime_event_elapsed_time(
    float* milliseconds, cudaEvent_t start, cudaEvent_t end,
    const std::function<cudaError_t(float*)>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    std::array<CudaEventProjection, 2> owned;
    std::size_t count{};
    CUresult prepared{CUDA_SUCCESS};
    for (CUevent event : {start, end}) {
        CudaEventProjection snapshot;
        const auto found = state.cooperative_router->event_projection(event, snapshot);
        if (found == CUDA_SUCCESS) {
            owned[count++] = std::move(snapshot);
        } else if (found != CUDA_ERROR_INVALID_HANDLE) {
            prepared = found;
        }
    }
    cudaError_t measured{cudaErrorUnknown};
    bool was_invoked{};
    const auto sample = [&] {
        was_invoked = true;
        measured = measure_runtime_event_timing(
            milliseconds, start, end,
            [&](CUevent event) {
                bool needs_native_boundary{};
                return cooperative_event_completion(state, event, false, needs_native_boundary);
            },
            original);
        return CUDA_SUCCESS;
    };
    const auto admitted = prepared == CUDA_SUCCESS ? state.cooperative_router->admit_event_metadata(
                                                         std::span{owned.data(), count}, sample)
                                                   : prepared;
    if (was_invoked) {
        return measured;
    }
    // Even an internal read-only admission failure retains one original SDK
    // validation and its native error precedence, with no sample publication.
    return measure_runtime_event_timing(
        milliseconds, start, end, [admitted](CUevent) { return admitted; }, original);
}

cudaError_t measure_runtime_event_timing(float* milliseconds, cudaEvent_t start, cudaEvent_t end,
                                         const std::function<CUresult(CUevent)>& observe,
                                         const std::function<cudaError_t(float*)>& original) {
    if (!observe || !original) {
        return cudaErrorNotSupported;
    }
    CUresult readiness{CUDA_SUCCESS};
    for (CUevent event : {start, end}) {
        const auto logical = observe(event);
        if (logical != CUDA_SUCCESS &&
            (readiness == CUDA_SUCCESS || readiness == CUDA_ERROR_NOT_READY)) {
            readiness = logical;
        }
    }
    float sample{};
    const auto native = original(milliseconds == nullptr ? nullptr : &sample);
    if (native != cudaSuccess) {
        return native;
    }
    if (readiness == CUDA_ERROR_NOT_READY) {
        return cudaErrorNotReady;
    }
    if (readiness != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    if (milliseconds != nullptr) {
        *milliseconds = sample;
    }
    return native;
}

std::optional<cudaError_t> intercept_normal_runtime_event_record(
    cudaEvent_t event, cudaStream_t stream, bool is_ptds,
    const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    CudaEventProjection projection;
    const auto found = router.event_projection(event, projection);
    if (found != CUDA_SUCCESS || !projection.allows_logical_binding() ||
        (!projection.is_logical() && !projection.has_native_completion() &&
         !projection.issued_native_prefix())) {
        const auto native = original();
        if (native != cudaSuccess) {
            return native;
        }
        return router.native_boundary([] { return CUDA_SUCCESS; },
                                      [] { return intercept_context_synchronize(); }) ==
                       CUDA_SUCCESS
                   ? cudaSuccess
                   : cudaErrorUnknown;
    }
    CUcontext context{};
    std::shared_ptr<CudaCooperativeFrontend> owner;
    const auto bound = router.event_frontend(event, context, owner);
    if (bound != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    const auto accepted = router.admit_event_metadata(std::span{&projection, 1}, [&] {
        return owner->admit_event_metadata(event, [&] {
            CudaEventProjection current;
            auto status = router.refresh_event_projection(projection, current);
            std::shared_ptr<CudaQueueEventRecordGuard> guard;
            if (status == CUDA_SUCCESS) {
                status = router.prepare_event_record_guard(current, guard);
            }
            if (status != CUDA_SUCCESS) {
                return status;  // No original effect has occurred.
            }
            was_invoked = true;
            try {
                native_status = original();
            } catch (...) {
                router.fail_context_owner(context, CUDA_ERROR_UNKNOWN);
                return CUDA_ERROR_UNKNOWN;
            }
            if (native_status != cudaSuccess) {
                return CUDA_SUCCESS;  // SDK failure accepts no logical record.
            }
            RuntimeMetadataEffectGuard effect(router, context);
            CUcontext stream_context{};
            {
                const NativeDriverCallScope native;
                status = is_ptds ? state.driver.stream_get_context_ptsz(stream, &stream_context)
                                 : state.driver.stream_get_context(stream, &stream_context);
            }
            if (status == CUDA_SUCCESS && stream_context != context) {
                status = CUDA_ERROR_INVALID_CONTEXT;
            }
            if (status == CUDA_SUCCESS) {
                status = router.with_context(context, [&] {
                    return owner->record_event(event, stream, is_ptds, std::move(guard));
                });
            }
            if (status == CUDA_SUCCESS) {
                effect.complete();
            }
            return status;
        });
    });
    return was_invoked && native_status != cudaSuccess ? native_status
           : accepted == CUDA_SUCCESS                  ? native_status
                                                       : cudaErrorUnknown;
}

std::optional<cudaError_t> intercept_normal_runtime_event_wait(
    cudaStream_t stream, cudaEvent_t event, bool is_ptds, unsigned int flags,
    const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    const auto complete_native = [&] {
        return router.native_boundary([] { return CUDA_SUCCESS; },
                                      [] { return intercept_context_synchronize(); }) ==
                       CUDA_SUCCESS
                   ? cudaSuccess
                   : cudaErrorUnknown;
    };
    CudaEventProjection projection;
    const auto found = router.event_projection(event, projection);
    if (flags != 0 || found != CUDA_SUCCESS || !projection.allows_logical_binding() ||
        (!projection.is_logical() && !projection.has_native_completion())) {
        const auto native = original();
        return native != cudaSuccess ? native : complete_native();
    }
    CUcontext context{};
    std::shared_ptr<CudaCooperativeFrontend> owner;
    if (router.event_frontend(event, context, owner) != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    bool needs_native_boundary{}, was_invoked{};
    cudaError_t native_status{cudaErrorUnknown};
    const auto accepted = owner->admit_event_metadata(event, [&] {
        CudaEventProjection current;
        auto status = router.refresh_event_projection(projection, current);
        if (status != CUDA_SUCCESS) {
            return status;
        }
        if (!current.is_logical() && !current.has_native_completion()) {
            needs_native_boundary = true;
            return CUDA_SUCCESS;  // Validate outside the contribution, without a logical import.
        }
        CudaFrontendEventWait snapshot;
        status = router.with_context(context,
                                     [&] { return owner->prepare_event_wait(event, snapshot); });
        if (status != CUDA_SUCCESS) {
            return status;
        }
        was_invoked = true;
        try {
            native_status = original();
        } catch (...) {
            router.fail_context_owner(context, CUDA_ERROR_UNKNOWN);
            return CUDA_ERROR_UNKNOWN;
        }
        if (native_status != cudaSuccess) {
            return CUDA_SUCCESS;
        }
        RuntimeMetadataEffectGuard effect(router, context);
        bool is_current{};
        status = router.matches_event_epoch(current, is_current);
        if (status == CUDA_SUCCESS && !is_current) {
            status = CUDA_ERROR_INVALID_HANDLE;
        }
        CUcontext stream_context{};
        if (status == CUDA_SUCCESS) {
            const NativeDriverCallScope native;
            status = is_ptds ? state.driver.stream_get_context_ptsz(stream, &stream_context)
                             : state.driver.stream_get_context(stream, &stream_context);
        }
        if (status == CUDA_SUCCESS && stream_context != context) {
            needs_native_boundary = true;
            effect.complete();
            return CUDA_SUCCESS;  // Cross-queue tokens are not inferred from SDK acceptance.
        }
        if (status == CUDA_SUCCESS) {
            status = router.with_context(
                context, [&] { return owner->wait_event_snapshot(stream, is_ptds, snapshot); });
        }
        if (status == CUDA_SUCCESS) {
            effect.complete();
        }
        return status;
    });
    if (was_invoked && native_status != cudaSuccess) {
        return native_status;
    }
    if (accepted != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    if (needs_native_boundary) {
        if (!was_invoked) {
            native_status = original();
        }
        return native_status != cudaSuccess ? native_status : complete_native();
    }
    return native_status;
}

std::optional<cudaError_t> intercept_normal_runtime_event_create(
    cudaEvent_t* event, unsigned int flags, const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary || (flags & cudaEventInterprocess) != 0U ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope scope;
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    const auto result = state.cooperative_router->normal_runtime_event_create(event, [&] {
        was_invoked = true;
        native_status = original();
        return native_status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    });
    return was_invoked && native_status != cudaSuccess ? native_status
           : result == CUDA_SUCCESS                    ? native_status
                                                       : cudaErrorUnknown;
}

cudaError_t intercept_cooperative_runtime_event_create(
    cudaEvent_t* event, unsigned int flags, const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    auto& gate = cooperative_process_call_gate();
    if (!original) {
        return cudaErrorInvalidValue;
    }
    if (!cooperative_runtime_preview_requested() || !gate.is_native_only_on_current_thread() ||
        is_inside_driver_call() || g_is_inside_cooperative_boundary) {
        return original();
    }
    if ((flags & cudaEventInterprocess) != 0U) {
        return cudaErrorNotSupported;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope scope;
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    bool has_native_effect{};
    const auto status = state.cooperative_router->source_runtime_event_create(
        gate, event,
        [&] {
            was_invoked = true;
            native_status = original();
            has_native_effect = native_status == cudaSuccess;
            return has_native_effect ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        },
        &has_native_effect);
    return was_invoked && native_status != cudaSuccess ? native_status
           : status == CUDA_SUCCESS                    ? native_status
                                                       : cudaErrorUnknown;
}

cudaError_t intercept_cooperative_runtime_event_destroy(
    cudaEvent_t event, const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    auto& gate = cooperative_process_call_gate();
    if (!original) {
        return cudaErrorInvalidValue;
    }
    if (!cooperative_runtime_preview_requested() || !gate.is_native_only_on_current_thread() ||
        is_inside_driver_call() || g_is_inside_cooperative_boundary) {
        return original();
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope scope;
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    bool has_native_effect{};
    const auto status = state.cooperative_router->source_event_destroy(
        gate, event,
        [&] {
            was_invoked = true;
            native_status = original();
            has_native_effect = native_status == cudaSuccess;
            // Effect signal only; never reinterpret Runtime's native error.
            return has_native_effect ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        },
        CudaSourceMemoryOrigin::kRuntime, &has_native_effect);
    return was_invoked && native_status != cudaSuccess ? native_status
           : status == CUDA_SUCCESS                    ? native_status
                                                       : cudaErrorUnknown;
}

cudaError_t intercept_cooperative_runtime_memory_boundary(
    const std::function<cudaError_t()>& operation, const bool& has_native_issue) {
    auto& gate = cooperative_process_call_gate();
    if (!cooperative_runtime_preview_requested() || !gate.is_native_only_on_current_thread() ||
        is_inside_driver_call() || g_is_inside_cooperative_boundary) {
        return operation();
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope scope;
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    const auto status = state.cooperative_router->source_memory_metadata(
        gate,
        [&] {
            was_invoked = true;
            native_status = operation();
            return native_status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        },
        CudaSourceMemoryOrigin::kRuntime, &has_native_issue);
    return was_invoked && native_status != cudaSuccess ? native_status
           : status == CUDA_SUCCESS                    ? native_status
                                                       : cudaErrorUnknown;
}

cudaError_t intercept_cooperative_runtime_native_boundary(
    const std::function<cudaError_t()>& operation) {
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary) {
        return operation();
    }
    InterceptorState& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router || !state.persistent_kernels) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope scope;
    cudaError_t original = cudaErrorUnknown;
    const auto boundary = state.cooperative_router->native_boundary(
        [&] {
            original = operation();
            return CUDA_SUCCESS;
        },
        [] { return intercept_context_synchronize(); });
    return boundary == CUDA_SUCCESS ? original : cudaErrorUnknown;
}

void* resolve_native_driver_function(const char* name, bool per_thread_default_stream,
                                     int cuda_version) {
    InterceptorState& state = get_state();
    if (!ensure_initialized(state) || name == nullptr) {
        return nullptr;
    }
    void* address = nullptr;
    const auto flags = per_thread_default_stream ? CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM
                                                 : CU_GET_PROC_ADDRESS_LEGACY_STREAM;
    if (state.driver.get_proc_address(name, &address, cuda_version, flags) != CUDA_SUCCESS) {
        return nullptr;
    }
    if (is_interceptor_driver_function(address)) {
        report_diagnostic(
            "[glimmer] native Driver lookup returned an interceptor; recursive dispatch "
            "rejected\n");
        return nullptr;
    }
    return address;
}

CUresult intercept_driver_stream_create(const std::function<CUresult()>& original) {
    InterceptorState& state = get_state();
    if (!should_route_cooperative(state) ||
        !cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return intercept_cooperative_native_boundary(original);
    }
    CudaProcessNativeToken token;
    CUcontext context{};
    const auto ready = source_native_current_context(state, token, context);
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    const auto observed = state.cooperative_router->observe_native_session_context(context);
    return observed == CUDA_SUCCESS ? original() : observed;
}

CUresult quarantine_cooperative_source_call(CUresult error) noexcept {
    auto* router = cooperative_source_router();
    return router ? router->quarantine_source_call(cooperative_process_call_gate(), error)
                  : CUDA_ERROR_NOT_SUPPORTED;
}

CUresult intercept_driver_copy(CUdeviceptr destination, CUdeviceptr source, void* host,
                               std::size_t memory_bytes, CUstream stream, unsigned int kind,
                               bool per_thread_default_stream) {
    if (kind > 2) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    static const auto native_copies = [] {
        std::array<void*, 6> addresses{};
        constexpr std::array<const char*, 3> names{"cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync",
                                                   "cuMemcpyDtoDAsync"};
        for (std::size_t index = 0; index < names.size(); ++index) {
            addresses[index * 2] = resolve_native_driver_function(names[index], false);
            addresses[index * 2 + 1] = resolve_native_driver_function(names[index], true);
        }
        return addresses;
    }();
    void* address =
        native_copies[static_cast<std::size_t>(kind) * 2 + (per_thread_default_stream ? 1 : 0)];
    if (address == nullptr) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const auto invoke = [&] {
        const NativeDriverCallScope scope;
        const auto result = kind == 0 ? reinterpret_cast<decltype(&cuMemcpyHtoDAsync)>(address)(
                                            destination, host, memory_bytes, stream)
                            : kind == 1 ? reinterpret_cast<decltype(&cuMemcpyDtoHAsync)>(address)(
                                              host, source, memory_bytes, stream)
                                        : reinterpret_cast<decltype(&cuMemcpyDtoDAsync)>(address)(
                                              destination, source, memory_bytes, stream);
        return result;
    };
    InterceptorState& state = get_state();
    if (!should_route_cooperative(state)) {
        return invoke();
    }
    if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return source_native_stream_execution(state, stream, per_thread_default_stream, invoke);
    }
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    const auto ready = cooperative_frontend_current(state, frontend);
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    bool can_queue = false;
    {
        const NativeDriverCallScope scope;
        CUcontext context{};
        const auto& dispatch = state.cooperative_router->copy_range_dispatch();
        can_queue = state.driver.context_get_current(&context) == CUDA_SUCCESS;
        if (kind != 1) {
            can_queue = can_queue && can_defer_cuda_copy_range(dispatch, context, destination,
                                                               memory_bytes, CU_MEMORYTYPE_DEVICE);
        }
        if (kind != 0) {
            can_queue = can_queue && can_defer_cuda_copy_range(dispatch, context, source,
                                                               memory_bytes, CU_MEMORYTYPE_DEVICE);
        }
        if (kind != 2) {
            can_queue = can_queue && can_defer_cuda_copy_range(dispatch, context,
                                                               reinterpret_cast<CUdeviceptr>(host),
                                                               memory_bytes, CU_MEMORYTYPE_HOST);
        }
    }
    if (!can_queue) {
        // Leave the native reentrancy scope before taking the ordered boundary.
        return intercept_cooperative_native_boundary(invoke);
    }
    return frontend->copy(
        stream, per_thread_default_stream,
        CudaQueuedCopy{static_cast<CudaQueuedCopyKind>(kind), destination, source, host,
                       memory_bytes, state.cooperative_router->resources()});
}

CUresult intercept_driver_uva_copy(CUdeviceptr destination, CUdeviceptr source,
                                   std::size_t memory_bytes, CUstream stream,
                                   bool per_thread_default_stream,
                                   const std::function<CUresult()>& original) {
    InterceptorState& state = get_state();
    if (!should_route_cooperative(state)) {
        return original();
    }
    if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return source_native_stream_execution(state, stream, per_thread_default_stream, original);
    }
    std::shared_ptr<CudaCooperativeFrontend> frontend;
    const auto ready = cooperative_frontend_current(state, frontend);
    if (ready != CUDA_SUCCESS) {
        return ready;
    }
    CUmemorytype destination_type{};
    CUmemorytype source_type{};
    CUcontext current_context = nullptr;
    bool can_queue = memory_bytes != 0 && destination != 0 && source != 0;
    {
        const NativeDriverCallScope scope;
        const auto attribute = state.cooperative_router->pointer_attribute();
        can_queue =
            can_queue && state.driver.context_get_current(&current_context) == CUDA_SUCCESS &&
            attribute(&destination_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, destination) ==
                CUDA_SUCCESS &&
            attribute(&source_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, source) == CUDA_SUCCESS;
        const auto& dispatch = state.cooperative_router->copy_range_dispatch();
        can_queue =
            can_queue &&
            can_defer_cuda_copy_range(dispatch, current_context, destination, memory_bytes,
                                      destination_type) &&
            can_defer_cuda_copy_range(dispatch, current_context, source, memory_bytes, source_type);
    }
    CudaQueuedCopy copy;
    copy.memory_bytes = memory_bytes;
    copy.resources = state.cooperative_router->resources();
    if (can_queue && destination_type == CU_MEMORYTYPE_DEVICE &&
        source_type == CU_MEMORYTYPE_HOST) {
        copy.kind = CudaQueuedCopyKind::kHostToDevice;
        copy.destination = destination;
        // NOLINTNEXTLINE(performance-no-int-to-ptr): required CUDA UVA host address ABI.
        copy.host_pointer = reinterpret_cast<void*>(source);
    } else if (can_queue && destination_type == CU_MEMORYTYPE_HOST &&
               source_type == CU_MEMORYTYPE_DEVICE) {
        copy.kind = CudaQueuedCopyKind::kDeviceToHost;
        copy.source = source;
        // NOLINTNEXTLINE(performance-no-int-to-ptr): required CUDA UVA host address ABI.
        copy.host_pointer = reinterpret_cast<void*>(destination);
    } else if (can_queue && destination_type == CU_MEMORYTYPE_DEVICE &&
               source_type == CU_MEMORYTYPE_DEVICE) {
        copy.kind = CudaQueuedCopyKind::kDeviceToDevice;
        copy.destination = destination;
        copy.source = source;
    } else {
        return intercept_cooperative_native_boundary(original);
    }
    return frontend->copy(stream, per_thread_default_stream, std::move(copy));
}

}  // namespace glimmer::interceptor

namespace glimmer::interceptor {

template <typename Allocate>
cudaError_t intercept_runtime_malloc_impl(void** device_pointer, std::size_t memory_bytes,
                                          Allocate&& allocate, RuntimeFreeFunction release,
                                          RuntimeGetDeviceFunction get_device,
                                          AllocationProvenance provenance) {
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

    const cudaError_t allocation_result = allocate();
    if (allocation_result != cudaSuccess) {
        return allocation_result;
    }

    if (*device_pointer == nullptr) {
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

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context), memory_bytes,
                                  AllocationScope::kContextBound, provenance)) {
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

cudaError_t intercept_runtime_malloc(void** device_pointer, std::size_t memory_bytes,
                                     RuntimeMallocFunction allocate, RuntimeFreeFunction release,
                                     RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_impl(
        device_pointer, memory_bytes,
        [device_pointer, memory_bytes, allocate] {
            mark_runtime_source_memory_issue();
            return allocate(device_pointer, memory_bytes);
        },
        release, get_device, AllocationProvenance::kIndependentDevice);
}

cudaError_t intercept_runtime_malloc_managed(void** device_pointer, std::size_t memory_bytes,
                                             unsigned int flags,
                                             RuntimeMallocManagedFunction allocate,
                                             RuntimeFreeFunction release,
                                             RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_impl(
        device_pointer, memory_bytes,
        [device_pointer, memory_bytes, flags, allocate] {
            mark_runtime_source_memory_issue();
            return allocate(device_pointer, memory_bytes, flags);
        },
        release, get_device, AllocationProvenance::kUnknown);
}

template <typename Allocate>
cudaError_t intercept_runtime_malloc_pitch_impl(void** device_pointer, std::size_t* pitch,
                                                std::size_t width_bytes, std::size_t height,
                                                Allocate&& allocate, RuntimeFreeFunction release,
                                                RuntimeGetDeviceFunction get_device) {
    InterceptorState& state = get_state();
    if (device_pointer == nullptr || pitch == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? allocate(device_pointer, pitch, width_bytes, height)
                   : cudaErrorUnknown;
    }

    if (state.quota_mode == QuotaMode::kDisabled) {
        return allocate(device_pointer, pitch, width_bytes, height);
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return cudaErrorInvalidValue;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    if (state.quota == nullptr || !state.quota->is_healthy()) {
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

    const std::optional<MemoryBytes> requested_bytes =
        checked_multiply(static_cast<MemoryBytes>(width_bytes), static_cast<MemoryBytes>(height));
    if (!requested_bytes.has_value()) {
        return cudaErrorInvalidValue;
    }
    auto reservation = state.quota->try_reserve(device, *requested_bytes);
    if (!reservation.has_value()) {
        return state.quota->is_healthy() ? cudaErrorMemoryAllocation : cudaErrorUnknown;
    }

    const cudaError_t allocation_result = allocate(device_pointer, pitch, width_bytes, height);
    if (allocation_result != cudaSuccess) {
        return allocation_result;
    }
    if (*device_pointer == nullptr) {
        reservation->abandon();
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    const auto cleanup_allocation = [device_pointer, release] {
        return release == nullptr ? cudaErrorUnknown : release(*device_pointer);
    };

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value() || static_cast<DeviceId>(context->device) != device) {
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return cudaErrorUnknown;
    }

    const std::optional<MemoryBytes> actual_bytes =
        checked_multiply(static_cast<MemoryBytes>(*pitch), static_cast<MemoryBytes>(height));
    if (!actual_bytes.has_value()) {
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        return cudaErrorInvalidValue;
    }

    std::optional<glimmer::control::MemoryReservation> adjustment;
    if (*actual_bytes > *requested_bytes) {
        adjustment = state.quota->try_reserve(device, *actual_bytes - *requested_bytes);
        if (!adjustment.has_value()) {
            const cudaError_t cleanup_result = cleanup_allocation();
            if (cleanup_result != cudaSuccess) {
                reservation->abandon();
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            return state.quota->is_healthy() ? cudaErrorMemoryAllocation : cudaErrorUnknown;
        }
    }

    if (!reservation->commit()) {
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            reservation->abandon();
            if (adjustment.has_value()) {
                adjustment->abandon();
            }
        }
        state.allocations.mark_accounting_degraded();
        return cleanup_result != cudaSuccess ? cleanup_result : cudaErrorUnknown;
    }

    MemoryBytes committed_bytes = *requested_bytes;
    if (adjustment.has_value()) {
        if (!adjustment->commit()) {
            const cudaError_t cleanup_result = cleanup_allocation();
            if (cleanup_result != cudaSuccess) {
                state.allocations.mark_accounting_degraded();
                return cleanup_result;
            }
            if (!state.quota->release(device, committed_bytes)) {
                state.allocations.mark_accounting_degraded();
            }
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        committed_bytes = *actual_bytes;
    } else if (*actual_bytes < *requested_bytes) {
        if (!state.quota->release(device, *requested_bytes - *actual_bytes)) {
            const cudaError_t cleanup_result = cleanup_allocation();
            state.allocations.mark_accounting_degraded();
            return cleanup_result == cudaSuccess ? cudaErrorUnknown : cleanup_result;
        }
        committed_bytes = *actual_bytes;
    }

    if (!state.allocations.record(make_allocation_identity(*device_pointer, *context),
                                  *actual_bytes, AllocationScope::kContextBound,
                                  AllocationProvenance::kIndependentDevice)) {
        state.allocations.mark_accounting_degraded();
        const cudaError_t cleanup_result = cleanup_allocation();
        if (cleanup_result != cudaSuccess) {
            state.allocations.mark_accounting_degraded();
            return cleanup_result;
        }
        if (!state.quota->release(device, committed_bytes)) {
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        return cudaErrorMemoryAllocation;
    }
    return cudaSuccess;
}

cudaError_t intercept_runtime_malloc_pitch(void** device_pointer, std::size_t* pitch,
                                           std::size_t width_bytes, std::size_t height,
                                           RuntimeMallocPitchFunction allocate,
                                           RuntimeFreeFunction release,
                                           RuntimeGetDeviceFunction get_device) {
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    return intercept_runtime_malloc_pitch_impl(
        device_pointer, pitch, width_bytes, height,
        [device_pointer, pitch, width_bytes, height, allocate](void**, std::size_t*, std::size_t,
                                                               std::size_t) {
            mark_runtime_source_memory_issue();
            return allocate(device_pointer, pitch, width_bytes, height);
        },
        release, get_device);
}

cudaError_t intercept_runtime_malloc_3d(struct cudaPitchedPtr* pitched_device_pointer,
                                        struct cudaExtent extent, RuntimeMalloc3DFunction allocate,
                                        RuntimeFreeFunction release,
                                        RuntimeGetDeviceFunction get_device) {
    if (pitched_device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    *pitched_device_pointer = {};
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }

    const std::optional<MemoryBytes> row_count = checked_multiply(
        static_cast<MemoryBytes>(extent.height), static_cast<MemoryBytes>(extent.depth));
    if (!row_count.has_value()) {
        return cudaErrorInvalidValue;
    }

    cudaPitchedPtr allocation{};
    void* device_pointer = nullptr;
    std::size_t pitch = 0;
    const cudaError_t result = intercept_runtime_malloc_pitch_impl(
        &device_pointer, &pitch, extent.width, *row_count,
        [&allocation, &device_pointer, &pitch, extent, allocate](void**, std::size_t*, std::size_t,
                                                                 std::size_t) {
            mark_runtime_source_memory_issue();
            const cudaError_t allocation_result = allocate(&allocation, extent);
            if (allocation_result == cudaSuccess) {
                device_pointer = allocation.ptr;
                pitch = allocation.pitch;
            }
            return allocation_result;
        },
        release, get_device);
    if (result == cudaSuccess) {
        *pitched_device_pointer = allocation;
    }
    return result;
}

cudaError_t intercept_runtime_free(void* device_pointer, RuntimeFreeFunction release) {
    InterceptorState& state = get_state();
    if (release == nullptr) {
        return cudaErrorNotSupported;
    }
    const auto invoke = [&] {
        mark_runtime_source_memory_issue();
        return release(device_pointer);
    };
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr ? invoke() : cudaErrorUnknown;
    }

    // A null release owns no allocation, even when the SDK binds its first
    // context here. Keep the original call and its nested memory boundary.
    if (state.quota_mode != QuotaMode::kEnabled || device_pointer == nullptr) {
        return invoke();
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        const cudaError_t free_result = invoke();
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
        return invoke();
    }

    const cudaError_t free_result = invoke();
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

cudaError_t intercept_runtime_ipc_get_mem_handle(cudaIpcMemHandle_t* handle, void* device_pointer,
                                                 RuntimeIpcGetMemHandleFunction get_handle) {
    if (handle == nullptr || device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_handle == nullptr) {
        return cudaErrorNotSupported;
    }
    return get_handle(handle, device_pointer);
}

cudaError_t intercept_runtime_ipc_open_mem_handle(void** device_pointer, cudaIpcMemHandle_t handle,
                                                  unsigned int flags,
                                                  RuntimeIpcOpenMemHandleFunction open_handle) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (open_handle == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return open_handle(device_pointer, handle, flags); });
}

cudaError_t intercept_runtime_ipc_close_mem_handle(void* device_pointer,
                                                   RuntimeIpcCloseMemHandleFunction close_handle) {
    if (device_pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (close_handle == nullptr) {
        return cudaErrorNotSupported;
    }
    return close_handle(device_pointer);
}

cudaError_t intercept_runtime_import_external_memory(
    cudaExternalMemory_t* external_memory, const struct cudaExternalMemoryHandleDesc* handle_desc,
    RuntimeImportExternalMemoryFunction import_memory) {
    if (external_memory == nullptr || handle_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (import_memory == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return import_memory(external_memory, handle_desc); });
}

cudaError_t intercept_runtime_external_memory_get_mapped_buffer(
    void** device_pointer, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryBufferDesc* buffer_desc,
    RuntimeExternalMemoryGetMappedBufferFunction get_buffer) {
    if (device_pointer == nullptr || external_memory == nullptr || buffer_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_buffer == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_buffer(device_pointer, external_memory, buffer_desc); });
}

cudaError_t intercept_runtime_external_memory_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaExternalMemory_t external_memory,
    const struct cudaExternalMemoryMipmappedArrayDesc* mipmap_desc,
    RuntimeExternalMemoryGetMappedMipmappedArrayFunction get_mipmap) {
    if (mipmap == nullptr || external_memory == nullptr || mipmap_desc == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_mipmap == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_mipmap(mipmap, external_memory, mipmap_desc); });
}

cudaError_t intercept_runtime_destroy_external_memory(
    cudaExternalMemory_t external_memory, RuntimeDestroyExternalMemoryFunction destroy_memory) {
    if (external_memory == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (destroy_memory == nullptr) {
        return cudaErrorNotSupported;
    }
    return destroy_memory(external_memory);
}

cudaError_t intercept_runtime_malloc_array(cudaArray_t* array,
                                           const struct cudaChannelFormatDesc* descriptor,
                                           std::size_t width, std::size_t height,
                                           unsigned int flags,
                                           RuntimeMallocArrayFunction allocate) {
    if (array == nullptr || descriptor == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return allocate(array, descriptor, width, height, flags); });
}

cudaError_t intercept_runtime_malloc_3d_array(cudaArray_t* array,
                                              const struct cudaChannelFormatDesc* descriptor,
                                              struct cudaExtent extent, unsigned int flags,
                                              RuntimeMalloc3DArrayFunction allocate) {
    if (array == nullptr || descriptor == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return allocate(array, descriptor, extent, flags); });
}

cudaError_t intercept_runtime_malloc_mipmapped_array(cudaMipmappedArray_t* mipmap,
                                                     const struct cudaChannelFormatDesc* descriptor,
                                                     struct cudaExtent extent,
                                                     unsigned int level_count, unsigned int flags,
                                                     RuntimeMallocMipmappedArrayFunction allocate) {
    if (mipmap == nullptr || descriptor == nullptr || level_count == 0) {
        return cudaErrorInvalidValue;
    }
    if (allocate == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return allocate(mipmap, descriptor, extent, level_count, flags); });
}

cudaError_t intercept_runtime_free_array(cudaArray_t array, RuntimeFreeArrayFunction release) {
    if (array == nullptr) {
        return cudaErrorInvalidValue;
    }
    return release == nullptr ? cudaErrorNotSupported : release(array);
}

cudaError_t intercept_runtime_free_mipmapped_array(cudaMipmappedArray_t mipmap,
                                                   RuntimeFreeMipmappedArrayFunction release) {
    if (mipmap == nullptr) {
        return cudaErrorInvalidValue;
    }
    return release == nullptr ? cudaErrorNotSupported : release(mipmap);
}

cudaError_t intercept_runtime_graph_add_mem_alloc_node(
    cudaGraphNode_t* graph_node, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    std::size_t dependency_count, struct cudaMemAllocNodeParams* parameters,
    RuntimeGraphAddMemAllocNodeFunction add_node) {
    if (graph_node == nullptr || graph == nullptr || parameters == nullptr ||
        (dependency_count != 0 && dependencies == nullptr)) {
        return cudaErrorInvalidValue;
    }
    if (add_node == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    // Graph memory nodes allocate outside the ordinary malloc/free ABI. We
    // reject them until their lifetime can be charged across graph launch,
    // auto-free, update, and process-exit transitions.
    return forward_untracked_runtime_call(state, [&] {
        return add_node(graph_node, graph, dependencies, dependency_count, parameters);
    });
}

cudaError_t intercept_runtime_graph_instantiation(
    cudaGraphExec_t* output, cudaGraph_t graph, bool should_prepare_effects,
    const std::function<cudaError_t(cudaGraphExec_t*)>& original) {
    static_assert(std::is_same_v<cudaGraph_t, CUgraph>);
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    if (!original) {
        return cudaErrorNotSupported;
    }
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary) {
        return original(output);
    }
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread()) {
        return cudaErrorNotSupported;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope boundary;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    CudaNativeGraphReservation reservation;
    if (output) {
        const auto prepared =
            router.prepare_graph(nullptr, graph, reservation, should_prepare_effects);
        if (prepared != CUDA_SUCCESS) {
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
        }
    }
    cudaError_t native_status{cudaErrorUnknown};
    bool was_invoked{};
    const auto owned = router.native_boundary(
        [&] {
            cudaGraphExec_t staged{};
            {
                const NativeDriverCallScope native;
                was_invoked = true;
                try {
                    native_status = original(output ? &staged : nullptr);
                } catch (...) {
                    // A staged handle may already be an actual SDK effect.
                    // Retain it as unprepared; never replay or guess cleanup.
                    native_status = cudaErrorUnknown;
                }
            }
            CUresult published{CUDA_SUCCESS};
            if (output) {
                std::optional<CUcontext> actual_context;
                if (staged) {
                    CUcontext current{};
                    const auto queried = state.driver.context_get_current(&current);
                    if (queried == CUDA_SUCCESS) {
                        actual_context = current;
                    }
                }
                published = router.graphs().publish(
                    reservation, staged,
                    native_status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN,
                    actual_context);
                if (staged || native_status == cudaSuccess) {
                    *output = staged;
                }
            }
            return published;
        },
        [] { return intercept_context_synchronize(); });
    return was_invoked && native_status != cudaSuccess ? native_status
           : owned == CUDA_SUCCESS                     ? native_status
                                                       : cudaErrorUnknown;
}

cudaError_t intercept_runtime_graph_exec_destroy(cudaGraphExec_t executable,
                                                 const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    if (!original) {
        return cudaErrorNotSupported;
    }
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary) {
        return original();
    }
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread()) {
        return cudaErrorNotSupported;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return cudaErrorNotSupported;
    }
    const CooperativeBoundaryScope boundary;
    auto& graphs = state.cooperative_router->graphs();
    graphs.invalidate(executable);
    const NativeDriverCallScope native;
    const auto status = original();
    if (status == cudaSuccess) {
        graphs.forget(executable);
    }
    return status;
}

cudaError_t intercept_runtime_graph_event_update(cudaGraphExec_t executable, cudaGraphNode_t node,
                                                 cudaEvent_t event, bool is_record,
                                                 const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaGraphNode_t, CUgraphNode>);
    static_assert(std::is_same_v<cudaEvent_t, CUevent>);
    if (!original) return cudaErrorNotSupported;
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary)
        return original();
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread()) {
        return cudaErrorNotSupported;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) return cudaErrorNotSupported;
    const CooperativeBoundaryScope boundary;
    auto& router = *state.cooperative_router;
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            router.prepare_graph_event_update(executable, node, event, is_record, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE) {
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
        }
    }
    const NativeDriverCallScope native;
    const auto status = original();
    const auto published =
        replacement ? router.graphs().publish_event_update(
                          replacement, status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN)
                    : CUDA_SUCCESS;
    return status != cudaSuccess || published == CUDA_SUCCESS ? status : cudaErrorUnknown;
}

cudaError_t intercept_runtime_graph_node_update(cudaGraphExec_t executable, cudaGraphNode_t node,
                                                std::optional<CUgraphNodeType> expected_type,
                                                const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaGraphNode_t, CUgraphNode>);
    if (!original) return cudaErrorNotSupported;
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary)
        return original();
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread())
        return cudaErrorNotSupported;
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) return cudaErrorNotSupported;
    const CooperativeBoundaryScope boundary;
    auto& graphs = state.cooperative_router->graphs();
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            graphs.prepare_node_update(executable, node, expected_type, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE)
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
    }
    const NativeDriverCallScope native;
    const auto status = original();
    const auto published =
        replacement ? graphs.publish_event_update(
                          replacement, status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN)
                    : CUDA_SUCCESS;
    return status != cudaSuccess || published == CUDA_SUCCESS ? status : cudaErrorUnknown;
}

cudaError_t intercept_runtime_graph_update(cudaGraphExec_t executable, cudaGraph_t graph,
                                           std::optional<cudaGraphNode_t> child_node,
                                           const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaGraph_t, CUgraph>);
    static_assert(std::is_same_v<cudaGraphNode_t, CUgraphNode>);
    if (!original) return cudaErrorNotSupported;
    if (!cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary)
        return original();
    auto& gate = cooperative_process_call_gate();
    if (!gate.is_entered_by_current_thread() || gate.is_native_only_on_current_thread())
        return cudaErrorNotSupported;
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) return cudaErrorNotSupported;
    const CooperativeBoundaryScope boundary;
    auto& router = *state.cooperative_router;
    CudaNativeGraphReservation replacement;
    if (executable) {
        const auto prepared =
            router.prepare_graph_update(executable, graph, child_node, replacement);
        if (prepared != CUDA_SUCCESS && prepared != CUDA_ERROR_INVALID_HANDLE)
            return prepared == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation
                                                        : cudaErrorUnknown;
    }
    const NativeDriverCallScope native;
    const auto status = original();
    const auto published =
        replacement ? router.graphs().publish_event_update(
                          replacement, status == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN)
                    : CUDA_SUCCESS;
    return status != cudaSuccess || published == CUDA_SUCCESS ? status : cudaErrorUnknown;
}

CUresult intercept_driver_capture(CUstream stream, bool is_ptds, CudaSourceCaptureKind kind,
                                  const std::function<CUresult()>& original) {
    if (!original) return CUDA_ERROR_NOT_SUPPORTED;
    if (!cooperative_driver_preview_requested() || is_inside_driver_call() ||
        is_inside_runtime_call())
        return original();
    auto& gate = cooperative_process_call_gate();
    auto& state = get_state();
    if (!gate.is_entered_by_current_thread() || !ensure_initialized(state) ||
        !state.cooperative_router)
        return CUDA_ERROR_NOT_SUPPORTED;
    const CooperativeBoundaryScope boundary;
    return state.cooperative_router->source_capture_root(gate, kind, [&](auto& dispatch) {
        CUcontext context{};
        CUresult queried{};
        {
            const NativeDriverCallScope native;
            queried = is_ptds ? state.driver.stream_get_context_ptsz(stream, &context)
                              : state.driver.stream_get_context(stream, &context);
        }
        if (queried == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED &&
            kind == CudaSourceCaptureKind::kEnd) {
            context = nullptr;
            queried = dispatch.lookup_origin_context(stream, is_ptds, context);
        }
        if (queried != CUDA_SUCCESS || !context)
            return queried == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : queried;
        return dispatch.invoke(context, stream, is_ptds, kind, original);
    });
}

cudaError_t intercept_runtime_capture(CudaSourceCaptureKind kind,
                                      const std::function<cudaError_t()>& original) noexcept {
    try {
        if (!original) return cudaErrorNotSupported;
        if (is_inside_runtime_call() || is_inside_driver_call()) return original();
        if (!cooperative_runtime_preview_requested())
            return cooperative_driver_preview_requested() ? cudaErrorNotSupported : original();
        auto& gate = cooperative_process_call_gate();
        auto& state = get_state();
        if (!gate.is_entered_by_current_thread() || !ensure_initialized(state) ||
            !state.cooperative_router)
            return cudaErrorNotSupported;
        const CooperativeBoundaryScope boundary;
        RuntimeCaptureObservation observation;
        cudaError_t native_status{cudaErrorUnknown};
        const auto owned =
            state.cooperative_router->source_capture_root(gate, kind, [&](auto& dispatch) {
                return invoke_runtime_capture_original(
                    [&](CUcontext context, CUstream stream, bool is_ptds,
                        CudaSourceCaptureKind actual_kind, const auto& native) {
                        return dispatch.invoke(context, stream, is_ptds, actual_kind, native);
                    },
                    original, observation, native_status,
                    [&](CUstream stream, bool is_ptds, CUcontext& context) {
                        return dispatch.lookup_origin_context(stream, is_ptds, context);
                    });
            });
        if (observation.was_runtime_invoked && native_status != cudaSuccess) return native_status;
        return owned == CUDA_SUCCESS && observation.was_runtime_invoked ? cudaSuccess
                                                                        : cudaErrorUnknown;
    } catch (...) {
        return cudaErrorUnknown;
    }
}

namespace {
CUresult execute_cooperative_graph(InterceptorState& state, CUgraphExec executable, CUstream stream,
                                   bool is_ptds, const std::function<CUresult()>& original) {
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
}  // namespace

cudaError_t execute_owned_runtime_graph(cudaGraphExec_t executable, cudaStream_t stream,
                                        bool is_ptds,
                                        const std::function<cudaError_t()>& original) noexcept {
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    try {
        auto& gate = cooperative_process_call_gate();
        if (!executable || !original) {
            return cudaErrorInvalidValue;
        }
        if (!cooperative_runtime_preview_requested() || !gate.is_entered_by_current_thread() ||
            is_inside_driver_call() || g_is_inside_cooperative_boundary) {
            return cudaErrorNotSupported;
        }
        auto& state = get_state();
        if (!ensure_initialized(state) || !state.cooperative_router) {
            return cudaErrorNotSupported;
        }
        const CooperativeBoundaryScope boundary;
        auto& router = *state.cooperative_router;
        if (!gate.is_native_only_on_current_thread() &&
            router.normal_owner_status() != CUDA_SUCCESS) {
            return cudaErrorUnknown;
        }
        return invoke_runtime_graph_original(
            [&](const auto& issue) {
                return execute_cooperative_graph(state, executable, stream, is_ptds, issue);
            },
            original);
    } catch (...) {
        return cudaErrorUnknown;
    }
}

cudaError_t intercept_runtime_graph_launch(const char* api_name, cudaGraphExec_t graph_exec,
                                           cudaStream_t stream, RuntimeGraphLaunchFunction launch,
                                           bool per_thread_default_stream) {
    if (cooperative_driver_preview_requested()) {
        if (!cooperative_runtime_preview_requested() || !launch) return cudaErrorNotSupported;
        return execute_owned_runtime_graph(graph_exec, stream, per_thread_default_stream,
                                           [&] { return launch(graph_exec, stream); });
    }
    if (graph_exec == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (launch == nullptr) {
        return cudaErrorNotSupported;
    }
    const bool scheduling_enforced = launch_scheduling_is_enforced();
    const bool trace_launch_timings = launch_timing_is_enabled();
    glimmer::control::LaunchGateTiming gate_timing;
    std::optional<glimmer::core::TaskId> launch_task;
    if (scheduling_enforced) {
        const auto capturing =
            runtime_launch_stream_is_capturing(stream, per_thread_default_stream);
        if (!capturing.has_value()) {
            return cudaErrorNotSupported;
        }
        if (capturing.value()) {
            return launch(graph_exec, stream);
        }
        launch_task = acquire_launch_slot(trace_launch_timings ? &gate_timing : nullptr);
        if (!launch_task.has_value()) {
            return cudaErrorNotSupported;
        }
    }

    const auto cuda_launch_started = trace_launch_timings ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
    const cudaError_t launch_result = launch(graph_exec, stream);
    const auto cuda_launch_finished = trace_launch_timings
                                          ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    if (launch_task.has_value() && launch_result != cudaSuccess) {
        fail_launch_slot(launch_task.value());
    }

    bool event_tracking_succeeded = false;
    bool batch_call_succeeded = true;
    std::uint64_t event_tracking_nanoseconds = 0;
    if (launch_task.has_value() && launch_result == cudaSuccess) {
        const auto event_tracking_started = trace_launch_timings
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        event_tracking_succeeded = track_launch_completion(
            launch_task.value(), per_thread_default_stream && stream == nullptr
                                     ? CU_STREAM_PER_THREAD
                                     : reinterpret_cast<CUstream>(stream));
        if (trace_launch_timings) {
            event_tracking_nanoseconds =
                elapsed_nanoseconds(event_tracking_started, std::chrono::steady_clock::now());
        }
        if (event_tracking_succeeded) {
            batch_call_succeeded = finish_launch_batch_call(launch_task.value());
        }
    }
    if (trace_launch_timings) {
        report_launch_timing_observed(
            launch_task.value_or(0), gate_timing,
            elapsed_nanoseconds(cuda_launch_started, cuda_launch_finished),
            event_tracking_nanoseconds, static_cast<std::uint32_t>(launch_result),
            event_tracking_succeeded);
    }
    if (launch_task.has_value() && launch_result == cudaSuccess &&
        (!event_tracking_succeeded || !batch_call_succeeded)) {
        return cudaErrorUnknown;
    }
    if (launch_result == cudaSuccess) {
        report_graph_launch_observed(api_name, reinterpret_cast<const void*>(stream));
    }
    return launch_result;
}

cudaError_t intercept_runtime_graphics_unregister_resource(
    cudaGraphicsResource_t resource, RuntimeGraphicsUnregisterResourceFunction unregister) {
    if (resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    return unregister == nullptr ? cudaErrorNotSupported : unregister(resource);
}

cudaError_t intercept_runtime_graphics_resource_set_map_flags(
    cudaGraphicsResource_t resource, unsigned int flags,
    RuntimeGraphicsResourceSetMapFlagsFunction set_flags) {
    if (resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    return set_flags == nullptr ? cudaErrorNotSupported : set_flags(resource, flags);
}

cudaError_t intercept_runtime_graphics_map_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsMapResourcesFunction map_resources) {
    if (count < 0 || (count != 0 && resources == nullptr)) {
        return cudaErrorInvalidValue;
    }
    if (map_resources == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(state,
                                          [&] { return map_resources(count, resources, stream); });
}

cudaError_t intercept_runtime_graphics_unmap_resources(
    int count, cudaGraphicsResource_t* resources, cudaStream_t stream,
    RuntimeGraphicsUnmapResourcesFunction unmap_resources) {
    if (count < 0 || (count != 0 && resources == nullptr)) {
        return cudaErrorInvalidValue;
    }
    return unmap_resources == nullptr ? cudaErrorNotSupported
                                      : unmap_resources(count, resources, stream);
}

cudaError_t intercept_runtime_graphics_resource_get_mapped_pointer(
    void** device_pointer, std::size_t* size, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedPointerFunction get_pointer) {
    if (device_pointer == nullptr || size == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_pointer == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_pointer(device_pointer, size, resource); });
}

cudaError_t intercept_runtime_graphics_subresource_get_mapped_array(
    cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int array_index,
    unsigned int mip_level, RuntimeGraphicsSubResourceGetMappedArrayFunction get_array) {
    if (array == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_array == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(
        state, [&] { return get_array(array, resource, array_index, mip_level); });
}

cudaError_t intercept_runtime_graphics_resource_get_mapped_mipmapped_array(
    cudaMipmappedArray_t* mipmap, cudaGraphicsResource_t resource,
    RuntimeGraphicsResourceGetMappedMipmappedArrayFunction get_mipmap) {
    if (mipmap == nullptr || resource == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (get_mipmap == nullptr) {
        return cudaErrorNotSupported;
    }
    InterceptorState& state = get_state();
    return forward_untracked_runtime_call(state, [&] { return get_mipmap(mipmap, resource); });
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
    const std::size_t physical_total_bytes = *total_bytes;
    const std::size_t physical_free_bytes = *free_bytes;
    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(device, *total_bytes, *free_bytes);
    if (!state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }
    *total_bytes = static_cast<std::size_t>(memory_info.total_bytes);
    *free_bytes = static_cast<std::size_t>(memory_info.free_bytes);
    report_memory_info_observed("cudaMemGetInfo", device, static_cast<std::uint64_t>(*total_bytes),
                                static_cast<std::uint64_t>(*free_bytes), physical_total_bytes,
                                physical_free_bytes);
    return cudaSuccess;
}

}  // namespace glimmer::interceptor

namespace glimmer::interceptor {

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

CUresult allocate_cooperative_progress(CUdeviceptr* pointer, std::size_t memory_bytes) noexcept {
    if (!pointer || memory_bytes == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer = 0;
    InterceptorState* state = nullptr;
    try {
        state = &get_state();
        return intercept_tracked_allocation(
            pointer, memory_bytes, [&] { return state->driver.mem_alloc(pointer, memory_bytes); },
            true, AllocationProvenance::kIndependentDevice);
    } catch (...) {
        if (state) {
            state->allocations.mark_accounting_degraded();
        }
        return CUDA_ERROR_UNKNOWN;
    }
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
    const auto ordered_release = [&] {
        return intercept_cooperative_runtime_native_boundary(
            [&] { return release(device_pointer, stream); });
    };
    if (!ensure_initialized(state)) {
        return std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") == nullptr
                   ? release(device_pointer, stream)
                   : cudaErrorUnknown;
    }
    if (state.quota_mode == QuotaMode::kDisabled) {
        return ordered_release();
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return ordered_release();
    }
    if (state.allocations.is_accounting_degraded() || state.quota == nullptr ||
        !state.quota->is_healthy()) {
        return cudaErrorUnknown;
    }

    const CUstream driver_stream = reinterpret_cast<CUstream>(stream);
    const std::optional<ContextIdentity> stream_identity =
        capture_stream_identity(state, driver_stream, false);
    if (!stream_identity.has_value()) {
        const cudaError_t free_result = ordered_release();
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
        return ordered_release();
    }
    RuntimeQueuedRelease pending{*release_ticket};
    const ReleaseExceptionGuard exception_guard(state.allocations, pending.ticket,
                                                pending.has_owned_release);
    cudaError_t free_result = cudaErrorUnknown;
    if (cooperative_runtime_preview_requested() &&
        can_queue_cooperative_release(state, *release_ticket)) {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            state.allocations.cancel_release(*release_ticket);
            return cudaErrorUnknown;
        }
        const RuntimeQueuedReleaseScope scope(pending);
        free_result = release(device_pointer, stream);
        if (free_result != cudaSuccess && pending.has_owned_release) {
            // Native validation cannot undo a release already accepted by the queue.
            state.allocations.mark_accounting_degraded();
            return cudaErrorUnknown;
        }
        if (free_result == cudaSuccess && (!pending.was_consumed || !pending.has_owned_release)) {
            state.allocations.mark_accounting_degraded();
            report_diagnostic("[glimmer] Runtime async free bypassed its owned Driver release\n");
            return cudaErrorUnknown;
        }
    } else {
        free_result = ordered_release();
    }
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

std::optional<cudaError_t> intercept_normal_runtime_device_completion(
    const std::function<cudaError_t()>& original,
    const RuntimeContextCompletionExecution& epoch_completion) {
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    auto submitted = state.allocations.submitted_async_releases(true);
    if (!submitted) {
        return cudaErrorUnknown;  // No SDK issue or accounting mutation.
    }
    std::optional<cudaError_t> recovered;
    if (epoch_completion || router.has_native_epoch_policy()) {
        recovered = invoke_runtime_context_completion_original(
            [&](const auto& completion) {
                const auto original_completion = [&] {
                    // Include only submissions joined before the owned SDK boundary.
                    // No future release may be finalized by an earlier proof.
                    submitted = state.allocations.submitted_async_releases(true);
                    return submitted ? completion() : CUDA_ERROR_UNKNOWN;
                };
                return epoch_completion ? epoch_completion(original_completion)
                                        : router.recover_current_native_epoch(
                                              cooperative_process_call_gate(), original_completion);
            },
            original, [&] { router.quarantine_normal_launch(nullptr, CUDA_ERROR_UNKNOWN); });
    }
    const auto native_status = recovered ? *recovered : original();
    if (native_status != cudaSuccess) {
        return native_status;
    }
    CUcontext context{};
    CUresult status{};
    {
        const NativeDriverCallScope native;
        status = state.driver.context_get_current(&context);
    }
    if (status != CUDA_SUCCESS || !context) {
        return cudaErrorUnknown;
    }
    std::shared_ptr<CudaCooperativeFrontend> owner;
    status = router.existing_frontend(context, owner);
    if (status == CUDA_SUCCESS && owner && !recovered) {
        status = router.with_context(context, [&] { return owner->synchronize_context(); });
    }
    if (status != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return native_status;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    const auto summaries =
        state.allocations.complete_submitted_context_releases(*submitted, context);
    if (!summaries) {
        state.allocations.mark_accounting_degraded();
        return cudaErrorUnknown;
    }
    finalize_async_releases(state, *summaries);
    return state.allocations.is_accounting_degraded() ? cudaErrorUnknown : native_status;
}

std::optional<cudaError_t> intercept_normal_runtime_stream_completion(
    cudaStream_t stream, bool is_ptds, bool should_wait,
    const std::function<cudaError_t()>& original) {
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    if (!original || !cooperative_runtime_preview_requested() || is_inside_driver_call() ||
        g_is_inside_cooperative_boundary ||
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return std::nullopt;
    }
    auto& state = get_state();
    if (!state.cooperative_router) {
        return std::nullopt;
    }
    const CooperativeBoundaryScope scope;
    auto& router = *state.cooperative_router;
    if (router.normal_owner_status() != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    const auto submitted = state.allocations.submitted_async_releases();
    if (!submitted) {
        return cudaErrorUnknown;  // No original issue or accounting mutation.
    }
    const auto native_status = original();
    if (native_status != cudaSuccess) {
        return native_status;  // Pending query remains nonblocking, with native TLS.
    }
    CUcontext context{};
    CUresult status{};
    {
        const NativeDriverCallScope native;
        status = is_ptds ? state.driver.stream_get_context_ptsz(stream, &context)
                         : state.driver.stream_get_context(stream, &context);
    }
    if (status != CUDA_SUCCESS || !context) {
        return cudaErrorUnknown;
    }
    std::shared_ptr<CudaCooperativeFrontend> owner;
    status = router.existing_frontend(context, owner);
    bool has_binding{};
    if (status == CUDA_SUCCESS && owner) {
        status = router.with_context(context, [&] {
            return owner->complete_stream_snapshot(stream, is_ptds, should_wait, has_binding);
        });
    }
    if (status == CUDA_ERROR_NOT_READY) {
        return cudaErrorNotReady;
    }
    if (status != CUDA_SUCCESS) {
        return cudaErrorUnknown;
    }
    if (state.quota_mode != QuotaMode::kEnabled) {
        return native_status;
    }
    if (state.allocations.is_accounting_degraded()) {
        return cudaErrorUnknown;
    }
    if (submitted->empty()) {
        return native_status;
    }
    status = router.with_context(context, [&] {
        if (state.allocations.is_accounting_degraded()) {
            return CUDA_ERROR_UNKNOWN;
        }
        const auto identity = capture_stream_identity(state, stream, is_ptds);
        if (!identity || identity->context != context) {
            return CUDA_ERROR_UNKNOWN;
        }
        const auto retire = [&](const AsyncStreamIdentity& proven) {
            const auto summaries =
                state.allocations.complete_submitted_async_releases(*submitted, proven);
            if (!summaries) {
                state.allocations.mark_accounting_degraded();
                return CUDA_ERROR_UNKNOWN;
            }
            finalize_async_releases(state, *summaries);
            return state.allocations.is_accounting_degraded() ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
        };
        auto completed = retire(make_async_stream_identity(stream, *identity, is_ptds));
        if (completed != CUDA_SUCCESS || stream != nullptr || is_ptds) {
            return completed;
        }
        // Runtime's existing null release namespace is caller-scoped even for
        // legacy aliases. Do not let legacy readiness retire pending PTDS frees.
        {
            const NativeDriverCallScope native;
            completed = state.driver.stream_query_ptsz(nullptr);
        }
        if (completed == CUDA_ERROR_NOT_READY) {
            return CUDA_SUCCESS;  // Retain the ambiguous namespace without blocking this query.
        }
        if (completed != CUDA_SUCCESS) {
            return completed;
        }
        if (owner) {
            bool has_thread_binding{};
            completed = owner->complete_stream_snapshot(nullptr, true, false, has_thread_binding);
            if (completed == CUDA_ERROR_NOT_READY) {
                return CUDA_SUCCESS;
            }
            if (completed != CUDA_SUCCESS) {
                return completed;
            }
        }
        return retire(make_runtime_async_stream_identity(stream, *identity));
    });
    return status == CUDA_SUCCESS ? native_status : cudaErrorUnknown;
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

CUresult intercept_mem_map_array_async(CUarrayMapInfo* map_info_list, unsigned int count,
                                       CUstream stream) {
    if (count != 0 && map_info_list == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mem_map_array_async(map_info_list, count, stream);
    }
    if (!ensure_initialized(state) || !state.driver.has_mem_map_array_async()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        // Sparse and deferred-mapping arrays can map tile-pool memory through
        // this API without exposing a stable allocation size at this boundary.
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mem_map_array_async(map_info_list, count, stream);
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

CUresult intercept_ipc_get_mem_handle(CUipcMemHandle* handle, CUdeviceptr device_pointer) {
    if (handle == nullptr || device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_ipc_get_mem_handle(); },
        [&state, handle, device_pointer] {
            return state.driver.ipc_get_mem_handle(handle, device_pointer);
        });
}

CUresult intercept_ipc_open_mem_handle(CUdeviceptr* device_pointer, CUipcMemHandle handle,
                                       unsigned int flags) {
    if (device_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.ipc_open_mem_handle(device_pointer, handle, flags);
    }
    if (!ensure_initialized(state) || !state.driver.has_ipc_open_mem_handle()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        // An IPC handle refers to physical memory owned and accounted by
        // another process. Without a trusted cross-process allocation record,
        // accepting it would let the importing process bypass its quota.
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.ipc_open_mem_handle(device_pointer, handle, flags);
}

CUresult intercept_ipc_close_mem_handle(CUdeviceptr device_pointer) {
    if (device_pointer == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_ipc_close_mem_handle(); },
        [&state, device_pointer] { return state.driver.ipc_close_mem_handle(device_pointer); });
}

CUresult intercept_import_external_memory(CUexternalMemory* external_memory,
                                          const CUDA_EXTERNAL_MEMORY_HANDLE_DESC* handle_desc) {
    if (external_memory == nullptr || handle_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.import_external_memory(external_memory, handle_desc);
    }
    if (!ensure_initialized(state) || !state.driver.has_import_external_memory()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        // External memory is allocated and owned by another API or process.
        // Its size and tenant ownership are not authenticated at this ABI.
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.import_external_memory(external_memory, handle_desc);
}

CUresult intercept_external_memory_get_mapped_buffer(
    CUdeviceptr* device_pointer, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC* buffer_desc) {
    if (device_pointer == nullptr || external_memory == nullptr || buffer_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.external_memory_get_mapped_buffer(device_pointer, external_memory,
                                                              buffer_desc);
    }
    if (!ensure_initialized(state) || !state.driver.has_external_memory_get_mapped_buffer()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.external_memory_get_mapped_buffer(device_pointer, external_memory,
                                                          buffer_desc);
}

CUresult intercept_external_memory_get_mapped_mipmapped_array(
    CUmipmappedArray* mipmap, CUexternalMemory external_memory,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC* mipmap_desc) {
    if (mipmap == nullptr || external_memory == nullptr || mipmap_desc == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.external_memory_get_mapped_mipmapped_array(mipmap, external_memory,
                                                                       mipmap_desc);
    }
    if (!ensure_initialized(state) ||
        !state.driver.has_external_memory_get_mapped_mipmapped_array()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.external_memory_get_mapped_mipmapped_array(mipmap, external_memory,
                                                                   mipmap_desc);
}

CUresult intercept_destroy_external_memory(CUexternalMemory external_memory) {
    if (external_memory == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_destroy_external_memory(); },
        [&state, external_memory] {
            return state.driver.destroy_external_memory(external_memory);
        });
}

CUresult intercept_array_create(CUarray* array, const CUDA_ARRAY_DESCRIPTOR* descriptor) {
    if (array == nullptr || descriptor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.array_create(array, descriptor);
    }
    if (!ensure_initialized(state) || !state.driver.has_array_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.array_create(array, descriptor);
}

CUresult intercept_array_3d_create(CUarray* array, const CUDA_ARRAY3D_DESCRIPTOR* descriptor) {
    if (array == nullptr || descriptor == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.array_3d_create(array, descriptor);
    }
    if (!ensure_initialized(state) || !state.driver.has_array_3d_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.array_3d_create(array, descriptor);
}

CUresult intercept_array_destroy(CUarray array) {
    if (array == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_array_destroy(); },
        [&state, array] { return state.driver.array_destroy(array); });
}

CUresult intercept_mipmapped_array_create(CUmipmappedArray* mipmap,
                                          const CUDA_ARRAY3D_DESCRIPTOR* descriptor,
                                          unsigned int level_count) {
    if (mipmap == nullptr || descriptor == nullptr || level_count == 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.mipmapped_array_create(mipmap, descriptor, level_count);
    }
    if (!ensure_initialized(state) || !state.driver.has_mipmapped_array_create()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.mipmapped_array_create(mipmap, descriptor, level_count);
}

CUresult intercept_mipmapped_array_destroy(CUmipmappedArray mipmap) {
    if (mipmap == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_mipmapped_array_destroy(); },
        [&state, mipmap] { return state.driver.mipmapped_array_destroy(mipmap); });
}

CUresult intercept_graphics_unregister_resource(CUgraphicsResource resource) {
    if (resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_graphics_unregister_resource(); },
        [&state, resource] { return state.driver.graphics_unregister_resource(resource); });
}

CUresult intercept_graphics_subresource_get_mapped_array(CUarray* array,
                                                         CUgraphicsResource resource,
                                                         unsigned int array_index,
                                                         unsigned int mip_level) {
    if (array == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_subresource_get_mapped_array(array, resource, array_index,
                                                                  mip_level);
    }
    if (!ensure_initialized(state) || !state.driver.has_graphics_subresource_get_mapped_array()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_subresource_get_mapped_array(array, resource, array_index,
                                                              mip_level);
}

CUresult intercept_graphics_resource_get_mapped_mipmapped_array(CUmipmappedArray* mipmap,
                                                                CUgraphicsResource resource) {
    if (mipmap == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_resource_get_mapped_mipmapped_array(mipmap, resource);
    }
    if (!ensure_initialized(state) ||
        !state.driver.has_graphics_resource_get_mapped_mipmapped_array()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_resource_get_mapped_mipmapped_array(mipmap, resource);
}

CUresult intercept_graphics_resource_get_mapped_pointer(CUdeviceptr* device_pointer,
                                                        std::size_t* size,
                                                        CUgraphicsResource resource) {
    if (device_pointer == nullptr || size == nullptr || resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_resource_get_mapped_pointer(device_pointer, size, resource);
    }
    if (!ensure_initialized(state) || !state.driver.has_graphics_resource_get_mapped_pointer()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_resource_get_mapped_pointer(device_pointer, size, resource);
}

CUresult intercept_graphics_resource_set_map_flags(CUgraphicsResource resource,
                                                   unsigned int flags) {
    if (resource == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_graphics_resource_set_map_flags(); },
        [&state, resource, flags] {
            return state.driver.graphics_resource_set_map_flags(resource, flags);
        });
}

CUresult intercept_graphics_map_resources(unsigned int count, CUgraphicsResource* resources,
                                          CUstream stream) {
    if (count != 0 && resources == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.graphics_map_resources(count, resources, stream);
    }
    if (!ensure_initialized(state) || !state.driver.has_graphics_map_resources()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (state.quota_mode == QuotaMode::kInvalidConfiguration) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return state.driver.graphics_map_resources(count, resources, stream);
}

CUresult intercept_graphics_unmap_resources(unsigned int count, CUgraphicsResource* resources,
                                            CUstream stream) {
    if (count != 0 && resources == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    return forward_optional_driver_call(
        state, [&state] { return state.driver.has_graphics_unmap_resources(); },
        [&state, count, resources, stream] {
            return state.driver.graphics_unmap_resources(count, resources, stream);
        });
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
        (glimmer::interceptor::is_inside_runtime_call() &&
         !runtime_cooperative_driver_routing_is_enabled())) {
        return per_thread_default_stream ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                         : state.driver.mem_free_async(device_pointer, stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_mem_free_async_ptsz()
                                   : !state.driver.has_mem_free_async())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        const auto ordered_release = [&] {
            return intercept_cooperative_native_boundary([&] {
                return intercept_mem_free_async(device_pointer, stream, per_thread_default_stream);
            });
        };
        if (is_inside_runtime_call()) {
            auto* pending = g_runtime_queued_release;
            if (pending == nullptr) {
                return ordered_release();
            }
            if (pending->was_consumed ||
                pending->ticket.identity.device_pointer != device_pointer ||
                !can_queue_cooperative_release(state, pending->ticket)) {
                return CUDA_ERROR_NOT_SUPPORTED;
            }
            pending->was_consumed = true;
            std::shared_ptr<CudaCooperativeFrontend> frontend;
            const auto ready = cooperative_frontend_current(state, frontend);
            if (ready != CUDA_SUCCESS) {
                return ready;
            }
            const auto admission = queue_cooperative_release(
                state, *frontend, stream, per_thread_default_stream, pending->ticket);
            pending->has_owned_release = admission.has_owned_release;
            return admission.status;
        }
        if (state.quota_mode != QuotaMode::kEnabled) {
            return ordered_release();
        }
        if (state.allocations.is_accounting_degraded()) {
            return CUDA_ERROR_UNKNOWN;
        }
        const auto context = capture_stream_identity(state, stream, per_thread_default_stream);
        if (!context.has_value()) {
            return ordered_release();
        }
        const auto async_stream =
            make_async_stream_identity(stream, *context, per_thread_default_stream);
        auto release_state = state.allocations.begin_async_release(
            make_allocation_identity(device_pointer, *context), async_stream);
        if (release_state.first == AllocationRegistry::ReleaseStatus::kUnknown) {
            release_state = state.allocations.begin_async_release_by_pointer(
                device_pointer, context->device, async_stream);
        }
        const auto& ticket = release_state.second;
        if (!ticket.has_value()) {
            return ordered_release();
        }
        bool has_owned_release = false;
        const ReleaseExceptionGuard exception_guard(state.allocations, *ticket, has_owned_release);
        if (!can_queue_cooperative_release(state, *ticket)) {
            state.allocations.cancel_release(*ticket);
            return ordered_release();
        }
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            state.allocations.cancel_release(*ticket);
            return ready;
        }
        const auto admission =
            queue_cooperative_release(state, *frontend, stream, per_thread_default_stream, *ticket);
        has_owned_release = admission.has_owned_release;
        if (admission.status != CUDA_SUCCESS) {
            if (admission.has_owned_release) {
                state.allocations.mark_accounting_degraded();
            } else {
                state.allocations.cancel_release(*ticket);
            }
            return admission.status;
        }
        if (!admission.has_owned_release || !state.allocations.commit_async_release(*ticket)) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        return CUDA_SUCCESS;
    }
    if (is_inside_runtime_call()) {
        return per_thread_default_stream ? state.driver.mem_free_async_ptsz(device_pointer, stream)
                                         : state.driver.mem_free_async(device_pointer, stream);
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
        (glimmer::interceptor::is_inside_runtime_call() &&
         !runtime_cooperative_driver_routing_is_enabled())) {
        return per_thread_default_stream ? state.driver.stream_query_ptsz(stream)
                                         : state.driver.stream_query(stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_stream_query_ptsz()
                                   : !state.driver.has_stream_query())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (should_route_cooperative(state) &&
        !cooperative_process_call_gate().is_native_only_on_current_thread()) {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            return ready;
        }
        const auto logical = frontend->query_stream(stream, per_thread_default_stream);
        if (logical != CUDA_SUCCESS) {
            return logical;
        }
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
        (glimmer::interceptor::is_inside_runtime_call() &&
         !runtime_cooperative_driver_routing_is_enabled())) {
        return per_thread_default_stream ? state.driver.stream_synchronize_ptsz(stream)
                                         : state.driver.stream_synchronize(stream);
    }
    if (!ensure_initialized(state) ||
        (per_thread_default_stream ? !state.driver.has_stream_synchronize_ptsz()
                                   : !state.driver.has_stream_synchronize())) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    if (should_route_cooperative(state) &&
        !cooperative_process_call_gate().is_native_only_on_current_thread()) {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            return ready;
        }
        const auto logical = frontend->synchronize_stream(stream, per_thread_default_stream);
        if (logical != CUDA_SUCCESS) {
            return logical;
        }
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
        (glimmer::interceptor::is_inside_runtime_call() &&
         !runtime_cooperative_driver_routing_is_enabled())) {
        return state.driver.stream_destroy(stream);
    }
    if (!ensure_initialized(state) || !state.driver.has_stream_destroy()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state) &&
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return state.cooperative_router->source_stream_destroy(
            cooperative_process_call_gate(), stream,
            [&] { return state.driver.stream_destroy(stream); },
            [&](CUcontext context) {
                if (state.quota_mode != QuotaMode::kEnabled) {
                    return CUDA_SUCCESS;
                }
                const AsyncStreamIdentity saved{.stream = stream, .context = context};
                if (state.allocations.is_accounting_degraded() ||
                    !state.allocations.detach_async_releases_for_stream(saved)) {
                    state.allocations.mark_accounting_degraded();
                    return CUDA_ERROR_UNKNOWN;
                }
                return CUDA_SUCCESS;
            });
    }
    if (should_route_cooperative(state)) {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            return ready;
        }
        const auto logical = frontend->close_stream(stream);
        if (logical != CUDA_SUCCESS && logical != CUDA_ERROR_INVALID_HANDLE) {
            return logical;
        }
        return intercept_cooperative_native_boundary(
            [&] { return intercept_stream_destroy(stream); });
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
        (glimmer::interceptor::is_inside_runtime_call() &&
         !runtime_cooperative_driver_routing_is_enabled())) {
        return state.driver.context_synchronize();
    }
    if (!ensure_initialized(state) || !state.driver.has_context_synchronize()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state) &&
        !cooperative_process_call_gate().is_native_only_on_current_thread()) {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            return ready;
        }
        const auto logical = frontend->synchronize_context();
        if (logical != CUDA_SUCCESS) {
            return logical;
        }
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

CUresult intercept_event_create(CUevent* event, unsigned int flags) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            return source_native_event_create(state, event, flags);
        }
        if (event == nullptr || (flags & CU_EVENT_INTERPROCESS) != 0) {
            return intercept_cooperative_native_boundary(
                [&] { return state.driver.event_create(event, flags); });
        }
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            return ready;
        }
        const auto created = state.driver.event_create(event, flags);
        if (created != CUDA_SUCCESS) {
            return created;
        }
        CUcontext context = nullptr;
        const auto current = state.driver.context_get_current(&context);
        return current == CUDA_SUCCESS ? state.cooperative_router->register_event(context, *event)
                                       : current;
    }
    return state.driver.event_create(event, flags);
}

CUresult intercept_event_record(CUevent event, CUstream stream) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        return cooperative_record(state, event, stream, false,
                                  [&] { return state.driver.event_record(event, stream); });
    }
    return state.driver.event_record(event, stream);
}

CUresult intercept_event_query(CUevent event) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state) &&
        !cooperative_process_call_gate().is_native_only_on_current_thread()) {
        bool needs_native_boundary{};
        const auto logical =
            cooperative_event_completion(state, event, false, needs_native_boundary);
        if (needs_native_boundary) {
            return intercept_cooperative_native_boundary(
                [&] { return state.driver.event_query(event); });
        }
        if (logical != CUDA_SUCCESS) {
            return logical;
        }
    }
    return state.driver.event_query(event);
}

CUresult intercept_event_destroy(CUevent event) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        if (!state.cooperative_router) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        auto& gate = cooperative_process_call_gate();
        if (gate.is_native_only_on_current_thread()) {
            return state.cooperative_router->source_event_destroy(
                gate, event, [&] { return state.driver.event_destroy(event); });
        }
        CudaEventProjection metadata;
        if (state.cooperative_router->event_projection(event, metadata) == CUDA_SUCCESS &&
            !metadata.allows_logical_binding()) {
            const auto destroyed = state.driver.event_destroy(event);
            return destroyed == CUDA_SUCCESS
                       ? state.cooperative_router->retire_metadata_event(metadata)
                       : destroyed;
        }
        CUcontext context = nullptr;
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto found = state.cooperative_router->event_frontend(event, context, frontend);
        if (found == CUDA_ERROR_INVALID_HANDLE) {
            return intercept_cooperative_native_boundary(
                [&] { return state.driver.event_destroy(event); });
        }
        if (found != CUDA_SUCCESS) {
            return found;
        }
        CudaEventProjection projection;
        const auto projected = state.cooperative_router->event_projection(event, projection);
        if (projected != CUDA_SUCCESS) {
            return projected;
        }
        if (!projection.is_logical() && !projection.has_native_completion()) {
            const auto completed =
                intercept_cooperative_native_boundary([] { return CUDA_SUCCESS; });
            if (completed != CUDA_SUCCESS) {
                return completed;
            }
        }
        const auto closed = state.cooperative_router->with_context(
            context, [&] { return frontend->close_event(event); });
        if (closed != CUDA_SUCCESS) {
            return closed;
        }
        const auto destroyed = state.driver.event_destroy(event);
        if (destroyed == CUDA_SUCCESS) {
            state.cooperative_router->forget_event(event);
        } else {
            frontend->fail_native_boundary(destroyed);
        }
        return destroyed;
    }
    return state.driver.event_destroy(event);
}

CUresult intercept_event_record_ptsz(CUevent event, CUstream stream) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        return cooperative_record(state, event, stream, true,
                                  [&] { return state.driver.event_record_ptsz(event, stream); });
    }
    return state.driver.event_record_ptsz(event, stream);
}

CUresult intercept_event_record_with_flags(CUevent event, CUstream stream, unsigned int flags) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            const CudaSourceEventOperation descriptor{CudaSourceEventOperation::Kind::kRecord,
                                                      event, flags};
            return source_native_stream_execution(
                state, stream, false,
                [&] { return state.driver.event_record_with_flags(event, stream, flags); },
                &descriptor);
        }
        if (flags == 0) {
            return cooperative_record(state, event, stream, false, [&] {
                return state.driver.event_record_with_flags(event, stream, flags);
            });
        }
        return intercept_cooperative_native_boundary(
            [&] { return state.driver.event_record_with_flags(event, stream, flags); });
    }
    return state.driver.event_record_with_flags(event, stream, flags);
}

CUresult intercept_event_record_with_flags_ptsz(CUevent event, CUstream stream,
                                                unsigned int flags) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            const CudaSourceEventOperation descriptor{CudaSourceEventOperation::Kind::kRecord,
                                                      event, flags};
            return source_native_stream_execution(
                state, stream, true,
                [&] { return state.driver.event_record_with_flags_ptsz(event, stream, flags); },
                &descriptor);
        }
        if (flags == 0) {
            return cooperative_record(state, event, stream, true, [&] {
                return state.driver.event_record_with_flags_ptsz(event, stream, flags);
            });
        }
        return intercept_cooperative_native_boundary(
            [&] { return state.driver.event_record_with_flags_ptsz(event, stream, flags); });
    }
    return state.driver.event_record_with_flags_ptsz(event, stream, flags);
}

CUresult intercept_event_synchronize(CUevent event) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state) &&
        !cooperative_process_call_gate().is_native_only_on_current_thread()) {
        bool needs_native_boundary{};
        const auto logical =
            cooperative_event_completion(state, event, true, needs_native_boundary);
        if (needs_native_boundary) {
            return intercept_cooperative_native_boundary(
                [&] { return state.driver.event_synchronize(event); });
        }
        if (logical != CUDA_SUCCESS) {
            return logical;
        }
    }
    return state.driver.event_synchronize(event);
}

CUresult intercept_event_elapsed_time(float* milliseconds, CUevent start, CUevent end) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state) &&
        !cooperative_process_call_gate().is_native_only_on_current_thread()) {
        for (CUevent event : {start, end}) {
            bool needs_native_boundary{};
            const auto logical =
                cooperative_event_completion(state, event, false, needs_native_boundary);
            if (needs_native_boundary) {
                return intercept_cooperative_native_boundary(
                    [&] { return state.driver.event_elapsed_time(milliseconds, start, end); });
            }
            if (logical != CUDA_SUCCESS) {
                return logical;
            }
        }
    }
    return state.driver.event_elapsed_time(milliseconds, start, end);
}

CUresult intercept_stream_wait_event(CUstream stream, CUevent event, unsigned int flags) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        const auto original = [&] { return state.driver.stream_wait_event(stream, event, flags); };
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            const CudaSourceEventOperation descriptor{CudaSourceEventOperation::Kind::kWait, event,
                                                      flags};
            return source_native_stream_execution(state, stream, false, original, &descriptor);
        }
        return flags == 0 ? cooperative_wait(state, stream, event, false, original)
                          : intercept_cooperative_native_boundary(original);
    }
    return state.driver.stream_wait_event(stream, event, flags);
}

CUresult intercept_stream_wait_event_ptsz(CUstream stream, CUevent event, unsigned int flags) {
    InterceptorState& state = get_state();
    if (!is_inside_driver_call() && !is_inside_runtime_call() && !ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        const auto original = [&] {
            return state.driver.stream_wait_event_ptsz(stream, event, flags);
        };
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            const CudaSourceEventOperation descriptor{CudaSourceEventOperation::Kind::kWait, event,
                                                      flags};
            return source_native_stream_execution(state, stream, true, original, &descriptor);
        }
        return flags == 0 ? cooperative_wait(state, stream, event, true, original)
                          : intercept_cooperative_native_boundary(original);
    }
    return state.driver.stream_wait_event_ptsz(stream, event, flags);
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

void report_kernel_launch_observed(const KernelLaunchObservation& observation) noexcept {
    try {
        InterceptorState& state = get_state();
        ensure_state_initialized(state);
        const bool observe_first_launch =
            state.scheduler_mode == glimmer::core::SchedulerMode::kObserve;
        if (!state.trace_kernel_launches && !observe_first_launch) {
            return;
        }
        const std::uint64_t launch_count =
            next_launch_observation_count(state.launch_observation_count);
        if (state.trace_kernel_launches || !state.launch_observation_reported.exchange(true)) {
            glimmer::interceptor::report_kernel_launch_diagnostic(observation, launch_count);
        }
    } catch (...) {
        glimmer::interceptor::report_diagnostic(
            "[glimmer] kernel launch observation initialization failed\n");
    }
}

void report_graph_launch_observed(const char* api_name, const void* stream) noexcept {
    try {
        InterceptorState& state = get_state();
        ensure_state_initialized(state);
        const bool observe_first_launch =
            state.scheduler_mode == glimmer::core::SchedulerMode::kObserve;
        if (!state.trace_kernel_launches && !observe_first_launch) {
            return;
        }
        const std::uint64_t launch_count =
            next_launch_observation_count(state.launch_observation_count);
        if (state.trace_kernel_launches || !state.launch_observation_reported.exchange(true)) {
            glimmer::interceptor::report_graph_launch_diagnostic(api_name, stream, launch_count);
        }
    } catch (...) {
        glimmer::interceptor::report_diagnostic(
            "[glimmer] graph launch observation initialization failed\n");
    }
}

void report_memory_info_observed(const char* api_name, std::int32_t device,
                                 std::uint64_t total_bytes, std::uint64_t free_bytes,
                                 std::uint64_t physical_total_bytes,
                                 std::uint64_t physical_free_bytes) noexcept {
    try {
        InterceptorState& state = get_state();
        if (!state.trace_memory_info) {
            return;
        }
        glimmer::interceptor::report_memory_info_diagnostic({
            .api_name = api_name,
            .device = device,
            .total_bytes = total_bytes,
            .free_bytes = free_bytes,
            .physical_total_bytes = physical_total_bytes,
            .physical_free_bytes = physical_free_bytes,
        });
    } catch (...) {
        glimmer::interceptor::report_diagnostic("[glimmer] memory info observation failed\n");
    }
}

bool launch_timing_is_enabled() noexcept {
    try {
        InterceptorState& state = get_state();
        return state.trace_launch_timings;
    } catch (...) {
        return false;
    }
}

void report_launch_timing_observed(glimmer::core::TaskId task_id,
                                   const glimmer::control::LaunchGateTiming& timing,
                                   std::uint64_t cuda_launch_nanoseconds,
                                   std::uint64_t event_tracking_nanoseconds,
                                   std::uint32_t cuda_launch_status,
                                   bool event_tracking_succeeded) noexcept {
    try {
        InterceptorState& state = get_state();
        if (!state.trace_launch_timings) {
            return;
        }
        glimmer::interceptor::report_launch_timing_diagnostic({
            .task_id = task_id,
            .remote = timing.remote,
            .lease_reused = timing.lease_reused,
            .acquire_nanoseconds = timing.elapsed_nanoseconds,
            .acquire_transport_nanoseconds = timing.transport_nanoseconds,
            .acquire_request_count = timing.request_count,
            .claim_poll_count = timing.claim_poll_count,
            .wait_request_count = timing.wait_request_count,
            .cuda_launch_nanoseconds = cuda_launch_nanoseconds,
            .event_tracking_nanoseconds = event_tracking_nanoseconds,
            .cuda_launch_status = cuda_launch_status,
            .event_tracking_succeeded = event_tracking_succeeded,
        });
    } catch (...) {
        glimmer::interceptor::report_diagnostic("[glimmer] launch timing observation failed\n");
    }
}

bool launch_scheduling_is_enforced() noexcept {
    try {
        InterceptorState& state = get_state();
        ensure_state_initialized(state);
        return state.scheduler_mode == glimmer::core::SchedulerMode::kEnforce &&
               !state.is_cooperative_requested;
    } catch (...) {
        return false;
    }
}

std::optional<bool> launch_stream_is_capturing(CUstream stream,
                                               bool per_thread_default_stream) noexcept {
    try {
        InterceptorState& state = get_state();
        if (!ensure_initialized(state)) {
            return std::nullopt;
        }
        CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;
        if (state.driver.stream_capture_status(stream, &status, per_thread_default_stream) !=
            CUDA_SUCCESS) {
            report_diagnostic("[glimmer] launch admission rejected: stream capture query failed\n");
            return std::nullopt;
        }
        switch (status) {
            case CU_STREAM_CAPTURE_STATUS_NONE:
                return false;
            case CU_STREAM_CAPTURE_STATUS_ACTIVE:
            case CU_STREAM_CAPTURE_STATUS_INVALIDATED:
                return true;
            default:
                return std::nullopt;
        }
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<glimmer::core::TaskId> acquire_launch_slot(
    glimmer::control::LaunchGateTiming* timing) noexcept {
    try {
        InterceptorState& state = get_state();
        ensure_state_initialized(state);
        if (state.scheduler_mode != glimmer::core::SchedulerMode::kEnforce ||
            !state.launch_scheduler_ready || state.launch_gate == nullptr) {
            return std::nullopt;
        }
        if (state.launch_batch_size > 1) {
            std::scoped_lock lock(state.launch_batch_mutex);
            for (auto& [task_id, batch] : state.launch_batches) {
                if (batch.remaining_launches == 0 || batch.close_requested) {
                    continue;
                }
                --batch.remaining_launches;
                ++batch.active_calls;
                g_thread_launch_batch = {.task_id = task_id,
                                         .final_launch = batch.remaining_launches == 0};
                if (timing != nullptr) {
                    *timing = {};
                    timing->remote = state.launch_gate->remote_mode();
                    timing->lease_reused = true;
                }
                return task_id;
            }
        }
        const auto task_id = state.launch_gate->acquire(std::chrono::milliseconds::zero(), timing);
        if (task_id.has_value()) {
            bool batch_inserted = false;
            try {
                std::scoped_lock lock(state.launch_batch_mutex);
                batch_inserted =
                    state.launch_batches
                        .try_emplace(task_id.value(),
                                     InterceptorState::LaunchBatchState{
                                         .remaining_launches = state.launch_batch_size - 1,
                                         .active_calls = 1,
                                         .close_requested = false})
                        .second;
                if (batch_inserted) {
                    g_thread_launch_batch = {.task_id = task_id.value(),
                                             .final_launch = state.launch_batch_size == 1};
                }
            } catch (...) {
                static_cast<void>(state.launch_gate->fail(task_id.value()));
                return std::nullopt;
            }
            if (!batch_inserted) {
                // A scheduler task ID must be unique while its launch batch is
                // active. Treat a violation as a failed admission instead of
                // corrupting the existing batch's active-call count.
                static_cast<void>(state.launch_gate->fail(task_id.value()));
                return std::nullopt;
            }
        }
        return task_id;
    } catch (...) {
        return std::nullopt;
    }
}

void fail_launch_slot(glimmer::core::TaskId task_id) noexcept {
    try {
        InterceptorState& state = get_state();
        {
            std::scoped_lock lock(state.launch_batch_mutex);
            state.launch_batches.erase(task_id);
        }
        if (g_thread_launch_batch.task_id == task_id) {
            g_thread_launch_batch = {};
        }
        if (state.launch_gate != nullptr) {
            if (state.launch_tracker != nullptr) {
                static_cast<void>(state.launch_tracker->fail(task_id));
            } else {
                static_cast<void>(state.launch_gate->fail(task_id));
            }
        }
    } catch (...) {
        glimmer::interceptor::report_diagnostic(
            "[glimmer] launch scheduler failure transition failed\n");
    }
}

bool track_launch_completion(glimmer::core::TaskId task_id, CUstream stream) noexcept {
    try {
        InterceptorState& state = get_state();
        const bool tracked =
            state.launch_tracker != nullptr && state.launch_tracker->track(task_id, stream);
        if (!tracked && g_thread_launch_batch.task_id == task_id) {
            {
                std::scoped_lock lock(state.launch_batch_mutex);
                state.launch_batches.erase(task_id);
            }
            g_thread_launch_batch = {};
        }
        return tracked;
    } catch (...) {
        fail_launch_slot(task_id);
        return false;
    }
}

bool finish_launch_batch_call(glimmer::core::TaskId task_id) noexcept {
    try {
        InterceptorState& state = get_state();
        if (g_thread_launch_batch.task_id != task_id) {
            return false;
        }
        bool should_close = false;
        {
            std::scoped_lock lock(state.launch_batch_mutex);
            const auto batch_iterator = state.launch_batches.find(task_id);
            if (batch_iterator == state.launch_batches.end() ||
                batch_iterator->second.active_calls == 0) {
                g_thread_launch_batch = {};
                return false;
            }
            --batch_iterator->second.active_calls;
            if (g_thread_launch_batch.final_launch) {
                batch_iterator->second.close_requested = true;
            }
            if (batch_iterator->second.close_requested &&
                batch_iterator->second.active_calls == 0) {
                state.launch_batches.erase(batch_iterator);
                should_close = true;
            }
        }
        g_thread_launch_batch = {};
        if (!should_close) {
            return true;
        }
        return state.launch_tracker != nullptr && state.launch_tracker->close_batch(task_id);
    } catch (...) {
        return false;
    }
}

namespace {

template <typename Available, typename Invoke>
CUresult intercept_driver_kernel_launch(InterceptorState& state,
                                        const KernelLaunchObservation& observation, CUstream stream,
                                        bool per_thread_default_stream, Available&& available,
                                        Invoke&& invoke) {
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return invoke();
    }

    if (!ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!available()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const bool trace_launch_timings = state.trace_launch_timings;
    glimmer::control::LaunchGateTiming gate_timing;
    std::optional<glimmer::core::TaskId> launch_task;
    if (launch_scheduling_is_enforced()) {
        const auto capturing = launch_stream_is_capturing(stream, per_thread_default_stream);
        if (!capturing.has_value()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        if (capturing.value()) {
            return invoke();
        }
        launch_task = acquire_launch_slot(trace_launch_timings ? &gate_timing : nullptr);
        if (!launch_task.has_value()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    }

    const auto cuda_launch_started = trace_launch_timings ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
    const CUresult launch_result = invoke();
    const auto cuda_launch_finished = trace_launch_timings
                                          ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    if (launch_task.has_value() && launch_result != CUDA_SUCCESS) {
        fail_launch_slot(launch_task.value());
    }
    bool event_tracking_succeeded = false;
    bool batch_call_succeeded = true;
    std::uint64_t event_tracking_nanoseconds = 0;
    if (launch_task.has_value() && launch_result == CUDA_SUCCESS) {
        const auto event_tracking_started = trace_launch_timings
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        event_tracking_succeeded = track_launch_completion(
            launch_task.value(),
            per_thread_default_stream && stream == nullptr ? CU_STREAM_PER_THREAD : stream);
        if (trace_launch_timings) {
            event_tracking_nanoseconds =
                elapsed_nanoseconds(event_tracking_started, std::chrono::steady_clock::now());
        }
        if (event_tracking_succeeded) {
            batch_call_succeeded = finish_launch_batch_call(launch_task.value());
        }
    }
    if (trace_launch_timings) {
        report_launch_timing_observed(
            launch_task.value_or(0), gate_timing,
            elapsed_nanoseconds(cuda_launch_started, cuda_launch_finished),
            event_tracking_nanoseconds, static_cast<std::uint32_t>(launch_result),
            event_tracking_succeeded);
    }
    if (launch_task.has_value() && launch_result == CUDA_SUCCESS &&
        (!event_tracking_succeeded || !batch_call_succeeded)) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (launch_result == CUDA_SUCCESS) {
        report_kernel_launch_observed(observation);
    }
    return launch_result;
}

}  // namespace

CUresult intercept_module_load_data(CUmodule* module, const void* image, bool uses_extended_loader,
                                    unsigned int option_count, CUjit_option* options,
                                    void** option_values) {
    InterceptorState& state = get_state();
    const auto invoke = [&] {
        return uses_extended_loader ? state.driver.module_load_data_ex(module, image, option_count,
                                                                       options, option_values)
                                    : state.driver.module_load_data(module, image);
    };
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return invoke();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state) &&
        cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return intercept_cooperative_module_boundary([&] {
            return state.persistent_kernels ? state.persistent_kernels->load_source_module(
                                                  module, image, uses_extended_loader, option_count,
                                                  options, option_values)
                                            : invoke();
        });
    }
    const auto load = [&] {
        return state.persistent_kernels == nullptr
                   ? invoke()
                   : state.persistent_kernels->load_module(module, image, uses_extended_loader,
                                                           option_count, options, option_values);
    };
    if (should_route_cooperative(state)) {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto result = cooperative_frontend_current(state, frontend);
        if (result != CUDA_SUCCESS) {
            return result;
        }
        return intercept_cooperative_native_boundary(load);
    }
    return load();
}

std::optional<RuntimeCooperativeLaunch> prepare_runtime_cooperative_launch(
    const void* function, dim3 grid, dim3 block, std::size_t shared_memory_bytes,
    RuntimeFuncGetNameFunction get_name, RuntimeGetFuncBySymbolFunction get_function) {
    if (function == nullptr || get_name == nullptr || get_function == nullptr) {
        return std::nullopt;
    }
    InterceptorState& state = get_state();
    if (!ensure_initialized(state) || !state.cooperative_router || !state.persistent_kernels) {
        return std::nullopt;
    }
    CUcontext context = nullptr;
    if (state.driver.context_get_current(&context) != CUDA_SUCCESS || context == nullptr) {
        return std::nullopt;
    }
    const auto image_path = loaded_image_path(function);
    if (!image_path) {
        return std::nullopt;
    }
    std::error_code path_error;
    const auto path = std::filesystem::absolute(*image_path, path_error);
    if (path_error) {
        return std::nullopt;
    }
    const auto binary_path = path.string();
    std::optional<RuntimeCooperativeLaunch> prepared;
    const auto prepare = [&]() -> cudaError_t {
        const char* name = nullptr;
        cudaFunction_t original = nullptr;
        if (get_name(&name, function) != cudaSuccess || name == nullptr ||
            get_function(&original, function) != cudaSuccess || original == nullptr) {
            return cudaSuccess;
        }
        // The selected Toolkit declares both types as CUfunc_st*. No opaque
        // host-function or Runtime stream/event handle is cast to a Driver ABI.
        RuntimeCooperativeLaunch plan{
            original, std::nullopt, {grid.x, grid.y, grid.z}, {block.x, block.y, block.z}};
        plan.context = context;
        plan.shared_memory_bytes = shared_memory_bytes;
        if (!state.persistent_kernels->module_source_is_prepared(original)) {
            const auto ready = intercept_cooperative_runtime_native_boundary([&] {
                state.persistent_kernels->prepare_module_source(original);
                return cudaSuccess;
            });
            if (ready != cudaSuccess) return ready;
        }
        plan.binary_variant = state.persistent_kernels->mapped_binary_kernel(original);
        plan.gemm_variant = state.persistent_kernels->mapped_gemm_kernel(original);
        const auto variant = state.persistent_kernels->runtime_kernel(
            context, function, name, binary_path, GLIMMER_CUOBJDUMP_PATH, original);
        const auto shape = state.persistent_kernels->validated_launch_shape(
            plan.grid, plan.block, PtxExecutionKind::kPreemptive);
        int resident_blocks = 0;
        CUcontext actual_context = nullptr;
        if (shared_memory_bytes == 0 && variant && variant->function != nullptr && shape &&
            variant->execution_kind == PtxExecutionKind::kPreemptive &&
            state.driver.context_get_current(&actual_context) == CUDA_SUCCESS &&
            actual_context == context &&
            state.driver.occupancy(&resident_blocks, variant->function,
                                   static_cast<int>(shape->block_threads), 0) == CUDA_SUCCESS &&
            resident_blocks > 0) {
            plan.variant = CooperativeKernelVariant{
                context, variant->function, variant->parameter_layout, *shape,
                std::min(state.persistent_worker_blocks, shape->logical_blocks)};
        }
        prepared = std::move(plan);
        return cudaSuccess;
    };
    if (state.persistent_kernels->runtime_cooperative_source_is_prepared(context, binary_path)) {
        if (prepare() != cudaSuccess) return std::nullopt;
    } else {
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        if (cooperative_frontend_current(state, frontend) != CUDA_SUCCESS ||
            intercept_cooperative_runtime_native_boundary(prepare) != cudaSuccess) {
            return std::nullopt;
        }
    }
    return prepared;
}

void quarantine_runtime_cooperative_launch(CUcontext context) noexcept {
    auto& state = get_state();
    if (state.cooperative_router)
        state.cooperative_router->quarantine_normal_launch(context, CUDA_ERROR_UNKNOWN);
}

std::optional<RuntimePersistentLaunch> prepare_runtime_persistent_launch(
    const void* function, dim3 grid, dim3 block, std::size_t shared_memory_bytes,
    cudaStream_t stream, bool per_thread_default_stream, RuntimeFuncGetNameFunction get_name,
    RuntimeGetFuncBySymbolFunction get_function) {
    try {
        InterceptorState& state = get_state();
        if (!ensure_initialized(state) || !state.is_runtime_persistent_enabled ||
            state.persistent_kernels == nullptr || function == nullptr || get_name == nullptr ||
            get_function == nullptr || grid.x == 0 || grid.y == 0 || grid.z == 0 || block.x == 0 ||
            block.y == 0 || block.z == 0 || shared_memory_bytes != 0 ||
            !state.driver.has_runtime_persistent_api()) {
            return std::nullopt;
        }
        CUcontext context = nullptr;
        CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
        if (state.driver.context_get_current(&context) != CUDA_SUCCESS || context == nullptr ||
            state.driver.stream_capture_status(reinterpret_cast<CUstream>(stream), &capture_status,
                                               per_thread_default_stream) != CUDA_SUCCESS ||
            capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
            return std::nullopt;
        }
        const char* name = nullptr;
        if (get_name(&name, function) != cudaSuccess || name == nullptr) {
            return std::nullopt;
        }
        const auto image_path = loaded_image_path(function);
        if (!image_path) {
            return std::nullopt;
        }
        std::error_code path_error;
        const auto binary_path = std::filesystem::absolute(*image_path, path_error);
        if (path_error) {
            return std::nullopt;
        }
        const auto variant = state.persistent_kernels->runtime_kernel(
            context, function, name, binary_path.string(), GLIMMER_CUOBJDUMP_PATH);
        if (!variant.has_value() || variant->execution_kind == PtxExecutionKind::kPreemptive) {
            return std::nullopt;
        }
        const auto shape = state.persistent_kernels->validated_launch_shape(
            {grid.x, grid.y, grid.z}, {block.x, block.y, block.z}, variant->execution_kind);
        if (!shape.has_value()) {
            return std::nullopt;
        }
        CUfunction device_function = nullptr;
        int resident_blocks = 0;
        if (state.driver.kernel_get_function(&device_function, variant->kernel) != CUDA_SUCCESS ||
            state.driver.occupancy(&resident_blocks, device_function,
                                   static_cast<int>(shape->block_threads), 0) != CUDA_SUCCESS ||
            resident_blocks <= 0) {
            return std::nullopt;
        }
        cudaFunction_t original_function = nullptr;
        if (get_function(&original_function, function) != cudaSuccess ||
            original_function == nullptr) {
            return std::nullopt;
        }
        CUcontext launch_context = nullptr;
        if (state.driver.context_get_current(&launch_context) != CUDA_SUCCESS ||
            launch_context != context) {
            return std::nullopt;
        }
        return RuntimePersistentLaunch{reinterpret_cast<CUfunction>(original_function),
                                       device_function,
                                       state.persistent_worker_blocks,
                                       variant->parameter_count,
                                       variant->execution_kind,
                                       shape->logical_blocks};
    } catch (...) {
        // No launch has happened; metadata/tool failures retain the original Runtime path.
        return std::nullopt;
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): trace schema names distinct block counts.
void report_runtime_persistent_launch(unsigned int logical_blocks,
                                      unsigned int worker_blocks) noexcept {
    try {
        if (!get_state().trace_kernel_launches) {
            return;
        }
        std::array<char, 160> message{};
        const auto written =
            std::snprintf(message.data(), message.size(),
                          "[glimmer] persistent_runtime logical_blocks=%u worker_blocks=%u\n",
                          logical_blocks, worker_blocks);
        if (written > 0 && static_cast<std::size_t>(written) < message.size()) {
            report_diagnostic(message.data());
        }
    } catch (...) {
    }
}

cudaError_t intercept_runtime_device_reset(RuntimeDeviceResetFunction reset) {
    if (reset == nullptr) {
        return cudaErrorNotSupported;
    }
    if (cooperative_runtime_preview_requested() &&
        (is_inside_driver_call() || g_is_inside_cooperative_boundary)) {
        return reset();
    }
    InterceptorState& state = get_state();
    CUcontext context = nullptr;
    const bool is_preview = cooperative_runtime_preview_requested();
    if (is_preview && !is_inside_driver_call() && !g_is_inside_cooperative_boundary) {
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            return cudaErrorNotSupported;
        }
        if (!ensure_initialized(state) || !state.cooperative_router || !state.persistent_kernels ||
            state.allocations.is_accounting_degraded() ||
            state.cooperative_router->normal_owner_status() != CUDA_SUCCESS) {
            return cudaErrorUnknown;
        }
        const auto current_status = state.driver.context_get_current(&context);
        if (current_status != CUDA_SUCCESS && current_status != CUDA_ERROR_NOT_INITIALIZED) {
            return cudaErrorUnknown;
        }
        if (current_status == CUDA_ERROR_NOT_INITIALIZED) {
            context = nullptr;  // Failed Driver output cannot identify a reset owner.
        }
        const CooperativeBoundaryScope scope;
        bool is_primary_context = false;
        if (!context) {
            // The qualified SDK leaves an unbound caller's active primary and
            // resources alive. Success is not destruction proof. Cold Runtime
            // initialization and last-error behavior remain owned by the SDK.
            return reset();
        } else if (state.quota_mode == QuotaMode::kEnabled) {
            if (state.driver.current_context_is_active_primary(context, is_primary_context) !=
                CUDA_SUCCESS) {
                // Uncertain identity/reference ownership cannot authorize reset
                // or a later retry that could release unproven allocation quota.
                state.allocations.mark_accounting_degraded();
                return cudaErrorUnknown;
            }
        }
        cudaError_t native_result = cudaErrorUnknown;
        bool was_called = false;
        const auto closed = state.cooperative_router->close_context(context, [&] {
            // Private retirement may require the reset target; actual SDK reset
            // must still run on the original bound caller.
            const auto retired = state.cooperative_router->with_context(context, [&] {
                state.persistent_kernels->forget_context(context);
                return CUDA_SUCCESS;
            });
            if (retired != CUDA_SUCCESS) {
                return retired;
            }
            was_called = true;
            native_result = reset();
            return native_result == cudaSuccess ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
        });
        if (!was_called) {
            return cudaErrorUnknown;
        }
        if (native_result != cudaSuccess) {
            return native_result;
        }
        if (closed != CUDA_SUCCESS) {
            return cudaErrorUnknown;
        }
        if (state.quota_mode == QuotaMode::kEnabled && is_primary_context) {
            const auto summaries = state.allocations.erase_context_by_device(context);
            if (!summaries) {
                state.allocations.mark_accounting_degraded();
                return cudaErrorUnknown;
            }
            finalize_async_releases(state, *summaries);
            if (state.allocations.is_accounting_degraded()) {
                return cudaErrorUnknown;
            }
        }
        return native_result;
    }
    const bool has_context = ensure_initialized(state) && state.persistent_kernels != nullptr &&
                             state.driver.context_get_current(&context) == CUDA_SUCCESS;
    const auto result = reset();
    if (result == cudaSuccess && has_context && context != nullptr) {
        state.persistent_kernels->forget_context(context);
    }
    return result;
}

CUresult intercept_library_load_data(CUlibrary* library, const void* image, CUjit_option* options,
                                     void** option_values, unsigned int option_count,
                                     CUlibraryOption* library_options, void** library_option_values,
                                     unsigned int library_option_count) {
    InterceptorState& state = get_state();
    const auto original = [&] {
        return state.driver.library_load_data(library, image, options, option_values, option_count,
                                              library_options, library_option_values,
                                              library_option_count);
    };
    if (is_inside_driver_call()) return original();
    ensure_state_initialized(state);
    return intercept_cooperative_module_boundary([&] {
        const auto result = original();
        const int saved_errno = errno;
        if (result == CUDA_SUCCESS && library && state.persistent_kernels) {
            static_assert(sizeof(CUlibraryOption) == sizeof(std::uint32_t));
            static_assert(CU_LIBRARY_BINARY_IS_PRESERVED == 1);
            state.persistent_kernels->observe_library_source(
                *library, qualified_library_source_options(option_count, library_options,
                                                           library_option_count)
                              ? image
                              : nullptr);
        }
        errno = saved_errno;
        return result;
    });
}

CUresult intercept_library_load_from_file(CUlibrary* library, const char* path,
                                          CUjit_option* options, void** option_values,
                                          unsigned int option_count,
                                          CUlibraryOption* library_options,
                                          void** library_option_values,
                                          unsigned int library_option_count) {
    InterceptorState& state = get_state();
    const auto original = [&] {
        return state.driver.library_load_from_file(library, path, options, option_values,
                                                   option_count, library_options,
                                                   library_option_values, library_option_count);
    };
    if (is_inside_driver_call()) return original();
    ensure_state_initialized(state);
    return intercept_cooperative_module_boundary([&] {
        const auto result = original();
        const int saved_errno = errno;
        if (result == CUDA_SUCCESS && library && state.persistent_kernels)
            state.persistent_kernels->retire_library_source(*library);
        errno = saved_errno;
        return result;
    });
}

CUresult intercept_library_get_module(CUmodule* module, CUlibrary library) {
    InterceptorState& state = get_state();
    if (is_inside_driver_call()) return state.driver.library_get_module(module, library);
    ensure_state_initialized(state);
    const auto result = state.driver.library_get_module(module, library);
    const int saved_errno = errno;
    if (result == CUDA_SUCCESS && module && state.persistent_kernels) {
        CUcontext context{};
        if (state.driver.context_get_current(&context) != CUDA_SUCCESS) context = nullptr;
        state.persistent_kernels->observe_library_module(library, *module, context);
    }
    errno = saved_errno;
    return result;
}

CUresult intercept_library_unload(CUlibrary library) {
    InterceptorState& state = get_state();
    if (is_inside_driver_call()) return state.driver.library_unload(library);
    ensure_state_initialized(state);
    const auto original = [&] {
        const auto result = state.driver.library_unload(library);
        const int saved_errno = errno;
        // An attempted unload is not permission to reuse uncertain associations.
        if (state.persistent_kernels) state.persistent_kernels->retire_library_source(library);
        errno = saved_errno;
        return result;
    };
    if (cooperative_process_call_gate().is_native_only_on_current_thread())
        return intercept_cooperative_module_boundary(original);
    if (!state.is_cooperative_requested || g_is_inside_cooperative_boundary) return original();
    if (!state.cooperative_router || !state.persistent_kernels) return CUDA_ERROR_NOT_SUPPORTED;
    // Libraries span contexts. This Driver boundary must also drain in a
    // Driver-only preview and when invoked by a Runtime-native SDK frame.
    const CooperativeBoundaryScope scope;
    return state.cooperative_router->native_boundary(
        original, [] { return intercept_context_synchronize(); });
}

CUresult intercept_module_get_function(CUfunction* function, CUmodule module, const char* name) {
    InterceptorState& state = get_state();
    if (is_inside_driver_call()) {
        return state.driver.module_get_function(function, module, name);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (is_inside_runtime_call()) {
        // The SDK's original lookup may run during Runtime-native preparation.
        // Retain identity only; private code loading requires a later drained boundary.
        return state.persistent_kernels
                   ? state.persistent_kernels->get_source_function(function, module, name)
                   : state.driver.module_get_function(function, module, name);
    }
    if (should_route_cooperative(state) &&
        !cooperative_process_call_gate().is_native_only_on_current_thread() &&
        state.persistent_kernels &&
        state.persistent_kernels->module_lookup_is_prepared(module, name)) {
        // Original SDK validation remains once. Private eager lookup is already
        // complete, so a warm function must not drain accepted low work.
        return state.persistent_kernels->get_source_function(function, module, name);
    }
    return intercept_cooperative_module_boundary([&] {
        if (!state.persistent_kernels) {
            return state.driver.module_get_function(function, module, name);
        }
        return cooperative_process_call_gate().is_native_only_on_current_thread()
                   ? state.persistent_kernels->get_source_function(function, module, name)
                   : state.persistent_kernels->get_function(function, module, name);
    });
}

CUresult intercept_module_unload(CUmodule module) {
    InterceptorState& state = get_state();
    if (is_inside_driver_call() || is_inside_runtime_call()) {
        return state.driver.module_unload(module);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    return intercept_cooperative_module_boundary([&] {
        if (!state.persistent_kernels) {
            return state.driver.module_unload(module);
        }
        return cooperative_process_call_gate().is_native_only_on_current_thread()
                   ? state.persistent_kernels->unload_source_module(module)
                   : state.persistent_kernels->unload_module(module);
    });
}

CUresult intercept_launch_kernel(CUfunction function, unsigned int grid_dim_x,
                                 unsigned int grid_dim_y, unsigned int grid_dim_z,
                                 unsigned int block_dim_x, unsigned int block_dim_y,
                                 unsigned int block_dim_z, unsigned int shared_memory_bytes,
                                 CUstream stream, void** kernel_parameters, void** extra,
                                 bool per_thread_default_stream) {
    InterceptorState& state = get_state();
    const KernelLaunchObservation observation{
        .api_name = per_thread_default_stream ? "cuLaunchKernel_ptsz" : "cuLaunchKernel",
        .grid_dim_x = grid_dim_x,
        .grid_dim_y = grid_dim_y,
        .grid_dim_z = grid_dim_z,
        .block_dim_x = block_dim_x,
        .block_dim_y = block_dim_y,
        .block_dim_z = block_dim_z,
        .shared_memory_bytes = shared_memory_bytes,
        .stream = stream};
    ensure_state_initialized(state);
    if (runtime_native_kernel_execution_is_pending() && is_inside_runtime_call() &&
        !is_inside_driver_call()) {
        const auto matched = dispatch_runtime_native_kernel(
            function, stream,
            {{grid_dim_x, grid_dim_y, grid_dim_z},
             {block_dim_x, block_dim_y, block_dim_z},
             shared_memory_bytes,
             per_thread_default_stream},
            [&] {
                return per_thread_default_stream
                           ? state.driver.launch_kernel_ptsz(function, grid_dim_x, grid_dim_y,
                                                             grid_dim_z, block_dim_x, block_dim_y,
                                                             block_dim_z, shared_memory_bytes,
                                                             stream, kernel_parameters, extra)
                           : state.driver.launch_kernel(function, grid_dim_x, grid_dim_y,
                                                        grid_dim_z, block_dim_x, block_dim_y,
                                                        block_dim_z, shared_memory_bytes, stream,
                                                        kernel_parameters, extra);
            });
        if (matched) return *matched;
    }
    if (should_route_cooperative(state)) {
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            const auto execute = [&] {
                return source_native_stream_execution(
                    state, stream, per_thread_default_stream, [&] {
                        return per_thread_default_stream
                                   ? state.driver.launch_kernel_ptsz(
                                         function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                                         block_dim_y, block_dim_z, shared_memory_bytes, stream,
                                         kernel_parameters, extra)
                                   : state.driver.launch_kernel(
                                         function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x,
                                         block_dim_y, block_dim_z, shared_memory_bytes, stream,
                                         kernel_parameters, extra);
                    });
            };
            const auto matched =
                dispatch_runtime_source_kernel(function,
                                               {{grid_dim_x, grid_dim_y, grid_dim_z},
                                                {block_dim_x, block_dim_y, block_dim_z},
                                                shared_memory_bytes,
                                                per_thread_default_stream},
                                               execute);
            const auto result = matched ? *matched : execute();
            if (result == CUDA_SUCCESS) {
                report_kernel_launch_observed(observation);
            }
            return result;
        }
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            return ready;
        }
        const std::array<unsigned int, 3> grid{grid_dim_x, grid_dim_y, grid_dim_z};
        const std::array<unsigned int, 3> block{block_dim_x, block_dim_y, block_dim_z};
        std::optional<CooperativeKernelVariant> variant;
        std::optional<MappedBinaryKernelVariant> mapped;
        std::optional<MappedGemmKernelVariant> gemm;
        std::optional<CudaQueuedKernel> binary_kernel;
        CUcontext binary_context{};
        if (is_inside_runtime_call()) {
            const auto plan = consume_runtime_cooperative_launch(
                function, {grid, block, shared_memory_bytes, per_thread_default_stream});
            if (!plan) {
                return CUDA_ERROR_NOT_SUPPORTED;
            }
            if (extra == nullptr) {
                variant = plan->variant;
                mapped = plan->binary_variant;
                gemm = plan->gemm_variant;
            }
        } else if (extra == nullptr) {
            if (!state.persistent_kernels->module_source_is_prepared(function)) {
                const auto prepared = intercept_cooperative_native_boundary([&] {
                    state.persistent_kernels->prepare_module_source(function);
                    return CUDA_SUCCESS;
                });
                if (prepared != CUDA_SUCCESS) {
                    return prepared;
                }
            }
            variant = state.persistent_kernels->cooperative_kernel(
                function, grid, block, shared_memory_bytes, stream, per_thread_default_stream);
            mapped = state.persistent_kernels->mapped_binary_kernel(function);
            gemm = state.persistent_kernels->mapped_gemm_kernel(function);
        }
        if (extra == nullptr && !variant && shared_memory_bytes == 0) {
            try {
                CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
                int resident{};
                if (mapped && state.driver.context_get_current(&binary_context) == CUDA_SUCCESS &&
                    binary_context == mapped->context &&
                    state.driver.stream_capture_status(stream, &capture,
                                                       per_thread_default_stream) == CUDA_SUCCESS &&
                    capture == CU_STREAM_CAPTURE_STATUS_NONE &&
                    state.driver.occupancy(&resident, mapped->function, 128, 0) == CUDA_SUCCESS &&
                    resident > 0) {
                    struct Owners {
                        std::shared_ptr<void> module;
                        std::shared_ptr<void> context;
                    };
                    auto context_owner = state.cooperative_router->resources();
                    if (mapped->resources && context_owner) {
                        auto owners = std::make_shared<Owners>(
                            Owners{mapped->resources, std::move(context_owner)});
                        binary_kernel = prepare_mapped_add_slices(
                            mapped->function, grid, block, kernel_parameters, std::move(owners),
                            state.trace_kernel_launches, mapped->mapping);
                    }
                }
            } catch (...) {
                // No private kernel was accepted or issued; preserve native fallback.
            }
        }
        if (extra == nullptr && !variant && !binary_kernel && gemm && shared_memory_bytes == 0) {
            try {
                CUdevice device{};
                CUstreamCaptureStatus capture = CU_STREAM_CAPTURE_STATUS_NONE;
                int resident{};
                if (state.driver.context_get_current(&binary_context) == CUDA_SUCCESS &&
                    binary_context == gemm->context &&
                    state.driver.context_get_device(&device) == CUDA_SUCCESS &&
                    state.driver.stream_capture_status(stream, &capture,
                                                       per_thread_default_stream) == CUDA_SUCCESS &&
                    capture == CU_STREAM_CAPTURE_STATUS_NONE &&
                    state.driver.occupancy(&resident, gemm->function, 256, 0) == CUDA_SUCCESS &&
                    resident > 0) {
                    struct Owners {
                        std::shared_ptr<void> module;
                        std::shared_ptr<void> context;
                    };
                    auto context_owner = state.cooperative_router->resources();
                    if (gemm->resources && context_owner) {
                        auto owners = std::make_shared<Owners>(
                            Owners{gemm->resources, std::move(context_owner)});
                        // End the private native scope before frontend issue or
                        // fallback; it must not bypass logical stream ordering.
                        const NativeDriverCallScope native;
                        binary_kernel = prepare_checked_mapped_gemm_slices(
                            gemm->function, grid, block, kernel_parameters, std::move(owners),
                            state.cooperative_router->copy_range_dispatch(), state.allocations,
                            binary_context, device, state.trace_kernel_launches, gemm->mapping);
                    }
                }
            } catch (...) {
                // Optional preparation has not accepted or issued any kernel.
            }
        }
        if (gemm && state.trace_kernel_launches) {
            const auto saved_errno = errno;
            report_diagnostic(binary_kernel ? "[glimmer] gemm_candidate prepared=1\n"
                                            : "[glimmer] gemm_candidate prepared=0\n");
            errno = saved_errno;
        }
        const auto result =
            binary_kernel.has_value()
                ? frontend->launch_sliced(binary_context, stream, per_thread_default_stream,
                                          std::move(*binary_kernel))
            : variant.has_value()
                ? frontend->launch(*variant, stream, per_thread_default_stream, grid, block,
                                   kernel_parameters, state.cooperative_router->resources())
                : state.cooperative_router->native_execution(
                      cooperative_process_call_gate(), frontend, stream, per_thread_default_stream,
                      [&] {
                          return per_thread_default_stream
                                     ? state.driver.launch_kernel_ptsz(
                                           function, grid_dim_x, grid_dim_y, grid_dim_z,
                                           block_dim_x, block_dim_y, block_dim_z,
                                           shared_memory_bytes, stream, kernel_parameters, extra)
                                     : state.driver.launch_kernel(
                                           function, grid_dim_x, grid_dim_y, grid_dim_z,
                                           block_dim_x, block_dim_y, block_dim_z,
                                           shared_memory_bytes, stream, kernel_parameters, extra);
                      });
        if (is_inside_runtime_call()) complete_runtime_cooperative_launch(result);
        if (result == CUDA_SUCCESS) {
            report_kernel_launch_observed(observation);
        }
        return result;
    }
    return intercept_driver_kernel_launch(
        state, observation, stream, per_thread_default_stream,
        [&] {
            return per_thread_default_stream ? state.driver.has_launch_kernel_ptsz()
                                             : state.driver.has_launch_kernel();
        },
        [&] {
            if (!is_inside_driver_call() && is_inside_runtime_call()) {
                const auto replacement = consume_runtime_driver_replacement(function);
                if (replacement.has_value()) {
                    return per_thread_default_stream
                               ? state.driver.launch_kernel_ptsz(
                                     replacement->persistent, replacement->worker_blocks, 1, 1,
                                     block_dim_x, block_dim_y, block_dim_z, shared_memory_bytes,
                                     stream, replacement->arguments, nullptr)
                               : state.driver.launch_kernel(
                                     replacement->persistent, replacement->worker_blocks, 1, 1,
                                     block_dim_x, block_dim_y, block_dim_z, shared_memory_bytes,
                                     stream, replacement->arguments, nullptr);
                }
            }
            if (state.persistent_kernels != nullptr && !is_inside_driver_call() &&
                !is_inside_runtime_call()) {
                return state.persistent_kernels->launch(
                    function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                    block_dim_z, shared_memory_bytes, stream, kernel_parameters, extra,
                    per_thread_default_stream);
            }
            return per_thread_default_stream
                       ? state.driver.launch_kernel_ptsz(
                             function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                             block_dim_z, shared_memory_bytes, stream, kernel_parameters, extra)
                       : state.driver.launch_kernel(
                             function, grid_dim_x, grid_dim_y, grid_dim_z, block_dim_x, block_dim_y,
                             block_dim_z, shared_memory_bytes, stream, kernel_parameters, extra);
        });
}

CUresult intercept_launch_kernel_ex(const CUlaunchConfig* config, CUfunction function,
                                    void** kernel_parameters, void** extra,
                                    bool per_thread_default_stream) {
    if (config == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    const KernelLaunchObservation observation{
        .api_name = per_thread_default_stream ? "cuLaunchKernelEx_ptsz" : "cuLaunchKernelEx",
        .grid_dim_x = config->gridDimX,
        .grid_dim_y = config->gridDimY,
        .grid_dim_z = config->gridDimZ,
        .block_dim_x = config->blockDimX,
        .block_dim_y = config->blockDimY,
        .block_dim_z = config->blockDimZ,
        .shared_memory_bytes = config->sharedMemBytes,
        .stream = config->hStream};
    ensure_state_initialized(state);
    if (should_route_cooperative(state)) {
        if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
            const auto execute = [&] {
                return source_native_stream_execution(
                    state, config->hStream, per_thread_default_stream, [&] {
                        return state.driver.launch_kernel_ex(config, function, kernel_parameters,
                                                             extra, per_thread_default_stream);
                    });
            };
            const auto matched = dispatch_runtime_source_kernel(
                function,
                {{config->gridDimX, config->gridDimY, config->gridDimZ},
                 {config->blockDimX, config->blockDimY, config->blockDimZ},
                 config->sharedMemBytes,
                 per_thread_default_stream},
                execute);
            const auto result = matched ? *matched : execute();
            if (result == CUDA_SUCCESS) {
                report_kernel_launch_observed(observation);
            }
            return result;
        }
        std::shared_ptr<CudaCooperativeFrontend> frontend;
        const auto ready = cooperative_frontend_current(state, frontend);
        if (ready != CUDA_SUCCESS) {
            return ready;
        }
        const auto result = state.cooperative_router->native_execution(
            cooperative_process_call_gate(), frontend, config->hStream, per_thread_default_stream,
            [&] {
                return state.driver.launch_kernel_ex(config, function, kernel_parameters, extra,
                                                     per_thread_default_stream);
            });
        if (result == CUDA_SUCCESS) {
            report_kernel_launch_observed(observation);
        }
        return result;
    }
    return intercept_driver_kernel_launch(
        state, observation, config->hStream, per_thread_default_stream,
        [&] { return state.driver.has_launch_kernel_ex(per_thread_default_stream); },
        [&] {
            return state.driver.launch_kernel_ex(config, function, kernel_parameters, extra,
                                                 per_thread_default_stream);
        });
}

CUresult intercept_graph_launch(CUgraphExec graph_exec, CUstream stream,
                                bool per_thread_default_stream) {
    if (graph_exec == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    InterceptorState& state = get_state();
    const auto invoke = [&state, graph_exec, stream, per_thread_default_stream] {
        return per_thread_default_stream ? state.driver.graph_launch_ptsz(graph_exec, stream)
                                         : state.driver.graph_launch(graph_exec, stream);
    };

    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return invoke();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (should_route_cooperative(state)) {
        if (!state.cooperative_router) return CUDA_ERROR_NOT_SUPPORTED;
        const CooperativeBoundaryScope boundary;
        return execute_cooperative_graph(state, graph_exec, stream, per_thread_default_stream,
                                         invoke);
    }
    const bool launch_available = per_thread_default_stream ? state.driver.has_graph_launch_ptsz()
                                                            : state.driver.has_graph_launch();
    if (!launch_available) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    const bool trace_launch_timings = state.trace_launch_timings;
    glimmer::control::LaunchGateTiming gate_timing;
    std::optional<glimmer::core::TaskId> launch_task;
    if (launch_scheduling_is_enforced()) {
        const auto capturing = launch_stream_is_capturing(stream, per_thread_default_stream);
        if (!capturing.has_value()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        if (capturing.value()) {
            return invoke();
        }
        launch_task = acquire_launch_slot(trace_launch_timings ? &gate_timing : nullptr);
        if (!launch_task.has_value()) {
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    }

    const auto cuda_launch_started = trace_launch_timings ? std::chrono::steady_clock::now()
                                                          : std::chrono::steady_clock::time_point{};
    const CUresult launch_result = invoke();
    const auto cuda_launch_finished = trace_launch_timings
                                          ? std::chrono::steady_clock::now()
                                          : std::chrono::steady_clock::time_point{};
    if (launch_task.has_value() && launch_result != CUDA_SUCCESS) {
        fail_launch_slot(launch_task.value());
    }

    bool event_tracking_succeeded = false;
    bool batch_call_succeeded = true;
    std::uint64_t event_tracking_nanoseconds = 0;
    if (launch_task.has_value() && launch_result == CUDA_SUCCESS) {
        const auto event_tracking_started = trace_launch_timings
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        event_tracking_succeeded = track_launch_completion(
            launch_task.value(),
            per_thread_default_stream && stream == nullptr ? CU_STREAM_PER_THREAD : stream);
        if (trace_launch_timings) {
            event_tracking_nanoseconds =
                elapsed_nanoseconds(event_tracking_started, std::chrono::steady_clock::now());
        }
        if (event_tracking_succeeded) {
            batch_call_succeeded = finish_launch_batch_call(launch_task.value());
        }
    }
    if (trace_launch_timings) {
        report_launch_timing_observed(
            launch_task.value_or(0), gate_timing,
            elapsed_nanoseconds(cuda_launch_started, cuda_launch_finished),
            event_tracking_nanoseconds, static_cast<std::uint32_t>(launch_result),
            event_tracking_succeeded);
    }
    if (launch_task.has_value() && launch_result == CUDA_SUCCESS &&
        (!event_tracking_succeeded || !batch_call_succeeded)) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (launch_result == CUDA_SUCCESS) {
        report_graph_launch_observed(
            per_thread_default_stream ? "cuGraphLaunch_ptsz" : "cuGraphLaunch",
            reinterpret_cast<const void*>(stream));
    }
    return launch_result;
}

CUresult intercept_mem_alloc(CUdeviceptr* device_pointer, std::size_t memory_bytes) {
    InterceptorState& state = get_state();
    return intercept_tracked_allocation(
        device_pointer, memory_bytes,
        [&state, device_pointer, memory_bytes] {
            return state.driver.mem_alloc(device_pointer, memory_bytes);
        },
        false, AllocationProvenance::kIndependentDevice);
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
                                  *actual_bytes, AllocationScope::kContextBound,
                                  AllocationProvenance::kIndependentDevice)) {
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

CudaCooperativeReleaseResult release_tracked_pointer(InterceptorState& state,
                                                     CUdeviceptr device_pointer,
                                                     bool is_private_progress) {
    const auto native_free = [&] {
        const auto result = state.driver.mem_free(device_pointer);
        if (result != CUDA_SUCCESS && is_private_progress) {
            state.allocations.mark_accounting_degraded();
        }
        return CudaCooperativeReleaseResult{result, result == CUDA_SUCCESS
                                                        ? CudaNativeReleaseState::kReleased
                                                        : CudaNativeReleaseState::kUncertain};
    };
    if (state.quota_mode != QuotaMode::kEnabled) {
        return native_free();
    }

    const std::optional<ContextIdentity> context = capture_context_identity(state);
    if (!context.has_value()) {
        auto free_result = native_free();
        if (free_result.status == CUDA_SUCCESS) {
            state.allocations.mark_accounting_degraded();
            free_result.status = CUDA_ERROR_UNKNOWN;
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
        return {CUDA_ERROR_INVALID_VALUE, CudaNativeReleaseState::kRetained};
    }
    if (release_status == AllocationRegistry::ReleaseStatus::kUnknown ||
        !release_ticket.has_value()) {
        return native_free();
    }

    auto free_result = native_free();
    if (free_result.status != CUDA_SUCCESS) {
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
        free_result.status = CUDA_ERROR_UNKNOWN;
    }
    return free_result;
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
    return release_tracked_pointer(state, device_pointer, false).status;
}

CudaCooperativeReleaseResult release_cooperative_progress(CUdeviceptr pointer) noexcept {
    InterceptorState* state = nullptr;
    try {
        state = &get_state();
        if (!ensure_initialized(*state)) {
            return {CUDA_ERROR_UNKNOWN, CudaNativeReleaseState::kRetained};
        }
        return release_tracked_pointer(*state, pointer, true);
    } catch (...) {
        if (state) {
            state->allocations.mark_accounting_degraded();
        }
        return {CUDA_ERROR_UNKNOWN, CudaNativeReleaseState::kUncertain};
    }
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

CUresult intercept_primary_context_release(CUdevice device, bool is_modern) {
    auto& state = get_state();
    const auto original = [&] { return state.driver.primary_context_release(device, is_modern); };
    if (is_inside_driver_call() || g_is_inside_cooperative_boundary) {
        return original();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!state.driver.has_primary_context_release(is_modern)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!should_route_cooperative(state)) {
        return original();
    }
    auto& gate = cooperative_process_call_gate();
    const ProcessCallGate::Call call(gate);
    if (call.is_native_only()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!state.cooperative_router || !state.persistent_kernels ||
        state.allocations.is_accounting_degraded()) {
        return CUDA_ERROR_UNKNOWN;
    }
    const CooperativeBoundaryScope boundary;
    CUresult native_result = CUDA_ERROR_UNKNOWN;
    bool was_called{};
    bool has_uncertain_lifetime{};
    const auto result = state.cooperative_router->release_primary_context(
        gate, device,
        [&] {
            was_called = true;
            native_result = original();
            return native_result;
        },
        [&](CUcontext target) {
            const auto retired = state.persistent_kernels->forget_context_checked(target);
            if (retired != CUDA_SUCCESS) {
                return retired;
            }
            if (state.quota_mode == QuotaMode::kEnabled) {
                const auto summaries = state.allocations.erase_context_by_device(target);
                if (!summaries) {
                    state.allocations.mark_accounting_degraded();
                    return CUDA_ERROR_UNKNOWN;
                }
                finalize_async_releases(state, *summaries);
                if (state.allocations.is_accounting_degraded()) {
                    return CUDA_ERROR_UNKNOWN;
                }
            }
            return CUDA_SUCCESS;
        },
        has_uncertain_lifetime);
    if (has_uncertain_lifetime) {
        // Failed native effects and internal lifetime proof retain usage. Do not
        // let a later memory operation turn unknown ownership into free quota.
        state.allocations.mark_accounting_degraded();
    }
    return was_called && native_result != CUDA_SUCCESS ? native_result : result;
}

CUresult intercept_primary_context_reset(CUdevice device, bool is_modern) {
    auto& state = get_state();
    const auto original = [&] { return state.driver.primary_context_reset(device, is_modern); };
    if (is_inside_driver_call() || g_is_inside_cooperative_boundary) {
        return original();
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!state.driver.has_primary_context_reset(is_modern)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!should_route_cooperative(state)) {
        return original();
    }
    if (cooperative_process_call_gate().is_native_only_on_current_thread()) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    if (!state.cooperative_router || !state.persistent_kernels ||
        state.allocations.is_accounting_degraded() ||
        state.cooperative_router->normal_owner_status() != CUDA_SUCCESS) {
        return CUDA_ERROR_UNKNOWN;
    }
    CUcontext target{};
    const auto identity = state.driver.active_primary_context(device, target);
    if (identity != CUDA_SUCCESS) {
        if (identity == CUDA_ERROR_INVALID_DEVICE || identity == CUDA_ERROR_NOT_INITIALIZED) {
            // Proven invalid/cold validation cannot destroy an existing owner.
            return original();
        }
        state.allocations.mark_accounting_degraded();
        return CUDA_ERROR_UNKNOWN;
    }
    if (!target) {
        return original();
    }
    const CooperativeBoundaryScope boundary;
    CUresult native_result = CUDA_ERROR_UNKNOWN;
    bool was_called{};
    const auto closed = state.cooperative_router->close_context(target, [&] {
        const auto retired = state.cooperative_router->with_context(target, [&] {
            state.persistent_kernels->forget_context(target);
            return CUDA_SUCCESS;
        });
        if (retired != CUDA_SUCCESS) {
            return retired;
        }
        // The native reset runs on its SDK/Driver caller, not the temporary
        // private cleanup target. Its original post-reset context is untouched.
        was_called = true;
        native_result = original();
        return native_result;
    });
    if (!was_called || native_result != CUDA_SUCCESS) {
        return was_called ? native_result : closed;
    }
    if (closed != CUDA_SUCCESS) {
        return closed;
    }
    if (state.quota_mode == QuotaMode::kEnabled) {
        const auto summaries = state.allocations.erase_context_by_device(target);
        if (!summaries) {
            state.allocations.mark_accounting_degraded();
            return CUDA_ERROR_UNKNOWN;
        }
        finalize_async_releases(state, *summaries);
        if (state.allocations.is_accounting_degraded()) {
            return CUDA_ERROR_UNKNOWN;
        }
    }
    return native_result;
}

CUresult intercept_context_destroy(CUcontext context, bool is_modern) {
    InterceptorState& state = get_state();
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        return state.driver.context_destroy(context, is_modern);
    }
    if (!ensure_initialized(state)) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (!state.driver.has_context_destroy(is_modern)) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    const auto retire_destroyed = [&](CUcontext target) {
        if (state.persistent_kernels) {
            const auto retired = state.persistent_kernels->forget_context_checked(target);
            if (retired != CUDA_SUCCESS) {
                state.allocations.mark_accounting_degraded();
                return retired;
            }
        }
        if (state.quota_mode == QuotaMode::kEnabled) {
            const auto summaries = state.allocations.erase_context_by_device(target);
            if (!summaries) {
                state.allocations.mark_accounting_degraded();
                return CUDA_ERROR_UNKNOWN;
            }
            finalize_async_releases(state, *summaries);
            if (state.allocations.is_accounting_degraded()) {
                return CUDA_ERROR_UNKNOWN;
            }
        }
        return CUDA_SUCCESS;
    };
    if (should_route_cooperative(state)) {
        if (!state.cooperative_router || state.allocations.is_accounting_degraded()) {
            return CUDA_ERROR_UNKNOWN;
        }
        const CooperativeBoundaryScope scope;
        CudaContextDestroyDisposition disposition;
        const auto result = state.cooperative_router->destroy_context(
            cooperative_process_call_gate(), context,
            [&] { return state.driver.context_destroy(context, is_modern); }, retire_destroyed,
            is_modern, disposition);
        if (disposition.has_uncertain_lifetime) {
            state.allocations.mark_accounting_degraded();
        }
        return result;
    }
    const CUresult destroy_result = state.driver.context_destroy(context, is_modern);
    if (destroy_result != CUDA_SUCCESS) {
        return destroy_result;
    }
    return retire_destroyed(context);
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
    const std::size_t physical_total_bytes = *total_bytes;
    const std::size_t physical_free_bytes = *free_bytes;
    const glimmer::control::MemoryInfo memory_info =
        state.quota->get_memory_info(device, *total_bytes, *free_bytes);
    if (!state.quota->is_healthy()) {
        return CUDA_ERROR_UNKNOWN;
    }
    *total_bytes = static_cast<std::size_t>(memory_info.total_bytes);
    *free_bytes = static_cast<std::size_t>(memory_info.free_bytes);
    report_memory_info_observed("cuMemGetInfo_v2", device, static_cast<std::uint64_t>(*total_bytes),
                                static_cast<std::uint64_t>(*free_bytes), physical_total_bytes,
                                physical_free_bytes);
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

bool initialize_runtime_driver_lookup() {
    // Runtime may query the Driver before entering any intercepted Runtime API.
    // Initialize outside RuntimeCallScope so its lookup does not use an empty dispatch table.
    auto& state = get_state();
    if (is_inside_driver_call()) {
        // Publish only original Driver symbols. A first trusted SDK lookup
        // must not initialize the cooperative owner or import private actors.
        return initialize_native_driver_dispatch(state);
    }
    return ensure_initialized(state);
}

CUresult intercept_get_proc_address(const char* symbol, void** function_pointer, int cuda_version,
                                    cuuint64_t flags) {
    InterceptorState& state = get_state();
    if (symbol == nullptr || symbol[0] == '\0' || function_pointer == nullptr) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (glimmer::interceptor::is_inside_driver_call() ||
        glimmer::interceptor::is_inside_runtime_call()) {
        const auto result =
            state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
        if (result == CUDA_SUCCESS) {
            replace_runtime_proc_address(state.driver, symbol, function_pointer, flags,
                                         cuda_version);
        }
        return result;
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

    if (void* intercepted_symbol = find_proc_address_interceptor(symbol, flags, cuda_version);
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
        CUresult result = CUDA_ERROR_NOT_SUPPORTED;
        if (state.driver.has_get_proc_address_v2()) {
            result = state.driver.get_proc_address_v2(symbol, function_pointer, cuda_version, flags,
                                                      symbol_status);
        } else if (state.driver.has_get_proc_address()) {
            result = state.driver.get_proc_address(symbol, function_pointer, cuda_version, flags);
        }
        if (result == CUDA_SUCCESS) {
            replace_runtime_proc_address(state.driver, symbol, function_pointer, flags,
                                         cuda_version);
        }
        return result;
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
        if (result == CUDA_ERROR_NOT_SUPPORTED && state.driver.has_get_proc_address()) {
            // Older driver loaders may export the v2 entry point while only
            // implementing the legacy resolver for a particular query. Keep
            // the application-visible lookup compatible with that driver.
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

    if (void* intercepted_symbol = find_proc_address_interceptor(symbol, flags, cuda_version);
        intercepted_symbol != nullptr) {
        *function_pointer = intercepted_symbol;
    }
    return CUDA_SUCCESS;
}

}  // namespace glimmer::interceptor
