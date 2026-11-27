#include "internal/driver_api_interceptor_state.h"

namespace glimmer::interceptor {

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

}  // namespace glimmer::interceptor
