#include "internal/driver_api_interceptor_state.h"

namespace glimmer::interceptor {

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
