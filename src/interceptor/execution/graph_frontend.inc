// Included in the owned frontend namespace. All metadata is protected by State's
// exclusion; RPC waits retain context/object pins and recheck the context epoch.
struct GraphReply {
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<std::int32_t> original{};
    std::uint64_t id{};
    ExecutionGraphObservation observation{};
    bool has_observation{};
};

struct GraphPreparationScope {
    bool is_previous{is_graph_preparation};
    explicit GraphPreparationScope(bool is_allowed) {
        is_graph_preparation = is_allowed;
    }
    ~GraphPreparationScope() {
        is_graph_preparation = is_previous;
    }
    GraphPreparationScope(const GraphPreparationScope&) = delete;
    GraphPreparationScope& operator=(const GraphPreparationScope&) = delete;
};

inline bool graph_quarantine_receipt(ExecutionOpcode operation,
                                     const ExecutionWireReply& reply) noexcept {
    if ((operation != ExecutionOpcode::kGraphControl &&
         operation != ExecutionOpcode::kGraphLaunch) ||
        reply.disposition != ExecutionWireDisposition::kRejected || reply.status ||
        !reply.original_status || *reply.original_status < 0 || reply.id || reply.address ||
        reply.parameter_bytes || !reply.arguments.empty() ||
        reply.data.size() != kExecutionGraphObservationBytes)
        return false;
    ExecutionPayloadReader reader(reply.data);
    ExecutionGraphObservation observation;
    if (!read_execution_graph_observation(reader, observation) || !reader.empty() ||
        !observation.is_quarantined)
        return false;
    // This exception only transports an actual error/unknown-effects receipt
    // to its typed validator. It cannot publish a capability or authorize work.
    return operation == ExecutionOpcode::kGraphControl ||
           (!observation.output_written && !observation.capture_status);
}

inline bool graph_capture_owner(const State& shared, const Entry* context,
                                const Entry* stream) noexcept {
    if (!shared.graph_capture) return false;
    const auto& capture = *shared.graph_capture;
    return capture.context == context && context && context->is_alive &&
           context->epoch == capture.epoch && !context->is_uncertain &&
           capture.caller == graph_caller && graph_caller && !capture.is_transition &&
           capture.status == cudaStreamCaptureStatusActive && (!stream || capture.stream == stream);
}

inline bool acquire_graph_caller(State& shared) noexcept {
    if (graph_caller) return true;
    if (!shared.next_graph_caller) return false;
    graph_caller = shared.next_graph_caller;
    shared.next_graph_caller = graph_caller == UINT64_MAX ? 0 : graph_caller + 1;
    return true;
}

inline GraphReply call_graph(State& shared, Exclusion& exclusion, Entry& context,
                             const ExecutionGraphRequest& request) {
    ExecutionPayloadWriter payload;
    if (!write_execution_graph_request(payload, request)) return {CUDA_ERROR_OUT_OF_MEMORY};
    if (context.active_calls >= kMaxHandles - 1 || context.active_mutations >= kMaxHandles - 1)
        return {CUDA_ERROR_OUT_OF_MEMORY};
    CounterPin owner(context.active_calls, exclusion);
    CounterPin mutation(context.active_mutations, exclusion);
    const auto epoch = context.epoch;
    const auto invalid = [&]() -> GraphReply {
        context.is_uncertain = true;
        return {};
    };
    std::optional<ExecutionWireReply> reply;
    try {
        reply = shared.call(exclusion, context, ExecutionOpcode::kGraphControl, std::move(payload));
    } catch (...) {
        // An exception is not proof that the authenticated request was never
        // consumed. Keep construction ownership and poison this context.
        return invalid();
    }
    if (!reply || !context.is_alive || context.epoch != epoch || context.is_uncertain ||
        reply->status < 0 || reply->status > 999 || reply->address || reply->parameter_bytes ||
        !reply->arguments.empty())
        return invalid();
    GraphReply result{static_cast<CUresult>(reply->status), reply->original_status, reply->id};
    if (!reply->data.empty()) {
        ExecutionPayloadReader reader(reply->data);
        if (!read_execution_graph_observation(reader, result.observation) || !reader.empty())
            return invalid();
        result.has_observation = true;
    }
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        // Session rejection describes failed ownership/policy proof, not
        // necessarily pre-native rejection. End may have actually run and
        // written output before its independent proof failed.
        if ((!reply->status && !result.observation.is_quarantined) || reply->id ||
            (!result.has_observation && reply->original_status))
            return invalid();
    } else if (reply->disposition != ExecutionWireDisposition::kComplete ||
               !result.has_observation || reply->status || result.observation.is_quarantined ||
               !reply->original_status) {
        return invalid();
    }
    const bool is_object_constructor = request.action == ExecutionGraphAction::kEnd ||
                                       request.action == ExecutionGraphAction::kInstantiate;
    if ((!request.has_output && (result.observation.output_written || result.id)) ||
        (result.original && *result.original < 0) || (result.id && !is_object_constructor) ||
        (result.id && (!result.observation.output_written || result.status != CUDA_SUCCESS ||
                       result.original != 0)) ||
        (is_object_constructor && result.status == CUDA_SUCCESS && result.original == 0 &&
         !result.observation.is_quarantined && (!request.has_output || !result.id)))
        return invalid();
    if (result.observation.is_quarantined) {
        context.is_uncertain = true;
        if (result.status == CUDA_SUCCESS) result.status = CUDA_ERROR_UNKNOWN;
    }
    return result;
}

inline CUresult reject_graph_capture(State& shared, Exclusion& exclusion) {
    if (!shared.graph_capture) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    auto& capture = *shared.graph_capture;
    if (capture.context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    if (capture.is_transition) return CUDA_ERROR_NOT_READY;
    if (capture.status == cudaStreamCaptureStatusInvalidated)
        return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    // Policy invalidation must reach the retained service construction owner;
    // it is not native End or permission to release construction exclusion.
    const auto context = capture.context;
    const auto request = ExecutionGraphRequest{.action = ExecutionGraphAction::kInvalidate,
                                               .stream = capture.stream->remote,
                                               .caller = capture.caller};
    capture.is_transition = true;
    const auto reply = call_graph(shared, exclusion, *context, request);
    if (shared.graph_capture) {
        shared.graph_capture->is_transition = false;
        shared.graph_capture->status = cudaStreamCaptureStatusInvalidated;
    }
    return context->is_uncertain ? CUDA_ERROR_UNKNOWN
           : reply.status == CUDA_SUCCESS || reply.status == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED
               ? CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED
               : reply.status;
}

inline std::optional<ExecutionWireReply> graph_rpc_gate(State& shared, Exclusion& exclusion,
                                                        Entry& context, ExecutionOpcode operation,
                                                        bool is_observation) {
    if (!shared.graph_capture || operation == ExecutionOpcode::kGraphControl) return std::nullopt;
    // Begin drains all prior physical work. These exact background operations
    // only retire already owned CPU frontier metadata; they issue no new work.
    if (is_observation && (operation == ExecutionOpcode::kQueryFrontier ||
                           operation == ExecutionOpcode::kReleaseFrontier ||
                           operation == ExecutionOpcode::kQueryAsyncFrontiers ||
                           operation == ExecutionOpcode::kReleaseAsyncFrontiers))
        return std::nullopt;
    // Existing cold carriers authenticate the session, not a wire caller token.
    // This lexical owner check is additional frontend thread provenance.
    if (is_graph_preparation && graph_capture_owner(shared, &context) &&
        (operation == ExecutionOpcode::kLoadPtx || operation == ExecutionOpcode::kLoadBinary ||
         operation == ExecutionOpcode::kLoadModuleFile ||
         operation == ExecutionOpcode::kGetFunction))
        return std::nullopt;
    return local_failure(reject_graph_capture(shared, exclusion));
}

inline CUresult capture_graph_command(State& shared, Exclusion& exclusion, Entry& context,
                                      Entry* stream, ExecutionGraphAction action,
                                      std::span<const std::byte> payload) {
    if (!stream || !graph_capture_owner(shared, &context, stream))
        return reject_graph_capture(shared, exclusion);
    const ExecutionGraphRequest request{
        .action = action, .stream = stream->remote, .caller = graph_caller, .payload = payload};
    shared.graph_capture->is_transition = true;
    const auto reply = call_graph(shared, exclusion, context, request);
    if (shared.graph_capture) {
        shared.graph_capture->is_transition = false;
        if (reply.has_observation)
            shared.graph_capture->status =
                static_cast<cudaStreamCaptureStatus>(reply.observation.capture_status);
    }
    // Node construction originals are Driver calls, unlike Runtime lifecycle.
    if (reply.status != CUDA_SUCCESS) return reply.status;
    if (reply.original && (*reply.original < 0 || *reply.original > 999)) {
        context.is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    return reply.original ? static_cast<CUresult>(*reply.original) : CUDA_SUCCESS;
}

inline cudaError_t graph_runtime_result(CUresult proof,
                                        std::optional<std::int32_t> original) noexcept {
    if (proof != CUDA_SUCCESS || !original) return runtime_result(proof);
    const auto result = static_cast<cudaError_t>(*original);
    if (result != cudaSuccess) runtime_last_error = result;
    return result;
}

inline bool graph_output_seed(void* output, ExecutionGraphRequest& request) noexcept {
    request.has_output = output != nullptr;
    return !output || (writable_host_range(output, sizeof(request.output_seed)) &&
                       self_copy(&request.output_seed, output, sizeof(request.output_seed), false));
}

inline CUresult graph_begin(State& shared, Exclusion& exclusion, cudaStream_t handle,
                            cudaStreamCaptureMode mode, std::optional<std::int32_t>& original) {
    if (mode < cudaStreamCaptureModeGlobal || mode > cudaStreamCaptureModeRelaxed)
        return CUDA_ERROR_INVALID_VALUE;
    if (shared.graph_capture) return reject_graph_capture(shared, exclusion);
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    if (!stream || !(stream->stream_flags & CU_STREAM_NON_BLOCKING))
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    if (!acquire_graph_caller(shared)) return CUDA_ERROR_OUT_OF_MEMORY;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (stream->active_calls == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin stream_use(stream->active_calls, exclusion);
    const auto epoch = context->epoch;
    const auto stopped = runtime_quiescent(shared, exclusion);
    if (stopped != CUDA_SUCCESS) return stopped;
    if (shared.graph_capture || !stream->is_alive || stream->is_retiring ||
        context->epoch != epoch || !context->is_alive || context->is_closing)
        return CUDA_ERROR_NOT_READY;
    shared.graph_capture =
        GraphCapture{context, stream, epoch, graph_caller, mode, cudaStreamCaptureStatusNone, true};
    const auto reply = call_graph(shared, exclusion, *context,
                                  {.action = ExecutionGraphAction::kBegin,
                                   .stream = stream->remote,
                                   .caller = graph_caller,
                                   .flags = static_cast<std::uint64_t>(mode)});
    original = reply.original;
    if (reply.has_observation && !reply.observation.is_quarantined) {
        if (reply.observation.capture_status == cudaStreamCaptureStatusNone) {
            shared.graph_capture.reset();
        } else {
            shared.graph_capture->status =
                static_cast<cudaStreamCaptureStatus>(reply.observation.capture_status);
            shared.graph_capture->is_transition = false;
        }
    } else if (reply.status != CUDA_SUCCESS && !context->is_uncertain && !reply.original) {
        shared.graph_capture.reset();  // Authenticated pre-native rejection only.
    } else {
        context->is_uncertain = true;
    }
    return reply.status;
}

inline CUresult publish_graph_object(State& shared, Entry& context, Entry& object, void* output,
                                     const GraphReply& reply) {
    if (!reply.observation.output_written) return reply.status;
    std::uint64_t value{};
    if (reply.id) {
        if (std::ranges::any_of(shared.entries, [&](const auto& entry) {
                return entry.get() != &object && entry->is_alive && entry->context == &context &&
                       entry->kind == object.kind && entry->remote == reply.id;
            })) {
            context.is_uncertain = true;
            return CUDA_ERROR_UNKNOWN;
        }
        object.remote = reply.id;
        object.epoch = context.epoch;
        object.is_alive = true;
        value = reinterpret_cast<std::uintptr_t>(&object);
    } else {
        // Preserve actual failed/partial output bytes without registering any
        // capability. Subsequent lookup still requires a live typed Entry.
        value = reply.observation.output;
    }
    if (!self_copy(output, &value, sizeof(value), true))
        return reply.status != CUDA_SUCCESS ? reply.status : CUDA_ERROR_INVALID_VALUE;
    return reply.status;
}

inline CUresult graph_end(State& shared, Exclusion& exclusion, cudaStream_t handle,
                          cudaGraph_t* output, std::optional<std::int32_t>& original) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    if (!shared.graph_capture || shared.graph_capture->context != context ||
        shared.graph_capture->stream != stream)
        return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    if (!acquire_graph_caller(shared)) return CUDA_ERROR_OUT_OF_MEMORY;
    auto& capture = *shared.graph_capture;
    if (capture.is_transition) return CUDA_ERROR_NOT_READY;
    if (capture.mode != cudaStreamCaptureModeRelaxed && capture.caller != graph_caller)
        return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    ExecutionGraphRequest request{
        .action = ExecutionGraphAction::kEnd, .stream = stream->remote, .caller = graph_caller};
    if (!graph_output_seed(output, request)) return CUDA_ERROR_INVALID_VALUE;
    auto* graph = shared.allocate_entry(Kind::kGraph, context);
    if (!graph) return CUDA_ERROR_OUT_OF_MEMORY;
    capture.is_transition = true;
    const auto reply = call_graph(shared, exclusion, *context, request);
    original = reply.original;
    if (reply.has_observation && !reply.observation.is_quarantined) {
        if (reply.observation.capture_status == cudaStreamCaptureStatusNone)
            shared.graph_capture.reset();
        else {
            shared.graph_capture->status =
                static_cast<cudaStreamCaptureStatus>(reply.observation.capture_status);
            shared.graph_capture->is_transition = false;
        }
    } else if (shared.graph_capture) {
        shared.graph_capture->is_transition = false;
    }
    return publish_graph_object(shared, *context, *graph, output, reply);
}

inline CUresult graph_capture_status(State& shared, Exclusion& exclusion, cudaStream_t handle,
                                     cudaStreamCaptureStatus* output,
                                     std::optional<std::int32_t>& original) {
    if (!writable(output)) return CUDA_ERROR_INVALID_VALUE;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    if (!shared.graph_capture)
        return publish(output, cudaStreamCaptureStatusNone) ? CUDA_SUCCESS
                                                            : CUDA_ERROR_INVALID_VALUE;
    auto& capture = *shared.graph_capture;
    if (capture.is_transition) return CUDA_ERROR_NOT_READY;
    if (capture.context != shared.context() || capture.stream != stream)
        return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    if (!acquire_graph_caller(shared)) return CUDA_ERROR_OUT_OF_MEMORY;
    if (capture.mode != cudaStreamCaptureModeRelaxed && capture.caller != graph_caller)
        return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    capture.is_transition = true;
    const auto reply = call_graph(
        shared, exclusion, *capture.context,
        {.action = ExecutionGraphAction::kState, .stream = stream->remote, .caller = graph_caller});
    original = reply.original;
    shared.graph_capture->is_transition = false;
    if (reply.has_observation)
        shared.graph_capture->status =
            static_cast<cudaStreamCaptureStatus>(reply.observation.capture_status);
    if (reply.status != CUDA_SUCCESS || (reply.original && *reply.original)) return reply.status;
    return publish(output, shared.graph_capture->status) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

inline CUresult graph_instantiate(State& shared, Exclusion& exclusion, cudaGraphExec_t* output,
                                  cudaGraph_t handle, unsigned long long flags,
                                  std::optional<std::int32_t>& original) {
    if (shared.graph_capture) return reject_graph_capture(shared, exclusion);
    auto* context = shared.context();
    auto* graph = shared.find(handle, Kind::kGraph);
    if (!context || !graph || graph->context != context || graph->epoch != context->epoch ||
        graph->is_retiring || graph->is_uncertain)
        return CUDA_ERROR_INVALID_HANDLE;
    if (graph->active_calls == kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin source(graph->active_calls, exclusion);
    ExecutionGraphRequest request{
        .action = ExecutionGraphAction::kInstantiate, .object = graph->remote, .flags = flags};
    if (!graph_output_seed(output, request)) return CUDA_ERROR_INVALID_VALUE;
    auto* executable = shared.allocate_entry(Kind::kGraphExecutable, context);
    if (!executable) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto reply = call_graph(shared, exclusion, *context, request);
    original = reply.original;
    return publish_graph_object(shared, *context, *executable, output, reply);
}

inline CUresult graph_destroy(State& shared, Exclusion& exclusion, const void* handle, Kind kind,
                              std::optional<std::int32_t>& original) {
    if (shared.graph_capture) return reject_graph_capture(shared, exclusion);
    auto* context = shared.context();
    auto* graph = shared.find(handle, kind);
    if (!context || !graph || graph->context != context || graph->epoch != context->epoch ||
        graph->is_retiring || graph->is_uncertain)
        return CUDA_ERROR_INVALID_HANDLE;
    FlagPin retiring(graph->is_retiring, exclusion);
    while (graph->active_calls) {
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (!context->is_alive || context->epoch != graph->epoch)
            return CUDA_ERROR_CONTEXT_IS_DESTROYED;
        if (context->is_uncertain) return CUDA_ERROR_UNKNOWN;
    }
    const auto reply =
        call_graph(shared, exclusion, *context,
                   {.action = kind == Kind::kGraph ? ExecutionGraphAction::kDestroyGraph
                                                   : ExecutionGraphAction::kDestroyExecutable,
                    .object = graph->remote});
    original = reply.original;
    if (reply.status == CUDA_SUCCESS && reply.original == 0) graph->is_alive = false;
    if (reply.observation.is_quarantined) graph->is_uncertain = true;
    return reply.status;
}

inline CUresult graph_launch(State& shared, Exclusion& exclusion, CUgraphExec handle,
                             CUstream target) {
    if (shared.graph_capture) return reject_graph_capture(shared, exclusion);
    if (!context_stack.empty() && context_stack.back()->is_uncertain) return CUDA_ERROR_UNKNOWN;
    auto* context = shared.context();
    auto* executable = shared.find(handle, Kind::kGraphExecutable);
    if (!context || !executable || executable->context != context ||
        executable->epoch != context->epoch || executable->is_retiring || executable->is_uncertain)
        return CUDA_ERROR_INVALID_HANDLE;
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, target, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    if (executable->active_calls == kMaxHandles ||
        (stream && stream->active_calls == kMaxHandles) || context->submissions == UINT64_MAX)
        return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin executable_use(executable->active_calls, exclusion);
    std::optional<CounterPin> stream_use;
    if (stream) stream_use.emplace(stream->active_calls, exclusion);
    ExecutionPayloadWriter payload;
    if (!write_execution_graph_launch(payload, {executable->remote, stream ? stream->remote : 0}))
        return CUDA_ERROR_OUT_OF_MEMORY;
    ++context->submissions;
    const auto reply =
        shared.call(exclusion, *context, ExecutionOpcode::kGraphLaunch, std::move(payload));
    if (!reply) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    ExecutionGraphObservation observation;
    ExecutionPayloadReader receipt(reply->data);
    const bool has_observation = !reply->data.empty();
    if (reply->status < 0 || reply->status > 999 || reply->id || reply->address ||
        reply->parameter_bytes || !reply->arguments.empty() ||
        (has_observation &&
         (!read_execution_graph_observation(receipt, observation) || !receipt.empty() ||
          observation.output_written || observation.capture_status)) ||
        (reply->disposition != ExecutionWireDisposition::kComplete &&
         reply->disposition != ExecutionWireDisposition::kRejected) ||
        (!reply->status && (!has_observation || !reply->original_status))) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    if (observation.is_quarantined) context->is_uncertain = true;
    const auto proof = status(reply);
    if (proof != CUDA_SUCCESS || !reply->original_status) return proof;
    if (*reply->original_status < 0 || *reply->original_status > 999) {
        context->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    }
    // An actual Driver issue error is still the first call's result when its
    // independent ownership proof succeeded. Quarantine forbids later issue;
    // it must not erase this already observed original error.
    const auto original = static_cast<CUresult>(*reply->original_status);
    return observation.is_quarantined && original == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : original;
}
