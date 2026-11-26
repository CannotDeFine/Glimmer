// Included inside Implementation: only this serialized service owner can
// reserve construction, submit helper work or admit an ordinary native issue.
static bool decode_graph_memory(ExecutionPayloadReader& input, CudaSessionMemory& memory) noexcept {
    std::uint32_t kind{}, asynchronous{};
    std::uint64_t destination_offset{}, source_offset{}, count{};
    if (!input.read_u32(kind) || kind > static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32) ||
        !input.read_u32(asynchronous) || asynchronous != 1 ||
        !input.read_u64(memory.destination.allocation) || !input.read_u64(destination_offset) ||
        !input.read_u64(memory.source.allocation) || !input.read_u64(source_offset) ||
        !input.read_u64(count) || !input.read_u32(memory.value) || !input.read_u64(memory.stream) ||
        !input.empty() || destination_offset > SIZE_MAX || source_offset > SIZE_MAX ||
        count > SIZE_MAX)
        return false;
    constexpr std::array kinds{CudaMemoryKind::kCopy, CudaMemoryKind::kFill8,
                               CudaMemoryKind::kFill16, CudaMemoryKind::kFill32};
    memory.kind = kinds[kind];
    memory.is_asynchronous = true;
    memory.destination.offset = static_cast<std::size_t>(destination_offset);
    memory.source.offset = static_cast<std::size_t>(source_offset);
    memory.count = static_cast<std::size_t>(count);
    return true;
}

CUresult graph_construction_ready(const Slot& owner) const noexcept {
    const auto helpers = process_memory_native_gate();
    if (helpers != CUDA_SUCCESS) return helpers;
    if (serial_pair || profile_pair || arrival_checkpoint) return CUDA_ERROR_NOT_READY;
    for (const auto& peer : memory_peers) {
        const auto observed = peer.backend->owner()->snapshot();
        if (observed.is_quarantined) return CUDA_ERROR_UNKNOWN;
        if (peer.cleanup || observed.has_native_call) return CUDA_ERROR_NOT_READY;
    }
    for (const auto& slot : slots) {
        if (!slot.session || slot.host_owner || slot.is_metadata_owner() || slot.is_nccl) continue;
        const auto observed = slot.session->scheduling_snapshot();
        if (observed.is_quarantined) return CUDA_ERROR_UNKNOWN;
        if (has_physical_reservation(observed) || has_private_boundaries(observed) ||
            !slot.host_functions.empty())
            return CUDA_ERROR_NOT_READY;
        if (&slot == &owner && (observed.submissions.queued || observed.submissions.paused))
            return CUDA_ERROR_NOT_READY;
    }
    return CUDA_SUCCESS;
}

void observe_graph_construction(Slot& slot) noexcept {
    if (!graph_construction || graph_construction->connection != slot.connection) return;
    const auto observed = slot.session->graph_snapshot();
    if (observed.status != CUDA_SUCCESS || observed.is_unknown) {
        graph_construction->is_unknown = true;
    } else if (!observed.has_capture) {
        // This snapshot is published only after Backend's real native None
        // observation. A successful transport response alone cannot clear it.
        graph_construction.reset();
    }
}

std::optional<ExecutionWireReply> execute_graph_control(Slot& slot, Slot::Request& request) {
    const auto reject = [](CUresult status) {
        return project(CudaSessionGraphResult{.status = status});
    };
    ExecutionPayloadReader input(request.message.payload);
    ExecutionGraphRequest wire;
    if (request.continuation || request.borrowed_descriptor() >= 0 ||
        !read_execution_graph_request(input, wire) || !input.empty())
        return reject(CUDA_ERROR_INVALID_VALUE);
    if (!slot.session || slot.host_owner || slot.is_metadata_owner() || slot.is_nccl)
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    if (graph_construction && graph_construction->connection != slot.connection)
        return std::nullopt;
    if (graph_construction && graph_construction->is_unknown)
        return project(
            CudaSessionGraphResult{.status = CUDA_ERROR_UNKNOWN, .is_quarantined = true});
    using Action = ExecutionGraphAction;
    if (wire.action == Action::kBegin && !graph_construction) {
        if (wire.flags > static_cast<std::uint64_t>(INT_MAX))
            return reject(CUDA_ERROR_INVALID_VALUE);
        // Open and authenticate before the final serialized idle check. An
        // original peer exit remains independently observable after socket loss.
        ExecutionOwnedDescriptor pidfd(
            static_cast<int>(::syscall(SYS_pidfd_open, slot.peer.pid, 0)));
        if (pidfd.get() < 0 || !memory_peer_matches(slot.peer))
            return reject(CUDA_ERROR_INVALID_CONTEXT);
        const auto ready = graph_construction_ready(slot);
        if (ready == CUDA_ERROR_NOT_READY) return std::nullopt;
        if (ready != CUDA_SUCCESS) return reject(ready);
        // No yield/native dispatch/helper submission between proof and lease.
        graph_construction.emplace(GraphConstruction{slot.connection, std::move(pidfd), false});
    }
    CudaSessionGraphResult result;
    switch (wire.action) {
        case Action::kBegin:
            if (wire.flags > static_cast<std::uint64_t>(INT_MAX))
                return reject(CUDA_ERROR_INVALID_VALUE);
            result = slot.session->begin_graph_capture(wire.stream, static_cast<int>(wire.flags),
                                                       wire.caller);
            break;
        case Action::kEnd:
            result = slot.session->end_graph_capture(wire.stream, wire.caller, wire.has_output,
                                                     wire.output_seed);
            break;
        case Action::kState:
            result = slot.session->graph_capture_state(wire.stream, wire.caller);
            break;
        case Action::kInstantiate:
            result = slot.session->instantiate_graph(wire.object, wire.flags, wire.has_output,
                                                     wire.output_seed);
            break;
        case Action::kDestroyGraph:
            result = slot.session->destroy_graph(wire.object);
            break;
        case Action::kDestroyExecutable:
            result = slot.session->destroy_graph_exec(wire.object);
            // No SDK entry means this is a known in-flight lease, not a retry
            // of a destroy whose side effects might already have occurred.
            if (result.status == CUDA_ERROR_NOT_READY && !result.original_status &&
                !result.is_quarantined)
                return std::nullopt;
            break;
        case Action::kCaptureKernel: {
            ExecutionPayloadReader node(wire.payload);
            CudaExecutionKernel kernel;
            bool asynchronous{};
            if (!graph_construction || !decode_kernel_request(node, kernel, asynchronous) ||
                asynchronous || kernel.stream != wire.stream)
                return reject(CUDA_ERROR_INVALID_VALUE);
            result = slot.session->capture_kernel(kernel, wire.caller);
            break;
        }
        case Action::kCaptureMemory: {
            ExecutionPayloadReader node(wire.payload);
            CudaSessionMemory memory;
            if (!graph_construction || !decode_graph_memory(node, memory) ||
                memory.stream != wire.stream)
                return reject(CUDA_ERROR_INVALID_VALUE);
            result = slot.session->capture_memory(memory, wire.caller);
            break;
        }
        case Action::kInvalidate:
            result = slot.session->invalidate_graph_capture(wire.caller);
            break;
    }
    observe_graph_construction(slot);
    return project(result);
}

static bool graph_construction_metadata(ExecutionOpcode operation) noexcept {
    // The exact owner's original code/ABI preparation may run without ordinary
    // queue issue. These old carriers authenticate Session, not logical caller;
    // frontend additionally checks its nonrecycled caller before cold preparation.
    switch (operation) {
        case ExecutionOpcode::kLoadPtx:
        case ExecutionOpcode::kLoadBinary:
        case ExecutionOpcode::kLoadModuleFile:
        case ExecutionOpcode::kLoadModuleJit:
        case ExecutionOpcode::kGetFunction:
        case ExecutionOpcode::kGetGlobal:
        case ExecutionOpcode::kSnapshot:
        case ExecutionOpcode::kCaptureFrontier:
        case ExecutionOpcode::kQueryFrontier:
        case ExecutionOpcode::kReleaseFrontier:
        case ExecutionOpcode::kQueryAsyncFrontiers:
        case ExecutionOpcode::kReleaseAsyncFrontiers:
            return true;
        default:
            return false;
    }
}

bool advance_graph_construction() noexcept {
    if (!graph_construction) return true;
    try {
        const auto owner =
            std::ranges::find(slots, graph_construction->connection, &Slot::connection);
        if (owner == slots.end() || !owner->session) {
            graph_construction->is_unknown = true;
            return true;
        }
        auto& slot = *owner;
        if (graph_construction->is_unknown) return true;
        if (slot.is_retiring || !transport.has_peer(slot.connection)) {
            pollfd descriptor{graph_construction->pidfd.get(), POLLIN, 0};
            // A socket close alone is not an authenticated peer death. Keep the
            // exact capture and its exclusion until the pidfd actually exits.
            if (::poll(&descriptor, 1, 0) != 1 || !(descriptor.revents & POLLIN)) return true;
            const auto aborted = slot.session->abort_graph_capture();
            observe_graph_construction(slot);
            if (graph_construction) {
                if (aborted.is_quarantined) graph_construction->is_unknown = true;
                return true;
            }
            retire(slot);
            return true;
        }
        while (!slot.requests.empty() && graph_construction) {
            auto& request = slot.requests.front();
            const auto operation = request.message.header.operation;
            std::optional<ExecutionWireReply> reply;
            if (request.continuation || request.receipt) {
                reply = rejection(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
            } else if (operation == ExecutionOpcode::kGraphControl) {
                reply = execute_graph_control(slot, request);
            } else if (graph_construction_metadata(operation)) {
                reply = execute_unobserved(slot, request, false, false);
                observe_graph_construction(slot);
            } else {
                const auto capture = slot.session->graph_snapshot();
                const auto invalidated = slot.session->invalidate_graph_capture(capture.caller);
                reply =
                    rejection(invalidated.is_quarantined ? CUDA_ERROR_UNKNOWN
                                                         : CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
                observe_graph_construction(slot);
            }
            if (!reply) break;
            if (!queue_reply(slot, request, *reply)) break;
            slot.requests.erase(slot.requests.begin());
        }
        return true;
    } catch (...) {
        if (graph_construction) graph_construction->is_unknown = true;
        return false;
    }
}
