// Included inside the frontend's private namespace. Process memory uses one
// authenticated carrier whose lifetime is independent of application contexts.
struct ProcessMemoryReply {
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<std::int32_t> original_status{};
    std::uint64_t id{}, address{};
    bool output_written{};
};

inline CUresult open_process_memory(State& shared, Exclusion& exclusion) {
    if (!shared.config) return CUDA_ERROR_NOT_INITIALIZED;
    if (!shared.process_memory_context) {
        auto* entry = shared.allocate_entry(Kind::kProcessMemoryContext, nullptr);
        if (!entry) return CUDA_ERROR_OUT_OF_MEMORY;
        entry->context = entry;
        entry->is_alive = true;
        shared.process_memory_context = entry;
    }
    return shared.open_context(exclusion, *shared.process_memory_context, 0);
}

inline void quarantine_process_memory(State& shared) noexcept {
    if (shared.process_memory_context) shared.process_memory_context->is_uncertain = true;
}

inline bool process_memory_output(void* output, ExecutionProcessMemoryRequest& request) noexcept {
    request.has_output = output != nullptr;
    request.output_seed = 0;
    // Seeded native observation preserves partial and equal-value writes. This
    // explicitly requires readable as well as writable nonnull output storage.
    return !output || (writable_host_range(output, sizeof(request.output_seed)) &&
                       self_copy(&request.output_seed, output, sizeof(request.output_seed), false));
}

inline ProcessMemoryReply call_process_memory(State& shared, Exclusion& exclusion, Entry& target,
                                              const ExecutionProcessMemoryRequest& request) {
    auto& carrier = *shared.process_memory_context;
    if (carrier.is_uncertain || target.is_uncertain) return {};
    if (carrier.active_calls >= kMaxHandles - 2 || carrier.active_mutations >= kMaxHandles - 2)
        return {CUDA_ERROR_OUT_OF_MEMORY};
    ExecutionPayloadWriter payload;
    if (!write_execution_process_memory_request(payload, request))
        return {CUDA_ERROR_OUT_OF_MEMORY};
    CounterPin owner(carrier.active_calls, exclusion);
    CounterPin owner_mutation(carrier.active_mutations, exclusion);
    const auto epoch = target.epoch;
    const auto reply =
        shared.call(exclusion, target, ExecutionOpcode::kProcessMemory, std::move(payload));
    const auto invalid = [&]() -> ProcessMemoryReply {
        target.is_uncertain = true;
        quarantine_process_memory(shared);
        return {};
    };
    if (!reply || !target.is_alive || target.epoch != epoch || target.is_uncertain ||
        carrier.is_uncertain || reply->status < 0 || reply->status > 999 ||
        reply->parameter_bytes > 1 || !reply->arguments.empty())
        return invalid();
    ExecutionProcessMemoryObservation observation;
    if (!reply->data.empty()) {
        ExecutionPayloadReader reader(reply->data);
        if (!read_execution_process_memory_observation(reader, observation) || !reader.empty())
            return invalid();
    } else if (reply->disposition != ExecutionWireDisposition::kRejected) {
        return invalid();
    }
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        if (!reply->status || reply->original_status || reply->id || reply->address ||
            observation.output_written)
            return invalid();
    } else if (reply->disposition == ExecutionWireDisposition::kComplete) {
        const bool is_runtime = request.operation == ExecutionProcessMemoryOperation::kTrimPool;
        if (!reply->original_status ||
            (!is_runtime && (*reply->original_status < 0 || *reply->original_status > 999)) ||
            (!request.has_output && (observation.output_written || reply->id || reply->address)))
            return invalid();
        if (request.operation != ExecutionProcessMemoryOperation::kTrimPool &&
            !reply->parameter_bytes && reply->status != *reply->original_status)
            return invalid();
        if (request.operation == ExecutionProcessMemoryOperation::kTrimPool &&
            !reply->parameter_bytes && reply->status != CUDA_SUCCESS)
            return invalid();
        if (reply->parameter_bytes && reply->status == CUDA_SUCCESS) return invalid();
    } else {
        return invalid();
    }
    if (reply->parameter_bytes) quarantine_process_memory(shared);
    return {static_cast<CUresult>(reply->status), reply->original_status, reply->id, reply->address,
            observation.output_written};
}

inline CUresult process_pool_id(State& shared, CUmemoryPool handle, std::uint64_t& id) noexcept {
    id = 0;
    if (!handle) return CUDA_SUCCESS;
    auto* entry = shared.find(handle, Kind::kMemoryPool);
    if (!entry || entry->context != shared.process_memory_context) return CUDA_ERROR_INVALID_HANDLE;
    if (entry->is_uncertain || !shared.process_memory_context ||
        shared.process_memory_context->is_uncertain)
        return CUDA_ERROR_UNKNOWN;
    id = entry->remote;
    return CUDA_SUCCESS;
}

inline CUresult query_process_pool(State& shared, Exclusion& exclusion, CUmemoryPool* output,
                                   CUdevice device, ExecutionProcessMemoryOperation operation) {
    ExecutionProcessMemoryRequest request;
    request.operation = operation;
    request.device = device;
    if (!process_memory_output(output, request)) return CUDA_ERROR_INVALID_VALUE;
    const auto initialized = ensure_driver_initialization(shared, exclusion);
    if (initialized != CUDA_SUCCESS) return initialized;
    const auto opened = open_process_memory(shared, exclusion);
    if (opened != CUDA_SUCCESS) return opened;
    // Pool queries commonly repeat the same handle. Reserve one reusable,
    // unpublished entry rather than consuming a handle on every observation.
    struct Publication {
        State& state;
        Exclusion& exclusion;
        Entry* entry{};
        ~Publication() {
            if (::getpid() == kProcess && exclusion.owns_lock() && entry && !entry->is_alive &&
                !state.process_memory_pool_spare)
                state.process_memory_pool_spare = entry;
        }
    } publication{shared, exclusion};
    if (output) {
        publication.entry = std::exchange(shared.process_memory_pool_spare, nullptr);
        if (!publication.entry)
            publication.entry =
                shared.allocate_entry(Kind::kMemoryPool, shared.process_memory_context);
        if (!publication.entry) return CUDA_ERROR_OUT_OF_MEMORY;
    }
    auto reply = call_process_memory(shared, exclusion, *shared.process_memory_context, request);
    if (!reply.original_status) return reply.status;
    const bool success = reply.status == CUDA_SUCCESS && *reply.original_status == CUDA_SUCCESS;
    const auto invalid = [&] {
        quarantine_process_memory(shared);
        return CUDA_ERROR_UNKNOWN;
    };
    if ((!output && (success || reply.id || reply.output_written)) ||
        (reply.id && (!reply.output_written || !reply.address ||
                      (reply.id & kExecutionProcessAllocationTag))) ||
        (success && (!reply.id || !reply.output_written || !reply.address)))
        return invalid();
    std::uint64_t bits = reply.address;
    if (reply.id) {
        auto found = std::ranges::find_if(shared.entries, [&](const auto& entry) {
            return entry->kind == Kind::kMemoryPool && entry->is_alive &&
                   entry->context == shared.process_memory_context && entry->remote == reply.id;
        });
        Entry* pool{};
        if (found != shared.entries.end()) {
            pool = found->get();
            if (pool->address != reply.address) return invalid();
        } else {
            if (!success || !publication.entry) return invalid();
            pool = publication.entry;
            pool->remote = reply.id;
            pool->address = reply.address;
            pool->is_alive = true;
        }
        bits = reinterpret_cast<std::uintptr_t>(pool);
    } else if (reply.output_written && reply.address &&
               !shared.process_memory_context->is_uncertain) {
        return invalid();  // Unknown error output may never gain pool authority.
    }
    if (reply.output_written && !self_copy(output, &bits, sizeof(bits), true)) {
        quarantine_process_memory(shared);
        return CUDA_ERROR_INVALID_VALUE;
    }
    return reply.status == CUDA_SUCCESS ? static_cast<CUresult>(*reply.original_status)
                                        : reply.status;
}

inline CUresult set_process_pool(State& shared, Exclusion& exclusion, CUdevice device,
                                 CUmemoryPool pool) {
    const auto initialized = ensure_driver_initialization(shared, exclusion);
    if (initialized != CUDA_SUCCESS) return initialized;
    const auto opened = open_process_memory(shared, exclusion);
    if (opened != CUDA_SUCCESS) return opened;
    ExecutionProcessMemoryRequest request;
    request.operation = ExecutionProcessMemoryOperation::kSetCurrentPool;
    request.device = device;
    const auto selected = process_pool_id(shared, pool, request.pool);
    if (selected != CUDA_SUCCESS) return selected;
    return call_process_memory(shared, exclusion, *shared.process_memory_context, request).status;
}

inline CUresult allocate_process_memory(State& shared, Exclusion& exclusion, CUdeviceptr* output,
                                        std::size_t bytes, CUmemoryPool pool, CUstream handle,
                                        bool from_pool) {
    ExecutionProcessMemoryRequest request;
    request.operation = from_pool ? ExecutionProcessMemoryOperation::kAllocateFromPoolAsync
                                  : ExecutionProcessMemoryOperation::kAllocateAsync;
    request.bytes = bytes;
    if (!process_memory_output(output, request)) return CUDA_ERROR_INVALID_VALUE;
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (context->is_closing) return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    if (context->active_calls >= kMaxHandles - 2 || context->active_mutations >= kMaxHandles - 2)
        return CUDA_ERROR_OUT_OF_MEMORY;
    CounterPin origin(context->active_calls, exclusion);
    CounterPin mutation(context->active_mutations, exclusion);
    const auto opened = open_process_memory(shared, exclusion);
    if (opened != CUDA_SUCCESS) return opened;
    if (from_pool) {
        const auto selected = process_pool_id(shared, pool, request.pool);
        if (selected != CUDA_SUCCESS) return selected;
    }
    Entry* stream{};
    const auto resolved = resolve_owned_stream(shared, exclusion, handle, stream);
    if (resolved != CUDA_SUCCESS) return resolved;
    std::optional<CounterPin> stream_use;
    if (stream) {
        if (stream->active_calls >= kMaxHandles) return CUDA_ERROR_OUT_OF_MEMORY;
        stream_use.emplace(stream->active_calls, exclusion);
        request.stream = stream->remote;
    }
    auto* allocation = bytes && output ? shared.allocate_entry(Kind::kProcessAllocation,
                                                               shared.process_memory_context)
                                       : nullptr;
    if (bytes && output && !allocation) return CUDA_ERROR_OUT_OF_MEMORY;
    if (context->submissions == UINT64_MAX) return CUDA_ERROR_OUT_OF_MEMORY;
    ++context->submissions;
    const auto reply = call_process_memory(shared, exclusion, *context, request);
    if (!reply.original_status) return reply.status;
    const bool success = reply.status == CUDA_SUCCESS && *reply.original_status == CUDA_SUCCESS;
    const auto invalid = [&] {
        quarantine_process_memory(shared);
        if (allocation) allocation->is_uncertain = true;
        return CUDA_ERROR_UNKNOWN;
    };
    if (success) {
        if (!output || !reply.output_written ||
            (bytes ? !execution_is_process_allocation(reply.id) || !reply.address ||
                         bytes - 1 > UINT64_MAX - reply.address
                   : reply.id || reply.address))
            return invalid();
        if (bytes) {
            const auto overlaps = [&](const auto& entry) {
                return entry.get() != allocation && entry->is_alive &&
                       (entry->kind == Kind::kAllocation || entry->kind == Kind::kModuleGlobal ||
                        entry->kind == Kind::kProcessAllocation) &&
                       (reply.address >= entry->address
                            ? reply.address - entry->address < entry->bytes
                            : entry->address - reply.address < bytes);
            };
            while (std::ranges::any_of(shared.entries, [&](const auto& entry) {
                return overlaps(entry) && shared.range_is_retiring(*entry);
            })) {
                if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
                if (context->is_uncertain || shared.process_memory_context->is_uncertain)
                    return invalid();
            }
            if (std::ranges::any_of(shared.entries, overlaps) ||
                std::ranges::any_of(shared.entries, [&](const auto& entry) {
                    return entry.get() != allocation && entry->kind == Kind::kProcessAllocation &&
                           entry->remote == reply.id;
                }))
                return invalid();
            allocation->remote = reply.id;
            allocation->address = reply.address;
            allocation->bytes = bytes;
            allocation->is_alive = true;
        }
    } else if (reply.id || (reply.output_written && reply.address &&
                            !shared.process_memory_context->is_uncertain)) {
        return invalid();
    }
    if (reply.output_written && !self_copy(output, &reply.address, sizeof(reply.address), true)) {
        quarantine_process_memory(shared);
        if (allocation) allocation->is_uncertain = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    return reply.status == CUDA_SUCCESS ? static_cast<CUresult>(*reply.original_status)
                                        : reply.status;
}

inline CUresult release_process_memory(State& shared, Exclusion& exclusion, Entry& allocation) {
    auto* context = shared.context();
    if (!context) return CUDA_ERROR_INVALID_CONTEXT;
    if (allocation.kind != Kind::kProcessAllocation || !allocation.is_alive ||
        allocation.is_retiring || allocation.context != shared.process_memory_context)
        return CUDA_ERROR_INVALID_VALUE;
    if (allocation.is_uncertain || shared.process_memory_context->is_uncertain)
        return CUDA_ERROR_UNKNOWN;
    FlagPin retiring(allocation.is_retiring, exclusion);
    while (allocation.active_calls) {
        if (!yield_frontend(exclusion)) return CUDA_ERROR_INVALID_CONTEXT;
        if (context->is_uncertain || shared.process_memory_context->is_uncertain)
            return CUDA_ERROR_UNKNOWN;
    }
    ExecutionPayloadWriter payload;
    if (!payload.write_u64(allocation.remote)) return CUDA_ERROR_OUT_OF_MEMORY;
    const auto reply =
        shared.call(exclusion, *context, ExecutionOpcode::kRelease, std::move(payload));
    if (!reply || reply->status < 0 || reply->status > 999 || reply->id || reply->address ||
        reply->parameter_bytes > 1 || !reply->arguments.empty() || !reply->data.empty() ||
        (reply->disposition != ExecutionWireDisposition::kComplete &&
         reply->disposition != ExecutionWireDisposition::kRejected) ||
        (reply->disposition == ExecutionWireDisposition::kComplete &&
         (!reply->original_status || *reply->original_status < 0 || *reply->original_status > 999 ||
          (!reply->parameter_bytes && reply->status != *reply->original_status))) ||
        (reply->parameter_bytes && !reply->status) ||
        (reply->disposition == ExecutionWireDisposition::kRejected &&
         (!reply->status || reply->original_status))) {
        allocation.is_uncertain = true;
        quarantine_process_memory(shared);
        return CUDA_ERROR_UNKNOWN;
    }
    if (reply->parameter_bytes ||
        (reply->original_status && *reply->original_status != CUDA_SUCCESS)) {
        allocation.is_uncertain = true;
        quarantine_process_memory(shared);
    }
    if (reply->status == CUDA_SUCCESS) allocation.is_alive = false;
    return static_cast<CUresult>(reply->status);
}
