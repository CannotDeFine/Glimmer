// Metadata keeps its DNM1 identity. Planning borrows authenticated execution
// authority; symbolic pointer slots do not confer host or device access.
namespace {
static_assert(std::endian::native == std::endian::little);
using BackendDirection = ExecutionDnnBackendAttributeDirection;
using BackendOperation = ExecutionDnnBackendAttributeOperation;
using BackendValueClass = ExecutionDnnBackendValueClass;

cudnnStatus_t backend_graph_status(const std::optional<ExecutionWireReply>& reply, Entry& owner,
                                   DnnEntry* entry, bool is_preparation, bool is_create,
                                   bool has_data, Entry* execution = nullptr,
                                   Entry* handle = nullptr) {
    const auto fault_context = [&] {
        if (execution) execution->is_uncertain = true;
        if (handle) handle->is_uncertain = true;
    };
    const auto invalid = [&] {
        fault_context();
        return dnn_invalid(owner, entry ? &entry->is_uncertain : nullptr);
    };
    if (!reply || reply->address || !reply->arguments.empty() ||
        reply->parameter_bytes > (execution ? 3U : 1U))
        return invalid();
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        const auto original = reply->original_status;
        if (reply->id || !reply->data.empty() || reply->status <= 0 || reply->status > 1023 ||
            (original &&
             (!execution || is_preparation || !dnn_profile::status_is_representable(*original) ||
              (!reply->parameter_bytes) ||
              (!(reply->parameter_bytes & 2U) && *original != CUDNN_STATUS_SUCCESS))))
            return invalid();
        if (reply->parameter_bytes & 1U)
            (void)dnn_invalid(owner, entry ? &entry->is_uncertain : nullptr);
        if (reply->parameter_bytes & 2U) fault_context();
        return dnn_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete) return invalid();
    if (is_preparation) {
        if (reply->id || reply->original_status || reply->status != CUDA_SUCCESS) return invalid();
    } else {
        if (!reply->original_status ||
            !dnn_profile::status_is_representable(*reply->original_status) ||
            reply->status != *reply->original_status)
            return invalid();
        if (*reply->original_status != CUDNN_STATUS_SUCCESS) {
            if (reply->id || !reply->data.empty() || (execution && reply->parameter_bytes))
                return invalid();
            // Configuration failure is recoverable and is not an ownership
            // fault. Only an explicit ownership fault closes the metadata owner.
            if (reply->parameter_bytes) (void)invalid();
            return static_cast<cudnnStatus_t>(*reply->original_status);
        }
        if ((is_create && !reply->id) || (!is_create && reply->id)) return invalid();
    }
    if (reply->parameter_bytes || !owner.is_alive || owner.is_uncertain || owner.is_closing ||
        (entry && (entry->is_uncertain || owner.epoch != entry->epoch)) ||
        (execution && (!execution->is_alive || execution->is_uncertain || execution->is_closing)) ||
        (handle && (!handle->is_alive || handle->is_uncertain)) ||
        (!has_data && !reply->data.empty()))
        return invalid();
    return CUDNN_STATUS_SUCCESS;
}

bool backend_requires_context(const DnnEntry& entry) {
    return entry.backend_kind == 2 || entry.backend_kind == 3 || entry.backend_kind == 4 ||
           entry.backend_kind == 5 || entry.backend_kind == 15;
}

bool backend_attribute_requires_context(const DnnEntry& entry, BackendOperation operation,
                                        std::uint32_t name) {
    if (entry.backend_kind == 38) return operation == BackendOperation::kSet && name == 2701;
    if (operation == BackendOperation::kGet &&
        ((entry.backend_kind == 2 && name >= 1303 && name <= 1305) ||
         (entry.backend_kind == 2 && name == 1307) || (entry.backend_kind == 4 && name == 204) ||
         (entry.backend_kind == 5 && (name == 400 || name == 407))))
        return false;
    return backend_requires_context(entry);
}

bool backend_property_frame_required(const std::optional<ExecutionWireReply>& reply) {
    // This exact policy receipt proves that no original Finalize was issued.
    // No native error, post-call fault or ambiguous transport result may retry.
    return reply && reply->disposition == ExecutionWireDisposition::kRejected &&
           reply->status == CUDA_ERROR_INVALID_CONTEXT && !reply->original_status && !reply->id &&
           !reply->address && !reply->parameter_bytes && reply->arguments.empty() &&
           reply->data.empty();
}

bool backend_context_valid(State& shared, Entry* context, std::uint64_t epoch) {
    return !context || (shared.context() == context && context->is_alive && context->is_active &&
                        context->connection && !context->is_uncertain && !context->is_closing &&
                        context->epoch == epoch);
}

struct BackendContextPins {
    std::optional<CounterPin> calls, mutations;
    BackendContextPins(Entry* context, Exclusion& exclusion) {
        if (context) {
            calls.emplace(context->active_calls, exclusion);
            mutations.emplace(context->active_mutations, exclusion);
        }
    }
};

bool backend_context_capacity(const Entry* context) {
    return !context ||
           (context->active_calls < kMaxHandles - 1 && context->active_mutations < kMaxHandles - 1);
}

struct BackendGraphRoute {
    Entry* execution{};
    Entry* handle{};
};

std::optional<ExecutionWireReply> backend_graph_call(State& shared, Exclusion& exclusion,
                                                     Entry& metadata, BackendGraphRoute route,
                                                     ExecutionOpcode operation,
                                                     ExecutionPayloadWriter&& payload) {
    if (!route.execution) return shared.call(exclusion, metadata, operation, std::move(payload));
    ExecutionDnnBackendContext envelope{metadata.connection->session(),
                                        route.handle ? route.handle->remote : 0, operation,
                                        payload.take()};
    ExecutionPayloadWriter output;
    if (!write_execution_dnn_backend_context(output, envelope))
        return local_failure(CUDA_ERROR_OUT_OF_MEMORY);
    return shared.call(exclusion, *route.execution, ExecutionOpcode::kDnnBackendContext,
                       std::move(output));
}

std::uint64_t backend_pointer_slot(State& shared, Entry& owner, std::uintptr_t address) {
    if (!address) return 0;
    const auto found = shared.dnn_backend_pointer_index.find(address);
    if (found != shared.dnn_backend_pointer_index.end()) {
        const auto& slot = shared.dnn_backend_pointer_slots.at(found->second - 1);
        if (slot.owner == &owner && slot.epoch == owner.epoch) return slot.id;
    }
    if (shared.dnn_backend_pointer_slots.size() == kExecutionDnnBackendMaxValueElements) return 0;
    const auto id = static_cast<std::uint64_t>(shared.dnn_backend_pointer_slots.size()) + 1;
    shared.dnn_backend_pointer_slots.push_back({&owner, owner.epoch, id, address});
    shared.dnn_backend_pointer_index.insert_or_assign(address, id);
    return id;
}

bool backend_reference_address(State& shared, Entry& metadata,
                               const ExecutionDnnBackendReference& reference,
                               std::uintptr_t& address) {
    if (reference.kind == ExecutionDnnBackendReferenceKind::kNull) {
        address = 0;
        return true;
    }
    if (reference.kind == ExecutionDnnBackendReferenceKind::kPointerSlot) {
        if (reference.owner != metadata.connection->session() || !reference.id ||
            reference.id > shared.dnn_backend_pointer_slots.size())
            return false;
        const auto& slot = shared.dnn_backend_pointer_slots[reference.id - 1];
        if (slot.owner != &metadata || slot.epoch != metadata.epoch) return false;
        address = slot.address;
        return true;
    }
    if (reference.kind != ExecutionDnnBackendReferenceKind::kHandle) return false;
    const auto found = std::ranges::find_if(shared.entries, [&](const auto& entry) {
        return entry->kind == Kind::kDnn && entry->dnn_backend_owner == reference.owner &&
               entry->remote == reference.id;
    });
    if (found == shared.entries.end()) return false;
    // Deliberately no is_alive check: return the exact old public token, not
    // renewed handle authority. Every later consuming API still checks liveness.
    address = reinterpret_cast<std::uintptr_t>(found->get());
    return true;
}

bool backend_scalar_enums_valid(const ExecutionDnnBackendAttributeValues& value) {
    // Representation envelopes of the installed unfixed SDK enums. This is
    // not a list of supported enumerators; original validation remains native.
    constexpr auto raw = UINT32_MAX;
    constexpr std::array<std::uint32_t, 30> bounds{raw,
                                                   dnn_profile::data_type_enum_max(),
                                                   raw,
                                                   raw,
                                                   raw,
                                                   raw,
                                                   raw,
                                                   1,
                                                   dnn_profile::behavior_note_enum_max(),
                                                   63,
                                                   1,
                                                   15,
                                                   7,
                                                   4095,
                                                   1023,
                                                   raw,
                                                   0,
                                                   1,
                                                   15,
                                                   7,
                                                   3,
                                                   7,
                                                   3,
                                                   raw,
                                                   raw,
                                                   1,
                                                   raw,
                                                   7,
                                                   1,
                                                   3};
    if (value.type >= bounds.size()) return false;
    const auto maximum = bounds[value.type];
    if (maximum == raw) return true;
    ExecutionPayloadReader reader(value.scalars);
    while (!reader.empty()) {
        std::uint32_t bits{};
        if (!reader.read_u32(bits) || bits > maximum) return false;
    }
    return true;
}

bool backend_preparation_matches(const ExecutionDnnBackendPrepare& request,
                                 const ExecutionDnnBackendAttributePreparation& value) {
    const auto& header = request.header;
    if ((!(header.presence & 1U) &&
         (value.input_elements || value.output_elements || value.output_bytes)) ||
        (!(header.presence & 2U) && value.count_bytes))
        return false;
    const auto value_class = execution_dnn_backend_value_class(header.type);
    if (request.operation == BackendOperation::kSet)
        return value.count_bytes == 0 && value.output_elements == 0 &&
               (value.direction == BackendDirection::kNone ||
                (value.direction == BackendDirection::kInput &&
                 (value_class == BackendValueClass::kScalar ||
                  value_class == BackendValueClass::kDescriptor ||
                  value_class == BackendValueClass::kHandle ||
                  value_class == BackendValueClass::kPointerSlot)));
    switch (value.direction) {
        case BackendDirection::kNone:
            return true;
        case BackendDirection::kInput:
        case BackendDirection::kDescriptorOutput:
            return value_class == BackendValueClass::kDescriptor;
        case BackendDirection::kScalarOutput: {
            const auto width = execution_dnn_backend_scalar_width(header.type);
            return value_class == BackendValueClass::kScalar && width &&
                   value.output_elements ==
                       value.output_bytes / width + (value.output_bytes % width != 0);
        }
        case BackendDirection::kHandleOutput:
            return value_class == BackendValueClass::kHandle;
        case BackendDirection::kPointerOutput:
            return value_class == BackendValueClass::kPointerSlot;
    }
    return false;
}

// All vectors are allocated before acquisition. No partial pin/flag survives
// a yield, including when the source also appears in a destination position.
struct BackendGraphPins {
    std::vector<std::shared_ptr<DnnEntry>> entries;
    std::vector<bool> exclusive;
    std::vector<Entry*> handles;
    Exclusion& exclusion;
    BackendGraphPins(std::vector<std::shared_ptr<DnnEntry>> values, std::vector<bool> mutations,
                     Exclusion& lock, std::vector<Entry*> native_handles = {})
        : entries(std::move(values)),
          exclusive(std::move(mutations)),
          handles(std::move(native_handles)),
          exclusion(lock) {
        for (std::size_t i = 0; i != entries.size(); ++i) {
            ++entries[i]->active_calls;
            if (exclusive[i]) entries[i]->is_retiring = true;
        }
        for (auto* handle : handles) ++handle->active_calls;
    }
    BackendGraphPins(const BackendGraphPins&) = delete;
    BackendGraphPins& operator=(const BackendGraphPins&) = delete;
    ~BackendGraphPins() {
        if (::getpid() != kProcess || !exclusion.owns_lock()) return;
        for (std::size_t i = 0; i != entries.size(); ++i) {
            --entries[i]->active_calls;
            if (exclusive[i]) entries[i]->is_retiring = false;
        }
        for (auto* handle : handles) --handle->active_calls;
    }
};

cudnnStatus_t backend_graph_lifecycle(cudnnBackendDescriptor_t descriptor,
                                      ExecutionOpcode operation) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        const auto entry = find_dnn_metadata(shared, descriptor, ExecutionDnnObjectKind::kBackend);
        if (!entry || !valid_dnn_metadata(*entry)) return CUDNN_STATUS_NOT_INITIALIZED;
        auto& owner = *entry->context;
        const bool needs_context =
            operation == ExecutionOpcode::kDnnBackendFinalize && backend_requires_context(*entry);
        auto* context = needs_context ? shared.context() : nullptr;
        if (needs_context && !context) return CUDNN_STATUS_NOT_INITIALIZED;
        const auto epoch = context ? context->epoch : 0;
        if (owner.active_calls >= kMaxHandles - 1 || owner.active_mutations >= kMaxHandles - 1)
            return dnn_profile::host_allocation_status();
        CounterPin origin(owner.active_calls, exclusion);
        CounterPin work(owner.active_mutations, exclusion);
        for (;;) {
            if (!valid_dnn_metadata(*entry) || !backend_context_valid(shared, context, epoch))
                return CUDNN_STATUS_NOT_INITIALIZED;
            if (!entry->is_retiring && !entry->active_calls) break;
            // Do not hold an object or handle flag while another caller finishes.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        if (!backend_context_capacity(context)) return dnn_profile::host_allocation_status();
        FlagPin retiring(entry->is_retiring, exclusion);
        CounterPin use(entry->active_calls, exclusion);
        BackendContextPins context_pins(context, exclusion);
        ExecutionPayloadWriter payload;
        if (!payload.write_u64(entry->remote)) return dnn_profile::host_allocation_status();
        auto reply = backend_graph_call(shared, exclusion, owner, {context, nullptr}, operation,
                                        std::move(payload));
        if (entry->backend_kind == 38 && operation == ExecutionOpcode::kDnnBackendFinalize &&
            backend_property_frame_required(reply)) {
            // Keep the same exclusive descriptor acquisition across both calls.
            // CPU-populated properties finish above even after origin teardown;
            // only the server's generation-bound state may request discovery.
            context = shared.context();
            const auto retry_epoch = context ? context->epoch : 0;
            if (!context || !valid_dnn_metadata(*entry) ||
                !backend_context_valid(shared, context, retry_epoch))
                return CUDNN_STATUS_NOT_INITIALIZED;
            if (!backend_context_capacity(context)) return dnn_profile::host_allocation_status();
            BackendContextPins retry_pins(context, exclusion);
            ExecutionPayloadWriter retry;
            if (!retry.write_u64(entry->remote)) return dnn_profile::host_allocation_status();
            reply = backend_graph_call(shared, exclusion, owner, {context, nullptr}, operation,
                                       std::move(retry));
        }
        const auto status =
            backend_graph_status(reply, owner, entry.get(), false, false, false, context);
        if (status == CUDNN_STATUS_SUCCESS && operation == ExecutionOpcode::kDnnBackendDestroy) {
            entry->is_alive = false;
            std::erase(shared.dnn_entries, entry);
        }
        return status;
    });
}

cudnnStatus_t backend_graph_attribute(cudnnBackendDescriptor_t descriptor,
                                      ExecutionDnnBackendAttributeHeader header,
                                      BackendOperation operation, const void* array,
                                      std::int64_t* count_output) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        if (!dnn_profile::supports_backend_attribute(header.name))
            return CUDNN_STATUS_NOT_SUPPORTED;
        auto source = find_dnn_metadata(shared, descriptor, ExecutionDnnObjectKind::kBackend);
        if (!source || !valid_dnn_metadata(*source)) return CUDNN_STATUS_NOT_INITIALIZED;
        auto& owner = *source->context;
        const auto epoch = owner.epoch;
        // Audited CPU copies/observations remain usable after origin teardown.
        // HANDLE Set still borrows a live execution owner, never a wire pointer.
        const bool needs_context =
            backend_attribute_requires_context(*source, operation, header.name);
        auto* context = needs_context ? shared.context() : nullptr;
        if (needs_context && !context) return CUDNN_STATUS_NOT_INITIALIZED;
        const auto context_epoch = context ? context->epoch : 0;
        if (owner.active_calls >= kMaxHandles - 1 || owner.active_mutations >= kMaxHandles - 1)
            return dnn_profile::host_allocation_status();
        CounterPin origin(owner.active_calls, exclusion);
        CounterPin work(owner.active_mutations, exclusion);
        header.descriptor = source->remote;
        const ExecutionDnnBackendPrepare request{operation, header};
        ExecutionDnnBackendAttributePreparation preparation;
        {
            while (source->is_retiring) {
                if (!yield_frontend(exclusion) || !valid_dnn_metadata(*source) ||
                    !backend_context_valid(shared, context, context_epoch))
                    return CUDNN_STATUS_NOT_INITIALIZED;
            }
            if (!backend_context_valid(shared, context, context_epoch))
                return CUDNN_STATUS_NOT_INITIALIZED;
            if (source->active_calls >= kMaxHandles - 1 || !backend_context_capacity(context))
                return dnn_profile::host_allocation_status();
            CounterPin use(source->active_calls, exclusion);
            BackendContextPins context_pins(context, exclusion);
            ExecutionPayloadWriter payload;
            if (!write_execution_dnn_backend_prepare(payload, request))
                return CUDNN_STATUS_BAD_PARAM;
            const auto reply =
                backend_graph_call(shared, exclusion, owner, {context, nullptr},
                                   ExecutionOpcode::kDnnBackendPrepare, std::move(payload));
            const auto status =
                backend_graph_status(reply, owner, source.get(), true, false, true, context);
            if (status != CUDNN_STATUS_SUCCESS) return status;
            ExecutionPayloadReader reader(reply->data);
            if (!read_execution_dnn_backend_preparation(reader, preparation) || !reader.empty() ||
                !backend_preparation_matches(request, preparation))
                return backend_graph_status(std::nullopt, owner, source.get(), true, false, true,
                                            context);
        }

        ExecutionDnnBackendAttribute command{header, preparation.generation, {header.type, {}, {}}};
        std::vector<cudnnBackendDescriptor_t> handles;
        std::vector<cudnnHandle_t> dnn_handles;
        std::vector<std::uintptr_t> addresses;
        const auto value_class = execution_dnn_backend_value_class(header.type);
        if (preparation.input_elements) {
            if (value_class == BackendValueClass::kScalar) {
                const auto width = execution_dnn_backend_scalar_width(header.type);
                command.values.scalars.resize(preparation.input_elements * width);
                if (!self_copy(command.values.scalars.data(), array, command.values.scalars.size(),
                               false) ||
                    !backend_scalar_enums_valid(command.values))
                    return CUDNN_STATUS_BAD_PARAM;
            } else if (value_class == BackendValueClass::kDescriptor) {
                handles.resize(preparation.input_elements);
                if (!self_copy(handles.data(), array, handles.size() * sizeof(handles[0]), false))
                    return CUDNN_STATUS_BAD_PARAM;
            } else if (value_class == BackendValueClass::kHandle) {
                dnn_handles.resize(preparation.input_elements);
                const auto bytes = std::as_writable_bytes(std::span{dnn_handles});
                if (!self_copy(bytes.data(), array, bytes.size(), false))
                    return CUDNN_STATUS_BAD_PARAM;
            } else if (value_class == BackendValueClass::kPointerSlot) {
                static_assert(sizeof(void*) == sizeof(std::uintptr_t));
                addresses.resize(preparation.input_elements);
                if (!self_copy(addresses.data(), array, addresses.size() * sizeof(addresses[0]),
                               false))
                    return CUDNN_STATUS_BAD_PARAM;
            } else {
                return dnn_invalid(owner, &source->is_uncertain);
            }
        }
        std::vector<std::shared_ptr<DnnEntry>> unique;
        std::vector<bool> exclusive;
        std::vector<Entry*> unique_handles;
        const bool has_reference_output =
            preparation.direction == BackendDirection::kHandleOutput ||
            preparation.direction == BackendDirection::kPointerOutput;
        std::vector<std::uintptr_t> output_addresses(
            has_reference_output ? preparation.output_elements : 0);
        command.values.references.resize(handles.size() + dnn_handles.size() + addresses.size());
        for (std::size_t i = 0; i != addresses.size(); ++i) {
            const auto id = backend_pointer_slot(shared, owner, addresses[i]);
            if (addresses[i] && !id) return dnn_profile::host_allocation_status();
            command.values.references[i] =
                id ? ExecutionDnnBackendReference{ExecutionDnnBackendReferenceKind::kPointerSlot,
                                                  owner.connection->session(), id}
                   : ExecutionDnnBackendReference{};
        }
        for (;;) {
            if (!owner.is_alive || owner.is_uncertain || owner.is_closing || owner.epoch != epoch ||
                !backend_context_valid(shared, context, context_epoch))
                return CUDNN_STATUS_NOT_INITIALIZED;
            source = find_dnn_metadata(shared, descriptor, ExecutionDnnObjectKind::kBackend);
            if (!source || source->context != &owner || !valid_dnn_metadata(*source) ||
                source->remote != header.descriptor)
                return CUDNN_STATUS_NOT_INITIALIZED;
            unique.clear();
            exclusive.clear();
            unique_handles.clear();
            unique.push_back(source);
            exclusive.push_back(operation == BackendOperation::kSet);
            for (std::size_t i = 0; i != handles.size(); ++i) {
                auto& reference = command.values.references[i];
                if (!handles[i]) {
                    reference = {};
                    continue;
                }
                const auto entry =
                    find_dnn_metadata(shared, handles[i], ExecutionDnnObjectKind::kBackend);
                if (!entry || entry->context != &owner || !valid_dnn_metadata(*entry))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                reference = {ExecutionDnnBackendReferenceKind::kDescriptor,
                             owner.connection->session(), entry->remote};
                const auto found = std::ranges::find(unique, entry);
                if (found == unique.end()) {
                    unique.push_back(entry);
                    exclusive.push_back(preparation.direction ==
                                        BackendDirection::kDescriptorOutput);
                } else if (preparation.direction == BackendDirection::kDescriptorOutput) {
                    exclusive[static_cast<std::size_t>(found - unique.begin())] = true;
                }
            }
            for (std::size_t i = 0; i != dnn_handles.size(); ++i) {
                auto& reference = command.values.references[i];
                if (!dnn_handles[i]) {
                    reference = {};
                    continue;
                }
                auto* handle = shared.find(dnn_handles[i], Kind::kDnn);
                if (!context || !handle || handle->context != context || handle->is_uncertain)
                    return CUDNN_STATUS_NOT_INITIALIZED;
                reference = {ExecutionDnnBackendReferenceKind::kHandle,
                             context->connection->session(), handle->remote};
                handle->dnn_backend_owner = reference.owner;
                if (std::ranges::find(unique_handles, handle) == unique_handles.end())
                    unique_handles.push_back(handle);
            }
            bool is_waiting{};
            for (std::size_t i = 0; i != unique.size(); ++i) {
                if (unique[i]->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |= unique[i]->is_retiring || (exclusive[i] && unique[i]->active_calls);
            }
            for (auto* handle : unique_handles) {
                if (handle->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |= handle->is_retiring;
            }
            if (!is_waiting) break;
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        if (!backend_context_capacity(context)) return dnn_profile::host_allocation_status();
        auto* native_handle = unique_handles.empty() ? nullptr : unique_handles.front();
        BackendGraphPins pins(std::move(unique), std::move(exclusive), exclusion,
                              std::move(unique_handles));
        BackendContextPins context_pins(context, exclusion);
        ExecutionPayloadWriter payload;
        if (!write_execution_dnn_backend_attribute(payload, command)) return CUDNN_STATUS_BAD_PARAM;
        const auto opcode = operation == BackendOperation::kSet ? ExecutionOpcode::kDnnBackendSet
                                                                : ExecutionOpcode::kDnnBackendGet;
        const auto reply = backend_graph_call(shared, exclusion, owner, {context, native_handle},
                                              opcode, std::move(payload));
        const auto status =
            backend_graph_status(reply, owner, source.get(), false, false,
                                 operation == BackendOperation::kGet, context, native_handle);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        if (std::ranges::any_of(pins.entries,
                                [](const auto& entry) { return !valid_dnn_metadata(*entry); }))
            return backend_graph_status(std::nullopt, owner, source.get(), false, false, true,
                                        context, native_handle);
        if (operation == BackendOperation::kSet) return CUDNN_STATUS_SUCCESS;

        ExecutionDnnBackendGetResult result;
        ExecutionPayloadReader reader(reply->data);
        const auto expected_bytes =
            has_reference_output
                ? 0
                : preparation.output_elements * execution_dnn_backend_scalar_width(header.type);
        if (!read_execution_dnn_backend_get_result(reader, result) || !reader.empty() ||
            result.values.type != header.type || result.count_bytes != preparation.count_bytes ||
            result.values.references.size() != output_addresses.size() ||
            result.values.scalars.size() != expected_bytes ||
            std::ranges::any_of(std::span{result.values.scalars}.subspan(preparation.output_bytes),
                                [](std::byte value) { return value != std::byte{}; }) ||
            !backend_scalar_enums_valid(result.values))
            return backend_graph_status(std::nullopt, owner, source.get(), false, false, true,
                                        context, native_handle);
        for (std::size_t i = 0; i != output_addresses.size(); ++i)
            if (!backend_reference_address(shared, owner, result.values.references[i],
                                           output_addresses[i]))
                return backend_graph_status(std::nullopt, owner, source.get(), false, false, true,
                                            context, native_handle);
        // Validate both spans before the first store. Scalar output precedes
        // count output in these native rows, including intentionally aliased spans.
        const auto output_bytes = has_reference_output
                                      ? output_addresses.size() * sizeof(output_addresses[0])
                                      : preparation.output_bytes;
        if ((output_bytes && !writable_host_range(const_cast<void*>(array), output_bytes)) ||
            (result.count_bytes && !writable_host_range(count_output, result.count_bytes)))
            return CUDNN_STATUS_BAD_PARAM;
        if (output_bytes &&
            !self_copy(const_cast<void*>(array),
                       has_reference_output ? static_cast<const void*>(output_addresses.data())
                                            : result.values.scalars.data(),
                       output_bytes, true))
            return CUDNN_STATUS_BAD_PARAM;
        if (result.count_bytes &&
            !self_copy(count_output, &result.count_bits, result.count_bytes, true))
            return CUDNN_STATUS_BAD_PARAM;
        return CUDNN_STATUS_SUCCESS;
    });
}

cudnnStatus_t backend_execute_status(const std::optional<ExecutionWireReply>& reply, Entry& handle,
                                     Entry& metadata,
                                     const std::array<std::shared_ptr<DnnEntry>, 2>& descriptors) {
    auto& context = *handle.context;
    const auto metadata_fault = [&] {
        metadata.is_uncertain = true;
        for (const auto& descriptor : descriptors) descriptor->is_uncertain = true;
    };
    const auto context_fault = [&] {
        context.is_uncertain = true;
        handle.is_uncertain = true;
    };
    const auto invalid = [&] {
        metadata_fault();
        context_fault();
        return CUDNN_STATUS_INTERNAL_ERROR;
    };
    if (!reply || reply->address || !reply->data.empty() || !reply->arguments.empty() ||
        reply->parameter_bytes > 3)
        return invalid();
    const auto original = reply->original_status;
    const auto flags = reply->parameter_bytes;
    if (reply->disposition == ExecutionWireDisposition::kRejected) {
        if (reply->status <= 0 || reply->status > 1023 ||
            (original &&
             (!reply->id || !dnn_profile::status_is_representable(*original) || !(flags & 2U))))
            return invalid();
        if (flags & 1U) metadata_fault();
        if (flags & 2U) context_fault();
        return dnn_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete || !reply->id || !original ||
        !dnn_profile::status_is_representable(*original) || reply->status != *original || flags)
        return invalid();
    if (*original != CUDNN_STATUS_SUCCESS) return static_cast<cudnnStatus_t>(*original);
    if (!handle.is_alive || handle.is_uncertain || !context.is_alive || context.is_uncertain ||
        !metadata.is_alive || metadata.is_uncertain ||
        std::ranges::any_of(
            descriptors, [](const auto& descriptor) { return !valid_dnn_metadata(*descriptor); }))
        return invalid();
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t submit_backend_graph(cudnnHandle_t handle, cudnnBackendDescriptor_t plan,
                                   cudnnBackendDescriptor_t variant) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        auto& context = *entry.context;
        const auto epoch = context.epoch;
        auto* metadata = shared.dnn_metadata_context;
        if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
            metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
        if (metadata->active_calls >= kMaxHandles - 1 || context.submissions == UINT64_MAX)
            return dnn_profile::host_allocation_status();
        CounterPin metadata_pin(metadata->active_calls, exclusion);
        const std::array handles{plan, variant};
        constexpr std::array<std::uint32_t, 2> kinds{5, 16};
        std::array<std::shared_ptr<DnnEntry>, 2> descriptors;
        const auto healthy = [&] {
            return entry.is_alive && !entry.is_uncertain &&
                   backend_context_valid(shared, &context, epoch) && metadata->is_alive &&
                   !metadata->is_uncertain && !metadata->is_closing;
        };
        for (;;) {
            if (!healthy()) return CUDNN_STATUS_NOT_INITIALIZED;
            bool waiting{};
            for (std::size_t i = 0; i != descriptors.size(); ++i) {
                auto& descriptor = descriptors[i];
                descriptor =
                    find_dnn_metadata(shared, handles[i], ExecutionDnnObjectKind::kBackend);
                if (!descriptor || descriptor->context != metadata ||
                    descriptor->backend_kind != kinds[i] || !valid_dnn_metadata(*descriptor))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (descriptor->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                waiting |= descriptor->is_retiring;
            }
            if (!waiting) break;
            // Neither descriptor is pinned while a previous mutation completes.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        CounterPin plan_pin(descriptors[0]->active_calls, exclusion);
        CounterPin variant_pin(descriptors[1]->active_calls, exclusion);
        const ExecutionDnnBackendExecutePrepare request{
            entry.remote, metadata->connection->session(), descriptors[0]->remote,
            descriptors[1]->remote};
        ExecutionPayloadWriter preparation_payload;
        if (!write_execution_dnn_backend_execute_prepare(preparation_payload, request))
            return dnn_profile::host_allocation_status();
        const auto prepared_reply =
            shared.call(exclusion, context, ExecutionOpcode::kPrepareDnnBackendExecute,
                        std::move(preparation_payload));
        const auto prepared_status = backend_graph_status(
            prepared_reply, *metadata, descriptors[0].get(), true, false, true, &context, &entry);
        if (prepared_status != CUDNN_STATUS_SUCCESS) return prepared_status;
        ExecutionDnnBackendExecutePreparation prepared;
        ExecutionPayloadReader reader(prepared_reply->data);
        if (!read_execution_dnn_backend_execute_preparation(reader, prepared) || !reader.empty() ||
            !healthy() || std::ranges::any_of(descriptors, [](const auto& descriptor) {
                return !valid_dnn_metadata(*descriptor);
            }))
            return backend_execute_status(std::nullopt, entry, *metadata, descriptors);

        ExecutionDnnBackendExecute command{request.handle,
                                           request.metadata,
                                           request.plan,
                                           request.variant,
                                           prepared.plan_generation,
                                           prepared.variant_generation,
                                           {}};
        command.bindings.reserve(prepared.requirements.size());
        std::array<std::optional<CounterPin>, kExecutionDnnBackendMaxBindings> allocation_pins,
            module_pins;
        for (std::size_t i = 0; i != prepared.requirements.size(); ++i) {
            const auto& requirement = prepared.requirements[i];
            ExecutionDnnBackendExecuteBinding binding{
                requirement.index, requirement.value_class, requirement.slot, 0, 0, {}};
            std::uintptr_t address{};
            if (requirement.slot &&
                !backend_reference_address(shared, *metadata,
                                           {ExecutionDnnBackendReferenceKind::kPointerSlot,
                                            metadata->connection->session(), requirement.slot},
                                           address))
                return backend_execute_status(std::nullopt, entry, *metadata, descriptors);
            // NULL stays NULL even for an active tensor with a nonzero extent.
            // Ignored entries preserve ordered slot identity without a read/lease.
            if (address && requirement.value_class != ExecutionDnnBackendBindingClass::kIgnored) {
                if (requirement.bytes > SIZE_MAX || address % requirement.alignment ||
                    requirement.bytes > UINTPTR_MAX - address)
                    return CUDNN_STATUS_BAD_PARAM;
                const auto bytes = static_cast<std::size_t>(requirement.bytes);
                if (requirement.value_class == ExecutionDnnBackendBindingClass::kByValue) {
                    binding.host_bytes.resize(bytes);
                    // NOLINTNEXTLINE(performance-no-int-to-ptr): checked original local address.
                    const auto* pointer = reinterpret_cast<const void*>(address);
                    if (bytes && !self_copy(binding.host_bytes.data(), pointer, bytes, false))
                        return CUDNN_STATUS_BAD_PARAM;
                } else {
                    auto* allocation = shared.range(static_cast<CUdeviceptr>(address), bytes);
                    if (!allocation || allocation->context != &context ||
                        allocation->is_uncertain || shared.range_is_retiring(*allocation))
                        return CUDNN_STATUS_BAD_PARAM;
                    if (allocation->active_calls >= kMaxHandles - 1)
                        return dnn_profile::host_allocation_status();
                    allocation_pins[i].emplace(allocation->active_calls, exclusion);
                    if (allocation->kind == Kind::kModuleGlobal) {
                        const auto module =
                            std::ranges::find_if(shared.entries, [&](const auto& value) {
                                return value->kind == Kind::kModule && value->context == &context &&
                                       value->remote == allocation->parent && value->is_alive &&
                                       !value->is_uncertain && !value->is_retiring;
                            });
                        if (module == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                        if ((*module)->active_calls >= kMaxHandles - 1)
                            return dnn_profile::host_allocation_status();
                        module_pins[i].emplace((*module)->active_calls, exclusion);
                    }
                    binding.allocation = allocation->remote;
                    binding.offset = static_cast<std::uint64_t>(address) - allocation->address;
                }
            }
            command.bindings.push_back(std::move(binding));
        }
        if (!healthy() || std::ranges::any_of(descriptors, [](const auto& descriptor) {
                return !valid_dnn_metadata(*descriptor);
            }))
            return CUDNN_STATUS_NOT_INITIALIZED;
        ExecutionPayloadWriter payload;
        if (!write_execution_dnn_backend_execute(payload, command))
            return dnn_profile::host_allocation_status();
        ++context.submissions;
        return backend_execute_status(
            shared.call(exclusion, context, ExecutionOpcode::kDnnBackendExecute,
                        std::move(payload)),
            entry, *metadata, descriptors);
    });
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t kind, cudnnBackendDescriptor_t* output) {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        if (!dnn_profile::supports_backend_kind(dnn_enum_bits(kind)))
            return CUDNN_STATUS_NOT_SUPPORTED;
        if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
        const auto opened = open_dnn_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return dnn_error(opened);
        auto& owner = *shared.dnn_metadata_context;
        if (!shared.has_dnn_tokens || shared.dnn_entries.size() == kMaxHandles ||
            owner.active_calls >= kMaxHandles - 1 || owner.active_mutations >= kMaxHandles - 1)
            return dnn_profile::host_allocation_status();
        auto entry = std::make_shared<DnnEntry>();
        entry->context = &owner;
        entry->epoch = owner.epoch;
        entry->kind = ExecutionDnnObjectKind::kBackend;
        entry->backend_kind = dnn_enum_bits(kind);
        entry->token = shared.next_dnn_token;
        if (shared.next_dnn_token == UINTPTR_MAX)
            shared.has_dnn_tokens = false;
        else
            ++shared.next_dnn_token;
        ExecutionPayloadWriter payload;
        if (!write_execution_dnn_backend_create(payload, {dnn_enum_bits(kind), 1}))
            return dnn_profile::host_allocation_status();
        shared.dnn_entries.push_back(entry);
        CounterPin origin(owner.active_calls, exclusion);
        CounterPin work(owner.active_mutations, exclusion);
        const auto reply =
            shared.call(exclusion, owner, ExecutionOpcode::kDnnBackendCreate, std::move(payload));
        const auto status = backend_graph_status(reply, owner, entry.get(), false, true, false);
        if (status != CUDNN_STATUS_SUCCESS) {
            if (!owner.is_uncertain) std::erase(shared.dnn_entries, entry);
            return status;
        }
        if (std::ranges::any_of(shared.dnn_entries, [&](const auto& candidate) {
                return candidate != entry && candidate->context == &owner && candidate->is_alive &&
                       candidate->remote == reply->id;
            }))
            return dnn_invalid(owner, &entry->is_uncertain);
        entry->remote = reply->id;
        entry->is_alive = true;
        const auto token =
            // NOLINTNEXTLINE(performance-no-int-to-ptr): ABI token, never dereferenced.
            reinterpret_cast<cudnnBackendDescriptor_t>(static_cast<std::uintptr_t>(entry->token));
        return publish(output, token) ? CUDNN_STATUS_SUCCESS
                                      : dnn_invalid(owner, &entry->is_uncertain);
    });
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor) {
    return backend_graph_lifecycle(descriptor, ExecutionOpcode::kDnnBackendDestroy);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendInitialize(cudnnBackendDescriptor_t descriptor) {
    return backend_graph_lifecycle(descriptor, ExecutionOpcode::kDnnBackendInitialize);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
    return backend_graph_lifecycle(descriptor, ExecutionOpcode::kDnnBackendFinalize);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                                                              cudnnBackendAttributeName_t name,
                                                              cudnnBackendAttributeType_t type,
                                                              std::int64_t count,
                                                              const void* array) {
    return backend_graph_attribute(
        descriptor, {0, dnn_enum_bits(name), dnn_enum_bits(type), count, array ? 1U : 0U},
        BackendOperation::kSet, array, nullptr);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendGetAttribute(cudnnBackendDescriptor_t descriptor,
                                                              cudnnBackendAttributeName_t name,
                                                              cudnnBackendAttributeType_t type,
                                                              std::int64_t requested,
                                                              std::int64_t* count, void* array) {
    return backend_graph_attribute(descriptor,
                                   {0, dnn_enum_bits(name), dnn_enum_bits(type), requested,
                                    (array ? 1U : 0U) | (count ? 2U : 0U)},
                                   BackendOperation::kGet, array, count);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendExecute(cudnnHandle_t handle,
                                                         cudnnBackendDescriptor_t plan,
                                                         cudnnBackendDescriptor_t variant) {
    return submit_backend_graph(handle, plan, variant);
}
#if CUDNN_MAJOR == 9
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendPopulateCudaGraph(cudnnHandle_t,
                                                                   cudnnBackendDescriptor_t,
                                                                   cudnnBackendDescriptor_t,
                                                                   cudaGraph_t) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnBackendUpdateCudaGraph(cudnnHandle_t,
                                                                 cudnnBackendDescriptor_t,
                                                                 cudnnBackendDescriptor_t,
                                                                 cudaGraph_t) {
    return dnn_unsupported();
}
#endif
// NOLINTEND(bugprone-easily-swappable-parameters)
