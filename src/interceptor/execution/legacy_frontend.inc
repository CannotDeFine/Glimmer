// SDK8 legacy ABIs only. No caller-native SDK executes in this translation unit.
#include <type_traits>

namespace {
static_assert(CUDNN_MAJOR == 8);
static_assert(sizeof(cudnnAlgorithm_t) == sizeof(std::uint32_t));
static_assert(offsetof(cudnnAlgorithm_t, algo) == 0);
static_assert(std::is_trivially_copyable_v<cudnnAlgorithm_t>);

class LegacyDescriptors final {
   public:
    bool add(const void* handle, ExecutionDnnObjectKind kind, bool is_mutating = false) {
        if (!handle) return true;
        const auto [found, inserted] = entries_.try_emplace(handle);
        auto& entry = found->second;
        if (inserted) entry.kind = kind;
        if (entry.kind != kind || entries_.size() > kMaxHandles) return false;
        entry.is_mutating |= is_mutating;
        return true;
    }
    std::uint64_t remote(const void* handle) const {
        return handle ? entries_.at(handle).entry->remote : 0;
    }
    bool healthy() const {
        return std::ranges::all_of(entries_, [](const auto& item) {
            return item.second.entry && valid_dnn_metadata(*item.second.entry);
        });
    }
    template <typename Function>
    cudnnStatus_t use(State& shared, Exclusion& exclusion, Entry& handle, Function&& function) {
        auto* metadata = shared.dnn_metadata_context;
        if (!metadata || !metadata->connection || !metadata->is_alive || metadata->is_uncertain ||
            metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
        const bool is_mutating =
            std::ranges::any_of(entries_, [](const auto& item) { return item.second.is_mutating; });
        if (metadata->active_calls >= kMaxHandles - 1 ||
            (is_mutating && metadata->active_mutations >= kMaxHandles - 1))
            return dnn_profile::host_allocation_status();
        CounterPin owner(metadata->active_calls, exclusion);
        std::optional<CounterPin> work;
        if (is_mutating) work.emplace(metadata->active_mutations, exclusion);
        auto& context = *handle.context;
        const auto epoch = context.epoch;
        for (;;) {
            if (!handle.is_alive || handle.is_uncertain || !context.is_alive ||
                context.is_uncertain || context.is_closing || context.epoch != epoch ||
                !metadata->is_alive || metadata->is_uncertain || metadata->is_closing)
                return CUDNN_STATUS_NOT_INITIALIZED;
            bool is_waiting{};
            for (auto& [token, value] : entries_) {
                value.entry = find_dnn_metadata(shared, token, value.kind);
                if (!value.entry || value.entry->context != metadata ||
                    !valid_dnn_metadata(*value.entry))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (value.entry->active_calls >= kMaxHandles - 1)
                    return dnn_profile::host_allocation_status();
                is_waiting |=
                    value.entry->is_retiring || (value.is_mutating && value.entry->active_calls);
            }
            if (!is_waiting) break;
            // No partial descriptor pins or mutation flags while yielding.
            // Input/output aliases share a single exclusive reservation.
            if (!yield_frontend(exclusion)) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        for (auto& [token, value] : entries_) {
            static_cast<void>(token);
            if (value.is_mutating) value.mutation.emplace(value.entry->is_retiring, exclusion);
            value.pin.emplace(value.entry->active_calls, exclusion);
        }
        return function(*metadata);
    }

   private:
    struct Reference {
        ExecutionDnnObjectKind kind{};
        bool is_mutating{};
        std::shared_ptr<DnnEntry> entry;
        std::optional<FlagPin> mutation;
        std::optional<CounterPin> pin;
    };
    std::unordered_map<const void*, Reference> entries_;
};

struct LegacyReceipt {
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    bool has_original{};
    ExecutionDnnLegacyOutput output{};
};

LegacyReceipt legacy_receipt(const std::optional<ExecutionWireReply>& reply, Entry& metadata,
                             Entry* handle = nullptr, bool is_compute = false) {
    const auto fault = [&](std::uint64_t flags) {
        if (flags & 1U) metadata.is_uncertain = true;
        if ((flags & 2U) && handle) handle->is_uncertain = handle->context->is_uncertain = true;
    };
    const auto invalid = [&] {
        fault(handle ? 3 : 1);
        return LegacyReceipt{};
    };
    if (!reply || reply->address || !reply->arguments.empty() ||
        reply->parameter_bytes > (handle ? 3U : 1U) || (!is_compute && reply->id))
        return invalid();
    const auto original = reply->original_status;
    if (original && !dnn_profile::status_is_representable(*original)) return invalid();
    LegacyReceipt result;
    result.has_original = original.has_value();
    if (original && !is_compute) {
        ExecutionPayloadReader input(reply->data);
        if (!read_execution_dnn_legacy_output(input, result.output) || !input.empty())
            return invalid();
    } else if (!reply->data.empty()) {
        return invalid();
    }
    if (reply->disposition == ExecutionWireDisposition::kComplete) {
        if (!original || reply->status != *original || (is_compute && !reply->id)) return invalid();
        result.status = static_cast<cudnnStatus_t>(*original);
        // A quarantined successful SDK call cannot establish public success.
        if (reply->parameter_bytes && result.status == CUDNN_STATUS_SUCCESS)
            result.status = CUDNN_STATUS_INTERNAL_ERROR;
    } else if (reply->disposition == ExecutionWireDisposition::kRejected) {
        constexpr auto driver_max =
            std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U;
        if (reply->status <= 0 || static_cast<std::uint32_t>(reply->status) > driver_max ||
            (original && (!reply->parameter_bytes || (is_compute && !reply->id))))
            return invalid();
        result.status = dnn_error(static_cast<CUresult>(reply->status));
    } else {
        return invalid();
    }
    fault(reply->parameter_bytes);
    if (result.status == CUDNN_STATUS_SUCCESS &&
        (!metadata.is_alive || metadata.is_uncertain || metadata.is_closing ||
         (handle && (!handle->is_alive || handle->is_uncertain || !handle->context->is_alive ||
                     handle->context->is_uncertain || handle->context->is_closing))))
        return invalid();
    return result;
}

bool legacy_scalar_output(const ExecutionDnnLegacyOutput& output, bool has_output,
                          std::size_t bytes) {
    return output.written <= 1 && (!output.written || has_output) && !output.non_null &&
           !output.offset && !output.extent && (!output.scalar || output.written) &&
           (bytes == sizeof(std::uint64_t) || output.scalar <= UINT32_MAX);
}
bool legacy_empty_output(const ExecutionDnnLegacyOutput& output) {
    return !output.written && !output.scalar && !output.non_null && !output.offset &&
           !output.extent;
}

template <typename Scalar>
bool legacy_seed(Scalar* output, std::uint64_t& seed) {
    static_assert(sizeof(Scalar) == 4 || sizeof(Scalar) == 8);
    seed = 0;
    return !output || (writable_host_range(output, sizeof(*output)) &&
                       self_copy(&seed, output, sizeof(*output), false));
}
cudnnStatus_t legacy_metadata_call(State& shared, Exclusion& exclusion, Entry& owner,
                                   ExecutionDnnLegacyMetadata request, void* output = nullptr) {
    ExecutionPayloadWriter payload;
    if (!write_execution_dnn_legacy_metadata(payload, request))
        return dnn_profile::host_allocation_status();
    const auto reply =
        shared.call(exclusion, owner, ExecutionOpcode::kDnnLegacyMetadata, std::move(payload));
    const auto receipt = legacy_receipt(reply, owner);
    const bool is_getter = request.kind == ExecutionDnnLegacyMetadataKind::kGetMath ||
                           request.kind == ExecutionDnnLegacyMetadataKind::kGetBias;
    if (receipt.has_original && !is_getter &&
        request.kind != ExecutionDnnLegacyMetadataKind::kVersionCheck &&
        *reply->original_status != CUDNN_STATUS_SUCCESS && !(reply->parameter_bytes & 1U))
        return dnn_invalid(owner);
    if (receipt.has_original &&
        !(is_getter ? legacy_scalar_output(receipt.output, request.has_output != 0, 4)
                    : legacy_empty_output(receipt.output)))
        return dnn_invalid(owner);
    if (receipt.output.written) {
        // Copy enum storage as bytes; do not evaluate an out-of-domain enum.
        if (!output || !writable_host_range(output, sizeof(std::uint32_t)) ||
            !self_copy(output, &receipt.output.scalar, sizeof(std::uint32_t), true))
            return CUDNN_STATUS_BAD_PARAM;
    }
    return receipt.status;
}
cudnnStatus_t legacy_version(ExecutionDnnLegacyVersionCheck check) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        const auto opened = open_dnn_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return dnn_error(opened);
        auto& owner = *shared.dnn_metadata_context;
        if (owner.active_calls >= kMaxHandles - 1) return dnn_profile::host_allocation_status();
        CounterPin pin(owner.active_calls, exclusion);
        return legacy_metadata_call(
            shared, exclusion, owner,
            {ExecutionDnnLegacyMetadataKind::kVersionCheck, 0, static_cast<std::uint32_t>(check)});
    });
}
template <typename Scalar>
cudnnStatus_t legacy_metadata(const void* descriptor, ExecutionDnnObjectKind object_kind,
                              ExecutionDnnLegacyMetadataKind kind, std::uint32_t value,
                              Scalar* output, bool is_mutating) noexcept {
    return dnn_metadata(
        descriptor, object_kind, is_mutating,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            std::uint64_t seed{};
            if (!legacy_seed(output, seed)) return CUDNN_STATUS_BAD_PARAM;
            return legacy_metadata_call(
                shared, exclusion, *entry.context,
                {kind, entry.remote, value, output ? 1U : 0U, static_cast<std::uint32_t>(seed)},
                output);
        });
}

using LegacyArrays = std::array<std::vector<cudnnTensorDescriptor_t>, 3>;
bool legacy_arrays(int count, const std::array<const cudnnTensorDescriptor_t*, 3>& pointers,
                   LegacyArrays& snapshots, std::size_t fixed_bytes) {
    if (count > static_cast<int>(kExecutionDnnMaxSequenceLengths)) return false;
    const auto length = count > 0 ? static_cast<std::size_t>(count) : 0;
    std::size_t total{};
    for (std::size_t index = 0; index != pointers.size(); ++index) {
        if (!pointers[index]) continue;
        // Exact fixed wire fields plus all captured ID arrays share the normal
        // transport limit; this introduces no smaller per-message policy cap.
        if (length > (kExecutionMaxPayloadBytes - fixed_bytes) / sizeof(std::uint64_t) - total)
            return false;
        total += length;
        snapshots[index].resize(length);
        if (!self_copy(snapshots[index].data(), pointers[index],
                       length * sizeof(cudnnTensorDescriptor_t), false))
            return false;
    }
    return true;
}
bool legacy_add_arrays(LegacyDescriptors& descriptors, const LegacyArrays& arrays) {
    for (const auto& array : arrays)
        for (const auto descriptor : array)
            if (!descriptors.add(descriptor, ExecutionDnnObjectKind::kTensor)) return false;
    return true;
}
void legacy_encode_array(ExecutionDnnLegacyDescriptorArray& output,
                         const cudnnTensorDescriptor_t* source,
                         const std::vector<cudnnTensorDescriptor_t>& snapshot,
                         const LegacyDescriptors& descriptors) {
    output.has_descriptors = source ? 1U : 0U;
    output.descriptors.reserve(snapshot.size());
    for (const auto descriptor : snapshot)
        output.descriptors.push_back(descriptors.remote(descriptor));
}

class LegacyBuffers final {
   public:
    cudnnStatus_t capture(State& shared, Exclusion& exclusion, Entry& context,
                          const std::array<const void*, 13>& pointers, std::size_t workspace_bytes,
                          std::size_t reserve_bytes) {
        for (std::size_t index = 0; index != pointers.size(); ++index) {
            const auto address = reinterpret_cast<CUdeviceptr>(pointers[index]);
            if (!address) continue;
            const auto bytes = index == 11 ? workspace_bytes : index == 12 ? reserve_bytes : 1;
            auto* allocation = shared.range(address, bytes);
            if (!allocation || allocation->context != &context || allocation->is_uncertain ||
                shared.range_is_retiring(*allocation))
                return CUDNN_STATUS_BAD_PARAM;
            if (allocation->active_calls >= kMaxHandles - pointers.size())
                return dnn_profile::host_allocation_status();
            owners_[index] = allocation;
            pins_[index].emplace(allocation->active_calls, exclusion);
            if (allocation->kind == Kind::kModuleGlobal) {
                const auto parent =
                    std::ranges::find_if(shared.entries, [&](const auto& candidate) {
                        return candidate->kind == Kind::kModule && candidate->context == &context &&
                               candidate->remote == allocation->parent && candidate->is_alive &&
                               !candidate->is_retiring && !candidate->is_uncertain;
                    });
                if (parent == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                if ((*parent)->active_calls >= kMaxHandles - pointers.size())
                    return dnn_profile::host_allocation_status();
                modules_[index] = parent->get();
                module_pins_[index].emplace((*parent)->active_calls, exclusion);
            }
            values[index] = {allocation->remote, address - allocation->address};
            available[index] = allocation->bytes - (address - allocation->address);
        }
        return CUDNN_STATUS_SUCCESS;
    }
    bool healthy() const {
        const auto valid = [](const Entry* entry) {
            return !entry || (entry->is_alive && !entry->is_uncertain);
        };
        return std::ranges::all_of(owners_, valid) && std::ranges::all_of(modules_, valid);
    }
    std::array<ExecutionDnnBuffer, 13> values{};
    std::array<std::uint64_t, 13> available{};

   private:
    std::array<Entry*, 13> owners_{}, modules_{};
    std::array<std::optional<CounterPin>, 13> pins_, module_pins_;
};
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI cudnnOpsInferVersionCheck() {
    return legacy_version(ExecutionDnnLegacyVersionCheck::kOpsInfer);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnOpsTrainVersionCheck() {
    return legacy_version(ExecutionDnnLegacyVersionCheck::kOpsTrain);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCnnInferVersionCheck() {
    return legacy_version(ExecutionDnnLegacyVersionCheck::kCnnInfer);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCnnTrainVersionCheck() {
    return legacy_version(ExecutionDnnLegacyVersionCheck::kCnnTrain);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnAdvInferVersionCheck() {
    return legacy_version(ExecutionDnnLegacyVersionCheck::kAdvInfer);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnAdvTrainVersionCheck() {
    return legacy_version(ExecutionDnnLegacyVersionCheck::kAdvTrain);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnCreateAlgorithmDescriptor(cudnnAlgorithmDescriptor_t* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kAlgorithm, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnDestroyAlgorithmDescriptor(cudnnAlgorithmDescriptor_t descriptor) {
    return destroy_dnn_metadata(descriptor, ExecutionDnnObjectKind::kAlgorithm, true);
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnSetAlgorithmDescriptor(cudnnAlgorithmDescriptor_t descriptor, cudnnAlgorithm_t algorithm) {
    std::uint32_t bits{};
    std::memcpy(&bits, &algorithm, sizeof(bits));
    return legacy_metadata(descriptor, ExecutionDnnObjectKind::kAlgorithm,
                           ExecutionDnnLegacyMetadataKind::kSetAlgorithm, bits,
                           static_cast<std::uint32_t*>(nullptr), true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNMatrixMathType(cudnnRNNDescriptor_t descriptor,
                                                               cudnnMathType_t type) {
    return legacy_metadata(descriptor, ExecutionDnnObjectKind::kRnn,
                           ExecutionDnnLegacyMetadataKind::kSetMath, dnn_enum_bits(type),
                           static_cast<std::uint32_t*>(nullptr), true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNMatrixMathType(cudnnRNNDescriptor_t descriptor,
                                                               cudnnMathType_t* type) {
    return legacy_metadata(descriptor, ExecutionDnnObjectKind::kRnn,
                           ExecutionDnnLegacyMetadataKind::kGetMath, 0, type, false);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNBiasMode(cudnnRNNDescriptor_t descriptor,
                                                         cudnnRNNBiasMode_t mode) {
    return legacy_metadata(descriptor, ExecutionDnnObjectKind::kRnn,
                           ExecutionDnnLegacyMetadataKind::kSetBias, dnn_enum_bits(mode),
                           static_cast<std::uint32_t*>(nullptr), true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNBiasMode(cudnnRNNDescriptor_t descriptor,
                                                         cudnnRNNBiasMode_t* mode) {
    return legacy_metadata(descriptor, ExecutionDnnObjectKind::kRnn,
                           ExecutionDnnLegacyMetadataKind::kGetBias, 0, mode, false);
}

namespace {
struct LegacyControlInputs {
    const void* rnn{};
    const void* dropout{};
    const void* input{};
    const void* filter{};
    const void* destination{};
    const void* algorithm{};
    const cudnnTensorDescriptor_t* sequence{};
    const void* weights{};
    void* output{};
};
cudnnStatus_t legacy_control(cudnnHandle_t handle, ExecutionDnnLegacyControl request,
                             const LegacyControlInputs& inputs) noexcept {
    const auto kind = request.kind;
    const bool is_pointer = kind == ExecutionDnnLegacyControlKind::kMatrixParameter ||
                            kind == ExecutionDnnLegacyControlKind::kBiasParameter;
    const bool is_scalar = kind == ExecutionDnnLegacyControlKind::kWorkspaceSize ||
                           kind == ExecutionDnnLegacyControlKind::kReserveSize ||
                           kind == ExecutionDnnLegacyControlKind::kParameterSize ||
                           kind == ExecutionDnnLegacyControlKind::kInferenceMaxCount;
    const bool is_mutating = is_pointer || kind == ExecutionDnnLegacyControlKind::kSetup ||
                             kind == ExecutionDnnLegacyControlKind::kClip ||
                             kind == ExecutionDnnLegacyControlKind::kSetAlgorithm;
    return dnn_handle(
        handle, false, is_mutating, [&](State& shared, Exclusion& exclusion, Entry& entry) {
            if (!inputs.rnn) return CUDNN_STATUS_NOT_INITIALIZED;
            const auto output_bytes = kind == ExecutionDnnLegacyControlKind::kInferenceMaxCount
                                          ? sizeof(std::int32_t)
                                          : sizeof(std::uint64_t);
            request.has_output = inputs.output ? 1U : 0U;
            if (inputs.output &&
                (!writable_host_range(inputs.output, output_bytes) ||
                 (is_scalar && !self_copy(&request.seed, inputs.output, output_bytes, false))))
                return CUDNN_STATUS_BAD_PARAM;
            LegacyArrays arrays;
            // 40-byte common control + signed count + presence/blob lengths.
            constexpr std::size_t fixed_array_bytes = 40 + 4 + 4 + 4;
            if (!legacy_arrays(request.sequence_length, {inputs.sequence, nullptr, nullptr}, arrays,
                               fixed_array_bytes))
                return CUDNN_STATUS_BAD_PARAM;
            LegacyDescriptors descriptors;
            if (!descriptors.add(inputs.rnn, ExecutionDnnObjectKind::kRnn,
                                 is_mutating && !is_pointer) ||
                !descriptors.add(inputs.dropout, ExecutionDnnObjectKind::kDropout) ||
                !descriptors.add(inputs.input, ExecutionDnnObjectKind::kTensor) ||
                !descriptors.add(inputs.filter, ExecutionDnnObjectKind::kFilter) ||
                !descriptors.add(inputs.destination, ExecutionDnnObjectKind::kFilter, true) ||
                !descriptors.add(inputs.algorithm, ExecutionDnnObjectKind::kAlgorithm) ||
                !legacy_add_arrays(descriptors, arrays))
                return CUDNN_STATUS_BAD_PARAM;
            return descriptors.use(shared, exclusion, entry, [&](Entry& metadata) {
                request.handle = entry.remote;
                request.metadata = metadata.connection->session();
                request.descriptor = descriptors.remote(inputs.rnn);
                request.setup.dropout = descriptors.remote(inputs.dropout);
                request.input = descriptors.remote(inputs.input);
                request.algorithm = descriptors.remote(inputs.algorithm);
                legacy_encode_array(request.sequence, inputs.sequence, arrays[0], descriptors);
                LegacyBuffers buffers;
                if (is_pointer) {
                    request.parameters.input = request.input;
                    request.input = 0;
                    request.parameters.filter = descriptors.remote(inputs.filter);
                    request.parameters.destination = descriptors.remote(inputs.destination);
                    std::array<const void*, 13> pointers{};
                    pointers[10] = inputs.weights;
                    const auto captured =
                        buffers.capture(shared, exclusion, *entry.context, pointers, 0, 0);
                    if (captured != CUDNN_STATUS_SUCCESS) return captured;
                    request.parameters.weights = buffers.values[10];
                }
                ExecutionPayloadWriter payload;
                if (!write_execution_dnn_legacy_control(payload, request))
                    return dnn_profile::host_allocation_status();
                const auto reply =
                    shared.call(exclusion, *entry.context, ExecutionOpcode::kDnnLegacyControl,
                                std::move(payload));
                const auto receipt = legacy_receipt(reply, metadata, &entry);
                const auto invalid = [&] {
                    metadata.is_uncertain = entry.is_uncertain = entry.context->is_uncertain = true;
                    return CUDNN_STATUS_INTERNAL_ERROR;
                };
                const auto& output = receipt.output;
                void* mapped{};
                if (receipt.has_original) {
                    if (is_mutating && *reply->original_status != CUDNN_STATUS_SUCCESS &&
                        !(reply->parameter_bytes & 1U))
                        return invalid();
                    if (is_pointer) {
                        if ((output.written != 0 && output.written != 2) || output.scalar ||
                            (output.written && !inputs.output) || output.non_null > 1 ||
                            (output.non_null && output.written != 2) ||
                            (!output.non_null && (output.offset || output.extent)))
                            return invalid();
                        if (output.non_null) {
                            const auto address = reinterpret_cast<std::uintptr_t>(inputs.weights);
                            if (!address || !inputs.destination || !output.extent ||
                                output.offset > buffers.available[10] ||
                                output.extent > buffers.available[10] - output.offset ||
                                output.offset > UINTPTR_MAX - address || !buffers.healthy())
                                return invalid();
                            // NOLINTNEXTLINE(performance-no-int-to-ptr): proved owned subregion.
                            mapped = reinterpret_cast<void*>(address + output.offset);
                        }
                    } else if (is_scalar) {
                        if (!legacy_scalar_output(output, inputs.output != nullptr, output_bytes))
                            return invalid();
                    } else if (!legacy_empty_output(output)) {
                        return invalid();
                    }
                }
                if (receipt.status == CUDNN_STATUS_SUCCESS &&
                    (!descriptors.healthy() || !buffers.healthy() || !entry.is_alive ||
                     entry.is_uncertain || !entry.context->is_alive || entry.context->is_uncertain))
                    return invalid();
                // Validate the complete receipt and mapping before any write. Actual
                // original failure/post-proof loss can still have a known scalar or
                // bounded pointer side effect; returning its error must not erase it.
                if (output.written) {
                    if (!writable_host_range(inputs.output, output_bytes))
                        return CUDNN_STATUS_BAD_PARAM;
                    const void* source = is_pointer ? static_cast<const void*>(&mapped)
                                                    : static_cast<const void*>(&output.scalar);
                    if (!self_copy(inputs.output, source, output_bytes, true))
                        return CUDNN_STATUS_BAD_PARAM;
                }
                return receipt.status;
            });
        });
}

cudnnStatus_t legacy_compute(cudnnHandle_t handle, const void* rnn,
                             ExecutionDnnLegacyComputeKind kind, int sequence_length,
                             const std::array<const cudnnTensorDescriptor_t*, 3>& sequences,
                             const std::array<const void*, 6>& states, const void* filter,
                             const std::array<const void*, 13>& pointers,
                             std::size_t workspace_bytes, std::size_t reserve_bytes) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!rnn) return CUDNN_STATUS_NOT_INITIALIZED;
        LegacyArrays arrays;
        // Common32 + three array headers24 + states48 + filter8 + buffers208
        // + raw workspace/reserve sizes16, matching the fieldwise codec.
        constexpr std::size_t fixed_compute_bytes = 32 + 24 + 48 + 8 + 208 + 16;
        if (!legacy_arrays(sequence_length, sequences, arrays, fixed_compute_bytes))
            return CUDNN_STATUS_BAD_PARAM;
        LegacyDescriptors descriptors;
        if (!descriptors.add(rnn, ExecutionDnnObjectKind::kRnn) ||
            !descriptors.add(filter, ExecutionDnnObjectKind::kFilter) ||
            !legacy_add_arrays(descriptors, arrays))
            return CUDNN_STATUS_BAD_PARAM;
        for (const auto state : states)
            if (!descriptors.add(state, ExecutionDnnObjectKind::kTensor))
                return CUDNN_STATUS_BAD_PARAM;
        return descriptors.use(shared, exclusion, entry, [&](Entry& metadata) {
            auto& context = *entry.context;
            if (context.submissions == UINT64_MAX) return dnn_profile::host_allocation_status();
            ExecutionDnnLegacyCompute request;
            request.kind = kind;
            request.handle = entry.remote;
            request.metadata = metadata.connection->session();
            request.descriptor = descriptors.remote(rnn);
            request.sequence_length = sequence_length;
            for (std::size_t index = 0; index != sequences.size(); ++index)
                legacy_encode_array(request.sequences[index], sequences[index], arrays[index],
                                    descriptors);
            for (std::size_t index = 0; index != states.size(); ++index)
                request.states[index] = descriptors.remote(states[index]);
            request.filter = descriptors.remote(filter);
            request.workspace_bytes = workspace_bytes;
            request.reserve_bytes = reserve_bytes;
            LegacyBuffers buffers;
            const auto captured = buffers.capture(shared, exclusion, context, pointers,
                                                  workspace_bytes, reserve_bytes);
            if (captured != CUDNN_STATUS_SUCCESS) return captured;
            request.buffers = buffers.values;
            ExecutionPayloadWriter payload;
            if (!write_execution_dnn_legacy_compute(payload, request))
                return dnn_profile::host_allocation_status();
            ++context.submissions;
            const auto receipt =
                legacy_receipt(shared.call(exclusion, context, ExecutionOpcode::kDnnLegacyCompute,
                                           std::move(payload)),
                               metadata, &entry, true);
            if (receipt.status == CUDNN_STATUS_SUCCESS &&
                (!descriptors.healthy() || !buffers.healthy() || !entry.is_alive ||
                 entry.is_uncertain || !context.is_alive || context.is_uncertain)) {
                metadata.is_uncertain = entry.is_uncertain = context.is_uncertain = true;
                return CUDNN_STATUS_INTERNAL_ERROR;
            }
            return receipt.status;
        });
    });
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed legacy SDK ABI.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNDescriptor_v6(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int hidden_size, int layers,
    cudnnDropoutDescriptor_t dropout, cudnnRNNInputMode_t input_mode,
    cudnnDirectionMode_t direction, cudnnRNNMode_t cell_mode, cudnnRNNAlgo_t algorithm,
    cudnnDataType_t math_precision) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kSetup;
    request.setup = {hidden_size,
                     layers,
                     0,
                     dnn_enum_bits(input_mode),
                     dnn_enum_bits(direction),
                     dnn_enum_bits(cell_mode),
                     dnn_enum_bits(algorithm),
                     dnn_enum_bits(math_precision)};
    return legacy_control(handle, request, {.rnn = descriptor, .dropout = dropout});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNSetClip(cudnnHandle_t handle,
                                                     cudnnRNNDescriptor_t descriptor,
                                                     cudnnRNNClipMode_t mode,
                                                     cudnnNanPropagation_t nan, double lower,
                                                     double upper) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kClip;
    request.clip = {dnn_enum_bits(mode), dnn_enum_bits(nan), std::bit_cast<std::uint64_t>(lower),
                    std::bit_cast<std::uint64_t>(upper)};
    return legacy_control(handle, request, {.rnn = descriptor});
}
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnGetRNNWorkspaceSize(cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
                         const cudnnTensorDescriptor_t* sequence, std::size_t* bytes) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kWorkspaceSize;
    request.sequence_length = sequence_length;
    return legacy_control(handle, request,
                          {.rnn = descriptor, .sequence = sequence, .output = bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNTrainingReserveSize(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* sequence, std::size_t* bytes) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kReserveSize;
    request.sequence_length = sequence_length;
    return legacy_control(handle, request,
                          {.rnn = descriptor, .sequence = sequence, .output = bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNParamsSize(cudnnHandle_t handle,
                                                           cudnnRNNDescriptor_t descriptor,
                                                           cudnnTensorDescriptor_t input,
                                                           std::size_t* bytes,
                                                           cudnnDataType_t type) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kParameterSize;
    request.data_type = dnn_enum_bits(type);
    return legacy_control(handle, request, {.rnn = descriptor, .input = input, .output = bytes});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerMatrixParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t input, cudnnFilterDescriptor_t filter, const void* weights,
    int linear_layer, cudnnFilterDescriptor_t destination, void** output) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kMatrixParameter;
    request.parameters.pseudo_layer = pseudo_layer;
    request.parameters.linear_layer = linear_layer;
    return legacy_control(handle, request,
                          {.rnn = descriptor,
                           .input = input,
                           .filter = filter,
                           .destination = destination,
                           .weights = weights,
                           .output = output});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNLinLayerBiasParams(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int pseudo_layer,
    cudnnTensorDescriptor_t input, cudnnFilterDescriptor_t filter, const void* weights,
    int linear_layer, cudnnFilterDescriptor_t destination, void** output) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kBiasParameter;
    request.parameters.pseudo_layer = pseudo_layer;
    request.parameters.linear_layer = linear_layer;
    return legacy_control(handle, request,
                          {.rnn = descriptor,
                           .input = input,
                           .filter = filter,
                           .destination = destination,
                           .weights = weights,
                           .output = output});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetRNNAlgorithmDescriptor(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, cudnnAlgorithmDescriptor_t algorithm) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kSetAlgorithm;
    return legacy_control(handle, request, {.rnn = descriptor, .algorithm = algorithm});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetRNNForwardInferenceAlgorithmMaxCount(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int* count) {
    ExecutionDnnLegacyControl request;
    request.kind = ExecutionDnnLegacyControlKind::kInferenceMaxCount;
    return legacy_control(handle, request, {.rnn = descriptor, .output = count});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNForwardTraining(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* x_descriptors, const void* x,
    cudnnTensorDescriptor_t hx_descriptor, const void* hx, cudnnTensorDescriptor_t cx_descriptor,
    const void* cx, cudnnFilterDescriptor_t filter, const void* weights,
    const cudnnTensorDescriptor_t* y_descriptors, void* y, cudnnTensorDescriptor_t hy_descriptor,
    void* hy, cudnnTensorDescriptor_t cy_descriptor, void* cy, void* workspace,
    std::size_t workspace_bytes, void* reserve, std::size_t reserve_bytes) {
    return legacy_compute(
        handle, descriptor, ExecutionDnnLegacyComputeKind::kForwardTraining, sequence_length,
        {x_descriptors, y_descriptors, nullptr},
        {hx_descriptor, cx_descriptor, hy_descriptor, cy_descriptor, nullptr, nullptr}, filter,
        {x, y, nullptr, nullptr, hx, cx, hy, cy, nullptr, nullptr, weights, workspace, reserve},
        workspace_bytes, reserve_bytes);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardData(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* y_descriptors, const void* y,
    const cudnnTensorDescriptor_t* dy_descriptors, const void* dy,
    cudnnTensorDescriptor_t dhy_descriptor, const void* dhy, cudnnTensorDescriptor_t dcy_descriptor,
    const void* dcy, cudnnFilterDescriptor_t filter, const void* weights,
    cudnnTensorDescriptor_t hx_descriptor, const void* hx, cudnnTensorDescriptor_t cx_descriptor,
    const void* cx, const cudnnTensorDescriptor_t* dx_descriptors, void* dx,
    cudnnTensorDescriptor_t dhx_descriptor, void* dhx, cudnnTensorDescriptor_t dcx_descriptor,
    void* dcx, void* workspace, std::size_t workspace_bytes, void* reserve,
    std::size_t reserve_bytes) {
    return legacy_compute(
        handle, descriptor, ExecutionDnnLegacyComputeKind::kBackwardData, sequence_length,
        {y_descriptors, dy_descriptors, dx_descriptors},
        {hx_descriptor, cx_descriptor, dhy_descriptor, dcy_descriptor, dhx_descriptor,
         dcx_descriptor},
        filter, {nullptr, y, dy, dx, hx, cx, dhy, dcy, dhx, dcx, weights, workspace, reserve},
        workspace_bytes, reserve_bytes);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnRNNBackwardWeights(
    cudnnHandle_t handle, cudnnRNNDescriptor_t descriptor, int sequence_length,
    const cudnnTensorDescriptor_t* x_descriptors, const void* x,
    cudnnTensorDescriptor_t hx_descriptor, const void* hx,
    const cudnnTensorDescriptor_t* y_descriptors, const void* y, const void* workspace,
    std::size_t workspace_bytes, cudnnFilterDescriptor_t filter, void* weights, const void* reserve,
    std::size_t reserve_bytes) {
    return legacy_compute(handle, descriptor, ExecutionDnnLegacyComputeKind::kBackwardWeights,
                          sequence_length, {x_descriptors, y_descriptors, nullptr},
                          {hx_descriptor, nullptr, nullptr, nullptr, nullptr, nullptr}, filter,
                          {x, y, nullptr, nullptr, hx, nullptr, nullptr, nullptr, nullptr, nullptr,
                           weights, workspace, reserve},
                          workspace_bytes, reserve_bytes);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
