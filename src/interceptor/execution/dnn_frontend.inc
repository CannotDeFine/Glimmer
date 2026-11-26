// Included only by the optional owned execution frontend after its common
// ownership/transport helpers. No vendor library is called in this process.
namespace {
namespace dnn_profile = glimmer::internal::cudnn_profile;
using DnnActivationHandle = cudnnActivationStruct*;
static_assert(CUDNN_DIM_MAX == kExecutionDnnMaxDimensions);
static_assert(sizeof(int) == sizeof(std::int32_t));

cudnnStatus_t dnn_error(CUresult error) noexcept {
    switch (error) {
        case CUDA_ERROR_OUT_OF_MEMORY:
            return dnn_profile::host_allocation_status();
        case CUDA_ERROR_INVALID_VALUE:
            return CUDNN_STATUS_BAD_PARAM;
        case CUDA_ERROR_NOT_SUPPORTED:
            return CUDNN_STATUS_NOT_SUPPORTED;
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_DEINITIALIZED:
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_INVALID_HANDLE:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return CUDNN_STATUS_NOT_INITIALIZED;
        default:
            return CUDNN_STATUS_INTERNAL_ERROR;
    }
}
cudnnStatus_t dnn_invalid(Entry& context, bool* uncertain = nullptr) noexcept {
    context.is_uncertain = true;
    if (uncertain) *uncertain = true;
    return CUDNN_STATUS_INTERNAL_ERROR;
}
cudnnStatus_t dnn_status(const std::optional<ExecutionWireReply>& reply, Entry& context,
                         bool* uncertain = nullptr) noexcept {
    if (!reply || reply->parameter_bytes > 1) return dnn_invalid(context, uncertain);
    if (reply->parameter_bytes) (void)dnn_invalid(context, uncertain);
    if (reply->disposition == ExecutionWireDisposition::kComplete && reply->original_status &&
        reply->status == *reply->original_status) {
        // Validate against this build's SDK enum domain before conversion.
        const auto value = *reply->original_status;
        if (!dnn_profile::status_is_representable(value)) return dnn_invalid(context, uncertain);
        return static_cast<cudnnStatus_t>(value);
    }
    // CUDA_ERROR_UNKNOWN is the largest enumerator in the supported Driver SDK.
    // Check the unfixed enum's representable range before converting wire bits.
    constexpr auto driver_max =
        std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U;
    if (reply->disposition != ExecutionWireDisposition::kRejected || reply->status <= 0 ||
        static_cast<std::uint32_t>(reply->status) > driver_max)
        return dnn_invalid(context, uncertain);
    return dnn_error(static_cast<CUresult>(reply->status));
}
template <typename Function>
cudnnStatus_t dnn_boundary(Function&& function) noexcept {
    if (::getpid() != kProcess) return CUDNN_STATUS_NOT_INITIALIZED;
    if (is_entered) return dnn_profile::profile_mismatch_status();
    if (!glimmer::interceptor::execution::dnn_application_profile_matches())
        return dnn_profile::profile_mismatch_status();
    Entered entered;
    try {
        auto& shared = state();
        Exclusion exclusion(shared.mutex);
        if (!shared.config) return CUDNN_STATUS_NOT_INITIALIZED;
        return function(shared, exclusion);
    } catch (const std::bad_alloc&) {
        return dnn_profile::host_allocation_status();
    } catch (...) {
        return CUDNN_STATUS_INTERNAL_ERROR;
    }
}
cudnnStatus_t dnn_unsupported() noexcept {
    // A direct guard must not leak a cuDNN 9-only error to a cuDNN 8 caller.
    // Never acquire loader exclusion from reentry or a forked resource owner.
    if (::getpid() != kProcess || is_entered ||
        !glimmer::interceptor::execution::dnn_application_profile_matches())
        return dnn_profile::profile_mismatch_status();
    return CUDNN_STATUS_NOT_SUPPORTED;
}
template <typename Value>
std::uint32_t dnn_enum_bits(const Value& value) noexcept {
    std::uint32_t bits{};
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
CUresult open_dnn_metadata(State& shared, Exclusion& exclusion) {
    if (!shared.dnn_metadata_context) {
        auto* context = shared.allocate_entry(Kind::kDnnMetadataContext, nullptr);
        if (!context) return CUDA_ERROR_OUT_OF_MEMORY;
        context->context = context;
        context->is_alive = true;
        shared.dnn_metadata_context = context;
    }
    return shared.open_context(exclusion, *shared.dnn_metadata_context, 0);
}
std::shared_ptr<DnnEntry> find_dnn_metadata(State& shared, const void* handle,
                                            ExecutionDnnObjectKind kind) {
    const auto token = reinterpret_cast<std::uintptr_t>(handle);
    const auto found = std::ranges::find_if(shared.dnn_entries, [=](const auto& entry) {
        return entry->token == token && entry->kind == kind && entry->is_alive;
    });
    return found == shared.dnn_entries.end() ? nullptr : *found;
}
bool valid_dnn_metadata(const DnnEntry& entry) {
    const auto& context = *entry.context;
    return entry.is_alive && !entry.is_uncertain && context.is_alive && !context.is_uncertain &&
           !context.is_closing && context.epoch == entry.epoch;
}
template <typename Handle>
cudnnStatus_t create_dnn_metadata(Handle* output, ExecutionDnnObjectKind kind,
                                  bool strict_receipt = false) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
        const auto opened = open_dnn_metadata(shared, exclusion);
        if (opened != CUDA_SUCCESS) return dnn_error(opened);
        auto* context = shared.dnn_metadata_context;
        if (!shared.has_dnn_tokens || shared.dnn_entries.size() == kMaxHandles ||
            context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return dnn_profile::host_allocation_status();
        auto entry = std::make_shared<DnnEntry>();
        entry->context = context;
        entry->kind = kind;
        entry->epoch = context->epoch;
        entry->token = shared.next_dnn_token;
        if (shared.next_dnn_token == UINTPTR_MAX)
            shared.has_dnn_tokens = false;
        else
            ++shared.next_dnn_token;
        ExecutionPayloadWriter input;
        if (!input.write_u32(static_cast<std::uint32_t>(kind)))
            return dnn_profile::host_allocation_status();
        shared.dnn_entries.push_back(entry);
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateDnnMetadata, std::move(input));
        if (strict_receipt && reply &&
            (reply->address || !reply->data.empty() || !reply->arguments.empty() ||
             (reply->disposition == ExecutionWireDisposition::kRejected &&
              (reply->id || reply->original_status))))
            return dnn_invalid(*context, &entry->is_uncertain);
        const auto result = dnn_status(reply, *context, &entry->is_uncertain);
        if (result != CUDNN_STATUS_SUCCESS) {
            if (!context->is_uncertain && reply && !reply->id)
                std::erase(shared.dnn_entries, entry);
            else
                (void)dnn_invalid(*context, &entry->is_uncertain);
            return result;
        }
        if (!reply->id ||
            (strict_receipt &&
             (reply->parameter_bytes || entry->is_uncertain || !context->is_alive ||
              context->is_uncertain || context->is_closing || context->epoch != entry->epoch)))
            return dnn_invalid(*context, &entry->is_uncertain);
        entry->remote = reply->id;
        entry->is_alive = true;
        // NOLINTNEXTLINE(performance-no-int-to-ptr): opaque non-dereferenced identity.
        const auto handle = reinterpret_cast<Handle>(static_cast<std::uintptr_t>(entry->token));
        return publish(output, handle) ? CUDNN_STATUS_SUCCESS
                                       : dnn_invalid(*context, &entry->is_uncertain);
    });
}
template <typename Handle, typename Function>
cudnnStatus_t dnn_metadata(Handle handle, ExecutionDnnObjectKind kind, bool is_mutating,
                           Function&& function) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        const auto entry = find_dnn_metadata(shared, handle, kind);
        if (!entry || !valid_dnn_metadata(*entry)) return CUDNN_STATUS_NOT_INITIALIZED;
        auto& context = *entry->context;
        if (entry->active_calls == kMaxHandles || context.active_calls >= kMaxHandles - 1 ||
            (is_mutating && context.active_mutations >= kMaxHandles - 1))
            return dnn_profile::host_allocation_status();
        CounterPin origin(context.active_calls, exclusion);
        std::optional<CounterPin> work;
        if (is_mutating) work.emplace(context.active_mutations, exclusion);
        while (entry->is_retiring)
            if (!yield_frontend(exclusion) || !valid_dnn_metadata(*entry))
                return CUDNN_STATUS_NOT_INITIALIZED;
        std::optional<FlagPin> retiring;
        if (is_mutating) {
            retiring.emplace(entry->is_retiring, exclusion);
            while (entry->active_calls)
                if (!yield_frontend(exclusion) || !valid_dnn_metadata(*entry))
                    return CUDNN_STATUS_NOT_INITIALIZED;
        }
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
template <typename Handle>
cudnnStatus_t destroy_dnn_metadata(Handle handle, ExecutionDnnObjectKind kind,
                                   bool strict_receipt = false) noexcept {
    return dnn_metadata(
        handle, kind, true, [=](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote)) return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kDestroyDnnMetadata, std::move(input));
            if (strict_receipt && reply &&
                (reply->id || reply->address || !reply->data.empty() || !reply->arguments.empty() ||
                 (reply->disposition == ExecutionWireDisposition::kRejected &&
                  reply->original_status)))
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            if (strict_receipt && reply &&
                ((result == CUDNN_STATUS_SUCCESS &&
                  (reply->parameter_bytes || !valid_dnn_metadata(entry))) ||
                 (result != CUDNN_STATUS_SUCCESS &&
                  reply->disposition == ExecutionWireDisposition::kComplete &&
                  reply->parameter_bytes != 1)))
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            if (result == CUDNN_STATUS_SUCCESS) {
                entry.is_alive = false;
                std::erase_if(shared.dnn_entries,
                              [&](const auto& value) { return value.get() == &entry; });
            }
            return result;
        });
}
cudnnStatus_t set_dnn_tensor(cudnnTensorDescriptor_t handle, ExecutionDnnTensorSetter setter,
                             const ExecutionDnnTensor& tensor) noexcept {
    return dnn_metadata(handle, ExecutionDnnObjectKind::kTensor, true,
                        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
                            if (tensor.dimensions < 0 || tensor.dimensions > CUDNN_DIM_MAX)
                                return CUDNN_STATUS_BAD_PARAM;
                            ExecutionPayloadWriter input;
                            if (!input.write_u64(entry.remote) ||
                                !input.write_u32(static_cast<std::uint32_t>(setter)) ||
                                !write_execution_dnn_tensor(input, tensor))
                                return dnn_profile::host_allocation_status();
                            return dnn_status(
                                shared.call(exclusion, *entry.context,
                                            ExecutionOpcode::kSetDnnTensor, std::move(input)),
                                *entry.context, &entry.is_uncertain);
                        });
}
cudnnStatus_t query_dnn_tensor(State& shared, Exclusion& exclusion, DnnEntry& entry, int requested,
                               bool is_four, ExecutionDnnTensor& output) {
    ExecutionPayloadWriter input;
    if (!input.write_u64(entry.remote) || !input.write_u32(static_cast<std::uint32_t>(requested)) ||
        !input.write_u32(is_four ? 1 : 0))
        return dnn_profile::host_allocation_status();
    const auto reply =
        shared.call(exclusion, *entry.context, ExecutionOpcode::kGetDnnTensor, std::move(input));
    const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
    if (result != CUDNN_STATUS_SUCCESS) return result;
    ExecutionPayloadReader reader(reply->data);
    ExecutionDnnTensor value;
    if (!read_execution_dnn_tensor(reader, value) || !reader.empty() ||
        value.type > dnn_profile::data_type_enum_max() || (is_four && value.dimensions != 4))
        return dnn_invalid(*entry.context, &entry.is_uncertain);
    output = value;
    return CUDNN_STATUS_SUCCESS;
}
template <typename Function>
cudnnStatus_t dnn_handle(cudnnHandle_t handle, bool is_mutating, bool is_command,
                         Function&& function) noexcept {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        auto* entry = shared.find(handle, Kind::kDnn);
        auto* context = shared.context();
        if (!entry || !context || entry->context != context || context->is_closing ||
            context->is_uncertain || entry->is_uncertain)
            return CUDNN_STATUS_NOT_INITIALIZED;
        if (context->active_calls >= kMaxHandles - 1 ||
            ((is_mutating || is_command) && context->active_mutations >= kMaxHandles - 1) ||
            entry->active_calls == kMaxHandles)
            return dnn_profile::host_allocation_status();
        const auto epoch = context->epoch;
        CounterPin origin(context->active_calls, exclusion);
        std::optional<CounterPin> work;
        if (is_mutating || is_command) work.emplace(context->active_mutations, exclusion);
        const auto valid = [&] {
            return entry->is_alive && !entry->is_uncertain && context->is_alive &&
                   !context->is_uncertain && context->epoch == epoch;
        };
        while (entry->is_retiring)
            if (!yield_frontend(exclusion) || !valid()) return CUDNN_STATUS_NOT_INITIALIZED;
        std::optional<FlagPin> retiring;
        if (is_mutating) {
            retiring.emplace(entry->is_retiring, exclusion);
            while (entry->active_calls)
                if (!yield_frontend(exclusion) || !valid()) return CUDNN_STATUS_NOT_INITIALIZED;
        }
        CounterPin use(entry->active_calls, exclusion);
        return function(shared, exclusion, *entry);
    });
}
}  // namespace

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kTensor);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t handle) {
    return destroy_dnn_metadata(handle, ExecutionDnnObjectKind::kTensor);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreateActivationDescriptor(DnnActivationHandle* output) {
    return create_dnn_metadata(output, ExecutionDnnObjectKind::kActivation);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroyActivationDescriptor(DnnActivationHandle handle) {
    return destroy_dnn_metadata(handle, ExecutionDnnObjectKind::kActivation);
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t handle,
                                                                cudnnTensorFormat_t format,
                                                                cudnnDataType_t type, int n, int c,
                                                                int h, int w) {
    return set_dnn_tensor(handle, ExecutionDnnTensorSetter::kFour,
                          {.type = dnn_enum_bits(type),
                           .format = dnn_enum_bits(format),
                           .dimensions = 4,
                           .sizes = {n, c, h, w}});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t handle,
                                                                  cudnnDataType_t type, int n,
                                                                  int c, int h, int w, int ns,
                                                                  int cs, int hs, int ws) {
    return set_dnn_tensor(handle, ExecutionDnnTensorSetter::kFourStrided,
                          {.type = dnn_enum_bits(type),
                           .dimensions = 4,
                           .sizes = {n, c, h, w},
                           .strides = {ns, cs, hs, ws}});
}
namespace {
cudnnStatus_t set_dnn_nd(cudnnTensorDescriptor_t handle, std::uint32_t type, int rank,
                         const int* sizes, const int* strides, std::uint32_t format,
                         bool is_packed) noexcept {
    if (::getpid() != kProcess) return CUDNN_STATUS_NOT_INITIALIZED;
    if (rank < 0 || rank > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
    ExecutionDnnTensor tensor{.type = type, .format = format, .dimensions = rank};
    const auto bytes = static_cast<std::size_t>(rank) * sizeof(int);
    if (!self_copy(tensor.sizes.data(), sizes, bytes, false) ||
        (!is_packed && !self_copy(tensor.strides.data(), strides, bytes, false)))
        return CUDNN_STATUS_BAD_PARAM;
    return set_dnn_tensor(
        handle, is_packed ? ExecutionDnnTensorSetter::kNdPacked : ExecutionDnnTensorSetter::kNd,
        tensor);
}
}  // namespace
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t handle,
                                                                cudnnDataType_t type, int rank,
                                                                const int* sizes,
                                                                const int* strides) {
    return set_dnn_nd(handle, dnn_enum_bits(type), rank, sizes, strides, 0, false);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensorNdDescriptorEx(cudnnTensorDescriptor_t handle,
                                                                  cudnnTensorFormat_t format,
                                                                  cudnnDataType_t type, int rank,
                                                                  const int* sizes) {
    return set_dnn_nd(handle, dnn_enum_bits(type), rank, sizes, nullptr, dnn_enum_bits(format),
                      true);
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t handle,
                                                                cudnnDataType_t* type, int* n,
                                                                int* c, int* h, int* w, int* ns,
                                                                int* cs, int* hs, int* ws) {
    return dnn_metadata(handle, ExecutionDnnObjectKind::kTensor, false,
                        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
                            if (!writable_host_range(type, sizeof(*type)))
                                return CUDNN_STATUS_BAD_PARAM;
                            const std::array outputs{n, c, h, w, ns, cs, hs, ws};
                            for (auto* output : outputs)
                                if (!writable_host_range(output, sizeof(int)))
                                    return CUDNN_STATUS_BAD_PARAM;
                            ExecutionDnnTensor value;
                            const auto result =
                                query_dnn_tensor(shared, exclusion, entry, 4, true, value);
                            if (result != CUDNN_STATUS_SUCCESS) return result;
                            const auto native_type = static_cast<cudnnDataType_t>(value.type);
                            if (!publish(type, native_type)) return CUDNN_STATUS_BAD_PARAM;
                            for (std::size_t i = 0; i != 4; ++i)
                                if (!publish(outputs[i], value.sizes[i]) ||
                                    !publish(outputs[i + 4], value.strides[i]))
                                    return CUDNN_STATUS_BAD_PARAM;
                            return CUDNN_STATUS_SUCCESS;
                        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t handle,
                                                                int requested,
                                                                cudnnDataType_t* type, int* rank,
                                                                int* sizes, int* strides) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kTensor, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (requested < 0 || !writable_host_range(type, sizeof(*type)) ||
                !writable_host_range(rank, sizeof(*rank)))
                return CUDNN_STATUS_BAD_PARAM;
            ExecutionDnnTensor value;
            const auto result = query_dnn_tensor(shared, exclusion, entry,
                                                 std::min(requested, CUDNN_DIM_MAX), false, value);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            const auto count = static_cast<std::size_t>(std::min(requested, value.dimensions));
            const auto bytes = count * sizeof(int);
            if (!writable_host_range(sizes, bytes) || !writable_host_range(strides, bytes))
                return CUDNN_STATUS_BAD_PARAM;
            const auto native_type = static_cast<cudnnDataType_t>(value.type);
            return publish(type, native_type) && publish(rank, value.dimensions) &&
                           self_copy(sizes, value.sizes.data(), bytes, true) &&
                           self_copy(strides, value.strides.data(), bytes, true)
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetTensorSizeInBytes(cudnnTensorDescriptor_t handle,
                                                               std::size_t* output) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kTensor, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote)) return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kDnnTensorSize, std::move(input));
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            if (reply->address > SIZE_MAX || !reply->data.empty())
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            return publish(output, static_cast<std::size_t>(reply->address))
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptor(DnnActivationHandle handle,
                                                                  cudnnActivationMode_t mode,
                                                                  cudnnNanPropagation_t nan,
                                                                  double coefficient) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kActivation, true,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote) ||
                !write_execution_dnn_activation(input, {dnn_enum_bits(mode), dnn_enum_bits(nan),
                                                        std::bit_cast<std::uint64_t>(coefficient)}))
                return dnn_profile::host_allocation_status();
            return dnn_status(shared.call(exclusion, *entry.context,
                                          ExecutionOpcode::kSetDnnActivation, std::move(input)),
                              *entry.context, &entry.is_uncertain);
        });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptor(DnnActivationHandle handle,
                                                                  cudnnActivationMode_t* mode,
                                                                  cudnnNanPropagation_t* nan,
                                                                  double* coefficient) {
    return dnn_metadata(
        handle, ExecutionDnnObjectKind::kActivation, false,
        [&](State& shared, Exclusion& exclusion, DnnEntry& entry) {
            if (!writable_host_range(mode, sizeof(*mode)) ||
                !writable_host_range(nan, sizeof(*nan)) ||
                !writable_host_range(coefficient, sizeof(*coefficient)))
                return CUDNN_STATUS_BAD_PARAM;
            ExecutionPayloadWriter input;
            if (!input.write_u64(entry.remote)) return dnn_profile::host_allocation_status();
            const auto reply = shared.call(exclusion, *entry.context,
                                           ExecutionOpcode::kGetDnnActivation, std::move(input));
            const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
            if (result != CUDNN_STATUS_SUCCESS) return result;
            ExecutionPayloadReader reader(reply->data);
            ExecutionDnnActivation value;
            if (!read_execution_dnn_activation(reader, value) || !reader.empty() ||
                value.mode > 7 || value.nan > 1)
                return dnn_invalid(*entry.context, &entry.is_uncertain);
            return publish(mode, static_cast<cudnnActivationMode_t>(value.mode)) &&
                           publish(nan, static_cast<cudnnNanPropagation_t>(value.nan)) &&
                           publish(coefficient, std::bit_cast<double>(value.coefficient_bits))
                       ? CUDNN_STATUS_SUCCESS
                       : CUDNN_STATUS_BAD_PARAM;
        });
}
// NOLINTEND(bugprone-easily-swappable-parameters)

extern "C" cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t* output) {
    return dnn_boundary([&](State& shared, Exclusion& exclusion) {
        if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
        const auto initialized = runtime_context(shared, exclusion);
        if (initialized != CUDA_SUCCESS) return dnn_error(initialized);
        auto* context = shared.context();
        if (!context || context->active_calls >= kMaxHandles - 1 ||
            context->active_mutations >= kMaxHandles - 1)
            return CUDNN_STATUS_NOT_INITIALIZED;
        CounterPin origin(context->active_calls, exclusion);
        CounterPin work(context->active_mutations, exclusion);
        auto* entry = shared.allocate_entry(Kind::kDnn, context);
        if (!entry) return dnn_profile::host_allocation_status();
        const auto reply =
            shared.call(exclusion, *context, ExecutionOpcode::kCreateDnn, ExecutionPayloadWriter{});
        const auto result = dnn_status(reply, *context, &entry->is_uncertain);
        if (result != CUDNN_STATUS_SUCCESS) {
            if (!context->is_uncertain && reply && !reply->id)
                std::erase_if(shared.entries,
                              [&](const auto& value) { return value.get() == entry; });
            else
                (void)dnn_invalid(*context, &entry->is_uncertain);
            return result;
        }
        if (!reply->id) return dnn_invalid(*context, &entry->is_uncertain);
        entry->remote = reply->id;
        entry->is_alive = true;
        return publish(output, reinterpret_cast<cudnnHandle_t>(entry))
                   ? CUDNN_STATUS_SUCCESS
                   : dnn_invalid(*context, &entry->is_uncertain);
    });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t handle) {
    return dnn_handle(handle, true, false, [](State& shared, Exclusion& exclusion, Entry& entry) {
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return dnn_profile::host_allocation_status();
        const auto reply =
            shared.call(exclusion, *entry.context, ExecutionOpcode::kDestroyDnn, std::move(input));
        const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
        if (result == CUDNN_STATUS_SUCCESS)
            entry.is_alive = false;
        else if (!reply || reply->original_status)
            entry.is_uncertain = true;
        return result;
    });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
    return dnn_handle(handle, true, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        Entry* binding{};
        const auto resolved =
            resolve_owned_stream(shared, exclusion, reinterpret_cast<CUstream>(stream), binding);
        if (resolved != CUDA_SUCCESS) return dnn_error(resolved);
        std::optional<CounterPin> stream_use;
        if (binding) {
            if (binding->active_calls >= kMaxHandles) return dnn_profile::host_allocation_status();
            stream_use.emplace(binding->active_calls, exclusion);
        }
        ExecutionPayloadWriter input;
        const auto remote = binding ? binding->remote : 0;
        if (!input.write_u64(entry.remote) || !input.write_u64(remote))
            return dnn_profile::host_allocation_status();
        const auto reply = shared.call(exclusion, *entry.context, ExecutionOpcode::kSetDnnStream,
                                       std::move(input));
        const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
        if (result == CUDNN_STATUS_SUCCESS) {
            entry.dnn_stream = remote;
            entry.dnn_public_stream = reinterpret_cast<std::uintptr_t>(stream);
        }
        return result;
    });
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetStream(cudnnHandle_t handle, cudaStream_t* output) {
    return dnn_handle(handle, false, false, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        if (!writable_host_range(output, sizeof(*output))) return CUDNN_STATUS_BAD_PARAM;
        ExecutionPayloadWriter input;
        if (!input.write_u64(entry.remote)) return dnn_profile::host_allocation_status();
        const auto reply = shared.call(exclusion, *entry.context, ExecutionOpcode::kGetDnnStream,
                                       std::move(input));
        const auto result = dnn_status(reply, *entry.context, &entry.is_uncertain);
        if (result != CUDNN_STATUS_SUCCESS) return result;
        if (reply->address != entry.dnn_stream || !reply->data.empty())
            return dnn_invalid(*entry.context, &entry.is_uncertain);
        if (entry.dnn_stream && !std::ranges::any_of(shared.entries, [&](const auto& stream) {
                return stream->kind == Kind::kStream && stream->context == entry.context &&
                       stream->remote == entry.dnn_stream && stream->is_alive;
            }))
            return dnn_invalid(*entry.context, &entry.is_uncertain);
        // Preserve an explicitly selected default-stream sentinel at the public ABI.
        // NOLINTNEXTLINE(performance-no-int-to-ptr): opaque public stream identity.
        return publish(output, reinterpret_cast<cudaStream_t>(entry.dnn_public_stream))
                   ? CUDNN_STATUS_SUCCESS
                   : CUDNN_STATUS_BAD_PARAM;
    });
}
namespace {
struct DnnComputeArguments {
    ExecutionDnnComputeKind kind{};
    DnnActivationHandle activation{};
    std::array<cudnnTensorDescriptor_t, 2> tensors{};
    const void* alpha{};
    const void* beta{};
    std::array<const void*, 2> buffers{};
};
cudnnStatus_t submit_dnn(cudnnHandle_t handle, const DnnComputeArguments& arguments) noexcept {
    return dnn_handle(handle, false, true, [&](State& shared, Exclusion& exclusion, Entry& entry) {
        auto* metadata = shared.dnn_metadata_context;
        if (!metadata || !metadata->connection || metadata->is_uncertain || metadata->is_closing)
            return CUDNN_STATUS_NOT_INITIALIZED;
        std::array<std::shared_ptr<DnnEntry>, 3> descriptors;
        const std::array<const void*, 3> handles{arguments.activation, arguments.tensors[0],
                                                 arguments.tensors[1]};
        for (;;) {
            bool waiting{};
            for (std::size_t i = 0; i != descriptors.size(); ++i) {
                if (i == 0 && arguments.kind == ExecutionDnnComputeKind::kAddTensor) continue;
                descriptors[i] = find_dnn_metadata(
                    shared, handles[i],
                    i == 0 ? ExecutionDnnObjectKind::kActivation : ExecutionDnnObjectKind::kTensor);
                if (!descriptors[i] || descriptors[i]->context != metadata ||
                    !valid_dnn_metadata(*descriptors[i]))
                    return CUDNN_STATUS_NOT_INITIALIZED;
                if (descriptors[i]->active_calls > kMaxHandles - descriptors.size())
                    return dnn_profile::host_allocation_status();
                waiting |= descriptors[i]->is_retiring;
            }
            if (!waiting) break;
            if (!yield_frontend(exclusion) || entry.context->is_uncertain ||
                metadata->is_uncertain || metadata->is_closing)
                return CUDNN_STATUS_NOT_INITIALIZED;
        }
        if (metadata->active_calls >= kMaxHandles - 1 || entry.context->submissions == UINT64_MAX)
            return dnn_profile::host_allocation_status();
        CounterPin origin(metadata->active_calls, exclusion);
        std::array<std::optional<CounterPin>, 3> pins;
        for (std::size_t i = 0; i != pins.size(); ++i)
            if (descriptors[i]) pins[i].emplace(descriptors[i]->active_calls, exclusion);
        // Pin every descriptor before releasing exclusion for the native type
        // query. Set/destroy cannot race a float query followed by double issue.
        ExecutionDnnTensor tensor;
        const auto queried =
            query_dnn_tensor(shared, exclusion, *descriptors[1], CUDNN_DIM_MAX, false, tensor);
        if (queried != CUDNN_STATUS_SUCCESS) return queried;
        ExecutionDnnCompute compute;
        compute.kind = arguments.kind;
        compute.handle = entry.remote;
        compute.metadata = metadata->connection->session();
        compute.activation = descriptors[0] ? descriptors[0]->remote : 0;
        compute.tensors = {descriptors[1]->remote, descriptors[2]->remote};
        compute.scalar_bytes = tensor.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
        if (!self_copy(compute.alpha.data(), arguments.alpha, compute.scalar_bytes, false) ||
            !self_copy(compute.beta.data(), arguments.beta, compute.scalar_bytes, false))
            return CUDNN_STATUS_BAD_PARAM;
        std::array<std::optional<CounterPin>, 2> buffer_pins, module_pins;
        for (std::size_t i = 0; i != compute.buffers.size(); ++i) {
            const auto address = reinterpret_cast<CUdeviceptr>(arguments.buffers[i]);
            auto* allocation = shared.range(address, 1);
            if (!allocation || allocation->context != entry.context ||
                shared.range_is_retiring(*allocation))
                return CUDNN_STATUS_BAD_PARAM;
            if (allocation->active_calls >= kMaxHandles - 2)
                return dnn_profile::host_allocation_status();
            buffer_pins[i].emplace(allocation->active_calls, exclusion);
            if (allocation->kind == Kind::kModuleGlobal) {
                const auto parent = std::ranges::find_if(shared.entries, [&](const auto& module) {
                    return module->kind == Kind::kModule && module->context == entry.context &&
                           module->remote == allocation->parent && module->is_alive &&
                           !module->is_retiring;
                });
                if (parent == shared.entries.end()) return CUDNN_STATUS_BAD_PARAM;
                if ((*parent)->active_calls >= kMaxHandles - 2)
                    return dnn_profile::host_allocation_status();
                module_pins[i].emplace((*parent)->active_calls, exclusion);
            }
            compute.buffers[i] = {allocation->remote, address - allocation->address};
        }
        ExecutionPayloadWriter input;
        if (!write_execution_dnn_compute(input, compute))
            return dnn_profile::host_allocation_status();
        ++entry.context->submissions;
        return dnn_status(
            shared.call(exclusion, *entry.context, ExecutionOpcode::kDnnCompute, std::move(input)),
            *entry.context, &entry.is_uncertain);
    });
}
}  // namespace
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact public SDK ABIs.
extern "C" cudnnStatus_t CUDNNWINAPI
cudnnActivationForward(cudnnHandle_t handle, DnnActivationHandle activation, const void* alpha,
                       cudnnTensorDescriptor_t x_descriptor, const void* x, const void* beta,
                       cudnnTensorDescriptor_t y_descriptor, void* y) {
    return submit_dnn(handle, {ExecutionDnnComputeKind::kActivationForward,
                               activation,
                               {x_descriptor, y_descriptor},
                               alpha,
                               beta,
                               {x, y}});
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnAddTensor(cudnnHandle_t handle, const void* alpha,
                                                    cudnnTensorDescriptor_t x_descriptor,
                                                    const void* x, const void* beta,
                                                    cudnnTensorDescriptor_t y_descriptor, void* y) {
    return submit_dnn(handle, {ExecutionDnnComputeKind::kAddTensor,
                               nullptr,
                               {x_descriptor, y_descriptor},
                               alpha,
                               beta,
                               {x, y}});
}
// Adjacent unsupported consumers must not dereference a remote descriptor in a
// client-native SDK. These exact typed rejections are not implemented compute.
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetActivationDescriptorSwishBeta(DnnActivationHandle,
                                                                           double) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnGetActivationDescriptorSwishBeta(DnnActivationHandle,
                                                                           double*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t, cudnnTensorDescriptor_t, void*,
                                                    const void*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnScaleTensor(cudnnHandle_t, cudnnTensorDescriptor_t, void*,
                                                      const void*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnTransformTensorEx(cudnnHandle_t,
                                                            cudnnTensorTransformStruct*,
                                                            const void*, cudnnTensorDescriptor_t,
                                                            const void*, const void*,
                                                            cudnnTensorDescriptor_t, void*) {
    return dnn_unsupported();
}
extern "C" cudnnStatus_t CUDNNWINAPI cudnnActivationBackward(cudnnHandle_t, DnnActivationHandle,
                                                             const void*, cudnnTensorDescriptor_t,
                                                             const void*, cudnnTensorDescriptor_t,
                                                             const void*, cudnnTensorDescriptor_t,
                                                             const void*, const void*,
                                                             cudnnTensorDescriptor_t, void*) {
    return dnn_unsupported();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
