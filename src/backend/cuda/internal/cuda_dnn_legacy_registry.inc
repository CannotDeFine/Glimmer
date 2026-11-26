// Included only by cuda_dnn_registry.cc, inside the backend namespace. Keeping
// the transaction beside the registry owner avoids exposing native objects.
namespace {
CudaDnnLegacyResult legacy_rejected(CUresult status) noexcept {
    CudaDnnLegacyResult output;
    output.result = rejected(status);
    if (status == CUDA_ERROR_NOT_READY) output.result.disposition = CudaDnnDisposition::kDeferred;
    return output;
}
#if CUDNN_MAJOR == 8
template <class T>
struct LegacyOutputCell {
    T value{};
    std::shared_ptr<CudaNativeOutputCell> cell;
    CUresult prepare(bool present) noexcept {
        if (!present) return CUDA_SUCCESS;
        auto created = CudaNativeOutputCell::create(std::as_bytes(std::span{&value, 1}));
        cell = std::move(created.cell);
        return created.status;
    }
    T* address() const noexcept {
        return cell ? static_cast<T*>(cell->data()) : nullptr;
    }
    CudaNativeOutputObservation observe() noexcept {
        return cell ? cell->snapshot(std::as_writable_bytes(std::span{&value, 1}))
                    : CudaNativeOutputObservation{CUDA_SUCCESS, false};
    }
};
bool legacy_array_valid(const CudaDnnLegacyDescriptorArray& array, int count) noexcept {
    if (array.has_descriptors > 1 || array.descriptors.size() > kCudaDnnMaxSequenceLengths)
        return false;
    return array.descriptors.size() ==
           (array.has_descriptors && count > 0 ? static_cast<std::size_t>(count) : 0);
}
#endif
}  // namespace

CudaDnnLegacyResult CudaDnnRegistry::legacy_metadata(const CudaDnnLegacyMetadata& input) noexcept {
    auto& state = *implementation_;
    if (const auto proof = state.gate(true); proof != CUDA_SUCCESS) return legacy_rejected(proof);
#if CUDNN_MAJOR != 8
    (void)input;
    return legacy_rejected(CUDA_ERROR_NOT_SUPPORTED);
#else
    using Kind = CudaDnnLegacyMetadataKind;
    const auto& sdk = state.dispatch.legacy;
    if (input.has_output > 1) return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
    if (input.kind == Kind::kVersionCheck) {
        if (input.object || input.has_output || input.seed || input.value < 1 || input.value > 6)
            return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
        const std::array checks{sdk.ops_infer_version, sdk.ops_train_version,
                                sdk.cnn_infer_version, sdk.cnn_train_version,
                                sdk.adv_infer_version, sdk.adv_train_version};
        const auto check = checks[input.value - 1];
        if (!check) return legacy_rejected(CUDA_ERROR_NOT_SUPPORTED);
        const DependencyCall active{state.rnn_calls->is_active};
        CudaDnnLegacyResult output;
        output.result = completed(check());
        return output;
    }
    const bool algorithm = input.kind == Kind::kSetAlgorithm;
    const bool math = input.kind == Kind::kSetMath || input.kind == Kind::kGetMath;
    const bool bias = input.kind == Kind::kSetBias || input.kind == Kind::kGetBias;
    const bool getter = input.kind == Kind::kGetMath || input.kind == Kind::kGetBias;
    if ((!algorithm && !math && !bias) || (!getter && (input.has_output || input.seed)) ||
        (getter && input.value) || (!algorithm && !getter && input.value > 3))
        return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
    const auto found = state.find(input.object);
    const auto kind = algorithm ? CudaDnnObjectKind::kAlgorithm : CudaDnnObjectKind::kRnn;
    if (found == state.objects.end() || (*found)->kind != kind || !(*found)->native ||
        (*found)->is_uncertain || (*found)->has_destroy_attempt)
        return legacy_rejected(CUDA_ERROR_INVALID_HANDLE);
    if ((!getter && found->use_count() != 1)) return legacy_rejected(CUDA_ERROR_NOT_READY);
    if ((algorithm && !sdk.algorithm_set) || (input.kind == Kind::kSetMath && !sdk.math_set) ||
        (input.kind == Kind::kGetMath && !sdk.math_get) ||
        (input.kind == Kind::kSetBias && !sdk.bias_set) ||
        (input.kind == Kind::kGetBias && !sdk.bias_get))
        return legacy_rejected(CUDA_ERROR_NOT_SUPPORTED);
    std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
    if (!algorithm) {
        if (const auto proof = state.pin_dependencies(**found, dependencies); proof != CUDA_SUCCESS)
            return legacy_rejected(proof);
    }
    LegacyOutputCell<std::uint32_t> cell{input.seed, {}};
    if (const auto proof = cell.prepare(getter && input.has_output); proof != CUDA_SUCCESS)
        return legacy_rejected(proof);
    const auto owner = *found;
    const DependencyCall active{state.rnn_calls->is_active};
    cudnnStatus_t native{};
    const auto rnn = static_cast<CudaDnnRnn>(owner->native);
    if (algorithm) {
        cudnnAlgorithm_t value{};
        static_assert(sizeof(value) == sizeof(input.value));
        static_assert(offsetof(cudnnAlgorithm_t, algo) == 0);
        static_assert(alignof(cudnnAlgorithm_t) == alignof(std::uint32_t));
        static_assert(std::is_trivially_copyable_v<cudnnAlgorithm_t>);
        std::memcpy(&value, &input.value, sizeof(value));
        native = sdk.algorithm_set(static_cast<cudnnAlgorithmDescriptor_t>(owner->native), value);
    } else if (input.kind == Kind::kSetMath)
        native = sdk.math_set(rnn, static_cast<cudnnMathType_t>(input.value));
    else if (input.kind == Kind::kSetBias)
        native = sdk.bias_set(rnn, static_cast<cudnnRNNBiasMode_t>(input.value));
    else if (input.kind == Kind::kGetMath)
        native = sdk.math_get(rnn, reinterpret_cast<cudnnMathType_t*>(cell.address()));
    else
        native = sdk.bias_get(rnn, reinterpret_cast<cudnnRNNBiasMode_t*>(cell.address()));
    CudaDnnLegacyResult output;
    output.result = completed(native);
    const auto observed = cell.observe();
    if (observed.status == CUDA_SUCCESS && observed.was_written)
        output.output = {.written = 1, .scalar = cell.value};
    const bool bad_output = observed.was_written && cell.value > 3;
    if ((!getter && native != CUDNN_STATUS_SUCCESS) || observed.status != CUDA_SUCCESS ||
        bad_output ||
        !dnn_profile::status_is_representable(std::bit_cast<std::int32_t>(raw_bits(native)))) {
        owner->is_uncertain = state.is_faulted = output.has_metadata_quarantine = true;
        if (observed.status != CUDA_SUCCESS || bad_output) {
            output.result.disposition = CudaDnnDisposition::kRejected;
            output.result.status =
                observed.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : observed.status;
        }
    }
    return output;
#endif
}

CudaDnnLegacyBorrowResult CudaDnnRegistry::borrow_legacy_rnn(
    const CudaSessionDnnLegacyRnn& input, std::shared_ptr<ExecutionHostStorage> storage) noexcept {
    auto& state = *implementation_;
    const auto reject = [](CUresult proof) {
        CudaDnnLegacyBorrowResult output;
        output.result = legacy_rejected(proof).result;
        return output;
    };
    if (const auto proof = state.gate(true); proof != CUDA_SUCCESS) return reject(proof);
#if CUDNN_MAJOR != 8
    (void)input;
    (void)storage;
    return reject(CUDA_ERROR_NOT_SUPPORTED);
#else
    if (!storage) return reject(CUDA_ERROR_INVALID_VALUE);
    if (input.sequence_length > static_cast<int>(kCudaDnnMaxSequenceLengths))
        return reject(CUDA_ERROR_INVALID_VALUE);
    for (const auto& array : input.sequences)
        if (!legacy_array_valid(array, input.sequence_length))
            return reject(CUDA_ERROR_INVALID_VALUE);
    try {
        CudaDnnLegacyBorrowResult output{{}, CudaDnnLegacyView{std::move(storage)}};
        // No SDK or callback occurs during this transaction; every successfully
        // captured shared owner is rolled back together on any later failure.
        auto rnn = borrow_rnn(input.descriptor);
        if (rnn.result.status != CUDA_SUCCESS ||
            rnn.result.disposition != CudaDnnDisposition::kComplete)
            return reject(rnn.result.status);
        output.view.rnn = std::move(rnn.view);
        const auto take = [&](std::uint64_t id, CudaDnnObjectKind kind,
                              CudaDnnObjectView& destination) {
            if (!id) return true;
            auto view = borrow(id, kind);
            if (!view) return false;
            destination = std::move(*view);
            return true;
        };
        for (std::size_t i = 0; i < input.sequences.size(); ++i) {
            output.view.sequences[i].resize(input.sequences[i].descriptors.size());
            for (std::size_t j = 0; j < input.sequences[i].descriptors.size(); ++j)
                if (!take(input.sequences[i].descriptors[j], CudaDnnObjectKind::kTensor,
                          output.view.sequences[i][j]))
                    return reject(CUDA_ERROR_INVALID_HANDLE);
        }
        for (std::size_t i = 0; i < input.states.size(); ++i)
            if (!take(input.states[i], CudaDnnObjectKind::kTensor, output.view.states[i]))
                return reject(CUDA_ERROR_INVALID_HANDLE);
        if (!take(input.filter, CudaDnnObjectKind::kFilter, output.view.filter))
            return reject(CUDA_ERROR_INVALID_HANDLE);
        output.view.algorithms = output.view.rnn.algorithms;
        output.result = {.disposition = CudaDnnDisposition::kComplete, .status = CUDA_SUCCESS};
        return output;
    } catch (...) {
        return reject(CUDA_ERROR_OUT_OF_MEMORY);
    }
#endif
}

CudaDnnLegacyResult CudaDnnRegistry::legacy_control(cudnnHandle_t handle,
                                                    const CudaDnnLegacyControl& input,
                                                    const CudaDnnLegacyContext& context) noexcept {
    auto& state = *implementation_;
    if (const auto proof = state.gate(true); proof != CUDA_SUCCESS) return legacy_rejected(proof);
#if CUDNN_MAJOR != 8
    (void)handle;
    (void)input;
    (void)context;
    return legacy_rejected(CUDA_ERROR_NOT_SUPPORTED);
#else
    using Kind = CudaDnnLegacyControlKind;
    const auto& sdk = state.dispatch.legacy;
    if (!handle || !context.check || !context.opaque || !context.session || input.has_output > 1)
        return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
    const bool setup = input.kind == Kind::kSetup;
    const bool clip = input.kind == Kind::kClip;
    const bool bind = input.kind == Kind::kSetAlgorithm;
    const bool array_query = input.kind == Kind::kWorkspaceSize || input.kind == Kind::kReserveSize;
    const bool parameter =
        input.kind == Kind::kMatrixParameter || input.kind == Kind::kBiasParameter;
    const bool size = input.kind == Kind::kParameterSize;
    const bool count = input.kind == Kind::kInferenceMaxCount;
    if ((!setup && !clip && !bind && !array_query && !parameter && !size && !count) ||
        ((setup || clip || bind) && (input.has_output || input.seed)) ||
        (setup && (input.setup.input_mode > 1 || input.setup.direction > 1 ||
                   input.setup.cell_mode > 3 || input.setup.algorithm > 7 ||
                   input.setup.math_precision > dnn_profile::data_type_enum_max())) ||
        (clip && (input.clip.mode > 1 || input.clip.nan > 1)) ||
        (size && input.data_type > dnn_profile::data_type_enum_max()) ||
        (count && input.seed > std::numeric_limits<std::uint32_t>::max()) ||
        (array_query && !legacy_array_valid(input.sequence, input.sequence_length)))
        return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
    if ((setup && !sdk.setup) || (clip && !sdk.clip) || (bind && !sdk.algorithm_bind) ||
        (input.kind == Kind::kWorkspaceSize && !sdk.workspace_size) ||
        (input.kind == Kind::kReserveSize && !sdk.reserve_size) || (size && !sdk.parameter_size) ||
        (count && !sdk.inference_max_count) ||
        (input.kind == Kind::kMatrixParameter && !sdk.matrix_parameter) ||
        (input.kind == Kind::kBiasParameter && !sdk.bias_parameter))
        return legacy_rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto parent_it = state.find(input.descriptor);
    if (parent_it == state.objects.end() || (*parent_it)->kind != CudaDnnObjectKind::kRnn ||
        !(*parent_it)->native || (*parent_it)->is_uncertain || (*parent_it)->has_destroy_attempt)
        return legacy_rejected(CUDA_ERROR_INVALID_HANDLE);
    if ((setup || clip || bind) && parent_it->use_count() != 1)
        return legacy_rejected(CUDA_ERROR_NOT_READY);
    try {
        const auto find = [&](std::uint64_t id,
                              CudaDnnObjectKind kind) -> std::shared_ptr<Implementation::Object>* {
            if (!id) return nullptr;
            const auto found = state.find(id);
            return found != state.objects.end() && (*found)->kind == kind && (*found)->native &&
                           !(*found)->is_uncertain && !(*found)->has_destroy_attempt
                       ? &*found
                       : nullptr;
        };
        auto* destination =
            parameter ? find(input.parameters.destination, CudaDnnObjectKind::kFilter) : nullptr;
        if (parameter && input.parameters.destination && !destination)
            return legacy_rejected(CUDA_ERROR_INVALID_HANDLE);
        // Test unique mutable ownership before retaining any aliased inputs.
        if (destination && destination->use_count() != 1)
            return legacy_rejected(CUDA_ERROR_NOT_READY);
        std::vector<std::shared_ptr<Implementation::Object>> owners;
        owners.reserve(input.sequence.descriptors.size() + 5);
        const auto pin = [&](std::uint64_t id, CudaDnnObjectKind kind, void*& native) {
            native = nullptr;
            if (!id) return true;
            const auto object = find(id, kind);
            if (!object) return false;
            owners.push_back(*object);
            native = (*object)->native;
            return true;
        };
        void *x{}, *filter{}, *algorithm{};
        if ((size && !pin(input.input, CudaDnnObjectKind::kTensor, x)) ||
            (parameter && (!pin(input.parameters.input, CudaDnnObjectKind::kTensor, x) ||
                           !pin(input.parameters.filter, CudaDnnObjectKind::kFilter, filter))) ||
            (bind && !pin(input.algorithm, CudaDnnObjectKind::kAlgorithm, algorithm)))
            return legacy_rejected(CUDA_ERROR_INVALID_HANDLE);
        std::vector<cudnnTensorDescriptor_t> sequence;
        if (array_query && input.sequence.has_descriptors) {
            sequence.resize(std::max<std::size_t>(1, input.sequence.descriptors.size()));
            for (std::size_t i = 0; i < input.sequence.descriptors.size(); ++i) {
                void* native{};
                if (!pin(input.sequence.descriptors[i], CudaDnnObjectKind::kTensor, native))
                    return legacy_rejected(CUDA_ERROR_INVALID_HANDLE);
                sequence[i] = static_cast<cudnnTensorDescriptor_t>(native);
            }
        }
        if (destination) owners.push_back(*destination);
        const auto parent = *parent_it;
        Implementation::DependencyUpdate update;
        std::array<std::shared_ptr<Implementation::Object>, 4> dependencies{};
        auto proof = setup ? state.prepare_dependencies(*parent, {input.setup.dropout, 0}, update)
                           : state.pin_dependencies(*parent, dependencies);
        if (proof != CUDA_SUCCESS) return legacy_rejected(proof);
        // Candidate algorithm storage is bounded and allocated before SDK entry.
        auto algorithm_retained = parent->algorithm_dependencies;
        if (bind && input.algorithm &&
            std::ranges::find(algorithm_retained, input.algorithm) == algorithm_retained.end()) {
            const auto slot = std::ranges::find(algorithm_retained, 0);
            if (slot == algorithm_retained.end()) return legacy_rejected(CUDA_ERROR_UNKNOWN);
            *slot = input.algorithm;
        }
        LegacyOutputCell<std::uint64_t> scalar{input.seed, {}};
        LegacyOutputCell<std::uint32_t> integer{static_cast<std::uint32_t>(input.seed), {}};
        // This address is an inert private sentinel, never dereferenced by the
        // SDK and never serialized. COW also detects an equal-value store.
        int sentinel{};
        LegacyOutputCell<void*> pointer{&sentinel, {}};
        if (input.has_output) {
            proof = parameter ? pointer.prepare(true)
                    : count   ? integer.prepare(true)
                              : scalar.prepare(true);
            if (proof != CUDA_SUCCESS) return legacy_rejected(proof);
        }
        const DependencyCall active{state.rnn_calls->is_active};
        if ((proof = context.check(context.opaque)) != CUDA_SUCCESS) {
            auto output = legacy_rejected(proof);
            output.has_context_loss = true;
            return output;
        }
        std::optional<CudaDeviceMemoryView> weights;
        std::size_t weight_bytes{};
        if (parameter && input.parameters.weights.allocation) {
            if (!filter || !state.dispatch.filter_size)
                return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
            const auto queried =
                state.dispatch.filter_size(static_cast<CudaDnnFilterHandle>(filter), &weight_bytes);
            if ((proof = context.check(context.opaque)) != CUDA_SUCCESS) {
                auto output = legacy_rejected(proof);
                output.has_context_loss = true;
                return output;
            }
            if (queried != CUDNN_STATUS_SUCCESS) return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
            weights = context.session->device_memory(input.parameters.weights, weight_bytes);
            if (!weights) return legacy_rejected(CUDA_ERROR_INVALID_VALUE);
        }
        const auto rnn = static_cast<CudaDnnRnn>(parent->native);
        cudnnStatus_t native{};
        if (setup)
            native = sdk.setup(handle, rnn, input.setup.hidden_size, input.setup.layers,
                               update.native[0],
                               static_cast<cudnnRNNInputMode_t>(input.setup.input_mode),
                               static_cast<cudnnDirectionMode_t>(input.setup.direction),
                               static_cast<cudnnRNNMode_t>(input.setup.cell_mode),
                               static_cast<cudnnRNNAlgo_t>(input.setup.algorithm),
                               static_cast<cudnnDataType_t>(input.setup.math_precision));
        else if (clip)
            native = sdk.clip(handle, rnn, static_cast<cudnnRNNClipMode_t>(input.clip.mode),
                              static_cast<cudnnNanPropagation_t>(input.clip.nan), input.clip.lower,
                              input.clip.upper);
        else if (bind)
            native =
                sdk.algorithm_bind(handle, rnn, static_cast<cudnnAlgorithmDescriptor_t>(algorithm));
        else if (array_query) {
            static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));
            const auto call =
                input.kind == Kind::kWorkspaceSize ? sdk.workspace_size : sdk.reserve_size;
            native = call(handle, rnn, input.sequence_length,
                          input.sequence.has_descriptors ? sequence.data() : nullptr,
                          reinterpret_cast<std::size_t*>(scalar.address()));
        } else if (size)
            native = sdk.parameter_size(handle, rnn, static_cast<cudnnTensorDescriptor_t>(x),
                                        reinterpret_cast<std::size_t*>(scalar.address()),
                                        static_cast<cudnnDataType_t>(input.data_type));
        else if (count)
            native =
                sdk.inference_max_count(handle, rnn, reinterpret_cast<int*>(integer.address()));
        else {
            const auto call =
                input.kind == Kind::kMatrixParameter ? sdk.matrix_parameter : sdk.bias_parameter;
            // NOLINTNEXTLINE(performance-no-int-to-ptr): service-owned allocation, checked above.
            const auto* base = weights ? reinterpret_cast<const void*>(weights->address) : nullptr;
            native = call(
                handle, rnn, input.parameters.pseudo_layer, static_cast<cudnnTensorDescriptor_t>(x),
                static_cast<CudaDnnFilterHandle>(filter), base, input.parameters.linear_layer,
                destination ? static_cast<CudaDnnFilterHandle>((*destination)->native) : nullptr,
                pointer.address());
        }
        CudaDnnLegacyResult output;
        output.result = completed(native);
        proof = context.check(context.opaque);
        output.has_context_loss = proof != CUDA_SUCCESS;
        const auto quarantine = [&] {
            state.is_faulted = output.has_metadata_quarantine = true;
            if (setup || clip || bind) parent->is_uncertain = true;
            if (destination) (*destination)->is_uncertain = true;
        };
        if (setup) {
            parent->dropout_dependencies = native == CUDNN_STATUS_SUCCESS && proof == CUDA_SUCCESS
                                               ? update.replacement
                                               : update.retained;
        }
        if (bind)
            parent->algorithm_dependencies = native == CUDNN_STATUS_SUCCESS && proof == CUDA_SUCCESS
                                                 ? std::array<std::uint64_t, 2>{input.algorithm, 0}
                                                 : algorithm_retained;
        if (((setup || clip || bind || parameter) && native != CUDNN_STATUS_SUCCESS) ||
            proof != CUDA_SUCCESS)
            quarantine();
        const auto observed = parameter ? pointer.observe()
                              : count   ? integer.observe()
                                        : scalar.observe();
        if (observed.status != CUDA_SUCCESS) {
            quarantine();
            if (proof == CUDA_SUCCESS) proof = observed.status;
        } else if (observed.was_written && !parameter) {
            output.output = {.written = 1, .scalar = count ? integer.value : scalar.value};
        } else if (observed.was_written && parameter) {
            if (!pointer.value)
                output.output.written = 2;
            else if (proof == CUDA_SUCCESS && weights && destination &&
                     state.dispatch.filter_size && pointer.value != &sentinel) {
                std::size_t extent{};
                const auto extent_status = state.dispatch.filter_size(
                    static_cast<CudaDnnFilterHandle>((*destination)->native), &extent);
                proof = context.check(context.opaque);
                output.has_context_loss |= proof != CUDA_SUCCESS;
                const auto address = reinterpret_cast<std::uintptr_t>(pointer.value);
                if (proof == CUDA_SUCCESS && extent_status == CUDNN_STATUS_SUCCESS && extent &&
                    address >= weights->address && address - weights->address <= weight_bytes &&
                    extent <= weight_bytes - (address - weights->address))
                    output.output = {.written = 2,
                                     .non_null = 1,
                                     .offset = address - weights->address,
                                     .extent = extent};
                else if (proof == CUDA_SUCCESS)
                    proof = CUDA_ERROR_UNKNOWN;
            } else if (proof == CUDA_SUCCESS)
                proof = CUDA_ERROR_UNKNOWN;
            if (pointer.value && !output.output.non_null) quarantine();
        }
        if (proof != CUDA_SUCCESS) {
            output.result.disposition = CudaDnnDisposition::kRejected;
            output.result.status = proof;
            // Context loss is independently checked; mapping/observation faults
            // quarantine metadata without inventing loss of the native context.
            output.has_context_loss |= context.check(context.opaque) != CUDA_SUCCESS;
        }
        return output;
    } catch (...) {
        return legacy_rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
#endif
}
