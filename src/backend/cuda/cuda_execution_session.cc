#include "internal/cuda_execution_session_state.h"

// Session lifecycle, contexts, modules, memory, streams, host memory, graphs,
// kernels, events, scheduling and shutdown.
namespace glimmer::backend::cuda {
struct CudaSessionHostFunction::State {
    std::atomic<bool> entered{}, completed{};
};
CudaSessionHostFunction::CudaSessionHostFunction() : state_(std::make_unique<State>()) {}
CudaSessionHostFunction::~CudaSessionHostFunction() = default;
bool CudaSessionHostFunction::entered() const noexcept {
    return state_->entered.load(std::memory_order_acquire);
}
void CudaSessionHostFunction::finish() noexcept {
    state_->completed.store(true, std::memory_order_release);
    state_->completed.notify_all();
}
void CUDA_CB CudaSessionHostFunction::wait(void* data) noexcept {
    auto& state = *static_cast<State*>(data);
    state.entered.store(true, std::memory_order_release);
    while (!state.completed.load(std::memory_order_acquire))
        state.completed.wait(false, std::memory_order_acquire);
}

std::size_t cuda_gemm_scalar_bytes(cublasComputeType_t compute, cudaDataType output) noexcept {
    return gemm_scalar_bytes(compute, output);
}

CUresult CudaSessionEvent::observe() const noexcept {
    if (!state_ || !lifetime_) return CUDA_ERROR_INVALID_HANDLE;
    if (static_cast<std::uint64_t>(::getpid()) != process_ || !resource_owner ||
        thread_.lock() != resource_owner)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (owner_.expired()) return CUDA_ERROR_INVALID_HANDLE;
    if (state_->record_status != CUDA_SUCCESS) return state_->record_status;
    return state_->frontier ? state_->frontier->observe() : CUDA_SUCCESS;
}
std::optional<CudaSessionEventIdentity> CudaSessionEvent::diagnostic_identity() const noexcept {
    if (!state_ || !lifetime_ || static_cast<std::uint64_t>(::getpid()) != process_ ||
        !resource_owner || thread_.lock() != resource_owner || owner_.expired())
        return std::nullopt;
    const auto activity = state_->activity.lock();
    if (!activity || activity->is_native_issue_active || activity->is_folded_active ||
        activity->is_rnn_query_active || activity->is_preparation_active ||
        !state_->identity.event_object)
        return std::nullopt;
    return state_->identity;
}

CUcontext CudaSessionExternalContext::native() const noexcept {
    return state_ && state_->process == ::getpid() ? state_->native : nullptr;
}
std::uint64_t CudaSessionExternalContext::id() const noexcept {
    return native() ? state_->id : 0;
}
int CudaSessionExternalContext::device_ordinal() const noexcept {
    return native() ? state_->device_ordinal : -1;
}
CudaSessionExternalContext CudaSessionExternalCommand::context() const noexcept {
    return state_ && state_->context.native() ? state_->context : CudaSessionExternalContext{};
}
CUstream CudaSessionExternalCommand::stream() const noexcept {
    return state_ && state_->context.native() ? state_->stream : nullptr;
}
std::uint64_t CudaSessionExternalCommand::stream_id() const noexcept {
    return state_ && state_->context.native() ? state_->stream_id : 0;
}
std::span<const CudaSessionExternalRange> CudaSessionExternalCommand::ranges() const noexcept {
    return state_ && state_->context.native()
               ? std::span<const CudaSessionExternalRange>{state_->ranges}
               : std::span<const CudaSessionExternalRange>{};
}

CudaExecutionSession::CudaExecutionSession(
    ExecutionSessionIdentity identity, std::uint64_t context_id, CudaSessionDispatch dispatch,
    CudaSessionLimits limits, CudaSessionPersistentConfig persistent,
    std::shared_ptr<CudaPtxCache> ptx_cache, std::shared_ptr<CudaProfileCache> profile_cache,
    bool requires_profile_preemption, CudaSessionStreamConfig streams,
    CudaExecutionDomainMember domain_member)
    : implementation_(std::make_unique<Implementation>(
          identity, context_id, std::move(dispatch), limits, persistent, std::move(ptx_cache),
          std::move(profile_cache), requires_profile_preemption, streams,
          std::move(domain_member))) {}

CUresult CudaExecutionSession::attach_process_memory(
    std::shared_ptr<CudaProcessMemoryBackend> owner) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return CUDA_ERROR_INVALID_CONTEXT;
    if (!owner || state.has_started || state.storage->context || state.process_memory ||
        state.next_id >= (UINT64_C(1) << 63))
        return CUDA_ERROR_INVALID_VALUE;
    state.process_memory = std::move(owner);
    return CUDA_SUCCESS;
}
CudaSessionProcessAllocationAcceptance CudaExecutionSession::allocate_process_memory(
    std::uint64_t sequence, const CudaProcessMemoryRequest& request) noexcept {
    auto& state = *implementation_;
    const auto invalid = [] {
        return CudaSessionProcessAllocationAcceptance{
            {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}}, {}};
    };
    if (!state.process_memory || state.process_memory->native_health() != CUDA_SUCCESS ||
        !state.owns() || state.is_closing || state.is_native_issue_active ||
        state.check_context() != CUDA_SUCCESS || !state.executor ||
        (request.operation != CudaProcessMemoryOperation::kAllocateAsync &&
         request.operation != CudaProcessMemoryOperation::kAllocateFromPoolAsync))
        return invalid();
    if (request.stream &&
        std::ranges::find(state.storage->streams, request.stream, &Implementation::Stream::id) ==
            state.storage->streams.end())
        return invalid();
    try {
        auto operation = std::make_shared<Implementation::ProcessAllocation>();
        operation->backend = state.process_memory;
        operation->request = request;
        operation->tail = std::make_shared<Implementation::ProcessTail>(state.storage);
        if (request.has_output) {
            auto created =
                CudaNativeOutputCell::create(std::as_bytes(std::span{&request.output_seed, 1}));
            if (created.status != CUDA_SUCCESS)
                return {{.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}}, {}};
            operation->output = std::move(created.cell);
        }
        const auto reserved = state.process_memory->reserve(request);
        if (reserved.status != CUDA_SUCCESS)
            return {{.status = reserved.status == CUDA_ERROR_OUT_OF_MEMORY
                                   ? ExecutionAcceptStatus::kOutOfMemory
                                   : ExecutionAcceptStatus::kInvalid,
                     .receipt = {}},
                    {}};
        operation->id = reserved.id;
        auto accepted = state.executor->accept_library(sequence, {request.stream, {}, operation});
        return {accepted, accepted.accepted() ? operation : nullptr};
    } catch (...) {
        return {{.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}}, {}};
    }
}
CudaExecutionSession::~CudaExecutionSession() {
    auto& state = *implementation_;
    if (getpid() != state.process) {
        (void)implementation_.release();
        return;
    }
    if (!state.owns() || shutdown().status != CUDA_SUCCESS) {
        const auto* retained = implementation_.release();
        try {
            spdlog::critical("Retaining unsettled CUDA session: client={}, context={}",
                             retained->identity.client_id, retained->context_id);
        } catch (...) {
        }
    }
}

CudaSessionResult CudaExecutionSession::initialize(int device_ordinal,
                                                   CudaSessionContextConfig config) noexcept {
    const auto flags = config.flags;
    auto& state = *implementation_;
    if (state.process_memory) {
        const auto health = state.process_memory->native_health();
        if (health != CUDA_SUCCESS) return rejected(health);
    }
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    if (state.is_closing) return rejected(CUDA_ERROR_DEINITIALIZED);
    if (device_ordinal < 0 || state.has_started) return rejected(CUDA_ERROR_INVALID_VALUE);
    const auto flags_status = context_flags_status(flags);
    if (flags_status != CUDA_SUCCESS) return rejected(flags_status);
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        const auto quiescent = state.domain_cold_status();
        if (quiescent == CUDA_ERROR_NOT_READY) return deferred_resource_release();
        if (quiescent != CUDA_SUCCESS) return rejected(quiescent);
        const auto effective_flags =
            flags | ((state.persistent.workers || (member.flags & CU_CTX_MAP_HOST))
                         ? static_cast<unsigned int>(CU_CTX_MAP_HOST)
                         : 0U);
        if (member.device_ordinal != device_ordinal || member.logical_flags != effective_flags)
            return rejected(CUDA_ERROR_INVALID_VALUE);
        const auto attached =
            state.storage->domain_member.attach({state.identity, state.context_id});
        if (attached != CUDA_SUCCESS) return rejected(attached);
    }
    // Sharing adds cold native observations. Exclude callback lifecycle reentry
    // before the first observation, without changing legacy initialization.
    std::optional<NativeExecutionScope> sharing_scope;
    if (state.persistent.uses_workload_agnostic || state.stream_config.uses_default_proxy)
        sharing_scope.emplace(state.is_native_issue_active);
    state.has_started = true;
    auto result = CUDA_SUCCESS;
    CUdevice device{};
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        device = member.native_device;
    } else {
        result = state.invoke(state.dispatch.initialize, 0);
        if (result == CUDA_SUCCESS)
            result = state.invoke(state.dispatch.device_get, &device, device_ordinal);
    }
    if (result == CUDA_SUCCESS) state.device_ordinal = device_ordinal;
    if (result != CUDA_SUCCESS) return rejected(result);
    state.native_device = device;
    if (state.persistent.uses_workload_agnostic) {
        int multiprocessors{}, native_threads{};
        result = state.invoke(state.dispatch.device_attribute, &multiprocessors,
                              CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        if (result != CUDA_SUCCESS) return rejected(result);
        result = state.invoke(state.dispatch.device_attribute, &native_threads,
                              CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
        if (result != CUDA_SUCCESS) return rejected(result);
        if (multiprocessors <= 0 || native_threads <= 0 ||
            state.persistent.sharing_threads_per_sm > static_cast<unsigned int>(native_threads))
            return rejected(CUDA_ERROR_INVALID_VALUE);
        state.sharing_sm_count = static_cast<unsigned int>(multiprocessors);
    }
    if (state.profile_cache) {
        CUuuid uuid{};
        int driver{};
        result = state.invoke(state.dispatch.device_uuid, &uuid, device);
        if (result == CUDA_SUCCESS) result = state.invoke(state.dispatch.driver_version, &driver);
        if (result != CUDA_SUCCESS) return rejected(result);
        if (driver <= 0) return rejected(CUDA_ERROR_INVALID_VALUE);
        static_assert(sizeof(uuid.bytes) == sizeof(state.profile_platform.uuid));
        std::memcpy(state.profile_platform.uuid.data(), uuid.bytes, sizeof(uuid.bytes));
        if (std::ranges::all_of(state.profile_platform.uuid,
                                [](auto value) { return value == std::byte{}; }))
            return rejected(CUDA_ERROR_INVALID_VALUE);
        state.profile_platform.driver_version = static_cast<unsigned int>(driver);
        state.profile_platform.toolkit_version = CUDA_VERSION;
        state.profile_platform.generator_abi = kCudaPtxGeneratorAbi;
    }
    CUcontext current{};
    result = state.invoke(state.dispatch.execution.context_get_current, &current);
    if (result != CUDA_SUCCESS) return rejected(result);
    if (current) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        state.storage->context = member.context.native;
        result = state.invoke(state.dispatch.context_set_current, state.storage->context);
    } else {
        result = state.invoke(
            state.dispatch.context_create, &state.storage->context,
            flags | (state.persistent.workers ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U),
            device);
    }
    CUcontext after{};
    const auto observed = state.invoke(state.dispatch.execution.context_get_current, &after);
    const bool known_extent =
        observed == CUDA_SUCCESS && (!after || after == state.storage->context);
    if (result != CUDA_SUCCESS || !state.storage->context) {
        const bool is_clean =
            !state.storage->context && clean_context_creation_rejection(result) && known_extent;
        state.is_creation_unproven =
            !known_extent || (!state.storage->context && !clean_context_creation_rejection(result));
        if (!is_clean) state.fault();
        auto failed = completed(result);
        if (result == CUDA_SUCCESS) {
            failed.disposition = CudaSessionDisposition::kRejected;
            failed.status = CUDA_ERROR_UNKNOWN;
        }
        return failed;
    }
    if (observed != CUDA_SUCCESS || after != state.storage->context) {
        state.is_creation_unproven = true;
        state.fault();
        auto failed = rejected(observed == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : observed);
        failed.original_status = CUDA_SUCCESS;
        return failed;
    }
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.materialize_configuration(true);
        if (configured != CUDA_SUCCESS) return rejected(configured);
    }
    if (state.stream_config.uses_default_proxy) {
        int native_priority{};
        if (state.stream_config.scheduling_priority) {
            const auto priorities = state.scheduling_stream_priorities(0);
            if (priorities.status != CUDA_SUCCESS) {
                auto failed = rejected(priorities.status);
                failed.original_status = CUDA_SUCCESS;
                return failed;  // The created/borrowed context remains owned.
            }
            native_priority = priorities.native;
        }
        result = state.check_context(true);
        if (result == CUDA_SUCCESS) {
            result = state.stream_config.scheduling_priority
                         ? state.invoke(state.dispatch.stream_create_priority,
                                        &state.storage->default_proxy, CU_STREAM_NON_BLOCKING,
                                        native_priority)
                         : state.invoke(state.dispatch.stream_create, &state.storage->default_proxy,
                                        CU_STREAM_NON_BLOCKING);
            if (result == CUDA_SUCCESS && state.storage->default_proxy &&
                state.stream_config.scheduling_priority) {
                int actual{};
                result = state.invoke(state.dispatch.stream_get_priority,
                                      state.storage->default_proxy, &actual);
                if (result == CUDA_SUCCESS && actual != native_priority)
                    result = CUDA_ERROR_UNKNOWN;
            }
        }
        const auto proof = state.check_context(true);
        if (result != CUDA_SUCCESS || proof != CUDA_SUCCESS || !state.storage->default_proxy ||
            state.storage->default_proxy == CU_STREAM_LEGACY ||
            state.storage->default_proxy == CU_STREAM_PER_THREAD) {
            state.storage->is_default_proxy_uncertain = true;
            state.fault();
            auto failed = rejected(proof != CUDA_SUCCESS    ? proof
                                   : result != CUDA_SUCCESS ? result
                                                            : CUDA_ERROR_UNKNOWN);
            failed.original_status = CUDA_SUCCESS;
            return failed;
        }
    }
    result = state.initialize_progress();
    if (result != CUDA_SUCCESS) {
        if (!state.clean_shared_allocation_rejection(result) ||
            !state.rollback_shared_initialization())
            state.fault();
        auto failed = rejected(result);
        failed.original_status = CUDA_SUCCESS;
        return failed;
    }
    try {
        auto execution = state.dispatch.execution;
        execution.upload_async = state.dispatch.upload_async;
        execution.download_async = state.dispatch.download_async;
        execution.stream_wait_event = state.dispatch.stream_wait_event;
        state.executor = std::make_unique<CudaExecutionDispatcher>(
            state.identity,
            CudaExecutionContext{state.context_id, state.storage->context, state.storage,
                                 state.process_memory ? state.process_memory->owner() : nullptr,
                                 state.process_memory
                                     ? std::function<CUresult()>{[memory = state.process_memory] {
                                           return memory->native_health();
                                       }}
                                     : std::function<CUresult()>{}},
            execution, state.limits.submissions, state.limits.max_objects + 28,
            state.profile_cache && state.profile_cache->is_learning(),
            CudaExecutionStreamConfig{state.storage->default_proxy,
                                      state.storage->default_proxy ? state.storage : nullptr},
            state.owner_activity,
            state.storage->domain_member
                ? std::function<CUresult()>{[member = state.storage->domain_member] {
                      return member.other_members_quiescent();
                  }}
                : std::function<CUresult()>{});
        result = state.executor->bind_executor();
        if (result != CUDA_SUCCESS) {
            state.executor.reset();
            state.fault();
            auto failed = rejected(result);
            failed.original_status = CUDA_SUCCESS;
            return failed;
        }
        return completed(result);
    } catch (...) {
        state.fault();
        auto failed = rejected(CUDA_ERROR_OUT_OF_MEMORY);
        failed.original_status = CUDA_SUCCESS;
        return failed;
    }
}

CudaSessionResult CudaExecutionSession::get_context_flags() noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (!state.dispatch.context_get_flags) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    unsigned int flags{};
    auto result = completed(state.invoke(state.dispatch.context_get_flags, &flags));
    if (result.status == CUDA_SUCCESS) result.id = flags;
    return result;
}

CudaSessionResult CudaExecutionSession::set_context_flags(unsigned int flags) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    const auto flags_status = context_flags_status(flags);
    if (flags_status != CUDA_SUCCESS) return rejected(flags_status);
    if (!state.dispatch.context_set_flags) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (state.uses_shared_domain) {
        auto gate = state.cold_gate(true);
        if (gate.status != CUDA_SUCCESS) return gate;
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        const auto configured = state.mutate_domain_configuration(
            CudaExecutionDomainConfigurationMutation::kFlags, 0, state.dispatch.context_set_flags,
            flags | (member.flags & CU_CTX_MAP_HOST) |
                (state.persistent.workers ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U));
        auto result = configured.status == CUDA_ERROR_NOT_READY ? deferred_resource_release()
                                                                : rejected(configured.status);
        result.original_status =
            configured.original_status
                ? std::optional{static_cast<CUresult>(*configured.original_status)}
                : std::nullopt;
        if (configured.original_status && configured.status == CUDA_SUCCESS) {
            result.disposition = CudaSessionDisposition::kComplete;
            result.status = static_cast<CUresult>(*configured.original_status);
        }
        return result;
    }
    state.has_context_configuration_attempt = true;
    if (state.control_epoch == UINT64_MAX) return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    ++state.control_epoch;
    const NativeExecutionScope active{state.is_native_issue_active};
    return completed(state.invoke(
        state.dispatch.context_set_flags,
        flags | (state.persistent.workers ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U)));
}

CudaContextControlResult CudaExecutionSession::context_control(CudaContextControl operation,
                                                               std::uint32_t argument,
                                                               std::uint64_t value,
                                                               bool is_runtime) noexcept {
    auto& state = *implementation_;
    CudaContextControlResult result;
    using Operation = CudaContextControl;
    const bool is_mutation =
        operation == Operation::kSetLimit || operation == Operation::kSetCache ||
        operation == Operation::kSetSharedMemory || operation == Operation::kResetPersistingCache;
    auto gate = is_mutation ? state.cold_gate(true) : rejected(state.check_context());
    if (!is_mutation && gate.status == CUDA_SUCCESS) {
        if (state.is_closing || state.is_faulted || !state.executor ||
            state.executor->snapshot().is_quarantined)
            gate.status = CUDA_ERROR_UNKNOWN;
        else
            gate = settled();
    }
    result.status = gate.status;
    if (gate.disposition != CudaSessionDisposition::kComplete) {
        result.disposition = gate.disposition;
        return result;
    }
    const bool has_argument =
        operation == Operation::kGetLimit || operation == Operation::kSetLimit ||
        operation == Operation::kSetCache || operation == Operation::kSetSharedMemory ||
        operation == Operation::kGetAffinity;
    if ((!has_argument && argument) || (operation != Operation::kSetLimit && value) ||
        value > std::numeric_limits<std::size_t>::max() ||
        ((operation == Operation::kGetLimit || operation == Operation::kSetLimit) &&
         argument > (is_runtime ? 7U : 15U)) ||
        ((operation == Operation::kSetCache || operation == Operation::kSetSharedMemory) &&
         argument > 3) ||
        (operation == Operation::kGetAffinity && argument > 1)) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }
    // Reject only values outside the SDK enum's representable range. Unknown
    // but representable enumerators reach the SDK and retain its original error.
    if (operation == Operation::kGetLimit && state.uses_shared_domain) {
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) {
            result.status = member.status;
            return result;
        }
        if (member.has_unobserved_configuration_effects) {
            result.status = CUDA_ERROR_NOT_READY;
            result.disposition = CudaSessionDisposition::kDeferred;
            return result;
        }
    }
    result.status = CUDA_ERROR_NOT_SUPPORTED;
    const auto complete = [&](auto status) {
        // Existing profiles are qualified for the context's initial controls.
        // A native mutation attempt may affect state even on an error. Do not
        // train/reuse those profiles afterward; ordinary configured execution
        // and persistent preparation remain enabled.
        if (is_mutation) state.has_context_configuration_attempt = true;
        result.disposition = CudaSessionDisposition::kComplete;
        result.status = CUDA_SUCCESS;
        result.original_status = static_cast<std::int32_t>(status);
        if (*result.original_status != 0) result.value = result.auxiliary = 0;
        return result;
    };
    const auto mutate = [&](auto callback, auto... arguments) {
        if (!state.uses_shared_domain) {
            if (state.control_epoch == UINT64_MAX) {
                result.status = CUDA_ERROR_OUT_OF_MEMORY;
                return result;
            }
            ++state.control_epoch;
            const NativeExecutionScope active{state.is_native_issue_active};
            return complete(state.invoke(callback, arguments...));
        }
        using Mutation = CudaExecutionDomainConfigurationMutation;
        const auto mutation = operation == Operation::kSetLimit   ? Mutation::kLimit
                              : operation == Operation::kSetCache ? Mutation::kCache
                              : operation == Operation::kSetSharedMemory
                                  ? Mutation::kSharedMemory
                                  : Mutation::kResetPersistingCache;
        const auto configured = state.mutate_domain_configuration(
            mutation, mutation == Mutation::kLimit ? argument : 0, callback, arguments...);
        result.status = configured.status;
        result.original_status = configured.original_status;
        result.disposition = configured.status == CUDA_ERROR_NOT_READY
                                 ? CudaSessionDisposition::kDeferred
                             : configured.original_status && configured.status == CUDA_SUCCESS
                                 ? CudaSessionDisposition::kComplete
                                 : CudaSessionDisposition::kRejected;
        return result;
    };
    const auto& sdk = state.dispatch.context;
    if (is_runtime) {
        const auto& runtime = sdk.runtime;
        switch (operation) {
            case Operation::kGetLimit: {
                if (!runtime.get_limit) return result;
                std::size_t bytes{};
                const auto status =
                    state.invoke(runtime.get_limit, &bytes, static_cast<cudaLimit>(argument));
                result.value = bytes;
                return complete(status);
            }
            case Operation::kSetLimit:
                if (!runtime.set_limit) return result;
                return mutate(runtime.set_limit, static_cast<cudaLimit>(argument),
                              static_cast<std::size_t>(value));
            case Operation::kGetCache: {
                if (!runtime.get_cache) return result;
                cudaFuncCache configuration{};
                const auto status = state.invoke(runtime.get_cache, &configuration);
                result.value = static_cast<std::uint32_t>(configuration);
                return complete(status);
            }
            case Operation::kSetCache:
                if (!runtime.set_cache) return result;
                return mutate(runtime.set_cache, static_cast<cudaFuncCache>(argument));
            case Operation::kGetSharedMemory: {
                if (!runtime.get_shared) return result;
                cudaSharedMemConfig configuration{};
                const auto status = state.invoke(runtime.get_shared, &configuration);
                result.value = static_cast<std::uint32_t>(configuration);
                return complete(status);
            }
            case Operation::kSetSharedMemory:
                if (!runtime.set_shared) return result;
                return mutate(runtime.set_shared, static_cast<cudaSharedMemConfig>(argument));
            case Operation::kResetPersistingCache:
                if (!runtime.reset_cache) return result;
                return mutate(runtime.reset_cache);
            default:
                return result;
        }
    }
    switch (operation) {
        case Operation::kGetLimit: {
            if (!sdk.get_limit) return result;
            std::size_t bytes{};
            const auto status = state.invoke(sdk.get_limit, &bytes, static_cast<CUlimit>(argument));
            result.value = bytes;
            return complete(status);
        }
        case Operation::kSetLimit:
            if (!sdk.set_limit) return result;
            return mutate(sdk.set_limit, static_cast<CUlimit>(argument),
                          static_cast<std::size_t>(value));
        case Operation::kGetCache: {
            if (!sdk.get_cache) return result;
            CUfunc_cache configuration{};
            const auto status = state.invoke(sdk.get_cache, &configuration);
            result.value = static_cast<std::uint32_t>(configuration);
            return complete(status);
        }
        case Operation::kSetCache:
            if (!sdk.set_cache) return result;
            return mutate(sdk.set_cache, static_cast<CUfunc_cache>(argument));
        case Operation::kGetSharedMemory: {
            if (!sdk.get_shared) return result;
            CUsharedconfig configuration{};
            const auto status = state.invoke(sdk.get_shared, &configuration);
            result.value = static_cast<std::uint32_t>(configuration);
            return complete(status);
        }
        case Operation::kSetSharedMemory:
            if (!sdk.set_shared) return result;
            return mutate(sdk.set_shared, static_cast<CUsharedconfig>(argument));
        case Operation::kResetPersistingCache:
            if (!sdk.reset_cache) return result;
            return mutate(sdk.reset_cache);
        case Operation::kGetApiVersion: {
            if (!sdk.api_version) return result;
            unsigned int version{};
            const auto status = state.invoke(sdk.api_version, state.storage->context, &version);
            result.value = version;
            return complete(status);
        }
        case Operation::kGetAffinity: {
            if (!sdk.affinity) return result;
            CUexecAffinityParam parameter{};
            const auto status =
                state.invoke(sdk.affinity, &parameter, static_cast<CUexecAffinityType>(argument));
            if (status == CUDA_SUCCESS &&
                (argument != 0 || parameter.type != CU_EXEC_AFFINITY_TYPE_SM_COUNT)) {
                state.fault();
                result.status = CUDA_ERROR_UNKNOWN;
                result.original_status = CUDA_SUCCESS;
                return result;
            }
            if (status == CUDA_SUCCESS) result.value = parameter.param.smCount.val;
            return complete(status);
        }
        case Operation::kGetPriorityRange: {
            if (!sdk.priority_range) return result;
            int least{}, greatest{};
            const auto status = state.invoke(sdk.priority_range, &least, &greatest);
            result.value = std::bit_cast<std::uint32_t>(least);
            result.auxiliary = std::bit_cast<std::uint32_t>(greatest);
            return complete(status);
        }
        default:
            return result;
    }
}

CudaSessionResult CudaExecutionSession::activate() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || state.has_destroy_attempt)
        return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        (state.uses_shared_domain && state.is_preparation_active)) {
        auto result = rejected(CUDA_ERROR_NOT_READY);
        result.disposition = CudaSessionDisposition::kDeferred;
        return result;
    }
    CUcontext previous{};
    std::optional<NativeExecutionScope> configuration_scope;
    if (state.storage->domain_member) {
        const auto gate = state.storage->domain_member.configuration_status();
        if (gate != CUDA_SUCCESS && gate != CUDA_ERROR_NOT_READY) return rejected(gate);
        const auto member = state.storage->domain_member.view();
        if (member.status != CUDA_SUCCESS) return rejected(member.status);
        if (member.can_virtualize_configuration) {
            configuration_scope.emplace(state.is_native_issue_active);
            const auto observed = state.dispatch.execution.context_get_current(&previous);
            if (observed != CUDA_SUCCESS) {
                state.fault();
                return rejected(observed);
            }
        }
    }
    auto selected =
        completed(state.invoke(state.dispatch.context_set_current, state.storage->context));
    if (selected.status != CUDA_SUCCESS || !state.storage->domain_member) return selected;
    const auto configured =
        state.storage->domain_member.materialize_configuration(configuration_scope.has_value());
    if ((configured == CUDA_ERROR_NOT_READY || configured == CUDA_ERROR_OUT_OF_MEMORY) &&
        configuration_scope) {
        // A clean, unissued configuration wait must not leak this activation's
        // TLS selection into a different logical owner's next scheduling turn.
        const auto restored = state.invoke(state.dispatch.context_set_current, previous);
        if (restored != CUDA_SUCCESS || state.is_faulted) {
            state.fault();
            return rejected(restored == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : restored);
        }
    }
    if (configured == CUDA_ERROR_NOT_READY) return deferred_resource_release();
    return configured == CUDA_SUCCESS ? selected : rejected(configured);
}
CudaSessionResult CudaExecutionSession::deactivate() noexcept {
    auto& state = *implementation_;
    if (state.owns() &&
        (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)) {
        auto result = rejected(CUDA_ERROR_NOT_READY);
        result.disposition = CudaSessionDisposition::kDeferred;
        return result;
    }
    if (state.owns() && state.executor && state.executor->has_active_enqueue_turn()) {
        // An actor must restore its own thread to NULL after beginning a source
        // turn, so another retained source can be visited. This changes ONLY the
        // owner's TLS context, never the worker's context or native issue grant.
        // Keep native callbacks/reentry excluded throughout both independent proofs.
        const NativeExecutionScope scope(state.is_native_issue_active);
        const auto before = state.check_context(true, true);
        if (before != CUDA_SUCCESS) return rejected(before);
        const auto selected = state.invoke(state.dispatch.context_set_current, nullptr);
        CUcontext current{};
        const auto proof = state.invoke(state.dispatch.execution.context_get_current, &current);
        if (proof != CUDA_SUCCESS || current != nullptr) {
            state.fault();
            return rejected(proof != CUDA_SUCCESS ? proof : CUDA_ERROR_INVALID_CONTEXT);
        }
        return completed(selected);
    }
    const auto result = state.check_context();
    if (result != CUDA_SUCCESS) return rejected(result);
    return completed(state.invoke(state.dispatch.context_set_current, nullptr));
}

bool CudaExecutionSession::set_preparation_observer(CudaPreparationObserver observer) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_preparation_active) return false;
    state.preparation_observer = observer;
    return true;
}
bool CudaExecutionSession::set_retirement_observer(
    std::shared_ptr<CudaRetirementObserver> observer) noexcept {
    auto& state = *implementation_;
    return state.owns() && state.executor &&
           state.executor->set_retirement_observer(std::move(observer));
}

CudaSessionResult CudaExecutionSession::load_ptx(std::string_view source) noexcept {
    return load_ptx_jit(source, {}, nullptr);
}
CudaSessionResult CudaExecutionSession::load_ptx_jit(
    std::string_view source, std::span<const CudaModuleJitOption> options,
    std::vector<CudaModuleJitOption>* outputs) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true, true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (source.empty() || source.find('\0') != std::string_view::npos)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->modules.size() == state.limits.max_objects ||
        source.size() >= state.limits.max_image_bytes - state.storage->image_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        Implementation::Module module;
        module.source = source;
        module.source_static_bytes = state.inspect_static_storage(source);
        if (!module.source_static_bytes) return rejected(CUDA_ERROR_NOT_SUPPORTED);
        module.charged_bytes = module.source.size() + 1;
        const auto attached = state.attach_jit(module, options, outputs);
        if (attached != CUDA_SUCCESS) return rejected(attached);
        const auto prepared = state.transform_source(module);
        if (prepared != CUDA_SUCCESS) return rejected(prepared);
        if (module.transformed) {
            const auto static_bytes = state.inspect_static_storage(
                module.transformed->image, kOwnedPtxTransformLimits.output_bytes);
            if (!static_bytes) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            module.prepared_static_budget = *static_bytes;
            const auto extra = module.transformed->image.size() + 1;
            if (extra >
                state.limits.max_image_bytes - state.storage->image_bytes - module.charged_bytes)
                return rejected(CUDA_ERROR_OUT_OF_MEMORY);
            module.charged_bytes += extra;
        }
        const bool is_combined = module.transformed && !module.jit;
        const auto* image = is_combined ? module.transformed->image.c_str() : module.source.c_str();
        if (!state.reserve_static(
                is_combined ? module.prepared_static_budget : *module.source_static_bytes,
                module.native_static_bytes))
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        auto result = completed(state.load_module_preparation(
            module, &module.native, image,
            is_combined ? CudaPreparationPhase::kLoadPrepared : CudaPreparationPhase::kLoadOriginal,
            outputs));
        if (result.status != CUDA_SUCCESS && !module.native && module.transformed && !module.jit &&
            !state.is_faulted) {
            // Retry original intake only before any native handle or launch is
            // published. Partial handles and malformed success never retry.
            module.transformed.reset();
            module.charged_bytes = module.source.size() + 1;
            state.release_static(module.native_static_bytes);
            if (!state.reserve_static(*module.source_static_bytes, module.native_static_bytes))
                return rejected(CUDA_ERROR_OUT_OF_MEMORY);
            result = completed(state.load_preparation(&module.native, module.source.c_str(),
                                                      CudaPreparationPhase::kLoadOriginal));
        }
        if (result.status != CUDA_SUCCESS && !module.native && !state.is_faulted) {
            state.release_static(module.native_static_bytes);
            return result;
        }
        if (module.jit && result.status == CUDA_SUCCESS && module.native && !state.is_faulted) {
            const auto prepared = state.prepare_shared_binary_module(module);
            if (prepared != CUDA_SUCCESS) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = prepared;
            }
        }
        module.id = state.consume_id();
        const auto id = module.id;
        state.storage->image_bytes += module.charged_bytes;
        const bool is_valid = result.status == CUDA_SUCCESS && module.native && !state.is_faulted;
        state.storage->modules.push_back(std::move(module));
        if (is_valid)
            result.id = id;
        else {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            if (result.status == CUDA_SUCCESS) {
                result.status = CUDA_ERROR_UNKNOWN;
            }
        }
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}

CudaSessionResult CudaExecutionSession::load_binary(std::span<const std::byte> image) noexcept {
    return load_binary_jit(image, {}, nullptr);
}
CudaSessionResult CudaExecutionSession::load_binary_jit(
    std::span<const std::byte> image, std::span<const CudaModuleJitOption> options,
    std::vector<CudaModuleJitOption>* outputs) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true, true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (image.empty()) return rejected(CUDA_ERROR_INVALID_IMAGE);
    if (!state.has_id_space || state.storage->modules.size() == state.limits.max_objects ||
        image.size() > state.limits.max_image_bytes - state.storage->image_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    if (!state.dispatch.device_attribute) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (!state.compute_capability) {
        int major{}, minor{};
        auto status =
            state.invoke(state.dispatch.device_attribute, &major,
                         CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, state.native_device);
        if (status == CUDA_SUCCESS)
            status =
                state.invoke(state.dispatch.device_attribute, &minor,
                             CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, state.native_device);
        if (status != CUDA_SUCCESS) return rejected(status);
        if (major <= 0 || major > 99 || minor < 0 || minor > 9)
            return rejected(CUDA_ERROR_NOT_SUPPORTED);
        state.compute_capability = static_cast<unsigned int>(major * 10 + minor);
    }
    try {
        Implementation::Module module;
        module.binary.assign(image.begin(), image.end());
        transform::CudaImageLimits limits;
        limits.image_bytes = std::min(state.limits.max_image_bytes, transform::kMaxCudaModuleBytes);
        limits.parameters = state.limits.max_parameters;
        limits.parameter_bytes = state.limits.submissions.max_parameter_bytes;
        limits.work_bytes = transform::kMaxCudaModuleWorkBytes;
        limits.compact_ptx_arities = true;
        {
            const PreparationScope phase(state.preparation_observer,
                                         CudaPreparationPhase::kInspectImage,
                                         state.is_preparation_active);
            module.metadata =
                transform::inspect_cuda_module(module.binary, *state.compute_capability, limits);
        }
        if (!module.metadata) return rejected(CUDA_ERROR_INVALID_IMAGE);
        module.charged_bytes = module.binary.size();
        auto remaining =
            state.limits.max_image_bytes - state.storage->image_bytes - module.charged_bytes;
        if (module.metadata->retained_bytes > remaining) return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        module.charged_bytes += module.metadata->retained_bytes;
        remaining -= module.metadata->retained_bytes;
        if (module.metadata->ptx) {
            module.source = std::move(*module.metadata->ptx);
            module.metadata->ptx.reset();
            module.source_static_bytes = state.inspect_static_storage(module.source);
            if (module.metadata->is_ptx_selected && !module.source_static_bytes)
                return rejected(CUDA_ERROR_NOT_SUPPORTED);
            const auto prepared = state.transform_source(module);
            if (prepared != CUDA_SUCCESS) return rejected(prepared);
        }
        const auto attached = state.attach_jit(module, options, outputs);
        if (attached != CUDA_SUCCESS) return rejected(attached);
        remaining =
            state.limits.max_image_bytes - state.storage->image_bytes - module.charged_bytes;
        if (module.transformed) {
            const auto static_bytes = state.inspect_static_storage(
                module.transformed->image, kOwnedPtxTransformLimits.output_bytes);
            if (!static_bytes) return rejected(CUDA_ERROR_NOT_SUPPORTED);
            module.prepared_static_budget = *static_bytes;
            if (module.transformed->image.size() >= remaining)
                return rejected(CUDA_ERROR_OUT_OF_MEMORY);
            module.charged_bytes += module.transformed->image.size() + 1;
        }
        const auto static_bytes = module.metadata->is_ptx_selected
                                      ? *module.source_static_bytes
                                      : module.metadata->static_storage_bytes;
        if (!state.reserve_static(static_bytes, module.native_static_bytes))
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        auto result =
            completed(state.load_module_preparation(module, &module.native, module.binary.data(),
                                                    CudaPreparationPhase::kLoadOriginal, outputs));
        if (result.status != CUDA_SUCCESS && !module.native && !state.is_faulted) {
            state.release_static(module.native_static_bytes);
            return result;
        }
        if (result.status == CUDA_SUCCESS && module.native && !state.is_faulted) {
            const auto prepared = state.prepare_shared_binary_module(module);
            if (prepared != CUDA_SUCCESS) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = prepared;
            }
        }
        module.id = state.consume_id();
        const auto id = module.id;
        const bool valid = result.status == CUDA_SUCCESS && module.native && !state.is_faulted;
        state.storage->image_bytes += module.charged_bytes;
        state.storage->modules.push_back(std::move(module));
        if (valid)
            result.id = id;
        else {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            if (result.status == CUDA_SUCCESS) {
                result.status = CUDA_ERROR_UNKNOWN;
            }
        }
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}

CudaSessionResult CudaExecutionSession::get_function(std::uint64_t module_id,
                                                     std::string_view name) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true, true);
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto module = std::ranges::find_if(
        state.storage->modules, [module_id](const auto& entry) { return entry.id == module_id; });
    if (module == state.storage->modules.end() || name.empty() || name.size() > 4096 ||
        name.find('\0') != std::string_view::npos)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (module->transformed &&
        std::ranges::any_of(module->transformed->kernels,
                            [&](const auto& entry) { return entry.persistent_name == name; }))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    auto existing = std::ranges::find_if(state.storage->functions, [&](const auto& entry) {
        return entry.module == module_id && entry.name == name;
    });
    if (existing == state.storage->functions.end() &&
        (!state.has_id_space || state.storage->functions.size() == state.limits.max_objects))
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    const PreparationScope phase(state.preparation_observer,
                                 CudaPreparationPhase::kValidateFunction,
                                 state.is_preparation_active);
    std::optional<CUresult> original_result;
    try {
        Implementation::Function function;
        function.module = module_id;
        function.name = name;
        function.arguments.reserve(state.limits.max_parameters);
        auto result = completed(state.invoke(state.dispatch.function_get, &function.native,
                                             module->native, function.name.c_str()));
        original_result = result.original_status;
        if (state.is_faulted) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        if (result.status != CUDA_SUCCESS) return result;
        if (!function.native) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        if (existing != state.storage->functions.end()) {
            if (existing->native != function.native) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
            result.id = existing->id;
            result.parameter_bytes = existing->parameter_bytes;
            result.arguments = existing->arguments;
            return result;
        }
        if (std::ranges::any_of(state.storage->functions, [&](const auto& alias) {
                return alias.module == module_id && alias.native == function.native &&
                       alias.has_configuration_attempt;
            })) {
            // A newly resolved public alias cannot reconstruct earlier public
            // settings. Preparing its private family could also reset a native
            // variant shared with an already configured alias.
            function.has_configuration_attempt = true;
            function.is_persistent_disabled = true;
            function.has_unknown_configuration = true;
            function.configuration.reset();
        }
        if (!state.has_graph_capture() && state.persistent.workers &&
            !function.has_configuration_attempt) {
            const NativeExecutionScope active(state.is_native_issue_active);
            const auto proof = [&] {
                const auto status = state.check_context(true);
                if (status != CUDA_SUCCESS) state.fault();
                return status;
            };
            const auto fail_preparation = [&](CUresult status) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = status;
                return result;
            };
            Implementation::FunctionCarveout* carveout{};
            if (state.dispatch.function_get_attribute) {
                auto found = std::ranges::find(module->function_carveouts, function.native,
                                               &Implementation::FunctionCarveout::native);
                if (found == module->function_carveouts.end()) {
                    std::size_t retained{};
                    for (const auto& owner : state.storage->modules)
                        retained += owner.function_carveouts.size();
                    if (retained >= state.limits.max_objects)
                        return fail_preparation(CUDA_ERROR_OUT_OF_MEMORY);
                    // Allocate before the observer/hint. A later preparation
                    // error must not make a retry recapture our physical hint.
                    module->function_carveouts.push_back({function.native, {}, false, false});
                    carveout = &module->function_carveouts.back();
                } else
                    carveout = &*found;
            }
            if (const auto status = proof(); status != CUDA_SUCCESS)
                return fail_preparation(status);
            if (carveout && !carveout->logical_value) {
                int observed{};
                const auto status = state.invoke(state.dispatch.function_get_attribute, &observed,
                                                 CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                                 function.native);
                if (const auto checked = proof(); checked != CUDA_SUCCESS)
                    return fail_preparation(checked);
                if (status != CUDA_SUCCESS) return fail_preparation(status);
                carveout->logical_value = observed;
                carveout->is_override_active = true;
            }
            // Private broadcast storage must not trigger a partition transition
            // behind an independent native helper. Keep both function domains
            // on the same physical hint while retaining the observed logical
            // value. A public setter ends that override; retries never renew it.
            if (!carveout || !carveout->is_hint_applied) {
                const auto configured =
                    state.invoke(state.dispatch.function_set_attribute, function.native,
                                 CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                 CU_SHAREDMEM_CARVEOUT_MAX_SHARED);
                if (const auto checked = proof(); checked != CUDA_SUCCESS)
                    return fail_preparation(checked);
                if (configured != CUDA_SUCCESS) return fail_preparation(configured);
                if (carveout) carveout->is_hint_applied = true;
            }
        }
        const transform::CudaImageKernel* binary_kernel{};
        std::optional<std::size_t> count;
        if (module->metadata && !module->metadata->is_ptx_selected) {
            const auto found = std::ranges::find(module->metadata->kernels, name,
                                                 &transform::CudaImageKernel::name);
            if (found != module->metadata->kernels.end()) {
                binary_kernel = &*found;
                count = found->arguments.size();
            }
        } else
            count = parameter_count(PtxTokens{module->source}, name, state.limits.max_parameters);
        if (!count) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_NOT_SUPPORTED;
            return result;
        }
        if (module->metadata) {
            const auto alternative = std::ranges::find(module->metadata->ptx_arities, name,
                                                       &transform::CudaImagePtxArity::name);
            if (alternative != module->metadata->ptx_arities.end() &&
                (alternative->has_conflicting_counts || alternative->parameter_count != *count)) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_INVALID_VALUE;
                return result;
            }
            for (const auto& source : module->metadata->validation_ptx) {
                const auto alternative =
                    parameter_count(PtxTokens{source}, name, state.limits.max_parameters);
                if (alternative && *alternative != *count) {
                    result.disposition = CudaSessionDisposition::kRejected;
                    result.status = CUDA_ERROR_INVALID_VALUE;
                    return result;
                }
            }
        }
        for (std::size_t index = 0; index < *count; ++index) {
            std::size_t offset{}, bytes{};
            const auto status = state.invoke(state.dispatch.parameter_info, function.native, index,
                                             &offset, &bytes);
            if (state.is_faulted || status != CUDA_SUCCESS || !bytes ||
                offset < function.parameter_bytes ||
                offset > state.limits.submissions.max_parameter_bytes ||
                bytes > state.limits.submissions.max_parameter_bytes - offset ||
                (binary_kernel && (offset != binary_kernel->arguments[index].offset ||
                                   bytes != binary_kernel->arguments[index].bytes))) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = state.is_faulted         ? CUDA_ERROR_UNKNOWN
                                : status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE
                                                         : status;
                return result;
            }
            function.arguments.push_back({offset, bytes});
            function.parameter_bytes = offset + bytes;
        }
        if (!state.has_graph_capture() && module->transformed && !function.is_persistent_disabled &&
            !module->binary.empty() && !module->has_preparation_attempt) {
            module->has_preparation_attempt = true;
            const auto status =
                state.reserve_static(module->prepared_static_budget, module->prepared_static_bytes)
                    ? state.load_module_preparation(*module, &module->prepared_native,
                                                    module->transformed->image.c_str(),
                                                    CudaPreparationPhase::kLoadPrepared)
                    : CUDA_ERROR_OUT_OF_MEMORY;
            if (state.is_faulted) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
            if (status != CUDA_SUCCESS || !module->prepared_native) {
                if (!module->prepared_native && status != CUDA_SUCCESS) {
                    state.release_static(module->prepared_static_bytes);
                    // Optional JIT failed before returning any private handle.
                    // Original module/function stay native; never retry this JIT.
                    const auto released = module->transformed->image.size() + 1;
                    module->charged_bytes -= released;
                    state.storage->image_bytes -= released;
                    module->transformed.reset();
                } else {
                    state.fault();
                    result.disposition = CudaSessionDisposition::kRejected;
                    result.status = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
                    return result;
                }
            }
        }
        if (!state.has_graph_capture() && module->transformed && !function.is_persistent_disabled) {
            for (const auto& variant : module->transformed->kernels) {
                if (variant.original_name != name) continue;
                const auto status = state.prepare_persistent_function(function, variant);
                if (status != CUDA_SUCCESS) {
                    result.disposition = CudaSessionDisposition::kRejected;
                    result.status = status;
                    return result;
                }
            }
        }
        if (state.profile_cache && !function.has_configuration_attempt)
            function.profile = state.profile_cache->bind(
                state.profile_platform,
                module->profile_source.empty() ? module->source : module->profile_source,
                module->binary, function.name);
        function.id = state.next_id;
        function.module = module_id;
        if (!state.executor->bind_function(function.id, function.native, function.parameter_bytes,
                                           state.storage)) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        result.id = state.consume_id();
        state.storage->functions.push_back(std::move(function));
        result.parameter_bytes = state.storage->functions.back().parameter_bytes;
        result.arguments = state.storage->functions.back().arguments;
        return result;
    } catch (...) {
        auto result = rejected(CUDA_ERROR_OUT_OF_MEMORY);
        result.original_status = original_result;
        return result;
    }
}

CudaPointerMetadataResult CudaExecutionSession::pointer_metadata(
    const CudaPointerQuery& query, CudaExecutionSession* querying, CudaExecutionSession* source,
    std::optional<CudaHostMemoryView> host) noexcept {
    auto& carrier = *implementation_;
    CudaPointerMetadataResult result;
    const auto reject = [&](CUresult status) {
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = status;
        result.observation.reset();
        return result;
    };
    std::array<Implementation*, 3> owners{};
    std::size_t count{};
    for (auto* candidate :
         std::array{&carrier, querying ? querying->implementation_.get() : nullptr,
                    source ? source->implementation_.get() : nullptr}) {
        if (!candidate || std::ranges::find(owners, candidate) != owners.end()) continue;
        if (!candidate->owns() || candidate->has_destroy_attempt)
            return reject(CUDA_ERROR_INVALID_CONTEXT);
        if (candidate->is_native_issue_active || candidate->is_folded_active ||
            candidate->is_rnn_query_active) {
            result.disposition = CudaSessionDisposition::kDeferred;
            result.status = CUDA_ERROR_NOT_READY;
            return result;
        }
        if (candidate->is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
        if (candidate->is_faulted ||
            (candidate->executor && candidate->executor->snapshot().is_quarantined))
            return reject(CUDA_ERROR_UNKNOWN);
        owners[count++] = candidate;
    }
    std::array<std::optional<NativeExecutionScope>, 3> scopes;
    for (std::size_t index = 0; index < count; ++index)
        scopes[index].emplace(owners[index]->is_native_issue_active);
    const auto fault = [&](CUresult status) {
        result.context_selection = CudaPointerContextSelection::kUnproven;
        for (std::size_t index = 0; index < count; ++index) owners[index]->fault();
        return reject(status);
    };
    const bool is_driver = query.domain == CudaPointerDomain::kDriver;
    if ((!is_driver && query.domain != CudaPointerDomain::kRuntime) ||
        (is_driver ? (query.selector != 3 && query.selector != 9 &&
                      !(query.selector == 1 && query.kind == CudaPointerKind::kProcessDevice))
                   : query.selector != 0))
        return reject(CUDA_ERROR_INVALID_VALUE);
    if (querying &&
        (!querying->implementation_->storage->context || !querying->implementation_->executor))
        return reject(CUDA_ERROR_INVALID_CONTEXT);
    CUdeviceptr input{}, device_alias{}, host_alias{};
    std::shared_ptr<const void> process_input;
    switch (query.kind) {
        case CudaPointerKind::kNull:
        case CudaPointerKind::kUnregistered:
            if (source || host || query.range.allocation || query.range.offset)
                return reject(CUDA_ERROR_INVALID_VALUE);
            if (query.kind == CudaPointerKind::kUnregistered)
                input = reinterpret_cast<CUdeviceptr>(&carrier.pointer_metadata_sentinel);
            host_alias = input;
            break;
        case CudaPointerKind::kDevice:
        case CudaPointerKind::kProcessDevice: {
            if (!source || host) return reject(CUDA_ERROR_INVALID_VALUE);
            auto& owner = *source->implementation_;
            const bool is_process = query.kind == CudaPointerKind::kProcessDevice;
            if (is_process
                    ? (!owner.process_memory || !(query.range.allocation & (UINT64_C(1) << 63)))
                    : (!owner.storage->context || !owner.executor))
                return reject(CUDA_ERROR_INVALID_CONTEXT);
            const auto found = owner.find_allocation(query.range.allocation);
            if (!found || query.range.offset >= found->bytes ||
                query.range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address ||
                (found->module && !std::ranges::any_of(owner.storage->modules, [&](const auto& m) {
                     return m.id == found->module && m.native;
                 })))
                return reject(CUDA_ERROR_INVALID_VALUE);
            input = device_alias = found->address + query.range.offset;
            if (is_process) {
                auto borrow = owner.process_memory->owner()->borrow(
                    query.range.allocation & ~(UINT64_C(1) << 63), query.range.offset, 1);
                if (borrow.status != CUDA_SUCCESS) return reject(borrow.status);
                process_input = std::move(borrow.lifetime);
            }
            break;
        }
        case CudaPointerKind::kHost:
            if (source || query.range.allocation || query.range.offset || !host || !host->pointer ||
                !host->address || !host->bytes || !host->lifetime)
                return reject(CUDA_ERROR_INVALID_VALUE);
            input = host_alias = reinterpret_cast<CUdeviceptr>(host->pointer);
            device_alias = host->address;
            break;
        default:
            return reject(CUDA_ERROR_INVALID_VALUE);
    }
    const auto& sdk = carrier.dispatch;
    if ((is_driver && !sdk.pointer_get_attribute) ||
        (!is_driver && (!sdk.runtime.pointer_attributes || !sdk.runtime.version)))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    // The initial context is deliberately not inferred from any resource owner.
    // cuInit may be the first Driver operation; NOT_INITIALIZED is a permitted
    // initial observation, not permission to skip all later context proofs.
    CUcontext before{};
    auto status = sdk.execution.context_get_current(&before);
    const bool was_uninitialized = status == CUDA_ERROR_NOT_INITIALIZED;
    if (status != CUDA_SUCCESS && !was_uninitialized) return fault(status);
    if (was_uninitialized && before) return fault(CUDA_ERROR_UNKNOWN);
    const auto initialized = sdk.initialize(0);
    CUcontext after{};
    status = sdk.execution.context_get_current(&after);
    if (initialized != CUDA_SUCCESS && was_uninitialized && status == CUDA_ERROR_NOT_INITIALIZED &&
        !after)
        return reject(initialized);
    if (status != CUDA_SUCCESS) return fault(status);
    if (after != before) return fault(CUDA_ERROR_INVALID_CONTEXT);
    if (initialized != CUDA_SUCCESS) return reject(initialized);
    const auto prove = [&](CUcontext expected) {
        CUcontext current{};
        const auto observed = sdk.execution.context_get_current(&current);
        return observed != CUDA_SUCCESS ? observed
               : current != expected    ? CUDA_ERROR_INVALID_CONTEXT
                                        : CUDA_SUCCESS;
    };
    if (const auto observed = prove(before); observed != CUDA_SUCCESS) return fault(observed);
    const auto desired = querying ? querying->implementation_->storage->context : nullptr;
    const auto selected = sdk.context_set_current(desired);
    if (const auto observed = prove(selected == CUDA_SUCCESS ? desired : before);
        observed != CUDA_SUCCESS)
        return fault(observed);
    if (selected != CUDA_SUCCESS) return reject(selected);
    result.context_selection = CudaPointerContextSelection::kQuery;
    if (!is_driver) {
        if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
        int version{};
        const auto version_status = sdk.runtime.version(&version);
        if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
        if (version_status != cudaSuccess || version != 12060 || CUDART_VERSION != 12060 ||
            sizeof(void*) != 8)
            return reject(CUDA_ERROR_NOT_SUPPORTED);
    }
    CUdeviceptr poison = std::numeric_limits<CUdeviceptr>::max();
    while (!poison || poison == input || poison == device_alias || poison == host_alias) --poison;
    constexpr auto scalar_poison = std::numeric_limits<std::int32_t>::min();
    static_assert(sizeof(int) == sizeof(std::int32_t));
    static_assert(sizeof(cudaMemoryType) == sizeof(std::int32_t));
    CUdeviceptr device_value = poison;
    int ordinal = scalar_poison;
    cudaPointerAttributes attributes{};
    std::memcpy(&attributes.type, &scalar_poison, sizeof(scalar_poison));
    attributes.device = scalar_poison;
    // These deliberately invalid output sentinels are compared as bits only;
    // neither the framework nor the original metadata getter dereferences them.
    // NOLINTBEGIN(performance-no-int-to-ptr)
    attributes.devicePointer = reinterpret_cast<void*>(poison);
    attributes.hostPointer = reinterpret_cast<void*>(poison);
    // NOLINTEND(performance-no-int-to-ptr)
    if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
    if (is_driver) {
        void* output = query.selector == 3 || query.selector == 1
                           ? static_cast<void*>(&device_value)
                           : static_cast<void*>(&ordinal);
        result.original_status = static_cast<std::int32_t>(sdk.pointer_get_attribute(
            output, static_cast<CUpointer_attribute>(query.selector), input));
    } else {
        // The Runtime ABI takes an address instead of Driver CUdeviceptr bits.
        // This validated source/sentinel is observed, never read as a pointee.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto* runtime_input = reinterpret_cast<const void*>(input);
        result.original_status =
            static_cast<std::int32_t>(sdk.runtime.pointer_attributes(&attributes, runtime_input));
    }
    if (const auto observed = prove(desired); observed != CUDA_SUCCESS) return fault(observed);
    CudaPointerObservation observation;
    enum class PointerField : std::uint32_t { kDevice = 4, kHost = 8 };
    const auto project_alias = [&](CUdeviceptr value, CUdeviceptr expected, PointerField effect,
                                   CudaPointerAlias& output) {
        if (value == poison) return true;
        if (value && (!expected || value != expected)) return false;
        observation.effects |= static_cast<std::uint32_t>(effect);
        output = value ? CudaPointerAlias::kInput : CudaPointerAlias::kNull;
        return true;
    };
    const bool succeeded = *result.original_status == 0;
    if (is_driver && query.selector == 1) {
        if (!project_alias(device_value, 0, PointerField::kDevice, observation.device_alias) ||
            (succeeded && observation.effects != 4))
            return fault(CUDA_ERROR_UNKNOWN);
    } else if (is_driver && query.selector == 3) {
        if (!project_alias(device_value, device_alias, PointerField::kDevice,
                           observation.device_alias) ||
            (succeeded &&
             (observation.effects != 4 || observation.device_alias != CudaPointerAlias::kInput)))
            return fault(CUDA_ERROR_UNKNOWN);
    } else if (is_driver) {
        if (ordinal != scalar_poison) {
            observation.effects = 2;
            observation.device = ordinal;
        }
        if (succeeded && (observation.effects != 2 || ordinal < 0 ||
                          (query.kind != CudaPointerKind::kDevice &&
                           query.kind != CudaPointerKind::kProcessDevice &&
                           query.kind != CudaPointerKind::kHost)))
            return fault(CUDA_ERROR_UNKNOWN);
    } else {
        std::int32_t type{};
        std::memcpy(&type, &attributes.type, sizeof(type));
        if (type != scalar_poison) {
            observation.effects |= 1;
            observation.type = type;
        }
        if (attributes.device != scalar_poison) {
            observation.effects |= 2;
            observation.device = attributes.device;
        }
        if (!project_alias(reinterpret_cast<CUdeviceptr>(attributes.devicePointer), device_alias,
                           PointerField::kDevice, observation.device_alias) ||
            !project_alias(reinterpret_cast<CUdeviceptr>(attributes.hostPointer), host_alias,
                           PointerField::kHost, observation.host_alias))
            return fault(CUDA_ERROR_UNKNOWN);
        const int expected_type =
            query.kind == CudaPointerKind::kDevice || query.kind == CudaPointerKind::kProcessDevice
                ? 2
            : query.kind == CudaPointerKind::kHost ? 1
                                                   : 0;
        if (succeeded &&
            (observation.effects != 15 || type != expected_type ||
             (expected_type ? attributes.device < 0 : attributes.device != -2) ||
             (!expected_type && reinterpret_cast<CUdeviceptr>(attributes.hostPointer) != input)))
            return fault(CUDA_ERROR_UNKNOWN);
    }
    result.disposition = CudaSessionDisposition::kComplete;
    result.status = CUDA_SUCCESS;
    result.observation = observation;
    return result;
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): distinct wire fields retain their typed ABI.
CudaFunctionControlResult CudaExecutionSession::function_control(CudaFunctionControl operation,
                                                                 std::uint64_t function_id,
                                                                 std::uint32_t argument,
                                                                 std::int32_t value) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    CudaFunctionControlResult result;
    const auto reject = [&](CUresult status) {
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = status;
        result.value = 0;
        result.attributes.reset();
        return result;
    };
    if (!state.owns() || !state.storage->context || state.has_destroy_attempt)
        return reject(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active) {
        result.disposition = CudaSessionDisposition::kDeferred;
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    if (operation == CudaFunctionControl::kGetAttributes && (argument != 0 || value != 0))
        return reject(CUDA_ERROR_INVALID_VALUE);
    // Acquire exclusion before even the first context observation: an injected
    // Driver callback must not destroy owners or consume another queued call.
    const NativeExecutionScope active(state.is_native_issue_active);
    const auto proof = [&] {
        const auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    if (state.is_closing) return reject(CUDA_ERROR_DEINITIALIZED);
    if (state.is_faulted || !state.executor || state.executor->snapshot().is_quarantined)
        return reject(CUDA_ERROR_UNKNOWN);
    using Operation = CudaFunctionControl;
    const bool is_query =
        operation == Operation::kGetAttribute || operation == Operation::kGetAttributes;
    if ((operation != Operation::kGetAttribute && operation != Operation::kSetAttribute &&
         operation != Operation::kSetCache && operation != Operation::kGetAttributes) ||
        (operation == Operation::kSetCache ? argument > 3 : argument > 31) ||
        (operation != Operation::kSetAttribute && value != 0))
        return reject(CUDA_ERROR_INVALID_VALUE);
    const auto function =
        std::ranges::find(state.storage->functions, function_id, &Implementation::Function::id);
    if (function == state.storage->functions.end() || !function->native ||
        !std::ranges::any_of(state.storage->modules, [&](const auto& module) {
            return module.id == function->module && module.native;
        }))
        return reject(CUDA_ERROR_INVALID_HANDLE);
    if ((is_query && !state.dispatch.function_get_attribute) ||
        (operation == Operation::kSetAttribute && !state.dispatch.function_set_attribute) ||
        (operation == Operation::kSetCache && !state.dispatch.function_set_cache))
        return reject(CUDA_ERROR_NOT_SUPPORTED);
    // The outer scope already owns cold_gate's exclusion. Only mutations need
    // its physical-quiescence condition; a getter never drains pending work.
    if (!is_query && !state.executor->resources_quiescent()) {
        result.disposition = CudaSessionDisposition::kDeferred;
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    if (operation == Operation::kGetAttributes) {
        constexpr std::array attributes{CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                                        CU_FUNC_ATTRIBUTE_NUM_REGS,
                                        CU_FUNC_ATTRIBUTE_PTX_VERSION,
                                        CU_FUNC_ATTRIBUTE_BINARY_VERSION,
                                        CU_FUNC_ATTRIBUTE_CACHE_MODE_CA,
                                        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                        CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                        CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET,
                                        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
                                        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT,
                                        CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH,
                                        CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE,
                                        CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED};
        CudaFunctionAttributes observed;
        for (std::size_t index = 0; index < attributes.size(); ++index) {
            // The initial proof covers the first callback. Later callbacks also
            // have their own pre-proof; the whole aggregate holds one exclusion.
            if (index != 0)
                if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
            int field{};
            result.original_status = state.invoke(state.dispatch.function_get_attribute, &field,
                                                  attributes[index], function->native);
            result.disposition = CudaSessionDisposition::kComplete;
            result.status = CUDA_SUCCESS;
            if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
            if (*result.original_status != CUDA_SUCCESS) return result;
            if (index < observed.sizes.size()) {
                if (field < 0) {
                    state.fault();
                    return reject(CUDA_ERROR_UNKNOWN);
                }
                observed.sizes[index] = static_cast<std::uint64_t>(field);
            } else {
                observed.values[index - observed.sizes.size()] = field;
            }
        }
        const auto module = std::ranges::find(state.storage->modules, function->module,
                                              &Implementation::Module::id);
        const auto carveout = std::ranges::find(module->function_carveouts, function->native,
                                                &Implementation::FunctionCarveout::native);
        if (carveout != module->function_carveouts.end() && carveout->is_override_active &&
            carveout->logical_value)
            observed.values[6] = *carveout->logical_value;
        result.attributes = observed;
        return result;
    }
    std::vector<Implementation::Function*> aliases;
    std::vector<CUfunction> private_targets;
    bool is_persistent_disabled{};
    if (!is_query) {
        try {
            // Existing owner/variant storage bounds these temporary lists.
            // Finish every allocation before the requested native mutation.
            std::size_t alias_count{}, private_count{};
            for (const auto& alias : state.storage->functions) {
                if (alias.module != function->module || alias.native != function->native) continue;
                if (alias_count == aliases.max_size() ||
                    alias.variants.size() > private_targets.max_size() - private_count)
                    return reject(CUDA_ERROR_OUT_OF_MEMORY);
                ++alias_count;
                private_count += alias.variants.size();
                is_persistent_disabled = is_persistent_disabled || alias.is_persistent_disabled;
            }
            aliases.reserve(alias_count);
            if (!is_persistent_disabled) private_targets.reserve(private_count);
            for (auto& alias : state.storage->functions) {
                if (alias.module != function->module || alias.native != function->native) continue;
                aliases.push_back(&alias);
                if (is_persistent_disabled) continue;
                for (const auto& variant : alias.variants) {
                    if (!variant.native) {
                        state.fault();
                        return reject(CUDA_ERROR_UNKNOWN);
                    }
                    if (variant.native != function->native &&
                        std::ranges::find(private_targets, variant.native) == private_targets.end())
                        private_targets.push_back(variant.native);
                }
            }
        } catch (...) {
            return reject(CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    const auto disable_private = [&] {
        is_persistent_disabled = true;
        for (auto* alias : aliases) alias->is_persistent_disabled = true;
    };
    const auto configure = [&](CUfunction native) {
        return operation == Operation::kSetAttribute
                   ? state.invoke(state.dispatch.function_set_attribute, native,
                                  static_cast<CUfunction_attribute>(argument), value)
                   : state.invoke(state.dispatch.function_set_cache, native,
                                  static_cast<CUfunc_cache>(argument));
    };
    int observed{};
    if (!is_query) {
        if (state.control_epoch == UINT64_MAX) return reject(CUDA_ERROR_OUT_OF_MEMORY);
        ++state.control_epoch;
        for (auto* alias : aliases) {
            alias->has_configuration_attempt = true;
            alias->configuration.reset();
        }
        if (is_persistent_disabled) disable_private();
    }
    result.original_status =
        is_query ? state.invoke(state.dispatch.function_get_attribute, &observed,
                                static_cast<CUfunction_attribute>(argument), function->native)
                 : configure(function->native);
    result.disposition = CudaSessionDisposition::kComplete;
    result.status = CUDA_SUCCESS;
    if (!is_query && *result.original_status != CUDA_SUCCESS) disable_private();
    if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
    if (*result.original_status != CUDA_SUCCESS) return result;
    const auto module =
        std::ranges::find(state.storage->modules, function->module, &Implementation::Module::id);
    const auto carveout = std::ranges::find(module->function_carveouts, function->native,
                                            &Implementation::FunctionCarveout::native);
    if (is_query) {
        result.value = argument == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
                               carveout != module->function_carveouts.end() &&
                               carveout->is_override_active && carveout->logical_value
                           ? *carveout->logical_value
                           : observed;
        return result;
    }
    if (operation == Operation::kSetAttribute &&
        argument == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
        carveout != module->function_carveouts.end())
        carveout->is_override_active = false;
    // An alias may lack complete historical configuration. A later setter
    // cannot restore that proof for the group, even if this setting succeeds.
    if (is_persistent_disabled) return result;
    for (const auto native : private_targets) {
        const auto configured = configure(native);
        if (configured != CUDA_SUCCESS) disable_private();
        if (const auto status = proof(); status != CUDA_SUCCESS) return reject(status);
        if (configured != CUDA_SUCCESS) {
            // Preserve requested original success. Future admission uses the
            // original function, never a partly configured private replacement.
            return result;
        }
    }
    if (operation == Operation::kSetCache)
        for (auto* alias : aliases) alias->cache_preference = argument + 1;
    if (operation == Operation::kSetAttribute && argument >= 16)
        for (auto* alias : aliases) alias->has_unknown_configuration = true;
    return result;
}

CudaSessionResult CudaExecutionSession::unload_module(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto found = std::ranges::find_if(state.storage->modules,
                                            [id](const auto& entry) { return entry.id == id; });
    if (found == state.storage->modules.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (state.has_external_module(id)) return deferred_resource_release();
    for (const auto& function : state.storage->functions)
        if (function.module == id && !state.executor->unbind_function(function.id)) {
            state.fault();
            return rejected(CUDA_ERROR_UNKNOWN);
        }
    if (found->prepared_native) {
        const auto status = state.invoke(state.dispatch.module_unload, found->prepared_native);
        if (status != CUDA_SUCCESS) {
            state.fault();
            return rejected(status);  // Original target unload was not called.
        }
        found->prepared_native = nullptr;
        state.release_static(found->prepared_static_bytes);
    }
    auto result = completed(state.invoke(state.dispatch.module_unload, found->native));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    std::erase_if(state.storage->functions,
                  [id](const auto& function) { return function.module == id; });
    std::erase_if(state.storage->allocations,
                  [&](const auto& allocation) { return allocation.module == id; });
    state.release_static(found->native_static_bytes);
    state.storage->image_bytes -= found->charged_bytes;
    state.storage->modules.erase(found);
    return result;
}

CudaSessionResult CudaExecutionSession::get_global(std::uint64_t module_id,
                                                   std::string_view name) noexcept {
    auto& state = *implementation_;
    const auto context = state.check_context();
    if (context != CUDA_SUCCESS) return rejected(context);
    if (state.is_faulted || state.is_closing || !state.executor ||
        state.executor->snapshot().is_quarantined)
        return rejected(CUDA_ERROR_UNKNOWN);
    if (!state.dispatch.module_global) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto module =
        std::ranges::find(state.storage->modules, module_id, &Implementation::Module::id);
    if (module == state.storage->modules.end() || name.empty() || name.size() > 4096 ||
        name.find('\0') != std::string_view::npos)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    try {
        const std::string symbol{name};
        CUdeviceptr address{};
        std::size_t bytes{};
        auto result = completed(state.invoke(state.dispatch.module_global, &address, &bytes,
                                             module->native, symbol.c_str()));
        const auto after = state.check_context();
        if (after != CUDA_SUCCESS) {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = after;
            return result;
        }
        if (result.status != CUDA_SUCCESS) return result;
        if (!address || !bytes || bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - address) {
            state.fault();
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        }
        auto existing =
            std::ranges::find_if(state.storage->allocations, [&](const auto& allocation) {
                return allocation.address == address && allocation.bytes == bytes &&
                       allocation.module == module_id;
            });
        if (existing != state.storage->allocations.end()) {
            result.id = existing->id;
        } else {
            const bool overlap =
                std::ranges::any_of(state.storage->allocations, [&](const auto& allocation) {
                    return address >= allocation.address
                               ? address - allocation.address < allocation.bytes
                               : allocation.address - address < bytes;
                });
            if (overlap) {
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_NOT_SUPPORTED;
                return result;
            }
            const auto maximum = std::numeric_limits<std::size_t>::max();
            bool has_overflow = bytes > maximum - module->discovered_static_bytes;
            module->discovered_static_bytes =
                has_overflow ? maximum : module->discovered_static_bytes + bytes;
            const auto new_charge =
                std::max(module->native_static_bytes, module->discovered_static_bytes);
            const auto extra = new_charge - module->native_static_bytes;
            if (!state.reserve_domain_bytes(extra)) {
                if (state.storage->domain_member)
                    state.storage->domain_member.retain_uncertain_device_bytes(extra);
                if (state.process_memory)
                    (void)state.process_memory->owner()->quarantine(CUDA_ERROR_OUT_OF_MEMORY);
                has_overflow = true;
            }
            const auto used = state.storage->device_bytes;
            has_overflow |= extra > maximum - used;
            state.storage->device_bytes = extra > maximum - used ? maximum : used + extra;
            module->native_static_bytes = new_charge;
            if (!state.has_id_space ||
                state.storage->allocations.size() == state.limits.max_objects) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_OUT_OF_MEMORY;
                return result;
            }
            const auto id = state.consume_id();
            state.storage->allocations.push_back({id, address, bytes, module_id});
            if (has_overflow || extra > state.limits.max_device_bytes -
                                            std::min(used, state.limits.max_device_bytes)) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_OUT_OF_MEMORY;
                return result;
            }
            result.id = id;
        }
        result.address = address;
        result.parameter_bytes = bytes;
        return result;
    } catch (const std::bad_alloc&) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    } catch (...) {
        state.fault();
        return rejected(CUDA_ERROR_UNKNOWN);
    }
}
CudaSessionResult CudaExecutionSession::allocate(std::size_t bytes) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (!bytes) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->allocations.size() == state.limits.max_objects ||
        bytes > state.limits.max_device_bytes - state.storage->device_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    if (!state.reserve_domain_bytes(bytes)) return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    Implementation::Allocation allocation{.id = state.next_id, .address = 0, .bytes = bytes};
    auto result = completed(state.invoke(state.dispatch.allocate, &allocation.address, bytes));
    if (result.status != CUDA_SUCCESS && !allocation.address &&
        (!state.uses_shared_domain || state.clean_shared_allocation_rejection(result.status))) {
        state.release_domain_bytes(bytes);
        return result;
    }
    const bool is_valid = result.status == CUDA_SUCCESS && !state.is_faulted &&
                          allocation.address &&
                          bytes - 1 <= std::numeric_limits<CUdeviceptr>::max() - allocation.address;
    (void)state.consume_id();
    state.storage->device_bytes += bytes;
    state.storage->allocations.push_back(allocation);
    if (is_valid) {
        result.id = allocation.id;
        result.address = allocation.address;
    } else {
        state.fault();
        if (result.status == CUDA_SUCCESS) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
        }
    }
    return result;
}
CudaSessionResult CudaExecutionSession::release(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (state.process_memory && (id & (UINT64_C(1) << 63))) {
        const auto observed = state.process_memory->release(id);
        auto result = observed.original_status
                          ? completed(static_cast<CUresult>(*observed.original_status))
                          : rejected(observed.status);
        result.status = observed.status;
        if (observed.status == CUDA_ERROR_NOT_READY && !observed.original_status)
            result.disposition = CudaSessionDisposition::kDeferred;
        if (observed.is_quarantined) state.fault();
        return result;
    }
    const auto found = std::ranges::find_if(state.storage->allocations,
                                            [id](const auto& entry) { return entry.id == id; });
    if (found == state.storage->allocations.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (found->module) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (state.has_external_allocation(id)) return deferred_resource_release();
    auto result = completed(state.invoke(state.dispatch.release, found->address));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    state.storage->device_bytes -= found->bytes;
    state.release_domain_bytes(found->bytes);
    state.storage->allocations.erase(found);
    return result;
}
CudaSessionResult CudaExecutionSession::upload(CudaAllocationRange range,
                                               std::span<const std::byte> bytes) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto found = state.find_allocation(range.allocation);
    if (!found || range.offset > found->bytes || bytes.size() > found->bytes - range.offset ||
        (!bytes.empty() && !bytes.data()))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (bytes.empty()) return settled();
    if (state.storage->default_proxy) {
        if (bytes.size() > state.stream_config.maximum_staging_bytes - state.storage->staging_bytes)
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        try {
            // Page-locked/pageable SDK staging may outlive its host return.
            // Own the exact bytes until the real legacy-to-proxy tail stops.
            auto staging = std::make_shared<Implementation::Staging>(state.storage, bytes.size());
            if (state.process_memory && (range.allocation & (UINT64_C(1) << 63))) {
                const auto view = device_memory(range, bytes.size());
                if (!view) return rejected(CUDA_ERROR_INVALID_VALUE);
                staging->process_lifetime = view->lifetime;
            }
            std::copy(bytes.begin(), bytes.end(), staging->data.get());
            const auto address = found->address + range.offset;
            const std::function<CUresult()> call = [sdk = state.dispatch.upload, address, staging] {
                return sdk(address, staging->data.get(), staging->bytes);
            };
            const NativeExecutionScope scope(state.is_native_issue_active);
            if (!state.reserve_domain_physical()) return rejected(CUDA_ERROR_UNKNOWN);
            const auto observed = state.executor->execute_synchronous_legacy(call, staging);
            state.update_domain_physical();
            auto result = observed.original_status ? completed(*observed.original_status)
                                                   : rejected(observed.status);
            result.frontier = observed.frontier;
            if (observed.status != CUDA_SUCCESS) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = observed.status;
            }
            return result;
        } catch (...) {
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    return completed(state.invoke(state.dispatch.upload, found->address + range.offset,
                                  bytes.data(), bytes.size()));
}
CudaSessionResult CudaExecutionSession::download(CudaAllocationRange range,
                                                 std::span<std::byte> bytes) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const auto found = state.find_allocation(range.allocation);
    if (!found || range.offset > found->bytes || bytes.size() > found->bytes - range.offset ||
        (!bytes.empty() && !bytes.data()))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (bytes.empty()) return settled();
    if (state.storage->default_proxy) {
        if (bytes.size() > state.stream_config.maximum_staging_bytes - state.storage->staging_bytes)
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        try {
            auto staging = std::make_shared<Implementation::Staging>(state.storage, bytes.size());
            if (state.process_memory && (range.allocation & (UINT64_C(1) << 63))) {
                const auto view = device_memory(range, bytes.size());
                if (!view) return rejected(CUDA_ERROR_INVALID_VALUE);
                staging->process_lifetime = view->lifetime;
            }
            const auto address = found->address + range.offset;
            const std::function<CUresult()> call = [sdk = state.dispatch.download, address,
                                                    staging] {
                return sdk(staging->data.get(), address, staging->bytes);
            };
            const NativeExecutionScope scope(state.is_native_issue_active);
            if (!state.reserve_domain_physical()) return rejected(CUDA_ERROR_UNKNOWN);
            const auto observed = state.executor->execute_synchronous_legacy(call, staging);
            state.update_domain_physical();
            auto result = observed.original_status ? completed(*observed.original_status)
                                                   : rejected(observed.status);
            result.frontier = observed.frontier;
            if (observed.status != CUDA_SUCCESS) {
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = observed.status;
            } else if (observed.original_status == CUDA_SUCCESS) {
                // The requested DtoH call itself supplies host-completion;
                // the private event preserves subsequent stream ordering.
                std::copy_n(staging->data.get(), staging->bytes, bytes.begin());
            }
            return result;
        } catch (...) {
            return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        }
    }
    return completed(state.invoke(state.dispatch.download, bytes.data(),
                                  found->address + range.offset, bytes.size()));
}

std::optional<CudaDeviceMemoryView> CudaExecutionSession::device_memory(
    CudaAllocationRange range, std::size_t bytes) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || state.is_faulted || state.is_closing)
        return std::nullopt;
    const auto found = state.find_allocation(range.allocation);
    if (!found || range.offset > found->bytes || bytes > found->bytes - range.offset ||
        range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
        return std::nullopt;
    if (state.process_memory && (range.allocation & (UINT64_C(1) << 63))) {
        auto captured = state.process_memory->owner()->borrow(
            range.allocation & ~(UINT64_C(1) << 63), range.offset, bytes);
        if (captured.status != CUDA_SUCCESS) return std::nullopt;
        return CudaDeviceMemoryView{captured.address, bytes, std::move(captured.lifetime)};
    }
    return CudaDeviceMemoryView{found->address + range.offset, bytes, state.storage};
}

CudaSessionExternalContextCapture CudaExecutionSession::capture_external_context() noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return {.status = status};
    std::erase_if(state.external_contexts, [](const auto& entry) { return entry.expired(); });
    if (state.external_contexts.size() >= state.limits.max_objects)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        CudaSessionExternalContext context;
        context.state_ = std::make_shared<const CudaSessionExternalContext::State>(
            CudaSessionExternalContext::State{state.context_id, state.storage->context,
                                              state.device_ordinal, state.process, state.storage});
        state.external_contexts.push_back(context.state_);
        return {.status = CUDA_SUCCESS, .context = std::move(context)};
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}

CudaSessionExternalCommandCapture CudaExecutionSession::capture_external_command(
    std::uint64_t stream, std::span<const CudaSessionExternalRangeRequest> ranges) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (const auto status = state.check_context(true); status != CUDA_SUCCESS)
        return {.status = status};
    if (ranges.size() > state.limits.max_parameters || (!ranges.empty() && !ranges.data()))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    CUstream native_stream = state.storage->default_proxy;
    if (stream) {
        const auto found =
            std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
        if (found == state.storage->streams.end()) return {.status = CUDA_ERROR_INVALID_HANDLE};
        native_stream = found->native;
    }
    std::erase_if(state.external_contexts, [](const auto& entry) { return entry.expired(); });
    std::erase_if(state.external_commands, [](const auto& entry) { return entry.expired(); });
    if (state.external_contexts.size() >= state.limits.max_objects ||
        state.external_commands.size() >= state.limits.submissions.max_commands)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    try {
        auto command = std::make_shared<CudaSessionExternalCommand::State>();
        command->stream_id = stream;
        command->stream = native_stream;
        command->ranges.reserve(ranges.size());
        command->parent_modules.reserve(ranges.size());
        for (const auto& request : ranges) {
            const auto found = state.find_allocation(request.range.allocation);
            if (!found || request.range.offset > found->bytes ||
                request.bytes > found->bytes - request.range.offset ||
                request.range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
                return {.status = CUDA_ERROR_INVALID_VALUE};
            const auto address = found->address + request.range.offset;
            if (request.bytes &&
                request.bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - address)
                return {.status = CUDA_ERROR_INVALID_VALUE};
            command->ranges.push_back({request.range, address, request.bytes});
            if (state.process_memory && (request.range.allocation & (UINT64_C(1) << 63))) {
                const auto view = device_memory(request.range, request.bytes);
                if (!view) return {.status = CUDA_ERROR_INVALID_VALUE};
                command->process_lifetimes.push_back(view->lifetime);
            }
            if (found->module) command->parent_modules.push_back(found->module);
        }
        command->context.state_ = std::make_shared<const CudaSessionExternalContext::State>(
            CudaSessionExternalContext::State{state.context_id, state.storage->context,
                                              state.device_ordinal, state.process, state.storage});
        // Capacity is reserved at construction. Publish only after all resource
        // checks/allocation succeed; the two weak insertions cannot allocate.
        state.external_contexts.push_back(command->context.state_);
        state.external_commands.push_back(command);
        CudaSessionExternalCommand capsule;
        capsule.state_ = std::move(command);
        return {.status = CUDA_SUCCESS, .command = std::move(capsule)};
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}

CudaExecutionExternalAcceptance CudaExecutionSession::accept_external_barrier(
    std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_external_barrier(stream, lifetime);
}
CudaExecutionExternalCohortAcceptance CudaExecutionSession::accept_external_cohort(
    std::span<const std::uint64_t> streams, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_external_cohort(streams, lifetime);
}
CUresult CudaExecutionSession::external_barrier_ready(
    const CudaExecutionExternalBarrier& barrier) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    return state.executor->external_barrier_ready(barrier);
}
CUresult CudaExecutionSession::begin_external_barrier(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (!state.reserve_domain_physical()) return CUDA_ERROR_UNKNOWN;
    const auto result = state.executor->begin_external_barrier(barrier);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::record_external_enqueue(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->record_external_enqueue(barrier);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::enqueue_host_function(
    const CudaExecutionExternalBarrier& barrier, std::uint64_t stream,
    const std::shared_ptr<CudaSessionHostFunction>& callback) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (!callback || !state.dispatch.launch_host_function) return CUDA_ERROR_NOT_SUPPORTED;
    CUstream native = state.storage->default_proxy;
    if (stream) {
        const auto found =
            std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
        if (found == state.storage->streams.end()) return CUDA_ERROR_INVALID_HANDLE;
        native = found->native;
    }
    const auto begun = begin_external_barrier(barrier);
    if (begun != CUDA_SUCCESS) return begun;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto original = state.dispatch.launch_host_function(
        native, &CudaSessionHostFunction::wait, callback->state_.get());
    const auto proof = state.check_context(true);
    if (original != CUDA_SUCCESS || proof != CUDA_SUCCESS) {
        const auto status = proof != CUDA_SUCCESS ? proof : original;
        (void)state.executor->fail_external_barrier(barrier, status);
        state.fault();
        state.update_domain_physical();
        return status;
    }
    const auto recorded = state.executor->record_external_enqueue(barrier);
    if (recorded != CUDA_SUCCESS) state.fault();
    state.update_domain_physical();
    return recorded;
}
CUresult CudaExecutionSession::cancel_external_barrier(
    const CudaExecutionExternalBarrier& barrier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    return state.executor->cancel_external_barrier(barrier);
}
CUresult CudaExecutionSession::fail_external_barrier(const CudaExecutionExternalBarrier& barrier,
                                                     CUresult failure) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->fail_external_barrier(barrier, failure);
    state.update_domain_physical();
    return result;
}

CudaExecutionEnqueueAcceptance CudaExecutionSession::accept_enqueue_turn(
    std::uint64_t stream, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_enqueue_turn(stream, lifetime);
}
CudaExecutionEnqueueAcceptance CudaExecutionSession::accept_enqueue_cohort(
    std::span<const std::uint64_t> streams, const std::shared_ptr<const void>& lifetime) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->accept_enqueue_cohort(streams, lifetime);
}
CUresult CudaExecutionSession::enqueue_turn_ready(
    const CudaExecutionEnqueueTurn& turn) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    return state.executor->enqueue_turn_ready(turn);
}
CUresult CudaExecutionSession::begin_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (!state.reserve_domain_physical()) return CUDA_ERROR_UNKNOWN;
    const auto result = state.executor->begin_enqueue_turn(turn);
    state.update_domain_physical();
    return result;
}
CudaExecutionIssueResult CudaExecutionSession::progress_enqueue_dependency(
    const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.state = CudaExecutionIssueState::kError,
                .native_status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = configured};
    }
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->progress_enqueue_dependency(turn);
    state.update_domain_physical();
    return result;
}
CudaExecutionDependencyInspection CudaExecutionSession::inspect_enqueue_dependency(
    const CudaExecutionEnqueueTurn& turn) const noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    return state.executor->inspect_enqueue_dependency(turn);
}
CudaExecutionDependencyInspection CudaExecutionSession::inspect_frontier_dependency(
    const CudaExecutionFrontier& frontier) const noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    return state.executor->inspect_frontier_dependency(frontier);
}
bool CudaExecutionSession::owns_frontier(const CudaExecutionFrontier& frontier) const noexcept {
    const auto& state = *implementation_;
    return state.owns() && state.executor && state.executor->owns_frontier(frontier);
}
CudaExecutionIssueResult CudaExecutionSession::progress_frontier_dependency(
    const CudaExecutionFrontier& frontier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.state = CudaExecutionIssueState::kError, .native_status = status};
    if (state.is_faulted)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->progress_frontier_dependency(frontier);
    state.update_domain_physical();
    return result;
}
CudaExecutionDependencyInspection CudaExecutionSession::inspect_frontier_retirement(
    const CudaExecutionFrontier& frontier) const noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    return state.executor->inspect_frontier_retirement(frontier);
}
CudaExecutionIssueResult CudaExecutionSession::progress_frontier_retirement(
    const CudaExecutionFrontier& frontier) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.state = CudaExecutionIssueState::kError, .native_status = status};
    if (state.is_faulted)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->progress_frontier_retirement(frontier);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::finish_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->finish_enqueue_turn(turn);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::cancel_enqueue_turn(const CudaExecutionEnqueueTurn& turn) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->cancel_enqueue_turn(turn);
    state.update_domain_physical();
    return result;
}
CUresult CudaExecutionSession::fail_enqueue_turn(const CudaExecutionEnqueueTurn& turn,
                                                 CUresult failure) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto result = state.executor->fail_enqueue_turn(turn, failure);
    state.update_domain_physical();
    return result;
}

CUresult CudaExecutionSession::prepare_external_bridge(
    CudaExternalExecutionBridge& bridge) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.executor->has_active_enqueue_turn()) return CUDA_ERROR_NOT_READY;
    if (state.is_faulted) return CUDA_ERROR_UNKNOWN;
    if (!bridge.matches_source(state.storage->context, state.context_id))
        return CUDA_ERROR_INVALID_HANDLE;
    const NativeExecutionScope scope(state.is_native_issue_active);
    return bridge.prepare_source();
}
CUresult CudaExecutionSession::join_external_bridge(CudaExternalExecutionBridge& bridge) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.executor->has_active_enqueue_turn()) return CUDA_ERROR_NOT_READY;
    if (!bridge.matches_source(state.storage->context, state.context_id))
        return CUDA_ERROR_INVALID_HANDLE;
    const NativeExecutionScope scope(state.is_native_issue_active);
    return bridge.join_source();
}
CUresult CudaExecutionSession::cleanup_external_bridge(
    CudaExternalExecutionBridge& bridge) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    if (state.executor->has_active_enqueue_turn()) return CUDA_ERROR_NOT_READY;
    if (!bridge.matches_source(state.storage->context, state.context_id))
        return CUDA_ERROR_INVALID_HANDLE;
    const NativeExecutionScope scope(state.is_native_issue_active);
    return bridge.cleanup_source();
}

CudaRuntimeMetadataResult CudaExecutionSession::runtime_metadata(
    const CudaRuntimeMetadataQuery& query) noexcept {
    auto& state = *implementation_;
    CudaRuntimeMetadataResult result;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) {
        result.status = checked;
        return result;
    }
    if (state.is_faulted || state.is_closing || !state.executor ||
        state.executor->snapshot().is_quarantined)
        return result;
    const auto& sdk = state.dispatch.runtime;
    if ((query.kind == CudaRuntimeMetadataKind::kAttribute &&
         (query.attribute < 0 || query.attribute >= static_cast<int>(cudaDevAttrMax))) ||
        (query.kind == CudaRuntimeMetadataKind::kPeerAttribute &&
         (query.attribute < 0 || query.attribute > 7))) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }
    if (!sdk.version || (query.kind == CudaRuntimeMetadataKind::kProperties && !sdk.properties) ||
        (query.kind == CudaRuntimeMetadataKind::kAttribute && !sdk.attribute) ||
        (query.kind == CudaRuntimeMetadataKind::kPeerAttribute && !sdk.peer_attribute) ||
        (query.kind == CudaRuntimeMetadataKind::kStreamPriorityRange &&
         !sdk.stream_priority_range) ||
        (query.kind == CudaRuntimeMetadataKind::kPeerAccess && !sdk.peer_access) ||
        (query.kind == CudaRuntimeMetadataKind::kMemoryInfo && !sdk.memory_info)) {
        result.status = CUDA_ERROR_NOT_SUPPORTED;
        return result;
    }
    try {
        // Fixed CUDA-12.6 SDK ABI: scalar/array properties only, no pointers or
        // resource handles. An unverified SDK cannot supply this wire profile.
        const auto version_status = state.invoke(sdk.version, &result.version);
        const auto after_version = state.check_context();
        if (after_version != CUDA_SUCCESS) {
            state.fault();
            result.status = after_version;
            return result;
        }
        if (version_status != cudaSuccess || result.version != 12060 || CUDART_VERSION != 12060 ||
            sizeof(void*) != 8) {
            result.status = CUDA_ERROR_NOT_SUPPORTED;
            return result;
        }
        cudaDeviceProp properties{};
        int value{};
        std::array<int, 2> priorities{};
        std::array<std::size_t, 2> memory{};
        const void* data{};
        std::size_t bytes{};
        switch (query.kind) {
            case CudaRuntimeMetadataKind::kProperties:
                result.original_status = state.invoke(sdk.properties, &properties, query.device);
                data = &properties;
                bytes = sizeof(properties);
                break;
            case CudaRuntimeMetadataKind::kAttribute:
                result.original_status =
                    state.invoke(sdk.attribute, &value,
                                 static_cast<cudaDeviceAttr>(query.attribute), query.device);
                data = &value;
                bytes = sizeof(value);
                break;
            case CudaRuntimeMetadataKind::kPeerAttribute:
                result.original_status = state.invoke(
                    sdk.peer_attribute, &value, static_cast<cudaDeviceP2PAttr>(query.attribute),
                    query.device, query.peer);
                data = &value;
                bytes = sizeof(value);
                break;
            case CudaRuntimeMetadataKind::kStreamPriorityRange:
                result.original_status =
                    state.invoke(sdk.stream_priority_range, &priorities[0], &priorities[1]);
                data = priorities.data();
                bytes = sizeof(priorities);
                break;
            case CudaRuntimeMetadataKind::kPeerAccess:
                result.original_status =
                    state.invoke(sdk.peer_access, &value, query.device, query.peer);
                data = &value;
                bytes = sizeof(value);
                break;
            case CudaRuntimeMetadataKind::kMemoryInfo:
                result.original_status = state.invoke(sdk.memory_info, &memory[0], &memory[1]);
                data = memory.data();
                bytes = sizeof(memory);
                break;
            default:
                result.status = CUDA_ERROR_INVALID_VALUE;
                return result;
        }
        const auto after = state.check_context();
        if (after != CUDA_SUCCESS) {
            state.fault();
            result.status = after;
            return result;
        }
        result.disposition = CudaSessionDisposition::kComplete;
        result.status = CUDA_SUCCESS;
        if (result.original_status == cudaSuccess) {
            const auto* first = static_cast<const std::byte*>(data);
            result.data.assign(first, first + bytes);
        }
        return result;
    } catch (const std::bad_alloc&) {
        result.status = CUDA_ERROR_OUT_OF_MEMORY;
        result.disposition = CudaSessionDisposition::kRejected;
        return result;
    } catch (...) {
        state.fault();
        result.status = CUDA_ERROR_UNKNOWN;
        result.disposition = CudaSessionDisposition::kRejected;
        return result;
    }
}
CudaSessionResult CudaExecutionSession::create_stream(unsigned int flags,
                                                      std::optional<int> priority) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (flags != CU_STREAM_DEFAULT && flags != CU_STREAM_NON_BLOCKING)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->streams.size() == state.limits.max_objects)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    if (priority && (!state.dispatch.stream_create_priority || !state.dispatch.stream_get_priority))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    std::optional<NativeExecutionScope> proxy_scope;
    if (state.stream_config.uses_default_proxy) proxy_scope.emplace(state.is_native_issue_active);
    int native_priority = priority.value_or(0), logical_priority = priority.value_or(0);
    if (state.stream_config.scheduling_priority) {
        const auto priorities = state.scheduling_stream_priorities(logical_priority);
        if (priorities.status != CUDA_SUCCESS) return rejected(priorities.status);
        native_priority = priorities.native;
        logical_priority = priorities.logical;
    }
    Implementation::Stream stream{
        .id = state.next_id, .native = nullptr, .is_nonblocking = flags == CU_STREAM_NON_BLOCKING};
    const auto native_flags =
        state.uses_shared_domain ? static_cast<unsigned int>(CU_STREAM_NON_BLOCKING) : flags;
    auto result =
        completed(priority || state.stream_config.scheduling_priority
                      ? state.invoke(state.dispatch.stream_create_priority, &stream.native,
                                     native_flags, native_priority)
                      : state.invoke(state.dispatch.stream_create, &stream.native, native_flags));
    if (result.status != CUDA_SUCCESS && !stream.native) {
        if (state.check_context(state.stream_config.uses_default_proxy) != CUDA_SUCCESS)
            state.fault();
        return result;
    }
    (void)state.consume_id();
    state.storage->streams.push_back(stream);
    if (state.storage->default_proxy && stream.native == state.storage->default_proxy) {
        state.storage->is_default_proxy_uncertain = true;
        state.fault();
        result.disposition = CudaSessionDisposition::kRejected;
        result.status = CUDA_ERROR_UNKNOWN;
        return result;
    }
    const auto after = state.check_context(state.stream_config.uses_default_proxy);
    if (after != CUDA_SUCCESS) {
        state.fault();
        auto failed = rejected(after);
        failed.original_status = result.original_status;
        return failed;
    }
    if ((priority || state.stream_config.scheduling_priority) && result.status == CUDA_SUCCESS &&
        stream.native) {
        int actual{};
        const auto queried =
            state.invoke(state.dispatch.stream_get_priority, stream.native, &actual);
        const auto restored = state.check_context(state.stream_config.uses_default_proxy);
        if (queried != CUDA_SUCCESS || restored != CUDA_SUCCESS ||
            (state.stream_config.scheduling_priority && actual != native_priority)) {
            state.fault();
            auto failed = rejected(queried != CUDA_SUCCESS    ? queried
                                   : restored != CUDA_SUCCESS ? restored
                                                              : CUDA_ERROR_UNKNOWN);
            failed.original_status = result.original_status;
            return failed;
        }
        result.address = std::bit_cast<std::uint32_t>(
            state.stream_config.scheduling_priority ? logical_priority : actual);
    }
    if (result.status == CUDA_SUCCESS && stream.native &&
        state.executor->bind_stream(stream.id, stream.native, state.storage,
                                    flags == CU_STREAM_NON_BLOCKING))
        result.id = stream.id;
    else {
        state.fault();
        if (result.status == CUDA_SUCCESS) {
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
        }
    }
    return result;
}
std::optional<bool> CudaExecutionSession::stream_is_nonblocking(std::uint64_t id) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt || state.is_faulted)
        return std::nullopt;
    if (!id) return false;
    const auto stream = std::ranges::find(state.storage->streams, id, &Implementation::Stream::id);
    if (stream == state.storage->streams.end()) return std::nullopt;
    return stream->is_nonblocking;
}

CudaSessionResult CudaExecutionSession::destroy_stream(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return gate;
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto found = std::ranges::find_if(state.storage->streams,
                                            [id](const auto& entry) { return entry.id == id; });
    if (found == state.storage->streams.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (state.has_external_stream(id)) return deferred_resource_release();
    if (std::ranges::any_of(state.storage->blas,
                            [id](const auto& handle) { return handle.stream == id; }))
        return rejected(CUDA_ERROR_NOT_READY);
#ifdef GLIMMER_HAS_CUDNN
    if (std::ranges::any_of(state.storage->dnn,
                            [id](const auto& handle) { return handle.stream == id; }))
        return rejected(CUDA_ERROR_NOT_READY);
#endif
    if (!state.executor->unbind_stream(id)) {
        state.fault();
        return rejected(CUDA_ERROR_UNKNOWN);
    }
    auto result = completed(state.invoke(state.dispatch.stream_destroy, found->native));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    state.storage->streams.erase(found);
    return result;
}

// Logical bytes, mapped extent and native flags are separately validated.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
CudaSessionResult CudaExecutionSession::import_host_memory(int descriptor, std::size_t bytes,
                                                           std::size_t mapping_bytes,
                                                           unsigned int flags,
                                                           bool is_pageable) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    auto gate = state.cold_gate(true);
    if (gate.status != CUDA_SUCCESS) return gate;
    if (!is_pageable && (!state.dispatch.host_register || !state.dispatch.host_unregister ||
                         !state.dispatch.host_map))
        return rejected(CUDA_ERROR_NOT_SUPPORTED);
    if (is_pageable && flags) return rejected(CUDA_ERROR_INVALID_VALUE);
    if (flags & ~static_cast<unsigned int>(CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP |
                                           CU_MEMHOSTALLOC_WRITECOMBINED))
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (flags & CU_MEMHOSTALLOC_WRITECOMBINED) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto page = ::sysconf(_SC_PAGESIZE);
    struct stat info {};
    const auto seals = descriptor < 0 ? -1 : ::fcntl(descriptor, F_GET_SEALS);
    constexpr int required_seals = F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
    if (descriptor < 0 || page <= 0 || !bytes || mapping_bytes < bytes ||
        mapping_bytes % static_cast<std::size_t>(page) ||
        mapping_bytes - bytes >= static_cast<std::size_t>(page) ||
        ::fstat(descriptor, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size <= 0 ||
        static_cast<std::uintmax_t>(info.st_size) != mapping_bytes || seals < 0 ||
        (seals & required_seals) != required_seals ||
        (::fcntl(descriptor, F_GETFL) & O_ACCMODE) != O_RDWR)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    if (!state.has_id_space || state.storage->host_allocations.size() == state.limits.max_objects ||
        mapping_bytes > state.limits.max_host_memory_bytes - state.storage->host_memory_bytes)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    try {
        auto mapping = std::make_shared<Implementation::HostMapping>();
        mapping->bytes = mapping_bytes;
        mapping->pointer =
            ::mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
        if (mapping->pointer == MAP_FAILED)
            return rejected(errno == ENOMEM ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_ERROR_INVALID_VALUE);
        // Capacity is reserved before any native resource exists. Unknown native
        // state retains this charged mapping, even when no public ID is returned.
        state.storage->host_allocations.push_back({0, bytes, flags, 0, std::move(mapping)});
        state.storage->host_memory_bytes += mapping_bytes;
        auto& allocation = state.storage->host_allocations.back();
        allocation.is_pageable = is_pageable;
        if (is_pageable) {
            // This mapping has never been registered. Preserve its host memory
            // class for the exact generic SDK call; physical borrowers still
            // retain the same charged mapping until independent retirement.
            allocation.id = state.consume_id();
            auto result = settled();
            result.id = allocation.id;
            return result;
        }
        const auto discard = [&](CudaSessionResult result) {
            if (::munmap(allocation.mapping->pointer, mapping_bytes) != 0) {
                allocation.mapping->can_unmap = false;
                allocation.is_uncertain = true;
                state.fault();
                result.disposition = CudaSessionDisposition::kRejected;
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
            allocation.mapping->pointer = MAP_FAILED;
            state.storage->host_memory_bytes -= mapping_bytes;
            state.storage->host_allocations.pop_back();
            return result;
        };
        CUdeviceptr before{};
        const auto observed =
            state.invoke(state.dispatch.host_map, &before, allocation.mapping->pointer, 0);
        if (state.is_faulted) {
            allocation.mapping->can_unmap = false;
            allocation.is_uncertain = true;
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        if (observed != CUDA_ERROR_INVALID_VALUE || before) {
            allocation.mapping->can_unmap = false;
            allocation.is_uncertain = true;
            state.fault();
            return rejected(observed == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : observed);
        }
        allocation.mapping->can_unmap = false;
        auto result = completed(
            state.invoke(state.dispatch.host_register, allocation.mapping->pointer, mapping_bytes,
                         CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP));
        const auto retain_fault = [&] {
            allocation.is_uncertain = true;
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = CUDA_ERROR_UNKNOWN;
            return result;
        };
        if (state.is_faulted) return retain_fault();
        if (result.status != CUDA_SUCCESS) {
            CUdeviceptr after{};
            const auto empty =
                state.invoke(state.dispatch.host_map, &after, allocation.mapping->pointer, 0);
            if (state.is_faulted) return retain_fault();
            const bool is_known_rejection =
                result.status == CUDA_ERROR_INVALID_VALUE ||
                result.status == CUDA_ERROR_OUT_OF_MEMORY ||
                result.status == CUDA_ERROR_NOT_SUPPORTED ||
                result.status == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
            if (is_known_rejection && empty == CUDA_ERROR_INVALID_VALUE && !after)
                return discard(result);
            allocation.is_uncertain = true;
            state.fault();
            return result;
        }
        const auto mapped = state.invoke(state.dispatch.host_map, &allocation.address,
                                         allocation.mapping->pointer, 0);
        if (state.is_faulted) return retain_fault();
        if (mapped != CUDA_SUCCESS || !allocation.address ||
            allocation.address % static_cast<std::size_t>(page) ||
            mapping_bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - allocation.address) {
            const auto failure = mapped == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : mapped;
            // No application work or public capability exists. One checked
            // unregister can settle this known successful private registration.
            allocation.has_unregister_attempt = true;
            const auto cleaned =
                state.invoke(state.dispatch.host_unregister, allocation.mapping->pointer);
            if (state.is_faulted) return retain_fault();
            result.disposition = CudaSessionDisposition::kRejected;
            result.status = failure;
            if (cleaned == CUDA_SUCCESS) return discard(result);
            allocation.is_uncertain = true;
            state.fault();
            return result;
        }
        allocation.id = state.consume_id();
        result.id = allocation.id;
        result.address = allocation.address;
        return result;
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionResult CudaExecutionSession::release_host_memory(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto status = state.check_context();
    if (status != CUDA_SUCCESS) return rejected(status);
    if (state.is_faulted || !id) return rejected(CUDA_ERROR_INVALID_VALUE);
    const auto found =
        std::ranges::find(state.storage->host_allocations, id, &Implementation::HostAllocation::id);
    if (found == state.storage->host_allocations.end()) return rejected(CUDA_ERROR_INVALID_VALUE);
    return state.release_host_at(
        static_cast<std::size_t>(found - state.storage->host_allocations.begin()));
}
std::optional<CudaHostMemoryView> CudaExecutionSession::host_memory(CudaAllocationRange range,
                                                                    std::size_t bytes) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.is_faulted || state.is_closing || !range.allocation)
        return std::nullopt;
    const auto found = std::ranges::find(state.storage->host_allocations, range.allocation,
                                         &Implementation::HostAllocation::id);
    if (found == state.storage->host_allocations.end() || found->is_retiring ||
        found->is_uncertain || range.offset > found->bytes || bytes > found->bytes - range.offset ||
        range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
        return std::nullopt;
    return CudaHostMemoryView{static_cast<std::byte*>(found->mapping->pointer) + range.offset,
                              found->is_pageable ? 0 : found->address + range.offset, bytes,
                              found->mapping, found->is_pageable};
}
std::vector<std::shared_ptr<const void>> CudaExecutionSession::host_lifetimes() {
    auto& state = *implementation_;
    if (!state.owns() || state.is_faulted || state.is_closing)
        throw std::logic_error("unavailable CUDA host owner");
    std::vector<std::shared_ptr<const void>> lifetimes;
    lifetimes.reserve(state.storage->host_allocations.size());
    for (const auto& allocation : state.storage->host_allocations)
        if (!allocation.is_retiring) lifetimes.push_back(allocation.mapping);
    return lifetimes;
}
ExecutionAcceptance CudaExecutionSession::accept_host_transfer(std::uint64_t sequence,
                                                               CudaAllocationRange device,
                                                               const CudaHostMemoryView& host,
                                                               std::uint64_t stream,
                                                               bool is_upload) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_native_issue_active || state.has_graph_capture() || host.is_pageable)
        return invalid;
    CUdeviceptr address{};
    if (device.allocation || device.offset || host.bytes) {
        const auto found = state.find_allocation(device.allocation);
        if (!found || device.offset > found->bytes || host.bytes > found->bytes - device.offset ||
            device.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
            return invalid;
        address = found->address + device.offset;
    }
    if (host.bytes && (!host.pointer || !host.lifetime)) return invalid;
    return state.executor->accept_memory(
        sequence, {.kind = is_upload ? CudaMemoryKind::kUpload : CudaMemoryKind::kDownload,
                   .destination = is_upload ? address : 0,
                   .source = is_upload ? 0 : address,
                   .count = host.bytes,
                   .stream = stream,
                   .is_asynchronous = true,
                   .host = host.pointer,
                   .host_lifetime = host.lifetime});
}

ExecutionAcceptance CudaExecutionSession::accept_memory(std::uint64_t sequence,
                                                        const CudaSessionMemory& memory) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_native_issue_active || state.has_graph_capture()) return invalid;
    std::size_t width{1};
    switch (memory.kind) {
        case CudaMemoryKind::kCopy:
        case CudaMemoryKind::kFill8:
            break;
        case CudaMemoryKind::kFill16:
            width = 2;
            break;
        case CudaMemoryKind::kFill32:
            width = 4;
            break;
        default:
            return invalid;
    }
    if (memory.count > std::numeric_limits<std::size_t>::max() / width) return invalid;
    const auto bytes = memory.count * width;
    const auto address = [&](CudaAllocationRange range) -> std::optional<CUdeviceptr> {
        if (!range.allocation && !range.offset && !bytes) return CUdeviceptr{};
        const auto found = state.find_allocation(range.allocation);
        if (!found || range.offset > found->bytes || bytes > found->bytes - range.offset ||
            range.offset > std::numeric_limits<CUdeviceptr>::max() - found->address)
            return std::nullopt;
        return found->address + range.offset;
    };
    const auto destination = address(memory.destination);
    auto source = std::optional<CUdeviceptr>{0};
    if (memory.kind == CudaMemoryKind::kCopy)
        source = address(memory.source);
    else if (memory.source.allocation || memory.source.offset)
        return invalid;
    if (!destination || !source) return invalid;
    return state.executor->accept_memory(sequence, {.kind = memory.kind,
                                                    .destination = *destination,
                                                    .source = *source,
                                                    .count = memory.count,
                                                    .value = memory.value,
                                                    .stream = memory.stream,
                                                    .is_asynchronous = memory.is_asynchronous});
}

CudaSessionGraphSnapshot CudaExecutionSession::graph_snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {};
    return state.graphs ? state.graphs->snapshot()
                        : CudaSessionGraphSnapshot{.status = CUDA_SUCCESS};
}
CudaSessionGraphResult CudaExecutionSession::begin_graph_capture(std::uint64_t stream, int mode,
                                                                 std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return {.status = gate.status};
    const auto found =
        std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
    if (!stream || found == state.storage->streams.end() || !found->is_nonblocking)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    if (!state.dispatch.graph.begin) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    try {
        if (!state.graphs)
            state.graphs = std::make_unique<CudaExecutionGraph>(
                state.dispatch.graph, state.storage->context,
                state.dispatch.execution.context_get_current, state.limits.max_objects,
                state.limits.submissions.max_commands);
        const auto context = capture_external_context();
        if (context.status != CUDA_SUCCESS) return {.status = context.status};
        auto lifetime = std::make_shared<const CudaSessionExternalContext>(context.context);
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->begin(stream, found->native, mode, caller, std::move(lifetime));
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
CudaSessionGraphResult CudaExecutionSession::end_graph_capture(std::uint64_t stream,
                                                               std::uint64_t caller,
                                                               bool has_output,
                                                               std::uint64_t seed) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->end(stream, caller, has_output, seed);
}
CudaSessionGraphResult CudaExecutionSession::graph_capture_state(std::uint64_t stream,
                                                                 std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (state.graphs && state.graphs->snapshot().has_capture) {
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->state(stream, caller);
    }
    CUstream native = state.storage->default_proxy;
    if (stream) {
        const auto found =
            std::ranges::find(state.storage->streams, stream, &Implementation::Stream::id);
        if (found == state.storage->streams.end()) return {.status = CUDA_ERROR_INVALID_HANDLE};
        native = found->native;
    }
    if (!state.dispatch.graph.state) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const NativeExecutionScope scope(state.is_native_issue_active);
    if (const auto proof = state.check_context(true); proof != CUDA_SUCCESS)
        return {.status = proof};
    cudaStreamCaptureStatus observed{};
    const auto original = state.dispatch.graph.state(native, &observed);
    auto proof = state.check_context(true);
    if (proof != CUDA_SUCCESS) state.fault();
    if (proof == CUDA_SUCCESS && original == cudaSuccess &&
        observed != cudaStreamCaptureStatusNone) {
        state.fault();
        proof = CUDA_ERROR_UNKNOWN;
    }
    return {.status = proof,
            .original_status = static_cast<std::int32_t>(original),
            .capture_status = static_cast<std::uint32_t>(observed),
            .is_quarantined = state.is_faulted};
}
CudaSessionGraphResult CudaExecutionSession::instantiate_graph(std::uint64_t graph,
                                                               std::uint64_t flags, bool has_output,
                                                               std::uint64_t seed) noexcept {
    auto& state = *implementation_;
    const auto gate = state.cold_gate();
    if (gate.status != CUDA_SUCCESS) return {.status = gate.status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->instantiate(graph, flags, has_output, seed);
}
CudaSessionGraphResult CudaExecutionSession::destroy_graph(std::uint64_t graph) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->destroy(graph);
}
CudaSessionGraphResult CudaExecutionSession::destroy_graph_exec(std::uint64_t executable) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->destroy_exec(executable);
}
CudaSessionGraphResult CudaExecutionSession::invalidate_graph_capture(
    std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    return state.graphs ? state.graphs->invalidate(caller)
                        : CudaSessionGraphResult{.status = CUDA_ERROR_INVALID_HANDLE};
}
CudaSessionGraphResult CudaExecutionSession::abort_graph_capture() noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_SUCCESS};
    const NativeExecutionScope scope(state.is_native_issue_active);
    return state.graphs->abort();
}
CudaSessionGraphResult CudaExecutionSession::capture_memory(const CudaSessionMemory& memory,
                                                            std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.graphs) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (!memory.is_asynchronous ||
        (memory.kind != CudaMemoryKind::kCopy && memory.kind != CudaMemoryKind::kFill8 &&
         memory.kind != CudaMemoryKind::kFill16 && memory.kind != CudaMemoryKind::kFill32))
        return state.graphs->invalidate(caller);
    const auto width = memory.kind == CudaMemoryKind::kFill32   ? 4U
                       : memory.kind == CudaMemoryKind::kFill16 ? 2U
                                                                : 1U;
    if (memory.count > SIZE_MAX / width) return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto bytes = memory.count * width;
    const std::array ranges{CudaSessionExternalRangeRequest{memory.destination, bytes},
                            CudaSessionExternalRangeRequest{memory.source, bytes}};
    const auto capture = capture_external_command(
        memory.stream, std::span{ranges}.first(memory.kind == CudaMemoryKind::kCopy ? 2 : 1));
    if (capture.status != CUDA_SUCCESS) return {.status = capture.status};
    const auto& sdk = state.dispatch.execution;
    if ((memory.kind == CudaMemoryKind::kCopy && !sdk.copy_device_async) ||
        (memory.kind == CudaMemoryKind::kFill8 && !sdk.fill8_async) ||
        (memory.kind == CudaMemoryKind::kFill16 && !sdk.fill16_async) ||
        (memory.kind == CudaMemoryKind::kFill32 && !sdk.fill32_async))
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    try {
        auto lifetime = std::make_shared<const CudaSessionExternalCommand>(capture.command);
        const auto& command = *capture.command.state_;
        const auto destination = command.ranges[0].address;
        const auto source = memory.kind == CudaMemoryKind::kCopy ? command.ranges[1].address : 0;
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->append(memory.stream, caller, std::move(lifetime), [&] {
            switch (memory.kind) {
                case CudaMemoryKind::kCopy:
                    return sdk.copy_device_async(destination, source, bytes, command.stream);
                case CudaMemoryKind::kFill8:
                    return sdk.fill8_async(destination, static_cast<unsigned char>(memory.value),
                                           memory.count, command.stream);
                case CudaMemoryKind::kFill16:
                    return sdk.fill16_async(destination, static_cast<unsigned short>(memory.value),
                                            memory.count, command.stream);
                case CudaMemoryKind::kFill32:
                    return sdk.fill32_async(destination, memory.value, memory.count,
                                            command.stream);
                default:
                    return CUDA_ERROR_INVALID_VALUE;
            }
        });
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
CudaSessionGraphResult CudaExecutionSession::capture_kernel(const CudaExecutionKernel& kernel,
                                                            std::uint64_t caller) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    if (!state.graphs) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto function =
        std::ranges::find(state.storage->functions, kernel.function, &Implementation::Function::id);
    const auto stream =
        std::ranges::find(state.storage->streams, kernel.stream, &Implementation::Stream::id);
    if (function == state.storage->functions.end() || stream == state.storage->streams.end() ||
        kernel.parameters.size() != function->parameter_bytes ||
        (!kernel.parameters.empty() && !kernel.parameters.data()) ||
        std::ranges::any_of(kernel.grid, [](auto value) { return !value; }) ||
        std::ranges::any_of(kernel.block, [](auto value) { return !value; }))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    try {
        std::erase_if(state.external_contexts, [](const auto& entry) { return entry.expired(); });
        std::erase_if(state.external_commands, [](const auto& entry) { return entry.expired(); });
        if (state.external_contexts.size() == state.limits.max_objects ||
            state.external_commands.size() == state.limits.submissions.max_commands)
            return {.status = CUDA_ERROR_OUT_OF_MEMORY};
        auto command = std::make_shared<CudaSessionExternalCommand::State>();
        command->stream_id = kernel.stream;
        command->stream = stream->native;
        command->parent_modules.push_back(function->module);
        command->ranges.reserve(state.storage->allocations.size());
        for (const auto& allocation : state.storage->allocations) {
            command->ranges.push_back({{allocation.id, 0}, allocation.address, allocation.bytes});
            if (allocation.module) command->parent_modules.push_back(allocation.module);
        }
        if (state.process_memory) {
            auto leases = state.process_memory->owner()->borrow_live();
            if (leases.status != CUDA_SUCCESS) return {.status = leases.status};
            if (leases.lifetime) command->process_lifetimes.push_back(std::move(leases.lifetime));
        }
        for (auto& lease : host_lifetimes()) command->process_lifetimes.push_back(std::move(lease));
        if (kernel.lifetime) command->process_lifetimes.push_back(kernel.lifetime);
        command->context.state_ = std::make_shared<const CudaSessionExternalContext::State>(
            CudaSessionExternalContext::State{state.context_id, state.storage->context,
                                              state.device_ordinal, state.process, state.storage});
        std::vector<std::byte> parameters(kernel.parameters.begin(), kernel.parameters.end());
        std::vector<void*> arguments;
        arguments.reserve(function->arguments.size());
        for (const auto& layout : function->arguments)
            arguments.push_back(parameters.data() + layout.offset);
        state.external_contexts.push_back(command->context.state_);
        state.external_commands.push_back(command);
        const NativeExecutionScope scope(state.is_native_issue_active);
        return state.graphs->append(kernel.stream, caller, command, [&] {
            return state.dispatch.execution.launch_kernel(
                function->native, kernel.grid[0], kernel.grid[1], kernel.grid[2], kernel.block[0],
                kernel.block[1], kernel.block[2], kernel.shared_memory_bytes, stream->native,
                arguments.empty() ? nullptr : arguments.data(), nullptr);
        });
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    }
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): distinct scoped wire identities.
CudaSessionGraphLaunchAcceptance CudaExecutionSession::accept_graph_launch(
    std::uint64_t sequence, std::uint64_t executable, std::uint64_t stream) noexcept {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || !state.graphs || state.is_faulted || state.is_closing ||
        state.is_native_issue_active || state.has_graph_capture())
        return {};
    auto prepared = state.graphs->prepare_launch(executable);
    if (prepared.status != CUDA_SUCCESS) return {};
    auto accepted = state.executor->accept_library(
        sequence, {.stream = stream, .parameters = {}, .call = prepared.operation});
    return {accepted, accepted.accepted() ? std::move(prepared.operation) : nullptr};
}

ExecutionAcceptance CudaExecutionSession::accept_uva_copy(std::uint64_t sequence,
                                                          const CudaSessionUvaCopy& copy) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_native_issue_active || state.has_graph_capture()) return invalid;
    CudaExecutionMemory memory{.kind = CudaMemoryKind::kCopyUva,
                               .count = copy.bytes,
                               .stream = copy.stream,
                               .is_asynchronous = copy.is_asynchronous};
    const auto resolve = [&](const CudaSessionUvaEndpoint& endpoint,
                             std::size_t index) -> std::optional<CUdeviceptr> {
        const bool has_device = endpoint.device.allocation || endpoint.device.offset;
        const auto& host = endpoint.host;
        const bool has_host =
            host.pointer || host.address || host.bytes || host.lifetime || host.is_pageable;
        switch (endpoint.kind) {
            case CudaUvaEndpointKind::kNull:
                if (copy.bytes || has_device || has_host) return std::nullopt;
                return CUdeviceptr{};
            case CudaUvaEndpointKind::kDevice: {
                if (has_host || !endpoint.device.allocation) return std::nullopt;
                const auto view = device_memory(endpoint.device, copy.bytes);
                if (!view) return std::nullopt;
                memory.endpoint_lifetimes[index] = view->lifetime;
                return view->address;
            }
            case CudaUvaEndpointKind::kHost: {
                if (has_device || !host.pointer || !host.lifetime || copy.bytes > host.bytes ||
                    (host.is_pageable && host.address))
                    return std::nullopt;
                const auto address =
                    static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(host.pointer));
                if (copy.bytes > std::numeric_limits<CUdeviceptr>::max() - address)
                    return std::nullopt;
                memory.endpoint_lifetimes[index] = host.lifetime;
                // The original generic SDK infers host/device direction. A
                // registered host's device alias is not its host endpoint.
                return address;
            }
        }
        return std::nullopt;
    };
    const auto destination = resolve(copy.destination, 0);
    const auto source = resolve(copy.source, 1);
    if (!destination || !source) return invalid;
    if (copy.bytes && copy.destination.kind == CudaUvaEndpointKind::kHost &&
        copy.source.kind == CudaUvaEndpointKind::kHost && *destination < *source + copy.bytes &&
        *source < *destination + copy.bytes)
        return invalid;
    memory.destination = *destination;
    memory.source = *source;
    return state.executor->accept_memory(sequence, memory);
}

ExecutionAcceptance CudaExecutionSession::accept_async_kernel(
    std::uint64_t sequence, const CudaExecutionKernel& input) noexcept {
    return accept_async_kernel(sequence, input, nullptr);
}

ExecutionAcceptance CudaExecutionSession::accept_async_kernel(std::uint64_t sequence,
                                                              const CudaExecutionKernel& input,
                                                              bool* requires_preparation) noexcept {
    if (requires_preparation) *requires_preparation = false;
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!state.owns() || !state.executor || state.has_destroy_attempt || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_closing) return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    const auto function =
        std::ranges::find(state.storage->functions, input.function, &Implementation::Function::id);
    if (function == state.storage->functions.end() ||
        input.parameters.size() != function->parameter_bytes ||
        (!input.parameters.empty() && !input.parameters.data()) ||
        std::ranges::any_of(input.grid, [](auto value) { return !value; }) ||
        std::ranges::any_of(input.block, [](auto value) { return !value; }) ||
        (input.stream && !std::ranges::any_of(state.storage->streams, [&](const auto& stream) {
             return stream.id == input.stream;
         })))
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    std::size_t capacity = input.parameters.size();
    const bool may_transform = state.persistent.workers && !function->variants.empty() &&
                               !function->is_persistent_disabled;
    const bool uses_original_admission =
        !may_transform && !state.profile_cache && !state.persistent.uses_workload_agnostic;
    if (!uses_original_admission) {
        std::erase_if(state.deferred_kernels, [](const auto& value) { return value.expired(); });
        if (state.deferred_kernels.size() == state.limits.submissions.max_commands)
            return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
    if (may_transform)
        for (const auto& variant : function->variants) capacity = std::max(capacity, variant.bytes);
    if (capacity > state.limits.submissions.max_parameter_bytes)
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    try {
        auto retained = input;
        if (!state.storage->host_allocations.empty()) {
            auto lifetimes = host_lifetimes();
            if (input.lifetime) lifetimes.push_back(input.lifetime);
            if (!lifetimes.empty())
                retained.lifetime = std::make_shared<std::vector<std::shared_ptr<const void>>>(
                    std::move(lifetimes));
        }
        if (uses_original_admission) {
            // Session observers come only from profile selection, as in the
            // prepared original path. No native preparation or profiling is due.
            retained.observation.reset();
            return state.executor->accept_async_original_kernel(sequence, retained);
        }
        auto deferred = std::make_shared<Implementation::DeferredKernel>();
        deferred->storage = state.storage;
        deferred->kernel = retained;
        deferred->kernel.parameters = {};
        deferred->parameter_bytes = input.parameters.size();
        if (may_transform) {
            if (!state.storage->is_progress_ready)
                return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
            const auto free = std::ranges::find(state.storage->occupied, false);
            if (free == state.storage->occupied.end())
                return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
            deferred->reserved_slot =
                static_cast<std::size_t>(free - state.storage->occupied.begin());
            *free = true;
        }
        auto kernel = input;
        kernel.lifetime = deferred;
        auto accepted = state.executor->accept_deferred_kernel(sequence, kernel, capacity);
        if (accepted.accepted()) {
            deferred->receipt = accepted.receipt;
            state.deferred_kernels.push_back(deferred);
            if (requires_preparation) *requires_preparation = true;
        }
        return accepted;
    } catch (...) {
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}

CudaSessionPendingResult CudaExecutionSession::pending_preparation(
    std::optional<std::uint64_t> sequence) const noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    CudaSessionPendingResult result;
    result.command.description_.control_epoch = state.control_epoch;
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return {.status = member.status};
        result.command.description_.member_generation = member.generation;
        result.command.description_.configuration_epoch = member.configuration_epoch;
        result.command.description_.physical_configuration_epoch =
            member.physical_configuration_epoch;
    }
    auto pending = state.executor->pending_preparation(sequence);
    if (pending.status != CUDA_SUCCESS || !pending.command) return {.status = pending.status};
    result.command.description_.command = pending.command.description();
    result.command.command_ = std::move(pending.command);
    result.command.owner_ = state.storage;
    return result;
}

CudaSessionKernelPreparationResult CudaExecutionSession::prepare_pending_kernel(
    const CudaSessionPendingCommand& command) noexcept {
    using Disposition = CudaSessionKernelPreparationDisposition;
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.status = CUDA_ERROR_NOT_READY, .disposition = Disposition::kDeferred};
    if (state.process_memory && state.process_memory->native_health() != CUDA_SUCCESS)
        return {.status = state.process_memory->native_health()};
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY, .disposition = Disposition::kDeferred};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (command.owner_.lock() != state.storage) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto current = pending_preparation(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    std::shared_ptr<Implementation::DeferredKernel> deferred;
    for (const auto& weak : state.deferred_kernels)
        if (auto value = weak.lock();
            value && value->receipt.sequence() == command.description_.command.sequence) {
            deferred = std::move(value);
            break;
        }
    if (!deferred) return {.status = CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionPendingCommand prepared_command;
    const auto uncertain = [&](CUresult failure) {
        const auto observed = state.executor->quarantine_deferred_kernel(
            prepared_command ? prepared_command : command.command_, failure);
        state.fault();
        return CudaSessionKernelPreparationResult{
            .status = observed == CUDA_SUCCESS ? failure : observed,
            .disposition = Disposition::kUncertain};
    };
    const auto rejected = [&](CUresult failure) {
        if (!state.executor->reject_command(deferred->receipt, failure)) return uncertain(failure);
        return CudaSessionKernelPreparationResult{.status = failure,
                                                  .disposition = Disposition::kRejected};
    };
    std::vector<std::byte> parameters;
    try {
        parameters.resize(deferred->parameter_bytes);
    } catch (...) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    }
    const auto copied = state.executor->copy_deferred_parameters(command.command_, parameters);
    if (copied != CUDA_SUCCESS) return rejected(copied);
    try {
        if (state.storage->domain_member) {
            const auto configured = state.storage->domain_member.configuration_status(true);
            if (configured == CUDA_ERROR_NOT_READY)
                return {.status = configured, .disposition = Disposition::kDeferred};
            if (configured != CUDA_SUCCESS) return uncertain(configured);
        }
        const auto before = state.check_context(true);
        if (before != CUDA_SUCCESS) return uncertain(before);
        auto kernel = deferred->kernel;
        kernel.parameters = parameters;
        CUresult failure = CUDA_SUCCESS;
        ExecutionAcceptance prepared;
        {
            const NativeExecutionScope preparation_scope(state.is_preparation_active);
            prepared = accept_kernel_impl(deferred->receipt.sequence(), kernel, &command.command_,
                                          &deferred->receipt, &deferred->reserved_slot, &failure,
                                          &prepared_command);
        }
        if (state.is_faulted)
            return uncertain(failure != CUDA_SUCCESS ? failure : CUDA_ERROR_UNKNOWN);
        if (!prepared.accepted())
            return rejected(failure == CUDA_SUCCESS ? CUDA_ERROR_INVALID_VALUE : failure);
        if (deferred->reserved_slot) {
            state.storage->occupied[*deferred->reserved_slot] = false;
            deferred->reserved_slot.reset();
        }
        const auto pending = pending_command(deferred->receipt.sequence());
        if (pending.status != CUDA_SUCCESS && pending.status != CUDA_ERROR_NOT_READY)
            return uncertain(pending.status);
        return {.status = CUDA_SUCCESS,
                .disposition = Disposition::kPrepared,
                .command = pending.command};
    } catch (...) {
        return uncertain(CUDA_ERROR_UNKNOWN);
    }
}

ExecutionAcceptance CudaExecutionSession::accept_kernel(std::uint64_t sequence,
                                                        const CudaExecutionKernel& input) noexcept {
    try {
        return accept_kernel_impl(sequence, input);
    } catch (...) {
        implementation_->fault();
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    }
}

ExecutionAcceptance CudaExecutionSession::accept_kernel_impl(
    std::uint64_t sequence, const CudaExecutionKernel& input,
    const CudaExecutionPendingCommand* preparation, const ExecutionReceipt* receipt,
    std::optional<std::size_t>* reserved_slot, CUresult* preparation_status,
    CudaExecutionPendingCommand* prepared_command) {
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    const auto note_failure = [&](CUresult failure) {
        if (preparation_status && *preparation_status == CUDA_SUCCESS)
            *preparation_status = failure;
    };
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
    if (state.is_native_issue_active || (state.persistent.uses_workload_agnostic &&
                                         (state.is_folded_active || state.is_rnn_query_active)))
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (state.is_closing && !preparation)
        return {.status = ExecutionAcceptStatus::kClosed, .receipt = {}};
    auto kernel = input;
    try {
        if (!preparation && !state.storage->host_allocations.empty()) {
            auto lifetimes = host_lifetimes();
            if (kernel.lifetime) lifetimes.push_back(kernel.lifetime);
            if (!lifetimes.empty())
                kernel.lifetime = std::make_shared<std::vector<std::shared_ptr<const void>>>(
                    std::move(lifetimes));
        }
    } catch (...) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
    const auto function = std::ranges::find_if(
        state.storage->functions, [&](const auto& entry) { return entry.id == kernel.function; });
    // Private library bindings are not public launch handles, even if guessed.
    if (function == state.storage->functions.end())
        return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    const bool sharing = state.persistent.uses_workload_agnostic;
    std::optional<NativeExecutionScope> sharing_scope;
    if (sharing) sharing_scope.emplace(state.is_native_issue_active);
    if (state.profile_cache && function->has_configuration_attempt &&
        !function->is_persistent_disabled && !function->has_unknown_configuration &&
        !function->configuration) {
        const NativeExecutionScope capture_scope(state.is_native_issue_active);
        function->configuration = capture_function_configuration(
            state.dispatch, Implementation::profile_targets(*function), function->cache_preference,
            [&] {
                if (state.check_context(true) == CUDA_SUCCESS) return true;
                state.fault();
                return false;
            });
        if (state.is_faulted) return {.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    }
    const auto context_configuration =
        state.profile_cache ? state.profile_context(sharing_scope.has_value() || preparation)
                            : std::optional<CudaProfileContextConfiguration>{};
    const CudaProfileGeometry profile_geometry{
        kernel.grid,
        kernel.block,
        kernel.shared_memory_bytes,
        state.persistent.kind,
        context_configuration.value_or(CudaProfileContextConfiguration{}),
        function->configuration.value_or(CudaProfileFunctionConfiguration{})};
    bool has_profile_allocation_failure{};
    const auto profile_choice = [&](std::span<const transform::PtxExecutionKind> kinds) {
        auto choice = !state.persistent.uses_workload_agnostic && state.profile_cache &&
                              context_configuration && function->configuration
                          ? state.profile_cache->choose(function->profile, profile_geometry,
                                                        state.persistent.workers, kinds,
                                                        state.requires_profile_preemption)
                          : CudaProfileChoice{};
        try {
            choice.observation =
                state.profile_observation(*function, profile_geometry, choice.observation);
        } catch (...) {
            if (choice.observation) choice.observation->finish(false);
            has_profile_allocation_failure = true;
            return CudaProfileChoice{};
        }
        return choice;
    };
    const auto publish_original = [&] {
        if (!preparation) return state.executor->accept_kernel(sequence, kernel);
        const auto proof = state.check_context(true);
        if (proof != CUDA_SUCCESS) {
            note_failure(proof);
            state.fault();
            return ExecutionAcceptance{.status = ExecutionAcceptStatus::kQuarantined,
                                       .receipt = {}};
        }
        const auto prepared =
            state.executor->complete_kernel_preparation(*preparation, kernel.observation);
        if (prepared.status != CUDA_SUCCESS) {
            note_failure(prepared.status);
            return ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
        }
        if (prepared_command) *prepared_command = prepared.command;
        return ExecutionAcceptance{.status = ExecutionAcceptStatus::kAccepted, .receipt = *receipt};
    };
    const auto accept_original = [&] {
        auto choice = profile_choice({});
        if (has_profile_allocation_failure) {
            note_failure(CUDA_ERROR_OUT_OF_MEMORY);
            return ExecutionAcceptance{.status = ExecutionAcceptStatus::kOutOfMemory,
                                       .receipt = {}};
        }
        kernel.observation = std::move(choice.observation);
        return publish_original();
    };
    if (!state.persistent.workers || function->variants.empty() || function->is_persistent_disabled)
        return accept_original();
    if (sharing && (!context_configuration || !function->configuration)) return accept_original();
    const auto proof = [&] {
        const auto status = state.check_context(sharing || preparation);
        if (status != CUDA_SUCCESS) {
            note_failure(status);
            if (sharing || preparation) state.fault();
        }
        return status == CUDA_SUCCESS;
    };
    auto invalid = ExecutionAcceptance{.status = ExecutionAcceptStatus::kInvalid, .receipt = {}};
    if (!proof() || !state.storage->is_progress_ready ||
        kernel.parameters.size() != function->parameter_bytes ||
        (!kernel.parameters.empty() && !kernel.parameters.data()) ||
        (kernel.stream && !std::ranges::any_of(state.storage->streams, [&](const auto& entry) {
             return entry.id == kernel.stream;
         })))
        return invalid;
    std::uint64_t total = 1, threads = 1;
    for (const auto dimension : kernel.grid) {
        if (!dimension || total > std::numeric_limits<std::uint32_t>::max() / dimension)
            return accept_original();
        total *= dimension;
    }
    for (const auto dimension : kernel.block) {
        if (!dimension ||
            threads > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) / dimension)
            return accept_original();
        threads *= dimension;
    }
    std::array<transform::PtxExecutionKind, 5> eligible{};
    std::array<CudaProfileSharingCandidate, 4> sharing_candidates{};
    std::size_t eligible_count{};
    const Implementation::PrivateFunction* selected{};
    for (const auto& variant : function->variants) {
        if (sharing && variant.kind == transform::PtxExecutionKind::kPreemptive) continue;
        if (variant.kind == transform::PtxExecutionKind::kStatic &&
            (kernel.grid[1] != 1 || kernel.grid[2] != 1))
            continue;
        int capacity{};
        const auto occupancy = state.invoke(state.dispatch.occupancy, &capacity, variant.native,
                                            static_cast<int>(threads), kernel.shared_memory_bytes);
        if ((sharing || preparation) && !proof()) return invalid;
        if (preparation && occupancy != CUDA_SUCCESS) {
            note_failure(occupancy);
            return invalid;
        }
        if (occupancy != CUDA_SUCCESS || capacity <= 0) continue;
        if (sharing) {
            const auto cap = static_cast<std::uint64_t>(state.sharing_sm_count) *
                             static_cast<unsigned int>(capacity);
            sharing_candidates[eligible_count] = {
                variant.kind,
                static_cast<unsigned int>(std::min<std::uint64_t>(state.persistent.workers, cap))};
        }
        eligible[eligible_count++] = variant.kind;
        if (variant.kind == state.persistent.kind) selected = &variant;
    }
    if (!eligible_count) return accept_original();
    auto choice =
        sharing && context_configuration && function->configuration
            ? state.profile_cache->choose_sharing(
                  function->profile, profile_geometry,
                  {state.sharing_sm_count, state.persistent.sharing_threads_per_sm,
                   state.persistent.workers, state.persistent.sharing_minimum_speed_percent},
                  std::span{sharing_candidates}.first(eligible_count))
            : profile_choice(std::span{eligible}.first(eligible_count));
    if (has_profile_allocation_failure) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
    kernel.observation = std::move(choice.observation);
    if (sharing && !choice.was_selected) return publish_original();
    if (choice.was_selected && !choice.workers) return publish_original();
    if (choice.was_selected) {
        const auto found = std::ranges::find(function->variants, choice.kind,
                                             &Implementation::PrivateFunction::kind);
        if (found == function->variants.end()) return invalid;
        selected = &*found;
    }
    if (!selected) return publish_original();
    const auto kind = selected->kind;
    const auto configured_workers = choice.was_selected ? choice.workers : state.persistent.workers;
    const auto free_slot =
        reserved_slot && *reserved_slot
            ? state.storage->occupied.begin() + static_cast<std::ptrdiff_t>(**reserved_slot)
            : std::ranges::find(state.storage->occupied, false);
    if (free_slot == state.storage->occupied.end()) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kFull, .receipt = {}};
    }
    try {
        const auto index = static_cast<std::size_t>(free_slot - state.storage->occupied.begin());
        const auto count = function->arguments.size();
        const auto workers =
            kind == transform::PtxExecutionKind::kStatic
                ? static_cast<unsigned int>(std::min<std::uint64_t>(configured_workers, total))
                : configured_workers;
        const std::array<std::size_t, 2> slice_offsets =
            kind == transform::PtxExecutionKind::kSliced
                ? std::array{selected->arguments[count + 3].offset,
                             selected->arguments[count + 4].offset}
                : std::array<std::size_t, 2>{};
        auto progress = std::make_shared<Implementation::Progress>(
            state.storage, state.dispatch,
            Implementation::ProgressParameters{
                .slot = index,
                .sequence = sequence,
                .total = total,
                .workers = workers,
                .max_workers = state.persistent.workers,
                .can_retarget_workers = true,
                .kind = kind,
                .slice_offsets = slice_offsets,
                .observes_block_completion = selected->observes_block_completion});
        *free_slot = true;
        if (reserved_slot) reserved_slot->reset();
        std::vector<std::byte> parameters(selected->bytes);
        if (!kernel.parameters.empty())
            std::memcpy(parameters.data(), kernel.parameters.data(), kernel.parameters.size());
        const std::size_t axes = kind == transform::PtxExecutionKind::kStatic ? 1 : 3;
        for (std::size_t axis = 0; axis < axes; ++axis)
            std::memcpy(parameters.data() + selected->arguments[count + axis].offset,
                        &kernel.grid[axis], sizeof(kernel.grid[axis]));
        if (kind == transform::PtxExecutionKind::kDynamic ||
            kind == transform::PtxExecutionKind::kPreemptive) {
            const auto counter = progress->counter();
            std::memcpy(parameters.data() + selected->arguments[count + 3].offset, &counter,
                        sizeof(counter));
        }
        if (kind == transform::PtxExecutionKind::kPreemptive) {
            const auto retreat = progress->retreat();
            std::memcpy(parameters.data() + selected->arguments[count + 4].offset, &retreat,
                        sizeof(retreat));
            if (selected->observes_block_completion) {
                const auto completion = progress->block_completion();
                std::memcpy(parameters.data() + selected->arguments[count + 5].offset, &completion,
                            sizeof(completion));
            }
        }
        std::erase_if(state.progress, [](const auto& entry) { return entry.expired(); });
        ExecutionAcceptance accepted;
        if (preparation) {
            const auto proof = state.check_context(true);
            if (proof != CUDA_SUCCESS) {
                note_failure(proof);
                state.fault();
                return {.status = ExecutionAcceptStatus::kQuarantined, .receipt = {}};
            }
            const CudaExecutionPersistentKernel prepared{selected->native, workers, parameters,
                                                         progress};
            const auto completed = state.executor->complete_kernel_preparation(
                *preparation, kernel.observation, &prepared);
            if (completed.status == CUDA_SUCCESS)
                accepted = {.status = ExecutionAcceptStatus::kAccepted, .receipt = *receipt};
            else
                note_failure(completed.status);
            if (completed.status == CUDA_SUCCESS && prepared_command)
                *prepared_command = completed.command;
        } else {
            accepted = state.executor->accept_persistent_kernel(
                sequence, kernel, {selected->native, workers, parameters, progress});
        }
        if (accepted.accepted()) {
            state.progress.push_back(progress);
            if (state.persistent_accepted != std::numeric_limits<std::uint64_t>::max())
                ++state.persistent_accepted;
        }
        return accepted;
    } catch (...) {
        note_failure(CUDA_ERROR_OUT_OF_MEMORY);
        return {.status = ExecutionAcceptStatus::kOutOfMemory, .receipt = {}};
    }
}
CudaExecutionIssueResult CudaExecutionSession::issue_next() noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.state = CudaExecutionIssueState::kIdle, .native_status = CUDA_ERROR_NOT_READY};
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.state = CudaExecutionIssueState::kError,
                .native_status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = configured};
    }
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->issue_next();
    state.update_domain_physical();
    return result;
}

CudaSessionPendingResult CudaExecutionSession::pending_command(
    std::optional<std::uint64_t> sequence) const noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    const NativeExecutionScope scope{state.is_native_issue_active};
    CudaSessionPendingResult result;
    result.command.description_.control_epoch = state.control_epoch;
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return {.status = member.status};
        result.command.description_.member_generation = member.generation;
        result.command.description_.configuration_epoch = member.configuration_epoch;
        result.command.description_.physical_configuration_epoch =
            member.physical_configuration_epoch;
    }
    auto pending = state.executor->pending_command(sequence);
    result.status = pending.status;
    if (pending.status != CUDA_SUCCESS || !pending.command) return {.status = pending.status};
    result.command.description_.command = pending.command.description();
    result.command.command_ = std::move(pending.command);
    result.command.owner_ = state.storage;
    return result;
}

CudaExecutionIssueResult CudaExecutionSession::issue(
    const CudaSessionPendingCommand& command) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture())
        return {.state = CudaExecutionIssueState::kIdle, .native_status = CUDA_ERROR_NOT_READY};
    const auto failure = [&](CUresult status) {
        return CudaExecutionIssueResult{.state = CudaExecutionIssueState::kError,
                                        .sequence = command.description_.command.sequence,
                                        .native_status = status};
    };
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return failure(CUDA_ERROR_NOT_READY);
    if (state.is_faulted) return failure(CUDA_ERROR_UNKNOWN);
    if (command.owner_.lock() != state.storage) return failure(CUDA_ERROR_INVALID_HANDLE);
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return failure(current.status);
    if (!current.command || current.command.description_ != command.description_)
        return failure(CUDA_ERROR_INVALID_HANDLE);
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS) return failure(configured);
    }
    if (!state.reserve_domain_physical()) return failure(CUDA_ERROR_UNKNOWN);
    const auto result = state.executor->issue(command.command_);
    state.update_domain_physical();
    return result;
}
CudaSessionPendingResult CudaExecutionSession::select_pending_kernel(
    const CudaSessionPendingCommand& command,
    const CudaExecutionPendingKernelSelection& selection) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (command.owner_.lock() != state.storage) return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (!selection.uses_original && selection.workers > state.persistent.workers)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    {
        const NativeExecutionScope active{state.is_native_issue_active};
        if (state.storage->domain_member) {
            const auto configured = state.storage->domain_member.configuration_status(true);
            if (configured != CUDA_SUCCESS) return {.status = configured};
        }
        const auto selected = state.executor->select_pending_kernel(command.command_, selection);
        if (state.executor->snapshot().is_quarantined) state.fault();
        if (selected.status != CUDA_SUCCESS) return {.status = selected.status};
    }
    return pending_command(command.description_.command.sequence);
}

CudaSessionPairKernelResult CudaExecutionSession::prepare_pair_kernel(
    const CudaSessionPendingCommand& command) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return {.status = CUDA_ERROR_NOT_READY};
    if (state.process_memory && state.process_memory->native_health() != CUDA_SUCCESS)
        return {.status = state.process_memory->native_health()};
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (command.owner_.lock() != state.storage) return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status();
        if (configured != CUDA_SUCCESS) return {.status = configured};
    }
    if (!state.next_pair_catalog) return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    if (!state.dispatch.device_attribute || !state.dispatch.function_get_attribute ||
        !state.dispatch.occupancy || !command.description_.command.is_persistent)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const auto function =
        std::ranges::find(state.storage->functions, command.description_.command.function,
                          &Implementation::Function::id);
    const auto progress = persistent_progress(command.description_.command.sequence);
    if (!progress) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const NativeExecutionScope scope{state.is_native_issue_active};
    const auto proof = [&] {
        CUresult status{CUDA_ERROR_UNKNOWN};
        try {
            status = state.check_context(true);
        } catch (...) {
            state.fault();
            return CUDA_ERROR_UNKNOWN;
        }
        if (status != CUDA_SUCCESS) state.fault();
        return status;
    };
    if (const auto status = proof(); status != CUDA_SUCCESS) return {.status = status};
    if (function == state.storage->functions.end()) {
        const auto is_private_binding = [&](const auto& ids, const auto& functions) {
            const auto id = command.description_.command.function;
            const auto found = std::ranges::find(ids, id);
            return id && found != ids.end() &&
                   functions[static_cast<std::size_t>(found - ids.begin())];
        };
        // Private replacement bindings are real persistent commands but have
        // no public-PTX measurement catalog. Do not classify an arbitrary
        // missing function, lost progress or native metadata error this way.
        if (progress->kind == transform::PtxExecutionKind::kPreemptive && progress->workers &&
            (is_private_binding(state.storage->gemm_ids, state.storage->gemm_functions) ||
             is_private_binding(state.storage->lt_ids, state.storage->lt_functions)))
            return {.status = CUDA_ERROR_NOT_SUPPORTED,
                    .disposition = CudaSessionPairKernelDisposition::kUnsupported};
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    }
    const auto variant = std::ranges::find(function->variants, progress->kind,
                                           &Implementation::PrivateFunction::kind);
    if (variant == function->variants.end() || !variant->native)
        return {.status = CUDA_ERROR_NOT_SUPPORTED};
    switch (progress->kind) {
        case transform::PtxExecutionKind::kPreemptive:
        case transform::PtxExecutionKind::kSliced:
            break;
        case transform::PtxExecutionKind::kStatic:
        case transform::PtxExecutionKind::kBlockStatic:
        case transform::PtxExecutionKind::kDynamic:
            return {.status = CUDA_ERROR_NOT_SUPPORTED,
                    .disposition = CudaSessionPairKernelDisposition::kUnsupported};
        default:
            return {.status = CUDA_ERROR_NOT_SUPPORTED};
    }
    const auto query = [&](auto callback, auto... arguments) {
        const auto before = proof();
        if (before != CUDA_SUCCESS) return before;
        CUresult status{CUDA_ERROR_UNKNOWN};
        try {
            status = state.invoke(callback, arguments...);
        } catch (...) {
            state.fault();
            return CUDA_ERROR_UNKNOWN;
        }
        const auto after = proof();
        return after == CUDA_SUCCESS ? status : after;
    };
    CudaSessionPairKernelResult result;
    auto& data = result.kernel.description_;
    data.pending = command.description_;
    data.geometry = {command.description_.command.grid, command.description_.command.block,
                     command.description_.command.shared_memory_bytes, state.persistent.kind};
    constexpr std::array attributes{CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
                                    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                                    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
                                    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR};
    std::array<unsigned int, attributes.size()> hardware{};
    for (std::size_t index = 0; index < attributes.size(); ++index) {
        int value{};
        const auto status =
            query(state.dispatch.device_attribute, &value, attributes[index], state.native_device);
        if (status != CUDA_SUCCESS) return {.status = status};
        if (value <= 0) return {.status = CUDA_ERROR_INVALID_VALUE};
        hardware[index] = static_cast<unsigned int>(value);
    }
    data.limits = {hardware[0], hardware[1], hardware[2], hardware[3], state.persistent.workers,
                   hardware[4]};
    int registers{}, shared{}, occupancy{};
    auto status = query(state.dispatch.function_get_attribute, &registers,
                        CU_FUNC_ATTRIBUTE_NUM_REGS, variant->native);
    if (status != CUDA_SUCCESS) return {.status = status};
    status = query(state.dispatch.function_get_attribute, &shared,
                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, variant->native);
    if (status != CUDA_SUCCESS) return {.status = status};
    std::uint64_t threads{1};
    for (const auto dimension : data.geometry.block) {
        if (!dimension || threads > static_cast<unsigned int>(INT32_MAX) / dimension)
            return {.status = CUDA_ERROR_INVALID_VALUE};
        threads *= dimension;
    }
    status = query(state.dispatch.occupancy, &occupancy, variant->native, static_cast<int>(threads),
                   data.geometry.shared_memory_bytes);
    if (status != CUDA_SUCCESS) return {.status = status};
    if (registers < 0 || shared < 0 || occupancy <= 0 || threads > data.limits.threads_per_sm ||
        static_cast<unsigned int>(occupancy) > data.limits.blocks_per_sm)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    data.variant = {progress->kind, progress->workers, static_cast<unsigned int>(registers),
                    static_cast<unsigned int>(occupancy), static_cast<std::uint64_t>(shared)};
    if (state.profile_cache && !state.profile_cache->is_learning() && function->configuration) {
        const auto configuration = state.profile_context(true);
        if (configuration) {
            data.profile = function->profile;
            data.geometry.context = *configuration;
            data.geometry.function = *function->configuration;
        }
    }
    if (state.storage->domain_member) {
        const auto member = state.storage->domain_member.view(true);
        if (member.status != CUDA_SUCCESS) return {.status = member.status};
        data.execution_domain = member.context.id;
        data.is_configuration_materialized = member.is_configuration_materialized;
    } else {
        data.is_configuration_materialized = true;
    }
    data.catalog_generation = state.next_pair_catalog;
    state.next_pair_catalog =
        state.next_pair_catalog == UINT64_MAX ? 0 : state.next_pair_catalog + 1;
    result.kernel.command_ = command;
    result.kernel.owner_ = state.storage;
    result.status = CUDA_SUCCESS;
    result.disposition = CudaSessionPairKernelDisposition::kPrepared;
    return result;
}

CudaSessionPairKernelResult CudaExecutionSession::refresh_pair_kernel(
    const CudaSessionPairKernel& kernel, const CudaSessionPendingCommand& command) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.has_destroy_attempt)
        return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return {.status = CUDA_ERROR_NOT_READY};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN};
    if (kernel.owner_.lock() != state.storage || command.owner_.lock() != state.storage ||
        !kernel.description_.catalog_generation)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    const auto current = pending_command(command.description_.command.sequence);
    if (current.status != CUDA_SUCCESS) return {.status = current.status};
    if (!current.command || current.command.description_ != command.description_)
        return {.status = CUDA_ERROR_INVALID_HANDLE};
    auto expected = kernel.description_.pending;
    expected.command.geometry_epoch = command.description_.command.geometry_epoch;
    expected.command.physical_grid = command.description_.command.physical_grid;
    expected.command.is_persistent = command.description_.command.is_persistent;
    if (expected != command.description_) return {.status = CUDA_ERROR_INVALID_HANDLE};
    CudaSessionPairKernelResult result;
    result.kernel = kernel;
    result.kernel.command_ = command;
    result.kernel.description_.pending = command.description_;
    result.kernel.description_.variant.workers = command.description_.command.is_persistent
                                                     ? command.description_.command.physical_grid[0]
                                                     : 0;
    result.status = CUDA_SUCCESS;
    result.disposition = CudaSessionPairKernelDisposition::kPrepared;
    return result;
}

bool CudaExecutionSession::can_share_configuration(
    const CudaExecutionSession& other) const noexcept {
    const auto& state = *implementation_;
    const auto& peer = *other.implementation_;
    return state.owns() && peer.owns() && !state.has_destroy_attempt && !peer.has_destroy_attempt &&
           !state.is_faulted && !peer.is_faulted &&
           state.storage->domain_member.can_share_configuration(peer.storage->domain_member);
}

core::ExecutionPairProposal CudaSessionPairSelection::proposal(
    std::array<std::uint64_t, 2> policy_ids) const noexcept {
    core::ExecutionPairProposal result;
    result.is_shared = choice_.is_shared;
    result.first_serial = choice_.is_shared ? 0 : choice_.first_original;
    for (std::size_t leg = 0; leg < kernels_.size(); ++leg) {
        const auto data = kernels_[leg].description();
        const auto& pending = data.pending;
        result.legs[leg] = {
            .session = policy_ids[leg],
            .command = {.sequence = pending.command.sequence,
                        .acceptance_order = pending.command.acceptance_order,
                        .physical_generation = pending.command.physical_generation,
                        .geometry_epoch = pending.command.geometry_epoch,
                        .command_epoch = pending.command.command_epoch,
                        .control_epoch = pending.control_epoch,
                        .member_generation = pending.member_generation,
                        .configuration_epoch = pending.configuration_epoch,
                        .physical_configuration_epoch = pending.physical_configuration_epoch,
                        .is_resume = pending.command.is_resume},
            .catalog_generation = data.catalog_generation,
            .workers = data.variant.workers};
    }
    return result;
}

std::array<core::ExecutionPairContext, 2> CudaSessionPairSelection::contexts() const noexcept {
    std::array<core::ExecutionPairContext, 2> result{};
    for (std::size_t leg = 0; leg < kernels_.size(); ++leg) {
        const auto data = kernels_[leg].description();
        result[leg] = {data.execution_domain, data.catalog_generation, data.variant.workers,
                       data.is_configuration_materialized};
    }
    return result;
}

namespace {
struct PairResourceCatalog {
    CudaPairProfileLimits limits;
    std::array<CudaProfileGeometry, 2> geometry;
    std::array<CudaPairProfileCandidate, 64> candidates{};
    std::size_t count{};
};
std::optional<PairResourceCatalog> pair_resource_catalog(
    const std::array<CudaSessionPairKernelDescription, 2>& kernels,
    std::span<const unsigned int> worker_counts) noexcept {
    if (worker_counts.empty() || worker_counts.size() > 8) return std::nullopt;
    PairResourceCatalog result;
    result.limits = kernels[0].limits;
    auto other = kernels[1].limits;
    result.limits.max_workers = other.max_workers =
        std::max(result.limits.max_workers, other.max_workers);
    if (result.limits != other) return std::nullopt;
    for (std::size_t index = 0; index < worker_counts.size(); ++index) {
        const auto workers = worker_counts[index];
        if (!workers || workers > static_cast<unsigned int>(INT32_MAX) || !result.limits.sm_count ||
            workers % result.limits.sm_count ||
            std::ranges::find(worker_counts.first(index), workers) !=
                worker_counts.first(index).end())
            return std::nullopt;
    }
    result.geometry = {kernels[0].geometry, kernels[1].geometry};
    for (const auto left : worker_counts)
        for (const auto right : worker_counts) {
            if (left > kernels[0].limits.max_workers || right > kernels[1].limits.max_workers)
                continue;
            CudaPairProfileCandidate candidate{{kernels[0].variant, kernels[1].variant}};
            candidate.legs[0].workers = left;
            candidate.legs[1].workers = right;
            if (cuda_pair_profile_candidate_fits(result.geometry, result.limits, candidate))
                result.candidates[result.count++] = candidate;
        }
    return result;
}
}  // namespace

CudaSessionPairSelectionResult CudaExecutionSession::inspect_pair(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    if (&first == &second || worker_counts.empty() || worker_counts.size() > 8)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    std::array<CudaExecutionSession*, 2> sessions{&first, &second};
    std::array<CudaSessionPairKernel, 2> kernels{first_kernel, second_kernel};
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto refreshed =
            sessions[leg]->refresh_pair_kernel(kernels[leg], kernels[leg].command_);
        if (refreshed.status != CUDA_SUCCESS) return {.status = refreshed.status};
        kernels[leg] = refreshed.kernel;
    }
    const auto cache = profiles.snapshot();
    if (!cache.is_valid || cache.is_learning) return {.status = CUDA_ERROR_INVALID_VALUE};
    const auto catalog =
        pair_resource_catalog({kernels[0].description_, kernels[1].description_}, worker_counts);
    if (!catalog) return {.status = CUDA_ERROR_INVALID_VALUE};
    const bool can_share = first.can_share_configuration(second);
    const CudaPairProfileDomain domain =
        can_share ? CudaPairProfileDomain{CudaPairProfileDomainKind::kSameContext,
                                          {kernels[0].description_.execution_domain,
                                           kernels[1].description_.execution_domain}}
                  : CudaPairProfileDomain{CudaPairProfileDomainKind::kDistinctContexts, {1, 2}};
    auto choice = profiles.lookup(
        {kernels[0].description_.profile, kernels[1].description_.profile}, catalog->geometry,
        domain, catalog->limits, std::span{catalog->candidates}.first(catalog->count));
    // Host overlap between separate native contexts is not spatial sharing.
    if (!can_share) {
        const auto originals = choice.original;
        const auto first_original = choice.first_original;
        choice = {.first_original = first_original, .original = originals};
    }
    CudaSessionPairSelectionResult result;
    result.selection.kernels_ = std::move(kernels);
    result.selection.choice_ = choice;
    return result;
}

CudaSessionPairChoiceResult CudaExecutionSession::evaluate_pair(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    const auto inspected =
        inspect_pair(first, first_kernel, second, second_kernel, profiles, worker_counts);
    return {inspected.status, inspected.selection.choice_};
}

CudaSessionPairSelectionResult CudaExecutionSession::select_pair(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    auto inspected =
        inspect_pair(first, first_kernel, second, second_kernel, profiles, worker_counts);
    if (inspected.status != CUDA_SUCCESS) return inspected;
    return configure_pair({&first, &second}, std::move(inspected.selection.kernels_),
                          inspected.selection.choice_);
}

CudaSessionPairSelectionResult CudaExecutionSession::configure_pair(
    std::array<CudaExecutionSession*, 2> sessions, std::array<CudaSessionPairKernel, 2> kernels,
    CudaPairProfileChoice choice,
    const std::array<std::shared_ptr<CudaExecutionObserver>, 2>& observations) noexcept {
    if (choice.first_original > 1 || (choice.is_shared && !choice.candidate) ||
        (bool(observations[0]) != bool(observations[1])) || (observations[0] && !choice.is_shared))
        return {.status = CUDA_ERROR_INVALID_VALUE};
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto& pending = kernels[leg].description_.pending.command;
        // A cache miss cannot restart a partially executed original. Keep its
        // prepared family/counter/observer and resume only the selected serial leg.
        const bool uses_original = !choice.is_shared && !pending.is_resume;
        const auto workers = choice.is_shared ? choice.configuration.legs[leg].workers
                             : uses_original  ? 0
                                              : pending.physical_grid[0];
        std::shared_ptr<CudaExecutionObserver> observation;
        try {
            auto& state = *sessions[leg]->implementation_;
            const auto function = std::ranges::find(state.storage->functions, pending.function,
                                                    &Implementation::Function::id);
            if (function == state.storage->functions.end())
                return {.status = CUDA_ERROR_INVALID_HANDLE};
            observation = state.profile_observation(*function, kernels[leg].description_.geometry,
                                                    observations[leg]);
        } catch (...) {
            return {.status = CUDA_ERROR_OUT_OF_MEMORY};
        }
        const auto selected = sessions[leg]->select_pending_kernel(
            kernels[leg].command_, {.workers = workers,
                                    .uses_original = uses_original,
                                    .observation = std::move(observation)});
        if (selected.status != CUDA_SUCCESS) return {.status = selected.status};
        const auto refreshed = sessions[leg]->refresh_pair_kernel(kernels[leg], selected.command);
        if (refreshed.status != CUDA_SUCCESS) return {.status = refreshed.status};
        kernels[leg] = refreshed.kernel;
    }
    CudaSessionPairSelectionResult result;
    result.selection.kernels_ = std::move(kernels);
    result.selection.choice_ = choice;
    result.selection.is_profile_sample_ = observations[0] != nullptr;
    return result;
}

CudaSessionPairSampleResult CudaExecutionSession::reserve_pair_sample(
    CudaExecutionSession& first, const CudaSessionPairKernel& first_kernel,
    CudaExecutionSession& second, const CudaSessionPairKernel& second_kernel,
    CudaPairProfileCache& profiles, std::span<const unsigned int> worker_counts) noexcept {
    if (&first == &second) return {.status = CUDA_ERROR_INVALID_VALUE};
    std::array<CudaExecutionSession*, 2> sessions{&first, &second};
    std::array<CudaSessionPairKernel, 2> kernels{first_kernel, second_kernel};
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto fresh = sessions[leg]->refresh_pair_kernel(kernels[leg], kernels[leg].command_);
        if (fresh.status != CUDA_SUCCESS) return {.status = fresh.status};
        kernels[leg] = fresh.kernel;
        const auto& command = kernels[leg].description_.pending.command;
        if (command.is_resume || command.physical_generation || command.command_epoch != 1)
            return {.status = CUDA_ERROR_NOT_SUPPORTED};
    }
    const auto cache = profiles.snapshot();
    if (!cache.is_valid || !cache.is_learning) return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!first.can_share_configuration(second)) return {.status = CUDA_ERROR_NOT_READY};
    const auto catalog =
        pair_resource_catalog({kernels[0].description_, kernels[1].description_}, worker_counts);
    if (!catalog) return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!catalog->count) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    const CudaPairProfileDomain domain{
        CudaPairProfileDomainKind::kSameContext,
        {kernels[0].description_.execution_domain, kernels[1].description_.execution_domain}};
    std::array<CudaPairProfileKernel, 2> bound{};
    for (std::size_t leg = 0; leg < bound.size(); ++leg) {
        if (!kernels[leg].description_.profile) return {.status = CUDA_ERROR_NOT_SUPPORTED};
        auto binding =
            profiles.bind_checked(kernels[leg].description_.profile, catalog->geometry[leg]);
        switch (binding.status) {
            case CudaPairProfileBindStatus::kBound:
                bound[leg] = std::move(binding.kernel);
                break;
            case CudaPairProfileBindStatus::kCapacity:
                return {.status = CUDA_ERROR_OUT_OF_MEMORY};
            case CudaPairProfileBindStatus::kMissingOriginal:
                return {.status = CUDA_ERROR_NOT_SUPPORTED};
            case CudaPairProfileBindStatus::kInvalid:
                return {.status = CUDA_ERROR_INVALID_VALUE};
        }
    }
    const auto candidates = std::span{catalog->candidates}.first(catalog->count);
    auto reserved = profiles.reserve_checked(bound, domain, catalog->limits, candidates);
    using Outcome = CudaPairProfileReserveStatus;
    if (reserved.status == Outcome::kBusy) return {.status = CUDA_ERROR_NOT_READY};
    if (reserved.status == Outcome::kCapacity) return {.status = CUDA_ERROR_OUT_OF_MEMORY};
    if (reserved.status == Outcome::kInvalid) return {.status = CUDA_ERROR_INVALID_VALUE};
    auto reservation = std::move(reserved.reservation);
    auto choice = profiles.choose(bound, domain, catalog->limits, candidates);
    if (reserved.status == Outcome::kReserved && reservation)
        choice = {.is_shared = true,
                  .candidate = reservation.candidate,
                  .configuration = reservation.configuration};
    else if (!choice.is_trained)
        return {.status = CUDA_ERROR_UNKNOWN};
    auto configured =
        configure_pair(sessions, std::move(kernels), choice, reservation.observations);
    if (configured.status != CUDA_SUCCESS) {
        for (const auto& observer : reservation.observations)
            if (observer) observer->finish(false);
        return {.status = configured.status};
    }
    return {CUDA_SUCCESS, std::move(configured.selection), std::move(reservation)};
}

CudaSessionPairIssueResult CudaExecutionSession::issue_pair(
    CudaExecutionSession& first, CudaExecutionSession& second,
    const CudaSessionPairSelection& selection, const core::ExecutionPairPlan& authorization,
    std::array<std::uint64_t, 2> policy_ids) noexcept {
    for (const auto* session : {&first, &second}) {
        if (session->implementation_->has_graph_capture()) return {.status = CUDA_ERROR_NOT_READY};
        const auto& memory = session->implementation_->process_memory;
        if (memory && memory->native_health() != CUDA_SUCCESS)
            return {.status = memory->native_health()};
    }
    if (&first == &second || !selection || !policy_ids[0] || !policy_ids[1] ||
        policy_ids[0] == policy_ids[1] || !authorization.is_valid ||
        !authorization.priority.is_valid ||
        authorization.priority.size > authorization.priority.decisions.size())
        return {.status = CUDA_ERROR_INVALID_VALUE};
    if (!authorization.grant_count) return {.status = CUDA_ERROR_NOT_READY};
    const auto proposed = selection.proposal(policy_ids);
    const auto count = proposed.is_shared ? 2U : 1U;
    if (authorization.grant_count != count || authorization.is_shared != proposed.is_shared ||
        proposed.first_serial > 1)
        return {.status = CUDA_ERROR_INVALID_VALUE};
    for (std::size_t index = 0; index < count; ++index) {
        const auto leg = proposed.is_shared ? index : proposed.first_serial;
        if (authorization.grants[index] != proposed.legs[leg])
            return {.status = CUDA_ERROR_INVALID_HANDLE};
    }
    std::size_t executes{};
    for (const auto& decision :
         std::span{authorization.priority.decisions}.first(authorization.priority.size)) {
        if (decision.action == core::ExecutionPriorityAction::kExecuteProgress ||
            decision.action == core::ExecutionPriorityAction::kRequestRetreat ||
            decision.action == core::ExecutionPriorityAction::kQuarantined)
            return {.status = CUDA_ERROR_NOT_READY};
        if (decision.action != core::ExecutionPriorityAction::kExecute) continue;
        const auto grant = std::ranges::find(std::span{authorization.grants}.first(count),
                                             decision.session, &core::ExecutionPairLeg::session);
        if (grant == std::span{authorization.grants}.first(count).end() ||
            decision.command != grant->command ||
            std::ranges::count(
                std::span{authorization.priority.decisions}.first(authorization.priority.size),
                decision.session, &core::ExecutionPriorityDecision::session) != 1)
            return {.status = CUDA_ERROR_INVALID_HANDLE};
        ++executes;
    }
    if (executes != count) return {.status = CUDA_ERROR_INVALID_HANDLE};
    std::array<CudaExecutionSession*, 2> sessions{&first, &second};
    // Validate BOTH opaque commands before any first-leg native effect. There
    // is no intake/poll/callback turn between these exact owner-serialized grants.
    for (std::size_t leg = 0; leg < sessions.size(); ++leg) {
        const auto& kernel = selection.kernels_[leg];
        const auto current = sessions[leg]->refresh_pair_kernel(kernel, kernel.command_);
        if (current.status != CUDA_SUCCESS) return {.status = current.status};
    }
    if (proposed.is_shared && !first.can_share_configuration(second))
        return {.status = CUDA_ERROR_NOT_READY};
    CudaSessionPairIssueResult result;
    for (std::size_t index = 0; index < count; ++index) {
        const auto leg = proposed.is_shared ? index : proposed.first_serial;
        const auto& kernel = selection.kernels_[leg];
        const auto current = sessions[leg]->refresh_pair_kernel(kernel, kernel.command_);
        if (current.status != CUDA_SUCCESS) {
            result.status = current.status;
            return result;
        }
        if (proposed.is_shared && !first.can_share_configuration(second)) {
            result.status = CUDA_ERROR_NOT_READY;
            return result;
        }
        const auto before =
            sessions[leg]->persistent_progress(kernel.description_.pending.command.sequence);
        result.attempted[leg] = true;
        auto& issued = result.issues[leg];
        issued = sessions[leg]->issue(kernel.command_);
        if (issued.state != CudaExecutionIssueState::kIssued ||
            issued.native_status != CUDA_SUCCESS || issued.completion_status != CUDA_SUCCESS ||
            sessions[leg]->snapshot().is_quarantined) {
            result.status = issued.native_status != CUDA_SUCCESS       ? issued.native_status
                            : issued.completion_status != CUDA_SUCCESS ? issued.completion_status
                                                                       : CUDA_ERROR_UNKNOWN;
            return result;
        }
        if (proposed.is_shared) {
            const auto after = sessions[leg]->persistent_progress(issued.sequence);
            if (!before || !after || before->physical_launches == UINT64_MAX ||
                after->physical_launches != before->physical_launches + 1 ||
                after->physical_stops >= after->physical_launches) {
                sessions[leg]->implementation_->fault();
                result.status = CUDA_ERROR_UNKNOWN;
                return result;
            }
        }
    }
    return result;
}

CudaSessionResult CudaExecutionSession::create_event(unsigned int flags) noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return rejected(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return rejected(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto checked = state.check_context(true);
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (state.is_faulted || state.is_closing) return rejected(CUDA_ERROR_UNKNOWN);
    if (!state.has_id_space || state.storage->events.size() == state.limits.max_objects)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    const auto cleaned = state.collect_events();
    if (cleaned != CUDA_SUCCESS) return rejected(cleaned);
    std::shared_ptr<CudaSessionEvent::State> generation;
    auto result = state.make_event(flags, generation);
    if (result.status != CUDA_SUCCESS) return result;
    result.id = state.consume_id();
    generation->identity = {result.id, 0, 0};
    state.storage->events.push_back({result.id, flags, std::move(generation)});
    return result;
}
CudaSessionResult CudaExecutionSession::destroy_event(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return rejected(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto checked = state.check_context(true);
    if (checked != CUDA_SUCCESS) return rejected(checked);
    const auto event = std::ranges::find(state.storage->events, id, &Implementation::Event::id);
    if (event == state.storage->events.end()) return rejected(CUDA_ERROR_INVALID_HANDLE);
    state.storage->events.erase(event);  // Existing immutable borrows remain valid.
    const auto cleaned = state.collect_events();
    return cleaned == CUDA_SUCCESS ? settled() : rejected(cleaned);
}
CudaSessionResult CudaExecutionSession::record_event(std::uint64_t sequence, std::uint64_t event,
                                                     std::uint64_t stream,
                                                     unsigned int flags) noexcept {
    return implementation_
        ->submit_event({.sequence = sequence, .event = event, .stream = stream, .flags = flags},
                       false, false)
        .result;
}
CudaSessionResult CudaExecutionSession::wait_event(std::uint64_t sequence, std::uint64_t event,
                                                   std::uint64_t stream,
                                                   unsigned int flags) noexcept {
    return implementation_
        ->submit_event({.sequence = sequence, .event = event, .stream = stream, .flags = flags},
                       true, false)
        .result;
}
CudaSessionEventSubmission CudaExecutionSession::record_event_with_frontier(
    std::uint64_t sequence, std::uint64_t event, std::uint64_t stream,
    unsigned int flags) noexcept {
    return implementation_->submit_event(
        {.sequence = sequence, .event = event, .stream = stream, .flags = flags}, false, true);
}
CudaSessionEventSubmission CudaExecutionSession::wait_event_with_frontier(
    std::uint64_t sequence, std::uint64_t event, std::uint64_t stream,
    unsigned int flags) noexcept {
    return implementation_->submit_event(
        {.sequence = sequence, .event = event, .stream = stream, .flags = flags}, true, true);
}
CudaSessionEventCapture CudaExecutionSession::capture_event(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || state.has_destroy_attempt || state.is_closing)
        return {.status = CUDA_ERROR_INVALID_CONTEXT, .event = {}};
    if (state.is_faulted || !state.executor || state.executor->snapshot().is_quarantined)
        return {.status = CUDA_ERROR_UNKNOWN, .event = {}};
    const auto event = std::ranges::find(state.storage->events, id, &Implementation::Event::id);
    if (event == state.storage->events.end()) return {};
    if (state.storage->event_borrows.load(std::memory_order_acquire) >= state.limits.max_objects)
        return {.status = CUDA_ERROR_OUT_OF_MEMORY, .event = {}};
    try {
        CudaSessionEvent result;
        result.state_ = event->current;
        result.owner_ = state.storage;
        result.thread_ = resource_owner;
        result.process_ = static_cast<std::uint64_t>(state.process);
        result.lifetime_ = std::make_shared<Implementation::EventLease>(state.storage);
        const auto generation = result.diagnostic_identity();
        return {CUDA_SUCCESS, std::move(result), event->current->frontier, generation};
    } catch (...) {
        return {.status = CUDA_ERROR_OUT_OF_MEMORY, .event = {}};
    }
}
CudaSessionResult CudaExecutionSession::wait_event(std::uint64_t sequence,
                                                   const CudaSessionEvent& source,
                                                   std::uint64_t stream,
                                                   unsigned int flags) noexcept {
    return wait_event_impl({.sequence = sequence, .stream = stream, .flags = flags}, source, false)
        .result;
}
CudaSessionEventSubmission CudaExecutionSession::wait_event_with_frontier(
    std::uint64_t sequence, const CudaSessionEvent& source, std::uint64_t stream,
    unsigned int flags) noexcept {
    return wait_event_impl({.sequence = sequence, .stream = stream, .flags = flags}, source, true);
}
CudaSessionEventSubmission CudaExecutionSession::wait_event_impl(EventCommand parameters,
                                                                 const CudaSessionEvent& source,
                                                                 bool captures_frontier) noexcept {
    const auto sequence = parameters.sequence;
    const auto stream = parameters.stream;
    const auto flags = parameters.flags;
    auto& state = *implementation_;
    const auto failure = [](CUresult status) {
        return CudaSessionEventSubmission{rejected(status), {}};
    };
    if (!state.owns()) return failure(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return failure(CUDA_ERROR_NOT_READY);
    const NativeExecutionScope scope(state.is_native_issue_active);
    const auto checked = state.check_context(true);
    if (checked != CUDA_SUCCESS) return failure(checked);
    if (!state.executor || state.is_faulted || state.is_closing) return failure(CUDA_ERROR_UNKNOWN);
    const auto dependency = source.observe();
    if (dependency != CUDA_SUCCESS && dependency != CUDA_ERROR_NOT_READY)
        return failure(dependency);
    if (flags) return failure(CUDA_ERROR_INVALID_VALUE);
    if (!state.dispatch.stream_wait_event) return failure(CUDA_ERROR_NOT_SUPPORTED);
    if (stream && !std::ranges::any_of(state.storage->streams,
                                       [stream](const auto& value) { return value.id == stream; }))
        return failure(CUDA_ERROR_INVALID_HANDLE);
    const auto cleaned = state.collect_events();
    if (cleaned != CUDA_SUCCESS) return failure(cleaned);
    if (state.executor->snapshot().outstanding >= state.limits.submissions.max_commands) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return {deferred, {}};
    }
    try {
        auto call = std::make_shared<Implementation::EventCall>();
        call->generation = source.state_;
        call->source = source;
        call->dispatch = state.dispatch;
        call->flags = flags;
        call->is_wait = true;
        CudaExecutionEvent command{stream, call, {}};
        if (source.owner_.lock() == state.storage)
            command.dependency = source.state_->frontier;
        else
            command.foreign_dependency = source.state_->frontier;
        auto submitted =
            captures_frontier
                ? state.executor->accept_event_with_frontier(sequence, command)
                : CudaExecutionEventAcceptance{state.executor->accept_event(sequence, command), {}};
        if (!submitted.acceptance.accepted()) return failure(CUDA_ERROR_OUT_OF_MEMORY);
        const auto generation = call->generation->identity;
        state.pending_events.push_back({submitted.acceptance.receipt, std::move(call)});
        auto result = settled();
        result.id = sequence;
        return {result, std::move(submitted.frontier), generation};
    } catch (...) {
        return failure(CUDA_ERROR_OUT_OF_MEMORY);
    }
}
CudaSessionResult CudaExecutionSession::query_event(const CudaSessionEvent& event,
                                                    bool should_synchronize) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (!event.state_ || event.owner_.lock() != state.storage)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    // An earlier record/preparation error is observable here, but does not
    // mean that this requested query/synchronize SDK operation was called.
    if (event.state_->record_status != CUDA_SUCCESS) return rejected(event.state_->record_status);
    if (event.state_->frontier) {
        const auto status = state.executor->query_frontier(*event.state_->frontier);
        if (status != CUDA_SUCCESS) return rejected(status);
    }
    if (should_synchronize) {
        if (!state.dispatch.event_synchronize) return rejected(CUDA_ERROR_NOT_SUPPORTED);
        return completed(state.invoke(state.dispatch.event_synchronize, event.state_->native));
    }
    return completed(state.invoke(state.dispatch.execution.event_query, event.state_->native));
}
CudaSessionResult CudaExecutionSession::event_elapsed_time(std::uint64_t start,
                                                           std::uint64_t end) noexcept {
    auto& state = *implementation_;
    const auto checked = state.check_context();
    if (checked != CUDA_SUCCESS) return rejected(checked);
    if (!state.dispatch.event_elapsed_time) return rejected(CUDA_ERROR_NOT_SUPPORTED);
    const auto first = capture_event(start);
    if (first.status != CUDA_SUCCESS) return rejected(first.status);
    const auto second = start == end ? first : capture_event(end);
    if (second.status != CUDA_SUCCESS) return rejected(second.status);
    for (const auto* event : {&first.event, &second.event}) {
        auto ready = query_event(*event);
        if (ready.status != CUDA_SUCCESS) {
            ready.original_status.reset();  // Preparatory observation, not elapsed-time issue.
            return ready;
        }
    }
    float milliseconds{};
    auto result = completed(state.invoke(state.dispatch.event_elapsed_time, &milliseconds,
                                         first.event.state_->native, second.event.state_->native));
    if (result.status == CUDA_SUCCESS) result.address = std::bit_cast<std::uint32_t>(milliseconds);
    return result;
}
CudaExecutionPollResult CudaExecutionSession::poll() noexcept {
    auto& state = *implementation_;
    if (state.has_graph_capture()) return {.status = CUDA_ERROR_NOT_READY};
    if (!state.owns() || !state.executor) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    auto result = state.executor->poll();
    if (result.status == CUDA_SUCCESS) result.status = state.collect_events();
    if (state.uses_shared_domain && result.status != CUDA_SUCCESS &&
        result.status != CUDA_ERROR_NOT_READY)
        state.fault();
    state.update_domain_physical();
    if (state.uses_shared_domain && state.is_faulted && result.status == CUDA_SUCCESS)
        result.status = CUDA_ERROR_UNKNOWN;
    return result;
}
CUresult CudaExecutionSession::request_retreat(std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return CUDA_ERROR_NOT_READY;
    const NativeExecutionScope active{state.is_native_issue_active};
    return state.executor->request_retreat(sequence);
}
CUresult CudaExecutionSession::retarget_workers(std::uint64_t sequence,
                                                CudaExecutionWorkerGeometry geometry) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active ||
        state.is_preparation_active)
        return CUDA_ERROR_NOT_READY;
    if (!geometry.workers || geometry.workers > state.persistent.workers)
        return CUDA_ERROR_INVALID_VALUE;
    const NativeExecutionScope active{state.is_native_issue_active};
    return state.executor->retarget_workers(sequence, geometry);
}

CudaExecutionIssueResult CudaExecutionSession::resume(std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted)
        return {.state = CudaExecutionIssueState::kError,
                .native_status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    if (state.storage->domain_member) {
        const auto configured = state.storage->domain_member.configuration_status(true);
        if (configured != CUDA_SUCCESS)
            return {.state = CudaExecutionIssueState::kError, .native_status = configured};
    }
    if (!state.reserve_domain_physical())
        return {.state = CudaExecutionIssueState::kError, .native_status = CUDA_ERROR_UNKNOWN};
    const auto result = state.executor->resume(sequence);
    state.update_domain_physical();
    return result;
}
std::optional<CudaSessionPersistentProgress> CudaExecutionSession::persistent_progress(
    std::uint64_t sequence) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return std::nullopt;
    for (const auto& entry : state.progress)
        if (const auto progress = entry.lock(); progress && progress->sequence == sequence)
            return progress->snapshot();
    return std::nullopt;
}
CudaExecutionPollResult CudaExecutionSession::drain_issued() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor) return {.status = CUDA_ERROR_INVALID_CONTEXT};
    if (state.is_native_issue_active || state.is_folded_active || state.is_rnn_query_active)
        return {.status = CUDA_ERROR_NOT_READY};
    const NativeExecutionScope active{state.is_native_issue_active};
    auto result = state.executor->drain_issued();
    if (state.uses_shared_domain && result.status != CUDA_SUCCESS &&
        result.status != CUDA_ERROR_NOT_READY)
        state.fault();
    state.update_domain_physical();
    if (state.uses_shared_domain && state.is_faulted && result.status == CUDA_SUCCESS)
        result.status = CUDA_ERROR_UNKNOWN;
    return result;
}
std::optional<ExecutionHostReply> CudaExecutionSession::host_result(
    const ExecutionReceipt& receipt) const noexcept {
    const auto& state = *implementation_;
    if (getpid() != state.process || !state.executor) return std::nullopt;
    return state.executor->host_result(receipt);
}
std::optional<std::array<std::uint64_t, 2>> CudaExecutionSession::native_kernel_pair()
    const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor || state.is_faulted) return std::nullopt;
    return state.executor->native_kernel_pair();
}
CudaExecutionFrontierResult CudaExecutionSession::capture_frontier(
    std::optional<std::uint64_t> stream, bool includes_submission_dependencies) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor)
        return {.status = CUDA_ERROR_INVALID_CONTEXT, .frontier = {}};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN, .frontier = {}};
    return state.executor->capture_frontier(stream, includes_submission_dependencies);
}
CudaExecutionFrontierResult CudaExecutionSession::capture_command_frontier(
    std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || !state.executor)
        return {.status = CUDA_ERROR_INVALID_CONTEXT, .frontier = {}};
    if (state.is_faulted) return {.status = CUDA_ERROR_UNKNOWN, .frontier = {}};
    return state.executor->capture_command_frontier(sequence);
}
CudaExecutionFrontierReservationResult CudaExecutionSession::reserve_command_frontier(
    std::uint64_t sequence) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_capture_gate(); status != CUDA_SUCCESS)
        return {.status = status};
    return state.executor->reserve_command_frontier(sequence);
}
CudaExecutionFrontierResult CudaExecutionSession::reserved_command_frontier(
    const CudaExecutionFrontierReservation& reservation) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS)
        return {.status = status, .frontier = {}};
    return state.executor->reserved_command_frontier(reservation);
}
CUresult CudaExecutionSession::cancel_frontier_reservation(
    const CudaExecutionFrontierReservation& reservation) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.external_transition_gate(); status != CUDA_SUCCESS) return status;
    return state.executor->cancel_frontier_reservation(reservation);
}
bool CudaExecutionSession::reject_command(const ExecutionReceipt& receipt,
                                          CUresult failure) noexcept {
    auto& state = *implementation_;
    if (state.external_transition_gate() != CUDA_SUCCESS || state.is_faulted) return false;
    return state.executor->reject_command(receipt, failure);
}
CUresult CudaExecutionSession::query_frontier(
    const CudaExecutionFrontier& frontier) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.executor) return CUDA_ERROR_INVALID_CONTEXT;
    return state.executor->query_frontier(frontier);
}
bool CudaExecutionSession::acknowledge_result(const ExecutionReceipt& receipt) noexcept {
    auto& state = *implementation_;
    return getpid() == state.process && state.executor &&
           state.executor->acknowledge_result(receipt);
}
CudaSessionSnapshot CudaExecutionSession::snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) {
        CudaSessionSnapshot rejected;
        rejected.is_quarantined = true;
        return rejected;
    }
    CudaSessionSnapshot result;
    result.device_bytes = state.storage->device_bytes;
    result.image_bytes = state.storage->image_bytes;
    result.allocations = state.storage->allocations.size();
    result.modules = state.storage->modules.size();
    result.streams = state.storage->streams.size();
    result.functions = state.storage->functions.size();
    result.private_host_bytes = state.storage->host_bytes;
    result.default_proxy_staging_bytes = state.storage->staging_bytes;
    result.persistent_slots =
        static_cast<std::size_t>(std::ranges::count(state.storage->occupied, true));
    result.persistent_functions = static_cast<std::size_t>(std::ranges::count_if(
        state.storage->functions, [](const auto& function) { return !function.variants.empty(); }));
    result.persistent_accepted = state.persistent_accepted;
    result.host_memory_bytes = state.storage->host_memory_bytes;
    result.host_allocations = state.storage->host_allocations.size();
    result.blas_handles = state.storage->blas.size();
    result.lt_handles = state.storage->lt.size();
    result.random_handles = state.storage->random.size();
    result.external_context_leases = static_cast<std::size_t>(std::ranges::count_if(
        state.external_contexts, [](const auto& entry) { return !entry.expired(); }));
    result.external_command_leases = static_cast<std::size_t>(std::ranges::count_if(
        state.external_commands, [](const auto& entry) { return !entry.expired(); }));
#ifdef GLIMMER_HAS_CUDNN
    result.dnn_handles = state.storage->dnn.size();
#endif
    // Preserve resource observation before the dispatcher's readiness callbacks.
    static_cast<CudaSessionSchedulingSnapshot&>(result) = scheduling_snapshot();
    return result;
}
CudaSessionHealthSnapshot CudaExecutionSession::health_snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {.is_quarantined = true};
    return {.has_context = state.storage->context != nullptr,
            .is_quarantined = state.is_faulted || state.is_creation_unproven ||
                              (state.executor && state.executor->is_quarantined())};
}
CudaSessionSchedulingSnapshot CudaExecutionSession::scheduling_snapshot() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return {.is_quarantined = true};
    CudaSessionSchedulingSnapshot result{
        .has_context = state.storage->context != nullptr,
        .is_quarantined = state.is_faulted || state.is_creation_unproven};
    if (state.executor) {
        const auto scheduling = state.executor->scheduling_snapshot();
        result.submissions = scheduling.submissions;
        result.external_barriers = scheduling.external_barriers;
        result.enqueue_turns = scheduling.enqueue_turns;
        result.external_work_in_flight = scheduling.external_work_in_flight;
        result.has_ready_work = scheduling.has_ready_work;
        result.is_quarantined = result.is_quarantined || result.submissions.is_quarantined;
        if (scheduling.has_pending_preparation && !state.has_destroy_attempt &&
            !state.is_native_issue_active && !state.is_folded_active &&
            !state.is_rnn_query_active && !state.is_preparation_active && !state.is_faulted) {
            result.has_pending_preparation =
                !state.storage->domain_member ||
                state.storage->domain_member.view(true).status == CUDA_SUCCESS;
        }
    }
    return result;
}
void CudaExecutionSession::close() noexcept {
    auto& state = *implementation_;
    if (state.owns() && !state.is_native_issue_active) {
        state.is_closing = true;
        if (state.executor) state.executor->close();
    }
}
bool CudaExecutionSession::can_cleanup_quarantined() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns() || !state.storage->context || !state.executor || state.is_faulted ||
        state.is_creation_unproven || state.has_destroy_attempt || state.is_native_issue_active ||
        state.is_folded_active || state.is_rnn_query_active || state.has_external_contexts() ||
        state.storage->event_borrows.load(std::memory_order_acquire))
        return false;
    const auto queue = state.executor->snapshot();
    if (!queue.is_quarantined || !state.executor->physical_readers_quiescent()) return false;
    const auto known = [](const auto& handle) {
        return handle.native && !handle.is_uncertain && !handle.has_destroy_attempt;
    };
    if (!std::ranges::all_of(state.storage->blas, known) ||
        !std::ranges::all_of(state.storage->lt, known) ||
        !std::ranges::all_of(state.storage->random, known))
        return false;
#ifdef GLIMMER_HAS_CUDNN
    if (!std::ranges::all_of(state.storage->dnn, known)) return false;
#endif
    return true;
}
std::size_t CudaExecutionSession::reject_queued(CUresult failure) noexcept {
    auto& state = *implementation_;
    return state.owns() && state.executor && !state.is_native_issue_active &&
                   !state.is_folded_active && !state.is_rnn_query_active
               ? state.executor->reject_queued(failure)
               : 0;
}
CudaSessionResult CudaExecutionSession::shutdown() noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.is_folded_active || state.is_rnn_query_active || state.is_native_issue_active) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    state.is_closing = true;
    if (!state.storage->context) {
        if (state.is_creation_unproven || (state.uses_shared_domain && state.is_faulted))
            return rejected(CUDA_ERROR_UNKNOWN);
        if (state.storage->domain_member) {
            const auto member = state.storage->domain_member.view();
            if (member.status != CUDA_SUCCESS) return rejected(member.status);
            if (member.is_attached && state.storage->domain_member.retire() != CUDA_SUCCESS)
                return rejected(CUDA_ERROR_UNKNOWN);
            state.storage->domain_member = {};
        }
        return settled();
    }
    if (state.process_memory) {
        const auto health = state.process_memory->native_health();
        if (health != CUDA_SUCCESS) return rejected(health);
    }
    if (state.has_destroy_attempt) return rejected(CUDA_ERROR_INVALID_CONTEXT);
    if (state.uses_shared_domain && state.is_faulted) return rejected(CUDA_ERROR_UNKNOWN);
    if (state.graphs) {
        const auto status = state.graphs->shutdown();
        if (status == CUDA_ERROR_NOT_READY) return deferred_resource_release();
        if (status != CUDA_SUCCESS) return rejected(status);
    }
    if (state.has_external_contexts()) return deferred_resource_release();
    if (state.storage->event_borrows.load(std::memory_order_acquire)) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    if (state.executor) {
        const auto status = state.check_context();
        if (status != CUDA_SUCCESS) return rejected(status);
    }
    if (state.executor && !state.executor->shutdown()) {
        auto deferred = rejected(CUDA_ERROR_NOT_READY);
        deferred.disposition = CudaSessionDisposition::kDeferred;
        return deferred;
    }
    state.update_domain_physical();
    if (state.uses_shared_domain) {
        const auto quiescent = state.domain_cold_status();
        if (quiescent == CUDA_ERROR_NOT_READY) return deferred_resource_release();
        if (quiescent != CUDA_SUCCESS) return rejected(quiescent);
    }
    for (auto& allocation : state.storage->host_allocations) allocation.is_retiring = true;
    while (!state.storage->host_allocations.empty()) {
        auto released = state.release_host_at(0);
        if (released.status != CUDA_SUCCESS) return released;
    }
#ifdef GLIMMER_HAS_CUDNN
    while (!state.storage->dnn.empty()) {
        auto& handle = state.storage->dnn.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.dnn->destroy, handle.native);
        if (status != CUDNN_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->dnn.pop_back();
        if (state.check_context() != CUDA_SUCCESS) {
            state.fault();
            return rejected(CUDA_ERROR_UNKNOWN);
        }
    }
#endif
    while (!state.storage->random.empty()) {
        auto& handle = state.storage->random.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.random.destroy, handle.native);
        if (status != CURAND_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            spdlog::error("Owned cuRAND shutdown failed: context={}, handle={}, library_status={}",
                          state.context_id, handle.id, static_cast<int>(status));
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->random.pop_back();
    }
    while (!state.storage->blas.empty()) {
        auto& handle = state.storage->blas.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.blas.destroy, handle.native);
        if (status != CUBLAS_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            spdlog::error("Owned cuBLAS shutdown failed: context={}, handle={}, library_status={}",
                          state.context_id, handle.id, static_cast<int>(status));
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->blas.pop_back();
    }
    while (!state.storage->lt.empty()) {
        auto& handle = state.storage->lt.back();
        if (handle.is_uncertain || handle.has_destroy_attempt) return rejected(CUDA_ERROR_UNKNOWN);
        handle.has_destroy_attempt = true;
        const auto status = state.invoke(state.dispatch.lt.destroy, handle.native);
        if (status != CUBLAS_STATUS_SUCCESS) {
            handle.is_uncertain = true;
            state.fault();
            spdlog::error(
                "Owned cuBLASLt shutdown failed: context={}, handle={}, library_status={}",
                state.context_id, handle.id, static_cast<int>(status));
            return rejected(CUDA_ERROR_UNKNOWN);
        }
        state.storage->lt.pop_back();
    }
    state.executor.reset();
    const auto cleanup = state.release_progress();
    if (cleanup != CUDA_SUCCESS) {
        state.fault();
        return rejected(cleanup);
    }
    if (state.storage->default_proxy) {
        if (state.storage->has_default_proxy_destroy_attempt ||
            state.storage->is_default_proxy_uncertain)
            return rejected(CUDA_ERROR_UNKNOWN);
        const NativeExecutionScope scope(state.is_native_issue_active);
        auto status = state.check_context(true);
        if (status != CUDA_SUCCESS) return rejected(status);
        state.storage->has_default_proxy_destroy_attempt = true;
        status = state.invoke(state.dispatch.stream_destroy, state.storage->default_proxy);
        const auto proof = state.check_context(true);
        if (status != CUDA_SUCCESS || proof != CUDA_SUCCESS) {
            state.storage->is_default_proxy_uncertain = true;
            state.fault();
            return rejected(proof != CUDA_SUCCESS ? proof : status);
        }
        state.storage->default_proxy = nullptr;
    }
    if (state.storage->domain_member) {
        const auto released = state.release_shared_resources();
        if (released != CUDA_SUCCESS) return rejected(released);
        const auto detached = state.invoke(state.dispatch.context_set_current, nullptr);
        if (detached != CUDA_SUCCESS || state.is_faulted) {
            state.fault();
            return rejected(detached == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : detached);
        }
        const auto retired = state.storage->domain_member.retire();
        if (retired != CUDA_SUCCESS) {
            state.fault();
            return rejected(retired);
        }
        state.storage->context = nullptr;
        state.storage->domain_member = {};
        return settled();
    }
    state.has_destroy_attempt = true;
    auto result = completed(state.invoke(state.dispatch.context_destroy, state.storage->context));
    if (result.status != CUDA_SUCCESS) {
        state.fault();
        return result;
    }
    state.storage->context = nullptr;
    state.storage->allocations.clear();
    state.storage->modules.clear();
    state.storage->streams.clear();
    state.storage->events.clear();
    state.pending_events.clear();
    for (auto& generation : state.storage->event_generations) generation->native = nullptr;
    state.storage->event_generations.clear();
    state.storage->functions.clear();
    if (state.process_memory && state.storage->device_bytes &&
        state.process_memory->owner()->release_context_bytes(state.storage->device_bytes) !=
            CUDA_SUCCESS) {
        state.fault();
        return rejected(CUDA_ERROR_UNKNOWN);
    }
    state.storage->device_bytes = 0;
    state.storage->gemm_static_bytes = 0;
    state.storage->gemm_module = nullptr;
    state.storage->image_bytes = 0;
    state.storage->counters = 0;
    state.storage->counter_bytes = 0;
    // Successful owned-context destruction, after checked host cleanup, settles
    // prior GPU uncertainty. This closed, started session still cannot reopen.
    state.is_faulted = false;
    return result;
}

}  // namespace glimmer::backend::cuda
