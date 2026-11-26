// Included by the optional owned frontend. All native NCCL execution belongs to
// the service; no lookup, error, or uncertain reply falls through to a caller SDK.
namespace {
static_assert(NCCL_VERSION_CODE == 22903 && sizeof(ncclConfig_t) == 88);
static_assert(sizeof(ncclUniqueId) == kExecutionNcclUniqueIdBytes);
static_assert(sizeof(ncclResult_t) == sizeof(std::int32_t));
static_assert(sizeof(std::size_t) == sizeof(std::uint64_t));

ncclResult_t nccl_policy_error(CUresult error) noexcept {
    switch (error) {
        case CUDA_ERROR_INVALID_VALUE:
        case CUDA_ERROR_INVALID_HANDLE:
            return ncclInvalidArgument;
        case CUDA_ERROR_INVALID_CONTEXT:
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
        case CUDA_ERROR_NOT_INITIALIZED:
        case CUDA_ERROR_NOT_SUPPORTED:
            return ncclInvalidUsage;
        default:
            return ncclSystemError;
    }
}
template <typename Function>
ncclResult_t nccl_boundary(Function&& function) noexcept {
    if (::getpid() != kProcess) return ncclInvalidUsage;
    if (is_entered) return ncclInvalidUsage;
    Entered entered;
    try {
        auto& shared = state();
        Exclusion exclusion(shared.mutex);
        if (!shared.config) return ncclInvalidUsage;
        return function(shared, exclusion);
    } catch (...) {
        return ncclSystemError;
    }
}
template <typename Value>
std::int32_t nccl_enum_bits(const Value& value) noexcept {
    std::int32_t bits{};
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
bool nccl_native_status(std::int32_t status) noexcept {
    // The installed unfixed enum includes ncclNumResults=8 and has range0..15.
    return status >= 0 && status <= 15;
}
NcclCaller* prepare_nccl_caller(State& shared) {
    if (nccl_caller) return nccl_caller;
    if (!shared.next_nccl_caller || shared.nccl_callers.size() == kMaxHandles) return nullptr;
    auto caller = std::make_unique<NcclCaller>();
    caller->thread = {static_cast<std::uint64_t>(kProcess), shared.next_nccl_caller};
    if (shared.next_nccl_caller == UINT64_MAX)
        shared.next_nccl_caller = 0;
    else
        ++shared.next_nccl_caller;
    nccl_caller = caller.get();
    shared.nccl_callers.push_back(std::move(caller));
    return nccl_caller;
}
CUresult open_nccl_context(State& shared, Exclusion& exclusion) {
    if (!shared.nccl_context) {
        auto* context = shared.allocate_entry(Kind::kNcclContext, nullptr);
        if (!context) return CUDA_ERROR_OUT_OF_MEMORY;
        context->context = context;
        context->is_alive = true;
        shared.nccl_context = context;
    }
    return shared.open_context(exclusion, *shared.nccl_context, 0);
}
std::shared_ptr<NcclEntry> find_nccl(State& shared, ncclComm_t handle) {
    const auto token = reinterpret_cast<std::uintptr_t>(handle);
    const auto found = std::ranges::find_if(shared.nccl_entries, [=](const auto& entry) {
        return entry->token == token && entry->is_alive && !entry->is_uncertain &&
               !entry->is_retiring;
    });
    return found == shared.nccl_entries.end() ? nullptr : *found;
}
bool nccl_communicator(State& shared, ncclComm_t handle, ExecutionNcclCommunicator& output,
                       std::shared_ptr<NcclEntry>& owner) {
    if (!handle) return true;  // Preserve original NULL-argument validation.
    owner = find_nccl(shared, handle);
    if (!owner) return false;
    output = owner->remote;
    return output.domain && output.object;
}
struct NcclPins {
    std::array<Entry*, 10> entries{};
    std::size_t count{};
    Exclusion& exclusion;
    bool should_retain{};
    explicit NcclPins(Exclusion& lock) : exclusion(lock) {}
    bool add(Entry* entry) noexcept {
        if (!entry) return true;
        if (std::find(entries.begin(), entries.begin() + count, entry) != entries.begin() + count)
            return true;
        if (count == entries.size() || entry->active_calls == kMaxHandles ||
            entry->active_mutations == kMaxHandles || !entry->is_alive || entry->is_uncertain ||
            entry->is_retiring || entry->is_closing)
            return false;
        ++entry->active_calls;
        ++entry->active_mutations;
        entries[count++] = entry;
        return true;
    }
    void quarantine() noexcept {
        should_retain = true;
        for (std::size_t index = 0; index != count; ++index) entries[index]->is_uncertain = true;
    }
    ~NcclPins() {
        if (should_retain || ::getpid() != kProcess || !exclusion.owns_lock()) return;
        for (std::size_t index = 0; index != count; ++index) {
            --entries[index]->active_calls;
            --entries[index]->active_mutations;
        }
    }
    NcclPins(const NcclPins&) = delete;
    NcclPins& operator=(const NcclPins&) = delete;
};
bool nccl_buffer(State& shared, const void* pointer, ExecutionNcclBuffer& output, NcclPins& pins) {
    if (!pointer) return true;
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    Entry* entry{};
    for (const auto& candidate : shared.entries) {
        if ((candidate->kind != Kind::kAllocation && candidate->kind != Kind::kModuleGlobal) ||
            !candidate->is_alive || address < candidate->address ||
            address - candidate->address > candidate->bytes)
            continue;
        if (entry) return false;
        entry = candidate.get();
    }
    if (!entry || shared.range_is_retiring(*entry) || entry->is_uncertain ||
        !entry->context->connection)
        return false;
    if (!pins.add(entry) || !pins.add(entry->context)) return false;
    if (entry->kind == Kind::kModuleGlobal) {
        const auto parent = std::ranges::find_if(shared.entries, [&](const auto& candidate) {
            return candidate->kind == Kind::kModule && candidate->is_alive &&
                   candidate->context == entry->context && candidate->remote == entry->parent;
        });
        if (parent == shared.entries.end() || !pins.add(parent->get())) return false;
    }
    output = {ExecutionNcclBufferKind::kOwned, entry->context->connection->session(), entry->remote,
              address - entry->address};
    return output.source && output.allocation;
}
CUresult nccl_stream(State& shared, Exclusion& exclusion, cudaStream_t handle,
                     ExecutionNcclStream& output, NcclPins& pins) {
    auto* source = shared.context();
    Entry* stream{};
    const auto native = reinterpret_cast<CUstream>(handle);
    if (native && native != CU_STREAM_LEGACY && native != CU_STREAM_PER_THREAD) {
        stream = shared.find(native, Kind::kStream);
        if (!stream) return CUDA_ERROR_INVALID_HANDLE;
        source = stream->context;
    } else {
        const auto result = resolve_owned_stream(shared, exclusion, native, stream);
        if (result != CUDA_SUCCESS) return result;
    }
    if (!source || !source->connection || !pins.add(source)) return CUDA_ERROR_INVALID_CONTEXT;
    if (!pins.add(stream)) return CUDA_ERROR_INVALID_HANDLE;
    if (stream)
        output = {ExecutionNcclStreamKind::kOwned, source->connection->session(), stream->remote};
    else
        output.kind = native == CU_STREAM_LEGACY ? ExecutionNcclStreamKind::kLegacy
                                                 : ExecutionNcclStreamKind::kDefault;
    return CUDA_SUCCESS;
}
bool capture_nccl_config(ncclConfig_t* pointer, std::optional<ExecutionNcclConfig>& output) {
    if (!pointer) return true;
    std::uint64_t declared{};
    if (!self_copy(&declared, pointer, sizeof(declared), false)) return false;
    if ((declared > 32 && declared < 40) || (declared > 48 && declared < 56)) return false;
    std::array<std::byte, 88> bytes{};
    if (!self_copy(bytes.data(), pointer, std::min<std::uint64_t>(declared, bytes.size()), false))
        return false;
    ExecutionNcclConfig config;
    config.declared_bytes = declared;
    std::memcpy(&config.magic, bytes.data() + 8, 4);
    std::memcpy(&config.version, bytes.data() + 12, 4);
    constexpr std::array<std::size_t, 14> offsets{16, 20, 24, 28, 40, 44, 56,
                                                  60, 64, 68, 72, 76, 80, 84};
    for (std::size_t index = 0; index != offsets.size(); ++index)
        std::memcpy(&config.fields[index], bytes.data() + offsets[index], 4);
    for (const auto offset : {std::size_t{32}, std::size_t{48}}) {
        if (declared < offset + sizeof(const char*)) continue;
        const char* text{};
        std::memcpy(&text, bytes.data() + offset, sizeof(text));
        if (!text) continue;
        auto owned = owned_string(text, kExecutionNcclConfigStringLimit + 1);
        if (!owned) return false;
        (offset == 32 ? config.net_name : config.comm_name) = std::move(owned);
    }
    output = std::move(config);
    return true;
}
struct NcclWrite {
    std::shared_ptr<NcclOutput> binding;
    std::array<std::byte, kExecutionNcclUniqueIdBytes> bytes{};
    std::array<bool, kExecutionNcclUniqueIdBytes> mask{};
    std::size_t count{};
};
bool prepare_nccl_write(const std::shared_ptr<NcclOutput>& output,
                        const ExecutionNcclObservation& observation, NcclWrite& write) {
    if (!output || !output->destination) return observation.effects == 0;
    if (output->phase == NcclOutputPhase::kUncertain) return false;
    write.binding = output;
    if (output->kind == NcclOutputKind::kUniqueId) {
        if (observation.effects & ~1U) return false;
        write.count = kExecutionNcclUniqueIdBytes;
        write.bytes = observation.unique_id;
        for (std::size_t index = 0; index != write.count; ++index)
            write.mask[index] =
                (std::to_integer<unsigned int>(observation.unique_id_mask[index / 8]) &
                 (1U << (index % 8))) != 0;
    } else if (output->kind == NcclOutputKind::kCommunicator) {
        if (observation.effects & ~2U) return false;
        if (!(observation.effects & 2U)) return true;
        if (!output->communicator || (observation.communicator.domain &&
                                      observation.communicator != output->communicator->remote))
            return false;
        const auto token = observation.communicator.domain ? output->communicator->token : 0;
        // NOLINTNEXTLINE(performance-no-int-to-ptr): opaque identity, never dereferenced.
        const auto handle = reinterpret_cast<ncclComm_t>(static_cast<std::uintptr_t>(token));
        static_assert(sizeof(ncclComm_t) == sizeof(token));
        write.count = sizeof(token);
        std::memcpy(write.bytes.data(), &handle, sizeof(token));
        std::fill_n(write.mask.begin(), write.count, true);
    } else {
        if (observation.effects & ~4U) return false;
        write.count = sizeof(std::int32_t);
        std::memcpy(write.bytes.data(), &observation.async_error, write.count);
        for (std::size_t index = 0; index != write.count; ++index)
            write.mask[index] = (observation.async_error_mask & (1U << index)) != 0;
    }
    return !write.count || writable_host_range(output->destination, write.count);
}
bool apply_nccl_write(const NcclWrite& write) noexcept {
    if (!write.binding) return true;
    const auto base = reinterpret_cast<std::uintptr_t>(write.binding->destination);
    for (std::size_t index = 0; index != write.count; ++index)
        if (write.mask[index]) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr): preflighted checked self-process write.
            if (!self_copy(reinterpret_cast<void*>(base + index), write.bytes.data() + index, 1,
                           true))
                return false;
        }
    return true;
}
struct NcclOutputReservation {
    State& shared;
    Exclusion& exclusion;
    std::shared_ptr<NcclOutput> output;
    NcclOutputReservation(State& owner, Exclusion& lock,
                          std::shared_ptr<NcclOutput> binding) noexcept
        : shared(owner), exclusion(lock), output(std::move(binding)) {}
    explicit operator bool() const noexcept {
        return static_cast<bool>(output);
    }
    ~NcclOutputReservation() {
        if (!output || ::getpid() != kProcess || !exclusion.owns_lock()) return;
        if (output->phase != NcclOutputPhase::kPrepared &&
            output->phase != NcclOutputPhase::kComplete)
            return;
        // Only this wrapper owns this reservation's terminal/no-entry cleanup.
        // Another caller's Prepared/InFlight zero-remote entry is never swept.
        std::erase(shared.nccl_outputs, output);
        if (output->communicator && !output->communicator->remote.domain &&
            !output->communicator->is_uncertain)
            std::erase(shared.nccl_entries, output->communicator);
    }
    NcclOutputReservation(const NcclOutputReservation&) = delete;
    NcclOutputReservation& operator=(const NcclOutputReservation&) = delete;
};
NcclOutputReservation reserve_nccl_output(State& shared, Exclusion& exclusion, void* destination,
                                          NcclOutputKind kind) {
    if (!destination) return {shared, exclusion, {}};
    const auto bytes = kind == NcclOutputKind::kUniqueId       ? sizeof(ncclUniqueId)
                       : kind == NcclOutputKind::kCommunicator ? sizeof(ncclComm_t)
                                                               : sizeof(ncclResult_t);
    if (!writable_host_range(destination, bytes) || shared.nccl_outputs.size() == kMaxHandles)
        return {shared, exclusion, {}};
    auto output = std::make_shared<NcclOutput>();
    output->kind = kind;
    output->destination = destination;
    if (kind == NcclOutputKind::kCommunicator) {
        if (!shared.next_nccl_token || shared.nccl_entries.size() == kMaxHandles)
            return {shared, exclusion, {}};
        auto entry = std::make_shared<NcclEntry>();
        entry->token = shared.next_nccl_token;
        shared.next_nccl_token =
            shared.next_nccl_token == UINTPTR_MAX ? 0 : shared.next_nccl_token + 1;
        shared.nccl_entries.push_back(entry);
        output->communicator = std::move(entry);
    }
    shared.nccl_outputs.push_back(output);
    return {shared, exclusion, std::move(output)};
}
ncclResult_t execute_nccl(State& shared, Exclusion& exclusion, ExecutionNcclCall call,
                          NcclPins& pins, const std::shared_ptr<NcclOutput>& output = {},
                          const std::shared_ptr<NcclEntry>& communicator = {}) {
    auto* caller = prepare_nccl_caller(shared);
    if (!caller || !caller->sequence) return ncclSystemError;
    if (caller->is_uncertain || (shared.nccl_context && shared.nccl_context->is_uncertain))
        return ncclSystemError;
    const auto opened = open_nccl_context(shared, exclusion);
    if (opened != CUDA_SUCCESS) return nccl_policy_error(opened);
    auto& transport = *shared.nccl_context;
    if (!pins.add(shared.context())) return ncclInvalidUsage;
    const bool is_start = std::holds_alternative<ExecutionNcclGroupStart>(call);
    const bool is_end = std::holds_alternative<ExecutionNcclGroupEnd>(call);
    if (is_start && caller->depth == UINT64_MAX) return ncclSystemError;
    if (is_start && !caller->depth) {
        if (!shared.next_nccl_group) return ncclSystemError;
        caller->group = shared.next_nccl_group;
        shared.next_nccl_group =
            shared.next_nccl_group == UINT64_MAX ? 0 : shared.next_nccl_group + 1;
    }
    const auto* source = shared.context();
    ExecutionNcclRequest request{{caller->thread, caller->sequence, caller->group,
                                  source && source->connection ? source->connection->session() : 0},
                                 std::move(call)};
    if (output) output->scope = request.scope;
    if (output && output->kind == NcclOutputKind::kCommunicator && caller->depth &&
        std::ranges::count_if(shared.nccl_outputs, [&](const auto& binding) {
            return binding->phase == NcclOutputPhase::kDeferred &&
                   binding->scope.thread == caller->thread && binding->scope.group == caller->group;
        }) >= static_cast<std::ptrdiff_t>(kExecutionNcclPublicationLimit))
        return ncclInvalidUsage;
    // All local context-stack/handle/publication capacity exists before entry.
    context_stack.reserve(std::max<std::size_t>(1, context_stack.size()));
    auto* primary = shared.primary_entry();
    if (!primary || !pins.add(primary)) return ncclSystemError;
    ExecutionPayloadWriter input;
    const auto caller_primary_domain =
        source && source->is_primary ? source->nccl_primary_domain : 0;
    if (!input.write_u64(caller_primary_domain) || !write_execution_nccl_request(input, request))
        return ncclInvalidArgument;
    const auto bytes = input.take();
    if (output) output->phase = NcclOutputPhase::kInFlight;
    bool was_submitted{};
    const auto reply =
        shared.rpc(exclusion, transport, ExecutionOpcode::kNcclCall, bytes, -1, &was_submitted);
    // The connection sequence may be independent; this caller sequence advances
    // once for every canonical sent RPC, including clean remote policy rejection.
    if (was_submitted) caller->sequence = caller->sequence == UINT64_MAX ? 0 : caller->sequence + 1;
    const auto fail = [&](std::optional<std::int32_t> original = {}) {
        caller->is_uncertain = transport.is_uncertain = true;
        if (output) output->phase = NcclOutputPhase::kUncertain;
        if (communicator) communicator->is_uncertain = true;
        pins.quarantine();
        return original && nccl_native_status(*original) ? static_cast<ncclResult_t>(*original)
                                                         : ncclSystemError;
    };
    if (!was_submitted && reply && reply->status > 0 && !transport.is_uncertain) {
        if (output) output->phase = NcclOutputPhase::kComplete;
        if (is_start && !caller->depth) caller->group = 0;
        return nccl_policy_error(static_cast<CUresult>(reply->status));
    }
    if (!reply) return fail();
    const auto original = reply->original_status;
    ExecutionNcclReply observed;
    ExecutionPayloadReader reader(reply->data);
    if (reply->address || !reply->arguments.empty() || reply->parameter_bytes > 1 ||
        !read_execution_nccl_reply(reader, observed) || !reader.empty() ||
        observed.original_status != original || (original && !nccl_native_status(*original)))
        return fail(original);
    if (!observed.is_accepted) {
        if (reply->disposition != ExecutionWireDisposition::kRejected || reply->status <= 0 ||
            reply->id || original || reply->parameter_bytes ||
            static_cast<std::uint32_t>(reply->status) >
                std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U)
            return fail(original);
        if (output) output->phase = NcclOutputPhase::kComplete;
        if (is_start && !caller->depth) caller->group = 0;
        return nccl_policy_error(static_cast<CUresult>(reply->status));
    }
    if (reply->disposition != ExecutionWireDisposition::kComplete ||
        reply->id != observed.operation || !original)
        return fail(original);
    if (reply->status < 0 ||
        static_cast<std::uint32_t>(reply->status) >
            std::bit_ceil(static_cast<std::uint32_t>(CUDA_ERROR_UNKNOWN) + 1U) - 1U)
        return fail(original);
    const bool has_ownership_fault = reply->status != 0 || reply->parameter_bytes != 0;
    if (output && output->communicator) {
        if (observed.reserved_communicator.domain) {
            const auto duplicate = std::ranges::any_of(shared.nccl_entries, [&](const auto& entry) {
                return entry != output->communicator &&
                       entry->remote == observed.reserved_communicator;
            });
            if (duplicate) return fail(original);
            output->communicator->remote = observed.reserved_communicator;
        }
    } else if (observed.reserved_communicator.domain) {
        return fail(original);
    }
    const auto effect = observed.context_effect;
    if (effect.kind == ExecutionNcclContextEffectKind::kSelectPrimary) {
        if (effect.device != 0 ||
            (primary->nccl_primary_domain && primary->nccl_primary_domain != effect.domain) ||
            primary->is_uncertain)
            return fail(original);
    } else if (effect.kind != ExecutionNcclContextEffectKind::kNone) {
        // Current-NULL is representable on the wire, but an observed current
        // context alone does not prove how a nested native context stack changed.
        // Retain/quarantine until that stack projection has an audited adapter.
        return fail(original);
    }
    std::array<NcclWrite, kExecutionNcclPublicationLimit + 1> writes{};
    if (!prepare_nccl_write(output, observed.observation, writes[0])) return fail(original);
    if (*original == ncclSuccess && output) {
        if (output->kind == NcclOutputKind::kUniqueId &&
            (observed.observation.effects != 1 ||
             !std::ranges::all_of(observed.observation.unique_id_mask,
                                  [](auto value) { return value == std::byte{255}; })))
            return fail(original);
        if (output->kind == NcclOutputKind::kAsyncError &&
            (observed.observation.effects != 4 || observed.observation.async_error_mask != 15))
            return fail(original);
        if (output->kind == NcclOutputKind::kCommunicator &&
            ((!caller->depth && observed.observation.effects != 2) ||
             (observed.observation.effects == 2 && !observed.observation.communicator.domain) ||
             !observed.reserved_communicator.domain))
            return fail(original);
    }
    if (is_end && caller->depth == 1 && *original == ncclSuccess) {
        const auto pending = std::ranges::count_if(shared.nccl_outputs, [&](const auto& binding) {
            return binding->phase == NcclOutputPhase::kDeferred &&
                   binding->scope.thread == caller->thread && binding->scope.group == caller->group;
        });
        if (pending != static_cast<std::ptrdiff_t>(observed.publications.size()))
            return fail(original);
    }
    for (std::size_t index = 0; index != observed.publications.size(); ++index) {
        const auto& publication = observed.publications[index];
        if (publication.scope == request.scope) return fail(original);
        const auto previous = std::span{observed.publications}.first(index);
        if (std::ranges::any_of(
                previous, [&](const auto& earlier) { return earlier.scope == publication.scope; }))
            return fail(original);
        const auto found = std::ranges::find_if(shared.nccl_outputs, [&](const auto& binding) {
            return binding->phase == NcclOutputPhase::kDeferred &&
                   binding->scope == publication.scope;
        });
        if (found == shared.nccl_outputs.end()) return fail(original);
        if (is_end) {
            if (caller->depth != 1 || publication.scope.thread != caller->thread ||
                publication.scope.group != caller->group)
                return fail(original);
        } else if (!std::holds_alternative<ExecutionNcclGetAsyncError>(request.call) ||
                   !communicator || (*found)->communicator != communicator) {
            return fail(original);
        }
        if (!prepare_nccl_write(*found, publication.observation, writes[index + 1]))
            return fail(original);
        if (*original == ncclSuccess &&
            (publication.observation.effects != 2 || !publication.observation.communicator.domain))
            return fail(original);
    }
    // Every scope, destination, token and effect is validated before any write.
    for (std::size_t index = 0; index <= observed.publications.size(); ++index)
        if (!apply_nccl_write(writes[index])) return fail(original);
    if (effect.kind == ExecutionNcclContextEffectKind::kSelectPrimary) {
        primary->nccl_primary_domain = effect.domain;
        if (context_stack.empty())
            context_stack.push_back(primary);
        else
            context_stack.back() = primary;
    }
    // The authenticated producer proves these output/context effects separately
    // from native async/resource ownership. A later carrier/proof fault cannot
    // erase an actual write (including NULL or an equal-value write), but never
    // authorizes retirement, replay, or use of an uncertain native owner.
    if (has_ownership_fault) return fail(original);
    if (output && output->communicator && observed.observation.communicator.domain)
        output->communicator->is_alive = true;
    for (std::size_t index = 0; index != observed.publications.size(); ++index) {
        const auto& binding = writes[index + 1].binding;
        if (binding && binding->communicator &&
            observed.publications[index].observation.communicator.domain)
            binding->communicator->is_alive = true;
        if (binding) {
            binding->phase = NcclOutputPhase::kComplete;
            std::erase(shared.nccl_outputs, binding);
        }
    }
    if (output) {
        output->phase = output->kind == NcclOutputKind::kCommunicator &&
                                !(observed.observation.effects & 2U) &&
                                (caller->depth || *original == ncclInProgress)
                            ? NcclOutputPhase::kDeferred
                            : NcclOutputPhase::kComplete;
        if (output->phase == NcclOutputPhase::kComplete) std::erase(shared.nccl_outputs, output);
    }
    if (is_start && *original == ncclSuccess)
        ++caller->depth;
    else if (is_start)
        return fail(original);
    if (is_end && (*original == ncclSuccess || *original == ncclInProgress)) {
        if (caller->depth) --caller->depth;
        if (!caller->depth) caller->group = 0;
    } else if (is_end && caller->depth) {
        return fail(original);
    }
    if (caller->depth && *original != ncclSuccess && *original != ncclInProgress)
        return fail(original);
    if (communicator && (std::holds_alternative<ExecutionNcclDestroy>(request.call) ||
                         std::holds_alternative<ExecutionNcclAbort>(request.call))) {
        // A grouped destructive inner remains sealed even after inner SUCCESS.
        communicator->is_retiring = true;
        if (*original == ncclSuccess && !request.scope.group)
            communicator->is_alive = false;
        else if (*original != ncclSuccess)
            communicator->is_uncertain = true;
    }
    return static_cast<ncclResult_t>(*original);
}
ncclResult_t initialize_nccl(ncclComm_t* destination, int ranks, ncclUniqueId id, int rank,
                             ncclConfig_t* config, bool has_config) noexcept {
    return nccl_boundary([&](State& shared, Exclusion& exclusion) {
        auto output =
            reserve_nccl_output(shared, exclusion, destination, NcclOutputKind::kCommunicator);
        if (destination && !output) return ncclInvalidArgument;
        ExecutionNcclInitRank init;
        init.has_output = destination != nullptr;
        init.ranks = ranks;
        init.rank = rank;
        std::memcpy(init.unique_id.data(), &id, sizeof(id));
        ExecutionNcclCall call = init;
        if (has_config) {
            ExecutionNcclInitRankConfig configured;
            configured.initialization = init;
            if (!capture_nccl_config(config, configured.config)) return ncclInvalidArgument;
            call = std::move(configured);
        }
        NcclPins pins(exclusion);
        return execute_nccl(shared, exclusion, std::move(call), pins, output.output);
    });
}
ncclResult_t collective_nccl(const void* send, void* receive, std::size_t count,
                             ncclDataType_t datatype, ncclComm_t handle, cudaStream_t stream,
                             std::optional<std::int32_t> reduction,
                             std::optional<int> root) noexcept {
    return nccl_boundary([&](State& shared, Exclusion& exclusion) {
        NcclPins pins(exclusion);
        ExecutionNcclAllGather collective;
        std::shared_ptr<NcclEntry> comm;
        if (!nccl_communicator(shared, handle, collective.communicator, comm))
            return ncclInvalidArgument;
        if (comm && comm->active_calls) return ncclInvalidUsage;
        const auto resolved = nccl_stream(shared, exclusion, stream, collective.stream, pins);
        if (resolved != CUDA_SUCCESS) return nccl_policy_error(resolved);
        if (!nccl_buffer(shared, send, collective.send, pins) ||
            !nccl_buffer(shared, receive, collective.receive, pins))
            return ncclInvalidArgument;
        collective.count = count;
        collective.datatype = nccl_enum_bits(datatype);
        std::optional<CounterPin> comm_pin;
        if (comm) comm_pin.emplace(comm->active_calls, exclusion);
        if (root) {
            ExecutionNcclBcast bcast{collective.communicator, collective.stream,
                                     collective.receive,      count,
                                     collective.datatype,     *root};
            return execute_nccl(shared, exclusion, bcast, pins, {}, comm);
        }
        if (reduction)
            return execute_nccl(shared, exclusion, ExecutionNcclAllReduce{collective, *reduction},
                                pins, {}, comm);
        return execute_nccl(shared, exclusion, collective, pins, {}, comm);
    });
}
}  // namespace

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed SDK ABIs.
extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId* destination) {
    return nccl_boundary([&](State& shared, Exclusion& exclusion) {
        auto output =
            reserve_nccl_output(shared, exclusion, destination, NcclOutputKind::kUniqueId);
        if (destination && !output) return ncclInvalidArgument;
        NcclPins pins(exclusion);
        return execute_nccl(shared, exclusion, ExecutionNcclGetUniqueId{destination != nullptr},
                            pins, output.output);
    });
}
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t* destination, int ranks, ncclUniqueId id,
                                         int rank) {
    return initialize_nccl(destination, ranks, id, rank, nullptr, false);
}
extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t* destination, int ranks, ncclUniqueId id,
                                               int rank, ncclConfig_t* config) {
    return initialize_nccl(destination, ranks, id, rank, config, true);
}
extern "C" ncclResult_t ncclCommGetAsyncError(ncclComm_t handle, ncclResult_t* destination) {
    return nccl_boundary([&](State& shared, Exclusion& exclusion) {
        ExecutionNcclGetAsyncError call;
        std::shared_ptr<NcclEntry> comm;
        if (!nccl_communicator(shared, handle, call.communicator, comm) ||
            (comm && comm->active_calls))
            return ncclInvalidArgument;
        auto output =
            reserve_nccl_output(shared, exclusion, destination, NcclOutputKind::kAsyncError);
        if (destination && !output) return ncclInvalidArgument;
        call.has_output = destination != nullptr;
        NcclPins pins(exclusion);
        std::optional<CounterPin> comm_pin;
        if (comm) comm_pin.emplace(comm->active_calls, exclusion);
        return execute_nccl(shared, exclusion, call, pins, output.output, comm);
    });
}
namespace {
ncclResult_t release_nccl(ncclComm_t handle, bool is_abort) noexcept {
    return nccl_boundary([&](State& shared, Exclusion& exclusion) {
        ExecutionNcclCommunicator identity;
        std::shared_ptr<NcclEntry> comm;
        if (!nccl_communicator(shared, handle, identity, comm) || (comm && comm->active_calls))
            return ncclInvalidArgument;
        NcclPins pins(exclusion);
        std::optional<CounterPin> comm_pin;
        if (comm) comm_pin.emplace(comm->active_calls, exclusion);
        ExecutionNcclCall call = is_abort ? ExecutionNcclCall{ExecutionNcclAbort{identity}}
                                          : ExecutionNcclCall{ExecutionNcclDestroy{identity}};
        return execute_nccl(shared, exclusion, std::move(call), pins, {}, comm);
    });
}
}  // namespace
extern "C" ncclResult_t ncclCommDestroy(ncclComm_t handle) {
    return release_nccl(handle, false);
}
extern "C" ncclResult_t ncclCommAbort(ncclComm_t handle) {
    return release_nccl(handle, true);
}
extern "C" ncclResult_t ncclGroupStart() {
    return nccl_boundary([](State& shared, Exclusion& exclusion) {
        NcclPins pins(exclusion);
        return execute_nccl(shared, exclusion, ExecutionNcclGroupStart{}, pins);
    });
}
extern "C" ncclResult_t ncclGroupEnd() {
    return nccl_boundary([](State& shared, Exclusion& exclusion) {
        NcclPins pins(exclusion);
        return execute_nccl(shared, exclusion, ExecutionNcclGroupEnd{}, pins);
    });
}
extern "C" ncclResult_t ncclAllGather(const void* send, void* receive, std::size_t count,
                                      ncclDataType_t datatype, ncclComm_t comm,
                                      cudaStream_t stream) {
    return collective_nccl(send, receive, count, datatype, comm, stream, {}, {});
}
extern "C" ncclResult_t ncclAllReduce(const void* send, void* receive, std::size_t count,
                                      ncclDataType_t datatype, ncclRedOp_t reduction,
                                      ncclComm_t comm, cudaStream_t stream) {
    return collective_nccl(send, receive, count, datatype, comm, stream, nccl_enum_bits(reduction),
                           {});
}
extern "C" ncclResult_t ncclBcast(void* buffer, std::size_t count, ncclDataType_t datatype,
                                  int root, ncclComm_t comm, cudaStream_t stream) {
    return collective_nccl(buffer, buffer, count, datatype, comm, stream, {}, root);
}
// NOLINTEND(bugprone-easily-swappable-parameters)
