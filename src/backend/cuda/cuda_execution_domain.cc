#include "internal/cuda_execution_domain.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> domain_thread_identity;

constexpr std::size_t kFirstLimitField = 3;
constexpr std::size_t kLimitFields = 16;
static_assert(kFirstLimitField + kLimitFields == kCudaExecutionDomainConfigurationFields);
struct DomainConfiguration {
    std::uint32_t available{1};
    std::array<std::uint64_t, kFirstLimitField + kLimitFields> values{};
    bool operator==(const DomainConfiguration&) const = default;
};
bool valid_flags(unsigned int flags) noexcept;

struct DomainState {
    const pid_t process{::getpid()};
    const std::weak_ptr<const void> owner{domain_thread_identity};
    const CudaExecutionDomainDispatch dispatch;
    const CudaExecutionDomainConfig config;
    const std::shared_ptr<CudaExecutionOwnerActivity> activity =
        std::make_shared<CudaExecutionOwnerActivity>();
    std::vector<std::shared_ptr<CudaExecutionDomainMemberState>> members;
    CUcontext context{};
    CUdevice native_device{};
    CUresult failure{CUDA_SUCCESS};
    std::size_t reserved_device_bytes{};
    std::size_t process_device_bytes{};
    std::uint64_t next_generation{1};
    DomainConfiguration baseline_configuration, physical_configuration;
    std::uint64_t physical_configuration_epoch{1};
    bool can_virtualize_configuration{};
    bool has_unobserved_configuration_effects{};
    bool has_initialize_attempt{}, has_native_owner{}, has_destroy_attempt{};
    bool is_initialized{}, is_closed{}, is_shutdown{};

    DomainState(CudaExecutionDomainDispatch callbacks, CudaExecutionDomainConfig settings)
        : dispatch(callbacks), config(settings) {
        baseline_configuration.values[0] = physical_configuration.values[0] = config.flags;
        can_virtualize_configuration = dispatch.configuration.get_flags != nullptr;
    }

    bool owns() const noexcept {
        // Test PID before touching inherited thread-local identity.
        return ::getpid() == process && domain_thread_identity &&
               owner.lock() == domain_thread_identity;
    }
    bool has_activity() const noexcept {
        return activity->is_native_issue_active || activity->is_folded_active ||
               activity->is_rnn_query_active || activity->is_preparation_active;
    }
    CUresult health() const noexcept {
        if (failure != CUDA_SUCCESS) return failure;
        return activity->is_quarantined ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
    }
    CUresult gate(bool owns_native_scope = false) const noexcept {
        if (!owns()) return CUDA_ERROR_INVALID_CONTEXT;
        if (has_activity() && !owns_native_scope) return CUDA_ERROR_NOT_READY;
        return health();
    }
    CUresult fail(CUresult status) noexcept {
        activity->is_quarantined = true;
        if (failure == CUDA_SUCCESS) failure = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        return failure;
    }
    struct Scope {
        explicit Scope(DomainState& state) : state_(state) {
            state_.activity->is_native_issue_active = true;
        }
        ~Scope() {
            state_.activity->is_native_issue_active = false;
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        DomainState& state_;
    };
    CUresult current(CUcontext& observed) {
        const auto status = dispatch.context_get_current(&observed);
        if (status != CUDA_SUCCESS) return fail(status);
        return health();
    }
    CUresult prove(CUcontext expected) {
        CUcontext observed{};
        const auto status = current(observed);
        if (status != CUDA_SUCCESS) return status;
        return observed == expected ? CUDA_SUCCESS : fail(CUDA_ERROR_INVALID_CONTEXT);
    }
    bool has_physical_work() const noexcept;
    CUresult capture_configuration(DomainConfiguration& result) {
        const auto& sdk = dispatch.configuration;
        const auto query = [&](auto callback, auto... arguments) {
            const auto status = callback(arguments...);
            const auto proof = prove(context);
            return proof == CUDA_SUCCESS ? status : proof;
        };
        result = {};
        unsigned int flags{};
        auto status = query(sdk.get_flags, &flags);
        if (status != CUDA_SUCCESS) return status;
        if (!valid_flags(flags)) return CUDA_ERROR_UNKNOWN;
        result.values[0] = flags;
        CUfunc_cache cache{};
        status = query(sdk.get_cache, &cache);
        if (status != CUDA_SUCCESS) return status;
        // Read the external ABI representation before evaluating an SDK enum.
        // GCC's bit_cast instrumentation still loads an invalid enum value.
        std::uint32_t cache_bits{};
        static_assert(sizeof(cache_bits) == sizeof(cache));
        std::memcpy(&cache_bits, &cache, sizeof(cache_bits));
        if (cache_bits > 3) return CUDA_ERROR_UNKNOWN;
        result.values[1] = cache_bits;
        CUsharedconfig shared{};
        status = query(sdk.get_shared, &shared);
        if (status != CUDA_SUCCESS) return status;
        std::uint32_t shared_bits{};
        static_assert(sizeof(shared_bits) == sizeof(shared));
        std::memcpy(&shared_bits, &shared, sizeof(shared_bits));
        if (shared_bits > 2) return CUDA_ERROR_UNKNOWN;
        result.values[2] = shared_bits;
        result.available = 7;
        for (std::size_t selector = 0; selector < kLimitFields; ++selector) {
            std::size_t bytes{};
            status = query(sdk.get_limit, &bytes, static_cast<CUlimit>(selector));
            if (health() != CUDA_SUCCESS) return health();
            if (status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_UNSUPPORTED_LIMIT ||
                status == CUDA_ERROR_NOT_SUPPORTED)
                continue;
            if (status != CUDA_SUCCESS) return status;
            const auto field = kFirstLimitField + selector;
            result.available |= std::uint32_t{1} << field;
            result.values[field] = bytes;
        }
        return CUDA_SUCCESS;
    }
    CUresult verify_configuration() {
        if (!can_virtualize_configuration || !has_unobserved_configuration_effects ||
            has_physical_work())
            return CUDA_SUCCESS;
        auto status = prove(context);
        if (status != CUDA_SUCCESS) return status;
        DomainConfiguration actual;
        status = capture_configuration(actual);
        if (status != CUDA_SUCCESS) return fail(status);
        if (actual != physical_configuration) return fail(CUDA_ERROR_UNKNOWN);
        has_unobserved_configuration_effects = false;
        return CUDA_SUCCESS;
    }
    CUresult materialize(const DomainConfiguration& target) {
        const auto verified = verify_configuration();
        if (verified != CUDA_SUCCESS) return verified;
        if (target == physical_configuration) return CUDA_SUCCESS;
        if (!can_virtualize_configuration) return CUDA_ERROR_NOT_SUPPORTED;
        if (has_physical_work()) return CUDA_ERROR_NOT_READY;
        if (physical_configuration_epoch == std::numeric_limits<std::uint64_t>::max())
            return CUDA_ERROR_OUT_OF_MEMORY;
        if (target.available != physical_configuration.available) return fail(CUDA_ERROR_UNKNOWN);
        auto status = prove(context);
        if (status != CUDA_SUCCESS) return status;
        const auto& sdk = dispatch.configuration;
        for (std::size_t field = 0; field < target.values.size(); ++field) {
            if (!(target.available & (std::uint32_t{1} << field)) ||
                target.values[field] == physical_configuration.values[field])
                continue;
            if (field == 0)
                status = sdk.set_flags(static_cast<unsigned int>(target.values[field]));
            else if (field == 1)
                status = sdk.set_cache(static_cast<CUfunc_cache>(target.values[field]));
            else if (field == 2)
                status = sdk.set_shared(static_cast<CUsharedconfig>(target.values[field]));
            else
                status = sdk.set_limit(static_cast<CUlimit>(field - kFirstLimitField),
                                       static_cast<std::size_t>(target.values[field]));
            const auto proof = prove(context);
            if (status != CUDA_SUCCESS) return fail(status);
            if (proof != CUDA_SUCCESS) return proof;
        }
        DomainConfiguration actual;
        status = capture_configuration(actual);
        if (status != CUDA_SUCCESS) return fail(status);
        if (actual != target) return fail(CUDA_ERROR_UNKNOWN);
        physical_configuration = actual;
        ++physical_configuration_epoch;
        return CUDA_SUCCESS;
    }
};

bool valid_identity(CudaExecutionDomainLogicalIdentity logical) noexcept {
    return logical.context && logical.session.client_id && logical.session.server_incarnation &&
           logical.session.first_sequence;
}
bool same_identity(CudaExecutionDomainLogicalIdentity left,
                   CudaExecutionDomainLogicalIdentity right) noexcept {
    return left.context == right.context && left.session.client_id == right.session.client_id &&
           left.session.server_incarnation == right.session.server_incarnation &&
           left.session.first_sequence == right.session.first_sequence;
}
bool valid_flags(unsigned int flags) noexcept {
    const auto scheduling = flags & CU_CTX_SCHED_MASK;
    return !(flags & ~static_cast<unsigned int>(CU_CTX_FLAGS_MASK)) &&
           !(flags & (CU_CTX_COREDUMP_ENABLE | CU_CTX_USER_COREDUMP_ENABLE)) &&
           (scheduling == CU_CTX_SCHED_AUTO || scheduling == CU_CTX_SCHED_SPIN ||
            scheduling == CU_CTX_SCHED_YIELD || scheduling == CU_CTX_SCHED_BLOCKING_SYNC);
}
bool clean_creation_error(CUresult status) noexcept {
    return status == CUDA_ERROR_OUT_OF_MEMORY || status == CUDA_ERROR_INVALID_VALUE ||
           status == CUDA_ERROR_INVALID_DEVICE || status == CUDA_ERROR_NOT_SUPPORTED ||
           status == CUDA_ERROR_NOT_PERMITTED;
}
std::size_t saturated_add(std::size_t left, std::size_t right) noexcept {
    return right > std::numeric_limits<std::size_t>::max() - left
               ? std::numeric_limits<std::size_t>::max()
               : left + right;
}
}  // namespace

struct CudaExecutionDomainMemberState {
    // The owner retains its implementation when native state or exported
    // members remain unsettled. A successful shutdown requires every external
    // member/view to be gone, so this backpointer never outlives that owner.
    DomainState* domain{};
    CudaExecutionDomainLogicalIdentity logical{};
    std::uint64_t generation{};
    std::uint64_t configuration_epoch{1};
    DomainConfiguration configuration;
    unsigned int initial_flags{};
    std::size_t reserved_device_bytes{};
    bool is_attached{}, is_retired{}, has_physical_work{};

    CUresult gate(bool owns_native_scope = false) const noexcept {
        if (!domain->owns()) return CUDA_ERROR_INVALID_CONTEXT;
        if (is_retired || domain->is_shutdown) return CUDA_ERROR_DEINITIALIZED;
        const auto status = domain->gate(owns_native_scope);
        if (status != CUDA_SUCCESS) return status;
        return domain->is_initialized ? CUDA_SUCCESS : CUDA_ERROR_NOT_INITIALIZED;
    }
};

bool DomainState::has_physical_work() const noexcept {
    return std::ranges::any_of(members,
                               [](const auto& member) { return member->has_physical_work; });
}

struct CudaExecutionDomainOwner::Implementation : DomainState {
    Implementation(CudaExecutionDomainDispatch callbacks, CudaExecutionDomainConfig settings)
        : DomainState(callbacks, settings) {
        members.reserve(config.maximum_members);
    }
    void prune() noexcept {
        std::erase_if(members, [](const auto& member) {
            return member.use_count() == 1 && !member->reserved_device_bytes &&
                   !member->has_physical_work && (member->is_retired || !member->is_attached);
        });
    }
    bool has_member_owners() const noexcept {
        return std::ranges::any_of(members, [](const auto& member) {
            return member.use_count() != 1 || member->reserved_device_bytes ||
                   member->has_physical_work || (member->is_attached && !member->is_retired);
        });
    }
};

CudaExecutionDomainOwner::CudaExecutionDomainOwner(CudaExecutionDomainDispatch dispatch,
                                                   CudaExecutionDomainConfig config) {
    if (!config.identity || config.device_ordinal < 0 || !config.maximum_members ||
        !config.maximum_device_bytes || !valid_flags(config.flags) || !dispatch.initialize ||
        !dispatch.device_get || !dispatch.context_create || !dispatch.context_set_current ||
        !dispatch.context_get_current || !dispatch.context_destroy)
        throw std::invalid_argument("invalid CUDA execution domain configuration");
    const auto& controls = dispatch.configuration;
    const std::array has_controls{controls.get_flags != nullptr,  controls.set_flags != nullptr,
                                  controls.get_limit != nullptr,  controls.set_limit != nullptr,
                                  controls.get_cache != nullptr,  controls.set_cache != nullptr,
                                  controls.get_shared != nullptr, controls.set_shared != nullptr};
    if (std::ranges::any_of(has_controls, [](bool present) { return present; }) &&
        !std::ranges::all_of(has_controls, [](bool present) { return present; }))
        throw std::invalid_argument("partial CUDA domain configuration dispatch");
    if (!domain_thread_identity) domain_thread_identity = std::make_shared<int>(0);
    implementation_ = std::make_unique<Implementation>(dispatch, config);
}

CudaExecutionDomainOwner::~CudaExecutionDomainOwner() {
    auto& state = *implementation_;
    if (!state.owns() ||
        (!state.is_shutdown &&
         (state.has_native_owner || state.has_member_owners() || state.reserved_device_bytes)))
        (void)implementation_.release();
}

CUresult CudaExecutionDomainOwner::reserve_process_bytes(std::size_t bytes) noexcept {
    auto& state = *implementation_;
    const auto status = state.gate(true);
    if (status != CUDA_SUCCESS) return status;
    if (state.is_closed || state.is_shutdown) return CUDA_ERROR_DEINITIALIZED;
    if (!state.is_initialized) return CUDA_ERROR_NOT_INITIALIZED;
    if (!bytes || state.reserved_device_bytes > state.config.maximum_device_bytes ||
        bytes > state.config.maximum_device_bytes - state.reserved_device_bytes)
        return CUDA_ERROR_OUT_OF_MEMORY;
    state.reserved_device_bytes += bytes;
    state.process_device_bytes += bytes;
    return CUDA_SUCCESS;
}
CUresult CudaExecutionDomainOwner::release_process_bytes(std::size_t bytes) noexcept {
    auto& state = *implementation_;
    const auto status = state.gate(true);
    if (status != CUDA_SUCCESS) return status;
    if (!bytes || bytes > state.process_device_bytes || bytes > state.reserved_device_bytes)
        return state.fail(CUDA_ERROR_INVALID_VALUE);
    state.process_device_bytes -= bytes;
    state.reserved_device_bytes -= bytes;
    return CUDA_SUCCESS;
}

CUresult CudaExecutionDomainOwner::initialize() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    if (state.is_closed) return CUDA_ERROR_DEINITIALIZED;
    if (state.has_initialize_attempt) return CUDA_ERROR_INVALID_VALUE;
    const DomainState::Scope scope(state);
    state.has_initialize_attempt = true;
    try {
        // Driver initialization is the prerequisite for meaningful context
        // queries. Exclusion is already active before this first SDK callback.
        auto status = state.dispatch.initialize(0);
        if (status != CUDA_SUCCESS) return status;
        if (state.health() != CUDA_SUCCESS) return state.health();
        CUcontext observed{};
        status = state.current(observed);
        if (status != CUDA_SUCCESS) return status;
        if (observed) return CUDA_ERROR_INVALID_CONTEXT;
        CUdevice device{-1};
        status = state.dispatch.device_get(&device, state.config.device_ordinal);
        const auto device_proof = state.prove(nullptr);
        if (device_proof != CUDA_SUCCESS) return device_proof;
        if (status != CUDA_SUCCESS) return status;
        if (device < 0) return state.fail(CUDA_ERROR_UNKNOWN);
        state.native_device = device;

        // A throw, a partial output or an unproven error after this boundary
        // retains the entire owner. No destructor guesses how to undo it.
        state.has_native_owner = true;
        status = state.dispatch.context_create(&state.context, state.config.flags, device);
        observed = nullptr;
        const auto create_proof = state.current(observed);
        if (create_proof != CUDA_SUCCESS) return create_proof;
        if (status != CUDA_SUCCESS) {
            if (!state.context && !observed && clean_creation_error(status)) {
                state.has_native_owner = false;
                return status;
            }
            return state.fail(status);
        }
        if (!state.context) return state.fail(CUDA_ERROR_UNKNOWN);
        if (observed != state.context) return state.fail(CUDA_ERROR_INVALID_CONTEXT);
        if (state.can_virtualize_configuration) {
            status = state.capture_configuration(state.baseline_configuration);
            if (status != CUDA_SUCCESS) return state.fail(status);
            state.physical_configuration = state.baseline_configuration;
        }
        status = state.dispatch.context_set_current(nullptr);
        const auto restore_proof = state.prove(nullptr);
        if (status != CUDA_SUCCESS) return state.fail(status);
        if (restore_proof != CUDA_SUCCESS) return restore_proof;
        state.is_initialized = true;
        return CUDA_SUCCESS;
    } catch (...) {
        return state.fail(CUDA_ERROR_UNKNOWN);
    }
}

CudaExecutionDomainMemberResult CudaExecutionDomainOwner::create_member(
    CudaExecutionDomainLogicalIdentity logical,
    std::optional<unsigned int> requested_flags) noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate, {}};
    if (state.is_closed) return {CUDA_ERROR_DEINITIALIZED, {}};
    if (!state.is_initialized) return {CUDA_ERROR_NOT_INITIALIZED, {}};
    if (!valid_identity(logical)) return {CUDA_ERROR_INVALID_VALUE, {}};
    auto logical_flags = state.config.flags;
    if (requested_flags) {
        if (!valid_flags(*requested_flags)) {
            const auto without_coredump =
                *requested_flags &
                ~static_cast<unsigned int>(CU_CTX_COREDUMP_ENABLE | CU_CTX_USER_COREDUMP_ENABLE);
            return {
                valid_flags(without_coredump) ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE,
                {}};
        }
        logical_flags = *requested_flags | (state.config.flags & CU_CTX_MAP_HOST);
        if (logical_flags != state.baseline_configuration.values[0] &&
            !state.can_virtualize_configuration)
            return {CUDA_ERROR_NOT_SUPPORTED, {}};
    }
    state.prune();
    if (std::ranges::any_of(state.members, [&](const auto& member) {
            return same_identity(member->logical, logical);
        }))
        return {CUDA_ERROR_INVALID_VALUE, {}};
    if (!state.next_generation || state.members.size() == state.config.maximum_members)
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    try {
        auto member = std::make_shared<CudaExecutionDomainMemberState>();
        member->domain = &state;
        member->logical = logical;
        member->generation = state.next_generation;
        member->configuration = state.baseline_configuration;
        member->initial_flags = logical_flags;
        if (requested_flags) member->configuration.values[0] = logical_flags;
        state.members.push_back(member);
        state.next_generation = state.next_generation == std::numeric_limits<std::uint64_t>::max()
                                    ? 0
                                    : state.next_generation + 1;
        CudaExecutionDomainMember result;
        result.state_ = std::move(member);
        return {CUDA_SUCCESS, std::move(result)};
    } catch (...) {
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    }
}

CudaExecutionDomainSnapshot CudaExecutionDomainOwner::snapshot() const noexcept {
    const auto& state = *implementation_;
    CudaExecutionDomainSnapshot result;
    if (!state.owns()) return result;
    result.status = state.health();
    result.retained_members = state.members.size();
    result.attached_members =
        static_cast<std::size_t>(std::ranges::count_if(state.members, [](const auto& member) {
            return member->is_attached && !member->is_retired;
        }));
    result.reserved_device_bytes = state.reserved_device_bytes;
    result.has_context = state.context != nullptr;
    result.is_closed = state.is_closed;
    result.is_shutdown = state.is_shutdown;
    result.is_quarantined = state.activity->is_quarantined;
    return result;
}

CUresult CudaExecutionDomainOwner::close() noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.has_activity()) return CUDA_ERROR_NOT_READY;
    state.is_closed = true;
    return state.health();
}

CUresult CudaExecutionDomainOwner::shutdown() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return gate;
    if (state.is_shutdown) return CUDA_SUCCESS;
    if (!state.is_closed || state.reserved_device_bytes || state.has_member_owners())
        return CUDA_ERROR_NOT_READY;
    if (!state.has_native_owner) {
        state.members.clear();
        state.is_shutdown = true;
        return CUDA_SUCCESS;
    }
    const DomainState::Scope scope(state);
    try {
        CUcontext observed{};
        const auto proof = state.current(observed);
        if (proof != CUDA_SUCCESS) return proof;
        if (observed) return CUDA_ERROR_INVALID_CONTEXT;
        if (!state.is_initialized || !state.context || state.has_destroy_attempt)
            return state.fail(CUDA_ERROR_UNKNOWN);
        state.has_destroy_attempt = true;
        const auto status = state.dispatch.context_destroy(state.context);
        const auto after = state.prove(nullptr);
        if (status != CUDA_SUCCESS) return state.fail(status);
        if (after != CUDA_SUCCESS) return after;
        state.context = nullptr;
        state.has_native_owner = false;
        state.is_shutdown = true;
        state.members.clear();
        return CUDA_SUCCESS;
    } catch (...) {
        return state.fail(CUDA_ERROR_UNKNOWN);
    }
}

CudaExecutionDomainMemberView CudaExecutionDomainMember::view(
    bool owns_native_scope) const noexcept {
    CudaExecutionDomainMemberView result;
    if (!state_) return result;
    result.status = state_->gate(owns_native_scope);
    if (result.status != CUDA_SUCCESS) return result;
    const auto& domain = *state_->domain;
    result.context = {domain.config.identity, domain.context, state_};
    result.logical = state_->logical;
    result.generation = state_->generation;
    result.device_ordinal = domain.config.device_ordinal;
    result.native_device = domain.native_device;
    result.flags = domain.config.flags;
    result.logical_flags = state_->initial_flags;
    result.is_attached = state_->is_attached;
    result.configuration_epoch = state_->configuration_epoch;
    result.physical_configuration_epoch = domain.physical_configuration_epoch;
    result.is_configuration_materialized = state_->configuration == domain.physical_configuration;
    result.can_virtualize_configuration = domain.can_virtualize_configuration;
    result.has_unobserved_configuration_effects = domain.has_unobserved_configuration_effects;
    return result;
}

bool CudaExecutionDomainMember::can_share_configuration(
    const CudaExecutionDomainMember& other) const noexcept {
    if (!state_ || !other.state_ || state_ == other.state_ || state_->gate() != CUDA_SUCCESS ||
        other.state_->gate() != CUDA_SUCCESS || state_->domain != other.state_->domain ||
        !state_->is_attached || !other.state_->is_attached)
        return false;
    return state_->configuration == other.state_->configuration &&
           state_->configuration == state_->domain->physical_configuration;
}

std::shared_ptr<CudaExecutionOwnerActivity> CudaExecutionDomainMember::owner_activity()
    const noexcept {
    if (!state_ || !state_->domain->owns() || state_->is_retired || state_->domain->is_shutdown)
        return {};
    return state_->domain->activity;
}

CUresult CudaExecutionDomainMember::attach(CudaExecutionDomainLogicalIdentity logical) noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate();
    if (status != CUDA_SUCCESS) return status;
    if (state_->domain->is_closed) return CUDA_ERROR_DEINITIALIZED;
    if (!same_identity(logical, state_->logical) || state_->is_attached)
        return CUDA_ERROR_INVALID_VALUE;
    state_->is_attached = true;
    return CUDA_SUCCESS;
}

CUresult CudaExecutionDomainMember::reserve_device_bytes(std::size_t bytes) noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate(true);
    if (status != CUDA_SUCCESS) return status;
    if (!state_->is_attached || !bytes) return CUDA_ERROR_INVALID_VALUE;
    auto& domain = *state_->domain;
    if (domain.reserved_device_bytes > domain.config.maximum_device_bytes ||
        bytes > domain.config.maximum_device_bytes - domain.reserved_device_bytes)
        return CUDA_ERROR_OUT_OF_MEMORY;
    domain.reserved_device_bytes += bytes;
    state_->reserved_device_bytes += bytes;
    return CUDA_SUCCESS;
}

CUresult CudaExecutionDomainMember::release_device_bytes(std::size_t bytes) noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate(true);
    if (status != CUDA_SUCCESS) return status;
    if (!state_->is_attached || !bytes || bytes > state_->reserved_device_bytes)
        return CUDA_ERROR_INVALID_VALUE;
    auto& domain = *state_->domain;
    if (bytes > domain.reserved_device_bytes) return domain.fail(CUDA_ERROR_UNKNOWN);
    state_->reserved_device_bytes -= bytes;
    domain.reserved_device_bytes -= bytes;
    return CUDA_SUCCESS;
}

CUresult CudaExecutionDomainMember::set_physical_reservation(bool has_physical_work) noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate(true);
    if (status != CUDA_SUCCESS) return status;
    if (!state_->is_attached) return CUDA_ERROR_INVALID_VALUE;
    if (has_physical_work && state_->configuration != state_->domain->physical_configuration)
        return CUDA_ERROR_NOT_READY;
    if (has_physical_work && state_->domain->can_virtualize_configuration)
        state_->domain->has_unobserved_configuration_effects = true;
    state_->has_physical_work = has_physical_work;
    return CUDA_SUCCESS;
}

CUresult CudaExecutionDomainMember::other_members_quiescent() const noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate(true);
    if (status != CUDA_SUCCESS) return status;
    return std::ranges::any_of(
               state_->domain->members,
               [&](const auto& other) { return other != state_ && other->has_physical_work; })
               ? CUDA_ERROR_NOT_READY
               : CUDA_SUCCESS;
}

CUresult CudaExecutionDomainMember::configuration_status(bool owns_native_scope) const noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate(owns_native_scope);
    if (status != CUDA_SUCCESS) return status;
    if (!state_->is_attached) return CUDA_ERROR_INVALID_VALUE;
    return state_->configuration == state_->domain->physical_configuration ? CUDA_SUCCESS
                                                                           : CUDA_ERROR_NOT_READY;
}

CudaExecutionDomainConfigurationView CudaExecutionDomainMember::configuration(
    bool owns_native_scope) const noexcept {
    const auto status = configuration_status(owns_native_scope);
    if (status != CUDA_SUCCESS) return {.status = status};
    const auto& domain = *state_->domain;
    if (!domain.can_virtualize_configuration) return {.status = CUDA_ERROR_NOT_SUPPORTED};
    if (domain.has_unobserved_configuration_effects) return {.status = CUDA_ERROR_NOT_READY};
    return {CUDA_SUCCESS, state_->configuration.available, state_->configuration.values};
}

CUresult CudaExecutionDomainMember::verify_profile_configuration(
    const CudaExecutionDomainConfigurationView& expected, bool owns_native_scope) noexcept {
    const auto status = configuration_status(owns_native_scope);
    if (status != CUDA_SUCCESS) return status;
    auto& domain = *state_->domain;
    if (!domain.can_virtualize_configuration) return CUDA_ERROR_NOT_SUPPORTED;
    if (expected.status != CUDA_SUCCESS || expected.available != state_->configuration.available ||
        expected.values != state_->configuration.values)
        return CUDA_ERROR_INVALID_VALUE;
    std::optional<DomainState::Scope> scope;
    if (!owns_native_scope) scope.emplace(domain);
    try {
        const auto proof = domain.prove(domain.context);
        if (proof != CUDA_SUCCESS) return proof;
        DomainConfiguration actual;
        const auto captured = domain.capture_configuration(actual);
        if (captured != CUDA_SUCCESS) return domain.fail(captured);
        if (actual != state_->configuration) return domain.fail(CUDA_ERROR_UNKNOWN);
        return CUDA_SUCCESS;
    } catch (...) {
        return domain.fail(CUDA_ERROR_UNKNOWN);
    }
}

CUresult CudaExecutionDomainMember::materialize_configuration(bool owns_native_scope) noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate(owns_native_scope);
    if (status != CUDA_SUCCESS) return status;
    if (!state_->is_attached) return CUDA_ERROR_INVALID_VALUE;
    auto& domain = *state_->domain;
    std::optional<DomainState::Scope> scope;
    if (!owns_native_scope) scope.emplace(domain);
    try {
        return domain.materialize(state_->configuration);
    } catch (...) {
        return domain.fail(CUDA_ERROR_UNKNOWN);
    }
}

CUresult CudaExecutionDomainMember::verify_materialized_configuration(
    bool owns_native_scope) noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate(owns_native_scope);
    if (status != CUDA_SUCCESS) return status;
    if (!state_->is_attached) return CUDA_ERROR_INVALID_VALUE;
    auto& domain = *state_->domain;
    std::optional<DomainState::Scope> scope;
    if (!owns_native_scope) scope.emplace(domain);
    try {
        return domain.verify_configuration();
    } catch (...) {
        return domain.fail(CUDA_ERROR_UNKNOWN);
    }
}

CudaExecutionDomainConfigurationResult CudaExecutionDomainMember::mutate_configuration(
    CudaExecutionDomainConfigurationMutation mutation, std::uint32_t selector,
    CudaExecutionDomainConfigurationCall call, bool owns_native_scope) noexcept {
    CudaExecutionDomainConfigurationResult result;
    if (!state_) return result;
    result.status = state_->gate(owns_native_scope);
    if (result.status != CUDA_SUCCESS) return result;
    auto& domain = *state_->domain;
    using Mutation = CudaExecutionDomainConfigurationMutation;
    if (!state_->is_attached || !call.state || !call.invoke ||
        (mutation != Mutation::kFlags && mutation != Mutation::kLimit &&
         mutation != Mutation::kCache && mutation != Mutation::kSharedMemory &&
         mutation != Mutation::kResetPersistingCache) ||
        (mutation == Mutation::kLimit ? selector >= kLimitFields : selector != 0)) {
        result.status = CUDA_ERROR_INVALID_VALUE;
        return result;
    }
    if (!domain.can_virtualize_configuration) {
        result.status = CUDA_ERROR_NOT_SUPPORTED;
        return result;
    }
    if (domain.has_physical_work()) {
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    if (state_->configuration_epoch == std::numeric_limits<std::uint64_t>::max() ||
        domain.physical_configuration_epoch == std::numeric_limits<std::uint64_t>::max()) {
        result.status = CUDA_ERROR_OUT_OF_MEMORY;
        return result;
    }
    std::optional<DomainState::Scope> scope;
    if (!owns_native_scope) scope.emplace(domain);
    try {
        result.status = domain.materialize(state_->configuration);
        if (result.status != CUDA_SUCCESS) return result;
        // Materialization may itself consume the final physical epoch.
        if (domain.physical_configuration_epoch == std::numeric_limits<std::uint64_t>::max()) {
            result.status = CUDA_ERROR_OUT_OF_MEMORY;
            return result;
        }
        result.status = domain.prove(domain.context);
        if (result.status != CUDA_SUCCESS) return result;
        result.original_status = call.invoke(call.state);
        ++state_->configuration_epoch;
        ++domain.physical_configuration_epoch;
        result.status = domain.prove(domain.context);
        if (result.status != CUDA_SUCCESS) return result;
        DomainConfiguration actual;
        result.status = domain.capture_configuration(actual);
        if (result.status != CUDA_SUCCESS) {
            (void)domain.fail(result.status);
            return result;
        }
        if (actual.available != domain.baseline_configuration.available) {
            result.status = domain.fail(CUDA_ERROR_UNKNOWN);
            return result;
        }
        state_->configuration = domain.physical_configuration = actual;
        return result;
    } catch (...) {
        result.status = domain.fail(CUDA_ERROR_UNKNOWN);
        return result;
    }
}

void CudaExecutionDomainMember::retain_uncertain_device_bytes(std::size_t bytes) noexcept {
    if (!state_ || !state_->domain->owns() || state_->is_retired || state_->domain->is_shutdown)
        return;
    auto& domain = *state_->domain;
    state_->reserved_device_bytes = saturated_add(state_->reserved_device_bytes, bytes);
    domain.reserved_device_bytes = saturated_add(domain.reserved_device_bytes, bytes);
    (void)domain.fail(CUDA_ERROR_UNKNOWN);
}

void CudaExecutionDomainMember::quarantine() noexcept {
    if (!state_ || !state_->domain->owns() || state_->is_retired || state_->domain->is_shutdown)
        return;
    (void)state_->domain->fail(CUDA_ERROR_UNKNOWN);
}

CUresult CudaExecutionDomainMember::retire() noexcept {
    if (!state_) return CUDA_ERROR_INVALID_HANDLE;
    const auto status = state_->gate();
    if (status != CUDA_SUCCESS) return status;
    if (!state_->is_attached) return CUDA_ERROR_INVALID_VALUE;
    if (state_->reserved_device_bytes || state_->has_physical_work) return CUDA_ERROR_NOT_READY;
    state_->is_retired = true;
    return CUDA_SUCCESS;
}

}  // namespace glimmer::backend::cuda
