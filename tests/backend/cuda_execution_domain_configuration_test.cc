#include "internal/cuda_execution_domain.h"

#include <array>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>

namespace {
using namespace glimmer::backend::cuda;
using Mutation = CudaExecutionDomainConfigurationMutation;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Native {
    CUcontext current{};
    unsigned int flags{CU_CTX_MAP_HOST};
    CUfunc_cache cache{CU_FUNC_CACHE_PREFER_NONE};
    CUsharedconfig shared{CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE};
    std::array<std::size_t, 3> limits{1024, 4096, 8192};
    unsigned int creates{}, destroys{}, setters{}, queries{}, requested{};
    CUresult next_query{CUDA_SUCCESS}, next_set{CUDA_SUCCESS};
    bool should_misround{}, should_change_context{}, should_corrupt_cache{},
        should_corrupt_shared{}, should_corrupt_flags{}, should_normalize_creation_flags{};
    std::function<void()> on_set;
    CUcontext context() {
        return reinterpret_cast<CUcontext>(this);
    }
    CUresult query() {
        ++queries;
        return std::exchange(next_query, CUDA_SUCCESS);
    }
    CUresult set() {
        ++setters;
        if (on_set) on_set();
        if (should_change_context) current = nullptr;
        return std::exchange(next_set, CUDA_SUCCESS);
    }
} native;

CUresult CUDAAPI initialize(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device_get(CUdevice* output, int ordinal) {
    *output = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_create(CUcontext* output, unsigned int flags, CUdevice) {
    ++native.creates;
    native.flags = flags | (native.should_normalize_creation_flags
                                ? static_cast<unsigned int>(CU_CTX_LMEM_RESIZE_TO_MAX)
                                : 0U);
    *output = native.current = native.context();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_set(CUcontext context) {
    native.current = context;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_get(CUcontext* output) {
    *output = native.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_destroy(CUcontext context) {
    expect(context == native.context() && !native.current, "destroy exact domain only from NULL");
    ++native.destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_flags(unsigned int* output) {
    *output = native.should_corrupt_flags ? 0x40000000U : native.flags;
    return native.query();
}
CUresult CUDAAPI set_flags(unsigned int flags) {
    const auto result = native.set();
    if (result == CUDA_SUCCESS) native.flags = flags;
    return result;
}
CUresult CUDAAPI get_cache(CUfunc_cache* output) {
    if (native.should_corrupt_cache) {
        const std::uint32_t bits = 7;
        std::memcpy(output, &bits, sizeof(bits));
    } else
        *output = native.cache;
    return native.query();
}
CUresult CUDAAPI set_cache(CUfunc_cache cache) {
    const auto result = native.set();
    if (result == CUDA_SUCCESS) native.cache = cache;
    return result;
}
CUresult CUDAAPI get_shared(CUsharedconfig* output) {
    if (native.should_corrupt_shared) {
        const std::uint32_t bits = UINT32_MAX;
        std::memcpy(output, &bits, sizeof(bits));
    } else
        *output = native.shared;
    return native.query();
}
CUresult CUDAAPI set_shared(CUsharedconfig shared) {
    const auto result = native.set();
    if (result == CUDA_SUCCESS) native.shared = shared;
    return result;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native CUDA ABI.
CUresult CUDAAPI get_limit(std::size_t* output, CUlimit selector) {
    const auto result = native.query();
    if (result != CUDA_SUCCESS) return result;
    const auto index = static_cast<unsigned int>(selector);
    if (index >= native.limits.size())
        return index == 14   ? CUDA_ERROR_NOT_SUPPORTED
               : index == 15 ? CUDA_ERROR_INVALID_VALUE
                             : CUDA_ERROR_UNSUPPORTED_LIMIT;
    *output = native.limits[index];
    return CUDA_SUCCESS;
}
CUresult CUDAAPI set_limit(CUlimit selector, std::size_t bytes) {
    const auto result = native.set();
    if (result != CUDA_SUCCESS) return result;
    const auto index = static_cast<unsigned int>(selector);
    if (index >= native.limits.size()) return CUDA_ERROR_UNSUPPORTED_LIMIT;
    native.limits[index] = ((bytes + 255) / 256) * 256 + (native.should_misround ? 256 : 0);
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CudaExecutionDomainDispatch dispatch() {
    return {
        initialize,
        device_get,
        context_create,
        context_set,
        context_get,
        context_destroy,
        {get_flags, set_flags, get_limit, set_limit, get_cache, set_cache, get_shared, set_shared}};
}
struct Call {
    std::function<std::int32_t()> function;
    CudaExecutionDomainConfigurationCall borrowed() {
        return {this, [](void* state) {
                    ++native.requested;
                    return static_cast<Call*>(state)->function();
                }};
    }
};
struct Fixture {
    CudaExecutionDomainOwner owner{dispatch(), {99, 0, CU_CTX_MAP_HOST, 3, 4096}};
    CudaExecutionDomainMember first, second;
    Fixture() {
        expect(owner.initialize() == CUDA_SUCCESS && native.queries == 19 && !native.current,
               "fresh bounded baseline queries flags/cache/shared/all16 limit selectors");
        auto a = owner.create_member({{1, 99}, 101});
        auto b = owner.create_member({{2, 99}, 202});
        expect(a.status == CUDA_SUCCESS && b.status == CUDA_SUCCESS, "two logical profiles");
        first = std::move(a.member);
        second = std::move(b.member);
        expect(first.attach({{1, 99}, 101}) == CUDA_SUCCESS &&
                   second.attach({{2, 99}, 202}) == CUDA_SUCCESS,
               "attach exact logical identities before configuration use");
        native.current = native.context();
    }
    void shutdown() {
        expect(first.retire() == CUDA_SUCCESS && second.retire() == CUDA_SUCCESS,
               "profile metadata cannot replace actual resource/physical retirement");
        first = {};
        second = {};
        native.current = nullptr;
        expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
                   native.creates == 1 && native.destroys == 1,
               "configuration teardown never resets or destroys a sibling/native context twice");
    }
};

void independent_normalized_profiles_and_epochs() {
    native = {};
    Fixture fixture;
    const auto first_epoch = fixture.first.view().configuration_epoch;
    Call change{[] { return static_cast<int>(set_limit(CU_LIMIT_STACK_SIZE, 1200)); }};
    const auto changed = fixture.first.mutate_configuration(Mutation::kLimit, 0, change.borrowed());
    expect(changed.status == CUDA_SUCCESS && changed.original_status == CUDA_SUCCESS &&
               native.limits[0] == 1280 && native.requested == 1 &&
               fixture.first.view().configuration_epoch == first_epoch + 1 &&
               fixture.second.view().configuration_epoch == 1 &&
               !fixture.second.view().is_configuration_materialized,
           "SDK-normalized mutation advances only its logical epoch and exact physical epoch");
    expect(fixture.second.configuration_status() == CUDA_ERROR_NOT_READY &&
               fixture.second.set_physical_reservation(true) == CUDA_ERROR_NOT_READY,
           "stale logical configuration cannot publish physical work");
    const auto setters = native.setters;
    expect(fixture.second.materialize_configuration() == CUDA_SUCCESS && native.limits[0] == 1024 &&
               native.setters == setters + 1 &&
               fixture.first.configuration_status() == CUDA_ERROR_NOT_READY,
           "sibling restores its own fresh baseline, only the differing native field");
    expect(fixture.first.materialize_configuration() == CUDA_SUCCESS && native.limits[0] == 1280,
           "original owner restores its saved normalized value, never rounds a caller guess");
    const auto queries = native.queries;
    expect(fixture.first.materialize_configuration() == CUDA_SUCCESS && native.queries == queries,
           "already materialized configuration performs no SDK calls");
    fixture.shutdown();
}

void held_physical_work_and_equal_profiles() {
    native = {};
    Fixture fixture;
    expect(fixture.first.set_physical_reservation(true) == CUDA_SUCCESS &&
               fixture.second.materialize_configuration() == CUDA_SUCCESS &&
               fixture.second.set_physical_reservation(true) == CUDA_SUCCESS,
           "equal baseline profiles do not serialize independent physical members");
    Call change{[] { return static_cast<int>(set_cache(CU_FUNC_CACHE_PREFER_SHARED)); }};
    expect(fixture.first.mutate_configuration(Mutation::kCache, 0, change.borrowed()).status ==
                   CUDA_ERROR_NOT_READY &&
               !native.requested,
           "own and sibling actual tails defer original SDK mutation before native entry");
    expect(fixture.first.set_physical_reservation(false) == CUDA_SUCCESS &&
               fixture.second.set_physical_reservation(false) == CUDA_SUCCESS,
           "trusted stop proof, not an ACK or configuration intent, clears physical work");
    expect(fixture.first.mutate_configuration(Mutation::kCache, 0, change.borrowed()).status ==
                   CUDA_SUCCESS &&
               fixture.first.set_physical_reservation(true) == CUDA_SUCCESS,
           "mutated first member may issue only its current profile");
    const auto setters = native.setters;
    expect(fixture.second.materialize_configuration() == CUDA_ERROR_NOT_READY &&
               native.setters == setters &&
               fixture.second.configuration_status() == CUDA_ERROR_NOT_READY,
           "different profile does not enter a synchronizing SDK while useful work is held");
    expect(fixture.first.set_physical_reservation(false) == CUDA_SUCCESS &&
               fixture.second.materialize_configuration() == CUDA_SUCCESS &&
               native.cache == CU_FUNC_CACHE_PREFER_NONE,
           "actual stop permits exact baseline materialization");
    fixture.shutdown();
}

void native_error_and_reentrant_lifetimes() {
    native = {};
    Fixture fixture;
    bool reentry_blocked{};
    native.on_set = [&] {
        const auto requested = native.requested;
        Call nested{[] { return 0; }};
        reentry_blocked =
            fixture.second.materialize_configuration() == CUDA_ERROR_NOT_READY &&
            fixture.first.mutate_configuration(Mutation::kFlags, 0, nested.borrowed()).status ==
                CUDA_ERROR_NOT_READY &&
            fixture.owner.close() == CUDA_ERROR_NOT_READY && native.requested == requested;
    };
    Call failed{[] {
        (void)set_limit(CU_LIMIT_STACK_SIZE, 4096);
        return static_cast<int>(CUDA_ERROR_INVALID_VALUE);
    }};
    const auto result = fixture.first.mutate_configuration(Mutation::kLimit, 0, failed.borrowed());
    expect(
        result.status == CUDA_SUCCESS && result.original_status == CUDA_ERROR_INVALID_VALUE &&
            reentry_blocked && !fixture.owner.snapshot().is_quarantined &&
            fixture.first.view().configuration_epoch == 2,
        "original error is preserved even if SDK changed state; callback lifecycle reentry blocks");
    native.on_set = {};
    expect(fixture.second.materialize_configuration() == CUDA_SUCCESS && native.limits[0] == 1024 &&
               fixture.first.materialize_configuration() == CUDA_SUCCESS &&
               native.limits[0] == 4096,
           "post-error actual snapshot, not original success inference, defines restoration");
    Call unchanged{[] { return static_cast<int>(CUDA_ERROR_UNSUPPORTED_LIMIT); }};
    const auto unchanged_result =
        fixture.first.mutate_configuration(Mutation::kLimit, 15, unchanged.borrowed());
    expect(unchanged_result.status == CUDA_SUCCESS &&
               unchanged_result.original_status == CUDA_ERROR_UNSUPPORTED_LIMIT &&
               fixture.first.view().configuration_epoch == 3,
           "unsupported actual SDK selector retains original error and invalidates old claims");
    fixture.shutdown();
}

void owner_validation_and_fresh_members() {
    native = {};
    Fixture fixture;
    Call flags{[] { return static_cast<int>(set_flags(CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST)); }};
    expect(fixture.first.mutate_configuration(Mutation::kFlags, 0, flags.borrowed()).status ==
               CUDA_SUCCESS,
           "ordinary safe flags are captured as an independent logical setting");
    auto fresh = fixture.owner.create_member({{3, 99}, 303});
    expect(fresh.status == CUDA_SUCCESS && fresh.member.attach({{3, 99}, 303}) == CUDA_SUCCESS &&
               fresh.member.configuration_status() == CUDA_ERROR_NOT_READY &&
               fresh.member.materialize_configuration() == CUDA_SUCCESS &&
               native.flags == CU_CTX_MAP_HOST,
           "fresh members inherit the original baseline, never a previous member's mutation");
    const auto requested = native.requested;
    expect(fixture.first.mutate_configuration(Mutation::kFlags, 1, flags.borrowed()).status ==
                   CUDA_ERROR_INVALID_VALUE &&
               fixture.first.mutate_configuration(Mutation::kLimit, 16, flags.borrowed()).status ==
                   CUDA_ERROR_INVALID_VALUE &&
               fixture.first.mutate_configuration(static_cast<Mutation>(7), 0, flags.borrowed())
                       .status == CUDA_ERROR_INVALID_VALUE &&
               fixture.first.mutate_configuration(Mutation::kFlags, 0, {}).status ==
                   CUDA_ERROR_INVALID_VALUE &&
               native.requested == requested,
           "unknown kinds/selectors/borrowed callbacks reject without native entry");
    const auto foreign = std::async(std::launch::async, [&] {
                             return fixture.first.materialize_configuration();
                         }).get();
    expect(foreign == CUDA_ERROR_INVALID_CONTEXT,
           "foreign host thread cannot mutate native context");
    const auto child = ::fork();
    expect(child >= 0, "fork configuration owner gate");
    if (child == 0)
        _exit(fixture.first.materialize_configuration() == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects before inherited native state or owner TLS");
    expect(fresh.member.retire() == CUDA_SUCCESS, "fresh member retires its metadata only");
    fresh.member = {};
    fixture.shutdown();
}

void retained_unknown(int scenario) {
    native = {};
    Fixture fixture;
    Call change{[] { return static_cast<int>(set_limit(CU_LIMIT_STACK_SIZE, 4096)); }};
    if (scenario == 0) {
        native.next_query = CUDA_ERROR_ILLEGAL_ADDRESS;
    } else if (scenario == 1) {
        native.should_change_context = true;
    } else if (scenario == 2) {
        change.function = []() -> std::int32_t {
            throw std::runtime_error("native callback exception");
        };
    } else if (scenario == 7) {
        native.should_corrupt_cache = true;
    } else if (scenario == 8) {
        native.should_corrupt_shared = true;
    } else if (scenario == 9) {
        native.should_corrupt_flags = true;
    }
    auto result = fixture.first.mutate_configuration(Mutation::kLimit, 0, change.borrowed());
    if (scenario >= 3) {
        if (scenario < 7)
            expect(result.status == CUDA_SUCCESS, "prepare a genuine different logical profile");
        if (scenario == 3) native.next_set = CUDA_ERROR_ILLEGAL_ADDRESS;
        if (scenario == 4) native.should_misround = true;
        if (scenario == 5) native.next_query = CUDA_ERROR_ILLEGAL_ADDRESS;
        if (scenario == 6) {
            expect(fixture.first.set_physical_reservation(true) == CUDA_SUCCESS &&
                       fixture.first.set_physical_reservation(false) == CUDA_SUCCESS,
                   "actual physical interval marks cached configuration unobserved");
            native.limits[0] = 8192;
        }
        if (scenario < 7) result.status = fixture.second.materialize_configuration();
    } else if (scenario != 2) {
        expect(result.original_status == CUDA_SUCCESS,
               "post-request failure keeps exact actual SDK status separately");
    }
    expect(result.status != CUDA_SUCCESS && fixture.owner.snapshot().is_quarantined &&
               !native.destroys,
           "unknown native configuration retains the physical cohort");
    const auto setters = native.setters, requested = native.requested;
    expect(
        fixture.first.materialize_configuration() != CUDA_SUCCESS &&
            fixture.second.mutate_configuration(Mutation::kLimit, 0, change.borrowed()).status !=
                CUDA_SUCCESS &&
            fixture.owner.close() != CUDA_SUCCESS && fixture.owner.shutdown() != CUDA_SUCCESS &&
            native.setters == setters && native.requested == requested && !native.destroys,
        "uncertain capture/restore never retries a setter, resets a context or invents retirement");
}

void partial_dispatch_rejects() {
    native = {};
    auto callbacks = dispatch();
    callbacks.configuration.set_limit = nullptr;
    bool rejected{};
    try {
        CudaExecutionDomainOwner owner(callbacks, {99, 0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    expect(rejected && !native.creates,
           "partial configuration capability rejects before SDK creation");
}
void cold_requested_flags_preserve_independent_profiles() {
    native = {};
    CudaExecutionDomainOwner owner(dispatch(), {99, 0, CU_CTX_MAP_HOST, 3, 4096});
    expect(owner.initialize() == CUDA_SUCCESS, "cold flag baseline initializes once");
    const auto calls = native.queries + native.setters;
    auto first = owner.create_member({{1, 99}, 101});
    auto second = owner.create_member({{2, 99}, 202}, CU_CTX_SCHED_YIELD);
    expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS &&
               native.queries + native.setters == calls &&
               first.member.view().logical_flags == CU_CTX_MAP_HOST &&
               second.member.view().flags == CU_CTX_MAP_HOST &&
               second.member.view().logical_flags == (CU_CTX_MAP_HOST | CU_CTX_SCHED_YIELD) &&
               !second.member.view().is_configuration_materialized,
           "requested flags seed only bounded logical metadata, never a speculative SDK call");
    for (const auto flags : {3U, 0x100U, 0x100U | CU_CTX_COREDUMP_ENABLE})
        expect(owner.create_member({{3, 99}, 303}, flags).status == CUDA_ERROR_INVALID_VALUE,
               "invalid flags reject without consuming a member identity");
    expect(owner.create_member({{3, 99}, 303}, CU_CTX_COREDUMP_ENABLE).status ==
                   CUDA_ERROR_NOT_SUPPORTED &&
               owner.snapshot().retained_members == 2 && native.queries + native.setters == calls,
           "coredump rejection cannot change security policy or publish a member");
    expect(first.member.attach({{1, 99}, 101}) == CUDA_SUCCESS &&
               second.member.attach({{2, 99}, 202}) == CUDA_SUCCESS,
           "two independent exact logical contexts attach");
    native.current = native.context();
    expect(first.member.set_physical_reservation(true) == CUDA_SUCCESS &&
               second.member.materialize_configuration() == CUDA_ERROR_NOT_READY &&
               native.setters == 0,
           "different initial flags cannot pass a sibling's real physical reservation");
    expect(first.member.set_physical_reservation(false) == CUDA_SUCCESS &&
               second.member.materialize_configuration() == CUDA_SUCCESS &&
               native.flags == (CU_CTX_MAP_HOST | CU_CTX_SCHED_YIELD) &&
               second.member.configuration().values[0] == native.flags,
           "only quiescent verified materialization installs the new member flags");
    expect(first.member.materialize_configuration() == CUDA_SUCCESS &&
               native.flags == CU_CTX_MAP_HOST &&
               second.member.view().logical_flags == (CU_CTX_MAP_HOST | CU_CTX_SCHED_YIELD),
           "switching restores the sibling baseline without changing another logical profile");
    expect(first.member.retire() == CUDA_SUCCESS && second.member.retire() == CUDA_SUCCESS,
           "flag profiles do not confer resource retirement authority");
    first.member = {};
    second.member = {};
    native.current = nullptr;
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               native.creates == 1 && native.destroys == 1,
           "one physical owner remains until all logical members retire");

    native = {};
    auto immutable_dispatch = dispatch();
    immutable_dispatch.configuration = {};
    CudaExecutionDomainOwner immutable(immutable_dispatch, {100, 0, CU_CTX_MAP_HOST, 1, 4096});
    expect(immutable.initialize() == CUDA_SUCCESS &&
               immutable.create_member({{3, 100}, 303}, CU_CTX_SCHED_SPIN).status ==
                   CUDA_ERROR_NOT_SUPPORTED,
           "different flags require full verified configuration dispatch, not silent inheritance");
    auto compatible = immutable.create_member({{3, 100}, 303}, 0);
    expect(compatible.status == CUDA_SUCCESS &&
               compatible.member.view().logical_flags == CU_CTX_MAP_HOST,
           "equivalent flags preserve the immutable-domain path");
    compatible.member = {};
    expect(immutable.close() == CUDA_SUCCESS && immutable.shutdown() == CUDA_SUCCESS,
           "rejected cold flags have no retained native resource");
}

void absent_initial_override_preserves_observed_baseline() {
    native = {};
    // Deliberate SDK seam normalization, not a claim about any installed Driver.
    native.should_normalize_creation_flags = true;
    CudaExecutionDomainOwner owner(dispatch(), {99, 0, CU_CTX_MAP_HOST, 1, 4096});
    expect(owner.initialize() == CUDA_SUCCESS, "observe an SDK-normalized creation baseline");
    auto member = owner.create_member({{1, 99}, 101});
    expect(member.status == CUDA_SUCCESS && member.member.attach({{1, 99}, 101}) == CUDA_SUCCESS,
           "attach the unchanged legacy member");
    expect(member.member.view().logical_flags == CU_CTX_MAP_HOST &&
               member.member.configuration().status == CUDA_SUCCESS &&
               member.member.configuration().values[0] ==
                   (CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX),
           "absent override retains the observed profile and old logical initialization contract");
    native.current = native.context();
    expect(member.member.materialize_configuration() == CUDA_SUCCESS && !native.setters &&
               native.flags == (CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX),
           "legacy activation cannot replace SDK-normalized flags with requested creation bytes");
    expect(member.member.retire() == CUDA_SUCCESS, "retire the observed baseline member");
    member.member = {};
    native.current = nullptr;
    expect(owner.close() == CUDA_SUCCESS && owner.shutdown() == CUDA_SUCCESS &&
               native.creates == 1 && native.destroys == 1,
           "normalized baseline retains exact one-owner cleanup");
}
}  // namespace

int main() {
    try {
        independent_normalized_profiles_and_epochs();
        held_physical_work_and_equal_profiles();
        native_error_and_reentrant_lifetimes();
        owner_validation_and_fresh_members();
        partial_dispatch_rejects();
        cold_requested_flags_preserve_independent_profiles();
        absent_initial_override_preserves_observed_baseline();
        for (int scenario = 0; scenario < 10; ++scenario) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated unknown-native configuration control");
            if (child == 0) {
                try {
                    retained_unknown(scenario);
                    _exit(0);  // Intentional quarantined native owner, not leak-free recovery.
                } catch (...) {
                    _exit(1);
                }
            }
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "isolated capture/restore/proof/exception failure retains without retry");
        }
        std::cout << "Shared configuration controls passed: healthy_groups=7; retained_faults=10\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
