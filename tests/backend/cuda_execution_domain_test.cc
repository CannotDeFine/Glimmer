#include "internal/cuda_execution_domain.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
using namespace glimmer::backend::cuda;

void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

enum class Call : std::size_t { kInitialize, kDevice, kCreate, kSet, kGet, kDestroy, kCount };

struct Context {
    bool is_alive{};
};

struct Native {
    Context physical, unrelated{true};
    CUcontext current{};
    std::vector<Call> calls;
    std::array<unsigned int, static_cast<std::size_t>(Call::kCount)> counts{};
    std::function<void(Call)> callback;
    Call failed_call{Call::kCount}, changed_call{Call::kCount};
    unsigned int failed_occurrence{1};
    CUresult failure{CUDA_ERROR_OUT_OF_MEMORY};
    bool has_partial_create{}, has_null_create{}, has_argument_error{};
    bool has_wrong_create_current{}, has_ineffective_detach{};
    unsigned int creates{}, destroys{};

    CUcontext physical_context() noexcept {
        return reinterpret_cast<CUcontext>(&physical);
    }
    CUcontext unrelated_context() noexcept {
        return reinterpret_cast<CUcontext>(&unrelated);
    }
    CUresult visit(Call call) {
        calls.push_back(call);
        const auto occurrence = ++counts[static_cast<std::size_t>(call)];
        if (callback) callback(call);
        if (call == changed_call) current = unrelated_context();
        return call == failed_call && occurrence == failed_occurrence ? failure : CUDA_SUCCESS;
    }
}* native;

CUresult CUDAAPI initialize(unsigned int flags) {
    if (flags != 0) native->has_argument_error = true;
    return native->visit(Call::kInitialize);
}

CUresult CUDAAPI device_get(CUdevice* output, int ordinal) {
    const auto status = native->visit(Call::kDevice);
    if (!output || ordinal != 3) {
        native->has_argument_error = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (status == CUDA_SUCCESS) *output = 41;
    return status;
}

CUresult CUDAAPI context_create(CUcontext* output, unsigned int flags, CUdevice device) {
    const auto status = native->visit(Call::kCreate);
    if (!output || flags != CU_CTX_MAP_HOST || device != 41 || native->physical.is_alive) {
        native->has_argument_error = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    ++native->creates;
    if ((status == CUDA_SUCCESS || native->has_partial_create) && !native->has_null_create) {
        native->physical.is_alive = true;
        *output = native->physical_context();
        native->current = native->has_wrong_create_current ? native->unrelated_context() : *output;
    }
    return status;
}

CUresult CUDAAPI context_set_current(CUcontext context) {
    const auto status = native->visit(Call::kSet);
    if (context != nullptr) native->has_argument_error = true;
    if (status == CUDA_SUCCESS && !native->has_ineffective_detach &&
        native->changed_call != Call::kSet)
        native->current = context;
    return status;
}

CUresult CUDAAPI context_get_current(CUcontext* output) {
    const auto status = native->visit(Call::kGet);
    if (!output) {
        native->has_argument_error = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (status == CUDA_SUCCESS) *output = native->current;
    return status;
}

CUresult CUDAAPI context_destroy(CUcontext context) {
    if (context != native->physical_context() || !native->physical.is_alive ||
        native->current != nullptr || native->destroys != 0) {
        native->has_argument_error = true;
        return CUDA_ERROR_INVALID_VALUE;
    }
    const auto status = native->visit(Call::kDestroy);
    ++native->destroys;
    if (status == CUDA_SUCCESS) native->physical.is_alive = false;
    return status;
}

CudaExecutionDomainDispatch dispatch() {
    return {initialize,          device_get,          context_create,
            context_set_current, context_get_current, context_destroy};
}

CudaExecutionDomainConfig config() {
    return {701, 3, CU_CTX_MAP_HOST, 2, 128};
}

CudaExecutionDomainLogicalIdentity logical(std::uint64_t client = 11) {
    return {{client, 71, 1}, client + 100};
}

bool same_logical(CudaExecutionDomainLogicalIdentity first,
                  CudaExecutionDomainLogicalIdentity second) {
    return first.session.client_id == second.session.client_id &&
           first.session.server_incarnation == second.session.server_incarnation &&
           first.session.first_sequence == second.session.first_sequence &&
           first.context == second.context;
}

struct Fixture {
    Native state;
    std::unique_ptr<CudaExecutionDomainOwner> domain;
    explicit Fixture(CudaExecutionDomainConfig options = config()) {
        native = &state;
        domain = std::make_unique<CudaExecutionDomainOwner>(dispatch(), options);
    }
    void start() {
        expect(domain->initialize() == CUDA_SUCCESS,
               "initialize the one physical execution domain");
        expect(state.current == nullptr && state.physical.is_alive && state.creates == 1 &&
                   state.destroys == 0 && !state.has_argument_error,
               "actual create and independent NULL restoration precede member publication");
        expect(state.calls == std::vector<Call>{Call::kInitialize, Call::kGet, Call::kDevice,
                                                Call::kGet, Call::kCreate, Call::kGet, Call::kSet,
                                                Call::kGet},
               "each context-changing initialization boundary has its own actual current proof");
    }
    CudaExecutionDomainMember member(CudaExecutionDomainLogicalIdentity identity = logical()) {
        auto result = domain->create_member(identity);
        expect(result.status == CUDA_SUCCESS && result.member,
               "bounded logical member receives an opaque trusted capability");
        return std::move(result.member);
    }
    void finish() {
        expect(domain->close() == CUDA_SUCCESS && domain->shutdown() == CUDA_SUCCESS,
               "explicitly destroy only a closed, independently retired physical domain");
        expect(state.destroys == 1 && !state.physical.is_alive && state.current == nullptr &&
                   !state.has_argument_error,
               "one exact original native destroy completes normal ownership");
    }
};

template <typename Function>
void isolated(Function function) {
    const auto child = fork();
    expect(child >= 0, "fork an intentionally retained unsafe-domain control");
    if (child == 0) {
        try {
            function();
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << "Isolated execution domain: " << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated retained-domain assertion completes without signaling or hidden recovery");
}

void cold_constructor_and_configuration() {
    Native state;
    native = &state;
    {
        CudaExecutionDomainOwner cold(dispatch(), config());
        const auto snapshot = cold.snapshot();
        expect(snapshot.status == CUDA_SUCCESS && !snapshot.has_context && !snapshot.is_closed &&
                   !snapshot.is_shutdown && !snapshot.is_quarantined && state.calls.empty(),
               "construction and cold snapshots perform no SDK work");
        expect(cold.create_member(logical()).status == CUDA_ERROR_NOT_INITIALIZED &&
                   state.calls.empty(),
               "cold owner cannot manufacture a native member");
    }
    expect(state.calls.empty(), "healthy uninitialized destruction has no SDK side effects");
    const auto rejects = [&](CudaExecutionDomainDispatch callbacks,
                             CudaExecutionDomainConfig options) {
        bool rejected{};
        try {
            CudaExecutionDomainOwner invalid(callbacks, options);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        expect(rejected && state.calls.empty(),
               "invalid cold contract rejects before any SDK callback");
    };
    for (unsigned int missing = 0; missing != 6; ++missing) {
        auto callbacks = dispatch();
        if (missing == 0) callbacks.initialize = nullptr;
        if (missing == 1) callbacks.device_get = nullptr;
        if (missing == 2) callbacks.context_create = nullptr;
        if (missing == 3) callbacks.context_set_current = nullptr;
        if (missing == 4) callbacks.context_get_current = nullptr;
        if (missing == 5) callbacks.context_destroy = nullptr;
        rejects(callbacks, config());
    }
    for (unsigned int field = 0; field != 5; ++field) {
        auto options = config();
        if (field == 0) options.identity = 0;
        if (field == 1) options.device_ordinal = -1;
        if (field == 2) options.flags = UINT32_MAX;
        if (field == 3) options.maximum_members = 0;
        if (field == 4) options.maximum_device_bytes = 0;
        rejects(dispatch(), options);
    }
    auto options = config();
    options.maximum_members = std::numeric_limits<std::size_t>::max();
    bool bounded{};
    try {
        CudaExecutionDomainOwner impossible(dispatch(), options);
    } catch (const std::exception&) {
        bounded = true;
    }
    expect(bounded && state.calls.empty(),
           "unrepresentable member capacity cannot publish an owner");
}

void exact_members_and_explicit_lifecycle() {
    Fixture fixture;
    fixture.start();
    const auto calls = fixture.state.calls.size();
    expect(fixture.domain->initialize() == CUDA_ERROR_INVALID_VALUE &&
               fixture.domain->shutdown() == CUDA_ERROR_NOT_READY &&
               fixture.state.calls.size() == calls,
           "initialization is once-only and shutdown never implicitly closes admission");
    auto first = fixture.member();
    auto second = fixture.member(logical(12));
    auto first_view = first.view();
    auto second_view = second.view();
    expect(first_view.status == CUDA_SUCCESS && second_view.status == CUDA_SUCCESS &&
               first_view.context.native == fixture.state.physical_context() &&
               second_view.context.native == first_view.context.native &&
               first_view.context.id == config().identity &&
               second_view.context.id == config().identity && first_view.context.lifetime &&
               second_view.context.lifetime && same_logical(first_view.logical, logical()) &&
               same_logical(second_view.logical, logical(12)) &&
               first_view.logical.context != second_view.logical.context &&
               first_view.generation != 0 && second_view.generation > first_view.generation &&
               first_view.device_ordinal == 3 && first_view.native_device == 41 &&
               first_view.flags == CU_CTX_MAP_HOST && !first_view.is_attached &&
               !second_view.is_attached,
           "native identity is common while logical identity and nonzero generation are exact");
    expect(first.owner_activity() && first.owner_activity() == second.owner_activity(),
           "members share exclusion and fault state, not logical resource identity");
    for (unsigned int field = 0; field != 4; ++field) {
        auto wrong = logical();
        if (field == 0) ++wrong.session.client_id;
        if (field == 1) ++wrong.session.server_incarnation;
        if (field == 2) ++wrong.session.first_sequence;
        if (field == 3) ++wrong.context;
        expect(first.attach(wrong) == CUDA_ERROR_INVALID_VALUE && !first.view().is_attached,
               "each independent logical identity field authenticates singleton attachment");
    }
    expect(first.attach(logical()) == CUDA_SUCCESS && second.attach(logical(12)) == CUDA_SUCCESS &&
               first.attach(logical()) == CUDA_ERROR_INVALID_VALUE && first.view().is_attached &&
               fixture.domain->snapshot().attached_members == 2 &&
               fixture.state.calls.size() == calls,
           "both exact members attach once without additional native context creation");
    expect(fixture.domain->close() == CUDA_SUCCESS && fixture.domain->close() == CUDA_SUCCESS &&
               fixture.domain->initialize() == CUDA_ERROR_DEINITIALIZED &&
               fixture.domain->create_member(logical(13)).status == CUDA_ERROR_DEINITIALIZED &&
               fixture.domain->shutdown() == CUDA_ERROR_NOT_READY,
           "close seals admission but neither retires attached members nor destroys context");
    expect(first.view().status == CUDA_SUCCESS && first.retire() == CUDA_SUCCESS &&
               first.retire() == CUDA_ERROR_DEINITIALIZED &&
               first.attach(logical()) == CUDA_ERROR_DEINITIALIZED &&
               first.view().status == CUDA_ERROR_DEINITIALIZED &&
               second.view().status == CUDA_SUCCESS &&
               fixture.domain->snapshot().attached_members == 1,
           "retiring one member invalidates that generation only");
    expect(second.retire() == CUDA_SUCCESS, "close preserves exact healthy member cleanup");
    first = {};
    second = {};
    expect(fixture.domain->shutdown() == CUDA_ERROR_NOT_READY && fixture.state.destroys == 0,
           "exported context lifetimes pin physical ownership after capability retirement");
    first_view = {};
    expect(fixture.domain->shutdown() == CUDA_ERROR_NOT_READY,
           "the second independent exported lifetime remains a real owner");
    second_view = {};
    fixture.finish();
    const auto destroyed_calls = fixture.state.calls.size();
    expect(fixture.domain->snapshot().is_shutdown && !fixture.domain->snapshot().has_context &&
               fixture.domain->snapshot().retained_members == 0 &&
               fixture.domain->shutdown() == CUDA_SUCCESS &&
               fixture.domain->close() == CUDA_SUCCESS &&
               fixture.state.calls.size() == destroyed_calls,
           "successful terminal shutdown is idempotent without native destroy replay");
}

void identity_capacity_and_generation() {
    Fixture fixture;
    fixture.start();
    const auto calls = fixture.state.calls.size();
    for (unsigned int field = 0; field != 4; ++field) {
        auto invalid = logical();
        if (field == 0) invalid.session.client_id = 0;
        if (field == 1) invalid.session.server_incarnation = 0;
        if (field == 2) invalid.session.first_sequence = 0;
        if (field == 3) invalid.context = 0;
        expect(fixture.domain->create_member(invalid).status == CUDA_ERROR_INVALID_VALUE &&
                   fixture.domain->snapshot().retained_members == 0,
               "invalid logical fields consume no member capacity");
    }
    auto first = fixture.member();
    const auto original_generation = first.view().generation;
    expect(fixture.domain->create_member(logical()).status == CUDA_ERROR_INVALID_VALUE,
           "retained logical identity cannot be duplicated");
    auto second = fixture.member(logical(12));
    expect(fixture.domain->create_member(logical(13)).status == CUDA_ERROR_OUT_OF_MEMORY &&
               fixture.domain->snapshot().retained_members == 2,
           "unattached exported capabilities count toward the bounded member pool");
    second = {};
    auto third = fixture.member(logical(13));
    expect(third.view().generation > original_generation &&
               fixture.domain->snapshot().retained_members == 2,
           "dropping unused uncharged capability permits bounded slot reuse, not ID reuse");
    expect(first.attach(logical()) == CUDA_SUCCESS && first.retire() == CUDA_SUCCESS,
           "prove old generation retirement before testing stale aliases");
    auto stale = first;
    first = {};
    expect(fixture.domain->create_member(logical()).status == CUDA_ERROR_INVALID_VALUE &&
               stale.reserve_device_bytes(1) == CUDA_ERROR_DEINITIALIZED &&
               stale.release_device_bytes(1) == CUDA_ERROR_DEINITIALIZED &&
               stale.set_physical_reservation(false) == CUDA_ERROR_DEINITIALIZED,
           "retired but exported generation neither authorizes reuse nor mutates accounting");
    stale = {};
    auto replacement = fixture.member();
    expect(replacement.view().generation > original_generation && !replacement.view().is_attached &&
               replacement.attach(logical()) == CUDA_SUCCESS &&
               replacement.retire() == CUDA_SUCCESS && fixture.state.calls.size() == calls,
           "fully released logical identity may return only with a fresh generation");
    replacement = {};
    third = {};
    fixture.finish();
}

void aggregate_bytes_and_refunds() {
    Fixture fixture;
    fixture.start();
    auto first = fixture.member();
    auto second = fixture.member(logical(12));
    expect(first.reserve_device_bytes(1) == CUDA_ERROR_INVALID_VALUE &&
               first.release_device_bytes(1) == CUDA_ERROR_INVALID_VALUE,
           "unattached capability cannot charge a Session resource");
    expect(first.attach(logical()) == CUDA_SUCCESS && second.attach(logical(12)) == CUDA_SUCCESS,
           "independent members own independently charged allocations");
    const auto calls = fixture.state.calls.size();
    expect(first.reserve_device_bytes(0) == CUDA_ERROR_INVALID_VALUE &&
               first.release_device_bytes(0) == CUDA_ERROR_INVALID_VALUE &&
               first.reserve_device_bytes(std::numeric_limits<std::size_t>::max()) ==
                   CUDA_ERROR_OUT_OF_MEMORY &&
               fixture.domain->snapshot().reserved_device_bytes == 0,
           "zero and overflow-sized requests reject without an arithmetic wrap or mutation");
    expect(first.reserve_device_bytes(80) == CUDA_SUCCESS &&
               second.reserve_device_bytes(48) == CUDA_SUCCESS &&
               fixture.domain->snapshot().reserved_device_bytes == 128 &&
               first.reserve_device_bytes(1) == CUDA_ERROR_OUT_OF_MEMORY &&
               second.reserve_device_bytes(1) == CUDA_ERROR_OUT_OF_MEMORY,
           "joint exact-boundary resources consume the physical domain quota");
    expect(first.release_device_bytes(81) == CUDA_ERROR_INVALID_VALUE &&
               second.release_device_bytes(49) == CUDA_ERROR_INVALID_VALUE &&
               fixture.domain->snapshot().reserved_device_bytes == 128 &&
               first.retire() == CUDA_ERROR_NOT_READY,
           "one member cannot refund the other's bytes or retire a charged allocation");
    expect(second.release_device_bytes(48) == CUDA_SUCCESS &&
               second.release_device_bytes(1) == CUDA_ERROR_INVALID_VALUE &&
               first.reserve_device_bytes(48) == CUDA_SUCCESS &&
               fixture.domain->snapshot().reserved_device_bytes == 128,
           "only trusted exact cleanup refunds aggregate capacity once");
    expect(first.release_device_bytes(128) == CUDA_SUCCESS &&
               fixture.domain->snapshot().reserved_device_bytes == 0 &&
               first.retire() == CUDA_SUCCESS && second.retire() == CUDA_SUCCESS &&
               fixture.state.calls.size() == calls,
           "quota bookkeeping is CPU-only and independent from native allocation cleanup");
    first = {};
    second = {};
    fixture.finish();
}

void physical_reservation_and_callback_exclusion() {
    Fixture fixture;
    fixture.start();
    auto first = fixture.member();
    auto second = fixture.member(logical(12));
    expect(first.set_physical_reservation(true) == CUDA_ERROR_INVALID_VALUE &&
               first.attach(logical()) == CUDA_SUCCESS &&
               second.attach(logical(12)) == CUDA_SUCCESS,
           "only attached logical Sessions publish trusted physical reservations");
    expect(
        first.set_physical_reservation(true) == CUDA_SUCCESS &&
            first.other_members_quiescent() == CUDA_SUCCESS &&
            second.other_members_quiescent() == CUDA_ERROR_NOT_READY &&
            first.retire() == CUDA_ERROR_NOT_READY,
        "self reservation is excluded from other-member check but still prevents self retirement");
    expect(second.set_physical_reservation(true) == CUDA_SUCCESS &&
               first.other_members_quiescent() == CUDA_ERROR_NOT_READY &&
               first.set_physical_reservation(false) == CUDA_SUCCESS &&
               second.other_members_quiescent() == CUDA_SUCCESS,
           "distinct members share physical exclusion without treating host return as stop");
    auto alias = second;
    second = {};
    expect(first.other_members_quiescent() == CUDA_ERROR_NOT_READY &&
               fixture.domain->snapshot().attached_members == 2,
           "dropping a capability does not clear its still-owned physical reservation");
    expect(alias.set_physical_reservation(false) == CUDA_SUCCESS,
           "trusted independent physical proof explicitly clears the retained reservation");
    auto activity = first.owner_activity();
    for (unsigned int field = 0; field != 4; ++field) {
        bool* flag = &activity->is_native_issue_active;
        if (field == 1) flag = &activity->is_folded_active;
        if (field == 2) flag = &activity->is_rnn_query_active;
        if (field == 3) flag = &activity->is_preparation_active;
        *flag = true;
        const auto calls = fixture.state.calls.size();
        expect(
            first.view().status == CUDA_ERROR_NOT_READY &&
                alias.view().status == CUDA_ERROR_NOT_READY &&
                first.view(true).status == CUDA_SUCCESS && first.retire() == CUDA_ERROR_NOT_READY &&
                alias.attach(logical(12)) == CUDA_ERROR_NOT_READY &&
                fixture.domain->initialize() == CUDA_ERROR_NOT_READY &&
                fixture.domain->create_member(logical(13)).status == CUDA_ERROR_NOT_READY &&
                fixture.domain->close() == CUDA_ERROR_NOT_READY &&
                fixture.domain->shutdown() == CUDA_ERROR_NOT_READY &&
                fixture.domain->snapshot().status == CUDA_SUCCESS,
            "all shared callback scopes block lifecycle reentry while CPU snapshots remain safe");
        expect(
            first.reserve_device_bytes(1) == CUDA_SUCCESS &&
                first.release_device_bytes(1) == CUDA_SUCCESS &&
                first.set_physical_reservation(true) == CUDA_SUCCESS &&
                alias.other_members_quiescent() == CUDA_ERROR_NOT_READY &&
                first.set_physical_reservation(false) == CUDA_SUCCESS &&
                fixture.state.calls.size() == calls,
            "trusted metadata updates inside owned native scope neither call SDK nor escape quota");
        *flag = false;
    }
    expect(first.retire() == CUDA_SUCCESS && alias.retire() == CUDA_SUCCESS,
           "only cleared physical flags and healthy callback exit permit retirement");
    first = {};
    alias = {};
    activity.reset();
    fixture.finish();
}

void native_callback_reentry() {
    Fixture fixture;
    unsigned int callbacks{};
    fixture.state.callback = [&](Call) {
        ++callbacks;
        const auto calls = fixture.state.calls.size();
        expect(
            fixture.domain->initialize() == CUDA_ERROR_NOT_READY &&
                fixture.domain->create_member(logical()).status == CUDA_ERROR_NOT_READY &&
                fixture.domain->close() == CUDA_ERROR_NOT_READY &&
                fixture.domain->shutdown() == CUDA_ERROR_NOT_READY &&
                fixture.domain->snapshot().status == CUDA_SUCCESS &&
                fixture.state.calls.size() == calls,
            "exclusion precedes every real callback including initialization and native destroy");
    };
    fixture.start();
    const auto initial_callbacks = callbacks;
    expect(initial_callbacks != 0, "native callback reentry controls actually executed");
    fixture.finish();
    expect(callbacks > initial_callbacks && callbacks == fixture.state.calls.size(),
           "cleanup callbacks have the same independent owner exclusion as creation");
}

void owner_thread_process_and_empty_capabilities() {
    CudaExecutionDomainMember empty;
    expect(!empty && empty.view().status == CUDA_ERROR_INVALID_HANDLE && !empty.owner_activity() &&
               empty.attach(logical()) == CUDA_ERROR_INVALID_HANDLE &&
               empty.reserve_device_bytes(1) == CUDA_ERROR_INVALID_HANDLE &&
               empty.release_device_bytes(1) == CUDA_ERROR_INVALID_HANDLE &&
               empty.set_physical_reservation(true) == CUDA_ERROR_INVALID_HANDLE &&
               empty.other_members_quiescent() == CUDA_ERROR_INVALID_HANDLE &&
               empty.retire() == CUDA_ERROR_INVALID_HANDLE,
           "empty capabilities never represent an owner or logical Session");
    empty.retain_uncertain_device_bytes(1);
    empty.quarantine();
    Fixture fixture;
    fixture.start();
    auto member = fixture.member();
    expect(
        member.attach(logical()) == CUDA_SUCCESS && member.reserve_device_bytes(16) == CUDA_SUCCESS,
        "owner retains a charged live member across foreign access controls");
    const auto calls = fixture.state.calls.size();
    const auto foreign = [&] {
        expect(
            fixture.domain->snapshot().status == CUDA_ERROR_INVALID_CONTEXT &&
                fixture.domain->initialize() == CUDA_ERROR_INVALID_CONTEXT &&
                fixture.domain->create_member(logical(12)).status == CUDA_ERROR_INVALID_CONTEXT &&
                fixture.domain->close() == CUDA_ERROR_INVALID_CONTEXT &&
                fixture.domain->shutdown() == CUDA_ERROR_INVALID_CONTEXT &&
                member.view().status == CUDA_ERROR_INVALID_CONTEXT &&
                member.view(true).status == CUDA_ERROR_INVALID_CONTEXT &&
                !member.owner_activity() &&
                member.attach(logical()) == CUDA_ERROR_INVALID_CONTEXT &&
                member.reserve_device_bytes(1) == CUDA_ERROR_INVALID_CONTEXT &&
                member.release_device_bytes(1) == CUDA_ERROR_INVALID_CONTEXT &&
                member.set_physical_reservation(false) == CUDA_ERROR_INVALID_CONTEXT &&
                member.other_members_quiescent() == CUDA_ERROR_INVALID_CONTEXT &&
                member.retire() == CUDA_ERROR_INVALID_CONTEXT,
            "foreign thread or fork cannot borrow owner identity even with exact capability bits");
        member.retain_uncertain_device_bytes(1000);
        member.quarantine();
    };
    std::async(std::launch::async, foreign).get();
    isolated(foreign);
    expect(fixture.state.calls.size() == calls &&
               fixture.domain->snapshot().reserved_device_bytes == 16 &&
               !fixture.domain->snapshot().is_quarantined && member.view().status == CUDA_SUCCESS,
           "foreign metadata calls cannot mutate quota, quarantine or native callback ledger");
    expect(member.release_device_bytes(16) == CUDA_SUCCESS && member.retire() == CUDA_SUCCESS,
           "original owner remains healthy after denied foreign access");
    member = {};
    fixture.finish();
}

void initial_context_and_clean_creation_failures() {
    {
        Fixture fixture;
        fixture.state.current = fixture.state.unrelated_context();
        expect(fixture.domain->initialize() == CUDA_ERROR_INVALID_CONTEXT &&
                   fixture.state.creates == 0 && fixture.state.destroys == 0 &&
                   fixture.state.current == fixture.state.unrelated_context() &&
                   !fixture.domain->snapshot().is_quarantined,
               "non-NULL entry rejects without stealing or restoring an unrelated native context");
        const auto calls = fixture.state.calls.size();
        fixture.state.current = nullptr;
        expect(fixture.domain->initialize() == CUDA_ERROR_INVALID_VALUE &&
                   fixture.domain->close() == CUDA_SUCCESS &&
                   fixture.domain->shutdown() == CUDA_SUCCESS &&
                   fixture.state.calls.size() == calls,
               "a clean failed initialization remains once-only and has no context to destroy");
    }
    const std::array clean_errors{CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_INVALID_VALUE,
                                  CUDA_ERROR_INVALID_DEVICE, CUDA_ERROR_NOT_SUPPORTED,
                                  CUDA_ERROR_NOT_PERMITTED};
    for (const auto error : clean_errors) {
        Fixture fixture;
        fixture.state.failed_call = Call::kCreate;
        fixture.state.failure = error;
        expect(fixture.domain->initialize() == error && fixture.state.creates == 1 &&
                   fixture.state.current == nullptr && !fixture.state.physical.is_alive &&
                   !fixture.domain->snapshot().has_context &&
                   !fixture.domain->snapshot().is_quarantined &&
                   fixture.domain->create_member(logical()).status == CUDA_ERROR_NOT_INITIALIZED,
               "whitelisted failed creation with independently proven NULL has no native owner");
        const auto calls = fixture.state.calls.size();
        expect(
            fixture.domain->initialize() == CUDA_ERROR_INVALID_VALUE &&
                fixture.domain->close() == CUDA_SUCCESS &&
                fixture.domain->shutdown() == CUDA_SUCCESS && fixture.state.destroys == 0 &&
                fixture.state.calls.size() == calls,
            "clean creation failure preserves its actual status without retry or invented destroy");
    }
    for (const auto failed : {Call::kInitialize, Call::kDevice}) {
        Fixture fixture;
        fixture.state.failed_call = failed;
        fixture.state.failure = CUDA_ERROR_INVALID_DEVICE;
        expect(fixture.domain->initialize() == CUDA_ERROR_INVALID_DEVICE &&
                   fixture.state.creates == 0 && !fixture.domain->snapshot().is_quarantined,
               "pre-creation original failure preserves status and performs no context allocation");
        const auto calls = fixture.state.calls.size();
        expect(fixture.domain->close() == CUDA_SUCCESS &&
                   fixture.domain->shutdown() == CUDA_SUCCESS &&
                   fixture.state.calls.size() == calls && fixture.state.destroys == 0,
               "known no-context prerequisite failure needs no native cleanup");
    }
}

void shutdown_requires_independent_null_entry() {
    Fixture fixture;
    fixture.start();
    expect(fixture.domain->close() == CUDA_SUCCESS, "seal a resource-free physical domain");
    fixture.state.current = fixture.state.unrelated_context();
    expect(fixture.domain->shutdown() == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.state.destroys == 0 &&
               fixture.state.current == fixture.state.unrelated_context() &&
               !fixture.domain->snapshot().is_quarantined,
           "wrong shutdown entry is a retryable owner activation error, not a destroy attempt");
    fixture.state.current = nullptr;
    fixture.finish();
}

enum class InitFailure {
    kInitialProof,
    kDeviceProof,
    kCreateProof,
    kRestoreProof,
    kDeviceContext,
    kPartialCreate,
    kNullSuccess,
    kWrongCreateContext,
    kDetachFailure,
    kDetachNoEffect,
    kUnknownCreateError,
    kCreateThrow
};

void uncertain_initialization_retains_without_replay() {
    const std::array failures{
        InitFailure::kInitialProof,   InitFailure::kDeviceProof,        InitFailure::kCreateProof,
        InitFailure::kRestoreProof,   InitFailure::kDeviceContext,      InitFailure::kPartialCreate,
        InitFailure::kNullSuccess,    InitFailure::kWrongCreateContext, InitFailure::kDetachFailure,
        InitFailure::kDetachNoEffect, InitFailure::kUnknownCreateError, InitFailure::kCreateThrow};
    for (const auto failure : failures)
        isolated([failure] {
            Fixture fixture;
            if (failure <= InitFailure::kRestoreProof) {
                fixture.state.failed_call = Call::kGet;
                fixture.state.failed_occurrence = static_cast<unsigned int>(failure) + 1;
                fixture.state.failure = CUDA_ERROR_INVALID_CONTEXT;
            }
            if (failure == InitFailure::kDeviceContext) fixture.state.changed_call = Call::kDevice;
            if (failure == InitFailure::kPartialCreate) {
                fixture.state.failed_call = Call::kCreate;
                fixture.state.has_partial_create = true;
            }
            if (failure == InitFailure::kNullSuccess) fixture.state.has_null_create = true;
            if (failure == InitFailure::kWrongCreateContext)
                fixture.state.has_wrong_create_current = true;
            if (failure == InitFailure::kDetachFailure) {
                fixture.state.failed_call = Call::kSet;
                fixture.state.failure = CUDA_ERROR_INVALID_VALUE;
            }
            if (failure == InitFailure::kDetachNoEffect)
                fixture.state.has_ineffective_detach = true;
            if (failure == InitFailure::kUnknownCreateError) {
                fixture.state.failed_call = Call::kCreate;
                fixture.state.failure = CUDA_ERROR_UNKNOWN;
            }
            if (failure == InitFailure::kCreateThrow)
                fixture.state.callback = [](Call call) {
                    if (call == Call::kCreate)
                        throw std::runtime_error("native create callback exception");
                };
            const auto status = fixture.domain->initialize();
            expect(status != CUDA_SUCCESS && fixture.domain->snapshot().is_quarantined &&
                       fixture.domain->snapshot().status == status && fixture.state.destroys == 0 &&
                       !fixture.state.has_argument_error,
                   "partial outputs, missing proof or unknown entry quarantine without guessing "
                   "cleanup");
            const auto calls = fixture.state.calls.size();
            expect(fixture.domain->initialize() == status &&
                       fixture.domain->create_member(logical()).status == status &&
                       fixture.domain->close() == status && fixture.domain->shutdown() == status &&
                       fixture.state.calls.size() == calls && fixture.state.destroys == 0,
                   "uncertain initialization cannot retry SDK or publish a usable logical member");
            fixture.domain.reset();
            expect(fixture.state.calls.size() == calls && fixture.state.destroys == 0,
                   "uncertain destructor is CPU-only and cannot manufacture stop proof");
        });
}

enum class DestroyFailure { kBeforeProof, kNativeFailure, kAfterProof, kChangedContext, kThrow };

void uncertain_shutdown_retains_without_replay() {
    const std::array failures{DestroyFailure::kBeforeProof, DestroyFailure::kNativeFailure,
                              DestroyFailure::kAfterProof, DestroyFailure::kChangedContext,
                              DestroyFailure::kThrow};
    for (const auto failure : failures)
        isolated([failure] {
            Fixture fixture;
            fixture.start();
            expect(fixture.domain->close() == CUDA_SUCCESS,
                   "seal before explicit faulted shutdown");
            if (failure == DestroyFailure::kBeforeProof || failure == DestroyFailure::kAfterProof) {
                fixture.state.failed_call = Call::kGet;
                fixture.state.failed_occurrence =
                    fixture.state.counts[static_cast<std::size_t>(Call::kGet)] +
                    (failure == DestroyFailure::kBeforeProof ? 1U : 2U);
                fixture.state.failure = CUDA_ERROR_INVALID_CONTEXT;
            }
            if (failure == DestroyFailure::kNativeFailure)
                fixture.state.failed_call = Call::kDestroy;
            if (failure == DestroyFailure::kChangedContext)
                fixture.state.changed_call = Call::kDestroy;
            if (failure == DestroyFailure::kThrow)
                fixture.state.callback = [](Call call) {
                    if (call == Call::kDestroy)
                        throw std::runtime_error("native destroy callback exception");
                };
            const auto status = fixture.domain->shutdown();
            expect(status != CUDA_SUCCESS && fixture.domain->snapshot().is_quarantined &&
                       !fixture.domain->snapshot().is_shutdown &&
                       fixture.domain->snapshot().status == status &&
                       fixture.state.counts[static_cast<std::size_t>(Call::kDestroy)] ==
                           (failure == DestroyFailure::kBeforeProof ? 0U : 1U) &&
                       !fixture.state.has_argument_error,
                   "destroy entry and independent post-proof are separate from actual successful "
                   "teardown");
            const auto calls = fixture.state.calls.size();
            expect(fixture.domain->shutdown() == status && fixture.domain->close() == status &&
                       fixture.state.calls.size() == calls,
                   "failed or unproven destroy remains permanently retained without native retry");
            fixture.domain.reset();
            expect(fixture.state.calls.size() == calls,
                   "destructor does not retry a destroy whose actual effects are unknown");
        });
}

void shared_uncertainty_never_refunds_or_clears_physical_work() {
    for (unsigned int kind = 0; kind != 3; ++kind)
        isolated([kind] {
            Fixture fixture;
            fixture.start();
            auto first = fixture.member();
            auto second = fixture.member(logical(12));
            expect(first.attach(logical()) == CUDA_SUCCESS &&
                       second.attach(logical(12)) == CUDA_SUCCESS &&
                       first.reserve_device_bytes(64) == CUDA_SUCCESS &&
                       second.reserve_device_bytes(32) == CUDA_SUCCESS &&
                       first.set_physical_reservation(true) == CUDA_SUCCESS,
                   "two independent resource owners precede shared uncertainty");
            const auto activity = second.owner_activity();
            const auto calls = fixture.state.calls.size();
            if (kind == 0) first.quarantine();
            if (kind == 1) first.retain_uncertain_device_bytes(64);
            if (kind == 2)
                first.retain_uncertain_device_bytes(std::numeric_limits<std::size_t>::max());
            const auto expected_bytes = kind == 0   ? std::size_t{96}
                                        : kind == 1 ? std::size_t{160}
                                                    : std::numeric_limits<std::size_t>::max();
            expect(fixture.domain->snapshot().is_quarantined && activity->is_quarantined &&
                       fixture.domain->snapshot().reserved_device_bytes == expected_bytes &&
                       first.view().status == CUDA_ERROR_UNKNOWN &&
                       second.view(true).status == CUDA_ERROR_UNKNOWN &&
                       first.release_device_bytes(64) == CUDA_ERROR_UNKNOWN &&
                       second.release_device_bytes(32) == CUDA_ERROR_UNKNOWN &&
                       first.set_physical_reservation(false) == CUDA_ERROR_UNKNOWN &&
                       second.other_members_quiescent() == CUDA_ERROR_UNKNOWN &&
                       first.retire() == CUDA_ERROR_UNKNOWN &&
                       second.retire() == CUDA_ERROR_UNKNOWN &&
                       fixture.domain->close() == CUDA_ERROR_UNKNOWN &&
                       fixture.domain->shutdown() == CUDA_ERROR_UNKNOWN &&
                       fixture.state.calls.size() == calls && fixture.state.destroys == 0,
                   "shared uncertainty retains exact or saturated bytes and forbids invented "
                   "refunds/stop");
            fixture.domain.reset();
            expect(
                fixture.state.calls.size() == calls && second.view().status == CUDA_ERROR_UNKNOWN,
                "exported capability remains safely faulted after the native owner is retained");
        });
}

void abandoned_member_and_owner_lifetimes() {
    isolated([] {
        Fixture fixture;
        fixture.start();
        auto first = fixture.member();
        auto second = fixture.member(logical(12));
        expect(
            first.attach(logical()) == CUDA_SUCCESS && second.attach(logical(12)) == CUDA_SUCCESS &&
                second.reserve_device_bytes(24) == CUDA_SUCCESS &&
                second.set_physical_reservation(true) == CUDA_SUCCESS,
            "attached abandoned member has real bytes and trusted unresolved physical reservation");
        second = {};
        expect(first.other_members_quiescent() == CUDA_ERROR_NOT_READY &&
                   fixture.domain->snapshot().attached_members == 2 &&
                   fixture.domain->snapshot().reserved_device_bytes == 24 &&
                   fixture.domain->create_member(logical(13)).status == CUDA_ERROR_OUT_OF_MEMORY &&
                   first.retire() == CUDA_SUCCESS,
               "dropping every external member alias cannot reclaim live internal ownership");
        first = {};
        expect(fixture.domain->close() == CUDA_SUCCESS &&
                   fixture.domain->shutdown() == CUDA_ERROR_NOT_READY,
               "closed owner cannot destroy context over abandoned member resources");
        const auto calls = fixture.state.calls.size();
        fixture.domain.reset();
        expect(fixture.state.calls.size() == calls && fixture.state.physical.is_alive &&
                   fixture.state.destroys == 0,
               "abandoned resource owner is retained, never cleaned by destructor");
    });
    isolated([] {
        Fixture fixture;
        fixture.start();
        auto member = fixture.member();
        expect(member.attach(logical()) == CUDA_SUCCESS,
               "member capability retains the physical owner independently of its facade");
        auto view = member.view();
        const auto calls = fixture.state.calls.size();
        fixture.domain.reset();
        expect(view.context.lifetime && member.view().status == CUDA_SUCCESS &&
                   member.reserve_device_bytes(8) == CUDA_SUCCESS &&
                   member.release_device_bytes(8) == CUDA_SUCCESS &&
                   member.retire() == CUDA_SUCCESS && fixture.state.calls.size() == calls &&
                   fixture.state.destroys == 0,
               "facade destruction cannot dangle exported member state or issue native cleanup");
        member = {};
        view = {};
        expect(fixture.state.calls.size() == calls && fixture.state.physical.is_alive,
               "last CPU lease release does not manufacture an explicit native shutdown call");
    });
}

void exited_owner_incarnation_cannot_be_recycled() {
    isolated([] {
        Native state;
        native = &state;
        std::unique_ptr<CudaExecutionDomainOwner> domain;
        CudaExecutionDomainMember member;
        std::exception_ptr failure;
        std::thread owner([&] {
            try {
                domain = std::make_unique<CudaExecutionDomainOwner>(dispatch(), config());
                expect(domain->initialize() == CUDA_SUCCESS, "dedicated owner initializes once");
                auto created = domain->create_member(logical());
                expect(created.status == CUDA_SUCCESS, "dedicated owner creates member");
                member = std::move(created.member);
                expect(member.attach(logical()) == CUDA_SUCCESS, "dedicated owner attaches member");
            } catch (...) {
                failure = std::current_exception();
            }
        });
        owner.join();
        if (failure) std::rethrow_exception(failure);
        const auto calls = state.calls.size();
        for (unsigned int attempt = 0; attempt != 4; ++attempt) {
            std::async(std::launch::async, [&] {
                CudaExecutionDomainOwner replacement(dispatch(), config());
                expect(replacement.snapshot().status == CUDA_SUCCESS,
                       "replacement worker has its own live nonrecycled owner incarnation");
                expect(domain->snapshot().status == CUDA_ERROR_INVALID_CONTEXT &&
                           domain->close() == CUDA_ERROR_INVALID_CONTEXT &&
                           domain->shutdown() == CUDA_ERROR_INVALID_CONTEXT &&
                           member.view().status == CUDA_ERROR_INVALID_CONTEXT &&
                           member.retire() == CUDA_ERROR_INVALID_CONTEXT,
                       "new real thread cannot inherit an exited owner's nonrecycled incarnation");
            }).get();
        }
        domain.reset();
        expect(state.calls.size() == calls && state.destroys == 0 && state.physical.is_alive,
               "foreign destruction retains native ownership without unsafe thread impersonation");
    });
}
}  // namespace

int main() {
    try {
        cold_constructor_and_configuration();
        exact_members_and_explicit_lifecycle();
        identity_capacity_and_generation();
        aggregate_bytes_and_refunds();
        physical_reservation_and_callback_exclusion();
        native_callback_reentry();
        owner_thread_process_and_empty_capabilities();
        initial_context_and_clean_creation_failures();
        shutdown_requires_independent_null_entry();
        uncertain_initialization_retains_without_replay();
        uncertain_shutdown_retains_without_replay();
        shared_uncertainty_never_refunds_or_clears_physical_work();
        abandoned_member_and_owner_lifetimes();
        exited_owner_incarnation_cannot_be_recycled();
        std::cout << "CUDA execution domain: 14 groups passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "CUDA execution domain: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
