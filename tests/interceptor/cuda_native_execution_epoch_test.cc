#include "internal/cuda_native_execution_epoch.h"

#include <array>
#include <atomic>
#include <iostream>
#include <stdexcept>

namespace {
using namespace glimmer::interceptor;
using namespace glimmer::control;
int context_storage;
int stream_storage;
const auto kContext = reinterpret_cast<CUcontext>(&context_storage);
const auto kStream = reinterpret_cast<CUstream>(&stream_storage);
constexpr NativeEpochScope kScope{1, 2, 3};
constexpr NativeEpochExtent kExtent{8, 2, 2, 1'000'000};
thread_local CUcontext current_context{};
thread_local std::array<CUcontext, 4> context_stack{};
thread_local unsigned int context_depth{};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Predicate>
void until(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (!predicate()) {
        require(std::chrono::steady_clock::now() < deadline, "bounded owner progress");
        std::this_thread::yield();
    }
}
template <typename Value>
Value result(std::future<Value> future) {
    require(future.valid(), "accepted control request");
    require(future.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
            "bounded control result");
    return future.get();
}
struct Fake {
    struct Event {
        bool is_recorded{};
        bool is_destroyed{};
    };
    std::array<Event, 17> events{};
    std::atomic<unsigned int> creates{0};
    std::atomic<unsigned int> records{0};
    std::atomic<unsigned int> queries{0};
    std::atomic<unsigned int> destroys{0};
    std::atomic<unsigned int> synchronizations{0};
    std::atomic<unsigned int> context_queries{0};
    std::atomic<unsigned int> context_pushes{0};
    std::atomic<int> pushed{0};
    std::atomic<int> native_scopes{0};
    std::atomic<bool> allow_predecessor{true};
    std::atomic<bool> allow_prefixes{true};
    std::atomic<bool> fail_query{false};
    std::atomic<bool> fail_push{false};
    std::atomic<bool> fail_context_query{false};
    std::atomic<bool> has_blocked_record{false};
    std::atomic<bool> allow_record{false};
    unsigned int block_record{};
    unsigned int fail_create{};
    bool has_failed_create_output{};
    unsigned int fail_record{};
    bool has_capture{};
    bool fail_destroy{};
    bool fail_synchronize{};
    bool fail_pop{};
    bool has_wrong_recovery_context{};
    CUstream expected_stream{kStream};
    std::thread::id ptds_caller{};
};
Fake* native{};
CUresult get_context(CUcontext* context) {
    ++native->context_queries;
    if (native->fail_context_query.load()) return CUDA_ERROR_UNKNOWN;
    *context = native->has_wrong_recovery_context && context_depth ? nullptr : current_context;
    return CUDA_SUCCESS;
}
CUresult push_context(CUcontext context) {
    ++native->context_pushes;
    if (native->fail_push.load()) return CUDA_ERROR_INVALID_CONTEXT;
    require(context_depth < context_stack.size(), "bounded fake context stack");
    context_stack[context_depth++] = current_context;
    current_context = context;
    ++native->pushed;
    return CUDA_SUCCESS;
}
CUresult pop_context(CUcontext* context) {
    require(context_depth > 0, "balanced fake context stack");
    *context = current_context;
    current_context = context_stack[--context_depth];
    --native->pushed;
    return native->fail_pop ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
CUresult stream_context(CUstream, CUcontext* context) {
    *context = kContext;
    return CUDA_SUCCESS;
}
CUresult stream_capture(CUstream, CUstreamCaptureStatus* capture) {
    *capture =
        native->has_capture ? CU_STREAM_CAPTURE_STATUS_ACTIVE : CU_STREAM_CAPTURE_STATUS_NONE;
    return CUDA_SUCCESS;
}
CUresult create_event(CUevent* event, unsigned int flags) {
    require(flags == CU_EVENT_DISABLE_TIMING, "private event flags");
    const auto index = native->creates++;
    if (index + 1 == native->fail_create) {
        if (native->has_failed_create_output)
            *event = reinterpret_cast<CUevent>(&native->events.at(index));
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *event = reinterpret_cast<CUevent>(&native->events.at(index));
    return CUDA_SUCCESS;
}
CUresult record_event(CUevent event, CUstream stream) {
    require(stream == native->expected_stream, "exact canonical stream prefix");
    if (stream == CU_STREAM_PER_THREAD)
        require(std::this_thread::get_id() == native->ptds_caller,
                "PTDS predecessor and prefixes record only on actual caller");
    auto* metadata = reinterpret_cast<Fake::Event*>(event);
    require(!metadata->is_destroyed, "no use after destruction");
    metadata->is_recorded = true;
    const auto count = ++native->records;
    if (count == native->block_record) {
        native->has_blocked_record = true;
        until([&] { return native->allow_record.load(); });
    }
    return count == native->fail_record ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
CUresult query_event(CUevent event) {
    ++native->queries;
    const auto* metadata = reinterpret_cast<Fake::Event*>(event);
    require(metadata->is_recorded && !metadata->is_destroyed,
            "only actual recorded live event queried");
    if (metadata == &native->events.front())
        return native->allow_predecessor.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    if (native->fail_query.load()) return CUDA_ERROR_UNKNOWN;
    return native->allow_prefixes.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult destroy_event(CUevent event) {
    auto* metadata = reinterpret_cast<Fake::Event*>(event);
    require(!metadata->is_destroyed, "exact once destroy");
    metadata->is_destroyed = true;
    ++native->destroys;
    return native->fail_destroy ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
CUresult synchronize_context() {
    require(current_context == kContext, "whole-scope proof uses exact owned context");
    ++native->synchronizations;
    return native->fail_synchronize ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
bool enter_native() noexcept {
    ++native->native_scopes;
    return false;
}
void leave_native(bool) noexcept {
    --native->native_scopes;
}
CudaNativeEpochDispatch dispatch() {
    return {get_context,    push_context, pop_context,  stream_context,
            stream_capture, create_event, record_event, query_event,
            destroy_event,  enter_native, leave_native, synchronize_context};
}
std::shared_ptr<AsyncCooperativeLaunchGate> gate(bool is_enabled = true) {
    LaunchGateOptions options;
    options.max_concurrent_launches = 1;
    options.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority;
    return std::make_shared<AsyncCooperativeLaunchGate>(
        options, 64, is_enabled ? std::optional{kExtent} : std::nullopt);
}
struct Fixture {
    Fake fake;
    Fixture() {
        native = &fake;
        current_context = kContext;
    }
    ~Fixture() {
        native = nullptr;
        current_context = nullptr;
    }
};
void wait_ready(CudaNativeExecutionEpoch& owner) {
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kReady; });
}
void issue(CudaNativeExecutionEpoch& owner,
           NativeEpochCallOutcome outcome = NativeEpochCallOutcome::kSuccess) {
    std::optional<NativeEpochIssue> call;
    until([&] {
        call = owner.begin_call();
        return call.has_value();
    });
    require(owner.original_returned(*call, outcome), "actual original return accounted once");
    require(!owner.original_returned(*call, outcome), "replayed original return rejected");
}
void assert_drained(CudaNativeExecutionEpoch& owner) {
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kComplete; });
    owner.shutdown();
    const auto state = owner.snapshot();
    require(state.status == CUDA_SUCCESS && state.execution && !state.execution->has_ownership,
            "physical proof and exact acknowledged terminal");
    require(state.created_events == state.destroyed_events && native->pushed == 0 &&
                native->native_scopes == 0,
            "all private events and native/context scopes drained");
}
void predecessor_and_capacity() {
    Fixture fixture;
    fixture.fake.allow_predecessor = false;
    auto control = gate();
    CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control, dispatch());
    until([&] { return fixture.fake.queries > 0; });
    require(owner.snapshot().grant.execution.task_id == 0 && !owner.begin_call(),
            "no grant/entry before predecessor proof");
    const auto ordinary = result(control->submit_and_claim());
    require(ordinary.claim.status == CooperativeClaimStatus::kGranted,
            "unproved predecessor consumes no new slot");
    require(result(control->complete(ordinary.claim.execution)), "ordinary synthetic stop");
    fixture.fake.allow_predecessor = true;
    wait_ready(owner);
    fixture.fake.allow_prefixes = false;
    for (unsigned int index = 0; index < 4; ++index) issue(owner);
    until([&] { return owner.snapshot().recorded_prefixes == 2; });
    require(!owner.begin_call(), "bounded pending prefix capacity");
    auto high = result(control->submit_and_claim({10, 1}));
    require(high.claim.status == CooperativeClaimStatus::kPending,
            "unproven prefixes retain shared slot");
    until([&] { return owner.snapshot().execution->phase == NativeEpochPhase::kStopping; });
    require(!owner.begin_call() && owner.snapshot().execution->has_ownership,
            "retreat stops acceptance without release");
    require(result(control->claim(high.task_id)).status == CooperativeClaimStatus::kPending,
            "retreat not completion");
    fixture.fake.allow_prefixes = true;
    assert_drained(owner);
    high.claim = result(control->claim(high.task_id));
    require(high.claim.status == CooperativeClaimStatus::kGranted,
            "high slot only after physical proof");
    require(result(control->complete(high.claim.execution)), "high synthetic stop");
}
void caller_and_call_cap() {
    Fixture fixture;
    auto control = gate();
    CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control, dispatch());
    wait_ready(owner);
    std::atomic<bool> foreign_accepted{false};
    std::thread other([&] {
        current_context = kContext;
        foreign_accepted = owner.begin_call().has_value();
    });
    other.join();
    require(!foreign_accepted, "foreign actual caller excluded");
    current_context = nullptr;
    require(!owner.begin_call(), "wrong actual context excluded");
    current_context = kContext;
    for (unsigned int index = 0; index < 8; ++index) issue(owner);
    assert_drained(owner);
    require(!owner.begin_call() && owner.snapshot().execution->issued_calls == 8 &&
                owner.snapshot().recorded_prefixes == 4,
            "exact call and prefix cap");
}
void idle_expiry() {
    Fixture fixture;
    auto control = gate();
    auto extent = kExtent;
    extent.acceptance_microseconds = 100'000;
    CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {extent, 0}, control, dispatch());
    wait_ready(owner);
    issue(owner);  // incomplete prefix, no later caller and no explicit close
    const auto deadline = owner.snapshot().grant.acceptance_deadline_nanoseconds;
    assert_drained(owner);
    require(owner.snapshot().execution->issued_calls == 1 &&
                owner.snapshot().recorded_prefixes == 1 &&
                owner.snapshot().grant.acceptance_deadline_nanoseconds == deadline,
            "idle fixed expiry closes actual partial prefix");
}
void recycled_caller_identity(bool per_thread_scope = false) {
    Fixture fixture;
    fixture.fake.expected_stream = per_thread_scope ? CU_STREAM_PER_THREAD : kStream;
    auto control = gate();
    auto extent = kExtent;
    if (per_thread_scope) extent.calls_per_prefix = 1;
    std::unique_ptr<CudaNativeExecutionEpoch> owner;
    std::thread::id original_caller;
    std::thread constructing([&] {
        current_context = kContext;
        original_caller = std::this_thread::get_id();
        fixture.fake.ptds_caller = original_caller;
        owner = std::make_unique<CudaNativeExecutionEpoch>(
            kContext, fixture.fake.expected_stream, kScope, NativeEpochLaunchRequest{extent, 0},
            control, dispatch());
        wait_ready(*owner);
    });
    constructing.join();
    bool has_recycled_id = false;
    bool has_accepted_call = false;
    for (unsigned int attempt = 0; attempt < 100 && !has_recycled_id; ++attempt) {
        std::thread replacement([&] {
            current_context = kContext;
            has_recycled_id = std::this_thread::get_id() == original_caller;
            const auto call = owner->begin_call();
            has_accepted_call = call.has_value();
            if (call)
                static_cast<void>(
                    owner->original_returned(*call, NativeEpochCallOutcome::kSuccess));
        });
        replacement.join();
        require(!has_accepted_call, "replacement caller cannot inherit exited caller lifetime");
    }
    require(has_recycled_id, "actual recycled std::thread identity exercised");
    owner->close();
    assert_drained(*owner);
}
void unavailable_terminal_control(bool per_thread_scope = false) {
    Fixture fixture;
    fixture.fake.expected_stream = per_thread_scope ? CU_STREAM_PER_THREAD : kStream;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    auto control = gate();
    auto extent = kExtent;
    if (per_thread_scope) extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, fixture.fake.expected_stream, kScope, {extent, 0},
                                   control, dispatch());
    wait_ready(owner);
    issue(owner);
    control->shutdown();  // metadata shutdown is never physical completion
    owner.close();
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
    owner.shutdown();
    const auto state = owner.snapshot();
    require(
        state.execution->has_ownership && state.execution->phase == NativeEpochPhase::kRetained &&
            state.execution->completed_calls == 1 && state.created_events == state.destroyed_events,
        "physical completion with rejected report retains capacity but can clean proven events");
    require(owner.recover_context_completion(kScope) == CUDA_ERROR_UNKNOWN &&
                fixture.fake.synchronizations == 0,
            "lost/rejected terminal cannot be retried through whole-context recovery");
}
std::unique_ptr<NativeExecutionEpoch> failed_construction(NativeEpochScope,
                                                          NativeEpochAuthorization,
                                                          NativeExecutionEpoch::TimePoint) {
    throw std::bad_alloc{};
}
void failed_model_allocation() {
    Fixture fixture;
    auto control = gate();
    CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control, dispatch(),
                                   {.create_execution = failed_construction});
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kComplete; });
    owner.shutdown();
    const auto state = owner.snapshot();
    require(state.grant.execution.task_id != 0 && !state.execution &&
                state.status == CUDA_ERROR_OUT_OF_MEMORY && state.recorded_prefixes == 0 &&
                state.created_events == state.destroyed_events,
            "allocation failure preserves grant and acknowledges unused failed extent");
    const auto next = result(control->submit_and_claim());
    require(next.claim.status == CooperativeClaimStatus::kGranted,
            "unused failed grant settled without stranding capacity");
    require(result(control->complete(next.claim.execution)), "unused probe synthetic stop");
}
void pending_cancel_and_disabled() {
    Fixture fixture;
    auto control = gate();
    const auto ordinary = result(control->submit_and_claim());
    CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control, dispatch());
    until([&] { return owner.snapshot().grant.execution.task_id != 0; });
    require(!owner.begin_call(), "pending cannot enter original");
    owner.shutdown();
    require(owner.snapshot().phase == CudaNativeEpochPhase::kComplete &&
                !owner.snapshot().execution &&
                owner.snapshot().created_events == owner.snapshot().destroyed_events,
            "pending typed cancel drains unused resources");
    require(result(control->complete(ordinary.claim.execution)),
            "held ordinary completes independently");
    auto disabled = gate(false);
    CudaNativeExecutionEpoch rejected(kContext, kStream, kScope, {kExtent, 0}, disabled,
                                      dispatch());
    until([&] { return rejected.snapshot().phase == CudaNativeEpochPhase::kRejected; });
    rejected.shutdown();
    require(!rejected.snapshot().execution && !rejected.begin_call(),
            "default off cannot create model");
}
void original_failure() {
    Fixture fixture;
    auto control = gate();
    CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control, dispatch());
    wait_ready(owner);
    issue(owner, NativeEpochCallOutcome::kUncertain);
    assert_drained(owner);
    require(owner.snapshot().execution->has_failed && owner.snapshot().execution->issued_calls == 1,
            "uncertain original counted and failed only after trailing proof");
}
void failed_proof(unsigned int mode, bool per_thread_scope = false) {
    Fixture fixture;
    fixture.fake.expected_stream = per_thread_scope ? CU_STREAM_PER_THREAD : kStream;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    fixture.fake.fail_create = mode == 0 ? 2 : 0;
    fixture.fake.fail_record = mode == 1 ? 1 : mode == 2 ? 2 : 0;
    auto control = gate();
    auto extent = kExtent;
    if (per_thread_scope) extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, fixture.fake.expected_stream, kScope, {extent, 0},
                                   control, dispatch());
    if (mode >= 2) {
        wait_ready(owner);
        fixture.fake.allow_prefixes = false;
        issue(owner);
        owner.close();
        if (mode == 3) {
            until([&] { return owner.snapshot().recorded_prefixes == 1; });
            fixture.fake.fail_query = true;
        }
    }
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
    owner.shutdown();
    require(!owner.begin_call() && owner.snapshot().status != CUDA_SUCCESS,
            "failed proof retains owner");
    if (mode >= 2) {
        require(owner.snapshot().execution->has_ownership &&
                    owner.snapshot().execution->has_failed &&
                    owner.snapshot().created_events > owner.snapshot().destroyed_events,
                "unproven private event retained");
        const auto probe = result(control->submit_and_claim());
        require(probe.claim.status == CooperativeClaimStatus::kPending,
                "failed proof never releases capacity");
        require(result(control->cancel_pending(probe.task_id)), "unused probe canceled");
    }
    require(fixture.fake.pushed == 0 && fixture.fake.native_scopes == 0,
            "failed scopes still balanced");
    require(current_context == kContext && context_depth == 0,
            "failed caller cleanup restores the actual context stack");
}
void shutdown_and_cleanup_failures() {
    {
        Fixture fixture;
        fixture.fake.allow_prefixes = false;
        auto control = gate();
        CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control, dispatch(),
                                       {.shutdown_grace = std::chrono::milliseconds{5}});
        wait_ready(owner);
        issue(owner);
        owner.shutdown();
        require(owner.snapshot().phase == CudaNativeEpochPhase::kRetained &&
                    owner.snapshot().execution->has_ownership,
                "bounded shutdown cannot substitute for physical stop");
    }
    {
        Fixture fixture;
        fixture.fake.fail_destroy = true;
        auto control = gate();
        CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control,
                                       dispatch());
        wait_ready(owner);
        owner.close();
        owner.shutdown();
        require(
            owner.snapshot().phase == CudaNativeEpochPhase::kRetained && fixture.fake.destroys == 3,
            "cleanup failure retained without retry");
    }
    {
        Fixture fixture;
        fixture.fake.has_capture = true;
        auto control = gate();
        CudaNativeExecutionEpoch owner(kContext, kStream, kScope, {kExtent, 0}, control,
                                       dispatch());
        until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
        owner.shutdown();
        require(fixture.fake.creates == 0 && owner.snapshot().grant.execution.task_id == 0,
                "capture rejected before event/admission effects");
    }
}

void recover_failed_prefix(unsigned int mode, bool per_thread_scope,
                           CUstream ordinary_stream = kStream) {
    Fixture fixture;
    fixture.fake.expected_stream = per_thread_scope  ? CU_STREAM_PER_THREAD
                                   : ordinary_stream ? ordinary_stream
                                                     : CU_STREAM_LEGACY;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    fixture.fake.allow_prefixes = false;
    fixture.fake.fail_record = mode == 0 ? 2 : 0;
    auto control = gate();
    auto extent = kExtent;
    if (per_thread_scope) extent.calls_per_prefix = 1;
    auto native_dispatch = dispatch();
    if (mode == 5) native_dispatch.context_synchronize = nullptr;
    CudaNativeExecutionEpoch owner(kContext, fixture.fake.expected_stream, kScope, {extent, 0},
                                   control, native_dispatch);
    wait_ready(owner);
    issue(owner);
    owner.close();
    if (mode != 0) {
        until([&] { return owner.snapshot().recorded_prefixes == 1; });
        fixture.fake.fail_query = true;
    }
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
    owner.shutdown();
    const auto retained = owner.snapshot();
    const auto queries = fixture.fake.queries.load();
    const auto records = fixture.fake.records.load();
    require(owner.recover_context_completion({1, 2, 4}) == CUDA_ERROR_INVALID_VALUE &&
                owner.snapshot().execution == retained.execution &&
                fixture.fake.synchronizations == 0,
            "wrong immutable scope performs no recovery effect");
    auto waiting = result(control->submit_and_claim({10, 1}));
    require(waiting.claim.status == CooperativeClaimStatus::kPending,
            "retained physical failure holds shared capacity before recovery");
    fixture.fake.fail_synchronize = mode == 2;
    fixture.fake.fail_destroy = mode == 3;
    fixture.fake.fail_push = mode == 6;
    fixture.fake.has_wrong_recovery_context = mode == 7;
    fixture.fake.fail_pop = mode == 8;
    if (mode == 4) control->shutdown();
    const auto recovered = owner.recover_context_completion(kScope);
    const bool succeeds = mode < 2;
    const auto after = owner.snapshot();
    require(recovered == (succeeds                 ? CUDA_SUCCESS
                          : mode == 5              ? CUDA_ERROR_NOT_SUPPORTED
                          : mode == 6 || mode == 7 ? CUDA_ERROR_INVALID_CONTEXT
                                                   : CUDA_ERROR_UNKNOWN) &&
                after.status == retained.status && after.execution->issued_calls == 1 &&
                fixture.fake.synchronizations == (mode >= 5 && mode <= 7 ? 0U : 1U) &&
                fixture.fake.queries == queries && fixture.fake.records == records &&
                !owner.begin_call() && fixture.fake.pushed == 0 &&
                fixture.fake.native_scopes == 0 && current_context == kContext &&
                context_depth == 0,
            "exact one original whole-context proof, no failed event replay and balanced caller");
    if (succeeds) {
        require(after.phase == CudaNativeEpochPhase::kComplete && !after.execution->has_ownership &&
                    after.execution->has_failed && after.execution->completed_calls == 1 &&
                    after.created_events == after.destroyed_events,
                "failed terminal acknowledged only after whole-scope proof and resource cleanup");
        waiting.claim = result(control->claim(waiting.task_id));
        require(waiting.claim.status == CooperativeClaimStatus::kGranted &&
                    result(control->complete(waiting.claim.execution)),
                "independently stopped failed epoch permits later priority capacity");
    } else {
        require(after.phase == CudaNativeEpochPhase::kRetained && after.execution->has_ownership,
                "failed proof/cleanup/control/missing SDK retains physical ownership");
        if (mode != 4) {
            require(result(control->claim(waiting.task_id)).status ==
                            CooperativeClaimStatus::kPending &&
                        result(control->cancel_pending(waiting.task_id)),
                    "failed recovery never releases a competing shared grant");
        }
    }
    const auto synchronizations = fixture.fake.synchronizations.load();
    const auto destroys = fixture.fake.destroys.load();
    require(owner.recover_context_completion(kScope) != CUDA_SUCCESS &&
                fixture.fake.synchronizations == synchronizations &&
                fixture.fake.destroys == destroys && fixture.fake.queries == queries,
            "repeated recovery cannot replay original synchronization, cleanup or report");
}

void recover_borrowed_completion(unsigned int mode, bool per_thread_scope) {
    Fixture fixture;
    fixture.fake.expected_stream = per_thread_scope ? CU_STREAM_PER_THREAD : kStream;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    fixture.fake.allow_prefixes = false;
    auto control = gate();
    auto extent = kExtent;
    if (per_thread_scope) extent.calls_per_prefix = 1;
    auto native_dispatch = dispatch();
    native_dispatch.context_synchronize = nullptr;  // only the borrowed actual SDK may prove stop
    CudaNativeExecutionEpoch owner(kContext, fixture.fake.expected_stream, kScope, {extent, 0},
                                   control, native_dispatch);
    wait_ready(owner);
    issue(owner);
    owner.close();
    until([&] { return owner.snapshot().recorded_prefixes == 1; });
    fixture.fake.fail_query = true;
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
    owner.shutdown();
    const auto before = owner.snapshot();
    const auto pushes = fixture.fake.context_pushes.load();
    const auto queries = fixture.fake.queries.load();
    unsigned int originals{};
    const auto caller = std::this_thread::get_id();
    if (mode == 1) current_context = nullptr;
    if (mode == 4) fixture.fake.fail_destroy = true;
    if (mode == 5) control->shutdown();
    const auto original = [&] {
        require(current_context == kContext && context_depth == 0 &&
                    std::this_thread::get_id() == caller && fixture.fake.native_scopes == 1,
                "borrowed SDK completion stays on actual caller/context without push or pop");
        ++originals;
        if (mode == 2) return CUDA_ERROR_LAUNCH_FAILED;
        if (mode == 3) current_context = nullptr;
        if (mode == 6) throw std::runtime_error("borrowed SDK completion exception");
        if (mode == 7) fixture.fake.fail_context_query = true;
        return CUDA_SUCCESS;
    };
    const auto recovered = owner.recover_context_completion(kScope, original);
    const auto after = owner.snapshot();
    require((recovered == CUDA_SUCCESS) == (mode == 0) && originals == (mode == 1 ? 0U : 1U) &&
                fixture.fake.synchronizations == 0 && fixture.fake.context_pushes == pushes &&
                fixture.fake.queries == queries && fixture.fake.native_scopes == 0 &&
                context_depth == 0 && after.status == before.status &&
                after.execution->issued_calls == 1 && after.execution->has_failed &&
                after.execution->has_ownership == (mode != 0),
            "one borrowed whole-context proof only releases exact successfully retired ownership");
    if (mode == 0)
        require(after.created_events == after.destroyed_events &&
                    after.phase == CudaNativeEpochPhase::kComplete,
                "borrowed success cleans every certainly live native resource");
    const auto destroys = fixture.fake.destroys.load();
    require(owner.recover_context_completion(kScope, original) != CUDA_SUCCESS &&
                originals == (mode == 1 ? 0U : 1U) && fixture.fake.destroys == destroys &&
                fixture.fake.queries == queries,
            "borrowed failure or success cannot replay SDK proof, cleanup or terminal reporting");
}

void recover_after_original_return(bool per_thread_scope) {
    Fixture fixture;
    fixture.fake.expected_stream = per_thread_scope ? CU_STREAM_PER_THREAD : kStream;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    fixture.fake.allow_prefixes = false;
    auto control = gate();
    auto extent = kExtent;
    if (per_thread_scope) extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, fixture.fake.expected_stream, kScope, {extent, 0},
                                   control, dispatch());
    wait_ready(owner);
    for (unsigned int call = 0; call < extent.calls_per_prefix; ++call) issue(owner);
    until([&] { return owner.snapshot().recorded_prefixes == 1; });
    const auto entered = owner.begin_call();
    require(entered.has_value(), "original entered before independent monitor failure");
    fixture.fake.fail_query = true;
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
    require(owner.recover_context_completion(kScope) == CUDA_ERROR_NOT_READY &&
                fixture.fake.synchronizations == 0,
            "active original excludes recovery before any CUDA/control effect");
    require(owner.original_returned(*entered, NativeEpochCallOutcome::kSuccess),
            "already entered original returns exactly once after retained failure");
    owner.shutdown();
    current_context = nullptr;
    require(owner.recover_context_completion(kScope) == CUDA_SUCCESS && !current_context &&
                context_depth == 0 && fixture.fake.pushed == 0 && fixture.fake.native_scopes == 0 &&
                owner.snapshot().execution->issued_calls == extent.calls_per_prefix + 1,
            "later independent whole-context proof restores a genuinely empty caller stack");
}
void canonical_stream_owner(CUstream stream) {
    Fixture fixture;
    fixture.fake.expected_stream = stream ? stream : CU_STREAM_LEGACY;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    auto control = gate();
    auto extent = kExtent;
    if (stream == CU_STREAM_PER_THREAD) extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, stream, kScope, {extent, 0}, control, dispatch());
    wait_ready(owner);
    fixture.fake.allow_prefixes = false;
    issue(owner);
    issue(owner);
    if (stream == CU_STREAM_PER_THREAD) {
        require(!owner.begin_call() && owner.snapshot().recorded_prefixes == 2,
                "caller PTDS uses both bounded slots without monitor record or unproved reuse");
        owner.close();
        fixture.fake.allow_prefixes = true;
        until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kComplete; });
        require(fixture.fake.destroys == 0, "PTDS cleanup waits for serialized monitor join");
    } else {
        fixture.fake.allow_prefixes = true;
        owner.close();
        until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kComplete; });
    }
    owner.shutdown();
    const auto completed = owner.snapshot();
    require(completed.execution && completed.execution->issued_calls == 2 &&
                completed.execution->completed_calls == 2 && !completed.execution->has_ownership &&
                completed.created_events == completed.destroyed_events &&
                fixture.fake.pushed == 0 && fixture.fake.native_scopes == 0,
            "canonical ownership, actual proof and cleanup settle exactly once");
    const auto destroyed = fixture.fake.destroys.load();
    owner.shutdown();
    require(fixture.fake.destroys == destroyed, "canonical cleanup never replays");
    require(current_context == kContext && context_depth == 0,
            "canonical cleanup preserves the actual caller context stack");
}
void ptds_failed_record_retains_original() {
    Fixture fixture;
    fixture.fake.expected_stream = CU_STREAM_PER_THREAD;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    fixture.fake.fail_record = 2;
    auto control = gate();
    auto extent = kExtent;
    extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, CU_STREAM_PER_THREAD, kScope, {extent, 0}, control,
                                   dispatch());
    wait_ready(owner);
    const auto call = owner.begin_call();
    require(call && owner.original_returned(*call, NativeEpochCallOutcome::kSuccess),
            "actual original return acceptance is distinct from failed private record");
    until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
    std::this_thread::sleep_for(std::chrono::milliseconds{2});
    require(owner.snapshot().phase == CudaNativeEpochPhase::kRetained,
            "later monitor progress cannot overwrite a caller-retained terminal owner");
    owner.shutdown();
    const auto retained = owner.snapshot();
    require(retained.execution && retained.execution->has_ownership &&
                retained.execution->issued_calls == 1 && fixture.fake.records == 2 &&
                retained.created_events > retained.destroyed_events && !owner.begin_call(),
            "failed caller record keeps exact capacity and unproved event without replay");
}
void ptds_failure_during_caller_record() {
    Fixture fixture;
    fixture.fake.expected_stream = CU_STREAM_PER_THREAD;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    fixture.fake.allow_prefixes = false;
    fixture.fake.block_record = 3;
    auto control = gate();
    auto extent = kExtent;
    extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, CU_STREAM_PER_THREAD, kScope, {extent, 0}, control,
                                   dispatch());
    wait_ready(owner);
    issue(owner);
    const auto second = owner.begin_call();
    require(second.has_value(), "second actual caller record reserved before entry");
    std::thread failure([&] {
        until([&] { return fixture.fake.has_blocked_record.load(); });
        fixture.fake.fail_query = true;
        until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kRetained; });
        require(fixture.fake.destroys == 0,
                "failed monitor cannot clean a concurrently returning caller record");
        require(owner.recover_context_completion(kScope) == CUDA_ERROR_NOT_READY &&
                    fixture.fake.synchronizations == 0,
                "returning PTDS record excludes recovery before SDK proof or monitor join");
        fixture.fake.allow_record = true;
    });
    require(owner.original_returned(*second, NativeEpochCallOutcome::kSuccess),
            "already returned original stays accepted across concurrent monitor failure");
    failure.join();
    owner.shutdown();
    const auto retained = owner.snapshot();
    require(retained.execution && retained.execution->issued_calls == 2 &&
                retained.execution->has_ownership && retained.proven_prefixes == 0 &&
                retained.created_events > retained.destroyed_events && fixture.fake.pushed == 0 &&
                fixture.fake.native_scopes == 0,
            "concurrent caller/monitor failure retains exact unproved work without replay");
}
void ptds_partial_and_cleanup_failures(unsigned int mode) {
    Fixture fixture;
    fixture.fake.expected_stream = CU_STREAM_PER_THREAD;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    fixture.fake.fail_create = mode == 0 ? 2 : 0;
    fixture.fake.has_failed_create_output = mode == 0;
    fixture.fake.has_capture = mode == 1;
    auto control = gate();
    auto extent = kExtent;
    extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, CU_STREAM_PER_THREAD, kScope, {extent, 0}, control,
                                   dispatch());
    if (mode >= 2) {
        wait_ready(owner);
        if (mode == 2) {
            const auto call = owner.begin_call();
            require(call.has_value(), "original accepted before caller context change");
            current_context = nullptr;
            require(owner.original_returned(*call, NativeEpochCallOutcome::kSuccess),
                    "private context error cannot replace returned original status");
            current_context = kContext;
        } else {
            owner.close();
            until([&] { return owner.snapshot().phase == CudaNativeEpochPhase::kComplete; });
            if (mode == 3)
                fixture.fake.fail_destroy = true;
            else
                fixture.fake.fail_push = true;
        }
    }
    owner.shutdown();
    const auto retained = owner.snapshot();
    require(retained.phase == CudaNativeEpochPhase::kRetained && !owner.begin_call(),
            "partial/capture/context/cleanup error is sticky");
    if (mode == 0)
        require(fixture.fake.creates == 2 && fixture.fake.destroys == 1 &&
                    !fixture.fake.events[1].is_destroyed,
                "ambiguous failed-create output is retained without destroy or retry");
    if (mode == 1)
        require(fixture.fake.creates == 0 && retained.grant.execution.task_id == 0,
                "caller PTDS capture rejection precedes native and control effects");
    if (mode == 2) {
        require(retained.execution->issued_calls == 1 && retained.execution->has_ownership &&
                    fixture.fake.records == 1,
                "wrong return context retains unproved original without another record");
        const auto context_queries = fixture.fake.context_queries.load();
        const auto context_pushes = fixture.fake.context_pushes.load();
        for (unsigned int attempt = 0; attempt < 2; ++attempt) {
            require(owner.recover_context_completion(kScope) == CUDA_ERROR_INVALID_CONTEXT &&
                        fixture.fake.synchronizations == 0 &&
                        fixture.fake.context_queries == context_queries &&
                        fixture.fake.context_pushes == context_pushes &&
                        owner.snapshot().execution->has_ownership,
                    "old context completion cannot recover an unverified original return context");
        }
    }
    if (mode >= 3)
        require(retained.execution && !retained.execution->has_ownership &&
                    fixture.fake.destroys == (mode == 3 ? 3U : 0U),
                "cleanup error cannot replay acknowledged control or failed native cleanup");
    const auto destroys = fixture.fake.destroys.load();
    owner.shutdown();
    require(fixture.fake.destroys == destroys && fixture.fake.pushed == 0 &&
                fixture.fake.native_scopes == 0 && current_context == kContext &&
                context_depth == 0,
            "partial/error cleanup is idempotent and restores caller scopes");
}
void ptds_unobservable_return_context() {
    Fixture fixture;
    fixture.fake.expected_stream = CU_STREAM_PER_THREAD;
    fixture.fake.ptds_caller = std::this_thread::get_id();
    auto control = gate();
    auto extent = kExtent;
    extent.calls_per_prefix = 1;
    CudaNativeExecutionEpoch owner(kContext, CU_STREAM_PER_THREAD, kScope, {extent, 0}, control,
                                   dispatch());
    wait_ready(owner);
    const auto call = owner.begin_call();
    require(call.has_value(), "original accepted before return context becomes unobservable");
    fixture.fake.fail_context_query = true;
    require(owner.original_returned(*call, NativeEpochCallOutcome::kSuccess),
            "unobservable private return context cannot replace actual original status");
    fixture.fake.fail_context_query = false;
    owner.shutdown();
    const auto retained = owner.snapshot();
    const auto queries = fixture.fake.queries.load();
    const auto destroys = fixture.fake.destroys.load();
    const auto context_queries = fixture.fake.context_queries.load();
    const auto context_pushes = fixture.fake.context_pushes.load();
    for (unsigned int attempt = 0; attempt < 2; ++attempt) {
        require(owner.recover_context_completion(kScope) == CUDA_ERROR_INVALID_CONTEXT &&
                    owner.snapshot().execution->has_ownership &&
                    fixture.fake.synchronizations == 0 && fixture.fake.queries == queries &&
                    fixture.fake.destroys == destroys &&
                    fixture.fake.context_queries == context_queries &&
                    fixture.fake.context_pushes == context_pushes,
                "later observable old context cannot heal unknown original return scope");
    }
    require(retained.status == CUDA_ERROR_UNKNOWN && retained.execution &&
                retained.execution->issued_calls == 1 && fixture.fake.records == 1 &&
                fixture.fake.pushed == 0 && fixture.fake.native_scopes == 0 &&
                current_context == kContext && context_depth == 0,
            "unobservable return retains first error, exact count and balanced caller scopes");
}
}  // namespace
int main() try {
    canonical_stream_owner(nullptr);
    canonical_stream_owner(CU_STREAM_LEGACY);
    canonical_stream_owner(CU_STREAM_PER_THREAD);
    for (unsigned int repetition = 0; repetition < 50; ++repetition)
        ptds_failed_record_retains_original();
    ptds_failure_during_caller_record();
    for (unsigned int mode = 0; mode < 5; ++mode) ptds_partial_and_cleanup_failures(mode);
    ptds_unobservable_return_context();
    predecessor_and_capacity();
    caller_and_call_cap();
    idle_expiry();
    recycled_caller_identity();
    recycled_caller_identity(true);
    unavailable_terminal_control();
    unavailable_terminal_control(true);
    failed_model_allocation();
    pending_cancel_and_disabled();
    original_failure();
    for (unsigned int mode = 0; mode < 4; ++mode) {
        failed_proof(mode);
        failed_proof(mode, true);
    }
    shutdown_and_cleanup_failures();
    for (unsigned int mode = 0; mode < 9; ++mode) {
        recover_failed_prefix(mode, false);
        recover_failed_prefix(mode, true);
    }
    recover_after_original_return(false);
    recover_after_original_return(true);
    recover_failed_prefix(0, false, CU_STREAM_LEGACY);
    recover_failed_prefix(1, false, nullptr);
    for (unsigned int mode = 0; mode < 8; ++mode) {
        recover_borrowed_completion(mode, false);
        recover_borrowed_completion(mode, true);
    }
    std::cout << "Private native epoch dispatch tests passed (synthetic physical proof)\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (...) {
    return 1;
}
