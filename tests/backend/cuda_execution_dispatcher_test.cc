#include "internal/cuda_execution_dispatcher.h"
#include "internal/cuda_retirement_observer.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
thread_local bool counts_process_checks{};
thread_local unsigned int process_checks{};
}  // namespace

// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming): GNU ld's exact ABI.
extern "C" pid_t __real_getpid() noexcept;
extern "C" pid_t __wrap_getpid() noexcept {
    if (counts_process_checks) ++process_checks;
    return __real_getpid();
}
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;

void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

struct Event {
    bool is_live{};
    bool is_recorded{};
    bool is_ready{};
};
struct FakeState {
    struct MemoryCall {
        CudaMemoryKind kind;
        CUdeviceptr destination{}, source{};
        std::size_t count{};
        std::uint32_t value{};
        CUstream stream{};
        bool is_asynchronous{};
    };
    char context_storage{}, other_context{}, function_storage{}, stream_storage{};
    CUresult context_status{CUDA_SUCCESS}, create_status{CUDA_SUCCESS};
    CUresult launch_status{CUDA_SUCCESS}, record_status{CUDA_SUCCESS};
    CUresult query_status{CUDA_ERROR_NOT_READY}, destroy_status{CUDA_SUCCESS};
    CUresult drain_status{CUDA_SUCCESS};
    CUresult memory_status{CUDA_SUCCESS};
    bool is_wrong_context{}, returns_null_event{}, returns_error_handle{};
    bool loses_context_on_query{}, loses_context_on_destroy{};
    unsigned int context_calls{}, launch_calls{}, record_calls{}, query_calls{};
    unsigned int destroy_calls{}, drain_calls{};
    std::array<Event, 32> events{};
    std::vector<std::uint32_t> values;
    std::vector<unsigned int> grids;
    std::vector<MemoryCall> memory_calls;
    std::function<void()> on_context;
};
FakeState state;

CUcontext context() {
    return reinterpret_cast<CUcontext>(&state.context_storage);
}
CUfunction function() {
    return reinterpret_cast<CUfunction>(&state.function_storage);
}
CUstream stream() {
    return reinterpret_cast<CUstream>(&state.stream_storage);
}
CUresult CUDAAPI get_current(CUcontext* current) {
    ++state.context_calls;
    if (state.on_context) state.on_context();
    *current =
        state.is_wrong_context ? reinterpret_cast<CUcontext>(&state.other_context) : context();
    return state.context_status;
}
CUresult CUDAAPI create_event(CUevent* event, unsigned int flags) {
    expect(flags == CU_EVENT_DISABLE_TIMING, "completion events must not collect timing");
    if (state.create_status != CUDA_SUCCESS && !state.returns_error_handle)
        return state.create_status;
    if (state.returns_null_event) return CUDA_SUCCESS;
    for (auto& entry : state.events) {
        if (entry.is_live) continue;
        entry = {.is_live = true};
        *event = reinterpret_cast<CUevent>(&entry);
        return state.create_status;
    }
    return CUDA_ERROR_OUT_OF_MEMORY;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native CUDA ABI.
CUresult CUDAAPI launch(CUfunction native, unsigned int x, unsigned int y, unsigned int z,
                        unsigned int bx, unsigned int by, unsigned int bz, unsigned int, CUstream,
                        void** parameters, void** extra) {
    expect(native == function() && x && y && z && bx && by && bz && !parameters,
           "native issue must use bound function and packed, owned parameters");
    ++state.launch_calls;
    state.grids.push_back(x);
    std::uint32_t value{};
    if (extra) {
        expect(extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER &&
                   extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[4] == CU_LAUNCH_PARAM_END &&
                   *static_cast<std::size_t*>(extra[3]) == sizeof(value),
               "packed ABI must preserve the exact parameter buffer size");
        std::memcpy(&value, extra[1], sizeof(value));
    }
    state.values.push_back(value);
    return state.launch_status;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CUresult CUDAAPI record_event(CUevent event, CUstream) {
    ++state.record_calls;
    if (state.record_status == CUDA_SUCCESS) reinterpret_cast<Event*>(event)->is_recorded = true;
    return state.record_status;
}
CUresult CUDAAPI query_event(CUevent event) {
    ++state.query_calls;
    if (state.loses_context_on_query) state.is_wrong_context = true;
    // Real Driver events may report SUCCESS without ever having been recorded.
    const auto& observed = *reinterpret_cast<Event*>(event);
    return observed.is_ready || !observed.is_recorded ? CUDA_SUCCESS : state.query_status;
}
CUresult CUDAAPI destroy_event(CUevent event) {
    ++state.destroy_calls;
    if (state.destroy_status == CUDA_SUCCESS) reinterpret_cast<Event*>(event)->is_live = false;
    if (state.loses_context_on_destroy) state.is_wrong_context = true;
    return state.destroy_status;
}
CUresult CUDAAPI drain_stream(CUstream) {
    ++state.drain_calls;
    return state.drain_status;
}
CudaExecutionDispatch dispatch() {
    CudaExecutionDispatch value{get_current, launch,        create_event, record_event,
                                query_event, destroy_event, drain_stream};
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA ABI.
    value.copy_device = +[](CUdeviceptr destination, CUdeviceptr source, std::size_t count) {
        state.memory_calls.push_back(
            {CudaMemoryKind::kCopy, destination, source, count, 0, nullptr, false});
        return state.memory_status;
    };
    value.copy_device_async =
        +[](CUdeviceptr destination, CUdeviceptr source, std::size_t count, CUstream stream) {
            state.memory_calls.push_back(
                {CudaMemoryKind::kCopy, destination, source, count, 0, stream, true});
            return state.memory_status;
        };
#define FILL_DISPATCH(BITS, TYPE, KIND)                                               \
    value.fill##BITS = +[](CUdeviceptr destination, TYPE bits, std::size_t count) {   \
        state.memory_calls.push_back(                                                 \
            {CudaMemoryKind::KIND, destination, 0, count, bits, nullptr, false});     \
        return state.memory_status;                                                   \
    };                                                                                \
    value.fill##BITS##_async =                                                        \
        +[](CUdeviceptr destination, TYPE bits, std::size_t count, CUstream stream) { \
            state.memory_calls.push_back(                                             \
                {CudaMemoryKind::KIND, destination, 0, count, bits, stream, true});   \
            return state.memory_status;                                               \
        }
    FILL_DISPATCH(8, unsigned char, kFill8);
    FILL_DISPATCH(16, unsigned short, kFill16);
    FILL_DISPATCH(32, unsigned int, kFill32);
#undef FILL_DISPATCH
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return value;
}
struct Fixture {
    std::shared_ptr<const void> resources{std::make_shared<int>(0)};
    CudaExecutionDispatcher executor;
    explicit Fixture(bool is_nonblocking = false, std::size_t binding_limit = 2,
                     bool is_profile_isolated = false,
                     std::function<CUresult()> exclusive_native_gate = {})
        : executor({1, 2}, {7, context(), resources}, dispatch(),
                   {.max_commands = 4, .max_parameter_bytes = 16}, binding_limit,
                   is_profile_isolated, {}, {}, std::move(exclusive_native_gate)) {
        expect(executor.bind_executor() == CUDA_SUCCESS &&
                   executor.bind_function(1, function(), sizeof(std::uint32_t), resources) &&
                   executor.bind_stream(1, stream(), resources, is_nonblocking),
               "bind actual owner and resources");
    }
    ExecutionAcceptance accept(std::uint64_t sequence, std::uint32_t value) {
        return executor.accept_kernel(sequence,
                                      {.function = 1,
                                       .stream = 1,
                                       .grid = {80, 1, 1},
                                       .block = {32, 1, 1},
                                       .parameters = std::as_bytes(std::span{&value, 1})});
    }
};

void scheduling_snapshot_preserves_state_and_bounds_process_checks() {
    state = {};
    Fixture fixture;
    const auto agrees = [&] {
        const auto aggregate = fixture.executor.scheduling_snapshot();
        const auto queue = fixture.executor.snapshot();
        expect(
            aggregate.submissions.outstanding == queue.outstanding &&
                aggregate.submissions.queued == queue.queued &&
                aggregate.submissions.issuing == queue.issuing &&
                aggregate.submissions.awaiting_physical_stop == queue.awaiting_physical_stop &&
                aggregate.submissions.paused == queue.paused &&
                aggregate.submissions.unacknowledged_replies == queue.unacknowledged_replies &&
                aggregate.submissions.retained_parameter_bytes == queue.retained_parameter_bytes &&
                aggregate.submissions.has_executor == queue.has_executor &&
                aggregate.submissions.is_closing == queue.is_closing &&
                aggregate.submissions.is_quarantined == queue.is_quarantined &&
                aggregate.external_barriers == fixture.executor.external_barrier_count() &&
                aggregate.enqueue_turns == fixture.executor.enqueue_turn_count() &&
                aggregate.external_work_in_flight == fixture.executor.started_external_count() &&
                aggregate.has_ready_work == fixture.executor.has_ready_work(),
            "aggregate scheduling projection preserves every legacy field");
    };
    agrees();
    const auto accepted = fixture.accept(1, 73);
    expect(accepted.accepted(), "snapshot guard admits the original kernel");
    const auto sdk_before = state.context_calls;
    process_checks = 0;
    counts_process_checks = true;
    const auto queued = fixture.executor.scheduling_snapshot();
    counts_process_checks = false;
    const auto checks = process_checks;
    expect(queued.submissions.queued == 1 && queued.has_ready_work &&
               state.context_calls == sdk_before,
           "snapshot preserves ready work without original SDK execution");
    agrees();
    bool excludes_reentry{};
    state.on_context = [&] {
        const auto nested = fixture.executor.scheduling_snapshot();
        excludes_reentry = nested.external_barriers == 0 && !nested.has_ready_work &&
                           nested.submissions.outstanding == 1;
    };
    expect(fixture.executor.issue_next().sequence == 1 && excludes_reentry &&
               state.values == std::vector<std::uint32_t>{73},
           "native reentry does not manufacture readiness or lose the actual command");
    state.on_context = {};
    agrees();
    expect(!fixture.executor.scheduling_snapshot().has_ready_work &&
               fixture.executor.scheduling_snapshot().submissions.awaiting_physical_stop == 1,
           "issued native work retains its actual physical tail");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(accepted.receipt),
           "original physical completion and reply retire normally");
    agrees();
    const auto barrier = fixture.executor.accept_external_barrier(1, fixture.resources);
    const auto turn = fixture.executor.accept_enqueue_turn(1, fixture.resources);
    expect(barrier.status == CUDA_SUCCESS && turn.status == CUDA_SUCCESS,
           "capture distinct external and native enqueue records");
    agrees();
    const auto external = fixture.executor.scheduling_snapshot();
    expect(external.external_barriers == 1 && external.enqueue_turns == 1 &&
               external.external_work_in_flight == 0 && !external.has_ready_work &&
               fixture.executor.begin_external_barrier(barrier.barrier) == CUDA_SUCCESS,
           "accepted unstarted external records are not physical work");
    agrees();
    expect(fixture.executor.scheduling_snapshot().external_work_in_flight == 1 &&
               fixture.executor.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "started reservation remains visible after cancelling a distinct unstarted turn");
    bool retains_reentrant_physical_work{};
    state.on_context = [&] {
        const auto nested = fixture.executor.scheduling_snapshot();
        retains_reentrant_physical_work = nested.external_work_in_flight == 1 &&
                                          nested.external_barriers == 0 && !nested.has_ready_work;
    };
    expect(fixture.executor.record_external_enqueue(barrier.barrier) == CUDA_SUCCESS &&
               retains_reentrant_physical_work,
           "callback suppression never hides started external physical ownership");
    state.on_context = {};
    expect(fixture.executor.poll().retired == 1 && fixture.executor.shutdown(),
           "started external work remains reserved until actual trailing event retirement");
    agrees();
    expect(checks >= 1 && checks <= 2,
           "one scheduling projection must not repeat five independent process checks");
}

struct Observation final : CudaExecutionObserver {
    unsigned int begins{}, finishes{};
    bool was_successful{};
    void begin() noexcept override {
        ++begins;
    }
    void finish(bool success) noexcept override {
        ++finishes;
        was_successful = success;
    }
};
void profile_observes_retirement_and_preserves_failures() {
    for (unsigned int mode = 0; mode < 4; ++mode) {
        state = {};
        Fixture fixture(true, 2, true);
        auto observation = std::make_shared<Observation>();
        const std::uint32_t argument = 73;
        const auto accepted =
            fixture.executor.accept_kernel(1, {.function = 1,
                                               .stream = 1,
                                               .grid = {80, 1, 1},
                                               .block = {32, 1, 1},
                                               .parameters = std::as_bytes(std::span{&argument, 1}),
                                               .observation = observation});
        if (mode == 1) state.launch_status = CUDA_ERROR_INVALID_VALUE;
        if (mode == 2) state.record_status = CUDA_ERROR_INVALID_VALUE;
        if (mode == 3) state.create_status = CUDA_ERROR_OUT_OF_MEMORY;
        const auto issued = fixture.executor.issue_next();
        expect(accepted.accepted() && observation->finishes == 0 &&
                   observation->begins == (mode == 3 ? 0U : 1U) &&
                   fixture.executor.acknowledge_result(accepted.receipt),
               "measurement starts only at real issue and never finishes at SDK return");
        if (mode == 3) {
            expect(issued.state == CudaExecutionIssueState::kRejected && !state.launch_calls &&
                       fixture.executor.shutdown(),
                   "pre-issue failure cannot manufacture a timing sample");
            continue;
        }
        const auto following = fixture.executor.accept_kernel(
            2, {.function = 1,
                .stream = 0,
                .grid = {80, 1, 1},
                .block = {32, 1, 1},
                .parameters = std::as_bytes(std::span{&argument, 1})});
        if (mode != 2)
            expect(following.accepted() &&
                       fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle,
                   "profile holds independent native streams until physical retirement");
        else
            expect(!following.accepted(), "unproven fence quarantines new work");
        state.query_status = CUDA_SUCCESS;
        const auto observed = mode == 2 ? fixture.executor.drain_issued() : fixture.executor.poll();
        expect(observed.retired == 1 && observation->begins == 1 && observation->finishes == 1 &&
                   observation->was_successful == (mode == 0),
               "only successful final physical retirement is a successful sample");
        if (mode != 2) {
            state.launch_status = CUDA_SUCCESS;
            expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
                       fixture.executor.acknowledge_result(following.receipt) &&
                       fixture.executor.poll().retired == 1,
                   "independent stream resumes after the measured command truly ends");
        }
        expect(fixture.executor.shutdown(), "profile failure controls preserve native cleanup");
    }
}

struct Progress final : CudaExecutionPersistentProgress {
    CUresult prepare_status{CUDA_SUCCESS}, retreat_status{CUDA_SUCCESS};
    CUresult readback_status{CUDA_SUCCESS}, stop_status{CUDA_SUCCESS};
    CUresult prefix_status{CUDA_SUCCESS};
    unsigned int prepare_calls{}, retreat_calls{}, readback_calls{}, stop_calls{}, prefix_calls{};
    CUresult retarget_status{CUDA_SUCCESS};
    unsigned int retarget_calls{}, workers{320};
    std::function<void()> on_retarget;
    std::vector<CudaExecutionParameterUpdate> updates;
    bool is_complete{};
    CUresult prepare_launch() noexcept override {
        ++prepare_calls;
        return prepare_status;
    }
    CUresult request_retreat() noexcept override {
        ++retreat_calls;
        return retreat_status;
    }
    CUresult retarget_workers(unsigned int requested) noexcept override {
        ++retarget_calls;
        if (on_retarget) on_retarget();
        if (retarget_status != CUDA_SUCCESS) return retarget_status;
        if (!requested || requested > 320) return CUDA_ERROR_INVALID_VALUE;
        workers = requested;
        return CUDA_SUCCESS;
    }
    CUresult record_readback(CUstream) noexcept override {
        ++readback_calls;
        return readback_status;
    }
    CUresult record_initialization(CUstream) noexcept override {
        ++prefix_calls;
        return prefix_status;
    }
    std::span<const CudaExecutionParameterUpdate> parameter_updates() const noexcept override {
        return updates;
    }
    CudaExecutionPersistentStop observe_physical_stop() noexcept override {
        ++stop_calls;
        return {stop_status, is_complete};
    }
};

ExecutionAcceptance accept_persistent(Fixture& fixture, const std::shared_ptr<Progress>& progress,
                                      std::uint32_t private_value = 73,
                                      std::uint64_t stream_id = 1) {
    const std::uint32_t original = 41;
    return fixture.executor.accept_persistent_kernel(
        1,
        {.function = 1,
         .stream = stream_id,
         .grid = {6400, 1, 1},
         .block = {32, 1, 1},
         .parameters = std::as_bytes(std::span{&original, 1})},
        {function(), 320, std::as_bytes(std::span{&private_value, 1}), progress});
}

void persistent_incomplete_stop_and_exact_resume() {
    state = {};
    Fixture fixture(true);
    auto progress = std::make_shared<Progress>();
    const auto low = accept_persistent(fixture, progress);
    const auto frontier = fixture.executor.capture_frontier(1);
    expect(low.accepted() && frontier.status == CUDA_SUCCESS && state.launch_calls == 0,
           "owned persistent acceptance is not original execution or logical completion");
    expect(fixture.executor.request_retreat(1) == CUDA_ERROR_NOT_SUPPORTED &&
               fixture.executor.resume(1).state == CudaExecutionIssueState::kError &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               state.values == std::vector<std::uint32_t>{73} && state.grids[0] == 320 &&
               progress->prepare_calls == 1 && progress->readback_calls == 1,
           "private ABI is copied and explicit physical workers replace only physical geometry");
    expect(fixture.executor.request_retreat(1) == CUDA_SUCCESS && progress->retreat_calls == 1 &&
               fixture.executor.poll().retained == 1 && progress->stop_calls == 0 &&
               fixture.executor.snapshot().paused == 0 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY &&
               fixture.executor.acknowledge_result(low.receipt),
           "retreat flag and SDK reply ACK cannot prove physical or logical completion");
    state.query_status = CUDA_SUCCESS;
    const auto partial = fixture.executor.poll();
    expect(partial.status == CUDA_SUCCESS && partial.retired == 0 && partial.paused == 1 &&
               partial.retained == 1 && progress->stop_calls == 1 && state.destroy_calls == 1 &&
               fixture.executor.snapshot().awaiting_physical_stop == 0 &&
               fixture.executor.snapshot().retained_parameter_bytes == 4 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
           "real incomplete stop destroys one event but retains logical task, bytes and frontier");
    const auto queries = state.query_calls;
    expect(fixture.executor.poll().paused == 1 && state.query_calls == queries &&
               progress->stop_calls == 1 && fixture.executor.request_retreat(1) == CUDA_SUCCESS &&
               progress->retreat_calls == 1,
           "paused work must not be queried, reobserved or mistaken for active GPU work");
    const std::uint32_t high_value = 11;
    const auto high =
        fixture.executor.accept_kernel(2, {.function = 1,
                                           .stream = 0,
                                           .grid = {80, 1, 1},
                                           .block = {32, 1, 1},
                                           .parameters = std::as_bytes(std::span{&high_value, 1})});
    const auto high_frontier = fixture.executor.capture_frontier(0);
    expect(high.accepted() && high_frontier.status == CUDA_SUCCESS &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               fixture.executor.acknowledge_result(high.receipt) &&
               fixture.executor.poll().retired == 1 &&
               fixture.executor.query_frontier(high_frontier.frontier) == CUDA_SUCCESS &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
           "other work may finish while the exact incomplete logical low frontier stays pending");
    state.create_status = CUDA_ERROR_OUT_OF_MEMORY;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kRejected &&
               fixture.executor.snapshot().paused == 1 && state.launch_calls == 2 &&
               !fixture.executor.snapshot().is_quarantined,
           "known pre-issue event allocation failure cannot issue, retire or lose paused work");
    state.create_status = CUDA_SUCCESS;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.resume(1).state == CudaExecutionIssueState::kError &&
               state.values == std::vector<std::uint32_t>{73, 11, 73} &&
               !fixture.executor.host_result(low.receipt) && fixture.executor.poll().paused == 1 &&
               progress->stop_calls == 2,
           "one continuation reuses exact owned ABI without publishing a second application reply");
    fixture.executor.close();
    progress->is_complete = true;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.poll().retired == 1 && progress->stop_calls == 3 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
               fixture.executor.snapshot().retained_parameter_bytes == 0 &&
               fixture.executor.snapshot().paused == 0 && state.drain_calls == 0 &&
               fixture.executor.shutdown(),
           "accepted continuation drains after close; only final real completion settles low");
}

void stopped_worker_retarget_preserves_the_exact_command() {
    state = {};
    Fixture fixture(true, 4);
    auto progress = std::make_shared<Progress>();
    const auto low = accept_persistent(fixture, progress);
    const auto frontier = fixture.executor.capture_command_frontier(1);
    expect(low.accepted() && frontier.status == CUDA_SUCCESS &&
               fixture.executor.retarget_workers(1, {80}) == CUDA_ERROR_NOT_READY &&
               fixture.executor.retarget_workers(1, {0}) == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.retarget_workers(1, {UINT32_MAX}) == CUDA_ERROR_INVALID_VALUE &&
               !progress->retarget_calls,
           "queued work and invalid geometry cannot reach the trusted retarget callback");
    expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               fixture.executor.acknowledge_result(low.receipt) &&
               fixture.executor.request_retreat(1) == CUDA_SUCCESS &&
               fixture.executor.retarget_workers(1, {80}) == CUDA_ERROR_NOT_READY &&
               fixture.executor.poll().paused == 0 && !progress->retarget_calls,
           "issue, host ACK and retreat intent are not physical-stop authority");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1 &&
               fixture.executor.retarget_workers(1, {80}) == CUDA_SUCCESS &&
               progress->workers == 80 && state.grids == std::vector<unsigned int>{320} &&
               fixture.executor.snapshot().paused == 1 &&
               fixture.executor.snapshot().retained_parameter_bytes == 4 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
           "retarget changes CPU geometry only while exact logical parameters and frontier stay");
    progress->retarget_status = CUDA_ERROR_INVALID_VALUE;
    expect(fixture.executor.retarget_workers(1, {160}) == CUDA_ERROR_INVALID_VALUE &&
               progress->workers == 80,
           "failed trusted retarget preserves the prior configuration");
    progress->retarget_status = CUDA_SUCCESS;
    const auto calls = progress->retarget_calls;
    state.is_wrong_context = true;
    expect(fixture.executor.retarget_workers(1, {160}) == CUDA_ERROR_INVALID_CONTEXT &&
               progress->retarget_calls == calls,
           "wrong current context rejects before retarget mutation");
    state.is_wrong_context = false;
    auto foreign =
        std::async(std::launch::async, [&] { return fixture.executor.retarget_workers(1, {160}); });
    expect(foreign.get() == CUDA_ERROR_INVALID_CONTEXT && progress->retarget_calls == calls,
           "another thread cannot retarget an owner-bound paused command");
    bool excludes_reentry{};
    progress->on_retarget = [&] {
        excludes_reentry = fixture.executor.retarget_workers(1, {160}) == CUDA_ERROR_NOT_READY &&
                           fixture.executor.resume(1).native_status == CUDA_ERROR_NOT_READY;
    };
    expect(fixture.executor.retarget_workers(1, {80}) == CUDA_SUCCESS && excludes_reentry &&
               fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.poll().paused == 1 &&
               state.grids == std::vector<unsigned int>{320, 80},
           "retarget callbacks cannot reenter and the next physical issue uses80 workers");
    progress->on_retarget = {};
    expect(fixture.executor.retarget_workers(1, {160}) == CUDA_SUCCESS,
           "same retained incomplete task can change its next interval again");
    progress->is_complete = true;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.poll().retired == 1 &&
               state.grids == std::vector<unsigned int>{320, 80, 160} &&
               state.values == std::vector<std::uint32_t>{73, 73, 73} &&
               !fixture.executor.host_result(low.receipt) &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
               fixture.executor.snapshot().retained_parameter_bytes == 0 &&
               fixture.executor.retarget_workers(1, {80}) == CUDA_ERROR_INVALID_HANDLE &&
               fixture.executor.shutdown(),
           "one final retirement settles the original frontier without replay or a second reply");
}

void pending_commands_bind_exact_records_without_sdk_work() {
    state = {};
    Fixture fixture(true, 4);
    const auto first = fixture.accept(1, 41);
    const auto second = fixture.accept(2, 73);
    const auto contexts = state.context_calls;
    const auto pending = fixture.executor.pending_command();
    expect(first.accepted() && second.accepted() && pending.status == CUDA_SUCCESS &&
               pending.command && pending.command.description().sequence == 1 &&
               pending.command.description().acceptance_order == 1 &&
               pending.command.description().physical_generation == 0 &&
               pending.command.description().geometry_epoch == 1 &&
               pending.command.description().function == 1 &&
               pending.command.description().stream == 1 &&
               pending.command.description().physical_grid == std::array{80U, 1U, 1U} &&
               !pending.command.description().is_persistent &&
               !pending.command.description().is_resume && state.context_calls == contexts &&
               !state.launch_calls && !state.record_calls && !state.query_calls,
           "CPU pending capture binds immutable logical/physical metadata without SDK work");
    expect(fixture.executor.pending_command(2).status == CUDA_ERROR_NOT_READY &&
               fixture.executor.pending_command(0).status == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.pending_command(91).status == CUDA_ERROR_INVALID_HANDLE,
           "explicit pending sequence never selects unrelated or dependency-blocked work");
    auto foreign = std::async(std::launch::async, [&] {
        return fixture.executor.pending_command().status == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.issue(pending.command).native_status == CUDA_ERROR_INVALID_CONTEXT;
    });
    expect(foreign.get() && state.context_calls == contexts, "foreign token checks enter no SDK");
    const auto child = ::fork();
    expect(child >= 0, "fork exact pending owner control");
    if (!child)
        ::_exit(fixture.executor.pending_command().status == CUDA_ERROR_INVALID_CONTEXT &&
                        fixture.executor.issue(pending.command).native_status ==
                            CUDA_ERROR_INVALID_CONTEXT
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork cannot borrow inherited pending execution authority");
    {
        Fixture other(true, 4);
        expect(other.executor.issue(pending.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
                   other.executor.shutdown(),
               "another dispatcher cannot use this record even with identical numeric identities");
    }
    bool excludes_reentry{};
    state.on_context = [&] {
        excludes_reentry =
            fixture.executor.pending_command().status == CUDA_ERROR_NOT_READY &&
            fixture.executor.issue(pending.command).native_status == CUDA_ERROR_NOT_READY;
    };
    expect(fixture.executor.issue(pending.command).state == CudaExecutionIssueState::kIssued &&
               excludes_reentry && state.values == std::vector<std::uint32_t>{41},
           "exact original issue preserves packed parameters and excludes native proof reentry");
    state.on_context = {};
    const auto next = fixture.executor.pending_command();
    const auto before_stale = state.context_calls;
    expect(next.command && next.command.description().sequence == 2 &&
               fixture.executor.issue(pending.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
               state.context_calls == before_stale && state.launch_calls == 1,
           "consumed token never falls through to another ready command or performs SDK proof");
    fixture.executor.close();
    expect(fixture.executor.issue(next.command).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.acknowledge_result(first.receipt) &&
               fixture.executor.acknowledge_result(second.receipt),
           "close permits exact draining of already accepted commands");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 2 && !pending.command && !next.command &&
               fixture.executor.issue(next.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
               fixture.executor.shutdown(),
           "weak tokens do not retain retired records/resources or invent later issue permission");
}

void pending_continuation_invalidates_old_geometry_and_physical_generations() {
    state = {};
    Fixture fixture(true, 4);
    auto progress = std::make_shared<Progress>();
    const auto accepted = accept_persistent(fixture, progress);
    const auto initial = fixture.executor.pending_command();
    expect(accepted.accepted() && initial.command &&
               fixture.executor.issue(initial.command).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.acknowledge_result(accepted.receipt) &&
               fixture.executor.request_retreat(1) == CUDA_SUCCESS &&
               !fixture.executor.pending_command().command,
           "SDK result, host ACK and retreat request cannot authorize incomplete continuation");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1, "actual physical stop precedes pending resume");
    const auto paused = fixture.executor.pending_command();
    expect(paused.command && paused.command.description().is_resume &&
               paused.command.description().physical_generation == 1 &&
               paused.command.description().geometry_epoch == 1 &&
               fixture.executor.retarget_workers(1, {80}) == CUDA_SUCCESS &&
               fixture.executor.retarget_workers(1, {320}) == CUDA_SUCCESS,
           "stopped command retains exact receipt/generation while geometry revisions advance");
    const auto before_stale = state.context_calls;
    const auto prepares = progress->prepare_calls;
    expect(fixture.executor.issue(paused.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
               state.context_calls == before_stale && progress->prepare_calls == prepares,
           "returning to the same worker count cannot revive an obsolete geometry token");
    const auto updated = fixture.executor.pending_command();
    expect(updated.command.description().geometry_epoch == 3 &&
               updated.command.description().physical_generation == 1,
           "new pending description includes current nonwrapping geometry revision");
    progress->retarget_status = CUDA_ERROR_INVALID_VALUE;
    expect(fixture.executor.retarget_workers(1, {80}) == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.pending_command().command.description() ==
                   updated.command.description(),
           "clean retarget rejection leaves the valid token unchanged");
    progress->prepare_status = CUDA_ERROR_INVALID_VALUE;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kRejected,
           "even an old direct continuation path records an actual preparation attempt");
    const auto failed_prepares = progress->prepare_calls;
    const auto failed_contexts = state.context_calls;
    expect(fixture.executor.issue(updated.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
               progress->prepare_calls == failed_prepares && state.context_calls == failed_contexts,
           "failed preparation cannot revive the same pending token or replay private work");
    const auto after_failure = fixture.executor.pending_command();
    expect(after_failure.command.description().command_epoch ==
                   updated.command.description().command_epoch + 1 &&
               after_failure.command.description().physical_generation == 1,
           "command-attempt epoch changes even before a new physical generation is issued");
    progress->prepare_status = CUDA_SUCCESS;
    expect(
        fixture.executor.issue(after_failure.command).state == CudaExecutionIssueState::kIssued &&
            fixture.executor.poll().paused == 1 &&
            fixture.executor.issue(after_failure.command).native_status ==
                CUDA_ERROR_INVALID_HANDLE,
        "a subsequent actual physical generation invalidates the preceding resume token");
    const auto final = fixture.executor.pending_command();
    expect(final.command.description().physical_generation == 2 &&
               final.command.description().geometry_epoch == 3,
           "next continuation binds the actual last stopped physical generation");
    progress->is_complete = true;
    expect(
        fixture.executor.issue(final.command).state == CudaExecutionIssueState::kIssued &&
            fixture.executor.poll().retired == 1 &&
            !fixture.executor.host_result(accepted.receipt) &&
            !fixture.executor.pending_command().command && fixture.executor.shutdown() &&
            state.values == std::vector<std::uint32_t>{73, 73, 73},
        "exact continuation keeps one host reply and original charged payload until real finish");
}

void pending_original_selection_keeps_exact_abi_and_retained_budget() {
    state = {};
    Fixture fixture(true, 4);
    auto progress = std::make_shared<Progress>();
    auto observation = std::make_shared<Observation>();
    const std::uint32_t original = 41;
    const std::array packed{original, 73U, 99U};
    const auto accepted = fixture.executor.accept_persistent_kernel(
        1,
        {.function = 1,
         .stream = 1,
         .grid = {6400, 1, 1},
         .block = {32, 1, 1},
         .parameters = std::as_bytes(std::span{&original, 1})},
        {function(), 320, std::as_bytes(std::span{packed}), progress});
    const auto pending = fixture.executor.pending_command();
    const auto before = state.context_calls;
    const auto selected = fixture.executor.select_pending_kernel(
        pending.command, {.uses_original = true, .observation = observation});
    expect(accepted.accepted() && selected.status == CUDA_SUCCESS && selected.command &&
               !selected.command.description().is_persistent &&
               selected.command.description().sequence == pending.command.description().sequence &&
               selected.command.description().acceptance_order ==
                   pending.command.description().acceptance_order &&
               selected.command.description().geometry_epoch == 2 &&
               selected.command.description().physical_generation == 0 &&
               selected.command.description().physical_grid == std::array{6400U, 1U, 1U} &&
               fixture.executor.snapshot().retained_parameter_bytes == sizeof(packed) &&
               !progress->prepare_calls && !progress->prefix_calls && !progress->retarget_calls &&
               !observation->begins && state.context_calls == before && !state.launch_calls,
           "CPU original selection keeps one exact command, owned prefix and full private charge");
    expect(fixture.executor.issue(pending.command).native_status == CUDA_ERROR_INVALID_HANDLE &&
               fixture.executor.select_pending_kernel(pending.command, {.workers = 80}).status ==
                   CUDA_ERROR_INVALID_HANDLE &&
               state.context_calls == before,
           "old selection tokens cannot issue or configure a different representation");
    expect(fixture.executor.issue(selected.command).state == CudaExecutionIssueState::kIssued &&
               state.values == std::vector<std::uint32_t>{41} &&
               state.grids == std::vector<unsigned int>{6400} && observation->begins == 1 &&
               !observation->finishes && fixture.executor.acknowledge_result(accepted.receipt) &&
               fixture.executor.snapshot().retained_parameter_bytes == sizeof(packed),
           "Driver sees exact original ABI length/value, not trailing private arguments");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 && observation->finishes == 1 &&
               observation->was_successful &&
               fixture.executor.snapshot().retained_parameter_bytes == 0 &&
               !fixture.executor.host_result(accepted.receipt) && fixture.executor.shutdown(),
           "one native result and actual physical stop retire the full payload exactly once");
}

void pending_worker_selection_preserves_initial_and_paused_ownership() {
    state = {};
    Fixture fixture(true, 4);
    auto progress = std::make_shared<Progress>();
    const auto accepted = accept_persistent(fixture, progress);
    const auto pending = fixture.executor.pending_command();
    const auto before = state.context_calls;
    expect(
        fixture.executor.select_pending_kernel(pending.command, {.uses_original = true}).status ==
                CUDA_ERROR_NOT_SUPPORTED &&
            fixture.executor.select_pending_kernel(pending.command, {}).status ==
                CUDA_ERROR_INVALID_VALUE &&
            fixture.executor
                    .select_pending_kernel(pending.command, {.workers = 80, .uses_original = true})
                    .status == CUDA_ERROR_INVALID_VALUE &&
            state.context_calls == before,
        "a noncanonical private prefix cannot masquerade as saved original arguments");
    auto foreign = std::async(std::launch::async, [&] {
        return fixture.executor.select_pending_kernel(pending.command, {.workers = 80}).status;
    });
    expect(foreign.get() == CUDA_ERROR_INVALID_CONTEXT && !progress->retarget_calls,
           "only the nonrecycled original owner can select pending workers");
    bool excludes_reentry{};
    progress->on_retarget = [&] {
        excludes_reentry =
            fixture.executor.pending_command().status == CUDA_ERROR_NOT_READY &&
            fixture.executor.select_pending_kernel(pending.command, {.workers = 80}).status ==
                CUDA_ERROR_NOT_READY &&
            fixture.executor.issue(pending.command).native_status == CUDA_ERROR_NOT_READY;
    };
    auto observation = std::make_shared<Observation>();
    const auto initial = fixture.executor.select_pending_kernel(
        pending.command, {.workers = 80, .observation = observation});
    expect(initial.status == CUDA_SUCCESS && initial.command && excludes_reentry &&
               state.context_calls == before && !state.launch_calls && !progress->prefix_calls &&
               initial.command.description().physical_grid == std::array{80U, 1U, 1U} &&
               initial.command.description().geometry_epoch == 2,
           "initial worker selection has no SDK, private initialization, reply or execution "
           "permission");
    progress->on_retarget = {};
    expect(fixture.executor.issue(initial.command).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.acknowledge_result(accepted.receipt) &&
               fixture.executor.select_pending_kernel(initial.command, {.workers = 160}).status ==
                   CUDA_ERROR_INVALID_HANDLE,
           "an actual issue consumes its original selection capability");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1, "actual event stop precedes retained selection");
    const auto paused = fixture.executor.pending_command();
    const auto stopped_contexts = state.context_calls;
    expect(fixture.executor
                       .select_pending_kernel(
                           paused.command,
                           {.workers = 160, .observation = std::make_shared<Observation>()})
                       .status == CUDA_ERROR_NOT_SUPPORTED &&
               fixture.executor.select_pending_kernel(paused.command, {.uses_original = true})
                       .status == CUDA_ERROR_NOT_SUPPORTED,
           "partial work cannot change its original observer or restart native computation");
    const auto continued = fixture.executor.select_pending_kernel(paused.command, {.workers = 160});
    expect(continued.status == CUDA_SUCCESS && continued.command.description().is_resume &&
               continued.command.description().physical_generation == 1 &&
               continued.command.description().geometry_epoch == 3 &&
               state.context_calls == stopped_contexts && observation->begins == 1 &&
               !observation->finishes && fixture.executor.snapshot().retained_parameter_bytes == 4,
           "known-stopped worker selection keeps physical generation, observer and charged counter "
           "payload");
    progress->is_complete = true;
    expect(fixture.executor.issue(continued.command).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.poll().retired == 1 && observation->begins == 1 &&
               observation->finishes == 1 && observation->was_successful &&
               state.grids == std::vector<unsigned int>{80, 160} &&
               state.values == std::vector<std::uint32_t>{73, 73} && fixture.executor.shutdown(),
           "original physical continuation finishes one observer and one logical command without "
           "reset");
}

void exact_command_frontiers_exclude_unrelated_work() {
    state = {};
    Fixture fixture(true, 4);
    auto progress = std::make_shared<Progress>();
    const auto low = accept_persistent(fixture, progress);
    const auto first = fixture.executor.capture_command_frontier(1);
    expect(low.accepted() && first.status == CUDA_SUCCESS &&
               fixture.executor.capture_command_frontier(0).status == CUDA_ERROR_INVALID_HANDLE &&
               fixture.executor.capture_command_frontier(2).status == CUDA_ERROR_INVALID_HANDLE &&
               fixture.executor.issue_next().sequence == 1,
           "exact command capture requires its accepted immutable identity");
    const std::uint32_t value = 29;
    const auto later =
        fixture.executor.accept_kernel(2, {.function = 1,
                                           .stream = 0,
                                           .grid = {80, 1, 1},
                                           .block = {32, 1, 1},
                                           .parameters = std::as_bytes(std::span{&value, 1})});
    const auto second = fixture.executor.capture_command_frontier(2);
    const auto prefix = fixture.executor.capture_frontier();
    expect(later.accepted() && second.status == CUDA_SUCCESS && prefix.status == CUDA_SUCCESS &&
               fixture.executor.issue_next().sequence == 2 &&
               fixture.executor.acknowledge_result(low.receipt) &&
               fixture.executor.acknowledge_result(later.receipt),
           "independent commands retain separate stop observations after host metadata ACK");
    state.events[1].is_ready = true;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.query_frontier(second.frontier) == CUDA_SUCCESS &&
               fixture.executor.query_frontier(first.frontier) == CUDA_ERROR_NOT_READY &&
               fixture.executor.query_frontier(prefix.frontier) == CUDA_ERROR_NOT_READY &&
               fixture.executor.capture_command_frontier(2).status == CUDA_ERROR_INVALID_HANDLE,
           "later exact completion excludes an earlier active epoch and survives record erasure");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1 &&
               fixture.executor.query_frontier(first.frontier) == CUDA_ERROR_NOT_READY &&
               fixture.executor.query_frontier(second.frontier) == CUDA_SUCCESS,
           "actual incomplete stop is not logical command completion");
    progress->is_complete = true;
    expect(fixture.executor.resume(1).sequence == 1 && fixture.executor.poll().retired == 1 &&
               fixture.executor.query_frontier(first.frontier) == CUDA_SUCCESS &&
               fixture.executor.query_frontier(prefix.frontier) == CUDA_SUCCESS &&
               fixture.executor.shutdown(),
           "exact retained continuation settles only its original command observation");
}

void memory_commands_keep_sdk_results_and_physical_ownership() {
    for (const auto kind : {CudaMemoryKind::kCopy, CudaMemoryKind::kFill8, CudaMemoryKind::kFill16,
                            CudaMemoryKind::kFill32})
        for (const bool is_asynchronous : {false, true})
            for (const bool has_sdk_error : {false, true}) {
                state = {};
                Fixture fixture(true);
                const std::uint32_t bits = kind == CudaMemoryKind::kCopy ? 0 : 37;
                CudaExecutionMemory command{
                    .kind = kind,
                    .destination = 0x2000,
                    .source = kind == CudaMemoryKind::kCopy ? CUdeviceptr{0x1000} : CUdeviceptr{},
                    .count = 17,
                    .value = bits,
                    .stream = is_asynchronous ? 1U : 0U,
                    .is_asynchronous = is_asynchronous};
                const auto accepted = fixture.executor.accept_memory(1, command);
                const auto frontier = fixture.executor.capture_command_frontier(1);
                command.destination = 0x3000;
                command.count = 3;
                state.memory_status = has_sdk_error ? CUDA_ERROR_INVALID_VALUE : CUDA_SUCCESS;
                expect(
                    accepted.accepted() && frontier.status == CUDA_SUCCESS &&
                        state.memory_calls.empty(),
                    "memory acceptance copies bounded metadata but cannot fabricate an SDK call");
                const auto issued = fixture.executor.issue_next();
                const auto reply = fixture.executor.host_result(accepted.receipt);
                expect(issued.state == CudaExecutionIssueState::kIssued && reply &&
                           reply->status == state.memory_status &&
                           reply->origin == ExecutionReplyOrigin::kNativeSdk &&
                           state.memory_calls.size() == 1 && state.launch_calls == 0,
                       "each copy/fill calls its exact original SDK once and preserves its result");
                const auto& call = state.memory_calls.front();
                expect(call.kind == kind && call.destination == 0x2000 && call.count == 17 &&
                           call.value == bits && call.is_asynchronous == is_asynchronous &&
                           call.stream == (is_asynchronous ? stream() : nullptr),
                       "typed count/value/stream and owned addresses survive caller mutation");
                expect(fixture.executor.acknowledge_result(accepted.receipt) &&
                           fixture.executor.poll().retained == 1 && state.destroy_calls == 0 &&
                           fixture.executor.query_frontier(frontier.frontier) ==
                               CUDA_ERROR_NOT_READY &&
                           !fixture.executor.resources_quiescent(),
                       "SDK result acknowledgment is not memory completion or resource release");
                state.query_status = CUDA_SUCCESS;
                expect(fixture.executor.poll().retired == 1 && state.destroy_calls == 1 &&
                           fixture.executor.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
                           state.drain_calls == 0 && fixture.executor.shutdown(),
                       "only actual trailing-event proof retires owned memory work");
            }
    state = {};
    Fixture fixture;
    CudaExecutionMemory invalid{.kind = CudaMemoryKind::kFill32, .destination = 0x2001, .count = 1};
    expect(!fixture.executor.accept_memory(1, invalid).accepted(), "reject misaligned typed fill");
    invalid.destination = 0x2000;
    invalid.count = std::numeric_limits<std::size_t>::max();
    expect(!fixture.executor.accept_memory(1, invalid).accepted(), "reject byte-size overflow");
    invalid.count = 1;
    invalid.stream = 1;
    expect(!fixture.executor.accept_memory(1, invalid).accepted(),
           "ordinary memory calls use legacy stream");
    invalid.is_asynchronous = true;
    const auto accepted = fixture.executor.accept_memory(1, invalid);
    expect(accepted.accepted(), "invalid metadata did not consume sequence");
    state.record_status = CUDA_ERROR_UNKNOWN;
    const auto issue = fixture.executor.issue_next();
    expect(issue.completion_status == CUDA_ERROR_UNKNOWN && fixture.executor.poll().retained == 1 &&
               state.query_calls == 0 && state.memory_calls.size() == 1,
           "failed trailing record never queries an empty event or repeats the memory SDK call");
    expect(fixture.executor.drain_issued().retired == 1 && state.memory_calls.size() == 1 &&
               fixture.executor.acknowledge_result(accepted.receipt) && fixture.executor.shutdown(),
           "only explicit successful recovery drain proves physical stop after record failure");
}

void memory_waits_for_logical_persistent_predecessor() {
    state = {};
    Fixture fixture(true, 4);
    auto progress = std::make_shared<Progress>();
    const auto low = accept_persistent(fixture, progress);
    expect(low.accepted() && fixture.executor.issue_next().sequence == 1,
           "start persistent producer");
    const auto dependent = fixture.executor.accept_memory(2, {.destination = 0x2000,
                                                              .source = 0x1000,
                                                              .count = 16,
                                                              .stream = 1,
                                                              .is_asynchronous = true});
    expect(dependent.accepted() &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle,
           "copy cannot pass incomplete work on the same logical stream");
    const auto independent = fixture.executor.accept_memory(
        3, {.kind = CudaMemoryKind::kFill32, .destination = 0x3000, .count = 1, .value = 1});
    expect(independent.accepted() && fixture.executor.issue_next().sequence == 3 &&
               state.memory_calls.size() == 1 &&
               fixture.executor.request_retreat(3) == CUDA_ERROR_NOT_SUPPORTED,
           "independent original memory work may issue but is honestly nonpreemptible");
    state.query_status = CUDA_SUCCESS;
    const auto stopped = fixture.executor.poll();
    expect(stopped.paused == 1 && stopped.retired == 1 &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle,
           "physical partial stop is not permission for the dependent copy");
    progress->is_complete = true;
    expect(
        fixture.executor.resume(1).sequence == 1 && fixture.executor.poll().retired == 1 &&
            fixture.executor.issue_next().sequence == 2 && fixture.executor.poll().retired == 1 &&
            state.memory_calls.size() == 2 && fixture.executor.acknowledge_result(low.receipt) &&
            fixture.executor.acknowledge_result(dependent.receipt) &&
            fixture.executor.acknowledge_result(independent.receipt) && fixture.executor.shutdown(),
        "final logical producer completion releases exactly the original dependent copy");
}

void persistent_logical_stream_ordering() {
    for (const bool is_nonblocking : {false, true}) {
        state = {};
        Fixture fixture(is_nonblocking);
        auto progress = std::make_shared<Progress>();
        const auto low = accept_persistent(fixture, progress);
        expect(low.accepted() &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
               "start persistent predecessor");
        const auto same_stream = fixture.accept(2, 29);
        expect(same_stream.accepted() &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
                   state.launch_calls == 1,
               "same logical stream cannot enqueue beyond a potentially partial physical epoch");
        const std::uint32_t value = 31;
        const auto legacy =
            fixture.executor.accept_kernel(3, {.function = 1,
                                               .stream = 0,
                                               .grid = {80, 1, 1},
                                               .block = {32, 1, 1},
                                               .parameters = std::as_bytes(std::span{&value, 1})});
        expect(legacy.accepted(), "accept later legacy operation");
        const auto issued = fixture.executor.issue_next();
        expect(issued.state == (is_nonblocking ? CudaExecutionIssueState::kIssued
                                               : CudaExecutionIssueState::kIdle) &&
                   (!is_nonblocking || issued.sequence == 3),
               "nonblocking independent work may pass; legacy/blocking logical fences may not");
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().paused == 1 &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
                   fixture.executor.snapshot().queued == (is_nonblocking ? 1U : 2U),
               "actual partial stop must not remove logical stream or legacy dependencies");
        progress->is_complete = true;
        expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued &&
                   fixture.executor.poll().retired == 1 &&
                   fixture.executor.issue_next().sequence == 2,
               "final continuation completion releases the earliest blocked logical command");
        if (!is_nonblocking)
            expect(fixture.executor.issue_next().sequence == 3,
                   "ordinary SDK ordering still pipelines native legacy successors");
        expect(fixture.executor.poll().retired == (is_nonblocking ? 1U : 2U) &&
                   fixture.executor.acknowledge_result(low.receipt) &&
                   fixture.executor.acknowledge_result(same_stream.receipt) &&
                   fixture.executor.acknowledge_result(legacy.receipt) &&
                   fixture.executor.shutdown(),
               "logical ordering drains each original and receipt once");
    }
    for (const bool is_nonblocking : {false, true}) {
        state = {};
        Fixture fixture(is_nonblocking);
        auto progress = std::make_shared<Progress>();
        const auto low = accept_persistent(fixture, progress, 73, 0);
        expect(low.accepted() &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
               "start legacy persistent predecessor");
        const auto later = fixture.accept(2, 17);
        expect(later.accepted() && fixture.executor.issue_next().state ==
                                       (is_nonblocking ? CudaExecutionIssueState::kIssued
                                                       : CudaExecutionIssueState::kIdle),
               "legacy predecessor fences blocking streams but not explicit nonblocking ones");
        progress->is_complete = true;
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().retired == (is_nonblocking ? 2U : 1U),
               "retire only physically issued commands");
        if (!is_nonblocking)
            expect(
                fixture.executor.issue_next().sequence == 2 && fixture.executor.poll().retired == 1,
                "known final legacy stop releases the blocked stream");
        expect(fixture.executor.acknowledge_result(low.receipt) &&
                   fixture.executor.acknowledge_result(later.receipt) &&
                   fixture.executor.shutdown(),
               "legacy dependency ownership drains");
    }
}

void persistent_preparation_and_completion_errors() {
    state = {};
    {
        Fixture fixture;
        auto progress = std::make_shared<Progress>();
        progress->prepare_status = CUDA_ERROR_INVALID_VALUE;
        const auto low = accept_persistent(fixture, progress);
        const auto rejected = fixture.executor.issue_next();
        const auto reply = fixture.executor.host_result(low.receipt);
        expect(rejected.state == CudaExecutionIssueState::kRejected && reply &&
                   reply->status == CUDA_ERROR_INVALID_VALUE &&
                   reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   state.launch_calls == 0 && state.record_calls == 0 &&
                   fixture.executor.acknowledge_result(low.receipt) && fixture.executor.shutdown(),
               "private preparation failure must not fabricate an original SDK return");
    }
    for (const bool fails_readback : {false, true}) {
        state = {};
        Fixture fixture;
        auto progress = std::make_shared<Progress>();
        if (fails_readback) progress->readback_status = CUDA_ERROR_INVALID_VALUE;
        const auto low = accept_persistent(fixture, progress);
        const auto frontier = fixture.executor.capture_frontier(1);
        const auto initial = fixture.executor.issue_next();
        expect(initial.native_status == CUDA_SUCCESS && frontier.status == CUDA_SUCCESS,
               "actual initial launch result is independent of later private errors");
        if (!fails_readback) {
            state.query_status = CUDA_SUCCESS;
            expect(fixture.executor.poll().paused == 1, "prepare a genuinely paused continuation");
            state.launch_status = CUDA_ERROR_INVALID_VALUE;
            state.query_status = CUDA_ERROR_NOT_READY;
            const auto resumed = fixture.executor.resume(1);
            expect(resumed.state == CudaExecutionIssueState::kIssued &&
                       resumed.native_status == CUDA_ERROR_INVALID_VALUE &&
                       progress->readback_calls == 1,
                   "failed continuation SDK call must not read or resume incomplete progress");
        }
        const auto reply = fixture.executor.host_result(low.receipt);
        expect(reply && reply->status == CUDA_SUCCESS &&
                   reply->origin == ExecutionReplyOrigin::kNativeSdk &&
                   fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_VALUE &&
                   fixture.executor.snapshot().is_quarantined &&
                   fixture.executor.poll().retained == 1 &&
                   fixture.executor.resume(1).state == CudaExecutionIssueState::kError,
               "async failure keeps original success, reports the actual error and waits for stop");
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().retired == 1 &&
                   fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_VALUE &&
                   fixture.executor.acknowledge_result(low.receipt) && fixture.executor.shutdown(),
               "known failed logical task may release only after real physical stop and cleanup");
    }
}

void persistent_progress_errors_are_not_completion() {
    state = {};
    Fixture fixture;
    auto progress = std::make_shared<Progress>();
    const auto low = accept_persistent(fixture, progress);
    const auto frontier = fixture.executor.capture_frontier(1);
    expect(low.accepted() && frontier.status == CUDA_SUCCESS &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
           "prepare real-issued persistent progress errors");
    progress->retreat_status = CUDA_ERROR_INVALID_VALUE;
    expect(fixture.executor.request_retreat(1) == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.snapshot().paused == 0 && progress->stop_calls == 0,
           "failed retreat publication cannot become a stop or logical completion");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1, "only the actual event may establish partial stop");
    progress->prepare_status = CUDA_ERROR_INVALID_VALUE;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kRejected &&
               fixture.executor.snapshot().paused == 1 && state.launch_calls == 1 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
           "known CPU preparation failure preserves the same stopped incomplete command");
    progress->prepare_status = CUDA_SUCCESS;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued,
           "known pre-issue rejection did not consume or duplicate continuation");
    progress->stop_status = CUDA_ERROR_INVALID_VALUE;
    const auto observed = fixture.executor.poll();
    const auto reply = fixture.executor.host_result(low.receipt);
    expect(observed.status == CUDA_ERROR_INVALID_VALUE && observed.retired == 1 && reply &&
               reply->status == CUDA_SUCCESS && fixture.executor.snapshot().is_quarantined &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.acknowledge_result(low.receipt) && fixture.executor.shutdown(),
           "invalid stopped progress is a terminal async failure, never completed compute");
}

void private_prefix_failure_is_not_a_launch(bool invalid_patch = false) {
    state = {};
    Fixture fixture;
    auto progress = std::make_shared<Progress>();
    progress->prefix_status = CUDA_ERROR_INVALID_VALUE;
    if (invalid_patch) progress->updates.push_back({1, 99});
    const auto low = accept_persistent(fixture, progress);
    const auto frontier = fixture.executor.capture_frontier();
    const auto issue = fixture.executor.issue_next();
    const auto reply = fixture.executor.host_result(low.receipt);
    expect(low.accepted() && frontier.status == CUDA_SUCCESS &&
               issue.state == CudaExecutionIssueState::kRejected && state.launch_calls == 0 &&
               state.record_calls == 1 && reply &&
               reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               reply->status == CUDA_ERROR_INVALID_VALUE && progress->readback_calls == 0 &&
               progress->prefix_calls == (invalid_patch ? 0U : 1U) &&
               fixture.executor.acknowledge_result(low.receipt) &&
               fixture.executor.poll().retained == 1 &&
               fixture.executor.snapshot().retained_parameter_bytes == 4,
           "failed asynchronous initialization must record a fence, not call the target or free "
           "private state");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 && state.destroy_calls == 1 &&
               progress->stop_calls == 0 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.shutdown(),
           "real private-prefix stop terminates failed logical work without fabricated compute");
}

void continuation_patch_failure_preserves_original_result() {
    state = {};
    Fixture fixture;
    auto progress = std::make_shared<Progress>();
    progress->updates.push_back({0, 91});
    const auto low = accept_persistent(fixture, progress);
    const auto frontier = fixture.executor.capture_frontier();
    expect(low.accepted() && frontier.status == CUDA_SUCCESS &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               state.values == std::vector<std::uint32_t>{91} &&
               fixture.executor.acknowledge_result(low.receipt),
           "initial private patch reaches actual SDK with one original result");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1, "real incomplete stop precedes continuation patch");
    progress->updates[0].offset = SIZE_MAX;
    const auto resumed = fixture.executor.resume(1);
    expect(resumed.state == CudaExecutionIssueState::kRejected &&
               resumed.native_status == CUDA_ERROR_INVALID_VALUE && state.launch_calls == 1 &&
               progress->prefix_calls == 1 && !fixture.executor.host_result(low.receipt) &&
               fixture.executor.snapshot().is_quarantined && fixture.executor.poll().retired == 1 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.shutdown(),
           "invalid continuation patch never issues compute or rewrites the acknowledged reply");
}

void persistent_ambiguous_cleanup_retains_once() {
    const auto child = fork();
    expect(child >= 0, "create isolated partial-stop cleanup quarantine");
    if (child == 0) {
        try {
            state = {};
            auto fixture = std::make_unique<Fixture>();
            auto progress = std::make_shared<Progress>();
            const auto low = accept_persistent(*fixture, progress);
            const auto frontier = fixture->executor.capture_frontier(1);
            expect(low.accepted() && frontier.status == CUDA_SUCCESS &&
                       fixture->executor.issue_next().state == CudaExecutionIssueState::kIssued,
                   "prepare physically issued partial stop");
            state.query_status = CUDA_SUCCESS;
            state.destroy_status = CUDA_ERROR_INVALID_HANDLE;
            expect(fixture->executor.poll().status == CUDA_ERROR_INVALID_HANDLE &&
                       fixture->executor.snapshot().is_quarantined && state.destroy_calls == 1 &&
                       fixture->executor.poll().retained == 1 &&
                       fixture->executor.drain_issued().retained == 1 && state.destroy_calls == 1 &&
                       fixture->executor.resume(1).state == CudaExecutionIssueState::kError &&
                       fixture->executor.query_frontier(frontier.frontier) ==
                           CUDA_ERROR_INVALID_HANDLE &&
                       fixture->executor.snapshot().retained_parameter_bytes == 4 &&
                       state.launch_calls == 1 && progress->stop_calls == 1 &&
                       !fixture->executor.shutdown(),
                   "ambiguous event destruction cannot retry, resume or release logical inputs");
            fixture.reset();
            expect(state.destroy_calls == 1, "unfinished destructor cannot retry native cleanup");
            _exit(0);
        } catch (...) {
            _exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "intentional unknown cleanup retention is bounded to the isolated child");
}

void native_dispatch_pipeline() {
    state = {};
    Fixture fixture;
    const auto first = fixture.accept(1, 41), second = fixture.accept(2, 73);
    expect(first.accepted() && second.accepted() && !fixture.executor.host_result(first.receipt),
           "acceptance must neither invoke CUDA nor publish SDK success");
    expect(
        fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
            fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
            state.launch_calls == 2 && state.values == std::vector<std::uint32_t>{41, 73} &&
            state.query_calls == 0 && state.drain_calls == 0,
        "two original native kernels must issue without polling or waiting for first completion");
    expect(fixture.executor.acknowledge_result(first.receipt) &&
               fixture.executor.acknowledge_result(second.receipt) &&
               fixture.executor.snapshot().retained_parameter_bytes == 8,
           "SDK reply ACK does not release in-flight parameters");
    expect(
        fixture.executor.poll().retained == 2 && state.destroy_calls == 0 && state.drain_calls == 0,
        "NOT_READY must preserve both native executions");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 2 && state.destroy_calls == 2 &&
               fixture.executor.snapshot().retained_parameter_bytes == 0 &&
               fixture.executor.shutdown(),
           "only actual stop queries and successful event cleanup retire parameters");
}

struct LibraryCall final : CudaExecutionLibraryCall {
    std::int32_t result{13};
    unsigned int calls{};
    std::uint32_t value{};
    CudaExecutionLibraryCompletion completed;
    mutable unsigned int completion_calls{};
    bool should_change_context{};
    std::int32_t issue(CUstream native, std::span<const std::byte> parameters) noexcept override {
        if (native != stream() || parameters.size() != sizeof(value)) return -1;
        std::memcpy(&value, parameters.data(), sizeof(value));
        ++calls;
        if (should_change_context) state.is_wrong_context = true;
        return result;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        ++completion_calls;
        return completed;
    }
};

void exclusive_library_gate_preserves_queued_work() {
    for (const bool uses_exact_progress : {false, true}) {
        state = {};
        bool has_other_physical_work{};
        bool has_reentry_rejection{true};
        unsigned int gate_calls{};
        Fixture* active{};
        Fixture fixture(true, 2, false, [&]() -> CUresult {
            ++gate_calls;
            const auto nested = active->executor.issue_next();
            has_reentry_rejection =
                has_reentry_rejection && nested.state == CudaExecutionIssueState::kError &&
                nested.native_status == CUDA_ERROR_NOT_READY && !active->accept(2, 99).accepted() &&
                active->executor.reject_queued(CUDA_ERROR_DEINITIALIZED) == 0;
            return has_other_physical_work ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
        });
        active = &fixture;
        auto call = std::make_shared<LibraryCall>();
        std::uint32_t value{53};
        const auto accepted =
            fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
        const auto captured = fixture.executor.capture_command_frontier(1);
        expect(accepted.accepted() && captured.status == CUDA_SUCCESS && gate_calls == 0,
               "opaque exclusivity is checked at issue, not admission");
        value = 91;
        has_other_physical_work = true;
        for (unsigned int attempt = 0; attempt != 2; ++attempt) {
            const auto deferred =
                uses_exact_progress
                    ? fixture.executor.progress_frontier_dependency(captured.frontier)
                    : fixture.executor.issue_next();
            const auto pending = fixture.executor.snapshot();
            expect(deferred.state == CudaExecutionIssueState::kIdle && deferred.sequence == 0 &&
                       deferred.native_status == CUDA_ERROR_NOT_READY && !deferred.library_status &&
                       !fixture.executor.host_result(accepted.receipt) && pending.queued == 1 &&
                       pending.issuing == 0 && pending.awaiting_physical_stop == 0 &&
                       pending.retained_parameter_bytes == sizeof(value) &&
                       !pending.is_quarantined && call->calls == 0 && call->completion_calls == 0 &&
                       state.launch_calls == 0 && state.record_calls == 0 &&
                       state.query_calls == 0 && state.destroy_calls == 0 &&
                       state.drain_calls == 0 &&
                       std::ranges::none_of(state.events,
                                            [](const auto& event) { return event.is_live; }) &&
                       captured.frontier.observe() == CUDA_ERROR_NOT_READY &&
                       !captured.frontier.retirement().is_retired,
                   "held cohort keeps exact library queued without native preparation or reply");
        }
        expect(gate_calls == 2 && has_reentry_rejection,
               "exclusive predicate runs outside queue locks under callback exclusion");
        has_other_physical_work = false;
        const auto issued = uses_exact_progress
                                ? fixture.executor.progress_frontier_dependency(captured.frontier)
                                : fixture.executor.issue_next();
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(issued.state == CudaExecutionIssueState::kIssued && issued.sequence == 1 &&
                   issued.native_status == CUDA_SUCCESS && issued.library_status == 13 && reply &&
                   reply->origin == ExecutionReplyOrigin::kNativeSdk && reply->status == 13 &&
                   gate_calls == 3 && call->calls == 1 && call->value == 53,
               "actual cohort stop allows exactly one original callback and exact vendor status");
        has_other_physical_work = true;
        const auto following = fixture.accept(2, 17);
        expect(following.accepted() &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
                   state.values == std::vector<std::uint32_t>{17} && gate_calls == 3 &&
                   fixture.executor.acknowledge_result(accepted.receipt) &&
                   fixture.executor.acknowledge_result(following.receipt),
               "reentry consumed no admission sequence and pure kernel bypasses opaque gate");
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().retired == 2 && captured.frontier.retirement().is_retired &&
                   call->calls == 1 && fixture.executor.shutdown(),
               "library gate does not replace actual physical retirement or cleanup");
    }
}

void exclusive_library_gate_does_not_serialize_persistent_kernels() {
    state = {};
    unsigned int gate_calls{};
    Fixture fixture(true, 2, false, [&]() -> CUresult {
        ++gate_calls;
        return CUDA_ERROR_NOT_READY;
    });
    auto progress = std::make_shared<Progress>();
    const auto accepted = accept_persistent(fixture, progress);
    expect(accepted.accepted() &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               gate_calls == 0 && state.launch_calls == 1 && progress->prepare_calls == 1,
           "persistent computation remains eligible while another member is physical");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1 &&
               fixture.executor.acknowledge_result(accepted.receipt),
           "persistent kernel retains its actual incomplete physical stop");
    progress->is_complete = true;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued &&
               gate_calls == 0 && state.launch_calls == 2 && fixture.executor.poll().retired == 1 &&
               fixture.executor.shutdown(),
           "exact persistent resume is not converted into an opaque exclusive call");
}

void exclusive_library_gate_failures_retain_owners() {
    for (unsigned int mode = 0; mode != 3; ++mode) {
        const auto child = ::fork();
        expect(child >= 0, "create isolated exclusive-gate uncertainty control");
        if (!child) {
            try {
                state = {};
                unsigned int gate_calls{};
                auto fixture = std::make_unique<Fixture>(true, 2, false, [&]() -> CUresult {
                    ++gate_calls;
                    if (mode == 1) throw std::runtime_error("unproven exclusive predicate");
                    if (mode == 2) {
                        state.is_wrong_context = true;
                        return CUDA_SUCCESS;
                    }
                    return CUDA_ERROR_OUT_OF_MEMORY;
                });
                auto call = std::make_shared<LibraryCall>();
                std::weak_ptr<LibraryCall> lease = call;
                const std::uint32_t value{61};
                const auto accepted = fixture->executor.accept_library(
                    1, {1, std::as_bytes(std::span{&value, 1}), call});
                const auto captured = fixture->executor.capture_command_frontier(1);
                expect(accepted.accepted() && captured.status == CUDA_SUCCESS,
                       "accept exact opaque call before uncertain dispatch gate");
                const auto issued = fixture->executor.issue_next();
                const auto expected = mode == 0   ? CUDA_ERROR_OUT_OF_MEMORY
                                      : mode == 1 ? CUDA_ERROR_UNKNOWN
                                                  : CUDA_ERROR_INVALID_CONTEXT;
                expect(issued.state == CudaExecutionIssueState::kError && issued.sequence == 0 &&
                           issued.native_status == expected && !issued.library_status &&
                           !fixture->executor.host_result(accepted.receipt) && call->calls == 0 &&
                           call->completion_calls == 0 && gate_calls == 1 &&
                           fixture->executor.snapshot().is_quarantined &&
                           captured.frontier.observe() == expected &&
                           !captured.frontier.retirement().is_retired,
                       "gate failure retains exact error without requested SDK provenance");
                state.is_wrong_context = false;
                call.reset();
                expect(
                    fixture->executor.issue_next().state == CudaExecutionIssueState::kError &&
                        gate_calls == 1 &&
                        !fixture->executor.reject_command(accepted.receipt,
                                                          CUDA_ERROR_DEINITIALIZED) &&
                        fixture->executor.reject_queued(CUDA_ERROR_DEINITIALIZED) == 0 &&
                        !fixture->executor.host_result(accepted.receipt) &&
                        fixture->executor.poll().retained == 1 &&
                        fixture->executor.drain_issued().retained == 1 &&
                        !fixture->executor.shutdown() && !lease.expired() &&
                        state.launch_calls == 0 && state.record_calls == 0 &&
                        state.query_calls == 0 && state.destroy_calls == 0 &&
                        state.drain_calls == 0,
                    "unknown gate cannot be retried, cancelled or drained into ownership release");
                fixture.reset();
                expect(
                    !lease.expired() && state.destroy_calls == 0 && state.drain_calls == 0,
                    "destruction retains uncertain opaque owners without speculative SDK cleanup");
                std::_Exit(EXIT_SUCCESS);
            } catch (const std::exception& error) {
                std::cerr << "exclusive gate mode " << mode << ": " << error.what() << '\n';
                std::_Exit(EXIT_FAILURE);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "exclusive-gate uncertainty child preserves all ownership guards");
    }
}

void library_calls_keep_typed_results_and_physical_leases() {
    state = {};
    Fixture fixture;
    auto call = std::make_shared<LibraryCall>();
    std::weak_ptr<LibraryCall> lease = call;
    std::uint32_t scalar{73};
    const auto bytes = std::as_bytes(std::span{&scalar, 1});
    expect(!fixture.executor.accept_library(1, {7, bytes, call}).accepted() &&
               !fixture.executor.accept_library(1, {1, bytes, {}}).accepted(),
           "library admission validates exact owned stream and trusted call");
    const auto first = fixture.executor.accept_library(1, {1, bytes, call});
    const auto second = fixture.accept(2, 41);
    const auto frontier = fixture.executor.capture_command_frontier(1);
    expect(first.accepted() && second.accepted() && call->calls == 0 &&
               frontier.status == CUDA_SUCCESS && !fixture.executor.host_result(first.receipt),
           "library acceptance copies arguments but is not original vendor success");
    scalar = 99;
    const auto issued = fixture.executor.issue_next();
    expect(
        issued.state == CudaExecutionIssueState::kIssued && issued.native_status == CUDA_SUCCESS &&
            issued.library_status == 13 && call->calls == 1 && call->value == 73 &&
            state.query_calls == 0 && fixture.executor.host_result(first.receipt)->status == 13 &&
            fixture.executor.host_result(first.receipt)->origin == ExecutionReplyOrigin::kNativeSdk,
        "original library status stays distinct from Driver status and uses owned bytes");
    expect(fixture.executor.request_retreat(1) == CUDA_ERROR_NOT_SUPPORTED &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               !fixture.executor.native_kernel_pair() && state.launch_calls == 1 &&
               state.drain_calls == 0,
           "original stream ordering pipelines a later kernel without inventing retreat or a "
           "kernel pair");
    call.reset();
    expect(fixture.executor.acknowledge_result(first.receipt) &&
               fixture.executor.acknowledge_result(second.receipt) && !lease.expired() &&
               fixture.executor.poll().retained == 2 &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
           "vendor error/ACK do not release potentially live library work");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 2 && lease.expired() &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_SUCCESS &&
               fixture.executor.shutdown(),
           "only actual physical retirement releases original library leases");
}
void library_waits_for_logical_persistent_predecessor() {
    state = {};
    Fixture fixture;
    auto progress = std::make_shared<Progress>();
    auto call = std::make_shared<LibraryCall>();
    const auto low = accept_persistent(fixture, progress);
    const std::uint32_t value{17};
    const auto library =
        fixture.executor.accept_library(2, {1, std::as_bytes(std::span{&value, 1}), call});
    expect(low.accepted() && library.accepted() && fixture.executor.issue_next().sequence == 1 &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
               call->calls == 0,
           "library cannot bypass an issued but logically incomplete persistent producer");
    state.query_status = CUDA_SUCCESS;
    expect(
        fixture.executor.poll().paused == 1 &&
            fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
            call->calls == 0,
        "actual incomplete retreat is not logical completion permission for a library successor");
    progress->is_complete = true;
    expect(fixture.executor.resume(1).state == CudaExecutionIssueState::kIssued &&
               fixture.executor.poll().retired == 1 &&
               fixture.executor.issue_next().sequence == 2 && call->calls == 1 &&
               call->value == 17 && fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(low.receipt) &&
               fixture.executor.acknowledge_result(library.receipt) && fixture.executor.shutdown(),
           "exactly-once final producer completion releases the original library command");
}
void library_preparation_and_record_failures() {
    for (const bool fails_record : {false, true}) {
        state = {};
        Fixture fixture;
        auto call = std::make_shared<LibraryCall>();
        std::uint32_t scalar{5};
        const auto accepted =
            fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&scalar, 1}), call});
        expect(accepted.accepted(), "accept original library operation");
        if (fails_record)
            state.record_status = CUDA_ERROR_INVALID_HANDLE;
        else
            state.create_status = CUDA_ERROR_OUT_OF_MEMORY;
        const auto issued = fixture.executor.issue_next();
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(reply.has_value(), "publish original or pre-issue rejection once");
        if (fails_record) {
            expect(issued.library_status == 13 && call->calls == 1 && reply->status == 13 &&
                       reply->origin == ExecutionReplyOrigin::kNativeSdk &&
                       fixture.executor.poll().retained == 1 && state.query_calls == 0 &&
                       fixture.executor.issue_next().state == CudaExecutionIssueState::kError,
                   "failed trailing record does not overwrite SDK status, prove stop or replay");
            expect(fixture.executor.drain_issued().retired == 1 && call->calls == 1,
                   "explicit real stream fence settles uncertain library work once");
        } else {
            expect(!issued.library_status && call->calls == 0 &&
                       reply->status == CUDA_ERROR_OUT_OF_MEMORY &&
                       reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue,
                   "event preparation failure is not a vendor return or library call");
        }
        expect(fixture.executor.acknowledge_result(accepted.receipt) && fixture.executor.shutdown(),
               "settled library error releases its receipt");
    }
}
void library_known_completion_error_still_retires() {
    for (const bool fails_record : {false, true}) {
        state = {};
        Fixture fixture;
        auto call = std::make_shared<LibraryCall>();
        call->completed.status = CUDA_ERROR_UNKNOWN;
        std::weak_ptr<LibraryCall> lease = call;
        const std::uint32_t scalar{11};
        const auto accepted =
            fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&scalar, 1}), call});
        const auto frontier = fixture.executor.capture_command_frontier(1);
        expect(accepted.accepted() && frontier.status == CUDA_SUCCESS,
               "accept trusted call before malformed native output");
        if (fails_record) state.record_status = CUDA_ERROR_INVALID_HANDLE;
        const auto issued = fixture.executor.issue_next();
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(issued.state == CudaExecutionIssueState::kIssued && issued.library_status == 13 &&
                   issued.native_status == CUDA_SUCCESS &&
                   issued.completion_status == CUDA_ERROR_UNKNOWN && reply &&
                   reply->origin == ExecutionReplyOrigin::kNativeSdk && reply->status == 13 &&
                   call->calls == 1 && call->completion_calls == 1 && state.record_calls == 1 &&
                   fixture.executor.snapshot().is_quarantined &&
                   fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_UNKNOWN,
               "known completion rejection preserves native status and records the owned event");
        call.reset();
        expect(fixture.executor.acknowledge_result(accepted.receipt) && !lease.expired() &&
                   !fixture.accept(2, 9).accepted(),
               "malformed outputs quarantine new work without dropping physical leases");
        state.query_status = CUDA_SUCCESS;
        if (fails_record)
            expect(fixture.executor.poll().retained == 1 && state.query_calls == 0 &&
                       fixture.executor.drain_issued().retired == 1 && state.drain_calls == 1,
                   "known ownership still permits real stream recovery after record failure");
        else
            expect(fixture.executor.poll().retired == 1 && state.query_calls == 1 &&
                       state.drain_calls == 0,
                   "known malformed outputs retire behind the actual trailing event");
        expect(lease.expired() && state.destroy_calls == 1 &&
                   fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_UNKNOWN &&
                   fixture.executor.shutdown(),
               "physical retirement releases known leases without erasing logical failure");
    }
}
void library_uncertain_completion_is_never_recovered() {
    for (const bool reports_success : {false, true}) {
        const auto child = fork();
        expect(child >= 0, "create isolated opaque-library physical uncertainty");
        if (child == 0) {
            try {
                state = {};
                auto fixture = std::make_unique<Fixture>();
                auto call = std::make_shared<LibraryCall>();
                call->completed = {reports_success ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT,
                                   true};
                call->should_change_context = true;
                call->result = reports_success ? 0 : 13;
                std::weak_ptr<LibraryCall> lease = call;
                const std::uint32_t scalar{17};
                const auto accepted = fixture->executor.accept_library(
                    1, {1, std::as_bytes(std::span{&scalar, 1}), call});
                expect(accepted.accepted(), "accept original opaque-work call");
                const auto issued = fixture->executor.issue_next();
                const auto reply = fixture->executor.host_result(accepted.receipt);
                expect(
                    issued.state == CudaExecutionIssueState::kIssued &&
                        issued.library_status == call->result &&
                        issued.completion_status ==
                            (reports_success ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_CONTEXT) &&
                        reply && reply->origin == ExecutionReplyOrigin::kNativeSdk &&
                        reply->status == call->result && call->calls == 1 &&
                        call->completion_calls == 1 && state.record_calls == 0 &&
                        fixture->executor.snapshot().is_quarantined,
                    "uncertain library proof preserves SDK status and skips unsafe recording");
                state.is_wrong_context = false;
                state.query_status = CUDA_SUCCESS;
                call.reset();
                expect(
                    fixture->executor.acknowledge_result(accepted.receipt) &&
                        fixture->executor.poll().retained == 1 &&
                        fixture->executor.drain_issued().retained == 1 &&
                        fixture->executor.poll().retired == 0 &&
                        fixture->executor.issue_next().state == CudaExecutionIssueState::kError &&
                        !fixture->executor.shutdown() && !lease.expired() &&
                        fixture->executor.snapshot().retained_parameter_bytes == sizeof(scalar) &&
                        state.record_calls == 0 && state.query_calls == 0 &&
                        state.drain_calls == 0 && state.destroy_calls == 0,
                    "later restored context, event success and drain cannot erase uncertainty");
                fixture.reset();
                expect(!lease.expired() && state.destroy_calls == 0,
                       "uncertain destruction retains private event and library leases");
                _exit(0);
            } catch (const std::exception& error) {
                std::cerr << "uncertain library mode " << reports_success << ": " << error.what()
                          << '\n';
                _exit(1);
            }
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "opaque-library completion uncertainty remains permanently quarantined");
    }
}

struct PrivateResourceState {
    unsigned int issues{}, native_calls{}, acquisitions{}, cleanups{}, destructors{};
    bool is_live{}, wrong_cleanup_context{};
    CUresult cleanup_status{CUDA_SUCCESS};
    std::function<void()> on_issue, on_cleanup;
};
struct PrivateResourceCall final : CudaExecutionLibraryCall {
    explicit PrivateResourceCall(std::shared_ptr<PrivateResourceState> value)
        : observed(std::move(value)) {}
    ~PrivateResourceCall() override {
        // Destruction intentionally does no native cleanup, even after failed
        // partial construction. Only the explicit physical-stop hook may do so.
        ++observed->destructors;
    }
    std::shared_ptr<PrivateResourceState> observed;
    CudaExecutionLibraryCompletion completed;
    std::int32_t native_status{};
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        ++observed->issues;
        ++observed->acquisitions;
        observed->is_live = true;
        observed->native_calls += completed.was_native_issued;
        if (observed->on_issue) observed->on_issue();
        return native_status;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completed;
    }
    CUresult cleanup_after_physical_stop() noexcept override {
        ++observed->cleanups;
        observed->wrong_cleanup_context |= state.is_wrong_context;
        if (observed->on_cleanup) observed->on_cleanup();
        if (observed->cleanup_status == CUDA_SUCCESS) observed->is_live = false;
        return observed->cleanup_status;
    }
};

void private_library_resources_retire_after_physical_stop() {
    for (unsigned int mode = 0; mode != 5; ++mode) {
        state = {};
        Fixture fixture;
        auto observed = std::make_shared<PrivateResourceState>();
        auto call = std::make_shared<PrivateResourceCall>(observed);
        std::weak_ptr<PrivateResourceCall> lease = call;
        if (mode == 1) call->native_status = 2000;
        if (mode == 2) call->completed = {CUDA_ERROR_OUT_OF_MEMORY, false, false};
        const std::uint32_t value{17};
        const auto accepted =
            fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
        expect(accepted.accepted() && !observed->acquisitions && !observed->cleanups,
               "queue admission constructs no private native resource");
        if (mode == 3) {
            expect(fixture.executor.reject_queued(CUDA_ERROR_DEINITIALIZED) == 1 &&
                       fixture.executor.host_result(accepted.receipt)->origin ==
                           ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       fixture.executor.acknowledge_result(accepted.receipt),
                   "pre-issue cancellation publishes no original native result");
            call.reset();
            expect(lease.expired() && observed->destructors == 1 && !observed->issues &&
                       !observed->cleanups && fixture.executor.shutdown(),
                   "cancellation destroys only CPU state without construction or cleanup hooks");
            continue;
        }
        if (mode == 4) state.record_status = CUDA_ERROR_INVALID_HANDLE;
        const auto issued = fixture.executor.issue_next();
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(reply && observed->issues == 1 && observed->acquisitions == 1 && observed->is_live &&
                   !observed->cleanups,
               "private construction inside issue does not retire at the host result");
        if (mode == 2)
            expect(issued.state == CudaExecutionIssueState::kRejected && !issued.library_status &&
                       reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       reply->status == CUDA_ERROR_OUT_OF_MEMORY && !observed->native_calls,
                   "partial preparation retains resources without inventing requested SDK status");
        else
            expect(issued.library_status == (mode == 1 ? 2000 : 0) &&
                       reply->origin == ExecutionReplyOrigin::kNativeSdk &&
                       reply->status == (mode == 1 ? 2000 : 0),
                   "private cleanup cannot replace an exact original success or failure");
        call.reset();
        expect(fixture.executor.acknowledge_result(accepted.receipt) &&
                   fixture.executor.poll().retained == 1 && !lease.expired() &&
                   !observed->cleanups && !observed->destructors,
               "ACK and pending event retain private resources and all call leases");
        state.query_status = CUDA_SUCCESS;
        const auto stopped = mode == 4 ? fixture.executor.drain_issued() : fixture.executor.poll();
        expect(stopped.retired == 1 && observed->cleanups == 1 && !observed->is_live &&
                   !observed->wrong_cleanup_context && observed->destructors == 1 &&
                   lease.expired() && state.destroy_calls == 1 &&
                   state.drain_calls == (mode == 4 ? 1U : 0U),
               "only proven stop runs cleanup once on the owner before releasing CPU leases");
        expect(!fixture.executor.poll().retired && !fixture.executor.drain_issued().retired &&
                   observed->cleanups == 1 && fixture.executor.shutdown(),
               "subsequent polling and shutdown never replay successful cleanup");
    }
}

void physical_quiescence_preserves_logical_quarantine() {
    for (const bool acknowledges_first : {false, true}) {
        state = {};
        Fixture fixture;
        expect(
            fixture.executor.physical_readers_quiescent() && fixture.executor.resources_quiescent(),
            "new owned dispatcher has neither physical readers nor logical quarantine");
        auto observed = std::make_shared<PrivateResourceState>();
        auto call = std::make_shared<PrivateResourceCall>(observed);
        call->completed = {CUDA_ERROR_OUT_OF_MEMORY, false, false};
        const std::uint32_t value{47};
        const auto accepted =
            fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
        expect(accepted.accepted() && !fixture.executor.physical_readers_quiescent(),
               "a queued command already owns physical-resource leases before native issue");
        const auto issued = fixture.executor.issue_next();
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(issued.state == CudaExecutionIssueState::kRejected && !issued.library_status &&
                   reply && reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   reply->status == CUDA_ERROR_OUT_OF_MEMORY && observed->is_live &&
                   !observed->native_calls && fixture.executor.snapshot().is_quarantined &&
                   !fixture.executor.physical_readers_quiescent(),
               "partial private preparation has real readers despite no requested native call");
        if (acknowledges_first)
            expect(fixture.executor.acknowledge_result(accepted.receipt),
                   "acknowledge private preparation failure before physical completion");
        expect(fixture.executor.poll().retained == 1 && !observed->cleanups &&
                   !fixture.executor.physical_readers_quiescent(),
               "ACK and pending-event observation cannot authorize quarantined cleanup");
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().retired == 1 && observed->cleanups == 1 &&
                   !observed->is_live && state.destroy_calls == 1,
               "physical stop and exactly-once private cleanup precede cleanup-only permission");
        if (!acknowledges_first) {
            expect(fixture.executor.snapshot().outstanding == 1 &&
                       !fixture.executor.physical_readers_quiescent(),
                   "empty physical records alone do not discard an unacknowledged queue receipt");
            expect(fixture.executor.acknowledge_result(accepted.receipt),
                   "acknowledge already physically cleaned preparation failure");
        }
        expect(fixture.executor.physical_readers_quiescent() &&
                   !fixture.executor.resources_quiescent() &&
                   fixture.executor.snapshot().is_quarantined && !fixture.accept(2, 53).accepted(),
               "cleanup-only physical proof never clears logical quarantine or admits new work");
        unsigned int callbacks{};
        state.on_context = [&] {
            ++callbacks;
            expect(!fixture.executor.physical_readers_quiescent(),
                   "even an empty dispatcher cannot grant cleanup from inside a native callback");
        };
        expect(!fixture.executor.poll().retired && callbacks == 1,
               "empty-record callback scope is tested independently of live reader rejection");
        state.on_context = {};
        expect(fixture.executor.physical_readers_quiescent() && fixture.executor.shutdown() &&
                   observed->cleanups == 1,
               "known clean quarantine may shut down without replay or repeated private cleanup");
    }
}

void clean_preparation_keeps_physical_ownership() {
    expect(!CudaExecutionLibraryCompletion{}.is_clean_preparation_rejection,
           "existing factories retain conservative completion classification by default");
    for (const bool acknowledges_first : {false, true}) {
        state = {};
        Fixture fixture(true);
        auto observed = std::make_shared<PrivateResourceState>();
        auto call = std::make_shared<PrivateResourceCall>(observed);
        call->completed = {CUDA_ERROR_INVALID_VALUE, false, false, true};
        call->native_status = 73;  // Not provenance: no requested library call occurred.
        std::weak_ptr<PrivateResourceCall> lease = call;
        const std::uint32_t value{59};
        const auto accepted =
            fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
        const auto frontier = fixture.executor.capture_command_frontier(1);
        unsigned int issue_callbacks{}, proof_callbacks{}, cleanup_callbacks{};
        const auto check_reentry = [&] {
            const auto context_calls = state.context_calls;
            expect(fixture.executor.issue_next().native_status == CUDA_ERROR_NOT_READY &&
                       fixture.executor.poll().status == CUDA_ERROR_NOT_READY &&
                       fixture.executor.drain_issued().status == CUDA_ERROR_NOT_READY &&
                       fixture.executor.resume(1).native_status == CUDA_ERROR_NOT_READY &&
                       fixture.executor.request_retreat(1) == CUDA_ERROR_NOT_READY &&
                       !fixture.accept(3, 61).accepted() &&
                       !fixture.executor.reject_queued(CUDA_ERROR_DEINITIALIZED) &&
                       !fixture.executor.physical_readers_quiescent() &&
                       !fixture.executor.shutdown() && state.context_calls == context_calls,
                   "clean preparation and its independent proof exclude all nested SDK work");
        };
        observed->on_issue = [&] {
            ++issue_callbacks;
            check_reentry();
        };
        observed->on_cleanup = [&] {
            ++cleanup_callbacks;
            expect(state.destroy_calls >= 1 && !state.events[0].is_live,
                   "private cleanup follows destruction of the proven trailing event");
            check_reentry();
        };
        state.on_context = [&] {
            if (observed->issues && !state.record_calls) {
                ++proof_callbacks;
                check_reentry();
            }
        };
        const auto issued = fixture.executor.issue_next();
        state.on_context = {};
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(accepted.accepted() && frontier.status == CUDA_SUCCESS &&
                   issued.state == CudaExecutionIssueState::kRejected && issued.sequence == 1 &&
                   issued.native_status == CUDA_ERROR_INVALID_VALUE &&
                   issued.completion_status == CUDA_SUCCESS && !issued.library_status && reply &&
                   reply->status == CUDA_ERROR_INVALID_VALUE &&
                   reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   issue_callbacks == 1 && proof_callbacks >= 1 && !cleanup_callbacks &&
                   observed->issues == 1 && !observed->native_calls && observed->is_live &&
                   !fixture.executor.snapshot().is_quarantined && state.record_calls == 1 &&
                   !state.query_calls && !state.drain_calls && !state.launch_calls &&
                   frontier.frontier.observe() == CUDA_ERROR_NOT_READY,
               "clean rejection publishes only preparation failure and retains a pending event");
        call.reset();
        if (acknowledges_first)
            expect(fixture.executor.acknowledge_result(accepted.receipt),
                   "acknowledge the clean preparation error before its actual physical stop");
        expect(fixture.executor.poll().retained == 1 && !lease.expired() && !observed->cleanups &&
                   !observed->destructors &&
                   fixture.executor.snapshot().retained_parameter_bytes == sizeof(value) &&
                   !fixture.executor.snapshot().is_quarantined &&
                   fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
               "pending event and ACK cannot release known preparation resources");

        const std::uint32_t next_value{67};
        const auto successor = fixture.executor.accept_kernel(
            2, {.function = 1,
                .stream = 0,
                .grid = {80, 1, 1},
                .block = {32, 1, 1},
                .parameters = std::as_bytes(std::span{&next_value, 1})});
        expect(successor.accepted() &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
                   state.launch_calls == 1 && state.values.back() == 67 && !lease.expired(),
               "a healthy successor can issue without releasing the rejected command's leases");
        state.events[1].is_ready = true;
        expect(fixture.executor.poll().retired == 1 &&
                   fixture.executor.acknowledge_result(successor.receipt) && !lease.expired() &&
                   !observed->cleanups && frontier.frontier.observe() == CUDA_ERROR_NOT_READY,
               "a successor stop is not a substitute for this command's own stop proof");
        state.events[0].is_ready = true;
        expect(fixture.executor.poll().retired == 1 && observed->cleanups == 1 &&
                   cleanup_callbacks == 1 && !observed->is_live &&
                   !observed->wrong_cleanup_context && lease.expired() &&
                   observed->destructors == 1 && frontier.frontier.observe() == CUDA_SUCCESS &&
                   !fixture.executor.snapshot().is_quarantined,
               "proven stop cleans once and settles a healthy frontier, not a native success");
        if (!acknowledges_first) {
            const auto retained_reply = fixture.executor.host_result(accepted.receipt);
            expect(retained_reply && retained_reply->status == CUDA_ERROR_INVALID_VALUE &&
                       retained_reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       fixture.executor.acknowledge_result(accepted.receipt),
                   "physical retirement does not overwrite or acknowledge the preparation reply");
        }
        expect(!fixture.executor.poll().retired && fixture.executor.resources_quiescent() &&
                   observed->cleanups == 1 && state.destroy_calls == 2 &&
                   fixture.executor.shutdown(),
               "clean rejection leaves a reusable owner without replay or duplicate cleanup");
    }
}

void canceled_clean_preparation_never_enters_library() {
    state = {};
    Fixture fixture(true);
    auto observed = std::make_shared<PrivateResourceState>();
    auto call = std::make_shared<PrivateResourceCall>(observed);
    call->completed = {CUDA_ERROR_INVALID_VALUE, false, false, true};
    std::weak_ptr<PrivateResourceCall> lease = call;
    const std::uint32_t value{71};
    const auto accepted =
        fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
    const auto frontier = fixture.executor.capture_command_frontier(1);
    const auto dependent = fixture.executor.accept_enqueue_turn(1, fixture.resources);
    const auto context_calls = state.context_calls;
    expect(accepted.accepted() && frontier.status == CUDA_SUCCESS &&
               dependent.status == CUDA_SUCCESS &&
               fixture.executor.reject_queued(CUDA_ERROR_DEINITIALIZED) == 1,
           "queued cancellation does not consult or enter the clean-preparation callback");
    const auto reply = fixture.executor.host_result(accepted.receipt);
    const auto retired = frontier.frontier.retirement();
    call.reset();
    expect(reply && reply->status == CUDA_ERROR_DEINITIALIZED &&
               reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               fixture.executor.acknowledge_result(accepted.receipt) && lease.expired() &&
               !observed->issues && !observed->acquisitions && !observed->cleanups &&
               observed->destructors == 1 && !state.record_calls && !state.destroy_calls &&
               frontier.frontier.observe() == CUDA_ERROR_DEINITIALIZED &&
               retired.status == CUDA_ERROR_DEINITIALIZED && retired.is_retired &&
               !fixture.executor.snapshot().is_quarantined,
           "cancellation releases only CPU state and publishes no requested SDK provenance");
    const auto dependency = fixture.executor.inspect_enqueue_dependency(dependent.turn);
    const auto progress = fixture.executor.progress_enqueue_dependency(dependent.turn);
    expect(
        dependency.status == CUDA_ERROR_DEINITIALIZED && dependency.has_failure_observation &&
            dependency.has_retired_failure && !dependency.foreign_blocker &&
            fixture.executor.enqueue_turn_ready(dependent.turn) == CUDA_ERROR_DEINITIALIZED &&
            fixture.executor.begin_enqueue_turn(dependent.turn) == CUDA_ERROR_DEINITIALIZED &&
            progress.state == CudaExecutionIssueState::kError &&
            progress.native_status == CUDA_ERROR_DEINITIALIZED && !progress.library_status &&
            fixture.executor.cancel_enqueue_turn(dependent.turn) == CUDA_SUCCESS &&
            dependent.frontier.observe() == CUDA_SUCCESS &&
            dependent.frontier.retirement().is_retired && state.context_calls == context_calls &&
            !state.launch_calls && !state.record_calls && !state.destroy_calls,
        "a pre-admitted dependent turn retains exact cancellation but can cancel without SDK work");
    const std::uint32_t next_value{73};
    const auto successor =
        fixture.executor.accept_kernel(2, {.function = 1,
                                           .stream = 0,
                                           .grid = {80, 1, 1},
                                           .block = {32, 1, 1},
                                           .parameters = std::as_bytes(std::span{&next_value, 1})});
    const auto issued = fixture.executor.issue_next();
    const auto successor_reply = fixture.executor.host_result(successor.receipt);
    expect(successor.accepted() && issued.state == CudaExecutionIssueState::kIssued &&
               issued.sequence == 2 && issued.native_status == CUDA_SUCCESS &&
               issued.completion_status == CUDA_SUCCESS && successor_reply &&
               successor_reply->origin == ExecutionReplyOrigin::kNativeSdk &&
               successor_reply->status == CUDA_SUCCESS && state.launch_calls == 1 &&
               state.values == std::vector<std::uint32_t>{next_value} && state.record_calls == 1 &&
               !observed->issues && !observed->acquisitions && !observed->cleanups &&
               frontier.frontier.observe() == CUDA_ERROR_DEINITIALIZED,
           "a later unrelated original issues once without erasing the canceled dependency error");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 && state.destroy_calls == 1 && !state.drain_calls &&
               fixture.executor.acknowledge_result(successor.receipt) &&
               frontier.frontier.retirement().is_retired && fixture.executor.shutdown(),
           "healthy successor retirement is independent of canceled requested SDK provenance");
}

void isolated_clean_preparation(const std::function<void()>& test, std::string_view label) {
    const auto child = fork();
    expect(child >= 0, "fork bounded clean-preparation uncertainty case");
    if (!child) {
        try {
            test();
            _exit(0);
        } catch (const std::exception& error) {
            std::cerr << label << ": " << error.what() << '\n';
            _exit(1);
        }
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status), label);
}

void contradictory_clean_preparation_flags() {
    for (const bool was_native_issued : {false, true})
        for (const bool is_cleanup_uncertain : {false, true})
            for (const auto preparation_status : {CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE}) {
                if (!was_native_issued && !is_cleanup_uncertain &&
                    preparation_status != CUDA_SUCCESS)
                    continue;  // The sole valid clean-rejection tuple is covered independently.
                for (const std::int32_t original_status : {0, 13}) {
                    if (!was_native_issued && original_status) continue;
                    isolated_clean_preparation(
                        [&] {
                            state = {};
                            auto fixture = std::make_unique<Fixture>();
                            auto observed = std::make_shared<PrivateResourceState>();
                            auto call = std::make_shared<PrivateResourceCall>(observed);
                            call->completed = {preparation_status, is_cleanup_uncertain,
                                               was_native_issued, true};
                            call->native_status = original_status;
                            std::weak_ptr<PrivateResourceCall> lease = call;
                            const std::uint32_t value{73};
                            const auto accepted = fixture->executor.accept_library(
                                1, {1, std::as_bytes(std::span{&value, 1}), call});
                            const auto frontier = fixture->executor.capture_command_frontier(1);
                            const auto issued = fixture->executor.issue_next();
                            const auto reply = fixture->executor.host_result(accepted.receipt);
                            expect(accepted.accepted() && reply &&
                                       fixture->executor.snapshot().is_quarantined &&
                                       issued.completion_status != CUDA_SUCCESS &&
                                       observed->native_calls == (was_native_issued ? 1U : 0U) &&
                                       !state.record_calls && !observed->cleanups,
                                   "contradictory clean flags never authorize an event or cleanup");
                            if (was_native_issued)
                                expect(issued.state == CudaExecutionIssueState::kIssued &&
                                           issued.library_status == original_status &&
                                           reply->status == original_status &&
                                           reply->origin == ExecutionReplyOrigin::kNativeSdk,
                                       "a contradictory flag cannot erase an observed SDK result");
                            else
                                expect(
                                    issued.state == CudaExecutionIssueState::kRejected &&
                                        !issued.library_status && reply->status != 0 &&
                                        reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue,
                                    "a contradictory flag cannot invent native provenance");
                            call.reset();
                            state.query_status = CUDA_SUCCESS;
                            expect(fixture->executor.acknowledge_result(accepted.receipt) &&
                                       fixture->executor.poll().retained == 1 &&
                                       fixture->executor.drain_issued().retained == 1 &&
                                       frontier.frontier.observe() != CUDA_SUCCESS &&
                                       !fixture->accept(2, 79).accepted() && !lease.expired() &&
                                       !observed->cleanups && !observed->destructors &&
                                       !state.query_calls && !state.drain_calls &&
                                       !state.destroy_calls && !fixture->executor.shutdown(),
                                   "later callback success cannot repair contradictory authority");
                            fixture.reset();
                            expect(!lease.expired() && !observed->destructors,
                                   "uncertain destruction remains CPU-only and retains ownership");
                        },
                        "contradictory clean-preparation flags fail closed");
                }
            }
}

enum class CleanPreparationFault {
    kIssueContext,
    kProofContext,
    kQueryContext,
    kDestroy,
    kDestroyContext,
    kCleanup,
    kCleanupContext
};

void clean_preparation_uncertainty(CleanPreparationFault fault) {
    state = {};
    auto fixture = std::make_unique<Fixture>();
    auto observed = std::make_shared<PrivateResourceState>();
    auto call = std::make_shared<PrivateResourceCall>(observed);
    call->completed = {CUDA_ERROR_INVALID_VALUE, false, false, true};
    std::weak_ptr<PrivateResourceCall> lease = call;
    if (fault == CleanPreparationFault::kIssueContext)
        observed->on_issue = [] { state.is_wrong_context = true; };
    if (fault == CleanPreparationFault::kProofContext)
        state.on_context = [&] {
            if (observed->issues) state.is_wrong_context = true;
        };
    if (fault == CleanPreparationFault::kQueryContext) state.loses_context_on_query = true;
    if (fault == CleanPreparationFault::kDestroy) state.destroy_status = CUDA_ERROR_INVALID_HANDLE;
    if (fault == CleanPreparationFault::kDestroyContext) state.loses_context_on_destroy = true;
    if (fault == CleanPreparationFault::kCleanup)
        observed->cleanup_status = CUDA_ERROR_INVALID_HANDLE;
    if (fault == CleanPreparationFault::kCleanupContext)
        observed->on_cleanup = [] { state.is_wrong_context = true; };
    const std::uint32_t value{83};
    const auto accepted =
        fixture->executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
    const auto frontier = fixture->executor.capture_command_frontier(1);
    const auto issued = fixture->executor.issue_next();
    state.on_context = {};
    const bool failed_before_event = fault == CleanPreparationFault::kIssueContext ||
                                     fault == CleanPreparationFault::kProofContext;
    expect(accepted.accepted() && issued.state == CudaExecutionIssueState::kRejected &&
               issued.native_status == CUDA_ERROR_INVALID_VALUE && !issued.library_status &&
               issued.completion_status ==
                   (failed_before_event ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS) &&
               state.record_calls == (failed_before_event ? 0U : 1U) &&
               frontier.frontier.observe() ==
                   (failed_before_event ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_NOT_READY),
           "independent context proof suppresses recording without replacing preparation status");
    state.is_wrong_context = false;
    state.query_status = CUDA_SUCCESS;
    const auto stopped = fixture->executor.poll();
    const auto reply = fixture->executor.host_result(accepted.receipt);
    const auto cleanup_calls = observed->cleanups;
    const auto destroy_calls = state.destroy_calls;
    expect(!stopped.retired && stopped.retained == 1 && stopped.status != CUDA_SUCCESS && reply &&
               reply->status == CUDA_ERROR_INVALID_VALUE &&
               reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               fixture->executor.snapshot().is_quarantined && !observed->native_calls &&
               !observed->wrong_cleanup_context &&
               cleanup_calls == (fault == CleanPreparationFault::kCleanup ||
                                         fault == CleanPreparationFault::kCleanupContext
                                     ? 1U
                                     : 0U),
           "post-preparation proof or cleanup failure retains every lease and its exact reply");
    state.is_wrong_context = false;
    state.loses_context_on_query = state.loses_context_on_destroy = false;
    state.destroy_status = observed->cleanup_status = CUDA_SUCCESS;
    observed->on_cleanup = {};
    call.reset();
    const auto query_calls = state.query_calls;
    expect(fixture->executor.acknowledge_result(accepted.receipt) &&
               fixture->executor.poll().retained == 1 &&
               fixture->executor.drain_issued().retained == 1 &&
               frontier.frontier.observe() != CUDA_SUCCESS && !lease.expired() &&
               observed->cleanups == cleanup_calls && state.destroy_calls == destroy_calls &&
               state.query_calls == query_calls && !state.drain_calls &&
               !fixture->executor.physical_readers_quiescent() && !fixture->executor.shutdown(),
           "restoring callback outcomes never retries uncertain native cleanup or stop proof");
    fixture.reset();
    expect(!lease.expired() && !observed->destructors && observed->cleanups == cleanup_calls,
           "uncertain clean-preparation records survive CPU-only dispatcher destruction");
}

void clean_preparation_failure_paths() {
    for (const auto fault :
         {CleanPreparationFault::kIssueContext, CleanPreparationFault::kProofContext,
          CleanPreparationFault::kQueryContext, CleanPreparationFault::kDestroy,
          CleanPreparationFault::kDestroyContext, CleanPreparationFault::kCleanup,
          CleanPreparationFault::kCleanupContext})
        isolated_clean_preparation([&] { clean_preparation_uncertainty(fault); },
                                   "clean preparation retains uncertain physical ownership");
    for (const bool fails_record : {false, true}) {
        state = {};
        Fixture fixture;
        auto observed = std::make_shared<PrivateResourceState>();
        auto call = std::make_shared<PrivateResourceCall>(observed);
        call->completed = {CUDA_ERROR_INVALID_VALUE, false, false, true};
        std::weak_ptr<PrivateResourceCall> lease = call;
        if (fails_record)
            state.record_status = CUDA_ERROR_INVALID_HANDLE;
        else
            state.query_status = CUDA_ERROR_LAUNCH_FAILED;
        const std::uint32_t value{89};
        const auto accepted =
            fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
        const auto frontier = fixture.executor.capture_command_frontier(1);
        const auto issued = fixture.executor.issue_next();
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(accepted.accepted() &&
                   issued.completion_status ==
                       (fails_record ? CUDA_ERROR_INVALID_HANDLE : CUDA_SUCCESS) &&
                   !issued.library_status && reply && reply->status == CUDA_ERROR_INVALID_VALUE &&
                   reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue,
               "clean classification does not suppress a trailing-event record failure");
        call.reset();
        const auto observed_stop = fixture.executor.poll();
        expect(observed_stop.retained == 1 && !observed_stop.retired &&
                   observed_stop.status ==
                       (fails_record ? CUDA_ERROR_INVALID_HANDLE : CUDA_ERROR_LAUNCH_FAILED) &&
                   fixture.executor.snapshot().is_quarantined && !observed->cleanups &&
                   !lease.expired() && state.query_calls == (fails_record ? 0U : 1U) &&
                   fixture.executor.acknowledge_result(accepted.receipt),
               "failed event proof retains leases and never queries a never-recorded event");
        state.query_status = CUDA_SUCCESS;
        if (fails_record) {
            state.drain_status = CUDA_ERROR_LAUNCH_FAILED;
            expect(!fixture.executor.drain_issued().retired && !observed->cleanups,
                   "a failed recovery drain supplies no stop or cleanup authority");
            state.drain_status = CUDA_SUCCESS;
        }
        const auto recovered =
            fails_record ? fixture.executor.drain_issued() : fixture.executor.poll();
        expect(recovered.retired == 1 && observed->cleanups == 1 && lease.expired() &&
                   observed->destructors == 1 && !observed->native_calls &&
                   frontier.frontier.observe() != CUDA_SUCCESS &&
                   fixture.executor.snapshot().is_quarantined &&
                   !fixture.accept(2, 97).accepted() && !fixture.executor.poll().retired &&
                   observed->cleanups == 1 && fixture.executor.shutdown(),
               "known-owner recovery releases resources but never clears a real physical fault");
    }
}

void private_library_cleanup_uncertainty(unsigned int mode) {
    state = {};
    auto fixture = std::make_unique<Fixture>();
    auto observed = std::make_shared<PrivateResourceState>();
    auto call = std::make_shared<PrivateResourceCall>(observed);
    std::weak_ptr<PrivateResourceCall> lease = call;
    if (mode == 0) observed->cleanup_status = CUDA_ERROR_INVALID_HANDLE;
    if (mode == 1) observed->on_cleanup = [] { state.is_wrong_context = true; };
    if (mode == 2) state.loses_context_on_query = true;
    if (mode == 3) state.destroy_status = CUDA_ERROR_INVALID_HANDLE;
    if (mode == 4) call->completed = {CUDA_ERROR_UNKNOWN, true, true};
    if (mode == 5) state.loses_context_on_destroy = true;
    const std::uint32_t value{23};
    const auto accepted =
        fixture->executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
    expect(accepted.accepted() && fixture->executor.issue_next().library_status == 0,
           "issued original success precedes independent private-resource failure");
    state.query_status = CUDA_SUCCESS;
    const auto first = fixture->executor.poll();
    const auto reply = fixture->executor.host_result(accepted.receipt);
    const auto cleanup_count = observed->cleanups;
    expect(!first.retired && first.retained == 1 && reply && reply->status == 0 &&
               reply->origin == ExecutionReplyOrigin::kNativeSdk &&
               fixture->executor.snapshot().is_quarantined &&
               !fixture->executor.physical_readers_quiescent() &&
               cleanup_count == (mode < 2 ? 1U : 0U) && !observed->wrong_cleanup_context,
           "failure/proof loss retains ownership without replacing original host provenance");
    state.is_wrong_context = false;
    state.loses_context_on_query = false;
    state.loses_context_on_destroy = false;
    state.destroy_status = CUDA_SUCCESS;
    observed->cleanup_status = CUDA_SUCCESS;
    observed->on_cleanup = {};
    call.reset();
    expect(fixture->executor.acknowledge_result(accepted.receipt) &&
               !fixture->executor.poll().retired && !fixture->executor.drain_issued().retired &&
               !fixture->executor.physical_readers_quiescent() && !fixture->executor.shutdown() &&
               !lease.expired() && observed->cleanups == cleanup_count && !observed->destructors,
           "later event/context/cleanup success cannot retry a destructive private operation");
    fixture.reset();
    expect(!lease.expired() && observed->cleanups == cleanup_count && !observed->destructors,
           "dispatcher destruction retains uncertain private resources without native destructor");
}

void private_library_cleanup_failures_are_isolated() {
    for (unsigned int mode = 0; mode != 6; ++mode) {
        const auto child = fork();
        expect(child >= 0, "fork bounded private cleanup uncertainty");
        if (!child) {
            try {
                private_library_cleanup_uncertainty(mode);
                _exit(0);
            } catch (const std::exception& error) {
                std::cerr << "private cleanup case " << mode << ": " << error.what() << '\n';
                _exit(1);
            }
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
               "isolated private cleanup ownership remains retained");
    }
}

void private_library_cleanup_reentry_keeps_records_stable() {
    state = {};
    Fixture fixture;
    auto observed = std::make_shared<PrivateResourceState>();
    auto call = std::make_shared<PrivateResourceCall>(observed);
    const std::uint32_t value{29};
    const auto first =
        fixture.executor.accept_library(1, {1, std::as_bytes(std::span{&value, 1}), call});
    const auto second = fixture.accept(2, 31);
    bool protected_reentry{};
    observed->on_cleanup = [&] {
        const auto native_calls = state.context_calls;
        protected_reentry = fixture.executor.issue_next().native_status == CUDA_ERROR_NOT_READY &&
                            fixture.executor.poll().status == CUDA_ERROR_NOT_READY &&
                            fixture.executor.drain_issued().status == CUDA_ERROR_NOT_READY &&
                            fixture.executor.resume(1).native_status == CUDA_ERROR_NOT_READY &&
                            fixture.executor.request_retreat(1) == CUDA_ERROR_NOT_READY &&
                            !fixture.accept(3, 37).accepted() &&
                            !fixture.executor.reject_queued(CUDA_ERROR_DEINITIALIZED) &&
                            !fixture.executor.physical_readers_quiescent() &&
                            !fixture.executor.shutdown() && !state.launch_calls &&
                            state.context_calls == native_calls;
    };
    expect(first.accepted() && second.accepted() && fixture.executor.issue_next().sequence == 1,
           "prepare a private cleanup callback and still-queued independent successor");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 && protected_reentry && observed->cleanups == 1 &&
               fixture.executor.acknowledge_result(first.receipt) &&
               fixture.executor.issue_next().sequence == 2 && state.launch_calls == 1 &&
               fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(second.receipt) && fixture.executor.shutdown(),
           "cleanup reentry neither consumes the next queue token nor invalidates active records");
}
void native_context_validation_excludes_reentry() {
    state = {};
    Fixture fixture;
    const auto accepted = fixture.accept(1, 41);
    unsigned int callbacks{};
    state.on_context = [&] {
        ++callbacks;
        const auto calls = state.context_calls;
        expect(fixture.executor.issue_next().native_status == CUDA_ERROR_NOT_READY &&
                   fixture.executor.poll().status == CUDA_ERROR_NOT_READY &&
                   fixture.executor.drain_issued().status == CUDA_ERROR_NOT_READY &&
                   !fixture.accept(2, 43).accepted() &&
                   !fixture.executor.reject_queued(CUDA_ERROR_DEINITIALIZED) &&
                   !fixture.executor.physical_readers_quiescent() && !fixture.executor.shutdown() &&
                   state.context_calls == calls,
               "callback scope begins before the first native context observation");
    };
    expect(accepted.accepted() && fixture.executor.issue_next().sequence == 1 && callbacks,
           "context-validation callback cannot recursively consume the original queue token");
    state.on_context = {};
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(accepted.receipt) && fixture.executor.shutdown(),
           "outer operation completes normally after blocked context-observation reentry");
}
void pre_native_rejection() {
    for (unsigned int mode = 0; mode < 3; ++mode) {
        state = {};
        Fixture fixture;
        state.create_status = mode == 1 ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
        state.returns_null_event = mode == 1;
        state.returns_error_handle = mode == 2;
        const auto accepted = fixture.accept(1, 9);
        expect(accepted.accepted(), "accept command before pre-native error");
        expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kRejected &&
                   state.launch_calls == 0 && state.record_calls == 0,
               "failed/malformed event creation cannot invoke original work");
        const auto reply = fixture.executor.host_result(accepted.receipt);
        expect(reply && reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   reply->status == (mode == 1 ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_OUT_OF_MEMORY) &&
                   state.destroy_calls == (mode == 2 ? 1U : 0U) &&
                   fixture.executor.snapshot().retained_parameter_bytes == 0 &&
                   fixture.executor.acknowledge_result(accepted.receipt) &&
                   fixture.executor.shutdown(),
               "pre-native error is explicitly tagged, retired and acknowledged once");
    }
}

void failed_record_never_queries_unrecorded_event() {
    state = {};
    Fixture fixture;
    state.launch_status = CUDA_ERROR_INVALID_VALUE;
    state.record_status = CUDA_ERROR_UNKNOWN;
    const auto accepted = fixture.accept(1, 1);
    expect(accepted.accepted(), "accept failed-record work");
    const auto issue = fixture.executor.issue_next();
    expect(issue.native_status == CUDA_ERROR_INVALID_VALUE &&
               issue.completion_status == CUDA_ERROR_UNKNOWN &&
               fixture.executor.snapshot().is_quarantined,
           "preserve original and trailing errors separately");
    expect(fixture.executor.poll().retired == 0 && state.query_calls == 0 &&
               fixture.executor.snapshot().retained_parameter_bytes == 4,
           "never-recorded event SUCCESS must not authorize early input release");
    state.drain_status = CUDA_ERROR_LAUNCH_FAILED;
    expect(fixture.executor.drain_issued().retired == 0 && state.destroy_calls == 0,
           "failed stream drain is not physical stop");
    state.drain_status = CUDA_SUCCESS;
    expect(fixture.executor.drain_issued().retired == 1 && state.destroy_calls == 1,
           "explicit successful recovery drain proves actual stopped execution");
    const auto reply = fixture.executor.host_result(accepted.receipt);
    expect(reply && reply->status == CUDA_ERROR_INVALID_VALUE &&
               reply->origin == ExecutionReplyOrigin::kNativeSdk &&
               fixture.executor.acknowledge_result(accepted.receipt) && fixture.executor.shutdown(),
           "recovery cannot overwrite the original SDK result");
}

void failed_query_and_queued_rejection() {
    state = {};
    Fixture fixture;
    const auto first = fixture.accept(1, 1), second = fixture.accept(2, 2);
    expect(first.accepted() && second.accepted() &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
           "prepare active original and queued successor");
    state.query_status = CUDA_ERROR_LAUNCH_FAILED;
    expect(fixture.executor.poll().status == CUDA_ERROR_LAUNCH_FAILED &&
               fixture.executor.snapshot().is_quarantined && state.destroy_calls == 0 &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kError &&
               fixture.accept(3, 3).status == ExecutionAcceptStatus::kQuarantined,
           "query failure retains input and forbids new acceptance/issue without replay");
    expect(fixture.executor.reject_queued(CUDA_SUCCESS) == 0 &&
               fixture.executor.reject_queued(CUDA_ERROR_DEINITIALIZED) == 1 &&
               fixture.executor.acknowledge_result(second.receipt) && state.launch_calls == 1,
           "queued shutdown rejection must not launch the successor or replay the original");
    expect(fixture.executor.drain_issued().retired == 1 &&
               fixture.executor.acknowledge_result(first.receipt) && fixture.executor.shutdown(),
           "exact independently drained original may retire after quarantine");
}

void owner_context_and_fork_guards() {
    state = {};
    Fixture fixture;
    const auto accepted = fixture.accept(1, 5);
    expect(accepted.accepted(), "prepare owner guards");
    const auto frontier = fixture.executor.capture_frontier();
    expect(frontier.status == CUDA_SUCCESS, "prepare process/executor-bound frontier");
    const auto sdk_before = state.context_calls;
    auto wrong = std::async(std::launch::async, [&] {
        const auto scheduling = fixture.executor.scheduling_snapshot();
        return scheduling.submissions.is_quarantined && scheduling.submissions.is_closing &&
               !scheduling.has_ready_work && scheduling.submissions.outstanding == 0 &&
               fixture.executor.bind_executor() == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.issue_next().native_status == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.poll().status == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.drain_issued().status == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.request_retreat(1) == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.resume(1).native_status == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.capture_frontier().status == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.capture_command_frontier(1).status == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_CONTEXT &&
               !fixture.accept(2, 2).accepted() && !fixture.executor.shutdown();
    });
    expect(wrong.get() && state.context_calls == sdk_before,
           "wrong executor must reject before any native SDK dispatch");
    const auto child = fork();
    expect(child >= 0, "fork isolated inherited dispatcher");
    if (child == 0) {
        const auto scheduling = fixture.executor.scheduling_snapshot();
        const bool guarded =
            scheduling.submissions.is_quarantined && scheduling.submissions.is_closing &&
            !scheduling.has_ready_work && scheduling.submissions.outstanding == 0 &&
            fixture.executor.bind_executor() == CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.issue_next().native_status == CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.poll().status == CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.request_retreat(1) == CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.resume(1).native_status == CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.capture_frontier().status == CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.capture_command_frontier(1).status == CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.query_frontier(frontier.frontier) == CUDA_ERROR_INVALID_CONTEXT &&
            !fixture.executor.shutdown() && state.context_calls == sdk_before;
        fixture.executor.~CudaExecutionDispatcher();
        _exit(guarded && state.context_calls == sdk_before ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork child must not enter inherited SDK or owner state");
    state.is_wrong_context = true;
    expect(fixture.executor.issue_next().native_status == CUDA_ERROR_INVALID_CONTEXT &&
               state.launch_calls == 0,
           "wrong current CUDA context must leave accepted work unissued");
    state.is_wrong_context = false;
    expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
           "restoring the correct current context permits original issue once");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(accepted.receipt) && fixture.executor.shutdown(),
           "guard rejection must not lose or duplicate accepted original work");
}

void immutable_binding_and_input_validation() {
    state = {};
    Fixture fixture;
    expect(!fixture.executor.bind_function(1, function(), 4, fixture.resources) &&
               !fixture.executor.bind_function(0, function(), 4, fixture.resources) &&
               !fixture.executor.bind_function(2, nullptr, 4, fixture.resources) &&
               !fixture.executor.bind_function(2, function(), 4, {}) &&
               !fixture.executor.bind_function(2, function(), 17, fixture.resources) &&
               fixture.executor.bind_function(2, function(), 4, fixture.resources) &&
               !fixture.executor.bind_function(3, function(), 4, fixture.resources) &&
               !fixture.executor.bind_stream(0, stream(), fixture.resources) &&
               !fixture.executor.bind_stream(2, CU_STREAM_PER_THREAD, fixture.resources) &&
               !fixture.executor.bind_stream(2, stream(), fixture.resources),
           "logical bindings are bounded, immutable, nonnull and owner-backed");
    CudaExecutionKernel invalid{
        .function = 1, .stream = 1, .grid = {80, 1, 1}, .block = {32, 1, 1}, .parameters = {}};
    expect(!fixture.executor.accept_kernel(1, invalid).accepted(), "wrong packed ABI size rejects");
    std::uint32_t value = 4;
    invalid.parameters = std::as_bytes(std::span{&value, 1});
    invalid.function = 99;
    expect(!fixture.executor.accept_kernel(1, invalid).accepted(), "foreign function ID rejects");
    invalid.function = 1;
    invalid.stream = 99;
    expect(!fixture.executor.accept_kernel(1, invalid).accepted(), "foreign stream ID rejects");
    invalid.stream = 1;
    invalid.grid[0] = 0;
    expect(!fixture.executor.accept_kernel(1, invalid).accepted() && state.launch_calls == 0 &&
               fixture.executor.snapshot().outstanding == 0 && fixture.executor.shutdown(),
           "invalid launch metadata must not consume sequence, resource capacity or native work");
    {
        CudaExecutionDispatcher never_bound({3, 4}, {7, context(), fixture.resources}, dispatch());
        // Empty partial initialization must not intentionally leak the context lease.
    }
}

void ambiguous_destroy_is_not_retried(bool before_native) {
    const auto child = fork();
    expect(child >= 0, "create isolated intentional cleanup quarantine");
    if (child == 0) {
        state = {};
        auto fixture = std::make_unique<Fixture>();
        const auto accepted = fixture->accept(1, 8);
        state.destroy_status = CUDA_ERROR_INVALID_HANDLE;
        if (before_native) {
            state.create_status = CUDA_ERROR_OUT_OF_MEMORY;
            state.returns_error_handle = true;
        }
        bool valid = accepted.accepted() && fixture->executor.issue_next().state ==
                                                (before_native ? CudaExecutionIssueState::kRejected
                                                               : CudaExecutionIssueState::kIssued);
        state.query_status = CUDA_SUCCESS;
        valid =
            valid && fixture->executor.poll().retired == 0 && state.destroy_calls == 1 &&
            fixture->executor.snapshot().is_quarantined &&
            fixture->executor.snapshot().retained_parameter_bytes == (before_native ? 0U : 4U) &&
            fixture->executor.acknowledge_result(accepted.receipt) &&
            !fixture->executor.physical_readers_quiescent() &&
            (!before_native || fixture->executor.snapshot().outstanding == 0) &&
            fixture->executor.poll().retired == 0 &&
            fixture->executor.drain_issued().retired == 0 && state.destroy_calls == 1 &&
            !fixture->executor.shutdown();
        fixture.reset();
        _exit(valid ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "ambiguous event cleanup must retain bounded ownership without double destruction");
}

void quiescent_unbind_never_reuses_ids() {
    state = {};
    Fixture fixture;
    expect(fixture.executor.resources_quiescent() && fixture.executor.unbind_function(1) &&
               fixture.executor.unbind_stream(1) && !fixture.executor.unbind_stream(0) &&
               !fixture.executor.bind_function(1, function(), 4, fixture.resources) &&
               !fixture.executor.bind_stream(1, stream(), fixture.resources) &&
               fixture.executor.bind_function(2, function(), 4, fixture.resources) &&
               fixture.executor.bind_stream(2, stream(), fixture.resources) &&
               !fixture.accept(1, 7).accepted() && state.launch_calls == 0 &&
               fixture.executor.shutdown(),
           "quiescent removal cannot rebind a stale resource identity or native handle");
}
void typed_native_kernel_pair() {
    state = {};
    Fixture fixture(true, 4);
    const auto memory = fixture.executor.accept_memory(1, {.kind = CudaMemoryKind::kFill8,
                                                           .destination = 0x1000,
                                                           .count = 16,
                                                           .stream = 1,
                                                           .is_asynchronous = true});
    const auto first = fixture.accept(2, 11), second = fixture.accept(3, 22),
               third = fixture.accept(4, 33);
    expect(memory.accepted() && first.accepted() && second.accepted() && third.accepted() &&
               !fixture.executor.native_kernel_pair(),
           "queued commands are not pending SDK kernel events");
    expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               !fixture.executor.native_kernel_pair(),
           "a memory event plus one kernel is not a kernel pair");
    expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               fixture.executor.native_kernel_pair() ==
                   std::optional{std::array<std::uint64_t, 2>{2, 3}},
           "exact typed kernel identities exclude the independent memory event");
    expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued &&
               !fixture.executor.native_kernel_pair(),
           "three kernels cannot masquerade as an exact pair");
    state.events[1].is_ready = true;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.native_kernel_pair() ==
                   std::optional{std::array<std::uint64_t, 2>{3, 4}},
           "retired kernel cannot remain in the observation");
    auto wrong_owner =
        std::async(std::launch::async, [&] { return !fixture.executor.native_kernel_pair(); });
    expect(wrong_owner.get(), "foreign threads cannot inspect native owner proof metadata");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 3 && !fixture.executor.native_kernel_pair() &&
               fixture.executor.acknowledge_result(memory.receipt) &&
               fixture.executor.acknowledge_result(first.receipt) &&
               fixture.executor.acknowledge_result(second.receipt) &&
               fixture.executor.acknowledge_result(third.receipt) && fixture.executor.shutdown(),
           "all command and event resources retire once");
}
void immutable_native_frontiers() {
    state = {};
    Fixture fixture;
    expect(fixture.executor.query_frontier({}) == CUDA_ERROR_INVALID_HANDLE &&
               fixture.executor.capture_frontier(99).status == CUDA_ERROR_INVALID_HANDLE,
           "default/foreign stream cannot fabricate an empty completed frontier");
    auto empty = fixture.executor.capture_frontier(0);
    expect(empty.status == CUDA_SUCCESS &&
               fixture.executor.query_frontier(empty.frontier) == CUDA_SUCCESS,
           "an explicitly captured empty stream is complete");
    empty.frontier = {};
    const auto first = fixture.accept(1, 11);
    auto prefix = fixture.executor.capture_frontier();
    const auto native_observations = state.context_calls;
    state.is_wrong_context = true;
    expect(prefix.frontier.observe() == CUDA_ERROR_NOT_READY &&
               state.context_calls == native_observations,
           "owner-thread logical observation does not inspect the current native context");
    state.is_wrong_context = false;
    auto other_observer = std::async(std::launch::async, [&] { return prefix.frontier.observe(); });
    expect(other_observer.get() == CUDA_ERROR_INVALID_CONTEXT,
           "a foreign thread cannot read mutable owner observations");
    const auto observer_child = ::fork();
    expect(observer_child >= 0, "fork immutable frontier observation guard");
    if (!observer_child) _exit(prefix.frontier.observe() == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int observer_status{};
    expect(::waitpid(observer_child, &observer_status, 0) == observer_child &&
               WIFEXITED(observer_status) && !WEXITSTATUS(observer_status),
           "fork cannot promote an inherited frontier into completion");
    expect(first.accepted() && prefix.status == CUDA_SUCCESS &&
               fixture.executor.query_frontier(prefix.frontier) == CUDA_ERROR_NOT_READY &&
               state.record_calls == 0 && state.query_calls == 0 && state.drain_calls == 0,
           "capture includes accepted unissued work without adding SDK/event/sync calls");
    expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
           "issue first original after immutable capture");
    auto independent = fixture.executor.capture_frontier(0);
    expect(independent.status == CUDA_SUCCESS &&
               fixture.executor.query_frontier(independent.frontier) == CUDA_SUCCESS,
           "busy nondefault work cannot enter an empty legacy-stream frontier");
    independent.frontier = {};
    const auto second = fixture.accept(2, 22);
    expect(second.accepted() &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
           "later original remains independent of the old wait membership");
    auto all = fixture.executor.capture_frontier(1);
    expect(all.status == CUDA_SUCCESS &&
               fixture.executor.capture_frontier().status == CUDA_ERROR_OUT_OF_MEMORY,
           "retained frontier metadata is bounded");
    state.events[0].is_ready = true;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.query_frontier(prefix.frontier) == CUDA_SUCCESS &&
               fixture.executor.query_frontier(all.frontier) == CUDA_ERROR_NOT_READY &&
               state.record_calls == 2 && state.drain_calls == 0,
           "future pending work cannot extend an earlier frontier or add a native dependency");
    auto wrong_owner = std::async(std::launch::async, [&] {
        return fixture.executor.query_frontier(prefix.frontier) == CUDA_ERROR_INVALID_CONTEXT &&
               fixture.executor.capture_frontier().status == CUDA_ERROR_INVALID_CONTEXT;
    });
    expect(wrong_owner.get(), "frontier mutation/query remains on the exact nonrecycled owner");
    {
        Fixture foreign;
        expect(foreign.executor.query_frontier(prefix.frontier) == CUDA_ERROR_INVALID_HANDLE,
               "numeric context/binding identity cannot substitute for the private instance "
               "capability");
    }
    state.query_status = CUDA_ERROR_LAUNCH_FAILED;
    expect(fixture.executor.poll().status == CUDA_ERROR_LAUNCH_FAILED &&
               fixture.executor.query_frontier(prefix.frontier) == CUDA_SUCCESS &&
               fixture.executor.query_frontier(all.frontier) == CUDA_ERROR_LAUNCH_FAILED,
           "later async failure does not contaminate an already physically complete prefix");
    expect(fixture.executor.drain_issued().retired == 1 &&
               fixture.executor.acknowledge_result(first.receipt) &&
               fixture.executor.acknowledge_result(second.receipt) && fixture.executor.shutdown(),
           "only real native stop retires the failed later command");
}
void canceled_frontier_and_expired_instance() {
    state = {};
    CudaExecutionFrontier expired;
    {
        Fixture fixture;
        const auto pending = fixture.accept(1, 7);
        auto captured = fixture.executor.capture_frontier();
        expect(pending.accepted() && captured.status == CUDA_SUCCESS &&
                   fixture.executor.reject_queued(CUDA_ERROR_DEINITIALIZED) == 1,
               "accept then cancel an exact ordinary frontier before SDK issue");
        const auto reply = fixture.executor.host_result(pending.receipt);
        const auto retired = captured.frontier.retirement();
        expect(reply && reply->status == CUDA_ERROR_DEINITIALIZED &&
                   reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   fixture.executor.query_frontier(captured.frontier) == CUDA_ERROR_DEINITIALIZED &&
                   retired.status == CUDA_ERROR_DEINITIALIZED && retired.is_retired &&
                   state.launch_calls == 0 && !state.record_calls && !state.destroy_calls &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
                   fixture.executor.acknowledge_result(pending.receipt) &&
                   fixture.executor.shutdown(),
               "known pre-issue cancellation settles without pretending native execution");
        expired = captured.frontier;
    }
    Fixture next;
    expect(next.executor.query_frontier(expired) == CUDA_ERROR_INVALID_HANDLE &&
               expired.retirement().status == CUDA_ERROR_INVALID_HANDLE &&
               !expired.retirement().is_retired && next.executor.shutdown(),
           "an expired owner cannot be revived by a new executor instance");
}
void upcoming_legacy_staging_dependencies() {
    for (const bool is_nonblocking : {false, true}) {
        state = {};
        Fixture fixture(is_nonblocking, 4);
        const auto first = fixture.accept(1, 11);
        auto logical = fixture.executor.capture_frontier(0);
        auto staging = fixture.executor.capture_frontier(0, true);
        expect(first.accepted() && logical.status == CUDA_SUCCESS &&
                   staging.status == CUDA_SUCCESS &&
                   fixture.executor.query_frontier(logical.frontier) == CUDA_SUCCESS &&
                   fixture.executor.query_frontier(staging.frontier) ==
                       (is_nonblocking ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY),
               "upcoming legacy CPU staging includes blocking predecessors, never independent "
               "streams");
        expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
               "original predecessor issues once");
        const auto later = fixture.accept(2, 22);
        expect(later.accepted() &&
                   fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
               "later work does not extend the captured upcoming-command prefix");
        state.events[0].is_ready = true;
        expect(fixture.executor.poll().retired == 1 &&
                   fixture.executor.query_frontier(staging.frontier) == CUDA_SUCCESS &&
                   state.drain_calls == 0,
               "CPU staging proceeds at its captured dependency, not context-wide completion");
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().retired == 1 &&
                   fixture.executor.acknowledge_result(first.receipt) &&
                   fixture.executor.acknowledge_result(later.receipt) &&
                   fixture.executor.shutdown(),
               "all original and later resources retire once");
    }
}
}  // namespace

namespace {
struct EventCall final : CudaExecutionEventCall {
    unsigned int calls{};
    mutable unsigned int dependency_observations{};
    CUresult status{CUDA_SUCCESS};
    std::optional<CudaExecutionFrontier> dependency;
    CUresult dependency_status() const noexcept override {
        ++dependency_observations;
        return dependency ? dependency->observe() : CUDA_SUCCESS;
    }
    CUresult issue(CUstream) noexcept override {
        ++calls;
        return status;
    }
};
struct RetirementObserver final : CudaRetirementObserver {
    std::array<std::uint64_t, 4> sequences{};
    std::array<int, 4> statuses{};
    std::size_t count{};
    // The test implements the exact observer interface.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void retired(std::uint64_t sequence, int status) noexcept override {
        if (count < sequences.size()) {
            sequences[count] = sequence;
            statuses[count++] = status;
        }
    }
};
void retirement_observer_is_passive_and_owner_bound() {
    state = {};
    Fixture fixture;
    auto observer = std::make_shared<RetirementObserver>();
    expect(fixture.executor.set_retirement_observer(observer), "owner installs observer");
    auto wrong = std::async(std::launch::async,
                            [&] { return fixture.executor.set_retirement_observer({}); });
    expect(!wrong.get(), "foreign thread cannot modify the observer");
    const auto accepted = fixture.accept(1, 42);
    expect(accepted.accepted() && fixture.executor.issue_next().sequence == 1,
           "ordinary command is issued");
    expect(fixture.executor.poll().retired == 0 && observer->count == 0,
           "an incomplete physical tail does not notify the observer");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 && observer->count == 1 &&
               observer->sequences[0] == 1 && observer->statuses[0] == CUDA_SUCCESS,
           "normal physical retirement notifies the observer exactly once");
    expect(fixture.executor.poll().retired == 0 && observer->count == 1,
           "later polls cannot repeat the retirement notification");
    expect(fixture.executor.acknowledge_result(accepted.receipt) && fixture.executor.shutdown(),
           "observed commands preserve ordinary acknowledgment and cleanup");
    state = {};
    Fixture failed;
    auto failed_observer = std::make_shared<RetirementObserver>();
    auto progress = std::make_shared<Progress>();
    progress->stop_status = CUDA_ERROR_INVALID_VALUE;
    const auto failed_acceptance = accept_persistent(failed, progress);
    expect(failed.executor.set_retirement_observer(failed_observer) &&
               failed_acceptance.accepted() &&
               failed.executor.issue_next().state == CudaExecutionIssueState::kIssued,
           "failed-progress fixture issues an actual persistent command");
    state.query_status = CUDA_SUCCESS;
    const auto result = failed.executor.poll();
    expect(result.retired == 1 && result.status == CUDA_ERROR_INVALID_VALUE &&
               failed_observer->count == 1 &&
               failed_observer->statuses[0] == CUDA_ERROR_INVALID_VALUE,
           "physical cleanup must not turn a progress failure into success evidence");
    expect(
        failed.executor.acknowledge_result(failed_acceptance.receipt) && failed.executor.shutdown(),
        "the failed but physically retired fixture releases its owner normally");
}
void diagnostic_frontier_identities_are_cpu_only() {
    state = {};
    CudaExecutionFrontier expired;
    {
        Fixture fixture(true, 16);
        const auto calls = state.context_calls;
        const auto first = fixture.executor.capture_frontier();
        const auto second = fixture.executor.capture_frontier();
        std::vector<std::uint64_t> identities;
        const auto distinct = [&](const CudaExecutionFrontier& frontier) {
            const auto identity = frontier.diagnostic_id();
            expect(identity && std::ranges::find(identities, identity) == identities.end(),
                   "every published capture has a distinct owner-scoped diagnostic ID");
            identities.push_back(identity);
            return identity;
        };
        const auto original = distinct(first.frontier);
        distinct(second.frontier);
        const auto copied = first.frontier;
        expect(copied.diagnostic_id() == original && !CudaExecutionFrontier{}.diagnostic_id() &&
                   first.frontier.retirement().is_retired && state.context_calls == calls,
               "empty captures remain distinct; reading an ID neither calls SDK nor proves stop");
        const std::array<std::uint64_t, 2> streams{0, 1};
        const auto cohort = fixture.executor.accept_external_cohort(streams, fixture.resources);
        expect(cohort.status == CUDA_SUCCESS && cohort.members.size() == 2,
               "private cohort admission succeeds without ordinary commands");
        for (const auto& member : cohort.members) {
            distinct(member.frontier);
            expect(!member.frontier.retirement().is_retired &&
                       fixture.executor.cancel_external_barrier(member.barrier) == CUDA_SUCCESS,
                   "a positive private diagnostic ID cannot settle an unstarted boundary");
        }
        const auto turn = fixture.executor.accept_enqueue_turn(1, fixture.resources);
        expect(turn.status == CUDA_SUCCESS, "private host-turn capture succeeds");
        distinct(turn.frontier);
        expect(!turn.frontier.retirement().is_retired &&
                   fixture.executor.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS &&
                   !fixture.executor.snapshot().outstanding && state.context_calls == calls,
               "private IDs consume no ordinary sequence, SDK callback or physical proof");

        const auto reserved = fixture.executor.reserve_command_frontier(1);
        const auto pending = fixture.executor.reserved_command_frontier(reserved.reservation);
        expect(reserved.status == CUDA_SUCCESS && pending.status == CUDA_ERROR_NOT_READY &&
                   !pending.frontier.diagnostic_id(),
               "a pending reservation never exposes a valid empty capture");
        auto event = std::make_shared<EventCall>();
        const auto accepted = fixture.executor.accept_event_with_frontier(1, {1, event, {}});
        const auto bound = fixture.executor.reserved_command_frontier(reserved.reservation);
        expect(accepted.acceptance.accepted() && bound.status == CUDA_SUCCESS,
               "diagnostic capture does not alter the first ordinary acceptance sequence");
        const auto observations = event->dependency_observations;
        expect(!fixture.executor.is_quarantined() && !fixture.executor.is_quarantined() &&
                   event->dependency_observations == observations,
               "health observation never invokes an unrelated dependency callback");
        expect(fixture.executor.scheduling_snapshot().has_ready_work &&
                   event->dependency_observations > observations,
               "scheduling still performs the real dependency readiness check");
        distinct(bound.frontier);
        const auto explicit_id = distinct(accepted.frontier);
        expect(accepted.frontier_capture == explicit_id,
               "guarded event factory returns its actual accepted frontier identity");
        auto wrong = std::async(std::launch::async, [&] {
            return !copied.diagnostic_id() && !accepted.frontier.diagnostic_id();
        });
        expect(wrong.get(), "foreign thread cannot read owner diagnostic identities");
        const auto child = ::fork();
        expect(child >= 0, "fork inherited diagnostic identity control");
        if (!child) ::_exit(!copied.diagnostic_id() && !accepted.frontier.diagnostic_id() ? 0 : 1);
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork identity reads reject before inherited owner state");
        unsigned reentries{};
        state.on_context = [&] {
            ++reentries;
            expect(!copied.diagnostic_id() && !accepted.frontier.diagnostic_id(),
                   "native callback reentry cannot read diagnostic identities");
        };
        expect(fixture.executor.issue_next().sequence == 1 && reentries && event->calls == 1,
               "diagnostic exclusion leaves the one real ordering issue unchanged");
        state.on_context = {};
        expect(copied.diagnostic_id() == original && !accepted.frontier.retirement().is_retired,
               "ID availability returns without manufacturing physical completion");
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().retired == 1 &&
                   fixture.executor.acknowledge_result(accepted.acceptance.receipt) &&
                   fixture.executor.shutdown(),
               "all diagnostic fixtures retain normal exactly-once native cleanup");
        fixture.executor.close();
        expect(copied.diagnostic_id() == original,
               "healthy dispatcher closure preserves diagnostics while its owner remains alive");
        expired = first.frontier;
    }
    expect(!expired.diagnostic_id(), "expired owner never leaves a reusable diagnostic token");
    std::uint64_t next = UINT64_MAX;
    expect(detail::take_diagnostic_id(next) == UINT64_MAX && !next &&
               !detail::take_diagnostic_id(next) && !detail::take_diagnostic_id(next),
           "diagnostic exhaustion remains unavailable without wrap or execution failure");
}
void reserved_frontiers_are_transactional_and_bounded() {
    state = {};
    Fixture fixture(true);
    auto first = fixture.executor.reserve_command_frontier(1);
    auto second = fixture.executor.reserve_command_frontier(1);
    const auto copied = first.reservation;
    expect(
        first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS &&
            fixture.executor.reserve_command_frontier(2).status == CUDA_ERROR_OUT_OF_MEMORY &&
            fixture.executor.reserved_command_frontier(copied).status == CUDA_ERROR_NOT_READY &&
            !fixture.executor.reserved_command_frontier(copied).frontier.retirement().is_retired &&
            fixture.executor.snapshot().outstanding == 0,
        "two observer reservations charge metadata only and pending is not completion");
    expect(!fixture.accept(2, 17).accepted() &&
               fixture.executor.reserved_command_frontier(first.reservation).status ==
                   CUDA_ERROR_NOT_READY &&
               fixture.executor.reserved_command_frontier(second.reservation).status ==
                   CUDA_ERROR_NOT_READY,
           "failed sequence admission binds no observer and remains retryable");
    auto event = std::make_shared<EventCall>();
    expect(fixture.executor.accept_event_with_frontier(1, {1, event, {}}).acceptance.status ==
                   ExecutionAcceptStatus::kOutOfMemory &&
               fixture.executor.reserved_command_frontier(first.reservation).status ==
                   CUDA_ERROR_NOT_READY &&
               !event->calls,
           "extra explicit event observer quota fails before acceptance and binds nothing");
    const auto accepted = fixture.accept(1, 19);
    auto left = fixture.executor.reserved_command_frontier(copied);
    auto right = fixture.executor.reserved_command_frontier(second.reservation);
    expect(accepted.accepted() && left.status == CUDA_SUCCESS && right.status == CUDA_SUCCESS &&
               left.frontier.observe() == CUDA_ERROR_NOT_READY &&
               right.frontier.observe() == CUDA_ERROR_NOT_READY &&
               fixture.executor.cancel_frontier_reservation(copied) == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.reserve_command_frontier(1).status == CUDA_ERROR_INVALID_VALUE,
           "one acceptance binds all preallocated observers exactly once");
    expect(fixture.executor.reject_command(accepted.receipt, CUDA_ERROR_DEINITIALIZED) &&
               left.frontier.observe() == CUDA_ERROR_DEINITIALIZED &&
               left.frontier.retirement().is_retired && right.frontier.retirement().is_retired &&
               fixture.executor.host_result(accepted.receipt)->origin ==
                   ExecutionReplyOrigin::kRejectedBeforeIssue &&
               !state.launch_calls && fixture.executor.acknowledge_result(accepted.receipt),
           "exact rejection settles both observations without original SDK provenance");
    first = {};
    second = {};
    right = {};
    expect(fixture.executor.reserve_command_frontier(2).status == CUDA_SUCCESS,
           "dropping one distinct observer refunds only its metadata slot");
    left = {};
    expect(fixture.executor.reserved_command_frontier(copied).frontier.observe() ==
               CUDA_ERROR_DEINITIALIZED,
           "reservation copy keeps its bound failure capture alive");
    fixture.executor.close();
    expect(fixture.executor.reserve_command_frontier(2).status == CUDA_ERROR_DEINITIALIZED &&
               fixture.executor.reserved_command_frontier(copied).status == CUDA_SUCCESS &&
               fixture.executor.shutdown(),
           "close denies new reservations but preserves existing read-only evidence");
}

void reserved_ancestor_capacity_rejects_before_acceptance() {
    state = {};
    Fixture fixture(true, 8);
    const auto context_calls = state.context_calls;
    ExecutionReceipt previous;
    for (std::uint64_t sequence = 1; sequence != 6; ++sequence) {
        const auto reserved = fixture.executor.reserve_command_frontier(sequence);
        const auto accepted = fixture.accept(sequence, 61);
        expect(reserved.status == CUDA_SUCCESS && accepted.accepted(),
               "build bounded exact erased-ancestor chain");
        if (previous)
            expect(fixture.executor.reject_command(previous, CUDA_ERROR_INVALID_VALUE) &&
                       fixture.executor.acknowledge_result(previous),
                   "erase previous command without erasing its captured failure");
        previous = accepted.receipt;
    }
    const auto pending = fixture.executor.reserve_command_frontier(6);
    expect(fixture.accept(6, 67).status == ExecutionAcceptStatus::kOutOfMemory &&
               fixture.executor.snapshot().outstanding == 1 &&
               fixture.executor.reserved_command_frontier(pending.reservation).status ==
                   CUDA_ERROR_NOT_READY &&
               !state.launch_calls && state.context_calls == context_calls,
           "bounded transitive capture fails before command or sequence publication");
    expect(fixture.executor.reject_command(previous, CUDA_ERROR_INVALID_VALUE) &&
               fixture.executor.acknowledge_result(previous),
           "release actual old target");
    const auto retried = fixture.accept(6, 71);
    expect(retried.accepted() &&
               fixture.executor.reserved_command_frontier(pending.reservation).status ==
                   CUDA_SUCCESS &&
               fixture.executor.reject_command(retried.receipt, CUDA_ERROR_DEINITIALIZED) &&
               fixture.executor.acknowledge_result(retried.receipt) && fixture.executor.shutdown(),
           "same reservation and sequence can retry after exact capacity becomes available");
}

void every_admission_funnel_binds_exact_reserved_frontier() {
    for (unsigned kind = 0; kind != 6; ++kind) {
        state = {};
        auto resources = std::make_shared<int>(0);
        auto native = dispatch();
        native.upload_async = +[](CUdeviceptr, const void*, std::size_t, CUstream) {
            ++state.launch_calls;
            return CUDA_SUCCESS;
        };
        CudaExecutionDispatcher executor({1, 2}, {7, context(), resources}, native,
                                         {.max_commands = 4, .max_parameter_bytes = 16}, 8);
        expect(executor.bind_executor() == CUDA_SUCCESS &&
                   executor.bind_function(1, function(), 4, resources) &&
                   executor.bind_stream(1, stream(), resources, true),
               "bind independent admission-funnel fixture");
        const auto reserved = executor.reserve_command_frontier(1);
        std::uint32_t value{37};
        const CudaExecutionKernel kernel{.function = 1,
                                         .stream = 1,
                                         .grid = {80, 1, 1},
                                         .block = {32, 1, 1},
                                         .parameters = std::as_bytes(std::span{&value, 1})};
        auto progress = std::make_shared<Progress>();
        progress->is_complete = true;
        auto call = std::make_shared<LibraryCall>();
        call->result = 0;
        auto event = std::make_shared<EventCall>();
        auto host_value = std::make_shared<std::uint32_t>(37);
        ExecutionAcceptance accepted;
        switch (kind) {
            case 0:
                accepted = executor.accept_kernel(1, kernel);
                break;
            case 1:
                accepted = executor.accept_persistent_kernel(
                    1, kernel, {function(), 320, kernel.parameters, progress});
                break;
            case 2:
                accepted = executor.accept_memory(1, {.kind = CudaMemoryKind::kFill8,
                                                      .destination = 0x1000,
                                                      .count = 8,
                                                      .stream = 1,
                                                      .is_asynchronous = true});
                break;
            case 3:
                accepted = executor.accept_memory(1, {.kind = CudaMemoryKind::kUpload,
                                                      .destination = 0x1000,
                                                      .count = 4,
                                                      .stream = 1,
                                                      .is_asynchronous = true,
                                                      .host = host_value.get(),
                                                      .host_lifetime = host_value});
                break;
            case 4:
                accepted = executor.accept_library(1, {1, kernel.parameters, call});
                break;
            default:
                accepted = executor.accept_event(1, {1, event, {}});
                break;
        }
        auto exact = executor.reserved_command_frontier(reserved.reservation);
        const auto later = executor.accept_kernel(2, kernel);
        const auto context_calls = state.context_calls;
        const auto queued = executor.inspect_frontier_dependency(exact.frontier);
        const auto queued_retirement = executor.inspect_frontier_retirement(exact.frontier);
        expect(queued.status == CUDA_ERROR_NOT_READY && queued.has_progress &&
                   queued_retirement.status == CUDA_ERROR_NOT_READY &&
                   queued_retirement.has_progress && state.context_calls == context_calls,
               "each exact queued admission exposes CPU-only actionable progress");
        expect(reserved.status == CUDA_SUCCESS && accepted.accepted() && later.accepted() &&
                   exact.status == CUDA_SUCCESS &&
                   executor.progress_frontier_dependency(exact.frontier).sequence == 1 &&
                   !executor.reject_command(accepted.receipt, CUDA_ERROR_DEINITIALIZED),
               "every ordinary path atomically binds and progresses only its exact target");
        const auto physical = executor.inspect_frontier_retirement(exact.frontier);
        expect(physical.status == CUDA_ERROR_NOT_READY && !physical.has_progress &&
                   !executor.inspect_frontier_dependency(exact.frontier).has_progress &&
                   !exact.frontier.retirement().is_retired && executor.snapshot().queued == 1,
               "issued held tail is not actionable despite later queued ordinary work");
        state.query_status = CUDA_SUCCESS;
        expect(executor.poll().retired == 1 && exact.frontier.retirement().is_retired &&
                   exact.frontier.observe() == CUDA_SUCCESS && executor.snapshot().queued == 1 &&
                   !executor.inspect_frontier_dependency(exact.frontier).has_progress &&
                   !executor.inspect_frontier_retirement(exact.frontier).has_progress &&
                   executor.acknowledge_result(accepted.receipt) &&
                   executor.reject_command(later.receipt, CUDA_ERROR_DEINITIALIZED) &&
                   executor.acknowledge_result(later.receipt) && executor.shutdown(),
               "target retirement excludes a later accepted ordinary command");
    }
}

void reserved_ancestry_survives_erasure_without_widening_retirement() {
    for (const bool after_issue : {false, true}) {
        state = {};
        Fixture fixture(true, 8);
        auto native = std::make_shared<EventCall>();
        native->status = CUDA_ERROR_INVALID_VALUE;
        const auto ancestor = fixture.executor.accept_event(1, {1, native, {}});
        if (after_issue)
            expect(fixture.executor.issue_next().native_status == CUDA_ERROR_INVALID_VALUE,
                   "capture after original failure but before trailing stop");
        const auto reserved = fixture.executor.reserve_command_frontier(2);
        const auto target = fixture.accept(2, 41);
        const auto exact = fixture.executor.reserved_command_frontier(reserved.reservation);
        if (!after_issue)
            expect(fixture.executor.issue_next().native_status == CUDA_ERROR_INVALID_VALUE,
                   "original ancestor fails after target admission");
        auto observed = fixture.executor.inspect_frontier_dependency(exact.frontier);
        expect(ancestor.accepted() && target.accepted() && observed.has_failure_observation &&
                   !observed.has_retired_failure && observed.status == CUDA_ERROR_INVALID_VALUE &&
                   !exact.frontier.retirement().is_retired,
               "retained ancestor failure is distinct from target physical membership");
        expect(
            fixture.executor.reject_command(target.receipt, CUDA_ERROR_DEINITIALIZED) &&
                exact.frontier.retirement().is_retired &&
                exact.frontier.observe() == CUDA_ERROR_DEINITIALIZED &&
                fixture.executor.inspect_frontier_retirement(exact.frontier).status ==
                    CUDA_SUCCESS &&
                !fixture.executor.inspect_frontier_retirement(exact.frontier)
                     .has_failure_observation &&
                !fixture.executor.inspect_frontier_dependency(exact.frontier).has_retired_failure,
            "known target rejection cannot falsely retire its still-live ancestor");
        state.query_status = CUDA_SUCCESS;
        expect(fixture.executor.poll().retired == 1, "retire actual ancestor tail");
        observed = fixture.executor.inspect_frontier_dependency(exact.frontier);
        expect(observed.status == CUDA_ERROR_INVALID_VALUE && observed.has_retired_failure &&
                   fixture.executor.progress_frontier_dependency(exact.frontier).native_status ==
                       CUDA_ERROR_INVALID_VALUE &&
                   native->calls == 1 && !state.launch_calls &&
                   fixture.executor.acknowledge_result(ancestor.receipt) &&
                   fixture.executor.acknowledge_result(target.receipt) &&
                   fixture.executor.shutdown(),
               "erased ancestor stays exact and cannot be replayed through target progress");
    }
    state = {};
    Fixture fixture(true, 8);
    const auto predecessor = fixture.accept(1, 79);
    const auto reserved = fixture.executor.reserve_command_frontier(2);
    const auto target = fixture.accept(2, 83);
    const auto exact = fixture.executor.reserved_command_frontier(reserved.reservation);
    expect(predecessor.accepted() && target.accepted() &&
               fixture.executor.reject_command(predecessor.receipt, CUDA_ERROR_OUT_OF_MEMORY),
           "erase an exact ordinary ancestor before ordinary target scheduling");
    const auto inspected = fixture.executor.inspect_frontier_dependency(exact.frontier);
    expect(inspected.status == CUDA_ERROR_OUT_OF_MEMORY && inspected.has_failure_observation &&
               inspected.has_retired_failure &&
               fixture.executor.reject_command(target.receipt, inspected.status),
           "exact retained ancestor authorizes only explicit never-attempted target rejection");
    const auto reply = fixture.executor.host_result(target.receipt);
    expect(reply && reply->status == CUDA_ERROR_OUT_OF_MEMORY &&
               reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               exact.frontier.retirement().is_retired && !state.launch_calls &&
               fixture.executor.inspect_frontier_dependency(exact.frontier).has_retired_failure &&
               fixture.executor.acknowledge_result(predecessor.receipt) &&
               fixture.executor.acknowledge_result(target.receipt) && fixture.executor.shutdown(),
           "targeted rejection preserves original ancestor evidence without issuing native work");
}

void reservation_owner_cancel_and_attempt_guards() {
    state = {};
    Fixture fixture(true, 8), foreign(true, 8);
    const auto pending = fixture.executor.reserve_command_frontier(1);
    const auto other = foreign.accept(1, 91);
    expect(foreign.executor.reserved_command_frontier(pending.reservation).status ==
                   CUDA_ERROR_INVALID_HANDLE &&
               foreign.executor.cancel_frontier_reservation(pending.reservation) ==
                   CUDA_ERROR_INVALID_HANDLE,
           "reservation is authenticated by exact owner, not numeric sequence");
    const auto accepted = fixture.accept(1, 27);
    expect(!fixture.executor.reject_command(other.receipt, CUDA_ERROR_DEINITIALIZED),
           "same-sequence foreign receipt cannot reject a target");
    std::async(std::launch::async, [&] {
        expect(
            fixture.executor.reserved_command_frontier(pending.reservation).status ==
                    CUDA_ERROR_INVALID_CONTEXT &&
                fixture.executor.reserve_command_frontier(2).status == CUDA_ERROR_INVALID_CONTEXT &&
                !fixture.executor.reject_command(accepted.receipt, CUDA_ERROR_DEINITIALIZED),
            "foreign CPU thread gains no observation or rejection authority");
    }).get();
    const auto child = fork();
    expect(child >= 0, "fork reservation owner control");
    if (!child) {
        const bool denied =
            fixture.executor.reserved_command_frontier(pending.reservation).status ==
                CUDA_ERROR_INVALID_CONTEXT &&
            fixture.executor.cancel_frontier_reservation(pending.reservation) ==
                CUDA_ERROR_INVALID_CONTEXT;
        _exit(denied ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects before inherited ownership access");
    state.on_context = [&] {
        expect(fixture.executor.reserve_command_frontier(2).status == CUDA_ERROR_NOT_READY &&
                   fixture.executor.reserved_command_frontier(pending.reservation).status ==
                       CUDA_ERROR_NOT_READY &&
                   !fixture.executor.reject_command(accepted.receipt, CUDA_ERROR_DEINITIALIZED),
               "first callback cannot reenter reservation or targeted rejection");
    };
    expect(fixture.executor.issue_next().state == CudaExecutionIssueState::kIssued,
           "original callback remains once-only");
    state.on_context = {};
    expect(!fixture.executor.reject_command(accepted.receipt, CUDA_ERROR_DEINITIALIZED) &&
               fixture.executor.host_result(accepted.receipt)->origin ==
                   ExecutionReplyOrigin::kNativeSdk,
           "issued command cannot be rewritten as a CPU cancellation");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(accepted.receipt) &&
               foreign.executor.reject_command(other.receipt, CUDA_ERROR_DEINITIALIZED) &&
               foreign.executor.acknowledge_result(other.receipt),
           "retire both actual owners");
    const auto unused = fixture.executor.reserve_command_frontier(2);
    fixture.executor.close();
    expect(fixture.executor.cancel_frontier_reservation(unused.reservation) == CUDA_SUCCESS &&
               fixture.executor.reserved_command_frontier(unused.reservation).status ==
                   CUDA_ERROR_INVALID_HANDLE &&
               fixture.executor.shutdown() && foreign.executor.shutdown(),
           "close permits pending cancellation, which never becomes a completion proof");
    CudaExecutionFrontierReservation expired;
    CudaExecutionFrontier old_capture;
    std::weak_ptr<const void> old_resources;
    {
        auto old = std::make_unique<Fixture>();
        old_resources = old->resources;
        expired = old->executor.reserve_command_frontier(1).reservation;
        const auto command = old->accept(1, 93);
        old_capture = old->executor.reserved_command_frontier(expired).frontier;
        expect(command.accepted() &&
                   old->executor.reject_command(command.receipt, CUDA_ERROR_DEINITIALIZED) &&
                   old->executor.acknowledge_result(command.receipt) && old->executor.shutdown(),
               "finish old instance before expiring its nonrecycled identity");
    }
    expect(
        old_resources.expired() && !old_capture.retirement().is_retired &&
            old_capture.observe() == CUDA_ERROR_INVALID_HANDLE &&
            fixture.executor.inspect_frontier_retirement(old_capture).status ==
                CUDA_ERROR_INVALID_HANDLE &&
            fixture.executor.progress_frontier_retirement(old_capture).native_status ==
                CUDA_ERROR_INVALID_HANDLE &&
            fixture.executor.reserved_command_frontier(expired).status == CUDA_ERROR_INVALID_HANDLE,
        "retained CPU reservation cannot pin native resources or revive an expired owner");
}
void reserved_progress_preserves_pauses_and_foreign_generations() {
    state = {};
    Fixture source(true, 8), target(true, 8);
    auto progress = std::make_shared<Progress>();
    const auto producer = accept_persistent(source, progress);
    const auto reservation = source.executor.reserve_command_frontier(2);
    auto event = std::make_shared<EventCall>();
    const auto recorded = source.executor.accept_event(2, {1, event, {}});
    const auto exact = source.executor.reserved_command_frontier(reservation.reservation);
    const auto target_reservation = target.executor.reserve_command_frontier(1);
    auto wait = std::make_shared<EventCall>();
    const auto waited = target.executor.accept_event(1, {1, wait, {}, exact.frontier});
    const auto target_cut =
        target.executor.reserved_command_frontier(target_reservation.reservation);
    const auto blocker = target.executor.inspect_frontier_dependency(target_cut.frontier);
    expect(producer.accepted() && recorded.accepted() && waited.accepted() &&
               blocker.status == CUDA_ERROR_NOT_READY && blocker.foreign_blocker &&
               blocker.foreign_blocker->same_capture(exact.frontier) &&
               !blocker.has_failure_observation &&
               source.executor.progress_frontier_dependency(exact.frontier).sequence == 1,
           "reserved target identifies its exact foreign generation without recursive issue");
    state.query_status = CUDA_SUCCESS;
    expect(source.executor.poll().paused == 1 &&
               !source.executor.reject_command(producer.receipt, CUDA_ERROR_DEINITIALIZED) &&
               exact.frontier.observe() == CUDA_ERROR_NOT_READY &&
               target.executor.progress_frontier_dependency(target_cut.frontier).state ==
                   CudaExecutionIssueState::kIdle &&
               !wait->calls,
           "paused original is never cancellable or a completed foreign prefix");
    progress->is_complete = true;
    expect(source.executor.progress_frontier_dependency(exact.frontier).sequence == 1 &&
               source.executor.poll().retired == 1 &&
               source.executor.progress_frontier_dependency(exact.frontier).sequence == 2 &&
               source.executor.poll().retired == 1 &&
               target.executor.progress_frontier_dependency(target_cut.frontier).sequence == 1 &&
               target.executor.poll().retired == 1 && wait->calls == 1,
           "exact original continuation, record and captured foreign wait issue once");
    expect(source.executor.acknowledge_result(producer.receipt) &&
               source.executor.acknowledge_result(recorded.receipt) &&
               target.executor.acknowledge_result(waited.receipt) && source.executor.shutdown() &&
               target.executor.shutdown(),
           "reserved foreign owners retire independently");
}

void attempted_preparation_never_allows_targeted_rejection() {
    isolated_clean_preparation(
        [] {
            state = {};
            Fixture fixture;
            const auto reserved = fixture.executor.reserve_command_frontier(1);
            const auto accepted = fixture.accept(1, 53);
            state.create_status = CUDA_ERROR_OUT_OF_MEMORY;
            state.returns_error_handle = true;
            state.destroy_status = CUDA_ERROR_INVALID_HANDLE;
            const auto issued = fixture.executor.issue_next();
            const auto frontier = fixture.executor.reserved_command_frontier(reserved.reservation);
            const auto reply = fixture.executor.host_result(accepted.receipt);
            const auto calls = state.context_calls;
            const auto destroys = state.destroy_calls;
            const auto inspected = fixture.executor.inspect_frontier_retirement(frontier.frontier);
            expect(
                issued.state == CudaExecutionIssueState::kRejected && !state.launch_calls &&
                    reply && reply->status == CUDA_ERROR_OUT_OF_MEMORY &&
                    reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                    !fixture.executor.reject_command(accepted.receipt, CUDA_ERROR_DEINITIALIZED) &&
                    !frontier.frontier.retirement().is_retired &&
                    inspected.status == CUDA_ERROR_UNKNOWN && !inspected.has_failure_observation &&
                    !inspected.has_retired_failure &&
                    fixture.executor.progress_frontier_retirement(frontier.frontier)
                            .native_status == CUDA_ERROR_UNKNOWN &&
                    state.context_calls == calls && state.destroy_calls == destroys &&
                    !fixture.executor.shutdown(),
                "private event preparation with unknown ownership is never unattempted "
                "cancellation");
        },
        "targeted rejection retains attempted uncertain preparation");
}

void retirement_progress_skips_only_settled_members() {
    state = {};
    Fixture fixture(true, 8);
    const auto rejected = fixture.accept(1, 11);
    auto unrelated_call = std::make_shared<EventCall>();
    const auto unrelated = fixture.executor.accept_event(2, {0, unrelated_call, {}});
    const auto live = fixture.accept(3, 23);
    const auto cut = fixture.executor.capture_frontier(1);
    const auto later = fixture.accept(4, 37);
    expect(rejected.accepted() && unrelated.accepted() && live.accepted() && later.accepted() &&
               cut.status == CUDA_SUCCESS &&
               fixture.executor.reject_command(rejected.receipt, CUDA_ERROR_INVALID_VALUE),
           "capture one rejected and one live member before later ordinary work");
    const auto pending = fixture.executor.inspect_frontier_retirement(cut.frontier);
    expect(cut.frontier.observe() == CUDA_ERROR_INVALID_VALUE &&
               !cut.frontier.retirement().is_retired && pending.status == CUDA_ERROR_NOT_READY &&
               !pending.has_failure_observation && !pending.has_retired_failure &&
               pending.has_progress &&
               !fixture.executor.inspect_frontier_dependency(cut.frontier).has_progress &&
               fixture.executor.progress_frontier_dependency(cut.frontier).native_status ==
                   CUDA_ERROR_INVALID_VALUE,
           "retirement selection alone skips the settled historical error");
    const auto progressed = fixture.executor.progress_frontier_retirement(cut.frontier);
    expect(progressed.state == CudaExecutionIssueState::kIssued && progressed.sequence == 3 &&
               state.values == std::vector<std::uint32_t>{23} && !unrelated_call->calls &&
               fixture.executor.progress_frontier_retirement(cut.frontier).state ==
                   CudaExecutionIssueState::kIdle &&
               !fixture.executor.inspect_frontier_retirement(cut.frontier).has_progress &&
               !cut.frontier.retirement().is_retired,
           "one exact live member issues once while later work stays queued");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1, "actual captured tail retires");
    const auto stopped = fixture.executor.inspect_frontier_retirement(cut.frontier);
    expect(stopped.status == CUDA_SUCCESS && !stopped.has_failure_observation &&
               !stopped.has_retired_failure && !stopped.has_progress &&
               cut.frontier.retirement().is_retired &&
               cut.frontier.retirement().status == CUDA_ERROR_INVALID_VALUE &&
               cut.frontier.observe() == CUDA_ERROR_INVALID_VALUE &&
               fixture.executor.progress_frontier_retirement(cut.frontier).state ==
                   CudaExecutionIssueState::kIdle &&
               state.launch_calls == 1 && !unrelated_call->calls &&
               fixture.executor.issue_next().sequence == 2 &&
               fixture.executor.poll().retired == 1 && unrelated_call->calls == 1 &&
               fixture.executor.issue_next().sequence == 4 && fixture.executor.poll().retired == 1,
           "quiescence is not native success and never widens the immutable cut");
    for (const auto* receipt :
         {&rejected.receipt, &unrelated.receipt, &live.receipt, &later.receipt})
        expect(fixture.executor.acknowledge_result(*receipt),
               "retirement control receipts persist");
    expect(fixture.executor.shutdown(), "mixed retirement owners clean up");
}

void retirement_progress_resumes_only_captured_paused_work() {
    state = {};
    Fixture fixture(true, 8);
    auto progress = std::make_shared<Progress>();
    const auto producer = accept_persistent(fixture, progress);
    const auto rejected = fixture.accept(2, 19);
    const auto cut = fixture.executor.capture_frontier();
    const auto later = fixture.accept(3, 31);
    expect(producer.accepted() && rejected.accepted() && later.accepted() &&
               fixture.executor.reject_command(rejected.receipt, CUDA_ERROR_OUT_OF_MEMORY) &&
               fixture.executor.progress_frontier_retirement(cut.frontier).sequence == 1,
           "settled failure does not hide an earlier captured persistent member");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().paused == 1 && state.launch_calls == 1 &&
               !cut.frontier.retirement().is_retired &&
               fixture.executor.inspect_frontier_retirement(cut.frontier).has_progress &&
               !fixture.executor.inspect_frontier_dependency(cut.frontier).has_progress,
           "incomplete physical stop remains a live logical dependency");
    progress->is_complete = true;
    const auto resumed = fixture.executor.progress_frontier_retirement(cut.frontier);
    expect(resumed.state == CudaExecutionIssueState::kIssued && resumed.sequence == 1 &&
               state.values == std::vector<std::uint32_t>{73, 73} &&
               fixture.executor.poll().retired == 1 &&
               fixture.executor.inspect_frontier_retirement(cut.frontier).status == CUDA_SUCCESS &&
               cut.frontier.retirement().status == CUDA_ERROR_OUT_OF_MEMORY &&
               cut.frontier.retirement().is_retired &&
               fixture.executor.progress_frontier_dependency(cut.frontier).native_status ==
                   CUDA_ERROR_OUT_OF_MEMORY &&
               fixture.executor.issue_next().sequence == 3 && fixture.executor.poll().retired == 1,
           "exact paused continuation preserves old errors and excludes the later command");
    for (const auto* receipt : {&producer.receipt, &rejected.receipt, &later.receipt})
        expect(fixture.executor.acknowledge_result(*receipt),
               "paused retirement receipt remains exact");
    expect(fixture.executor.shutdown(), "paused retirement owns every physical tail");
}

void retirement_progress_waits_for_native_error_tails() {
    state = {};
    Fixture fixture(true, 8);
    auto failed = std::make_shared<EventCall>();
    failed->status = CUDA_ERROR_INVALID_VALUE;
    const auto first = fixture.executor.accept_event(1, {1, failed, {}});
    const auto live = fixture.accept(2, 41);
    const auto cut = fixture.executor.capture_frontier();
    expect(first.accepted() && live.accepted() &&
               fixture.executor.issue_next().native_status == CUDA_ERROR_INVALID_VALUE,
           "requested event error leaves a real healthy trailing event");
    const auto pending = fixture.executor.inspect_frontier_retirement(cut.frontier);
    expect(pending.status == CUDA_ERROR_INVALID_VALUE && pending.has_failure_observation &&
               !pending.has_retired_failure && !cut.frontier.retirement().is_retired &&
               fixture.executor.progress_frontier_retirement(cut.frontier).native_status ==
                   CUDA_ERROR_INVALID_VALUE &&
               !state.launch_calls && failed->calls == 1 && fixture.executor.poll().retired == 0,
           "unretired original error cannot authorize another issue or pretend to be a clean stop");
    state.query_status = CUDA_SUCCESS;
    expect(
        fixture.executor.poll().retired == 1 &&
            fixture.executor.inspect_frontier_retirement(cut.frontier).status ==
                CUDA_ERROR_NOT_READY &&
            !fixture.executor.inspect_frontier_retirement(cut.frontier).has_failure_observation &&
            fixture.executor.progress_frontier_retirement(cut.frontier).sequence == 2 &&
            fixture.executor.poll().retired == 1 && failed->calls == 1 && state.launch_calls == 1 &&
            cut.frontier.retirement().status == CUDA_ERROR_INVALID_VALUE &&
            cut.frontier.retirement().is_retired &&
            fixture.executor.acknowledge_result(first.receipt) &&
            fixture.executor.acknowledge_result(live.receipt) && fixture.executor.shutdown(),
        "actual error-tail retirement permits only the remaining captured operation");
}

void retirement_progress_preserves_old_foreign_error_generation() {
    state = {};
    Fixture source(true, 8), target(true, 8);
    auto failed = std::make_shared<EventCall>();
    failed->status = CUDA_ERROR_INVALID_VALUE;
    const auto first = source.executor.accept_event_with_frontier(1, {1, failed, {}});
    auto wait = std::make_shared<EventCall>();
    wait->dependency = first.frontier;
    const auto waited =
        target.executor.accept_event_with_frontier(1, {1, wait, {}, first.frontier});
    auto later = std::make_shared<EventCall>();
    const auto rerecorded = source.executor.accept_event_with_frontier(2, {1, later, {}});
    expect(first.acceptance.accepted() && waited.acceptance.accepted() &&
               rerecorded.acceptance.accepted() &&
               source.executor.issue_next().native_status == CUDA_ERROR_INVALID_VALUE,
           "foreign wait captures the old failing generation before a replacement exists");
    const auto pending = target.executor.inspect_frontier_retirement(waited.frontier);
    expect(pending.status == CUDA_ERROR_INVALID_VALUE && pending.foreign_blocker &&
               pending.foreign_blocker->same_capture(first.frontier) &&
               !pending.has_failure_observation && !pending.has_retired_failure &&
               target.executor.progress_frontier_retirement(waited.frontier).native_status ==
                   CUDA_ERROR_INVALID_VALUE &&
               !wait->calls && !later->calls,
           "foreign live error remains an exact blocker, never recursive source progress");
    state.query_status = CUDA_SUCCESS;
    expect(source.executor.poll().retired == 1 && first.frontier.retirement().is_retired,
           "original source independently proves its failed generation stopped");
    const auto ready = target.executor.inspect_frontier_retirement(waited.frontier);
    const auto rejected = target.executor.progress_frontier_retirement(waited.frontier);
    const auto reply = target.executor.host_result(waited.acceptance.receipt);
    expect(
        ready.status == CUDA_ERROR_NOT_READY && !ready.foreign_blocker &&
            !ready.has_failure_observation &&
            rejected.state == CudaExecutionIssueState::kRejected && rejected.sequence == 1 &&
            rejected.native_status == CUDA_ERROR_INVALID_VALUE && reply &&
            reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
            reply->status == CUDA_ERROR_INVALID_VALUE && !wait->calls && !later->calls &&
            waited.frontier.retirement().is_retired &&
            waited.frontier.retirement().status == CUDA_ERROR_INVALID_VALUE &&
            target.executor.inspect_frontier_retirement(waited.frontier).status == CUDA_SUCCESS &&
            source.executor.issue_next().sequence == 2 && source.executor.poll().retired == 1 &&
            source.executor.acknowledge_result(first.acceptance.receipt) &&
            source.executor.acknowledge_result(rerecorded.acceptance.receipt) &&
            target.executor.acknowledge_result(waited.acceptance.receipt) &&
            source.executor.shutdown() && target.executor.shutdown(),
        "one actual wait preparation rejects exactly; old failure never becomes a native wait");

    // A callback-only error has no authenticated captured frontier. Even a
    // numerically matching retired error elsewhere cannot authorize progress.
    auto failed_again = std::make_shared<EventCall>();
    failed_again->status = CUDA_ERROR_INVALID_VALUE;
    Fixture unbound_source(true, 8), unbound_target(true, 8);
    const auto origin =
        unbound_source.executor.accept_event_with_frontier(1, {1, failed_again, {}});
    auto unbound_wait = std::make_shared<EventCall>();
    unbound_wait->dependency = origin.frontier;
    const auto unbound =
        unbound_target.executor.accept_event_with_frontier(1, {1, unbound_wait, {}});
    expect(origin.acceptance.accepted() && unbound.acceptance.accepted() &&
               unbound_source.executor.issue_next().native_status == CUDA_ERROR_INVALID_VALUE &&
               unbound_source.executor.poll().retired == 1,
           "callback-only foreign failure has no trusted bridge metadata");
    const auto unauthenticated =
        unbound_target.executor.inspect_frontier_retirement(unbound.frontier);
    expect(
        unauthenticated.status == CUDA_ERROR_INVALID_VALUE &&
            !unauthenticated.has_failure_observation && !unauthenticated.has_retired_failure &&
            !unauthenticated.foreign_blocker &&
            unbound_target.executor.progress_frontier_retirement(unbound.frontier).native_status ==
                CUDA_ERROR_INVALID_VALUE &&
            !unbound_wait->calls &&
            unbound_target.executor.reject_command(unbound.acceptance.receipt,
                                                   CUDA_ERROR_DEINITIALIZED) &&
            unbound_source.executor.acknowledge_result(origin.acceptance.receipt) &&
            unbound_target.executor.acknowledge_result(unbound.acceptance.receipt) &&
            unbound_source.executor.shutdown() && unbound_target.executor.shutdown(),
        "raw callback failure cannot be silently skipped or supply stopped-owner proof");
}

void retirement_progress_preserves_preparation_rejection() {
    state = {};
    Fixture fixture(true, 8);
    auto progress = std::make_shared<Progress>();
    progress->prepare_status = CUDA_ERROR_OUT_OF_MEMORY;
    const auto first = accept_persistent(fixture, progress);
    const auto second = fixture.accept(2, 59);
    const auto cut = fixture.executor.capture_frontier();
    const auto later = fixture.accept(3, 61);
    const auto rejected = fixture.executor.progress_frontier_retirement(cut.frontier);
    const auto reply = fixture.executor.host_result(first.receipt);
    expect(
        first.accepted() && second.accepted() && later.accepted() &&
            rejected.state == CudaExecutionIssueState::kRejected && rejected.sequence == 1 &&
            rejected.native_status == CUDA_ERROR_OUT_OF_MEMORY && reply &&
            reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue && !state.launch_calls &&
            progress->prepare_calls == 1 && cut.frontier.observe() == CUDA_ERROR_OUT_OF_MEMORY &&
            !cut.frontier.retirement().is_retired &&
            fixture.executor.progress_frontier_retirement(cut.frontier).sequence == 2 &&
            state.values == std::vector<std::uint32_t>{59},
        "actual failed preparation stays rejected while the next grant selects only a live member");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.inspect_frontier_retirement(cut.frontier).status == CUDA_SUCCESS &&
               cut.frontier.retirement().status == CUDA_ERROR_OUT_OF_MEMORY &&
               cut.frontier.retirement().is_retired &&
               fixture.executor.issue_next().sequence == 3 &&
               fixture.executor.poll().retired == 1 && progress->prepare_calls == 1,
           "retirement never retries failed preparation or consumes later admission");
    for (const auto* receipt : {&first.receipt, &second.receipt, &later.receipt})
        expect(fixture.executor.acknowledge_result(*receipt),
               "preparation result remains independently owned");
    expect(fixture.executor.shutdown(), "preparation rejection has no speculative cleanup");
}

void retirement_progress_authenticates_before_quiescence() {
    state = {};
    Fixture fixture(true, 8), foreign(true, 8);
    const auto accepted = fixture.accept(1, 67);
    const auto cut = fixture.executor.capture_frontier();
    expect(accepted.accepted() &&
               fixture.executor.reject_command(accepted.receipt, CUDA_ERROR_UNKNOWN) &&
               fixture.executor.inspect_frontier_retirement(cut.frontier).status == CUDA_SUCCESS &&
               cut.frontier.observe() == CUDA_ERROR_UNKNOWN && cut.frontier.retirement().is_retired,
           "settled UNKNOWN is skipped by observation proof, not numeric status filtering");
    const auto denied = [&](const CudaExecutionDispatcher& executor, CUresult expected) {
        const auto observed = executor.inspect_frontier_retirement(cut.frontier);
        expect(observed.status == expected && !observed.has_failure_observation &&
                   !observed.has_retired_failure,
               "gate error is never a retained failure or retirement proof");
    };
    denied(foreign.executor, CUDA_ERROR_INVALID_HANDLE);
    std::async(std::launch::async, [&] {
        denied(fixture.executor, CUDA_ERROR_INVALID_CONTEXT);
        expect(fixture.executor.progress_frontier_retirement(cut.frontier).native_status ==
                   CUDA_ERROR_INVALID_CONTEXT,
               "foreign thread cannot progress even a settled cut");
    }).get();
    const auto child = fork();
    expect(child >= 0, "fork retirement-only authentication control");
    if (!child) {
        const auto observed = fixture.executor.inspect_frontier_retirement(cut.frontier);
        _exit(observed.status == CUDA_ERROR_INVALID_CONTEXT && !observed.has_failure_observation &&
                      fixture.executor.progress_frontier_retirement(cut.frontier).native_status ==
                          CUDA_ERROR_INVALID_CONTEXT
                  ? 0
                  : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork denies before touching inherited callback state");
    const auto later = fixture.accept(2, 71);
    unsigned reentries{};
    state.on_context = [&] {
        ++reentries;
        denied(fixture.executor, CUDA_ERROR_NOT_READY);
        expect(fixture.executor.progress_frontier_retirement(cut.frontier).native_status ==
                   CUDA_ERROR_NOT_READY,
               "settled cut does not bypass callback exclusion");
    };
    expect(later.accepted() && fixture.executor.issue_next().sequence == 2 && reentries != 0,
           "ordinary callbacks remain independent of old failed observation");
    state.on_context = {};
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(accepted.receipt) &&
               fixture.executor.acknowledge_result(later.receipt) && fixture.executor.shutdown() &&
               foreign.executor.shutdown(),
           "authentication controls retain exact resource cleanup");
}
void event_dependencies_keep_persistent_logical_completion() {
    state = {};
    Fixture fixture(true, 8);
    expect(fixture.executor.bind_stream(2, reinterpret_cast<CUstream>(&state.other_context),
                                        fixture.resources, true),
           "independent nonblocking event consumer stream");
    auto progress = std::make_shared<Progress>();
    const auto low = accept_persistent(fixture, progress);
    expect(low.accepted() && fixture.executor.issue_next().sequence == 1,
           "start persistent producer");
    auto record = std::make_shared<EventCall>();
    const auto recorded = fixture.executor.accept_event(2, {1, record, {}});
    const auto generation = fixture.executor.capture_command_frontier(2);
    auto wait = std::make_shared<EventCall>();
    const auto waited = fixture.executor.accept_event(3, {2, wait, generation.frontier});
    auto independent = std::make_shared<EventCall>();
    const auto other = fixture.executor.accept_event(4, {0, independent, {}});
    expect(recorded.accepted() && generation.status == CUDA_SUCCESS && waited.accepted() &&
               other.accepted() && fixture.executor.issue_next().sequence == 4 &&
               independent->calls == 1 && !record->calls && !wait->calls &&
               !fixture.executor.native_kernel_pair(),
           "event dependency cannot stall unrelated streams or masquerade as native kernel proof");
    state.query_status = CUDA_SUCCESS;
    const auto partial = fixture.executor.poll();
    expect(partial.paused == 1 && partial.retired == 1 &&
               fixture.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
               fixture.executor.query_frontier(generation.frontier) == CUDA_ERROR_NOT_READY,
           "physical incomplete retreat cannot record or release a logical event");
    progress->is_complete = true;
    expect(fixture.executor.resume(1).sequence == 1 && fixture.executor.poll().retired == 1 &&
               fixture.executor.issue_next().sequence == 2 && record->calls == 1 && !wait->calls &&
               fixture.executor.poll().retired == 1 &&
               fixture.executor.issue_next().sequence == 3 && wait->calls == 1 &&
               fixture.executor.poll().retired == 1 &&
               fixture.executor.query_frontier(generation.frontier) == CUDA_SUCCESS,
           "one logical completion orders one record then one captured-generation wait");
    for (const auto* receipt : {&low.receipt, &recorded.receipt, &waited.receipt, &other.receipt})
        expect(fixture.executor.acknowledge_result(*receipt), "event host result retires once");
    expect(fixture.executor.shutdown(), "event dependency resources retire normally");
}
void event_dependency_errors_never_issue_or_retry_waits() {
    state = {};
    Fixture fixture(true, 8);
    auto record = std::make_shared<EventCall>();
    record->status = CUDA_ERROR_INVALID_VALUE;
    const auto first = fixture.executor.accept_event(1, {1, record, {}});
    const auto generation = fixture.executor.capture_command_frontier(1);
    auto wait = std::make_shared<EventCall>();
    const auto second = fixture.executor.accept_event(2, {0, wait, generation.frontier});
    expect(
        first.accepted() && second.accepted() &&
            fixture.executor.issue_next().native_status == CUDA_ERROR_INVALID_VALUE &&
            fixture.executor.issue_next().state == CudaExecutionIssueState::kRejected &&
            record->calls == 1 && !wait->calls &&
            fixture.executor.host_result(second.receipt)->origin ==
                ExecutionReplyOrigin::kRejectedBeforeIssue &&
            !fixture.executor.native_kernel_pair(),
        "failed original event record rejects its dependent wait without issuing or replaying it");
    state.query_status = CUDA_SUCCESS;
    expect(fixture.executor.poll().retired == 1 &&
               fixture.executor.acknowledge_result(first.receipt) &&
               fixture.executor.acknowledge_result(second.receipt) && fixture.executor.shutdown(),
           "event error preserves independent physical cleanup");
}
void foreign_event_dependency_preserves_incomplete_persistent_work() {
    state = {};
    Fixture source(true, 8), target(true, 8);
    auto progress = std::make_shared<Progress>();
    const auto low = accept_persistent(source, progress);
    auto record = std::make_shared<EventCall>();
    const auto recorded = source.executor.accept_event(2, {1, record, {}});
    const auto generation = source.executor.capture_command_frontier(2);
    auto wait = std::make_shared<EventCall>();
    wait->dependency = generation.frontier;
    expect(!target.executor.accept_event(1, {1, wait, generation.frontier}).accepted(),
           "ordinary executor tokens cannot cross ownership without a trusted bridge");
    const auto waited = target.executor.accept_event(1, {1, wait, {}});
    expect(low.accepted() && recorded.accepted() && waited.accepted() &&
               source.executor.issue_next().sequence == 1 && !target.executor.has_ready_work() &&
               target.executor.issue_next().state == CudaExecutionIssueState::kIdle,
           "foreign logical dependency withholds runnable demand and native wait issue");
    state.query_status = CUDA_SUCCESS;
    expect(source.executor.poll().paused == 1 && !target.executor.has_ready_work() &&
               target.executor.issue_next().state == CudaExecutionIssueState::kIdle && !wait->calls,
           "foreign consumer cannot treat a persistent partial stop as source completion");
    progress->is_complete = true;
    expect(source.executor.resume(1).sequence == 1 && source.executor.poll().retired == 1 &&
               source.executor.issue_next().sequence == 2 && source.executor.poll().retired == 1 &&
               target.executor.has_ready_work() && target.executor.issue_next().sequence == 1 &&
               wait->calls == 1 && target.executor.poll().retired == 1 &&
               source.executor.acknowledge_result(low.receipt) &&
               source.executor.acknowledge_result(recorded.receipt) &&
               target.executor.acknowledge_result(waited.receipt) && source.executor.shutdown() &&
               target.executor.shutdown(),
           "one final producer continuation enables exactly one foreign wait and normal cleanup");
}
}  // namespace
int main() {
    try {
        native_dispatch_pipeline();
        retirement_observer_is_passive_and_owner_bound();
        diagnostic_frontier_identities_are_cpu_only();
        reserved_frontiers_are_transactional_and_bounded();
        reserved_ancestor_capacity_rejects_before_acceptance();
        every_admission_funnel_binds_exact_reserved_frontier();
        reserved_ancestry_survives_erasure_without_widening_retirement();
        reservation_owner_cancel_and_attempt_guards();
        reserved_progress_preserves_pauses_and_foreign_generations();
        attempted_preparation_never_allows_targeted_rejection();
        retirement_progress_skips_only_settled_members();
        retirement_progress_resumes_only_captured_paused_work();
        retirement_progress_waits_for_native_error_tails();
        retirement_progress_preserves_old_foreign_error_generation();
        retirement_progress_preserves_preparation_rejection();
        retirement_progress_authenticates_before_quiescence();
        event_dependencies_keep_persistent_logical_completion();
        event_dependency_errors_never_issue_or_retry_waits();
        foreign_event_dependency_preserves_incomplete_persistent_work();
        exclusive_library_gate_preserves_queued_work();
        exclusive_library_gate_does_not_serialize_persistent_kernels();
        exclusive_library_gate_failures_retain_owners();
        library_calls_keep_typed_results_and_physical_leases();
        library_preparation_and_record_failures();
        library_known_completion_error_still_retires();
        library_uncertain_completion_is_never_recovered();
        private_library_resources_retire_after_physical_stop();
        physical_quiescence_preserves_logical_quarantine();
        clean_preparation_keeps_physical_ownership();
        canceled_clean_preparation_never_enters_library();
        contradictory_clean_preparation_flags();
        clean_preparation_failure_paths();
        private_library_cleanup_failures_are_isolated();
        private_library_cleanup_reentry_keeps_records_stable();
        native_context_validation_excludes_reentry();
        library_waits_for_logical_persistent_predecessor();
        persistent_incomplete_stop_and_exact_resume();
        stopped_worker_retarget_preserves_the_exact_command();
        pending_commands_bind_exact_records_without_sdk_work();
        pending_continuation_invalidates_old_geometry_and_physical_generations();
        pending_original_selection_keeps_exact_abi_and_retained_budget();
        pending_worker_selection_preserves_initial_and_paused_ownership();
        exact_command_frontiers_exclude_unrelated_work();
        memory_commands_keep_sdk_results_and_physical_ownership();
        memory_waits_for_logical_persistent_predecessor();
        persistent_logical_stream_ordering();
        persistent_preparation_and_completion_errors();
        persistent_progress_errors_are_not_completion();
        private_prefix_failure_is_not_a_launch();
        private_prefix_failure_is_not_a_launch(true);
        continuation_patch_failure_preserves_original_result();
        profile_observes_retirement_and_preserves_failures();
        persistent_ambiguous_cleanup_retains_once();
        pre_native_rejection();
        failed_record_never_queries_unrecorded_event();
        failed_query_and_queued_rejection();
        owner_context_and_fork_guards();
        immutable_binding_and_input_validation();
        ambiguous_destroy_is_not_retried(false);
        ambiguous_destroy_is_not_retried(true);
        quiescent_unbind_never_reuses_ids();
        immutable_native_frontiers();
        typed_native_kernel_pair();
        upcoming_legacy_staging_dependencies();
        canceled_frontier_and_expired_instance();
        scheduling_snapshot_preserves_state_and_bounds_process_checks();
        std::cout << "CUDA execution dispatcher ownership checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
