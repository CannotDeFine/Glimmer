#include "internal/cuda_execution_dispatcher.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;

void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

enum class Callback { kContext, kCreate, kRecord, kQuery, kDestroy, kSynchronize, kLaunch };
struct Event {
    bool live{}, recorded{}, ready{};
    CUstream stream{};
};
struct Native {
    CUcontext current{};
    CUresult context_status{CUDA_SUCCESS}, create_status{CUDA_SUCCESS};
    CUresult record_status{CUDA_SUCCESS}, query_status{CUDA_SUCCESS};
    CUresult destroy_status{CUDA_SUCCESS}, synchronize_status{CUDA_SUCCESS};
    bool null_create{}, error_handle{};
    std::array<Event, 64> events{};
    std::size_t next_event{};
    unsigned contexts{}, creates{}, records{}, queries{}, destroys{}, synchronizes{}, launches{};
    unsigned unrecorded_queries{};
    std::vector<std::uint32_t> arguments;
    std::vector<CUstream> launched_streams;
    std::function<void(Callback)> callback;

    [[nodiscard]] unsigned operations() const {
        return contexts + creates + records + queries + destroys + synchronizes + launches;
    }
    void visit(Callback site) {
        if (callback) callback(site);
    }
    void ready_all() {
        for (auto& event : events) event.ready = true;
    }
};
Native native;

CUresult CUDAAPI current_context(CUcontext* output) {
    ++native.contexts;
    native.visit(Callback::kContext);
    *output = native.current;
    return native.context_status;
}
CUresult CUDAAPI event_create(CUevent* output, unsigned flags) {
    ++native.creates;
    expect(flags == CU_EVENT_DISABLE_TIMING, "barrier events never collect timing");
    if (!native.null_create && (native.create_status == CUDA_SUCCESS || native.error_handle)) {
        expect(native.next_event < native.events.size(), "bounded independent event storage");
        auto& event = native.events[native.next_event++];
        event = {.live = true};
        *output = reinterpret_cast<CUevent>(&event);
    }
    native.visit(Callback::kCreate);
    return native.create_status;
}
CUresult CUDAAPI event_record(CUevent event, CUstream stream) {
    ++native.records;
    auto& value = *reinterpret_cast<Event*>(event);
    expect(value.live && !value.recorded, "each private event is recorded at most once");
    if (native.record_status == CUDA_SUCCESS) {
        value.recorded = true;
        value.stream = stream;
    }
    native.visit(Callback::kRecord);
    return native.record_status;
}
CUresult CUDAAPI event_query(CUevent event) {
    ++native.queries;
    const auto& value = *reinterpret_cast<Event*>(event);
    expect(value.live, "query requires an independently live native event");
    native.unrecorded_queries += !value.recorded;
    native.visit(Callback::kQuery);
    // Deliberately permissive: an unrecorded native event can report success.
    // Safety must come from the dispatcher's enqueue state, not this fake SDK.
    if (!value.recorded) return CUDA_SUCCESS;
    if (native.query_status != CUDA_SUCCESS) return native.query_status;
    return value.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent event) {
    ++native.destroys;
    auto& value = *reinterpret_cast<Event*>(event);
    expect(value.live, "destroy never retries an already destroyed event");
    if (native.destroy_status == CUDA_SUCCESS) value.live = false;
    native.visit(Callback::kDestroy);
    return native.destroy_status;
}
CUresult CUDAAPI synchronize(CUstream) {
    ++native.synchronizes;
    native.visit(Callback::kSynchronize);
    return native.synchronize_status;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA launch ABI.
CUresult CUDAAPI launch(CUfunction function, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                        unsigned by, unsigned bz, unsigned, CUstream stream, void** arguments,
                        void** extra) {
    ++native.launches;
    expect(function && gx && gy == 1 && gz == 1 && bx == 32 && by == 1 && bz == 1 && !arguments &&
               extra && extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER &&
               extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE && extra[4] == CU_LAUNCH_PARAM_END &&
               *static_cast<const std::size_t*>(extra[3]) == sizeof(std::uint32_t),
           "ordinary/private launch keeps independently checked packed ABI");
    std::uint32_t value{};
    std::memcpy(&value, extra[1], sizeof(value));
    native.arguments.push_back(value);
    native.launched_streams.push_back(stream);
    native.visit(Callback::kLaunch);
    return CUDA_SUCCESS;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

CudaExecutionDispatch dispatch() {
    return {current_context, launch,        event_create, event_record,
            event_query,     event_destroy, synchronize};
}

struct Configuration {
    std::size_t commands{16}, bindings{32};
    bool isolated{};
};
struct Owner {
    char context_storage{}, function_storage{};
    std::array<char, 3> streams{};
    std::shared_ptr<const void> binding = std::make_shared<int>(19);
    CudaExecutionDispatcher executor;

    explicit Owner(Configuration configuration = {})
        : executor({71, 93}, {1, context(), binding}, dispatch(),
                   {.max_commands = configuration.commands, .max_parameter_bytes = 256},
                   configuration.bindings, configuration.isolated) {
        activate();
        expect(executor.bind_executor() == CUDA_SUCCESS &&
                   executor.bind_function(1, function(), sizeof(std::uint32_t), binding) &&
                   executor.bind_stream(1, stream(1), binding) &&
                   executor.bind_stream(2, stream(2), binding, true) &&
                   executor.bind_stream(3, stream(3), binding),
               "independent owner binds function, blocking and nonblocking streams");
    }
    [[nodiscard]] CUcontext context() {
        return reinterpret_cast<CUcontext>(&context_storage);
    }
    [[nodiscard]] CUfunction function() {
        return reinterpret_cast<CUfunction>(&function_storage);
    }
    [[nodiscard]] CUstream stream(std::size_t id) {
        return id == 0 ? nullptr : reinterpret_cast<CUstream>(&streams.at(id - 1));
    }
    void activate() {
        native.current = context();
    }
    ExecutionAcceptance kernel(std::uint64_t sequence, std::uint64_t stream_id = 1) {
        std::uint32_t value = static_cast<std::uint32_t>(sequence * 17);
        auto result =
            executor.accept_kernel(sequence, {.function = 1,
                                              .stream = stream_id,
                                              .grid = {80, 1, 1},
                                              .block = {32, 1, 1},
                                              .parameters = std::as_bytes(std::span{&value, 1})});
        value = 999;
        return result;
    }
    CudaExecutionExternalAcceptance barrier(std::uint64_t stream_id = 1) {
        return executor.accept_external_barrier(stream_id, std::make_shared<int>(31));
    }
    void finish() {
        native.ready_all();
        expect(executor.poll().status == CUDA_SUCCESS && executor.external_barrier_count() == 0 &&
                   executor.physical_readers_quiescent() && executor.shutdown(),
               "known completion releases all records before owner shutdown");
    }
};

struct Progress final : CudaExecutionPersistentProgress {
    unsigned prepares{}, retreats{}, readbacks{}, stops{};
    bool complete{};
    CUresult prepare_launch() noexcept override {
        ++prepares;
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        ++retreats;
        return CUDA_SUCCESS;
    }
    CUresult record_readback(CUstream) noexcept override {
        ++readbacks;
        return CUDA_SUCCESS;
    }
    CudaExecutionPersistentStop observe_physical_stop() noexcept override {
        ++stops;
        return {CUDA_SUCCESS, complete};
    }
};

void cancel_cohort(Owner& owner, const CudaExecutionExternalCohortAcceptance& cohort) {
    expect(cohort.status == CUDA_SUCCESS && !cohort.members.empty(),
           "cancellation requires an actually accepted nonempty cohort");
    for (const auto& member : cohort.members)
        expect(owner.executor.cancel_external_barrier(member.barrier) == CUDA_SUCCESS &&
                   member.frontier.observe() == CUDA_SUCCESS,
               "each unstarted member settles exactly its own no-work frontier");
}

void cohort_atomic_admission() {
    native = {};
    {
        Owner owner;
        const auto calls = native.operations();
        const std::array<std::uint64_t, 3> invalid{1, 99, 2};
        auto lifetime = std::make_shared<int>(52);
        const std::weak_ptr<const void> weak = lifetime;
        const auto rejected = owner.executor.accept_external_cohort(invalid, lifetime);
        lifetime.reset();
        const std::array<std::uint64_t, 1> valid{1};
        const auto empty = owner.executor.accept_external_cohort({}, owner.binding);
        const auto absent = owner.executor.accept_external_cohort(valid, {});
        expect(rejected.status == CUDA_ERROR_INVALID_HANDLE && rejected.members.empty() &&
                   empty.status == CUDA_ERROR_INVALID_VALUE && empty.members.empty() &&
                   absent.status == CUDA_ERROR_INVALID_VALUE && absent.members.empty() &&
                   weak.expired() && owner.executor.external_barrier_count() == 0 &&
                   native.operations() == calls,
               "invalid later stream, empty input and missing union owner publish no partial pins");
        const std::array<std::uint64_t, 5> streams{3, 0, 3, 2, 0};
        const auto cohort = owner.executor.accept_external_cohort(streams, owner.binding);
        expect(cohort.status == CUDA_SUCCESS && cohort.members.size() == 3 &&
                   cohort.members[0].stream == 3 && cohort.members[1].stream == 0 &&
                   cohort.members[2].stream == 2 && owner.executor.external_barrier_count() == 3 &&
                   owner.executor.snapshot().outstanding == 0 &&
                   owner.executor.snapshot().unacknowledged_replies == 0 &&
                   native.operations() == calls,
               "deduplication preserves first appearance without native calls or application "
               "receipts");
        cancel_cohort(owner, cohort);
        const auto command = owner.kernel(1);
        expect(command.accepted() && owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt),
               "failed and successful cohort admissions consume no application sequence");
        owner.finish();
    }
    native = {};
    {
        Owner owner({.commands = 2});
        const std::array<std::uint64_t, 3> oversized{1, 1, 1};
        const auto rejected = owner.executor.accept_external_cohort(oversized, owner.binding);
        expect(rejected.status == CUDA_ERROR_INVALID_VALUE && rejected.members.empty() &&
                   owner.executor.external_barrier_count() == 0,
               "raw input is bounded before duplicate compression");
        const std::array<std::uint64_t, 2> duplicates{1, 1};
        const auto first = owner.executor.accept_external_cohort(duplicates, owner.binding);
        const auto second = owner.barrier(2);
        expect(first.status == CUDA_SUCCESS && first.members.size() == 1 &&
                   second.status == CUDA_SUCCESS && owner.executor.external_barrier_count() == 2,
               "only unique streams consume command slots after bounded input validation");
        cancel_cohort(owner, first);
        expect(owner.executor.cancel_external_barrier(second.barrier) == CUDA_SUCCESS,
               "single admission remains compatible with cohort quota");
        owner.finish();
    }
    native = {};
    {
        Owner owner({.commands = 3});
        const auto command = owner.kernel(1);
        const std::array<std::uint64_t, 3> too_many{0, 1, 2};
        const auto rejected = owner.executor.accept_external_cohort(too_many, owner.binding);
        expect(command.accepted() && rejected.status == CUDA_ERROR_OUT_OF_MEMORY &&
                   rejected.members.empty() && owner.executor.external_barrier_count() == 0 &&
                   owner.executor.snapshot().queued == 1,
               "insufficient aggregate command capacity cannot publish a first cohort member");
        const std::array<std::uint64_t, 2> fits{0, 2};
        const auto cohort = owner.executor.accept_external_cohort(fits, owner.binding);
        expect(cohort.status == CUDA_SUCCESS && cohort.members.size() == 2,
               "failed aggregate admission leaves the complete remaining quota reusable");
        cancel_cohort(owner, cohort);
        expect(owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt),
               "ordinary predecessor survives failed aggregate admission unchanged");
        owner.finish();
    }
    native = {};
    {
        Owner owner({.bindings = 4});
        const std::array<std::uint64_t, 3> first_streams{0, 1, 2};
        auto first = owner.executor.accept_external_cohort(first_streams, owner.binding);
        const std::array<std::uint64_t, 2> next_streams{2, 3};
        const auto rejected = owner.executor.accept_external_cohort(next_streams, owner.binding);
        expect(first.status == CUDA_SUCCESS && rejected.status == CUDA_ERROR_OUT_OF_MEMORY &&
                   rejected.members.empty() && owner.executor.external_barrier_count() == 3,
               "insufficient frontier capacity rejects the entire second cohort");
        cancel_cohort(owner, first);
        const auto retained = owner.executor.accept_external_cohort(next_streams, owner.binding);
        expect(retained.status == CUDA_ERROR_OUT_OF_MEMORY && retained.members.empty() &&
                   owner.executor.external_barrier_count() == 0,
               "settled but live exact-tail handles still consume their frontier registrations");
        first.members.clear();
        const std::array<std::uint64_t, 4> all_streams{0, 1, 2, 3};
        const auto reused = owner.executor.accept_external_cohort(all_streams, owner.binding);
        expect(reused.status == CUDA_SUCCESS && reused.members.size() == 4,
               "expired observations refund all bounded slots without a leaked partial admission");
        cancel_cohort(owner, reused);
        owner.finish();
    }
}

void cohort_union_readiness_matrix() {
    struct Row {
        std::uint64_t predecessor;
        std::array<std::uint64_t, 2> streams;
        bool dependent;
    };
    // These expected dependencies are literal CUDA same-stream/legacy rules,
    // not values computed by the dispatcher's readiness implementation.
    constexpr std::array rows{Row{1, {3, 2}, false}, Row{3, {1, 2}, false}, Row{1, {2, 1}, true},
                              Row{0, {2, 3}, true},  Row{2, {0, 1}, false}, Row{3, {0, 2}, true},
                              Row{0, {2, 2}, false}, Row{2, {3, 2}, true},  Row{1, {0, 1}, true},
                              Row{3, {1, 0}, true}};
    for (const bool external_predecessor : {false, true}) {
        for (const auto& row : rows) {
            native = {};
            Owner owner;
            const auto ordinary =
                external_predecessor ? ExecutionAcceptance{} : owner.kernel(1, row.predecessor);
            const auto previous = external_predecessor ? owner.barrier(row.predecessor)
                                                       : CudaExecutionExternalAcceptance{};
            expect(external_predecessor ? previous.status == CUDA_SUCCESS : ordinary.accepted(),
                   "union readiness has an actual accepted predecessor");
            const auto cohort = owner.executor.accept_external_cohort(row.streams, owner.binding);
            expect(cohort.status == CUDA_SUCCESS && !cohort.members.empty(),
                   "bounded stream row accepts one coherent cohort");
            const auto calls = native.operations();
            for (const auto& member : cohort.members) {
                expect(owner.executor.external_barrier_ready(member.barrier) ==
                           (row.dependent ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS),
                       "every member waits on the union, even if its own stream is independent");
                if (row.dependent)
                    expect(owner.executor.begin_external_barrier(member.barrier) ==
                               CUDA_ERROR_NOT_READY,
                           "begin cannot bypass another member's unissued prefix");
            }
            expect(native.operations() == calls,
                   "logical readiness and blocked begin remain CPU-only");
            if (external_predecessor) {
                expect(owner.executor.cancel_external_barrier(previous.barrier) == CUDA_SUCCESS,
                       "known unentered predecessor cancellation releases its logical dependency");
            } else {
                expect(owner.executor.issue_next().sequence == 1 &&
                           owner.executor.acknowledge_result(ordinary.receipt),
                       "ordinary native enqueue preserves the existing FIFO readiness exemption");
            }
            for (const auto& member : cohort.members)
                expect(owner.executor.external_barrier_ready(member.barrier) == CUDA_SUCCESS &&
                           owner.executor.begin_external_barrier(member.barrier) == CUDA_SUCCESS,
                       "all sibling prefixes can begin before any sibling has enqueued");
            for (const auto& member : cohort.members)
                expect(owner.executor.record_external_enqueue(member.barrier) == CUDA_SUCCESS,
                       "each member separately proves its exact-stream enqueue");
            owner.finish();
            expect(native.unrecorded_queries == 0,
                   "coherent logical readiness never authorizes querying an unrecorded tail");
        }
    }
}

void cohort_intervening_producer_and_successors() {
    native = {};
    Owner owner;
    // An inner grouped call may retain an existing frontier, but must not put
    // a barrier into the source queue before the actual outer-End boundary.
    const auto inner_frontier = owner.executor.capture_frontier(1);
    expect(
        inner_frontier.status == CUDA_SUCCESS && inner_frontier.frontier.observe() == CUDA_SUCCESS,
        "inner-time observation contains no invented future producer");
    const auto producer = owner.kernel(1, 1);
    expect(producer.accepted() && owner.executor.has_ready_work(),
           "ordinary producer remains admissible after inner capture and before cohort End");
    const std::array<std::uint64_t, 2> streams{2, 1};
    const auto cohort = owner.executor.accept_external_cohort(streams, owner.binding);
    expect(cohort.status == CUDA_SUCCESS && cohort.members.size() == 2 &&
               owner.executor.external_barrier_ready(cohort.members[0].barrier) ==
                   CUDA_ERROR_NOT_READY &&
               owner.executor.external_barrier_ready(cohort.members[1].barrier) ==
                   CUDA_ERROR_NOT_READY &&
               inner_frontier.frontier.observe() == CUDA_SUCCESS,
           "outer-End union observes the late producer, not the stale inner-time frontier");
    expect(owner.executor.issue_next().sequence == 1 &&
               owner.executor.acknowledge_result(producer.receipt),
           "intervening producer actually issues before either cohort prefix");
    for (const auto& member : cohort.members)
        expect(owner.executor.begin_external_barrier(member.barrier) == CUDA_SUCCESS,
               "all incoming prefixes begin before outgoing joins");
    const auto same = owner.kernel(2, 1);
    const auto other = owner.kernel(3, 2);
    expect(same.accepted() && other.accepted() && !owner.executor.has_ready_work() &&
               owner.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
               owner.executor.poll().retired == 0 && native.unrecorded_queries == 0,
           "ordinary successors cannot treat begun-but-unrecorded members as enqueue proof");
    expect(owner.executor.record_external_enqueue(cohort.members[1].barrier) == CUDA_SUCCESS &&
               owner.executor.issue_next().sequence == 2 &&
               owner.executor.acknowledge_result(same.receipt) && !owner.executor.has_ready_work(),
           "one recorded tail releases only the ordinary successor with matching stream authority");
    expect(owner.executor.record_external_enqueue(cohort.members[0].barrier) == CUDA_SUCCESS &&
               owner.executor.issue_next().sequence == 3 &&
               owner.executor.acknowledge_result(other.receipt) &&
               native.arguments == std::vector<std::uint32_t>{17, 34, 51} &&
               native.launched_streams ==
                   std::vector<CUstream>{owner.stream(1), owner.stream(1), owner.stream(2)},
           "cohort members preserve ordinary application sequence, copied arguments and exact "
           "streams");
    owner.finish();
    expect(native.records == 5 && native.destroys == 5,
           "three original commands and two external tails retain five distinct physical events");
}

void cohort_distinct_tail_lifetimes() {
    native = {};
    Owner owner;
    auto lifetime = std::make_shared<int>(61);
    const std::weak_ptr<const void> weak = lifetime;
    const std::array<std::uint64_t, 4> streams{0, 1, 0, 2};
    const auto cohort = owner.executor.accept_external_cohort(streams, lifetime);
    lifetime.reset();
    expect(cohort.status == CUDA_SUCCESS && cohort.members.size() == 3 && !weak.expired(),
           "one immutable union owner covers all deduplicated stream members");
    for (const auto& member : cohort.members)
        expect(owner.executor.begin_external_barrier(member.barrier) == CUDA_SUCCESS,
               "legacy and blocking siblings do not deadlock their own coherent prefixes");
    expect(native.creates == 3 && native.records == 0 && owner.executor.poll().retired == 0 &&
               native.queries == 0 && native.synchronizes == 0 && !weak.expired(),
           "three begun events are not three physical completion observations");
    expect(owner.executor.record_external_enqueue(cohort.members[0].barrier) == CUDA_SUCCESS,
           "first member has an actual recorded tail");
    native.events[0].ready = true;
    expect(owner.executor.poll().retired == 1 && native.destroys == 1 && !weak.expired() &&
               cohort.members[0].frontier.observe() == CUDA_SUCCESS &&
               cohort.members[1].frontier.observe() == CUDA_ERROR_NOT_READY &&
               cohort.members[2].frontier.observe() == CUDA_ERROR_NOT_READY,
           "retiring the first exact tail cannot settle siblings or release the union owner");
    expect(owner.executor.record_external_enqueue(cohort.members[1].barrier) == CUDA_SUCCESS &&
               owner.executor.record_external_enqueue(cohort.members[2].barrier) == CUDA_SUCCESS,
           "remaining members record distinct exact-stream tails");
    native.events[2].ready = true;
    expect(owner.executor.poll().retired == 1 && native.destroys == 2 && !weak.expired() &&
               cohort.members[2].frontier.observe() == CUDA_SUCCESS &&
               cohort.members[1].frontier.observe() == CUDA_ERROR_NOT_READY &&
               !owner.executor.physical_readers_quiescent(),
           "out-of-order stop of the last stream cannot release a still-pending middle stream");
    native.events[1].ready = true;
    expect(
        owner.executor.poll().retired == 1 && weak.expired() && native.destroys == 3 &&
            native.unrecorded_queries == 0 && native.events[0].stream == nullptr &&
            native.events[1].stream == owner.stream(1) &&
            native.events[2].stream == owner.stream(2) && owner.executor.shutdown(),
        "only every real exact tail retires the common lease, with no unrecorded query or replay");
}

void cohort_persistent_and_profile_isolation() {
    native = {};
    {
        Owner owner;
        const auto progress = std::make_shared<Progress>();
        const std::uint32_t original = 41, prepared = 73;
        const auto command = owner.executor.accept_persistent_kernel(
            1,
            {.function = 1,
             .stream = 1,
             .grid = {6400, 1, 1},
             .block = {32, 1, 1},
             .parameters = std::as_bytes(std::span{&original, 1})},
            {owner.function(), 320, std::as_bytes(std::span{&prepared, 1}), progress});
        const std::array<std::uint64_t, 2> streams{2, 1};
        const auto cohort = owner.executor.accept_external_cohort(streams, owner.binding);
        expect(command.accepted() && cohort.status == CUDA_SUCCESS &&
                   owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt) &&
                   owner.executor.request_retreat(1) == CUDA_SUCCESS,
               "cohort union includes an actually issued persistent predecessor");
        native.ready_all();
        expect(owner.executor.poll().paused == 1 && progress->stops == 1,
               "persistent predecessor has a real but incomplete physical stop");
        for (const auto& member : cohort.members)
            expect(
                owner.executor.external_barrier_ready(member.barrier) == CUDA_ERROR_NOT_READY &&
                    owner.executor.begin_external_barrier(member.barrier) == CUDA_ERROR_NOT_READY,
                "even the independent member cannot bypass the union's paused logical work");
        progress->complete = true;
        expect(owner.executor.resume(1).state == CudaExecutionIssueState::kIssued,
               "same persistent command resumes without a cohort-induced fallback");
        native.ready_all();
        expect(owner.executor.poll().retired == 1 && progress->stops == 2 &&
                   native.arguments == std::vector<std::uint32_t>{73, 73},
               "only complete continuation settles the union predecessor");
        for (const auto& member : cohort.members)
            expect(owner.executor.begin_external_barrier(member.barrier) == CUDA_SUCCESS,
                   "all coherent prefixes can begin after final logical completion");
        for (const auto& member : cohort.members)
            expect(owner.executor.record_external_enqueue(member.barrier) == CUDA_SUCCESS,
                   "persistent predecessor does not merge the separate external tails");
        owner.finish();
    }
    native = {};
    {
        Owner owner({.isolated = true});
        const auto command = owner.kernel(1, 2);
        const std::array<std::uint64_t, 2> streams{0, 1};
        const auto cohort = owner.executor.accept_external_cohort(streams, owner.binding);
        expect(command.accepted() && cohort.status == CUDA_SUCCESS &&
                   owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt),
               "profile-isolated independent predecessor is enqueued but not retired");
        for (const auto& member : cohort.members)
            expect(owner.executor.external_barrier_ready(member.barrier) == CUDA_ERROR_NOT_READY,
                   "profile isolation remains stronger than independent-stream native FIFO");
        native.ready_all();
        expect(owner.executor.poll().retired == 1, "isolated predecessor has actual stop proof");
        for (const auto& member : cohort.members)
            expect(owner.executor.begin_external_barrier(member.barrier) == CUDA_SUCCESS,
                   "profile isolation ignores internal siblings after the real prior work retires");
        for (const auto& member : cohort.members)
            expect(owner.executor.record_external_enqueue(member.barrier) == CUDA_SUCCESS,
                   "profile-isolated siblings retain separate native enqueue boundaries");
        owner.finish();
    }
}

void cohort_member_preparation_failure() {
    native = {};
    Owner owner;
    auto lifetime = std::make_shared<int>(62);
    const std::weak_ptr<const void> weak = lifetime;
    const std::array<std::uint64_t, 2> streams{0, 1};
    const auto cohort = owner.executor.accept_external_cohort(streams, lifetime);
    lifetime.reset();
    expect(cohort.status == CUDA_SUCCESS && cohort.members.size() == 2,
           "member failure control starts with one atomically published cohort");
    native.create_status = CUDA_ERROR_OUT_OF_MEMORY;
    expect(
        owner.executor.begin_external_barrier(cohort.members[0].barrier) ==
                CUDA_ERROR_OUT_OF_MEMORY &&
            cohort.members[0].frontier.observe() == CUDA_ERROR_OUT_OF_MEMORY &&
            cohort.members[1].frontier.observe() == CUDA_ERROR_NOT_READY && !weak.expired() &&
            owner.executor.external_barrier_count() == 1 &&
            !owner.executor.snapshot().is_quarantined,
        "known no-event preparation failure settles only that member and retains the union lease");
    native.create_status = CUDA_SUCCESS;
    expect(owner.executor.external_barrier_ready(cohort.members[1].barrier) == CUDA_SUCCESS &&
               owner.executor.begin_external_barrier(cohort.members[1].barrier) == CUDA_SUCCESS &&
               owner.executor.record_external_enqueue(cohort.members[1].barrier) == CUDA_SUCCESS,
           "surviving member retains independent valid authority after known sibling preparation "
           "failure");
    owner.finish();
    expect(weak.expired() && native.creates == 2 && native.records == 1 && native.destroys == 1,
           "one known failed allocation and one real tail cause exactly one event destruction");
}

void pending_has_no_physical_proof() {
    native = {};
    Owner owner;
    auto lifetime = std::make_shared<int>(42);
    const std::weak_ptr<const void> weak = lifetime;
    auto accepted = owner.executor.accept_external_barrier(1, lifetime);
    lifetime.reset();
    expect(accepted.status == CUDA_SUCCESS && !weak.expired() &&
               owner.executor.external_barrier_count() == 1 &&
               owner.executor.snapshot().outstanding == 0 &&
               owner.executor.snapshot().unacknowledged_replies == 0 &&
               !owner.executor.has_ready_work() && !owner.executor.native_kernel_pair() &&
               owner.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
               owner.executor.capture_command_frontier(0).status == CUDA_ERROR_INVALID_HANDLE,
           "external acceptance has no application sequence, issue or host receipt");
    const auto all = owner.executor.capture_frontier();
    const auto stream = owner.executor.capture_frontier(1);
    const auto other = owner.executor.capture_frontier(2);
    expect(all.status == CUDA_SUCCESS && stream.status == CUDA_SUCCESS &&
               other.status == CUDA_SUCCESS && all.frontier.observe() == CUDA_ERROR_NOT_READY &&
               stream.frontier.observe() == CUDA_ERROR_NOT_READY &&
               other.frontier.observe() == CUDA_SUCCESS,
           "only context and exact stream frontiers include pending external work");
    expect(owner.executor.record_external_enqueue(accepted.barrier) == CUDA_ERROR_INVALID_VALUE &&
               owner.executor.fail_external_barrier(accepted.barrier, CUDA_ERROR_UNKNOWN) ==
                   CUDA_ERROR_INVALID_VALUE &&
               owner.executor.external_barrier_ready(accepted.barrier) == CUDA_SUCCESS,
           "record and fail cannot manufacture an unstarted native operation");
    for (int phase = 0; phase != 2; ++phase) {
        for (int repeat = 0; repeat != 3; ++repeat) {
            const auto polled = owner.executor.poll();
            const auto drained = owner.executor.drain_issued();
            expect(polled.retired == 0 && drained.retired == 0 && !weak.expired() &&
                       !owner.executor.physical_readers_quiescent() && native.queries == 0 &&
                       native.synchronizes == 0 && native.destroys == 0,
                   "no enqueue means no event query, stream drain or physical retirement");
        }
        if (phase == 0) {
            expect(owner.executor.begin_external_barrier(accepted.barrier) == CUDA_SUCCESS &&
                       native.creates == 1 && native.records == 0,
                   "begin creates one unrecorded event, not a native application call");
        }
    }
    expect(owner.executor.begin_external_barrier(accepted.barrier) == CUDA_ERROR_INVALID_VALUE &&
               owner.executor.cancel_external_barrier(accepted.barrier) == CUDA_ERROR_NOT_READY &&
               owner.executor.fail_external_barrier(accepted.barrier, CUDA_SUCCESS) ==
                   CUDA_ERROR_INVALID_VALUE &&
               native.creates == 1,
           "duplicate begin, started cancel and success-as-failure cannot change the phase");
    owner.executor.close();
    expect(owner.barrier().status == CUDA_ERROR_DEINITIALIZED && !owner.executor.shutdown() &&
               native.synchronizes == 0 && native.queries == 0 && !weak.expired(),
           "close cannot convert an unrecorded event into proof");
    auto copied_token = accepted.barrier;
    accepted.barrier = {};
    expect(owner.executor.record_external_enqueue(copied_token) == CUDA_SUCCESS &&
               native.records == 1 && native.events[0].stream == owner.stream(1) &&
               owner.executor.record_external_enqueue(copied_token) == CUDA_ERROR_INVALID_VALUE &&
               owner.executor.poll().retired == 0 && !weak.expired(),
           "trusted exact-stream enqueue permits only a real pending tail query");
    native.ready_all();
    expect(owner.executor.poll().retired == 1 && weak.expired() && native.destroys == 1 &&
               accepted.frontier.observe() == CUDA_SUCCESS &&
               all.frontier.observe() == CUDA_SUCCESS &&
               stream.frontier.observe() == CUDA_SUCCESS &&
               owner.executor.begin_external_barrier(copied_token) == CUDA_ERROR_INVALID_HANDLE &&
               native.unrecorded_queries == 0 && owner.executor.shutdown(),
           "exactly one physical tail retires the owner and consumes every token copy");
}

void sequence_continuity_and_native_pair() {
    native = {};
    Owner owner;
    const auto first = owner.kernel(1);
    auto lifetime = std::make_shared<int>(7);
    const std::weak_ptr<const void> weak = lifetime;
    const auto external = owner.executor.accept_external_barrier(1, lifetime);
    lifetime.reset();
    const auto second = owner.kernel(2);
    expect(first.accepted() && second.accepted() && first.receipt.sequence() == 1 &&
               second.receipt.sequence() == 2 && external.status == CUDA_SUCCESS &&
               owner.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_NOT_READY &&
               owner.executor.issue_next().sequence == 1 && native.launches == 1,
           "private acceptance order never consumes or fabricates an application sequence");
    const auto reply = owner.executor.host_result(first.receipt);
    expect(reply && reply->origin == ExecutionReplyOrigin::kNativeSdk && reply->status == 0 &&
               owner.executor.acknowledge_result(first.receipt) &&
               owner.executor.external_barrier_ready(external.barrier) == CUDA_SUCCESS &&
               owner.executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS &&
               !owner.executor.has_ready_work() &&
               owner.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
               !owner.executor.native_kernel_pair() && !weak.expired(),
           "prior original enqueue grants readiness, but its ACK is not external enqueue");
    expect(owner.executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS &&
               owner.executor.has_ready_work() && owner.executor.issue_next().sequence == 2 &&
               owner.executor.acknowledge_result(second.receipt) &&
               native.arguments == std::vector<std::uint32_t>{17, 34},
           "real external enqueue releases the copied ordinary successor ABI");
    const auto pair = owner.executor.native_kernel_pair();
    expect(pair && *pair == std::array<std::uint64_t, 2>{1, 2} &&
               owner.executor.snapshot().outstanding == 2 &&
               owner.executor.external_barrier_count() == 1 && !weak.expired(),
           "native pair audit names only the two issued ordinary kernels, never the barrier");
    owner.finish();
    expect(weak.expired() && native.launches == 2 && native.records == 3 && native.destroys == 3,
           "ordinary and external tails retire independently exactly once");
}

void stream_ordering_matrix() {
    // Rows independently specify CUDA legacy/blocking dependencies. Distinct
    // ordinary blocking streams do not depend on each other without legacy0.
    struct Row {
        std::uint64_t before, after;
        bool dependent;
    };
    constexpr std::array rows{Row{1, 1, true},  Row{0, 1, true},  Row{1, 0, true},
                              Row{0, 2, false}, Row{2, 0, false}, Row{1, 3, false},
                              Row{2, 2, true},  Row{1, 2, false}, Row{2, 1, false}};
    for (const auto row : rows) {
        native = {};
        Owner owner;
        const auto first = owner.barrier(row.before);
        const auto second = owner.barrier(row.after);
        expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS &&
                   owner.executor.external_barrier_ready(first.barrier) == CUDA_SUCCESS &&
                   owner.executor.external_barrier_ready(second.barrier) ==
                       (row.dependent ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS),
               "barrier readiness follows the independently enumerated stream dependency table");
        const auto direct = owner.executor.capture_frontier(row.after);
        const auto dependency = owner.executor.capture_frontier(row.after, true);
        expect(
            direct.status == CUDA_SUCCESS && dependency.status == CUDA_SUCCESS &&
                owner.executor.cancel_external_barrier(second.barrier) == CUDA_SUCCESS &&
                direct.frontier.observe() ==
                    (row.before == row.after ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS) &&
                dependency.frontier.observe() ==
                    (row.dependent ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS),
            "dependency frontiers include legacy predecessors but exact stream frontiers do not");
        const auto command = owner.kernel(1, row.after);
        expect(command.accepted() && owner.executor.has_ready_work() == !row.dependent,
               "an external predecessor gates ordinary issue using the same logical order");
        expect(
            owner.executor.cancel_external_barrier(first.barrier) == CUDA_SUCCESS &&
                first.frontier.observe() == CUDA_SUCCESS && native.creates == 0 &&
                owner.executor.external_barrier_ready(first.barrier) == CUDA_ERROR_INVALID_HANDLE,
            "pre-begin cancellation settles only known no-work without an SDK call");
        expect(owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt),
               "cancel releases ordinary issue");
        owner.finish();
    }
    for (const auto row : rows) {
        native = {};
        Owner owner;
        const auto command = owner.kernel(1, row.before);
        const auto external = owner.barrier(row.after);
        expect(command.accepted() && external.status == CUDA_SUCCESS &&
                   owner.executor.external_barrier_ready(external.barrier) ==
                       (row.dependent ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS),
               "queued ordinary predecessors use acceptance order and the same independent stream "
               "table");
        expect(owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt) &&
                   owner.executor.external_barrier_ready(external.barrier) == CUDA_SUCCESS &&
                   owner.executor.cancel_external_barrier(external.barrier) == CUDA_SUCCESS,
               "ordinary enqueue releases CPU readiness without claiming physical completion");
        owner.finish();
    }
}

void persistent_partial_and_profile_isolation() {
    native = {};
    {
        Owner owner;
        const auto progress = std::make_shared<Progress>();
        const std::uint32_t original = 41, prepared = 73;
        const auto first = owner.executor.accept_persistent_kernel(
            1,
            {.function = 1,
             .stream = 1,
             .grid = {6400, 1, 1},
             .block = {32, 1, 1},
             .parameters = std::as_bytes(std::span{&original, 1})},
            {owner.function(), 320, std::as_bytes(std::span{&prepared, 1}), progress});
        const auto external = owner.barrier();
        const auto independent = owner.barrier(2);
        expect(
            first.accepted() && external.status == CUDA_SUCCESS &&
                independent.status == CUDA_SUCCESS && owner.executor.issue_next().sequence == 1 &&
                owner.executor.acknowledge_result(first.receipt) &&
                owner.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.executor.external_barrier_ready(independent.barrier) == CUDA_SUCCESS &&
                owner.executor.cancel_external_barrier(independent.barrier) == CUDA_SUCCESS &&
                owner.executor.request_retreat(1) == CUDA_SUCCESS,
            "persistent enqueue is not logical completion, while nonblocking independence remains");
        native.ready_all();
        expect(
            owner.executor.poll().paused == 1 && progress->stops == 1 &&
                owner.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.executor.begin_external_barrier(external.barrier) == CUDA_ERROR_NOT_READY,
            "a proven incomplete stop cannot let an external successor overtake the logical task");
        progress->complete = true;
        expect(owner.executor.resume(1).state == CudaExecutionIssueState::kIssued,
               "continuation reuses the same accepted persistent task");
        native.ready_all();
        expect(owner.executor.poll().retired == 1 && progress->stops == 2 &&
                   owner.executor.external_barrier_ready(external.barrier) == CUDA_SUCCESS &&
                   native.arguments == std::vector<std::uint32_t>{73, 73} &&
                   owner.executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS &&
                   owner.executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS,
               "only final persistent completion enables the exact external successor");
        owner.finish();
    }
    native = {};
    {
        Owner owner;
        const auto external = owner.barrier();
        const auto progress = std::make_shared<Progress>();
        const std::uint32_t original = 41, prepared = 73;
        const auto successor = owner.executor.accept_persistent_kernel(
            1,
            {.function = 1,
             .stream = 1,
             .grid = {6400, 1, 1},
             .block = {32, 1, 1},
             .parameters = std::as_bytes(std::span{&original, 1})},
            {owner.function(), 320, std::as_bytes(std::span{&prepared, 1}), progress});
        expect(external.status == CUDA_SUCCESS && successor.accepted() &&
                   owner.executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS &&
                   owner.executor.issue_next().state == CudaExecutionIssueState::kIdle &&
                   native.launches == 0 && progress->prepares == 0 &&
                   owner.executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS &&
                   owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(successor.receipt) &&
                   native.arguments == std::vector<std::uint32_t>{73},
               "an external predecessor blocks private launch until actual enqueue, without native "
               "fallback");
        progress->complete = true;
        owner.finish();
        expect(progress->stops == 1 && native.destroys == 2,
               "private successor and external predecessor retain separate completion events");
    }
    native = {};
    {
        Owner owner({.isolated = true});
        const auto first = owner.kernel(1, 2);
        const auto external = owner.barrier(1);
        expect(
            first.accepted() && owner.executor.issue_next().sequence == 1 &&
                owner.executor.acknowledge_result(first.receipt) &&
                owner.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_NOT_READY,
            "profile isolation requires actual predecessor retirement across independent streams");
        native.ready_all();
        expect(owner.executor.poll().retired == 1 &&
                   owner.executor.external_barrier_ready(external.barrier) == CUDA_SUCCESS &&
                   owner.executor.cancel_external_barrier(external.barrier) == CUDA_SUCCESS,
               "isolated readiness changes only after real stop proof");
        owner.finish();
    }
}

void quotas_cancellation_and_close() {
    native = {};
    Owner owner({.commands = 2});
    const auto before = native.operations();
    expect(owner.executor.accept_external_barrier(99, owner.binding).status ==
               CUDA_ERROR_INVALID_HANDLE,
           "unknown stream cannot become external authority");
    expect(owner.executor.accept_external_barrier(1, {}).status == CUDA_ERROR_INVALID_VALUE &&
               native.operations() == before && owner.executor.external_barrier_count() == 0,
           "invalid stream or absent trusted lifetime rejects transactionally without SDK work");
    const auto first = owner.barrier();
    const auto second = owner.kernel(1, 2);
    expect(first.status == CUDA_SUCCESS && second.accepted() &&
               owner.barrier().status == CUDA_ERROR_OUT_OF_MEMORY &&
               owner.executor.snapshot().queued == 1 &&
               owner.executor.external_barrier_count() == 1,
           "external and normal records share one bounded command capacity");
    expect(owner.executor.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1 &&
               owner.executor.external_barrier_count() == 1 &&
               first.frontier.observe() == CUDA_ERROR_NOT_READY &&
               owner.executor.acknowledge_result(second.receipt) && native.creates == 0,
           "ordinary queued rejection never silently cancels an external actor");
    const auto third = owner.barrier(2);
    expect(third.status == CUDA_SUCCESS &&
               owner.executor.cancel_external_barrier(third.barrier) == CUDA_SUCCESS &&
               owner.executor.cancel_external_barrier(first.barrier) == CUDA_SUCCESS &&
               owner.executor.cancel_external_barrier(first.barrier) == CUDA_ERROR_INVALID_HANDLE &&
               native.creates == 0 && owner.executor.snapshot().outstanding == 0,
           "known cancellation refunds the bounded record without consuming sequence2");
    const auto command = owner.kernel(2);
    expect(command.accepted() && owner.executor.issue_next().sequence == 2 &&
               owner.executor.acknowledge_result(command.receipt),
           "sequence remains contiguous after cancellation");
    owner.finish();
    expect(owner.barrier().status == CUDA_ERROR_DEINITIALIZED,
           "closed executor cannot accept a barrier");
}

void frontier_capacity_refund() {
    native = {};
    Owner owner({.bindings = 4});
    std::array<CudaExecutionExternalAcceptance, 4> barriers;
    for (auto& accepted : barriers) {
        accepted = owner.barrier(2);
        expect(accepted.status == CUDA_SUCCESS, "each barrier consumes one bounded frontier slot");
    }
    expect(owner.barrier().status == CUDA_ERROR_OUT_OF_MEMORY,
           "frontier quota limits live observation handles");
    for (auto& accepted : barriers)
        expect(owner.executor.cancel_external_barrier(accepted.barrier) == CUDA_SUCCESS,
               "settled no-work frontiers retain observation identity");
    expect(owner.barrier().status == CUDA_ERROR_OUT_OF_MEMORY,
           "still-live settled frontier handles do not silently evade their quota");
    barriers = {};
    const auto reused = owner.barrier();
    expect(
        reused.status == CUDA_SUCCESS &&
            owner.executor.cancel_external_barrier(reused.barrier) == CUDA_SUCCESS,
        "expired frontier registrations refund bounded capacity without recycling token identity");
    owner.finish();
}

void owner_token_thread_and_process() {
    native = {};
    Owner first;
    const auto external = first.barrier();
    Owner second;
    const auto calls = native.operations();
    expect(
        second.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_INVALID_HANDLE &&
            second.executor.begin_external_barrier(external.barrier) == CUDA_ERROR_INVALID_HANDLE &&
            second.executor.record_external_enqueue(external.barrier) ==
                CUDA_ERROR_INVALID_HANDLE &&
            second.executor.cancel_external_barrier(external.barrier) ==
                CUDA_ERROR_INVALID_HANDLE &&
            second.executor.fail_external_barrier(external.barrier, CUDA_ERROR_UNKNOWN) ==
                CUDA_ERROR_INVALID_HANDLE &&
            native.operations() == calls,
        "another dispatcher cannot use a numeric-looking foreign token");
    first.activate();
    const std::array<std::uint64_t, 2> cohort_streams{1, 2};
    bool thread_ok{};
    std::thread thread([&] {
        thread_ok =
            first.barrier().status == CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.accept_external_cohort(cohort_streams, first.binding).status ==
                CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.begin_external_barrier(external.barrier) == CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.record_external_enqueue(external.barrier) ==
                CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.cancel_external_barrier(external.barrier) ==
                CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.fail_external_barrier(external.barrier, CUDA_ERROR_UNKNOWN) ==
                CUDA_ERROR_INVALID_CONTEXT &&
            external.frontier.observe() == CUDA_ERROR_INVALID_CONTEXT;
    });
    thread.join();
    expect(thread_ok && native.operations() == calls,
           "wrong thread rejects before every native callback");
    const auto child = fork();
    expect(child >= 0, "fork owner-scope control");
    if (child == 0) {
        const bool valid =
            first.barrier().status == CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.accept_external_cohort(cohort_streams, first.binding).status ==
                CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.begin_external_barrier(external.barrier) == CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.record_external_enqueue(external.barrier) ==
                CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.cancel_external_barrier(external.barrier) ==
                CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.fail_external_barrier(external.barrier, CUDA_ERROR_UNKNOWN) ==
                CUDA_ERROR_INVALID_CONTEXT &&
            first.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_INVALID_CONTEXT &&
            external.frontier.observe() == CUDA_ERROR_INVALID_CONTEXT &&
            native.operations() == calls;
        std::_Exit(valid ? 0 : 1);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects before inherited lock/native access");
    expect(first.executor.cancel_external_barrier(external.barrier) == CUDA_SUCCESS &&
               first.executor.begin_external_barrier({}) == CUDA_ERROR_INVALID_HANDLE,
           "empty and consumed tokens carry no authority");
    first.finish();
    second.activate();
    second.finish();
}

void mixed_drain_respects_unfinished_enqueue() {
    struct Row {
        std::uint64_t ordinary, external;
        bool blocks_drain;
    };
    constexpr std::array rows{Row{1, 1, true},  Row{0, 1, true},  Row{1, 0, true},
                              Row{2, 0, false}, Row{0, 2, false}, Row{1, 3, false}};
    for (const auto row : rows) {
        native = {};
        Owner owner;
        const auto command = owner.kernel(1, row.ordinary);
        expect(command.accepted() && owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt),
               "issue real ordinary predecessor");
        const auto external = owner.barrier(row.external);
        expect(external.status == CUDA_SUCCESS &&
                   owner.executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS,
               "external actor is started but its enqueue has not been proved");
        const auto first = owner.executor.drain_issued();
        expect(native.synchronizes == (row.blocks_drain ? 0U : 1U) &&
                   native.queries == (row.blocks_drain ? 1U : 0U) &&
                   first.retired == (row.blocks_drain ? 0U : 1U) &&
                   external.frontier.observe() == CUDA_ERROR_NOT_READY &&
                   native.unrecorded_queries == 0,
               "drain cannot synchronize a pending external suffix or its legacy dependency");
        native.ready_all();
        const auto second = owner.executor.drain_issued();
        expect(second.retired == (row.blocks_drain ? 1U : 0U) &&
                   native.synchronizes == (row.blocks_drain ? 0U : 1U) &&
                   owner.executor.external_barrier_count() == 1 && native.destroys == 1 &&
                   native.unrecorded_queries == 0,
               "the earlier recorded event can settle its command without settling later work");
        expect(owner.executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS,
               "trusted actor eventually proves exact-stream enqueue");
        expect(owner.executor.drain_issued().retired == 1 &&
                   native.synchronizes == (row.blocks_drain ? 1U : 2U) &&
                   external.frontier.observe() == CUDA_SUCCESS && owner.executor.shutdown(),
               "only after enqueue proof may blocking drain settle the actual external tail");
    }
}

void callback_exclusion() {
    for (const bool drain : {false, true}) {
        native = {};
        Owner owner;
        const auto external = owner.barrier();
        expect(external.status == CUDA_SUCCESS, "reentry control has one live owner token");
        const std::array<std::uint64_t, 2> cohort_streams{0, 1};
        std::array<unsigned, 7> visited{};
        bool correct = true;
        native.callback = [&](Callback site) {
            ++visited[static_cast<std::size_t>(site)];
            const auto before = native.operations();
            const auto count = owner.executor.external_barrier_count();
            correct &=
                owner.barrier().status == CUDA_ERROR_NOT_READY &&
                owner.executor.accept_external_cohort(cohort_streams, owner.binding).status ==
                    CUDA_ERROR_NOT_READY &&
                owner.executor.external_barrier_ready(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.executor.begin_external_barrier(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.executor.record_external_enqueue(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.executor.cancel_external_barrier(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.executor.fail_external_barrier(external.barrier, CUDA_ERROR_UNKNOWN) ==
                    CUDA_ERROR_NOT_READY &&
                owner.executor.poll().status == CUDA_ERROR_NOT_READY &&
                owner.executor.drain_issued().status == CUDA_ERROR_NOT_READY &&
                owner.executor.issue_next().native_status == CUDA_ERROR_NOT_READY &&
                owner.executor.capture_frontier().status == CUDA_ERROR_NOT_READY &&
                owner.executor.reject_queued(CUDA_ERROR_INVALID_VALUE) == 0 &&
                !owner.executor.physical_readers_quiescent() && !owner.executor.shutdown();
            owner.executor.close();
            correct &= native.operations() == before &&
                       owner.executor.external_barrier_count() == count &&
                       !owner.executor.snapshot().is_closing;
        };
        expect(owner.executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS &&
                   owner.executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS,
               "native begin and record callbacks finish despite rejected nested transitions");
        native.ready_all();
        const auto result = drain ? owner.executor.drain_issued() : owner.executor.poll();
        native.callback = {};
        expect(result.retired == 1 && correct &&
                   visited[static_cast<std::size_t>(Callback::kContext)] > 0 &&
                   visited[static_cast<std::size_t>(Callback::kCreate)] == 1 &&
                   visited[static_cast<std::size_t>(Callback::kRecord)] == 1 &&
                   visited[static_cast<std::size_t>(Callback::kDestroy)] == 1 &&
                   visited[static_cast<std::size_t>(drain ? Callback::kSynchronize
                                                          : Callback::kQuery)] == 1 &&
                   owner.executor.shutdown(),
               "exclusion covers initial/post proofs, event allocation/record/stop/destroy without "
               "nested SDK calls");
    }
}

void known_preparation_failure() {
    for (const bool partial_handle : {false, true}) {
        native = {};
        Owner owner;
        auto lifetime = std::make_shared<int>(5);
        const std::weak_ptr<const void> weak = lifetime;
        const auto external = owner.executor.accept_external_barrier(1, lifetime);
        lifetime.reset();
        native.create_status = CUDA_ERROR_OUT_OF_MEMORY;
        native.error_handle = partial_handle;
        expect(external.status == CUDA_SUCCESS &&
                   owner.executor.begin_external_barrier(external.barrier) ==
                       CUDA_ERROR_OUT_OF_MEMORY &&
                   external.frontier.observe() == CUDA_ERROR_OUT_OF_MEMORY &&
                   owner.executor.external_barrier_count() == 0 && weak.expired() &&
                   !owner.executor.snapshot().is_quarantined &&
                   owner.executor.snapshot().outstanding == 0 &&
                   owner.executor.snapshot().unacknowledged_replies == 0 && native.creates == 1 &&
                   native.destroys == (partial_handle ? 1U : 0U) && native.records == 0 &&
                   native.launches == 0 && native.queries == 0 &&
                   owner.executor.begin_external_barrier(external.barrier) ==
                       CUDA_ERROR_INVALID_HANDLE,
               "known failed event creation settles only preparation and verified one-time partial "
               "cleanup");
        native.create_status = CUDA_SUCCESS;
        native.error_handle = false;
        const auto command = owner.kernel(1);
        expect(command.accepted() && owner.executor.issue_next().sequence == 1 &&
                   owner.executor.acknowledge_result(command.receipt),
               "known private preparation failure neither poisons ownership nor consumes "
               "application sequence1");
        owner.finish();
    }
}

void current_context_gates() {
    native = {};
    Owner owner;
    const auto external = owner.barrier();
    expect(owner.executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS &&
               owner.executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS,
           "context gate control has a genuinely recorded native event");
    const auto queries = native.queries;
    char foreign{};
    native.current = reinterpret_cast<CUcontext>(&foreign);
    expect(owner.executor.poll().status == CUDA_ERROR_INVALID_CONTEXT &&
               owner.executor.drain_issued().status == CUDA_ERROR_INVALID_CONTEXT &&
               native.queries == queries && native.synchronizes == 0 &&
               !owner.executor.snapshot().is_quarantined &&
               external.frontier.observe() == CUDA_ERROR_NOT_READY,
           "collection entry rejects wrong current context before SDK and preserves retryable "
           "pending state");
    owner.activate();
    native.context_status = CUDA_ERROR_NOT_INITIALIZED;
    expect(owner.executor.poll().status == CUDA_ERROR_NOT_INITIALIZED && native.queries == queries,
           "failed initial context getter preserves its exact Driver status without tail access");
    native.context_status = CUDA_SUCCESS;
    owner.activate();
    owner.finish();
}

void accepted_before_close_can_finish() {
    native = {};
    Owner owner;
    const auto external = owner.barrier(0);
    expect(external.status == CUDA_SUCCESS, "legacy default stream is a bound external stream");
    owner.executor.close();
    const std::array<std::uint64_t, 2> streams{0, 1};
    const auto rejected = owner.executor.accept_external_cohort(streams, owner.binding);
    expect(
        owner.barrier().status == CUDA_ERROR_DEINITIALIZED &&
            rejected.status == CUDA_ERROR_DEINITIALIZED && rejected.members.empty() &&
            owner.executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS &&
            owner.executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS &&
            native.events[0].stream == nullptr && owner.executor.drain_issued().retired == 1 &&
            external.frontier.observe() == CUDA_SUCCESS && owner.executor.shutdown(),
        "close forbids new admission but permits the already accepted exact legacy tail to finish");
}

enum class Failure {
    kSuccessNull,
    kBeginPreContext,
    kCreatePostContext,
    kPartialDestroy,
    kPartialDestroyPostContext,
    kRecord,
    kRecordPreContext,
    kRecordPostContext,
    kQuery,
    kQueryPostContext,
    kQueryNotReadyPostContext,
    kDestroy,
    kDestroyPostContext,
    kDrain,
    kDrainPostContext,
    kCollectionPreRecordContext,
    kRetirementPreContext,
    kRetirementFinalContext,
    kExplicitFailure,
};

void uncertain_case(Failure failure) {
    native = {};
    auto owner = std::make_unique<Owner>();
    auto lifetime = std::make_shared<int>(8);
    const std::weak_ptr<const void> weak = lifetime;
    const auto external = owner->executor.accept_external_barrier(1, lifetime);
    lifetime.reset();
    expect(external.status == CUDA_SUCCESS, "uncertainty case accepted before native work");
    char foreign{};
    const auto lose_at = [&](Callback expected) {
        native.callback = [&, expected](Callback actual) {
            if (actual == expected) native.current = reinterpret_cast<CUcontext>(&foreign);
        };
    };
    CUresult expected = CUDA_ERROR_UNKNOWN;
    CUresult observed = CUDA_SUCCESS;
    if (failure == Failure::kSuccessNull) {
        native.null_create = true;
        observed = owner->executor.begin_external_barrier(external.barrier);
    } else if (failure == Failure::kBeginPreContext) {
        native.current = reinterpret_cast<CUcontext>(&foreign);
        expected = CUDA_ERROR_INVALID_CONTEXT;
        observed = owner->executor.begin_external_barrier(external.barrier);
        expect(native.creates == 0, "failed begin context proof stops before event allocation");
    } else if (failure == Failure::kCreatePostContext) {
        lose_at(Callback::kCreate);
        expected = CUDA_ERROR_INVALID_CONTEXT;
        observed = owner->executor.begin_external_barrier(external.barrier);
    } else if (failure == Failure::kPartialDestroy ||
               failure == Failure::kPartialDestroyPostContext) {
        native.create_status = CUDA_ERROR_OUT_OF_MEMORY;
        native.error_handle = true;
        if (failure == Failure::kPartialDestroy) {
            native.destroy_status = CUDA_ERROR_UNKNOWN;
        } else {
            lose_at(Callback::kDestroy);
        }
        // Cleanup uncertainty is the authoritative preparation failure, not an
        // invented return from the application operation that never ran.
        observed = owner->executor.begin_external_barrier(external.barrier);
        expected =
            failure == Failure::kPartialDestroy ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_CONTEXT;
    } else {
        expect(owner->executor.begin_external_barrier(external.barrier) == CUDA_SUCCESS,
               "real private event exists before recording/collection failure");
        if (failure == Failure::kExplicitFailure) {
            const auto calls = native.operations();
            expect(
                owner->executor.fail_external_barrier(external.barrier, CUDA_ERROR_LAUNCH_FAILED) ==
                        CUDA_SUCCESS &&
                    native.operations() == calls,
                "trusted explicit failure marks uncertainty with no SDK work or fake host result");
            observed = external.frontier.observe();
            expected = CUDA_ERROR_LAUNCH_FAILED;
        } else if (failure == Failure::kRecord || failure == Failure::kRecordPreContext ||
                   failure == Failure::kRecordPostContext) {
            if (failure == Failure::kRecord) native.record_status = CUDA_ERROR_UNKNOWN;
            if (failure == Failure::kRecordPreContext)
                native.context_status = CUDA_ERROR_INVALID_CONTEXT;
            if (failure == Failure::kRecordPostContext) lose_at(Callback::kRecord);
            if (failure != Failure::kRecord) expected = CUDA_ERROR_INVALID_CONTEXT;
            observed = owner->executor.record_external_enqueue(external.barrier);
        } else {
            expect(owner->executor.record_external_enqueue(external.barrier) == CUDA_SUCCESS,
                   "only actual recorded enqueue permits stop/cleanup observations");
            native.ready_all();
            if (failure == Failure::kQuery) native.query_status = CUDA_ERROR_UNKNOWN;
            if (failure == Failure::kQueryPostContext ||
                failure == Failure::kQueryNotReadyPostContext) {
                lose_at(Callback::kQuery);
                expected = CUDA_ERROR_INVALID_CONTEXT;
                if (failure == Failure::kQueryNotReadyPostContext) native.events[0].ready = false;
            }
            if (failure == Failure::kDestroy) native.destroy_status = CUDA_ERROR_UNKNOWN;
            if (failure == Failure::kDestroyPostContext) {
                lose_at(Callback::kDestroy);
                expected = CUDA_ERROR_INVALID_CONTEXT;
            }
            if (failure == Failure::kDrain) native.synchronize_status = CUDA_ERROR_UNKNOWN;
            if (failure == Failure::kDrainPostContext) {
                lose_at(Callback::kSynchronize);
                expected = CUDA_ERROR_INVALID_CONTEXT;
            }
            unsigned proof_calls{};
            if (failure == Failure::kCollectionPreRecordContext ||
                failure == Failure::kRetirementPreContext ||
                failure == Failure::kRetirementFinalContext) {
                const unsigned fail_on = failure == Failure::kCollectionPreRecordContext ? 2U
                                         : failure == Failure::kRetirementPreContext     ? 4U
                                                                                         : 6U;
                native.callback = [&, fail_on](Callback site) {
                    if (site == Callback::kContext && ++proof_calls == fail_on)
                        native.context_status = CUDA_ERROR_INVALID_CONTEXT;
                };
                expected = CUDA_ERROR_INVALID_CONTEXT;
            }
            observed = (failure == Failure::kDrain || failure == Failure::kDrainPostContext)
                           ? owner->executor.drain_issued().status
                           : owner->executor.poll().status;
        }
    }
    expect(observed == expected && owner->executor.snapshot().is_quarantined &&
               owner->executor.external_barrier_count() == 1 && !weak.expired() &&
               !owner->executor.physical_readers_quiescent() &&
               owner->executor.snapshot().outstanding == 0 &&
               owner->executor.snapshot().unacknowledged_replies == 0 &&
               external.frontier.observe() != CUDA_SUCCESS &&
               external.frontier.observe() != CUDA_ERROR_NOT_READY,
           "unknown boundary retains exact owners and a failed frontier without inventing native "
           "status");
    native.callback = {};
    native.context_status = CUDA_SUCCESS;
    native.create_status = CUDA_SUCCESS;
    native.record_status = CUDA_SUCCESS;
    native.query_status = CUDA_SUCCESS;
    native.destroy_status = CUDA_SUCCESS;
    native.synchronize_status = CUDA_SUCCESS;
    owner->activate();
    native.ready_all();
    const auto creates = native.creates;
    const auto records = native.records;
    const auto queries = native.queries;
    const auto destroys = native.destroys;
    const auto synchronizes = native.synchronizes;
    for (int repeat = 0; repeat != 3; ++repeat) {
        expect(owner->executor.poll().retired == 0 && owner->executor.drain_issued().retired == 0 &&
                   !owner->executor.shutdown() && !weak.expired(),
               "later permissive SDK responses cannot repair an unknown prior ownership boundary");
        (void)owner->executor.begin_external_barrier(external.barrier);
        (void)owner->executor.record_external_enqueue(external.barrier);
        (void)owner->executor.cancel_external_barrier(external.barrier);
        (void)owner->executor.fail_external_barrier(external.barrier, CUDA_ERROR_UNKNOWN);
    }
    expect(native.creates == creates && native.records == records && native.queries == queries &&
               native.destroys == destroys && native.synchronizes == synchronizes &&
               native.launches == 0,
           "uncertainty never replays enqueue, rechecks stop or retries private cleanup");
    owner.reset();
    expect(!weak.expired() && native.creates == creates && native.records == records &&
               native.queries == queries && native.destroys == destroys &&
               native.synchronizes == synchronizes,
           "dispatcher destruction remains CPU-only and retains uncertain native owners");
}

void isolated_uncertainty_matrix() {
    constexpr unsigned count = static_cast<unsigned>(Failure::kExplicitFailure) + 1;
    for (unsigned which = 0; which != count; ++which) {
        const auto child = fork();
        expect(child >= 0, "fork isolated permanent-retention control");
        if (child == 0) {
            try {
                uncertain_case(static_cast<Failure>(which));
                std::_Exit(0);
            } catch (const std::exception& error) {
                std::cerr << "external uncertainty case " << which << ": " << error.what() << '\n';
                std::_Exit(1);
            }
        }
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
               "isolated external uncertainty case " + std::to_string(which));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--uncertainty-case") {
            const auto which = std::stoul(argv[2]);
            expect(which <= static_cast<unsigned>(Failure::kExplicitFailure), "bounded case index");
            uncertain_case(static_cast<Failure>(which));
            std::_Exit(0);
        }
        const bool cohort_only = argc == 2 && std::string_view(argv[1]) == "--cohort";
        expect(argc == 1 || cohort_only, "no unexpected test arguments");
        cohort_atomic_admission();
        cohort_union_readiness_matrix();
        cohort_intervening_producer_and_successors();
        cohort_distinct_tail_lifetimes();
        cohort_persistent_and_profile_isolation();
        cohort_member_preparation_failure();
        if (cohort_only) {
            owner_token_thread_and_process();
            callback_exclusion();
            accepted_before_close_can_finish();
            std::cout << "external cohort: 9 groups, 20 union-readiness rows; "
                         "GPU_initialized=false\n";
            return 0;
        }
        pending_has_no_physical_proof();
        sequence_continuity_and_native_pair();
        stream_ordering_matrix();
        persistent_partial_and_profile_isolation();
        quotas_cancellation_and_close();
        frontier_capacity_refund();
        owner_token_thread_and_process();
        mixed_drain_respects_unfinished_enqueue();
        callback_exclusion();
        known_preparation_failure();
        current_context_gates();
        accepted_before_close_can_finish();
        isolated_uncertainty_matrix();
        std::cout << "external barrier: 19 groups, 19 isolated uncertainty cases; "
                     "GPU_initialized=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
