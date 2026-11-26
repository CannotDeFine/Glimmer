#include "internal/cuda_nccl_coordinator.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
using Kind = CudaNcclNativeOperation;
using Phase = CudaNcclCoordinatorPhase;
void expect(bool value, std::string_view message) {
    if (value) return;
    std::cerr << "NCCL context coordinator: " << message << '\n';
    std::_Exit(1);
}
template <typename Predicate>
void await(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!predicate()) {
        expect(std::chrono::steady_clock::now() < deadline, "bounded CPU fixture progress");
        std::this_thread::yield();
    }
}
struct Event {
    bool live{}, recorded{};
};
struct Native {
    std::array<char, 4> contexts{};
    std::array<char, 8> comms{};
    std::array<Event, 32> events{};
    std::atomic<unsigned> sets{}, gets{}, calls{}, records{};
    std::atomic<bool> entered{}, release{true};
    CUcontext expected_entry{}, post{};
    ncclResult_t result{ncclSuccess};
    unsigned next_comm{}, next_event{}, fail_set{}, fail_get{}, mismatch_set{};
    bool hold{}, change_context{true}, immediate_init{};
    CUcontext context(std::size_t index) {
        return reinterpret_cast<CUcontext>(&contexts.at(index));
    }
} native;
thread_local CUcontext current{};
thread_local unsigned depth{};
thread_local std::vector<ncclComm_t*> deferred;
CUresult CUDAAPI set_current(CUcontext value) {
    const auto index = ++native.sets;
    current = index == native.mismatch_set ? native.context(3) : value;
    return index == native.fail_set ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* output) {
    const auto index = ++native.gets;
    *output = current;
    return index == native.fail_get ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
}
ncclResult_t visit() {
    expect(current == native.expected_entry, "packet sees actual requested caller entry");
    ++native.calls;
    native.entered = true;
    if (native.hold) await([] { return native.release.load(); });
    if (native.change_context) current = native.post;
    return native.result;
}
ncclResult_t uid(ncclUniqueId* output) {
    if (output) std::memset(output, 0x37, sizeof(*output));
    return visit();
}
ncclResult_t init(ncclComm_t* output, int, ncclUniqueId, int) {
    if (output) {
        if (depth && !native.immediate_init)
            deferred.push_back(output);
        else {
            expect(native.next_comm < native.comms.size(), "bounded fake communicator storage");
            *output = reinterpret_cast<ncclComm_t>(&native.comms[native.next_comm++]);
        }
    }
    return visit();
}
ncclResult_t init_config(ncclComm_t* output, int ranks, ncclUniqueId id, int rank,
                         ncclConfig_t* config) {
    expect(!config, "original NULL config");
    return init(output, ranks, id, rank);
}
ncclResult_t query(ncclComm_t, ncclResult_t* output) {
    if (output) *output = ncclSuccess;
    return visit();
}
ncclResult_t destroy(ncclComm_t) {
    return visit();
}
ncclResult_t abort_comm(ncclComm_t) {
    return visit();
}
ncclResult_t start() {
    if (native.result == ncclSuccess) ++depth;
    return visit();
}
ncclResult_t end() {
    if (depth && native.result == ncclSuccess && !--depth) {
        for (auto* output : deferred) {
            expect(native.next_comm < native.comms.size(), "bounded deferred fake output");
            *output = reinterpret_cast<ncclComm_t>(&native.comms[native.next_comm++]);
        }
        deferred.clear();
    }
    return visit();
}
ncclResult_t gather(const void*, void*, std::size_t, ncclDataType_t, ncclComm_t, cudaStream_t) {
    return visit();
}
ncclResult_t reduce(const void*, void*, std::size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t,
                    cudaStream_t) {
    return visit();
}
ncclResult_t bcast(void*, std::size_t, ncclDataType_t, int, ncclComm_t, cudaStream_t) {
    return visit();
}
CudaNcclNativeDispatch dispatch() {
    return {uid, init, init_config, query, destroy, abort_comm, start, end, gather, reduce, bcast};
}
CUresult CUDAAPI event_create(CUevent* output, unsigned) {
    expect(native.next_event < native.events.size(), "bounded event storage");
    auto& event = native.events[native.next_event++];
    event.live = true;
    *output = reinterpret_cast<CUevent>(&event);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_record(CUevent value, CUstream) {
    auto& event = *reinterpret_cast<Event*>(value);
    expect(event.live && !event.recorded, "posted prefix recorded once");
    event.recorded = true;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_query(CUevent value) {
    const auto& event = *reinterpret_cast<Event*>(value);
    expect(event.live && event.recorded, "no unrecorded prefix query");
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_destroy(CUevent value) {
    auto& event = *reinterpret_cast<Event*>(value);
    expect(event.live, "one event cleanup");
    event.live = false;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI synchronize(CUstream) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI launch(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned, CUstream, void**, void**) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
struct Source {
    std::shared_ptr<const void> lifetime = std::make_shared<int>(1);
    CudaExecutionDispatcher dispatcher;
    Source()
        : dispatcher({5, 6}, {50, native.context(1), lifetime},
                     {get_current, launch, event_create, event_record, event_query, event_destroy,
                      synchronize}) {
        current = native.context(1);
        expect(dispatcher.bind_executor() == CUDA_SUCCESS, "bind real source dispatcher");
    }
    void complete(CudaNcclCoordinator& coordinator, std::uint64_t operation) {
        current = native.context(1);
        const auto tail = dispatcher.capture_frontier();
        expect(tail.status == CUDA_SUCCESS && tail.frontier.observe() == CUDA_SUCCESS,
               "empty fake source is independently physically quiescent");
        expect(coordinator.bind_tails(operation, std::span{&tail.frontier, 1}) == CUDA_SUCCESS &&
                   coordinator.advance(false) == CUDA_SUCCESS,
               "real frontier, not caller completion boolean");
    }
    void shutdown() {
        current = native.context(1);
        expect(dispatcher.poll().status == CUDA_SUCCESS && dispatcher.shutdown(),
               "source prefix native retirement");
    }
};
struct Owner {
    std::shared_ptr<const void> lifetime = std::make_shared<int>(2);
    CudaNcclCoordinator coordinator{dispatch(), {set_current, get_current}};
    std::uint64_t sequence{};
    CudaNcclCoordinatorDomain domain(std::size_t index) const {
        return {
            100 + index, static_cast<int>(index), {200 + index, native.context(index), lifetime}};
    }
    CudaNcclCoordinatorRequest request(Kind kind) const {
        CudaNcclCoordinatorRequest result;
        result.caller = {19, {21, 23}};
        result.sequence = sequence + 1;
        result.domain = domain(0);
        result.caller_context = domain(1);
        result.arguments.operation = kind;
        result.arguments.ranks = 1;
        result.arguments.resources_lifetime = lifetime;
        if (kind == Kind::kAllGather || kind == Kind::kAllReduce || kind == Kind::kBcast) {
            result.prepare_enqueue = [] { return CUDA_SUCCESS; };
            result.record_enqueue = [] {
                ++native.records;
                return CUDA_SUCCESS;
            };
        }
        return result;
    }
    std::uint64_t submit(CudaNcclCoordinatorRequest request) {
        const auto accepted = coordinator.submit(std::move(request));
        expect(accepted.carrier == CUDA_SUCCESS && accepted.operation, "actual lane admission");
        ++sequence;
        return accepted.operation;
    }
    CudaNcclCoordinatorObservation returned(std::uint64_t operation) {
        await([&] {
            expect(coordinator.advance(false) == CUDA_SUCCESS, "CPU-only collection");
            return coordinator.observe(operation).reply_ready;
        });
        return coordinator.observe(operation);
    }
    void shutdown() {
        expect(coordinator.close() == CUDA_SUCCESS, "close coordinator");
        await([&] {
            const auto status = coordinator.shutdown();
            expect(status == CUDA_SUCCESS || status == CUDA_ERROR_NOT_READY, "shutdown status");
            return status == CUDA_SUCCESS;
        });
    }
};
void transition(const CudaNcclCoordinatorObservation& observation, CudaNcclContextKind before,
                CudaNcclContextKind after) {
    expect(observation.context_transition.has_value(), "actual context observation retained");
    expect(observation.context_transition->before.kind == before &&
               observation.context_transition->after.kind == after,
           "actual typed identities, never operation-derived effect");
}
void all_operations() {
    Owner owner;
    Source source;
    native.expected_entry = native.context(1);
    for (unsigned raw = 1; raw <= 11; ++raw) {
        const auto kind = static_cast<Kind>(raw);
        native.post = raw % 3 == 0 ? nullptr : raw % 3 == 1 ? native.context(1) : native.context(0);
        auto request = owner.request(kind);
        const auto operation = owner.submit(std::move(request));
        const auto result = owner.returned(operation);
        expect(result.carrier == CUDA_SUCCESS && result.original_status == 0,
               "all eleven real callbacks preserve original result");
        transition(result, CudaNcclContextKind::kCaller,
                   !native.post                       ? CudaNcclContextKind::kNull
                   : native.post == native.context(1) ? CudaNcclContextKind::kCaller
                                                      : CudaNcclContextKind::kDomain);
        if (raw >= 9) source.complete(owner.coordinator, operation);
        expect(owner.coordinator.release_reply(operation) == CUDA_SUCCESS, "release reply");
    }
    expect(native.calls == 11, "no original replay");
    source.shutdown();
    owner.shutdown();
}
void null_and_legacy() {
    Owner owner;
    auto request = owner.request(Kind::kGetUniqueId);
    request.caller_context = CudaNcclCoordinatorDomain{};
    native.expected_entry = nullptr;
    native.post = native.context(0);
    auto operation = owner.submit(std::move(request));
    auto result = owner.returned(operation);
    transition(result, CudaNcclContextKind::kNull, CudaNcclContextKind::kDomain);
    expect(result.carrier == CUDA_SUCCESS &&
               owner.coordinator.release_reply(operation) == CUDA_SUCCESS,
           "explicit NULL differs from absent observation");
    request = owner.request(Kind::kGetUniqueId);
    request.caller_context.reset();
    native.expected_entry = native.context(0);
    operation = owner.submit(std::move(request));
    result = owner.returned(operation);
    expect(result.carrier == CUDA_SUCCESS && !result.context_transition &&
               owner.coordinator.release_reply(operation) == CUDA_SUCCESS,
           "legacy fixed-domain request unchanged");
    owner.shutdown();
}
void nested_member_domains() {
    Owner owner;
    native.expected_entry = native.context(1);
    native.post = native.context(0);
    for (unsigned index = 0; index != 2; ++index) {
        const auto id = owner.submit(owner.request(Kind::kGroupStart));
        expect(owner.returned(id).original_status == 0 &&
                   owner.coordinator.release_reply(id) == CUDA_SUCCESS,
               "actual nested group starts");
    }
    auto request = owner.request(Kind::kInitRank);
    request.domain = owner.domain(2);
    request.arguments.has_output = true;
    native.post = native.context(2);
    const auto inner = owner.submit(std::move(request));
    const auto result = owner.returned(inner);
    expect(result.reserved_communicator && !result.output, "group init output is delayed");
    expect(owner.coordinator.release_reply(inner) == CUDA_SUCCESS,
           "released inner remains retained");
    native.post = native.context(0);
    auto id = owner.submit(owner.request(Kind::kGroupEnd));
    expect(owner.returned(id).publications.empty() &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS,
           "inner End publishes no outer outputs");
    native.post = native.context(2);
    id = owner.submit(owner.request(Kind::kGroupEnd));
    const auto outer = owner.returned(id);
    transition(outer, CudaNcclContextKind::kCaller, CudaNcclContextKind::kDomain);
    expect(outer.carrier == CUDA_SUCCESS && outer.context_transition->after.identity == 102 &&
               outer.publications.size() == 1 && outer.publications[0].operation == inner &&
               outer.publications[0].output->communicator,
           "actual member domain and delayed native publication survive outer End");
    expect(owner.coordinator.release_reply(id) == CUDA_SUCCESS, "release outer reply");
    request = owner.request(Kind::kDestroy);
    request.domain = owner.domain(2);
    request.communicator = result.reserved_communicator;
    id = owner.submit(std::move(request));
    expect(owner.returned(id).original_status == 0 &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS,
           "actual copied owner destruction");
    owner.shutdown();
}
void pinned_grouped_initialization() {
    Owner owner;
    native.expected_entry = native.context(1);
    native.post = native.context(0);
    native.immediate_init = true;
    auto request = owner.request(Kind::kGroupStart);
    request.arguments.output_contract = CudaNcclNativeOutputContract::kPinnedHostStable;
    auto id = owner.submit(std::move(request));
    expect(owner.returned(id).original_status == 0 &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS,
           "pinned original native Start enters the stable worker TLS");
    request = owner.request(Kind::kInitRank);
    request.arguments.has_output = true;
    request.arguments.output_contract = CudaNcclNativeOutputContract::kPinnedHostStable;
    const auto inner = owner.submit(std::move(request));
    const auto initialized = owner.returned(inner);
    expect(initialized.carrier == CUDA_SUCCESS && initialized.original_status == 0 &&
               initialized.reserved_communicator && initialized.output &&
               initialized.output->output_written == true && initialized.output->communicator &&
               !initialized.physical_complete && !initialized.enqueue_complete &&
               owner.coordinator.release_reply(inner) == CUDA_SUCCESS,
           "actual pinned group-inner packet publishes proved synchronous output without group "
           "completion");
    request = owner.request(Kind::kGroupEnd);
    request.arguments.output_contract = CudaNcclNativeOutputContract::kPinnedHostStable;
    id = owner.submit(std::move(request));
    const auto ended = owner.returned(id);
    expect(ended.carrier == CUDA_SUCCESS && ended.original_status == 0 &&
               ended.publications.size() == 1 && ended.publications[0].operation == inner &&
               ended.publications[0].output &&
               ended.publications[0].output->output_written == true &&
               ended.publications[0].output->communicator == initialized.output->communicator &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS,
           "exact same durable pinned cell remains owned and observable at actual outer End");
    request = owner.request(Kind::kDestroy);
    request.communicator = initialized.reserved_communicator;
    id = owner.submit(std::move(request));
    expect(owner.returned(id).original_status == 0 &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS && native.calls == 4,
           "actual destruction retires the one published symbolic owner without replay");
    owner.shutdown();
}

void active_source_turn() {
    Owner owner;
    Source source;
    native.expected_entry = native.context(1);
    native.post = native.context(0);
    native.hold = true;
    native.release = false;
    const auto accepted = source.dispatcher.accept_enqueue_turn(0, source.lifetime);
    expect(accepted.status == CUDA_SUCCESS &&
               source.dispatcher.begin_enqueue_turn(accepted.turn) == CUDA_SUCCESS,
           "source owner grants real host turn");
    auto request = owner.request(Kind::kGetUniqueId);
    request.source_turns.push_back(accepted.worker);
    const auto id = owner.submit(std::move(request));
    await([] { return native.entered.load(); });
    expect(source.dispatcher.has_active_enqueue_turn() &&
               source.dispatcher.finish_enqueue_turn(accepted.turn) == CUDA_ERROR_NOT_READY,
           "blocked original callback is not a terminal ticket");
    const auto snapshot = owner.coordinator.snapshot();
    expect(snapshot.carrier == CUDA_SUCCESS && snapshot.operations.size() == 1 &&
               snapshot.operations[0].active_call && snapshot.operations[0].has_source_turns &&
               snapshot.operations[0].devices == std::vector<int>({0, 1}),
           "snapshot binds active source exclusion and both gated devices");
    native.release = true;
    expect(owner.returned(id).original_status == 0, "actual held native return");
    expect(source.dispatcher.finish_enqueue_turn(accepted.turn) == CUDA_SUCCESS &&
               !source.dispatcher.has_active_enqueue_turn(),
           "once-bound exact ticket proves posted-prefix release");
    expect(owner.coordinator.release_reply(id) == CUDA_SUCCESS, "release host receipt");
    source.shutdown();
    owner.shutdown();
}
void authorized_continuation() {
    Owner owner;
    Source source;
    native.expected_entry = native.context(1);
    native.post = native.context(0);
    auto request = owner.request(Kind::kInitRank);
    request.arguments.has_output = true;
    auto id = owner.submit(std::move(request));
    const auto init_result = owner.returned(id);
    const auto comm = init_result.reserved_communicator;
    expect(comm && owner.coordinator.release_reply(id) == CUDA_SUCCESS, "initialize real owner");
    native.result = ncclInProgress;
    request = owner.request(Kind::kAllReduce);
    request.communicator = comm;
    const auto batch = owner.submit(std::move(request));
    expect(owner.returned(batch).phase == Phase::kWaitingAsync && native.records == 0,
           "actual InProgress reserves batch without outgoing record");
    native.result = ncclSuccess;
    request = owner.request(Kind::kGetAsyncError);
    request.communicator = comm;
    request.arguments.has_output = true;
    id = owner.submit(std::move(request));
    expect(owner.returned(id).original_status == 0, "real async query");
    for (unsigned index = 0; index != 32; ++index)
        expect(owner.coordinator.advance(false) == CUDA_SUCCESS, "CPU-only progress");
    const auto snapshot = owner.coordinator.snapshot();
    expect(snapshot.carrier == CUDA_SUCCESS && native.records == 0,
           "CPU observation cannot issue unauthorized continuation");
    bool ready{};
    for (const auto& operation : snapshot.operations)
        if (operation.operation == batch) ready = operation.ready_continuation;
    expect(ready && owner.coordinator.continue_enqueue(batch) == CUDA_SUCCESS &&
               owner.coordinator.continue_enqueue(batch) == CUDA_ERROR_NOT_READY,
           "one exact granted continuation, no duplicate");
    await([&] {
        expect(owner.coordinator.advance(false) == CUDA_SUCCESS, "collect granted record");
        return owner.coordinator.observe(batch).enqueue_complete;
    });
    expect(native.records == 1 && native.calls == 3, "record once without packet replay");
    source.complete(owner.coordinator, batch);
    expect(owner.coordinator.release_reply(batch) == CUDA_SUCCESS &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS,
           "release retained replies");
    request = owner.request(Kind::kDestroy);
    request.communicator = comm;
    id = owner.submit(std::move(request));
    expect(owner.returned(id).original_status == 0 &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS,
           "native communicator cleanup");
    source.shutdown();
    owner.shutdown();
}
void entry_device_exclusion() {
    Owner owner;
    native.expected_entry = native.context(1);
    native.post = native.context(1);
    native.hold = true;
    native.release = false;
    const auto first = owner.submit(owner.request(Kind::kGetUniqueId));
    await([] { return native.entered.load(); });
    auto request = owner.request(Kind::kGetUniqueId);
    request.caller.thread = {31, 33};
    request.sequence = 1;
    request.domain = owner.domain(1);
    const auto second = owner.coordinator.submit(std::move(request));
    expect(second.carrier == CUDA_SUCCESS, "independent caller accepted on entry's device");
    for (unsigned index = 0; index != 64; ++index) {
        expect(owner.coordinator.advance(false) == CUDA_SUCCESS, "nonblocking owner while held");
        expect(native.calls == 1 && !owner.coordinator.observe(second.operation).reply_ready,
               "entry device is gated with primary device before any original entry");
        std::this_thread::yield();
    }
    native.release = true;
    expect(owner.returned(first).carrier == CUDA_SUCCESS &&
               owner.returned(second.operation).carrier == CUDA_SUCCESS && native.calls == 2,
           "both callers progress once after complete device-set release");
    expect(owner.coordinator.release_reply(first) == CUDA_SUCCESS &&
               owner.coordinator.release_reply(second.operation) == CUDA_SUCCESS,
           "release both exact callbacks");
    owner.shutdown();
}
void invalid_entries() {
    Owner owner;
    auto request = owner.request(Kind::kGetUniqueId);
    request.caller_context->context.lifetime.reset();
    expect(owner.coordinator.submit(request).carrier == CUDA_ERROR_INVALID_VALUE,
           "missing entry lease rejects before callback");
    request.caller_context = CudaNcclCoordinatorDomain{};
    request.caller_context->identity = 9;
    expect(owner.coordinator.submit(request).carrier == CUDA_ERROR_INVALID_VALUE,
           "noncanonical NULL rejects");
    request = owner.request(Kind::kGetUniqueId);
    request.source_turns.resize(257);
    expect(owner.coordinator.submit(request).carrier == CUDA_ERROR_INVALID_VALUE &&
               native.calls == 0 && native.sets == 0,
           "raw turn bound before lane selection");
    native.expected_entry = native.context(1);
    native.post = native.context(0);
    const auto id = owner.submit(owner.request(Kind::kGetUniqueId));
    expect(owner.returned(id).original_status == 0 &&
               owner.coordinator.release_reply(id) == CUDA_SUCCESS,
           "rejection did not consume sequence");
    owner.shutdown();
}
void native_error(unsigned raw) {
    Owner owner;
    native.expected_entry = native.context(1);
    native.post = native.context(0);
    native.result = ncclInvalidArgument;
    const auto id = owner.submit(owner.request(static_cast<Kind>(raw)));
    const auto result = owner.returned(id);
    expect(result.original_status == static_cast<int>(ncclInvalidArgument) && native.calls == 1,
           "all eleven native errors keep actual original, never retry");
    transition(result, CudaNcclContextKind::kCaller, CudaNcclContextKind::kDomain);
    // Deliberate uncertain native group/collective owners remain child-contained.
}
void failure(unsigned scenario) {
    Owner owner;
    native.expected_entry = native.context(1);
    native.post = native.context(1);
    native.result = ncclInvalidArgument;
    switch (scenario) {
        case 0:
            native.fail_set = 2;
            break;
        case 1:
            native.fail_get = 3;
            break;
        case 2:
            native.post = native.context(3);
            break;
        case 3:
            native.fail_get = 4;
            break;
        case 4:
            native.fail_set = 3;
            break;
        case 5:
            native.fail_get = 5;
            break;
        case 6:
            native.fail_get = 6;
            break;
        case 7:
            native.mismatch_set = 3;
            break;
        case 8:
            native.fail_set = 3;
            native.result = ncclSuccess;
            break;
        default:
            expect(false, "known failure fixture");
    }
    auto request = owner.request(Kind::kGetUniqueId);
    request.arguments.output_contract = CudaNcclNativeOutputContract::kPinnedHostStable;
    request.arguments.has_output = true;
    const auto id = owner.submit(std::move(request));
    const auto result = owner.returned(id);
    expect(result.phase == Phase::kUncertain && result.carrier != CUDA_SUCCESS,
           "independent native/proof failure retains uncertainty");
    const bool invoked = scenario >= 2;
    expect(result.original_status.has_value() == invoked && native.calls == (invoked ? 1U : 0U),
           "never invent original on pre-entry failure");
    if (invoked)
        expect(result.original_status == static_cast<int>(native.result),
               "later fault never overwrites original SDK status, including success");
    expect(result.output.has_value() == invoked,
           "pinned output snapshot precedes fallible later context observations");
    if (invoked)
        expect(
            result.output->output_written == true &&
                result.output->unique_id.front() == std::byte{0x37},
            "actual native write/value survives later context proof loss independently of status");
    expect(result.context_transition.has_value() == (scenario >= 4),
           "known actual transition survives later fault; failed observation has no effect");
    if (scenario == 2 || scenario == 3)
        expect(native.sets == 2, "no restore/select after unknown context or failed observation");
    const auto snapshot = owner.coordinator.snapshot();
    expect(snapshot.carrier == CUDA_SUCCESS && snapshot.operations[0].uncertain &&
               owner.coordinator.continue_enqueue(id) == CUDA_ERROR_NOT_READY,
           "uncertainty is visible, never a continuation grant");
}
void invalid_source_endpoint() {
    Owner owner;
    native.expected_entry = native.context(1);
    native.post = native.context(0);
    auto request = owner.request(Kind::kGetUniqueId);
    request.source_turns.emplace_back();
    const auto id = owner.submit(std::move(request));
    const auto result = owner.returned(id);
    expect(result.phase == Phase::kUncertain && !result.original_status && native.calls == 0 &&
               native.sets <= 1 && !result.context_transition,
           "bad source endpoint retains admitted ticket without original or context selection");
}
template <typename Function>
void isolated(Function function, bool has_intentional_retention = false) {
    const auto child = ::fork();
    expect(child >= 0, "CPU child creation");
    if (!child) {
        function();
        if (has_intentional_retention) std::_Exit(0);
        std::exit(EXIT_SUCCESS);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated deterministic case passed");
}
}  // namespace
int main() {
    isolated(all_operations);
    isolated(null_and_legacy);
    isolated(nested_member_domains);
    isolated(pinned_grouped_initialization);
    isolated(active_source_turn);
    isolated(authorized_continuation);
    isolated(entry_device_exclusion);
    isolated(invalid_entries);
    for (unsigned raw = 1; raw <= 11; ++raw) isolated([raw] { native_error(raw); }, true);
    for (unsigned scenario = 0; scenario != 9; ++scenario)
        isolated([scenario] { failure(scenario); }, true);
    isolated(invalid_source_endpoint, true);
    std::cout << "NCCL context coordinator: 8 groups and 21 isolated failures passed\n";
}
