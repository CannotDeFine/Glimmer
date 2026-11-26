#include "internal/cuda_nccl_coordinator.h"
#include "internal/cuda_execution_session.h"
#include "internal/cuda_external_execution_bridge.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
using Kind = CudaNcclNativeOperation;
using Phase = CudaNcclCoordinatorPhase;
void expect(bool value, std::string_view message) {
    if (value) return;
    // Never unwind fixture storage through an intentionally retained worker.
    std::cerr << "NCCL coordinator: " << message << '\n';
    std::_Exit(1);
}
template <typename Predicate>
void await(Predicate predicate, std::string_view message) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!predicate()) {
        expect(std::chrono::steady_clock::now() < deadline, message);
        std::this_thread::yield();
    }
}
struct NativeComm {
    bool live{};
};
struct Context {
    int device{};
};
struct Event {
    bool live{}, recorded{}, ready{};
    CUcontext context{};
};
struct Native {
    std::array<Context, 3> contexts{{{0}, {1}, {0}}};
    std::array<NativeComm, 32> comms{};
    std::array<Event, 32> events{};
    std::array<std::atomic<unsigned>, 12> calls{};
    std::atomic<unsigned> next_comm{}, prepares{}, records{}, proofs{}, event_queries{};
    unsigned next_event{};
    ncclResult_t returned{ncclSuccess};
    int async_state{};
    bool return_null{}, loses_context{};
    std::function<void()> callback;
    std::function<void(Kind)> collective_callback;
    CudaNcclCoordinator* coordinator{};
    ncclComm_t make_comm() {
        const auto index = next_comm.fetch_add(1);
        expect(index < comms.size(), "bounded native communicator pool");
        comms[index].live = true;
        return reinterpret_cast<ncclComm_t>(&comms[index]);
    }
    void visit(Kind kind) {
        ++calls[static_cast<unsigned>(kind)];
        if (coordinator) {
            expect(coordinator->advance() == CUDA_ERROR_INVALID_CONTEXT &&
                       coordinator->close() == CUDA_ERROR_INVALID_CONTEXT,
                   "native worker cannot reenter mutable owner coordinator");
        }
        if (callback) callback();
    }
} native;
thread_local CUcontext current{};
thread_local unsigned group_depth{};
thread_local std::vector<ncclComm_t*> pending_init;
thread_local std::vector<ncclComm_t> pending_destroy;

CUresult CUDAAPI set_current(CUcontext context) {
    current = context;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* output) {
    ++native.proofs;
    *output = current;
    return native.loses_context ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
}
ncclResult_t unique_id(ncclUniqueId* output) {
    native.visit(Kind::kGetUniqueId);
    if (output) std::memset(output, 0x3a, sizeof(*output));
    return native.returned;
}
ncclResult_t initialize(ncclComm_t* output, int ranks, ncclUniqueId id, int rank) {
    native.visit(Kind::kInitRank);
    expect(ranks == 2 && rank == 1 && id.internal[0] == 0x22, "exact signed init/UID ABI");
    if (output) {
        if (group_depth)
            pending_init.push_back(output);
        else if (!native.return_null)
            *output = native.make_comm();
    }
    return native.returned;
}
ncclResult_t initialize_config(ncclComm_t* output, int ranks, ncclUniqueId id, int rank,
                               ncclConfig_t* config) {
    native.visit(Kind::kInitRankConfig);
    expect(ranks == 2 && rank == 1 && id.internal[0] == 0x22 && !config,
           "Config NULL remains native NULL, not fabricated defaults");
    if (output) {
        if (group_depth)
            pending_init.push_back(output);
        else if (!native.return_null)
            *output = native.make_comm();
    }
    return native.returned;
}
ncclResult_t async_error(ncclComm_t communicator, ncclResult_t* output) {
    native.visit(Kind::kGetAsyncError);
    if (communicator)
        expect(reinterpret_cast<NativeComm*>(communicator)->live,
               "no query after native destruction");
    if (output) std::memcpy(output, &native.async_state, sizeof(native.async_state));
    return native.returned;
}
ncclResult_t destroy(ncclComm_t communicator) {
    native.visit(Kind::kDestroy);
    if (communicator) {
        expect(reinterpret_cast<NativeComm*>(communicator)->live, "destroy actual live owner once");
        if (group_depth)
            pending_destroy.push_back(communicator);
        else if (native.returned == ncclSuccess)
            reinterpret_cast<NativeComm*>(communicator)->live = false;
    }
    return native.returned;
}
ncclResult_t abort_comm(ncclComm_t communicator) {
    native.visit(Kind::kAbort);
    if (communicator) {
        expect(reinterpret_cast<NativeComm*>(communicator)->live, "abort actual live owner once");
        if (group_depth)
            pending_destroy.push_back(communicator);
        else if (native.returned == ncclSuccess)
            reinterpret_cast<NativeComm*>(communicator)->live = false;
    }
    return native.returned;
}
ncclResult_t group_start() {
    native.visit(Kind::kGroupStart);
    ++group_depth;
    return native.returned;
}
ncclResult_t group_end() {
    native.visit(Kind::kGroupEnd);
    if (!group_depth) return ncclInvalidUsage;
    if (--group_depth == 0) {
        for (auto* cell : pending_init)
            if (!native.return_null) *cell = native.make_comm();
        pending_init.clear();
        for (const auto communicator : pending_destroy)
            reinterpret_cast<NativeComm*>(communicator)->live = false;
        pending_destroy.clear();
    }
    return native.returned;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected NCCL ABI.
ncclResult_t gather(const void* send, void* receive, std::size_t count, ncclDataType_t type,
                    ncclComm_t communicator, cudaStream_t stream) {
    native.visit(Kind::kAllGather);
    if (native.collective_callback) native.collective_callback(Kind::kAllGather);
    expect(send && receive && count == 4 && type == ncclFloat32 && stream,
           "exact collective input ABI, no numeric vendor emulation claim");
    if (communicator)
        expect(reinterpret_cast<NativeComm*>(communicator)->live, "collective live owner");
    return native.returned;
}
ncclResult_t reduce(const void* send, void* receive, std::size_t count, ncclDataType_t type,
                    ncclRedOp_t reduction, ncclComm_t communicator, cudaStream_t stream) {
    native.visit(Kind::kAllReduce);
    if (native.collective_callback) native.collective_callback(Kind::kAllReduce);
    expect(send && receive && count == 4 && type == ncclFloat32 && reduction == ncclSum && stream,
           "exact AllReduce inputs");
    if (communicator)
        expect(reinterpret_cast<NativeComm*>(communicator)->live, "reduce live owner");
    return native.returned;
}
ncclResult_t broadcast(void* buffer, std::size_t count, ncclDataType_t type, int root,
                       ncclComm_t communicator, cudaStream_t stream) {
    native.visit(Kind::kBcast);
    if (native.collective_callback) native.collective_callback(Kind::kBcast);
    expect(buffer && count == 4 && type == ncclFloat32 && root == 1 && stream,
           "exact Bcast inputs");
    if (communicator) expect(reinterpret_cast<NativeComm*>(communicator)->live, "Bcast live owner");
    return native.returned;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CudaNcclNativeDispatch dispatch() {
    return {unique_id, initialize, initialize_config, async_error,
            destroy,   abort_comm, group_start,       group_end,
            gather,    reduce,     broadcast};
}
enum class CallerSlot : unsigned { kFirst, kSecond, kThird };
struct Owner {
    std::shared_ptr<const void> context_lifetime = std::make_shared<int>(9);
    std::array<std::uint64_t, 8> sequences{};
    std::array<float, 8> buffers{};
    char stream{};
    CudaNcclCoordinator coordinator;
    explicit Owner(CudaNcclCoordinatorLimits limits = {})
        : coordinator(dispatch(), {set_current, get_current}, limits) {
        native.coordinator = &coordinator;
    }
    ~Owner() {
        native.coordinator = nullptr;
    }
    CudaNcclCoordinatorRequest request(Kind kind, std::uint64_t comm = 0,
                                       CallerSlot caller = CallerSlot::kFirst) {
        CudaNcclCoordinatorRequest result;
        const auto caller_index = static_cast<unsigned>(caller);
        result.caller = {17, {caller_index + 1, 1}};
        result.sequence = sequences[caller_index] + 1;
        result.domain = {
            70, 0, {80, reinterpret_cast<CUcontext>(&native.contexts[0]), context_lifetime}};
        result.communicator = comm;
        result.arguments.operation = kind;
        result.arguments.has_output = true;
        result.arguments.ranks = 2;
        result.arguments.rank = 1;
        result.arguments.root = 1;
        result.arguments.unique_id.fill(std::byte{0x22});
        result.arguments.count = 4;
        result.arguments.datatype = ncclFloat32;
        result.arguments.send = buffers.data();
        result.arguments.receive = buffers.data() + 4;
        result.arguments.stream = reinterpret_cast<cudaStream_t>(&stream);
        result.arguments.resources_lifetime = std::make_shared<int>(4);
        return result;
    }
    CudaNcclCoordinatorAcceptance submit(CudaNcclCoordinatorRequest request) {
        const auto caller = static_cast<std::size_t>(request.caller.thread[0] - 1);
        const auto result = coordinator.submit(std::move(request));
        expect(result.carrier == CUDA_SUCCESS && result.operation, "actual coordinator submission");
        ++sequences[caller];
        return result;
    }
    CudaNcclCoordinatorObservation returned(std::uint64_t id) {
        await(
            [&] {
                expect(coordinator.advance() == CUDA_SUCCESS, "nonblocking owner advancement");
                return coordinator.observe(id).reply_ready;
            },
            "bounded native callback return");
        auto result = coordinator.observe(id);
        expect(result.original_status.has_value(), "requested result is actual native status");
        return result;
    }
    CudaNcclCoordinatorAcceptance call(Kind kind, std::uint64_t comm = 0,
                                       CallerSlot caller = CallerSlot::kFirst) {
        const auto accepted = submit(request(kind, comm, caller));
        expect(returned(accepted.operation).original_status == native.returned,
               "exact requested NCCL status");
        return accepted;
    }
    void release(std::uint64_t id) {
        expect(coordinator.release_reply(id) == CUDA_SUCCESS, "release exact host reply");
    }
    void shutdown() {
        expect(coordinator.close() == CUDA_SUCCESS, "close admission without SDK teardown");
        await(
            [&] {
                expect(coordinator.advance() == CUDA_SUCCESS, "finish known owner transitions");
                const auto status = coordinator.shutdown();
                expect(status == CUDA_SUCCESS || status == CUDA_ERROR_NOT_READY,
                       "bounded known shutdown status");
                return status == CUDA_SUCCESS;
            },
            "only known retired native ownership permits coordinator shutdown");
    }
};
void incoming_hook(CudaNcclCoordinatorRequest& request) {
    request.prepare_enqueue = [] {
        ++native.prepares;
        return CUDA_SUCCESS;
    };
}
void outgoing_hook(CudaNcclCoordinatorRequest& request) {
    request.record_enqueue = [] {
        ++native.records;
        return CUDA_SUCCESS;
    };
}
void hooks(CudaNcclCoordinatorRequest& request) {
    incoming_hook(request);
    outgoing_hook(request);
}

CUresult CUDAAPI create_event(CUevent* output, unsigned) {
    expect(native.next_event < native.events.size(), "bounded physical event inventory");
    auto& event = native.events[native.next_event++];
    event = {.live = true, .context = current};
    *output = reinterpret_cast<CUevent>(&event);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI record_event(CUevent value, CUstream) {
    auto& event = *reinterpret_cast<Event*>(value);
    expect(event.live && !event.recorded && event.context == current,
           "physical event recorded once on its owner");
    event.recorded = true;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI query_event(CUevent value) {
    ++native.event_queries;
    const auto& event = *reinterpret_cast<Event*>(value);
    expect(event.live && event.recorded && event.context == current,
           "never query an unrecorded or foreign native tail");
    return event.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI destroy_event(CUevent value) {
    auto& event = *reinterpret_cast<Event*>(value);
    expect(event.live && event.context == current, "event cleanup exactly once on its owner");
    event.live = false;
    return CUDA_SUCCESS;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact unused Driver callback ABI.
CUresult CUDAAPI launch(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned, CUstream, void**, void**) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
CUresult CUDAAPI synchronize(CUstream) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
struct Tails {
    std::shared_ptr<const void> lifetime = std::make_shared<int>(11);
    char stream{};
    CudaExecutionDispatcher dispatcher;
    CudaExecutionExternalCohortAcceptance cohort;
    explicit Tails(std::size_t count = 1)
        : dispatcher({1, 2}, {90, reinterpret_cast<CUcontext>(&native.contexts[2]), lifetime},
                     {get_current, launch, create_event, record_event, query_event, destroy_event,
                      synchronize}) {
        activate();
        expect(dispatcher.bind_executor() == CUDA_SUCCESS &&
                   dispatcher.bind_stream(1, reinterpret_cast<CUstream>(&stream), lifetime),
               "real owner dispatcher binds physical source streams");
        const std::array<std::uint64_t, 2> streams{0, 1};
        cohort = dispatcher.accept_external_cohort(std::span{streams}.first(count), lifetime);
        expect(cohort.status == CUDA_SUCCESS && cohort.members.size() == count,
               "actual coherent source tails");
        for (const auto& member : cohort.members)
            expect(dispatcher.begin_external_barrier(member.barrier) == CUDA_SUCCESS,
                   "begin exact ready source boundary");
    }
    void activate() {
        current = reinterpret_cast<CUcontext>(&native.contexts[2]);
    }
    void bind(CudaNcclCoordinator& coordinator, std::uint64_t id) {
        std::vector<CudaExecutionFrontier> frontiers;
        for (const auto& member : cohort.members) frontiers.push_back(member.frontier);
        expect(coordinator.bind_tails(id, frontiers) == CUDA_SUCCESS &&
                   coordinator.bind_tails(id, frontiers) == CUDA_ERROR_INVALID_VALUE,
               "bind actual immutable tails once, never a loose completion boolean");
    }
    void record() {
        activate();
        for (const auto& member : cohort.members)
            expect(dispatcher.record_external_enqueue(member.barrier) == CUDA_SUCCESS,
                   "record tail only after enqueue proof");
    }
    void finish() {
        activate();
        for (auto& event : native.events) event.ready = true;
        expect(dispatcher.poll().status == CUDA_SUCCESS && dispatcher.shutdown(),
               "actual physical retirement and cleanup");
    }
};

CudaNativeExecutionCallObservation housekeeping_returned(Owner& owner, std::uint64_t id) {
    await(
        [&] {
            expect(owner.coordinator.advance() == CUDA_SUCCESS,
                   "observe housekeeping without owner SDK calls");
            const auto phase = owner.coordinator.observe_housekeeping(id).phase;
            return phase != CudaNativeExecutionCallPhase::kPending &&
                   phase != CudaNativeExecutionCallPhase::kEntered;
        },
        "bounded housekeeping callback completion");
    // Raw ticket publication may follow the loop's last owner advance. Commit
    // that terminal observation before asserting reusable caller permissions.
    expect(owner.coordinator.advance() == CUDA_SUCCESS,
           "owner consumes the returned housekeeping ticket");
    return owner.coordinator.observe_housekeeping(id);
}

// Required unused session dispatch entries reject if accidentally reached. The
// source fixture captures a real default-stream/context capsule, not fake ranges.
CudaSessionDispatch bridge_source_dispatch() {
    CudaSessionDispatch result;
    result.initialize = [](unsigned) { return CUDA_SUCCESS; };
    result.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    result.context_create = [](CUcontext* output, unsigned, CUdevice) {
        current = *output = reinterpret_cast<CUcontext>(&native.contexts[2]);
        return CUDA_SUCCESS;
    };
    result.context_set_current = set_current;
    result.context_destroy = [](CUcontext context) {
        expect(context == current && context == reinterpret_cast<CUcontext>(&native.contexts[2]),
               "source context cleanup is exact");
        current = nullptr;
        return CUDA_SUCCESS;
    };
    result.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.function_get = [](CUfunction*, CUmodule, const char*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    result.allocate = [](CUdeviceptr*, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.release = [](CUdeviceptr) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.upload = [](CUdeviceptr, const void*, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.download = [](void*, CUdeviceptr, std::size_t) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.stream_create = [](CUstream*, unsigned) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.stream_destroy = [](CUstream) { return CUDA_ERROR_NOT_SUPPORTED; };
    result.execution = {
        get_current, launch, create_event, record_event, query_event, destroy_event, [](CUstream) {
            return CUDA_SUCCESS;
        }};
    return result;
}
CUresult CUDAAPI wait_bridge_event(CUstream, CUevent handle, unsigned flags) {
    const auto& event = *reinterpret_cast<Event*>(handle);
    expect(flags == 0 && event.live && event.recorded && event.context != current,
           "actual bridge waits on recorded event from the other domain");
    return CUDA_SUCCESS;
}

void eleven_routes_and_tails() {
    Owner owner;
    const auto uid = owner.call(Kind::kGetUniqueId);
    expect(owner.coordinator.observe(uid.operation).output->unique_id[127] == std::byte{0x3a},
           "complete UID snapshot");
    owner.release(uid.operation);
    const auto init = owner.call(Kind::kInitRank);
    const auto communicator = init.reserved_communicator;
    expect(communicator && owner.coordinator.observe(init.operation).output->communicator,
           "reserved identity publishes actual native owner");
    owner.release(init.operation);
    const auto query = owner.call(Kind::kGetAsyncError, communicator, CallerSlot::kSecond);
    expect(owner.coordinator.observe(query.operation).output->async_error == 0,
           "cross-caller sequential owned query");
    owner.release(query.operation);
    for (const auto kind : {Kind::kAllGather, Kind::kAllReduce, Kind::kBcast}) {
        Tails tails;
        auto request = owner.request(kind, communicator, CallerSlot::kSecond);
        auto resources = std::make_shared<int>(99);
        const std::weak_ptr<int> weak = resources;
        request.arguments.resources_lifetime = resources;
        hooks(request);
        const auto accepted = owner.submit(std::move(request));
        resources.reset();
        tails.bind(owner.coordinator, accepted.operation);
        expect(owner.returned(accepted.operation).enqueue_complete && !weak.expired(),
               "native success proves enqueue, not stop");
        owner.release(accepted.operation);
        expect(owner.coordinator.advance() == CUDA_SUCCESS && !weak.expired(),
               "reply release retains all physical resources");
        tails.record();
        expect(owner.coordinator.advance() == CUDA_SUCCESS && !weak.expired(),
               "recording is not tail completion");
        tails.finish();
        expect(owner.coordinator.advance() == CUDA_SUCCESS && weak.expired(),
               "actual every-tail success releases resources");
    }
    owner.release(owner.call(Kind::kDestroy, communicator, CallerSlot::kSecond).operation);
    const auto next = owner.call(Kind::kInitRankConfig);
    owner.release(next.operation);
    owner.release(owner.call(Kind::kAbort, next.reserved_communicator).operation);
    owner.shutdown();
}

void grouped_publications_and_repeated_collectives() {
    Owner owner;
    owner.release(owner.call(Kind::kGroupStart).operation);
    owner.release(owner.call(Kind::kGroupStart).operation);
    const auto first = owner.call(Kind::kInitRank);
    const auto second = owner.call(Kind::kInitRankConfig);
    expect(!owner.coordinator.observe(first.operation).output &&
               !owner.coordinator.observe(second.operation).output,
           "inner init return does not read deferred native output cells");
    owner.release(first.operation);
    owner.release(second.operation);
    owner.release(owner.call(Kind::kGroupEnd).operation);
    expect(!owner.coordinator.observe(first.operation).output,
           "nested End is not publication boundary");
    const auto ended = owner.call(Kind::kGroupEnd);
    const auto receipt = owner.coordinator.observe(ended.operation);
    expect(receipt.publications.size() == 2 &&
               receipt.publications[0].operation == first.operation &&
               receipt.publications[1].reserved_communicator == second.reserved_communicator &&
               receipt.publications[0].output->communicator &&
               receipt.publications[1].output->communicator,
           "outer End owns both actual deferred publications after inner replies released");
    owner.release(ended.operation);
    owner.release(owner.call(Kind::kGroupStart, 0, CallerSlot::kSecond).operation);
    const auto before_prepare = native.prepares.load();
    const auto before_record = native.records.load();
    for (unsigned index = 0; index != 2; ++index) {
        auto request =
            owner.request(Kind::kAllReduce, first.reserved_communicator, CallerSlot::kSecond);
        incoming_hook(request);
        const auto reduce = owner.submit(std::move(request));
        expect(owner.returned(reduce.operation).original_status == 0,
               "each grouped collective executes after its incoming hook");
        owner.release(reduce.operation);
    }
    expect(native.prepares == before_prepare + 2 && native.records == before_record,
           "repeated group inners prepare independently without an outgoing tail");
    Tails tails(2);
    auto end_request = owner.request(Kind::kGroupEnd, 0, CallerSlot::kSecond);
    outgoing_hook(end_request);
    const auto end = owner.submit(std::move(end_request));
    tails.bind(owner.coordinator, end.operation);
    expect(owner.returned(end.operation).enqueue_complete &&
               native.prepares == before_prepare + 2 && native.records == before_record + 1,
           "whole group executes only the aggregate outgoing hook at outer End");
    owner.release(end.operation);
    tails.record();
    tails.finish();
    expect(owner.coordinator.advance() == CUDA_SUCCESS,
           "retire all grouped members after both source tails");
    owner.release(owner.call(Kind::kGroupStart).operation);
    owner.release(owner.call(Kind::kDestroy, first.reserved_communicator).operation);
    owner.release(owner.call(Kind::kAbort, second.reserved_communicator).operation);
    const auto queries = native.calls[4].load();
    expect(
        owner.coordinator.submit(owner.request(Kind::kGetAsyncError, first.reserved_communicator))
                .carrier == CUDA_ERROR_INVALID_HANDLE,
        "destructive entry seals even before outer End");
    owner.release(owner.call(Kind::kGroupEnd).operation);
    expect(native.calls[4] == queries, "no post-destructive query to prove terminal ownership");
    owner.shutdown();
}

void eager_grouped_incoming_prefixes() {
    struct EagerState {
        std::atomic<unsigned> produced{}, prepared{}, consumed{}, outgoing{};
        std::array<std::atomic<unsigned>, 3> output{};
    };
    constexpr std::array kinds{Kind::kAllReduce, Kind::kAllGather, Kind::kBcast};
    constexpr std::array<unsigned, 3> values{17, 41, 83};
    const auto state = std::make_shared<EagerState>();
    Owner owner;
    const auto init = owner.call(Kind::kInitRank);
    owner.release(init.operation);
    const auto before_prepare = native.prepares.load();
    const auto before_record = native.records.load();
    auto invalid_metadata = owner.request(Kind::kGetUniqueId);
    incoming_hook(invalid_metadata);
    expect(
        owner.coordinator.submit(std::move(invalid_metadata)).carrier == CUDA_ERROR_INVALID_VALUE,
        "noncollective metadata cannot carry an incoming enqueue hook");
    owner.release(owner.call(Kind::kGroupStart).operation);
    owner.release(owner.call(Kind::kGroupStart).operation);
    const auto before_collective = native.calls[static_cast<unsigned>(Kind::kAllReduce)].load();
    for (unsigned variant = 0; variant != 3; ++variant) {
        auto invalid = owner.request(Kind::kAllReduce, init.reserved_communicator);
        if (variant != 0) incoming_hook(invalid);
        if (variant == 1) outgoing_hook(invalid);
        if (variant == 2) invalid.arguments.resources_lifetime.reset();
        expect(owner.coordinator.submit(std::move(invalid)).carrier == CUDA_ERROR_INVALID_VALUE,
               "grouped inner requires incoming owner proof and forbids an early outgoing hook");
    }
    expect(native.prepares == before_prepare && native.records == before_record &&
               native.calls[static_cast<unsigned>(Kind::kAllReduce)] == before_collective,
           "rejected hook shapes run neither private endpoints nor requested native callbacks");

    // The provider deliberately performs work inside each collective, while its
    // native TLS group is still open. GroupEnd is not an incoming dependency.
    // These immutable endpoints retain only fixture payload, never a session,
    // frontier or coordinator. Atomic markers model distinct producer prefixes.
    native.collective_callback = [state, kinds, values, before_record](Kind kind) {
        const auto index = state->consumed.load();
        expect(index < kinds.size() && kind == kinds[index] && group_depth != 0 &&
                   state->produced == index + 1 && state->prepared == index + 1 &&
                   state->outgoing == 0 && native.records == before_record,
               "eager inner native work requires its own incoming prefix before first access");
        state->output[index] = values[index];
        state->consumed = index + 1;
    };
    std::array<std::weak_ptr<int>, 3> retained;
    for (unsigned index = 0; index != kinds.size(); ++index) {
        // Later producers remain possible between native inner calls: no
        // outgoing source barrier or CPU wait is installed by an inner call.
        state->produced = index + 1;
        auto request = owner.request(kinds[index], init.reserved_communicator);
        const auto sequence = request.sequence;
        auto resources = std::make_shared<int>(static_cast<int>(index));
        retained[index] = resources;
        request.arguments.resources_lifetime = resources;
        request.prepare_enqueue = [state, index]() -> CUresult {
            expect(state->produced == index + 1 && state->prepared == index &&
                       state->consumed == index && state->outgoing == 0,
                   "per-inner incoming endpoint observes the fresh intervening producer");
            ++native.prepares;
            state->prepared = index + 1;
            return CUDA_SUCCESS;
        };
        const auto accepted = owner.submit(std::move(request));
        resources.reset();
        const auto observed = owner.returned(accepted.operation);
        expect(owner.sequences[0] == sequence && observed.original_status == 0 &&
                   !observed.enqueue_complete && !observed.physical_complete &&
                   state->output[index] == values[index] && !retained[index].expired(),
               "rejected admissions consume no sequence; eager native result is not group stop");
        owner.release(accepted.operation);
        if (index == 0) {
            auto invalid_end = owner.request(Kind::kGroupEnd);
            outgoing_hook(invalid_end);
            expect(owner.coordinator.submit(std::move(invalid_end)).carrier ==
                       CUDA_ERROR_INVALID_VALUE,
                   "nested End cannot publish an aggregate outgoing event");
            owner.release(owner.call(Kind::kGroupEnd).operation);
            expect(state->outgoing == 0 && native.records == before_record,
                   "nested End does not close the outer enqueue frontier");
        }
    }
    native.collective_callback = {};
    for (unsigned variant = 0; variant != 3; ++variant) {
        auto invalid_end = owner.request(Kind::kGroupEnd);
        if (variant != 0) outgoing_hook(invalid_end);
        if (variant == 1) incoming_hook(invalid_end);
        if (variant == 2) invalid_end.arguments.resources_lifetime.reset();
        expect(owner.coordinator.submit(std::move(invalid_end)).carrier == CUDA_ERROR_INVALID_VALUE,
               "outer End requires retained outgoing authority and forbids a late incoming hook");
    }
    Tails tails;
    auto end_request = owner.request(Kind::kGroupEnd);
    end_request.record_enqueue = [state]() -> CUresult {
        expect(group_depth == 0 && state->prepared == 3 && state->consumed == 3 &&
                   state->outgoing == 0,
               "outgoing endpoint follows the actual outer native End exactly once");
        ++native.records;
        ++state->outgoing;
        return CUDA_SUCCESS;
    };
    const auto ended = owner.submit(std::move(end_request));
    tails.bind(owner.coordinator, ended.operation);
    expect(owner.returned(ended.operation).enqueue_complete &&
               native.prepares == before_prepare + 3 && native.records == before_record + 1 &&
               state->outgoing == 1,
           "all three eager native entries are prefixed independently and joined once at End");
    owner.release(ended.operation);
    for (const auto& resource : retained)
        expect(!resource.expired(), "released inner replies retain every source owner until tail");
    tails.record();
    tails.finish();
    expect(owner.coordinator.advance() == CUDA_SUCCESS,
           "actual tail retirement, not eager return, releases the group");
    for (const auto& resource : retained)
        expect(resource.expired(), "all inner source owners release after the exact physical tail");
    owner.release(owner.call(Kind::kDestroy, init.reserved_communicator).operation);
    owner.shutdown();
}

void incoming_failure_retains_group() {
    const auto child = ::fork();
    expect(child >= 0, "isolate uncertain incoming prefix failure");
    if (child == 0) {
        Owner owner;
        const auto init = owner.call(Kind::kInitRank);
        owner.release(init.operation);
        owner.release(owner.call(Kind::kGroupStart).operation);
        const auto calls = native.calls[static_cast<unsigned>(Kind::kAllReduce)].load();
        const auto prepares = native.prepares.load();
        const auto records = native.records.load();
        auto resources = std::make_shared<int>(77);
        const std::weak_ptr<int> weak = resources;
        auto request = owner.request(Kind::kAllReduce, init.reserved_communicator);
        request.arguments.resources_lifetime = resources;
        request.prepare_enqueue = []() -> CUresult {
            ++native.prepares;
            return CUDA_ERROR_LAUNCH_FAILED;
        };
        const auto failed = owner.submit(std::move(request));
        resources.reset();
        await(
            [&] {
                expect(owner.coordinator.advance() == CUDA_SUCCESS,
                       "owner observes failed private incoming callback");
                return owner.coordinator.observe(failed.operation).reply_ready;
            },
            "bounded incoming failure observation");
        const auto observed = owner.coordinator.observe(failed.operation);
        expect(observed.phase == Phase::kUncertain &&
                   observed.carrier == CUDA_ERROR_LAUNCH_FAILED && !observed.original_status &&
                   !observed.enqueue_complete && !observed.physical_complete &&
                   native.calls[static_cast<unsigned>(Kind::kAllReduce)] == calls &&
                   native.prepares == prepares + 1 && native.records == records && !weak.expired(),
               "failed incoming proof prevents native entry without fabricating native status");
        owner.release(failed.operation);
        expect(owner.coordinator.advance() == CUDA_SUCCESS && !weak.expired() &&
                   native.prepares == prepares + 1 && native.records == records &&
                   owner.coordinator.close() == CUDA_SUCCESS &&
                   owner.coordinator.shutdown() == CUDA_ERROR_NOT_READY,
               "uncertain source ownership and open native group remain retained without replay");
        std::_Exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "incoming failure retains uncertain state only until isolated process exit");
}

void async_progress_and_native_errors() {
    Owner owner;
    native.returned = ncclInProgress;
    const auto init = owner.call(Kind::kInitRankConfig);
    expect(owner.coordinator.observe(init.operation).phase == Phase::kWaitingAsync &&
               owner.coordinator.observe(init.operation).output->communicator,
           "direct InProgress preserves real owner and original requested status");
    owner.release(init.operation);
    native.returned = ncclSuccess;
    native.async_state = 7;
    owner.release(owner.call(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond)
                      .operation);
    expect(owner.coordinator.observe(init.operation).phase == Phase::kWaitingAsync,
           "async7 is not ready");
    native.async_state = 0;
    owner.release(owner.call(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond)
                      .operation);
    expect(owner.coordinator.advance() == CUDA_SUCCESS,
           "native query0 advances initialization without replay");
    Tails tails;
    auto request = owner.request(Kind::kAllGather, init.reserved_communicator);
    hooks(request);
    native.returned = ncclInProgress;
    const auto accepted = owner.submit(std::move(request));
    tails.bind(owner.coordinator, accepted.operation);
    const auto record_count = native.records.load();
    expect(owner.returned(accepted.operation).phase == Phase::kWaitingAsync &&
               native.records == record_count,
           "InProgress cannot install an early tail");
    native.returned = ncclRemoteError;
    native.async_state = 0;
    const auto failed_query =
        owner.call(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond);
    owner.release(failed_query.operation);
    expect(!owner.coordinator.observe(accepted.operation).enqueue_complete &&
               native.records == record_count,
           "native query error with written0 cannot promote enqueue proof");
    native.returned = ncclSuccess;
    owner.release(owner.call(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond)
                      .operation);
    await(
        [&] {
            expect(owner.coordinator.advance() == CUDA_SUCCESS,
                   "internal original-lane continuation");
            return owner.coordinator.observe(accepted.operation).enqueue_complete;
        },
        "async record continuation");
    expect(native.records == record_count + 1,
           "actual successful async query records once, without replay");
    owner.release(accepted.operation);
    tails.record();
    tails.finish();
    expect(owner.coordinator.advance() == CUDA_SUCCESS, "retire async physical owners");
    owner.release(
        owner.call(Kind::kDestroy, init.reserved_communicator, CallerSlot::kSecond).operation);
    owner.shutdown();
}

void authentication_limits_and_null() {
    CudaNcclCoordinatorLimits limits;
    limits.lanes.maximum_lanes = 2;
    limits.maximum_communicators = 1;
    Owner owner(limits);
    for (unsigned key = 1; key != 7; ++key) {
        auto bad = owner.request(Kind::kGetUniqueId);
        bad.caller.thread[0] = key + 100;
        bad.sequence = 2;
        expect(owner.coordinator.submit(std::move(bad)).carrier == CUDA_ERROR_INVALID_VALUE,
               "rejected first sequences cannot exhaust caller quota");
    }
    const auto init = owner.call(Kind::kInitRank);
    owner.release(init.operation);
    expect(owner.coordinator.submit(owner.request(Kind::kInitRank)).carrier ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "communicator reservation quota before native init");
    const auto calls = native.calls[4].load();
    for (unsigned variant = 0; variant != 6; ++variant) {
        auto bad =
            owner.request(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond);
        if (variant == 0) bad.caller.peer = 999;
        if (variant == 1) ++bad.domain.identity;
        if (variant == 2) ++bad.domain.context.id;
        if (variant == 3)
            bad.domain.context.native = reinterpret_cast<CUcontext>(&native.contexts[1]);
        if (variant == 4) bad.domain.context.lifetime = std::make_shared<int>(9);
        if (variant == 5) bad.communicator = 999999;
        expect(owner.coordinator.submit(std::move(bad)).carrier == CUDA_ERROR_INVALID_HANDLE,
               "peer/domain/context/generation ownership precedes native callback");
    }
    auto raw = owner.request(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond);
    raw.arguments.communicator.native = reinterpret_cast<ncclComm_t>(&native.comms[0]);
    expect(owner.coordinator.submit(std::move(raw)).carrier == CUDA_ERROR_INVALID_VALUE &&
               native.calls[4] == calls,
           "raw communicator input forbidden, rejection issues no requested SDK call");
    owner.release(owner.call(Kind::kDestroy, init.reserved_communicator).operation);
    native.returned = ncclInvalidArgument;
    const auto null_query = owner.call(Kind::kGetAsyncError, 0, CallerSlot::kSecond);
    expect(owner.coordinator.observe(null_query.operation).original_status == 4,
           "explicit NULL uses native ABI status");
    owner.release(null_query.operation);
    auto absent = owner.request(Kind::kInitRankConfig);
    absent.arguments.has_output = false;
    const auto no_output = owner.submit(std::move(absent));
    expect(!no_output.reserved_communicator &&
               owner.returned(no_output.operation).original_status == 4,
           "absent init output reaches original without fabricating communicator");
    owner.release(no_output.operation);
    native.returned = ncclSuccess;
    owner.shutdown();
}

void pending_continuation_reserves_every_participant() {
    Owner owner;
    const auto init = owner.call(Kind::kInitRank);
    owner.release(init.operation);
    Tails tails;
    auto request = owner.request(Kind::kAllGather, init.reserved_communicator);
    hooks(request);
    native.returned = ncclInProgress;
    const auto batch = owner.submit(std::move(request));
    tails.bind(owner.coordinator, batch.operation);
    expect(owner.returned(batch.operation).phase == Phase::kWaitingAsync,
           "batch waits for actual async progress");
    native.returned = ncclSuccess;
    native.async_state = 0;
    std::atomic<bool> hold_uid{true}, uid_entered{}, hold_query{true}, query_entered{};
    native.callback = [&] {
        if (current == reinterpret_cast<CUcontext>(&native.contexts[1])) {
            uid_entered = true;
            await([&] { return !hold_uid.load(); }, "release distinct-device original lane");
        } else if (hold_query.load()) {
            query_entered = true;
            await([&] { return !hold_query.load(); }, "release held foreign-caller query");
        }
    };
    const auto first_query = owner.submit(
        owner.request(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond));
    await([&] { return query_entered.load(); }, "first actual async query entered");
    auto uid_request = owner.request(Kind::kGetUniqueId);
    uid_request.domain = {
        71, 1, {81, reinterpret_cast<CUcontext>(&native.contexts[1]), owner.context_lifetime}};
    const auto uid = owner.submit(std::move(uid_request));
    await([&] { return uid_entered.load(); },
          "independent device occupies continuation's original lane");
    hold_query = false;
    expect(owner.returned(first_query.operation).original_status == 0,
           "first query proves async success");
    owner.release(first_query.operation);
    query_entered = false;
    hold_query = true;
    const auto second_query = owner.submit(
        owner.request(Kind::kGetAsyncError, init.reserved_communicator, CallerSlot::kSecond));
    await([&] { return query_entered.load(); }, "second caller holds participant reservation");
    const auto records = native.records.load();
    hold_uid = false;
    expect(owner.returned(uid.operation).original_status == 0, "original lane becomes available");
    owner.release(uid.operation);
    expect(owner.coordinator.advance() == CUDA_SUCCESS && native.records == records &&
               !owner.coordinator.observe(batch.operation).enqueue_complete,
           "continuation cannot overwrite active participant on another caller");
    auto blocked = owner.request(Kind::kAllReduce, init.reserved_communicator, CallerSlot::kThird);
    hooks(blocked);
    const auto collectives = native.calls[10].load();
    expect(owner.coordinator.submit(std::move(blocked)).carrier == CUDA_ERROR_NOT_READY &&
               native.calls[10] == collectives,
           "pending batch remains reserved while outgoing continuation admission is deferred");
    hold_query = false;
    expect(owner.returned(second_query.operation).original_status == 0,
           "held query completes exactly once");
    owner.release(second_query.operation);
    await(
        [&] {
            expect(owner.coordinator.advance() == CUDA_SUCCESS,
                   "retry continuation after participant release");
            return owner.coordinator.observe(batch.operation).enqueue_complete;
        },
        "outgoing continuation eventually records");
    native.callback = {};
    expect(native.records == records + 1, "deferred record runs once without reissuing collective");
    owner.release(batch.operation);
    tails.record();
    tails.finish();
    expect(owner.coordinator.advance() == CUDA_SUCCESS, "physical batch retires after actual tail");
    owner.release(owner.call(Kind::kDestroy, init.reserved_communicator).operation);
    owner.shutdown();
}

void progress_storage_is_independent() {
    CudaNcclCoordinatorLimits limits;
    limits.maximum_operations = 1;
    limits.maximum_retained_groups = 1;
    limits.maximum_progress_operations = 1;
    Owner owner(limits);
    native.returned = ncclInProgress;
    const auto direct = owner.call(Kind::kInitRankConfig);
    owner.release(direct.operation);
    native.returned = ncclSuccess;
    native.async_state = 0;
    owner.release(
        owner.call(Kind::kGetAsyncError, direct.reserved_communicator, CallerSlot::kSecond)
            .operation);
    expect(owner.coordinator.advance() == CUDA_SUCCESS,
           "independent query slot releases sole normal init record");
    owner.release(owner.call(Kind::kDestroy, direct.reserved_communicator).operation);
    owner.release(owner.call(Kind::kGroupStart).operation);
    const auto grouped = owner.call(Kind::kInitRank);
    owner.release(grouped.operation);
    expect(owner.coordinator.submit(owner.request(Kind::kGetUniqueId)).carrier ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "unfinished grouped member still consumes sole normal slot");
    native.returned = ncclInProgress;
    const auto end = owner.call(Kind::kGroupEnd);
    expect(owner.coordinator.observe(end.operation).publications.size() == 1 &&
               owner.coordinator.observe(end.operation).phase == Phase::kWaitingAsync,
           "reserved End slot remains available despite full normal pool");
    owner.release(end.operation);
    native.returned = ncclSuccess;
    owner.release(
        owner.call(Kind::kGetAsyncError, grouped.reserved_communicator, CallerSlot::kSecond)
            .operation);
    expect(owner.coordinator.advance() == CUDA_SUCCESS,
           "third independent pool progresses End7 and its member");
    owner.release(owner.call(Kind::kDestroy, grouped.reserved_communicator).operation);
    owner.release(owner.call(Kind::kGroupStart).operation);
    const auto starts = native.calls[7].load();
    expect(owner.coordinator.submit(owner.request(Kind::kGroupStart, 0, CallerSlot::kSecond))
                       .carrier == CUDA_ERROR_OUT_OF_MEMORY &&
               native.calls[7] == starts,
           "open group reserves its own End credit before any second native Start");
    owner.release(owner.call(Kind::kGroupEnd).operation);
    owner.release(owner.call(Kind::kGroupStart, 0, CallerSlot::kSecond).operation);
    owner.release(owner.call(Kind::kGroupEnd, 0, CallerSlot::kSecond).operation);
    owner.shutdown();
}

void owner_incarnation_is_not_thread_id() {
    Owner owner;
    std::thread foreign([&] {
        expect(owner.coordinator.advance() == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.coordinator.observe(1).carrier == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.coordinator.release_reply(1) == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.coordinator.close() == CUDA_ERROR_INVALID_CONTEXT &&
                   owner.coordinator.shutdown() == CUDA_ERROR_INVALID_CONTEXT,
               "unrelated CPU thread cannot mutate or observe live owner registry");
    });
    foreign.join();
    owner.shutdown();
    const auto child = ::fork();
    expect(child >= 0, "expired owner incarnation isolation");
    if (child == 0) {
        std::unique_ptr<CudaNcclCoordinator> abandoned;
        std::thread creator([&] {
            abandoned = std::make_unique<CudaNcclCoordinator>(
                dispatch(), CudaNativeExecutionLaneDispatch{set_current, get_current});
        });
        creator.join();
        std::thread replacement([&] {
            CudaNcclCoordinator replacement_owner(
                dispatch(), CudaNativeExecutionLaneDispatch{set_current, get_current});
            expect(abandoned->advance() == CUDA_ERROR_INVALID_CONTEXT &&
                       abandoned->observe(1).carrier == CUDA_ERROR_INVALID_CONTEXT &&
                       abandoned->release_reply(1) == CUDA_ERROR_INVALID_CONTEXT &&
                       abandoned->close() == CUDA_ERROR_INVALID_CONTEXT &&
                       abandoned->shutdown() == CUDA_ERROR_INVALID_CONTEXT,
                   "expired owner TLS identity rejects replacement even if thread ID is recycled");
        });
        replacement.join();
        // Retaining an owner whose incarnation expired is not clean shutdown.
        std::_Exit(0);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "expired owner regression retains unknown state only in isolated child");
}

void housekeeping_bridge_cleanup() {
    current = nullptr;
    CudaExecutionSession source({301, 303}, 17, bridge_source_dispatch());
    expect(source.initialize(0).status == CUDA_SUCCESS, "actual source session initializes");
    auto captured = source.capture_external_command(0, {});
    expect(captured.status == CUDA_SUCCESS, "actual default-stream source capsule");
    auto boundary = std::make_shared<CudaSessionExternalCommand>(captured.command);
    const auto tail = source.accept_external_barrier(0, boundary);
    expect(
        tail.status == CUDA_SUCCESS && source.begin_external_barrier(tail.barrier) == CUDA_SUCCESS,
        "actual source external tail is accepted and begun");
    Owner owner;
    const auto init = owner.call(Kind::kInitRank);
    owner.release(init.operation);
    const auto scope = owner.request(Kind::kGetUniqueId);
    auto bridge = std::make_unique<CudaExternalExecutionBridge>(
        captured.command, tail.frontier,
        CudaExternalExecutionDomain{scope.domain.context, scope.domain.device_ordinal,
                                    reinterpret_cast<CUstream>(&owner.stream),
                                    owner.context_lifetime},
        CudaExternalExecutionBridgeDispatch{get_current, create_event, record_event, destroy_event,
                                            wait_bridge_event});
    auto worker = bridge->worker();
    expect(source.prepare_external_bridge(*bridge) == CUDA_SUCCESS,
           "source prefix event precedes worker begin");
    auto request = owner.request(Kind::kAllGather, init.reserved_communicator);
    request.arguments.resources_lifetime = boundary;
    request.prepare_enqueue = [worker]() mutable { return worker.begin(); };
    request.record_enqueue = [worker]() mutable { return worker.record_enqueue(); };
    const auto collective = owner.submit(std::move(request));
    const std::array tails{tail.frontier};
    expect(owner.coordinator.bind_tails(collective.operation, tails) == CUDA_SUCCESS &&
               owner.returned(collective.operation).enqueue_complete,
           "real NCCL lane binds and records actual bridge worker endpoint");
    owner.release(collective.operation);
    expect(source.join_external_bridge(*bridge) == CUDA_SUCCESS &&
               source.record_external_enqueue(tail.barrier) == CUDA_SUCCESS,
           "actual outgoing bridge wait precedes exact source tail event");
    for (auto& event : native.events) event.ready = true;
    const auto stopped = source.poll();
    expect(stopped.status == CUDA_SUCCESS && stopped.retired == 1 &&
               tail.frontier.observe() == CUDA_SUCCESS &&
               owner.coordinator.advance() == CUDA_SUCCESS,
           "physical source tail completes before either bridge event cleanup");
    expect(source.cleanup_external_bridge(*bridge) == CUDA_SUCCESS && !bridge->completed(),
           "source event cleanup alone does not retire original-worker event");
    owner.release(owner.call(Kind::kDestroy, init.reserved_communicator).operation);
    expect(owner.coordinator.close() == CUDA_SUCCESS,
           "close still permits accepted-owner housekeeping");
    const auto cleanup = owner.coordinator.submit_housekeeping(
        scope.caller, scope.domain, boundary, [worker]() mutable { return worker.cleanup(); });
    expect(cleanup.status == CUDA_SUCCESS && cleanup.job,
           "original-lane event cleanup admitted after close");
    const auto observed = housekeeping_returned(owner, cleanup.job);
    expect(
        observed.phase == CudaNativeExecutionCallPhase::kReturned &&
            observed.status == CUDA_SUCCESS && observed.was_invoked && bridge->completed(),
        "actual bridge enforces same nonrecycled worker plus source-cleanup-before-worker-cleanup");
    expect(owner.coordinator.shutdown() == CUDA_ERROR_NOT_READY,
           "unreleased healthy housekeeping still owns its bounded record and resources");
    expect(owner.coordinator.release_housekeeping(cleanup.job) == CUDA_SUCCESS,
           "explicit housekeeping release after exact healthy callback proof");
    worker = {};
    bridge.reset();
    captured.command = {};
    boundary.reset();
    owner.shutdown();
    expect(source.shutdown().status == CUDA_SUCCESS,
           "source ownership cleans normally after bridge and tail retirement");
}

void housekeeping_bounds_and_identity() {
    CudaNcclCoordinatorLimits limits;
    limits.maximum_housekeeping_operations = 1;
    limits.maximum_caller_domains = 2;
    limits.maximum_operations = 1;
    Owner owner(limits);
    const auto scope = owner.request(Kind::kGetUniqueId);
    const auto noop = [] { return CUDA_SUCCESS; };
    expect(owner.coordinator
                   .submit_housekeeping(scope.caller, scope.domain, owner.context_lifetime, noop)
                   .status == CUDA_ERROR_INVALID_HANDLE,
           "housekeeping cannot create a previously unauthenticated native lane");
    std::thread::id original_worker;
    native.callback = [&] { original_worker = std::this_thread::get_id(); };
    const auto original = owner.call(Kind::kGetUniqueId);
    native.callback = {};
    const auto proofs = native.proofs.load();
    for (unsigned variant = 0; variant != 7; ++variant) {
        auto key = scope.caller;
        auto domain = scope.domain;
        if (variant == 0) ++key.peer;
        if (variant == 1) ++key.thread[1];
        if (variant == 2) ++domain.identity;
        if (variant == 3) ++domain.context.id;
        if (variant == 4) domain.context.native = reinterpret_cast<CUcontext>(&native.contexts[1]);
        if (variant == 5) domain.context.lifetime = std::make_shared<int>(9);
        if (variant == 6) ++domain.device_ordinal;
        expect(owner.coordinator.submit_housekeeping(key, domain, owner.context_lifetime, noop)
                       .status == CUDA_ERROR_INVALID_HANDLE,
               "housekeeping validates exact peer/incarnation/domain/context/lifetime/device");
    }
    const std::array duplicate{0, 0};
    const std::array missing{1};
    const std::array unknown{0, 1};
    expect(owner.coordinator.submit_housekeeping(scope.caller, scope.domain, {}, noop).status ==
                   CUDA_ERROR_INVALID_VALUE &&
               owner.coordinator
                       .submit_housekeeping(scope.caller, scope.domain, owner.context_lifetime, {})
                       .status == CUDA_ERROR_INVALID_VALUE &&
               owner.coordinator
                       .submit_housekeeping(scope.caller, scope.domain, owner.context_lifetime,
                                            noop, duplicate)
                       .status == CUDA_ERROR_INVALID_VALUE &&
               owner.coordinator
                       .submit_housekeeping(scope.caller, scope.domain, owner.context_lifetime,
                                            noop, missing)
                       .status == CUDA_ERROR_INVALID_VALUE &&
               owner.coordinator
                       .submit_housekeeping(scope.caller, scope.domain, owner.context_lifetime,
                                            noop, unknown)
                       .status == CUDA_ERROR_INVALID_HANDLE &&
               native.proofs == proofs,
           "invalid housekeeping has no native context or callable effects");
    std::atomic<bool> entered{}, released{};
    std::atomic<unsigned> calls{};
    auto resources = std::make_shared<int>(12);
    const std::weak_ptr<int> weak = resources;
    const auto job =
        owner.coordinator.submit_housekeeping(scope.caller, scope.domain, resources, [&] {
            expect(std::this_thread::get_id() == original_worker && group_depth == 0,
                   "housekeeping reuses actual original NCCL worker and native TLS");
            expect(
                owner.coordinator.advance() == CUDA_ERROR_INVALID_CONTEXT &&
                    owner.coordinator.release_housekeeping(1) == CUDA_ERROR_INVALID_CONTEXT &&
                    owner.coordinator.observe_housekeeping(1).status == CUDA_ERROR_INVALID_CONTEXT,
                "worker callback cannot reenter any owner housekeeping state");
            ++calls;
            entered = true;
            await([&] { return released.load(); }, "release bounded housekeeping callback");
            return CUDA_SUCCESS;
        });
    expect(job.status == CUDA_SUCCESS && job.job,
           "housekeeping quota is independent of full normal reply storage");
    resources.reset();
    await([&] { return entered.load(); }, "housekeeping entered actual stable lane");
    expect(!weak.expired() &&
               owner.coordinator.release_housekeeping(job.job) == CUDA_ERROR_NOT_READY &&
               owner.coordinator.submit(owner.request(Kind::kGetUniqueId)).carrier ==
                   CUDA_ERROR_NOT_READY,
           "held housekeeping retains resource and does not overwrite public caller reservation");
    released = true;
    expect(housekeeping_returned(owner, job.job).status == CUDA_SUCCESS && !weak.expired() &&
               calls == 1,
           "healthy return keeps lifetime until explicit release and never replays");
    expect(owner.coordinator
                   .submit_housekeeping(scope.caller, scope.domain, owner.context_lifetime, noop)
                   .status == CUDA_ERROR_OUT_OF_MEMORY,
           "unreleased housekeeping consumes only its independent bounded quota");
    owner.release(original.operation);
    owner.release(owner.call(Kind::kGroupStart).operation);
    expect(owner.coordinator.release_housekeeping(job.job) == CUDA_SUCCESS && weak.expired() &&
               owner.coordinator.release_housekeeping(job.job) == CUDA_ERROR_INVALID_HANDLE,
           "explicit release refunds once without consuming or skipping public NCCL sequence");
    const auto in_group = owner.coordinator.submit_housekeeping(
        scope.caller, scope.domain, owner.context_lifetime, [&] {
            expect(std::this_thread::get_id() == original_worker && group_depth == 1,
                   "same-lane housekeeping preserves actual open native group TLS");
            return CUDA_SUCCESS;
        });
    expect(in_group.status == CUDA_SUCCESS &&
               housekeeping_returned(owner, in_group.job).status == CUDA_SUCCESS &&
               owner.coordinator.release_housekeeping(in_group.job) == CUDA_SUCCESS,
           "housekeeping neither fabricates End nor advances public group sequence");
    owner.release(owner.call(Kind::kGroupEnd).operation);
    auto second = owner.request(Kind::kGetUniqueId);
    auto second_lifetime = std::make_shared<int>(51);
    second.domain = {
        71, 1, {81, reinterpret_cast<CUcontext>(&native.contexts[1]), second_lifetime}};
    auto second_domain = second.domain;
    const std::weak_ptr<int> second_weak = second_lifetime;
    const auto accepted_second = owner.submit(std::move(second));
    expect(owner.returned(accepted_second.operation).original_status == 0,
           "second exact domain is accepted on the same lane");
    owner.release(accepted_second.operation);
    // Use the recorded identity for both devices; unknown device sets were
    // rejected above rather than delegated to a new unowned execution domain.
    const std::array devices{1, 0};
    const auto multi = owner.coordinator.submit_housekeeping(scope.caller, second_domain,
                                                             second_lifetime, noop, devices);
    expect(multi.status == CUDA_SUCCESS &&
               housekeeping_returned(owner, multi.job).status == CUDA_SUCCESS &&
               owner.coordinator.release_housekeeping(multi.job) == CUDA_SUCCESS,
           "validated multi-device housekeeping shares the original hub gates");
    second_domain.context.lifetime.reset();
    second_lifetime.reset();
    expect(second_weak.expired(), "domain history retains no permanent context ownership");
    second_domain.context.lifetime = std::make_shared<int>(51);
    expect(owner.coordinator
                   .submit_housekeeping(scope.caller, second_domain, second_domain.context.lifetime,
                                        noop)
                   .status == CUDA_ERROR_INVALID_HANDLE,
           "expired domain identity cannot be reauthenticated using reused raw context values");
    auto rebound = owner.request(Kind::kGetUniqueId);
    rebound.domain = second_domain;
    expect(owner.coordinator.submit(std::move(rebound)).carrier == CUDA_ERROR_INVALID_HANDLE,
           "public admission cannot silently rebind an accepted domain identity");
    auto third = owner.request(Kind::kGetUniqueId);
    third.domain.identity = 72;
    expect(owner.coordinator.submit(std::move(third)).carrier == CUDA_ERROR_OUT_OF_MEMORY,
           "bounded weak domain history is preflighted before public native acceptance");
    owner.shutdown();
}

void housekeeping_uncertainty_and_fork() {
    for (unsigned scenario = 0; scenario != 3; ++scenario) {
        const auto child = ::fork();
        expect(child >= 0, "isolated housekeeping uncertainty");
        if (child == 0) {
            Owner owner;
            owner.release(owner.call(Kind::kGetUniqueId).operation);
            const auto scope = owner.request(Kind::kGetUniqueId);
            auto resources = std::make_shared<int>(7);
            const std::weak_ptr<int> weak = resources;
            const auto job = owner.coordinator.submit_housekeeping(
                scope.caller, scope.domain, resources, [scenario]() -> CUresult {
                    if (scenario == 0) return CUDA_ERROR_LAUNCH_FAILED;
                    if (scenario == 1) {
                        native.loses_context = true;
                        return CUDA_SUCCESS;
                    }
                    throw 7;
                });
            expect(job.status == CUDA_SUCCESS, "uncertain callback admitted once");
            resources.reset();
            const auto observed = housekeeping_returned(owner, job.job);
            expect(observed.phase == CudaNativeExecutionCallPhase::kUncertain &&
                       observed.was_invoked && observed.status != CUDA_SUCCESS && !weak.expired() &&
                       owner.coordinator.release_housekeeping(job.job) != CUDA_SUCCESS &&
                       owner.coordinator.close() == CUDA_SUCCESS &&
                       owner.coordinator.shutdown() == CUDA_ERROR_NOT_READY,
                   "callback/proof/exception uncertainty never releases or invents cleanup");
            std::_Exit(0);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "unknown housekeeping owner retained only until isolated process exit");
    }
    Owner owner;
    owner.release(owner.call(Kind::kGetUniqueId).operation);
    const auto scope = owner.request(Kind::kGetUniqueId);
    const auto child = ::fork();
    expect(child >= 0, "housekeeping fork boundary");
    if (child == 0) {
        std::_Exit(owner.coordinator
                                   .submit_housekeeping(scope.caller, scope.domain,
                                                        owner.context_lifetime,
                                                        [] { return CUDA_SUCCESS; })
                                   .status == CUDA_ERROR_INVALID_CONTEXT &&
                           owner.coordinator.observe_housekeeping(1).status ==
                               CUDA_ERROR_INVALID_CONTEXT &&
                           owner.coordinator.release_housekeeping(1) == CUDA_ERROR_INVALID_CONTEXT
                       ? 0
                       : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork rejects housekeeping before inherited hub synchronization or native callbacks");
    owner.shutdown();
}

void isolated_uncertainties() {
    for (unsigned scenario = 0; scenario != 5; ++scenario) {
        const auto child = ::fork();
        expect(child >= 0, "isolated native uncertainty fork");
        if (child == 0) {
            Owner owner;
            if (scenario == 0) {
                owner.release(owner.call(Kind::kGroupStart).operation);
                expect(owner.coordinator.close() == CUDA_SUCCESS &&
                           owner.coordinator.shutdown() == CUDA_ERROR_NOT_READY,
                       "idle open native TLS group cannot stop hub or invent End");
            } else if (scenario == 1) {
                native.returned = ncclRemoteError;
                native.return_null = true;
                const auto failed = owner.call(Kind::kInitRankConfig);
                expect(owner.coordinator.observe(failed.operation).phase == Phase::kUncertain,
                       "error plus NULL seed is not proof of no native owner");
                owner.release(failed.operation);
            } else if (scenario == 2) {
                const auto init = owner.call(Kind::kInitRank);
                owner.release(init.operation);
                owner.release(owner.call(Kind::kGroupStart).operation);
                owner.release(owner.call(Kind::kDestroy, init.reserved_communicator).operation);
                native.returned = ncclInProgress;
                const auto ended = owner.call(Kind::kGroupEnd);
                expect(owner.coordinator.observe(ended.operation).phase == Phase::kUncertain,
                       "destructive End7 has no legal post-poll completion channel");
                const auto queries = native.calls[4].load();
                expect(owner.coordinator
                                   .submit(owner.request(Kind::kGetAsyncError,
                                                         init.reserved_communicator))
                                   .carrier == CUDA_ERROR_INVALID_HANDLE &&
                           native.calls[4] == queries,
                       "never query possibly freed grouped owner");
            } else if (scenario == 3) {
                native.callback = [] { native.loses_context = true; };
                const auto accepted = owner.submit(owner.request(Kind::kGetUniqueId));
                const auto observed = owner.returned(accepted.operation);
                expect(
                    observed.phase == Phase::kUncertain && observed.original_status == 0 &&
                        observed.carrier == CUDA_ERROR_INVALID_CONTEXT,
                    "post-call context loss preserves actual original status but retains owners");
            } else {
                const auto init = owner.call(Kind::kInitRank);
                owner.release(init.operation);
                native.returned = ncclInProgress;
                const auto destructive = owner.call(Kind::kAbort, init.reserved_communicator);
                expect(owner.coordinator.observe(destructive.operation).phase == Phase::kUncertain,
                       "direct destructive7 seals owner and retains without post-query");
            }
            expect(owner.coordinator.close() == CUDA_SUCCESS &&
                       owner.coordinator.shutdown() == CUDA_ERROR_NOT_READY,
                   "unknown native ownership cannot report clean shutdown");
            std::_Exit(0);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated uncertainty retains all state until process exit");
    }
    Owner owner;
    const auto child = ::fork();
    expect(child >= 0, "inherited coordinator guard fork");
    if (child == 0) {
        std::_Exit(owner.coordinator.advance() == CUDA_ERROR_INVALID_CONTEXT &&
                           owner.coordinator.observe(1).carrier == CUDA_ERROR_INVALID_CONTEXT &&
                           owner.coordinator.close() == CUDA_ERROR_INVALID_CONTEXT
                       ? 0
                       : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "PID rejection before inherited TLS and hub synchronization");
    owner.shutdown();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view{argv[1]} == "--eager-grouped-prefix") {
        eager_grouped_incoming_prefixes();
        incoming_failure_retains_group();
        std::cout << "NCCL coordinator: per-inner incoming prefixes and deferred outgoing tail; "
                     "no GPU or multi-rank qualification\n";
        return 0;
    }
    expect(argc == 1, "only the explicit focused eager-grouped-prefix mode is supported");
    eleven_routes_and_tails();
    grouped_publications_and_repeated_collectives();
    eager_grouped_incoming_prefixes();
    incoming_failure_retains_group();
    async_progress_and_native_errors();
    authentication_limits_and_null();
    pending_continuation_reserves_every_participant();
    progress_storage_is_independent();
    owner_incarnation_is_not_thread_id();
    housekeeping_bridge_cleanup();
    housekeeping_bounds_and_identity();
    housekeeping_uncertainty_and_fork();
    isolated_uncertainties();
    for (unsigned kind = 1; kind <= 11; ++kind)
        expect(native.calls[kind] != 0, "all eleven real injected ABIs executed");
    std::cout << "NCCL coordinator: thirteen ownership groups; eleven native ABI routes; "
                 "no GPU or multi-rank qualification\n";
}
