#include "nccl_actor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::backend::cuda;
namespace wire = glimmer::control;
void expect(bool value, std::string_view message) {
    if (value) return;
    std::cerr << "NCCL actor: " << message << '\n';
    std::_Exit(1);
}
template <typename Predicate>
void await(Predicate predicate) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!predicate()) {
        expect(std::chrono::steady_clock::now() < limit, "bounded CPU progress");
        std::this_thread::yield();
    }
}
struct AuditTrace {
    static constexpr std::size_t capacity = 16384;
    std::unique_ptr<NcclActorAuditRecord[]> records{
        std::make_unique<NcclActorAuditRecord[]>(capacity)};
    std::size_t size{};
    NcclActor* actor{};
    bool overflow{}, wrong_thread{}, reentry_failed{};
    const std::thread::id owner{std::this_thread::get_id()};
    static void receive(void* opaque, const NcclActorAuditRecord& record) noexcept {
        auto& trace = *static_cast<AuditTrace*>(opaque);
        trace.wrong_thread |= std::this_thread::get_id() != trace.owner;
        if (trace.size == capacity)
            trace.overflow = true;
        else
            trace.records[trace.size++] = record;
        // Reentrant inspection/configuration must stop before observation or
        // native work. The callback copies fixed storage, never owning actors.
        trace.reentry_failed |= !trace.actor ||
                                trace.actor->snapshot().status != CUDA_ERROR_NOT_READY ||
                                trace.actor->set_observer({});
    }
    NcclActorObserver observer() {
        return {this, receive};
    }
    template <typename Predicate>
    bool contains(Predicate predicate) const {
        for (std::size_t index = 0; index != size; ++index)
            if (predicate(records[index])) return true;
        return false;
    }
    void validate(bool stopped = true) const {
        static_assert(std::is_trivially_copyable_v<NcclActorAuditRecord>);
        static_assert(
            [] {
                std::uint64_t next = UINT64_MAX;
                return glimmer::backend::cuda::detail::take_diagnostic_id(next) == UINT64_MAX &&
                       !glimmer::backend::cuda::detail::take_diagnostic_id(next) && !next;
            }(),
            "the production serial allocator exhausts without wrapping");
        expect(size && !overflow && !wrong_thread && !reentry_failed,
               "bounded owner-only nonreentrant fixed-record observer");
        expect(records[0].event == NcclActorAuditEvent::kBegin &&
                   records[0].subject == NcclActorAuditSubject::kActor,
               "cold observer has an explicit start record");
        std::vector<bool> matched(size);
        std::vector<NcclActorAuditRecord> submissions;
        std::uint32_t contract{};
        for (std::size_t index = 0; index != size; ++index) {
            const auto& row = records[index];
            expect(row.actor_serial == index + 1, "actor serial is contiguous and never zero");
            const bool has_communicator = row.communicator_object != 0;
            expect((row.communicator_domain != 0) == has_communicator &&
                       (row.communicator_context != 0) == has_communicator,
                   "communicator identity is one complete symbolic tuple or all zero");
            if (row.subject == NcclActorAuditSubject::kRequest) {
                if (!contract) contract = row.output_contract;
                expect((contract == 1 || contract == 2) && row.output_contract == contract,
                       "every native request preserves the trusted cold output contract");
                if (row.event == NcclActorAuditEvent::kNativeSubmit) {
                    expect(std::ranges::none_of(
                               submissions, [&](const auto& item) { return item.id == row.id; }),
                           "one actual native acceptance per actor request");
                    submissions.push_back(row);
                }
                const auto accepted =
                    std::ranges::find(submissions, row.id, &NcclActorAuditRecord::id);
                if (accepted == submissions.end())
                    expect(!has_communicator,
                           "input claims and cold preparation cannot publish native identity");
                else
                    expect(row.communicator_domain == accepted->communicator_domain &&
                               row.communicator_context == accepted->communicator_context &&
                               row.communicator_object == accepted->communicator_object,
                           "accepted communicator tuple survives every later row and erasure");
                if (row.native_kind == 1 || row.native_kind == 7 || row.native_kind == 8)
                    expect(!has_communicator, "noncommunicator operations never invent an owner");
            } else
                expect(!row.output_contract && !has_communicator,
                       "ordinary and owner records cannot inherit native output/resource tags");
            if (!row.has_context)
                expect(!row.before_kind && !row.before_identity && !row.before_context &&
                           !row.before_device && !row.after_kind && !row.after_identity &&
                           !row.after_context && !row.after_device,
                       "absent actual context observation keeps every transition field zero");
            if (row.event == NcclActorAuditEvent::kAdmission && row.accepted)
                expect(row.id && row.sequence_consumed && row.status == CUDA_SUCCESS &&
                           row.reservation && row.generation && row.cutoff,
                       "accepted admission binds actual actor identity and batch");
            if (row.event == NcclActorAuditEvent::kNativeSubmit)
                expect(row.accepted && row.native_sequence && row.native_operation &&
                           row.native_kind >= 1 && row.native_kind <= 11,
                       "native submit names an actual accepted coordinator operation");
            if (row.event == NcclActorAuditEvent::kGrantEnd) {
                bool found{};
                for (auto prior = index; prior != 0;) {
                    --prior;
                    const auto& begin = records[prior];
                    if (matched[prior] || begin.event != NcclActorAuditEvent::kGrantBegin ||
                        begin.subject != row.subject || begin.id != row.id ||
                        begin.attempt != row.attempt ||
                        begin.input_reservation != row.input_reservation ||
                        begin.input_generation != row.input_generation ||
                        begin.input_cutoff != row.input_cutoff)
                        continue;
                    matched[prior] = found = true;
                    break;
                }
                expect(found && row.input_reservation && row.input_generation && row.input_cutoff,
                       "each completed attempt retains its exact independently supplied grant");
            }
        }
        for (std::size_t index = 0; index != size; ++index)
            if (records[index].event == NcclActorAuditEvent::kGrantBegin)
                expect(matched[index], "healthy fixture has no fabricated or unmatched grant end");
        if (stopped)
            expect(records[size - 1].event == NcclActorAuditEvent::kStopped &&
                       records[size - 1].retired,
                   "explicit healthy shutdown, not host return, closes the actor trace");
    }
};
AuditTrace* active_audit{};
struct Event {
    bool live{}, recorded{};
    CUstream stream{};
};
struct Fill32Command {
    CUdeviceptr destination{};
    unsigned value{};
    std::size_t count{};
    CUstream stream{};
};
struct Native {
    std::array<int, 4> contexts{};
    std::array<int, 16> streams{}, comms{};
    std::array<Event, 128> events{};
    std::array<std::array<float, 16>, 8> memory{};
    unsigned next_context{}, next_stream{}, next_memory{}, next_comm{};
    unsigned records{}, destroys{}, primary_retains{}, primary_releases{};
    unsigned uploads{}, fills{};
    std::array<std::atomic<unsigned>, 11> calls{};
    std::atomic<bool> hold{}, entered{}, release{true}, stop{true};
    std::atomic<int> result{}, async_result{};
    bool clear_current{};
    bool upload_loses_context{};
    std::atomic<bool> fail_worker_restore{}, fail_worker_observation{};
    CUresult upload_status{CUDA_SUCCESS};
    CUcontext primary() {
        return reinterpret_cast<CUcontext>(&contexts[3]);
    }
} native;
thread_local CUcontext current{};
thread_local unsigned depth{};
thread_local bool reject_restore{}, reject_observation{};
std::thread::id owner;
CUresult CUDAAPI set_current(CUcontext value) {
    if (reject_restore) {
        reject_restore = false;
        return CUDA_ERROR_INVALID_CONTEXT;
    }
    current = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* value) {
    if (reject_observation) {
        reject_observation = false;
        return CUDA_ERROR_UNKNOWN;
    }
    *value = current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI fill32_async(CUdeviceptr destination, unsigned value, std::size_t count,
                              CUstream stream) {
    const Fill32Command command{destination, value, count, stream};
    expect(command.stream && command.count == 8, "exact original ordinary fill geometry");
    ++native.fills;
    // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Fake Driver allocation.
    auto* output = reinterpret_cast<std::byte*>(command.destination);
    for (std::size_t index = 0; index != command.count; ++index)
        std::memcpy(output + index * sizeof(command.value), &command.value, sizeof(command.value));
    return CUDA_SUCCESS;
}
ncclResult_t visit(unsigned operation) {
    expect(std::this_thread::get_id() != owner, "real distinct native lane");
    ++native.calls[operation - 1];
    native.entered = true;
    if (native.hold) await([] { return native.release.load(); });
    if (native.clear_current) current = nullptr;
    reject_restore = native.fail_worker_restore.load();
    reject_observation = native.fail_worker_observation.load();
    return static_cast<ncclResult_t>(native.result.load());
}
ncclResult_t uid(ncclUniqueId* output) {
    if (output) std::memset(output, 0x6d, sizeof(*output));
    return visit(1);
}
ncclResult_t init(ncclComm_t* output, int, ncclUniqueId, int) {
    if (output) *output = reinterpret_cast<ncclComm_t>(&native.comms.at(native.next_comm++));
    current = native.primary();
    return visit(2);
}
ncclResult_t init_config(ncclComm_t* output, int, ncclUniqueId, int, ncclConfig_t* config) {
    expect(!config, "exact NULL config, not fabricated defaults");
    if (output) *output = reinterpret_cast<ncclComm_t>(&native.comms.at(native.next_comm++));
    current = native.primary();
    return visit(3);
}
ncclResult_t query(ncclComm_t, ncclResult_t* output) {
    const auto raw = native.async_result.load();
    if (output) std::memcpy(output, &raw, sizeof(raw));
    return visit(4);
}
ncclResult_t destroy(ncclComm_t) {
    return visit(5);
}
ncclResult_t abort_comm(ncclComm_t) {
    return visit(6);
}
ncclResult_t start() {
    ++depth;
    return visit(7);
}
ncclResult_t end() {
    expect(depth != 0, "actual lane TLS group depth");
    --depth;
    return visit(8);
}
ncclResult_t gather(const void* input, void* output, std::size_t count, ncclDataType_t type,
                    ncclComm_t, cudaStream_t stream) {
    expect(stream && type == ncclFloat32, "real exact source stream/type");
    std::memcpy(output, input, count * sizeof(float));
    return visit(9);
}
ncclResult_t reduce(const void* input, void* output, std::size_t count, ncclDataType_t type,
                    ncclRedOp_t op, ncclComm_t, cudaStream_t stream) {
    expect(stream && type == ncclFloat32 && op == ncclSum, "actual reduction arguments");
    std::memcpy(output, input, count * sizeof(float));
    return visit(10);
}
ncclResult_t bcast(void* output, std::size_t count, ncclDataType_t type, int root, ncclComm_t,
                   cudaStream_t stream) {
    expect(output && count == 8 && type == ncclFloat32 && root == 0 && stream,
           "actual broadcast arguments");
    return visit(11);
}
CudaNcclNativeDispatch calls() {
    return {uid, init, init_config, query, destroy, abort_comm, start, end, gather, reduce, bcast};
}
CUresult CUDAAPI stream_create(CUstream* output, unsigned) {
    *output = reinterpret_cast<CUstream>(&native.streams.at(native.next_stream++));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_destroy(CUstream) {
    return CUDA_SUCCESS;
}
CudaNcclPrimaryDomainDispatch primary_dispatch() {
    CudaNcclPrimaryDomainDispatch sdk;
    sdk.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.primary_retain = [](CUcontext* output, CUdevice) {
        ++native.primary_retains;
        *output = native.primary();
        return CUDA_SUCCESS;
    };
    sdk.primary_release = [](CUdevice) {
        ++native.primary_releases;
        return CUDA_SUCCESS;
    };
    sdk.context_get_current = get_current;
    sdk.context_set_current = set_current;
    sdk.stream_create = stream_create;
    sdk.stream_query = [](CUstream) { return CUDA_SUCCESS; };
    sdk.stream_destroy = stream_destroy;
    return sdk;
}
CudaSessionDispatch session_dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned, CUdevice) {
        expect(native.next_context < 3, "bounded source contexts");
        *output = current = reinterpret_cast<CUcontext>(&native.contexts[native.next_context++]);
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = set_current;
    sdk.context_destroy = [](CUcontext context) {
        expect(current == context, "exact source cleanup context");
        current = nullptr;
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        expect(bytes == 64, "bounded complete allocation");
        *output = reinterpret_cast<CUdeviceptr>(native.memory.at(native.next_memory++).data());
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr) { return CUDA_SUCCESS; };
    sdk.upload = [](CUdeviceptr destination, const void* source, std::size_t bytes) {
        ++native.uploads;
        if (native.upload_status != CUDA_SUCCESS) return native.upload_status;
        // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Fake Driver allocation.
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        if (native.upload_loses_context) current = nullptr;
        return CUDA_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- Exact Driver ABI.
    sdk.download = [](void* destination, CUdeviceptr source, std::size_t bytes) {
        // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Fake Driver allocation.
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        return CUDA_SUCCESS;
    };
    sdk.stream_create = stream_create;
    sdk.stream_destroy = stream_destroy;
    sdk.execution.context_get_current = get_current;
    sdk.execution.launch_kernel = [](CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, unsigned, CUstream, void**,
                                     void**) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.execution.event_create = [](CUevent* output, unsigned) {
        const auto found = std::ranges::find(native.events, false, &Event::live);
        expect(found != native.events.end(), "bounded independent event storage");
        *found = {true, false, nullptr};
        *output = reinterpret_cast<CUevent>(&*found);
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent value, CUstream stream) {
        auto& event = *reinterpret_cast<Event*>(value);
        expect(event.live && !event.recorded, "fresh exact posted-prefix event");
        event.recorded = true;
        event.stream = stream;
        ++native.records;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent value) {
        expect(reinterpret_cast<Event*>(value)->recorded, "no query of unrecorded event");
        return native.stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent value) {
        auto& event = *reinterpret_cast<Event*>(value);
        expect(event.live, "event cleanup once");
        event.live = false;
        ++native.destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        return native.stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.stream_wait_event = [](CUstream, CUevent value, unsigned flags) {
        expect(reinterpret_cast<Event*>(value)->recorded && !flags, "wait for actual prefix");
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_wait_event = sdk.stream_wait_event;
    sdk.execution.fill32_async = fill32_async;
    return sdk;
}
// This independently injected fake explicitly writes every present output in
// the callbacks above. Its oracle does not use the packet's changed-byte mask.
CUresult effects(void*, const NcclActorEffectInput& input,
                 wire::ExecutionNcclObservation& result) noexcept {
    if (!input.native) return CUDA_ERROR_NOT_READY;
    if (std::holds_alternative<wire::ExecutionNcclGetUniqueId>(input.request.call)) {
        result.effects = 1;
        result.unique_id_mask.fill(std::byte{0xff});
        result.unique_id.fill(std::byte{0x6d});
    } else if (std::holds_alternative<wire::ExecutionNcclGetAsyncError>(input.request.call)) {
        result.effects = 4;
        result.async_error_mask = 15;
        result.async_error = input.native->async_error;
    } else {
        result.effects = 2;
        if (input.native->communicator) result.communicator = input.reserved_communicator;
    }
    return CUDA_SUCCESS;
}
struct Harness {
    std::vector<std::shared_ptr<CudaExecutionSession>> sources;
    std::vector<std::uint64_t> allocations;
    std::unique_ptr<NcclActor> actor;
    std::uint64_t sequence{}, group{};
    CUresult source_fence{CUDA_SUCCESS};
    bool source_visible{true};
    CUresult cleanup_status{CUDA_ERROR_NOT_READY};
    unsigned cleanup_proofs{};
    std::uint64_t cleanup_connection{};
    std::shared_ptr<const void> cleanup_helper;
    bool cleanup_reentry{};
    explicit Harness(bool producer = true, unsigned source_count = 1, bool stable_output = false,
                     NcclActorConfig config = {}, bool retirement_proof = false) {
        owner = std::this_thread::get_id();
        for (unsigned i = 0; i != source_count; ++i) {
            auto source = std::make_shared<CudaExecutionSession>(
                glimmer::backend::ExecutionSessionIdentity{101 + i, 201 + i}, 301 + i,
                session_dispatch(), CudaSessionLimits{}, CudaSessionPersistentConfig{}, nullptr,
                nullptr, false, CudaSessionStreamConfig{true});
            expect(source->initialize(0).status == CUDA_SUCCESS, "source initialized");
            const auto allocation = source->allocate(64);
            expect(allocation.status == CUDA_SUCCESS, "source allocation");
            allocations.push_back(allocation.id);
            // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Fake Driver allocation.
            auto* values = reinterpret_cast<float*>(allocation.address);
            for (unsigned n = 0; n != 16; ++n) values[n] = static_cast<float>(n + 1 + i * 100);
            expect(source->deactivate().status == CUDA_SUCCESS, "source leaves NULL owner entry");
            sources.push_back(std::move(source));
        }
        NcclActorResolver resolver;
        resolver.session = [&](std::uint64_t peer, std::uint64_t session) {
            return source_visible && peer == 17 && session >= 1 && session <= sources.size()
                       ? sources[session - 1]
                       : std::shared_ptr<CudaExecutionSession>{};
        };
        resolver.stream = [&](std::uint64_t peer, const wire::ExecutionNcclScope& scope,
                              const wire::ExecutionNcclStream& stream) {
            const auto id = stream.kind == wire::ExecutionNcclStreamKind::kOwned
                                ? stream.owner
                                : (scope.context ? scope.context : 1);
            if (peer != 17 || !id || id > sources.size()) return NcclActorSource{};
            return NcclActorSource{id, sources[id - 1], stream.object};
        };
        resolver.admit_source = [&](std::uint64_t peer, std::uint64_t source) {
            expect(peer == 17 && source && source <= sources.size(),
                   "source fence uses authenticated existing owner identity");
            return source_fence;
        };
        if (retirement_proof)
            resolver.cleanup_owner_retired = [&](std::uint64_t peer, std::uint64_t connection,
                                                 const std::shared_ptr<const void>& helper) {
                expect(peer == 17 && connection == cleanup_connection && helper == cleanup_helper,
                       "retirement proof authenticates the exact trusted helper identity");
                ++cleanup_proofs;
                if (cleanup_reentry)
                    expect(actor->snapshot().status == CUDA_ERROR_NOT_READY &&
                               actor->complete_cleanup_retirement({}) == CUDA_ERROR_NOT_READY,
                           "trusted proof callback cannot reenter actor state");
                return cleanup_status;
            };
        NcclActorEffectProducer output;
        if (producer) output = {nullptr, effects, std::make_shared<int>(0)};
        if (stable_output) output.output_contract = CudaNcclNativeOutputContract::kPinnedHostStable;
        actor = std::make_unique<NcclActor>(
            config, calls(), CudaNativeExecutionLaneDispatch{set_current, get_current},
            primary_dispatch(), std::move(resolver), std::move(output));
        if (active_audit && !active_audit->actor) {
            active_audit->actor = actor.get();
            expect(actor->set_observer(active_audit->observer()), "cold optional audit installed");
        }
    }
    wire::ExecutionNcclRequest request(wire::ExecutionNcclCall call, std::uint64_t context = 0) {
        return {{{91, 92}, ++sequence, group, context}, std::move(call)};
    }
    NcclActorGrant grant(std::uint64_t id) {
        const auto snapshot = actor->snapshot();
        expect(snapshot.status == CUDA_SUCCESS, "honest bounded snapshot");
        for (const auto& batch : snapshot.reservations)
            for (const auto& item : batch.requests)
                if (item.request == id) return item.grant;
        expect(false, "exact retained request in snapshot");
        return {};
    }
    void poll() {
        for (const auto& source : sources) {
            expect(source->activate().status == CUDA_SUCCESS, "poll exact source");
            const auto polled = source->poll().status;
            expect(polled == CUDA_SUCCESS || polled == CUDA_ERROR_NOT_READY,
                   "physical poll respects active native host exclusion");
            expect(source->deactivate().status == CUDA_SUCCESS, "poll restores owner NULL");
        }
    }
    void step(bool allow_issue = true) {
        expect(actor->observe() == CUDA_SUCCESS, "CPU observation succeeds");
        const auto snapshot = actor->snapshot();
        expect(snapshot.status == CUDA_SUCCESS, "snapshot succeeds");
        if (allow_issue)
            for (const auto& batch : snapshot.reservations)
                for (const auto& item : batch.requests)
                    if (item.is_ready) {
                        const auto result = actor->issue(item.request, item.grant);
                        expect(result == CUDA_SUCCESS || result == CUDA_ERROR_NOT_READY,
                               "exact granted stage");
                    }
        poll();
        expect(actor->observe() == CUDA_SUCCESS, "consume actual source tails");
    }
    std::pair<std::uint64_t, NcclActorReply> run(wire::ExecutionNcclCall call,
                                                 std::uint64_t context = 0,
                                                 std::uint64_t primary = 0) {
        const auto prepared = actor->prepare({17, 0}, request(std::move(call), context), primary);
        expect(prepared.status == CUDA_SUCCESS && prepared.request, "clean preparation");
        await([&] {
            step();
            return actor->reply(prepared.request).ready;
        });
        return {prepared.request, actor->reply(prepared.request)};
    }
    void release(std::uint64_t id) {
        expect(actor->release_reply(id) == CUDA_SUCCESS, "public ACK is not stop proof");
    }
    wire::ExecutionNcclCommunicator init_comm(bool configured = false) {
        wire::ExecutionNcclInitRank input{true, 1, 0, {}};
        auto [id, result] =
            configured ? run(wire::ExecutionNcclInitRankConfig{input, {}}) : run(input);
        expect(result.envelope->original_status == 0 && result.envelope->observation.effects == 2,
               "actual original init/status/write");
        expect(result.envelope->context_effect.kind ==
                   wire::ExecutionNcclContextEffectKind::kSelectPrimary,
               "actual observed primary effect, not opcode guess");
        const auto comm = result.envelope->observation.communicator;
        expect(comm.domain == 1 && comm.object, "symbolic owned communicator");
        release(id);
        return comm;
    }
    wire::ExecutionNcclAllGather collective(wire::ExecutionNcclCommunicator comm,
                                            unsigned source = 1) {
        return {comm,
                {},
                {wire::ExecutionNcclBufferKind::kOwned, source, allocations[source - 1], 0},
                {wire::ExecutionNcclBufferKind::kOwned, source, allocations[source - 1], 32},
                8,
                7};
    }
    void finish() {
        await([&] {
            step();
            return actor->snapshot().reservations.empty();
        });
        expect(actor->close() == CUDA_SUCCESS, "close actor admission");
        await([&] { return actor->shutdown() == CUDA_SUCCESS; });
        for (std::size_t i = 0; i != sources.size(); ++i) {
            expect(sources[i]->activate().status == CUDA_SUCCESS, "source cleanup entry");
            expect(sources[i]->release(allocations[i]).status == CUDA_SUCCESS,
                   "capsules released after proof");
            expect(sources[i]->shutdown().status == CUDA_SUCCESS, "source shutdown");
        }
        expect(native.primary_retains == native.primary_releases,
               "actual primary cleanup balanced");
    }
};

wire::ExecutionSourceContinuation ordinary_scope(std::uint64_t sequence) {
    return {0, {91, 92}, 77, sequence, 1, 0};
}
NcclActorContinuationTarget ordinary_target(wire::ExecutionOpcode operation,
                                            std::uint64_t sequence) {
    NcclActorContinuationTarget result;
    result.connection = 1;
    result.wire_sequence = sequence;
    result.operation = operation;
    result.request_lifetime = std::make_shared<int>(17);
    return result;
}
NcclActorContinuationSnapshot ordinary_snapshot(Harness& test, std::uint64_t id) {
    const auto state = test.actor->snapshot();
    expect(state.status == CUDA_SUCCESS, "ordinary snapshot is CPU-only and authenticated");
    for (const auto& batch : state.reservations)
        for (const auto& item : batch.continuations)
            if (item.continuation == id) return item;
    expect(false, "exact ordinary continuation remains in its original batch");
    return {};
}
NcclActorGrant open_ordinary_group(Harness& test, std::uint64_t group = 77) {
    test.group = group;
    auto [id, reply] = test.run(wire::ExecutionNcclGroupStart{});
    expect(reply.envelope->original_status == 0, "actual native Start before ordinary work");
    const auto grant = test.grant(id);
    test.release(id);
    return grant;
}
void close_ordinary_group(Harness& test) {
    auto [id, reply] = test.run(wire::ExecutionNcclGroupEnd{});
    expect(reply.envelope->original_status == 0, "actual exact native End");
    test.group = 0;
    test.release(id);
}
NcclActorContinuationOutcome ordinary_reply(std::uint64_t id = 0) {
    NcclActorContinuationOutcome result;
    result.kind = NcclActorContinuationOutcomeKind::kReturned;
    result.status = CUDA_SUCCESS;
    wire::ExecutionWireReply reply;
    reply.disposition = wire::ExecutionWireDisposition::kComplete;
    reply.id = id;
    result.reply = std::move(reply);
    return result;
}
NcclActorContinuationPermit ordinary_begin(Harness& test, std::uint64_t id) {
    const auto admitted = test.actor->begin_continuation(id, ordinary_snapshot(test, id).grant);
    expect(admitted.status == CUDA_SUCCESS && admitted.permit.continuation == id &&
               admitted.permit.attempt,
           "fresh exact ordinary attempt capability");
    return admitted.permit;
}
void ordinary_return(Harness& test, std::uint64_t id, NcclActorContinuationOutcome outcome) {
    const auto permit = ordinary_begin(test, id);
    expect(test.actor->finish_continuation_attempt(permit, std::move(outcome)) == CUDA_SUCCESS,
           "actual ordinary return consumes exactly its one-use attempt");
    expect(test.actor->release_continuation_reply(id) == CUDA_SUCCESS,
           "release ordinary reply independently from its native tail");
}

void all_routes() {
    Harness test;
    auto [uid_id, unique] = test.run(wire::ExecutionNcclGetUniqueId{true});
    expect(unique.envelope->observation.unique_id[127] == std::byte{0x6d}, "complete UID bytes");
    test.release(uid_id);
    const auto first = test.init_comm();
    const auto second = test.init_comm(true);
    auto [query_id, query_result] = test.run(wire::ExecutionNcclGetAsyncError{first, true}, 1, 1);
    expect(query_result.envelope->observation.async_error == 0, "exact async original output");
    test.release(query_id);
    auto input = test.collective(first);
    auto [gather_id, gathered] = test.run(input, 1);
    expect(gathered.envelope->original_status == 0, "actual gather original");
    test.release(gather_id);
    auto [reduce_id, reduced] = test.run(wire::ExecutionNcclAllReduce{input, 0}, 1);
    expect(reduced.envelope->original_status == 0, "actual reduce original");
    test.release(reduce_id);
    auto [bcast_id, broadcast] =
        test.run(wire::ExecutionNcclBcast{first, {}, input.receive, 8, 7, 0}, 1);
    expect(broadcast.envelope->original_status == 0, "actual bcast original");
    test.release(bcast_id);
    for (unsigned i = 0; i != 8; ++i)
        expect(native.memory[0][i + 8] == static_cast<float>(i + 1), "all fake numerical outputs");
    test.group = 44;
    auto [start_id, started] = test.run(wire::ExecutionNcclGroupStart{});
    test.release(start_id);
    auto [end_id, ended] = test.run(wire::ExecutionNcclGroupEnd{});
    test.group = 0;
    test.release(end_id);
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{first});
    test.release(destroy_id);
    auto [abort_id, aborted] = test.run(wire::ExecutionNcclAbort{second});
    test.release(abort_id);
    test.finish();
    for (const auto& count : native.calls)
        expect(count.load() != 0, "every one of eleven native callbacks actually invoked");
}
void authorization() {
    Harness test;
    const auto before = native.primary_retains;
    const auto prepared =
        test.actor->prepare({17, 0}, test.request(wire::ExecutionNcclGetUniqueId{true}));
    expect(prepared.status == CUDA_SUCCESS && native.primary_retains == before,
           "prepare never initializes SDK");
    const auto grant = test.grant(prepared.request);
    for (unsigned i = 0; i != 8; ++i) test.step(false);
    expect(native.calls[0] == 0, "CPU observation never grants native issue");
    auto wrong = grant;
    ++wrong.generation;
    expect(test.actor->issue(prepared.request, wrong) == CUDA_ERROR_INVALID_VALUE,
           "stale generation rejects");
    const auto stale_progress = test.actor->progress_source(prepared.request, wrong, 1);
    expect(stale_progress.status == CUDA_ERROR_INVALID_VALUE && !stale_progress.source_session &&
               !stale_progress.issue.sequence,
           "source progress cannot bypass exact batch generation");
    const auto stale_inspection = test.actor->inspect_source(prepared.request, wrong, 1);
    expect(stale_inspection.status == CUDA_ERROR_INVALID_VALUE &&
               !stale_inspection.source_session && !stale_inspection.dependency.foreign_blocker,
           "CPU dependency inspection cannot bypass the original batch generation");
    auto wrong_cutoff = grant;
    ++wrong_cutoff.cutoff;
    expect(test.actor->inspect_source(prepared.request, wrong_cutoff, 1).status ==
               CUDA_ERROR_INVALID_VALUE,
           "CPU dependency inspection rejects a stale stage cutoff");
    const auto early_inspection = test.actor->inspect_source(prepared.request, grant, 1);
    expect(early_inspection.status == CUDA_ERROR_NOT_READY && !early_inspection.source_session &&
               !early_inspection.dependency.foreign_blocker,
           "CPU dependency inspection cannot invent an unaccepted source turn");
    CUresult foreign_inspection{};
    std::thread inspection_thread([&] {
        foreign_inspection = test.actor->inspect_source(prepared.request, grant, 1).status;
    });
    inspection_thread.join();
    expect(foreign_inspection == CUDA_ERROR_INVALID_CONTEXT && native.primary_retains == before &&
               native.calls[0] == 0,
           "foreign inspection never activates an owner or invokes a native callback");
    const auto early_progress = test.actor->progress_source(prepared.request, grant, 1);
    expect(early_progress.status == CUDA_ERROR_NOT_READY && !early_progress.issue.sequence &&
               native.primary_retains == before && native.calls[0] == 0,
           "source progress requires a previously accepted exact source turn");
    expect(test.actor->issue(prepared.request, grant) == CUDA_SUCCESS, "one exact initial grant");
    expect(test.actor->issue(prepared.request, grant) == CUDA_ERROR_INVALID_VALUE,
           "old stage grant cannot replay");
    expect(
        test.actor->inspect_source(prepared.request, grant, 1).status == CUDA_ERROR_INVALID_VALUE,
        "an earlier inspected stage grant cannot authorize a later native stage");
    await([&] {
        test.step();
        return test.actor->reply(prepared.request).ready;
    });
    test.release(prepared.request);
    test.finish();
    expect(native.calls[0] == 1, "exactly one original UID callback");
}
void source_dependency_inspection() {
    Harness test(true, 2);
    const auto comm = test.init_comm();
    {
        auto& origin = *test.sources[0];
        auto& target = *test.sources[1];
        expect(origin.activate().status == CUDA_SUCCESS, "select captured foreign origin");
        const auto event = origin.create_event(0);
        expect(event.status == CUDA_SUCCESS && event.id, "create exact source event owner");
        const auto recorded = origin.record_event_with_frontier(1, event.id, 0);
        expect(recorded.result.status == CUDA_SUCCESS && recorded.result.id == 1,
               "atomically accept original record and its exact frontier");
        const auto captured = origin.capture_event(event.id);
        expect(captured.status == CUDA_SUCCESS && captured.dependency,
               "capture the original immutable event generation");
        expect(origin.deactivate().status == CUDA_SUCCESS, "restore NULL after origin capture");
        expect(target.activate().status == CUDA_SUCCESS, "select exact target source");
        const auto waited = target.wait_event_with_frontier(1, captured.event, 0);
        expect(waited.result.status == CUDA_SUCCESS && waited.result.id == 1,
               "admit an actual foreign wait before the native source turn");
        expect(target.deactivate().status == CUDA_SUCCESS, "restore NULL before actor entry");
        const auto prepared =
            test.actor->prepare({17, 0}, test.request(test.collective(comm, 2), 2));
        expect(prepared.status == CUDA_SUCCESS && prepared.request,
               "prepare actual native request with the blocked target source");
        expect(test.actor->issue(prepared.request, test.grant(prepared.request)) ==
                   CUDA_ERROR_NOT_READY,
               "exact setup admits the source turn but cannot begin its blocked foreign prefix");
        const auto grant = test.grant(prepared.request);
        const auto records = native.records;
        for (unsigned attempt = 0; attempt != 8; ++attempt) {
            const auto inspected = test.actor->inspect_source(prepared.request, grant, 2);
            expect(inspected.status == CUDA_SUCCESS && inspected.source_session == 2 &&
                       inspected.dependency.status == CUDA_ERROR_NOT_READY &&
                       inspected.dependency.foreign_blocker &&
                       inspected.dependency.foreign_blocker->same_capture(*captured.dependency),
                   "CPU-only inspection exposes only the actual old foreign dependency capture");
        }
        expect(!current && native.records == records && native.calls[8] == 0,
               "inspection never selects a context, records a prefix or invokes NCCL");
        expect(test.actor->inspect_source(prepared.request, grant, 99).status ==
                   CUDA_ERROR_INVALID_VALUE,
               "a valid batch cannot inspect an unrelated source identity");
        expect(test.actor->reject_source(prepared.request, grant, 2, CUDA_ERROR_INVALID_VALUE) ==
                   CUDA_ERROR_INVALID_VALUE,
               "a valid pending turn is not authority to invent a settled dependency error");
        const auto blocked = test.actor->progress_source(prepared.request, grant, 2);
        expect(blocked.status == CUDA_SUCCESS &&
                   blocked.issue.state == CudaExecutionIssueState::kIdle &&
                   !blocked.issue.sequence && native.records == records,
               "root-only progress cannot issue the foreign owner's ungranted command");
        expect(origin.activate().status == CUDA_SUCCESS, "select exact required leaf owner");
        const auto issued = origin.progress_frontier_dependency(*captured.dependency);
        expect(
            issued.state == CudaExecutionIssueState::kIssued && issued.sequence == 1 &&
                issued.native_status == CUDA_SUCCESS && origin.poll().status == CUDA_SUCCESS,
            "one exact leaf record issues and physically retires without source-wide scheduling");
        expect(origin.deactivate().status == CUDA_SUCCESS, "leaf progress restores proven NULL");
        const auto progressed = test.actor->progress_source(prepared.request, grant, 2);
        expect(progressed.status == CUDA_SUCCESS && progressed.source_session == 2 &&
                   progressed.issue.state == CudaExecutionIssueState::kIssued &&
                   progressed.issue.sequence == 1 && !current && native.calls[8] == 0,
               "the original root grant advances only its actual now-ready local wait");
        await([&] {
            test.step();
            return test.actor->reply(prepared.request).ready;
        });
        const auto completed = test.actor->reply(prepared.request);
        expect(
            completed.envelope && completed.envelope->original_status == 0 && native.calls[8] == 1,
            "the original collective runs exactly once after the actual required dependency");
        test.release(prepared.request);
        for (std::size_t index = 0; index != 16; ++index) {
            expect(native.memory[0][index] == static_cast<float>(index + 1),
                   "every origin element survives unrelated native source computation");
            expect(native.memory[1][index] == static_cast<float>(101 + index % 8),
                   "every target input and output element matches the exact native collective");
        }
        auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
        test.release(destroy_id);
        expect(origin.activate().status == CUDA_SUCCESS &&
                   origin.destroy_event(event.id).status == CUDA_SUCCESS &&
                   origin.deactivate().status == CUDA_SUCCESS,
               "retire the public event while its completed immutable capture remains valid");
    }
    test.finish();
}
void grouping() {
    Harness test(true, 2);
    const auto comm = test.init_comm();
    test.group = 99;
    auto [start_id, started] = test.run(wire::ExecutionNcclGroupStart{});
    const auto original = test.grant(start_id);
    test.release(start_id);
    auto [nested_id, nested] = test.run(wire::ExecutionNcclGroupStart{});
    test.release(nested_id);
    for (unsigned source = 1; source <= 2; ++source) {
        auto [id, result] = test.run(test.collective(comm, source), source);
        const auto grant = test.grant(id);
        expect(grant.reservation == original.reservation && grant.generation == original.generation,
               "released inners retain original batch authority");
        expect(!result.native->physical_complete, "inner host return is not whole-group stop");
        test.release(id);
    }
    auto [nested_end, inner] = test.run(wire::ExecutionNcclGroupEnd{});
    test.release(nested_end);
    auto [end_id, ended] = test.run(wire::ExecutionNcclGroupEnd{});
    test.group = 0;
    const auto end_grant = test.grant(end_id);
    expect(end_grant.reservation == original.reservation &&
               end_grant.generation == original.generation,
           "End retains original reservation, not fresh connection grant");
    test.release(end_id);
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
    expect(native.calls[8] == 2 && native.calls[6] == 2 && native.calls[7] == 2,
           "real grouped inners and nested TLS End calls");
}
void errors_and_identity() {
    Harness test;
    auto invalid = test.request(wire::ExecutionNcclGetUniqueId{true});
    invalid.scope.context = 999;
    expect(test.actor->prepare({17, 0}, invalid).status == CUDA_ERROR_INVALID_CONTEXT,
           "foreign source rejected before SDK");
    auto primary = test.request(wire::ExecutionNcclGetUniqueId{true});
    expect(test.actor->prepare({17, 0}, primary, 1).status == CUDA_ERROR_INVALID_CONTEXT,
           "unobserved primary is not authority");
    native.result = 4;
    auto [id, result] = test.run(wire::ExecutionNcclGetUniqueId{true});
    expect(result.envelope->original_status == 4 && result.envelope->observation.effects == 1,
           "actual native error retains exact typed output effect");
    test.release(id);
    native.result = 0;
    auto [next_id, next] = test.run(wire::ExecutionNcclGetUniqueId{true});
    test.release(next_id);
    expect(native.calls[0] == 2,
           "clean rejected public sequence does not gap coordinator sequence");
    test.finish();
}
void grouped_output() {
    Harness test(true, 1, true);
    test.group = 18;
    auto [start_id, started] = test.run(wire::ExecutionNcclGroupStart{});
    test.release(start_id);
    auto [init_id, initialized] = test.run(wire::ExecutionNcclInitRank{true, 1, 0, {}});
    expect(initialized.envelope->observation.effects == 2 &&
               initialized.envelope->observation.communicator.object,
           "pinned stable grouped Init publishes actual synchronous output before End");
    const auto comm = initialized.envelope->observation.communicator;
    const auto reservation = test.grant(init_id);
    test.release(init_id);
    auto [end_id, ended] = test.run(wire::ExecutionNcclGroupEnd{});
    test.group = 0;
    expect(ended.envelope->publications.empty(),
           "already published stable cell is not written twice");
    expect(test.grant(end_id).reservation == reservation.reservation,
           "grouped output retains batch ownership");
    test.release(end_id);
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
}
void async_continuation() {
    Harness test;
    const auto comm = test.init_comm();
    native.result = 7;
    auto [id, pending] = test.run(test.collective(comm), 1);
    const auto original = test.grant(id);
    expect(pending.envelope->original_status == 7 && !pending.native->enqueue_complete,
           "native in-progress is not enqueue completion");
    test.release(id);
    native.result = 0;
    auto [query_id, queried] = test.run(wire::ExecutionNcclGetAsyncError{comm, true});
    expect(test.grant(query_id).reservation == original.reservation,
           "async query binds original retained reservation");
    test.release(query_id);
    await([&] {
        test.step();
        return test.actor->snapshot().reservations.empty();
    });
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
    expect(native.calls[8] == 1, "async record continuation never replays collective");
}
void prepared_group_credit() {
    NcclActorConfig config;
    config.maximum_groups = 1;
    config.coordinator.maximum_retained_groups = 2;
    Harness test(true, 1, false, config);
    test.group = 41;
    const auto first = test.actor->prepare({17, 0}, test.request(wire::ExecutionNcclGroupStart{}));
    expect(first.status == CUDA_SUCCESS && first.request,
           "first prepared Start reserves an End slot");
    const auto second =
        test.actor->prepare({17, 0}, {{{93, 94}, 1, 51, 0}, wire::ExecutionNcclGroupStart{}});
    expect(second.status == CUDA_ERROR_OUT_OF_MEMORY && !second.request,
           "prepared Start consumes group credit before native TLS opens");
    expect(native.calls[6] == 0 && native.primary_retains == 0,
           "both preparation decisions are CPU-only, no native Start to strand");
    await([&] {
        test.step();
        return test.actor->reply(first.request).ready;
    });
    test.release(first.request);
    auto [end_id, ended] = test.run(wire::ExecutionNcclGroupEnd{});
    test.group = 0;
    expect(ended.envelope->original_status == 0, "first group actually ended");
    const auto retained_reply =
        test.actor->prepare({17, 0}, {{{93, 94}, 2, 51, 0}, wire::ExecutionNcclGroupStart{}});
    expect(retained_reply.status == CUDA_ERROR_OUT_OF_MEMORY && !retained_reply.request,
           "retained End receipt does not prematurely refund group credit");
    test.release(end_id);
    expect(test.actor->snapshot().reservations.empty(),
           "complete released group refunds its reservation");
    const auto admitted =
        test.actor->prepare({17, 0}, {{{93, 94}, 3, 51, 0}, wire::ExecutionNcclGroupStart{}});
    expect(admitted.status == CUDA_SUCCESS && admitted.request,
           "fresh group admitted after credit refund");
    await([&] {
        test.step();
        return test.actor->reply(admitted.request).ready;
    });
    test.release(admitted.request);
    const auto last =
        test.actor->prepare({17, 0}, {{{93, 94}, 4, 51, 0}, wire::ExecutionNcclGroupEnd{}});
    expect(last.status == CUDA_SUCCESS && last.request, "reserved End remains available");
    await([&] {
        test.step();
        return test.actor->reply(last.request).ready;
    });
    test.release(last.request);
    test.finish();
    expect(native.calls[6] == 2 && native.calls[7] == 2,
           "rejected Starts never entered native TLS or gapped accepted sequence");
}
void async_group_credit() {
    NcclActorConfig config;
    config.maximum_groups = 1;
    config.coordinator.maximum_retained_groups = 2;
    Harness test(true, 1, false, config);
    const auto comm = test.init_comm();
    test.group = 71;
    auto [start_id, started] = test.run(wire::ExecutionNcclGroupStart{});
    const auto batch = test.grant(start_id);
    test.release(start_id);
    auto [inner_id, inner] = test.run(test.collective(comm), 1);
    test.release(inner_id);
    native.result = 7;
    auto [end_id, ended] = test.run(wire::ExecutionNcclGroupEnd{});
    test.group = 0;
    expect(ended.envelope->original_status == 7 && !ended.native->physical_complete,
           "actual End7 closes native TLS but leaves its group retained");
    test.release(end_id);
    native.result = 0;
    const auto rejected =
        test.actor->prepare({17, 0}, {{{93, 94}, 1, 81, 0}, wire::ExecutionNcclGroupStart{}});
    expect(rejected.status == CUDA_ERROR_OUT_OF_MEMORY && !rejected.request,
           "released unfinished End7 retains its reserved End credit");
    expect(native.calls[6] == 1, "no second native Start with an unavailable End slot");
    native.stop = false;
    const auto records_before_query = native.records;
    const auto retained_end_id = end_id;
    auto [query_id, queried] = test.run(wire::ExecutionNcclGetAsyncError{comm, true});
    expect(queried.envelope->original_status == 0 &&
               test.grant(query_id).reservation == batch.reservation &&
               test.grant(query_id).generation == batch.generation,
           "group capacity does not disable exact original-batch async progress");
    test.release(query_id);
    await([&] {
        test.step();
        const auto result = test.actor->reply(retained_end_id);
        const auto snapshot = test.actor->snapshot();
        expect(snapshot.status == CUDA_SUCCESS, "held-tail reservation snapshot");
        bool waiting{};
        for (const auto& reservation : snapshot.reservations)
            for (const auto& request : reservation.requests)
                if (request.request == retained_end_id)
                    waiting = request.stage == NcclActorStage::kWaitingTails;
        return waiting && result.native && result.native->enqueue_complete &&
               !result.native->physical_complete;
    });
    expect(native.records > records_before_query,
           "real final source tail was recorded and remains held");
    const auto held_tail =
        test.actor->prepare({17, 0}, {{{93, 94}, 2, 81, 0}, wire::ExecutionNcclGroupStart{}});
    expect(held_tail.status == CUDA_ERROR_OUT_OF_MEMORY && !held_tail.request,
           "real async readiness does not refund an unretired source tail");
    native.stop = true;
    await([&] {
        test.step();
        return test.actor->snapshot().reservations.empty();
    });
    const auto admitted =
        test.actor->prepare({17, 0}, {{{93, 94}, 3, 81, 0}, wire::ExecutionNcclGroupStart{}});
    expect(admitted.status == CUDA_SUCCESS && admitted.request,
           "group credit refunded only after cleanup of actual source tails");
    await([&] {
        test.step();
        return test.actor->reply(admitted.request).ready;
    });
    test.release(admitted.request);
    const auto last =
        test.actor->prepare({17, 0}, {{{93, 94}, 4, 81, 0}, wire::ExecutionNcclGroupEnd{}});
    expect(last.status == CUDA_SUCCESS && last.request,
           "new native group can always reach its reserved End");
    await([&] {
        test.step();
        return test.actor->reply(last.request).ready;
    });
    test.release(last.request);
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
    expect(native.calls[6] == 2 && native.calls[7] == 2 && native.calls[8] == 1 &&
               native.calls[3] == 1,
           "credit repair adds no native replay or invented async completion");
}
void retained_tail() {
    Harness test;
    const auto comm = test.init_comm();
    native.stop = false;
    auto [id, result] = test.run(test.collective(comm), 1);
    expect(!result.native->physical_complete,
           "held exact source event prevents physical completion");
    test.release(id);
    expect(!test.actor->snapshot().reservations.empty(), "ACK retains source tail ownership");
    expect(test.sources[0]->activate().status == CUDA_SUCCESS, "source owner selected");
    expect(test.sources[0]->release(test.allocations[0]).status == CUDA_ERROR_NOT_READY,
           "immutable admitted buffer lease prevents early free");
    expect(test.sources[0]->deactivate().status == CUDA_SUCCESS, "restore NULL");
    native.stop = true;
    await([&] {
        test.step();
        return test.actor->snapshot().reservations.empty();
    });
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
}
void held_callback_and_context() {
    Harness test;
    const auto comm = test.init_comm();
    native.hold = true;
    native.release = false;
    native.entered = false;
    const auto prepared = test.actor->prepare({17, 0}, test.request(test.collective(comm), 1));
    expect(prepared.status == CUDA_SUCCESS, "held collective preparation");
    expect(test.actor->issue(prepared.request, test.grant(prepared.request)) == CUDA_SUCCESS,
           "begin exact source/native submission");
    await([] { return native.entered.load(); });
    const auto record_count = native.records;
    for (unsigned i = 0; i != 8; ++i)
        expect(test.actor->observe() == CUDA_SUCCESS,
               "bounded CPU observation while worker blocked");
    expect(!test.actor->reply(prepared.request).ready && native.records == record_count,
           "no guessed host return or unauthorized prefix recording");
    native.release = true;
    await([&] {
        expect(test.actor->observe() == CUDA_SUCCESS, "consume callback");
        const auto snapshot = test.actor->snapshot();
        for (const auto& batch : snapshot.reservations)
            for (const auto& item : batch.requests)
                if (item.request == prepared.request)
                    return item.stage == NcclActorStage::kFinishSources;
        return false;
    });
    expect(native.records == record_count, "CPU observe does not finish source turn");
    expect(test.actor->issue(prepared.request, test.grant(prepared.request)) == CUDA_SUCCESS,
           "new exact grant records source prefix");
    await([&] {
        test.step();
        return test.actor->reply(prepared.request).ready;
    });
    test.release(prepared.request);
    native.hold = false;
    native.clear_current = true;
    auto [clear_id, cleared] = test.run(wire::ExecutionNcclGetUniqueId{true}, 1);
    expect(cleared.envelope->context_effect.kind ==
               wire::ExecutionNcclContextEffectKind::kClearCurrent,
           "actual source-to-NULL context effect");
    test.release(clear_id);
    native.clear_current = false;
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
}
void no_producer() {
    Harness test(false);
    const auto prepared =
        test.actor->prepare({17, 0}, test.request(wire::ExecutionNcclGetUniqueId{true}));
    expect(prepared.status == CUDA_SUCCESS,
           "missing producer still permits private native evidence");
    await([&] {
        test.step();
        const auto result = test.actor->reply(prepared.request);
        return result.native && result.native->reply_ready;
    });
    const auto result = test.actor->reply(prepared.request);
    expect(!result.ready && !result.envelope && result.native->original_status == 0,
           "byte differences cannot fabricate public output writes");
    expect(test.actor->release_reply(prepared.request) == CUDA_ERROR_NOT_READY,
           "unproved public output is not acknowledged");
    // Intentionally isolated child: unavailable public proof keeps the receipt.
}
void owner_and_disconnect() {
    Harness test;
    CUresult foreign{};
    std::thread other([&] { foreign = test.actor->observe(); });
    other.join();
    expect(foreign == CUDA_ERROR_INVALID_CONTEXT,
           "foreign thread rejected before state observation");
    const auto child = ::fork();
    expect(child >= 0, "fixture fork");
    if (!child) std::_Exit(test.actor->observe() == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork gate before inherited state");
    auto [id, result] = test.run(wire::ExecutionNcclGetUniqueId{true});
    test.release(id);
    expect(test.actor->disconnect(17) == CUDA_SUCCESS, "known peer disconnect");
    expect(
        test.actor->prepare({17, 0}, test.request(wire::ExecutionNcclGetUniqueId{true})).status ==
            CUDA_ERROR_DEINITIALIZED,
        "disconnected caller does not enqueue");
    test.finish();
}

void ordinary_authorization() {
    Harness test;
    const auto original = open_ordinary_group(test);
    const auto empty = test.actor->snapshot();
    expect(empty.reservations.size() == 1 && empty.reservations[0].requests.empty(),
           "released Start has no pending NCCL request to lend fabricated authority");
    auto input = ordinary_scope(1);
    auto target = ordinary_target(wire::ExecutionOpcode::kGetContextFlags, 101);
    const auto native_records = native.records;
    const auto prepared = test.actor->prepare_continuation({17, 0}, input, target);
    expect(prepared.status == CUDA_SUCCESS, "ordinary work attaches to empty open batch");
    const auto snapshot = ordinary_snapshot(test, prepared.request);
    expect(snapshot.grant.reservation == original.reservation &&
               snapshot.grant.generation == original.generation &&
               snapshot.grant.cutoff != original.cutoff && snapshot.is_ready,
           "original reservation/generation and fresh exact ordinary cutoff");
    auto stale = snapshot.grant;
    ++stale.generation;
    expect(
        test.actor->begin_continuation(prepared.request, stale).status == CUDA_ERROR_INVALID_VALUE,
        "stale generation cannot enter an ordinary handler");
    CUresult foreign{};
    std::thread thread(
        [&] { foreign = test.actor->begin_continuation(prepared.request, snapshot.grant).status; });
    thread.join();
    expect(foreign == CUDA_ERROR_INVALID_CONTEXT, "ordinary permit is owner-thread bound");
    const auto child = ::fork();
    expect(child >= 0, "ordinary actor fork check");
    if (!child)
        std::_Exit(test.actor->begin_continuation(prepared.request, snapshot.grant).status ==
                           CUDA_ERROR_INVALID_CONTEXT
                       ? 0
                       : 1);
    int child_status{};
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               !WEXITSTATUS(child_status),
           "fork cannot consume an inherited ordinary permit");
    const auto permit = ordinary_begin(test, prepared.request);
    expect(test.actor->observe() == CUDA_ERROR_NOT_READY &&
               test.actor->snapshot().status == CUDA_ERROR_NOT_READY &&
               test.actor->cancel_continuation(prepared.request) == CUDA_ERROR_NOT_READY,
           "armed synchronous attempt excludes actor callback reentry");
    auto wrong_permit = permit;
    ++wrong_permit.attempt;
    expect(test.actor->finish_continuation_attempt(wrong_permit, ordinary_reply()) ==
               CUDA_ERROR_INVALID_VALUE,
           "wrong permit cannot consume the actual attempt");
    NcclActorContinuationOutcome deferred;
    deferred.kind = NcclActorContinuationOutcomeKind::kDeferred;
    deferred.status = CUDA_ERROR_NOT_READY;
    expect(test.actor->finish_continuation_attempt(permit, std::move(deferred)) == CUDA_SUCCESS,
           "proven unentered Deferred remains the same ordinary call");
    expect(test.actor->finish_continuation_attempt(permit, ordinary_reply()) ==
                   CUDA_ERROR_INVALID_VALUE &&
               test.actor->begin_continuation(prepared.request, snapshot.grant).status ==
                   CUDA_ERROR_INVALID_VALUE,
           "old permit and old cutoff cannot replay");
    ordinary_return(test, prepared.request, ordinary_reply());
    expect(
        test.actor->prepare_continuation({17, 0}, input, target).status == CUDA_ERROR_INVALID_VALUE,
        "normal ordinary sequence is consumed once, not a retry namespace");
    input.ordinary_sequence = 2;
    input.group = 78;
    expect(
        test.actor->prepare_continuation({17, 0}, input, target).status == CUDA_ERROR_INVALID_VALUE,
        "wrong live group cleanly rejects");
    input.group = 77;
    expect(
        test.actor->prepare_continuation({17, 0}, input, target).status == CUDA_ERROR_INVALID_VALUE,
        "canonical policy rejection still consumed its ordinary sequence");
    input.ordinary_sequence = 3;
    auto invalid = target;
    invalid.operation = wire::ExecutionOpcode::kNcclCall;
    expect(test.actor->prepare_continuation({17, 0}, input, invalid).status ==
               CUDA_ERROR_INVALID_VALUE,
           "nested native call is not a canonical ordinary request");
    for (const auto operation :
         {wire::ExecutionOpcode::kQueryStagingIdle, wire::ExecutionOpcode::kQueryHostFunction,
          wire::ExecutionOpcode::kFinishHostFunction}) {
        invalid.operation = operation;
        expect(test.actor->prepare_continuation({17, 0}, input, invalid).status ==
                   CUDA_ERROR_INVALID_VALUE,
               "independent helper progress never borrows an application continuation");
    }
    const auto cancelled = test.actor->prepare_continuation({17, 0}, input, target);
    expect(cancelled.status == CUDA_SUCCESS &&
               test.actor->cancel_continuation(cancelled.request) == CUDA_SUCCESS &&
               test.actor->cancel_continuation(cancelled.request) == CUDA_ERROR_INVALID_HANDLE,
           "exact unentered cancellation refunds only its independent slot");
    input.ordinary_sequence = 4;
    target.operation = wire::ExecutionOpcode::kEnqueueHostFunction;
    const auto callback = test.actor->prepare_continuation({17, 0}, input, target);
    expect(callback.status == CUDA_SUCCESS &&
               test.actor->cancel_continuation(callback.request) == CUDA_SUCCESS,
           "source host-function admission is an ordinary continuation, not a native call");
    expect(native.records == native_records && native.calls[6] == 1 && native.calls[7] == 0,
           "all ordinary bookkeeping leaves SDK and NCCL counters unchanged");
    close_ordinary_group(test);
    test.finish();
}

void ordinary_command_and_retirement() {
    Harness test;
    const auto original = open_ordinary_group(test);
    auto& source = *test.sources[0];
    const auto prepared = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(1), ordinary_target(wire::ExecutionOpcode::kMemory, 201));
    expect(prepared.status == CUDA_SUCCESS, "ordinary queued command prepared");
    const auto prefix = source.capture_frontier();
    expect(prefix.status == CUDA_SUCCESS && prefix.frontier.retirement().is_retired,
           "pre-call prefix is exact and already physically quiescent");
    expect(test.actor->bind_continuation_dependency(
               prepared.request, {1, test.sources[0], prefix.frontier, {}}) == CUDA_SUCCESS,
           "bind exact cold prefix");
    const auto permit = ordinary_begin(test, prepared.request);
    expect(source.activate().status == CUDA_SUCCESS, "runtime selects exact source for admission");
    const auto reservation = source.reserve_command_frontier(1);
    expect(reservation.status == CUDA_SUCCESS, "reserve exact target before acceptance");
    CudaSessionMemory memory;
    memory.kind = CudaMemoryKind::kFill32;
    memory.destination = {test.allocations[0], 4 * sizeof(float)};
    memory.count = 8;
    memory.value = 0x3f800000U;
    memory.is_asynchronous = true;
    const auto accepted = source.accept_memory(1, memory);
    const auto bound = source.reserved_command_frontier(reservation.reservation);
    expect(accepted.accepted() && bound.status == CUDA_SUCCESS,
           "actual atomic target admission binds its preallocated observation");
    expect(source.deactivate().status == CUDA_SUCCESS, "admission leaves NULL context");
    NcclActorContinuationOutcome outcome;
    outcome.kind = NcclActorContinuationOutcomeKind::kCommandAccepted;
    outcome.status = CUDA_SUCCESS;
    outcome.binding =
        NcclActorContinuationBinding{1, test.sources[0], bound.frontier, accepted.receipt};
    expect(test.actor->finish_continuation_attempt(permit, std::move(outcome)) == CUDA_SUCCESS,
           "store exact receipt and target without allocating a replacement capture");
    const auto pending = ordinary_snapshot(test, prepared.request);
    expect(!pending.is_ready && pending.dependencies == std::vector<std::uint64_t>{1},
           "queued target is a required leaf, not permission to rerun admission");
    const auto inspected =
        test.actor->inspect_continuation_source(prepared.request, pending.grant, 1);
    expect(inspected.status == CUDA_SUCCESS &&
               inspected.binding.frontier.same_capture(bound.frontier) &&
               !inspected.binding.frontier.same_capture(prefix.frontier),
           "accepted target supersedes the same-source pre-call capture");
    expect(test.actor->cancel_continuation(prepared.request) == CUDA_ERROR_NOT_READY &&
               test.actor->complete_continuation(prepared.request, ordinary_reply(1)) ==
                   CUDA_ERROR_NOT_READY,
           "accepted command needs actual host provenance");
    native.stop = false;
    expect(source.activate().status == CUDA_SUCCESS, "select granted exact command owner");
    const auto issued = source.progress_frontier_dependency(inspected.binding.frontier);
    expect(issued.state == CudaExecutionIssueState::kIssued && issued.sequence == 1 &&
               native.fills == 1,
           "one runtime-authorized target runs exactly once");
    expect(source.deactivate().status == CUDA_SUCCESS, "actual issue restores owner NULL");
    auto returned = ordinary_reply(1);
    returned.reply->original_status = 0;
    expect(test.actor->complete_continuation(prepared.request, std::move(returned)) == CUDA_SUCCESS,
           "consume original result without a second native permission");
    const auto receipt = test.actor->continuation_reply(prepared.request);
    expect(receipt.ready && !receipt.is_retired && receipt.host_reply &&
               receipt.host_reply->origin == glimmer::backend::ExecutionReplyOrigin::kNativeSdk,
           "host return is stored separately from unretired physical tail");
    expect(source.acknowledge_result(accepted.receipt), "runtime ACK follows actor receipt cache");
    expect(test.actor->continuation_reply(prepared.request).host_reply->status == CUDA_SUCCESS,
           "cached exact provenance survives backend ACK");
    expect(test.actor->release_continuation_reply(prepared.request) == CUDA_SUCCESS,
           "ordinary reply release cannot retire its tail");
    close_ordinary_group(test);
    const auto held = test.actor->snapshot();
    expect(held.reservations.size() == 1 && !held.reservations[0].has_open_group &&
               !held.reservations[0].has_physical_work &&
               held.reservations[0].generation == original.generation,
           "End and ACK retain identity without duplicating source physical ownership");
    for (std::size_t index = 0; index != 16; ++index)
        expect(native.memory[0][index] ==
                   (index >= 4 && index < 12 ? 1.0F : static_cast<float>(index + 1)),
               "complete original output and both four-value guards");
    native.stop = true;
    test.finish();
}

void ordinary_rejected_prefix() {
    Harness test;
    open_ordinary_group(test);
    auto& source = *test.sources[0];
    const auto prepared = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(1), ordinary_target(wire::ExecutionOpcode::kMemory, 301));
    const auto permit = ordinary_begin(test, prepared.request);
    expect(source.activate().status == CUDA_SUCCESS, "select clean rejection owner");
    const auto reservation = source.reserve_command_frontier(1);
    CudaSessionMemory memory;
    memory.kind = CudaMemoryKind::kFill32;
    memory.destination = {test.allocations[0], 16};
    memory.count = 8;
    memory.is_asynchronous = true;
    const auto accepted = source.accept_memory(1, memory);
    const auto target = source.reserved_command_frontier(reservation.reservation);
    expect(accepted.accepted() && target.status == CUDA_SUCCESS, "exact clean-rejection target");
    expect(source.deactivate().status == CUDA_SUCCESS, "restore owner before CPU binding");
    NcclActorContinuationOutcome bound;
    bound.kind = NcclActorContinuationOutcomeKind::kCommandAccepted;
    bound.status = CUDA_SUCCESS;
    bound.binding =
        NcclActorContinuationBinding{1, test.sources[0], target.frontier, accepted.receipt};
    expect(test.actor->finish_continuation_attempt(permit, std::move(bound)) == CUDA_SUCCESS,
           "bind before runtime's exact targeted rejection");
    expect(source.reject_command(accepted.receipt, CUDA_ERROR_OUT_OF_MEMORY),
           "only exact never-attempted backend command is rejected");
    auto failed = ordinary_reply(1);
    failed.reply->disposition = wire::ExecutionWireDisposition::kRejected;
    failed.reply->status = CUDA_ERROR_OUT_OF_MEMORY;
    expect(test.actor->complete_continuation(prepared.request, std::move(failed)) == CUDA_SUCCESS,
           "known rejected-before-issue error with independent retirement stays healthy");
    const auto observed = test.actor->continuation_reply(prepared.request);
    expect(observed.ready && observed.is_retired && observed.host_reply &&
               observed.host_reply->origin ==
                   glimmer::backend::ExecutionReplyOrigin::kRejectedBeforeIssue &&
               !observed.reply->original_status && native.fills == 0,
           "no fabricated SDK status or compute for the clean rejection");
    expect(source.acknowledge_result(accepted.receipt) &&
               test.actor->release_continuation_reply(prepared.request) == CUDA_SUCCESS,
           "clean receipt and independent metadata can be acknowledged");
    const auto next = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(2), ordinary_target(wire::ExecutionOpcode::kGetContextFlags, 302));
    expect(next.status == CUDA_SUCCESS &&
               test.actor->bind_continuation_dependency(
                   next.request, {1, test.sources[0], target.frontier, {}}) == CUDA_SUCCESS,
           "cold handler binds exact retired historical failure without clearing it");
    const auto inspected = test.actor->inspect_continuation_source(
        next.request, ordinary_snapshot(test, next.request).grant, 1);
    expect(inspected.dependency.status == CUDA_SUCCESS &&
               inspected.binding.frontier.observe() == CUDA_ERROR_OUT_OF_MEMORY &&
               inspected.binding.frontier.retirement().is_retired,
           "quiescence view preserves historical error without blocking cold metadata");
    ordinary_return(test, next.request, ordinary_reply());
    close_ordinary_group(test);
    test.finish();
}

void ordinary_import_cleanup() {
    NcclActorConfig config;
    config.maximum_groups = 1;
    config.maximum_continuations = 1;
    config.maximum_cleanup_parents = 2;
    Harness test(true, 1, false, config);
    const auto helper = std::make_shared<int>(51);
    std::array<NcclActorGrant, 2> groups;
    for (std::size_t index = 0; index != 2; ++index) {
        const auto group = 77 + index;
        groups[index] = open_ordinary_group(test, group);
        auto input = ordinary_scope(index + 1);
        input.group = group;
        input.origin_source = 0;
        auto target = ordinary_target(wire::ExecutionOpcode::kImportHostMemory, 401 + index);
        target.connection = 51;
        target.helper_owner = helper;
        target.is_context_free_helper = true;
        const auto prepared = test.actor->prepare_continuation({17, 0}, input, target);
        expect(prepared.status == CUDA_SUCCESS,
               "context-free HST1/STG1 import reserves independent parent credit");
        auto permit = ordinary_begin(test, prepared.request);
        NcclActorContinuationOutcome pending;
        pending.kind = NcclActorContinuationOutcomeKind::kHostJobPending;
        pending.status = CUDA_SUCCESS;
        pending.pending_lifetime = helper;
        pending.pending_sequence = target.wire_sequence;
        expect(test.actor->finish_continuation_attempt(permit, pending) == CUDA_SUCCESS,
               "retain exactly accepted asynchronous helper mailbox job");
        expect(!test.actor->continuation_reply(prepared.request).ready &&
                   test.actor->cancel_continuation(prepared.request) == CUDA_ERROR_NOT_READY,
               "pending helper call cannot be declared returned or cancelled");
        if (index == 0) {
            NcclActorContinuationOutcome deferred;
            deferred.kind = NcclActorContinuationOutcomeKind::kDeferred;
            deferred.status = CUDA_ERROR_NOT_READY;
            expect(test.actor->complete_continuation(prepared.request, std::move(deferred)) ==
                       CUDA_SUCCESS,
                   "real no-entry helper Deferred keeps same ordinary sequence");
            permit = ordinary_begin(test, prepared.request);
            expect(test.actor->finish_continuation_attempt(permit, pending) == CUDA_SUCCESS,
                   "another helper attempt needs a fresh Core-stage permit");
        }
        auto imported = ordinary_reply(900 + index);
        imported.reply->original_status = 0;
        imported.reply->data.assign(4, std::byte{});
        imported.imported_allocation = 900 + index;
        expect(test.actor->complete_continuation(prepared.request, std::move(imported)) ==
                   CUDA_SUCCESS,
               "actual successful import ID binds parent before any ACK");
        const auto reply = test.actor->continuation_reply(prepared.request);
        expect(reply.ready && reply.is_retired && reply.reply->id == 900 + index,
               "ordinary result fields are unchanged by actor bookkeeping");
        expect(test.actor->release_continuation_reply(prepared.request) == CUDA_SUCCESS,
               "parent survives import reply release");
        close_ordinary_group(test);
        await([&] {
            test.step();
            const auto state = test.actor->snapshot();
            return std::ranges::all_of(state.reservations, [](const auto& batch) {
                return batch.requests.empty() && !batch.has_physical_work && !batch.has_open_group;
            });
        });
    }
    expect(groups[0].reservation != groups[1].reservation &&
               groups[0].generation != groups[1].generation,
           "long-lived parent metadata does not occupy maximum_groups End credit");
    const auto retained = test.actor->snapshot();
    expect(retained.reservations.size() == 2 &&
               std::ranges::none_of(retained.reservations,
                                    [](const auto& batch) {
                                        return batch.has_physical_work || batch.has_unknown_state;
                                    }),
           "parent metadata retains identity, never physical/progress authority");
    for (std::size_t index : {std::size_t{1}, std::size_t{0}}) {
        auto input = ordinary_scope(1);
        input.flags = wire::kExecutionSourceContinuationCleanup;
        input.group = 77 + index;
        input.origin_source = 0;
        input.cleanup_parent = index + 1;
        auto target = ordinary_target(wire::ExecutionOpcode::kReleaseHostMemory, 501 + index);
        target.connection = 51;
        target.helper_owner = helper;
        target.is_context_free_helper = true;
        target.cleanup_allocation = 899;
        expect(test.actor->prepare_continuation({17, 0}, input, target).status ==
                   CUDA_ERROR_INVALID_HANDLE,
               "a parent cannot release any other allocation or consume its legitimate sequence");
        target.cleanup_allocation = 900 + index;
        auto wrong = target;
        wrong.helper_owner = std::make_shared<int>(51);
        expect(test.actor->prepare_continuation({17, 0}, input, wrong).status ==
                   CUDA_ERROR_INVALID_HANDLE,
               "numeric helper identity without exact retained owner is not authority");
        const auto prepared = test.actor->prepare_continuation({17, 0}, input, target);
        expect(prepared.status == CUDA_SUCCESS,
               "independent per-parent cleanup sequence can arrive in either order after End");
        const auto snapshot = ordinary_snapshot(test, prepared.request);
        expect(snapshot.grant.reservation == groups[index].reservation &&
                   snapshot.grant.generation == groups[index].generation,
               "cleanup keeps its original group, not a newer ambient caller batch");
        auto released = ordinary_reply();
        released.reply->original_status = 0;
        ordinary_return(test, prepared.request, std::move(released));
        input.ordinary_sequence = 2;
        expect(test.actor->prepare_continuation({17, 0}, input, target).status ==
                   CUDA_ERROR_INVALID_HANDLE,
               "successful exact parent cleanup is single-use, not replayable");
    }
    test.finish();
}

void ordinary_limits_and_failed_import() {
    NcclActorConfig config;
    config.maximum_continuations = config.maximum_cleanup_parents = 1;
    Harness test(true, 1, false, config);
    open_ordinary_group(test);
    const auto helper = std::make_shared<int>(62);
    auto target = ordinary_target(wire::ExecutionOpcode::kImportHostMemory, 601);
    target.connection = 62;
    target.helper_owner = helper;
    auto first = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(first.status == CUDA_SUCCESS, "first import consumes reserved parent capacity");
    auto failed = ordinary_reply();
    failed.reply->disposition = wire::ExecutionWireDisposition::kRejected;
    failed.reply->status = CUDA_ERROR_INVALID_VALUE;
    const auto permit = ordinary_begin(test, first.request);
    expect(test.actor->finish_continuation_attempt(permit, std::move(failed)) == CUDA_SUCCESS,
           "clean failed import creates no usable cleanup parent");
    expect(test.actor->prepare_continuation({17, 0}, ordinary_scope(2), target).status ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "unreleased reply still charges ordinary pool");
    expect(test.actor->release_continuation_reply(first.request) == CUDA_SUCCESS,
           "failed import refunds reserved parent only after its own reply release");
    auto second = test.actor->prepare_continuation({17, 0}, ordinary_scope(3), target);
    expect(second.status == CUDA_SUCCESS,
           "both bounded pools can be reused without sequence replay");
    auto imported = ordinary_reply(950);
    imported.reply->original_status = 0;
    imported.imported_allocation = 950;
    ordinary_return(test, second.request, std::move(imported));
    expect(test.actor->prepare_continuation({17, 0}, ordinary_scope(4), target).status ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "live successful parent remains separately charged");
    const auto normal = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(5), ordinary_target(wire::ExecutionOpcode::kGetContextFlags, 602));
    expect(normal.status == CUDA_SUCCESS, "parent capacity cannot steal non-import ordinary slots");
    ordinary_return(test, normal.request, ordinary_reply());
    close_ordinary_group(test);
    auto cleanup = ordinary_scope(1);
    cleanup.flags = wire::kExecutionSourceContinuationCleanup;
    cleanup.cleanup_parent = 1;
    target.operation = wire::ExecutionOpcode::kReleaseHostMemory;
    target.cleanup_allocation = 950;
    expect(test.actor->prepare_continuation({17, 0}, cleanup, target).status ==
               CUDA_ERROR_INVALID_HANDLE,
           "failed import ordinal never becomes a cleanup capability");
    cleanup.cleanup_parent = 3;
    const auto released = test.actor->prepare_continuation({17, 0}, cleanup, target);
    expect(released.status == CUDA_SUCCESS, "correct successful import ordinal owns cleanup");
    auto result = ordinary_reply();
    result.reply->original_status = 0;
    ordinary_return(test, released.request, std::move(result));
    test.finish();
}

void ordinary_synchronous_tail() {
    Harness test;
    open_ordinary_group(test);
    auto& source = *test.sources[0];
    std::array<float, 16> values;
    for (std::size_t index = 0; index != values.size(); ++index)
        values[index] = static_cast<float>(index * 3 + 7);
    const auto prepared = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(1), ordinary_target(wire::ExecutionOpcode::kUpload, 701));
    const auto permit = ordinary_begin(test, prepared.request);
    native.stop = false;
    expect(source.activate().status == CUDA_SUCCESS, "select exact synchronous source");
    const auto uploaded = source.upload({test.allocations[0], 0}, std::as_bytes(std::span{values}));
    expect(uploaded.status == CUDA_SUCCESS && uploaded.original_status == CUDA_SUCCESS &&
               source.owns_frontier(uploaded.frontier),
           "real synchronous call returns its private tail");
    expect(source.deactivate().status == CUDA_SUCCESS, "sync callback restores proven NULL");
    auto result = ordinary_reply();
    result.reply->original_status = 0;
    result.binding = NcclActorContinuationBinding{1, test.sources[0], uploaded.frontier, {}};
    expect(test.actor->finish_continuation_attempt(permit, std::move(result)) == CUDA_SUCCESS &&
               test.actor->release_continuation_reply(prepared.request) == CUDA_SUCCESS,
           "sync return and reply ACK retain private tail independently");
    expect(!ordinary_snapshot(test, prepared.request).is_retired && native.uploads == 1,
           "no duplicate synchronous call or invented GPU stop");
    close_ordinary_group(test);
    native.stop = true;
    for (std::size_t index = 0; index != values.size(); ++index)
        expect(native.memory[0][index] == values[index], "complete unchanged synchronous output");
    test.finish();
}

void ordinary_ordering_reply_before_issue() {
    Harness test;
    open_ordinary_group(test);
    auto& source = *test.sources[0];
    expect(source.activate().status == CUDA_SUCCESS, "select ordering resource owner");
    const auto event = source.create_event(0);
    expect(event.status == CUDA_SUCCESS && source.deactivate().status == CUDA_SUCCESS,
           "create genuine user event before ordinary handler");
    const auto prepared = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(1), ordinary_target(wire::ExecutionOpcode::kRecordEvent, 851));
    const auto permit = ordinary_begin(test, prepared.request);
    expect(source.activate().status == CUDA_SUCCESS, "select exact event admission owner");
    const auto accepted = source.record_event_with_frontier(1, event.id, 0);
    expect(accepted.result.status == CUDA_SUCCESS && accepted.result.id == 1 &&
               source.deactivate().status == CUDA_SUCCESS,
           "actual event acceptance returns target frontier without issuing it");
    auto returned = ordinary_reply(1);
    returned.binding = NcclActorContinuationBinding{1, test.sources[0], accepted.frontier, {}};
    const auto records = native.records;
    expect(test.actor->finish_continuation_attempt(permit, std::move(returned)) == CUDA_SUCCESS &&
               test.actor->release_continuation_reply(prepared.request) == CUDA_SUCCESS,
           "ordinary event reply release precedes actual source issue");
    const auto pending = ordinary_snapshot(test, prepared.request);
    expect(pending.stage == NcclActorContinuationStage::kWaitingRetirement &&
               pending.dependencies == std::vector<std::uint64_t>{1} && !pending.is_retired,
           "released RPC retains exact source progress authority until retirement");
    const auto inspection =
        test.actor->inspect_continuation_source(prepared.request, pending.grant, 1);
    expect(inspection.status == CUDA_SUCCESS &&
               inspection.binding.frontier.same_capture(accepted.frontier) &&
               native.records == records,
           "CPU inspection never issues the already acknowledged event command");
    auto stale = pending.grant;
    ++stale.cutoff;
    expect(test.actor->inspect_continuation_source(prepared.request, stale, 1).status ==
               CUDA_ERROR_INVALID_VALUE,
           "released request still requires the exact current stage");
    native.stop = false;
    expect(source.activate().status == CUDA_SUCCESS, "select granted retirement source");
    const auto issued = source.progress_frontier_retirement(inspection.binding.frontier);
    expect(issued.state == CudaExecutionIssueState::kIssued && issued.sequence == 1 &&
               source.deactivate().status == CUDA_SUCCESS,
           "one exact leaf performs actual native event recording");
    close_ordinary_group(test);
    expect(!test.actor->snapshot().reservations[0].has_physical_work &&
               !ordinary_snapshot(test, prepared.request).is_retired,
           "source owns live event tail; batch metadata does not claim another physical owner");
    native.stop = true;
    await([&] {
        test.step();
        return test.actor->snapshot().reservations.empty();
    });
    expect(source.activate().status == CUDA_SUCCESS &&
               source.destroy_event(event.id).status == CUDA_SUCCESS &&
               source.deactivate().status == CUDA_SUCCESS,
           "destroy actual user event only after exact ordering tail stops");
    test.finish();
}

void ordinary_synchronous_error_tail() {
    Harness test;
    open_ordinary_group(test);
    auto& source = *test.sources[0];
    const auto prepared = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(1), ordinary_target(wire::ExecutionOpcode::kUpload, 855));
    const auto permit = ordinary_begin(test, prepared.request);
    native.stop = false;
    native.upload_status = CUDA_ERROR_INVALID_VALUE;
    expect(source.activate().status == CUDA_SUCCESS, "select genuine original-error source");
    const std::array<float, 16> values{};
    const auto uploaded = source.upload({test.allocations[0], 0}, std::as_bytes(std::span{values}));
    expect(uploaded.status == CUDA_ERROR_INVALID_VALUE &&
               uploaded.original_status == CUDA_ERROR_INVALID_VALUE &&
               !source.snapshot().is_quarantined && source.owns_frontier(uploaded.frontier),
           "actual SDK error remains distinct from healthy private-tail proof");
    expect(source.deactivate().status == CUDA_SUCCESS, "restore NULL after actual SDK error");
    auto result = ordinary_reply();
    result.reply->status = uploaded.status;
    result.reply->original_status = uploaded.original_status;
    result.binding = NcclActorContinuationBinding{1, test.sources[0], uploaded.frontier, {}};
    expect(test.actor->finish_continuation_attempt(permit, std::move(result)) == CUDA_SUCCESS,
           "unchanged native error is not itself proof of tail retirement or uncertainty");
    const auto observed = test.actor->continuation_reply(prepared.request);
    expect(observed.ready && !observed.is_retired &&
               observed.reply->original_status == CUDA_ERROR_INVALID_VALUE &&
               !test.actor->snapshot().reservations[0].has_unknown_state,
           "retained exact error reply waits for real healthy completion evidence");
    expect(test.actor->release_continuation_reply(prepared.request) == CUDA_SUCCESS,
           "original error ACK leaves the private tail retained");
    close_ordinary_group(test);
    native.stop = true;
    test.finish();
    expect(native.uploads == 1, "native error is never replayed");
    for (std::size_t index = 0; index != values.size(); ++index)
        expect(native.memory[0][index] == static_cast<float>(index + 1),
               "full output remains unchanged on the fake SDK's actual no-write error");
}

void ordinary_context_free_metadata() {
    Harness test;
    open_ordinary_group(test);
    auto input = ordinary_scope(1);
    input.origin_source = 0;
    auto target = ordinary_target(wire::ExecutionOpcode::kPointerMetadata, 861);
    expect(test.actor->prepare_continuation({17, 0}, input, target).status ==
               CUDA_ERROR_INVALID_CONTEXT,
           "untrusted zero origin cannot authorize ordinary metadata");
    ++input.ordinary_sequence;
    target.is_context_free_helper = true;
    const auto prepared = test.actor->prepare_continuation({17, 0}, input, target);
    expect(prepared.status == CUDA_SUCCESS && !target.helper_owner,
           "trusted CPU metadata role needs no invented source or HostOwner");
    ordinary_return(test, prepared.request, ordinary_reply());
    ++input.ordinary_sequence;
    target.operation = wire::ExecutionOpcode::kImportHostMemory;
    expect(test.actor->prepare_continuation({17, 0}, input, target).status ==
               CUDA_ERROR_INVALID_HANDLE,
           "metadata role tag cannot substitute for import owner");
    close_ordinary_group(test);
    test.finish();
}

void ordinary_retired_source_parent() {
    Harness test;
    open_ordinary_group(test);
    auto target = ordinary_target(wire::ExecutionOpcode::kImportHostMemory, 871);
    target.helper_owner = std::make_shared<int>(71);
    const auto prepared = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(prepared.status == CUDA_SUCCESS, "source-bound helper import preparation");
    auto imported = ordinary_reply(971);
    imported.reply->original_status = 0;
    imported.imported_allocation = 971;
    ordinary_return(test, prepared.request, std::move(imported));
    close_ordinary_group(test);
    test.source_visible = false;  // Retired Slot no longer authenticates fresh source requests.
    auto cleanup = ordinary_scope(1);
    cleanup.flags = wire::kExecutionSourceContinuationCleanup;
    cleanup.cleanup_parent = 1;
    target.operation = wire::ExecutionOpcode::kReleaseHostMemory;
    target.cleanup_allocation = 971;
    const auto release = test.actor->prepare_continuation({17, 0}, cleanup, target);
    expect(release.status == CUDA_SUCCESS, "actual import parent retains retiring source identity");
    const auto cut = test.sources[0]->capture_frontier();
    expect(cut.status == CUDA_SUCCESS &&
               test.actor->bind_continuation_dependency(
                   release.request, {1, test.sources[0], cut.frontier, {}}) == CUDA_SUCCESS,
           "cleanup binds only its retained original Session despite resolver retirement");
    const auto inspected = test.actor->inspect_continuation_source(
        release.request, ordinary_snapshot(test, release.request).grant, 1);
    expect(inspected.status == CUDA_SUCCESS && inspected.dependency.status == CUDA_SUCCESS,
           "retained cleanup cut remains independently inspectable");
    auto result = ordinary_reply();
    result.reply->original_status = 0;
    ordinary_return(test, release.request, std::move(result));
    test.finish();
}

void ordinary_explicit_close_parent() {
    Harness test;
    open_ordinary_group(test);
    auto imported_target = ordinary_target(wire::ExecutionOpcode::kImportHostMemory, 881);
    imported_target.connection = 81;
    imported_target.helper_owner = std::make_shared<int>(81);
    const auto imported =
        test.actor->prepare_continuation({17, 0}, ordinary_scope(1), imported_target);
    expect(imported.status == CUDA_SUCCESS,
           "import parent before actual source context destruction");
    auto imported_reply = ordinary_reply(981);
    imported_reply.reply->original_status = 0;
    imported_reply.imported_allocation = 981;
    ordinary_return(test, imported.request, std::move(imported_reply));

    auto closing = ordinary_target(wire::ExecutionOpcode::kClose, 882);
    closing.closing_source = test.sources[0];
    auto invalid = closing;
    invalid.operation = wire::ExecutionOpcode::kGetContextFlags;
    expect(test.actor->prepare_continuation({17, 0}, ordinary_scope(2), invalid).status ==
               CUDA_ERROR_INVALID_VALUE,
           "Close lease cannot authorize another ordinary operation");
    invalid = closing;
    ++invalid.connection;
    expect(test.actor->prepare_continuation({17, 0}, ordinary_scope(2), invalid).status ==
               CUDA_ERROR_INVALID_VALUE,
           "Close lease cannot impersonate another target connection");
    test.source_visible = false;
    const auto prepared = test.actor->prepare_continuation({17, 0}, ordinary_scope(2), closing);
    expect(prepared.status == CUDA_SUCCESS, "exact own Close uses sealed target capability only");
    const auto cut = test.sources[0]->capture_frontier();
    expect(cut.status == CUDA_SUCCESS &&
               test.actor->bind_continuation_dependency(
                   prepared.request, {1, test.sources[0], cut.frontier, {}}) == CUDA_SUCCESS,
           "exact closing source permits its own immutable quiescence cut");
    const auto permit = ordinary_begin(test, prepared.request);
    expect(test.sources[0]->activate().status == CUDA_SUCCESS, "select actual Close context");
    const auto closed = test.sources[0]->shutdown();
    expect(closed.status == CUDA_SUCCESS && !current && !test.sources[0]->snapshot().has_context,
           "actual healthy context shutdown, not merely retained shared ownership");
    auto reply = ordinary_reply();
    reply.reply->original_status = closed.original_status;
    expect(test.actor->finish_continuation_attempt(permit, std::move(reply)) == CUDA_SUCCESS &&
               test.actor->release_continuation_reply(prepared.request) == CUDA_SUCCESS,
           "Close reply retains no fabricated live native context");
    // The helper parent retains the old Session identity but needs no new
    // native capture. The completed source is no longer an active poll owner.
    test.sources.clear();
    test.allocations.clear();
    close_ordinary_group(test);
    auto cleanup = ordinary_scope(1);
    cleanup.flags = wire::kExecutionSourceContinuationCleanup;
    cleanup.cleanup_parent = 1;
    imported_target.operation = wire::ExecutionOpcode::kReleaseHostMemory;
    imported_target.cleanup_allocation = 981;
    const auto released = test.actor->prepare_continuation({17, 0}, cleanup, imported_target);
    expect(released.status == CUDA_SUCCESS,
           "exact helper cleanup after End and real source death needs no context resurrection");
    auto released_reply = ordinary_reply();
    released_reply.reply->original_status = 0;
    ordinary_return(test, released.request, std::move(released_reply));
    test.finish();
}

void ordinary_lost_proof() {
    Harness test;
    open_ordinary_group(test);
    const auto prepared = test.actor->prepare_continuation(
        {17, 0}, ordinary_scope(1), ordinary_target(wire::ExecutionOpcode::kUpload, 801));
    const auto permit = ordinary_begin(test, prepared.request);
    auto& source = *test.sources[0];
    expect(source.activate().status == CUDA_SUCCESS, "select isolated proof-loss source");
    native.upload_loses_context = true;
    const std::array<float, 16> values{};
    const auto uploaded = source.upload({test.allocations[0], 0}, std::as_bytes(std::span{values}));
    expect(uploaded.status != CUDA_SUCCESS && uploaded.original_status == CUDA_SUCCESS &&
               source.snapshot().is_quarantined && source.owns_frontier(uploaded.frontier),
           "actual successful original upload followed by independent context-proof loss");
    auto result = ordinary_reply();
    result.kind = NcclActorContinuationOutcomeKind::kUncertain;
    result.status = uploaded.status;
    result.reply->disposition = wire::ExecutionWireDisposition::kRejected;
    result.reply->status = uploaded.status;
    result.reply->original_status = 0;
    result.reply->parameter_bytes = 1;
    result.binding = NcclActorContinuationBinding{1, test.sources[0], uploaded.frontier, {}};
    expect(test.actor->finish_continuation_attempt(permit, std::move(result)) == uploaded.status,
           "actual requested success survives stronger carrier fault");
    const auto observed = test.actor->continuation_reply(prepared.request);
    expect(observed.ready && !observed.is_retired && observed.reply->original_status == 0 &&
               test.actor->cancel_continuation(prepared.request) == CUDA_ERROR_NOT_READY,
           "unknown private tail is neither cancelled nor falsely retired");
    const auto record_count = native.records;
    for (unsigned attempt = 0; attempt != 8; ++attempt) (void)test.actor->observe();
    expect(native.uploads == 1 && native.records == record_count &&
               test.actor->snapshot().reservations[0].has_unknown_state,
           "unknown bookkeeping never replays SDK or invents stop");
    // Isolated intentional native-ownership retention; not a leak-free recovery.
}

void ordinary_source_fence() {
    Harness test;
    const auto comm = test.init_comm();
    const auto prepared = test.actor->prepare({17, 0}, test.request(test.collective(comm), 1));
    expect(prepared.status == CUDA_SUCCESS, "materialize CPU request before source fence");
    test.source_fence = CUDA_ERROR_NOT_READY;
    const auto records = native.records;
    const auto grant = test.grant(prepared.request);
    expect(test.actor->issue(prepared.request, grant) == CUDA_ERROR_NOT_READY &&
               native.records == records && native.calls[8] == 0 &&
               test.sources[0]->snapshot().enqueue_turns == 0,
           "source fence rejects capture before any turn or original SDK entry");
    test.source_fence = CUDA_SUCCESS;
    await([&] {
        test.step();
        return test.actor->reply(prepared.request).ready;
    });
    test.release(prepared.request);
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
}

NcclActorContinuationTarget cleanup_import_target(Harness& test, std::uint64_t connection) {
    test.cleanup_connection = connection;
    test.cleanup_helper = std::make_shared<int>(91);
    auto target = ordinary_target(wire::ExecutionOpcode::kImportHostMemory, 1001);
    target.connection = connection;
    target.helper_owner = test.cleanup_helper;
    return target;
}
void return_import(Harness& test, NcclActorPreparation prepared, std::uint64_t allocation) {
    auto reply = ordinary_reply(allocation);
    reply.reply->original_status = CUDA_SUCCESS;
    reply.imported_allocation = allocation;
    ordinary_return(test, prepared.request, std::move(reply));
}
void cleanup_retirement_refund() {
    NcclActorConfig config;
    config.maximum_cleanup_parents = 2;
    Harness test(true, 1, false, config, true);
    open_ordinary_group(test);
    auto target = cleanup_import_target(test, 91);
    for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
        const auto prepared =
            test.actor->prepare_continuation({17, 0}, ordinary_scope(sequence), target);
        expect(prepared.status == CUDA_SUCCESS, "two exact imports share one retained helper");
        return_import(test, prepared, 1000 + sequence);
    }
    close_ordinary_group(test);
    expect(
        test.actor->begin_cleanup_retirement({18, 0}, 91, target.helper_owner).status ==
                CUDA_ERROR_INVALID_HANDLE &&
            test.actor->begin_cleanup_retirement({17, 1}, 91, target.helper_owner).status ==
                CUDA_ERROR_INVALID_HANDLE &&
            test.actor->begin_cleanup_retirement({17, 0}, 92, target.helper_owner).status ==
                CUDA_ERROR_INVALID_HANDLE &&
            test.actor->begin_cleanup_retirement({17, 0}, 91, std::make_shared<int>(91)).status ==
                CUDA_ERROR_INVALID_HANDLE,
        "peer, priority, connection and actual helper identity are independent authority");
    const auto sealed = test.actor->begin_cleanup_retirement({17, 0}, 91, target.helper_owner);
    expect(sealed.status == CUDA_SUCCESS && sealed.ticket.identity && sealed.ticket.parents == 2 &&
               !test.cleanup_proofs,
           "begin seals exact two-parent set without asking for native shutdown proof");
    expect(test.actor->begin_cleanup_retirement({17, 0}, 91, target.helper_owner).status ==
               CUDA_ERROR_INVALID_VALUE,
           "duplicate seal cannot create another retirement owner");
    auto wrong = sealed.ticket;
    --wrong.parents;
    expect(test.actor->complete_cleanup_retirement(wrong) == CUDA_ERROR_INVALID_VALUE &&
               test.actor->complete_cleanup_retirement({}) == CUDA_ERROR_INVALID_VALUE &&
               !test.cleanup_proofs,
           "count mutation and empty tickets never invoke proof callback");
    {
        Harness other(true, 0);
        expect(
            other.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_INVALID_VALUE &&
                other.actor->close() == CUDA_SUCCESS,
            "another actor cannot consume a foreign retirement ticket");
        await([&] { return other.actor->shutdown() == CUDA_SUCCESS; });
    }
    CUresult foreign_status{};
    std::thread foreign(
        [&] { foreign_status = test.actor->complete_cleanup_retirement(sealed.ticket); });
    foreign.join();
    expect(foreign_status == CUDA_ERROR_INVALID_CONTEXT, "retirement keeps owner-thread identity");
    const auto child = ::fork();
    expect(child >= 0, "fork retirement authority control");
    if (!child)
        std::_Exit(test.actor->complete_cleanup_retirement(sealed.ticket) ==
                           CUDA_ERROR_INVALID_CONTEXT
                       ? 0
                       : 1);
    int child_status{};
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "fork cannot consume inherited parent authority");
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_NOT_READY &&
               test.cleanup_proofs == 1,
           "unfinished actual helper retains every parent and ticket");
    test.cleanup_status = CUDA_SUCCESS;
    test.cleanup_reentry = true;
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_SUCCESS &&
               test.cleanup_proofs == 2 &&
               test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_INVALID_VALUE,
           "actual proof consumes exact set once, independently from lost delivery");
    open_ordinary_group(test, 78);
    auto next_scope = ordinary_scope(3);
    next_scope.group = 78;
    target = cleanup_import_target(test, 92);
    const auto reused = test.actor->prepare_continuation({17, 0}, next_scope, target);
    expect(reused.status == CUDA_SUCCESS, "retired helper refunds bounded parent capacity");
    return_import(test, reused, 1003);
    close_ordinary_group(test);
    const auto next = test.actor->begin_cleanup_retirement({17, 0}, 92, target.helper_owner);
    expect(next.status == CUDA_SUCCESS && next.ticket.parents == 1 &&
               next.ticket.identity > sealed.ticket.identity,
           "refunded metadata never reuses retired ticket identity");
    expect(test.actor->complete_cleanup_retirement(next.ticket) == CUDA_SUCCESS,
           "second independently authenticated owner also retires once");
    test.finish();
}
void cleanup_retirement_active_and_sealed() {
    Harness test(true, 1, false, {}, true);
    open_ordinary_group(test);
    auto target = cleanup_import_target(test, 93);
    const auto imported = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(imported.status == CUDA_SUCCESS &&
               test.actor->begin_cleanup_retirement({17, 0}, 93, target.helper_owner).status ==
                   CUDA_ERROR_NOT_READY,
           "unbound reserved parent cannot be reclaimed");
    const auto permit = ordinary_begin(test, imported.request);
    expect(test.actor->begin_cleanup_retirement({17, 0}, 93, target.helper_owner).status ==
               CUDA_ERROR_NOT_READY,
           "armed handler excludes retirement before its result is classified");
    NcclActorContinuationOutcome pending;
    pending.kind = NcclActorContinuationOutcomeKind::kHostJobPending;
    pending.status = CUDA_SUCCESS;
    pending.pending_lifetime = target.helper_owner;
    pending.pending_sequence = target.wire_sequence;
    expect(test.actor->finish_continuation_attempt(permit, pending) == CUDA_SUCCESS &&
               test.actor->begin_cleanup_retirement({17, 0}, 93, target.helper_owner).status ==
                   CUDA_ERROR_NOT_READY,
           "in-flight helper mailbox is retained even if client disappears");
    auto returned = ordinary_reply(1004);
    returned.reply->original_status = CUDA_SUCCESS;
    returned.imported_allocation = 1004;
    expect(
        test.actor->complete_continuation(imported.request, std::move(returned)) == CUDA_SUCCESS &&
            test.actor->begin_cleanup_retirement({17, 0}, 93, target.helper_owner).status ==
                CUDA_ERROR_NOT_READY,
        "known import is bound before abandoned reply is released");
    expect(test.actor->release_continuation_reply(imported.request) == CUDA_SUCCESS,
           "consume abandoned known result without claiming native cleanup");
    auto cleanup_scope = ordinary_scope(1);
    cleanup_scope.flags = wire::kExecutionSourceContinuationCleanup;
    cleanup_scope.cleanup_parent = 1;
    auto cleanup = target;
    cleanup.operation = wire::ExecutionOpcode::kReleaseHostMemory;
    cleanup.cleanup_allocation = 1004;
    const auto active = test.actor->prepare_continuation({17, 0}, cleanup_scope, cleanup);
    expect(active.status == CUDA_SUCCESS &&
               test.actor->begin_cleanup_retirement({17, 0}, 93, target.helper_owner).status ==
                   CUDA_ERROR_NOT_READY,
           "a live cleanup continuation cannot race wholesale helper retirement");
    expect(test.actor->cancel_continuation(active.request) == CUDA_SUCCESS,
           "cancel only the known unentered cleanup attempt");
    const auto sealed = test.actor->begin_cleanup_retirement({17, 0}, 93, target.helper_owner);
    expect(sealed.status == CUDA_SUCCESS, "inactive known parent is now sealable");
    expect(test.actor->prepare_continuation({17, 0}, ordinary_scope(2), target).status ==
                   CUDA_ERROR_DEINITIALIZED &&
               test.actor->prepare_continuation({17, 0}, ordinary_scope(2), target).status ==
                   CUDA_ERROR_INVALID_VALUE,
           "sealed helper rejects new import while consuming its canonical dense ordinal");
    cleanup_scope.ordinary_sequence = 2;
    expect(test.actor->prepare_continuation({17, 0}, cleanup_scope, cleanup).status ==
                   CUDA_ERROR_DEINITIALIZED &&
               test.actor->prepare_continuation({17, 0}, cleanup_scope, cleanup).status ==
                   CUDA_ERROR_INVALID_VALUE,
           "sealed helper rejects parent cleanup without replaying its independent sequence");
    test.cleanup_status = CUDA_SUCCESS;
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_SUCCESS,
           "healthy closed helper settles parent but not native group");
    const auto still_open = test.actor->snapshot();
    expect(still_open.reservations.size() == 1 && still_open.reservations[0].has_open_group &&
               still_open.reservations[0].has_physical_work,
           "parent cleanup does not manufacture GroupEnd or physical stop");
    close_ordinary_group(test);
    test.finish();
}
void cleanup_retirement_named_close() {
    Harness test(true, 1, false, {}, true);
    open_ordinary_group(test);
    auto target = cleanup_import_target(test, 94);
    const auto imported = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(imported.status == CUDA_SUCCESS, "known parent for named terminal Close");
    return_import(test, imported, 1005);
    target.operation = wire::ExecutionOpcode::kGetContextFlags;
    const auto wrong = test.actor->prepare_continuation({17, 0}, ordinary_scope(2), target);
    expect(wrong.status == CUDA_SUCCESS &&
               test.actor->begin_cleanup_retirement({17, 0}, 94, target.helper_owner, wrong.request)
                       .status == CUDA_ERROR_INVALID_HANDLE,
           "naming an arbitrary continuation cannot bypass the active-owner gate");
    expect(test.actor->cancel_continuation(wrong.request) == CUDA_SUCCESS,
           "discard unentered non-Close control");
    target.operation = wire::ExecutionOpcode::kClose;
    const auto closing = test.actor->prepare_continuation({17, 0}, ordinary_scope(3), target);
    expect(closing.status == CUDA_SUCCESS, "accept exact normal helper Close");
    auto& source = *test.sources[0];
    expect(source.activate().status == CUDA_SUCCESS, "select real dependency source");
    const auto event = source.create_event(0);
    const auto queued = source.record_event_with_frontier(1, event.id, 0);
    expect(event.status == CUDA_SUCCESS && queued.result.status == CUDA_SUCCESS &&
               source.deactivate().status == CUDA_SUCCESS &&
               test.actor->bind_continuation_dependency(
                   closing.request, {1, test.sources[0], queued.frontier, {}}) == CUDA_SUCCESS,
           "terminal Close retains exact not-yet-issued pre-call cut");
    expect(test.actor->begin_cleanup_retirement({17, 0}, 94, target.helper_owner, closing.request)
                       .status == CUDA_ERROR_NOT_READY &&
               native.records == 0,
           "named exception does not issue its unresolved source dependencies");
    expect(source.activate().status == CUDA_SUCCESS &&
               source.progress_frontier_retirement(queued.frontier).state ==
                   CudaExecutionIssueState::kIssued &&
               source.deactivate().status == CUDA_SUCCESS,
           "separately authorized exact dependency performs real source work");
    test.poll();
    auto permit = ordinary_begin(test, closing.request);
    NcclActorContinuationOutcome deferred;
    deferred.kind = NcclActorContinuationOutcomeKind::kDeferred;
    deferred.status = CUDA_ERROR_NOT_READY;
    expect(test.actor->finish_continuation_attempt(permit, deferred) == CUDA_SUCCESS,
           "proven no-entry Deferred keeps named Close safely Blocked");
    const auto sealed =
        test.actor->begin_cleanup_retirement({17, 0}, 94, target.helper_owner, closing.request);
    expect(sealed.status == CUDA_SUCCESS &&
               test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_NOT_READY &&
               !test.cleanup_proofs,
           "Blocked Close can seal, but must independently complete before shutdown proof");
    auto stale = ordinary_snapshot(test, closing.request).grant;
    ++stale.cutoff;
    expect(
        test.actor->begin_continuation(closing.request, stale).status == CUDA_ERROR_INVALID_VALUE,
        "sealing does not replace the exact ordinary Core grant");
    permit = ordinary_begin(test, closing.request);
    auto closed = ordinary_reply();
    closed.reply->original_status = CUDA_SUCCESS;
    expect(test.actor->finish_continuation_attempt(permit, closed) == CUDA_SUCCESS,
           "actual cached Close result independently completes the named continuation");
    test.cleanup_status = CUDA_SUCCESS;
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_SUCCESS &&
               test.cleanup_proofs == 1 &&
               test.actor->release_continuation_reply(closing.request) == CUDA_SUCCESS,
           "real closed proof consumes parents before reply ACK, without a self-deadlock");
    close_ordinary_group(test);
    expect(source.activate().status == CUDA_SUCCESS &&
               source.destroy_event(event.id).status == CUDA_SUCCESS &&
               source.deactivate().status == CUDA_SUCCESS,
           "exact user event cleanup after actual retirement");
    test.finish();
}
void cleanup_retirement_after_admission_close() {
    Harness test(true, 1, false, {}, true);
    open_ordinary_group(test);
    const auto target = cleanup_import_target(test, 95);
    const auto imported = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(imported.status == CUDA_SUCCESS, "known parent survives admission closure");
    return_import(test, imported, 1006);
    close_ordinary_group(test);
    expect(test.sources[0]->activate().status == CUDA_SUCCESS &&
               test.sources[0]->shutdown().status == CUDA_SUCCESS && !current,
           "successful source Close does not invalidate the helper's retained parent identity");
    test.sources.clear();
    test.allocations.clear();
    test.source_visible = false;
    expect(test.actor->disconnect(17) == CUDA_SUCCESS && test.actor->close() == CUDA_SUCCESS,
           "peer and actor admission close do not invalidate accepted cleanup ownership");
    const auto sealed = test.actor->begin_cleanup_retirement({17, 0}, 95, target.helper_owner);
    expect(sealed.status == CUDA_SUCCESS, "begin still authenticates retained exact parent");
    test.cleanup_status = CUDA_SUCCESS;
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_SUCCESS,
           "cleanup uses retained origin, not a resurrected source or peer admission");
    test.finish();
}
void cleanup_retirement_cancelled_close() {
    Harness test(true, 1, false, {}, true);
    open_ordinary_group(test);
    auto target = cleanup_import_target(test, 98);
    const auto imported = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(imported.status == CUDA_SUCCESS, "known parent before terminal Close disconnect");
    return_import(test, imported, 1009);
    target.operation = wire::ExecutionOpcode::kClose;
    const auto closing = test.actor->prepare_continuation({17, 0}, ordinary_scope(2), target);
    expect(closing.status == CUDA_SUCCESS, "accept named terminal helper Close");
    const auto sealed =
        test.actor->begin_cleanup_retirement({17, 0}, 98, target.helper_owner, closing.request);
    expect(sealed.status == CUDA_SUCCESS, "seal before actual Close attempt");
    const auto permit = ordinary_begin(test, closing.request);
    NcclActorContinuationOutcome pending;
    pending.kind = NcclActorContinuationOutcomeKind::kHostJobPending;
    pending.status = CUDA_SUCCESS;
    pending.pending_lifetime = target.helper_owner;
    pending.pending_sequence = target.wire_sequence;
    expect(test.actor->finish_continuation_attempt(permit, pending) == CUDA_SUCCESS &&
               test.actor->cancel_continuation(closing.request) == CUDA_ERROR_NOT_READY &&
               test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_NOT_READY &&
               !test.cleanup_proofs,
           "entered shutdown mailbox is not cancelled or mistaken for independent completion");
    NcclActorContinuationOutcome deferred;
    deferred.kind = NcclActorContinuationOutcomeKind::kDeferred;
    deferred.status = CUDA_ERROR_NOT_READY;
    expect(test.actor->complete_continuation(closing.request, deferred) == CUDA_SUCCESS &&
               test.actor->cancel_continuation(closing.request) == CUDA_SUCCESS,
           "only actual consumed no-entry Deferred permits exact abandoned Close cancellation");
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_NOT_READY &&
               test.cleanup_proofs == 1,
           "cancelled/released Close still needs actual helper shutdown, never ACK as proof");
    close_ordinary_group(test);
    expect(test.actor->disconnect(17) == CUDA_SUCCESS && test.actor->close() == CUDA_SUCCESS,
           "exact cleanup remains possible after caller admission ends");
    test.cleanup_status = CUDA_SUCCESS;
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_SUCCESS &&
               test.cleanup_proofs == 2,
           "ordinary authorized shutdown proof completes the same cancelled named Close ticket");
    test.finish();
}
void cleanup_retirement_missing_proof() {
    Harness test;
    open_ordinary_group(test);
    const auto target = cleanup_import_target(test, 96);
    const auto imported = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(imported.status == CUDA_SUCCESS, "known parent without installed proof producer");
    return_import(test, imported, 1007);
    close_ordinary_group(test);
    const auto sealed = test.actor->begin_cleanup_retirement({17, 0}, 96, target.helper_owner);
    expect(sealed.status == CUDA_SUCCESS &&
               test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_NOT_SUPPORTED &&
               test.actor->close() == CUDA_SUCCESS &&
               test.actor->shutdown() == CUDA_ERROR_NOT_READY,
           "missing actual helper proof retains parents; no default fabricated success");
}
void cleanup_retirement_uncertain_proof() {
    Harness test(true, 1, false, {}, true);
    open_ordinary_group(test);
    const auto target = cleanup_import_target(test, 97);
    const auto imported = test.actor->prepare_continuation({17, 0}, ordinary_scope(1), target);
    expect(imported.status == CUDA_SUCCESS, "known parent before actual cleanup uncertainty");
    return_import(test, imported, 1008);
    close_ordinary_group(test);
    const auto sealed = test.actor->begin_cleanup_retirement({17, 0}, 97, target.helper_owner);
    expect(sealed.status == CUDA_SUCCESS, "reserve before uncertainty appears");
    test.cleanup_status = CUDA_ERROR_UNKNOWN;
    expect(test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_UNKNOWN &&
               test.cleanup_proofs == 1,
           "unexpected native shutdown proof failure retains and marks exact parents uncertain");
    test.cleanup_status = CUDA_SUCCESS;
    expect(
        test.actor->complete_cleanup_retirement(sealed.ticket) == CUDA_ERROR_UNKNOWN &&
            test.cleanup_proofs == 1 && test.actor->snapshot().reservations[0].has_unknown_state &&
            test.actor->close() == CUDA_SUCCESS && test.actor->shutdown() == CUDA_ERROR_NOT_READY,
        "unknown cleanup is sticky, never retried or refunded after a later guessed success");
}

AuditTrace audited(void (*scenario)(), bool stopped = true) {
    AuditTrace trace;
    active_audit = &trace;
    scenario();
    active_audit = nullptr;
    trace.validate(stopped);
    return trace;
}
void audit_configuration() {
    AuditTrace trace;
    Harness test;
    trace.actor = test.actor.get();
    expect(test.actor->set_observer({}) && !test.actor->set_observer({&trace, nullptr}),
           "disabled default is cold and rejects a dangling context-only observer");
    bool foreign{};
    std::thread thread([&] { foreign = test.actor->set_observer(trace.observer()); });
    thread.join();
    expect(!foreign && !trace.size, "foreign thread cannot install or emit diagnostics");
    const auto child = ::fork();
    expect(child >= 0, "fork diagnostic installation control");
    if (!child) std::_Exit(test.actor->set_observer(trace.observer()) ? 1 : 0);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork installation rejects before inherited mutable state");
    expect(test.actor->set_observer(trace.observer()) &&
               !test.actor->set_observer(trace.observer()) && !test.actor->set_observer({}),
           "one cold observer cannot be replaced or disabled mid-lifecycle");
    const auto prepared =
        test.actor->prepare({17, 0}, test.request(wire::ExecutionNcclGetUniqueId{true}));
    expect(prepared.status == CUDA_SUCCESS, "diagnostic controls leave admission unchanged");
    const auto count = trace.size;
    for (unsigned iteration = 0; iteration != 8; ++iteration) {
        expect(
            test.actor->snapshot().status == CUDA_SUCCESS && test.actor->observe() == CUDA_SUCCESS,
            "unchanged CPU observation remains healthy");
    }
    expect(trace.size == count && native.calls[0] == 0 && native.primary_retains == 0,
           "unchanged observation emits nothing and grants no native work");
    await([&] {
        test.step();
        return test.actor->reply(prepared.request).ready;
    });
    test.release(prepared.request);
    test.finish();
    trace.validate();
    expect(native.calls[0] == 1, "diagnostic callback never replays original work");
}
void audit_native_routes() {
    const auto trace = audited(all_routes);
    for (std::uint32_t operation = 1; operation <= 11; ++operation)
        expect(trace.contains([&](const auto& row) {
            return row.event == NcclActorAuditEvent::kNativeSubmit &&
                   row.native_kind == operation && row.native_operation;
        }),
               "all eleven real submissions have typed coordinator identities");
    std::array<std::uint64_t, 2> objects{};
    std::uint64_t context{};
    for (std::size_t index = 0; index != trace.size; ++index) {
        const auto& row = trace.records[index];
        if (row.event != NcclActorAuditEvent::kNativeSubmit) continue;
        if (row.native_kind == 2 || row.native_kind == 3) {
            expect(row.communicator_domain == 1 && row.communicator_context &&
                       row.communicator_context != 301 &&
                       row.communicator_object == row.allocation && row.allocation,
                   "Init acceptance names its actual reserved object and retained primary context");
            objects[row.native_kind - 2] = row.communicator_object;
            if (!context) context = row.communicator_context;
            expect(context == row.communicator_context,
                   "both initializers share the retained domain");
        } else if (row.native_kind >= 4 && row.native_kind != 7 && row.native_kind != 8) {
            const auto expected = objects[row.native_kind == 6 ? 1 : 0];
            expect(expected && row.communicator_object == expected &&
                       row.communicator_domain == 1 && row.communicator_context == context,
                   "query, collectives and destruction preserve their exact input communicator");
        }
    }
    expect(objects[0] && objects[1] && objects[0] != objects[1],
           "separate accepted initializers never alias symbolic communicator objects");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kObservation && row.has_async &&
               row.has_original && row.original_status == 0 && row.async_status == 0 &&
               row.output_contract == 1 && !row.has_output_written;
    }),
           "deferred async snapshot presence is not an affirmative output-write proof");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kSource && row.capture && row.source &&
               row.action == NcclActorAuditAction::kBegin;
    }),
           "actual accepted source turn emits a real immutable capture join");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kObservation && row.native_kind == 2 &&
               row.has_context && row.before_kind == 1 && !row.before_identity &&
               !row.before_context && !row.before_device && row.after_kind == 3 &&
               row.after_identity == 1 && row.after_context && row.after_device == 1;
    }),
           "actual NULL-to-primary transition contains symbolic IDs and no native pointers");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kObservation && row.native_kind == 4 &&
               row.origin_source == 1 && row.caller_primary_domain == 1 && row.has_context &&
               row.before_kind == 2 && row.before_identity == 1 && row.before_context &&
               row.before_context != 301 && row.before_device == 1;
    }),
           "observed primary caller selection stays distinct from an existing source session");
}
void audit_group_members() {
    const auto trace = audited(grouping);
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kGroup && row.depth == 2 && row.index == 1;
    }) && trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kGroup && row.depth == 0 && row.index == 1;
    }),
           "actual nested group changes preserve exact old and new depth");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kBinding &&
               row.binding == NcclActorAuditBinding::kGroupMember && row.native_kind == 8 &&
               row.related != row.id && row.count == 2;
    }),
           "outer End explicitly joins both retained grouped member identities");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kReply && row.native_kind == 9 &&
               row.has_original && !row.physical_complete && row.open_group;
    }),
           "grouped inner host reply is never reported as physical completion");
}
void audit_original_errors() {
    const auto trace = audited(errors_and_identity);
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kAdmission && !row.accepted &&
               row.sequence_consumed && !row.native_sequence && !row.native_operation;
    }),
           "clean rejected public ordinal does not invent a native operation");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kObservation && row.has_original &&
               row.original_status == 4 && row.carrier == CUDA_SUCCESS;
    }),
           "real native error remains independent from healthy carrier proof");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kAdmission && row.caller_primary_domain == 1 &&
               !row.accepted && !row.has_context && !row.native_operation &&
               row.status == CUDA_ERROR_INVALID_CONTEXT;
    }),
           "an unobserved primary claim is diagnostic input, never context authority");
}
enum class AuditContextCall { kUniqueId, kCommunicator };
void audit_context_failure(bool failure_after_observation,
                           AuditContextCall operation = AuditContextCall::kUniqueId) {
    const bool communicator_call = operation == AuditContextCall::kCommunicator;
    AuditTrace trace;
    active_audit = &trace;
    Harness test(true, 1, communicator_call);
    active_audit = nullptr;
    const auto comm = communicator_call ? test.init_comm() : wire::ExecutionNcclCommunicator{};
    native.result = 4;
    native.fail_worker_restore = failure_after_observation;
    native.fail_worker_observation = !failure_after_observation;
    wire::ExecutionNcclCall call = wire::ExecutionNcclGetUniqueId{true};
    if (communicator_call) call = wire::ExecutionNcclGetAsyncError{comm, true};
    const auto prepared = test.actor->prepare({17, 0}, test.request(std::move(call), 1));
    expect(prepared.status == CUDA_SUCCESS &&
               test.actor->issue(prepared.request, test.grant(prepared.request)) == CUDA_SUCCESS,
           "one accepted original call before independent context proof failure");
    await([&] {
        expect(test.actor->observe() == CUDA_SUCCESS, "observe actual failed carrier receipt");
        const auto observed = test.actor->reply(prepared.request);
        return observed.native && observed.native->reply_ready;
    });
    const auto result = test.actor->reply(prepared.request);
    expect(result.native && result.native->original_status == 4 &&
               result.native->carrier != CUDA_SUCCESS && !result.native->physical_complete &&
               result.native->context_transition.has_value() == failure_after_observation,
           "actual original status survives failed restoration; missing observation stays absent");
    trace.validate(false);
    expect(trace.contains([&](const auto& row) {
        if (row.event != NcclActorAuditEvent::kObservation || row.id != prepared.request ||
            !row.has_original || row.original_status != 4 || row.carrier == CUDA_SUCCESS)
            return false;
        if (!failure_after_observation) return !row.has_context;
        return row.has_context && row.before_kind == 2 && row.before_identity == 1 &&
               row.before_context == 301 && row.before_device == 1 && row.after_kind == 2 &&
               row.after_identity == 1 && row.after_context == 301 && row.after_device == 1;
    }),
           "audit preserves only independently observed symbolic transition through later failure");
    if (communicator_call)
        expect(trace.contains([&](const auto& row) {
            return row.event == NcclActorAuditEvent::kObservation && row.id == prepared.request &&
                   row.communicator_domain == comm.domain &&
                   row.communicator_object == comm.object && row.communicator_context &&
                   row.communicator_context != 301 && row.output_contract == 2 && row.has_async &&
                   row.has_output_written && row.output_written && row.has_original &&
                   row.original_status == 4 && row.carrier != CUDA_SUCCESS &&
                   !row.physical_complete;
        }),
               "accepted identity and actual pinned write survive later carrier proof failure");
    expect(native.calls[communicator_call ? 3 : 0] == 1 &&
               !test.actor->snapshot().reservations.empty(),
           "failed carrier is retained without original replay or fabricated physical stop");
    // Each caller is an isolated intentional-retention child.
}
void audit_context_restore_failure() {
    audit_context_failure(true);
}
void audit_context_observation_failure() {
    audit_context_failure(false);
}
void audit_communicator_restore_failure() {
    audit_context_failure(true, AuditContextCall::kCommunicator);
}
void audit_async_states() {
    AuditTrace trace;
    active_audit = &trace;
    Harness test;
    active_audit = nullptr;
    const auto comm = test.init_comm();
    test.group = 71;
    auto [start_id, started] = test.run(wire::ExecutionNcclGroupStart{});
    test.release(start_id);
    auto [inner_id, inner] = test.run(test.collective(comm), 1);
    test.release(inner_id);
    native.result = 7;
    auto ended_call = test.run(wire::ExecutionNcclGroupEnd{});
    const auto end_id = ended_call.first;
    const auto& ended = ended_call.second;
    test.group = 0;
    expect(ended.envelope->original_status == 7 && !ended.native->physical_complete,
           "actual End7 retains asynchronous group ownership");
    test.release(end_id);
    native.result = 0;
    native.async_result = 7;
    auto [pending_id, pending] = test.run(wire::ExecutionNcclGetAsyncError{comm, true});
    expect(pending.envelope->original_status == 0 && pending.envelope->observation.async_error == 7,
           "original query success and async7 are independent native domains");
    test.release(pending_id);
    expect(!test.actor->reply(end_id).native->physical_complete,
           "query7 cannot retire the original End7 batch");
    native.async_result = 0;
    native.stop = false;
    auto [ready_id, ready] = test.run(wire::ExecutionNcclGetAsyncError{comm, true});
    test.release(ready_id);
    await([&] {
        test.step();
        const auto observed = test.actor->reply(end_id);
        return observed.native && observed.native->enqueue_complete &&
               !observed.native->physical_complete;
    });
    native.stop = true;
    await([&] {
        test.step();
        return test.actor->snapshot().reservations.empty();
    });
    auto [destroy_id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
    test.release(destroy_id);
    test.finish();
    trace.validate();
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kObservation && row.native_kind == 8 &&
               row.has_original && row.original_status == 7 && !row.physical_complete;
    }) && trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kObservation && row.native_kind == 4 &&
               row.has_original && row.original_status == 0 && row.has_async &&
               row.async_status == 7;
    }),
           "typed audit cannot conflate async status with original success or physical stop");
}
void audit_null_communicators() {
    AuditTrace trace;
    active_audit = &trace;
    Harness test(true, 1, true);
    active_audit = nullptr;
    native.result = 4;
    auto [init_id, initialized] = test.run(wire::ExecutionNcclInitRank{false, 1, 0, {}});
    expect(initialized.envelope->original_status == 4 &&
               !initialized.envelope->reserved_communicator.object,
           "omitted Init output has an actual native error but no reserved object");
    test.release(init_id);
    const auto query_call = test.run(wire::ExecutionNcclGetAsyncError{{}, true});
    const auto query_id = query_call.first;
    const auto& queried = query_call.second;
    expect(queried.envelope->original_status == 4 && queried.envelope->observation.effects == 4,
           "native NULL communicator error may still produce an observed scalar write");
    test.release(query_id);
    const auto rejected = test.actor->prepare(
        {17, 0}, test.request(wire::ExecutionNcclGetAsyncError{{1, 999}, true}));
    expect(
        rejected.status == CUDA_ERROR_INVALID_HANDLE && !rejected.request && native.calls[3] == 1,
        "unowned symbolic input rejects before coordinator acceptance and native entry");
    native.result = 0;
    test.finish();
    trace.validate();
    expect(!trace.contains([](const auto& row) { return row.communicator_object != 0; }),
           "NULL inputs and rejected claims never fabricate accepted communicator identity");
    expect(trace.contains([&](const auto& row) {
        return row.event == NcclActorAuditEvent::kObservation && row.id == query_id &&
               row.output_contract == 2 && row.has_async && row.has_output_written &&
               row.output_written && row.original_status == 4;
    }),
           "affirmative scalar-write evidence does not create a communicator or native success");
}
void audit_multiple_communicators() {
    AuditTrace trace;
    active_audit = &trace;
    Harness test(true, 1, true);
    active_audit = nullptr;
    const auto first = test.init_comm();
    const auto second = test.init_comm(true);
    test.group = 73;
    auto [start_id, started] = test.run(wire::ExecutionNcclGroupStart{});
    test.release(start_id);
    auto [first_id, first_result] = test.run(test.collective(first), 1);
    test.release(first_id);
    auto [second_id, second_result] = test.run(test.collective(second), 1);
    test.release(second_id);
    expect(
        first_result.envelope->original_status == 0 && second_result.envelope->original_status == 0,
        "both distinct communicator members really return success inside the open group");
    native.result = 7;
    const auto end_call = test.run(wire::ExecutionNcclGroupEnd{});
    const auto end_id = end_call.first;
    const auto& ended = end_call.second;
    test.group = 0;
    expect(ended.envelope->original_status == 7 && !ended.native->physical_complete,
           "outer End7 retains both otherwise successful participants");
    test.release(end_id);
    native.result = 0;
    native.stop = false;
    auto [first_query, first_ready] = test.run(wire::ExecutionNcclGetAsyncError{first, true});
    test.release(first_query);
    for (unsigned iteration = 0; iteration != 8; ++iteration) test.step();
    const auto only_first_ready = test.actor->reply(end_id);
    expect(first_ready.envelope->original_status == 0 && only_first_ready.native &&
               !only_first_ready.native->enqueue_complete &&
               !only_first_ready.native->physical_complete,
           "one actual communicator query cannot discharge the other's End7 obligation");
    native.async_result = 7;
    auto [pending_query, second_pending] = test.run(wire::ExecutionNcclGetAsyncError{second, true});
    test.release(pending_query);
    expect(second_pending.envelope->observation.async_error == 7 &&
               !test.actor->reply(end_id).native->enqueue_complete,
           "the second communicator's observed async7 remains pending");
    native.async_result = 0;
    auto [second_query, second_ready] = test.run(wire::ExecutionNcclGetAsyncError{second, true});
    test.release(second_query);
    await([&] {
        test.step();
        const auto observed = test.actor->reply(end_id);
        return observed.native && observed.native->enqueue_complete &&
               !observed.native->physical_complete;
    });
    expect(second_ready.envelope->observation.async_error == 0,
           "both communicator readiness results still do not prove held source tails");
    native.stop = true;
    await([&] {
        test.step();
        return test.actor->snapshot().reservations.empty();
    });
    for (const auto comm : {first, second}) {
        auto [id, destroyed] = test.run(wire::ExecutionNcclDestroy{comm});
        test.release(id);
    }
    test.finish();
    trace.validate();
    for (const auto member : {first_id, second_id})
        expect(trace.contains([&](const auto& row) {
            return row.event == NcclActorAuditEvent::kBinding && row.id == end_id &&
                   row.binding == NcclActorAuditBinding::kGroupMember && row.related == member &&
                   row.count == 2 && !row.communicator_object;
        }),
               "End identifies every exact member without inventing one aggregate communicator");
    for (const auto query : {first_query, pending_query, second_query}) {
        const auto expected = query == first_query ? first : second;
        expect(trace.contains([&](const auto& row) {
            return row.event == NcclActorAuditEvent::kObservation && row.id == query &&
                   row.communicator_domain == expected.domain &&
                   row.communicator_object == expected.object && row.communicator_context &&
                   row.output_contract == 2 && row.has_output_written && row.output_written &&
                   row.has_async && row.has_original && row.original_status == 0 && row.has_context;
        }),
               "every actual query exposes only its exact accepted communicator and pinned write");
    }
    expect(trace.contains([&](const auto& row) {
        return row.event == NcclActorAuditEvent::kRetirement && row.id == end_id &&
               row.action == NcclActorAuditAction::kRemove && row.physical_complete;
    }),
           "complete multi-communicator readiness retains its independent physical retirement");
}
void audit_physical_retention() {
    const auto trace = audited(retained_tail);
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kRelease && row.native_kind == 9 &&
               row.reply_released && !row.physical_complete;
    }) && trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kBatch && row.has_physical_work;
    }),
           "host ACK cannot erase held native physical ownership");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kRetirement &&
               row.action == NcclActorAuditAction::kRemove && row.native_kind == 9 &&
               row.physical_complete;
    }),
           "request removal follows explicit native physical proof");
}
void audit_source_progress() {
    const auto trace = audited(source_dependency_inspection);
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kGrantEnd &&
               row.action == NcclActorAuditAction::kObserve && row.source == 2 &&
               row.command_sequence == 1 && row.outcome == 2 && row.native_status == CUDA_SUCCESS &&
               row.completion_status == CUDA_SUCCESS;
    }),
           "exact source progress carries actual ordinary issue and original root grant");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kGrantEnd &&
               row.action == NcclActorAuditAction::kReject &&
               row.status == CUDA_ERROR_INVALID_VALUE;
    }),
           "failed rejection authorization is not silently converted to a clean rejection");
}
void audit_ordinary_lifetime() {
    const auto trace = audited(ordinary_command_and_retirement);
    std::uint64_t id{}, target_capture{};
    for (std::size_t index = 0; index != trace.size; ++index) {
        const auto& row = trace.records[index];
        if (row.subject == NcclActorAuditSubject::kContinuation &&
            row.event == NcclActorAuditEvent::kBinding &&
            row.binding == NcclActorAuditBinding::kTarget) {
            id = row.id;
            target_capture = row.capture;
            expect(target_capture && row.command_sequence == 1,
                   "ordinary target binds the actual accepted receipt and immutable capture");
        }
    }
    expect(id && trace.contains([&](const auto& row) {
        return row.subject == NcclActorAuditSubject::kContinuation && row.id == id &&
               row.event == NcclActorAuditEvent::kRelease && row.has_host && !row.retired &&
               row.capture == target_capture;
    }) && trace.contains([&](const auto& row) {
        return row.subject == NcclActorAuditSubject::kContinuation && row.id == id &&
               row.event == NcclActorAuditEvent::kRetirement && row.retired &&
               row.capture == target_capture;
    }),
           "same stored capture survives reply release through independent retirement");
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kBatch && !row.open_group && row.index &&
               !row.has_physical_work;
    }),
           "ordinary retained metadata never fabricates duplicate native physical ownership");
}
void audit_ordinary_retries() {
    const auto trace = audited(ordinary_authorization);
    expect(trace.contains([](const auto& row) {
        return row.subject == NcclActorAuditSubject::kContinuation &&
               row.event == NcclActorAuditEvent::kObservation && row.outcome == 1 &&
               row.status == CUDA_ERROR_NOT_READY;
    }) && trace.contains([](const auto& row) {
        return row.subject == NcclActorAuditSubject::kContinuation &&
               row.event == NcclActorAuditEvent::kRetirement &&
               row.action == NcclActorAuditAction::kCancel && row.retired;
    }),
           "ordinary deferred retries and never-entered cancellation stay explicit");
}
void audit_parent_retirement() {
    const auto trace = audited(cleanup_retirement_refund);
    std::uint64_t ticket{};
    expect(trace.contains([&](const auto& row) {
        if (row.event != NcclActorAuditEvent::kCleanup ||
            row.action != NcclActorAuditAction::kSeal || row.count != 2)
            return false;
        ticket = row.ticket;
        return ticket != 0;
    }),
           "two-phase cleanup publishes the exact bounded sealed parent set");
    std::size_t members{};
    for (std::size_t index = 0; index != trace.size; ++index) {
        const auto& row = trace.records[index];
        if (row.subject == NcclActorAuditSubject::kParent && row.ticket == ticket &&
            row.action == NcclActorAuditAction::kSeal)
            ++members;
    }
    expect(members == 2 && trace.contains([&](const auto& row) {
        return row.event == NcclActorAuditEvent::kCleanup && row.ticket == ticket &&
               row.status == CUDA_ERROR_NOT_READY && !row.retired;
    }) && trace.contains([&](const auto& row) {
        return row.event == NcclActorAuditEvent::kCleanup && row.ticket == ticket &&
               row.status == CUDA_SUCCESS && row.retired;
    }),
           "unfinished helper proof retains exact parents before authoritative retirement");
}
void audit_closed_source_capture() {
    const auto trace = audited(ordinary_explicit_close_parent);
    std::uint64_t capture{}, id{};
    for (std::size_t index = 0; index != trace.size; ++index) {
        const auto& row = trace.records[index];
        if (row.event == NcclActorAuditEvent::kBinding &&
            row.binding == NcclActorAuditBinding::kPrecall &&
            row.action == NcclActorAuditAction::kBind &&
            row.ordinary_opcode == static_cast<std::uint32_t>(wire::ExecutionOpcode::kClose)) {
            id = row.id;
            capture = row.capture;
        }
    }
    expect(capture && trace.contains([&](const auto& row) {
        return row.subject == NcclActorAuditSubject::kContinuation && row.id == id &&
               row.event == NcclActorAuditEvent::kBinding &&
               row.action == NcclActorAuditAction::kRemove && row.capture == capture;
    }),
           "healthy actual source shutdown cannot erase its previously captured diagnostic ID");
}
void audit_disconnect() {
    const auto trace = audited(owner_and_disconnect);
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kDisconnect && row.peer == 17 &&
               !row.physical_complete && !row.retired;
    }) && trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kAdmission && !row.accepted &&
               row.status == CUDA_ERROR_DEINITIALIZED;
    }),
           "disconnect is admission closure, not native cancellation or stop proof");
}
void audit_uncertain_retention() {
    const auto trace = audited(ordinary_lost_proof, false);
    expect(trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kBatch && row.unknown;
    }) && !trace.contains([](const auto& row) {
        return row.event == NcclActorAuditEvent::kStopped;
    }),
           "actual proof loss remains unknown, never diagnostic physical completion");
}
}  // namespace
int main() {
    const std::array tests{all_routes,
                           authorization,
                           source_dependency_inspection,
                           grouping,
                           errors_and_identity,
                           grouped_output,
                           async_continuation,
                           prepared_group_credit,
                           async_group_credit,
                           retained_tail,
                           held_callback_and_context,
                           no_producer,
                           owner_and_disconnect,
                           ordinary_authorization,
                           ordinary_command_and_retirement,
                           ordinary_rejected_prefix,
                           ordinary_import_cleanup,
                           ordinary_limits_and_failed_import,
                           ordinary_synchronous_tail,
                           ordinary_synchronous_error_tail,
                           ordinary_ordering_reply_before_issue,
                           ordinary_context_free_metadata,
                           ordinary_retired_source_parent,
                           ordinary_explicit_close_parent,
                           ordinary_lost_proof,
                           ordinary_source_fence,
                           cleanup_retirement_refund,
                           cleanup_retirement_active_and_sealed,
                           cleanup_retirement_named_close,
                           cleanup_retirement_after_admission_close,
                           cleanup_retirement_cancelled_close,
                           cleanup_retirement_missing_proof,
                           cleanup_retirement_uncertain_proof,
                           audit_configuration,
                           audit_native_routes,
                           audit_group_members,
                           audit_original_errors,
                           audit_context_restore_failure,
                           audit_context_observation_failure,
                           audit_communicator_restore_failure,
                           audit_async_states,
                           audit_null_communicators,
                           audit_multiple_communicators,
                           audit_physical_retention,
                           audit_source_progress,
                           audit_ordinary_lifetime,
                           audit_ordinary_retries,
                           audit_parent_retirement,
                           audit_closed_source_capture,
                           audit_disconnect,
                           audit_uncertain_retention};
    for (std::size_t i = 0; i != tests.size(); ++i) {
        const auto child = ::fork();
        expect(child >= 0, "isolated fixture process");
        if (!child) {
            tests[i]();
            // Missing output proof and actual native proof loss retain owners.
            // Healthy children must run the leak sanitizer's normal exit hook.
            if (tests[i] == no_producer || tests[i] == ordinary_lost_proof ||
                tests[i] == cleanup_retirement_missing_proof ||
                tests[i] == cleanup_retirement_uncertain_proof ||
                tests[i] == audit_context_restore_failure ||
                tests[i] == audit_context_observation_failure ||
                tests[i] == audit_communicator_restore_failure ||
                tests[i] == audit_uncertain_retention)
                std::_Exit(0);
            std::exit(EXIT_SUCCESS);
        }
        int status{};
        expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
               "independent actor scenario");
    }
    std::cout << "NCCL actor: " << tests.size() << " isolated CPU composition groups passed\n";
}
