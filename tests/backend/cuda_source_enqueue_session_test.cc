#include "internal/cuda_execution_session.h"
#include "internal/cuda_native_execution_lanes.h"
#ifdef GLIMMER_HAS_CUDNN
#include "internal/cuda_dnn_registry.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
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
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
void expect(bool value, std::string_view message) {
    if (value) return;
    std::cerr << "source enqueue: " << message << '\n';
    std::_Exit(1);
}
void success(const CudaSessionResult& result) {
    expect(result.status == CUDA_SUCCESS && result.disposition == CudaSessionDisposition::kComplete,
           "session operation completed");
}
template <typename Predicate>
void await(Predicate predicate) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!predicate()) {
        expect(std::chrono::steady_clock::now() < until, "bounded independent worker handshake");
        std::this_thread::yield();
    }
}
struct Event {
    bool alive{}, recorded{};
    CUstream stream{};
};
struct Stream {
    bool alive{};
    unsigned flags{};
};
struct Visit {
    char kind{};
    CUstream stream{};
    CUevent event{};
};
struct Driver {
    int context{}, foreign_context{};
    std::array<int, 4> additional_contexts{};
    unsigned contexts_created{};
    std::array<Event, 128> events{};
    std::array<Stream, 16> streams{};
    std::array<std::byte, 128> memory{};
    std::vector<Visit> visits;
    unsigned creates{}, destroys{}, releases{}, queries{}, syncs{}, uploads{}, downloads{};
    unsigned fills{}, record_attempts{}, event_creates{}, event_destroys{}, fail_record_attempt{},
        fail_query_attempt{};
    bool stop{};
    bool loses_native_context{};
    CUstream blocked_stream{};
    CUresult wait_status{CUDA_SUCCESS}, record_status{CUDA_SUCCESS};
    CUresult native_status{CUDA_SUCCESS};
    CUresult event_create_status{CUDA_SUCCESS};
    const void* upload_source{};
    void* download_destination{};
    std::function<void()> proof, event_factory_proof, event_destroy_proof;
    cudaStream_t blas_stream{}, dnn_stream{};
    int blas{}, dnn{};
} driver;
thread_local CUcontext current{};
CUresult CUDAAPI set_current(CUcontext context) {
    current = context;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* output) {
    if (driver.proof) driver.proof();
    *output = current;
    return CUDA_SUCCESS;
}
CUstream proxy() {
    return reinterpret_cast<CUstream>(&driver.streams[0]);
}
void reset() {
    driver = {};
    current = nullptr;
    driver.visits.reserve(1024);
}

// Exact injected public SDK signatures, never a GPU or a replacement oracle.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
CudaSessionDispatch dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* device, int ordinal) {
        *device = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        expect(driver.contexts_created <= driver.additional_contexts.size(),
               "bounded distinct native context inventory");
        auto* context = driver.contexts_created == 0
                            ? &driver.context
                            : &driver.additional_contexts[driver.contexts_created - 1];
        ++driver.contexts_created;
        *output = current = reinterpret_cast<CUcontext>(context);
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = set_current;
    sdk.context_destroy = [](CUcontext context) {
        expect(context == current, "owned context cleanup");
        current = nullptr;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = get_current;
    sdk.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.module_unload = [](CUmodule) { return CUDA_SUCCESS; };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_INVALID_VALUE;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t size) {
        expect(size <= driver.memory.size(), "bounded test allocation");
        *output = reinterpret_cast<CUdeviceptr>(driver.memory.data());
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr) {
        ++driver.releases;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr destination, const void* source, std::size_t bytes) {
        ++driver.uploads;
        driver.upload_source = source;
        driver.visits.push_back({'U', nullptr, nullptr});
        // The Driver passes this owned fixture address as an integer.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        if (driver.loses_native_context)
            current = reinterpret_cast<CUcontext>(&driver.foreign_context);
        return driver.native_status;
    };
    sdk.download = [](void* destination, CUdeviceptr source, std::size_t bytes) {
        ++driver.downloads;
        driver.download_destination = destination;
        driver.visits.push_back({'D', nullptr, nullptr});
        // The Driver passes this owned fixture address as an integer.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        if (driver.loses_native_context)
            current = reinterpret_cast<CUcontext>(&driver.foreign_context);
        return driver.native_status;
    };
    sdk.stream_create = [](CUstream* output, unsigned flags) {
        expect(driver.creates < driver.streams.size(), "bounded stream inventory");
        auto& stream = driver.streams[driver.creates++];
        stream = {true, flags};
        *output = reinterpret_cast<CUstream>(&stream);
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream value) {
        auto& stream = *reinterpret_cast<Stream*>(value);
        expect(stream.alive, "stream destruction once");
        stream.alive = false;
        ++driver.destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, unsigned, CUstream stream, void**, void**) {
        driver.visits.push_back({'K', stream, nullptr});
        return CUDA_SUCCESS;
    };
    sdk.execution.event_create = [](CUevent* output, unsigned) {
        if (driver.event_factory_proof) driver.event_factory_proof();
        ++driver.event_creates;
        if (driver.event_create_status != CUDA_SUCCESS) return driver.event_create_status;
        const auto available = std::ranges::find(driver.events, false, &Event::alive);
        expect(available != driver.events.end(), "bounded event inventory");
        *available = {true, false, nullptr};
        *output = reinterpret_cast<CUevent>(&*available);
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent value, CUstream stream) {
        auto& event = *reinterpret_cast<Event*>(value);
        expect(event.alive && !event.recorded, "record a fresh exact event");
        ++driver.record_attempts;
        if (driver.record_attempts == driver.fail_record_attempt) return CUDA_ERROR_INVALID_VALUE;
        if (driver.record_status != CUDA_SUCCESS) return driver.record_status;
        event.recorded = true;
        event.stream = stream;
        driver.visits.push_back({'R', stream, value});
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent value) {
        expect(reinterpret_cast<Event*>(value)->recorded, "never query an unrecorded prefix");
        ++driver.queries;
        if (driver.queries == driver.fail_query_attempt) return CUDA_ERROR_INVALID_CONTEXT;
        return driver.stop && (!driver.blocked_stream ||
                               reinterpret_cast<Event*>(value)->stream != driver.blocked_stream)
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent value) {
        if (driver.event_destroy_proof) driver.event_destroy_proof();
        auto& event = *reinterpret_cast<Event*>(value);
        expect(event.alive, "destroy an event once");
        event.alive = false;
        ++driver.event_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        ++driver.syncs;
        return driver.stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.stream_wait_event = [](CUstream stream, CUevent value, unsigned flags) {
        const auto& event = *reinterpret_cast<Event*>(value);
        expect(event.alive && event.recorded && flags == 0, "wait only a real recorded dependency");
        driver.visits.push_back({'W', stream, value});
        return driver.wait_status;
    };
    sdk.execution.stream_wait_event = sdk.stream_wait_event;
    sdk.execution.fill8_async = [](CUdeviceptr, unsigned char, std::size_t, CUstream stream) {
        driver.visits.push_back({'F', stream, nullptr});
        return CUDA_SUCCESS;
    };
    sdk.execution.fill8 = [](CUdeviceptr, unsigned char, std::size_t) {
        ++driver.fills;
        driver.visits.push_back({'S', nullptr, nullptr});
        if (driver.loses_native_context)
            current = reinterpret_cast<CUcontext>(&driver.foreign_context);
        return driver.native_status;
    };
    sdk.execution.copy_device = [](CUdeviceptr, CUdeviceptr, std::size_t) {
        driver.visits.push_back({'C', nullptr, nullptr});
        return CUDA_SUCCESS;
    };
    sdk.blas.create = [](cublasHandle_t* output) {
        *output = reinterpret_cast<cublasHandle_t>(&driver.blas);
        driver.blas_stream = nullptr;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.destroy = [](cublasHandle_t) { return CUBLAS_STATUS_SUCCESS; };
    sdk.blas.set_stream = [](cublasHandle_t, cudaStream_t stream) {
        driver.blas_stream = stream;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.get_stream = [](cublasHandle_t, cudaStream_t* stream) {
        *stream = driver.blas_stream;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                        const float*, const float*, int, const float*, int, const float*, float*,
                        int) { return CUBLAS_STATUS_SUCCESS; };
#ifdef GLIMMER_HAS_CUDNN
    auto dnn = std::make_shared<CudaDnnDispatch>();
    dnn->create = [](cudnnHandle_t* output) {
        *output = reinterpret_cast<cudnnHandle_t>(&driver.dnn);
        driver.dnn_stream = nullptr;
        return CUDNN_STATUS_SUCCESS;
    };
    dnn->destroy = [](cudnnHandle_t) { return CUDNN_STATUS_SUCCESS; };
    dnn->set_stream = [](cudnnHandle_t, cudaStream_t stream) {
        driver.dnn_stream = stream;
        return CUDNN_STATUS_SUCCESS;
    };
    dnn->get_stream = [](cudnnHandle_t, cudaStream_t* stream) {
        *stream = driver.dnn_stream;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.dnn = std::move(dnn);
#endif
    return sdk;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

std::unique_ptr<CudaExecutionSession> session(bool uses_proxy = true,
                                              std::size_t staging_bytes = std::size_t{64} * 1024 *
                                                                          1024) {
    auto result = std::make_unique<CudaExecutionSession>(
        ExecutionSessionIdentity{11, 12, 1}, 13, dispatch(), CudaSessionLimits{},
        CudaSessionPersistentConfig{}, nullptr, nullptr, false,
        CudaSessionStreamConfig{uses_proxy, staging_bytes});
    success(result->initialize(0));
    return result;
}
ExecutionReceipt fill(CudaExecutionSession& owner, std::uint64_t sequence, std::uint64_t allocation,
                      std::uint64_t stream = 0, bool asynchronous = true) {
    const auto accepted = owner.accept_memory(sequence, {.kind = CudaMemoryKind::kFill8,
                                                         .destination = {allocation, 0},
                                                         .count = 8,
                                                         .value = 5,
                                                         .stream = stream,
                                                         .is_asynchronous = asynchronous});
    expect(accepted.accepted(), "ordinary fill admitted exactly once");
    return accepted.receipt;
}
void issue(CudaExecutionSession& owner, const ExecutionReceipt& receipt) {
    const auto issued = owner.issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued &&
               issued.native_status == CUDA_SUCCESS && issued.completion_status == CUDA_SUCCESS,
           "ordinary native issue succeeds");
    expect(owner.host_result(receipt).has_value() && owner.acknowledge_result(receipt),
           "ordinary reply independently acknowledged");
}
void retire(CudaExecutionSession& owner) {
    driver.stop = true;
    expect(owner.poll().status == CUDA_SUCCESS, "actual event stop permits cleanup");
}
void stop(CudaNativeExecutionLanes& lanes) {
    expect(lanes.request_stop() == CUDA_SUCCESS, "stop real workers");
    await([&] { return lanes.shutdown() == CUDA_SUCCESS; });
}

void actual_worker_and_posted_prefix() {
    reset();
    auto owner = session();
    const auto allocation = owner->allocate(32);
    success(allocation);
    auto resource = std::make_shared<int>(7);
    const std::weak_ptr<int> retained = resource;
    const auto turn = owner->accept_enqueue_turn(0, resource);
    expect(turn.status == CUDA_SUCCESS && owner->snapshot().submissions.outstanding == 0 &&
               owner->snapshot().enqueue_turns == 1 &&
               owner->snapshot().external_work_in_flight == 0,
           "private admission allocates no ordinary sequence or reply");
    expect(owner->begin_enqueue_turn(turn.turn) == CUDA_SUCCESS &&
               owner->snapshot().external_work_in_flight == 1 &&
               turn.worker.begin() == CUDA_ERROR_INVALID_CONTEXT,
           "source is not its native worker");
    CudaNativeExecutionLanes lanes({set_current, get_current});
    auto entered = std::make_shared<std::atomic<bool>>(false);
    auto leave = std::make_shared<std::atomic<bool>>(false);
    const auto call =
        lanes.submit({21, {22, 0}}, {.sequence = 1,
                                     .device_ordinal = 0,
                                     .context = {13, current, resource},
                                     .arguments_lifetime = resource,
                                     .invoke = [worker = turn.worker, entered, leave] {
                                         const auto status = worker.begin();
                                         if (status != CUDA_SUCCESS) return status;
                                         entered->store(true);
                                         await([&] { return leave->load(); });
                                         return CUDA_SUCCESS;
                                     }});
    expect(call.status == CUDA_SUCCESS, "actual lane admission");
    await([&] { return entered->load(); });
    expect(turn.worker.bind_ticket(call.ticket) == CUDA_SUCCESS,
           "worker entry may precede exact owner ticket binding");
    const auto source_context = current;
    driver.proof = [&] {
        expect(owner->deactivate().status == CUDA_ERROR_NOT_READY,
               "owner TLS-transition proof cannot recursively deactivate the source");
    };
    success(owner->deactivate());
    expect(current == nullptr,
           "source host exclusion still permits checked owner-only TLS restoration");
    driver.proof = {};
    success(owner->activate());
    expect(current == source_context, "owner may revisit the exact retained source context");
    const auto visits = driver.visits.size();
    expect(
        owner->finish_enqueue_turn(turn.turn) == CUDA_ERROR_NOT_READY &&
            owner->issue_next().native_status == CUDA_ERROR_NOT_READY &&
            owner->progress_enqueue_dependency(turn.turn).native_status == CUDA_ERROR_NOT_READY &&
            owner->inspect_enqueue_dependency(turn.turn).status == CUDA_ERROR_NOT_READY &&
            !owner->inspect_enqueue_dependency(turn.turn).has_progress &&
            owner->poll().status == CUDA_ERROR_NOT_READY && driver.visits.size() == visits &&
            owner->snapshot().external_work_in_flight == 1,
        "held host callback excludes native source issue and retirement");
    leave->store(true);
    await([&] { return call.ticket.observe().phase == CudaNativeExecutionCallPhase::kReturned; });
    expect(owner->finish_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "exact returned healthy ticket publishes only the posted prefix");
    const auto frontier = owner->capture_frontier(0);
    expect(frontier.status == CUDA_SUCCESS && frontier.frontier.observe() == CUDA_ERROR_NOT_READY,
           "posted prefix is independently queryable, not host-return GPU completion");
    expect(!owner->inspect_frontier_dependency(turn.frontier).has_progress &&
               !owner->inspect_frontier_retirement(turn.frontier).has_progress,
           "finished external prefix still awaiting GPU stop has no native issue/resume action");
    expect(driver.visits.back().kind == 'R' && driver.visits.back().stream == proxy(),
           "prefix recorded on exact proxy");
    // A later ordinary producer remains legal before any hypothetical outer End.
    const auto ordinary = fill(*owner, 1, allocation.id);
    issue(*owner, ordinary);
    expect(owner->snapshot().enqueue_turns == 1 && owner->snapshot().external_work_in_flight == 1 &&
               driver.syncs == 0,
           "posted inner prefix neither drains nor retains host exclusion");
    const auto later = fill(*owner, 2, allocation.id);
    const auto broad = owner->capture_frontier(0);
    expect(broad.status == CUDA_SUCCESS, "later accepted ordinary command has its own frontier");
    expect(lanes.release(call.ticket) == CUDA_SUCCESS, "native packet host storage release");
    stop(lanes);
    resource.reset();
    expect(!retained.expired(), "prefix resource lives through physical stop");
    retire(*owner);
    expect(frontier.frontier.observe() == CUDA_SUCCESS && owner->snapshot().enqueue_turns == 0 &&
               owner->snapshot().external_work_in_flight == 0,
           "event stop retires prefix without inventing an application receipt");
    expect(
        turn.frontier.observe() == CUDA_SUCCESS && broad.frontier.observe() == CUDA_ERROR_NOT_READY,
        "acceptance frontier excludes later queued ordinary work, unlike a broad capture");
    issue(*owner, later);
    retire(*owner);
    success(owner->release(allocation.id));
    success(owner->shutdown());
}

void started_external_reservations_are_not_acceptance() {
    reset();
    auto owner = session();
    const auto resource = std::make_shared<int>(0);
    const auto first = owner->accept_external_barrier(0, resource);
    const auto second = owner->accept_external_barrier(0, resource);
    expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS &&
               owner->snapshot().external_barriers == 2 &&
               owner->snapshot().external_work_in_flight == 0,
           "accepted unstarted barriers do not manufacture physical reservations");
    expect(owner->begin_external_barrier(first.barrier) == CUDA_SUCCESS &&
               owner->snapshot().external_work_in_flight == 1,
           "only the begun external record reserves physical ownership");
    const auto queries = driver.queries;
    expect(owner->poll().status == CUDA_SUCCESS && driver.queries == queries &&
               owner->snapshot().external_work_in_flight == 1,
           "unrecorded started boundary stays reserved without querying its event");
    const auto source_context = current;
    const auto visits = driver.visits.size();
    current = nullptr;
    expect(owner->snapshot().external_work_in_flight == 1 && driver.visits.size() == visits,
           "started reservation inspection is CPU-only and independent of current context");
    current = source_context;
    expect(owner->record_external_enqueue(first.barrier) == CUDA_SUCCESS &&
               owner->begin_external_barrier(second.barrier) == CUDA_SUCCESS &&
               owner->snapshot().external_work_in_flight == 2,
           "each started retained record contributes exactly one reservation");
    expect(owner->record_external_enqueue(second.barrier) == CUDA_SUCCESS &&
               owner->poll().status == CUDA_SUCCESS &&
               owner->snapshot().external_work_in_flight == 2,
           "recorded but pending tails retain both physical reservations");
    retire(*owner);
    expect(owner->snapshot().external_barriers == 0 &&
               owner->snapshot().external_work_in_flight == 0 &&
               first.frontier.observe() == CUDA_SUCCESS &&
               second.frontier.observe() == CUDA_SUCCESS,
           "only actual retirement clears the started reservations");
    const auto cancelled = owner->accept_external_barrier(0, resource);
    expect(cancelled.status == CUDA_SUCCESS && owner->snapshot().external_work_in_flight == 0 &&
               owner->cancel_external_barrier(cancelled.barrier) == CUDA_SUCCESS &&
               owner->snapshot().external_work_in_flight == 0,
           "known unstarted cancellation never creates physical demand");
    success(owner->shutdown());
}

void logical_edges_and_native_synchronous_calls() {
    reset();
    auto owner = session();
    const auto memory = owner->allocate(32);
    success(memory);
    const auto blocking = owner->create_stream(CU_STREAM_DEFAULT);
    success(blocking);
    const auto independent = owner->create_stream(CU_STREAM_NON_BLOCKING);
    success(independent);
    const auto first = fill(*owner, 1, memory.id);
    issue(*owner, first);
    const auto first_event = driver.visits.back().event;
    const auto next = fill(*owner, 2, memory.id, blocking.id);
    issue(*owner, next);
    expect(std::ranges::any_of(driver.visits,
                               [&](const Visit& visit) {
                                   return visit.kind == 'W' && visit.event == first_event &&
                                          visit.stream != proxy();
                               }),
           "blocking stream receives logical legacy predecessor event");
    const auto waits = std::ranges::count(driver.visits, 'W', &Visit::kind);
    const auto third = fill(*owner, 3, memory.id, independent.id);
    issue(*owner, third);
    expect(std::ranges::count(driver.visits, 'W', &Visit::kind) == waits,
           "independent nonblocking stream has no invented legacy dependency");
    const auto fourth = fill(*owner, 4, memory.id);
    issue(*owner, fourth);
    expect(std::ranges::count(driver.visits, 'W', &Visit::kind) == waits + 1,
           "logical zero receives blocking-stream predecessor, not nonblocking work");
    const auto synchronous = fill(*owner, 5, memory.id, 0, false);
    issue(*owner, synchronous);
    const auto sync = std::ranges::find(driver.visits, 'S', &Visit::kind);
    expect(sync != driver.visits.end() && sync->stream == nullptr && (sync + 1)->kind == 'R' &&
               (sync + 1)->stream == nullptr && (sync + 2)->kind == 'W' &&
               (sync + 2)->stream == proxy() && (sync + 2)->event == (sync + 1)->event,
           "original synchronous fill has real native-legacy to proxy bridge");
    retire(*owner);
    std::array<std::byte, 16> input{}, output{};
    input.fill(std::byte{0x5a});
    driver.stop = false;
    const auto uploaded = owner->upload({memory.id, 0}, input);
    success(uploaded);
    expect(driver.uploads == 1 && owner->snapshot().enqueue_turns == 1 &&
               owner->snapshot().external_work_in_flight == 1,
           "synchronous HtoD original called once with owned posted-prefix lifetime");
    expect(uploaded.frontier.observe() == CUDA_ERROR_NOT_READY &&
               !uploaded.frontier.retirement().is_retired,
           "synchronous host return exposes its actual pending private prefix");
    expect(owner->snapshot().default_proxy_staging_bytes == input.size(),
           "aggregate staging charge survives synchronous host return");
    retire(*owner);
    expect(owner->snapshot().default_proxy_staging_bytes == 0 &&
               owner->snapshot().external_work_in_flight == 0 &&
               uploaded.frontier.retirement().is_retired,
           "only actual physical prefix retirement refunds staging");
    driver.stop = false;
    const auto downloaded = owner->download({memory.id, 0}, output);
    success(downloaded);
    expect(driver.downloads == 1 && output == input && driver.syncs == 0,
           "synchronous DtoH publishes complete bytes with no broad synchronize");
    const auto later = fill(*owner, 6, memory.id, independent.id);
    expect(downloaded.frontier.observe() == CUDA_ERROR_NOT_READY,
           "download result carries only its actual accepted legacy tail");
    retire(*owner);
    expect(downloaded.frontier.retirement().is_retired &&
               owner->snapshot().submissions.queued == 1 &&
               owner->reject_command(later, CUDA_ERROR_DEINITIALIZED) &&
               owner->acknowledge_result(later),
           "later ordinary admission cannot widen a returned synchronous tail");
    success(owner->release(memory.id));
    success(owner->destroy_stream(blocking.id));
    success(owner->destroy_stream(independent.id));
    success(owner->shutdown());
    expect(driver.destroys == 3, "two public streams and private proxy cleaned once");
}

void session_reservations_preserve_owner_and_admission() {
    reset();
    auto owner = session();
    const auto memory = owner->allocate(16);
    success(memory);
    const auto first = owner->reserve_command_frontier(1);
    const auto second = owner->reserve_command_frontier(1);
    const auto pending = owner->reserve_command_frontier(2);
    expect(
        first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS &&
            pending.status == CUDA_SUCCESS &&
            owner->reserved_command_frontier(first.reservation).status == CUDA_ERROR_NOT_READY &&
            !owner->reserved_command_frontier(first.reservation).frontier.retirement().is_retired,
        "Session pending reservations are CPU metadata, not completion proof");
    const auto invalid = owner->accept_memory(1, {.kind = CudaMemoryKind::kFill8,
                                                  .destination = {memory.id, 16},
                                                  .count = 1,
                                                  .stream = 0,
                                                  .is_asynchronous = true});
    expect(!invalid.accepted() && owner->snapshot().submissions.outstanding == 0 &&
               owner->reserved_command_frontier(second.reservation).status == CUDA_ERROR_NOT_READY,
           "range rejection admits nothing and leaves all exact observers retryable");
    const auto receipt = fill(*owner, 1, memory.id);
    const auto exact = owner->reserved_command_frontier(first.reservation);
    const auto other = owner->reserved_command_frontier(second.reservation);
    expect(exact.status == CUDA_SUCCESS && other.status == CUDA_SUCCESS &&
               !exact.frontier.retirement().is_retired,
           "normal Session memory admission binds both observers without a second submit");
    driver.event_factory_proof = [&] {
        expect(
            owner->reserve_command_frontier(3).status == CUDA_ERROR_NOT_READY &&
                owner->reserved_command_frontier(first.reservation).status ==
                    CUDA_ERROR_NOT_READY &&
                owner->cancel_frontier_reservation(pending.reservation) == CUDA_ERROR_NOT_READY &&
                !owner->reject_command(receipt, CUDA_ERROR_DEINITIALIZED),
            "Session native factory cannot reenter any capture or cancellation path");
    };
    issue(*owner, receipt);
    driver.event_factory_proof = {};
    expect(!owner->reject_command(receipt, CUDA_ERROR_DEINITIALIZED) &&
               !exact.frontier.retirement().is_retired,
           "actual requested native work cannot be reclassified as rejected");
    expect(std::ranges::count(driver.visits, 'F', &Visit::kind) == 1 && driver.fills == 0,
           "one original asynchronous fill is observed without a synchronous substitute");
    retire(*owner);
    expect(exact.frontier.retirement().is_retired && other.frontier.retirement().is_retired,
           "both observers retire from one real native event");
    const auto cancelled = fill(*owner, 2, memory.id);
    expect(
        owner->reject_command(cancelled, CUDA_ERROR_DEINITIALIZED) &&
            owner->reserved_command_frontier(pending.reservation).frontier.observe() ==
                CUDA_ERROR_DEINITIALIZED &&
            owner->reserved_command_frontier(pending.reservation)
                .frontier.retirement()
                .is_retired &&
            owner->host_result(cancelled)->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
            owner->acknowledge_result(cancelled) &&
            std::ranges::count(driver.visits, 'F', &Visit::kind) == 1 && driver.fills == 0,
        "targeted Session rejection retains exact host provenance and no native callback");
    const auto unused = owner->reserve_command_frontier(3);
    success(owner->release(memory.id));
    owner->close();
    expect(owner->reserve_command_frontier(3).status == CUDA_ERROR_DEINITIALIZED &&
               owner->cancel_frontier_reservation(unused.reservation) == CUDA_SUCCESS &&
               owner->reserved_command_frontier(first.reservation).status == CUDA_SUCCESS,
           "closing Session permits only existing reservation observation/cancellation");
    success(owner->shutdown());
}

void session_retirement_progress_keeps_exact_cut() {
    reset();
    auto owner = session();
    const auto memory = owner->allocate(16);
    success(memory);
    const auto rejected = fill(*owner, 1, memory.id);
    const auto live = fill(*owner, 2, memory.id);
    const auto cut = owner->capture_frontier();
    const auto later = fill(*owner, 3, memory.id);
    expect(cut.status == CUDA_SUCCESS && owner->reject_command(rejected, CUDA_ERROR_INVALID_VALUE),
           "Session captures mixed rejected/live work before later admission");
    const auto pending = owner->inspect_frontier_retirement(cut.frontier);
    expect(pending.status == CUDA_ERROR_NOT_READY && !pending.has_failure_observation &&
               !pending.has_retired_failure && cut.frontier.observe() == CUDA_ERROR_INVALID_VALUE &&
               owner->inspect_frontier_dependency(cut.frontier).has_retired_failure,
           "Session retirement-only view does not overwrite the original dependency error");
    unsigned reentries{};
    driver.event_factory_proof = [&] {
        ++reentries;
        const auto blocked = owner->inspect_frontier_retirement(cut.frontier);
        expect(blocked.status == CUDA_ERROR_NOT_READY && !blocked.has_failure_observation &&
                   !blocked.has_retired_failure &&
                   owner->progress_frontier_retirement(cut.frontier).native_status ==
                       CUDA_ERROR_NOT_READY,
               "Session factory cannot recursively progress a retained cut");
    };
    const auto progressed = owner->progress_frontier_retirement(cut.frontier);
    driver.event_factory_proof = {};
    expect(progressed.state == CudaExecutionIssueState::kIssued && progressed.sequence == 2 &&
               reentries != 0 && std::ranges::count(driver.visits, 'F', &Visit::kind) == 1 &&
               owner->host_result(live)->origin == ExecutionReplyOrigin::kNativeSdk &&
               !cut.frontier.retirement().is_retired,
           "one real captured command issues with exact provenance, not a broad drain");
    retire(*owner);
    const auto stopped = owner->inspect_frontier_retirement(cut.frontier);
    expect(stopped.status == CUDA_SUCCESS && !stopped.has_failure_observation &&
               !stopped.has_retired_failure && cut.frontier.retirement().is_retired &&
               cut.frontier.retirement().status == CUDA_ERROR_INVALID_VALUE &&
               owner->progress_frontier_retirement(cut.frontier).state ==
                   CudaExecutionIssueState::kIdle &&
               std::ranges::count(driver.visits, 'F', &Visit::kind) == 1 &&
               owner->acknowledge_result(rejected) && owner->acknowledge_result(live),
           "Session quiescence view excludes later work and preserves both original receipts");
    issue(*owner, later);
    retire(*owner);
    expect(std::ranges::count(driver.visits, 'F', &Visit::kind) == 2,
           "later ordinary work still requires its separate original issue");
    success(owner->release(memory.id));
    success(owner->shutdown());
}

void same_session_cohort_and_staging_limit() {
    reset();
    auto owner = session(true, 8);
    const auto memory = owner->allocate(32);
    success(memory);
    std::array<std::byte, 16> bytes{};
    expect(owner->upload({memory.id, 0}, bytes).status == CUDA_ERROR_OUT_OF_MEMORY &&
               driver.uploads == 0 && owner->snapshot().default_proxy_staging_bytes == 0,
           "explicit staging cap rejects before original or prefix side effects");
    const auto blocking = owner->create_stream(CU_STREAM_DEFAULT);
    success(blocking);
    const std::array members{std::uint64_t{0}, blocking.id, std::uint64_t{0}};
    issue(*owner, fill(*owner, 1, memory.id));
    issue(*owner, fill(*owner, 2, memory.id, blocking.id));
    auto lifetime = std::make_shared<int>(0);
    const auto turn = owner->accept_enqueue_cohort(members, lifetime);
    expect(turn.status == CUDA_SUCCESS && owner->begin_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "deduplicated same-session cohort acquires one atomic predecessor union");
    CudaNativeExecutionLanes lanes({set_current, get_current});
    const auto call =
        lanes.submit({21, {23, 0}}, {.sequence = 1,
                                     .device_ordinal = 0,
                                     .context = {13, current, lifetime},
                                     .arguments_lifetime = lifetime,
                                     .invoke = [worker = turn.worker] { return worker.begin(); }});
    expect(call.status == CUDA_SUCCESS && turn.worker.bind_ticket(call.ticket) == CUDA_SUCCESS,
           "one exact ticket covers all cohort members");
    await([&] { return call.ticket.observe().phase == CudaNativeExecutionCallPhase::kReturned; });
    const auto before = driver.visits.size();
    expect(owner->finish_enqueue_turn(turn.turn) == CUDA_SUCCESS &&
               driver.visits.size() == before + 2 && driver.visits[before].stream == proxy() &&
               driver.visits[before + 1].stream != proxy(),
           "finish records one posted prefix on every distinct source stream");
    const auto first = owner->capture_frontier(0);
    const auto second = owner->capture_frontier(blocking.id);
    expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS,
           "each member frontier retains the whole atomic prefix");
    driver.stop = true;
    driver.blocked_stream = driver.visits[before + 1].stream;
    expect(owner->poll().status == CUDA_SUCCESS && owner->snapshot().enqueue_turns == 1 &&
               owner->snapshot().external_work_in_flight == 1 &&
               first.frontier.observe() == CUDA_ERROR_NOT_READY &&
               second.frontier.observe() == CUDA_ERROR_NOT_READY,
           "one completed member is not whole-cohort completion");
    driver.blocked_stream = nullptr;
    retire(*owner);
    expect(first.frontier.observe() == CUDA_SUCCESS && second.frontier.observe() == CUDA_SUCCESS &&
               owner->snapshot().enqueue_turns == 0 &&
               owner->snapshot().external_work_in_flight == 0,
           "all real member events settle one record");
    expect(lanes.release(call.ticket) == CUDA_SUCCESS,
           "cohort ticket released after returned call");
    stop(lanes);
    success(owner->release(memory.id));
    success(owner->destroy_stream(blocking.id));
    success(owner->shutdown());
}

void queued_predecessor_cancellation_and_owner_gates() {
    reset();
    auto owner = session();
    const auto memory = owner->allocate(32);
    success(memory);
    const auto first = fill(*owner, 1, memory.id);
    auto lifetime = std::make_shared<int>(0);
    const auto turn = owner->accept_enqueue_turn(0, lifetime);
    expect(owner->enqueue_turn_ready(turn.turn) == CUDA_ERROR_NOT_READY,
           "queued predecessor blocks external host entry");
    issue(*owner, first);
    expect(owner->enqueue_turn_ready(turn.turn) == CUDA_SUCCESS, "issued prefix permits host turn");
    std::thread foreign([&] {
        expect(owner->begin_enqueue_turn(turn.turn) == CUDA_ERROR_INVALID_CONTEXT,
               "foreign CPU thread cannot mutate source owner");
        expect(owner->progress_enqueue_dependency(turn.turn).native_status ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   !owner->inspect_enqueue_dependency(turn.turn).has_progress,
               "foreign CPU thread cannot progress a retained source cut");
    });
    foreign.join();
    const auto child = fork();
    expect(child >= 0, "fork child created");
    if (child == 0) {
        expect(owner->cancel_enqueue_turn(turn.turn) == CUDA_ERROR_INVALID_CONTEXT,
               "fork cannot enter inherited source state");
        expect(owner->progress_enqueue_dependency(turn.turn).native_status ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   !owner->inspect_enqueue_dependency(turn.turn).has_progress,
               "fork cannot use an inherited dependency grant");
        std::_Exit(0);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork gate child successful");
    driver.proof = [&] {
        expect(owner->begin_enqueue_turn(turn.turn) == CUDA_ERROR_NOT_READY &&
                   owner->issue_next().native_status == CUDA_ERROR_NOT_READY,
               "native proof callback cannot reenter source issue");
    };
    expect(owner->begin_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "begin after queued native prefix");
    driver.proof = {};
    expect(owner->cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "known unentered cancellation succeeds");
    expect(owner->progress_enqueue_dependency(turn.turn).native_status == CUDA_ERROR_INVALID_HANDLE,
           "cancelled turn cannot retain source progress authority");
    expect(!owner->inspect_enqueue_dependency(turn.turn).has_progress,
           "cancelled turn cannot advertise actionable source work");
    CUresult cancelled_entry{CUDA_SUCCESS};
    std::thread cancelled_worker([&] { cancelled_entry = turn.worker.begin(); });
    cancelled_worker.join();
    expect(cancelled_entry == CUDA_ERROR_NOT_READY,
           "known unentered cancellation prevents future distinct worker entry");
    retire(*owner);
    success(owner->release(memory.id));
    success(owner->shutdown());
}

void exact_dependency_cut_excludes_unrelated_work() {
    reset();
    auto owner = session();
    const auto memory = owner->allocate(32);
    success(memory);
    const auto unrelated = owner->create_stream(CU_STREAM_NON_BLOCKING);
    const auto blocking = owner->create_stream(CU_STREAM_DEFAULT);
    success(unrelated);
    success(blocking);
    const auto first = fill(*owner, 1, memory.id, unrelated.id);
    const auto required = fill(*owner, 2, memory.id, blocking.id);
    const std::array streams{std::uint64_t{0}, blocking.id};
    const auto turn = owner->accept_enqueue_cohort(streams, std::make_shared<int>(0));
    const auto later_same = fill(*owner, 3, memory.id, blocking.id);
    const auto later_unrelated = fill(*owner, 4, memory.id, unrelated.id);
    expect(turn.status == CUDA_SUCCESS &&
               owner->enqueue_turn_ready(turn.turn) == CUDA_ERROR_NOT_READY &&
               owner->inspect_enqueue_dependency(turn.turn).has_progress,
           "cohort cut retains its queued blocking predecessor");
    unsigned reentries{};
    driver.proof = [&] {
        ++reentries;
        const auto nested = owner->progress_enqueue_dependency(turn.turn);
        expect(nested.state == CudaExecutionIssueState::kError &&
                   nested.native_status == CUDA_ERROR_NOT_READY &&
                   !owner->inspect_enqueue_dependency(turn.turn).has_progress,
               "first and later context proofs exclude dependency-progress reentry");
    };
    const auto progressed = owner->progress_enqueue_dependency(turn.turn);
    driver.proof = {};
    expect(reentries && progressed.state == CudaExecutionIssueState::kIssued &&
               progressed.sequence == 2 && progressed.native_status == CUDA_SUCCESS &&
               progressed.completion_status == CUDA_SUCCESS && owner->host_result(required) &&
               owner->acknowledge_result(required),
           "exact filtered predecessor preserves its normal host receipt");
    expect(!owner->host_result(first) && !owner->host_result(later_same) &&
               !owner->host_result(later_unrelated) &&
               std::ranges::count(driver.visits, 'F', &Visit::kind) == 1,
           "dependency progress issues neither earlier independent nor later same-stream work");
    const auto idle = owner->progress_enqueue_dependency(turn.turn);
    expect(idle.state == CudaExecutionIssueState::kIdle && idle.native_status == CUDA_SUCCESS &&
               owner->enqueue_turn_ready(turn.turn) == CUDA_SUCCESS &&
               !owner->inspect_enqueue_dependency(turn.turn).has_progress &&
               std::ranges::count(driver.visits, 'F', &Visit::kind) == 1,
           "ready cut never drains unrelated queued work");
    expect(owner->cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "unused ready cohort is CPU-cancelled");
    issue(*owner, first);
    issue(*owner, later_same);
    issue(*owner, later_unrelated);
    retire(*owner);
    success(owner->release(memory.id));
    success(owner->destroy_stream(blocking.id));
    success(owner->destroy_stream(unrelated.id));
    success(owner->shutdown());
}

class DependencyEvent final : public CudaExecutionEventCall {
   public:
    explicit DependencyEvent(CudaExecutionFrontier foreign = {}) : foreign_(std::move(foreign)) {}
    CUresult issue(CUstream stream) noexcept override {
        ++calls;
        driver.visits.push_back({'E', stream, nullptr});
        return issue_status;
    }
    CUresult dependency_status() const noexcept override {
        if (dependency_failure != CUDA_SUCCESS) return dependency_failure;
        return has_foreign_dependency ? foreign_.observe() : CUDA_SUCCESS;
    }
    unsigned calls{};
    bool has_foreign_dependency{};
    CUresult dependency_failure{CUDA_SUCCESS};
    CUresult issue_status{CUDA_SUCCESS};

   private:
    CudaExecutionFrontier foreign_;
};

void event_dependency_closure_and_foreign_owner() {
    reset();
    current = reinterpret_cast<CUcontext>(&driver.context);
    auto lifetime = std::make_shared<int>(0);
    CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime}, dispatch().execution);
    expect(
        owner.bind_executor() == CUDA_SUCCESS &&
            owner.bind_function(1, reinterpret_cast<CUfunction>(&driver.context), 0, lifetime) &&
            owner.bind_stream(1, reinterpret_cast<CUstream>(&driver.streams[0]), lifetime, true) &&
            owner.bind_stream(2, reinterpret_cast<CUstream>(&driver.streams[1]), lifetime, true) &&
            owner.bind_stream(3, reinterpret_cast<CUstream>(&driver.streams[2]), lifetime),
        "independent and blocking stream bindings");
    const auto kernel = [](std::uint64_t stream) {
        return CudaExecutionKernel{.function = 1,
                                   .stream = stream,
                                   .grid = {8, 1, 1},
                                   .block = {32, 1, 1},
                                   .parameters = {}};
    };
    const auto first = owner.accept_kernel(1, kernel(1));
    const auto producer = owner.accept_kernel(2, kernel(2));
    const auto producer_frontier = owner.capture_command_frontier(2);
    auto event = std::make_shared<DependencyEvent>();
    const auto wait = owner.accept_event(
        3, {.stream = 3, .call = event, .dependency = producer_frontier.frontier});
    const std::array streams{std::uint64_t{0}, std::uint64_t{3}};
    const auto turn = owner.accept_enqueue_cohort(streams, lifetime);
    const auto later = owner.accept_kernel(4, kernel(2));
    expect(first.accepted() && producer.accepted() && producer_frontier.status == CUDA_SUCCESS &&
               wait.accepted() && turn.status == CUDA_SUCCESS && later.accepted(),
           "local event frontier and later ordinary command admitted");
    expect(owner.inspect_enqueue_dependency(turn.turn).has_progress,
           "exact local event closure exposes its ready nonblocking producer");
    const auto selected = owner.progress_enqueue_dependency(turn.turn);
    expect(selected.state == CudaExecutionIssueState::kIssued && selected.sequence == 2 &&
               owner.acknowledge_result(producer.receipt) && event->calls == 0 &&
               !owner.host_result(first.receipt) && !owner.host_result(later.receipt),
           "transitive local event edge selects its nonblocking producer, not earliest ready work");
    const auto pending = owner.progress_enqueue_dependency(turn.turn);
    expect(pending.state == CudaExecutionIssueState::kIdle &&
               pending.native_status == CUDA_ERROR_NOT_READY && event->calls == 0 &&
               !owner.inspect_enqueue_dependency(turn.turn).has_progress &&
               std::ranges::count(driver.visits, 'K', &Visit::kind) == 1,
           "unsettled explicit frontier is not replaced by mere native enqueue");
    driver.stop = true;
    expect(owner.poll().retired == 1 && producer_frontier.frontier.observe() == CUDA_SUCCESS,
           "actual producer event settles the local event dependency");
    const auto waited = owner.progress_enqueue_dependency(turn.turn);
    expect(waited.state == CudaExecutionIssueState::kIssued && waited.sequence == 3 &&
               event->calls == 1 && owner.acknowledge_result(wait.receipt) &&
               owner.enqueue_turn_ready(turn.turn) == CUDA_SUCCESS &&
               owner.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "one required event callback runs after its physical predecessor");
    expect(owner.issue_next().sequence == 1 && owner.acknowledge_result(first.receipt) &&
               owner.issue_next().sequence == 4 && owner.acknowledge_result(later.receipt) &&
               owner.poll().retired == 3,
           "unrelated and later commands remain available to ordinary scheduling");

    CudaExecutionDispatcher foreign({21, 22, 1}, {23, current, lifetime}, dispatch().execution);
    expect(foreign.bind_executor() == CUDA_SUCCESS &&
               foreign.bind_function(1, reinterpret_cast<CUfunction>(&driver.foreign_context), 0,
                                     lifetime),
           "a distinct source owner retains its own command frontier");
    const auto foreign_producer = foreign.accept_kernel(1, kernel(0));
    const auto foreign_frontier = foreign.capture_command_frontier(1);
    auto foreign_event = std::make_shared<DependencyEvent>(foreign_frontier.frontier);
    foreign_event->has_foreign_dependency = true;
    const auto independent = owner.accept_kernel(5, kernel(1));
    const auto foreign_wait =
        owner.accept_event(6, {.stream = 3,
                               .call = foreign_event,
                               .dependency = std::nullopt,
                               .foreign_dependency = foreign_frontier.frontier});
    const auto blocked = owner.accept_enqueue_turn(0, lifetime);
    expect(foreign_producer.accepted() && foreign_frontier.status == CUDA_SUCCESS &&
               independent.accepted() && foreign_wait.accepted() && blocked.status == CUDA_SUCCESS,
           "trusted foreign-generation observer admitted without transplanting frontier authority");
    const auto visits = driver.visits.size();
    const auto opaque = owner.progress_enqueue_dependency(blocked.turn);
    const auto discovered = owner.inspect_enqueue_dependency(blocked.turn);
    expect(discovered.status == CUDA_ERROR_NOT_READY && discovered.foreign_blocker &&
               !discovered.has_progress &&
               discovered.foreign_blocker->same_capture(foreign_frontier.frontier) &&
               foreign.owns_frontier(*discovered.foreign_blocker) &&
               !owner.owns_frontier(*discovered.foreign_blocker) &&
               foreign.inspect_frontier_dependency(*discovered.foreign_blocker).has_progress &&
               !owner.inspect_frontier_dependency(*discovered.foreign_blocker).has_progress &&
               owner.progress_frontier_dependency(*discovered.foreign_blocker).native_status ==
                   CUDA_ERROR_INVALID_HANDLE,
           "inspection returns exact foreign identity without granting this owner foreign issue");
    expect(opaque.state == CudaExecutionIssueState::kIdle &&
               opaque.native_status == CUDA_ERROR_NOT_READY && driver.visits.size() == visits &&
               !owner.host_result(independent.receipt) &&
               !foreign.host_result(foreign_producer.receipt),
           "opaque foreign blocker stays unresolved, with no cross-owner or unrelated issue");
    const auto mixed = owner.capture_frontier();
    const auto actionable = owner.inspect_frontier_dependency(mixed.frontier);
    expect(mixed.status == CUDA_SUCCESS && actionable.status == CUDA_ERROR_NOT_READY &&
               actionable.has_progress && actionable.foreign_blocker &&
               actionable.foreign_blocker->same_capture(foreign_frontier.frontier) &&
               owner.progress_frontier_dependency(mixed.frontier).sequence == 5 &&
               owner.acknowledge_result(independent.receipt) && foreign_event->calls == 0 &&
               !foreign.host_result(foreign_producer.receipt) &&
               !owner.inspect_frontier_dependency(mixed.frontier).has_progress,
           "explicit mixed cut progresses its local member without claiming foreign authority");
    expect(foreign.progress_frontier_dependency(*discovered.foreign_blocker).sequence == 1 &&
               foreign.acknowledge_result(foreign_producer.receipt) && foreign.poll().retired == 1,
           "only separately authorized foreign progress can establish its real completion");
    const auto resolved = owner.progress_enqueue_dependency(blocked.turn);
    expect(resolved.state == CudaExecutionIssueState::kIssued && resolved.sequence == 6 &&
               foreign_event->calls == 1 && owner.acknowledge_result(foreign_wait.receipt) &&
               owner.cancel_enqueue_turn(blocked.turn) == CUDA_SUCCESS,
           "actual foreign completion unblocks exactly the retained wait");
    expect(owner.poll().retired == 2 && foreign.shutdown() && owner.shutdown(),
           "all real issued records stop before independent owner cleanup");
}

void chained_session_generations_survive_rerecord_and_close() {
    reset();
    auto leaf = session();
    const auto leaf_memory = leaf->allocate(32);
    const auto leaf_stream = leaf->create_stream(CU_STREAM_NON_BLOCKING);
    const auto leaf_event = leaf->create_event(0);
    success(leaf_memory);
    success(leaf_stream);
    success(leaf_event);
    auto fresh = leaf->capture_event(leaf_event.id);
    expect(
        fresh.status == CUDA_SUCCESS && !fresh.dependency && fresh.event.observe() == CUDA_SUCCESS,
        "never-recorded generation has no invented dependency cut");
    fresh.event = {};
    const auto leaf_unrelated = fill(*leaf, 1, leaf_memory.id, leaf_stream.id);
    const auto producer = fill(*leaf, 2, leaf_memory.id);
    const auto old_submission = leaf->record_event_with_frontier(3, leaf_event.id, 0, 0);
    success(old_submission.result);
    auto old = leaf->capture_event(leaf_event.id);
    const auto leaf_later = fill(*leaf, 4, leaf_memory.id);
    const auto new_submission = leaf->record_event_with_frontier(5, leaf_event.id, 0, 0);
    success(new_submission.result);
    auto newer = leaf->capture_event(leaf_event.id);
    expect(old.status == CUDA_SUCCESS && old.dependency && newer.dependency &&
               !old.dependency->same_capture(*newer.dependency) &&
               old.dependency->same_capture(old_submission.frontier) &&
               newer.dependency->same_capture(new_submission.frontier) &&
               !old_submission.frontier.retirement().is_retired,
           "rerecord publishes a distinct immutable command generation");
    success(leaf->destroy_event(leaf_event.id));
    leaf->close();
    success(leaf->deactivate());

    auto middle = session();
    const auto middle_memory = middle->allocate(32);
    const auto middle_stream = middle->create_stream(CU_STREAM_NON_BLOCKING);
    const auto middle_event = middle->create_event(0);
    success(middle_memory);
    success(middle_stream);
    success(middle_event);
    const auto middle_unrelated = fill(*middle, 1, middle_memory.id, middle_stream.id);
    const auto middle_wait = middle->wait_event_with_frontier(2, old.event, 0, 0);
    success(middle_wait.result);
    const auto middle_record = middle->record_event_with_frontier(3, middle_event.id, 0, 0);
    success(middle_record.result);
    auto intermediate = middle->capture_event(middle_event.id);
    const auto middle_later = fill(*middle, 4, middle_memory.id);
    expect(intermediate.status == CUDA_SUCCESS && intermediate.dependency &&
               intermediate.dependency->same_capture(middle_record.frontier) &&
               !middle_wait.frontier.same_capture(middle_record.frontier),
           "intermediate event retains its own exact record cut");
    success(middle->deactivate());

    auto root = session();
    const auto root_memory = root->allocate(32);
    const auto root_stream = root->create_stream(CU_STREAM_NON_BLOCKING);
    success(root_memory);
    success(root_stream);
    const auto root_unrelated = fill(*root, 1, root_memory.id, root_stream.id);
    const auto root_wait = root->wait_event_with_frontier(2, intermediate.event, 0, 0);
    success(root_wait.result);
    const auto turn = root->accept_enqueue_turn(0, std::make_shared<int>(0));
    const auto root_later = fill(*root, 3, root_memory.id);
    expect(turn.status == CUDA_SUCCESS,
           "root native reservation captures only the admitted prefix");

    unsigned native_proofs{};
    driver.proof = [&] { ++native_proofs; };
    current = nullptr;
    const auto first = root->inspect_enqueue_dependency(turn.turn);
    const auto second = middle->inspect_frontier_dependency(*intermediate.dependency);
    const auto last = leaf->inspect_frontier_dependency(*old.dependency);
    expect(first.status == CUDA_ERROR_NOT_READY && first.foreign_blocker &&
               first.foreign_blocker->same_capture(*intermediate.dependency) &&
               second.status == CUDA_ERROR_NOT_READY && second.foreign_blocker &&
               second.foreign_blocker->same_capture(*old.dependency) &&
               last.status == CUDA_ERROR_NOT_READY && !last.foreign_blocker && native_proofs == 0 &&
               middle->owns_frontier(*first.foreign_blocker) &&
               leaf->owns_frontier(*second.foreign_blocker) &&
               !root->owns_frontier(*second.foreign_blocker),
           "three-owner discovery is CPU-only under NULL current and authenticates exact edges");
    driver.proof = {};
    expect(root->progress_frontier_dependency(*old.dependency).native_status ==
               CUDA_ERROR_INVALID_HANDLE,
           "target source cannot use the foreign cut as local issue authority");
    std::thread other([&] {
        expect(!old.dependency->same_capture(*old.dependency) &&
                   !leaf->owns_frontier(*old.dependency) &&
                   leaf->inspect_frontier_dependency(*old.dependency).status ==
                       CUDA_ERROR_INVALID_CONTEXT,
               "foreign CPU incarnation cannot discover or authorize inherited cuts");
    });
    other.join();
    const auto child = fork();
    expect(child >= 0, "fork exact-generation inspection control");
    if (!child) {
        expect(
            !old.dependency->same_capture(*old.dependency) &&
                root->inspect_enqueue_dependency(turn.turn).status == CUDA_ERROR_INVALID_CONTEXT &&
                leaf->progress_frontier_dependency(*old.dependency).native_status ==
                    CUDA_ERROR_INVALID_CONTEXT,
            "fork rejects before source callbacks or thread-local authority");
        std::_Exit(0);
    }
    int child_status{};
    expect(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "fork inspection control completes");

    success(leaf->activate());
    driver.proof = [&] {
        expect(leaf->inspect_frontier_dependency(*old.dependency).status == CUDA_ERROR_NOT_READY &&
                   leaf->progress_frontier_dependency(*old.dependency).native_status ==
                       CUDA_ERROR_NOT_READY,
               "native proof excludes recursive inspect and issue even on the exact cut");
    };
    const auto produced = leaf->progress_frontier_dependency(*old.dependency);
    driver.proof = {};
    expect(produced.state == CudaExecutionIssueState::kIssued && produced.sequence == 2 &&
               leaf->acknowledge_result(producer),
           "closed source progresses only its retained old-generation producer");
    expect(leaf->progress_frontier_dependency(*old.dependency).sequence == 3 &&
               old.event.observe() == CUDA_ERROR_NOT_READY &&
               root->inspect_enqueue_dependency(turn.turn).foreign_blocker,
           "record host return is not event-generation physical completion");
    retire(*leaf);
    expect(old.event.observe() == CUDA_SUCCESS &&
               leaf->inspect_frontier_dependency(*old.dependency).status == CUDA_SUCCESS &&
               old_submission.frontier.retirement().is_retired &&
               !new_submission.frontier.retirement().is_retired &&
               leaf->progress_frontier_dependency(*old.dependency).state ==
                   CudaExecutionIssueState::kIdle &&
               newer.event.observe() == CUDA_ERROR_NOT_READY,
           "settled old generation neither resumes nor completes later rerecord");
    success(middle->activate());
    expect(middle->progress_frontier_dependency(*intermediate.dependency).sequence == 2 &&
               middle->progress_frontier_dependency(*intermediate.dependency).sequence == 3,
           "middle source separately advances exactly its foreign wait and captured record");
    retire(*middle);
    expect(middle_wait.frontier.retirement().is_retired &&
               middle_record.frontier.retirement().is_retired &&
               !root_wait.frontier.retirement().is_retired,
           "each target wait has independent stop proof, not its source's completion");
    success(root->activate());
    expect(root->progress_enqueue_dependency(turn.turn).sequence == 2 &&
               root->inspect_enqueue_dependency(turn.turn).status == CUDA_SUCCESS &&
               root->cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "original cut becomes ready only after actual transitive event completion");
    retire(*root);
    expect(root_wait.frontier.retirement().status == CUDA_SUCCESS &&
               root_wait.frontier.retirement().is_retired,
           "exact foreign wait retires without including later ordinary work");
    expect(!leaf->host_result(leaf_unrelated) && !leaf->host_result(leaf_later) &&
               !middle->host_result(middle_unrelated) && !middle->host_result(middle_later) &&
               !root->host_result(root_unrelated) && !root->host_result(root_later),
           "no transitive progress touches any earlier independent or later ordinary request");

    issue(*root, root_unrelated);
    issue(*root, root_later);
    retire(*root);
    success(root->shutdown());
    intermediate.event = {};
    success(middle->activate());
    issue(*middle, middle_unrelated);
    issue(*middle, middle_later);
    retire(*middle);
    success(middle->destroy_event(middle_event.id));
    success(middle->shutdown());
    old.event = {};
    newer.event = {};
    success(leaf->activate());
    issue(*leaf, leaf_unrelated);
    issue(*leaf, leaf_later);
    expect(leaf->issue_next().sequence == 5,
           "later rerecord still requires ordinary authorization");
    retire(*leaf);
    success(leaf->shutdown());
    expect(!old.dependency->same_capture(*old.dependency) && !leaf->owns_frontier(*old.dependency),
           "destroyed source incarnation cannot authorize a retained stale token");
    expect(old_submission.frontier.retirement().status == CUDA_ERROR_INVALID_HANDLE &&
               !old_submission.frontier.retirement().is_retired,
           "expired source owner cannot furnish stale retirement proof");
}

void failed_dependency_inspection_is_not_completion() {
    reset();
    current = reinterpret_cast<CUcontext>(&driver.context);
    auto lifetime = std::make_shared<int>(0);
    CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime}, dispatch().execution);
    expect(owner.bind_executor() == CUDA_SUCCESS, "failed-cut owner bound");
    auto call = std::make_shared<DependencyEvent>();
    const auto accepted = owner.accept_event(1, {.stream = 0, .call = call, .dependency = {}});
    const auto frontier = owner.capture_command_frontier(1);
    expect(accepted.accepted() && frontier.status == CUDA_SUCCESS, "failed-cut record captured");
    CudaExecutionDispatcher target({21, 22, 1}, {23, current, lifetime}, dispatch().execution);
    expect(target.bind_executor() == CUDA_SUCCESS, "dependent failure owner bound");
    auto wait = std::make_shared<DependencyEvent>(frontier.frontier);
    wait->has_foreign_dependency = true;
    const auto waiting = target.accept_event_with_frontier(
        1, {.stream = 0, .call = wait, .dependency = {}, .foreign_dependency = frontier.frontier});
    const auto turn = target.accept_enqueue_turn(0, lifetime);
    expect(waiting.acceptance.accepted() && turn.status == CUDA_SUCCESS,
           "foreign failure edge retained");
    call->dependency_failure = CUDA_ERROR_INVALID_VALUE;
    const auto visits = driver.visits.size();
    expect(
        owner.inspect_frontier_dependency(frontier.frontier).status == CUDA_ERROR_INVALID_VALUE &&
            driver.visits.size() == visits,
        "CPU inspection reports actual preparatory dependency failure, not readiness");
    const auto rejected = owner.progress_frontier_dependency(frontier.frontier);
    expect(rejected.state == CudaExecutionIssueState::kRejected && rejected.sequence == 1 &&
               rejected.native_status == CUDA_ERROR_INVALID_VALUE && call->calls == 0 &&
               owner.acknowledge_result(accepted.receipt) &&
               frontier.frontier.observe() == CUDA_ERROR_INVALID_VALUE &&
               owner.inspect_frontier_dependency(frontier.frontier).status ==
                   CUDA_ERROR_INVALID_VALUE &&
               owner.progress_frontier_dependency(frontier.frontier).state ==
                   CudaExecutionIssueState::kError &&
               driver.visits.size() == visits,
           "known rejection keeps exact receipt/failure and never invokes/replays the event");
    const auto failed_origin = target.inspect_enqueue_dependency(turn.turn);
    expect(failed_origin.status == CUDA_ERROR_INVALID_VALUE && failed_origin.foreign_blocker &&
               failed_origin.foreign_blocker->same_capture(frontier.frontier) &&
               failed_origin.foreign_blocker->retirement().is_retired &&
               !failed_origin.has_failure_observation && !failed_origin.has_retired_failure &&
               !waiting.frontier.retirement().is_retired,
           "failed old foreign origin remains authenticated, independently of target retirement");
    const auto rejected_wait = target.issue_next();
    const auto retained = target.inspect_enqueue_dependency(turn.turn);
    expect(rejected_wait.state == CudaExecutionIssueState::kRejected &&
               rejected_wait.sequence == 1 && retained.status == CUDA_ERROR_INVALID_VALUE &&
               retained.has_failure_observation && retained.has_retired_failure &&
               retained.foreign_blocker &&
               retained.foreign_blocker->same_capture(frontier.frontier) &&
               target.enqueue_turn_ready(turn.turn) == CUDA_ERROR_INVALID_VALUE &&
               target.begin_enqueue_turn(turn.turn) == CUDA_ERROR_INVALID_VALUE &&
               target.acknowledge_result(waiting.acceptance.receipt) && wait->calls == 0 &&
               waiting.frontier.retirement().status == CUDA_ERROR_INVALID_VALUE &&
               waiting.frontier.retirement().is_retired && driver.visits.size() == visits &&
               target.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS && target.shutdown() &&
               owner.shutdown(),
           "failed foreign observation rejects the actual wait without fabricating completion");
}

void erased_transitive_predecessor_preserves_exact_failure() {
    for (const auto failure :
         {CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN, CUDA_ERROR_OUT_OF_MEMORY}) {
        reset();
        current = reinterpret_cast<CUcontext>(&driver.context);
        auto lifetime = std::make_shared<int>(0);
        CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime}, dispatch().execution);
        expect(owner.bind_executor() == CUDA_SUCCESS &&
                   owner.bind_stream(1, reinterpret_cast<CUstream>(&driver.streams[0]), lifetime,
                                     true) &&
                   owner.bind_stream(2, reinterpret_cast<CUstream>(&driver.streams[1]), lifetime,
                                     true),
               "independent local event sources bound");
        auto unrelated = std::make_shared<DependencyEvent>();
        auto source = std::make_shared<DependencyEvent>();
        auto wait = std::make_shared<DependencyEvent>();
        auto later = std::make_shared<DependencyEvent>();
        const auto independent = owner.accept_event_with_frontier(1, {2, unrelated, {}});
        const auto producer = owner.accept_event_with_frontier(2, {1, source, {}});
        const auto target = owner.accept_event_with_frontier(3, {0, wait, producer.frontier});
        const auto turn = owner.accept_enqueue_turn(0, lifetime);
        const auto successor = owner.accept_event_with_frontier(4, {0, later, {}});
        expect(independent.acceptance.accepted() && producer.acceptance.accepted() &&
                   target.acceptance.accepted() && turn.status == CUDA_SUCCESS &&
                   successor.acceptance.accepted(),
               "turn retains exact transitive event roots, excluding independent and later work");
        unrelated->dependency_failure = CUDA_ERROR_NOT_SUPPORTED;
        expect(owner.issue_next().sequence == 1 &&
                   owner.acknowledge_result(independent.acceptance.receipt) &&
                   owner.inspect_enqueue_dependency(turn.turn).status == CUDA_ERROR_NOT_READY &&
                   !owner.inspect_enqueue_dependency(turn.turn).has_failure_observation,
               "erased earlier independent error cannot contaminate this accepted cut");
        source->dependency_failure = failure;
        const auto rejected = owner.issue_next();
        expect(rejected.state == CudaExecutionIssueState::kRejected && rejected.sequence == 2 &&
                   rejected.native_status == failure &&
                   owner.acknowledge_result(producer.acceptance.receipt) &&
                   owner.snapshot().queued == 2 && source->calls == 0,
               "ordinary scheduling rejects and erases the real ancestor before actor inspection");
        const auto visits = driver.visits.size();
        const auto inspected = owner.inspect_enqueue_dependency(turn.turn);
        const auto through_frontier = owner.inspect_frontier_dependency(turn.frontier);
        expect(inspected.status == failure && inspected.has_failure_observation &&
                   inspected.has_retired_failure && !inspected.foreign_blocker &&
                   !inspected.has_progress && !through_frontier.has_progress &&
                   through_frontier.status == failure && through_frontier.has_retired_failure &&
                   owner.enqueue_turn_ready(turn.turn) == failure &&
                   owner.begin_enqueue_turn(turn.turn) == failure &&
                   owner.progress_enqueue_dependency(turn.turn).native_status == failure &&
                   !target.frontier.retirement().is_retired &&
                   !turn.frontier.retirement().is_retired &&
                   !successor.frontier.retirement().is_retired &&
                   owner.started_external_count() == 0 && driver.visits.size() == visits &&
                   wait->calls == 0 && later->calls == 0,
               "known settled ancestor failure survives erasure without granting whole-cut stop");
        std::thread foreign([&] {
            const auto invalid = owner.inspect_enqueue_dependency(turn.turn);
            expect(invalid.status == CUDA_ERROR_INVALID_CONTEXT &&
                       !invalid.has_failure_observation && !invalid.has_retired_failure &&
                       !invalid.has_progress,
                   "foreign thread cannot mint an otherwise real settled-failure proof");
        });
        foreign.join();
        expect(owner.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS &&
                   turn.frontier.retirement().is_retired,
               "unbegun failed cut alone cancels without any SDK callbacks");
        const auto rejected_target = owner.issue_next();
        expect(rejected_target.state == CudaExecutionIssueState::kRejected &&
                   rejected_target.sequence == 3 &&
                   owner.acknowledge_result(target.acceptance.receipt) && wait->calls == 0,
               "retained actual target event still has its independent ordinary error receipt");
        expect(owner.issue_next().sequence == 4 &&
                   owner.acknowledge_result(successor.acceptance.receipt) && later->calls == 1,
               "later ordinary work is not rejected or replayed with the failed cut");
        driver.stop = true;
        expect(owner.poll().retired == 1 && owner.shutdown(),
               "known rejected predecessor controls cleanly retire all actual ownership");
    }
}

class RejectedProgress final : public CudaExecutionPersistentProgress {
   public:
    CUresult prepare_launch() noexcept override {
        ++preparations;
        return failure;
    }
    CUresult request_retreat() noexcept override {
        return CUDA_SUCCESS;
    }
    CUresult record_readback(CUstream) noexcept override {
        expect(false, "preissue rejection cannot enqueue persistent readback");
        return CUDA_ERROR_UNKNOWN;
    }
    CudaExecutionPersistentStop observe_physical_stop() noexcept override {
        expect(false, "preissue rejection cannot invent persistent physical work");
        return {CUDA_ERROR_UNKNOWN, false};
    }
    CUresult failure{CUDA_SUCCESS};
    unsigned preparations{};
};

void erased_ordinary_rejections_preserve_exact_failure() {
    enum class Operation { kMemory, kKernel, kPersistent };
    for (const auto operation : {Operation::kMemory, Operation::kKernel, Operation::kPersistent}) {
        for (const bool is_queued_rejection : {false, true}) {
            for (const auto failure :
                 {CUDA_ERROR_INVALID_VALUE, CUDA_ERROR_UNKNOWN, CUDA_ERROR_OUT_OF_MEMORY}) {
                reset();
                current = reinterpret_cast<CUcontext>(&driver.context);
                auto lifetime = std::make_shared<int>(0);
                CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime},
                                              dispatch().execution);
                expect(owner.bind_executor() == CUDA_SUCCESS &&
                           owner.bind_function(1, reinterpret_cast<CUfunction>(&driver.context), 0,
                                               lifetime),
                       "ordinary rejection owner and original kernel bound");
                const CudaExecutionKernel kernel{.function = 1,
                                                 .stream = 0,
                                                 .grid = {8, 1, 1},
                                                 .block = {32, 1, 1},
                                                 .parameters = {}};
                auto progress = std::make_shared<RejectedProgress>();
                ExecutionAcceptance accepted;
                if (operation == Operation::kMemory)
                    accepted = owner.accept_memory(
                        1, {.kind = CudaMemoryKind::kFill8,
                            .destination = reinterpret_cast<CUdeviceptr>(driver.memory.data()),
                            .count = 8,
                            .value = 5,
                            .is_asynchronous = true});
                else if (operation == Operation::kKernel)
                    accepted = owner.accept_kernel(1, kernel);
                else
                    accepted = owner.accept_persistent_kernel(
                        1, kernel,
                        {reinterpret_cast<CUfunction>(&driver.context), 2, {}, progress});
                const auto original = owner.capture_command_frontier(1);
                const auto turn = owner.accept_enqueue_turn(0, lifetime);
                expect(accepted.accepted() && original.status == CUDA_SUCCESS &&
                           turn.status == CUDA_SUCCESS,
                       "ordinary predecessor and exact dependent turn accepted before rejection");
                progress->failure = failure;
                driver.event_create_status = failure;
                if (is_queued_rejection) {
                    expect(owner.reject_queued(failure) == 1,
                           "queued ordinary predecessor rejected and erased exactly once");
                } else {
                    const auto rejected = owner.issue_next();
                    expect(rejected.state == CudaExecutionIssueState::kRejected &&
                               rejected.sequence == 1 && rejected.native_status == failure &&
                               rejected.completion_status == CUDA_SUCCESS,
                           "original ordinary preparation error preserves exact rejection receipt");
                }
                const auto receipt = owner.host_result(accepted.receipt);
                const auto inspected = owner.inspect_enqueue_dependency(turn.turn);
                const auto captured = owner.inspect_frontier_dependency(turn.frontier);
                const auto retired = original.frontier.retirement();
                expect(
                    receipt && receipt->status == failure &&
                        receipt->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                        owner.acknowledge_result(accepted.receipt) && inspected.status == failure &&
                        inspected.has_failure_observation && inspected.has_retired_failure &&
                        captured.status == failure && captured.has_failure_observation &&
                        captured.has_retired_failure && retired.status == failure &&
                        retired.is_retired && owner.enqueue_turn_ready(turn.turn) == failure &&
                        owner.begin_enqueue_turn(turn.turn) == failure &&
                        owner.progress_enqueue_dependency(turn.turn).native_status == failure &&
                        !turn.frontier.retirement().is_retired &&
                        owner.started_external_count() == 0 && !owner.snapshot().is_quarantined &&
                        driver.visits.empty() && !driver.event_destroys &&
                        progress->preparations ==
                            (!is_queued_rejection && operation == Operation::kPersistent ? 1U
                                                                                         : 0U) &&
                        driver.event_creates ==
                            (!is_queued_rejection && operation != Operation::kPersistent ? 1U : 0U),
                    "erased memory/kernel rejection remains exact settled failure without "
                    "requested native work, whole-turn retirement or retry");
                expect(owner.reject_queued(CUDA_ERROR_NOT_SUPPORTED) == 0 &&
                           owner.inspect_enqueue_dependency(turn.turn).status == failure &&
                           owner.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS && owner.shutdown(),
                       "later rejection cannot replace the retained original failure; only the "
                       "unbegun turn cancels");
            }
        }
    }
}

void pending_native_failure_is_not_a_clean_rejection() {
    for (const bool admission_after_issue : {false, true}) {
        reset();
        current = reinterpret_cast<CUcontext>(&driver.context);
        auto lifetime = std::make_shared<int>(0);
        CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime}, dispatch().execution);
        expect(owner.bind_executor() == CUDA_SUCCESS, "pending error owner bound");
        auto call = std::make_shared<DependencyEvent>();
        const auto accepted = owner.accept_event_with_frontier(1, {0, call, {}});
        CudaExecutionEnqueueAcceptance turn;
        if (!admission_after_issue) turn = owner.accept_enqueue_turn(0, lifetime);
        call->issue_status = CUDA_ERROR_INVALID_VALUE;
        expect(accepted.acceptance.accepted() &&
                   (admission_after_issue || turn.status == CUDA_SUCCESS),
               "event and optional earlier dependent turn accepted before actual native error");
        const auto issued = owner.issue_next();
        expect(issued.state == CudaExecutionIssueState::kIssued &&
                   issued.native_status == CUDA_ERROR_INVALID_VALUE && call->calls == 1 &&
                   owner.acknowledge_result(accepted.acceptance.receipt),
               "requested event error has a real trailing event, not no-native rejection");
        if (admission_after_issue) turn = owner.accept_enqueue_turn(0, lifetime);
        expect(turn.status == CUDA_SUCCESS, "exact turn admitted on either side of original issue");
        const auto pending = owner.inspect_enqueue_dependency(turn.turn);
        expect(
            pending.status == CUDA_ERROR_INVALID_VALUE && pending.has_failure_observation &&
                !pending.has_retired_failure && !pending.has_progress &&
                !accepted.frontier.retirement().is_retired &&
                owner.begin_enqueue_turn(turn.turn) == CUDA_ERROR_INVALID_VALUE &&
                owner.started_external_count() == 0 && !owner.snapshot().is_quarantined,
            "known native-error observation remains pending until its healthy tail actually stops");
        unsigned proof_visits{};
        driver.proof = [&] {
            ++proof_visits;
            const auto reentrant = owner.inspect_enqueue_dependency(turn.turn);
            expect(reentrant.status == CUDA_ERROR_NOT_READY && !reentrant.has_failure_observation &&
                       !reentrant.has_retired_failure && !reentrant.has_progress,
                   "reentry cannot reuse a retained failure as fresh clean proof");
        };
        expect(owner.poll().retired == 0 && proof_visits != 0,
               "unready tail remains retained without stream synchronization");
        driver.proof = {};
        driver.stop = true;
        expect(owner.poll().retired == 1, "actual event stop retires the failed native record");
        const auto settled = owner.inspect_enqueue_dependency(turn.turn);
        expect(settled.status == CUDA_ERROR_INVALID_VALUE && settled.has_failure_observation &&
                   settled.has_retired_failure && !settled.has_progress &&
                   accepted.frontier.retirement().is_retired && call->calls == 1 &&
                   owner.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS && owner.shutdown(),
               "only observed native retirement allows grant-bound clean cancellation");
    }
}

void atomic_event_frontiers_and_retirement() {
    reset();
    current = reinterpret_cast<CUcontext>(&driver.context);
    auto lifetime = std::make_shared<int>(0);
    auto owner = std::make_unique<CudaExecutionDispatcher>(
        ExecutionSessionIdentity{11, 12, 1}, CudaExecutionContext{13, current, lifetime},
        dispatch().execution, ExecutionSubmissionLimits{1, 1024}, 1);
    expect(owner->bind_executor() == CUDA_SUCCESS, "atomic event owner bound");
    auto occupied = owner->capture_frontier();
    auto event = std::make_shared<DependencyEvent>();
    const CudaExecutionEvent command{0, event, {}};
    const auto refused = owner->accept_event_with_frontier(1, command);
    expect(occupied.status == CUDA_SUCCESS && occupied.frontier.retirement().is_retired &&
               refused.acceptance.status == ExecutionAcceptStatus::kOutOfMemory &&
               refused.frontier.retirement().status == CUDA_ERROR_INVALID_HANDLE &&
               !refused.frontier.retirement().is_retired && owner->snapshot().outstanding == 0 &&
               event->calls == 0,
           "frontier quota rejects before command acceptance, sequence use or native issue");
    occupied.frontier = {};
    auto accepted = owner->accept_event_with_frontier(1, command);
    expect(accepted.acceptance.accepted() &&
               accepted.frontier.retirement().status == CUDA_ERROR_NOT_READY &&
               !accepted.frontier.retirement().is_retired &&
               owner->accept_event_with_frontier(2, command).acceptance.status ==
                   ExecutionAcceptStatus::kFull,
           "same-sequence retry succeeds and bounded command exhaustion remains transactional");
    std::thread foreign([&] {
        expect(accepted.frontier.retirement().status == CUDA_ERROR_INVALID_CONTEXT &&
                   !accepted.frontier.retirement().is_retired,
               "foreign CPU thread never acquires retirement proof");
    });
    foreign.join();
    const auto child = fork();
    expect(child >= 0, "fork retirement proof control");
    if (!child) {
        expect(accepted.frontier.retirement().status == CUDA_ERROR_INVALID_CONTEXT &&
                   !accepted.frontier.retirement().is_retired,
               "inherited frontier fails before thread or callback authority");
        std::_Exit(0);
    }
    int child_status{};
    expect(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "fork retirement proof control completes");
    unsigned reentries{};
    driver.proof = [&] {
        ++reentries;
        const auto observed = accepted.frontier.retirement();
        expect(observed.status == CUDA_ERROR_NOT_READY && !observed.is_retired,
               "SDK callback cannot publish a reentrant retirement proof");
    };
    expect(owner->issue_next().sequence == 1 && event->calls == 1 && reentries != 0 &&
               owner->acknowledge_result(accepted.acceptance.receipt),
           "first requested event issues once under callback exclusion");
    driver.proof = {};
    driver.stop = true;
    expect(owner->poll().retired == 1 && accepted.frontier.retirement().is_retired &&
               owner->accept_event_with_frontier(2, command).acceptance.status ==
                   ExecutionAcceptStatus::kOutOfMemory &&
               owner->snapshot().outstanding == 0,
           "retired but live capture still consumes its bounded metadata slot only");
    accepted.frontier = {};
    auto retried = owner->accept_event_with_frontier(2, command);
    expect(retried.acceptance.accepted() && owner->issue_next().sequence == 2 &&
               owner->acknowledge_result(retried.acceptance.receipt) &&
               owner->poll().retired == 1 && retried.frontier.retirement().is_retired &&
               owner->shutdown(),
           "both command and frontier quota failures preserve exact retry sequence");
    owner.reset();
    expect(retried.frontier.retirement().status == CUDA_ERROR_INVALID_HANDLE &&
               !retried.frontier.retirement().is_retired,
           "destroyed dispatcher cannot vouch for retained observation tokens");
}

void retirement_scans_all_captured_records() {
    reset();
    current = reinterpret_cast<CUcontext>(&driver.context);
    auto lifetime = std::make_shared<int>(0);
    CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime}, dispatch().execution);
    expect(owner.bind_executor() == CUDA_SUCCESS, "multi-record retirement owner bound");
    auto first = std::make_shared<DependencyEvent>();
    auto second = std::make_shared<DependencyEvent>();
    auto later = std::make_shared<DependencyEvent>();
    const auto one = owner.accept_event_with_frontier(1, {0, first, {}});
    const auto two = owner.accept_event_with_frontier(2, {0, second, {}});
    const auto both = owner.capture_frontier();
    const auto three = owner.accept_event_with_frontier(3, {0, later, {}});
    expect(one.acceptance.accepted() && two.acceptance.accepted() && three.acceptance.accepted() &&
               both.status == CUDA_SUCCESS,
           "immutable capture excludes later accepted record");
    first->dependency_failure = CUDA_ERROR_INVALID_VALUE;
    expect(owner.progress_frontier_dependency(one.frontier).state ==
                   CudaExecutionIssueState::kRejected &&
               owner.acknowledge_result(one.acceptance.receipt) &&
               one.frontier.retirement().is_retired &&
               both.frontier.observe() == CUDA_ERROR_INVALID_VALUE &&
               both.frontier.retirement().status == CUDA_ERROR_INVALID_VALUE &&
               !both.frontier.retirement().is_retired,
           "first known rejection does not hide a later captured pending record");
    second->dependency_failure = CUDA_ERROR_NOT_SUPPORTED;
    expect(owner.progress_frontier_dependency(two.frontier).native_status ==
                   CUDA_ERROR_INVALID_VALUE &&
               owner.inspect_frontier_dependency(two.frontier).has_retired_failure,
           "dependency-success progression still refuses the exact settled ancestor failure");
    const auto second_rejected = owner.issue_next();
    expect(second_rejected.state == CudaExecutionIssueState::kRejected &&
               second_rejected.sequence == 2 &&
               second_rejected.native_status == CUDA_ERROR_NOT_SUPPORTED &&
               owner.acknowledge_result(two.acceptance.receipt) &&
               both.frontier.retirement().status == CUDA_ERROR_INVALID_VALUE &&
               both.frontier.retirement().is_retired &&
               two.frontier.retirement().status == CUDA_ERROR_NOT_SUPPORTED &&
               !three.frontier.retirement().is_retired && first->calls == 0 && second->calls == 0,
           "all-record stop is independent of first-error precedence and later membership");
    unsigned reentries{};
    driver.proof = [&] {
        ++reentries;
        expect(both.frontier.observe() == CUDA_ERROR_INVALID_VALUE &&
                   both.frontier.retirement().status == CUDA_ERROR_NOT_READY &&
                   !both.frontier.retirement().is_retired,
               "dispatcher callback cannot reuse even an already retired failure proof");
    };
    expect(owner.issue_next().sequence == 3 && reentries != 0 &&
               owner.acknowledge_result(three.acceptance.receipt),
           "uncaptured successor remains independently executable");
    driver.proof = {};
    driver.stop = true;
    expect(owner.poll().retired == 1 && owner.shutdown(), "retirement controls clean up exactly");
}

void session_event_admission_is_transactional() {
    reset();
    CudaSessionLimits limits;
    limits.max_objects = 2;
    limits.submissions.max_commands = 1;
    CudaExecutionSession owner({11, 12, 1}, 13, dispatch(), limits);
    success(owner.initialize(0));
    const auto event = owner.create_event(0);
    success(event);
    std::vector<CudaExecutionFrontier> occupied;
    occupied.reserve(64);
    CUresult exhausted{CUDA_SUCCESS};
    while (occupied.size() != 64) {
        auto captured = owner.capture_frontier();
        if (captured.status != CUDA_SUCCESS) {
            exhausted = captured.status;
            break;
        }
        occupied.push_back(std::move(captured.frontier));
    }
    expect(!occupied.empty() && exhausted == CUDA_ERROR_OUT_OF_MEMORY,
           "bounded independent capture quota is filled before record admission");
    const auto failed = owner.record_event_with_frontier(1, event.id, 0);
    auto initial = owner.capture_event(event.id);
    expect(failed.result.status == CUDA_ERROR_OUT_OF_MEMORY && !failed.result.original_status &&
               !failed.frontier.retirement().is_retired && !initial.dependency &&
               owner.snapshot().submissions.outstanding == 0 && driver.record_attempts == 0,
           "failed atomic record publishes neither a command nor a replacement generation");
    occupied.clear();
    // Release the initial native-generation borrow before the retry needs its
    // second bounded native event slot.
    initial.event = {};
    const auto record = owner.record_event_with_frontier(1, event.id, 0);
    success(record.result);
    auto captured = owner.capture_event(event.id);
    expect(captured.dependency && captured.dependency->same_capture(record.frontier),
           "record publication and returned target frontier are one atomic capture");
    const auto deferred = owner.wait_event_with_frontier(2, event.id, 0);
    expect(deferred.result.disposition == CudaSessionDisposition::kDeferred &&
               !deferred.result.original_status && !deferred.frontier.retirement().is_retired &&
               owner.snapshot().submissions.outstanding == 1 && driver.record_attempts == 0,
           "full event command capacity defers without consuming its next sequence");
    expect(owner.issue_next().sequence == 1, "atomic record retains the original request sequence");
    retire(owner);
    const auto wait = owner.wait_event_with_frontier(2, event.id, 0);
    success(wait.result);
    expect(!wait.frontier.same_capture(record.frontier) && !wait.frontier.retirement().is_retired &&
               record.frontier.retirement().is_retired && owner.issue_next().sequence == 2,
           "local wait retry gets its own exact target rather than the source event frontier");
    retire(owner);
    expect(wait.frontier.retirement().is_retired, "actual local wait stop establishes retirement");
    captured.event = {};
    unsigned factory_reentries{};
    driver.event_factory_proof = [&] {
        ++factory_reentries;
        expect(record.frontier.observe() == CUDA_SUCCESS &&
                   record.frontier.retirement().status == CUDA_ERROR_NOT_READY &&
                   !record.frontier.retirement().is_retired &&
                   owner.wait_event_with_frontier(3, event.id, 0).result.status ==
                       CUDA_ERROR_NOT_READY,
               "Session event factory excludes settled proof and recursive admission");
    };
    const auto rerecord = owner.record_event_with_frontier(3, event.id, 0);
    driver.event_factory_proof = {};
    success(rerecord.result);
    expect(factory_reentries == 1 && record.frontier.retirement().is_retired &&
               !rerecord.frontier.retirement().is_retired && owner.issue_next().sequence == 3,
           "settled observation resumes only after the complete Session callback scope");
    retire(owner);
    success(owner.destroy_event(event.id));
    success(owner.shutdown());
}

void event_lifecycle_excludes_retirement_reentry() {
    reset();
    auto owner = session();
    const auto event = owner->create_event(0);
    success(event);
    const auto recorded = owner->record_event_with_frontier(1, event.id, 0);
    success(recorded.result);
    expect(owner->issue_next().sequence == 1, "create a real completed event observation");
    retire(*owner);
    expect(recorded.frontier.retirement().is_retired,
           "completed source frontier is available outside lifecycle callbacks");
    unsigned proof_visits{}, factory_visits{}, destroy_visits{};
    const auto excluded = [&] {
        const auto retired = recorded.frontier.retirement();
        expect(recorded.frontier.observe() == CUDA_SUCCESS &&
                   retired.status == CUDA_ERROR_NOT_READY && !retired.is_retired &&
                   owner->inspect_frontier_retirement(recorded.frontier).status ==
                       CUDA_ERROR_NOT_READY &&
                   owner->progress_frontier_retirement(recorded.frontier).native_status ==
                       CUDA_ERROR_NOT_READY &&
                   owner->create_event(0).status == CUDA_ERROR_NOT_READY &&
                   owner->destroy_event(event.id).status == CUDA_ERROR_NOT_READY &&
                   owner->record_event_with_frontier(2, event.id, 0).result.status ==
                       CUDA_ERROR_NOT_READY,
               "event lifecycle callbacks cannot reuse settled proof or mutate the event ledger");
    };
    driver.proof = [&] {
        ++proof_visits;
        excluded();
    };
    driver.event_factory_proof = [&] {
        ++factory_visits;
        excluded();
    };
    const auto created = owner->create_event(0);
    driver.proof = {};
    driver.event_factory_proof = {};
    success(created);
    expect(proof_visits != 0 && factory_visits == 1 && recorded.frontier.retirement().is_retired,
           "CreateEvent excludes the initial proof and native factory until ledger publication");
    proof_visits = 0;
    driver.proof = [&] {
        ++proof_visits;
        excluded();
    };
    driver.event_destroy_proof = [&] {
        ++destroy_visits;
        excluded();
    };
    const auto destroyed = owner->destroy_event(created.id);
    driver.proof = {};
    driver.event_destroy_proof = {};
    success(destroyed);
    expect(proof_visits != 0 && destroy_visits == 1 && recorded.frontier.retirement().is_retired,
           "DestroyEvent excludes the initial proof and native cleanup until ledger retirement");
    const auto next = owner->record_event_with_frontier(2, event.id, 0);
    success(next.result);
    expect(owner->issue_next().sequence == 2,
           "reentrant event attempts consume neither public IDs nor command sequence");
    retire(*owner);
    success(owner->destroy_event(event.id));
    success(owner->shutdown());
}

void library_defaults_and_disabled_compatibility() {
    for (const bool enabled : {false, true}) {
        reset();
        auto owner = session(enabled);
        expect(driver.creates == (enabled ? 1U : 0U), "default-off creates no hidden stream");
        const auto blas = owner->create_blas();
        expect(
            blas.status == CUDA_SUCCESS && blas.id &&
                driver.blas_stream == reinterpret_cast<cudaStream_t>(enabled ? proxy() : nullptr),
            "BLAS native default is the configured logical default");
        const auto stream = owner->create_stream(CU_STREAM_NON_BLOCKING);
        success(stream);
        expect(
            owner->set_blas_stream(blas.id, stream.id).status == CUDA_SUCCESS &&
                owner->get_blas_attribute(blas.id, CudaBlasAttribute::kStream).value == stream.id,
            "explicit BLAS stream remains explicit");
        expect(owner->set_blas_stream(blas.id, 0).status == CUDA_SUCCESS &&
                   owner->get_blas_attribute(blas.id, CudaBlasAttribute::kStream).value == 0,
               "native proxy is projected to public stream zero");
#ifdef GLIMMER_HAS_CUDNN
        const auto dnn = owner->create_dnn();
        expect(dnn.status == CUDA_SUCCESS && dnn.id &&
                   driver.dnn_stream == reinterpret_cast<cudaStream_t>(enabled ? proxy() : nullptr),
               "DNN native default follows the same mapping");
        expect(owner->set_dnn_stream(dnn.id, stream.id).status == CUDA_SUCCESS &&
                   owner->get_dnn_stream(dnn.id).value == stream.id &&
                   owner->set_dnn_stream(dnn.id, 0).status == CUDA_SUCCESS &&
                   owner->get_dnn_stream(dnn.id).value == 0,
               "DNN Set/GetStream0 round trip");
        expect(owner->destroy_dnn(dnn.id).status == CUDA_SUCCESS, "DNN cleanup");
#endif
        expect(owner->destroy_blas(blas.id).status == CUDA_SUCCESS, "BLAS cleanup");
        success(owner->destroy_stream(stream.id));
        success(owner->shutdown());
        expect(driver.destroys == (enabled ? 2U : 1U), "proxy cleanup follows handles");
    }
}

class PausedProgress final : public CudaExecutionPersistentProgress {
   public:
    CUresult prepare_launch() noexcept override {
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        return CUDA_SUCCESS;
    }
    CUresult record_readback(CUstream) noexcept override {
        return CUDA_SUCCESS;
    }
    CudaExecutionPersistentStop observe_physical_stop() noexcept override {
        return {CUDA_SUCCESS, ++stops == 2};
    }
    unsigned stops{};
};
void paused_predecessor_and_capacity() {
    for (const bool through_frontier : {false, true}) {
        reset();
        current = reinterpret_cast<CUcontext>(&driver.context);
        auto lifetime = std::make_shared<int>(0);
        CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime}, dispatch().execution,
                                      {.max_commands = 2, .max_parameter_bytes = 1024});
        expect(
            owner.bind_executor() == CUDA_SUCCESS &&
                owner.bind_function(1, reinterpret_cast<CUfunction>(&driver.context), 0, lifetime),
            "direct dispatcher original binding");
        auto progress = std::make_shared<PausedProgress>();
        const auto admitted = owner.accept_persistent_kernel(
            1,
            {.function = 1, .stream = 0, .grid = {8, 1, 1}, .block = {32, 1, 1}, .parameters = {}},
            {reinterpret_cast<CUfunction>(&driver.context), 2, {}, progress});
        expect(admitted.accepted() && owner.issue_next().state == CudaExecutionIssueState::kIssued,
               "persistent predecessor first interval issued");
        expect(owner.acknowledge_result(admitted.receipt), "persistent host result acknowledged");
        const auto exact = owner.capture_command_frontier(1);
        expect(exact.status == CUDA_SUCCESS, "persistent original retains an exact command cut");
        driver.stop = true;
        expect(owner.poll().paused == 1, "real first stop leaves logical predecessor paused");
        const auto turn = owner.accept_enqueue_turn(0, lifetime);
        expect(turn.status == CUDA_SUCCESS &&
                   owner.enqueue_turn_ready(turn.turn) == CUDA_ERROR_NOT_READY &&
                   owner.accept_enqueue_turn(0, lifetime).status == CUDA_ERROR_OUT_OF_MEMORY,
               "paused predecessor holds turn and bounded prefix capacity");
        expect(owner.inspect_frontier_dependency(exact.frontier).status == CUDA_ERROR_NOT_READY &&
                   owner.inspect_frontier_dependency(exact.frontier).has_progress &&
                   owner.inspect_enqueue_dependency(turn.turn).has_progress &&
                   !owner.inspect_enqueue_dependency(turn.turn).foreign_blocker,
               "paused local dependency is not misclassified as a foreign or completed prefix");
        const auto resumed = through_frontier ? owner.progress_frontier_dependency(exact.frontier)
                                              : owner.progress_enqueue_dependency(turn.turn);
        expect(resumed.state == CudaExecutionIssueState::kIssued && resumed.sequence == 1 &&
                   owner.poll().retired == 1 && owner.enqueue_turn_ready(turn.turn) == CUDA_SUCCESS,
               "complete resumed logical predecessor alone releases readiness");
        expect(owner.cancel_enqueue_turn(turn.turn) == CUDA_SUCCESS && owner.shutdown(),
               "unused turn cancellation permits exact normal shutdown");
    }
}

void uncertainty_retains_without_retry() {
    const auto child = fork();
    expect(child >= 0, "isolated uncertain owner child");
    if (child == 0) {
        reset();
        auto owner = session();
        const auto memory = owner->allocate(32);
        success(memory);
        const auto blocking = owner->create_stream(CU_STREAM_DEFAULT);
        success(blocking);
        issue(*owner, fill(*owner, 1, memory.id));
        const auto turn = owner->accept_enqueue_turn(blocking.id, std::make_shared<int>(0));
        driver.wait_status = CUDA_ERROR_INVALID_VALUE;
        expect(owner->begin_enqueue_turn(turn.turn) == CUDA_ERROR_INVALID_VALUE &&
                   owner->snapshot().is_quarantined && owner->snapshot().enqueue_turns == 1 &&
                   owner->snapshot().external_work_in_flight == 1,
               "failed real dependency wait retains source owner before native packet");
        const auto visits = driver.visits.size();
        const auto cancelled = owner->cancel_enqueue_turn(turn.turn);
        const auto progressed = owner->progress_enqueue_dependency(turn.turn);
        const auto inspected = owner->inspect_enqueue_dependency(turn.turn);
        const auto frontier_progress = owner->progress_frontier_dependency(turn.frontier);
        const auto frontier_inspection = owner->inspect_frontier_dependency(turn.frontier);
        const auto retirement = turn.frontier.retirement();
        const auto stopped = owner->shutdown();
        const auto snapshot = owner->snapshot();
        const bool is_retained =
            cancelled != CUDA_SUCCESS && progressed.state == CudaExecutionIssueState::kError &&
            progressed.native_status == CUDA_ERROR_UNKNOWN &&
            inspected.status == CUDA_ERROR_UNKNOWN && !inspected.has_failure_observation &&
            !inspected.has_retired_failure && !inspected.has_progress &&
            frontier_progress.state == CudaExecutionIssueState::kError &&
            frontier_progress.native_status == CUDA_ERROR_UNKNOWN &&
            frontier_inspection.status == CUDA_ERROR_UNKNOWN && !frontier_inspection.has_progress &&
            !retirement.is_retired && stopped.status != CUDA_SUCCESS &&
            driver.visits.size() == visits && driver.destroys == 0 && driver.releases == 0 &&
            snapshot.external_work_in_flight == 1;
        if (!is_retained)
            std::cerr << "uncertain prefix diagnostics: cancelled=" << cancelled
                      << "; progress_state=" << static_cast<unsigned>(progressed.state)
                      << "; progress_native=" << progressed.native_status
                      << "; inspect=" << inspected.status
                      << "; frontier_state=" << static_cast<unsigned>(frontier_progress.state)
                      << "; frontier_native=" << frontier_progress.native_status
                      << "; frontier_inspect=" << frontier_inspection.status
                      << "; retirement_status=" << retirement.status
                      << "; is_retired=" << retirement.is_retired << "; shutdown=" << stopped.status
                      << "; visits_before=" << visits << "; visits_after=" << driver.visits.size()
                      << "; destroys=" << driver.destroys << "; releases=" << driver.releases
                      << "; enqueue_turns=" << snapshot.enqueue_turns
                      << "; external_work_in_flight=" << snapshot.external_work_in_flight
                      << "; quarantined=" << snapshot.is_quarantined << '\n';
        expect(is_retained, "uncertain prefix never retries or speculatively frees owners");
        std::_Exit(0);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "uncertainty retention child complete");
}
void rejected_preparation_keeps_cleanup_uncertainty() {
    reset();
    current = reinterpret_cast<CUcontext>(&driver.context);
    auto lifetime = std::make_shared<int>(0);
    auto sdk = dispatch().execution;
    sdk.event_create = [](CUevent* output, unsigned int) {
        ++driver.event_creates;
        driver.events[0] = {true, false, nullptr};
        *output = reinterpret_cast<CUevent>(&driver.events[0]);
        return CUDA_ERROR_INVALID_VALUE;
    };
    sdk.event_destroy = [](CUevent event) {
        expect(event == reinterpret_cast<CUevent>(&driver.events[0]),
               "cleanup attempts only its actual partial factory output");
        ++driver.event_destroys;
        return CUDA_ERROR_INVALID_CONTEXT;
    };
    CudaExecutionDispatcher owner({11, 12, 1}, {13, current, lifetime}, sdk);
    expect(owner.bind_executor() == CUDA_SUCCESS, "partial factory failure owner bound");
    const auto accepted =
        owner.accept_memory(1, {.kind = CudaMemoryKind::kFill8,
                                .destination = reinterpret_cast<CUdeviceptr>(driver.memory.data()),
                                .count = 8,
                                .value = 5,
                                .is_asynchronous = true});
    const auto captured = owner.capture_command_frontier(1);
    const auto turn = owner.accept_enqueue_turn(0, lifetime);
    expect(accepted.accepted() && captured.status == CUDA_SUCCESS && turn.status == CUDA_SUCCESS,
           "partial factory failure starts with an exact ordinary predecessor");
    const auto issued = owner.issue_next();
    const auto receipt = owner.host_result(accepted.receipt);
    const auto inspection = owner.inspect_enqueue_dependency(turn.turn);
    expect(issued.state == CudaExecutionIssueState::kRejected &&
               issued.native_status == CUDA_ERROR_INVALID_VALUE && receipt &&
               receipt->status == CUDA_ERROR_INVALID_VALUE &&
               receipt->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               owner.acknowledge_result(accepted.receipt) &&
               captured.frontier.observe() == CUDA_ERROR_INVALID_CONTEXT &&
               !captured.frontier.retirement().is_retired &&
               inspection.status == CUDA_ERROR_UNKNOWN && !inspection.has_failure_observation &&
               !inspection.has_retired_failure && owner.snapshot().is_quarantined &&
               owner.progress_enqueue_dependency(turn.turn).native_status == CUDA_ERROR_UNKNOWN &&
               !owner.shutdown() && driver.event_creates == 1 && driver.event_destroys == 1 &&
               driver.visits.empty() && driver.events[0].alive,
           "cleanup uncertainty remains stronger than the unchanged preparation-error receipt; "
           "neither error permits settled proof, native entry or repeated cleanup");
}

template <typename Function>
void isolated_failure(Function function) {
    const auto child = fork();
    expect(child >= 0, "create isolated native/proof failure control");
    if (!child) {
        function();
        std::_Exit(0);
    }
    int status{};
    expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated native/proof failure control passed");
}
void synchronous_native_failure(unsigned operation, bool loses_context, CUresult original) {
    reset();
    auto owner = session(true, 16);
    const auto memory = owner->allocate(32);
    success(memory);
    std::array<std::byte, 16> input{}, output{};
    input.fill(std::byte{0x5a});
    output.fill(std::byte{0x33});
    driver.memory.fill(std::byte{0x6b});
    driver.native_status = original;
    driver.loses_native_context = loses_context;
    CudaExecutionFrontier synchronous_frontier;
    if (operation < 2) {
        const auto result = operation == 0 ? owner->upload({memory.id, 0}, input)
                                           : owner->download({memory.id, 0}, output);
        expect(result.original_status == original,
               "synchronous upload/download preserves actual requested native error or success");
        synchronous_frontier = result.frontier;
        expect(synchronous_frontier.observe() ==
                       (loses_context ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_NOT_READY) &&
                   !synchronous_frontier.retirement().is_retired,
               "original error and late proof loss both preserve exact unretired private tail");
        expect(result.status == (loses_context ? CUDA_ERROR_INVALID_CONTEXT : original) &&
                   result.disposition == (loses_context ? CudaSessionDisposition::kRejected
                                                        : CudaSessionDisposition::kComplete),
               "later context proof failure is a separate carrier, not a fabricated original");
        expect(driver.uploads == (operation == 0 ? 1U : 0U) &&
                   driver.downloads == (operation == 1 ? 1U : 0U),
               "only the requested synchronous callback runs exactly once");
        expect(owner->snapshot().default_proxy_staging_bytes == input.size() &&
                   owner->snapshot().enqueue_turns == 1 &&
                   owner->snapshot().external_work_in_flight == 1,
               "owned staging and exact posted-prefix record survive host return/error");
        if (operation == 0) {
            expect(driver.upload_source && driver.upload_source != input.data(),
                   "original HtoD receives framework-owned storage, never borrowed request bytes");
            input.fill(std::byte{0x7c});
            expect(std::ranges::all_of(
                       std::span{static_cast<const std::byte*>(driver.upload_source), input.size()},
                       [](auto value) { return value == std::byte{0x5a}; }),
                   "retained native staging remains intact after caller input changes");
        } else {
            expect(driver.download_destination && driver.download_destination != output.data() &&
                       std::ranges::all_of(output,
                                           [](auto value) { return value == std::byte{0x33}; }),
                   "native error or lost proof cannot publish staged DtoH bytes to the caller");
        }
    } else {
        const auto receipt = fill(*owner, 1, memory.id, 0, false);
        const auto issued = owner->issue_next();
        const auto reply = owner->host_result(receipt);
        expect(issued.state == CudaExecutionIssueState::kIssued &&
                   issued.native_status == original && reply && reply->status == original &&
                   owner->acknowledge_result(receipt),
               "synchronous fill preserves original status despite later post-call proof failure");
        expect(!loses_context || issued.completion_status != CUDA_SUCCESS,
               "lost context proof cannot report successful physical completion status");
        expect(driver.fills == 1 && owner->snapshot().submissions.outstanding == 1,
               "failed/lost-proof synchronous fill keeps its actual native ownership record");
    }
    if (loses_context) {
        expect(owner->snapshot().is_quarantined,
               "actual post-call context loss quarantines source");
        const auto records = driver.record_attempts;
        const auto queries = driver.queries;
        const auto visits = driver.visits.size();
        driver.stop = true;
        expect(owner->poll().status != CUDA_SUCCESS && owner->shutdown().status != CUDA_SUCCESS &&
                   owner->release(memory.id).status != CUDA_SUCCESS &&
                   driver.record_attempts == records && driver.queries == queries &&
                   driver.visits.size() == visits && driver.event_destroys == 0 &&
                   driver.releases == 0 && driver.destroys == 0,
               "lost proof never retries the native callback, records a false tail, queries, or "
               "frees");
        if (operation < 2)
            expect(owner->snapshot().default_proxy_staging_bytes == input.size(),
                   "unknown physical ownership cannot refund retained staging charge");
        return;  // Intentional retained uncertainty is contained in this child.
    }
    retire(*owner);
    if (operation < 2)
        expect(synchronous_frontier.observe() == CUDA_SUCCESS &&
                   synchronous_frontier.retirement().is_retired,
               "healthy native error is independent of real successful tail retirement");
    expect(owner->snapshot().default_proxy_staging_bytes == 0,
           "known native error still refunds staging only after actual exact event stop");
    success(owner->release(memory.id));
    success(owner->shutdown());
}
void cohort_second_event_failure(bool query_failure) {
    reset();
    auto owner = session();
    const auto blocking = owner->create_stream(CU_STREAM_DEFAULT);
    success(blocking);
    auto resource = std::make_shared<int>(91);
    const std::weak_ptr<int> retained = resource;
    const std::array streams{std::uint64_t{0}, blocking.id};
    auto turn = owner->accept_enqueue_cohort(streams, resource);
    expect(turn.status == CUDA_SUCCESS && owner->begin_enqueue_turn(turn.turn) == CUDA_SUCCESS,
           "two-stream exact cohort source turn begun");
    CudaNativeExecutionLanes lanes({set_current, get_current});
    auto call =
        lanes.submit({21, {25, 0}}, {.sequence = 1,
                                     .device_ordinal = 0,
                                     .context = {13, current, resource},
                                     .arguments_lifetime = resource,
                                     .invoke = [worker = turn.worker] { return worker.begin(); }});
    expect(call.status == CUDA_SUCCESS && turn.worker.bind_ticket(call.ticket) == CUDA_SUCCESS,
           "actual exact cohort ticket bound");
    await([&] { return call.ticket.observe().phase == CudaNativeExecutionCallPhase::kReturned; });
    if (!query_failure) driver.fail_record_attempt = driver.record_attempts + 2;
    expect(owner->finish_enqueue_turn(turn.turn) ==
               (query_failure ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE),
           "second recorded member failure cannot publish a complete cohort");
    expect(lanes.release(call.ticket) == CUDA_SUCCESS, "release known returned host ticket only");
    call.ticket = {};
    stop(lanes);
    turn.worker = {};
    resource.reset();
    expect(!retained.expired(),
           "source record independently retains complete cohort resource owner");
    driver.stop = true;
    if (query_failure) driver.fail_query_attempt = driver.queries + 2;
    const auto polled = owner->poll();
    expect(polled.status == (query_failure ? CUDA_ERROR_INVALID_CONTEXT : CUDA_ERROR_NOT_READY) &&
               owner->snapshot().is_quarantined && owner->snapshot().enqueue_turns == 1 &&
               owner->snapshot().external_work_in_flight == 1 &&
               turn.frontier.observe() != CUDA_SUCCESS && !retained.expired(),
           "first completed event is never false whole-cohort physical completion");
    expect(
        driver.queries == (query_failure ? 2U : 0U) && driver.event_destroys == 0 &&
            std::ranges::count(driver.events, true, &Event::alive) == 2,
        "all native cohort events remain owned, and an unrecorded second event is never queried");
    const auto records = driver.record_attempts;
    const auto queries = driver.queries;
    expect(owner->finish_enqueue_turn(turn.turn) != CUDA_SUCCESS &&
               owner->cancel_enqueue_turn(turn.turn) != CUDA_SUCCESS &&
               owner->shutdown().status != CUDA_SUCCESS && driver.record_attempts == records &&
               driver.queries == queries && driver.event_destroys == 0 && driver.destroys == 0 &&
               !retained.expired(),
           "cohort proof failure retains every owner without SDK retry or speculative cleanup");
}
}  // namespace

int main() {
    actual_worker_and_posted_prefix();
    started_external_reservations_are_not_acceptance();
    logical_edges_and_native_synchronous_calls();
    session_reservations_preserve_owner_and_admission();
    session_retirement_progress_keeps_exact_cut();
    same_session_cohort_and_staging_limit();
    queued_predecessor_cancellation_and_owner_gates();
    exact_dependency_cut_excludes_unrelated_work();
    event_dependency_closure_and_foreign_owner();
    chained_session_generations_survive_rerecord_and_close();
    failed_dependency_inspection_is_not_completion();
    erased_transitive_predecessor_preserves_exact_failure();
    erased_ordinary_rejections_preserve_exact_failure();
    pending_native_failure_is_not_a_clean_rejection();
    atomic_event_frontiers_and_retirement();
    retirement_scans_all_captured_records();
    session_event_admission_is_transactional();
    event_lifecycle_excludes_retirement_reentry();
    library_defaults_and_disabled_compatibility();
    paused_predecessor_and_capacity();
    uncertainty_retains_without_retry();
    for (unsigned operation = 0; operation != 3; ++operation) {
        isolated_failure([operation] {
            synchronous_native_failure(operation, false, CUDA_ERROR_INVALID_VALUE);
        });
        for (const auto original : {CUDA_SUCCESS, CUDA_ERROR_INVALID_VALUE})
            isolated_failure(
                [operation, original] { synchronous_native_failure(operation, true, original); });
    }
    isolated_failure([] { cohort_second_event_failure(false); });
    isolated_failure([] { cohort_second_event_failure(true); });
    isolated_failure(rejected_preparation_keeps_cleanup_uncertainty);
    std::cout << "Source enqueue/proxy groups passed: 21; isolated native/cohort controls: 12\n";
}
