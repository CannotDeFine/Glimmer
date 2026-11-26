#include "internal/cuda_admission_coordinator.h"
#include "internal/cuda_native_execution_epoch.h"
#include "internal/cuda_paused_native_executor.h"
#include "internal/native_execution_heartbeat.h"
#include "internal/owner_control_pump.h"
#include "internal/shared_control_channel.h"
#include "glimmer/control/task_execution_protocol.h"
#include "glimmer/control/native_epoch_protocol.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <future>
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
template <typename T>
decltype(auto) checked(T&& value) noexcept {
    if (!value.has_value()) std::abort();
    return *value;
}

using namespace glimmer::control;
using namespace glimmer::control::internal;
using namespace glimmer::interceptor;
using namespace std::chrono_literals;
using Gate = AsyncCooperativeLaunchGate;
using Clock = std::chrono::steady_clock;
constexpr NativeEpochExtent kEpochExtent{8, 1, 2, 1'000'000};
const auto kContext = reinterpret_cast<CUcontext>(0x1000);
const auto kOtherContext = reinterpret_cast<CUcontext>(0x4000);
const auto kStream = reinterpret_cast<CUstream>(0x2000);
thread_local CUcontext current_context{};
thread_local std::array<CUcontext, 4> context_stack{};
thread_local std::size_t context_depth{};
thread_local bool is_native{};

void require(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
std::size_t inventory(const char* path) {
    DIR* directory = ::opendir(path);
    require(directory != nullptr, "owned CPU resource inventory opens");
    std::size_t count{};
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    require(errno == 0 && ::closedir(directory) == 0, "CPU resource inventory closes");
    return count;
}
template <typename Predicate>
void await(Predicate predicate, const char* message) {
    const auto deadline = Clock::now() + 1s;
    while (!predicate()) {
        require(Clock::now() < deadline, message);
        std::this_thread::yield();
    }
}
template <typename Result>
bool ready(std::future<Result>& future) {
    return future.valid() && future.wait_for(0s) == std::future_status::ready;
}
class Descriptor final {
   public:
    explicit Descriptor(int fd) : fd_(fd) {
        require(fd >= 0, "owned descriptor creates");
    }
    ~Descriptor() {
        require(::close(fd_) == 0, "owned descriptor closes");
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const {
        return fd_;
    }

   private:
    int fd_;
};
class Peer final {
   public:
    explicit Peer(std::chrono::milliseconds transport_timeout = 1000ms)
        : listener_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {
        std::array<char, 64> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-owner-actors-XXXXXX";
        std::memcpy(pattern.data(), prefix.data(), prefix.size());
        require(::mkdtemp(pattern.data()) != nullptr, "owned directory creates");
        directory_ = pattern.data();
        path_ = directory_ + "/control.sock";
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        require(path_.size() < sizeof(address.sun_path), "owned socket path fits");
        std::memcpy(address.sun_path, path_.c_str(), path_.size() + 1);
        require(::bind(listener_.get(), reinterpret_cast<const sockaddr*>(&address),
                       sizeof(address)) == 0 &&
                    ::listen(listener_.get(), 4) == 0,
                "owned socket listens");
        std::thread startup([&] {
            gate = Gate::create_owner_pumped({.control_socket = path_},
                                             {.max_queued_requests = 16,
                                              .max_cpu_owners = 8,
                                              .transport_timeout = transport_timeout},
                                             kEpochExtent);
        });
        peer_ = std::make_unique<Descriptor>(
            ::accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC));
        std::string open;
        for (std::size_t index = 0; index < kOwnerControlOpen.size(); ++index) {
            char byte{};
            require(::recv(peer_->get(), &byte, 1, 0) == 1, "actual cold OPEN arrives");
            open.push_back(byte);
        }
        require(open == kOwnerControlOpen &&
                    ::send(peer_->get(), kOwnerControlReady.data(), kOwnerControlReady.size(),
                           MSG_NOSIGNAL) == static_cast<ssize_t>(kOwnerControlReady.size()),
                "explicit actual V3 negotiation completes");
        require(channel_.create_server({17, 29}, SharedControlMode::kReplyHintV3) &&
                    channel_.send_offer(peer_->get(), 1000ms),
                "authenticated sealed offer sends");
        startup.join();
        require(gate != nullptr, "cold public gate creates before actual actor");
    }
    ~Peer() {
        gate.reset();
        cleanup_files();
    }
    void cleanup_files() {
        if (has_removed_files_) return;
        require(::unlink(path_.c_str()) == 0 && ::rmdir(directory_.c_str()) == 0,
                "only owned socket and empty directory retire");
        has_removed_files_ = true;
    }
    Peer(const Peer&) = delete;
    Peer& operator=(const Peer&) = delete;
    TaskExecutionProtocolRequest receive(TaskExecutionProtocolOperation operation,
                                         TaskExecutionIdentity expected = {17, 0}) {
        std::string line;
        require(
            channel_.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine,
            "one actual actor request arrives");
        const auto parsed = parse_task_execution_protocol_request(line);
        require(parsed.parsed() && checked(parsed.request).operation == operation &&
                    checked(parsed.request).generation == expected.generation,
                "actual actor operation and fresh generation match");
        if (operation != TaskExecutionProtocolOperation::kExecutionAcquire &&
            operation != TaskExecutionProtocolOperation::kSubmit)
            require(checked(parsed.request).task_id == expected.task_id,
                    "actual task identity matches");
        return checked(parsed.request);
    }
    TaskExecutionProtocolRequest receive(TaskExecutionProtocolOperation operation,
                                         std::uint64_t generation) {
        return receive(operation, TaskExecutionIdentity{17, generation});
    }
    void reply(TaskExecutionProtocolResponseKind kind,
               TaskProtocolState state = TaskProtocolState::kQueued, bool should_retreat = false,
               TaskExecutionIdentity execution = {17, 29}) {
        TaskExecutionProtocolResponse response;
        response.kind = kind;
        response.task_id = execution.task_id;
        if (kind == TaskExecutionProtocolResponseKind::kExecutionLease ||
            kind == TaskExecutionProtocolResponseKind::kRetreat)
            response.generation = execution.generation;
        response.state = state;
        response.should_retreat = should_retreat;
        response.tenant_id = "default";
        response.memory_bytes = 1;
        response.weight = 1;
        response.work_units = 1;
        const auto wire = format_task_execution_protocol_response(response);
        require(wire && channel_.publish_response_with_hint(peer_->get(), *wire),
                "actual canonical reply body publishes before hint");
    }
    void no_request() {
        std::string line;
        require(channel_.receive_request(peer_->get(), 1, line) == SharedControlReadStatus::kIdle,
                "withheld ACK permits no independent wire");
    }
    void reject_terminal() {
        TaskExecutionProtocolResponse response;
        response.kind = TaskExecutionProtocolResponseKind::kError;
        response.error = TaskProtocolErrorCode::kUnknownTask;
        const auto wire = format_task_execution_protocol_response(response);
        require(wire && channel_.publish_response_with_hint(peer_->get(), *wire),
                "actual definite failed terminal response publishes");
    }
    NativeEpochRequest receive_epoch() {
        std::string line;
        require(
            channel_.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine,
            "one actual epoch control request arrives");
        const auto request = parse_native_epoch_request(line);
        require(request.has_value() && request->request_sequence != 0,
                "actual epoch request has fresh canonical correlation");
        return checked(request);
    }
    void reply_epoch(const NativeEpochRequest& request, NativeEpochResponseKind kind,
                     const NativeEpochGrant& grant) {
        NativeEpochResponse response;
        response.kind = kind;
        response.operation = request.operation;
        response.request_sequence = request.request_sequence;
        response.grant = grant;
        response.issued_calls = request.issued_calls;
        const auto wire = format_native_epoch_response(response);
        require(wire && channel_.publish_response_with_hint(peer_->get(), *wire),
                "actual epoch reply echoes exact sequence and identity before hint");
    }
    std::shared_ptr<Gate> gate;

   private:
    Descriptor listener_;
    std::unique_ptr<Descriptor> peer_;
    SharedControlChannel channel_;
    std::string directory_;
    std::string path_;
    bool has_removed_files_{};
};

struct Fake final {
    struct Event {
        bool was_recorded{};
    };
    static inline Fake* active{};
    std::atomic<bool> is_complete{};
    std::atomic<unsigned int> calls{}, launches{}, creates{}, destroys{}, ptds_records{};
    std::atomic<unsigned int> queries{};
    std::atomic<CUresult> query_error{CUDA_SUCCESS};
    std::atomic<std::int64_t> clock_nanoseconds{};
    CUresult drain_error{CUDA_SUCCESS};
    std::thread::id ptds_caller{};
    Fake() {
        require(active == nullptr, "one fake dispatch lifetime");
        active = this;
    }
    ~Fake() {
        require(creates == destroys, "all ordinary physical events retire");
        active = nullptr;
    }
    static Clock::time_point now() noexcept {
        return Clock::time_point{std::chrono::nanoseconds{active->clock_nanoseconds.load()}};
    }
    static bool enter() noexcept {
        const bool old = is_native;
        is_native = true;
        return old;
    }
    static void leave(bool old) noexcept {
        is_native = old;
    }
    static CUresult current(CUcontext* context) {
        *context = current_context;
        return CUDA_SUCCESS;
    }
    static CUresult push(CUcontext context) {
        require(is_native && context_depth < context_stack.size(),
                "owner binds bounded actual native context stack");
        context_stack[context_depth++] = current_context;
        current_context = context;
        return CUDA_SUCCESS;
    }
    static CUresult pop(CUcontext* context) {
        require(context_depth > 0, "owner pops only its actual bound native context");
        *context = current_context;
        current_context = context_stack[--context_depth];
        return CUDA_SUCCESS;
    }
    static void native_call() {
        require(is_native && (current_context == kContext || current_context == kOtherContext),
                "private CUDA keeps its actual native context");
        ++active->calls;
    }
    static CUresult create(CUevent* output, unsigned int flags) {
        native_call();
        require(flags == CU_EVENT_DISABLE_TIMING, "private event collects no timing");
        *output = reinterpret_cast<CUevent>(new Event{});
        ++active->creates;
        return CUDA_SUCCESS;
    }
    static CUresult record(CUevent event, CUstream stream) {
        native_call();
        require(stream == kStream || (stream == CU_STREAM_PER_THREAD &&
                                      std::this_thread::get_id() == active->ptds_caller),
                "actual native stream and PTDS recording caller remain unchanged");
        reinterpret_cast<Event*>(event)->was_recorded = true;
        return CUDA_SUCCESS;
    }
    static CUresult record_ptds(CUevent event, CUstream stream) {
        ++active->ptds_records;
        return record(event, stream);
    }
    static CUresult query(CUevent event) {
        native_call();
        ++active->queries;
        require(reinterpret_cast<Event*>(event)->was_recorded,
                "only actual recorded fence queries");
        if (const auto result = active->query_error.load(); result != CUDA_SUCCESS) return result;
        return active->is_complete.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    }
    static CUresult destroy(CUevent event) {
        native_call();
        delete reinterpret_cast<Event*>(event);
        ++active->destroys;
        return CUDA_SUCCESS;
    }
    static CUresult drain(CUstream stream) {
        native_call();
        require(stream == kStream, "physical stop drains actual stream");
        if (active->drain_error != CUDA_SUCCESS) return active->drain_error;
        active->is_complete = true;
        return CUDA_SUCCESS;
    }
    static CUresult launch(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int, CUstream stream, void**,
                           void**) {
        native_call();
        require(stream == kStream, "physical kernel uses actual stream");
        ++active->launches;
        return CUDA_SUCCESS;
    }
    static CudaQueueWorkerDispatch worker_dispatch() {
        CudaQueueDispatch dispatch{};
        dispatch.context_get_current = current;
        dispatch.launch_kernel = launch;
        dispatch.event_create = create;
        dispatch.event_record = record;
        dispatch.event_query = query;
        dispatch.event_destroy = destroy;
        dispatch.stream_synchronize = drain;
        dispatch.stream_wait_event = wait;
        dispatch.event_synchronize = synchronize_event;
        return {dispatch, push, pop, enter, leave};
    }
    static CUresult wait(CUstream stream, CUevent event, unsigned int flags) {
        native_call();
        require(stream == kStream && flags == 0 && reinterpret_cast<Event*>(event)->was_recorded,
                "native import waits actual recorded prefix on actual stream");
        return CUDA_SUCCESS;
    }
    static CUresult synchronize_event(CUevent event) {
        native_call();
        require(reinterpret_cast<Event*>(event)->was_recorded,
                "only recorded native prefix can prove stop");
        active->is_complete = true;
        return CUDA_SUCCESS;
    }
    static CUresult stream_context(CUstream, CUcontext* context) {
        *context = kContext;
        return CUDA_SUCCESS;
    }
    static CUresult capture(CUstream, CUstreamCaptureStatus* status) {
        *status = CU_STREAM_CAPTURE_STATUS_NONE;
        return CUDA_SUCCESS;
    }
    static CUresult in_context(CUcontext context,
                               const CudaPausedNativeExecutor::NativeOperation& operation) {
        const auto old = current_context;
        current_context = context;
        const auto result = operation();
        current_context = old;
        return result;
    }
};
struct Continuation final : CudaQueueContinuation {
    CUresult prepare_launch(std::array<unsigned int, 3>&, OwnedKernelArguments&) noexcept override {
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        return CUDA_SUCCESS;
    }
    CUresult stage_completion(CUstream) noexcept override {
        return CUDA_SUCCESS;
    }
    CUresult inspect_stopped(bool& is_complete) noexcept override {
        is_complete = false;
        return CUDA_SUCCESS;
    }
};
struct Coordinator final : CudaQueueCoordinator {
    explicit Coordinator(std::shared_ptr<Gate> gate)
        : actual(std::move(gate), {.heartbeat_interval = 1ms, .clock_now = Fake::now}) {}
    CudaAdmissionCoordinator actual;
    std::atomic<std::uint64_t> passes{};
    std::atomic<bool> has_stopped{};
    CUresult progress(CudaOperationQueue& queue) noexcept override {
        const auto result = actual.progress(queue);
        ++passes;
        return result;
    }
    void stopped(CudaOperationQueue& queue, CUresult result) noexcept override {
        actual.stopped(queue, result);
        has_stopped = true;
    }
    CUresult stopped_progress() noexcept override {
        return actual.stopped_progress();
    }
    CUresult native_boundary_status() const noexcept override {
        return actual.native_boundary_status();
    }
};
void add_kernel(CudaQueueWorker& worker, Coordinator& coordinator, bool can_retreat) {
    require(worker.invoke([&](CudaOperationQueue& queue) {
        auto resources = std::make_shared<int>(0);
        const auto stream = queue.bind_stream(kStream, OrderedStreamKind::kNonBlocking, resources);
        unsigned int value = 7;
        std::array<void*, 1> arguments{&value};
        const std::array<PtxParameterLayout, 1> layout{{{4, 4}}};
        auto owned = OwnedKernelArguments::copy(layout, arguments.data());
        if (!owned) return CUDA_ERROR_OUT_OF_MEMORY;
        const auto token = queue.submit(
            stream, CudaQueuedKernel{reinterpret_cast<CUfunction>(0x3000),
                                     {1, 1, 1},
                                     {32, 1, 1},
                                     0,
                                     std::move(*owned),
                                     resources,
                                     can_retreat ? std::make_shared<Continuation>() : nullptr});
        return token && coordinator.actual.track(queue, token) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    }) == CUDA_SUCCESS,
            "actual worker owns and tracks kernel metadata");
}
enum class StopPhase : std::uint8_t {
    kSubmitting,
    kPending,
    kClaiming,
    kRetreating,
    kHeartbeating,
    kSuspending,
    kCompleting,
    kFailing
};
void delayed_stop(StopPhase phase) {
    Fake fake;
    Peer peer;
    auto coordinator = std::make_shared<Coordinator>(peer.gate);
    CudaQueueWorker worker(kContext, Fake::worker_dispatch(), coordinator,
                           {.control_gate = peer.gate});
    require(worker.status() == CUDA_SUCCESS, "actual selected worker starts");
    const bool can_retreat = phase == StopPhase::kRetreating || phase == StopPhase::kSuspending;
    if (phase != StopPhase::kFailing) add_kernel(worker, *coordinator, can_retreat);
    auto native_acquisition = phase == StopPhase::kFailing ? peer.gate->submit_and_claim()
                                                           : std::future<CooperativeSubmission>{};
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionAcquire));
    if (phase == StopPhase::kFailing) {
        peer.reply(TaskExecutionProtocolResponseKind::kExecutionLease);
        await([&] { return ready(native_acquisition); },
              "actual native prefix receives fresh grant");
        const auto granted = native_acquisition.get();
        require(granted.claim.status == CooperativeClaimStatus::kGranted &&
                    worker.invoke([&](CudaOperationQueue& queue) {
                        auto prefix = std::make_shared<Fake::Event>();
                        const auto event = reinterpret_cast<CUevent>(prefix.get());
                        const auto recorded = Fake::record(event, kStream);
                        if (recorded != CUDA_SUCCESS) return recorded;
                        const auto stream =
                            queue.bind_stream(kStream, OrderedStreamKind::kNonBlocking, prefix);
                        const auto token =
                            queue.submit(stream, CudaQueuedNativeFence{event, prefix});
                        return token && coordinator->actual.adopt_native_execution(
                                            queue, token, granted.claim.execution,
                                            CUDA_ERROR_LAUNCH_FAILED)
                                   ? CUDA_SUCCESS
                                   : CUDA_ERROR_UNKNOWN;
                    }) == CUDA_SUCCESS,
                "actual recorded failed native prefix transfers exact grant to coordinator");
        fake.is_complete = true;
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionFail, 29));
    } else if (phase == StopPhase::kPending || phase == StopPhase::kClaiming) {
        peer.reply(TaskExecutionProtocolResponseKind::kAccepted);
        await([&] { return peer.gate->owner_pump_snapshot().outstanding_requests == 0; },
              "accepted pending original settles");
        const auto passes = coordinator->passes.load();
        await([&] { return coordinator->passes > passes + 2; }, "worker consumes pending result");
        if (phase == StopPhase::kClaiming) {
            fake.clock_nanoseconds = 2'000'000;
            static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionClaim));
        }
    } else if (phase != StopPhase::kSubmitting) {
        peer.reply(TaskExecutionProtocolResponseKind::kExecutionLease);
        await([&] { return fake.launches.load() == 1; }, "only fresh grant issues physical work");
        if (phase == StopPhase::kRetreating || phase == StopPhase::kSuspending) {
            static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kRetreat, 29));
            if (phase == StopPhase::kSuspending) {
                peer.reply(TaskExecutionProtocolResponseKind::kRetreat, TaskProtocolState::kRunning,
                           true);
                fake.is_complete = true;
                static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kSuspend, 29));
            }
        } else if (phase == StopPhase::kHeartbeating) {
            fake.clock_nanoseconds = 2'000'000;
            static_cast<void>(
                peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
        } else {
            fake.is_complete = true;
            static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionComplete, 29));
        }
    }
    auto shutdown = std::async(std::launch::async, [&] { return worker.shutdown(); });
    await([&] { return coordinator->has_stopped.load(); },
          "actual physical shutdown precedes CPU drain");
    const auto stopped_calls = fake.calls.load();
    if (phase == StopPhase::kPending)
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kCancel));
    require(!ready(shutdown), "withheld actual reply cannot finish selected owner shutdown");
    // receive_request rejects while the server owns an unanswered head. That
    // is not an idle/successor oracle; keep the actual reply withheld instead.
    std::this_thread::sleep_for(2ms);
    require(!ready(shutdown) && peer.gate->owner_pump_snapshot().outstanding_requests == 1,
            "unanswered head retains its one actual original");
    require(fake.calls == stopped_calls, "stopped CPU drain makes no private CUDA probe");
    if (phase == StopPhase::kPending) {
        peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kCancelled);
    } else if (phase == StopPhase::kCompleting || phase == StopPhase::kFailing) {
        peer.reply(TaskExecutionProtocolResponseKind::kState, phase == StopPhase::kCompleting
                                                                  ? TaskProtocolState::kCompleted
                                                                  : TaskProtocolState::kFailed);
    } else {
        if (phase == StopPhase::kSubmitting || phase == StopPhase::kClaiming)
            peer.reply(TaskExecutionProtocolResponseKind::kExecutionLease);
        else if (phase == StopPhase::kRetreating)
            peer.reply(TaskExecutionProtocolResponseKind::kRetreat, TaskProtocolState::kRunning);
        else
            peer.reply(TaskExecutionProtocolResponseKind::kState,
                       phase == StopPhase::kSuspending ? TaskProtocolState::kQueued
                                                       : TaskProtocolState::kRunning);
        const bool is_cancel = phase == StopPhase::kSuspending;
        static_cast<void>(peer.receive(is_cancel ? TaskExecutionProtocolOperation::kExecutionCancel
                                                 : TaskExecutionProtocolOperation::kExecutionFail,
                                       29));
        require(!ready(shutdown), "physical stop alone cannot substitute terminal ACK");
        std::this_thread::sleep_for(2ms);
        require(!ready(shutdown), "withheld terminal remains unacknowledged");
        require(fake.calls == stopped_calls, "terminal CPU wait cannot relaunch stopped work");
        peer.reply(TaskExecutionProtocolResponseKind::kState,
                   is_cancel ? TaskProtocolState::kCancelled : TaskProtocolState::kFailed);
    }
    await([&] { return ready(shutdown); }, "actual ACK lets sole worker finish without self-wait");
    require(shutdown.get() == CUDA_SUCCESS && worker.status() == CUDA_ERROR_DEINITIALIZED &&
                !peer.gate->owner_pump_snapshot().is_quarantined &&
                peer.gate->owner_pump_snapshot().eligible_cpu_owners == 0,
            "actual stopped owner drains and detaches cleanly");
    peer.no_request();  // Only after actual response publication is idle meaningful.
    const auto stats = coordinator->actual.stats();
    require(stats.acknowledged_completions == (phase == StopPhase::kCompleting ? 1U : 0U),
            "completion stats require actual success ACK");
    require(stats.acknowledged_failures ==
                (phase != StopPhase::kPending && phase != StopPhase::kSuspending &&
                         phase != StopPhase::kCompleting
                     ? 1U
                     : 0U),
            "failed stopped work keeps actual failure semantics");
}

void paused_worker_and_checkpoint() {
    Fake fake;
    Peer peer;
    auto coordinator = std::make_shared<Coordinator>(peer.gate);
    CudaQueueWorker worker(kContext, Fake::worker_dispatch(), coordinator,
                           {.control_gate = peer.gate});
    CudaNativeBoundaryToken pause;
    require(worker.try_pause_native(pause) == CUDA_SUCCESS,
            "empty actual native pause proves boundary");
    const auto passes = coordinator->passes.load();
    const auto calls = fake.calls.load();
    auto heartbeat = peer.gate->heartbeat({17, 29});
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
    require(!ready(heartbeat), "paused CPU actor cannot guess heartbeat ACK");
    std::this_thread::sleep_for(2ms);
    require(!ready(heartbeat), "withheld paused heartbeat stays unready");
    require(fake.calls == calls && coordinator->passes == passes,
            "paused worker advances control without coordinator or CUDA probe");
    peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kRunning);
    await([&] { return ready(heartbeat); }, "paused sole actor observes actual heartbeat reply");
    require(heartbeat.get(), "paused original heartbeat result remains actual");
    {
        NativeExecutionHeartbeat owner(
            [&](TaskExecutionIdentity execution) {
                auto original = peer.gate->heartbeat(execution);
                while (original.valid() && original.wait_for(1ms) != std::future_status::ready)
                    if (peer.gate->owner_pump_snapshot().is_quarantined) return false;
                return original.valid() && original.get();
            },
            1ms);
        NativeExecutionHeartbeat::Session session(owner, {17, 29});
        require(session.valid(), "actual separate native heartbeat session starts");
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
        auto stopped = std::async(std::launch::async, [&] { return session.stop(); });
        std::this_thread::sleep_for(2ms);
        require(!ready(stopped) && fake.calls == calls && coordinator->passes == passes,
                "session stop retains real heartbeat while sole worker stays privately paused");
        peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kRunning);
        await([&] { return ready(stopped); },
              "actual paused CPU pump releases heartbeat session stop");
        require(stopped.get(), "session stop observes actual original heartbeat ACK");
    }
    require(worker.finish_native(pause, [](CudaOperationQueue&) { return CUDA_SUCCESS; }) ==
                CUDA_SUCCESS,
            "trusted empty paused transaction finishes");
    CudaWorkerCheckpoint checkpoint;
    require(worker.checkpoint(checkpoint) == CUDA_SUCCESS &&
                peer.gate->owner_pump_snapshot().eligible_cpu_owners == 0 &&
                !peer.gate->heartbeat({17, 29}).valid(),
            "joined checkpoint withdraws last actor without stranding acceptance");
    current_context = kContext;
    require(worker.resume_checkpoint(checkpoint) == CUDA_SUCCESS,
            "independently bound checkpoint attaches fresh CPU actor");
    current_context = nullptr;
    auto resumed = peer.gate->heartbeat({17, 29});
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
    peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kRunning);
    await([&] { return ready(resumed); }, "resumed fresh actor advances actual CPU reply");
    require(resumed.get() && worker.shutdown() == CUDA_SUCCESS,
            "resumed actor retires normally without closing process gate");
}

void native_only_actual_caller() {
    Fake fake;
    Peer peer;
    {
        CudaPausedNativeExecutor executor(peer.gate,
                                          {Fake::create, Fake::record, Fake::record_ptds,
                                           Fake::query, Fake::destroy, Fake::enter, Fake::leave},
                                          Fake::in_context);
        const pid_t child = ::fork();
        require(child >= 0, "owned inherited native actor child creates");
        if (child == 0) {
            CudaPausedNativePermit inherited;
            const auto result =
                executor.acquire(kContext, kStream, false, std::make_shared<int>(0), {}, inherited);
            ::_exit(result == CUDA_ERROR_UNKNOWN && !inherited ? 0 : 1);
        }
        int child_status{};
        require(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                    WEXITSTATUS(child_status) == 0,
                "inherited actual native admission rejects before mutex/CUDA and child reaps");
        // The empty native monitor is the only registered CPU executor. No
        // cooperative frontend or dummy queue exists in this case.
        auto idle_heartbeat = peer.gate->heartbeat({17, 29});
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
        peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kRunning);
        await([&] { return ready(idle_heartbeat); },
              "empty native-only monitor advances shared CPU FIFO");
        require(idle_heartbeat.get() && fake.calls == 0,
                "empty native monitor makes no CUDA query");
        std::atomic<unsigned int> originals{};
        auto caller = std::async(std::launch::async, [&] {
            const auto caller_id = std::this_thread::get_id();
            CudaPausedNativePermit permit;
            const auto acquired =
                executor.acquire(kContext, kStream, true, std::make_shared<int>(0), {}, permit);
            if (acquired != CUDA_SUCCESS) return acquired;
            const auto result = executor.issue(permit, [&] {
                require(std::this_thread::get_id() == caller_id && current_context == kContext &&
                            is_native,
                        "original stays on actual caller context, never CPU pump");
                ++originals;
                return CUDA_SUCCESS;
            });
            return result.was_invoked && result.tracking_status == CUDA_SUCCESS
                       ? result.status
                       : CUDA_ERROR_UNKNOWN;
        });
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kSubmit));
        require(originals == 0, "submission hint conveys no execution authority");
        peer.reply(TaskExecutionProtocolResponseKind::kAccepted);
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionClaim));
        require(originals == 0, "fresh native claim must receive actual authorization");
        peer.reply(TaskExecutionProtocolResponseKind::kExecutionLease);
        await([&] { return ready(caller); },
              "native-only monitor grants actual caller without dummy worker");
        require(caller.get() == CUDA_SUCCESS && originals == 1 && fake.ptds_records == 1,
                "native operation and actual PTDS trailing record happen exactly once");
        fake.is_complete = true;
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionComplete, 29));
        auto drained = std::async(std::launch::async, [&] { return executor.drain(); });
        require(!ready(drained), "native physical fence cannot replace actual terminal ACK");
        peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kCompleted);
        await([&] { return ready(drained); }, "native-only owner consumes actual terminal reply");
        require(drained.get() == CUDA_SUCCESS && executor.stats().completed == 1 &&
                    executor.stats().retained == 0,
                "native-only physical owner drains actual completion");
    }
    require(peer.gate->owner_pump_snapshot().eligible_cpu_owners == 0 &&
                !peer.gate->owner_pump_snapshot().is_quarantined,
            "empty process monitor retires cleanly after joining");
}

void retiring_actor() {
    Peer peer;
    auto owner = peer.gate->register_cpu_owner();
    auto request = peer.gate->heartbeat({17, 29});
    require(owner.begin_retirement() &&
                peer.gate->owner_pump_snapshot().accepting_cpu_owners == 0 &&
                !peer.gate->heartbeat({17, 29}).valid() && !owner.detach(),
            "retirement atomically withdraws acceptance but retains accepted drain eligibility");
    auto executor = peer.gate->try_acquire_cpu(owner);
    require(executor.has_value(), "retiring actor can still acquire CPU lease");
    static_cast<void>(checked(executor).progress());
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
    peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kRunning);
    await(
        [&] {
            static_cast<void>(checked(executor).progress());
            return ready(request);
        },
        "retiring last actor settles original actual reply");
    require(request.get(), "retirement keeps original result semantics");
    executor.reset();
    require(owner.detach(), "settled retired actor detaches");
    auto fresh = peer.gate->register_cpu_owner();
    require(static_cast<bool>(fresh) &&
                peer.gate->owner_pump_snapshot().accepting_cpu_owners == 1 &&
                fresh.begin_retirement() && fresh.detach(),
            "a later fresh actor can reactivate process session without reconnect");
}

void retirement_producer_race() {
    Peer peer;
    for (unsigned int iteration = 0; iteration < 16; ++iteration) {
        auto owner = peer.gate->register_cpu_owner();
        std::atomic<bool> can_start{};
        std::future<bool> original;
        std::thread producer([&] {
            while (!can_start.load(std::memory_order_acquire)) std::this_thread::yield();
            original = peer.gate->heartbeat({17, 29});
        });
        can_start.store(true, std::memory_order_release);
        require(owner.begin_retirement(), "last-actor withdrawal races only FIFO acceptance");
        producer.join();
        const auto snapshot = peer.gate->owner_pump_snapshot();
        require(snapshot.accepting_cpu_owners == 0 && snapshot.eligible_cpu_owners == 1 &&
                    snapshot.outstanding_requests == (original.valid() ? 1U : 0U) &&
                    !peer.gate->heartbeat({17, 29}).valid(),
                "racing producer is either rejected or retained under executable ownership");
        if (original.valid()) {
            require(!owner.detach(), "accepted racing request prevents last-owner detachment");
            auto executor = peer.gate->try_acquire_cpu(owner);
            require(executor.has_value(), "retiring actor retains its real CPU lease eligibility");
            static_cast<void>(checked(executor).progress());
            static_cast<void>(
                peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
            peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kRunning);
            await(
                [&] {
                    static_cast<void>(checked(executor).progress());
                    return ready(original);
                },
                "accepted racing original settles only through actual reply");
            require(original.get(), "racing acceptance retains original actual result");
        }
        require(owner.detach() && !peer.gate->owner_pump_snapshot().is_quarantined,
                "settled or rejected race permits clean final detach");
        peer.no_request();
    }
}

void actual_transport_deadline() {
    Fake fake;
    Peer peer(30ms);
    auto coordinator = std::make_shared<Coordinator>(peer.gate);
    CudaQueueWorker worker(kContext, Fake::worker_dispatch(), coordinator,
                           {.control_gate = peer.gate});
    auto heartbeat = peer.gate->heartbeat({17, 29});
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
    // The real remote-operation deadline, not socket readiness or a fabricated
    // reply, determines uncertainty. Never get an original while unready.
    await([&] { return peer.gate->owner_pump_snapshot().is_quarantined; },
          "actual sole worker contains unanswered transport at its finite deadline");
    await([&] { return ready(heartbeat); }, "transport uncertainty resolves the accepted original");
    require(!heartbeat.get(), "transport uncertainty never acknowledges the original");
    require(!peer.gate->heartbeat({17, 29}).valid(), "quarantine rejects new publications");
    require(worker.shutdown() == CUDA_ERROR_UNKNOWN, "uncertain worker shutdown cannot succeed");
    require(fake.calls == 0, "transport uncertainty cannot trigger a CUDA effect");
    require(peer.gate->owner_pump_snapshot().eligible_cpu_owners == 0,
            "uncertain worker retires its CPU ownership");
}

void actual_retirement_deadline_child() {
    const pid_t child = ::fork();
    require(child >= 0, "owned retirement-containment child creates");
    if (child == 0) {
        Fake fake;
        Peer peer(5s);  // Longer than the actual actor's fixed two-second retirement budget.
        auto coordinator = std::make_shared<Coordinator>(peer.gate);
        CudaQueueWorker worker(kContext, Fake::worker_dispatch(), coordinator,
                               {.control_gate = peer.gate});
        auto heartbeat = peer.gate->heartbeat({17, 29});
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionHeartbeat, 29));
        CudaWorkerCheckpoint checkpoint;
        const auto started = Clock::now();
        auto stopped =
            std::async(std::launch::async, [&] { return worker.checkpoint(checkpoint); });
        await([&] { return peer.gate->owner_pump_snapshot().accepting_cpu_owners == 0; },
              "actual checkpoint atomically withdraws last actor acceptance");
        require(!ready(heartbeat) && !peer.gate->heartbeat({17, 29}).valid() && fake.calls == 0,
                "withdrawn last actor retains unanswered original without private CUDA probes");
        require(peer.gate->owner_pump_snapshot().outstanding_requests == 1,
                "healthy retiring actor still owns the accepted unanswered original");
        require(stopped.wait_for(4s) == std::future_status::ready,
                "actual retirement is bounded independently of the longer transport deadline");
        const auto result = stopped.get();
        const auto elapsed = Clock::now() - started;
        const auto snapshot = peer.gate->owner_pump_snapshot();
        require(result == CUDA_ERROR_UNKNOWN && worker.status() == result && !checkpoint &&
                    elapsed >= 1500ms && elapsed < 4s && snapshot.is_quarantined &&
                    snapshot.accepting_cpu_owners == 0 && !ready(heartbeat) && fake.calls == 0,
                "retirement timeout retains the unready original and authority, never a fake ACK");
        // A poisoned snapshot deliberately masks counters. Its zero count is
        // not evidence of release; the retained original remains unready.
        peer.cleanup_files();
        // The failed checkpoint/accepted original are intentionally retained
        // containment. This is not a leak-free physical cleanup assertion.
        ::_exit(0);
    }
    int status{};
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "owned retirement-containment child passes and reaps");
}

void two_context_terminal_fifo() {
    Fake fake;
    Peer peer;
    auto first_coordinator = std::make_shared<Coordinator>(peer.gate);
    CudaQueueWorker first(kContext, Fake::worker_dispatch(), first_coordinator,
                          {.control_gate = peer.gate});
    add_kernel(first, *first_coordinator, false);
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionAcquire));
    peer.reply(TaskExecutionProtocolResponseKind::kExecutionLease);
    await([&] { return fake.launches.load() == 1; },
          "first context issues only its authorized generation");
    auto stopped = std::async(std::launch::async, [&] { return first.shutdown(); });
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionFail, 29));
    auto second_coordinator = std::make_shared<Coordinator>(peer.gate);
    CudaQueueWorker second(kOtherContext, Fake::worker_dispatch(), second_coordinator,
                           {.control_gate = peer.gate});
    add_kernel(second, *second_coordinator, false);
    await([&] { return peer.gate->owner_pump_snapshot().outstanding_requests == 2; },
          "distinct contexts share the exact accepted predecessor and successor FIFO");
    std::this_thread::sleep_for(2ms);
    require(
        fake.launches == 1 && !ready(stopped),
        "second context cannot issue or finish predecessor while actual terminal ACK is withheld");
    peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kFailed);
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionAcquire));
    // Acceptance is not authorization: the second context still needs its
    // independently validated task/generation on the same process endpoint.
    require(fake.launches == 1, "successor wire alone cannot authorize another context");
    peer.reply(TaskExecutionProtocolResponseKind::kExecutionLease, TaskProtocolState::kRunning,
               false, {18, 30});
    static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionComplete,
                                   TaskExecutionIdentity{18, 30}));
    require(fake.launches == 2, "fresh distinct generation issues exactly once on second context");
    peer.reply(TaskExecutionProtocolResponseKind::kState, TaskProtocolState::kCompleted, false,
               {18, 30});
    await([&] { return ready(stopped); },
          "one owner detaches while other context retains shared CPU progress");
    require(stopped.get() == CUDA_SUCCESS && second.shutdown() == CUDA_SUCCESS &&
                peer.gate->owner_pump_snapshot().eligible_cpu_owners == 0 &&
                !peer.gate->owner_pump_snapshot().is_quarantined,
            "two actual contexts drain one FIFO without endpoint replacement or replay");
}

void actual_epoch(bool is_ptds) {
    Fake fake;
    Peer peer;
    fake.is_complete = true;
    fake.ptds_caller = std::this_thread::get_id();
    current_context = kContext;
    const auto stream = is_ptds ? CU_STREAM_PER_THREAD : kStream;
    {
        CudaNativeExecutionEpoch epoch(
            kContext, stream, {1, 2, 3}, {.extent = kEpochExtent}, peer.gate,
            {Fake::current, Fake::push, Fake::pop, Fake::stream_context, Fake::capture,
             Fake::create, Fake::record, Fake::query, Fake::destroy, Fake::enter, Fake::leave});
        const auto acquisition = peer.receive_epoch();
        require(acquisition.operation == NativeEpochOperation::kAcquire &&
                    acquisition.extent == kEpochExtent,
                "actual epoch acquires exact authorized extent after predecessor proof");
        const auto deadline = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  (Clock::now() + 900ms).time_since_epoch())
                                  .count();
        require(deadline > 0, "native epoch deadline has checked monotonic representation");
        const NativeEpochGrant grant{{17, 29}, kEpochExtent, static_cast<std::uint64_t>(deadline)};
        peer.reply_epoch(acquisition, NativeEpochResponseKind::kGranted, grant);
        await([&] { return epoch.snapshot().phase == CudaNativeEpochPhase::kReady; },
              "sole actual epoch monitor advances its fresh CPU grant");
        const auto issue = epoch.begin_call();
        require(issue.has_value() && current_context == kContext &&
                    std::this_thread::get_id() == fake.ptds_caller,
                "epoch actual caller independently enters exact granted issue");
        // This original success is published by its real caller. Only original
        // dispatch recording/query can turn it into physical prefix proof.
        require(epoch.original_returned(checked(issue), NativeEpochCallOutcome::kSuccess),
                "actual caller return publishes once without guessed physical completion");
        epoch.close();
        bool has_terminal = false;
        for (unsigned int index = 0; index < 4 && !has_terminal; ++index) {
            const auto request = peer.receive_epoch();
            require(request.execution.task_id == grant.execution.task_id &&
                        request.execution.generation == grant.execution.generation,
                    "epoch retains exact task and generation");
            if (request.operation == NativeEpochOperation::kComplete) {
                has_terminal = true;
                const auto snapshot = epoch.snapshot();
                require(request.issued_calls == 1 && snapshot.proven_prefixes == 1 &&
                            snapshot.phase == CudaNativeEpochPhase::kReporting,
                        "actual physical prefix proof precedes one-count terminal report");
                const auto calls = fake.calls.load();
                std::this_thread::sleep_for(2ms);
                require(epoch.snapshot().phase == CudaNativeEpochPhase::kReporting &&
                            fake.calls == calls,
                        "withheld epoch ACK cannot release capacity or repeat private CUDA proof");
                peer.reply_epoch(request, NativeEpochResponseKind::kAcknowledged, grant);
            } else {
                require(request.operation == NativeEpochOperation::kRetreat ||
                            request.operation == NativeEpochOperation::kHeartbeat,
                        "only retained original epoch control may precede terminal");
                peer.reply_epoch(request,
                                 request.operation == NativeEpochOperation::kRetreat
                                     ? NativeEpochResponseKind::kRetreat
                                     : NativeEpochResponseKind::kAcknowledged,
                                 grant);
            }
        }
        require(has_terminal, "bounded actual epoch control reaches physical terminal");
        epoch.shutdown();
        const auto snapshot = epoch.snapshot();
        require(snapshot.phase == CudaNativeEpochPhase::kComplete &&
                    snapshot.status == CUDA_SUCCESS &&
                    snapshot.created_events == snapshot.destroyed_events &&
                    peer.gate->owner_pump_snapshot().eligible_cpu_owners == 0 &&
                    !peer.gate->owner_pump_snapshot().is_quarantined,
                "actual epoch consumes real ACK and retires CPU/physical ownership cleanly");
    }
    current_context = nullptr;
}

void failed_epoch_joined_recovery() {
    Fake fake;
    Peer peer;
    fake.is_complete = true;
    fake.ptds_caller = std::this_thread::get_id();
    current_context = kContext;
    {
        // This is the existing process native monitor, not a recovery-specific
        // executor or the original caller impersonating a CPU actor.
        CudaPausedNativeExecutor master(peer.gate,
                                        {Fake::create, Fake::record, Fake::record_ptds, Fake::query,
                                         Fake::destroy, Fake::enter, Fake::leave},
                                        Fake::in_context);
        CudaNativeExecutionEpoch epoch(
            kContext, kStream, {1, 2, 3}, {.extent = kEpochExtent}, peer.gate,
            {Fake::current, Fake::push, Fake::pop, Fake::stream_context, Fake::capture,
             Fake::create, Fake::record, Fake::query, Fake::destroy, Fake::enter, Fake::leave});
        const auto acquisition = peer.receive_epoch();
        require(acquisition.operation == NativeEpochOperation::kAcquire,
                "recovery owner has independent actual epoch acquisition");
        const auto deadline = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  (Clock::now() + 900ms).time_since_epoch())
                                  .count();
        require(deadline > 0, "recovery grant has checked monotonic deadline");
        const NativeEpochGrant grant{{17, 29}, kEpochExtent, static_cast<std::uint64_t>(deadline)};
        peer.reply_epoch(acquisition, NativeEpochResponseKind::kGranted, grant);
        await([&] { return epoch.snapshot().phase == CudaNativeEpochPhase::kReady; },
              "actual recovery grant reaches original caller");
        const auto issue = epoch.begin_call();
        require(issue.has_value(), "recovery original enters only under actual fresh grant");
        fake.query_error = CUDA_ERROR_LAUNCH_FAILED;
        require(epoch.original_returned(checked(issue), NativeEpochCallOutcome::kSuccess),
                "original caller publishes actual successful return before physical-query error");
        epoch.close();
        await([&] { return epoch.snapshot().phase == CudaNativeEpochPhase::kRetained; },
              "actual failed private query retains physical ownership");
        epoch.shutdown();
        require(peer.gate->owner_pump_snapshot().eligible_cpu_owners == 1 &&
                    !peer.gate->owner_pump_snapshot().is_quarantined,
                "joined failed epoch leaves the actual empty process monitor CPU-eligible");
        const auto queries = fake.queries.load();
        unsigned int proofs{};
        std::thread responder([&] {
            auto terminal = peer.receive_epoch();
            for (unsigned int count = 0;
                 terminal.operation != NativeEpochOperation::kFail && count < 4; ++count) {
                require(terminal.operation == NativeEpochOperation::kRetreat ||
                            terminal.operation == NativeEpochOperation::kHeartbeat,
                        "previously accepted control drains without inventing a terminal ACK");
                peer.reply_epoch(terminal,
                                 terminal.operation == NativeEpochOperation::kRetreat
                                     ? NativeEpochResponseKind::kRetreat
                                     : NativeEpochResponseKind::kAcknowledged,
                                 grant);
                terminal = peer.receive_epoch();
            }
            require(terminal.operation == NativeEpochOperation::kFail &&
                        terminal.issued_calls == 1 &&
                        terminal.execution.task_id == grant.execution.task_id &&
                        terminal.execution.generation == grant.execution.generation,
                    "whole-context recovery submits exactly one failed actual terminal");
            std::this_thread::sleep_for(2ms);
            require(epoch.snapshot().phase == CudaNativeEpochPhase::kRetained &&
                        fake.queries == queries,
                    "withheld real recovery ACK neither completes ownership nor retries query");
            peer.reply_epoch(terminal, NativeEpochResponseKind::kAcknowledged, grant);
        });
        const auto recovered = epoch.recover_context_completion({1, 2, 3}, [&] {
            require(current_context == kContext && std::this_thread::get_id() == fake.ptds_caller &&
                        is_native && fake.queries == queries,
                    "independent context proof stays on actual caller without failed-query replay");
            ++proofs;
            return CUDA_SUCCESS;
        });
        responder.join();
        const auto snapshot = epoch.snapshot();
        require(recovered == CUDA_SUCCESS && proofs == 1 &&
                    snapshot.phase == CudaNativeEpochPhase::kComplete &&
                    snapshot.status == CUDA_ERROR_LAUNCH_FAILED &&
                    snapshot.created_events == snapshot.destroyed_events &&
                    fake.queries == queries &&
                    epoch.recover_context_completion({1, 2, 3}) != CUDA_SUCCESS,
                "actual ACK retires failed epoch once while preserving its first error");
    }
    current_context = nullptr;
    require(peer.gate->owner_pump_snapshot().eligible_cpu_owners == 0 &&
                !peer.gate->owner_pump_snapshot().is_quarantined,
            "recovery process monitor finally retires without orphaned CPU work");
}

void failed_stop_child(bool has_unknown_physical_stop) {
    // Physical uncertainty intentionally retains resources. Isolate that
    // containment from the ordinary parent leak/resource-cleanup assertions.
    const pid_t child = ::fork();
    require(child >= 0, "owned physical containment child creates");
    if (child == 0) {
        std::cout << "Containment: unknown physical stop=" << has_unknown_physical_stop << '\n';
        Fake fake;
        Peer peer;
        auto coordinator = std::make_shared<Coordinator>(peer.gate);
        CudaQueueWorker worker(kContext, Fake::worker_dispatch(), coordinator,
                               {.control_gate = peer.gate});
        add_kernel(worker, *coordinator, false);
        static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionAcquire));
        peer.reply(TaskExecutionProtocolResponseKind::kExecutionLease);
        await([&] { return fake.launches.load() == 1; },
              "containment child issues once under fresh grant");
        if (has_unknown_physical_stop) fake.drain_error = CUDA_ERROR_LAUNCH_FAILED;
        auto shutdown = std::async(std::launch::async, [&] { return worker.shutdown(); });
        await([&] { return coordinator->has_stopped.load(); },
              "containment child attempts actual physical stop");
        if (!has_unknown_physical_stop) {
            static_cast<void>(peer.receive(TaskExecutionProtocolOperation::kExecutionFail, 29));
            peer.reject_terminal();
        }
        await([&] { return ready(shutdown); },
              "failed stop has bounded CPU containment without unready get");
        const auto result = shutdown.get();
        const auto snapshot = peer.gate->owner_pump_snapshot();
        require(
            result == (has_unknown_physical_stop ? CUDA_ERROR_LAUNCH_FAILED : CUDA_ERROR_UNKNOWN) &&
                worker.status() == result && snapshot.is_quarantined &&
                !peer.gate->submit_and_claim().valid() && fake.launches == 1 &&
                coordinator->actual.stats().acknowledged_completions == 0 &&
                coordinator->actual.stats().acknowledged_failures == 0,
            "unknown physical stop or negative actual ACK preserves sticky failure and CPU "
            "quarantine");
        // A healthy idle check is meaningful only for the unknown-stop case:
        // no terminal report may be emitted without real physical-stop proof.
        if (has_unknown_physical_stop) peer.no_request();
        peer.cleanup_files();  // CPU filesystem cleanup does not release uncertain GPU resources.
        ::_exit(0);
    }
    int status{};
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "owned physical containment child passes and reaps");
}
}  // namespace

int main() {
    const auto descriptors = inventory("/proc/self/fd");
    const auto threads = inventory("/proc/self/task");
    for (const auto phase : {StopPhase::kSubmitting, StopPhase::kPending, StopPhase::kClaiming,
                             StopPhase::kRetreating, StopPhase::kHeartbeating,
                             StopPhase::kSuspending, StopPhase::kCompleting, StopPhase::kFailing}) {
        std::cout << "Stopped phase " << static_cast<unsigned int>(phase) << '\n';
        delayed_stop(phase);
    }
    paused_worker_and_checkpoint();
    native_only_actual_caller();
    retiring_actor();
    retirement_producer_race();
    actual_transport_deadline();
    actual_retirement_deadline_child();
    two_context_terminal_fifo();
    actual_epoch(false);
    actual_epoch(true);
    failed_epoch_joined_recovery();
    failed_stop_child(false);
    failed_stop_child(true);
    // A checked pthread join may precede proc task removal. Bound only final
    // owned cleanup; never retry a gate, workload, or failed test.
    await([&] { return inventory("/proc/self/task") == threads; },
          "every owned actor and caller thread retires");
    require(inventory("/proc/self/fd") == descriptors, "ordinary parent leaks no CPU descriptor");
}
