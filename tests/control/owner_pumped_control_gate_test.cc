#include "glimmer/control/async_cooperative_launch_gate.h"
#include "glimmer/control/unix_socket_server.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/unix_socket_task_execution_server.h"
#include "internal/process_control_mailbox.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <sys/socket.h>
#include <sys/syscall.h>
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
using namespace std::chrono_literals;
using Gate = AsyncCooperativeLaunchGate;
using Clock = std::chrono::steady_clock;
constexpr NativeEpochExtent kExtent{8, 4, 2, 1'000'000};
constexpr TaskExecutionIdentity kExecution{17, 29};
void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}
class Directory final {
   public:
    Directory() {
        std::array<char, 48> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-owner-gate-XXXXXX";
        std::memcpy(pattern.data(), prefix.data(), prefix.size());
        require(::mkdtemp(pattern.data()) != nullptr, "owned directory creates");
        path = pattern.data();
    }
    ~Directory() {
        require(::rmdir(path.c_str()) == 0, "owned empty directory retires");
    }
    Directory(const Directory&) = delete;
    Directory& operator=(const Directory&) = delete;
    std::string path;
};
class Descriptor final {
   public:
    explicit Descriptor(int value) : value_(value) {
        require(value >= 0, "owned descriptor creates");
    }
    ~Descriptor() {
        require(::close(value_) == 0, "owned descriptor retires");
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const {
        return value_;
    }

   private:
    int value_;
};
std::size_t inventory(const char* path) {
    DIR* directory = ::opendir(path);
    require(directory != nullptr, "CPU resource inventory opens");
    std::size_t count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    require(errno == 0 && ::closedir(directory) == 0, "CPU resource inventory closes");
    return count;
}
void await_joined_thread(long thread_id) {
    require(thread_id > 0, "owned startup thread has checked Linux identity");
    const auto path = "/proc/self/task/" + std::to_string(thread_id);
    const auto deadline = Clock::now() + 1s;
    for (;;) {
        if (::access(path.c_str(), F_OK) != 0) {
            require(errno == ENOENT, "joined thread retires from proc inventory");
            return;
        }
        require(Clock::now() < deadline, "joined startup task must retire within bounded cleanup");
        std::this_thread::yield();
    }
}
template <typename Result>
bool ready(std::future<Result>& future) {
    return future.valid() && future.wait_for(0s) == std::future_status::ready;
}
class Peer final {
   public:
    explicit Peer(Gate::OwnerPumpOptions bounds = {.max_queued_requests = 4, .max_cpu_owners = 2})
        : listener_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        require(path.size() < sizeof(address.sun_path), "owned socket path fits");
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        require(::bind(listener_.get(), reinterpret_cast<const sockaddr*>(&address),
                       sizeof(address)) == 0 &&
                    ::listen(listener_.get(), 4) == 0,
                "owned socket listens");
        const auto threads = inventory("/proc/self/task");
        long startup_thread = 0;
        std::thread startup([&] {
            startup_thread = ::syscall(SYS_gettid);
            gate = Gate::create_owner_pumped({.control_socket = path}, bounds, kExtent);
        });
        peer_ = std::make_unique<Descriptor>(
            ::accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC));
        std::string open;
        for (std::size_t index = 0; index < kOwnerControlOpen.size(); ++index) {
            char byte = 0;
            require(::recv(peer_->get(), &byte, 1, 0) == 1, "cold OPEN arrives");
            open.push_back(byte);
        }
        require(open == kOwnerControlOpen &&
                    ::send(peer_->get(), kOwnerControlReady.data(), kOwnerControlReady.size(),
                           MSG_NOSIGNAL) == static_cast<ssize_t>(kOwnerControlReady.size()),
                "explicit V3 negotiates");
        require(channel_.create_server({17, 29}, SharedControlMode::kReplyHintV3) &&
                    channel_.send_offer(peer_->get(), 1000ms),
                "sealed actual offer sends");
        startup.join();
        // pthread_join can precede removal of the exiting task's proc entry.
        // Wait only for our exact joined helper, never a new executor thread.
        await_joined_thread(startup_thread);
        require(gate && inventory("/proc/self/task") == threads,
                "public factory creates no control thread");
    }
    ~Peer() {
        require(::unlink(path.c_str()) == 0, "owned socket inode retires");
    }
    Peer(const Peer&) = delete;
    Peer& operator=(const Peer&) = delete;
    TaskExecutionProtocolRequest receive() {
        std::string line;
        require(
            channel_.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine,
            "one actual request arrives");
        const auto parsed = parse_task_execution_protocol_request(line);
        require(parsed.parsed(), "actual canonical request parses");
        return checked(parsed.request);
    }
    void reply(std::string_view line) {
        require(channel_.publish_response_with_hint(peer_->get(), line),
                "actual body precedes hint");
    }
    void reply(const TaskExecutionProtocolResponse& response) {
        const auto wire = format_task_execution_protocol_response(response);
        require(wire.has_value(), "response schema encodes");
        reply(checked(wire));
    }
    void no_request() {
        std::string line;
        require(channel_.receive_request(peer_->get(), 1, line) == SharedControlReadStatus::kIdle,
                "no unpublished successor reaches service");
    }
    Directory directory;
    const std::string path{directory.path + "/control.sock"};
    std::unique_ptr<Gate> gate;

   private:
    Descriptor listener_;
    std::unique_ptr<Descriptor> peer_;
    SharedControlChannel channel_;
};
TaskExecutionProtocolResponse response(TaskExecutionProtocolResponseKind kind) {
    TaskExecutionProtocolResponse value{};
    value.kind = kind;
    if (kind != TaskExecutionProtocolResponseKind::kError &&
        kind != TaskExecutionProtocolResponseKind::kEmpty)
        value.task_id = kExecution.task_id;
    if (kind == TaskExecutionProtocolResponseKind::kExecutionLease ||
        kind == TaskExecutionProtocolResponseKind::kExecutionHandoff ||
        kind == TaskExecutionProtocolResponseKind::kRetreat)
        value.generation = kExecution.generation;
    value.tenant_id = "default";
    value.memory_bytes = 1;
    value.weight = 1;
    value.work_units = 1;
    if (kind == TaskExecutionProtocolResponseKind::kExecutionHandoff)
        value.completed_execution = kExecution;
    return value;
}
template <typename Result>
Result resolve(Gate::CpuExecutor& executor, std::future<Result> future) {
    require(future.valid(), "public acceptance returns original valid future");
    const auto deadline = Clock::now() + 2s;
    while (!ready(future) && Clock::now() < deadline) {
        const auto progress = executor.progress();
        require(progress != Gate::CpuProgress::kRejected, "valid CPU lease progresses");
        if (!ready(future)) static_cast<void>(executor.wait_until(deadline));
    }
    require(ready(future), "actual typed result becomes ready within finite test bound");
    return future.get();
}
void test_wait_migration_and_retirement() {
    Peer peer;
    auto& gate = *peer.gate;
    require(!gate.submit().valid(), "metadata producer is not an eligible CPU actor");
    auto first = gate.register_cpu_owner();
    auto second = gate.register_cpu_owner();
    require(first && second && !gate.register_cpu_owner(), "cold CPU owner bound rejects excess");
    auto signal = gate.subscribe_progress();
    auto other_signal = gate.subscribe_progress();
    require(signal && other_signal, "independent weak subscriptions accept");
    auto submitted = gate.submit_and_claim({.should_measure_timing = true});
    auto executor = gate.try_acquire_cpu(first);
    require(executor && !gate.try_acquire_cpu(second) && !first.detach() &&
                executor->progress() == Gate::CpuProgress::kPending && !ready(submitted),
            "one CPU executor starts without synchronous get");
    const auto wire = peer.receive();
    require(wire.operation == TaskExecutionProtocolOperation::kExecutionAcquire,
            "canonical public acquisition maps exactly");
    std::thread wrong_thread([&] {
        require(executor->progress() == Gate::CpuProgress::kRejected &&
                    executor->wait_until(Clock::now()).status == Gate::CpuWaitStatus::kRejected,
                "wrong thread cannot advance or consume wait");
    });
    wrong_thread.join();
    const pid_t child = ::fork();
    require(child >= 0, "owned fork creates");
    if (child == 0) {
        signal->notify();
        const bool rejected = !first && !gate.register_cpu_owner() && !gate.submit().valid() &&
                              checked(executor).progress() == Gate::CpuProgress::kRejected &&
                              !signal->wait_for(0us);
        ::_exit(rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "inherited CPU methods reject without parent transport mutation");
    signal->notify();
    require(checked(executor).wait_until(Clock::now() + 1s).has_activity && !ready(submitted) &&
                signal->wait_for(0us) && !other_signal->wait_for(0us),
            "actor wake is local hint, neither reply nor other consumer broadcast");
    executor.reset();
    require(first.detach() && !second.detach(),
            "last eligible CPU actor cannot abandon accepted work");
    auto migrated = gate.try_acquire_cpu(second);
    require(migrated.has_value(), "migration obtains new thread-bound CPU lease");
    executor.emplace(std::move(checked(migrated)));
    require(executor && executor->progress() == Gate::CpuProgress::kPending && !ready(submitted),
            "migration retains exact in-flight request");
    peer.reply(response(TaskExecutionProtocolResponseKind::kExecutionLease));
    bool has_reply = false;
    const auto deadline = Clock::now() + 1s;
    while (!has_reply && Clock::now() < deadline)
        has_reply = executor->wait_until(deadline).has_reply;
    require(has_reply && !ready(submitted),
            "actual socket readiness alone cannot publish original future");
    const auto granted = resolve(*executor, std::move(submitted));
    require(granted.claim.status == CooperativeClaimStatus::kGranted &&
                granted.claim.execution.task_id == kExecution.task_id &&
                granted.claim.execution.generation == kExecution.generation &&
                granted.claim.timing.enqueued_nanoseconds > 0 && signal->wait_for(0us) &&
                other_signal->wait_for(0us),
            "validated actual result precedes independent original signals and timing");
    auto complete = gate.complete(kExecution);
    require(executor->progress() == Gate::CpuProgress::kPending &&
                peer.receive().operation == TaskExecutionProtocolOperation::kExecutionComplete,
            "accepted terminal report starts");
    peer.gate.reset();
    require(!ready(complete), "facade retirement does not fake actual terminal ACK");
    auto acknowledged = response(TaskExecutionProtocolResponseKind::kState);
    acknowledged.state = TaskProtocolState::kCompleted;
    peer.reply(acknowledged);
    require(resolve(*executor, std::move(complete)),
            "retained CPU handles finish actual report after facade retirement");
    executor.reset();
    require(second.detach(), "last CPU owner detaches only after actual drain");
}
void test_queue_bound_and_fusion() {
    Peer peer{{.max_queued_requests = 2, .max_cpu_owners = 1}};
    auto owner = peer.gate->register_cpu_owner();
    auto first_signal = peer.gate->subscribe_progress();
    auto second_signal = peer.gate->subscribe_progress();
    require(first_signal && second_signal && !peer.gate->subscribe_progress(),
            "subscriber bound is enforced");
    second_signal.reset();
    require(peer.gate->subscribe_progress() != nullptr,
            "expired subscriber slot reuses cold capacity");
    auto old = peer.gate->complete(kExecution);
    auto next = peer.gate->submit_and_claim({.should_measure_timing = true});
    require(old.valid() && next.valid() && !peer.gate->submit().valid(),
            "queued public bound preserves both originals");
    auto executor = peer.gate->try_acquire_cpu(owner);
    require(executor && executor->progress() == Gate::CpuProgress::kPending,
            "pair starts in one bounded step");
    const auto request = peer.receive();
    require(request.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire &&
                request.task_id == 17 && request.generation == 29 && !ready(old) && !ready(next),
            "one actual wire, no fabricated early original results");
    auto queued = peer.gate->submit();
    auto other = peer.gate->submit();
    require(queued.valid() && other.valid() && !peer.gate->submit().valid() &&
                peer.gate->owner_pump_snapshot().outstanding_requests == 4,
            "active originals do not consume legacy queued slots, all remain bounded");
    auto handoff = response(TaskExecutionProtocolResponseKind::kExecutionHandoff);
    handoff.task_id = 19;
    handoff.generation = 31;
    peer.reply(handoff);
    require(checked(executor).progress() == Gate::CpuProgress::kCompleted && ready(old) &&
                ready(next) && old.get(),
            "actual combined response settles old before successor originals");
    const auto successor = next.get();
    require(successor.claim.execution.task_id == 19 && successor.claim.execution.generation == 31 &&
                successor.claim.timing.enqueued_nanoseconds > 0,
            "successor keeps original identity and timing");
    for (auto* future : {&queued, &other}) {
        require(checked(executor).progress() == Gate::CpuProgress::kPending &&
                    peer.receive().operation == TaskExecutionProtocolOperation::kSubmit,
                "independent queued originals retain order");
        peer.reply(response(TaskExecutionProtocolResponseKind::kAccepted));
        require(resolve(checked(executor), std::move(*future)) == 17,
                "accepted original ID is decoded");
    }
    peer.gate->shutdown();
    require(!peer.gate->submit().valid() && !peer.gate->subscribe_progress() &&
                peer.gate->owner_pump_snapshot().is_closed,
            "explicit close rejects new work without blocking drain");
    executor.reset();
    require(owner.detach(), "settled queue closes cleanly");
}
void test_quarantine_and_owner_loss() {
    {
        Peer peer;
        auto owner = peer.gate->register_cpu_owner();
        auto executor = peer.gate->try_acquire_cpu(owner);
        auto old = peer.gate->complete(kExecution);
        auto next = peer.gate->submit_and_claim();
        auto never_published = peer.gate->submit();
        require(executor && executor->progress() == Gate::CpuProgress::kPending &&
                    peer.receive().operation ==
                        TaskExecutionProtocolOperation::kExecutionCompleteAcquire,
                "negative pair publishes once");
        auto rejected = response(TaskExecutionProtocolResponseKind::kError);
        rejected.error = TaskProtocolErrorCode::kUnknownTask;
        auto observer = std::async(std::launch::async, [&] {
            old.wait();
            return peer.gate->remote_outcome_uncertain() && !peer.gate->submit().valid();
        });
        peer.reply(rejected);
        require(checked(executor).progress() == Gate::CpuProgress::kQuarantined && observer.get() &&
                    ready(old) && ready(next) && !old.get() &&
                    next.get().claim.status != CooperativeClaimStatus::kGranted &&
                    peer.gate->owner_pump_snapshot().is_quarantined && !peer.gate->submit().valid(),
                "actual failed old terminal quarantines without positive successor");
        require(checked(executor).progress() == Gate::CpuProgress::kCompleted &&
                    ready(never_published) && !never_published.get(),
                "unpublished queued original rejects during containment");
        require(peer.gate->owner_pump_snapshot().is_quarantined,
                "one rejected original completion never clears session quarantine");
        peer.no_request();
        executor.reset();
        require(owner.detach() && peer.gate->owner_pump_snapshot().is_quarantined &&
                    !peer.gate->register_cpu_owner(),
                "CPU detach after settled negative outcomes cannot repair domain quarantine");
    }
    {
        Peer peer;
        std::future<std::optional<glimmer::core::TaskId>> abandoned;
        {
            auto owner = peer.gate->register_cpu_owner();
            abandoned = peer.gate->submit();
            require(abandoned.valid() && !owner.detach(),
                    "busy last CPU owner cannot detach behind metadata producer");
        }
        require(peer.gate->owner_pump_snapshot().is_quarantined && !ready(abandoned) &&
                    !peer.gate->submit().valid(),
                "premature owner destruction poisons, never fabricates drain or ACK");
    }
}
bool registered(void*, glimmer::core::TaskId) noexcept {
    return true;
}
void rollback(void*, glimmer::core::TaskId) noexcept {}
void test_all_public_methods_on_actual_service() {
    Directory directory;
    auto admission = std::make_unique<TaskExecutionAdmissionService>(
        1024,
        glimmer::core::SchedulerOptions{.scheduling_policy =
                                            glimmer::core::SchedulingPolicy::kPriority},
        5000ms, true, nullptr, nullptr, nullptr, nullptr, kExtent);
    TaskExecutionEndpoint endpoint{std::move(admission), registered, rollback, nullptr};
    UnixSocketTaskExecutionServer server{endpoint,
                                         {.socket_path = directory.path + "/control.sock",
                                          .allowed_uid = std::nullopt,
                                          .io_timeout_ms = 1000}};
    require(server.start(), "actual authenticated service starts");
    std::jthread acceptor([&](const std::stop_token& stop) {
        while (!stop.stop_requested())
            require(
                server.serve_one_for(10) != UnixSocketServeStatus::kError || stop.stop_requested(),
                "actual acceptor stays healthy");
    });
    {
        auto gate = Gate::create_owner_pumped({.control_socket = directory.path + "/control.sock"},
                                              {}, kExtent);
        require(gate != nullptr, "public factory binds real process-authenticated service");
        auto owner = gate->register_cpu_owner();
        auto executor = gate->try_acquire_cpu(owner);
        require(executor.has_value(), "actual CPU owner leases");
        const auto id = resolve(checked(executor), gate->submit());
        require(id.has_value(), "public submit original ID");
        auto claim = resolve(checked(executor), gate->claim(checked(id), true));
        require(claim.status == CooperativeClaimStatus::kGranted &&
                    claim.timing.enqueued_nanoseconds > 0 &&
                    resolve(checked(executor), gate->retreat_requested(claim.execution)) == false &&
                    resolve(checked(executor), gate->heartbeat(claim.execution)) &&
                    resolve(checked(executor), gate->suspend(claim.execution)),
                "public claim/probe/heartbeat/suspend have actual results");
        claim = resolve(checked(executor), gate->claim(checked(id)));
        require(claim.status == CooperativeClaimStatus::kGranted &&
                    resolve(checked(executor), gate->complete(claim.execution)),
                "fresh resumed generation completes");
        auto acquired = resolve(checked(executor), gate->submit_and_claim());
        require(acquired.claim.status == CooperativeClaimStatus::kGranted &&
                    resolve(checked(executor), gate->fail(acquired.claim.execution)),
                "public failed physical-declared report acknowledges");
        acquired = resolve(checked(executor), gate->submit_and_claim());
        require(acquired.claim.status == CooperativeClaimStatus::kGranted &&
                    resolve(checked(executor), gate->suspend(acquired.claim.execution)) &&
                    resolve(checked(executor), gate->cancel_stopped(acquired.claim.execution)),
                "public stopped cancellation acknowledges");
        const auto queued = resolve(checked(executor), gate->submit());
        require(queued && resolve(checked(executor), gate->cancel_pending(*queued)),
                "never-dispatched cancellation acknowledges");
        auto epoch = resolve(checked(executor), gate->acquire_native_epoch({.extent = kExtent}));
        require(epoch.status == NativeEpochClaimStatus::kGranted &&
                    resolve(checked(executor), gate->retreat_native_epoch(epoch.grant)) == false &&
                    resolve(checked(executor), gate->heartbeat_native_epoch(epoch.grant)) &&
                    resolve(checked(executor), gate->finish_native_epoch(epoch.grant, 2, false)),
                "public native acquire/probe/heartbeat/complete preserve exact extent");
        epoch = resolve(checked(executor), gate->acquire_native_epoch({.extent = kExtent}));
        require(epoch.status == NativeEpochClaimStatus::kGranted &&
                    resolve(checked(executor), gate->finish_native_epoch(epoch.grant, 2, true)),
                "native failed report acknowledges its own count");
        const auto blocker = resolve(checked(executor), gate->submit_and_claim());
        epoch = resolve(checked(executor), gate->acquire_native_epoch({.extent = kExtent}));
        require(blocker.claim.status == CooperativeClaimStatus::kGranted &&
                    epoch.status == NativeEpochClaimStatus::kPending &&
                    resolve(checked(executor), gate->claim_native_epoch(epoch.grant)).status ==
                        NativeEpochClaimStatus::kPending &&
                    resolve(checked(executor), gate->cancel_native_epoch(epoch.grant)) &&
                    resolve(checked(executor), gate->complete(blocker.claim.execution)),
                "pending native claim/cancel cannot bypass existing owner");
        const auto status = endpoint.lease_status();
        require(status && status->running_tasks == 0 && status->queued_tasks == 0 &&
                    gate->owner_pump_snapshot().outstanding_requests == 0,
                "actual service and original promises drain");
        gate->shutdown();
        executor.reset();
        require(owner.detach(), "actual last CPU actor retires after drain");
    }
    require(endpoint.begin_shutdown(), "actual service stops admitting work");
    acceptor.request_stop();
    acceptor.join();
    server.stop();
}
void test_validation_and_private_roles() {
    require(!Gate::create_owner_pumped({}, {}) &&
                !Gate::create_owner_pumped({.control_socket = "/nonexistent/glimmer"},
                                           {.max_queued_requests = 0}) &&
                !Gate::create_owner_pumped({.control_socket = "/nonexistent/glimmer"},
                                           {.max_cpu_owners = 0}) &&
                !Gate::create_owner_pumped({.control_socket = "/nonexistent/glimmer"},
                                           {.transport_timeout = 0ms}) &&
                !Gate::create_owner_pumped({.control_socket = "/nonexistent/glimmer"}, {},
                                           NativeEpochExtent{}),
            "invalid cold options reject before domain acceptance and never legacy fallback");
    ProcessControlMailbox mailbox;
    const auto producer = mailbox.register_producer();
    require(!producer, "uninitialized producer rejects");
}
}  // namespace
int main() {
    // Initialize lazy thread-runtime bookkeeping before resource baselines.
    std::thread warmup([] {});
    warmup.join();
    const auto descriptors = inventory("/proc/self/fd");
    const auto threads = inventory("/proc/self/task");
    test_validation_and_private_roles();
    test_wait_migration_and_retirement();
    test_queue_bound_and_fusion();
    test_quarantine_and_owner_loss();
    test_all_public_methods_on_actual_service();
    const auto cleanup_deadline = Clock::now() + 1s;
    while (inventory("/proc/self/task") != threads && Clock::now() < cleanup_deadline)
        std::this_thread::yield();
    require(inventory("/proc/self/fd") == descriptors && inventory("/proc/self/task") == threads,
            "all success/rejection/fork/containment CPU resources retire");
}
