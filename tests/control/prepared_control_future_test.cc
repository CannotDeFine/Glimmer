#include "internal/process_control_mailbox.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <string>
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
using namespace std::chrono_literals;
using Clock = RemoteControlOperation::Clock;
constexpr NativeEpochExtent kExtent{8, 4, 2, 1'000'000};

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class Descriptor final {
   public:
    explicit Descriptor(int fd) : fd_(fd) {
        require(fd >= 0, "owned descriptor must create");
    }
    ~Descriptor() {
        require(::close(fd_) == 0, "owned descriptor must close");
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
    explicit Peer(ProcessControlMailbox& mailbox)
        : listener_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {
        std::array<char, 48> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-control-future-XXXXXX";
        std::memcpy(pattern.data(), prefix.data(), prefix.size());
        require(::mkdtemp(pattern.data()) != nullptr, "owned directory must create");
        directory_ = pattern.data();
        path_ = directory_ + "/control.sock";
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        require(path_.size() < sizeof(address.sun_path), "owned path must fit");
        std::memcpy(address.sun_path, path_.c_str(), path_.size() + 1);
        require(::bind(listener_.get(), reinterpret_cast<const sockaddr*>(&address),
                       sizeof(address)) == 0 &&
                    ::listen(listener_.get(), 4) == 0,
                "owned socket must listen");
        auto status = OwnedExchangeColdStatus::kRejected;
        std::thread startup([&] { status = mailbox.initialize({path_, 1000ms}); });
        peer_ = std::make_unique<Descriptor>(
            ::accept4(listener_.get(), nullptr, nullptr, SOCK_CLOEXEC));
        std::string open;
        for (std::size_t index = 0; index < kOwnerControlOpen.size(); ++index) {
            char byte = 0;
            require(::recv(peer_->get(), &byte, 1, 0) == 1, "cold OPEN must arrive");
            open.push_back(byte);
        }
        require(open == kOwnerControlOpen &&
                    ::send(peer_->get(), kOwnerControlReady.data(), kOwnerControlReady.size(),
                           MSG_DONTWAIT | MSG_NOSIGNAL) ==
                        static_cast<ssize_t>(kOwnerControlReady.size()),
                "V3 must negotiate explicitly");
        require(channel_.create_server({17, 29}, SharedControlMode::kReplyHintV3) &&
                    channel_.send_offer(peer_->get(), 1000ms),
                "sealed offer must send");
        startup.join();
        require(status == OwnedExchangeColdStatus::kReady, "cold startup must finish");
    }
    ~Peer() {
        require(::unlink(path_.c_str()) == 0 && ::rmdir(directory_.c_str()) == 0,
                "only owned socket and empty directory must remove");
    }
    Peer(const Peer&) = delete;
    Peer& operator=(const Peer&) = delete;
    std::string receive() {
        std::string line;
        require(
            channel_.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine,
            "actual head must arrive");
        return line;
    }
    void no_request() {
        // Call only after actual response publication: a pending old request
        // is not an idle-slot oracle in the server-side channel.
        std::string line;
        require(channel_.receive_request(peer_->get(), 1, line) == SharedControlReadStatus::kIdle,
                "no independent successor may publish");
    }
    void reply(std::string_view line) {
        require(channel_.publish_response_with_hint(peer_->get(), line),
                "actual body must publish before hint");
    }

   private:
    Descriptor listener_;
    std::unique_ptr<Descriptor> peer_;
    SharedControlChannel channel_;
    std::string directory_;
    std::string path_;
};

template <typename Result>
bool ready(std::future<Result>& future) {
    return future.valid() && future.wait_for(0s) == std::future_status::ready;
}
void check_timing(CooperativeRequestTiming timing, bool measured) {
    require(measured ? timing.enqueued_nanoseconds > 0 &&
                           timing.started_nanoseconds >= timing.enqueued_nanoseconds &&
                           timing.finished_nanoseconds >= timing.started_nanoseconds
                     : timing.enqueued_nanoseconds == 0 && timing.started_nanoseconds == 0 &&
                           timing.finished_nanoseconds == 0,
            "original measured timing is ordered; unmeasured metadata stays zero");
}

template <typename Result>
Result scripted(RemoteCooperativeCommand command, std::string_view response,
                bool measured = false) {
    ProcessControlMailbox mailbox;
    Peer peer{mailbox};
    const auto owner = mailbox.register_owner();
    auto signal = std::make_shared<AsyncCooperativeLaunchGate::ProgressSignal>();
    auto prepared = PreparedControlFuture::prepare<Result>(command, {}, signal, measured);
    require(prepared != nullptr, "typed operation must prepare before acceptance");
    auto future = prepared->template take_future<Result>();
    require(future.valid() && !prepared->template take_future<Result>().valid(),
            "original future is obtained exactly once");
    auto ticket = mailbox.submit(owner, std::move(prepared));
    auto lease = mailbox.try_acquire(owner);
    const auto now = Clock::now();
    require(ticket && lease && lease->progress(now) == ControlMailboxProgress::kPending,
            "typed head publishes through one executor");
    const auto parsed = parse_task_execution_protocol_request(peer.receive());
    require(parsed.parsed() && checked(parsed.request).operation == command.operation,
            "exact prepared command must arrive");
    signal->notify();
    require(signal->wait_for(0us) && !ready(future) && !ticket.is_ready() &&
                checked(lease).progress(now) == ControlMailboxProgress::kPending,
            "spurious consumer hints cannot fabricate an original result");
    peer.reply(response);
    require(!ready(future) && checked(lease).progress(now) == ControlMailboxProgress::kCompleted &&
                ready(future) && ticket.is_ready() && signal->wait_for(0us),
            "real validation publishes original future before ticket and consumer hint");
    auto original = future.get();
    require(!future.valid() && ticket.take_result() && !ticket.take_result(),
            "original future and independent diagnostic ticket each consume once");
    lease.reset();
    require(mailbox.detach_owner(owner), "settled owner must detach");
    return original;
}

void test_all_original_types_and_timing() {
    require(scripted<std::optional<glimmer::core::TaskId>>(
                {TaskExecutionProtocolOperation::kSubmit}, "GLIMMER_TASK_EXEC_V1 OK 17\n") == 17,
            "original submit result is retained");
    for (const bool measured : {false, true}) {
        const auto claim = scripted<CooperativeClaim>(
            {TaskExecutionProtocolOperation::kExecutionClaim, {17, 0}},
            "GLIMMER_TASK_EXEC_V1 EXEC_LEASE 17 default 1 1 1 29\n", measured);
        require(
            claim.status == CooperativeClaimStatus::kGranted && claim.execution.generation == 29,
            "claim retains actual generation");
        check_timing(claim.timing, measured);
        const auto submission = scripted<CooperativeSubmission>(
            {TaskExecutionProtocolOperation::kExecutionAcquire},
            "GLIMMER_TASK_EXEC_V1 EXEC_LEASE 17 default 1 1 1 29\n", measured);
        require(submission.task_id == 17 && submission.claim.execution.generation == 29,
                "submission retains actual authorization");
        check_timing(submission.claim.timing, measured);
        const auto handoff = scripted<CooperativeHandoff>(
            {TaskExecutionProtocolOperation::kExecutionCompleteAcquire, {17, 29}},
            "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 17 29 18 30\n", measured);
        require(handoff.is_completed && handoff.submission.task_id == 18 &&
                    handoff.submission.claim.execution.generation == 30,
                "combined result retains old ACK and independent fresh generation");
        check_timing(handoff.submission.claim.timing, measured);
    }
    require(scripted<std::optional<bool>>({TaskExecutionProtocolOperation::kRetreat, {17, 29}},
                                          "GLIMMER_TASK_EXEC_V1 RETREAT 17 29 0\n") == false,
            "false retreat is an actual optional result, not missing response");
    require(scripted<bool>({TaskExecutionProtocolOperation::kExecutionComplete, {17, 29}},
                           "GLIMMER_TASK_EXEC_V1 STATE 17 COMPLETED\n"),
            "terminal future requires an actual ACK");

    ProcessControlMailbox mailbox;
    Peer peer{mailbox};
    const auto owner = mailbox.register_owner();
    auto prepared = PreparedControlFuture::prepare<NativeEpochClaim>(
        RemoteNativeCommand{.operation = NativeEpochOperation::kAcquire,
                            .settings = {.extent = kExtent}},
        {}, kExtent);
    require(prepared != nullptr, "native promise must prepare");
    auto future = prepared->take_future<NativeEpochClaim>();
    auto ticket = mailbox.submit(owner, std::move(prepared));
    auto lease = mailbox.try_acquire(owner);
    require(ticket && lease && lease->progress() == ControlMailboxProgress::kPending,
            "native original request publishes");
    const auto request = parse_native_epoch_request(peer.receive());
    require(request && request->request_sequence != 0, "native request must correlate");
    NativeEpochResponse response{};
    response.kind = NativeEpochResponseKind::kPending;
    response.grant = {.execution = {17, 0}, .extent = kExtent};
    response.request_sequence = checked(request).request_sequence;
    const auto wire = format_native_epoch_response(response);
    require(wire.has_value() && !ready(future), "native pending is not a hint-only result");
    peer.reply(checked(wire));
    require(checked(lease).progress() == ControlMailboxProgress::kCompleted && ready(future),
            "canonical native original result publishes after correlation validation");
    const auto original = future.get();
    require(original.status == NativeEpochClaimStatus::kPending &&
                original.grant.execution.task_id == 17 && ticket.take_result(),
            "known pending native ownership must not be erased");
    lease.reset();
    require(mailbox.detach_owner(owner), "native CPU owner detaches without claiming GPU stop");
}

void test_future_before_migrated_successor() {
    ProcessControlMailbox mailbox{{.max_outstanding_requests = 2, .max_owners = 2}};
    Peer peer{mailbox};
    const auto first = mailbox.register_owner();
    const auto second = mailbox.register_owner();
    auto old_signal = std::make_shared<AsyncCooperativeLaunchGate::ProgressSignal>();
    auto next_signal = std::make_shared<AsyncCooperativeLaunchGate::ProgressSignal>();
    auto old = PreparedControlFuture::prepare<bool>(
        {TaskExecutionProtocolOperation::kExecutionComplete, {17, 29}}, {}, old_signal);
    auto next = PreparedControlFuture::prepare<CooperativeSubmission>(
        {TaskExecutionProtocolOperation::kExecutionAcquire}, {}, next_signal);
    require(old && next, "adjacent independent original records prepare");
    auto old_future = old->take_future<bool>();
    auto next_future = next->take_future<CooperativeSubmission>();
    auto old_ticket = mailbox.submit(first, std::move(old));
    auto next_ticket = mailbox.submit(second, std::move(next));
    auto lease = mailbox.try_acquire(first);
    const auto now = Clock::now();
    require(old_ticket && next_ticket && lease &&
                lease->progress(now) == ControlMailboxProgress::kPending,
            "only actual old report publishes");
    const auto request = parse_task_execution_protocol_request(peer.receive());
    require(request.parsed() && checked(request.request).operation ==
                                    TaskExecutionProtocolOperation::kExecutionComplete,
            "old terminal precedes new acquisition");
    for (std::size_t index = 0; index < 16; ++index)
        require(checked(lease).progress(now) == ControlMailboxProgress::kPending &&
                    !ready(old_future) && !ready(next_future),
                "withheld real ACK keeps both unready");
    lease.reset();
    peer.reply("GLIMMER_TASK_EXEC_V1 STATE 17 COMPLETED\n");
    peer.no_request();
    auto publishing = mailbox.try_acquire(second);
    require(publishing && publishing->progress(now) == ControlMailboxProgress::kCompleted &&
                ready(old_future) && !ready(next_future) && old_signal->wait_for(0us) &&
                !next_signal->wait_for(0us),
            "only old original result/signal publishes");
    peer.no_request();
    publishing.reset();
    std::thread migrated([&] {
        auto active = mailbox.try_acquire(first);
        require(active && ready(old_future) &&
                    active->progress(now) == ControlMailboxProgress::kPending,
                "new executor cannot acquire before original old future publication");
        const auto successor = parse_task_execution_protocol_request(peer.receive());
        require(successor.parsed() && successor.request->operation ==
                                          TaskExecutionProtocolOperation::kExecutionAcquire,
                "fresh independent successor arrives exactly once after old future");
        peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_LEASE 18 default 1 1 1 30\n");
        require(active->progress(now) == ControlMailboxProgress::kCompleted,
                "successor still requires actual fresh grant");
    });
    migrated.join();
    require(old_future.get() && next_future.get().claim.execution.generation == 30 &&
                old_ticket.take_result() && next_ticket.take_result() && next_signal->wait_for(0us),
            "independent consumers receive original typed outcomes exactly once");
    require(mailbox.detach_owner(first) && mailbox.detach_owner(second),
            "both settled owners detach");
}

void test_failed_terminal_and_queued_future() {
    for (const bool has_timeout : {false, true}) {
        ProcessControlMailbox mailbox;
        Peer peer{mailbox};
        const auto owner = mailbox.register_owner();
        auto old = PreparedControlFuture::prepare<bool>(
            {TaskExecutionProtocolOperation::kExecutionComplete, {17, 29}}, {});
        auto next = PreparedControlFuture::prepare<NativeEpochClaim>(
            RemoteNativeCommand{NativeEpochOperation::kClaim,
                                {.execution = {18, 0}, .extent = kExtent}},
            {}, kExtent);
        require(old && next, "failure records prepare");
        auto old_future = old->take_future<bool>();
        auto next_future = next->take_future<NativeEpochClaim>();
        auto old_ticket = mailbox.submit(owner, std::move(old));
        auto next_ticket = mailbox.submit(owner, std::move(next));
        auto lease = mailbox.try_acquire(owner);
        const auto now = Clock::now();
        require(old_ticket && next_ticket && lease &&
                    lease->progress(now) == ControlMailboxProgress::kPending,
                "failed-terminal fixture must publish actual old report");
        static_cast<void>(peer.receive());
        if (!has_timeout) peer.reply("GLIMMER_TASK_EXEC_V1 ERROR INVALID_REQUEST\n");
        require(checked(lease).progress(has_timeout ? now + 1000ms : now) ==
                        ControlMailboxProgress::kQuarantined &&
                    ready(old_future) && !old_future.get() && mailbox.snapshot().is_quarantined,
                "negative terminal is never a release ACK");
        const auto outcome = old_ticket.take_result();
        require(outcome && outcome->was_published &&
                    outcome->step == (has_timeout ? RemoteControlStep::kUncertain
                                                  : RemoteControlStep::kFinished),
                "definite failure stays distinct from ambiguous timeout");
        require(checked(lease).progress(now) == ControlMailboxProgress::kCompleted &&
                    ready(next_future),
                "never-published successor settles negatively without replay");
        const auto rejected = next_future.get();
        const auto next_outcome = next_ticket.take_result();
        require(rejected.status == NativeEpochClaimStatus::kUncertain &&
                    rejected.grant.execution.task_id == 18 && next_outcome &&
                    !next_outcome->was_published &&
                    next_outcome->step == RemoteControlStep::kRejected,
                "queued original future preserves known native pending ownership");
        if (!has_timeout) peer.no_request();
        require(checked(lease).progress(now) == ControlMailboxProgress::kQuarantined,
                "zero queued work is not successful control drain");
        lease.reset();
        require(mailbox.detach_owner(owner) && !mailbox.register_owner(),
                "last CPU detach cannot repair terminal uncertainty");
    }
}

void test_preparation_rejection_and_fork() {
    require(!PreparedControlFuture::prepare<bool>(
                {TaskExecutionProtocolOperation::kExecutionAcquire}, {}) &&
                !PreparedControlFuture::prepare<bool>({TaskExecutionProtocolOperation::kStats}, {}),
            "wrong original type and unsupported command reject before acceptance");
    ProcessControlMailbox mailbox{{.max_outstanding_requests = 1, .max_owners = 1}};
    Peer peer{mailbox};
    const auto owner = mailbox.register_owner();
    auto not_obtained = PreparedControlFuture::prepare<CooperativeSubmission>(
        {TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    require(!mailbox.submit(owner, std::move(not_obtained)),
            "acceptance requires an original future");
    auto accepted = PreparedControlFuture::prepare<CooperativeSubmission>(
        {TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    require(accepted && !accepted->take_future<bool>().valid(),
            "wrong retrieval cannot consume original future");
    auto future = accepted->take_future<CooperativeSubmission>();
    auto ticket = mailbox.submit(owner, std::move(accepted));
    auto rejected = PreparedControlFuture::prepare<CooperativeSubmission>(
        {TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    auto rejected_future = rejected->take_future<CooperativeSubmission>();
    require(ticket && !mailbox.submit(owner, std::move(rejected)),
            "active head counts in exact bound");
    require(ready(rejected_future),
            "rejected private promise is retired, never a valid public acceptance");
    bool has_broken_promise = false;
    try {
        static_cast<void>(rejected_future.get());
    } catch (const std::future_error& error) {
        has_broken_promise = error.code() == std::future_errc::broken_promise;
    }
    require(has_broken_promise, "pre-acceptance rejection must not publish a control result");
    auto inherited = PreparedControlFuture::prepare<CooperativeSubmission>(
        {TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    const pid_t child = ::fork();
    require(child >= 0, "owned child must create");
    if (child == 0) ::_exit(inherited->take_future<CooperativeSubmission>().valid() ? 1 : 0);
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "inherited future retrieval rejects; child is reaped");
    auto lease = mailbox.try_acquire(owner);
    require(lease && lease->progress() == ControlMailboxProgress::kPending,
            "parent head remains usable");
    static_cast<void>(peer.receive());
    mailbox.close();
    auto closed = PreparedControlFuture::prepare<CooperativeSubmission>(
        {TaskExecutionProtocolOperation::kExecutionAcquire}, {});
    auto closed_future = closed->take_future<CooperativeSubmission>();
    require(!mailbox.submit(owner, std::move(closed)) && ready(closed_future),
            "closed rejects new acceptance");
    peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_LEASE 17 default 1 1 1 29\n");
    require(checked(lease).progress() == ControlMailboxProgress::kCompleted && ready(future),
            "closed drains actual accepted original future");
    lease.reset();
    require(mailbox.detach_owner(owner) && future.get().claim.execution.generation == 29 &&
                ticket.take_result(),
            "exact parent authorization survives rejected fork and close");
}

std::size_t descriptors() {
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "owned inventory opens");
    std::size_t count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    require(errno == 0 && ::closedir(directory) == 0, "owned inventory closes");
    return count;
}
}  // namespace

int main() {
    const auto before = descriptors();
    test_all_original_types_and_timing();
    test_future_before_migrated_successor();
    test_failed_terminal_and_queued_future();
    test_preparation_rejection_and_fork();
    require(descriptors() == before, "all original-future CPU descriptors retire");
}
