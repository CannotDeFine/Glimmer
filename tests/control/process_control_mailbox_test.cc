#include "internal/process_control_mailbox.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <stdexcept>
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
        constexpr std::string_view prefix = "/tmp/glimmer-process-mailbox-XXXXXX";
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
        require(status == OwnedExchangeColdStatus::kReady &&
                    mailbox.initialize({path_, 1000ms}) == OwnedExchangeColdStatus::kRejected,
                "cold startup must finish exactly once before acceptance");
    }
    ~Peer() {
        require(::unlink(path_.c_str()) == 0 && ::rmdir(directory_.c_str()) == 0,
                "only owned socket and empty directory must remove");
    }
    Peer(const Peer&) = delete;
    Peer& operator=(const Peer&) = delete;
    TaskExecutionProtocolRequest receive() {
        std::string line;
        require(
            channel_.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine,
            "one actual head must arrive");
        const auto request = parse_task_execution_protocol_request(line);
        require(request.parsed(), "actual canonical head must parse");
        return checked(request.request);
    }
    void no_request() {
        std::string line;
        require(channel_.receive_request(peer_->get(), 1, line) == SharedControlReadStatus::kIdle,
                "unacknowledged head cannot publish independent successor");
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

std::unique_ptr<RemoteControlOperation> acquire() {
    return RemoteControlOperation::prepare(
        RemoteCooperativeCommand{TaskExecutionProtocolOperation::kExecutionAcquire}, {});
}
ControlRequestOutcome take(ControlCompletionTicket& ticket) {
    require(ticket.is_ready(), "completion ticket requires actual ready outcome");
    const auto result = ticket.take_result();
    require(result.has_value() && !ticket.take_result(), "outcome consumes exactly once");
    return checked(result);
}

void test_fifo_migration_and_close() {
    ProcessControlMailbox mailbox{{.max_outstanding_requests = 2, .max_owners = 2}};
    Peer peer{mailbox};
    const auto first = mailbox.register_owner();
    const auto second = mailbox.register_owner();
    require(first && second && !mailbox.register_owner(), "owner capacity is bounded");
    auto completed = mailbox.submit(
        first,
        RemoteControlOperation::prepare(
            RemoteCooperativeCommand{TaskExecutionProtocolOperation::kExecutionComplete, {17, 29}},
            {}));
    auto successor = mailbox.submit(second, acquire());
    auto copied = completed;
    require(completed && successor && !completed.is_ready() && !successor.is_ready() &&
                !mailbox.submit(first, acquire()),
            "active and queued acceptance share exact bound");
    auto lease = mailbox.try_acquire(first);
    require(lease && !mailbox.try_acquire(second) && !mailbox.detach_owner(first),
            "one registered live lease owns progress");
    const auto now = Clock::now();
    require(checked(lease).progress(now) == ControlMailboxProgress::kPending,
            "only FIFO head publishes");
    const auto request = peer.receive();
    require(request.operation == TaskExecutionProtocolOperation::kExecutionComplete &&
                request.task_id == 17 && request.generation == 29,
            "exact old terminal identity must precede successor");
    const auto wait = checked(lease).wait_state();
    require(wait && wait->deadline == now + 1000ms, "published deadline remains fixed");
    auto foreign_thread = ControlMailboxProgress::kIdle;
    std::thread contender([&] { foreign_thread = lease->progress(now); });
    contender.join();
    require(foreign_thread == ControlMailboxProgress::kRejected,
            "lease progress is acquiring-thread-bound");
    for (std::size_t index = 0; index < 16; ++index)
        require(checked(lease).progress(now) == ControlMailboxProgress::kPending &&
                    !completed.is_ready() && !successor.is_ready(),
                "withheld actual ACK cannot fulfill results");
    lease.reset();
    require(mailbox.detach_owner(first) && !mailbox.try_acquire(first) &&
                !mailbox.submit(first, acquire()) && !mailbox.detach_owner(second),
            "nonlast detach retains head; last detach cannot strand accepted work");
    const auto replacement = mailbox.register_owner();
    require(replacement && !mailbox.try_acquire(first),
            "slot reuse cannot revive stale owner generation");
    require(mailbox.detach_owner(replacement), "unused replacement owner detaches");
    std::thread migrated([&] {
        auto active = mailbox.try_acquire(second);
        require(active && active->wait_state()->descriptor == wait->descriptor &&
                    active->wait_state()->deadline == wait->deadline &&
                    active->progress(now) == ControlMailboxProgress::kPending,
                "migration preserves exact endpoint/head/ticket/deadline without replay");
        peer.reply("GLIMMER_TASK_EXEC_V1 STATE 17 COMPLETED\n");
        peer.no_request();
        require(active->progress(now) == ControlMailboxProgress::kCompleted &&
                    completed.is_ready() && !successor.is_ready(),
                "old actual outcome publishes before independent successor");
        require(active->progress(now) == ControlMailboxProgress::kPending,
                "independent successor publishes only on subsequent bounded step");
        require(peer.receive().operation == TaskExecutionProtocolOperation::kExecutionAcquire,
                "exact accepted successor must arrive once");
        peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_LEASE 18 default 1 1 1 30\n");
        require(active->progress(now) == ControlMailboxProgress::kCompleted,
                "successor needs actual fresh authorization");
    });
    migrated.join();
    const auto old = take(completed);
    require(old.step == RemoteControlStep::kFinished && old.was_published &&
                std::get<bool>(old.result) && !copied.take_result(),
            "copied ticket cannot duplicate actual old ACK");
    const auto next = take(successor);
    require(std::get<CooperativeSubmission>(next.result).claim.execution.generation == 30,
            "successor result retains new generation");
    auto final = mailbox.submit(
        second,
        RemoteControlOperation::prepare(
            RemoteCooperativeCommand{TaskExecutionProtocolOperation::kExecutionComplete, {18, 30}},
            {}));
    mailbox.close();
    require(final && !mailbox.submit(second, acquire()) && !mailbox.register_owner() &&
                !mailbox.detach_owner(second),
            "close rejects only new work and preserves final progress source");
    auto final_lease = mailbox.try_acquire(second);
    require(final_lease && final_lease->progress(now) == ControlMailboxProgress::kPending,
            "closed mailbox drains accepted final report");
    static_cast<void>(peer.receive());
    peer.reply("GLIMMER_TASK_EXEC_V1 STATE 18 COMPLETED\n");
    require(checked(final_lease).progress(now) == ControlMailboxProgress::kCompleted &&
                std::get<bool>(take(final).result),
            "actual final ACK must settle before last owner exit");
    final_lease.reset();
    require(mailbox.detach_owner(second) && mailbox.snapshot().unsettled_requests == 0 &&
                !mailbox.snapshot().is_quarantined,
            "clean closed drain retains no uncertain record");
}

void test_uncertainty_and_queued_rejection() {
    ProcessControlMailbox mailbox{{.max_outstanding_requests = 3, .max_owners = 1}};
    Peer peer{mailbox};
    const auto owner = mailbox.register_owner();
    auto head = mailbox.submit(owner, acquire());
    constexpr NativeEpochExtent extent{8, 4, 2, 1'000'000};
    const NativeEpochGrant pending{.execution = {17, 0}, .extent = extent};
    auto queued = mailbox.submit(
        owner, RemoteControlOperation::prepare(
                   RemoteNativeCommand{NativeEpochOperation::kClaim, pending}, {}, extent));
    auto last = mailbox.submit(owner, acquire());
    auto lease = mailbox.try_acquire(owner);
    const auto now = Clock::now();
    require(lease && lease->progress(now) == ControlMailboxProgress::kPending,
            "uncertain test publishes real head");
    static_cast<void>(peer.receive());
    require(checked(lease).progress(now + 1000ms) == ControlMailboxProgress::kQuarantined &&
                head.is_ready() && !queued.is_ready() && !last.is_ready(),
            "withheld reply timeout quarantines exact head without successor publication");
    const auto uncertain = take(head);
    require(uncertain.step == RemoteControlStep::kUncertain && uncertain.was_published &&
                mailbox.snapshot().is_quarantined && !mailbox.submit(owner, acquire()),
            "published uncertainty is permanent, not a new logical session");
    require(checked(lease).progress(now) == ControlMailboxProgress::kCompleted,
            "never-published claim settles without transport");
    const auto rejected = take(queued);
    require(rejected.step == RemoteControlStep::kRejected && !rejected.was_published &&
                std::get<NativeEpochClaim>(rejected.result).grant.execution.task_id == 17,
            "queued rejection preserves known pending identity without issuing claim");
    require(checked(lease).progress(now) == ControlMailboxProgress::kCompleted &&
                !take(last).was_published &&
                checked(lease).progress(now) == ControlMailboxProgress::kQuarantined,
            "all queued outcomes settle once; zero unsettled is not successful drain");
    lease.reset();
    require(mailbox.detach_owner(owner) && mailbox.snapshot().is_quarantined &&
                mailbox.snapshot().registered_owners == 0 && !mailbox.register_owner() &&
                mailbox.initialize({"/tmp/not-a-recovery", 1000ms}) ==
                    OwnedExchangeColdStatus::kRejected,
            "last CPU owner exit cannot reset retained quarantine or reconnect");
}

void test_fork_foreign_and_completion_lifetime() {
    ControlCompletionTicket retained;
    {
        ProcessControlMailbox mailbox{{.max_outstanding_requests = 1, .max_owners = 1}};
        Peer peer{mailbox};
        const auto owner = mailbox.register_owner();
        ProcessControlMailbox foreign;
        require(!foreign.try_acquire(owner) && !foreign.submit(owner, acquire()),
                "foreign retained mailbox identity cannot acquire or accept");
        retained = mailbox.submit(owner, acquire());
        auto lease = mailbox.try_acquire(owner);
        const auto now = Clock::now();
        require(lease && lease->progress(now) == ControlMailboxProgress::kPending,
                "actual fork head publishes");
        static_cast<void>(peer.receive());
        const pid_t child = ::fork();
        require(child >= 0, "owned fork child creates");
        if (child == 0) {
            const bool rejected =
                checked(lease).progress(now) == ControlMailboxProgress::kRejected &&
                !mailbox.try_acquire(owner) && !mailbox.register_owner() &&
                !mailbox.detach_owner(owner) && !mailbox.submit(owner, acquire()) &&
                !retained.take_result();
            lease.reset();
            ::_exit(rejected ? 0 : 1);
        }
        int status = 0;
        require(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork rejection child must terminate and be reaped");
        peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_LEASE 17 default 1 1 1 29\n");
        require(checked(lease).progress(now) == ControlMailboxProgress::kCompleted &&
                    retained.is_ready(),
                "child rejection cannot abandon or consume parent operation");
        const pid_t ready_child = ::fork();
        require(ready_child >= 0, "ready-result child creates");
        if (ready_child == 0) ::_exit(retained.is_ready() || retained.take_result() ? 1 : 0);
        require(::waitpid(ready_child, &status, 0) == ready_child && WIFEXITED(status) &&
                    WEXITSTATUS(status) == 0,
                "ready positive authority cannot be inherited through ticket");
        lease.reset();
        require(mailbox.detach_owner(owner), "settled owner detaches");
    }
    require(std::get<CooperativeSubmission>(take(retained).result).claim.execution.generation == 29,
            "completed CPU result outlives mailbox without borrowing destroyed endpoint");
    auto prepared = acquire();
    require(prepared->is_prepared() && prepared->reject_unpublished() && !prepared->is_prepared() &&
                !prepared->reject_unpublished() && !prepared->was_published() &&
                prepared->take_result(),
            "prepared-only rejection settles once before any transport and cannot reset");
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
    test_fifo_migration_and_close();
    test_uncertainty_and_queued_rejection();
    test_fork_foreign_and_completion_lifetime();
    require(descriptors() == before, "every normal/closed/quarantined/fork CPU descriptor retires");
}
