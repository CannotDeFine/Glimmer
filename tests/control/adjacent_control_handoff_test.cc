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
        constexpr std::string_view prefix = "/tmp/glimmer-control-handoff-XXXXXX";
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
    TaskExecutionProtocolRequest receive() {
        std::string line;
        require(
            channel_.receive_request(peer_->get(), 1000, line) == SharedControlReadStatus::kLine,
            "one actual wire must arrive");
        const auto parsed = parse_task_execution_protocol_request(line);
        require(parsed.parsed(), "actual canonical request must parse");
        return checked(parsed.request);
    }
    void no_request() {
        // An idle oracle is valid only after actual response publication.
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
template <typename Result>
struct Accepted {
    std::future<Result> future;
    ControlCompletionTicket ticket;
};
template <typename Result>
Accepted<Result> accept(ProcessControlMailbox& mailbox, const ControlOwnerToken& owner,
                        RemoteCooperativeCommand command, const LaunchGateOptions& options = {},
                        std::shared_ptr<AsyncCooperativeLaunchGate::ProgressSignal> signal = {}) {
    auto prepared = PreparedControlFuture::prepare<Result>(command, options, std::move(signal));
    require(prepared != nullptr, "owned original promise must prepare");
    auto future = prepared->template take_future<Result>();
    auto ticket = mailbox.submit(owner, std::move(prepared));
    require(future.valid() && ticket, "fixture original request must be accepted");
    return {std::move(future), std::move(ticket)};
}
struct Pair {
    Accepted<bool> old;
    Accepted<CooperativeSubmission> next;
};
Pair accept_pair(ProcessControlMailbox& mailbox, const ControlOwnerToken& first,
                 const ControlOwnerToken& second, CooperativeLaunchRequest settings = {},
                 const LaunchGateOptions& options = {}) {
    auto old = accept<bool>(
        mailbox, first, {TaskExecutionProtocolOperation::kExecutionComplete, {17, 29}}, options);
    auto next = accept<CooperativeSubmission>(
        mailbox, second, {TaskExecutionProtocolOperation::kExecutionAcquire, {}, settings},
        options);
    return {std::move(old), std::move(next)};
}
void check_pair_request(const TaskExecutionProtocolRequest& request) {
    require(request.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire &&
                request.task_id == 17 && request.generation == 29,
            "one wire combines only the exact accepted old report and adjacent acquisition");
}

void test_withheld_pair_migration_close_and_bounds() {
    ProcessControlMailbox mailbox{
        {.max_outstanding_requests = 2, .max_owners = 2, .should_combine_adjacent_requests = true}};
    Peer peer{mailbox};
    const auto first = mailbox.register_owner();
    const auto second = mailbox.register_owner();
    LaunchGateOptions options;
    options.tenant_id = "owned";
    options.tenant_weight = 3;
    options.task_priority = 5;
    auto pair = accept_pair(
        mailbox, first, second,
        {.relative_priority = 2, .work_units = 7, .should_measure_timing = true}, options);
    auto full = PreparedControlFuture::prepare<CooperativeSubmission>(
        {TaskExecutionProtocolOperation::kExecutionAcquire}, options);
    auto full_future = full->take_future<CooperativeSubmission>();
    require(full_future.valid() && !mailbox.submit(first, std::move(full)),
            "two originals count separately against capacity");
    auto lease = mailbox.try_acquire(first);
    const auto now = Clock::now();
    require(lease && lease->progress(now) == ControlMailboxProgress::kPending,
            "one prepared combined head must publish");
    const auto request = peer.receive();
    check_pair_request(request);
    require(request.admission.tenant_id == "owned" && request.admission.weight == 3 &&
                request.admission.priority == 7 && request.admission.work_units == 7,
            "candidate must copy exact already-owned successor admission");
    const auto wait = checked(lease).wait_state();
    require(wait && wait->deadline == now + 1000ms && mailbox.snapshot().unsettled_requests == 2,
            "one pending wire retains both bounded records and fixed deadline");
    for (std::size_t index = 0; index < 16; ++index)
        require(checked(lease).progress(now) == ControlMailboxProgress::kPending &&
                    !ready(pair.old.future) && !ready(pair.next.future),
                "withheld real combined reply cannot fabricate either original result");
    const pid_t child = ::fork();
    require(child >= 0, "owned fork child creates");
    if (child == 0) {
        const bool rejected = checked(lease).progress(now) == ControlMailboxProgress::kRejected;
        lease.reset();
        ::_exit(rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "inherited paired executor rejects and child is reaped");
    lease.reset();
    require(mailbox.detach_owner(first) && !mailbox.detach_owner(second),
            "nonlast migration retains pair; last detach cannot strand accepted work");
    mailbox.close();
    std::thread migrated([&] {
        auto active = mailbox.try_acquire(second);
        require(active && active->wait_state()->descriptor == wait->descriptor &&
                    active->wait_state()->deadline == wait->deadline,
                "migration keeps exact pending combined endpoint and deadline");
        peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 17 29 18 30\n");
        peer.no_request();
        require(active->progress(now) == ControlMailboxProgress::kCompleted &&
                    ready(pair.old.future) && ready(pair.next.future),
                "real combined validation publishes both original futures before retirement");
    });
    migrated.join();
    require(pair.old.future.get(), "old bool future retains actual ACK");
    const auto submission = pair.next.future.get();
    const auto timing = submission.claim.timing;
    require(submission.task_id == 18 && submission.claim.execution.generation == 30 &&
                timing.enqueued_nanoseconds > 0 &&
                timing.started_nanoseconds >= timing.enqueued_nanoseconds &&
                timing.finished_nanoseconds >= timing.started_nanoseconds,
            "new original future retains fresh authorization and its own timing");
    const auto old = pair.old.ticket.take_result();
    const auto next = pair.next.ticket.take_result();
    require(old && next && old->was_published && next->was_published &&
                std::get<bool>(old->result) &&
                std::get<CooperativeSubmission>(next->result).task_id == 18 &&
                !pair.old.ticket.take_result() && !pair.next.ticket.take_result(),
            "both diagnostic results refer to the actual combined wire and consume once");
    require(mailbox.detach_owner(second) && mailbox.snapshot().unsettled_requests == 0 &&
                !mailbox.snapshot().is_quarantined,
            "closed CPU drain settles both results before last owner exit");
}

void test_acknowledged_old_with_new_denial() {
    ProcessControlMailbox mailbox{{.should_combine_adjacent_requests = true}};
    Peer peer{mailbox};
    const auto owner = mailbox.register_owner();
    auto pair = accept_pair(mailbox, owner, owner);
    auto lease = mailbox.try_acquire(owner);
    require(lease && lease->progress() == ControlMailboxProgress::kPending,
            "denial pair publishes");
    check_pair_request(peer.receive());
    peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 17 29 0 0\n");
    require(
        checked(lease).progress() == ControlMailboxProgress::kCompleted && pair.old.future.get(),
        "new admission rejection cannot erase an actual old completion ACK");
    const auto denied = pair.next.future.get();
    require(denied.task_id == 0 && denied.claim.status == CooperativeClaimStatus::kFailed &&
                !mailbox.snapshot().is_quarantined,
            "no guessed new grant or lost old ACK");
    auto fresh = accept<CooperativeSubmission>(mailbox, owner,
                                               {TaskExecutionProtocolOperation::kExecutionAcquire});
    require(checked(lease).progress() == ControlMailboxProgress::kPending,
            "later request needs separate fresh authorization");
    require(peer.receive().operation == TaskExecutionProtocolOperation::kExecutionAcquire,
            "later acquisition is independent");
    peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_LEASE 19 default 1 1 1 31\n");
    require(checked(lease).progress() == ControlMailboxProgress::kCompleted &&
                fresh.future.get().claim.execution.generation == 31,
            "later generation comes only from actual reply");
    lease.reset();
    require(mailbox.detach_owner(owner), "settled denial owner detaches");
}

void test_pair_uncertainty_and_queued_rejection() {
    constexpr std::array<std::string_view, 4> responses{
        "GLIMMER_TASK_EXEC_V1 ERROR INVALID_REQUEST\n", "WRONG\n",
        "GLIMMER_TASK_EXEC_V1 EXEC_HANDOFF 16 29 18 30\n", ""};
    for (std::size_t index = 0; index < responses.size(); ++index) {
        ProcessControlMailbox mailbox{{.max_outstanding_requests = 3,
                                       .max_owners = 1,
                                       .should_combine_adjacent_requests = true}};
        Peer peer{mailbox};
        const auto owner = mailbox.register_owner();
        auto pair = accept_pair(mailbox, owner, owner);
        auto queued = PreparedControlFuture::prepare<NativeEpochClaim>(
            RemoteNativeCommand{NativeEpochOperation::kClaim,
                                {.execution = {19, 0}, .extent = kExtent}},
            {}, kExtent);
        auto queued_future = queued->take_future<NativeEpochClaim>();
        auto queued_ticket = mailbox.submit(owner, std::move(queued));
        auto lease = mailbox.try_acquire(owner);
        const auto now = Clock::now();
        require(queued_ticket && lease && lease->progress(now) == ControlMailboxProgress::kPending,
                "failure fixture publishes one real pair");
        check_pair_request(peer.receive());
        if (!responses[index].empty()) peer.reply(responses[index]);
        require(
            checked(lease).progress(responses[index].empty() ? now + 1000ms : now) ==
                    ControlMailboxProgress::kQuarantined &&
                !pair.old.future.get() && pair.next.future.get().task_id == 0 &&
                !ready(queued_future),
            "unacknowledged/foreign/malformed/timeout pair cannot fund successor authorization");
        const auto old = pair.old.ticket.take_result();
        const auto next = pair.next.ticket.take_result();
        const auto expected =
            index == 0 ? RemoteControlStep::kFinished : RemoteControlStep::kUncertain;
        require(old && next && old->step == expected && next->step == expected &&
                    old->was_published && next->was_published,
                "both failed originals retain actual combined publication and exact failure class");
        require(checked(lease).progress(now) == ControlMailboxProgress::kCompleted &&
                    ready(queued_future),
                "queued never-published request settles without transport");
        const auto rejected = queued_future.get();
        const auto outcome = queued_ticket.take_result();
        require(rejected.status == NativeEpochClaimStatus::kUncertain &&
                    rejected.grant.execution.task_id == 19 && outcome && !outcome->was_published &&
                    outcome->step == RemoteControlStep::kRejected,
                "known queued native identity survives pair quarantine without replay");
        if (index == 0) peer.no_request();
        require(checked(lease).progress(now) == ControlMailboxProgress::kQuarantined,
                "zero unsettled originals cannot become successful drain");
        lease.reset();
        require(mailbox.detach_owner(owner) && !mailbox.register_owner(),
                "detaching cannot repair pair quarantine");
    }
}

void test_intervening_started_and_unencodable() {
    for (std::size_t mode = 0; mode < 3; ++mode) {
        ProcessControlMailbox mailbox{{.should_combine_adjacent_requests = true}};
        Peer peer{mailbox};
        const auto owner = mailbox.register_owner();
        auto old = accept<bool>(mailbox, owner,
                                {TaskExecutionProtocolOperation::kExecutionComplete, {17, 29}});
        auto lease = mailbox.try_acquire(owner);
        require(lease.has_value(), "independent fixture owns lease");
        Accepted<bool> intervening;
        if (mode == 0)
            intervening = accept<bool>(
                mailbox, owner, {TaskExecutionProtocolOperation::kExecutionHeartbeat, {19, 31}});
        if (mode == 1) {
            require(checked(lease).progress() == ControlMailboxProgress::kPending,
                    "old report already issued");
            require(peer.receive().operation == TaskExecutionProtocolOperation::kExecutionComplete,
                    "already-issued report remains independent");
        }
        auto next =
            accept<CooperativeSubmission>(mailbox, owner,
                                          {TaskExecutionProtocolOperation::kExecutionAcquire,
                                           {},
                                           {.work_units = mode == 2 ? 0U : 1U}});
        if (mode != 1) {
            require(checked(lease).progress() == ControlMailboxProgress::kPending,
                    "old report starts independently");
            require(peer.receive().operation == TaskExecutionProtocolOperation::kExecutionComplete,
                    "fusion cannot cross intervening work or fail combined encoding");
        }
        peer.reply("GLIMMER_TASK_EXEC_V1 STATE 17 COMPLETED\n");
        peer.no_request();
        require(checked(lease).progress() == ControlMailboxProgress::kCompleted &&
                    ready(old.future) && old.future.get() && !ready(next.future),
                "actual old ACK and original future precede any independent successor");
        if (mode == 0) {
            require(
                checked(lease).progress() == ControlMailboxProgress::kPending &&
                    peer.receive().operation == TaskExecutionProtocolOperation::kExecutionHeartbeat,
                "intervening original operation remains in FIFO");
            peer.reply("GLIMMER_TASK_EXEC_V1 STATE 19 RUNNING\n");
            require(checked(lease).progress() == ControlMailboxProgress::kCompleted &&
                        intervening.future.get(),
                    "intervening operation needs its actual reply");
        }
        if (mode == 2) {
            require(checked(lease).progress() == ControlMailboxProgress::kCompleted &&
                        next.future.get().task_id == 0,
                    "unencodable successor is a definite negative, not a hidden split publication");
            peer.no_request();
            const auto outcome = next.ticket.take_result();
            require(outcome && !outcome->was_published,
                    "locally rejected successor never publishes domain work");
        } else {
            require(
                checked(lease).progress() == ControlMailboxProgress::kPending &&
                    peer.receive().operation == TaskExecutionProtocolOperation::kExecutionAcquire,
                "independent acquisition follows old original future and intervening work");
            peer.reply("GLIMMER_TASK_EXEC_V1 EXEC_LEASE 18 default 1 1 1 30\n");
            require(checked(lease).progress() == ControlMailboxProgress::kCompleted &&
                        next.future.get().claim.execution.generation == 30,
                    "independent result remains fresh");
        }
        lease.reset();
        require(mailbox.detach_owner(owner), "all independent original outcomes settle");
    }
}

void test_concurrent_producers() {
    constexpr std::size_t producers = 6;
    ProcessControlMailbox mailbox{{.max_outstanding_requests = producers * 2,
                                   .max_owners = 1,
                                   .should_combine_adjacent_requests = true}};
    Peer peer{mailbox};
    const auto owner = mailbox.register_owner();
    std::array<Accepted<bool>, producers> reports;
    std::array<Accepted<CooperativeSubmission>, producers> submissions;
    std::array<std::thread, producers> threads;
    for (std::size_t index = 0; index < producers; ++index) {
        threads[index] = std::thread([&, index] {
            reports[index] = accept<bool>(
                mailbox, owner,
                {TaskExecutionProtocolOperation::kExecutionComplete, {17 + index, 29 + index}});
            submissions[index] = accept<CooperativeSubmission>(
                mailbox, owner,
                {TaskExecutionProtocolOperation::kExecutionAcquire,
                 {},
                 {.work_units = static_cast<std::uint32_t>(index + 1)}});
        });
    }
    for (auto& thread : threads) thread.join();
    require(mailbox.snapshot().unsettled_requests == producers * 2,
            "all accepted originals retain the declared bound");
    std::array<bool, producers> seen_old{};
    std::array<bool, producers> seen_new{};
    auto lease = mailbox.try_acquire(owner);
    require(lease.has_value(), "concurrent accepted originals share one executor");
    for (std::size_t wires = 0; wires < producers * 2 && mailbox.snapshot().unsettled_requests != 0;
         ++wires) {
        require(lease->progress() == ControlMailboxProgress::kPending,
                "one accepted FIFO wire publishes");
        const auto request = peer.receive();
        TaskExecutionProtocolResponse response{};
        const bool has_old =
            request.operation == TaskExecutionProtocolOperation::kExecutionComplete ||
            request.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire;
        const bool has_new =
            request.operation == TaskExecutionProtocolOperation::kExecutionAcquire ||
            request.operation == TaskExecutionProtocolOperation::kExecutionCompleteAcquire;
        require(has_old || has_new, "only actual accepted operation types may publish");
        if (has_old) {
            require(request.task_id >= 17 && request.task_id < 17 + producers &&
                        request.generation == request.task_id + 12 &&
                        !seen_old[request.task_id - 17],
                    "concurrent pairing cannot substitute or repeat an old identity");
            seen_old[request.task_id - 17] = true;
            response.kind = TaskExecutionProtocolResponseKind::kState;
            response.task_id = request.task_id;
            response.state = TaskProtocolState::kCompleted;
            if (has_new) response.completed_execution = {request.task_id, request.generation};
        }
        if (has_new) {
            const auto units = request.admission.work_units;
            require(units >= 1 && units <= producers && !seen_new[units - 1],
                    "concurrent preparation preserves each exact original successor admission");
            seen_new[units - 1] = true;
            response.kind = has_old ? TaskExecutionProtocolResponseKind::kExecutionHandoff
                                    : TaskExecutionProtocolResponseKind::kExecutionLease;
            response.task_id = 200 + units;
            response.generation = 500 + units;
            response.tenant_id = "default";
            response.memory_bytes = 1;
            response.weight = 1;
            response.work_units = units;
        }
        const auto encoded = format_task_execution_protocol_response(response);
        require(encoded.has_value(), "actual concurrent reply must encode");
        peer.reply(*encoded);
        require(lease->progress() == ControlMailboxProgress::kCompleted,
                "actual reply settles one original or exact adjacent pair");
    }
    require(mailbox.snapshot().unsettled_requests == 0,
            "all concurrently accepted originals settle");
    for (std::size_t index = 0; index < producers; ++index) {
        require(seen_old[index] && seen_new[index] && reports[index].future.get(),
                "every original report has exactly one actual ACK");
        const auto result = submissions[index].future.get();
        require(result.task_id == 201 + index && result.claim.execution.generation == 501 + index,
                "every original submission consumes its own fresh actual grant");
    }
    lease.reset();
    require(mailbox.detach_owner(owner), "concurrent producer CPU ownership drains");
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
    test_withheld_pair_migration_close_and_bounds();
    test_acknowledged_old_with_new_denial();
    test_pair_uncertainty_and_queued_rejection();
    test_intervening_started_and_unencodable();
    test_concurrent_producers();
    require(descriptors() == before,
            "all paired/independent/quarantined/fork CPU descriptors retire");
}
