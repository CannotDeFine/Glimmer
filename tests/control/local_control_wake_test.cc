#include "internal/local_control_wake.h"
#include "internal/process_control_mailbox.h"
#include "glimmer/control/unix_socket_server.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/unix_socket_task_execution_server.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <future>
#include <iostream>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
template <typename T>
decltype(auto) checked(T&& value) noexcept {
    if (!value.has_value()) std::abort();
    return *value;
}

using namespace glimmer::control;
using namespace glimmer::control::internal;
using namespace std::chrono_literals;
using Clock = OwnedControlExchange::Clock;
void require(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_local_activity_and_backpressure() {
    LocalControlWake wake;
    require(wake.notify() && wake.wait_until({Clock::now() + 1s, {}}).has_activity,
            "publication before wait remains observable");
    std::vector<std::thread> producers;
    for (std::size_t index = 0; index < 4; ++index)
        producers.emplace_back([&] {
            for (std::size_t count = 0; count < 1000; ++count)
                require(wake.notify(), "concurrent full wake buffer coalesces readable hints");
        });
    for (auto& producer : producers) producer.join();
    std::size_t drains = 0;
    for (;;) {
        const auto outcome = wake.wait_until({Clock::now(), {}});
        if (outcome.status == ControlWaitStatus::kDue) break;
        require(
            outcome.status == ControlWaitStatus::kReady && outcome.has_activity && ++drains <= 64,
            "local hint drain is bounded and contains no authority");
    }
    require(drains > 0, "real concurrent hints must have been readable");
    std::thread notifier([&] {
        std::this_thread::sleep_for(5ms);
        require(wake.notify(), "in-flight wait notify succeeds");
    });
    const auto ready = wake.wait_until({Clock::now() + 1s, {}});
    notifier.join();
    require(ready.status == ControlWaitStatus::kReady && ready.has_activity,
            "publication racing a sleeping wait cannot be lost");
    const auto deadline = Clock::now() + 200us;
    require(wake.wait_until({deadline, {}}).status == ControlWaitStatus::kDue &&
                Clock::now() >= deadline,
            "submillisecond absolute timer never returns before due");
    require(wake.wait_until({Clock::time_point::max(), {}}).status == ControlWaitStatus::kRejected,
            "unbounded wait plan rejects without consuming a result");
    const pid_t child = ::fork();
    require(child >= 0, "owned fork creates");
    if (child == 0) {
        const bool rejected =
            !wake.notify() &&
            wake.wait_until({Clock::now() + 1s, {}}).status == ControlWaitStatus::kRejected &&
            wake.close();
        ::_exit(rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0 && wake.notify() &&
                wake.wait_until({Clock::now() + 1s, {}}).has_activity,
            "fork CPU close cannot shutdown or consume parent wake");
    require(wake.close() && wake.close() && !wake.notify() &&
                wake.wait_until({Clock::now(), {}}).status == ControlWaitStatus::kError,
            "partial/repeated cleanup is safe and closed wake cannot recover");
}

void test_reply_closure_and_earliest_deadline() {
    LocalControlWake wake;
    std::array<int, 2> reply{-1, -1};
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, reply.data()) == 0,
            "owned reply pair creates");
    constexpr char hint = 'R';
    require(::send(reply[1], &hint, 1, MSG_DONTWAIT | MSG_NOSIGNAL) == 1,
            "real ready reply hint publishes");
    auto outcome =
        wake.wait_until({Clock::now() + 1s, OwnedExchangeWait{reply[0], Clock::now() + 1s}});
    require(
        outcome.status == ControlWaitStatus::kReady && outcome.has_reply && !outcome.has_activity,
        "reply ready before wait is observed independently of actor notification");
    char byte = 0;
    require(::recv(reply[0], &byte, 1, MSG_DONTWAIT) == 1 && byte == hint,
            "wait does not consume authenticated reply bytes");
    const auto deadline = Clock::now() + 200us;
    require(wake.wait_until({Clock::now() + 1s, OwnedExchangeWait{reply[0], deadline}}).status ==
                    ControlWaitStatus::kDue &&
                Clock::now() >= deadline,
            "transport deadline wins without extension or millisecond rounding");
    require(::close(reply[1]) == 0, "owned peer closes");
    outcome = wake.wait_until({Clock::now() + 1s, OwnedExchangeWait{reply[0], Clock::now() + 1s}});
    require(outcome.has_reply && outcome.has_peer_closure,
            "real peer closure wakes but never acknowledges physical stop");
    require(::close(reply[0]) == 0, "owned response descriptor closes");
}

class Directory final {
   public:
    Directory() {
        std::array<char, 48> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-control-wait-XXXXXX";
        std::memcpy(pattern.data(), prefix.data(), prefix.size());
        require(::mkdtemp(pattern.data()) != nullptr, "owned service directory creates");
        path = pattern.data();
    }
    ~Directory() {
        require(::rmdir(path.c_str()) == 0, "owned empty service directory retires");
    }
    Directory(const Directory&) = delete;
    Directory& operator=(const Directory&) = delete;
    std::string path;
};
struct Barrier {
    std::atomic<bool> is_entered{false};
    std::promise<void> release;
    std::shared_future<void> ready{release.get_future().share()};
};
bool registered(void* context, glimmer::core::TaskId) noexcept {
    auto& barrier = *static_cast<Barrier*>(context);
    barrier.is_entered.store(true, std::memory_order_release);
    try {
        barrier.ready.wait();
        return true;
    } catch (...) {
        return false;
    }
}
void rollback(void*, glimmer::core::TaskId) noexcept {}

void test_authenticated_mailbox_wait() {
    Directory directory;
    Barrier barrier;
    auto admission = std::make_unique<TaskExecutionAdmissionService>(
        1024,
        glimmer::core::SchedulerOptions{.scheduling_policy =
                                            glimmer::core::SchedulingPolicy::kPriority},
        5000ms, true);
    TaskExecutionEndpoint endpoint{std::move(admission), registered, rollback, &barrier};
    UnixSocketTaskExecutionServer server{endpoint,
                                         {.socket_path = directory.path + "/control.sock",
                                          .allowed_uid = std::nullopt,
                                          .io_timeout_ms = 1000}};
    require(server.start(), "actual authenticated service starts");
    std::jthread acceptor([&](const std::stop_token& stop) {
        while (!stop.stop_requested())
            require(
                server.serve_one_for(10) != UnixSocketServeStatus::kError || stop.stop_requested(),
                "owned acceptor only stops on explicit teardown");
    });
    {
        ProcessControlMailbox mailbox;
        require(mailbox.initialize({directory.path + "/control.sock", 1000ms}) ==
                    OwnedExchangeColdStatus::kReady,
                "real service binds V3 before domain acceptance");
        const auto owner = mailbox.register_owner();
        auto lease = mailbox.try_acquire(owner);
        auto ticket = mailbox.submit(
            owner,
            RemoteControlOperation::prepare(
                RemoteCooperativeCommand{TaskExecutionProtocolOperation::kExecutionAcquire}, {}));
        require(lease && ticket && lease->wait_until(Clock::now() + 1s).has_activity &&
                    !ticket.is_ready() && lease->progress() == ControlMailboxProgress::kPending,
                "prepared FIFO readiness is not a result or synchronous dispatch");
        const auto observed = Clock::now() + 1s;
        while (!barrier.is_entered.load(std::memory_order_acquire) && Clock::now() < observed)
            std::this_thread::yield();
        require(barrier.is_entered.load(std::memory_order_acquire),
                "actual request reaches blocked registrar");
        const auto fixed = checked(checked(lease).wait_state()).deadline;
        auto wait = checked(lease).wait_until(Clock::now() + 500us);
        if (wait.has_activity) wait = checked(lease).wait_until(Clock::now() + 500us);
        require(wait.status == ControlWaitStatus::kDue && !ticket.is_ready() &&
                    checked(lease).progress() == ControlMailboxProgress::kPending &&
                    checked(checked(lease).wait_state()).deadline == fixed,
                "due physical/control timer cannot fabricate grant or renew transport deadline");
        require(mailbox.notify_activity() &&
                    checked(lease).wait_until(Clock::now() + 1s).has_activity && !ticket.is_ready(),
                "spurious actor wake cannot complete withheld actual reply");
        barrier.release.set_value();
        const auto limit = Clock::now() + 1s;
        bool has_reply = false;
        while (!has_reply && Clock::now() < limit)
            has_reply = checked(lease).wait_until(limit).has_reply;
        require(has_reply && !ticket.is_ready() &&
                    checked(lease).progress() == ControlMailboxProgress::kCompleted &&
                    ticket.is_ready(),
                "actual authenticated reply wakes wait; only validated progress publishes result");
        const auto outcome = ticket.take_result();
        require(outcome && outcome->was_published && !ticket.take_result(),
                "actual result consumes once");
        const auto granted = std::get<CooperativeSubmission>(checked(outcome).result);
        require(granted.claim.status == CooperativeClaimStatus::kGranted,
                "real generation comes only from service");
        auto complete = mailbox.submit(
            owner, RemoteControlOperation::prepare(
                       RemoteCooperativeCommand{TaskExecutionProtocolOperation::kExecutionComplete,
                                                granted.claim.execution},
                       {}));
        mailbox.close();
        require(complete && checked(lease).progress() == ControlMailboxProgress::kPending,
                "closed mailbox keeps accepted final report");
        const auto finish_limit = Clock::now() + 1s;
        auto progress = ControlMailboxProgress::kPending;
        while (progress == ControlMailboxProgress::kPending && Clock::now() < finish_limit) {
            static_cast<void>(checked(lease).wait_until(finish_limit));
            progress = checked(lease).progress();
        }
        const auto status = endpoint.lease_status();
        require(progress == ControlMailboxProgress::kCompleted &&
                    std::get<bool>(checked(complete.take_result()).result) && status &&
                    status->running_tasks == 0,
                "real CPU-declared terminal ACK drains before teardown");
        lease.reset();
        require(mailbox.detach_owner(owner), "last actor exits only after settled actual outcomes");
    }
    require(endpoint.begin_shutdown(), "stop new work before teardown");
    acceptor.request_stop();
    acceptor.join();
    server.stop();
}

std::size_t descriptors() {
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "inventory opens");
    std::size_t count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    require(errno == 0 && ::closedir(directory) == 0, "inventory closes");
    return count;
}
}  // namespace
int main() {
    const auto before = descriptors();
    test_local_activity_and_backpressure();
    test_reply_closure_and_earliest_deadline();
    test_authenticated_mailbox_wait();
    require(descriptors() == before,
            "all normal/concurrent/fork/reply/closed CPU descriptors retire");
}
