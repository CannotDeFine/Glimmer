#include "glimmer/control/async_cooperative_launch_gate.h"
#include "glimmer/control/native_epoch_protocol.h"
#include "glimmer/control/unix_socket_server.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <poll.h>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
using namespace glimmer::control;
using namespace std::chrono_literals;
constexpr NativeEpochExtent kExtent{8, 4, 2, 1'000'000};
constexpr NativeEpochGrant kGrant{{10, 20}, kExtent, 1};
const std::string kAcquire = "GLIMMER_EPOCH_V2 ACQUIRE default 1 1 8 0 8 4 2 1000000\n";
const std::string kProbe = "GLIMMER_EPOCH_V2 HEARTBEAT 10 20\n";
const std::string kProbeReply = "GLIMMER_EPOCH_V2 ACK HEARTBEAT 10 20 0\n";

template <typename T>
decltype(auto) checked(T&& value) {
    if (!value.has_value()) std::abort();
    return *value;
}

void require(bool value, const char* message) {
    if (!value) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class SocketUnavailable final : public std::runtime_error {
   public:
    SocketUnavailable() : std::runtime_error("Unix sockets unavailable") {}
};

class TemporaryDirectory final {
   public:
    TemporaryDirectory() {
        std::array<char, 48> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-epoch-client-XXXXXX";
        std::copy(prefix.begin(), prefix.end(), pattern.begin());
        const auto* directory = ::mkdtemp(pattern.data());
        require(directory != nullptr, "isolated socket directory");
        path = directory;
    }
    ~TemporaryDirectory() {
        require(::rmdir(path.c_str()) == 0, "owned empty directory removed");
    }
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    TemporaryDirectory(TemporaryDirectory&&) = delete;
    TemporaryDirectory& operator=(TemporaryDirectory&&) = delete;
    std::string path;
};

struct Step {
    std::string request;
    std::optional<std::string> response;
};

class ScriptedPeer final {
   public:
    TemporaryDirectory directory;
    const std::string path{directory.path + "/control.sock"};
    ScriptedPeer(std::vector<Step> script, bool should_block_first = false)
        : script_(std::move(script)), is_blocking_(should_block_first) {
        listener_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
        if (listener_ < 0) {
            throw SocketUnavailable{};
        }
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::copy(path.begin(), path.end(), address.sun_path);
        if (::bind(listener_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0 ||
            ::listen(listener_, 4) != 0) {
            require(::close(listener_) == 0, "unbound listener closed");
            listener_ = -1;
            static_cast<void>(::unlink(path.c_str()));
            throw SocketUnavailable{};
        }
        thread_ = std::thread([this] { run(); });
    }
    ~ScriptedPeer() {
        release();
        join();
        require(::close(listener_) == 0, "listener closed");
        require(::unlink(path.c_str()) == 0, "owned socket removed");
    }
    ScriptedPeer(const ScriptedPeer&) = delete;
    ScriptedPeer& operator=(const ScriptedPeer&) = delete;
    ScriptedPeer(ScriptedPeer&&) = delete;
    ScriptedPeer& operator=(ScriptedPeer&&) = delete;
    void await_first() {
        std::unique_lock lock(mutex_);
        require(condition_.wait_for(lock, 2s, [&] { return has_entered_; }),
                "accepted request observed");
    }
    void release() {
        const std::lock_guard lock(mutex_);
        is_blocking_ = false;
        condition_.notify_all();
    }
    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

   private:
    static void readable(int descriptor) {
        pollfd state{descriptor, POLLIN, 0};
        int status;
        do {
            status = ::poll(&state, 1, 2000);
        } while (status < 0 && errno == EINTR);
        require(status > 0, "expected exact request arrives within deadline");
    }
    void run() {
        int client = -1;
        std::string first_grant;
        std::set<std::uint64_t> sequences;
        for (std::size_t index = 0; index < script_.size(); ++index) {
            if (client < 0) {
                readable(listener_);
                client = ::accept4(listener_, nullptr, nullptr, SOCK_CLOEXEC);
                require(client >= 0, "scripted connection accepted");
            }
            std::string request;
            for (;;) {
                readable(client);
                char byte{};
                require(::recv(client, &byte, 1, 0) == 1, "exact frame received");
                request += byte;
                require(request.size() <= kTaskProtocolMaxLineBytes, "bounded request frame");
                if (byte == '\n') {
                    break;
                }
            }
            std::uint64_t sequence = 0;
            if (request.starts_with(kNativeEpochProtocolVersion)) {
                const auto parsed = parse_native_epoch_request(request);
                require(parsed && sequences.insert(parsed->request_sequence).second,
                        "real unique nonzero native request correlation");
                sequence = checked(parsed).request_sequence;
                request = request.substr(0, request.find_last_of(' ')) + "\n";
            }
            require(request == script_[index].request,
                    "exact wire order: no replay/fallback/coalescing");
            if (index == 0) {
                std::unique_lock lock(mutex_);
                has_entered_ = true;
                condition_.notify_all();
                require(condition_.wait_for(lock, 2s, [&] { return !is_blocking_; }),
                        "blocked fixture released");
            }
            auto response = script_[index].response;
            if (response == "REPLAY_FIRST_GRANT") {
                require(!first_grant.empty(), "a real earlier grant is available to replay");
                response = first_grant;
            } else if (response) {
                const bool wrong_sequence = response->starts_with("WRONG_SEQUENCE ");
                if (wrong_sequence) {
                    *response = response->substr(std::string_view("WRONG_SEQUENCE ").size());
                }
                if (response->starts_with(kNativeEpochProtocolVersion)) {
                    response->pop_back();
                    *response +=
                        " " +
                        std::to_string(wrong_sequence ? std::numeric_limits<std::uint64_t>::max()
                                                      : sequence) +
                        "\n";
                    if (response->find(" GRANTED ") != std::string::npos && first_grant.empty()) {
                        first_grant = *response;
                    }
                }
            }
            if (!response) {
                require(::shutdown(client, SHUT_RDWR) == 0 && ::close(client) == 0,
                        "lose response without replay");
                client = -1;
            } else {
                std::size_t sent = 0;
                while (sent < checked(response).size()) {
                    const auto count = ::send(client, checked(response).data() + sent,
                                              checked(response).size() - sent, MSG_NOSIGNAL);
                    require(count > 0, "scripted reply delivered");
                    sent += static_cast<std::size_t>(count);
                }
            }
        }
        if (client >= 0) {
            require(::close(client) == 0, "completed scripted connection closed");
        }
    }
    std::vector<Step> script_;
    bool is_blocking_{};
    bool has_entered_{};
    std::mutex mutex_;
    std::condition_variable condition_;
    int listener_{-1};
    std::thread thread_;
};

void shared_policy(CooperativeLaunchGate& gate) {
    const auto low = gate.acquire_native_epoch({kExtent, 1});
    require(low.status == NativeEpochClaimStatus::kGranted && low.grant.extent == kExtent,
            "distinct native grant");
    require(!gate.complete(low.grant.execution) && !gate.suspend(low.grant.execution),
            "unit reports cannot consume epoch");
    const auto high = gate.submit_and_claim({.relative_priority = 9});
    require(high.claim.status == CooperativeClaimStatus::kPending &&
                gate.retreat_native_epoch(low.grant) == true,
            "ordinary priority shares native capacity and requests retreat");
    auto stale = low.grant;
    ++stale.execution.generation;
    require(!gate.finish_native_epoch(stale, 1, false) && !gate.retreat_native_epoch(stale) &&
                gate.heartbeat_native_epoch(low.grant),
            "stale generation cannot release epoch");
    require(!gate.finish_native_epoch(low.grant, 9, false) &&
                gate.finish_native_epoch(low.grant, 3, false) &&
                !gate.finish_native_epoch(low.grant, 3, false),
            "exact terminal count/cap and no replay");
    const auto unit = gate.claim(high.task_id);
    require(unit.status == CooperativeClaimStatus::kGranted,
            "original unit claims after physical stop declaration");
    auto fake = kGrant;
    fake.execution = unit.execution;
    require(!gate.finish_native_epoch(fake, 0, false), "native reports cannot consume unit");
    const auto pending = gate.acquire_native_epoch({kExtent});
    require(pending.status == NativeEpochClaimStatus::kPending &&
                pending.grant.execution.generation == 0 &&
                !gate.cancel_pending(pending.grant.execution.task_id),
            "pending approval is not dispatch or unit cancellation");
    require(gate.cancel_native_epoch(pending.grant) && !gate.cancel_native_epoch(pending.grant),
            "unused native cancellation once");
    const auto next = gate.acquire_native_epoch({kExtent, 10});
    require(next.status == NativeEpochClaimStatus::kPending && gate.complete(unit.execution),
            "second pending epoch shares unit owner");
    const auto granted = gate.claim_native_epoch(next.grant);
    require(granted.status == NativeEpochClaimStatus::kGranted &&
                granted.grant.execution.task_id == next.grant.execution.task_id &&
                granted.grant.extent == next.grant.extent &&
                gate.finish_native_epoch(granted.grant, 0, true),
            "pending exact extent and failure report");
}

void test_local() {
    CooperativeLaunchGate disabled;
    require(disabled.acquire_native_epoch({kExtent}).status == NativeEpochClaimStatus::kRejected &&
                checked(disabled.local_stats()).total_task_count == 0,
            "native policy defaults off before mutation");
    CooperativeLaunchGate gate(
        {.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority, .control_socket = {}},
        kExtent);
    shared_policy(gate);
    require(checked(gate.local_stats()).total_task_count == 0 &&
                checked(gate.local_stats()).quota.allocated_bytes == 0,
            "same local owner drains/forgets all unit/native tasks");
    AsyncCooperativeLaunchGate async({}, 8, kExtent);
    auto signal = async.subscribe_progress();
    const auto epoch = async.acquire_native_epoch({kExtent}).get();
    require(epoch.status == NativeEpochClaimStatus::kGranted && signal && signal->wait_for(1s),
            "typed async grant publishes progress");
    require(async.heartbeat_native_epoch(epoch.grant).get() &&
                async.retreat_native_epoch(epoch.grant).get() == false,
            "native async identity observation");
    const auto pending = async.acquire_native_epoch({kExtent}).get();
    require(pending.status == NativeEpochClaimStatus::kPending &&
                async.cancel_native_epoch(pending.grant).get(),
            "typed async pending cancellation");
    auto complete = async.finish_native_epoch(epoch.grant, 2, false);
    async.shutdown();
    require(complete.get() && !async.acquire_native_epoch({kExtent}).valid() &&
                !async.subscribe_progress(),
            "shutdown drains accepted terminal metadata only");
}

void test_bad_acquisition() {
    const std::vector<std::optional<std::string>> responses{
        std::nullopt,
        "GLIMMER_EPOCH_V1 GRANTED 10 20 8 4 2 1000000 1\n",
        "WRONG_SEQUENCE GLIMMER_EPOCH_V2 GRANTED 10 20 8 4 2 1000000 1\n",
        "malformed\n",
        "GLIMMER_TASK_V1 EXEC_LEASE 10 default 1 1 1 20\n",
        "GLIMMER_EPOCH_V2 GRANTED 10 20 9 4 2 1000000 1\n",
        "GLIMMER_EPOCH_V2 GRANTED 10 20 8 4 2 1000000 18446744073709551615\n",
        "GLIMMER_EPOCH_V2 PENDING 10 8 4 3 1000000\n",
        "GLIMMER_EPOCH_V2 ERROR INTERNAL_ERROR\n"};
    for (const auto& response : responses) {
        ScriptedPeer peer({{kAcquire, response}, {kProbe, kProbeReply}});
        CooperativeLaunchGate gate({.control_socket = peer.path}, kExtent);
        const auto result = gate.acquire_native_epoch({kExtent});
        require(result.status == NativeEpochClaimStatus::kUncertain &&
                    result.grant.execution.task_id == 0,
                "lost/wrong/legacy response does not invent a grant");
        require(gate.heartbeat_native_epoch(kGrant),
                "explicit next operation follows, no implicit retry/downgrade");
        peer.join();
    }
    ScriptedPeer rejected(
        {{kAcquire, "GLIMMER_EPOCH_V2 ERROR QUOTA_EXCEEDED\n"}, {kProbe, kProbeReply}});
    CooperativeLaunchGate gate({.control_socket = rejected.path}, kExtent);
    require(gate.acquire_native_epoch({kExtent}).status == NativeEpochClaimStatus::kRejected &&
                gate.heartbeat_native_epoch(kGrant),
            "explicit service rejection differs from uncertain mutation");
}

void test_bad_pending_and_reports() {
    auto pending = kGrant;
    pending.execution.generation = 0;
    pending.acceptance_deadline_nanoseconds = 0;
    const std::vector<std::optional<std::string>> claims{
        std::nullopt, "GLIMMER_EPOCH_V2 GRANTED 11 20 8 4 2 1000000 1\n",
        "GLIMMER_EPOCH_V2 GRANTED 10 20 8 3 2 1000000 1\n",
        "GLIMMER_EPOCH_V2 ERROR INVALID_REQUEST\n"};
    for (const auto& response : claims) {
        ScriptedPeer peer({{"GLIMMER_EPOCH_V2 CLAIM 10\n", response}, {kProbe, kProbeReply}});
        CooperativeLaunchGate gate({.control_socket = peer.path}, kExtent);
        const auto result = gate.claim_native_epoch(pending);
        require(result.status == NativeEpochClaimStatus::kUncertain &&
                    result.grant.execution.task_id == 10 && result.grant.extent == kExtent &&
                    result.grant.execution.generation == 0 && gate.heartbeat_native_epoch(kGrant),
                "uncertain claim preserves pending identity/extent without retry");
    }
    const std::vector<std::optional<std::string>> terminals{
        std::nullopt,
        "malformed\n",
        "GLIMMER_TASK_V1 STATE 10 COMPLETED\n",
        "WRONG_SEQUENCE GLIMMER_EPOCH_V2 ACK COMPLETE 10 20 2\n",
        "GLIMMER_EPOCH_V2 ACK COMPLETE 11 20 2\n",
        "GLIMMER_EPOCH_V2 ACK COMPLETE 10 21 2\n",
        "GLIMMER_EPOCH_V2 ACK FAIL 10 20 2\n",
        "GLIMMER_EPOCH_V2 ACK COMPLETE 10 20 3\n"};
    for (const auto& response : terminals) {
        ScriptedPeer peer(
            {{"GLIMMER_EPOCH_V2 COMPLETE 10 20 2\n", response}, {kProbe, kProbeReply}});
        CooperativeLaunchGate gate({.control_socket = peer.path}, kExtent);
        require(!gate.finish_native_epoch(kGrant, 2, false) && gate.heartbeat_native_epoch(kGrant),
                "wrong/lost terminal cannot acknowledge release or replay");
    }
    ScriptedPeer expired(
        {{kAcquire, "GLIMMER_EPOCH_V2 GRANTED 10 20 8 4 2 1000000 1\n"}, {kProbe, kProbeReply}});
    CooperativeLaunchGate gate({.control_socket = expired.path}, kExtent);
    const auto grant = gate.acquire_native_epoch({kExtent});
    require(grant.status == NativeEpochClaimStatus::kGranted &&
                grant.grant.acceptance_deadline_nanoseconds == 1 &&
                gate.heartbeat_native_epoch(grant.grant),
            "expired grant identity remains owned; heartbeat cannot reset deadline");
}

void test_bad_observations_and_disabled_policy() {
    for (const auto& reply :
         {"GLIMMER_EPOCH_V2 RETREAT 11 20 1\n", "GLIMMER_EPOCH_V2 RETREAT 10 21 1\n",
          "GLIMMER_EPOCH_V2 ACK HEARTBEAT 10 20 0\n"}) {
        ScriptedPeer peer({{"GLIMMER_EPOCH_V2 RETREAT 10 20\n", reply}, {kProbe, kProbeReply}});
        CooperativeLaunchGate gate({.control_socket = peer.path}, kExtent);
        require(!gate.retreat_native_epoch(kGrant) && gate.heartbeat_native_epoch(kGrant),
                "wrong retreat response stays unknown without replay");
    }
    auto pending = kGrant;
    pending.execution.generation = 0;
    pending.acceptance_deadline_nanoseconds = 0;
    ScriptedPeer cancelled(
        {{"GLIMMER_EPOCH_V2 CANCEL_PENDING 10\n", "GLIMMER_EPOCH_V2 ACK CANCEL_PENDING 11 0 0\n"},
         {kProbe, kProbeReply}});
    CooperativeLaunchGate cancel_gate({.control_socket = cancelled.path}, kExtent);
    require(!cancel_gate.cancel_native_epoch(pending) && cancel_gate.heartbeat_native_epoch(kGrant),
            "wrong pending cancellation cannot acknowledge release");
    ScriptedPeer disabled(std::vector<Step>{
        {"GLIMMER_TASK_V1 EXEC_HEARTBEAT 1 1\n", "GLIMMER_TASK_V1 STATE 1 RUNNING\n"}});
    CooperativeLaunchGate default_gate({.control_socket = disabled.path});
    require(
        default_gate.acquire_native_epoch({kExtent}).status == NativeEpochClaimStatus::kRejected &&
            default_gate.heartbeat({1, 1}),
        "default-off native request never reaches transport");
    ScriptedPeer excessive(std::vector<Step>{{kProbe, kProbeReply}});
    CooperativeLaunchGate bounded({.control_socket = excessive.path}, kExtent);
    auto extent = kExtent;
    ++extent.max_pending_prefixes;
    require(bounded.acquire_native_epoch({extent}).status == NativeEpochClaimStatus::kRejected &&
                bounded.heartbeat_native_epoch(kGrant),
            "client rejects over-policy extent before transport");
}

void test_replayed_grant() {
    const auto deadline = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              (std::chrono::steady_clock::now() + 1s).time_since_epoch())
                              .count();
    const auto fresh =
        "GLIMMER_EPOCH_V2 GRANTED 30 40 8 4 2 1000000 " + std::to_string(deadline) + "\n";
    ScriptedPeer peer(
        {{kAcquire, fresh},
         {"GLIMMER_EPOCH_V2 COMPLETE 30 40 2\n", "GLIMMER_EPOCH_V2 ACK COMPLETE 30 40 2\n"},
         {kAcquire, "REPLAY_FIRST_GRANT"},
         {kProbe, kProbeReply}});
    CooperativeLaunchGate gate({.control_socket = peer.path}, kExtent);
    const auto first = gate.acquire_native_epoch({kExtent});
    require(first.status == NativeEpochClaimStatus::kGranted &&
                gate.finish_native_epoch(first.grant, 2, false),
            "first actual grant and terminal response accepted");
    const auto replay = gate.acquire_native_epoch({kExtent});
    require(
        replay.status == NativeEpochClaimStatus::kUncertain && replay.grant.execution.task_id == 0,
        "replayed prior grant cannot fund a new epoch");
    require(gate.heartbeat_native_epoch(kGrant),
            "explicit next operation, no replay recovery request");
}

void test_async_capacity_and_order() {
    ScriptedPeer peer(
        {{kAcquire, "GLIMMER_EPOCH_V2 GRANTED 10 20 8 4 2 1000000 1\n"},
         {"GLIMMER_EPOCH_V2 COMPLETE 10 20 2\n", "GLIMMER_EPOCH_V2 ACK COMPLETE 10 20 2\n"}},
        true);
    AsyncCooperativeLaunchGate gate({.control_socket = peer.path}, 1, kExtent);
    auto first = gate.acquire_native_epoch({kExtent});
    require(first.valid(), "native request accepted without submitter transport");
    peer.await_first();
    require(first.wait_for(0ms) == std::future_status::timeout,
            "accepted remote operation is still blocked independently");
    auto second = gate.finish_native_epoch(kGrant, 2, false);
    require(second.valid() && !gate.heartbeat_native_epoch(kGrant).valid(),
            "bounded shared mailbox rejects before execution");
    std::promise<void> entering;
    std::thread closer([&] {
        entering.set_value();
        gate.shutdown();
    });
    entering.get_future().get();
    require(first.wait_for(0ms) == std::future_status::timeout,
            "shutdown waits accepted operation, not submitter");
    peer.release();
    closer.join();
    require(first.get().status == NativeEpochClaimStatus::kGranted && second.get() &&
                !gate.acquire_native_epoch({kExtent}).valid(),
            "shutdown drains both accepted native results");
    peer.join();

    ScriptedPeer interleaved(
        {{"GLIMMER_TASK_V1 EXEC_HEARTBEAT 1 1\n", "GLIMMER_TASK_V1 STATE 1 RUNNING\n"},
         {"GLIMMER_TASK_V1 EXEC_COMPLETE 1 1\n", "GLIMMER_TASK_V1 STATE 1 COMPLETED\n"},
         {"GLIMMER_EPOCH_V2 HEARTBEAT 2 2\n", "GLIMMER_EPOCH_V2 ACK HEARTBEAT 2 2 0\n"},
         {"GLIMMER_TASK_V1 EXEC_ACQUIRE default 1 1 1\n",
          "GLIMMER_TASK_V1 EXEC_LEASE 3 default 1 1 1 3\n"}},
        true);
    AsyncCooperativeLaunchGate ordered({.control_socket = interleaved.path}, 8, kExtent);
    auto blocked = ordered.heartbeat({1, 1});
    interleaved.await_first();
    auto completion = ordered.complete({1, 1});
    auto native = kGrant;
    native.execution = {2, 2};
    auto middle = ordered.heartbeat_native_epoch(native);
    auto acquisition = ordered.submit_and_claim();
    interleaved.release();
    ordered.shutdown();
    require(blocked.get() && completion.get() && middle.get() &&
                acquisition.get().claim.status == CooperativeClaimStatus::kGranted,
            "ordinary pair cannot coalesce across native operation");
    interleaved.join();
}

bool registered(void*, glimmer::core::TaskId) noexcept {
    return true;
}

void test_real_remote() {
    TemporaryDirectory directory;
    glimmer::core::Scheduler scheduler(
        100, {.scheduling_policy = glimmer::core::SchedulingPolicy::kPriority});
    TaskAdmissionService admission(scheduler, {}, true, nullptr, nullptr, nullptr, nullptr,
                                   kExtent);
    TaskControlEndpoint endpoint(admission, registered, nullptr);
    const auto path = directory.path + "/control.sock";
    UnixSocketControlServer server(endpoint, {.socket_path = path, .allowed_uid = std::nullopt});
    if (!server.start()) {
        throw SocketUnavailable{};
    }
    std::jthread acceptor([&](const std::stop_token& stop) {
        while (!stop.stop_requested()) {
            require(server.serve_one_for(10) != UnixSocketServeStatus::kError,
                    "authenticated server accepts");
        }
    });
    {
        CooperativeLaunchGate gate({.control_socket = path}, kExtent);
        shared_policy(gate);
        require(!gate.local_stats(), "remote path never switches local");
    }
    acceptor.request_stop();
    acceptor.join();
    if (const auto* transport = std::getenv("GLIMMER_CONTROL_TRANSPORT");
        transport != nullptr && std::string_view(transport) == "shared_memory") {
        require(server.shared_memory_request_count() > 0 &&
                    server.shared_memory_request_count() == server.served_request_count(),
                "real endpoint must use strict shared memory for all V2 and unit operations");
    }
    server.stop();
    require(scheduler.stats().running_task_count == 0 && scheduler.stats().queued_task_count == 0 &&
                scheduler.stats().quota.allocated_bytes == 0,
            "actual shared remote policy drains");
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc == 1) {
        test_local();
    } else {
        // Raw scripted peers intentionally assert socket bytes without an
        // upgrade handshake. This mode repeats only the real bound endpoint.
        if (argc == 2 && std::string_view(argv[1]) == "--shared-socket") {
            const char* transport = std::getenv("GLIMMER_CONTROL_TRANSPORT");
            require(transport != nullptr && std::string_view(transport) == "shared_memory",
                    "shared fixture must require strict negotiation");
            test_real_remote();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--replay") {
            test_replayed_grant();
            return 0;
        }
        require(argc == 2 && std::string_view(argv[1]) == "--socket", "known test mode");
        try {
            test_bad_acquisition();
            test_bad_pending_and_reports();
            test_bad_observations_and_disabled_policy();
            test_replayed_grant();
            test_async_capacity_and_order();
            test_real_remote();
        } catch (const SocketUnavailable&) {
            return 77;
        }
    }
    std::cout << "Native epoch gate tests passed\n";
    return 0;
} catch (const SocketUnavailable&) {
    return 77;
} catch (const std::exception& error) {
    std::cerr << "Native epoch gate test error: " << error.what() << '\n';
    return EXIT_FAILURE;
} catch (...) {
    std::cerr << "Native epoch gate test unknown error\n";
    return EXIT_FAILURE;
}
