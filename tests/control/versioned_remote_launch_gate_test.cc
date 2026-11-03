#include "internal/versioned_remote_launch_gate.h"
#include "internal/shared_control_channel.h"
#include "internal/socket_line_reader.h"
#include "internal/versioned_control_test_hooks.h"
#include "glimmer/control/task_execution_endpoint.h"
#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_task_execution_server.h"
#include "glimmer/control/unix_socket_server.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <poll.h>
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

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::abort();
    }
}

bool registered(void*, glimmer::core::TaskId) noexcept {
    return true;
}
void rollback(void*, glimmer::core::TaskId) noexcept {}

class Service final {
   public:
    Service()
        : endpoint(
              std::make_unique<TaskExecutionAdmissionService>(
                  100,
                  glimmer::core::SchedulerOptions{.max_running_tasks = 1, .max_queued_tasks = 1},
                  0ms, true),
              registered, rollback, nullptr),
          directory(make_directory()),
          path(directory + "/control.sock"),
          server(endpoint, {.socket_path = path}) {
        expect(server.start(), "real execution endpoint binds in private directory");
        thread = std::jthread([this](const std::stop_token& stop) {
            while (!stop.stop_requested())
                expect(server.serve_one_for(10) != UnixSocketServeStatus::kError,
                       "actual execution endpoint stays healthy");
        });
    }
    ~Service() {
        thread.request_stop();
        thread.join();
        server.stop();
        expect(::rmdir(directory.c_str()) == 0, "remove owned empty directory");
    }
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;
    TaskExecutionEndpoint endpoint;
    const std::string directory;
    const std::string path;
    UnixSocketTaskExecutionServer server;
    std::jthread thread;

   private:
    static std::string make_directory() {
        std::array<char, 50> path{};
        constexpr char prefix[] = "/tmp/glimmer-versioned-gate-XXXXXX";
        std::memcpy(path.data(), prefix, sizeof(prefix));
        expect(::mkdtemp(path.data()) != nullptr, "create private directory");
        return path.data();
    }
};

void actual_shared_service() {
    Service service;
    VersionedRemoteLaunchGate gate({.control_socket = service.path},
                                   VersionedControlTransport::kSharedMemory);
    LaunchGateTiming timing;
    const auto first = gate.acquire(&timing);
    expect(first.has_value() && timing.remote && timing.request_count == 1 &&
               timing.transport_nanoseconds > 0,
           "actual shared service grants one timed ordinary task");
    auto next = std::async(std::launch::async, [&] { return gate.acquire(); });
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    for (;;) {
        const auto status = service.endpoint.lease_status();
        if (status && status->queued_tasks == 1) break;
        expect(std::chrono::steady_clock::now() < deadline, "second task is queued");
        std::this_thread::sleep_for(1ms);
    }
    expect(!gate.acquire() && !gate.quarantined() && !gate.retained(),
           "an explicit full queue rejects without quarantining the session");
    expect(gate.heartbeat(checked(first)) && gate.complete(checked(first)),
           "queued polling cannot starve heartbeat or terminal report");
    expect(next.wait_for(2s) == std::future_status::ready, "second task receives admission");
    const auto second = next.get();
    expect(second && checked(second) != checked(first) && gate.heartbeat(checked(second)),
           "second task is distinct");
    const pid_t child = ::fork();
    expect(child >= 0, "fork actual session");
    if (child == 0) {
        const bool rejected = !gate.heartbeat(checked(second)) && gate.quarantined();
        ::_exit(rejected ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork child rejects inherited state without blocking parent");
    expect(gate.fail(checked(second)) && !gate.complete(checked(second)) && !gate.quarantined(),
           "exact failed terminal retires once; parent survives fork");
    const auto ledger = service.endpoint.lease_status();
    expect(ledger && ledger->running_tasks == 0 && ledger->queued_tasks == 0,
           "real service has no remaining ordinary work");
}

void actual_legacy_fallback() {
    std::array<char, 50> pattern{};
    constexpr char prefix[] = "/tmp/glimmer-gate-legacy-XXXXXX";
    std::memcpy(pattern.data(), prefix, sizeof(prefix));
    expect(::mkdtemp(pattern.data()) != nullptr, "create private legacy directory");
    const std::string directory = pattern.data();
    const std::string path = directory + "/control.sock";
    glimmer::core::Scheduler scheduler(100);
    TaskAdmissionService admission(scheduler, 0ms, true);
    TaskControlEndpoint endpoint(admission, registered, nullptr);
    UnixSocketControlServer server(endpoint, {.socket_path = path, .allowed_uid = std::nullopt});
    expect(server.start(), "start actual legacy socket server");
    std::jthread acceptor([&](const std::stop_token& stop) {
        while (!stop.stop_requested())
            expect(server.serve_one_for(10) != UnixSocketServeStatus::kError,
                   "legacy accept loop remains healthy");
    });
    {
        VersionedRemoteLaunchGate gate({.control_socket = path}, VersionedControlTransport::kAuto);
        const auto task = gate.acquire();
        expect(task && gate.heartbeat(*task) && gate.complete(*task) && !gate.quarantined(),
               "auto negotiates explicit unsupported, then uses one legacy socket session");
        expect(server.served_request_count() == 4,
               "exactly one negotiation and three domain requests were served");
    }
    acceptor.request_stop();
    acceptor.join();
    server.stop();
    expect(::rmdir(directory.c_str()) == 0, "remove only owned legacy directory");
}

void retained_service_lifecycle() {
    auto service = std::make_unique<Service>();
    VersionedRemoteLaunchGate gate({.control_socket = service->path},
                                   VersionedControlTransport::kSharedMemory);
    const auto running = gate.acquire();
    expect(running == 1, "first service owns actual initial task");
    auto pending = std::async(std::launch::async, [&] { return gate.acquire(); });
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    for (;;) {
        const auto status = service->endpoint.lease_status();
        if (status && status->queued_tasks == 1) break;
        expect(std::chrono::steady_clock::now() < deadline, "actual second task becomes pending");
        std::this_thread::sleep_for(1ms);
    }
    testing::drop_next_task_response.store(true);
    expect(pending.wait_for(2s) == std::future_status::ready && !pending.get(),
           "lost actual claim response does not grant physical work");
    const auto snapshot = gate.retained();
    const auto ledger = service->endpoint.lease_status();
    expect(snapshot && snapshot->size() == 2 && gate.quarantined() && ledger &&
               ledger->running_tasks == 1 && ledger->queued_tasks == 1,
           "quarantine retains both original service ledger entries");
    expect(service->endpoint.begin_shutdown(), "stop new service admission");
    const auto draining = service->endpoint.lease_status();
    expect(draining && draining->running_tasks == 1,
           "shutdown does not invent physical completion for a running task");
    // This fixture's registrar allocated no external resource and no CUDA
    // work was ever issued. The entire service is stopped and destroyed,
    // rather than interpreting a client disconnect as a successful cancel.
    service.reset();
    auto replacement = std::make_unique<Service>();
    VersionedRemoteLaunchGate fresh({.control_socket = replacement->path},
                                    VersionedControlTransport::kSharedMemory);
    const auto reused = fresh.acquire();
    expect(reused == running && !gate.fail(checked(running)) && fresh.heartbeat(checked(reused)) &&
               fresh.complete(checked(reused)),
           "an old retained identity cannot affect a new service that reuses its task ID");
}

void unsupported_acquire_falls_back_once() {
    std::array<char, 50> pattern{};
    constexpr char prefix[] = "/tmp/glimmer-gate-fallback-XXXXXX";
    std::memcpy(pattern.data(), prefix, sizeof(prefix));
    expect(::mkdtemp(pattern.data()) != nullptr, "create fallback directory");
    const std::string directory = pattern.data();
    const std::string path = directory + "/control.sock";
    const int listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(listener >= 0, "create fallback listener");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "fallback path fits");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    expect(::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0 &&
               ::listen(listener, 4) == 0,
           "bind fallback listener");
    std::thread peer([&] {
        const int socket = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(socket >= 0, "accept one fallback connection");
        SocketLineReader reader(socket);
        std::string line;
        expect(
            reader.read_line(line, 1000) == SocketLineStatus::kLine && line == kSharedControlOpen,
            "fallback begins with shared negotiation");
        expect(::send(socket, kSharedControlReady.data(), kSharedControlReady.size(),
                      MSG_NOSIGNAL) == static_cast<ssize_t>(kSharedControlReady.size()),
               "shared fallback ready sends");
        SharedControlChannel channel;
        expect(channel.create_server({17, 29}) && channel.send_offer(socket, 1s),
               "shared fallback offer sends");
        const auto step = [&](std::string_view prefix, std::string_view reply) {
            expect(channel.receive_request(socket, 1000, line) == SharedControlReadStatus::kLine &&
                       line.starts_with(prefix) && channel.publish_response(reply),
                   "fallback publishes each expected operation exactly once");
        };
        step("GLIMMER_TASK_V1 ACQUIRE ", "GLIMMER_TASK_V1 ERROR INVALID_REQUEST\n");
        step("GLIMMER_TASK_V1 SUBMIT ", "GLIMMER_TASK_V1 OK 17\n");
        step("GLIMMER_TASK_V1 CLAIM 17\n", "GLIMMER_TASK_V1 LEASE 17 default 1 1 1\n");
        step("GLIMMER_TASK_V1 COMPLETE 17\n", "GLIMMER_TASK_V1 STATE 17 COMPLETED\n");
        expect(::close(socket) == 0, "close exact completed session");
    });
    {
        VersionedRemoteLaunchGate gate({.control_socket = path},
                                       VersionedControlTransport::kSharedMemory);
        LaunchGateTiming timing;
        const auto task = gate.acquire(&timing);
        expect(task == 17 && timing.request_count == 3 && timing.claim_poll_count == 1 &&
                   gate.complete(17) && !gate.quarantined(),
               "only explicit pre-publication unsupported ACQUIRE permits SUBMIT fallback");
    }
    peer.join();
    expect(::close(listener) == 0 && ::unlink(path.c_str()) == 0 && ::rmdir(directory.c_str()) == 0,
           "remove only owned fallback resources");
}

void fair_broker_order() {
    std::array<char, 50> pattern{};
    constexpr char prefix[] = "/tmp/glimmer-gate-fair-XXXXXX";
    std::memcpy(pattern.data(), prefix, sizeof(prefix));
    expect(::mkdtemp(pattern.data()) != nullptr, "create fair broker directory");
    const std::string directory = pattern.data();
    const std::string path = directory + "/control.sock";
    const int listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(listener >= 0, "create fair broker listener");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "fair broker path fits");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    expect(::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0 &&
               ::listen(listener, 4) == 0,
           "bind fair broker listener");
    std::promise<void> has_blocked;
    std::promise<void> release;
    std::promise<void> acquire_in_flight;
    std::promise<void> release_acquire;
    std::vector<std::string> order;
    std::thread peer([&] {
        const int socket = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(socket >= 0, "accept fair broker session");
        SocketLineReader reader(socket);
        std::string line;
        expect(
            reader.read_line(line, 1000) == SocketLineStatus::kLine && line == kSharedControlOpen,
            "fair broker negotiates shared session");
        expect(::send(socket, kSharedControlReady.data(), kSharedControlReady.size(),
                      MSG_NOSIGNAL) == static_cast<ssize_t>(kSharedControlReady.size()),
               "fair broker sends ready");
        SharedControlChannel channel;
        expect(channel.create_server({17, 29}) && channel.send_offer(socket, 1s),
               "fair broker sends sealed offer");
        auto read = [&] {
            expect(channel.receive_request(socket, 1000, line) == SharedControlReadStatus::kLine,
                   "fair broker receives one exact request");
            const auto parsed = parse_task_protocol_request(line);
            expect(parsed.parsed(), "fair broker receives canonical request");
            return checked(parsed.request);
        };
        auto send = [&](const TaskProtocolResponse& response) {
            const auto encoded = format_task_protocol_response(response);
            expect(encoded && channel.publish_response(*encoded), "fair broker responds once");
        };
        auto lease = [&](glimmer::core::TaskId id) {
            TaskProtocolResponse response{};
            response.kind = TaskProtocolResponseKind::kLease;
            response.task_id = id;
            response.tenant_id = "default";
            response.memory_bytes = response.weight = response.work_units = 1;
            send(response);
        };
        auto state = [&](glimmer::core::TaskId id, TaskProtocolState value) {
            TaskProtocolResponse response{};
            response.kind = TaskProtocolResponseKind::kState;
            response.task_id = id;
            response.state = value;
            send(response);
        };
        for (glimmer::core::TaskId id = 1; id <= 10; ++id) {
            expect(read().operation == TaskProtocolOperation::kAcquire, "seed one running task");
            lease(id);
        }
        const auto blocked = read();
        expect(blocked.operation == TaskProtocolOperation::kHeartbeat && blocked.task_id == 1,
               "first urgent request occupies the transport");
        has_blocked.set_value();
        release.get_future().wait();
        state(1, TaskProtocolState::kRunning);
        glimmer::core::TaskId next = 11;
        for (unsigned int count = 0; count < 11; ++count) {
            const auto request = read();
            order.push_back(line);
            if (request.operation == TaskProtocolOperation::kAcquire)
                lease(next++);
            else {
                expect(request.operation == TaskProtocolOperation::kHeartbeat,
                       "only queued heartbeats or acquisitions are sent");
                state(request.task_id, TaskProtocolState::kRunning);
            }
        }
        for (glimmer::core::TaskId id = 1; id <= 12; ++id) {
            const auto request = read();
            expect(request.operation == TaskProtocolOperation::kFail && request.task_id == id,
                   "each physical fixture task retires in order");
            state(id, TaskProtocolState::kFailed);
        }
        expect(read().operation == TaskProtocolOperation::kAcquire,
               "final acquisition is already published");
        acquire_in_flight.set_value();
        release_acquire.get_future().wait();
        lease(13);
        expect(::close(socket) == 0, "close fully retired fair broker session");
    });
    {
        VersionedRemoteLaunchGate gate({.control_socket = path},
                                       VersionedControlTransport::kSharedMemory);
        for (glimmer::core::TaskId id = 1; id <= 10; ++id)
            expect(gate.acquire() == id, "seed distinct real gate identities");
        auto blocked = std::async(std::launch::async, [&] { return gate.heartbeat(1); });
        expect(has_blocked.get_future().wait_for(2s) == std::future_status::ready,
               "first heartbeat reaches peer");
        const auto wait_lanes = [&](std::uint64_t urgent, std::uint64_t normal) {
            const auto deadline = std::chrono::steady_clock::now() + 500ms;
            for (;;) {
                const auto lanes = gate.test_lanes();
                expect(lanes.has_value(), "broker lane snapshot remains available");
                if (checked(lanes).urgent_next == urgent && checked(lanes).normal_next == normal)
                    return;
                expect(std::chrono::steady_clock::now() < deadline, "broker request queues");
                std::this_thread::yield();
            }
        };
        std::vector<std::future<bool>> heartbeats;
        for (glimmer::core::TaskId id = 2; id <= 10; ++id) {
            heartbeats.push_back(
                std::async(std::launch::async, [&, id] { return gate.heartbeat(id); }));
            wait_lanes(id, 10);
        }
        auto first = std::async(std::launch::async, [&] { return gate.acquire(); });
        wait_lanes(10, 11);
        auto second = std::async(std::launch::async, [&] { return gate.acquire(); });
        wait_lanes(10, 12);
        release.set_value();
        expect(blocked.get(), "blocked heartbeat completes");
        for (auto& heartbeat : heartbeats) expect(heartbeat.get(), "queued heartbeat completes");
        expect(first.get() == 11 && second.get() == 12, "normal lane retains FIFO order");
        for (glimmer::core::TaskId id = 1; id <= 12; ++id)
            expect(gate.fail(id), "retire every seeded task");
        auto final = std::async(std::launch::async, [&] { return gate.acquire(); });
        expect(acquire_in_flight.get_future().wait_for(2s) == std::future_status::ready,
               "acquisition is in flight before local quarantine");
        gate.quarantine();
        release_acquire.set_value();
        expect(!final.get() && gate.quarantined(),
               "a late lease never grants work after local quarantine");
        const auto retained = gate.retained();
        expect(retained && retained->size() == 1 &&
                   (*retained)[0].phase ==
                       VersionedRemoteLaunchGate::RetainedPhase::kUnknownIdentity &&
                   !gate.heartbeat(13) && !gate.fail(13),
               "in-flight ownership remains unknown without further publication");
    }
    peer.join();
    expect(order.size() == 11, "all queued requests were observed");
    for (unsigned int index = 0; index < 7; ++index)
        expect(order[index] == "GLIMMER_TASK_V1 HEARTBEAT " + std::to_string(index + 2) + "\n",
               "urgent lane retains FIFO order");
    expect(order[7].starts_with("GLIMMER_TASK_V1 ACQUIRE ") &&
               order[8] == "GLIMMER_TASK_V1 HEARTBEAT 9\n" &&
               order[9] == "GLIMMER_TASK_V1 HEARTBEAT 10\n" &&
               order[10].starts_with("GLIMMER_TASK_V1 ACQUIRE "),
           "normal requests make progress after eight consecutive urgent requests");
    expect(::close(listener) == 0 && ::unlink(path.c_str()) == 0 && ::rmdir(directory.c_str()) == 0,
           "remove only owned fair broker resources");
}

enum class Fault : std::uint8_t {
    kBadAcquire,
    kLostClaim,
    kLostCompletion,
    kQuarantinedCompletion
};

void uncertain_reply(Fault fault) {
    std::array<char, 50> pattern{};
    constexpr char prefix[] = "/tmp/glimmer-gate-fault-XXXXXX";
    std::memcpy(pattern.data(), prefix, sizeof(prefix));
    expect(::mkdtemp(pattern.data()) != nullptr, "create owned fault directory");
    const std::string directory = pattern.data();
    const std::string path = directory + "/control.sock";
    const int listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(listener >= 0, "create owned fault listener");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    expect(path.size() < sizeof(address.sun_path), "private fault path fits");
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    expect(::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0 &&
               ::listen(listener, 4) == 0,
           "bind owned fault listener");
    std::promise<void> release;
    std::promise<void> terminal_in_flight;
    std::promise<void> terminal_returned;
    std::thread peer([&] {
        const int socket = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(socket >= 0, "accept one fault session");
        SocketLineReader reader(socket);
        std::string line;
        expect(
            reader.read_line(line, 1000) == SocketLineStatus::kLine && line == kSharedControlOpen,
            "only shared negotiation precedes the first request");
        expect(::send(socket, kSharedControlReady.data(), kSharedControlReady.size(),
                      MSG_NOSIGNAL) == static_cast<ssize_t>(kSharedControlReady.size()),
               "send explicit shared ready");
        SharedControlChannel channel;
        expect(channel.create_server({17, 29}) && channel.send_offer(socket, 1s),
               "publish authenticated service offer");
        expect(channel.receive_request(socket, 1000, line) == SharedControlReadStatus::kLine &&
                   line.starts_with("GLIMMER_TASK_V1 ACQUIRE "),
               "first ordinary acquisition arrives once");
        if (fault == Fault::kBadAcquire) {
            expect(channel.publish_response("GLIMMER_TASK_V1 LEASE 17 foreign 1 1 1\n"),
                   "publish a foreign acquisition response");
            release.get_future().wait();
        } else {
            const auto first = fault == Fault::kLostClaim
                                   ? "GLIMMER_TASK_V1 OK 17\n"
                                   : "GLIMMER_TASK_V1 LEASE 17 default 1 1 1\n";
            expect(channel.publish_response(first), "publish one actual first result");
            expect(channel.receive_request(socket, 1000, line) == SharedControlReadStatus::kLine,
                   "expected next mutation arrives once");
            expect(line == (fault == Fault::kLostClaim ? "GLIMMER_TASK_V1 CLAIM 17\n"
                                                       : "GLIMMER_TASK_V1 COMPLETE 17\n"),
                   "claim or terminal identity remains exact");
            if (fault == Fault::kQuarantinedCompletion) {
                terminal_in_flight.set_value();
                release.get_future().wait();
                expect(channel.publish_response("GLIMMER_TASK_V1 STATE 17 COMPLETED\n"),
                       "a delayed terminal acknowledgement is delivered");
                terminal_returned.get_future().wait();
            }
        }
        expect(::close(socket) == 0, "close without replaying a lost result");
    });
    {
        VersionedRemoteLaunchGate gate({.control_socket = path},
                                       VersionedControlTransport::kSharedMemory);
        const auto task = gate.acquire();
        if (fault == Fault::kQuarantinedCompletion) {
            expect(task == 17, "terminal fixture first grants a task");
            auto terminal = std::async(std::launch::async, [&] { return gate.complete(17); });
            expect(terminal_in_flight.get_future().wait_for(2s) == std::future_status::ready,
                   "terminal is in flight before local quarantine");
            gate.quarantine();
            release.set_value();
            expect(!terminal.get(), "late terminal acknowledgement does not clear quarantine");
            terminal_returned.set_value();
        } else if (fault == Fault::kLostCompletion) {
            expect(task == 17 && !gate.complete(17), "lost terminal cannot release capacity");
        } else {
            expect(!task, "untrusted or lost acquisition cannot authorize work");
        }
        const auto retained = gate.retained();
        expect(gate.quarantined() && retained && retained->size() == 1 &&
                   (*retained)[0].task == (fault == Fault::kBadAcquire ? 0U : 17U) &&
                   (*retained)[0].phase ==
                       (fault == Fault::kBadAcquire
                            ? VersionedRemoteLaunchGate::RetainedPhase::kUnknownIdentity
                            : VersionedRemoteLaunchGate::RetainedPhase::kUncertain),
               "uncertainty retains exact known identity or an explicit unknown marker");
        expect(!gate.acquire() && !gate.heartbeat(17) && !gate.fail(17),
               "quarantine forbids all later domain publications");
        if (fault == Fault::kBadAcquire) release.set_value();
    }
    peer.join();
    pollfd ready{listener, POLLIN, 0};
    expect(::poll(&ready, 1, 0) == 0, "no implicit reconnect or replay");
    expect(::close(listener) == 0 && ::unlink(path.c_str()) == 0 && ::rmdir(directory.c_str()) == 0,
           "retire only owned fault resources");
}
}  // namespace

int main() {
    actual_shared_service();
    actual_legacy_fallback();
    retained_service_lifecycle();
    unsupported_acquire_falls_back_once();
    fair_broker_order();
    uncertain_reply(Fault::kBadAcquire);
    uncertain_reply(Fault::kLostClaim);
    uncertain_reply(Fault::kLostCompletion);
    uncertain_reply(Fault::kQuarantinedCompletion);
}
