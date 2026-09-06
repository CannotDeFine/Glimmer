#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_server.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

using glimmer::control::TaskAdmissionService;
using glimmer::control::TaskControlEndpoint;
using glimmer::control::TaskProtocolResponseKind;
using glimmer::control::TaskResourceRegistrar;
using glimmer::control::UnixSocketControlConfig;
using glimmer::control::UnixSocketControlServer;
using glimmer::core::Scheduler;
using glimmer::core::TaskId;

bool register_resource(void* context, TaskId) noexcept {
    return context != nullptr;
}

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int connect_client(const char* path) {
    const int client_fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(client_fd >= 0, "client socket should be created");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path, std::strlen(path) + 1);
    const auto address_length =
        static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + std::strlen(path) + 1);
    expect(::connect(client_fd, reinterpret_cast<const sockaddr*>(&address), address_length) == 0,
           "client should connect");
    return client_fd;
}

std::string exchange(UnixSocketControlServer& server, const std::string& path,
                     std::string_view request) {
    const int client_fd = connect_client(path.c_str());
    const std::size_t request_size = request.size();
    expect(::send(client_fd, request.data(), request_size, MSG_NOSIGNAL) ==
               static_cast<ssize_t>(request_size),
           "client should send request");
    expect(server.serve_one(), "server should serve one client");
    char response[256]{};
    const ssize_t received = ::recv(client_fd, response, sizeof(response) - 1, 0);
    expect(received > 0, "client should receive response");
    static_cast<void>(::close(client_fd));
    return std::string(response, static_cast<std::size_t>(received));
}

void send_request(int client_fd, std::string_view request) {
    const ssize_t sent = ::send(client_fd, request.data(), request.size(), MSG_NOSIGNAL);
    expect(sent == static_cast<ssize_t>(request.size()), "persistent client should send request");
}

std::string receive_response(int client_fd) {
    std::string response;
    char value = '\0';
    while (response.size() <= glimmer::control::kTaskProtocolMaxLineBytes) {
        const ssize_t received = ::recv(client_fd, &value, 1, 0);
        expect(received == 1, "persistent client should receive response");
        response.push_back(value);
        if (value == '\n') {
            return response;
        }
    }
    expect(false, "persistent response should stay within protocol bounds");
    return {};
}

bool test_server_lifecycle_and_authentication() {
    const std::string path = "/tmp/glimmer-control-test-" + std::to_string(::getpid()) + ".sock";
    static_cast<void>(::unlink(path.c_str()));
    Scheduler scheduler(100);
    TaskAdmissionService admission_service(scheduler, std::chrono::milliseconds::zero(), true);
    TaskControlEndpoint endpoint(admission_service, register_resource, &scheduler);
    UnixSocketControlServer server(
        endpoint, UnixSocketControlConfig{.socket_path = path, .allowed_uid = ::geteuid()});
    if (!server.start()) {
        if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
            std::cerr << "Skipping Unix socket test: local socket operations are unavailable\n";
            return false;
        }
        expect(false, "server should start");
    }
    expect(server.running(), "server should report running");
    const std::string response = exchange(server, path, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1\n");
    const auto parsed = glimmer::control::parse_task_protocol_response(response);
    expect(parsed.parsed(), "socket response should parse");
    expect(
        parsed.response.has_value() && parsed.response->kind == TaskProtocolResponseKind::kAccepted,
        "socket request should be admitted");
    const TaskId submitted_task_id =
        parsed.response.value_or(glimmer::control::TaskProtocolResponse{}).task_id;
    const std::string lease_response =
        exchange(server, path, "GLIMMER_TASK_V1 CLAIM " + std::to_string(submitted_task_id) + "\n");
    const auto lease = glimmer::control::parse_task_protocol_response(lease_response);
    expect(lease.parsed() && lease.response.has_value() &&
               lease.response->kind == TaskProtocolResponseKind::kLease,
           "socket peer should receive a process-bound lease");
    const TaskId lease_id =
        lease.response.value_or(glimmer::control::TaskProtocolResponse{}).task_id;
    const std::string complete_request =
        "GLIMMER_TASK_V1 COMPLETE " + std::to_string(lease_id) + "\n";
    const std::string complete_response = exchange(server, path, complete_request);
    const auto completed = glimmer::control::parse_task_protocol_response(complete_response);
    expect(completed.parsed() && completed.response.has_value() &&
               completed.response->kind == TaskProtocolResponseKind::kState &&
               completed.response->state == glimmer::control::TaskProtocolState::kCompleted,
           "socket peer should complete its own lease");

    const int persistent_fd = connect_client(path.c_str());
    send_request(persistent_fd, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1\n");
    expect(server.serve_one(), "server should accept a persistent client");
    const auto persistent_submit =
        glimmer::control::parse_task_protocol_response(receive_response(persistent_fd));
    expect(persistent_submit.parsed() && persistent_submit.response.has_value() &&
               persistent_submit.response->kind == TaskProtocolResponseKind::kAccepted,
           "persistent client should receive its first response");
    const TaskId persistent_submitted_task_id =
        persistent_submit.response.value_or(glimmer::control::TaskProtocolResponse{}).task_id;
    send_request(persistent_fd,
                 "GLIMMER_TASK_V1 CLAIM " + std::to_string(persistent_submitted_task_id) + "\n");
    const auto persistent_claim =
        glimmer::control::parse_task_protocol_response(receive_response(persistent_fd));
    expect(persistent_claim.parsed() && persistent_claim.response.has_value() &&
               persistent_claim.response->kind == TaskProtocolResponseKind::kLease,
           "persistent client should reuse its connection for CLAIM");
    const TaskId persistent_task_id =
        persistent_claim.response.value_or(glimmer::control::TaskProtocolResponse{}).task_id;
    send_request(persistent_fd,
                 "GLIMMER_TASK_V1 COMPLETE " + std::to_string(persistent_task_id) + "\n");
    const auto persistent_complete =
        glimmer::control::parse_task_protocol_response(receive_response(persistent_fd));
    expect(
        persistent_complete.parsed() && persistent_complete.response.has_value() &&
            persistent_complete.response->kind == TaskProtocolResponseKind::kState &&
            persistent_complete.response->state == glimmer::control::TaskProtocolState::kCompleted,
        "persistent client should complete its lease without reconnecting");
    static_cast<void>(::close(persistent_fd));
    server.stop();
    expect(!server.running(), "server should stop");
    expect(::access(path.c_str(), F_OK) != 0 && errno == ENOENT,
           "server stop should remove its socket");
    return true;
}

bool test_server_rejects_wrong_uid() {
    const std::string path = "/tmp/glimmer-control-auth-" + std::to_string(::getpid()) + ".sock";
    static_cast<void>(::unlink(path.c_str()));
    Scheduler scheduler(100);
    TaskAdmissionService admission_service(scheduler);
    TaskControlEndpoint endpoint(admission_service, register_resource, &scheduler);
    const std::uint32_t wrong_uid = static_cast<std::uint32_t>(::geteuid()) + 1;
    UnixSocketControlServer server(
        endpoint, UnixSocketControlConfig{.socket_path = path, .allowed_uid = wrong_uid});
    if (!server.start()) {
        if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
            std::cerr << "Skipping Unix socket authentication test: local socket operations are "
                         "unavailable\n";
            return false;
        }
        expect(false, "authentication test server should start");
    }
    const int client_fd = connect_client(path.c_str());
    const char request[] = "GLIMMER_TASK_V1 QUERY 1\n";
    expect(::send(client_fd, request, sizeof(request) - 1, MSG_NOSIGNAL) ==
               static_cast<ssize_t>(sizeof(request) - 1),
           "unauthorized client should send request");
    expect(server.serve_one(), "server should close unauthorized client");
    char response[8]{};
    errno = 0;
    const ssize_t received = ::recv(client_fd, response, sizeof(response), 0);
    expect(received == 0 || (received < 0 && errno == ECONNRESET),
           "unauthorized client should receive no response bytes");
    static_cast<void>(::close(client_fd));
    return true;
}

void expect_closed(int client_fd) {
    pollfd ready{.fd = client_fd, .events = POLLIN, .revents = 0};
    expect(::poll(&ready, 1, 3000) > 0, "server should close the rejected stream");
    char byte = '\0';
    const ssize_t received = ::recv(client_fd, &byte, 1, 0);
    expect(received == 0 || (received < 0 && errno == ECONNRESET),
           "rejected connection should have no further response");
}

bool test_buffered_requests_and_rejection() {
    const std::string path = "/tmp/glimmer-control-framing-" + std::to_string(::getpid()) + ".sock";
    Scheduler scheduler(100);
    TaskAdmissionService admission(scheduler);
    TaskControlEndpoint endpoint(admission, register_resource, &scheduler);
    UnixSocketControlServer server(
        endpoint, UnixSocketControlConfig{
                      .socket_path = path, .allowed_uid = ::geteuid(), .io_timeout_ms = 20});
    if (!server.start()) {
        if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
            std::cerr << "Skipping server framing test: Unix sockets unavailable\n";
            return false;
        }
        expect(false, "framing server should start");
    }

    const int coalesced = connect_client(path.c_str());
    // Queue both requests before accept: a single receive can contain them.
    send_request(coalesced, "GLIMMER_TASK_V1 SUBMIT tenant-a 20 1 1\nGLIMMER_TASK_V1 CANCEL 1\n");
    expect(server.serve_one(), "server should accept coalesced requests");
    expect(receive_response(coalesced) == "GLIMMER_TASK_V1 OK 1\n",
           "coalesced first request should be admitted once");
    expect(receive_response(coalesced) == "GLIMMER_TASK_V1 STATE 1 CANCELLED\n",
           "coalesced second request should be handled in order");
    static_cast<void>(::close(coalesced));

    const int oversized = connect_client(path.c_str());
    send_request(oversized, std::string(glimmer::control::kTaskProtocolMaxLineBytes + 1, 'x') +
                                "GLIMMER_TASK_V1 SUBMIT injected 20 1 1\n");
    expect(server.serve_one(), "server should accept the invalid stream for rejection");
    expect(receive_response(oversized) == "GLIMMER_TASK_V1 ERROR INVALID_REQUEST\n",
           "oversized line should receive one protocol error");
    expect_closed(oversized);
    expect(scheduler.stats().total_task_count == 1,
           "the suffix of an oversized line must never submit a task");
    static_cast<void>(::close(oversized));

    const int partial = connect_client(path.c_str());
    send_request(partial, "GLIMMER_TASK_V1 SUBMIT");
    expect(server.serve_one(), "server should accept the partial line");
    expect_closed(partial);
    static_cast<void>(::close(partial));

    const int eof = connect_client(path.c_str());
    send_request(eof, "GLIMMER_TASK_V1 QUERY 1");
    expect(::shutdown(eof, SHUT_WR) == 0 && server.serve_one(),
           "server should accept an EOF-terminated query");
    expect(receive_response(eof) == "GLIMMER_TASK_V1 STATE 1 CANCELLED\n",
           "EOF-framed request compatibility should be preserved");
    expect_closed(eof);
    static_cast<void>(::close(eof));

    const int idle = connect_client(path.c_str());
    expect(server.serve_one(), "server should accept idle client");
    server.stop();
    expect_closed(idle);
    static_cast<void>(::close(idle));
    return true;
}

}  // namespace

int main() {
    if (!test_server_lifecycle_and_authentication() || !test_server_rejects_wrong_uid() ||
        !test_buffered_requests_and_rejection()) {
        return 77;
    }
    return EXIT_SUCCESS;
}
