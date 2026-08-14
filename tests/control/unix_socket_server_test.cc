#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_server.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

void test_server_lifecycle_and_authentication() {
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
            return;
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
    const std::string lease_response = exchange(server, path, "GLIMMER_TASK_V1 CLAIM\n");
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
    server.stop();
    expect(!server.running(), "server should stop");
    expect(::access(path.c_str(), F_OK) != 0 && errno == ENOENT,
           "server stop should remove its socket");
}

void test_server_rejects_wrong_uid() {
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
            return;
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
}

}  // namespace

int main() {
    test_server_lifecycle_and_authentication();
    test_server_rejects_wrong_uid();
    return EXIT_SUCCESS;
}
