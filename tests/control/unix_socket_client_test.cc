#include "glimmer/control/task_protocol.h"
#include "glimmer/control/unix_socket_client.h"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <poll.h>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

using glimmer::control::UnixSocketControlClient;
constexpr std::string_view kRequest = "GLIMMER_TASK_V1 QUERY 1\n";

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class Descriptor final {
   public:
    explicit Descriptor(int value) : value_(value) {
        expect(value >= 0, "descriptor should be valid");
    }
    ~Descriptor() {
        static_cast<void>(::close(value_));
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    [[nodiscard]] int get() const {
        return value_;
    }

   private:
    int value_;
};

class Listener final {
   public:
    Listener() : descriptor_(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0)) {
        std::array<char, 40> pattern{};
        const std::string name = "/tmp/glimmer-client-test-XXXXXX";
        std::memcpy(pattern.data(), name.c_str(), name.size() + 1);
        const char* directory = ::mkdtemp(pattern.data());
        expect(directory != nullptr, "private test directory should be created");
        directory_ = directory;
        path_ = directory_ + "/control.sock";
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, path_.c_str(), path_.size() + 1);
        const auto size =
            static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path_.size() + 1);
        if (::bind(descriptor_.get(), reinterpret_cast<const sockaddr*>(&address), size) != 0) {
            const int bind_errno = errno;
            static_cast<void>(::rmdir(directory_.c_str()));
            if (bind_errno == EPERM || bind_errno == EACCES || bind_errno == ENOSYS) {
                std::cerr << "Skipping client transport test: socket binding unavailable\n";
                std::exit(77);
            }
            expect(false, "fake service should bind");
        }
        expect(::listen(descriptor_.get(), 4) == 0, "fake service should listen");
    }
    ~Listener() {
        static_cast<void>(::unlink(path_.c_str()));
        static_cast<void>(::rmdir(directory_.c_str()));
    }
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    [[nodiscard]] const std::string& path() const {
        return path_;
    }

    [[nodiscard]] int accept() const {
        pollfd ready{.fd = descriptor_.get(), .events = POLLIN, .revents = 0};
        expect(::poll(&ready, 1, 3000) > 0 && (ready.revents & POLLIN) != 0,
               "client should connect within the test deadline");
        return ::accept4(descriptor_.get(), nullptr, nullptr, SOCK_CLOEXEC);
    }

   private:
    Descriptor descriptor_;
    std::string directory_;
    std::string path_;
};

void receive_request(int descriptor) {
    // Independent byte-wise oracle: the peer does not use the optimized reader.
    for (const char expected : kRequest) {
        pollfd ready{.fd = descriptor, .events = POLLIN, .revents = 0};
        expect(::poll(&ready, 1, 3000) > 0, "request should arrive before the deadline");
        char actual = '\0';
        expect(::recv(descriptor, &actual, 1, 0) == 1 && actual == expected,
               "exactly one unchanged request should arrive");
    }
}

void send_reply(int descriptor, std::string_view reply) {
    while (!reply.empty()) {
        const ssize_t count = ::send(descriptor, reply.data(), reply.size(), MSG_NOSIGNAL);
        expect(count > 0, "fake service should send reply");
        reply.remove_prefix(static_cast<std::size_t>(count));
    }
}

void test_persistent_read_ahead() {
    Listener listener;
    std::thread service([&] {
        const Descriptor peer(listener.accept());
        receive_request(peer.get());
        // Deliberately coalesce frames to check transport ordering. The real
        // endpoint normally answers sequential requests, not unsolicited lines.
        send_reply(peer.get(), "first\nsecond\n");
        receive_request(peer.get());
    });
    UnixSocketControlClient client(listener.path());
    expect(client.request(kRequest) == "first\n", "first frame should be returned alone");
    expect(client.request(kRequest) == "second\n", "next frame must not be lost or merged");
    service.join();
}

void test_response_limits_eof_and_reconnect() {
    Listener listener;
    const std::string maximum(glimmer::control::kTaskProtocolMaxLineBytes - 1, 'x');
    const std::array<std::string, 5> replies{maximum + "\n", maximum + "x\n", "fresh\n", "eof", ""};
    std::thread service([&] {
        for (const auto& reply : replies) {
            const Descriptor peer(listener.accept());
            receive_request(peer.get());
            send_reply(peer.get(), reply);
        }
    });
    // Change timeout to force a new connection for each EOF fixture. After the
    // oversized reply, the identical client must reconnect without a retry.
    UnixSocketControlClient first(listener.path(), 1000);
    expect(first.request(kRequest) == maximum + "\n", "maximum reply should be accepted");
    UnixSocketControlClient second(listener.path(), 1001);
    expect(!second.request(kRequest), "newline must count toward the response limit");
    expect(second.request(kRequest) == "fresh\n", "failure should discard connection state");
    UnixSocketControlClient third(listener.path(), 1002);
    expect(third.request(kRequest) == "eof", "bounded EOF-terminated reply remains supported");
    UnixSocketControlClient fourth(listener.path(), 1003);
    expect(!fourth.request(kRequest), "empty EOF should be a transport failure");
    service.join();
}

void test_path_and_timeout_reset() {
    Listener first;
    Listener second;
    std::thread service([&] {
        const Descriptor peer(first.accept());
        receive_request(peer.get());
        send_reply(peer.get(), "first\nstale\n");
        const Descriptor replacement(first.accept());
        receive_request(replacement.get());
        send_reply(replacement.get(), "timeout\nstale\n");
        const Descriptor other(second.accept());
        receive_request(other.get());
        send_reply(other.get(), "path\n");
    });
    UnixSocketControlClient client(first.path(), 1000);
    expect(client.request(kRequest) == "first\n", "first connection should respond");
    UnixSocketControlClient changed_timeout(first.path(), 1001);
    expect(changed_timeout.request(kRequest) == "timeout\n",
           "timeout change should discard read-ahead");
    UnixSocketControlClient changed_path(second.path(), 1001);
    expect(changed_path.request(kRequest) == "path\n", "path change should discard read-ahead");
    service.join();
}

void test_partial_timeout_does_not_retry() {
    Listener listener;
    std::thread service([&] {
        const Descriptor peer(listener.accept());
        receive_request(peer.get());
        send_reply(peer.get(), "partial");
        pollfd ready{.fd = peer.get(), .events = POLLIN, .revents = 0};
        expect(::poll(&ready, 1, 3000) > 0, "timed-out client should disconnect");
        char value = '\0';
        expect(::recv(peer.get(), &value, 1, 0) == 0, "failed request should not be resent");
        const Descriptor replacement(listener.accept());
        receive_request(replacement.get());
        send_reply(replacement.get(), "fresh\n");
    });
    UnixSocketControlClient client(listener.path(), 100);
    expect(!client.request(kRequest), "partial response must time out and discard connection");
    expect(client.request(kRequest) == "fresh\n", "next explicit request should reconnect");
    service.join();
}

void test_fork_discards_parent_read_ahead() {
    Listener listener;
    std::thread service([&] {
        const Descriptor peer(listener.accept());
        receive_request(peer.get());
        send_reply(peer.get(), "parent\nstale\n");
    });
    UnixSocketControlClient client(listener.path());
    expect(client.request(kRequest) == "parent\n", "parent should populate its receive buffer");
    service.join();
    // Fork only after joining the peer thread. The child inherits the client's
    // TLS storage but must establish its own authenticated connection.
    const pid_t child = ::fork();
    expect(child >= 0, "fork should succeed");
    if (child == 0) {
        ::_exit(client.request(kRequest) == "child\n" ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    const Descriptor peer(listener.accept());
    receive_request(peer.get());
    send_reply(peer.get(), "child\n");
    int status = 0;
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "child must not use the parent's descriptor or prefetched response");
}

}  // namespace

int main() {
    const int probe = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (probe < 0) {
        if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
            std::cerr << "Skipping client transport test: Unix sockets unavailable\n";
            return 77;
        }
        return EXIT_FAILURE;
    }
    static_cast<void>(::close(probe));
    test_persistent_read_ahead();
    test_response_limits_eof_and_reconnect();
    test_path_and_timeout_reset();
    test_partial_timeout_does_not_retry();
    test_fork_discards_parent_read_ahead();
    return EXIT_SUCCESS;
}
