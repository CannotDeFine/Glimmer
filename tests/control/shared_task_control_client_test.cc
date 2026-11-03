#include "glimmer/control/shared_task_control_client.h"
#include "glimmer/control/versioned_task_control_client.h"
#include "internal/shared_control_channel.h"
#include "internal/socket_line_reader.h"

#include <chrono>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <future>
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
using namespace glimmer::control;
using namespace glimmer::control::internal;

static_assert(static_cast<unsigned>(VersionedControlTransport::kSocket) == 0);
static_assert(static_cast<unsigned>(VersionedControlTransport::kSharedMemory) == 1);
static_assert(static_cast<unsigned>(VersionedControlTransport::kAuto) == 2);

void expect(bool value, const char* message) {
    if (!value) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void offer(int listener, ServiceIncarnation identity, bool reply, std::promise<void>& closed) {
    const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    expect(fd >= 0, "accept client");
    SocketLineReader reader(fd);
    std::string line;
    expect(reader.read_line(line, 2000) == SocketLineStatus::kLine && line == kSharedControlOpen,
           "client must negotiate shared transport");
    expect(::send(fd, kSharedControlReady.data(), kSharedControlReady.size(), MSG_NOSIGNAL) ==
               static_cast<ssize_t>(kSharedControlReady.size()),
           "server sends ready");
    SharedControlChannel channel;
    expect(channel.create_server(identity) && channel.send_offer(fd, std::chrono::seconds(2)),
           "server offers channel");
    if (reply) {
        for (int i = 0; i < 3; ++i) {
            expect(channel.receive_request(fd, 2000, line) == SharedControlReadStatus::kLine &&
                       line == "TEST\n" && channel.publish_response("OK\n"),
                   "concurrent requests are serialized and delivered once");
        }
        // The client signals after it has consumed the response.
        closed.get_future().wait();
    } else {
        expect(channel.receive_request(fd, 2000, line) == SharedControlReadStatus::kError,
               "changed service never receives a domain request");
    }
    static_cast<void>(::close(fd));
}

void offer_lost_reply(int listener, bool should_reply, std::promise<void>& release) {
    const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    expect(fd >= 0, "accept lost-reply client");
    SocketLineReader reader(fd);
    std::string line;
    expect(reader.read_line(line, 2000) == SocketLineStatus::kLine && line == kSharedControlOpen,
           "negotiate lost-reply channel");
    expect(::send(fd, kSharedControlReady.data(), kSharedControlReady.size(), MSG_NOSIGNAL) ==
               static_cast<ssize_t>(kSharedControlReady.size()),
           "send lost-reply ready");
    SharedControlChannel channel;
    expect(channel.create_server({3, 4}) && channel.send_offer(fd, std::chrono::seconds(2)),
           "offer stable incarnation");
    expect(channel.receive_request(fd, 2000, line) == SharedControlReadStatus::kLine &&
               line == (should_reply ? "NEXT\n" : "LOST\n"),
           "receive exactly the explicitly submitted request");
    if (should_reply) {
        expect(channel.publish_response("OK\n"), "reply to explicit next request");
        release.get_future().wait();
    }
    static_cast<void>(::close(fd));
}

void test_rejected_negotiation(int listener, const std::string& path, bool automatic) {
    for (const std::string_view reply :
         {"malformed\n", "GLIMMER_SHM_V1 READY\n", "GLIMMER_SHM_V2 READY\n", "partial"}) {
        std::thread peer([&] {
            const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
            expect(fd >= 0, "accept malformed-negotiation client");
            SocketLineReader reader(fd);
            std::string line;
            expect(reader.read_line(line, 2000) == SocketLineStatus::kLine &&
                       line == kSharedControlOpen,
                   "client sends only negotiation before shared channel is ready");
            expect(::send(fd, reply.data(), reply.size(), MSG_NOSIGNAL) ==
                       static_cast<ssize_t>(reply.size()),
                   "send malformed or incomplete offer");
            pollfd ready{.fd = fd, .events = POLLIN, .revents = 0};
            expect(::poll(&ready, 1, 2000) > 0, "client finishes failed negotiation");
            char byte = 0;
            if (reply == kSharedControlReady) {
                expect(::recv(fd, &byte, 1, 0) == 1 && byte == 'O',
                       "ready response allows only descriptor acknowledgment");
                expect(::poll(&ready, 1, 2000) > 0, "missing descriptor offer times out");
            }
            expect(::recv(fd, &byte, 1, 0) == 0,
                   "failed negotiation closes without publishing domain work");
            expect(::close(fd) == 0, "close rejected negotiation peer");
        });
        if (automatic) {
            VersionedTaskControlClient client(path, VersionedControlTransport::kAuto, 200);
            expect(!client.request("GLIMMER_TASK_EXEC_V1 STATS\n"),
                   "auto mode rejects malformed or incomplete negotiation");
        } else {
            SharedTaskControlClient client(path, 200);
            expect(!client.request("GLIMMER_TASK_EXEC_V1 STATS\n"),
                   "shared mode rejects malformed or incomplete negotiation");
        }
        peer.join();
        pollfd pending{.fd = listener, .events = POLLIN, .revents = 0};
        expect(::poll(&pending, 1, 0) == 0, "rejected negotiation does not reconnect");
    }
}

void test_explicit_fallback(int listener, const std::string& path, std::string_view unsupported) {
    std::thread peer([&] {
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(fd >= 0, "accept fallback client");
        SocketLineReader reader(fd);
        std::string line;
        expect(
            reader.read_line(line, 2000) == SocketLineStatus::kLine && line == kSharedControlOpen,
            "auto mode sends negotiation first");
        expect(::send(fd, unsupported.data(), unsupported.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(unsupported.size()),
               "explicit unsupported response sends");
        expect(reader.read_line(line, 2000) == SocketLineStatus::kLine && line == "FIRST\n",
               "fallback publishes once on the same socket");
        constexpr std::string_view response = "OK\n";
        expect(::send(fd, response.data(), response.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(response.size()),
               "fallback response sends");
        expect(reader.read_line(line, 2000) == SocketLineStatus::kLine && line == "LOST\n",
               "second socket request publishes once");
        expect(::close(fd) == 0, "close without the second response");
    });
    VersionedTaskControlClient client(path, VersionedControlTransport::kAuto, 1000);
    expect(client.request("FIRST\n") == "OK\n", "explicit unsupported enables fallback");
    const pid_t child = ::fork();
    expect(child >= 0, "fork fallback client");
    if (child == 0) {
        ::_exit(client.request("CHILD\n").has_value() ? EXIT_FAILURE : EXIT_SUCCESS);
    }
    int status = 0;
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork child rejects inherited fallback client");
    expect(!client.request("LOST\n") && !client.request("AFTER\n"),
           "lost socket response retires session without replay or reconnect");
    peer.join();
    pollfd pending{.fd = listener, .events = POLLIN, .revents = 0};
    expect(::poll(&pending, 1, 0) == 0, "retired fallback session never reconnects");
}

void test_strict_shared_rejects_unsupported(int listener, const std::string& path) {
    std::thread peer([&] {
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(fd >= 0, "accept strict shared client");
        SocketLineReader reader(fd);
        std::string line;
        expect(
            reader.read_line(line, 2000) == SocketLineStatus::kLine && line == kSharedControlOpen,
            "strict client negotiates before publication");
        expect(::send(fd, kSharedControlUnsupported.data(), kSharedControlUnsupported.size(),
                      MSG_NOSIGNAL) == static_cast<ssize_t>(kSharedControlUnsupported.size()),
               "strict peer sends unsupported");
        expect(reader.read_line(line, 2000) == SocketLineStatus::kClosed,
               "strict shared mode closes without domain request");
        expect(::close(fd) == 0, "close strict peer");
    });
    VersionedTaskControlClient client(path, VersionedControlTransport::kSharedMemory, 1000);
    expect(!client.request("NEVER\n"), "strict shared mode cannot fall back");
    peer.join();
}

void test_socket_mode_and_invalid_selection(int listener, const std::string& path) {
    VersionedTaskControlClient invalid(path, static_cast<VersionedControlTransport>(255), 1000);
    expect(!invalid.request("NEVER\n"), "invalid mode is rejected before connecting");
    pollfd pending{.fd = listener, .events = POLLIN, .revents = 0};
    expect(::poll(&pending, 1, 0) == 0, "invalid mode never contacts service");
    std::thread peer([&] {
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(fd >= 0, "accept explicit socket client");
        SocketLineReader reader(fd);
        std::string line;
        expect(reader.read_line(line, 2000) == SocketLineStatus::kLine && line == "SOCKET\n",
               "explicit socket mode does not send a shared-channel negotiation");
        constexpr std::string_view response = "OK\n";
        expect(::send(fd, response.data(), response.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(response.size()),
               "socket response sends");
        expect(reader.read_line(line, 2000) == SocketLineStatus::kClosed,
               "explicit socket session closes cleanly");
        expect(::close(fd) == 0, "close socket peer");
    });
    {
        VersionedTaskControlClient client(path, VersionedControlTransport::kSocket, 1000);
        expect(client.request("SOCKET\n") == "OK\n", "explicit socket request succeeds");
    }
    peer.join();
}

void test_pinned_auto_cannot_downgrade(int listener, const std::string& path) {
    std::promise<void> consumed;
    std::promise<void> disconnected;
    std::thread peer([&] {
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(fd >= 0, "accept shared auto client");
        SocketLineReader reader(fd);
        std::string line;
        expect(
            reader.read_line(line, 2000) == SocketLineStatus::kLine && line == kSharedControlOpen,
            "auto client negotiates shared channel");
        expect(::send(fd, kSharedControlReady.data(), kSharedControlReady.size(), MSG_NOSIGNAL) ==
                   static_cast<ssize_t>(kSharedControlReady.size()),
               "send shared ready");
        SharedControlChannel channel;
        expect(channel.create_server({7, 8}) && channel.send_offer(fd, std::chrono::seconds(2)) &&
                   channel.receive_request(fd, 2000, line) == SharedControlReadStatus::kLine &&
                   line == "FIRST\n" && channel.publish_response("OK\n"),
               "auto client pins an actual shared service");
        consumed.get_future().wait();
        expect(::close(fd) == 0, "disconnect pinned channel");
        disconnected.set_value();
        const int replacement = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(replacement >= 0, "accept pinned reconnect");
        SocketLineReader next(replacement);
        expect(next.read_line(line, 2000) == SocketLineStatus::kLine && line == kSharedControlOpen,
               "pinned client negotiates again");
        expect(
            ::send(replacement, kSharedControlUnsupported.data(), kSharedControlUnsupported.size(),
                   MSG_NOSIGNAL) == static_cast<ssize_t>(kSharedControlUnsupported.size()),
            "replacement reports unsupported");
        expect(next.read_line(line, 2000) == SocketLineStatus::kClosed,
               "pinned client cannot publish on a downgraded socket");
        expect(::close(replacement) == 0, "close replacement peer");
    });
    VersionedTaskControlClient client(path, VersionedControlTransport::kAuto, 1000);
    expect(client.request("FIRST\n") == "OK\n", "auto client establishes shared identity");
    consumed.set_value();
    disconnected.get_future().wait();
    expect(!client.request("LOST\n"), "disconnected request fails without retry");
    expect(!client.request("NEVER\n"), "pinned client rejects unsupported downgrade");
    peer.join();
}

void test_auto_reconnect_identity(int listener, const std::string& path,
                                  ServiceIncarnation next_identity) {
    const bool same = next_identity == ServiceIncarnation{7, 8};
    std::promise<void> consumed;
    std::promise<void> disconnected;
    std::promise<void> second_consumed;
    std::thread peer([&] {
        const auto setup = [&](ServiceIncarnation identity, SharedControlChannel& channel) {
            const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
            expect(fd >= 0, "accept shared identity connection");
            SocketLineReader reader(fd);
            std::string line;
            expect(reader.read_line(line, 2000) == SocketLineStatus::kLine &&
                       line == kSharedControlOpen,
                   "identity connection negotiates");
            expect(::send(fd, kSharedControlReady.data(), kSharedControlReady.size(),
                          MSG_NOSIGNAL) == static_cast<ssize_t>(kSharedControlReady.size()) &&
                       channel.create_server(identity) &&
                       channel.send_offer(fd, std::chrono::seconds(2)),
                   "identity connection offers an authenticated channel");
            return fd;
        };
        SharedControlChannel first_channel;
        const int first_fd = setup({7, 8}, first_channel);
        std::string line;
        expect(
            first_channel.receive_request(first_fd, 2000, line) == SharedControlReadStatus::kLine &&
                line == "FIRST\n" && first_channel.publish_response("OK\n"),
            "initial identity serves one request");
        consumed.get_future().wait();
        expect(::close(first_fd) == 0, "disconnect initial identity");
        disconnected.set_value();
        SharedControlChannel second_channel;
        const int second_fd = setup(next_identity, second_channel);
        line.clear();
        const auto received = second_channel.receive_request(second_fd, 2000, line);
        if (same) {
            expect(received == SharedControlReadStatus::kLine && line == "SECOND\n" &&
                       second_channel.publish_response("SAME\n"),
                   "same incarnation serves only the explicit next request");
            second_consumed.get_future().wait();
        } else {
            expect(received == SharedControlReadStatus::kError && line.empty(),
                   "changed incarnation receives no domain publication");
        }
        expect(::close(second_fd) == 0, "close second identity connection");
    });
    VersionedTaskControlClient client(path, VersionedControlTransport::kAuto, 1000);
    expect(client.request("FIRST\n") == "OK\n", "auto pins initial shared identity");
    consumed.set_value();
    disconnected.get_future().wait();
    expect(!client.request("LOST\n"), "lost shared request is not replayed");
    const auto response = client.request("SECOND\n");
    expect(same ? response == "SAME\n" : !response,
           "reconnect accepts only the same pair of identity words");
    second_consumed.set_value();
    peer.join();
}

void test_concurrent_auto_fallback(int listener, const std::string& path, bool lose_reply) {
    constexpr std::size_t count = 4;
    std::thread peer([&] {
        const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
        expect(fd >= 0, "accept concurrent client");
        SocketLineReader reader(fd);
        std::string line;
        expect(
            reader.read_line(line, 2000) == SocketLineStatus::kLine && line == kSharedControlOpen,
            "concurrent client negotiates once");
        expect(::send(fd, kSharedControlUnsupported.data(), kSharedControlUnsupported.size(),
                      MSG_NOSIGNAL) == static_cast<ssize_t>(kSharedControlUnsupported.size()),
               "concurrent peer enables fallback");
        std::array<bool, count> seen{};
        for (std::size_t i = 0; i < (lose_reply ? 1U : count); ++i) {
            expect(reader.read_line(line, 2000) == SocketLineStatus::kLine,
                   "one concurrent domain request arrives");
            bool matched = false;
            for (std::size_t n = 0; n < count; ++n) {
                if (line == "REQ-" + std::to_string(n) + "\n") {
                    expect(!seen[n], "domain request is not duplicated");
                    seen[n] = true;
                    matched = true;
                }
            }
            expect(matched, "concurrent domain request is valid");
            if (lose_reply) break;
            expect(::send(fd, line.data(), line.size(), MSG_NOSIGNAL) ==
                       static_cast<ssize_t>(line.size()),
                   "each concurrent request receives its own response");
        }
        if (!lose_reply) {
            expect(reader.read_line(line, 2000) == SocketLineStatus::kClosed,
                   "concurrent client closes after all requests");
        }
        expect(::close(fd) == 0, "close concurrent peer");
    });
    {
        VersionedTaskControlClient client(path, VersionedControlTransport::kAuto, 1000);
        std::array<std::thread, count> workers;
        std::barrier start(static_cast<std::ptrdiff_t>(count));
        for (std::size_t n = 0; n < count; ++n) {
            workers[n] = std::thread([&, n] {
                const auto line = "REQ-" + std::to_string(n) + "\n";
                start.arrive_and_wait();
                const auto response = client.request(line);
                expect(lose_reply ? !response : response == line,
                       "serialized request has its own response or fails after loss");
            });
        }
        for (auto& worker : workers) worker.join();
    }
    peer.join();
    pollfd pending{.fd = listener, .events = POLLIN, .revents = 0};
    expect(::poll(&pending, 1, 0) == 0, "concurrent failure does not reconnect or replay");
}

}  // namespace

int main() {
    char directory[] = "/tmp/glimmer-shared-client-XXXXXX";
    expect(::mkdtemp(directory) != nullptr, "create private socket directory");
    const std::string path = std::string(directory) + "/control.sock";
    const int listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    expect(listener >= 0, "create listener");
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
    expect(::bind(listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) == 0 &&
               ::listen(listener, 2) == 0,
           "bind private listener");

    std::promise<void> first_response;
    std::promise<void> first_closed;
    std::promise<void> unused;
    std::promise<void> lost_returned;
    std::promise<void> next_returned;
    std::thread server([&] {
        offer(listener, {1, 2}, true, first_response);
        first_closed.set_value();
        offer(listener, {1, 3}, false, unused);
        offer(listener, {3, 2}, false, unused);
        offer(listener, {3, 4}, false, unused);
        offer_lost_reply(listener, false, unused);
        lost_returned.get_future().wait();
        offer_lost_reply(listener, true, next_returned);
    });
    SharedTaskControlClient client(path, 2000);
    expect(!client.request("TEST\nOTHER\n"), "multiple lines reject before publication");
    auto first = std::async(std::launch::async, [&] { return client.request("TEST\n"); });
    auto second = std::async(std::launch::async, [&] { return client.request("TEST\n"); });
    expect(first.get() == "OK\n" && second.get() == "OK\n", "shared requests receive replies");
    const pid_t child = ::fork();
    expect(child >= 0, "fork for inherited client check");
    if (child == 0) {
        ::_exit(client.request("TEST\n").has_value() ? EXIT_FAILURE : EXIT_SUCCESS);
    }
    int status = 0;
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork child cannot use inherited session");
    expect(client.request("TEST\n") == "OK\n", "parent session survives fork");
    first_response.set_value();
    first_closed.get_future().wait();
    expect(!client.request("TEST\n"), "lost request is not replayed");
    expect(!client.request("TEST\n"), "changed second incarnation word is rejected");
    expect(!client.request("TEST\n"), "changed first incarnation word is rejected");
    expect(!client.request("TEST\n"), "both changed incarnation words are rejected");
    SharedTaskControlClient another_client(path, 2000);
    expect(!another_client.request("LOST\n"), "lost reply returns without replay");
    lost_returned.set_value();
    expect(another_client.request("NEXT\n") == "OK\n", "only explicit next call reconnects");
    next_returned.set_value();
    server.join();
    test_rejected_negotiation(listener, path, false);
    test_rejected_negotiation(listener, path, true);
    test_explicit_fallback(listener, path, kSharedControlUnsupported);
    test_explicit_fallback(listener, path, "GLIMMER_TASK_V1 ERROR UNSUPPORTED_VERSION\n");
    test_strict_shared_rejects_unsupported(listener, path);
    test_socket_mode_and_invalid_selection(listener, path);
    test_pinned_auto_cannot_downgrade(listener, path);
    test_auto_reconnect_identity(listener, path, {7, 8});
    test_auto_reconnect_identity(listener, path, {7, 9});
    test_auto_reconnect_identity(listener, path, {9, 8});
    test_concurrent_auto_fallback(listener, path, false);
    test_concurrent_auto_fallback(listener, path, true);
    static_cast<void>(::close(listener));
    expect(::unlink(path.c_str()) == 0 && ::rmdir(directory) == 0, "remove private socket");
    return EXIT_SUCCESS;
}
