#include "glimmer/control/shared_task_control_client.h"
#include "internal/shared_control_channel.h"
#include "internal/socket_line_reader.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace glimmer::control;
using namespace glimmer::control::internal;

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
    expect(!client.request("TEST\n"), "new service incarnation is rejected");
    SharedTaskControlClient another_client(path, 2000);
    expect(!another_client.request("LOST\n"), "lost reply returns without replay");
    lost_returned.set_value();
    expect(another_client.request("NEXT\n") == "OK\n", "only explicit next call reconnects");
    next_returned.set_value();
    server.join();
    static_cast<void>(::close(listener));
    expect(::unlink(path.c_str()) == 0 && ::rmdir(directory) == 0, "remove private socket");
    return EXIT_SUCCESS;
}
