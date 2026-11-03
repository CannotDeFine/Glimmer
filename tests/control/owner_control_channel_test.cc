#include "internal/shared_control_channel.h"

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <iostream>
#include <poll.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {
using namespace glimmer::control;
using namespace glimmer::control::internal;

void require(bool value, const char* message) {
    if (!value) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class Pair final {
   public:
    std::array<int, 2> descriptors{-1, -1};
    Pair() {
        if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, descriptors.data()) != 0) {
            if (errno == EPERM || errno == EACCES || errno == ENOSYS) std::exit(77);
            require(false, "socketpair must create");
        }
    }
    ~Pair() {
        for (const int descriptor : descriptors)
            require(::close(descriptor) == 0, "owned socket must close");
    }
    Pair(const Pair&) = delete;
    Pair& operator=(const Pair&) = delete;
};

class Session final {
   public:
    Pair pair;
    SharedControlChannel server;
    SharedControlChannel client;
    Session() {
        require(server.create_server({17, 29}, SharedControlMode::kReplyHintV3),
                "V3 sealed server must create");
        std::thread offer([&] {
            require(server.send_offer(pair.descriptors[0], std::chrono::milliseconds{1000}),
                    "V3 offer must send");
        });
        require(client.receive_offer(pair.descriptors[1], std::chrono::milliseconds{1000},
                                     SharedControlMode::kReplyHintV3),
                "V3 offer must validate");
        offer.join();
        require(client.service_incarnation() == ServiceIncarnation{17, 29},
                "authenticated immutable incarnation must survive offer");
    }
    void begin() {
        require(client.begin_request(pair.descriptors[1], "request\n") ==
                    SharedControlPublication::kPublished,
                "request must publish exactly once");
    }
    void receive() {
        std::string line;
        require(server.receive_request(pair.descriptors[0], 1000, line) ==
                        SharedControlReadStatus::kLine &&
                    line == "request\n",
                "service must receive actual request once");
    }
    void reply() {
        require(server.publish_response_with_hint(pair.descriptors[0], "response\n"),
                "actual response and nonblocking hint must publish");
    }
    void consume() {
        std::string line;
        require(client.poll_response(pair.descriptors[1], line) == SharedControlReadStatus::kLine &&
                    line == "response\n",
                "only matching published response may complete");
    }
};

void send_bytes(int descriptor, std::string_view bytes) {
    require(::send(descriptor, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL) ==
                static_cast<ssize_t>(bytes.size()),
            "scripted peer must send its bounded bytes");
}

std::size_t descriptor_count() {
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "descriptor inventory must open");
    std::size_t count = 0;
    errno = 0;
    while (const auto* entry = ::readdir(directory)) {
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    }
    require(errno == 0 && ::closedir(directory) == 0, "descriptor inventory must close");
    return count;
}

void test_pending_readiness_and_bounds() {
    Session session;
    std::string line;
    require(!session.client.publish_request("legacy\n") &&
                !session.server.publish_response("legacy\n") &&
                !session.client.receive_response(session.pair.descriptors[1], 1, line),
            "legacy methods cannot silently operate on V3");
    for (const std::string& invalid :
         {std::string{}, std::string{"missing"}, std::string{"two\nlines\n"}, std::string(257, 'x'),
          std::string("embedded\0\n", 10)})
        require(session.client.begin_request(session.pair.descriptors[1], invalid) ==
                    SharedControlPublication::kRejected,
                "invalid request must reject before publication and leave channel "
                "usable");
    for (std::size_t index = 0; index < 1000; ++index) {
        session.begin();
        require(session.client.begin_request(session.pair.descriptors[1], "second\n") ==
                    SharedControlPublication::kRejected,
                "busy channel cannot publish a second operation");
        session.receive();
        // Withhold the actual response. Polling must return without a futex
        // wait, synchronous request, assumed success or response release.
        for (std::size_t attempt = 0; attempt < 8; ++attempt)
            require(session.client.poll_response(session.pair.descriptors[1], line) ==
                        SharedControlReadStatus::kIdle,
                    "unready poll must return idle while retaining its request");
        session.reply();
        pollfd reply{.fd = session.pair.descriptors[1], .events = POLLIN, .revents = 0};
        require(::poll(&reply, 1, 0) == 1 && (reply.revents & POLLIN) != 0,
                "reply publication must make a real descriptor ready before waiting");
        session.consume();
        require(session.client.poll_response(session.pair.descriptors[1], line) ==
                    SharedControlReadStatus::kError,
                "response cannot complete twice");
    }
}

void test_late_hint_and_hint_without_response() {
    Session session;
    session.begin();
    session.receive();
    session.reply();
    char hint = 0;
    require(::recv(session.pair.descriptors[1], &hint, 1, MSG_DONTWAIT) == 1 && hint == 'R',
            "script must temporarily withhold the real hint");
    session.consume();
    session.begin();
    send_bytes(session.pair.descriptors[0], "R");
    std::string line;
    require(session.client.poll_response(session.pair.descriptors[1], line) ==
                SharedControlReadStatus::kIdle,
            "late previous hint cannot complete current request");
    session.receive();
    session.reply();
    session.consume();

    Session spurious;
    spurious.begin();
    send_bytes(spurious.pair.descriptors[0], "R");
    require(spurious.client.poll_response(spurious.pair.descriptors[1], line) ==
                SharedControlReadStatus::kIdle,
            "even a syntactically valid hint is not a control result");
}

void test_malformed_hints_and_closure() {
    for (const auto* bytes : {"X", "RR", "RRR"}) {
        Session session;
        session.begin();
        send_bytes(session.pair.descriptors[0], bytes);
        std::string line;
        require(session.client.poll_response(session.pair.descriptors[1], line) ==
                        SharedControlReadStatus::kError &&
                    session.client.begin_request(session.pair.descriptors[1], "retry\n") ==
                        SharedControlPublication::kRejected,
                "malformed/excess/future hints must fail closed without replay");
    }
    for (const bool has_response : {false, true}) {
        Session session;
        session.begin();
        session.receive();
        if (has_response) session.reply();
        require(::shutdown(session.pair.descriptors[0], SHUT_RDWR) == 0,
                "scripted server closure must succeed");
        std::string line;
        const auto status = session.client.poll_response(session.pair.descriptors[1], line);
        require(status == (has_response ? SharedControlReadStatus::kLine
                                        : SharedControlReadStatus::kError) &&
                    (!has_response || line == "response\n") &&
                    session.client.begin_request(session.pair.descriptors[1], "successor\n") ==
                        SharedControlPublication::kRejected,
                "closure preserves only already published response and blocks "
                "successor");
    }
    Session failed_hint;
    failed_hint.begin();
    failed_hint.receive();
    require(::shutdown(failed_hint.pair.descriptors[1], SHUT_RD) == 0,
            "scripted receive shutdown must succeed");
    require(!failed_hint.server.publish_response_with_hint(failed_hint.pair.descriptors[0],
                                                           "response\n"),
            "hint send failure must not be silently accepted or retried");
    failed_hint.consume();
    require(failed_hint.client.begin_request(failed_hint.pair.descriptors[1], "successor\n") ==
                SharedControlPublication::kRejected,
            "published response cannot repair a broken notification connection");
}

void test_process_socket_and_version_ownership() {
    Session session;
    Pair other;
    require(session.client.begin_request(other.descriptors[1], "wrong\n") ==
                SharedControlPublication::kRejected,
            "borrowed socket identity must match negotiated endpoint");
    session.begin();
    const auto child = ::fork();
    require(child >= 0, "fork fixture must create owned child");
    if (child == 0) {
        std::string line;
        const bool rejected =
            session.client.begin_request(session.pair.descriptors[1], "fork\n") ==
                SharedControlPublication::kRejected &&
            session.client.poll_response(session.pair.descriptors[1], line) ==
                SharedControlReadStatus::kError &&
            session.server.receive_request(session.pair.descriptors[0], 1, line) ==
                SharedControlReadStatus::kError;
        ::_exit(rejected ? 0 : 1);
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork endpoint rejection child must exit and be reaped");
    session.receive();
    require(!session.server.publish_response_with_hint(other.descriptors[0], "wrong\n"),
            "server reply socket must match without consuming the pending response");
    session.reply();
    session.consume();
    for (const bool is_server_v3 : {false, true}) {
        Pair pair;
        SharedControlChannel server;
        SharedControlChannel client;
        require(server.create_server({17, 29}, is_server_v3 ? SharedControlMode::kReplyHintV3
                                                            : SharedControlMode::kV2),
                "mismatch fixture server must create");
        std::thread offer([&] {
            require(server.send_offer(pair.descriptors[0], std::chrono::milliseconds{1000}),
                    "mismatch offer must send");
        });
        require(!client.receive_offer(
                    pair.descriptors[1], std::chrono::milliseconds{1000},
                    is_server_v3 ? SharedControlMode::kV2 : SharedControlMode::kReplyHintV3) &&
                    !client.active(),
                "V2/V3 offer mismatch cannot silently downgrade or retain descriptors");
        offer.join();
    }
}

}  // namespace

int main() {
    const auto descriptors = descriptor_count();
    test_pending_readiness_and_bounds();
    test_late_hint_and_hint_without_response();
    test_malformed_hints_and_closure();
    test_process_socket_and_version_ownership();
    require(descriptor_count() == descriptors, "all descriptors must be retired");
}
