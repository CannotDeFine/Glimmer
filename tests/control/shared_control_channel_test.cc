#include "glimmer/control/task_protocol.h"
#include "internal/shared_control_channel.h"

#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <poll.h>
#include <span>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

// GNU ld --wrap requires this exact ABI name.
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __real_poll(pollfd*, nfds_t, int);
namespace {
struct PollPublication {
    int descriptor{-1};
    void (*publish)(void*){};
    void* context{};
    bool triggered{};
};
thread_local PollPublication poll_publication;
}  // namespace
// Only this test target wraps poll. Production keeps the actual POSIX call.
// GNU ld --wrap requires this exact ABI name.
// NOLINTNEXTLINE(bugprone-reserved-identifier,readability-identifier-naming)
extern "C" int __wrap_poll(pollfd* descriptors, nfds_t count, int timeout) {
    if (count == 1 && timeout == 0 && descriptors[0].fd == poll_publication.descriptor &&
        poll_publication.publish) {
        const auto callback = poll_publication.publish;
        poll_publication.publish = nullptr;
        poll_publication.triggered = true;
        callback(poll_publication.context);
    }
    return __real_poll(descriptors, count, timeout);
}

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
            if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
                std::exit(77);
            }
            require(false, "socket pair must create");
        }
    }
    ~Pair() {
        for (const int descriptor : descriptors) {
            require(::close(descriptor) == 0, "socket descriptor must close");
        }
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
        require(server.create_server(), "sealed channel must create");
        std::thread offer([&] {
            require(server.send_offer(pair.descriptors[0], std::chrono::milliseconds{1000}),
                    "server offer must succeed");
        });
        require(client.receive_offer(pair.descriptors[1], std::chrono::milliseconds{1000}),
                "client offer must validate");
        offer.join();
    }
};

void test_sequences_bounds_idle_and_disconnect() {
    Session session;
    std::string request;
    std::string response;
    require(!session.client.receive_response(session.pair.descriptors[1], 10, response) &&
                !session.server.publish_response("unexpected\n") &&
                !session.server.publish_request("unexpected\n") &&
                session.server.receive_request(session.pair.descriptors[0], 5, request) ==
                    SharedControlReadStatus::kIdle,
            "roles, empty capacity and idle timeout must reject without mutation");
    for (std::size_t i = 0; i < 1000; ++i) {
        const auto line = "request-" + std::to_string(i) + "\n";
        const auto reply = "response-" + std::to_string(i) + "\n";
        require(session.client.publish_request(line) &&
                    !session.client.publish_request("second\n") &&
                    session.server.receive_request(session.pair.descriptors[0], 1000, request) ==
                        SharedControlReadStatus::kLine &&
                    request == line &&
                    session.server.receive_request(session.pair.descriptors[0], 1, request) ==
                        SharedControlReadStatus::kError &&
                    session.server.publish_response(reply) &&
                    !session.server.publish_response("duplicate\n") &&
                    session.client.receive_response(session.pair.descriptors[1], 1000, response) &&
                    response == reply,
                "one outstanding request and exact matching sequence must be "
                "conserved");
    }
    const std::string maximum(kTaskProtocolMaxLineBytes - 1, 'x');
    require(session.client.publish_request(maximum + "\n") &&
                session.server.receive_request(session.pair.descriptors[0], 1000, request) ==
                    SharedControlReadStatus::kLine &&
                request == maximum + "\n" && session.server.publish_response(request) &&
                session.client.receive_response(session.pair.descriptors[1], 1000, response) &&
                response == request,
            "newline-inclusive maximum must round trip");
    require(session.client.publish_request("pending\n") &&
                !session.client.receive_response(session.pair.descriptors[1], 5, response) &&
                !session.client.publish_request("retry\n"),
            "response timeout cannot release outstanding capacity or retry");
    require(::shutdown(session.pair.descriptors[1], SHUT_RDWR) == 0,
            "client disconnect must succeed");
    // A published but unread request can still be consumed exactly once; peer
    // disconnection is not proof that a domain operation never reached service.
    require(session.server.receive_request(session.pair.descriptors[0], 1000, request) ==
                    SharedControlReadStatus::kLine &&
                request == "pending\n",
            "ambiguous disconnected publication remains one request");
    session.server.reset();
    session.client.reset();
    require(!session.server.active() && !session.client.active(), "reset must clear all mappings");
    for (const std::string& line : std::array<std::string, 5>{
             "", "missing", "two\nlines\n", maximum + "x\n", std::string("embedded\0\n", 10)}) {
        Session invalid;
        require(!invalid.client.publish_request(line), "invalid framing must never publish");
        require(invalid.server.receive_request(invalid.pair.descriptors[0], 2, request) ==
                    SharedControlReadStatus::kIdle,
                "invalid framing must not deliver an operation");
    }
}

class SpinEnvironment final {
   public:
    explicit SpinEnvironment(const char* value) {
        if (const char* previous = std::getenv("GLIMMER_CONTROL_SPIN_US")) previous_ = previous;
        require(::setenv("GLIMMER_CONTROL_SPIN_US", value, 1) == 0,
                "spin configuration must publish before any worker starts");
    }
    ~SpinEnvironment() {
        require(previous_ ? ::setenv("GLIMMER_CONTROL_SPIN_US", previous_->c_str(), 1) == 0
                          : ::unsetenv("GLIMMER_CONTROL_SPIN_US") == 0,
                "spin configuration must restore after workers join");
    }
    SpinEnvironment(const SpinEnvironment&) = delete;
    SpinEnvironment& operator=(const SpinEnvironment&) = delete;

   private:
    std::optional<std::string> previous_;
};

void test_publication_between_sequence_and_peer_close() {
    for (const bool is_response : {false, true}) {
        Session session;
        std::string line;
        if (is_response) {
            require(session.client.publish_request("request\n") &&
                        session.server.receive_request(session.pair.descriptors[0], 1000, line) ==
                            SharedControlReadStatus::kLine,
                    "prepare pending request before response race");
        }
        struct Publication {
            Session* session;
            bool is_response;
        } publication{&session, is_response};
        poll_publication = {
            is_response ? session.pair.descriptors[1] : session.pair.descriptors[0],
            [](void* value) {
                auto& pending = *static_cast<Publication*>(value);
                auto& current = *pending.session;
                require(pending.is_response ? current.server.publish_response("committed\n")
                                            : current.client.publish_request("committed\n"),
                        "publish exact sequence only after the consumer's first read");
                require(::shutdown(current.pair.descriptors[pending.is_response ? 0 : 1],
                                   SHUT_RDWR) == 0,
                        "owned peer closes immediately after committing publication");
            },
            &publication, false};
        const bool received =
            is_response ? session.client.receive_response(session.pair.descriptors[1], 1000, line)
                        : session.server.receive_request(session.pair.descriptors[0], 1000, line) ==
                              SharedControlReadStatus::kLine;
        const bool triggered = poll_publication.triggered;
        poll_publication = {};
        require(triggered && received && line == "committed\n",
                "matching committed publication must survive the exact "
                "sequence/liveness race");
        require(is_response
                    ? !session.client.receive_response(session.pair.descriptors[1], 1, line)
                    : session.server.receive_request(session.pair.descriptors[0], 1, line) ==
                          SharedControlReadStatus::kError,
                "committed publication is consumable only once, never new "
                "authority after closure");
    }
    Session absent;
    require(::shutdown(absent.pair.descriptors[1], SHUT_RDWR) == 0,
            "close peer without publishing an operation");
    std::string line;
    require(absent.server.receive_request(absent.pair.descriptors[0], 1000, line) ==
                SharedControlReadStatus::kError,
            "closure alone cannot manufacture publication");
}

void test_spin_budget_configuration() {
    const std::array<std::pair<const char*, int>, 6> policies{
        {{"0", 0}, {"1", 1}, {"20", 20}, {"100", 100}, {"1000", 1000}, {"0001", 1}}};
    for (const auto& [value, microseconds] : policies) {
        const auto parsed = parse_shared_control_spin_budget(value);
        require(parsed && parsed->count() == microseconds,
                "bounded decimal spin policy must parse");
        const SpinEnvironment setting(value);
        SharedControlChannel server;
        require(server.create_server(), "valid spin policy must permit sealed channel creation");
    }
    for (const auto* value : {"", "1001", "-1", "+1", " 1", "1 ", "1x", "999999", "00000"}) {
        require(!parse_shared_control_spin_budget(value), "invalid spin policy must reject");
        const SpinEnvironment setting(value);
        Pair pair;
        SharedControlChannel server;
        SharedControlChannel client;
        require(!server.create_server() && !server.active() &&
                    !client.receive_offer(pair.descriptors[1], std::chrono::milliseconds{10}) &&
                    !client.active(),
                "invalid policy must reject both endpoints before offers or "
                "publication");
        pollfd ready{.fd = pair.descriptors[0], .events = POLLIN, .revents = 0};
        require(::poll(&ready, 1, 0) == 0, "invalid receive policy cannot send acknowledgment");
    }
    require(!parse_shared_control_spin_budget(std::string_view("1\0", 2)),
            "embedded null policy must reject");
    const SpinEnvironment setting("1000");
    Session session;
    std::string line;
    require(session.server.receive_request(session.pair.descriptors[0], 0, line) ==
                SharedControlReadStatus::kIdle,
            "spin cannot extend an expired absolute deadline");
    require(::shutdown(session.pair.descriptors[1], SHUT_RDWR) == 0 &&
                session.server.receive_request(session.pair.descriptors[0], 1000, line) ==
                    SharedControlReadStatus::kError,
            "spin must preserve disconnected peer rejection");
}

void test_concurrent_spinning_round_trips() {
    const SpinEnvironment setting("1000");
    Session session;
    std::barrier ready(2);
    std::thread server([&] {
        for (int i = 0; i < 100; ++i) {
            ready.arrive_and_wait();
            std::string line;
            require(session.server.receive_request(session.pair.descriptors[0], 1000, line) ==
                            SharedControlReadStatus::kLine &&
                        line == "request\n" && session.server.publish_response("reply\n"),
                    "concurrent sequence publication must survive spin/futex "
                    "transitions");
        }
    });
    for (int i = 0; i < 100; ++i) {
        ready.arrive_and_wait();
        std::string line;
        require(session.client.publish_request("request\n") &&
                    session.client.receive_response(session.pair.descriptors[1], 1000, line) &&
                    line == "reply\n",
                "each spinning request must retain its exact matching response");
    }
    server.join();
}

std::size_t descriptor_count() {
    DIR* directory = ::opendir("/proc/self/fd");
    require(directory != nullptr, "descriptor inventory must open");
    std::size_t count = 0;
    while (::readdir(directory) != nullptr) {
        ++count;
    }
    require(::closedir(directory) == 0, "descriptor inventory must close");
    return count;
}

void send_descriptors(int socket, std::span<const int> descriptors, char marker) {
    const auto count = descriptors.size();
    char acknowledgment = 0;
    require(::recv(socket, &acknowledgment, 1, 0) == 1 && acknowledgment == 'O',
            "client must acknowledge before descriptor offer");
    iovec payload{.iov_base = &marker, .iov_len = 1};
    alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(9 * sizeof(int))> ancillary{};
    msghdr message{};
    message.msg_iov = &payload;
    message.msg_iovlen = 1;
    if (count != 0) {
        message.msg_control = ancillary.data();
        message.msg_controllen = CMSG_SPACE(count * sizeof(int));
        auto* header = CMSG_FIRSTHDR(&message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len = CMSG_LEN(count * sizeof(int));
        std::memcpy(CMSG_DATA(header), descriptors.data(), count * sizeof(int));
    }
    require(::sendmsg(socket, &message, MSG_NOSIGNAL) == 1, "raw offer must send");
}

void test_bad_offers_and_cleanup() {
    const auto before = descriptor_count();
    for (std::size_t count : {0U, 1U, 2U, 4U, 5U, 9U}) {
        Pair pair;
        std::array<int, 9> wrong{};
        wrong.fill(pair.descriptors[0]);
        SharedControlChannel client;
        std::thread service([&] {
            send_descriptors(pair.descriptors[0], std::span<const int>{wrong.data(), count}, 'S');
        });
        require(!client.receive_offer(pair.descriptors[1], std::chrono::milliseconds{1000}) &&
                    !client.active(),
                "missing, excess, truncated and nonsealed descriptors must reject");
        service.join();
    }
    require(descriptor_count() == before, "malformed offers must not leak received descriptors");
    for (int i = 0; i < 25; ++i) {
        Session session;
    }
    require(descriptor_count() == before, "repeated channel teardown must release all descriptors");
}

// Independent private-wire view for adversarial bounds/sequence and kernel
// sealing checks. This test does not call production publication helpers.
struct alignas(64) RawSlot {
    std::atomic<std::uint32_t> sequence{0};
    std::atomic<std::uint32_t> length{0};
    std::array<std::atomic<unsigned char>, kTaskProtocolMaxLineBytes> bytes{};
    ServiceIncarnation incarnation{};
};

void test_invalid_incarnation_offers() {
    const auto before = descriptor_count();
    for (const ServiceIncarnation incarnation :
         {ServiceIncarnation{}, ServiceIncarnation{1, 0}, ServiceIncarnation{0, 1}}) {
        SharedControlChannel invalid;
        require(!invalid.create_server(incarnation) && !invalid.active(),
                "server must not substitute predictable identity for unavailable "
                "entropy");
        Pair pair;
        std::array<int, 2> descriptors{};
        for (std::size_t i = 0; i < descriptors.size(); ++i) {
            descriptors[i] = ::memfd_create("invalid-incarnation", MFD_CLOEXEC | MFD_ALLOW_SEALING);
            require(descriptors[i] >= 0 &&
                        ::ftruncate(descriptors[i], static_cast<off_t>(sizeof(RawSlot))) == 0,
                    "malformed identity fixture backing must create");
            void* mapping = ::mmap(nullptr, sizeof(RawSlot), PROT_READ | PROT_WRITE, MAP_SHARED,
                                   descriptors[i], 0);
            require(mapping != MAP_FAILED, "malformed fixture mapping must create");
            std::construct_at(static_cast<RawSlot*>(mapping));
            static_cast<RawSlot*>(mapping)->incarnation = incarnation;
            require(::fcntl(descriptors[i], F_ADD_SEALS,
                            F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL |
                                (i == 1 ? F_SEAL_FUTURE_WRITE : 0)) == 0 &&
                        ::munmap(mapping, sizeof(RawSlot)) == 0,
                    "fixture must otherwise have valid sealed backing");
        }
        std::thread offer([&] { send_descriptors(pair.descriptors[0], descriptors, 'S'); });
        SharedControlChannel client;
        require(!client.receive_offer(pair.descriptors[1], std::chrono::milliseconds{1000}) &&
                    !client.active(),
                "absent or partial authenticated incarnation must reject before "
                "publication");
        offer.join();
        for (const int descriptor : descriptors) {
            require(::close(descriptor) == 0, "fixture backing must release");
        }
    }
    require(descriptor_count() == before, "invalid incarnation offers must not leak descriptors");
}

void test_seals_and_hostile_publication() {
    for (const auto kind : {0, 1, 2}) {
        Pair pair;
        SharedControlChannel server;
        require(server.create_server(), "raw channel must create");
        std::thread offer([&] {
            require(server.send_offer(pair.descriptors[0], std::chrono::milliseconds{1000}),
                    "raw offer must transfer");
        });
        const char acknowledgment = 'O';
        require(::send(pair.descriptors[1], &acknowledgment, 1, MSG_NOSIGNAL) == 1,
                "raw client must acknowledge");
        char marker = 0;
        iovec payload{.iov_base = &marker, .iov_len = 1};
        alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(2 * sizeof(int))> ancillary{};
        msghdr message{};
        message.msg_iov = &payload;
        message.msg_iovlen = 1;
        message.msg_control = ancillary.data();
        message.msg_controllen = ancillary.size();
        require(::recvmsg(pair.descriptors[1], &message, MSG_CMSG_CLOEXEC) == 1 && marker == 'S',
                "independent client must receive exact marker");
        const auto* header = CMSG_FIRSTHDR(&message);
        require(header != nullptr && header->cmsg_len == CMSG_LEN(2 * sizeof(int)),
                "offer must have two descriptors");
        std::array<int, 2> descriptors{};
        std::memcpy(descriptors.data(), CMSG_DATA(header), sizeof(descriptors));
        offer.join();
        for (const int descriptor : descriptors) {
            require((::fcntl(descriptor, F_GETFD) & FD_CLOEXEC) != 0,
                    "received descriptor must be close-on-exec");
        }
        for (std::size_t i = 0; i < 2; ++i) {
            require(::ftruncate(descriptors[i], 0) != 0 && errno == EPERM,
                    "peer cannot shrink backing or trigger server SIGBUS");
            require(::ftruncate(descriptors[i], 4096) != 0 && errno == EPERM,
                    "peer cannot grow backing");
        }
        void* response =
            ::mmap(nullptr, sizeof(RawSlot), PROT_READ | PROT_WRITE, MAP_SHARED, descriptors[1], 0);
        require(response == MAP_FAILED && errno == EPERM,
                "client cannot obtain writable response mapping");
        const char byte = 'x';
        require(::pwrite(descriptors[1], &byte, 1, 0) != 1 && errno == EPERM,
                "response file writes must also reject");
        void* request =
            ::mmap(nullptr, sizeof(RawSlot), PROT_READ | PROT_WRITE, MAP_SHARED, descriptors[0], 0);
        require(request != MAP_FAILED, "client owns writable request mapping");
        auto& slot = *static_cast<RawSlot*>(request);
        std::barrier ready(2);
        auto result = std::async(std::launch::async, [&] {
            ready.arrive_and_wait();
            std::string line;
            return server.receive_request(pair.descriptors[0], 1000, line);
        });
        ready.arrive_and_wait();
        slot.length.store(kind == 1 ? 257 : 2, std::memory_order_relaxed);
        slot.bytes[0].store('x', std::memory_order_relaxed);
        slot.bytes[1].store(kind == 2 ? 'x' : '\n', std::memory_order_relaxed);
        slot.sequence.store(kind == 0 ? 2 : 1, std::memory_order_release);
        require(result.get() == SharedControlReadStatus::kError,
                "skipped sequence, oversized payload and malformed framing must "
                "reject");
        require(::munmap(request, sizeof(RawSlot)) == 0, "raw mapping must release");
        for (const int descriptor : descriptors) {
            require(::close(descriptor) == 0, "raw transferred descriptor must release");
        }
    }
}

void test_v2_process_and_socket_ownership() {
    Session session;
    Pair other;
    const auto child = ::fork();
    require(child >= 0, "fork V2 endpoint owner");
    if (child == 0) {
        std::string line;
        const bool rejected =
            !session.client.active() && !session.server.active() &&
            !session.client.publish_request("child\n") &&
            session.server.receive_request(session.pair.descriptors[0], 0, line) ==
                SharedControlReadStatus::kError &&
            !session.server.send_offer(session.pair.descriptors[0], std::chrono::milliseconds{0});
        ::_exit(rejected ? 0 : 1);
    }
    int status{};
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork child cannot use parent V2 endpoints");
    require(session.client.publish_request("parent\n"), "parent retains its V2 request channel");
    std::string line;
    require(session.server.receive_request(other.descriptors[0], 0, line) ==
                SharedControlReadStatus::kError,
            "V2 server rejects a different liveness socket");
    require(session.server.receive_request(session.pair.descriptors[0], 1000, line) ==
                    SharedControlReadStatus::kLine &&
                line == "parent\n",
            "wrong socket cannot consume the parent's request");
    require(session.server.publish_response("reply\n"), "parent publishes V2 response");
    require(!session.client.receive_response(other.descriptors[1], 0, line),
            "V2 client rejects a different liveness socket");
    require(session.client.receive_response(session.pair.descriptors[1], 1000, line) &&
                line == "reply\n",
            "wrong socket cannot consume the parent's response");
    require(!session.server.send_offer(other.descriptors[0], std::chrono::milliseconds{0}),
            "an offered backing cannot be shared with a second socket");
}

}  // namespace

int main() {
    test_spin_budget_configuration();
    test_v2_process_and_socket_ownership();
    test_concurrent_spinning_round_trips();
    for (const auto* budget : {"0", "20", "1000"}) {
        const SpinEnvironment setting(budget);
        test_publication_between_sequence_and_peer_close();
        test_sequences_bounds_idle_and_disconnect();
        test_bad_offers_and_cleanup();
        test_invalid_incarnation_offers();
        test_seals_and_hostile_publication();
    }
    return EXIT_SUCCESS;
}
