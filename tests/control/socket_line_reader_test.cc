#include "internal/socket_line_reader.h"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

using glimmer::control::kTaskProtocolMaxLineBytes;
using glimmer::control::internal::SocketLineReader;
using glimmer::control::internal::SocketLineStatus;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

class SocketPair final {
   public:
    SocketPair() {
        expect(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, descriptors_.data()) == 0,
               "socket pair should open");
    }
    ~SocketPair() {
        for (const int descriptor : descriptors_) {
            static_cast<void>(::close(descriptor));
        }
    }
    SocketPair(const SocketPair&) = delete;
    SocketPair& operator=(const SocketPair&) = delete;

    [[nodiscard]] int reader_fd() const {
        return descriptors_[0];
    }
    [[nodiscard]] int writer_fd() const {
        return descriptors_[1];
    }

    void send(std::string_view data) const {
        while (!data.empty()) {
            const ssize_t count = ::send(writer_fd(), data.data(), data.size(), MSG_NOSIGNAL);
            if (count < 0 && errno == EINTR) {
                continue;
            }
            expect(count > 0, "socket data should be sent");
            data.remove_prefix(static_cast<std::size_t>(count));
        }
    }
    void end_input() const {
        expect(::shutdown(writer_fd(), SHUT_WR) == 0, "writer should half-close");
    }

   private:
    std::array<int, 2> descriptors_{-1, -1};
};

void test_coalesced_fragmented_and_eof_lines() {
    SocketPair pair;
    SocketLineReader reader(pair.reader_fd());
    const std::string maximum(kTaskProtocolMaxLineBytes - 1, 'x');
    // The second line straddles the fixed receive buffer, even if all writes
    // coalesce. Subsequent lines must remain available without more socket data.
    pair.send("a\n" + maximum + "\n\ntail");
    pair.end_input();
    std::string line;
    for (const std::string& expected :
         {std::string("a\n"), maximum + "\n", std::string("\n"), std::string("tail")}) {
        expect(reader.read_line(line, 1000) == SocketLineStatus::kLine && line == expected,
               "buffered lines should preserve their framing and order");
    }
    expect(reader.read_line(line, 1000) == SocketLineStatus::kClosed && line.empty(),
           "EOF without data should close the connection");
}

void test_byte_fragments_and_many_lines() {
    SocketPair pair;
    SocketLineReader reader(pair.reader_fd());
    const std::string expected = "GLIMMER_TASK_V1 STATE 123 COMPLETED\n";
    std::thread writer([&] {
        for (int iteration = 0; iteration < 100; ++iteration) {
            for (const char value : expected) {
                pair.send(std::string_view(&value, 1));
            }
        }
        pair.end_input();
    });
    std::string line;
    for (int iteration = 0; iteration < 100; ++iteration) {
        expect(reader.read_line(line, 1000) == SocketLineStatus::kLine && line == expected,
               "fragmented stream should yield exactly one complete line per read");
    }
    writer.join();
    expect(reader.read_line(line, 1000) == SocketLineStatus::kClosed,
           "no extra line should appear after the sender finishes");
}

void test_line_limits() {
    for (const bool newline : {false, true}) {
        SocketPair pair;
        SocketLineReader reader(pair.reader_fd());
        const std::string maximum(kTaskProtocolMaxLineBytes - (newline ? 1 : 0), 'x');
        pair.send(maximum + (newline ? "\n" : ""));
        pair.end_input();
        std::string line;
        expect(reader.read_line(line, 1000) == SocketLineStatus::kLine &&
                   line.size() == kTaskProtocolMaxLineBytes,
               "the inclusive protocol limit should be accepted");
    }
    for (const bool newline : {false, true}) {
        SocketPair pair;
        SocketLineReader reader(pair.reader_fd());
        pair.send(std::string(kTaskProtocolMaxLineBytes, 'x') + (newline ? "\n" : "x"));
        std::string line;
        expect(reader.read_line(line, 1000) == SocketLineStatus::kTooLong &&
                   line.size() == kTaskProtocolMaxLineBytes + 1,
               "one excess byte should reject immediately, with or without newline");
    }
}

void test_timeouts_reset_and_shutdown() {
    SocketPair pair;
    SocketLineReader reader(pair.reader_fd());
    std::string line;
    expect(reader.read_line(line, 1) == SocketLineStatus::kIdle,
           "an empty idle connection may be retried");
    pair.send("one\ntwo\n");
    expect(reader.read_line(line, 1000) == SocketLineStatus::kLine && line == "one\n",
           "an idle timeout must not poison the next read");
    const std::atomic<bool> stopping{true};
    expect(reader.read_line(line, 1000, &stopping) == SocketLineStatus::kError,
           "shutdown should reject even prefetched data");
    SocketPair replacement;
    reader.reset(replacement.reader_fd());
    replacement.send("new\n");
    expect(reader.read_line(line, 1000) == SocketLineStatus::kLine && line == "new\n",
           "reset must discard bytes from the previous connection");
    replacement.send("partial");
    expect(reader.read_line(line, 1) == SocketLineStatus::kError,
           "a partial-line timeout must fail, not become an idle timeout");
    reader.reset();
    expect(reader.read_line(line, 1000) == SocketLineStatus::kError,
           "an invalid descriptor should fail without polling");
    reader.reset(pair.reader_fd());
    expect(reader.read_line(line, 0) == SocketLineStatus::kError &&
               reader.read_line(line, 60'001) == SocketLineStatus::kError,
           "invalid timeouts should fail without polling");
}

}  // namespace

int main() {
    std::array<int, 2> probe{};
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, probe.data()) != 0) {
        if (errno == EPERM || errno == EACCES || errno == ENOSYS) {
            std::cerr << "Skipping socket framing test: Unix sockets unavailable\n";
            return 77;
        }
        return EXIT_FAILURE;
    }
    const char value = 'x';
    const ssize_t sent = ::send(probe[1], &value, 1, MSG_NOSIGNAL);
    const int send_errno = errno;
    for (const int descriptor : probe) {
        static_cast<void>(::close(descriptor));
    }
    if (sent != 1) {
        if (send_errno == EPERM || send_errno == EACCES || send_errno == ENOSYS) {
            std::cerr << "Skipping socket framing test: socket I/O unavailable\n";
            return 77;
        }
        return EXIT_FAILURE;
    }
    test_coalesced_fragmented_and_eof_lines();
    test_byte_fragments_and_many_lines();
    test_line_limits();
    test_timeouts_reset_and_shutdown();
    return EXIT_SUCCESS;
}
