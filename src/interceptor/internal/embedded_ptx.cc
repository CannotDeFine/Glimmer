#include "embedded_ptx.h"

#include "ptx_transform.h"

#include <array>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace glimmer::interceptor {
namespace {

constexpr auto kMaxToolOutputBytes = kMaxPersistentPtxBytes * 4;
constexpr std::size_t kMaxImages = 32;

struct Pipe {
    std::array<int, 2> descriptors{-1, -1};
    ~Pipe() {
        for (const auto descriptor : descriptors) {
            if (descriptor >= 0) {
                (void)::close(descriptor);
            }
        }
    }
};

struct SpawnActions {
    posix_spawn_file_actions_t actions{};
    bool is_initialized = false;
    ~SpawnActions() {
        if (is_initialized) {
            (void)::posix_spawn_file_actions_destroy(&actions);
        }
    }
};

struct Child {
    pid_t pid = -1;
    ~Child() {
        if (pid > 0) {
            // Only the child created by this invocation is terminated/reaped.
            (void)::kill(pid, SIGKILL);
            int status = 0;
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }
};

std::optional<std::string> read_output(int descriptor, Child& child,
                                       std::chrono::milliseconds timeout) {
    std::string output;
    std::array<char, 8192> buffer{};
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    bool has_eof = false;
    int status = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!has_eof) {
            const auto count = ::read(descriptor, buffer.data(), buffer.size());
            if (count > 0) {
                const auto bytes = static_cast<std::size_t>(count);
                if (bytes > kMaxToolOutputBytes - output.size()) {
                    return std::nullopt;
                }
                output.append(buffer.data(), bytes);
                continue;
            }
            if (count == 0) {
                has_eof = true;
            } else if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK) {
                return std::nullopt;
            }
        }
        const auto reaped = ::waitpid(child.pid, &status, WNOHANG);
        if (reaped == child.pid) {
            child.pid = -1;
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                return std::nullopt;
            }
            // The pipe can still contain buffered bytes after child exit.
            while (!has_eof) {
                const auto count = ::read(descriptor, buffer.data(), buffer.size());
                if (count == 0) {
                    has_eof = true;
                } else if (count > 0) {
                    const auto bytes = static_cast<std::size_t>(count);
                    if (bytes > kMaxToolOutputBytes - output.size()) {
                        return std::nullopt;
                    }
                    output.append(buffer.data(), bytes);
                } else if (errno != EINTR) {
                    return std::nullopt;
                }
            }
            return output;
        }
        if (reaped < 0 && errno != EINTR) {
            // ECHILD means another application handler already reaped it.
            if (errno == ECHILD) {
                child.pid = -1;
            }
            return std::nullopt;
        }
        pollfd poll_descriptor{descriptor, POLLIN, 0};
        if (::poll(&poll_descriptor, 1, 10) < 0 && errno != EINTR) {
            return std::nullopt;
        }
    }
    return std::nullopt;
}

}  // namespace

std::optional<std::vector<std::string>> parse_embedded_ptx(std::string_view output) {
    if (output.size() > kMaxToolOutputBytes || output.find('\0') != std::string_view::npos) {
        return std::nullopt;
    }
    std::vector<std::string> images;
    std::size_t cursor = 0;
    while (cursor < output.size()) {
        auto begin = output.find(".version ", cursor);
        if (begin == std::string_view::npos) {
            break;
        }
        if (begin > 0 && output[begin - 1] != '\n') {
            cursor = begin + 1;
            continue;
        }
        auto end = output.find("\nFatbin ", begin);
        if (end == std::string_view::npos) {
            end = output.size();
        }
        const auto module = output.substr(begin, end - begin);
        if (module.size() > kMaxPersistentPtxBytes || images.size() == kMaxImages) {
            return std::nullopt;
        }
        images.emplace_back(module);
        cursor = end;
    }
    return images.empty() ? std::nullopt : std::optional{std::move(images)};
}

std::optional<std::vector<std::string>> extract_embedded_ptx(const std::string& tool,
                                                             const std::string& binary,
                                                             std::chrono::milliseconds timeout) {
    if (tool.empty() || tool.front() != '/' || binary.empty() || binary.front() != '/' ||
        tool.find('\0') != std::string::npos || binary.find('\0') != std::string::npos ||
        timeout.count() <= 0) {
        return std::nullopt;
    }
    Pipe pipe;
    if (::pipe2(pipe.descriptors.data(), O_CLOEXEC | O_NONBLOCK) != 0) {
        return std::nullopt;
    }
    if (::fcntl(pipe.descriptors[1], F_SETFL, 0) < 0) {
        return std::nullopt;
    }
    SpawnActions actions;
    if (::posix_spawn_file_actions_init(&actions.actions) != 0) {
        return std::nullopt;
    }
    actions.is_initialized = true;
    if (::posix_spawn_file_actions_adddup2(&actions.actions, pipe.descriptors[1], STDOUT_FILENO) !=
            0 ||
        ::posix_spawn_file_actions_addopen(&actions.actions, STDERR_FILENO, "/dev/null", O_WRONLY,
                                           0) != 0 ||
        ::posix_spawn_file_actions_addopen(&actions.actions, STDIN_FILENO, "/dev/null", O_RDONLY,
                                           0) != 0 ||
        ::posix_spawn_file_actions_addclose(&actions.actions, pipe.descriptors[0]) != 0 ||
        ::posix_spawn_file_actions_addclose(&actions.actions, pipe.descriptors[1]) != 0) {
        return std::nullopt;
    }
    std::vector<char*> environment;
    for (auto* entry = environ; *entry != nullptr; ++entry) {
        const std::string_view value(*entry);
        if (!value.starts_with("LD_PRELOAD=") && !value.starts_with("GLIMMER_")) {
            environment.push_back(*entry);
        }
    }
    environment.push_back(nullptr);
    // posix_spawn does not modify the argv strings.
    std::array<char*, 4> arguments{const_cast<char*>(tool.c_str()), const_cast<char*>("--dump-ptx"),
                                   const_cast<char*>(binary.c_str()), nullptr};
    Child child;
    if (::posix_spawn(&child.pid, tool.c_str(), &actions.actions, nullptr, arguments.data(),
                      environment.data()) != 0) {
        child.pid = -1;
        return std::nullopt;
    }
    // On Linux close releases the descriptor even on EINTR. Do not retry a
    // potentially recycled descriptor in another application thread.
    const auto close_result = ::close(pipe.descriptors[1]);
    pipe.descriptors[1] = -1;
    if (close_result != 0) {
        return std::nullopt;
    }
    const auto output = read_output(pipe.descriptors[0], child, timeout);
    return output.has_value() ? parse_embedded_ptx(*output) : std::nullopt;
}

}  // namespace glimmer::interceptor
