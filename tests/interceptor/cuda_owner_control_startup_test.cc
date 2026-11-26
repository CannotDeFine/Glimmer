#include "internal/shared_control_channel.h"
#include "internal/socket_line_reader.h"
#include "glimmer/control/task_execution_protocol.h"

#include <cuda.h>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <poll.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

extern char** environ;
extern "C" unsigned int glimmer_owner_startup_launch_count();
extern "C" unsigned int glimmer_owner_startup_last_grid();

namespace {
template <typename T>
decltype(auto) checked(T&& value) noexcept {
    if (!value.has_value()) std::abort();
    return *value;
}

using namespace glimmer::control;
using namespace glimmer::control::internal;
using namespace std::chrono_literals;
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
std::size_t inventory(const char* path) {
    DIR* directory = ::opendir(path);
    require(directory != nullptr, "owned CPU inventory opens");
    std::size_t count{};
    errno = 0;
    while (const auto* entry = ::readdir(directory))
        if (std::strcmp(entry->d_name, ".") != 0 && std::strcmp(entry->d_name, "..") != 0) ++count;
    require(errno == 0 && ::closedir(directory) == 0, "owned CPU inventory closes");
    return count;
}
class Descriptor final {
   public:
    explicit Descriptor(int fd) : fd_(fd) {
        require(fd >= 0, "owned descriptor creates");
    }
    ~Descriptor() {
        if (::close(fd_) != 0) std::abort();
    }
    Descriptor(const Descriptor&) = delete;
    Descriptor& operator=(const Descriptor&) = delete;
    int get() const {
        return fd_;
    }

   private:
    int fd_;
};
class Listener final {
   public:
    Listener() : descriptor(::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0)) {
        std::array<char, 64> pattern{};
        constexpr std::string_view prefix = "/tmp/glimmer-owner-startup-XXXXXX";
        std::memcpy(pattern.data(), prefix.data(), prefix.size());
        require(::mkdtemp(pattern.data()) != nullptr, "owned directory creates");
        directory_ = pattern.data();
        path = directory_ + "/control.sock";
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        require(path.size() < sizeof(address.sun_path), "owned socket path fits");
        std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
        require(::bind(descriptor.get(), reinterpret_cast<const sockaddr*>(&address),
                       sizeof(address)) == 0 &&
                    ::listen(descriptor.get(), 4) == 0,
                "owned listener binds");
    }
    ~Listener() {
        if (::unlink(path.c_str()) != 0 || ::rmdir(directory_.c_str()) != 0) std::abort();
    }
    Listener(const Listener&) = delete;
    Listener& operator=(const Listener&) = delete;
    Descriptor descriptor;
    std::string path;

   private:
    std::string directory_;
};
enum class Mode : std::uint8_t { kOwner, kThread, kInvalid, kUnsupported };
struct Scenario {
    const char* name;
    Mode mode;
    std::vector<std::string> configuration;
    bool has_socket{true};
};
const char* child_mode(Mode mode) {
    switch (mode) {
        case Mode::kOwner:
            return "owner";
        case Mode::kThread:
            return "thread";
        case Mode::kInvalid:
            return "invalid";
        case Mode::kUnsupported:
            return "unsupported";
    }
    std::exit(EXIT_FAILURE);
}
int child(std::string_view mode, int exit_descriptor) {
    const Descriptor exit_signal(exit_descriptor);
    const auto threads = inventory("/proc/self/task");
    require(glimmer_owner_startup_launch_count() == 0, "fresh ordinary child has issued no kernel");
    const auto initialized = cuInit(0);
    if (mode == "owner" || mode == "thread") {
        require(initialized == CUDA_SUCCESS, "valid public selector initializes original Driver");
        require(
            inventory("/proc/self/task") == threads + (mode == "owner" ? 1U : 2U),
            "cold public owner has only process monitor; thread route keeps independent executor");
    } else if (mode == "invalid") {
        require(initialized == CUDA_ERROR_UNKNOWN,
                "invalid public startup rejects before Driver readiness");
    } else {
        require(mode == "unsupported" && initialized == CUDA_SUCCESS,
                "original Driver initialization is not authorization after unsupported control "
                "startup");
    }
    const auto launched = cuLaunchKernel(reinterpret_cast<CUfunction>(0x3000), 7, 1, 1, 32, 1, 1, 0,
                                         nullptr, nullptr, nullptr);
    std::cout << "Source result=" << static_cast<int>(launched)
              << " original launches=" << glimmer_owner_startup_launch_count() << '\n';
    if (mode == "owner" || mode == "thread") {
        require(launched == CUDA_SUCCESS && glimmer_owner_startup_launch_count() == 1 &&
                    glimmer_owner_startup_last_grid() == 7,
                "original native kernel forwards only with fresh actual control authorization");
        require(cuCtxSynchronize() == CUDA_SUCCESS,
                "ordinary application drains its real context boundary");
        // Keep this ordinary application alive while its existing actor reports
        // the actual terminal. This test-only signal cannot authorize CUDA or
        // fulfill any gate future; it only prevents premature process exit.
        pollfd pending{exit_signal.get(), POLLIN, 0};
        char byte{};
        require(::poll(&pending, 1, 3000) == 1 && (pending.revents & POLLIN) != 0 &&
                    ::recv(exit_signal.get(), &byte, 1, MSG_DONTWAIT) == 1 && byte == 'x',
                "parent observes actual terminal response before application exits");
    } else {
        require(launched != CUDA_SUCCESS && glimmer_owner_startup_launch_count() == 0,
                "failed explicit selection cannot forward original kernel through fallback");
        const auto again = cuLaunchKernel(reinterpret_cast<CUfunction>(0x3000), 9, 1, 1, 32, 1, 1,
                                          0, nullptr, nullptr, nullptr);
        require(again != CUDA_SUCCESS && glimmer_owner_startup_launch_count() == 0,
                "failed startup remains sticky on another source call");
    }
    return EXIT_SUCCESS;
}
void send_text(int fd, std::string_view line) {
    require(::send(fd, line.data(), line.size(), MSG_DONTWAIT | MSG_NOSIGNAL) ==
                static_cast<ssize_t>(line.size()),
            "actual finite handshake sends");
}
TaskExecutionProtocolRequest receive(SharedControlChannel& channel, int fd) {
    std::string line;
    require(channel.receive_request(fd, 1000, line) == SharedControlReadStatus::kLine,
            "public source publishes one actual request");
    const auto parsed = parse_task_execution_protocol_request(line);
    require(parsed.parsed(), "actual public source request parses");
    return checked(parsed.request);
}
void reply(SharedControlChannel& channel, int fd, TaskExecutionProtocolResponseKind kind,
           TaskProtocolState state = TaskProtocolState::kQueued) {
    TaskExecutionProtocolResponse response;
    response.kind = kind;
    response.task_id = 17;
    if (kind == TaskExecutionProtocolResponseKind::kExecutionLease ||
        kind == TaskExecutionProtocolResponseKind::kRetreat)
        response.generation = 29;
    response.state = state;
    response.tenant_id = "default";
    response.memory_bytes = 1;
    response.weight = 1;
    response.work_units = 1;
    const auto wire = format_task_execution_protocol_response(response);
    require(wire && (channel.sends_reply_hints() ? channel.publish_response_with_hint(fd, *wire)
                                                 : channel.publish_response(*wire)),
            "actual public response body publishes through negotiated protocol");
}
class Child final {
   public:
    explicit Child(pid_t pid) : pid_(pid) {}
    ~Child() {
        if (is_reaped_) return;
        if (::kill(pid_, SIGKILL) != 0 && errno != ESRCH) std::abort();
        int status{};
        pid_t result{};
        do {
            result = ::waitpid(pid_, &status, 0);
        } while (result < 0 && errno == EINTR);
        if (result != pid_) std::abort();
    }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    void wait() {
        int status{};
        pid_t result{};
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        do {
            result = ::waitpid(pid_, &status, WNOHANG);
            if (result == 0) {
                require(std::chrono::steady_clock::now() < deadline,
                        "owned application exits within its finite fixture bound");
                std::this_thread::sleep_for(1ms);
            }
        } while (result == 0 || (result < 0 && errno == EINTR));
        require(result == pid_, "owned ordinary application is reaped");
        is_reaped_ = true;
        require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "owned ordinary application passes");
    }

   private:
    pid_t pid_;
    bool is_reaped_{};
};
void serve(Listener& listener, Mode mode, Child& application, int exit_descriptor) {
    pollfd pending{listener.descriptor.get(), POLLIN, 0};
    require(::poll(&pending, 1, 2000) == 1 && (pending.revents & POLLIN) != 0,
            "owned application connects within cold startup bound");
    Descriptor peer(::accept4(listener.descriptor.get(), nullptr, nullptr, SOCK_CLOEXEC));
    const auto expected = mode == Mode::kThread ? kSharedControlOpen : kOwnerControlOpen;
    std::string open;
    SocketLineReader reader(peer.get());
    require(reader.read_line(open, 1000) == SocketLineStatus::kLine,
            "actual complete cold OPEN arrives within its absolute framing bound");
    require(open == expected, "default/thread uses V2; owner explicitly negotiates V3 only");
    if (mode == Mode::kUnsupported) {
        send_text(peer.get(), kOwnerControlUnsupported);
        application.wait();
        return;
    }
    send_text(peer.get(), mode == Mode::kThread ? kSharedControlReady : kOwnerControlReady);
    SharedControlChannel channel;
    require(
        channel.create_server({17, 29}, mode == Mode::kThread ? SharedControlMode::kV2
                                                              : SharedControlMode::kReplyHintV3) &&
            channel.send_offer(peer.get(), 1000ms),
        "actual sealed offer sends");
    auto request = receive(channel, peer.get());
    if (request.operation == TaskExecutionProtocolOperation::kSubmit) {
        reply(channel, peer.get(), TaskExecutionProtocolResponseKind::kAccepted);
        request = receive(channel, peer.get());
        require(request.operation == TaskExecutionProtocolOperation::kExecutionClaim &&
                    request.task_id == 17,
                "native source independently claims accepted identity");
    } else {
        require(request.operation == TaskExecutionProtocolOperation::kExecutionAcquire,
                "cooperative native reservation acquires a fresh unit execution");
    }
    reply(channel, peer.get(), TaskExecutionProtocolResponseKind::kExecutionLease);
    bool has_terminal = false;
    for (unsigned int count = 0; count < 8 && !has_terminal; ++count) {
        request = receive(channel, peer.get());
        require(request.task_id == 17 && request.generation == 29,
                "public source retains exact authorized identity/generation");
        if (request.operation == TaskExecutionProtocolOperation::kExecutionComplete) {
            reply(channel, peer.get(), TaskExecutionProtocolResponseKind::kState,
                  TaskProtocolState::kCompleted);
            has_terminal = true;
        } else {
            require(request.operation == TaskExecutionProtocolOperation::kExecutionHeartbeat,
                    "only existing heartbeat may precede actual native terminal");
            reply(channel, peer.get(), TaskExecutionProtocolResponseKind::kState,
                  TaskProtocolState::kRunning);
        }
    }
    require(has_terminal,
            "public source reports physical completion through actual control endpoint");
    send_text(exit_descriptor, "x");
    application.wait();  // Healthy service ownership survives the application boundary.
}
void scenario(const Scenario& settings) {
    std::cout << "Public selector: " << settings.name << '\n';
    Listener listener;
    std::vector<std::string> environment;
    for (char** entry = environ; *entry; ++entry)
        if (!std::string_view(*entry).starts_with("GLIMMER_")) environment.emplace_back(*entry);
    environment.emplace_back("GLIMMER_CONTROL_SPIN_US=20");
    environment.emplace_back("GLIMMER_CONTROL_TRANSPORT=shared_memory");
    environment.insert(environment.end(), settings.configuration.begin(),
                       settings.configuration.end());
    if (settings.has_socket)
        environment.emplace_back("GLIMMER_SCHEDULER_CONTROL_SOCKET=" + listener.path);
    std::vector<char*> variables;
    variables.reserve(environment.size() + 1);
    for (auto& value : environment) variables.push_back(value.data());
    variables.push_back(nullptr);
    std::array<int, 2> exit_pair{};
    require(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, exit_pair.data()) == 0,
            "test-only process-exit signal creates");
    const Descriptor parent_exit(exit_pair[0]);
    const Descriptor child_exit(exit_pair[1]);
    const auto exit_argument = std::to_string(child_exit.get());
    std::array<char*, 5> arguments{const_cast<char*>("owner-control-startup"),
                                   const_cast<char*>("--child"),
                                   const_cast<char*>(child_mode(settings.mode)),
                                   const_cast<char*>(exit_argument.c_str()), nullptr};
    const pid_t pid = ::fork();
    require(pid >= 0, "owned fresh application child creates");
    if (pid == 0) {
        if (::fcntl(child_exit.get(), F_SETFD, 0) != 0) ::_exit(126);
        ::execve("/proc/self/exe", arguments.data(), variables.data());
        ::_exit(127);
    }
    Child application(pid);
    if (settings.mode != Mode::kInvalid)
        serve(listener, settings.mode, application, parent_exit.get());
    else
        application.wait();
    pollfd pending{listener.descriptor.get(), POLLIN, 0};
    require(
        ::poll(&pending, 1, 0) == 0,
        "invalid startup never connects and unsupported/healthy startup never silently reconnects");
}
}  // namespace
int run(int argc, char** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--child") == 0)
        return child(argv[2], std::stoi(argv[3]));
    require(argc == 1, "public startup harness accepts only its own exec children");
    const auto descriptors = inventory("/proc/self/fd");
    const auto threads = inventory("/proc/self/task");
    const std::vector<std::string> valid{"GLIMMER_SCHEDULER_MODE=enforce",
                                         "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1"};
    auto owner = valid;
    owner.emplace_back("GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR=owner");
    scenario({"default", Mode::kThread, valid});
    auto independent = valid;
    independent.emplace_back("GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR=thread");
    scenario({"thread", Mode::kThread, independent});
    scenario({"owner", Mode::kOwner, owner});
    scenario({"unsupported_v3", Mode::kUnsupported, owner});
    scenario({"owner_no_socket", Mode::kInvalid, owner, false});
    for (const auto* selector : {"", "unknown", "OWNER", "thread "}) {
        auto invalid = valid;
        invalid.emplace_back(std::string("GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR=") + selector);
        scenario({"invalid_selector", Mode::kInvalid, invalid});
    }
    scenario({"owner_observe",
              Mode::kInvalid,
              {"GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR=owner", "GLIMMER_SCHEDULER_MODE=observe",
               "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1"}});
    scenario({"owner_disabled",
              Mode::kInvalid,
              {"GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR=owner", "GLIMMER_SCHEDULER_MODE=off",
               "GLIMMER_COOPERATIVE_RUNTIME_PREVIEW=1"}});
    scenario({"owner_driver_only",
              Mode::kInvalid,
              {"GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR=owner", "GLIMMER_SCHEDULER_MODE=enforce",
               "GLIMMER_COOPERATIVE_DRIVER_PREVIEW=1"}});
    scenario({"owner_no_preview",
              Mode::kInvalid,
              {"GLIMMER_EXPERIMENTAL_CONTROL_EXECUTOR=owner", "GLIMMER_SCHEDULER_MODE=enforce"}});
    require(inventory("/proc/self/fd") == descriptors && inventory("/proc/self/task") == threads,
            "ordinary parent retains no socket, mapping, child or thread resources");
    return EXIT_SUCCESS;
}
int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
