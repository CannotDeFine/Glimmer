#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr std::string_view kVariable = "GLIMMER_EXECUTION_PREPARATION_TIMEOUT_MS";
constexpr char kPtx[] =
    ".version 7.0\n.target sm_70\n.address_size 64\n.visible .entry kernel(){ret;}\n";
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Directory {
    std::string path;
    Directory() {
        char pattern[] = "/tmp/glimmer-preparation-config-XXXXXX";
        const auto* created = ::mkdtemp(pattern);
        expect(created, "create private CPU fixture directory");
        path = created;
    }
    ~Directory() {
        if (::rmdir(path.c_str()) != 0) std::cerr << "Preparation fixture directory not empty\n";
    }
};
void exercise(bool is_valid, bool has_short_deadline) {
    CUcontext context{};
    const auto created = cuCtxCreate(&context, 0, 0);
    if (!is_valid) {
        expect(created == CUDA_ERROR_NOT_INITIALIZED && !context,
               "invalid present preparation configuration fails closed without output or IPC");
        return;
    }
    expect(created == CUDA_SUCCESS && context, "valid policy preserves ordinary bootstrap");
    if (has_short_deadline) {
        CUmodule module{};
        expect(cuModuleLoadData(&module, kPtx) == CUDA_ERROR_UNKNOWN && !module,
               "frontend applies the explicitly short cold-load deadline");
        expect(cuModuleLoadData(&module, kPtx) == CUDA_ERROR_INVALID_CONTEXT && !module &&
                   cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN,
               "expired owned preparation remains uncertain and cannot reconnect or replay");
    } else {
        expect(cuCtxDestroy(context) == CUDA_SUCCESS,
               "valid finite preparation policy does not change ordinary close");
    }
}
void run_case(std::string_view value, bool is_valid, bool has_short_deadline) {
    Directory directory;
    const auto path = directory.path + "/service.sock";
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated CPU-only endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "obtain exact expected server identity");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure frontend before its first owned call");
    if (value == "unset")
        expect(::unsetenv(kVariable.data()) == 0, "preserve unset legacy policy");
    else
        expect(::setenv(kVariable.data(), std::string(value).c_str(), 1) == 0,
               "set exact independent configuration case");
    auto client = std::async(std::launch::async, [=] { exercise(is_valid, has_short_deadline); });
    unsigned int hellos{}, loads{}, closes{};
    std::optional<ExecutionReceivedRequest> delayed;
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    try {
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(Clock::now() < deadline && server.poll_for(1), "bounded frontend CPU fixture");
            auto request = server.take_request();
            if (!request) continue;
            if (request->message.header.operation == ExecutionOpcode::kLoadPtx) {
                expect(!delayed, "an unresolved preparation request is never replayed");
                ++loads;
                delayed = std::move(request);
                continue;
            }
            if (request->message.header.operation == ExecutionOpcode::kHello)
                ++hellos;
            else {
                expect(request->message.header.operation == ExecutionOpcode::kClose,
                       "only bootstrap, one cold load and ordinary close can be issued");
                ++closes;
            }
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = CUDA_SUCCESS;
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {41, 43};
            const auto payload = encode_execution_reply(reply);
            expect(payload && server.reply(request->connection, {header, *payload}),
                   "return the exact authenticated fake original result");
        }
        client.get();
        expect(server.poll_for(0) && !server.take_request(), "no hidden extra owned request");
        expect(hellos == (is_valid ? 1U : 0U) && loads == (has_short_deadline ? 1U : 0U) &&
                   closes == (is_valid && !has_short_deadline ? 1U : 0U),
               "configuration rejection and expiry create no duplicate operation or resource");
    } catch (...) {
        server.stop();
        throw;
    }
    server.stop();
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--case") {
            const auto kind = std::string_view(argv[3]);
            expect(kind == "valid" || kind == "invalid" || kind == "short", "known fixture case");
            run_case(argv[2], kind != "invalid", kind == "short");
            return EXIT_SUCCESS;
        }
        expect(argc == 1, "unknown preparation fixture argument");
        constexpr std::array cases{std::pair{"unset", "valid"},
                                   std::pair{"1", "valid"},
                                   std::pair{"600000", "valid"},
                                   std::pair{"00001000", "valid"},
                                   std::pair{"", "invalid"},
                                   std::pair{"0", "invalid"},
                                   std::pair{"600001", "invalid"},
                                   std::pair{"-1", "invalid"},
                                   std::pair{"+1", "invalid"},
                                   std::pair{" 1", "invalid"},
                                   std::pair{"1 ", "invalid"},
                                   std::pair{"1x", "invalid"},
                                   std::pair{"18446744073709551616", "invalid"},
                                   std::pair{"000000000000000000000000000000001", "invalid"},
                                   std::pair{"500", "short"}};
        for (const auto& [value, kind] : cases) {
            const auto child = ::fork();
            expect(child >= 0, "fork independent fresh frontend configuration process");
            if (!child) {
                ::execl("/proc/self/exe", argv[0], "--case", value, kind, nullptr);
                _exit(127);
            }
            int status{};
            expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                       WEXITSTATUS(status) == 0,
                   "fresh process preserves strict configuration and one-way uncertainty");
        }
        std::cout << "Preparation configuration checks passed in15 fresh CPU-only processes\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
