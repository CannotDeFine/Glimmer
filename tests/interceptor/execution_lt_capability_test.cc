#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cublasLt.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_lt_native_calls();
namespace {
using namespace glimmer::control;
using namespace std::chrono_literals;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct EndpointDirectory {
    const char* path{};
    ~EndpointDirectory() {
        if (path && ::rmdir(path) != 0)
            std::fprintf(stderr, "Capability fixture directory cleanup failed: %s\n", path);
    }
    EndpointDirectory(const EndpointDirectory&) = delete;
    EndpointDirectory& operator=(const EndpointDirectory&) = delete;
    explicit EndpointDirectory(const char* value) : path(value) {}
    bool close() noexcept {
        if (::rmdir(path) != 0) return false;
        path = nullptr;
        return true;
    }
};
void client(unsigned int scenario, std::atomic<bool>& pending) {
    cublasLtMatmulAlgo_t algorithm{};
    algorithm.data[0] = 77;
    const auto get = reinterpret_cast<decltype(&cublasLtMatmulAlgoCapGetAttribute)>(
        ::dlsym(RTLD_DEFAULT, "cublasLtMatmulAlgoCapGetAttribute"));
    expect(get == &cublasLtMatmulAlgoCapGetAttribute,
           "cached capability remains the public wrapper");
    struct Output {
        std::uint32_t before{17}, value{19}, after{23};
        std::size_t written{29};
    } output;
    const auto call = [&](Output& value) {
        return get(&algorithm, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, &value.value,
                   sizeof(value.value), &value.written);
    };
    if (scenario == 8) {
        alignas(std::size_t) std::array<std::byte, 16> alias{};
        expect(get(&algorithm, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, alias.data(), 8,
                   reinterpret_cast<std::size_t*>(alias.data())) == CUBLAS_STATUS_NOT_SUPPORTED &&
                   get(nullptr, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, nullptr, 4097, nullptr) ==
                       CUBLAS_STATUS_NOT_SUPPORTED,
               "unsupported aliases/oversize reject without reading input or opening metadata");
        return;
    }
    if (scenario == 9) {
        auto first = std::async(std::launch::async, [&] { return call(output); });
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (!pending.load()) {
            expect(std::chrono::steady_clock::now() < deadline, "bounded first metadata request");
            std::this_thread::yield();
        }
        expect(first.wait_for(0ms) == std::future_status::timeout,
               "first original observation is held");
        Output other;
        expect(call(other) == CUBLAS_STATUS_SUCCESS && first.get() == CUBLAS_STATUS_SUCCESS &&
                   output.value == 3 && other.value == 3 && output.written == 4 &&
                   other.written == 4,
               "withheld reply permits independent concurrent capability progress");
        return;
    }
    const auto status = call(output);
    if (scenario) {
        expect(status == CUBLAS_STATUS_INTERNAL_ERROR && output.value == 19 &&
                   output.written == 29 && output.before == 17 && output.after == 23,
               "malformed/lost receipt cannot publish any output");
        expect(call(output) == CUBLAS_STATUS_NOT_INITIALIZED,
               "uncertain endpoint cannot retry the original metadata query");
        return;
    }
    expect(status == CUBLAS_STATUS_SUCCESS && output.value == 3 && output.written == 4 &&
               output.before == 17 && output.after == 23,
           "exact native success and output guards");
    expect(call(output) == CUBLAS_STATUS_INVALID_VALUE && output.value == 0xeeeeeeeeU &&
               output.written == 99 && output.before == 17 && output.after == 23,
           "original error-side output effects are not discarded");
    expect(get(&algorithm, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK, nullptr, 0, &output.written) ==
                   CUBLAS_STATUS_SUCCESS &&
               output.written == 4,
           "size-only query preserves presence and original required byte count");
}
void run(unsigned int scenario) {
    char pattern[] = "/tmp/glimmer-lt-capability-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    expect(directory, "private endpoint directory");
    EndpointDirectory directory_owner(directory);
    ExecutionSocketConfig configuration;
    configuration.socket_path = std::string(directory) + "/service.sock";
    configuration.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(configuration));
    expect(server.start(), "start actual authenticated no-GPU endpoint");
    const auto identity = current_process_identity();
    expect(identity &&
               ::setenv("GLIMMER_EXECUTION_SOCKET",
                        (std::string(directory) + "/service.sock").c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(identity->pid).c_str(), 1) ==
                   0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START",
                        std::to_string(identity->start_time_ticks).c_str(), 1) == 0,
           "bind exact process-owned endpoint");
    std::atomic<bool> pending{};
    auto application = std::async(std::launch::async, [&] { client(scenario, pending); });
    std::optional<ExecutionReceivedRequest> held;
    unsigned int queries{};
    const auto publish = [&](const ExecutionReceivedRequest& request,
                             const ExecutionWireReply& reply) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {173, 179};
        const auto bytes = encode_execution_reply(reply);
        expect(bytes && server.reply(request.connection, {header, *bytes}),
               "publish authenticated reply");
    };
    const auto deadline = std::chrono::steady_clock::now() + 15s;
    try {
        while (application.wait_for(0ms) != std::future_status::ready) {
            expect(server.poll_for(1), "progress actual socket");
            if (auto request = server.take_request()) {
                ExecutionPayloadReader input(request->message.payload);
                ExecutionWireReply reply;
                reply.disposition = ExecutionWireDisposition::kComplete;
                if (request->message.header.operation == ExecutionOpcode::kHello) {
                    std::uint32_t role{}, flags{};
                    expect(input.read_u32(role) && role == kExecutionLtMetadataRole &&
                               input.read_u32(flags) && !flags && input.empty(),
                           "CPU metadata-only hello");
                    const auto ownership = encode_execution_bootstrap_ownership(
                        ExecutionBootstrapOwnership::kMetadataRetained);
                    reply.data.assign(ownership->begin(), ownership->end());
                } else {
                    expect(request->message.header.operation ==
                               ExecutionOpcode::kLtAlgorithmCapability,
                           "no GPU or resource command in capability fixture");
                    std::span<const std::byte> algorithm, initial;
                    std::uint32_t attribute{}, presence{};
                    std::uint64_t bytes{}, written{};
                    expect(input.read_blob(algorithm) && algorithm.size() == 64 &&
                               input.read_u32(attribute) && attribute == 11 &&
                               input.read_u64(bytes) && input.read_u32(presence) &&
                               input.read_blob(initial) && input.read_u64(written) &&
                               input.empty() &&
                               ((bytes == 4 && presence == 3 && initial.size() == 4) ||
                                (bytes == 0 && presence == 2 && initial.empty())),
                           "exact bounded capability frame");
                    ++queries;
                    reply.original_status = 0;
                    reply.address = 4;
                    reply.data.assign(initial.begin(), initial.end());
                    if (bytes) {
                        const std::uint32_t value = scenario == 0 && queries == 2 ? 0xeeeeeeeeU : 3;
                        std::memcpy(reply.data.data(), &value, sizeof(value));
                    }
                    if (!scenario && queries == 2) {
                        reply.original_status = reply.status = 7;
                        reply.address = 99;
                    }
                    if (scenario == 1) reply.original_status.reset();
                    if (scenario == 2) reply.data.pop_back();
                    if (scenario == 3) reply.id = 1;
                    if (scenario == 4) reply.status = 7;
                    if (scenario == 5) reply.address = 5;
                    if (scenario == 6) reply.parameter_bytes = 2;
                    if (scenario == 7) {
                        server.close_peer(request->connection);
                        continue;
                    }
                    if (scenario == 9 && queries == 1) {
                        held = std::move(request);
                        pending.store(true);
                        continue;
                    }
                    if (scenario == 10) reply.original_status = reply.status = -1;
                    if (scenario == 11) {
                        reply.disposition = ExecutionWireDisposition::kRejected;
                        reply.original_status.reset();
                        reply.status = -1;
                    }
                    if (scenario == 12) reply.original_status = reply.status = 32;
                }
                publish(*request, reply);
                if (held) {
                    publish(*held, reply);
                    held.reset();
                }
            }
            expect(std::chrono::steady_clock::now() < deadline, "bounded capability receipt test");
        }
        application.get();
        expect(queries == (scenario == 0   ? 3U
                           : scenario == 8 ? 0U
                           : scenario == 9 ? 2U
                                           : 1U),
               "every actual query occurs exactly once; no replay after quarantine");
    } catch (...) {
        server.stop();
        throw;
    }
    server.stop();
    expect(directory_owner.close() && glimmer_lt_native_calls() == 0,
           "normal endpoint cleanup and no caller-native capability consumer");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            for (unsigned int scenario = 0; scenario <= 12; ++scenario) {
                const auto number = std::to_string(scenario);
                const auto child = ::fork();
                expect(child >= 0, "fresh process before frontend initialization");
                if (!child) {
                    ::execl(argv[0], argv[0], "--case", number.c_str(), nullptr);
                    ::_exit(127);
                }
                int status{};
                expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                           WEXITSTATUS(status) == 0,
                       "isolated capability scenario failed");
            }
        } else {
            expect(argc == 3 && std::string_view(argv[1]) == "--case", "explicit bounded case");
            const auto scenario = std::stoul(argv[2]);
            expect(scenario <= 12, "known capability case");
            run(static_cast<unsigned int>(scenario));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
