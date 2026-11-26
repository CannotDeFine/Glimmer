#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"
#include "execution_async_fixture.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fatbinary_section.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>

// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" void** __cudaRegisterFatBinary(void*);
extern "C" void __cudaRegisterFatBinaryEnd(void**);
extern "C" void __cudaUnregisterFatBinary(void**);
extern "C" void __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*,
                                       dim3*, dim3*, int*);
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
struct Signals {
    std::atomic<bool> peek_returned{}, sync_returned{}, query_held{}, query_released{},
        metadata_released{};
} signals;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void await(const std::atomic<bool>& value) {
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (!value.load(std::memory_order_acquire)) {
        expect(Clock::now() < deadline, "bounded actual transport progress");
        std::this_thread::yield();
    }
}
struct Result {
    cudaError_t peek{}, sync{}, first_error{}, after_background{};
    bool peek_before_query_release{};
};
Result application() {
    CUcontext context{};
    expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS, "create the ordinary owned context");
    alignas(8) std::array<std::byte, 24> image{};
    const std::array<std::uint64_t, 2> prefix{0x00100001ba55ed50, 8};
    std::memcpy(image.data(), prefix.data(), sizeof(prefix));
    image[16] = std::byte{7};
    __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                reinterpret_cast<const unsigned long long*>(image.data()), nullptr};
    auto** registration = __cudaRegisterFatBinary(&wrapper);
    expect(registration != nullptr, "snapshot ordinary Runtime registration");
    const int host{};
    char name[] = "kernel";
    __cudaRegisterFunction(registration, reinterpret_cast<const char*>(&host), name, name, -1,
                           nullptr, nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFatBinaryEnd(registration);
    const auto admission = cudaLaunchKernel(&host, dim3{320}, dim3{32}, nullptr, 0, nullptr);
    if (admission != cudaSuccess)
        throw std::runtime_error("ordinary remote admission status=" +
                                 std::to_string(static_cast<int>(admission)));
    Result result;
    result.peek = cudaPeekAtLastError();
    result.peek_before_query_release = !signals.query_released.load(std::memory_order_acquire);
    signals.peek_returned.store(true, std::memory_order_release);
    await(signals.query_held);
    result.sync = cudaDeviceSynchronize();
    result.first_error = cudaGetLastError();
    signals.sync_returned.store(true, std::memory_order_release);
    await(signals.metadata_released);
    result.after_background = cudaGetLastError();
    // Clear a faulty duplicate only to finish both policies' actual cleanup;
    // the immutable returned result below still rejects it after normal exit.
    (void)cudaGetLastError();
    expect(cuCtxDestroy(context) == CUDA_SUCCESS, "normal exact context cleanup");
    __cudaUnregisterFatBinary(registration);
    expect(cudaGetLastError() == cudaSuccess, "registration cleanup preserves clear TLS");
    return result;
}
void run(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    config.io_timeout_ms = 3000;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start the real authenticated CPU endpoint");
    const auto process = current_process_identity();
    expect(process.has_value(), "read the exact endpoint identity");
    const auto pid = std::to_string(process->pid);
    const auto start = std::to_string(process->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select the endpoint before any frontend entry");
    ExecutionAsyncFixture ownership;
    std::optional<ExecutionReceivedRequest> held;
    Clock::time_point held_since{};
    unsigned int launches{}, queries{}, releases{}, closes{};
    bool cleanup_fallback{};
    const auto respond = [&](const ExecutionReceivedRequest& request,
                             const ExecutionWireReply& reply) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {101, 103};
        auto encoded = encode_execution_reply(reply);
        expect(encoded && server.reply(request.connection, {header, std::move(*encoded)}),
               "deliver the exact real reply without changing its contents");
    };
    auto client = std::async(std::launch::async, application);
    const auto deadline = Clock::now() + std::chrono::seconds{12};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress real socket traffic");
        if (held && (signals.sync_returned.load(std::memory_order_acquire) ||
                     Clock::now() - held_since >= std::chrono::seconds{1})) {
            cleanup_fallback = !signals.sync_returned.load(std::memory_order_acquire);
            respond(*held, ownership.observe(*held));
            signals.query_released.store(true, std::memory_order_release);
            held.reset();
        }
        if (auto request = server.take_request()) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            switch (request->message.header.operation) {
                case ExecutionOpcode::kHello:
                    break;
                case ExecutionOpcode::kLoadBinary:
                    reply.original_status = CUDA_SUCCESS;
                    reply.id = 1;
                    break;
                case ExecutionOpcode::kGetFunction:
                    reply.id = 2;
                    break;
                case ExecutionOpcode::kLaunch: {
                    ExecutionPayloadReader input(request->message.payload);
                    ExecutionAsyncFixture::zero_argument_launch(input, 2);
                    reply = ownership.admit(request->connection, 0);
                    ownership.complete(request->connection, reply.id,
                                       CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES);
                    ++launches;
                    break;
                }
                case ExecutionOpcode::kQueryAsyncFrontiers:
                    expect(!held && queries++ == 0, "hold only the first exact status Query");
                    held_since = Clock::now();
                    held = std::move(request);
                    signals.query_held.store(true, std::memory_order_release);
                    continue;
                case ExecutionOpcode::kReleaseAsyncFrontiers:
                    reply = ownership.observe(*request);
                    ++releases;
                    break;
                case ExecutionOpcode::kCaptureFrontier:
                    reply = ownership.capture(request->connection);
                    break;
                case ExecutionOpcode::kQueryFrontier:
                case ExecutionOpcode::kReleaseFrontier:
                    reply = ownership.observe(*request);
                    break;
                case ExecutionOpcode::kClose:
                    ownership.close(request->connection);
                    ++closes;
                    break;
                default:
                    throw std::runtime_error("unexpected observer transport operation");
            }
            respond(*request, reply);
            if (request->message.header.operation == ExecutionOpcode::kReleaseAsyncFrontiers)
                signals.metadata_released.store(true, std::memory_order_release);
        }
        expect(Clock::now() < deadline, "bounded causal observation, not a latency benchmark");
    }
    const auto result = client.get();
    server.stop();
    expect(!held && launches == 1 && queries == 1 && releases == 1 && closes == 1,
           "all exact command/frontier ownership cleans up before checking policy");
    std::cout << "Async status normal cleanup: local_peek_before_release="
              << result.peek_before_query_release << "; cleanup_fallback=" << cleanup_fallback
              << "; duplicate_status=" << static_cast<int>(result.after_background) << '\n';
    expect(!cleanup_fallback && result.peek_before_query_release && result.peek == cudaSuccess,
           "local Peek and independent synchronization must finish before status reply release");
    expect(result.sync == cudaErrorLaunchOutOfResources &&
               result.first_error == cudaErrorLaunchOutOfResources &&
               result.after_background == cudaSuccess,
           "sync reports the real failure once; late cache cannot duplicate its notification");
}
}  // namespace

int main() {
    char pattern[] = "/tmp/glimmer-async-status-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
