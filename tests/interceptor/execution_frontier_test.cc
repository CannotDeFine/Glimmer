#include "glimmer/control/execution_socket.h"
#include "execution_async_fixture.h"
#include "internal/process_identity.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult result) {
    expect(result == CUDA_SUCCESS, "public Driver operation succeeds");
}
std::atomic<bool> waiter_observed{}, first_wait_done{};
void application() {
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    CUmodule module{};
    check(cuModuleLoadData(&module, "owned zero-parameter fixture"));
    CUfunction function{};
    check(cuModuleGetFunction(&function, module, "kernel"));
    const auto launch = [&] {
        check(cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr));
    };
    launch();
    auto waiter = std::async(std::launch::async, [context] {
        check(cuCtxSetCurrent(context));
        return cuCtxSynchronize();
    });
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    while (!waiter_observed.load(std::memory_order_acquire)) {
        expect(Clock::now() < deadline, "waiter reaches the real transport");
        std::this_thread::yield();
    }
    // This command is required for the first one to finish. A process-wide
    // blocking drain prevents this call and creates the old failure.
    launch();
    expect(
        waiter.get() == CUDA_SUCCESS,
        "legacy blocking drain prevents the releasing submission while its waiter owns exclusion");
    first_wait_done.store(true, std::memory_order_release);
    // The second command remains physically pending until this third command.
    // Finishing the first wait before here proves future commands stayed out.
    launch();
    check(cuCtxSynchronize());
    check(cuModuleUnload(module));
    check(cuCtxDestroy(context));
}
void run(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated liveness endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted owner identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select only this owned endpoint");
    ExecutionAsyncFixture asynchronous;
    unsigned int launches{}, captures{}, releases{}, queries{};
    bool has_legacy_drain{}, completed_while_future_pending{};
    std::optional<ExecutionReceivedRequest> legacy;
    Clock::time_point legacy_deadline;
    const auto reply_to = [&](const ExecutionReceivedRequest& request,
                              const ExecutionWireReply& reply) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {41, 43};
        auto encoded = encode_execution_reply(reply);
        expect(encoded && server.reply(request.connection, {header, std::move(*encoded)}),
               "reply once to exact request");
    };
    auto client = std::async(std::launch::async, application);
    const auto deadline = Clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress authenticated concurrent submissions");
        if (legacy && Clock::now() >= legacy_deadline) {
            expect(launches == 1, "old waiter exclusion blocks the releasing original");
            std::cerr << "Legacy drain watchdog: only_one_original=true; "
                         "releasing_submission_blocked=true\n";
            ExecutionWireReply failure;
            failure.disposition = ExecutionWireDisposition::kComplete;
            failure.status = CUDA_ERROR_UNKNOWN;
            reply_to(*legacy, failure);
            legacy.reset();
        }
        if (auto request = server.take_request()) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = 0;
            ExecutionPayloadReader input(request->message.payload);
            switch (request->message.header.operation) {
                case ExecutionOpcode::kLoadPtx:
                    reply.id = 1;
                    break;
                case ExecutionOpcode::kGetFunction:
                    reply.id = 2;
                    break;
                case ExecutionOpcode::kLaunch: {
                    ExecutionAsyncFixture::zero_argument_launch(input, 2);
                    ++launches;
                    reply = asynchronous.admit(request->connection, 0);
                    expect(reply.id == launches, "admission sequence is an exact command identity");
                    if (launches == 2) asynchronous.complete(request->connection, 1, CUDA_SUCCESS);
                    if (launches == 3) {
                        completed_while_future_pending =
                            first_wait_done.load(std::memory_order_acquire);
                        asynchronous.complete(request->connection, 2, CUDA_SUCCESS);
                        asynchronous.complete(request->connection, 3, CUDA_SUCCESS);
                    }
                    break;
                }
                case ExecutionOpcode::kDrain:
                    has_legacy_drain = true;
                    legacy = std::move(*request);
                    legacy_deadline = Clock::now() + std::chrono::milliseconds{150};
                    waiter_observed.store(true, std::memory_order_release);
                    continue;
                case ExecutionOpcode::kCaptureFrontier: {
                    std::uint32_t scope{};
                    std::uint64_t stream{};
                    expect(input.read_u32(scope) && scope == 0 && input.read_u64(stream) &&
                               stream == 0 && input.empty(),
                           "context capture has explicit scope and exact payload");
                    ++captures;
                    reply = asynchronous.capture(request->connection);
                    break;
                }
                case ExecutionOpcode::kQueryFrontier:
                case ExecutionOpcode::kReleaseFrontier:
                case ExecutionOpcode::kQueryAsyncFrontiers:
                case ExecutionOpcode::kReleaseAsyncFrontiers: {
                    reply = asynchronous.observe(*request);
                    if (request->message.header.operation ==
                        ExecutionOpcode::kReleaseAsyncFrontiers) {
                        const auto released = decode_execution_async_frontier_batch(reply.data);
                        expect(released.has_value(), "complete released frontier batch identity");
                        releases += static_cast<unsigned int>(released->size());
                    } else if (request->message.header.operation ==
                               ExecutionOpcode::kReleaseFrontier) {
                        ++releases;
                    } else {
                        ++queries;
                        // Admission polling alone must not release the application:
                        // it waits until the first actual immutable wait exists.
                        if (captures != 0) waiter_observed.store(true, std::memory_order_release);
                    }
                    break;
                }
                case ExecutionOpcode::kClose:
                    asynchronous.close(request->connection);
                    break;
                default:
                    break;
            }
            reply_to(*request, reply);
        }
        expect(Clock::now() < deadline, "bounded no-GPU liveness proof");
    }
    client.get();
    expect(!has_legacy_drain && completed_while_future_pending && launches == 3 && captures == 2 &&
               queries >= 2 && releases == captures + launches && asynchronous.empty(),
           "immutable wait completes before future work, without leaked capabilities or original "
           "replay");
    server.stop();
    std::cout
        << "Concurrent public frontier verified: releasing_submission=true; immutable_prefix=true; "
           "legacy_drain=false; actual_gpu_qualified=false\n";
}
}  // namespace
int main() {
    char pattern[] = "/tmp/glimmer-frontier-XXXXXX";
    const auto directory = ::mkdtemp(pattern);
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
