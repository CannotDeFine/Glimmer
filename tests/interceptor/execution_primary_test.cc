#include "glimmer/control/execution_socket.h"
#include "execution_async_fixture.h"
#include "internal/process_identity.h"

#include <cuda.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_execution_lookup_count(unsigned int);
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
std::atomic<bool> waiter_observed{}, should_fail_set{}, should_fail_close{},
    should_delay_frontier{}, delayed_wait_observed{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status) {
    expect(status == CUDA_SUCCESS, "public primary operation succeeds");
}
void application() {
    void* address{};
    CUdriverProcAddressQueryResult query_status{};
    expect(cuGetProcAddress("cuDevicePrimaryCtxRetain", &address, 12000, 0, &query_status) ==
                   CUDA_SUCCESS &&
               query_status == CU_GET_PROC_ADDRESS_SUCCESS &&
               address == reinterpret_cast<void*>(&cuDevicePrimaryCtxRetain),
           "primary SDK lookup selects remote ownership");
    const auto retained = reinterpret_cast<decltype(&cuDevicePrimaryCtxRetain)>(address);
    unsigned int flags{};
    int active{};
    CUcontext primary{}, again{}, current{}, user{};
    check(cuDevicePrimaryCtxGetState(0, &flags, &active));
    expect(!flags && !active, "primary starts inactive without a native session");
    expect(cuDevicePrimaryCtxRelease(0) == CUDA_ERROR_INVALID_CONTEXT &&
               retained(reinterpret_cast<CUcontext*>(1), 0) == CUDA_ERROR_INVALID_VALUE &&
               retained(&primary, -1) == CUDA_ERROR_INVALID_DEVICE &&
               cuDevicePrimaryCtxSetFlags(0, 3) == CUDA_ERROR_INVALID_VALUE &&
               cuDevicePrimaryCtxSetFlags(0, CU_CTX_MAP_HOST) == CUDA_ERROR_INVALID_VALUE &&
               cuDevicePrimaryCtxSetFlags(0, CU_CTX_MAP_HOST | CU_CTX_SCHED_YIELD) ==
                   CUDA_ERROR_INVALID_VALUE &&
               cuDevicePrimaryCtxSetFlags(0, CU_CTX_COREDUMP_ENABLE) == CUDA_ERROR_NOT_SUPPORTED,
           "invalid inputs and unauthorized flags do not construct a remote resource");
    check(cuDevicePrimaryCtxSetFlags(0, CU_CTX_SCHED_YIELD));
    check(retained(&primary, 0));
    check(cuDevicePrimaryCtxRetain(&again, 0));
    check(cuCtxGetCurrent(&current));
    expect(primary == again && !current, "retain shares identity without pushing current");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active));
    expect(flags == CU_CTX_SCHED_YIELD && active == 1, "activation preserves configured flags");
    std::vector<std::future<void>> threads;
    for (unsigned int index = 0; index < 8; ++index)
        threads.push_back(std::async(std::launch::async, [primary, retained] {
            CUcontext observed{};
            check(retained(&observed, 0));
            expect(observed == primary, "process-wide retain identity crosses threads");
            check(cuCtxGetCurrent(&observed));
            expect(!observed, "retaining thread does not inherit another current stack");
            check(cuDevicePrimaryCtxRelease(0));
        }));
    for (auto& thread : threads) thread.get();
    check(cuCtxSetCurrent(primary));
    CUdevice device{-1};
    check(cuCtxGetDevice(&device));
    check(cuCtxGetFlags(&flags));
    expect(device == 0 && flags == (CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST),
           "primary metadata belongs to the correct remote generation");
    expect(cuDevicePrimaryCtxSetFlags(0, CU_CTX_MAP_HOST | CU_CTX_SCHED_SPIN) ==
               CUDA_ERROR_INVALID_VALUE,
           "invalid primary setter flags reject before native mutation on an active owner");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active));
    expect(flags == CU_CTX_SCHED_YIELD && active == 1,
           "rejected primary flags cannot change active metadata");
    should_fail_set.store(true, std::memory_order_release);
    expect(cuDevicePrimaryCtxSetFlags(0, CU_CTX_SCHED_SPIN) == CUDA_ERROR_NOT_SUPPORTED,
           "actual flag SDK error remains unchanged");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active));
    expect(flags == CU_CTX_SCHED_YIELD, "failed native flag mutation does not update the cache");
    check(cuDevicePrimaryCtxSetFlags(0, CU_CTX_SCHED_SPIN));
    check(cuCtxSetFlags(CU_CTX_SCHED_YIELD));
    expect(cuCtxDestroy(primary) == CUDA_ERROR_INVALID_CONTEXT,
           "user destruction cannot retire a primary identity");
    check(cuCtxCreate(&user, 0, 0));
    check(cuDevicePrimaryCtxRelease(0));
    check(cuDevicePrimaryCtxReset(0));
    check(cuCtxGetCurrent(&current));
    expect(current == user, "noncurrent primary reset preserves the caller's user context");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active));
    expect(!flags && !active, "reset clears state but retains reference accounting");
    check(cuCtxSetCurrent(primary));
    check(cuCtxGetFlags(&flags));
    expect(flags == CU_CTX_MAP_HOST, "inactive primary metadata remains addressable");
    check(retained(&again, 0));
    expect(primary == again, "reactivation preserves the primary handle, not resource epochs");
    CUmodule module{};
    CUfunction function{};
    CUdeviceptr memory{};
    check(cuModuleLoadData(&module, "owned zero-argument primary fixture"));
    check(cuModuleGetFunction(&function, module, "kernel"));
    check(cuMemAlloc(&memory, 4));
    const auto launch = [&] {
        check(cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr));
    };
    launch();
    check(cuDevicePrimaryCtxRelease(0));
    auto release = std::async(std::launch::async, [] { return cuDevicePrimaryCtxRelease(0); });
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (!waiter_observed.load(std::memory_order_acquire)) {
        expect(Clock::now() < deadline, "last release reaches its actual completion frontier");
        std::this_thread::yield();
    }
    launch();  // Required to complete the captured work; exclusion must be released.
    check(release.get());
    check(cuCtxGetCurrent(&current));
    expect(current == primary, "last release does not pop the caller's thread stack");
    check(cuDevicePrimaryCtxGetState(0, &flags, &active));
    expect(!flags && !active && cuMemFree(memory) == CUDA_ERROR_INVALID_VALUE &&
               cuModuleUnload(module) == CUDA_ERROR_INVALID_HANDLE &&
               cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) ==
                   CUDA_ERROR_INVALID_HANDLE &&
               cuDevicePrimaryCtxRelease(0) == CUDA_ERROR_INVALID_CONTEXT,
           "last release invalidates children and cannot consume a reference twice");
    check(retained(&again, 0));
    expect(again == primary, "same logical primary receives another isolated native generation");
    should_delay_frontier.store(true, std::memory_order_release);
    auto old_wait = std::async(std::launch::async, [primary] {
        check(cuCtxSetCurrent(primary));
        return cuCtxSynchronize();
    });
    const auto old_wait_deadline = Clock::now() + std::chrono::seconds{4};
    while (!delayed_wait_observed.load(std::memory_order_acquire)) {
        expect(Clock::now() < old_wait_deadline, "old wait holds a real generation-bound token");
        std::this_thread::yield();
    }
    check(cuDevicePrimaryCtxReset(0));
    expect(old_wait.get() == CUDA_ERROR_CONTEXT_IS_DESTROYED,
           "reset cannot retarget an old wait into a new native connection");
    check(retained(&again, 0));
    expect(again == primary, "reset keeps stable handle identity and outstanding references");
    check(cuDevicePrimaryCtxRelease(0));
    check(cuDevicePrimaryCtxRelease(0));
    check(cuCtxSetCurrent(user));
    check(cuCtxDestroy(user));
    check(cuCtxGetCurrent(&current));
    expect(current == primary, "destroying the user restores the unchanged underlying stack");
    check(cuCtxSetCurrent(nullptr));
    const auto child = ::fork();
    expect(child >= 0, "fork primary ownership fixture");
    if (child == 0) _exit(retained(&again, 0) == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int child_status{};
    expect(::waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
               WEXITSTATUS(child_status) == 0,
           "fork rejects before copied guards, locks and SDK lookup");
    check(retained(&again, 0));
    check(cuCtxSetCurrent(again));
    should_fail_close.store(true, std::memory_order_release);
    expect(cuDevicePrimaryCtxRelease(0) == CUDA_ERROR_UNKNOWN &&
               retained(&again, 0) == CUDA_ERROR_UNKNOWN &&
               cuDevicePrimaryCtxReset(0) == CUDA_ERROR_UNKNOWN &&
               cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_ERROR_UNKNOWN &&
               cuCtxGetCurrent(&current) == CUDA_ERROR_UNKNOWN,
           "uncertain teardown retains ownership, never reconnects or fabricates inactivity");
    expect(glimmer_execution_lookup_count(2) == 0,
           "no primary/resource call falls back to the caller's native Driver");
}

void run(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated primary endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted endpoint identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select the actual owned endpoint before public operations");
    std::jthread watchdog([](const std::stop_token& stop) {
        const auto deadline = Clock::now() + std::chrono::seconds{12};
        while (!stop.stop_requested()) {
            if (Clock::now() >= deadline) _exit(97);
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    });
    auto client = std::async(std::launch::async, application);
    std::map<std::uint64_t, unsigned int> context_flags;
    ExecutionAsyncFixture asynchronous;
    std::set<std::uint64_t> captured_frontiers;
    unsigned int hellos{}, closes{}, launches{}, captures{}, releases{}, abandoned{};
    std::uint64_t primary_activations{};
    std::uint64_t delayed_frontier{}, delayed_connection{};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress authenticated primary traffic");
        const auto request = server.take_request();
        if (!request) continue;
        ExecutionPayloadReader input(request->message.payload);
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.original_status = 0;
        auto& flags = context_flags[request->connection];
        switch (request->message.header.operation) {
            case ExecutionOpcode::kHello: {
                ++hellos;
                const auto descriptor =
                    decode_execution_context_bootstrap(request->message.payload);
                const bool is_user = hellos == 2;
                expect(descriptor &&
                           descriptor->purpose ==
                               (is_user ? ExecutionContextPurpose::kExplicitUser
                                        : ExecutionContextPurpose::kProcessPrimary) &&
                           descriptor->source_generation ==
                               (is_user ? 1 : 2 * primary_activations++ + 1),
                       "cold factories distinguish explicit user and exact primary generations");
                flags = descriptor->flags;
                break;
            }
            case ExecutionOpcode::kGetContextFlags:
                reply.id = flags;
                break;
            case ExecutionOpcode::kSetContextFlags:
                if (should_fail_set.exchange(false, std::memory_order_acq_rel)) {
                    reply.status = CUDA_ERROR_NOT_SUPPORTED;
                    reply.original_status = reply.status;
                } else
                    expect(input.read_u32(flags) && input.empty(), "exact native flag mutation");
                break;
            case ExecutionOpcode::kClose:
                ++closes;
                if (delayed_frontier && delayed_connection == request->connection) {
                    asynchronous.abandon(request->connection, delayed_frontier);
                    captured_frontiers.erase(delayed_frontier);
                    ++abandoned;
                    delayed_frontier = 0;
                }
                asynchronous.close(request->connection);
                if (should_fail_close.load(std::memory_order_acquire)) {
                    reply.status = CUDA_ERROR_UNKNOWN;
                    reply.original_status = reply.status;
                }
                break;
            case ExecutionOpcode::kLoadPtx:
            case ExecutionOpcode::kGetFunction:
                reply.id = 3;
                break;
            case ExecutionOpcode::kAllocate:
                reply.id = 3;
                reply.address = 0x200000;
                break;
            case ExecutionOpcode::kLaunch:
                ExecutionAsyncFixture::zero_argument_launch(input, 3);
                ++launches;
                reply = asynchronous.admit(request->connection, 0);
                expect(reply.id == launches, "both accepted primary commands share one generation");
                if (launches == 2) {
                    asynchronous.complete(request->connection, 1, CUDA_SUCCESS);
                    asynchronous.complete(request->connection, 2, CUDA_SUCCESS);
                }
                break;
            case ExecutionOpcode::kCaptureFrontier: {
                std::uint32_t scope{};
                std::uint64_t stream{};
                expect(input.read_u32(scope) && scope == 0 && input.read_u64(stream) &&
                           stream == 0 && input.empty(),
                       "primary waits capture an exact context prefix");
                ++captures;
                reply = asynchronous.capture(request->connection);
                captured_frontiers.insert(reply.id);
                if (should_delay_frontier.exchange(false, std::memory_order_acq_rel)) {
                    delayed_frontier = reply.id;
                    delayed_connection = request->connection;
                }
                break;
            }
            case ExecutionOpcode::kQueryAsyncFrontiers:
            case ExecutionOpcode::kReleaseAsyncFrontiers:
                reply = asynchronous.observe(*request);
                if (request->message.header.operation == ExecutionOpcode::kReleaseAsyncFrontiers) {
                    const auto released = decode_execution_async_frontier_batch(reply.data);
                    expect(released.has_value(), "complete released primary batch identity");
                    releases += static_cast<unsigned int>(released->size());
                }
                break;
            case ExecutionOpcode::kQueryFrontier:
            case ExecutionOpcode::kReleaseFrontier: {
                std::uint64_t id{};
                expect(input.read_u64(id) && input.empty(), "exact primary frontier identity");
                reply = asynchronous.observe(*request);
                if (request->message.header.operation == ExecutionOpcode::kReleaseFrontier) {
                    captured_frontiers.erase(id);
                    ++releases;
                } else if (id == delayed_frontier) {
                    reply.status = CUDA_ERROR_NOT_READY;
                    reply.parameter_bytes = 0;
                    delayed_wait_observed.store(true, std::memory_order_release);
                } else if (launches == 1 && captured_frontiers.contains(id)) {
                    waiter_observed.store(true, std::memory_order_release);
                }
                break;
            }
            default:
                throw std::runtime_error("unexpected public primary operation");
        }
        auto header = request->message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request->connection;
        header.incarnation = {17, 19};
        auto payload = encode_execution_reply(reply);
        expect(payload && server.reply(request->connection, {header, std::move(*payload)}),
               "return the exact original/preparation result");
    }
    client.get();
    watchdog.request_stop();
    expect(hellos == 6 && primary_activations == 5 && closes == 6 && launches == 2 &&
               captures >= 7 && releases + abandoned == captures + launches && abandoned == 1 &&
               asynchronous.empty() && captured_frontiers.empty(),
           "retains share native generations; stable last-release cleanup executes once");
    server.stop();
    std::cout << "Public process-primary ownership verified: stable_identity=true; "
                 "noncurrent_reset=true; concurrent_release=true; native_caller_calls=0\n";
}
}  // namespace
int main() {
    char pattern[] = "/tmp/glimmer-primary-XXXXXX";
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
