#include "glimmer/control/execution_socket.h"
#include "execution_async_fixture.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fatbinary_section.h>

#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
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
std::atomic<bool> pending_observed{}, close_observed{};
std::atomic<bool> runtime_open_observed{}, runtime_open_release{}, runtime_load_observed{},
    runtime_load_release{};
std::atomic<bool> runtime_race_observed{}, runtime_race_release{}, runtime_unregister_entered{};
std::atomic<bool> free_observed{};
std::atomic<bool> retirement_query_observed{}, retirement_seal_observed{};
void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Submission fixture invariant: " << message << '\n';
        throw std::runtime_error(message);
    }
}
void check(CUresult result) {
    expect(result == CUDA_SUCCESS, "public Driver operation succeeds");
}
void wait_for(const std::atomic<bool>& condition) {
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (!condition.load(std::memory_order_acquire)) {
        expect(Clock::now() < deadline, "bounded actual endpoint observation");
        std::this_thread::yield();
    }
}
void runtime_application() {
    alignas(8) std::array<std::byte, 24> image{};
    const std::array<std::uint64_t, 2> prefix{0x00100001ba55ed50, 8};
    std::memcpy(image.data(), prefix.data(), sizeof(prefix));
    image[16] = std::byte{7};
    __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                reinterpret_cast<const unsigned long long*>(image.data()), nullptr};
    auto** registration = __cudaRegisterFatBinary(&wrapper);
    expect(registration, "register an owned Runtime image before concurrent initialization");
    const int host_identity{};
    const auto* host = reinterpret_cast<const char*>(&host_identity);
    char name[] = "kernel";
    __cudaRegisterFunction(registration, host, name, name, -1, nullptr, nullptr, nullptr, nullptr,
                           nullptr);
    __cudaRegisterFatBinaryEnd(registration);
    image[16] = std::byte{99};
    std::barrier start(9);
    std::vector<std::future<cudaError_t>> callers;
    for (unsigned int index = 0; index != 8; ++index)
        callers.push_back(std::async(std::launch::async, [&] {
            start.arrive_and_wait();
            return cudaLaunchKernel(host, dim3{320}, dim3{32}, nullptr, 0, nullptr);
        }));
    start.arrive_and_wait();
    wait_for(runtime_open_observed);
    std::vector<void**> images;
    images.push_back(__cudaRegisterFatBinary(&wrapper));
    expect(images.back(), "independent metadata does not wait for native bootstrap reply");
    runtime_open_release.store(true, std::memory_order_release);
    wait_for(runtime_load_observed);
    // Reallocate the process image vector while its first owned loader waits.
    for (unsigned int index = 0; index != 64; ++index) {
        auto** extra = __cudaRegisterFatBinary(&wrapper);
        expect(extra, "independent metadata remains available during module preparation");
        images.push_back(extra);
    }
    runtime_load_release.store(true, std::memory_order_release);
    for (auto& caller : callers)
        expect(caller.get() == cudaSuccess,
               "concurrent cold callers share one prepared Runtime binding");
    const int second_identity{};
    const auto* second_host = reinterpret_cast<const char*>(&second_identity);
    __cudaRegisterFunction(images.front(), second_host, name, name, -1, nullptr, nullptr, nullptr,
                           nullptr, nullptr);
    __cudaRegisterFatBinaryEnd(images.front());
    auto raced = std::async(std::launch::async, [&] {
        return cudaLaunchKernel(second_host, dim3{320}, dim3{32}, nullptr, 0, nullptr);
    });
    wait_for(runtime_race_observed);
    auto unregistering = std::async(std::launch::async, [&] {
        runtime_unregister_entered.store(true, std::memory_order_release);
        __cudaUnregisterFatBinary(images.front());
        return cudaGetLastError();
    });
    wait_for(runtime_unregister_entered);
    const auto sealed = cudaLaunchKernel(second_host, dim3{320}, dim3{32}, nullptr, 0, nullptr);
    (void)cudaGetLastError();
    runtime_race_release.store(true, std::memory_order_release);
    const auto raced_result = raced.get(), unregister_result = unregistering.get();
    expect(sealed == cudaErrorInvalidDeviceFunction || sealed == cudaErrorInvalidResourceHandle,
           "unregister seals new lookups without clearing an in-flight loader's owned module");
    expect(raced_result == cudaErrorInvalidResourceHandle && unregister_result == cudaSuccess,
           "pending loader publishes its owned module before once-only unregister cleanup");
    for (std::size_t index = 1; index < images.size(); ++index)
        __cudaUnregisterFatBinary(images[index]);
    __cudaUnregisterFatBinary(registration);
    expect(cudaGetLastError() == cudaSuccess && cudaDeviceReset() == cudaSuccess,
           "Runtime image/primary retires once after all concurrent SDK results");
}
void application() {
    CUcontext context{}, other{};
    CUmodule module{}, other_module{};
    CUfunction function{}, other_function{};
    check(cuCtxCreate(&context, 0, 0));
    check(cuModuleLoadData(&module, "zero-argument source"));
    check(cuModuleGetFunction(&function, module, "kernel"));
    check(cuCtxCreate(&other, 0, 0));
    check(cuModuleLoadData(&other_module, "other zero-argument source"));
    check(cuModuleGetFunction(&other_function, other_module, "kernel"));
    check(cuCtxSetCurrent(context));
    const auto launch = [](CUfunction kernel) {
        return cuLaunchKernel(kernel, 320, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr);
    };
    check(launch(function));
    auto dependent = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return launch(function);
    });
    wait_for(pending_observed);
    auto destroyer = std::async(std::launch::async, [&] { return cuCtxDestroy(context); });
    check(launch(function));
    expect(dependent.get() == CUDA_ERROR_INVALID_VALUE,
           "deferred pre-admission rejection survives independent submission exactly once");
    wait_for(close_observed);
    expect(cuCtxDestroy(context) == CUDA_ERROR_CONTEXT_IS_DESTROYED &&
               launch(function) == CUDA_ERROR_CONTEXT_IS_DESTROYED,
           "exclusive close rejects duplicate teardown and new issue without another RPC");
    check(cuCtxSetCurrent(other));
    check(launch(other_function));
    check(destroyer.get());
    expect(cuCtxSetCurrent(context) == CUDA_ERROR_INVALID_CONTEXT,
           "completed close invalidates original context without reviving stale handles");
    CUdeviceptr old_pointer{}, new_pointer{};
    check(cuMemAlloc(&old_pointer, 64));
    auto freeing = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(other));
        return cuMemFree(old_pointer);
    });
    wait_for(free_observed);
    expect(cuMemFree(old_pointer) == CUDA_ERROR_INVALID_HANDLE,
           "a pending free cannot submit the same native destruction again");
    auto allocating = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(other));
        return cuMemAlloc(&new_pointer, 64);
    });
    check(allocating.get());
    check(freeing.get());
    expect(new_pointer == old_pointer,
           "legitimate native address reuse survives reversed host result delivery");
    check(cuMemFree(new_pointer));
    check(cuModuleUnload(other_module));
    check(cuCtxDestroy(other));
    check(cuCtxSetCurrent(nullptr));  // Remove the destroyed lower context-stack frame.
    runtime_application();
    CUcontext primary{};
    check(cuDevicePrimaryCtxRetain(&primary, 0));
    check(cuCtxSetCurrent(primary));
    auto old_wait = std::async(std::launch::async, [primary] {
        check(cuCtxSetCurrent(primary));
        return cuCtxSynchronize();
    });
    wait_for(retirement_query_observed);
    auto reset = std::async(std::launch::async, [] { return cuDevicePrimaryCtxReset(0); });
    const auto seal_deadline = Clock::now() + std::chrono::seconds{2};
    for (;;) {
        const auto observed = cuStreamQuery(nullptr);
        if (observed == CUDA_ERROR_CONTEXT_IS_DESTROYED) break;
        check(observed);
        expect(Clock::now() < seal_deadline, "primary close seals before pending metadata replies");
        std::this_thread::yield();
    }
    retirement_seal_observed.store(true, std::memory_order_release);
    check(reset.get());
    expect(old_wait.get() == CUDA_ERROR_CONTEXT_IS_DESTROYED,
           "an old pending query drains without poisoning or retargeting the closing context");
    check(cuDevicePrimaryCtxRelease(0));
    check(cuCtxSetCurrent(nullptr));
}
void run(const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start actual authenticated public-ABI endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted native owner identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "bind public wrappers only to this authenticated endpoint");
    ExecutionAsyncFixture asynchronous;
    std::uint64_t retirement_frontier{};
    bool has_retirement_token{};
    std::set<std::uint64_t> retirement_captures;
    const auto respond = [&](const ExecutionReceivedRequest& request,
                             CUresult status = CUDA_SUCCESS) {
        try {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = status;
            reply.original_status = status;
            if (request.message.header.operation == ExecutionOpcode::kLoadPtx) reply.id = 1;
            if (request.message.header.operation == ExecutionOpcode::kLoadBinary)
                reply.id = 1000 + request.message.header.sequence;
            if (request.message.header.operation == ExecutionOpcode::kGetFunction)
                reply.id = request.connection == 3 ? 12 : 2;
            const auto operation = request.message.header.operation;
            ExecutionPayloadReader input(request.message.payload);
            if (operation == ExecutionOpcode::kLaunch) {
                ExecutionAsyncFixture::zero_argument_launch(input,
                                                            request.connection == 3 ? 12 : 2);
                if (status == CUDA_SUCCESS) {
                    reply = asynchronous.admit(request.connection, 0);
                    asynchronous.complete(request.connection, reply.id, CUDA_SUCCESS);
                } else {
                    reply.disposition = ExecutionWireDisposition::kRejected;
                    reply.original_status.reset();
                }
            } else if (operation == ExecutionOpcode::kCaptureFrontier) {
                std::uint32_t scope{};
                std::uint64_t stream{};
                expect(input.read_u32(scope) && scope <= 2 && input.read_u64(stream) &&
                           stream == 0 && input.empty(),
                       "capture the exact typed context/default-stream prefix");
                reply = asynchronous.capture(request.connection,
                                             scope ? std::optional{stream} : std::nullopt);
                if (request.connection == 4 && !retirement_frontier) {
                    retirement_frontier = reply.id;
                    has_retirement_token = true;
                }
                if (request.connection == 4) retirement_captures.insert(reply.id);
            } else if (operation == ExecutionOpcode::kQueryAsyncFrontiers ||
                       operation == ExecutionOpcode::kReleaseAsyncFrontiers) {
                expect(status == CUDA_SUCCESS, "batch metadata does not borrow a held capture");
                reply = asynchronous.observe(request);
            } else if (operation == ExecutionOpcode::kQueryFrontier ||
                       operation == ExecutionOpcode::kReleaseFrontier) {
                reply = asynchronous.observe(request);
                if (status != CUDA_SUCCESS) {
                    expect(operation == ExecutionOpcode::kQueryFrontier &&
                               status == CUDA_ERROR_NOT_READY,
                           "only the independently held metadata query may defer completion");
                    reply.status = status;
                    reply.parameter_bytes = 0;
                }
                if (operation == ExecutionOpcode::kReleaseFrontier && request.connection == 4) {
                    std::uint64_t id{};
                    expect(input.read_u64(id) && input.empty(),
                           "exact released retirement identity");
                    if (id == retirement_frontier) has_retirement_token = false;
                    retirement_captures.erase(id);
                }
            } else if (operation == ExecutionOpcode::kClose) {
                if (request.connection == 4) {
                    // A concurrent query may have captured before Close sealed
                    // the epoch, then returned CONTEXT_IS_DESTROYED without a
                    // Release RPC. Actual Close retires those metadata tokens.
                    // Only Capture IDs enter this set, never ASY1 ownership.
                    for (const auto frontier : retirement_captures)
                        asynchronous.abandon(request.connection, frontier);
                    retirement_captures.clear();
                    has_retirement_token = false;
                }
                asynchronous.close(request.connection);
            }
            auto header = request.message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request.connection;
            header.incarnation = {71, 73};
            const auto bytes = encode_execution_reply(reply);
            expect(bytes && server.reply(request.connection, {header, *bytes}),
                   "return only the exact owned original result");
        } catch (const std::exception& error) {
            std::cerr << "Submission reply failure: connection=" << request.connection
                      << "; sequence=" << request.message.header.sequence
                      << "; operation=" << static_cast<unsigned>(request.message.header.operation)
                      << "; cause=" << error.what() << '\n';
            throw;
        }
    };
    unsigned int launches{}, other_launches{}, closes{}, bootstraps{}, allocations{}, frees{},
        runtime_loads{}, runtime_functions{}, runtime_launches{}, runtime_unloads{},
        retirement_closes{};
    std::optional<ExecutionReceivedRequest> dependent, closing, runtime_open, runtime_load,
        runtime_race, freeing, retirement_query;
    Clock::time_point dependent_deadline, close_deadline, retirement_deadline;
    bool has_watchdog{};
    std::atomic<std::uint64_t> last_connection{}, last_sequence{};
    std::atomic<std::uint32_t> last_operation{};
    std::jthread watchdog([&](const std::stop_token& stop) {
        const auto until = Clock::now() + std::chrono::seconds{8};
        while (!stop.stop_requested()) {
            if (Clock::now() >= until) {
                // An owner-side failure must not disappear behind a future's
                // join while its application still waits for an unanswered RPC.
                std::cerr << "Submission fixture watchdog: connection=" << last_connection.load()
                          << "; sequence=" << last_sequence.load()
                          << "; operation=" << last_operation.load() << '\n';
                _exit(98);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    });
    auto client = std::async(std::launch::async, application);
    const auto deadline = Clock::now() + std::chrono::seconds{6};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress real multiplexed endpoint");
        if (runtime_open && runtime_open_release.load(std::memory_order_acquire)) {
            respond(*runtime_open);
            runtime_open.reset();
        }
        if (runtime_load && runtime_load_release.load(std::memory_order_acquire)) {
            respond(*runtime_load);
            runtime_load.reset();
        }
        if (runtime_race && runtime_race_release.load(std::memory_order_acquire)) {
            respond(*runtime_race);
            runtime_race.reset();
        }
        if (retirement_query && (retirement_seal_observed.load(std::memory_order_acquire) ||
                                 Clock::now() >= retirement_deadline)) {
            if (!retirement_seal_observed.load(std::memory_order_acquire)) {
                std::cerr << "Primary retirement watchdog: metadata_wait_prevents_sealing=true\n";
                has_watchdog = true;
            }
            respond(*retirement_query, CUDA_ERROR_NOT_READY);
            retirement_query.reset();
        }
        if (dependent && Clock::now() >= dependent_deadline) {
            std::cerr << "Pending admission watchdog: releasing_submission_blocked=true\n";
            respond(*dependent, CUDA_ERROR_UNKNOWN);
            dependent.reset();
            has_watchdog = true;
        }
        if (closing && Clock::now() >= close_deadline) {
            std::cerr << "Close SDK watchdog: unrelated_context_blocked=true\n";
            respond(*closing, CUDA_ERROR_UNKNOWN);
            closing.reset();
            has_watchdog = true;
        }
        if (auto request = server.take_request()) {
            const auto operation = request->message.header.operation;
            last_connection.store(request->connection);
            last_sequence.store(request->message.header.sequence);
            last_operation.store(static_cast<std::uint32_t>(operation));
            if (operation == ExecutionOpcode::kHello) ++bootstraps;
            if (request->connection == 4) {
                if (operation == ExecutionOpcode::kQueryFrontier) {
                    ExecutionPayloadReader input(request->message.payload);
                    std::uint64_t id{};
                    expect(input.read_u64(id) && input.empty(), "exact retirement frontier query");
                    if (id == retirement_frontier) {
                        retirement_query = std::move(*request);
                        retirement_deadline = Clock::now() + std::chrono::milliseconds{250};
                        retirement_query_observed.store(true, std::memory_order_release);
                        continue;
                    }
                }
                if (operation == ExecutionOpcode::kClose) {
                    expect(!retirement_query &&
                               retirement_seal_observed.load(std::memory_order_acquire),
                           "close waits actual accepted metadata replies after sealing new calls");
                    ++retirement_closes;
                }
            }
            if (request->connection == 2 && operation == ExecutionOpcode::kRelease) {
                ++frees;
                ExecutionPayloadReader input(request->message.payload);
                std::uint64_t id{};
                expect(input.read_u64(id) && id == 100 + frees && input.empty(),
                       "address reuse frees only its exact new native allocation identity");
                if (frees == 1) {
                    freeing = std::move(*request);
                    free_observed.store(true, std::memory_order_release);
                    continue;
                }
            }
            if (request->connection == 2 && operation == ExecutionOpcode::kAllocate) {
                ++allocations;
                ExecutionPayloadReader input(request->message.payload);
                std::uint64_t bytes_requested{};
                expect(input.read_u64(bytes_requested) && bytes_requested == 64 && input.empty(),
                       "both concurrent allocations preserve their original requested extent");
                ExecutionWireReply reply;
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.original_status = 0;
                reply.id = 100 + allocations;
                reply.address = 0x100000;
                auto header = request->message.header;
                header.kind = ExecutionMessageKind::kReply;
                header.session = request->connection;
                header.incarnation = {71, 73};
                const auto bytes = encode_execution_reply(reply);
                expect(bytes && server.reply(request->connection, {header, *bytes}),
                       "return an actual allocation identity before the prior free reply");
                if (allocations == 2) {
                    expect(freeing.has_value(),
                           "address reuse follows an actual pending native free");
                    respond(*freeing);
                    freeing.reset();
                }
                continue;
            }
            if (request->connection == 3) {
                if (operation == ExecutionOpcode::kHello) {
                    runtime_open = std::move(*request);
                    runtime_open_observed.store(true, std::memory_order_release);
                    continue;
                }
                if (operation == ExecutionOpcode::kLoadBinary) {
                    ++runtime_loads;
                    ExecutionPayloadReader input(request->message.payload);
                    std::span<const std::byte> bytes;
                    expect(input.read_blob(bytes) && bytes.size() == 24 &&
                               bytes[16] == (runtime_loads == 1 ? std::byte{7} : std::byte{99}) &&
                               input.empty(),
                           "pending loader owns original image bytes despite caller overwrite");
                    if (runtime_loads == 1) {
                        runtime_load = std::move(*request);
                        runtime_load_observed.store(true, std::memory_order_release);
                    } else {
                        runtime_race = std::move(*request);
                        runtime_race_observed.store(true, std::memory_order_release);
                    }
                    continue;
                }
                if (operation == ExecutionOpcode::kGetFunction) ++runtime_functions;
                if (operation == ExecutionOpcode::kLaunch) ++runtime_launches;
                if (operation == ExecutionOpcode::kUnloadModule) ++runtime_unloads;
            }
            if (operation == ExecutionOpcode::kLaunch && request->connection == 1) {
                ++launches;
                if (launches == 2) {
                    dependent = std::move(*request);
                    dependent_deadline = Clock::now() + std::chrono::milliseconds{250};
                    pending_observed.store(true, std::memory_order_release);
                    continue;
                }
                if (launches == 3 && dependent) {
                    respond(*request);
                    respond(*dependent, CUDA_ERROR_INVALID_VALUE);
                    dependent.reset();
                    continue;
                }
            }
            if (operation == ExecutionOpcode::kClose && request->connection == 1) {
                expect(!dependent && launches == 3,
                       "close cannot overtake unresolved admission ownership");
                ++closes;
                closing = std::move(*request);
                close_deadline = Clock::now() + std::chrono::milliseconds{250};
                close_observed.store(true, std::memory_order_release);
                continue;
            }
            if (operation == ExecutionOpcode::kLaunch && request->connection == 2) {
                ++other_launches;
                expect(closing.has_value(),
                       "unrelated context issues during the actual close wait");
                respond(*request);
                respond(*closing);
                closing.reset();
                continue;
            }
            respond(*request);
        }
        expect(Clock::now() < deadline, "bounded public nonblocking submission control");
    }
    client.get();
    watchdog.request_stop();
    expect(
        !has_watchdog && !dependent && !closing && launches == 3 && other_launches == 1 &&
            closes == 1 && bootstraps == 4 && allocations == 2 && frees == 2 && !freeing &&
            runtime_loads == 2 && runtime_functions == 1 && runtime_launches == 8 &&
            runtime_unloads == 2 && !runtime_open && !runtime_load && !runtime_race &&
            retirement_closes == 1 && !retirement_query && asynchronous.empty() &&
            !has_retirement_token && retirement_captures.empty(),
        "exact original/request lifetimes without replay, duplicate destruction or false success");
    server.stop();
    std::cout
        << "Public multiplexed submission verified: independent_release=true; "
           "pre_admission_error=true; exclusive_close=true; unrelated_context=true; "
           "runtime_concurrent_cache=true; pending_query_retirement=true; gpu_qualified=false\n";
}
}  // namespace
int main() {
    char pattern[] = "/tmp/glimmer-submission-XXXXXX";
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
