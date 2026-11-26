#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_host_function_native_calls();
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc_ptsz(cudaStream_t, cudaHostFn_t, void*);
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
// Literal future wire IDs allow this normal-cleanup regression to run against
// the retained pre-route frontend without changing its production headers.
constexpr auto kEnqueue = static_cast<ExecutionOpcode>(134);
constexpr auto kQuery = static_cast<ExecutionOpcode>(135);
constexpr auto kFinish = static_cast<ExecutionOpcode>(136);
enum class Mode { kOrdinary, kSaturated, kNotRun, kBadQuery, kPrimaryReset };
enum class Route { kExplicit, kPerThread, kPtdsNull, kPtdsExplicit, kPtdsLegacy, kPtdsSpecial };
bool uses_proxy(Route route) {
    return route == Route::kPerThread || route == Route::kPtdsNull || route == Route::kPtdsSpecial;
}
bool uses_explicit_stream(Route route) {
    return route == Route::kExplicit || route == Route::kPtdsExplicit;
}
cudaStream_t callback_stream(Route route, CUstream stream) {
    if (route == Route::kPtdsNull) return nullptr;
    if (route == Route::kPtdsLegacy) return cudaStreamLegacy;
    return reinterpret_cast<cudaStream_t>(uses_proxy(route) ? CU_STREAM_PER_THREAD : stream);
}
decltype(&cudaLaunchHostFunc) callback_entry(Route route) {
    const auto* symbol = route == Route::kExplicit || route == Route::kPerThread
                             ? "cudaLaunchHostFunc"
                             : "cudaLaunchHostFunc_ptsz";
    auto cached = reinterpret_cast<decltype(&cudaLaunchHostFunc)>(::dlsym(RTLD_DEFAULT, symbol));
    const auto direct = route == Route::kExplicit || route == Route::kPerThread
                            ? &cudaLaunchHostFunc
                            : &cudaLaunchHostFunc_ptsz;
    if (cached != direct)
        throw std::runtime_error("cached callback must select the public wrapper");
    return cached;
}
const char* route_option(Route route) {
    switch (route) {
        case Route::kExplicit:
            return "--explicit";
        case Route::kPerThread:
            return "--per-thread";
        case Route::kPtdsNull:
            return "--ptsz-null";
        case Route::kPtdsExplicit:
            return "--ptsz-explicit";
        case Route::kPtdsLegacy:
            return "--ptsz-legacy";
        case Route::kPtdsSpecial:
            return "--ptsz-special";
    }
    throw std::runtime_error("invalid callback route");
}
void expect(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
template <class Predicate>
void await(Predicate&& predicate) {
    const auto until = Clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        expect(Clock::now() < until, "bounded asynchronous host-function observation");
        std::this_thread::yield();
    }
}
struct Latches {
    std::atomic<bool> returned{}, predecessor_stopped{}, started{}, may_finish{};
    std::atomic<bool> holds_metadata{};
    std::atomic<bool> rejects_next_frontier{};
    std::atomic<unsigned int> held_metadata{};
    std::atomic<unsigned int> callbacks{};
    std::atomic<CUresult> reentry{CUDA_SUCCESS};
    std::thread::id caller;
    std::atomic<bool> is_worker{};
};
void CUDART_CB callback(void* opaque) {
    auto& latches = *static_cast<Latches*>(opaque);
    latches.is_worker = std::this_thread::get_id() != latches.caller;
    unsigned int flags{};
    latches.reentry = cuCtxGetFlags(&flags);
    latches.started = true;
    while (!latches.may_finish) std::this_thread::yield();
    ++latches.callbacks;
}
bool application(Latches& latches, bool is_saturated, Route route) {
    latches.caller = std::this_thread::get_id();
    CUcontext context{};
    CUstream stream{};
    expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS &&
               cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS,
           "owned context and stream setup");
    const auto launch = callback_entry(route);
    const auto selected = callback_stream(route, stream);
    const auto result = launch(selected, callback, &latches);
    const bool accepted = result == cudaSuccess;
    latches.returned = true;
    if (accepted) {
        expect(launch(selected, nullptr, &latches) == cudaErrorInvalidValue,
               "null callback is rejected locally without creating an owner callback");
        await([&] { return latches.started.load(); });
        expect(latches.predecessor_stopped && latches.callbacks == 0 && latches.is_worker,
               "callback follows physical predecessor and does not require submitting thread");
        unsigned int flags{99};
        expect(cuCtxGetFlags(&flags) == CUDA_SUCCESS && flags == 0,
               "application metadata progresses while callback holds no frontend lock");
        expect(launch(selected, callback, &latches) == cudaSuccess,
               "second callback queues while first is still running");
        std::vector<std::future<CUresult>> queries;
        if (is_saturated) {
            latches.holds_metadata = true;
            for (unsigned int index = 0; index < 9; ++index)
                queries.push_back(std::async(std::launch::async, [context] {
                    if (const auto selected = cuCtxSetCurrent(context); selected != CUDA_SUCCESS)
                        return selected;
                    unsigned int queried_flags{99};
                    const auto queried = cuCtxGetFlags(&queried_flags);
                    if (queried != CUDA_SUCCESS) return queried;
                    return queried_flags == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
                }));
            await([&] { return latches.held_metadata.load() == 7; });
            expect(latches.callbacks == 0,
                   "seven ordinary transport receipts are held before callback return");
        }
        latches.may_finish = true;
        for (auto& query : queries)
            expect(query.get() == CUDA_SUCCESS,
                   "saturated metadata callers drain after both callback receipts");
    }
    expect(cuStreamDestroy(stream) == CUDA_SUCCESS && cuCtxDestroy(context) == CUDA_SUCCESS,
           "normal stream/context cleanup waits for exact callback return receipts");
    return accepted && latches.callbacks == 2 && latches.reentry == CUDA_ERROR_NOT_PERMITTED &&
           glimmer_host_function_native_calls() == 0;
}
bool failed_application(Latches& latches, bool has_bad_query, Route route) {
    CUcontext context{};
    CUstream stream{};
    expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS &&
               cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS,
           "failure case owns its normal context and stream");
    const auto launch = callback_entry(route);
    latches.may_finish = true;
    const auto selected = callback_stream(route, stream);
    expect(launch(selected, callback, &latches) == cudaSuccess,
           "failure is asynchronous after successful host-function admission");
    latches.returned = true;
    if (has_bad_query) {
        if (!uses_explicit_stream(route)) {
            expect(cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN &&
                       cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN &&
                       cuStreamDestroy(stream) == CUDA_ERROR_INVALID_HANDLE &&
                       launch(selected, callback, &latches) == cudaErrorUnknown,
                   "uncertain per-thread permission retains context/proxy pins without cleanup");
            return latches.callbacks == 0 && !latches.started &&
                   glimmer_host_function_native_calls() == 0;
        }
        const auto stream_status = cuStreamDestroy(stream);
        expect(
            (stream_status == CUDA_ERROR_UNKNOWN || stream_status == CUDA_ERROR_INVALID_HANDLE) &&
                cuStreamDestroy(stream) == CUDA_ERROR_INVALID_HANDLE &&
                cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN &&
                cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN &&
                launch(reinterpret_cast<cudaStream_t>(stream), callback, &latches) ==
                    cudaErrorUnknown,
            "mismatched permission retains pins, invalidates current-context use and rejects "
            "cleanup");
    } else {
        expect(cuStreamDestroy(stream) == CUDA_SUCCESS,
               "known-not-run receipt releases exact callback pins before normal stream cleanup");
        if (!uses_explicit_stream(route)) {
            CUresult observed{};
            await([&] {
                observed = cuCtxSynchronize();
                return observed != CUDA_SUCCESS;
            });
            expect(observed == CUDA_ERROR_LAUNCH_FAILED,
                   "known per-thread callback absence becomes observable independently of an "
                   "unrelated nonblocking stream");
        }
        expect(cuCtxSynchronize() == CUDA_ERROR_LAUNCH_FAILED &&
                   cuCtxSynchronize() == CUDA_ERROR_LAUNCH_FAILED &&
                   launch(nullptr, callback, &latches) == cudaErrorLaunchFailure,
               "physical prefix success cannot erase a known callback predecessor error");
        expect(cuCtxDestroy(context) == CUDA_SUCCESS,
               "known absent callback does not quarantine otherwise proven context cleanup");
    }
    return latches.callbacks == 0 && !latches.started && glimmer_host_function_native_calls() == 0;
}
bool primary_reset_application(Latches& latches, Route route) {
    latches.caller = std::this_thread::get_id();
    latches.may_finish = true;
    cudaStream_t stream{};
    const auto selected_stream = [&] {
        return callback_stream(route, reinterpret_cast<CUstream>(stream));
    };
    const auto launch = callback_entry(route);
    expect(cudaSetDevice(0) == cudaSuccess &&
               cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
               launch(selected_stream(), callback, &latches) == cudaSuccess,
           "ordinary Runtime primary context accepts its first asynchronous callback");
    latches.returned = true;
    expect(cudaStreamDestroy(stream) == cudaSuccess && latches.callbacks == 0,
           "known absent callback permits physically proven Runtime stream cleanup");
    if (!uses_explicit_stream(route)) {
        cudaError_t observed{};
        await([&] {
            observed = cudaDeviceSynchronize();
            return observed != cudaSuccess;
        });
        expect(observed == cudaErrorInvalidValue,
               "per-thread known absence reaches its originating primary epoch before reset");
    }
    expect(cudaDeviceSynchronize() == cudaErrorInvalidValue &&
               launch(nullptr, callback, &latches) == cudaErrorInvalidValue,
           "nonfatal callback predecessor error remains sticky before successful reset");
    latches.rejects_next_frontier = true;
    expect(cudaDeviceReset() == cudaErrorInvalidResourceHandle &&
               launch(nullptr, callback, &latches) == cudaErrorInvalidValue,
           "cleanup cannot bypass actual physical-frontier errors or prematurely clear callback "
           "errors");
    expect(cudaDeviceReset() == cudaSuccess,
           "independently proven physical stop and Close allow primary reset after known absence");
    expect(
        cudaSetDevice(0) == cudaSuccess &&
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
            launch(selected_stream(), callback, &latches) == cudaSuccess &&
            cudaStreamDestroy(stream) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
            cudaDeviceReset() == cudaSuccess,
        "new primary epoch accepts callback token reuse and clears only the retired epoch error");
    return latches.callbacks == 1 && latches.is_worker &&
           latches.reentry == CUDA_ERROR_NOT_PERMITTED && glimmer_host_function_native_calls() == 0;
}
void run(Mode mode, Route route) {
    const bool uses_per_thread = uses_proxy(route);
    const bool is_saturated = mode == Mode::kSaturated;
    const bool has_failure = mode == Mode::kNotRun || mode == Mode::kBadQuery;
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-host-function-" + std::to_string(::getpid());
    config.max_pending_requests = 8;
    ExecutionSocketServer server(config);
    expect(server.start(), "start authenticated callback fixture");
    const auto owner = current_process_identity();
    expect(owner.has_value(), "exact current owner identity");
    const auto pid = std::to_string(owner->pid), start = std::to_string(owner->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select remote execution before first application call");
    Latches latches;
    auto client = std::async(std::launch::async, [&] {
        if (mode == Mode::kPrimaryReset) return primary_reset_application(latches, route);
        return has_failure ? failed_application(latches, mode == Mode::kBadQuery, route)
                           : application(latches, is_saturated, route);
    });
    std::uint64_t accepted{}, completed{}, query_replies{}, cancelled{};
    std::vector<ExecutionReceivedRequest> pending, metadata;
    unsigned int metadata_queries{}, opened{}, closed{}, stream_creations{};
    const auto reply = [&](const ExecutionReceivedRequest& request,
                           const ExecutionWireReply& result) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {43, 47};
        const auto encoded = encode_execution_reply(result);
        expect(encoded && server.reply(request.connection, {header, *encoded}),
               "exact original request receipt");
    };
    const auto until = Clock::now() + std::chrono::seconds(12);
    unsigned int turns{};
    while (client.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < until, "bounded actual callback IPC progress");
        if (completed == 2) {
            for (const auto& request : metadata) {
                ExecutionWireReply result;
                result.disposition = ExecutionWireDisposition::kComplete;
                result.original_status = 0;
                reply(request, result);
            }
            metadata.clear();
        }
        if (latches.returned && ++turns >= 8) latches.predecessor_stopped = true;
        for (auto item = pending.begin(); item != pending.end();) {
            ExecutionPayloadReader input(item->message.payload);
            std::uint64_t id{};
            expect(input.read_u64(id) && input.empty(), "exact held callback identity");
            if (latches.predecessor_stopped &&
                id == (mode == Mode::kPrimaryReset ? 1 : completed + 1)) {
                ExecutionWireReply result;
                result.disposition = ExecutionWireDisposition::kComplete;
                result.id = id;
                if (mode == Mode::kNotRun || (mode == Mode::kPrimaryReset && accepted == 1)) {
                    result.status =
                        mode == Mode::kNotRun ? CUDA_ERROR_LAUNCH_FAILED : CUDA_ERROR_INVALID_VALUE;
                    result.parameter_bytes = 1;
                    ++cancelled;
                } else if (mode == Mode::kBadQuery)
                    ++result.id;
                reply(*item, result);
                ++query_replies;
                item = pending.erase(item);
            } else
                ++item;
        }
        auto request = server.take_request();
        if (!request) continue;
        ExecutionPayloadReader input(request->message.payload);
        ExecutionWireReply result;
        result.disposition = ExecutionWireDisposition::kComplete;
        const auto operation = request->message.header.operation;
        std::uint64_t id{};
        std::uint32_t returned{};
        if (operation == kEnqueue) {
            const auto expected_stream = uses_per_thread               ? 8U
                                         : uses_explicit_stream(route) ? 7U
                                                                       : 0U;
            expect(input.read_u64(id) && id == expected_stream && input.empty(),
                   "callback binds exact explicit or per-thread proxy stream");
            ++accepted;
            result.id = mode == Mode::kPrimaryReset ? 1 : accepted;
        } else if (operation == kQuery) {
            pending.push_back(std::move(*request));
            continue;
        } else if (operation == kFinish) {
            expect(!has_failure, "failed readiness cannot authorize callback invocation or finish");
            expect(input.read_u64(id) && id == (mode == Mode::kPrimaryReset ? 1 : completed + 1) &&
                       input.read_u32(returned) && returned == 1 && input.empty() &&
                       latches.callbacks >= completed + 1,
                   "finish follows exactly one actual callback return in stream order");
            ++completed;
            result.id = id;
        } else if (operation == ExecutionOpcode::kCreateStream) {
            const bool is_proxy = uses_per_thread && stream_creations % 2 != 0;
            std::uint32_t flags{};
            expect(input.read_u32(flags) && input.empty() &&
                       flags == (is_proxy ? CU_STREAM_DEFAULT : CU_STREAM_NON_BLOCKING),
                   "per-thread callbacks own a regular proxy, not the independent control stream");
            result.id = is_proxy ? 8 : 7;
            result.original_status = 0;
            ++stream_creations;
        } else if (operation == ExecutionOpcode::kGetContextFlags) {
            ++metadata_queries;
            if (latches.holds_metadata && completed != 2) {
                metadata.push_back(std::move(*request));
                ++latches.held_metadata;
                continue;
            }
            result.original_status = 0;
        } else if (operation == ExecutionOpcode::kCaptureFrontier) {
            expect(mode == Mode::kNotRun || mode == Mode::kPrimaryReset,
                   "only known-not-run error uses ordinary prefix observation");
            result.id = 41;
        } else if (operation == ExecutionOpcode::kQueryFrontier ||
                   operation == ExecutionOpcode::kReleaseFrontier) {
            expect((mode == Mode::kNotRun || mode == Mode::kPrimaryReset) && input.read_u64(id) &&
                       id == 41 && input.empty(),
                   "known callback error remains independent of exact successful physical prefix");
            if (operation == ExecutionOpcode::kQueryFrontier &&
                latches.rejects_next_frontier.exchange(false))
                result.status = CUDA_ERROR_INVALID_HANDLE;
        } else {
            expect(operation == ExecutionOpcode::kHello || operation == ExecutionOpcode::kClose ||
                       operation == ExecutionOpcode::kDestroyStream ||
                       operation == ExecutionOpcode::kRuntimeMetadata,
                   "only required ordinary setup/cleanup operations reach fake SDK");
            if (operation == ExecutionOpcode::kDestroyStream ||
                operation == ExecutionOpcode::kClose) {
                expect(mode != Mode::kBadQuery,
                       "uncertain permission cannot send remote stream or context cleanup");
                expect((!uses_explicit_stream(route) &&
                        operation == ExecutionOpcode::kDestroyStream) ||
                           completed + cancelled == accepted,
                       "cleanup never overtakes a callback boundary or known-not-run receipt");
            }
            if (operation == ExecutionOpcode::kHello) ++opened;
            if (operation == ExecutionOpcode::kClose) ++closed;
            result.original_status = 0;
        }
        reply(*request, result);
    }
    const bool passed = client.get();
    expect(pending.empty() && metadata.empty(),
           "normal cleanup leaves no callback readiness or held metadata request");
    server.stop();
    const auto expected_completed = has_failure ? 0U : (mode == Mode::kPrimaryReset ? 1U : 2U);
    expect(passed && accepted == (has_failure ? 1U : 2U) && completed == expected_completed &&
               query_replies == accepted &&
               stream_creations ==
                   (uses_per_thread ? 2U : 1U) * (mode == Mode::kPrimaryReset ? 2U : 1U),
           "exact callback successes or known/unknown failures preserve no-fallback ownership");
    if (mode == Mode::kPrimaryReset)
        expect(opened == 2 && closed == 2 && cancelled == 1,
               "two real primary ownership generations close normally around one known callback "
               "error");
    if (is_saturated)
        expect(
            latches.held_metadata == 7 && metadata_queries == 10,
            "reserved eighth slot serves Finish1, Query2 and Finish2 before nine ordinary callers");
}
void isolated_bad_query(Route route) {
    const auto child = ::fork();
    expect(child >= 0, "create isolated unknown-callback-lifetime process");
    if (!child) {
        // A fresh executable is required: inherited frontend process identity
        // must reject CUDA in a fork child before touching any inherited mutex.
        ::execl("/proc/self/exe", "glimmer_execution_host_function_test", "--bad-query-child",
                route_option(route), static_cast<char*>(nullptr));
        ::_exit(127);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
               WEXITSTATUS(status) == EXIT_SUCCESS,
           "isolated malformed-query observation finishes without manufacturing cleanup");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        Route route{Route::kExplicit};
        std::string_view argument;
        bool has_route{};
        for (int index = 1; index < argc; ++index) {
            const std::string_view option{argv[index]};
            if (option == "--saturated" || option == "--not-run" || option == "--bad-query" ||
                option == "--bad-query-child" || option == "--primary-reset") {
                expect(argument.empty(), "only one failure/pressure mode is allowed");
                argument = option;
                continue;
            }
            expect(!has_route, "only one callback stream route is allowed");
            has_route = true;
            if (option == "--explicit")
                route = Route::kExplicit;
            else if (option == "--per-thread")
                route = Route::kPerThread;
            else if (option == "--ptsz-null")
                route = Route::kPtdsNull;
            else if (option == "--ptsz-explicit")
                route = Route::kPtdsExplicit;
            else if (option == "--ptsz-legacy")
                route = Route::kPtdsLegacy;
            else if (option == "--ptsz-special")
                route = Route::kPtdsSpecial;
            else
                throw std::runtime_error("unknown callback stream route");
        }
        if (argument == "--bad-query") {
            isolated_bad_query(route);
        } else {
            const auto mode = argument == "--saturated"         ? Mode::kSaturated
                              : argument == "--not-run"         ? Mode::kNotRun
                              : argument == "--bad-query-child" ? Mode::kBadQuery
                              : argument == "--primary-reset"   ? Mode::kPrimaryReset
                                                                : Mode::kOrdinary;
            run(mode, route);
            if (mode == Mode::kBadQuery)
                ::_exit(
                    EXIT_SUCCESS);  // Intentional uncertainty remains pinned until process death.
        }
        std::cout << "Transparent asynchronous host-function checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
