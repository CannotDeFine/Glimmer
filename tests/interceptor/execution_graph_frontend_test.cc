#include "glimmer/control/execution_graph_protocol.h"
#include "glimmer/control/execution_socket.h"
#include "execution_async_fixture.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>

namespace {
using namespace glimmer::control;
enum class Scenario { kNormal, kMalformed, kPartialOutput, kLaunchFailure, kMalformedQuarantine };
constexpr std::uint64_t kStream = 1, kModule = 2, kFunction = 3, kAllocation = 4;
constexpr std::uint64_t kGraph = 5, kExecutable = 6, kAddress = 0x100000;
constexpr std::uint64_t kNativeGraph = 0xd0000, kNativeExecutable = 0xe0000;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "public owned Driver graph call succeeds");
}
void check(cudaError_t value) {
    expect(value == cudaSuccess, "public owned Runtime graph call succeeds");
}
template <typename Pointer>
Pointer address(std::uintptr_t bits) {
    // Synthetic SDK output bits are never dereferenced by this CPU fixture.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<Pointer>(bits);
}

void application(Scenario scenario) {
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUmodule module{};
    check(cuModuleLoadData(&module, "owned graph CPU fixture"));
    CUfunction function{};
    check(cuModuleGetFunction(&function, module, "accumulate"));
    CUdeviceptr allocation{};
    check(cuMemAlloc(&allocation, 64));
    cudaStreamCaptureStatus state{cudaStreamCaptureStatusInvalidated};
    check(cudaStreamIsCapturing(stream, &state));
    expect(state == cudaStreamCaptureStatusNone, "idle state is not an active capture");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    check(cudaStreamIsCapturing(stream, &state));
    expect(state == cudaStreamCaptureStatusActive, "actual owner observation reports Active");
    auto wrong_thread = std::async(std::launch::async, [=] {
        check(cuCtxSetCurrent(context));
        auto untouched = address<cudaGraph_t>(0x1234);
        expect(cudaStreamEndCapture(stream, &untouched) == cudaErrorStreamCaptureWrongThread &&
                   untouched == address<cudaGraph_t>(0x1234),
               "non-RELAXED End enforces caller token without consuming capture");
    });
    wrong_thread.get();
    std::uint32_t scalar = 7;
    void* arguments[]{&allocation, &scalar};
    check(cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0, stream, arguments, nullptr));
    scalar = 99;
    check(cuMemsetD32Async(allocation + 16, 9, 4, stream));
    check(cuMemcpyDtoDAsync(allocation + 32, allocation, 16, stream));
    auto graph = address<cudaGraph_t>(0x1234);
    check(cudaStreamEndCapture(stream, &graph));
    expect(graph && reinterpret_cast<std::uintptr_t>(graph) != kNativeGraph,
           "native graph address never becomes a public capability");
    check(cudaStreamIsCapturing(stream, &state));
    expect(state == cudaStreamCaptureStatusNone, "proved End releases local capture state");
    auto executable = address<cudaGraphExec_t>(0x1122334455667788ULL);
    expect(static_cast<int>(cudaGraphInstantiateWithFlags(&executable, graph, 10719)) == 10719 &&
               executable == address<cudaGraphExec_t>(0x1122334455667788ULL) &&
               static_cast<int>(cudaPeekAtLastError()) == 10719 &&
               static_cast<int>(cudaGetLastError()) == 10719,
           "Runtime errors above API-failure-base retain their domain and untouched output");
    expect(cudaGraphInstantiateWithFlags(address<cudaGraphExec_t*>(1), graph, 0) ==
               cudaErrorInvalidValue,
           "inaccessible output rejects before any remote constructor");
    const auto instantiated = cudaGraphInstantiateWithFlags(&executable, graph, 0);
    if (scenario == Scenario::kMalformed || scenario == Scenario::kPartialOutput) {
        const auto expected =
            scenario == Scenario::kPartialOutput ? 0x11223344aabbccddULL : 0x1122334455667788ULL;
        expect(instantiated == cudaErrorUnknown &&
                   reinterpret_cast<std::uintptr_t>(executable) == expected,
               "failed proof preserves real partial output but never mints a capability");
        expect(cudaGraphLaunch(executable, stream) == cudaErrorUnknown,
               "quarantined receipt cannot authorize a subsequent graph replay");
        return;
    }
    check(instantiated);
    expect(executable && reinterpret_cast<std::uintptr_t>(executable) != kNativeExecutable &&
               reinterpret_cast<void*>(executable) != reinterpret_cast<void*>(graph),
           "graph and executable use distinct typed local tokens");
    expect(
        cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(graph), stream) ==
                cudaErrorInvalidResourceHandle &&
            cudaGraphDestroy(address<cudaGraph_t>(kNativeGraph)) == cudaErrorInvalidResourceHandle,
        "wrong-kind and raw SDK pointers cannot become execution authority");
    if (scenario == Scenario::kNormal) {
        CUcontext foreign{};
        check(cuCtxCreate(&foreign, 0, 0));
        expect(cudaGraphLaunch(executable, nullptr) == cudaErrorInvalidResourceHandle,
               "a second authenticated context cannot borrow a graph capability");
        check(cuCtxDestroy(foreign));
    }
    check(cudaGraphDestroy(graph));
    expect(cudaGraphDestroy(graph) == cudaErrorInvalidResourceHandle,
           "source graph cannot be destroyed twice");
    if (scenario == Scenario::kLaunchFailure || scenario == Scenario::kMalformedQuarantine) {
        const auto expected =
            scenario == Scenario::kLaunchFailure ? CUDA_ERROR_LAUNCH_FAILED : CUDA_ERROR_UNKNOWN;
        expect(cuGraphLaunch(executable, stream) == expected,
               "only an exact quarantined native receipt reports Driver 719; malformed rejects");
        expect(cuGraphLaunch(executable, stream) == CUDA_ERROR_UNKNOWN &&
                   cudaGraphLaunch(executable, stream) == cudaErrorUnknown,
               "quarantine rejects later Driver and Runtime replays without another RPC");
        return;
    }
    check(cuGraphLaunch(executable, stream));
    check(cudaGraphLaunch(executable, stream));
    check(cuGraphLaunch(executable, stream));
    check(cuStreamSynchronize(stream));
    check(cudaGraphExecDestroy(executable));
    expect(cuGraphLaunch(executable, stream) == CUDA_ERROR_INVALID_HANDLE,
           "destroyed executable cannot replay");

    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    expect(cudaStreamSynchronize(stream) == cudaErrorStreamCaptureUnsupported,
           "unsupported synchronization invalidates instead of waiting for End");
    check(cudaStreamIsCapturing(stream, &state));
    expect(state == cudaStreamCaptureStatusInvalidated,
           "policy invalidation is visible without claiming native execution");
    graph = address<cudaGraph_t>(0x1234);
    expect(cudaStreamEndCapture(stream, &graph) == cudaErrorStreamCaptureInvalidated && !graph,
           "real End cleanup can succeed while policy proof rejects its graph");
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
    auto relaxed = std::async(std::launch::async, [=] {
        check(cuCtxSetCurrent(context));
        cudaGraph_t result{};
        check(cudaStreamEndCapture(stream, &result));
        return result;
    });
    graph = relaxed.get();
    check(cudaGraphDestroy(graph));

    check(cuMemFree(allocation));
    check(cuModuleUnload(module));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
}

void run(const std::string& path, Scenario scenario) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated graph endpoint fixture");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "obtain trusted endpoint process identity");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select owned frontend before its first call");
    auto client = std::async(std::launch::async, [=] { application(scenario); });
    struct EndpointStop {
        ExecutionSocketServer& server;
        ~EndpointStop() {
            server.stop();
        }
    } endpoint_stop{server};
    ExecutionAsyncFixture asynchronous;
    std::uint64_t owner{}, caller{}, previous_caller{};
    std::uint32_t capture_status{}, mode{};
    unsigned int begins{}, ends{}, states{}, kernels{}, memories{}, launches{}, invalidates{},
        graph_destroys{}, exec_destroys{}, constructors{}, closes{};
    bool is_source_destroyed{}, has_fault{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(std::chrono::steady_clock::now() < deadline, "bounded public graph fixture");
        expect(server.poll_for(1), "progress authenticated graph traffic");
        auto request = server.take_request();
        if (!request) continue;
        ExecutionPayloadReader input(request->message.payload);
        const auto operation = request->message.header.operation;
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.original_status = 0;
        if (operation == ExecutionOpcode::kHello) {
            if (!owner) owner = request->connection;
        } else if (operation == ExecutionOpcode::kCreateStream) {
            std::uint32_t flags{};
            expect(input.read_u32(flags) && flags == CU_STREAM_NON_BLOCKING && input.empty(),
                   "explicit nonblocking capture stream");
            reply.id = kStream;
        } else if (operation == ExecutionOpcode::kLoadPtx) {
            expect(!capture_status, "fixture preloads its Driver function outside capture");
            reply.id = kModule;
        } else if (operation == ExecutionOpcode::kGetFunction) {
            reply.id = kFunction;
            reply.parameter_bytes = 12;
            reply.arguments = {{0, 8}, {8, 4}};
        } else if (operation == ExecutionOpcode::kAllocate) {
            reply.id = kAllocation;
            reply.address = kAddress;
        } else if (operation == ExecutionOpcode::kCaptureFrontier) {
            expect(!capture_status, "capture-time synchronization never reaches ordinary queue");
            std::uint32_t kind{};
            std::uint64_t stream{};
            expect(input.read_u32(kind) && kind <= 2 && input.read_u64(stream) && input.empty(),
                   "capture Begin drains an exact ordinary frontier");
            reply = asynchronous.capture(request->connection,
                                         kind ? std::optional{stream} : std::nullopt);
        } else if (operation == ExecutionOpcode::kQueryFrontier ||
                   operation == ExecutionOpcode::kReleaseFrontier ||
                   operation == ExecutionOpcode::kQueryAsyncFrontiers ||
                   operation == ExecutionOpcode::kReleaseAsyncFrontiers) {
            reply = asynchronous.observe(*request);
        } else if (operation == ExecutionOpcode::kGraphControl) {
            ExecutionGraphRequest graph;
            expect(request->connection == owner && read_execution_graph_request(input, graph) &&
                       input.empty(),
                   "one exact authenticated graph request with no unparsed tail");
            ExecutionGraphObservation observation{.capture_status = capture_status};
            switch (graph.action) {
                case ExecutionGraphAction::kBegin:
                    expect(!capture_status && graph.stream == kStream && graph.caller &&
                               graph.flags == begins,
                           "GLOBAL/THREAD_LOCAL/RELAXED modes reach original construction");
                    previous_caller = caller;
                    caller = graph.caller;
                    expect(!previous_caller || previous_caller == caller,
                           "one live caller retains its nonrecycled token");
                    mode = static_cast<std::uint32_t>(graph.flags);
                    capture_status = cudaStreamCaptureStatusActive;
                    ++begins;
                    break;
                case ExecutionGraphAction::kState:
                    expect(graph.stream == kStream && graph.caller == caller && capture_status,
                           "active status query preserves exact capture ownership");
                    ++states;
                    break;
                case ExecutionGraphAction::kCaptureKernel: {
                    expect(capture_status == cudaStreamCaptureStatusActive &&
                               graph.stream == kStream && graph.caller == caller,
                           "kernel node uses exact capture caller and stream");
                    ExecutionPayloadReader node(graph.payload);
                    std::uint64_t function{}, stream{};
                    expect(node.read_u64(function) && function == kFunction &&
                               node.read_u64(stream) && stream == kStream,
                           "nested original launch references owned resources");
                    for (const auto expected : std::array{320U, 1U, 1U, 32U, 1U, 1U, 0U}) {
                        std::uint32_t dimension{};
                        expect(node.read_u32(dimension) && dimension == expected,
                               "complete captured kernel dimensions");
                    }
                    std::span<const std::byte> parameters;
                    expect(node.read_blob(parameters) && parameters.size() == 12 && node.empty(),
                           "capture does not carry an asynchronous ordinary-launch marker");
                    std::uint64_t pointer{};
                    std::uint32_t scalar{};
                    std::memcpy(&pointer, parameters.data(), 8);
                    std::memcpy(&scalar, parameters.data() + 8, 4);
                    expect(pointer == kAddress && scalar == 7,
                           "kernel node snapshots actual scalar and device pointer bytes");
                    ++kernels;
                    break;
                }
                case ExecutionGraphAction::kCaptureMemory: {
                    expect(capture_status == cudaStreamCaptureStatusActive &&
                               graph.stream == kStream && graph.caller == caller,
                           "memory node uses exact capture caller and stream");
                    ExecutionPayloadReader node(graph.payload);
                    std::uint32_t kind{}, async{}, value{};
                    std::uint64_t destination{}, offset{}, source{}, source_offset{}, count{},
                        stream{};
                    expect(node.read_u32(kind) && node.read_u32(async) && async == 1 &&
                               node.read_u64(destination) && destination == kAllocation &&
                               node.read_u64(offset) && node.read_u64(source) &&
                               node.read_u64(source_offset) && node.read_u64(count) &&
                               node.read_u32(value) && node.read_u64(stream) && stream == kStream &&
                               node.empty(),
                           "memory node owns complete ordinary grammar without execution tail");
                    expect(memories == 0
                               ? kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32) &&
                                     offset == 16 && !source && !source_offset && count == 4 &&
                                     value == 9
                               : kind == static_cast<std::uint32_t>(ExecutionMemoryKind::kCopy) &&
                                     offset == 32 && source == kAllocation && !source_offset &&
                                     count == 16 && !value,
                           "fill and D2D ranges remain exact");
                    ++memories;
                    break;
                }
                case ExecutionGraphAction::kEnd:
                    expect(capture_status && graph.stream == kStream && graph.has_output &&
                               (graph.caller == caller || mode == cudaStreamCaptureModeRelaxed),
                           "only a valid End reaches the native construction owner");
                    if (mode == cudaStreamCaptureModeRelaxed)
                        expect(
                            graph.caller != caller && graph.caller != previous_caller,
                            "a distinct relaxed caller is explicit, not a recycled OS thread id");
                    observation.output_written = true;
                    if (capture_status == cudaStreamCaptureStatusInvalidated) {
                        reply.status = CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
                        reply.disposition = ExecutionWireDisposition::kRejected;
                    } else {
                        reply.id = kGraph + ends;
                        observation.output = kNativeGraph + ends;
                    }
                    capture_status = cudaStreamCaptureStatusNone;
                    ++ends;
                    break;
                case ExecutionGraphAction::kInstantiate:
                    expect(!capture_status && graph.object == kGraph && graph.has_output,
                           "constructor uses a logical source graph and copied output seed");
                    ++constructors;
                    if (graph.flags == 10719) {
                        reply.original_status = 10719;
                    } else {
                        expect(!graph.flags, "fixed supported constructor flags");
                        if (scenario == Scenario::kMalformed) {
                            // Claimed SDK success without an executable identity is malformed.
                            observation.output_written = true;
                            observation.output = kNativeExecutable;
                            has_fault = true;
                        } else if (scenario == Scenario::kPartialOutput) {
                            reply.status = CUDA_ERROR_UNKNOWN;
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.original_status = cudaErrorInvalidValue;
                            observation.output_written = observation.is_quarantined = true;
                            observation.output = 0x11223344aabbccddULL;
                            has_fault = true;
                        } else {
                            reply.id = kExecutable;
                            observation.output_written = true;
                            observation.output = kNativeExecutable;
                        }
                    }
                    break;
                case ExecutionGraphAction::kDestroyGraph:
                    expect(
                        !capture_status && graph.object == (graph_destroys ? kGraph + 2 : kGraph),
                        "only a live logical source graph may be destroyed");
                    is_source_destroyed = true;
                    ++graph_destroys;
                    break;
                case ExecutionGraphAction::kDestroyExecutable:
                    expect(!capture_status && graph.object == kExecutable && launches == 3,
                           "executable lifetime remains independent from source destruction");
                    ++exec_destroys;
                    break;
                case ExecutionGraphAction::kInvalidate:
                    expect(capture_status == cudaStreamCaptureStatusActive &&
                               graph.stream == kStream && graph.caller == caller,
                           "unsupported work records policy invalidation on the exact owner");
                    capture_status = cudaStreamCaptureStatusInvalidated;
                    reply.status = CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
                    reply.disposition = ExecutionWireDisposition::kRejected;
                    reply.original_status.reset();
                    ++invalidates;
                    break;
            }
            observation.capture_status = capture_status;
            ExecutionPayloadWriter payload;
            expect(write_execution_graph_observation(payload, observation),
                   "encode exact SDK graph observation");
            reply.data = payload.take();
        } else if (operation == ExecutionOpcode::kGraphLaunch) {
            ExecutionGraphLaunch launch;
            expect(
                !capture_status && is_source_destroyed &&
                    read_execution_graph_launch(input, launch) && input.empty() &&
                    launch.executable == kExecutable && launch.stream == kStream,
                "both public launch ABIs replay one retained executable after source destruction");
            ExecutionGraphObservation observation;
            if (scenario == Scenario::kLaunchFailure ||
                scenario == Scenario::kMalformedQuarantine) {
                expect(launches == 0, "actual native failure is never replayed");
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.original_status = CUDA_ERROR_LAUNCH_FAILED;
                // A status-zero rejection with a native error but without the
                // declared quarantine proof cannot use the narrow RPC exception.
                observation.is_quarantined = scenario == Scenario::kLaunchFailure;
                has_fault = true;
            }
            ExecutionPayloadWriter payload;
            expect(write_execution_graph_observation(payload, observation),
                   "graph replay observation");
            reply.data = payload.take();
            ++launches;
        } else if (operation == ExecutionOpcode::kClose) {
            expect(!capture_status, "normal context close follows proven capture retirement");
            asynchronous.close(request->connection);
            ++closes;
        } else {
            expect(!capture_status && (operation == ExecutionOpcode::kRelease ||
                                       operation == ExecutionOpcode::kDestroyStream ||
                                       operation == ExecutionOpcode::kUnloadModule),
                   "no captured kernel, copy, helper or synchronization executes ordinarily");
        }
        auto header = request->message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request->connection;
        header.incarnation = {193, 197};
        auto encoded = encode_execution_reply(reply);
        expect(encoded && server.reply(request->connection, {header, std::move(*encoded)}),
               "deliver one authenticated graph receipt");
    }
    client.get();
    expect(kernels == 1 && memories == 2 && constructors == 2,
           "captured operations and constructors occur exactly once");
    if (scenario == Scenario::kNormal)
        expect(begins == 3 && ends == 3 && states == 2 && launches == 3 && invalidates == 1 &&
                   graph_destroys == 2 && exec_destroys == 1 && closes == 2 && asynchronous.empty(),
               "all live resources retire normally; invalid requests add no native operations");
    else if (scenario == Scenario::kLaunchFailure || scenario == Scenario::kMalformedQuarantine)
        expect(has_fault && begins == 1 && ends == 1 && launches == 1 && graph_destroys == 1 &&
                   exec_destroys == 0 && closes == 0,
               "native issue failure reports once, retains ownership and fabricates no cleanup");
    else
        expect(has_fault && begins == 1 && ends == 1 && launches == 0 && closes == 0,
               "unknown effects retain ownership without replay or fabricated cleanup");
    server.stop();
    std::cout << "Owned graph frontend boundary verified; SDK/GPU evidence is separate\n";
}
}  // namespace

int main(int argc, char** argv) {
    Scenario scenario = Scenario::kNormal;
    if (argc == 2) {
        const std::string_view mode{argv[1]};
        if (mode == "--malformed")
            scenario = Scenario::kMalformed;
        else if (mode == "--partial-output")
            scenario = Scenario::kPartialOutput;
        else if (mode == "--launch-failure")
            scenario = Scenario::kLaunchFailure;
        else if (mode == "--malformed-quarantine")
            scenario = Scenario::kMalformedQuarantine;
        else
            return EXIT_FAILURE;
    } else if (argc != 1) {
        return EXIT_FAILURE;
    }
    char pattern[] = "/tmp/glimmer-graph-frontend-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock", scenario);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
