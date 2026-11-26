#include "internal/cuda_execution_session.h"

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Node {
    enum class Kind { kFill, kCopy, kKernel } kind{};
    CUdeviceptr destination{}, source{};
    std::size_t count{};
    std::uint32_t value{};
};
struct Graph {
    std::vector<Node> nodes;
};
struct Native {
    CUcontext current{};
    CUstream capture_stream{};
    std::vector<Node> captured;
    std::map<CUdeviceptr, std::vector<std::byte>> allocations;
    std::map<CUevent, bool> events;
    std::map<cudaGraph_t, Graph*> graphs;
    std::map<cudaGraphExec_t, Graph*> executables;
    bool ready{true}, fail_state{}, fail_destroy{}, fail_context{}, fail_end_output{},
        fail_instantiate{}, fail_launch{};
    unsigned int private_events{}, kernel_calls{}, launches{}, graph_destroys{}, exec_destroys{},
        loads{}, frees{};
    int function_tag{};
} native;
void* pointer(CUdeviceptr address) {
    return std::bit_cast<void*>(address);
}
void execute(const Node& node) {
    if (node.kind == Node::Kind::kCopy) {
        std::memcpy(pointer(node.destination), pointer(node.source), node.count);
        return;
    }
    for (std::size_t index = 0; index != node.count; ++index) {
        std::uint32_t value = node.value;
        if (node.kind == Node::Kind::kKernel) {
            std::uint32_t old{}, input{};
            std::memcpy(&old, pointer(node.destination + index * 4), 4);
            std::memcpy(&input, pointer(node.source + index * 4), 4);
            value += old + input;
        }
        std::memcpy(pointer(node.destination + index * 4), &value, 4);
    }
}
CUresult CUDAAPI initialize(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device(CUdevice* value, int ordinal) {
    *value = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI current(CUcontext* output) {
    *output = native.current;
    return native.fail_context ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
}
CUresult CUDAAPI create_context(CUcontext* output, unsigned int, CUdevice) {
    *output = reinterpret_cast<CUcontext>(new int(0));
    native.current = *output;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI set_context(CUcontext value) {
    native.current = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI destroy_context(CUcontext value) {
    delete reinterpret_cast<int*>(value);
    native.current = nullptr;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI create_stream(CUstream* output, unsigned int) {
    *output = reinterpret_cast<CUstream>(new int(0));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI destroy_stream(CUstream value) {
    delete reinterpret_cast<int*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_create(CUevent* output, unsigned int) {
    expect(!native.capture_stream, "private event creation must never enter capture");
    ++native.private_events;
    *output = reinterpret_cast<CUevent>(new int(0));
    native.events.emplace(*output, false);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_record(CUevent event, CUstream) {
    expect(!native.capture_stream, "private event record must never enter capture");
    native.events.at(event) = true;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_query(CUevent event) {
    expect(!native.capture_stream, "private event poll must never enter capture");
    return native.ready && native.events.at(event) ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent event) {
    expect(native.events.erase(event) == 1, "event destroyed exactly once");
    delete reinterpret_cast<int*>(event);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI synchronize(CUstream) {
    return native.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI wait_event(CUstream, CUevent event, unsigned int) {
    return native.events.contains(event) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_HANDLE;
}
CUresult CUDAAPI allocate(CUdeviceptr* output, std::size_t bytes) {
    std::vector<std::byte> data(bytes);
    *output = reinterpret_cast<CUdeviceptr>(data.data());
    native.allocations.emplace(*output, std::move(data));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI release(CUdeviceptr address) {
    expect(native.allocations.erase(address) == 1, "free actual allocation once");
    ++native.frees;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI upload(CUdeviceptr destination, const void* source, std::size_t bytes) {
    std::memcpy(pointer(destination), source, bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI download(void* destination, CUdeviceptr source, std::size_t bytes) {
    std::memcpy(destination, pointer(source), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI fill(CUdeviceptr destination, unsigned int value, std::size_t count,
                      CUstream stream) {
    Node node{Node::Kind::kFill, destination, 0, count, value};
    if (native.capture_stream == stream)
        native.captured.push_back(node);
    else
        execute(node);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI copy(CUdeviceptr destination, CUdeviceptr source, std::size_t bytes,
                      CUstream stream) {
    Node node{Node::Kind::kCopy, destination, source, bytes, 0};
    if (native.capture_stream == stream)
        native.captured.push_back(node);
    else
        execute(node);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI load(CUmodule* output, const void*) {
    ++native.loads;
    *output = reinterpret_cast<CUmodule>(new int(0));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI unload(CUmodule module) {
    delete reinterpret_cast<int*>(module);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI function(CUfunction* output, CUmodule, const char*) {
    *output = reinterpret_cast<CUfunction>(&native.function_tag);
    return CUDA_SUCCESS;
}
// Original cuFuncGetParamInfo ABI keeps its two independent output pointers.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CUresult CUDAAPI parameter(CUfunction, std::size_t index, std::size_t* offset, std::size_t* bytes) {
    constexpr std::array offsets{0U, 8U, 16U, 20U};
    if (index >= offsets.size()) return CUDA_ERROR_INVALID_VALUE;
    *offset = offsets[index];
    *bytes = index < 2 ? 8 : 4;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI kernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, CUstream stream, void** arguments,
                        void**) {
    ++native.kernel_calls;
    Node node{.kind = Node::Kind::kKernel};
    std::uint32_t count{};
    std::memcpy(&node.destination, arguments[0], 8);
    std::memcpy(&node.source, arguments[1], 8);
    std::memcpy(&count, arguments[2], 4);
    std::memcpy(&node.value, arguments[3], 4);
    node.count = count;
    if (native.capture_stream == stream)
        native.captured.push_back(node);
    else
        execute(node);
    return CUDA_SUCCESS;
}
cudaError_t CUDARTAPI begin(cudaStream_t stream, cudaStreamCaptureMode) {
    if (native.capture_stream) return cudaErrorIllegalState;
    native.capture_stream = stream;
    native.captured.clear();
    return cudaSuccess;
}
cudaError_t CUDARTAPI state(cudaStream_t stream, cudaStreamCaptureStatus* output) {
    if (native.fail_state) return cudaErrorUnknown;
    *output = native.capture_stream == stream ? cudaStreamCaptureStatusActive
                                              : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
cudaError_t CUDARTAPI end(cudaStream_t stream, cudaGraph_t* output) {
    if (!output) return cudaErrorInvalidValue;
    if (stream != native.capture_stream) return cudaErrorIllegalState;
    auto* graph = new Graph{native.captured};
    *output = reinterpret_cast<cudaGraph_t>(graph);
    native.graphs.emplace(*output, graph);
    native.capture_stream = nullptr;
    return native.fail_end_output ? cudaErrorUnknown : cudaSuccess;
}
cudaError_t CUDARTAPI instantiate(cudaGraphExec_t* output, cudaGraph_t graph, unsigned long long) {
    if (!output) return cudaErrorInvalidValue;
    if (native.fail_instantiate) {
        *output = nullptr;
        return cudaErrorInvalidValue;
    }
    auto* executable = new Graph{*native.graphs.at(graph)};
    *output = reinterpret_cast<cudaGraphExec_t>(executable);
    native.executables.emplace(*output, executable);
    return cudaSuccess;
}
cudaError_t CUDARTAPI destroy(cudaGraph_t graph) {
    ++native.graph_destroys;
    if (native.fail_destroy) return cudaErrorUnknown;
    delete native.graphs.at(graph);
    native.graphs.erase(graph);
    return cudaSuccess;
}
cudaError_t CUDARTAPI destroy_exec(cudaGraphExec_t executable) {
    ++native.exec_destroys;
    delete native.executables.at(executable);
    native.executables.erase(executable);
    return cudaSuccess;
}
CUresult CUDAAPI graph_launch(CUgraphExec executable, CUstream) {
    ++native.launches;
    for (const auto& node : native.executables.at(executable)->nodes) execute(node);
    return native.fail_launch ? CUDA_ERROR_LAUNCH_FAILED : CUDA_SUCCESS;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{
        {current, kernel, event_create, event_record, event_query, event_destroy, synchronize},
        initialize,
        device,
        create_context,
        set_context,
        destroy_context,
        load,
        unload,
        function,
        parameter,
        allocate,
        release,
        upload,
        download,
        create_stream,
        destroy_stream};
    result.execution.fill32_async = fill;
    result.execution.copy_device_async = copy;
    result.stream_wait_event = wait_event;
    result.graph = {begin, end, state, instantiate, destroy, destroy_exec, graph_launch};
    return result;
}
constexpr std::string_view kSource = R"(.version 7.0
.target sm_70
.address_size 64
.visible .entry add(.param .u64 output, .param .u64 input, .param .u32 count, .param .u32 delta) { ret; }
)";
void lifecycle() {
    native = {};
    CudaExecutionSession session({1, 1}, 1, dispatch());
    expect(session.initialize(0).status == CUDA_SUCCESS, "context initialize");
    const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
    const auto consumer = session.create_stream(CU_STREAM_NON_BLOCKING);
    const auto input = session.allocate(64), copied = session.allocate(64),
               output = session.allocate(64);
    expect(stream.status == CUDA_SUCCESS && consumer.status == CUDA_SUCCESS &&
               input.status == CUDA_SUCCESS && copied.status == CUDA_SUCCESS &&
               output.status == CUDA_SUCCESS,
           "owned capture resources");
    std::array<std::uint32_t, 16> values{};
    values.fill(1);
    expect(session.upload({output.id, 0}, std::as_bytes(std::span{values})).status == CUDA_SUCCESS,
           "initial output");
    const auto events_before = native.private_events;
    const auto begun = session.begin_graph_capture(stream.id, cudaStreamCaptureModeGlobal, 11);
    expect(begun.status == CUDA_SUCCESS && begun.original_status == cudaSuccess &&
               session.graph_snapshot().has_capture,
           "actual SDK begins capture with authenticated logical caller");
    expect(session.end_graph_capture(stream.id, 12, true, 0).status ==
                   CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD &&
               session.graph_snapshot().has_capture,
           "wrong logical caller cannot consume capture");
    expect(session.create_event(0).status == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED &&
               session.poll().status == CUDA_ERROR_NOT_READY,
           "ordinary private work cannot enter capture");
    const auto module = session.load_ptx(kSource);
    const auto function_id = session.get_function(module.id, "add");
    expect(module.status == CUDA_SUCCESS && function_id.status == CUDA_SUCCESS && native.loads == 1,
           "first function resolves while capturing without application warmup");
    expect(session.capture_memory({.kind = CudaMemoryKind::kFill32,
                                   .destination = {input.id, 0},
                                   .count = 16,
                                   .value = 2,
                                   .stream = stream.id,
                                   .is_asynchronous = true},
                                  11)
                   .original_status == CUDA_SUCCESS,
           "capture original fill");
    expect(session.capture_memory({.kind = CudaMemoryKind::kCopy,
                                   .destination = {copied.id, 0},
                                   .source = {input.id, 0},
                                   .count = 64,
                                   .stream = stream.id,
                                   .is_asynchronous = true},
                                  11)
                   .original_status == CUDA_SUCCESS,
           "capture original device copy");
    struct Parameters {
        CUdeviceptr output, input;
        std::uint32_t count, delta;
    } parameters{output.address, copied.address, 16, 3};
    expect(session.capture_kernel({.function = function_id.id,
                                   .stream = stream.id,
                                   .grid = {1, 1, 1},
                                   .block = {16, 1, 1},
                                   .parameters = std::as_bytes(std::span{&parameters, 1})},
                                  11)
                   .original_status == CUDA_SUCCESS,
           "capture original kernel with copied parameters");
    parameters.delta = 999;
    expect(native.private_events == events_before && native.launches == 0 &&
               *static_cast<std::uint32_t*>(pointer(output.address)) == 1,
           "construction has no numerical effect or private event");
    const auto ended = session.end_graph_capture(stream.id, 11, true, 0);
    expect(ended.status == CUDA_SUCCESS && ended.original_status == cudaSuccess && ended.id &&
               !session.graph_snapshot().has_capture,
           "End owns a real graph and proves capture None");
    native.fail_instantiate = true;
    const auto failed = session.instantiate_graph(ended.id, 0, true, 99);
    expect(failed.status == CUDA_SUCCESS && failed.original_status == cudaErrorInvalidValue &&
               failed.output_written && !failed.output && !failed.id && !failed.is_quarantined,
           "known constructor error-side NULL remains distinct from policy failure");
    native.fail_instantiate = false;
    const auto executable = session.instantiate_graph(ended.id, 0, true, 0);
    expect(executable.status == CUDA_SUCCESS && executable.id && executable.id != ended.id,
           "independent typed executable identity");
    expect(session.destroy_graph_exec(ended.id).status == CUDA_ERROR_INVALID_HANDLE,
           "graph token cannot alias an executable");
    expect(session.destroy_graph(ended.id).status == CUDA_SUCCESS && native.graphs.empty(),
           "source graph destroys independently before all replay");
    expect(session.unload_module(module.id).status == CUDA_ERROR_NOT_READY &&
               session.release(output.id).status == CUDA_ERROR_NOT_READY,
           "executable retains original module and opaque-kernel device references");
    for (std::uint64_t sequence = 1; sequence != 4; ++sequence) {
        native.ready = false;
        auto accepted = session.accept_graph_launch(sequence, executable.id, stream.id);
        expect(accepted.acceptance.accepted() && session.issue_next().sequence == sequence &&
                   accepted.operation->reply()->original_status == CUDA_SUCCESS,
               "actual whole-graph replay host result");
        expect(session.acknowledge_result(accepted.acceptance.receipt), "replay host ACK");
        expect(session.destroy_graph_exec(executable.id).status == CUDA_ERROR_NOT_READY,
               "host ACK cannot release executable before physical tail");
        native.ready = true;
        expect(session.poll().status == CUDA_SUCCESS, "actual independent tail observation");
    }
    expect(native.launches == 3, "three original non-idempotent launches");
    const auto event = session.create_event(CU_EVENT_DISABLE_TIMING);
    expect(event.status == CUDA_SUCCESS &&
               session.record_event(4, event.id, stream.id).status == CUDA_SUCCESS &&
               session.issue_next().sequence == 4,
           "ordinary event follows replay");
    expect(session.poll().status == CUDA_SUCCESS, "record's exact physical frontier retires");
    expect(session.wait_event(5, event.id, consumer.id).status == CUDA_SUCCESS &&
               session.issue_next().sequence == 5 && session.poll().status == CUDA_SUCCESS,
           "consumer stream waits on ordinary replay completion event");
    expect(session.download({output.id, 0}, std::as_writable_bytes(std::span{values})).status ==
               CUDA_SUCCESS,
           "read all final values");
    for (auto value : values) expect(value == 16, "independent repeated-replay full-value oracle");
    expect(session.destroy_graph_exec(executable.id).status == CUDA_SUCCESS &&
               native.executables.empty(),
           "executable destroys once after every replay tail");
    expect(session.destroy_graph(ended.id).status == CUDA_ERROR_INVALID_HANDLE &&
               !session.accept_graph_launch(6, executable.id, stream.id).acceptance.accepted(),
           "stale typed graph and executable reject");
    expect(session.unload_module(module.id).status == CUDA_SUCCESS,
           "captured module releases after executable destruction");
    expect(session.release(input.id).status == CUDA_SUCCESS,
           "captured fill allocation releases after executable destruction");
    expect(session.release(copied.id).status == CUDA_SUCCESS,
           "captured copy allocation releases after executable destruction");
    expect(session.release(output.id).status == CUDA_SUCCESS,
           "captured kernel allocation releases after executable destruction");
    // The real SDK also releases streams/events at context destruction. Destroy
    // them explicitly here so the fake SDK proves each resource was unpinned.
    expect(session.destroy_event(event.id).status == CUDA_SUCCESS,
           "ordinary replay completion event destroys after consumer retirement");
    expect(session.destroy_stream(stream.id).status == CUDA_SUCCESS,
           "capture stream destroys after executable destruction");
    expect(session.destroy_stream(consumer.id).status == CUDA_SUCCESS,
           "consumer stream destroys after wait retirement");
    expect(session.shutdown().status == CUDA_SUCCESS, "graph Session shuts down normally");
    expect(native.graphs.empty() && native.executables.empty() && native.events.empty() &&
               native.allocations.empty(),
           "all SDK objects and resource leases have normal cleanup");
}
void policy_and_failure_controls() {
    for (int scenario = 0; scenario != 5; ++scenario) {
        const auto child = ::fork();
        expect(child >= 0, "fork bounded graph ownership control");
        if (!child) {
            try {
                native = {};
                CudaExecutionSession session({2, 1}, 2, dispatch());
                expect(session.initialize(0).status == CUDA_SUCCESS, "failure context");
                const auto stream = session.create_stream(CU_STREAM_NON_BLOCKING);
                const auto allocation = scenario == 4 ? session.allocate(64) : CudaSessionResult{};
                expect(session.begin_graph_capture(stream.id, cudaStreamCaptureModeRelaxed, 31)
                               .status == CUDA_SUCCESS,
                       "relaxed capture begins");
                if (!scenario) {
                    expect(session.invalidate_graph_capture(32).status ==
                               CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                           "unsupported capture policy invalidates without work");
                    const auto ended = session.end_graph_capture(stream.id, 32, true, 99);
                    expect(ended.status == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED &&
                               ended.original_status == cudaSuccess && ended.output_written &&
                               !ended.output && !ended.id && !ended.is_quarantined &&
                               !session.graph_snapshot().has_capture &&
                               native.graph_destroys == 1 &&
                               session.shutdown().status == CUDA_SUCCESS,
                           "proven policy-invalid End cleans native graph once");
                } else if (scenario == 4) {
                    expect(allocation.status == CUDA_SUCCESS && allocation.id &&
                               session.capture_memory({.kind = CudaMemoryKind::kFill32,
                                                       .destination = {allocation.id, 0},
                                                       .count = 16,
                                                       .value = 7,
                                                       .stream = stream.id,
                                                       .is_asynchronous = true},
                                                      31)
                                       .original_status == CUDA_SUCCESS,
                           "launch failure retains a graph-owned device allocation");
                    const auto ended = session.end_graph_capture(stream.id, 31, true, 0);
                    const auto executable = session.instantiate_graph(ended.id, 0, true, 0);
                    expect(ended.status == CUDA_SUCCESS && ended.id &&
                               executable.status == CUDA_SUCCESS && executable.id,
                           "launch failure owns actual source graph and executable");
                    native.fail_launch = true;
                    auto accepted = session.accept_graph_launch(1, executable.id, stream.id);
                    expect(accepted.acceptance.accepted(), "failed replay was reliably accepted");
                    (void)session.issue_next();
                    const auto reply = accepted.operation->reply();
                    expect(
                        reply && reply->original_status == 719 && reply->status == CUDA_SUCCESS &&
                            reply->is_quarantined && native.launches == 1 &&
                            *static_cast<std::uint32_t*>(pointer(allocation.address)) == 7,
                        "actual Driver launch error preserves 719 and independent context proof");
                    expect(session.graph_snapshot().is_unknown &&
                               !session.accept_graph_launch(2, executable.id, stream.id)
                                    .acceptance.accepted() &&
                               session.destroy_graph_exec(executable.id).is_quarantined &&
                               session.destroy_graph(ended.id).is_quarantined &&
                               session.release(allocation.id).status == CUDA_ERROR_UNKNOWN &&
                               session.shutdown().status == CUDA_ERROR_UNKNOWN,
                           "uncertain replay blocks new launch and resource destruction");
                    (void)session.issue_next();
                    expect(native.launches == 1 && native.graph_destroys == 0 &&
                               native.exec_destroys == 0 && native.frees == 0 &&
                               native.graphs.size() == 1 && native.executables.size() == 1 &&
                               native.allocations.size() == 1,
                           "launch uncertainty retains resources without replay or cleanup retry");
                } else {
                    if (scenario == 1) native.fail_end_output = true;
                    if (scenario == 2) native.fail_state = true;
                    if (scenario == 3) {
                        native.fail_destroy = true;
                        (void)session.invalidate_graph_capture(31);
                    }
                    const auto ended = session.end_graph_capture(stream.id, 31, true, 0);
                    expect(ended.is_quarantined && !ended.id && !native.graphs.empty(),
                           "partial output, lost state proof and failed cleanup retain graph "
                           "ownership");
                    const auto destroys = native.graph_destroys;
                    expect(session.end_graph_capture(stream.id, 31, true, 0).is_quarantined &&
                               native.graph_destroys == destroys &&
                               session.shutdown().status != CUDA_SUCCESS,
                           "unknown ownership forbids replay and cleanup retries");
                }
                std::_Exit(0);
            } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                std::_Exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated graph failure control passes");
    }
}
}  // namespace
int main() {
    try {
        lifecycle();
        policy_and_failure_controls();
        std::cout << "Owned graph construction and replay checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
