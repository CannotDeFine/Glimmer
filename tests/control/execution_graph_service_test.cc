#include "runtime.h"
#include "glimmer/control/execution_graph_protocol.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using Clock = std::chrono::steady_clock;
using Action = ExecutionGraphAction;
constexpr std::size_t kWords = 32, kBytes = kWords * 4;
constexpr std::uint32_t kGuard = 0x728392a1, kFill = 0x37461526;
void expect(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "graph service: " << message << '\n';
    std::_Exit(1);
}
struct Node {
    CUdeviceptr address{};
    std::uint32_t value{};
    std::size_t count{};
};
struct Graph {
    std::vector<Node> nodes;
};
struct Native {
    CUcontext current{};
    CUstream capture{};
    std::vector<Node> nodes;
    std::set<CUcontext> contexts;
    std::set<CUstream> streams;
    std::set<CUevent> events;
    std::map<CUdeviceptr, std::vector<std::byte>> memory;
    std::atomic<bool> ready{true};
    std::atomic<unsigned int> allocations{}, launches{}, exec_destroys{}, graph_destroys{};
} native;
void ordinary() {
    expect(!native.capture, "ordinary SDK work cannot enter while capture lease is active");
}
CUresult CUDAAPI initialize(unsigned int) {
    ordinary();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device(CUdevice* value, int ordinal) {
    *value = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI current(CUcontext* value) {
    *value = native.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_create(CUcontext* value, unsigned int, CUdevice) {
    ordinary();
    *value = reinterpret_cast<CUcontext>(new int(0));
    native.contexts.insert(*value);
    native.current = *value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_set(CUcontext value) {
    expect(!native.capture || value == native.current, "capture never switches native context");
    native.current = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_destroy(CUcontext value) {
    ordinary();
    expect(native.contexts.erase(value) == 1, "exact context destroyed once");
    if (native.current == value) native.current = nullptr;
    delete reinterpret_cast<int*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_create(CUevent* value, unsigned int) {
    ordinary();
    *value = reinterpret_cast<CUevent>(new bool(false));
    native.events.insert(*value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_record(CUevent value, CUstream) {
    ordinary();
    *reinterpret_cast<bool*>(value) = true;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_query(CUevent value) {
    ordinary();
    expect(native.events.contains(value) && *reinterpret_cast<bool*>(value), "recorded tail only");
    return native.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent value) {
    ordinary();
    expect(native.events.erase(value) == 1, "exact event release");
    delete reinterpret_cast<bool*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI synchronize(CUstream) {
    ordinary();
    return native.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI kernel(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
    expect(false, "memory-only service input never launches a kernel");
    return CUDA_ERROR_UNKNOWN;
}
CUresult CUDAAPI module_load(CUmodule*, const void*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI module_unload(CUmodule) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI function(CUfunction*, CUmodule, const char*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI parameter(CUfunction, std::size_t, std::size_t*, std::size_t*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI allocate(CUdeviceptr* value, std::size_t bytes) {
    ordinary();
    std::vector<std::byte> data(bytes);
    *value = reinterpret_cast<CUdeviceptr>(data.data());
    native.memory.emplace(*value, std::move(data));
    ++native.allocations;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI release(CUdeviceptr value) {
    ordinary();
    expect(native.ready && native.memory.erase(value) == 1, "release after actual graph tails");
    return CUDA_SUCCESS;
}
auto range(CUdeviceptr address, std::size_t bytes) {
    const auto found = std::ranges::find_if(native.memory, [&](const auto& item) {
        return address >= item.first && address - item.first <= item.second.size() &&
               bytes <= item.second.size() - (address - item.first);
    });
    expect(found != native.memory.end(), "all captured native ranges remain live");
    return found->second.data() + (address - found->first);
}
CUresult CUDAAPI upload(CUdeviceptr address, const void* data, std::size_t bytes) {
    ordinary();
    std::memcpy(range(address, bytes), data, bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI download(void* data, CUdeviceptr address, std::size_t bytes) {
    ordinary();
    std::memcpy(data, range(address, bytes), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_create(CUstream* value, unsigned int) {
    ordinary();
    *value = reinterpret_cast<CUstream>(new int(0));
    native.streams.insert(*value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_destroy(CUstream value) {
    ordinary();
    expect(native.streams.erase(value) == 1, "stream remains until graph owner release");
    delete reinterpret_cast<int*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI fill(CUdeviceptr address, unsigned int value, std::size_t count, CUstream stream) {
    expect(native.capture == stream, "captured fill must use exact construction stream");
    (void)range(address, count * 4);
    native.nodes.push_back({address, value, count});
    return CUDA_SUCCESS;
}
cudaError_t CUDARTAPI begin(cudaStream_t stream, cudaStreamCaptureMode) {
    expect(!native.capture && native.streams.contains(stream), "real begin once on owned stream");
    native.capture = stream;
    native.nodes.clear();
    return cudaSuccess;
}
cudaError_t CUDARTAPI state(cudaStream_t stream, cudaStreamCaptureStatus* value) {
    if (!value) return cudaErrorInvalidValue;
    *value = stream == native.capture && native.capture ? cudaStreamCaptureStatusActive
                                                        : cudaStreamCaptureStatusNone;
    return cudaSuccess;
}
cudaError_t CUDARTAPI end(cudaStream_t stream, cudaGraph_t* value) {
    if (!value) return cudaErrorInvalidValue;
    expect(stream == native.capture, "end consumes exact native capture");
    *value = reinterpret_cast<cudaGraph_t>(new Graph{native.nodes});
    native.capture = nullptr;
    native.nodes.clear();
    return cudaSuccess;
}
cudaError_t CUDARTAPI instantiate(cudaGraphExec_t* value, cudaGraph_t graph,
                                  unsigned long long flags) {
    ordinary();
    if (!value || flags) return cudaErrorInvalidValue;
    *value = reinterpret_cast<cudaGraphExec_t>(new Graph{*reinterpret_cast<Graph*>(graph)});
    return cudaSuccess;
}
cudaError_t CUDARTAPI graph_destroy(cudaGraph_t graph) {
    ordinary();
    delete reinterpret_cast<Graph*>(graph);
    ++native.graph_destroys;
    return cudaSuccess;
}
cudaError_t CUDARTAPI exec_destroy(cudaGraphExec_t graph) {
    ordinary();
    expect(native.ready, "graph exec SDK destruction must wait for actual physical stop");
    delete reinterpret_cast<Graph*>(graph);
    ++native.exec_destroys;
    return cudaSuccess;
}
CUresult CUDAAPI replay(CUgraphExec graph, CUstream stream) {
    ordinary();
    expect(native.streams.contains(stream), "replay only on original owned stream");
    for (const auto& node : reinterpret_cast<Graph*>(graph)->nodes) {
        auto* data = range(node.address, node.count * 4);
        for (std::size_t word = 0; word < node.count; ++word)
            std::memcpy(data + word * 4, &node.value, 4);
    }
    ++native.launches;
    return CUDA_SUCCESS;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{
        {current, kernel, event_create, event_record, event_query, event_destroy, synchronize},
        initialize,
        device,
        context_create,
        context_set,
        context_destroy,
        module_load,
        module_unload,
        function,
        parameter,
        allocate,
        release,
        upload,
        download,
        stream_create,
        stream_destroy};
    result.execution.fill32_async = fill;
    result.graph = {begin, end, state, instantiate, graph_destroy, exec_destroy, replay};
    return result;
}
std::vector<std::byte> number(std::uint64_t value) {
    ExecutionPayloadWriter output;
    expect(output.write_u64(value), "encode identity");
    return output.take();
}
ExecutionWireReply completed(std::optional<ExecutionWireReply> result) {
    expect(result && result->status == CUDA_SUCCESS &&
               result->disposition == ExecutionWireDisposition::kComplete,
           "complete authenticated service receipt");
    return *result;
}
ExecutionWireReply graph(ExecutionSocketClient& client, const ExecutionGraphRequest& value) {
    ExecutionPayloadWriter output;
    expect(write_execution_graph_request(output, value), "encode graph call");
    auto reply = client.exchange(ExecutionOpcode::kGraphControl, output.take());
    expect(reply.has_value(), "actual service graph reply");
    ExecutionPayloadReader input(reply->data);
    ExecutionGraphObservation observation;
    expect(read_execution_graph_observation(input, observation) && input.empty() &&
               !observation.is_quarantined,
           "graph observation remains bounded and known");
    return *reply;
}
void check(ExecutionSocketClient& client, std::uint64_t allocation, bool filled) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(allocation) && input.write_u64(0) && input.write_u64(kBytes),
           "read image");
    const auto reply = completed(client.exchange(ExecutionOpcode::kDownload, input.take()));
    expect(reply.data.size() == kBytes, "full image, never checksum-only");
    for (std::size_t index = 0; index < kWords; ++index) {
        std::uint32_t value{};
        std::memcpy(&value, reply.data.data() + index * 4, 4);
        expect(value == (filled && index >= 4 && index < 28 ? kFill : kGuard),
               "every captured output and guard matches independent oracle");
    }
}
void client_path(const ExecutionSocketClientConfig& config) {
    ExecutionSocketClient client(config), other(config);
    completed(client.open());
    completed(other.open());
    ExecutionPayloadWriter stream_payload;
    expect(stream_payload.write_u32(CU_STREAM_NON_BLOCKING), "explicit nonblocking stream");
    const auto stream =
        completed(client.exchange(ExecutionOpcode::kCreateStream, stream_payload.take())).id;
    const auto allocation =
        completed(client.exchange(ExecutionOpcode::kAllocate, number(kBytes))).id;
    std::array<std::uint32_t, kWords> image;
    image.fill(kGuard);
    ExecutionPayloadWriter upload_payload;
    expect(upload_payload.write_u64(allocation) && upload_payload.write_u64(0) &&
               upload_payload.write_blob(std::as_bytes(std::span{image})),
           "upload full image");
    completed(client.exchange(ExecutionOpcode::kUpload, upload_payload.take()));
    completed(graph(client, {.action = Action::kBegin, .stream = stream, .caller = 11}));
    auto competing = std::async(std::launch::async, [&] {
        return completed(other.exchange(ExecutionOpcode::kAllocate, number(64))).id;
    });
    expect(competing.wait_for(std::chrono::milliseconds{30}) == std::future_status::timeout &&
               native.allocations == 1,
           "other peer SDK allocation is parked during capture");
    ExecutionPayloadWriter node;
    expect(node.write_u32(static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32)) &&
               node.write_u32(1) && node.write_u64(allocation) && node.write_u64(16) &&
               node.write_u64(0) && node.write_u64(0) && node.write_u64(24) &&
               node.write_u32(kFill) && node.write_u64(stream),
           "exact captured range");
    const auto payload = node.take();
    completed(graph(
        client,
        {.action = Action::kCaptureMemory, .stream = stream, .caller = 11, .payload = payload}));
    const auto wrong =
        graph(client, {.action = Action::kEnd, .has_output = true, .stream = stream, .caller = 12});
    expect(wrong.status == CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD && !wrong.id,
           "foreign logical caller cannot consume native capture");
    const auto source =
        completed(
            graph(client,
                  {.action = Action::kEnd, .has_output = true, .stream = stream, .caller = 11}))
            .id;
    expect(competing.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
           "parked peer resumes after proven native None");
    const auto other_allocation = competing.get();
    check(client, allocation, false);
    const auto executable =
        completed(
            graph(client, {.action = Action::kInstantiate, .has_output = true, .object = source}))
            .id;
    completed(graph(client, {.action = Action::kDestroyGraph, .object = source}));
    for (unsigned int repeat = 0; repeat < 3; ++repeat) {
        if (repeat == 2) native.ready = false;
        ExecutionPayloadWriter launch;
        expect(write_execution_graph_launch(launch, {executable, stream}), "replay actual exec");
        const auto replay =
            completed(client.exchange(ExecutionOpcode::kGraphLaunch, launch.take()));
        ExecutionGraphObservation observed;
        ExecutionPayloadReader result(replay.data);
        expect(!replay.id && !replay.address && !replay.parameter_bytes &&
                   replay.arguments.empty() && replay.original_status == CUDA_SUCCESS &&
                   read_execution_graph_observation(result, observed) && result.empty() &&
                   !observed.output_written && !observed.is_quarantined && !observed.capture_status,
               "replay reply has exact graph shape, never a generic command ID");
        if (repeat < 2) check(client, allocation, true);
    }
    auto destroying = std::async(std::launch::async, [&] {
        return graph(client, {.action = Action::kDestroyExecutable, .object = executable});
    });
    expect(destroying.wait_for(std::chrono::milliseconds{30}) == std::future_status::timeout &&
               native.exec_destroys == 0,
           "host reply cannot retire a graph's held physical tail");
    native.ready = true;
    expect(destroying.wait_for(std::chrono::seconds{2}) == std::future_status::ready,
           "exec destroy progresses after actual tail retirement");
    completed(destroying.get());
    check(client, allocation, true);
    completed(graph(client, {.action = Action::kBegin, .stream = stream, .caller = 11}));
    const auto unsupported = client.exchange(ExecutionOpcode::kMemory, payload);
    expect(unsupported && unsupported->status == CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
           "raw unsupported capture route cannot bypass frontend containment");
    const auto invalidated =
        graph(client, {.action = Action::kEnd, .has_output = true, .stream = stream, .caller = 11});
    expect(invalidated.status == CUDA_ERROR_STREAM_CAPTURE_INVALIDATED && !invalidated.id,
           "policy-invalid capture actually ends without publishing a graph");
    completed(client.exchange(ExecutionOpcode::kRelease, number(allocation)));
    completed(other.exchange(ExecutionOpcode::kRelease, number(other_allocation)));
    completed(client.exchange(ExecutionOpcode::kDestroyStream, number(stream)));
    completed(client.exchange(ExecutionOpcode::kClose, {}));
    completed(other.exchange(ExecutionOpcode::kClose, {}));
}
}  // namespace

int main() {
    std::array<char, 64> directory{};
    const std::string pattern = "/tmp/glimmer-graph-service-XXXXXX";
    std::ranges::copy(pattern, directory.begin());
    expect(::mkdtemp(directory.data()), "create unique socket directory");
    const std::string path = std::string(directory.data()) + "/socket";
    const auto identity = current_process_identity();
    expect(identity.has_value(), "resolve trusted server PID/start identity");
    ExecutionSocketClientConfig client_config{path, *identity, 5000};
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    config.priority_idle_us = 0;
    {
        ExecutionService service(config, dispatch());
        expect(service.start(), "start real authenticated service");
        auto client = std::async(std::launch::async, [&] { client_path(client_config); });
        const auto deadline = Clock::now() + std::chrono::seconds{20};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
            expect(service.step(1) && Clock::now() < deadline, "bounded actual service progress");
        client.get();
        while (!service.finished())
            expect(service.step(1) && Clock::now() < deadline, "normal owner cleanup");
        expect(service.request_stop(), "no native owner retained at normal shutdown");
    }
    expect(native.memory.empty() && native.contexts.empty() && native.streams.empty() &&
               native.events.empty() && !native.capture && native.launches == 3 &&
               native.exec_destroys == 1 && native.graph_destroys == 2,
           "complete images and all native identities retired once");
    expect(::rmdir(directory.data()) == 0, "remove exact empty test directory");
    return EXIT_SUCCESS;
}
