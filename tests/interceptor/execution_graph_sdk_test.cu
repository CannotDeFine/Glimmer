#include <cuda.h>
#include <cuda_runtime.h>

#include "execution_cached_runtime_fixture.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <dlfcn.h>

namespace {
constexpr std::uint32_t kCount = 8193, kGuards = 32, kReplays = 3;
constexpr std::uint32_t kFill = 0x01010101U, kScalar = 7;
constexpr std::size_t kExtent = kCount + 2 * kGuards;
constexpr std::size_t kBytes = kExtent * sizeof(std::uint32_t);

// The original CUDA kernel ABI intentionally has independent count and scalar
// arguments; the full-value oracle detects interchanging their copied bytes.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
__global__ void graph_accumulate(std::uint32_t* output, const std::uint32_t* input,
                                 const std::uint32_t* fill, std::uint32_t count,
                                 std::uint32_t scalar) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = output[index] * 2U + input[index] * scalar + fill[index];
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(result));
}
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " + std::to_string(result));
}

struct GraphApi {
    // Explicit ABI aliases avoid NVCC preserving SDK default arguments in
    // decltype-based template arguments. Every caller still supplies all fields.
    using Instantiate = cudaError_t(CUDARTAPI*)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
    using CopyAsync = cudaError_t(CUDARTAPI*)(void*, const void*, std::size_t, cudaMemcpyKind,
                                              cudaStream_t);
    using FillAsync = cudaError_t(CUDARTAPI*)(void*, int, std::size_t, cudaStream_t);
    using RecordEvent = cudaError_t(CUDARTAPI*)(cudaEvent_t, cudaStream_t);
    using WaitEvent = cudaError_t(CUDARTAPI*)(cudaStream_t, cudaEvent_t, unsigned int);
    ExecutionCachedRuntime runtime;
    decltype(&cudaStreamBeginCapture) begin{&cudaStreamBeginCapture};
    decltype(&cudaStreamEndCapture) end{&cudaStreamEndCapture};
    decltype(&cudaStreamIsCapturing) state{&cudaStreamIsCapturing};
    Instantiate instantiate{static_cast<Instantiate>(&cudaGraphInstantiateWithFlags)};
    decltype(&cudaGraphLaunch) launch{&cudaGraphLaunch};
    decltype(&cudaGraphDestroy) destroy_graph{&cudaGraphDestroy};
    decltype(&cudaGraphExecDestroy) destroy_exec{&cudaGraphExecDestroy};
    CopyAsync copy_async{static_cast<CopyAsync>(&cudaMemcpyAsync)};
    FillAsync fill_async{static_cast<FillAsync>(&cudaMemsetAsync)};
    decltype(&cudaEventCreateWithFlags) create_event{&cudaEventCreateWithFlags};
    RecordEvent record_event{static_cast<RecordEvent>(&cudaEventRecord)};
    WaitEvent wait_event{static_cast<WaitEvent>(&cudaStreamWaitEvent)};
    decltype(&cudaEventDestroy) destroy_event{&cudaEventDestroy};
    decltype(&cuGraphLaunch) driver_launch{&cuGraphLaunch};
    void* driver{};
    GraphApi() = default;
    GraphApi(const GraphApi&) = delete;
    GraphApi& operator=(const GraphApi&) = delete;
    ~GraphApi() {
        if (driver && ::dlclose(driver) != 0) std::cerr << "Driver cache cleanup failed\n";
    }
    void cache() {
        runtime.enable();
#define CACHE(FIELD, NAME) FIELD = runtime.lookup<decltype(FIELD)>(#NAME)
        CACHE(begin, cudaStreamBeginCapture);
        CACHE(end, cudaStreamEndCapture);
        CACHE(state, cudaStreamIsCapturing);
        CACHE(launch, cudaGraphLaunch);
        CACHE(destroy_graph, cudaGraphDestroy);
        CACHE(destroy_exec, cudaGraphExecDestroy);
        CACHE(create_event, cudaEventCreateWithFlags);
        CACHE(destroy_event, cudaEventDestroy);
#undef CACHE
        instantiate = runtime.lookup<Instantiate>("cudaGraphInstantiateWithFlags");
        copy_async = runtime.lookup<CopyAsync>("cudaMemcpyAsync");
        fill_async = runtime.lookup<FillAsync>("cudaMemsetAsync");
        record_event = runtime.lookup<RecordEvent>("cudaEventRecord");
        wait_event = runtime.lookup<WaitEvent>("cudaStreamWaitEvent");
        driver = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!driver) throw std::runtime_error("load Driver graph provider");
        auto* symbol = ::dlsym(driver, "cuGraphLaunch");
        if (!symbol) throw std::runtime_error("cache original Driver graph launch ABI");
        driver_launch = reinterpret_cast<decltype(driver_launch)>(symbol);
    }
    void close() {
        if (driver) {
            auto* owned = driver;
            driver = nullptr;
            if (::dlclose(owned) != 0) throw std::runtime_error("close Driver provider");
        }
        if (!runtime.close()) throw std::runtime_error("close Runtime provider");
    }
};

std::uint32_t initial(std::size_t array, std::size_t index) {
    return static_cast<std::uint32_t>(0x10000U + array * 0x1000U + index);
}

void run(bool is_cached) {
    GraphApi api;
    if (is_cached) api.cache();
    check(api.runtime.select(0), "select device");
    std::array<std::uint32_t*, 4> device{};
    std::array<std::vector<std::uint32_t>, 4> expected;
    for (std::size_t array = 0; array < device.size(); ++array) {
        expected[array].resize(kExtent);
        for (std::size_t index = 0; index < kExtent; ++index)
            expected[array][index] = initial(array, index);
        check(api.runtime.allocate(reinterpret_cast<void**>(&device[array]), kBytes), "allocate");
        check(
            api.runtime.copy(device[array], expected[array].data(), kBytes, cudaMemcpyHostToDevice),
            "initial upload");
    }
    cudaStream_t capture_stream{}, consumer_stream{};
    cudaEvent_t completed{};
    check(api.runtime.create_stream(&capture_stream, cudaStreamNonBlocking), "capture stream");
    check(api.runtime.create_stream(&consumer_stream, cudaStreamNonBlocking), "consumer stream");
    check(api.create_event(&completed, cudaEventDisableTiming), "ordinary completion event");
    cudaStreamCaptureStatus capture_status{};
    check(api.state(capture_stream, &capture_status), "initial capture state");
    if (capture_status != cudaStreamCaptureStatusNone) throw std::runtime_error("initial state");

    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    check(api.begin(capture_stream, cudaStreamCaptureModeGlobal), "begin capture");
    check(api.state(capture_stream, &capture_status), "active capture state");
    if (capture_status != cudaStreamCaptureStatusActive) throw std::runtime_error("active state");
    const auto active_bytes = static_cast<std::size_t>(kCount) * sizeof(std::uint32_t);
    check(api.copy_async(device[1] + kGuards, device[0] + kGuards, active_bytes,
                         cudaMemcpyDeviceToDevice, capture_stream),
          "capture device copy");
    check(api.fill_async(device[2] + kGuards, 1, active_bytes, capture_stream), "capture fill");
    auto* output = device[3] + kGuards;
    auto* input = device[1] + kGuards;
    auto* fill = device[2] + kGuards;
    auto count = kCount;
    auto scalar = kScalar;
    // The first use of this exact kernel occurs inside capture. No application
    // warmup, Glimmer helper, environment-selected oracle or private API.
    if (is_cached) {
        void* arguments[]{&output, &input, &fill, &count, &scalar};
        check(api.runtime.launch(reinterpret_cast<const void*>(&graph_accumulate), dim3{320},
                                 dim3{32}, arguments, 0, capture_stream),
              "capture cached kernel");
    } else {
        graph_accumulate<<<320, 32, 0, capture_stream>>>(output, input, fill, count, scalar);
        check(api.runtime.last_error(), "capture compiler kernel");
    }
    scalar = 99;
    count = 1;
    check(api.end(capture_stream, &graph), "end capture");
    if (!graph) throw std::runtime_error("missing graph");
    check(api.state(capture_stream, &capture_status), "ended capture state");
    if (capture_status != cudaStreamCaptureStatusNone) throw std::runtime_error("ended state");
    check(api.instantiate(&executable, graph, 0), "instantiate original graph");
    if (!executable) throw std::runtime_error("missing graph executable");
    check(api.destroy_graph(graph), "destroy source before any replay");
    graph = nullptr;

    std::uint64_t checked_values{}, checksum{1469598103934665603ULL};
    std::vector<std::uint32_t> observed(kExtent);
    const auto verify = [&] {
        for (std::size_t array = 0; array < device.size(); ++array) {
            check(api.runtime.copy(observed.data(), device[array], kBytes, cudaMemcpyDeviceToHost),
                  "complete output snapshot");
            for (std::size_t index = 0; index < kExtent; ++index) {
                if (observed[index] != expected[array][index])
                    throw std::runtime_error(
                        "graph full-value/guard mismatch array=" + std::to_string(array) +
                        " index=" + std::to_string(index));
                ++checked_values;
                checksum = (checksum ^ observed[index]) * 1099511628211ULL;
            }
        }
    };
    verify();  // Capture/instantiate/source destruction must not execute nodes.
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    for (std::uint32_t replay = 0; replay < kReplays; ++replay) {
        for (std::size_t index = kGuards; index < kGuards + kCount; ++index) {
            const auto input_value =
                static_cast<std::uint32_t>(index + static_cast<std::size_t>(replay) * 13U);
            expected[0][index] = input_value;
            expected[1][index] = input_value;
            expected[2][index] = kFill;
            expected[3][index] = expected[3][index] * 2U + input_value * kScalar + kFill;
        }
        check(api.runtime.copy(device[0], expected[0].data(), kBytes, cudaMemcpyHostToDevice),
              "update replay input");
        if (replay % 2U == 0)
            check(api.driver_launch(executable, capture_stream), "actual Driver graph replay");
        else
            check(api.launch(executable, capture_stream), "Runtime graph replay");
        check(api.record_event(completed, capture_stream), "record graph completion prefix");
        check(api.wait_event(consumer_stream, completed, 0), "consumer waits graph prefix");
        // The async download must observe the graph through its event dependency,
        // before any device-wide or capture-stream synchronization can hide a bug.
        check(api.copy_async(observed.data(), device[3], kBytes, cudaMemcpyDeviceToHost,
                             consumer_stream),
              "dependent consumer download");
        check(api.runtime.synchronize_stream(consumer_stream), "consumer physical completion");
        if (observed != expected[3]) throw std::runtime_error("graph event ordering mismatch");
        verify();
    }
    check(api.runtime.synchronize_stream(capture_stream), "capture stream completion");
    check(api.destroy_exec(executable), "destroy executable");
    check(api.destroy_event(completed), "destroy event");
    check(api.runtime.destroy_stream(consumer_stream), "destroy consumer stream");
    check(api.runtime.destroy_stream(capture_stream), "destroy capture stream");
    for (auto* pointer : device) check(api.runtime.release(pointer), "release allocation");
    check(api.runtime.reset(), "normal Runtime teardown");
    api.close();
    std::cout << "{\"status\":\"graph_sdk_oracle_passed\",\"cached\":"
              << (is_cached ? "true" : "false") << ",\"replays\":" << kReplays
              << ",\"driver_replays\":2,\"runtime_replays\":1,\"checked_values\":" << checked_values
              << ",\"checksum\":" << checksum
              << ",\"capture_executed\":false,\"source_destroyed_before_replay\":true,"
                 "\"full_device_qualified\":false,\"performance_qualified\":false}\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--cached-lookups"))
            throw std::runtime_error("usage: execution_graph_sdk_test [--cached-lookups]");
        run(argc == 2);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
