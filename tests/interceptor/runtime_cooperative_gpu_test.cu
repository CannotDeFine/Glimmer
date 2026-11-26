#include <cuda_runtime.h>
#include <cuda.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string_view>
#include <thread>
#include <type_traits>

namespace {
constexpr unsigned int kBlocks = 105;
constexpr unsigned int kThreads = 64;
constexpr std::size_t kCount = kBlocks * kThreads;
struct KernelArguments {
    unsigned int* output;
    unsigned int work;
    unsigned int increment;
};
static_assert(sizeof(KernelArguments) == 16);

__global__ void low_kernel(KernelArguments arguments, unsigned int* marker) {
    __shared__ unsigned int logical_block;
    if (threadIdx.x == 0) {
        logical_block = blockIdx.x;
        // Business progress only; the application knows no retreat protocol.
        *marker = 1;
        __threadfence_system();
    }
    __syncthreads();
    const auto index = logical_block * blockDim.x + threadIdx.x;
    for (unsigned int iteration = 0; iteration < arguments.work; ++iteration) {
        const unsigned int value = static_cast<volatile unsigned int*>(arguments.output)[index];
        (void)value;
    }
    arguments.output[index] += arguments.increment;
}

__global__ void high_kernel(unsigned int* marker) {
    *marker += 1;
    __threadfence_system();
}

__global__ void graph_kernel(unsigned int* marker, unsigned int* data) {
    for (unsigned int iteration = 0; iteration < 500000; ++iteration) {
        const unsigned int value = static_cast<volatile unsigned int*>(data)[0];
        (void)value;
    }
    *marker += 1;
    __threadfence_system();
}

bool check(cudaError_t result, const char* operation) {
    if (result == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
    return false;
}

struct RuntimeApi {
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    Launch launch = cudaLaunchKernel;
    decltype(&cudaMemcpyAsync) copy = cudaMemcpyAsync;
    decltype(&cudaEventRecord) record = cudaEventRecord;
    decltype(&cudaStreamWaitEvent) wait = cudaStreamWaitEvent;
    decltype(&cudaStreamQuery) query = cudaStreamQuery;
    decltype(&cudaStreamSynchronize) synchronize = cudaStreamSynchronize;
};

template <typename Function>
bool resolve(Function& output, const char* symbol) {
    output = reinterpret_cast<Function>(::dlsym(RTLD_DEFAULT, symbol));
    if (output == nullptr) {
        std::cerr << "missing Runtime alias " << symbol << '\n';
        return false;
    }
    return true;
}

bool check_public_graphs(const RuntimeApi& api, cudaStream_t low, cudaStream_t origin,
                         cudaStream_t branch, cudaEvent_t fork, unsigned int* output,
                         unsigned int* input, unsigned int* readback, unsigned int* device_markers,
                         unsigned int* markers, bool is_ptds, bool uses_driver,
                         unsigned long long constructor_flags) {
    static_assert(std::is_same_v<cudaStream_t, CUstream>);
    static_assert(std::is_same_v<cudaGraph_t, CUgraph>);
    static_assert(std::is_same_v<cudaGraphExec_t, CUgraphExec>);
    auto begin = &cudaStreamBeginCapture;
    auto end = &cudaStreamEndCapture;
    auto replay = &cudaGraphLaunch;
    if (is_ptds &&
        (!resolve(begin, "cudaStreamBeginCapture_ptsz") ||
         !resolve(end, "cudaStreamEndCapture_ptsz") || !resolve(replay, "cudaGraphLaunch_ptsz"))) {
        return false;
    }
    cudaEvent_t join{};
    using DriverBegin = CUresult(CUDAAPI*)(CUstream, CUstreamCaptureMode);
    using DriverEnd = CUresult(CUDAAPI*)(CUstream, CUgraph*);
    using DriverReplay = CUresult(CUDAAPI*)(CUgraphExec, CUstream);
    using DriverConstruct = decltype(&cuGraphInstantiateWithFlags);
    void* driver_handle = uses_driver ? dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL) : nullptr;
    if (uses_driver && !driver_handle) return false;
    const auto driver_begin = uses_driver
                                  ? reinterpret_cast<DriverBegin>(dlsym(
                                        driver_handle, is_ptds ? "cuStreamBeginCapture_v2_ptsz"
                                                               : "cuStreamBeginCapture_v2"))
                                  : nullptr;
    const auto driver_end =
        uses_driver
            ? reinterpret_cast<DriverEnd>(
                  dlsym(driver_handle, is_ptds ? "cuStreamEndCapture_ptsz" : "cuStreamEndCapture"))
            : nullptr;
    const auto driver_replay =
        uses_driver ? reinterpret_cast<DriverReplay>(
                          dlsym(driver_handle, is_ptds ? "cuGraphLaunch_ptsz" : "cuGraphLaunch"))
                    : nullptr;
    const auto driver_construct =
        uses_driver
            ? reinterpret_cast<DriverConstruct>(dlsym(driver_handle, "cuGraphInstantiateWithFlags"))
            : nullptr;
    if (uses_driver && (!driver_begin || !driver_end || !driver_replay || !driver_construct))
        return false;
    const auto driver_check = [](CUresult result, const char* operation) {
        if (result == CUDA_SUCCESS) return true;
        std::cerr << operation << " failed with Driver status " << result << '\n';
        return false;
    };
    const auto begin_capture = [&] {
        return uses_driver
                   ? driver_check(driver_begin(origin, CU_STREAM_CAPTURE_MODE_GLOBAL),
                                  "public Driver begin")
                   : check(begin(origin, cudaStreamCaptureModeGlobal), "public graph begin");
    };
    const auto end_capture = [&](cudaGraph_t* target) {
        return uses_driver ? driver_check(driver_end(origin, target), "public Driver end")
                           : check(end(origin, target), "public graph end");
    };
    const auto launch_graph = [&](cudaGraphExec_t target, cudaStream_t stream) {
        return uses_driver ? driver_check(driver_replay(target, stream), "public Driver replay")
                           : check(replay(target, stream), "public graph replay");
    };
    const auto construct_graph = [&](cudaGraphExec_t* target, cudaGraph_t source) {
        return uses_driver ? driver_check(driver_construct(target, source, constructor_flags),
                                          "public Driver graph instantiate")
                           : check(cudaGraphInstantiate(target, source, constructor_flags),
                                   "public Runtime graph instantiate");
    };
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    KernelArguments arguments{output, 0, 7};
    auto* low_marker = device_markers;
    auto* high_marker = device_markers + 1;
    std::array<void*, 2> low_parameters{&arguments, &low_marker};
    std::array<void*, 1> high_parameters{&high_marker};
    const auto bytes = kCount * sizeof(unsigned int);
    KernelArguments prelude{output, 50000, 1};
    std::array<void*, 2> prelude_parameters{&prelude, &low_marker};
    if (!check(api.copy(output, input, bytes, cudaMemcpyHostToDevice, low), "pre-capture upload") ||
        !check(api.launch(reinterpret_cast<const void*>(low_kernel), dim3(kBlocks), dim3(kThreads),
                          prelude_parameters.data(), 0, low),
               "pre-capture low launch"))
        return false;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::atomic_ref<unsigned int>(markers[0]).load(std::memory_order_acquire) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    if (!check(cudaEventCreateWithFlags(&join, cudaEventDisableTiming), "graph join event") ||
        !begin_capture() ||
        !check(api.copy(output, input, bytes, cudaMemcpyHostToDevice, origin), "captured upload") ||
        !check(api.record(fork, origin), "captured fork record") ||
        !check(api.wait(branch, fork, 0), "captured branch wait") ||
        !check(api.launch(reinterpret_cast<const void*>(low_kernel), dim3(kBlocks), dim3(kThreads),
                          low_parameters.data(), 0, branch),
               "captured original low kernel") ||
        !check(api.record(join, branch), "captured join record") ||
        !check(api.wait(origin, join, 0), "captured origin wait") ||
        !check(api.launch(reinterpret_cast<const void*>(high_kernel), dim3(1), dim3(1),
                          high_parameters.data(), 0, origin),
               "captured original high kernel") ||
        !check(api.copy(readback, output, bytes, cudaMemcpyDeviceToHost, origin),
               "captured readback") ||
        !end_capture(&graph) || !check(api.synchronize(low), "pre-capture low completion")) {
        return false;
    }
    // Construction cannot execute the business kernel or copy the output.
    if (markers[0] != 1 || markers[1] != 0 || readback[0] != 0xdeadbeef) {
        std::cerr << "public capture executed construction work\n";
        return false;
    }
    std::size_t count{};
    std::array<cudaGraphNode_t, 4> nodes{};
    if (!check(cudaGraphGetNodes(graph, nullptr, &count), "original graph node count") ||
        count != nodes.size() ||
        !check(cudaGraphGetNodes(graph, nodes.data(), &count), "original graph nodes")) {
        std::cerr << "public capture changed original graph node count\n";
        return false;
    }
    unsigned int low_nodes{}, high_nodes{};
    for (const auto node : nodes) {
        cudaGraphNodeType type{};
        if (!check(cudaGraphNodeGetType(node, &type), "original node type")) return false;
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams parameters{};
        if (!check(cudaGraphKernelNodeGetParams(node, &parameters), "original kernel params"))
            return false;
        if (parameters.func == reinterpret_cast<void*>(low_kernel) &&
            parameters.gridDim.x == kBlocks && parameters.blockDim.x == kThreads)
            ++low_nodes;
        if (parameters.func == reinterpret_cast<void*>(high_kernel) && parameters.gridDim.x == 1 &&
            parameters.blockDim.x == 1)
            ++high_nodes;
    }
    if (low_nodes != 1 || high_nodes != 1 || !construct_graph(&executable, graph) ||
        !check(cudaGraphDestroy(graph), "public source graph destroy")) {
        std::cerr << "public capture changed original kernel identity\n";
        return false;
    }
    for (unsigned int repetition = 1; repetition <= 2; ++repetition) {
        if (!launch_graph(executable, origin) ||
            !check(api.synchronize(origin), "public graph physical completion") ||
            markers[1] != repetition ||
            !std::all_of(readback, readback + kCount, [](auto value) { return value == 7; })) {
            std::cerr << "public graph dependency or exact output mismatch\n";
            return false;
        }
    }
    if (!check(cudaGraphExecDestroy(executable), "public captured executable destroy") ||
        !check(cudaGraphCreate(&graph, 0), "public explicit graph create"))
        return false;
    cudaGraphNode_t node{};
    cudaKernelNodeParams parameters{};
    std::array<void*, 2> graph_parameters{&high_marker, &output};
    parameters.func = reinterpret_cast<void*>(graph_kernel);
    parameters.gridDim = dim3(1);
    parameters.blockDim = dim3(1);
    parameters.kernelParams = graph_parameters.data();
    if (!check(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &parameters),
               "public explicit original kernel node") ||
        !construct_graph(&executable, graph) ||
        !check(cudaGraphDestroy(graph), "public explicit source graph destroy") ||
        !launch_graph(executable, origin) ||
        !check(api.synchronize(origin), "public explicit graph completion") || markers[1] != 3 ||
        !begin_capture() ||
        !check(api.launch(reinterpret_cast<const void*>(high_kernel), dim3(1), dim3(1),
                          high_parameters.data(), 0, origin),
               "second captured original kernel") ||
        !launch_graph(executable, branch) || !end_capture(&graph) ||
        !check(api.synchronize(branch), "source uncaptured graph completion") || markers[1] != 4)
        return false;
    count = 0;
    cudaGraphExec_t nested_executable{};
    if (!check(cudaGraphGetNodes(graph, nullptr, &count), "captured graph-launch node count") ||
        count != 1 || !construct_graph(&nested_executable, graph) ||
        !check(cudaGraphExecDestroy(executable), "public explicit executable destroy") ||
        !check(cudaGraphDestroy(graph), "nested source graph destroy") ||
        !launch_graph(nested_executable, origin) ||
        !check(api.synchronize(origin), "nested whole-graph completion") || markers[1] != 5 ||
        !check(cudaGraphExecDestroy(nested_executable), "public nested executable destroy") ||
        !check(cudaEventDestroy(join), "public graph join event destroy") ||
        (driver_handle && dlclose(driver_handle) != 0))
        return false;
    markers[0] = 0;
    markers[1] = 0;
    std::fill_n(readback, kCount, 0xdeadbeef);
    return true;
}

bool run_round(std::string_view mode) {
    const bool uses_graph = mode.starts_with("--graph");
    const bool uses_driver_graph = mode.find("-driver") != std::string_view::npos;
    const unsigned long long graph_flags =
        mode.find("-flags") != std::string_view::npos
            ? cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority
            : 0;
    const bool uses_ptds = mode == "--ptds" || mode == "--async-release-ptds" ||
                           mode == "--graph-ptds" || mode == "--graph-driver-ptds" ||
                           mode == "--graph-null-ptds";
    const bool uses_async_release = mode == "--async-release" || mode == "--async-release-ptds";
    const bool releases_before_sync = mode == "--release";
    const bool uses_native_high = mode == "--native-high";
    const bool uses_compiler_launch = mode == "--compiler";
    const bool uses_fresh_function = mode == "--fresh-function";
    RuntimeApi api;
    if (uses_ptds && (!resolve(api.launch, "cudaLaunchKernel_ptsz") ||
                      !resolve(api.copy, "cudaMemcpyAsync_ptsz") ||
                      !resolve(api.record, "cudaEventRecord_ptsz") ||
                      !resolve(api.wait, "cudaStreamWaitEvent_ptsz") ||
                      !resolve(api.query, "cudaStreamQuery_ptsz") ||
                      !resolve(api.synchronize, "cudaStreamSynchronize_ptsz"))) {
        return false;
    }
    cudaStream_t low = nullptr;
    cudaStream_t high = nullptr;
    cudaStream_t consumer = nullptr;
    cudaEvent_t done = nullptr;
    unsigned int* output = nullptr;
    unsigned int* input = nullptr;
    unsigned int* readback = nullptr;
    unsigned int* waited = nullptr;
    unsigned int* markers = nullptr;
    unsigned int* device_markers = nullptr;
    if (!check(cudaSetDevice(0), "set device") ||
        !check(cudaStreamCreateWithPriority(&low, cudaStreamNonBlocking, 0), "low stream") ||
        !check(cudaStreamCreateWithPriority(&high, cudaStreamNonBlocking, -1), "high stream") ||
        !check(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "consumer stream") ||
        !check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "event") ||
        !check(cudaMalloc(&output, kCount * sizeof(unsigned int)), "output") ||
        !check(cudaMallocHost(&input, kCount * sizeof(unsigned int)), "input") ||
        !check(cudaMallocHost(&readback, kCount * sizeof(unsigned int)), "readback") ||
        !check(cudaMallocHost(&waited, kCount * sizeof(unsigned int)), "waited") ||
        !check(cudaHostAlloc(&markers, 2 * sizeof(unsigned int), cudaHostAllocMapped), "markers") ||
        !check(cudaHostGetDevicePointer(&device_markers, markers, 0), "mapped markers")) {
        return false;
    }
    cudaFuncAttributes attributes{};
    if (!check(cudaFuncGetAttributes(&attributes, low_kernel), "low attributes") ||
        (!uses_fresh_function &&
         !check(cudaFuncGetAttributes(&attributes, high_kernel), "high attributes"))) {
        return false;
    }
    std::fill_n(input, kCount, 0);
    std::fill_n(readback, kCount, 0xdeadbeef);
    std::fill_n(waited, kCount, 0xdeadbeef);
    markers[0] = 0;
    markers[1] = 0;
    const auto low_stream = uses_ptds ? nullptr : low;
    const auto graph_origin = mode == "--graph-null-ptds" ? nullptr : consumer;
    if (uses_graph &&
        !check_public_graphs(api, low, graph_origin, high, done, output, input, readback,
                             device_markers, markers, uses_ptds, uses_driver_graph, graph_flags))
        return false;
    auto release_async = &cudaFreeAsync;
    if (uses_async_release) {
        if (uses_ptds && !resolve(release_async, "cudaFreeAsync_ptsz")) {
            return false;
        }
        if (release_async(output + 1, consumer) != cudaErrorInvalidValue ||
            cudaGetLastError() != cudaErrorInvalidValue || cudaGetLastError() != cudaSuccess) {
            std::cerr << "invalid first Runtime free must preserve native immediate/last error\n";
            return false;
        }
    }
    const std::size_t copy_bytes = kCount * sizeof(unsigned int);
    for (const auto& invalid_copy :
         {std::pair{cudaMemcpyHostToDevice, std::pair<void*, const void*>{output, input}},
          std::pair{cudaMemcpyDeviceToHost, std::pair<void*, const void*>{readback, output}},
          std::pair{cudaMemcpyDeviceToDevice, std::pair<void*, const void*>{output, output}}}) {
        if (api.copy(invalid_copy.second.first, invalid_copy.second.second, copy_bytes + 1,
                     invalid_copy.first, low_stream) != cudaErrorInvalidValue ||
            cudaGetLastError() != cudaErrorInvalidValue || cudaGetLastError() != cudaSuccess) {
            std::cerr << "oversized Runtime copy must preserve native immediate/last error\n";
            return false;
        }
    }
    const auto submit_low = [&] {
        if (!check(api.copy(output, input, kCount * sizeof(unsigned int), cudaMemcpyHostToDevice,
                            low_stream),
                   "upload")) {
            return false;
        }
        KernelArguments arguments{output, 50000, 1};
        auto* low_marker = device_markers;
        std::array<void*, 2> parameters{&arguments, &low_marker};
        cudaError_t low_result = cudaSuccess;
        if (uses_compiler_launch) {
            low_kernel<<<kBlocks, kThreads, 0, low_stream>>>(arguments, low_marker);
            low_result = cudaGetLastError();
        } else {
            low_result = api.launch(reinterpret_cast<const void*>(low_kernel), dim3(kBlocks),
                                    dim3(kThreads), parameters.data(), 0, low_stream);
        }
        if (!check(low_result, "low launch") ||
            !check(cudaPeekAtLastError(), "accepted low last error")) {
            return false;
        }
        arguments = {};
        low_marker = nullptr;
        parameters.fill(nullptr);
        if (!check(api.copy(readback, output, kCount * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            low_stream),
                   "queued readback") ||
            !check(api.record(done, low_stream), "queued event") ||
            !check(api.wait(consumer, done, 0), "queued wait") ||
            !check(api.copy(waited, output, kCount * sizeof(unsigned int), cudaMemcpyDeviceToHost,
                            consumer),
                   "waited readback")) {
            return false;
        }
        return true;
    };
    bool submitted = false;
    if (mode == "--thread") {
        // No cudaSetDevice on this fresh caller: Runtime owns lazy binding.
        std::thread caller([&] { submitted = submit_low(); });
        caller.join();
    } else {
        submitted = submit_low();
    }
    if (!submitted) {
        return false;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (std::atomic_ref<unsigned int>(markers[0]).load(std::memory_order_acquire) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::cerr << "Runtime low did not publish application progress\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (cudaEventQuery(done) != cudaErrorNotReady || api.query(low_stream) != cudaErrorNotReady) {
        std::cerr << "Runtime logical work became ready prematurely\n";
        return false;
    }
    if (uses_async_release) {
        // Both readbacks were submitted before free; no new use of output is
        // legal after this API returns. A conservative host drain here would
        // prevent the next high launch from arriving before low completes.
        if (!check(release_async(output, consumer), "queued output release before high arrival") ||
            !check(cudaGetLastError(), "accepted async free last error")) {
            return false;
        }
        output = nullptr;
    }
    auto* high_marker = device_markers + 1;
    std::array<void*, 1> high_parameters{&high_marker};
    cudaError_t high_result = cudaSuccess;
    if (uses_compiler_launch) {
        (void)cudaGetLastError();
        high_kernel<<<1, 1, 0, high>>>(high_marker);
        high_result = cudaGetLastError();
    } else {
        high_result = api.launch(reinterpret_cast<const void*>(high_kernel), dim3(1), dim3(1),
                                 high_parameters.data(), uses_native_high ? 4 : 0, high);
    }
    if (!check(high_result, "high launch") || !check(api.synchronize(high), "high sync") ||
        markers[1] != 1) {
        std::cerr << "Runtime high did not execute exactly once\n";
        return false;
    }
    const bool checks_quota =
        uses_async_release && std::getenv("GLIMMER_MEMORY_LIMIT_BYTES") != nullptr;
    std::size_t retained_free{};
    std::size_t quota_total{};
    if (checks_quota &&
        !check(cudaMemGetInfo(&retained_free, &quota_total), "retained free quota")) {
        return false;
    }
    if (releases_before_sync) {
        if (!check(cudaFree(output), "pending output release") ||
            !check(cudaFreeHost(input), "pending host release")) {
            return false;
        }
        output = nullptr;
        input = nullptr;
    }
    if (!check(cudaEventSynchronize(done), "logical event sync") ||
        !check(api.synchronize(consumer), "logical consumer sync") ||
        !check(cudaDeviceSynchronize(), "logical device sync") ||
        !std::all_of(readback, readback + kCount, [](auto value) { return value == 1; }) ||
        !std::all_of(waited, waited + kCount, [](auto value) { return value == 1; })) {
        std::cerr << "Runtime exact logical output mismatch\n";
        return false;
    }
    if (checks_quota) {
        std::size_t recovered_free{};
        std::size_t recovered_total{};
        if (!check(cudaMemGetInfo(&recovered_free, &recovered_total), "completed free quota") ||
            recovered_total != quota_total || recovered_free != retained_free + copy_bytes) {
            std::cerr << "Runtime queued release must retain then recover its quota exactly once\n";
            return false;
        }
    }
    // Native Runtime maps the matched Driver rejection and owns last-error TLS.
    auto* invalid_marker = device_markers + 1;
    void* invalid_parameters[]{&invalid_marker};
    const auto invalid = api.launch(reinterpret_cast<const void*>(high_kernel), dim3(0), dim3(1),
                                    invalid_parameters, 0, high);
    if (invalid != cudaErrorInvalidConfiguration || cudaGetLastError() != invalid ||
        cudaGetLastError() != cudaSuccess) {
        std::cerr << "Runtime native last-error semantics changed\n";
        return false;
    }
    if (!check(cudaEventDestroy(done), "event destroy") ||
        !check(cudaStreamDestroy(low), "low stream destroy") ||
        !check(cudaStreamDestroy(high), "high stream destroy") ||
        !check(cudaStreamDestroy(consumer), "consumer stream destroy") ||
        (output && !check(cudaFree(output), "output release")) ||
        (input && !check(cudaFreeHost(input), "input release")) ||
        !check(cudaFreeHost(readback), "readback release") ||
        !check(cudaFreeHost(waited), "waited release") ||
        !check(cudaFreeHost(markers), "marker release") ||
        !check(cudaDeviceReset(), "device reset")) {
        return false;
    }
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string_view mode = argc > 1 ? argv[1] : "";
    if (!run_round(mode) || !run_round(mode)) {
        return EXIT_FAILURE;
    }
    std::cout << "runtime_cooperative_transparency status=ok logical_threads=" << kCount << '\n';
    return EXIT_SUCCESS;
}
