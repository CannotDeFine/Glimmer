#include "persistent_ptx_fixture.h"
#include "cooperative_ptx_fixture.h"

#include <cuda.h>
#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kThreads = 64;
constexpr std::uint32_t kCapacity = kThreads * 257 + 8;
constexpr std::uint32_t kSentinel = 0xdeadbeef;

bool check(CUresult result, std::string_view operation) {
    if (result != CUDA_SUCCESS) {
        std::cerr << operation << " failed with CUresult " << static_cast<int>(result) << '\n';
        return false;
    }
    return true;
}

struct Resources {
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    std::array<CUstream, 2> streams{};
    std::array<CUdeviceptr, 2> outputs{};

    ~Resources() {
        (void)cleanup();
    }

    bool cleanup() {
        bool passed = true;
        for (auto& output : outputs) {
            if (output != 0) {
                passed &= check(cuMemFree(output), "free output");
                output = 0;
            }
        }
        for (auto& stream : streams) {
            if (stream != nullptr) {
                passed &= check(cuStreamDestroy(stream), "destroy stream");
                stream = nullptr;
            }
        }
        if (module != nullptr) {
            passed &= check(cuModuleUnload(module), "unload module");
            module = nullptr;
        }
        if (context != nullptr) {
            passed &= check(cuCtxDestroy(context), "destroy context");
            context = nullptr;
        }
        return passed;
    }
};

using LoadModule = CUresult (*)(CUmodule*, const void*);
using GetFunction = CUresult (*)(CUfunction*, CUmodule, const char*);
using Launch = CUresult (*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int, CUstream, void**, void**);

bool load(Resources& resources, unsigned int path, const void* image = kPersistentTestPtx) {
    if (path == 0) {
        return check(cuModuleLoadData(&resources.module, image), "load PTX");
    }
    if (path == 1) {
        return check(cuModuleLoadDataEx(&resources.module, image, 0, nullptr, nullptr),
                     "load PTX with zero JIT options");
    }
    if (path == 2) {
        void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (driver == nullptr) {
            std::cerr << "open Driver failed\n";
            return false;
        }
        const auto loader = reinterpret_cast<LoadModule>(dlsym(driver, "cuModuleLoadData"));
        const bool passed =
            loader != nullptr && check(loader(&resources.module, image), "dlsym load PTX");
        return dlclose(driver) == 0 && passed;
    }
    void* address = nullptr;
    if (!check(cuGetProcAddress("cuModuleLoadData", &address, CUDA_VERSION,
                                CU_GET_PROC_ADDRESS_DEFAULT, nullptr),
               "resolve module loader") ||
        address == nullptr) {
        return false;
    }
    return check(reinterpret_cast<LoadModule>(address)(&resources.module, image),
                 "proc-address load PTX");
}

bool run_cases(Resources& resources) {
    void* address = nullptr;
    if (!check(cuGetProcAddress("cuModuleGetFunction", &address, CUDA_VERSION,
                                CU_GET_PROC_ADDRESS_DEFAULT, nullptr),
               "resolve function lookup") ||
        address == nullptr) {
        return false;
    }
    CUfunction function = nullptr;
    if (!check(reinterpret_cast<GetFunction>(address)(&function, resources.module, "write_indices"),
               "find application function")) {
        return false;
    }
    std::array<Launch, 2> launchers{};
    for (std::size_t index = 0; index < launchers.size(); ++index) {
        const auto flags = index == 0 ? CU_GET_PROC_ADDRESS_DEFAULT
                                      : CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
        if (!check(cuGetProcAddress("cuLaunchKernel", &address, CUDA_VERSION, flags, nullptr),
                   "resolve kernel launch") ||
            address == nullptr) {
            return false;
        }
        launchers[index] = reinterpret_cast<Launch>(address);
    }
    std::vector<std::uint32_t> result(kCapacity);
    for (const auto count : {1U, 17U, 64U, 65U, 255U, 1025U, kThreads * 257 - 11}) {
        std::array<std::uint32_t, 2> counts{count, count + 3};
        std::array<std::uint32_t, 2> grids{};
        for (std::size_t index = 0; index < resources.outputs.size(); ++index) {
            grids[index] = (counts[index] + kThreads - 1) / kThreads;
            void* arguments[] = {&resources.outputs[index], &counts[index]};
            // Also exercise the PTDS default stream, without application changes.
            const auto stream = index == 1 && count == 1025 ? nullptr : resources.streams[index];
            const auto initialization_stream = stream == nullptr ? CU_STREAM_PER_THREAD : stream;
            if (!check(cuMemsetD32Async(resources.outputs[index], kSentinel, kCapacity,
                                        initialization_stream),
                       "initialize output on launch stream") ||
                !check(launchers[index](function, grids[index], 1, 1, kThreads, 1, 1, 0, stream,
                                        arguments, nullptr),
                       "launch application kernel")) {
                return false;
            }
        }
        if (!check(cuCtxSynchronize(), "synchronize application work")) {
            return false;
        }
        for (std::size_t index = 0; index < resources.outputs.size(); ++index) {
            if (!check(cuMemcpyDtoH(result.data(), resources.outputs[index],
                                    result.size() * sizeof(std::uint32_t)),
                       "read output")) {
                return false;
            }
            for (std::size_t element = 0; element < result.size(); ++element) {
                const auto expected = element < counts[index]
                                          ? static_cast<std::uint32_t>(element) + grids[index]
                                          : kSentinel;
                if (result[element] != expected) {
                    std::cerr << "output failed at " << element << " count=" << counts[index]
                              << " grid=" << grids[index] << " value=" << result[element]
                              << " expected=" << expected << '\n';
                    return false;
                }
            }
        }
    }
    CUfunction barrier = nullptr;
    void* arguments[] = {&resources.outputs[0]};
    if (!check(cuModuleGetFunction(&barrier, resources.module, "native_barrier"),
               "find native entry") ||
        !check(cuLaunchKernel(barrier, 17, 1, 1, 64, 1, 1, 0, resources.streams[0], arguments,
                              nullptr),
               "launch unsupported native entry") ||
        !check(cuStreamSynchronize(resources.streams[0]), "synchronize native entry") ||
        !check(cuMemcpyDtoH(result.data(), resources.outputs[0], 17 * sizeof(std::uint32_t)),
               "read native output")) {
        return false;
    }
    for (std::size_t index = 0; index < 17; ++index) {
        if (result[index] != 17) {
            std::cerr << "native fallback validation failed\n";
            return false;
        }
    }
    return true;
}

struct GraphResources {
    CUgraph graph = nullptr;
    CUgraphExec executable = nullptr;
    ~GraphResources() {
        (void)cleanup();
    }
    bool cleanup() {
        bool passed = true;
        if (executable != nullptr) {
            passed &= check(cuGraphExecDestroy(executable), "destroy graph executable");
            executable = nullptr;
        }
        if (graph != nullptr) {
            passed &= check(cuGraphDestroy(graph), "destroy graph");
            graph = nullptr;
        }
        return passed;
    }
};

struct EventResource {
    CUevent handle = nullptr;
    ~EventResource() {
        (void)cleanup();
    }
    bool cleanup() {
        if (handle == nullptr) {
            return true;
        }
        const auto result = cuEventDestroy(handle);
        if (result == CUDA_SUCCESS) {
            handle = nullptr;
        }
        return check(result, "destroy 3D completion event");
    }
};

bool run_cooperative(Resources& resources, bool should_capture) {
    constexpr std::uint32_t k_elements = 7 * 5 * 3 * kThreads;
    CUfunction function = nullptr;
    CUdeviceptr marker = resources.outputs[0] + (kCapacity - 1) * sizeof(std::uint32_t);
    std::uint32_t work = 0;
    void* arguments[] = {&resources.outputs[0], &marker, &work};
    if (!check(cuModuleGetFunction(&function, resources.module, "cooperative"),
               "find 3D shared entry") ||
        !check(cuMemsetD32Async(resources.outputs[0], 0, kCapacity, resources.streams[0]),
               "clear 3D output")) {
        return false;
    }
    for (unsigned int repetition = 0; repetition < 2; ++repetition) {
        if (!check(cuLaunchKernel(function, 7, 5, 3, 8, 4, 2, 0, resources.streams[0], arguments,
                                  nullptr),
                   "ordinary 3D shared launch")) {
            return false;
        }
    }
    EventResource event;
    if (!check(cuEventCreate(&event.handle, CU_EVENT_DISABLE_TIMING),
               "create 3D completion event")) {
        return false;
    }
    std::vector<std::uint32_t> result(k_elements + 8);
    bool passed =
        check(cuEventRecord(event.handle, resources.streams[0]), "record 3D completion") &&
        check(cuStreamWaitEvent(resources.streams[1], event.handle, 0), "wait for 3D work") &&
        check(cuMemcpyDtoDAsync(resources.outputs[1], resources.outputs[0],
                                result.size() * sizeof(std::uint32_t), resources.streams[1]),
              "cross-stream 3D copy") &&
        check(cuStreamSynchronize(resources.streams[1]), "finish dependent 3D copy") &&
        check(cuMemcpyDtoH(result.data(), resources.outputs[1],
                           result.size() * sizeof(std::uint32_t)),
              "read dependent 3D copy") &&
        check(cuEventQuery(event.handle), "query completed 3D event");
    passed &= event.cleanup();
    if (!passed) {
        return false;
    }
    for (std::size_t element = 0; element < result.size(); ++element) {
        if (result[element] != (element < k_elements ? 2U : 0U)) {
            std::cerr << "Driver 3D shared output mismatch at " << element << '\n';
            return false;
        }
    }
    if (should_capture) {
        GraphResources graph;
        if (!check(cuStreamBeginCapture(resources.streams[0], CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                   "capture 3D work")) {
            return false;
        }
        const auto launch_result =
            cuLaunchKernel(function, 7, 5, 3, 8, 4, 2, 0, resources.streams[0], arguments, nullptr);
        const auto capture_result = cuStreamEndCapture(resources.streams[0], &graph.graph);
        CUgraphNode node = nullptr;
        std::size_t node_count = 1;
        CUDA_KERNEL_NODE_PARAMS parameters{};
        if (!check(launch_result, "captured 3D launch") ||
            !check(capture_result, "end 3D capture") ||
            !check(cuGraphGetNodes(graph.graph, &node, &node_count), "get 3D graph node") ||
            node_count != 1 ||
            !check(cuGraphKernelNodeGetParams(node, &parameters), "inspect 3D graph") ||
            parameters.func != function || parameters.gridDimX != 7 || parameters.gridDimY != 5 ||
            parameters.gridDimZ != 3 || parameters.blockDimX != 8 || parameters.blockDimY != 4 ||
            parameters.blockDimZ != 2 ||
            !check(cuGraphInstantiateWithFlags(&graph.executable, graph.graph, 0),
                   "instantiate 3D graph") ||
            !check(cuGraphLaunch(graph.executable, resources.streams[0]), "execute 3D graph") ||
            !check(cuStreamSynchronize(resources.streams[0]), "finish 3D graph") ||
            !check(cuMemcpyDtoH(result.data(), resources.outputs[0],
                                result.size() * sizeof(std::uint32_t)),
                   "read 3D graph results")) {
            return false;
        }
        for (std::size_t element = 0; element < result.size(); ++element) {
            if (result[element] != (element < k_elements ? 3U : 0U)) {
                return false;
            }
        }
        if (!graph.cleanup()) {
            return false;
        }
    }
    return true;
}

bool run_capture(Resources& resources) {
    GraphResources graph;
    CUfunction function = nullptr;
    std::uint32_t count = 1025;
    void* arguments[] = {&resources.outputs[0], &count};
    if (!check(cuModuleGetFunction(&function, resources.module, "write_indices"),
               "find capture kernel") ||
        !check(cuMemsetD32Async(resources.outputs[0], kSentinel, kCapacity, resources.streams[0]),
               "initialize captured output") ||
        !check(cuStreamSynchronize(resources.streams[0]), "finish output initialization") ||
        !check(cuStreamBeginCapture(resources.streams[0], CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
               "begin capture")) {
        return false;
    }
    const auto launch_result = cuLaunchKernel(function, 17, 1, 1, kThreads, 1, 1, 0,
                                              resources.streams[0], arguments, nullptr);
    const auto capture_result = cuStreamEndCapture(resources.streams[0], &graph.graph);
    if (!check(launch_result, "capture launch") || !check(capture_result, "end capture")) {
        return false;
    }
    std::size_t node_count = 0;
    std::array<CUgraphNode, 1> nodes{};
    CUDA_KERNEL_NODE_PARAMS parameters{};
    if (!check(cuGraphGetNodes(graph.graph, nullptr, &node_count), "count captured nodes") ||
        node_count != 1 ||
        !check(cuGraphGetNodes(graph.graph, nodes.data(), &node_count), "get captured node") ||
        !check(cuGraphKernelNodeGetParams(nodes[0], &parameters), "get captured parameters") ||
        parameters.func != function || parameters.gridDimX != 17 ||
        parameters.blockDimX != kThreads) {
        std::cerr << "capture must retain original application function and dimensions\n";
        return false;
    }
    if (!check(cuGraphInstantiateWithFlags(&graph.executable, graph.graph, 0),
               "instantiate graph") ||
        !check(cuGraphLaunch(graph.executable, resources.streams[0]), "launch graph") ||
        !check(cuStreamSynchronize(resources.streams[0]), "complete graph")) {
        return false;
    }
    std::vector<std::uint32_t> result(count + 8);
    if (!check(cuMemcpyDtoH(result.data(), resources.outputs[0],
                            result.size() * sizeof(std::uint32_t)),
               "read captured results")) {
        return false;
    }
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto expected = index < count ? static_cast<std::uint32_t>(index) + 17 : kSentinel;
        if (result[index] != expected) {
            std::cerr << "captured graph result validation failed\n";
            return false;
        }
    }
    return graph.cleanup();
}

bool run(bool should_capture) {
    Resources resources;
    CUdevice device = 0;
    if (!check(cuInit(0), "initialize Driver") || !check(cuDeviceGet(&device, 0), "get device") ||
        !check(cuCtxCreate(&resources.context, 0, device), "create context")) {
        return false;
    }
    for (std::size_t index = 0; index < resources.outputs.size(); ++index) {
        if (!check(cuStreamCreate(&resources.streams[index], CU_STREAM_NON_BLOCKING),
                   "create stream") ||
            !check(cuMemAlloc(&resources.outputs[index], kCapacity * sizeof(std::uint32_t)),
                   "allocate output")) {
            return false;
        }
    }
    for (unsigned int path = 0; path < 4; ++path) {
        if (!load(resources, path) || !run_cases(resources) ||
            (should_capture && !run_capture(resources)) ||
            !check(cuModuleUnload(resources.module), "unload before reload")) {
            return false;
        }
        resources.module = nullptr;
        if (!load(resources, path, kCooperativePtx) ||
            !run_cooperative(resources, should_capture) ||
            !check(cuModuleUnload(resources.module), "unload 3D module")) {
            return false;
        }
        resources.module = nullptr;
    }
    return resources.cleanup();
}

}  // namespace

int main(int argc, char** argv) {
    const bool should_capture = argc == 2 && std::string_view(argv[1]) == "--capture";
    const bool passed = run(should_capture);
    std::cout << "persistent_transparency status=" << (passed ? "ok" : "failed") << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
