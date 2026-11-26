#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

// Ordinary nvcc kernels: no Glimmer interface or application-owned PTX.
extern "C" __global__ void runtime_write_indices(std::uint32_t* output, std::uint32_t count) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    output[index] = index + gridDim.x;
}

extern "C" __global__ void runtime_native_barrier(std::uint32_t* output) {
    __syncthreads();
    if (threadIdx.x == 0) {
        output[blockIdx.x] = 17;
    }
}

struct LaunchValues {
    std::uint32_t increment;
    std::uint32_t seed;
    std::uint32_t multiplier;
    std::uint32_t unused;
};

extern "C" __global__ void runtime_cooperative_3d(std::uint32_t* output, LaunchValues values) {
    __shared__ std::uint32_t shared_values[64];
    const auto thread = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const auto logical_block = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    const auto threads = blockDim.x * blockDim.y * blockDim.z;
    shared_values[thread] =
        logical_block * values.multiplier + thread + values.increment + gridDim.z * values.seed;
    __syncthreads();
    output[logical_block * threads + thread] += shared_values[63 - thread];
}

namespace {

constexpr std::uint32_t kThreads = 64;
constexpr std::uint32_t kCapacity = kThreads * 257 + 8;
constexpr std::uint32_t kSentinel = 0xffffffff;

bool check(cudaError_t result, std::string_view operation) {
    if (result != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(result) << '\n';
        return false;
    }
    return true;
}

struct Resources {
    std::array<cudaStream_t, 2> streams{};
    std::array<std::uint32_t*, 2> outputs{};
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaEvent_t event = nullptr;
    ~Resources() {
        (void)cleanup();
    }

    bool cleanup() {
        bool passed = true;
        if (event != nullptr) {
            passed &= check(cudaEventDestroy(event), "destroy dependency event");
            event = nullptr;
        }
        if (executable != nullptr) {
            passed &= check(cudaGraphExecDestroy(executable), "destroy executable");
            executable = nullptr;
        }
        if (graph != nullptr) {
            passed &= check(cudaGraphDestroy(graph), "destroy graph");
            graph = nullptr;
        }
        for (auto& output : outputs) {
            if (output != nullptr) {
                passed &= check(cudaFree(output), "free output");
                output = nullptr;
            }
        }
        for (auto& stream : streams) {
            if (stream != nullptr) {
                passed &= check(cudaStreamDestroy(stream), "destroy stream");
                stream = nullptr;
            }
        }
        return passed;
    }
};

bool validate(std::uint32_t* device_output, std::uint32_t count, std::uint32_t grid) {
    std::vector<std::uint32_t> output(kCapacity);
    if (!check(cudaMemcpy(output.data(), device_output, output.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "read output")) {
        return false;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto expected = index < count ? static_cast<std::uint32_t>(index) + grid : kSentinel;
        if (output[index] != expected) {
            std::cerr << "Runtime output mismatch index=" << index << " count=" << count << '\n';
            return false;
        }
    }
    return true;
}

bool run_cooperative(Resources& resources) {
    constexpr std::uint32_t kElements = 7 * 5 * 3 * kThreads;
    const LaunchValues values{11, 13, 17, 19};
    if (!check(cudaMemsetAsync(resources.outputs[0], 0, kCapacity * sizeof(std::uint32_t),
                               resources.streams[0]),
               "clear shared output") ||
        !check(cudaEventCreateWithFlags(&resources.event, cudaEventDisableTiming),
               "create dependency event")) {
        return false;
    }
    for (unsigned int repetition = 0; repetition < 2; ++repetition) {
        runtime_cooperative_3d<<<dim3{7, 5, 3}, dim3{8, 4, 2}, 0, resources.streams[0]>>>(
            resources.outputs[0], values);
        if (!check(cudaGetLastError(), "ordinary shared 3D launch")) {
            return false;
        }
    }
    std::vector<std::uint32_t> output(kElements + 8);
    if (!check(cudaEventRecord(resources.event, resources.streams[0]),
               "record shared completion") ||
        !check(cudaStreamWaitEvent(resources.streams[1], resources.event, 0),
               "wait for logical completion") ||
        !check(cudaMemcpyAsync(resources.outputs[1], resources.outputs[0],
                               output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
                               resources.streams[1]),
               "dependent device copy") ||
        !check(cudaStreamSynchronize(resources.streams[1]), "finish dependent copy") ||
        !check(cudaEventQuery(resources.event), "query logical completion") ||
        !check(cudaMemcpy(output.data(), resources.outputs[1],
                          output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "read shared results")) {
        return false;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto expected =
            index < kElements
                ? 2U * (static_cast<std::uint32_t>(index / kThreads) * values.multiplier + 63U -
                        static_cast<std::uint32_t>(index % kThreads) + values.increment +
                        3U * values.seed)
                : 0U;
        if (output[index] != expected) {
            std::cerr << "Runtime shared 3D output mismatch at " << index << '\n';
            return false;
        }
    }
    return true;
}

bool run_cooperative_capture(Resources& resources) {
    const LaunchValues values{11, 13, 17, 19};
    if (!check(cudaStreamBeginCapture(resources.streams[0], cudaStreamCaptureModeThreadLocal),
               "capture shared 3D work")) {
        return false;
    }
    runtime_cooperative_3d<<<dim3{7, 5, 3}, dim3{8, 4, 2}, 0, resources.streams[0]>>>(
        resources.outputs[0], values);
    const auto launch_result = cudaGetLastError();
    const auto capture_result = cudaStreamEndCapture(resources.streams[0], &resources.graph);
    cudaGraphNode_t node = nullptr;
    std::size_t node_count = 1;
    cudaKernelNodeParams parameters{};
    if (!check(launch_result, "capture shared launch") ||
        !check(capture_result, "end shared capture") ||
        !check(cudaGraphGetNodes(resources.graph, &node, &node_count), "get shared graph node") ||
        node_count != 1 ||
        !check(cudaGraphKernelNodeGetParams(node, &parameters), "inspect shared graph") ||
        parameters.func != reinterpret_cast<void*>(runtime_cooperative_3d) ||
        parameters.gridDim.x != 7 || parameters.gridDim.y != 5 || parameters.gridDim.z != 3 ||
        parameters.blockDim.x != 8 || parameters.blockDim.y != 4 || parameters.blockDim.z != 2 ||
        !check(cudaGraphInstantiate(&resources.executable, resources.graph, 0),
               "instantiate shared graph") ||
        !check(cudaGraphLaunch(resources.executable, resources.streams[0]),
               "launch shared graph") ||
        !check(cudaStreamSynchronize(resources.streams[0]), "finish shared graph")) {
        return false;
    }
    constexpr std::uint32_t kElements = 7 * 5 * 3 * kThreads;
    std::vector<std::uint32_t> output(kElements + 8);
    if (!check(cudaMemcpy(output.data(), resources.outputs[0],
                          output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "read shared graph")) {
        return false;
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
        const auto expected =
            index < kElements
                ? 3U * (static_cast<std::uint32_t>(index / kThreads) * values.multiplier + 63U -
                        static_cast<std::uint32_t>(index % kThreads) + values.increment +
                        3U * values.seed)
                : 0U;
        if (output[index] != expected) {
            return false;
        }
    }
    if (!check(cudaGraphExecDestroy(resources.executable), "destroy shared executable")) {
        return false;
    }
    resources.executable = nullptr;
    if (!check(cudaGraphDestroy(resources.graph), "destroy shared graph")) {
        return false;
    }
    resources.graph = nullptr;
    return true;
}

bool run_lazy_runtime_thread(Resources& resources) {
    using ContextQuery = CUresult (*)(CUcontext*);
    void* driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (driver == nullptr) {
        return false;
    }
    const auto query = reinterpret_cast<ContextQuery>(dlsym(driver, "cuCtxGetCurrent"));
    bool passed = false;
    std::thread worker([&] {
        CUcontext context = nullptr;
        if (query == nullptr || query(&context) != CUDA_SUCCESS || context != nullptr) {
            std::cerr << "new Runtime thread did not start without a bound CUDA context\n";
            return;
        }
        std::uint32_t count = 1025;
        void* arguments[] = {&resources.outputs[0], &count};
        passed =
            check(cudaLaunchKernel(reinterpret_cast<const void*>(runtime_write_indices), dim3{17},
                                   dim3{kThreads}, arguments, 0, resources.streams[0]),
                  "launch on unbound Runtime thread") &&
            check(cudaGetLastError(), "lazy Runtime thread last error");
    });
    worker.join();
    const bool library_closed = dlclose(driver) == 0;
    return passed && library_closed &&
           check(cudaStreamSynchronize(resources.streams[0]), "finish lazy thread launch") &&
           validate(resources.outputs[0], 1025, 17);
}

bool run(bool should_capture) {
    Resources resources;
    for (std::size_t index = 0; index < resources.outputs.size(); ++index) {
        if (!check(cudaStreamCreateWithFlags(&resources.streams[index], cudaStreamNonBlocking),
                   "create stream") ||
            !check(cudaMalloc(&resources.outputs[index], kCapacity * sizeof(std::uint32_t)),
                   "allocate output")) {
            return false;
        }
    }
    if (!check(cudaMemsetAsync(resources.outputs[0], 0xff, kCapacity * sizeof(std::uint32_t),
                               resources.streams[0]),
               "initialize lazy thread output") ||
        !run_lazy_runtime_thread(resources)) {
        return false;
    }
    using Launch = cudaError_t (*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    const auto legacy = reinterpret_cast<Launch>(dlsym(RTLD_DEFAULT, "cudaLaunchKernel"));
    if (legacy == nullptr) {
        return false;
    }
    for (const std::uint32_t count : {1U, 65U, 1025U, kThreads * 257 - 11}) {
        const auto grid = (count + kThreads - 1) / kThreads;
        for (std::size_t index = 0; index < resources.outputs.size(); ++index) {
            const auto stream = index == 1 && count == 1025 ? nullptr : resources.streams[index];
            // This fixture uses PTDS for ordinary nvcc launches.
            const auto initialization_stream = stream == nullptr ? cudaStreamPerThread : stream;
            if (!check(cudaMemsetAsync(resources.outputs[index], 0xff,
                                       kCapacity * sizeof(std::uint32_t), initialization_stream),
                       "initialize output")) {
                return false;
            }
            if (index == 0) {
                void* arguments[] = {&resources.outputs[index], const_cast<std::uint32_t*>(&count)};
                if (!check(legacy(reinterpret_cast<const void*>(runtime_write_indices), dim3{grid},
                                  dim3{kThreads}, arguments, 0, stream),
                           "legacy public launch")) {
                    return false;
                }
            } else {
                runtime_write_indices<<<grid, kThreads, 0, stream>>>(resources.outputs[index],
                                                                     count);
                if (!check(cudaGetLastError(), "ordinary nvcc launch")) {
                    return false;
                }
            }
        }
        if (!check(cudaDeviceSynchronize(), "finish Runtime work")) {
            return false;
        }
        for (auto* output : resources.outputs) {
            if (!validate(output, count, grid)) {
                return false;
            }
        }
    }
    runtime_native_barrier<<<17, kThreads, 0, resources.streams[0]>>>(resources.outputs[0]);
    std::array<std::uint32_t, 17> barrier_output{};
    if (!check(cudaGetLastError(), "barrier launch") ||
        !check(cudaStreamSynchronize(resources.streams[0]), "finish native barrier") ||
        !check(cudaMemcpy(barrier_output.data(), resources.outputs[0], sizeof(barrier_output),
                          cudaMemcpyDeviceToHost),
               "read native barrier")) {
        return false;
    }
    for (const auto value : barrier_output) {
        if (value != 17) {
            return false;
        }
    }
    if (!run_cooperative(resources) || (should_capture && !run_cooperative_capture(resources))) {
        return false;
    }
    // Compare Runtime error-state behavior, not only successful Driver results.
    (void)cudaGetLastError();
    std::uint32_t error_count = 1025;
    void* error_arguments[] = {&resources.outputs[0], &error_count};
    const auto invalid = legacy(reinterpret_cast<const void*>(runtime_write_indices), dim3{17},
                                dim3{0}, error_arguments, 0, resources.streams[0]);
    if (invalid == cudaSuccess || cudaPeekAtLastError() != invalid ||
        cudaGetLastError() != invalid || cudaPeekAtLastError() != cudaSuccess) {
        std::cerr << "Runtime last-error semantics changed\n";
        return false;
    }
    if (!check(legacy(reinterpret_cast<const void*>(runtime_write_indices), dim3{17},
                      dim3{kThreads}, error_arguments, 0, resources.streams[0]),
               "valid launch after native error") ||
        !check(cudaGetLastError(), "Runtime error state after valid variant") ||
        !check(cudaStreamSynchronize(resources.streams[0]), "finish post-error launch")) {
        return false;
    }
    if (should_capture) {
        if (!check(cudaMemsetAsync(resources.outputs[0], 0xff, kCapacity * sizeof(std::uint32_t),
                                   resources.streams[0]),
                   "initialize graph output") ||
            !check(cudaStreamSynchronize(resources.streams[0]), "finish graph initialization") ||
            !check(cudaStreamBeginCapture(resources.streams[0], cudaStreamCaptureModeGlobal),
                   "begin capture")) {
            return false;
        }
        constexpr std::size_t kCapturedLaunches = 8;
        for (std::size_t launch = 0; launch < kCapturedLaunches; ++launch) {
            runtime_write_indices<<<17, kThreads, 0, resources.streams[0]>>>(resources.outputs[0],
                                                                             1025);
        }
        const auto launch_result = cudaGetLastError();
        const auto capture_result = cudaStreamEndCapture(resources.streams[0], &resources.graph);
        std::size_t node_count = kCapturedLaunches;
        std::array<cudaGraphNode_t, kCapturedLaunches> nodes{};
        if (!check(launch_result, "captured launch") || !check(capture_result, "end capture") ||
            !check(cudaGraphGetNodes(resources.graph, nodes.data(), &node_count),
                   "get graph nodes") ||
            node_count != kCapturedLaunches) {
            return false;
        }
        for (const auto node : nodes) {
            cudaKernelNodeParams parameters{};
            if (!check(cudaGraphKernelNodeGetParams(node, &parameters),
                       "get original graph kernel") ||
                parameters.func != reinterpret_cast<void*>(runtime_write_indices) ||
                parameters.gridDim.x != 17 || parameters.blockDim.x != kThreads) {
                return false;
            }
        }
        if (!check(cudaGraphInstantiate(&resources.executable, resources.graph, 0),
                   "instantiate graph") ||
            !check(cudaGraphLaunch(resources.executable, resources.streams[0]), "execute graph") ||
            !check(cudaStreamSynchronize(resources.streams[0]), "finish graph") ||
            !validate(resources.outputs[0], 1025, 17)) {
            return false;
        }
    }
    return resources.cleanup();
}

}  // namespace

int main(int argc, char** argv) {
    const bool should_capture = argc == 2 && std::string_view(argv[1]) == "--capture";
    if (!check(cudaSetDevice(0), "initialize Runtime outside intercepted launch/allocation")) {
        return EXIT_FAILURE;
    }
    for (int round = 0; round < 2; ++round) {
        if (!run(should_capture) || !check(cudaDeviceReset(), "reset Runtime context")) {
            return EXIT_FAILURE;
        }
    }
    std::cout << "persistent_runtime_transparency status=ok\n";
    return EXIT_SUCCESS;
}
