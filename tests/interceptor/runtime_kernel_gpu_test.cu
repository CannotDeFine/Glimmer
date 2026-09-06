#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

constexpr std::uint32_t kExpectedValue = 42;
constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;

__global__ void write_value(std::uint32_t* output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = kExpectedValue;
    }
}

bool check(cudaError_t result, std::string_view operation) {
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed with cudaError " << static_cast<int>(result) << ": "
              << cudaGetErrorString(result) << '\n';
    return false;
}

}  // namespace

int main() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    bool workload_succeeded =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo initial") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes;

    std::uint32_t* device_output = nullptr;
    workload_succeeded = check(cudaMalloc(&device_output, sizeof(kExpectedValue)), "cudaMalloc") &&
                         workload_succeeded;

    if (workload_succeeded) {
        workload_succeeded =
            check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo allocated") &&
            total_bytes == kQuotaBytes && free_bytes == kQuotaBytes - sizeof(kExpectedValue) &&
            workload_succeeded;
    }

    if (workload_succeeded) {
        const int launch_count = std::getenv("GLIMMER_TEST_BATCHED") == nullptr ? 1 : 2;
        for (int launch_index = 0; launch_index < launch_count && workload_succeeded;
             ++launch_index) {
            write_value<<<1, 1>>>(device_output);
            workload_succeeded = check(cudaGetLastError(), "kernel launch") && workload_succeeded;
        }
        workload_succeeded =
            check(cudaDeviceSynchronize(), "cudaDeviceSynchronize") && workload_succeeded;
    }

    cudaGraph_t graph = nullptr;
    if (workload_succeeded && std::getenv("GLIMMER_TEST_EXTENDED") != nullptr) {
        using ExtendedLaunch = cudaError_t (*)(const cudaLaunchConfig_t*, const void*, void**);
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeIgnore;
        cudaLaunchConfig_t config{};
        config.gridDim = dim3{1, 1, 1};
        config.blockDim = dim3{1, 1, 1};
        config.attrs = &attribute;
        config.numAttrs = 1;
        void* arguments[] = {&device_output};
        for (const char* name : {"cudaLaunchKernelExC", "cudaLaunchKernelExC_ptsz"}) {
            const auto launch = reinterpret_cast<ExtendedLaunch>(dlsym(RTLD_DEFAULT, name));
            workload_succeeded = launch != nullptr && workload_succeeded;
            if (launch != nullptr) {
                workload_succeeded = check(cudaMemset(device_output, 0, sizeof(kExpectedValue)),
                                           "cudaMemset extended") &&
                                     workload_succeeded;
                workload_succeeded =
                    check(launch(&config, reinterpret_cast<const void*>(write_value), arguments),
                          name) &&
                    workload_succeeded;
                std::uint32_t value = 0;
                workload_succeeded =
                    check(cudaDeviceSynchronize(), "extended synchronize") &&
                    check(cudaMemcpy(&value, device_output, sizeof(value), cudaMemcpyDeviceToHost),
                          "extended copy") &&
                    value == kExpectedValue && workload_succeeded;
            }
        }
    }
    cudaGraphExec_t graph_exec = nullptr;
    if (workload_succeeded) {
        workload_succeeded = check(cudaGraphCreate(&graph, 0), "cudaGraphCreate");
    }
    if (workload_succeeded) {
        void* graph_kernel_arguments[] = {&device_output};
        cudaKernelNodeParams graph_kernel_params{};
        graph_kernel_params.func = reinterpret_cast<void*>(write_value);
        graph_kernel_params.gridDim = dim3{1, 1, 1};
        graph_kernel_params.blockDim = dim3{1, 1, 1};
        graph_kernel_params.kernelParams = graph_kernel_arguments;
        cudaGraphNode_t graph_node = nullptr;
        workload_succeeded =
            check(cudaGraphAddKernelNode(&graph_node, graph, nullptr, 0, &graph_kernel_params),
                  "cudaGraphAddKernelNode");
    }
    if (workload_succeeded) {
        workload_succeeded =
            check(cudaGraphInstantiate(&graph_exec, graph, 0), "cudaGraphInstantiate");
    }
    if (workload_succeeded) {
        workload_succeeded = check(cudaGraphLaunch(graph_exec, nullptr), "cudaGraphLaunch") &&
                             check(cudaDeviceSynchronize(), "cudaDeviceSynchronize graph") &&
                             workload_succeeded;
    }
    if (graph_exec != nullptr) {
        workload_succeeded =
            check(cudaGraphExecDestroy(graph_exec), "cudaGraphExecDestroy") && workload_succeeded;
    }
    if (graph != nullptr) {
        workload_succeeded =
            check(cudaGraphDestroy(graph), "cudaGraphDestroy") && workload_succeeded;
    }

    std::uint32_t host_output = 0;
    if (workload_succeeded) {
        const cudaError_t copy_result =
            cudaMemcpy(&host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost);
        workload_succeeded =
            check(copy_result, "cudaMemcpy") && host_output == kExpectedValue && workload_succeeded;
    }

    const bool released = device_output == nullptr || check(cudaFree(device_output), "cudaFree");
    const bool restored =
        released && check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo restored") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes;
    if (!workload_succeeded || !released || !restored) {
        std::cerr << "test assertion failed: CUDA Runtime kernel workload validation or cleanup\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
