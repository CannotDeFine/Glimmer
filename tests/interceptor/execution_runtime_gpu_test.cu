#include <cuda_runtime.h>

#include "execution_binary_kernel_fixture.cu"
#include "execution_cached_runtime_fixture.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned int kWorkers = 320, kThreads = 32, kLogicalBlocks = 65536;
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(status));
}
void run(bool is_low, bool is_cached) {
    ExecutionCachedRuntime runtime;
    if (is_cached) runtime.enable();
    check(runtime.select(0), "select device");
    check(runtime.release(nullptr), "lazy Runtime initialization");
    if (is_cached) {
        CUcontext context{};
        CUdeviceptr scratch{};
        if (runtime.current(&context) != CUDA_SUCCESS || !context ||
            runtime.driver_allocate(&scratch, 16) != CUDA_SUCCESS)
            throw std::runtime_error("Runtime-cached Driver allocation/current context");
        check(runtime.release(reinterpret_cast<void*>(scratch)),
              "release Runtime-cached Driver allocation");
    }
    const auto count = (is_low ? kLogicalBlocks : kWorkers) * kThreads;
    std::vector<std::uint32_t> values(count);
    std::uint32_t* output{};
    check(runtime.allocate(reinterpret_cast<void**>(&output), values.size() * sizeof(values[0])),
          "allocate");
    cudaStream_t stream{};
    check(runtime.create_stream(&stream, cudaStreamNonBlocking), "create stream");
    check(runtime.copy(output, values.data(), values.size() * sizeof(values[0]),
                       cudaMemcpyHostToDevice),
          "upload");
    // Ordinary application warmup materializes the exact host-function binding.
    // Both native and preload execute it; it is outside the priority phase.
    if (is_cached) {
        std::uint32_t count_zero{};
        ExecutionBinaryPair pair{};
        unsigned long long duration{};
        void* low_arguments[]{&output, &count_zero, &pair};
        void* high_arguments[]{&output, &duration};
        check(runtime.launch(is_low ? reinterpret_cast<const void*>(&binary_add)
                                    : reinterpret_cast<const void*>(&binary_high),
                             dim3{is_low ? 1U : kWorkers}, dim3{kThreads},
                             is_low ? low_arguments : high_arguments, 0, stream),
              "cached warmup");
    } else if (is_low)
        binary_add<<<1, kThreads, 0, stream>>>(output, 0, ExecutionBinaryPair{});
    else
        binary_high<<<kWorkers, kThreads, 0, stream>>>(output, 0);
    check(runtime.last_error(), "warmup launch");
    check(runtime.synchronize_stream(stream), "warmup completion");
    check(runtime.copy(output, values.data(), values.size() * sizeof(values[0]),
                       cudaMemcpyHostToDevice),
          "restore input");
    const auto* role = is_low ? "low" : "high";
    std::cout << "Ordinary Runtime application ready: role=" << role << '\n' << std::flush;
    std::string signal;
    if (!std::getline(std::cin, signal) || signal != "run")
        throw std::runtime_error("missing run signal");
    if (is_low) {
        ExecutionBinaryPair parameters{1, 8192, {0, 0}};
        if (is_cached) {
            auto argument_count = count;
            void* arguments[]{&output, &argument_count, &parameters};
            check(runtime.launch(reinterpret_cast<const void*>(&binary_add), dim3{kLogicalBlocks},
                                 dim3{kThreads}, arguments, 0, stream),
                  "cached low launch");
        } else
            binary_add<<<kLogicalBlocks, kThreads, 0, stream>>>(output, count, parameters);
        check(runtime.last_error(), "low launch");
        parameters.add = 99;  // The accepted launch already owns its by-value bytes.
    } else {
        for (unsigned int index = 0; index < 2; ++index) {
            if (is_cached) {
                unsigned long long duration = 100000000;
                void* arguments[]{&output, &duration};
                check(runtime.launch(reinterpret_cast<const void*>(&binary_high), dim3{kWorkers},
                                     dim3{kThreads}, arguments, 0, stream),
                      "cached high launch");
            } else
                binary_high<<<kWorkers, kThreads, 0, stream>>>(output, 100000000);
            check(runtime.last_error(), "high launch");
        }
    }
    check(runtime.synchronize_stream(stream), "logical stream completion");
    check(runtime.copy(values.data(), output, values.size() * sizeof(values[0]), cudaMemcpyDefault),
          "download");
    for (const auto value : values)
        if (value != (is_low ? 1U : 2U))
            throw std::runtime_error("once-only Runtime output mismatch");
    check(runtime.synchronize_device(), "device completion");
    check(runtime.release(output), "free output");
    check(runtime.destroy_stream(stream), "destroy stream");
    check(runtime.reset(), "reset actual owned generation");
    if (!runtime.close()) throw std::runtime_error("close exact SDK cache reference");
    std::cout << "Ordinary Runtime priority application verified: role=" << role
              << "; elements=" << count
              << "; entry=" << (is_cached ? "runtime-cached-launch" : "nvcc-triple-chevron")
              << "; workers=" << kWorkers << "; cached_runtime=" << (is_cached ? "true" : "false")
              << '\n';
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if ((argc != 2 && argc != 3) ||
            (std::string(argv[1]) != "--low" && std::string(argv[1]) != "--high") ||
            (argc == 3 && std::string(argv[2]) != "--cached-lookups"))
            throw std::runtime_error("select --low or --high");
        run(std::string(argv[1]) == "--low", argc == 3);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
