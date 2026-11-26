#include <cuda.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <poll.h>
#include <unistd.h>

namespace {
void check(CUresult status) {
    if (status != CUDA_SUCCESS) throw std::runtime_error("Driver " + std::to_string(status));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS " + std::to_string(status));
}
void run(bool is_device_tool, bool device_scalars) {
    const int dimension = is_device_tool ? 2048 : 8192;
    constexpr int depth = 4096;
    check(cuInit(0));
    CUdevice device{};
    CUcontext context{};
    check(cuDeviceGet(&device, 0));
    check(cuCtxCreate(&context, 0, device));
    cublasHandle_t handle{};
    check(cublasCreate_v2(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cublasSetStream_v2(handle, reinterpret_cast<cudaStream_t>(stream)));
    const auto input_elements = static_cast<std::size_t>(dimension) * depth;
    const auto output_elements = static_cast<std::size_t>(dimension) * dimension;
    constexpr std::size_t guard = 32;
    std::vector<float> input(input_elements + guard, 1.0F), output(output_elements + guard, 6.0F);
    CUdeviceptr a{}, b{}, c{}, scalars{};
    check(cuMemAlloc(&a, input.size() * sizeof(float)));
    check(cuMemAlloc(&b, input.size() * sizeof(float)));
    check(cuMemAlloc(&c, output.size() * sizeof(float)));
    check(cuMemcpyHtoD(a, input.data(), input.size() * sizeof(float)));
    check(cuMemcpyHtoD(b, input.data(), input.size() * sizeof(float)));
    check(cuMemcpyHtoD(c, output.data(), output.size() * sizeof(float)));
    const std::array<float, 2> coefficients{2.0F, 0.5F};
    if (device_scalars) {
        check(cuMemAlloc(&scalars, sizeof(coefficients)));
        check(cuMemcpyHtoD(scalars, coefficients.data(), sizeof(coefficients)));
        check(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE));
    }
    std::cout << "Ordinary SGEMM priority application ready: role=low; pid=" << ::getpid()
              << "; tiles=" << (dimension / 64) * (dimension / 64) << '\n'
              << std::flush;
    pollfd start{STDIN_FILENO, POLLIN, 0};
    std::string command;
    if (::poll(&start, 1, 30000) != 1 || !std::getline(std::cin, command) || command != "run")
        throw std::runtime_error("bounded ordinary workload start barrier");
    // Original SDK ABI only: no Glimmer headers, progress buffers or launch edits.
    // NOLINTBEGIN(performance-no-int-to-ptr)
    const auto* alpha = device_scalars ? reinterpret_cast<const float*>(scalars) : &coefficients[0];
    const auto* beta =
        device_scalars ? reinterpret_cast<const float*>(scalars + sizeof(float)) : &coefficients[1];
    check(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension, dimension, depth, alpha,
                         reinterpret_cast<const float*>(a), dimension,
                         reinterpret_cast<const float*>(b), depth, beta,
                         reinterpret_cast<float*>(c), dimension));
    // NOLINTEND(performance-no-int-to-ptr)
    check(cuStreamSynchronize(stream));
    check(cuMemcpyDtoH(output.data(), c, output.size() * sizeof(float)));
    for (std::size_t index = 0; index < output.size(); ++index)
        if (output[index] != (index < output_elements ? 2.0F * depth + 3.0F : 6.0F))
            throw std::runtime_error("once-only beta-C numerical or padding mismatch at " +
                                     std::to_string(index));
    check(cuMemcpyDtoH(input.data(), a, input.size() * sizeof(float)));
    if (!std::ranges::all_of(input, [](float value) { return value == 1.0F; }))
        throw std::runtime_error("GEMM changed its original A input");
    check(cuMemcpyDtoH(input.data(), b, input.size() * sizeof(float)));
    if (!std::ranges::all_of(input, [](float value) { return value == 1.0F; }))
        throw std::runtime_error("GEMM changed its original B input");
    check(cublasDestroy_v2(handle));
    if (scalars) check(cuMemFree(scalars));
    check(cuMemFree(a));
    check(cuMemFree(b));
    check(cuMemFree(c));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    std::cout << "Ordinary SGEMM priority application verified: role=low; values=" << output.size()
              << "; input_values=" << input.size() * 2
              << "; beta_once=true; guards=true; "
                 "full_device_qualified=false; performance_qualified=false; device_scalars="
              << std::boolalpha << device_scalars << '\n';
}
}  // namespace
int main(int argc, char** argv) {
    try {
        bool is_device_tool{}, device_scalars{};
        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--device-tool")
                is_device_tool = true;
            else if (option == "--device-scalars")
                device_scalars = true;
            else
                throw std::runtime_error("unknown SGEMM priority fixture option");
        }
        run(is_device_tool, device_scalars);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
