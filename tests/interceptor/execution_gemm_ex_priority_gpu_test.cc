#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
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
template <typename Element>
Element encode(float value) {
    if constexpr (std::is_same_v<Element, __half>)
        return __float2half(value);
    else
        return value;
}
template <typename Element>
float decode(Element value) {
    if constexpr (std::is_same_v<Element, __half>)
        return __half2float(value);
    else
        return value;
}
template <typename Input, typename Output>
void run(bool is_device_tool, bool device_scalars, std::string_view format, bool strided,
         bool broadcasts) {
    const int dimension = strided ? (is_device_tool ? 1024 : 4096) : (is_device_tool ? 2048 : 8192);
    const int batches = strided ? 3 : 1;
    constexpr int depth = 8192;
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
    const auto input_stride = static_cast<std::size_t>(dimension) * depth;
    const auto output_stride = static_cast<std::size_t>(dimension) * dimension;
    const auto input_elements = input_stride * static_cast<unsigned int>(broadcasts ? 1 : batches);
    const auto output_elements = output_stride * static_cast<unsigned int>(batches);
    constexpr std::size_t guard = 32;
    std::vector<Input> input(input_elements + guard, encode<Input>(1.0F));
    std::vector<Output> output(output_elements + guard, encode<Output>(6.0F));
    CUdeviceptr a{}, b{}, c{}, scalars{};
    check(cuMemAlloc(&a, input.size() * sizeof(Input)));
    check(cuMemAlloc(&b, input.size() * sizeof(Input)));
    check(cuMemAlloc(&c, output.size() * sizeof(Output)));
    check(cuMemcpyHtoD(a, input.data(), input.size() * sizeof(Input)));
    check(cuMemcpyHtoD(b, input.data(), input.size() * sizeof(Input)));
    check(cuMemcpyHtoD(c, output.data(), output.size() * sizeof(Output)));
    // The final 1027 is exact in both output formats, so omission or replay of
    // beta-C cannot hide behind half-precision rounding at a large magnitude.
    const std::array<float, 2> coefficients{0.125F, 0.5F};
    if (device_scalars) {
        check(cuMemAlloc(&scalars, sizeof(coefficients)));
        check(cuMemcpyHtoD(scalars, coefficients.data(), sizeof(coefficients)));
        check(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_DEVICE));
    }
    std::cout << "Ordinary GemmEx priority ready: role=low; pid=" << ::getpid()
              << "; tiles=" << (dimension / 64) * (dimension / 64) * batches
              << "; format=" << format << '\n'
              << std::flush;
    pollfd start{STDIN_FILENO, POLLIN, 0};
    std::string command;
    if (::poll(&start, 1, 30000) != 1 || !std::getline(std::cin, command) || command != "run")
        throw std::runtime_error("bounded original workload start barrier");
    constexpr auto input_type = std::is_same_v<Input, __half> ? CUDA_R_16F : CUDA_R_32F;
    constexpr auto output_type = std::is_same_v<Output, __half> ? CUDA_R_16F : CUDA_R_32F;
    // NOLINTBEGIN(performance-no-int-to-ptr): exact original SDK GPU address ABI.
    const auto* alpha =
        device_scalars ? reinterpret_cast<const void*>(scalars) : coefficients.data();
    const auto* beta = device_scalars ? reinterpret_cast<const void*>(scalars + sizeof(float))
                                      : coefficients.data() + 1;
    if (strided)
        check(cublasGemmStridedBatchedEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension, dimension, depth, alpha,
            reinterpret_cast<const void*>(a), input_type, dimension,
            broadcasts ? 0 : static_cast<long long>(input_stride), reinterpret_cast<const void*>(b),
            input_type, depth, broadcasts ? 0 : static_cast<long long>(input_stride), beta,
            reinterpret_cast<void*>(c), output_type, dimension,
            static_cast<long long>(output_stride), batches, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
    else
        check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, dimension, dimension, depth, alpha,
                           reinterpret_cast<const void*>(a), input_type, dimension,
                           reinterpret_cast<const void*>(b), input_type, depth, beta,
                           reinterpret_cast<void*>(c), output_type, dimension, CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT));
    // NOLINTEND(performance-no-int-to-ptr)
    check(cuStreamSynchronize(stream));
    check(cuMemcpyDtoH(output.data(), c, output.size() * sizeof(Output)));
    for (std::size_t index = 0; index < output.size(); ++index)
        if (decode(output[index]) != (index < output_elements ? 1027.0F : 6.0F))
            throw std::runtime_error("typed once-only beta-C or padding mismatch at " +
                                     std::to_string(index));
    check(cuMemcpyDtoH(input.data(), a, input.size() * sizeof(Input)));
    if (!std::ranges::all_of(input, [](auto value) { return decode(value) == 1.0F; }))
        throw std::runtime_error("original A bytes changed");
    check(cuMemcpyDtoH(input.data(), b, input.size() * sizeof(Input)));
    if (!std::ranges::all_of(input, [](auto value) { return decode(value) == 1.0F; }))
        throw std::runtime_error("original B bytes changed");
    check(cublasDestroy_v2(handle));
    if (scalars) check(cuMemFree(scalars));
    check(cuMemFree(a));
    check(cuMemFree(b));
    check(cuMemFree(c));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    std::cout << "Ordinary GemmEx priority verified: role=low; values=" << output.size()
              << "; input_values=" << input.size() * 2
              << "; beta_once=true; guards=true; format=" << format
              << "; device_scalars=" << std::boolalpha << device_scalars
              << "; full_device_qualified=false; performance_qualified=false";
    if (strided) std::cout << "; strided=true; batches=3";
    if (broadcasts) std::cout << "; broadcast=true";
    std::cout << '\n';
}
}  // namespace
int main(int argc, char** argv) {
    try {
        bool device_tool{}, device_scalars{}, strided{}, broadcasts{};
        std::string_view format = "half";
        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--device-tool")
                device_tool = true;
            else if (option == "--device-scalars")
                device_scalars = true;
            else if (option == "--strided")
                strided = true;
            else if (option == "--broadcast")
                broadcasts = strided = true;
            else if (option == "--format" && index + 1 < argc)
                format = argv[++index];
            else
                throw std::runtime_error("unknown typed priority option");
        }
        if (format == "half")
            run<__half, __half>(device_tool, device_scalars, format, strided, broadcasts);
        else if (format == "mixed")
            run<__half, float>(device_tool, device_scalars, format, strided, broadcasts);
        else if (format == "float")
            run<float, float>(device_tool, device_scalars, format, strided, broadcasts);
        else
            throw std::runtime_error("unknown typed priority format");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
