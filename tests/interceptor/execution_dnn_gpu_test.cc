#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <type_traits>

// This ordinary installed-SDK input intentionally exercises legacy ABI calls.
// Compile during implementation; execute only through the bounded GPU collector.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void driver(CUresult status) {
    expect(status == CUDA_SUCCESS, "original Driver operation succeeds");
}
void dnn(cudnnStatus_t status) {
    expect(status == CUDNN_STATUS_SUCCESS, "original cuDNN operation succeeds");
}
template <typename Value>
void run(cudnnHandle_t handle, cudnnActivationDescriptor_t activation) {
    constexpr std::size_t kCount = 32, kGuard = 4, kTotal = kCount + 2 * kGuard;
    constexpr auto kType = std::is_same_v<Value, double> ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    constexpr Value kSentinel = Value{-1234};
    std::array<Value, kTotal> source{}, expected{}, actual{};
    source.fill(kSentinel);
    expected.fill(kSentinel);
    for (std::size_t index = 0; index < kCount; ++index) {
        source[index + kGuard] = static_cast<Value>(static_cast<int>(index) - 17);
        expected[index + kGuard] = static_cast<Value>(index) * Value{0.25};
    }
    cudnnTensorDescriptor_t x{}, y{};
    dnn(cudnnCreateTensorDescriptor(&x));
    dnn(cudnnCreateTensorDescriptor(&y));
    dnn(cudnnSetTensor4dDescriptor(x, CUDNN_TENSOR_NCHW, kType, 1, 1, 4, 8));
    const std::array<int, 4> dimensions{1, 1, 4, 8}, strides{32, 32, 8, 1};
    dnn(cudnnSetTensorNdDescriptor(y, kType, 4, dimensions.data(), strides.data()));
    std::size_t bytes{};
    dnn(cudnnGetTensorSizeInBytes(x, &bytes));
    expect(bytes == kCount * sizeof(Value), "native tensor byte extent");
    cudnnDataType_t queried{};
    int rank{};
    std::array<int, 12> shape{}, steps{};
    shape.fill(-7);
    steps.fill(-9);
    dnn(cudnnGetTensorNdDescriptor(y, 12, &queried, &rank, shape.data(), steps.data()));
    expect(queried == kType && rank == 4 &&
               std::equal(dimensions.begin(), dimensions.end(), shape.begin()) &&
               std::equal(strides.begin(), strides.end(), steps.begin()) && shape[4] == -7 &&
               steps[4] == -9,
           "oversized native descriptor query preserves the unwritten suffix");
    CUdeviceptr input{}, output{};
    driver(cuMemAlloc(&input, sizeof(source)));
    driver(cuMemAlloc(&output, sizeof(expected)));
    driver(cuMemcpyHtoD(input, source.data(), sizeof(source)));
    driver(cuMemcpyHtoD(output, expected.data(), sizeof(expected)));
    // NOLINTBEGIN(performance-no-int-to-ptr): original CUDA address ABI.
    const auto* x_pointer = reinterpret_cast<const void*>(input + kGuard * sizeof(Value));
    auto* y_pointer = reinterpret_cast<void*>(output + kGuard * sizeof(Value));
    // NOLINTEND(performance-no-int-to-ptr)
    const Value alpha{1.5}, beta{-0.25};
    for (unsigned int iteration = 0; iteration < 4; ++iteration) {
        const bool is_activation = iteration % 2 != 0;
        if (is_activation)
            dnn(cudnnActivationForward(handle, activation, &alpha, x, x_pointer, &beta, y,
                                       y_pointer));
        else
            dnn(cudnnAddTensor(handle, &alpha, x, x_pointer, &beta, y, y_pointer));
        for (std::size_t index = 0; index < kCount; ++index) {
            const auto value = source[index + kGuard];
            expected[index + kGuard] = alpha * (is_activation ? std::max(value, Value{0}) : value) +
                                       beta * expected[index + kGuard];
        }
        driver(cuCtxSynchronize());
        driver(cuMemcpyDtoH(actual.data(), output, sizeof(actual)));
        for (std::size_t index = 0; index < kTotal; ++index) {
            const auto tolerance = std::is_same_v<Value, double> ? 1.0e-12 : 1.0e-6;
            expect(
                std::isfinite(actual[index]) &&
                    std::abs(static_cast<double>(actual[index] - expected[index])) <=
                        tolerance * std::max(1.0, std::abs(static_cast<double>(expected[index]))),
                "every result and both guards match the non-idempotent oracle");
        }
        driver(cuMemcpyDtoH(actual.data(), input, sizeof(actual)));
        expect(actual == source, "read-only input and both guards remain unchanged");
    }
    driver(cuMemFree(output));
    driver(cuMemFree(input));
    dnn(cudnnDestroyTensorDescriptor(y));
    dnn(cudnnDestroyTensorDescriptor(x));
}
}  // namespace
int main() {
    try {
        // Descriptor metadata intentionally exists before the application context.
        cudnnActivationDescriptor_t activation{};
        dnn(cudnnCreateActivationDescriptor(&activation));
        dnn(cudnnSetActivationDescriptor(activation, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN,
                                         0));
        driver(cuInit(0));
        CUdevice device{};
        driver(cuDeviceGet(&device, 0));
        CUcontext context{};
        driver(cuCtxCreate(&context, 0, device));
        CUstream stream{};
        driver(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        cudnnHandle_t handle{};
        dnn(cudnnCreate(&handle));
        dnn(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
        cudaStream_t observed{};
        dnn(cudnnGetStream(handle, &observed));
        expect(observed == reinterpret_cast<cudaStream_t>(stream),
               "original public stream identity");
        run<float>(handle, activation);
        run<double>(handle, activation);
        dnn(cudnnSetStream(handle, nullptr));
        dnn(cudnnDestroy(handle));
        driver(cuStreamDestroy(stream));
        driver(cuCtxDestroy(context));
        dnn(cudnnDestroyActivationDescriptor(activation));
        std::cout << "cuDNN tensor input passed: types=2; compute_calls=8; "
                     "full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
