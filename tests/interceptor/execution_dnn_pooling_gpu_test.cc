#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Ordinary installed-SDK input, never an automatic GPU CTest. Compile during
// development; run only through a bounded, isolated native/owned collector.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
bool cleanup_failed{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status) != 0)
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (release() != 0) cleanup_failed = true;
    }
    Handle* output() {
        return &value_;
    }
    Handle get() const {
        return value_;
    }
    void close() {
        checked(release(), "native cleanup");
    }

   private:
    int release() noexcept {
        if (!value_) return 0;
        // Unknown failed destruction is recorded and never retried.
        return static_cast<int>(Destroy(std::exchange(value_, Handle{})));
    }
    Handle value_{};
};
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Pooling = Resource<cudnnPoolingDescriptor_t, cudnnDestroyPoolingDescriptor>;
using Memory = Resource<CUdeviceptr, cuMemFree>;
constexpr std::size_t kGuard = 8;
constexpr std::array kPoolingModes{CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                   CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                                   CUDNN_POOLING_MAX_DETERMINISTIC};
constexpr std::array kAlgorithms{CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG};
constexpr std::array kSoftmaxModes{CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL};
constexpr std::size_t elements(const std::array<int, 5>& dimensions) {
    std::size_t result{1};
    for (auto dimension : dimensions) result *= static_cast<std::size_t>(dimension);
    return result;
}
constexpr std::array<int, 5> strides(const std::array<int, 5>& dimensions) {
    std::array<int, 5> result{1, 1, 1, 1, 1};
    for (std::size_t i = result.size() - 1; i != 0; --i) result[i - 1] = result[i] * dimensions[i];
    return result;
}
template <typename Value, int Spatial>
struct Descriptors {
    static constexpr auto kType =
        std::is_same_v<Value, double> ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    static constexpr auto kInput =
        Spatial == 2 ? std::array{2, 3, 4, 5, 1} : std::array{1, 2, 3, 4, 5};
    static constexpr auto kWindow = Spatial == 2 ? std::array{2, 3, 1} : std::array{2, 2, 3};
    static constexpr auto kPadding = Spatial == 2 ? std::array{1, 1, 0} : std::array{0, 1, 1};
    static constexpr auto kStrides = Spatial == 2 ? std::array{2, 2, 1} : std::array{1, 2, 2};
    static constexpr auto kOutput = [] {
        auto result = kInput;
        for (std::size_t i = 0; i < 3; ++i)
            result[i + 2] = (kInput[i + 2] + 2 * kPadding[i] - kWindow[i]) / kStrides[i] + 1;
        return result;
    }();
    Tensor input, output;
    std::array<Pooling, kPoolingModes.size()> pooling;
    Descriptors() {
        checked(cudnnCreateTensorDescriptor(input.output()), "create input tensor");
        checked(cudnnCreateTensorDescriptor(output.output()), "create output tensor");
        const auto input_strides = strides(kInput), output_strides = strides(kOutput);
        checked(cudnnSetTensorNdDescriptor(input.get(), kType, Spatial + 2, kInput.data(),
                                           input_strides.data()),
                "set packed Nd input");
        checked(cudnnSetTensorNdDescriptor(output.get(), kType, Spatial + 2, kOutput.data(),
                                           output_strides.data()),
                "set packed Nd output");
        for (std::size_t i = 0; i < pooling.size(); ++i) {
            checked(cudnnCreatePoolingDescriptor(pooling[i].output()), "create pooling descriptor");
            checked(cudnnSetPoolingNdDescriptor(pooling[i].get(), kPoolingModes[i],
                                                CUDNN_PROPAGATE_NAN, Spatial, kWindow.data(),
                                                kPadding.data(), kStrides.data()),
                    "set Nd pooling");
            int rank{};
            cudnnPoolingMode_t mode{};
            cudnnNanPropagation_t nan{};
            std::array<int, 8> window{}, padding{}, steps{};
            window.fill(-71);
            padding.fill(-72);
            steps.fill(-73);
            checked(cudnnGetPoolingNdDescriptor(pooling[i].get(), 8, &mode, &nan, &rank,
                                                window.data(), padding.data(), steps.data()),
                    "get Nd pooling");
            expect(rank == Spatial && mode == kPoolingModes[i] && nan == CUDNN_PROPAGATE_NAN,
                   "exact native pooling enum/rank metadata");
            for (std::size_t j = 0; j < static_cast<std::size_t>(Spatial); ++j)
                expect(
                    window[j] == kWindow[j] && padding[j] == kPadding[j] && steps[j] == kStrides[j],
                    "exact native pooling arrays");
            std::array<int, 8> shape{};
            shape.fill(-74);
            checked(cudnnGetPoolingNdForwardOutputDim(pooling[i].get(), input.get(), Spatial + 2,
                                                      shape.data()),
                    "get pooling output shape");
            expect(std::equal(kOutput.begin(), kOutput.begin() + Spatial + 2, shape.begin()),
                   "native output dimensions agree with independent integer formula");
            for (std::size_t j = static_cast<std::size_t>(Spatial + 2); j < shape.size(); ++j)
                expect(shape[j] == -74, "output shape leaves caller tail unchanged");
        }
    }
    void close() {
        for (auto& operation : pooling) operation.close();
        output.close();
        input.close();
    }
};
template <typename Value>
std::vector<Value> initial(std::size_t count, Value guard, int modulus, int shift) {
    std::vector<Value> values(count + 2 * kGuard, guard);
    for (std::size_t i = 0; i < count; ++i)
        values[kGuard + i] =
            static_cast<Value>(static_cast<int>(i % static_cast<std::size_t>(modulus)) - shift) *
            Value{0.125};
    return values;
}
template <typename Value>
void verify_output(const std::vector<Value>& actual, const std::vector<Value>& expected) {
    expect(actual.size() == expected.size(), "complete result extent");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (i < kGuard || i >= actual.size() - kGuard) {
            expect(actual[i] == expected[i], "both output guards remain exact");
            continue;
        }
        const long double tolerance = std::is_same_v<Value, double> ? 2.0e-11L : 5.0e-5L;
        expect(std::isfinite(actual[i]) &&
                   std::abs(static_cast<long double>(actual[i]) - expected[i]) <=
                       tolerance * std::max(1.0L, std::abs(static_cast<long double>(expected[i]))),
               "every output matches the independent repeated CPU oracle");
    }
}
template <typename Value>
void download(CUdeviceptr memory, std::vector<Value>& values) {
    checked(cuMemcpyDtoH(values.data(), memory, values.size() * sizeof(Value)),
            "download full range");
}
template <typename Value>
void upload(CUdeviceptr memory, const std::vector<Value>& values) {
    checked(cuMemcpyHtoD(memory, values.data(), values.size() * sizeof(Value)),
            "upload full range");
}
template <typename Value, int Spatial>
std::vector<Value> pool_reference(const std::vector<Value>& input, std::size_t mode) {
    using Shape = Descriptors<Value, Spatial>;
    const auto& x = Shape::kInput;
    const auto& y = Shape::kOutput;
    std::vector<Value> result(elements(y));
    const bool is_maximum = mode == 0 || mode == 3;
    for (int n = 0; n < y[0]; ++n)
        for (int c = 0; c < y[1]; ++c)
            for (int a = 0; a < y[2]; ++a)
                for (int b = 0; b < y[3]; ++b)
                    for (int d = 0; d < y[4]; ++d) {
                        long double value =
                            is_maximum ? -std::numeric_limits<long double>::infinity() : 0.0L;
                        int valid{};
                        for (int wa = 0; wa < Shape::kWindow[0]; ++wa)
                            for (int wb = 0; wb < Shape::kWindow[1]; ++wb)
                                for (int wd = 0; wd < Shape::kWindow[2]; ++wd) {
                                    const int ia = a * Shape::kStrides[0] - Shape::kPadding[0] + wa;
                                    const int ib = b * Shape::kStrides[1] - Shape::kPadding[1] + wb;
                                    const int id = d * Shape::kStrides[2] - Shape::kPadding[2] + wd;
                                    if (ia < 0 || ia >= x[2] || ib < 0 || ib >= x[3] || id < 0 ||
                                        id >= x[4])
                                        continue;
                                    const auto index = static_cast<std::size_t>(
                                        ((((n * x[1] + c) * x[2] + ia) * x[3] + ib) * x[4] + id));
                                    const long double element = input[kGuard + index];
                                    value = is_maximum ? std::max(value, element) : value + element;
                                    ++valid;
                                }
                        expect(valid != 0, "every pooling window has actual input");
                        if (!is_maximum)
                            value /= mode == 1
                                         ? Shape::kWindow[0] * Shape::kWindow[1] * Shape::kWindow[2]
                                         : valid;
                        const auto index = static_cast<std::size_t>(
                            ((((n * y[1] + c) * y[2] + a) * y[3] + b) * y[4] + d));
                        result[index] = static_cast<Value>(value);
                    }
    return result;
}
template <typename Value>
std::vector<Value> softmax_reference(const std::vector<Value>& input,
                                     const std::array<int, 5>& shape, bool is_channel,
                                     bool is_log) {
    const auto channels = static_cast<std::size_t>(shape[1]);
    const auto spatial = static_cast<std::size_t>(shape[2] * shape[3] * shape[4]);
    const auto batch = channels * spatial;
    std::vector<Value> result(elements(shape));
    const auto groups = is_channel ? spatial : 1;
    const auto group_size = is_channel ? channels : batch;
    for (std::size_t n = 0; n < static_cast<std::size_t>(shape[0]); ++n)
        for (std::size_t group = 0; group < groups; ++group) {
            const auto offset = [&](std::size_t index) {
                return n * batch + (is_channel ? index * spatial + group : index);
            };
            long double maximum = -std::numeric_limits<long double>::infinity();
            for (std::size_t i = 0; i < group_size; ++i)
                maximum = std::max(maximum, static_cast<long double>(input[kGuard + offset(i)]));
            long double sum{};
            for (std::size_t i = 0; i < group_size; ++i)
                sum += std::exp(static_cast<long double>(input[kGuard + offset(i)]) - maximum);
            for (std::size_t i = 0; i < group_size; ++i) {
                const long double centered = input[kGuard + offset(i)] - maximum;
                result[offset(i)] = static_cast<Value>(is_log ? centered - std::log(sum)
                                                              : std::exp(centered) / sum);
            }
        }
    return result;
}
struct Evidence {
    unsigned int pooling_calls{}, softmax_calls{}, inplace_calls{};
};
template <typename Value, int Spatial>
void run_pooling(cudnnHandle_t handle, CUstream stream,
                 const Descriptors<Value, Spatial>& descriptors, Evidence& evidence) {
    using Shape = Descriptors<Value, Spatial>;
    const auto source = initial(elements(Shape::kInput), Value{-1234}, 29, 15);
    const auto destination = initial(elements(Shape::kOutput), Value{-2345}, 13, 5);
    Memory x, y;
    checked(cuMemAlloc(x.output(), source.size() * sizeof(Value)), "allocate pooling input");
    checked(cuMemAlloc(y.output(), destination.size() * sizeof(Value)), "allocate pooling output");
    upload(x.get(), source);
    const Value alpha{1.25}, beta{-0.5};
    for (std::size_t mode = 0; mode < kPoolingModes.size(); ++mode) {
        upload(y.get(), destination);
        auto expected = destination;
        const auto pooled = pool_reference<Value, Spatial>(source, mode);
        for (int repetition = 0; repetition < 3; ++repetition) {
            // NOLINTBEGIN(performance-no-int-to-ptr): original Driver device address ABI.
            checked(cudnnPoolingForward(
                        handle, descriptors.pooling[mode].get(), &alpha, descriptors.input.get(),
                        reinterpret_cast<const void*>(x.get() + kGuard * sizeof(Value)), &beta,
                        descriptors.output.get(),
                        reinterpret_cast<void*>(y.get() + kGuard * sizeof(Value))),
                    "pooling forward");
            // NOLINTEND(performance-no-int-to-ptr)
            ++evidence.pooling_calls;
            for (std::size_t i = 0; i < pooled.size(); ++i)
                expected[kGuard + i] = alpha * pooled[i] + beta * expected[kGuard + i];
            checked(cuStreamSynchronize(stream), "pooling physical completion");
            auto actual = destination;
            download(y.get(), actual);
            verify_output(actual, expected);
            auto observed_input = source;
            download(x.get(), observed_input);
            expect(observed_input == source, "pooling preserves entire input and both guards");
        }
    }
    y.close();
    x.close();
}
template <typename Value>
void run_softmax(cudnnHandle_t handle, CUstream stream, const Descriptors<Value, 2>& descriptors,
                 Evidence& evidence) {
    using Shape = Descriptors<Value, 2>;
    const auto source = initial(elements(Shape::kInput), Value{-3456}, 19, 9);
    const auto destination = initial(elements(Shape::kInput), Value{-4567}, 11, 5);
    Memory x, y;
    checked(cuMemAlloc(x.output(), source.size() * sizeof(Value)), "allocate softmax input");
    checked(cuMemAlloc(y.output(), destination.size() * sizeof(Value)), "allocate softmax output");
    const Value alpha{1.25}, beta{-0.25};
    for (const auto algorithm : kAlgorithms)
        for (const auto mode : kSoftmaxModes)
            for (const bool is_inplace : {false, true}) {
                upload(x.get(), source);
                upload(y.get(), destination);
                auto expected = is_inplace ? source : destination;
                for (int repetition = 0; repetition < 3; ++repetition) {
                    const auto reference = softmax_reference(
                        is_inplace ? expected : source, Shape::kInput,
                        mode == CUDNN_SOFTMAX_MODE_CHANNEL, algorithm == CUDNN_SOFTMAX_LOG);
                    // NOLINTBEGIN(performance-no-int-to-ptr): original Driver device address ABI.
                    checked(cudnnSoftmaxForward(
                                handle, algorithm, mode, &alpha, descriptors.input.get(),
                                reinterpret_cast<const void*>(x.get() + kGuard * sizeof(Value)),
                                &beta, descriptors.input.get(),
                                reinterpret_cast<void*>((is_inplace ? x.get() : y.get()) +
                                                        kGuard * sizeof(Value))),
                            "softmax forward");
                    // NOLINTEND(performance-no-int-to-ptr)
                    ++evidence.softmax_calls;
                    if (is_inplace) ++evidence.inplace_calls;
                    for (std::size_t i = 0; i < reference.size(); ++i)
                        expected[kGuard + i] = alpha * reference[i] + beta * expected[kGuard + i];
                    checked(cuStreamSynchronize(stream), "softmax physical completion");
                    auto actual = expected;
                    download(is_inplace ? x.get() : y.get(), actual);
                    verify_output(actual, expected);
                    auto unchanged = is_inplace ? destination : source;
                    download(is_inplace ? y.get() : x.get(), unchanged);
                    expect(unchanged == (is_inplace ? destination : source),
                           "softmax preserves the entire non-output allocation and its guards");
                }
            }
    y.close();
    x.close();
}
}  // namespace

int main() {
    try {
        // All descriptor graphs exist before explicit Driver initialization.
        Descriptors<float, 2> single_two;
        Descriptors<double, 2> double_two;
        Descriptors<float, 3> single_three;
        Descriptors<double, 3> double_three;
        checked(cuInit(0), "initialize Driver");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select device");
        Resource<CUcontext, cuCtxDestroy> context;
        checked(cuCtxCreate(context.output(), 0, device), "create context");
        Resource<CUstream, cuStreamDestroy> stream;
        checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING), "create stream");
        Resource<cudnnHandle_t, cudnnDestroy> handle;
        checked(cudnnCreate(handle.output()), "create cuDNN handle");
        checked(cudnnSetStream(handle.get(), reinterpret_cast<cudaStream_t>(stream.get())),
                "bind explicit stream");
        cudaStream_t observed{};
        checked(cudnnGetStream(handle.get(), &observed), "query explicit stream");
        expect(observed == reinterpret_cast<cudaStream_t>(stream.get()), "exact stream identity");
        Evidence evidence;
        run_pooling(handle.get(), stream.get(), single_two, evidence);
        run_pooling(handle.get(), stream.get(), double_two, evidence);
        run_pooling(handle.get(), stream.get(), single_three, evidence);
        run_pooling(handle.get(), stream.get(), double_three, evidence);
        run_softmax(handle.get(), stream.get(), single_two, evidence);
        run_softmax(handle.get(), stream.get(), double_two, evidence);
        expect(evidence.pooling_calls == 48 && evidence.softmax_calls == 72 &&
                   evidence.inplace_calls == 36,
               "complete bounded operation matrix executed");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind stream");
        handle.close();
        stream.close();
        context.close();
        double_three.close();
        single_three.close();
        double_two.close();
        single_two.close();
        expect(!cleanup_failed, "checked once-only native cleanup");
        std::cout << "cuDNN pooling/softmax input passed: types=2; pooling_spatial_ranks=2,3; "
                     "pooling_modes=4; softmax_algorithms=3; softmax_modes=2; pooling_calls="
                  << evidence.pooling_calls << "; softmax_calls=" << evidence.softmax_calls
                  << "; inplace_calls=" << evidence.inplace_calls
                  << "; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
