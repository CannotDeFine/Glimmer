#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Ordinary installed-SDK input. Compile during development, never register as
// an automatic GPU CTest; native/owned runs require the bounded GPU collector.
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
        // Never retry a native destruction whose outcome is uncertain.
        return static_cast<int>(Destroy(std::exchange(value_, Handle{})));
    }
    Handle value_{};
};
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Lrn = Resource<cudnnLRNDescriptor_t, cudnnDestroyLRNDescriptor>;
using Memory = Resource<CUdeviceptr, cuMemFree>;
constexpr std::array<int, 4> kShape{2, 7, 2, 3};
constexpr std::array<int, 4> kInputStrides{42, 6, 3, 1};
// Padded NCHW and NHWC layouts have genuinely different logical indexing and
// untouched holes, not merely different descriptor labels for packed storage.
constexpr std::array kOutputStrides{std::array{64, 8, 4, 1}, std::array{48, 1, 24, 8}};
constexpr std::size_t kGuard = 8;
constexpr std::size_t kLogicalCount = 84;
constexpr std::size_t extent(const std::array<int, 4>& strides) {
    std::size_t result{1};
    for (std::size_t i = 0; i < strides.size(); ++i)
        result += static_cast<std::size_t>((kShape[i] - 1) * strides[i]);
    return result;
}
constexpr std::size_t offset(const std::array<int, 4>& strides, int n, int c, int h, int w) {
    return static_cast<std::size_t>(n * strides[0] + c * strides[1] + h * strides[2] +
                                    w * strides[3]);
}
template <typename Function>
void each_element(Function function) {
    for (int n = 0; n < kShape[0]; ++n)
        for (int c = 0; c < kShape[1]; ++c)
            for (int h = 0; h < kShape[2]; ++h)
                for (int w = 0; w < kShape[3]; ++w) function(n, c, h, w);
}
struct Parameters {
    unsigned int n;
    double alpha, beta, k;
};
// The first descriptor is left at native creation defaults. The remaining
// windows distinguish odd/even look-behind and channel-edge clipping.
constexpr std::array kParameters{Parameters{5, 1.0e-4, 0.75, 2.0},
                                 Parameters{3, 0.375, 0.625, 1.25},
                                 Parameters{4, 0.3125, 1.125, 1.5}};
template <typename Value>
struct Descriptors {
    Tensor input;
    std::array<Tensor, 2> output;
    std::array<Lrn, kParameters.size()> lrn;
    Descriptors() {
        constexpr auto type = std::is_same_v<Value, double> ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
        checked(cudnnCreateTensorDescriptor(input.output()), "create input tensor");
        checked(
            cudnnSetTensorNdDescriptor(input.get(), type, 4, kShape.data(), kInputStrides.data()),
            "set input tensor");
        for (std::size_t i = 0; i < output.size(); ++i) {
            checked(cudnnCreateTensorDescriptor(output[i].output()), "create output tensor");
            checked(cudnnSetTensorNdDescriptor(output[i].get(), type, 4, kShape.data(),
                                               kOutputStrides[i].data()),
                    "set strided output");
            std::size_t bytes{};
            checked(cudnnGetTensorSizeInBytes(output[i].get(), &bytes), "strided output extent");
            expect(bytes == extent(kOutputStrides[i]) * sizeof(Value),
                   "native strided byte hull agrees with independent address formula");
        }
        for (std::size_t i = 0; i < lrn.size(); ++i) {
            checked(cudnnCreateLRNDescriptor(lrn[i].output()), "create LRN descriptor");
            if (i != 0) {
                const auto& p = kParameters[i];
                checked(cudnnSetLRNDescriptor(lrn[i].get(), p.n, p.alpha, p.beta, p.k),
                        "set nondefault LRN descriptor");
            }
        }
    }
    void close() {
        for (auto& operation : lrn) operation.close();
        for (auto& tensor : output) tensor.close();
        input.close();
    }
};
template <typename Value>
Value source_value(int n, int c, int h, int w) {
    return static_cast<Value>(n * 13 + c * 5 + h * 3 + w - 19) * Value{0.125};
}
template <typename Value>
Value destination_value(int n, int c, int h, int w) {
    return static_cast<Value>(n * 7 - c * 2 + h * 3 - w) * Value{0.0625};
}
template <typename Value>
void upload(CUdeviceptr memory, const std::vector<Value>& values) {
    checked(cuMemcpyHtoD(memory, values.data(), values.size() * sizeof(Value)),
            "upload full storage");
}
template <typename Value>
void download(CUdeviceptr memory, std::vector<Value>& values) {
    checked(cuMemcpyDtoH(values.data(), memory, values.size() * sizeof(Value)),
            "download full storage");
}
template <typename Value>
void verify(const std::vector<Value>& actual, const std::vector<Value>& expected,
            const std::vector<bool>& writable) {
    expect(actual.size() == expected.size() && writable.size() == expected.size(),
           "complete storage extent");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!writable[i]) {
            expect(actual[i] == expected[i],
                   "every guard, hole and read-only element remains exact");
            continue;
        }
        const long double tolerance = std::is_same_v<Value, double> ? 2.0e-11L : 5.0e-5L;
        expect(std::isfinite(actual[i]) &&
                   std::abs(static_cast<long double>(actual[i]) - expected[i]) <=
                       tolerance * std::max(1.0L, std::abs(static_cast<long double>(expected[i]))),
               "every repeated non-idempotent output matches the independent oracle");
    }
}
template <typename Value>
std::array<Value, kLogicalCount> reference_lrn(const Parameters& parameters) {
    std::array<Value, kLogicalCount> output{};
    const int behind = (static_cast<int>(parameters.n) - 1) / 2;
    const int ahead = static_cast<int>(parameters.n) - behind - 1;
    // The SDK specifies conversion of descriptor doubles to the tensor type.
    const auto scale =
        static_cast<long double>(static_cast<Value>(parameters.alpha)) / parameters.n;
    const auto power = static_cast<long double>(static_cast<Value>(parameters.beta));
    const auto bias = static_cast<long double>(static_cast<Value>(parameters.k));
    each_element([&](int n, int c, int h, int w) {
        long double sum{};
        for (int channel = std::max(0, c - behind); channel <= std::min(kShape[1] - 1, c + ahead);
             ++channel) {
            const long double value = source_value<Value>(n, channel, h, w);
            sum += value * value;
        }
        output[offset(kInputStrides, n, c, h, w)] =
            static_cast<Value>(static_cast<long double>(source_value<Value>(n, c, h, w)) /
                               std::pow(bias + scale * sum, power));
    });
    return output;
}
struct Evidence {
    unsigned int lrn_calls{}, default_lrn_calls{}, transform_calls{};
};
template <typename Value>
void run_lrn(cudnnHandle_t handle, CUstream stream, const Descriptors<Value>& descriptors,
             Evidence& evidence) {
    std::vector<Value> source(extent(kInputStrides) + 2 * kGuard, Value{-1234});
    std::vector<Value> destination(extent(kOutputStrides[0]) + 2 * kGuard, Value{-2345});
    std::vector<bool> writable(destination.size());
    each_element([&](int n, int c, int h, int w) {
        source[kGuard + offset(kInputStrides, n, c, h, w)] = source_value<Value>(n, c, h, w);
        const auto y = kGuard + offset(kOutputStrides[0], n, c, h, w);
        destination[y] = destination_value<Value>(n, c, h, w);
        writable[y] = true;
    });
    Memory x, y;
    checked(cuMemAlloc(x.output(), source.size() * sizeof(Value)), "allocate LRN input");
    checked(cuMemAlloc(y.output(), destination.size() * sizeof(Value)), "allocate LRN output");
    upload(x.get(), source);
    const Value alpha{1.25}, beta{-0.5};
    for (std::size_t operation = 0; operation < kParameters.size(); ++operation) {
        upload(y.get(), destination);
        auto expected = destination;
        const auto normalized = reference_lrn<Value>(kParameters[operation]);
        for (unsigned int iteration = 0; iteration < 3; ++iteration) {
            // NOLINTBEGIN(performance-no-int-to-ptr): original Driver address ABI.
            checked(cudnnLRNCrossChannelForward(
                        handle, descriptors.lrn[operation].get(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
                        &alpha, descriptors.input.get(),
                        reinterpret_cast<const void*>(x.get() + kGuard * sizeof(Value)), &beta,
                        descriptors.output[0].get(),
                        reinterpret_cast<void*>(y.get() + kGuard * sizeof(Value))),
                    "LRN forward");
            // NOLINTEND(performance-no-int-to-ptr)
            ++evidence.lrn_calls;
            if (operation == 0) ++evidence.default_lrn_calls;
            each_element([&](int n, int c, int h, int w) {
                const auto index = kGuard + offset(kOutputStrides[0], n, c, h, w);
                expected[index] =
                    alpha * normalized[offset(kInputStrides, n, c, h, w)] + beta * expected[index];
            });
            checked(cuStreamSynchronize(stream), "LRN physical completion");
            auto actual = destination;
            download(y.get(), actual);
            verify(actual, expected, writable);
            auto unchanged = source;
            download(x.get(), unchanged);
            expect(unchanged == source, "entire LRN input and its guards remain unchanged");
        }
    }
    y.close();
    x.close();
}
template <typename Value>
void run_transform(cudnnHandle_t handle, CUstream stream, const Descriptors<Value>& descriptors,
                   Evidence& evidence) {
    const Value alpha{1.5}, beta{-0.25};
    for (std::size_t layout = 0; layout < kOutputStrides.size(); ++layout) {
        const auto& strides = kOutputStrides[layout];
        const auto source_base = kGuard;
        const auto destination_base = source_base + extent(kInputStrides) + 2 * kGuard;
        std::vector<Value> initial(destination_base + extent(strides) + kGuard, Value{-3456});
        std::vector<bool> writable(initial.size());
        each_element([&](int n, int c, int h, int w) {
            initial[source_base + offset(kInputStrides, n, c, h, w)] =
                source_value<Value>(n, c, h, w);
            const auto y = destination_base + offset(strides, n, c, h, w);
            initial[y] = destination_value<Value>(n, c, h, w);
            writable[y] = true;
        });
        // One allocation, two disjoint native byte hulls with a protected gap.
        Memory storage;
        checked(cuMemAlloc(storage.output(), initial.size() * sizeof(Value)),
                "allocate disjoint transform views");
        upload(storage.get(), initial);
        auto expected = initial;
        for (unsigned int iteration = 0; iteration < 3; ++iteration) {
            // NOLINTBEGIN(performance-no-int-to-ptr): original Driver address ABI.
            checked(cudnnTransformTensor(
                        handle, &alpha, descriptors.input.get(),
                        reinterpret_cast<const void*>(storage.get() + source_base * sizeof(Value)),
                        &beta, descriptors.output[layout].get(),
                        reinterpret_cast<void*>(storage.get() + destination_base * sizeof(Value))),
                    "strided tensor transform");
            // NOLINTEND(performance-no-int-to-ptr)
            ++evidence.transform_calls;
            each_element([&](int n, int c, int h, int w) {
                const auto index = destination_base + offset(strides, n, c, h, w);
                expected[index] = alpha * source_value<Value>(n, c, h, w) + beta * expected[index];
            });
            checked(cuStreamSynchronize(stream), "transform physical completion");
            auto actual = initial;
            download(storage.get(), actual);
            verify(actual, expected, writable);
        }
        storage.close();
    }
}
}  // namespace

int main() {
    try {
        // Configure both complete metadata graphs before explicit CUDA init.
        Descriptors<float> single;
        Descriptors<double> dual;
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
        run_lrn(handle.get(), stream.get(), single, evidence);
        run_lrn(handle.get(), stream.get(), dual, evidence);
        run_transform(handle.get(), stream.get(), single, evidence);
        run_transform(handle.get(), stream.get(), dual, evidence);
        expect(evidence.lrn_calls == 18 && evidence.default_lrn_calls == 6 &&
                   evidence.transform_calls == 12,
               "complete default/odd/even LRN and two-layout transform matrix");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind stream");
        handle.close();
        stream.close();
        context.close();
        dual.close();
        single.close();
        expect(!cleanup_failed, "checked once-only native cleanup");
        std::cout << "cuDNN LRN/transform input passed: types=2; lrn_calls=" << evidence.lrn_calls
                  << "; default_lrn_calls=" << evidence.default_lrn_calls
                  << "; transform_calls=" << evidence.transform_calls
                  << "; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
