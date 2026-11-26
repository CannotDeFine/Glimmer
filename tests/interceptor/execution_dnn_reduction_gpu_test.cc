#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// Ordinary installed-SDK input, not an injected provider or automatic GPU test.
// Compilation does not qualify numerical execution, full-device use or priority.
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
        return static_cast<int>(Destroy(std::exchange(value_, Handle{})));
    }
    Handle value_{};
};
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
using Reduction = Resource<cudnnReduceTensorDescriptor_t, cudnnDestroyReduceTensorDescriptor>;
using Memory = Resource<CUdeviceptr, cuMemFree>;
constexpr std::size_t kGuard = 32;
constexpr std::size_t kByteGuard = 256;
constexpr std::size_t kMaximumAuxiliaryBytes = 64 * 1024 * 1024;
constexpr std::array kInputShape{2, 3, 2, 4};
constexpr std::array kInputStrides{40, 12, 5, 1};
constexpr std::array kOutputShape{2, 1, 1, 1};
// ADD exercises output holes. Indexed MIN/MAX use packed output so the
// independently checked index slots do not depend on an undocumented index
// storage convention for nonpacked output descriptors.
constexpr std::array kOutputStrides{std::array{3, 1, 1, 1}, std::array{1, 1, 1, 1}};
constexpr std::size_t kInputExtent = 73, kOutputExtent = 4;
constexpr std::array kOperations{CUDNN_REDUCE_TENSOR_ADD, CUDNN_REDUCE_TENSOR_MIN,
                                 CUDNN_REDUCE_TENSOR_MAX};
template <typename Value>
constexpr cudnnDataType_t data_type() {
    if constexpr (std::is_same_v<Value, double>) return CUDNN_DATA_DOUBLE;
    if constexpr (std::is_same_v<Value, std::int8_t>) return CUDNN_DATA_INT8;
    return CUDNN_DATA_FLOAT;
}
template <typename Input, typename Output>
struct Descriptors {
    Tensor input;
    std::array<Tensor, 2> output;
    std::array<Reduction, kOperations.size()> reductions;
    Descriptors() {
        checked(cudnnCreateTensorDescriptor(input.output()), "create input descriptor");
        checked(cudnnSetTensorNdDescriptor(input.get(), data_type<Input>(), 4, kInputShape.data(),
                                           kInputStrides.data()),
                "set input strides");
        std::size_t bytes{};
        checked(cudnnGetTensorSizeInBytes(input.get(), &bytes), "input byte hull");
        expect(bytes == kInputExtent * sizeof(Input), "independent input hull");
        for (std::size_t i = 0; i != output.size(); ++i) {
            checked(cudnnCreateTensorDescriptor(output[i].output()), "create output descriptor");
            checked(cudnnSetTensorNdDescriptor(output[i].get(), data_type<Output>(), 4,
                                               kOutputShape.data(), kOutputStrides[i].data()),
                    "set output strides");
            checked(cudnnGetTensorSizeInBytes(output[i].get(), &bytes), "output byte hull");
            expect(bytes == (i == 0 ? kOutputExtent : 2) * sizeof(Output),
                   "independent output hull");
        }
        for (std::size_t i = 0; i != reductions.size(); ++i) {
            checked(cudnnCreateReduceTensorDescriptor(reductions[i].output()),
                    "create reduction descriptor");
            checked(
                cudnnSetReduceTensorDescriptor(
                    reductions[i].get(), kOperations[i],
                    std::is_same_v<Input, double> ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT,
                    CUDNN_PROPAGATE_NAN,
                    i == 0 ? CUDNN_REDUCE_TENSOR_NO_INDICES : CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
                    CUDNN_32BIT_INDICES),
                "set reduction descriptor");
        }
    }
    void close() {
        for (auto& reduction : reductions) reduction.close();
        for (auto& tensor : output) tensor.close();
        input.close();
    }
};
template <typename Value>
void upload(CUdeviceptr memory, const std::vector<Value>& values) {
    checked(cuMemcpyHtoD(memory, values.data(), values.size() * sizeof(Value)),
            "upload complete allocation");
}
template <typename Value>
void download(CUdeviceptr memory, std::vector<Value>& values) {
    checked(cuMemcpyDtoH(values.data(), memory, values.size() * sizeof(Value)),
            "download complete allocation");
}
struct Auxiliary {
    Memory memory;
    std::vector<std::byte> original, observed;
    std::size_t bytes;
    explicit Auxiliary(std::size_t requested)
        : original(std::max(requested, std::size_t{8}) + 2 * kByteGuard, std::byte{0xa7}),
          observed(original),
          bytes(requested) {
        expect(requested <= kMaximumAuxiliaryBytes, "bounded auxiliary allocation");
        checked(cuMemAlloc(memory.output(), original.size()), "allocate guarded auxiliary");
        upload(memory.get(), original);
    }
    void* pointer() const {
        return reinterpret_cast<void*>(memory.get() + kByteGuard);
    }
    void* end() const {
        return reinterpret_cast<void*>(memory.get() + original.size());
    }
    void verify(bool untouched) {
        download(memory.get(), observed);
        for (std::size_t i = 0; i != observed.size(); ++i)
            if (untouched || i < kByteGuard || i >= kByteGuard + bytes)
                expect(observed[i] == original[i],
                       "every auxiliary guard and byte outside declared range is unchanged");
    }
};
constexpr std::size_t input_offset(int n, int c, int h, int w) {
    return static_cast<std::size_t>(n * 40 + c * 12 + h * 5 + w);
}
constexpr int source(int n, int c, int h, int w) {
    // Unique, deliberately non-monotonic values identify the flattened index
    // independently of its strided storage offset and the unreduced batch.
    return ((c * 8 + h * 4 + w) * 7 + 5) % 24 - 12 + n * 2;
}
struct ExpectedReduction {
    long double value{};
    std::uint32_t index{};
};
ExpectedReduction oracle(int n, cudnnReduceTensorOp_t operation) {
    ExpectedReduction result{
        operation == CUDNN_REDUCE_TENSOR_ADD ? 0.0L : static_cast<long double>(source(n, 0, 0, 0)),
        0};
    for (int c = 0; c != 3; ++c)
        for (int h = 0; h != 2; ++h)
            for (int w = 0; w != 4; ++w) {
                const auto value = static_cast<long double>(source(n, c, h, w));
                if (operation == CUDNN_REDUCE_TENSOR_ADD)
                    result.value += value;
                else if ((operation == CUDNN_REDUCE_TENSOR_MIN && value < result.value) ||
                         (operation == CUDNN_REDUCE_TENSOR_MAX && value > result.value))
                    result = {value, static_cast<std::uint32_t>(c * 8 + h * 4 + w)};
            }
    return result;
}
struct Evidence {
    unsigned calls{}, indexed_calls{}, mixed_calls{}, ignored_null_calls{}, ignored_end_calls{};
};
template <typename Input, typename Output>
void run(cudnnHandle_t handle, CUstream stream, const Descriptors<Input, Output>& descriptors,
         Evidence& evidence) {
    using Scalar = std::conditional_t<std::is_same_v<Input, double>, double, float>;
    constexpr bool mixed = !std::is_same_v<Input, Output>;
    constexpr Scalar alpha = mixed ? Scalar{1} : Scalar{0.75};
    constexpr Scalar beta = mixed ? Scalar{1} : Scalar{0.25};
    std::vector<Input> input(kInputExtent + 2 * kGuard, Input{99});
    for (int n = 0; n != 2; ++n)
        for (int c = 0; c != 3; ++c)
            for (int h = 0; h != 2; ++h)
                for (int w = 0; w != 4; ++w)
                    input[kGuard + input_offset(n, c, h, w)] =
                        static_cast<Input>(source(n, c, h, w));
    Memory input_memory, output_memory;
    checked(cuMemAlloc(input_memory.output(), input.size() * sizeof(Input)), "allocate input");
    checked(cuMemAlloc(output_memory.output(), (kOutputExtent + 2 * kGuard) * sizeof(Output)),
            "allocate output");
    upload(input_memory.get(), input);
    for (std::size_t operation = 0; operation != kOperations.size(); ++operation) {
        const auto output = descriptors.output[operation == 0 ? 0 : 1].get();
        const std::size_t output_stride = operation == 0 ? 3 : 1;
        std::size_t indices_bytes{}, workspace_bytes{};
        checked(cudnnGetReductionIndicesSize(handle, descriptors.reductions[operation].get(),
                                             descriptors.input.get(), output, &indices_bytes),
                "query native indices size");
        checked(cudnnGetReductionWorkspaceSize(handle, descriptors.reductions[operation].get(),
                                               descriptors.input.get(), output, &workspace_bytes),
                "query native workspace size");
        expect(indices_bytes <= kMaximumAuxiliaryBytes && workspace_bytes <= kMaximumAuxiliaryBytes,
               "native size queries remain within the experiment memory budget");
        expect(operation == 0 || indices_bytes >= 2 * sizeof(std::uint32_t),
               "both flattened output indices fit the native declared space");
        Auxiliary indices(operation == 0 ? 8 : indices_bytes), workspace(workspace_bytes);
        std::vector<Output> expected(kOutputExtent + 2 * kGuard, Output{97});
        expected[kGuard] = Output{-3};
        expected[kGuard + output_stride] = Output{-1};
        auto observed = expected;
        upload(output_memory.get(), expected);
        for (unsigned repetition = 0; repetition != 3; ++repetition) {
            void* indices_pointer = indices.pointer();
            std::size_t declared_indices_bytes = indices_bytes;
            if (operation == 0) {
                indices_pointer = repetition == 0   ? nullptr
                                  : repetition == 1 ? indices.end()
                                                    : indices.pointer();
                declared_indices_bytes = repetition == 0 ? 19 : repetition == 1 ? 0 : 8;
                evidence.ignored_null_calls += repetition == 0 ? 1U : 0U;
                evidence.ignored_end_calls += repetition == 1 ? 1U : 0U;
            }
            void* workspace_pointer = workspace_bytes   ? workspace.pointer()
                                      : repetition == 0 ? nullptr
                                      : repetition == 1 ? workspace.end()
                                                        : workspace.pointer();
            checked(cudnnReduceTensor(
                        handle, descriptors.reductions[operation].get(), indices_pointer,
                        declared_indices_bytes, workspace_pointer, workspace_bytes, &alpha,
                        descriptors.input.get(),
                        reinterpret_cast<const void*>(input_memory.get() + kGuard * sizeof(Input)),
                        &beta, output,
                        reinterpret_cast<void*>(output_memory.get() + kGuard * sizeof(Output))),
                    "original device reduction");
            checked(cuStreamSynchronize(stream), "complete owned reduction stream");
            for (int n = 0; n != 2; ++n) {
                const auto offset = kGuard + static_cast<std::size_t>(n) * output_stride;
                const auto result = oracle(n, kOperations[operation]);
                const auto value = static_cast<long double>(alpha) * result.value +
                                   static_cast<long double>(beta) * expected[offset];
                expect(value >= -128 && value <= 127, "mixed integer oracle avoids clipping");
                expected[offset] = static_cast<Output>(value);
            }
            download(output_memory.get(), observed);
            for (std::size_t i = 0; i != observed.size(); ++i) {
                if (i == kGuard || i == kGuard + output_stride) {
                    const long double tolerance =
                        std::is_same_v<Output, double> ? 1.0e-11L : 1.0e-4L;
                    expect(
                        std::isfinite(static_cast<long double>(observed[i])) &&
                            std::abs(static_cast<long double>(observed[i]) - expected[i]) <=
                                tolerance * (1 + std::abs(static_cast<long double>(expected[i]))),
                        "complete numerical reduction matches independent scalar oracle");
                } else
                    expect(observed[i] == expected[i],
                           "every output guard and stride hole is exact");
            }
            auto input_observed = input;
            download(input_memory.get(), input_observed);
            expect(input_observed == input, "input elements, padding and guards are unchanged");
            indices.verify(operation == 0);
            workspace.verify(workspace_bytes == 0);
            if (operation != 0) {
                std::array<std::uint32_t, 2> actual{};
                std::memcpy(actual.data(), indices.observed.data() + kByteGuard, sizeof(actual));
                for (int n = 0; n != 2; ++n)
                    expect(actual[static_cast<std::size_t>(n)] ==
                               oracle(n, kOperations[operation]).index,
                           "indices are relative flattened reduced coordinates, not byte offsets");
                ++evidence.indexed_calls;
            }
            ++evidence.calls;
            evidence.mixed_calls += mixed ? 1U : 0U;
        }
        workspace.memory.close();
        indices.memory.close();
    }
    output_memory.close();
    input_memory.close();
}
}  // namespace

int main() {
    try {
        Descriptors<float, float> single;
        Descriptors<double, double> dual;
        Descriptors<std::int8_t, float> mixed_input;
        Descriptors<float, std::int8_t> mixed_output;
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
        Evidence evidence;
        run(handle.get(), stream.get(), single, evidence);
        run(handle.get(), stream.get(), dual, evidence);
        run(handle.get(), stream.get(), mixed_input, evidence);
        run(handle.get(), stream.get(), mixed_output, evidence);
        expect(evidence.calls == 36 && evidence.indexed_calls == 24 && evidence.mixed_calls == 18 &&
                   evidence.ignored_null_calls == 4 && evidence.ignored_end_calls == 4,
               "complete original/mixed/indexed/optional-pointer matrix");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind stream");
        handle.close();
        stream.close();
        context.close();
        mixed_output.close();
        mixed_input.close();
        dual.close();
        single.close();
        expect(!cleanup_failed, "checked once-only cleanup");
        std::cout << "cuDNN reduction input passed: calls=" << evidence.calls
                  << "; indexed_calls=" << evidence.indexed_calls
                  << "; mixed_calls=" << evidence.mixed_calls
                  << "; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
