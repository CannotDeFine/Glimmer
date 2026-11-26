#include <cuda.h>
#include <cudnn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

// Ordinary installed-SDK query input, compiled but not automatically executed.
// It allocates no tensor/workspace buffers and runs no backward computation.
// Later native/owned comparison must match SDK, context and descriptor inputs.
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
        const auto status = release();
        if (status != 0) {
            cleanup_failed = true;
            try {
                std::cerr << "Cleanup failed without retry: " << status << '\n';
            } catch (...) {
            }
        }
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
using Filter = Resource<cudnnFilterDescriptor_t, cudnnDestroyFilterDescriptor>;
using Convolution = Resource<cudnnConvolutionDescriptor_t, cudnnDestroyConvolutionDescriptor>;
template <typename Value>
std::array<Value, 3> guarded() {
    return {static_cast<Value>(0x1234567), std::numeric_limits<Value>::max(),
            static_cast<Value>(0x2345678)};
}
template <typename Value>
void guards(const std::array<Value, 3>& output) {
    const auto initial = guarded<Value>();
    expect(output.front() == initial.front() && output.back() == initial.back(),
           "complete host output prefix/suffix remain exact");
}
struct Shape {
    cudnnDataType_t type;
    int rank;
    std::array<int, 5> dx, filter;
    std::array<int, 3> padding, strides, dilation;
};
constexpr std::array kShapes{
    Shape{CUDNN_DATA_FLOAT, 4, {2, 3, 7, 6, 0}, {4, 3, 3, 2, 0}, {1, 0, 0}, {2, 1, 0}, {1, 1, 0}},
    Shape{CUDNN_DATA_DOUBLE, 4, {1, 2, 9, 8, 0}, {3, 2, 2, 3, 0}, {0, 1, 0}, {1, 2, 0}, {2, 1, 0}},
    Shape{CUDNN_DATA_FLOAT, 5, {2, 2, 5, 6, 7}, {3, 2, 2, 3, 2}, {0, 1, 0}, {1, 2, 1}, {1, 1, 1}}};
struct Descriptors {
    Tensor dx, dy;
    Filter filter;
    Convolution convolution;
    void initialize(const Shape& shape) {
        checked(cudnnCreateTensorDescriptor(dx.output()), "create dx");
        checked(cudnnCreateTensorDescriptor(dy.output()), "create dy");
        checked(cudnnCreateFilterDescriptor(filter.output()), "create filter");
        checked(cudnnCreateConvolutionDescriptor(convolution.output()), "create convolution");
        checked(cudnnSetFilterNdDescriptor(filter.get(), shape.type, CUDNN_TENSOR_NCHW, shape.rank,
                                           shape.filter.data()),
                "set original filter");
        checked(cudnnSetConvolutionNdDescriptor(
                    convolution.get(), shape.rank - 2, shape.padding.data(), shape.strides.data(),
                    shape.dilation.data(), CUDNN_CROSS_CORRELATION, shape.type),
                "set original convolution");
        auto output_shape = shape.dx;
        output_shape[1] = shape.filter[0];
        for (int axis = 2; axis != shape.rank; ++axis) {
            const auto spatial = static_cast<std::size_t>(axis - 2);
            const auto index = static_cast<std::size_t>(axis);
            output_shape[index] = (shape.dx[index] + 2 * shape.padding[spatial] -
                                   shape.dilation[spatial] * (shape.filter[index] - 1) - 1) /
                                      shape.strides[spatial] +
                                  1;
        }
        const auto set_tensor = [&](cudnnTensorDescriptor_t tensor,
                                    const std::array<int, 5>& sizes) {
            std::array<int, 5> strides{};
            int stride = 1;
            for (int axis = shape.rank; axis != 0; --axis) {
                const auto index = static_cast<std::size_t>(axis - 1);
                strides[index] = stride;
                stride *= sizes[index];
            }
            checked(cudnnSetTensorNdDescriptor(tensor, shape.type, shape.rank, sizes.data(),
                                               strides.data()),
                    "set independently derived contiguous tensor");
        };
        set_tensor(dx.get(), shape.dx);
        set_tensor(dy.get(), output_shape);
        std::array<int, 5> native_shape{};
        checked(cudnnGetConvolutionNdForwardOutputDim(convolution.get(), dx.get(), filter.get(),
                                                      shape.rank, native_shape.data()),
                "check independently derived dy dimensions");
        expect(native_shape == output_shape,
               "all dy dimensions match CPU formula and differ from dx dimensions");
    }
    void close() {
        convolution.close();
        filter.close();
        dy.close();
        dx.close();
    }
};
struct Observation {
    cudnnStatus_t status{};
    std::size_t bytes{};
    bool operator==(const Observation&) const = default;
};
}  // namespace

int main() {
    try {
        std::array<Descriptors, kShapes.size()> descriptors;
        for (std::size_t index = 0; index != descriptors.size(); ++index)
            descriptors[index].initialize(kShapes[index]);
        checked(cuInit(0), "initialize Driver after metadata");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select device");
        Resource<CUcontext, cuCtxDestroy> context;
        checked(cuCtxCreate(context.output(), 0, device), "create context");
        Resource<CUstream, cuStreamDestroy> stream;
        checked(cuStreamCreate(stream.output(), CU_STREAM_NON_BLOCKING), "create explicit stream");
        Resource<cudnnHandle_t, cudnnDestroy> handle;
        checked(cudnnCreate(handle.output()), "create original handle");
        std::array<int, 2> baseline_counts{};
        std::array<std::array<Observation, 8>, kShapes.size()> baseline{};
        unsigned int count_calls{}, workspace_calls{}, null_workspace_calls{};
        for (unsigned int pass = 0; pass != 4; ++pass) {
            checked(
                cudnnSetStream(handle.get(),
                               pass % 2 ? reinterpret_cast<cudaStream_t>(stream.get()) : nullptr),
                "alternate default and explicit streams");
            auto data_count = guarded<int>();
            auto filter_count = guarded<int>();
            checked(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle.get(), &data_count[1]),
                    "query original backward-data count");
            checked(
                cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(handle.get(), &filter_count[1]),
                "query original backward-filter count");
            count_calls += 2;
            guards(data_count);
            guards(filter_count);
            expect(data_count[1] > 0 && filter_count[1] > 0,
                   "both original counts are positive without a fabricated fixed capacity");
            const std::array counts{data_count[1], filter_count[1]};
            if (pass == 0) baseline_counts = counts;
            expect(counts == baseline_counts, "native counts stable across repetition/stream");
            std::cout << "pass=" << pass << " backward_data_count=" << counts[0]
                      << " backward_filter_count=" << counts[1] << '\n';
            // Never execute MaxCount with nullptr: this configured SDK stores
            // through it unconditionally. Glimmer tests containment with fakes.
            for (std::size_t shape = 0; shape != descriptors.size(); ++shape) {
                auto& value = descriptors[shape];
                bool has_success{};
                for (unsigned int algorithm = 0; algorithm != 8; ++algorithm) {
                    auto bytes = guarded<std::size_t>();
                    const auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
                        handle.get(), value.filter.get(), value.dy.get(), value.convolution.get(),
                        value.dx.get(), static_cast<cudnnConvolutionBwdDataAlgo_t>(algorithm),
                        &bytes[1]);
                    ++workspace_calls;
                    guards(bytes);
                    if (status != CUDNN_STATUS_SUCCESS)
                        expect(bytes == guarded<std::size_t>(),
                               "native failure leaves complete host output unchanged");
                    has_success = has_success || status == CUDNN_STATUS_SUCCESS;
                    const Observation observed{status, bytes[1]};
                    if (pass == 0) baseline[shape][algorithm] = observed;
                    expect(observed == baseline[shape][algorithm],
                           "exact status/full-width size stable across repetition/stream");
                    std::cout << "pass=" << pass << " shape=" << shape << " algorithm=" << algorithm
                              << " status=" << static_cast<int>(status) << " bytes=" << bytes[1]
                              << '\n';
                }
                expect(has_success, "each fixed float/double 2D/3D shape has a valid algorithm");
                expect(cudnnGetConvolutionBackwardDataWorkspaceSize(
                           handle.get(), value.filter.get(), value.dy.get(),
                           value.convolution.get(), value.dx.get(),
                           CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr) == CUDNN_STATUS_BAD_PARAM,
                       "workspace preserves its native null-output BAD_PARAM path");
                ++null_workspace_calls;
            }
        }
        expect(count_calls == 8 && workspace_calls == 96 && null_workspace_calls == 12,
               "complete three-query matrix");
        checked(cudnnSetStream(handle.get(), nullptr), "unbind explicit stream");
        handle.close();
        stream.close();
        context.close();
        for (auto& value : descriptors) value.close();
        expect(!cleanup_failed, "all cleanup was checked without retries");
        std::cout << "complete count_calls=" << count_calls
                  << " workspace_calls=" << workspace_calls
                  << " null_workspace_calls=" << null_workspace_calls
                  << " explicit_compute_calls=0 native_owned_comparison_required=true"
                     " full_device_qualified=false priority_qualified=false\n";
        expect(static_cast<bool>(std::cout), "complete query receipt is written");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
