#include <cuda.h>
#include <cudnn.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

// Compile-only ordinary SDK input for later matched native/owned execution.
// Record every field observable through the supported tensor/filter/convolution
// getters, not an invented folding algorithm. TensorTransform's getter remains
// an explicit Glimmer guard, so transform internals are NOT observed here. A
// complete transform-state/numerical acceptance requires separate future work.
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
        if (const auto status = release(); status != 0) {
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
using Transform = Resource<cudnnTensorTransformDescriptor_t, cudnnDestroyTensorTransformDescriptor>;
constexpr int kSentinel = 0x2345678;
struct GuardedDimensions {
    int prefix{kSentinel};
    std::array<int, 8> values{kSentinel, kSentinel, kSentinel, kSentinel,
                              kSentinel, kSentinel, kSentinel, kSentinel};
    int suffix{kSentinel};
    std::array<int, 8> checked_values(int rank) const {
        expect(rank >= 0 && rank <= 8, "native rank fits fixed dimension capacity");
        expect(prefix == kSentinel && suffix == kSentinel, "dimension array guards exact");
        std::array<int, 8> result{};
        for (std::size_t index = 0; index != result.size(); ++index) {
            if (index < static_cast<std::size_t>(rank))
                result[index] = values[index];
            else
                expect(values[index] == kSentinel, "all unused output dimensions untouched");
        }
        return result;
    }
};
struct TensorState {
    int type{}, rank{};
    std::array<int, 8> dimensions{}, strides{};
    std::size_t bytes{};
    bool operator==(const TensorState&) const = default;
};
struct FilterState {
    int type{}, format{}, rank{};
    std::array<int, 8> dimensions{};
    std::size_t bytes{};
    bool operator==(const FilterState&) const = default;
};
struct ConvolutionState {
    int rank{}, mode{}, type{}, group{}, math{}, reorder{};
    std::array<int, 8> padding{}, strides{}, dilation{};
    bool operator==(const ConvolutionState&) const = default;
};
TensorState observe(cudnnTensorDescriptor_t descriptor) {
    cudnnDataType_t type{};
    int rank{};
    GuardedDimensions dimensions, strides;
    TensorState value;
    checked(cudnnGetTensorNdDescriptor(descriptor, 8, &type, &rank, dimensions.values.data(),
                                       strides.values.data()),
            "observe complete tensor metadata");
    value.type = static_cast<int>(type);
    value.rank = rank;
    value.dimensions = dimensions.checked_values(rank);
    value.strides = strides.checked_values(rank);
    checked(cudnnGetTensorSizeInBytes(descriptor, &value.bytes), "observe tensor byte size");
    return value;
}
FilterState observe(cudnnFilterDescriptor_t descriptor) {
    cudnnDataType_t type{};
    cudnnTensorFormat_t format{};
    int rank{};
    GuardedDimensions dimensions;
    FilterState value;
    checked(
        cudnnGetFilterNdDescriptor(descriptor, 8, &type, &format, &rank, dimensions.values.data()),
        "observe complete filter metadata");
    value.type = static_cast<int>(type);
    value.format = static_cast<int>(format);
    value.rank = rank;
    value.dimensions = dimensions.checked_values(rank);
    checked(cudnnGetFilterSizeInBytes(descriptor, &value.bytes), "observe filter byte size");
    return value;
}
ConvolutionState observe(cudnnConvolutionDescriptor_t descriptor) {
    cudnnConvolutionMode_t mode{};
    cudnnDataType_t type{};
    cudnnMathType_t math{};
    cudnnReorderType_t reorder{};
    GuardedDimensions padding, strides, dilation;
    ConvolutionState value;
    // The configured getter writes the requested count, not min(count, rank).
    // This matrix is strictly 2D; request two before checking untouched tails.
    checked(cudnnGetConvolutionNdDescriptor(descriptor, 2, &value.rank, padding.values.data(),
                                            strides.values.data(), dilation.values.data(), &mode,
                                            &type),
            "observe complete convolution metadata");
    expect(value.rank == 2, "all observed convolution descriptors remain two-dimensional");
    value.mode = static_cast<int>(mode);
    value.type = static_cast<int>(type);
    value.padding = padding.checked_values(value.rank);
    value.strides = strides.checked_values(value.rank);
    value.dilation = dilation.checked_values(value.rank);
    checked(cudnnGetConvolutionGroupCount(descriptor, &value.group), "observe groups");
    checked(cudnnGetConvolutionMathType(descriptor, &math), "observe math type");
    checked(cudnnGetConvolutionReorderType(descriptor, &reorder), "observe reorder type");
    value.math = static_cast<int>(math);
    value.reorder = static_cast<int>(reorder);
    return value;
}
void print_array(const char* label, const std::array<int, 8>& values) {
    std::cout << ' ' << label << '=';
    for (const auto value : values) std::cout << value << ',';
}
void print(const char* label, const TensorState& value) {
    std::cout << label << " type=" << value.type << " rank=" << value.rank
              << " bytes=" << value.bytes;
    print_array("dimensions", value.dimensions);
    print_array("strides", value.strides);
    std::cout << '\n';
}
void print(const char* label, const FilterState& value) {
    std::cout << label << " type=" << value.type << " format=" << value.format
              << " rank=" << value.rank << " bytes=" << value.bytes;
    print_array("dimensions", value.dimensions);
    std::cout << '\n';
}
void print(const char* label, const ConvolutionState& value) {
    std::cout << label << " type=" << value.type << " mode=" << value.mode << " rank=" << value.rank
              << " group=" << value.group << " math=" << value.math << " reorder=" << value.reorder;
    print_array("padding", value.padding);
    print_array("strides", value.strides);
    print_array("dilation", value.dilation);
    std::cout << '\n';
}
struct Snapshot {
    FilterState filter;
    TensorState diff;
    ConvolutionState convolution;
    TensorState grad;
    bool operator==(const Snapshot&) const = default;
    void print_all(const char* label) const {
        std::cout << "snapshot=" << label << '\n';
        print("filter", filter);
        print("diff", diff);
        print("convolution", convolution);
        print("grad", grad);
    }
};
struct Shape {
    cudnnDataType_t type;
    std::array<int, 4> grad, filter;
    std::array<int, 2> padding, strides;
};
constexpr std::array kShapes{Shape{CUDNN_DATA_FLOAT, {2, 8, 8, 10}, {16, 8, 3, 3}, {1, 1}, {2, 2}},
                             Shape{CUDNN_DATA_HALF, {1, 16, 9, 8}, {8, 16, 3, 2}, {1, 0}, {2, 1}}};
struct Descriptors {
    Filter filter, folded_filter;
    Tensor diff, grad, padded_diff, folded_grad;
    Convolution convolution, folded_convolution;
    std::array<Transform, 4> transforms;
    void initialize(const Shape& shape) {
        for (auto* value : {&filter, &folded_filter})
            checked(cudnnCreateFilterDescriptor(value->output()), "create filter");
        for (auto* value : {&diff, &grad, &padded_diff, &folded_grad})
            checked(cudnnCreateTensorDescriptor(value->output()), "create tensor");
        for (auto* value : {&convolution, &folded_convolution})
            checked(cudnnCreateConvolutionDescriptor(value->output()), "create convolution");
        for (auto& value : transforms)
            checked(cudnnCreateTensorTransformDescriptor(value.output()), "create transform");
        checked(cudnnSetFilterNdDescriptor(filter.get(), shape.type, CUDNN_TENSOR_NCHW, 4,
                                           shape.filter.data()),
                "initialize filter");
        const std::array dilation{1, 1};
        checked(cudnnSetConvolutionNdDescriptor(convolution.get(), 2, shape.padding.data(),
                                                shape.strides.data(), dilation.data(),
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                "initialize positive-stride convolution");
        auto diff_shape = shape.grad;
        diff_shape[1] = shape.filter[0];
        for (std::size_t axis = 2; axis != diff_shape.size(); ++axis)
            diff_shape[axis] =
                (shape.grad[axis] + 2 * shape.padding[axis - 2] - shape.filter[axis]) /
                    shape.strides[axis - 2] +
                1;
        const auto set_tensor = [&](cudnnTensorDescriptor_t value, const auto& dimensions) {
            std::array<int, 4> strides{};
            int stride = 1;
            for (std::size_t axis = strides.size(); axis != 0; --axis) {
                strides[axis - 1] = stride;
                stride *= dimensions[axis - 1];
            }
            checked(
                cudnnSetTensorNdDescriptor(value, shape.type, 4, dimensions.data(), strides.data()),
                "initialize tensor");
        };
        set_tensor(grad.get(), shape.grad);
        set_tensor(diff.get(), diff_shape);
    }
    Snapshot input() const {
        return {observe(filter.get()), observe(diff.get()), observe(convolution.get()),
                observe(grad.get())};
    }
    Snapshot output(bool is_alias) const {
        return {observe(folded_filter.get()), observe(padded_diff.get()),
                observe(is_alias ? convolution.get() : folded_convolution.get()),
                observe(folded_grad.get())};
    }
    void close() {
        for (auto& value : transforms) value.close();
        folded_convolution.close();
        convolution.close();
        folded_grad.close();
        padded_diff.close();
        grad.close();
        diff.close();
        folded_filter.close();
        filter.close();
    }
};
}  // namespace

int main() {
    try {
        checked(cuInit(0), "initialize Driver");
        CUdevice device{};
        checked(cuDeviceGet(&device, 0), "select device");
        Resource<CUcontext, cuCtxDestroy> context;
        checked(cuCtxCreate(context.output(), 0, device), "create owned context");
        Resource<cudnnHandle_t, cudnnDestroy> handle;
        checked(cudnnCreate(handle.output()), "create original handle");
        unsigned int calls{};
        for (std::size_t shape = 0; shape != kShapes.size(); ++shape)
            for (const auto format : {CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC})
                for (const bool has_last : {false, true})
                    for (const bool is_alias : {false, true}) {
                        Snapshot baseline;
                        for (unsigned int repeat = 0; repeat != 2; ++repeat) {
                            Descriptors descriptors;
                            descriptors.initialize(kShapes[shape]);
                            const auto before = descriptors.input();
                            const auto status = cudnnGetFoldedConvBackwardDataDescriptors(
                                handle.get(), descriptors.filter.get(), descriptors.diff.get(),
                                descriptors.convolution.get(), descriptors.grad.get(), format,
                                descriptors.folded_filter.get(), descriptors.padded_diff.get(),
                                is_alias ? descriptors.convolution.get()
                                         : descriptors.folded_convolution.get(),
                                descriptors.folded_grad.get(), descriptors.transforms[0].get(),
                                descriptors.transforms[1].get(), descriptors.transforms[2].get(),
                                has_last ? descriptors.transforms[3].get() : nullptr);
                            ++calls;
                            std::cout << "shape=" << shape << " format=" << static_cast<int>(format)
                                      << " has_last=" << has_last
                                      << " convolution_alias=" << is_alias << " repeat=" << repeat
                                      << " original_status=" << static_cast<int>(status) << '\n';
                            // An actual unsupported case remains a failed input,
                            // not successful coverage or a synthetic getter result.
                            checked(status, "single original Folded mutation");
                            const auto after = descriptors.input();
                            expect(after.filter == before.filter && after.diff == before.diff &&
                                       after.grad == before.grad &&
                                       (is_alias || after.convolution == before.convolution),
                                   "all nonaliased input fields unchanged");
                            const auto output = descriptors.output(is_alias);
                            if (repeat == 0) baseline = output;
                            expect(output == baseline,
                                   "all observable output fields repeat exactly");
                            before.print_all("input-before");
                            after.print_all("input-after");
                            output.print_all("output");
                            descriptors.close();
                        }
                    }
        expect(calls == 32, "complete bounded Folded matrix");
        handle.close();
        context.close();
        expect(!cleanup_failed, "all cleanup checked without retries");
        std::cout << "complete original_folded_calls=" << calls
                  << " observable_tensor_filter_convolution_fields=true"
                     " transform_internal_fields_observed=false transform_getter_calls=0"
                     " complete_transform_oracle=false native_owned_comparison_required=true"
                     " explicit_device_allocations=0 explicit_compute_calls=0"
                     " full_device_qualified=false priority_qualified=false\n";
        expect(static_cast<bool>(std::cout), "complete observable snapshot receipt written");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
