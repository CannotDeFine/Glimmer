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

// Ordinary installed-SDK input, compiled only during development. It makes no
// Driver/device calls and does not qualify GPU computation or another SDK's
// InitTransformDest classification. Native and owned execution remain separate.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
namespace {
bool cleanup_failed{};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS)
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
        if (release() != CUDNN_STATUS_SUCCESS) cleanup_failed = true;
    }
    Handle* output() {
        return &value_;
    }
    Handle get() const {
        return value_;
    }
    void close() {
        checked(release(), "native descriptor cleanup");
    }

   private:
    cudnnStatus_t release() noexcept {
        if (!value_) return CUDNN_STATUS_SUCCESS;
        return Destroy(std::exchange(value_, Handle{}));
    }
    Handle value_{};
};
using OpTensor = Resource<cudnnOpTensorDescriptor_t, cudnnDestroyOpTensorDescriptor>;
using Transform = Resource<cudnnTensorTransformDescriptor_t, cudnnDestroyTensorTransformDescriptor>;
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
constexpr std::uint64_t kBefore = 0x123456789abcdef0, kAfter = 0xfedcba9876543210;
template <typename Value>
struct Guarded {
    std::uint64_t before{kBefore};
    Value value;
    std::uint64_t after{kAfter};
    void verify() const {
        expect(before == kBefore && after == kAfter, "all host-output guard words remain exact");
    }
};
struct Evidence {
    unsigned op_sets{}, op_gets{}, transform_sets{}, destinations{}, aliases{};
};
void pointwise(Evidence& evidence) {
    OpTensor descriptor;
    checked(cudnnCreateOpTensorDescriptor(descriptor.output()), "create pointwise metadata");
    constexpr std::array operations{CUDNN_OP_TENSOR_ADD, CUDNN_OP_TENSOR_MUL,  CUDNN_OP_TENSOR_MIN,
                                    CUDNN_OP_TENSOR_MAX, CUDNN_OP_TENSOR_SQRT, CUDNN_OP_TENSOR_NOT};
    for (const auto operation : operations) {
        for (const auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE}) {
            for (const auto nan : {CUDNN_NOT_PROPAGATE_NAN, CUDNN_PROPAGATE_NAN}) {
                checked(cudnnSetOpTensorDescriptor(descriptor.get(), operation, type, nan),
                        "configure original pointwise descriptor");
                ++evidence.op_sets;
                for (unsigned mask = 0; mask != 8; ++mask) {
                    constexpr auto untouched_operation = static_cast<cudnnOpTensorOp_t>(7);
                    Guarded<cudnnOpTensorOp_t> actual_operation{.value = untouched_operation};
                    Guarded<cudnnDataType_t> actual_type{.value = CUDNN_DATA_INT32};
                    Guarded<cudnnNanPropagation_t> actual_nan{.value = CUDNN_NOT_PROPAGATE_NAN};
                    checked(cudnnGetOpTensorDescriptor(descriptor.get(),
                                                       mask & 1 ? &actual_operation.value : nullptr,
                                                       mask & 2 ? &actual_type.value : nullptr,
                                                       mask & 4 ? &actual_nan.value : nullptr),
                            "query independent nullable pointwise outputs");
                    ++evidence.op_gets;
                    actual_operation.verify();
                    actual_type.verify();
                    actual_nan.verify();
                    expect(actual_operation.value == (mask & 1 ? operation : untouched_operation) &&
                               actual_type.value == (mask & 2 ? type : CUDNN_DATA_INT32) &&
                               actual_nan.value == (mask & 4 ? nan : CUDNN_NOT_PROPAGATE_NAN),
                           "every present field matches configured metadata; absent fields stay "
                           "exact");
                }
            }
        }
    }
    descriptor.close();
}
struct TensorState {
    cudnnDataType_t type{};
    int dimensions{};
    std::array<int, 4> shape{}, strides{};
    friend bool operator==(const TensorState&, const TensorState&) = default;
};
TensorState query(cudnnTensorDescriptor_t tensor) {
    Guarded<cudnnDataType_t> type{.value = CUDNN_DATA_INT32};
    Guarded<int> rank{.value = -19};
    Guarded<std::array<int, 4>> shape{.value = {-7, -7, -7, -7}};
    Guarded<std::array<int, 4>> strides{.value = {-9, -9, -9, -9}};
    checked(cudnnGetTensorNdDescriptor(tensor, 4, &type.value, &rank.value, shape.value.data(),
                                       strides.value.data()),
            "query complete initialized destination tensor");
    type.verify();
    rank.verify();
    shape.verify();
    strides.verify();
    return {type.value, rank.value, shape.value, strides.value};
}
std::array<int, 4> packed_strides(const std::array<int, 4>& shape, cudnnTensorFormat_t format) {
    if (format == CUDNN_TENSOR_NHWC)
        return {shape[1] * shape[2] * shape[3], 1, shape[3] * shape[1], shape[1]};
    return {shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1};
}
std::size_t packed_bytes(const std::array<int, 4>& shape, cudnnDataType_t type) {
    std::size_t result = type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float);
    for (const auto size : shape) {
        expect(size > 0 && result <= std::numeric_limits<std::size_t>::max() /
                                         static_cast<std::size_t>(size),
               "bounded independent packed-size oracle");
        result *= static_cast<std::size_t>(size);
    }
    return result;
}
void transformations(Evidence& evidence) {
    Transform transform;
    Tensor source, destination;
    checked(cudnnCreateTensorTransformDescriptor(transform.output()), "create transform metadata");
    checked(cudnnCreateTensorDescriptor(source.output()), "create source metadata");
    checked(cudnnCreateTensorDescriptor(destination.output()), "create destination metadata");
    constexpr std::array<int, 4> shape{2, 3, 4, 5}, strides{100, 30, 7, 1};
    constexpr std::array<std::int32_t, 4> before{0, 0, 1, 2}, after{0, 0, 2, 1};
    // Exactly rank-2 elements: reading a rank-sized fold array is a defect.
    constexpr std::array<std::uint32_t, 2> identity_fold{1, 1};
    for (const auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE}) {
        for (const auto format : {CUDNN_TENSOR_NCHW, CUDNN_TENSOR_NHWC}) {
            checked(cudnnSetTensorNdDescriptor(source.get(), type, 4, shape.data(), strides.data()),
                    "configure independent padded-stride source");
            const TensorState expected_source{type, 4, shape, strides};
            expect(query(source.get()) == expected_source, "source metadata matches raw inputs");
            for (unsigned mask = 0; mask != 8; ++mask) {
                checked(cudnnSetTensorTransformDescriptor(
                            transform.get(), 4, format, mask & 1 ? before.data() : nullptr,
                            mask & 2 ? after.data() : nullptr,
                            mask & 4 ? identity_fold.data() : nullptr, CUDNN_TRANSFORM_FOLD),
                        "configure independent optional padding/fold pointers");
                ++evidence.transform_sets;
                auto expected_shape = shape;
                for (std::size_t i = 0; i != shape.size(); ++i)
                    expected_shape[i] += (mask & 1 ? before[i] : 0) + (mask & 2 ? after[i] : 0);
                Guarded<std::size_t> bytes{.value = SIZE_MAX};
                const bool has_size = (mask & 1) != 0;
                checked(cudnnInitTransformDest(transform.get(), source.get(), destination.get(),
                                               has_size ? &bytes.value : nullptr),
                        "initialize existing destination with optional size output");
                ++evidence.destinations;
                bytes.verify();
                const TensorState expected{type, 4, expected_shape,
                                           packed_strides(expected_shape, format)};
                expect(
                    query(destination.get()) == expected &&
                        query(source.get()) == expected_source &&
                        bytes.value == (has_size ? packed_bytes(expected_shape, type) : SIZE_MAX),
                    "complete packed destination and byte size match independent padding oracle");
            }
            // Avoid assuming undocumented nonidentity unfolding or padding
            // arithmetic: this checks its explicitly identity configuration.
            checked(cudnnSetTensorTransformDescriptor(transform.get(), 4, format, nullptr, nullptr,
                                                      identity_fold.data(), CUDNN_TRANSFORM_UNFOLD),
                    "configure identity unfolding");
            ++evidence.transform_sets;
            Guarded<std::size_t> bytes{.value = SIZE_MAX};
            checked(cudnnInitTransformDest(transform.get(), source.get(), destination.get(),
                                           &bytes.value),
                    "initialize identity-unfold destination");
            ++evidence.destinations;
            const TensorState expected{type, 4, shape, packed_strides(shape, format)};
            expect(query(destination.get()) == expected && query(source.get()) == expected_source &&
                       bytes.value == packed_bytes(shape, type),
                   "identity unfolding preserves source and produces packed destination");
            bytes.verify();
            // The exact same object is both input and output, not a newly
            // created handle and not two mutually blocking immutable borrows.
            checked(cudnnInitTransformDest(transform.get(), source.get(), source.get(), nullptr),
                    "initialize aliased source/destination identity transform");
            ++evidence.destinations;
            ++evidence.aliases;
            expect(query(source.get()) == expected, "identity alias retains exact packed metadata");
        }
    }
    destination.close();
    source.close();
    transform.close();
}
}  // namespace
int main() {
    try {
        Evidence evidence;
        pointwise(evidence);
        transformations(evidence);
        expect(evidence.op_sets == 24 && evidence.op_gets == 192 && evidence.transform_sets == 36 &&
                   evidence.destinations == 40 && evidence.aliases == 4,
               "complete nullable-output, layout, scalar-type and identity-alias matrix");
        expect(!cleanup_failed, "all descriptor destruction statuses were checked");
        std::cout
            << "cuDNN descriptor input passed: op_gets=" << evidence.op_gets
            << "; destination_initializations=" << evidence.destinations
            << "; explicit_device_calls=0; full_device_qualified=false; priority_qualified=false\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#pragma GCC diagnostic pop
