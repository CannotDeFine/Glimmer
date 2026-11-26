#include <cuda.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Compile-only ordinary SDK input. The original API observes numeric addresses
// and descriptor layouts; no device allocation, transfer or compute is requested.
namespace {
constexpr std::size_t kMaxWeightBytes = 64 * 1024;
bool has_failure{};
unsigned int calls{}, configurations{};
void expect(bool value, const char* message) {
    if (!value) {
        has_failure = true;
        throw std::runtime_error(message);
    }
}
template <typename Status>
void checked(Status status, const char* operation) {
    if (static_cast<int>(status)) {
        has_failure = true;
        throw std::runtime_error(std::string(operation) + ": " +
                                 std::to_string(static_cast<int>(status)));
    }
}
template <typename Handle, auto Destroy>
class Resource final {
   public:
    Resource() = default;
    Resource(const Resource&) = delete;
    Resource& operator=(const Resource&) = delete;
    ~Resource() {
        if (value_ && !has_failure && static_cast<int>(Destroy(std::exchange(value_, {}))))
            has_failure = true;
    }
    Handle get() const {
        return value_;
    }
    Handle* output() {
        return &value_;
    }

   private:
    Handle value_{};
};
using Context = Resource<CUcontext, cuCtxDestroy>;
using Handle = Resource<cudnnHandle_t, cudnnDestroy>;
using Attention = Resource<cudnnAttnDescriptor_t, cudnnDestroyAttnDescriptor>;
using Tensor = Resource<cudnnTensorDescriptor_t, cudnnDestroyTensorDescriptor>;
struct Shape {
    cudnnDataType_t type{};
    int rank{};
    std::array<int, 8> dimensions{}, strides{};
    bool operator==(const Shape&) const = default;
};
Shape observe(cudnnTensorDescriptor_t tensor) {
    struct Guarded {
        std::uint64_t before{0x123456789abcdef0ULL};
        std::array<int, 8> values;
        std::uint64_t after{0xfedcba9876543210ULL};
        Guarded() {
            values.fill(-8193);
        }
    } dimensions, strides;
    Shape result;
    checked(cudnnGetTensorNdDescriptor(tensor, 8, &result.type, &result.rank,
                                       dimensions.values.data(), strides.values.data()),
            "read original attention parameter descriptor");
    expect(result.rank == 0 || result.rank == 3, "bounded original attention parameter rank");
    for (const auto* value : {&dimensions, &strides}) {
        expect(value->before == 0x123456789abcdef0ULL && value->after == 0xfedcba9876543210ULL,
               "descriptor getter preserves every outer guard");
        for (std::size_t i = static_cast<std::size_t>(result.rank); i != value->values.size(); ++i)
            expect(value->values[i] == -8193, "descriptor getter preserves every unused entry");
    }
    std::copy_n(dimensions.values.begin(), result.rank, result.dimensions.begin());
    std::copy_n(strides.values.begin(), result.rank, result.strides.begin());
    return result;
}
struct Observation {
    bool is_present{};
    std::uint64_t offset{};
    Shape shape;
    bool operator==(const Observation&) const = default;
};
Observation query(cudnnHandle_t handle, cudnnAttnDescriptor_t attention,
                  cudnnTensorDescriptor_t tensor, unsigned int kind, const void* base,
                  std::size_t bytes) {
    struct Guarded {
        std::uint64_t before{0x718293a4b5c6d7e8ULL};
        void* value{reinterpret_cast<void*>(std::uintptr_t{0x7654321})};
        std::uint64_t after{0x8e7d6c5b4a392817ULL};
    } output;
    checked(cudnnGetMultiHeadAttnWeights(handle, attention,
                                         static_cast<cudnnMultiHeadAttnWeightKind_t>(kind), bytes,
                                         base, tensor, &output.value),
            "observe original attention parameter identity");
    ++calls;
    expect(output.before == 0x718293a4b5c6d7e8ULL && output.after == 0x8e7d6c5b4a392817ULL,
           "layout address output preserves adjacent guards");
    Observation result;
    result.is_present = output.value != nullptr;
    if (result.is_present) {
        const auto pointer = reinterpret_cast<std::uintptr_t>(output.value);
        const auto original = reinterpret_cast<std::uintptr_t>(base);
        expect(original && pointer >= original && pointer - original < bytes,
               "nonnull parameter lies strictly inside the declared numeric interval");
        result.offset = pointer - original;
    }
    result.shape = observe(tensor);
    std::cout << "attn_weights_case=" << calls << "; configuration=" << configurations
              << "; kind=" << kind << "; base_present=" << (base != nullptr)
              << "; capacity=" << bytes << "; present=" << result.is_present
              << "; offset=" << result.offset << "; type=" << static_cast<int>(result.shape.type)
              << "; rank=" << result.shape.rank << "; dimensions=";
    for (const auto value : result.shape.dimensions) std::cout << value << ',';
    std::cout << "; strides=";
    for (const auto value : result.shape.strides) std::cout << value << ',';
    std::cout << '\n';
    return result;
}
void check_elements(const Observation& result, unsigned int kind, cudnnDataType_t type,
                    std::vector<bool>& covered) {
    expect(result.shape.type == type && result.shape.rank == 3 && result.shape.dimensions[0] == 1 &&
               result.shape.dimensions[1] == 2 && result.shape.dimensions[2] == (kind < 4 ? 2 : 1),
           "all logical parameter shapes follow the original one-head projection contract");
    for (std::size_t i = 0; i != 3; ++i)
        expect(result.shape.strides[i] >= 0 &&
                   (result.shape.dimensions[i] <= 1 || result.shape.strides[i] > 0),
               "active original strides describe bounded distinct elements");
    const auto width = type == CUDNN_DATA_DOUBLE ? std::size_t{8} : std::size_t{4};
    for (std::size_t row = 0; row != 2; ++row)
        for (int col = 0; col != result.shape.dimensions[2]; ++col) {
            const auto index = row * static_cast<std::size_t>(result.shape.strides[1]) +
                               static_cast<std::size_t>(col) * result.shape.strides[2];
            expect(result.offset <= covered.size() &&
                       index <= (covered.size() - result.offset) / width,
                   "checked native-stride scalar offset");
            const auto offset = static_cast<std::size_t>(result.offset) + index * width;
            expect(
                offset <= covered.size() && width <= covered.size() - offset && offset % width == 0,
                "every original mapped scalar is complete and aligned within capacity");
            for (std::size_t i = 0; i != width; ++i) {
                expect(!covered[offset + i],
                       "all distinct logical parameters have disjoint storage");
                covered[offset + i] = true;
            }
        }
}
void run_configuration(cudnnHandle_t handle, cudnnDataType_t type, bool projection, bool biases) {
    ++configurations;
    Attention attention;
    Tensor tensor;
    checked(cudnnCreateAttnDescriptor(attention.output()), "create original attention metadata");
    checked(cudnnCreateTensorDescriptor(tensor.output()), "create original mutable layout output");
    const auto projected = projection ? 2 : 0;
    checked(cudnnSetAttnDescriptor(attention.get(), biases ? CUDNN_ATTN_ENABLE_PROJ_BIASES : 0U, 1,
                                   .75, type, type, CUDNN_DEFAULT_MATH, nullptr, nullptr, 2, 2, 2,
                                   projected, projected, projected, projected, 2, 3, 1, 1),
            "configure original one-head attention projections");
    std::size_t bytes{}, workspace{}, reserve{};
    checked(cudnnGetMultiHeadAttnBuffers(handle, attention.get(), &bytes, &workspace, &reserve),
            "obtain original parameter capacity without assuming vendor packing");
    expect(bytes <= kMaxWeightBytes && (projection ? bytes != 0 : bytes == 0),
           "bounded original parameter capacity matches presence of projections");
    std::vector<std::byte> host(bytes + 512, std::byte{0xa7});
    std::vector<bool> covered(bytes);
    // Two numeric/device-like addresses intentionally have no readable payload.
    // The API is address arithmetic only; they establish no device ownership.
    const std::array<const void*, 3> bases{
        host.data() + 256, reinterpret_cast<const void*>(std::uintptr_t{1}),
        reinterpret_cast<const void*>(std::uintptr_t{0x1000000000ULL})};
    for (unsigned int kind = 0; kind != 8; ++kind) {
        const auto original = query(handle, attention.get(), tensor.get(), kind, bases[0], bytes);
        const auto is_present = projection && (kind < 4 || biases);
        expect(original.is_present == is_present,
               "disabled projections/biases have original null mappings");
        if (is_present)
            check_elements(original, kind, type, covered);
        else
            expect(original.shape.rank == 0, "absent original parameter zeroes the tensor rank");
        for (std::size_t i = 1; i != bases.size(); ++i)
            expect(query(handle, attention.get(), tensor.get(), kind, bases[i], bytes) == original,
                   "all original shapes, strides and relative offsets are base-independent");
        if (!projection)
            expect(query(handle, attention.get(), tensor.get(), kind, nullptr, 0) == original,
                   "null base and zero capacity preserve original absent-parameter success");
    }
    const auto scalars = projection ? (biases ? 24 : 16) : 0;
    const auto width = type == CUDNN_DATA_DOUBLE ? 8 : 4;
    expect(std::ranges::count(covered, true) == scalars * width &&
               std::ranges::all_of(host, [](std::byte byte) { return byte == std::byte{0xa7}; }),
           "complete logical mapping and all host payload/guard bytes are unchanged");
}
}  // namespace

int main() {
    try {
        {
            expect(
                cudnnGetVersion() == 91002,
                "numeric-base input requires the statically audited cuDNN 9.10.2 implementation");
            checked(cuInit(0), "initialize original Driver");
            CUdevice device{};
            checked(cuDeviceGet(&device, 0), "select diagnostic context device");
            Context context;
            checked(cuCtxCreate(context.output(), 0, device), "create diagnostic context");
            Handle handle;
            checked(cudnnCreate(handle.output()), "create original cuDNN handle");
            for (const auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE}) {
                run_configuration(handle.get(), type, true, false);
                run_configuration(handle.get(), type, true, true);
                run_configuration(handle.get(), type, false, false);
            }
            expect(configurations == 6 && calls == 160, "complete original layout query matrix");
        }
        expect(!has_failure, "all known original owners cleaned up successfully");
        std::cout << "cuDNN attention layout input passed: configurations=" << configurations
                  << "; layout_queries=" << calls
                  << "; explicit_device_allocations=0; explicit_device_transfers=0;"
                     " explicit_compute_calls=0; numerical_gradients_qualified=false;"
                     " full_device_qualified=false; priority_qualified=false\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
