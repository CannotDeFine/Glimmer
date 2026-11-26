#pragma once

#include "dnn_convolution_fixture.h"

#include <cmath>

namespace glimmer::tests::dnn_reduction {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace foundation = glimmer::tests::dnn_convolution;
struct Descriptor {
    bool alive{}, configured{};
    CudaDnnReductionState value{};
};
struct Invocation {
    cudnnHandle_t handle{};
    CudaDnnReduction reduction{};
    CudaDnnReductionState parameters{};
    cudnnTensorDescriptor_t input_descriptor{}, output_descriptor{};
    const void* input{};
    void* output{};
    void* indices{};
    void* workspace{};
    std::size_t indices_bytes{}, workspace_bytes{};
    double alpha{}, beta{};
    cudaStream_t stream{};
};
struct State {
    std::array<Descriptor, 16> descriptors{};
    unsigned int creates{}, destroys{}, sets{}, computes{}, indices_queries{}, workspace_queries{};
    Invocation last;
};
inline State state;
inline std::atomic<bool> fail_compute{}, fail_query{}, override_query{};
inline std::atomic<std::size_t> query_value{}, required_workspace{32};
inline std::atomic<std::size_t> observed_indices_bytes{}, observed_workspace_bytes{};
inline std::atomic<bool> observed_null_indices{}, observed_null_workspace{};
inline void reset() {
    foundation::reset();
    state = {};
    fail_compute.store(false);
    fail_query.store(false);
    override_query.store(false);
    query_value.store(0);
    required_workspace.store(32);
    observed_indices_bytes.store(0);
    observed_workspace_bytes.store(0);
    observed_null_indices.store(false);
    observed_null_workspace.store(false);
}
inline Descriptor& descriptor(CudaDnnReduction pointer) {
    base::owner();
    for (auto& item : state.descriptors)
        if (reinterpret_cast<CudaDnnReduction>(&item) == pointer) {
            base::require(item.alive);
            return item;
        }
    throw std::runtime_error("unknown independent native reduction descriptor");
}
inline std::size_t width(std::uint32_t type) {
    base::require(type == CUDNN_DATA_DOUBLE || type == CUDNN_DATA_FLOAT || type == CUDNN_DATA_INT8);
    return type == CUDNN_DATA_DOUBLE ? 8 : type == CUDNN_DATA_FLOAT ? 4 : 1;
}
inline std::size_t elements(const CudaDnnTensor& tensor) {
    std::size_t result{1};
    for (int i = 0; i < tensor.dimensions; ++i)
        result *= static_cast<std::size_t>(tensor.sizes[static_cast<std::size_t>(i)]);
    return result;
}
inline std::size_t bytes(const CudaDnnTensor& tensor) {
    std::size_t result{1};
    for (int i = 0; i < tensor.dimensions; ++i)
        result += static_cast<std::size_t>(tensor.sizes[static_cast<std::size_t>(i)] - 1) *
                  static_cast<std::size_t>(tensor.strides[static_cast<std::size_t>(i)]);
    return result * width(tensor.type);
}
inline std::size_t offset(const CudaDnnTensor& tensor, std::size_t logical) {
    std::size_t result{};
    for (int axis = tensor.dimensions; axis-- > 0;) {
        const auto i = static_cast<std::size_t>(axis);
        const auto dimension = static_cast<std::size_t>(tensor.sizes[i]);
        result += logical % dimension * static_cast<std::size_t>(tensor.strides[i]);
        logical /= dimension;
    }
    return result;
}
inline double read(const void* pointer, std::uint32_t type, std::size_t index) {
    if (type == CUDNN_DATA_DOUBLE) return static_cast<const double*>(pointer)[index];
    if (type == CUDNN_DATA_FLOAT) return static_cast<const float*>(pointer)[index];
    return static_cast<const std::int8_t*>(pointer)[index];
}
inline void write(void* pointer, std::uint32_t type, std::size_t index, double value) {
    if (type == CUDNN_DATA_DOUBLE)
        static_cast<double*>(pointer)[index] = value;
    else if (type == CUDNN_DATA_FLOAT)
        static_cast<float*>(pointer)[index] = static_cast<float>(value);
    else {
        // Mixed-width oracle cases deliberately use exact in-range integers.
        base::require(value >= -128 && value <= 127 && value == std::trunc(value));
        static_cast<std::int8_t*>(pointer)[index] = static_cast<std::int8_t>(value);
    }
}
inline bool needs_indices(const CudaDnnReductionState& value) {
    return value.indices == CUDNN_REDUCE_TENSOR_FLATTENED_INDICES &&
           (value.operation == CUDNN_REDUCE_TENSOR_MIN ||
            value.operation == CUDNN_REDUCE_TENSOR_MAX);
}
template <typename Scalar>
inline void compute(const Invocation& call) {
    const auto& x = base::object(call.input_descriptor, CudaDnnObjectKind::kTensor).tensor;
    const auto& y = base::object(call.output_descriptor, CudaDnnObjectKind::kTensor).tensor;
    base::require(x.dimensions == y.dimensions);
    const auto outputs = elements(y);
    std::vector<std::vector<Scalar>> groups(outputs);
    for (std::size_t logical = 0; logical != elements(x); ++logical) {
        auto rest = logical;
        std::size_t group{}, multiplier{1};
        for (int axis = x.dimensions; axis-- > 0;) {
            const auto i = static_cast<std::size_t>(axis);
            const auto coordinate = rest % static_cast<std::size_t>(x.sizes[i]);
            rest /= static_cast<std::size_t>(x.sizes[i]);
            if (y.sizes[i] != 1) group += coordinate * multiplier;
            multiplier *= static_cast<std::size_t>(y.sizes[i]);
        }
        groups[group].push_back(static_cast<Scalar>(read(call.input, x.type, offset(x, logical))));
    }
    for (std::size_t group = 0; group != outputs; ++group) {
        const auto& values = groups[group];
        const auto operation = call.parameters.operation;
        Scalar result{};
        std::uint32_t selected{};
        if (operation == CUDNN_REDUCE_TENSOR_MUL || operation == CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS)
            result = 1;
        else if (operation == CUDNN_REDUCE_TENSOR_MIN || operation == CUDNN_REDUCE_TENSOR_MAX)
            result = values.front();
        for (std::size_t i = 0; i != values.size(); ++i) {
            const auto value = values[i];
            switch (operation) {
                case CUDNN_REDUCE_TENSOR_ADD:
                case CUDNN_REDUCE_TENSOR_AVG:
                    result += value;
                    break;
                case CUDNN_REDUCE_TENSOR_MUL:
                    result *= value;
                    break;
                case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
                    if (value != 0) result *= value;
                    break;
                case CUDNN_REDUCE_TENSOR_MIN:
                case CUDNN_REDUCE_TENSOR_MAX:
                    if ((operation == CUDNN_REDUCE_TENSOR_MIN && value < result) ||
                        (operation == CUDNN_REDUCE_TENSOR_MAX && value > result)) {
                        result = value;
                        selected = static_cast<std::uint32_t>(i);
                    }
                    break;
                case CUDNN_REDUCE_TENSOR_AMAX:
                    result = std::max(result, std::abs(value));
                    break;
                case CUDNN_REDUCE_TENSOR_NORM1:
                    result += std::abs(value);
                    break;
                case CUDNN_REDUCE_TENSOR_NORM2:
                    result += value * value;
                    break;
                default:
                    throw std::runtime_error("unimplemented independent reduction oracle");
            }
        }
        if (operation == CUDNN_REDUCE_TENSOR_AVG) result /= static_cast<Scalar>(values.size());
        if (operation == CUDNN_REDUCE_TENSOR_NORM2) result = std::sqrt(result);
        const auto destination = offset(y, group);
        write(call.output, y.type, destination,
              static_cast<Scalar>(call.alpha) * result +
                  static_cast<Scalar>(call.beta) *
                      static_cast<Scalar>(read(call.output, y.type, destination)));
        if (needs_indices(call.parameters))
            static_cast<std::uint32_t*>(call.indices)[group] = selected;
    }
    if (required_workspace.load())
        std::fill_n(static_cast<std::byte*>(call.workspace), required_workspace.load(),
                    std::byte{0x5a});
}
inline cudnnStatus_t query(cudnnHandle_t handle, CudaDnnReduction reduction,
                           cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t y,
                           std::size_t* result, bool indices) {
    (void)base::handle(handle);
    const auto& item = descriptor(reduction);
    base::require(item.configured && base::object(x, CudaDnnObjectKind::kTensor).configured &&
                  base::object(y, CudaDnnObjectKind::kTensor).configured && result);
    ++(indices ? state.indices_queries : state.workspace_queries);
    // The native error intentionally touches the private output, never the caller's.
    *result = SIZE_MAX - 3;
    if (fail_query.exchange(false)) return CUDNN_STATUS_NOT_SUPPORTED_SHAPE;
    *result = override_query.load() ? query_value.load()
              : indices             ? (needs_indices(item.value)
                                           ? elements(base::object(y, CudaDnnObjectKind::kTensor).tensor) *
                                     sizeof(std::uint32_t)
                                           : 0)
                                    : required_workspace.load();
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = foundation::dispatch();
    sdk.tensor_size = [](cudnnTensorDescriptor_t native, std::size_t* result) {
        const auto& tensor = base::object(native, CudaDnnObjectKind::kTensor);
        base::require(tensor.configured && result);
        *result = bytes(tensor.tensor);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.reduction_create = [](CudaDnnReduction* output) {
        base::owner();
        for (auto& item : state.descriptors)
            if (!item.alive) {
                item = {};
                item.alive = true;
                *output = reinterpret_cast<CudaDnnReduction>(&item);
                ++state.creates;
                return CUDNN_STATUS_SUCCESS;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    sdk.reduction_destroy = [](CudaDnnReduction native) {
        descriptor(native).alive = false;
        ++state.destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.reduction_set = [](CudaDnnReduction native, decltype(CUDNN_REDUCE_TENSOR_ADD) operation,
                           cudnnDataType_t type, cudnnNanPropagation_t nan,
                           decltype(CUDNN_REDUCE_TENSOR_NO_INDICES) indices,
                           decltype(CUDNN_32BIT_INDICES) indices_type) {
        auto& item = descriptor(native);
        ++state.sets;
        if (operation > CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS || indices_type != CUDNN_32BIT_INDICES)
            return CUDNN_STATUS_NOT_SUPPORTED;
        item.value = {static_cast<std::uint32_t>(operation), static_cast<std::uint32_t>(type),
                      static_cast<std::uint32_t>(nan), static_cast<std::uint32_t>(indices),
                      static_cast<std::uint32_t>(indices_type)};
        item.configured = true;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.reduction_indices_size = [](cudnnHandle_t handle, CudaDnnReduction reduction,
                                    cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t y,
                                    std::size_t* result) {
        return query(handle, reduction, x, y, result, true);
    };
    sdk.reduction_workspace_size = [](cudnnHandle_t handle, CudaDnnReduction reduction,
                                      cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t y,
                                      std::size_t* result) {
        return query(handle, reduction, x, y, result, false);
    };
    sdk.reduce_tensor = [](cudnnHandle_t handle, CudaDnnReduction reduction, void* indices,
                           std::size_t indices_bytes, void* workspace, std::size_t workspace_bytes,
                           const void* alpha, cudnnTensorDescriptor_t xd, const void* input,
                           const void* beta, cudnnTensorDescriptor_t yd, void* output) {
        const auto& owner = base::handle(handle);
        const auto& item = descriptor(reduction);
        const auto& x = base::object(xd, CudaDnnObjectKind::kTensor);
        const auto& y = base::object(yd, CudaDnnObjectKind::kTensor);
        base::require(item.configured && x.configured && y.configured && alpha && beta && input &&
                      output);
        Invocation call{handle,          reduction, item.value, xd,          yd,
                        input,           output,    indices,    workspace,   indices_bytes,
                        workspace_bytes, 0,         0,          owner.stream};
        if (x.tensor.type == CUDNN_DATA_DOUBLE) {
            base::require(reinterpret_cast<std::uintptr_t>(alpha) % 8 == 0 &&
                          reinterpret_cast<std::uintptr_t>(beta) % 8 == 0);
            std::memcpy(&call.alpha, alpha, 8);
            std::memcpy(&call.beta, beta, 8);
        } else {
            float a{}, b{};
            std::memcpy(&a, alpha, 4);
            std::memcpy(&b, beta, 4);
            call.alpha = a;
            call.beta = b;
        }
        state.last = call;
        observed_indices_bytes.store(indices_bytes);
        observed_workspace_bytes.store(workspace_bytes);
        observed_null_indices.store(!indices);
        observed_null_workspace.store(!workspace);
        ++state.computes;
        if (fail_compute.exchange(false)) return CUDNN_STATUS_EXECUTION_FAILED;
        const auto minimum = elements(y.tensor) * sizeof(std::uint32_t);
        if ((needs_indices(item.value) && (!indices || indices_bytes < minimum)) ||
            (required_workspace.load() &&
             (!workspace || workspace_bytes < required_workspace.load())))
            return CUDNN_STATUS_INVALID_VALUE;
        if (x.tensor.type == CUDNN_DATA_DOUBLE)
            compute<double>(call);
        else
            compute<float>(call);
        return CUDNN_STATUS_SUCCESS;
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}

// Fixed independent arithmetic and index expectations, not the grouping loop above.
struct Case {
    CudaDnnReductionState reduction{CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
                                    CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
                                    CUDNN_32BIT_INDICES};
    CudaDnnTensor x{CUDNN_DATA_FLOAT, 0, 4, {2, 3, 2, 2}, {40, 11, 4, 1}};
    CudaDnnTensor y{CUDNN_DATA_FLOAT, 0, 4, {2, 1, 1, 2}, {9, 8, 4, 2}};
    double alpha{1.5}, beta{-0.25};
};
inline constexpr std::array<double, 24> kInput{1, -2, 3,  0, 4, 2,  -1, 5, -3, 1,  2, -4,
                                               2, 3,  -2, 1, 0, -1, 4,  2, 5,  -3, 1, -2};
inline constexpr std::array<std::uint32_t, 4> kMinimumIndices{4, 5, 1, 4},
    kMaximumIndices{2, 3, 4, 0};
inline std::array<double, 4> expected(std::uint32_t operation) {
    switch (operation) {
        case CUDNN_REDUCE_TENSOR_ADD:
            return {6, 2, 10, 0};
        case CUDNN_REDUCE_TENSOR_MUL:
            return {72, 0, 0, -36};
        case CUDNN_REDUCE_TENSOR_MIN:
            return {-3, -4, -2, -3};
        case CUDNN_REDUCE_TENSOR_MAX:
            return {4, 5, 5, 3};
        case CUDNN_REDUCE_TENSOR_AMAX:
            return {4, 5, 5, 3};
        case CUDNN_REDUCE_TENSOR_AVG:
            return {1, 1.0 / 3, 5.0 / 3, 0};
        case CUDNN_REDUCE_TENSOR_NORM1:
            return {14, 14, 14, 12};
        case CUDNN_REDUCE_TENSOR_NORM2:
            return {std::sqrt(40.0), std::sqrt(50.0), std::sqrt(50.0), std::sqrt(28.0)};
        case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
            return {72, 80, -80, -36};
        default:
            throw std::runtime_error("missing fixed reduction expectation");
    }
}
inline Case make_case(std::uint32_t operation, bool is_double = false) {
    Case result;
    result.reduction.operation = operation;
    result.reduction.type = result.x.type = result.y.type =
        is_double ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    if (operation == CUDNN_REDUCE_TENSOR_MIN || operation == CUDNN_REDUCE_TENSOR_MAX)
        result.reduction.indices = CUDNN_REDUCE_TENSOR_FLATTENED_INDICES;
    return result;
}
inline std::vector<std::byte> image(const Case& value, bool output) {
    const auto& tensor = output ? value.y : value.x;
    std::vector<std::byte> result(bytes(tensor) + 32, std::byte{0xcd});
    for (std::size_t i = 0; i != elements(tensor); ++i)
        write(result.data() + 16, tensor.type, offset(tensor, i),
              output ? static_cast<double>(4 * (i + 1)) : kInput[i]);
    return result;
}
inline void verify(const Case& value, const std::vector<std::byte>& actual,
                   const std::vector<std::byte>& original, bool unchanged) {
    base::require(actual.size() == original.size());
    if (unchanged) {
        base::require(actual == original);
        return;
    }
    const auto result = expected(value.reduction.operation);
    auto holes = actual;
    for (std::size_t i = 0; i != result.size(); ++i) {
        const auto index = offset(value.y, i);
        const auto found = read(actual.data() + 16, value.y.type, index);
        const auto wanted = value.alpha * result[i] + value.beta * static_cast<double>(4 * (i + 1));
        const auto tolerance = value.y.type == CUDNN_DATA_DOUBLE ? 1e-12 : 3e-6;
        base::require(std::abs(found - wanted) <= tolerance * std::max(1.0, std::abs(wanted)));
        const auto byte = 16 + index * width(value.y.type);
        std::copy_n(original.begin() + static_cast<std::ptrdiff_t>(byte), width(value.y.type),
                    holes.begin() + static_cast<std::ptrdiff_t>(byte));
    }
    base::require(holes == original);
}
inline void verify_auxiliary(std::vector<std::byte> indices, std::vector<std::byte> workspace,
                             const Case& value, bool unchanged) {
    if (!unchanged) {
        if (needs_indices(value.reduction)) {
            const auto& expected_indices = value.reduction.operation == CUDNN_REDUCE_TENSOR_MIN
                                               ? kMinimumIndices
                                               : kMaximumIndices;
            base::require(std::memcmp(indices.data() + 16, expected_indices.data(), 16) == 0);
            std::fill_n(indices.begin() + 16, 16, std::byte{0xcd});
        }
        for (std::size_t i = 0; i != required_workspace.load(); ++i) {
            base::require(workspace.at(16 + i) == std::byte{0x5a});
            workspace[16 + i] = std::byte{0xcd};
        }
    }
    base::require(
        std::ranges::all_of(indices, [](auto byte) { return byte == std::byte{0xcd}; }) &&
        std::ranges::all_of(workspace, [](auto byte) { return byte == std::byte{0xcd}; }));
}
}  // namespace glimmer::tests::dnn_reduction
