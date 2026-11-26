#pragma once

#include "dnn_convolution_fixture.h"

#include <cmath>

namespace glimmer::tests::dnn_pooling {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace foundation = glimmer::tests::dnn_convolution;
struct Pooling {
    bool alive{}, configured{};
    CudaDnnPoolingState value;
};
struct Invocation {
    cudnnHandle_t handle{};
    CudaDnnPooling pooling{};
    cudnnTensorDescriptor_t x_descriptor{}, y_descriptor{};
    const void* x{};
    void* y{};
    std::uint32_t algorithm{}, mode{};
    double alpha{}, beta{};
    cudaStream_t stream{};
};
struct State {
    std::array<Pooling, 16> pooling{};
    unsigned int creates{}, destroys{}, sets{}, gets{}, shapes{}, pools{}, softmaxes{};
    cudnnStatus_t compute_status{CUDNN_STATUS_SUCCESS};
    Invocation last;
};
struct Control {
    std::atomic<bool> fail_compute{}, fail_get{};
};
inline State state;
inline Control control;
inline void reset() {
    foundation::reset();
    state = {};
    control.fail_compute.store(false);
    control.fail_get.store(false);
}
inline Pooling& object(CudaDnnPooling pointer) {
    base::owner();
    for (auto& item : state.pooling)
        if (reinterpret_cast<CudaDnnPooling>(&item) == pointer) {
            base::require(item.alive);
            return item;
        }
    throw std::runtime_error("unknown independent pooling descriptor");
}
inline std::size_t elements(const CudaDnnTensor& tensor) {
    std::size_t count{1};
    for (int i = 0; i != tensor.dimensions; ++i)
        count *= static_cast<std::size_t>(tensor.sizes[static_cast<std::size_t>(i)]);
    return count;
}
inline std::size_t offset(const CudaDnnTensor& tensor, std::size_t index) {
    std::size_t result{};
    for (int axis = tensor.dimensions; axis-- > 0;) {
        const auto dim = static_cast<std::size_t>(axis);
        const auto size = static_cast<std::size_t>(tensor.sizes[dim]);
        result += index % size * static_cast<std::size_t>(tensor.strides[dim]);
        index /= size;
    }
    return result;
}
template <typename Scalar>
inline std::vector<Scalar> snapshot(const CudaDnnTensor& tensor, const void* pointer) {
    const auto* values = static_cast<const Scalar*>(pointer);
    std::vector<Scalar> output(elements(tensor));
    for (std::size_t i = 0; i != output.size(); ++i) output[i] = values[offset(tensor, i)];
    return output;
}
template <typename Scalar>
inline void compute(const Invocation& call, bool is_pooling) {
    const auto& x = base::object(call.x_descriptor, CudaDnnObjectKind::kTensor).tensor;
    const auto& y = base::object(call.y_descriptor, CudaDnnObjectKind::kTensor).tensor;
    base::require(x.dimensions == 4 && y.dimensions == 4 && x.type == y.type);
    const auto source = snapshot<Scalar>(x, call.x);
    const auto old = snapshot<Scalar>(y, call.y);
    std::vector<Scalar> result(old.size());
    if (is_pooling) {
        const auto& pooling = object(call.pooling).value;
        base::require(pooling.dimensions == 2);
        for (int n = 0; n != y.sizes[0]; ++n)
            for (int c = 0; c != y.sizes[1]; ++c)
                for (int h = 0; h != y.sizes[2]; ++h)
                    for (int w = 0; w != y.sizes[3]; ++w) {
                        const bool is_max = pooling.mode == CUDNN_POOLING_MAX ||
                                            pooling.mode == CUDNN_POOLING_MAX_DETERMINISTIC;
                        Scalar value = is_max ? -std::numeric_limits<Scalar>::infinity() : 0;
                        unsigned int count{};
                        bool has_nan{};
                        for (int kh = 0; kh != pooling.window[0]; ++kh)
                            for (int kw = 0; kw != pooling.window[1]; ++kw) {
                                const auto ih = h * pooling.strides[0] - pooling.padding[0] + kh;
                                const auto iw = w * pooling.strides[1] - pooling.padding[1] + kw;
                                if (ih < 0 || iw < 0 || ih >= x.sizes[2] || iw >= x.sizes[3])
                                    continue;
                                const auto i =
                                    ((n * x.sizes[1] + c) * x.sizes[2] + ih) * x.sizes[3] + iw;
                                const auto sample = source[static_cast<std::size_t>(i)];
                                has_nan |= std::isnan(sample);
                                value = is_max ? std::fmax(value, sample) : value + sample;
                                ++count;
                            }
                        if (!is_max) {
                            const auto divisor =
                                pooling.mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                    ? static_cast<unsigned int>(pooling.window[0] *
                                                                pooling.window[1])
                                    : count;
                            value /= static_cast<Scalar>(divisor);
                        }
                        if (has_nan && pooling.nan == CUDNN_PROPAGATE_NAN)
                            value = std::numeric_limits<Scalar>::quiet_NaN();
                        const auto i = ((n * y.sizes[1] + c) * y.sizes[2] + h) * y.sizes[3] + w;
                        result[static_cast<std::size_t>(i)] = value;
                    }
    } else {
        base::require(x.sizes == y.sizes);
        const auto spatial = static_cast<std::size_t>(x.sizes[2] * x.sizes[3]);
        const auto batch = static_cast<std::size_t>(x.sizes[1]) * spatial;
        for (int n = 0; n != x.sizes[0]; ++n) {
            const auto groups = call.mode == CUDNN_SOFTMAX_MODE_CHANNEL ? spatial : 1;
            const auto width = call.mode == CUDNN_SOFTMAX_MODE_CHANNEL
                                   ? static_cast<std::size_t>(x.sizes[1])
                                   : batch;
            for (std::size_t group = 0; group != groups; ++group) {
                const auto index = [&](std::size_t element) {
                    return static_cast<std::size_t>(n) * batch +
                           (groups == 1 ? element : element * spatial + group);
                };
                Scalar maximum = source[index(0)];
                for (std::size_t i = 1; i != width; ++i)
                    maximum = std::max(maximum, source[index(i)]);
                const auto shift = call.algorithm == CUDNN_SOFTMAX_FAST ? Scalar{0} : maximum;
                Scalar denominator{};
                for (std::size_t i = 0; i != width; ++i)
                    denominator += std::exp(source[index(i)] - shift);
                for (std::size_t i = 0; i != width; ++i)
                    result[index(i)] = call.algorithm == CUDNN_SOFTMAX_LOG
                                           ? source[index(i)] - shift - std::log(denominator)
                                           : std::exp(source[index(i)] - shift) / denominator;
            }
        }
    }
    auto* destination = static_cast<Scalar*>(call.y);
    for (std::size_t i = 0; i != result.size(); ++i)
        destination[offset(y, i)] =
            static_cast<Scalar>(call.alpha) * result[i] + static_cast<Scalar>(call.beta) * old[i];
}
inline cudnnStatus_t forward(cudnnHandle_t handle, CudaDnnPooling pooling, std::uint32_t algorithm,
                             std::uint32_t mode, const void* alpha,
                             cudnnTensorDescriptor_t x_descriptor, const void* x, const void* beta,
                             cudnnTensorDescriptor_t y_descriptor, void* y) {
    const auto& native_handle = base::handle(handle);
    const auto& input = base::object(x_descriptor, CudaDnnObjectKind::kTensor);
    const auto& output = base::object(y_descriptor, CudaDnnObjectKind::kTensor);
    base::require(input.configured && output.configured && x && y && alpha && beta);
    if (pooling)
        ++state.pools;
    else
        ++state.softmaxes;
    Invocation call{handle, pooling, x_descriptor,        y_descriptor, x, y, algorithm, mode,
                    0,      0,       native_handle.stream};
    if (input.tensor.type == CUDNN_DATA_DOUBLE) {
        base::require(reinterpret_cast<std::uintptr_t>(alpha) % alignof(double) == 0 &&
                      reinterpret_cast<std::uintptr_t>(beta) % alignof(double) == 0);
        std::memcpy(&call.alpha, alpha, sizeof(double));
        std::memcpy(&call.beta, beta, sizeof(double));
    } else {
        float a{}, b{};
        std::memcpy(&a, alpha, sizeof(a));
        std::memcpy(&b, beta, sizeof(b));
        call.alpha = a;
        call.beta = b;
    }
    state.last = call;
    if (control.fail_compute.exchange(false)) return CUDNN_STATUS_EXECUTION_FAILED;
    if (state.compute_status != CUDNN_STATUS_SUCCESS) return state.compute_status;
    if (!pooling && (algorithm > CUDNN_SOFTMAX_LOG || mode > CUDNN_SOFTMAX_MODE_CHANNEL))
        return CUDNN_STATUS_BAD_PARAM;
    if (input.tensor.type == CUDNN_DATA_DOUBLE)
        compute<double>(call, pooling != nullptr);
    else
        compute<float>(call, pooling != nullptr);
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = foundation::dispatch();
    sdk.pooling_create = [](CudaDnnPooling* output) {
        base::owner();
        for (auto& item : state.pooling)
            if (!item.alive) {
                item = {};
                item.alive = true;
                *output = reinterpret_cast<CudaDnnPooling>(&item);
                ++state.creates;
                return CUDNN_STATUS_SUCCESS;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    sdk.pooling_destroy = [](CudaDnnPooling pointer) {
        object(pointer).alive = false;
        ++state.destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.pooling_set_nd = [](CudaDnnPooling pointer, decltype(CUDNN_POOLING_MAX) mode,
                            cudnnNanPropagation_t nan, int rank, const int* window,
                            const int* padding, const int* strides) {
        auto& item = object(pointer);
        ++state.sets;
        if (rank < 0) return CUDNN_STATUS_BAD_PARAM;
        if (rank < 1 || rank > 6) return CUDNN_STATUS_NOT_SUPPORTED;
        item.value = {static_cast<std::uint32_t>(mode), static_cast<std::uint32_t>(nan), rank};
        std::copy_n(window, rank, item.value.window.begin());
        std::copy_n(padding, rank, item.value.padding.begin());
        std::copy_n(strides, rank, item.value.strides.begin());
        item.configured = true;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.pooling_get_nd = [](CudaDnnPooling pointer, int requested,
                            decltype(CUDNN_POOLING_MAX)* mode, cudnnNanPropagation_t* nan,
                            int* rank, int* window, int* padding, int* strides) {
        const auto& item = object(pointer);
        ++state.gets;
        *mode = static_cast<decltype(CUDNN_POOLING_MAX)>(item.value.mode);
        *nan = static_cast<cudnnNanPropagation_t>(item.value.nan);
        *rank = item.value.dimensions;
        if (requested > 8) return CUDNN_STATUS_NOT_SUPPORTED;
        if (requested < 0 || !item.configured || control.fail_get.exchange(false))
            return CUDNN_STATUS_BAD_PARAM;
        const auto count = std::min(requested, *rank);
        std::copy_n(item.value.window.begin(), count, window);
        std::copy_n(item.value.padding.begin(), count, padding);
        std::copy_n(item.value.strides.begin(), count, strides);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.pooling_output_nd = [](CudaDnnPooling pointer, cudnnTensorDescriptor_t tensor, int rank,
                               int* sizes) {
        const auto& item = object(pointer);
        const auto& input = base::object(tensor, CudaDnnObjectKind::kTensor).tensor;
        ++state.shapes;
        if (rank != input.dimensions || rank != item.value.dimensions + 2)
            return CUDNN_STATUS_BAD_PARAM;
        sizes[0] = input.sizes[0];
        sizes[1] = input.sizes[1];
        for (int i = 2; i != rank; ++i) {
            const auto axis = static_cast<std::size_t>(i - 2);
            sizes[i] = 1 + (input.sizes[static_cast<std::size_t>(i)] +
                            2 * item.value.padding[axis] - item.value.window[axis]) /
                               item.value.strides[axis];
        }
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.pooling_forward = [](cudnnHandle_t handle, CudaDnnPooling pooling, const void* alpha,
                             cudnnTensorDescriptor_t xd, const void* x, const void* beta,
                             cudnnTensorDescriptor_t yd, void* y) {
        return forward(handle, pooling, 0, 0, alpha, xd, x, beta, yd, y);
    };
    sdk.softmax_forward = [](cudnnHandle_t handle, cudnnSoftmaxAlgorithm_t algorithm,
                             cudnnSoftmaxMode_t mode, const void* alpha, cudnnTensorDescriptor_t xd,
                             const void* x, const void* beta, cudnnTensorDescriptor_t yd, void* y) {
        return forward(handle, nullptr, static_cast<std::uint32_t>(algorithm),
                       static_cast<std::uint32_t>(mode), alpha, xd, x, beta, yd, y);
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}

// Test inputs and fixed expected values are independent of the generic
// arithmetic callbacks above. This fixture executes CPU arithmetic, never GPU.
struct Case {
    bool is_pooling{}, is_inplace{};
    std::uint32_t algorithm{}, mode{};
    CudaDnnPoolingState pooling;
    CudaDnnTensor x, y;
    std::vector<double> input, initial, expected;
};
inline Case pool_case(std::uint32_t mode, bool short_output = false, bool negative_padding = false,
                      bool has_nan = false) {
    Case value;
    value.is_pooling = true;
    value.pooling = {mode, CUDNN_PROPAGATE_NAN, 2, {2, 3}, {1, 1}, {1, 2}};
    value.x = {0, 0, 4, {1, 1, 3, 4}, {40, 40, 9, 2}};
    value.y = {0, 0, 4, {1, 1, short_output ? 3 : 4, 2}, {32, 32, 5, 2}};
    value.input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    if (mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING)
        value.expected = {0.5, 1.5, 7.0 / 3, 5, 5, 9, 19.0 / 6, 5.5};
    else if (mode == CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING)
        value.expected = {1.5, 3, 3.5, 5, 7.5, 9, 9.5, 11};
    else
        value.expected = {2, 4, 6, 8, 10, 12, 10, 12};
    if (short_output) value.expected.resize(6);
    if (negative_padding) {
        value.pooling = {CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, {2, 2}, {-1, -1}, {1, 1}};
        value.x = {0, 0, 4, {1, 1, 4, 5}, {60, 60, 12, 2}};
        value.y = {0, 0, 4, {1, 1, 1, 2}, {8, 8, 6, 2}};
        value.input.resize(20);
        for (std::size_t i = 0; i != value.input.size(); ++i) value.input[i] = i + 1;
        value.expected = {13, 14};
    }
    if (has_nan) {
        value.input[0] = std::numeric_limits<double>::quiet_NaN();
        value.expected[0] = value.expected[2] = std::numeric_limits<double>::quiet_NaN();
    }
    value.initial.resize(value.expected.size());
    for (std::size_t i = 0; i != value.initial.size(); ++i) value.initial[i] = 2 + i * 0.25;
    return value;
}
inline Case softmax_case(std::uint32_t algorithm, std::uint32_t mode, bool is_inplace = false) {
    Case value;
    value.algorithm = algorithm;
    value.mode = mode;
    value.is_inplace = is_inplace;
    value.x = {0, 0, 4, {2, 2, 2, 2}, {32, 12, 5, 2}};
    value.y = {0, 0, 4, {2, 2, 2, 2}, {44, 18, 7, 2}};
    constexpr std::array<double, 16> weights{1, 2, 4, 8, 2, 1, 8, 4, 4, 8, 1, 2, 8, 4, 2, 1};
    constexpr std::array<double, 16> channels{1.0 / 3, 2.0 / 3, 1.0 / 3, 2.0 / 3, 2.0 / 3, 1.0 / 3,
                                              2.0 / 3, 1.0 / 3, 1.0 / 3, 2.0 / 3, 1.0 / 3, 2.0 / 3,
                                              2.0 / 3, 1.0 / 3, 2.0 / 3, 1.0 / 3};
    for (std::size_t i = 0; i != weights.size(); ++i) {
        value.input.push_back(std::log(weights[i]));
        value.initial.push_back(is_inplace ? value.input.back() : 2 + i * 0.25);
        const auto probability = mode == CUDNN_SOFTMAX_MODE_CHANNEL ? channels[i] : weights[i] / 30;
        value.expected.push_back(algorithm == CUDNN_SOFTMAX_LOG ? std::log(probability)
                                                                : probability);
    }
    if (is_inplace) value.y = value.x;
    return value;
}
template <typename Scalar>
inline std::vector<std::byte> image(const CudaDnnTensor& tensor,
                                    const std::vector<double>& values) {
    std::vector<std::byte> bytes(foundation::tensor_bytes(tensor) + 32, std::byte{0xcd});
    for (std::size_t i = 0; i != values.size(); ++i) {
        const auto value = static_cast<Scalar>(values[i]);
        std::memcpy(bytes.data() + 16 + offset(tensor, i) * sizeof(Scalar), &value, sizeof(value));
    }
    return bytes;
}
template <typename Scalar>
inline void verify(const Case& value, const std::vector<std::byte>& actual,
                   const std::vector<std::byte>& original, bool failed = false) {
    if (failed) {
        base::require(actual == original);
        return;
    }
    auto guarded = actual;
    for (std::size_t i = 0; i != value.expected.size(); ++i) {
        const auto start = 16 + offset(value.y, i) * sizeof(Scalar);
        Scalar observed{};
        std::memcpy(&observed, actual.data() + start, sizeof(observed));
        const auto expected =
            1.25 * value.expected[i] - 0.5 * static_cast<Scalar>(value.initial[i]);
        base::require((std::isnan(expected) && std::isnan(observed)) ||
                      std::abs(static_cast<double>(observed) - expected) <=
                          (sizeof(Scalar) == 4 ? 2e-5 : 1e-12));
        std::copy_n(original.data() + start, sizeof(Scalar), guarded.data() + start);
    }
    base::require(guarded == original);  // Both guards and every stride hole.
}
}  // namespace glimmer::tests::dnn_pooling
