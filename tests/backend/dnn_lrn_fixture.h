#pragma once

#include "dnn_convolution_fixture.h"

#include <cmath>

namespace glimmer::tests::dnn_lrn {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace foundation = glimmer::tests::dnn_convolution;
inline constexpr CudaDnnLrnState kNativeDefaults{5, 1e-4, 0.75, 2};
struct Descriptor {
    bool alive{};
    CudaDnnLrnState value{kNativeDefaults};
};
struct Invocation {
    cudnnHandle_t handle{};
    CudaDnnLrn lrn{};
    CudaDnnLrnState parameters{};
    cudnnTensorDescriptor_t x_descriptor{}, y_descriptor{};
    const void* x{};
    void* y{};
    std::uint32_t mode{};
    double alpha{}, beta{};
    cudaStream_t stream{};
};
struct State {
    std::array<Descriptor, 16> descriptors{};
    unsigned int creates{}, destroys{}, sets{}, lrns{}, transforms{};
    cudnnStatus_t compute_status{CUDNN_STATUS_SUCCESS};
    Invocation last;
};
inline State state;
inline std::atomic<bool> fail_compute{};
inline void reset() {
    foundation::reset();
    state = {};
    fail_compute.store(false);
}
inline Descriptor& descriptor(CudaDnnLrn pointer) {
    base::owner();
    for (auto& item : state.descriptors)
        if (reinterpret_cast<CudaDnnLrn>(&item) == pointer) {
            base::require(item.alive);
            return item;
        }
    throw std::runtime_error("unknown independent native LRN descriptor");
}
inline std::size_t offset(const CudaDnnTensor& tensor, std::size_t index) {
    std::size_t result{};
    for (int axis = tensor.dimensions; axis-- > 0;) {
        const auto i = static_cast<std::size_t>(axis);
        const auto size = static_cast<std::size_t>(tensor.sizes[i]);
        result += index % size * static_cast<std::size_t>(tensor.strides[i]);
        index /= size;
    }
    return result;
}
template <typename Scalar>
inline void compute(const Invocation& call) {
    const auto& x = base::object(call.x_descriptor, CudaDnnObjectKind::kTensor).tensor;
    const auto& y = base::object(call.y_descriptor, CudaDnnObjectKind::kTensor).tensor;
    base::require(x.dimensions == 4 && y.dimensions == 4 && x.type == y.type && x.sizes == y.sizes);
    const auto spatial = static_cast<std::size_t>(x.sizes[2] * x.sizes[3]);
    const auto channels = static_cast<std::size_t>(x.sizes[1]);
    const auto count = static_cast<std::size_t>(x.sizes[0]) * channels * spatial;
    const auto* input = static_cast<const Scalar*>(call.x);
    auto* output = static_cast<Scalar*>(call.y);
    std::vector<Scalar> source(count), old(count);
    for (std::size_t i = 0; i != count; ++i) {
        source[i] = input[offset(x, i)];
        old[i] = output[offset(y, i)];
    }
    for (std::size_t index = 0; index != count; ++index) {
        auto value = source[index];
        if (call.lrn) {
            const auto channel = index / spatial % channels;
            const auto batch = index / (channels * spatial);
            const auto pixel = index % spatial;
            const auto behind = (call.parameters.n - 1) / 2;
            const auto ahead = call.parameters.n - behind - 1;
            Scalar sum{};
            for (std::size_t c = 0; c != channels; ++c)
                if (c + behind >= channel && c <= channel + ahead) {
                    const auto neighbor = source[(batch * channels + c) * spatial + pixel];
                    sum += neighbor * neighbor;
                }
            const auto denominator = static_cast<Scalar>(call.parameters.k) +
                                     static_cast<Scalar>(call.parameters.alpha) /
                                         static_cast<Scalar>(call.parameters.n) * sum;
            value /= std::pow(denominator, static_cast<Scalar>(call.parameters.beta));
        }
        output[offset(y, index)] =
            static_cast<Scalar>(call.alpha) * value + static_cast<Scalar>(call.beta) * old[index];
    }
}
inline cudnnStatus_t forward(cudnnHandle_t handle, CudaDnnLrn lrn, std::uint32_t mode,
                             const void* alpha, cudnnTensorDescriptor_t xd, const void* x,
                             const void* beta, cudnnTensorDescriptor_t yd, void* y) {
    const auto& native_handle = base::handle(handle);
    const auto& input = base::object(xd, CudaDnnObjectKind::kTensor);
    const auto& output = base::object(yd, CudaDnnObjectKind::kTensor);
    base::require(input.configured && output.configured && alpha && beta && x && y);
    if (lrn)
        ++state.lrns;
    else
        ++state.transforms;
    Invocation call{handle,
                    lrn,
                    lrn ? descriptor(lrn).value : CudaDnnLrnState{},
                    xd,
                    yd,
                    x,
                    y,
                    mode,
                    0,
                    0,
                    native_handle.stream};
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
    if (!lrn) {
        const auto a = reinterpret_cast<std::uintptr_t>(x), b = reinterpret_cast<std::uintptr_t>(y);
        base::require(a >= b + foundation::tensor_bytes(output.tensor) ||
                      b >= a + foundation::tensor_bytes(input.tensor));
    }
    if (fail_compute.exchange(false)) return CUDNN_STATUS_EXECUTION_FAILED;
    if (state.compute_status != CUDNN_STATUS_SUCCESS) return state.compute_status;
    if (mode) return CUDNN_STATUS_BAD_PARAM;
    if (input.tensor.type == CUDNN_DATA_DOUBLE)
        compute<double>(call);
    else
        compute<float>(call);
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = foundation::dispatch();
    sdk.lrn_create = [](CudaDnnLrn* output) {
        base::owner();
        for (auto& item : state.descriptors)
            if (!item.alive) {
                item = {};
                item.alive = true;
                *output = reinterpret_cast<CudaDnnLrn>(&item);
                ++state.creates;
                return CUDNN_STATUS_SUCCESS;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    sdk.lrn_destroy = [](CudaDnnLrn pointer) {
        descriptor(pointer).alive = false;
        ++state.destroys;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.lrn_set = [](CudaDnnLrn pointer, unsigned int n, double alpha, double beta, double k) {
        auto& item = descriptor(pointer);
        ++state.sets;
        if (n < CUDNN_LRN_MIN_N || n > CUDNN_LRN_MAX_N || alpha < 0 || beta < CUDNN_LRN_MIN_BETA ||
            k < CUDNN_LRN_MIN_K)
            return CUDNN_STATUS_BAD_PARAM;
        item.value = {n, alpha, beta, k};
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.lrn_forward = [](cudnnHandle_t handle, CudaDnnLrn lrn, cudnnLRNMode_t mode,
                         const void* alpha, cudnnTensorDescriptor_t xd, const void* x,
                         const void* beta, cudnnTensorDescriptor_t yd, void* y) {
        return forward(handle, lrn, static_cast<std::uint32_t>(mode), alpha, xd, x, beta, yd, y);
    };
    sdk.transform_tensor = [](cudnnHandle_t handle, const void* alpha, cudnnTensorDescriptor_t xd,
                              const void* x, const void* beta, cudnnTensorDescriptor_t yd,
                              void* y) {
        return forward(handle, nullptr, 0, alpha, xd, x, beta, yd, y);
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}

// Full independent test oracles: explicit channel-window squared sums, not
// the loop used by the fake original implementation above.
struct Case {
    bool is_transform{}, use_defaults{}, shares_allocation{};
    std::uint32_t mode{};
    CudaDnnLrnState parameters{kNativeDefaults};
    CudaDnnTensor x{0, 0, 4, {2, 5, 1, 2}, {60, 9, 4, 2}};
    CudaDnnTensor y{0, 0, 4, {2, 5, 1, 2}, {78, 13, 5, 3}};
    std::array<double, 20> input{1,  2,  2,  3,  3,  4,  4,  5,  5,  6,
                                 -2, -1, -3, -2, -4, -3, -5, -4, -6, -5};
};
inline Case lrn_case(unsigned int n = 5) {
    Case value;
    value.use_defaults = n == 5;
    if (n == 3) value.parameters = {3, 0.37500000000000006, 0.5000000000000001, 1.2500000000000002};
    if (n == 4) value.parameters = {4, 0.5, 1.25, 0.75};
    return value;
}
inline Case transform_case(bool shares_allocation) {
    Case value;
    value.is_transform = true;
    value.shares_allocation = shares_allocation;
    return value;
}
template <typename Scalar>
inline Scalar normalized(const Case& value, std::size_t index) {
    constexpr std::array<double, 20> n3{5,  13, 14, 29, 29, 50, 50, 77, 41, 61,
                                        13, 5,  29, 14, 50, 29, 77, 50, 61, 41};
    constexpr std::array<double, 20> n4{14, 29, 30, 54, 54, 86, 50, 77, 41, 61,
                                        29, 14, 54, 30, 86, 54, 77, 50, 61, 41};
    constexpr std::array<double, 20> n5{14, 29, 30, 54, 55, 90, 54, 86, 50, 77,
                                        29, 14, 54, 30, 90, 55, 86, 54, 77, 50};
    const auto input = static_cast<Scalar>(value.input[index]);
    if (value.is_transform) return input;
    const auto& sums = value.parameters.n == 3 ? n3 : value.parameters.n == 4 ? n4 : n5;
    const auto denominator =
        static_cast<Scalar>(value.parameters.k) + static_cast<Scalar>(value.parameters.alpha) /
                                                      static_cast<Scalar>(value.parameters.n) *
                                                      static_cast<Scalar>(sums[index]);
    return input / std::pow(denominator, static_cast<Scalar>(value.parameters.beta));
}
template <typename Scalar>
inline std::vector<std::byte> image(const Case& value, bool output) {
    const auto& tensor = output ? value.y : value.x;
    std::vector<std::byte> bytes(foundation::tensor_bytes(tensor) + 32, std::byte{0xcd});
    for (std::size_t i = 0; i != value.input.size(); ++i) {
        const auto item = static_cast<Scalar>(output ? 2 + i * 0.125 : value.input[i]);
        std::memcpy(bytes.data() + 16 + offset(tensor, i) * sizeof(Scalar), &item, sizeof(item));
    }
    return bytes;
}
template <typename Scalar>
inline void verify(const Case& value, const std::vector<std::byte>& actual,
                   const std::vector<std::byte>& original, bool failed) {
    if (failed) {
        base::require(actual == original);
        return;
    }
    auto guards = actual;
    for (std::size_t i = 0; i != value.input.size(); ++i) {
        const auto start = 16 + offset(value.y, i) * sizeof(Scalar);
        Scalar observed{};
        std::memcpy(&observed, actual.data() + start, sizeof(observed));
        const auto expected = Scalar{1.5} * normalized<Scalar>(value, i) -
                              Scalar{0.25} * static_cast<Scalar>(2 + i * 0.125);
        base::require(std::abs(observed - expected) <= (sizeof(Scalar) == 4 ? 3e-6 : 1e-12));
        std::copy_n(original.data() + start, sizeof(Scalar), guards.data() + start);
    }
    base::require(guards == original);
}
}  // namespace glimmer::tests::dnn_lrn
