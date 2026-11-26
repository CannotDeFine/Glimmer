#pragma once

#include "dnn_convolution_fixture.h"

#include <cmath>

namespace glimmer::tests::dnn_batch_norm {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace foundation = glimmer::tests::dnn_convolution;
struct Invocation {
    CudaDnnBatchNormKind kind{};
    std::uint32_t mode{}, operation{};
    cudnnHandle_t handle{};
    std::array<void*, 7> descriptors{};
    std::array<void*, 13> buffers{};
    std::size_t workspace_bytes{}, reserve_bytes{};
    double epsilon{}, average_factor{};
    std::array<double, 4> coefficients{};
    cudaStream_t stream{};
};
struct State {
    unsigned int forwards{}, backwards{};
    std::array<unsigned int, 3> queries{};
    Invocation last;
};
inline State state;
inline std::atomic<bool> fail_compute{}, fail_query{}, override_query{};
inline std::atomic<std::size_t> query_value{}, required_workspace{32}, required_reserve{32};
inline std::atomic<std::size_t> observed_workspace{}, observed_reserve{};
inline std::atomic<std::uint32_t> observed_nulls{};
inline void reset() {
    foundation::reset();
    state = {};
    fail_compute.store(false);
    fail_query.store(false);
    override_query.store(false);
    query_value.store(0);
    required_workspace.store(32);
    required_reserve.store(32);
    observed_workspace.store(0);
    observed_reserve.store(0);
    observed_nulls.store(0);
}
inline const CudaDnnTensor& tensor(void* native) {
    const auto& object = base::object(native, CudaDnnObjectKind::kTensor);
    base::require(object.configured);
    return object.tensor;
}
inline std::size_t count(const CudaDnnTensor& value) {
    std::size_t result{1};
    for (int i = 0; i != value.dimensions; ++i)
        result *= static_cast<std::size_t>(value.sizes[static_cast<std::size_t>(i)]);
    return result;
}
inline std::size_t offset(const CudaDnnTensor& value, std::size_t logical) {
    std::size_t result{};
    for (int axis = value.dimensions; axis-- > 0;) {
        const auto i = static_cast<std::size_t>(axis);
        const auto size = static_cast<std::size_t>(value.sizes[i]);
        result += logical % size * static_cast<std::size_t>(value.strides[i]);
        logical /= size;
    }
    return result;
}
inline std::size_t group(std::uint32_t mode, std::size_t logical) {
    return mode == CUDNN_BATCHNORM_PER_ACTIVATION ? logical % 4 : logical / 2 % 2;
}
inline std::size_t groups(std::uint32_t mode) {
    return mode == CUDNN_BATCHNORM_PER_ACTIVATION ? 4 : 2;
}
template <typename Scalar>
inline void native_compute(const Invocation& call) {
    const auto& x = tensor(call.descriptors[0]);
    const auto& params = tensor(call.descriptors[5]);
    base::require(x.sizes[0] == 2 && x.sizes[1] == 2 && x.sizes[2] == 1 && x.sizes[3] == 2);
    const auto number = groups(call.mode), samples = 8 / number;
    auto read = [&](std::size_t slot, std::size_t descriptor, std::size_t logical) {
        return static_cast<const Scalar*>(
            call.buffers[slot])[offset(tensor(call.descriptors[descriptor]), logical)];
    };
    auto parameter = [&](std::size_t slot, std::size_t logical) -> Scalar& {
        return static_cast<Scalar*>(call.buffers[slot])[offset(params, logical)];
    };
    std::array<Scalar, 4> means{}, variance{}, inverse{};
    for (std::size_t i = 0; i != 8; ++i) means[group(call.mode, i)] += read(0, 0, i);
    for (std::size_t g = 0; g != number; ++g) means[g] /= static_cast<Scalar>(samples);
    for (std::size_t i = 0; i != 8; ++i) {
        const auto g = group(call.mode, i);
        const auto delta = read(0, 0, i) - means[g];
        variance[g] += delta * delta / static_cast<Scalar>(samples);
    }
    for (std::size_t g = 0; g != number; ++g)
        inverse[g] = Scalar{1} / std::sqrt(variance[g] + static_cast<Scalar>(call.epsilon));
    const bool forward = call.kind == CudaDnnBatchNormKind::kForward;
    if (!forward && call.buffers[9])
        for (std::size_t g = 0; g != number; ++g) {
            means[g] = parameter(9, g);
            inverse[g] = parameter(10, g);
        }
    auto* reserve = static_cast<std::byte*>(call.buffers[12]);
    if (forward) {
        if (required_reserve.load()) std::fill_n(reserve, required_reserve.load(), std::byte{0xa7});
        for (std::size_t i = 0; i != 8; ++i) {
            const auto g = group(call.mode, i);
            auto normalized =
                parameter(5, g) * (read(0, 0, i) - means[g]) * inverse[g] + parameter(6, g);
            if (call.operation == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION)
                normalized += read(3, 3, i);
            const bool active = !call.operation || normalized > 0;
            if (required_reserve.load()) reserve[i] = active ? std::byte{1} : std::byte{0};
            if (call.operation) normalized = std::max(Scalar{}, normalized);
            auto& y = static_cast<Scalar*>(call.buffers[1])[offset(tensor(call.descriptors[1]), i)];
            y = static_cast<Scalar>(call.coefficients[0]) * normalized +
                static_cast<Scalar>(call.coefficients[1]) * y;
        }
        for (std::size_t g = 0; g != number; ++g) {
            if (call.buffers[7]) {
                const auto factor = static_cast<Scalar>(call.average_factor);
                parameter(7, g) = factor * means[g] + (Scalar{1} - factor) * parameter(7, g);
                parameter(8, g) = factor * variance[g] * static_cast<Scalar>(samples) /
                                      static_cast<Scalar>(samples - 1) +
                                  (Scalar{1} - factor) * parameter(8, g);
            }
            if (call.buffers[9]) {
                parameter(9, g) = means[g];
                parameter(10, g) = inverse[g];
            }
        }
    } else {
        if (required_reserve.load())
            for (std::size_t i = 8; i != required_reserve.load(); ++i)
                base::require(reserve[i] == std::byte{0xa7});
        std::array<Scalar, 8> gradient{}, normalized{};
        std::array<Scalar, 4> sum{}, scale_gradient{};
        for (std::size_t i = 0; i != 8; ++i) {
            const auto g = group(call.mode, i);
            normalized[i] = (read(0, 0, i) - means[g]) * inverse[g];
            gradient[i] = read(2, 2, i);
            if (call.operation) gradient[i] *= reserve[i] == std::byte{1} ? Scalar{1} : Scalar{};
            sum[g] += gradient[i];
            scale_gradient[g] += gradient[i] * normalized[i];
            if (call.operation == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION)
                static_cast<Scalar*>(call.buffers[3])[offset(tensor(call.descriptors[3]), i)] =
                    gradient[i];
        }
        for (std::size_t i = 0; i != 8; ++i) {
            const auto g = group(call.mode, i);
            const auto result = parameter(5, g) * inverse[g] / static_cast<Scalar>(samples) *
                                (static_cast<Scalar>(samples) * gradient[i] - sum[g] -
                                 normalized[i] * scale_gradient[g]);
            auto& dx =
                static_cast<Scalar*>(call.buffers[4])[offset(tensor(call.descriptors[4]), i)];
            dx = static_cast<Scalar>(call.coefficients[0]) * result +
                 static_cast<Scalar>(call.coefficients[1]) * dx;
        }
        for (std::size_t g = 0; g != number; ++g) {
            parameter(7, g) = static_cast<Scalar>(call.coefficients[2]) * scale_gradient[g] +
                              static_cast<Scalar>(call.coefficients[3]) * parameter(7, g);
            parameter(8, g) = static_cast<Scalar>(call.coefficients[2]) * sum[g] +
                              static_cast<Scalar>(call.coefficients[3]) * parameter(8, g);
        }
    }
    if (required_workspace.load())
        std::fill_n(static_cast<std::byte*>(call.buffers[11]), required_workspace.load(),
                    forward ? std::byte{0x4f} : std::byte{0x42});
}
inline cudnnStatus_t issue(Invocation call, const std::array<const void*, 4>& coefficients) {
    call.stream = base::handle(call.handle).stream;
    for (std::size_t i = 0; i != (call.kind == CudaDnnBatchNormKind::kForward ? 2U : 4U); ++i) {
        base::require(coefficients[i]);
        if (tensor(call.descriptors[0]).type == CUDNN_DATA_DOUBLE) {
            base::require(reinterpret_cast<std::uintptr_t>(coefficients[i]) % 8 == 0);
            std::memcpy(&call.coefficients[i], coefficients[i], 8);
        } else {
            float value{};
            std::memcpy(&value, coefficients[i], 4);
            call.coefficients[i] = value;
        }
    }
    state.last = call;
    ++(call.kind == CudaDnnBatchNormKind::kForward ? state.forwards : state.backwards);
    observed_workspace.store(call.workspace_bytes);
    observed_reserve.store(call.reserve_bytes);
    std::uint32_t nulls{};
    for (std::size_t i = 0; i != call.buffers.size(); ++i)
        if (!call.buffers[i]) nulls |= 1U << i;
    observed_nulls.store(nulls);
    if (fail_compute.exchange(false)) return CUDNN_STATUS_EXECUTION_FAILED;
    if (call.mode == 3 || call.operation == 3) return CUDNN_STATUS_NOT_SUPPORTED;
    const bool forward = call.kind == CudaDnnBatchNormKind::kForward;
    if ((forward && static_cast<bool>(call.buffers[7]) != static_cast<bool>(call.buffers[8])) ||
        static_cast<bool>(call.buffers[9]) != static_cast<bool>(call.buffers[10]) ||
        (call.operation && (!call.descriptors[6] || !call.buffers[12])) ||
        (call.operation == 2 && (!call.descriptors[3] || !call.buffers[3])))
        return CUDNN_STATUS_BAD_PARAM;
    if ((required_workspace.load() &&
         (!call.buffers[11] || call.workspace_bytes < required_workspace.load())) ||
        (required_reserve.load() &&
         (!call.buffers[12] || call.reserve_bytes < required_reserve.load())))
        return CUDNN_STATUS_INVALID_VALUE;
    if (call.descriptors[6])
        base::require(base::object(call.descriptors[6], CudaDnnObjectKind::kActivation).configured);
    if (tensor(call.descriptors[0]).type == CUDNN_DATA_DOUBLE)
        native_compute<double>(call);
    else
        native_compute<float>(call);
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t query(cudnnHandle_t handle, std::size_t kind, std::uint32_t mode,
                           std::uint32_t operation, const std::array<void*, 7>& descriptors,
                           std::size_t* bytes) {
    (void)base::handle(handle);
    base::require(bytes && descriptors[0]);
    for (std::size_t i = 0; i != descriptors.size(); ++i)
        if (descriptors[i])
            (void)base::object(descriptors[i], i == 6 ? CudaDnnObjectKind::kActivation
                                                      : CudaDnnObjectKind::kTensor);
    ++state.queries.at(kind - 1);
    *bytes = SIZE_MAX - 3;
    if (fail_query.exchange(false)) return CUDNN_STATUS_NOT_SUPPORTED_SHAPE;
    if (mode == 3 || operation == 3) return CUDNN_STATUS_NOT_SUPPORTED;
    *bytes = override_query.load() ? query_value.load()
             : kind == 3           ? required_reserve.load()
                                   : required_workspace.load();
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = foundation::dispatch();
    sdk.batch_norm_forward_workspace =
        [](cudnnHandle_t h, cudnnBatchNormMode_t m, cudnnBatchNormOps_t o,
           cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t z, cudnnTensorDescriptor_t y,
           cudnnTensorDescriptor_t p, CudaDnnActivation a, std::size_t* bytes) {
            return query(h, 1, static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(o),
                         {x, y, nullptr, z, nullptr, p, a}, bytes);
        };
    sdk.batch_norm_backward_workspace =
        [](cudnnHandle_t h, cudnnBatchNormMode_t m, cudnnBatchNormOps_t o,
           cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t y, cudnnTensorDescriptor_t dy,
           cudnnTensorDescriptor_t dz, cudnnTensorDescriptor_t dx, cudnnTensorDescriptor_t p,
           CudaDnnActivation a, std::size_t* bytes) {
            return query(h, 2, static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(o),
                         {x, y, dy, dz, dx, p, a}, bytes);
        };
    sdk.batch_norm_reserve_size = [](cudnnHandle_t h, cudnnBatchNormMode_t m, cudnnBatchNormOps_t o,
                                     CudaDnnActivation a, cudnnTensorDescriptor_t x,
                                     std::size_t* bytes) {
        return query(h, 3, static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(o),
                     {x, nullptr, nullptr, nullptr, nullptr, nullptr, a}, bytes);
    };
    sdk.batch_norm_forward =
        [](cudnnHandle_t h, cudnnBatchNormMode_t m, cudnnBatchNormOps_t o, const void* alpha,
           const void* beta, cudnnTensorDescriptor_t xd, const void* x, cudnnTensorDescriptor_t zd,
           const void* z, cudnnTensorDescriptor_t yd, void* y, cudnnTensorDescriptor_t p,
           const void* scale, const void* bias, double factor, void* mean, void* variance,
           double epsilon, void* saved_mean, void* saved_inverse, CudaDnnActivation a, void* ws,
           std::size_t wb, void* reserve, std::size_t rb) {
            Invocation call{CudaDnnBatchNormKind::kForward,
                            static_cast<std::uint32_t>(m),
                            static_cast<std::uint32_t>(o),
                            h,
                            {xd, yd, nullptr, zd, nullptr, p, a},
                            {const_cast<void*>(x), y, nullptr, const_cast<void*>(z), nullptr,
                             const_cast<void*>(scale), const_cast<void*>(bias), mean, variance,
                             saved_mean, saved_inverse, ws, reserve},
                            wb,
                            rb,
                            epsilon,
                            factor};
            return issue(call, {alpha, beta, nullptr, nullptr});
        };
    sdk.batch_norm_backward =
        [](cudnnHandle_t h, cudnnBatchNormMode_t m, cudnnBatchNormOps_t o, const void* ad,
           const void* bd, const void* ap, const void* bp, cudnnTensorDescriptor_t xd,
           const void* x, cudnnTensorDescriptor_t yd, const void* y, cudnnTensorDescriptor_t dyd,
           const void* dy, cudnnTensorDescriptor_t dzd, void* dz, cudnnTensorDescriptor_t dxd,
           void* dx, cudnnTensorDescriptor_t p, const void* scale, const void* bias, void* dscale,
           void* dbias, double epsilon, const void* saved_mean, const void* saved_inverse,
           CudaDnnActivation a, void* ws, std::size_t wb, void* reserve, std::size_t rb) {
            Invocation call{
                CudaDnnBatchNormKind::kBackward,
                static_cast<std::uint32_t>(m),
                static_cast<std::uint32_t>(o),
                h,
                {xd, yd, dyd, dzd, dxd, p, a},
                {const_cast<void*>(x), const_cast<void*>(y), const_cast<void*>(dy), dz, dx,
                 const_cast<void*>(scale), const_cast<void*>(bias), dscale, dbias,
                 const_cast<void*>(saved_mean), const_cast<void*>(saved_inverse), ws, reserve},
                wb,
                rb,
                epsilon,
                0};
            return issue(call, {ad, bd, ap, bp});
        };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}

struct Case {
    bool is_double{}, omit_statistics{}, ignored_pointers{};
    std::uint32_t mode{CUDNN_BATCHNORM_SPATIAL}, operation{CUDNN_BATCHNORM_OPS_BN};
    double epsilon{0.25}, factor{0.375};
    std::array<double, 2> forward{1.5, -0.25};
    std::array<double, 4> backward{0.75, 0.125, 1.25, -0.5};
};
inline std::array<bool, 7> descriptor_mask(const Case& value, CudaDnnBatchNormKind kind) {
    if (kind == CudaDnnBatchNormKind::kReserve)
        return {true,
                false,
                false,
                false,
                false,
                false,
                value.operation != 0 || value.ignored_pointers};
    const bool forward = kind == CudaDnnBatchNormKind::kForward;
    return {true,
            forward || value.operation != 0,
            !forward,
            value.operation == 2,
            !forward,
            true,
            value.operation != 0 || value.ignored_pointers};
}
inline std::array<bool, 13> buffer_mask(const Case& value, CudaDnnBatchNormKind kind) {
    const bool forward = kind == CudaDnnBatchNormKind::kForward;
    return {true,
            forward || value.operation != 0 || value.ignored_pointers,
            !forward,
            value.operation == 2 || value.ignored_pointers,
            !forward,
            true,
            forward || value.operation != 0 || value.ignored_pointers,
            !forward || !value.omit_statistics,
            !forward || !value.omit_statistics,
            !value.omit_statistics,
            !value.omit_statistics,
            true,
            true};
}
inline std::array<CudaDnnTensor, 6> tensors(const Case& value) {
    const std::uint32_t type = value.is_double ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    std::array<CudaDnnTensor, 6> result{CudaDnnTensor{type, 0, 4, {2, 2, 1, 2}, {16, 6, 4, 2}},
                                        CudaDnnTensor{type, 0, 4, {2, 2, 1, 2}, {21, 8, 4, 3}},
                                        CudaDnnTensor{type, 0, 4, {2, 2, 1, 2}, {24, 9, 5, 2}},
                                        CudaDnnTensor{type, 0, 4, {2, 2, 1, 2}, {28, 11, 5, 2}},
                                        CudaDnnTensor{type, 0, 4, {2, 2, 1, 2}, {30, 12, 5, 3}},
                                        CudaDnnTensor{type, 0, 4, {1, 2, 1, 1}, {20, 7, 4, 2}}};
    if (value.mode == CUDNN_BATCHNORM_PER_ACTIVATION) result[5].sizes[3] = 2;
    return result;
}
inline constexpr std::array<double, 8> kInput{1, 3, 2, 6, 5, 7, 4, 8},
    kResidual{-1, 0.25, 2, -3, 0.5, -2, -0.25, 1}, kGradient{1, -2, 3, 0.5, -1, 2, -0.5, 4};
inline double parameter_initial(std::size_t slot, std::size_t index, const Case& value) {
    if (slot == 5)
        return value.mode == CUDNN_BATCHNORM_PER_ACTIVATION
                   ? std::array{1.25, 0.5, -0.75, 1.5}[index]
                   : std::array{1.25, -0.75}[index];
    if (slot == 6) return 0.5 + static_cast<double>(index) * 0.75;
    if (slot == 7) return 2 + static_cast<double>(index);
    if (slot == 8) return 3 + static_cast<double>(index);
    return -11 - static_cast<double>(index);
}
using Images = std::array<std::vector<std::byte>, 13>;
inline double image_read(const std::vector<std::byte>& image, const CudaDnnTensor& tensor,
                         std::size_t logical) {
    const auto byte = 16 + offset(tensor, logical) * (tensor.type == CUDNN_DATA_DOUBLE ? 8U : 4U);
    if (tensor.type == CUDNN_DATA_DOUBLE) {
        double value{};
        std::memcpy(&value, image.data() + byte, 8);
        return value;
    }
    float value{};
    std::memcpy(&value, image.data() + byte, 4);
    return value;
}
inline void image_write(std::vector<std::byte>& image, const CudaDnnTensor& tensor,
                        std::size_t logical, double value) {
    const auto byte = 16 + offset(tensor, logical) * (tensor.type == CUDNN_DATA_DOUBLE ? 8U : 4U);
    if (tensor.type == CUDNN_DATA_DOUBLE)
        std::memcpy(image.data() + byte, &value, 8);
    else {
        const auto converted = static_cast<float>(value);
        std::memcpy(image.data() + byte, &converted, 4);
    }
}
inline Images images(const Case& value) {
    const auto descriptors = tensors(value);
    Images result;
    for (std::size_t slot = 0; slot != result.size(); ++slot) {
        if (slot >= 11) {
            result[slot].assign(64, std::byte{0xcd});
            continue;
        }
        const auto& descriptor = descriptors[slot < 5 ? slot : 5];
        result[slot].assign(foundation::tensor_bytes(descriptor) + 32, std::byte{0xcd});
        for (std::size_t i = 0; i != count(descriptor); ++i)
            image_write(result[slot], descriptor, i,
                        slot == 0   ? kInput[i]
                        : slot == 2 ? kGradient[i]
                        : slot == 3 ? kResidual[i]
                        : slot >= 5 ? parameter_initial(slot, i, value)
                                    : 2 + static_cast<double>(i) * 0.125);
    }
    return result;
}
// Independent oracle uses fixed means/centered inputs and an explicit dense
// normalization Jacobian, not the native fixture's compact gradient reduction.
inline double centered(const Case& value, std::size_t i) {
    return value.mode == CUDNN_BATCHNORM_PER_ACTIVATION
               ? std::array{-2., -2., -1., -1., 2., 2., 1., 1.}[i]
               : std::array{-3., -1., -3., 1., 1., 3., -1., 3.}[i];
}
inline double mean(const Case& value, std::size_t g) {
    return value.mode == CUDNN_BATCHNORM_PER_ACTIVATION ? std::array{3., 5., 3., 7.}[g]
                                                        : std::array{4., 5.}[g];
}
inline double variance(const Case& value, std::size_t g) {
    return value.mode == CUDNN_BATCHNORM_PER_ACTIVATION ? (g < 2 ? 4. : 1.) : 5.;
}
inline double inverse(const Case& value, std::size_t g) {
    return 1 / std::sqrt(variance(value, g) + value.epsilon);
}
inline bool active(const Case& value, std::size_t i) {
    const auto g = group(value.mode, i);
    return !value.operation ||
           parameter_initial(5, g, value) * centered(value, i) * inverse(value, g) +
                   parameter_initial(6, g, value) + (value.operation == 2 ? kResidual[i] : 0) >
               0;
}
inline Images oracle_forward(const Case& value, Images result) {
    const auto desc = tensors(value);
    for (std::size_t i = 0; i != 8; ++i) {
        const auto g = group(value.mode, i);
        auto output = parameter_initial(5, g, value) * centered(value, i) * inverse(value, g) +
                      parameter_initial(6, g, value) + (value.operation == 2 ? kResidual[i] : 0);
        if (value.operation) output = std::max(0., output);
        image_write(
            result[1], desc[1], i,
            value.forward[0] * output + value.forward[1] * image_read(result[1], desc[1], i));
    }
    if (!value.omit_statistics)
        for (std::size_t g = 0; g != groups(value.mode); ++g) {
            const auto samples = static_cast<double>(8 / groups(value.mode));
            image_write(result[7], desc[5], g,
                        value.factor * mean(value, g) +
                            (1 - value.factor) * parameter_initial(7, g, value));
            image_write(result[8], desc[5], g,
                        value.factor * variance(value, g) * samples / (samples - 1) +
                            (1 - value.factor) * parameter_initial(8, g, value));
            image_write(result[9], desc[5], g, mean(value, g));
            image_write(result[10], desc[5], g, inverse(value, g));
        }
    if (required_workspace.load())
        std::fill_n(result[11].begin() + 16, required_workspace.load(), std::byte{0x4f});
    if (required_reserve.load()) {
        std::fill_n(result[12].begin() + 16, required_reserve.load(), std::byte{0xa7});
        for (std::size_t i = 0; i != 8; ++i)
            result[12][16 + i] = active(value, i) ? std::byte{1} : std::byte{0};
    }
    return result;
}
inline Images oracle_backward(const Case& value, Images result) {
    const auto desc = tensors(value);
    const auto samples = static_cast<double>(8 / groups(value.mode));
    for (std::size_t i = 0; i != 8; ++i) {
        const auto g = group(value.mode, i);
        const auto inv = inverse(value, g);
        double gradient{};
        for (std::size_t j = 0; j != 8; ++j)
            if (group(value.mode, j) == g)
                gradient += (((i == j ? 1. : 0.) - 1 / samples) * inv -
                             centered(value, i) * centered(value, j) * inv * inv * inv / samples) *
                            (active(value, j) ? kGradient[j] : 0);
        gradient *= parameter_initial(5, g, value);
        image_write(
            result[4], desc[4], i,
            value.backward[0] * gradient + value.backward[1] * image_read(result[4], desc[4], i));
        if (value.operation == 2)
            image_write(result[3], desc[3], i, active(value, i) ? kGradient[i] : 0);
    }
    for (std::size_t g = 0; g != groups(value.mode); ++g) {
        double scale{}, bias{};
        for (std::size_t i = 0; i != 8; ++i)
            if (group(value.mode, i) == g && active(value, i)) {
                scale += kGradient[i] * centered(value, i) * inverse(value, g);
                bias += kGradient[i];
            }
        image_write(
            result[7], desc[5], g,
            value.backward[2] * scale + value.backward[3] * image_read(result[7], desc[5], g));
        image_write(
            result[8], desc[5], g,
            value.backward[2] * bias + value.backward[3] * image_read(result[8], desc[5], g));
    }
    if (required_workspace.load())
        std::fill_n(result[11].begin() + 16, required_workspace.load(), std::byte{0x42});
    return result;
}
inline void verify(const Case& value, Images actual, Images expected) {
    const auto descriptors = tensors(value);
    for (std::size_t slot = 0; slot != actual.size(); ++slot) {
        base::require(actual[slot].size() == expected[slot].size());
        if (slot < 11) {
            const auto& descriptor = descriptors[slot < 5 ? slot : 5];
            for (std::size_t i = 0; i != count(descriptor); ++i) {
                const auto found = image_read(actual[slot], descriptor, i);
                const auto wanted = image_read(expected[slot], descriptor, i);
                base::require(std::abs(found - wanted) <=
                              (value.is_double ? 1e-12 : 5e-6) * std::max(1., std::abs(wanted)));
                image_write(actual[slot], descriptor, i, 0);
                image_write(expected[slot], descriptor, i, 0);
            }
        }
        base::require(actual[slot] == expected[slot]);
    }
}
}  // namespace glimmer::tests::dnn_batch_norm
