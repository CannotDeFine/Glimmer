#pragma once

#include "internal/cuda_dnn_registry.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace glimmer::tests::dnn {
using backend::cuda::CudaDnnActivation;
using backend::cuda::CudaDnnActivationState;
using backend::cuda::CudaDnnDispatch;
using backend::cuda::CudaDnnObjectKind;
using backend::cuda::CudaDnnTensor;

struct Object {
    bool alive{}, configured{};
    CudaDnnObjectKind kind{};
    CudaDnnTensor tensor{};
    CudaDnnActivationState activation{};
    std::size_t bytes{4096};
    unsigned int destroy_attempts{};
};
struct Handle {
    bool alive{};
    CUcontext context{};
    cudaStream_t stream{};
    unsigned int destroy_attempts{};
};
struct Invocation {
    cudnnHandle_t handle{};
    CudaDnnActivation activation{};
    cudnnTensorDescriptor_t x_descriptor{}, y_descriptor{};
    const void* x{};
    void* y{};
    double alpha{}, beta{};
    bool double_scalars{};
};
struct State {
    std::thread::id owner{std::this_thread::get_id()};
    pid_t process{::getpid()};
    CUcontext (*current)(){};
    std::array<Object, 32> objects{};
    std::array<Handle, 16> handles{};
    unsigned int creates{}, destroys{}, sets{}, gets{}, sizes{};
    std::array<unsigned int, 4> tensor_setters{};
    unsigned int handle_creates{}, handle_destroys{}, stream_sets{}, stream_gets{};
    unsigned int activation_calls{}, add_calls{};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS},
        set_status{CUDNN_STATUS_SUCCESS}, get_status{CUDNN_STATUS_SUCCESS},
        size_status{CUDNN_STATUS_SUCCESS};
    cudnnStatus_t handle_create_status{CUDNN_STATUS_SUCCESS},
        handle_destroy_status{CUDNN_STATUS_SUCCESS}, stream_status{CUDNN_STATUS_SUCCESS},
        compute_status{CUDNN_STATUS_SUCCESS};
    bool partial_create{}, null_create{}, duplicate_create{};
    bool partial_handle_create{}, null_handle_create{}, duplicate_handle_create{};
    int reported_rank{-1};
    std::optional<std::uint32_t> reported_type, reported_mode, reported_nan;
    // Fail this destruction, then allow later calls to detect accidental retry.
    unsigned int fail_destroy_call{};
    Invocation last_call{};
};
inline State state;

inline void require(bool condition) {
    if (!condition) throw std::runtime_error("cuDNN original SDK fixture invariant");
}
inline void reset(CUcontext (*current)() = nullptr) {
    state = {};
    state.current = current;
}
inline void owner() {
    require(state.owner == std::this_thread::get_id() && state.process == ::getpid());
}
inline Object& object(void* native, CudaDnnObjectKind kind) {
    owner();
    for (auto& candidate : state.objects)
        if (&candidate == native) {
            require(candidate.alive && candidate.kind == kind);
            return candidate;
        }
    throw std::runtime_error("unknown native cuDNN descriptor");
}
inline Handle& handle(cudnnHandle_t native) {
    owner();
    for (auto& candidate : state.handles)
        if (reinterpret_cast<cudnnHandle_t>(&candidate) == native) {
            require(candidate.alive &&
                    (!state.current || (state.current() && candidate.context == state.current())));
            return candidate;
        }
    throw std::runtime_error("unknown native cuDNN handle");
}
inline Object* create(CudaDnnObjectKind kind) {
    owner();
    ++state.creates;
    if (state.null_create || (state.create_status != CUDNN_STATUS_SUCCESS && !state.partial_create))
        return nullptr;
    if (state.duplicate_create)
        for (auto& candidate : state.objects)
            if (candidate.alive && candidate.kind == kind) return &candidate;
    for (auto& candidate : state.objects)
        if (!candidate.alive) {
            candidate = {};
            candidate.alive = true;
            candidate.kind = kind;
            return &candidate;
        }
    throw std::runtime_error("native cuDNN fixture descriptor capacity");
}
inline cudnnStatus_t destroy(void* native, CudaDnnObjectKind kind) {
    auto& candidate = object(native, kind);
    ++state.destroys;
    require(++candidate.destroy_attempts == 1);
    const auto status = state.fail_destroy_call == state.destroys ? CUDNN_STATUS_INTERNAL_ERROR
                                                                  : state.destroy_status;
    if (status == CUDNN_STATUS_SUCCESS) candidate.alive = false;
    return status;
}
inline cudnnStatus_t set_tensor(cudnnTensorDescriptor_t native, CudaDnnTensor tensor,
                                unsigned int setter) {
    auto& candidate = object(native, CudaDnnObjectKind::kTensor);
    ++state.sets;
    ++state.tensor_setters.at(setter);
    if (state.set_status != CUDNN_STATUS_SUCCESS) return state.set_status;
    if (tensor.type > CUDNN_DATA_FP4_E2M1 || tensor.format > 2)
        return CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
    if (tensor.dimensions < 1 || tensor.dimensions > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
    for (int i = 0; i < tensor.dimensions; ++i)
        if (tensor.sizes[static_cast<std::size_t>(i)] <= 0) return CUDNN_STATUS_BAD_PARAM;
    candidate.tensor = tensor;
    candidate.configured = true;
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnTensor tensor(cudnnDataType_t type, int dimensions, const int* sizes,
                            const int* strides, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    CudaDnnTensor result{.type = static_cast<std::uint32_t>(type),
                         .format = static_cast<std::uint32_t>(format),
                         .dimensions = dimensions};
    require(dimensions >= 0 && dimensions <= CUDNN_DIM_MAX);
    std::copy_n(sizes, dimensions, result.sizes.begin());
    if (strides)
        std::copy_n(strides, dimensions, result.strides.begin());
    else {
        int stride = 1;
        for (int i = dimensions; i != 0; --i) {
            const auto index = static_cast<std::size_t>(i - 1);
            result.strides[index] = stride;
            // Fixtures use small dimensions; invalid shapes must still reach
            // the fake vendor's status path without arithmetic overflow.
            if (result.sizes[index] > 0 &&
                stride <= std::numeric_limits<int>::max() / result.sizes[index])
                stride *= result.sizes[index];
        }
    }
    return result;
}
inline cudnnStatus_t get_tensor(cudnnTensorDescriptor_t native, int requested,
                                cudnnDataType_t* type, int* dimensions, int* sizes, int* strides) {
    auto& candidate = object(native, CudaDnnObjectKind::kTensor);
    ++state.gets;
    *type = static_cast<cudnnDataType_t>(candidate.tensor.type);
    if (state.reported_type) std::memcpy(type, &*state.reported_type, sizeof(*type));
    *dimensions = state.reported_rank >= 0 ? state.reported_rank : candidate.tensor.dimensions;
    const auto copied = std::min(requested, candidate.tensor.dimensions);
    std::copy_n(candidate.tensor.sizes.begin(), copied, sizes);
    std::copy_n(candidate.tensor.strides.begin(), copied, strides);
    if (state.get_status != CUDNN_STATUS_SUCCESS) return state.get_status;
    return candidate.configured ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}
inline cudnnStatus_t compute(Invocation invocation, const void* alpha, const void* beta) {
    (void)handle(invocation.handle);
    auto& x = object(invocation.x_descriptor, CudaDnnObjectKind::kTensor);
    auto& y = object(invocation.y_descriptor, CudaDnnObjectKind::kTensor);
    require(x.configured && y.configured && alpha && beta && invocation.x && invocation.y);
    if (invocation.activation)
        require(object(invocation.activation, CudaDnnObjectKind::kActivation).configured);
    invocation.double_scalars = x.tensor.type == CUDNN_DATA_DOUBLE;
    if (invocation.double_scalars) {
        require(reinterpret_cast<std::uintptr_t>(alpha) % alignof(double) == 0 &&
                reinterpret_cast<std::uintptr_t>(beta) % alignof(double) == 0);
        std::memcpy(&invocation.alpha, alpha, sizeof(double));
        std::memcpy(&invocation.beta, beta, sizeof(double));
    } else {
        require(reinterpret_cast<std::uintptr_t>(alpha) % alignof(float) == 0 &&
                reinterpret_cast<std::uintptr_t>(beta) % alignof(float) == 0);
        float a{}, b{};
        std::memcpy(&a, alpha, sizeof(float));
        std::memcpy(&b, beta, sizeof(float));
        invocation.alpha = a;
        invocation.beta = b;
    }
    state.last_call = invocation;
    return state.compute_status;
}
inline CudaDnnDispatch dispatch() {
    CudaDnnDispatch sdk;
    sdk.create = [](cudnnHandle_t* output) {
        owner();
        ++state.handle_creates;
        require(!state.current || state.current());
        *output = nullptr;
        if (state.null_handle_create ||
            (state.handle_create_status != CUDNN_STATUS_SUCCESS && !state.partial_handle_create))
            return state.handle_create_status;
        if (state.duplicate_handle_create)
            for (auto& candidate : state.handles)
                if (candidate.alive) {
                    *output = reinterpret_cast<cudnnHandle_t>(&candidate);
                    return state.handle_create_status;
                }
        for (auto& candidate : state.handles)
            if (!candidate.alive) {
                candidate = {};
                candidate.alive = true;
                candidate.context = state.current ? state.current() : nullptr;
                *output = reinterpret_cast<cudnnHandle_t>(&candidate);
                return state.handle_create_status;
            }
        throw std::runtime_error("native cuDNN fixture handle capacity");
    };
    sdk.destroy = [](cudnnHandle_t native) {
        auto& candidate = handle(native);
        ++state.handle_destroys;
        require(++candidate.destroy_attempts == 1);
        if (state.handle_destroy_status == CUDNN_STATUS_SUCCESS) candidate.alive = false;
        return state.handle_destroy_status;
    };
    sdk.set_stream = [](cudnnHandle_t native, cudaStream_t stream) {
        auto& candidate = handle(native);
        ++state.stream_sets;
        if (state.stream_status == CUDNN_STATUS_SUCCESS) candidate.stream = stream;
        return state.stream_status;
    };
    sdk.get_stream = [](cudnnHandle_t native, cudaStream_t* stream) {
        auto& candidate = handle(native);
        ++state.stream_gets;
        *stream = candidate.stream;
        return state.stream_status;
    };
    sdk.tensor_create = [](cudnnTensorDescriptor_t* output) {
        *output = reinterpret_cast<cudnnTensorDescriptor_t>(create(CudaDnnObjectKind::kTensor));
        return state.create_status;
    };
    sdk.tensor_destroy = [](cudnnTensorDescriptor_t native) {
        return destroy(native, CudaDnnObjectKind::kTensor);
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): original SDK signatures.
    sdk.tensor_set_4d = [](cudnnTensorDescriptor_t native, cudnnTensorFormat_t format,
                           cudnnDataType_t type, int n, int c, int h, int w) {
        const std::array dimensions{n, c, h, w};
        return set_tensor(native, tensor(type, 4, dimensions.data(), nullptr, format), 0);
    };
    sdk.tensor_set_4d_ex = [](cudnnTensorDescriptor_t native, cudnnDataType_t type, int n, int c,
                              int h, int w, int ns, int cs, int hs, int ws) {
        const std::array dimensions{n, c, h, w}, strides{ns, cs, hs, ws};
        return set_tensor(native, tensor(type, 4, dimensions.data(), strides.data()), 1);
    };
    sdk.tensor_set_nd = [](cudnnTensorDescriptor_t native, cudnnDataType_t type, int dimensions,
                           const int* sizes, const int* strides) {
        return set_tensor(native, tensor(type, dimensions, sizes, strides), 2);
    };
    sdk.tensor_set_nd_ex = [](cudnnTensorDescriptor_t native, cudnnTensorFormat_t format,
                              cudnnDataType_t type, int dimensions, const int* sizes) {
        return set_tensor(native, tensor(type, dimensions, sizes, nullptr, format), 3);
    };
    sdk.tensor_get_4d = [](cudnnTensorDescriptor_t native, cudnnDataType_t* type, int* n, int* c,
                           int* h, int* w, int* ns, int* cs, int* hs, int* ws) {
        int rank{};
        std::array<int, 4> dimensions{}, strides{};
        const auto status = get_tensor(native, 4, type, &rank, dimensions.data(), strides.data());
        const std::array outputs{n, c, h, w}, stride_outputs{ns, cs, hs, ws};
        for (std::size_t i = 0; i != 4; ++i) {
            *outputs[i] = dimensions[i];
            *stride_outputs[i] = strides[i];
        }
        return status == CUDNN_STATUS_SUCCESS && rank != 4 ? CUDNN_STATUS_BAD_PARAM : status;
    };
    sdk.tensor_get_nd = &get_tensor;
    sdk.tensor_size = [](cudnnTensorDescriptor_t native, std::size_t* bytes) {
        auto& candidate = object(native, CudaDnnObjectKind::kTensor);
        ++state.sizes;
        *bytes = candidate.bytes;
        if (state.size_status != CUDNN_STATUS_SUCCESS) return state.size_status;
        return candidate.configured ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
    };
    sdk.activation_create = [](CudaDnnActivation* output) {
        *output = reinterpret_cast<CudaDnnActivation>(create(CudaDnnObjectKind::kActivation));
        return state.create_status;
    };
    sdk.activation_destroy = [](CudaDnnActivation native) {
        return destroy(native, CudaDnnObjectKind::kActivation);
    };
    sdk.activation_set = [](CudaDnnActivation native, cudnnActivationMode_t mode,
                            cudnnNanPropagation_t nan, double coefficient) {
        auto& candidate = object(native, CudaDnnObjectKind::kActivation);
        ++state.sets;
        if (state.set_status != CUDNN_STATUS_SUCCESS) return state.set_status;
        if (static_cast<unsigned int>(mode) > CUDNN_ACTIVATION_SWISH)
            return CUDNN_STATUS_NOT_SUPPORTED;
        candidate.activation = {static_cast<std::uint32_t>(mode), static_cast<std::uint32_t>(nan),
                                coefficient};
        candidate.configured = true;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.activation_get = [](CudaDnnActivation native, cudnnActivationMode_t* mode,
                            cudnnNanPropagation_t* nan, double* coefficient) {
        auto& candidate = object(native, CudaDnnObjectKind::kActivation);
        ++state.gets;
        *mode = static_cast<cudnnActivationMode_t>(candidate.activation.mode);
        *nan = static_cast<cudnnNanPropagation_t>(candidate.activation.nan);
        if (state.reported_mode) std::memcpy(mode, &*state.reported_mode, sizeof(*mode));
        if (state.reported_nan) std::memcpy(nan, &*state.reported_nan, sizeof(*nan));
        *coefficient = candidate.activation.coefficient;
        if (state.get_status != CUDNN_STATUS_SUCCESS) return state.get_status;
        return candidate.configured ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
    };
    sdk.activation_forward = [](cudnnHandle_t native, CudaDnnActivation activation,
                                const void* alpha, cudnnTensorDescriptor_t x_descriptor,
                                const void* x, const void* beta,
                                cudnnTensorDescriptor_t y_descriptor, void* y) {
        ++state.activation_calls;
        return compute({native, activation, x_descriptor, y_descriptor, x, y}, alpha, beta);
    };
    sdk.add_tensor = [](cudnnHandle_t native, const void* alpha,
                        cudnnTensorDescriptor_t x_descriptor, const void* x, const void* beta,
                        cudnnTensorDescriptor_t y_descriptor, void* y) {
        ++state.add_calls;
        return compute({native, nullptr, x_descriptor, y_descriptor, x, y}, alpha, beta);
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return sdk;
}
}  // namespace glimmer::tests::dnn
