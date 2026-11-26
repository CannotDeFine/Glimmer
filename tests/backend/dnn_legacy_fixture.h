#pragma once

#include "internal/cuda_dnn_legacy.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>
#include <unistd.h>

// Independent CPU ownership/ABI scaffolding for the legacy fixture. It does
// not import SDK9 empirical or numerical oracles, simulate NVIDIA RNN math,
// initialize CUDA, or use enum names unavailable in the selected SDK8 headers.
namespace glimmer::tests::dnn_legacy {
using namespace backend::cuda;
namespace foundation = ::glimmer::tests::dnn_legacy;
namespace base = ::glimmer::tests::dnn_legacy;
namespace metadata = ::glimmer::tests::dnn_legacy;
struct Object {
    bool alive{}, configured{};
    CudaDnnObjectKind kind{};
    CudaDnnTensor tensor{};
    CudaDnnFilter filter{};
    CudaDnnActivationState activation{};
    struct Rnn {
        CudaDnnDropout dropout{};
    } rnn;
};
struct Handle {
    bool alive{};
    CUcontext context{};
    cudaStream_t stream{};
};
struct Resource {
    CUcontext context{};
    std::unique_ptr<int> identity;
};
struct Memory {
    CUcontext context{};
    std::vector<std::byte> bytes;
};
struct Driver {
    CUcontext current{};
    std::map<CUcontext, std::unique_ptr<int>> contexts;
    std::map<CUstream, Resource> streams;
    std::map<CUevent, Resource> events;
    std::map<CUdeviceptr, Memory> memory;
    std::map<void*, std::size_t> host_mappings;
    unsigned int frees{};
};
struct State {
    const std::thread::id thread{std::this_thread::get_id()};
    const pid_t process{::getpid()};
    std::array<Object, 32> objects{};
    std::array<Handle, 16> handles{};
};
struct Control {
    std::atomic<bool> can_stop{true};
};
inline Driver driver;
inline std::unique_ptr<State> state;
inline Control control;
inline thread_local bool arm_allocation_failure{}, fail_next_allocation{};
inline thread_local unsigned int allocation_failures{};
inline void require(bool condition) {
    if (!condition) throw std::runtime_error("legacy fixture ownership or exact ABI invariant");
}
inline void owner() {
    require(state && state->thread == std::this_thread::get_id() && state->process == ::getpid());
}
inline void reset() {
    driver = {};
    state = std::make_unique<State>();
    control.can_stop.store(true);
    arm_allocation_failure = fail_next_allocation = false;
    allocation_failures = 0;
}
inline Object& object(void* pointer, CudaDnnObjectKind kind) {
    owner();
    for (auto& value : state->objects)
        if (&value == pointer) {
            require(value.alive && value.kind == kind);
            return value;
        }
    throw std::runtime_error("unknown legacy native descriptor");
}
inline Object& descriptor(void* pointer, CudaDnnObjectKind kind) {
    return object(pointer, kind);
}
inline void* create(CudaDnnObjectKind kind) {
    owner();
    for (auto& value : state->objects)
        if (!value.alive) {
            value = {};
            value.alive = true;
            value.kind = kind;
            return &value;
        }
    throw std::runtime_error("legacy fixture descriptor capacity");
}
inline cudnnStatus_t destroy(void* pointer, CudaDnnObjectKind kind) {
    object(pointer, kind).alive = false;
    return CUDNN_STATUS_SUCCESS;
}
inline Handle& handle(cudnnHandle_t pointer) {
    owner();
    for (auto& value : state->handles)
        if (reinterpret_cast<cudnnHandle_t>(&value) == pointer) {
            require(value.alive && driver.current && value.context == driver.current);
            return value;
        }
    throw std::runtime_error("unknown legacy native handle");
}
inline cudnnStatus_t set_tensor(cudnnTensorDescriptor_t pointer, cudnnDataType_t type, int count,
                                const int* sizes, const int* strides) {
    auto& value = object(pointer, CudaDnnObjectKind::kTensor);
    if (type != CUDNN_DATA_FLOAT || count < 1 || count > CUDNN_DIM_MAX || !sizes)
        return CUDNN_STATUS_BAD_PARAM;
    CudaDnnTensor tensor{.type = static_cast<std::uint32_t>(type), .dimensions = count};
    int stride = 1;
    for (int i = count; i > 0; --i) {
        const auto index = static_cast<std::size_t>(i - 1);
        require(sizes[index] > 0 && sizes[index] <= 1024 && stride <= 1024 * 1024);
        tensor.sizes[index] = sizes[index];
        tensor.strides[index] = strides ? strides[index] : stride;
        stride *= sizes[index];
    }
    value.tensor = tensor;
    value.configured = true;
    return CUDNN_STATUS_SUCCESS;
}
inline cudnnStatus_t get_tensor(cudnnTensorDescriptor_t pointer, int capacity,
                                cudnnDataType_t* type, int* rank, int* sizes, int* strides) {
    const auto& value = object(pointer, CudaDnnObjectKind::kTensor);
    if (!value.configured) return CUDNN_STATUS_BAD_PARAM;
    require(capacity >= value.tensor.dimensions && type && rank && sizes && strides);
    *type = static_cast<cudnnDataType_t>(value.tensor.type);
    *rank = value.tensor.dimensions;
    std::copy_n(value.tensor.sizes.begin(), *rank, sizes);
    std::copy_n(value.tensor.strides.begin(), *rank, strides);
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch() {
    CudaDnnDispatch sdk;
    sdk.version = [] { return static_cast<std::size_t>(CUDNN_VERSION); };
    sdk.create = [](cudnnHandle_t* output) {
        owner();
        require(driver.current && output);
        for (auto& value : state->handles)
            if (!value.alive) {
                value = {true, driver.current, nullptr};
                *output = reinterpret_cast<cudnnHandle_t>(&value);
                return CUDNN_STATUS_SUCCESS;
            }
        return CUDNN_STATUS_INTERNAL_ERROR;
    };
    sdk.destroy = [](cudnnHandle_t pointer) {
        handle(pointer).alive = false;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.set_stream = [](cudnnHandle_t pointer, cudaStream_t stream) {
        handle(pointer).stream = stream;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.get_stream = [](cudnnHandle_t pointer, cudaStream_t* output) {
        *output = handle(pointer).stream;
        if (arm_allocation_failure) {
            arm_allocation_failure = false;
            fail_next_allocation = true;
        }
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.tensor_create = [](cudnnTensorDescriptor_t* output) {
        *output = reinterpret_cast<cudnnTensorDescriptor_t>(create(CudaDnnObjectKind::kTensor));
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.tensor_destroy = [](cudnnTensorDescriptor_t pointer) {
        return destroy(pointer, CudaDnnObjectKind::kTensor);
    };
    sdk.tensor_set_nd = set_tensor;
    sdk.tensor_set_nd_ex = [](cudnnTensorDescriptor_t pointer, cudnnTensorFormat_t,
                              cudnnDataType_t type, int count, const int* sizes) {
        return set_tensor(pointer, type, count, sizes, nullptr);
    };
    sdk.tensor_set_4d = [](cudnnTensorDescriptor_t pointer, cudnnTensorFormat_t,
                           cudnnDataType_t type, int n, int c, int h, int w) {
        const std::array sizes{n, c, h, w};
        return set_tensor(pointer, type, 4, sizes.data(), nullptr);
    };
    sdk.tensor_set_4d_ex = [](cudnnTensorDescriptor_t pointer, cudnnDataType_t type, int n, int c,
                              int h, int w, int ns, int cs, int hs, int ws) {
        const std::array sizes{n, c, h, w}, strides{ns, cs, hs, ws};
        return set_tensor(pointer, type, 4, sizes.data(), strides.data());
    };
    sdk.tensor_get_nd = get_tensor;
    sdk.tensor_get_4d = [](cudnnTensorDescriptor_t pointer, cudnnDataType_t* type, int* n, int* c,
                           int* h, int* w, int* ns, int* cs, int* hs, int* ws) {
        int rank{};
        std::array<int, 4> sizes{}, strides{};
        const auto result = get_tensor(pointer, 4, type, &rank, sizes.data(), strides.data());
        if (result != CUDNN_STATUS_SUCCESS || rank != 4) return CUDNN_STATUS_BAD_PARAM;
        *n = sizes[0];
        *c = sizes[1];
        *h = sizes[2];
        *w = sizes[3];
        *ns = strides[0];
        *cs = strides[1];
        *hs = strides[2];
        *ws = strides[3];
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.tensor_size = [](cudnnTensorDescriptor_t pointer, std::size_t* output) {
        const auto& value = object(pointer, CudaDnnObjectKind::kTensor);
        if (!value.configured) return CUDNN_STATUS_BAD_PARAM;
        std::size_t elements = 1;
        for (int i = 0; i < value.tensor.dimensions; ++i) {
            const auto index = static_cast<std::size_t>(i);
            require(value.tensor.strides[index] > 0);
            elements += static_cast<std::size_t>(value.tensor.sizes[index] - 1) *
                        static_cast<std::size_t>(value.tensor.strides[index]);
        }
        *output = elements * sizeof(float);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.activation_create = [](CudaDnnActivation* output) {
        *output = static_cast<CudaDnnActivation>(create(CudaDnnObjectKind::kActivation));
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.activation_destroy = [](CudaDnnActivation pointer) {
        return destroy(pointer, CudaDnnObjectKind::kActivation);
    };
    sdk.activation_set = [](CudaDnnActivation pointer, cudnnActivationMode_t mode,
                            cudnnNanPropagation_t nan, double coefficient) {
        auto& value = object(pointer, CudaDnnObjectKind::kActivation);
        value.activation = {static_cast<std::uint32_t>(mode), static_cast<std::uint32_t>(nan),
                            coefficient};
        value.configured = true;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.activation_get = [](CudaDnnActivation pointer, cudnnActivationMode_t* mode,
                            cudnnNanPropagation_t* nan, double* coefficient) {
        const auto& value = object(pointer, CudaDnnObjectKind::kActivation);
        if (!value.configured) return CUDNN_STATUS_BAD_PARAM;
        *mode = static_cast<cudnnActivationMode_t>(value.activation.mode);
        *nan = static_cast<cudnnNanPropagation_t>(value.activation.nan);
        *coefficient = value.activation.coefficient;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.filter_create = [](CudaDnnFilterHandle* output) {
        *output = static_cast<CudaDnnFilterHandle>(create(CudaDnnObjectKind::kFilter));
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.filter_destroy = [](CudaDnnFilterHandle pointer) {
        return destroy(pointer, CudaDnnObjectKind::kFilter);
    };
    sdk.filter_set_nd = [](CudaDnnFilterHandle pointer, cudnnDataType_t type,
                           cudnnTensorFormat_t format, int count, const int* sizes) {
        auto& value = object(pointer, CudaDnnObjectKind::kFilter);
        require(type == CUDNN_DATA_FLOAT && format == CUDNN_TENSOR_NCHW && count > 0 &&
                count <= CUDNN_DIM_MAX);
        value.filter = {
            static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(format), count, {}};
        std::copy_n(sizes, count, value.filter.sizes.begin());
        value.configured = true;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.filter_get_nd = [](CudaDnnFilterHandle pointer, int capacity, cudnnDataType_t* type,
                           cudnnTensorFormat_t* format, int* rank, int* sizes) {
        const auto& value = object(pointer, CudaDnnObjectKind::kFilter);
        if (!value.configured) return CUDNN_STATUS_BAD_PARAM;
        require(capacity >= value.filter.dimensions);
        *type = static_cast<cudnnDataType_t>(value.filter.type);
        *format = static_cast<cudnnTensorFormat_t>(value.filter.format);
        *rank = value.filter.dimensions;
        std::copy_n(value.filter.sizes.begin(), *rank, sizes);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.filter_size = [](CudaDnnFilterHandle pointer, std::size_t* output) {
        const auto& value = object(pointer, CudaDnnObjectKind::kFilter);
        if (!value.configured) return CUDNN_STATUS_BAD_PARAM;
        std::size_t bytes = sizeof(float);
        for (int i = 0; i < value.filter.dimensions; ++i) {
            const auto extent = value.filter.sizes[static_cast<std::size_t>(i)];
            require(extent > 0 && extent <= 1024);
            bytes *= static_cast<std::size_t>(extent);
        }
        *output = bytes;
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.rnn_create = [](CudaDnnRnn* output) {
        *output = static_cast<CudaDnnRnn>(create(CudaDnnObjectKind::kRnn));
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.rnn_destroy = [](CudaDnnRnn pointer) { return destroy(pointer, CudaDnnObjectKind::kRnn); };
    sdk.dropout_create = [](CudaDnnDropout* output) {
        *output = static_cast<CudaDnnDropout>(create(CudaDnnObjectKind::kDropout));
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.dropout_destroy = [](CudaDnnDropout pointer) {
        return destroy(pointer, CudaDnnObjectKind::kDropout);
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) {
        owner();
        return CUDA_SUCCESS;
    };
    sdk.device_get = [](CUdevice* device, int ordinal) {
        *device = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        auto token = std::make_unique<int>(0);
        *output = reinterpret_cast<CUcontext>(token.get());
        driver.contexts.emplace(*output, std::move(token));
        driver.current = *output;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext context) {
        require(!context || driver.contexts.contains(context));
        driver.current = context;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        *output = driver.current;
        return CUDA_SUCCESS;
    };
    sdk.context_synchronize = [] {
        require(driver.current);
        return CUDA_SUCCESS;
    };
    sdk.context_destroy = [](CUcontext context) {
        require(context == driver.current && driver.contexts.contains(context) &&
                driver.host_mappings.empty());
        for (const auto& value : state->handles) require(!value.alive || value.context != context);
        std::erase_if(driver.memory,
                      [=](const auto& entry) { return entry.second.context == context; });
        std::erase_if(driver.streams,
                      [=](const auto& entry) { return entry.second.context == context; });
        std::erase_if(driver.events,
                      [=](const auto& entry) { return entry.second.context == context; });
        driver.contexts.erase(context);
        driver.current = nullptr;
        return CUDA_SUCCESS;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        require(driver.current);
        Memory memory{driver.current, std::vector<std::byte>(bytes, std::byte{0xcd})};
        *output = reinterpret_cast<CUdeviceptr>(memory.bytes.data());
        require(driver.memory.emplace(*output, std::move(memory)).second);
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr address) {
        require(driver.memory.at(address).context == driver.current);
        driver.memory.erase(address);
        ++driver.frees;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr output, const void* input, std::size_t bytes) {
        std::memcpy(reinterpret_cast<void*>(output), input, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download = [](void* output, CUdeviceptr input, std::size_t bytes) {
        std::memcpy(output, reinterpret_cast<const void*>(input), bytes);
        return CUDA_SUCCESS;
    };
    sdk.host_register = [](void* address, std::size_t bytes, unsigned int) {
        require(driver.current && driver.host_mappings.emplace(address, bytes).second);
        return CUDA_SUCCESS;
    };
    sdk.host_unregister = [](void* address) {
        require(driver.current && driver.host_mappings.erase(address) == 1);
        return CUDA_SUCCESS;
    };
    sdk.host_map = [](CUdeviceptr* output, void* address, unsigned int) {
        require(driver.current && driver.host_mappings.contains(address));
        *output = reinterpret_cast<CUdeviceptr>(address);
        return CUDA_SUCCESS;
    };
    sdk.stream_create = [](CUstream* output, unsigned int) {
        require(driver.current);
        auto token = std::make_unique<int>(0);
        *output = reinterpret_cast<CUstream>(token.get());
        driver.streams.emplace(*output, Resource{driver.current, std::move(token)});
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream stream) {
        require(driver.streams.at(stream).context == driver.current &&
                driver.streams.erase(stream) == 1);
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.execution.event_create = [](CUevent* output, unsigned int) {
        require(driver.current);
        auto token = std::make_unique<int>(0);
        *output = reinterpret_cast<CUevent>(token.get());
        driver.events.emplace(*output, Resource{driver.current, std::move(token)});
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent event, CUstream) {
        require(driver.events.at(event).context == driver.current);
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent event) {
        require(driver.events.at(event).context == driver.current);
        return control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent event) {
        require(driver.events.at(event).context == driver.current &&
                driver.events.erase(event) == 1);
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        return control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.device_uuid = [](CUuuid* output, CUdevice) {
        *output = {};
        output->bytes[0] = 61;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* output) {
        *output = 12060;
        return CUDA_SUCCESS;
    };
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_legacy
