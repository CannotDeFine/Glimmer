#include "runtime.h"
#include "glimmer/control/execution_socket.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "internal/process_identity.h"
#include "../backend/dnn_fixture.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_native_calls();

namespace {
using namespace glimmer::app;
using namespace glimmer::backend::cuda;
namespace dnn = glimmer::tests::dnn;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location location = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public cuDNN status " + std::to_string(status) + " at line " +
                                 std::to_string(location.line()));
}
void driver_check(CUresult status,
                  std::source_location location = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public Driver status " + std::to_string(status) + " at line " +
                                 std::to_string(location.line()));
}
struct Controls {
    std::atomic<bool> fail_tensor{true}, fail_handle{}, fail_get_stream{}, fail_compute{};
    std::atomic<bool> partial_tensor{}, hold_completion{}, can_stop{true};
    std::atomic<unsigned int> context_creates{}, context_destroys{}, compute_calls{};
    std::atomic<unsigned int> host_registers{}, host_unregisters{};
    std::atomic<std::uintptr_t> native_handle{}, native_tensor{};
} controls;
struct Memory {
    CUcontext context{};
    std::vector<std::byte> bytes;
};
struct Stream {
    CUcontext context{};
};
struct Event {
    bool is_pending{};
};
struct Driver {
    CUcontext current{};
    bool has_library_ownership{};
    std::map<CUcontext, std::unique_ptr<int>> contexts;
    std::map<CUdeviceptr, Memory> memory;
    std::map<CUstream, std::unique_ptr<Stream>> streams;
    std::map<CUevent, std::unique_ptr<Event>> events;
    std::map<void*, std::size_t> host_mappings;
    unsigned int context_destroys{}, frees{}, stream_destroys{}, event_destroys{};
};
// CUDA current context and the fixture's owned resource records belong to the
// actual SDK caller thread. Pageable transfers use a separate HostOwner worker.
thread_local Driver native;
CUcontext current() {
    return native.current;
}
void tensor_bytes(cudnnTensorDescriptor_t descriptor, std::size_t* bytes) {
    const auto& object = dnn::object(descriptor, CudaDnnObjectKind::kTensor);
    std::size_t elements{1};
    for (int i = 0; i < object.tensor.dimensions; ++i) {
        const auto index = static_cast<std::size_t>(i);
        expect(object.tensor.sizes[index] > 0 && object.tensor.strides[index] > 0,
               "fixture tensor hull is positive and small");
        elements += static_cast<std::size_t>(object.tensor.sizes[index] - 1) *
                    static_cast<std::size_t>(object.tensor.strides[index]);
    }
    *bytes = elements * (object.tensor.type == CUDNN_DATA_DOUBLE ? sizeof(double) : sizeof(float));
}
template <typename Scalar>
void compute_values(const dnn::Invocation& call) {
    auto* destination = static_cast<Scalar*>(call.y);
    const auto* source = static_cast<const Scalar*>(call.x);
    const auto& tensor = dnn::object(call.x_descriptor, CudaDnnObjectKind::kTensor).tensor;
    expect(tensor.dimensions == 4 && tensor.sizes[0] == 1 && tensor.sizes[1] == 1 &&
               tensor.sizes[2] == 2 && tensor.sizes[3] == 3,
           "CPU arithmetic oracle only executes its declared six-element tensor");
    for (std::size_t index = 0; index < 6; ++index) {
        const auto input = call.activation ? std::max(Scalar{}, source[index]) : source[index];
        destination[index] = static_cast<Scalar>(call.alpha) * input +
                             static_cast<Scalar>(call.beta) * destination[index];
    }
}
cudnnStatus_t compute(dnn::Invocation call, const void* alpha, const void* beta) {
    const auto status = dnn::compute(call, alpha, beta);
    ++controls.compute_calls;
    if (controls.fail_compute.exchange(false)) return CUDNN_STATUS_EXECUTION_FAILED;
    if (status != CUDNN_STATUS_SUCCESS) return status;
    if (dnn::state.last_call.double_scalars)
        compute_values<double>(dnn::state.last_call);
    else
        compute_values<float>(dnn::state.last_call);
    return status;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* output, int ordinal) {
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.device_uuid = [](CUuuid* output, CUdevice) {
        *output = {};
        output->bytes[0] = 7;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* output) {
        *output = 12060;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice) {
        auto context = std::make_unique<int>(0);
        *output = reinterpret_cast<CUcontext>(context.get());
        native.contexts.emplace(*output, std::move(context));
        native.current = *output;
        ++controls.context_creates;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext context) {
        expect(!context || native.contexts.contains(context),
               "activate only a service-owned context");
        native.current = context;
        return CUDA_SUCCESS;
    };
    sdk.context_destroy = [](CUcontext context) {
        expect(context == native.current && native.contexts.contains(context),
               "destroy only the current owned context");
        if (native.has_library_ownership)
            for (const auto& handle : dnn::state.handles)
                expect(!handle.alive || handle.context != context,
                       "all native cuDNN handles retire before their context");
        expect(native.host_mappings.empty(), "host owner unregisters every mapping before context");
        std::erase_if(native.memory,
                      [context](const auto& item) { return item.second.context == context; });
        std::erase_if(native.streams,
                      [context](const auto& item) { return item.second->context == context; });
        native.contexts.erase(context);
        native.current = nullptr;
        ++native.context_destroys;
        ++controls.context_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        *output = native.current;
        return CUDA_SUCCESS;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.execution.event_create = [](CUevent* output, unsigned int) {
        auto event = std::make_unique<Event>();
        *output = reinterpret_cast<CUevent>(event.get());
        native.events.emplace(*output, std::move(event));
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent event, CUstream) {
        native.events.at(event)->is_pending = controls.hold_completion.load();
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent event) {
        return native.events.at(event)->is_pending && !controls.can_stop.load()
                   ? CUDA_ERROR_NOT_READY
                   : CUDA_SUCCESS;
    };
    sdk.execution.event_destroy = [](CUevent event) {
        expect(native.events.erase(event) == 1, "private event retires exactly once");
        ++native.event_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        return controls.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.module_load = [](CUmodule*, const void*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.module_unload = [](CUmodule) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        Memory memory{native.current, std::vector<std::byte>(bytes)};
        *output = reinterpret_cast<CUdeviceptr>(memory.bytes.data());
        expect(native.current && *output && !native.memory.contains(*output),
               "independent live allocation belongs to the active context");
        native.memory.emplace(*output, std::move(memory));
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr address) {
        expect(native.memory.at(address).context == native.current,
               "device release is performed by its exact owner");
        native.memory.erase(address);
        ++native.frees;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr destination, const void* source, std::size_t bytes) {
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download = [](void* destination, CUdeviceptr source, std::size_t bytes) {
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        return CUDA_SUCCESS;
    };
    sdk.upload_async = [](CUdeviceptr destination, const void* source, std::size_t bytes,
                          CUstream) {
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download_async = [](void* destination, CUdeviceptr source, std::size_t bytes, CUstream) {
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        return CUDA_SUCCESS;
    };
    sdk.host_register = [](void* pointer, std::size_t bytes, unsigned int) {
        expect(native.current && pointer && bytes &&
                   native.host_mappings.emplace(pointer, bytes).second,
               "staging host registration belongs to the independent current owner");
        ++controls.host_registers;
        return CUDA_SUCCESS;
    };
    sdk.host_unregister = [](void* pointer) {
        expect(native.current && native.host_mappings.erase(pointer) == 1,
               "staging host mapping unregisters exactly once on its actual owner");
        ++controls.host_unregisters;
        return CUDA_SUCCESS;
    };
    sdk.host_map = [](CUdeviceptr* output, void* pointer, unsigned int) {
        expect(native.current, "host map query executes on its actual current owner");
        if (!native.host_mappings.contains(pointer)) {
            // Import probes for proven absence before registering the private
            // mapping. An unregistered pointer is a normal SDK rejection.
            *output = 0;
            return CUDA_ERROR_INVALID_VALUE;
        }
        *output = reinterpret_cast<CUdeviceptr>(pointer);
        return CUDA_SUCCESS;
    };
    sdk.stream_create = [](CUstream* output, unsigned int) {
        auto stream = std::make_unique<Stream>(Stream{native.current});
        *output = reinterpret_cast<CUstream>(stream.get());
        native.streams.emplace(*output, std::move(stream));
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream stream) {
        expect(native.streams.at(stream)->context == native.current,
               "stream retirement uses the actual owned context");
        native.streams.erase(stream);
        ++native.stream_destroys;
        return CUDA_SUCCESS;
    };
    auto original = dnn::dispatch();
    original.tensor_create = [](cudnnTensorDescriptor_t* output) {
        if (controls.fail_tensor.exchange(false)) {
            ++dnn::state.creates;
            *output = nullptr;
            return CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED;
        }
        const auto status = dnn::dispatch().tensor_create(output);
        controls.native_tensor.store(reinterpret_cast<std::uintptr_t>(*output));
        return controls.partial_tensor.exchange(false) ? CUDNN_STATUS_INTERNAL_ERROR : status;
    };
    original.tensor_size = [](cudnnTensorDescriptor_t descriptor, std::size_t* bytes) {
        ++dnn::state.sizes;
        if (!dnn::object(descriptor, CudaDnnObjectKind::kTensor).configured) {
            *bytes = 991;
            return CUDNN_STATUS_BAD_PARAM;
        }
        tensor_bytes(descriptor, bytes);
        return CUDNN_STATUS_SUCCESS;
    };
    original.create = [](cudnnHandle_t* output) {
        native.has_library_ownership = true;
        if (controls.fail_handle.exchange(false)) {
            ++dnn::state.handle_creates;
            *output = nullptr;
            return CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED;
        }
        const auto status = dnn::dispatch().create(output);
        controls.native_handle.store(reinterpret_cast<std::uintptr_t>(*output));
        return status;
    };
    original.get_stream = [](cudnnHandle_t handle, cudaStream_t* output) {
        const auto status = dnn::dispatch().get_stream(handle, output);
        return controls.fail_get_stream.exchange(false) ? CUDNN_STATUS_BAD_PARAM : status;
    };
    original.activation_forward = [](cudnnHandle_t handle, CudaDnnActivation activation,
                                     const void* alpha, cudnnTensorDescriptor_t x_descriptor,
                                     const void* x, const void* beta,
                                     cudnnTensorDescriptor_t y_descriptor, void* y) {
        ++dnn::state.activation_calls;
        return compute({handle, activation, x_descriptor, y_descriptor, x, y}, alpha, beta);
    };
    original.add_tensor = [](cudnnHandle_t handle, const void* alpha,
                             cudnnTensorDescriptor_t x_descriptor, const void* x, const void* beta,
                             cudnnTensorDescriptor_t y_descriptor, void* y) {
        ++dnn::state.add_calls;
        return compute({handle, nullptr, x_descriptor, y_descriptor, x, y}, alpha, beta);
    };
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(original);
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
template <typename Function>
Function symbol(void* library, const char* name, Function direct) {
    auto* found = ::dlsym(library, name);
    expect(
        found && found == ::dlsym(RTLD_DEFAULT, name) && found == reinterpret_cast<void*>(direct),
        "all public cuDNN lookup paths select the controlled wrapper");
    return reinterpret_cast<Function>(found);
}

// The public legacy tensor API is the compatibility target of this fixture.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
void check_symbols(void* library) {
#define DNN_SYMBOL(NAME) (void)symbol(library, #NAME, &NAME)
    DNN_SYMBOL(cudnnCreate);
    DNN_SYMBOL(cudnnDestroy);
    DNN_SYMBOL(cudnnSetStream);
    DNN_SYMBOL(cudnnGetStream);
    DNN_SYMBOL(cudnnCreateTensorDescriptor);
    DNN_SYMBOL(cudnnDestroyTensorDescriptor);
    DNN_SYMBOL(cudnnSetTensor4dDescriptor);
    DNN_SYMBOL(cudnnSetTensor4dDescriptorEx);
    DNN_SYMBOL(cudnnGetTensor4dDescriptor);
    DNN_SYMBOL(cudnnSetTensorNdDescriptor);
    DNN_SYMBOL(cudnnSetTensorNdDescriptorEx);
    DNN_SYMBOL(cudnnGetTensorNdDescriptor);
    DNN_SYMBOL(cudnnGetTensorSizeInBytes);
    DNN_SYMBOL(cudnnCreateActivationDescriptor);
    DNN_SYMBOL(cudnnDestroyActivationDescriptor);
    DNN_SYMBOL(cudnnSetActivationDescriptor);
    DNN_SYMBOL(cudnnGetActivationDescriptor);
    DNN_SYMBOL(cudnnActivationForward);
    DNN_SYMBOL(cudnnAddTensor);
#undef DNN_SYMBOL
}
void metadata(cudnnTensorDescriptor_t& tensor, cudnnActivationDescriptor_t& activation) {
    tensor = reinterpret_cast<cudnnTensorDescriptor_t>(std::uintptr_t{0x12345});
    expect(cudnnCreateTensorDescriptor(&tensor) ==
                   CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED &&
               reinterpret_cast<std::uintptr_t>(tensor) == 0x12345,
           "native known-empty create failure preserves caller output");
    check(cudnnCreateTensorDescriptor(&tensor));
    expect(reinterpret_cast<std::uintptr_t>(tensor) != controls.native_tensor.load(),
           "caller descriptor is opaque, not a native server pointer");
    std::size_t bytes = 17;
    expect(cudnnGetTensorSizeInBytes(tensor, &bytes) == CUDNN_STATUS_BAD_PARAM && bytes == 17,
           "failed native size query cannot publish partial output");
    std::array<int, 12> dimensions, strides;
    dimensions.fill(91);
    strides.fill(92);
    auto type = CUDNN_DATA_DOUBLE;
    int rank = 93;
    expect(cudnnGetTensorNdDescriptor(tensor, 12, &type, &rank, dimensions.data(),
                                      strides.data()) == CUDNN_STATUS_BAD_PARAM &&
               type == CUDNN_DATA_DOUBLE && rank == 93 && dimensions[0] == 91 && strides[0] == 92,
           "failed original tensor query preserves all output fields");
    check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 3, 4, 5));
    check(cudnnGetTensor4dDescriptor(tensor, &type, &dimensions[0], &dimensions[1], &dimensions[2],
                                     &dimensions[3], &strides[0], &strides[1], &strides[2],
                                     &strides[3]));
    expect(type == CUDNN_DATA_FLOAT &&
               std::equal(dimensions.begin(), dimensions.begin() + 4,
                          std::array{2, 3, 4, 5}.begin()) &&
               std::equal(strides.begin(), strides.begin() + 4, std::array{60, 20, 5, 1}.begin()),
           "four-dimensional original tensor fields round trip exactly");
    check(cudnnSetTensor4dDescriptorEx(tensor, CUDNN_DATA_FLOAT, 2, 3, 4, 5, 100, 30, 6, 1));
    check(cudnnGetTensorSizeInBytes(tensor, &bytes));
    expect(bytes == 183 * sizeof(float), "original strided size uses the complete storage hull");
    const std::array<int, 4> shape{2, 3, 4, 5}, step{60, 20, 5, 1};
    check(cudnnSetTensorNdDescriptor(tensor, CUDNN_DATA_DOUBLE, 4, shape.data(), step.data()));
    check(
        cudnnSetTensorNdDescriptorEx(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 4, shape.data()));
    dimensions.fill(91);
    strides.fill(92);
    check(cudnnGetTensorNdDescriptor(tensor, 12, &type, &rank, dimensions.data(), strides.data()));
    expect(type == CUDNN_DATA_FLOAT && rank == 4 && dimensions[3] == 5 && strides[0] == 60 &&
               dimensions[4] == 91 && dimensions[11] == 91 && strides[4] == 92,
           "legal oversized capacity publishes only actual rank, not unwritten array tails");
    dimensions.fill(91);
    strides.fill(92);
    check(cudnnGetTensorNdDescriptor(tensor, 2, &type, &rank, dimensions.data(), strides.data()));
    expect(rank == 4 && dimensions[0] == 2 && dimensions[1] == 3 && dimensions[2] == 91 &&
               strides[0] == 60 && strides[1] == 20 && strides[2] == 92,
           "short query publishes requested prefix and retains actual native rank");
    expect(cudnnSetTensorNdDescriptor(tensor, static_cast<cudnnDataType_t>(31), 4, shape.data(),
                                      step.data()) == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
           "representable unsupported type preserves detailed original status");
    check(cudnnCreateActivationDescriptor(&activation));
    auto mode = CUDNN_ACTIVATION_TANH;
    auto nan = CUDNN_PROPAGATE_NAN;
    double coefficient = 91;
    expect(cudnnGetActivationDescriptor(activation, &mode, &nan, &coefficient) ==
                   CUDNN_STATUS_BAD_PARAM &&
               mode == CUDNN_ACTIVATION_TANH && nan == CUDNN_PROPAGATE_NAN && coefficient == 91,
           "native activation query failure preserves every caller output");
    check(cudnnSetActivationDescriptor(activation, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN,
                                       0.125));
    check(cudnnGetActivationDescriptor(activation, &mode, &nan, &coefficient));
    expect(mode == CUDNN_ACTIVATION_RELU && nan == CUDNN_NOT_PROPAGATE_NAN && coefficient == 0.125,
           "activation mode, NaN policy and full double coefficient round trip");
    expect(controls.context_creates.load() == 0,
           "all descriptor APIs execute without creating any CUDA context");
    const auto child = ::fork();
    expect(child >= 0, "fork exclusion fixture");
    if (!child) {
        std::size_t untouched = 887;
        const bool rejected =
            cudnnGetTensorSizeInBytes(tensor, &untouched) == CUDNN_STATUS_NOT_INITIALIZED &&
            untouched == 887;
        ::_exit(rejected ? 0 : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork cannot issue inherited authenticated descriptor commands");
}
template <typename Scalar>
void arithmetic(cudnnHandle_t handle, cudnnTensorDescriptor_t x, cudnnTensorDescriptor_t y,
                cudnnActivationDescriptor_t activation, CUdeviceptr source,
                CUdeviceptr destination) {
    constexpr auto type = sizeof(Scalar) == sizeof(double) ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    check(cudnnSetTensor4dDescriptor(x, CUDNN_TENSOR_NCHW, type, 1, 1, 2, 3));
    check(cudnnSetTensor4dDescriptor(y, CUDNN_TENSOR_NCHW, type, 1, 1, 2, 3));
    const std::array<Scalar, 6> input{-3, -2, -1, 0, 1, 2};
    std::array<Scalar, 6> output{1, 2, 3, 4, 5, 6};
    const Scalar alpha = 1.5, beta = -0.5;
    driver_check(cuMemcpyHtoD(source, input.data(), sizeof(input)));
    for (const bool is_activation : {true, false}) {
        const auto old = output;
        driver_check(cuMemcpyHtoD(destination, old.data(), sizeof(old)));
        check(is_activation
                  ? cudnnActivationForward(handle, activation, &alpha, x,
                                           reinterpret_cast<const void*>(source), &beta, y,
                                           reinterpret_cast<void*>(destination))
                  : cudnnAddTensor(handle, &alpha, x, reinterpret_cast<const void*>(source), &beta,
                                   y, reinterpret_cast<void*>(destination)));
        driver_check(cuMemcpyDtoH(output.data(), destination, sizeof(output)));
        for (std::size_t index = 0; index < output.size(); ++index)
            expect(output[index] ==
                       alpha * (is_activation ? std::max(Scalar{}, input[index]) : input[index]) +
                           beta * old[index],
                   "complete CPU-injected arithmetic output matches the unchanged oracle");
    }
}
void application(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent caller-native cuDNN provider");
    check_symbols(library);
    const auto cached_size =
        symbol(library, "cudnnGetTensorSizeInBytes", &cudnnGetTensorSizeInBytes);
    cudnnTensorDescriptor_t x{};
    cudnnActivationDescriptor_t activation{};
    metadata(x, activation);
    CUcontext first{}, second{};
    driver_check(cuCtxCreate(&first, 0, 0));
    controls.fail_handle.store(true);
    auto handle = reinterpret_cast<cudnnHandle_t>(std::uintptr_t{0x98765});
    expect(cudnnCreate(&handle) == CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED &&
               reinterpret_cast<std::uintptr_t>(handle) == 0x98765,
           "known-empty native handle failure leaves caller token untouched");
    check(cudnnCreate(&handle));
    expect(reinterpret_cast<std::uintptr_t>(handle) != controls.native_handle.load(),
           "native context handle is never exposed as a public token");
    CUstream stream{};
    driver_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
    cudaStream_t observed{};
    check(cudnnGetStream(handle, &observed));
    expect(observed == reinterpret_cast<cudaStream_t>(stream), "owned stream identity round trips");
    controls.fail_get_stream.store(true);
    observed = reinterpret_cast<cudaStream_t>(std::uintptr_t{0x7878});
    expect(cudnnGetStream(handle, &observed) == CUDNN_STATUS_BAD_PARAM &&
               reinterpret_cast<std::uintptr_t>(observed) == 0x7878,
           "native get-stream failure preserves caller output");
    driver_check(cuCtxCreate(&second, 0, 0));
    expect(cudnnGetStream(handle, &observed) == CUDNN_STATUS_NOT_INITIALIZED &&
               reinterpret_cast<std::uintptr_t>(observed) == 0x7878,
           "cross-context handle cannot issue to another native owner");
    cudnnHandle_t other{};
    check(cudnnCreate(&other));
    check(cudnnDestroy(other));
    driver_check(cuCtxDestroy(second));
    driver_check(cuCtxSetCurrent(first));
    std::size_t bytes{};
    expect(::dlclose(library) == 0, "close explicit vendor lookup reference");
    check(cached_size(x, &bytes));
    expect(bytes == 480, "cached controlled wrapper survives explicit provider close");
    cudnnTensorDescriptor_t y{};
    check(cudnnCreateTensorDescriptor(&y));
    CUdeviceptr source{}, destination{};
    driver_check(cuMemAlloc(&source, 64));
    driver_check(cuMemAlloc(&destination, 64));
    arithmetic<float>(handle, x, y, activation, source, destination);
    arithmetic<double>(handle, x, y, activation, source, destination);
    check(cudnnSetStream(handle, nullptr));
    observed = reinterpret_cast<cudaStream_t>(std::uintptr_t{1});
    check(cudnnGetStream(handle, &observed));
    expect(!observed, "default stream returns the public null representation");
    driver_check(cuStreamDestroy(stream));
    controls.hold_completion.store(true);
    controls.can_stop.store(false);
    controls.fail_compute.store(true);
    const double alpha = 2, beta = 0.5;
    expect(cudnnAddTensor(handle, &alpha, x, reinterpret_cast<const void*>(source), &beta, y,
                          reinterpret_cast<void*>(destination)) == CUDNN_STATUS_EXECUTION_FAILED,
           "actual original compute failure is not replaced by successful queue admission");
    auto descriptor_release =
        std::async(std::launch::async, [x] { return cudnnDestroyTensorDescriptor(x); });
    auto handle_release = std::async(std::launch::async, [first, handle] {
        driver_check(cuCtxSetCurrent(first));
        const auto result = cudnnDestroy(handle);
        driver_check(cuCtxSetCurrent(nullptr));
        return result;
    });
    auto memory_release = std::async(std::launch::async, [first, source] {
        driver_check(cuCtxSetCurrent(first));
        const auto result = cuMemFree(source);
        driver_check(cuCtxSetCurrent(nullptr));
        return result;
    });
    const bool descriptor_pending =
        descriptor_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool handle_pending =
        handle_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool memory_pending =
        memory_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    controls.can_stop.store(true);
    controls.hold_completion.store(false);
    check(descriptor_release.get());
    check(handle_release.get());
    driver_check(memory_release.get());
    expect(descriptor_pending && handle_pending && memory_pending,
           "native error and host ACK cannot release descriptors, handle or memory before stop");
    expect(cudnnDestroyTensorDescriptor(x) == CUDNN_STATUS_NOT_INITIALIZED &&
               cudnnDestroy(handle) == CUDNN_STATUS_NOT_INITIALIZED,
           "stale public tokens cannot repeat native destruction");
    check(cudnnDestroyTensorDescriptor(y));
    driver_check(cuMemFree(destination));
    driver_check(cuCtxDestroy(first));
    check(cudnnDestroyActivationDescriptor(activation));
    expect(controls.compute_calls.load() == 5 && glimmer_dnn_native_calls() == 0,
           "only the service SDK executes each computation exactly once");
}
void application_quarantine(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent quarantine provider");
    check_symbols(library);
    cudnnTensorDescriptor_t first{};
    check(cudnnCreateTensorDescriptor(&first));
    controls.partial_tensor.store(true);
    auto failed = reinterpret_cast<cudnnTensorDescriptor_t>(std::uintptr_t{0x12345});
    expect(cudnnCreateTensorDescriptor(&failed) != CUDNN_STATUS_SUCCESS &&
               reinterpret_cast<std::uintptr_t>(failed) == 0x12345 &&
               cudnnDestroyTensorDescriptor(first) != CUDNN_STATUS_SUCCESS &&
               glimmer_dnn_native_calls() == 0 && ::dlclose(library) == 0,
           "partial native creation never publishes, replays or guesses descriptor cleanup");
}
#pragma GCC diagnostic pop

void run(const char* provider, bool quarantine, bool profile) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "authenticated local service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-dnn-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 32;
    config.exit_when_idle = true;
    if (profile) {
        config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
        config.uses_original_kernels = true;
    }
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure real authenticated frontend transport");
    dnn::reset(current);
    controls.fail_tensor.store(!quarantine);
    ExecutionService service(std::move(config), dispatch());
    expect(service.start(), "start actual owned execution service");
    auto client = std::async(std::launch::async, [&] {
        if (quarantine)
            application_quarantine(provider);
        else
            application(provider);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{60};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public cuDNN IPC progress");
    client.get();
    expect(glimmer_dnn_native_calls() == 0, "caller-native cuDNN provider was never invoked");
    if (quarantine) {
        expect(dnn::state.creates == 2 && dnn::state.destroys == 0 &&
                   controls.context_creates.load() == 0,
               "uncertain CPU metadata retains ownership without native cleanup or GPU creation");
        return;
    }
    expect(service.request_stop(), "seal public cuDNN service");
    while (!service.finished())
        expect(service.step(1) && Clock::now() < deadline, "bounded physical shutdown");
    expect(dnn::state.creates == 4 && dnn::state.destroys == 3 && dnn::state.handle_creates == 3 &&
               dnn::state.handle_destroys == 2 && dnn::state.activation_calls == 2 &&
               dnn::state.add_calls == 3 && controls.context_creates.load() == 3 &&
               controls.context_destroys.load() == 3 && controls.host_registers.load() &&
               controls.host_registers.load() == controls.host_unregisters.load() &&
               native.context_destroys == 2 && native.frees == 2 && native.stream_destroys == 1 &&
               native.contexts.empty() && native.memory.empty() && native.streams.empty() &&
               native.events.empty(),
           "all original metadata, handles, buffers, events and contexts retire exactly once");
}

glimmer::control::ExecutionWireReply malformed_reply(unsigned int index) {
    using glimmer::control::ExecutionWireDisposition;
    glimmer::control::ExecutionWireReply reply;
    switch (index) {
        case 0:
            reply.status = -1;
            break;
        case 1:
            reply.status = 1024;
            break;
        case 2:
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = CUDA_ERROR_INVALID_VALUE;  // Missing original SDK status.
            break;
        case 3:
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = CUDA_ERROR_INVALID_VALUE;
            reply.original_status = CUDNN_STATUS_BAD_PARAM;  // Mismatched status domains.
            break;
        case 4:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.id = 19;  // Structurally valid, but descriptor creation cannot be replacement.
            break;
        case 5:
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = -1;
            reply.original_status = -1;
            break;
        case 6:
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = 8192;
            reply.original_status = 8192;
            break;
        case 7:
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
            reply.original_status = reply.status;
            break;
        case 8:
            reply.status = CUDA_ERROR_INVALID_VALUE;
            break;
        default:
            throw std::runtime_error("unknown hostile-reply fixture case");
    }
    return reply;
}

void malformed_case(unsigned int index) {
    using namespace glimmer::control;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual malformed-reply server identity");
    ExecutionSocketConfig config;
    config.socket_path =
        "/tmp/glimmer-execution-dnn-malformed-" + std::to_string(::getpid()) + ".sock";
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "authenticate isolated fake reply endpoint");
    // On a server-side assertion, close transport before waiting for the
    // asynchronous caller so a malformed-fixture failure cannot deadlock.
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start actual hostile-reply transport");
    const bool is_valid = index >= 7;
    client = std::async(std::launch::async, [=] {
        cudnnTensorDescriptor_t known{};
        check(cudnnCreateTensorDescriptor(&known));
        auto failed = reinterpret_cast<cudnnTensorDescriptor_t>(std::uintptr_t{0x12345});
        const auto actual = cudnnCreateTensorDescriptor(&failed);
        const auto expected = index == 7   ? CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE
                              : index == 8 ? CUDNN_STATUS_BAD_PARAM
                                           : CUDNN_STATUS_INTERNAL_ERROR;
        expect(actual == expected && reinterpret_cast<std::uintptr_t>(failed) == 0x12345,
               "status domain validation preserves exact errors and factory output");
        std::size_t bytes = 991;
        const auto queried = cudnnGetTensorSizeInBytes(known, &bytes);
        if (is_valid) {
            expect(queried == CUDNN_STATUS_SUCCESS && bytes == 64,
                   "valid native or policy rejection does not quarantine metadata");
            check(cudnnDestroyTensorDescriptor(known));
        } else {
            expect(queried == CUDNN_STATUS_NOT_INITIALIZED && bytes == 991,
                   "malformed status envelope quarantines known metadata before another request");
        }
        expect(glimmer_dnn_native_calls() == 0, "hostile reply never dispatches caller-native SDK");
    });
    unsigned int hellos{}, creations{}, queries{}, destructions{};
    unsigned int bindings{};
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < deadline, "bounded hostile-reply progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "real transport authenticates hostile-reply peer");
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = CUDNN_STATUS_SUCCESS;
            ExecutionPayloadReader input(request->message.payload);
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "descriptor frontend opens only independent CPU metadata role");
                ++hellos;
                reply.original_status.reset();
                const auto ownership = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(ownership.has_value(), "encode metadata-only bootstrap ownership");
                reply.data.assign(ownership->begin(), ownership->end());
            } else if (request->message.header.operation == ExecutionOpcode::kBindDnnProfile) {
                std::uint32_t version{};
                expect(hellos == 1 && !bindings && !creations &&
                           read_execution_dnn_profile(input, version) && version == CUDNN_VERSION,
                       "profile binds exact headers after Hello and before descriptor traffic");
                ++bindings;
                reply.original_status.reset();
                ExecutionPayloadWriter output;
                expect(write_execution_dnn_profile(output, version), "encode matching DNN profile");
                reply.data = output.take();
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && kind == 1 && input.empty(),
                       "exact descriptor creation payload");
                if (++creations == 1)
                    reply.id = 11;
                else
                    reply = malformed_reply(index);
            } else {
                std::uint64_t id{};
                expect(input.read_u64(id) && id == 11 && input.empty(),
                       "follow-up uses the exact earlier owned descriptor");
                if (request->message.header.operation == ExecutionOpcode::kDnnTensorSize) {
                    ++queries;
                    reply.address = 64;
                } else {
                    expect(
                        request->message.header.operation == ExecutionOpcode::kDestroyDnnMetadata,
                        "hostile-reply fixture never accepts unrelated SDK operations");
                    ++destructions;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded.has_value(), "encode intentionally hostile status envelope");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {17, 29};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver exact authenticated hostile reply");
        }
    }
    client.get();
    expect(bindings == 1, "one canonical profile binding precedes hostile SDK reply cases");
    expect(hellos == 1 && creations == 2 && queries == (is_valid ? 1U : 0U) &&
               destructions == (is_valid ? 1U : 0U),
           "quarantine emits no later remote calls; valid errors preserve normal ownership");
    server.stop();
}

void malformed_replies(const char* executable, const char* provider) {
    for (unsigned int index = 0; index != 9; ++index) {
        const auto value = std::to_string(index);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated frontend status case");
        if (!child) {
            // exec starts a fresh frontend process identity; plain fork must
            // reject inherited handles before reaching the transport.
            ::execl("/proc/self/exe", executable, provider, "--malformed-case", value.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("isolated hostile cuDNN reply case failed: " + value);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string_view(argv[2]) == "--malformed-replies") {
            malformed_replies(argv[0], argv[1]);
            return 0;
        }
        if (argc == 4 && std::string_view(argv[2]) == "--malformed-case") {
            const std::string_view value{argv[3]};
            expect(value.size() == 1 && value[0] >= '0' && value[0] <= '8',
                   "bounded isolated hostile-reply case");
            malformed_case(static_cast<unsigned int>(value[0] - '0'));
            ::_exit(0);  // Do not fabricate remote cleanup after malformed ownership.
        }
        const bool quarantine = argc == 3 && std::string_view(argv[2]) == "--quarantine";
        const bool profile = argc == 3 && std::string_view(argv[2]) == "--profile";
        expect(argc == 2 || quarantine || profile, "provider path and optional execution mode");
        run(argv[1], quarantine, profile);
        if (quarantine) ::_exit(0);
        std::cout << "Public cuDNN CPU IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        controls.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
