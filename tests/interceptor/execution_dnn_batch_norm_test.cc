#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/dnn_batch_norm_fixture.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <source_location>
#include <dlfcn.h>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" unsigned int glimmer_dnn_batch_norm_native_calls();
namespace {
using namespace glimmer::app;
using glimmer::backend::cuda::CudaDnnBatchNormKind;
namespace fixture = glimmer::tests::dnn_batch_norm;
namespace foundation = fixture::foundation;
namespace base = glimmer::tests::dnn;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public BN SDK status " + std::to_string(status) + " at line " +
                                 std::to_string(where.line()));
}
void check(CUresult status, std::source_location where = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public BN Driver status " + std::to_string(status) + " at line " +
                                 std::to_string(where.line()));
}
template <typename Function>
void symbol(void* library, const char* name, Function direct) {
    expect(::dlsym(library, name) == reinterpret_cast<void*>(direct) &&
               ::dlsym(RTLD_DEFAULT, name) == reinterpret_cast<void*>(direct),
           "direct/global/explicit provider selects the controlled BN ABI");
}
void lookup(void* library) {
#define CHECK(NAME) symbol(library, #NAME, &(NAME))
    CHECK(cudnnBatchNormalizationForwardTrainingEx);
    CHECK(cudnnBatchNormalizationBackwardEx);
    CHECK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize);
    CHECK(cudnnGetBatchNormalizationBackwardExWorkspaceSize);
    CHECK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize);
    CHECK(cudnnBatchNormalizationForwardTraining);
    CHECK(cudnnBatchNormalizationForwardInference);
    CHECK(cudnnBatchNormalizationBackward);
    CHECK(cudnnDeriveBNTensorDescriptor);
#undef CHECK
}
struct Descriptors {
    std::array<cudnnTensorDescriptor_t, 6> tensors{};
    cudnnActivationDescriptor_t activation{};
};
void configure(const Descriptors& d, const fixture::Case& value) {
    const auto tensors = fixture::tensors(value);
    for (std::size_t i = 0; i != tensors.size(); ++i)
        check(cudnnSetTensorNdDescriptor(
            d.tensors[i], static_cast<cudnnDataType_t>(tensors[i].type), tensors[i].dimensions,
            tensors[i].sizes.data(), tensors[i].strides.data()));
    check(cudnnSetActivationDescriptor(d.activation, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN,
                                       0));
}
Descriptors metadata() {
    Descriptors d;
    for (auto& tensor : d.tensors) check(cudnnCreateTensorDescriptor(&tensor));
    check(cudnnCreateActivationDescriptor(&d.activation));
    configure(d, {});
    expect(cudnnDeriveBNTensorDescriptor(d.tensors[5], d.tensors[0], CUDNN_BATCHNORM_SPATIAL) ==
                   CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnBatchNormalizationForwardInference(nullptr, CUDNN_BATCHNORM_SPATIAL, nullptr,
                                                       nullptr, nullptr, nullptr, nullptr, nullptr,
                                                       nullptr, nullptr, nullptr, nullptr, nullptr,
                                                       0.25) == CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnBatchNormalizationForwardTraining(
                   nullptr, CUDNN_BATCHNORM_SPATIAL, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, nullptr, nullptr, nullptr, 0.5, nullptr, nullptr, 0.25, nullptr,
                   nullptr) == CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnBatchNormalizationBackward(
                   nullptr, CUDNN_BATCHNORM_SPATIAL, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                   0.25, nullptr, nullptr) == CUDNN_STATUS_NOT_SUPPORTED &&
               !foundation::control.context_creates.load(),
           "four reference stubs remain typed guards and metadata needs no CUDA context");
    return d;
}
struct Cached {
    decltype(&cudnnBatchNormalizationForwardTrainingEx) forward{};
    decltype(&cudnnBatchNormalizationBackwardEx) backward{};
    decltype(&cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize) forward_size{};
    decltype(&cudnnGetBatchNormalizationBackwardExWorkspaceSize) backward_size{};
    decltype(&cudnnGetBatchNormalizationTrainingExReserveSpaceSize) reserve_size{};
};
std::array<cudnnTensorDescriptor_t, 6> descriptor_arguments(const Descriptors& d,
                                                            const fixture::Case& value,
                                                            CudaDnnBatchNormKind kind) {
    auto result = d.tensors;
    const auto mask = fixture::descriptor_mask(value, kind);
    for (std::size_t i = 0; i != result.size(); ++i)
        if (!mask[i]) result[i] = nullptr;
    return result;
}
cudnnStatus_t query(Cached cached, cudnnHandle_t handle, const Descriptors& d,
                    const fixture::Case& value, CudaDnnBatchNormKind kind, std::size_t* bytes) {
    const auto t = descriptor_arguments(d, value, kind);
    const auto activation = fixture::descriptor_mask(value, kind)[6] ? d.activation : nullptr;
    const auto mode = static_cast<cudnnBatchNormMode_t>(value.mode);
    const auto operation = static_cast<cudnnBatchNormOps_t>(value.operation);
    if (kind == CudaDnnBatchNormKind::kForward)
        return cached.forward_size(handle, mode, operation, t[0], t[3], t[1], t[5], activation,
                                   bytes);
    if (kind == CudaDnnBatchNormKind::kBackward)
        return cached.backward_size(handle, mode, operation, t[0], t[1], t[2], t[3], t[4], t[5],
                                    activation, bytes);
    return cached.reserve_size(handle, mode, operation, activation, t[0], bytes);
}
void queries(Cached cached, cudnnHandle_t handle, const Descriptors& d, const fixture::Case& value,
             bool detailed = false) {
    for (auto kind : {CudaDnnBatchNormKind::kForward, CudaDnnBatchNormKind::kBackward,
                      CudaDnnBatchNormKind::kReserve}) {
        std::size_t bytes = 991;
        const auto status = query(cached, handle, d, value, kind, &bytes);
        if (value.mode == 3 || value.operation == 3) {
            expect(status == CUDNN_STATUS_NOT_SUPPORTED && bytes == 991,
                   "representable unsupported BN query reaches native and preserves caller output");
            continue;
        }
        check(status);
        expect(
            bytes == (kind == CudaDnnBatchNormKind::kReserve ? fixture::required_reserve.load()
                                                             : fixture::required_workspace.load()),
            "all three native BN size results are independent");
        if (detailed) {
            fixture::override_query.store(true);
            fixture::query_value.store(SIZE_MAX - 9);
            check(query(cached, handle, d, value, kind, &bytes));
            expect(bytes == SIZE_MAX - 9, "BN public size_t result is never truncated");
            fixture::override_query.store(false);
            bytes = 177;
            fixture::fail_query.store(true);
            expect(
                query(cached, handle, d, value, kind, &bytes) == CUDNN_STATUS_NOT_SUPPORTED_SHAPE &&
                    bytes == 177,
                "native BN query errors suppress all output publication");
        }
    }
}
struct Storage {
    std::array<CUdeviceptr, 13> addresses{};
    fixture::Images images;
};
Storage allocate(const fixture::Case& value) {
    Storage storage{{}, fixture::images(value)};
    for (std::size_t i = 0; i != storage.addresses.size(); ++i) {
        check(cuMemAlloc(&storage.addresses[i], storage.images[i].size()));
        check(
            cuMemcpyHtoD(storage.addresses[i], storage.images[i].data(), storage.images[i].size()));
    }
    return storage;
}
void verify(const fixture::Case& value, const Storage& storage, const fixture::Images& expected) {
    auto actual = storage.images;
    for (std::size_t i = 0; i != actual.size(); ++i)
        check(cuMemcpyDtoH(actual[i].data(), storage.addresses[i], actual[i].size()));
    fixture::verify(value, actual, expected);
}
enum class Fault {
    kNone,
    kForwardError,
    kBackwardError,
    kPartialRunning,
    kPartialSaved,
    kShortWorkspace,
    kShortReserve,
    kNullAuxiliary,
    kZeroEnd,
    kZeroInterior
};
cudnnStatus_t invoke(Cached cached, cudnnHandle_t handle, const Descriptors& d,
                     const fixture::Case& value, const Storage& storage, CudaDnnBatchNormKind kind,
                     Fault fault) {
    const bool forward = kind == CudaDnnBatchNormKind::kForward;
    const auto t = descriptor_arguments(d, value, kind);
    const auto activation = fixture::descriptor_mask(value, kind)[6] ? d.activation : nullptr;
    const auto mask = fixture::buffer_mask(value, kind);
    std::array<void*, 13> b{};
    for (std::size_t i = 0; i != b.size(); ++i)
        if (mask[i]) b[i] = reinterpret_cast<void*>(storage.addresses[i] + 16);
    std::size_t wb = fixture::required_workspace.load(), rb = fixture::required_reserve.load();
    if (fault == Fault::kNullAuxiliary) {
        b[11] = b[12] = nullptr;
        wb = SIZE_MAX - 11;
        rb = SIZE_MAX - 13;
    } else if (fault == Fault::kZeroEnd) {
        b[11] = reinterpret_cast<void*>(storage.addresses[11] + storage.images[11].size());
        b[12] = reinterpret_cast<void*>(storage.addresses[12] + storage.images[12].size());
    } else if (fault == Fault::kShortWorkspace)
        wb = 31;
    else if (fault == Fault::kShortReserve)
        rb = 31;
    else if (fault == Fault::kPartialRunning && forward)
        b[8] = nullptr;
    else if (fault == Fault::kPartialSaved)
        b[10] = nullptr;
    const auto coefficients =
        forward ? std::array{value.forward[0], value.forward[1], 0., 0.} : value.backward;
    std::array<float, 4> narrowed{};
    std::array<const void*, 4> c{};
    for (std::size_t i = 0; i != c.size(); ++i) {
        narrowed[i] = static_cast<float>(coefficients[i]);
        c[i] = value.is_double ? static_cast<const void*>(&coefficients[i]) : &narrowed[i];
    }
    const auto mode = static_cast<cudnnBatchNormMode_t>(value.mode);
    const auto operation = static_cast<cudnnBatchNormOps_t>(value.operation);
    const auto status =
        forward
            ? cached.forward(handle, mode, operation, c[0], c[1], t[0], b[0], t[3], b[3], t[1],
                             b[1], t[5], b[5], b[6], value.factor, b[7], b[8], value.epsilon, b[9],
                             b[10], activation, b[11], wb, b[12], rb)
            : cached.backward(handle, mode, operation, c[0], c[1], c[2], c[3], t[0], b[0], t[1],
                              b[1], t[2], b[2], t[3], b[3], t[4], b[4], t[5], b[5], b[6], b[7],
                              b[8], value.epsilon, b[9], b[10], activation, b[11], wb, b[12], rb);
    std::uint32_t nulls{};
    for (std::size_t i = 0; i != b.size(); ++i)
        if (!b[i]) nulls |= 1U << i;
    expect(fixture::observed_workspace.load() == wb && fixture::observed_reserve.load() == rb &&
               fixture::observed_nulls.load() == nulls,
           "BN public path preserves every optional pointer and independent full-width count");
    return status;
}
void arithmetic(Cached cached, cudnnHandle_t handle, const Descriptors& d, fixture::Case value,
                Fault fault = Fault::kNone, bool detailed = false) {
    const bool zero =
        fault == Fault::kNullAuxiliary || fault == Fault::kZeroEnd || fault == Fault::kZeroInterior;
    fixture::required_workspace.store(zero ? 0 : 32);
    fixture::required_reserve.store(zero ? 0 : 32);
    configure(d, value);
    queries(cached, handle, d, value, detailed);
    const auto storage = allocate(value);
    auto expected = storage.images;
    for (auto kind : {CudaDnnBatchNormKind::kForward, CudaDnnBatchNormKind::kBackward}) {
        const bool forward = kind == CudaDnnBatchNormKind::kForward;
        const bool fails = fault == (forward ? Fault::kForwardError : Fault::kBackwardError);
        fixture::fail_compute.store(fails);
        const auto wanted = fails ? CUDNN_STATUS_EXECUTION_FAILED
                            : value.mode == 3 || value.operation == 3 ? CUDNN_STATUS_NOT_SUPPORTED
                            : fault == Fault::kPartialRunning || fault == Fault::kPartialSaved
                                ? CUDNN_STATUS_BAD_PARAM
                            : fault == Fault::kShortWorkspace || fault == Fault::kShortReserve
                                ? CUDNN_STATUS_INVALID_VALUE
                                : CUDNN_STATUS_SUCCESS;
        expect(invoke(cached, handle, d, value, storage, kind, fault) == wanted,
               "BN public reply reports actual original compute status");
        if (wanted == CUDNN_STATUS_SUCCESS)
            expected = forward ? fixture::oracle_forward(value, expected)
                               : fixture::oracle_backward(value, expected);
        verify(value, storage, expected);
        if (wanted != CUDNN_STATUS_SUCCESS) break;
    }
    for (const auto address : storage.addresses) check(cuMemFree(address));
}
void pending(Cached cached, cudnnHandle_t handle, const Descriptors& d, CUcontext context) {
    fixture::Case value{.operation = CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION};
    fixture::required_workspace.store(32);
    fixture::required_reserve.store(32);
    configure(d, value);
    const auto storage = allocate(value);
    check(invoke(cached, handle, d, value, storage, CudaDnnBatchNormKind::kForward, Fault::kNone));
    verify(value, storage, fixture::oracle_forward(value, storage.images));
    foundation::control.can_stop.store(false);
    fixture::fail_compute.store(true);
    expect(invoke(cached, handle, d, value, storage, CudaDnnBatchNormKind::kBackward,
                  Fault::kNone) == CUDNN_STATUS_EXECUTION_FAILED,
           "native backward error does not imply GPU completion");
    queries(cached, handle, d, value);
    auto activation = std::async(std::launch::async, [&] {
        return cudnnSetActivationDescriptor(d.activation, CUDNN_ACTIVATION_RELU,
                                            CUDNN_NOT_PROPAGATE_NAN, 0);
    });
    auto tensor =
        std::async(std::launch::async, [&] { return cudnnDestroyTensorDescriptor(d.tensors[0]); });
    auto release_handle = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto result = cudnnDestroy(handle);
        check(cuCtxSetCurrent(nullptr));
        return result;
    });
    std::array<std::future<CUresult>, 13> frees;
    for (std::size_t i = 0; i != frees.size(); ++i)
        frees[i] = std::async(std::launch::async, [&, i] {
            check(cuCtxSetCurrent(context));
            const auto result = cuMemFree(storage.addresses[i]);
            check(cuCtxSetCurrent(nullptr));
            return result;
        });
    bool blocked =
        activation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        tensor.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        release_handle.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    for (auto& release : frees)
        blocked &= release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    check(activation.get());
    check(tensor.get());
    check(release_handle.get());
    for (auto& release : frees) check(release.get());
    expect(blocked, "sixteen BN metadata/handle/buffer releases await physical completion");
}
void application(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent BN provider");
    lookup(library);
    const Cached cached{reinterpret_cast<decltype(Cached::forward)>(
                            ::dlsym(library, "cudnnBatchNormalizationForwardTrainingEx")),
                        reinterpret_cast<decltype(Cached::backward)>(
                            ::dlsym(library, "cudnnBatchNormalizationBackwardEx")),
                        reinterpret_cast<decltype(Cached::forward_size)>(::dlsym(
                            library, "cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize")),
                        reinterpret_cast<decltype(Cached::backward_size)>(
                            ::dlsym(library, "cudnnGetBatchNormalizationBackwardExWorkspaceSize")),
                        reinterpret_cast<decltype(Cached::reserve_size)>(::dlsym(
                            library, "cudnnGetBatchNormalizationTrainingExReserveSpaceSize"))};
    const auto d = metadata();
    expect(::dlclose(library) == 0, "cached BN wrappers survive explicit provider close");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cudnnHandle_t handle{};
    check(cudnnCreate(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
    for (bool is_double : {false, true}) {
        if (is_double) check(cudnnSetStream(handle, nullptr));
        for (std::uint32_t operation = 0; operation != 3; ++operation)
            arithmetic(cached, handle, d, {.is_double = is_double, .operation = operation},
                       Fault::kNone, !is_double && operation == 0);
        arithmetic(cached, handle, d,
                   {.is_double = is_double, .mode = CUDNN_BATCHNORM_PER_ACTIVATION});
    }
    arithmetic(cached, handle, d, {.mode = CUDNN_BATCHNORM_SPATIAL_PERSISTENT});
    arithmetic(cached, handle, d, {.omit_statistics = true});
    arithmetic(cached, handle, d, {.ignored_pointers = true});
    for (auto fault : {Fault::kNullAuxiliary, Fault::kZeroEnd, Fault::kZeroInterior,
                       Fault::kForwardError, Fault::kBackwardError, Fault::kPartialRunning,
                       Fault::kPartialSaved, Fault::kShortWorkspace, Fault::kShortReserve})
        arithmetic(cached, handle, d, {}, fault);
    arithmetic(cached, handle, d, {.mode = 3});
    arithmetic(cached, handle, d, {.operation = 3});
    pending(cached, handle, d, context);
    for (std::size_t i = 1; i != d.tensors.size(); ++i)
        check(cudnnDestroyTensorDescriptor(d.tensors[i]));
    check(cudnnDestroyActivationDescriptor(d.activation));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    expect(!glimmer_dnn_batch_norm_native_calls(),
           "caller-native BN provider executes no SDK function");
}
void run(const char* provider) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual BN service process identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-bn-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 512;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated original-owned BN IPC");
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start independent BN service");
    client = std::async(std::launch::async, [=] { application(provider); });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public BN progress");
    client.get();
    expect(service.request_stop() && service.finished(), "all BN owners physically settle");
    expect(
        fixture::state.forwards == 23 && fixture::state.backwards == 16 &&
            base::state.creates == 7 && base::state.destroys == 7 &&
            base::state.handle_destroys == 1 && foundation::driver.frees == 299 &&
            foundation::driver.memory.empty() && foundation::driver.events.empty() &&
            foundation::driver.streams.empty() && foundation::driver.contexts.empty() &&
            foundation::control.context_creates.load() == 2 &&
            foundation::control.context_destroys.load() == 2 &&
            foundation::control.host_registers.load() > 0 &&
            foundation::control.host_registers.load() ==
                foundation::control.host_unregisters.load() &&
            !foundation::control.context_synchronizes.load() &&
            !glimmer_dnn_batch_norm_native_calls(),
        "39 original BN calls and 299 device allocations retire exactly once without query drains");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2, "independent BN provider path required");
        run(argv[1]);
        std::cout << "BN Ex public CPU numerical, query, routing and lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#pragma GCC diagnostic pop
