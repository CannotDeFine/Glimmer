#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/dnn_lrn_fixture.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <source_location>
#include <dlfcn.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_lrn_native_calls();
namespace {
using namespace glimmer::app;
namespace fixture = glimmer::tests::dnn_lrn;
namespace foundation = fixture::foundation;
namespace base = glimmer::tests::dnn;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public LRN SDK status " + std::to_string(status) + " at line " +
                                 std::to_string(where.line()));
}
void check(CUresult status, std::source_location where = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public LRN Driver status " + std::to_string(status) +
                                 " at line " + std::to_string(where.line()));
}
template <typename Function>
void symbol(void* library, const char* name, Function direct) {
    expect(::dlsym(library, name) == reinterpret_cast<void*>(direct) &&
               ::dlsym(RTLD_DEFAULT, name) == reinterpret_cast<void*>(direct),
           "direct/global/explicit provider selects the controlled LRN/transform ABI");
}
void lookup(void* library) {
#define CHECK(NAME) symbol(library, #NAME, &(NAME))
    CHECK(cudnnCreateLRNDescriptor);
    CHECK(cudnnDestroyLRNDescriptor);
    CHECK(cudnnSetLRNDescriptor);
    CHECK(cudnnLRNCrossChannelForward);
    CHECK(cudnnTransformTensor);
    CHECK(cudnnGetLRNDescriptor);
    CHECK(cudnnLRNCrossChannelBackward);
    CHECK(cudnnDivisiveNormalizationForward);
    CHECK(cudnnDivisiveNormalizationBackward);
#undef CHECK
}
struct Descriptors {
    cudnnTensorDescriptor_t x{}, y{};
};
void configure(const Descriptors& d, const fixture::Case& value) {
    check(cudnnSetTensorNdDescriptor(d.x, static_cast<cudnnDataType_t>(value.x.type),
                                     value.x.dimensions, value.x.sizes.data(),
                                     value.x.strides.data()));
    check(cudnnSetTensorNdDescriptor(d.y, static_cast<cudnnDataType_t>(value.y.type),
                                     value.y.dimensions, value.y.sizes.data(),
                                     value.y.strides.data()));
}
void set_lrn(cudnnLRNDescriptor_t descriptor, const fixture::Case& value) {
    const auto& p = value.parameters;
    check(cudnnSetLRNDescriptor(descriptor, p.n, p.alpha, p.beta, p.k));
}
Descriptors metadata() {
    Descriptors d;
    check(cudnnCreateTensorDescriptor(&d.x));
    check(cudnnCreateTensorDescriptor(&d.y));
    cudnnLRNDescriptor_t descriptor{};
    check(cudnnCreateLRNDescriptor(&descriptor));
    set_lrn(descriptor, fixture::lrn_case(3));
    unsigned int n = 991;
    double alpha = 11, beta = 12, k = 13;
    expect(
        cudnnGetLRNDescriptor(descriptor, &n, &alpha, &beta, &k) == CUDNN_STATUS_NOT_SUPPORTED &&
            n == 991 && alpha == 11 && beta == 12 && k == 13 &&
            cudnnLRNCrossChannelBackward(nullptr, descriptor, CUDNN_LRN_CROSS_CHANNEL_DIM1, nullptr,
                                         d.y, nullptr, d.y, nullptr, d.x, nullptr, nullptr, d.x,
                                         nullptr) == CUDNN_STATUS_NOT_SUPPORTED &&
            cudnnDivisiveNormalizationForward(
                nullptr, descriptor, CUDNN_DIVNORM_PRECOMPUTED_MEANS, nullptr, d.x, nullptr,
                nullptr, nullptr, nullptr, nullptr, d.y, nullptr) == CUDNN_STATUS_NOT_SUPPORTED &&
            cudnnDivisiveNormalizationBackward(nullptr, descriptor, CUDNN_DIVNORM_PRECOMPUTED_MEANS,
                                               nullptr, d.x, nullptr, nullptr, nullptr, nullptr,
                                               nullptr, nullptr, d.y, nullptr,
                                               nullptr) == CUDNN_STATUS_NOT_SUPPORTED,
        "four reference stubs remain typed guards and never modify caller outputs");
    check(cudnnDestroyLRNDescriptor(descriptor));
    expect(cudnnDestroyLRNDescriptor(descriptor) == CUDNN_STATUS_NOT_INITIALIZED &&
               !foundation::control.context_creates.load(),
           "LRN create/set/destroy is context independent with stale-handle exclusion");
    return d;
}
struct Cached {
    decltype(&cudnnLRNCrossChannelForward) lrn{};
    decltype(&cudnnTransformTensor) transform{};
};
template <typename Scalar>
void arithmetic(Cached cached, cudnnHandle_t handle, const Descriptors& d, fixture::Case value,
                bool fails = false) {
    value.x.type = value.y.type = sizeof(Scalar) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    configure(d, value);
    cudnnLRNDescriptor_t descriptor{};
    if (!value.is_transform) {
        check(cudnnCreateLRNDescriptor(&descriptor));
        if (!value.use_defaults) set_lrn(descriptor, value);
    }
    const auto input = fixture::image<Scalar>(value, false);
    const auto initial = fixture::image<Scalar>(value, true);
    CUdeviceptr x{}, y{};
    check(cuMemAlloc(&x, input.size() + (value.shares_allocation ? initial.size() : 0)));
    if (value.shares_allocation)
        y = x + input.size();
    else
        check(cuMemAlloc(&y, initial.size()));
    check(cuMemcpyHtoD(x, input.data(), input.size()));
    check(cuMemcpyHtoD(y, initial.data(), initial.size()));
    const Scalar alpha{1.5}, beta{-0.25};
    if (value.is_transform && value.shares_allocation) {
        expect(cached.transform(handle, &alpha, d.x, reinterpret_cast<void*>(x + 16), &beta, d.y,
                                reinterpret_cast<void*>(x + 16)) == CUDNN_STATUS_BAD_PARAM &&
                   cached.transform(handle, &alpha, d.x, reinterpret_cast<void*>(x + 16), &beta,
                                    d.y, reinterpret_cast<void*>(x + 24)) == CUDNN_STATUS_BAD_PARAM,
               "exact and partial overlap reject before original transform, despite owned range");
    }
    fixture::fail_compute.store(fails);
    const auto status =
        value.is_transform ? cached.transform(handle, &alpha, d.x, reinterpret_cast<void*>(x + 16),
                                              &beta, d.y, reinterpret_cast<void*>(y + 16))
                           : cached.lrn(handle, descriptor, static_cast<cudnnLRNMode_t>(value.mode),
                                        &alpha, d.x, reinterpret_cast<void*>(x + 16), &beta, d.y,
                                        reinterpret_cast<void*>(y + 16));
    expect(status == (fails        ? CUDNN_STATUS_EXECUTION_FAILED
                      : value.mode ? CUDNN_STATUS_BAD_PARAM
                                   : CUDNN_STATUS_SUCCESS),
           "public LRN/transform preserves actual original status rather than admission success");
    std::vector<std::byte> actual(initial.size()), unchanged(input.size());
    check(cuMemcpyDtoH(actual.data(), y, actual.size()));
    check(cuMemcpyDtoH(unchanged.data(), x, unchanged.size()));
    expect(unchanged == input, "full public source and stride holes remain unchanged");
    fixture::verify<Scalar>(value, actual, initial, fails || value.mode);
    check(cuMemFree(x));
    if (!value.shares_allocation) check(cuMemFree(y));
    if (descriptor) check(cudnnDestroyLRNDescriptor(descriptor));
}
template <typename Scalar>
void matrix(Cached cached, cudnnHandle_t handle, const Descriptors& d) {
    for (const auto n : {5U, 3U, 4U}) arithmetic<Scalar>(cached, handle, d, fixture::lrn_case(n));
    arithmetic<Scalar>(cached, handle, d, fixture::transform_case(false));
    arithmetic<Scalar>(cached, handle, d, fixture::transform_case(true));
}
void pending_error(Cached cached, cudnnHandle_t handle, const Descriptors& d, CUcontext context) {
    auto value = fixture::lrn_case(4);
    configure(d, value);
    cudnnLRNDescriptor_t descriptor{};
    check(cudnnCreateLRNDescriptor(&descriptor));
    set_lrn(descriptor, value);
    const auto input = fixture::image<float>(value, false);
    const auto initial = fixture::image<float>(value, true);
    CUdeviceptr x{}, y{};
    check(cuMemAlloc(&x, input.size()));
    check(cuMemAlloc(&y, initial.size()));
    check(cuMemcpyHtoD(x, input.data(), input.size()));
    check(cuMemcpyHtoD(y, initial.data(), initial.size()));
    const float alpha = 1.5F, beta = -0.25F;
    foundation::control.can_stop.store(false);
    fixture::fail_compute.store(true);
    expect(cached.lrn(handle, descriptor, CUDNN_LRN_CROSS_CHANNEL_DIM1, &alpha, d.x,
                      reinterpret_cast<void*>(x + 16), &beta, d.y,
                      reinterpret_cast<void*>(y + 16)) == CUDNN_STATUS_EXECUTION_FAILED,
           "native LRN error returns while original event still lacks completion");
    auto mutation = std::async(std::launch::async, [&] {
        const auto& p = value.parameters;
        return cudnnSetLRNDescriptor(descriptor, p.n, p.alpha, p.beta, p.k);
    });
    auto tensor_release =
        std::async(std::launch::async, [&] { return cudnnDestroyTensorDescriptor(d.y); });
    auto handle_release = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto status = cudnnDestroy(handle);
        check(cuCtxSetCurrent(nullptr));
        return status;
    });
    std::array releases{std::async(std::launch::async,
                                   [&] {
                                       check(cuCtxSetCurrent(context));
                                       const auto status = cuMemFree(x);
                                       check(cuCtxSetCurrent(nullptr));
                                       return status;
                                   }),
                        std::async(std::launch::async, [&] {
                            check(cuCtxSetCurrent(context));
                            const auto status = cuMemFree(y);
                            check(cuCtxSetCurrent(nullptr));
                            return status;
                        })};
    bool pending =
        mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        tensor_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        handle_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    for (auto& release : releases)
        pending &= release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    check(mutation.get());
    check(tensor_release.get());
    check(handle_release.get());
    for (auto& release : releases) check(release.get());
    expect(pending,
           "LRN error retains immutable operation/tensors, handle and both allocation owners");
    check(cudnnDestroyLRNDescriptor(descriptor));
}
void application(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent LRN caller provider");
    lookup(library);
    const Cached cached{reinterpret_cast<decltype(&cudnnLRNCrossChannelForward)>(
                            ::dlsym(library, "cudnnLRNCrossChannelForward")),
                        reinterpret_cast<decltype(&cudnnTransformTensor)>(
                            ::dlsym(library, "cudnnTransformTensor"))};
    const auto descriptors = metadata();
    expect(::dlclose(library) == 0, "cached wrappers outlive explicit provider handle");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cudnnHandle_t handle{};
    check(cudnnCreate(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
    matrix<float>(cached, handle, descriptors);
    check(cudnnSetStream(handle, nullptr));
    matrix<double>(cached, handle, descriptors);
    arithmetic<float>(cached, handle, descriptors, fixture::lrn_case(3), true);
    arithmetic<double>(cached, handle, descriptors, fixture::transform_case(true), true);
    auto bad_mode = fixture::lrn_case();
    bad_mode.mode = 1;
    arithmetic<float>(cached, handle, descriptors, bad_mode);
    pending_error(cached, handle, descriptors, context);
    check(cudnnDestroyTensorDescriptor(descriptors.x));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    expect(glimmer_dnn_lrn_native_calls() == 0, "caller-provider never executes a native LRN call");
}
void run(const char* provider) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual LRN service process identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-lrn-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 64;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated original-owned LRN IPC");
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start independent LRN service");
    client = std::async(std::launch::async, [=] { application(provider); });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public LRN progress");
    client.get();
    expect(service.request_stop() && service.finished(), "all original LRN owners settle");
    expect(fixture::state.lrns == 9 && fixture::state.transforms == 5 &&
               fixture::state.creates == 10 && fixture::state.destroys == 10 &&
               fixture::state.sets == 8 && base::state.creates == 2 && base::state.destroys == 2 &&
               base::state.handle_creates == 1 && base::state.handle_destroys == 1 &&
               foundation::driver.frees == 25 && foundation::driver.memory.empty() &&
               foundation::driver.streams.empty() && foundation::driver.events.empty() &&
               foundation::driver.contexts.empty() &&
               foundation::control.context_creates.load() == 2 &&
               foundation::control.context_destroys.load() == 2 &&
               foundation::control.host_registers.load() &&
               foundation::control.host_registers.load() ==
                   foundation::control.host_unregisters.load() &&
               glimmer_dnn_lrn_native_calls() == 0,
           "exact original computations/default-preserving setters and once-only cleanup");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2, "independent LRN provider path");
        run(argv[1]);
        std::cout << "LRN/transform public CPU numerical and ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
