#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/dnn_pooling_fixture.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <source_location>
#include <dlfcn.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_pooling_native_calls();
namespace {
using namespace glimmer::app;
namespace fixture = glimmer::tests::dnn_pooling;
namespace foundation = fixture::foundation;
namespace base = glimmer::tests::dnn;
using Clock = std::chrono::steady_clock;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public pooling SDK status " + std::to_string(status) +
                                 " at line " + std::to_string(where.line()));
}
void check(CUresult status, std::source_location where = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public pooling Driver status " + std::to_string(status) +
                                 " at line " + std::to_string(where.line()));
}
template <typename Function>
void symbol(void* library, const char* name, Function direct) {
    expect(::dlsym(library, name) == reinterpret_cast<void*>(direct) &&
               ::dlsym(RTLD_DEFAULT, name) == reinterpret_cast<void*>(direct),
           "direct/global/explicit provider lookup all select the controlled pooling ABI");
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
void lookup(void* library) {
#define CHECK(NAME) symbol(library, #NAME, &(NAME))
    CHECK(cudnnCreatePoolingDescriptor);
    CHECK(cudnnDestroyPoolingDescriptor);
    CHECK(cudnnSetPoolingNdDescriptor);
    CHECK(cudnnGetPoolingNdDescriptor);
    CHECK(cudnnGetPoolingNdForwardOutputDim);
    CHECK(cudnnPoolingForward);
    CHECK(cudnnSoftmaxForward);
    CHECK(cudnnSetPooling2dDescriptor);
    CHECK(cudnnGetPooling2dDescriptor);
    CHECK(cudnnGetPooling2dForwardOutputDim);
    CHECK(cudnnPoolingBackward);
    CHECK(cudnnSoftmaxBackward);
#undef CHECK
}
struct Descriptors {
    cudnnPoolingDescriptor_t pooling{};
    cudnnTensorDescriptor_t x{}, y{};
};
void configure(const Descriptors& d, const fixture::Case& value) {
    check(cudnnSetTensorNdDescriptor(d.x, static_cast<cudnnDataType_t>(value.x.type),
                                     value.x.dimensions, value.x.sizes.data(),
                                     value.x.strides.data()));
    check(cudnnSetTensorNdDescriptor(d.y, static_cast<cudnnDataType_t>(value.y.type),
                                     value.y.dimensions, value.y.sizes.data(),
                                     value.y.strides.data()));
    if (value.is_pooling)
        check(cudnnSetPoolingNdDescriptor(
            d.pooling, static_cast<cudnnPoolingMode_t>(value.pooling.mode),
            static_cast<cudnnNanPropagation_t>(value.pooling.nan), value.pooling.dimensions,
            value.pooling.window.data(), value.pooling.padding.data(),
            value.pooling.strides.data()));
}
Descriptors metadata() {
    Descriptors d;
    check(cudnnCreatePoolingDescriptor(&d.pooling));
    check(cudnnCreateTensorDescriptor(&d.x));
    check(cudnnCreateTensorDescriptor(&d.y));
    configure(d, fixture::pool_case(CUDNN_POOLING_MAX));
    std::array<int, 8> window, padding, strides;
    window.fill(91);
    padding.fill(92);
    strides.fill(93);
    int rank{};
    cudnnPoolingMode_t mode{};
    cudnnNanPropagation_t nan{};
    check(cudnnGetPoolingNdDescriptor(d.pooling, 8, &mode, &nan, &rank, window.data(),
                                      padding.data(), strides.data()));
    expect(rank == 2 && mode == CUDNN_POOLING_MAX && nan == CUDNN_PROPAGATE_NAN && window[0] == 2 &&
               window[1] == 3 && window[2] == 91 && padding[0] == 1 && padding[1] == 1 &&
               padding[2] == 92 && strides[0] == 1 && strides[1] == 2 && strides[2] == 93,
           "pooling getter publishes only the original actual-rank prefix");
    for (const auto requested : {-1, 9}) {
        rank = 991;
        mode = CUDNN_POOLING_MAX_DETERMINISTIC;
        nan = CUDNN_NOT_PROPAGATE_NAN;
        window.fill(91);
        padding.fill(92);
        strides.fill(93);
        expect(cudnnGetPoolingNdDescriptor(d.pooling, requested, &mode, &nan, &rank, window.data(),
                                           padding.data(), strides.data()) ==
                       (requested < 0 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_NOT_SUPPORTED) &&
                   rank == 991 && mode == CUDNN_POOLING_MAX_DETERMINISTIC &&
                   nan == CUDNN_NOT_PROPAGATE_NAN && window[0] == 91 && padding[0] == 92 &&
                   strides[0] == 93,
               "invalid pooling request preserves native error and every output family");
    }
    foundation::control.can_stop.store(true);
    fixture::control.fail_get.store(true);
    expect(cudnnGetPoolingNdDescriptor(d.pooling, 2, &mode, &nan, &rank, window.data(),
                                       padding.data(), strides.data()) == CUDNN_STATUS_BAD_PARAM &&
               rank == 991 && mode == CUDNN_POOLING_MAX_DETERMINISTIC,
           "detailed native getter failure does not publish touched scalar outputs");
    std::array<int, 5> shape{0, 0, 0, 0, 991};
    check(cudnnGetPoolingNdForwardOutputDim(d.pooling, d.x, 4, shape.data()));
    expect(shape == std::array<int, 5>{1, 1, 4, 2, 991}, "exact original pooling output shape");
    shape.fill(991);
    expect(cudnnGetPoolingNdForwardOutputDim(d.pooling, d.x, 5, shape.data()) ==
                   CUDNN_STATUS_BAD_PARAM &&
               std::ranges::all_of(shape, [](int value) { return value == 991; }),
           "pooling output rank is an exact native request, not a clamped capacity");
    configure(d, fixture::pool_case(CUDNN_POOLING_MAX, false, true));
    check(cudnnGetPoolingNdForwardOutputDim(d.pooling, d.x, 4, shape.data()));
    expect(std::equal(shape.begin(), shape.begin() + 4, std::array{1, 1, 1, 2}.begin()),
           "negative padding preserves original cropping output semantics");
    configure(d, fixture::pool_case(CUDNN_POOLING_MAX));
    expect(!foundation::control.context_creates.load(),
           "all pooling metadata routes precede CUDA context initialization");
    expect(cudnnSetPooling2dDescriptor(d.pooling, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, 3, 1,
                                       1, 1, 2) == CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnGetPooling2dDescriptor(d.pooling, &mode, &nan, &rank, &rank, &rank, &rank,
                                           &rank, &rank) == CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnGetPooling2dForwardOutputDim(d.pooling, d.x, &rank, &rank, &rank, &rank) ==
                   CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnPoolingBackward(nullptr, d.pooling, nullptr, d.y, nullptr, d.y, nullptr, d.x,
                                    nullptr, nullptr, d.x, nullptr) == CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnSoftmaxBackward(nullptr, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                                    nullptr, d.y, nullptr, d.y, nullptr, nullptr, d.x,
                                    nullptr) == CUDNN_STATUS_NOT_SUPPORTED,
           "adjacent unimplemented reference routes remain typed rejection guards only");
    return d;
}
struct Cached {
    decltype(&cudnnPoolingForward) pooling{};
    decltype(&cudnnSoftmaxForward) softmax{};
};
template <typename Scalar>
void arithmetic(Cached cached, cudnnHandle_t handle, const Descriptors& d, fixture::Case value,
                bool fails = false) {
    value.x.type = value.y.type = sizeof(Scalar) == 8 ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
    configure(d, value);
    const auto input = fixture::image<Scalar>(value.x, value.input);
    const auto initial = fixture::image<Scalar>(value.y, value.initial);
    CUdeviceptr x{}, y{};
    check(cuMemAlloc(&x, input.size()));
    check(cuMemcpyHtoD(x, input.data(), input.size()));
    if (value.is_inplace) {
        y = x;
    } else {
        check(cuMemAlloc(&y, initial.size()));
        check(cuMemcpyHtoD(y, initial.data(), initial.size()));
    }
    const Scalar alpha{1.25}, beta{-0.5};
    fixture::control.fail_compute.store(fails);
    const auto status =
        value.is_pooling
            ? cached.pooling(handle, d.pooling, &alpha, d.x, reinterpret_cast<void*>(x + 16), &beta,
                             d.y, reinterpret_cast<void*>(y + 16))
            : cached.softmax(handle, static_cast<cudnnSoftmaxAlgorithm_t>(value.algorithm),
                             static_cast<cudnnSoftmaxMode_t>(value.mode), &alpha, d.x,
                             reinterpret_cast<void*>(x + 16), &beta, d.y,
                             reinterpret_cast<void*>(y + 16));
    expect(status == (fails                  ? CUDNN_STATUS_EXECUTION_FAILED
                      : value.algorithm == 3 ? CUDNN_STATUS_BAD_PARAM
                                             : CUDNN_STATUS_SUCCESS),
           "public arithmetic preserves actual native status, including algorithm 3");
    std::vector<std::byte> actual(initial.size());
    check(cuMemcpyDtoH(actual.data(), y, actual.size()));
    fixture::verify<Scalar>(value, actual, value.is_inplace ? input : initial,
                            fails || value.algorithm == 3);
    if (!value.is_inplace) {
        std::vector<std::byte> unchanged(input.size());
        check(cuMemcpyDtoH(unchanged.data(), x, unchanged.size()));
        expect(unchanged == input, "public out-of-place input and every stride hole remain intact");
    }
    check(cuMemFree(x));
    if (!value.is_inplace) check(cuMemFree(y));
}
template <typename Scalar>
void matrix(Cached cached, cudnnHandle_t handle, const Descriptors& d) {
    for (std::uint32_t mode = 0; mode != 4; ++mode)
        arithmetic<Scalar>(cached, handle, d, fixture::pool_case(mode));
    arithmetic<Scalar>(cached, handle, d,
                       fixture::pool_case(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, true));
    arithmetic<Scalar>(cached, handle, d, fixture::pool_case(CUDNN_POOLING_MAX, false, true));
    arithmetic<Scalar>(cached, handle, d,
                       fixture::pool_case(CUDNN_POOLING_MAX, false, false, true));
    for (std::uint32_t algorithm = 0; algorithm != 3; ++algorithm)
        for (std::uint32_t mode = 0; mode != 2; ++mode)
            arithmetic<Scalar>(cached, handle, d, fixture::softmax_case(algorithm, mode));
    arithmetic<Scalar>(cached, handle, d, fixture::softmax_case(1, 0, true));
    arithmetic<Scalar>(cached, handle, d, fixture::softmax_case(2, 1, true));
}
void pending_error(Cached cached, cudnnHandle_t handle, const Descriptors& d, CUcontext context) {
    const auto value = fixture::pool_case(CUDNN_POOLING_MAX);
    configure(d, value);
    const auto input = fixture::image<float>(value.x, value.input);
    const auto initial = fixture::image<float>(value.y, value.initial);
    CUdeviceptr x{}, y{};
    check(cuMemAlloc(&x, input.size()));
    check(cuMemAlloc(&y, initial.size()));
    check(cuMemcpyHtoD(x, input.data(), input.size()));
    check(cuMemcpyHtoD(y, initial.data(), initial.size()));
    const float alpha = 1.25F, beta = -0.5F;
    foundation::control.can_stop.store(false);
    fixture::control.fail_compute.store(true);
    expect(cached.pooling(handle, d.pooling, &alpha, d.x, reinterpret_cast<void*>(x + 16), &beta,
                          d.y, reinterpret_cast<void*>(y + 16)) == CUDNN_STATUS_EXECUTION_FAILED,
           "native pooling error returns before independent event completion");
    auto mutation = std::async(std::launch::async, [&] {
        return cudnnSetPoolingNdDescriptor(
            d.pooling, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 2, value.pooling.window.data(),
            value.pooling.padding.data(), value.pooling.strides.data());
    });
    auto descriptor_release =
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
        descriptor_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        handle_release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    for (auto& release : releases)
        pending &= release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    check(mutation.get());
    check(descriptor_release.get());
    check(handle_release.get());
    for (auto& release : releases) check(release.get());
    expect(pending, "error reply retains pooling mutation, tensor, handle and both allocations");
    expect(cudnnDestroyTensorDescriptor(d.y) == CUDNN_STATUS_NOT_INITIALIZED,
           "stale tensor identity cannot destroy native metadata twice");
}
void application(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent pooling/softmax caller provider");
    lookup(library);
    const Cached cached{
        reinterpret_cast<decltype(&cudnnPoolingForward)>(::dlsym(library, "cudnnPoolingForward")),
        reinterpret_cast<decltype(&cudnnSoftmaxForward)>(::dlsym(library, "cudnnSoftmaxForward"))};
    const auto descriptors = metadata();
    expect(::dlclose(library) == 0, "close provider while cached controlled entries remain valid");
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
    arithmetic<float>(cached, handle, descriptors, fixture::pool_case(CUDNN_POOLING_MAX), true);
    arithmetic<double>(cached, handle, descriptors, fixture::softmax_case(1, 1), true);
    arithmetic<float>(cached, handle, descriptors, fixture::softmax_case(3, 0));
    pending_error(cached, handle, descriptors, context);
    check(cudnnDestroyTensorDescriptor(descriptors.x));
    check(cudnnDestroyPoolingDescriptor(descriptors.pooling));
    expect(cudnnDestroyPoolingDescriptor(descriptors.pooling) == CUDNN_STATUS_NOT_INITIALIZED,
           "stale pooling descriptor is rejected locally");
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    expect(glimmer_dnn_pooling_native_calls() == 0,
           "caller-provider SDK invocation count stays zero");
}
#pragma GCC diagnostic pop
void run(const char* provider) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual pooling service process identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-pooling-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 64;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure genuine authenticated pooling IPC");
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start independent original-owned pooling service");
    client = std::async(std::launch::async, [=] { application(provider); });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded pooling public progress");
    client.get();
    expect(service.request_stop() && service.finished(), "all pooling service owners settle");
    expect(
        fixture::state.pools == 16 && fixture::state.softmaxes == 18 &&
            fixture::state.creates == 1 && fixture::state.destroys == 1 &&
            base::state.creates == 2 && base::state.destroys == 2 &&
            base::state.handle_creates == 1 && base::state.handle_destroys == 1 &&
            foundation::driver.frees == 64 && foundation::driver.memory.empty() &&
            foundation::driver.events.empty() && foundation::driver.streams.empty() &&
            foundation::driver.contexts.empty() &&
            foundation::control.context_creates.load() == 2 &&
            foundation::control.context_destroys.load() == 2 &&
            foundation::control.host_registers.load() &&
            foundation::control.host_registers.load() ==
                foundation::control.host_unregisters.load() &&
            glimmer_dnn_pooling_native_calls() == 0,
        "service executes every CPU original and reclaims native/staging resources exactly once");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2, "independent pooling provider path");
        run(argv[1]);
        std::cout << "Pooling/softmax public CPU numerical and IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
