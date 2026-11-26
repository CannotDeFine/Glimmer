#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_backward_convolution_fixture.h"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_backward_convolution_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_backward_convolution;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kHostileCases = 22;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t result) {
    expect(result == CUDNN_STATUS_SUCCESS, "original public backward-query SDK operation succeeds");
}
void driver_check(CUresult result) {
    expect(result == CUDA_SUCCESS, "public owned Driver setup/cleanup succeeds");
}
struct Cached {
    std::array<decltype(&cudnnGetConvolutionBackwardDataAlgorithmMaxCount), 2> counts{};
    decltype(&cudnnGetConvolutionBackwardDataWorkspaceSize) workspace{};
};
Cached lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent backward-query caller provider");
    const std::array names{"cudnnGetConvolutionBackwardDataAlgorithmMaxCount",
                           "cudnnGetConvolutionBackwardFilterAlgorithmMaxCount",
                           "cudnnGetConvolutionBackwardDataWorkspaceSize"};
    const std::array<void*, 3> direct{
        reinterpret_cast<void*>(&cudnnGetConvolutionBackwardDataAlgorithmMaxCount),
        reinterpret_cast<void*>(&cudnnGetConvolutionBackwardFilterAlgorithmMaxCount),
        reinterpret_cast<void*>(&cudnnGetConvolutionBackwardDataWorkspaceSize)};
    Cached result;
    for (std::size_t i = 0; i != 3; ++i) {
        auto* selected = ::dlsym(library, names[i]);
        expect(selected == direct[i] && ::dlsym(RTLD_DEFAULT, names[i]) == direct[i],
               "direct/global/explicit-provider routing selects exact controlled query");
        if (i < 2)
            result.counts[i] = reinterpret_cast<decltype(result.counts)::value_type>(selected);
        else
            result.workspace = reinterpret_cast<decltype(result.workspace)>(selected);
    }
    expect(::dlclose(library) == 0, "cached queries survive explicit provider close");
    return result;
}
template <typename Value>
struct Guarded {
    std::uint64_t before{0x123456789abcdef0ULL};
    Value value{71};
    std::uint64_t after{0xfedcba9876543210ULL};
    void verify(Value expected) const {
        expect(
            value == expected && before == 0x123456789abcdef0ULL && after == 0xfedcba9876543210ULL,
            "complete scalar output and both guards remain exact");
    }
};
struct Application {
    Cached cached;
    cudnnTensorDescriptor_t dy{}, dx{};
    cudnnFilterDescriptor_t w{};
    cudnnConvolutionDescriptor_t convolution{};
    CUcontext context{};
    cudnnHandle_t handle{};
    explicit Application(const char* provider) : cached(lookup(provider)) {
        check(cudnnCreateTensorDescriptor(&dy));
        check(cudnnCreateFilterDescriptor(&w));
        check(cudnnCreateConvolutionDescriptor(&convolution));
        check(cudnnCreateTensorDescriptor(&dx));
        check(cudnnSetTensor4dDescriptor(dy, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 2, 5));
        check(cudnnSetTensor4dDescriptor(dx, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 4, 6));
        check(cudnnSetFilter4dDescriptor(w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, 2, 3, 2));
        check(cudnnSetConvolution2dDescriptor(convolution, 0, 0, 1, 1, 1, 1,
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        driver_check(cuCtxCreate(&context, 0, 0));
        check(cudnnCreate(&handle));
    }
    cudnnStatus_t workspace(std::size_t* output, unsigned int algorithm = 5) const {
        return cached.workspace(handle, w, dy, convolution, dx,
                                static_cast<cudnnConvolutionBwdDataAlgo_t>(algorithm), output);
    }
    void close(bool descriptors_destroyed = false) {
        check(cudnnDestroy(handle));
        if (!descriptors_destroyed) {
            check(cudnnDestroyTensorDescriptor(dy));
            check(cudnnDestroyTensorDescriptor(dx));
            check(cudnnDestroyFilterDescriptor(w));
            check(cudnnDestroyConvolutionDescriptor(convolution));
        }
        driver_check(cuCtxDestroy(context));
    }
};
void values(Application& app) {
    Guarded<int> count;
    check(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(app.handle, &count.value));
    count.verify(11);
    check(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(app.handle, &count.value));
    count.verify(13);
    for (std::size_t direction = 0; direction != 2; ++direction) {
        for (const int value : {0, 65, INT32_MAX}) {
            fixture::control.counts[direction].store(value);
            check(app.cached.counts[direction](app.handle, &count.value));
            count.verify(value);
        }
        fixture::control.counts[direction].store(static_cast<int>(11 + 2 * direction));
        fixture::control.statuses[direction].store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
        count.value = 71;
        expect(app.cached.counts[direction](app.handle, &count.value) ==
                   CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
               "native count failure preserves detailed status despite touching private output");
        count.verify(71);
        fixture::control.statuses[direction].store(CUDNN_STATUS_SUCCESS);
        const auto calls = fixture::calls();
        expect(app.cached.counts[direction](app.handle, nullptr) == CUDNN_STATUS_BAD_PARAM &&
                   fixture::calls() == calls,
               "null count rejects before unsafe native MaxCount invocation");
    }
    Guarded<std::size_t> bytes;
    for (const auto value :
         {std::size_t{0}, (std::size_t{1} << 40) + 7, std::numeric_limits<std::size_t>::max()}) {
        fixture::control.bytes.store(value);
        check(app.workspace(&bytes.value));
        bytes.verify(value);
    }
    fixture::control.bytes.store(47);
    bytes.value = 71;
    fixture::control.statuses[2].store(CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT);
    expect(app.workspace(&bytes.value) == CUDNN_STATUS_BAD_PARAM_SIZE_INSUFFICIENT,
           "failed native size query never publishes its touched private value");
    bytes.verify(71);
    fixture::control.statuses[2].store(CUDNN_STATUS_SUCCESS);
    expect(app.workspace(nullptr) == CUDNN_STATUS_BAD_PARAM &&
               fixture::control.null_workspace.load() == 1,
           "null workspace output reaches the original callback unchanged");
    for (const auto algorithm : {6U, 7U}) {
        expect(app.workspace(&bytes.value, algorithm) == CUDNN_STATUS_NOT_SUPPORTED,
               "representable unsupported backward algorithm preserves native status");
        bytes.verify(71);
    }
    auto* inaccessible = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(inaccessible != MAP_FAILED, "create unreadable scalar output guard");
    const auto calls = fixture::calls();
    const auto invalid_count = app.cached.counts[0](app.handle, static_cast<int*>(inaccessible));
    const auto invalid_bytes = app.workspace(static_cast<std::size_t*>(inaccessible));
    expect(::munmap(inaccessible, 4096) == 0 && invalid_count == CUDNN_STATUS_BAD_PARAM &&
               invalid_bytes == CUDNN_STATUS_BAD_PARAM && fixture::calls() == calls,
           "invalid output ranges reject before remote/native call");
    expect(app.cached.workspace(app.handle, reinterpret_cast<cudnnFilterDescriptor_t>(app.dy),
                                app.dy, app.convolution, app.dx, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                &bytes.value) == CUDNN_STATUS_NOT_INITIALIZED &&
               fixture::calls() == calls,
           "wrong descriptor kind cannot become a native filter");
    bytes.verify(71);
    CUcontext foreign{};
    driver_check(cuCtxCreate(&foreign, 0, 0));
    expect(app.cached.counts[0](app.handle, &count.value) == CUDNN_STATUS_NOT_INITIALIZED &&
               fixture::calls() == calls,
           "handle cannot be queried from a different owned application context");
    driver_check(cuCtxDestroy(foreign));
    driver_check(cuCtxSetCurrent(app.context));
    const auto child = ::fork();
    expect(child >= 0, "fork inherited backward-query ownership check");
    if (!child) {
        ::_exit(app.cached.counts[0](app.handle, &count.value) == CUDNN_STATUS_NOT_INITIALIZED &&
                        app.workspace(&bytes.value) == CUDNN_STATUS_NOT_INITIALIZED &&
                        !glimmer_dnn_backward_convolution_native_calls()
                    ? 0
                    : 1);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "inherited fork invokes neither remote owner nor caller SDK");
}
void query_exclusion(Application& app) {
    fixture::control.pauses.store(true);
    fixture::control.can_finish.store(false);
    fixture::control.entered.store(false);
    auto query = std::async(std::launch::async, [&] {
        driver_check(cuCtxSetCurrent(app.context));
        Guarded<std::size_t> bytes;
        check(app.workspace(&bytes.value));
        bytes.verify(47);
        driver_check(cuCtxSetCurrent(nullptr));
    });
    const auto deadline = Clock::now() + std::chrono::seconds{3};
    while (!fixture::control.entered.load() && Clock::now() < deadline) std::this_thread::yield();
    const bool entered = fixture::control.entered.load();
    std::atomic<unsigned int> started{};
    std::array<std::future<cudnnStatus_t>, 4> mutations{
        std::async(std::launch::async,
                   [&] {
                       ++started;
                       return cudnnSetTensor4dDescriptor(app.dy, CUDNN_TENSOR_NCHW,
                                                         CUDNN_DATA_FLOAT, 1, 3, 2, 5);
                   }),
        std::async(std::launch::async,
                   [&] {
                       ++started;
                       return cudnnSetFilter4dDescriptor(app.w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                         3, 2, 3, 2);
                   }),
        std::async(std::launch::async,
                   [&] {
                       ++started;
                       return cudnnSetConvolution2dDescriptor(app.convolution, 0, 0, 1, 1, 1, 1,
                                                              CUDNN_CROSS_CORRELATION,
                                                              CUDNN_DATA_FLOAT);
                   }),
        std::async(std::launch::async, [&] {
            ++started;
            return cudnnSetTensor4dDescriptor(app.dx, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 4,
                                              6);
        })};
    auto handle = std::async(std::launch::async, [&] {
        driver_check(cuCtxSetCurrent(app.context));
        ++started;
        const auto result = cudnnSetStream(app.handle, nullptr);
        driver_check(cuCtxSetCurrent(nullptr));
        return result;
    });
    const auto start_deadline = Clock::now() + std::chrono::seconds{3};
    while (started.load() != 5 && Clock::now() < start_deadline) std::this_thread::yield();
    const bool all_started = started.load() == 5;
    bool all_waiting = query.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
    for (auto& mutation : mutations)
        all_waiting &=
            mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    all_waiting &= handle.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    fixture::control.can_finish.store(true);
    query.get();
    for (auto& mutation : mutations) check(mutation.get());
    check(handle.get());
    fixture::control.pauses.store(false);
    expect(entered && all_started && all_waiting,
           "active original query and metadata/handle transitions finish without conflicting "
           "mutation");
    // This observes combined frontend/service exclusion, not frontend pins independently
    // of the service's synchronous original-callback serialization.
}
void pending(Application& app) {
    CUstream stream{};
    driver_check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(app.handle, reinterpret_cast<cudaStream_t>(stream)));
    constexpr std::array<std::size_t, 4> sizes{192, 144, 120, 64};
    std::array<CUdeviceptr, 4> buffers{};
    for (std::size_t i = 0; i != 4; ++i) driver_check(cuMemAlloc(&buffers[i], sizes[i]));
    foundation::control.can_stop.store(false);
    foundation::control.fail_forward.store(true);
    const float alpha = 1, beta = 0;
    expect(cudnnConvolutionForward(
               app.handle, &alpha, app.dx, reinterpret_cast<void*>(buffers[0]), app.w,
               reinterpret_cast<void*>(buffers[1]), app.convolution,
               CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, reinterpret_cast<void*>(buffers[3]), 64,
               &beta, app.dy, reinterpret_cast<void*>(buffers[2])) == CUDNN_STATUS_EXECUTION_FAILED,
           "prior native error leaves an event-owned original call pending");
    const auto synchronizes = fixture::control.synchronizes.load();
    Guarded<int> count;
    Guarded<std::size_t> bytes;
    for (std::size_t i = 0; i != 2; ++i) {
        check(app.cached.counts[i](app.handle, &count.value));
        count.verify(static_cast<int>(11 + 2 * i));
    }
    check(app.workspace(&bytes.value));
    bytes.verify(47);
    expect(!foundation::control.can_stop.load() &&
               fixture::control.synchronizes.load() == synchronizes,
           "all queries complete while physical event remains NOT_READY without any drain");
    std::atomic<unsigned int> started{};
    std::array<std::future<cudnnStatus_t>, 4> descriptors{
        std::async(std::launch::async,
                   [&] {
                       ++started;
                       return cudnnDestroyTensorDescriptor(app.dy);
                   }),
        std::async(std::launch::async,
                   [&] {
                       ++started;
                       return cudnnDestroyFilterDescriptor(app.w);
                   }),
        std::async(std::launch::async,
                   [&] {
                       ++started;
                       return cudnnDestroyConvolutionDescriptor(app.convolution);
                   }),
        std::async(std::launch::async, [&] {
            ++started;
            return cudnnDestroyTensorDescriptor(app.dx);
        })};
    auto change_stream = std::async(std::launch::async, [&] {
        driver_check(cuCtxSetCurrent(app.context));
        ++started;
        const auto status = cudnnSetStream(app.handle, nullptr);
        driver_check(cuCtxSetCurrent(nullptr));
        return status;
    });
    const auto start_deadline = Clock::now() + std::chrono::seconds{3};
    while (started.load() != 5 && Clock::now() < start_deadline) std::this_thread::yield();
    const bool all_started = started.load() == 5;
    bool pending_descriptors = true;
    for (auto& result : descriptors)
        pending_descriptors &=
            result.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool pending_handle =
        change_stream.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    for (auto& result : descriptors) check(result.get());
    check(change_stream.get());
    expect(all_started && pending_descriptors && pending_handle,
           "read-only queries never release prior descriptor/handle leases");
    for (const auto buffer : buffers) driver_check(cuMemFree(buffer));
    driver_check(cuStreamDestroy(stream));
    app.close(true);
}
void missing(Application& app, unsigned int index) {
    Guarded<int> count;
    Guarded<std::size_t> bytes;
    for (unsigned int repetition = 0; repetition != 2; ++repetition) {
        const auto result = index < 2 ? app.cached.counts[index](app.handle, &count.value)
                                      : app.workspace(&bytes.value);
        expect(result == CUDNN_STATUS_NOT_SUPPORTED,
               "missing original capability is clean NOT_SUPPORTED");
        count.verify(71);
        bytes.verify(71);
        const auto other = (index + 1) % 2;
        int value{};
        check(app.cached.counts[other](app.handle, &value));
        expect(value == static_cast<int>(11 + 2 * other), "clean rejection permits another query");
    }
    expect(!fixture::control.calls[index].load(), "missing callback has no native invocation");
    app.close();
}
void poison(Application& app, unsigned int index) {
    Guarded<int> count;
    Guarded<std::size_t> bytes;
    if (index < 2)
        fixture::control.counts[index].store(-1);
    else
        fixture::control.changes_context.store(true);
    const auto result = index < 2 ? app.cached.counts[index](app.handle, &count.value)
                                  : app.workspace(&bytes.value);
    expect(result == (index < 2 ? CUDNN_STATUS_INTERNAL_ERROR : CUDNN_STATUS_NOT_INITIALIZED),
           "negative count/context loss cannot expose native SUCCESS as public success");
    count.verify(71);
    bytes.verify(71);
    const auto calls = fixture::calls();
    expect(app.cached.counts[0](app.handle, &count.value) == CUDNN_STATUS_NOT_INITIALIZED &&
               fixture::calls() == calls && !fixture::control.synchronizes.load(),
           "poisoned public handle never retries query or synchronizes");
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "authenticate local backward-query endpoint");
}
void run(const char* provider, ExecutionSchedulingMode mode, int absent = -1, int poisoned = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-backward-query-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config),
                             fixture::session_dispatch(static_cast<unsigned int>(absent + 1)));
    expect(service.start(), "start original owned query service");
    client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (poisoned >= 0)
            poison(app, static_cast<unsigned int>(poisoned));
        else if (absent >= 0)
            missing(app, static_cast<unsigned int>(absent));
        else {
            values(app);
            query_exclusion(app);
            pending(app);
        }
    });
    const auto deadline = Clock::now() + std::chrono::seconds{20};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline,
               "bounded public backward-query progress");
    client.get();
    expect(!glimmer_dnn_backward_convolution_native_calls(),
           "no caller-native SDK query or setup ever executes");
    if (poisoned >= 0) {
        expect(
            fixture::calls() == 1 && !base::state.handle_destroys &&
                !foundation::control.context_destroys.load() && !foundation::driver.frees,
            "isolated poisoned owner neither fabricates recovery nor prematurely cleans resources");
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished(), "settle original query service ownership");
    expect(base::state.creates == 2 && base::state.destroys == 2 &&
               foundation::state.creates == 2 && foundation::state.destroys == 2 &&
               base::state.handle_creates == 1 && base::state.handle_destroys == 1 &&
               foundation::driver.memory.empty() && foundation::driver.events.empty() &&
               foundation::driver.contexts.empty(),
           "known descriptors/handle/buffers/contexts retire once after actual physical proof");
}
struct Hostile {
    ExecutionWireReply reply;
    bool clean{}, disconnect{};
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    int count{71};
};
Hostile hostile(unsigned int index) {
    Hostile result;
    auto& reply = result.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    switch (index) {
        case 0:
            reply.parameter_bytes = 1;
            reply.address = 5;
            break;
        case 1:
            reply.address = UINT32_MAX;
            break;
        case 2:
            reply.address = std::uint64_t{1} << 32;
            break;
        case 3:
            reply.data = {std::byte{7}};
            break;
        case 4:
            reply.id = 7;
            break;
        case 5:
            reply.original_status.reset();
            break;
        case 6:
            reply.original_status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 7:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            break;
        case 8:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = -1;
            break;
        case 9:
        case 19:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 1;
            reply.original_status = index == 9 ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
            result.status = CUDNN_STATUS_NOT_INITIALIZED;
            break;
        case 10:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.address = 5;
            break;
        case 11:
            result.disconnect = true;
            break;
        case 12:
        case 21:
            result.clean = true;
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = index == 12 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE;
            result.status = index == 12 ? CUDNN_STATUS_NOT_SUPPORTED : CUDNN_STATUS_BAD_PARAM;
            break;
        case 13:
            result.clean = true;
            reply.status = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
            reply.original_status = reply.status;
            result.status = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
            break;
        case 14:
        case 20:
            result.clean = true;
            reply.address = index == 14 ? INT32_MAX : 0;
            result.count = index == 14 ? INT32_MAX : 0;
            result.status = CUDNN_STATUS_SUCCESS;
            break;
        case 15:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = 1024;
            break;
        case 16:
            reply.parameter_bytes = 2;
            break;
        case 17:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            reply.id = 7;
            break;
        case 18:
            reply.status = 8192;
            reply.original_status = 8192;
            break;
        default:
            throw std::runtime_error("invalid hostile backward-query case");
    }
    return result;
}
void malformed_case(const char* provider, unsigned int index) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "hostile backward-query endpoint identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-backward-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    const auto test = hostile(index);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated hostile query server");
    client = std::async(std::launch::async, [=] {
        const auto cached = lookup(provider);
        CUcontext context{};
        driver_check(cuCtxCreate(&context, 0, 0));
        cudnnHandle_t handle{};
        check(cudnnCreate(&handle));
        Guarded<int> count;
        expect(cached.counts[0](handle, &count.value) == test.status,
               "hostile/native/policy query envelope has exact public status");
        count.verify(test.count);
        if (test.clean) {
            check(cached.counts[1](handle, &count.value));
            count.verify(61);
        } else {
            expect(cached.counts[1](handle, &count.value) == CUDNN_STATUS_NOT_INITIALIZED,
                   "uncertain handle rejects subsequent queries locally");
            count.verify(test.count);
        }
        expect(!glimmer_dnn_backward_convolution_native_calls(),
               "hostile replies cannot trigger caller fallback");
    });
    unsigned int hellos{}, creates{}, queries{};
    bool disconnected{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline, "bounded hostile backward-query progress");
        if (disconnected) {
            std::this_thread::yield();
            continue;
        }
        expect(server.poll_for(1), "poll hostile authenticated transport");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity,
                   "hostile fixture preserves actual peer authentication");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                expect(decode_execution_context_bootstrap(request->message.payload) ==
                           ExecutionContextBootstrap{ExecutionContextPurpose::kExplicitUser, 0, 1},
                       "ordinary typed context is exact and cannot carry a DNM1 role");
                ++hellos;
                const auto ownership =
                    encode_execution_bootstrap_ownership(ExecutionBootstrapOwnership::kRetained);
                expect(ownership.has_value(), "encode actual context ownership protocol");
                reply.data.assign(ownership->begin(), ownership->end());
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnn) {
                expect(input.empty(), "empty original handle-creation request");
                ++creates;
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = 19;
            } else {
                ExecutionDnnBackwardConvolutionQuery query;
                expect(request->message.header.operation ==
                               ExecutionOpcode::kDnnBackwardConvolutionQuery &&
                           request->message.payload.size() == 60 &&
                           read_execution_dnn_backward_convolution_query(input, query) &&
                           input.empty() && query.handle == 19 && query.has_output == 1 &&
                           !query.metadata && !query.algorithm &&
                           query.descriptors == std::array<std::uint64_t, 4>{},
                       "count request contains only exact owned handle authority");
                ++queries;
                if (queries == 1) {
                    expect(query.kind == ExecutionDnnBackwardConvolutionQueryKind::kDataMaxCount,
                           "first direction reaches correct opcode");
                    if (test.disconnect) {
                        server.stop();
                        disconnected = true;
                        continue;
                    }
                    reply = test.reply;
                } else {
                    expect(
                        test.clean && queries == 2 &&
                            query.kind == ExecutionDnnBackwardConvolutionQueryKind::kFilterMaxCount,
                        "only clean receipt permits other-direction follow-up");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    reply.address = 61;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded.has_value(), "encode hostile backward-query reply");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *encoded}), "deliver hostile reply");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && creates == 1 && queries == (test.clean ? 2U : 1U),
           "no extra bootstrap, retry, descriptor call or hidden query follows quarantine");
    server.stop();
}
void isolated(const char* executable, const char* provider, std::string_view mode) {
    const auto count = mode == "--malformed-replies" ? kHostileCases : 3U;
    const char* selected = mode == "--malformed-replies" ? "--malformed-case"
                           : mode == "--poison"          ? "--poison-case"
                                                         : "--missing-case";
    for (unsigned int i = 0; i != count; ++i) {
        const auto index = std::to_string(i);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated public backward-query case");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider, selected, index.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error(std::string(selected) + " failed: " + index);
    }
}
unsigned int index_argument(const char* text) {
    const std::string_view value{text};
    expect(!value.empty() && value.size() <= 2, "bounded isolated query index");
    unsigned int result{};
    for (const char digit : value) {
        expect(digit >= '0' && digit <= '9', "decimal fixture index");
        result = 10 * result + static_cast<unsigned int>(digit - '0');
    }
    return result;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider path and optional query mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--missing-callback" || mode == "--poison" || mode == "--malformed-replies") {
            expect(argc == 3, "isolated parent mode has no extra argument");
            isolated(argv[0], argv[1], mode);
        } else if (mode == "--missing-case" || mode == "--poison-case" ||
                   mode == "--malformed-case") {
            expect(argc == 4, "isolated case needs index");
            const auto index = index_argument(argv[3]);
            if (mode == "--malformed-case") {
                expect(index < kHostileCases, "known malformed backward-query case");
                malformed_case(argv[1], index);
                ::_exit(0);
            }
            expect(index < 3, "known missing/poison query case");
            run(argv[1], ExecutionSchedulingMode::kNaive,
                mode == "--missing-case" ? static_cast<int>(index) : -1,
                mode == "--poison-case" ? static_cast<int>(index) : -1);
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known scheduling mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "Backward-convolution original query CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        fixture::control.can_finish.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
