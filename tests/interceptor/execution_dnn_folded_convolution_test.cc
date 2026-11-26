#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_folded_convolution_fixture.h"

#include <cerrno>
#include <future>
#include <iostream>
#include <dlfcn.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_folded_convolution_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_folded_convolution;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
using Clock = std::chrono::steady_clock;
using Fold = decltype(&cudnnGetFoldedConvBackwardDataDescriptors);
constexpr unsigned int kHostileCases = 24;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t result) {
    expect(result == CUDNN_STATUS_SUCCESS, "public original metadata operation succeeds");
}
void driver(CUresult result) {
    expect(result == CUDA_SUCCESS, "public owned Driver operation succeeds");
}
Fold lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent Folded caller provider");
    auto* address = ::dlsym(library, "cudnnGetFoldedConvBackwardDataDescriptors");
    expect(address == reinterpret_cast<void*>(&cudnnGetFoldedConvBackwardDataDescriptors) &&
               ::dlsym(RTLD_DEFAULT, "cudnnGetFoldedConvBackwardDataDescriptors") == address,
           "direct/global/explicit provider selects the controlled Folded wrapper");
    expect(::dlclose(library) == 0, "cached wrapper survives explicit provider close");
    return reinterpret_cast<Fold>(address);
}
struct Set {
    cudnnFilterDescriptor_t filter{};
    cudnnTensorDescriptor_t diff{};
    cudnnConvolutionDescriptor_t convolution{};
    cudnnTensorDescriptor_t grad{};
};
struct Application {
    Fold folded;
    Set input, output;
    std::array<cudnnTensorTransformDescriptor_t, 4> transforms{};
    CUcontext context{};
    cudnnHandle_t handle{};
    explicit Application(const char* provider, bool configure = true) : folded(lookup(provider)) {
        for (auto* group : {&input, &output}) {
            check(cudnnCreateFilterDescriptor(&group->filter));
            check(cudnnCreateTensorDescriptor(&group->diff));
            check(cudnnCreateConvolutionDescriptor(&group->convolution));
            check(cudnnCreateTensorDescriptor(&group->grad));
        }
        for (auto& value : transforms) check(cudnnCreateTensorTransformDescriptor(&value));
        if (configure) reset_inputs();
        driver(cuCtxCreate(&context, 0, 0));
        check(cudnnCreate(&handle));
    }
    void reset_inputs() {
        check(cudnnSetFilter4dDescriptor(input.filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, 2, 3,
                                         2));
        check(cudnnSetTensor4dDescriptor(input.diff, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 2,
                                         5));
        check(cudnnSetConvolution2dDescriptor(input.convolution, 0, 0, 1, 1, 1, 1,
                                              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        check(cudnnSetTensor4dDescriptor(input.grad, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 4,
                                         6));
    }
    cudnnStatus_t call(const Set& source, const Set& destination,
                       std::array<cudnnTensorTransformDescriptor_t, 4> conversions,
                       cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) const {
        return folded(handle, source.filter, source.diff, source.convolution, source.grad, format,
                      destination.filter, destination.diff, destination.convolution,
                      destination.grad, conversions[0], conversions[1], conversions[2],
                      conversions[3]);
    }
    cudnnStatus_t call() const {
        return call(input, output, transforms);
    }
    void close() {
        check(cudnnDestroy(handle));
        for (auto* group : {&input, &output}) {
            check(cudnnDestroyFilterDescriptor(group->filter));
            check(cudnnDestroyTensorDescriptor(group->diff));
            check(cudnnDestroyConvolutionDescriptor(group->convolution));
            check(cudnnDestroyTensorDescriptor(group->grad));
        }
        for (const auto value : transforms) check(cudnnDestroyTensorTransformDescriptor(value));
        driver(cuCtxDestroy(context));
    }
};
void tensor(cudnnTensorDescriptor_t descriptor, std::array<int, 4> expected) {
    cudnnDataType_t type{};
    int rank{};
    std::array<int, 8> sizes{}, strides{};
    sizes.fill(77);
    strides.fill(88);
    check(cudnnGetTensorNdDescriptor(descriptor, 8, &type, &rank, sizes.data(), strides.data()));
    expect(type == CUDNN_DATA_FLOAT && rank == 4 &&
               std::equal(expected.begin(), expected.end(), sizes.begin()) && sizes[4] == 77 &&
               strides[4] == 88,
           "native descriptor mutation reaches complete public tensor output without tail writes");
}
void ordinary(Application& app) {
    check(cudnnGetFoldedConvBackwardDataDescriptors(
        app.handle, app.input.filter, app.input.diff, app.input.convolution, app.input.grad,
        CUDNN_TENSOR_NCHW, app.output.filter, app.output.diff, app.output.convolution,
        app.output.grad, app.transforms[0], app.transforms[1], app.transforms[2],
        app.transforms[3]));
    tensor(app.output.diff, {102, 3, 2, 5});
    tensor(app.output.grad, {103, 2, 4, 6});
    tensor(app.input.diff, {1, 3, 2, 5});
    std::array<int, 8> sizes{};
    sizes.fill(73);
    int rank{};
    cudnnDataType_t type{};
    cudnnTensorFormat_t format{};
    check(cudnnGetFilterNdDescriptor(app.output.filter, 8, &type, &format, &rank, sizes.data()));
    expect(rank == 4 && sizes[0] == 101 && sizes[1] == 2 && sizes[2] == 3 && sizes[3] == 2 &&
               sizes[4] == 73,
           "independent synthetic filter mutation and trailing sentinel are observable");
    auto optional = app.transforms;
    optional[3] = nullptr;
    check(app.call(app.input, app.output, optional));
    auto repeated = app.transforms;
    repeated.fill(app.transforms[0]);
    check(app.call(app.input, app.input, repeated));
    tensor(app.input.diff, {102, 3, 2, 5});
    app.reset_inputs();
    Set duplicate = app.output;
    duplicate.grad = duplicate.diff;
    check(app.call(app.input, duplicate, repeated));
    tensor(app.output.diff, {103, 2, 4, 6});
    const auto calls = fixture::control.folds.load();
    optional[0] = nullptr;
    expect(app.call(app.input, app.output, optional) == CUDNN_STATUS_NOT_INITIALIZED,
           "mandatory null output rejects before native issue");
    auto wrong = app.output;
    wrong.filter = reinterpret_cast<cudnnFilterDescriptor_t>(app.output.diff);
    expect(app.call(app.input, wrong, app.transforms) == CUDNN_STATUS_NOT_INITIALIZED &&
               fixture::control.folds == calls,
           "wrong-kind identity cannot become a native output filter");
    fixture::control.rank_override = 0;
    expect(app.call() == CUDNN_STATUS_BAD_PARAM,
           "fresh convolution rank is rejected before unsafe Folded");
    fixture::control.rank_override = 2;
    fixture::control.stride_override = 0;
    expect(app.call() == CUDNN_STATUS_BAD_PARAM,
           "zero native stride cannot reach integer division");
    fixture::control.stride_override = 1;
    fixture::control.getter_status = CUDNN_STATUS_BAD_PARAM;
    expect(app.call() == CUDNN_STATUS_INTERNAL_ERROR,
           "preflight getter error is policy failure, not fabricated original Folded BAD_PARAM");
    fixture::control.getter_status = CUDNN_STATUS_SUCCESS;
    check(app.call());
    CUcontext foreign{};
    driver(cuCtxCreate(&foreign, 0, 0));
    expect(app.call() == CUDNN_STATUS_NOT_INITIALIZED,
           "exact owned handle requires its own current context");
    driver(cuCtxDestroy(foreign));
    driver(cuCtxSetCurrent(app.context));
    const auto child = ::fork();
    expect(child >= 0, "fork inherited Folded ownership");
    if (!child)
        ::_exit(app.call() == CUDNN_STATUS_NOT_INITIALIZED &&
                        !glimmer_dnn_folded_convolution_native_calls()
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "inherited fork cannot invoke remote or caller-native Folded");
}
void reciprocal(Application& app) {
    fixture::control.pause = true;
    fixture::control.release = false;
    fixture::control.entered = false;
    auto first = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        const auto status = app.call();
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    const auto deadline = Clock::now() + std::chrono::seconds{3};
    while (!fixture::control.entered.load() && Clock::now() < deadline) std::this_thread::yield();
    const bool entered = fixture::control.entered.load();
    std::atomic<unsigned int> started{};
    auto reverse = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        ++started;
        const auto status = app.call(app.output, app.input, app.transforms);
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    auto getter = std::async(std::launch::async, [&] {
        ++started;
        tensor(app.output.diff, {102, 3, 2, 5});
    });
    while (started.load() != 2 && Clock::now() < deadline) std::this_thread::yield();
    const bool waiting =
        started == 2 &&
        reverse.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        getter.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    fixture::control.release = true;
    check(first.get());
    check(reverse.get());
    getter.get();
    expect(entered && waiting,
           "reciprocal twelve-slot mutation and read complete without partial-pin deadlock");
    fixture::control.pause = false;
    app.reset_inputs();
}
void pending(Application& app) {
    std::array<CUdeviceptr, 4> memory{};
    constexpr std::array<std::size_t, 4> bytes{192, 144, 120, 64};
    for (std::size_t i = 0; i != 4; ++i) driver(cuMemAlloc(&memory[i], bytes[i]));
    const float alpha = 1, beta = 0;
    foundation::control.can_stop = false;
    foundation::control.fail_forward = true;
    expect(cudnnConvolutionForward(
               app.handle, &alpha, app.input.grad, reinterpret_cast<void*>(memory[0]),
               app.input.filter, reinterpret_cast<void*>(memory[1]), app.input.convolution,
               CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, reinterpret_cast<void*>(memory[3]),
               bytes[3], &beta, app.input.diff,
               reinterpret_cast<void*>(memory[2])) == CUDNN_STATUS_EXECUTION_FAILED,
           "prior original failure still owns its NOT_READY physical event");
    check(app.call());
    auto aliased = app.output;
    aliased.filter = app.input.filter;
    std::atomic<bool> started{};
    const auto calls = fixture::control.folds.load(), gets = fixture::control.gets.load();
    auto blocked = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        started = true;
        const auto status = app.call(app.input, aliased, app.transforms);
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    const auto deadline = Clock::now() + std::chrono::seconds{3};
    while (!started.load() && Clock::now() < deadline) std::this_thread::yield();
    const bool waiting =
        started && blocked.wait_for(std::chrono::milliseconds{25}) != std::future_status::ready;
    const bool untouched = fixture::control.folds == calls && fixture::control.gets == gets &&
                           !fixture::control.synchronizes && !foundation::control.can_stop;
    foundation::control.can_stop = true;
    check(blocked.get());
    expect(waiting && untouched,
           "physical immutable destination borrower blocks getter and mutation without drain");
    for (const auto allocation : memory) driver(cuMemFree(allocation));
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               ticks = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", ticks.c_str(), 1) == 0,
           "authenticate local Folded endpoint");
}
void poison(Application& app, unsigned int mode) {
    if (mode == 0) fixture::control.fold_status = CUDNN_STATUS_BAD_PARAM;
    if (mode == 1 || mode == 2) fixture::control.change_after_get = true;
    if (mode == 2) fixture::control.getter_status = CUDNN_STATUS_BAD_PARAM;
    if (mode == 3 || mode == 4) fixture::control.change_after_fold = true;
    if (mode == 4) fixture::control.fold_status = CUDNN_STATUS_EXECUTION_FAILED;
    if (mode == 5) fixture::control.rank_override = 7;
    const auto result = app.call();
    expect(result == (mode == 0   ? CUDNN_STATUS_BAD_PARAM
                      : mode == 5 ? CUDNN_STATUS_INTERNAL_ERROR
                                  : CUDNN_STATUS_NOT_INITIALIZED),
           "partial native error and context proof have exact distinct public failure mapping");
    const auto calls = fixture::control.folds.load(), gets = fixture::control.gets.load();
    expect(app.call() == CUDNN_STATUS_NOT_INITIALIZED && fixture::control.folds == calls &&
               fixture::control.gets == gets,
           "quarantine prevents local retry of Folded");
    if (mode == 0 || mode == 5) {
        check(cudnnSetStream(app.handle, nullptr));
        check(cudnnDestroy(app.handle));
        driver(cuCtxDestroy(app.context));
    } else if (mode == 1 || mode == 2) {
        tensor(app.input.diff, {1, 3, 2, 5});
    }
}
void run(const char* provider, ExecutionSchedulingMode mode, unsigned int missing = 0,
         int poisoned = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual Folded service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path = "/tmp/glimmer-folded-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch(missing));
    expect(service.start(), "start original Folded service");
    client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (poisoned >= 0)
            poison(app, static_cast<unsigned int>(poisoned));
        else {
            if (missing) {
                expect(app.call() == CUDNN_STATUS_NOT_SUPPORTED,
                       "missing callback preserves clean policy failure");
                tensor(app.input.diff, {1, 3, 2, 5});
                check(cudnnSetStream(app.handle, nullptr));
            } else {
                ordinary(app);
                reciprocal(app);
                pending(app);
            }
            app.close();
        }
    });
    const auto deadline = Clock::now() + std::chrono::seconds{25};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded Folded public IPC progress");
    client.get();
    expect(!glimmer_dnn_folded_convolution_native_calls() && !fixture::control.synchronizes,
           "all public calls use original owner, never caller SDK or added GPU synchronization");
    if (poisoned >= 0) {
        const bool known = poisoned == 0 || poisoned == 5;
        expect(base::state.handle_destroys == (known ? 1U : 0U) && !base::state.destroys &&
                   !foundation::state.destroys && !fixture::transform::state.destroys,
               "metadata uncertainty does not erase proven handle ownership or destroy partial "
               "outputs");
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished(), "all known Folded service owners retire");
    expect(base::state.creates == 4 && base::state.destroys == 4 &&
               foundation::state.creates == 4 && foundation::state.destroys == 4 &&
               fixture::transform::state.creates == 4 && fixture::transform::state.destroys == 4 &&
               base::state.handle_creates == 1 && base::state.handle_destroys == 1 &&
               foundation::driver.memory.empty() && foundation::driver.events.empty() &&
               foundation::driver.contexts.empty(),
           "all twelve descriptors, handle, physical buffers and context retire exactly once");
}
struct Hostile {
    ExecutionWireReply reply;
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    bool clean{}, disconnect{};
};
Hostile hostile(unsigned int index) {
    Hostile test;
    auto& reply = test.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    switch (index) {
        case 0:
            reply.parameter_bytes = 1;
            break;
        case 1:
            reply.parameter_bytes = 2;
            break;
        case 2:
            reply.parameter_bytes = 3;
            break;
        case 3:
            reply.parameter_bytes = 4;
            break;
        case 4:
            reply.id = 7;
            break;
        case 5:
            reply.address = 7;
            break;
        case 6:
            reply.data = {std::byte{7}};
            break;
        case 7:
            reply.original_status.reset();
            break;
        case 8:
            reply.original_status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 9:
            reply.status = 8192;
            reply.original_status = 8192;
            break;
        case 10:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            break;
        case 11:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.parameter_bytes = 1;
            test.status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 12:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 3;
            test.status = CUDNN_STATUS_NOT_INITIALIZED;
            break;
        case 13:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 2;
            break;  // Native provenance without metadata quarantine.
        case 14:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            test.status = CUDNN_STATUS_NOT_SUPPORTED;
            test.clean = true;
            break;
        case 15:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_INVALID_VALUE;
            test.status = CUDNN_STATUS_BAD_PARAM;
            test.clean = true;
            break;
        case 16:
            test.clean = true;
            test.status = CUDNN_STATUS_SUCCESS;
            break;
        case 17:
            test.disconnect = true;
            break;
        case 18:
        case 19:
        case 20:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = index == 18 ? -1 : index == 19 ? 0 : 1024;
            break;
        case 21:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            // Deliver a wire-valid replacement, not an encoder rejection.
            // Metadata mutation must never accept replacement-plan identity.
            reply.original_status.reset();
            reply.id = 17;
            break;
        case 22:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 2;
            test.status = CUDNN_STATUS_NOT_INITIALIZED;
            break;
        case 23:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.parameter_bytes = 3;
            break;
        default:
            throw std::runtime_error("unknown hostile Folded case");
    }
    return test;
}
void malformed_case(const char* provider, unsigned int index) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated hostile Folded identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-folded-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    const auto test = hostile(index);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start hostile Folded receipt endpoint");
    client = std::async(std::launch::async, [=] {
        Application app(provider, false);
        expect(app.call() == test.status,
               "Folded reply publishes only canonical original/policy status");
        expect(app.call() == (test.clean ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_INITIALIZED),
               "only clean reply permits another remote Folded invocation");
        expect(!glimmer_dnn_folded_convolution_native_calls(),
               "hostile receipt cannot trigger caller fallback");
    });
    unsigned int hellos{}, creates{}, handles{}, calls{};
    std::uint64_t metadata{}, execution{};
    bool disconnected{};
    constexpr std::array<std::uint32_t, 12> kinds{3, 1, 4, 1, 3, 1, 4, 1, 10, 10, 10, 10};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline, "bounded hostile Folded progress");
        if (disconnected) {
            std::this_thread::yield();
            continue;
        }
        expect(server.poll_for(1), "poll hostile Folded endpoint");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "real authenticated peer preserved in hostile test");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                ++hellos;
                ExecutionBootstrapOwnership ownership = ExecutionBootstrapOwnership::kRetained;
                if (!decode_execution_context_bootstrap(request->message.payload)) {
                    std::uint32_t role{}, flags{};
                    expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                               input.read_u32(flags) && !flags && input.empty() && !metadata,
                           "metadata Hello is exactly DNM1 role plus zero flags");
                    metadata = request->connection;
                    ownership = ExecutionBootstrapOwnership::kMetadataRetained;
                } else {
                    const auto descriptor =
                        decode_execution_context_bootstrap(request->message.payload);
                    expect(!execution && descriptor ==
                                             ExecutionContextBootstrap{
                                                 ExecutionContextPurpose::kExplicitUser, 0, 1},
                           "ordinary typed context cannot alias DNM1 metadata");
                    execution = request->connection;
                }
                const auto encoded = encode_execution_bootstrap_ownership(ownership);
                expect(encoded.has_value(), "encode exact retained role");
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(request->connection == metadata && creates < kinds.size() &&
                           input.read_u32(kind) && kind == kinds[creates] && input.empty(),
                       "twelve typed metadata creates retain positional identities");
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = ++creates;
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnn) {
                expect(request->connection == execution && input.empty() && !handles++,
                       "owned native handle create");
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = 41;
            } else {
                ExecutionDnnFoldedConvBackwardDataDescriptors operation;
                expect(request->message.header.operation ==
                               ExecutionOpcode::kDnnFoldedConvBackwardDataDescriptors &&
                           request->connection == execution &&
                           request->message.payload.size() == 116 &&
                           read_execution_dnn_folded_conv_backward_data_descriptors(input,
                                                                                    operation) &&
                           input.empty() && operation.handle == 41 &&
                           operation.metadata == metadata && !operation.format &&
                           operation.inputs == std::array<std::uint64_t, 4>{1, 2, 3, 4} &&
                           operation.outputs ==
                               std::array<std::uint64_t, 8>{5, 6, 7, 8, 9, 10, 11, 12},
                       "exact 116-byte Folded authority and all twelve positional IDs");
                if (++calls == 1) {
                    if (test.disconnect) {
                        server.stop();
                        disconnected = true;
                        continue;
                    }
                    reply = test.reply;
                } else {
                    expect(calls == 2 && test.clean,
                           "uncertain operation is never retried remotely");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded.has_value(), "encode hostile Folded receipt");
            const auto decoded = decode_execution_reply(*encoded);
            expect(decoded && decoded->disposition == reply.disposition,
                   "hostile envelope reaches route validation through the actual wire codec");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver exact hostile receipt");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 2 && creates == 12 && handles == 1 && calls == (test.clean ? 2U : 1U),
           "no hidden native setup, metadata mutation or replay after quarantine");
    server.stop();
}
void isolated(const char* executable, const char* provider, std::string_view mode) {
    const unsigned int count = mode == "--malformed-replies" ? kHostileCases
                               : mode == "--poison"          ? 6
                                                             : 2;
    const char* selected = mode == "--malformed-replies" ? "--malformed-case"
                           : mode == "--poison"          ? "--poison-case"
                                                         : "--missing-case";
    for (unsigned int i = 0; i != count; ++i) {
        const auto index = std::to_string(i);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated Folded public case");
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
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional Folded mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--missing-callback" || mode == "--poison" || mode == "--malformed-replies") {
            expect(argc == 3, "parent isolated mode takes no case index");
            isolated(argv[0], argv[1], mode);
        } else if (mode == "--missing-case" || mode == "--poison-case" ||
                   mode == "--malformed-case") {
            expect(argc == 4, "isolated case needs one bounded decimal index");
            const std::string_view text{argv[3]};
            expect(!text.empty() && text.size() <= 2, "bounded case index length");
            unsigned int index{};
            for (const char digit : text) {
                expect(digit >= '0' && digit <= '9', "decimal case index");
                index = 10 * index + static_cast<unsigned int>(digit - '0');
            }
            if (mode == "--malformed-case") {
                expect(index < kHostileCases, "known hostile index");
                malformed_case(argv[1], index);
                ::_exit(0);
            }
            expect(index < (mode == "--missing-case" ? 2U : 6U), "known isolation index");
            run(argv[1], ExecutionSchedulingMode::kNaive, mode == "--missing-case" ? index + 1 : 0,
                mode == "--poison-case" ? static_cast<int>(index) : -1);
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known scheduler mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "Folded descriptor CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.release = true;
        foundation::control.can_stop = true;
        std::cerr << error.what() << '\n';
        return 1;
    }
}
