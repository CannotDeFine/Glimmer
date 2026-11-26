#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_rnn_compute_fixture.h"

#include <cerrno>
#include <atomic>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <dlfcn.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_rnn_compute_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_rnn_compute;
namespace foundation = fixture::foundation;
using Clock = std::chrono::steady_clock;
struct TailHold {
    std::atomic<bool> is_armed{}, has_entered{};
};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudnnStatus_t value) {
    expect(value == CUDNN_STATUS_SUCCESS, "public original SDK success");
}
void driver(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned public Driver operation");
}
struct Cached {
    decltype(&cudnnRNNForward) forward{};
    decltype(&cudnnRNNBackwardData_v8) data{};
    decltype(&cudnnRNNBackwardWeights_v8) weights{};
    decltype(&cudnnBuildRNNDynamic) build{};
};
Cached lookup(const char* provider) {
    auto* library = dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent original caller provider");
    Cached api;
#define LOOKUP(member, symbol)                                                                     \
    do {                                                                                           \
        auto* value = dlsym(library, #symbol);                                                     \
        expect(value == reinterpret_cast<void*>(&symbol) && value == dlsym(RTLD_DEFAULT, #symbol), \
               "direct/global/provider routing identity");                                         \
        api.member = reinterpret_cast<decltype(api.member)>(value);                                \
    } while (false)
    LOOKUP(forward, cudnnRNNForward);
    LOOKUP(data, cudnnRNNBackwardData_v8);
    LOOKUP(weights, cudnnRNNBackwardWeights_v8);
    LOOKUP(build, cudnnBuildRNNDynamic);
#undef LOOKUP
    expect(dlclose(library) == 0, "cached function remains valid after provider close");
    return api;
}
struct Application {
    Cached api;
    cudnnRNNDescriptor_t rnn{};
    std::array<cudnnRNNDataDescriptor_t, 2> data{};
    std::array<cudnnTensorDescriptor_t, 2> tensors{};
    cudnnDropoutDescriptor_t dropout{};
    CUcontext context{};
    CUstream stream{};
    cudnnHandle_t handle{};
    std::array<CUdeviceptr, 17> memory{};
    std::array<std::size_t, 17> bytes{};
    unsigned int scalar{};
    Application(const char* provider, bool use_double = false) : api(lookup(provider)) {
        scalar = use_double ? 8 : 4;
        const auto type = use_double ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
        check(cudnnCreateRNNDescriptor(&rnn));
        check(cudnnCreateDropoutDescriptor(&dropout));
        check(cudnnSetRNNDescriptor_v8(rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                       CUDNN_RNN_SINGLE_INP_BIAS, CUDNN_UNIDIRECTIONAL,
                                       CUDNN_LINEAR_INPUT, type, type, CUDNN_DEFAULT_MATH, 1, 1, 1,
                                       1, dropout, 0));
        alignas(double) std::array<std::byte, 8> padding{};
        for (auto& descriptor : data) {
            check(cudnnCreateRNNDataDescriptor(&descriptor));
            check(cudnnSetRNNDataDescriptor(descriptor, type,
                                            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, 3, 2, 1,
                                            fixture::kLengths.data(), padding.data()));
        }
        const int dimensions[]{1, 2, 1}, strides[]{2, 1, 1};
        for (auto& descriptor : tensors) {
            check(cudnnCreateTensorDescriptor(&descriptor));
            check(cudnnSetTensorNdDescriptor(descriptor, type, 3, dimensions, strides));
        }
        driver(cuCtxCreate(&context, 0, 0));
        check(cudnnCreate(&handle));
        if (!use_double) {
            driver(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
            check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
        }
        bytes = {8,          6 * scalar,  6 * scalar, 6 * scalar,
                 6 * scalar, 2 * scalar,  2 * scalar, 2 * scalar,
                 2 * scalar, 2 * scalar,  2 * scalar, 2 * scalar,
                 2 * scalar, 12 * scalar, 32,         fixture::kReserve * scalar,
                 12 * scalar};
        for (std::size_t i = 0; i != memory.size(); ++i) {
            driver(cuMemAlloc(&memory[i], bytes[i] + 32));
            std::vector<std::byte> initial(bytes[i] + 32, std::byte{0xcd});
            std::fill(initial.begin() + 16, initial.end() - 16, std::byte{});
            driver(cuMemcpyHtoD(memory[i], initial.data(), initial.size()));
        }
        driver(cuMemcpyHtoD(memory[0] + 16, fixture::kLengths.data(), 8));
        if (use_double)
            initialize<double>();
        else
            initialize<float>();
    }
    template <typename T, std::size_t N>
    void upload(std::size_t slot, const std::array<double, N>& source) {
        std::array<T, N> output{};
        std::transform(source.begin(), source.end(), output.begin(),
                       [](double x) { return static_cast<T>(x); });
        driver(cuMemcpyHtoD(memory[slot] + 16, output.data(), sizeof(output)));
    }
    template <typename T>
    void initialize() {
        upload<T>(1, fixture::kInput);
        upload<T>(3, fixture::kDy);
        upload<T>(13, fixture::kWeight);
        upload<T>(5, fixture::kHidden);
        upload<T>(9, fixture::kCell);
        upload<T>(7, fixture::kDhidden);
        upload<T>(11, fixture::kDcell);
        std::array<double, 12> initial;
        initial.fill(.125);
        upload<T>(16, initial);
    }
    void* pointer(std::size_t slot) const {
        return reinterpret_cast<void*>(memory[slot] + 16);
    }
    cudnnStatus_t invoke(unsigned int kind, bool nullable = false, bool set = false) const {
        const auto* lengths = nullable ? nullptr : static_cast<const int32_t*>(pointer(0));
        const auto hidden = [&](std::size_t slot) { return nullable ? nullptr : pointer(slot); };
        if (kind == 0)
            return api.forward(handle, rnn, CUDNN_FWD_MODE_TRAINING, lengths, data[0], pointer(1),
                               data[1], pointer(2), tensors[0], hidden(5), hidden(6), tensors[1],
                               hidden(9), hidden(10), bytes[13], pointer(13), bytes[14],
                               pointer(14), bytes[15], pointer(15));
        if (kind == 1)
            return api.data(handle, rnn, lengths, data[1], pointer(2), pointer(3), data[0],
                            pointer(4), tensors[0], hidden(5), hidden(7), hidden(8), tensors[1],
                            hidden(9), hidden(11), hidden(12), bytes[13], pointer(13), bytes[14],
                            pointer(14), bytes[15], pointer(15));
        return api.weights(handle, rnn, set ? CUDNN_WGRAD_MODE_SET : CUDNN_WGRAD_MODE_ADD, lengths,
                           data[0], pointer(1), tensors[0], hidden(5), data[1], pointer(2),
                           bytes[16], pointer(16), bytes[14], pointer(14), bytes[15], pointer(15));
    }
    template <typename T>
    std::vector<T> read(std::size_t slot) const {
        std::vector<T> result(bytes[slot] / sizeof(T));
        driver(cuMemcpyDtoH(result.data(), memory[slot] + 16, bytes[slot]));
        return result;
    }
    void guards() const {
        for (std::size_t i = 0; i != memory.size(); ++i) {
            std::vector<std::byte> output(bytes[i] + 32);
            driver(cuMemcpyDtoH(output.data(), memory[i], output.size()));
            for (std::size_t j = 0; j != 16; ++j)
                expect(output[j] == std::byte{0xcd} && output[16 + bytes[i] + j] == std::byte{0xcd},
                       "public complete allocation guards remain untouched");
        }
    }
    void close() {
        for (const auto allocation : memory) driver(cuMemFree(allocation));
        check(cudnnDestroy(handle));
        check(cudnnDestroyRNNDescriptor(rnn));
        for (const auto descriptor : data) check(cudnnDestroyRNNDataDescriptor(descriptor));
        for (const auto descriptor : tensors) check(cudnnDestroyTensorDescriptor(descriptor));
        check(cudnnDestroyDropoutDescriptor(dropout));
        if (stream) driver(cuStreamDestroy(stream));
        driver(cuCtxDestroy(context));
    }
};
template <typename T>
void numerical(Application& app) {
    check(app.api.build(app.handle, app.rnn, 2));
    check(app.invoke(0));
    check(app.invoke(1));
    check(app.invoke(2));
    check(app.invoke(2));
    const auto tolerance = sizeof(T) == 8 ? 2e-8 : 3e-6;
    const auto dx = app.read<T>(4), dhx = app.read<T>(8), dcx = app.read<T>(12),
               dw = app.read<T>(16);
    for (std::size_t i = 0; i != 6; ++i)
        expect(std::abs(dx[i] - fixture::derivative(0, i)) < tolerance,
               "public full input gradients");
    for (std::size_t i = 0; i != 12; ++i)
        expect(std::abs(dw[i] - (.125 + 2 * fixture::derivative(1, i))) < tolerance,
               "public full accumulated weight gradients preserve opaque byte transfer");
    for (std::size_t i = 0; i != 2; ++i)
        expect(std::abs(dhx[i] - fixture::derivative(2, i)) < tolerance &&
                   std::abs(dcx[i] - fixture::derivative(3, i)) < tolerance,
               "public hidden/cell gradients");
    const auto y = app.read<T>(2), hy = app.read<T>(6), cy = app.read<T>(10);
    constexpr std::array expected_y{.09480791354991329,   -.0708897405454054,  .1632489323985552,
                                    -.007892195938427039, .051490083047405634, 0.0};
    constexpr std::array expected_c{.10812837416783348, -.01670197275016795};
    constexpr std::array expected_h{.051490083047405634, -.007892195938427039};
    for (std::size_t i = 0; i != y.size(); ++i)
        expect(std::abs(y[i] - expected_y[i]) < tolerance, "every independent forward output");
    for (std::size_t i = 0; i != hy.size(); ++i)
        expect(std::abs(hy[i] - expected_h[i]) < tolerance &&
                   std::abs(cy[i] - expected_c[i]) < tolerance,
               "complete final hidden and cell outputs");
    const auto input = app.read<T>(1), weight = app.read<T>(13), hidden = app.read<T>(5),
               cell = app.read<T>(9);
    for (std::size_t i = 0; i != input.size(); ++i)
        expect(input[i] == static_cast<T>(fixture::kInput[i]), "complete readonly input unchanged");
    for (std::size_t i = 0; i != weight.size(); ++i)
        expect(weight[i] == static_cast<T>(fixture::kWeight[i]),
               "complete readonly weights unchanged");
    for (std::size_t i = 0; i != hidden.size(); ++i)
        expect(hidden[i] == static_cast<T>(fixture::kHidden[i]) &&
                   cell[i] == static_cast<T>(fixture::kCell[i]),
               "complete readonly state unchanged");
    double loss{};
    for (std::size_t i = 0; i != 6; ++i) loss += y[i] * fixture::kDy[i];
    for (std::size_t i = 0; i != 2; ++i)
        loss += hy[i] * fixture::kDhidden[i] + cy[i] * fixture::kDcell[i];
    expect(std::abs(loss - fixture::loss(fixture::kInput, fixture::kWeight, fixture::kHidden,
                                         fixture::kCell)) < 2e-6,
           "public complete forward loss agrees with independent oracle");
    app.guards();
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "authenticated endpoint");
}
void wait_flag(const std::atomic<bool>& flag) {
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (!flag.load()) {
        expect(Clock::now() < deadline, "bounded started handshake");
        std::this_thread::yield();
    }
}
void run(const char* provider, ExecutionSchedulingMode mode, unsigned int missing = 0,
         int poison = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "peer identity");
    ExecutionServiceConfig config;
    config.transport.socket_path = "/tmp/glimmer-rnn-compute-" + std::to_string(getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    const auto tail_hold = std::make_shared<TailHold>();
    std::optional<std::pair<unsigned int, std::size_t>> before_proof;
    fixture::state.callback = [&, tail_hold](unsigned int phase) {
        // Hold only the selected native call, not an earlier profile-mode tail.
        if (phase == 2 && tail_hold->is_armed.exchange(false)) {
            foundation::control.can_stop.store(false);
            tail_hold->has_entered.store(true);
        }
        if (poison >= 0 && poison < 4) {
            // Runs on the service owner at the first authoritative observation,
            // after all pageable uploads but before the requested RNN issues.
            // Transfer events belong to healthy setup, not this RNN proof.
            if (phase == 1 && !before_proof)
                before_proof = {fixture::control.records.load(), foundation::driver.memory.size()};
        }
    };
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch(missing));
    expect(service.start(), "start real owned RNN service");
    client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (missing) {
            const auto status = missing == 4 ? app.api.build(app.handle, app.rnn, 2)
                                             : app.invoke(missing < 4 ? missing - 1 : 0);
            expect(status == CUDNN_STATUS_NOT_SUPPORTED && !fixture::calls(),
                   "missing original/preflight callback cleanly rejects without fallback");
            app.close();
            return;
        }
        if (poison >= 0) {
            fixture::control.arithmetic.store(false);
            if (poison == 0) fixture::control.context_loss.store(2);
            if (poison == 1) {
                fixture::control.status.store(CUDNN_STATUS_EXECUTION_FAILED);
                fixture::control.sync_status.store(CUDA_ERROR_LAUNCH_FAILED);
            }
            if (poison == 2) fixture::control.stream_change.store(1);
            if (poison == 3) fixture::control.stream_status.store(CUDNN_STATUS_BAD_PARAM);
            if (poison == 4) fixture::control.corrupt.store(1);
            if (poison == 5) fixture::control.build_status.store(CUDNN_STATUS_EXECUTION_FAILED);
            const auto result = poison == 5 ? app.api.build(app.handle, app.rnn, 2) : app.invoke(0);
            const auto expected = poison == 0   ? CUDNN_STATUS_NOT_INITIALIZED
                                  : poison == 5 ? CUDNN_STATUS_EXECUTION_FAILED
                                                : CUDNN_STATUS_INTERNAL_ERROR;
            expect(result == expected, "uncertain proof never projects successful original status");
            const auto calls = fixture::calls();
            expect(app.invoke(0) == CUDNN_STATUS_NOT_INITIALIZED && fixture::calls() == calls,
                   "quarantined owner refuses repeat locally without second native invocation");
            if (poison >= 4) {
                check(cudnnDestroy(app.handle));
                driver(cuCtxDestroy(app.context));
            }
            return;
        }
        numerical<float>(app);
        const auto child = fork();
        expect(child >= 0, "fork inherited public handles");
        if (!child)
            _exit(app.api.build(app.handle, app.rnn, 2) == CUDNN_STATUS_NOT_INITIALIZED ? 0 : 1);
        int child_status{};
        expect(waitpid(child, &child_status, 0) == child && WIFEXITED(child_status) &&
                   !WEXITSTATUS(child_status),
               "fork-inherited authority never reaches native Build");
        CUcontext other{};
        driver(cuCtxCreate(&other, 0, 0));
        const auto before = fixture::calls();
        expect(app.invoke(0) == CUDNN_STATUS_NOT_INITIALIZED && fixture::calls() == before,
               "cross-context handle cannot issue original RNN work");
        driver(cuCtxDestroy(other));
        driver(cuCtxSetCurrent(app.context));
        fixture::control.arithmetic.store(false);
        const auto tensor_gets = fixture::control.tensor_gets.load();
        check(app.invoke(0, true));
        expect(fixture::control.tensor_gets.load() == tensor_gets,
               "null active h/c pointers skip shape queries for supplied unused descriptors");
        const auto syncs = fixture::control.syncs.load();
        fixture::control.status.store(CUDNN_STATUS_EXECUTION_FAILED);
        expect(app.invoke(0) == CUDNN_STATUS_EXECUTION_FAILED &&
                   fixture::control.syncs.load() == syncs + 1,
               "partial native error preserved after one context-wide proof");
        fixture::control.status.store(CUDNN_STATUS_SUCCESS);
        expect(app.invoke(2, false, true) == CUDNN_STATUS_NOT_SUPPORTED,
               "signed original weight ADD/SET mode is not rewritten");
        tail_hold->is_armed.store(true);
        check(app.invoke(0));
        expect(tail_hold->has_entered.load() && !tail_hold->is_armed.load(),
               "selected original call arms its own physical tail exactly once");
        std::atomic<bool> started{};
        auto build = std::async(std::launch::async, [&] {
            driver(cuCtxSetCurrent(app.context));
            started.store(true);
            const auto result = app.api.build(app.handle, app.rnn, 3);
            driver(cuCtxSetCurrent(nullptr));
            return result;
        });
        wait_flag(started);
        const bool blocked =
            build.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
        foundation::control.can_stop.store(true);
        check(build.get());
        expect(blocked, "host ACK does not permit dynamic population before physical event");
        app.close();
        fixture::control.arithmetic.store(true);
        Application double_app(provider, true);
        numerical<double>(double_app);
        double_app.close();
    });
    const auto deadline = Clock::now() + std::chrono::seconds{25};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded RNN IPC progress");
    client.get();
    fixture::state.callback = {};
    expect(!glimmer_dnn_rnn_compute_native_calls(), "zero caller-native calls through every route");
    if (poison < 0)
        expect(foundation::driver.memory.empty() && foundation::driver.contexts.empty(),
               "normal real IPC leaves no allocation/context ownership");
    if (poison >= 0 && poison < 4)
        expect(before_proof && before_proof->second &&
                   fixture::control.records.load() == before_proof->first &&
                   !fixture::base::state.handle_destroys &&
                   foundation::driver.memory.size() == before_proof->second,
               "failed proof never releases physical owners");
}

// Hostile replies use real authenticated IPC, not calls to the reply helper.
// Build needs only a parent; compute uses typed RNNData identities with null
// ranges so the fake peer can test the receipt without pretending device work.
void malformed(const char* provider, unsigned int index) {
    const bool build = index >= 16;
    const auto test = index % 16;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "peer identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-rnn-compute-hostile-" + std::to_string(getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "hostile peer starts");
    const bool clean = test == 12 || test == 13 || test == 14;
    const auto expected = test == 12                 ? CUDNN_STATUS_NOT_SUPPORTED
                          : test == 13               ? CUDNN_STATUS_SUCCESS
                          : test == 14 || test == 15 ? CUDNN_STATUS_BAD_PARAM
                                                     : CUDNN_STATUS_INTERNAL_ERROR;
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnRNNDescriptor_t rnn{};
        check(cudnnCreateRNNDescriptor(&rnn));
        std::array<cudnnRNNDataDescriptor_t, 2> data{};
        if (!build)
            for (auto& value : data) check(cudnnCreateRNNDataDescriptor(&value));
        CUcontext context{};
        driver(cuCtxCreate(&context, 0, 0));
        cudnnHandle_t handle{};
        check(cudnnCreate(&handle));
        const auto invoke = [&] {
            return build
                       ? api.build(handle, rnn, -7)
                       : api.forward(handle, rnn, CUDNN_FWD_MODE_TRAINING, nullptr, data[0],
                                     nullptr, data[1], nullptr, nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, 0, nullptr, 0, nullptr, 0, nullptr);
        };
        expect(invoke() == expected, "hostile result has exact failure/native-control status");
        const auto follow = invoke();
        expect(follow == (clean && !(build && test == 14) ? CUDNN_STATUS_SUCCESS
                                                          : CUDNN_STATUS_NOT_INITIALIZED),
               "only canonical healthy receipt permits another RPC");
        expect(!glimmer_dnn_rnn_compute_native_calls(), "malformed reply cannot trigger local SDK");
    });
    unsigned int operations{}, hellos{}, creates{}, handles{};
    std::uint64_t metadata_connection{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded hostile reply progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "authenticated request owner");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            const auto opcode = request->message.header.operation;
            if (opcode == ExecutionOpcode::kHello) {
                ++hellos;
                auto ownership = ExecutionBootstrapOwnership::kRetained;
                if (!decode_execution_context_bootstrap(request->message.payload)) {
                    std::uint32_t role{}, flags{};
                    expect(input.read_u32(role) && role == 0x314d4e44U && input.read_u32(flags) &&
                               !flags && input.empty(),
                           "exact DNM1 bootstrap");
                    metadata_connection = request->connection;
                    ownership = ExecutionBootstrapOwnership::kMetadataRetained;
                }
                const auto encoded = encode_execution_bootstrap_ownership(ownership);
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (opcode == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && input.empty() && kind == (creates ? 11U : 13U),
                       "exact parent/data create sequence");
                reply.id = 17 + creates++;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (opcode == ExecutionOpcode::kCreateDnn) {
                expect(input.empty(), "empty handle creation");
                ++handles;
                reply.id = 23;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else {
                if (build) {
                    ExecutionDnnBuildRnnDynamic value;
                    expect(opcode == ExecutionOpcode::kDnnBuildRnnDynamic &&
                               read_execution_dnn_build_rnn_dynamic(input, value) &&
                               input.empty() && value.handle == 23 &&
                               value.metadata == metadata_connection && value.descriptor == 17 &&
                               value.mini_batch == -7,
                           "precise original signed Build request");
                } else {
                    ExecutionDnnRnnCompute value;
                    expect(opcode == ExecutionOpcode::kDnnRnnCompute &&
                               read_execution_dnn_rnn_compute(input, value) && input.empty() &&
                               value.handle == 23 && value.metadata == metadata_connection &&
                               value.descriptors == std::array<std::uint64_t, 5>{17, 18, 19, 0, 0},
                           "precise typed compute request");
                }
                ++operations;
                reply.id = build ? 0 : 1;
                reply.original_status = CUDNN_STATUS_SUCCESS;
                if (operations == 1) switch (test) {
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
                            reply.id = build ? 7 : 0;
                            break;
                        case 4:
                            reply.address = 1;
                            break;
                        case 5:
                            reply.data = {std::byte{1}};
                            break;
                        case 6:
                            reply.original_status.reset();
                            break;
                        case 7:
                            reply.status = CUDNN_STATUS_BAD_PARAM;
                            break;
                        case 8:
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.status = -1;
                            reply.original_status.reset();
                            break;
                        case 9:
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.status = 1024;
                            reply.original_status.reset();
                            break;
                        case 10:
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.status = CUDA_ERROR_UNKNOWN;
                            break;
                        case 11:
                            reply.parameter_bytes = 4;
                            break;
                        case 12:
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.id = 0;
                            reply.status = CUDA_ERROR_NOT_SUPPORTED;
                            reply.original_status.reset();
                            break;
                        case 13:
                            break;
                        case 14:
                            reply.status = CUDNN_STATUS_BAD_PARAM;
                            reply.original_status = reply.status;
                            reply.parameter_bytes = build ? 1 : 0;
                            break;
                        case 15:
                            reply.disposition = ExecutionWireDisposition::kRejected;
                            reply.status = CUDA_ERROR_INVALID_VALUE;
                            reply.parameter_bytes = 2;
                            break;
                    }
            }
            const auto payload = encode_execution_reply(reply);
            expect(payload.has_value() && decode_execution_reply(*payload).has_value(),
                   "wire-valid hostile receipt");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *payload}),
                   "deliver hostile envelope");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 2 && creates == (build ? 1U : 3U) && handles == 1 &&
               operations == (clean && !(build && test == 14) ? 2U : 1U),
           "exact RPC count without retry/fallback");
    server.stop();
}
void isolated(const char* executable, const char* provider, std::string_view mode) {
    const auto count = mode == "--malformed-replies" ? 32U : mode == "--poison" ? 6U : 5U;
    const auto selected = mode == "--malformed-replies" ? "--malformed-case"
                          : mode == "--poison"          ? "--poison-case"
                                                        : "--missing-case";
    for (unsigned int i = 0; i != count; ++i) {
        const auto number = std::to_string(i);
        const auto child = fork();
        expect(child >= 0, "isolated case fork");
        if (!child) {
            execl("/proc/self/exe", executable, provider, selected, number.c_str(),
                  static_cast<char*>(nullptr));
            _exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error(std::string(selected) + " failed " + number);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional test mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--poison" || mode == "--missing-callback")
            isolated(argv[0], argv[1], mode);
        else if (argc == 4) {
            const auto index = static_cast<unsigned int>(std::stoul(argv[3]));
            if (mode == "--malformed-case") {
                expect(index < 32, "known malformed case");
                malformed(argv[1], index);
                _exit(0);
            }
            expect(
                (mode == "--poison-case" && index < 6) || (mode == "--missing-case" && index < 5),
                "known native case");
            run(argv[1], ExecutionSchedulingMode::kNaive, mode == "--missing-case" ? index + 1 : 0,
                mode == "--poison-case" ? static_cast<int>(index) : -1);
            // Unproven GPU owners are deliberately retained. The parent verifies
            // this isolated child's assertions and exit, not leak-free cleanup.
            if (mode == "--poison-case") _exit(0);
        } else {
            expect(mode.empty() || mode == "--profile" || mode == "--priority",
                   "known scheduler mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "RNN CPU arithmetic and real IPC ownership checks passed\n";
    } catch (const std::exception& error) {
        fixture::control.release.store(true);
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
