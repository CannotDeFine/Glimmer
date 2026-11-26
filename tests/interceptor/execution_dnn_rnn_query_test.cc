#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_rnn_query_fixture.h"

#include <cerrno>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_rnn_query_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_rnn_query;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kHostileCases = 40;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudnnStatus_t status) {
    expect(status == CUDNN_STATUS_SUCCESS, "public original SDK success");
}
void driver(CUresult status) {
    expect(status == CUDA_SUCCESS, "public owned Driver success");
}
struct Cached {
    decltype(&cudnnGetRNNWeightSpaceSize) weight{};
    decltype(&cudnnGetRNNTempSpaceSizes) temp{};
    decltype(&cudnnGetMultiHeadAttnBuffers) attention{};
    decltype(&cudnnRNNSetClip_v8) clip{};
    decltype(&cudnnRNNGetClip_v8) get_clip{};
};
Cached lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent RNN query provider");
    Cached api;
#define LOOKUP(member, symbol)                                                                    \
    do {                                                                                          \
        auto* value = ::dlsym(library, #symbol);                                                  \
        expect(                                                                                   \
            value == reinterpret_cast<void*>(&symbol) && value == ::dlsym(RTLD_DEFAULT, #symbol), \
            "controlled direct/global/explicit lookup");                                          \
        api.member = reinterpret_cast<decltype(api.member)>(value);                               \
    } while (false)
    LOOKUP(weight, cudnnGetRNNWeightSpaceSize);
    LOOKUP(temp, cudnnGetRNNTempSpaceSizes);
    LOOKUP(attention, cudnnGetMultiHeadAttnBuffers);
    LOOKUP(clip, cudnnRNNSetClip_v8);
    LOOKUP(get_clip, cudnnRNNGetClip_v8);
#undef LOOKUP
    expect(::dlclose(library) == 0, "cached query pointers survive provider close");
    return api;
}
struct Guarded {
    std::uint64_t before{0x123456789abcdef0ULL};
    std::size_t value{71};
    std::uint64_t after{0xfedcba9876543210ULL};
    void verify(std::size_t expected) const {
        expect(
            before == 0x123456789abcdef0ULL && after == 0xfedcba9876543210ULL && value == expected,
            "full-width public output and both guards remain exact");
    }
};
struct Application {
    Cached api;
    cudnnRNNDescriptor_t rnn{};
    cudnnAttnDescriptor_t attention{};
    cudnnRNNDataDescriptor_t input{};
    cudnnDropoutDescriptor_t dropout{};
    CUcontext context{};
    cudnnHandle_t handle{};
    explicit Application(const char* provider) : api(lookup(provider)) {
        check(cudnnCreateRNNDescriptor(&rnn));
        check(cudnnCreateAttnDescriptor(&attention));
        check(cudnnCreateRNNDataDescriptor(&input));
        check(cudnnCreateDropoutDescriptor(&dropout));
        check(cudnnSetRNNDescriptor_v8(rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                       CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                       CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                       CUDNN_DEFAULT_MATH, 4, 4, 4, 1, dropout, 0));
        check(cudnnSetAttnDescriptor(attention, 0, 1, 1.0, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                     CUDNN_DEFAULT_MATH, dropout, dropout, 4, 4, 4, 0, 0, 0, 0, 3,
                                     3, 2, 1));
        configure_input();
        driver(cuCtxCreate(&context, 0, 0));
        check(cudnnCreate(&handle));
    }
    void configure_input() const {
        const int lengths[]{3, 2};
        float padding{};
        check(cudnnSetRNNDataDescriptor(input, CUDNN_DATA_FLOAT,
                                        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, 3, 2, 4, lengths,
                                        &padding));
    }
    cudnnStatus_t query(unsigned int kind, unsigned int mask,
                        const std::array<std::size_t*, 3>& values) const {
        const auto slot = [&](std::size_t i) { return mask & (1U << i) ? values[i] : nullptr; };
        if (!kind) return api.weight(handle, rnn, slot(0));
        if (kind == 1)
            return api.temp(handle, rnn, CUDNN_FWD_MODE_INFERENCE, input, slot(1), slot(2));
        return api.attention(handle, attention, slot(0), slot(1), slot(2));
    }
    void close() {
        check(cudnnDestroy(handle));
        check(cudnnDestroyRNNDescriptor(rnn));
        check(cudnnDestroyAttnDescriptor(attention));
        check(cudnnDestroyRNNDataDescriptor(input));
        check(cudnnDestroyDropoutDescriptor(dropout));
        driver(cuCtxDestroy(context));
    }
};
void wait_flag(const std::atomic<bool>& flag) {
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (!flag.load()) {
        expect(Clock::now() < deadline, "bounded started callback handshake");
        std::this_thread::yield();
    }
}
void values(Application& app) {
    for (unsigned int kind = 0; kind != 3; ++kind) {
        const auto allowed = kind == 0 ? 1U : kind == 1 ? 6U : 7U;
        for (unsigned int mask = 0; mask != 8; ++mask) {
            if (mask & ~allowed) continue;
            std::array<Guarded, 3> outputs;
            const auto expected = (!kind && !mask) || (kind == 2 && (mask & 3) != 3)
                                      ? CUDNN_STATUS_BAD_PARAM
                                      : CUDNN_STATUS_SUCCESS;
            expect(app.query(kind, mask,
                             {&outputs[0].value, &outputs[1].value, &outputs[2].value}) == expected,
                   "every nullable output mask reaches original validation unchanged");
            for (std::size_t i = 0; i != 3; ++i)
                outputs[i].verify(expected == CUDNN_STATUS_SUCCESS && (mask & (1U << i))
                                      ? fixture::control.sizes[i].load()
                                      : 71);
        }
    }
    Guarded alias;
    check(app.query(1, 6, {nullptr, &alias.value, &alias.value}));
    alias.verify(41);
    check(app.query(2, 7, {&alias.value, &alias.value, &alias.value}));
    alias.verify(29);  // Native weight, reserve, then workspace: not wire-slot order.
    {
        Guarded shared, workspace;
        check(app.query(2, 7, {&shared.value, &workspace.value, &shared.value}));
        shared.verify(41);  // Reserve overwrites weight even without workspace aliasing.
        workspace.verify(29);
    }
    {
        Guarded shared, reserve;
        check(app.query(2, 7, {&shared.value, &shared.value, &reserve.value}));
        shared.verify(29);
        reserve.verify(41);
    }
    {
        Guarded weight, shared;
        check(app.query(2, 7, {&weight.value, &shared.value, &shared.value}));
        weight.verify(17);
        shared.verify(29);  // Workspace overwrites reserve, independently of weight.
    }
    for (const auto bytes : {std::size_t{0}, (std::size_t{1} << 40) + 7, SIZE_MAX}) {
        for (auto& value : fixture::control.sizes) value.store(bytes);
        std::array<Guarded, 3> outputs;
        check(app.query(2, 7, {&outputs[0].value, &outputs[1].value, &outputs[2].value}));
        for (const auto& output : outputs) output.verify(bytes);
    }
    for (std::size_t i = 0; i != 3; ++i) fixture::control.sizes[i].store(17 + 12 * i);
    for (unsigned int kind = 0; kind != 3; ++kind) {
        fixture::control.statuses[kind].store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
        std::array<Guarded, 3> outputs;
        expect(app.query(kind,
                         kind == 0   ? 1
                         : kind == 1 ? 6
                                     : 7,
                         {&outputs[0].value, &outputs[1].value, &outputs[2].value}) ==
                   CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
               "original detailed query error survives private partial writes");
        for (const auto& output : outputs) output.verify(71);
        fixture::control.statuses[kind].store(CUDNN_STATUS_SUCCESS);
    }
    Guarded output;
    fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM);
    expect(app.api.temp(app.handle, app.rnn, CUDNN_FWD_MODE_INFERENCE, app.input, &output.value,
                        nullptr) == CUDNN_STATUS_INTERNAL_ERROR,
           "private getter error is policy UNKNOWN, never fabricated original query BAD_PARAM");
    output.verify(71);
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    fixture::control.corrupt.store(3);
    expect(app.api.temp(app.handle, app.rnn, CUDNN_FWD_MODE_INFERENCE, app.input, &output.value,
                        nullptr) == CUDNN_STATUS_BAD_PARAM,
           "unsafe workspace preflight rejects without original query");
    check(app.api.temp(app.handle, app.rnn, CUDNN_FWD_MODE_INFERENCE, app.input, nullptr,
                       &output.value));
    output.verify(41);
    fixture::control.corrupt.store(0);
    check(app.api.weight(app.handle, app.rnn, &output.value));
    output.verify(17);
    for (const auto range : {std::array<double, 2>{-INFINITY, INFINITY}, {-0.0, 0.0}, {-4, 9}})
        check(
            app.api.clip(app.rnn, CUDNN_RNN_CLIP_MINMAX, CUDNN_PROPAGATE_NAN, range[0], range[1]));
    expect(cudnnDestroyDropoutDescriptor(app.dropout) == CUDNN_STATUS_BAD_PARAM,
           "Clip does not detach original dropout dependencies");
    cudnnRNNClipMode_t mode = CUDNN_RNN_CLIP_MINMAX;
    cudnnNanPropagation_t nan = CUDNN_PROPAGATE_NAN;
    double lower = 7, upper = 9;
    expect(app.api.get_clip(app.rnn, &mode, &nan, &lower, &upper) == CUDNN_STATUS_NOT_SUPPORTED &&
               mode == CUDNN_RNN_CLIP_MINMAX && nan == CUDNN_PROPAGATE_NAN && lower == 7 &&
               upper == 9,
           "typed GetClip guard publishes no invented state");
    const auto calls = fixture::calls();
    expect(app.api.weight(app.handle, reinterpret_cast<cudnnRNNDescriptor_t>(app.input),
                          &output.value) == CUDNN_STATUS_NOT_INITIALIZED &&
               app.api.temp(app.handle, app.rnn, static_cast<cudnnForwardMode_t>(2), app.input,
                            &output.value, nullptr) == CUDNN_STATUS_BAD_PARAM &&
               fixture::calls() == calls,
           "wrong kinds and invalid enum representations never dispatch");
    const auto child = ::fork();
    expect(child >= 0, "fork inherited public token test");
    if (!child)
        ::_exit(app.api.weight(app.handle, app.rnn, &output.value) == CUDNN_STATUS_NOT_INITIALIZED
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "forked process cannot reuse owned query authority");
}
void publication(Application& app) {
    auto* page = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(page != MAP_FAILED, "map hostile public output page");
    Guarded first;
    const auto calls = fixture::calls();
    expect(app.api.temp(app.handle, app.rnn, CUDNN_FWD_MODE_INFERENCE, app.input, &first.value,
                        static_cast<std::size_t*>(page)) == CUDNN_STATUS_BAD_PARAM &&
               fixture::calls() == calls,
           "all output spans validate before native call and any scalar publication");
    first.verify(71);
    expect(::mprotect(page, 4096, PROT_READ | PROT_WRITE) == 0,
           "make later output temporarily writable");
    fixture::control.pause.store(true);
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    auto result = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        const auto status = app.api.temp(app.handle, app.rnn, CUDNN_FWD_MODE_INFERENCE, app.input,
                                         &first.value, static_cast<std::size_t*>(page));
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    wait_flag(fixture::control.entered);
    const auto protected_result = ::mprotect(page, 4096, PROT_NONE);
    fixture::control.release.store(true);
    expect(result.get() == CUDNN_STATUS_BAD_PARAM && protected_result == 0,
           "post-RPC validation checks all spans before publishing the first output");
    first.verify(71);
    expect(::munmap(page, 4096) == 0, "release only owned hostile page");
}
void exclusion(Application& app) {
    fixture::control.pause.store(true);
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    auto query = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        std::size_t bytes{};
        const auto result =
            app.api.temp(app.handle, app.rnn, CUDNN_FWD_MODE_INFERENCE, app.input, &bytes, nullptr);
        driver(cuCtxSetCurrent(nullptr));
        return result;
    });
    wait_flag(fixture::control.entered);
    std::atomic<unsigned int> started{};
    auto clip = std::async(std::launch::async, [&] {
        ++started;
        return app.api.clip(app.rnn, CUDNN_RNN_CLIP_MINMAX, CUDNN_NOT_PROPAGATE_NAN, -1, 1);
    });
    auto input = std::async(std::launch::async, [&] {
        ++started;
        app.configure_input();
    });
    const auto deadline = Clock::now() + std::chrono::seconds{3};
    while (started.load() != 2 && Clock::now() < deadline) std::this_thread::yield();
    const bool began = started.load() == 2;
    const bool blocked =
        clip.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        input.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    fixture::control.release.store(true);
    check(query.get());
    check(clip.get());
    input.get();
    expect(began && blocked,
           "active query and input/parent mutations finish without overlapping original calls");
    // This is combined frontend/service exclusion; direct backend callback tests
    // independently prove immutable leases and scoped reentry protection.
}
void pending(Application& app) {
    cudnnTensorDescriptor_t tensor{};
    check(cudnnCreateTensorDescriptor(&tensor));
    check(cudnnSetTensor4dDescriptor(tensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1));
    std::array<CUdeviceptr, 2> buffers{};
    for (auto& buffer : buffers) driver(cuMemAlloc(&buffer, 4));
    const float alpha = 1, beta = 0;
    foundation::control.can_stop.store(false);
    expect(cudnnAddTensor(app.handle, &alpha, tensor, reinterpret_cast<void*>(buffers[0]), &beta,
                          tensor,
                          reinterpret_cast<void*>(buffers[1])) == CUDNN_STATUS_EXECUTION_FAILED,
           "prior original error still holds a trailing physical event");
    std::array<Guarded, 3> outputs;
    const auto synchronizes = fixture::control.synchronizes.load();
    for (unsigned int kind = 0; kind != 3; ++kind)
        check(app.query(kind,
                        kind == 0   ? 1
                        : kind == 1 ? 6
                                    : 7,
                        {&outputs[0].value, &outputs[1].value, &outputs[2].value}));
    expect(!foundation::control.can_stop.load() &&
               fixture::control.synchronizes.load() == synchronizes,
           "all public queries return with unrelated work NOT_READY, without drain");
    std::atomic<bool> started{};
    auto destroy = std::async(std::launch::async, [&] {
        started.store(true);
        return cudnnDestroyTensorDescriptor(tensor);
    });
    wait_flag(started);
    const bool blocked =
        destroy.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    check(destroy.get());
    expect(blocked, "queries and host ACK never release prior physical tensor lease");
    for (const auto buffer : buffers) driver(cuMemFree(buffer));
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated endpoint");
}
void run(const char* provider, ExecutionSchedulingMode mode, unsigned int missing = 0,
         int poison = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "service peer identity");
    ExecutionServiceConfig config;
    config.transport.socket_path = "/tmp/glimmer-rnn-query-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch(missing));
    expect(service.start(), "start owned RNN size-query service");
    client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (missing) {
            std::array<Guarded, 3> output;
            const auto status =
                missing == 4
                    ? app.api.clip(app.rnn, CUDNN_RNN_CLIP_NONE, CUDNN_NOT_PROPAGATE_NAN, 0, 0)
                    : app.query(missing == 1   ? 0
                                : missing == 3 ? 2
                                               : 1,
                                missing == 1   ? 1
                                : missing == 3 ? 7
                                               : 2,
                                {&output[0].value, &output[1].value, &output[2].value});
            expect(status == CUDNN_STATUS_NOT_SUPPORTED,
                   "missing query/preflight/Clip callback is clean policy");
            for (const auto& value : output) value.verify(71);
            if (missing == 5) {
                for (const auto mask : {0U, 4U}) {
                    check(
                        app.query(1, mask, {&output[0].value, &output[1].value, &output[2].value}));
                    output[0].verify(71);
                    output[1].verify(71);
                    output[2].verify(mask ? 41 : 71);
                }
                expect(!fixture::control.gets.load(),
                       "missing private getter does not block zero-output or reserve-only Temp");
            }
            check(app.query(missing == 1 ? 2 : 0, missing == 1 ? 7 : 1,
                            {&output[0].value, &output[1].value, &output[2].value}));
        } else if (poison >= 0) {
            Guarded output;
            if (poison < 2) fixture::control.context_loss.store(poison + 1);
            if (poison == 2) fixture::control.corrupt.store(1);
            if (poison == 3) fixture::control.statuses[3].store(CUDNN_STATUS_BAD_PARAM);
            const auto status =
                poison == 3
                    ? app.api.clip(app.rnn, CUDNN_RNN_CLIP_MINMAX, CUDNN_PROPAGATE_NAN, -1, 1)
                    : app.api.temp(app.handle, app.rnn, CUDNN_FWD_MODE_INFERENCE, app.input,
                                   &output.value, nullptr);
            expect(status == (poison < 2    ? CUDNN_STATUS_NOT_INITIALIZED
                              : poison == 2 ? CUDNN_STATUS_INTERNAL_ERROR
                                            : CUDNN_STATUS_BAD_PARAM),
                   "physical proof or metadata uncertainty cannot expose original success");
            output.verify(71);
            const auto calls = fixture::calls();
            expect(app.api.weight(app.handle, app.rnn, &output.value) ==
                           CUDNN_STATUS_NOT_INITIALIZED &&
                       fixture::calls() == calls,
                   "quarantine rejects repeat locally without retry");
            if (poison >= 2) {
                check(cudnnDestroy(app.handle));
                driver(cuCtxDestroy(app.context));
            }
            return;
        } else {
            values(app);
            publication(app);
            exclusion(app);
            pending(app);
        }
        app.close();
    });
    const auto deadline = Clock::now() + std::chrono::seconds{20};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public RNN query progress");
    client.get();
    expect(!glimmer_dnn_rnn_query_native_calls(), "zero caller-native SDK setup/query/mutation");
    if (poison >= 0) {
        expect(!fixture::dropout::state.destroys && !fixture::control.synchronizes.load() &&
                   base::state.handle_destroys == (poison >= 2 ? 1U : 0U),
               "metadata-only poison preserves proven execution cleanup; lost context does not");
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished(), "retire ordinary service");
    expect(fixture::metadata::state.creates == 2 && fixture::metadata::state.destroys == 2 &&
               fixture::sequence::state.destroys == 1 && fixture::dropout::state.destroys == 1 &&
               base::state.handle_destroys == 1 && foundation::driver.contexts.empty() &&
               foundation::driver.memory.empty() && foundation::driver.events.empty() &&
               !fixture::control.synchronizes.load(),
           "exact ownership cleanup without query-induced synchronization");
}
struct Hostile {
    ExecutionWireReply reply;
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    std::size_t output{71};
    bool healthy{}, metadata_only{}, execution_only{}, disconnect{};
};
std::vector<std::byte> sizes(std::uint64_t first, std::uint64_t second = 0,
                             std::uint64_t third = 0) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_rnn_query_result(writer, {{first, second, third}}),
           "encode complete size receipt");
    return writer.take();
}
Hostile hostile(unsigned int index) {
    Hostile result;
    auto& reply = result.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    if (index < 28) {
        reply.data = sizes(17);
        switch (index) {
            case 0:
                reply.id = 7;
                break;
            case 1:
                reply.address = 7;
                break;
            case 2:
                reply.parameter_bytes = 1;
                reply.arguments = {{0, 1}};
                break;
            case 3:
                reply.parameter_bytes = 4;
                break;
            case 4:
            case 5:
            case 6:
                reply.parameter_bytes = index - 3;
                break;
            case 7:
                reply.original_status.reset();
                break;
            case 8:
                reply.original_status = CUDNN_STATUS_BAD_PARAM;
                break;
            case 9:
                reply.data.pop_back();
                break;
            case 10:
                reply.data.push_back(std::byte{1});
                break;
            case 11:
                reply.data = sizes(17, 29);
                break;
            case 12:
                reply.status = CUDNN_STATUS_BAD_PARAM;
                reply.original_status = reply.status;
                break;
            case 13:
            case 14:
            case 15:
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = index == 13 ? -1 : index == 14 ? 1024 : CUDA_ERROR_UNKNOWN;
                reply.data.clear();
                if (index != 15) reply.original_status.reset();
                break;
            case 16:
                reply.disposition = ExecutionWireDisposition::kComputeReplacement;
                reply.original_status.reset();
                reply.id = 7;
                reply.data.clear();
                break;
            case 17:
                result.disconnect = true;
                break;
            case 18:
                reply.status = 8192;
                reply.original_status = reply.status;
                break;
            case 19:
                reply.data.clear();
                break;
            case 20:
                reply.status = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
                reply.original_status = reply.status;
                reply.data.clear();
                result.status = CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE;
                result.healthy = true;
                break;
            case 21:
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = CUDA_ERROR_NOT_SUPPORTED;
                reply.original_status.reset();
                reply.data.clear();
                result.status = CUDNN_STATUS_NOT_SUPPORTED;
                result.healthy = true;
                break;
            case 22:
            case 23:
                result.output = index == 22 ? 0 : SIZE_MAX;
                reply.data = sizes(result.output);
                result.status = CUDNN_STATUS_SUCCESS;
                result.healthy = true;
                break;
            case 24:
            case 25:
            case 26:
            case 27:
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = index == 24 ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_CONTEXT;
                reply.parameter_bytes = index == 24 ? 1 : index == 26 ? 3 : 2;
                reply.data.clear();
                if (index == 24 || index == 27) reply.original_status.reset();
                if (index == 26) reply.original_status = CUDNN_STATUS_BAD_PARAM;
                result.status =
                    index == 24 ? CUDNN_STATUS_INTERNAL_ERROR : CUDNN_STATUS_NOT_INITIALIZED;
                result.metadata_only = index == 24;
                result.execution_only = index == 25 || index == 27;
                break;
            default:
                throw std::runtime_error("unknown query hostile case");
        }
    } else {
        switch (index - 28) {
            case 0:
                reply.id = 7;
                break;
            case 1:
                reply.address = 7;
                break;
            case 2:
                reply.data = {std::byte{1}};
                break;
            case 3:
                reply.parameter_bytes = 1;
                reply.arguments = {{0, 1}};
                break;
            case 4:
                reply.parameter_bytes = 1;
                break;
            case 5:
                reply.original_status.reset();
                break;
            case 6:
                reply.original_status = CUDNN_STATUS_BAD_PARAM;
                break;
            case 7:
            case 8:
                reply.status = CUDNN_STATUS_BAD_PARAM;
                reply.original_status = reply.status;
                if (index == 36) {
                    reply.parameter_bytes = 1;
                    result.status = CUDNN_STATUS_BAD_PARAM;
                }
                break;
            case 9:
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.original_status.reset();
                reply.status = CUDA_ERROR_NOT_SUPPORTED;
                result.status = CUDNN_STATUS_NOT_SUPPORTED;
                result.healthy = true;
                break;
            case 10:
                result.status = CUDNN_STATUS_SUCCESS;
                result.healthy = true;
                break;
            case 11:
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.original_status.reset();
                reply.status = -1;
                break;
            default:
                throw std::runtime_error("unknown Clip hostile case");
        }
        // Clip uses a host-only metadata receipt: even malformed Clip replies
        // cannot erase an unrelated execution owner's cleanup authority.
        result.metadata_only = !result.healthy;
    }
    return result;
}
void malformed_case(const char* provider, unsigned int index) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated hostile peer identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-rnn-query-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    const auto test = hostile(index);
    const bool is_clip = index >= 28;
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start hostile real IPC peer");
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnRNNDescriptor_t descriptor{};
        check(cudnnCreateRNNDescriptor(&descriptor));
        CUcontext context{};
        driver(cuCtxCreate(&context, 0, 0));
        cudnnHandle_t handle{};
        check(cudnnCreate(&handle));
        Guarded output;
        const auto status =
            is_clip ? api.clip(descriptor, CUDNN_RNN_CLIP_MINMAX, CUDNN_NOT_PROPAGATE_NAN, -0.0, 3)
                    : api.weight(handle, descriptor, &output.value);
        expect(status == test.status, "hostile receipt preserves precise error/control status");
        output.verify(is_clip ? 71 : test.output);
        if (test.healthy) {
            check(api.weight(handle, descriptor, &output.value));
            output.verify(61);
        } else {
            expect(api.weight(handle, descriptor, &output.value) == CUDNN_STATUS_NOT_INITIALIZED,
                   "poisoned query authority rejects locally without extra RPC");
            output.verify(is_clip ? 71 : test.output);
            if (test.execution_only)
                check(api.clip(descriptor, CUDNN_RNN_CLIP_NONE, CUDNN_NOT_PROPAGATE_NAN, 0, 0));
            else if (test.metadata_only)
                check(cudnnDestroy(handle));
            else
                expect(api.clip(descriptor, CUDNN_RNN_CLIP_NONE, CUDNN_NOT_PROPAGATE_NAN, 0, 0) ==
                           CUDNN_STATUS_NOT_INITIALIZED,
                       "malformed query receipt faults metadata as well as execution");
        }
        expect(!glimmer_dnn_rnn_query_native_calls(),
               "hostile peer never triggers caller-native fallback");
    });
    unsigned int metadata_hellos{}, context_hellos{}, creates{}, handle_creates{}, operations{},
        extra{};
    std::uint64_t metadata_connection{};
    bool disconnected{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline, "bounded hostile query progress");
        if (disconnected) {
            std::this_thread::yield();
            continue;
        }
        expect(server.poll_for(1), "poll hostile peer");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "preserve actual authenticated peer");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            const auto operation = request->message.header.operation;
            if (operation == ExecutionOpcode::kHello) {
                ExecutionBootstrapOwnership ownership;
                if (decode_execution_context_bootstrap(request->message.payload)) {
                    ++context_hellos;
                    ownership = ExecutionBootstrapOwnership::kRetained;
                } else {
                    std::uint32_t role{}, flags{};
                    expect(input.read_u32(role) && input.read_u32(flags) && input.empty() &&
                               role == 0x314d4e44U && flags == 0,
                           "exact DNM1 role/zero-flag bootstrap");
                    ++metadata_hellos;
                    metadata_connection = request->connection;
                    ownership = ExecutionBootstrapOwnership::kMetadataRetained;
                }
                const auto encoded = encode_execution_bootstrap_ownership(ownership);
                expect(encoded.has_value(), "encode bootstrap owner");
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && input.empty() &&
                           kind == static_cast<std::uint32_t>(ExecutionDnnObjectKind::kRnn),
                       "hostile fixture creates only exact RNN kind");
                ++creates;
                reply.id = 17;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (operation == ExecutionOpcode::kCreateDnn) {
                expect(input.empty(), "empty original-handle creation");
                ++handle_creates;
                reply.id = 19;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (operation == ExecutionOpcode::kDestroyDnn) {
                std::uint64_t handle{};
                expect(test.metadata_only && input.read_u64(handle) && input.empty() &&
                           handle == 19 && !extra++,
                       "only metadata poison permits proven handle cleanup probe");
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else {
                if (operation == ExecutionOpcode::kDnnRnnQuery) {
                    ExecutionDnnRnnQuery query;
                    expect(read_execution_dnn_rnn_query(input, query) && input.empty() &&
                               query.kind == ExecutionDnnRnnQueryKind::kWeightSpace &&
                               query.handle == 19 && query.metadata == metadata_connection &&
                               query.descriptor == 17 && !query.input && !query.mode &&
                               query.outputs == 1 && request->message.payload.size() == 44,
                           "precise handle/metadata/descriptor query authority");
                } else {
                    std::uint64_t descriptor{};
                    ExecutionDnnRnnClip clip;
                    expect(operation == ExecutionOpcode::kSetDnnRnnClip &&
                               input.read_u64(descriptor) && descriptor == 17 &&
                               read_execution_dnn_rnn_clip(input, clip) && input.empty(),
                           "precise Clip request");
                    if (!operations)
                        expect(is_clip && clip.lower_bits == UINT64_C(0x8000000000000000),
                               "raw Clip negative zero reaches wire");
                    else
                        expect(test.execution_only && !extra++,
                               "only execution-only poison permits metadata Clip probe");
                }
                ++operations;
                if (operations == 1) {
                    if (test.disconnect) {
                        server.stop();
                        disconnected = true;
                        continue;
                    }
                    reply = test.reply;
                } else {
                    expect(test.healthy || test.execution_only,
                           "no unauthorized follow-up after quarantine");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    if (operation == ExecutionOpcode::kDnnRnnQuery) reply.data = sizes(61);
                }
            }
            const auto payload = encode_execution_reply(reply);
            expect(payload.has_value() && decode_execution_reply(*payload).has_value(),
                   "hostile envelope remains wire-valid");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *payload}),
                   "send hostile/control receipt");
        }
    }
    client.get();
    profiles.verify();
    expect(metadata_hellos == 1 && context_hellos == 1 && creates == 1 && handle_creates == 1 &&
               operations == (test.healthy || test.execution_only ? 2U : 1U) &&
               extra == (test.metadata_only || test.execution_only ? 1U : 0U),
           "exact bootstrap/call counts prove no hidden fallback, retry or poisoned RPC");
    server.stop();
}
void isolated(const char* executable, const char* provider, std::string_view mode) {
    const auto count = mode == "--malformed-replies" ? kHostileCases : mode == "--poison" ? 4U : 5U;
    const auto selected = mode == "--malformed-replies" ? "--malformed-case"
                          : mode == "--poison"          ? "--poison-case"
                                                        : "--missing-case";
    for (unsigned int i = 0; i != count; ++i) {
        const auto text = std::to_string(i);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated public query case");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider, selected, text.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error(std::string(selected) + " failed: " + text);
    }
}
unsigned int index_argument(const char* argument) {
    const std::string_view text{argument};
    expect(!text.empty() && text.size() <= 2, "bounded isolated index");
    unsigned int value{};
    for (const char digit : text) {
        expect(digit >= '0' && digit <= '9', "decimal case index");
        value = value * 10 + static_cast<unsigned int>(digit - '0');
    }
    return value;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider path and optional public test mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--missing-callback" || mode == "--poison") {
            expect(argc == 3, "parent mode requires no index");
            isolated(argv[0], argv[1], mode);
        } else if (mode == "--malformed-case" || mode == "--missing-case" ||
                   mode == "--poison-case") {
            expect(argc == 4, "isolated mode requires index");
            const auto index = index_argument(argv[3]);
            if (mode == "--malformed-case") {
                expect(index < kHostileCases, "known hostile case");
                malformed_case(argv[1], index);
                ::_exit(0);
            }
            expect(index < (mode == "--missing-case" ? 5U : 4U), "known native error case");
            run(argv[1], ExecutionSchedulingMode::kNaive, mode == "--missing-case" ? index + 1 : 0,
                mode == "--poison-case" ? static_cast<int>(index) : -1);
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known scheduling mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "RNN size-query and Clip CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.release.store(true);
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
