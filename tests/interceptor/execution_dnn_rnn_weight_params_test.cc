#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_rnn_weight_params_fixture.h"

#include <cerrno>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_rnn_weight_params_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_rnn_weight_params;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
using Clock = std::chrono::steady_clock;
using Api = decltype(&cudnnGetRNNWeightParams);
constexpr unsigned int kHostileCases = 30;
void expect(bool value, const char* text) {
    if (!value) throw std::runtime_error(text);
}
void check(cudnnStatus_t status) {
    expect(status == CUDNN_STATUS_SUCCESS, "public original SDK success");
}
void driver(CUresult status) {
    expect(status == CUDA_SUCCESS, "public owned Driver success");
}
Api lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent WeightParams provider");
    auto* symbol = ::dlsym(library, "cudnnGetRNNWeightParams");
    expect(symbol == reinterpret_cast<void*>(&cudnnGetRNNWeightParams) &&
               symbol == ::dlsym(RTLD_DEFAULT, "cudnnGetRNNWeightParams"),
           "direct/global/explicit provider lookup is controlled");
    expect(::dlclose(library) == 0, "close provider while retaining controlled cached pointer");
    return reinterpret_cast<Api>(symbol);
}
struct Guarded {
    std::uint64_t before{0x123456789abcdef0ULL};
    void* value{reinterpret_cast<void*>(0x718U)};
    std::uint64_t after{0xfedcba9876543210ULL};
    void verify(std::uintptr_t expected = 0x718U) const {
        expect(before == 0x123456789abcdef0ULL && after == 0xfedcba9876543210ULL &&
                   reinterpret_cast<std::uintptr_t>(value) == expected,
               "pointer output and full adjacent guards remain exact");
    }
};
struct Application {
    Api api;
    cudnnRNNDescriptor_t rnn{};
    cudnnDropoutDescriptor_t dropout{};
    std::array<cudnnTensorDescriptor_t, 2> destinations{};
    CUcontext context{};
    cudnnHandle_t handle{};
    CUdeviceptr allocation{};
    explicit Application(const char* provider) : api(lookup(provider)) {
        check(cudnnCreateRNNDescriptor(&rnn));
        check(cudnnCreateDropoutDescriptor(&dropout));
        configure();
        const int sizes[]{1, 1, 1}, strides[]{1, 1, 1};
        for (auto& tensor : destinations) {
            check(cudnnCreateTensorDescriptor(&tensor));
            check(cudnnSetTensorNdDescriptor(tensor, CUDNN_DATA_FLOAT, 3, sizes, strides));
        }
        driver(cuCtxCreate(&context, 0, 0));
        check(cudnnCreate(&handle));
        driver(cuMemAlloc(&allocation, 544));
    }
    void configure() const {
        check(cudnnSetRNNDescriptor_v8(rnn, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM,
                                       CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                       CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                       CUDNN_DEFAULT_MATH, 2, 2, 2, 1, dropout, 0));
    }
    cudnnStatus_t query(std::array<void**, 2> output, unsigned int mask = 3, bool raw = false,
                        int linear = 0, std::size_t bytes = 512) const {
        return api(handle, rnn, 0, bytes, raw ? nullptr : reinterpret_cast<void*>(allocation + 16),
                   linear, mask & 1 ? destinations[0] : nullptr, output[0],
                   mask & 2 ? destinations[1] : nullptr, output[1]);
    }
    void close() {
        driver(cuMemFree(allocation));
        check(cudnnDestroy(handle));
        for (const auto tensor : destinations) check(cudnnDestroyTensorDescriptor(tensor));
        check(cudnnDestroyRNNDescriptor(rnn));
        check(cudnnDestroyDropoutDescriptor(dropout));
        driver(cuCtxDestroy(context));
    }
};
void wait_flag(const std::atomic<bool>& flag) {
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (!flag.load()) {
        expect(Clock::now() < deadline, "bounded started/callback handshake");
        std::this_thread::yield();
    }
}
void shape(cudnnTensorDescriptor_t tensor, int last) {
    cudnnDataType_t type = CUDNN_DATA_DOUBLE;
    int rank = -1;
    std::array<int, 4> sizes{-7, -7, -7, -7}, strides{-9, -9, -9, -9};
    check(cudnnGetTensorNdDescriptor(tensor, 3, &type, &rank, sizes.data(), strides.data()));
    expect(type == CUDNN_DATA_FLOAT && rank == 3 && sizes == std::array<int, 4>{1, 2, last, -7} &&
               strides == std::array<int, 4>{2 * last, last, 1, -9},
           "original destination descriptor mutation and query guards are exact");
}
void ordinary(Application& app) {
    for (unsigned int mask = 0; mask != 4; ++mask) {
        std::array<Guarded, 2> output;
        check(app.query(
            {mask & 1 ? &output[0].value : nullptr, mask & 2 ? &output[1].value : nullptr}, mask));
        output[0].verify(mask & 1 ? app.allocation + 16 : 0x718U);
        output[1].verify(mask & 2 ? app.allocation + 272 : 0x718U);
    }
    shape(app.destinations[0], 2);
    shape(app.destinations[1], 1);
    for (int linear = 0; linear != 8; ++linear) {
        std::array<Guarded, 2> output;
        check(app.query({&output[0].value, &output[1].value}, 3, true, linear));
        output[0].verify(32U * linear);
        output[1].verify(256U + 16U * linear);
    }
    Guarded alias;
    check(app.query({&alias.value, &alias.value}));
    alias.verify(app.allocation + 272);  // Native matrix then bias pointer publication.
    std::array<Guarded, 2> output;
    check(app.api(app.handle, app.rnn, 0, 512, reinterpret_cast<void*>(app.allocation + 16), 0,
                  app.destinations[0], &output[0].value, app.destinations[0], &output[1].value));
    output[0].verify(app.allocation + 16);
    output[1].verify(app.allocation + 272);
    shape(app.destinations[0], 1);  // Native bias mutation wins the descriptor alias too.
    check(app.query({&alias.value, &alias.value}, 3, true, 8, 0));
    alias.verify(0);
    check(app.api(app.handle, app.rnn, 0, 0, reinterpret_cast<void*>(app.allocation + 544), 8,
                  app.destinations[0], &output[0].value, app.destinations[1], &output[1].value));
    output[0].verify(0);
    output[1].verify(0);
    check(
        app.api(app.handle, app.rnn, 0, SIZE_MAX, nullptr, 8, nullptr, nullptr, nullptr, nullptr));
    const auto calls = fixture::control.calls.load();
    expect(app.query({nullptr, &alias.value}) == CUDNN_STATUS_BAD_PARAM &&
               app.api(app.handle, app.rnn, 0, SIZE_MAX, reinterpret_cast<void*>(app.allocation), 0,
                       app.destinations[0], &alias.value, nullptr,
                       nullptr) == CUDNN_STATUS_BAD_PARAM &&
               fixture::control.calls.load() == calls,
           "unsafe native output-null pair and oversized owned interval reject before SDK");
    alias.verify(0);
    expect(app.api(app.handle, reinterpret_cast<cudnnRNNDescriptor_t>(app.destinations[0]), 0, 512,
                   nullptr, 0, app.destinations[0], &alias.value, nullptr,
                   nullptr) == CUDNN_STATUS_NOT_INITIALIZED,
           "wrong typed identity is not native authority");
    CUcontext other{};
    driver(cuCtxCreate(&other, 0, 0));
    expect(app.query({&alias.value, &alias.value}) == CUDNN_STATUS_NOT_INITIALIZED,
           "handle from another current context is rejected");
    driver(cuCtxDestroy(other));
    driver(cuCtxSetCurrent(app.context));
    const auto child = ::fork();
    expect(child >= 0, "fork inherited query identity");
    if (!child)
        ::_exit(app.query({&alias.value, &alias.value}) == CUDNN_STATUS_NOT_INITIALIZED ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork cannot reuse original descriptor or allocation authority");
    driver(cuMemFree(app.allocation));
    expect(app.query({&alias.value, &alias.value}) == CUDNN_STATUS_BAD_PARAM,
           "freed allocation is not made live by a saved native offset");
    driver(cuMemAlloc(&app.allocation, 544));
}
void publication(Application& app) {
    auto* page = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(page != MAP_FAILED, "map hostile output page");
    Guarded first;
    const auto calls = fixture::control.calls.load();
    expect(app.query({&first.value, static_cast<void**>(page)}) == CUDNN_STATUS_BAD_PARAM &&
               fixture::control.calls.load() == calls,
           "all output spans checked before invoking original");
    first.verify();
    expect(::mprotect(page, 4096, PROT_READ | PROT_WRITE) == 0, "temporarily writable output");
    fixture::control.pause.store(true);
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    auto result = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        const auto status = app.query({&first.value, static_cast<void**>(page)});
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    wait_flag(fixture::control.entered);
    const auto protection = ::mprotect(page, 4096, PROT_NONE);
    fixture::control.release.store(true);
    expect(result.get() == CUDNN_STATUS_BAD_PARAM && protection == 0,
           "post-RPC all-span check rejects local output loss without partial publication");
    first.verify();
    expect(::munmap(page, 4096) == 0, "release hostile page");
    shape(app.destinations[0], 2);
    shape(app.destinations[1], 1);
    check(app.query({&first.value, &first.value}));
    first.verify(app.allocation + 272);
}
void exclusion(Application& app) {
    fixture::control.pause.store(true);
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    auto query = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        void* pointer{};
        const auto status = app.query({&pointer, &pointer});
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    wait_flag(fixture::control.entered);
    std::atomic<bool> started{};
    auto mutation = std::async(std::launch::async, [&] {
        started.store(true);
        app.configure();
    });
    wait_flag(started);
    const bool blocked =
        mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    fixture::control.release.store(true);
    check(query.get());
    mutation.get();
    expect(blocked, "parent mutation and active query are serialized without partial-pin deadlock");
    // Reciprocal mutable destinations must acquire all unique pins atomically.
    auto swapped = [&](bool reverse) {
        driver(cuCtxSetCurrent(app.context));
        void* pointers[2]{};
        const auto status =
            app.api(app.handle, app.rnn, 0, 512, nullptr, 0, app.destinations[reverse ? 1 : 0],
                    &pointers[0], app.destinations[reverse ? 0 : 1], &pointers[1]);
        driver(cuCtxSetCurrent(nullptr));
        return status;
    };
    auto first = std::async(std::launch::async, [&] { return swapped(false); });
    auto second = std::async(std::launch::async, [&] { return swapped(true); });
    check(first.get());
    check(second.get());
}
void pending(Application& app) {
    const int sizes[]{1, 1, 1}, strides[]{1, 1, 1};
    check(cudnnSetTensorNdDescriptor(app.destinations[0], CUDNN_DATA_FLOAT, 3, sizes, strides));
    std::array<CUdeviceptr, 2> buffers{};
    for (auto& buffer : buffers) driver(cuMemAlloc(&buffer, 4));
    const float alpha = 1, beta = 0;
    foundation::control.can_stop.store(false);
    expect(cudnnAddTensor(app.handle, &alpha, app.destinations[0],
                          reinterpret_cast<void*>(buffers[0]), &beta, app.destinations[0],
                          reinterpret_cast<void*>(buffers[1])) == CUDNN_STATUS_EXECUTION_FAILED,
           "prior original error retains physical event");
    Guarded output;
    const auto syncs = fixture::query::control.synchronizes.load();
    check(app.api(app.handle, app.rnn, 0, 512, nullptr, 0, app.destinations[1], &output.value,
                  app.destinations[1], &output.value));
    output.verify(256);
    std::atomic<bool> started{};
    auto wait = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        started.store(true);
        void* address{};
        const auto status = app.query({&address, &address});
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    wait_flag(started);
    const auto before = fixture::control.calls.load();
    const bool blocked = wait.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    expect(!foundation::control.can_stop.load() &&
               fixture::query::control.synchronizes.load() == syncs &&
               fixture::control.calls.load() == before,
           "independent host query does not synchronize; borrowed destination stays unissued");
    foundation::control.can_stop.store(true);
    check(wait.get());
    expect(blocked, "host ACK cannot release destination before physical completion");
    for (const auto buffer : buffers) driver(cuMemFree(buffer));
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated local peer");
}
void run(const char* provider, ExecutionSchedulingMode mode, bool missing = false,
         int poison = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "real service peer identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-weight-params-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch(missing));
    expect(service.start(), "start owned WeightParams service");
    client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (missing) {
            Guarded output;
            expect(app.query({&output.value, &output.value}) == CUDNN_STATUS_NOT_SUPPORTED,
                   "missing original callback reports clean NOT_SUPPORTED");
            output.verify();
            app.configure();
        } else if (poison >= 0) {
            std::array<Guarded, 2> output;
            if (poison < 4) fixture::control.corrupt.store(poison + 1);
            if (poison == 4) fixture::control.status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
            if (poison == 7) fixture::control.context_loss.store(true);
            const auto status = app.query({&output[0].value, &output[1].value}, poison == 6 ? 1 : 3,
                                          false, 0, poison == 5 ? 16 : 512);
            expect(status == (poison < 4    ? CUDNN_STATUS_INTERNAL_ERROR
                              : poison == 4 ? CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE
                              : poison == 7 ? CUDNN_STATUS_NOT_INITIALIZED
                                            : CUDNN_STATUS_BAD_PARAM),
                   "exact original failure or containment/context policy reaches caller");
            for (const auto& value : output) value.verify();
            const auto calls = fixture::control.calls.load();
            expect(
                app.query({&output[0].value, &output[1].value}) == CUDNN_STATUS_NOT_INITIALIZED &&
                    fixture::control.calls.load() == calls,
                "uncertain mutable descriptors never retry or publish");
            if (poison != 7) {
                driver(cuMemFree(app.allocation));
                check(cudnnDestroy(app.handle));
                driver(cuCtxDestroy(app.context));
            }
            return;
        } else {
            ordinary(app);
            publication(app);
            exclusion(app);
            pending(app);
        }
        app.close();
    });
    const auto deadline = Clock::now() + std::chrono::seconds{20};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded WeightParams IPC progress");
    client.get();
    expect(!glimmer_dnn_rnn_weight_params_native_calls(), "zero native caller fallback");
    if (poison >= 0) {
        expect(base::state.handle_destroys == (poison == 7 ? 0U : 1U) &&
                   !fixture::query::control.synchronizes.load(),
               "metadata-only failure permits proven cleanup; context loss does not");
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished(), "ordinary service retires");
    expect(fixture::metadata::state.destroys == 1 && fixture::dropout::state.destroys == 1 &&
               base::state.handle_destroys == 1 && foundation::driver.contexts.empty() &&
               foundation::driver.memory.empty() && foundation::driver.events.empty() &&
               !fixture::query::control.synchronizes.load(),
           "original owners and complete allocations retire once without query-induced sync");
}

// Hostile wire tests below use raw null-base mode. The ordinary service tests
// above separately establish reconstruction against an actually owned base.
struct Hostile {
    ExecutionWireReply reply;
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    bool healthy{}, metadata_only{}, execution_only{};
    std::array<std::uintptr_t, 2> expected{0x718, 0x718};
};
std::vector<std::byte> offsets(std::uint32_t mask = 2, std::uint64_t first = 0,
                               std::uint64_t second = 256) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(mask) && writer.write_u64(first) && writer.write_u64(second),
           "encode exact raw offset fixture");
    return std::move(writer).take();
}
Hostile hostile(unsigned int index) {
    Hostile result;
    auto& reply = result.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    reply.data = offsets();
    switch (index) {
        case 0:
            reply.id = 7;
            break;
        case 1:
            reply.address = 7;
            break;
        case 2:
            reply.parameter_bytes = 4;
            break;
        case 3:
            reply.parameter_bytes = 1;
            reply.arguments = {{0, 1}};
            break;
        case 4:
            reply.parameter_bytes = 1;
            break;
        case 5:
            reply.parameter_bytes = 2;
            break;
        case 6:
            reply.parameter_bytes = 3;
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
            reply.data = offsets(4, 0, 0);
            break;
        case 12:
            reply.data = offsets(2, 1, 256);
            break;
        case 13:
            reply.data = offsets(3, 0, 256);
            break;  // Null-base nonnull offset zero is impossible.
        case 14:
            reply.data = offsets(2, 0, 512);
            break;
        case 15:
            reply.data = offsets(2, 0, UINT64_MAX);
            break;
        case 16:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.data.clear();
            break;  // Issued errors must quarantine metadata.
        case 17:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.parameter_bytes = 1;
            break;  // Native error cannot carry successful offsets.
        case 18:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.parameter_bytes = 1;
            reply.data.clear();
            result.status = CUDNN_STATUS_BAD_PARAM;
            result.metadata_only = true;
            break;
        case 19:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            reply.original_status.reset();
            reply.data.clear();
            result.status = CUDNN_STATUS_NOT_SUPPORTED;
            result.healthy = true;
            break;
        case 20:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes = 1;
            reply.data.clear();
            result.metadata_only = true;
            break;
        case 21:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 3;
            reply.data.clear();
            result.status = CUDNN_STATUS_NOT_INITIALIZED;
            break;
        case 22:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 2;
            reply.original_status.reset();
            reply.data.clear();
            result.status = CUDNN_STATUS_NOT_INITIALIZED;
            result.execution_only = true;
            break;
        case 23:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = -1;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 24:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 1024;
            reply.original_status.reset();
            reply.data.clear();
            break;
        case 25:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes = 1;
            reply.data.clear();
            reply.original_status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 26:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.data.clear();
            break;
        case 27:
            result.status = CUDNN_STATUS_SUCCESS;
            result.healthy = true;
            result.expected = {0, 256};
            break;
        case 28:
            reply.data = offsets(0, 0, 0);
            result.status = CUDNN_STATUS_SUCCESS;
            result.healthy = true;
            result.expected = {0, 0};
            break;
        case 29:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 0;
            reply.original_status.reset();
            reply.data.clear();
            break;
        default:
            throw std::runtime_error("unknown hostile case");
    }
    return result;
}
void malformed_case(const char* provider, unsigned int index) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated hostile peer identity");
    ExecutionSocketConfig config;
    config.socket_path =
        "/tmp/glimmer-weight-params-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    const auto test = hostile(index);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start hostile real IPC peer");
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnRNNDescriptor_t parent{};
        std::array<cudnnTensorDescriptor_t, 2> tensors{};
        check(cudnnCreateRNNDescriptor(&parent));
        for (auto& tensor : tensors) check(cudnnCreateTensorDescriptor(&tensor));
        CUcontext context{};
        driver(cuCtxCreate(&context, 0, 0));
        cudnnHandle_t handle{};
        check(cudnnCreate(&handle));
        std::array<Guarded, 2> output;
        auto invoke = [&] {
            return api(handle, parent, 0, 512, nullptr, 0, tensors[0], &output[0].value, tensors[1],
                       &output[1].value);
        };
        expect(invoke() == test.status, "hostile receipt preserves precise status contract");
        for (std::size_t i = 0; i != 2; ++i) output[i].verify(test.expected[i]);
        if (test.healthy) {
            check(invoke());
            output[0].verify(0);
            output[1].verify(256);
        } else {
            expect(invoke() == CUDNN_STATUS_NOT_INITIALIZED,
                   "poisoned owners reject locally, no extra RPC");
            for (std::size_t i = 0; i != 2; ++i) output[i].verify(test.expected[i]);
            if (test.metadata_only)
                check(cudnnDestroy(handle));
            else if (test.execution_only)
                check(cudnnDestroyTensorDescriptor(tensors[0]));
            else
                expect(cudnnDestroyTensorDescriptor(tensors[0]) == CUDNN_STATUS_NOT_INITIALIZED,
                       "malformed receipt faults metadata as well as execution");
        }
        expect(!glimmer_dnn_rnn_weight_params_native_calls(),
               "hostile transport never falls back natively");
    });
    unsigned int hellos{}, contexts{}, creates{}, handles{}, queries{}, probes{};
    std::uint64_t metadata_connection{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded hostile peer progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "actual authenticated peer preserved");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            const auto operation = request->message.header.operation;
            if (operation == ExecutionOpcode::kHello) {
                auto ownership = ExecutionBootstrapOwnership::kRetained;
                if (decode_execution_context_bootstrap(request->message.payload))
                    ++contexts;
                else {
                    std::uint32_t role{}, flags{};
                    expect(input.read_u32(role) && input.read_u32(flags) && input.empty() &&
                               role == 0x314d4e44U && !flags,
                           "exact DNM1 bootstrap contract");
                    ++hellos;
                    metadata_connection = request->connection;
                    ownership = ExecutionBootstrapOwnership::kMetadataRetained;
                }
                const auto encoded = encode_execution_bootstrap_ownership(ownership);
                expect(encoded.has_value(), "encode actual bootstrap owner");
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && input.empty() && kind == (creates ? 1U : 13U),
                       "exact typed RNN then tensor creations");
                reply.id = 17 + creates++;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (operation == ExecutionOpcode::kCreateDnn) {
                expect(input.empty(), "empty original-handle creation");
                ++handles;
                reply.id = 23;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (operation == ExecutionOpcode::kDnnRnnWeightParams) {
                ExecutionDnnRnnWeightParams value;
                expect(read_execution_dnn_rnn_weight_params(input, value) && input.empty() &&
                           request->message.payload.size() == 76 && value.handle == 23 &&
                           value.metadata == metadata_connection && value.descriptor == 17 &&
                           !value.pseudo_layer && !value.linear_layer &&
                           !value.weights.allocation && !value.weights.offset &&
                           value.weight_bytes == 512 &&
                           value.destinations == std::array<std::uint64_t, 2>{18, 19} &&
                           value.outputs == 3,
                       "precise context/metadata/descriptor/raw-offset query");
                if (!queries++)
                    reply = test.reply;
                else {
                    expect(test.healthy, "no poisoned retry reaches transport");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    reply.data = offsets();
                }
            } else if (operation == ExecutionOpcode::kDestroyDnn) {
                std::uint64_t handle{};
                expect(test.metadata_only && !probes++ && input.read_u64(handle) && handle == 23 &&
                           input.empty(),
                       "metadata-only quarantine leaves proven handle cleanup");
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else {
                std::uint64_t id{};
                expect(operation == ExecutionOpcode::kDestroyDnnMetadata && test.execution_only &&
                           !probes++ && input.read_u64(id) && id == 18 && input.empty(),
                       "execution-only quarantine leaves healthy metadata cleanup");
                reply.original_status = CUDNN_STATUS_SUCCESS;
            }
            const auto payload = encode_execution_reply(reply);
            expect(payload && decode_execution_reply(*payload).has_value(),
                   "hostile envelope is wire-valid");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *payload}),
                   "send hostile/control reply");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && contexts == 1 && creates == 3 && handles == 1 &&
               queries == (test.healthy ? 2U : 1U) &&
               probes == (test.metadata_only || test.execution_only ? 1U : 0U),
           "exact RPC counts prove no hidden fallback or poisoned submission");
    server.stop();
}
void isolated(const char* executable, const char* provider, bool malformed) {
    const auto mode = malformed ? "--malformed-case" : "--poison-case";
    for (unsigned int i = 0; i != (malformed ? kHostileCases : 8U); ++i) {
        const auto text = std::to_string(i);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated mutation/receipt case");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider, mode, text.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error(std::string(mode) + " failed: " + text);
    }
}
unsigned int case_index(const char* text) {
    unsigned int index{};
    const std::string_view value{text};
    expect(!value.empty() && value.size() < 3, "bounded case argument");
    for (const char digit : value) {
        expect(digit >= '0' && digit <= '9', "decimal case argument");
        index = index * 10 + static_cast<unsigned int>(digit - '0');
    }
    return index;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--poison") {
            expect(argc == 3, "parent mode takes no index");
            isolated(argv[0], argv[1], mode == "--malformed-replies");
        } else if (mode == "--malformed-case" || mode == "--poison-case") {
            expect(argc == 4, "isolated mode takes index");
            const auto index = case_index(argv[3]);
            expect(index < (mode == "--malformed-case" ? kHostileCases : 8U), "known case");
            if (mode == "--malformed-case") {
                malformed_case(argv[1], index);
                ::_exit(0);
            }
            run(argv[1], ExecutionSchedulingMode::kNaive, false, static_cast<int>(index));
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority" ||
                                 mode == "--missing-callback"),
                   "known mode");
            run(argv[1],
                mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                       : ExecutionSchedulingMode::kNaive,
                mode == "--missing-callback");
        }
        std::cout << "WeightParams CPU/local-IPC mapping checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.release.store(true);
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
