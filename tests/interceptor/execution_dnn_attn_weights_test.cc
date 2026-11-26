#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_attn_weights_fixture.h"

#include <cerrno>
#include <cstdlib>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_attn_weights_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_attn_weights;
namespace foundation = fixture::foundation;
namespace base = fixture::base;
using Clock = std::chrono::steady_clock;
using Api = decltype(&cudnnGetMultiHeadAttnWeights);
constexpr unsigned int kHostileCases = 34;
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
    expect(library, "load independent attention-layout provider");
    auto* symbol = ::dlsym(library, "cudnnGetMultiHeadAttnWeights");
    expect(symbol == reinterpret_cast<void*>(&cudnnGetMultiHeadAttnWeights) &&
               symbol == ::dlsym(RTLD_DEFAULT, "cudnnGetMultiHeadAttnWeights"),
           "direct/global/explicit-provider lookup is controlled");
    expect(!::dlclose(library), "cached controlled pointer survives provider close");
    return reinterpret_cast<Api>(symbol);
}
struct Guarded {
    std::uint64_t before{0x123456789abcdef0ULL};
    void* value{reinterpret_cast<void*>(0x718U)};
    std::uint64_t after{0xfedcba9876543210ULL};
    void verify(std::uintptr_t expected = 0x718U) const {
        expect(before == 0x123456789abcdef0ULL && after == 0xfedcba9876543210ULL &&
                   reinterpret_cast<std::uintptr_t>(value) == expected,
               "complete pointer output and adjacent guards remain exact");
    }
};
struct Application {
    Api api;
    cudnnAttnDescriptor_t parent{};
    std::array<cudnnDropoutDescriptor_t, 2> dropouts{};
    std::array<cudnnTensorDescriptor_t, 2> destinations{};
    CUcontext context{};
    cudnnHandle_t handle{};
    explicit Application(const char* provider) : api(lookup(provider)) {
        check(cudnnCreateAttnDescriptor(&parent));
        for (auto& dropout : dropouts) check(cudnnCreateDropoutDescriptor(&dropout));
        configure();
        const int sizes[]{1, 1, 1}, strides[]{1, 1, 1};
        for (auto& tensor : destinations) {
            check(cudnnCreateTensorDescriptor(&tensor));
            check(cudnnSetTensorNdDescriptor(tensor, CUDNN_DATA_FLOAT, 3, sizes, strides));
        }
        driver(cuCtxCreate(&context, 0, 0));
        check(cudnnCreate(&handle));
    }
    void configure(bool projected = true, bool alias = false) const {
        const auto size = projected ? 2 : 0;
        check(cudnnSetAttnDescriptor(parent, CUDNN_ATTN_ENABLE_PROJ_BIASES, 1, .75,
                                     CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH,
                                     dropouts[0], dropouts[alias ? 0 : 1], 2, 2, 2, size, size,
                                     size, size, 2, 3, 1, 1));
    }
    cudnnStatus_t query(void** output, const void* weights = reinterpret_cast<void*>(0x10000),
                        unsigned int kind = 0, std::size_t bytes = 512,
                        unsigned int target = 0) const {
        return api(handle, parent, static_cast<cudnnMultiHeadAttnWeightKind_t>(kind), bytes,
                   weights, destinations[target], output);
    }
    void close() {
        check(cudnnDestroy(handle));
        for (const auto tensor : destinations) check(cudnnDestroyTensorDescriptor(tensor));
        check(cudnnDestroyAttnDescriptor(parent));
        for (const auto dropout : dropouts) check(cudnnDestroyDropoutDescriptor(dropout));
        driver(cuCtxDestroy(context));
    }
};
void wait_flag(const std::atomic<bool>& flag) {
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (!flag.load()) {
        expect(Clock::now() < deadline, "bounded callback/started handshake");
        std::this_thread::yield();
    }
}
void shape(cudnnTensorDescriptor_t tensor, int last) {
    cudnnDataType_t type = CUDNN_DATA_DOUBLE;
    int rank = -1;
    std::array<int, 4> sizes{-7, -7, -7, -7}, strides{-9, -9, -9, -9};
    check(cudnnGetTensorNdDescriptor(tensor, 3, &type, &rank, sizes.data(), strides.data()));
    expect(type == CUDNN_DATA_FLOAT && rank == 3 && sizes == std::array<int, 4>{1, 2, last, -7} &&
               strides == std::array<int, 4>{2, 1, 2, -9},
           "original non-row-major destination metadata and trailing guards preserved");
}
void ordinary(Application& app) {
    std::array<std::byte, 512> host;
    host.fill(std::byte{0xa5});
    auto* inaccessible = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(inaccessible != MAP_FAILED, "map unreadable host base");
    CUdeviceptr memory{};
    driver(cuMemAlloc(&memory, 512));
    const std::array<const void*, 5> bases{host.data(), inaccessible, reinterpret_cast<void*>(1),
                                           reinterpret_cast<void*>(0x1000000000ULL),
                                           reinterpret_cast<void*>(memory)};
    for (const auto address : bases)
        for (unsigned int kind = 0; kind != 8; ++kind) {
            Guarded output;
            check(app.query(&output.value, address, kind));
            output.verify(reinterpret_cast<std::uintptr_t>(address) +
                          (kind < 4 ? 32U * kind : 192U + 16U * (kind - 4)));
            shape(app.destinations[0], kind < 4 ? 2 : 1);
        }
    expect(std::ranges::all_of(host, [](auto value) { return value == std::byte{0xa5}; }),
           "numeric helper never reads or changes complete host payload");
    driver(cuMemFree(memory));
    Guarded output;
    check(app.query(&output.value, reinterpret_cast<void*>(memory), 1));
    output.verify(memory + 32);
    expect(cuMemFree(reinterpret_cast<CUdeviceptr>(output.value)) == CUDA_ERROR_INVALID_VALUE,
           "stale numeric base remains arithmetic but result grants no allocation authority");
    expect(!::munmap(inaccessible, 4096), "release unreadable host mapping");
    const auto calls = fixture::control.calls.load();
    expect(app.query(&output.value, reinterpret_cast<void*>(UINTPTR_MAX - 7)) ==
                   CUDNN_STATUS_BAD_PARAM &&
               app.query(&output.value, reinterpret_cast<void*>(1), 8) == CUDNN_STATUS_BAD_PARAM &&
               fixture::control.calls.load() == calls,
           "overflow/kind reject before original");
    output.verify(memory + 32);
    app.configure(true, true);
    check(app.query(&output.value));
    app.configure(false);
    for (const void* address : std::array<const void*, 2>{nullptr, reinterpret_cast<void*>(1)}) {
        check(app.query(&output.value, address, 0, 0));
        output.verify(0);
    }
    app.configure();
    expect(app.api(app.handle, reinterpret_cast<cudnnAttnDescriptor_t>(app.destinations[0]),
                   CUDNN_MH_ATTN_Q_WEIGHTS, 512, host.data(), app.destinations[0],
                   &output.value) == CUDNN_STATUS_NOT_INITIALIZED,
           "typed descriptor authority is not an address");
    CUcontext other{};
    driver(cuCtxCreate(&other, 0, 0));
    expect(app.query(&output.value) == CUDNN_STATUS_NOT_INITIALIZED,
           "cross-context handle rejected");
    driver(cuCtxDestroy(other));
    driver(cuCtxSetCurrent(app.context));
    const auto child = ::fork();
    expect(child >= 0, "fork inherited caller identity");
    if (!child) ::_exit(app.query(&output.value) == CUDNN_STATUS_NOT_INITIALIZED ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork cannot inherit metadata/execution authority");
}
void publication(Application& app) {
    auto* page = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(page != MAP_FAILED, "map inaccessible output");
    const auto calls = fixture::control.calls.load();
    expect(app.query(static_cast<void**>(page)) == CUDNN_STATUS_BAD_PARAM &&
               fixture::control.calls.load() == calls,
           "unwritable output rejected before SDK");
    expect(!::mprotect(page, 4096, PROT_READ | PROT_WRITE), "temporarily writable output");
    *static_cast<void**>(page) = reinterpret_cast<void*>(0x718);
    fixture::control.pause.store(true);
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    auto pending = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        const auto status = app.query(static_cast<void**>(page));
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    wait_flag(fixture::control.entered);
    const auto protection = ::mprotect(page, 4096, PROT_NONE);
    fixture::control.release.store(true);
    expect(pending.get() == CUDNN_STATUS_BAD_PARAM && !protection,
           "lost output permission has no publication");
    expect(!::mprotect(page, 4096, PROT_READ), "inspect failed output without writable permission");
    expect(*static_cast<void**>(page) == reinterpret_cast<void*>(0x718),
           "post-RPC output remains untouched");
    expect(!::munmap(page, 4096), "release output page");
    shape(app.destinations[0], 2);
    Guarded output;
    check(app.query(&output.value));
    output.verify(0x10000);
}
void physical(Application& app) {
    const int sizes[]{1, 1, 1}, strides[]{1, 1, 1};
    check(cudnnSetTensorNdDescriptor(app.destinations[0], CUDNN_DATA_FLOAT, 3, sizes, strides));
    std::array<CUdeviceptr, 2> buffers{};
    for (auto& buffer : buffers) driver(cuMemAlloc(&buffer, 4));
    foundation::control.can_stop.store(false);
    const float alpha = 1, beta = 0;
    expect(cudnnAddTensor(app.handle, &alpha, app.destinations[0],
                          reinterpret_cast<void*>(buffers[0]), &beta, app.destinations[0],
                          reinterpret_cast<void*>(buffers[1])) == CUDNN_STATUS_EXECUTION_FAILED,
           "original failure ACK is not physical retirement");
    Guarded other;
    check(app.query(&other.value, reinterpret_cast<void*>(1), 1, 512, 1));
    other.verify(33);
    const auto calls = fixture::control.calls.load();
    std::atomic<bool> started{};
    auto pending = std::async(std::launch::async, [&] {
        driver(cuCtxSetCurrent(app.context));
        started.store(true);
        Guarded output;
        const auto status = app.query(&output.value);
        if (status == CUDNN_STATUS_SUCCESS) output.verify(0x10000);
        driver(cuCtxSetCurrent(nullptr));
        return status;
    });
    wait_flag(started);
    const bool blocked =
        pending.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    const bool unchanged =
        fixture::control.calls.load() == calls && !fixture::query::control.synchronizes.load();
    foundation::control.can_stop.store(true);
    check(pending.get());
    expect(blocked && unchanged,
           "borrowed destination waits without draining; fresh destination may query");
    for (const auto buffer : buffers) driver(cuMemFree(buffer));
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(!::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) &&
               !::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1),
           "authenticated local endpoint");
}
void run(const char* provider, ExecutionSchedulingMode mode, unsigned int missing = 0,
         int poison = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "real service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-attn-layout-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    if (missing >= 3) fixture::control.version.store(missing == 3 ? 91001 : 91003);
    std::future<void> client;
    ExecutionService service(std::move(config),
                             fixture::session_dispatch(missing < 3 ? missing : 0));
    expect(service.start(), "start owned attention-layout service");
    client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (missing) {
            Guarded output;
            expect(app.query(&output.value) == CUDNN_STATUS_NOT_SUPPORTED,
                   "unsupported callback/version clean policy");
            output.verify();
            app.configure();
        } else if (poison >= 0) {
            Guarded output;
            if (poison == 5) fixture::control.status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
            if (poison == 6 || poison == 7) fixture::control.corrupt.store(poison - 5);
            if (poison == 8) fixture::control.context_loss.store(true);
            if (poison == 9) fixture::control.version_context_loss.store(true);
            const auto status =
                app.api(app.handle, app.parent, CUDNN_MH_ATTN_Q_WEIGHTS,
                        poison == 4 ? 0 : (poison == 10 ? SIZE_MAX : 512),
                        poison == 3 || poison == 10 ? nullptr : reinterpret_cast<void*>(0x10000),
                        poison == 0 || poison == 2 ? nullptr : app.destinations[0],
                        poison == 0 || poison == 1 ? nullptr : &output.value);
            expect(status == (poison == 5                  ? CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE
                              : poison == 6 || poison == 7 ? CUDNN_STATUS_INTERNAL_ERROR
                              : poison == 8 || poison == 9 ? CUDNN_STATUS_NOT_INITIALIZED
                                                           : CUDNN_STATUS_BAD_PARAM),
                   "exact native error or containment policy, with no pointer publication");
            output.verify();
            const auto calls = fixture::control.calls.load();
            expect(app.query(&output.value) == CUDNN_STATUS_NOT_INITIALIZED &&
                       fixture::control.calls.load() == calls,
                   "poisoned descriptors never retry the native helper");
            if (poison != 8 && poison != 9) {
                check(cudnnDestroy(app.handle));
                driver(cuCtxDestroy(app.context));
            }
            return;
        } else {
            ordinary(app);
            publication(app);
            physical(app);
        }
        app.close();
    });
    const auto deadline = Clock::now() + std::chrono::seconds{20};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded attention-layout IPC progress");
    client.get();
    expect(!glimmer_dnn_attn_weights_native_calls(), "zero caller-native dispatch");
    if (poison >= 0) {
        expect(base::state.handle_destroys == (poison == 8 || poison == 9 ? 0U : 1U) &&
                   !fixture::query::control.synchronizes.load(),
               "context loss differs from metadata-only cleanup");
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished(), "healthy service retires");
    expect(fixture::metadata::state.destroys == 1 && fixture::dropout::state.destroys == 2 &&
               base::state.handle_destroys == 1 && foundation::driver.contexts.empty() &&
               foundation::driver.memory.empty() && foundation::driver.events.empty() &&
               !fixture::query::control.synchronizes.load() &&
               (missing ? !fixture::control.calls.load() : true),
           "all original owners retire once, queries never introduce context synchronization");
}

struct Hostile {
    ExecutionWireReply reply;
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    bool healthy{}, metadata_only{}, execution_only{};
    std::uintptr_t expected{0x718};
    bool has_base{true}, has_destination{true}, has_output{true};
    std::size_t bytes{512};
};
std::vector<std::byte> offset(std::uint32_t non_null = 1, std::uint64_t value = 32) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(non_null) && writer.write_u64(value),
           "encode independent 12-byte offset");
    return std::move(writer).take();
}
Hostile hostile(unsigned int index) {
    Hostile result;
    auto& reply = result.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    reply.data = offset();
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
            reply.data = offset(2);
            break;
        case 12:
            reply.data = offset(0, 32);
            break;
        case 13:
            reply.data = offset(1, 512);
            break;
        case 14:
            reply.data = offset(1, UINT64_MAX);
            break;
        case 15:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.data.clear();
            break;
        case 16:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.parameter_bytes = 1;
            break;
        case 17:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.parameter_bytes = 1;
            reply.data.clear();
            result.status = CUDNN_STATUS_BAD_PARAM;
            result.metadata_only = true;
            break;
        case 18:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            reply.original_status.reset();
            reply.data.clear();
            result.status = CUDNN_STATUS_NOT_SUPPORTED;
            result.healthy = true;
            break;
        case 19:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes = 1;
            reply.data.clear();
            result.metadata_only = true;
            break;
        case 20:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 3;
            reply.data.clear();
            result.status = CUDNN_STATUS_NOT_INITIALIZED;
            break;
        case 21:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.parameter_bytes = 2;
            reply.original_status.reset();
            reply.data.clear();
            result.status = CUDNN_STATUS_NOT_INITIALIZED;
            result.execution_only = true;
            break;
        case 22:
        case 23:
        case 24:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = index == 22 ? -1 : index == 23 ? 1024 : 0;
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
            result.expected = 0x10020;
            break;
        case 28:
            reply.data = offset(1, 0);
            result.status = CUDNN_STATUS_SUCCESS;
            result.healthy = true;
            result.expected = 0x10000;
            break;
        case 29:
            reply.data = offset(0, 0);
            result.status = CUDNN_STATUS_SUCCESS;
            result.healthy = true;
            result.expected = 0;
            break;
        case 30:
            result.has_destination = false;
            reply.data = offset(0, 0);
            break;
        case 31:
            result.has_output = false;
            reply.data = offset(0, 0);
            break;
        case 32:
            result.has_base = false;
            reply.data = offset(0, 0);
            break;
        case 33:
            result.has_base = false;
            result.bytes = 0;
            reply.data = offset(0, 0);
            result.status = CUDNN_STATUS_SUCCESS;
            result.healthy = true;
            result.expected = 0;
            break;
        default:
            throw std::runtime_error("unknown hostile case");
    }
    return result;
}
void malformed_case(const char* provider, unsigned int index) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated hostile peer");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-attn-layout-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    const auto test = hostile(index);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start hostile IPC peer");
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnAttnDescriptor_t parent{};
        cudnnTensorDescriptor_t tensor{};
        check(cudnnCreateAttnDescriptor(&parent));
        check(cudnnCreateTensorDescriptor(&tensor));
        CUcontext context{};
        driver(cuCtxCreate(&context, 0, 0));
        cudnnHandle_t handle{};
        check(cudnnCreate(&handle));
        Guarded output;
        const auto invoke = [&] {
            return api(handle, parent, CUDNN_MH_ATTN_Q_WEIGHTS, test.bytes,
                       test.has_base ? reinterpret_cast<void*>(0x10000) : nullptr,
                       test.has_destination ? tensor : nullptr,
                       test.has_output ? &output.value : nullptr);
        };
        expect(invoke() == test.status, "exact hostile/control receipt status");
        output.verify(test.expected);
        if (test.healthy) {
            check(invoke());
            output.verify(test.has_base ? 0x10020 : 0);
        } else {
            expect(invoke() == CUDNN_STATUS_NOT_INITIALIZED,
                   "poisoned authority rejects locally without retry");
            output.verify(test.expected);
            if (test.metadata_only)
                check(cudnnDestroy(handle));
            else if (test.execution_only)
                check(cudnnDestroyTensorDescriptor(tensor));
            else
                expect(cudnnDestroyTensorDescriptor(tensor) == CUDNN_STATUS_NOT_INITIALIZED,
                       "malformed receipt quarantines both domains");
        }
        expect(!glimmer_dnn_attn_weights_native_calls(), "no caller-native hostile fallback");
    });
    unsigned int hellos{}, contexts{}, creates{}, handles{}, queries{}, probes{};
    std::uint64_t metadata_connection{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded hostile peer progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "authenticated peer identity preserved");
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
                           "exact independent DNM1 bootstrap, not execution authority");
                    ++hellos;
                    metadata_connection = request->connection;
                    ownership = ExecutionBootstrapOwnership::kMetadataRetained;
                }
                const auto encoded = encode_execution_bootstrap_ownership(ownership);
                expect(encoded.has_value(), "bootstrap ownership");
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && input.empty() && kind == (creates ? 1U : 14U),
                       "typed attention and tensor creations");
                reply.id = 17 + creates++;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (operation == ExecutionOpcode::kCreateDnn) {
                expect(input.empty(), "original handle create payload");
                ++handles;
                reply.id = 23;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (operation == ExecutionOpcode::kDnnAttnWeights) {
                ExecutionDnnAttnWeights value;
                expect(read_execution_dnn_attn_weights(input, value) && input.empty() &&
                           request->message.payload.size() == 52 && value.handle == 23 &&
                           value.metadata == metadata_connection && value.descriptor == 17 &&
                           !value.kind && value.has_base == test.has_base &&
                           value.weight_bytes == test.bytes &&
                           value.destination == (test.has_destination ? 18U : 0U) &&
                           value.has_output == test.has_output,
                       "wire contains presence, never caller base address or allocation authority");
                if (!queries++)
                    reply = test.reply;
                else {
                    expect(test.healthy, "no poisoned retry crosses transport");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    reply.data = test.has_base ? offset() : offset(0, 0);
                }
            } else if (operation == ExecutionOpcode::kDestroyDnn) {
                std::uint64_t handle{};
                expect(test.metadata_only && !probes++ && input.read_u64(handle) && handle == 23 &&
                           input.empty(),
                       "metadata-only quarantine leaves handle healthy");
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else {
                std::uint64_t id{};
                expect(operation == ExecutionOpcode::kDestroyDnnMetadata && test.execution_only &&
                           !probes++ && input.read_u64(id) && id == 18 && input.empty(),
                       "execution-only quarantine leaves metadata healthy");
                reply.original_status = CUDNN_STATUS_SUCCESS;
            }
            const auto payload = encode_execution_reply(reply);
            expect(payload && decode_execution_reply(*payload),
                   "hostile envelope remains structurally wire-valid");
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
    expect(hellos == 1 && contexts == 1 && creates == 2 && handles == 1 &&
               queries == (test.healthy ? 2U : 1U) &&
               probes == (test.metadata_only || test.execution_only ? 1U : 0U),
           "exact RPC counts prove quarantine and no hidden queries/fallback");
    server.stop();
}
unsigned int case_index(const char* text) {
    unsigned int result{};
    const std::string_view value{text};
    expect(!value.empty() && value.size() < 3, "bounded case index");
    for (const char digit : value) {
        expect(digit >= '0' && digit <= '9', "decimal case index");
        result = result * 10 + digit - '0';
    }
    return result;
}
void isolated(const char* executable, const char* provider, const char* mode, unsigned int count) {
    for (unsigned int i = 0; i != count; ++i) {
        const auto text = std::to_string(i);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated case");
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
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--poison" || mode == "--missing-callback") {
            expect(argc == 3, "parent mode takes no index");
            isolated(argv[0], argv[1],
                     mode == "--malformed-replies" ? "--malformed-case"
                     : mode == "--poison"          ? "--poison-case"
                                                   : "--missing-case",
                     mode == "--malformed-replies" ? kHostileCases
                     : mode == "--poison"          ? 11U
                                                   : 4U);
        } else if (mode == "--malformed-case" || mode == "--poison-case" ||
                   mode == "--missing-case") {
            expect(argc == 4, "isolated mode takes index");
            const auto index = case_index(argv[3]);
            expect(index < (mode == "--malformed-case" ? kHostileCases
                            : mode == "--poison-case"  ? 11U
                                                       : 4U),
                   "known case");
            if (mode == "--malformed-case") {
                malformed_case(argv[1], index);
                ::_exit(0);
            }
            run(argv[1], ExecutionSchedulingMode::kNaive, mode == "--missing-case" ? index + 1 : 0,
                mode == "--poison-case" ? static_cast<int>(index) : -1);
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "Attention-layout CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.release.store(true);
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
