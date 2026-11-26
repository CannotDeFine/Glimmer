#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fatbinary_section.h>

#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>

// Exact compiler callback ABI; no vendor Runtime is loaded by this fixture.
// NOLINTBEGIN(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" void** __cudaRegisterFatBinary(void*);
extern "C" void __cudaRegisterFatBinaryEnd(void**);
extern "C" void __cudaUnregisterFatBinary(void**);
extern "C" void __cudaRegisterFunction(void**, const char*, char*, const char*, int, uint3*, uint3*,
                                       dim3*, dim3*, int*);
// NOLINTEND(bugprone-reserved-identifier, readability-identifier-naming)
extern "C" unsigned int glimmer_execution_lookup_count(unsigned int);

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr std::array kScenarios{"complete",
                                "replacement",
                                "error-complete",
                                "error-original",
                                "error-id",
                                "error-address",
                                "error-extent",
                                "error-arguments",
                                "error-data",
                                "duplicate-sequence",
                                "duplicate-frontier",
                                "clean-rejection",
                                "runtime-complete",
                                "batch-missing",
                                "batch-reordered",
                                "batch-duplicate",
                                "batch-sequence",
                                "batch-partial-cache",
                                "batch-retirement",
                                "batch-status",
                                "batch-envelope",
                                "batch-query-lost",
                                "batch-release-missing",
                                "batch-release-reordered",
                                "batch-release-lost",
                                "batch-release-error",
                                "batch-mixed",
                                "batch-capacity",
                                "batch-success-release",
                                "busy-retry",
                                "busy-wrong-operation",
                                "busy-status",
                                "busy-original",
                                "busy-id",
                                "busy-address",
                                "busy-extent",
                                "busy-arguments",
                                "busy-data"};

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
cudaError_t await_runtime_error() {
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    cudaError_t value{};
    while ((value = cudaPeekAtLastError()) == cudaSuccess) {
        expect(Clock::now() < deadline, "bounded eventual background error notification");
        std::this_thread::yield();
    }
    return value;
}
bool is_duplicate(std::string_view scenario) {
    return scenario == "duplicate-sequence" || scenario == "duplicate-frontier";
}
void application(std::string_view scenario) {
    CUcontext context{};
    expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS, "create independent public context");
    CUmodule module{};
    CUfunction function{};
    const int host_identity{};
    const bool is_runtime = scenario == "runtime-complete";
    if (is_runtime) {
        alignas(8) std::array<std::byte, 24> image{};
        const std::array<std::uint64_t, 2> prefix{0x00100001ba55ed50, 8};
        std::memcpy(image.data(), prefix.data(), sizeof(prefix));
        image[16] = std::byte{7};
        __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                    reinterpret_cast<const unsigned long long*>(image.data()),
                                    nullptr};
        auto** registration = __cudaRegisterFatBinary(&wrapper);
        expect(registration != nullptr, "snapshot ordinary Runtime compiler registration");
        char name[] = "kernel";
        __cudaRegisterFunction(registration, reinterpret_cast<const char*>(&host_identity), name,
                               name, -1, nullptr, nullptr, nullptr, nullptr, nullptr);
        __cudaRegisterFatBinaryEnd(registration);
        expect(cudaGetLastError() == cudaSuccess, "valid registration leaves Runtime TLS clear");
    } else {
        const auto loaded = cuModuleLoadData(&module, "owned zero-argument receipt fixture");
        if (scenario == "busy-wrong-operation") {
            expect(loaded == CUDA_SUCCESS && module &&
                       cuModuleGetFunction(&function, module, "kernel") == CUDA_ERROR_UNKNOWN &&
                       !function && cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN,
                   "busy is not a metadata success or permission to retry a different operation");
            return;
        }
        expect(loaded == CUDA_SUCCESS &&
                   cuModuleGetFunction(&function, module, "kernel") == CUDA_SUCCESS,
               "prepare public Driver function metadata");
    }
    const auto launch = [&] {
        return is_runtime ? static_cast<int>(cudaLaunchKernel(&host_identity, dim3{320}, dim3{32},
                                                              nullptr, 0, nullptr))
                          : static_cast<int>(cuLaunchKernel(function, 320, 1, 1, 32, 1, 1, 0,
                                                            nullptr, nullptr, nullptr));
    };
    if (scenario == "busy-retry") {
        expect(launch() == CUDA_SUCCESS && cuCtxSynchronize() == CUDA_SUCCESS &&
                   cuModuleUnload(module) == CUDA_SUCCESS && cuCtxDestroy(context) == CUDA_SUCCESS,
               "verified unaccepted busy replies precede one exact accepted command and normal "
               "cleanup");
        return;
    }
    if (scenario == "clean-rejection") {
        expect(launch() == CUDA_ERROR_OUT_OF_MEMORY,
               "exact clean rejection preserves the pre-admission Driver error");
        expect(launch() == CUDA_SUCCESS, "explicit later call can admit after a clean rejection");
        expect(cuCtxSynchronize() == CUDA_SUCCESS && cuModuleUnload(module) == CUDA_SUCCESS &&
                   cuCtxDestroy(context) == CUDA_SUCCESS,
               "clean rejection leaves normal observation and teardown usable");
        return;
    }
    if (is_duplicate(scenario))
        expect(launch() == CUDA_SUCCESS, "first exact token has a pending original owner");
    expect(launch() == CUDA_ERROR_UNKNOWN,
           "wrong success, malformed rejection or duplicate token cannot publish CUDA success");
    if (is_runtime) {
        expect(cudaPeekAtLastError() == cudaErrorUnknown &&
                   cudaGetLastError() == cudaErrorUnknown && cudaGetLastError() == cudaSuccess,
               "immediately reported Runtime uncertainty is consumed exactly once");
    } else {
        expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess,
               "Driver receipt uncertainty cannot manufacture a Runtime-origin notification");
    }
    expect(launch() == (is_runtime ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_INVALID_HANDLE) &&
               cuCtxSynchronize() == CUDA_ERROR_INVALID_CONTEXT &&
               cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN,
           "uncertain owner rejects replay, observation and invented successful teardown");
    // Deliberately uncertain contexts are isolated in this fresh test process.
    // The endpoint owns no CUDA resource: this proves quarantine/no replay,
    // not successful physical cleanup or numerical execution.
}

ExecutionWireReply launch_reply(std::string_view scenario, unsigned int count) {
    ExecutionWireReply reply;
    if (scenario.starts_with("busy-") && (scenario != "busy-retry" || count < 3)) {
        reply.disposition = ExecutionWireDisposition::kAdmissionBusy;
        return reply;
    }
    if (scenario == "clean-rejection" && count == 1) {
        reply.status = CUDA_ERROR_OUT_OF_MEMORY;
        return reply;
    }
    if (scenario == "clean-rejection" || scenario == "busy-retry" || is_duplicate(scenario)) {
        reply.disposition = ExecutionWireDisposition::kAdmissionAccepted;
        reply.id = scenario == "duplicate-frontier" && count == 2 ? 18 : 17;
        reply.address = scenario == "duplicate-sequence" && count == 2 ? 31 : 29;
        return reply;
    }
    if (scenario == "complete" || scenario == "runtime-complete") {
        reply.disposition = ExecutionWireDisposition::kComplete;
    } else if (scenario == "replacement") {
        reply.disposition = ExecutionWireDisposition::kComputeReplacement;
        reply.id = 17;
    } else {
        reply.status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        if (scenario == "error-complete")
            reply.disposition = ExecutionWireDisposition::kComplete;
        else if (scenario == "error-original")
            reply.original_status = reply.status;
        else if (scenario == "error-id")
            reply.id = 17;
        else if (scenario == "error-address")
            reply.address = 29;
        else if (scenario == "error-extent")
            reply.parameter_bytes = 1;
        else if (scenario == "error-arguments") {
            reply.parameter_bytes = 8;
            reply.arguments = {{0, 8}};
        } else if (scenario == "error-data")
            reply.data = {std::byte{1}};
        else
            throw std::runtime_error("unknown receipt scenario");
    }
    return reply;
}

struct BatchState {
    std::atomic<bool> observes{}, physical_ready{}, capacity_probe{}, eight_released{};
    std::atomic<int> capacity_result{cudaErrorUnknown};
    std::atomic<bool> capacity_returned_early{}, capacity_tls_preserved{};
};
struct BatchApplication {
    CUcontext context{};
    void** registration{};
    int host_identity{};
    void open() {
        expect(cuCtxCreate(&context, 0, 0) == CUDA_SUCCESS, "create public batch receipt context");
        alignas(8) std::array<std::byte, 24> image{};
        const std::array<std::uint64_t, 2> prefix{0x00100001ba55ed50, 8};
        std::memcpy(image.data(), prefix.data(), sizeof(prefix));
        image[16] = std::byte{7};
        __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                    reinterpret_cast<const unsigned long long*>(image.data()),
                                    nullptr};
        registration = __cudaRegisterFatBinary(&wrapper);
        expect(registration != nullptr, "register owned public batch Runtime image");
        char name[] = "kernel";
        __cudaRegisterFunction(registration, reinterpret_cast<const char*>(&host_identity), name,
                               name, -1, nullptr, nullptr, nullptr, nullptr, nullptr);
        __cudaRegisterFatBinaryEnd(registration);
        expect(cudaGetLastError() == cudaSuccess, "batch registration preserves clear TLS");
    }
    cudaError_t launch() const {
        return cudaLaunchKernel(&host_identity, dim3{320}, dim3{32}, nullptr, 0, nullptr);
    }
    void close() const {
        // Unregister synchronizes a live binding and must preserve that
        // generation's error history. Retire the proven-stopped context first,
        // as the actual late-error fixture resets before unregistering its image.
        expect(cuCtxDestroy(context) == CUDA_SUCCESS,
               "all exact batch frontiers retire before normal context destruction");
        __cudaUnregisterFatBinary(registration);
        expect(cudaGetLastError() == cudaSuccess,
               "dead module bindings unregister normally without replaying consumed errors");
    }
};
void batch_application(std::string_view scenario, BatchState& state) {
    BatchApplication app;
    app.open();
    if (scenario == "batch-success-release") {
        state.observes.store(true, std::memory_order_release);
        for (unsigned int index = 0; index != 9; ++index) {
            expect(app.launch() == cudaSuccess, "admit each exact successful Runtime launch");
            expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess &&
                       cudaPeekAtLastError() == cudaSuccess,
                   "repeated error queries preserve verified success without consuming errors");
            if (index == 0)
                expect(cuStreamSynchronize(reinterpret_cast<CUstream>(app.context)) ==
                           CUDA_ERROR_INVALID_HANDLE,
                       "a context handle is not a stream and must not flush retained successes");
        }
        const auto deadline = Clock::now() + std::chrono::seconds{5};
        while (!state.eight_released.load(std::memory_order_acquire)) {
            expect(Clock::now() < deadline, "idle caller still receives autonomous batch progress");
            std::this_thread::yield();
        }
        expect(cuCtxSynchronize() == CUDA_SUCCESS,
               "explicit synchronization flushes proven metadata");
        app.close();
        return;
    }
    const bool is_capacity = scenario == "batch-capacity";
    const unsigned int count = is_capacity ? 256 : 2;
    for (unsigned int index = 0; index < count; ++index)
        expect(app.launch() == cudaSuccess, "every exact initial asynchronous launch is admitted");
    if (is_capacity) {
        state.capacity_probe.store(true, std::memory_order_release);
        const auto result = app.launch();
        state.capacity_returned_early.store(!state.observes.load(), std::memory_order_release);
        state.capacity_result.store(result, std::memory_order_release);
        state.capacity_tls_preserved.store(
            cudaGetLastError() == result && cudaGetLastError() == cudaSuccess,
            std::memory_order_release);
    }
    state.observes.store(true, std::memory_order_release);
    if (is_capacity) {
        expect(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess &&
                   app.launch() == cudaSuccess && cudaPeekAtLastError() == cudaSuccess,
               "one full retirement batch frees every slot and admits a fresh exact owner");
        app.close();
        return;
    }
    if (scenario == "batch-mixed") {
        expect(await_runtime_error() == cudaErrorLaunchOutOfResources &&
                   cudaGetLastError() == cudaErrorLaunchOutOfResources &&
                   cudaGetLastError() == cudaSuccess,
               "mixed pending error and retired success preserve exactly one Runtime error");
        expect(cuCtxSynchronize() == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
               "physical pending error remains available after its notification is consumed");
        state.physical_ready.store(true, std::memory_order_release);
        expect(cudaPeekAtLastError() == cudaSuccess &&
                   cuCtxSynchronize() == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
               "later physical retirement releases the token without erasing error history");
        app.close();
        return;
    }
    const auto expected =
        scenario == "batch-release-error" ? cudaErrorLaunchOutOfResources : cudaErrorUnknown;
    if (scenario.starts_with("batch-release-") && scenario != "batch-release-error") {
        // Force a deferred successful Release before checking its hostile receipt.
        const auto first = cudaPeekAtLastError();
        if (first == cudaSuccess) {
            const auto synchronized = cuCtxSynchronize();
            expect(synchronized == CUDA_ERROR_UNKNOWN || synchronized == CUDA_ERROR_INVALID_CONTEXT,
                   "lost/malformed Release either fails this synchronization or already "
                   "quarantined it");
        } else
            expect(first == cudaErrorUnknown, "immediate Release uncertainty remains visible");
    }
    expect(await_runtime_error() == expected && cudaGetLastError() == expected &&
               await_runtime_error() == cudaErrorUnknown &&
               cudaGetLastError() == cudaErrorUnknown && cudaGetLastError() == cudaSuccess,
           "batch uncertainty preserves earlier native error and notifies each origin once");
    expect(app.launch() == cudaErrorUnknown && cuCtxSynchronize() == CUDA_ERROR_INVALID_CONTEXT &&
               cuCtxDestroy(app.context) == CUDA_ERROR_UNKNOWN,
           "uncertain batch outcomes forbid replay, single-token fallback and fabricated cleanup");
    // No native resource exists in this hostile endpoint. The frontend must
    // retain its unresolved ownership metadata until this fresh process exits.
}
void overwrite_u32(std::vector<std::byte>& bytes, std::size_t offset, std::uint32_t value) {
    expect(offset <= bytes.size() && bytes.size() - offset >= 4,
           "mutate only a bounded field of the actual versioned observation grammar");
    for (unsigned int index = 0; index < 4; ++index)
        bytes[offset + index] = std::byte{static_cast<unsigned char>(value >> (8 * index))};
}
void run_batch(std::string_view scenario, const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.io_timeout_ms = 1000;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start independent bounded batch receipt endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticate actual batch receipt endpoint");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure batch endpoint before public SDK entry");
    BatchState state;
    auto client = std::async(std::launch::async, [&] { batch_application(scenario, state); });
    std::map<std::uint64_t, ExecutionAsyncFrontierKey> retained;
    struct Captured {
        std::int32_t status{};
        bool is_retired{};
    };
    std::map<std::uint64_t, Captured> captured;
    std::uint64_t next_capture{100000};
    unsigned int launches{}, closes{}, released_items{}, release_batches{}, single_async_queries{},
        single_async_releases{}, corruptions{};
    std::size_t max_reaper{}, max_probe{}, full_releases{};
    const bool is_capacity = scenario == "batch-capacity";
    const bool is_mixed = scenario == "batch-mixed";
    const bool is_success_release = scenario == "batch-success-release";
    const bool is_clean = is_capacity || is_mixed || is_success_release;
    std::vector<std::size_t> successful_release_sizes;
    const auto observes = [&] { return state.observes.load(std::memory_order_acquire); };
    const auto observation = [&](ExecutionAsyncFrontierKey key) {
        const bool is_error = (is_mixed || scenario == "batch-release-error") && key.sequence == 18;
        return ExecutionAsyncFrontierObservation{
            key,
            observes() ? (is_error ? CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES : CUDA_SUCCESS)
                       : CUDA_ERROR_NOT_READY,
            observes() &&
                (!is_error || !is_mixed || state.physical_ready.load(std::memory_order_acquire))};
    };
    std::optional<ExecutionWireMessage> held_observer_reply;
    std::uint64_t held_connection{};
    bool has_held_observer{}, has_released_observer{};
    std::optional<Clock::time_point> capacity_time;
    const auto deadline = Clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        if (state.capacity_probe.load(std::memory_order_acquire) && !capacity_time)
            capacity_time = Clock::now();
        if (capacity_time && max_probe >= 248 &&
            Clock::now() - *capacity_time >= std::chrono::milliseconds{20})
            state.observes.store(true, std::memory_order_release);
        if (held_observer_reply && observes()) {
            expect(server.reply(held_connection, *held_observer_reply),
                   "release only the retained exact observer response after independent recovery");
            held_observer_reply.reset();
            has_released_observer = true;
        }
        expect(server.poll_for(1), "progress actual public batch transport");
        if (auto request = server.take_request()) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            const auto operation = request->message.header.operation;
            ExecutionPayloadReader input(request->message.payload);
            std::uint64_t id{}, stream{};
            switch (operation) {
                case ExecutionOpcode::kHello:
                    break;
                case ExecutionOpcode::kLoadBinary:
                    reply.original_status = CUDA_SUCCESS;
                    reply.id = 1;
                    break;
                case ExecutionOpcode::kGetFunction:
                    reply.id = 2;
                    break;
                case ExecutionOpcode::kLaunch: {
                    std::uint32_t value{};
                    std::span<const std::byte> parameters;
                    expect(input.read_u64(id) && id == 2 && input.read_u64(stream) && !stream,
                           "exact owned zero-argument Runtime launch");
                    for (unsigned int index = 0; index < 7; ++index)
                        expect(input.read_u32(value), "full original launch geometry");
                    expect(input.read_blob(parameters) && parameters.empty() &&
                               input.read_u32(value) && value == kExecutionAsyncLaunchMagic &&
                               input.empty(),
                           "batch fixture retains explicit ordinary ASY1 admission");
                    ++launches;
                    const ExecutionAsyncFrontierKey key{1000 + launches, 17 + launches};
                    expect(retained.emplace(key.id, key).second,
                           "every admitted command owns a distinct nonrecycled frontier");
                    reply.disposition = ExecutionWireDisposition::kAdmissionAccepted;
                    reply.id = key.sequence;
                    reply.address = key.id;
                    break;
                }
                case ExecutionOpcode::kQueryAsyncFrontiers:
                case ExecutionOpcode::kReleaseAsyncFrontiers: {
                    const auto keys =
                        decode_execution_async_frontier_batch(request->message.payload);
                    expect(keys && !keys->empty() && keys->size() <= 256,
                           "public frontend uses versioned bounded exact observation keys");
                    for (const auto key : *keys) {
                        const auto found = retained.find(key.id);
                        expect(found != retained.end() && found->second == key,
                               "whole batch matches independently retained command identities");
                    }
                    if (operation == ExecutionOpcode::kQueryAsyncFrontiers) {
                        if (!observes()) {
                            if (state.capacity_probe.load(std::memory_order_acquire))
                                max_probe = std::max(max_probe, keys->size());
                            else {
                                max_reaper = std::max(max_reaper, keys->size());
                                expect(keys->size() <= 8,
                                       "ordinary admission still reaps only eight");
                            }
                        }
                        std::vector<ExecutionAsyncFrontierObservation> observations;
                        for (const auto key : *keys)
                            observations.push_back(
                                !is_clean && keys->size() != 2
                                    ? ExecutionAsyncFrontierObservation{key, CUDA_ERROR_NOT_READY,
                                                                        false}
                                    : observation(key));
                        const bool corrupts = observes() && !is_clean && keys->size() == 2 &&
                                              !scenario.starts_with("batch-release-");
                        if (corrupts) {
                            expect(keys->size() == 2 && !corruptions++,
                                   "corrupt only the selected complete two-record batch once");
                            if (scenario == "batch-query-lost") {
                                server.close_peer(request->connection);
                                continue;
                            }
                            if (scenario == "batch-missing") observations.pop_back();
                            if (scenario == "batch-reordered")
                                std::swap(observations.front(), observations.back());
                            if (scenario == "batch-partial-cache")
                                observations.front().status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
                            if (scenario == "batch-sequence" || scenario == "batch-partial-cache")
                                observations.back().key.sequence += 7;
                        }
                        const auto encoded =
                            encode_execution_async_frontier_observations(observations);
                        expect(encoded.has_value(), "encode exact valid independent observations");
                        reply.data = *encoded;
                        if (corrupts) {
                            if (scenario == "batch-duplicate")
                                std::copy_n(reply.data.begin() + 8, 24, reply.data.begin() + 32);
                            else if (scenario == "batch-retirement")
                                overwrite_u32(reply.data, 28, 2);
                            else if (scenario == "batch-status")
                                overwrite_u32(reply.data, 24, 1000);
                            else if (scenario == "batch-envelope")
                                reply.status = CUDA_ERROR_OUT_OF_MEMORY;
                        }
                    } else {
                        if (is_success_release) successful_release_sizes.push_back(keys->size());
                        if (is_success_release && keys->size() == 8)
                            state.eight_released.store(true, std::memory_order_release);
                        ++release_batches;
                        for (const auto key : *keys)
                            expect(
                                observation(key).is_retired,
                                "Release never includes an error whose physical tail is pending");
                        released_items += static_cast<unsigned int>(keys->size());
                        if (keys->size() == 256) ++full_releases;
                        for (const auto key : *keys) retained.erase(key.id);
                        auto echo = *keys;
                        if (!is_clean) {
                            expect(keys->size() == 2 && !corruptions++,
                                   "lose or corrupt exactly one fully validated release outcome");
                            if (scenario == "batch-release-lost") {
                                server.close_peer(request->connection);
                                continue;
                            }
                            if (scenario == "batch-release-reordered")
                                std::swap(echo.front(), echo.back());
                            else
                                echo.pop_back();
                        }
                        const auto encoded = encode_execution_async_frontier_batch(echo);
                        expect(encoded.has_value(), "encode actual release identity echo");
                        reply.data = *encoded;
                    }
                    break;
                }
                case ExecutionOpcode::kCaptureFrontier: {
                    std::uint32_t scope{};
                    expect(input.read_u32(scope) && !scope && input.read_u64(stream) && !stream &&
                               input.empty(),
                           "ordinary context synchronization captures an independent prefix");
                    Captured prefix{CUDA_SUCCESS, true};
                    for (const auto& [key_id, key] : retained) {
                        (void)key_id;
                        const auto value = observation(key);
                        if (prefix.status == CUDA_SUCCESS && value.status != CUDA_SUCCESS)
                            prefix.status = value.status;
                        prefix.is_retired = prefix.is_retired && value.is_retired;
                    }
                    reply.id = next_capture++;
                    reply.address = launches ? 17 + launches : 0;
                    captured.emplace(reply.id, prefix);
                    break;
                }
                case ExecutionOpcode::kQueryFrontier:
                case ExecutionOpcode::kReleaseFrontier: {
                    expect(input.read_u64(id) && input.empty(), "exact ordinary single frontier");
                    const auto found = captured.find(id);
                    if (found != captured.end()) {
                        if (operation == ExecutionOpcode::kReleaseFrontier)
                            captured.erase(found);
                        else {
                            reply.status = found->second.status;
                            reply.parameter_bytes = found->second.is_retired ? 1 : 0;
                        }
                    } else {
                        expect(is_mixed && retained.contains(id),
                               "hostile batch result never falls back to individual token RPCs");
                        const auto observed = observation(retained.at(id));
                        if (operation == ExecutionOpcode::kReleaseFrontier) {
                            ++single_async_releases;
                            expect(observed.is_retired, "single cleanup also needs physical proof");
                            retained.erase(id);
                        } else {
                            ++single_async_queries;
                            reply.status = observed.status;
                            reply.parameter_bytes = observed.is_retired ? 1 : 0;
                        }
                    }
                    break;
                }
                case ExecutionOpcode::kUnloadModule:
                    expect(is_clean && retained.empty(), "module cleanup follows exact retirement");
                    break;
                case ExecutionOpcode::kClose:
                    expect(is_clean && retained.empty(), "uncertain contexts never close");
                    ++closes;
                    break;
                default:
                    throw std::runtime_error("unexpected remote action in batch receipt fixture");
            }
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {61, 67};
            auto encoded = encode_execution_reply(reply);
            if (is_capacity && operation == ExecutionOpcode::kQueryAsyncFrontiers &&
                !has_held_observer && !state.capacity_probe.load(std::memory_order_acquire)) {
                expect(encoded.has_value(), "retain the actual ordinary observer response");
                held_observer_reply = ExecutionWireMessage{header, std::move(*encoded)};
                held_connection = request->connection;
                has_held_observer = true;
                continue;
            }
            expect(encoded && server.reply(request->connection, {header, std::move(*encoded)}),
                   "deliver independently constructed exact or adversarial batch carrier");
        }
        expect(Clock::now() < deadline, "bounded batch observation and ownership checks");
    }
    client.get();
    const bool retained_uncertain_scope =
        scenario.starts_with("batch-release-") && scenario != "batch-release-error" &&
        captured.size() == 1 && captured.begin()->second.status == CUDA_SUCCESS &&
        captured.begin()->second.is_retired;
    expect(
        glimmer_execution_lookup_count(2) == 0 && (captured.empty() || retained_uncertain_scope) &&
            launches == (is_capacity ? (state.capacity_result.load() == cudaSuccess ? 258U : 257U)
                         : is_success_release ? 9U
                                              : 2U),
        "batch outcomes never replay original launches or invoke caller-native fallback");
    if (is_clean) {
        expect(closes == 1 && retained.empty() && released_items == launches &&
                   !single_async_releases && !corruptions,
               "healthy observations retire every exact token before normal cleanup");
        if (is_success_release) {
            std::cout << "Successful async metadata normal cleanup: launches=" << launches
                      << "; release_batches=" << release_batches
                      << "; released_items=" << released_items
                      << "; first_batch=" << successful_release_sizes.front() << '\n'
                      << std::flush;
            expect(
                successful_release_sizes == std::vector<std::size_t>{8, 1} &&
                    release_batches == 2 && !single_async_queries,
                "eight verified successes amortize Release; synchronization flushes the remainder");
        } else if (is_capacity)
            expect(max_reaper <= 8 && max_probe >= 248 && max_probe <= 256 && !single_async_queries,
                   "full256 pressure recovers while at most eight records belong to the observer");
        else
            expect(release_batches == 2 && single_async_queries == 0,
                   "mixed pending failure and retired success release in two independently proven "
                   "phases");
    } else {
        const bool loses_release = scenario.starts_with("batch-release-");
        expect(!closes && corruptions == 1 && !single_async_queries && !single_async_releases &&
                   release_batches == (loses_release ? 1U : 0U) &&
                   retained.size() == (loses_release ? 0U : 2U),
               "unknown query/release outcomes retain frontend ownership without hidden recovery");
    }
    server.stop();
    if (is_capacity)
        expect(state.capacity_result.load() == cudaSuccess &&
                   !state.capacity_returned_early.load() && state.capacity_tls_preserved.load() &&
                   has_held_observer && has_released_observer && launches == 258,
               "after exact metadata cleanup, full local tracking waits for the held observer "
               "without OOM");
}
void run(std::string_view scenario, const std::string& path) {
    const bool is_clean = scenario == "clean-rejection" || scenario == "busy-retry";
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.io_timeout_ms = 1000;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start independent authenticated malicious-receipt endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "obtain exact trusted server process identity");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure endpoint before any public frontend call");
    auto client = std::async(std::launch::async, [scenario] { application(scenario); });
    std::set<std::uint64_t> captured;
    std::uint64_t next_capture{100};
    unsigned int launches{}, closes{}, async_queries{}, async_releases{}, function_requests{},
        admission_replies{};
    std::vector<std::byte> first_launch_payload;
    const auto deadline = Clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress authenticated request/response traffic");
        if (auto request = server.take_request()) {
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            std::uint64_t id{}, stream{};
            switch (request->message.header.operation) {
                case ExecutionOpcode::kHello:
                    break;
                case ExecutionOpcode::kLoadPtx:
                case ExecutionOpcode::kLoadBinary:
                    reply.original_status = CUDA_SUCCESS;
                    reply.id = 1;
                    break;
                case ExecutionOpcode::kGetFunction:
                    ++function_requests;
                    if (scenario == "busy-wrong-operation")
                        reply.disposition = ExecutionWireDisposition::kAdmissionBusy;
                    else
                        reply.id = 2;
                    break;
                case ExecutionOpcode::kLaunch: {
                    std::uint32_t value{};
                    std::span<const std::byte> parameters;
                    expect(input.read_u64(id) && id == 2 && input.read_u64(stream) && !stream,
                           "exact owned function/default stream on the wire");
                    for (unsigned int index = 0; index < 7; ++index)
                        expect(input.read_u32(value), "complete original launch geometry");
                    expect(input.read_blob(parameters) && parameters.empty() &&
                               input.read_u32(value) && value == kExecutionAsyncLaunchMagic &&
                               input.empty(),
                           "ordinary frontend explicitly selects only the strict ASY1 contract");
                    if (scenario == "busy-retry") {
                        if (first_launch_payload.empty())
                            first_launch_payload = request->message.payload;
                        else
                            expect(first_launch_payload == request->message.payload,
                                   "busy retries retain the original copied argument/geometry "
                                   "payload");
                    }
                    reply = launch_reply(scenario, ++launches);
                    if (reply.disposition == ExecutionWireDisposition::kAdmissionAccepted)
                        ++admission_replies;
                    break;
                }
                case ExecutionOpcode::kCaptureFrontier: {
                    std::uint32_t scope{};
                    expect(input.read_u32(scope) && !scope && input.read_u64(stream) && !stream &&
                               input.empty(),
                           "capture only the ordinary context prefix");
                    reply.id = next_capture++;
                    reply.address = launches ? 17 : 0;
                    captured.insert(reply.id);
                    break;
                }
                case ExecutionOpcode::kQueryFrontier:
                case ExecutionOpcode::kReleaseFrontier: {
                    expect(input.read_u64(id) && input.empty(), "exact frontier observation");
                    const bool is_release =
                        request->message.header.operation == ExecutionOpcode::kReleaseFrontier;
                    if (captured.contains(id)) {
                        if (is_release)
                            captured.erase(id);
                        else
                            reply.parameter_bytes = 1;
                    } else {
                        expect(id == 29, "observation never invents a second exact token");
                        if (is_release) {
                            expect(is_clean,
                                   "pending duplicate owner cannot be physically released");
                            ++async_releases;
                        } else {
                            ++async_queries;
                            reply.status = is_clean ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
                            reply.parameter_bytes = is_clean ? 1 : 0;
                        }
                    }
                    break;
                }
                case ExecutionOpcode::kQueryAsyncFrontiers:
                case ExecutionOpcode::kReleaseAsyncFrontiers: {
                    const auto keys =
                        decode_execution_async_frontier_batch(request->message.payload);
                    expect(keys && keys->size() == 1 && keys->front().id == 29 &&
                               keys->front().sequence == 17,
                           "batch reaping observes only the first exact admitted owner");
                    if (request->message.header.operation ==
                        ExecutionOpcode::kReleaseAsyncFrontiers) {
                        expect(is_clean, "pending duplicate owner cannot be released by a batch");
                        ++async_releases;
                        reply.data = *encode_execution_async_frontier_batch(*keys);
                    } else {
                        ++async_queries;
                        const std::array observations{ExecutionAsyncFrontierObservation{
                            keys->front(), is_clean ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY,
                            is_clean}};
                        reply.data = *encode_execution_async_frontier_observations(observations);
                    }
                    break;
                }
                case ExecutionOpcode::kUnloadModule:
                    expect(is_clean, "uncertain modules are retained");
                    break;
                case ExecutionOpcode::kClose:
                    expect(is_clean, "uncertain owners are never closed");
                    ++closes;
                    break;
                default:
                    throw std::runtime_error("unexpected remote action after receipt quarantine");
            }
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {61, 67};
            auto encoded = encode_execution_reply(reply);
            if (request->message.header.operation == ExecutionOpcode::kLaunch &&
                scenario.starts_with("busy-") && scenario != "busy-retry" &&
                scenario != "busy-wrong-operation") {
                expect(encoded.has_value(), "start from a canonical empty busy response");
                if (scenario == "busy-status")
                    (*encoded)[4] = std::byte{1};
                else if (scenario == "busy-original")
                    (*encoded)[8] = std::byte{1};
                else if (scenario == "busy-id")
                    (*encoded)[16] = std::byte{1};
                else if (scenario == "busy-address")
                    (*encoded)[24] = std::byte{1};
                else if (scenario == "busy-extent")
                    (*encoded)[32] = std::byte{1};
                else {
                    reply.disposition = ExecutionWireDisposition::kComplete;
                    if (scenario == "busy-arguments") {
                        reply.parameter_bytes = 8;
                        reply.arguments = {{0, 8}};
                    } else
                        reply.data = {std::byte{1}};
                    encoded = encode_execution_reply(reply);
                    expect(encoded.has_value(), "mutate only otherwise valid reply framing");
                    (*encoded)[0] = std::byte{5};
                }
            }
            expect(encoded && server.reply(request->connection, {header, std::move(*encoded)}),
                   "send codec-valid but operation-specific adversarial carrier");
        }
        expect(Clock::now() < deadline, "bound malformed-receipt observation and quarantine");
    }
    client.get();
    expect(glimmer_execution_lookup_count(2) == 0,
           "no rejected carrier can invoke caller-native resource or kernel fallback");
    expect(launches == (scenario == "busy-retry"             ? 3U
                        : scenario == "busy-wrong-operation" ? 0U
                        : is_clean || is_duplicate(scenario) ? 2U
                                                             : 1U),
           "uncertainty never retries/replays another remote original");
    expect(closes == (is_clean ? 1U : 0U) && captured.empty(),
           "only proven clean rejection allows normal context teardown");
    expect(async_releases == (is_clean ? 1U : 0U) &&
               (is_clean ? async_queries == 1U : is_duplicate(scenario) || async_queries == 0U),
           "exact pending ownership is observed, never discarded for a malformed second reply");
    server.stop();
    if (scenario == "busy-retry")
        expect(admission_replies == 1 && function_requests == 1,
               "busy attempts never become duplicate accepted commands or native metadata calls");
    if (scenario == "busy-wrong-operation")
        expect(function_requests == 1 && !launches,
               "wrong-operation busy quarantines without metadata retry or kernel issue");
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return EXIT_FAILURE;
    const std::string_view scenario(argv[1]);
    bool is_known{};
    for (const auto* candidate : kScenarios) is_known = is_known || scenario == candidate;
    if (!is_known) return EXIT_FAILURE;
    char pattern[] = "/tmp/glimmer-async-receipt-XXXXXX";
    const auto directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        if (scenario.starts_with("batch-"))
            run_batch(scenario, std::string(directory) + "/service.sock");
        else
            run(scenario, std::string(directory) + "/service.sock");
    } catch (const std::exception& error) {
        std::cerr << scenario << ": " << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
