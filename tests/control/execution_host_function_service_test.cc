#include "runtime.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Stream {
    unsigned int prefix{};
    bool is_non_blocking{};
};
struct Context {
    std::array<unsigned int, 4> functions{1, 2, 4, 8};
    Stream legacy;
    std::vector<std::unique_ptr<Stream>> streams;
    unsigned int modules{};
};
struct Event {
    unsigned int kernel{};
    bool is_recorded{};
};
struct Native {
    std::atomic<unsigned int> issued{}, stopped{}, failures{}, launches{}, creates{}, destroys{},
        event_creates{}, event_destroys{}, stream_creates{}, stream_destroys{}, module_loads{},
        module_unloads{}, context_reclaimed_streams{}, context_reclaimed_modules{};
    std::mutex contexts_mutex;
    std::vector<Context*> contexts;
};
Native* native;
thread_local Context* current_context;

CUresult CUDAAPI initialize(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device_get(CUdevice* result, int ordinal) {
    *result = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_get(CUcontext* result) {
    *result = reinterpret_cast<CUcontext>(current_context);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_create(CUcontext* result, unsigned int, CUdevice) {
    current_context = new Context;
    {
        std::lock_guard lock(native->contexts_mutex);
        native->contexts.push_back(current_context);
    }
    *result = reinterpret_cast<CUcontext>(current_context);
    ++native->creates;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_set(CUcontext value) {
    current_context = reinterpret_cast<Context*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_destroy(CUcontext value) {
    auto* context = reinterpret_cast<Context*>(value);
    std::lock_guard lock(native->contexts_mutex);
    const auto owned = std::ranges::find(native->contexts, context);
    expect(context && owned != native->contexts.end(),
           "native context cleanup destroys an exact live owner only once");
    const auto stopped = native->stopped.load();
    const auto is_stopped = [stopped](const Stream& stream) {
        return (stopped & stream.prefix) == stream.prefix;
    };
    expect(is_stopped(context->legacy) &&
               std::ranges::all_of(context->streams,
                                   [&](const auto& stream) { return is_stopped(*stream); }),
           "context cleanup cannot reclaim streams with unproved physical work");
    // cuCtxDestroy owns implicit stream/module cleanup and may target a
    // noncurrent context. Count those resources separately from explicit SDK
    // destruction instead of requiring APIs the native contract does not need.
    native->context_reclaimed_streams += static_cast<unsigned int>(context->streams.size());
    native->context_reclaimed_modules += context->modules;
    native->contexts.erase(owned);
    if (context == current_context) current_context = nullptr;
    delete context;
    ++native->destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_load(CUmodule* result, const void*) {
    *result = reinterpret_cast<CUmodule>(current_context);
    ++current_context->modules;
    ++native->module_loads;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_unload(CUmodule value) {
    auto* context = reinterpret_cast<Context*>(value);
    expect(context && context == current_context && context->modules != 0,
           "explicit module cleanup consumes its live owning context exactly once");
    --context->modules;
    ++native->module_unloads;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI function_get(CUfunction* result, CUmodule, const char* name) {
    const std::string_view text(name);
    expect(text.size() == 2 && text[0] == 'k' && text[1] >= '1' && text[1] <= '4',
           "only original named functions enter the native fake");
    *result = reinterpret_cast<CUfunction>(
        &current_context->functions[static_cast<std::size_t>(text[1] - '1')]);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI parameter_info(CUfunction, std::size_t, std::size_t*, std::size_t*) {
    throw std::runtime_error("zero-argument original functions need no SDK parameter probe");
}
CUresult CUDAAPI allocate(CUdeviceptr*, std::size_t) {
    return CUDA_ERROR_OUT_OF_MEMORY;
}
CUresult CUDAAPI release(CUdeviceptr) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI upload(CUdeviceptr, const void*, std::size_t) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI download(void*, CUdeviceptr, std::size_t) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI stream_create(CUstream* result, unsigned int flags) {
    current_context->streams.push_back(std::make_unique<Stream>());
    current_context->streams.back()->is_non_blocking = (flags & CU_STREAM_NON_BLOCKING) != 0;
    *result = reinterpret_cast<CUstream>(current_context->streams.back().get());
    ++native->stream_creates;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_destroy(CUstream value) {
    const auto removed = std::erase_if(current_context->streams, [value](const auto& stream) {
        return reinterpret_cast<CUstream>(stream.get()) == value;
    });
    expect(removed == 1, "destroy only the exact owned stream once");
    ++native->stream_destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_create(CUevent* result, unsigned int) {
    *result = reinterpret_cast<CUevent>(new Event);
    ++native->event_creates;
    return CUDA_SUCCESS;
}
Stream& ordered_stream(CUstream stream) {
    auto& selected = stream ? *reinterpret_cast<Stream*>(stream) : current_context->legacy;
    // Native SDK enqueue is asynchronous. Preserve its device-side ordering
    // independently of the callback's CPU boundary and command-local stop bits.
    // A legacy submission joins preceding blocking streams, never nonblocking
    // streams; a blocking submission joins the preceding legacy prefix.
    if (!stream) {
        for (const auto& previous : current_context->streams)
            if (!previous->is_non_blocking) selected.prefix |= previous->prefix;
    } else if (!selected.is_non_blocking) {
        selected.prefix |= current_context->legacy.prefix;
    }
    return selected;
}
CUresult CUDAAPI event_record(CUevent value, CUstream stream) {
    reinterpret_cast<Event*>(value)->kernel = ordered_stream(stream).prefix;
    reinterpret_cast<Event*>(value)->is_recorded = true;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_query(CUevent value) {
    expect(reinterpret_cast<Event*>(value)->is_recorded,
           "unrecorded event can never supply physical predecessor proof");
    const auto kernel = reinterpret_cast<Event*>(value)->kernel;
    if (native->failures.load() & kernel) return CUDA_ERROR_LAUNCH_FAILED;
    return (native->stopped.load() & kernel) == kernel ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent value) {
    delete reinterpret_cast<Event*>(value);
    ++native->event_destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_synchronize(CUstream) {
    throw std::runtime_error(
        "host-function readiness cannot replace event proof by synchronization");
}
CUresult CUDAAPI launch(CUfunction function, unsigned int, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, CUstream stream, void**, void**) {
    const auto kernel = *reinterpret_cast<unsigned int*>(function);
    ordered_stream(stream).prefix |= kernel;
    native->issued.fetch_or(kernel);
    ++native->launches;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_register(void*, std::size_t, unsigned int) {
    throw std::runtime_error("callback role checks must not register application host memory");
}
CUresult CUDAAPI host_unregister(void*) {
    throw std::runtime_error("callback role checks must not unregister application host memory");
}
CUresult CUDAAPI host_map(CUdeviceptr*, void*, unsigned int) {
    throw std::runtime_error("callback role checks must not map application host memory");
}
const CudaSessionDispatch kDispatch = [] {
    CudaSessionDispatch dispatch{{context_get, launch, event_create, event_record, event_query,
                                  event_destroy, stream_synchronize},
                                 initialize,
                                 device_get,
                                 context_create,
                                 context_set,
                                 context_destroy,
                                 module_load,
                                 module_unload,
                                 function_get,
                                 parameter_info,
                                 allocate,
                                 release,
                                 upload,
                                 download,
                                 stream_create,
                                 stream_destroy};
    dispatch.host_register = host_register;
    dispatch.host_unregister = host_unregister;
    dispatch.host_map = host_map;
    return dispatch;
}();

std::vector<std::byte> by_id(std::uint64_t id) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(id), "encode one scoped identity");
    return input.take();
}
std::vector<std::byte> finish_payload(std::uint64_t id, unsigned int returned = 1) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(id) && input.write_u32(returned), "encode callback return observation");
    return input.take();
}
ExecutionWireReply original(std::optional<ExecutionWireReply> result) {
    expect(result && result->status == CUDA_SUCCESS && result->original_status == CUDA_SUCCESS &&
               result->disposition == ExecutionWireDisposition::kComplete,
           "native SDK receipt is complete and distinct from synthetic callback metadata");
    return std::move(*result);
}
ExecutionWireReply synthetic(std::optional<ExecutionWireReply> result, std::uint64_t id) {
    expect(result && result->status == CUDA_SUCCESS && !result->original_status &&
               result->disposition == ExecutionWireDisposition::kComplete && result->id == id &&
               !result->address && !result->parameter_bytes && result->arguments.empty() &&
               result->data.empty(),
           "exact callback receipt grants no original SDK or extra resource authority");
    return std::move(*result);
}
void rejected(std::optional<ExecutionWireReply> result) {
    expect(result && result->status != CUDA_SUCCESS && !result->original_status &&
               result->disposition == ExecutionWireDisposition::kRejected,
           "invalid callback control rejects without original SDK or completion authority");
}
std::uint64_t submit(ExecutionSocketClient& client, ExecutionOpcode operation,
                     std::span<const std::byte> payload = {}) {
    const auto result = client.submit(operation, payload);
    expect(result.status == ExecutionSocketSubmitStatus::kAccepted,
           "bounded authenticated callback request is accepted by transport");
    return result.sequence;
}
ExecutionWireReply receive(ExecutionSocketClient& client, std::uint64_t sequence) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    for (;;) {
        if (auto result = client.take_reply(sequence)) return std::move(*result);
        expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
               "bounded receipt progress through actual production service");
    }
}
ExecutionWireReply request(ExecutionSocketClient& client, ExecutionOpcode operation,
                           std::span<const std::byte> payload = {}) {
    return receive(client, submit(client, operation, payload));
}
std::uint64_t enqueue(ExecutionSocketClient& client, std::uint64_t stream) {
    auto result = request(client, ExecutionOpcode::kEnqueueHostFunction, by_id(stream));
    const auto id = result.id;
    expect(id != 0, "accepted host-function identity is nonzero");
    synthetic(std::move(result), id);
    return id;
}
std::uint64_t create_stream(ExecutionSocketClient& client, unsigned int flags) {
    ExecutionPayloadWriter input;
    expect(input.write_u32(flags), "encode stream flags");
    return original(request(client, ExecutionOpcode::kCreateStream, input.take())).id;
}
std::array<std::uint64_t, 4> functions(ExecutionSocketClient& client) {
    ExecutionPayloadWriter source;
    expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                               ".visible .entry k1() { ret; }\n"
                               ".visible .entry k2() { ret; }\n"
                               ".visible .entry k3() { ret; }\n"
                               ".visible .entry k4() { ret; }"),
           "encode immutable original function source");
    const auto module = original(request(client, ExecutionOpcode::kLoadPtx, source.take()));
    std::array<std::uint64_t, 4> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("k" + std::to_string(index + 1)),
               "encode original named function lookup");
        result[index] = original(request(client, ExecutionOpcode::kGetFunction, lookup.take())).id;
    }
    return result;
}
std::vector<std::byte> kernel(std::uint64_t function, std::uint64_t stream) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(function) && input.write_u64(stream), "encode exact original handles");
    for (unsigned int axis = 0; axis < 6; ++axis)
        expect(input.write_u32(1), "encode original launch geometry");
    expect(input.write_u32(0) && input.write_blob({}), "encode unchanged zero-argument ABI");
    return input.take();
}
ExecutionServiceConfig config_for(const std::string& path) {
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 16;
    config.uses_original_kernels = true;
    config.priority_idle_us = 0;
    config.exit_when_idle = true;
    return config;
}
ExecutionSocketClientConfig client_config(const std::string& path) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "resolve exact trusted process identity");
    ExecutionSocketClientConfig config{path, *identity, 5000};
    config.max_pending_requests = 16;
    return config;
}
void advance(ExecutionService& service, std::future<void>& client) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded real owner-loop and authenticated client progress");
    client.get();
}
void verify_retired_resources() {
    expect(
        native->creates == native->destroys && native->event_creates == native->event_destroys &&
            native->stream_creates == native->stream_destroys + native->context_reclaimed_streams &&
            native->module_loads == native->module_unloads + native->context_reclaimed_modules,
        "all actual owned contexts, events, streams and modules retire exactly once");
    std::lock_guard lock(native->contexts_mutex);
    expect(native->contexts.empty(), "normal cleanup retains no live native context identity");
}
void drain(ExecutionService& service) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "normal callback/session cleanup finishes without fabricated completion");
    expect(service.request_stop(), "stop normally auto-retired callback service");
    verify_retired_resources();
}

void physical_ordering(const std::string& path) {
    Native state;
    native = &state;
    ExecutionService service(config_for(path), kDispatch);
    expect(service.start(), "start actual callback-ordering service");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient source(client_config(path));
        original(source.open());
        const auto entry = functions(source);
        const auto blocking = create_stream(source, CU_STREAM_DEFAULT);
        const auto independent = create_stream(source, CU_STREAM_NON_BLOCKING);
        original(request(source, ExecutionOpcode::kLaunch, kernel(entry[0], blocking)));
        expect(state.issued == 1 && state.stopped == 0,
               "real predecessor SDK success is independent of physical event stop");
        const auto first = enqueue(source, blocking);
        const auto second = enqueue(source, blocking);
        expect(second > first, "callback identities are nonrecycled within the source");
        rejected(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(second)));

        const auto same_stream =
            submit(source, ExecutionOpcode::kLaunch, kernel(entry[1], blocking));
        const auto legacy = submit(source, ExecutionOpcode::kLaunch, kernel(entry[2], 0));
        const auto unrelated =
            submit(source, ExecutionOpcode::kLaunch, kernel(entry[3], independent));
        const auto draining = submit(source, ExecutionOpcode::kDrain);
        const auto first_query = submit(source, ExecutionOpcode::kQueryHostFunction, by_id(first));
        const auto second_query =
            submit(source, ExecutionOpcode::kQueryHostFunction, by_id(second));
        original(receive(source, unrelated));
        for (unsigned int observation = 0; observation < 8; ++observation)
            expect(source.poll_for(1) && state.issued == 9 && !source.take_reply(first_query) &&
                       !source.take_reply(second_query) && !source.take_reply(same_stream) &&
                       !source.take_reply(legacy) && !source.take_reply(draining),
                   "independent stream progresses but no SDK return or CPU wait grants a callback");

        state.stopped.fetch_or(1);
        synthetic(receive(source, first_query), first);
        expect(!source.take_reply(second_query) && !source.take_reply(same_stream) &&
                   !source.take_reply(legacy) && !source.take_reply(draining) && state.issued == 9,
               "first callback permission preserves second callback and same/legacy stream holds");
        rejected(request(source, ExecutionOpcode::kQueryHostFunction, by_id(first)));
        rejected(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(second)));
        synthetic(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(first)),
                  first);
        synthetic(receive(source, second_query), second);
        expect(state.issued == 9 && !source.take_reply(same_stream) && !source.take_reply(legacy),
               "first callback return cannot release the second callback's boundary");
        synthetic(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(second)),
                  second);
        original(receive(source, same_stream));
        original(receive(source, legacy));
        expect(state.issued == 15 && !source.take_reply(draining),
               "after callback return native same-stream and legacy enqueue may pipeline");
        ExecutionPayloadWriter capture;
        expect(capture.write_u32(1) && capture.write_u64(0),
               "capture only the actual native legacy suffix");
        const auto legacy_frontier =
            request(source, ExecutionOpcode::kCaptureFrontier, capture.take());
        // Four ordinary kernels consumed public command identities. The two
        // internal callback barriers do not advance that public high-water mark.
        expect(legacy_frontier.status == CUDA_SUCCESS && !legacy_frontier.original_status &&
                   legacy_frontier.disposition == ExecutionWireDisposition::kComplete &&
                   legacy_frontier.id != 0 && legacy_frontier.address == 4 &&
                   legacy_frontier.parameter_bytes == 0 && legacy_frontier.arguments.empty() &&
                   legacy_frontier.data.empty(),
               "legacy capture carries only its exact identity and four-command high-water mark");
        // Make the legacy operation's own completion eligible while keeping its
        // blocking-stream predecessor held. A missing native dependency would
        // now falsely complete this exact frontier, independently of k4.
        state.stopped.fetch_or(4);
        const auto held =
            request(source, ExecutionOpcode::kQueryFrontier, by_id(legacy_frontier.id));
        expect(held.status == CUDA_ERROR_NOT_READY && !held.original_status &&
                   held.disposition == ExecutionWireDisposition::kComplete && !held.id &&
                   !held.address && held.parameter_bytes == 0 && held.arguments.empty() &&
                   held.data.empty() && !source.take_reply(draining),
               "native enqueue and local completion cannot bypass the legacy predecessor event");
        state.stopped.fetch_or(2);
        const auto legacy_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        for (;;) {
            const auto completed =
                request(source, ExecutionOpcode::kQueryFrontier, by_id(legacy_frontier.id));
            expect(!completed.original_status &&
                       completed.disposition == ExecutionWireDisposition::kComplete &&
                       !completed.id && !completed.address && completed.arguments.empty() &&
                       completed.data.empty(),
                   "frontier observation grants no callback or extra resource authority");
            if (completed.status == CUDA_ERROR_NOT_READY) {
                expect(completed.parameter_bytes == 0 &&
                           std::chrono::steady_clock::now() < legacy_deadline,
                       "legacy native dependency retires after its exact predecessor event");
                continue;
            }
            expect(completed.status == CUDA_SUCCESS && completed.parameter_bytes == 1,
                   "only the fully retired native legacy prefix publishes the physical-stop bit");
            break;
        }
        expect(!source.take_reply(draining) && (state.stopped.load() & 8U) == 0,
               "legacy physical completion neither requires nor retires the nonblocking suffix");
        synthetic(request(source, ExecutionOpcode::kReleaseFrontier, by_id(legacy_frontier.id)), 0);
        state.stopped.store(15);
        synthetic(receive(source, draining), 0);
        for (const auto id : {first, second}) {
            rejected(request(source, ExecutionOpcode::kQueryHostFunction, by_id(id)));
            rejected(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(id)));
        }
        original(request(source, ExecutionOpcode::kClose));
    });
    advance(service, client);
    drain(service);
    expect(state.launches == 4, "all four original commands issue exactly once");
}

void invalid_inputs_and_quota(const std::string& path) {
    Native state;
    native = &state;
    auto config = config_for(path);
    config.session_limits.max_objects = 2;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start bounded callback-validation service");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient source(client_config(path));
        original(source.open());
        for (const auto operation :
             {ExecutionOpcode::kEnqueueHostFunction, ExecutionOpcode::kQueryHostFunction,
              ExecutionOpcode::kFinishHostFunction}) {
            rejected(request(source, operation));
            auto trailing =
                operation == ExecutionOpcode::kFinishHostFunction ? finish_payload(1) : by_id(1);
            trailing.push_back(std::byte{0});
            rejected(request(source, operation, trailing));
        }
        rejected(request(source, ExecutionOpcode::kEnqueueHostFunction, by_id(UINT64_MAX)));
        for (const auto id : {std::uint64_t{0}, UINT64_MAX}) {
            rejected(request(source, ExecutionOpcode::kQueryHostFunction, by_id(id)));
            rejected(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(id)));
        }
        const auto first = enqueue(source, 0);
        const auto second = enqueue(source, 0);
        rejected(request(source, ExecutionOpcode::kEnqueueHostFunction, by_id(0)));
        rejected(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(first, 2)));

        ExecutionSocketClient foreign(client_config(path));
        original(foreign.open());
        rejected(request(foreign, ExecutionOpcode::kQueryHostFunction, by_id(first)));
        rejected(request(foreign, ExecutionOpcode::kFinishHostFunction, finish_payload(first)));
        original(request(foreign, ExecutionOpcode::kClose));

        ExecutionOwnedDescriptor descriptor(::open("/dev/null", O_RDONLY | O_CLOEXEC));
        expect(descriptor.get() >= 0, "open bounded borrowed descriptor for rejection control");
        for (const auto operation :
             {ExecutionOpcode::kEnqueueHostFunction, ExecutionOpcode::kQueryHostFunction,
              ExecutionOpcode::kFinishHostFunction}) {
            const auto input = operation == ExecutionOpcode::kFinishHostFunction
                                   ? finish_payload(first)
                                   : by_id(first);
            expect(source.submit(operation, input, descriptor.get()).status ==
                       ExecutionSocketSubmitStatus::kInvalid,
                   "transport cannot attach descriptors to any callback operation");
        }
        for (const auto id : {first, second}) {
            synthetic(request(source, ExecutionOpcode::kQueryHostFunction, by_id(id)), id);
            synthetic(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(id)),
                      id);
        }
        const auto third = enqueue(source, 0);
        expect(third > second, "capacity reuse never recycles callback identity");
        synthetic(request(source, ExecutionOpcode::kQueryHostFunction, by_id(third)), third);
        synthetic(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(third)),
                  third);
        original(request(source, ExecutionOpcode::kClose));
        expect(state.launches == 0 && state.event_creates == 0,
               "CPU callback acceptance, validation and completion create no native GPU work");
    });
    advance(service, client);
    drain(service);
}

void metadata_role_rejection(const std::string& path) {
    Native state;
    native = &state;
    auto config = config_for(path);
    config.exit_when_idle = false;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start helper-role callback exclusion service");
    auto client = std::async(std::launch::async, [&] {
        for (const auto role :
             {kExecutionPointerMetadataRole, kExecutionHostOwnerRole, kExecutionStagingOwnerRole}) {
            ExecutionSocketClient helper(client_config(path));
            ExecutionPayloadWriter bootstrap;
            expect(bootstrap.write_u32(role) && bootstrap.write_u32(0),
                   "encode actual helper role");
            const auto opened = helper.open(bootstrap.take());
            expect(opened && opened->status == CUDA_SUCCESS, "actual helper bootstrap succeeds");
            rejected(request(helper, ExecutionOpcode::kEnqueueHostFunction, by_id(0)));
            rejected(request(helper, ExecutionOpcode::kQueryHostFunction, by_id(1)));
            rejected(request(helper, ExecutionOpcode::kFinishHostFunction, finish_payload(1)));
            synthetic(request(helper, ExecutionOpcode::kClose), 0);
        }
    });
    advance(service, client);
    expect(service.request_stop(), "stop normally retired helper-only service");
    verify_retired_resources();
    expect(state.launches == 0 && state.event_creates == 0,
           "helper roles never gain GPU or callback execution authority");
}

void close_waits_for_callback(const std::string& path) {
    Native state;
    native = &state;
    ExecutionService service(config_for(path), kDispatch);
    expect(service.start(), "start close/metadata-progress service");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient source(client_config(path));
        original(source.open());
        const auto id = enqueue(source, 0);
        const auto closing = submit(source, ExecutionOpcode::kClose);
        const auto querying = submit(source, ExecutionOpcode::kQueryHostFunction, by_id(id));
        synthetic(receive(source, querying), id);
        expect(!source.take_reply(closing) && state.destroys == 0,
               "close cannot overtake a granted callback boundary");
        synthetic(request(source, ExecutionOpcode::kFinishHostFunction, finish_payload(id)), id);
        original(receive(source, closing));
    });
    advance(service, client);
    drain(service);
}

void disconnect_before_permission(const std::string& path) {
    Native state;
    native = &state;
    ExecutionService service(config_for(path), kDispatch);
    expect(service.start(), "start unpermitted callback disconnect service");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient source(client_config(path));
        original(source.open());
        const auto entry = functions(source);
        original(request(source, ExecutionOpcode::kLaunch, kernel(entry[0], 0)));
        const auto id = enqueue(source, 0);
        const auto querying = submit(source, ExecutionOpcode::kQueryHostFunction, by_id(id));
        const auto later = submit(source, ExecutionOpcode::kLaunch, kernel(entry[1], 0));
        for (unsigned int observation = 0; observation < 8; ++observation)
            expect(source.poll_for(1) && !source.take_reply(querying) && !source.take_reply(later),
                   "held predecessor prevents permission before real client disconnect");
    });
    advance(service, client);
    for (unsigned int turn = 0; turn < 20; ++turn)
        expect(service.step(0), "observe actual socket loss without inventing physical stop");
    expect(state.launches == 1 && state.destroys == 0 && !service.finished(),
           "canceling a never-permitted CPU callback cannot finish preceding GPU ownership");
    state.stopped.store(15);
    drain(service);
    expect(state.launches == 1,
           "queued successor never issues after disconnected source retirement");
}

enum class RetainedScenario { kGrantedDisconnect, kCallbackException, kPredecessorError };
void retained_uncertainty(const std::string& path, RetainedScenario scenario) {
    const auto child = ::fork();
    expect(child >= 0, "isolate deliberately retained uncertain native ownership");
    if (child == 0) {
        try {
            Native state;
            native = &state;
            ExecutionService service(config_for(path), kDispatch);
            expect(service.start(), "start isolated callback-retention service");
            auto client = std::async(std::launch::async, [&] {
                ExecutionSocketClient source(client_config(path));
                original(source.open());
                const auto entry = functions(source);
                if (scenario == RetainedScenario::kPredecessorError)
                    original(request(source, ExecutionOpcode::kLaunch, kernel(entry[0], 0)));
                const auto id = enqueue(source, 0);
                const auto later = submit(source, ExecutionOpcode::kLaunch, kernel(entry[1], 0));
                const auto querying =
                    submit(source, ExecutionOpcode::kQueryHostFunction, by_id(id));
                if (scenario == RetainedScenario::kPredecessorError) {
                    for (unsigned int observation = 0; observation < 8; ++observation)
                        expect(
                            source.poll_for(1) && !source.take_reply(querying),
                            "query cannot publish callback permission before failing predecessor");
                    state.failures.store(1);
                    const auto failed = receive(source, querying);
                    expect(failed.status != CUDA_SUCCESS && !failed.original_status &&
                               failed.disposition == ExecutionWireDisposition::kComplete &&
                               failed.id == id && failed.parameter_bytes == 1 && !failed.address &&
                               failed.arguments.empty() && failed.data.empty(),
                           "known-not-run callback error cancels only its CPU boundary");
                } else {
                    synthetic(receive(source, querying), id);
                    if (scenario == RetainedScenario::kCallbackException)
                        rejected(request(source, ExecutionOpcode::kFinishHostFunction,
                                         finish_payload(id, 0)));
                }
                expect((state.issued.load() & 2) == 0,
                       "uncertain or failed callback never permits successor native issue");
                // The stream successor has not entered the SDK. Its receipt may
                // already be an explicit error after the predecessor failed.
                if (const auto reply = source.take_reply(later))
                    expect(reply->status != CUDA_SUCCESS && !reply->original_status,
                           "no source failure creates a successful original successor");
            });
            advance(service, client);
            for (unsigned int turn = 0; turn < 40; ++turn)
                expect(service.step(0),
                       "observe retained callback/error ownership after disconnect");
            expect(!service.finished() && state.destroys == 0 && (state.issued.load() & 2) == 0,
                   "socket loss and callback exceptions cannot release unproven ownership");
            // The contract intentionally retains uncertainty until process death.
            // Other scenarios separately require normal destructor/leak-free cleanup.
            ::_exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            ::_exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
               WEXITSTATUS(status) == EXIT_SUCCESS,
           "isolated callback-retention scenario passes without signaling any process");
}
}  // namespace

int main() {
    try {
        const auto path = "/tmp/glimmer-host-function-service-" + std::to_string(::getpid());
        physical_ordering(path + "-ordering");
        invalid_inputs_and_quota(path + "-validation");
        metadata_role_rejection(path + "-roles");
        close_waits_for_callback(path + "-close");
        disconnect_before_permission(path + "-before");
        retained_uncertainty(path + "-after", RetainedScenario::kGrantedDisconnect);
        retained_uncertainty(path + "-exception", RetainedScenario::kCallbackException);
        retained_uncertainty(path + "-error", RetainedScenario::kPredecessorError);
        std::cout << "Actual service host-function ordering, ownership and failure checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
