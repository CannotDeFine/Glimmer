#include "internal/cuda_execution_session.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;

void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned setup or cleanup succeeds");
}
void deferred(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kDeferred &&
               result.status == CUDA_ERROR_NOT_READY && !result.original_status,
           "retained owner defers cleanup without requested native provenance");
}
struct NativeContext {
    bool alive{};
    int device{};
};
struct NativeModule {
    bool alive{};
    CUcontext context{};
    char function{};
    std::array<std::byte, 32> global{};
};
struct NativeStream {
    bool alive{};
    CUcontext context{};
};
struct NativeEvent {
    bool alive{};
    CUcontext context{};
    bool is_recorded{};
};
struct Memory {
    CUcontext context{};
    std::vector<std::byte> bytes;
};
enum class Cleanup { kAllocation, kModule, kStream };
struct Driver {
    CUcontext current{};
    std::array<NativeContext, 8> contexts{};
    std::array<NativeModule, 16> modules{};
    std::array<NativeStream, 16> streams{};
    std::array<NativeEvent, 16> events{};
    std::map<CUdeviceptr, Memory> memory;
    std::size_t calls{}, proofs{}, context_creates{}, context_destroys{}, frees{}, unloads{};
    std::size_t stream_destroys{}, launches{}, event_creates{}, event_queries{}, syncs{};
    bool can_stop{};
    CUdeviceptr next_allocation_address{};
    CUresult proof_status{CUDA_SUCCESS}, release_status{CUDA_SUCCESS};
    std::function<void()> proof_callback;
    std::function<void()> load_callback;
    std::function<void(Cleanup)> cleanup_callback;
} driver;

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact injected CUDA ABI signatures.
CudaSessionDispatch dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int flags) {
        ++driver.calls;
        expect(flags == 0, "initialization flags remain zero");
        return CUDA_SUCCESS;
    };
    sdk.device_get = [](CUdevice* device, int ordinal) {
        ++driver.calls;
        *device = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned int, CUdevice device) {
        ++driver.calls;
        expect(driver.context_creates < driver.contexts.size(), "bounded fake context inventory");
        auto& context = driver.contexts[driver.context_creates++];
        context = {true, device};
        *output = reinterpret_cast<CUcontext>(&context);
        driver.current = *output;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext context) {
        ++driver.calls;
        driver.current = context;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* output) {
        ++driver.calls;
        ++driver.proofs;
        if (driver.proof_callback) driver.proof_callback();
        *output = driver.current;
        return driver.proof_status;
    };
    sdk.context_destroy = [](CUcontext context) {
        ++driver.calls;
        expect(context && context == driver.current,
               "context destruction requires the exact owner");
        auto& native = *reinterpret_cast<NativeContext*>(context);
        expect(native.alive, "native context is destroyed once");
        native.alive = false;
        ++driver.context_destroys;
        std::erase_if(driver.memory,
                      [context](const auto& entry) { return entry.second.context == context; });
        for (auto& module : driver.modules)
            if (module.context == context) module.alive = false;
        for (auto& stream : driver.streams)
            if (stream.context == context) stream.alive = false;
        driver.current = nullptr;
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule* output, const void*) {
        ++driver.calls;
        if (driver.load_callback) driver.load_callback();
        const auto found = std::ranges::find(driver.modules, false, &NativeModule::alive);
        expect(found != driver.modules.end(), "bounded fake module inventory");
        found->alive = true;
        found->context = driver.current;
        *output = reinterpret_cast<CUmodule>(&*found);
        return CUDA_SUCCESS;
    };
    sdk.module_unload = [](CUmodule module) {
        ++driver.calls;
        if (driver.cleanup_callback) driver.cleanup_callback(Cleanup::kModule);
        auto& native = *reinterpret_cast<NativeModule*>(module);
        expect(native.alive && native.context == driver.current, "unload exact live module owner");
        native.alive = false;
        ++driver.unloads;
        return CUDA_SUCCESS;
    };
    sdk.module_global = [](CUdeviceptr* output, std::size_t* bytes, CUmodule module, const char*) {
        ++driver.calls;
        auto& native = *reinterpret_cast<NativeModule*>(module);
        expect(native.alive && native.context == driver.current,
               "global belongs to the current module");
        *output = reinterpret_cast<CUdeviceptr>(native.global.data());
        *bytes = native.global.size();
        return CUDA_SUCCESS;
    };
    sdk.function_get = [](CUfunction* output, CUmodule module, const char*) {
        ++driver.calls;
        auto& native = *reinterpret_cast<NativeModule*>(module);
        *output = reinterpret_cast<CUfunction>(&native.function);
        return CUDA_SUCCESS;
    };
    sdk.parameter_info = [](CUfunction, std::size_t index, std::size_t* offset,
                            std::size_t* bytes) {
        ++driver.calls;
        expect(index == 0, "independent original ABI has exactly one pointer parameter");
        *offset = 0;
        *bytes = 8;
        return CUDA_SUCCESS;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        ++driver.calls;
        Memory value{driver.current, std::vector<std::byte>(bytes)};
        *output = driver.next_allocation_address
                      ? driver.next_allocation_address
                      : reinterpret_cast<CUdeviceptr>(value.bytes.data());
        driver.next_allocation_address = 0;
        expect(driver.memory.emplace(*output, std::move(value)).second,
               "distinct live allocations");
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr pointer) {
        ++driver.calls;
        ++driver.frees;
        if (driver.cleanup_callback) driver.cleanup_callback(Cleanup::kAllocation);
        if (driver.release_status != CUDA_SUCCESS) return driver.release_status;
        const auto found = driver.memory.find(pointer);
        expect(found != driver.memory.end() && found->second.context == driver.current,
               "release exact live allocation owner once");
        driver.memory.erase(found);
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr, const void*, std::size_t) {
        ++driver.calls;
        return CUDA_SUCCESS;
    };
    sdk.download = [](void*, CUdeviceptr, std::size_t) {
        ++driver.calls;
        return CUDA_SUCCESS;
    };
    sdk.stream_create = [](CUstream* output, unsigned int) {
        ++driver.calls;
        const auto found = std::ranges::find(driver.streams, false, &NativeStream::alive);
        expect(found != driver.streams.end(), "bounded fake stream inventory");
        *found = {true, driver.current};
        *output = reinterpret_cast<CUstream>(&*found);
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream stream) {
        ++driver.calls;
        if (driver.cleanup_callback) driver.cleanup_callback(Cleanup::kStream);
        auto& native = *reinterpret_cast<NativeStream*>(stream);
        expect(native.alive && native.context == driver.current, "destroy exact live stream once");
        native.alive = false;
        ++driver.stream_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) {
        ++driver.calls;
        ++driver.launches;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_create = [](CUevent* output, unsigned int) {
        ++driver.calls;
        const auto found = std::ranges::find(driver.events, false, &NativeEvent::alive);
        expect(found != driver.events.end(), "bounded physical event inventory");
        *found = {true, driver.current};
        *output = reinterpret_cast<CUevent>(&*found);
        ++driver.event_creates;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent event, CUstream stream) {
        ++driver.calls;
        auto& native = *reinterpret_cast<NativeEvent*>(event);
        expect(native.alive && !native.is_recorded && native.context == driver.current,
               "record exactly one event on its actual native context");
        expect(!stream || reinterpret_cast<NativeStream*>(stream)->context == driver.current,
               "external tail records on its exact source stream");
        native.is_recorded = true;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent event) {
        ++driver.calls;
        ++driver.event_queries;
        expect(reinterpret_cast<NativeEvent*>(event)->is_recorded,
               "accepted or started unrecorded tails are never queried");
        return driver.can_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent event) {
        ++driver.calls;
        auto& native = *reinterpret_cast<NativeEvent*>(event);
        expect(native.alive && native.context == driver.current, "destroy one proven event");
        native.alive = false;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        ++driver.calls;
        ++driver.syncs;
        return driver.can_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    return sdk;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

constexpr std::string_view kSource =
    ".version 7.0\n.target sm_70\n.address_size 64\n"
    ".visible .global .align 4 .u32 variable[8];\n"
    ".visible .entry kernel(.param .u64 pointer) { ret; }\n";
struct Owner {
    CudaExecutionSession session;
    CUcontext native{};
    explicit Owner(CudaSessionLimits limits = {}, std::uint64_t id = 17)
        : session(ExecutionSessionIdentity{11, 13}, id, dispatch(), limits) {
        success(session.initialize(5));
        native = driver.current;
    }
    CudaSessionResult allocation() {
        auto result = session.allocate(32);
        success(result);
        return result;
    }
    CudaSessionResult stream() {
        auto result = session.create_stream(0);
        success(result);
        return result;
    }
    CudaSessionResult module() {
        auto result = session.load_ptx(kSource);
        success(result);
        return result;
    }
    void close() {
        success(session.activate());
        success(session.shutdown());
    }
};
struct LeaseCounts {
    std::size_t contexts{}, commands{};
};
auto scheduling_fields(const CudaSessionSchedulingSnapshot& value) {
    const auto& queue = value.submissions;
    return std::tuple(value.has_context, value.is_quarantined, value.has_ready_work,
                      value.external_barriers, value.enqueue_turns, value.external_work_in_flight,
                      value.has_pending_preparation, queue.outstanding, queue.queued, queue.issuing,
                      queue.awaiting_physical_stop, queue.paused, queue.unacknowledged_replies,
                      queue.retained_parameter_bytes, queue.retained_host_bytes, queue.has_executor,
                      queue.is_closing, queue.is_quarantined);
}
CudaSessionSchedulingSnapshot check_scheduling(const CudaExecutionSession& session) {
    const auto before = driver.calls;
    const auto health = session.health_snapshot();
    const auto light = session.scheduling_snapshot();
    const auto full = session.snapshot();
    expect(health.has_context == light.has_context && health.is_quarantined == light.is_quarantined,
           "owner health agrees with full scheduling state without requesting readiness");
    expect(scheduling_fields(light) == scheduling_fields(full),
           "fresh lightweight and diagnostic observations agree on all scheduling fields");
    expect(driver.calls == before, "neither observation issues SDK work or native proof");
    return light;
}
void counts(const CudaExecutionSession& session, LeaseCounts expected) {
    check_scheduling(session);
    const auto value = session.snapshot();
    expect(value.external_context_leases == expected.contexts &&
               value.external_command_leases == expected.commands,
           "live unique capsule registrations match the independent owner count");
}
void absent(const CudaSessionExternalContextCapture& result, CUresult status) {
    expect(result.status == status && !result.context.native() && result.context.id() == 0 &&
               result.context.device_ordinal() == -1,
           "rejected context capture publishes no authority");
}
void absent(const CudaSessionExternalCommandCapture& result, CUresult status) {
    expect(result.status == status && !result.command.context().native() &&
               !result.command.stream() && result.command.stream_id() == 0 &&
               result.command.ranges().empty(),
           "rejected command capture publishes no partial authority");
}
CudaSessionExternalContext context(Owner& owner) {
    auto captured = owner.session.capture_external_context();
    expect(captured.status == CUDA_SUCCESS && captured.context.native() == owner.native &&
               captured.context.id() == 17 && captured.context.device_ordinal() == 5,
           "capsule records the actual source context and device, not an execution-context claim");
    return std::move(captured.context);
}
CudaSessionExternalCommand command(Owner& owner, std::uint64_t stream,
                                   std::span<const CudaSessionExternalRangeRequest> ranges) {
    auto captured = owner.session.capture_external_command(stream, ranges);
    expect(captured.status == CUDA_SUCCESS && captured.command.context().native() == owner.native &&
               captured.command.stream_id() == stream &&
               captured.command.ranges().size() == ranges.size(),
           "capture returns the complete ordered resource set");
    return std::move(captured.command);
}
void context_only_and_default() {
    driver = {};
    Owner owner;
    CudaSessionExternalContext empty;
    CudaSessionExternalCommand no_command;
    expect(!empty.native() && empty.id() == 0 && empty.device_ordinal() == -1 &&
               !no_command.context().native() && no_command.ranges().empty() &&
               !no_command.stream() && no_command.stream_id() == 0,
           "empty capsules are inert");
    auto retained = context(owner);
    auto copy = retained;
    counts(owner.session, {1, 0});
    const auto memory = owner.allocation();
    const auto stream = owner.stream();
    const auto module = owner.module();
    success(owner.session.release(memory.id));
    success(owner.session.destroy_stream(stream.id));
    success(owner.session.unload_module(module.id));
    expect(driver.frees == 1 && driver.stream_destroys == 1 && driver.unloads == 1,
           "context-only retention permits unrelated allocation, stream and module cleanup");
    retained = {};
    counts(owner.session, {1, 0});
    const auto before = driver.calls;
    std::thread worker([held = std::move(copy), expected = owner.native]() mutable {
        expect(held.native() == expected && held.id() == 17 && held.device_ordinal() == 5,
               "worker reads immutable context identity");
        auto another = held;
        held = {};
        another = {};
    });
    worker.join();
    expect(driver.calls == before, "foreign-thread context copy and release invoke no SDK");
    counts(owner.session, {0, 0});
    auto default_command = command(owner, 0, {});
    expect(!default_command.stream() && default_command.ranges().empty(),
           "default stream and empty ranges remain a valid context-retaining command");
    counts(owner.session, {1, 1});
    default_command = {};
    counts(owner.session, {0, 0});
    owner.close();
    expect(driver.context_destroys == 1 && driver.event_creates == 0 && driver.syncs == 0,
           "retention alone neither launches, synchronizes nor creates physical events");
}
void exact_resources_and_context_copy() {
    driver = {};
    Owner owner;
    const auto first = owner.allocation();
    const auto unrelated = owner.allocation();
    const auto stream = owner.stream();
    const auto other_stream = owner.stream();
    const auto module = owner.module();
    const auto other_module = owner.module();
    const auto global = owner.session.get_global(module.id, "variable");
    success(global);
    std::array<CudaSessionExternalRangeRequest, 3> requests{
        {{{first.id, 3}, 7}, {{first.id, 32}, 0}, {{global.id, 4}, 8}}};
    auto retained = command(owner, stream.id, requests);
    const auto captured_ranges = retained.ranges();
    expect(captured_ranges[0].range.allocation == first.id &&
               captured_ranges[0].range.offset == 3 &&
               captured_ranges[0].address == first.address + 3 && captured_ranges[0].bytes == 7 &&
               captured_ranges[1].address == first.address + 32 && captured_ranges[1].bytes == 0 &&
               captured_ranges[2].address == global.address + 4 && captured_ranges[2].bytes == 8,
           "duplicate/interior/zero-endpoint/global ranges preserve exact order and extents");
    const auto actual_stream = retained.stream();
    expect(actual_stream && reinterpret_cast<NativeStream*>(actual_stream)->context == owner.native,
           "capsule retains the actual stream from the source context");
    requests = {};
    expect(retained.ranges()[0].bytes == 7 && retained.ranges()[2].range.allocation == global.id,
           "capture owns its range metadata independently of caller storage");
    success(owner.session.release(unrelated.id));
    success(owner.session.destroy_stream(other_stream.id));
    success(owner.session.unload_module(other_module.id));
    const auto frees = driver.frees;
    const auto unloads = driver.unloads;
    const auto destroys = driver.stream_destroys;
    deferred(owner.session.release(first.id));
    deferred(owner.session.destroy_stream(stream.id));
    deferred(owner.session.unload_module(module.id));
    expect(driver.frees == frees && driver.unloads == unloads && driver.stream_destroys == destroys,
           "target cleanup cannot reach native callbacks while exact command owners remain");
    auto copy = retained;
    auto source_context = retained.context();
    retained = {};
    counts(owner.session, {1, 1});
    const auto before = driver.calls;
    std::thread worker([held = std::move(copy), actual_stream]() mutable {
        expect(held.ranges().size() == 3 && held.stream() == actual_stream,
               "worker reads immutable stream/range identities");
        auto another = held;
        held = {};
        another = {};
    });
    worker.join();
    expect(driver.calls == before, "command destruction is CPU-only even on the final worker copy");
    counts(owner.session, {1, 0});
    success(owner.session.release(first.id));
    success(owner.session.destroy_stream(stream.id));
    success(owner.session.unload_module(module.id));
    deferred(owner.session.shutdown());
    expect(driver.context_destroys == 0,
           "context() copy prolongs source context after command release");
    absent(owner.session.capture_external_context(), CUDA_ERROR_DEINITIALIZED);
    absent(owner.session.capture_external_command(0, {}), CUDA_ERROR_DEINITIALIZED);
    source_context = {};
    counts(owner.session, {0, 0});
    success(owner.session.shutdown());
    expect(driver.context_destroys == 1 && driver.memory.empty(),
           "known owner cleans up once after final capsule");
}
void rejected_ranges_and_foreign_ids() {
    driver = {};
    CudaSessionLimits limits;
    limits.max_parameters = 2;
    Owner owner(limits);
    const auto memory = owner.allocation();
    const auto stream = owner.stream();
    const auto stale = owner.allocation();
    success(owner.session.release(stale.id));
    const auto stale_stream = owner.stream();
    success(owner.session.destroy_stream(stale_stream.id));
    const std::array<CudaSessionExternalRangeRequest, 8> invalid{
        {{{0, 0}, 0},
         {{stale.id, 0}, 1},
         {{memory.id, 32}, 1},
         {{memory.id, 33}, 0},
         {{memory.id, std::numeric_limits<std::size_t>::max()}, 1},
         {{memory.id, 1}, std::numeric_limits<std::size_t>::max()},
         {{UINT64_MAX, 0}, 0},
         {{memory.id, 0}, 33}}};
    for (const auto& bad : invalid) {
        const std::array<CudaSessionExternalRangeRequest, 2> ranges{{{{memory.id, 0}, 4}, bad}};
        absent(owner.session.capture_external_command(stream.id, ranges), CUDA_ERROR_INVALID_VALUE);
        counts(owner.session, {0, 0});
    }
    const std::array<CudaSessionExternalRangeRequest, 3> too_many{
        {{{memory.id, 0}, 0}, {{memory.id, 1}, 0}, {{memory.id, 2}, 0}}};
    absent(owner.session.capture_external_command(stream.id, too_many), CUDA_ERROR_INVALID_VALUE);
    absent(owner.session.capture_external_command(stale_stream.id, {}), CUDA_ERROR_INVALID_HANDLE);
    absent(owner.session.capture_external_command(UINT64_MAX, {}), CUDA_ERROR_INVALID_HANDLE);
    const std::array<CudaSessionExternalRangeRequest, 2> endpoints{
        {{{memory.id, 0}, 0}, {{memory.id, 32}, 0}}};
    auto zero_extent = command(owner, stream.id, endpoints);
    deferred(owner.session.release(memory.id));
    expect(zero_extent.ranges()[1].address == memory.address + 32,
           "zero-byte ranges at the exact range-count limit still retain their allocation");
    zero_extent = {};
    success(owner.session.deactivate());
    CudaSessionLimits other_limits;
    other_limits.first_resource_id = 10000;
    Owner other(other_limits, 19);
    const auto foreign = other.allocation();
    const auto foreign_stream = other.stream();
    absent(owner.session.capture_external_context(), CUDA_ERROR_INVALID_CONTEXT);
    success(other.session.deactivate());
    success(owner.session.activate());
    const std::array<CudaSessionExternalRangeRequest, 2> ranges{
        {{{memory.id, 0}, 4}, {{foreign.id, 0}, 4}}};
    absent(owner.session.capture_external_command(stream.id, ranges), CUDA_ERROR_INVALID_VALUE);
    absent(owner.session.capture_external_command(foreign_stream.id, {}),
           CUDA_ERROR_INVALID_HANDLE);
    counts(owner.session, {0, 0});
    success(owner.session.release(memory.id));
    success(owner.session.destroy_stream(stream.id));
    expect(driver.launches == 0 && driver.event_queries == 0 && driver.syncs == 0,
           "all-or-none rejection leaves no partial pins, command, poll or drain");
    owner.close();
    success(other.session.activate());
    success(other.session.release(foreign.id));
    success(other.session.destroy_stream(foreign_stream.id));
    other.close();
    expect(driver.context_destroys == 2 && driver.memory.empty(),
           "two distinct session owners clean independently");
}
void bounded_registrations() {
    driver = {};
    CudaSessionLimits limits;
    limits.max_objects = 2;
    limits.submissions.max_commands = 1;
    Owner owner(limits);
    auto first = context(owner);
    auto second = context(owner);
    absent(owner.session.capture_external_context(), CUDA_ERROR_OUT_OF_MEMORY);
    absent(owner.session.capture_external_command(0, {}), CUDA_ERROR_OUT_OF_MEMORY);
    auto copy = first;
    first = {};
    counts(owner.session, {2, 0});
    absent(owner.session.capture_external_context(), CUDA_ERROR_OUT_OF_MEMORY);
    copy = {};
    first = context(owner);
    counts(owner.session, {2, 0});
    first = {};
    second = {};
    for (unsigned int repeat = 0; repeat != 4; ++repeat) {
        auto held = command(owner, 0, {});
        auto held_copy = held;
        auto context_copy = held.context();
        auto separate = context(owner);
        counts(owner.session, {2, 1});
        absent(owner.session.capture_external_command(0, {}), CUDA_ERROR_OUT_OF_MEMORY);
        absent(owner.session.capture_external_context(), CUDA_ERROR_OUT_OF_MEMORY);
        held = {};
        held_copy = {};
        counts(owner.session, {2, 0});
        absent(owner.session.capture_external_command(0, {}), CUDA_ERROR_OUT_OF_MEMORY);
        context_copy = {};
        auto reused = command(owner, 0, {});
        counts(owner.session, {2, 1});
        reused = {};
        separate = {};
        counts(owner.session, {0, 0});
    }
    // Distinguish the command bound from the context bound: one free context
    // slot does not permit a second live command registration.
    auto held = command(owner, 0, {});
    counts(owner.session, {1, 1});
    absent(owner.session.capture_external_command(0, {}), CUDA_ERROR_OUT_OF_MEMORY);
    auto separate = context(owner);
    counts(owner.session, {2, 1});
    held = {};
    separate = {};
    owner.close();
}
void native_address_bounds_and_preparation() {
    driver = {};
    Owner owner;
    // A synthetic native address exercises checked pointer arithmetic without
    // allocating or dereferencing memory at that address.
    driver.next_allocation_address = std::numeric_limits<CUdeviceptr>::max() - 3;
    const auto memory = owner.session.allocate(4);
    success(memory);
    for (const auto bad : std::array<CudaSessionExternalRangeRequest, 2>{
             {{{memory.id, 0}, 5}, {{memory.id, 4}, 0}}}) {
        absent(owner.session.capture_external_command(0, std::span{&bad, 1}),
               CUDA_ERROR_INVALID_VALUE);
        counts(owner.session, {0, 0});
    }
    const std::array<CudaSessionExternalRangeRequest, 1> last_byte{{{{memory.id, 3}, 1}}};
    auto held = command(owner, 0, last_byte);
    expect(held.ranges()[0].address == std::numeric_limits<CUdeviceptr>::max(),
           "last representable byte is valid without overflowing the inclusive extent");
    held = {};
    success(owner.session.release(memory.id));
    unsigned int callbacks{};
    driver.load_callback = [&] {
        ++callbacks;
        const auto calls = driver.calls;
        absent(owner.session.capture_external_context(), CUDA_ERROR_NOT_READY);
        absent(owner.session.capture_external_command(0, {}), CUDA_ERROR_NOT_READY);
        expect(driver.calls == calls,
               "cold native preparation cannot recursively publish capsules");
    };
    const auto module = owner.module();
    driver.load_callback = {};
    expect(callbacks == 1,
           "preparation exclusion is tested at an actual original module-load callback");
    counts(owner.session, {0, 0});
    success(owner.session.unload_module(module.id));
    owner.close();
}
void pending_work_is_not_a_barrier() {
    driver = {};
    Owner owner;
    const auto memory = owner.allocation();
    const auto stream = owner.stream();
    const auto module = owner.module();
    const auto function = owner.session.get_function(module.id, "kernel");
    success(function);
    std::array<std::byte, 8> parameters{};
    std::memcpy(parameters.data(), &memory.address, sizeof(memory.address));
    const CudaExecutionKernel kernel{.function = function.id,
                                     .stream = stream.id,
                                     .grid = {1, 1, 1},
                                     .block = {32, 1, 1},
                                     .parameters = parameters};
    const auto accepted = owner.session.accept_kernel(1, kernel);
    expect(accepted.status == ExecutionAcceptStatus::kAccepted, "real queued kernel accepted");
    expect(check_scheduling(owner.session).submissions.queued == 1,
           "lightweight observation retains accepted queued work");
    const std::array<CudaSessionExternalRangeRequest, 1> ranges{{{{memory.id, 0}, 32}}};
    auto queued_capture = command(owner, stream.id, ranges);
    auto queued_context = context(owner);
    expect(driver.launches == 0 && driver.event_creates == 0 && driver.event_queries == 0 &&
               driver.syncs == 0 && owner.session.snapshot().submissions.queued == 1,
           "capture neither drains nor consumes existing queued work");
    expect(owner.session.issue_next().state == CudaExecutionIssueState::kIssued &&
               owner.session.acknowledge_result(accepted.receipt),
           "queued work issues and ACKs normally while retained");
    const auto queries = driver.event_queries;
    const auto syncs = driver.syncs;
    auto active_capture = command(owner, stream.id, ranges);
    expect(driver.launches == 1 && driver.event_queries == queries && driver.syncs == syncs,
           "capture while issued does not poll, synchronize or replay");
    queued_capture = {};
    queued_context = {};
    active_capture = {};
    counts(owner.session, {0, 0});
    expect(owner.session.snapshot().submissions.awaiting_physical_stop == 1,
           "dropping every external capsule does not invent a physical stop after host ACK");
    deferred(owner.session.release(memory.id));
    driver.can_stop = true;
    expect(owner.session.poll().retired == 1, "independent actual event stop retires once");
    expect(check_scheduling(owner.session).submissions.outstanding == 0,
           "lightweight observation retires work only after actual stop");
    auto retained = command(owner, stream.id, ranges);
    const auto canceled = owner.session.accept_kernel(2, kernel);
    expect(canceled.status == ExecutionAcceptStatus::kAccepted &&
               owner.session.reject_queued(CUDA_ERROR_INVALID_VALUE) == 1 &&
               owner.session.acknowledge_result(canceled.receipt),
           "queued cancellation remains independent of external holds");
    expect(driver.launches == 1, "canceled work never reaches native launch");
    expect(check_scheduling(owner.session).submissions.outstanding == 0,
           "lightweight observation reflects acknowledged queued cancellation");
    deferred(owner.session.release(memory.id));
    retained = {};
    success(owner.session.release(memory.id));
    success(owner.session.destroy_stream(stream.id));
    success(owner.session.unload_module(module.id));
    owner.close();
}
void thread_and_process_boundaries() {
    driver = {};
    Owner owner;
    const auto memory = owner.allocation();
    const auto stream = owner.stream();
    const std::array<CudaSessionExternalRangeRequest, 1> ranges{{{{memory.id, 5}, 9}}};
    auto held = command(owner, stream.id, ranges);
    auto held_context = context(owner);
    const auto before = driver.calls;
    std::thread wrong_owner([&] {
        const auto state = check_scheduling(owner.session);
        expect(state.is_quarantined && !state.has_context && !state.submissions.outstanding,
               "wrong-thread scheduling observation rejects without exposing owner state");
        absent(owner.session.capture_external_context(), CUDA_ERROR_INVALID_CONTEXT);
        absent(owner.session.capture_external_command(stream.id, ranges),
               CUDA_ERROR_INVALID_CONTEXT);
        expect(held.context().native() == owner.native &&
                   held.ranges()[0].address == memory.address + 5,
               "cross-thread observations transfer no mutable session authority");
    });
    wrong_owner.join();
    expect(driver.calls == before, "wrong-thread capture rejects before the first native proof");
    const auto pid = ::fork();
    expect(pid >= 0, "fork capsule process-boundary control");
    if (pid == 0) {
        try {
            expect(!held_context.native() && held_context.id() == 0 &&
                       held_context.device_ordinal() == -1 && !held.context().native() &&
                       !held.stream() && held.stream_id() == 0 && held.ranges().empty(),
                   "forked inherited getters expose no native identities or ranges");
            const auto state = check_scheduling(owner.session);
            expect(state.is_quarantined && !state.has_context && !state.submissions.has_executor,
                   "forked scheduling observation rejects before inherited queue locks");
            absent(owner.session.capture_external_context(), CUDA_ERROR_INVALID_CONTEXT);
            absent(owner.session.capture_external_command(stream.id, ranges),
                   CUDA_ERROR_INVALID_CONTEXT);
            held = {};
            held_context = {};
            expect(driver.calls == before,
                   "inherited copies and release cannot invoke native callbacks");
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork process-boundary assertions pass");
    expect(held.context().native() == owner.native && held.ranges().size() == 1,
           "child release cannot retire parent-process owners");
    deferred(owner.session.release(memory.id));
    held = {};
    held_context = {};
    success(owner.session.release(memory.id));
    success(owner.session.destroy_stream(stream.id));
    owner.close();
}
void external_barrier_retains_actual_source() {
    driver = {};
    Owner owner;
    const auto memory = owner.allocation();
    const auto stream = owner.stream();
    const auto module = owner.module();
    const auto global = owner.session.get_global(module.id, "variable");
    success(global);
    const std::array<CudaSessionExternalRangeRequest, 2> ranges{
        {{{memory.id, 2}, 12}, {{global.id, 4}, 8}}};
    auto lifetime = std::make_shared<CudaSessionExternalCommand>(command(owner, stream.id, ranges));
    const auto before = driver.calls;
    const auto external = owner.session.accept_external_barrier(stream.id, lifetime);
    expect(external.status == CUDA_SUCCESS && driver.calls == before &&
               external.frontier.observe() == CUDA_ERROR_NOT_READY &&
               owner.session.external_barrier_ready(external.barrier) == CUDA_SUCCESS,
           "session accepts a CPU-only boundary with no native call or completion claim");
    lifetime.reset();
    counts(owner.session, {1, 1});
    const auto snapshot = owner.session.snapshot();
    expect(snapshot.external_barriers == 1 && snapshot.submissions.outstanding == 0 &&
               !snapshot.has_ready_work && !owner.session.native_kernel_pair(),
           "external boundary is counted separately without fake ordinary work or audit credit");
    const auto stream_frontier = owner.session.capture_frontier(stream.id);
    const auto context_frontier = owner.session.capture_frontier();
    expect(stream_frontier.status == CUDA_SUCCESS && context_frontier.status == CUDA_SUCCESS &&
               stream_frontier.frontier.observe() == CUDA_ERROR_NOT_READY &&
               context_frontier.frontier.observe() == CUDA_ERROR_NOT_READY &&
               owner.session.capture_command_frontier(0).status == CUDA_ERROR_INVALID_HANDLE,
           "real session stream/context frontiers include the non-command barrier");
    deferred(owner.session.release(memory.id));
    deferred(owner.session.destroy_stream(stream.id));
    deferred(owner.session.unload_module(module.id));
    owner.session.close();
    expect(owner.session.accept_external_barrier(stream.id, std::make_shared<int>(0)).status ==
               CUDA_ERROR_DEINITIALIZED,
           "close rejects new boundaries but does not abandon accepted source owners");
    expect(owner.session.begin_external_barrier(external.barrier) == CUDA_SUCCESS &&
               owner.session.begin_external_barrier(external.barrier) == CUDA_ERROR_INVALID_VALUE,
           "already accepted work can begin exactly once after close");
    expect(check_scheduling(owner.session).external_work_in_flight == 1,
           "lightweight observation preserves started external physical reservation");
    const auto queries = driver.event_queries;
    const auto syncs = driver.syncs;
    expect(owner.session.poll().retained == 1 && owner.session.drain_issued().retained == 1 &&
               driver.event_queries == queries && driver.syncs == syncs &&
               owner.session.cancel_external_barrier(external.barrier) == CUDA_ERROR_NOT_READY,
           "no poll, drain, close or cancel turns an unrecorded external tail into stop proof");
    deferred(owner.session.shutdown());
    expect(owner.session.record_external_enqueue(external.barrier) == CUDA_SUCCESS &&
               owner.session.record_external_enqueue(external.barrier) == CUDA_ERROR_INVALID_VALUE,
           "trusted exact-stream enqueue records the actual tail once");
    expect(owner.session.poll().retained == 1 && driver.event_queries == queries + 1,
           "recorded tail remains pinned until independent native completion");
    deferred(owner.session.release(memory.id));
    driver.can_stop = true;
    expect(owner.session.poll().retired == 1 && owner.session.snapshot().external_barriers == 0 &&
               external.frontier.observe() == CUDA_SUCCESS &&
               stream_frontier.frontier.observe() == CUDA_SUCCESS &&
               context_frontier.frontier.observe() == CUDA_SUCCESS,
           "actual event completion retires the boundary and every captured prefix");
    counts(owner.session, {0, 0});
    expect(owner.session.cancel_external_barrier(external.barrier) == CUDA_ERROR_INVALID_HANDLE,
           "retired barrier token cannot mutate a later source lifetime");
    success(owner.session.release(memory.id));
    success(owner.session.destroy_stream(stream.id));
    success(owner.session.unload_module(module.id));
    success(owner.session.shutdown());
    expect(driver.frees == 1 && driver.stream_destroys == 1 && driver.unloads == 1 &&
               driver.context_destroys == 1 && driver.launches == 0,
           "all exact source owners clean once without a fabricated application issue");
}
void external_barrier_session_exclusion() {
    driver = {};
    Owner owner;
    const auto stream = owner.stream();
    const std::array cohort_streams{stream.id, std::uint64_t{0}};
    auto lifetime = std::make_shared<CudaSessionExternalCommand>(command(owner, stream.id, {}));
    success(owner.session.deactivate());
    const auto detached_calls = driver.calls;
    const auto canceled = owner.session.accept_external_barrier(stream.id, lifetime);
    expect(canceled.status == CUDA_SUCCESS &&
               owner.session.external_barrier_ready(canceled.barrier) == CUDA_SUCCESS &&
               owner.session.cancel_external_barrier(canceled.barrier) == CUDA_SUCCESS &&
               canceled.frontier.observe() == CUDA_SUCCESS && driver.calls == detached_calls,
           "CPU acceptance/readiness/cancel need owner identity but not a current native context");
    success(owner.session.activate());
    const auto external = owner.session.accept_external_barrier(stream.id, lifetime);
    expect(external.status == CUDA_SUCCESS, "second source boundary accepted");
    const auto wrong_calls = driver.calls;
    std::thread wrong_owner([&] {
        expect(owner.session.accept_external_cohort(cohort_streams, lifetime).status ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   owner.session.accept_external_barrier(stream.id, lifetime).status ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   owner.session.external_barrier_ready(external.barrier) ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   owner.session.begin_external_barrier(external.barrier) ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   owner.session.record_external_enqueue(external.barrier) ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   owner.session.cancel_external_barrier(external.barrier) ==
                       CUDA_ERROR_INVALID_CONTEXT &&
                   owner.session.fail_external_barrier(external.barrier, CUDA_ERROR_UNKNOWN) ==
                       CUDA_ERROR_INVALID_CONTEXT,
               "every facade transition rejects a foreign thread before native callbacks");
    });
    wrong_owner.join();
    expect(driver.calls == wrong_calls, "wrong owner causes no SDK query");
    unsigned int callbacks{};
    driver.proof_callback = [&] {
        ++callbacks;
        const auto calls = driver.calls;
        expect(
            owner.session.accept_external_cohort(cohort_streams, lifetime).status ==
                    CUDA_ERROR_NOT_READY &&
                owner.session.accept_external_barrier(stream.id, lifetime).status ==
                    CUDA_ERROR_NOT_READY &&
                owner.session.external_barrier_ready(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.session.begin_external_barrier(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.session.record_external_enqueue(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.session.cancel_external_barrier(external.barrier) == CUDA_ERROR_NOT_READY &&
                owner.session.fail_external_barrier(external.barrier, CUDA_ERROR_UNKNOWN) ==
                    CUDA_ERROR_NOT_READY,
            "native scope excludes every recursively requested transition");
        deferred(owner.session.destroy_stream(stream.id));
        deferred(owner.session.shutdown());
        expect(driver.calls == calls, "reentry cannot enter a second SDK proof or cleanup");
    };
    expect(owner.session.begin_external_barrier(external.barrier) == CUDA_SUCCESS,
           "outer native preparation remains intact after rejected recursive transitions");
    driver.proof_callback = {};
    expect(callbacks == 2, "begin brackets event creation with two guarded context proofs");
    expect(owner.session.record_external_enqueue(external.barrier) == CUDA_SUCCESS,
           "healthy external tail can record after exclusion ends");
    driver.can_stop = true;
    expect(owner.session.poll().retired == 1, "second source boundary physically retires");
    lifetime.reset();
    counts(owner.session, {0, 0});
    success(owner.session.destroy_stream(stream.id));
    owner.close();
}
void external_cohort_retains_actual_session_sources() {
    driver = {};
    Owner owner;
    const auto first = owner.allocation(), second = owner.allocation();
    const auto stream = owner.stream();
    const std::array<CudaSessionExternalRangeRequest, 1> first_range{{{{first.id, 0}, 8}}};
    const std::array<CudaSessionExternalRangeRequest, 1> second_range{{{{second.id, 2}, 12}}};
    auto sources = std::make_shared<std::vector<CudaSessionExternalCommand>>();
    sources->push_back(command(owner, stream.id, first_range));
    sources->push_back(command(owner, 0, second_range));
    const std::array bad_streams{stream.id, UINT64_MAX};
    const auto calls = driver.calls;
    expect(owner.session.accept_external_cohort(bad_streams, sources).status ==
                   CUDA_ERROR_INVALID_HANDLE &&
               owner.session.snapshot().external_barriers == 0 && driver.calls == calls,
           "session cohort validates every source before publishing or calling SDK");
    success(owner.session.deactivate());
    const std::array streams{stream.id, std::uint64_t{0}, stream.id};
    const auto detached_calls = driver.calls;
    auto accepted = owner.session.accept_external_cohort(streams, sources);
    expect(accepted.status == CUDA_SUCCESS && accepted.members.size() == 2 &&
               accepted.members[0].stream == stream.id && accepted.members[1].stream == 0 &&
               driver.calls == detached_calls,
           "CPU-only session cohort merges exact duplicate stream identities while detached");
    sources.reset();
    counts(owner.session, {2, 2});
    success(owner.session.activate());
    const auto context_tail = owner.session.capture_frontier();
    expect(context_tail.status == CUDA_SUCCESS, "context frontier captures the whole cohort");
    for (const auto& member : accepted.members) {
        expect(owner.session.external_barrier_ready(member.barrier) == CUDA_SUCCESS,
               "legacy and blocking source siblings share one union readiness cut");
        expect(owner.session.begin_external_barrier(member.barrier) == CUDA_SUCCESS,
               "both coherent source boundaries begin before either enqueue is recorded");
    }
    owner.session.close();
    expect(owner.session.accept_external_cohort(streams, std::make_shared<int>(0)).status ==
               CUDA_ERROR_DEINITIALIZED,
           "closing rejects new cohorts but preserves accepted native ownership");
    deferred(owner.session.release(first.id));
    deferred(owner.session.release(second.id));
    deferred(owner.session.destroy_stream(stream.id));
    expect(owner.session.record_external_enqueue(accepted.members[0].barrier) == CUDA_SUCCESS,
           "first actual source stream tail records independently");
    driver.can_stop = true;
    const auto queries = driver.event_queries;
    const auto partial = owner.session.poll();
    expect(partial.retired == 1 && partial.retained == 1 && driver.event_queries == queries + 1 &&
               accepted.members[0].frontier.observe() == CUDA_SUCCESS &&
               accepted.members[1].frontier.observe() == CUDA_ERROR_NOT_READY &&
               context_tail.frontier.observe() == CUDA_ERROR_NOT_READY,
           "one physical tail cannot retire its unrecorded coherent sibling");
    counts(owner.session, {2, 2});
    deferred(owner.session.release(first.id));
    deferred(owner.session.release(second.id));
    expect(owner.session.record_external_enqueue(accepted.members[1].barrier) == CUDA_SUCCESS &&
               owner.session.poll().retired == 1 &&
               accepted.members[1].frontier.observe() == CUDA_SUCCESS &&
               context_tail.frontier.observe() == CUDA_SUCCESS,
           "all exact native tails must stop before common source owners retire");
    counts(owner.session, {0, 0});
    success(owner.session.release(first.id));
    success(owner.session.release(second.id));
    success(owner.session.destroy_stream(stream.id));
    success(owner.session.shutdown());
    expect(driver.frees == 2 && driver.stream_destroys == 1 && driver.context_destroys == 1 &&
               driver.launches == 0,
           "owned cohort cleanup is exact without a synthetic application kernel or receipt");
}

void proof_and_reentry() {
    driver = {};
    CudaExecutionSession cold(ExecutionSessionIdentity{11, 13}, 23, dispatch());
    expect(!check_scheduling(cold).has_context, "cold scheduling observation creates no context");
    absent(cold.capture_external_context(), CUDA_ERROR_INVALID_CONTEXT);
    absent(cold.capture_external_command(0, {}), CUDA_ERROR_INVALID_CONTEXT);
    expect(driver.calls == 0, "uninitialized capture cannot initialize or create a context");
    success(cold.shutdown());
    Owner owner;
    const auto memory = owner.allocation();
    const auto stream = owner.stream();
    const auto module = owner.module();
    const std::array<CudaSessionExternalRangeRequest, 1> ranges{{{{memory.id, 0}, 4}}};
    for (bool is_command : {false, true}) {
        unsigned int callbacks{};
        driver.proof_callback = [&] {
            ++callbacks;
            const auto calls = driver.calls;
            check_scheduling(owner.session);
            absent(owner.session.capture_external_context(), CUDA_ERROR_NOT_READY);
            absent(owner.session.capture_external_command(stream.id, ranges), CUDA_ERROR_NOT_READY);
            deferred(owner.session.release(memory.id));
            deferred(owner.session.destroy_stream(stream.id));
            deferred(owner.session.unload_module(module.id));
            deferred(owner.session.shutdown());
            expect(driver.calls == calls,
                   "capture scope excludes recursive owner proof and all target cleanup");
        };
        if (is_command) {
            auto value = command(owner, stream.id, ranges);
            expect(value.ranges().size() == 1,
                   "outer command capture survives rejected recursive cleanup");
        } else {
            auto value = context(owner);
            expect(value.native() == owner.native,
                   "outer context capture survives rejected recursive cleanup");
        }
        driver.proof_callback = {};
        expect(callbacks == 1, "capture performs exactly one guarded current-context proof");
        counts(owner.session, {0, 0});
    }
    driver.proof_status = CUDA_ERROR_OUT_OF_MEMORY;
    absent(owner.session.capture_external_context(), CUDA_ERROR_OUT_OF_MEMORY);
    absent(owner.session.capture_external_command(stream.id, ranges), CUDA_ERROR_OUT_OF_MEMORY);
    counts(owner.session, {0, 0});
    driver.proof_status = CUDA_SUCCESS;
    success(owner.session.deactivate());
    absent(owner.session.capture_external_context(), CUDA_ERROR_INVALID_CONTEXT);
    absent(owner.session.capture_external_command(stream.id, ranges), CUDA_ERROR_INVALID_CONTEXT);
    success(owner.session.activate());
    auto retained = context(owner);
    owner.session.close();
    absent(owner.session.capture_external_context(), CUDA_ERROR_DEINITIALIZED);
    absent(owner.session.capture_external_command(stream.id, ranges), CUDA_ERROR_DEINITIALIZED);
    deferred(owner.session.shutdown());
    expect(driver.context_destroys == 0, "closing does not weaken a live context hold");
    retained = {};
    success(owner.session.shutdown());
    expect(driver.context_destroys == 1, "known context closes after final retention release");
    expect(!check_scheduling(owner.session).has_context,
           "closed scheduling observation does not resurrect the context");
}
void cleanup_cannot_publish_new_capsules() {
    for (const auto operation : {Cleanup::kAllocation, Cleanup::kModule, Cleanup::kStream}) {
        driver = {};
        Owner owner;
        const auto memory = owner.allocation();
        const auto stream = owner.stream();
        const auto module = owner.module();
        const auto global = owner.session.get_global(module.id, "variable");
        success(global);
        const std::array<CudaSessionExternalRangeRequest, 1> ranges{
            {{{operation == Cleanup::kModule ? global.id : memory.id, 0}, 4}}};
        unsigned int callbacks{};
        driver.cleanup_callback = [&](Cleanup observed) {
            expect(observed == operation,
                   "exact requested cleanup callback reaches the regression boundary");
            ++callbacks;
            const auto calls = driver.calls;
            absent(owner.session.capture_external_context(), CUDA_ERROR_NOT_READY);
            absent(owner.session.capture_external_command(stream.id, ranges), CUDA_ERROR_NOT_READY);
            counts(owner.session, {0, 0});
            expect(driver.calls == calls,
                   "cleanup exclusion rejects capture before another context callback");
        };
        CudaSessionResult result;
        if (operation == Cleanup::kAllocation)
            result = owner.session.release(memory.id);
        else if (operation == Cleanup::kModule)
            result = owner.session.unload_module(module.id);
        else
            result = owner.session.destroy_stream(stream.id);
        success(result);
        expect(result.original_status == CUDA_SUCCESS && callbacks == 1,
               "native cleanup completes exactly once without publishing a capsule for dead "
               "resources");
        driver.cleanup_callback = {};
        counts(owner.session, {0, 0});
        if (operation != Cleanup::kAllocation) success(owner.session.release(memory.id));
        if (operation != Cleanup::kModule) success(owner.session.unload_module(module.id));
        if (operation != Cleanup::kStream) success(owner.session.destroy_stream(stream.id));
        expect(driver.frees == 1 && driver.unloads == 1 && driver.stream_destroys == 1,
               "all independent resources have exactly one original cleanup");
        owner.close();
    }
}
void retained_uncertainty() {
    const auto pid = ::fork();
    expect(pid >= 0, "fork isolated intentional uncertainty");
    if (pid == 0) {
        try {
            driver = {};
            // Deliberately uncertain native ownership is retained until process
            // exit. No successful cleanup or leak-freedom is claimed for it.
            auto* owner = new Owner;
            const auto held_memory = owner->allocation();
            const auto failed_memory = owner->allocation();
            const std::array<CudaSessionExternalRangeRequest, 1> ranges{
                {{{held_memory.id, 0}, 32}}};
            auto retained = command(*owner, 0, ranges);
            driver.release_status = CUDA_ERROR_UNKNOWN;
            const auto failed = owner->session.release(failed_memory.id);
            expect(failed.original_status == CUDA_ERROR_UNKNOWN &&
                       owner->session.snapshot().is_quarantined,
                   "actual unrelated native failure establishes uncertainty independently");
            absent(owner->session.capture_external_context(), CUDA_ERROR_UNKNOWN);
            absent(owner->session.capture_external_command(0, {}), CUDA_ERROR_UNKNOWN);
            deferred(owner->session.shutdown());
            counts(owner->session, {1, 1});
            expect(
                driver.context_destroys == 0 && driver.frees == 1 && driver.memory.size() == 2 &&
                    retained.ranges()[0].address == held_memory.address,
                "keeping the capsule retains exact owners without inventing completion or retry");
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated retained-uncertainty assertions pass");
}
void faulted_session_can_fence_started_boundary() {
    const auto pid = ::fork();
    expect(pid >= 0, "fork isolated session-proof fault");
    if (pid == 0) {
        try {
            driver = {};
            auto owner = std::make_unique<Owner>();
            const auto memory = owner->allocation();
            const auto module = owner->module();
            const auto function = owner->session.get_function(module.id, "kernel");
            success(function);
            const std::array<CudaSessionExternalRangeRequest, 1> ranges{{{{memory.id, 0}, 32}}};
            const auto external = owner->session.accept_external_barrier(
                0, std::make_shared<CudaSessionExternalCommand>(command(*owner, 0, ranges)));
            const auto waiting =
                owner->session.accept_external_barrier(0, std::make_shared<int>(0));
            expect(external.status == CUDA_SUCCESS && waiting.status == CUDA_SUCCESS &&
                       owner->session.begin_external_barrier(external.barrier) == CUDA_SUCCESS,
                   "real accepted source boundary starts before unrelated metadata proof failure");
            // Resource cleanup is correctly forbidden while barriers exist.
            // Fail a permitted read-only metadata entry's actual native proof
            // instead; never bypass resource quiescence to manufacture a fault.
            driver.proof_status = CUDA_ERROR_NOT_INITIALIZED;
            const auto failed =
                owner->session.function_control(CudaFunctionControl::kGetAttribute, function.id, 0);
            expect(failed.status == CUDA_ERROR_NOT_INITIALIZED && !failed.original_status &&
                       owner->session.snapshot().is_quarantined,
                   "actual metadata owner-proof failure faults the session without vendor status");
            const auto state = check_scheduling(owner->session);
            expect(state.is_quarantined && state.external_work_in_flight == 1,
                   "faulted scheduling observation retains the actual started boundary");
            driver.proof_status = CUDA_SUCCESS;
            expect(
                owner->session.external_barrier_ready(waiting.barrier) == CUDA_ERROR_UNKNOWN &&
                    owner->session.begin_external_barrier(waiting.barrier) == CUDA_ERROR_UNKNOWN &&
                    owner->session.record_external_enqueue(external.barrier) == CUDA_SUCCESS &&
                    owner->session.cancel_external_barrier(waiting.barrier) == CUDA_SUCCESS,
                "fault forbids new native entry but permits known tail and no-work cleanup");
            driver.can_stop = true;
            expect(owner->session.poll().retired == 1 &&
                       external.frontier.observe() == CUDA_SUCCESS &&
                       owner->session.snapshot().external_barriers == 0 &&
                       owner->session.snapshot().is_quarantined,
                   "independent tail retirement never claims unrelated session fault recovery");
            counts(owner->session, {0, 0});
            expect(driver.context_destroys == 0 && driver.frees == 0 && driver.unloads == 0,
                   "uncertain session retains native context/resources until child process exit");
            std::_Exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            std::_Exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "isolated faulted-session tail assertions pass");
}
}  // namespace

int main() {
    try {
        context_only_and_default();
        exact_resources_and_context_copy();
        rejected_ranges_and_foreign_ids();
        bounded_registrations();
        native_address_bounds_and_preparation();
        pending_work_is_not_a_barrier();
        thread_and_process_boundaries();
        external_barrier_retains_actual_source();
        external_barrier_session_exclusion();
        external_cohort_retains_actual_session_sources();
        proof_and_reentry();
        cleanup_cannot_publish_new_capsules();
        retained_uncertainty();
        faulted_session_can_fence_started_boundary();
        std::cout << "External resource capsule contracts passed; no NCCL execution or GPU-stop "
                     "proof claimed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
