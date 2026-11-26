#include "runtime.h"
#include "glimmer/control/execution_pointer_protocol.h"
#include "glimmer/control/execution_process_memory_protocol.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <span>
#include <string>
#include <vector>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using Clock = std::chrono::steady_clock;
using Operation = ExecutionProcessMemoryOperation;
constexpr std::size_t kWords = 24, kGuardWords = 4, kValues = 16;
constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);
constexpr std::uint32_t kGuard = 0xa1b2c3d4, kValue = 0x1728394a;
constexpr std::uint64_t kSeed = 0x123456789abcdef0;

void expect(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "process-memory service: " << message << '\n';
    // A failed native ownership assertion must not unwind through noexcept SDK
    // callbacks or attempt cleanup using the very ownership under examination.
    std::_Exit(1);
}
struct Event {
    bool recorded{};
};
struct Native {
    CUcontext current{};
    std::array<char, 3> pool_handles{};
    CUmemoryPool selected{};
    unsigned int selected_queries{};
    std::set<CUcontext> contexts;
    std::set<CUstream> streams;
    std::set<CUevent> events;
    std::map<CUdeviceptr, std::vector<std::byte>> memory;
    std::atomic<unsigned int> creates{}, destroys{}, allocation_calls{}, live_allocations{},
        frees{}, fills{}, pointer_calls{}, selections{}, trims{};
    std::atomic<bool> ready{true}, observe_second_pool{};
} native;
CUmemoryPool pool(unsigned int index) {
    return reinterpret_cast<CUmemoryPool>(&native.pool_handles.at(index));
}
CUresult CUDAAPI initialize(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device_get(CUdevice* output, int ordinal) {
    *output = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_get(CUcontext* output) {
    *output = native.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_create(CUcontext* output, unsigned int, CUdevice) {
    *output = reinterpret_cast<CUcontext>(new int(0));
    expect(native.contexts.insert(*output).second, "create distinct live native context");
    native.current = *output;
    ++native.creates;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_set(CUcontext value) {
    expect(!value || native.contexts.contains(value), "select only a live native context");
    native.current = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_destroy(CUcontext value) {
    expect(native.ready && native.contexts.erase(value) == 1,
           "destroy exact native context once, after physical stop");
    if (native.current == value) native.current = nullptr;
    delete reinterpret_cast<int*>(value);
    ++native.destroys;
    // Process allocations deliberately do not belong to this context.
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_create(CUevent* output, unsigned int) {
    *output = reinterpret_cast<CUevent>(new Event);
    expect(native.events.insert(*output).second, "create exact event identity");
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_record(CUevent value, CUstream stream) {
    expect(native.current && native.events.contains(value) &&
               (!stream || native.streams.contains(stream)),
           "record allocation or consumer tail on its actual native stream");
    reinterpret_cast<Event*>(value)->recorded = true;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_query(CUevent value) {
    expect(native.events.contains(value) && reinterpret_cast<Event*>(value)->recorded,
           "only a recorded live event supplies stop proof");
    return native.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent value) {
    expect(native.events.erase(value) == 1, "retire event exactly once");
    delete reinterpret_cast<Event*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI synchronize(CUstream) {
    return native.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI launch(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
    expect(false, "no kernel SDK path belongs to this process-memory fixture");
    return CUDA_ERROR_UNKNOWN;
}
CUresult CUDAAPI module_load(CUmodule*, const void*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI module_unload(CUmodule) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI function_get(CUfunction*, CUmodule, const char*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI parameter_info(CUfunction, std::size_t, std::size_t*, std::size_t*) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
auto allocation_containing(CUdeviceptr address, std::size_t bytes) {
    return std::ranges::find_if(native.memory, [&](const auto& entry) {
        return address >= entry.first && address - entry.first <= entry.second.size() &&
               bytes <= entry.second.size() - static_cast<std::size_t>(address - entry.first);
    });
}
CUresult CUDAAPI allocate(CUdeviceptr* output, std::size_t bytes) {
    expect(native.current, "actual allocation SDK executes in a live selected context");
    ++native.allocation_calls;
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    if (bytes == 13) {
        *output = 0;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (!bytes) {
        *output = 0;
        return CUDA_SUCCESS;
    }
    std::vector<std::byte> storage(bytes, std::byte{0x5a});
    *output = reinterpret_cast<CUdeviceptr>(storage.data());
    expect(native.memory.emplace(*output, std::move(storage)).second,
           "native allocation publishes a new independent byte image");
    ++native.live_allocations;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI release(CUdeviceptr address) {
    expect(native.current && native.ready && native.memory.erase(address) == 1,
           "synchronous free consumes exactly one live allocation after stop");
    --native.live_allocations;
    ++native.frees;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI upload(CUdeviceptr address, const void* input, std::size_t bytes) {
    const auto found = allocation_containing(address, bytes);
    expect(native.current && found != native.memory.end(),
           "upload stays inside actual owned native allocation");
    std::memcpy(found->second.data() + (address - found->first), input, bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI download(void* output, CUdeviceptr address, std::size_t bytes) {
    const auto found = allocation_containing(address, bytes);
    expect(native.current && native.ready && found != native.memory.end(),
           "download uses a live complete process allocation in the consumer context");
    std::memcpy(output, found->second.data() + (address - found->first), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_create(CUstream* output, unsigned int) {
    *output = reinterpret_cast<CUstream>(new int(0));
    expect(native.streams.insert(*output).second, "create one native stream");
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_destroy(CUstream value) {
    expect(native.ready && native.streams.erase(value) == 1, "destroy exact stopped stream once");
    delete reinterpret_cast<int*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI default_pool(CUmemoryPool* output, CUdevice device) {
    if (!output || device) return CUDA_ERROR_INVALID_VALUE;
    *output = pool(0);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI current_pool(CUmemoryPool* output, CUdevice device) {
    if (!output || device) return CUDA_ERROR_INVALID_VALUE;
    // Inject exactly one independently trusted SDK current-pool observation.
    // The first query proves materialization; only the requested getter may
    // supply this fixture observation. The default remains immutable and this
    // seam does not bypass the production materialization proof.
    if (native.selected == pool(0) && ++native.selected_queries == 2 &&
        native.observe_second_pool.exchange(false))
        native.selected = pool(1);
    *output = native.selected;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI select_pool(CUdevice device, CUmemoryPool value) {
    ++native.selections;
    if (device || !value) return CUDA_ERROR_INVALID_VALUE;
    expect(value == pool(0) || value == pool(1) || value == pool(2),
           "caller integers never reach SDK as native pool authority");
    native.selected = value;
    native.selected_queries = 0;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI allocate_async(CUdeviceptr* output, std::size_t bytes, CUstream stream) {
    expect(!stream || native.streams.contains(stream), "async allocator receives original stream");
    expect(native.selected == pool(0), "default allocation materializes this peer's logical pool");
    return allocate(output, bytes);
}
CUresult CUDAAPI allocate_from(CUdeviceptr* output, std::size_t bytes, CUmemoryPool value,
                               CUstream stream) {
    expect(!stream || native.streams.contains(stream), "explicit allocator receives owned stream");
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    expect(value == pool(1), "explicit allocation resolves the second trusted capability");
    return allocate(output, bytes);
}
cudaError_t CUDARTAPI trim(cudaMemPool_t value, std::size_t bytes) {
    ++native.trims;
    if (!value) return cudaErrorInvalidValue;
    expect(reinterpret_cast<CUmemoryPool>(value) == pool(1),
           "Runtime trim resolves the second owned pool");
    return bytes == 10719 ? static_cast<cudaError_t>(10719) : cudaSuccess;
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): original CUDA ABI order.
CUresult CUDAAPI fill(CUdeviceptr address, unsigned int value, std::size_t count, CUstream stream) {
    const auto found = allocation_containing(address, count * 4);
    expect(native.current && (!stream || native.streams.contains(stream)) &&
               found != native.memory.end(),
           "same-stream fill host enqueue requires ownership, not physical allocation completion");
    for (std::size_t index = 0; index != count; ++index)
        std::memcpy(found->second.data() + (address - found->first) + index * 4, &value, 4);
    ++native.fills;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI pointer_attribute(void* output, CUpointer_attribute selector,
                                   CUdeviceptr address) {
    ++native.pointer_calls;
    expect(output && allocation_containing(address, 1) != native.memory.end(),
           "pointer SDK sees only an authenticated live process allocation");
    if (selector == CU_POINTER_ATTRIBUTE_CONTEXT) {
        *static_cast<CUcontext*>(output) = nullptr;
    } else if (selector == CU_POINTER_ATTRIBUTE_DEVICE_POINTER) {
        *static_cast<CUdeviceptr*>(output) = address;
    } else {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return CUDA_SUCCESS;
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{
        {context_get, launch, event_create, event_record, event_query, event_destroy, synchronize},
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
    result.execution.fill32_async = fill;
    result.pointer_get_attribute = pointer_attribute;
    result.process_memory = {initialize,     context_get,   default_pool, current_pool, select_pool,
                             allocate_async, allocate_from, trim,         release};
    return result;
}

std::vector<std::byte> by_id(std::uint64_t id) {
    ExecutionPayloadWriter output;
    expect(output.write_u64(id), "encode exact scoped identity");
    return output.take();
}
std::vector<std::byte> role(std::uint32_t value, std::uint32_t flags = 0) {
    ExecutionPayloadWriter output;
    expect(output.write_u32(value) && output.write_u32(flags), "encode metadata bootstrap");
    return output.take();
}
ExecutionWireReply complete(std::optional<ExecutionWireReply> reply, bool is_native = true) {
    expect(reply && !reply->status && reply->disposition == ExecutionWireDisposition::kComplete &&
               (is_native ? reply->original_status == CUDA_SUCCESS : !reply->original_status),
           "complete receipt preserves actual SDK versus CPU metadata provenance");
    return std::move(*reply);
}
void rejected(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->status && !reply->original_status && !reply->id && !reply->address &&
               !reply->parameter_bytes && reply->arguments.empty() &&
               reply->disposition == ExecutionWireDisposition::kRejected && reply->data.empty(),
           "unauthorized operation rejects before SDK entry and publishes no capability");
}
void open_metadata(ExecutionSocketClient& client, std::uint32_t kind) {
    const auto reply = complete(client.open(role(kind)), false);
    expect(decode_execution_bootstrap_ownership(reply.data) ==
               ExecutionBootstrapOwnership::kMetadataRetained,
           "actual service retains CPU metadata ownership without fabricating a CUDA context");
}
ExecutionWireReply process(ExecutionSocketClient& client, ExecutionProcessMemoryRequest request) {
    ExecutionPayloadWriter output;
    expect(write_execution_process_memory_request(output, request), "encode canonical PMM1 call");
    const auto reply = client.exchange(ExecutionOpcode::kProcessMemory, output.take());
    expect(reply.has_value(), "receive bounded PMM1 receipt through actual service");
    return *reply;
}
void observation(const ExecutionWireReply& reply, std::int32_t status, bool written) {
    ExecutionPayloadReader input(reply.data);
    ExecutionProcessMemoryObservation effect;
    expect(reply.disposition == ExecutionWireDisposition::kComplete &&
               reply.original_status == status && !reply.parameter_bytes &&
               reply.arguments.empty() &&
               read_execution_process_memory_observation(input, effect) && input.empty() &&
               effect.output_written == written,
           "native result and independently observed output effects survive service projection");
}
ExecutionWireReply query_pool(ExecutionSocketClient& client, Operation operation) {
    auto reply = process(client, {operation, 0, true, 0, 0, 0, kSeed});
    observation(reply, CUDA_SUCCESS, true);
    expect(!reply.status && reply.id && reply.address, "successful getter owns a real capability");
    return reply;
}
ExecutionWireReply pointer(ExecutionSocketClient& client, std::uint64_t carrier,
                           std::uint64_t allocation, std::uint64_t query = 0) {
    ExecutionPayloadWriter output;
    expect(write_execution_pointer_query(
               output, {ExecutionPointerDomain::kDriver, 1, ExecutionPointerKind::kProcessDevice,
                        query, carrier, allocation, 4}),
           "encode process-owned pointer source independent of allocating context");
    auto reply = client.exchange(ExecutionOpcode::kPointerMetadata, output.take());
    expect(reply.has_value(), "receive pointer observation");
    return *reply;
}
void null_context(ExecutionSocketClient& client, std::uint64_t carrier, std::uint64_t allocation) {
    const auto reply = complete(pointer(client, carrier, allocation));
    ExecutionPayloadReader input(reply.data);
    ExecutionPointerObservation observed;
    expect(read_execution_pointer_observation(input, observed) && input.empty() &&
               observed.effects == 4 && observed.device_alias == ExecutionPointerAlias::kNull,
           "actual Context=NULL is an observed pointer effect, not missing context authority");
}
void fill_range(ExecutionSocketClient& client, std::uint64_t id, std::uint64_t stream,
                std::size_t offset, std::size_t words, std::uint32_t value) {
    ExecutionPayloadWriter input;
    expect(input.write_u32(static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32)) &&
               input.write_u32(1) && input.write_u64(id) && input.write_u64(offset) &&
               input.write_u64(0) && input.write_u64(0) && input.write_u64(words) &&
               input.write_u32(value) && input.write_u64(stream),
           "encode original full-range async fill");
    complete(client.exchange(ExecutionOpcode::kMemory, input.take()));
}
struct ImageExpectation {
    std::uint64_t id;
    std::uint32_t value;
};
void verify_image(ExecutionSocketClient& client, ImageExpectation image) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(image.id) && input.write_u64(0) && input.write_u64(kBytes),
           "request entire allocation including both guards");
    const auto reply = complete(client.exchange(ExecutionOpcode::kDownload, input.take()));
    expect(reply.data.size() == kBytes, "complete byte image is returned");
    for (std::size_t index = 0; index != kWords; ++index) {
        std::uint32_t actual{};
        std::memcpy(&actual, reply.data.data() + index * 4, 4);
        expect(actual ==
                   (index < kGuardWords || index >= kGuardWords + kValues ? kGuard : image.value),
               "every value and both guards match the independent mathematical image");
    }
}
template <typename Value>
void send_value(int socket, const Value& value) {
    expect(::send(socket, &value, sizeof(value), MSG_NOSIGNAL) == sizeof(value),
           "send exact independent peer fixture barrier");
}
template <typename Value>
Value receive_value(int socket) {
    pollfd input{socket, POLLIN, 0};
    expect(::poll(&input, 1, 10000) == 1, "bounded independent peer barrier");
    Value value{};
    expect(::recv(socket, &value, sizeof(value), 0) == sizeof(value), "receive exact peer barrier");
    return value;
}
struct Capabilities {
    std::uint64_t carrier{}, context{}, allocation{}, pool{};
};
void foreign_peer(int barrier, const ExecutionSocketClientConfig& config) {
    const auto stolen = receive_value<Capabilities>(barrier);
    ExecutionSocketClient carrier(config), metadata(config), execution(config);
    open_metadata(carrier, kExecutionProcessMemoryRole);
    open_metadata(metadata, kExecutionPointerMetadataRole);
    complete(execution.open());
    rejected(process(carrier, {Operation::kSetCurrentPool, 0, false, stolen.pool}));
    rejected(process(carrier, {Operation::kTrimPool, 0, false, stolen.pool, 1}));
    rejected(process(execution, {Operation::kAllocateFromPoolAsync, 0, true, stolen.pool, kBytes}));
    rejected(execution.exchange(ExecutionOpcode::kRelease, by_id(stolen.allocation)));
    rejected(pointer(metadata, stolen.carrier, stolen.allocation));
    rejected(pointer(metadata, stolen.carrier, stolen.allocation, stolen.context));
    // Actual SO_PEERCRED/pid-start identity creates another owner, not authority
    // from equal numeric capability values in the first peer's namespace.
    const auto orphan = process(execution, {Operation::kAllocateAsync, 0, true, 0, kBytes});
    observation(orphan, CUDA_SUCCESS, true);
    expect(execution_is_process_allocation(orphan.id), "second peer owns its own allocation");
    complete(execution.exchange(ExecutionOpcode::kClose));
    complete(metadata.exchange(ExecutionOpcode::kClose), false);
    complete(carrier.exchange(ExecutionOpcode::kClose), false);
    send_value(barrier, 'C');
    expect(receive_value<char>(barrier) == 'X', "peer remains alive after all socket Close calls");
    ::close(barrier);
    std::_Exit(0);
}

void public_path(const ExecutionSocketClientConfig& config, int barrier) {
    {
        ExecutionSocketClient invalid(config);
        const auto denied = invalid.open(role(kExecutionProcessMemoryRole, 1));
        expect(denied && denied->status && !denied->original_status &&
                   denied->disposition == ExecutionWireDisposition::kRejected &&
                   decode_execution_bootstrap_ownership(denied->data) ==
                       ExecutionBootstrapOwnership::kNone,
               "invalid PMM1 bootstrap flags retain no metadata or native ownership");
    }
    ExecutionSocketClient carrier(config), metadata(config), first(config);
    open_metadata(carrier, kExecutionProcessMemoryRole);
    open_metadata(metadata, kExecutionPointerMetadataRole);
    expect(native.creates == 0, "PMM1 and PMD1 bootstrap create no CUDA contexts");
    rejected(process(carrier, {Operation::kAllocateAsync, 0, true, 0, kBytes}));
    rejected(
        carrier.exchange(ExecutionOpcode::kRelease, by_id(kExecutionProcessAllocationTag | 1)));
    rejected(carrier.exchange(ExecutionOpcode::kDrain));

    const auto initial = query_pool(carrier, Operation::kGetDefaultPool);
    expect(query_pool(carrier, Operation::kGetCurrentPool).id == initial.id,
           "peer initializes logical current pool from trusted default");
    // A controlled fake-SDK current-pool observation tests two-capability
    // translation without exposing private owner publish APIs. This is not a
    // claim of native pool creation or changes to a physical default pool.
    native.observe_second_pool = true;
    const auto alternate = query_pool(carrier, Operation::kGetCurrentPool);
    expect(initial.id != alternate.id && initial.address != alternate.address,
           "two distinct SDK observations produce distinct pool capabilities");
    observation(process(carrier, {Operation::kSetCurrentPool, 0, false, alternate.id}),
                CUDA_SUCCESS, false);
    expect(query_pool(carrier, Operation::kGetCurrentPool).id == alternate.id,
           "setter really changes subsequent logical getter");
    observation(process(carrier, {Operation::kSetCurrentPool, 0, false, initial.id}), CUDA_SUCCESS,
                false);
    expect(query_pool(carrier, Operation::kGetCurrentPool).id == initial.id,
           "restoring first pool is not a constant or dropped setter");
    auto trimmed = process(carrier, {Operation::kTrimPool, 0, false, alternate.id, 10719});
    observation(trimmed, 10719, false);
    expect(!trimmed.status, "Runtime original error above 10000 is not mapped to Driver proof");
    observation(process(carrier, {Operation::kTrimPool, 0, false, alternate.id}), cudaSuccess,
                false);
    const auto null_getter = process(carrier, {Operation::kGetDefaultPool});
    observation(null_getter, CUDA_ERROR_INVALID_VALUE, false);
    expect(null_getter.status == CUDA_ERROR_INVALID_VALUE && !null_getter.id,
           "native null-output error creates no pool capability");
    rejected(process(carrier, {Operation::kGetDefaultPool, 1, true}));
    rejected(process(carrier, {Operation::kSetCurrentPool, 0, false, initial.address}));

    complete(first.open());
    rejected(process(first, {Operation::kGetDefaultPool, 0, true}));
    ExecutionPayloadWriter flags;
    expect(flags.write_u32(CU_STREAM_NON_BLOCKING), "encode original stream flags");
    const auto stream = complete(first.exchange(ExecutionOpcode::kCreateStream, flags.take())).id;
    const auto failed = process(first, {Operation::kAllocateAsync, 0, true, 0, 13, stream, kSeed});
    observation(failed, CUDA_ERROR_OUT_OF_MEMORY, true);
    expect(failed.status == CUDA_ERROR_OUT_OF_MEMORY && !failed.id && !failed.address,
           "native error-side NULL write survives without minting allocation authority");
    const auto zero = process(first, {Operation::kAllocateAsync, 0, true, 0, 0, stream, kSeed});
    observation(zero, CUDA_SUCCESS, true);
    expect(!zero.id && !zero.address, "zero-sized success preserves actual NULL output");
    rejected(process(first, {Operation::kAllocateAsync, 0, true, 0, kBytes, 99999}));

    native.ready = false;
    const auto a = process(first, {Operation::kAllocateAsync, 0, true, 0, kBytes, stream});
    const auto b =
        process(first, {Operation::kAllocateFromPoolAsync, 0, true, alternate.id, kBytes, stream});
    for (const auto* allocation : {&a, &b}) {
        observation(*allocation, CUDA_SUCCESS, true);
        expect(execution_is_process_allocation(allocation->id) && allocation->address,
               "actual successful native allocation gets a tagged process capability");
        fill_range(first, allocation->id, stream, 0, kWords, kGuard);
        fill_range(first, allocation->id, stream, kGuardWords * 4, kValues, kValue);
    }
    expect(native.fills == 4 && native.frees == 0 && !native.ready,
           "allocation and same-stream consumers enqueue before physical event readiness");
    native.ready = true;
    complete(first.exchange(ExecutionOpcode::kDrain), false);
    null_context(metadata, carrier.session(), a.id);
    const auto pointer_calls = native.pointer_calls.load();
    rejected(pointer(metadata, first.session(), a.id));
    expect(native.pointer_calls == pointer_calls, "execution context is not a process carrier");

    send_value(barrier, Capabilities{carrier.session(), first.session(), a.id, alternate.id});
    expect(receive_value<char>(barrier) == 'C', "foreign peer completed authenticated controls");
    expect(native.live_allocations == 3 && native.frees == 0,
           "another live peer's closed sockets do not authorize freeing its allocation");
    const auto old_carrier = carrier.session();
    complete(carrier.exchange(ExecutionOpcode::kClose), false);
    expect(native.live_allocations == 3 && native.frees == 0,
           "metadata carrier Close is not process death or process-memory cleanup");
    ExecutionSocketClient replacement(config);
    open_metadata(replacement, kExecutionProcessMemoryRole);
    expect(query_pool(replacement, Operation::kGetCurrentPool).id == initial.id,
           "new same-peer carrier attaches existing process pool owner");
    rejected(pointer(metadata, old_carrier, a.id));
    null_context(metadata, replacement.session(), a.id);
    complete(first.exchange(ExecutionOpcode::kDestroyStream, by_id(stream)));
    complete(first.exchange(ExecutionOpcode::kClose));
    expect(native.live_allocations == 3 && native.frees == 0,
           "source A Close destroys context but preserves every process allocation");
    null_context(metadata, replacement.session(), a.id);

    ExecutionSocketClient second(config);
    complete(second.open());
    for (const auto* allocation : {&a, &b}) {
        verify_image(second, {allocation->id, kValue});
        fill_range(second, allocation->id, 0, kGuardWords * 4, kValues, kValue + 1);
        verify_image(second, {allocation->id, kValue + 1});
        const auto freed =
            complete(second.exchange(ExecutionOpcode::kRelease, by_id(allocation->id)));
        expect(!freed.id && !freed.address && !freed.parameter_bytes && freed.data.empty(),
               "tagged synchronous free keeps the ordinary release reply contract");
        rejected(second.exchange(ExecutionOpcode::kRelease, by_id(allocation->id)));
        rejected(pointer(metadata, replacement.session(), allocation->id));
    }
    expect(native.frees == 2 && native.live_allocations == 1,
           "consumer B frees both allocations exactly once, never another peer's orphan");
    complete(second.exchange(ExecutionOpcode::kClose));
    complete(metadata.exchange(ExecutionOpcode::kClose), false);
    complete(replacement.exchange(ExecutionOpcode::kClose), false);
    send_value(barrier, 'X');
}
}  // namespace

int main() {
    std::array<char, 64> directory{};
    std::strcpy(directory.data(), "/tmp/glimmer-process-memory-service-XXXXXX");
    expect(::mkdtemp(directory.data()), "create uniquely owned fixture directory");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "resolve trusted server PID/start identity");
    const std::string path = std::string(directory.data()) + "/service.sock";
    ExecutionSocketClientConfig client_config{path, *identity, 5000};
    int barriers[2]{};
    expect(::socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, barriers) == 0,
           "create bounded peer coordination without service authority");
    const auto peer = ::fork();
    expect(peer >= 0, "create independent authenticated process before service threads exist");
    if (!peer) {
        ::close(barriers[0]);
        foreign_peer(barriers[1], client_config);
    }
    ::close(barriers[1]);
    native.selected = pool(2);  // Another native actor's selection must survive every call.
    {
        ExecutionServiceConfig config;
        config.transport.socket_path = path;
        config.transport.max_peers = 16;
        config.transport.max_pending_requests = 16;
        config.uses_original_kernels = true;
        config.exit_when_idle = true;
        config.priority_idle_us = 0;
        ExecutionService service(config, dispatch());
        expect(service.start(), "start actual service runtime and authenticated transport");
        auto client =
            std::async(std::launch::async, [&] { public_path(client_config, barriers[0]); });
        const auto deadline = Clock::now() + std::chrono::seconds{20};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
            expect(service.step(1) && Clock::now() < deadline,
                   "bounded real owner-loop progress for complete public process-memory chain");
        client.get();
        while (!service.finished())
            expect(service.step(1) && Clock::now() < deadline,
                   "actual peer exit permits idle cleanup only after every physical tail");
        expect(service.request_stop(), "normal service shutdown proves all native cleanup");
        expect(native.selected == pool(2), "all pool operations restore outside actor's selection");
    }
    int status{};
    expect(::waitpid(peer, &status, 0) == peer && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "independent peer completed every permission and socket-lifetime assertion");
    ::close(barriers[0]);
    expect(native.frees == 3 && native.live_allocations == 0 && native.memory.empty() &&
               native.creates == native.destroys && native.contexts.empty() &&
               native.events.empty() && native.streams.empty(),
           "all allocations, contexts, streams and recorded tails retire exactly once");
    expect(::rmdir(directory.data()) == 0,
           "remove only the empty uniquely owned fixture directory");
    std::cout << "process-memory actual service/runtime/session CPU fixture passed\n";
}
