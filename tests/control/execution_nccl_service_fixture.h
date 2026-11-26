#pragma once

#include "runtime.h"
#include "glimmer/control/execution_nccl_protocol.h"
#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace glimmer::tests::nccl_service {
using namespace app;
using namespace control;
using namespace backend::cuda;
using Clock = std::chrono::steady_clock;
inline constexpr unsigned kUserEventFlags = CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING;
inline void expect(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "NCCL service: " << message << '\n';
    std::_Exit(1);
}
template <typename Predicate>
void await(Predicate predicate) {
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (!predicate()) {
        expect(Clock::now() < deadline, "bounded independently owned progress");
        std::this_thread::yield();
    }
}
struct Stream;
struct RegisteredHost {
    void* pointer{};
    std::size_t bytes{};
    CUcontext context{};
    std::thread::id worker;
    unsigned pending_copies{};
    bool live{}, was_mapped{};
};
struct Command {
    enum class Kind { kFill, kCopy, kWait, kBroadcast } kind{};
    CUdeviceptr source{}, destination{};
    std::size_t bytes{}, cutoff{};
    std::uint32_t value{};
    Stream* dependency{};
    RegisteredHost* registration{};
};
struct Stream {
    CUcontext context{};
    unsigned flags{};
    bool live{};
    std::vector<Command> commands;
    std::size_t completed{};
};
struct DeferredGroupCommand {
    Stream* stream{};
    Command command;
};
struct Event {
    bool live{}, recorded{};
    Stream* stream{};
    std::size_t cutoff{};
    unsigned flags{};
    std::uint64_t incarnation{};
    CUcontext context{};
};
struct DriverIssue {
    enum class Kind { kFill, kRecordEvent, kWaitEvent } kind{};
    Stream* stream{};
    std::uint64_t event{};
    unsigned event_flags{};
    std::size_t cutoff{};
    std::uint32_t value{};
};
struct Allocation {
    std::array<std::uint32_t, 4> before{};
    std::array<std::uint32_t, 16> values{};
    std::array<std::uint32_t, 4> after{};
    bool live{};
    CUcontext context{};
};
struct Comm {
    bool live{};
};
inline constexpr std::size_t kProgressCommands = 32, kProgressHostSlotBytes = 64;
inline constexpr std::uint64_t kProgressGuard = 0x3141592653589793;
template <std::size_t Bytes>
struct alignas(64) ProgressArena {
    std::array<std::uint64_t, 8> before{};
    std::array<std::byte, Bytes> values{};
    std::array<std::uint64_t, 8> after{};
    void prepare() {
        before.fill(kProgressGuard);
        after.fill(kProgressGuard);
    }
    void verify() const {
        expect(std::ranges::all_of(before, [](auto value) { return value == kProgressGuard; }) &&
                   std::ranges::all_of(after, [](auto value) { return value == kProgressGuard; }),
               "private progress storage boundary guards are unchanged");
    }
};
struct ProgressStorage {
    ProgressArena<kProgressCommands * sizeof(std::uint64_t)> counters;
    ProgressArena<kProgressCommands * kProgressHostSlotBytes> host;
    bool is_required{}, has_counter_allocation{}, is_counter_live{}, has_counter_release{},
        has_host_allocation{}, is_host_live{}, has_host_mapping{}, has_host_release{};
    void verify_cleanup() const {
        if (!is_required) {
            expect(!has_counter_allocation && !has_host_allocation && !has_host_mapping,
                   "nonpersistent source never owns private progress storage");
            return;
        }
        counters.verify();
        host.verify();
        expect(has_counter_allocation && has_counter_release && !is_counter_live &&
                   has_host_allocation && has_host_mapping && has_host_release && !is_host_live,
               "each exact context releases its separately allocated and mapped progress owners");
    }
};
struct Provider {
    std::mutex mutex;
    std::thread::id owner;
    std::array<int, 8> contexts{};
    std::array<bool, 8> live_contexts{};
    std::array<std::thread::id, 8> context_threads{};
    std::array<ProgressStorage, 8> progress{};
    std::array<Stream, 8> legacy{};
    std::array<Stream, 32> streams{};
    std::array<Allocation, 16> allocations{};
    std::array<Event, 256> events{};
    std::array<Comm, 128> communicators{};
    std::array<RegisteredHost, 8> registrations{};
    std::array<std::atomic<unsigned>, 11> calls{};
    std::vector<std::pair<unsigned, std::thread::id>> ledger;
    std::vector<ncclComm_t> async_queries;
    std::vector<DriverIssue> driver_issues;
    unsigned next_context{}, next_stream{}, next_allocation{}, next_communicator{};
    unsigned primary_retains{}, primary_releases{}, context_destroys{}, stream_destroys{}, waits{};
    unsigned implicit_allocations{}, implicit_streams{}, implicit_events{};
    unsigned next_registration{};
    bool has_host_registration{};
    std::atomic<unsigned> host_registers{}, host_unregisters{}, host_maps{}, host_probes{},
        host_uploads{}, host_downloads{}, release_context_proofs{};
    std::atomic<CUcontext> observed_release_context{};
    std::atomic<unsigned> hold_release_proof_at{};
    std::atomic<bool> release_proof_entered{}, may_return_release_proof{true};
    std::atomic<Stream*> host_copy_stream{};
    std::atomic<bool> hold_next_host_copy{};
    std::atomic<bool> hold_host_registration{}, host_registration_entered{},
        may_return_host_registration{true};
    std::size_t maximum_pending_requests{2};
    std::uint64_t next_event_incarnation{1};
    std::atomic<bool> events_ready{true}, stop_after_collective{}, hold_uid{}, uid_entered{},
        release_uid{true}, uid_returned{}, corrupt_uid_context{};
    std::atomic<int> collective_status{}, async_value{}, user_record_status{};
    std::atomic<unsigned> denied_steps{};
    std::atomic<unsigned> event_records{}, event_queries{}, event_retires{};
    std::atomic<unsigned> platform_uuid_queries{}, platform_driver_queries{};
    std::atomic<unsigned> driver_callbacks{};
    std::atomic<unsigned> user_record_errors{};
    std::atomic<Stream*> held_event_stream{};
    std::atomic<Stream*> user_record_failure_stream{};
    std::atomic<bool> should_defer_group_publication{};
    std::atomic<unsigned> eager_group_commands{}, deferred_group_commands{},
        published_group_commands{};
    CUcontext primary() {
        return reinterpret_cast<CUcontext>(&contexts.back());
    }
    bool is_helper_context(CUcontext context) const {
        for (std::size_t index = 0; index != next_context; ++index)
            if (reinterpret_cast<const void*>(&contexts[index]) == context)
                return live_contexts[index] && context_threads[index] != owner &&
                       context_threads[index] == std::this_thread::get_id();
        return false;
    }
    ProgressStorage& source_progress(CUcontext context) {
        const auto owners = std::span{contexts}.first(next_context);
        const auto found = std::ranges::find_if(owners, [context](auto& owner) {
            return reinterpret_cast<CUcontext>(&owner) == context;
        });
        expect(found != owners.end() &&
                   live_contexts[static_cast<std::size_t>(found - owners.begin())],
               "private progress resource belongs to one exact live source context");
        return progress[static_cast<std::size_t>(found - owners.begin())];
    }
    Stream& stream(CUstream handle, CUcontext context) {
        if (!handle) {
            const auto found = std::ranges::find_if(contexts, [&](const auto& value) {
                return reinterpret_cast<CUcontext>(const_cast<int*>(&value)) == context;
            });
            expect(found != contexts.end(), "legacy stream has an exact known current context");
            auto& result = legacy[static_cast<std::size_t>(found - contexts.begin())];
            result.context = context;
            result.live = true;
            return result;
        }
        const auto found = std::ranges::find_if(streams, [&](const auto& value) {
            return reinterpret_cast<CUstream>(const_cast<Stream*>(&value)) == handle && value.live;
        });
        expect(found != streams.end(), "only authenticated live source/proxy streams reach SDK");
        return *found;
    }
    void drain(Stream& stream, std::size_t cutoff, unsigned recursion = 0) {
        expect(cutoff <= stream.commands.size() && recursion < 64,
               "bounded acyclic actual stream dependencies");
        while (stream.completed < cutoff) {
            const auto command = stream.commands[stream.completed];
            if (command.kind == Command::Kind::kWait) {
                expect(command.dependency, "real event dependency is retained");
                drain(*command.dependency, command.cutoff, recursion + 1);
            } else if (command.kind == Command::Kind::kFill) {
                // Fake SDK device allocations are actual owned host storage.
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                auto* destination = reinterpret_cast<std::uint32_t*>(command.destination);
                std::fill_n(destination, command.bytes / sizeof(std::uint32_t), command.value);
            } else if (command.kind == Command::Kind::kCopy) {
                if (command.registration)
                    expect(command.registration->live && command.registration->pending_copies,
                           "queued host copy retains the exact live registered mapping");
                // Fake SDK addresses are never caller-native fallback pointers.
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                auto* destination = reinterpret_cast<void*>(command.destination);
                // NOLINTNEXTLINE(performance-no-int-to-ptr)
                const auto* source = reinterpret_cast<const void*>(command.source);
                std::memcpy(destination, source, command.bytes);
                if (command.registration) --command.registration->pending_copies;
            }
            ++stream.completed;
        }
    }
    void verify_guards() const {
        for (const auto& memory : allocations) {
            if (!memory.live) continue;
            expect(std::ranges::all_of(memory.before,
                                       [](auto value) { return value == 0x12345678; }) &&
                       std::ranges::all_of(memory.after,
                                           [](auto value) { return value == 0x87654321; }),
                   "every fake allocation boundary guard is unchanged");
        }
        for (const auto& registration : registrations) {
            if (!registration.live) continue;
            const auto bytes =
                std::span{static_cast<const std::byte*>(registration.pointer), registration.bytes};
            expect(std::ranges::all_of(bytes.subspan(64),
                                       [](auto value) { return value == std::byte{0x5a}; }),
                   "all unused registered memfd page bytes remain unchanged");
        }
    }
    void enqueue_host_copy(CUdeviceptr device, const void* host, std::size_t bytes, CUstream handle,
                           CUcontext context, bool is_upload) {
        expect(has_host_registration && bytes == 64,
               "only complete 64-byte registered transfers, never private progress copies");
        const auto registered = std::ranges::find_if(registrations, [host](const auto& value) {
            return value.live && value.was_mapped && value.pointer == host;
        });
        const auto allocation = std::ranges::find_if(allocations, [&](const auto& value) {
            return value.live && value.context == context &&
                   reinterpret_cast<CUdeviceptr>(value.values.data()) == device;
        });
        expect(registered != registrations.end() && allocation != allocations.end(),
               "host copy uses one exact registered mapping and current-context allocation");
        auto& selected = stream(handle, context);
        const auto host_address = reinterpret_cast<CUdeviceptr>(host);
        selected.commands.push_back({Command::Kind::kCopy, is_upload ? host_address : device,
                                     is_upload ? device : host_address, bytes, 0, 0, nullptr,
                                     &*registered});
        ++registered->pending_copies;
        // Publish the complete fake callback state before the client can use
        // either atomic count as cross-thread observation of this copy.
        host_copy_stream = &selected;
        if (hold_next_host_copy.exchange(false)) held_event_stream = &selected;
        if (is_upload)
            ++host_uploads;
        else
            ++host_downloads;
    }
};
inline Provider provider;
inline thread_local CUcontext current{};
inline thread_local unsigned group_depth{};
inline thread_local std::vector<DeferredGroupCommand> deferred_group_commands;
inline CUresult CUDAAPI set_current(CUcontext context) {
    ++provider.driver_callbacks;
    current = context;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI get_current(CUcontext* output) {
    ++provider.driver_callbacks;
    if (current && current == provider.observed_release_context.load() &&
        std::this_thread::get_id() != provider.owner) {
        const auto proof = ++provider.release_context_proofs;
        if (proof == provider.hold_release_proof_at.load()) {
            provider.release_proof_entered = true;
            await([] { return provider.may_return_release_proof.load(); });
        }
    }
    *output = current;
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI stream_create(CUstream* output, unsigned flags) {
    ++provider.driver_callbacks;
    std::lock_guard lock(provider.mutex);
    expect(provider.next_stream < provider.streams.size(), "bounded exact stream storage");
    auto& stream = provider.streams[provider.next_stream++];
    stream.context = current;
    stream.flags = flags;
    stream.live = true;
    *output = reinterpret_cast<CUstream>(&stream);
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI stream_query(CUstream handle) {
    ++provider.driver_callbacks;
    if (!provider.events_ready.load()) return CUDA_ERROR_NOT_READY;
    std::lock_guard lock(provider.mutex);
    auto& stream = provider.stream(handle, current);
    provider.drain(stream, stream.commands.size());
    return CUDA_SUCCESS;
}
inline CUresult CUDAAPI stream_destroy(CUstream handle) {
    ++provider.driver_callbacks;
    std::lock_guard lock(provider.mutex);
    auto& stream = provider.stream(handle, current);
    expect(stream.completed == stream.commands.size(),
           "stream cleanup follows actual command stop");
    stream.live = false;
    ++provider.stream_destroys;
    return CUDA_SUCCESS;
}
inline ncclResult_t visit(unsigned operation) {
    expect(std::this_thread::get_id() != provider.owner,
           "native NCCL runs on a real distinct stable lane");
    ++provider.calls[operation - 1];
    {
        std::lock_guard lock(provider.mutex);
        provider.ledger.push_back({operation, std::this_thread::get_id()});
    }
    if (operation == 1 && provider.hold_uid.load()) {
        provider.uid_entered = true;
        await([] { return provider.release_uid.load(); });
    }
    return ncclSuccess;
}
inline ncclResult_t uid(ncclUniqueId* output) {
    const auto status = visit(1);
    if (!output) return ncclInvalidArgument;
    std::memset(output, 0x6d, sizeof(*output));
    if (provider.corrupt_uid_context.load())
        current = reinterpret_cast<CUcontext>(&provider.contexts[6]);
    provider.uid_returned = true;
    return status;
}
inline ncclResult_t initialize_comm(ncclComm_t* output, unsigned operation) {
    const auto status = visit(operation);
    if (!output) return ncclInvalidArgument;
    std::lock_guard lock(provider.mutex);
    expect(provider.next_communicator < provider.communicators.size(),
           "bounded native communicator storage");
    auto& communicator = provider.communicators[provider.next_communicator++];
    communicator.live = true;
    *output = reinterpret_cast<ncclComm_t>(&communicator);
    current = provider.primary();
    return status;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed SDK ABIs.
inline ncclResult_t init(ncclComm_t* output, int ranks, ncclUniqueId, int rank) {
    expect(ranks == 1 && rank == 0, "exact original rank scalars");
    return initialize_comm(output, 2);
}
inline ncclResult_t init_config(ncclComm_t* output, int ranks, ncclUniqueId, int rank,
                                ncclConfig_t* config) {
    expect(ranks == 1 && rank == 0 && !config, "exact NULL configuration and rank scalars");
    return initialize_comm(output, 3);
}
inline ncclResult_t query(ncclComm_t handle, ncclResult_t* output) {
    const auto status = visit(4);
    if (!handle || !output) return ncclInvalidArgument;
    {
        std::lock_guard lock(provider.mutex);
        const auto found =
            std::ranges::find_if(provider.communicators, [handle](auto& communicator) {
                return handle == reinterpret_cast<ncclComm_t>(&communicator);
            });
        expect(found != provider.communicators.end() && found->live,
               "query uses an exact live native communicator");
        provider.async_queries.push_back(handle);
    }
    const auto value = provider.async_value.load();
    std::memcpy(output, &value, sizeof(value));
    return status;
}
inline ncclResult_t destroy(ncclComm_t handle) {
    const auto status = visit(5);
    if (!handle) return ncclInvalidArgument;
    std::lock_guard lock(provider.mutex);
    auto& communicator = *reinterpret_cast<Comm*>(handle);
    expect(communicator.live, "destroy exact live native communicator once");
    communicator.live = false;
    return status;
}
inline ncclResult_t abort_comm(ncclComm_t handle) {
    const auto status = visit(6);
    if (!handle) return ncclInvalidArgument;
    std::lock_guard lock(provider.mutex);
    auto& communicator = *reinterpret_cast<Comm*>(handle);
    expect(communicator.live, "abort exact live native communicator once");
    communicator.live = false;
    return status;
}
inline ncclResult_t start() {
    ++group_depth;
    return visit(7);
}
inline ncclResult_t end() {
    expect(group_depth != 0, "exact native lane group TLS");
    --group_depth;
    const auto status = visit(8);
    if (!group_depth) {
        std::lock_guard lock(provider.mutex);
        for (const auto& pending : deferred_group_commands) {
            expect(pending.stream && pending.stream->live,
                   "deferred outer group retains each exact native source stream");
            pending.stream->commands.push_back(pending.command);
            ++provider.published_group_commands;
        }
        deferred_group_commands.clear();
    }
    return status;
}
inline ncclResult_t collective(const void* source, void* destination, std::size_t count,
                               ncclDataType_t datatype, cudaStream_t handle, unsigned operation) {
    const auto status = visit(operation);
    expect(handle && datatype == ncclFloat32 && count == 8,
           "exact authenticated source stream, extent and datatype");
    {
        std::lock_guard lock(provider.mutex);
        auto& stream = provider.stream(reinterpret_cast<CUstream>(handle), current);
        const Command command{operation == 11 ? Command::Kind::kBroadcast : Command::Kind::kCopy,
                              reinterpret_cast<CUdeviceptr>(source),
                              reinterpret_cast<CUdeviceptr>(destination), count * sizeof(float)};
        if (group_depth && provider.should_defer_group_publication.load()) {
            deferred_group_commands.push_back({&stream, command});
            ++provider.deferred_group_commands;
        } else {
            stream.commands.push_back(command);
            if (group_depth) ++provider.eager_group_commands;
        }
    }
    if (provider.stop_after_collective.load()) provider.events_ready = false;
    const auto requested = provider.collective_status.load();
    return requested ? static_cast<ncclResult_t>(requested) : status;
}
inline ncclResult_t gather(const void* source, void* destination, std::size_t count,
                           ncclDataType_t datatype, ncclComm_t, cudaStream_t stream) {
    return collective(source, destination, count, datatype, stream, 9);
}
inline ncclResult_t reduce(const void* source, void* destination, std::size_t count,
                           ncclDataType_t datatype, ncclRedOp_t reduction, ncclComm_t,
                           cudaStream_t stream) {
    expect(reduction == ncclSum, "exact native reduction selector");
    return collective(source, destination, count, datatype, stream, 10);
}
inline ncclResult_t bcast(void* buffer, std::size_t count, ncclDataType_t datatype, int root,
                          ncclComm_t, cudaStream_t stream) {
    expect(root == 0, "exact broadcast root");
    return collective(buffer, buffer, count, datatype, stream, 11);
}
inline CudaNcclNativeDispatch nccl_dispatch() {
    return {uid, init, init_config, query, destroy, abort_comm, start, end, gather, reduce, bcast};
}
inline CudaSessionDispatch session_dispatch() {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned) {
        ++provider.driver_callbacks;
        return CUDA_SUCCESS;
    };
    sdk.device_get = [](CUdevice* output, int ordinal) {
        ++provider.driver_callbacks;
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.device_uuid = [](CUuuid* output, CUdevice device) {
        ++provider.driver_callbacks;
        expect(device == 0, "fake platform UUID belongs to the exact selected device");
        *output = {};
        output->bytes[0] = 0x47;
        output->bytes[15] = 1;
        ++provider.platform_uuid_queries;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* output) {
        ++provider.driver_callbacks;
        *output = 12060;
        ++provider.platform_driver_queries;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* output, unsigned flags, CUdevice) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        expect(provider.next_context < provider.contexts.size() - 1, "bounded source contexts");
        provider.live_contexts[provider.next_context] = true;
        provider.context_threads[provider.next_context] = std::this_thread::get_id();
        provider.progress[provider.next_context].is_required =
            std::this_thread::get_id() == provider.owner && (flags & CU_CTX_MAP_HOST) != 0;
        *output = current =
            reinterpret_cast<CUcontext>(&provider.contexts[provider.next_context++]);
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = set_current;
    sdk.context_destroy = [](CUcontext context) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        const auto owners = std::span{provider.contexts}.first(provider.next_context);
        const auto found = std::ranges::find_if(owners, [context](auto& owner) {
            return reinterpret_cast<CUcontext>(&owner) == context;
        });
        expect(current == context && context && found != owners.end(),
               "source context cleanup uses an exact created non-primary owner");
        const auto index = static_cast<std::size_t>(found - owners.begin());
        expect(provider.live_contexts[index], "destroy an exact source context only once");
        provider.verify_guards();
        provider.progress[index].verify_cleanup();
        expect(std::ranges::none_of(
                   provider.registrations,
                   [context](const auto& value) { return value.live && value.context == context; }),
               "helper context destruction never implicitly unregisters a live mapping");
        const auto retire_stream = [](Stream& stream) {
            expect(stream.completed == stream.commands.size(),
                   "implicit stream cleanup follows actual completed commands, never drains them");
            stream.live = false;
        };
        // cuCtxDestroy releases only resources associated with this context.
        // Preserve every tensor and other context's owners for later oracles.
        for (auto& stream : provider.streams)
            if (stream.live && stream.context == context) {
                retire_stream(stream);
                ++provider.implicit_streams;
            }
        for (auto& stream : provider.legacy)
            if (stream.live && stream.context == context) retire_stream(stream);
        for (auto& event : provider.events)
            if (event.live && event.context == context) {
                expect(!event.recorded || (event.stream && event.stream->context == context &&
                                           event.stream->completed >= event.cutoff),
                       "implicit event cleanup cannot invent its captured prefix completion");
                if (event.recorded) ++provider.event_retires;
                event.live = false;
                ++provider.implicit_events;
            }
        for (auto& allocation : provider.allocations)
            if (allocation.live && allocation.context == context) {
                allocation.live = false;
                ++provider.implicit_allocations;
            }
        provider.live_contexts[index] = false;
        current = nullptr;
        ++provider.context_destroys;
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule*, const void*) {
        ++provider.driver_callbacks;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.module_unload = [](CUmodule) {
        ++provider.driver_callbacks;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) {
        ++provider.driver_callbacks;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        ++provider.driver_callbacks;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.allocate = [](CUdeviceptr* output, std::size_t bytes) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& progress = provider.source_progress(current);
        if (progress.is_required && !progress.has_counter_allocation) {
            expect(bytes == progress.counters.values.size(),
                   "private progress allocation covers exactly 32 eight-byte counters");
            progress.counters.prepare();
            progress.has_counter_allocation = progress.is_counter_live = true;
            *output = reinterpret_cast<CUdeviceptr>(progress.counters.values.data());
            return CUDA_SUCCESS;
        }
        expect(bytes == 64 && provider.next_allocation < provider.allocations.size(),
               "bounded complete source allocation");
        auto& memory = provider.allocations[provider.next_allocation++];
        memory.before.fill(0x12345678);
        memory.after.fill(0x87654321);
        memory.live = true;
        memory.context = current;
        *output = reinterpret_cast<CUdeviceptr>(memory.values.data());
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr address) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& progress = provider.source_progress(current);
        if (address == reinterpret_cast<CUdeviceptr>(progress.counters.values.data())) {
            expect(progress.has_counter_allocation && progress.is_counter_live &&
                       !progress.has_counter_release && progress.has_host_release &&
                       !progress.is_host_live,
                   "free exact private counter owner once, after its host mapping release");
            progress.counters.verify();
            progress.has_counter_release = true;
            progress.is_counter_live = false;
            return CUDA_SUCCESS;
        }
        const auto found = std::ranges::find_if(provider.allocations, [=](const auto& memory) {
            return memory.live && reinterpret_cast<CUdeviceptr>(memory.values.data()) == address;
        });
        expect(found != provider.allocations.end(), "release exact live allocation");
        provider.verify_guards();
        found->live = false;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr destination, const void* source, std::size_t bytes) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& legacy = provider.stream(nullptr, current);
        provider.drain(legacy, legacy.commands.size());
        // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Exact fake Driver allocation.
        std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download = [](void* destination, CUdeviceptr source, std::size_t bytes) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& legacy = provider.stream(nullptr, current);
        provider.drain(legacy, legacy.commands.size());
        // NOLINTNEXTLINE(performance-no-int-to-ptr) -- Exact fake Driver allocation.
        std::memcpy(destination, reinterpret_cast<const void*>(source), bytes);
        return CUDA_SUCCESS;
    };
    sdk.host_allocate = [](void** output, std::size_t bytes, unsigned flags) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& progress = provider.source_progress(current);
        expect(progress.is_required && progress.is_counter_live && !progress.has_host_allocation &&
                   bytes == progress.host.values.size() && flags == CU_MEMHOSTALLOC_DEVICEMAP,
               "allocate one exact aligned private 32-slot host progress owner");
        progress.host.prepare();
        progress.has_host_allocation = progress.is_host_live = true;
        *output = progress.host.values.data();
        return CUDA_SUCCESS;
    };
    sdk.host_map = [](CUdeviceptr* output, void* host, unsigned flags) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& progress = provider.source_progress(current);
        if (provider.has_host_registration && std::this_thread::get_id() != provider.owner) {
            expect(output && host && !flags && provider.is_helper_context(current),
                   "registered map query has exact helper context, output and flags");
            const auto found = std::ranges::find_if(provider.registrations, [&](const auto& value) {
                return value.live && value.context == current && value.pointer == host;
            });
            if (found == provider.registrations.end()) {
                expect(!*output, "unregistered map probe starts with an untouched null output");
                ++provider.host_probes;
                return CUDA_ERROR_INVALID_VALUE;
            }
            expect(found->worker == std::this_thread::get_id() && !found->was_mapped,
                   "registered map publication stays on its original helper worker");
            found->was_mapped = true;
            *output = reinterpret_cast<CUdeviceptr>(host);
            ++provider.host_maps;
            return CUDA_SUCCESS;
        }
        expect(progress.is_host_live && !progress.has_host_mapping && !flags &&
                   host == progress.host.values.data(),
               "map the exact current-context host progress owner once, never caller storage");
        progress.host.verify();
        progress.has_host_mapping = true;
        *output = reinterpret_cast<CUdeviceptr>(host);
        return CUDA_SUCCESS;
    };
    sdk.host_release = [](void* host) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& progress = provider.source_progress(current);
        expect(progress.is_host_live && progress.has_host_mapping && !progress.has_host_release &&
                   host == progress.host.values.data(),
               "release the exact mapped private host progress owner once");
        progress.host.verify();
        progress.has_host_release = true;
        progress.is_host_live = false;
        return CUDA_SUCCESS;
    };
    if (provider.has_host_registration) {
        sdk.host_register = [](void* pointer, std::size_t bytes, unsigned flags) {
            ++provider.driver_callbacks;
            {
                std::lock_guard lock(provider.mutex);
                const auto page = ::sysconf(_SC_PAGESIZE);
                expect(page > 0 && bytes == static_cast<std::size_t>(page) && pointer &&
                           reinterpret_cast<std::uintptr_t>(pointer) % bytes == 0 &&
                           flags == (CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP) &&
                           provider.is_helper_context(current) &&
                           !provider.source_progress(current).is_required &&
                           provider.next_registration < provider.registrations.size(),
                       "register exact sealed-page helper storage, separate from private progress");
                expect(std::ranges::none_of(provider.registrations,
                                            [pointer](const auto& value) {
                                                return value.live && value.pointer == pointer;
                                            }),
                       "one native registration per live mapping");
                provider.registrations[provider.next_registration++] = {
                    pointer, bytes, current, std::this_thread::get_id(), 0, true, false};
                provider.verify_guards();
                ++provider.host_registers;
            }
            if (provider.hold_host_registration.load()) {
                provider.host_registration_entered = true;
                await([] { return provider.may_return_host_registration.load(); });
            }
            return CUDA_SUCCESS;
        };
        sdk.host_unregister = [](void* pointer) {
            ++provider.driver_callbacks;
            std::lock_guard lock(provider.mutex);
            const auto found = std::ranges::find_if(provider.registrations, [&](const auto& value) {
                return value.live && value.pointer == pointer && value.context == current;
            });
            expect(found != provider.registrations.end() && found->was_mapped &&
                       found->worker == std::this_thread::get_id() && !found->pending_copies,
                   "unregister once on the exact helper after every modeled copy actually stops");
            provider.verify_guards();
            found->live = false;
            ++provider.host_unregisters;
            return CUDA_SUCCESS;
        };
    }
    sdk.upload_async = [](CUdeviceptr destination, const void* source, std::size_t bytes,
                          CUstream stream) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        provider.enqueue_host_copy(destination, source, bytes, stream, current, true);
        return CUDA_SUCCESS;
    };
    sdk.download_async = [](void* destination, CUdeviceptr source, std::size_t bytes,
                            CUstream stream) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        provider.enqueue_host_copy(source, destination, bytes, stream, current, false);
        return CUDA_SUCCESS;
    };
    sdk.occupancy = [](int*, CUfunction, int, std::size_t) {
        ++provider.driver_callbacks;
        expect(false, "event-only configured source must not fabricate kernel occupancy");
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.function_set_attribute = [](CUfunction, CUfunction_attribute, int) {
        ++provider.driver_callbacks;
        expect(false, "event-only configured source must not configure a compute replacement");
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.stream_create = stream_create;
    sdk.stream_destroy = stream_destroy;
    sdk.execution.context_get_current = get_current;
    sdk.execution.launch_kernel = [](CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                     unsigned, unsigned, CUstream, void**, void**) {
        ++provider.driver_callbacks;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    sdk.execution.event_create = [](CUevent* output, unsigned flags) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        const auto found = std::ranges::find(provider.events, false, &Event::live);
        expect(found != provider.events.end(), "bounded private exact event storage");
        *found = {true, false, nullptr, 0, flags, provider.next_event_incarnation++, current};
        *output = reinterpret_cast<CUevent>(&*found);
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent handle, CUstream selected) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& event = *reinterpret_cast<Event*>(handle);
        expect(event.live && !event.recorded, "record exact private event once");
        if (event.flags == kUserEventFlags &&
            (!provider.user_record_failure_stream.load() ||
             provider.user_record_failure_stream.load() == &provider.stream(selected, current))) {
            const auto requested = provider.user_record_status.exchange(0);
            if (requested) {
                ++provider.user_record_errors;
                return static_cast<CUresult>(requested);
            }
        }
        event.stream = &provider.stream(selected, current);
        event.cutoff = event.stream->commands.size();
        event.recorded = true;
        provider.driver_issues.push_back({DriverIssue::Kind::kRecordEvent, event.stream,
                                          event.incarnation, event.flags, event.cutoff});
        ++provider.event_records;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent handle) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& event = *reinterpret_cast<Event*>(handle);
        expect(event.live && event.recorded, "never query unrecorded/retired event");
        ++provider.event_queries;
        if (!provider.events_ready.load() || provider.held_event_stream.load() == event.stream)
            return CUDA_ERROR_NOT_READY;
        provider.drain(*event.stream, event.cutoff);
        return CUDA_SUCCESS;
    };
    sdk.execution.event_destroy = [](CUevent handle) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& event = *reinterpret_cast<Event*>(handle);
        expect(event.live, "destroy exact private event once");
        if (event.recorded) ++provider.event_retires;
        event.live = false;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = stream_query;
    sdk.stream_wait_event = [](CUstream selected, CUevent handle, unsigned flags) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        const auto& event = *reinterpret_cast<Event*>(handle);
        expect(event.live && event.recorded && !flags, "wait on a real exact recorded prefix");
        auto& stream = provider.stream(selected, current);
        stream.commands.push_back({Command::Kind::kWait, 0, 0, 0, event.cutoff, 0, event.stream});
        provider.driver_issues.push_back(
            {DriverIssue::Kind::kWaitEvent, &stream, event.incarnation, event.flags, event.cutoff});
        ++provider.waits;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_wait_event = sdk.stream_wait_event;
    sdk.execution.fill32_async = [](CUdeviceptr destination, unsigned value, std::size_t count,
                                    CUstream handle) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        auto& stream = provider.stream(handle, current);
        stream.commands.push_back({Command::Kind::kFill, 0, destination, count * 4, 0, value});
        provider.driver_issues.push_back({DriverIssue::Kind::kFill, &stream, 0, 0, 0, value});
        return CUDA_SUCCESS;
    };
    sdk.execution.copy_device_async = [](CUdeviceptr destination, CUdeviceptr source,
                                         std::size_t bytes, CUstream handle) {
        ++provider.driver_callbacks;
        std::lock_guard lock(provider.mutex);
        provider.stream(handle, current)
            .commands.push_back({Command::Kind::kCopy, source, destination, bytes});
        return CUDA_SUCCESS;
    };
    return sdk;
}
// NOLINTEND(bugprone-easily-swappable-parameters)
inline ExecutionNcclDispatch execution_dispatch() {
    CudaNcclPrimaryDomainDispatch primary;
    primary.device_get = [](CUdevice* output, int ordinal) {
        ++provider.driver_callbacks;
        *output = ordinal;
        return CUDA_SUCCESS;
    };
    primary.primary_retain = [](CUcontext* output, CUdevice) {
        ++provider.driver_callbacks;
        ++provider.primary_retains;
        *output = provider.primary();
        return CUDA_SUCCESS;
    };
    primary.primary_release = [](CUdevice) {
        ++provider.driver_callbacks;
        ++provider.primary_releases;
        return CUDA_SUCCESS;
    };
    primary.context_get_current = get_current;
    primary.context_set_current = set_current;
    primary.stream_create = stream_create;
    primary.stream_query = stream_query;
    primary.stream_destroy = stream_destroy;
    return {nccl_dispatch(), primary};
}
inline void describe_failure(const std::optional<ExecutionWireReply>& reply, const char* operation,
                             const ExecutionSocketClient& client, std::uint64_t sequence = 0) {
    std::cerr << "NCCL service operation failure: operation=" << operation
              << " session=" << client.session() << " sequence=" << sequence;
    if (reply) {
        std::cerr << " disposition=" << static_cast<unsigned>(reply->disposition)
                  << " status=" << reply->status << " original=";
        if (reply->original_status)
            std::cerr << *reply->original_status;
        else
            std::cerr << "unavailable";
        std::cerr << " id=" << reply->id << " quarantine=" << reply->parameter_bytes;
    } else {
        std::cerr << " reply=unavailable";
    }
    std::cerr << " native_calls=";
    for (const auto& count : provider.calls) std::cerr << count.load() << ',';
    std::cerr << " events_ready=" << provider.events_ready.load() << '\n';
}
inline ExecutionWireReply success(std::optional<ExecutionWireReply> reply, const char* operation,
                                  const ExecutionSocketClient& client, std::uint64_t sequence = 0) {
    const bool complete =
        reply && reply->disposition == ExecutionWireDisposition::kComplete && !reply->status;
    if (!complete) describe_failure(reply, operation, client, sequence);
    expect(complete, "actual service operation succeeds on authenticated transport");
    return std::move(*reply);
}
inline ExecutionSocketClientConfig client_config(const std::string& path, TaskPeerIdentity peer) {
    return {path, peer, 10000, kExecutionMaxPayloadBytes, provider.maximum_pending_requests};
}
}  // namespace glimmer::tests::nccl_service
