#include "runtime.h"
#include "cleanup_idle.h"
#include "profile_archive.h"
#include "internal/process_identity.h"

#include <array>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/ostream_sink.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void completed_cleanup_eligibility() {
    CudaSessionResult release;
    release.disposition = CudaSessionDisposition::kComplete;
    release.status = CUDA_SUCCESS;
    release.original_status = CUDA_SUCCESS;
    expect(completed_idle_cleanup(1, release), "successful native release qualifies");
    expect(!completed_idle_cleanup(kExecutionProcessAllocationTag | 1, release),
           "process allocations do not qualify");
    release.original_status.reset();
    expect(!completed_idle_cleanup(1, release), "missing native release does not qualify");
    release.original_status = CUDA_ERROR_INVALID_VALUE;
    expect(!completed_idle_cleanup(1, release), "failed native release does not qualify");
    release.original_status = CUDA_SUCCESS;
    release.status = CUDA_ERROR_INVALID_VALUE;
    expect(!completed_idle_cleanup(1, release), "failed release status does not qualify");
    release.status = CUDA_SUCCESS;
    for (const auto disposition :
         {CudaSessionDisposition::kRejected, CudaSessionDisposition::kDeferred}) {
        release.disposition = disposition;
        expect(!completed_idle_cleanup(1, release), "incomplete release does not qualify");
    }
    CudaSessionBlasResult destroy;
    destroy.disposition = CudaSessionDisposition::kComplete;
    destroy.status = CUDA_SUCCESS;
    destroy.original_status = CUBLAS_STATUS_SUCCESS;
    expect(completed_idle_cleanup(destroy), "successful native BLAS destroy qualifies");
    destroy.original_status.reset();
    expect(!completed_idle_cleanup(destroy), "missing native BLAS destroy does not qualify");
    destroy.original_status = CUBLAS_STATUS_INTERNAL_ERROR;
    expect(!completed_idle_cleanup(destroy), "failed native BLAS destroy does not qualify");
    destroy.original_status = CUBLAS_STATUS_SUCCESS;
    destroy.status = CUDA_ERROR_INVALID_VALUE;
    expect(!completed_idle_cleanup(destroy), "failed BLAS destroy status does not qualify");
    destroy.status = CUDA_SUCCESS;
    for (const auto disposition :
         {CudaSessionDisposition::kRejected, CudaSessionDisposition::kDeferred}) {
        destroy.disposition = disposition;
        expect(!completed_idle_cleanup(destroy), "incomplete BLAS destroy does not qualify");
    }
}
struct HostDelete {
    void operator()(std::byte* value) const noexcept {
        ::operator delete(value, std::align_val_t{64});
    }
};
struct Context {
    char original{}, transformed{};
    std::vector<std::byte> counters;
    std::unique_ptr<std::byte, HostDelete> host;
    bool is_destroyed{};
    std::array<char, 2> streams{};
    std::array<bool, 2> live_streams{};
    unsigned int created_streams{};
};
struct Event {
    Context* context{};
    unsigned int epoch{};
    bool is_destroyed{};
    unsigned int command{};
};
struct Native {
    Context* current{};
    Context* mode_low_context{};
    std::vector<std::unique_ptr<Context>> contexts;
    std::vector<std::unique_ptr<Event>> events;
    unsigned int low_launches{}, high_launches{}, initializations{}, host_frees{}, destroys{},
        synchronizes{}, event_queries{}, preparation_attempts{};
    bool allow_low_stop{}, allow_high_stop{}, saw_retreat{}, is_native_low{}, fails_first_low{};
    bool allows_naive_overlap{};
    bool is_sliced{};
    bool observes_completion{};
    bool uses_original_mode{true};
    bool fails_high_observation{};
    bool is_intake_race{}, is_intake_armed{}, has_intake_poll{}, saw_intake_retreat{};
    bool is_retired_epoch_case{}, has_retired_epoch{};
    bool selection_is_equal{}, selection_is_configured{};
    unsigned int low_commands{1}, stream_destroys{};
    unsigned int profile_identities{};
    unsigned int sharing_attribute_calls{};
    unsigned int preparation_cleanup_attempts{};
    CUresult preparation_cleanup_error{CUDA_SUCCESS};
    std::uint64_t profile_time{};
    std::uint64_t priority_time{1000};
    std::array<std::byte, 1234> application_storage{};
    char blas_handle{};
    unsigned int application_releases{}, blas_destroys{};
    std::array<unsigned int, 2> command_launches{};
    std::array<bool, 2> allow_low_stops{}, saw_retreats{};
} native;
std::byte* device_bytes(CUdeviceptr address, std::size_t bytes) {
    const auto base = reinterpret_cast<CUdeviceptr>(native.current->counters.data());
    expect(address >= base && address - base <= native.current->counters.size() &&
               bytes <= native.current->counters.size() - (address - base),
           "fake device access stays in the current owned allocation");
    return native.current->counters.data() + (address - base);
}
CUresult CUDAAPI initialize(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device_get(CUdevice* value, int ordinal) {
    *value = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI current(CUcontext* value) {
    *value = reinterpret_cast<CUcontext>(native.current);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_create(CUcontext* value, unsigned int, CUdevice) {
    native.contexts.push_back(std::make_unique<Context>());
    native.current = native.contexts.back().get();
    *value = reinterpret_cast<CUcontext>(native.current);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_set(CUcontext value) {
    native.current = reinterpret_cast<Context*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context_destroy(CUcontext value) {
    auto* context = reinterpret_cast<Context*>(value);
    expect(context == native.current && !context->is_destroyed, "exact current context cleanup");
    // Native context destruction reclaims still-owned streams without separate
    // cuStreamDestroy calls. Count each resource once through either path.
    for (auto& is_live : context->live_streams)
        if (is_live) {
            is_live = false;
            ++native.stream_destroys;
        }
    context->is_destroyed = true;
    native.current = nullptr;
    ++native.destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_load(CUmodule* value, const void*) {
    *value = reinterpret_cast<CUmodule>(native.current);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_unload(CUmodule value) {
    expect(value == reinterpret_cast<CUmodule>(native.current), "module context isolation");
    return CUDA_SUCCESS;
}
CUresult CUDAAPI function_get(CUfunction* value, CUmodule module, const char* name) {
    expect(module == reinterpret_cast<CUmodule>(native.current), "function context isolation");
    *value = reinterpret_cast<CUfunction>(std::string_view(name).find("__glimmer_ptb") ==
                                                  std::string_view::npos
                                              ? &native.current->original
                                              : &native.current->transformed);
    return CUDA_SUCCESS;
}
// Exact CUDA parameter metadata ABI.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CUresult CUDAAPI parameter_info(CUfunction function, std::size_t index, std::size_t* offset,
                                std::size_t* bytes) {
    constexpr std::array<std::size_t, 6> offsets{0, 4, 8, 16, 24, 32};
    expect(function == reinterpret_cast<CUfunction>(&native.current->transformed) &&
               index < offsets.size() - (native.observes_completion ? 0 : 1),
           "only valid private parameter indices queried");
    *offset = native.is_sliced ? index * 4 : offsets[index];
    *bytes = native.is_sliced || index < 3 ? 4 : 8;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI allocate(CUdeviceptr* value, std::size_t bytes) {
    native.current->counters.resize(bytes);
    *value = reinterpret_cast<CUdeviceptr>(native.current->counters.data());
    return CUDA_SUCCESS;
}
CUresult CUDAAPI release(CUdeviceptr value) {
    expect(value == reinterpret_cast<CUdeviceptr>(native.current->counters.data()),
           "release only owned private counters");
    native.current->counters.clear();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI upload(CUdeviceptr, const void*, std::size_t) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI download(void*, CUdeviceptr, std::size_t) {
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI stream_create(CUstream* value, unsigned int flags) {
    if (native.low_commands != 2) return CUDA_ERROR_NOT_SUPPORTED;
    expect(flags == CU_STREAM_NON_BLOCKING && native.current->host &&
               native.current->created_streams < 2,
           "create only prepared independent low streams");
    const auto index = native.current->created_streams++;
    native.current->live_streams[index] = true;
    *value = reinterpret_cast<CUstream>(&native.current->streams[index]);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_destroy(CUstream stream) {
    for (unsigned int index = 0; index < native.current->created_streams; ++index)
        if (stream == reinterpret_cast<CUstream>(&native.current->streams[index])) {
            expect(native.current->live_streams[index],
                   "destroy each owned independent stream once");
            native.current->live_streams[index] = false;
            ++native.stream_destroys;
            return CUDA_SUCCESS;
        }
    return CUDA_ERROR_INVALID_HANDLE;
}
CUresult CUDAAPI event_create(CUevent* value, unsigned int) {
    native.events.push_back(std::make_unique<Event>(Event{
        native.current, native.current->host ? native.low_launches : native.high_launches, false}));
    *value = reinterpret_cast<CUevent>(native.events.back().get());
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_record(CUevent value, CUstream stream) {
    auto* event = reinterpret_cast<Event*>(value);
    expect(event->context == native.current, "record in actual context");
    if (native.low_commands == 2 && native.current->host) {
        expect(stream == reinterpret_cast<CUstream>(&native.current->streams[0]) ||
                   stream == reinterpret_cast<CUstream>(&native.current->streams[1]),
               "low event belongs to its exact independent stream");
        event->command = stream == reinterpret_cast<CUstream>(&native.current->streams[0]) ? 0 : 1;
        event->epoch = native.command_launches[event->command] - 1;
    }
    return CUDA_SUCCESS;
}
CUresult CUDAAPI event_query(CUevent value) {
    auto& event = *reinterpret_cast<Event*>(value);
    expect(event.context == native.current && !event.is_destroyed, "query live owned event");
    if (event.context->host) {
        std::uint32_t retreat{};
        std::memcpy(&retreat, event.context->host.get() + std::size_t{event.command} * 64,
                    sizeof(retreat));
        native.saw_retreat = native.saw_retreat || retreat != 0;
        native.saw_retreats[event.command] = native.saw_retreats[event.command] || retreat != 0;
        if (native.is_retired_epoch_case && event.command == 0 && native.low_launches == 2) {
            native.has_retired_epoch = true;
            return CUDA_SUCCESS;
        }
        if (native.is_intake_race && event.epoch == 0 &&
            (native.is_intake_armed || native.has_intake_poll)) {
            if (native.is_intake_armed) {
                native.is_intake_armed = false;
                native.has_intake_poll = true;
                native.saw_intake_retreat = retreat != 0;
                if (!retreat) {
                    // The fake GPU finishes on this physical query unless the
                    // owner has already published the real retreat control.
                    const std::uint64_t complete = 8;
                    std::memcpy(event.context->counters.data(), &complete, sizeof(complete));
                    std::memcpy(event.context->host.get() + 16, &complete, sizeof(complete));
                }
            }
            return CUDA_SUCCESS;
        }
        const bool can_stop = native.low_commands == 1 ? native.allow_low_stop
                                                       : native.allow_low_stops[event.command];
        if (event.epoch == 0 &&
            !(can_stop && (retreat || native.is_native_low || native.is_sliced)))
            return CUDA_ERROR_NOT_READY;
        return CUDA_SUCCESS;
    }
    return native.allow_high_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI event_destroy(CUevent value) {
    auto& event = *reinterpret_cast<Event*>(value);
    expect(event.context == native.current && !event.is_destroyed, "destroy event once");
    event.is_destroyed = true;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI stream_sync(CUstream) {
    ++native.synchronizes;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI launch(CUfunction function, unsigned int grid_x, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, unsigned int, CUstream stream,
                        void**, void** extra) {
    const auto command = native.low_commands == 2 && native.current->host &&
                                 stream == reinterpret_cast<CUstream>(&native.current->streams[1])
                             ? 1U
                             : 0U;
    if (function == reinterpret_cast<CUfunction>(&native.current->transformed)) {
        if (native.is_sliced) {
            expect(grid_x == 3 && extra && *static_cast<std::size_t*>(extra[3]) == 20,
                   "slice launch uses bounded explicit workers and its own private ABI");
            std::array<std::uint32_t, 2> slice{};
            std::memcpy(slice.data(), static_cast<const std::byte*>(extra[1]) + 12, 8);
            const auto issued = native.command_launches[command];
            expect(issued < 3 && slice[0] == issued * 3 && slice[1] == (issued == 2 ? 2U : 3U) &&
                       (!issued || (native.high_launches == 2 && native.allow_high_stop)),
                   "low slice resumes at the next disjoint interval only after high physical stop");
            ++native.low_launches;
            ++native.command_launches[command];
            return CUDA_SUCCESS;
        }
        expect(grid_x == 320 && extra &&
                   *static_cast<std::size_t*>(extra[3]) == (native.observes_completion ? 40U : 32U),
               "server alone selects workers and private ABI");
        CUdeviceptr counter{};
        std::memcpy(&counter, static_cast<const std::byte*>(extra[1]) + 16, sizeof(counter));
        std::uint64_t before{};
        std::memcpy(&before, device_bytes(counter, sizeof(before)), sizeof(before));
        const auto issued = native.command_launches[command];
        if (native.fails_first_low && command == 0 && issued == 0) {
            expect(before == 0,
                   "failed original low launch retains its initialized private prefix");
            ++native.low_launches;
            ++native.command_launches[command];
            return CUDA_ERROR_INVALID_VALUE;
        }
        expect(
            (issued == 0 && before == 0) ||
                (issued == 1 && before == 3 && native.high_launches == 2 && native.allow_high_stop),
            "resume retains exact progress after actual high stop");
        const std::uint64_t after = issued == 0 ? 3 : 8;
        std::memcpy(device_bytes(counter, sizeof(after)), &after, sizeof(after));
        ++native.low_launches;
        ++native.command_launches[command];
    } else if (native.current->host && native.is_native_low) {
        expect(function == reinterpret_cast<CUfunction>(&native.current->original) &&
                   native.command_launches[command] == 0,
               "unsupported low work executes original exactly once");
        ++native.low_launches;
        ++native.command_launches[command];
    } else {
        expect(!native.is_intake_race || native.has_intake_poll,
               "early demand cannot issue high work before actual physical low observation");
        expect(function == reinterpret_cast<CUfunction>(&native.current->original) &&
                   !native.current->host && native.low_launches == native.low_commands &&
                   (native.low_commands == 1 ? native.allow_low_stop : native.allow_low_stops[1]),
               "high uses original ABI only after actual low event proof");
        ++native.high_launches;
    }
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_allocate(void** value, std::size_t bytes, unsigned int) {
    native.current->host.reset(
        static_cast<std::byte*>(::operator new(bytes, std::align_val_t{64})));
    *value = native.current->host.get();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_release(void* value) {
    expect(value == native.current->host.get(), "exact private host release");
    native.current->host.reset();
    ++native.host_frees;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_map(CUdeviceptr* value, void* host, unsigned int) {
    *value = reinterpret_cast<CUdeviceptr>(host);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI upload_async(CUdeviceptr value, const void* source, std::size_t bytes, CUstream) {
    ++native.initializations;
    std::memcpy(device_bytes(value, bytes), source, bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI download_async(void* value, CUdeviceptr source, std::size_t bytes, CUstream) {
    std::memcpy(value, device_bytes(source, bytes), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI occupancy(int* value, CUfunction, int, std::size_t) {
    *value = 4;
    return CUDA_SUCCESS;
}
const CudaSessionDispatch kDispatch{
    {current, launch, event_create, event_record, event_query, event_destroy, stream_sync},
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
    stream_destroy,
    host_allocate,
    host_release,
    host_map,
    upload_async,
    download_async,
    occupancy,
    nullptr,
    nullptr,
    nullptr,
    [](CUfunction, CUfunction_attribute attribute, int value) {
        expect(attribute == CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT &&
                   value == CU_SHAREDMEM_CARVEOUT_MAX_SHARED,
               "native helpers and private variants share a function preference");
        return CUDA_SUCCESS;
    }};

ExecutionWireReply success(std::optional<ExecutionWireReply> value) {
    expect(value && value->status == 0 && value->original_status == 0 &&
               value->disposition == ExecutionWireDisposition::kComplete,
           "actual SDK success, not early acceptance");
    return std::move(*value);
}
void send_byte(int socket, char value) {
    expect(::send(socket, &value, 1, MSG_NOSIGNAL) == 1, "send fixture barrier");
}
void wait_byte(int socket, char expected) {
    pollfd descriptor{socket, POLLIN, 0};
    expect(::poll(&descriptor, 1, 5000) == 1, "bounded fixture barrier wait");
    char value{};
    expect(::recv(socket, &value, 1, 0) == 1 && value == expected, "exact fixture barrier");
}
struct ClientMode {
    bool is_low{}, should_disconnect{}, is_native_low{};
    unsigned int low_commands{1};
    bool fails_first_low{};
    bool is_mode_control{};
    unsigned int high_commands{2};
    bool fails_high_observation{};
    bool fails_first_preparation{};
    unsigned int logical_blocks{8}, block_threads{32};
    bool holds_rejected_successor{};
    bool submits_batch{};
    bool sends_metadata_only{};
    unsigned int cleanup_case{};
};
void run_client(int barrier, const std::string& path, TaskPeerIdentity server, ClientMode mode) {
    wait_byte(barrier, 'C');
    ExecutionSocketClient client({path, server, 3000});
    success(client.open());
    if (mode.cleanup_case) {
        std::uint64_t allocation{}, handle{};
        if (mode.cleanup_case == 1 || mode.cleanup_case == 4 || mode.cleanup_case == 5) {
            ExecutionPayloadWriter size;
            expect(size.write_u64(1234), "cleanup test allocation size");
            allocation = success(client.exchange(ExecutionOpcode::kAllocate, size.take())).id;
        }
        if (mode.cleanup_case == 2 || mode.cleanup_case == 5)
            handle = success(client.exchange(ExecutionOpcode::kCreateBlas)).id;
        send_byte(barrier, 'R');
        wait_byte(barrier, 'G');
        ExecutionPayloadWriter id;
        expect(id.write_u64(mode.cleanup_case == 2   ? handle
                            : mode.cleanup_case == 3 ? 999
                                                     : allocation),
               "cleanup test resource identity");
        const auto first = client.exchange(
            mode.cleanup_case == 2 ? ExecutionOpcode::kDestroyBlas : ExecutionOpcode::kRelease,
            id.take());
        if (mode.cleanup_case == 3)
            expect(first && first->status != 0 && !first->original_status,
                   "unknown allocation is rejected without a native call");
        else
            (void)success(first);
        send_byte(barrier, '1');
        if (mode.cleanup_case == 5) {
            wait_byte(barrier, '2');
            ExecutionPayloadWriter blas;
            expect(blas.write_u64(handle), "second cleanup resource identity");
            (void)success(client.exchange(ExecutionOpcode::kDestroyBlas, blas.take()));
            send_byte(barrier, '3');
        }
        wait_byte(barrier, 'X');
        success(client.exchange(ExecutionOpcode::kClose));
        return;
    }
    ExecutionPayloadWriter source;
    const auto ptx = mode.is_low && mode.is_native_low
                         ? ".version 7.0\n.target sm_70\n.address_size 64\n"
                           ".visible .entry kernel() { .reg .b64 %clock; "
                           // A legal nonzero barrier keeps this deliberately
                           // native control independent of read-only timer support.
                           "bar.sync 15; mov.u64 %clock, %globaltimer; ret; }"
                         : ".version 7.0\n.target sm_70\n.address_size 64\n"
                           ".visible .entry kernel() { ret; }";
    const auto source_text =
        mode.is_mode_control && mode.is_low ? std::string("// mode-low\n") + ptx : std::string(ptx);
    expect(source.write_string(source_text), "original source");
    const auto module = success(client.exchange(ExecutionOpcode::kLoadPtx, source.take()));
    ExecutionPayloadWriter lookup;
    expect(lookup.write_u64(module.id) && lookup.write_string("kernel"), "original name");
    const auto function = success(client.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
    expect(function.arguments.empty() && function.parameter_bytes == 0, "original empty ABI");
    std::array<std::uint64_t, 2> streams{};
    if (mode.is_low && mode.low_commands == 2)
        for (auto& stream : streams) {
            ExecutionPayloadWriter flags;
            expect(flags.write_u32(CU_STREAM_NON_BLOCKING), "independent low stream flags");
            stream = success(client.exchange(ExecutionOpcode::kCreateStream, flags.take())).id;
        }
    send_byte(barrier, 'R');
    wait_byte(barrier, 'G');
    std::vector<std::uint64_t> batch;
    if (mode.sends_metadata_only) {
        const auto submitted = client.submit(ExecutionOpcode::kSnapshot);
        expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
               "flush exact high metadata request before physical query");
        send_byte(barrier, 'P');
        wait_byte(barrier, 'S');
        std::optional<ExecutionWireReply> reply;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!(reply = client.take_reply(submitted.sequence)))
            expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                   "bounded metadata-only host result");
        expect(reply && reply->status == CUDA_SUCCESS && !reply->original_status &&
                   reply->disposition == ExecutionWireDisposition::kComplete,
               "CPU snapshot success is not an original SDK return");
    }
    const auto commands = mode.sends_metadata_only ? 0U
                          : mode.is_low            ? mode.low_commands
                                                   : mode.high_commands;
    for (unsigned int index = 0; index < commands; ++index) {
        ExecutionPayloadWriter arguments;
        expect(arguments.write_u64(function.id) &&
                   arguments.write_u64(mode.is_low ? streams[index] : 0),
               "original handles");
        for (const auto dimension :
             std::array<std::uint32_t, 6>{mode.logical_blocks, 1, 1, mode.block_threads, 1, 1})
            expect(arguments.write_u32(dimension), "original geometry");
        expect(arguments.write_u32(0) && arguments.write_blob({}), "no private client arguments");
        if (mode.submits_batch) {
            const auto accepted = client.submit(ExecutionOpcode::kLaunch, arguments.take());
            expect(accepted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
                   "send both exact original commands before waiting for host replies");
            batch.push_back(accepted.sequence);
            continue;
        }
        if (mode.holds_rejected_successor && index == 1) {
            const auto submitted = client.submit(ExecutionOpcode::kLaunch, arguments.take());
            expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted && client.poll_for(0),
                   "submit successor after exact preparation rejection");
            send_byte(barrier, 'P');
            wait_byte(barrier, 'S');
            expect(client.poll_for(0) && !client.take_reply(submitted.sequence),
                   "uncertain cleanup cannot authorize a successor receipt");
            return;
        }
        const auto reply = client.exchange(ExecutionOpcode::kLaunch, arguments.take());
        if (mode.fails_first_preparation && index == 0)
            expect(reply && reply->status == CUDA_ERROR_OUT_OF_MEMORY && !reply->original_status &&
                       reply->id == 1 && reply->disposition == ExecutionWireDisposition::kRejected,
                   "preparation rejection is not an original SDK return");
        else if (mode.is_low && mode.fails_first_low && index == 0)
            expect(reply && reply->status == CUDA_ERROR_INVALID_VALUE &&
                       reply->original_status == CUDA_ERROR_INVALID_VALUE && reply->id == 1,
                   "one low command's actual SDK error cannot replace another command's reply");
        else
            expect(success(reply).id == index + 1, "one SDK result per original launch");
    }
    if (mode.submits_batch) {
        send_byte(barrier, 'P');
        wait_byte(barrier, 'S');
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        for (std::size_t index = 0; index < batch.size(); ++index) {
            std::optional<ExecutionWireReply> reply;
            while (!(reply = client.take_reply(batch[index])))
                expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                       "bounded exact asynchronous host result");
            expect(success(std::move(reply)).id == index + 1,
                   "one original host result for each batched logical command");
        }
    }
    if (mode.should_disconnect) return;  // Accepted physical work still belongs to the owner.
    ExecutionPayloadWriter scope;
    expect(scope.write_u32(0) && scope.write_u64(0), "logical context scope");
    const auto frontier = client.exchange(ExecutionOpcode::kCaptureFrontier, scope.take());
    if (mode.fails_high_observation) {
        expect(
            frontier && frontier->status == CUDA_ERROR_INVALID_VALUE && !frontier->original_status,
            "failed high event observation preserves its asynchronous error at the frontier");
        return;  // Disconnect only after receiving the genuine historical failure.
    }
    expect(frontier && frontier->status == 0 && !frontier->original_status, "CPU-only frontier");
    ExecutionPayloadWriter identifier;
    expect(identifier.write_u64(frontier->id), "exact frontier identity");
    const auto bytes = identifier.take();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (true) {
        const auto observed = client.exchange(ExecutionOpcode::kQueryFrontier, bytes);
        expect(observed && !observed->original_status, "frontier is not an original SDK call");
        if (observed->status == 0) break;
        expect(
            observed->status == CUDA_ERROR_NOT_READY && std::chrono::steady_clock::now() < deadline,
            "bounded actual logical wait");
    }
    const auto released = client.exchange(ExecutionOpcode::kReleaseFrontier, bytes);
    expect(released && released->status == 0, "release metadata only");
    success(client.exchange(ExecutionOpcode::kClose));
}
struct Child {
    pid_t pid{-1};
    int socket{-1};
    bool is_ready{};
    Child(const std::string& path, TaskPeerIdentity server, ClientMode mode) {
        std::array<int, 2> sockets{};
        expect(::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sockets.data()) == 0,
               "private fixture barrier pair");
        pid = ::fork();
        if (pid < 0) {
            (void)::close(sockets[0]);
            (void)::close(sockets[1]);
            throw std::runtime_error("fork independent authenticated client");
        }
        if (pid == 0) {
            (void)::close(sockets[0]);
            int result = EXIT_SUCCESS;
            try {
                run_client(sockets[1], path, server, mode);
            } catch (const std::exception& error) {
                std::cerr << "Priority child: " << error.what() << '\n';
                result = EXIT_FAILURE;
            }
            (void)::close(sockets[1]);
            _exit(result);
        }
        (void)::close(sockets[1]);
        socket = sockets[0];
    }
    ~Child() {
        if (socket >= 0) (void)::close(socket);
        if (pid > 0) {
            int status{};
            while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
            }
        }
    }
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    TaskPeerIdentity identity() const {
        std::uint64_t start{};
        expect(read_process_start_time(static_cast<std::uint32_t>(pid), &start),
               "actual child start");
        return {static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()), start};
    }
    void observe_ready() {
        if (is_ready) return;
        char value{};
        const auto count = ::recv(socket, &value, 1, MSG_DONTWAIT);
        if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return;
        expect(count == 1 && value == 'R', "actual child resources ready");
        is_ready = true;
    }
    bool join_success() {
        int status{};
        const auto result = ::waitpid(pid, &status, 0);
        expect(result == pid, "join owned child");
        pid = -1;
        expect(WIFEXITED(status), "owned child exits normally, never through a signal");
        return WEXITSTATUS(status) == 0;
    }
    void join() {
        expect(join_success(), "child original-ABI checks passed");
    }
};
void retired_epoch_retreat(const std::string& path) {
    native = {};
    native.is_native_low = true;
    native.low_commands = 2;
    native.is_retired_epoch_case = true;
    const auto server = current_process_identity();
    expect(server.has_value(), "real parent identity");
    ClientMode high_mode;
    high_mode.submits_batch = true;
    Child low(path, *server, {true, false, true, 2}), high(path, *server, high_mode);
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 8;
    config.transport.io_timeout_ms = 3000;
    config.session_limits.submissions.max_commands = 2;
    config.persistent.workers = 320;
    config.should_audit_native = true;
    config.client_priorities = {{low.identity(), 0}, {high.identity(), 100}};
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start actual retired-epoch priority service");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1), "prepare exact low/high resources");
        low.observe_ready();
        high.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded retired-epoch bootstrap");
    }
    send_byte(low.socket, 'G');
    while (native.low_launches != 2) {
        expect(service.step(1), "issue two independent original low commands");
        expect(std::chrono::steady_clock::now() < deadline, "bounded exact low issue");
    }
    while (!native.has_retired_epoch)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "ordinary physical observation retires the first original before high arrival");
    expect(native.has_retired_epoch, "post-issue poll physically retires the first original");
    send_byte(high.socket, 'G');
    wait_byte(high.socket, 'P');
    expect(service.step(0), "receive high demand before another physical observation");
    send_byte(high.socket, 'S');
    unsigned int turns{}, high_waits{};
    while (!service.finished()) {
        if (++turns == 5) native.allow_low_stops[1] = true;
        expect(service.step(1), "retire exact physical owners under either policy");
        if (native.high_launches == 2 && !native.allow_high_stop && ++high_waits == 5)
            native.allow_high_stop = true;
        expect(std::chrono::steady_clock::now() < deadline, "bounded retired-epoch cleanup");
    }
    const bool has_low_success = low.join_success();
    high.join();
    expect(service.request_stop() && native.low_launches == 2 && native.high_launches == 2 &&
               native.initializations == 0 && native.host_frees == 1 && native.destroys == 2 &&
               native.stream_destroys == 2 && native.synchronizes == 0,
           "all original commands issue once and exact resources retire before policy assertion");
    std::cout << "Retired epoch: low_success=" << has_low_success
              << "; low_launches=2; high_launches=2; normal_cleanup=true\n";
    expect(has_low_success,
           "known retired low epoch must not create an invalid-handle policy failure");
}
void priority_intake_race(const std::string& path, bool sends_metadata_only) {
    native = {};
    native.is_intake_race = true;
    const auto server = current_process_identity();
    expect(server.has_value(), "real parent identity");
    ClientMode high_mode;
    high_mode.submits_batch = !sends_metadata_only;
    high_mode.sends_metadata_only = sends_metadata_only;
    Child low(path, *server, {true}), high(path, *server, high_mode);
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.transport.max_pending_requests = 8;
    config.session_limits.submissions.max_commands = 2;
    config.persistent.workers = 320;
    config.client_priorities = {{low.identity(), 0}, {high.identity(), 100}};
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start authenticated intake-race service");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1), "prepare intake-race original resources");
        low.observe_ready();
        high.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded intake-race bootstrap");
    }
    send_byte(low.socket, 'G');
    while (!native.low_launches) {
        expect(service.step(1), "issue actual low before high request");
        expect(std::chrono::steady_clock::now() < deadline, "bounded initial low issue");
    }
    send_byte(high.socket, 'G');
    wait_byte(high.socket, 'P');  // Both complete wire frames precede the next owner turn.
    native.is_intake_armed = true;
    native.allow_low_stop = true;
    expect(service.step(0), "consume high demand at the final low physical-query window");
    send_byte(high.socket, 'S');
    unsigned int high_waits{};
    native.allow_high_stop = sends_metadata_only;
    while (!service.finished()) {
        expect(service.step(1), "drain exact high and retained low work");
        if (!sends_metadata_only && native.high_launches == 2 && !native.allow_high_stop) {
            expect(native.low_launches == 1, "high host return cannot authorize low resume");
            if (++high_waits == 20) native.allow_high_stop = true;
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded intake-race cleanup");
    }
    low.join();
    high.join();
    expect(service.request_stop(), "close intake-race transport after physical retirement");
    expect(native.has_intake_poll && native.initializations == 1 && native.host_frees == 1 &&
               native.destroys == 2 && native.synchronizes == 0 &&
               native.high_launches == (sends_metadata_only ? 0U : 2U) &&
               native.low_launches == (native.saw_intake_retreat ? 2U : 1U),
           "both policies fully retire exact work and resources before the ordering assertion");
    std::cout << "Intake race: metadata=" << sends_metadata_only
              << "; retreat_before_poll=" << native.saw_intake_retreat
              << "; low_launches=" << native.low_launches
              << "; high_launches=" << native.high_launches << "; normal_cleanup=true\n";
    expect(native.saw_intake_retreat != sends_metadata_only,
           "received GPU demand must request retreat before polling away its physical window; "
           "metadata alone must not request retreat");
}
struct AuditCapture {
    std::ostringstream messages;
    std::shared_ptr<spdlog::logger> logger{spdlog::default_logger()};
    std::shared_ptr<spdlog::sinks::ostream_sink_mt> sink{
        std::make_shared<spdlog::sinks::ostream_sink_mt>(messages)};
    AuditCapture() {
        sink->set_pattern("%v");
        logger->sinks().push_back(sink);
    }
    ~AuditCapture() {
        std::erase(logger->sinks(), sink);
    }
    AuditCapture(const AuditCapture&) = delete;
    AuditCapture& operator=(const AuditCapture&) = delete;
};
void cleanup_idle_service(const std::string& path, unsigned int scenario) {
    std::cout << "Cleanup idle service scenario " << scenario << '\n' << std::flush;
    native = {};
    const auto server = current_process_identity();
    expect(server.has_value(), "actual cleanup test owner identity");
    ClientMode high_mode;
    high_mode.cleanup_case = scenario;
    Child low(path, *server, {.is_low = true}), high(path, *server, high_mode);
    auto dispatch = kDispatch;
    dispatch.allocate = [](CUdeviceptr* address, std::size_t bytes) {
        if (bytes == native.application_storage.size()) {
            *address = reinterpret_cast<CUdeviceptr>(native.application_storage.data());
            return CUDA_SUCCESS;
        }
        return allocate(address, bytes);
    };
    dispatch.release = [](CUdeviceptr address) {
        if (address == reinterpret_cast<CUdeviceptr>(native.application_storage.data())) {
            ++native.application_releases;
            return CUDA_SUCCESS;
        }
        return release(address);
    };
    dispatch.blas.create = [](cublasHandle_t* handle) {
        *handle = reinterpret_cast<cublasHandle_t>(&native.blas_handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.destroy = [](cublasHandle_t handle) {
        expect(handle == reinterpret_cast<cublasHandle_t>(&native.blas_handle),
               "destroy the exact application BLAS handle");
        ++native.blas_destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    dispatch.blas.set_stream = [](cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; };
    dispatch.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                             const float*, const float*, int, const float*, int, const float*,
                             float*, int) { return CUBLAS_STATUS_NOT_SUPPORTED; };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact CUDA launch ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int gx, unsigned int gy,
                                          unsigned int gz, unsigned int bx, unsigned int by,
                                          unsigned int bz, unsigned int shared, CUstream stream,
                                          void** parameters, void** extra) {
        const auto status =
            launch(function, gx, gy, gz, bx, by, bz, shared, stream, parameters, extra);
        if (status == CUDA_SUCCESS && native.current->host) {
            const std::uint64_t complete = 8;
            std::memcpy(native.current->counters.data(), &complete, sizeof(complete));
        }
        return status;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    dispatch.execution.event_query = [](CUevent value) {
        const auto& event = *reinterpret_cast<Event*>(value);
        if (!event.context->host) return event_query(value);
        return CUDA_SUCCESS;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.persistent.workers = 320;
    config.client_priorities = {{low.identity(), 0}, {high.identity(), 100}};
    config.priority_idle_us = scenario == 4 ? 0 : 100;
    config.priority_clock_ns = []() noexcept { return native.priority_time; };
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start actual cleanup idle service");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "prepare actual cleanup test peers");
        low.observe_ready();
        high.observe_ready();
    }
    const auto wait_for_high = [&](char expected) {
        while (true) {
            expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
                   "bounded actual cleanup request");
            char value{};
            const auto count = ::recv(high.socket, &value, 1, MSG_DONTWAIT);
            if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) continue;
            expect(count == 1 && value == expected, "exact cleanup test reply barrier");
            return;
        }
    };
    send_byte(high.socket, 'G');
    wait_for_high('1');
    expect(native.application_releases == (scenario == 1 || scenario == 4 || scenario == 5) &&
               native.blas_destroys == (scenario == 2),
           "actual cleanup invokes exactly the intended native operation");
    for (int index = 0; index != 5; ++index) expect(service.step(1), "observe first drain");
    if (scenario == 5) {
        native.priority_time += 99999;
        send_byte(high.socket, '2');
        wait_for_high('3');
        expect(native.application_releases == 1 && native.blas_destroys == 1,
               "both successive native cleanups completed");
        for (int index = 0; index != 5; ++index) expect(service.step(1), "observe second drain");
    }
    send_byte(low.socket, 'G');
    if (scenario != 3 && scenario != 4) {
        for (int index = 0; index != 20; ++index) {
            expect(service.step(1) && native.low_launches == 0,
                   "successful cleanup keeps low work waiting at the fixed clock");
        }
        native.priority_time += scenario == 5 ? 1 : 99999;
        for (int index = 0; index != 5; ++index)
            expect(service.step(1) && native.low_launches == 0,
                   "the last nanosecond or old deadline cannot issue low work");
        native.priority_time += scenario == 5 ? 99999 : 1;
    }
    while (!native.low_launches)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "low work eventually issues without a stale cleanup hold");
    expect(native.low_launches == 1, "exactly one low command issues after the hold");
    while (!native.destroys)
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "low work completes before the high client disconnects");
    send_byte(high.socket, 'X');
    while (!service.finished()) {
        native.priority_time += 100000;
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "both cleanup test clients retire");
    }
    low.join();
    high.join();
    expect(native.destroys == 2 && service.request_stop(), "cleanup test retires both contexts");
}
void priority_handoff(const std::string& path, bool should_disconnect = false,
                      bool is_native_low = false, unsigned int low_commands = 1,
                      bool fails_first_low = false, bool is_sliced = false,
                      bool uses_saved_profile = false, bool observes_completion = false,
                      bool checks_idle_window = false) {
    native = {};
    native.is_native_low = is_native_low;
    native.low_commands = low_commands;
    native.fails_first_low = fails_first_low;
    native.is_sliced = is_sliced;
    native.observes_completion = observes_completion;
    const auto server = current_process_identity();
    expect(server.has_value(), "real parent identity");
    Child low(path, *server,
              {true, should_disconnect, is_native_low, low_commands, fails_first_low}),
        high(path, *server, {false, should_disconnect, is_native_low});
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.session_limits.submissions.max_commands = 2;
    config.persistent.workers = is_sliced ? 3 : 320;
    config.persistent.observes_block_completion = observes_completion;
    config.should_audit_native = config.should_audit_priority = observes_completion;
    if (is_sliced) config.persistent.kind = glimmer::transform::PtxExecutionKind::kSliced;
    config.client_priorities = {{low.identity(), 0}, {high.identity(), 100}};
    config.exit_when_idle = true;
    if (checks_idle_window)
        config.priority_clock_ns = []() noexcept { return native.priority_time; };
    auto dispatch = kDispatch;
    if (uses_saved_profile) {
        config.profile.workers = {320};
        config.profile.warmup_calls = 0;
        config.profile.samples_per_candidate = 1;
        CudaProfileCache measurements(config.profile,
                                      []() noexcept { return native.profile_time; });
        const auto key =
            measurements.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                              ".version 7.0\n.target sm_70\n.address_size 64\n"
                              ".visible .entry kernel() { ret; }",
                              {}, "kernel");
        for (const auto duration : {1U, 100U}) {
            auto choice = measurements.choose(
                key, {{8, 1, 1}, {32, 1, 1}, 0, glimmer::transform::PtxExecutionKind::kPreemptive},
                320, true);
            expect(choice.observation != nullptr, "prepare original-faster priority fixture");
            native.profile_time = 100;
            choice.observation->begin();
            native.profile_time += duration;
            choice.observation->finish(true);
        }
        const auto identity = glimmer::app::profile_runtime_identity();
        expect(identity.has_value(), "bind profile to the actual fixture execution program");
        const auto archive = measurements.encode(*identity);
        config.profile_input = path + ".cache";
        expect(archive && glimmer::app::write_profile_archive(config.profile_input, *archive),
               "publish owned priority measurement input");
        dispatch.device_uuid = [](CUuuid* uuid, CUdevice) {
            ++native.profile_identities;
            *uuid = {};
            uuid->bytes[0] = 1;
            return CUDA_SUCCESS;
        };
        dispatch.driver_version = [](int* version) {
            *version = 12060;
            return CUDA_SUCCESS;
        };
    }
    const auto low_peer = low.identity();
    AuditCapture capture;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start actual authenticated priority owner");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1), "prepare original client resources");
        low.observe_ready();
        high.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded client bootstrap");
    }
    send_byte(low.socket, 'G');
    bool is_high_requested{};
    bool exposed_retreat_publication{};
    unsigned int retreat_waits{}, second_retreat_waits{}, high_waits{}, independent_issue_waits{};
    unsigned int idle_turns{};
    bool resumed_before_idle_expiry{};
    while (!service.finished()) {
        expect(service.step(1), "advance physical priority owner");
        if (checks_idle_window && native.allow_high_stop) {
            // Keep the clock fixed while actual high events and source cleanup
            // drain; even many real owner turns cannot spend synthetic time.
            if (idle_turns < 60 && native.low_launches > low_commands)
                resumed_before_idle_expiry = true;
            ++idle_turns;
            if (idle_turns == 40) native.priority_time += 99999;
            if (idle_turns >= 60) native.priority_time += 100000;
        }
        if (low_commands == 2 && native.low_launches == 1 && !is_high_requested)
            expect(++independent_issue_waits < 200,
                   "independent low command must issue while predecessor is incomplete");
        if (native.low_launches == low_commands && !is_high_requested) {
            if (observes_completion) {
                expect(!service.block_completion_observation(low_peer),
                       "native return/claims do not fabricate an application publication");
                const auto owner = std::ranges::find_if(
                    native.contexts, [](const auto& context) { return context->host != nullptr; });
                expect(owner != native.contexts.end(), "exact live observed mapped owner");
                auto& word = *reinterpret_cast<std::uint32_t*>((*owner)->host.get() + 4);
                std::atomic_ref(word).store(1, std::memory_order_release);  // Fake GPU only.
                const auto publication = service.block_completion_observation(low_peer);
                expect(publication && publication->progress.has_completed_block &&
                           publication->progress.physical_launches == 1 &&
                           !publication->progress.physical_stops &&
                           !service.block_completion_observation(high.identity()),
                       "only exact active low source may expose the independent fake word");
                expect(
                    !service.publish_block_completion_audit(high.identity(), 100, *publication) &&
                        !service.publish_block_completion_audit(low_peer, 100, *publication) &&
                        service.publish_block_completion_audit(low_peer, 0, *publication),
                    "only the exact pinned live peer can publish the actual block witness");
            }
            send_byte(high.socket, 'G');
            is_high_requested = true;
        }
        if ((std::ranges::all_of(std::span{native.saw_retreats}.first(low_commands),
                                 [](bool value) { return value; }) ||
             ((is_native_low || is_sliced) && is_high_requested)) &&
            !native.allow_low_stop) {
            expect(native.high_launches == 0 && native.low_launches == low_commands,
                   "retreat flag is not physical stop authority");
            if (observes_completion && service.block_completion_observation(low_peer))
                exposed_retreat_publication = true;
            if (++retreat_waits == 20) {
                native.allow_low_stop = true;
                native.allow_low_stops[0] = true;
            }
        }
        if (low_commands == 2 && native.allow_low_stops[0] && !native.allow_low_stops[1]) {
            expect(native.high_launches == 0 && native.low_launches == 2,
                   "one stopped low does not authorize high issue while the other remains active");
            if (++second_retreat_waits == 20) native.allow_low_stops[1] = true;
        }
        if (native.high_launches == 2 && !native.allow_high_stop) {
            expect(native.low_launches == low_commands,
                   "two high host replies do not authorize any low resume");
            if (++high_waits == 20) native.allow_high_stop = true;
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded complete priority handoff");
    }
    low.join();
    high.join();
    if (observes_completion) {
        expect(service.request_stop(), "close and drain ordered block witness audit");
        const auto text = capture.messages.str();
        const auto issued = text.find("Priority execution audit: event=low_issued,");
        const auto block = text.find("Block completion audit:");
        expect(issued != std::string::npos && block != std::string::npos && issued < block,
               "the block witness follows the actual issue in the same audit FIFO");
    }
    std::cout << "Priority case: low_commands=" << low_commands << "; native=" << is_native_low
              << "; disconnect=" << should_disconnect << "; low_launches=" << native.low_launches
              << "; high_launches=" << native.high_launches
              << "; initializations=" << native.initializations << "; stop_waits=" << retreat_waits
              << ',' << second_retreat_waits << "; high_waits=" << high_waits
              << "; streams=" << native.stream_destroys << '\n';
    expect(retreat_waits == 20 && high_waits == 20 &&
               second_retreat_waits == (low_commands == 2 ? 20U : 0U) &&
               native.low_launches == low_commands * (is_native_low ? 1U
                                                      : is_sliced   ? 3U
                                                                    : 2U) -
                                          static_cast<unsigned int>(fails_first_low) &&
               native.high_launches == 2 &&
               native.initializations == (is_native_low || is_sliced ? 0U : low_commands) &&
               native.stream_destroys == (low_commands == 2 ? 2U : 0U) &&
               native.saw_retreat == (!is_native_low && !is_sliced) && native.host_frees == 1 &&
               native.destroys == 2 && native.synchronizes == 0 && service.request_stop(),
           "real stop, native high pipeline, exact no-reset resume and once-only private cleanup");
    for (const auto& event : native.events)
        expect(event->is_destroyed, "all real event leases freed");
    expect(!resumed_before_idle_expiry && (!checks_idle_window || idle_turns >= 60),
           "actual low continuation must wait through the post-drain priority idle window");
    expect(!exposed_retreat_publication,
           "Retreat-requested publication remained exposed before physical stop");
    if (uses_saved_profile) {
        const auto snapshot = service.profile_snapshot();
        expect(snapshot.loaded_cases == 1 && snapshot.hits == 1 && snapshot.samples == 0 &&
                   !snapshot.is_learning && native.profile_identities == 1 &&
                   ::unlink((path + ".cache").c_str()) == 0,
               "cached priority retains incomplete retreat/resume despite a faster original; "
               "high peers never consult profile state");
    }
}
void immediate_sliced_batch_uses_exact_resume_phase(const std::string& path) {
    native = {};
    native.is_sliced = true;
    native.low_commands = 2;
    const auto server = current_process_identity();
    expect(server.has_value(), "exact batch owner identity");
    ClientMode low_mode{.is_low = true, .low_commands = 2, .submits_batch = true};
    Child low(path, *server, low_mode);
    AuditCapture capture;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.session_limits.submissions.max_commands = 2;
    config.persistent.workers = 3;
    config.persistent.kind = glimmer::transform::PtxExecutionKind::kSliced;
    config.client_priorities = {{low.identity(), 0, ExecutionClientKernelMode::kConfigured}};
    config.should_audit_native = config.should_audit_priority = true;
    config.exit_when_idle = true;
    auto dispatch = kDispatch;
    dispatch.execution.event_query = [](CUevent value) {
        const auto& event = *reinterpret_cast<Event*>(value);
        expect(event.context == native.current && !event.is_destroyed,
               "only actual live event observations permit immediate slice stop");
        return event.context->host || native.allow_high_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int grid_x, unsigned int,
                                          unsigned int, unsigned int, unsigned int, unsigned int,
                                          unsigned int, CUstream stream, void**, void** extra) {
        expect(native.current->host && !native.high_launches &&
                   function == reinterpret_cast<CUfunction>(&native.current->transformed) &&
                   grid_x == 3 && extra && *static_cast<std::size_t*>(extra[3]) == 20,
               "exact independently terminating low slices retain their private ABI");
        const auto command =
            stream == reinterpret_cast<CUstream>(&native.current->streams[0]) ? 0U : 1U;
        expect(stream == reinterpret_cast<CUstream>(&native.current->streams[command]),
               "each batched command keeps its independent stream");
        std::array<std::uint32_t, 2> slice{};
        std::memcpy(slice.data(), static_cast<const std::byte*>(extra[1]) + 12, sizeof(slice));
        const auto index = native.command_launches[command]++;
        expect(index < 3 && slice == std::array{index * 3U, index == 2 ? 2U : 3U},
               "every immediate continuation advances one disjoint exactly-once interval");
        ++native.low_launches;
        return CUDA_SUCCESS;
    };
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start exact immediate-slice service control");
    send_byte(low.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    const auto step = [&] {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded actual exact batch service progress");
    };
    while (!low.is_ready) {
        step();
        low.observe_ready();
    }
    send_byte(low.socket, 'G');
    wait_byte(low.socket, 'P');
    const auto count = [&](std::string_view marker) {
        const auto messages = capture.messages.str();
        std::size_t result{}, position{};
        while ((position = messages.find(marker, position)) != std::string::npos) {
            ++result;
            position += marker.size();
        }
        return result;
    };
    send_byte(low.socket, 'S');
    while (!service.finished()) step();
    low.join();
    expect(native.low_launches == 6 && native.command_launches == std::array{3U, 3U} &&
               count("Native admission audit:") == 2 &&
               count("Priority execution audit: event=low_issued,") == 2 &&
               count("Priority execution audit: event=low_resumed,") == 4 &&
               native.host_frees == 1 && native.destroys == 1 && !native.synchronizes,
           "actual token phase, not stale pause bookkeeping, classifies every exact continuation");
    for (const auto& event : native.events)
        expect(event->is_destroyed, "immediate exact-batch event cleanup is once-only");
}

void immediate_high_completion_audit(const std::string& path, bool audits_native,
                                     bool audits_priority, bool fails_observation = false) {
    native = {};
    native.fails_high_observation = fails_observation;
    // Add/remove this synchronous sink only while no service/worker is active.
    AuditCapture capture;
    const auto server = current_process_identity();
    expect(server.has_value(), "actual short-command server identity");
    Child high(path, *server, {false, false, false, 1, false, false, 1, fails_observation});
    auto dispatch = kDispatch;
    dispatch.execution.event_query = [](CUevent value) {
        auto& event = *reinterpret_cast<Event*>(value);
        expect(event.context == native.current && !event.is_destroyed,
               "short high completion observes only its live original event");
        ++native.event_queries;
        if (native.fails_high_observation && native.event_queries == 1)
            return CUDA_ERROR_INVALID_VALUE;
        return CUDA_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int gx, unsigned int,
                                          unsigned int, unsigned int tx, unsigned int, unsigned int,
                                          unsigned int, CUstream, void**, void**) {
        expect(function == reinterpret_cast<CUfunction>(&native.current->original) && gx == 8 &&
                   tx == 32 && !native.current->host && native.high_launches++ == 0,
               "one original short high SDK issue, no persistent substitute");
        return CUDA_SUCCESS;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.client_priorities = {{high.identity(), 100}};
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    config.should_audit_native = audits_native;
    config.should_audit_priority = audits_priority;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start authenticated immediate-high owner");
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!high.is_ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded immediate-high resource preparation");
        high.observe_ready();
    }
    send_byte(high.socket, 'G');
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded immediate-high physical completion and cleanup");
    high.join();
    for (const auto& event : native.events) {
        expect(event->is_destroyed, "short high event lease retires exactly once");
    }
    const auto text = capture.messages.str();
    const auto issued = text.find("Priority execution audit: event=high_issued,");
    const auto done = text.find("Priority execution audit: event=high_physical_done,");
    if (audits_priority && fails_observation)
        expect(issued != std::string::npos && done == std::string::npos,
               "failed/quarantined high observation must never publish physical success");
    else if (audits_priority)
        expect(issued != std::string::npos && done != std::string::npos && issued < done &&
                   text.find("Priority execution audit: event=high_physical_done,", done + 1) ==
                       std::string::npos,
               "actual short high completion must publish exactly one post-issue physical proof");
    else
        expect(issued == std::string::npos && done == std::string::npos,
               "priority audit off emits no physical or issue diagnostic");
    expect(native.event_queries == (fails_observation ? 2U : 1U) && native.high_launches == 1 &&
               native.destroys == 1 && native.synchronizes == 0 && service.request_stop(),
           "short high audit cannot add issues, synchronization or native resources");
}

void unissued_low_rejection_audit(const std::string& path, bool audits_native,
                                  bool audits_priority) {
    native = {};
    native.allow_high_stop = true;
    AuditCapture capture;
    const auto server = current_process_identity();
    expect(server.has_value(), "authenticate rejected low and live successor");
    // Priority comes only from the authenticated service table. The client
    // submits two ordinary default-stream commands with no private resources.
    Child low(path, *server, {.fails_first_preparation = true});
    auto dispatch = kDispatch;
    dispatch.execution.event_create = [](CUevent* output, unsigned int flags) {
        expect(native.current != nullptr, "prepare event only on an owned context");
        if (++native.preparation_attempts == 1) {
            *output = nullptr;
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        return event_create(output, flags);
    };
    dispatch.execution.event_query = [](CUevent value) {
        const auto& event = *reinterpret_cast<Event*>(value);
        expect(event.context == native.current && !event.is_destroyed,
               "successor observes its real live event");
        ++native.event_queries;
        return CUDA_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int gx, unsigned int,
                                          unsigned int, unsigned int tx, unsigned int, unsigned int,
                                          unsigned int, CUstream, void**, void**) {
        expect(function == reinterpret_cast<CUfunction>(&native.current->original) && gx == 8 &&
                   tx == 32 && !native.current->host && native.low_launches++ == 0,
               "only the healthy successor reaches the original SDK exactly once");
        return CUDA_SUCCESS;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.client_priorities = {{low.identity(), 0}, {*server, 100}};
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    config.should_audit_native = audits_native;
    config.should_audit_priority = audits_priority;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start low preparation-rejection audit owner");
    send_byte(low.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded rejected-low resource preparation");
        low.observe_ready();
    }
    send_byte(low.socket, 'G');
    while (!service.finished())
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "preparation rejection leaves the queue healthy for its successor");
    low.join();
    std::array<unsigned int, 2> issued{}, done{};
    unsigned int native_issues{};
    std::istringstream messages(capture.messages.str());
    for (std::string line; std::getline(messages, line);) {
        if (line.find("Native execution audit:") != std::string::npos) ++native_issues;
        const auto is_issue =
            line.find("Priority execution audit: event=low_issued,") != std::string::npos;
        const auto is_done =
            line.find("Priority execution audit: event=low_logical_done,") != std::string::npos;
        if (!is_issue && !is_done) continue;
        for (std::size_t index = 0; index != issued.size(); ++index) {
            const auto sequence = "sequence=" + std::to_string(index + 1) + ",";
            if (line.find(sequence) == std::string::npos) continue;
            if (is_issue) ++issued[index];
            if (is_done) ++done[index];
        }
    }
    expect(issued[0] == 0 && done[0] == 0 && issued[1] == (audits_priority ? 1U : 0U) &&
               done[1] == (audits_priority ? 1U : 0U) && native_issues == (audits_native ? 1U : 0U),
           "unissued rejection has no orphan completion or fabricated compute evidence");
    expect(native.preparation_attempts == 2 && native.low_launches == 1 &&
               native.events.size() == 1 && native.events.front()->is_destroyed &&
               native.event_queries == 1 && native.destroys == 1 && native.stream_destroys == 0 &&
               native.synchronizes == 0 && service.request_stop(),
           "successor physical retirement and all native cleanup remain exactly once");
}

void uncertain_preparation_rejection_audit(const std::string& path, CUresult cleanup_error) {
    const auto isolated = ::fork();
    expect(isolated >= 0, "isolate intentionally retained uncertain preparation resources");
    if (isolated == 0) {
        try {
            native = {};
            native.preparation_cleanup_error = cleanup_error;
            AuditCapture capture;
            const auto server = current_process_identity();
            expect(server.has_value(), "authenticate uncertain preparation owner");
            Child low(path, *server,
                      {.fails_first_preparation = true, .holds_rejected_successor = true});
            auto dispatch = kDispatch;
            dispatch.execution.event_create = [](CUevent* output, unsigned int flags) {
                expect(++native.preparation_attempts == 1 &&
                           event_create(output, flags) == CUDA_SUCCESS && *output,
                       "one actual partial event factory output");
                return CUDA_ERROR_OUT_OF_MEMORY;
            };
            dispatch.execution.event_destroy = [](CUevent value) {
                const auto& event = *reinterpret_cast<Event*>(value);
                expect(event.context == native.current && !event.is_destroyed &&
                           ++native.preparation_cleanup_attempts == 1,
                       "partial factory cleanup attempted exactly once on its owner");
                return native.preparation_cleanup_error;
            };
            ExecutionServiceConfig config;
            config.transport.socket_path = path;
            config.transport.io_timeout_ms = 3000;
            config.client_priorities = {{low.identity(), 0}, {*server, 100}};
            config.uses_original_kernels = true;
            config.exit_when_idle = true;
            config.should_audit_native = true;
            config.should_audit_priority = true;
            ExecutionService service(std::move(config), dispatch);
            expect(service.start(), "start uncertain preparation audit owner");
            send_byte(low.socket, 'C');
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!low.is_ready) {
                expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
                       "bounded uncertain preparation setup");
                low.observe_ready();
            }
            send_byte(low.socket, 'G');
            while (true) {
                expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
                       "bounded original preparation reply and successor submission");
                char marker{};
                const auto received = ::recv(low.socket, &marker, 1, MSG_DONTWAIT);
                if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR))
                    continue;
                expect(received == 1 && marker == 'P', "successor remains owned by its caller");
                break;
            }
            for (unsigned int iteration = 0; iteration != 32; ++iteration)
                expect(service.step(0) && !service.finished(),
                       "uncertain preparation remains retained across scheduling passes");
            send_byte(low.socket, 'S');
            low.join();
            const auto messages = capture.messages.str();
            const auto failure = messages.find("event=policy_failure,");
            expect(failure != std::string::npos, "uncertain preparation records policy failure");
            const auto failure_line =
                messages.substr(failure, messages.find('\n', failure) - failure);
            expect(failure_line.find("status=" + std::to_string(static_cast<int>(cleanup_error))) !=
                           std::string::npos &&
                       messages.find("event=policy_failure,", failure + 1) == std::string::npos &&
                       messages.find("event=low_issued,") == std::string::npos &&
                       messages.find("event=low_logical_done,") == std::string::npos &&
                       messages.find("Native execution audit:") == std::string::npos,
                   "same or stronger cleanup failure is never clean rejection or compute proof");
            expect(native.preparation_attempts == 1 && native.preparation_cleanup_attempts == 1 &&
                       native.low_launches == 0 && native.high_launches == 0 &&
                       native.event_queries == 0 && native.synchronizes == 0 &&
                       native.destroys == 0 && native.events.size() == 1 &&
                       !native.events.front()->is_destroyed,
                   "no successor SDK entry, cleanup retry or invented uncertain physical stop");
            // Unknown resources deliberately survive the service owner. Exit
            // the isolated process without invoking cleanup or claiming no leaks.
            ::_exit(EXIT_SUCCESS);
        } catch (const std::exception& error) {
            std::cerr << "Uncertain preparation child: " << error.what() << '\n';
            ::_exit(EXIT_FAILURE);
        }
    }
    int status{};
    expect(::waitpid(isolated, &status, 0) == isolated && WIFEXITED(status) &&
               WEXITSTATUS(status) == EXIT_SUCCESS,
           "isolated uncertain preparation assertions passed");
    expect(::unlink(path.c_str()) == 0, "remove terminated isolated owner's fixture socket");
}

void naive_and_original_execution(const std::string& path, bool uses_naive,
                                  bool has_priority_table = true, bool uses_original = true) {
    native = {};
    native.allows_naive_overlap = uses_naive;
    native.uses_original_mode = uses_original;
    const auto server = current_process_identity();
    expect(server.has_value(), "actual execution-mode server identity");
    Child low(path, *server, {true, false, false, 1, false, true}),
        high(path, *server, {false, false, false, 1, false, true});
    auto dispatch = kDispatch;
    dispatch.module_load = [](CUmodule* output, const void* source) {
        expect(!native.uses_original_mode ||
                   (!native.current->host && native.current->counters.empty()),
               "explicit original mode allocates no private persistent preparation");
        if (std::string_view(static_cast<const char*>(source)).find("// mode-low") !=
            std::string_view::npos)
            native.mode_low_context = native.current;
        *output = reinterpret_cast<CUmodule>(native.current);
        return CUDA_SUCCESS;
    };
    dispatch.execution.event_query = [](CUevent value) {
        const auto& event = *reinterpret_cast<Event*>(value);
        expect(event.context == native.current && !event.is_destroyed,
               "mode control observes only an actual current live event");
        const bool ready = event.context == native.mode_low_context ? native.allow_low_stop
                                                                    : native.allow_high_stop;
        return ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original CUDA ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int grid_x, unsigned int,
                                          unsigned int, unsigned int threads, unsigned int,
                                          unsigned int, unsigned int, CUstream, void**,
                                          void** extra) {
        if (native.uses_original_mode) {
            expect(function == reinterpret_cast<CUfunction>(&native.current->original) &&
                       grid_x == 8 && threads == 32 && !native.current->host &&
                       native.current->counters.empty(),
                   "transformable source uses the original application ABI, never a private "
                   "substitute");
        } else {
            expect(function == reinterpret_cast<CUfunction>(&native.current->transformed) &&
                       grid_x == 320 && threads == 32 && extra &&
                       *static_cast<std::size_t*>(extra[3]) == 32,
                   "naive permits explicitly configured persistent preparation on both peers");
            CUdeviceptr counter{};
            std::memcpy(&counter, static_cast<const std::byte*>(extra[1]) + 16, sizeof(counter));
            std::uint64_t before{};
            std::memcpy(&before, device_bytes(counter, sizeof(before)), sizeof(before));
            expect(before == 0, "each naive private command starts once without replay");
            constexpr std::uint64_t complete = 8;
            std::memcpy(device_bytes(counter, sizeof(complete)), &complete, sizeof(complete));
        }
        if (native.current == native.mode_low_context) {
            expect(native.low_launches++ == 0, "original low computation executes exactly once");
        } else {
            expect(
                native.low_launches == 1 && (native.allows_naive_overlap || native.allow_low_stop),
                "only naive may issue high before actual original low stop");
            expect(
                native.high_launches < 2 &&
                    (native.uses_original_mode ? !native.allow_high_stop
                                               : !native.high_launches || native.allow_high_stop),
                "native highs pipeline, while private highs retain their logical stream "
                "dependency");
            ++native.high_launches;
        }
        return CUDA_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.session_limits.submissions.max_commands = 2;
    config.persistent.workers = uses_naive ? 320 : 0;
    config.uses_original_kernels = uses_original;
    config.scheduling_mode = uses_naive ? glimmer::core::ExecutionSchedulingMode::kNaive
                                        : glimmer::core::ExecutionSchedulingMode::kPriority;
    if (has_priority_table)
        config.client_priorities = {{low.identity(), 0}, {high.identity(), 100}};
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start original/naive service with actual authenticated peers");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1), "bootstrap unchanged original execution clients");
        low.observe_ready();
        high.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded original-mode bootstrap");
    }
    send_byte(low.socket, 'G');
    bool high_requested{};
    unsigned int original_waits{}, pending_pipeline_waits{}, private_dependency_waits{};
    while (!service.finished()) {
        expect(service.step(1), "advance actual original-mode resource owner");
        if (native.low_launches && !high_requested) {
            send_byte(high.socket, 'G');
            high_requested = true;
        }
        if (high_requested && !native.allow_low_stop) {
            if (uses_naive) {
                expect(!native.mode_low_context->is_destroyed,
                       "naive high admission cannot retire or reclaim pending low work");
            } else {
                expect(native.high_launches == 0 && native.low_launches == 1,
                       "priority-original waits for actual nonpreemptible low completion");
                if (++original_waits == 20) native.allow_low_stop = true;
            }
        }
        if (!uses_original && native.high_launches == 1 && !native.allow_high_stop) {
            expect(!native.allow_low_stop && native.low_launches == 1,
                   "naive high private work is pending concurrently with low physical ownership");
            if (++private_dependency_waits == 20) native.allow_high_stop = true;
        }
        if (!uses_original && native.high_launches == 2 && !native.allow_low_stop)
            native.allow_low_stop = true;
        if (uses_original && native.high_launches == 2 && !native.allow_high_stop) {
            expect(native.low_launches == 1,
                   "two high host replies never replay original low work");
            if (uses_naive)
                expect(!native.allow_low_stop, "naive high pipeline precedes original low stop");
            if (++pending_pipeline_waits == 20) {
                native.allow_high_stop = true;
                native.allow_low_stop = true;
            }
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded original/naive cleanup");
    }
    low.join();
    high.join();
    expect(native.low_launches == 1 && native.high_launches == 2 &&
               original_waits == (uses_naive ? 0U : 20U) &&
               pending_pipeline_waits == (uses_original ? 20U : 0U) &&
               private_dependency_waits == (uses_original ? 0U : 20U) &&
               native.initializations == (uses_original ? 0U : 3U) &&
               native.host_frees == (uses_original ? 0U : 2U) && !native.saw_retreat &&
               native.destroys == 2 && native.synchronizes == 0 && service.request_stop(),
           "scheduler and kernel modes preserve selected preparation, native pipeline, "
           "dependencies and exact cleanup");
    for (const auto& event : native.events)
        expect(event->is_destroyed, "original-mode event retires once");
    const auto cache = service.ptx_cache_snapshot();
    expect(cache.is_valid && cache.lookups == (uses_original ? 0U : 2U) &&
               cache.transforms == (uses_original ? 0U : 2U),
           "original mode never consults transformation cache; preemptive mode keeps exact "
           "distinct sources");
}

enum class ClientSelectionCase {
    kConfiguredEqual,
    kConfiguredPriority,
    kAutomaticEqual,
    kOriginalPriority,
    kConfiguredEqualProfile
};

void client_kernel_selection(const std::string& path, ClientSelectionCase scenario) {
    native = {};
    native.selection_is_equal = scenario == ClientSelectionCase::kConfiguredEqual ||
                                scenario == ClientSelectionCase::kAutomaticEqual ||
                                scenario == ClientSelectionCase::kConfiguredEqualProfile;
    native.selection_is_configured = scenario != ClientSelectionCase::kAutomaticEqual &&
                                     scenario != ClientSelectionCase::kOriginalPriority;
    const bool uses_profile = scenario == ClientSelectionCase::kConfiguredEqualProfile;
    AuditCapture capture;
    const auto server = current_process_identity();
    expect(server.has_value(), "actual client-selection server identity");
    Child low(path, *server, {true, false, false, 1, false, true}),
        high(path, *server, {false, false, false, 1, false, true});
    const auto low_peer = low.identity(), high_peer = high.identity();
    auto dispatch = kDispatch;
    dispatch.module_load = [](CUmodule* output, const void* source) {
        if (std::string_view(static_cast<const char*>(source)).find("// mode-low") !=
            std::string_view::npos)
            native.mode_low_context = native.current;
        *output = reinterpret_cast<CUmodule>(native.current);
        return CUDA_SUCCESS;
    };
    dispatch.execution.event_query = [](CUevent value) {
        const auto& event = *reinterpret_cast<Event*>(value);
        expect(event.context == native.current && !event.is_destroyed,
               "selection control observes an actual live current-context event");
        if (event.context != native.mode_low_context)
            return native.allow_high_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
        if (event.context->host) {
            std::uint32_t retreat{};
            std::memcpy(&retreat, event.context->host.get(), sizeof(retreat));
            native.saw_retreat = native.saw_retreat || retreat != 0;
            expect(!native.selection_is_equal || !retreat,
                   "numeric equals never request each other's retreat");
            if (!native.selection_is_equal && event.epoch == 0)
                return native.allow_low_stop && retreat ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
        }
        return native.allow_low_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original Driver ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int grid_x, unsigned int,
                                          unsigned int, unsigned int threads, unsigned int,
                                          unsigned int, unsigned int, CUstream, void**,
                                          void** extra) {
        const bool is_low = native.current == native.mode_low_context;
        if (is_low && native.selection_is_configured) {
            expect(function == reinterpret_cast<CUfunction>(&native.current->transformed) &&
                       grid_x == 320 && threads == 32 && extra &&
                       *static_cast<std::size_t*>(extra[3]) == 32,
                   "configured peer uses the actual private function and explicit 320 workers");
            CUdeviceptr counter{};
            std::memcpy(&counter, static_cast<const std::byte*>(extra[1]) + 16, sizeof(counter));
            std::uint64_t before{};
            std::memcpy(&before, device_bytes(counter, sizeof(before)), sizeof(before));
            expect((native.low_launches == 0 && before == 0) ||
                       (!native.selection_is_equal && native.low_launches == 1 && before == 3 &&
                        native.high_launches == 2 && native.allow_high_stop),
                   "configured resume preserves the counter and waits for physical high stop");
            const std::uint64_t after = native.selection_is_equal || native.low_launches ? 8 : 3;
            std::memcpy(device_bytes(counter, sizeof(after)), &after, sizeof(after));
            ++native.low_launches;
        } else {
            expect(
                function == reinterpret_cast<CUfunction>(&native.current->original) &&
                    grid_x == 8 && threads == 32 && !native.current->host &&
                    native.current->counters.empty(),
                "original and automatic-high peers retain the original ABI without private state");
            if (is_low)
                expect(native.low_launches++ == 0, "original lower peer executes exactly once");
            else {
                expect(native.low_launches == 1 &&
                           (native.selection_is_equal || native.allow_low_stop) &&
                           !native.allow_high_stop && native.high_launches < 2,
                       "only equal priority permits high issue before the actual low stop");
                ++native.high_launches;
            }
        }
        return CUDA_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.session_limits.submissions.max_commands = 2;
    config.persistent.workers = 320;
    config.should_audit_priority = true;
    config.exit_when_idle = true;
    const auto low_mode = native.selection_is_configured ? ExecutionClientKernelMode::kConfigured
                          : native.selection_is_equal    ? ExecutionClientKernelMode::kAuto
                                                         : ExecutionClientKernelMode::kOriginal;
    const auto high_mode = scenario == ClientSelectionCase::kAutomaticEqual
                               ? ExecutionClientKernelMode::kAuto
                               : ExecutionClientKernelMode::kOriginal;
    const auto high_priority = native.selection_is_equal ? 0U : 100U;
    config.client_priorities = {{low_peer, 0, low_mode}, {high_peer, high_priority, high_mode}};
    if (uses_profile) {
        config.profile.workers = {320};
        config.profile.warmup_calls = 0;
        config.profile.samples_per_candidate = 1;
        CudaProfileCache measurements(config.profile,
                                      []() noexcept { return native.profile_time; });
        const auto key =
            measurements.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                              "// mode-low\n.version 7.0\n.target sm_70\n.address_size 64\n"
                              ".visible .entry kernel() { ret; }",
                              {}, "kernel");
        for (const auto duration : {1U, 100U}) {
            auto choice = measurements.choose(
                key, {{8, 1, 1}, {32, 1, 1}, 0, glimmer::transform::PtxExecutionKind::kPreemptive},
                320, true);
            expect(choice.observation != nullptr, "prepare original-faster equal-priority profile");
            native.profile_time = 100;
            choice.observation->begin();
            native.profile_time += duration;
            choice.observation->finish(true);
        }
        const auto identity = glimmer::app::profile_runtime_identity();
        expect(identity.has_value(), "bind equal-priority profile to this exact program");
        const auto archive = measurements.encode(*identity);
        config.profile_input = path + ".cache";
        expect(archive && glimmer::app::write_profile_archive(config.profile_input, *archive),
               "publish bounded original-faster profile fixture");
        dispatch.device_uuid = [](CUuuid* uuid, CUdevice) {
            ++native.profile_identities;
            *uuid = {};
            uuid->bytes[0] = 1;
            return CUDA_SUCCESS;
        };
        dispatch.driver_version = [](int* version) {
            *version = 12060;
            return CUDA_SUCCESS;
        };
    }
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start explicit trusted client-selection service");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1), "bootstrap actual authenticated selection clients");
        low.observe_ready();
        high.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded client-selection bootstrap");
    }
    send_byte(low.socket, 'G');
    bool high_requested{};
    unsigned int low_waits{}, high_waits{};
    while (!service.finished()) {
        expect(service.step(1), "advance numeric policy with independently selected functions");
        if (native.low_launches && !high_requested) {
            send_byte(high.socket, 'G');
            high_requested = true;
        }
        if (high_requested && !native.selection_is_equal && !native.allow_low_stop) {
            expect(!native.high_launches && native.low_launches == 1,
                   "numeric higher priority still waits for physical low completion");
            if ((!native.selection_is_configured || native.saw_retreat) && ++low_waits == 20)
                native.allow_low_stop = true;
        }
        if (native.high_launches == 2 && !native.allow_high_stop) {
            expect(
                native.low_launches == 1 && (!native.selection_is_equal || !native.allow_low_stop),
                "equal high overlaps pending low; unequal low cannot resume on host replies");
            if (++high_waits == 20) {
                native.allow_high_stop = true;
                native.allow_low_stop = true;
            }
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded selection cleanup");
    }
    low.join();
    high.join();
    const bool resumes = native.selection_is_configured && !native.selection_is_equal;
    expect(
        native.low_launches == (resumes ? 2U : 1U) && native.high_launches == 2 &&
            low_waits == (native.selection_is_equal ? 0U : 20U) && high_waits == 20 &&
            native.saw_retreat == resumes &&
            native.initializations == (native.selection_is_configured ? 1U : 0U) &&
            native.host_frees == (native.selection_is_configured ? 1U : 0U) &&
            native.destroys == 2 && !native.synchronizes && service.request_stop(),
        "selection preserves exact initialization, numeric physical gates and once-only cleanup");
    for (const auto& event : native.events)
        expect(event->is_destroyed, "selection event retires exactly once");
    const auto cache = service.ptx_cache_snapshot();
    expect(cache.is_valid && cache.lookups == (native.selection_is_configured ? 1U : 0U) &&
               cache.transforms == (native.selection_is_configured ? 1U : 0U),
           "explicit original and automatic equal peers never consult the transform cache");
    const auto logs = capture.messages.str();
    const auto identity_text = [](const TaskPeerIdentity& peer) {
        return ", pid=" + std::to_string(peer.pid) +
               ", start=" + std::to_string(peer.start_time_ticks);
    };
    const std::string requested_low = native.selection_is_configured ? "configured"
                                      : native.selection_is_equal    ? "auto"
                                                                     : "original";
    const auto low_selection =
        identity_text(low_peer) + ", priority=0, requested=" + requested_low +
        ", effective=" + (native.selection_is_configured ? "configured" : "original") +
        ", retained_epochs=" +
        (scenario == ClientSelectionCase::kAutomaticEqual ? "false" : "true");
    const auto high_selection =
        identity_text(high_peer) + ", priority=" + std::to_string(high_priority) +
        ", requested=" + (high_mode == ExecutionClientKernelMode::kAuto ? "auto" : "original") +
        ", effective=original, retained_epochs=false";
    expect(
        logs.find(low_selection) != std::string::npos &&
            logs.find(high_selection) != std::string::npos,
        "cold selection receipt binds exact authenticated identity, numeric priority and cohort");
    expect((logs.find("event=low_issued") != std::string::npos) ==
               (scenario != ClientSelectionCase::kAutomaticEqual),
           "configured highest belongs to retained audit cohort, automatic equal does not");
    expect((logs.find("event=low_resumed") != std::string::npos) == resumes,
           "only genuinely lower configured work resumes");
    if (uses_profile) {
        const auto profile = service.profile_snapshot();
        expect(profile.loaded_cases == 1 && profile.hits == 1 && !profile.samples &&
                   !profile.is_learning && native.profile_identities == 1 &&
                   ::unlink((path + ".cache").c_str()) == 0,
               "configured highest rejects faster native profile; original peer never consults it");
    }
}

void profile_serializes_service_work(const std::string& path) {
    native = {};
    const auto server = current_process_identity();
    expect(server.has_value(), "actual isolated-profile owner identity");
    Child low(path, *server, {true, false, false, 1, false, true}),
        high(path, *server, {false, false, false, 1, false, true});
    auto dispatch = kDispatch;
    dispatch.device_uuid = [](CUuuid* uuid, CUdevice) {
        *uuid = {};
        uuid->bytes[0] = 1;
        return CUDA_SUCCESS;
    };
    dispatch.driver_version = [](int* version) {
        *version = 12060;
        return CUDA_SUCCESS;
    };
    dispatch.module_load = [](CUmodule* output, const void* source) {
        if (std::string_view(static_cast<const char*>(source)).find("// mode-low") !=
            std::string_view::npos)
            native.mode_low_context = native.current;
        *output = reinterpret_cast<CUmodule>(native.current);
        return CUDA_SUCCESS;
    };
    dispatch.execution.event_query = [](CUevent value) {
        const auto& event = *reinterpret_cast<Event*>(value);
        expect(event.context == native.current && !event.is_destroyed,
               "profile observes real current native events");
        return (native.current == native.mode_low_context ? native.allow_low_stop
                                                          : native.allow_high_stop)
                   ? CUDA_SUCCESS
                   : CUDA_ERROR_NOT_READY;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int gx, unsigned int,
                                          unsigned int, unsigned int, unsigned int, unsigned int,
                                          unsigned int, CUstream, void**, void**) {
        expect(function == reinterpret_cast<CUfunction>(&native.current->original) && gx == 8,
               "original profile executes unchanged application kernels exactly once");
        if (native.current == native.mode_low_context) {
            expect(native.low_launches++ == 0, "one low request, one actual launch");
        } else {
            expect(native.allow_low_stop && native.low_launches == 1 &&
                       (!native.high_launches || native.allow_high_stop),
                   "profile excludes both another peer and a prior same-peer physical interval");
            ++native.high_launches;
        }
        return CUDA_SUCCESS;
    };
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
    config.uses_original_kernels = true;
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.exit_when_idle = true;
    ExecutionService service(std::move(config), dispatch);
    expect(service.start(), "start explicit profile mode without a priority table");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded profile clients prepare ordinary original resources");
        low.observe_ready();
        high.observe_ready();
    }
    send_byte(low.socket, 'G');
    bool high_requested{};
    unsigned int low_waits{}, high_waits{};
    while (!service.finished()) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded once-only isolated-profile execution");
        if (native.low_launches && !high_requested) {
            send_byte(high.socket, 'G');
            high_requested = true;
        }
        if (native.low_launches && !native.allow_low_stop) {
            expect(!native.high_launches && service.profile_snapshot().samples == 0,
                   "host result never publishes profile completion or admits another peer");
            if (++low_waits == 20) native.allow_low_stop = true;
        }
        if (native.high_launches == 1 && !native.allow_high_stop) {
            expect(service.profile_snapshot().samples == 1,
                   "first high remains unmeasured until its actual stop");
            if (++high_waits == 20) native.allow_high_stop = true;
        }
    }
    low.join();
    high.join();
    const auto profile = service.profile_snapshot();
    expect(low_waits == 20 && high_waits == 20 && native.low_launches == 1 &&
               native.high_launches == 2 && profile.samples == 2 && profile.trained_cases == 2 &&
               profile.hits == 1 && !profile.rejected_samples && !profile.cancelled &&
               native.destroys == 2 && service.request_stop(),
           "actual IPC profiling reuses completed observations without replay or leaked ownership");
}

enum class SharingCase {
    kQualified,
    kGlobalCap,
    kBelowFloor,
    kIncompleteProfile,
    kFittingGeometry
};
struct SharingNative {
    unsigned int logical_blocks{}, expected_workers{}, private_launches{};
} sharing;

std::string sharing_source(bool is_low) {
    return std::string(is_low ? "// mode-low\n" : "") +
           ".version 7.0\n.target sm_70\n.address_size 64\n"
           ".visible .entry kernel() { ret; }";
}

void write_sharing_profile(ExecutionServiceConfig& config, SharingCase scenario) {
    CudaProfileCache measurements(config.profile, []() noexcept { return native.profile_time; });
    const std::array kinds{glimmer::transform::PtxExecutionKind::kDynamic};
    const CudaProfileGeometry geometry{
        {sharing.logical_blocks, 1, 1}, {256, 1, 1}, 0, kinds.front()};
    // Each complete case includes original and all three geometries. The
    // fastest private candidate, 320, exceeds the later sharing resource cap.
    const std::array<unsigned int, 4> durations{
        100, 130, scenario == SharingCase::kBelowFloor ? 120U : 105U, 80};
    for (const bool is_low : {false, true}) {
        const auto key =
            measurements.bind({{std::byte{1}}, 12060, CUDA_VERSION, kCudaPtxGeneratorAbi},
                              sharing_source(is_low), {}, "kernel");
        const auto count = scenario == SharingCase::kIncompleteProfile ? 3U : 4U;
        for (unsigned int index = 0; index < count; ++index) {
            auto choice = measurements.choose(key, geometry, 320, kinds);
            expect(choice.observation && choice.workers == (index ? 80U << (index - 1) : 0),
                   "independent broad profile observes each original/private geometry once");
            native.profile_time += 1000;
            choice.observation->begin();
            native.profile_time += durations[index];
            choice.observation->finish(true);
        }
    }
    const auto identity = glimmer::app::profile_runtime_identity();
    expect(identity.has_value(), "sharing archive uses the exact running program identity");
    const auto archive = measurements.encode(*identity);
    expect(archive && glimmer::app::write_profile_archive(config.profile_input, *archive),
           "write independently measured complete supersets, never forged selected decisions");
}

CudaSessionDispatch sharing_dispatch() {
    auto dispatch = kDispatch;
    dispatch.device_uuid = [](CUuuid* uuid, CUdevice device) {
        expect(device == 0, "sharing identity is the selected owned device");
        ++native.profile_identities;
        *uuid = {};
        uuid->bytes[0] = 1;
        return CUDA_SUCCESS;
    };
    dispatch.driver_version = [](int* version) {
        *version = 12060;
        return CUDA_SUCCESS;
    };
    dispatch.device_attribute = [](int* value, CUdevice_attribute attribute, CUdevice device) {
        expect(device == 0, "sharing hardware bounds belong to the selected device");
        ++native.sharing_attribute_calls;
        if (attribute == CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            *value = 80;
        else if (attribute == CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
            *value = 2048;
        else
            throw std::runtime_error("unexpected sharing cold hardware query");
        return CUDA_SUCCESS;
    };
    dispatch.module_load = [](CUmodule* output, const void* source) {
        if (std::string_view(static_cast<const char*>(source)).find("// mode-low") !=
            std::string_view::npos)
            native.mode_low_context = native.current;
        *output = reinterpret_cast<CUmodule>(native.current);
        return CUDA_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact Driver ABI.
    dispatch.parameter_info = [](CUfunction function, std::size_t index, std::size_t* offset,
                                 std::size_t* bytes) {
        constexpr std::array<std::size_t, 4> offsets{0, 4, 8, 16};
        expect(function == reinterpret_cast<CUfunction>(&native.current->transformed) &&
                   index < offsets.size(),
               "Dynamic sharing validates the real four-argument private ABI");
        *offset = offsets[index];
        *bytes = index == 3 ? 8 : 4;
        return CUDA_SUCCESS;
    };
    dispatch.execution.event_query = [](CUevent value) {
        const auto& event = *reinterpret_cast<Event*>(value);
        expect(event.context == native.current && !event.is_destroyed,
               "sharing physical proof uses a live event in its actual owning context");
        ++native.event_queries;
        return native.allow_low_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact Driver ABI.
    dispatch.execution.launch_kernel = [](CUfunction function, unsigned int grid_x,
                                          unsigned int grid_y, unsigned int grid_z,
                                          unsigned int threads_x, unsigned int threads_y,
                                          unsigned int threads_z, unsigned int shared_bytes,
                                          CUstream stream, void** parameters, void** extra) {
        expect(native.current && native.current->host && !native.allow_low_stop && grid_y == 1 &&
                   grid_z == 1 && threads_x == 256 && threads_y == 1 && threads_z == 1 &&
                   !shared_bytes && !stream && !parameters,
               "two real clients issue unchanged geometry before either physical completion");
        if (sharing.expected_workers) {
            expect(function == reinterpret_cast<CUfunction>(&native.current->transformed) &&
                       grid_x == sharing.expected_workers && extra &&
                       *static_cast<std::size_t*>(extra[3]) == 24,
                   "slower qualified Dynamic candidate actually issues 160, not global 320");
            std::array<std::uint32_t, 3> logical{};
            std::memcpy(logical.data(), extra[1], sizeof(logical));
            expect(logical == std::array<std::uint32_t, 3>{sharing.logical_blocks, 1, 1},
                   "resource-limited private launch retains the complete original logical grid");
            CUdeviceptr counter{};
            std::memcpy(&counter, static_cast<const std::byte*>(extra[1]) + 16, sizeof(counter));
            std::uint64_t before{};
            std::memcpy(&before, device_bytes(counter, sizeof(before)), sizeof(before));
            expect(!before, "each sharing command starts with its own fresh physical counter");
            const std::uint64_t complete = sharing.logical_blocks;
            std::memcpy(device_bytes(counter, sizeof(complete)), &complete, sizeof(complete));
            ++sharing.private_launches;
        } else {
            expect(function == reinterpret_cast<CUfunction>(&native.current->original) &&
                       grid_x == sharing.logical_blocks,
                   "unqualified sharing requests execute original, not configured private work");
        }
        if (native.current == native.mode_low_context)
            expect(native.low_launches++ == 0,
                   "first sharing client executes once, without replay");
        else
            expect(native.high_launches++ == 0 && native.low_launches == 1,
                   "second sharing client issues while first event is deliberately pending");
        return CUDA_SUCCESS;
    };
    // NOLINTEND(bugprone-easily-swappable-parameters)
    return dispatch;
}

void workload_agnostic_execution(const std::string& path, SharingCase scenario) {
    native = {};
    sharing = {
        scenario == SharingCase::kFittingGeometry ? 80U : 1024U,
        scenario == SharingCase::kQualified || scenario == SharingCase::kGlobalCap ? 160U : 0U, 0};
    AuditCapture capture;
    const auto server = current_process_identity();
    expect(server.has_value(), "sharing server has an authenticated process identity");
    Child low(path, *server,
              {.is_low = true,
               .is_mode_control = true,
               .high_commands = 1,
               .logical_blocks = sharing.logical_blocks,
               .block_threads = 256}),
        high(path, *server,
             {.is_mode_control = true,
              .high_commands = 1,
              .logical_blocks = sharing.logical_blocks,
              .block_threads = 256});
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.io_timeout_ms = 3000;
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kWorkloadAgnostic;
    config.persistent.workers = scenario == SharingCase::kGlobalCap ? 160U : 320U;
    config.persistent.kind = glimmer::transform::PtxExecutionKind::kDynamic;
    config.persistent.sharing_threads_per_sm = scenario == SharingCase::kGlobalCap ? 1024U : 512U;
    config.persistent.sharing_minimum_speed_percent = 90;
    config.profile.workers = {80, 160, 320};
    config.profile.kinds = {glimmer::transform::PtxExecutionKind::kDynamic};
    config.profile.warmup_calls = 0;
    config.profile.samples_per_candidate = 1;
    config.profile_input = path + ".cache";
    config.should_audit_native = true;
    config.exit_when_idle = true;
    write_sharing_profile(config, scenario);
    ExecutionService service(std::move(config), sharing_dispatch());
    expect(service.start(), "start readonly workload-agnostic service with no priority table");
    send_byte(low.socket, 'C');
    send_byte(high.socket, 'C');
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!low.is_ready || !high.is_ready) {
        expect(service.step(1), "prepare both actual sharing clients");
        low.observe_ready();
        high.observe_ready();
        expect(std::chrono::steady_clock::now() < deadline, "bounded sharing bootstrap");
    }
    send_byte(low.socket, 'G');
    bool second_requested{};
    unsigned int pending_advances{};
    while (!service.finished()) {
        expect(service.step(1), "advance sharing while preserving actual physical ownership");
        if (native.low_launches && !second_requested) {
            send_byte(high.socket, 'G');
            second_requested = true;
        }
        if (native.high_launches && !native.allow_low_stop) {
            expect(native.low_launches == 1 && native.events.size() == 2 &&
                       std::ranges::none_of(
                           native.events, [](const auto& event) { return event->is_destroyed; }) &&
                       !native.destroys && !native.host_frees &&
                       service.profile_snapshot().samples == 0,
                   "SDK replies do not retire either pending owner or learn ordinary samples");
            if (++pending_advances == 20) native.allow_low_stop = true;
        }
        expect(std::chrono::steady_clock::now() < deadline, "bounded sharing completion");
    }
    low.join();
    high.join();
    const auto profile = service.profile_snapshot();
    const auto complete_cases = scenario == SharingCase::kIncompleteProfile ? 0U : 2U;
    expect(pending_advances == 20 && native.low_launches == 1 && native.high_launches == 1 &&
               sharing.private_launches == (sharing.expected_workers ? 2U : 0U) &&
               native.initializations == (sharing.expected_workers ? 2U : 0U) &&
               native.host_frees == 2 && native.destroys == 2 && !native.synchronizes &&
               native.profile_identities == 2 && native.sharing_attribute_calls == 4 &&
               profile.loaded_cases == complete_cases && !profile.samples && !profile.is_learning &&
               !profile.rejected_samples && !profile.cancelled && service.request_stop(),
           "sharing retains exact once-only issue, readonly profile reuse and physical cleanup");
    expect(!sharing.expected_workers || profile.hits == 2,
           "resource-limited decisions reuse both broad complete profile records");
    for (const auto& event : native.events)
        expect(event->is_destroyed, "every sharing event is destroyed after its actual proof");
    std::istringstream lines(capture.messages.str());
    std::string line;
    unsigned int persistent_records{};
    while (std::getline(lines, line)) {
        if (line.find("Native persistent audit:") == std::string::npos) continue;
        ++persistent_records;
        expect(
            line.find("total_blocks=1024, physical_launches=1, workers=160") != std::string::npos,
            "trusted audit reports actual selected workers, never the configured maximum");
    }
    expect(
        persistent_records == sharing.private_launches && ::unlink((path + ".cache").c_str()) == 0,
        "native fallback has no fabricated persistent receipt and owned archive is removed");
}

void invalid_sharing_configuration(const std::string& path) {
    native = {};
    const auto peer = current_process_identity();
    expect(peer.has_value(), "configuration control has an actual process identity");
    const auto identity = glimmer::app::profile_runtime_identity();
    expect(identity.has_value(), "configuration controls bind an actual profile archive");
    CudaProfileCache empty_profile;
    const auto archive = empty_profile.encode(*identity);
    const auto profile_path = path + ".cache";
    expect(archive && glimmer::app::write_profile_archive(profile_path, *archive),
           "configuration negatives must not pass because an unrelated archive is missing");
    for (unsigned int index = 0; index < 12; ++index) {
        ExecutionServiceConfig config;
        config.transport.socket_path = path;
        config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kWorkloadAgnostic;
        config.persistent.workers = 320;
        config.profile_input = profile_path;
        auto dispatch = sharing_dispatch();
        if (index == 0) config.profile_input.clear();
        if (index == 1) config.persistent.workers = 0;
        if (index == 2) config.uses_original_kernels = true;
        if (index == 3) config.persistent.replace_sgemm = true;
        if (index == 4) config.client_priorities = {{*peer, 100}};
        if (index == 5) config.profile_output = path + ".output";
        if (index == 6) config.persistent.sharing_threads_per_sm = 0;
        if (index == 7) config.persistent.sharing_minimum_speed_percent = 0;
        if (index == 8) config.persistent.sharing_minimum_speed_percent = 10001;
        if (index == 9) dispatch.device_attribute = nullptr;
        if (index == 10) config.persistent.sharing_threads_per_sm = 65537;
        bool was_rejected{};
        try {
            ExecutionService service(std::move(config), dispatch);
            expect(index == 11 && !service.profile_snapshot().is_learning,
                   "only the otherwise-identical valid readonly configuration is accepted");
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        expect(was_rejected == (index != 11) && native.contexts.empty() && native.events.empty() &&
                   !native.profile_identities && !native.sharing_attribute_calls &&
                   ::access(path.c_str(), F_OK) != 0,
               "invalid sharing policy rejects before SDK identity, resources or publication");
    }
    expect(::unlink(profile_path.c_str()) == 0, "remove the owned configuration-control archive");
}

void stale_priority_rejected_before_sdk(const std::string& path,
                                        glimmer::core::ExecutionSchedulingMode mode =
                                            glimmer::core::ExecutionSchedulingMode::kPriority) {
    native = {};
    const auto server = current_process_identity();
    expect(server.has_value(), "actual server identity for rejection");
    auto stale = *server;
    ++stale.start_time_ticks;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.persistent.workers = 320;
    config.client_priorities = {{stale, 100}};
    config.scheduling_mode = mode;
    ExecutionService service(std::move(config), kDispatch);
    expect(service.start(), "start strict priority rejection service");
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient connection({path, *server, 3000});
        const auto rejected = connection.open();
        expect(
            rejected && rejected->status == CUDA_ERROR_NOT_PERMITTED && !rejected->original_status,
            "stale PID/start cannot self-promote priority");
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded authenticated rejection");
    }
    client.get();
    expect(native.contexts.empty() && native.events.empty() && native.initializations == 0 &&
               service.request_stop(),
           "unlisted identity never reaches native context or pool");
}
void invalid_priority_configuration(const std::string& path) {
    const auto peer = current_process_identity();
    expect(peer.has_value(), "actual configuration identity");
    const auto contexts_before = native.contexts.size(), events_before = native.events.size();
    for (unsigned int mode = 0; mode < 16; ++mode) {
        ExecutionServiceConfig config;
        config.transport.socket_path = path;
        config.persistent.workers = mode == 0 ? 0 : 320;
        config.client_priorities = {{*peer, 100}};
        if (mode == 1) config.client_priorities.push_back({*peer, 0});
        if (mode == 2) ++config.client_priorities.front().peer.uid;
        if (mode == 3)
            config.scheduling_mode = static_cast<glimmer::core::ExecutionSchedulingMode>(255);
        if (mode == 4) {
            config.uses_original_kernels = true;
            config.persistent.replace_sgemm = true;
        }
        if (mode == 5) {
            config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
            config.client_priorities.clear();
            config.transport.max_peers = glimmer::core::kMaxExecutionPriorityPeers + 1;
        }
        if (mode == 6)
            config.persistent.kind = static_cast<glimmer::transform::PtxExecutionKind>(7);
        if (mode == 7) {
            config.persistent.kind = glimmer::transform::PtxExecutionKind::kDynamic;
            config.persistent.replace_sgemm = true;
        }
        if (mode == 8) {
            config.client_priorities.clear();
            config.persistent.kind = glimmer::transform::PtxExecutionKind::kSliced;
            config.persistent.workers = 0;
        }
        if (mode == 9)
            config.client_priorities.front().kernel_mode =
                static_cast<ExecutionClientKernelMode>(255);
        if (mode >= 10) {
            config.client_priorities.front().kernel_mode = ExecutionClientKernelMode::kConfigured;
            if (mode == 10) config.uses_original_kernels = true;
            if (mode == 11) config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
            if (mode == 12)
                config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
            if (mode == 13) config.persistent.workers = 0;
            if (mode == 14)
                config.client_priorities.push_back(
                    {*peer, 100, ExecutionClientKernelMode::kOriginal});
        }
        if (mode == 15) config.priority_idle_us = 1000001;
        bool was_rejected{};
        try {
            ExecutionService service(std::move(config), kDispatch);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        expect(was_rejected,
               "invalid geometry, identity, scheduler or conflicting original/replacement modes "
               "reject");
    }
    expect(native.contexts.size() == contexts_before && native.events.size() == events_before &&
               ::access(path.c_str(), F_OK) != 0,
           "invalid trusted selections reject before native resources or socket publication");
}
}  // namespace
int main(int argc, char** argv) {
    char pattern[] = "/tmp/glimmer-exec-priority-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        completed_cleanup_eligibility();
        if (argc == 2 && std::string_view(argv[1]) == "--retired-retreat") {
            retired_epoch_retreat(std::string(directory) + "/retired-retreat.sock");
            expect(::rmdir(directory) == 0, "remove empty retired-retreat directory");
            std::cout << "Known-retired epoch retreat checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--intake-race") {
            priority_intake_race(std::string(directory) + "/metadata-intake.sock", true);
            priority_intake_race(std::string(directory) + "/gpu-intake.sock", false);
            expect(::rmdir(directory) == 0, "remove empty intake-race directory");
            std::cout << "Priority demand intake and metadata-negative checks passed\n";
            return EXIT_SUCCESS;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--priority-idle") {
            for (unsigned int scenario = 1; scenario <= 5; ++scenario)
                cleanup_idle_service(std::string(directory) + "/cleanup-idle.sock", scenario);
            priority_handoff(std::string(directory) + "/idle-window.sock", false, false, 1, false,
                             false, false, false, true);
            expect(::rmdir(directory) == 0, "remove empty idle-window fixture directory");
            std::cout << "Physical-drain priority idle window checks passed\n";
            return EXIT_SUCCESS;
        }
        immediate_sliced_batch_uses_exact_resume_phase(std::string(directory) +
                                                       "/exact-batch.sock");
        if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--workload-agnostic")) {
            for (const auto scenario :
                 {SharingCase::kQualified, SharingCase::kGlobalCap, SharingCase::kBelowFloor,
                  SharingCase::kIncompleteProfile, SharingCase::kFittingGeometry})
                workload_agnostic_execution(std::string(directory) + "/sharing.sock", scenario);
            invalid_sharing_configuration(std::string(directory) + "/sharing-invalid.sock");
            if (argc == 2) {
                expect(::rmdir(directory) == 0, "remove empty sharing fixture directory");
                std::cout << "Workload-agnostic selection and software overlap checks passed\n";
                return EXIT_SUCCESS;
            }
        }
        if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--client-kernels")) {
            for (const auto scenario :
                 {ClientSelectionCase::kConfiguredEqual, ClientSelectionCase::kConfiguredPriority,
                  ClientSelectionCase::kAutomaticEqual, ClientSelectionCase::kOriginalPriority,
                  ClientSelectionCase::kConfiguredEqualProfile})
                client_kernel_selection(std::string(directory) + "/client-kernels.sock", scenario);
            invalid_priority_configuration(std::string(directory) + "/client-kernels-invalid.sock");
            if (argc == 2) {
                expect(::rmdir(directory) == 0, "remove empty client-selection directory");
                std::cout << "Trusted client kernel selection checks passed\n";
                return EXIT_SUCCESS;
            }
        }
        for (const auto& mode :
             std::array<std::array<bool, 2>, 3>{{{true, true}, {false, true}, {false, false}}})
            unissued_low_rejection_audit(std::string(directory) + "/unissued-low.sock", mode[0],
                                         mode[1]);
        for (const auto failure : {CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_INVALID_CONTEXT})
            uncertain_preparation_rejection_audit(
                std::string(directory) + "/uncertain-preparation.sock", failure);
        if (argc == 2 && std::string_view(argv[1]) == "--unissued-audit") {
            expect(::rmdir(directory) == 0, "remove empty unissued-audit directory");
            std::cout << "Unissued low rejection audit checks passed\n";
            return EXIT_SUCCESS;
        }
        immediate_high_completion_audit(std::string(directory) + "/short-native-audit.sock", true,
                                        true);
        immediate_high_completion_audit(std::string(directory) + "/short-priority-audit.sock",
                                        false, true);
        immediate_high_completion_audit(std::string(directory) + "/short-audit-off.sock", false,
                                        false);
        immediate_high_completion_audit(std::string(directory) + "/failed-native-audit.sock", true,
                                        true, true);
        immediate_high_completion_audit(std::string(directory) + "/failed-priority-audit.sock",
                                        false, true, true);
        invalid_priority_configuration(std::string(directory) + "/invalid.sock");
        stale_priority_rejected_before_sdk(std::string(directory) + "/stale.sock");
        stale_priority_rejected_before_sdk(std::string(directory) + "/stale-naive.sock",
                                           glimmer::core::ExecutionSchedulingMode::kNaive);
        naive_and_original_execution(std::string(directory) + "/naive-original.sock", true);
        naive_and_original_execution(std::string(directory) + "/naive-unranked.sock", true, false);
        naive_and_original_execution(std::string(directory) + "/naive-preemptive.sock", true, true,
                                     false);
        naive_and_original_execution(std::string(directory) + "/priority-original.sock", false);
        profile_serializes_service_work(std::string(directory) + "/profile.sock");
        priority_handoff(std::string(directory) + "/priority.sock");
        priority_handoff(std::string(directory) + "/observed-priority.sock", false, false, 1, false,
                         false, false, true);
        priority_handoff(std::string(directory) + "/cached-priority.sock", false, false, 1, false,
                         false, true);
        priority_handoff(std::string(directory) + "/orphan.sock", true);
        priority_handoff(std::string(directory) + "/native-low.sock", false, true);
        priority_handoff(std::string(directory) + "/multi-low.sock", false, false, 2);
        priority_handoff(std::string(directory) + "/multi-orphan.sock", true, false, 2);
        priority_handoff(std::string(directory) + "/multi-native.sock", false, true, 2);
        priority_handoff(std::string(directory) + "/multi-sdk-error.sock", false, false, 2, true);
        priority_handoff(std::string(directory) + "/sliced.sock", false, false, 1, false, true);
        priority_handoff(std::string(directory) + "/sliced-orphan.sock", true, false, 1, false,
                         true);
        priority_handoff(std::string(directory) + "/sliced-multi.sock", false, false, 2, false,
                         true);
        std::cout << "Authenticated priority physical-stop and exact-resume checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
