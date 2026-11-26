#include "execution_nccl_service_fixture.h"

#include <charconv>
#include <cerrno>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <fcntl.h>
#include <sys/mman.h>

namespace {
using namespace glimmer::tests::nccl_service;
using glimmer::core::ExecutionSchedulingMode;
constexpr std::uint32_t kGuard = 0x11223344;
// Selected only before constructing the service or starting fixture workers.
bool audits_semantic{};
bool returns_async_outer_end{};
bool injects_host_function_event_failure{};
bool uses_async_cut_kernel{};
struct AsyncCutModule;
struct AsyncCutFunction {
    AsyncCutModule* module{};
    bool is_private{};
};
struct AsyncCutModule {
    CUcontext context{};
    bool is_live{};
    std::array<AsyncCutFunction, 2> functions{};
};
std::array<AsyncCutModule, 4> async_cut_modules;
std::atomic<unsigned> async_cut_preparations{}, async_cut_launches{}, async_cut_module_releases{};
std::atomic<bool> async_cut_end_requested{};
constexpr std::string_view kAsyncCutSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry kernel(.param .u64 output, .param .u32 value) {
 .reg .b32 %r<3>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output];
 ld.param.u32 %r1, [value];
 mov.u32 %r2, %ctaid.x;
 mul.wide.u32 %rd2, %r2, 4;
 add.u64 %rd3, %rd1, %rd2;
 st.global.u32 [%rd3], %r1;
 ret;
}
)ptx";
void install_async_cut_kernel(CudaSessionDispatch& sdk) {
    sdk.module_load = [](CUmodule* output, const void* image) {
        ++provider.driver_callbacks;
        expect(current && image, "async cut module requires actual selected source context");
        const auto free = std::ranges::find(async_cut_modules, false, &AsyncCutModule::is_live);
        expect(free != async_cut_modules.end(), "bounded scenario-owned native module arena");
        free->context = current;
        free->is_live = true;
        free->functions = {{{&*free, false}, {&*free, true}}};
        *output = reinterpret_cast<CUmodule>(&*free);
        return CUDA_SUCCESS;
    };
    sdk.module_unload = [](CUmodule native) {
        ++provider.driver_callbacks;
        auto* module = reinterpret_cast<AsyncCutModule*>(native);
        expect(module && module->is_live && module->context == current,
               "unload exact live async-cut source module once");
        module->is_live = false;
        ++async_cut_module_releases;
        return CUDA_SUCCESS;
    };
    sdk.function_get = [](CUfunction* output, CUmodule native, const char* name) {
        ++provider.driver_callbacks;
        auto* module = reinterpret_cast<AsyncCutModule*>(native);
        expect(module && module->is_live && module->context == current && name,
               "resolve only the exact selected source module");
        *output = reinterpret_cast<CUfunction>(
            &module->functions[std::string_view{name} == "kernel" ? 0 : 1]);
        return CUDA_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact cuFuncGetParamInfo callback ABI.
    sdk.parameter_info = [](CUfunction native, std::size_t index, std::size_t* offset,
                            std::size_t* bytes) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        ++provider.driver_callbacks;
        const auto* function = reinterpret_cast<AsyncCutFunction*>(native);
        expect(function && function->module->is_live && function->module->context == current,
               "ABI metadata stays on the exact live function owner");
        constexpr std::array<std::size_t, 7> offsets{0, 8, 12, 16, 20, 24, 32};
        constexpr std::array<std::size_t, 7> widths{8, 4, 4, 4, 4, 8, 8};
        if (index >= (function->is_private ? offsets.size() : 2)) return CUDA_ERROR_INVALID_VALUE;
        *offset = offsets[index];
        *bytes = widths[index];
        return CUDA_SUCCESS;
    };
    sdk.function_set_attribute = [](CUfunction native, CUfunction_attribute, int) {
        ++provider.driver_callbacks;
        const auto* function = reinterpret_cast<AsyncCutFunction*>(native);
        expect(function && function->module->is_live && function->module->context == current,
               "prepare private function metadata only on its exact source owner");
        return CUDA_SUCCESS;
    };
    sdk.occupancy = [](int* capacity, CUfunction native, int, std::size_t) {
        ++provider.driver_callbacks;
        const auto* function = reinterpret_cast<AsyncCutFunction*>(native);
        expect(function && function->is_private && function->module->is_live &&
                   function->module->context == current && provider.calls[6] == 1,
               "actual deferred preparation uses the exact source owner after GroupStart");
        // A real zero-capacity candidate falls back to the original function.
        // This exposes a native preparation callback without teaching this
        // source-order fixture to emulate the separate persistent counter ABI.
        *capacity = 0;
        const auto ordinal = ++async_cut_preparations;
        expect(
            ordinal <= 2 && !provider.calls[7] && async_cut_end_requested.load() == (ordinal == 2),
            "each kernel prepares only under its collective or newly requested End cut");
        return CUDA_SUCCESS;
    };
    // NOLINTBEGIN(bugprone-easily-swappable-parameters): exact cuLaunchKernel callback ABI.
    sdk.execution.launch_kernel = [](CUfunction native, unsigned x, unsigned y, unsigned z,
                                     unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                     CUstream handle, void** arguments, void** extra) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        ++provider.driver_callbacks;
        const auto* function = reinterpret_cast<AsyncCutFunction*>(native);
        expect(function && !function->is_private && function->module->is_live &&
                   function->module->context == current && x == 8 && y == 1 && z == 1 && bx == 1 &&
                   by == 1 && bz == 1 && !shared && !arguments && extra &&
                   *static_cast<const std::size_t*>(extra[3]) == 12,
               "issue only the exact original eight-element kernel and copied public ABI");
        CUdeviceptr output{};
        std::uint32_t value{};
        const auto* parameters = static_cast<const std::byte*>(extra[1]);
        std::memcpy(&output, parameters, sizeof(output));
        std::memcpy(&value, parameters + sizeof(output), sizeof(value));
        std::lock_guard lock(provider.mutex);
        const auto allocation = std::ranges::find_if(provider.allocations, [&](const auto& memory) {
            return memory.live && memory.context == current &&
                   reinterpret_cast<CUdeviceptr>(memory.values.data() + 4) == output;
        });
        expect(allocation != provider.allocations.end(),
               "kernel writes only eight interior values");
        auto& stream = provider.stream(handle, current);
        stream.commands.push_back({Command::Kind::kFill, 0, output, 32, 0, value});
        const auto ordinal = ++async_cut_launches;
        expect(
            ordinal <= 2 && !provider.calls[7] && async_cut_end_requested.load() == (ordinal == 2),
            "each exact source prefix issues before SDK entry under its own native cut");
        return CUDA_SUCCESS;
    };
}
std::atomic<bool> fails_next_private_event{};
std::atomic<unsigned> private_event_failures{};
CUresult host_function_event_create(CUevent* output, unsigned flags) {
    if (flags == CU_EVENT_DISABLE_TIMING && fails_next_private_event.exchange(false)) {
        ++provider.driver_callbacks;
        ++private_event_failures;
        *output = nullptr;
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    return session_dispatch().execution.event_create(output, flags);
}
ncclResult_t async_outer_end() {
    const auto result = end();
    if (result != ncclSuccess || group_depth) return result;
    provider.events_ready = false;
    return ncclInProgress;
}
struct ExpectedScalar {
    float value{};
};
enum class PeerAdmission { kParentOnly, kConfiguredParent, kAuthenticatedProcesses };
struct FixtureLimits {
    std::size_t foreign_edges{4096}, pending{2}, commands{32};
};
struct Caller {
    std::array<std::uint64_t, 2> thread{};
    std::uint64_t sequence{1}, group{}, depth{}, next_group{1}, context{}, primary_domain{};
    std::uint64_t ordinary_sequence{1};
};
struct Source {
    ExecutionSocketClient client;
    std::vector<std::uint64_t> allocations, streams, events;
    std::uint64_t next_sequence{1};
    Caller* caller{};
    ExecutionSocketSubmission submit(ExecutionOpcode operation,
                                     std::span<const std::byte> payload = {}) {
        if (!caller || !caller->depth) return client.submit(operation, payload);
        const ExecutionSourceContinuation scope{
            0, caller->thread, caller->group, caller->ordinary_sequence, client.session(), 0};
        const auto encoded = encode_execution_source_continuation(scope, operation, payload);
        expect(encoded.has_value(), "encode exact live-group ordinary request");
        const auto result = client.submit(ExecutionOpcode::kSourceContinuation, *encoded);
        if (result.status == ExecutionSocketSubmitStatus::kAccepted) ++caller->ordinary_sequence;
        return result;
    }
    ExecutionWireReply complete(std::optional<ExecutionWireReply> reply, const char* operation) {
        return success(std::move(reply), operation, client, next_sequence++);
    }
    std::optional<ExecutionWireReply> exchange(ExecutionOpcode operation,
                                               std::span<const std::byte> payload = {}) {
        const auto submitted = submit(operation, payload);
        if (submitted.status != ExecutionSocketSubmitStatus::kAccepted) return std::nullopt;
        expect(submitted.sequence == next_sequence,
               "source exchange preserves the exact submitted wire sequence");
        const auto deadline = Clock::now() + std::chrono::seconds{10};
        for (;;) {
            if (auto reply = client.take_reply(submitted.sequence)) return reply;
            if (Clock::now() >= deadline || !client.poll_for(1))
                return client.take_reply(submitted.sequence);
        }
    }
    Source(const std::string& path, TaskPeerIdentity peer, Caller* live_caller = nullptr)
        : client(client_config(path, peer)), caller(live_caller) {
        if (!caller || !caller->depth) {
            (void)complete(client.open(), "source.Hello");
            return;
        }
        const ExecutionSourceContinuation scope{
            0, caller->thread, caller->group, caller->ordinary_sequence, 0, 0};
        const auto opened = client.submit_open({}, scope);
        expect(opened.status == ExecutionSocketSubmitStatus::kAccepted && opened.sequence == 1,
               "accept exact cold source bootstrap without waiting for another native inner");
        ++caller->ordinary_sequence;
        std::optional<ExecutionWireReply> reply;
        await([&] {
            expect(client.poll_for(0), "poll exact wrapped cold source bootstrap");
            reply = client.take_reply(opened.sequence);
            return reply.has_value();
        });
        (void)complete(std::move(reply), "source.WrappedHello");
    }
    std::uint64_t allocate() {
        ExecutionPayloadWriter input;
        expect(input.write_u64(64), "encode complete owned allocation");
        const auto reply =
            complete(exchange(ExecutionOpcode::kAllocate, input.take()), "source.Allocate");
        expect(reply.id && reply.address, "actual service owns numeric allocation and identity");
        allocations.push_back(reply.id);
        return reply.id;
    }
    std::uint64_t stream(unsigned flags) {
        ExecutionPayloadWriter input;
        expect(input.write_u32(flags), "encode exact owned stream flags");
        const auto reply =
            complete(exchange(ExecutionOpcode::kCreateStream, input.take()), "source.CreateStream");
        expect(reply.id, "actual source stream has authenticated identity");
        streams.push_back(reply.id);
        return reply.id;
    }
    void upload(std::uint64_t allocation) {
        std::array<std::uint32_t, 16> initial;
        initial.fill(kGuard);
        ExecutionPayloadWriter input;
        expect(input.write_u64(allocation) && input.write_u64(0) &&
                   input.write_blob(std::as_bytes(std::span{initial})),
               "encode unchanged full source tensor");
        (void)complete(exchange(ExecutionOpcode::kUpload, input.take()), "source.Upload");
    }
    static std::vector<std::byte> fill_payload(std::uint64_t allocation, std::uint64_t stream,
                                               float value) {
        ExecutionPayloadWriter input;
        expect(input.write_u32(static_cast<std::uint32_t>(ExecutionMemoryKind::kFill32)) &&
                   input.write_u32(1) && input.write_u64(allocation) && input.write_u64(16) &&
                   input.write_u64(0) && input.write_u64(0) && input.write_u64(8) &&
                   input.write_u32(std::bit_cast<std::uint32_t>(value)) && input.write_u64(stream),
               "encode exact eight-element asynchronous source producer");
        return input.take();
    }
    void fill(std::uint64_t allocation, std::uint64_t stream, float value) {
        (void)complete(exchange(ExecutionOpcode::kMemory, fill_payload(allocation, stream, value)),
                       "source.Fill32");
    }
    ExecutionSocketSubmission submit_fill(std::uint64_t allocation, std::uint64_t stream,
                                          float value) {
        const auto submitted =
            submit(ExecutionOpcode::kMemory, fill_payload(allocation, stream, value));
        expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted &&
                   submitted.sequence == next_sequence++,
               "submit exact queued source producer without pretending it issued");
        return submitted;
    }
    ExecutionWireReply settle(const ExecutionSocketSubmission& submitted, const char* operation) {
        std::optional<ExecutionWireReply> reply;
        await([&] {
            expect(client.poll_for(0), "poll exact retained source command receipt");
            reply = client.take_reply(submitted.sequence);
            return reply.has_value();
        });
        return success(std::move(reply), operation, client, submitted.sequence);
    }
    void drain() {
        (void)complete(exchange(ExecutionOpcode::kDrain), "source.Drain");
    }
    std::uint64_t event() {
        ExecutionPayloadWriter input;
        expect(input.write_u32(kUserEventFlags), "encode distinct valid public event flags");
        const auto reply =
            complete(exchange(ExecutionOpcode::kCreateEvent, input.take()), "source.CreateEvent");
        expect(reply.id, "public event has exact authenticated source identity");
        events.push_back(reply.id);
        return reply.id;
    }
    void event_command(ExecutionOpcode operation, std::uint64_t event, std::uint64_t stream) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(event) && input.write_u64(stream) && input.write_u32(0),
               "encode exact local event command");
        const auto reply = complete(exchange(operation, input.take()), "source.Event");
        expect(reply.id && !reply.original_status,
               "event admission is not an invented requested native result");
    }
    ExecutionWireReply foreign_wait(std::uint64_t source, std::uint64_t event,
                                    std::uint64_t stream) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(source) && input.write_u64(event) && input.write_u64(stream) &&
                   input.write_u32(0),
               "encode exact authenticated foreign event generation capture");
        auto reply = exchange(ExecutionOpcode::kWaitForeignEvent, input.take());
        if (!reply) describe_failure(reply, "source.WaitForeignEvent", client, next_sequence);
        ++next_sequence;
        expect(reply.has_value(), "foreign event admission returns a bounded actual response");
        return std::move(*reply);
    }
    void destroy_event(std::uint64_t event) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(event), "encode exact event metadata deletion");
        (void)complete(exchange(ExecutionOpcode::kDestroyEvent, input.take()),
                       "source.DestroyEvent");
        const auto found = std::ranges::find(events, event);
        expect(found != events.end(), "delete only the owned live event metadata");
        events.erase(found);
    }
    void expect_outstanding(std::uint64_t expected) {
        const auto reply = complete(exchange(ExecutionOpcode::kSnapshot), "source.Snapshot");
        ExecutionPayloadReader input(reply.data);
        std::array<std::uint64_t, 7> fields{};
        for (auto& field : fields) expect(input.read_u64(field), "complete exact source snapshot");
        std::uint32_t quarantine{};
        expect(input.read_u32(quarantine) && input.empty() && !quarantine && fields[0] == expected,
               "actual source snapshot proves queued command admission without native issue");
    }
    void verify(std::uint64_t allocation, ExpectedScalar expected) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(allocation) && input.write_u64(0) && input.write_u64(64),
               "encode complete numerical and interior-guard readback");
        const auto reply =
            complete(exchange(ExecutionOpcode::kDownload, input.take()), "source.Download");
        expect(reply.data.size() == 64, "complete source tensor is returned");
        std::array<std::uint32_t, 16> observed{};
        std::memcpy(observed.data(), reply.data.data(), reply.data.size());
        for (std::size_t index = 0; index != observed.size(); ++index)
            expect(observed[index] == (index >= 4 && index < 12
                                           ? std::bit_cast<std::uint32_t>(expected.value)
                                           : kGuard),
                   "every numerical element and interior guard matches exact command order");
    }
    void close() {
        while (!events.empty()) destroy_event(events.back());
        for (const auto allocation : allocations) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(allocation), "encode exact allocation retirement");
            (void)complete(exchange(ExecutionOpcode::kRelease, input.take()), "source.Release");
        }
        for (const auto stream : streams) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(stream), "encode exact stream retirement");
            (void)complete(exchange(ExecutionOpcode::kDestroyStream, input.take()),
                           "source.DestroyStream");
        }
        (void)complete(exchange(ExecutionOpcode::kClose), "source.Close");
    }
};
struct HostPage {
    ExecutionOwnedDescriptor descriptor;
    void* mapping{MAP_FAILED};
    std::size_t bytes{};
    std::array<std::uint32_t, 16> expected{};
    HostPage() : descriptor(::memfd_create("glimmer-nccl-host", MFD_ALLOW_SEALING | MFD_CLOEXEC)) {
        const auto page = ::sysconf(_SC_PAGESIZE);
        expect(page > 64 && descriptor.get() >= 0, "create real bounded host-transfer memfd");
        bytes = static_cast<std::size_t>(page);
        expect(::ftruncate(descriptor.get(), static_cast<off_t>(bytes)) == 0 &&
                   ::fcntl(descriptor.get(), F_ADD_SEALS,
                           F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
               "seal the exact memfd extent without preventing shared numerical writes");
        mapping = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor.get(), 0);
        expect(mapping != MAP_FAILED, "map the independent caller-side page");
        std::memset(mapping, 0x5a, bytes);
        for (std::size_t index = 0; index != expected.size(); ++index)
            expected[index] = std::bit_cast<std::uint32_t>(static_cast<float>(index) + 0.25F);
        std::memcpy(mapping, expected.data(), 64);
    }
    ~HostPage() {
        if (mapping != MAP_FAILED)
            expect(::munmap(mapping, bytes) == 0, "release exact client memfd mapping once");
    }
    HostPage(const HostPage&) = delete;
    HostPage& operator=(const HostPage&) = delete;
    void verify() const {
        std::array<std::uint32_t, 16> observed{};
        std::memcpy(observed.data(), mapping, 64);
        expect(observed == expected,
               "all sixteen host-transfer values match the independent oracle");
        const auto page = std::span{static_cast<const std::byte*>(mapping), bytes};
        expect(std::ranges::all_of(page.subspan(64),
                                   [](auto value) { return value == std::byte{0x5a}; }),
               "every unused memfd page byte remains an unchanged guard");
    }
};
struct HostHelper {
    ExecutionSocketClient client;
    Caller& caller;
    std::uint64_t origin{}, allocation{}, cleanup_sequence{1};
    bool is_staging{};
    std::optional<ExecutionSourceContinuation> parent;
    HostHelper(const std::string& path, TaskPeerIdentity peer, Caller& owner, std::uint64_t source,
               bool staging)
        : client(client_config(path, peer)), caller(owner), origin(source), is_staging(staging) {
        ExecutionPayloadWriter input;
        expect(input.write_u32(staging ? kExecutionStagingOwnerRole : kExecutionHostOwnerRole) &&
                   input.write_u32(CU_CTX_MAP_HOST),
               "encode exact supported helper bootstrap");
        std::optional<ExecutionSourceContinuation> scope;
        if (caller.depth)
            scope = ExecutionSourceContinuation{
                0, caller.thread, caller.group, caller.ordinary_sequence, origin, 0};
        const auto accepted = client.submit_open(input.take(), scope);
        expect(accepted.status == ExecutionSocketSubmitStatus::kAccepted,
               "submit actual authenticated helper bootstrap");
        if (scope) ++caller.ordinary_sequence;
        const auto reply = settle(accepted, "helper.Hello");
        expect(reply.original_status == CUDA_SUCCESS &&
                   decode_execution_bootstrap_ownership(reply.data) ==
                       ExecutionBootstrapOwnership::kRetained,
               "real helper owns its created context, not a fabricated bootstrap response");
    }
    ExecutionWireReply settle(const ExecutionSocketSubmission& request, const char* label) {
        std::optional<ExecutionWireReply> reply;
        await([&] {
            expect(client.poll_for(0), "poll exact helper request on healthy transport");
            reply = client.take_reply(request.sequence);
            return reply.has_value();
        });
        return success(std::move(reply), label, client, request.sequence);
    }
    ExecutionSocketSubmission submit(ExecutionOpcode operation, std::span<const std::byte> payload,
                                     int descriptor = -1, bool is_cleanup = false) {
        std::optional<ExecutionSourceContinuation> scope;
        if (is_cleanup) {
            expect(parent.has_value(),
                   "only an actual wrapped import has cleanup-parent authority");
            scope = *parent;
            scope->flags = kExecutionSourceContinuationCleanup;
            scope->cleanup_parent = parent->ordinary_sequence;
            scope->ordinary_sequence = cleanup_sequence;
        } else if (caller.depth) {
            scope = ExecutionSourceContinuation{
                0, caller.thread, caller.group, caller.ordinary_sequence, origin, 0};
        }
        auto request = ExecutionSocketSubmission{};
        if (scope) {
            const auto encoded = encode_execution_source_continuation(*scope, operation, payload);
            expect(encoded.has_value(),
                   "encode full helper request and independent cleanup ordinal");
            request = client.submit(ExecutionOpcode::kSourceContinuation, *encoded, descriptor);
        } else {
            request = client.submit(operation, payload, descriptor);
        }
        expect(request.status == ExecutionSocketSubmitStatus::kAccepted,
               "submit helper through real descriptor-carrying transport");
        if (scope) {
            if (is_cleanup)
                ++cleanup_sequence;
            else {
                ++caller.ordinary_sequence;
                if (operation == ExecutionOpcode::kImportHostMemory) parent = scope;
            }
        }
        return request;
    }
    void import(HostPage& page) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(64) && input.write_u64(page.bytes) && input.write_u32(0),
               "import exactly 64 logical bytes within the real sealed page");
        const auto request =
            submit(ExecutionOpcode::kImportHostMemory, input.take(), page.descriptor.get());
        const auto reply = settle(request, "helper.Import");
        expect(reply.id && reply.address && reply.original_status == CUDA_SUCCESS,
               "actual successful native registration publishes one owned imported allocation");
        allocation = reply.id;
        native_address = reply.address;
    }
    ExecutionSocketSubmission release(bool cleanup) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(allocation), "release exact imported allocation");
        return submit(ExecutionOpcode::kReleaseHostMemory, input.take(), -1, cleanup);
    }
    void close() {
        const auto reply = settle(submit(ExecutionOpcode::kClose, {}), "helper.Close");
        expect(!reply.original_status,
               "aggregate helper shutdown fabricates no requested SDK status");
    }
    std::vector<std::byte> transfer_payload(std::uint64_t device, std::uint64_t stream, bool upload,
                                            std::uint64_t count = 64) const {
        ExecutionPayloadWriter input;
        expect(input.write_u64(client.session()) && input.write_u64(allocation) &&
                   input.write_u64(0) && input.write_u64(device) && input.write_u64(0) &&
                   input.write_u64(count) && input.write_u64(stream) &&
                   input.write_u32(upload ? 1 : 0),
               "encode exact full host transfer and independently authenticated helper owner");
        return input.take();
    }
    ExecutionOpcode transfer_opcode() const {
        return is_staging ? ExecutionOpcode::kStagedHostTransfer : ExecutionOpcode::kHostTransfer;
    }
    CUcontext native_context() const {
        std::lock_guard lock(provider.mutex);
        const auto found = std::ranges::find_if(provider.registrations, [&](const auto& value) {
            return value.live && reinterpret_cast<CUdeviceptr>(value.pointer) == native_address;
        });
        expect(found != provider.registrations.end(),
               "identify only the exact observed host mapping");
        return found->context;
    }
    CUdeviceptr native_address{};
};
void verify_full(Source& source, std::uint64_t allocation,
                 const std::array<std::uint32_t, 16>& expected) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(allocation) && input.write_u64(0) && input.write_u64(64),
           "read back every byte of the host-transfer device tensor");
    const auto reply = source.complete(source.exchange(ExecutionOpcode::kDownload, input.take()),
                                       "source.FullHostTransferReadback");
    std::array<std::uint32_t, 16> observed{};
    expect(reply.data.size() == 64, "full host-transfer readback has exact extent");
    std::memcpy(observed.data(), reply.data.data(), 64);
    expect(observed == expected, "all sixteen device values match the independent host oracle");
}
struct NativeReply {
    ExecutionWireReply outer;
    ExecutionNcclReply envelope;
};
struct Endpoint {
    ExecutionSocketClient client;
    Endpoint(const std::string& path, TaskPeerIdentity peer) : client(client_config(path, peer)) {
        ExecutionPayloadWriter input;
        expect(input.write_u32(kExecutionNcclRole) && input.write_u32(0),
               "encode real NCL1 bootstrap");
        (void)success(client.open(input.take()), "NCL1.Hello", client, 1);
    }
    std::vector<std::byte> request(Caller& caller, ExecutionNcclCall call) {
        const bool starts = std::holds_alternative<ExecutionNcclGroupStart>(call);
        if (starts && !caller.depth) caller.group = caller.next_group++;
        ExecutionNcclRequest request{
            {caller.thread, caller.sequence++, caller.group, caller.context}, std::move(call)};
        ExecutionPayloadWriter input;
        expect(
            input.write_u64(caller.primary_domain) && write_execution_nccl_request(input, request),
            "encode explicit primary provenance and exact authenticated full scope");
        return input.take();
    }
    NativeReply decode(std::optional<ExecutionWireReply> response) {
        if (!response) describe_failure(response, "NCL1.Call", client);
        expect(response.has_value(), "actual runtime returns an authenticated NCCL reply");
        NativeReply result{std::move(*response), {}};
        ExecutionPayloadReader input(result.outer.data);
        expect(read_execution_nccl_reply(input, result.envelope) && input.empty() &&
                   !result.outer.address && result.outer.arguments.empty() &&
                   result.outer.parameter_bytes <= 1,
               "runtime emits complete canonical NCR1 envelope, not CPU metadata success");
        expect(result.envelope.original_status == result.outer.original_status,
               "carrier and actual requested native status remain separate and correlated");
        if (result.envelope.is_accepted)
            expect(result.outer.id == result.envelope.operation && result.outer.id,
                   "accepted actor identity matches exact outer receipt");
        return result;
    }
    NativeReply call(Caller& caller, ExecutionNcclCall value, int expected = 0) {
        const bool starts = std::holds_alternative<ExecutionNcclGroupStart>(value);
        const bool ends = std::holds_alternative<ExecutionNcclGroupEnd>(value);
        const auto variant = value.index() + 1;
        const auto public_sequence = caller.sequence;
        const auto payload = request(caller, std::move(value));
        const auto describe_call = [&] {
            std::cerr << "NCCL public call failure: variant=" << variant
                      << " public_sequence=" << public_sequence << " group=" << caller.group
                      << " depth=" << caller.depth << '\n';
        };
        auto response = client.exchange(ExecutionOpcode::kNcclCall, payload);
        if (!response) describe_call();
        if (response &&
            (response->disposition != ExecutionWireDisposition::kComplete || response->status ||
             response->parameter_bytes || response->original_status != expected)) {
            describe_call();
            describe_failure(response, "NCL1.Call", client);
        }
        auto result = decode(std::move(response));
        const bool complete = result.outer.disposition == ExecutionWireDisposition::kComplete &&
                              !result.outer.status && !result.outer.parameter_bytes &&
                              result.envelope.is_accepted &&
                              result.envelope.original_status == expected;
        if (!complete) describe_call();
        expect(complete, "actual eleven-route native invocation preserves exact requested result");
        if (starts && expected == 0) ++caller.depth;
        if (ends && (expected == 0 || expected == 7) && caller.depth && !--caller.depth)
            caller.group = 0;
        if (result.envelope.context_effect.kind == ExecutionNcclContextEffectKind::kSelectPrimary) {
            expect(result.envelope.context_effect.device == 0 &&
                       result.envelope.context_effect.domain == 1,
                   "native observed primary is server-owned device/domain, not source session");
            caller.primary_domain = result.envelope.context_effect.domain;
            caller.context = 0;
        }
        return result;
    }
    ExecutionNcclUniqueId uid(Caller& caller) {
        const auto result = call(caller, ExecutionNcclGetUniqueId{true});
        expect(result.envelope.observation.effects == 1 &&
                   std::ranges::all_of(result.envelope.observation.unique_id_mask,
                                       [](auto value) { return value == std::byte{255}; }) &&
                   std::ranges::all_of(result.envelope.observation.unique_id,
                                       [](auto value) { return value == std::byte{0x6d}; }),
               "real native COW observation projects every UID byte");
        return result.envelope.observation.unique_id;
    }
    ExecutionNcclCommunicator initialize(Caller& caller, const ExecutionNcclUniqueId& uid,
                                         bool configured = false) {
        ExecutionNcclInitRank init{true, 1, 0, uid};
        ExecutionNcclCall operation = configured
                                          ? ExecutionNcclCall{ExecutionNcclInitRankConfig{init, {}}}
                                          : ExecutionNcclCall{init};
        const auto result = call(caller, std::move(operation));
        expect(
            result.envelope.observation.effects == 2 &&
                result.envelope.observation.communicator == result.envelope.reserved_communicator &&
                result.envelope.reserved_communicator.domain == 1 &&
                result.envelope.reserved_communicator.object,
            "actual synchronous pinned init write returns an owned symbolic communicator");
        return result.envelope.reserved_communicator;
    }
    void close() {
        (void)success(client.exchange(ExecutionOpcode::kClose), "NCL1.Close", client);
    }
    NativeReply rejected(Caller& caller, ExecutionNcclCall call) {
        auto reply =
            decode(client.exchange(ExecutionOpcode::kNcclCall, request(caller, std::move(call))));
        expect(reply.outer.disposition == ExecutionWireDisposition::kRejected &&
                   reply.outer.status && !reply.outer.id && !reply.outer.original_status &&
                   !reply.outer.parameter_bytes && !reply.envelope.is_accepted,
               "clean ownership/capacity rejection invokes no original and fabricates no native "
               "result");
        return reply;
    }
};
ExecutionNcclAllGather collective(ExecutionNcclCommunicator communicator, const Source& source,
                                  std::uint64_t send, std::uint64_t receive, std::uint64_t stream) {
    return {communicator,
            {ExecutionNcclStreamKind::kOwned, source.client.session(), stream},
            {ExecutionNcclBufferKind::kOwned, source.client.session(), send, 16},
            {ExecutionNcclBufferKind::kOwned, source.client.session(), receive, 16},
            8,
            static_cast<std::int32_t>(ncclFloat32)};
}
template <typename Client, typename Observe>
void run(Client client, Observe observe,
         ExecutionSchedulingMode mode = ExecutionSchedulingMode::kPriority,
         bool retains_uncertainty = false, PeerAdmission admission = PeerAdmission::kParentOnly,
         FixtureLimits limits = {}, std::vector<ExecutionClientPriority> extra_priorities = {}) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "read real authenticated service identity");
    const auto path = "/tmp/glimmer-nccl-service-" + std::to_string(::getpid()) + ".sock";
    provider.owner = std::this_thread::get_id();
    provider.maximum_pending_requests = limits.pending;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_peers = 8;
    config.transport.max_pending_requests = limits.pending;
    config.transport.io_timeout_ms = 10000;
    config.session_limits.max_objects = 64;
    config.session_limits.submissions.max_commands = limits.commands;
    config.exit_when_idle = true;
    config.uses_nccl = true;
    config.uses_original_kernels = true;
    config.maximum_foreign_event_edges = limits.foreign_edges;
    config.scheduling_mode = mode;
    config.should_audit_native = audits_semantic;
    if (mode == ExecutionSchedulingMode::kProfile)
        config.profile.kinds.clear();
    else if (admission == PeerAdmission::kParentOnly)
        config.client_priorities = {{*identity, 7, ExecutionClientKernelMode::kOriginal}};
    else if (admission == PeerAdmission::kConfiguredParent) {
        config.uses_original_kernels = false;
        config.persistent.workers = 320;
        config.client_priorities = {{*identity, 7, ExecutionClientKernelMode::kConfigured}};
    }
    config.client_priorities.insert(config.client_priorities.end(), extra_priorities.begin(),
                                    extra_priorities.end());
    config.should_audit_priority = audits_semantic && !config.client_priorities.empty();
    auto native = execution_dispatch();
    if (returns_async_outer_end) native.native.group_end = async_outer_end;
    auto driver = session_dispatch();
    if (uses_async_cut_kernel) install_async_cut_kernel(driver);
    if (injects_host_function_event_failure)
        driver.execution.event_create = host_function_event_create;
    ExecutionService service(std::move(config), driver, native);
    expect(service.start(), "start actual CUDA execution runtime with NCL1 actor and Core policy");
    auto future = std::async(std::launch::async, [=] { client(path, *identity); });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (future.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && service.step(1),
               "bounded actual service/Core/actor progress");
        observe(service);
    }
    future.get();
    if (retains_uncertainty) {
        for (unsigned step = 0; step != 32; ++step) {
            if (!service.step(0)) break;
            observe(service);
        }
        expect(!service.request_stop() && !service.finished() && !provider.primary_releases &&
                   !provider.calls[4] && !provider.calls[5] && !provider.calls[7],
               "disconnect/unknown ownership remains retained without fabricated End/Abort/Destroy "
               "or primary release");
        return;
    }
    await([&] {
        if (service.finished()) return true;
        expect(service.step(0), "complete runtime cleanup progress");
        return service.finished();
    });
    expect(service.request_stop(),
           "stop once after actual idle cleanup; never step a stopped service");
    expect(provider.primary_retains == 1 && provider.primary_releases == 1,
           "actual primary owner has one retain and one proven release");
    for (const auto& progress : provider.progress) progress.verify_cleanup();
    if (provider.has_host_registration)
        expect(provider.host_registers == provider.host_unregisters &&
                   provider.host_registers == provider.host_maps &&
                   provider.host_registers == provider.host_probes &&
                   std::ranges::all_of(provider.registrations,
                                       [](const auto& registration) {
                                           return !registration.live &&
                                                  !registration.pending_copies;
                                       }),
               "every imported mapping has exact probe/register/map/unregister and stopped copies");
    if (mode == ExecutionSchedulingMode::kProfile)
        expect(provider.platform_uuid_queries && provider.platform_driver_queries,
               "profile source initialization observed deterministic nonzero fake platform "
               "identity through the typed Driver callbacks");
    const bool has_cleaned =
        std::ranges::none_of(provider.events, &Event::live) &&
        std::ranges::none_of(provider.communicators, &Comm::live) &&
        std::ranges::none_of(provider.streams, &Stream::live) &&
        std::ranges::none_of(provider.legacy, &Stream::live) &&
        std::ranges::none_of(provider.allocations, &Allocation::live) &&
        std::ranges::none_of(provider.registrations, &RegisteredHost::live) &&
        std::ranges::none_of(provider.live_contexts, [](bool live) { return live; });
    if (!has_cleaned)
        std::cerr << "NCCL service cleanup failure: events="
                  << std::ranges::count_if(provider.events, &Event::live)
                  << " streams=" << std::ranges::count_if(provider.streams, &Stream::live)
                  << " legacy=" << std::ranges::count_if(provider.legacy, &Stream::live)
                  << " allocations="
                  << std::ranges::count_if(provider.allocations, &Allocation::live)
                  << " communicators=" << std::ranges::count_if(provider.communicators, &Comm::live)
                  << " contexts=" << std::ranges::count(provider.live_contexts, true) << '\n';
    expect(
        has_cleaned,
        "no private native event, stream, allocation or communicator survives successful cleanup");
}
void ordinary() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto send = source.allocate(), later_send = source.allocate(),
                       first = source.allocate(), second = source.allocate();
            for (const auto allocation : {send, later_send, first, second})
                source.upload(allocation);
            const auto stream = source.stream(CU_STREAM_DEFAULT);
            Endpoint endpoint(path, peer);
            Caller caller{{37, 41}};
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto direct = endpoint.initialize(caller, uid);
            const auto configured = endpoint.initialize(caller, uid, true);
            source.fill(send, stream, 2.5F);
            endpoint.call(caller, collective(direct, source, send, first, stream));
            source.verify(first, {2.5F});
            endpoint.call(caller, ExecutionNcclGroupStart{});
            endpoint.call(caller, ExecutionNcclGroupStart{});
            const auto grouped = endpoint.initialize(caller, uid);
            expect(caller.depth == 2 && grouped.object,
                   "nested initialization exposes its independently proved immediate host output");
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            expect(!caller.depth && !caller.group,
                   "initialization-only group closes before any communication group begins");
            endpoint.call(caller, ExecutionNcclGroupStart{});
            endpoint.call(caller, ExecutionNcclGroupStart{});
            source.fill(send, 0, 3.75F);
            endpoint.call(caller, collective(grouped, source, send, first, stream));
            source.fill(later_send, stream, 6.25F);
            endpoint.call(caller, ExecutionNcclAllReduce{
                                      collective(grouped, source, later_send, second, stream),
                                      static_cast<std::int32_t>(ncclSum)});
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            if (provider.should_defer_group_publication.load())
                expect(provider.deferred_group_commands == 2 && !provider.published_group_commands,
                       "nested End cannot publish the provider's deferred outer-group work");
            else
                expect(provider.eager_group_commands == 2,
                       "eager provider publishes every actual inner before the outer End");
            endpoint.call(
                caller, ExecutionNcclBcast{
                            grouped,
                            {ExecutionNcclStreamKind::kOwned, source.client.session(), stream},
                            {ExecutionNcclBufferKind::kOwned, source.client.session(), second, 16},
                            8,
                            static_cast<std::int32_t>(ncclFloat32),
                            0});
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            source.verify(send, {3.75F});
            source.verify(later_send, {6.25F});
            source.verify(first, {3.75F});
            source.verify(second, {6.25F});
            const auto queried = endpoint.call(caller, ExecutionNcclGetAsyncError{grouped, true});
            expect(queried.envelope.observation.effects == 4 &&
                       queried.envelope.observation.async_error_mask == 15 &&
                       queried.envelope.observation.async_error == 0,
                   "actual original async query emits full independently proved scalar");
            Caller other{{43, 47}};
            (void)endpoint.uid(other);
            endpoint.call(caller, ExecutionNcclDestroy{direct});
            endpoint.call(caller, ExecutionNcclAbort{configured});
            endpoint.call(caller, ExecutionNcclDestroy{grouped});
            endpoint.close();
            source.close();
        },
        [](ExecutionService&) {});
    expect(
        std::ranges::all_of(provider.calls, [](const auto& count) { return count.load() != 0; }) &&
            provider.waits,
        "all eleven actual native callbacks and real source dependency waits executed");
    std::vector<std::thread::id> lanes;
    for (const auto& entry : provider.ledger)
        if (std::ranges::find(lanes, entry.second) == lanes.end()) lanes.push_back(entry.second);
    expect(lanes.size() == 2, "two nonrecycled wire callers own two actual stable native lanes");
}
void ordinary_deferred() {
    provider.should_defer_group_publication = true;
    ordinary();
    expect(provider.deferred_group_commands == 3 && provider.published_group_commands == 3 &&
               !provider.eager_group_commands,
           "all three deferred native inners publish once at the actual outer End");
}
void high_client(const std::string& path, TaskPeerIdentity service, int signal_descriptor) {
    unsigned char signal{};
    ssize_t bytes{};
    do {
        bytes = ::read(signal_descriptor, &signal, sizeof(signal));
    } while (bytes < 0 && errno == EINTR);
    expect(bytes == 1 && signal == 1 && ::close(signal_descriptor) == 0,
           "high process starts only after actual native GroupStart returned");
    Source source(path, service);
    const auto allocation = source.allocate();
    source.upload(allocation);
    const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
    source.fill(allocation, stream, 17.5F);
    source.drain();
    source.verify(allocation, {17.5F});
    source.close();
}
void ordinary_high_arrival_impl(bool uses_host_transfer) {
    provider.has_host_registration = uses_host_transfer;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticated owner identity before spawning high peer");
    const auto path = "/tmp/glimmer-nccl-service-" + std::to_string(::getpid()) + ".sock";
    std::array<char, 4096> executable_bytes{};
    const auto executable_size =
        ::readlink("/proc/self/exe", executable_bytes.data(), executable_bytes.size());
    expect(
        executable_size > 0 && static_cast<std::size_t>(executable_size) < executable_bytes.size(),
        "resolve actual unchanged fixture executable before fresh exec");
    const std::string executable{executable_bytes.data(),
                                 static_cast<std::size_t>(executable_size)};
    int descriptors[2]{};
    expect(::pipe2(descriptors, O_CLOEXEC) == 0, "bounded owned high-peer start pipe");
    ExecutionOwnedDescriptor signal_read(descriptors[0]), signal_write(descriptors[1]);
    const std::array<std::string, 3> arguments{std::to_string(identity->pid),
                                               std::to_string(identity->start_time_ticks),
                                               std::to_string(signal_read.get())};
    const auto child = ::fork();
    expect(child >= 0, "spawn a distinct real high-priority authenticated process");
    if (!child) {
        if (::close(signal_write.get()) != 0 || ::fcntl(signal_read.get(), F_SETFD, 0) != 0)
            ::_exit(126);
        ::execl(executable.c_str(), executable.c_str(), "--high", path.c_str(),
                arguments[0].c_str(), arguments[1].c_str(), arguments[2].c_str(),
                static_cast<char*>(nullptr));
        ::_exit(127);
    }
    signal_read = ExecutionOwnedDescriptor{};
    TaskPeerIdentity high{static_cast<std::uint32_t>(child),
                          static_cast<std::uint32_t>(::geteuid()), 0};
    expect(read_process_start_time(high.pid, &high.start_time_ticks),
           "bind immutable high priority to actual PID and process incarnation");
    std::atomic<bool> pressure{}, group_is_open{};
    unsigned pressure_observations{};
    run(
        [&](const std::string& socket_path, TaskPeerIdentity peer) {
            Endpoint endpoint(socket_path, peer);
            Caller caller{{229, 233}};
            const auto uid = endpoint.uid(caller);
            const auto communicator = endpoint.initialize(caller, uid);
            endpoint.call(caller, ExecutionNcclGroupStart{});
            group_is_open = true;
            const unsigned char signal = 1;
            expect(::write(signal_write.get(), &signal, sizeof(signal)) == 1,
                   "release high peer after returned Start and before any inner collective");
            await([&] { return pressure.load(); });
            if (uses_host_transfer) {
                HostPage page;
                HostHelper helper(socket_path, peer, caller, 0, false);
                helper.import(page);
                Source source(socket_path, peer, &caller);
                const auto send = source.allocate(), receive = source.allocate();
                source.upload(send);
                source.upload(receive);
                const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
                caller.context = source.client.session();
                provider.hold_next_host_copy = true;
                const auto copied =
                    source.complete(source.exchange(helper.transfer_opcode(),
                                                    helper.transfer_payload(send, stream, true)),
                                    "source.HeldHostUpload");
                expect(copied.original_status == CUDA_SUCCESS && provider.host_uploads == 1 &&
                           provider.held_event_stream && !provider.host_unregisters,
                       "HST1 host return does not retire its real accepted mapping lease");
                provider.observed_release_context = helper.native_context();
                const auto release = helper.release(true);
                await([&] {
                    expect(helper.client.poll_for(0), "drive exact background cleanup request");
                    expect(!helper.client.take_reply(release.sequence),
                           "held imported mapping cannot publish a successful release");
                    return provider.release_context_proofs.load() >= 2;
                });
                // Only this release can query the helper context now. Its
                // second worker entry proves the first completed Deferred was
                // consumed, rather than a request merely waiting at a fence.
                expect(!provider.host_unregisters && provider.host_registers == 1 &&
                           provider.host_maps == 1 && provider.host_probes == 1,
                       "origin-zero cleanup really retries no-entry Deferred without an SDK free");
                endpoint.call(caller, collective(communicator, source, send, receive, stream));
                group_is_open = false;
                endpoint.call(caller, ExecutionNcclGroupEnd{});
                expect(!provider.host_unregisters && !helper.client.take_reply(release.sequence),
                       "End and ordinary ACK preserve the independent cleanup parent and lease");
                provider.observed_release_context = nullptr;
                provider.held_event_stream = nullptr;
                const auto freed = helper.settle(release, "helper.CrossEndCleanup");
                expect(freed.original_status == CUDA_SUCCESS && provider.host_unregisters == 1,
                       "fresh cleanup permission unregisters once after the actual source tail");
                int status{};
                expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                           !WEXITSTATUS(status),
                       "authenticated higher-priority work completes after the real low tails");
                verify_full(source, send, page.expected);
                auto gathered = page.expected;
                for (std::size_t index = 0; index != gathered.size(); ++index)
                    if (index < 4 || index >= 12) gathered[index] = kGuard;
                verify_full(source, receive, gathered);
                page.verify();
                endpoint.call(caller, ExecutionNcclDestroy{communicator});
                endpoint.close();
                source.close();
                helper.close();
                return;
            }
            {
                // Host registration is absent. This canonical no-entry policy
                // error still consumes its exact ordinary ordinal; an invalid
                // nested native carrier is separately rejected by the codec.
                ExecutionSocketClient helper(client_config(socket_path, peer));
                ExecutionPayloadWriter bootstrap;
                expect(bootstrap.write_u32(kExecutionHostOwnerRole) && bootstrap.write_u32(0),
                       "encode unsupported helper bootstrap without native entry");
                const ExecutionSourceContinuation scope{
                    0, caller.thread, caller.group, caller.ordinary_sequence, 0, 0};
                const auto submitted = helper.submit_open(bootstrap.take(), scope);
                expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
                       "canonical wrapped helper Hello is accepted for authenticated validation");
                ++caller.ordinary_sequence;
                std::optional<ExecutionWireReply> reply;
                await([&] {
                    expect(helper.poll_for(0), "poll exact unsupported helper bootstrap result");
                    reply = helper.take_reply(submitted.sequence);
                    return reply.has_value();
                });
                expect(reply->disposition == ExecutionWireDisposition::kRejected &&
                           reply->status == CUDA_ERROR_NOT_SUPPORTED &&
                           decode_execution_bootstrap_ownership(reply->data) ==
                               ExecutionBootstrapOwnership::kNone,
                       "early bootstrap policy rejection preserves dense ordinal and BOO1None");
            }
            // The higher-priority Hello is really received, not a sleeping
            // client or an inferred pending request. Each following call must
            // use the existing batch's exact progress authority.
            Source source(socket_path, peer, &caller);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            caller.context = source.client.session();
            source.fill(send, stream, 23.25F);
            const auto event = source.event();
            source.event_command(ExecutionOpcode::kRecordEvent, event, stream);
            source.event_command(ExecutionOpcode::kStreamWaitEvent, event, stream);
            source.drain();
            endpoint.call(caller, collective(communicator, source, send, receive, stream));
            // This source was never borrowed by NCCL; exact Close must not be
            // rejected merely because receive already sealed its own slot.
            Source closing(socket_path, peer, &caller);
            closing.close();
            expect(pressure.load(), "ordinary progress does not consume unrelated high demand");
            group_is_open = false;
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "high process runs after actual End/tails and exits normally without signals");
            source.verify(send, {23.25F});
            source.verify(receive, {23.25F});
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
            source.close();
        },
        [&](ExecutionService& service) {
            if (!group_is_open.load()) return;
            const auto pending = service.pending_priority_requests(9);
            if (pending) {
                pressure = true;
                ++pressure_observations;
            } else {
                expect(!pressure.load(),
                       "unrelated high bootstrap cannot enter the retained opaque open group");
            }
        },
        ExecutionSchedulingMode::kPriority, false, PeerAdmission::kParentOnly, {},
        {{high, 9, ExecutionClientKernelMode::kOriginal}});
    expect(pressure_observations && provider.calls[6] == 1 && provider.calls[7] == 1 &&
               provider.calls[8] == 1,
           "actual high-pressure group makes once-only ordinary and native progress");
}
void ordinary_high_arrival() {
    ordinary_high_arrival_impl(false);
}
void host_high_arrival() {
    ordinary_high_arrival_impl(true);
}
void staging_host_continuations() {
    provider.has_host_registration = true;
    std::atomic<unsigned> fenced_steps{}, phase{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{239, 241}};
            const auto communicator = endpoint.initialize(caller, endpoint.uid(caller));
            endpoint.call(caller, ExecutionNcclGroupStart{});
            Source source(path, peer, &caller);
            const auto allocation = source.allocate();
            source.upload(allocation);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            HostPage upload;
            HostHelper staging(path, peer, caller, source.client.session(), true);
            staging.import(upload);
            const auto reject_transfer = [&](ExecutionOpcode opcode, std::uint64_t count) {
                const auto copies = provider.host_uploads.load() + provider.host_downloads.load();
                const auto reply = source.exchange(
                    opcode, staging.transfer_payload(allocation, stream, true, count));
                ++source.next_sequence;
                expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
                           reply->status == CUDA_ERROR_INVALID_VALUE && !reply->original_status &&
                           provider.host_uploads.load() + provider.host_downloads.load() == copies,
                       "wrong helper role or 65-byte range rejects before any native transfer");
            };
            reject_transfer(ExecutionOpcode::kHostTransfer, 64);
            reject_transfer(ExecutionOpcode::kStagedHostTransfer, 65);
            provider.hold_next_host_copy = true;
            const auto copied =
                source.complete(source.exchange(staging.transfer_opcode(),
                                                staging.transfer_payload(allocation, stream, true)),
                                "source.StagingUpload");
            expect(copied.original_status == CUDA_SUCCESS && provider.host_uploads == 1,
                   "staged upload host return preserves the original once-only call");
            provider.observed_release_context = staging.native_context();
            const auto release = staging.release(true);
            phase = 1;
            await([&] {
                expect(staging.client.poll_for(0), "deliver source-bound cleanup over real socket");
                expect(!staging.client.take_reply(release.sequence),
                       "captured source tail prevents premature cleanup reply");
                return fenced_steps.load() >= 16;
            });
            expect(!provider.release_context_proofs && !provider.host_unregisters,
                   "same-origin cleanup is fenced before worker entry, not mislabeled Deferred");
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            expect(!provider.host_unregisters && !staging.client.take_reply(release.sequence),
                   "source-bound cleanup and actual mapping ownership survive End");
            phase = 0;
            provider.observed_release_context = nullptr;
            provider.held_event_stream = nullptr;
            expect(staging.settle(release, "helper.StagedCrossEndRelease").original_status ==
                       CUDA_SUCCESS,
                   "exact successful unregister remains original SDK provenance");
            verify_full(source, allocation, upload.expected);
            upload.verify();
            staging.close();

            endpoint.call(caller, ExecutionNcclGroupStart{});
            HostPage download;
            std::memset(download.mapping, 0x3c, 64);
            HostHelper receiver(path, peer, caller, source.client.session(), true);
            receiver.import(download);
            provider.hold_next_host_copy = true;
            const auto transfer = source.submit(
                receiver.transfer_opcode(), receiver.transfer_payload(allocation, stream, false));
            expect(transfer.status == ExecutionSocketSubmitStatus::kAccepted &&
                       transfer.sequence == source.next_sequence++,
                   "submit one actual STG1 download without pretending its event completed");
            await([&] {
                expect(source.client.poll_for(0), "drive staged download to its native callback");
                expect(!source.client.take_reply(transfer.sequence),
                       "STG1 D2H reply waits its exact physical transfer tail");
                return provider.host_downloads.load() == 1;
            });
            expect(provider.held_event_stream.load(), "hold the actual staged download stream");
            provider.held_event_stream = nullptr;
            expect(source.settle(transfer, "source.StagedDownload").original_status == CUDA_SUCCESS,
                   "completed staged readback retains exact requested status");
            download.verify();
            verify_full(source, allocation, upload.expected);
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            source.close();
            const auto closed_origin_release = receiver.release(true);
            expect(receiver.settle(closed_origin_release, "helper.ClosedOriginCleanup")
                           .original_status == CUDA_SUCCESS,
                   "saved cleanup parent works after its exact source context closes");
            receiver.close();
            endpoint.close();
        },
        [&](ExecutionService& service) {
            if (phase == 1 && service.pending_priority_requests(7)) ++fenced_steps;
        });
    expect(fenced_steps >= 16 && provider.host_registers == 2 && provider.host_maps == 2 &&
               provider.host_probes == 2 && provider.host_unregisters == 2 &&
               provider.host_uploads == 1 && provider.host_downloads == 1,
           "both STG1 generations consume exact full buffers and retire each real mapping once");
}
void pre_group_host_release() {
    provider.has_host_registration = true;
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{251, 257}};
            const auto communicator = endpoint.initialize(caller, endpoint.uid(caller));
            HostPage page;
            HostHelper helper(path, peer, caller, 0, false);
            helper.import(page);
            expect(!helper.parent, "unwrapped pre-group import creates no cleanup-parent claim");
            endpoint.call(caller, ExecutionNcclGroupStart{});
            const auto released = helper.settle(helper.release(false), "helper.OrdinaryRelease");
            expect(released.original_status == CUDA_SUCCESS && provider.host_unregisters == 1 &&
                       !helper.parent,
                   "ordinary wrapped release uses live group authority, not an invented parent");
            HostPage terminal_page;
            HostHelper terminal(path, peer, caller, 0, false);
            terminal.import(terminal_page);
            expect(terminal.parent.has_value(), "in-group import binds its exact cleanup parent");
            terminal.close();
            expect(provider.host_unregisters == 2,
                   "named wrapped helper Close retires its own parent without self-deadlock");
            terminal_page.verify();
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            page.verify();
            helper.close();
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
        },
        [](ExecutionService&) {});
    expect(provider.host_registers == 2 && provider.host_unregisters == 2,
           "pre-group release and wrapped terminal Close each unregister exactly once");
}
void host_close_after_end() {
    provider.has_host_registration = true;
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{263, 269}};
            const auto communicator = endpoint.initialize(caller, endpoint.uid(caller));
            endpoint.call(caller, ExecutionNcclGroupStart{});
            HostPage page;
            HostHelper helper(path, peer, caller, 0, false);
            helper.import(page);
            Source source(path, peer, &caller);
            const auto allocation = source.allocate();
            source.upload(allocation);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            provider.hold_next_host_copy = true;
            (void)source.complete(
                source.exchange(helper.transfer_opcode(),
                                helper.transfer_payload(allocation, stream, true)),
                "source.CloseHeldUpload");
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            provider.observed_release_context = helper.native_context();
            const auto closed = helper.submit(ExecutionOpcode::kClose, {});
            await([&] {
                expect(helper.client.poll_for(0),
                       "deliver explicit Close with retained import parent");
                expect(!helper.client.take_reply(closed.sequence),
                       "Close cannot invent stop for an accepted registered-host transfer");
                return provider.release_context_proofs.load() >= 2;
            });
            expect(!provider.host_unregisters && provider.host_uploads == 1,
                   "real helper shutdown defers while the actual source mapping lease lives");
            provider.observed_release_context = nullptr;
            provider.held_event_stream = nullptr;
            const auto reply = helper.settle(closed, "helper.CloseAfterEnd");
            expect(!reply.original_status && provider.host_unregisters == 1,
                   "Close retires its saved parent only after one real unregister and shutdown");
            verify_full(source, allocation, page.expected);
            page.verify();
            source.close();
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
        },
        [](ExecutionService&) {});
}
void host_import_reply_loss() {
    provider.has_host_registration = true;
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{271, 277}};
            const auto communicator = endpoint.initialize(caller, endpoint.uid(caller));
            endpoint.call(caller, ExecutionNcclGroupStart{});
            HostPage page;
            auto helper = std::make_unique<HostHelper>(path, peer, caller, 0, false);
            ExecutionPayloadWriter input;
            expect(input.write_u64(64) && input.write_u64(page.bytes) && input.write_u32(0),
                   "encode descriptor-backed import whose actual reply will be disconnected");
            provider.hold_host_registration = true;
            provider.may_return_host_registration = false;
            const auto imported = helper->submit(ExecutionOpcode::kImportHostMemory, input.take(),
                                                 page.descriptor.get());
            await([&] {
                expect(helper->client.poll_for(0), "deliver held actual import request");
                expect(!helper->client.take_reply(imported.sequence),
                       "held native registration cannot have a completed import reply");
                return provider.host_registration_entered.load();
            });
            helper.reset();
            expect(
                provider.host_registers == 1 && !provider.host_maps && !provider.host_unregisters,
                "disconnect preserves the actually entered native registration and output owner");
            provider.may_return_host_registration = true;
            await([] { return provider.host_maps.load() == 1; });
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            await([] { return provider.host_unregisters.load() == 1; });
            page.verify();
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
        },
        [](ExecutionService&) {});
    expect(
        provider.host_registers == 1 && provider.host_maps == 1 && provider.host_unregisters == 1,
        "lost import reply still consumes the mailbox and refunds its real owner exactly once");
}
void host_wrapped_close_deferred() {
    provider.has_host_registration = true;
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{281, 283}};
            const auto communicator = endpoint.initialize(caller, endpoint.uid(caller));
            endpoint.call(caller, ExecutionNcclGroupStart{});
            HostPage page;
            HostHelper helper(path, peer, caller, 0, false);
            helper.import(page);
            Source source(path, peer, &caller);
            const auto allocation = source.allocate();
            source.upload(allocation);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            provider.hold_next_host_copy = true;
            (void)source.complete(
                source.exchange(helper.transfer_opcode(),
                                helper.transfer_payload(allocation, stream, true)),
                "source.WrappedCloseHeldUpload");
            provider.observed_release_context = helper.native_context();
            const auto closed = helper.submit(ExecutionOpcode::kClose, {});
            await([&] {
                expect(helper.client.poll_for(0), "deliver named terminal Close inside live group");
                expect(!helper.client.take_reply(closed.sequence),
                       "named Close cannot return success for an actually held source mapping");
                return provider.release_context_proofs.load() >= 2;
            });
            expect(caller.depth == 1 && !provider.calls[7] && !provider.host_unregisters,
                   "one completed Deferred shutdown is consumed and retried before native End");
            provider.observed_release_context = nullptr;
            provider.held_event_stream = nullptr;
            const auto reply = helper.settle(closed, "helper.WrappedCloseAfterDeferred");
            expect(!reply.original_status && provider.host_unregisters == 1 && !provider.calls[7],
                   "fresh exact Close grant cleans once without End or fabricated SDK provenance");
            verify_full(source, allocation, page.expected);
            page.verify();
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            source.close();
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
        },
        [](ExecutionService&) {});
}
void host_wrapped_close_retry_disconnect() {
    provider.has_host_registration = true;
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{293, 307}};
            const auto communicator = endpoint.initialize(caller, endpoint.uid(caller));
            endpoint.call(caller, ExecutionNcclGroupStart{});
            HostPage page;
            auto helper = std::make_unique<HostHelper>(path, peer, caller, 0, false);
            helper->import(page);
            Source source(path, peer, &caller);
            const auto allocation = source.allocate();
            source.upload(allocation);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            provider.hold_next_host_copy = true;
            (void)source.complete(
                source.exchange(helper->transfer_opcode(),
                                helper->transfer_payload(allocation, stream, true)),
                "source.DisconnectedCloseHeldUpload");
            provider.observed_release_context = helper->native_context();
            provider.hold_release_proof_at = 2;
            provider.may_return_release_proof = false;
            const auto closed = helper->submit(ExecutionOpcode::kClose, {});
            await([&] {
                expect(helper->client.poll_for(0), "deliver named Close and its actual retry");
                expect(!helper->client.take_reply(closed.sequence),
                       "neither Deferred shutdown nor held retry supplies application success");
                return provider.release_proof_entered.load();
            });
            expect(provider.release_context_proofs == 2 && !provider.host_unregisters,
                   "first Deferred was consumed before the distinct second shutdown attempt");
            // This deliberately disconnects an already accepted second attempt,
            // not the unobservable service interval before its retry grant.
            helper.reset();
            expect(!provider.host_unregisters && provider.host_uploads == 1,
                   "socket loss cannot free a mapping used by real pending source work");
            provider.held_event_stream = nullptr;
            provider.hold_release_proof_at = 0;
            provider.may_return_release_proof = true;
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            await([] { return provider.host_unregisters.load() == 1; });
            provider.observed_release_context = nullptr;
            verify_full(source, allocation, page.expected);
            page.verify();
            source.close();
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
        },
        [](ExecutionService&) {});
    expect(provider.host_registers == 1 && provider.host_unregisters == 1,
           "accepted retry disconnect retires exact native ownership without a Close reply");
}
void mixed_group_rejections() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto send = source.allocate(), first = source.allocate(),
                       second = source.allocate();
            for (const auto allocation : {send, first, second}) source.upload(allocation);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            source.fill(send, stream, 8.25F);
            Endpoint endpoint(path, peer);
            Caller caller{{151, 157}};
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto direct = endpoint.initialize(caller, uid);
            endpoint.call(caller, ExecutionNcclGroupStart{});
            endpoint.call(caller, ExecutionNcclGroupStart{});
            const auto grouped = endpoint.initialize(caller, uid);
            const auto before_collective = provider.calls[8].load();
            const auto rejected_collective =
                endpoint.rejected(caller, collective(grouped, source, send, first, stream));
            expect(rejected_collective.outer.status == CUDA_ERROR_INVALID_VALUE &&
                       provider.calls[8] == before_collective && caller.depth == 2,
                   "collective after initialization gets a bounded policy rejection without "
                   "native entry or lost nested End credit");
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            endpoint.call(caller, collective(grouped, source, send, first, stream));
            source.verify(first, {8.25F});
            endpoint.call(caller, ExecutionNcclGroupStart{});
            endpoint.call(caller, ExecutionNcclGroupStart{});
            endpoint.call(caller, collective(direct, source, send, second, stream));
            const auto before_init = provider.calls[1].load();
            const auto rejected_init =
                endpoint.rejected(caller, ExecutionNcclInitRank{true, 1, 0, uid});
            expect(rejected_init.outer.status == CUDA_ERROR_INVALID_VALUE &&
                       provider.calls[1] == before_init && caller.depth == 2,
                   "initialization after collective gets a bounded policy rejection without "
                   "native entry or lost nested End credit");
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            source.verify(second, {8.25F});
            endpoint.call(caller, ExecutionNcclGroupStart{});
            endpoint.call(caller, collective(grouped, source, send, first, stream));
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            source.verify(send, {8.25F});
            source.verify(first, {8.25F});
            source.verify(second, {8.25F});
            endpoint.call(caller, ExecutionNcclDestroy{direct});
            endpoint.call(caller, ExecutionNcclDestroy{grouped});
            endpoint.close();
            source.close();
        },
        [](ExecutionService&) {});
    expect(provider.calls[1] == 2 && provider.calls[8] == 3 && provider.calls[6] == 5 &&
               provider.calls[7] == 5 && provider.calls[4] == 2,
           "both clean public rejections preserve dense subsequent native sequencing, nested "
           "End, next-group use and exact healthy cleanup");
}
Stream* last_native_stream() {
    std::lock_guard lock(provider.mutex);
    expect(provider.next_stream, "actual typed Driver stream factory completed");
    return &provider.streams[provider.next_stream - 1];
}
void asynchronous_source_cut() {
    uses_async_cut_kernel = true;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "authenticate deferred-source fixture owner");
    const auto socket_path = "/tmp/glimmer-nccl-service-" + std::to_string(::getpid()) + ".sock";
    std::array<char, 4096> executable_bytes{};
    const auto executable_size =
        ::readlink("/proc/self/exe", executable_bytes.data(), executable_bytes.size());
    expect(
        executable_size > 0 && static_cast<std::size_t>(executable_size) < executable_bytes.size(),
        "resolve unchanged deferred-source fixture executable");
    const std::string executable{executable_bytes.data(),
                                 static_cast<std::size_t>(executable_size)};
    int descriptors[2]{};
    expect(::pipe2(descriptors, O_CLOEXEC) == 0, "create owned high-demand start pipe");
    ExecutionOwnedDescriptor signal_read(descriptors[0]), signal_write(descriptors[1]);
    const std::array<std::string, 3> arguments{std::to_string(identity->pid),
                                               std::to_string(identity->start_time_ticks),
                                               std::to_string(signal_read.get())};
    const auto child = ::fork();
    expect(child >= 0, "spawn real higher-priority peer before creating service threads");
    if (!child) {
        if (::close(signal_write.get()) != 0 || ::fcntl(signal_read.get(), F_SETFD, 0) != 0)
            ::_exit(126);
        ::execl(executable.c_str(), executable.c_str(), "--high", socket_path.c_str(),
                arguments[0].c_str(), arguments[1].c_str(), arguments[2].c_str(),
                static_cast<char*>(nullptr));
        ::_exit(127);
    }
    signal_read = ExecutionOwnedDescriptor{};
    TaskPeerIdentity high{static_cast<std::uint32_t>(child),
                          static_cast<std::uint32_t>(::geteuid()), 0};
    expect(read_process_start_time(high.pid, &high.start_time_ticks),
           "bind high demand to the exact authenticated process incarnation");
    std::atomic<unsigned> phase{};
    std::atomic<bool> pressure{}, group_is_open{}, observed_unprepared{}, observed_suffix_blocked{};
    unsigned unprepared_steps{}, suffix_steps{};
    constexpr float prefix_value = 19.25F, suffix_value = 27.75F;
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            // This source models ordinary launches from a caller outside the
            // other caller's live native group. ASY1 must never be wrapped SCN1.
            Source source(path, peer);
            ExecutionPayloadWriter allocate;
            expect(allocate.write_u64(64), "encode complete async source allocation");
            const auto send =
                source.complete(source.exchange(ExecutionOpcode::kAllocate, allocate.take()),
                                "source.AsyncAllocate");
            source.allocations.push_back(send.id);
            const auto receive = source.allocate();
            source.upload(send.id);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            ExecutionPayloadWriter image;
            expect(image.write_string(kAsyncCutSource), "encode real transformable source PTX");
            const auto module = source.complete(
                source.exchange(ExecutionOpcode::kLoadPtx, image.take()), "source.AsyncModule");
            ExecutionPayloadWriter lookup;
            expect(lookup.write_u64(module.id) && lookup.write_string("kernel"),
                   "resolve exact public kernel before entering native group");
            const auto function =
                source.complete(source.exchange(ExecutionOpcode::kGetFunction, lookup.take()),
                                "source.AsyncFunction");
            expect(function.parameter_bytes == 12 && !async_cut_preparations,
                   "cold binding prepares metadata, not a pending command occupancy catalog");
            Caller caller{{379, 383}};
            Endpoint endpoint(path, peer);
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto communicator = endpoint.initialize(caller, uid);
            endpoint.call(caller, ExecutionNcclGroupStart{});
            group_is_open = true;
            const unsigned char signal = 1;
            expect(::write(signal_write.get(), &signal, sizeof(signal)) == 1,
                   "admit real high demand only after the exact low native group opens");
            await([&] { return pressure.load(); });
            // A same-priority open group alone permits ordinary preparation.
            // Real higher-priority demand now denies that permission, while
            // the retained native group can still receive exact progress cuts.
            ExecutionPayloadWriter host_stream;
            expect(host_stream.write_u64(stream), "encode exact CPU callback boundary");
            const auto callback = source.complete(
                source.exchange(ExecutionOpcode::kEnqueueHostFunction, host_stream.take()),
                "source.AsyncBoundary");
            const auto raw_submit = [&](ExecutionOpcode operation,
                                        std::span<const std::byte> bytes = {}) {
                const auto submitted = source.client.submit(operation, bytes);
                expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted &&
                           submitted.sequence == source.next_sequence++,
                       "ordinary async source preserves exact raw request sequence");
                return submitted;
            };
            const auto launch = [&](float value) {
                std::array<std::byte, 12> parameters{};
                const auto output = send.address + 16;
                const auto bits = std::bit_cast<std::uint32_t>(value);
                std::memcpy(parameters.data(), &output, 8);
                std::memcpy(parameters.data() + 8, &bits, 4);
                ExecutionPayloadWriter input;
                expect(input.write_u64(function.id) && input.write_u64(stream) &&
                           input.write_u32(8) && input.write_u32(1) && input.write_u32(1) &&
                           input.write_u32(1) && input.write_u32(1) && input.write_u32(1) &&
                           input.write_u32(0) && input.write_blob(parameters) &&
                           input.write_u32(kExecutionAsyncLaunchMagic),
                       "encode ordinary reliable async launch, never a continuation envelope");
                return raw_submit(ExecutionOpcode::kLaunch, input.take());
            };
            const auto admitted = [&](const ExecutionSocketSubmission& submitted) {
                std::optional<ExecutionWireReply> reply;
                await([&] {
                    expect(source.client.poll_for(0), "poll ordinary admission receipt");
                    reply = source.client.take_reply(submitted.sequence);
                    return reply.has_value();
                });
                expect(reply->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
                           !reply->status && reply->id && reply->address &&
                           !reply->original_status && !reply->parameter_bytes &&
                           reply->arguments.empty() && reply->data.empty(),
                       "raw ASY1 returns exact reliable admission, not native completion");
                return *reply;
            };
            const auto prefix = admitted(launch(prefix_value));
            source.expect_outstanding(1);
            phase = 1;
            await([&] { return observed_unprepared.load(); });
            phase = 0;  // The next turn may legitimately prepare the captured prefix.
            const auto collective_request = endpoint.client.submit(
                ExecutionOpcode::kNcclCall,
                endpoint.request(caller,
                                 collective(communicator, source, send.id, receive, stream)));
            expect(
                collective_request.status == ExecutionSocketSubmitStatus::kAccepted,
                "submit actual native collective requiring the captured unprepared source prefix");
            // The exact actor cut must authorize preparation while ordinary
            // source scheduling remains barred by higher-priority demand.
            await([&] {
                expect(endpoint.client.poll_for(0),
                       "flush the exact native collective before waiting for source preparation");
                return async_cut_preparations.load() == 1;
            });
            // Native turns freeze their dependency cut, not later CPU admission.
            // The accepted suffix must not borrow that cut's native authority.
            const auto tail = admitted(launch(suffix_value));
            expect(tail.id > prefix.id && tail.address != prefix.address,
                   "later reliable admission owns a distinct exact sequence and observation");
            source.expect_outstanding(2);
            phase = 2;
            await([&] { return observed_suffix_blocked.load(); });
            ExecutionPayloadWriter query;
            expect(query.write_u64(callback.id),
                   "query independently captured callback predecessor");
            (void)source.settle(raw_submit(ExecutionOpcode::kQueryHostFunction, query.take()),
                                "source.AsyncBoundaryReady");
            ExecutionPayloadWriter finish;
            expect(finish.write_u64(callback.id) && finish.write_u32(1),
                   "settle only the known returned CPU boundary");
            phase = 3;
            (void)source.settle(raw_submit(ExecutionOpcode::kFinishHostFunction, finish.take()),
                                "source.AsyncBoundaryFinish");
            std::optional<ExecutionWireReply> collective_reply;
            await([&] {
                expect(endpoint.client.poll_for(0), "poll exact native collective result");
                collective_reply = endpoint.client.take_reply(collective_request.sequence);
                return collective_reply.has_value();
            });
            const auto completed = endpoint.decode(std::move(collective_reply));
            expect(completed.outer.disposition == ExecutionWireDisposition::kComplete &&
                       !completed.outer.status && completed.outer.original_status == ncclSuccess &&
                       completed.envelope.is_accepted && async_cut_launches == 1 &&
                       async_cut_preparations == 1,
                   "actual native collective progresses only its prefix, never the later ordinary "
                   "suffix");
            ExecutionPayloadWriter observation;
            expect(observation.write_u64(tail.address),
                   "query the exact accepted suffix observation before native End");
            const auto observed = raw_submit(ExecutionOpcode::kQueryFrontier, observation.take());
            std::optional<ExecutionWireReply> pending;
            await([&] {
                expect(source.client.poll_for(0), "poll exact accepted suffix observation");
                pending = source.client.take_reply(observed.sequence);
                return pending.has_value();
            });
            expect(pending->disposition == ExecutionWireDisposition::kComplete &&
                       pending->status == CUDA_ERROR_NOT_READY && !pending->original_status &&
                       !pending->id && !pending->address && !pending->parameter_bytes &&
                       pending->arguments.empty() && pending->data.empty() &&
                       async_cut_preparations == 1 && async_cut_launches == 1,
                   "exact accepted suffix stays physically unretired without native authority");
            // End captures a new source prefix containing the admitted suffix.
            // Its own dependency grant must issue that prefix before SDK End.
            async_cut_end_requested = true;
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            group_is_open = false;
            expect(provider.calls[7] == 1 && async_cut_preparations == 2 && async_cut_launches == 2,
                   "actual End enters once only after its exact newly captured prefix issues");
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "actual high peer completes and closes normally after the native group retires");
            source.drain();
            source.verify(send.id, {suffix_value});
            source.verify(receive, {prefix_value});
            for (const auto frontier : {prefix.address, tail.address}) {
                ExecutionPayloadWriter release;
                expect(release.write_u64(frontier),
                       "release exact async observation after real drain");
                (void)source.complete(
                    source.exchange(ExecutionOpcode::kReleaseFrontier, release.take()),
                    "source.AsyncObservationRelease");
            }
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
            ExecutionPayloadWriter unload;
            expect(unload.write_u64(module.id),
                   "retire exact source module after both kernels stop");
            (void)source.complete(source.exchange(ExecutionOpcode::kUnloadModule, unload.take()),
                                  "source.AsyncModuleRelease");
            source.close();
        },
        [&](ExecutionService& service) {
            if (group_is_open && !provider.calls[7]) {
                const auto pending = service.pending_priority_requests(9);
                if (pending)
                    pressure = true;
                else
                    expect(!pressure,
                           "real high demand stays pending behind the retained native group");
            }
            if (phase == 1) {
                expect(
                    pressure && !async_cut_preparations && !async_cut_launches &&
                        provider.calls[6] == 1 && !provider.calls[7] && !provider.calls[8],
                    "actual higher-priority demand denies preparation before source-cut capture");
                if (++unprepared_steps >= 32) observed_unprepared = true;
            } else if (phase == 2) {
                expect(async_cut_preparations == 1 && !async_cut_launches && !provider.calls[7] &&
                           !provider.calls[8],
                       "exact prefix is prepared, callback and admitted suffix remain unexecuted");
                if (++suffix_steps >= 32) observed_suffix_blocked = true;
            }
        },
        ExecutionSchedulingMode::kPriority, false, PeerAdmission::kConfiguredParent, {.pending = 4},
        {{high, 9, ExecutionClientKernelMode::kOriginal}});
    expect(observed_unprepared && observed_suffix_blocked && async_cut_end_requested &&
               provider.calls[7] == 1 && async_cut_preparations == 2 && async_cut_launches == 2 &&
               async_cut_module_releases &&
               std::ranges::none_of(async_cut_modules, &AsyncCutModule::is_live),
           "distinct collective and End cuts each prepare and execute only their exact prefix "
           "with clean owners");
}

void ordinary_host_function_continuation() {
    std::atomic<unsigned> phase{};
    std::atomic<bool> observed_blocked{}, observed_permitted{};
    unsigned blocked_steps{}, permitted_steps{};
    Stream* source_native{};
    constexpr float before = 31.25F, after = 42.75F;
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Caller caller{{337, 347}};
            Source source(path, peer, &caller);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            source_native = last_native_stream();
            Endpoint endpoint(path, peer);
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto communicator = endpoint.initialize(caller, uid);
            endpoint.call(caller, ExecutionNcclGroupStart{});
            source.fill(send, stream, before);
            source.drain();
            ExecutionPayloadWriter enqueue;
            expect(enqueue.write_u64(stream), "encode ordinary SCN1 host-function target stream");
            const auto accepted = source.complete(
                source.exchange(ExecutionOpcode::kEnqueueHostFunction, enqueue.take()),
                "source.WrappedHostFunction");
            const auto exact_host_receipt = [&](const ExecutionWireReply& reply) {
                expect(reply.id == accepted.id && reply.id && !reply.original_status &&
                           !reply.address && !reply.parameter_bytes && reply.arguments.empty() &&
                           reply.data.empty(),
                       "host-function ownership receipt is exact synthetic CPU metadata");
            };
            exact_host_receipt(accepted);
            const auto later = source.submit_fill(send, stream, after);
            const auto ordinary_sequence = caller.ordinary_sequence;
            const auto helper = [&](ExecutionOpcode operation,
                                    std::span<const std::byte> bytes = {}) {
                const auto submitted = source.client.submit(operation, bytes);
                expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted &&
                           submitted.sequence == source.next_sequence++,
                       "callback helper is unwrapped exact-source metadata, not another SCN1 turn");
                return submitted;
            };
            // The wrapped later fill captures a real source admission fence
            // containing the host boundary. Snapshot proves receive progress
            // without issuing the fill or supplying callback completion.
            const auto marker = helper(ExecutionOpcode::kSnapshot);
            (void)source.settle(marker, "source.HostFunctionFenceSnapshot");
            phase = 1;
            await([&] {
                expect(source.client.poll_for(0) && !source.client.take_reply(later.sequence),
                       "later SCN1 fill cannot enter before callback permission or return");
                return observed_blocked.load();
            });
            ExecutionPayloadWriter query;
            expect(query.write_u64(accepted.id), "query only the exact accepted host token");
            const auto ready = helper(ExecutionOpcode::kQueryHostFunction, query.take());
            exact_host_receipt(source.settle(ready, "source.UnwrappedHostFunctionReady"));
            phase = 2;
            await([&] {
                expect(source.client.poll_for(0) && !source.client.take_reply(later.sequence),
                       "permission alone cannot settle the callback or release its source fence");
                return observed_permitted.load();
            });
            ExecutionPayloadWriter finish;
            expect(finish.write_u64(accepted.id) && finish.write_u32(1),
                   "finish only the granted CPU token after the independent return boundary");
            phase = 3;  // Later issue is legal as soon as the owner consumes Finish.
            const auto completed = helper(ExecutionOpcode::kFinishHostFunction, finish.take());
            exact_host_receipt(source.settle(completed, "source.UnwrappedHostFunctionFinish"));
            (void)source.settle(later, "source.FillAfterHostFunctionReturn");
            expect(caller.ordinary_sequence == ordinary_sequence,
                   "query and finish consume no live-group ordinary continuation ordinal");
            endpoint.call(caller, collective(communicator, source, send, receive, stream));
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            source.verify(send, {after});
            source.verify(receive, {after});
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
            source.close();
        },
        [&](ExecutionService&) {
            const auto observed = phase.load();
            if (observed != 1 && observed != 2) return;
            std::lock_guard lock(provider.mutex);
            expect(provider.calls[6] == 1 && !provider.calls[7] && !provider.calls[8] &&
                       std::ranges::count_if(provider.driver_issues,
                                             [&](const auto& issue) {
                                                 return issue.kind == DriverIssue::Kind::kFill &&
                                                        issue.stream == source_native;
                                             }) == 1,
                   "live native group and actual later source issue remain blocked by the CPU "
                   "boundary, not by an invented native completion");
            if (observed == 1 && ++blocked_steps >= 32) observed_blocked = true;
            if (observed == 2 && ++permitted_steps >= 32) observed_permitted = true;
        },
        ExecutionSchedulingMode::kPriority, false, PeerAdmission::kParentOnly, {.pending = 4});
    expect(observed_blocked && observed_permitted && provider.calls[6] == 1 &&
               provider.calls[7] == 1 && provider.calls[8] == 1 && provider.calls[4] == 1 &&
               std::ranges::count_if(provider.driver_issues,
                                     [&](const auto& issue) {
                                         return issue.kind == DriverIssue::Kind::kFill &&
                                                issue.stream == source_native;
                                     }) == 2,
           "SCN1 host boundary, fenced helper progress, later compute and actual NCCL cleanup "
           "each complete exactly once with full numerical output");
}
void host_function_source_disconnect() {
    std::atomic<unsigned> phase{};
    std::atomic<bool> observed_prepared{}, observed_retention{};
    unsigned prepared_steps{}, retained_steps{};
    Allocation* receive_native{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Caller caller{{349, 353}};
            auto source = std::make_unique<Source>(path, peer, &caller);
            const auto send = source->allocate(), receive = source->allocate();
            source->upload(send);
            source->upload(receive);
            {
                std::lock_guard lock(provider.mutex);
                expect(provider.next_allocation == 2, "identify exact retained target tensor");
                receive_native = &provider.allocations[provider.next_allocation - 1];
            }
            const auto stream = source->stream(CU_STREAM_NON_BLOCKING);
            source->fill(send, stream, 53.25F);
            source->drain();
            Endpoint endpoint(path, peer);
            const auto uid = endpoint.uid(caller);
            caller.context = source->client.session();
            const auto communicator = endpoint.initialize(caller, uid);
            endpoint.call(caller, ExecutionNcclGroupStart{});
            ExecutionPayloadWriter enqueue;
            expect(enqueue.write_u64(stream), "enqueue exact pre-permission source callback");
            const auto accepted = source->complete(
                source->exchange(ExecutionOpcode::kEnqueueHostFunction, enqueue.take()),
                "source.HostFunctionBeforeDisconnect");
            expect(accepted.id && !accepted.original_status && !accepted.parameter_bytes,
                   "actual service accepted a host boundary without callback permission");
            const auto dependent = endpoint.client.submit(
                ExecutionOpcode::kNcclCall,
                endpoint.request(caller, collective(communicator, *source, send, receive, stream)));
            expect(dependent.status == ExecutionSocketSubmitStatus::kAccepted,
                   "submit exact native dependent on the retained host-function source");
            phase = 1;
            await([&] {
                expect(
                    endpoint.client.poll_for(0) && !endpoint.client.take_reply(dependent.sequence),
                    "native dependent remains blocked before source callback permission");
                return observed_prepared.load();
            });
            source.reset();  // No Query/Finish was sent; the application function never ran.
            phase = 2;
            // A new authenticated round trip on the still-live NCL1 carrier
            // proves owner/transport progress after the source peer closed.
            // This rejected metadata request cannot grant native execution.
            const auto marker = endpoint.client.submit(ExecutionOpcode::kSnapshot);
            expect(marker.status == ExecutionSocketSubmitStatus::kAccepted,
                   "keep the native carrier alive after its exact source disappears");
            std::optional<ExecutionWireReply> receipt;
            await([&] {
                expect(
                    endpoint.client.poll_for(0) && !endpoint.client.take_reply(dependent.sequence),
                    "source disconnect is not successful callback completion");
                receipt = endpoint.client.take_reply(marker.sequence);
                return receipt.has_value();
            });
            expect(receipt->disposition == ExecutionWireDisposition::kRejected &&
                       receipt->status == CUDA_ERROR_NOT_SUPPORTED && !receipt->id &&
                       !receipt->original_status && !receipt->parameter_bytes,
                   "live native carrier answers only the unsupported metadata rejection");
            phase = 3;
            await([&] {
                expect(
                    endpoint.client.poll_for(0) && !endpoint.client.take_reply(dependent.sequence),
                    "known absent callback cannot authorize an already accepted native suffix");
                return observed_retention.load();
            });
            // Deliberately no End/Abort/Destroy: retained uncertainty cannot
            // authorize fabricated cleanup of the actor or its source owners.
        },
        [&](ExecutionService& service) {
            const auto observed = phase.load();
            if (!observed) return;
            std::lock_guard lock(provider.mutex);
            expect(!service.finished() && provider.calls[6] == 1 && !provider.calls[7] &&
                       !provider.calls[8] && !provider.primary_releases &&
                       !provider.context_destroys && !provider.stream_destroys &&
                       receive_native->live &&
                       std::ranges::all_of(receive_native->values,
                                           [](auto value) { return value == kGuard; }),
                   "unrun source callback retains native dependents and every target word "
                   "without inventing compute or resource retirement");
            if (observed == 1 && service.pending_priority_requests(7) == 1 &&
                ++prepared_steps >= 32)
                observed_prepared = true;
            if (observed == 3 && ++retained_steps >= 32) observed_retention = true;
        },
        ExecutionSchedulingMode::kPriority, true);
    expect(observed_prepared && observed_retention && !provider.calls[8],
           "actual prepared native request never bypasses a disconnected unpermitted callback");
}
void host_function_query_error() {
    injects_host_function_event_failure = true;
    std::atomic<unsigned> phase{};
    std::atomic<bool> observed_prepared{}, observed_retention{};
    unsigned prepared_steps{}, retained_steps{};
    Allocation* receive_native{};
    Stream* independent_native{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Caller caller{{359, 367}};
            Source source(path, peer, &caller);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            {
                std::lock_guard lock(provider.mutex);
                expect(provider.next_allocation == 2, "identify exact callback-error target");
                receive_native = &provider.allocations[provider.next_allocation - 1];
            }
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            const auto independent = source.stream(CU_STREAM_NON_BLOCKING);
            independent_native = last_native_stream();
            const auto event = source.event();
            source.fill(send, stream, 61.25F);
            source.drain();
            Endpoint endpoint(path, peer);
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto communicator = endpoint.initialize(caller, uid);

            ExecutionPayloadWriter enqueue;
            expect(enqueue.write_u64(stream), "enqueue callback before the unrelated event error");
            const auto accepted = source.complete(
                source.exchange(ExecutionOpcode::kEnqueueHostFunction, enqueue.take()),
                "source.HostFunctionBeforeEventFailure");
            expect(accepted.id && !accepted.original_status && !accepted.parameter_bytes,
                   "real callback admission precedes the independent asynchronous error");
            // The public event generation uses distinct flags. Only the next
            // private issue-preparation event fails, with no partial output or
            // native record. This exercises actual backend retirement/error
            // propagation rather than injecting a service policy failure.
            fails_next_private_event = true;
            source.event_command(ExecutionOpcode::kRecordEvent, event, independent);
            ExecutionPayloadWriter query_event;
            expect(query_event.write_u64(event), "observe exact independently failed event");
            const auto event_reply =
                source.exchange(ExecutionOpcode::kQueryEvent, query_event.take());
            ++source.next_sequence;
            expect(event_reply && event_reply->disposition == ExecutionWireDisposition::kRejected &&
                       event_reply->status == CUDA_ERROR_OUT_OF_MEMORY &&
                       !event_reply->original_status && private_event_failures == 1 &&
                       !fails_next_private_event,
                   "known private-event OOM is real preparation failure, not fabricated SDK issue");
            source.expect_outstanding(0);  // The Session remains healthy, not quarantined.

            // Successful GroupStart after the failure also proves Core has no
            // unknown-state gate yet. Its collective captures only stream S,
            // not the independent nonblocking stream T that failed above.
            endpoint.call(caller, ExecutionNcclGroupStart{});
            const auto dependent = endpoint.client.submit(
                ExecutionOpcode::kNcclCall,
                endpoint.request(caller, collective(communicator, source, send, receive, stream)));
            expect(dependent.status == ExecutionSocketSubmitStatus::kAccepted,
                   "accept actual native dependent despite unrelated sticky asynchronous error");
            phase = 1;
            await([&] {
                expect(
                    endpoint.client.poll_for(0) && !endpoint.client.take_reply(dependent.sequence),
                    "accepted actor remains blocked by the unpermitted callback");
                return observed_prepared.load();
            });
            ExecutionPayloadWriter query;
            expect(query.write_u64(accepted.id), "query exact callback with retained native owner");
            const auto submitted =
                source.client.submit(ExecutionOpcode::kQueryHostFunction, query.take());
            expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted &&
                       submitted.sequence == source.next_sequence++,
                   "callback query remains unwrapped CPU metadata inside the live group");
            std::optional<ExecutionWireReply> reply;
            await([&] {
                expect(source.client.poll_for(0), "observe real callback error receipt");
                reply = source.client.take_reply(submitted.sequence);
                return reply.has_value();
            });
            expect(reply->status == CUDA_ERROR_OUT_OF_MEMORY && !reply->original_status,
                   "callback query preserves the independent original error without SDK status");
            phase = 2;
            await([&] {
                expect(
                    endpoint.client.poll_for(0) && !endpoint.client.take_reply(dependent.sequence),
                    "query error is not successful callback completion for retained actor");
                return observed_retention.load();
            });
            expect(reply->disposition == ExecutionWireDisposition::kRejected && !reply->id &&
                       !reply->address && !reply->parameter_bytes && reply->arguments.empty() &&
                       reply->data.empty(),
                   "retained failed callback has no known-removed-boundary marker");
            // No callback permission/Finish or native End/Abort/Destroy exists.
            // Deliberate retention is inspected in this isolated process only.
        },
        [&](ExecutionService& service) {
            const auto observed = phase.load();
            if (!observed) return;
            std::lock_guard lock(provider.mutex);
            expect(!service.finished() && private_event_failures == 1 && provider.calls[6] == 1 &&
                       !provider.calls[7] && !provider.calls[8] && !provider.primary_releases &&
                       !provider.context_destroys && !provider.stream_destroys &&
                       receive_native->live &&
                       std::ranges::all_of(receive_native->values,
                                           [](auto value) { return value == kGuard; }) &&
                       std::ranges::none_of(provider.driver_issues,
                                            [&](const auto& issue) {
                                                return issue.kind ==
                                                           DriverIssue::Kind::kRecordEvent &&
                                                       issue.stream == independent_native;
                                            }),
                   "unrelated clean event OOM cannot turn an unrun callback into native authority");
            if (observed == 1 && service.pending_priority_requests(7) == 1 &&
                ++prepared_steps >= 32)
                observed_prepared = true;
            if (observed == 2 && ++retained_steps >= 32) observed_retention = true;
        },
        ExecutionSchedulingMode::kPriority, true);
    expect(observed_prepared && observed_retention && private_event_failures == 1 &&
               !provider.calls[8],
           "async-only Query failure preserves the exact retained native dependency");
}
void accepted_foreign_wait(const ExecutionWireReply& reply) {
    expect(reply.disposition == ExecutionWireDisposition::kComplete && !reply.status && reply.id &&
               !reply.original_status && !reply.address && !reply.parameter_bytes &&
               reply.arguments.empty() && reply.data.empty(),
           "actual foreign-wait admission owns an exact command without fabricated SDK status");
}
void foreign_registry_bounds() {
    for (const auto capacity : std::array<std::size_t, 2>{0, 65537}) {
        ExecutionServiceConfig config;
        config.transport.socket_path =
            "/tmp/glimmer-nccl-invalid-edge-capacity-" + std::to_string(::getpid()) + ".sock";
        config.uses_nccl = true;
        config.uses_original_kernels = true;
        config.maximum_foreign_event_edges = capacity;
        bool is_rejected{};
        try {
            ExecutionService service(std::move(config), session_dispatch(), execution_dispatch());
        } catch (const std::invalid_argument& error) {
            is_rejected =
                std::string_view{error.what()} == "invalid foreign event dependency capacity";
        }
        expect(is_rejected && !provider.driver_callbacks &&
                   std::ranges::all_of(provider.calls,
                                       [](const auto& count) { return !count.load(); }),
               "zero and oversized foreign registries reject construction before every Driver "
               "and NCCL callback");
    }
}
void foreign_chain(bool should_disconnect) {
    std::atomic<bool> waiting{}, observed_hold{};
    unsigned held_steps{};
    Stream *first_native{}, *later_native{}, *middle_native{}, *last_native{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            auto first = std::make_unique<Source>(path, peer);
            Source middle(path, peer), last(path, peer);
            const auto first_input = first->allocate(), later_input = first->allocate(),
                       middle_input = middle.allocate(), last_input = last.allocate(),
                       output = last.allocate();
            for (const auto allocation : {first_input, later_input}) first->upload(allocation);
            middle.upload(middle_input);
            last.upload(last_input);
            last.upload(output);
            const auto first_stream = first->stream(CU_STREAM_NON_BLOCKING);
            first_native = last_native_stream();
            const auto later_stream = first->stream(CU_STREAM_NON_BLOCKING);
            later_native = last_native_stream();
            const auto middle_stream = middle.stream(CU_STREAM_NON_BLOCKING);
            middle_native = last_native_stream();
            const auto last_stream = last.stream(CU_STREAM_NON_BLOCKING);
            last_native = last_native_stream();
            const auto first_event = first->event(), middle_event = middle.event();
            Endpoint endpoint(path, peer);
            Caller caller{{163, 167}};
            const auto uid = endpoint.uid(caller);
            caller.context = last.client.session();
            const auto communicator = endpoint.initialize(caller, uid);
            first->drain();
            middle.drain();
            last.drain();
            provider.held_event_stream = first_native;
            first->fill(first_input, first_stream, 4.25F);
            first->event_command(ExecutionOpcode::kRecordEvent, first_event, first_stream);
            accepted_foreign_wait(
                middle.foreign_wait(first->client.session(), first_event, middle_stream));
            const auto middle_fill = middle.submit_fill(middle_input, middle_stream, 6.5F);
            middle.event_command(ExecutionOpcode::kRecordEvent, middle_event, middle_stream);
            middle.expect_outstanding(3);
            accepted_foreign_wait(
                last.foreign_wait(middle.client.session(), middle_event, last_stream));
            const auto last_fill = last.submit_fill(last_input, last_stream, 10.25F);
            last.expect_outstanding(2);
            // The target already captured the old generation. The new record
            // has a different nonblocking stream and must never replace it.
            first->fill(later_input, later_stream, 9.5F);
            first->event_command(ExecutionOpcode::kRecordEvent, first_event, later_stream);
            first->destroy_event(first_event);
            middle.destroy_event(middle_event);
            const auto submission = endpoint.client.submit(
                ExecutionOpcode::kNcclCall,
                endpoint.request(caller,
                                 collective(communicator, last, last_input, output, last_stream)));
            expect(submission.status == ExecutionSocketSubmitStatus::kAccepted &&
                       endpoint.client.poll_for(0),
                   "actual NCCL request waits behind an authenticated three-source event chain");
            if (should_disconnect) first.reset();
            waiting = true;
            await([&] {
                expect(
                    endpoint.client.poll_for(0) && !endpoint.client.take_reply(submission.sequence),
                    "held old-generation source cannot produce an early NCCL receipt");
                return observed_hold.load();
            });
            provider.held_event_stream = nullptr;
            (void)middle.settle(middle_fill, "middle.Fill32");
            (void)last.settle(last_fill, "last.Fill32");
            std::optional<ExecutionWireReply> response;
            await([&] {
                expect(endpoint.client.poll_for(0), "poll exact three-source NCCL receipt");
                response = endpoint.client.take_reply(submission.sequence);
                return response.has_value();
            });
            const auto actual = endpoint.decode(std::move(response));
            expect(actual.outer.disposition == ExecutionWireDisposition::kComplete &&
                       !actual.outer.status && !actual.outer.parameter_bytes &&
                       actual.envelope.is_accepted && actual.envelope.original_status == 0,
                   "three-source event chain reaches one actual successful collective");
            if (first) {
                first->verify(first_input, {4.25F});
                first->verify(later_input, {9.5F});
            }
            middle.verify(middle_input, {6.5F});
            last.verify(last_input, {10.25F});
            last.verify(output, {10.25F});
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
            if (first) first->close();
            middle.close();
            last.close();
        },
        [&](ExecutionService&) {
            if (!waiting.load() || observed_hold.load()) return;
            expect(!provider.calls[8] && !provider.context_destroys,
                   "retained source generations and disconnected owner survive until real stop");
            if (++held_steps == 32) observed_hold = true;
        },
        ExecutionSchedulingMode::kPriority, false, PeerAdmission::kParentOnly,
        {.foreign_edges = 2});
    const auto recorded_on = [&](Stream* stream) {
        const auto found = std::ranges::find_if(provider.driver_issues, [&](const auto& issue) {
            return issue.kind == DriverIssue::Kind::kRecordEvent && issue.stream == stream &&
                   issue.event_flags == kUserEventFlags;
        });
        expect(found != provider.driver_issues.end(), "actual public native event record exists");
        return found->event;
    };
    const auto old_first = recorded_on(first_native), new_first = recorded_on(later_native),
               old_middle = recorded_on(middle_native);
    expect(old_first != new_first, "rerecord owns a distinct immutable native event generation");
    for (const auto& issue : provider.driver_issues) {
        if (issue.kind != DriverIssue::Kind::kWaitEvent || issue.event_flags != kUserEventFlags)
            continue;
        expect((issue.stream == middle_native && issue.event == old_first) ||
                   (issue.stream == last_native && issue.event == old_middle),
               "actual native waits use only captured old generations, never the replacement");
    }
    for (const auto& expected :
         {std::pair{middle_native, old_first}, std::pair{last_native, old_middle}})
        expect(std::ranges::any_of(provider.driver_issues,
                                   [&](const auto& issue) {
                                       return issue.kind == DriverIssue::Kind::kWaitEvent &&
                                              issue.stream == expected.first &&
                                              issue.event == expected.second;
                                   }),
               "both actual foreign-wait callbacks ran on exact target streams");
    constexpr std::array<float, 5> values{4.25F, 9.5F, 6.5F, 10.25F, 10.25F};
    for (std::size_t allocation = 0; allocation != values.size(); ++allocation)
        for (std::size_t index = 0; index != provider.allocations[allocation].values.size();
             ++index)
            expect(provider.allocations[allocation].values[index] ==
                       (index >= 4 && index < 12 ? std::bit_cast<std::uint32_t>(values[allocation])
                                                 : kGuard),
                   "every source tensor and interior guard survives event deletion/disconnect");
    expect(provider.calls[8] == 1 && provider.calls[4] == 1,
           "three-source lifecycle invokes the collective and final native destroy exactly once");
    if (should_disconnect)
        expect(
            provider.implicit_allocations == 2 && provider.implicit_streams == 2,
            "disconnected source context releases only its two retained allocations and streams");
}
void foreign_chain_live() {
    foreign_chain(false);
}
void foreign_chain_disconnect() {
    foreign_chain(true);
}
void foreign_edge_quota_and_low_events() {
    std::atomic<bool> waiting{}, observed_hold{};
    unsigned held_steps{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer), target(path, peer);
            const auto input = source.allocate(), output = target.allocate();
            source.upload(input);
            target.upload(output);
            const auto source_stream = source.stream(CU_STREAM_NON_BLOCKING);
            auto* native_source = last_native_stream();
            const auto target_stream = target.stream(CU_STREAM_NON_BLOCKING);
            const auto source_event = source.event(), local_event = target.event();
            Endpoint endpoint(path, peer);
            Caller caller{{173, 179}};
            (void)endpoint.uid(caller);
            source.drain();
            target.drain();
            provider.held_event_stream = native_source;
            source.fill(input, source_stream, 7.25F);
            source.event_command(ExecutionOpcode::kRecordEvent, source_event, source_stream);
            accepted_foreign_wait(
                target.foreign_wait(source.client.session(), source_event, target_stream));
            target.event_command(ExecutionOpcode::kRecordEvent, local_event, target_stream);
            target.event_command(ExecutionOpcode::kStreamWaitEvent, local_event, target_stream);
            target.expect_outstanding(3);
            const auto exhausted =
                target.foreign_wait(source.client.session(), source_event, target_stream);
            expect(exhausted.disposition == ExecutionWireDisposition::kRejected &&
                       exhausted.status == CUDA_ERROR_OUT_OF_MEMORY && !exhausted.id &&
                       !exhausted.original_status,
                   "bounded foreign registry rejects before another target wait or SDK entry");
            waiting = true;
            await([&] { return observed_hold.load(); });
            provider.held_event_stream = nullptr;
            source.drain();
            target.drain();
            accepted_foreign_wait(
                target.foreign_wait(source.client.session(), source_event, target_stream));
            target.drain();
            target.fill(output, target_stream, 13.5F);
            source.verify(input, {7.25F});
            target.verify(output, {13.5F});
            endpoint.close();
            source.close();
            target.close();
        },
        [&](ExecutionService&) {
            if (!waiting.load() || observed_hold.load()) return;
            expect(!provider.context_destroys && !provider.calls[8],
                   "event-only configured-low epochs do not fabricate compute or early cleanup");
            {
                std::lock_guard lock(provider.mutex);
                const auto progress = std::span{provider.progress}.first(provider.next_context);
                expect(progress.size() == 2 &&
                           std::ranges::all_of(progress,
                                               [](const auto& owner) {
                                                   owner.counters.verify();
                                                   owner.host.verify();
                                                   return owner.is_required &&
                                                          owner.is_counter_live &&
                                                          owner.is_host_live &&
                                                          owner.has_host_mapping &&
                                                          !owner.has_counter_release &&
                                                          !owner.has_host_release;
                                               }),
                       "held configured-low event tails retain both exact private progress owners");
            }
            if (++held_steps == 32) observed_hold = true;
        },
        ExecutionSchedulingMode::kPriority, false, PeerAdmission::kConfiguredParent,
        {.foreign_edges = 1});
    expect(provider.calls[0] == 1 && !provider.calls[8],
           "record/local-wait/foreign-wait low epochs retire and refund without compute credit");
    const auto public_records =
        std::ranges::count_if(provider.driver_issues, [](const auto& issue) {
            return issue.kind == DriverIssue::Kind::kRecordEvent &&
                   issue.event_flags == kUserEventFlags;
        });
    const auto public_waits = std::ranges::count_if(provider.driver_issues, [](const auto& issue) {
        return issue.kind == DriverIssue::Kind::kWaitEvent && issue.event_flags == kUserEventFlags;
    });
    expect(public_records == 2 && public_waits == 3,
           "two records, one local wait and two admitted foreign waits enter the SDK exactly once; "
           "quota rejection enters none");
}
void foreign_record_failure() {
    std::atomic<bool> waiting{}, observed_hold{};
    unsigned held_steps{};
    Stream* failed_stream{};
    Stream* target_native{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Source origin(path, peer), source(path, peer), target(path, peer);
            const auto origin_input = origin.allocate(), input = target.allocate(),
                       output = target.allocate();
            origin.upload(origin_input);
            target.upload(input);
            target.upload(output);
            const auto origin_stream = origin.stream(CU_STREAM_NON_BLOCKING);
            auto* native_origin = last_native_stream();
            const auto source_stream = source.stream(CU_STREAM_NON_BLOCKING);
            failed_stream = last_native_stream();
            const auto target_stream = target.stream(CU_STREAM_NON_BLOCKING);
            target_native = last_native_stream();
            const auto origin_event = origin.event(), source_event = source.event();
            Endpoint endpoint(path, peer);
            Caller caller{{181, 191}};
            const auto uid = endpoint.uid(caller);
            caller.context = target.client.session();
            const auto communicator = endpoint.initialize(caller, uid);
            origin.drain();
            source.drain();
            target.drain();
            endpoint.call(caller, ExecutionNcclGroupStart{});
            provider.held_event_stream = native_origin;
            origin.fill(origin_input, origin_stream, 6.75F);
            origin.event_command(ExecutionOpcode::kRecordEvent, origin_event, origin_stream);
            accepted_foreign_wait(
                source.foreign_wait(origin.client.session(), origin_event, source_stream));
            source.event_command(ExecutionOpcode::kRecordEvent, source_event, source_stream);
            source.expect_outstanding(2);
            accepted_foreign_wait(
                target.foreign_wait(source.client.session(), source_event, target_stream));
            target.expect_outstanding(1);
            // The failed record cannot enter before its foreign predecessor.
            // Both downstream captures already exist; only the real user record
            // fails, never the separate dispatcher completion/proof events.
            provider.user_record_failure_stream = failed_stream;
            provider.user_record_status = CUDA_ERROR_INVALID_VALUE;
            const auto submission = endpoint.client.submit(
                ExecutionOpcode::kNcclCall,
                endpoint.request(caller,
                                 collective(communicator, target, input, output, target_stream)));
            expect(submission.status == ExecutionSocketSubmitStatus::kAccepted &&
                       endpoint.client.poll_for(0),
                   "admit exact NCCL source cut before releasing the delayed actual record");
            waiting = true;
            await([&] {
                expect(
                    endpoint.client.poll_for(0) && !endpoint.client.take_reply(submission.sequence),
                    "held prerequisite cannot fabricate a rejected or successful collective");
                return observed_hold.load();
            });
            provider.held_event_stream = nullptr;
            std::optional<ExecutionWireReply> response;
            await([&] {
                expect(endpoint.client.poll_for(0), "poll real failed-prefix NCCL rejection");
                response = endpoint.client.take_reply(submission.sequence);
                return response.has_value();
            });
            const auto actual = endpoint.decode(std::move(response));
            expect(actual.outer.disposition == ExecutionWireDisposition::kRejected &&
                       actual.outer.status == CUDA_ERROR_INVALID_VALUE && !actual.outer.id &&
                       !actual.outer.original_status && !actual.outer.parameter_bytes &&
                       !actual.envelope.is_accepted && !provider.calls[8] &&
                       provider.user_record_errors == 1,
                   "actual failed source record rejects the dependent NCCL request before any "
                   "collective or original-status invention");
            endpoint.call(caller, ExecutionNcclGroupEnd{});
            (void)endpoint.call(caller, ExecutionNcclGetAsyncError{communicator, true});
            origin.verify(origin_input, {6.75F});
            target.verify(input, {std::bit_cast<float>(kGuard)});
            target.verify(output, {std::bit_cast<float>(kGuard)});
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
            origin.close();
            source.close();
            target.close();
        },
        [&](ExecutionService&) {
            if (!waiting.load() || observed_hold.load()) return;
            expect(!provider.user_record_errors && !provider.calls[8],
                   "the injected real SDK error remains gated behind its exact old prerequisite");
            if (++held_steps == 32) observed_hold = true;
        },
        ExecutionSchedulingMode::kPriority, false, PeerAdmission::kParentOnly,
        {.foreign_edges = 2});
    expect(provider.calls[6] == 1 && provider.calls[7] == 1 && provider.calls[4] == 1 &&
               provider.calls[3] == 1 && !provider.calls[8] && provider.user_record_errors == 1,
           "known failed-prefix rejection preserves valid End, dense later native calls and "
           "healthy cleanup without replay");
    expect(std::ranges::none_of(provider.driver_issues,
                                [&](const auto& issue) {
                                    return issue.kind == DriverIssue::Kind::kWaitEvent &&
                                           issue.event_flags == kUserEventFlags &&
                                           issue.stream == target_native;
                                }),
           "a failed foreign prefix never enters its target native stream-wait callback");
    expect(std::ranges::none_of(provider.driver_issues,
                                [&](const auto& issue) {
                                    return issue.kind == DriverIssue::Kind::kRecordEvent &&
                                           issue.event_flags == kUserEventFlags &&
                                           issue.stream == failed_stream;
                                }),
           "the failed user record is not replayed or recorded as successful physical work");
}
void deferred_event_admission_fifo() {
    std::atomic<bool> waiting{}, observed_hold{};
    unsigned held_steps{};
    Stream* later_native{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto first_input = source.allocate(), later_input = source.allocate();
            source.upload(first_input);
            source.upload(later_input);
            const auto held_stream = source.stream(CU_STREAM_NON_BLOCKING);
            auto* held_native = last_native_stream();
            const auto later_stream = source.stream(CU_STREAM_NON_BLOCKING);
            later_native = last_native_stream();
            const auto rerecorded = source.event(), other_event = source.event();
            Endpoint endpoint(path, peer);
            Caller caller{{193, 197}};
            (void)endpoint.uid(caller);
            source.drain();
            provider.held_event_stream = held_native;
            source.fill(first_input, held_stream, 20.25F);
            source.event_command(ExecutionOpcode::kRecordEvent, rerecorded, held_stream);
            source.event_command(ExecutionOpcode::kStreamWaitEvent, rerecorded, held_stream);
            source.event_command(ExecutionOpcode::kRecordEvent, other_event, held_stream);
            source.expect_outstanding(4);
            const auto submit_event = [&](ExecutionOpcode operation) {
                ExecutionPayloadWriter input;
                expect(input.write_u64(rerecorded) && input.write_u64(later_stream) &&
                           input.write_u32(0),
                       "encode exact deferred event generation and target stream");
                const auto submitted = source.client.submit(operation, input.take());
                expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted &&
                           submitted.sequence == source.next_sequence++,
                       "submit ordered event without pretending backend admission succeeded");
                return submitted;
            };
            const auto earlier_record = submit_event(ExecutionOpcode::kRecordEvent);
            const auto later_wait = submit_event(ExecutionOpcode::kStreamWaitEvent);
            const auto later_fill = source.submit_fill(later_input, later_stream, 22.5F);
            // The queue really contains four physical commands; these three
            // requests are not an artificial registry-quota rejection. A fourth
            // transport slot permits metadata to answer while all three wait.
            source.expect_outstanding(4);
            waiting = true;
            await([&] {
                expect(source.client.poll_for(0) &&
                           !source.client.take_reply(earlier_record.sequence) &&
                           !source.client.take_reply(later_wait.sequence) &&
                           !source.client.take_reply(later_fill.sequence),
                       "later event and ordinary calls cannot overtake a genuinely deferred "
                       "earlier event while Snapshot still answers");
                return observed_hold.load();
            });
            provider.held_event_stream = nullptr;
            const auto recorded = source.settle(earlier_record, "source.DeferredRecordEvent");
            const auto waited = source.settle(later_wait, "source.OrderedLocalWait");
            (void)source.settle(later_fill, "source.OrderedFill32");
            expect(recorded.id == 5 && waited.id == 6 && !recorded.original_status &&
                       !waited.original_status,
                   "deferred event and later wait keep dense admission sequences without "
                   "fabricating requested native statuses");
            source.drain();
            source.verify(first_input, {20.25F});
            source.verify(later_input, {22.5F});
            endpoint.close();
            source.close();
        },
        [&](ExecutionService&) {
            if (!waiting.load() || observed_hold.load()) return;
            std::lock_guard lock(provider.mutex);
            expect(std::ranges::none_of(
                       provider.driver_issues,
                       [&](const auto& issue) { return issue.stream == later_native; }),
                   "no later-stream event or ordinary native call enters while the earlier "
                   "event cannot obtain actual queue capacity");
            if (++held_steps == 32) observed_hold = true;
        },
        ExecutionSchedulingMode::kPriority, false, PeerAdmission::kParentOnly,
        {.pending = 4, .commands = 4});
    const auto recorded = std::ranges::find_if(provider.driver_issues, [&](const auto& issue) {
        return issue.kind == DriverIssue::Kind::kRecordEvent && issue.stream == later_native &&
               issue.event_flags == kUserEventFlags;
    });
    expect(recorded != provider.driver_issues.end(), "deferred native event eventually records");
    const auto waited = std::ranges::find_if(provider.driver_issues, [&](const auto& issue) {
        return issue.kind == DriverIssue::Kind::kWaitEvent && issue.stream == later_native &&
               issue.event == recorded->event;
    });
    const auto filled = std::ranges::find_if(provider.driver_issues, [&](const auto& issue) {
        return issue.kind == DriverIssue::Kind::kFill && issue.stream == later_native &&
               issue.value == std::bit_cast<std::uint32_t>(22.5F);
    });
    expect(waited != provider.driver_issues.end() && filled != provider.driver_issues.end() &&
               recorded < waited && waited < filled,
           "actual SDK order is deferred rerecord, wait on its exact new generation, then "
           "ordinary fill; the old generation never replaces the accepted capture");
    for (const auto kind :
         {DriverIssue::Kind::kRecordEvent, DriverIssue::Kind::kWaitEvent, DriverIssue::Kind::kFill})
        expect(std::ranges::count_if(provider.driver_issues,
                                     [&](const auto& issue) {
                                         return issue.stream == later_native &&
                                                issue.kind == kind &&
                                                (kind == DriverIssue::Kind::kFill ||
                                                 issue.event_flags == kUserEventFlags);
                                     }) == 1,
               "each ordered public event and ordinary native call enters exactly once");
}
void asynchronous() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            Endpoint endpoint(path, peer);
            Caller caller{{53, 59}};
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto communicator = endpoint.initialize(caller, uid);
            source.fill(send, stream, 9.5F);
            provider.collective_status = ncclInProgress;
            provider.async_value = ncclInProgress;
            endpoint.call(caller, collective(communicator, source, send, receive, stream),
                          ncclInProgress);
            const auto pending =
                endpoint.call(caller, ExecutionNcclGetAsyncError{communicator, true});
            expect(pending.envelope.observation.async_error == ncclInProgress,
                   "original async query, not an invented completion poll, reports pending native "
                   "work");
            provider.collective_status = 0;
            provider.async_value = 0;
            const auto ready =
                endpoint.call(caller, ExecutionNcclGetAsyncError{communicator, true});
            expect(ready.envelope.observation.async_error == 0,
                   "requested readiness enables original-batch record continuation");
            source.verify(receive, {9.5F});
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
            source.close();
        },
        [](ExecutionService&) {});
    expect(provider.calls[8] == 1 && provider.calls[3] == 2,
           "async continuation never replays original collective or adds hidden query");
}
void asynchronous_two_communicators() {
    // This callback choice is cold and scenario-local, before runtime/worker creation.
    returns_async_outer_end = true;
    std::atomic<unsigned> phase{};
    std::atomic<bool> observed_pending{}, observed_held_tail{};
    Stream* first_native{};
    Stream* second_native{};
    std::size_t records_before_readiness{};
    unsigned pending_steps{}, tail_steps{};
    const auto source_records = [&] {
        std::lock_guard lock(provider.mutex);
        std::size_t records{};
        for (const auto& issue : provider.driver_issues)
            if (issue.kind == DriverIssue::Kind::kRecordEvent &&
                (issue.stream == first_native || issue.stream == second_native))
                ++records;
        return records;
    };
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto first_input = source.allocate(), second_input = source.allocate(),
                       first_output = source.allocate(), second_output = source.allocate();
            for (const auto allocation : {first_input, second_input, first_output, second_output})
                source.upload(allocation);
            const auto first_stream = source.stream(CU_STREAM_NON_BLOCKING);
            const auto second_stream = source.stream(CU_STREAM_NON_BLOCKING);
            {
                std::lock_guard lock(provider.mutex);
                expect(provider.next_stream >= 2, "two distinct native source streams exist");
                first_native = &provider.streams[provider.next_stream - 2];
                second_native = &provider.streams[provider.next_stream - 1];
                expect(first_native != second_native && first_native->live && second_native->live &&
                           first_native->context == second_native->context,
                       "the two exact native streams belong to the same live source context");
            }
            Endpoint endpoint(path, peer);
            Caller caller{{163, 167}};
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto first = endpoint.initialize(caller, uid);
            const auto second = endpoint.initialize(caller, uid);
            expect(first != second && first.object && second.object,
                   "two independently owned communicator tokens are not interchangeable");
            source.fill(first_input, first_stream, 12.5F);
            source.fill(second_input, second_stream, 27.25F);
            source.drain();
            endpoint.call(caller, ExecutionNcclGroupStart{});
            endpoint.call(caller,
                          collective(first, source, first_input, first_output, first_stream));
            endpoint.call(caller,
                          collective(second, source, second_input, second_output, second_stream));
            endpoint.call(caller, ExecutionNcclGroupEnd{}, ncclInProgress);
            expect(
                !caller.depth && !caller.group && !provider.events_ready.load(),
                "actual outer End7 closes host TLS but retains independently held physical work");
            records_before_readiness = source_records();
            const auto drain = source.submit(ExecutionOpcode::kDrain);
            expect(drain.status == ExecutionSocketSubmitStatus::kAccepted &&
                       drain.sequence == source.next_sequence++,
                   "submit exact source Drain without manufacturing physical completion");
            provider.async_value = 0;
            const auto first_ready = endpoint.call(caller, ExecutionNcclGetAsyncError{first, true});
            expect(first_ready.envelope.observation.effects == 4 &&
                       first_ready.envelope.observation.async_error_mask == 15 &&
                       first_ready.envelope.observation.async_error == 0,
                   "first exact communicator has actual affirmative whole-field readiness proof");
            provider.async_value = ncclInProgress;
            const auto second_pending =
                endpoint.call(caller, ExecutionNcclGetAsyncError{second, true});
            expect(second_pending.envelope.observation.effects == 4 &&
                       second_pending.envelope.observation.async_error_mask == 15 &&
                       second_pending.envelope.observation.async_error == ncclInProgress,
                   "the other communicator independently remains pending after the first async0");
            phase = 1;
            await([&] {
                expect(source.client.poll_for(0) && !source.client.take_reply(drain.sequence),
                       "one ready communicator cannot complete the exact group's source Drain");
                return observed_pending.load();
            });
            provider.async_value = 0;
            const auto second_ready =
                endpoint.call(caller, ExecutionNcclGetAsyncError{second, true});
            expect(
                second_ready.envelope.observation.effects == 4 &&
                    second_ready.envelope.observation.async_error_mask == 15 &&
                    second_ready.envelope.observation.async_error == 0,
                "only the second exact pinned written async0 discharges its remaining readiness");
            phase = 2;
            await([&] {
                expect(source.client.poll_for(0) && !source.client.take_reply(drain.sequence),
                       "both readiness replies still cannot replace actual source physical tails");
                return observed_held_tail.load();
            });
            provider.events_ready = true;
            (void)source.settle(drain, "source.TwoCommunicatorDrain");
            source.verify(first_input, {12.5F});
            source.verify(second_input, {27.25F});
            source.verify(first_output, {12.5F});
            source.verify(second_output, {27.25F});
            endpoint.call(caller, ExecutionNcclDestroy{first});
            endpoint.call(caller, ExecutionNcclDestroy{second});
            endpoint.close();
            source.close();
        },
        [&](ExecutionService&) {
            if (phase.load() == 1 && !observed_pending.load()) {
                expect(provider.calls[3] == 2 && provider.calls[8] == 2 && provider.calls[7] == 1 &&
                           !provider.calls[4] && !provider.events_ready.load() &&
                           source_records() == records_before_readiness,
                       "another communicator's async0 cannot record this group's final source "
                       "enqueue boundary or invoke hidden readiness/cleanup callbacks");
                if (++pending_steps == 32) observed_pending = true;
            } else if (phase.load() == 2 && !observed_held_tail.load()) {
                expect(
                    provider.calls[3] == 3 && !provider.calls[4] && !provider.events_ready.load(),
                    "actual final tails remain owned after both requested readiness replies");
                if (source_records() > records_before_readiness && ++tail_steps == 32)
                    observed_held_tail = true;
            }
        });
    expect(provider.calls[1] == 2 && provider.calls[8] == 2 && provider.calls[3] == 3 &&
               provider.calls[4] == 2 && provider.calls[6] == 1 && provider.calls[7] == 1 &&
               observed_pending.load() && observed_held_tail.load(),
           "two real communicator readiness proofs and actual stopped tails retire exactly once");
    expect(provider.async_queries ==
               std::vector<ncclComm_t>{reinterpret_cast<ncclComm_t>(&provider.communicators[0]),
                                       reinterpret_cast<ncclComm_t>(&provider.communicators[1]),
                                       reinterpret_cast<ncclComm_t>(&provider.communicators[1])},
           "actual native query handles are exactly A, B, B, not just symbolic reply identities");
}
void capabilities_and_sequences() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            Endpoint endpoint(path, peer);
            Caller caller{{61, 67}};
            const auto uid = endpoint.uid(caller);
            caller.context = source.client.session();
            const auto communicator = endpoint.initialize(caller, uid);
            auto bad = collective(communicator, source, send, receive, stream);
            bad.receive.offset = 60;
            endpoint.rejected(caller, bad);
            bad = collective(communicator, source, send, receive, stream);
            bad.send.allocation = UINT64_MAX;
            endpoint.rejected(caller, bad);
            bad = collective(communicator, source, send, receive, stream);
            bad.stream.owner = UINT64_MAX;
            endpoint.rejected(caller, bad);
            bad = collective(communicator, source, send, receive, stream);
            bad.communicator.domain = 2;
            endpoint.rejected(caller, bad);
            const auto proven_primary = caller.primary_domain;
            caller.primary_domain = UINT64_MAX;
            endpoint.rejected(caller, ExecutionNcclGetUniqueId{true});
            caller.primary_domain = proven_primary;
            expect(!provider.calls[8],
                   "bad ranges/allocations/stream owners/communicators never enter original SDK");
            (void)endpoint.uid(caller);
            const auto repeated = endpoint.request(caller, ExecutionNcclGetUniqueId{true});
            const auto first =
                endpoint.decode(endpoint.client.exchange(ExecutionOpcode::kNcclCall, repeated));
            expect(first.envelope.is_accepted && first.envelope.original_status == 0,
                   "one exact canonical scope is accepted once");
            const auto duplicate =
                endpoint.decode(endpoint.client.exchange(ExecutionOpcode::kNcclCall, repeated));
            expect(!duplicate.envelope.is_accepted && duplicate.outer.status &&
                       !duplicate.outer.original_status,
                   "replayed full scope cannot cause a second original invocation");
            (void)endpoint.uid(caller);
            source.fill(send, stream, 7.25F);
            endpoint.call(caller, collective(communicator, source, send, receive, stream));
            source.verify(receive, {7.25F});
            Caller malformed{{151, 157}};
            auto truncated = endpoint.request(malformed, ExecutionNcclGetUniqueId{true});
            truncated.pop_back();
            --malformed.sequence;  // A noncanonical body never consumed this caller scope.
            const auto rejected =
                endpoint.decode(endpoint.client.exchange(ExecutionOpcode::kNcclCall, truncated));
            expect(!rejected.envelope.is_accepted && rejected.outer.status &&
                       !rejected.outer.original_status,
                   "malformed operation body rejects without SDK entry or native status invention");
            (void)endpoint.uid(malformed);
            endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.close();
            source.close();
        },
        [](ExecutionService&) {});
    expect(provider.calls[8] == 1 && provider.calls[0] == 5,
           "canonical clean rejections advance only wire caller sequencing, never replay SDK work");
}
struct ForeignCapabilities {
    std::uint64_t source{}, send{}, receive{}, stream{};
    ExecutionNcclCommunicator parent;
};
void foreign_client(const std::string& path, TaskPeerIdentity peer,
                    const ForeignCapabilities& tokens) {
    Endpoint foreign(path, peer);
    Caller caller{{79, 83}};
    const auto uid = foreign.uid(caller);
    const auto own = foreign.initialize(caller, uid);
    ExecutionNcclAllGather request{
        own,
        {ExecutionNcclStreamKind::kOwned, tokens.source, tokens.stream},
        {ExecutionNcclBufferKind::kOwned, tokens.source, tokens.send, 16},
        {ExecutionNcclBufferKind::kOwned, tokens.source, tokens.receive, 16},
        8,
        static_cast<std::int32_t>(ncclFloat32)};
    foreign.rejected(caller, request);
    request.communicator = tokens.parent;
    foreign.rejected(caller, request);
    (void)foreign.uid(caller);
    foreign.call(caller, ExecutionNcclDestroy{own});
    foreign.close();
}
void foreign_process_capability() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            Endpoint endpoint(path, peer);
            Caller parent{{71, 73}};
            const auto uid = endpoint.uid(parent);
            parent.context = source.client.session();
            const auto communicator = endpoint.initialize(parent, uid);
            std::array<char, 4096> executable_bytes{};
            const auto executable_size =
                ::readlink("/proc/self/exe", executable_bytes.data(), executable_bytes.size());
            expect(executable_size > 0 &&
                       static_cast<std::size_t>(executable_size) < executable_bytes.size(),
                   "resolve exact executable before forking a clean foreign client");
            const std::string executable{executable_bytes.data(),
                                         static_cast<std::size_t>(executable_size)};
            const std::array<std::string, 9> arguments{path,
                                                       std::to_string(peer.pid),
                                                       std::to_string(peer.start_time_ticks),
                                                       std::to_string(source.client.session()),
                                                       std::to_string(send),
                                                       std::to_string(receive),
                                                       std::to_string(stream),
                                                       std::to_string(communicator.domain),
                                                       std::to_string(communicator.object)};
            const auto child = ::fork();
            expect(child >= 0, "fork distinct authenticated process capability consumer");
            if (!child) {
                // Fresh exec avoids inheriting another thread's active runtime
                // allocations and permits a meaningful normal-exit LSan check.
                ::execl(executable.c_str(), executable.c_str(), "--foreign", arguments[0].c_str(),
                        arguments[1].c_str(), arguments[2].c_str(), arguments[3].c_str(),
                        arguments[4].c_str(), arguments[5].c_str(), arguments[6].c_str(),
                        arguments[7].c_str(), arguments[8].c_str(), static_cast<char*>(nullptr));
                ::_exit(127);
            }
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "foreign process cannot borrow parent source or communicator authority");
            expect(!provider.calls[8],
                   "cross-process token guesses never enter any collective callback");
            endpoint.call(parent, ExecutionNcclDestroy{communicator});
            endpoint.close();
            source.close();
        },
        [](ExecutionService&) {}, ExecutionSchedulingMode::kPriority, false,
        PeerAdmission::kAuthenticatedProcesses);
    expect(provider.calls[1] == 2 && provider.calls[4] == 2,
           "each process's independently authenticated communicator retires once");
}
void bounded_communicator_capacity() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{89, 97}};
            const auto uid = endpoint.uid(caller);
            std::vector<ExecutionNcclCommunicator> communicators;
            communicators.reserve(64);
            for (unsigned index = 0; index != 64; ++index)
                communicators.push_back(endpoint.initialize(caller, uid));
            endpoint.rejected(caller, ExecutionNcclInitRank{true, 1, 0, uid});
            expect(provider.calls[1] == 64,
                   "bounded communicator quota rejects before original init");
            const auto observed =
                endpoint.call(caller, ExecutionNcclGetAsyncError{communicators.front(), true});
            expect(observed.envelope.observation.async_error == 0,
                   "full normal communicator capacity cannot starve required progress query");
            endpoint.call(caller, ExecutionNcclAbort{communicators.front()});
            communicators.erase(communicators.begin());
            const auto replacement = endpoint.initialize(caller, uid);
            for (const auto communicator : communicators)
                endpoint.call(caller, ExecutionNcclDestroy{communicator});
            endpoint.call(caller, ExecutionNcclDestroy{replacement});
            endpoint.close();
        },
        [](ExecutionService&) {});
    expect(provider.calls[1] == 65 && provider.calls[3] == 1 && provider.calls[5] == 1 &&
               provider.calls[4] == 64,
           "normal-capacity rejection, reserved progress and exact post-retirement quota reuse are "
           "real");
}
void profile_physical_isolation() {
    std::atomic<bool> waiting{}, observed_denial{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            Endpoint endpoint(path, peer);
            Caller owner{{101, 103}}, unrelated{{107, 109}};
            const auto uid = endpoint.uid(owner);
            owner.context = source.client.session();
            const auto communicator = endpoint.initialize(owner, uid);
            source.fill(send, stream, 11.5F);
            provider.stop_after_collective = true;
            endpoint.call(owner, collective(communicator, source, send, receive, stream));
            const auto submission =
                endpoint.client.submit(ExecutionOpcode::kNcclCall,
                                       endpoint.request(unrelated, ExecutionNcclGetUniqueId{true}));
            expect(submission.status == ExecutionSocketSubmitStatus::kAccepted,
                   "unrelated canonical SDK request enters actual transport while GPU tail remains "
                   "unproven");
            expect(endpoint.client.poll_for(0),
                   "flush unrelated request to the real authenticated socket");
            waiting = true;
            await([&] {
                expect(endpoint.client.poll_for(0),
                       "progress actual delayed request transport without execution authority");
                expect(!endpoint.client.take_reply(submission.sequence),
                       "profile isolation denies unrelated SDK issue while physical work remains");
                return observed_denial.load();
            });
            provider.stop_after_collective = false;
            provider.events_ready = true;
            std::optional<ExecutionWireReply> result;
            await([&] {
                expect(endpoint.client.poll_for(0), "poll real policy-delayed native receipt");
                result = endpoint.client.take_reply(submission.sequence);
                return result.has_value();
            });
            const auto granted = endpoint.decode(std::move(result));
            expect(
                granted.envelope.is_accepted && granted.envelope.original_status == 0 &&
                    !granted.outer.status,
                "profile isolation permits unrelated Core issue only after actual physical stop");
            source.verify(receive, {11.5F});
            endpoint.call(owner, ExecutionNcclDestroy{communicator});
            endpoint.close();
            source.close();
        },
        [&](ExecutionService&) {
            if (!waiting.load() || observed_denial.load()) return;
            expect(provider.calls[0] == 1 && !provider.events_ready.load(),
                   "read-only runtime observation and an absent Core grant never invoke unrelated "
                   "SDK work");
            if (++provider.denied_steps == 32) observed_denial = true;
        },
        ExecutionSchedulingMode::kProfile);
    expect(provider.calls[0] == 2 && provider.calls[8] == 1,
           "held physical owner neither loses its tail nor replays work before later authorized "
           "issue");
}
void priority_equal_tail_overlap() {
    std::atomic<bool> granted{}, observed_overlap{};
    unsigned observed_steps{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto send = source.allocate(), receive = source.allocate();
            source.upload(send);
            source.upload(receive);
            const auto stream = source.stream(CU_STREAM_NON_BLOCKING);
            Endpoint endpoint(path, peer);
            Caller owner{{113, 127}}, unrelated{{131, 137}};
            const auto uid = endpoint.uid(owner);
            owner.context = source.client.session();
            const auto communicator = endpoint.initialize(owner, uid);
            source.fill(send, stream, 12.5F);
            provider.stop_after_collective = true;
            endpoint.call(owner, collective(communicator, source, send, receive, stream));
            const auto overlapping = endpoint.call(unrelated, ExecutionNcclGetUniqueId{true});
            expect(overlapping.envelope.observation.effects == 1 && !provider.events_ready.load(),
                   "equal-priority unrelated UID receives a real Core grant before physical tail "
                   "completion");
            granted = true;
            await([&] { return observed_overlap.load(); });
            provider.stop_after_collective = false;
            provider.events_ready = true;
            source.verify(receive, {12.5F});
            endpoint.call(owner, ExecutionNcclDestroy{communicator});
            endpoint.close();
            source.close();
        },
        [&](ExecutionService&) {
            if (!granted.load() || observed_overlap.load()) return;
            expect(provider.calls[0] == 2 && !provider.events_ready.load(),
                   "same-priority grant does not fabricate completion of another physical tail");
            if (++observed_steps == 32) observed_overlap = true;
        });
    expect(provider.calls[0] == 2 && provider.calls[8] == 1,
           "equal-priority overlap and later physical cleanup each execute exactly once");
}
void private_source_boundary() {
    std::atomic<bool> waiting{}, observed_denial{};
    unsigned observation_steps{}, first_queries{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            Source source(path, peer);
            const auto allocation = source.allocate();
            provider.events_ready = false;
            source.upload(allocation);
            const auto ordinary_is_empty = [](const ExecutionWireReply& reply) {
                ExecutionPayloadReader input(reply.data);
                std::array<std::uint64_t, 7> fields{};
                for (auto& field : fields)
                    expect(input.read_u64(field), "complete actual source snapshot field");
                std::uint32_t quarantine{};
                expect(input.read_u32(quarantine) && input.empty() && !quarantine && !fields[0] &&
                           !fields[1] && !fields[2] && fields[3] == 64,
                       "held synchronous upload has zero ordinary submissions or retained "
                       "parameters but an exact live allocation");
            };
            ordinary_is_empty(source.complete(source.client.exchange(ExecutionOpcode::kSnapshot),
                                              "source.Snapshot"));
            const auto drain = source.client.submit(ExecutionOpcode::kDrain);
            expect(drain.status == ExecutionSocketSubmitStatus::kAccepted &&
                       drain.sequence == source.next_sequence++,
                   "submit exact source Drain while only a private physical boundary remains");
            expect(source.client.poll_for(0),
                   "flush private-boundary Drain without blocking owner");
            auto native = std::async(std::launch::async, [path, peer] {
                Endpoint endpoint(path, peer);
                Caller caller{{139, 149}};
                (void)endpoint.uid(caller);
                endpoint.close();
            });
            const auto marker = source.client.submit(ExecutionOpcode::kSnapshot);
            expect(marker.status == ExecutionSocketSubmitStatus::kAccepted &&
                       marker.sequence == source.next_sequence++,
                   "metadata request shares the transport with a physically blocked Drain");
            std::optional<ExecutionWireReply> marker_reply;
            await([&] {
                expect(source.client.poll_for(0), "owner and transport remain nonblocking");
                expect(!source.client.take_reply(drain.sequence),
                       "Drain cannot report completion from zero ordinary submissions alone");
                marker_reply = source.client.take_reply(marker.sequence);
                return marker_reply.has_value();
            });
            ordinary_is_empty(success(std::move(marker_reply), "source.SnapshotAfterDrain",
                                      source.client, marker.sequence));
            waiting = true;
            await([&] {
                expect(source.client.poll_for(0), "held Drain keeps healthy transport progress");
                expect(
                    !source.client.take_reply(drain.sequence) &&
                        native.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready,
                    "profile denies NCL1 bootstrap and Drain while source private work is "
                    "physically unproven");
                return observed_denial.load();
            });
            provider.events_ready = true;
            std::optional<ExecutionWireReply> drained;
            await([&] {
                expect(source.client.poll_for(0), "poll Drain after exact event readiness");
                drained = source.client.take_reply(drain.sequence);
                return drained.has_value();
            });
            (void)success(std::move(drained), "source.Drain", source.client, drain.sequence);
            native.get();
            // Every uploaded word is the guard pattern, including the interior.
            source.verify(allocation, {std::bit_cast<float>(kGuard)});
            source.fill(allocation, 0, 14.5F);
            source.verify(allocation, {14.5F});
            source.close();
        },
        [&](ExecutionService&) {
            if (!waiting.load() || observed_denial.load()) return;
            expect(!provider.events_ready.load() &&
                       provider.event_records.load() > provider.event_retires.load() &&
                       std::ranges::all_of(provider.calls,
                                           [](const auto& count) { return !count.load(); }),
                   "an actual source-only private boundary reserves physical work without an "
                   "NCCL batch or callback");
            if (!observation_steps) first_queries = provider.event_queries.load();
            if (++observation_steps == 32) {
                expect(provider.event_queries.load() > first_queries,
                       "owner polls the actual private event while Drain and new work wait");
                observed_denial = true;
            }
        },
        ExecutionSchedulingMode::kProfile);
    expect(provider.calls[0] == 1 && provider.event_records == provider.event_retires,
           "physical readiness releases Drain and later NCL1 grant with exact event cleanup");
}
void blocked_callback_disconnect() {
    std::atomic<bool> disconnected{};
    run(
        [&](const std::string& path, TaskPeerIdentity peer) {
            auto endpoint = std::make_unique<Endpoint>(path, peer);
            Caller caller{{113, 127}};
            (void)endpoint->uid(caller);
            provider.uid_returned = false;
            provider.uid_entered = false;
            provider.release_uid = false;
            provider.hold_uid = true;
            const auto submitted =
                endpoint->client.submit(ExecutionOpcode::kNcclCall,
                                        endpoint->request(caller, ExecutionNcclGetUniqueId{true}));
            expect(submitted.status == ExecutionSocketSubmitStatus::kAccepted,
                   "actual held native callback is submitted once");
            await([&] {
                expect(endpoint->client.poll_for(0), "send held request before disconnect");
                return provider.uid_entered.load();
            });
            endpoint.reset();
            disconnected = true;
            await([] { return provider.uid_returned.load(); });
        },
        [&](ExecutionService& service) {
            if (!disconnected.load() || provider.release_uid.load()) return;
            expect(
                !service.finished() && !provider.primary_releases && provider.calls[0] == 2 &&
                    !provider.calls[4] && !provider.calls[5],
                "socket disconnect cannot cancel an entered SDK call or destroy its primary owner");
            if (++provider.denied_steps == 32) provider.release_uid = true;
        });
    expect(provider.calls[0] == 2, "known eventual native return permits cleanup without replay");
}
void open_group_disconnect() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            auto endpoint = std::make_unique<Endpoint>(path, peer);
            Caller caller{{131, 137}};
            (void)endpoint->uid(caller);
            endpoint->call(caller, ExecutionNcclGroupStart{});
            endpoint.reset();
        },
        [](ExecutionService&) {}, ExecutionSchedulingMode::kPriority, true);
    expect(provider.calls[6] == 1 && !provider.calls[7],
           "disconnect keeps actual open native lane TLS; it never fabricates GroupEnd");
}
void unknown_context_retention() {
    run(
        [](const std::string& path, TaskPeerIdentity peer) {
            Endpoint endpoint(path, peer);
            Caller caller{{139, 149}};
            provider.corrupt_uid_context = true;
            const auto result = endpoint.decode(
                endpoint.client.exchange(ExecutionOpcode::kNcclCall,
                                         endpoint.request(caller, ExecutionNcclGetUniqueId{true})));
            expect(result.envelope.is_accepted && result.envelope.original_status == 0 &&
                       result.outer.status && result.outer.parameter_bytes == 1 &&
                       result.envelope.observation.effects == 1 &&
                       std::ranges::all_of(result.envelope.observation.unique_id_mask,
                                           [](auto value) { return value == std::byte{255}; }) &&
                       std::ranges::all_of(result.envelope.observation.unique_id,
                                           [](auto value) { return value == std::byte{0x6d}; }),
                   "proved native UID write/status survive a later unknown current-context proof");
        },
        [](ExecutionService&) {}, ExecutionSchedulingMode::kPriority, true);
    expect(provider.calls[0] == 1,
           "unknown native/proof ownership is retained without hidden recovery SDK calls");
}
void isolated(void (*scenario)(), const char* name, bool retains_native_owners = false) {
    const auto child = ::fork();
    expect(child >= 0, "fork fresh isolated runtime fixture");
    if (!child) {
        scenario();
        // Only explicitly marked unknown/disconnected cases retain native
        // ownership. Every healthy scenario must run global/TLS/LSan exit checks.
        if (retains_native_owners) std::_Exit(EXIT_SUCCESS);
        std::exit(EXIT_SUCCESS);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           name);
}
}  // namespace
int main(int argc, char** argv) {
    if (argc == 3 && std::string_view{argv[1]} == "--semantic") {
        audits_semantic = true;
        bool retains_native_owners{};
        const std::string_view scenario{argv[2]};
        if (scenario == "ordinary")
            ordinary();
        else if (scenario == "deferred")
            ordinary_deferred();
        else if (scenario == "foreign")
            foreign_chain_live();
        else if (scenario == "host")
            host_wrapped_close_deferred();
        else if (scenario == "host-high")
            host_high_arrival();
        else if (scenario == "async")
            asynchronous();
        else if (scenario == "async-two")
            asynchronous_two_communicators();
        else if (scenario == "host-function")
            ordinary_host_function_continuation();
        else if (scenario == "async-source-cut")
            asynchronous_source_cut();
        else if (scenario == "host-function-disconnect") {
            retains_native_owners = true;
            host_function_source_disconnect();
        } else if (scenario == "host-function-query-error") {
            retains_native_owners = true;
            host_function_query_error();
        } else {
            std::cerr << "Unknown semantic-audit fixture scenario\n";
            return EXIT_FAILURE;
        }
        std::cout << "Actual semantic service audit scenario passed\n";
        if (retains_native_owners) {
            std::cout.flush();
            std::_Exit(EXIT_SUCCESS);
        }
        return EXIT_SUCCESS;
    }
    if (argc != 1) {
        expect((argc == 6 && std::string_view{argv[1]} == "--high") ||
                   (argc == 11 && std::string_view{argv[1]} == "--foreign"),
               "known bounded clean-process fixture entry");
        const auto number = [](const char* input) {
            const std::string_view text{input};
            std::uint64_t value{};
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
            expect(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size(),
                   "exact trusted numeric foreign-client argument");
            return value;
        };
        const auto pid = number(argv[3]);
        expect(pid && pid <= UINT32_MAX, "bounded authenticated service PID");
        if (argc == 6 && std::string_view{argv[1]} == "--high") {
            const auto descriptor = number(argv[5]);
            expect(descriptor <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
                   "bounded exact inherited high-peer signal descriptor");
            high_client(argv[2],
                        {static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()),
                         number(argv[4])},
                        static_cast<int>(descriptor));
            return EXIT_SUCCESS;
        }
        expect(argc == 11 && std::string_view{argv[1]} == "--foreign",
               "known clean foreign-client entry");
        foreign_client(argv[2],
                       {static_cast<std::uint32_t>(pid), static_cast<std::uint32_t>(::geteuid()),
                        number(argv[4])},
                       {.source = number(argv[5]),
                        .send = number(argv[6]),
                        .receive = number(argv[7]),
                        .stream = number(argv[8]),
                        .parent = {number(argv[9]), number(argv[10])}});
        return EXIT_SUCCESS;
    }
    isolated(private_source_boundary,
             "actual private source boundary holds Drain and profile admission until stop");
    isolated(ordinary, "all eleven actual service routes and inter-inner source ordering pass");
    isolated(ordinary_deferred,
             "all eleven actual service routes preserve deferred outer-group source ordering");
    isolated(ordinary_high_arrival,
             "higher-priority arrival after Start permits exact ordinary progress before End");
    isolated(asynchronous_source_cut,
             "actual NCCL collective and End cuts prepare only their exact async prefixes");
    isolated(
        ordinary_host_function_continuation,
        "SCN1 host functions retain later source work while unwrapped helpers cross its fence");
    isolated(host_function_source_disconnect,
             "disconnected unpermitted source callbacks retain already accepted native dependents",
             true);
    isolated(host_function_query_error,
             "async-only callback query errors retain already accepted native dependents", true);
    isolated(host_high_arrival,
             "actual HST1 Deferred cleanup survives End and authenticated higher-priority demand");
    isolated(staging_host_continuations,
             "actual STG1 transfer fences, full readback and closed-origin cleanup survive End");
    isolated(pre_group_host_release,
             "ordinary in-group release preserves an unwrapped pre-group host import");
    isolated(host_close_after_end,
             "explicit helper Close after End waits its real mapping lease and retires parent");
    isolated(host_import_reply_loss,
             "disconnected successful import consumes its worker result and retires its parent");
    isolated(host_wrapped_close_deferred,
             "wrapped terminal Close consumes Deferred and retries within the live group");
    isolated(host_wrapped_close_retry_disconnect,
             "disconnected accepted wrapped-Close retry retains owners until actual cleanup");
    isolated(mixed_group_rejections,
             "both invalid mixed-group orders reject while valid End and later use survive");
    isolated(foreign_registry_bounds,
             "invalid foreign registry capacities reject before any native callback");
    isolated(foreign_chain_live,
             "actual three-source Priority lifecycle preserves old foreign generations");
    isolated(foreign_chain_disconnect,
             "actual three-source foreign registry retains a disconnected source owner");
    isolated(foreign_edge_quota_and_low_events,
             "actual bounded foreign edges refund through configured-low event accounting");
    isolated(foreign_record_failure,
             "actual source record error rejects its NCCL prefix without invoking the collective");
    isolated(deferred_event_admission_fifo,
             "actual full-queue event deferral preserves same-connection admission FIFO");
    isolated(asynchronous, "actual service async query/progress/final-tail lifecycle passes");
    isolated(capabilities_and_sequences,
             "actual service validates exact capability and caller sequences");
    isolated(foreign_process_capability,
             "actual service rejects foreign-process resource authority");
    isolated(bounded_communicator_capacity,
             "actual bounded communicator quota preserves progress and reuse");
    isolated(profile_physical_isolation,
             "actual Core profile isolation denies new issue until physical stop");
    isolated(priority_equal_tail_overlap,
             "actual Core priority mode permits equal-priority issue across a held tail");
    isolated(blocked_callback_disconnect,
             "entered native call survives socket disconnect until actual return");
    isolated(open_group_disconnect, "disconnected open native group remains safely retained", true);
    isolated(unknown_context_retention,
             "actual output survives unknown context while owners stay retained", true);
    std::cout << "Actual NCCL service socket integration: 30 groups passed\n";
}
