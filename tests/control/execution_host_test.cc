#include "runtime.h"
#include "host_owner.h"
#include "host_stage_metrics.h"
#include "glimmer/control/execution_process_memory_protocol.h"
#include "glimmer/core/execution_priority.h"
#include "internal/process_identity.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Context {
    std::map<CUdeviceptr, std::unique_ptr<std::byte[]>> memory;
    void* registration{};
};
thread_local Context* current_context;
struct Native {
    std::atomic<unsigned int> creates{}, destroys{}, registrations{}, unregistrations{}, launches{},
        copies{}, event_destroys{}, event_creates{}, stopped_prefix{}, unregister_attempts{},
        destroy_attempts{};
    std::atomic<bool> has_registration_wait{}, may_register{}, may_stop{}, observes_shutdown{};
    std::atomic<unsigned int> shutdown_proofs{};
    std::atomic<int> unregister_status{}, destroy_status{};
} native;
CUresult CUDAAPI initialize(unsigned int) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI device_get(CUdevice* result, int ordinal) {
    *result = ordinal;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI get_current(CUcontext* result) {
    *result = reinterpret_cast<CUcontext>(current_context);
    if (native.observes_shutdown && current_context && current_context->registration)
        ++native.shutdown_proofs;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI create_context(CUcontext* result, unsigned int, CUdevice) {
    expect(!current_context, "each native factory starts on an empty thread");
    current_context = new Context;
    *result = reinterpret_cast<CUcontext>(current_context);
    ++native.creates;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI set_context(CUcontext value) {
    current_context = reinterpret_cast<Context*>(value);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI destroy_context(CUcontext value) {
    auto* context = reinterpret_cast<Context*>(value);
    expect(context && context == current_context && !context->registration,
           "only its stable owner destroys an unregistered context");
    ++native.destroy_attempts;
    if (native.destroy_status) return static_cast<CUresult>(native.destroy_status.load());
    delete context;
    current_context = nullptr;
    ++native.destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_load(CUmodule* result, const void*) {
    *result = reinterpret_cast<CUmodule>(current_context);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI module_unload(CUmodule) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI function_get(CUfunction* result, CUmodule, const char*) {
    *result = reinterpret_cast<CUfunction>(current_context);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI parameter_info(CUfunction, std::size_t, std::size_t*, std::size_t*) {
    throw std::runtime_error("zero-argument native function needs no parameter probe");
}
CUresult CUDAAPI allocate(CUdeviceptr* result, std::size_t bytes) {
    auto storage = std::make_unique<std::byte[]>(bytes);
    *result = reinterpret_cast<CUdeviceptr>(storage.get());
    current_context->memory.emplace(*result, std::move(storage));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI release(CUdeviceptr address) {
    return current_context->memory.erase(address) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
void* cpu_storage(CUdeviceptr address) {
    // This fake SDK represents device addresses with its own actual CPU storage.
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    return reinterpret_cast<void*>(address);
}
CUresult CUDAAPI upload(CUdeviceptr destination, const void* source, std::size_t bytes) {
    std::memcpy(cpu_storage(destination), source, bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI download(void* destination, CUdeviceptr source, std::size_t bytes) {
    std::memcpy(destination, cpu_storage(source), bytes);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI create_stream(CUstream*, unsigned int) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI destroy_stream(CUstream) {
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI create_event(CUevent* result, unsigned int) {
    *result = reinterpret_cast<CUevent>(new unsigned int(++native.event_creates));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI record_event(CUevent, CUstream) {
    return CUDA_SUCCESS;
}
CUresult CUDAAPI query_event(CUevent event) {
    return native.may_stop.load() ||
                   *reinterpret_cast<unsigned int*>(event) <= native.stopped_prefix
               ? CUDA_SUCCESS
               : CUDA_ERROR_NOT_READY;
}
CUresult CUDAAPI destroy_event(CUevent event) {
    delete reinterpret_cast<unsigned int*>(event);
    ++native.event_destroys;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI synchronize(CUstream) {
    throw std::runtime_error(
        "host paths must not fabricate completion using stream synchronization");
}
CUresult CUDAAPI launch(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, CUstream, void**, void**) {
    ++native.launches;
    native.may_register.store(true);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_register(void* pointer, std::size_t bytes, unsigned int flags) {
    expect(current_context && !current_context->registration && bytes == 4096 &&
               flags == (CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
           "real native registration uses exact independently owned context/backing");
    native.has_registration_wait.store(true);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!native.may_register.load()) {
        expect(std::chrono::steady_clock::now() < deadline,
               "GPU owner must issue the independent registration-releasing kernel");
        std::this_thread::yield();
    }
    current_context->registration = pointer;
    ++native.registrations;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_unregister(void* pointer) {
    expect(current_context && pointer == current_context->registration,
           "unregister only the exact registration on its native actor");
    ++native.unregister_attempts;
    if (native.unregister_status) return static_cast<CUresult>(native.unregister_status.load());
    current_context->registration = nullptr;
    ++native.unregistrations;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI host_map(CUdeviceptr* result, void* pointer, unsigned int) {
    if (pointer != current_context->registration) return CUDA_ERROR_INVALID_VALUE;
    *result = reinterpret_cast<CUdeviceptr>(pointer);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI upload_async(CUdeviceptr destination, const void* source, std::size_t bytes,
                              CUstream) {
    ++native.copies;
    return upload(destination, source, bytes);
}
CUresult CUDAAPI download_async(void* destination, CUdeviceptr source, std::size_t bytes,
                                CUstream) {
    ++native.copies;
    return download(destination, source, bytes);
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch result{
        {get_current, launch, create_event, record_event, query_event, destroy_event, synchronize},
        initialize,
        device_get,
        create_context,
        set_context,
        destroy_context,
        module_load,
        module_unload,
        function_get,
        parameter_info,
        allocate,
        release,
        upload,
        download,
        create_stream,
        destroy_stream};
    result.host_map = host_map;
    result.host_register = host_register;
    result.host_unregister = host_unregister;
    result.upload_async = upload_async;
    result.download_async = download_async;
    return result;
}
ExecutionWireReply actual(std::optional<ExecutionWireReply> result) {
    expect(result && result->status == CUDA_SUCCESS && result->original_status == CUDA_SUCCESS &&
               result->disposition == ExecutionWireDisposition::kComplete,
           "actual target native SDK result, not admission acknowledgment");
    return std::move(*result);
}
std::vector<std::byte> by_id(std::uint64_t id) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(id), "encode scoped resource identity");
    return input.take();
}
std::vector<std::byte> launch_payload(std::uint64_t id) {
    ExecutionPayloadWriter input;
    expect(input.write_u64(id) && input.write_u64(0), "encode function and legacy stream");
    for (int axis = 0; axis < 6; ++axis) expect(input.write_u32(1), "encode native geometry");
    expect(input.write_u32(0) && input.write_blob({}), "encode original empty parameter ABI");
    return input.take();
}
void shared_host_service(const std::string& path, bool should_disconnect, bool is_staging = false,
                         bool closes_before_disconnect = false,
                         std::optional<bool> skips_helper_plans = std::nullopt) {
    for (auto* count : {&native.creates, &native.destroys, &native.registrations,
                        &native.unregistrations, &native.launches, &native.copies,
                        &native.event_destroys, &native.event_creates, &native.stopped_prefix})
        count->store(0);
    native.has_registration_wait = native.may_register = native.may_stop = false;
    native.observes_shutdown = false;
    native.shutdown_proofs = 0;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.transport.max_pending_requests = 4;
    config.exit_when_idle = true;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted native service identity");
    std::shared_ptr<HostStageMetrics> metrics;
    if (skips_helper_plans) {
        config.persistent.workers = 320;
        config.client_priorities = {{*identity, 0, ExecutionClientKernelMode::kOriginal}};
        config.should_skip_idle_helper_plans = *skips_helper_plans;
        metrics = std::make_shared<HostStageMetrics>();
        config.host_stage_metrics = metrics;
    }
    ExecutionService service(config, dispatch());
    expect(service.start(), "start authenticated service with native host actor");
    std::atomic<bool> has_requested_free{};
    std::atomic<bool> has_disconnected_close{};
    std::atomic<bool> has_requested_download{}, has_host_freed{};
    auto client = std::async(std::launch::async, [&] {
        ExecutionSocketClient application({path, *identity, 5000});
        actual(application.open());
        ExecutionPayloadWriter source;
        expect(source.write_string(".version 7.0\n.target sm_70\n.address_size 64\n"
                                   ".visible .entry kernel() { ret; }"),
               "encode original PTX");
        const auto module = actual(application.exchange(ExecutionOpcode::kLoadPtx, source.take()));
        ExecutionPayloadWriter lookup;
        expect(lookup.write_u64(module.id) && lookup.write_string("kernel"), "lookup original");
        const auto function =
            actual(application.exchange(ExecutionOpcode::kGetFunction, lookup.take()));
        const auto device = actual(application.exchange(ExecutionOpcode::kAllocate, by_id(256)));
        ExecutionSocketClient host({path, *identity, 5000});
        ExecutionPayloadWriter bootstrap;
        expect(bootstrap.write_u32(is_staging ? kExecutionStagingOwnerRole
                                              : kExecutionHostOwnerRole) &&
                   bootstrap.write_u32(CU_CTX_MAP_HOST),
               "host-only native role is explicit and separate from application stack");
        actual(host.open(bootstrap.take()));
        const auto forbidden = host.exchange(ExecutionOpcode::kLaunch, launch_payload(function.id));
        expect(forbidden && forbidden->status == CUDA_ERROR_NOT_SUPPORTED &&
                   !forbidden->original_status,
               "host owner cannot acquire application execution authority");
        ExecutionOwnedDescriptor backing(
            ::memfd_create("glimmer-host-service", MFD_ALLOW_SEALING | MFD_CLOEXEC));
        expect(
            backing.get() >= 0 && ::ftruncate(backing.get(), 4096) == 0 &&
                ::fcntl(backing.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
            "create size-sealed shared backing");
        void* mapping = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, backing.get(), 0);
        expect(mapping != MAP_FAILED, "application maps the same shared bytes");
        ExecutionPayloadWriter import;
        expect(import.write_u64(256) && import.write_u64(4096) && import.write_u32(0),
               "encode import");
        const auto pending =
            host.submit(ExecutionOpcode::kImportHostMemory, import.take(), backing.get());
        expect(pending.status == ExecutionSocketSubmitStatus::kAccepted && host.poll_for(0),
               "admit import");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!native.has_registration_wait.load()) {
            expect(host.poll_for(0) && std::chrono::steady_clock::now() < deadline,
                   "observe withheld registration");
            std::this_thread::yield();
        }
        actual(application.exchange(ExecutionOpcode::kLaunch, launch_payload(function.id)));
        std::optional<ExecutionWireReply> imported;
        while (!(imported = host.take_reply(pending.sequence)))
            expect(host.poll_for(1), "wait original import result");
        const auto resource = actual(std::move(imported));
        const auto idle_before =
            host.exchange(ExecutionOpcode::kQueryStagingIdle, by_id(resource.id));
        expect(idle_before &&
                   (is_staging ? idle_before->status == CUDA_SUCCESS &&
                                     idle_before->id == resource.id && !idle_before->original_status
                               : idle_before->status == CUDA_ERROR_NOT_SUPPORTED),
               "idle observation is private STG1 metadata, not a public host-memory operation");
        for (const auto invalid : {std::uint64_t{0}, UINT64_MAX}) {
            const auto rejected = host.exchange(ExecutionOpcode::kQueryStagingIdle, by_id(invalid));
            expect(rejected && rejected->status != CUDA_SUCCESS && !rejected->original_status,
                   "missing and zero mapping IDs cannot manufacture an idle receipt");
        }
        auto malformed_idle = by_id(resource.id);
        malformed_idle.push_back(std::byte{0});
        const auto malformed = host.exchange(ExecutionOpcode::kQueryStagingIdle, malformed_idle);
        expect(malformed && malformed->status != CUDA_SUCCESS && !malformed->original_status,
               "trailing idle-query input rejects without native work");
        const auto transfer = [&](std::uint64_t owner, std::uint64_t allocation,
                                  unsigned int direction, bool has_wrong_role = false) {
            ExecutionPayloadWriter input;
            expect(input.write_u64(owner) && input.write_u64(allocation) && input.write_u64(0) &&
                       input.write_u64(device.id) && input.write_u64(0) && input.write_u64(256) &&
                       input.write_u64(0) && input.write_u32(direction),
                   "encode identity-bound host transfer");
            return application.exchange(is_staging != has_wrong_role
                                            ? ExecutionOpcode::kStagedHostTransfer
                                            : ExecutionOpcode::kHostTransfer,
                                        input.take());
        };
        for (const auto owner : {application.session(), UINT64_MAX}) {
            const auto rejected = transfer(owner, resource.id, 1);
            expect(rejected && rejected->status == CUDA_ERROR_INVALID_VALUE &&
                       !rejected->original_status,
                   "wrong host-owner connection never claims a native copy");
        }
        const auto wrong_role = transfer(host.session(), resource.id, 1, true);
        expect(
            wrong_role && wrong_role->status == CUDA_ERROR_INVALID_VALUE &&
                !wrong_role->original_status,
            "public and private staging roles cannot be interchanged or acquire argument leases");
        std::memset(mapping, 37, 256);
        actual(transfer(host.session(), resource.id, 1));
        // The fake device is distinct storage. Changing it stands for an earlier
        // GPU result; subsequent H2D must read the ordered D2H result in place.
        std::memset(cpu_storage(device.address), 61, 256);
        has_requested_download.store(is_staging);
        actual(transfer(host.session(), resource.id, 0));
        if (is_staging)
            expect(native.stopped_prefix >= 3 && !native.may_stop,
                   "pageable download reply waits for its own actual event without a global drain");
        actual(transfer(host.session(), resource.id, 1));
        for (std::size_t index = 0; index < 256; ++index)
            expect(static_cast<const unsigned char*>(mapping)[index] == 61 &&
                       static_cast<const unsigned char*>(cpu_storage(device.address))[index] == 61,
                   "shared D2H-to-H2D alias dataflow has no early snapshot or copyback");
        actual(application.exchange(ExecutionOpcode::kLaunch, launch_payload(function.id)));
        std::optional<std::uint64_t> idle_sequence;
        if (is_staging && !should_disconnect) {
            const auto query = host.submit(ExecutionOpcode::kQueryStagingIdle, by_id(resource.id));
            expect(query.status == ExecutionSocketSubmitStatus::kAccepted,
                   "accept exact idle query after SDK transfer return");
            idle_sequence = query.sequence;
            for (unsigned int turn = 0; turn < 8; ++turn)
                expect(host.poll_for(1) && !host.take_reply(*idle_sequence) &&
                           native.unregistrations == 0 && native.stopped_prefix == 3,
                       "SDK copy result cannot authorize reuse while the physical event is held");
        }
        has_requested_free.store(true);
        if (should_disconnect) {
            if (closes_before_disconnect) {
                native.observes_shutdown = true;
                const auto closing = host.submit(ExecutionOpcode::kClose);
                expect(closing.status == ExecutionSocketSubmitStatus::kAccepted,
                       "accept ordinary no-parent Close with actual pending mapping leases");
                while (native.shutdown_proofs < 2) {
                    expect(host.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                           "actual second shutdown entry proves the first Deferred was consumed");
                }
                expect(!host.take_reply(closing.sequence) && native.unregistrations == 0,
                       "no-parent Close cannot complete before physical mapping lease retirement");
                has_disconnected_close = true;
            }
            expect(::munmap(mapping, 4096) == 0, "client unmaps independently before disconnect");
            return;
        }
        if (idle_sequence) {
            std::optional<ExecutionWireReply> idle;
            while (!(idle = host.take_reply(*idle_sequence)))
                expect(host.poll_for(1) && std::chrono::steady_clock::now() < deadline,
                       "bounded actual physical idle receipt");
            expect(idle->status == CUDA_SUCCESS && idle->id == resource.id &&
                       idle->disposition == ExecutionWireDisposition::kComplete &&
                       !idle->original_status && native.unregistrations == 0 && !native.may_stop &&
                       native.stopped_prefix == 4,
                   "idle receipt preserves registration and excludes unrelated later GPU work");
        }
        actual(host.exchange(ExecutionOpcode::kReleaseHostMemory, by_id(resource.id)));
        expect(native.unregistrations == 1, "native free waits through actual physical leases");
        if (is_staging)
            expect(!native.may_stop && native.stopped_prefix == 4,
                   "private staging lifetime excludes the unrelated subsequent kernel event");
        has_host_freed.store(true);
        const auto stale_idle =
            host.exchange(ExecutionOpcode::kQueryStagingIdle, by_id(resource.id));
        expect(stale_idle && stale_idle->status != CUDA_SUCCESS && !stale_idle->original_status,
               "released mapping never authorizes reuse");
        const auto stale = transfer(host.session(), resource.id, 1);
        expect(stale && stale->status == CUDA_ERROR_INVALID_VALUE && !stale->original_status,
               "released resource rejects without native work");
        const auto closed = host.exchange(ExecutionOpcode::kClose);
        expect(closed && closed->status == CUDA_SUCCESS && !closed->original_status,
               "host close settles multiple cleanup operations, not a fabricated original");
        expect(::munmap(mapping, 4096) == 0, "client mapping is independently owned");
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    unsigned int pending_steps{};
    unsigned int download_steps{};
    while (!service.finished()) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded real service progress");
        if (has_requested_download && !has_requested_free && ++download_steps == 20) {
            expect(native.copies == 2 && native.unregistrations == 0,
                   "native download SDK return is not physical completion or copyback permission");
            native.stopped_prefix.store(3);
        }
        if (has_requested_free.load() && !native.may_stop) {
            expect((is_staging || native.unregistrations == 0) && native.destroys == 0,
                   "reply/disconnect cannot release physically outstanding host users");
            if (++pending_steps >= 20 &&
                (!closes_before_disconnect || has_disconnected_close.load())) {
                if (is_staging)
                    native.stopped_prefix.store(4);
                else
                    native.may_stop.store(true);
            }
        }
        if (is_staging && has_host_freed) native.may_stop.store(true);
    }
    client.get();
    expect(native.launches == 2 && native.copies == 3 && native.registrations == 1 &&
               native.unregistrations == 1 && native.creates == 2 && native.destroys == 2 &&
               native.event_destroys == 5 && service.request_stop(),
           "native contexts, registration, commands and trailing events retire exactly once");
    if (metrics) {
        std::uint64_t skipped{}, physical_samples{};
        for (std::size_t index = 0; index < HostStageMetrics::bucket_count; ++index) {
            const auto& bucket = metrics->bucket(index);
            skipped += bucket.slot_plans[static_cast<std::size_t>(
                HostStageMetrics::SlotPlanReason::kSkippedHelper)];
            // The compute connection opens first, followed by its host helper.
            physical_samples += bucket.queues[1].physical;
        }
        expect(metrics->valid() && physical_samples && (skipped != 0) == *skips_helper_plans,
               "real helper reservation remains globally observed with the exact opt-in skip "
               "behavior");
    }
}
void host_actor_guards_and_deferred_shutdown() {
    native.may_register = true;
    HostOwner owner({101, 102}, 103, dispatch(), {}, 0);
    auto foreign = std::async(std::launch::async, [&] {
        return !owner.submit({.sequence = 1}) && owner.snapshot().session.is_quarantined &&
               !owner.borrow({1, 0}, 1) && !owner.allocation_idle(1) &&
               !owner.permit_shutdown(71) && !owner.take_shutdown_reply(71);
    });
    expect(foreign.get(), "foreign root thread rejects before actor state access");
    expect(owner.submit({.sequence = 1, .flags = CU_CTX_MAP_HOST}), "grant one native factory");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    const auto await = [&](std::uint64_t sequence) {
        std::optional<HostOwnerReply> result;
        while (!(result = owner.take_reply(sequence))) {
            expect(std::chrono::steady_clock::now() < deadline, "bounded native actor result");
            std::this_thread::yield();
        }
        return *result;
    };
    const auto initialized = await(1);
    expect(initialized.result.status == CUDA_SUCCESS &&
               initialized.ownership == ExecutionBootstrapOwnership::kRetained,
           "actor factory reports independently retained native ownership");
    ExecutionOwnedDescriptor backing(
        ::memfd_create("glimmer-host-actor", MFD_ALLOW_SEALING | MFD_CLOEXEC));
    expect(backing.get() >= 0 && ::ftruncate(backing.get(), 4096) == 0 &&
               ::fcntl(backing.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
           "create bounded actor backing");
    expect(owner.submit({.operation = HostOwnerOperation::kImport,
                         .sequence = 2,
                         .bytes = 256,
                         .mapping_bytes = 4096},
                        backing.get()),
           "grant import");
    expect(!owner.submit({.sequence = 3}) && !owner.take_reply(3),
           "single native flight/unconsumed result cannot be overwritten or reaped by another "
           "identity");
    const auto imported = await(2);
    expect(imported.result.status == CUDA_SUCCESS,
           "actor publishes successful native host metadata");
    expect(owner.allocation_idle(imported.result.id) == true && !owner.allocation_idle(0) &&
               !owner.allocation_idle(UINT64_MAX),
           "only exact live unborrowed mappings can authorize reuse");
    auto view = owner.borrow({imported.result.id, 0}, 256);
    expect(owner.allocation_idle(imported.result.id) == false,
           "SDK registration success is not idle while an independent physical lease remains");
    view.reset();
    expect(owner.allocation_idle(imported.result.id) == true,
           "idle observes actual final lease retirement without unregistering");
    auto retained = owner.lifetimes();
    expect(owner.allocation_idle(imported.result.id) == false,
           "aggregate lifetime borrowers also prevent reuse");
    retained.clear();
    view = owner.borrow({imported.result.id, 0}, 256);
    expect(view && !owner.borrow({imported.result.id, 255}, 2),
           "root metadata validates borrow range");
    const auto child = ::fork();
    expect(child >= 0, "fork with an idle native host actor");
    if (child == 0)
        ::_exit(!owner.submit({.sequence = 3}) && owner.snapshot().session.is_quarantined &&
                        !owner.borrow({imported.result.id, 0}, 1) &&
                        !owner.allocation_idle(imported.result.id) && !owner.permit_shutdown(71) &&
                        !owner.take_shutdown_reply(71)
                    ? EXIT_SUCCESS
                    : EXIT_FAILURE);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "fork child rejects before inherited locks or native session access");
    expect(owner.submit({.operation = HostOwnerOperation::kRelease,
                         .sequence = 3,
                         .allocation = imported.result.id}),
           "grant explicit free");
    expect(!owner.borrow({imported.result.id, 0}, 1) && !owner.allocation_idle(imported.result.id),
           "seal root metadata before worker free starts");
    const auto deferred = await(3);
    expect(deferred.result.disposition == CudaSessionDisposition::kDeferred &&
               !deferred.result.original_status && !owner.snapshot().has_reservation,
           "physical-lease-deferred free relinquishes native reservation without any SDK result");
    using namespace glimmer::core;
    const std::array<ExecutionPriorityPeer, 2> peers{
        {{103, 1, true, owner.snapshot().has_reservation, false, false},
         {104, 2, true, false, false, false}}};
    const auto plan = plan_execution_priority(peers);
    expect(plan.is_valid && plan.decisions[1].action == ExecutionPriorityAction::kExecute,
           "deferred low host free cannot block the higher GPU command needed to retire its lease");
    owner.close();
    expect(!owner.borrow({imported.result.id, 0}, 1) &&
               !owner.allocation_idle(imported.result.id) && !owner.submit({.sequence = 4}),
           "close seals metadata and stops admission without releasing a live strong view");
    const auto before = native.unregistrations.load();
    owner.permit_shutdown();
    while (owner.snapshot().has_reservation) {
        expect(std::chrono::steady_clock::now() < deadline,
               "deferred actor shutdown does not block root");
        std::this_thread::yield();
    }
    expect(!owner.snapshot().is_closed && native.unregistrations == before,
           "cleanup cannot unregister retained external users");
    expect(!owner.permit_shutdown(0) && !owner.take_shutdown_reply(71) && owner.permit_shutdown(71),
           "explicit shutdown requires a nonzero exact accepted sequence");
    while (owner.snapshot().has_reservation) {
        expect(std::chrono::steady_clock::now() < deadline,
               "explicit shutdown publishes actual Deferred without blocking owner");
        std::this_thread::yield();
    }
    expect(!owner.snapshot().is_closed && !owner.permit_shutdown(72) &&
               !owner.take_shutdown_reply(72) && native.unregistrations == before,
           "idle reservation is not consumed-result proof or another shutdown grant");
    owner.close();
    owner.permit_shutdown();
    const auto shutdown_deferred = owner.take_shutdown_reply(71);
    expect(shutdown_deferred &&
               shutdown_deferred->disposition == CudaSessionDisposition::kDeferred &&
               shutdown_deferred->status == CUDA_ERROR_NOT_READY &&
               !shutdown_deferred->original_status && !owner.take_shutdown_reply(71) &&
               native.unregistrations == before,
           "Close and legacy permission cannot discard or overwrite the exact Deferred result");
    view.reset();
    expect(owner.permit_shutdown(72), "fresh explicit grant follows actual Deferred consumption");
    std::optional<CudaSessionResult> shutdown_completed;
    while (!(shutdown_completed = owner.take_shutdown_reply(72))) {
        expect(std::chrono::steady_clock::now() < deadline, "bounded final actor cleanup");
        std::this_thread::yield();
    }
    expect(shutdown_completed->status == CUDA_SUCCESS &&
               shutdown_completed->disposition == CudaSessionDisposition::kComplete &&
               owner.snapshot().is_closed && native.unregistrations == before + 1 &&
               !owner.take_shutdown_reply(72) && !owner.permit_shutdown(73),
           "exact final receipt proves cleanup once, cannot replay or accept another grant");
}
void host_publication_wakes_after_normal_cleanup() {
    native.may_register = true;
    const auto creates = native.creates.load(), destroys = native.destroys.load();
    const auto registrations = native.registrations.load();
    const auto unregistrations = native.unregistrations.load();
    auto wake = std::make_shared<ExecutionSocketWake>();
    const auto receives_wake = [&] {
        pollfd observed{wake->descriptor(), POLLIN, 0};
        const auto ready = ::poll(&observed, 1, 100);
        const bool is_ready = ready == 1 && observed.revents == POLLIN;
        expect(ready >= 0 && wake->consume(), "bounded normal worker notification drain");
        return is_ready;
    };
    bool initialized_wake{}, imported_wake{}, released_wake{}, shutdown_wake{};
    {
        HostOwner owner({301, 302}, 303, dispatch(), {}, 0, wake);
        expect(wake->notify() && !owner.take_reply(1) && !owner.take_shutdown_reply(1) &&
                   wake->consume(),
               "a CPU wake cannot invent a host mailbox result");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
        const auto await = [&](std::uint64_t sequence) {
            std::optional<HostOwnerReply> result;
            while (!(result = owner.take_reply(sequence))) {
                expect(std::chrono::steady_clock::now() < deadline, "bounded actual host result");
                std::this_thread::yield();
            }
            expect(result->result.status == CUDA_SUCCESS, "normal native host result unchanged");
            return *result;
        };
        expect(owner.submit({.sequence = 1, .flags = CU_CTX_MAP_HOST}), "grant actual host setup");
        (void)await(1);
        initialized_wake = receives_wake();
        ExecutionOwnedDescriptor backing(
            ::memfd_create("glimmer-host-wake", MFD_ALLOW_SEALING | MFD_CLOEXEC));
        expect(backing.get() >= 0 && ::ftruncate(backing.get(), 4096) == 0 &&
                   ::fcntl(backing.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) ==
                       0 &&
                   owner.submit({.operation = HostOwnerOperation::kImport,
                                 .sequence = 2,
                                 .bytes = 256,
                                 .mapping_bytes = 4096},
                                backing.get()),
               "grant exact bounded host import");
        const auto imported = await(2);
        imported_wake = receives_wake();
        expect(owner.submit({.operation = HostOwnerOperation::kRelease,
                             .sequence = 3,
                             .allocation = imported.result.id}),
               "grant actual release");
        (void)await(3);
        released_wake = receives_wake();
        owner.close();
        expect(owner.permit_shutdown(4), "grant exact native shutdown");
        std::optional<CudaSessionResult> shutdown;
        while (!(shutdown = owner.take_shutdown_reply(4))) {
            expect(std::chrono::steady_clock::now() < deadline, "bounded actual native shutdown");
            std::this_thread::yield();
        }
        expect(shutdown->status == CUDA_SUCCESS &&
                   shutdown->disposition == CudaSessionDisposition::kComplete,
               "shutdown wake cannot replace exact physical cleanup result");
        shutdown_wake = receives_wake();
    }
    expect(native.creates == creates + 1 && native.destroys == destroys + 1 &&
               native.registrations == registrations + 1 &&
               native.unregistrations == unregistrations + 1,
           "host wake regression finishes normal native/context cleanup before assertion");
    expect(initialized_wake && imported_wake && released_wake && shutdown_wake,
           "every actual host mailbox publication must wake the waiting owner");
}
void explicit_shutdown_failure_retains(bool fails_unregister) {
    const auto child = ::fork();
    expect(child >= 0, "isolate intentional unknown host shutdown retention");
    if (child == 0) {
        native.may_register = true;
        HostOwner owner({201, 202}, 203, dispatch(), {}, 0);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
        const auto await_reply = [&](std::uint64_t sequence) {
            std::optional<HostOwnerReply> reply;
            while (!(reply = owner.take_reply(sequence))) {
                expect(std::chrono::steady_clock::now() < deadline,
                       "bounded real host mailbox setup before failed shutdown");
                std::this_thread::yield();
            }
            expect(reply->result.status == CUDA_SUCCESS, "successful setup before failure");
        };
        expect(owner.submit({.sequence = 1, .flags = CU_CTX_MAP_HOST}),
               "accept retained failure factory");
        await_reply(1);
        ExecutionOwnedDescriptor backing(
            ::memfd_create("glimmer-host-shutdown-failure", MFD_ALLOW_SEALING | MFD_CLOEXEC));
        expect(
            backing.get() >= 0 && ::ftruncate(backing.get(), 4096) == 0 &&
                ::fcntl(backing.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
            "sealed native registration before shutdown failure");
        expect(owner.submit({.operation = HostOwnerOperation::kImport,
                             .sequence = 2,
                             .bytes = 256,
                             .mapping_bytes = 4096},
                            backing.get()),
               "accept actual import before shutdown failure");
        await_reply(2);
        const auto unregisters = native.unregister_attempts.load();
        const auto destroys = native.destroy_attempts.load();
        if (fails_unregister)
            native.unregister_status = CUDA_ERROR_INVALID_VALUE;
        else
            native.destroy_status = CUDA_ERROR_INVALID_VALUE;
        owner.close();
        expect(owner.permit_shutdown(81), "accept exactly one explicit failing shutdown");
        std::optional<CudaSessionResult> reply;
        while (!(reply = owner.take_shutdown_reply(81))) {
            expect(std::chrono::steady_clock::now() < deadline,
                   "bounded actual failed shutdown result");
            std::this_thread::yield();
        }
        expect(reply->status == CUDA_ERROR_INVALID_VALUE &&
                   reply->original_status == CUDA_ERROR_INVALID_VALUE &&
                   !owner.snapshot().is_closed && owner.snapshot().session.is_quarantined &&
                   !owner.permit_shutdown(82) && !owner.take_shutdown_reply(81) &&
                   native.unregister_attempts == unregisters + 1 &&
                   native.destroy_attempts == destroys + (fails_unregister ? 0U : 1U),
               "actual original error survives; unknown shutdown cannot retry or prove cleanup");
        // Retained uncertain registration/context ownership is intentional.
        // This child does not claim leak-free recovery or destructor cleanup.
        ::_exit(EXIT_SUCCESS);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
               WEXITSTATUS(status) == EXIT_SUCCESS,
           "isolated unknown cleanup retains exactly once without replay");
}
struct PoolNative {
    char identity{};
    std::atomic<unsigned int> initializations{}, context_queries{}, default_queries{},
        current_queries{}, selections{}, allocations{}, trims{}, releases{};
    std::array<unsigned int, 8> calls() const {
        return {initializations.load(), context_queries.load(), default_queries.load(),
                current_queries.load(), selections.load(),      allocations.load(),
                trims.load(),           releases.load()};
    }
} pool_native;
CUresult CUDAAPI pool_initialize(unsigned int flags) {
    ++pool_native.initializations;
    return initialize(flags);
}
CUresult CUDAAPI pool_context(CUcontext* output) {
    ++pool_native.context_queries;
    return get_current(output);
}
CUmemoryPool native_pool() {
    return reinterpret_cast<CUmemoryPool>(&pool_native.identity);
}
CUresult CUDAAPI pool_default(CUmemoryPool* output, CUdevice device) {
    ++pool_native.default_queries;
    if (!output || device) return CUDA_ERROR_INVALID_VALUE;
    *output = native_pool();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI pool_current(CUmemoryPool* output, CUdevice device) {
    ++pool_native.current_queries;
    if (!output || device) return CUDA_ERROR_INVALID_VALUE;
    *output = native_pool();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI pool_select(CUdevice device, CUmemoryPool pool) {
    ++pool_native.selections;
    return !device && pool == native_pool() ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
CUresult CUDAAPI pool_allocate(CUdeviceptr*, std::size_t, CUstream) {
    ++pool_native.allocations;
    return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI pool_allocate_from(CUdeviceptr*, std::size_t, CUmemoryPool, CUstream) {
    ++pool_native.allocations;
    return CUDA_ERROR_NOT_SUPPORTED;
}
cudaError_t CUDARTAPI pool_trim(cudaMemPool_t pool, std::size_t) {
    ++pool_native.trims;
    return reinterpret_cast<CUmemoryPool>(pool) == native_pool() ? cudaSuccess
                                                                 : cudaErrorInvalidValue;
}
CUresult CUDAAPI pool_release(CUdeviceptr) {
    ++pool_native.releases;
    return CUDA_ERROR_NOT_SUPPORTED;
}
CudaSessionDispatch pool_dispatch() {
    auto functions = dispatch();
    functions.process_memory = {pool_initialize,    pool_context, pool_default,
                                pool_current,       pool_select,  pool_allocate,
                                pool_allocate_from, pool_trim,    pool_release};
    return functions;
}
ExecutionWireReply pool_call(ExecutionSocketClient& client,
                             const ExecutionProcessMemoryRequest& request) {
    ExecutionPayloadWriter payload;
    expect(write_execution_process_memory_request(payload, request),
           "encode bounded process-memory operation around host cleanup failure");
    auto reply = client.exchange(ExecutionOpcode::kProcessMemory, payload.take());
    expect(reply.has_value(), "bounded actual PMM1 response after host ownership transition");
    return std::move(*reply);
}
void quarantined_service_close_reply(bool fails_unregister,
                                     std::optional<bool> skips_helper_plans = std::nullopt) {
    const auto child = ::fork();
    expect(child >= 0, "isolate intentional quarantined service cleanup retention");
    if (child == 0) {
        native.may_register = true;
        const auto identity = current_process_identity();
        expect(identity.has_value(), "authenticate cleanup-error priority client");
        const auto path = "/tmp/glimmer-host-close-error-" + std::to_string(::getpid());
        ExecutionServiceConfig config;
        config.transport.socket_path = path;
        config.uses_original_kernels = true;
        config.client_priorities = {{*identity, 7, ExecutionClientKernelMode::kOriginal}};
        if (skips_helper_plans) {
            config.uses_original_kernels = false;
            config.persistent.workers = 320;
            config.client_priorities = {{*identity, 0, ExecutionClientKernelMode::kOriginal}};
            config.should_skip_idle_helper_plans = *skips_helper_plans;
        }
        // Only this regression enables process-memory support. Existing host
        // fixtures retain their original dispatch and cleanup assertions.
        ExecutionService service(config, pool_dispatch());
        expect(service.start(), "start actual priority service for cleanup-error reply");
        const auto unregisters = native.unregister_attempts.load();
        const auto destroys = native.destroy_attempts.load();
        auto client = std::async(std::launch::async, [&] {
            ExecutionSocketClient process({path, *identity, 3000});
            ExecutionPayloadWriter process_bootstrap;
            expect(process_bootstrap.write_u32(kExecutionProcessMemoryRole) &&
                       process_bootstrap.write_u32(0),
                   "encode authenticated context-free process-memory carrier");
            const auto opened = process.open(process_bootstrap.take());
            expect(opened && opened->status == CUDA_SUCCESS && !opened->original_status &&
                       opened->disposition == ExecutionWireDisposition::kComplete &&
                       decode_execution_bootstrap_ownership(opened->data) ==
                           ExecutionBootstrapOwnership::kMetadataRetained,
                   "PMM1 carrier is authenticated before any host cleanup failure");
            using Operation = ExecutionProcessMemoryOperation;
            const auto pool = actual(pool_call(process, {Operation::kGetDefaultPool, 0, true}));
            ExecutionPayloadReader observed_pool(pool.data);
            ExecutionProcessMemoryObservation effect;
            expect(pool.id && pool.address == reinterpret_cast<std::uintptr_t>(native_pool()) &&
                       !pool.parameter_bytes &&
                       read_execution_process_memory_observation(observed_pool, effect) &&
                       observed_pool.empty() && effect.output_written,
                   "real injected SDK output creates pool authority before failure");
            const std::array requests{
                ExecutionProcessMemoryRequest{Operation::kGetDefaultPool, 0, true},
                ExecutionProcessMemoryRequest{Operation::kGetCurrentPool, 0, true},
                ExecutionProcessMemoryRequest{Operation::kSetCurrentPool, 0, false, pool.id},
                ExecutionProcessMemoryRequest{Operation::kTrimPool, 0, false, pool.id}};
            for (const auto& request : requests) (void)actual(pool_call(process, request));
            expect(pool_native.default_queries && pool_native.current_queries &&
                       pool_native.selections && pool_native.trims,
                   "healthy query/set/trim all reach SDK before the regression fault");
            const auto pool_calls = pool_native.calls();
            ExecutionSocketClient host({path, *identity, 3000});
            ExecutionPayloadWriter bootstrap;
            expect(bootstrap.write_u32(kExecutionHostOwnerRole) &&
                       bootstrap.write_u32(CU_CTX_MAP_HOST),
                   "encode explicit host role for error polling");
            actual(host.open(bootstrap.take()));
            ExecutionOwnedDescriptor backing(
                ::memfd_create("glimmer-host-close-error", MFD_ALLOW_SEALING | MFD_CLOEXEC));
            expect(backing.get() >= 0 && ::ftruncate(backing.get(), 4096) == 0 &&
                       ::fcntl(backing.get(), F_ADD_SEALS,
                               F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) == 0,
                   "sealed imported resource before actual service close error");
            ExecutionPayloadWriter import;
            expect(import.write_u64(256) && import.write_u64(4096) && import.write_u32(0),
                   "encode cleanup-error import");
            const auto pending =
                host.submit(ExecutionOpcode::kImportHostMemory, import.take(), backing.get());
            expect(pending.status == ExecutionSocketSubmitStatus::kAccepted,
                   "accept cleanup-error import");
            std::optional<ExecutionWireReply> imported;
            while (!(imported = host.take_reply(pending.sequence)))
                expect(host.poll_for(1), "wait for actual imported resource");
            (void)actual(std::move(imported));
            if (fails_unregister)
                native.unregister_status = CUDA_ERROR_INVALID_VALUE;
            else
                native.destroy_status = CUDA_ERROR_INVALID_VALUE;
            const auto closed = host.exchange(ExecutionOpcode::kClose);
            expect(closed && closed->status == CUDA_ERROR_INVALID_VALUE &&
                       closed->original_status == CUDA_ERROR_INVALID_VALUE &&
                       closed->disposition == ExecutionWireDisposition::kComplete,
                   "quarantine cannot suppress the already accepted actual cleanup-error reply");
            for (const auto& request : requests) {
                const auto denied = pool_call(process, request);
                expect(denied.status == CUDA_ERROR_UNKNOWN && !denied.original_status &&
                           denied.disposition == ExecutionWireDisposition::kRejected &&
                           !denied.id && !denied.address && !denied.parameter_bytes &&
                           denied.arguments.empty() && denied.data.empty() &&
                           pool_native.calls() == pool_calls,
                       "completed HostOwner quarantine fences every later pool SDK entry, "
                       "even when no native reservation remains");
            }
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{4};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
            expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
                   "bounded no-new-grant polling of actual failed Close");
        client.get();
        expect(!service.finished() && native.unregister_attempts == unregisters + 1 &&
                   native.destroy_attempts == destroys + (fails_unregister ? 0U : 1U),
               "actual unknown cleanup is retained without retry after error delivery");
        // Unknown native ownership deliberately survives this isolated process.
        ::_exit(EXIT_SUCCESS);
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
               WEXITSTATUS(status) == EXIT_SUCCESS,
           "actual priority service preserves cleanup errors after quarantine");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        std::optional<bool> skips_helper_plans;
        if (argc == 2 && std::string_view(argv[1]) == "--helper-plans-on")
            skips_helper_plans = true;
        else if (argc == 2 && std::string_view(argv[1]) == "--helper-plans-off")
            skips_helper_plans = false;
        else
            expect(argc == 1, "recognized helper planning fixture mode");
        const auto path = "/tmp/glimmer-host-service-" + std::to_string(::getpid());
        shared_host_service(path + "-free", false, false, false, skips_helper_plans);
        shared_host_service(path + "-disconnect", true, false, false, skips_helper_plans);
        shared_host_service(path + "-staging", false, true, false, skips_helper_plans);
        shared_host_service(path + "-close-disconnect", true, false, true, skips_helper_plans);
        host_actor_guards_and_deferred_shutdown();
        host_publication_wakes_after_normal_cleanup();
        explicit_shutdown_failure_retains(true);
        explicit_shutdown_failure_retains(false);
        quarantined_service_close_reply(true, skips_helper_plans);
        quarantined_service_close_reply(false, skips_helper_plans);
        std::cout << "Native host actor and shared asynchronous service checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
