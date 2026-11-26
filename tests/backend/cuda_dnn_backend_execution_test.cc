#include "dnn_backend_graph_fixture.h"
#include "internal/cuda_dnn_backend_execution.h"

#include <array>
#include <functional>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
namespace graph = glimmer::tests::dnn_backend_graph;
using PointerKind = CudaDnnBackendPointerKind;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Pack {
    bool alive{}, finalized{};
    unsigned int destroys{};
    std::vector<std::int64_t> uids;
    std::vector<void*> pointers;
    void* workspace{};
};
struct Event {
    bool alive{}, recorded{}, ready{};
};
struct Invocation {
    Pack* pack{};
    std::vector<std::int64_t> uids;
    std::vector<void*> pointers;
    std::array<std::vector<std::byte>, 3> host;
    void* workspace{};
};
struct NativeState {
    char context{}, foreign{}, handle{}, stream{};
    void* plan{};
    cudnnBackendDescriptor_t alias_create{};
    std::array<Pack, 16> packs;
    std::array<Event, 16> events;
    unsigned int creates{}, sets{}, finalizes{}, executes{}, destroys{}, synchronizes{};
    unsigned int records{}, event_destroys{}, stream_drains{}, stream_gets{};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, execute_status{CUDNN_STATUS_SUCCESS};
    cudnnStatus_t destroy_status{CUDNN_STATUS_SUCCESS};
    CUresult synchronize_status{CUDA_SUCCESS}, record_status{CUDA_SUCCESS};
    unsigned int fail_phase{}, lose_context_phase{};
    bool wrong_context{}, wrong_stream{}, null_create{}, partial_create{}, partial_write{};
    std::function<void(unsigned int)> callback;
    std::vector<Invocation> calls;
};
NativeState state;
CUcontext context() {
    return reinterpret_cast<CUcontext>(&state.context);
}
CUstream stream() {
    return reinterpret_cast<CUstream>(&state.stream);
}
cudnnHandle_t handle() {
    return reinterpret_cast<cudnnHandle_t>(&state.handle);
}
void callback(unsigned int phase) {
    if (state.callback) state.callback(phase);
    if (state.lose_context_phase == phase) state.wrong_context = true;
}
CUresult CUDAAPI current(CUcontext* output) {
    *output = state.wrong_context ? reinterpret_cast<CUcontext>(&state.foreign) : context();
    return CUDA_SUCCESS;
}
CUresult CUDAAPI synchronize() {
    ++state.synchronizes;
    callback(9);
    return state.synchronize_status;
}
Pack& pack(cudnnBackendDescriptor_t native) {
    for (auto& value : state.packs)
        if (&value == native) {
            expect(value.alive, "native private pack is still owned");
            return value;
        }
    throw std::runtime_error("foreign private native pack");
}
CudaDnnDispatch library() {
    auto sdk = graph::dispatch();
    sdk.backend_create = [](cudnnBackendDescriptorType_t kind, cudnnBackendDescriptor_t* output) {
        expect(kind == CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR && output,
               "only original private VariantPack construction is admitted");
        ++state.creates;
        *output = nullptr;
        callback(1);
        if (state.alias_create) {
            *output = state.alias_create;
            return CUDNN_STATUS_SUCCESS;
        }
        if (state.null_create ||
            (state.create_status != CUDNN_STATUS_SUCCESS && !state.partial_create))
            return state.create_status;
        for (auto& value : state.packs)
            if (!value.alive && !value.destroys) {
                value.alive = true;
                *output = &value;
                return state.create_status;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    sdk.backend_set = [](cudnnBackendDescriptor_t native, cudnnBackendAttributeName_t name,
                         cudnnBackendAttributeType_t type, std::int64_t count, const void* data) {
        auto& value = pack(native);
        ++state.sets;
        expect(data && count >= 0 && count <= 100 && !value.finalized,
               "private setters preserve bounded present arrays and original order");
        unsigned int phase{};
        if (name == CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS) {
            expect(type == CUDNN_TYPE_INT64, "UID original ABI");
            const auto* first = static_cast<const std::int64_t*>(data);
            value.uids.assign(first, first + count);
            phase = 2;
        } else if (name == CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS) {
            expect(type == CUDNN_TYPE_VOID_PTR, "pointer original ABI");
            auto* const* first = static_cast<void* const*>(data);
            value.pointers.assign(first, first + count);
            phase = 3;
        } else {
            expect(name == CUDNN_ATTR_VARIANT_PACK_WORKSPACE && type == CUDNN_TYPE_VOID_PTR &&
                       count == 1,
                   "workspace original ABI");
            std::memcpy(&value.workspace, data, sizeof(value.workspace));
            phase = 4;
        }
        callback(phase);
        // Deliberately mutate the private descriptor before an injected error.
        return state.fail_phase == phase ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_SUCCESS;
    };
    sdk.backend_finalize = [](cudnnBackendDescriptor_t native) {
        auto& value = pack(native);
        ++state.finalizes;
        expect(value.uids.size() == value.pointers.size(), "complete private pack pairing");
        value.finalized = true;
        callback(5);
        return state.fail_phase == 5 ? CUDNN_STATUS_INTERNAL_ERROR : CUDNN_STATUS_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact cuDNN BackendExecute ABI.
    sdk.backend_execute = [](cudnnHandle_t native_handle, cudnnBackendDescriptor_t native_plan,
                             cudnnBackendDescriptor_t native_pack) {
        expect(native_handle == handle() && native_plan == state.plan,
               "trusted plan and handle identity preserved without fabricating planning");
        auto& value = pack(native_pack);
        expect(value.finalized, "private preparation precedes original Execute");
        ++state.executes;
        Invocation copied{&value, value.uids, value.pointers, {}, value.workspace};
        if (!value.pointers.empty()) {
            expect(value.pointers.size() == 5 && value.pointers[0] && !value.pointers[1],
                   "device and explicit NULL positions preserved");
            constexpr std::array<std::size_t, 3> widths{2, 4, 8};
            for (std::size_t i = 0; i != widths.size(); ++i) {
                const auto* bytes = static_cast<const std::byte*>(value.pointers[i + 2]);
                expect(bytes && reinterpret_cast<std::uintptr_t>(bytes) % 24 == 0,
                       "non-power-of-two by-value alignment is honored");
                copied.host[i].assign(bytes, bytes + widths[i]);
            }
            if (state.partial_write) *static_cast<std::byte*>(value.pointers[0]) = std::byte{0x5a};
        }
        state.calls.push_back(std::move(copied));
        callback(6);
        return state.execute_status;
    };
    sdk.get_stream = [](cudnnHandle_t native, cudaStream_t* output) {
        expect(native == handle(), "stream proof uses original owned handle");
        ++state.stream_gets;
        *output = reinterpret_cast<cudaStream_t>(state.wrong_stream ? nullptr : stream());
        callback(7);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.backend_destroy = [](cudnnBackendDescriptor_t native) {
        auto& value = pack(native);
        ++state.destroys;
        expect(++value.destroys == 1, "native private pack destruction is exactly once");
        callback(8);
        if (state.destroy_status == CUDNN_STATUS_SUCCESS) value.alive = false;
        return state.destroy_status;
    };
    return sdk;
}
CudaExecutionDispatch driver() {
    CudaExecutionDispatch sdk;
    sdk.context_get_current = current;
    sdk.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int, unsigned int,
                           unsigned int, unsigned int, unsigned int, CUstream, void**,
                           void**) { return CUDA_ERROR_NOT_SUPPORTED; };
    sdk.event_create = [](CUevent* output, unsigned int flags) {
        expect(flags == CU_EVENT_DISABLE_TIMING, "non-timing physical completion event");
        for (auto& event : state.events)
            if (!event.alive) {
                event = {.alive = true};
                *output = reinterpret_cast<CUevent>(&event);
                return CUDA_SUCCESS;
            }
        return CUDA_ERROR_OUT_OF_MEMORY;
    };
    sdk.event_record = [](CUevent native, CUstream actual) {
        expect(actual == stream(), "event records on checked original stream");
        ++state.records;
        if (state.record_status == CUDA_SUCCESS) reinterpret_cast<Event*>(native)->recorded = true;
        return state.record_status;
    };
    sdk.event_query = [](CUevent native) {
        const auto& value = *reinterpret_cast<Event*>(native);
        return !value.recorded || value.ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.event_destroy = [](CUevent native) {
        auto& value = *reinterpret_cast<Event*>(native);
        expect(value.alive && !state.wrong_context, "event cleanup on original owner");
        value.alive = false;
        ++state.event_destroys;
        return CUDA_SUCCESS;
    };
    sdk.stream_synchronize = [](CUstream actual) {
        expect(actual == stream(), "known stream recovery");
        ++state.stream_drains;
        return CUDA_SUCCESS;
    };
    return sdk;
}
struct Backing {
    std::array<std::byte, 64> data;
};
struct Capsule {
    std::shared_ptr<CudaExecutionLibraryCall> call;
    std::shared_ptr<CudaDnnBackendExecutionCompletion> result;
};
struct Owner {
    std::unique_ptr<CudaDnnRegistry> registry;
    std::unique_ptr<CudaExecutionDispatcher> executor;
    std::shared_ptr<Backing> data{std::make_shared<Backing>()};
    std::shared_ptr<Backing> workspace{std::make_shared<Backing>()};
    std::uint64_t descriptor{};
    Owner() {
        graph::reset();
        state = {};
        registry = std::make_unique<CudaDnnRegistry>(graph::dispatch());
        // A real registry lease/global call gate, but deliberately not a graph
        // planning oracle. Fake Execute independently validates this sentinel.
        const auto created = registry->create_backend(21);
        expect(created.original_status == CUDNN_STATUS_SUCCESS, "create lease sentinel");
        descriptor = created.id;
        const auto view = registry->borrow_backend(descriptor);
        expect(bool(view), "owned sentinel has physical lifetime");
        state.plan = view->object.native;
        executor = std::make_unique<CudaExecutionDispatcher>(
            ExecutionSessionIdentity{901, 902}, CudaExecutionContext{903, context(), data},
            driver(), ExecutionSubmissionLimits{.max_commands = 4, .max_parameter_bytes = 64});
        expect(executor->bind_executor() == CUDA_SUCCESS &&
                   executor->bind_stream(1, stream(), std::make_shared<int>(0), true),
               "bind independent owner-thread dispatcher");
        data->data.fill(std::byte{0xa5});
        workspace->data.fill(std::byte{0xc7});
    }
    CudaDnnBackendExecutionArguments arguments(unsigned char seed = 1) {
        const auto view = registry->borrow_backend_call(descriptor);
        expect(bool(view), "real registry composite call lease");
        CudaDnnBackendExecutionArguments result;
        result.plan = *view;
        result.uids = {11, 11, 7, 7, 13};
        result.pointers.push_back({PointerKind::kDevice,
                                   {reinterpret_cast<CUdeviceptr>(data->data.data() + 8), 24, data},
                                   {},
                                   1});
        result.pointers.push_back({PointerKind::kNull, {}, {}, 1});
        for (const auto size : {2U, 4U, 8U}) {
            std::vector<std::byte> bytes(size);
            for (unsigned int i = 0; i != size; ++i)
                bytes[i] = static_cast<std::byte>(seed + size + i);
            result.pointers.push_back({PointerKind::kHostValue, {}, std::move(bytes), 24});
        }
        result.workspace = {reinterpret_cast<CUdeviceptr>(workspace->data.data() + 8), 24,
                            workspace};
        return result;
    }
    Capsule capsule(CudaDnnBackendExecutionArguments arguments) {
        auto result = std::make_shared<CudaDnnBackendExecutionCompletion>();
        auto call = make_cuda_dnn_backend_call(std::make_shared<const CudaDnnDispatch>(library()),
                                               handle(), std::move(arguments),
                                               {context(), current, synchronize}, result);
        return {std::move(call), std::move(result)};
    }
    ExecutionAcceptance accept(std::uint64_t sequence, const Capsule& capsule) {
        return executor->accept_library(sequence, {1, {}, capsule.call});
    }
    void close() {
        expect(executor->shutdown(), "all issued private packs physically retired");
        executor.reset();
        registry->close();
        expect(registry->shutdown().status == CUDA_SUCCESS && graph::alive() == 0,
               "plan lease releases without a context/registry ownership cycle");
    }
};

void concurrent_packs() {
    Owner owner;
    auto first_arguments = owner.arguments(1), second_arguments = owner.arguments(71);
    std::array<std::vector<std::byte>, 3> first_bytes, second_bytes;
    for (std::size_t i = 0; i != 3; ++i) {
        first_bytes[i] = first_arguments.pointers[i + 2].value;
        second_bytes[i] = second_arguments.pointers[i + 2].value;
    }
    auto first = owner.capsule(std::move(first_arguments));
    auto second = owner.capsule(std::move(second_arguments));
    const auto one = owner.accept(1, first), two = owner.accept(2, second);
    expect(one.accepted() && two.accepted() && !state.creates && !state.sets && !state.executes,
           "factory and queue admission create no private native resources");
    expect(owner.executor->issue_next().library_status == CUDNN_STATUS_SUCCESS &&
               owner.executor->issue_next().library_status == CUDNN_STATUS_SUCCESS,
           "two independent original Executes issue before either physically retires");
    expect(state.calls.size() == 2 && state.calls[0].pack != state.calls[1].pack &&
               state.calls[0].uids == std::vector<std::int64_t>({11, 11, 7, 7, 13}) &&
               state.calls[1].uids == state.calls[0].uids && state.calls[0].host == first_bytes &&
               state.calls[1].host == second_bytes &&
               state.calls[0].workspace == owner.workspace->data.data() + 8 && !state.synchronizes,
           "private packs preserve duplicate UID order, exact snapshots and independent identity");
    for (std::size_t i = 2; i != 5; ++i)
        expect(state.calls[0].pointers[i] != state.calls[1].pointers[i],
               "simultaneous commands never share mutable host-value storage");
    std::weak_ptr<CudaExecutionLibraryCall> first_call = first.call, second_call = second.call;
    std::weak_ptr<Backing> workspace = owner.workspace;
    first.call.reset();
    second.call.reset();
    owner.workspace.reset();
    expect(
        owner.executor->acknowledge_result(one.receipt) &&
            owner.executor->acknowledge_result(two.receipt) &&
            owner.executor->poll().retained == 2 && !workspace.expired() && !first_call.expired() &&
            !second_call.expired() && !state.destroys &&
            owner.registry->destroy(owner.descriptor).disposition == CudaDnnDisposition::kDeferred,
        "ACK leaves native packs, host bytes, workspace and plan physically leased");
    state.events[0].ready = true;
    expect(owner.executor->poll().retired == 1 && first_call.expired() && !second_call.expired() &&
               !workspace.expired() && state.destroys == 1,
           "one completed event cannot retire the other command's private resources");
    state.events[1].ready = true;
    expect(owner.executor->poll().retired == 1 && second_call.expired() && workspace.expired() &&
               state.destroys == 2 && state.event_destroys == 2,
           "last physical proof destroys exactly one remaining pack and releases its leases");
    owner.close();
}

void canceled_and_empty() {
    Owner owner;
    auto canceled = owner.capsule(owner.arguments());
    const auto accepted = owner.accept(1, canceled);
    expect(accepted.accepted() && owner.executor->reject_queued(CUDA_ERROR_DEINITIALIZED) == 1 &&
               owner.executor->host_result(accepted.receipt)->origin ==
                   ExecutionReplyOrigin::kRejectedBeforeIssue &&
               owner.executor->acknowledge_result(accepted.receipt),
           "queued cancellation has no original Execute result");
    canceled.call.reset();
    expect(!state.creates && !state.destroys && !state.executes && !canceled.result->is_ready,
           "canceled capsule has neither private construction nor native destructor work");
    auto arguments = owner.arguments();
    arguments.uids.clear();
    arguments.pointers.clear();
    arguments.workspace = {};
    auto empty = owner.capsule(std::move(arguments));
    const auto next = owner.accept(2, empty);
    expect(next.accepted() && owner.executor->issue_next().library_status == CUDNN_STATUS_SUCCESS &&
               state.calls[0].uids.empty() && !state.calls[0].workspace,
           "empty present arrays and NULL workspace preserve original private ABI");
    state.events[0].ready = true;
    empty.call.reset();
    expect(owner.executor->poll().retired == 1 &&
               owner.executor->acknowledge_result(next.receipt) && state.destroys == 1,
           "empty original work still has exactly one physical cleanup");
    owner.close();
}

void invalid_trusted_inputs() {
    Owner owner;
    for (unsigned int mode = 0; mode != 14; ++mode) {
        auto arguments = owner.arguments();
        auto sdk = library();
        auto proof = CudaDnnFindContext{context(), current, synchronize};
        switch (mode) {
            case 0:
                sdk.backend_create = nullptr;
                break;
            case 1:
                sdk.backend_set = nullptr;
                break;
            case 2:
                sdk.backend_finalize = nullptr;
                break;
            case 3:
                sdk.backend_destroy = nullptr;
                break;
            case 4:
                sdk.backend_execute = nullptr;
                break;
            case 5:
                sdk.get_stream = nullptr;
                break;
            case 6:
                proof.synchronize = nullptr;
                break;
            case 7:
                arguments.plan.call.reset();
                break;
            case 8:
                arguments.pointers[0].device.lifetime.reset();
                break;
            case 9:
                arguments.pointers[1].value = {std::byte{1}};
                break;
            case 10:
                arguments.pointers[2].alignment = 0;
                break;
            case 11:
                arguments.pointers[2].alignment = (std::size_t{2} << 20) + 1;
                break;
            case 12:
                arguments.uids.pop_back();
                break;
            default:
                arguments.workspace.lifetime.reset();
                break;
        }
        bool rejected{};
        try {
            (void)make_cuda_dnn_backend_call(std::make_shared<const CudaDnnDispatch>(sdk), handle(),
                                             std::move(arguments), proof,
                                             std::make_shared<CudaDnnBackendExecutionCompletion>());
        } catch (const std::exception&) {
            rejected = true;
        }
        expect(rejected && !state.creates && !state.executes && !state.destroys,
               "missing callbacks and malformed trusted bindings fail without native resources");
    }
    owner.close();
}

void preparation_errors() {
    for (unsigned int phase = 1; phase != 6; ++phase) {
        Owner owner;
        auto capsule = owner.capsule(owner.arguments());
        state.fail_phase = phase;
        if (phase == 1) {
            state.create_status = CUDNN_STATUS_BAD_PARAM;
        }
        const auto accepted = owner.accept(1, capsule);
        const auto issued = owner.executor->issue_next();
        const auto host = owner.executor->host_result(accepted.receipt);
        expect(accepted.accepted() && issued.state == CudaExecutionIssueState::kRejected &&
                   !issued.library_status && host &&
                   host->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   host->status == CUDA_ERROR_UNKNOWN && capsule.result->is_ready &&
                   !capsule.result->result.original_status &&
                   capsule.result->preparation_status ==
                       (phase == 5 ? CUDNN_STATUS_INTERNAL_ERROR : CUDNN_STATUS_BAD_PARAM) &&
                   state.creates == 1 && !state.executes && !state.destroys,
               "partial original private construction cannot invent requested Execute provenance");
        expect(owner.executor->acknowledge_result(accepted.receipt) &&
                   owner.executor->poll().retained == 1 && !state.destroys,
               "partially written private descriptor remains owned before physical stop");
        state.events[0].ready = true;
        capsule.call.reset();
        expect(owner.executor->poll().retired == 1 && state.destroys == (phase == 1 ? 0U : 1U) &&
                   !state.executes,
               "private preparation failure cleans its partial native pack exactly once");
        owner.close();
    }
}

void original_error_and_record_recovery() {
    for (const bool record_failure : {false, true}) {
        Owner owner;
        auto capsule = owner.capsule(owner.arguments());
        state.execute_status = CUDNN_STATUS_EXECUTION_FAILED;
        state.partial_write = true;
        if (record_failure) state.record_status = CUDA_ERROR_INVALID_HANDLE;
        const auto accepted = owner.accept(1, capsule);
        const auto issued = owner.executor->issue_next();
        const auto host = owner.executor->host_result(accepted.receipt);
        expect(issued.library_status == CUDNN_STATUS_EXECUTION_FAILED && host &&
                   host->origin == ExecutionReplyOrigin::kNativeSdk &&
                   host->status == CUDNN_STATUS_EXECUTION_FAILED && state.synchronizes == 1 &&
                   state.stream_gets == 1 && owner.data->data[8] == std::byte{0x5a} &&
                   owner.data->data[7] == std::byte{0xa5} &&
                   owner.data->data[32] == std::byte{0xa5} &&
                   capsule.result->result.original_status == CUDNN_STATUS_EXECUTION_FAILED &&
                   !capsule.result->is_cleanup_uncertain,
               "partial original writes preserve exact native error and require context-wide stop "
               "proof");
        expect(owner.executor->acknowledge_result(accepted.receipt) &&
                   owner.executor->poll().retained == 1 && !state.destroys,
               "native error and ACK never release a potentially active original call");
        state.events[0].ready = true;
        capsule.call.reset();
        const auto stopped =
            record_failure ? owner.executor->drain_issued() : owner.executor->poll();
        expect(stopped.retired == 1 && state.destroys == 1 &&
                   state.stream_drains == (record_failure ? 1U : 0U),
               "failed record permits only an explicit checked stream recovery before cleanup");
        owner.close();
    }
}

void reentry() {
    Owner owner;
    auto capsule = owner.capsule(owner.arguments());
    std::array<unsigned int, 10> phases{};
    state.callback = [&](unsigned int phase) {
        ++phases[phase];
        const auto native_calls = graph::control.callbacks.load();
        expect(owner.registry->create_backend(21).status == CUDA_ERROR_NOT_READY &&
                   !owner.registry->borrow_backend_call(owner.descriptor) &&
                   owner.registry->destroy(owner.descriptor).status == CUDA_ERROR_NOT_READY &&
                   owner.registry->shutdown().disposition == CudaDnnDisposition::kDeferred &&
                   graph::control.callbacks.load() == native_calls,
               "private preparation, Execute, proof and cleanup hold registry call exclusion");
    };
    state.execute_status = CUDNN_STATUS_BAD_PARAM;
    const auto accepted = owner.accept(1, capsule);
    expect(owner.executor->issue_next().library_status == CUDNN_STATUS_BAD_PARAM,
           "original native error exercises hidden-stream proof callback");
    state.events[0].ready = true;
    capsule.call.reset();
    expect(
        owner.executor->poll().retired == 1 && owner.executor->acknowledge_result(accepted.receipt),
        "cleanup exits registry exclusion without deadlock");
    state.callback = {};
    for (unsigned int phase = 1; phase != 10; ++phase)
        expect(phases[phase] == 1, "each original private/proof/cleanup callback runs once");
    owner.close();
}

void uncertain(unsigned int mode) {
    auto owner = std::make_unique<Owner>();
    auto capsule = owner->capsule(owner->arguments());
    if (mode == 0) state.lose_context_phase = 3;
    if (mode == 1) state.lose_context_phase = 6;
    if (mode == 2) {
        state.execute_status = CUDNN_STATUS_EXECUTION_FAILED;
        state.synchronize_status = CUDA_ERROR_LAUNCH_FAILED;
    }
    if (mode == 3) state.wrong_stream = true;
    if (mode == 4) state.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
    if (mode == 5) state.lose_context_phase = 8;
    if (mode == 6) {
        state.create_status = CUDNN_STATUS_BAD_PARAM;
        state.partial_create = true;
    }
    if (mode == 7) state.null_create = true;
    const auto accepted = owner->accept(1, capsule);
    const auto issued = owner->executor->issue_next();
    const auto host = owner->executor->host_result(accepted.receipt);
    expect(accepted.accepted() && host &&
               (mode == 0 || mode >= 6
                    ? (!issued.library_status && !capsule.result->result.original_status &&
                       host->origin == ExecutionReplyOrigin::kRejectedBeforeIssue)
                    : (issued.library_status.has_value() &&
                       host->origin == ExecutionReplyOrigin::kNativeSdk &&
                       capsule.result->result.original_status == state.execute_status)),
           "proof failure does not fabricate or erase original Execute provenance");
    state.wrong_context = false;
    state.events[0].ready = true;
    expect(!owner->executor->poll().retired && capsule.result->is_cleanup_uncertain &&
               capsule.result->has_context_loss == (mode == 0 || mode == 1 || mode == 5),
           "opaque proof loss or failed native destruction retains the capsule");
    const auto destroyed = state.destroys, executed = state.executes;
    std::weak_ptr<CudaExecutionLibraryCall> lease = capsule.call;
    std::weak_ptr<Backing> workspace = owner->workspace;
    capsule.call.reset();
    owner->workspace.reset();
    state.wrong_context = false;
    state.lose_context_phase = 0;
    state.wrong_stream = false;
    state.synchronize_status = CUDA_SUCCESS;
    state.destroy_status = CUDNN_STATUS_SUCCESS;
    expect(owner->executor->acknowledge_result(accepted.receipt) &&
               !owner->executor->poll().retired && !owner->executor->drain_issued().retired &&
               !owner->executor->shutdown() && !lease.expired() && !workspace.expired() &&
               state.destroys == destroyed && state.executes == executed &&
               destroyed == (mode == 4 || mode == 5 ? 1U : 0U),
           "restored proof cannot replay Execute or retry a possibly destructive private cleanup");
    owner.reset();
    expect(!lease.expired() && !workspace.expired() && state.destroys == destroyed &&
               graph::alive() == 1,
           "CPU destruction retains uncertain plan, pack, allocation and host-value owners");
}

void duplicate_native_ownership(unsigned int mode) {
    auto owner = std::make_unique<Owner>();
    auto first = owner->capsule(owner->arguments(1));
    Capsule second;
    ExecutionAcceptance accepted_first, accepted_second;
    unsigned int public_objects = 1;
    if (mode == 2) {
        accepted_first = owner->accept(1, first);
        expect(accepted_first.accepted() &&
                   owner->executor->issue_next().library_status == CUDNN_STATUS_SUCCESS,
               "first original command is physically pending before duplicate Create");
        state.alias_create = state.calls.front().pack;
        second = owner->capsule(owner->arguments(61));
        accepted_second = owner->accept(2, second);
    } else {
        if (mode == 0)
            state.alias_create = state.plan;
        else {
            const auto other = owner->registry->create_backend(21);
            expect(other.original_status == CUDNN_STATUS_SUCCESS,
                   "create independent public alias");
            const auto view = owner->registry->borrow_backend(other.id);
            expect(view && view->object.native != state.plan, "alias is not the capsule's plan");
            state.alias_create = view->object.native;
            public_objects = 2;
        }
        accepted_first = owner->accept(1, first);
    }
    const auto issued = owner->executor->issue_next();
    const auto rejected =
        owner->executor->host_result(mode == 2 ? accepted_second.receipt : accepted_first.receipt);
    expect(!issued.library_status && rejected &&
               rejected->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               owner->registry->snapshot().is_quarantined && first.result->is_cleanup_uncertain &&
               !first.result->has_context_loss && !state.destroys && !graph::state.destroys,
           "contradictory private Create quarantines both native and registry ownership without "
           "Destroy");
    if (mode == 2) {
        const auto original = owner->executor->host_result(accepted_first.receipt);
        expect(
            second.result->is_cleanup_uncertain && !second.result->result.original_status &&
                original && original->origin == ExecutionReplyOrigin::kNativeSdk &&
                original->status == CUDNN_STATUS_SUCCESS &&
                first.result->result.original_status == CUDNN_STATUS_SUCCESS &&
                state.executes == 1 && state.sets == 3,
            "second alias does not erase first Execute provenance or repeat preparation/Execute");
    } else
        expect(!state.executes && !state.sets,
               "public-object alias never reaches a private setter");
    for (auto& event : state.events) event.ready = true;
    std::weak_ptr<CudaExecutionLibraryCall> first_lease = first.call, second_lease = second.call;
    first.call.reset();
    second.call.reset();
    expect(owner->executor->acknowledge_result(accepted_first.receipt) &&
               (mode != 2 || owner->executor->acknowledge_result(accepted_second.receipt)) &&
               !owner->executor->poll().retired && !owner->executor->drain_issued().retired &&
               !owner->executor->shutdown() && !first_lease.expired() &&
               (mode != 2 || !second_lease.expired()) && !state.destroys && !graph::state.destroys,
           "even complete events cannot choose which contradictory owner may destroy the pointer");
    owner.reset();
    expect(!first_lease.expired() && (mode != 2 || !second_lease.expired()) && !state.destroys &&
               !graph::state.destroys && graph::alive() == public_objects,
           "CPU teardown cannot destroy either aliased public or in-flight private native owner");
}
}  // namespace

int main() {
    try {
        concurrent_packs();
        canceled_and_empty();
        invalid_trusted_inputs();
        preparation_errors();
        original_error_and_record_recovery();
        reentry();
        for (unsigned int mode = 0; mode != 8; ++mode) {
            const auto child = fork();
            expect(child >= 0, "fork isolated retained private resources");
            if (!child) {
                try {
                    uncertain(mode);
                    _exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "backend capsule uncertainty " << mode << ": " << error.what()
                              << '\n';
                    _exit(1);
                }
            }
            int status{};
            expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                   "isolated backend capsule ownership failure");
        }
        for (unsigned int mode = 0; mode != 3; ++mode) {
            const auto child = fork();
            expect(child >= 0, "fork contradictory private native pointer ownership");
            if (!child) {
                try {
                    duplicate_native_ownership(mode);
                    _exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "backend capsule alias " << mode << ": " << error.what() << '\n';
                    _exit(1);
                }
            }
            int status{};
            expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                   "isolated backend capsule alias retention");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "backend execution capsule: " << error.what() << '\n';
        return 1;
    }
}
