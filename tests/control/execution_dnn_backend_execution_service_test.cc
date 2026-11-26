#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_backend_protocol.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_backend_graph_fixture.h"
#include "../backend/dnn_convolution_fixture.h"

#include <cstdlib>
#include <cerrno>
#include <future>
#include <iostream>
#include <set>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
namespace metadata = glimmer::tests::dnn_backend_graph;
namespace driver = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
using Native = cudnnBackendDescriptor_t;
using Values = ExecutionDnnBackendAttributeValues;
using Header = ExecutionDnnBackendAttributeHeader;
using Class = ExecutionDnnBackendBindingClass;
enum class Mode { kClose, kDisconnect, kForeignPending, kQueuedDemand, kProfileRebindPending };

// The existing independent Tensor model supplies native observations. These
// additional callbacks model CPU graph copying and a deliberately small injected
// Execute operation, not the vendor's engine selection or numerical algorithm.
struct Control {
    std::atomic<unsigned int> executions{}, private_creates{}, private_destroys{};
    std::atomic<int> execute_status{};
    std::atomic<bool> is_executing{}, fail_private_finalize{};
    std::atomic<bool> hold_next_context{}, release_foreign{true};
    std::atomic<std::uintptr_t> held_context{};
    std::atomic<unsigned int> service_steps{}, blocked_queries{};
    std::atomic<unsigned int> profile_callbacks{}, profile_waits{};
};
Control control;
CudaDnnDispatch original_metadata;
std::set<Native> private_packs;  // Service owner thread only.
constexpr std::size_t kWorkspaceBytes = 32;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool is_planning(std::uint32_t kind) {
    return kind == 2 || kind == 3 || kind == 5 || kind == 15 || kind == 16;
}
bool is_child(std::uint32_t name) {
    return name == 801 || name == 1300 || name == 300 || name == 401;
}
void require_device(const void* pointer, std::size_t bytes) {
    const auto address = reinterpret_cast<CUdeviceptr>(pointer);
    for (const auto& [start, memory] : driver::driver.memory)
        if (address >= start && address - start <= memory.bytes.size() &&
            bytes <= memory.bytes.size() - (address - start)) {
            expect(memory.context == driver::current(), "native buffer has current context owner");
            return;
        }
    throw std::runtime_error("native Execute received an unowned or short allocation");
}
CudaSessionDispatch dispatch(bool has_profile_predecessor = false) {
    original_metadata = metadata::dispatch();
    auto sdk = original_metadata;
    sdk.create = driver::dispatch().create;
    sdk.backend_create = [](cudnnBackendDescriptorType_t kind, Native* output) {
        ++control.profile_callbacks;
        const auto status = original_metadata.backend_create(kind, output);
        if (status == CUDNN_STATUS_SUCCESS && kind == CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR &&
            control.is_executing.load()) {
            expect(private_packs.insert(*output).second, "fresh private pack identity");
            ++control.private_creates;
        }
        return status;
    };
    sdk.backend_destroy = [](Native value) {
        ++control.profile_callbacks;
        const bool is_private = private_packs.erase(value) != 0;
        const auto status = original_metadata.backend_destroy(value);
        if (is_private) ++control.private_destroys;
        return status;
    };
    sdk.backend_set = [](Native native, cudnnBackendAttributeName_t attribute,
                         cudnnBackendAttributeType_t value_type, std::int64_t count,
                         const void* input) {
        ++control.profile_callbacks;
        auto& object = metadata::object(native).snapshot;
        if (!is_planning(object.kind))
            return original_metadata.backend_set(native, attribute, value_type, count, input);
        metadata::callback();
        const auto name = static_cast<std::uint32_t>(attribute);
        const auto type = static_cast<std::uint32_t>(value_type);
        if (object.is_finalized || !input || count <= 0 || count > 100)
            return CUDNN_STATUS_BAD_PARAM;
        metadata::Attribute value{type, count, {}, {}};
        if (is_child(name)) {
            expect(type == 15, "descriptor-copy original ABI");
            const auto* sources = static_cast<const Native*>(input);
            for (std::int64_t i = 0; i != count; ++i)
                value.children.push_back(std::make_shared<const metadata::Snapshot>(
                    metadata::object(sources[i]).snapshot));
        } else {
            const auto width = type == 0 || type == 6 ? sizeof(void*) : metadata::width(type);
            const auto* first = static_cast<const std::byte*>(input);
            value.bytes.assign(first, first + static_cast<std::size_t>(count) * width);
            if (name == 800) {
                expect(count == 1 && type == 0, "original graph handle ABI");
                const auto handle = *static_cast<const cudnnHandle_t*>(input);
                (void)base::handle(handle);
            }
        }
        if (name == 801) {
            auto& previous = object.attributes[name];
            previous.type = type;
            previous.count += count;
            previous.children.insert(previous.children.end(), value.children.begin(),
                                     value.children.end());
        } else {
            object.attributes[name] = std::move(value);
        }
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.backend_finalize = [](Native native) {
        ++control.profile_callbacks;
        if (private_packs.contains(native) && control.fail_private_finalize.load())
            return CUDNN_STATUS_NOT_SUPPORTED;
        if (is_planning(metadata::object(native).snapshot.kind) &&
            metadata::object(native).snapshot.kind != 16)
            expect(driver::current(), "native planning has authorized context");
        return original_metadata.backend_finalize(native);
    };
    sdk.backend_get = [](Native native, cudnnBackendAttributeName_t attribute,
                         cudnnBackendAttributeType_t value_type, std::int64_t requested,
                         std::int64_t* count, void* output) {
        ++control.profile_callbacks;
        const auto& object = metadata::object(native).snapshot;
        if (!is_planning(object.kind))
            return original_metadata.backend_get(native, attribute, value_type, requested, count,
                                                 output);
        metadata::callback();
        const auto name = static_cast<std::uint32_t>(attribute);
        if (!object.is_finalized || (!count && !output) || (output && requested != 1))
            return CUDNN_STATUS_BAD_PARAM;
        if (name != 402) return CUDNN_STATUS_NOT_SUPPORTED;
        expect(value_type == CUDNN_TYPE_INT64, "workspace query uses native INT64");
        const std::int64_t bytes = kWorkspaceBytes;
        if (output) std::memcpy(output, &bytes, sizeof(bytes));
        if (count) *count = 1;
        return CUDNN_STATUS_SUCCESS;
    };
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): exact original SDK ABI.
    sdk.backend_execute = [](cudnnHandle_t handle, Native plan, Native pack) {
        ++control.profile_callbacks;
        (void)base::handle(handle);
        if (control.hold_next_context.exchange(false))
            control.held_context.store(reinterpret_cast<std::uintptr_t>(driver::current()));
        expect(metadata::object(plan).snapshot.kind == 5 &&
                   metadata::object(plan).snapshot.is_finalized && private_packs.contains(pack),
               "only copied finalized plan and command-private pack reach original Execute");
        const auto& values = metadata::object(pack).snapshot.attributes;
        const auto& uids = values.at(1000).bytes;
        const auto& pointers = values.at(1001).bytes;
        expect(uids.size() == 5 * sizeof(std::int64_t) && pointers.size() == 5 * sizeof(void*),
               "ordered duplicate UID table reaches original unchanged");
        void* input{};
        void* output{};
        void* coefficient{};
        for (std::size_t index = 0; index != 5; ++index) {
            std::int64_t uid{};
            void* pointer{};
            std::memcpy(&uid, uids.data() + index * sizeof(uid), sizeof(uid));
            std::memcpy(&pointer, pointers.data() + index * sizeof(pointer), sizeof(pointer));
            if (uid == 11) input = pointer;  // Deliberate last-match native model.
            if (uid == 12) output = pointer;
            if (uid == 13) coefficient = pointer;
            if (uid == 999)
                expect(pointer, "ignored non-NULL slot retains identity without payload access");
        }
        void* workspace{};
        std::memcpy(&workspace, values.at(1003).bytes.data(), sizeof(workspace));
        require_device(input, 4 * sizeof(float));
        require_device(output, 4 * sizeof(float));
        require_device(workspace, kWorkspaceBytes);
        expect(coefficient && reinterpret_cast<std::uintptr_t>(coefficient) % alignof(float) == 0,
               "by-value bytes have owned natural alignment");
        float scale{};
        std::memcpy(&scale, coefficient, sizeof(scale));
        const auto* x = static_cast<const float*>(input);
        auto* y = static_cast<float*>(output);
        for (std::size_t i = 0; i != 4; ++i) y[i] = scale * x[i];
        static_cast<std::byte*>(workspace)[0] = std::byte{0x5a};
        ++control.executions;
        return static_cast<cudnnStatus_t>(control.execute_status.load());
    };
    auto result = driver::session_dispatch();
    result.execution.event_query = [](CUevent event) {
        const auto& owner = driver::driver.events.at(event);
        expect(owner.context == driver::current(), "event observation uses its actual context");
        if (reinterpret_cast<std::uintptr_t>(owner.context) == control.held_context.load() &&
            !control.release_foreign.load()) {
            ++control.blocked_queries;
            return CUDA_ERROR_NOT_READY;
        }
        return driver::control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    result.execution.stream_synchronize = [](CUstream) {
        if (reinterpret_cast<std::uintptr_t>(driver::current()) == control.held_context.load() &&
            !control.release_foreign.load())
            return CUDA_ERROR_NOT_READY;
        return driver::control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    if (has_profile_predecessor)
        result.stream_wait_event = [](CUstream stream, CUevent event, unsigned int flags) {
            expect(driver::current() && !flags && driver::driver.events.contains(event) &&
                       (!stream || driver::driver.streams.at(stream).context == driver::current()),
                   "owned foreign event wait uses the selected target context and stream");
            ++control.profile_waits;
            return CUDA_SUCCESS;
        };
    result.dnn = std::make_shared<const CudaDnnDispatch>(sdk);
    result.verified_dnn_header_version = CUDNN_VERSION;
    return result;
}

// Both the GCC build and the older scoped Clang analysis understand these
// call-site default arguments; no production diagnostics depend on them.
ExecutionWireReply success(std::optional<ExecutionWireReply> reply, bool original = false,
                           const char* file = __builtin_FILE(),
                           unsigned int line = __builtin_LINE()) {
    if (!reply || reply->disposition != ExecutionWireDisposition::kComplete || reply->status ||
        reply->parameter_bytes || !reply->arguments.empty() ||
        (original ? reply->original_status != 0 : reply->original_status.has_value()))
        throw std::runtime_error(
            std::string(file) + ':' + std::to_string(line) +
            " expected success: status=" + std::to_string(reply ? reply->status : -1) +
            " original=" + std::to_string(reply ? reply->original_status.value_or(-1) : -1) +
            " flags=" + std::to_string(reply ? reply->parameter_bytes : 0) +
            " execute=" + std::to_string(control.executions.load()) +
            " private_create=" + std::to_string(control.private_creates.load()) +
            " private_destroy=" + std::to_string(control.private_destroys.load()) +
            " can_stop=" + std::to_string(driver::control.can_stop.load()));
    return std::move(*reply);
}
void rejected(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kRejected && reply->status &&
               !reply->original_status && !reply->parameter_bytes && !reply->id &&
               !reply->address && reply->data.empty() && reply->arguments.empty(),
           "pre-issue rejection has no native provenance, output or quarantine");
}
std::vector<std::byte> identity(std::uint64_t value) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(value), "encode identity");
    return writer.take();
}
void bind_dnn_profile(ExecutionSocketClient& client) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_profile(writer, CUDNN_VERSION), "encode selected DNN profile");
    const auto reply = success(client.exchange(ExecutionOpcode::kBindDnnProfile, writer.take()));
    ExecutionPayloadReader reader(reply.data);
    std::uint32_t version{};
    expect(!reply.id && !reply.address && read_execution_dnn_profile(reader, version) &&
               version == CUDNN_VERSION && reader.empty(),
           "profile binding echoes exact headers without object authority");
}
void open_metadata(ExecutionSocketClient& client) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u32(kExecutionDnnMetadataRole) && writer.write_u32(0), "metadata role");
    const auto reply = success(client.open(writer.take()));
    expect(decode_execution_bootstrap_ownership(reply.data) ==
               ExecutionBootstrapOwnership::kMetadataRetained,
           "DNM1 owns metadata without a context");
    bind_dnn_profile(client);
}
std::vector<std::byte> prepared_bytes(ExecutionDnnBackendAttributeOperation operation,
                                      Header header) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_prepare(writer, {operation, header}),
           "encode attribute prepare");
    return writer.take();
}
std::vector<std::byte> attribute_bytes(const ExecutionDnnBackendAttribute& attribute) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_attribute(writer, attribute), "encode attribute");
    return writer.take();
}
struct Client {
    ExecutionSocketClient& metadata;
    ExecutionSocketClient& execution;
    std::uint64_t handle{};

    std::optional<ExecutionWireReply> exchange(ExecutionOpcode operation,
                                               std::vector<std::byte> bytes, bool context = false) {
        if (!context) return metadata.exchange(operation, bytes);
        ExecutionPayloadWriter writer;
        expect(write_execution_dnn_backend_context(
                   writer, {metadata.session(), handle, operation, std::move(bytes)}),
               "encode authenticated planning envelope");
        return execution.exchange(ExecutionOpcode::kDnnBackendContext, writer.take());
    }
    std::uint64_t create(std::uint32_t kind) {
        ExecutionPayloadWriter writer;
        expect(write_execution_dnn_backend_create(writer, {kind, 1}), "encode backend create");
        const auto reply =
            success(metadata.exchange(ExecutionOpcode::kDnnBackendCreate, writer.take()), true);
        expect(reply.id && !reply.address && reply.data.empty(), "opaque created backend identity");
        return reply.id;
    }
    ExecutionDnnBackendAttributePreparation prepare(Header header, bool context = false) {
        auto reply = success(
            exchange(ExecutionOpcode::kDnnBackendPrepare,
                     prepared_bytes(ExecutionDnnBackendAttributeOperation::kSet, header), context));
        ExecutionPayloadReader reader(reply.data);
        ExecutionDnnBackendAttributePreparation result;
        expect(read_execution_dnn_backend_preparation(reader, result) && reader.empty(),
               "exact preparation");
        return result;
    }
    void set(Header header, Values values, bool context = false) {
        const auto preparation = prepare(header, context);
        success(
            exchange(ExecutionOpcode::kDnnBackendSet,
                     attribute_bytes({header, preparation.generation, std::move(values)}), context),
            true);
    }
    template <typename T>
    void scalars(std::uint64_t object, std::uint32_t name, std::uint32_t type,
                 std::span<const T> values, bool context = false) {
        set({object, name, type, static_cast<std::int64_t>(values.size()), 1},
            {type, metadata::bytes(values), {}}, context);
    }
    template <typename T>
    void scalar(std::uint64_t object, std::uint32_t name, std::uint32_t type, T value,
                bool context = false) {
        scalars(object, name, type, std::span<const T>{&value, 1}, context);
    }
    void reference(std::uint64_t object, std::uint32_t name, std::uint64_t child,
                   bool context = false) {
        set({object, name, 15, 1, 1},
            {15, {}, {{ExecutionDnnBackendReferenceKind::kDescriptor, metadata.session(), child}}},
            context);
    }
    void finalize(std::uint64_t object, bool context = false) {
        success(exchange(ExecutionOpcode::kDnnBackendFinalize, identity(object), context), true);
    }
    void destroy(std::uint64_t object) {
        success(metadata.exchange(ExecutionOpcode::kDnnBackendDestroy, identity(object)), true);
    }
    std::uint64_t tensor(std::int64_t uid, bool by_value = false) {
        const auto object = create(17);
        scalar(object, 900, 3, std::int64_t{4});
        scalar(object, 901, 1, std::uint32_t{0});
        const std::array<std::int64_t, 4> sizes{1, 1, 1, by_value ? 1 : 4}, strides{4, 4, 4, 1};
        scalars(object, 902, 3, std::span<const std::int64_t>{sizes});
        scalars(object, 903, 3, std::span<const std::int64_t>{strides});
        scalar(object, 906, 3, uid);
        if (by_value) scalar(object, 908, 2, true);
        finalize(object);
        return object;
    }
    std::uint64_t plan() {
        const auto x = tensor(11), y = tensor(12), coefficient = tensor(13, true),
                   pointwise = create(0), operation = create(13);
        scalar(pointwise, 0, 14, std::uint32_t{0});
        scalar(pointwise, 1, 1, std::uint32_t{0});
        finalize(pointwise);
        reference(operation, 750, pointwise);
        reference(operation, 751, x);
        reference(operation, 752, y);
        reference(operation, 753, coefficient);
        finalize(operation);
        for (const auto source : {x, y, coefficient, pointwise}) destroy(source);
        const auto graph = create(15);
        set({graph, 800, 0, 1, 1},
            {0, {}, {{ExecutionDnnBackendReferenceKind::kHandle, execution.session(), handle}}},
            true);
        reference(graph, 801, operation, true);
        destroy(operation);
        finalize(graph, true);
        const auto engine = create(2);
        reference(engine, 1300, graph, true);
        scalar(engine, 1301, 3, std::int64_t{0}, true);
        finalize(engine, true);
        destroy(graph);
        const auto config = create(3);
        reference(config, 300, engine, true);
        finalize(config, true);
        destroy(engine);
        const auto result = create(5);
        reference(result, 401, config, true);
        finalize(result, true);
        destroy(config);
        return result;
    }
    void configure_variant(std::uint64_t object) {
        const std::array<std::int64_t, 5> uids{11, 12, 13, 999, 11};
        scalars(object, 1000, 3, std::span<const std::int64_t>{uids});
        Values pointers{6, {}, {}};
        for (std::uint64_t slot = 101; slot != 106; ++slot)
            pointers.references.push_back(
                {ExecutionDnnBackendReferenceKind::kPointerSlot, metadata.session(), slot});
        set({object, 1001, 6, 5, 1}, std::move(pointers));
        set({object, 1003, 6, 1, 1},
            {6, {}, {{ExecutionDnnBackendReferenceKind::kPointerSlot, metadata.session(), 106}}});
        finalize(object);
    }
};

std::vector<std::byte> execute_bytes(const ExecutionDnnBackendExecute& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_execute(writer, value), "encode final Execute");
    return writer.take();
}
std::vector<std::byte> prepare_execute_bytes(const ExecutionDnnBackendExecutePrepare& value) {
    ExecutionPayloadWriter writer;
    expect(write_execution_dnn_backend_execute_prepare(writer, value),
           "encode Execute preparation");
    return writer.take();
}
ExecutionDnnBackendExecutePreparation prepare_execute(
    ExecutionSocketClient& execution, const ExecutionDnnBackendExecutePrepare& value) {
    auto reply = success(execution.exchange(ExecutionOpcode::kPrepareDnnBackendExecute,
                                            prepare_execute_bytes(value)));
    ExecutionPayloadReader reader(reply.data);
    ExecutionDnnBackendExecutePreparation result;
    expect(read_execution_dnn_backend_execute_preparation(reader, result) && reader.empty(),
           "exact Execute preparation");
    return result;
}
std::uint64_t allocate(ExecutionSocketClient& client, std::size_t bytes) {
    const auto reply = success(client.exchange(ExecutionOpcode::kAllocate, identity(bytes)), true);
    expect(reply.id && reply.address, "owned device allocation");
    return reply.id;
}
void upload(ExecutionSocketClient& client, std::uint64_t id, std::span<const std::byte> bytes) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(id) && writer.write_u64(0) && writer.write_blob(bytes),
           "encode upload");
    success(client.exchange(ExecutionOpcode::kUpload, writer.take()), true);
}
std::vector<std::byte> download(ExecutionSocketClient& client, std::uint64_t id,
                                std::size_t bytes) {
    ExecutionPayloadWriter writer;
    expect(writer.write_u64(id) && writer.write_u64(0) && writer.write_u64(bytes),
           "encode download");
    return success(client.exchange(ExecutionOpcode::kDownload, writer.take()), true).data;
}
ExecutionSocketSubmission submit(ExecutionSocketClient& client, ExecutionOpcode opcode,
                                 std::span<const std::byte> bytes = {}) {
    const auto result = client.submit(opcode, bytes);
    expect(result.status == ExecutionSocketSubmitStatus::kAccepted && result.sequence,
           "asynchronous transport submission accepted");
    return result;
}
std::optional<ExecutionWireReply> await_reply(ExecutionSocketClient& client,
                                              ExecutionSocketSubmission submission) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while (std::chrono::steady_clock::now() < deadline) {
        expect(client.poll_for(1), "bounded asynchronous reply transport remains healthy");
        if (auto result = client.take_reply(submission.sequence)) return result;
    }
    return std::nullopt;
}
void observe_pending(ExecutionSocketClient& client, ExecutionSocketSubmission submission) {
    const auto steps = control.service_steps.load();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
    while (control.service_steps.load() < steps + 8) {
        expect(client.poll_for(1) && std::chrono::steady_clock::now() < deadline,
               "service advances while the forbidden operation remains pending");
        expect(!client.take_reply(submission.sequence),
               "no premature reply while cleanup is blocked");
    }
}
void profile_rebind_pending(ExecutionSocketClient& execution, ExecutionSocketClient& source,
                            ExecutionSocketClient& metadata_client,
                            const ExecutionDnnBackendExecute& command) {
    ExecutionPayloadWriter event_flags;
    expect(event_flags.write_u32(CU_EVENT_DISABLE_TIMING), "encode owned predecessor event");
    const auto event =
        success(source.exchange(ExecutionOpcode::kCreateEvent, event_flags.take()), true).id;
    expect(event != 0, "predecessor event has an owned identity");
    driver::control.can_stop.store(false);
    ExecutionPayloadWriter record;
    expect(record.write_u64(event) && record.write_u64(0) && record.write_u32(0),
           "record exact predecessor event on source default stream");
    expect(success(source.exchange(ExecutionOpcode::kRecordEvent, record.take())).id != 0,
           "record admission retains its native physical frontier");
    ExecutionPayloadWriter wait;
    expect(wait.write_u64(source.session()) && wait.write_u64(event) && wait.write_u64(0) &&
               wait.write_u32(0),
           "capture authenticated foreign event as target predecessor");
    expect(success(execution.exchange(ExecutionOpcode::kWaitForeignEvent, wait.take())).id != 0,
           "target predecessor is accepted without claiming native completion");

    const auto prior_executions = control.executions.load();
    const auto prior_packs = control.private_creates.load();
    control.execute_status.store(CUDNN_STATUS_NOT_SUPPORTED);
    const auto pending =
        submit(execution, ExecutionOpcode::kDnnBackendExecute, execute_bytes(command));
    const auto barrier = submit(execution, ExecutionOpcode::kSnapshot);
    expect(barrier.sequence == pending.sequence + 1, "snapshot follows the exact DNN request");
    const auto snapshot = success(await_reply(execution, barrier));
    ExecutionPayloadReader observed(snapshot.data);
    std::array<std::uint64_t, 7> state{};
    for (auto& value : state) expect(observed.read_u64(value), "decode complete queue snapshot");
    std::uint32_t quarantined{};
    expect(observed.read_u32(quarantined) && observed.empty() && !quarantined && state[0] == 2 &&
               state[1] == 0 && !execution.take_reply(pending.sequence) &&
               control.executions.load() == prior_executions &&
               control.private_creates.load() == prior_packs && !control.profile_waits.load(),
           "snapshot proves accepted wait and DNN receipts, neither natively issued");

    const auto destroy_plan =
        submit(metadata_client, ExecutionOpcode::kDnnBackendDestroy, identity(command.plan));
    observe_pending(metadata_client, destroy_plan);
    ExecutionPayloadWriter wrong_profile;
    expect(write_execution_dnn_profile(wrong_profile, CUDNN_VERSION + 1U),
           "encode an exact but mismatched profile after accepted native ownership");
    const auto mismatch = await_reply(
        execution, submit(execution, ExecutionOpcode::kBindDnnProfile, wrong_profile.take()));
    rejected(mismatch);
    expect(mismatch->status == CUDA_ERROR_NOT_SUPPORTED &&
               !execution.take_reply(pending.sequence) &&
               !metadata_client.take_reply(destroy_plan.sequence),
           "sticky rebind rejection neither replaces accepted result nor releases its plan lease");
    const auto callbacks = control.profile_callbacks.load();
    const auto fresh = await_reply(
        execution, submit(execution, ExecutionOpcode::kDnnBackendExecute, execute_bytes(command)));
    rejected(fresh);
    expect(fresh->status == CUDA_ERROR_NOT_SUPPORTED &&
               control.profile_callbacks.load() == callbacks &&
               control.executions.load() == prior_executions &&
               control.private_creates.load() == prior_packs,
           "fresh DNN work after mismatch reaches no native metadata, preparation or Execute call");

    const auto synchronizes = driver::control.context_synchronizes.load();
    driver::control.can_stop.store(true);
    const auto original = await_reply(execution, pending);
    expect(
        original && original->disposition == ExecutionWireDisposition::kComplete &&
            original->status == CUDNN_STATUS_NOT_SUPPORTED &&
            original->original_status == CUDNN_STATUS_NOT_SUPPORTED && original->id &&
            !original->parameter_bytes && !original->address && original->data.empty() &&
            original->arguments.empty() && control.executions.load() == prior_executions + 1 &&
            driver::control.context_synchronizes.load() == synchronizes + 1,
        "accepted DNN returns its exact original native error once after predecessor retirement");
    success(execution.exchange(ExecutionOpcode::kDrain));
    success(source.exchange(ExecutionOpcode::kDrain));
    success(await_reply(metadata_client, destroy_plan), true);
    expect(control.profile_waits.load() == 1 && control.private_creates.load() == prior_packs + 1 &&
               control.private_destroys.load() == prior_packs + 1 &&
               !execution.take_reply(pending.sequence),
           "physical retirement releases one private pack and plan lease without replay or "
           "duplicate reply");
    const auto retired_callbacks = control.profile_callbacks.load();
    const auto later =
        execution.exchange(ExecutionOpcode::kDnnBackendExecute, execute_bytes(command));
    rejected(later);
    expect(later->status == CUDA_ERROR_NOT_SUPPORTED &&
               control.profile_callbacks.load() == retired_callbacks &&
               control.executions.load() == prior_executions + 1,
           "retiring accepted work does not restore admission after the sticky mismatch");
    success(source.exchange(ExecutionOpcode::kDestroyEvent, identity(event)));
}
void run_client(const std::string& path, TaskPeerIdentity peer, Mode mode) {
    ExecutionSocketClient metadata_client({path, peer, 3000}), other_metadata({path, peer, 3000}),
        other_execution({path, peer, 3000});
    auto execution_owner =
        std::make_unique<ExecutionSocketClient>(ExecutionSocketClientConfig{path, peer, 3000});
    auto& execution = *execution_owner;
    open_metadata(metadata_client);
    open_metadata(other_metadata);
    success(execution.open(), true);
    bind_dnn_profile(execution);
    success(other_execution.open(), true);
    bind_dnn_profile(other_execution);
    const auto handle = success(execution.exchange(ExecutionOpcode::kCreateDnn), true).id;
    Client client{metadata_client, execution, handle};
    const auto plan = client.plan(), variant = client.create(16);
    client.configure_variant(variant);
    ExecutionDnnBackendExecutePrepare request{handle, metadata_client.session(), plan, variant};
    const auto original = prepare_execute(execution, request);
    expect(original.requirements.size() == 6 && original.plan_generation &&
               original.variant_generation,
           "five original ordered slots plus workspace");
    for (std::size_t i = 0; i != 5; ++i) {
        const auto& value = original.requirements[i];
        expect(value.index == i && value.slot == i + 101 &&
                   value.role == ExecutionDnnBackendBindingRole::kTensor,
               "requirements retain table order and opaque slot identities");
        expect(value.value_class == (i == 2   ? Class::kByValue
                                     : i == 3 ? Class::kIgnored
                                              : Class::kDevice) &&
                   value.bytes == (i == 2   ? 4U
                                   : i == 3 ? 0U
                                            : 16U),
               "native Tensor observations determine extents");
    }
    expect(original.requirements.back().index == kExecutionDnnBackendWorkspaceIndex &&
               original.requirements.back().slot == 106 &&
               original.requirements.back().bytes == kWorkspaceBytes,
           "private original workspace query determines exact required bytes");
    rejected(metadata_client.exchange(ExecutionOpcode::kPrepareDnnBackendExecute,
                                      prepare_execute_bytes(request)));
    auto foreign = request;
    foreign.metadata = other_metadata.session();
    rejected(execution.exchange(ExecutionOpcode::kPrepareDnnBackendExecute,
                                prepare_execute_bytes(foreign)));
    foreign = request;
    foreign.metadata = other_execution.session();
    rejected(execution.exchange(ExecutionOpcode::kPrepareDnnBackendExecute,
                                prepare_execute_bytes(foreign)));
    foreign = request;
    foreign.plan = variant;
    rejected(execution.exchange(ExecutionOpcode::kPrepareDnnBackendExecute,
                                prepare_execute_bytes(foreign)));
    expect(control.executions.load() == 0 && control.private_creates.load() == 0,
           "planning cannot execute or create private packs");

    const auto x_first = allocate(execution, 24), x_last = allocate(execution, 24),
               y = allocate(execution, 24), workspace = allocate(execution, 40),
               short_range = allocate(execution, 15);
    const std::array<float, 6> first{91, 100, 100, 100, 100, 92}, last{91, 1, -2, 3, -4, 92},
        output{71, 0, 0, 0, 0, 72};
    upload(execution, x_first, std::as_bytes(std::span{first}));
    upload(execution, x_last, std::as_bytes(std::span{last}));
    upload(execution, y, std::as_bytes(std::span{output}));
    ExecutionDnnBackendExecute command{
        handle,  metadata_client.session(), plan,
        variant, original.plan_generation,  original.variant_generation,
        {}};
    for (const auto& value : original.requirements) {
        ExecutionDnnBackendExecuteBinding binding{value.index, value.value_class, value.slot, 0, 0,
                                                  {}};
        if (value.index == 0) binding.allocation = x_first;
        if (value.index == 1) binding.allocation = y;
        if (value.index == 4) binding.allocation = x_last;
        if (value.index == kExecutionDnnBackendWorkspaceIndex) binding.allocation = workspace;
        if (binding.allocation) binding.offset = 4;
        if (value.index == 2) binding.host_bytes = metadata::scalar(2.5F);
        command.bindings.push_back(std::move(binding));
    }
    std::vector<std::uint64_t> foreign_allocations;
    do {
        foreign_allocations.push_back(allocate(other_execution, 32));
    } while (foreign_allocations.back() <= short_range);
    expect(foreign_allocations.back() > short_range, "foreign-only numeric allocation identity");
    for (unsigned int corruption = 0; corruption != 9; ++corruption) {
        auto bad = command;
        if (corruption == 0) ++bad.plan_generation;
        if (corruption == 1) ++bad.bindings[0].slot;
        if (corruption == 2) bad.bindings[0].allocation = foreign_allocations.back();
        if (corruption == 3) {
            bad.bindings[0].allocation = short_range;
            bad.bindings[0].offset = 0;
        }
        if (corruption == 4) bad.bindings[0].offset = 12;
        if (corruption == 5) bad.metadata = other_metadata.session();
        if (corruption == 6) bad.bindings[2].host_bytes.resize(8);
        if (corruption == 7) bad.bindings.back().offset = 20;
        if (corruption == 8) bad.bindings[0].allocation = UINT64_MAX;
        rejected(execution.exchange(ExecutionOpcode::kDnnBackendExecute, execute_bytes(bad)));
    }
    success(metadata_client.exchange(ExecutionOpcode::kDnnBackendInitialize, identity(variant)),
            true);
    client.configure_variant(variant);
    rejected(execution.exchange(ExecutionOpcode::kDnnBackendExecute, execute_bytes(command)));
    const auto fresh = prepare_execute(execution, request);
    expect(fresh.variant_generation != original.variant_generation,
           "reset and rebind changes authoritative generation");
    command.variant_generation = fresh.variant_generation;
    expect(control.executions.load() == 0 && control.private_creates.load() == 0,
           "all rejected bindings remain pre-issue");

    control.is_executing.store(true);
    driver::control.can_stop.store(false);
    auto reply = success(
        execution.exchange(ExecutionOpcode::kDnnBackendExecute, execute_bytes(command)), true);
    expect(reply.id && reply.data.empty() && control.executions.load() == 1 &&
               control.private_creates.load() == 1 && control.private_destroys.load() == 0,
           "original host ACK precedes physical private-pack cleanup");
    const auto while_pending = prepare_execute(execution, request);
    expect(while_pending.plan_generation == fresh.plan_generation &&
               control.private_destroys.load() == 0,
           "read-only planning neither drains nor consumes the pending physical lease");
    driver::control.can_stop.store(true);
    success(execution.exchange(ExecutionOpcode::kDrain));
    expect(control.private_destroys.load() == 1,
           "private pack destroys exactly once after physical stop");
    const auto result = download(execution, y, sizeof(output));
    const std::array<float, 6> expected{71, 2.5F, -5, 7.5F, -10, 72};
    expect(result == metadata::bytes(std::span<const float>{expected}),
           "last duplicate UID, by-value capture and all output guards");
    expect(download(execution, x_first, sizeof(first)) ==
                   metadata::bytes(std::span<const float>{first}) &&
               download(execution, x_last, sizeof(last)) ==
                   metadata::bytes(std::span<const float>{last}),
           "all original input and guard bytes remain intact");
    std::vector<std::byte> expected_workspace(40, std::byte{0xcd});
    expected_workspace[4] = std::byte{0x5a};
    expect(download(execution, workspace, expected_workspace.size()) == expected_workspace,
           "workspace writes remain inside its original owned interval");

    control.execute_status.store(CUDNN_STATUS_NOT_SUPPORTED);
    const auto synchronizes = driver::control.context_synchronizes.load();
    const auto native_error =
        execution.exchange(ExecutionOpcode::kDnnBackendExecute, execute_bytes(command));
    expect(native_error && native_error->disposition == ExecutionWireDisposition::kComplete &&
               native_error->status == CUDNN_STATUS_NOT_SUPPORTED &&
               native_error->original_status == CUDNN_STATUS_NOT_SUPPORTED && native_error->id &&
               !native_error->parameter_bytes &&
               driver::control.context_synchronizes.load() == synchronizes + 1,
           "issued original error survives hidden-stream stop proof unchanged");
    success(execution.exchange(ExecutionOpcode::kDrain));
    expect(control.executions.load() == 2 && control.private_destroys.load() == 2,
           "native-error private pack also retires once");
    control.execute_status.store(CUDNN_STATUS_SUCCESS);
    if (mode == Mode::kProfileRebindPending) {
        profile_rebind_pending(execution, other_execution, metadata_client, command);
        expect(download(execution, y, sizeof(output)) ==
                       metadata::bytes(std::span<const float>{expected}) &&
                   download(execution, x_first, sizeof(first)) ==
                       metadata::bytes(std::span<const float>{first}) &&
                   download(execution, x_last, sizeof(last)) ==
                       metadata::bytes(std::span<const float>{last}) &&
                   download(execution, workspace, expected_workspace.size()) == expected_workspace,
               "accepted native error retains exact numerical effects and every input/output "
               "guard");
        control.is_executing.store(false);
        client.destroy(variant);  // The accepted command's plan was retired by its pending destroy.
        for (const auto allocation : foreign_allocations)
            success(other_execution.exchange(ExecutionOpcode::kRelease, identity(allocation)),
                    true);
        success(execution.exchange(ExecutionOpcode::kClose), true);
        success(other_execution.exchange(ExecutionOpcode::kClose), true);
        success(metadata_client.exchange(ExecutionOpcode::kClose), true);
        success(other_metadata.exchange(ExecutionOpcode::kClose), true);
        return;
    }
    const bool has_peer_command = mode == Mode::kForeignPending || mode == Mode::kQueuedDemand;
    ExecutionDnnBackendExecute peer_command = command;
    if (has_peer_command) {
        peer_command.handle =
            success(other_execution.exchange(ExecutionOpcode::kCreateDnn), true).id;
        const auto peer_preparation = prepare_execute(
            other_execution, {peer_command.handle, metadata_client.session(), plan, variant});
        expect(peer_preparation.plan_generation == command.plan_generation &&
                   peer_preparation.variant_generation == command.variant_generation,
               "copied same-device plan is usable with another authenticated execution owner");
        peer_command.bindings[0].allocation = foreign_allocations[0];
        peer_command.bindings[1].allocation = foreign_allocations[1];
        peer_command.bindings[4].allocation = foreign_allocations[2];
        peer_command.bindings.back().allocation = foreign_allocations[3];
        peer_command.bindings.back().offset = 0;
        upload(other_execution, foreign_allocations[0], std::as_bytes(std::span{first}));
        upload(other_execution, foreign_allocations[1], std::as_bytes(std::span{output}));
        upload(other_execution, foreign_allocations[2], std::as_bytes(std::span{last}));
    }
    if (mode == Mode::kForeignPending) {
        control.release_foreign.store(false);
        control.hold_next_context.store(true);
        success(other_execution.exchange(ExecutionOpcode::kDnnBackendExecute,
                                         execute_bytes(peer_command)),
                true);
        expect(control.executions.load() == 3 && control.private_destroys.load() == 2 &&
                   control.held_context.load(),
               "another context has genuine issued physical work");
    }
    control.fail_private_finalize.store(true);
    const auto private_error =
        execution.exchange(ExecutionOpcode::kDnnBackendExecute, execute_bytes(command));
    expect(private_error && private_error->disposition == ExecutionWireDisposition::kRejected &&
               private_error->status == CUDA_ERROR_UNKNOWN && private_error->id &&
               !private_error->original_status && private_error->parameter_bytes == 2 &&
               !private_error->address && private_error->data.empty() &&
               private_error->arguments.empty(),
           "private-preparation failure faults admission but never invents Execute provenance");
    control.fail_private_finalize.store(false);
    // The rejected preparation quarantines new queue admissions, not proven
    // native ownership. Close still waits for physical stop and cleans once.
    if (mode == Mode::kDisconnect) {
        execution_owner.reset();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
        while (!driver::control.context_destroys.load()) {
            expect(std::chrono::steady_clock::now() < deadline,
                   "disconnected clean-quarantined session completes proven cleanup");
            std::this_thread::yield();
        }
    } else if (mode == Mode::kForeignPending) {
        const auto close = submit(execution, ExecutionOpcode::kClose);
        observe_pending(execution, close);
        expect(control.blocked_queries.load() && !driver::control.context_destroys.load() &&
                   control.executions.load() == 3 && control.private_destroys.load() == 3,
               "foreign physical work forbids cleanup without issuing or destroying that peer");
        control.release_foreign.store(true);
        success(await_reply(execution, close), true);
        success(other_execution.exchange(ExecutionOpcode::kDrain));
    } else if (mode == Mode::kQueuedDemand) {
        const auto queued = submit(other_execution, ExecutionOpcode::kDnnBackendExecute,
                                   execute_bytes(peer_command));
        const auto barrier = submit(other_execution, ExecutionOpcode::kSnapshot);
        expect(barrier.sequence == queued.sequence + 1, "same-connection ordered barrier");
        success(await_reply(other_execution, barrier));
        // The later metadata reply proves ingestion of the earlier Execute,
        // not admission or permission to issue it while another owner is faulted.
        expect(!other_execution.take_reply(queued.sequence),
               "ingested queued demand has no premature execution reply");
        observe_pending(other_execution, queued);
        expect(control.executions.load() == 2 && control.private_creates.load() == 3 &&
                   !driver::control.context_destroys.load(),
               "queued demand has no issue authority while another owner is quarantined");
        success(execution.exchange(ExecutionOpcode::kClose), true);
        success(await_reply(other_execution, queued), true);
        success(other_execution.exchange(ExecutionOpcode::kDrain));
        expect(driver::control.context_destroys.load() == 1,
               "cleanup does not deadlock against queued-only demand");
    } else {
        success(execution.exchange(ExecutionOpcode::kClose), true);
    }
    control.is_executing.store(false);
    const auto expected_executions = has_peer_command ? 3U : 2U;
    expect(control.executions.load() == expected_executions &&
               control.private_creates.load() == expected_executions + 1 &&
               control.private_destroys.load() == expected_executions + 1,
           "private preparation error cleans its native pack at proven context shutdown");
    // Copied host planning metadata outlives the original native handle/context.
    client.destroy(plan);
    client.destroy(variant);
    for (const auto allocation : foreign_allocations)
        success(other_execution.exchange(ExecutionOpcode::kRelease, identity(allocation)), true);
    success(other_execution.exchange(ExecutionOpcode::kClose), true);
    success(metadata_client.exchange(ExecutionOpcode::kClose), true);
    success(other_metadata.exchange(ExecutionOpcode::kClose), true);
}
void run(const std::string& path, TaskPeerIdentity peer, Mode mode) {
    metadata::reset();
    driver::reset();
    control.executions.store(0);
    control.private_creates.store(0);
    control.private_destroys.store(0);
    control.execute_status.store(CUDNN_STATUS_SUCCESS);
    control.is_executing.store(false);
    control.fail_private_finalize.store(false);
    control.hold_next_context.store(false);
    control.release_foreign.store(true);
    control.held_context.store(0);
    control.service_steps.store(0);
    control.blocked_queries.store(0);
    control.profile_callbacks.store(0);
    control.profile_waits.store(0);
    private_packs.clear();
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    // The queued-demand ingestion barrier needs one request beyond the blocked
    // Execute; client submission capacity alone does not grant server capacity.
    config.transport.max_pending_requests = 2;
    config.uses_original_kernels = true;
    config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
    ExecutionService service(config, dispatch(mode == Mode::kProfileRebindPending));
    expect(service.start(), "start authenticated backend execution service");
    auto client = std::async(std::launch::async, [&] { run_client(path, peer, mode); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1) && std::chrono::steady_clock::now() < deadline,
               "bounded planning/execution service exchange");
        ++control.service_steps;
    }
    client.get();
    expect(service.request_stop() && metadata::alive() == 0 && private_packs.empty() &&
               driver::driver.contexts.empty() && driver::driver.memory.empty() &&
               driver::driver.events.empty() &&
               base::state.handle_creates == base::state.handle_destroys,
           "every metadata/native handle/allocation/event owner cleans once");
}
}  // namespace

int main(int argc, char** argv) {
    Mode mode = Mode::kClose;
    if (argc == 2 && std::string_view(argv[1]) == "--disconnect")
        mode = Mode::kDisconnect;
    else if (argc == 2 && std::string_view(argv[1]) == "--foreign-pending")
        mode = Mode::kForeignPending;
    else if (argc == 2 && std::string_view(argv[1]) == "--queued-demand")
        mode = Mode::kQueuedDemand;
    else if (argc == 2 && std::string_view(argv[1]) == "--profile-rebind-pending")
        mode = Mode::kProfileRebindPending;
    else if (argc != 1)
        return EXIT_FAILURE;
    char pattern[] = "/tmp/glimmer-backend-execute-service-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    const auto child = ::fork();
    if (child < 0) {
        (void)::rmdir(directory);
        return EXIT_FAILURE;
    }
    if (child > 0) {
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        const bool success =
            waited == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS;
        return ::rmdir(directory) == 0 && success ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    int result = EXIT_SUCCESS;
    try {
        const auto peer = current_process_identity();
        expect(peer.has_value(), "authenticated process identity");
        run(std::string(directory) + "/execute.sock", *peer, mode);
        std::cout << "Backend graph authenticated planning and execution checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    return result;
}
