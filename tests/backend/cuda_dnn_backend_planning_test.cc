#include "dnn_convolution_fixture.h"
#include "internal/cuda_dnn_backend_execution.h"

#include <functional>
#include <iostream>
#include <map>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
namespace foundation = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
using Header = CudaDnnBackendAttributeHeader;
using Values = CudaDnnBackendAttributeValues;
using Operation = CudaDnnBackendContextOperation;
using AttributeOperation = CudaDnnBackendAttributeOperation;
using Native = cudnnBackendDescriptor_t;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void native(const CudaDnnResult& value, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    expect(value.disposition == CudaDnnDisposition::kComplete && value.status == CUDA_SUCCESS &&
               value.original_status == status,
           "exact original native metadata result");
}
void driver(const CudaSessionResult& value) {
    expect(value.disposition == CudaSessionDisposition::kComplete && value.status == CUDA_SUCCESS,
           "owned fake Driver result");
}
template <typename T>
std::vector<std::byte> bytes(std::span<const T> value) {
    const auto view = std::as_bytes(value);
    return {view.begin(), view.end()};
}
template <typename T>
std::vector<std::byte> scalar(T value) {
    return bytes(std::span<const T>{&value, 1});
}
struct Snapshot;
struct Attribute {
    std::uint32_t type{};
    std::vector<std::byte> scalars;
    std::vector<std::shared_ptr<const Snapshot>> children;
    std::vector<void*> pointers;
};
struct Snapshot {
    std::uint32_t kind{};
    bool finalized{};
    std::map<std::uint32_t, Attribute> attributes;
};
struct Object {
    bool alive{};
    Snapshot value;
};
struct State {
    std::array<Object, 128> objects;
    unsigned int creates{}, destroys{}, initializes{}, sets{}, gets{}, finalizes{}, versions{};
    unsigned int switches{}, stream_gets{};
    Native last_destination{};
    std::vector<void*> logged_handles;
    std::function<void(unsigned int)> callback;
    cudnnStatus_t set_status{CUDNN_STATUS_SUCCESS}, get_status{CUDNN_STATUS_SUCCESS};
    cudnnStatus_t finalize_status{CUDNN_STATUS_SUCCESS};
    bool lose_context{};
};
State state;
constexpr std::int64_t kWorkspace = INT64_C(0x100000017);
void callback(unsigned int phase) {
    if (state.callback) state.callback(phase);
    if (state.lose_context && phase != 0) foundation::driver.current = nullptr;
}
Object& object(Native native) {
    for (auto& value : state.objects)
        if (&value == native) {
            expect(value.alive, "independent original descriptor still alive");
            return value;
        }
    throw std::runtime_error("unknown original planning descriptor");
}
bool child(std::uint32_t name) {
    return name == 801 || name == 1300 || name == 300 || name == 401;
}
std::size_t width(std::uint32_t type) {
    return type == 3 || type == 5 ? 8 : type == 2 ? 1 : 4;
}
CudaDnnDispatch library() {
    auto sdk = foundation::dispatch();
    sdk.version = [] {
        ++state.versions;
        callback(0);
        return std::size_t{91002};
    };
    sdk.backend_create = [](cudnnBackendDescriptorType_t kind, Native* output) {
        expect(output, "original output present");
        ++state.creates;
        for (auto& value : state.objects)
            if (!value.alive) {
                value = {};
                value.alive = true;
                value.value.kind = static_cast<std::uint32_t>(kind);
                *output = &value;
                return CUDNN_STATUS_SUCCESS;
            }
        return CUDNN_STATUS_ALLOC_FAILED;
    };
    sdk.backend_destroy = [](Native descriptor) {
        auto& value = object(descriptor);
        ++state.destroys;
        // Config/plan copies are host-owned resources in this fixture; no
        // permanent context or original source-handle pin is invented.
        value.alive = false;
        callback(5);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.backend_initialize = [](Native descriptor) {
        auto& value = object(descriptor).value;
        ++state.initializes;
        const auto kind = value.kind;
        value = {};
        value.kind = kind;
        callback(4);
        return CUDNN_STATUS_SUCCESS;
    };
    sdk.backend_set = [](Native descriptor, cudnnBackendAttributeName_t native_name,
                         cudnnBackendAttributeType_t native_type, std::int64_t count,
                         const void* input) {
        auto& value = object(descriptor).value;
        ++state.sets;
        const auto name = static_cast<std::uint32_t>(native_name);
        const auto type = static_cast<std::uint32_t>(native_type);
        if (name == 400 || name == 800) {
            expect(type == 0, "original handle attribute type");
            state.logged_handles.clear();
            // Native logging visits the original positive prefix before the
            // functional count check. The second handle must not be replaced
            // with a dummy or the currently selected frame handle.
            if (input && count > 0 && count <= 100) {
                auto* const* handles = static_cast<void* const*>(input);
                state.logged_handles.assign(handles, handles + count);
                for (auto* handle : state.logged_handles)
                    if (handle) (void)base::handle(static_cast<cudnnHandle_t>(handle));
            }
            if (!input || count != 1) {
                callback(1);
                return CUDNN_STATUS_BAD_PARAM;
            }
            if (name == 800 && !state.logged_handles.front()) {
                callback(1);
                return CUDNN_STATUS_BAD_PARAM_NULL_POINTER;
            }
        }
        if (value.finalized || !input || count <= 0 || count > 100) return CUDNN_STATUS_BAD_PARAM;
        Attribute attribute{type, {}, {}, {}};
        if (child(name)) {
            expect(type == 15, "original descriptor-copy ABI");
            auto* const* inputs = static_cast<Native const*>(input);
            for (std::int64_t i = 0; i != count; ++i)
                attribute.children.push_back(
                    std::make_shared<const Snapshot>(object(inputs[i]).value));
        } else if (type == 0 || type == 6) {
            auto* const* inputs = static_cast<void* const*>(input);
            attribute.pointers.assign(inputs, inputs + count);
            if (name == 800 && inputs[0]) (void)base::handle(static_cast<cudnnHandle_t>(inputs[0]));
        } else {
            const auto* first = static_cast<const std::byte*>(input);
            attribute.scalars.assign(first, first + static_cast<std::size_t>(count) * width(type));
        }
        if (name == 801) {
            auto& existing = value.attributes[name];
            existing.type = type;
            existing.children.insert(existing.children.end(), attribute.children.begin(),
                                     attribute.children.end());
        } else
            value.attributes[name] = std::move(attribute);
        callback(1);
        // A failing setter can have already changed the independent native state.
        return state.set_status;
    };
    sdk.backend_finalize = [](Native descriptor) {
        auto& value = object(descriptor).value;
        ++state.finalizes;
        if (value.kind == 2 || value.kind == 3 || value.kind == 5 || value.kind == 15)
            expect(foundation::current(), "planning finalization has a current device context");
        value.finalized = state.finalize_status == CUDNN_STATUS_SUCCESS;
        callback(3);
        return state.finalize_status;
    };
    sdk.backend_get = [](Native descriptor, cudnnBackendAttributeName_t native_name,
                         cudnnBackendAttributeType_t native_type, std::int64_t requested,
                         std::int64_t* count, void* output) {
        auto& value = object(descriptor).value;
        ++state.gets;
        const auto name = static_cast<std::uint32_t>(native_name);
        const auto type = static_cast<std::uint32_t>(native_type);
        if (!value.finalized || (!count && !output) || (output && requested != 1))
            return CUDNN_STATUS_BAD_PARAM;
        if ((value.kind == 15 && (name == 800 || name == 801)) ||
            (value.kind == 16 && name != 1003))
            return CUDNN_STATUS_NOT_SUPPORTED;
        if (output) {
            if (child(name)) {
                expect(type == 15 && value.attributes.at(name).children.size() == 1,
                       "original child getter uses one precreated destination");
                Native destination{};
                std::memcpy(&destination, output, sizeof(destination));
                auto& target = object(destination).value;
                const auto& source = *value.attributes.at(name).children.front();
                expect(target.kind == source.kind, "matching native destination kind");
                target = source;
                state.last_destination = destination;
            } else if (type == 0 || type == 6) {
                void* result{};
                const auto found = value.attributes.find(name);
                if (found != value.attributes.end() && !found->second.pointers.empty())
                    result = found->second.pointers.front();
                // Pointer identity only: never read a host slot or old handle.
                std::memcpy(output, &result, sizeof(result));
            } else {
                const auto result = name == 802   ? scalar(std::int64_t{7})
                                    : name == 402 ? scalar(kWorkspace)
                                                  : value.attributes.at(name).scalars;
                expect(result.size() >= width(type), "complete original scalar storage");
                std::memcpy(output, result.data(), width(type));
            }
        }
        if (count) *count = 1;
        callback(2);
        return state.get_status;
    };
    sdk.get_stream = [](cudnnHandle_t handle, cudaStream_t* stream) {
        ++state.stream_gets;
        return base::dispatch().get_stream(handle, stream);
    };
    return sdk;
}

struct Peer {
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t owner{}, handle{};
    CUcontext context{};
};
struct Finalization {
    std::size_t peer{};
    std::uint64_t descriptor{};
};
struct Owner {
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> registry;
    std::vector<Peer> peers;
    Owner() {
        foundation::reset();
        state = {};
        sdk = foundation::session_dispatch();
        sdk.context_set_current = [](CUcontext context) {
            ++state.switches;
            return foundation::session_dispatch().context_set_current(context);
        };
        sdk.dnn = std::make_shared<const CudaDnnDispatch>(library());
        registry = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        peers.reserve(3);
    }
    std::size_t add_peer(int ordinal) {
        expect(sdk.context_set_current(nullptr) == CUDA_SUCCESS, "detach before new context");
        const auto id = std::uint64_t{1001} + peers.size();
        auto session =
            std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{991, id}, id, sdk);
        driver(session->initialize(ordinal));
        const auto handle = session->create_dnn();
        native(handle);
        peers.push_back({std::move(session), id, handle.id, foundation::current()});
        return peers.size() - 1;
    }
    std::uint64_t create(std::uint32_t kind) {
        const auto value = registry->create_backend(kind);
        native(value);
        return value.id;
    }
    CudaDnnBackendContextResult call(std::size_t peer, const CudaDnnBackendContextRequest& request,
                                     bool activate = true) {
        auto& owner = peers.at(peer);
        if (activate) driver(owner.session->activate());
        return owner.session->backend_context(owner.owner, owner.handle, *registry, request);
    }
    CudaDnnBackendAttributePreparation prepare(std::size_t peer, AttributeOperation operation,
                                               const Header& header) {
        CudaDnnBackendContextRequest request;
        request.operation = Operation::kPrepare;
        request.attribute_operation = operation;
        request.header = header;
        auto result = call(peer, request);
        expect(result.result.status == CUDA_SUCCESS && !result.result.original_status &&
                   result.preparation.generation,
               "session-authorized preparation has no invented native metadata status");
        return result.preparation;
    }
    CudaDnnBackendAttributeResult attribute(std::size_t peer, Operation operation,
                                            const Header& header, const Values& values = {}) {
        const auto ready = prepare(
            peer,
            operation == Operation::kSet ? AttributeOperation::kSet : AttributeOperation::kGet,
            header);
        CudaDnnBackendContextRequest request;
        request.operation = operation;
        request.header = header;
        request.generation = ready.generation;
        request.values = values;
        return call(peer, request).attribute;
    }
    void finalize(Finalization target) {
        CudaDnnBackendContextRequest request;
        request.operation = Operation::kFinalize;
        request.id = target.descriptor;
        native(call(target.peer, request).result);
    }
    void set_child(std::size_t peer, std::uint64_t target, std::uint32_t name,
                   std::initializer_list<std::uint64_t> children) {
        Values values;
        values.descriptors = children;
        native(attribute(peer, Operation::kSet,
                         {target, name, 15, static_cast<std::int64_t>(children.size()), 1}, values)
                   .result);
    }
    template <typename T>
    void set_scalar(std::size_t peer, std::uint64_t target, std::uint32_t name, std::uint32_t type,
                    T value) {
        Values values;
        values.scalars = scalar(value);
        native(attribute(peer, Operation::kSet, {target, name, type, 1, 1}, values).result);
    }
    std::shared_ptr<const CudaDnnBackendSnapshot> snapshot(std::uint64_t id) {
        const auto value = registry->borrow_backend(id);
        expect(bool(value), "owned metadata snapshot");
        return value->snapshot;
    }
    Native native_id(std::uint64_t id) {
        const auto value = registry->borrow_backend(id);
        expect(bool(value), "owned native metadata identity");
        return value->object.native;
    }
    void close_peer(std::size_t peer) {
        auto& value = peers.at(peer);
        if (!value.session) return;
        driver(value.session->activate());
        driver(value.session->shutdown());
        value.session.reset();
    }
    void close() {
        for (std::size_t i = 0; i != peers.size(); ++i) close_peer(i);
        expect(!foundation::current(), "context-first shutdown left no current CUDA owner");
        registry->close();
        expect(registry->shutdown().status == CUDA_SUCCESS && registry->snapshot().is_closed &&
                   state.creates == state.destroys,
               "all copied planning objects clean exactly once after their contexts close");
    }
};
const CudaDnnBackendSnapshotAttribute& saved(const CudaDnnBackendSnapshot& snapshot,
                                             std::uint32_t name) {
    const auto found =
        std::ranges::find(snapshot.attributes, name, &CudaDnnBackendSnapshotAttribute::name);
    expect(found != snapshot.attributes.end(), "expected immutable copied attribute");
    return *found;
}

void copied_planning_and_context_lifetimes() {
    Owner owner;
    const auto first = owner.add_peer(0);
    const auto operation_a = owner.create(13), operation_b = owner.create(13);
    owner.set_scalar(first, operation_a, 754, 4, 1.25F);
    owner.set_scalar(first, operation_b, 754, 4, 2.5F);
    owner.finalize({.peer = first, .descriptor = operation_a});
    owner.finalize({.peer = first, .descriptor = operation_b});
    const auto graph = owner.create(15);
    Values handle;
    handle.handles.push_back({owner.peers[first].owner, owner.peers[first].handle});
    native(owner.attribute(first, Operation::kSet, {graph, 800, 0, 1, 1}, handle).result);
    owner.set_child(first, graph, 801, {operation_a});
    owner.set_child(first, graph, 801, {operation_b});
    owner.finalize({.peer = first, .descriptor = graph});
    const auto& originals = object(owner.native_id(graph)).value.attributes.at(801).children;
    const auto copied = owner.snapshot(graph);
    expect(originals.size() == 2 && saved(*copied, 801).descriptors.size() == 2 &&
               originals[0]->attributes.at(754).scalars == scalar(1.25F) &&
               originals[1]->attributes.at(754).scalars == scalar(2.5F) &&
               saved(*saved(*copied, 801).descriptors[0], 754).scalars == scalar(1.25F) &&
               saved(*saved(*copied, 801).descriptors[1], 754).scalars == scalar(2.5F),
           "repeated original OPS appends both independent copied trees in order");
    native(owner.registry->initialize_backend(operation_a));
    native(owner.registry->destroy(operation_a));
    native(owner.registry->destroy(operation_b));
    const auto engine = owner.create(2);
    owner.set_child(first, engine, 1300, {graph});
    owner.set_scalar(first, engine, 1301, 3, std::int64_t{3});
    owner.finalize({.peer = first, .descriptor = engine});
    const auto config = owner.create(3);
    owner.set_child(first, config, 300, {engine});
    owner.finalize({.peer = first, .descriptor = config});
    const auto plan = owner.create(5);
    owner.set_child(first, plan, 401, {config});
    native(owner.attribute(first, Operation::kSet, {plan, 400, 0, 1, 1}, handle).result);
    owner.finalize({.peer = first, .descriptor = plan});
    for (const auto source : {graph, engine, config}) {
        native(owner.registry->initialize_backend(source));
        native(owner.registry->destroy(source));
    }
    const auto destination = owner.create(3);
    const auto identity = owner.native_id(destination);
    Values targets;
    targets.descriptors.push_back(destination);
    const auto result = owner.attribute(first, Operation::kGet, {plan, 401, 15, 1, 3}, targets);
    native(result.result);
    expect(state.last_destination == identity && owner.native_id(destination) == identity &&
               result.count == 1 && result.count_bytes == 8,
           "native descriptor Get copies into the caller's original destination identity");
    const auto& native_config = object(identity).value;
    const auto& native_engine = *native_config.attributes.at(300).children.front();
    const auto& native_graph = *native_engine.attributes.at(1300).children.front();
    const auto& native_ops = native_graph.attributes.at(801).children;
    expect(native_ops.size() == 2 && native_ops[1]->attributes.at(754).scalars == scalar(2.5F),
           "native plan copied value graph survives reset/destruction of every original source");
    const auto generation = owner.snapshot(destination)->generation;
    state.get_status = CUDNN_STATUS_BAD_PARAM;
    const auto partial = owner.attribute(first, Operation::kGet, {plan, 401, 15, 1, 3}, targets);
    native(partial.result, CUDNN_STATUS_BAD_PARAM);
    expect(!partial.count_bytes && partial.values.descriptors.empty() &&
               !owner.snapshot(destination)->is_valid &&
               owner.snapshot(destination)->generation > generation &&
               owner.snapshot(plan)->is_valid && !owner.registry->snapshot().is_quarantined,
           "partially written Get destination invalidates only its configuration and publishes no "
           "output");
    state.get_status = CUDNN_STATUS_SUCCESS;
    native(owner.registry->initialize_backend(destination));
    native(owner.attribute(first, Operation::kGet, {plan, 401, 15, 1, 3}, targets).result);
    expect(owner.snapshot(destination)->is_valid && owner.native_id(destination) == identity,
           "original reset and successful copied Get recover destination without replacing its "
           "handle");
    const auto workspace = owner.attribute(first, Operation::kGet, {plan, 402, 3, 1, 3});
    native(workspace.result);
    expect(workspace.values.scalars == scalar(kWorkspace) && workspace.count == 1,
           "original full-width workspace result is never truncated to32bits");
    const auto source_lifetime = owner.snapshot(plan)->context->lifetime;
    owner.close_peer(first);
    expect(source_lifetime.expired(),
           "planning provenance does not permanently pin source context");
    const Header historical_header{plan, 400, 0, 1, 3};
    const auto historical_preparation =
        owner.registry->prepare_backend_attribute(AttributeOperation::kGet, historical_header);
    expect(historical_preparation.result.status == CUDA_SUCCESS &&
               !historical_preparation.result.original_status && !foundation::current(),
           "historical Plan HANDLE Get needs no revived native context or handle authority");
    const auto detached_historical = owner.registry->get_backend_attribute(
        historical_header, historical_preparation.generation, {});
    native(detached_historical.result);
    expect(
        detached_historical.values.handles.size() == 1 &&
            detached_historical.values.handles[0].owner == handle.handles[0].owner &&
            detached_historical.values.handles[0].id == handle.handles[0].id &&
            !foundation::current(),
        "detached historical Get returns only stored symbolic identity after native handle death");
    const auto second = owner.add_peer(0);
    const auto historical = owner.attribute(second, Operation::kGet, {plan, 400, 0, 1, 3});
    native(historical.result);
    expect(
        historical.values.handles.size() == 1 &&
            historical.values.handles[0].owner == handle.handles[0].owner &&
            historical.values.handles[0].id == handle.handles[0].id,
        "same-device frame returns historical symbolic handle without reviving native authority");
    native(owner.attribute(second, Operation::kGet, {plan, 402, 3, 1, 3}).result);
    const auto foreign = owner.add_peer(1);
    driver(owner.peers[foreign].session->activate());
    CudaDnnBackendContextRequest query;
    query.operation = Operation::kPrepare;
    query.attribute_operation = AttributeOperation::kGet;
    query.header = {plan, 402, 3, 1, 3};
    const auto calls = state.gets + state.versions;
    const auto rejected = owner.call(foreign, query);
    expect(rejected.result.status == CUDA_ERROR_INVALID_CONTEXT &&
               !rejected.result.original_status && state.gets + state.versions == calls,
           "foreign device cannot use a copied graph through coincident metadata IDs");
    owner.close();
}

void variant_slots_and_generations() {
    Owner owner;
    const auto peer = owner.add_peer(0);
    const auto variant = owner.create(16);
    Values uids;
    const std::array<std::int64_t, 3> ordered{17, 17, -9};
    uids.scalars = bytes(std::span<const std::int64_t>{ordered});
    native(owner.attribute(peer, Operation::kSet, {variant, 1000, 3, 3, 1}, uids).result);
    Values slots;
    slots.pointer_slots = {71, 0, 71};
    native(owner.attribute(peer, Operation::kSet, {variant, 1001, 6, 3, 1}, slots).result);
    slots.pointer_slots = {UINT64_MAX};
    const auto prepared = owner.prepare(peer, AttributeOperation::kSet, {variant, 1003, 6, 1, 1});
    native(owner.attribute(peer, Operation::kSet, {variant, 1003, 6, 1, 1}, slots).result);
    const auto& pointers = object(owner.native_id(variant)).value.attributes.at(1001).pointers;
    expect(pointers.size() == 3 && pointers[0] && !pointers[1] && pointers[0] == pointers[2],
           "symbolic repeated/null slots materialize stable inert native identities");
    const auto before = state.sets;
    CudaDnnBackendContextRequest stale;
    stale.operation = Operation::kSet;
    stale.header = {variant, 1003, 6, 1, 1};
    stale.generation = prepared.generation;
    stale.values = slots;
    const auto rejected = owner.call(peer, stale);
    expect(!rejected.result.original_status && state.sets == before,
           "stale generation cannot overwrite more recent native slot state");
    owner.finalize({.peer = peer, .descriptor = variant});
    const auto result = owner.attribute(peer, Operation::kGet, {variant, 1003, 6, 1, 3});
    native(result.result);
    expect(result.values.pointer_slots == slots.pointer_slots && result.values.scalars.empty() &&
               owner.snapshot(variant)->variant_uids == std::vector<std::int64_t>({17, 17, -9}),
           "workspace Get returns original symbolic identity, never a native/caller address");
    native(owner.registry->initialize_backend(variant));
    expect(
        owner.snapshot(variant)->pointer_slots.empty() && !owner.snapshot(variant)->workspace_slot,
        "original reset invalidates prior VariantPack slot and UID state");
    state.set_status = CUDNN_STATUS_BAD_PARAM;
    const auto failed = owner.attribute(peer, Operation::kSet, {variant, 1003, 6, 1, 1}, slots);
    native(failed.result, CUDNN_STATUS_BAD_PARAM);
    expect(
        !owner.snapshot(variant)->is_valid && !owner.registry->snapshot().is_quarantined,
        "partial native slot setter error dirties configuration without inventing ownership loss");
    state.set_status = CUDNN_STATUS_SUCCESS;
    native(owner.registry->initialize_backend(variant));
    state.finalize_status = CUDNN_STATUS_BAD_PARAM;
    CudaDnnBackendContextRequest finalize;
    finalize.operation = Operation::kFinalize;
    finalize.id = variant;
    native(owner.call(peer, finalize).result, CUDNN_STATUS_BAD_PARAM);
    expect(!owner.snapshot(variant)->is_valid && !owner.registry->snapshot().is_quarantined,
           "original Finalize error remains a recoverable configuration failure");
    state.finalize_status = CUDNN_STATUS_SUCCESS;
    native(owner.registry->initialize_backend(variant));
    owner.close();
}

void frame_and_reentry() {
    Owner owner;
    const auto peer = owner.add_peer(0);
    const auto graph = owner.create(15);
    const auto missing =
        owner.registry->prepare_backend_attribute(AttributeOperation::kGet, {graph, 802, 3, 1, 3});
    expect(missing.result.status == CUDA_ERROR_INVALID_CONTEXT && !missing.result.original_status,
           "DNM1-only metadata cannot authorize context-bound planning");
    Values handle;
    handle.handles = {{owner.peers[peer].owner, owner.peers[peer].handle + 1}};
    const auto wrong = owner.attribute(peer, Operation::kSet, {graph, 800, 0, 1, 1}, handle);
    expect(!wrong.result.original_status && wrong.result.status == CUDA_ERROR_INVALID_HANDLE,
           "typed historical or foreign handle IDs are not current handle authority");
    handle.handles[0].id = owner.peers[peer].handle;
    const auto ready = owner.prepare(peer, AttributeOperation::kSet, {graph, 800, 0, 1, 1});
    unsigned int callbacks{};
    state.callback = [&](unsigned int) {
        ++callbacks;
        auto& session = *owner.peers[peer].session;
        const auto switches = state.switches, streams = state.stream_gets;
        CudaDnnBackendContextRequest nested;
        nested.operation = Operation::kFinalize;
        nested.id = graph;
        expect(
            session.backend_context(owner.peers[peer].owner, owner.peers[peer].handle,
                                    *owner.registry, nested)
                        .result.status == CUDA_ERROR_NOT_READY &&
                session.activate().status == CUDA_ERROR_NOT_READY &&
                session.deactivate().status == CUDA_ERROR_NOT_READY &&
                session.get_dnn_stream(owner.peers[peer].handle).status == CUDA_ERROR_NOT_READY &&
                session.destroy_dnn(owner.peers[peer].handle).status == CUDA_ERROR_NOT_READY &&
                session.shutdown().status == CUDA_ERROR_NOT_READY && state.switches == switches &&
                state.stream_gets == streams,
            "live backend frame excludes native callback reentry into session ownership");
        expect(owner.registry->create_backend(15).status == CUDA_ERROR_NOT_READY &&
                   !owner.registry->borrow_backend(graph),
               "planning version/native callbacks also hold global registry exclusion");
    };
    CudaDnnBackendContextRequest set;
    set.operation = Operation::kSet;
    set.header = {graph, 800, 0, 1, 1};
    set.generation = ready.generation;
    set.values = handle;
    native(owner.call(peer, set).result);
    state.callback = {};
    expect(callbacks == 2, "version and original setter callback reentry both exercised");
    owner.finalize({.peer = peer, .descriptor = graph});
    const auto query = owner.prepare(peer, AttributeOperation::kGet, {graph, 802, 3, 1, 3});
    CudaDnnBackendContextRequest get;
    get.operation = Operation::kGet;
    get.header = {graph, 802, 3, 1, 3};
    get.generation = query.generation;
    state.lose_context = true;
    const auto lost = owner.call(peer, get);
    expect(
        lost.result.disposition == CudaDnnDisposition::kRejected &&
            lost.result.status == CUDA_ERROR_INVALID_CONTEXT &&
            lost.result.original_status == CUDNN_STATUS_SUCCESS &&
            lost.attribute.values.scalars.empty() && lost.attribute.has_context_loss,
        "post-native frame loss retains exact original status but publishes no successful value");
    state.lose_context = false;
    expect(owner.sdk.context_set_current(owner.peers[peer].context) == CUDA_SUCCESS,
           "restore known context only to clean proven host metadata ownership");
    owner.close();
}

void multiple_handle_materialization() {
    Owner owner;
    const auto peer = owner.add_peer(0);
    const auto additional = owner.peers[peer].session->create_dnn();
    native(additional);
    expect(additional.id != owner.peers[peer].handle,
           "two live handles have distinct owned identities in one session");
    const auto identity = owner.peers[peer].owner;
    Values handles;
    handles.handles = {{identity, additional.id}, {identity, owner.peers[peer].handle}};
    const std::vector<void*> expected{&base::state.handles[1], &base::state.handles[0]};
    for (const auto kind : {15U, 5U}) {
        const auto descriptor = owner.create(kind);
        const auto name = kind == 15 ? 800U : 400U;
        const auto before = state.sets;
        const auto rejected =
            owner.attribute(peer, Operation::kSet, {descriptor, name, 0, 2, 1}, handles);
        native(rejected.result, CUDNN_STATUS_BAD_PARAM);
        expect(
            state.sets == before + 1 && state.logged_handles == expected &&
                object(owner.native_id(descriptor)).value.attributes.empty() &&
                !owner.snapshot(descriptor)->is_valid && !owner.registry->snapshot().is_quarantined,
            "all same-session handles reach native logging in original order before count error");
        native(owner.registry->initialize_backend(descriptor));
        Values null_handle;
        null_handle.handles = {{0, 0}};
        const auto null_result =
            owner.attribute(peer, Operation::kSet, {descriptor, name, 0, 1, 1}, null_handle);
        native(null_result.result,
               kind == 15 ? CUDNN_STATUS_BAD_PARAM_NULL_POINTER : CUDNN_STATUS_SUCCESS);
        expect(
            state.logged_handles == std::vector<void*>{nullptr} &&
                !owner.registry->snapshot().is_quarantined,
            "Graph null-handle error and Plan null-handle success retain exact native semantics");
        if (kind == 15)
            native(owner.registry->initialize_backend(descriptor));
        else {
            owner.finalize({.peer = peer, .descriptor = descriptor});
            const auto result = owner.attribute(peer, Operation::kGet, {descriptor, 400, 0, 1, 3});
            native(result.result);
            expect(result.values.handles.size() == 1 && !result.values.handles[0].owner &&
                       !result.values.handles[0].id,
                   "original successful null Plan handle remains a null symbolic reference");
        }
    }
    owner.close();
}

std::shared_ptr<CudaDnnBackendSnapshot> observed_tensor(std::int64_t uid,
                                                        std::array<std::int64_t, 2> dimensions,
                                                        std::uint32_t type,
                                                        std::array<std::int64_t, 2> strides) {
    auto result = std::make_shared<CudaDnnBackendSnapshot>();
    result->kind = 17;
    result->is_valid = result->is_finalized = true;
    result->generation = 1;
    CudaDnnBackendTensorObservation tensor;
    tensor.uid = uid;
    tensor.type = type;
    tensor.alignment = {16, CudaDnnBackendTensorAlignmentSource::kNativeSet};
    tensor.dimension_count = tensor.stride_count = 2;
    tensor.vector_count = 1;
    tensor.vectorized_dimension = -1;
    std::ranges::copy(dimensions, tensor.dimensions.begin());
    std::ranges::copy(strides, tensor.strides.begin());
    result->tensor = tensor;
    return result;
}
CudaDnnBackendSnapshot graph_of(
    std::initializer_list<std::shared_ptr<const CudaDnnBackendSnapshot>> children) {
    CudaDnnBackendSnapshot graph;
    graph.kind = 15;
    graph.is_valid = graph.is_finalized = true;
    graph.generation = 1;
    CudaDnnBackendSnapshotAttribute attribute;
    attribute.name = 801;
    attribute.type = 15;
    attribute.descriptors = children;
    graph.attributes.push_back(std::move(attribute));
    return graph;
}
void independent_storage_requirements() {
    // These are trusted observation inputs, not invented native Finalize
    // acceptance or owned allocations. Exact extents have independent constants.
    const auto check = [](const CudaDnnBackendSnapshot& tensor, std::size_t expected) {
        const auto result = cuda_dnn_backend_tensor_requirements(tensor);
        expect(result.status == CUDA_SUCCESS && result.tensors.size() == 1 &&
                   result.tensors.front().bytes == expected,
               "independent last reachable element/packed bit rounding extent");
    };
    auto plain = observed_tensor(17, {2, 3}, 0, {8, 2});
    check(*plain, 52);  // Float indices 0..12, including stride holes.
    for (const std::int64_t count : {4, 32}) {
        auto vector = observed_tensor(18, {2, 3}, 3, {3, 1});
        vector->tensor->vector_count = count;
        vector->tensor->vectorized_dimension = 1;
        check(*vector, count == 4 ? 24 : 192);
    }
    for (const std::uint32_t type : {11U, 16U})
        expect(cuda_dnn_backend_tensor_requirements(*observed_tensor(19, {2, 5}, type, {5, 1}))
                       .status == CUDA_ERROR_NOT_SUPPORTED,
               "Boolean and FP4 byte addressing remains unqualified, not inferred from bit width");
    check(*observed_tensor(21, {0, 3}, 0, {9, 1}), 0);
    check(*observed_tensor(21, {INT64_MAX, 0}, 0, {INT64_MAX, INT64_MAX}), 0);
    auto host = observed_tensor(22, {1, 2}, 1, {5, 3});
    host->tensor->is_by_value = true;
    check(*host, 32);  // Four reachable double slots, not two dense values.
    expect(cuda_dnn_backend_tensor_requirements(*host).tensors.front().is_by_value,
           "by-value class is separate from the exact strided capture extent");
    for (const std::uint32_t type : {5U, 7U, 8U, 31U}) {
        const auto result =
            cuda_dnn_backend_tensor_requirements(*observed_tensor(23, {2, 3}, type, {3, 1}));
        expect(result.status == CUDA_ERROR_NOT_SUPPORTED,
               "legacy vector enums and unknown storage are not silently treated as scalar INT8");
    }
    auto swizzled = *plain;
    swizzled.tensor->reordering = 1;
    expect(cuda_dnn_backend_tensor_requirements(swizzled).status == CUDA_ERROR_NOT_SUPPORTED,
           "special reordering remains an explicit unqualified footprint");
    for (unsigned int mode = 0; mode != 5; ++mode) {
        auto invalid = *plain;
        if (mode == 0) invalid.tensor->dimensions[0] = -1;
        if (mode == 1) invalid.tensor->strides[0] = -1;
        if (mode == 2) invalid.tensor->alignment.bytes = 0;
        if (mode == 3) invalid.tensor->stride_count = 1;
        if (mode == 4) {
            invalid.tensor->dimensions[0] = INT64_MAX;
            invalid.tensor->strides[0] = INT64_MAX;
        }
        expect(cuda_dnn_backend_tensor_requirements(invalid).status == CUDA_ERROR_INVALID_VALUE,
               "invalid/overflowed observations confer no storage authority");
    }
    auto larger = observed_tensor(17, {3, 3}, 0, {8, 2});
    plain->tensor->alignment.bytes = 24;
    auto virtual_tensor = observed_tensor(99, {2, 3}, 0, {3, 1});
    virtual_tensor->tensor->is_virtual = true;
    const auto shared_branch = std::make_shared<const CudaDnnBackendSnapshot>(graph_of({plain}));
    const auto dag = graph_of({shared_branch, shared_branch, larger, virtual_tensor});
    const auto unioned = cuda_dnn_backend_tensor_requirements(dag);
    expect(unioned.status == CUDA_SUCCESS && unioned.tensors.size() == 1 &&
               unioned.tensors[0].uid == 17 && unioned.tensors[0].bytes == 84 &&
               unioned.tensors[0].alignment == 48,
           "shared copied DAG deduplicates traversal, unions UID max extent and alignment LCM");
    larger->tensor->is_by_value = true;
    expect(cuda_dnn_backend_tensor_requirements(graph_of({plain, larger})).status ==
               CUDA_ERROR_INVALID_VALUE,
           "one UID cannot simultaneously authorize host and device storage");
    larger->tensor->is_by_value = false;
    plain->tensor->alignment.bytes = INT64_MAX;
    larger->tensor->alignment.bytes = INT64_MAX - 1;
    expect(cuda_dnn_backend_tensor_requirements(graph_of({plain, larger})).status ==
               CUDA_ERROR_INVALID_VALUE,
           "alignment LCM overflow is rejected rather than wrapped into weaker alignment");
    plain->tensor.reset();
    expect(cuda_dnn_backend_tensor_requirements(*plain).status == CUDA_ERROR_NOT_SUPPORTED,
           "caller setter snapshots cannot substitute for missing native Tensor observations");
}
}  // namespace

int main() {
    try {
        copied_planning_and_context_lifetimes();
        variant_slots_and_generations();
        frame_and_reentry();
        multiple_handle_materialization();
        independent_storage_requirements();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "backend graph planning: " << error.what() << '\n';
        return 1;
    }
}
