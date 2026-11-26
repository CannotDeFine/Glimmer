#include "dnn_backend_heuristics_fixture.h"
#include "internal/cuda_dnn_backend_execution.h"

#include <iostream>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_backend_heuristics;
namespace graph = fixture::graph;
namespace driver = fixture::driver;
using Native = fixture::Native;
using Header = CudaDnnBackendAttributeHeader;
using Values = CudaDnnBackendAttributeValues;
using Operation = CudaDnnBackendContextOperation;
using AttributeOperation = CudaDnnBackendAttributeOperation;
void expect(bool condition, const char* message) {
    fixture::require(condition, message);
}
void native(const CudaDnnResult& result, cudnnStatus_t status = CUDNN_STATUS_SUCCESS) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == status,
           "exact original requested metadata result");
}
void checked(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned fake Driver operation");
}
struct Owner {
    static constexpr std::uint64_t k_owner = 921;
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> registry;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t handle{};
    Owner() {
        fixture::reset();
        sdk = fixture::session_dispatch();
        registry = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        session = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{917, k_owner},
                                                         k_owner, sdk);
        checked(session->initialize(0));
        const auto result = session->create_dnn();
        native(result);
        handle = result.id;
    }
    std::uint64_t create(std::uint32_t kind) {
        const auto result = registry->create_backend(kind);
        native(result);
        return result.id;
    }
    CudaDnnBackendContextResult call(const CudaDnnBackendContextRequest& request) {
        return session->backend_context(k_owner, handle, *registry, request);
    }
    CudaDnnBackendAttributePreparation prepare(AttributeOperation operation, const Header& header) {
        CudaDnnBackendContextRequest request;
        request.operation = Operation::kPrepare;
        request.attribute_operation = operation;
        request.header = header;
        const auto result = call(request);
        expect(result.result.status == CUDA_SUCCESS && !result.result.original_status &&
                   result.preparation.generation,
               "private preparation has generation but no requested native provenance");
        return result.preparation;
    }
    CudaDnnBackendAttributeResult attribute(Operation operation, const Header& header,
                                            const Values& values = {}) {
        const auto ready = prepare(
            operation == Operation::kSet ? AttributeOperation::kSet : AttributeOperation::kGet,
            header);
        CudaDnnBackendContextRequest request;
        request.operation = operation;
        request.header = header;
        request.generation = ready.generation;
        request.values = values;
        return call(request).attribute;
    }
    void finalize(std::uint64_t id) {
        CudaDnnBackendContextRequest request;
        request.operation = Operation::kFinalize;
        request.id = id;
        native(call(request).result);
    }
    template <typename T>
    void scalar(std::uint64_t id, std::uint32_t name, std::uint32_t type, T value) {
        Values values;
        values.scalars = graph::scalar(value);
        native(attribute(Operation::kSet, {id, name, type, 1, 1}, values).result);
    }
    template <typename T, std::size_t N>
    void array(std::uint64_t id, std::uint32_t name, std::uint32_t type,
               const std::array<T, N>& value) {
        Values values;
        values.scalars = graph::bytes(std::span<const T>{value});
        native(attribute(Operation::kSet, {id, name, type, N, 1}, values).result);
    }
    void child(std::uint64_t id, std::uint32_t name,
               std::initializer_list<std::uint64_t> children) {
        Values values;
        values.descriptors = children;
        native(attribute(Operation::kSet,
                         {id, name, 15, static_cast<std::int64_t>(children.size()), 1}, values)
                   .result);
    }
    CudaDnnBackendAttributeResult get(const Header& header,
                                      std::initializer_list<std::uint64_t> destinations = {}) {
        Values values;
        values.descriptors = destinations;
        return attribute(Operation::kGet, header, values);
    }
    std::shared_ptr<const CudaDnnBackendSnapshot> snapshot(std::uint64_t id) {
        auto value = registry->borrow_backend(id);
        expect(bool(value), "observable owned metadata snapshot");
        return value->snapshot;
    }
    Native original(std::uint64_t id) {
        auto value = registry->borrow_backend(id);
        expect(bool(value), "owned original descriptor identity");
        return value->object.native;
    }
    std::uint64_t tensor(std::int64_t uid, bool by_value = false) {
        const auto id = create(17);
        scalar(id, 900, 3, std::int64_t{4});
        scalar(id, 901, 1, std::uint32_t{0});
        array(id, 902, 3, std::array<std::int64_t, 4>{1, 1, 1, by_value ? 1 : 4});
        array(id, 903, 3, std::array<std::int64_t, 4>{4, 4, 4, 1});
        scalar(id, 906, 3, uid);
        if (by_value) scalar(id, 908, 2, true);
        finalize(id);
        expect(snapshot(id)->tensor.has_value(), "independent original Tensor observations");
        return id;
    }
    struct Chain {
        std::array<std::uint64_t, 3> tensors{};
        std::uint64_t pointwise{}, operation{}, graph{}, heuristic{};
    };
    Chain chain() {
        Chain result;
        result.tensors = {tensor(11), tensor(12), tensor(13, true)};
        result.pointwise = create(0);
        scalar(result.pointwise, 0, 14, std::uint32_t{1});
        scalar(result.pointwise, 1, 1, std::uint32_t{0});
        finalize(result.pointwise);
        result.operation = create(13);
        child(result.operation, 750, {result.pointwise});
        child(result.operation, 751, {result.tensors[0]});
        child(result.operation, 752, {result.tensors[2]});
        child(result.operation, 753, {result.tensors[1]});
        finalize(result.operation);
        result.graph = create(15);
        Values handle_value;
        handle_value.handles.push_back({k_owner, handle});
        native(attribute(Operation::kSet, {result.graph, 800, 0, 1, 1}, handle_value).result);
        child(result.graph, 801, {result.operation});
        finalize(result.graph);
        result.heuristic = create(4);
        scalar(result.heuristic, 200, 8, std::uint32_t{0});
        child(result.heuristic, 201, {result.graph});
        finalize(result.heuristic);
        return result;
    }
    std::uint64_t generated(const Chain& chain) {
        const auto config = create(3);
        const auto result = get({chain.heuristic, 202, 15, 1, 3}, {config});
        native(result.result);
        expect(result.count == 1 && snapshot(config)->is_finalized &&
                   snapshot(config)->generated_graph && snapshot(config)->attributes.empty(),
               "actual generated config has immutable graph lineage");
        return config;
    }
    void erase_sources(const Chain& chain) {
        for (const auto id : {chain.heuristic, chain.graph, chain.operation, chain.pointwise,
                              chain.tensors[0], chain.tensors[1], chain.tensors[2]}) {
            native(registry->initialize_backend(id));
            native(registry->destroy(id));
        }
    }
    std::uint64_t allocation(std::span<const std::byte> contents) {
        const auto result = session->allocate(contents.size());
        checked(result);
        checked(session->upload({result.id, 0}, contents));
        return result.id;
    }
    template <std::size_t Count>
    std::vector<std::byte> download(std::uint64_t id) {
        std::vector<std::byte> output(Count);
        checked(session->download({id, 0}, output));
        return output;
    }
    void close() {
        checked(session->shutdown());
        session.reset();
        expect(!driver::current(), "context-first native planning teardown");
        registry->close();
        expect(registry->shutdown().status == CUDA_SUCCESS && registry->snapshot().is_closed &&
                   graph::state.creates == graph::state.destroys &&
                   fixture::state.private_packs.empty(),
               "all original and private native descriptors retire exactly once");
    }
};
template <typename T>
T scalar_get(Owner& owner, std::uint64_t id, std::uint32_t name, std::uint32_t type) {
    const auto result = owner.get({id, name, type, 1, 3});
    native(result.result);
    expect(result.count == 1 && result.values.scalars.size() == sizeof(T),
           "complete scalar getter publication");
    T value{};
    std::memcpy(&value, result.values.scalars.data(), sizeof(value));
    return value;
}
void heuristic_counts_and_tails() {
    Owner owner;
    const auto chain = owner.chain();
    expect(owner.snapshot(chain.heuristic)->generated_array_capacity == 4,
           "private finalized heuristic observation records raw upper bound");
    for (const auto requested : {INT64_MIN, INT64_C(-1), INT64_C(0), INT64_C(1), INT64_MAX}) {
        const auto result = owner.get({chain.heuristic, 202, 15, requested, 2});
        native(result.result);
        expect(result.count == 4 && result.count_bytes == 8 && result.values.descriptors.empty(),
               "count-only raw heuristic N is independent of signed requested count");
    }
    for (const auto requested : {0, 1, 2, 4, 6}) {
        std::array<std::uint64_t, 4> destinations{};
        std::array<std::uint64_t, 4> generations{};
        for (std::size_t i = 0; i != destinations.size(); ++i) {
            destinations[i] = owner.create(3);
            generations[i] = owner.snapshot(destinations[i])->generation;
        }
        const Header header{chain.heuristic, 202, 15, requested, 3};
        const auto ready = owner.prepare(AttributeOperation::kGet, header);
        const auto captured = static_cast<std::size_t>(std::min(requested, 4));
        expect(ready.input_elements == captured,
               "preparation captures bounded potentially accessed N prefix, not emitted M");
        Values values;
        values.descriptors.assign(destinations.begin(), destinations.begin() + captured);
        const auto result = owner.attribute(Operation::kGet, header, values);
        native(result.result);
        const auto emitted = static_cast<std::size_t>(std::min(requested, 2));
        expect(result.count == static_cast<std::int64_t>(emitted) && result.count_bytes == 8,
               "requested getter publishes actual successfully emitted M, not raw N");
        for (std::size_t i = 0; i != destinations.size(); ++i) {
            const auto snapshot = owner.snapshot(destinations[i]);
            expect((snapshot->generation > generations[i]) == (i < emitted),
                   "only actually written destination generations change");
            expect(snapshot->is_finalized == (i < emitted),
                   "untouched tail is never fabricated as a finalized native result");
        }
        for (const auto id : destinations) native(owner.registry->destroy(id));
    }
    const auto first = owner.create(3), second = owner.create(3);
    const auto tail = owner.get({chain.heuristic, 202, 15, 4, 3}, {first, second, first, first});
    native(tail.result);
    expect(tail.count == 2 && owner.snapshot(first)->is_valid && owner.snapshot(second)->is_valid,
           "aliased unwritten heuristic tail must not overwrite or invalidate written prefix");
    expect(owner.get({chain.heuristic, 202, 15, -1, 3}).result.original_status ==
               CUDNN_STATUS_BAD_PARAM,
           "negative array request retains original early status without reading destinations");
    owner.close();
}

void partial_and_alias_failures() {
    for (const bool duplicate : {false, true}) {
        Owner owner;
        const auto chain = owner.chain();
        const auto first = owner.create(3), second = owner.create(3);
        const auto source_generation = owner.snapshot(chain.heuristic)->generation;
        const auto first_generation = owner.snapshot(first)->generation;
        if (!duplicate) fixture::state.fail_after = 0;
        const auto failed =
            owner.get({chain.heuristic, 202, 15, 2, 3}, {first, duplicate ? first : second});
        native(failed.result, CUDNN_STATUS_BAD_PARAM);
        expect(!failed.count_bytes && failed.values.descriptors.empty() &&
                   owner.snapshot(first)->generation > first_generation &&
                   !owner.snapshot(first)->is_valid &&
                   owner.snapshot(chain.heuristic)->generation == source_generation &&
                   owner.snapshot(chain.heuristic)->is_valid &&
                   !owner.registry->snapshot().is_quarantined,
               "partial original error invalidates destination proof without count publication "
               "or ownership quarantine");
        expect(graph::object(owner.original(first)).snapshot.is_finalized,
               "native first copy happened before injected failure or finalized-alias rejection");
        if (!duplicate)
            expect(!owner.snapshot(second)->is_valid,
                   "uncertain partially written upper prefix cannot retain stale authority");
        fixture::state.fail_after = -1;
        native(owner.registry->initialize_backend(first));
        native(owner.registry->initialize_backend(second));
        native(owner.get({chain.heuristic, 202, 15, 2, 3}, {first, second}).result);
        expect(owner.snapshot(first)->is_valid && owner.snapshot(second)->is_valid,
               "explicit original Initialize restores destination configuration usability");
        owner.close();
    }
}

void normalized_knobs_and_aliases() {
    Owner owner;
    const auto chain = owner.chain();
    const auto config = owner.generated(chain);
    const auto engine = owner.create(2);
    native(owner.get({config, 300, 15, 1, 3}, {engine}).result);
    expect(owner.snapshot(engine)->generated_graph &&
               owner.snapshot(engine)->generated_array_capacity == 3 &&
               scalar_get<std::int64_t>(owner, engine, 1301, 3) == 100,
           "generated Engine observation retains graph and bounded native knob count");
    const auto info_a = owner.create(8), info_b = owner.create(8);
    const auto ready = owner.prepare(AttributeOperation::kGet, {engine, 1302, 15, 0, 3});
    expect(ready.input_elements == 3,
           "Engine request zero still captures all three native destination writes");
    const auto infos = owner.get({engine, 1302, 15, 0, 3}, {info_a, info_b, info_a});
    native(infos.result);
    expect(infos.count == 3 && scalar_get<std::uint32_t>(owner, info_a, 1200, 9) == 2 &&
               scalar_get<std::uint32_t>(owner, info_b, 1200, 9) == 1 &&
               scalar_get<std::int64_t>(owner, info_a, 1201, 3) == 18 &&
               scalar_get<std::int64_t>(owner, info_a, 1202, 3) == 3 &&
               scalar_get<std::int64_t>(owner, info_a, 1203, 3) == 2,
           "sequential Engine destination aliases retain the last actually written KnobInfo");
    const auto choice_a = owner.create(7), choice_b = owner.create(7), tail = owner.create(7);
    const auto tail_generation = owner.snapshot(tail)->generation;
    const auto choices =
        owner.get({config, 302, 15, 5, 3}, {choice_a, choice_b, choice_a, tail, choice_b});
    native(choices.result);
    expect(choices.count == 3 && scalar_get<std::uint32_t>(owner, choice_a, 600, 9) == 2 &&
               scalar_get<std::int64_t>(owner, choice_a, 601, 3) == 33 &&
               scalar_get<std::uint32_t>(owner, choice_b, 600, 9) == 1 &&
               owner.snapshot(tail)->generation == tail_generation,
           "Config validates all requested positions but commits only normalized written prefix; "
           "unwritten tail aliases never win");
    const auto zero = owner.get({config, 302, 15, 0, 2});
    native(zero.result);
    expect(zero.count == 0, "Config zero count-only is not a total-count query");
    const auto total = owner.get({config, 302, 15, 43, 2});
    native(total.result);
    expect(total.count == 3, "Config bounded capacity query sees normalized native count");
    const auto fresh_config = owner.create(3), low = owner.create(7), high = owner.create(7);
    owner.scalar(low, 600, 9, std::uint32_t{1});
    owner.scalar(low, 601, 3, std::int64_t{10});
    owner.finalize(low);
    owner.scalar(high, 600, 9, std::uint32_t{2});
    owner.scalar(high, 601, 3, std::int64_t{20});
    owner.finalize(high);
    owner.child(fresh_config, 300, {engine});
    owner.child(fresh_config, 302, {high, low});
    owner.finalize(fresh_config);
    native(owner.registry->initialize_backend(choice_a));
    native(owner.registry->initialize_backend(choice_b));
    native(owner.get({fresh_config, 302, 15, 2, 3}, {choice_a, choice_b}).result);
    expect(scalar_get<std::uint32_t>(owner, choice_a, 600, 9) == 1 &&
               scalar_get<std::int64_t>(owner, choice_a, 601, 3) == 10 &&
               scalar_get<std::uint32_t>(owner, choice_b, 600, 9) == 2,
           "queried normalized choices follow native keyed order, not public Set vector order");
    owner.close();
}

void generated_execution() {
    Owner owner;
    const auto chain = owner.chain();
    const auto config = owner.generated(chain);
    owner.erase_sources(chain);
    const auto plan = owner.create(5);
    owner.child(plan, 401, {config});
    owner.finalize(plan);
    native(owner.registry->initialize_backend(config));
    native(owner.registry->destroy(config));
    const auto copied_config = owner.create(3);
    native(owner.get({plan, 401, 15, 1, 3}, {copied_config}).result);
    const auto requirements = cuda_dnn_backend_tensor_requirements(*owner.snapshot(copied_config));
    expect(requirements.status == CUDA_SUCCESS && requirements.tensors.size() == 3,
           "generated Config copied out of Plan retains requirements after every source dies");
    for (const auto& requirement : requirements.tensors)
        expect(requirement.uid >= 11 && requirement.uid <= 13 && requirement.alignment == 4 &&
                   requirement.bytes == (requirement.uid == 13 ? 4U : 16U) &&
                   requirement.is_by_value == (requirement.uid == 13),
               "generated lineage supplies independently known full tensor footprints");
    const auto variant = owner.create(16);
    owner.array(variant, 1000, 3, std::array<std::int64_t, 3>{11, 12, 13});
    Values slots;
    slots.pointer_slots = {101, 102, 103};
    native(owner.attribute(Operation::kSet, {variant, 1001, 6, 3, 1}, slots).result);
    slots.pointer_slots = {104};
    native(owner.attribute(Operation::kSet, {variant, 1003, 6, 1, 1}, slots).result);
    owner.finalize(variant);
    const auto ready = owner.session->prepare_dnn_backend_execute(Owner::k_owner, owner.handle,
                                                                  *owner.registry, plan, variant);
    expect(ready.result.status == CUDA_SUCCESS && !ready.result.original_status &&
               ready.requirements.size() == 4,
           "generated plan reaches real session execution preparation without Execute provenance");
    const std::array<float, 6> input{91, 1, -2, 3, -4, 92}, output{71, 0, 0, 0, 0, 72};
    const auto input_id = owner.allocation(graph::bytes(std::span<const float>{input}));
    const auto output_id = owner.allocation(graph::bytes(std::span<const float>{output}));
    const std::vector<std::byte> work(40, std::byte{0xcd});
    const auto workspace_id = owner.allocation(work);
    CudaSessionDnnBackendExecute problem{
        owner.handle, plan, variant, ready.plan_generation, ready.variant_generation, {}};
    for (const auto& requirement : ready.requirements) {
        CudaDnnBackendExecuteBinding binding;
        binding.index = requirement.index;
        binding.value_class = requirement.value_class;
        binding.slot = requirement.slot;
        if (requirement.index == 0) binding.device = {input_id, 4};
        if (requirement.index == 1) binding.device = {output_id, 4};
        if (requirement.index == 2) binding.host_bytes = graph::scalar(2.5F);
        if (requirement.index == UINT32_MAX) {
            expect(requirement.bytes == 32, "original plan workspace determines required interval");
            binding.device = {workspace_id, 4};
        }
        problem.bindings.push_back(std::move(binding));
    }
    auto short_range = problem;
    short_range.bindings[1].device.offset = 12;
    const auto rejected =
        owner.session->accept_dnn_backend_execute(1, *owner.registry, Owner::k_owner, short_range);
    expect(!rejected.native.acceptance.accepted() && !rejected.native.original_status &&
               !fixture::state.executes && !fixture::state.private_creates,
           "generated lineage rejects insufficient owned output range before original Execute");
    driver::control.can_stop.store(false);
    auto accepted =
        owner.session->accept_dnn_backend_execute(1, *owner.registry, Owner::k_owner, problem);
    expect(accepted.native.acceptance.accepted() && !fixture::state.private_creates,
           "actual session admission creates no private native pack");
    problem.bindings[2].host_bytes = graph::scalar(99.0F);
    fixture::state.issuing = true;
    expect(owner.session->issue_next().library_status == CUDNN_STATUS_SUCCESS &&
               fixture::state.executes == 1 && fixture::state.private_creates == 1 &&
               !fixture::state.private_destroys,
           "original Execute receives generated plan only after private preparation");
    expect(owner.session->acknowledge_result(accepted.native.acceptance.receipt) &&
               owner.session->poll().retained == 1 && !fixture::state.private_destroys &&
               owner.registry->destroy(plan).disposition == CudaDnnDisposition::kDeferred &&
               owner.session->release(input_id).disposition == CudaSessionDisposition::kDeferred,
           "host ACK retains private pack, generated plan and actual device memory until event");
    const auto pending = owner.session->prepare_dnn_backend_execute(Owner::k_owner, owner.handle,
                                                                    *owner.registry, plan, variant);
    expect(pending.result.status == CUDA_SUCCESS && !fixture::state.private_destroys,
           "read-only generated requirements do not drain unrelated pending physical work");
    driver::control.can_stop.store(true);
    expect(owner.session->poll().retired == 1 && fixture::state.private_destroys == 1,
           "physical proof destroys exactly one command-private pack");
    const std::array<float, 6> expected{71, 2.5F, -5, 7.5F, -10, 72};
    expect(
        owner.download<sizeof(output)>(output_id) ==
                graph::bytes(std::span<const float>{expected}) &&
            owner.download<sizeof(input)>(input_id) == graph::bytes(std::span<const float>{input}),
        "all numerical outputs, input bytes and guards match the independent CPU operation");
    auto expected_work = work;
    expected_work[4] = std::byte{0x5a};
    expect(owner.download<40>(workspace_id) == expected_work,
           "full workspace and guards remain within the exact owned interval");
    const auto synchronizes = driver::control.context_synchronizes.load();
    fixture::state.execute_status = CUDNN_STATUS_NOT_SUPPORTED;
    const auto native_error =
        owner.session->accept_dnn_backend_execute(2, *owner.registry, Owner::k_owner, problem);
    expect(native_error.native.acceptance.accepted() &&
               owner.session->issue_next().library_status == CUDNN_STATUS_NOT_SUPPORTED &&
               driver::control.context_synchronizes.load() == synchronizes + 1 &&
               native_error.result->result.original_status == CUDNN_STATUS_NOT_SUPPORTED,
           "actual original error preserves provenance and obtains hidden-stream stop proof");
    expect(owner.session->acknowledge_result(native_error.native.acceptance.receipt) &&
               owner.session->poll().retired == 1 && fixture::state.private_destroys == 2,
           "original-error private pack also cleans exactly once");
    owner.close();
}

void reentry_and_stale_generation() {
    Owner owner;
    const auto chain = owner.chain();
    const auto destination = owner.create(3);
    bool called{};
    fixture::state.callback = [&] {
        if (called) return;
        called = true;
        expect(owner.registry->destroy(chain.heuristic).status == CUDA_ERROR_NOT_READY &&
                   owner.registry->shutdown().status == CUDA_ERROR_NOT_READY &&
                   owner.session->shutdown().status == CUDA_ERROR_NOT_READY,
               "generated native getter cannot reenter metadata or session destruction");
    };
    native(owner.get({chain.heuristic, 202, 15, 1, 3}, {destination}).result);
    expect(called, "original generated getter callback actually ran");
    fixture::state.callback = {};
    const Header header{chain.heuristic, 202, 15, 1, 3};
    const auto before = owner.prepare(AttributeOperation::kGet, header);
    native(owner.registry->initialize_backend(chain.heuristic));
    CudaDnnBackendContextRequest stale;
    stale.operation = Operation::kGet;
    stale.header = header;
    stale.generation = before.generation;
    stale.values.descriptors = {destination};
    const auto gets = fixture::state.planning_gets;
    const auto result = owner.call(stale).attribute;
    expect(result.result.disposition == CudaDnnDisposition::kRejected &&
               !result.result.original_status && fixture::state.planning_gets == gets,
           "stale prepared generation cannot invoke the original getter");
    owner.close();
}

void successful_rebind_replaces_lineage() {
    Owner owner;
    const auto make_graph = [&](std::int64_t uid, const std::array<std::int64_t, 4>& dimensions) {
        const auto tensor = owner.create(17);
        owner.scalar(tensor, 900, 3, std::int64_t{4});
        owner.scalar(tensor, 901, 1, std::uint32_t{0});
        owner.scalar(tensor, 906, 3, uid);
        owner.array(tensor, 902, 3, dimensions);
        owner.array(tensor, 903, 3, std::array<std::int64_t, 4>{7, 7, 7, 1});
        owner.finalize(tensor);
        const auto operation = owner.create(13);
        owner.child(operation, 751, {tensor});
        owner.finalize(operation);
        const auto graph_id = owner.create(15);
        Values handle;
        handle.handles.push_back({Owner::k_owner, owner.handle});
        native(owner.attribute(Operation::kSet, {graph_id, 800, 0, 1, 1}, handle).result);
        owner.child(graph_id, 801, {operation});
        owner.finalize(graph_id);
        native(owner.registry->destroy(tensor));
        native(owner.registry->destroy(operation));
        return graph_id;
    };
    const auto graph_a = make_graph(41, {1, 1, 1, 3});
    const auto graph_b = make_graph(42, {1, 1, 1, 7});
    const auto check_plan = [&](std::uint64_t config) {
        const auto plan = owner.create(5);
        owner.child(plan, 401, {config});
        owner.finalize(plan);
        const auto required = cuda_dnn_backend_tensor_requirements(*owner.snapshot(plan));
        expect(required.status == CUDA_SUCCESS && required.tensors.size() == 1 &&
                   required.tensors[0].uid == 42 && required.tensors[0].bytes == 28,
               "successful A-to-B rebind grants only B's independently different UID and extent");
    };
    const auto heuristic = owner.create(4);
    owner.scalar(heuristic, 200, 8, std::uint32_t{0});
    owner.child(heuristic, 201, {graph_a});
    owner.child(heuristic, 201, {graph_b});
    owner.finalize(heuristic);
    const auto generated = owner.create(3);
    native(owner.get({heuristic, 202, 15, 1, 3}, {generated}).result);
    check_plan(generated);

    const auto engine_a = owner.create(2), engine_b = owner.create(2);
    owner.child(engine_a, 1300, {graph_a});
    owner.scalar(engine_a, 1301, 3, std::int64_t{0});
    owner.finalize(engine_a);
    owner.child(engine_b, 1300, {graph_a});
    owner.scalar(engine_b, 1301, 3, std::int64_t{1});
    owner.child(engine_b, 1300, {graph_b});
    owner.finalize(engine_b);
    const auto config = owner.create(3);
    owner.child(config, 300, {engine_a});
    owner.child(config, 300, {engine_b});
    owner.finalize(config);
    check_plan(config);
    owner.close();
}
}  // namespace
int main() {
    try {
        heuristic_counts_and_tails();
        partial_and_alias_failures();
        normalized_knobs_and_aliases();
        generated_execution();
        reentry_and_stale_generation();
        successful_rebind_replaces_lineage();
        std::cout << "Independent generated planning and execution regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
