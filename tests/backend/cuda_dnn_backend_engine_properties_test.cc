#include "dnn_backend_engine_properties_fixture.h"
#include "internal/cuda_dnn_backend_execution.h"

#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_backend_engine_properties;
namespace driver = fixture::driver;
using Header = CudaDnnBackendAttributeHeader;
using Values = CudaDnnBackendAttributeValues;
using Operation = CudaDnnBackendContextOperation;
using AttributeOperation = CudaDnnBackendAttributeOperation;

void expect(bool condition, const char* message) {
    fixture::require(condition, message);
}
void native(const CudaDnnResult& result, cudnnStatus_t expected = CUDNN_STATUS_SUCCESS) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == expected,
           "exact requested native status, distinct from private preparation");
}
void policy(const CudaDnnResult& result, CUresult expected) {
    expect(result.disposition == CudaDnnDisposition::kRejected && result.status == expected &&
               !result.original_status,
           "clean preissue policy result has no invented native provenance");
}
void checked(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "successful owned fake Driver lifecycle");
}
struct Owner {
    static constexpr std::uint64_t k_owner = 963;
    CudaSessionDispatch sdk;
    std::unique_ptr<CudaDnnRegistry> registry;
    std::unique_ptr<CudaExecutionSession> session;
    std::uint64_t handle{};
    explicit Owner(bool has_getter = true) {
        fixture::reset();
        sdk = fixture::session_dispatch();
        if (!has_getter) {
            auto dnn = *sdk.dnn;
            dnn.backend_get = nullptr;
            sdk.dnn = std::make_shared<const CudaDnnDispatch>(dnn);
        }
        registry = std::make_unique<CudaDnnRegistry>(*sdk.dnn);
        session = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{961, k_owner},
                                                         k_owner, sdk);
        checked(session->initialize(0));
        const auto created = session->create_dnn();
        native(created);
        handle = created.id;
    }
    std::uint64_t create(std::uint32_t kind) {
        const auto result = registry->create_backend(kind);
        native(result);
        return result.id;
    }
    CudaDnnBackendContextResult framed(const CudaDnnBackendContextRequest& request) {
        expect(bool(session), "frame requires a live session");
        return session->backend_context(k_owner, handle, *registry, request);
    }
    CudaDnnResult finalize(std::uint64_t id, bool has_frame = false) {
        if (!has_frame) return registry->finalize_backend(id);
        CudaDnnBackendContextRequest request;
        request.operation = Operation::kFinalize;
        request.id = id;
        return framed(request).result;
    }
    CudaDnnBackendAttributePreparation prepare(AttributeOperation operation, const Header& header,
                                               bool has_frame = false) {
        if (!has_frame) return registry->prepare_backend_attribute(operation, header);
        CudaDnnBackendContextRequest request;
        request.operation = Operation::kPrepare;
        request.attribute_operation = operation;
        request.header = header;
        return framed(request).preparation;
    }
    CudaDnnBackendAttributeResult complete(Operation operation, const Header& header,
                                           std::uint64_t generation, const Values& values = {},
                                           bool has_frame = false) {
        if (!has_frame)
            return operation == Operation::kGet
                       ? registry->get_backend_attribute(header, generation, values)
                       : registry->set_backend_attribute(header, generation, values);
        CudaDnnBackendContextRequest request;
        request.operation = operation;
        request.header = header;
        request.generation = generation;
        request.values = values;
        return framed(request).attribute;
    }
    CudaDnnBackendAttributeResult attribute(Operation operation, const Header& header,
                                            const Values& values = {}, bool has_frame = false) {
        const auto ready = prepare(
            operation == Operation::kGet ? AttributeOperation::kGet : AttributeOperation::kSet,
            header, has_frame);
        if (ready.result.status != CUDA_SUCCESS) {
            CudaDnnBackendAttributeResult result;
            result.result = ready.result;
            return result;
        }
        expect(ready.generation && !ready.result.original_status,
               "private preparation provides a generation, not requested SDK status");
        return complete(operation, header, ready.generation, values, has_frame);
    }
    CudaDnnBackendAttributeResult get(const Header& header,
                                      std::initializer_list<std::uint64_t> destinations = {}) {
        Values values;
        values.descriptors = destinations;
        return attribute(Operation::kGet, header, values);
    }
    template <typename T>
    CudaDnnBackendAttributeResult set_scalar(std::uint64_t id, std::uint32_t name,
                                             std::uint32_t type, T value) {
        Values values;
        values.scalars = fixture::scalar(value);
        return attribute(Operation::kSet, {id, name, type, 1, 1}, values);
    }
    void json(std::uint64_t id, const std::string& value) {
        Values values;
        values.scalars = fixture::bytes(std::span<const char>{value.data(), value.size()});
        native(attribute(Operation::kSet,
                         {id, 2702, 24, static_cast<std::int64_t>(value.size()), 1}, values)
                   .result);
    }
    void handle_copy(std::uint64_t id) {
        Values values;
        values.handles.push_back({k_owner, handle});
        native(attribute(Operation::kSet, {id, 2701, 0, 1, 1}, values, true).result);
    }
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): positional descriptor attribute helper.
    void copy_in(std::uint64_t parent, std::uint32_t name, std::uint64_t child) {
        Values values;
        values.descriptors = {child};
        native(attribute(Operation::kSet, {parent, name, 15, 1, 1}, values, true).result);
    }
    std::shared_ptr<const CudaDnnBackendSnapshot> snapshot(std::uint64_t id) {
        auto borrowed = registry->borrow_backend(id);
        expect(bool(borrowed), "owned snapshot is observable without retaining a native borrow");
        return borrowed->snapshot;
    }
    fixture::Native original(std::uint64_t id) {
        auto borrowed = registry->borrow_backend(id);
        expect(bool(borrowed), "owned native identity");
        return borrowed->object.native;
    }
    void end_context() {
        checked(session->shutdown());
        session.reset();
        expect(!driver::current() && driver::control.context_destroys.load() == 1,
               "source handle and context actually die before metadata-only work");
    }
    void close() {
        if (session) end_context();
        registry->close();
        expect(registry->shutdown().status == CUDA_SUCCESS && registry->snapshot().is_closed &&
                   fixture::state.creates == fixture::state.destroys,
               "context-free metadata cleanup destroys every native object exactly once");
    }
};
template <typename T>
T scalar_get(Owner& owner, std::uint64_t id, std::uint32_t name, std::uint32_t type) {
    const auto result = owner.get({id, name, type, 1, 3});
    native(result.result);
    expect(
        result.count == 1 && result.count_bytes == 8 && result.values.scalars.size() == sizeof(T),
        "exact scalar and count publication widths");
    T value{};
    std::memcpy(&value, result.values.scalars.data(), sizeof(value));
    return value;
}

void note_counts_and_exact_prefixes() {
    Owner owner;
    const auto engine = owner.create(2);
    native(owner.finalize(engine, true));
    struct Row {
        std::uint32_t name, type;
        std::vector<std::uint32_t> values;
    };
    for (const auto& row : {Row{1303, 11, {1, 4, 9}}, Row{1305, 19, {0, 3}}}) {
        for (const auto requested :
             {INT64_MIN, INT64_C(-1), INT64_C(0), INT64_C(1), INT64_C(2), INT64_C(20), INT64_MAX}) {
            const auto expected = requested < 0 ? 0
                                  : requested == 0
                                      ? row.values.size()
                                      : std::min<std::uint64_t>(requested, row.values.size());
            const auto count = owner.get({engine, row.name, row.type, requested, 2});
            native(count.result);
            expect(count.count == static_cast<std::int64_t>(expected) && count.count_bytes == 8 &&
                       count.values.scalars.empty(),
                   "count-only signed request uses independent note cardinality");
            if (requested < 0) continue;
            for (const std::uint32_t presence : {1U, 3U}) {
                const Header header{engine, row.name, row.type, requested, presence};
                const auto prepared = owner.prepare(AttributeOperation::kGet, header);
                expect(prepared.result.status == CUDA_SUCCESS &&
                           prepared.output_elements == expected &&
                           prepared.output_bytes == expected * 4,
                       "R0 writes all notes; large capacity only authorizes exact bounded prefix");
                const auto result = owner.complete(Operation::kGet, header, prepared.generation);
                native(result.result);
                expect(result.values.scalars ==
                               fixture::bytes(
                                   std::span<const std::uint32_t>{row.values}.first(expected)) &&
                           result.count_bytes == (presence == 3 ? 8U : 0U),
                       "numeric and behavior note bytes are not confused with knob capacity");
            }
        }
        native(owner.get({engine, row.name, row.type, -1, 3}).result, CUDNN_STATUS_BAD_PARAM);
        native(owner.get({engine, row.name, row.type, 0, 0}).result, CUDNN_STATUS_BAD_PARAM);
    }
    for (const bool empty : {false, true}) {
        fixture::state.numerical = empty ? std::vector<std::uint32_t>{}
                                         : std::vector<std::uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        fixture::state.behavior =
            empty ? std::vector<std::uint32_t>{} : std::vector<std::uint32_t>{0, 1, 2, 3};
        const auto boundary = owner.create(2);
        native(owner.finalize(boundary, true));
        for (const auto& row :
             {Row{1303, 11, fixture::state.numerical}, Row{1305, 19, fixture::state.behavior}}) {
            const auto result = owner.get({boundary, row.name, row.type, 0, 3});
            native(result.result);
            expect(result.count == static_cast<std::int64_t>(row.values.size()) &&
                       result.values.scalars ==
                           fixture::bytes(std::span<const std::uint32_t>{row.values}),
                   "empty and maximum native note sets preserve their complete exact prefix");
        }
    }
    owner.close();
}

void layout_arrays_and_aliases() {
    Owner owner;
    const auto engine = owner.create(2);
    const auto first = owner.create(9), second = owner.create(9), tail = owner.create(9);
    native(owner.finalize(engine, true));
    const auto untouched = owner.snapshot(tail)->generation;
    native(owner.get({engine, 1304, 15, 0, 3}).result);
    expect(fixture::state.written.empty(), "LayoutInfo R0 writes zero destinations, not all");
    for (const auto request : {INT64_MIN, INT64_C(-1), INT64_C(0), INT64_C(1), INT64_C(99)}) {
        const auto result = owner.get({engine, 1304, 15, request, 2});
        native(result.result);
        expect(result.count == (request <= 0 ? 0 : std::min<std::int64_t>(request, 3)),
               "LayoutInfo count-only prefix is bounded by original generated cardinality");
    }
    const Header four{engine, 1304, 15, 4, 3};
    const auto prepared = owner.prepare(AttributeOperation::kGet, four);
    expect(prepared.input_elements == 4 && prepared.output_bytes == 0,
           "all requested destination identities, including unwritten tail, are captured");
    const auto result = owner.get(four, {first, second, first, tail});
    native(result.result);
    expect(result.count == 3 && fixture::state.validated.size() == 4 &&
               fixture::state.written.size() == 3 && owner.snapshot(tail)->generation == untouched,
           "actual emitted prefix only mutates destinations; caller identity array has no output "
           "bytes");
    expect(scalar_get<std::int64_t>(owner, first, 1100, 3) == 303 &&
               scalar_get<std::int64_t>(owner, second, 1100, 3) == 202,
           "duplicate destinations use last actually written copy, not first or unwritten tail");
    native(owner.get(four, {first, second, tail, first}).result);
    expect(scalar_get<std::int64_t>(owner, first, 1100, 3) == 101,
           "unwritten aliased fourth target does not replay the final source element");
    const auto layouts = owner.get({tail, 1101, 12, 0, 3});
    native(layouts.result);
    expect(layouts.count == 3 &&
               layouts.values.scalars ==
                   fixture::bytes(std::span<const std::uint32_t>{fixture::state.layouts[2].types}),
           "LayoutType R0 publishes all independent enum values");
    for (const auto requested : {INT64_C(-1), INT64_C(0), INT64_C(1), INT64_C(2), INT64_MAX}) {
        const auto expected = requested < 0    ? 0
                              : requested == 0 ? 3
                                               : std::min<std::int64_t>(requested, 3);
        const auto count = owner.get({tail, 1101, 12, requested, 2});
        native(count.result);
        expect(count.count == expected, "LayoutType signed count-only semantics");
        if (requested < 0) continue;
        for (const std::uint32_t presence : {1U, 3U}) {
            const auto result = owner.get({tail, 1101, 12, requested, presence});
            native(result.result);
            expect(result.values.scalars ==
                       fixture::bytes(
                           std::span<const std::uint32_t>{fixture::state.layouts[2].types}.first(
                               static_cast<std::size_t>(expected))),
                   "LayoutType output prefix is independent of optional count pointer");
        }
    }
    for (const auto requested : {INT64_MIN, INT64_C(0), INT64_C(2), INT64_MAX}) {
        const auto count = owner.get({first, 1100, 3, requested, 2});
        native(count.result);
        expect(count.count == 1, "UID count-only ignores signed capacity");
    }
    for (const auto requested : {INT64_C(0), INT64_C(2)})
        native(owner.get({first, 1100, 3, requested, 3}).result, CUDNN_STATUS_BAD_PARAM);

    const auto wrong = owner.create(38);
    auto before = fixture::state.get_calls[1304];
    policy(owner.get(four, {first, second, tail, wrong}).result, CUDA_ERROR_INVALID_HANDLE);
    expect(fixture::state.get_calls[1304] == before,
           "unwritten wrong-kind tail is rejected before native");
    auto borrow = owner.registry->borrow_backend(tail);
    const auto deferred = owner.get(four, {first, second, first, tail});
    expect(deferred.result.disposition == CudaDnnDisposition::kDeferred &&
               deferred.result.status == CUDA_SUCCESS && !deferred.result.original_status &&
               fixture::state.get_calls[1304] == before,
           "unwritten destination still needs exclusive mutable ownership");
    borrow.reset();
    policy(owner.prepare(AttributeOperation::kGet, {engine, 1304, 15, 65537, 1}).result,
           CUDA_ERROR_INVALID_VALUE);
    owner.close();
}

void layout_partial_errors_and_reset() {
    Owner owner;
    const auto engine = owner.create(2), a = owner.create(9), b = owner.create(9);
    native(owner.finalize(engine, true));
    fixture::state.fail_after_copy = 1;
    const auto result = owner.get({engine, 1304, 15, 2, 3}, {a, b});
    native(result.result, CUDNN_STATUS_BAD_PARAM);
    expect(result.count_bytes == 0 && result.values.scalars.empty() &&
               !owner.snapshot(a)->is_valid && !owner.snapshot(b)->is_valid &&
               owner.snapshot(engine)->is_valid && !owner.registry->snapshot().is_quarantined &&
               fixture::state.written.size() == 1,
           "partial error invalidates possible destinations, never publishes count or faults "
           "ownership");
    fixture::state.fail_after_copy = 0;
    native(owner.registry->initialize_backend(a));
    expect(
        fixture::object(owner.original(a)).layout.uid == 101 &&
            owner.snapshot(a)->attributes.empty(),
        "Layout Initialize keeps native opaque bytes but clears framework copied-value authority");
    native(owner.get({a, 1100, 3, 1, 3}).result, CUDNN_STATUS_NOT_INITIALIZED);
    native(owner.set_scalar(a, 1100, 3, std::int64_t{9}).result, CUDNN_STATUS_NOT_SUPPORTED);
    native(owner.registry->initialize_backend(a));
    native(owner.registry->initialize_backend(b));
    // A generated child can carry an unfinalized header even when its parent is finalized.
    fixture::state.layouts[0].header.finalized = false;
    native(owner.registry->initialize_backend(engine));
    native(owner.finalize(engine, true));
    native(owner.get({engine, 1304, 15, 1, 3}, {a}).result);
    expect(!owner.snapshot(a)->is_finalized,
           "parent finalization is not invented child finalization");
    native(owner.get({a, 1100, 3, 1, 3}).result, CUDNN_STATUS_NOT_INITIALIZED);
    owner.close();
}

void property_discovery_and_full_reset() {
    Owner owner;
    const auto property = owner.create(38);
    auto calls = fixture::state.finalizes;
    policy(owner.finalize(property), CUDA_ERROR_INVALID_CONTEXT);
    expect(fixture::state.finalizes == calls && !owner.registry->snapshot().is_quarantined,
           "fresh default discovery rejects before original; exact clean retry remains possible");
    native(owner.set_scalar(property, 2700, 23, std::int32_t{-17}).result);
    expect(fixture::object(owner.original(property)).property.id == -17,
           "native setter receives the exact signed ID, not the current session ordinal");
    // Do not invent successful native discovery for an invalid physical ID.
    native(owner.registry->initialize_backend(property));
    native(owner.finalize(property, true));
    expect(
        fixture::state.discoveries == 1 && scalar_get<std::int32_t>(owner, property, 2700, 23) == 0,
        "default device discovery occurs exactly once under its authenticated frame");
    native(owner.registry->initialize_backend(property));
    const auto& reset = fixture::object(owner.original(property)).property;
    expect(reset.id == 0 && !reset.populated && !reset.json_mode && reset.json.empty() &&
               std::all_of(reset.copied_value.begin(), reset.copied_value.end(),
                           [](auto byte) { return byte == std::byte{}; }),
           "Property Initialize fully resets ID, flags, JSON, and copied native data");
    calls = fixture::state.finalizes;
    policy(owner.finalize(property), CUDA_ERROR_INVALID_CONTEXT);
    expect(fixture::state.finalizes == calls,
           "reset property no longer has CPU-ready value provenance");
    native(owner.finalize(property, true));
    expect(scalar_get<std::int32_t>(owner, property, 2700, 23) == 0,
           "reset default ID is exactly zero");
    native(owner.get({property, 2701, 0, 1, 2}).result, CUDNN_STATUS_NOT_SUPPORTED);
    owner.close();
}

void handle_copy_survives_context_death() {
    Owner owner;
    const auto property = owner.create(38);
    owner.handle_copy(property);
    const auto bytes = fixture::object(owner.original(property)).property.copied_value;
    expect(fixture::state.handle_copies == 1,
           "native setter copied the live authenticated handle value");
    owner.end_context();
    const auto discoveries = fixture::state.discoveries;
    native(owner.finalize(property));
    expect(fixture::state.discoveries == discoveries &&
               fixture::object(owner.original(property)).property.copied_value == bytes &&
               scalar_get<std::int32_t>(owner, property, 2700, 23) == 0,
           "populated value needs neither its original handle nor context for Finalize/Get");
    native(owner.registry->initialize_backend(property));
    policy(owner.finalize(property), CUDA_ERROR_INVALID_CONTEXT);
    owner.close();
}

void json_exact_bytes_and_unsafe_forms() {
    Owner owner;
    const auto property = owner.create(38);
    // Raw transport, not vendor parser acceptance: the independent model stores
    // all bytes, including an embedded NUL and a non-NUL final byte.
    std::string json = "{\"synthetic\":\"" + std::string(330, 'q') + "\"}";
    json[40] = '\0';
    owner.json(property, json);
    owner.end_context();
    native(owner.finalize(property));
    for (const auto requested : {INT64_MIN, INT64_C(-1), INT64_C(0)}) {
        const auto result = owner.get({property, 2702, 24, requested, 2});
        native(result.result);
        expect(result.count == static_cast<std::int64_t>(json.size()) && result.count_bytes == 8 &&
                   result.values.scalars.empty(),
               "JSON nonpositive count-only returns exact full byte count, without strlen");
    }
    for (const auto requested : {static_cast<std::int64_t>(json.size()), INT64_MAX}) {
        const auto ready =
            owner.prepare(AttributeOperation::kGet, {property, 2702, 24, requested, 3});
        expect(ready.result.status == CUDA_SUCCESS && ready.output_bytes == json.size() &&
                   ready.output_elements == json.size(),
               "JSON caller capacity does not size storage; exact native N may exceed 256 bytes");
        const auto result =
            owner.complete(Operation::kGet, {property, 2702, 24, requested, 3}, ready.generation);
        native(result.result);
        expect(result.count == static_cast<std::int64_t>(json.size()) &&
                   result.values.scalars == fixture::bytes(std::span<const char>{json}),
               "JSON publishes N exact bytes, embedded NUL preserved and no extra terminator");
    }
    const auto short_result = owner.get({property, 2702, 24, 1, 3});
    native(short_result.result, fixture::status(2006));
    expect(short_result.count_bytes == 0 && short_result.values.scalars.empty() &&
               owner.snapshot(property)->is_valid,
           "safe short JSON getter keeps original 2006 and leaves outputs/source untouched");
    for (const Header unsafe :
         {Header{property, 2702, 24, 0, 1}, Header{property, 2702, 24, INT64_MAX, 1},
          Header{property, 2702, 24, INT64_MAX, 2}}) {
        const auto before = fixture::state.get_history.size();
        const auto result = owner.prepare(AttributeOperation::kGet, unsafe);
        expect(result.result.status != CUDA_SUCCESS && !result.result.original_status &&
                   std::ranges::all_of(std::span{fixture::state.get_history}.subspan(before),
                                       [](const auto& call) {
                                           return call.name == 2702 && call.requested == 0 &&
                                                  call.has_count && !call.has_array;
                                       }),
               "unsafe requested form never issues; only safe bounded private count observation is "
               "allowed");
    }
    native(owner.registry->initialize_backend(property));
    for (const auto count : {INT64_MIN, INT64_C(-1), INT64_C(65537)}) {
        const auto before = fixture::state.sets;
        const auto result = owner.prepare(AttributeOperation::kSet, {property, 2702, 24, count, 1});
        expect(result.result.status != CUDA_SUCCESS && !result.result.original_status &&
                   fixture::state.sets == before,
               "unsafe negative or over-quota JSON source never reaches SDK string replacement");
    }
    owner.close();
}

void property_copies_preserve_flags_and_lifetimes() {
    for (const bool finalized : {false, true}) {
        Owner owner;
        const auto property = owner.create(38);
        const std::string json = "{\"synthetic_copy\":19}";
        owner.json(property, json);
        if (finalized) native(owner.finalize(property));
        const std::array<std::pair<std::uint32_t, std::uint32_t>, 3> rows{
            {{4, 204}, {2, 1307}, {5, 407}}};
        std::array<std::uint64_t, 3> parents{}, destinations{};
        for (std::size_t i = 0; i < rows.size(); ++i) {
            parents[i] = owner.create(rows[i].first);
            destinations[i] = owner.create(38);
            owner.copy_in(parents[i], rows[i].second, property);
            native(owner.finalize(parents[i], true));
        }
        native(owner.registry->initialize_backend(property));
        native(owner.registry->destroy(property));
        owner.end_context();
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto name = rows[i].second;
            for (const auto requested : {INT64_MIN, INT64_C(0), INT64_MAX}) {
                const auto count = owner.get({parents[i], name, 15, requested, 2});
                native(count.result);
                expect(count.count == 1, "property copy count-only ignores signed requested count");
            }
            native(owner.get({parents[i], name, 15, 2, 3}).result, CUDNN_STATUS_NOT_SUPPORTED);
            const auto copied = owner.get({parents[i], name, 15, 1, 3}, {destinations[i]});
            native(copied.result);
            expect(copied.count == 1 &&
                       owner.snapshot(destinations[i])->is_finalized == finalized &&
                       fixture::object(owner.original(destinations[i])).property.json == json,
                   "CPU copy after source/context death preserves child header and deep bytes");
            native(owner.registry->destroy(parents[i]));
            if (!finalized) {
                native(owner.get({destinations[i], 2700, 23, 1, 3}).result,
                       CUDNN_STATUS_NOT_INITIALIZED);
                native(owner.finalize(destinations[i]));
            }
            const auto output = owner.get({destinations[i], 2702, 24, INT64_MAX, 3});
            native(output.result);
            expect(output.values.scalars == fixture::bytes(std::span<const char>{json}),
                   "copied JSON is independent of destroyed parent and reset source");
        }
        owner.close();
    }
}

void property_failure_recovery_and_generation() {
    Owner owner;
    const auto property = owner.create(38);
    fixture::state.set_status = CUDNN_STATUS_BAD_PARAM;
    Values value;
    const std::string json = "{\"partial\":1}";
    value.scalars = fixture::bytes(std::span<const char>{json});
    const auto failed = owner.attribute(
        Operation::kSet, {property, 2702, 24, static_cast<std::int64_t>(json.size()), 1}, value);
    native(failed.result, CUDNN_STATUS_BAD_PARAM);
    expect(!owner.snapshot(property)->is_valid && !owner.registry->snapshot().is_quarantined,
           "partial native mutation invalidates semantic configuration, not ownership");
    const auto before = fixture::state.finalizes;
    const auto rejected = owner.finalize(property);
    expect(rejected.status != CUDA_SUCCESS && !rejected.original_status &&
               fixture::state.finalizes == before,
           "failed setter cannot manufacture CPU-ready Finalize permission");
    fixture::state.set_status = CUDNN_STATUS_SUCCESS;
    native(owner.registry->initialize_backend(property));
    const Header header{property, 2702, 24, static_cast<std::int64_t>(json.size()), 1};
    const auto old = owner.prepare(AttributeOperation::kSet, header);
    native(owner.registry->initialize_backend(property));
    const auto set_calls = fixture::state.sets;
    policy(owner.complete(Operation::kSet, header, old.generation, value).result,
           CUDA_ERROR_INVALID_VALUE);
    expect(fixture::state.sets == set_calls,
           "Initialize invalidates prepared generation before any native mutation");
    owner.json(property, json);
    fixture::state.finalize_status = CUDNN_STATUS_BAD_PARAM;
    native(owner.finalize(property), CUDNN_STATUS_BAD_PARAM);
    expect(!owner.registry->snapshot().is_quarantined,
           "original parse failure does not quarantine unrelated ownership");
    fixture::state.finalize_status = CUDNN_STATUS_SUCCESS;
    native(owner.registry->initialize_backend(property));
    policy(owner.finalize(property), CUDA_ERROR_INVALID_CONTEXT);
    owner.close();
}

void property_copy_errors_and_json_bound() {
    Owner owner;
    const auto property = owner.create(38), destination = owner.create(38),
               parent = owner.create(2);
    const std::string json(65536, 'x');
    owner.json(property, json);
    native(owner.finalize(property));
    const auto bounded = owner.get({property, 2702, 24, INT64_MAX, 3});
    native(bounded.result);
    expect(bounded.count == 65536 &&
               bounded.values.scalars == fixture::bytes(std::span<const char>{json}),
           "maximum admitted JSON bytes are fully returned, not truncated to stack scratch");
    owner.copy_in(parent, 1307, property);
    native(owner.finalize(parent, true));
    const auto null_target = owner.get({parent, 1307, 15, 1, 3}, {0});
    native(null_target.result, CUDNN_STATUS_BAD_PARAM);
    expect(!null_target.count_bytes && null_target.values.scalars.empty(),
           "safe null destination preserves original error without output publication");
    fixture::state.fail_after_copy = 1;
    const auto partial = owner.get({parent, 1307, 15, 1, 3}, {destination});
    native(partial.result, CUDNN_STATUS_BAD_PARAM);
    expect(!partial.count_bytes && !owner.snapshot(destination)->is_valid &&
               owner.snapshot(parent)->is_valid && !owner.registry->snapshot().is_quarantined &&
               fixture::object(owner.original(destination)).property.json == json,
           "partial deep copy error invalidates destination semantics without rollback or "
           "ownership loss");
    fixture::state.fail_after_copy = 0;
    native(owner.registry->initialize_backend(destination));
    fixture::state.failed_get = 2700;  // Optional copied-header observation fails.
    native(owner.get({parent, 1307, 15, 1, 3}, {destination}).result);
    expect(
        !owner.registry->snapshot().is_quarantined,
        "failed optional destination observation preserves requested copy success and ownership");
    fixture::state.failed_get.reset();
    expect(scalar_get<std::int32_t>(owner, destination, 2700, 23) == 0,
           "subsequent original scalar query observes the copied value");
    const auto missing = owner.create(2);
    native(owner.finalize(missing, true));
    native(owner.get({missing, 1307, 15, 1, 3}, {destination}).result, fixture::status(2002));
    owner.close();
}

void optional_observation_and_reentry() {
    for (const unsigned int mode : {0U, 1U, 2U}) {
        Owner owner(mode != 0);
        const auto engine = owner.create(2);
        if (mode == 1) fixture::state.failed_get = 1303;
        if (mode == 2) fixture::state.corrupt_count = 1303;
        native(owner.finalize(engine, true));
        expect(
            owner.snapshot(engine)->is_valid && owner.snapshot(engine)->is_finalized &&
                !owner.registry->snapshot().is_quarantined,
            "missing/error/malformed optional observer never replaces original Finalize success");
        const auto before = fixture::state.get_history.size();
        const auto prepared = owner.prepare(AttributeOperation::kGet, {engine, 1303, 11, 0, 3});
        expect(prepared.result.status != CUDA_SUCCESS && !prepared.result.original_status &&
                   std::ranges::all_of(std::span{fixture::state.get_history}.subspan(before),
                                       [](const auto& call) {
                                           return call.name == 1303 && call.requested == 0 &&
                                                  call.has_count && !call.has_array;
                                       }),
               "missing prefix proof permits safe private retry, never the requested array call");
        owner.close();
    }
    {
        Owner owner;
        const auto property = owner.create(38);
        owner.json(property, "{\"optional\":1}");
        fixture::state.failed_get = 2702;
        native(owner.finalize(property));
        expect(owner.snapshot(property)->is_valid && !owner.registry->snapshot().is_quarantined,
               "private JSON observation failure does not replace successful original Finalize");
        const auto prepared =
            owner.prepare(AttributeOperation::kGet, {property, 2702, 24, INT64_MAX, 3});
        expect(prepared.result.status != CUDA_SUCCESS && !prepared.result.original_status,
               "unknown JSON length cannot authorize requested array writes");
        fixture::state.failed_get.reset();
        native(owner.get({property, 2702, 24, INT64_MAX, 3}).result);
        owner.close();
    }
    Owner owner;
    const auto property = owner.create(38);
    owner.json(property, "{\"callback\":1}");
    unsigned int callbacks{};
    fixture::state.callback = [&] {
        ++callbacks;
        const auto creates = fixture::state.creates;
        const auto finalizes = fixture::state.finalizes;
        const auto nested = owner.registry->create_backend(38);
        expect(nested.status == CUDA_ERROR_NOT_READY && !nested.original_status &&
                   fixture::state.creates == creates,
               "registry callback reentry cannot issue another native lifecycle call");
        expect(!owner.registry->borrow_backend(property),
               "active native scope excludes new metadata borrows");
        const auto nested_finalize = owner.finalize(property);
        expect(nested_finalize.status == CUDA_ERROR_NOT_READY && !nested_finalize.original_status &&
                   fixture::state.finalizes == finalizes,
               "version, original, and private observer share one exclusion scope");
    };
    native(owner.finalize(property));
    fixture::state.callback = {};
    expect(callbacks >= 2, "original and private observation callbacks actually exercised reentry");
    owner.close();
}

void context_loss_case() {
    Owner owner;
    const auto property = owner.create(38);
    bool lost{};
    fixture::state.callback = [&] {
        if (!lost && fixture::state.finalizes) {
            lost = true;
            driver::driver.current = nullptr;
        }
    };
    // Do not perform context-dependent discovery after the callback destroys
    // proof: use a copied value, but explicitly request a framed operation.
    owner.handle_copy(property);
    const auto before = fixture::state.gets;
    const auto result = owner.finalize(property, true);
    expect(lost && result.status != CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS && fixture::state.gets == before &&
               owner.session->snapshot().is_quarantined,
           "post-native frame loss preserves issued status and stops all private observers");
    fixture::state.callback = {};
    // Deliberately no unsafe cleanup. The caller runs this case in an isolated
    // child so unproven native ownership remains retained until process exit.
}

std::shared_ptr<CudaDnnBackendSnapshot> tensor(std::int64_t uid) {
    auto result = std::make_shared<CudaDnnBackendSnapshot>();
    result->kind = 17;
    result->generation = 1;
    result->is_valid = result->is_finalized = true;
    CudaDnnBackendTensorObservation observation;
    observation.uid = uid;
    observation.type = 0;
    observation.alignment = {16, CudaDnnBackendTensorAlignmentSource::kNativeSet};
    observation.dimension_count = observation.stride_count = 2;
    observation.dimensions = {2, 3};
    observation.strides = {8, 2};
    observation.vector_count = 1;
    observation.vectorized_dimension = -1;
    result->tensor = observation;
    return result;
}
void descriptive_edges_never_grant_tensor_authority() {
    const auto real_tensor = tensor(71);
    auto graph = std::make_shared<CudaDnnBackendSnapshot>();
    graph->kind = 15;
    graph->generation = 1;
    graph->is_valid = graph->is_finalized = true;
    CudaDnnBackendSnapshotAttribute tensors;
    tensors.name = 801;
    tensors.type = 15;
    tensors.descriptors = {real_tensor};
    graph->attributes.push_back(tensors);
    auto property = std::make_shared<CudaDnnBackendSnapshot>();
    property->kind = 38;
    property->generation = 1;
    property->is_valid = true;
    // Descriptive bytes/children cannot become a second executable tensor DAG.
    CudaDnnBackendSnapshotAttribute forged;
    forged.name = 2702;
    forged.type = 24;
    forged.scalars = fixture::scalar(std::int32_t{12345});
    forged.descriptors = {tensor(999)};
    property->attributes.push_back(forged);
    for (const auto& [kind, name] :
         {std::pair{2U, 1307U}, std::pair{4U, 204U}, std::pair{5U, 407U}}) {
        CudaDnnBackendSnapshot parent;
        parent.kind = kind;
        parent.generation = 1;
        parent.is_valid = parent.is_finalized = true;
        parent.generated_graph = graph;
        CudaDnnBackendSnapshotAttribute description;
        description.name = name;
        description.type = 15;
        description.descriptors = {property};
        parent.attributes.push_back(description);
        const auto result = cuda_dnn_backend_tensor_requirements(parent);
        expect(result.status == CUDA_SUCCESS && result.tensors.size() == 1 &&
                   result.tensors[0].uid == 71 && result.tensors[0].bytes == 52 &&
                   result.tensors[0].alignment == 16,
               "known descriptive edges ignore unfinalized property and forged nested tensor while "
               "preserving real extent");
        parent.attributes[0].name = 9999;
        expect(cuda_dnn_backend_tensor_requirements(parent).status != CUDA_SUCCESS,
               "kind38 is not globally ignored through arbitrary graph attributes");
        parent.attributes[0].name = name;
        parent.attributes[0].descriptors = {tensor(999)};
        const auto wrong_child = cuda_dnn_backend_tensor_requirements(parent);
        expect(wrong_child.status != CUDA_SUCCESS || wrong_child.tensors.size() != 1,
               "property-name exemption cannot silently hide an actual Tensor child");
        parent.attributes[0].descriptors = {property};
        auto missing = std::make_shared<CudaDnnBackendSnapshot>(*graph);
        auto invalid_tensor = std::make_shared<CudaDnnBackendSnapshot>(*real_tensor);
        invalid_tensor->tensor.reset();
        missing->attributes[0].descriptors = {invalid_tensor};
        parent.generated_graph = missing;
        expect(cuda_dnn_backend_tensor_requirements(parent).status != CUDA_SUCCESS,
               "descriptive data cannot repair missing authoritative Tensor observations");
    }
}
}  // namespace

int main() {
    try {
        note_counts_and_exact_prefixes();
        layout_arrays_and_aliases();
        layout_partial_errors_and_reset();
        property_discovery_and_full_reset();
        handle_copy_survives_context_death();
        json_exact_bytes_and_unsafe_forms();
        property_copies_preserve_flags_and_lifetimes();
        property_failure_recovery_and_generation();
        property_copy_errors_and_json_bound();
        optional_observation_and_reentry();
        descriptive_edges_never_grant_tensor_authority();
        const auto child = ::fork();
        expect(child >= 0, "isolated context-loss fixture fork");
        if (!child) {
            try {
                context_loss_case();
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "context loss: " << error.what() << '\n';
                ::_exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated native context-loss proof");
        std::cout << "independent Engine/LayoutInfo/DeviceProperty regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
