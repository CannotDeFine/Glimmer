#include "dnn_backend_graph_fixture.h"

#include <bit>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_backend_graph;
using Operation = CudaDnnBackendAttributeOperation;
using Direction = CudaDnnBackendAttributeDirection;
using Header = CudaDnnBackendAttributeHeader;
using Values = CudaDnnBackendAttributeValues;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "exact original success receipt");
}
void native_error(const CudaDnnResult& result, cudnnStatus_t error) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == error,
           "ordinary native failure keeps exact status and known native ownership");
}
class Owner {
   public:
    explicit Owner(unsigned int missing = 0) : registry(initialize(missing)) {}
    CudaDnnRegistry registry;
    std::uint64_t create(std::uint32_t kind) {
        const auto result = registry.create_backend(kind);
        success(result);
        expect(result.id != 0, "created object has a nonzero public ID");
        return result.id;
    }
    std::shared_ptr<const CudaDnnBackendSnapshot> snapshot(std::uint64_t id) {
        const auto view = registry.borrow_backend(id);
        expect(view && view->object.kind == CudaDnnObjectKind::kBackend && view->snapshot,
               "owned backend snapshot is observable");
        return view->snapshot;
    }
    CudaDnnBackendAttributePreparation prepare(Operation operation, const Header& header) {
        auto result = registry.prepare_backend_attribute(operation, header);
        expect(result.result.disposition == CudaDnnDisposition::kComplete &&
                   result.result.status == CUDA_SUCCESS && !result.result.original_status &&
                   result.generation,
               "private preparation has generation but no requested-native provenance");
        return result;
    }
    CudaDnnBackendAttributeResult set(const Header& header, const Values& values) {
        const auto ready = prepare(Operation::kSet, header);
        return registry.set_backend_attribute(header, ready.generation, values);
    }
    CudaDnnBackendAttributeResult get(const Header& header, const Values& values = {}) {
        const auto ready = prepare(Operation::kGet, header);
        return registry.get_backend_attribute(header, ready.generation, values);
    }
    template <typename T>
    void scalar(std::uint64_t id, std::uint32_t name, std::uint32_t type, T value) {
        success(set({id, name, type, 1, 1}, {fixture::scalar(value), {}}).result);
    }
    template <std::size_t Count>
    void array(std::uint64_t id, std::uint32_t name,
               const std::array<std::int64_t, Count>& values) {
        success(set({id, name, 3, static_cast<std::int64_t>(Count), 1},
                    {fixture::bytes(std::span<const std::int64_t>{values}), {}})
                    .result);
    }
    void tensor(std::uint64_t id, std::int64_t uid) {
        scalar(id, 900, 3, std::int64_t{16});
        scalar(id, 901, 1, std::uint32_t{CUDNN_DATA_FLOAT});
        array(id, 902, std::array<std::int64_t, 4>{1, 2, 3, 4});
        array(id, 903, std::array<std::int64_t, 4>{24, 12, 4, 1});
        scalar(id, 906, 3, uid);
        scalar(id, 907, 2, false);
        success(registry.finalize_backend(id));
    }
    void close() {
        registry.close();
        expect(registry.shutdown().status == CUDA_SUCCESS && registry.snapshot().is_closed,
               "all proven native graph ownership can shut down");
        expect(!fixture::alive() && fixture::state.creates == fixture::state.destroys &&
                   !fixture::foundation::control.driver_calls.load(),
               "every original descriptor destroys once without Driver/context work");
    }

   private:
    static CudaDnnDispatch initialize(unsigned int missing) {
        fixture::reset();
        return fixture::dispatch(missing);
    }
};

const CudaDnnBackendSnapshotAttribute& attribute(const CudaDnnBackendSnapshot& snapshot,
                                                 std::uint32_t name) {
    for (const auto& value : snapshot.attributes)
        if (value.name == name) return value;
    throw std::runtime_error("expected independent copied semantic attribute");
}

void kinds_and_scalars() {
    Owner owner;
    for (const auto kind : {0U, 1U, 2U, 3U, 5U, 7U, 10U, 11U, 12U, 13U, 15U, 16U, 17U, 18U, 19U,
                            21U, 22U, 24U, 32U}) {
        const auto id = owner.create(kind);
        const auto initial = owner.snapshot(id);
        expect(initial->kind == kind && initial->is_valid && !initial->is_finalized,
               "every admitted family preserves fresh configuration without hidden finalize");
        const auto creates = fixture::state.creates;
        const auto destroys = fixture::state.destroys;
        success(owner.registry.initialize_backend(id));
        const auto reset = owner.snapshot(id);
        expect(reset->generation > initial->generation && reset->is_valid && !reset->is_finalized &&
                   fixture::state.creates == creates && fixture::state.destroys == destroys,
               "original one-argument Initialize advances generation without destroy/create");
        success(owner.registry.destroy(id));
    }
    expect(!fixture::state.sets && !fixture::state.finalizes,
           "lifecycle foundation introduces no hidden setter/finalize");
    // LayoutInfo/DeviceProperty lifecycle now has its own independent fixture.
    for (const auto kind : {34U, 63U, UINT32_MAX}) {
        const auto calls = fixture::state.creates;
        const auto result = owner.registry.create_backend(kind);
        expect(!result.original_status && result.status == CUDA_ERROR_NOT_SUPPORTED &&
                   fixture::state.creates == calls,
               "unaudited descriptor families fail before native create");
    }
    struct ScalarCase {
        std::uint32_t kind, name, type;
        std::vector<std::byte> value;
    };
    const std::array cases{
        ScalarCase{0, 0, 14, fixture::scalar(std::uint32_t{CUDNN_POINTWISE_RELU_FWD})},
        ScalarCase{0, 1, 1, fixture::scalar(std::uint32_t{CUDNN_DATA_DOUBLE})},
        ScalarCase{0, 3, 5, fixture::scalar(std::bit_cast<double>(UINT64_C(0x8000000000000000)))},
        ScalarCase{1, 100, 1, fixture::scalar(std::uint32_t{CUDNN_DATA_FLOAT})},
        ScalarCase{1, 101, 7, fixture::scalar(std::uint32_t{CUDNN_CROSS_CORRELATION})},
        ScalarCase{18, 1500, 1, fixture::scalar(std::uint32_t{CUDNN_DATA_DOUBLE})},
        ScalarCase{21, 1600, 18, fixture::scalar(std::uint32_t{CUDNN_REDUCE_TENSOR_ADD})},
        ScalarCase{21, 1601, 1, fixture::scalar(std::uint32_t{CUDNN_DATA_DOUBLE})}};
    for (const auto& item : cases) {
        const auto id = owner.create(item.kind);
        success(owner.set({id, item.name, item.type, 1, 1}, {item.value, {}}).result);
        {
            const auto view = owner.registry.borrow_backend(id);
            expect(
                view &&
                    fixture::object(view->object.native).snapshot.attributes.at(item.name).bytes ==
                        item.value,
                "native setter receives exact independently expected scalar bytes");
        }
        success(owner.registry.finalize_backend(id));
        const auto result = owner.get({id, item.name, item.type, 1, 3});
        if (item.kind == 0) {
            native_error(result.result, CUDNN_STATUS_NOT_SUPPORTED);
            expect(result.values.scalars.empty() && !result.count_bytes &&
                       owner.snapshot(id)->is_valid,
                   "PointwiseDescriptor getter is original NOT_SUPPORTED, not synthesized state");
        } else {
            success(result.result);
            expect(
                result.count_bytes == 8 && result.count == 1 && result.values.scalars == item.value,
                "independent scalar families retain complete original value bits");
            for (const auto requested : {0, 2}) {
                const Header header{id, item.name, item.type, requested, 3};
                const auto prepared = owner.prepare(Operation::kGet, header);
                expect(!prepared.output_elements && !prepared.output_bytes,
                       "invalid fixed-scalar capacity grants no caller output bytes");
                const auto calls = fixture::state.gets;
                const auto failed =
                    owner.registry.get_backend_attribute(header, prepared.generation, {});
                native_error(failed.result, CUDNN_STATUS_BAD_PARAM);
                expect(fixture::state.gets == calls + 1 && fixture::state.last_count == requested &&
                           failed.values.scalars.empty() && !failed.count_bytes &&
                           owner.snapshot(id)->is_valid &&
                           !owner.registry.snapshot().is_quarantined,
                       "fixed-scalar count error reaches native without output or owner poisoning");
                const auto count_only = owner.get({id, item.name, item.type, requested, 2});
                success(count_only.result);
                expect(
                    count_only.count == 1 && count_only.count_bytes == 8 &&
                        count_only.values.scalars.empty(),
                    "null scalar array preserves native count-only behavior at the same capacity");
            }
        }
        success(owner.registry.destroy(id));
    }
    owner.close();
}

void tensor_prefix_and_padding() {
    Owner owner;
    const auto tensor = owner.create(17);
    owner.tensor(tensor, 7);
    const std::array<std::int64_t, 4> dimensions{1, 2, 3, 4};
    for (const auto requested : {0, 1, 2, 4, 6}) {
        for (const std::uint32_t presence : {1U, 2U, 3U}) {
            const Header header{tensor, 902, 3, requested, presence};
            const auto prepared = owner.prepare(Operation::kGet, header);
            const auto copied = presence & 1 ? std::min(requested, 4) : 0;
            expect(prepared.input_elements == 0 &&
                       prepared.output_elements == static_cast<std::uint32_t>(copied) &&
                       prepared.count_bytes == (presence & 2 ? 8U : 0U),
                   "prepare exposes actual requested prefix independently of count nullness");
            const auto result =
                owner.registry.get_backend_attribute(header, prepared.generation, {});
            success(result.result);
            const auto expected =
                fixture::bytes(std::span<const std::int64_t>{dimensions}.first(copied));
            expect(result.values.scalars == expected &&
                       result.count == (presence & 2 ? (presence & 1 ? copied : 4) : 0),
                   "Tensor native present-array count is copied prefix, count-only returns rank");
        }
    }
    for (const auto count : {INT64_MIN, INT64_MAX}) {
        const auto result = owner.get({tensor, 902, 3, count, 2});
        success(result.result);
        expect(result.count == 4 && result.values.scalars.empty(),
               "count-only signed extrema need no caller-array materialization");
    }
    for (const auto type : {3U, 4U, 5U, 23U}) {
        const auto matmul = owner.create(18);
        const Header header{matmul, 1503, type, 1, 1};
        const auto ready = owner.prepare(Operation::kSet, header);
        expect(ready.input_elements == (type == 4 || type == 23 ? 2U : 1U),
               "Matmul padding has eight-byte original input even for four-byte type");
        const auto value = fixture::scalar(UINT64_C(0x8123456789abcdef));
        success(owner.registry.set_backend_attribute(header, ready.generation, {value, {}}).result);
        expect(fixture::state.last_count == 1, "padding extent does not rewrite original count");
        success(owner.registry.finalize_backend(matmul));
        auto query = header;
        query.presence = 3;
        const auto output = owner.prepare(Operation::kGet, query);
        expect(output.output_elements == ready.input_elements,
               "padding output uses exact eight-byte write extent, not returned count");
        expect(output.output_bytes == 8, "padding bytes are separate from type and element count");
        const auto result = owner.registry.get_backend_attribute(query, output.generation, {});
        success(result.result);
        expect(result.count == 1 && result.values.scalars == value,
               "complete padding bytes are preserved independently of native count one");
    }
    const auto partial = owner.create(19);
    owner.scalar(partial, 1524, 3, UINT64_C(0xaabbccdd01234567));
    success(owner.registry.finalize_backend(partial));
    const Header partial_query{partial, 1524, 3, 1, 3};
    const auto partial_prepared = owner.prepare(Operation::kGet, partial_query);
    expect(partial_prepared.output_elements == 1 && partial_prepared.output_bytes == 4,
           "Matmul operation INT64 attribute writes only a four-byte prefix");
    const auto partial_result =
        owner.registry.get_backend_attribute(partial_query, partial_prepared.generation, {});
    success(partial_result.result);
    expect(partial_result.values.scalars == fixture::scalar(UINT64_C(0x01234567)) &&
               partial_result.count_bytes == 8 && partial_result.count == 1,
           "unwritten high typed-storage bytes are canonical zero, not caller publication");
    const auto wide = owner.create(17);
    const std::array<std::int64_t, 10> wide_dimensions{1, 2, 3, 4, 5, 6, 7, 8, 9, 901};
    const std::array<std::int64_t, 10> wide_strides{362880, 181440, 60480, 15120, 3024,
                                                    504,    72,     9,     1,     902};
    const Header wide_set{wide, 902, 3, 10, 1};
    const auto wide_prepared = owner.prepare(Operation::kSet, wide_set);
    expect(wide_prepared.input_elements == 10,
           "native logging retains full requested10 input despite functional cap9");
    success(owner.registry
                .set_backend_attribute(
                    wide_set, wide_prepared.generation,
                    {fixture::bytes(std::span<const std::int64_t>{wide_dimensions}), {}})
                .result);
    expect(fixture::state.last_count == 10, "Tensor original count10 is not silently rewritten");
    owner.array(wide, 903, std::array<std::int64_t, 2>{2, 1});
    const auto independent = owner.snapshot(wide);
    expect(attribute(*independent, 902).scalars.size() == 9 * sizeof(std::int64_t) &&
               attribute(*independent, 903).scalars.size() == 2 * sizeof(std::int64_t),
           "Tensor DIM and STRIDE retain independent effective native ranks");
    owner.array(wide, 903, wide_strides);
    success(owner.registry.finalize_backend(wide));
    for (const auto name : {902U, 903U}) {
        const Header query{wide, name, 3, 10, 3};
        const auto ready = owner.prepare(Operation::kGet, query);
        const auto result = owner.registry.get_backend_attribute(query, ready.generation, {});
        success(result.result);
        const auto& expected = name == 902 ? wide_dimensions : wide_strides;
        expect(ready.output_elements == 9 && ready.output_bytes == 72 && result.count == 9 &&
                   result.values.scalars ==
                       fixture::bytes(std::span<const std::int64_t>{expected}.first(9)),
               "Tensor query publishes allnine retained values, not legacy8 or requested10");
    }
    owner.close();
}

void native_tensor_observation() {
    using AlignmentSource = CudaDnnBackendTensorAlignmentSource;
    Owner owner;
    const auto tensor = owner.create(17);
    const auto fresh = owner.snapshot(tensor);
    expect(!fresh->tensor && fresh->tensor_alignment && fresh->tensor_alignment->bytes == 1 &&
               fresh->tensor_alignment->source == AlignmentSource::kNativeDefault,
           "fresh alignment has audited constructor provenance, not a fake successful getter");
    owner.scalar(tensor, 901, 1, std::uint32_t{CUDNN_DATA_INT8});
    owner.array(tensor, 902, std::array<std::int64_t, 3>{1, 1, 4});
    owner.array(tensor, 903, std::array<std::int64_t, 3>{16, 4, 1});
    owner.scalar(tensor, 906, 3, std::int64_t{31});
    owner.scalar(tensor, 908, 2, true);
    owner.scalar(tensor, 909, 20, std::uint32_t{2});
    const auto before_gets = fixture::state.gets;
    success(owner.registry.finalize_backend(tensor));
    const auto finalized = owner.snapshot(tensor);
    expect(finalized->tensor && fixture::state.gets == before_gets + 9 &&
               finalized->tensor->type == CUDNN_DATA_INT8 && finalized->tensor->uid == 31 &&
               finalized->tensor->vector_count == 1 &&
               finalized->tensor->vectorized_dimension == -1 &&
               finalized->tensor->reordering == 2 && !finalized->tensor->is_virtual &&
               finalized->tensor->is_by_value && finalized->tensor->dimension_count == 3 &&
               finalized->tensor->stride_count == 3 && finalized->tensor->dimensions[2] == 4 &&
               finalized->tensor->strides[0] == 16 && finalized->tensor->dimensions[3] == 0 &&
               finalized->tensor->alignment.bytes == 1 &&
               finalized->tensor->alignment.source == AlignmentSource::kNativeDefault,
           "nine exact original observations include native defaults and typed partial reordering");
    native_error(owner.get({tensor, 900, 3, 1, 3}).result, CUDNN_STATUS_NOT_SUPPORTED);
    const Header reorder{tensor, 909, 3, 1, 3};
    const auto prepared = owner.prepare(Operation::kGet, reorder);
    expect(prepared.output_elements == 1 && prepared.output_bytes == 4 && prepared.count_bytes == 8,
           "actual INT64 reordering query writes four caller bytes, not eight");
    const auto observed = owner.registry.get_backend_attribute(reorder, prepared.generation, {});
    success(observed.result);
    expect(observed.count == 1 && observed.values.scalars == fixture::scalar(std::int64_t{2}),
           "typed reordering transport zero-pads the native four-byte output");
    native_error(owner.get({tensor, 909, 20, 1, 3}).result, CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE);

    const auto operation = owner.create(22);
    success(owner.set({operation, 1610, 15, 1, 1}, {{}, {tensor}}).result);
    success(owner.registry.finalize_backend(operation));
    success(owner.registry.initialize_backend(tensor));
    expect(!owner.snapshot(tensor)->tensor && owner.snapshot(tensor)->tensor_alignment &&
               owner.snapshot(tensor)->tensor_alignment->source == AlignmentSource::kNativeDefault,
           "Initialize removes old authoritative observation and restores only audited default");
    success(owner.registry.destroy(tensor));
    const auto destination = owner.create(17);
    success(owner.get({operation, 1610, 15, 1, 3}, {{}, {destination}}).result);
    success(owner.registry.destroy(operation));
    const auto copy = owner.snapshot(destination);
    expect(copy->tensor && copy->tensor->uid == 31 && copy->tensor->alignment.bytes == 1 &&
               finalized->tensor->uid == 31,
           "copied tensor observations outlive reset/destruction of both source and parent");
    const auto rejected =
        owner.set({destination, 906, 3, 1, 1}, {fixture::scalar(std::int64_t{9}), {}});
    native_error(rejected.result, CUDNN_STATUS_BAD_PARAM);
    expect(!owner.snapshot(destination)->tensor && !owner.snapshot(destination)->tensor_alignment &&
               !owner.registry.snapshot().is_quarantined,
           "issued configuration rejection removes stale extent proof without losing ownership");
    success(owner.registry.initialize_backend(destination));
    owner.tensor(destination, 47);
    const auto recovered = owner.snapshot(destination);
    expect(recovered->tensor && recovered->tensor->uid == 47 &&
               recovered->tensor->alignment.bytes == 16 &&
               recovered->tensor->alignment.source == AlignmentSource::kNativeSet,
           "successful audited alignment Set and Finalize reestablish native observations");
    owner.close();
}

void unavailable_tensor_observation() {
    for (unsigned int mode = 0; mode != 15; ++mode) {
        Owner owner(mode == 0 ? 7 : 0);
        const auto tensor = owner.create(17);
        owner.scalar(tensor, 900, 3, std::int64_t{16});
        owner.scalar(tensor, 901, 1, std::uint32_t{CUDNN_DATA_FLOAT});
        owner.array(tensor, 902, std::array<std::int64_t, 2>{2, 3});
        owner.array(tensor, 903, std::array<std::int64_t, 2>{3, 1});
        if (mode >= 1 && mode <= 9) fixture::control.fail_get_name.store(900 + mode);
        if (mode == 10) fixture::control.get_status.store(CUDNN_STATUS_EXECUTION_FAILED);
        if (mode == 11) fixture::control.corrupt_count.store(true);
        if (mode == 12) fixture::control.corrupt_enum.store(true);
        if (mode == 13) fixture::control.corrupt_bool.store(true);
        if (mode == 14) fixture::control.finalize_status.store(CUDNN_STATUS_BAD_PARAM);
        const auto result = owner.registry.finalize_backend(tensor);
        if (mode == 14)
            native_error(result, CUDNN_STATUS_BAD_PARAM);
        else
            success(result);
        const auto observed = owner.snapshot(tensor);
        expect(!observed->tensor && !owner.registry.snapshot().is_quarantined &&
                   fixture::state.finalizes == 1 && observed->is_finalized == (mode != 14) &&
                   observed->is_valid == (mode != 14),
               "unavailable/malformed private observations never replace original Finalize status");
        if (mode == 14)
            expect(!fixture::state.gets && !observed->tensor_alignment,
                   "failed Finalize clears earlier alignment authority without private getters");
        owner.close();
    }
}

void child_copy_and_mutable_get() {
    Owner owner;
    const auto input = owner.create(17);
    owner.tensor(input, 7);
    const auto old = owner.snapshot(input);
    const auto operation = owner.create(22);
    // Audited ReductionOperation reads one descriptor despite nonpositive count.
    for (const auto count : {INT64_MIN, std::int64_t{-1}, std::int64_t{0}, std::int64_t{1}}) {
        const auto ready = owner.prepare(Operation::kSet, {operation, 1610, 15, count, 1});
        expect(ready.input_elements == 1, "descriptor setter read extent is not original count");
        success(owner.registry
                    .set_backend_attribute({operation, 1610, 15, count, 1}, ready.generation,
                                           {{}, {input}})
                    .result);
        expect(fixture::state.last_count == count, "signed original descriptor count is unchanged");
    }
    const auto captured = owner.snapshot(operation);
    expect(attribute(*captured, 1610).descriptors.size() == 1 &&
               attribute(*attribute(*captured, 1610).descriptors[0], 906).scalars ==
                   fixture::scalar(std::int64_t{7}),
           "parent semantic snapshot stores copied child values");
    success(owner.registry.initialize_backend(input));
    owner.tensor(input, 107);
    success(owner.registry.destroy(input));
    expect(old->is_finalized && attribute(*old, 906).scalars == fixture::scalar(std::int64_t{7}),
           "immutable old semantic snapshot survives original reset and destruction");
    success(owner.registry.finalize_backend(operation));
    const auto destination = owner.create(17);
    const auto before = owner.snapshot(destination);
    const auto native_target = [&] {
        auto view = owner.registry.borrow_backend(destination);
        expect(view.has_value(), "destination exists");
        return view->object.native;
    }();
    const Header query{operation, 1610, 15, 1, 3};
    const auto prepared = owner.prepare(Operation::kGet, query);
    expect(prepared.direction == Direction::kDescriptorOutput && prepared.input_elements == 1 &&
               prepared.output_elements == 0,
           "descriptor getter captures existing target ID, never new handle output storage");
    {
        const auto held = owner.registry.borrow_backend(destination);
        const auto gets = fixture::state.gets;
        const auto deferred =
            owner.registry.get_backend_attribute(query, prepared.generation, {{}, {destination}});
        expect(held && deferred.result.disposition == CudaDnnDisposition::kDeferred &&
                   fixture::state.gets == gets,
               "external immutable target lease defers before original Get mutation");
    }
    const auto copied =
        owner.registry.get_backend_attribute(query, prepared.generation, {{}, {destination}});
    success(copied.result);
    expect(copied.values.descriptors.empty() && copied.values.scalars.empty() &&
               copied.count == 1 && fixture::state.last_destination == native_target,
           "Get copies into unchanged target identity without publishing reference tokens");
    const auto after = owner.snapshot(destination);
    expect(after->generation > before->generation && after->is_finalized && after->is_valid &&
               attribute(*after, 906).scalars == fixture::scalar(std::int64_t{7}),
           "mutable getter installs the original copied semantic state on target");
    success(owner.registry.destroy(operation));
    const auto uid = owner.get({destination, 906, 3, 1, 3});
    success(uid.result);
    expect(uid.values.scalars == fixture::scalar(std::int64_t{7}),
           "original copied target outlives parent and original source objects");
    owner.close();
}

void mutation_recovery() {
    Owner owner;
    const auto tensor = owner.create(17);
    const auto unaffected = owner.create(21);
    owner.scalar(tensor, 906, 3, std::int64_t{7});
    auto generation = owner.snapshot(tensor)->generation;
    fixture::control.set_status.store(CUDNN_STATUS_BAD_PARAM);
    const auto failed = owner.set({tensor, 906, 3, 1, 1}, {fixture::scalar(std::int64_t{99}), {}});
    native_error(failed.result, CUDNN_STATUS_BAD_PARAM);
    expect(!failed.is_configuration_valid && failed.generation > generation &&
               !owner.snapshot(tensor)->is_valid && owner.snapshot(unaffected)->is_valid &&
               !owner.registry.snapshot().is_quarantined,
           "partial Set dirties affected semantics but does not quarantine ownership");
    fixture::control.set_status.store(CUDNN_STATUS_SUCCESS);
    owner.scalar(tensor, 906, 3, std::int64_t{100});
    expect(!owner.snapshot(tensor)->is_valid,
           "later successful setter cannot erase unknown prior partial configuration");
    success(owner.registry.initialize_backend(tensor));
    expect(owner.snapshot(tensor)->is_valid && owner.snapshot(tensor)->attributes.empty(),
           "successful original reset restores fresh known semantics");
    owner.scalar(tensor, 906, 3, std::int64_t{11});
    fixture::control.finalize_status.store(CUDNN_STATUS_BAD_PARAM);
    native_error(owner.registry.finalize_backend(tensor), CUDNN_STATUS_BAD_PARAM);
    expect(!owner.snapshot(tensor)->is_valid && !owner.registry.snapshot().is_quarantined,
           "Finalize configuration failure remains recoverable");
    fixture::control.finalize_status.store(CUDNN_STATUS_SUCCESS);
    success(owner.registry.initialize_backend(tensor));
    owner.tensor(tensor, 7);
    const std::array<std::int64_t, 4> logging_values{71, -82, 93, INT64_MIN};
    const auto logging_bytes = fixture::bytes(std::span<const std::int64_t>{logging_values});
    const Header finalized_set{tensor, 906, 3, 4, 1};
    const auto prepared = owner.prepare(Operation::kSet, finalized_set);
    expect(prepared.input_elements == 4,
           "finalized scalar Set still captures the full positive native logging prefix");
    fixture::state.capture_set_logging = true;
    const auto logged = owner.registry.set_backend_attribute(finalized_set, prepared.generation,
                                                             {logging_bytes, {}});
    fixture::state.capture_set_logging = false;
    native_error(logged.result, CUDNN_STATUS_BAD_PARAM);
    expect(fixture::state.logged_set_bytes == logging_bytes && fixture::state.last_count == 4 &&
               !logged.is_configuration_valid && !owner.registry.snapshot().is_quarantined &&
               owner.snapshot(unaffected)->is_valid,
           "native pre-validation logging observes all four values without ownership quarantine");
    {
        const auto native = owner.registry.borrow_backend(tensor);
        expect(native && fixture::object(native->object.native).snapshot.attributes.at(906).bytes ==
                             fixture::scalar(std::int64_t{7}),
               "finalized rejection does not mutate the original scalar after logging it");
    }
    success(owner.registry.initialize_backend(tensor));
    owner.tensor(tensor, 7);
    expect(owner.snapshot(tensor)->is_valid,
           "original Initialize recovers finalized-Set configuration rejection");
    fixture::control.get_status.store(CUDNN_STATUS_INTERNAL_ERROR);
    const auto read = owner.get({tensor, 906, 3, 1, 3});
    native_error(read.result, CUDNN_STATUS_INTERNAL_ERROR);
    expect(read.values.scalars.empty() && !read.count_bytes && owner.snapshot(tensor)->is_valid,
           "failed scalar read publishes no private writes and does not dirty its source");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    const auto operation = owner.create(22);
    success(owner.set({operation, 1610, 15, 1, 1}, {{}, {tensor}}).result);
    success(owner.registry.finalize_backend(operation));
    const auto destination = owner.create(17);
    owner.tensor(destination, 53);
    expect(owner.snapshot(destination)->tensor.has_value(),
           "failed destination copy starts with an established observation");
    fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM);
    const auto mutation = owner.get({operation, 1610, 15, 1, 3}, {{}, {destination}});
    native_error(mutation.result, CUDNN_STATUS_BAD_PARAM);
    expect(!owner.snapshot(destination)->is_valid && owner.snapshot(operation)->is_valid &&
               !owner.snapshot(destination)->tensor &&
               !owner.snapshot(destination)->tensor_alignment && owner.snapshot(tensor)->is_valid &&
               mutation.values.descriptors.empty() && !mutation.count_bytes &&
               !owner.registry.snapshot().is_quarantined,
           "failed native descriptor copy dirties target only and suppresses count publication");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    success(owner.registry.initialize_backend(destination));
    expect(owner.snapshot(destination)->is_valid, "failed copied destination can reset safely");
    owner.close();
}

void generations_and_reentry() {
    Owner owner;
    const auto id = owner.create(17);
    const Header header{id, 906, 3, 1, 1};
    const auto old = owner.prepare(Operation::kSet, header);
    owner.scalar(id, 906, 3, std::int64_t{1});
    auto calls = fixture::state.sets;
    const auto stale = owner.registry.set_backend_attribute(header, old.generation,
                                                            {fixture::scalar(std::int64_t{2}), {}});
    expect(stale.result.status == CUDA_ERROR_INVALID_VALUE && !stale.result.original_status &&
               fixture::state.sets == calls,
           "stale preparation cannot change a later native generation");
    {
        const auto lease = owner.registry.borrow_backend(id);
        const auto prepared = owner.prepare(Operation::kSet, header);
        expect(
            lease &&
                owner.registry.initialize_backend(id).disposition ==
                    CudaDnnDisposition::kDeferred &&
                owner.registry.finalize_backend(id).disposition == CudaDnnDisposition::kDeferred &&
                owner.registry.destroy(id).disposition == CudaDnnDisposition::kDeferred &&
                owner.registry
                        .set_backend_attribute(header, prepared.generation,
                                               {fixture::scalar(std::int64_t{3}), {}})
                        .result.disposition == CudaDnnDisposition::kDeferred &&
                fixture::state.sets == calls,
            "immutable object lifetime defers every mutating metadata operation");
    }
    bool entered{};
    fixture::state.callback = [&] {
        entered = true;
        const auto native_calls = fixture::control.callbacks.load();
        expect(!owner.registry.borrow_backend(id) &&
                   owner.registry.create_backend(17).status == CUDA_ERROR_NOT_READY &&
                   owner.registry.initialize_backend(id).status == CUDA_ERROR_NOT_READY &&
                   owner.registry.destroy(id).status == CUDA_ERROR_NOT_READY &&
                   owner.registry.shutdown().disposition == CudaDnnDisposition::kDeferred &&
                   fixture::control.callbacks.load() == native_calls,
               "native callback reentry cannot borrow, mutate or destroy accepted owners");
    };
    owner.scalar(id, 906, 3, std::int64_t{4});
    fixture::state.callback = {};
    expect(entered, "reentry assertions ran inside a real native callback");
    const auto prepared = owner.prepare(Operation::kSet, header);
    fixture::state.callback = [&] { owner.registry.close(); };
    success(owner.registry
                .set_backend_attribute(header, prepared.generation,
                                       {fixture::scalar(std::int64_t{5}), {}})
                .result);
    fixture::state.callback = {};
    expect(fixture::state.sets == calls + 2 && !fixture::state.destroys &&
               !owner.registry.create_backend(17).original_status,
           "callback close is non-destructive and accepted operation finishes once");
    owner.close();
}

void admission_controls() {
    for (unsigned int missing = 1; missing <= 7; ++missing) {
        Owner owner(missing);
        if (missing <= 3) {
            const auto result = owner.registry.create_backend(17);
            expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                       !fixture::state.creates,
                   "missing version/create/destroy rejects before native creation");
        } else {
            const auto id = owner.create(17);
            CudaDnnResult result;
            if (missing == 4) result = owner.registry.initialize_backend(id);
            if (missing == 5) result = owner.registry.finalize_backend(id);
            if (missing == 6 || missing == 7)
                result =
                    owner.registry
                        .prepare_backend_attribute(missing == 6 ? Operation::kSet : Operation::kGet,
                                                   {id, 906, 3, 1, 1})
                        .result;
            expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status,
                   "missing callback is clean policy error without fake native status");
        }
        owner.close();
    }
    Owner owner;
    fixture::control.version.store(91003);
    expect(owner.registry.create_backend(17).status == CUDA_ERROR_NOT_SUPPORTED &&
               !fixture::state.creates,
           "unqualified native version rejects before creation");
    fixture::control.version.store(91002);
    const auto id = owner.create(17);
    const auto pointwise = owner.create(0);
    for (const auto count : {std::int64_t{-1}, std::int64_t{0}, INT64_MIN}) {
        const Header invalid_count{pointwise, 0, 14, count, 1};
        const auto ready = owner.prepare(Operation::kSet, invalid_count);
        expect(!ready.input_elements, "Pointwise count rejection precedes any caller bytes");
        const auto result =
            owner.registry.set_backend_attribute(invalid_count, ready.generation, {});
        native_error(result.result, CUDNN_STATUS_BAD_PARAM);
    }
    const auto reduction = owner.create(22);
    for (const auto count : {std::int64_t{0}, std::int64_t{-1}}) {
        const Header null_array{reduction, 1610, 15, count, 0};
        const auto ready = owner.prepare(Operation::kSet, null_array);
        expect(!ready.input_elements, "absent descriptor array captures no invented pointer");
        success(owner.registry.set_backend_attribute(null_array, ready.generation, {}).result);
        expect(!fixture::state.last_has_array && fixture::state.last_count == count,
               "safe native child-clear receives exact NULL array and original signed count");
    }
    for (const auto operation : {Operation::kSet, Operation::kGet}) {
        const auto result =
            owner.registry.prepare_backend_attribute(operation, {id, 913, 15, 1, 1});
        expect(result.result.status == CUDA_ERROR_NOT_SUPPORTED && !result.result.original_status,
               "unsafe ragged ABI cannot escape through either metadata direction");
    }
    const auto prepared = owner.prepare(Operation::kSet, {id, 906, 3, 1, 1});
    const auto native_sets = fixture::state.sets;
    for (const auto& values : {Values{{std::byte{1}}, {}}, Values{{}, {id}}}) {
        const auto invalid =
            owner.registry.set_backend_attribute({id, 906, 3, 1, 1}, prepared.generation, values);
        expect(!invalid.result.original_status && fixture::state.sets == native_sets,
               "wrong materialized class or byte span rejects before native dereference");
    }
    owner.close();
}

void uncertain_owner(unsigned int which) {
    Owner owner;
    if (which == 0) fixture::state.null_create = true;
    if (which == 1) {
        fixture::state.partial_create = true;
        fixture::state.create_status = CUDNN_STATUS_BAD_PARAM;
    }
    if (which == 2) {
        (void)owner.create(17);
        fixture::state.duplicate_create = true;
    }
    if (which < 3) {
        const auto result = owner.registry.create_backend(17);
        expect(result.disposition == CudaDnnDisposition::kRejected && !result.id &&
                   owner.registry.snapshot().is_quarantined,
               "null/partial/duplicate native creation retains uncertain ownership");
    } else {
        const auto id = owner.create(17);
        fixture::state.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
        const auto result = owner.registry.destroy(id);
        expect(result.original_status == CUDNN_STATUS_INTERNAL_ERROR &&
                   owner.registry.snapshot().is_quarantined,
               "uncertain destroy status remains distinct from configuration failure");
    }
    const auto destroys = fixture::state.destroys;
    owner.registry.close();
    (void)owner.registry.shutdown();
    (void)owner.registry.shutdown();
    expect(fixture::state.destroys == destroys && !owner.registry.snapshot().is_closed,
           "unknown native ownership never destroys twice or falsely closes");
}
}  // namespace

int main() {
    try {
        kinds_and_scalars();
        tensor_prefix_and_padding();
        native_tensor_observation();
        unavailable_tensor_observation();
        child_copy_and_mutable_get();
        mutation_recovery();
        generations_and_reentry();
        admission_controls();
        for (unsigned int which = 0; which != 4; ++which) {
            const auto child = fork();
            expect(child >= 0, "fork isolated uncertain owner");
            if (!child) {
                try {
                    uncertain_owner(which);
                    _exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "backend graph ownership case " << which << ": " << error.what()
                              << '\n';
                    _exit(1);
                }
            }
            int status{};
            expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                   "isolated graph uncertain ownership regression");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "backend graph registry: " << error.what() << '\n';
        return 1;
    }
}
