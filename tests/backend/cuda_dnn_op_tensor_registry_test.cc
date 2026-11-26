#include "dnn_op_tensor_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_op_tensor;
namespace base = fixture::base;
using Kind = CudaDnnObjectKind;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "original metadata success is preserved");
}
CudaDnnTensor tensor() {
    return {.type = CUDNN_DATA_DOUBLE,
            .format = CUDNN_TENSOR_NCHW,
            .dimensions = 4,
            .sizes = {2, 3, 4, 5},
            .strides = {60, 20, 5, 1}};
}
CudaDnnTensorTransformState transform() {
    return {.dimensions = 4,
            .format = CUDNN_TENSOR_NHWC,
            .direction = CUDNN_TRANSFORM_FOLD,
            .presence = 7,
            .before = {0, 0, 1, 2},
            .after = {0, 0, 2, 1},
            .fold = {1, 1}};
}
void ordinary() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto operation = registry.create(Kind::kOpTensor);
    const auto conversion = registry.create(Kind::kTensorTransform);
    success(operation);
    success(conversion);
    expect(!fixture::state.sets && !fixture::driver::control.context_creates.load(),
           "both creates preserve native defaults without hidden setter or CUDA context");
    const auto initial = registry.get_op_tensor(operation.id);
    success(initial);
    expect(initial.op_tensor.operation == CUDNN_OP_TENSOR_ADD &&
               initial.op_tensor.type == CUDNN_DATA_FLOAT &&
               initial.op_tensor.nan == CUDNN_NOT_PROPAGATE_NAN,
           "original OpTensor defaults are queried, not synthesized");
    success(registry.set_op_tensor(operation.id,
                                   {CUDNN_OP_TENSOR_MAX, CUDNN_DATA_DOUBLE, CUDNN_PROPAGATE_NAN}));
    for (unsigned int mask = 0; mask != 8; ++mask) {
        const auto result = registry.get_op_tensor(operation.id, mask);
        success(result);
        expect(result.op_tensor.operation ==
                       (mask & 1U ? static_cast<unsigned int>(CUDNN_OP_TENSOR_MAX) : 0U) &&
                   result.op_tensor.type ==
                       (mask & 2U ? static_cast<unsigned int>(CUDNN_DATA_DOUBLE) : 0U) &&
                   result.op_tensor.nan ==
                       (mask & 4U ? static_cast<unsigned int>(CUDNN_PROPAGATE_NAN) : 0U),
               "all eight optional-output masks preserve original null pointers");
    }
    fixture::control.get_status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    const auto failed = registry.get_op_tensor(operation.id);
    expect(failed.original_status == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE &&
               !failed.op_tensor.operation && !failed.op_tensor.type && !failed.op_tensor.nan &&
               !registry.snapshot().is_quarantined,
           "native getter error publishes none of its already-written outputs");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    fixture::control.invalid_output_mask.store(7);
    success(registry.get_op_tensor(operation.id, 0));
    fixture::control.invalid_output_mask.store(0);
    const auto before = fixture::state.sets;
    expect(
        registry.set_op_tensor(operation.id, {8, 0, 0}).status == CUDA_ERROR_INVALID_VALUE &&
            registry.set_op_tensor(operation.id, {0, 32, 0}).status == CUDA_ERROR_INVALID_VALUE &&
            registry.set_op_tensor(operation.id, {0, 0, 2}).status == CUDA_ERROR_INVALID_VALUE &&
            fixture::state.sets == before,
        "invalid enum representation rejects without invoking native setter");
    expect(
        registry.set_op_tensor(operation.id, {7, 0, 0}).original_status == CUDNN_STATUS_BAD_PARAM,
        "representable unsupported operation reaches original SDK validation");
    {
        const auto lease = registry.borrow(operation.id, Kind::kOpTensor);
        expect(lease &&
                   registry.set_op_tensor(operation.id, {}).disposition ==
                       CudaDnnDisposition::kDeferred &&
                   registry.destroy(operation.id).disposition == CudaDnnDisposition::kDeferred,
               "immutable OpTensor lease excludes mutation and destruction");
        success(registry.get_op_tensor(operation.id));
    }
    auto conversion_state = transform();
    conversion_state.before[0] = std::numeric_limits<std::int32_t>::min();
    conversion_state.after[3] = std::numeric_limits<std::int32_t>::max();
    conversion_state.fold[0] = 0x80000000U;
    conversion_state.fold[1] = 0xffffffffU;
    for (unsigned int mask = 0; mask != 8; ++mask) {
        conversion_state.presence = mask;
        success(registry.set_tensor_transform(conversion.id, conversion_state));
        const auto& seen = fixture::state.last_transform;
        expect(seen.presence == mask && seen.dimensions == 4 &&
                   seen.before[0] == (mask & 1U ? conversion_state.before[0] : 0) &&
                   seen.after[3] == (mask & 2U ? conversion_state.after[3] : 0) &&
                   seen.fold[0] == (mask & 4U ? 0x80000000U : 0U) &&
                   seen.fold[1] == (mask & 4U ? 0xffffffffU : 0U) && !seen.fold[2],
               "raw signed padding, full-width folding and independent array nullness survive");
    }
    for (unsigned int rank : {0U, 1U, 2U, 3U, 8U}) {
        conversion_state = {};
        conversion_state.dimensions = rank;
        const auto result = registry.set_tensor_transform(conversion.id, conversion_state);
        expect(
            result.original_status == (rank <= 2 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_SUCCESS) &&
                fixture::state.last_transform.dimensions == rank,
            "bounded ranks including zero reach native validation without clipping");
    }
    const auto sets = fixture::state.sets;
    for (unsigned int rank : {9U, 0xffffffffU}) {
        conversion_state.dimensions = rank;
        expect(registry.set_tensor_transform(conversion.id, conversion_state).status ==
                   CUDA_ERROR_INVALID_VALUE,
               "transport rank overflow rejects before accessing arrays");
    }
    expect(fixture::state.sets == sets, "invalid ranks execute no original setter");
    success(registry.set_tensor_transform(conversion.id, transform()));
    const auto source = registry.create(Kind::kTensor).id;
    const auto destination = registry.create(Kind::kTensor).id;
    success(registry.set_tensor(source, CudaDnnTensorSetter::kNd, tensor()));
    {
        const auto lease = registry.borrow(destination, Kind::kTensor);
        const auto calls = fixture::state.initializes;
        expect(lease &&
                   registry.init_transform_dest(conversion.id, source, destination).disposition ==
                       CudaDnnDisposition::kDeferred &&
                   fixture::state.initializes == calls,
               "destination immutable physical lease defers initialization before native mutation");
    }
    {
        const auto source_lease = registry.borrow(source, Kind::kTensor);
        const auto transform_lease = registry.borrow(conversion.id, Kind::kTensorTransform);
        expect(source_lease && transform_lease, "source and transform have immutable leases");
        expect(registry.set_tensor_transform(conversion.id, transform()).disposition ==
                       CudaDnnDisposition::kDeferred &&
                   registry.destroy(conversion.id).disposition == CudaDnnDisposition::kDeferred,
               "immutable transform lease excludes its own mutation and destruction");
        const auto initialized = registry.init_transform_dest(conversion.id, source, destination);
        success(initialized);
        expect(initialized.value == 2U * 3U * 7U * 8U * 8U,
               "packed destination reports exact native bytes");
    }
    const auto output = registry.get_tensor(destination, 4).tensor;
    expect(output.sizes == std::array<int, 8>{2, 3, 7, 8} &&
               output.strides == std::array<int, 8>{168, 1, 24, 3} &&
               registry.get_tensor(source, 4).tensor.sizes == tensor().sizes,
           "complete packed NHWC destination and immutable source match independent shape oracle");
    fixture::control.reported_bytes.store(std::numeric_limits<std::size_t>::max() - 7);
    expect(registry.init_transform_dest(conversion.id, source, destination).value ==
               std::numeric_limits<std::size_t>::max() - 7,
           "size result is not narrowed to 32 bits");
    expect(registry.init_transform_dest(conversion.id, source, destination, false).value == 0 &&
               !fixture::state.last_has_size,
           "absent size pointer reaches native and does not invent output");
    conversion_state = {.dimensions = 4};
    success(registry.set_tensor_transform(conversion.id, conversion_state));
    success(registry.init_transform_dest(conversion.id, source, source));
    expect(fixture::state.last_source == fixture::state.last_destination &&
               registry.get_tensor(source, 4).tensor.sizes == tensor().sizes,
           "source/destination alias reaches original unchanged without self-borrow deadlock");
    {
        const auto lease = registry.borrow(source, Kind::kTensor);
        expect(lease && registry.init_transform_dest(conversion.id, source, source).disposition ==
                            CudaDnnDisposition::kDeferred,
               "real external alias lease still excludes mutation");
    }
    expect(registry.init_transform_dest(operation.id, source, destination).status ==
                   CUDA_ERROR_INVALID_HANDLE &&
               registry.init_transform_dest(conversion.id, source, 999).status ==
                   CUDA_ERROR_INVALID_HANDLE &&
               registry.get_op_tensor(conversion.id).status == CUDA_ERROR_INVALID_HANDLE &&
               registry.get_op_tensor(operation.id, 8).status == CUDA_ERROR_INVALID_VALUE,
           "wrong-kind, stale and invalid masks reject before original callbacks");
    const auto foreign =
        std::async(std::launch::async, [&] { return registry.get_op_tensor(operation.id); }).get();
    expect(foreign.status == CUDA_ERROR_UNKNOWN, "foreign owner thread cannot invoke metadata");
    const auto child = ::fork();
    expect(child >= 0, "fork inherited registry authority check");
    if (!child)
        ::_exit(registry.get_op_tensor(operation.id).status == CUDA_ERROR_UNKNOWN &&
                        registry.init_transform_dest(conversion.id, source, destination).status ==
                            CUDA_ERROR_UNKNOWN
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "inherited process cannot reuse registry identity or invoke original metadata");
    success(registry.shutdown());
    expect(fixture::state.creates == 2 && fixture::state.destroys == 2 &&
               base::state.creates == 2 && base::state.destroys == 2 &&
               fixture::state.query_masks == 255 && fixture::state.array_masks == 255,
           "all metadata ownership closes exactly once after complete mask coverage");
}
void missing() {
    for (unsigned int missing = 0; missing != 8; ++missing) {
        fixture::reset();
        auto sdk = fixture::dispatch();
        if (missing == 0) sdk.op_tensor_create = nullptr;
        if (missing == 1) sdk.op_tensor_destroy = nullptr;
        if (missing == 2) sdk.op_tensor_set = nullptr;
        if (missing == 3) sdk.op_tensor_get = nullptr;
        if (missing == 4) sdk.tensor_transform_create = nullptr;
        if (missing == 5) sdk.tensor_transform_destroy = nullptr;
        if (missing == 6) sdk.tensor_transform_set = nullptr;
        if (missing == 7) sdk.init_transform_dest = nullptr;
        CudaDnnRegistry registry(sdk);
        const auto op = registry.create(Kind::kOpTensor);
        const auto conversion = registry.create(Kind::kTensorTransform);
        CudaDnnResult result;
        if (missing <= 1) result = op;
        if (missing == 2) result = registry.set_op_tensor(op.id, {});
        if (missing == 3) result = registry.get_op_tensor(op.id);
        if (missing == 4 || missing == 5) result = conversion;
        if (missing == 6) result = registry.set_tensor_transform(conversion.id, transform());
        if (missing == 7) {
            const auto source = registry.create(Kind::kTensor).id;
            result = registry.init_transform_dest(conversion.id, source, source);
        }
        expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                   !registry.snapshot().is_quarantined && !fixture::state.initializes,
               "missing callbacks reject cleanly before native access and do not quarantine");
        success(registry.shutdown());
    }
}
void isolated(unsigned int mode) {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto kind = mode & 1U ? Kind::kTensorTransform : Kind::kOpTensor;
    if (mode < 6) {
        if (mode < 2) {
            fixture::state.partial_create = true;
            fixture::state.create_status = CUDNN_STATUS_ALLOC_FAILED;
        } else if (mode < 4) {
            fixture::state.null_create = true;
        } else {
            success(registry.create(kind));
            fixture::state.duplicate_create = true;
        }
        expect(!registry.create(kind).id && registry.snapshot().is_quarantined,
               "partial, null and duplicate native ownership cannot publish metadata");
    } else if (mode == 6) {
        const auto op = registry.create(Kind::kOpTensor).id;
        fixture::control.invalid_output_mask.store(4);
        const auto malformed = registry.get_op_tensor(op);
        expect(malformed.original_status == CUDNN_STATUS_SUCCESS &&
                   malformed.status == CUDA_ERROR_UNKNOWN && !malformed.op_tensor.operation &&
                   !malformed.op_tensor.type && !malformed.op_tensor.nan &&
                   registry.snapshot().is_quarantined,
               "last malformed getter field suppresses all earlier fields transactionally");
    } else if (mode == 7) {
        const auto op = registry.create(Kind::kOpTensor).id;
        fixture::state.destroy_status = CUDNN_STATUS_INTERNAL_ERROR;
        expect(registry.destroy(op).original_status == CUDNN_STATUS_INTERNAL_ERROR,
               "uncertain destroy preserves original status");
    } else {
        const auto conversion = registry.create(Kind::kTensorTransform).id;
        const auto source = registry.create(Kind::kTensor).id;
        const auto destination = registry.create(Kind::kTensor).id;
        success(registry.set_tensor(source, CudaDnnTensorSetter::kNd, tensor()));
        success(registry.set_tensor_transform(conversion, transform()));
        const auto original = mode == 8   ? CUDNN_STATUS_BAD_PARAM
                              : mode == 9 ? CUDNN_STATUS_ALLOC_FAILED
                                          : CUDNN_STATUS_NOT_SUPPORTED;
        fixture::control.init_status.store(original);
        const auto failed = registry.init_transform_dest(conversion, source, destination);
        expect(
            failed.original_status == original && !failed.value &&
                registry.snapshot().is_quarantined &&
                !registry.borrow(destination, Kind::kTensor) &&
                base::object(fixture::state.last_destination, Kind::kTensor).tensor.sizes[0] == 777,
            "partially mutating original failure preserves error but suppresses size and reuse");
    }
    const auto attempts = fixture::state.destroys;
    expect(registry.shutdown().disposition != CudaDnnDisposition::kComplete &&
               registry.shutdown().disposition != CudaDnnDisposition::kComplete &&
               fixture::state.destroys == attempts && !registry.snapshot().is_closed,
           "ambiguous descriptor cleanup is not retried or falsely closed");
}
}  // namespace
int main() {
    try {
        ordinary();
        missing();
        for (unsigned int mode = 0; mode != 11; ++mode) {
            const auto pid = ::fork();
            expect(pid >= 0, "fork isolated quarantine test");
            if (!pid) {
                try {
                    isolated(mode);
                    ::_exit(0);
                } catch (const std::exception& error) {
                    std::cerr << "OpTensor isolated " << mode << ": " << error.what() << '\n';
                    ::_exit(1);
                }
            }
            int status{};
            expect(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
                   "isolated metadata ownership case passed");
        }
        std::cout << "OpTensor/Transform CPU registry checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
