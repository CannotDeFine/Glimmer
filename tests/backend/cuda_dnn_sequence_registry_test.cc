#include "dnn_sequence_fixture.h"

#include <future>
#include <iostream>
#include <sys/wait.h>

namespace {
using namespace glimmer::backend::cuda;
namespace fixture = glimmer::tests::dnn_sequence;
using Kind = CudaDnnObjectKind;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void success(const CudaDnnResult& result) {
    expect(result.disposition == CudaDnnDisposition::kComplete && result.status == CUDA_SUCCESS &&
               result.original_status == CUDNN_STATUS_SUCCESS,
           "sequence metadata preserves original success");
}
void verify(const CudaDnnResult& result, const CudaDnnSeqDataState& input,
            CudaDnnSeqDataQuery query) {
    success(result);
    const auto& output = result.seq_data;
    const auto dimensions = static_cast<std::size_t>(std::min(query.dimensions_requested, 4));
    const auto lengths = std::min(query.lengths_requested, input.sequence_count);
    expect(
        output.type == input.type && output.dimensions == 4 &&
            output.sequence_count == input.sequence_count &&
            output.presence == ((query.outputs & 4U ? 1U : 0U) | (query.outputs & 8U ? 2U : 0U) |
                                (query.outputs & 32U ? 4U : 0U) | (query.outputs & 64U ? 8U : 0U)),
        "private same-call scalar authority and requested pointer mask are exact");
    for (std::size_t index = 0; index != 4; ++index) {
        expect(output.sizes[index] ==
                       (query.outputs & 4U && index < dimensions ? input.sizes[index] : 0) &&
                   output.axes[index] ==
                       (query.outputs & 8U && index < dimensions ? input.axes[index] : 0),
               "getter arrays contain only requested bounded native prefixes");
    }
    expect(output.lengths.size() == (query.outputs & 32U ? lengths : 0),
           "getter captures min(raw requested, actual count), never SIZE_MAX allocation");
    expect(std::equal(output.lengths.begin(), output.lengths.end(), input.lengths.begin()),
           "complete requested sequence length values survive");
    const auto width = query.outputs & 64U ? fixture::padding_width(input.type, false) : 0;
    expect(output.padding_bytes == width &&
               std::equal(output.padding.begin(), output.padding.begin() + width,
                          input.padding.begin()),
           "same-call actual type determines exact padding output width");
}
void ordinary() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto rnn = registry.create(Kind::kRnnData).id;
    const auto seq = registry.create(Kind::kSeqData).id;
    expect(rnn && seq && !fixture::state.rnn_sets && !fixture::state.seq_sets,
           "original creates execute without hidden configuration");
    CudaDnnSeqDataState fresh{.dimensions = 4, .lengths = {}};
    verify(registry.get_seq_data(seq, {INT32_MAX, UINT64_MAX, 127}), fresh,
           {INT32_MAX, UINT64_MAX, 127});
    for (const auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF}) {
        auto input = fixture::rnn(type);
        for (unsigned int layout = 0; layout != 3; ++layout) {
            input.layout = layout;
            success(registry.set_rnn_data(rnn, input));
            expect(fixture::state.last_rnn.lengths == input.lengths &&
                       fixture::state.last_rnn.padding == input.padding &&
                       fixture::state.last_rnn.layout == layout,
                   "RNN lengths, layout and exact scalar byte snapshot survive");
        }
    }
    for (const auto type : {CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF}) {
        const auto input = fixture::seq(type);
        success(registry.set_seq_data(seq, input));
        for (unsigned int mask = 0; mask != 128; ++mask)
            for (const int requested : {0, 1, 3, 4, INT32_MAX})
                for (const std::uint64_t count :
                     std::array<std::uint64_t, 6>{0, 1, 3, 4, 5, UINT64_MAX}) {
                    const CudaDnnSeqDataQuery query{requested, count, mask};
                    verify(registry.get_seq_data(seq, query), input, query);
                    expect(
                        fixture::state.last_query.dimensions_requested == requested &&
                            fixture::state.last_query.lengths_requested == count &&
                            fixture::state.last_query.outputs == (mask | 1U | 2U | 16U),
                        "original getter receives raw capacities and independent nullable outputs");
                }
    }
    const auto bfloat = fixture::seq(CUDNN_DATA_BFLOAT16);
    success(registry.set_seq_data(seq, bfloat));
    verify(registry.get_seq_data(seq, {4, 4, 63}), bfloat, {4, 4, 63});
    const auto bf_error = registry.get_seq_data(seq, {4, 4, 127});
    expect(bf_error.original_status == CUDNN_STATUS_INTERNAL_ERROR &&
               bf_error.seq_data.lengths.empty() && !bf_error.seq_data.dimensions &&
               !registry.snapshot().is_quarantined,
           "BF16 native getter failure publishes none of its already-written private outputs");
    success(registry.set_seq_data(seq, fixture::seq()));
    fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH);
    const auto error = registry.get_seq_data(seq, {4, 4, 127});
    expect(error.original_status == CUDNN_STATUS_BAD_PARAM_SHAPE_MISMATCH &&
               error.seq_data.lengths.empty() && !error.seq_data.dimensions &&
               !registry.snapshot().is_quarantined,
           "detailed native getter error does not quarantine or publish output");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    const auto before = fixture::control.callbacks.load();
    auto invalid = fixture::seq();
    invalid.sizes[1] = INT32_MAX;
    invalid.sizes[2] = 2;
    expect(registry.set_seq_data(seq, invalid).status == CUDA_ERROR_INVALID_VALUE,
           "native signed multiplication overflow is rejected before issue");
    invalid = fixture::seq();
    invalid.sizes[1] = static_cast<int>(kCudaDnnMaxSequenceLengths + 1);
    invalid.sizes[2] = 1;
    invalid.sequence_count = kCudaDnnMaxSequenceLengths + 1;
    expect(registry.set_seq_data(seq, invalid).status == CUDA_ERROR_INVALID_VALUE,
           "healthy stored-count quota cannot be exceeded");
    invalid = fixture::seq();
    invalid.lengths.pop_back();
    expect(registry.set_seq_data(seq, invalid).status == CUDA_ERROR_INVALID_VALUE,
           "partial length snapshots cannot reach native reads");
    invalid = fixture::seq();
    invalid.axes[2] = 4;
    expect(registry.set_seq_data(seq, invalid).status == CUDA_ERROR_INVALID_VALUE,
           "unrepresentable raw axes cannot become SDK enum objects");
    expect(registry.get_seq_data(seq, {4, 4, 128}).status == CUDA_ERROR_INVALID_VALUE &&
               fixture::control.callbacks.load() == before && !registry.snapshot().is_quarantined,
           "clean policy failures neither issue nor poison metadata");
    for (const auto id : {rnn, seq}) {
        const auto kind = id == rnn ? Kind::kRnnData : Kind::kSeqData;
        const auto held = registry.borrow(id, kind);
        expect(held.has_value(), "borrow exact original sequence identity");
        const auto calls = fixture::control.callbacks.load();
        const auto set = id == rnn ? registry.set_rnn_data(id, fixture::rnn())
                                   : registry.set_seq_data(id, fixture::seq());
        expect(set.disposition == CudaDnnDisposition::kDeferred &&
                   registry.destroy(id).disposition == CudaDnnDisposition::kDeferred &&
                   calls == fixture::control.callbacks.load(),
               "immutable borrower defers mutation and destruction before any native call");
    }
    expect(registry.set_seq_data(rnn, fixture::seq()).status == CUDA_ERROR_INVALID_HANDLE &&
               registry.set_rnn_data(seq, fixture::rnn()).status == CUDA_ERROR_INVALID_HANDLE,
           "descriptor kinds cannot be confused");
    const auto foreign = std::async(std::launch::async, [&] {
                             return registry.get_seq_data(seq, {4, 4, 127});
                         }).get();
    expect(foreign.status == CUDA_ERROR_UNKNOWN, "foreign thread cannot invoke native metadata");
    const auto child = ::fork();
    expect(child >= 0, "fork inherited metadata owner");
    if (!child) ::_exit(registry.get_seq_data(seq, {}).status == CUDA_ERROR_UNKNOWN ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork never reuses inherited registry authority");
    unsigned int reentries{};
    fixture::state.callback = [&] {
        ++reentries;
        expect(registry.get_seq_data(seq, {}).status == CUDA_ERROR_NOT_READY &&
                   registry.set_seq_data(seq, fixture::seq()).status == CUDA_ERROR_NOT_READY &&
                   registry.set_rnn_data(rnn, fixture::rnn()).status == CUDA_ERROR_NOT_READY &&
                   registry.destroy(seq).status == CUDA_ERROR_NOT_READY &&
                   registry.create(Kind::kSeqData).status == CUDA_ERROR_NOT_READY &&
                   !registry.borrow(seq, Kind::kSeqData) &&
                   registry.shutdown().disposition == CudaDnnDisposition::kDeferred,
               "scoped native callback cannot recursively mutate, borrow or retire owners");
    };
    success(registry.set_seq_data(seq, fixture::seq()));
    success(registry.set_rnn_data(rnn, fixture::rnn()));
    success(registry.get_seq_data(seq, {4, 4, 127}));
    fixture::state.callback = {};
    expect(reentries == 3, "guard spans both setters and getter validation");
    success(registry.shutdown());
    expect(fixture::state.creates == 2 && fixture::state.destroys == 2 &&
               fixture::foundation::is_resource_free(),
           "all original metadata retires exactly once without Driver calls");
}
void close_during_call() {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto seq = registry.create(Kind::kSeqData).id;
    fixture::state.callback = [&] {
        registry.close();
        expect(!fixture::state.destroys && registry.snapshot().is_closing,
               "close records intent without deleting an active original");
    };
    success(registry.set_seq_data(seq, fixture::seq()));
    fixture::state.callback = {};
    expect(registry.set_seq_data(seq, fixture::seq()).disposition == CudaDnnDisposition::kRejected,
           "already-admitted call finishes but subsequent mutation rejects closing owner");
    success(registry.shutdown());
    expect(fixture::state.destroys == 1, "closed original retires once after callback returns");
}
void missing() {
    for (unsigned int missing = 0; missing != 7; ++missing) {
        fixture::reset();
        auto sdk = fixture::dispatch();
        if (missing == 0) sdk.rnn_data_create = nullptr;
        if (missing == 1) sdk.rnn_data_destroy = nullptr;
        if (missing == 2) sdk.rnn_data_set = nullptr;
        if (missing == 3) sdk.seq_data_create = nullptr;
        if (missing == 4) sdk.seq_data_destroy = nullptr;
        if (missing == 5) sdk.seq_data_set = nullptr;
        if (missing == 6) sdk.seq_data_get = nullptr;
        CudaDnnRegistry registry(sdk);
        const auto rnn = registry.create(Kind::kRnnData), seq = registry.create(Kind::kSeqData);
        auto result = missing < 2 ? rnn : seq;
        if (missing == 2) result = registry.set_rnn_data(rnn.id, fixture::rnn());
        if (missing == 5) result = registry.set_seq_data(seq.id, fixture::seq());
        if (missing == 6) result = registry.get_seq_data(seq.id, {});
        expect(result.status == CUDA_ERROR_NOT_SUPPORTED && !result.original_status &&
                   !registry.snapshot().is_quarantined,
               "missing capability is clean, with no fabricated native status");
        success(registry.shutdown());
    }
}
void poison(unsigned int which) {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto kind = which == 0 ? Kind::kRnnData : Kind::kSeqData;
    const auto id = registry.create(kind).id;
    if (kind == Kind::kRnnData)
        success(registry.set_rnn_data(id, fixture::rnn()));
    else
        success(registry.set_seq_data(id, fixture::seq()));
    CudaDnnResult result;
    if (which <= 1) {
        fixture::control.set_status.store(which ? CUDNN_STATUS_ALLOC_FAILED
                                                : CUDNN_STATUS_BAD_PARAM);
        result = which ? registry.set_seq_data(id, fixture::seq())
                       : registry.set_rnn_data(id, fixture::rnn());
        expect(result.original_status == fixture::control.set_status.load(),
               "issued partial setter error keeps exact original status");
        const auto lease = &fixture::state.objects[0];
        expect(which ? lease->seq.sizes[0] == 777 : lease->rnn.vector_size == 777,
               "error fixture actually changed metadata before reporting failure");
    } else if (which == 7 || which == 8) {
        auto input = fixture::seq();
        if (which == 7)
            input.axes = {3, 2, 1, 0};
        else {
            input.type = CUDNN_DATA_INT8;
            input.presence = 7;
            input.padding_bytes = 0;
            input.padding = {};
        }
        result = registry.set_seq_data(id, input);
        expect(
            result.original_status ==
                (which == 7 ? CUDNN_STATUS_BAD_PARAM : CUDNN_STATUS_INTERNAL_ERROR),
            "invalid vector-axis position and late unsupported storage keep exact native errors");
        if (which == 8)
            expect(fixture::state.objects[0].seq.type == CUDNN_DATA_INT8,
                   "unsupported storage error follows real partial descriptor mutation");
    } else {
        fixture::control.malformed.store(static_cast<int>(which - 1));
        result = registry.get_seq_data(id, {4, 4, 127});
        expect(result.original_status == CUDNN_STATUS_SUCCESS &&
                   result.disposition == CudaDnnDisposition::kRejected &&
                   result.seq_data.lengths.empty(),
               "malformed native success preserves provenance without publishing values");
    }
    const auto calls = fixture::control.callbacks.load();
    expect(registry.snapshot().is_quarantined && !registry.borrow(id, kind) &&
               registry.destroy(id).disposition == CudaDnnDisposition::kRejected &&
               registry.get_seq_data(id, {}).disposition == CudaDnnDisposition::kRejected &&
               calls == fixture::control.callbacks.load() && !fixture::state.destroys,
           "uncertain metadata never replays or destroys partially changed originals");
    ::_exit(0);
}
void ownership_failure(unsigned int which) {
    fixture::reset();
    CudaDnnRegistry registry(fixture::dispatch());
    const auto kind = which % 2 ? Kind::kSeqData : Kind::kRnnData;
    if (which < 2) {
        fixture::state.partial_create = true;
        fixture::state.create_status = CUDNN_STATUS_ALLOC_FAILED;
    } else if (which < 4) {
        fixture::state.null_create = true;
    } else {
        success(registry.create(kind));
        if (which < 6)
            fixture::state.duplicate_create = true;
        else
            fixture::state.destroy_status = CUDNN_STATUS_BAD_PARAM;
    }
    const auto result = which < 6 ? registry.create(kind) : registry.destroy(1);
    expect(registry.snapshot().is_quarantined &&
               result.original_status == (which < 2   ? CUDNN_STATUS_ALLOC_FAILED
                                          : which < 6 ? CUDNN_STATUS_SUCCESS
                                                      : CUDNN_STATUS_BAD_PARAM),
           "partial, null, duplicate creation and destroy failure retain exact native provenance");
    const auto destroys = fixture::state.destroys;
    expect(registry.shutdown().disposition != CudaDnnDisposition::kComplete &&
               fixture::state.destroys == destroys && destroys == (which >= 6 ? 1U : 0U),
           "uncertain native lifetime is never retried or destroyed twice");
    ::_exit(0);
}
}  // namespace
int main() {
    try {
        ordinary();
        close_during_call();
        missing();
        for (unsigned int which = 0; which != 9; ++which) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated sequence poison case");
            if (!child) poison(which);
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "isolated sequence poison case passes");
        }
        for (unsigned int which = 0; which != 8; ++which) {
            const auto child = ::fork();
            expect(child >= 0, "fork isolated sequence ownership failure");
            if (!child) ownership_failure(which);
            int status{};
            expect(
                ::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
                "isolated sequence ownership failure passes");
        }
        std::cout << "Sequence registry lifecycle and full optional-output matrix passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
