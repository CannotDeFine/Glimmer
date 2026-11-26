#pragma once

#include "dnn_library_info_fixture.h"

#include <chrono>
#include <functional>

// An independent, bounded CPU model of the audited metadata ABI. This is not
// vendor-runtime, sequence computation or GPU qualification.
namespace glimmer::tests::dnn_sequence {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace foundation = glimmer::tests::dnn_library_info;
using Kind = CudaDnnObjectKind;
struct Object {
    bool alive{};
    Kind kind{};
    unsigned int destroy_attempts{};
    CudaDnnRnnDataState rnn{};
    CudaDnnSeqDataState seq{.dimensions = 4, .lengths = {}};
};
struct State {
    std::array<Object, 32> objects{};
    unsigned int creates{}, destroys{}, rnn_sets{}, seq_sets{}, gets{};
    CudaDnnRnnDataState last_rnn{};
    CudaDnnSeqDataState last_seq{};
    CudaDnnSeqDataQuery last_query{};
    std::function<void()> callback;
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS};
    bool partial_create{}, null_create{}, duplicate_create{};
};
struct Control {
    std::atomic<unsigned int> callbacks{};
    std::atomic<int> set_status{CUDNN_STATUS_SUCCESS}, get_status{CUDNN_STATUS_SUCCESS};
    std::atomic<int> malformed{};
    std::atomic<bool> pause_set{}, entered{}, release{true};
};
inline State state;
inline Control control;
inline void reset() {
    foundation::reset();
    state = {};
    control.callbacks.store(0);
    control.set_status.store(CUDNN_STATUS_SUCCESS);
    control.get_status.store(CUDNN_STATUS_SUCCESS);
    control.malformed.store(0);
    control.pause_set.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline std::size_t padding_width(std::uint32_t type, bool sequence) {
    if (type == CUDNN_DATA_FLOAT) return 4;
    if (type == CUDNN_DATA_DOUBLE) return 8;
    if (type == CUDNN_DATA_HALF || (sequence && type == CUDNN_DATA_BFLOAT16)) return 2;
    return 0;
}
inline Object& object(void* native, Kind kind) {
    base::owner();
    for (auto& candidate : state.objects)
        if (&candidate == native) {
            base::require(candidate.alive && candidate.kind == kind);
            return candidate;
        }
    throw std::runtime_error("unknown original sequence descriptor");
}
inline Object* create(Kind kind) {
    base::owner();
    ++state.creates;
    if (state.null_create || (state.create_status != CUDNN_STATUS_SUCCESS && !state.partial_create))
        return nullptr;
    if (state.duplicate_create)
        for (auto& candidate : state.objects)
            if (candidate.alive && candidate.kind == kind) return &candidate;
    for (auto& candidate : state.objects)
        if (!candidate.alive) {
            candidate = {};
            candidate.alive = true;
            candidate.kind = kind;
            return &candidate;
        }
    throw std::runtime_error("sequence fixture capacity");
}
inline cudnnStatus_t destroy(void* native, Kind kind) {
    auto& candidate = object(native, kind);
    ++state.destroys;
    base::require(++candidate.destroy_attempts == 1);
    if (state.destroy_status == CUDNN_STATUS_SUCCESS) candidate.alive = false;
    return state.destroy_status;
}
inline void entered(bool setter) {
    ++control.callbacks;
    if (state.callback) state.callback();
    if (setter && control.pause_set.exchange(false)) {
        control.entered.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!control.release.load()) {
            base::require(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }
    }
}
inline CudaDnnDispatch dispatch() {
    auto sdk = base::dispatch();
    sdk.rnn_data_create = [](CudaDnnRnnData* output) {
        *output = reinterpret_cast<CudaDnnRnnData>(create(Kind::kRnnData));
        return state.create_status;
    };
    sdk.rnn_data_destroy = [](CudaDnnRnnData native) { return destroy(native, Kind::kRnnData); };
    sdk.seq_data_create = [](CudaDnnSeqData* output) {
        *output = reinterpret_cast<CudaDnnSeqData>(create(Kind::kSeqData));
        return state.create_status;
    };
    sdk.seq_data_destroy = [](CudaDnnSeqData native) { return destroy(native, Kind::kSeqData); };
    sdk.rnn_data_set = [](CudaDnnRnnData native, cudnnDataType_t type, cudnnRNNDataLayout_t layout,
                          int maximum, int batch, int width, const int* lengths, void* padding) {
        auto& entry = object(native, Kind::kRnnData);
        ++state.rnn_sets;
        entered(true);
        CudaDnnRnnDataState input{.type = static_cast<std::uint32_t>(type),
                                  .layout = static_cast<std::uint32_t>(layout),
                                  .presence = (lengths ? 1U : 0U) | (padding ? 2U : 0U),
                                  .max_sequence_length = maximum,
                                  .batch_size = batch,
                                  .vector_size = width,
                                  .lengths = {}};
        state.last_rnn = input;
        if (!lengths) return CUDNN_STATUS_BAD_PARAM;
        if (input.type > 16) return CUDNN_STATUS_BAD_PARAM;
        if (input.type > 2) return CUDNN_STATUS_NOT_SUPPORTED;
        if (input.layout > 2 || maximum <= 0 || batch <= 0 || width <= 0)
            return CUDNN_STATUS_BAD_PARAM;
        if (maximum > 65535) return CUDNN_STATUS_NOT_SUPPORTED;
        base::require(static_cast<std::size_t>(batch) <= kCudaDnnMaxSequenceLengths);
        input.lengths.assign(lengths, lengths + batch);
        for (const int length : input.lengths)
            if (length < 0 || length > maximum) return CUDNN_STATUS_BAD_PARAM;
        if (input.layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED &&
            (input.lengths.front() != maximum || input.lengths.back() <= 0 ||
             !std::is_sorted(input.lengths.begin(), input.lengths.end(), std::greater<int>{})))
            return CUDNN_STATUS_BAD_PARAM;
        input.padding_bytes = padding ? padding_width(input.type, false) : 0;
        if (padding) std::memcpy(input.padding.data(), padding, input.padding_bytes);
        state.last_rnn = input;
        // Deliberately mutate before an injected error; no rollback is promised.
        entry.rnn = input;
        if (control.set_status.load() != CUDNN_STATUS_SUCCESS) entry.rnn.vector_size = 777;
        return static_cast<cudnnStatus_t>(control.set_status.load());
    };
    sdk.seq_data_set = [](CudaDnnSeqData native, cudnnDataType_t type, int dimensions,
                          const int* sizes, const cudnnSeqDataAxis_t* axes, std::size_t count,
                          const int* lengths, void* padding) {
        auto& entry = object(native, Kind::kSeqData);
        ++state.seq_sets;
        entered(true);
        CudaDnnSeqDataState input{.type = static_cast<std::uint32_t>(type),
                                  .presence = (sizes ? 1U : 0U) | (axes ? 2U : 0U) |
                                              (lengths ? 4U : 0U) | (padding ? 8U : 0U),
                                  .dimensions = dimensions,
                                  .sequence_count = count,
                                  .lengths = {}};
        state.last_seq = input;
        if (input.type > 16 || dimensions <= 0) return CUDNN_STATUS_BAD_PARAM;
        if (dimensions != 4) return CUDNN_STATUS_NOT_SUPPORTED;
        base::require(sizes);
        std::copy_n(sizes, 4, input.sizes.begin());
        state.last_seq = input;
        for (const int size : input.sizes)
            if (size <= 0) return CUDNN_STATUS_BAD_PARAM;
        const auto actual = static_cast<std::uint64_t>(sizes[1]) * sizes[2];
        base::require(actual <= INT32_MAX);
        if (actual != count) return CUDNN_STATUS_BAD_PARAM;
        base::require(count <= kCudaDnnMaxSequenceLengths && lengths && axes);
        input.lengths.assign(lengths, lengths + count);
        for (const int length : input.lengths)
            if (length <= 0 || length > sizes[0]) return CUDNN_STATUS_BAD_PARAM;
        for (std::size_t i = 0; i != 4; ++i) input.axes[i] = static_cast<std::uint32_t>(axes[i]);
        auto sorted = input.axes;
        // Original 9.10.2 sorts only the first three axes and requires the
        // vector axis to remain last, not an arbitrary four-axis permutation.
        std::sort(sorted.begin(), sorted.begin() + 3);
        if (sorted != std::array<std::uint32_t, 4>{0, 1, 2, 3}) return CUDNN_STATUS_BAD_PARAM;
        entry.seq = input;  // Native fields change before late storage-type validation.
        if (!padding_width(input.type, true)) return CUDNN_STATUS_INTERNAL_ERROR;
        input.padding_bytes = padding ? padding_width(input.type, true) : 0;
        if (padding) std::memcpy(input.padding.data(), padding, input.padding_bytes);
        state.last_seq = input;
        entry.seq = input;
        if (control.set_status.load() != CUDNN_STATUS_SUCCESS) {
            entry.seq.sizes[0] = 777;
            entry.seq.lengths = {777};
        }
        return static_cast<cudnnStatus_t>(control.set_status.load());
    };
    sdk.seq_data_get = [](CudaDnnSeqData native, cudnnDataType_t* type, int* dimensions,
                          int requested, int* sizes, cudnnSeqDataAxis_t* axes, std::size_t* count,
                          std::size_t requested_lengths, int* lengths, void* padding) {
        const auto& entry = object(native, Kind::kSeqData).seq;
        ++state.gets;
        entered(false);
        base::require(type && dimensions && count);
        state.last_query = {requested, requested_lengths,
                            1U | 2U | 16U | (sizes ? 4U : 0U) | (axes ? 8U : 0U) |
                                (lengths ? 32U : 0U) | (padding ? 64U : 0U)};
        if (requested < 0) return CUDNN_STATUS_BAD_PARAM;
        const auto copied = std::min(requested, 4);
        *type = static_cast<cudnnDataType_t>(entry.type);
        *dimensions = 4;
        *count = entry.sequence_count;
        if (sizes) std::copy_n(entry.sizes.begin(), copied, sizes);
        if (axes)
            for (int i = 0; i != copied; ++i)
                axes[i] = static_cast<cudnnSeqDataAxis_t>(entry.axes[i]);
        if (lengths)
            std::copy_n(entry.lengths.begin(), std::min(requested_lengths, entry.lengths.size()),
                        lengths);
        if (padding) {
            if (entry.type > 2) return CUDNN_STATUS_INTERNAL_ERROR;
            std::memcpy(padding, entry.padding.data(), padding_width(entry.type, false));
        }
        // Corruption happens only after bounded writes, never by overrunning the fixture.
        const int malformed = control.malformed.load();
        if (malformed == 1) *dimensions = 3;
        if (malformed == 2) *dimensions = -1;
        if (malformed == 3) *count = kCudaDnnMaxSequenceLengths + 1;
        if (malformed == 4) {
            const std::uint32_t invalid = 32;
            std::memcpy(type, &invalid, 4);
        }
        if (malformed == 5 && axes && copied) {
            const std::uint32_t invalid = 4;
            std::memcpy(axes, &invalid, 4);
        }
        return static_cast<cudnnStatus_t>(control.get_status.load());
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
inline CudaDnnRnnDataState rnn(std::uint32_t type = CUDNN_DATA_FLOAT) {
    CudaDnnRnnDataState input{
        .type = type,
        .presence = 3,
        .padding_bytes = static_cast<std::uint32_t>(padding_width(type, false)),
        .max_sequence_length = 5,
        .batch_size = 3,
        .vector_size = 7,
        .lengths = {5, 3, 1}};
    const std::uint64_t bits = type == CUDNN_DATA_DOUBLE  ? 0x4004000000000000ULL
                               : type == CUDNN_DATA_FLOAT ? 0x40200000ULL
                                                          : 0x4100ULL;
    std::memcpy(input.padding.data(), &bits, input.padding_bytes);
    return input;
}
inline CudaDnnSeqDataState seq(std::uint32_t type = CUDNN_DATA_FLOAT) {
    CudaDnnSeqDataState input{
        .type = type,
        .presence = 15,
        .padding_bytes = static_cast<std::uint32_t>(padding_width(type, true)),
        .dimensions = 4,
        .sizes = {5, 2, 2, 7},
        .axes = {1, 0, 2, 3},
        .sequence_count = 4,
        .lengths = {5, 4, 3, 2}};
    const std::uint64_t bits = type == CUDNN_DATA_DOUBLE     ? 0x4004000000000000ULL
                               : type == CUDNN_DATA_FLOAT    ? 0x40200000ULL
                               : type == CUDNN_DATA_BFLOAT16 ? 0x4020ULL
                                                             : 0x4100ULL;
    std::memcpy(input.padding.data(), &bits, input.padding_bytes);
    return input;
}
}  // namespace glimmer::tests::dnn_sequence
