#pragma once

#include "dnn_attn_weights_fixture.h"
#include "../interceptor/attention_oracle.h"

// CPU-only ABI and ownership fixture. Parameter packing and reserve contents
// below are synthetic and are never a model of cuDNN's opaque representation.
namespace glimmer::tests::dnn_attn_compute {
using namespace backend::cuda;
namespace layout = glimmer::tests::dnn_attn_weights;
namespace query = layout::query;
namespace foundation = layout::foundation;
namespace base = layout::base;
namespace metadata = layout::metadata;
namespace dropout = layout::dropout;
namespace sequence = query::sequence;
namespace oracle = glimmer::tests::attention_oracle;
using Kind = CudaDnnAttnComputeKind;
inline constexpr std::size_t kWeights = 24, kReserve = 16;
inline constexpr std::array<std::array<std::uint32_t, 4>, 6> kAxes{
    {{0, 1, 2, 3}, {0, 2, 1, 3}, {1, 0, 2, 3}, {1, 2, 0, 3}, {2, 0, 1, 3}, {2, 1, 0, 3}}};
struct Control {
    std::array<std::atomic<unsigned int>, 3> calls{};
    std::atomic<unsigned int> gets{}, seq_gets{}, length_gets{}, records{}, polls{}, syncs{},
        stream_gets{}, context_sets{}, context_gets{}, partial_writes{};
    std::atomic<int> native_status{CUDNN_STATUS_SUCCESS}, get_status{CUDNN_STATUS_SUCCESS},
        seq_status{CUDNN_STATUS_SUCCESS}, sync_status{CUDA_SUCCESS},
        stream_status{CUDNN_STATUS_SUCCESS}, event_status{CUDA_SUCCESS}, corruption{},
        context_loss{};
    std::atomic<bool> arithmetic{true}, stream_change{}, partial_write{}, pause{}, entered{},
        release{true};
};
struct Invocation {
    Kind kind{};
    cudnnHandle_t handle{};
    std::array<void*, 5> descriptors{};
    std::array<const void*, 15> buffers{};
    std::array<std::size_t, 3> bytes{};
    std::uint32_t mode{};
    int current{};
    std::array<const int*, 2> windows{};
};
struct State {
    Invocation last;
    std::array<std::vector<int>, 2> observed_windows;
    std::function<void(unsigned int)> callback;
    bool issued{};
};
inline Control control;
inline State state;
inline void reset() {
    layout::reset();
    state = {};
    for (auto& value : control.calls) value.store(0);
    control.gets.store(0);
    control.seq_gets.store(0);
    control.length_gets.store(0);
    control.records.store(0);
    control.polls.store(0);
    control.syncs.store(0);
    control.stream_gets.store(0);
    control.context_sets.store(0);
    control.context_gets.store(0);
    control.partial_writes.store(0);
    control.native_status.store(CUDNN_STATUS_SUCCESS);
    control.get_status.store(CUDNN_STATUS_SUCCESS);
    control.seq_status.store(CUDNN_STATUS_SUCCESS);
    control.sync_status.store(CUDA_SUCCESS);
    control.stream_status.store(CUDNN_STATUS_SUCCESS);
    control.event_status.store(CUDA_SUCCESS);
    control.corruption.store(0);
    control.context_loss.store(0);
    control.arithmetic.store(true);
    control.stream_change.store(false);
    control.partial_write.store(false);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline void observe(unsigned int phase) {
    if (state.callback) state.callback(phase);
    if (phase == 3 && control.pause.exchange(false)) {
        control.entered.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!control.release.load()) {
            base::require(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }
    }
    if (control.context_loss.load() == static_cast<int>(phase))
        foundation::driver.current = nullptr;
}
inline std::size_t elements(const CudaDnnSeqDataState& data) {
    std::size_t result = 1;
    for (const auto size : data.sizes) {
        base::require(size > 0);
        result *= static_cast<std::size_t>(size);
    }
    return result;
}
inline std::size_t index(const CudaDnnSeqDataState& data, std::size_t t, std::size_t b,
                         std::size_t beam, std::size_t component) {
    const std::array coordinates{t, b, beam, component};
    std::size_t result{};
    for (const auto axis : data.axes)
        result = result * static_cast<std::size_t>(data.sizes[axis]) + coordinates[axis];
    return result;
}
inline oracle::Inputs inputs(std::size_t b, std::size_t beam) {
    auto value = oracle::seeded_inputs();
    const auto shift = .0625 * static_cast<double>(2 * b + beam);
    for (auto& x : value.queries) x += shift;
    for (auto& x : value.keys) x -= shift;
    for (auto& x : value.values) x += 2 * shift;
    for (auto& x : value.residual) x -= shift;
    value.has_residual = true;
    return value;
}
inline oracle::Upstream upstream(std::size_t b, std::size_t beam) {
    auto value = oracle::seeded_upstream();
    for (auto& x : value.out) x += .03125 * static_cast<double>(2 * b + beam);
    return value;
}
inline std::array<double, kWeights> flatten(const oracle::Parameters& value) {
    std::array<double, kWeights> result{};
    for (std::size_t kind = 0; kind != 4; ++kind) {
        std::copy(value.matrices[kind].begin(), value.matrices[kind].end(),
                  result.begin() + 6 * kind);
        std::copy(value.biases[kind].begin(), value.biases[kind].end(),
                  result.begin() + 6 * kind + 4);
    }
    return result;
}
template <typename Scalar>
inline void compute(const Invocation& call) {
    const auto& parent = metadata::object(call.descriptors[0], CudaDnnObjectKind::kAttn).attn;
    std::array<CudaDnnSeqDataState, 4> data;
    for (std::size_t i = 0; i != 4; ++i)
        data[i] = sequence::object(call.descriptors[i + 1], CudaDnnObjectKind::kSeqData).seq;
    const auto in = [&](std::size_t slot) {
        return static_cast<const Scalar*>(call.buffers[slot]);
    };
    const auto out = [&](std::size_t slot) { return const_cast<Scalar*>(in(slot)); };
    oracle::Parameters parameters;
    for (std::size_t kind = 0; kind != 4; ++kind) {
        for (std::size_t i = 0; i != 4; ++i) parameters.matrices[kind][i] = in(11)[6 * kind + i];
        for (std::size_t i = 0; i != 2; ++i) parameters.biases[kind][i] = in(11)[6 * kind + 4 + i];
    }
    auto* reserve = out(14);
    if (call.kind == Kind::kForward && reserve) {
        reserve[0] = 17;
        for (std::size_t t = 0; t != 2; ++t) {
            reserve[1 + t] = static_cast<Scalar>(call.windows[0][t]);
            reserve[3 + t] = static_cast<Scalar>(call.windows[1][t]);
        }
    }
    if (call.kind != Kind::kForward)
        base::require(reserve && (reserve[0] == 17 || reserve[0] == 19));
    if (call.kind == Kind::kBackwardWeights && call.mode == CUDNN_WGRAD_MODE_SET)
        std::fill_n(out(12), kWeights, Scalar{});
    for (int b = 0; b != data[0].sizes[1]; ++b)
        for (int beam = 0; beam != data[0].sizes[2]; ++beam) {
            oracle::Inputs input;
            input.scale = std::bit_cast<double>(parent.scale_bits);
            input.has_residual = call.buffers[5] != nullptr;
            for (std::size_t t = 0; t != 2; ++t) {
                input.lower[t] = call.kind == Kind::kBackwardWeights
                                     ? static_cast<int>(reserve[1 + t])
                                     : call.windows[0][t];
                input.upper[t] = call.kind == Kind::kBackwardWeights
                                     ? static_cast<int>(reserve[3 + t])
                                     : call.windows[1][t];
            }
            for (std::size_t t = 0; t != 2; ++t)
                for (std::size_t c = 0; c != 2; ++c) {
                    input.queries[2 * t + c] = in(2)[index(data[0], t, b, beam, c)];
                    if (input.has_residual)
                        input.residual[2 * t + c] = in(5)[index(data[0], t, b, beam, c)];
                }
            for (std::size_t t = 0; t != 3; ++t)
                for (std::size_t c = 0; c != 2; ++c) {
                    input.keys[2 * t + c] = in(3)[index(data[1], t, b, beam, c)];
                    input.values[2 * t + c] = in(4)[index(data[2], t, b, beam, c)];
                }
            if (call.kind == Kind::kForward) {
                const auto value = oracle::forward(parameters, input);
                for (std::size_t t = 0; t != 2; ++t)
                    for (std::size_t c = 0; c != 2; ++c)
                        if (call.current < 0 || call.current == static_cast<int>(t))
                            out(6)[index(data[3], t, b, beam, c)] =
                                static_cast<Scalar>(value.out[2 * t + c]);
            } else {
                oracle::Upstream derivative;
                for (std::size_t t = 0; t != 2; ++t)
                    for (std::size_t c = 0; c != 2; ++c)
                        derivative.out[2 * t + c] = in(7)[index(data[3], t, b, beam, c)];
                const auto value = oracle::backward(parameters, input, derivative);
                if (call.kind == Kind::kBackwardData) {
                    for (std::size_t t = 0; t != 2; ++t)
                        for (std::size_t c = 0; c != 2; ++c)
                            out(8)[index(data[0], t, b, beam, c)] =
                                static_cast<Scalar>(value.queries[2 * t + c]);
                    for (std::size_t t = 0; t != 3; ++t)
                        for (std::size_t c = 0; c != 2; ++c) {
                            out(9)[index(data[1], t, b, beam, c)] =
                                static_cast<Scalar>(value.keys[2 * t + c]);
                            out(10)[index(data[2], t, b, beam, c)] =
                                static_cast<Scalar>(value.values[2 * t + c]);
                        }
                } else {
                    const auto gradient = flatten(value.parameters);
                    for (std::size_t i = 0; i != kWeights; ++i)
                        out(12)[i] += static_cast<Scalar>(gradient[i]);
                }
            }
        }
    if (call.kind == Kind::kBackwardData) reserve[0] = 19;
}
inline cudnnStatus_t invoke(Invocation call) {
    query::parent(call.handle, call.descriptors[0], CudaDnnObjectKind::kAttn);
    ++control.calls[static_cast<std::size_t>(call.kind) - 1];
    state.last = call;
    state.issued = true;
    observe(3);
    if (control.stream_change.load())
        base::handle(call.handle).stream = reinterpret_cast<cudaStream_t>(0x4240);
    for (std::size_t i = 1; i != 5; ++i)
        if (!call.descriptors[i]) return CUDNN_STATUS_BAD_PARAM;
    for (const auto slot : {2U, 3U, 4U})
        if (!call.buffers[slot]) return CUDNN_STATUS_BAD_PARAM;
    for (std::size_t i = 0; i != 3; ++i)
        if (call.bytes[i] && !call.buffers[i == 0 ? 11 : i + 12]) return CUDNN_STATUS_BAD_PARAM;
    if (call.kind == Kind::kBackwardWeights &&
        (!call.buffers[7] || !call.buffers[12] || !call.buffers[14]))
        return CUDNN_STATUS_BAD_PARAM;
    if (call.kind != Kind::kBackwardWeights) {
        if (!call.windows[0] || !call.windows[1] || !call.buffers[0] || !call.buffers[1])
            return CUDNN_STATUS_BAD_PARAM;
        if (call.kind == Kind::kForward &&
            (!call.buffers[6] || (call.current >= 0 && call.buffers[14])))
            return CUDNN_STATUS_BAD_PARAM;
        if (call.kind == Kind::kBackwardData &&
            (!call.buffers[7] || !call.buffers[8] || !call.buffers[9] || !call.buffers[10] ||
             !call.buffers[14] || !call.bytes[2]))
            return CUDNN_STATUS_BAD_PARAM;
        const auto& q = sequence::object(call.descriptors[1], CudaDnnObjectKind::kSeqData).seq;
        if (call.kind == Kind::kForward && call.current >= q.sizes[0])
            return CUDNN_STATUS_BAD_PARAM;
        if (call.kind == Kind::kForward && call.current >= 0 &&
            call.current >= *std::max_element(q.lengths.begin(), q.lengths.end()))
            return CUDNN_STATUS_SUCCESS;
        const auto count =
            call.current >= 0 && call.kind == Kind::kForward
                ? 1
                : metadata::object(call.descriptors[0], CudaDnnObjectKind::kAttn).attn.sizes[7];
        const auto first = call.current >= 0 && call.kind == Kind::kForward ? call.current : 0;
        for (std::size_t i = 0; i != 2; ++i)
            state.observed_windows[i].assign(call.windows[i] + first,
                                             call.windows[i] + first + count);
    }
    if (control.native_status.load() != CUDNN_STATUS_SUCCESS) {
        if (control.partial_write.load()) {
            // Deliberately model an issued operation that writes before failing.
            // This is a lifecycle probe, not a cuDNN partial-output value oracle.
            const auto slot = call.kind == Kind::kForward        ? 6
                              : call.kind == Kind::kBackwardData ? 8
                                                                 : 12;
            base::require(call.buffers[slot]);
            *static_cast<std::byte*>(const_cast<void*>(call.buffers[slot])) = std::byte{0x67};
            ++control.partial_writes;
        }
        return static_cast<cudnnStatus_t>(control.native_status.load());
    }
    if (control.arithmetic.load()) {
        const auto type =
            metadata::object(call.descriptors[0], CudaDnnObjectKind::kAttn).attn.enums[0];
        if (type == CUDNN_DATA_DOUBLE)
            compute<double>(call);
        else
            compute<float>(call);
    }
    return CUDNN_STATUS_SUCCESS;
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto sdk = layout::dispatch();
    sdk.version = [] {
        const auto version = layout::dispatch().version();
        observe(6);
        return version;
    };
    const auto sequences = sequence::dispatch();
    sdk.seq_data_create = sequences.seq_data_create;
    sdk.seq_data_destroy = sequences.seq_data_destroy;
    sdk.seq_data_set = sequences.seq_data_set;
    sdk.attn_get = [](CudaDnnAttn native, unsigned int* mode, int* heads, double* scale,
                      cudnnDataType_t* type, cudnnDataType_t* precision, cudnnMathType_t* math,
                      CudaDnnDropout* first, CudaDnnDropout* second, int* q, int* k, int* v,
                      int* qp, int* kp, int* vp, int* op, int* maxq, int* maxk, int* batch,
                      int* beam) {
        ++control.gets;
        const auto& object = metadata::object(native, CudaDnnObjectKind::kAttn).attn;
        for (const auto dependency : object.dropouts) metadata::live(dependency);
        base::require(type && maxq && !mode && !heads && !scale && !precision && !math && !first &&
                      !second && !q && !k && !v && !qp && !kp && !vp && !op && !maxk && !batch &&
                      !beam);
        *type = static_cast<cudnnDataType_t>(object.enums[0]);
        *maxq = object.sizes[7];
        if (control.corruption.load() == 1) {
            const std::uint32_t bits = 32;
            std::memcpy(type, &bits, 4);
        }
        observe(1);
        return static_cast<cudnnStatus_t>(control.get_status.load());
    };
    sdk.seq_data_get = [](CudaDnnSeqData native, cudnnDataType_t* type, int* dimensions,
                          int requested, int* sizes, cudnnSeqDataAxis_t* axes, std::size_t* count,
                          std::size_t capacity, int* lengths, void* padding) {
        ++control.seq_gets;
        control.length_gets += capacity != 0;
        base::require(type && dimensions && sizes && axes && count && requested == 4 && !padding);
        const auto& value = sequence::object(native, CudaDnnObjectKind::kSeqData).seq;
        *type = static_cast<cudnnDataType_t>(value.type);
        *dimensions = value.dimensions;
        *count = value.sequence_count;
        std::copy(value.sizes.begin(), value.sizes.end(), sizes);
        for (std::size_t i = 0; i != 4; ++i)
            axes[i] = static_cast<cudnnSeqDataAxis_t>(value.axes[i]);
        base::require(capacity <= value.lengths.size() && (!capacity || lengths));
        if (capacity) std::copy_n(value.lengths.begin(), capacity, lengths);
        if (control.corruption.load() == 2) *dimensions = 5;
        if (control.corruption.load() == 3) {
            const std::uint32_t bits = 4;
            std::memcpy(axes, &bits, 4);
        }
        if (control.corruption.load() == 4) ++*count;
        if (control.corruption.load() == 5 && capacity) ++sizes[0];
        observe(2);
        return static_cast<cudnnStatus_t>(control.seq_status.load());
    };
    sdk.get_stream = [](cudnnHandle_t handle, cudaStream_t* stream) {
        ++control.stream_gets;
        const auto status = base::dispatch().get_stream(handle, stream);
        if (!state.issued) return status;
        observe(4);
        return static_cast<cudnnStatus_t>(control.stream_status.load());
    };
    sdk.attn_forward = [](cudnnHandle_t h, CudaDnnAttn a, int current, const int* lower,
                          const int* upper, const int* seqQ, const int* seqK, CudaDnnSeqData qd,
                          const void* q, const void* residual, CudaDnnSeqData kd, const void* k,
                          CudaDnnSeqData vd, const void* v, CudaDnnSeqData od, void* out,
                          std::size_t wb, const void* w, std::size_t sb, void* work, std::size_t rb,
                          void* reserve) {
        return invoke({Kind::kForward,
                       h,
                       {a, qd, kd, vd, od},
                       {seqQ, seqK, q, k, v, residual, out, nullptr, nullptr, nullptr, nullptr, w,
                        nullptr, work, reserve},
                       {wb, sb, rb},
                       0,
                       current,
                       {lower, upper}});
    };
    sdk.attn_backward_data = [](cudnnHandle_t h, CudaDnnAttn a, const int* lower, const int* upper,
                                const int* seqQ, const int* seqK, CudaDnnSeqData od,
                                const void* dout, CudaDnnSeqData qd, void* dq, const void* q,
                                CudaDnnSeqData kd, void* dk, const void* k, CudaDnnSeqData vd,
                                void* dv, const void* v, std::size_t wb, const void* w,
                                std::size_t sb, void* work, std::size_t rb, void* reserve) {
        return invoke(
            {Kind::kBackwardData,
             h,
             {a, qd, kd, vd, od},
             {seqQ, seqK, q, k, v, nullptr, nullptr, dout, dq, dk, dv, w, nullptr, work, reserve},
             {wb, sb, rb},
             0,
             0,
             {lower, upper}});
    };
    sdk.attn_backward_weights =
        [](cudnnHandle_t h, CudaDnnAttn a, cudnnWgradMode_t mode, CudaDnnSeqData qd, const void* q,
           CudaDnnSeqData kd, const void* k, CudaDnnSeqData vd, const void* v, CudaDnnSeqData od,
           const void* dout, std::size_t wb, const void* w, void* dw, std::size_t sb, void* work,
           std::size_t rb, void* reserve) {
            return invoke({Kind::kBackwardWeights,
                           h,
                           {a, qd, kd, vd, od},
                           {nullptr, nullptr, q, k, v, nullptr, nullptr, dout, nullptr, nullptr,
                            nullptr, w, dw, work, reserve},
                           {wb, sb, rb},
                           static_cast<std::uint32_t>(mode),
                           0,
                           {}});
        };
    if (missing == 1) sdk.attn_forward = nullptr;
    if (missing == 2) sdk.attn_backward_data = nullptr;
    if (missing == 3) sdk.attn_backward_weights = nullptr;
    if (missing == 4) sdk.attn_get = nullptr;
    if (missing == 5) sdk.seq_data_get = nullptr;
    if (missing == 6) sdk.version = nullptr;
    return sdk;
}
inline CudaSessionDispatch session_dispatch(unsigned int missing = 0) {
    auto sdk = query::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch(missing));
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    sdk.context_set_current = [](CUcontext context) {
        ++control.context_sets;
        return foundation::session_dispatch().context_set_current(context);
    };
    sdk.execution.context_get_current = [](CUcontext* context) {
        ++control.context_gets;
        return foundation::session_dispatch().execution.context_get_current(context);
    };
    sdk.context_synchronize = [] {
        ++control.syncs;
        observe(5);
        return static_cast<CUresult>(control.sync_status.load());
    };
    sdk.execution.event_record = [](CUevent event, CUstream) {
        base::require(foundation::driver.events.contains(event));
        ++control.records;
        return static_cast<CUresult>(control.event_status.load());
    };
    sdk.execution.event_query = [](CUevent event) {
        base::require(foundation::driver.events.contains(event));
        ++control.polls;
        return foundation::control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    return sdk;
}
inline unsigned int calls() {
    unsigned int total{};
    for (const auto& value : control.calls) total += value.load();
    return total;
}
}  // namespace glimmer::tests::dnn_attn_compute
