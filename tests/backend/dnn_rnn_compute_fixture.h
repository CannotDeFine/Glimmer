#pragma once

#include "dnn_rnn_query_fixture.h"

#include <numeric>

// Independent CPU LSTM and ABI oracle. The twelve weights and reserve layout
// below are fixture-local, NOT NVIDIA's opaque weight/reserve representation.
namespace glimmer::tests::dnn_rnn_compute {
using namespace backend::cuda;
namespace query = glimmer::tests::dnn_rnn_query;
namespace foundation = query::foundation;
namespace base = query::base;
namespace metadata = query::metadata;
namespace sequence = query::sequence;
namespace dropout = query::dropout;
using Kind = CudaDnnRnnComputeKind;
inline constexpr std::size_t kSteps = 3, kBatch = 2, kCells = 6, kWeights = 12;
inline constexpr std::size_t kReserve = 1 + 12 * kCells;
inline constexpr std::array<double, kCells> kInput{.2, -.4, .7, .1, -.3, .9};
inline constexpr std::array<double, kWeights> kWeight{.3,  -.2, .1,  -.4, .25, .2,
                                                      .15, .35, -.1, .6,  -.3, .05};
inline constexpr std::array<double, kBatch> kHidden{.12, -.21}, kCell{.23, -.17}, kDhidden{.4, -.2},
    kDcell{-.3, .1};
inline constexpr std::array<double, kCells> kDy{.2, -.1, .3, .4, -.5, .7};
inline constexpr std::array<int, kBatch> kLengths{3, 2};
struct Control {
    std::array<std::atomic<unsigned int>, 4> calls{};
    std::atomic<unsigned int> gets{}, tensor_gets{}, sizes{}, records{}, polls{}, syncs{};
    std::atomic<unsigned int> context_gets{}, context_sets{}, stream_gets{}, stream_syncs{};
    std::atomic<int> status{CUDNN_STATUS_SUCCESS}, get_status{CUDNN_STATUS_SUCCESS},
        build_status{CUDNN_STATUS_SUCCESS}, sync_status{CUDA_SUCCESS}, stream_status{}, corrupt{},
        context_loss{}, stream_change{}, event_status{CUDA_SUCCESS};
    std::atomic<bool> arithmetic{true}, pause{}, entered{}, release{true}, event_context_loss{};
};
struct Invocation {
    Kind kind{};
    cudnnHandle_t handle{};
    std::array<void*, 5> descriptors{};
    std::array<const void*, 16> buffers{};
    std::array<std::size_t, 3> bytes{};
    std::uint32_t mode{};
    cudaStream_t stream{};
};
struct State {
    Invocation last;
    std::function<void(unsigned int)> callback;
    std::map<void*, int> built;
    unsigned int unloads{};
    int mini_batch{};
    bool issued{};
};
inline Control control;
inline State state;
inline void reset() {
    query::reset();
    state = {};
    for (auto& count : control.calls) count.store(0);
    control.gets.store(0);
    control.tensor_gets.store(0);
    control.sizes.store(0);
    control.records.store(0);
    control.polls.store(0);
    control.syncs.store(0);
    control.context_gets.store(0);
    control.context_sets.store(0);
    control.stream_gets.store(0);
    control.stream_syncs.store(0);
    control.status.store(CUDNN_STATUS_SUCCESS);
    control.get_status.store(CUDNN_STATUS_SUCCESS);
    control.build_status.store(CUDNN_STATUS_SUCCESS);
    control.sync_status.store(CUDA_SUCCESS);
    control.stream_status.store(CUDNN_STATUS_SUCCESS);
    control.corrupt.store(0);
    control.context_loss.store(0);
    control.stream_change.store(0);
    control.event_status.store(CUDA_SUCCESS);
    control.event_context_loss.store(false);
    control.arithmetic.store(true);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline void observe(unsigned int phase) {
    if (state.callback) state.callback(phase);
    if (phase == 2 && control.pause.exchange(false)) {
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
inline std::size_t index(const CudaDnnRnnDataState& data, std::size_t t, std::size_t b) {
    if (data.layout == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED)
        return b * static_cast<std::size_t>(data.max_sequence_length) + t;
    if (data.layout != CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED)
        return t * static_cast<std::size_t>(data.batch_size) + b;
    std::size_t result{};
    for (std::size_t previous = 0; previous < t; ++previous)
        for (const int length : data.lengths) result += length > static_cast<int>(previous);
    for (std::size_t previous = 0; previous < b; ++previous)
        result += data.lengths[previous] > static_cast<int>(t);
    return result;
}
inline std::size_t elements(const CudaDnnRnnDataState& data) {
    return data.layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
               ? static_cast<std::size_t>(
                     std::accumulate(data.lengths.begin(), data.lengths.end(), 0))
               : static_cast<std::size_t>(data.max_sequence_length * data.batch_size);
}
template <typename Scalar>
inline void compute(const Invocation& call) {
    const auto& xdesc = sequence::object(call.descriptors[1], CudaDnnObjectKind::kRnnData).rnn;
    const auto& ydesc = sequence::object(call.descriptors[2], CudaDnnObjectKind::kRnnData).rnn;
    base::require(xdesc.vector_size == 1 && ydesc.vector_size == 1 &&
                  xdesc.max_sequence_length == 3 && xdesc.batch_size == 2);
    const auto in = [&](std::size_t slot) {
        return static_cast<const Scalar*>(call.buffers[slot]);
    };
    const auto out = [&](std::size_t slot) {
        return const_cast<Scalar*>(static_cast<const Scalar*>(call.buffers[slot]));
    };
    auto* reserve = out(15);
    if (call.kind == Kind::kForward) {
        auto* y = out(2);
        const auto* weights = in(13);
        if (reserve) reserve[0] = 1;
        std::fill_n(y, elements(ydesc), Scalar{});
        for (std::size_t b = 0; b < kBatch; ++b) {
            Scalar h = in(5) ? in(5)[b] : Scalar{};
            Scalar c = in(9) ? in(9)[b] : Scalar{};
            for (int t = 0; t < xdesc.lengths[b]; ++t) {
                const auto step = static_cast<std::size_t>(t);
                const auto x = in(1)[index(xdesc, step, b)];
                std::array<Scalar, 4> gates;
                for (std::size_t gate = 0; gate < gates.size(); ++gate) {
                    const Scalar z =
                        weights[3 * gate] * x + weights[3 * gate + 1] * h + weights[3 * gate + 2];
                    gates[gate] = gate == 3 ? std::tanh(z) : Scalar{1} / (Scalar{1} + std::exp(-z));
                }
                if (reserve) {
                    auto* saved = reserve + 1 + 12 * (step * kBatch + b);
                    saved[0] = h;
                    saved[1] = c;
                    std::copy(gates.begin(), gates.end(), saved + 2);
                }
                c = gates[1] * c + gates[0] * gates[3];
                h = gates[2] * std::tanh(c);
                y[index(ydesc, step, b)] = h;
                if (reserve) {
                    auto* saved = reserve + 1 + 12 * (step * kBatch + b);
                    saved[6] = c;
                    saved[7] = h;
                }
            }
            if (out(6)) out(6)[b] = h;
            if (out(10)) out(10)[b] = c;
        }
    } else if (call.kind == Kind::kBackwardData) {
        base::require(reserve && reserve[0] == Scalar{1});
        std::fill_n(out(4), elements(xdesc), Scalar{});
        for (std::size_t b = 0; b < kBatch; ++b) {
            Scalar dh = in(7) ? in(7)[b] : Scalar{};
            Scalar dc = in(11) ? in(11)[b] : Scalar{};
            for (int t = xdesc.lengths[b] - 1; t >= 0; --t) {
                const auto step = static_cast<std::size_t>(t);
                auto* saved = reserve + 1 + 12 * (step * kBatch + b);
                dh += in(3)[index(ydesc, step, b)];
                const auto tanh_c = std::tanh(saved[6]);
                dc += dh * saved[4] * (Scalar{1} - tanh_c * tanh_c);
                std::array<Scalar, 4> dz{dc * saved[5] * saved[2] * (Scalar{1} - saved[2]),
                                         dc * saved[1] * saved[3] * (Scalar{1} - saved[3]),
                                         dh * tanh_c * saved[4] * (Scalar{1} - saved[4]),
                                         dc * saved[2] * (Scalar{1} - saved[5] * saved[5])};
                dc *= saved[3];
                dh = 0;
                Scalar dx{};
                for (std::size_t gate = 0; gate != 4; ++gate) {
                    dx += dz[gate] * in(13)[3 * gate];
                    dh += dz[gate] * in(13)[3 * gate + 1];
                    saved[8 + gate] = dz[gate];
                }
                out(4)[index(xdesc, step, b)] = dx;
            }
            if (out(8)) out(8)[b] = dh;
            if (out(12)) out(12)[b] = dc;
        }
        reserve[0] = 2;
    } else {
        base::require(reserve && reserve[0] == Scalar{2});
        for (std::size_t b = 0; b < kBatch; ++b)
            for (int t = 0; t < xdesc.lengths[b]; ++t) {
                const auto step = static_cast<std::size_t>(t);
                const auto* saved = reserve + 1 + 12 * (step * kBatch + b);
                for (std::size_t gate = 0; gate != 4; ++gate) {
                    const auto dz = saved[8 + gate];
                    out(13)[3 * gate] += dz * in(1)[index(xdesc, step, b)];
                    out(13)[3 * gate + 1] += dz * saved[0];
                    out(13)[3 * gate + 2] += dz;
                }
            }
    }
}
// An independent forward-only loss, evaluated with central differences by the
// consumers. It neither reads the fake reserve nor shares the reverse pass.
inline double loss(std::array<double, kCells> x, std::array<double, kWeights> weights,
                   std::array<double, kBatch> hidden, std::array<double, kBatch> cell) {
    double result{};
    for (std::size_t batch = 0; batch != kBatch; ++batch) {
        for (int t = 0; t != kLengths[batch]; ++t) {
            const auto i = static_cast<std::size_t>(t) * kBatch + batch;
            const auto affine = [&](std::size_t gate) {
                return weights[gate * 3] * x[i] + weights[gate * 3 + 1] * hidden[batch] +
                       weights[gate * 3 + 2];
            };
            const auto sigmoid = [](double value) { return .5 + .5 * std::tanh(value / 2); };
            cell[batch] =
                sigmoid(affine(1)) * cell[batch] + sigmoid(affine(0)) * std::tanh(affine(3));
            hidden[batch] = sigmoid(affine(2)) * std::tanh(cell[batch]);
            result += hidden[batch] * kDy[i];
        }
        result += hidden[batch] * kDhidden[batch] + cell[batch] * kDcell[batch];
    }
    return result;
}
inline double derivative(unsigned int group, std::size_t index) {
    auto x = kInput;
    auto w = kWeight;
    auto h = kHidden;
    auto c = kCell;
    const auto value = [&](double delta) {
        auto xx = x;
        auto ww = w;
        auto hh = h;
        auto cc = c;
        (group == 0   ? xx[index]
         : group == 1 ? ww[index]
         : group == 2 ? hh[index]
                      : cc[index]) += delta;
        return loss(xx, ww, hh, cc);
    };
    return (value(1e-5) - value(-1e-5)) / 2e-5;
}
inline cudnnStatus_t invoke(Invocation call) {
    auto& handle = base::handle(call.handle);
    const auto& parent = metadata::object(call.descriptors[0], CudaDnnObjectKind::kRnn).rnn;
    metadata::live(parent.dropout);
    for (std::size_t i = 1; i != 3; ++i)
        (void)sequence::object(call.descriptors[i], CudaDnnObjectKind::kRnnData);
    for (std::size_t i = 3; i != 5; ++i)
        if (call.descriptors[i])
            (void)base::object(call.descriptors[i], CudaDnnObjectKind::kTensor);
    call.stream = handle.stream;
    state.last = call;
    ++control.calls[static_cast<std::size_t>(call.kind) - 1];
    state.issued = true;
    const auto& data = sequence::object(call.descriptors[1], CudaDnnObjectKind::kRnnData).rnn;
    const auto scalar = data.type == CUDNN_DATA_DOUBLE ? 8U : 4U;
    cudnnStatus_t status = static_cast<cudnnStatus_t>(control.status.load());
    if (call.kind == Kind::kBackwardWeights && call.mode == CUDNN_WGRAD_MODE_SET)
        status = CUDNN_STATUS_NOT_SUPPORTED;
    if (call.bytes[0] < kWeights * scalar || !call.buffers[13] || !call.buffers[14] ||
        call.bytes[1] < 16 ||
        ((call.kind != Kind::kForward || call.mode == CUDNN_FWD_MODE_TRAINING) &&
         (!call.buffers[15] || call.bytes[2] < kReserve * scalar)))
        status = CUDNN_STATUS_BAD_PARAM;
    if (status == CUDNN_STATUS_SUCCESS && control.arithmetic.load()) {
        if (data.type == CUDNN_DATA_DOUBLE)
            compute<double>(call);
        else
            compute<float>(call);
    } else if (status != CUDNN_STATUS_SUCCESS && call.buffers[14] && call.bytes[1]) {
        // A late failure may have changed device storage before reporting an error.
        *const_cast<std::byte*>(static_cast<const std::byte*>(call.buffers[14])) = std::byte{0x5a};
    }
    observe(2);
    if (control.stream_change.load()) handle.stream = reinterpret_cast<cudaStream_t>(0x9a70);
    return status;
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto sdk = query::dispatch();
    sdk.rnn_data_get = [](CudaDnnRnnData native, cudnnDataType_t* type,
                          cudnnRNNDataLayout_t* layout, int* maximum, int* batch, int* width,
                          int requested, int* lengths, void* padding) {
        const auto& value = sequence::object(native, CudaDnnObjectKind::kRnnData).rnn;
        ++control.gets;
        base::require(type && layout && maximum && batch && width && padding && requested >= 0 &&
                      static_cast<std::size_t>(requested) <= kCudaDnnMaxSequenceLengths &&
                      (!requested || lengths) &&
                      reinterpret_cast<std::uintptr_t>(padding) % 8 == 0);
        *type = static_cast<cudnnDataType_t>(value.type);
        *layout = static_cast<cudnnRNNDataLayout_t>(value.layout);
        *maximum = value.max_sequence_length;
        *batch = value.batch_size;
        *width = value.vector_size;
        for (int i = 0; i != requested; ++i)
            lengths[i] = static_cast<std::size_t>(i) < value.lengths.size() ? value.lengths[i] : 0;
        std::memcpy(padding, value.padding.data(), sequence::padding_width(value.type, false));
        const auto corruption = control.corrupt.load();
        if (corruption == 1) {
            const std::uint32_t bits = 32;
            std::memcpy(type, &bits, 4);
        }
        if (corruption == 2) {
            const std::uint32_t bits = 4;
            std::memcpy(layout, &bits, 4);
        }
        if (corruption == 3) *maximum = 0;
        if (corruption == 4) *batch = static_cast<int>(kCudaDnnMaxSequenceLengths) + 1;
        if (corruption == 5 && requested) lengths[0] = -1;
        if (corruption == 6 && requested) ++*batch;
        observe(1);
        return static_cast<cudnnStatus_t>(control.get_status.load());
    };
    sdk.tensor_get_nd = [](cudnnTensorDescriptor_t d, int n, cudnnDataType_t* type, int* rank,
                           int* sizes, int* strides) {
        ++control.tensor_gets;
        return base::dispatch().tensor_get_nd(d, n, type, rank, sizes, strides);
    };
    sdk.tensor_size = [](cudnnTensorDescriptor_t d, std::size_t* bytes) {
        ++control.sizes;
        return foundation::dispatch().tensor_size(d, bytes);
    };
    sdk.get_stream = [](cudnnHandle_t handle, cudaStream_t* output) {
        ++control.stream_gets;
        const auto status = base::dispatch().get_stream(handle, output);
        if (!state.issued) return status;
        observe(4);
        return static_cast<cudnnStatus_t>(control.stream_status.load());
    };
    sdk.rnn_forward = [](cudnnHandle_t h, CudaDnnRnn r, cudnnForwardMode_t mode, const int32_t* seq,
                         CudaDnnRnnData xd, const void* x, CudaDnnRnnData yd, void* y,
                         cudnnTensorDescriptor_t hd, const void* hx, void* hy,
                         cudnnTensorDescriptor_t cd, const void* cx, void* cy, std::size_t wb,
                         const void* w, std::size_t sb, void* s, std::size_t rb, void* reserve) {
        return invoke({Kind::kForward,
                       h,
                       {r, xd, yd, hd, cd},
                       {seq, x, y, nullptr, nullptr, hx, hy, nullptr, nullptr, cx, cy, nullptr,
                        nullptr, w, s, reserve},
                       {wb, sb, rb},
                       static_cast<std::uint32_t>(mode)});
    };
    sdk.rnn_backward_data = [](cudnnHandle_t h, CudaDnnRnn r, const int32_t* seq, CudaDnnRnnData yd,
                               const void* y, const void* dy, CudaDnnRnnData xd, void* dx,
                               cudnnTensorDescriptor_t hd, const void* hx, const void* dhy,
                               void* dhx, cudnnTensorDescriptor_t cd, const void* cx,
                               const void* dcy, void* dcx, std::size_t wb, const void* w,
                               std::size_t sb, void* s, std::size_t rb, void* reserve) {
        return invoke(
            {Kind::kBackwardData,
             h,
             {r, xd, yd, hd, cd},
             {seq, nullptr, y, dy, dx, hx, nullptr, dhy, dhx, cx, nullptr, dcy, dcx, w, s, reserve},
             {wb, sb, rb},
             0});
    };
    sdk.rnn_backward_weights = [](cudnnHandle_t h, CudaDnnRnn r, cudnnWgradMode_t mode,
                                  const int32_t* seq, CudaDnnRnnData xd, const void* x,
                                  cudnnTensorDescriptor_t hd, const void* hx, CudaDnnRnnData yd,
                                  const void* y, std::size_t wb, void* w, std::size_t sb, void* s,
                                  std::size_t rb, void* reserve) {
        return invoke({Kind::kBackwardWeights,
                       h,
                       {r, xd, yd, hd, nullptr},
                       {seq, x, y, nullptr, nullptr, hx, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, w, s, reserve},
                       {wb, sb, rb},
                       static_cast<std::uint32_t>(mode)});
    };
    sdk.rnn_build_dynamic = [](cudnnHandle_t handle, CudaDnnRnn native, int batch) {
        query::parent(handle, native, CudaDnnObjectKind::kRnn);
        ++control.calls[3];
        state.mini_batch = batch;
        observe(3);
        if (batch <= 0) return CUDNN_STATUS_BAD_PARAM;
        state.built[native] = batch;  // Possible partial population on original failure.
        return static_cast<cudnnStatus_t>(control.build_status.load());
    };
    sdk.rnn_destroy = [](CudaDnnRnn native) {
        state.unloads += static_cast<unsigned int>(state.built.erase(native));
        return metadata::dispatch().rnn_destroy(native);
    };
    if (missing == 1) sdk.rnn_forward = nullptr;
    if (missing == 2) sdk.rnn_backward_data = nullptr;
    if (missing == 3) sdk.rnn_backward_weights = nullptr;
    if (missing == 4) sdk.rnn_build_dynamic = nullptr;
    if (missing == 5) sdk.rnn_data_get = nullptr;
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
    sdk.execution.context_get_current = [](CUcontext* output) {
        ++control.context_gets;
        return foundation::session_dispatch().execution.context_get_current(output);
    };
    sdk.execution.event_create = [](CUevent* output, unsigned int flags) {
        const auto status = foundation::session_dispatch().execution.event_create(output, flags);
        if (control.event_context_loss.load()) foundation::driver.current = nullptr;
        return status;
    };
    sdk.execution.stream_synchronize = [](CUstream stream) {
        ++control.stream_syncs;
        return query::session_dispatch().execution.stream_synchronize(stream);
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
    unsigned int count{};
    for (const auto& value : control.calls) count += value.load();
    return count;
}
}  // namespace glimmer::tests::dnn_rnn_compute
