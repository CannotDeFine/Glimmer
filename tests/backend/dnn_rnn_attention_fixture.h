#pragma once

#include "dnn_dropout_fixture.h"
#include "dnn_library_info_fixture.h"

#include <chrono>
#include <functional>

// Independent host-ABI and shallow-pointer lifetime oracle. No RNN/attention
// algorithm, vendor configuration support or GPU behavior is simulated here.
namespace glimmer::tests::dnn_rnn_attention {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace dropout = glimmer::tests::dnn_dropout;
namespace foundation = glimmer::tests::dnn_library_info;
using Kind = CudaDnnObjectKind;
struct Rnn {
    std::array<std::uint32_t, 8> enums{};
    std::array<std::int32_t, 4> sizes{};
    cudnnDropoutDescriptor_t dropout{};
    std::uint32_t flags{};
};
struct Attn {
    std::uint32_t mode{};
    int heads{};
    std::uint64_t scale_bits{};
    std::array<std::uint32_t, 3> enums{};
    std::array<cudnnDropoutDescriptor_t, 2> dropouts{};
    std::array<int, 11> sizes{};
};
struct Object {
    bool alive{};
    Kind kind{};
    Rnn rnn;
    Attn attn;
    unsigned int destroys{};
};
struct State {
    std::array<Object, 16> objects{};
    std::vector<std::pair<Kind, void*>> destruction;
    unsigned int creates{}, destroys{}, rnn_sets{}, attn_sets{};
    Rnn last_rnn;
    Attn last_attn;
    std::function<void()> callback;
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS};
    bool partial_create{}, null_create{}, duplicate_create{};
};
struct Control {
    std::atomic<int> set_status{CUDNN_STATUS_SUCCESS};
    std::atomic<unsigned int> callbacks{};
    std::atomic<bool> pause{}, entered{}, release{true};
};
inline State state;
inline Control control;
inline void reset() {
    dropout::reset();
    foundation::reset();
    state = {};
    control.set_status.store(CUDNN_STATUS_SUCCESS);
    control.callbacks.store(0);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline void callback(bool setter = false) {
    if (state.callback) state.callback();
    if (!setter) return;
    ++control.callbacks;
    if (control.pause.exchange(false)) {
        control.entered.store(true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!control.release.load()) {
            base::require(std::chrono::steady_clock::now() < deadline);
            std::this_thread::yield();
        }
    }
}
inline Object& object(void* native, Kind kind) {
    base::owner();
    for (auto& entry : state.objects)
        if (&entry == native) {
            base::require(entry.alive && entry.kind == kind);
            return entry;
        }
    throw std::runtime_error("unknown original RNN/attention descriptor");
}
inline void live(cudnnDropoutDescriptor_t descriptor) {
    if (descriptor) (void)dropout::descriptor(descriptor);
}
inline Object* create(Kind kind) {
    base::owner();
    callback();
    ++state.creates;
    if (state.null_create || (state.create_status != CUDNN_STATUS_SUCCESS && !state.partial_create))
        return nullptr;
    if (state.duplicate_create)
        for (auto& entry : state.objects)
            if (entry.alive && entry.kind == kind) return &entry;
    for (auto& entry : state.objects)
        if (!entry.alive) {
            entry = {};
            entry.alive = true;
            entry.kind = kind;
            return &entry;
        }
    throw std::runtime_error("bounded RNN/attention original capacity");
}
inline cudnnStatus_t destroy(void* native, Kind kind) {
    auto& entry = object(native, kind);
    callback();
    if (kind == Kind::kRnn)
        live(entry.rnn.dropout);
    else
        for (auto* dependency : entry.attn.dropouts) live(dependency);
    base::require(++entry.destroys == 1);
    ++state.destroys;
    state.destruction.emplace_back(kind, native);
    if (state.destroy_status == CUDNN_STATUS_SUCCESS) entry.alive = false;
    return state.destroy_status;
}
inline CudaDnnDispatch dispatch() {
    auto sdk = dropout::dispatch();
    sdk.dropout_destroy = [](cudnnDropoutDescriptor_t native) {
        for (const auto& entry : state.objects)
            if (entry.alive) {
                base::require(entry.rnn.dropout != native);
                for (auto* dependency : entry.attn.dropouts) base::require(dependency != native);
            }
        state.destruction.emplace_back(Kind::kDropout, native);
        return dropout::destroy(native);
    };
    sdk.rnn_create = [](CudaDnnRnn* output) {
        *output = reinterpret_cast<CudaDnnRnn>(create(Kind::kRnn));
        return state.create_status;
    };
    sdk.rnn_destroy = [](CudaDnnRnn native) { return destroy(native, Kind::kRnn); };
    sdk.attn_create = [](CudaDnnAttn* output) {
        *output = reinterpret_cast<CudaDnnAttn>(create(Kind::kAttn));
        return state.create_status;
    };
    sdk.attn_destroy = [](CudaDnnAttn native) { return destroy(native, Kind::kAttn); };
    sdk.rnn_set_v8 = [](CudaDnnRnn native, cudnnRNNAlgo_t algorithm, cudnnRNNMode_t cell,
                        cudnnRNNBiasMode_t bias, cudnnDirectionMode_t direction,
                        cudnnRNNInputMode_t input, cudnnDataType_t type, cudnnDataType_t precision,
                        cudnnMathType_t math, std::int32_t input_size, std::int32_t hidden_size,
                        std::int32_t projection_size, std::int32_t layers,
                        cudnnDropoutDescriptor_t dependency, std::uint32_t flags) {
        auto& entry = object(native, Kind::kRnn);
        callback(true);
        live(dependency);
        state.last_rnn = {{static_cast<std::uint32_t>(algorithm), static_cast<std::uint32_t>(cell),
                           static_cast<std::uint32_t>(bias), static_cast<std::uint32_t>(direction),
                           static_cast<std::uint32_t>(input), static_cast<std::uint32_t>(type),
                           static_cast<std::uint32_t>(precision), static_cast<std::uint32_t>(math)},
                          {input_size, hidden_size, projection_size, layers},
                          dependency,
                          flags};
        ++state.rnn_sets;
        entry.rnn = state.last_rnn;  // Deliberate pointer replacement even on native error.
        return static_cast<cudnnStatus_t>(control.set_status.load());
    };
    sdk.attn_set = [](CudaDnnAttn native, unsigned int mode, int heads, double scale,
                      cudnnDataType_t type, cudnnDataType_t precision, cudnnMathType_t math,
                      cudnnDropoutDescriptor_t first, cudnnDropoutDescriptor_t second,
                      int query_size, int key_size, int value_size, int query_projection,
                      int key_projection, int value_projection, int output_projection,
                      int max_query, int max_key, int max_batch, int max_beam) {
        auto& entry = object(native, Kind::kAttn);
        callback(true);
        live(first);
        live(second);
        state.last_attn = {
            mode,
            heads,
            std::bit_cast<std::uint64_t>(scale),
            {static_cast<std::uint32_t>(type), static_cast<std::uint32_t>(precision),
             static_cast<std::uint32_t>(math)},
            {first, second},
            {query_size, key_size, value_size, query_projection, key_projection, value_projection,
             output_projection, max_query, max_key, max_batch, max_beam}};
        ++state.attn_sets;
        if (control.set_status.load() == CUDNN_STATUS_SUCCESS)
            entry.attn = state.last_attn;
        else
            entry.attn.dropouts[0] = first;  // A genuine old/new mixture after partial failure.
        return static_cast<cudnnStatus_t>(control.set_status.load());
    };
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_rnn_attention
