#pragma once

#include "dnn_rnn_query_fixture.h"

// Synthetic host-only layout arithmetic, never a vendor weight-layout oracle.
// The configured native helper treats a nonnull base as a numerical addend.
namespace glimmer::tests::dnn_attn_weights {
using namespace backend::cuda;
namespace query = glimmer::tests::dnn_rnn_query;
namespace foundation = query::foundation;
namespace base = query::base;
namespace metadata = query::metadata;
namespace dropout = query::dropout;
struct Control {
    std::atomic<unsigned int> calls{}, versions{};
    std::atomic<std::size_t> version{91002};
    std::atomic<int> status{CUDNN_STATUS_SUCCESS}, corrupt{};
    std::atomic<bool> context_loss{}, version_context_loss{}, impossible_success{}, pause{},
        entered{}, release{true};
};
struct Invocation {
    cudnnHandle_t handle{};
    CudaDnnAttn parent{};
    std::uint32_t kind{};
    std::size_t bytes{};
    const void* base{};
    cudnnTensorDescriptor_t destination{};
    bool has_output{};
};
struct State {
    Invocation last;
    std::function<void()> callback, version_callback;
};
inline Control control;
inline State state;
inline void reset() {
    query::reset();
    state = {};
    control.calls.store(0);
    control.versions.store(0);
    control.version.store(91002);
    control.status.store(CUDNN_STATUS_SUCCESS);
    control.corrupt.store(0);
    control.context_loss.store(false);
    control.version_context_loss.store(false);
    control.impossible_success.store(false);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto sdk = query::dispatch();
    sdk.version = [] {
        ++control.versions;
        if (state.version_callback) state.version_callback();
        if (control.version_context_loss.load()) foundation::driver.current = nullptr;
        return control.version.load();
    };
    sdk.attn_weights = [](cudnnHandle_t handle, CudaDnnAttn parent,
                          cudnnMultiHeadAttnWeightKind_t kind, std::size_t bytes,
                          const void* weights, cudnnTensorDescriptor_t destination, void** output) {
        query::parent(handle, parent, CudaDnnObjectKind::kAttn);
        ++control.calls;
        state.last = {
            handle,      parent,           static_cast<std::uint32_t>(kind), bytes, weights,
            destination, output != nullptr};
        if (state.callback) state.callback();
        if (control.pause.exchange(false)) {
            control.entered.store(true);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
            while (!control.release.load()) {
                base::require(std::chrono::steady_clock::now() < deadline);
                std::this_thread::yield();
            }
        }
        if (control.context_loss.load()) foundation::driver.current = nullptr;
        // Deliberately hostile injected receipt: no private output write can
        // make these impossible successful nullable calls trustworthy.
        if (control.impossible_success.load()) return CUDNN_STATUS_SUCCESS;
        if (!destination || !output || (bytes && !weights)) return CUDNN_STATUS_BAD_PARAM;
        const auto& config = metadata::object(parent, CudaDnnObjectKind::kAttn).attn;
        const bool projected = std::any_of(config.sizes.begin() + 3, config.sizes.begin() + 7,
                                           [](int value) { return value != 0; });
        if (bytes < (projected ? 512U : 0U) || state.last.kind > 7) return CUDNN_STATUS_BAD_PARAM;
        auto& tensor = base::object(destination, CudaDnnObjectKind::kTensor);
        if (!projected) {
            *output = nullptr;
            tensor.tensor = {};
            tensor.configured = true;
        } else {
            const auto bias = state.last.kind >= 4;
            const auto offset = bias ? 192U + 16U * (state.last.kind - 4) : 32U * state.last.kind;
            *output = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(weights) + offset);
            // Deliberately column-major: callers must use the actual strides.
            tensor.tensor = {CUDNN_DATA_FLOAT, 0, 3, {1, 2, bias ? 1 : 2}, {2, 1, 2}};
            tensor.configured = true;
        }
        if (control.corrupt.load() == 1)
            *output = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(weights) + bytes);
        if (control.corrupt.load() == 2)
            *output = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max());
        return static_cast<cudnnStatus_t>(control.status.load());
    };
    if (missing == 1) sdk.attn_weights = nullptr;
    if (missing == 2) sdk.version = nullptr;
    return sdk;
}
inline CudaSessionDispatch session_dispatch(unsigned int missing = 0) {
    auto sdk = query::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch(missing));
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_attn_weights
