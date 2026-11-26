#pragma once

#include "dnn_rnn_query_fixture.h"

// A deliberately synthetic host layout. This proves ABI, pointer containment and
// ownership, not the vendor's opaque RNN layout or its parameter arithmetic.
namespace glimmer::tests::dnn_rnn_weight_params {
using namespace backend::cuda;
namespace query = glimmer::tests::dnn_rnn_query;
namespace foundation = query::foundation;
namespace base = query::base;
namespace metadata = query::metadata;
namespace dropout = query::dropout;
struct Control {
    std::atomic<unsigned int> calls{};
    std::atomic<int> status{CUDNN_STATUS_SUCCESS}, corrupt{};
    std::atomic<bool> context_loss{}, pause{}, entered{}, release{true};
};
struct Invocation {
    cudnnHandle_t handle{};
    CudaDnnRnn parent{};
    std::int32_t pseudo_layer{}, linear_layer{};
    std::size_t bytes{};
    const void* weights{};
    std::array<cudnnTensorDescriptor_t, 2> destinations{};
    unsigned int outputs{};
};
struct State {
    Invocation last;
    std::function<void()> callback;
};
inline Control control;
inline State state;
inline void reset() {
    query::reset();
    state = {};
    control.calls.store(0);
    control.status.store(CUDNN_STATUS_SUCCESS);
    control.corrupt.store(0);
    control.context_loss.store(false);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline void observe() {
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
}
inline CudaDnnDispatch dispatch(bool missing = false) {
    auto sdk = query::dispatch();
    sdk.rnn_weight_params = [](cudnnHandle_t handle, CudaDnnRnn parent, std::int32_t pseudo,
                               std::size_t bytes, const void* weights, std::int32_t linear,
                               cudnnTensorDescriptor_t matrix, void** matrix_address,
                               cudnnTensorDescriptor_t bias, void** bias_address) {
        query::parent(handle, parent, CudaDnnObjectKind::kRnn);
        ++control.calls;
        state.last = {
            handle, parent,  pseudo,         linear,
            bytes,  weights, {matrix, bias}, (matrix_address ? 1U : 0U) | (bias_address ? 2U : 0U)};
        observe();
        if (pseudo != 0 || linear < 0 || linear > 8) return CUDNN_STATUS_BAD_PARAM;
        const auto begin = reinterpret_cast<std::uintptr_t>(weights);
        const std::array destinations{matrix, bias};
        const std::array addresses{matrix_address, bias_address};
        for (std::size_t slot = 0; slot != 2; ++slot) {
            if (!destinations[slot]) {
                if (addresses[slot]) return CUDNN_STATUS_BAD_PARAM;
                continue;
            }
            // Configured originals unconditionally dereference this argument.
            base::require(addresses[slot]);
            auto& object = base::object(destinations[slot], CudaDnnObjectKind::kTensor);
            if (linear == 8) {
                *addresses[slot] = nullptr;
                object.tensor = {};
                object.configured = true;
                continue;
            }
            const auto offset = slot ? 256U + 16U * static_cast<unsigned int>(linear)
                                     : 32U * static_cast<unsigned int>(linear);
            // Match the audited ordering: address first, capacity check second,
            // destination metadata mutation third. Failed outputs stay private.
            *addresses[slot] = reinterpret_cast<void*>(begin + offset);
            if (offset > bytes || (slot ? 8U : 16U) > bytes - offset) return CUDNN_STATUS_BAD_PARAM;
            object.tensor = {
                CUDNN_DATA_FLOAT, 0, 3, {1, 2, slot ? 1 : 2}, {slot ? 2 : 4, slot ? 1 : 2, 1}};
            object.configured = true;
        }
        if (bias_address && control.corrupt.load()) {
            const auto mode = control.corrupt.load();
            *bias_address =
                reinterpret_cast<void*>(mode == 1   ? begin + bytes
                                        : mode == 2 ? begin + bytes + 1
                                        : mode == 3 ? begin - 1
                                                    : std::numeric_limits<std::uintptr_t>::max());
        }
        return static_cast<cudnnStatus_t>(control.status.load());
    };
    if (missing) sdk.rnn_weight_params = nullptr;
    return sdk;
}
inline CudaSessionDispatch session_dispatch(bool missing = false) {
    auto sdk = query::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch(missing));
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_rnn_weight_params
