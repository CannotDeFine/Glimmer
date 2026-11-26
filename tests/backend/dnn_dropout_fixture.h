#pragma once

#include "dnn_convolution_fixture.h"

#include <bit>
#include <cmath>

namespace glimmer::tests::dnn_dropout {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace driver = glimmer::tests::dnn_convolution;

inline constexpr std::size_t kStateBytes = 32;
struct Descriptor {
    bool alive{};
    std::uint32_t probability_bits{0x3e800000U};
    void* states{};
    std::size_t bytes{};
    unsigned long long seed{};
    unsigned int destroy_attempts{};
};
struct Invocation {
    cudnnDropoutDescriptor_t descriptor{};
    cudnnHandle_t handle{};
    cudaStream_t stream{};
    std::uint32_t probability_bits{};
    void* states{};
    std::size_t bytes{};
    unsigned long long seed{};
    bool is_restore{};
};
struct State {
    std::array<Descriptor, 32> descriptors{};
    unsigned int creates{}, destroys{}, queries{}, sets{}, restores{}, initializations{};
    std::size_t reported_bytes{kStateBytes};
    cudnnStatus_t create_status{CUDNN_STATUS_SUCCESS}, destroy_status{CUDNN_STATUS_SUCCESS};
    bool partial_create{}, null_create{}, duplicate_create{};
    Invocation last;
};
inline State state;
struct Control {
    std::atomic<int> query_status{CUDNN_STATUS_SUCCESS}, set_status{CUDNN_STATUS_SUCCESS},
        restore_status{CUDNN_STATUS_SUCCESS};
    std::atomic<bool> should_mutate_on_error{}, should_change_context{};
    std::atomic<unsigned int> callbacks{};
};
inline Control control;

inline void reset() {
    driver::reset();
    state = {};
    control.query_status.store(CUDNN_STATUS_SUCCESS);
    control.set_status.store(CUDNN_STATUS_SUCCESS);
    control.restore_status.store(CUDNN_STATUS_SUCCESS);
    control.should_mutate_on_error.store(false);
    control.should_change_context.store(false);
    control.callbacks.store(0);
}
inline Descriptor& descriptor(cudnnDropoutDescriptor_t native) {
    base::owner();
    for (auto& value : state.descriptors)
        if (reinterpret_cast<cudnnDropoutDescriptor_t>(&value) == native) {
            base::require(value.alive);
            return value;
        }
    throw std::runtime_error("unknown independent dropout descriptor");
}
inline cudnnStatus_t create(cudnnDropoutDescriptor_t* output) {
    base::owner();
    ++state.creates;
    if (state.null_create ||
        (state.create_status != CUDNN_STATUS_SUCCESS && !state.partial_create)) {
        *output = nullptr;
        return state.create_status;
    }
    if (state.duplicate_create)
        for (auto& value : state.descriptors)
            if (value.alive) {
                *output = reinterpret_cast<cudnnDropoutDescriptor_t>(&value);
                return state.create_status;
            }
    for (auto& value : state.descriptors)
        if (!value.alive) {
            value = {};
            value.alive = true;
            *output = reinterpret_cast<cudnnDropoutDescriptor_t>(&value);
            return state.create_status;
        }
    throw std::runtime_error("bounded independent dropout descriptor capacity");
}
inline cudnnStatus_t destroy(cudnnDropoutDescriptor_t native) {
    auto& value = descriptor(native);
    ++state.destroys;
    base::require(++value.destroy_attempts == 1);
    if (state.destroy_status == CUDNN_STATUS_SUCCESS) value.alive = false;
    // The caller owns state storage; descriptor destruction must not access it.
    return state.destroy_status;
}
inline cudnnStatus_t states_size(cudnnHandle_t handle, std::size_t* bytes) {
    (void)base::handle(handle);
    ++state.queries;
    *bytes = state.reported_bytes;
    return static_cast<cudnnStatus_t>(control.query_status.load());
}
inline cudnnStatus_t configure(cudnnDropoutDescriptor_t native, cudnnHandle_t handle,
                               float probability, void* states, std::size_t bytes,
                               unsigned long long seed, bool is_restore) {
    auto& value = descriptor(native);
    const auto& native_handle = base::handle(handle);
    state.last = {native,
                  handle,
                  native_handle.stream,
                  std::bit_cast<std::uint32_t>(probability),
                  states,
                  bytes,
                  seed,
                  is_restore};
    ++(is_restore ? state.restores : state.sets);
    control.callbacks.fetch_add(1);
    auto status = static_cast<cudnnStatus_t>(is_restore ? control.restore_status.load()
                                                        : control.set_status.load());
    if (!std::isfinite(probability) || probability < 0 || probability > 1 ||
        (states && bytes < kStateBytes))
        status = CUDNN_STATUS_BAD_PARAM;
    if (status != CUDNN_STATUS_SUCCESS && !control.should_mutate_on_error.load()) return status;
    value.probability_bits = state.last.probability_bits;
    if (states) {
        value.states = states;
        value.bytes = bytes;
        value.seed = seed;
        if (!is_restore) {
            // This deterministic CPU byte writer checks transport and lifetime,
            // not cuRAND's algorithm, GPU execution or vendor RNG equivalence.
            auto* output = static_cast<unsigned char*>(states);
            for (std::size_t i = 0; i != kStateBytes; ++i)
                output[i] = static_cast<unsigned char>((seed >> (8 * (i % 8))) ^ (0x5bU + i));
            ++state.initializations;
        }
    }
    if (control.should_change_context.load()) driver::driver.current = nullptr;
    return status;
}
inline cudnnStatus_t set(cudnnDropoutDescriptor_t native, cudnnHandle_t handle, float probability,
                         void* states, std::size_t bytes, unsigned long long seed) {
    return configure(native, handle, probability, states, bytes, seed, false);
}
inline cudnnStatus_t restore(cudnnDropoutDescriptor_t native, cudnnHandle_t handle,
                             float probability, void* states, std::size_t bytes,
                             unsigned long long seed) {
    return configure(native, handle, probability, states, bytes, seed, true);
}
inline CudaDnnDispatch dispatch() {
    auto sdk = base::dispatch();
    sdk.dropout_create = create;
    sdk.dropout_destroy = destroy;
    sdk.dropout_states_size = states_size;
    sdk.dropout_set = set;
    sdk.dropout_restore = restore;
    return sdk;
}
inline CudaSessionDispatch session_dispatch() {
    auto sdk = driver::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch());
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    return sdk;
}
}  // namespace glimmer::tests::dnn_dropout
