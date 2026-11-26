#pragma once

#include "dnn_convolution_fixture.h"
#include "dnn_rnn_attention_fixture.h"
#include "dnn_sequence_fixture.h"

#include <cmath>

// Independent ABI/ownership oracle. Injected sizes are not an implementation
// or qualification of NVIDIA's sizing algorithms or GPU execution.
namespace glimmer::tests::dnn_rnn_query {
using namespace backend::cuda;
namespace base = glimmer::tests::dnn;
namespace foundation = glimmer::tests::dnn_convolution;
namespace metadata = glimmer::tests::dnn_rnn_attention;
namespace sequence = glimmer::tests::dnn_sequence;
namespace dropout = glimmer::tests::dnn_dropout;
struct Control {
    std::array<std::atomic<std::size_t>, 3> sizes{};
    std::array<std::atomic<int>, 4> statuses{};
    std::array<std::atomic<unsigned int>, 4> calls{};
    std::atomic<unsigned int> gets{}, polls{}, synchronizes{}, mask{}, mode{};
    std::atomic<int> get_status{CUDNN_STATUS_SUCCESS}, corrupt{}, context_loss{};
    std::atomic<bool> pause{}, entered{}, release{true};
};
struct State {
    std::function<void()> callback;
    cudnnHandle_t handle{};
    void* descriptor{};
    void* input{};
    std::array<std::uint32_t, 2> clip_enums{};
    std::array<std::uint64_t, 2> clip_bits{};
};
inline Control control;
inline State state;
inline void reset() {
    metadata::reset();
    sequence::reset();
    foundation::reset();  // Establish the real fake-Driver current-context source last.
    state = {};
    for (std::size_t i = 0; i != 3; ++i) control.sizes[i].store(17 + 12 * i);
    for (auto& status : control.statuses) status.store(CUDNN_STATUS_SUCCESS);
    for (auto& calls : control.calls) calls.store(0);
    control.gets.store(0);
    control.polls.store(0);
    control.synchronizes.store(0);
    control.mask.store(0);
    control.mode.store(0);
    control.get_status.store(CUDNN_STATUS_SUCCESS);
    control.corrupt.store(0);
    control.context_loss.store(0);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline void observe(unsigned int phase) {
    if (state.callback) state.callback();
    if (control.pause.exchange(false)) {
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
inline void parent(cudnnHandle_t handle, void* native, CudaDnnObjectKind kind) {
    (void)base::handle(handle);
    const auto& object = metadata::object(native, kind);
    if (kind == CudaDnnObjectKind::kRnn)
        metadata::live(object.rnn.dropout);
    else
        for (const auto dependency : object.attn.dropouts) metadata::live(dependency);
    state.handle = handle;
    state.descriptor = native;
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto sdk = foundation::dispatch();
    const auto owners = metadata::dispatch();
    sdk.rnn_create = owners.rnn_create;
    sdk.rnn_destroy = owners.rnn_destroy;
    sdk.rnn_set_v8 = owners.rnn_set_v8;
    sdk.attn_create = owners.attn_create;
    sdk.attn_destroy = owners.attn_destroy;
    sdk.attn_set = owners.attn_set;
    sdk.dropout_create = owners.dropout_create;
    sdk.dropout_destroy = owners.dropout_destroy;
    sdk.dropout_set = owners.dropout_set;
    sdk.dropout_restore = owners.dropout_restore;
    sdk.dropout_states_size = owners.dropout_states_size;
    const auto sequences = sequence::dispatch();
    sdk.rnn_data_create = sequences.rnn_data_create;
    sdk.rnn_data_destroy = sequences.rnn_data_destroy;
    sdk.rnn_data_set = sequences.rnn_data_set;
    sdk.rnn_weight_space_size = [](cudnnHandle_t handle, CudaDnnRnn descriptor,
                                   std::size_t* output) {
        parent(handle, descriptor, CudaDnnObjectKind::kRnn);
        ++control.calls[0];
        control.mask.store(output ? 1 : 0);
        if (output) *output = control.sizes[0].load();
        observe(2);
        return output ? static_cast<cudnnStatus_t>(control.statuses[0].load())
                      : CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_temp_space_sizes = [](cudnnHandle_t handle, CudaDnnRnn descriptor,
                                  cudnnForwardMode_t mode, CudaDnnRnnData input, std::size_t* work,
                                  std::size_t* reserve) {
        parent(handle, descriptor, CudaDnnObjectKind::kRnn);
        (void)sequence::object(input, CudaDnnObjectKind::kRnnData);
        state.input = input;
        ++control.calls[1];
        control.mask.store((work ? 2U : 0U) | (reserve ? 4U : 0U));
        control.mode.store(static_cast<unsigned int>(mode));
        if (work) *work = control.sizes[1].load();
        if (reserve) *reserve = control.sizes[2].load();
        observe(2);
        return static_cast<cudnnStatus_t>(control.statuses[1].load());
    };
    sdk.attn_buffers = [](cudnnHandle_t handle, CudaDnnAttn descriptor, std::size_t* weight,
                          std::size_t* work, std::size_t* reserve) {
        parent(handle, descriptor, CudaDnnObjectKind::kAttn);
        ++control.calls[2];
        control.mask.store((weight ? 1U : 0U) | (work ? 2U : 0U) | (reserve ? 4U : 0U));
        // Audited successful store order, deliberately not array-slot order.
        if (weight) *weight = control.sizes[0].load();
        if (reserve) *reserve = control.sizes[2].load();
        if (work) *work = control.sizes[1].load();
        observe(2);
        return weight && work ? static_cast<cudnnStatus_t>(control.statuses[2].load())
                              : CUDNN_STATUS_BAD_PARAM;
    };
    sdk.rnn_set_clip_v8 = [](CudaDnnRnn native, cudnnRNNClipMode_t mode, cudnnNanPropagation_t nan,
                             double lower, double upper) {
        const auto& object = metadata::object(native, CudaDnnObjectKind::kRnn);
        metadata::live(object.rnn.dropout);
        state.descriptor = native;
        ++control.calls[3];
        state.clip_enums = {static_cast<std::uint32_t>(mode), static_cast<std::uint32_t>(nan)};
        state.clip_bits = {std::bit_cast<std::uint64_t>(lower),
                           std::bit_cast<std::uint64_t>(upper)};
        observe(3);
        // Injected failures intentionally occur after private mutation. Do not
        // infer rollback from the configured SDK's ordinary validation order.
        if (std::isnan(lower) || std::isnan(upper) || lower > upper) return CUDNN_STATUS_BAD_PARAM;
        return static_cast<cudnnStatus_t>(control.statuses[3].load());
    };
    sdk.rnn_data_get = [](CudaDnnRnnData native, cudnnDataType_t* type,
                          cudnnRNNDataLayout_t* layout, int* maximum, int* batch, int* width,
                          int requested, int* lengths, void* padding) {
        const auto& value = sequence::object(native, CudaDnnObjectKind::kRnnData).rnn;
        ++control.gets;
        base::require(type && layout && maximum && batch && width && requested == 0 && !lengths &&
                      padding && reinterpret_cast<std::uintptr_t>(padding) % alignof(double) == 0);
        *type = static_cast<cudnnDataType_t>(value.type);
        *layout = static_cast<cudnnRNNDataLayout_t>(value.layout);
        *maximum = value.max_sequence_length;
        *batch = value.batch_size;
        *width = value.vector_size;
        if (control.corrupt.load() == 1) {
            const std::uint32_t raw = 32;
            std::memcpy(type, &raw, sizeof(raw));
        }
        if (control.corrupt.load() == 2) {
            const std::uint32_t raw = 4;
            std::memcpy(layout, &raw, sizeof(raw));
        }
        if (control.corrupt.load() == 3) *maximum = 0;
        if (control.corrupt.load() == 4) *maximum = -1;
        const auto bytes = sequence::padding_width(value.type, false);
        if (bytes) std::memcpy(padding, value.padding.data(), bytes);
        observe(1);
        return static_cast<cudnnStatus_t>(control.get_status.load());
    };
    sdk.add_tensor = [](cudnnHandle_t handle, const void*, cudnnTensorDescriptor_t x,
                        const void* input, const void*, cudnnTensorDescriptor_t y, void* output) {
        (void)base::handle(handle);
        (void)base::object(x, CudaDnnObjectKind::kTensor);
        (void)base::object(y, CudaDnnObjectKind::kTensor);
        base::require(input && output);
        ++base::state.add_calls;
        return CUDNN_STATUS_EXECUTION_FAILED;  // Physical lifetime, not a numerical oracle.
    };
    if (missing == 1) sdk.rnn_weight_space_size = nullptr;
    if (missing == 2) sdk.rnn_temp_space_sizes = nullptr;
    if (missing == 3) sdk.attn_buffers = nullptr;
    if (missing == 4) sdk.rnn_set_clip_v8 = nullptr;
    if (missing == 5) sdk.rnn_data_get = nullptr;
    return sdk;
}
inline CudaSessionDispatch session_dispatch(unsigned int missing = 0) {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch(missing));
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    sdk.execution.event_query = [](CUevent event) {
        ++control.polls;
        base::require(foundation::driver.events.contains(event));
        return foundation::control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        ++control.synchronizes;
        return foundation::control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.context_synchronize = [] {
        ++control.synchronizes;
        return foundation::control.can_stop.load() ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.device_uuid = [](CUuuid* output, CUdevice) {
        *output = {};
        output->bytes[0] = 61;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* output) {
        *output = 12060;
        return CUDA_SUCCESS;
    };
    return sdk;
}
inline unsigned int calls() {
    unsigned int result{};
    for (const auto& count : control.calls) result += count.load();
    return result;
}
}  // namespace glimmer::tests::dnn_rnn_query
