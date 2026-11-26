#pragma once

#include "dnn_convolution_fixture.h"

namespace glimmer::tests::dnn_backward_convolution {
namespace foundation = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
using namespace glimmer::backend::cuda;

inline constexpr CudaDnnTensor kDy{CUDNN_DATA_FLOAT, 0, 4, {1, 3, 2, 5}, {30, 10, 5, 1}};
inline constexpr CudaDnnTensor kDx{CUDNN_DATA_FLOAT, 0, 4, {1, 2, 4, 6}, {48, 24, 6, 1}};
inline constexpr CudaDnnFilter kFilter{CUDNN_DATA_FLOAT, 0, 4, {3, 2, 3, 2}};
inline constexpr CudaDnnConvolutionState kConvolution{
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT, 2, {0, 0}, {1, 1}, {1, 1}};

struct Control {
    std::array<std::atomic<int>, 2> counts{};
    std::array<std::atomic<int>, 3> statuses{};
    std::array<std::atomic<unsigned int>, 3> calls{};
    std::atomic<std::size_t> bytes{47};
    std::atomic<unsigned int> null_workspace{}, algorithm{}, synchronizes{}, event_queries{};
    std::atomic<bool> changes_context{}, pauses{}, entered{}, can_finish{true};
};
inline Control control;
struct Invocation {
    cudnnHandle_t handle{};
    std::array<void*, 4> descriptors{};  // Original ABI: w, dy, convolution, dx.
};
inline Invocation last;
inline void reset() {
    foundation::reset();
    last = {};
    control.counts[0].store(11);
    control.counts[1].store(13);
    for (auto& status : control.statuses) status.store(CUDNN_STATUS_SUCCESS);
    for (auto& calls : control.calls) calls.store(0);
    control.bytes.store(47);
    control.null_workspace.store(0);
    control.algorithm.store(0);
    control.synchronizes.store(0);
    control.event_queries.store(0);
    control.changes_context.store(false);
    control.pauses.store(false);
    control.entered.store(false);
    control.can_finish.store(true);
}
inline void finish() {
    control.entered.store(true);
    if (control.pauses.load())
        while (!control.can_finish.load()) std::this_thread::yield();
    if (control.changes_context.load()) foundation::driver.current = nullptr;
}
inline cudnnStatus_t count(cudnnHandle_t handle, int* output, unsigned int index) {
    (void)base::handle(handle);
    base::require(output != nullptr);  // Actual MaxCount has no safe native null-output path.
    ++control.calls[index];
    last.handle = handle;
    *output = control.counts[index].load();
    finish();
    return static_cast<cudnnStatus_t>(control.statuses[index].load());
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto sdk = foundation::dispatch();
    sdk.convolution_backward_data_max_count = [](cudnnHandle_t handle, int* output) {
        return count(handle, output, 0);
    };
    sdk.convolution_backward_filter_max_count = [](cudnnHandle_t handle, int* output) {
        return count(handle, output, 1);
    };
    sdk.convolution_backward_data_workspace =
        [](cudnnHandle_t handle, cudnnFilterDescriptor_t w, cudnnTensorDescriptor_t dy,
           cudnnConvolutionDescriptor_t convolution, cudnnTensorDescriptor_t dx,
           cudnnConvolutionBwdDataAlgo_t algorithm, std::size_t* output) {
            (void)base::handle(handle);
            base::require(foundation::descriptor(w, CudaDnnObjectKind::kFilter).filter.sizes ==
                              kFilter.sizes &&
                          base::object(dy, CudaDnnObjectKind::kTensor).tensor.sizes == kDy.sizes &&
                          base::object(dx, CudaDnnObjectKind::kTensor).tensor.sizes == kDx.sizes &&
                          foundation::descriptor(convolution, CudaDnnObjectKind::kConvolution)
                                  .convolution.strides == kConvolution.strides);
            ++control.calls[2];
            control.algorithm.store(static_cast<unsigned int>(algorithm));
            last = {handle, {w, dy, convolution, dx}};
            if (output)
                *output = control.bytes.load();  // Even failures touch only private output.
            else
                ++control.null_workspace;
            finish();
            if (!output) return CUDNN_STATUS_BAD_PARAM;
            if (static_cast<unsigned int>(algorithm) >= 6) return CUDNN_STATUS_NOT_SUPPORTED;
            return static_cast<cudnnStatus_t>(control.statuses[2].load());
        };
    if (missing == 1) sdk.convolution_backward_data_max_count = nullptr;
    if (missing == 2) sdk.convolution_backward_filter_max_count = nullptr;
    if (missing == 3) sdk.convolution_backward_data_workspace = nullptr;
    return sdk;
}
inline CudaSessionDispatch session_dispatch(unsigned int missing = 0) {
    auto sdk = foundation::session_dispatch();
    sdk.dnn = std::make_shared<const CudaDnnDispatch>(dispatch(missing));
    sdk.verified_dnn_header_version = CUDNN_VERSION;
    sdk.execution.event_query = [](CUevent event) {
        ++control.event_queries;
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
        output->bytes[0] = 47;
        return CUDA_SUCCESS;
    };
    sdk.driver_version = [](int* output) {
        *output = 12060;
        return CUDA_SUCCESS;
    };
    return sdk;
}
inline unsigned int calls() {
    unsigned int total{};
    for (const auto& calls : control.calls) total += calls.load();
    return total;
}
}  // namespace glimmer::tests::dnn_backward_convolution
