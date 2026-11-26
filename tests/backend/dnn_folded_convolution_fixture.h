#pragma once

#include "dnn_op_tensor_fixture.h"

// Synthetic host descriptor mutations prove ABI, alias and ownership behavior.
// Their markers deliberately do not emulate or qualify cuDNN's folding math.
namespace glimmer::tests::dnn_folded_convolution {
using namespace glimmer::backend::cuda;
namespace transform = glimmer::tests::dnn_op_tensor;
namespace foundation = glimmer::tests::dnn_convolution;
namespace base = glimmer::tests::dnn;
using Kind = CudaDnnObjectKind;
inline constexpr std::array kInputs{Kind::kFilter, Kind::kTensor, Kind::kConvolution,
                                    Kind::kTensor};
inline constexpr std::array kOutputs{
    Kind::kFilter,          Kind::kTensor,          Kind::kConvolution,     Kind::kTensor,
    Kind::kTensorTransform, Kind::kTensorTransform, Kind::kTensorTransform, Kind::kTensorTransform};
inline constexpr CudaDnnTensor kDiff{CUDNN_DATA_FLOAT, 0, 4, {1, 3, 2, 5}, {30, 10, 5, 1}};
inline constexpr CudaDnnTensor kGrad{CUDNN_DATA_FLOAT, 0, 4, {1, 2, 4, 6}, {48, 24, 6, 1}};
inline constexpr CudaDnnFilter kFilter{CUDNN_DATA_FLOAT, 0, 4, {3, 2, 3, 2}};
inline constexpr CudaDnnConvolutionState kConvolution{
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT, 2, {0, 0}, {1, 1}, {1, 1}};
struct Control {
    std::atomic<unsigned int> gets{}, folds{}, event_queries{}, synchronizes{};
    std::atomic<int> getter_status{CUDNN_STATUS_SUCCESS}, fold_status{CUDNN_STATUS_SUCCESS};
    std::atomic<int> rank_override{2}, stride_override{1};
    std::atomic<unsigned int> mode_override{CUDNN_CROSS_CORRELATION};
    std::atomic<bool> change_after_get{}, change_after_fold{}, pause{}, entered{}, release{true};
};
inline Control control;
struct Invocation {
    cudnnHandle_t handle{};
    std::array<void*, 4> inputs{};
    std::array<void*, 8> outputs{};
    std::uint32_t format{};
    int requested{}, final_source_stride{};
};
inline Invocation last;
inline void (*reenter_get)(){};
inline void (*reenter_fold)(){};
inline void reset() {
    transform::reset();
    last = {};
    reenter_get = nullptr;
    reenter_fold = nullptr;
    control.gets.store(0);
    control.folds.store(0);
    control.event_queries.store(0);
    control.synchronizes.store(0);
    control.getter_status.store(CUDNN_STATUS_SUCCESS);
    control.fold_status.store(CUDNN_STATUS_SUCCESS);
    control.rank_override.store(2);
    control.stride_override.store(1);
    control.mode_override.store(CUDNN_CROSS_CORRELATION);
    control.change_after_get.store(false);
    control.change_after_fold.store(false);
    control.pause.store(false);
    control.entered.store(false);
    control.release.store(true);
}
inline CudaDnnDispatch dispatch(unsigned int missing = 0) {
    auto sdk = foundation::dispatch();
    const auto extra = transform::dispatch();
    sdk.tensor_transform_create = extra.tensor_transform_create;
    sdk.tensor_transform_destroy = extra.tensor_transform_destroy;
    sdk.tensor_transform_set = extra.tensor_transform_set;
    sdk.convolution_get_nd = [](cudnnConvolutionDescriptor_t native, int requested, int* rank,
                                int* padding, int* stride, int* dilation,
                                cudnnConvolutionMode_t* mode, cudnnDataType_t* type) {
        const auto& value = foundation::descriptor(native, Kind::kConvolution).convolution;
        ++control.gets;
        last.requested = requested;
        base::require(requested == 2);
        *rank = control.rank_override.load();
        std::copy_n(value.padding.begin(), 2, padding);
        std::copy_n(value.dilation.begin(), 2, dilation);
        stride[0] = control.stride_override.load();
        stride[1] = value.strides[1];
        const auto bits = control.mode_override.load();
        std::memcpy(mode, &bits, sizeof(bits));
        *type = static_cast<cudnnDataType_t>(value.type);
        if (reenter_get) reenter_get();
        if (control.change_after_get.load()) foundation::driver.current = nullptr;
        return static_cast<cudnnStatus_t>(control.getter_status.load());
    };
    sdk.folded_backward_data =
        [](cudnnHandle_t handle, cudnnFilterDescriptor_t w, cudnnTensorDescriptor_t dy,
           cudnnConvolutionDescriptor_t convolution, cudnnTensorDescriptor_t dx,
           cudnnTensorFormat_t format, cudnnFilterDescriptor_t fw, cudnnTensorDescriptor_t pdy,
           cudnnConvolutionDescriptor_t fc, cudnnTensorDescriptor_t fdx,
           cudnnTensorTransformDescriptor_t wt, cudnnTensorTransformDescriptor_t dyt,
           cudnnTensorTransformDescriptor_t dxt, cudnnTensorTransformDescriptor_t unfold) {
            (void)base::handle(handle);
            ++control.folds;
            last.handle = handle;
            last.inputs = {w, dy, convolution, dx};
            last.outputs = {fw, pdy, fc, fdx, wt, dyt, dxt, unfold};
            last.format = static_cast<std::uint32_t>(format);
            control.entered.store(true);
            if (control.pause.load()) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
                while (!control.release.load()) {
                    base::require(std::chrono::steady_clock::now() < deadline);
                    std::this_thread::yield();
                }
            }
            if (reenter_fold) reenter_fold();
            // Copy and mutate in positional order; aliases retain observable
            // read-after-write semantics instead of cloned input snapshots.
            auto& filter = foundation::descriptor(fw, Kind::kFilter);
            filter.filter = foundation::descriptor(w, Kind::kFilter).filter;
            filter.filter.sizes[0] = 101;
            filter.configured = true;
            auto& diff = base::object(pdy, Kind::kTensor);
            diff.tensor = base::object(dy, Kind::kTensor).tensor;
            diff.tensor.sizes[0] = 102;
            diff.configured = true;
            transform::object(wt, Kind::kTensorTransform).transform.dimensions = 104;
            if (control.change_after_fold.load()) foundation::driver.current = nullptr;
            const auto status = static_cast<cudnnStatus_t>(control.fold_status.load());
            if (status != CUDNN_STATUS_SUCCESS) return status;  // Three writes already happened.
            auto& conv = foundation::descriptor(fc, Kind::kConvolution);
            conv.convolution = foundation::descriptor(convolution, Kind::kConvolution).convolution;
            conv.convolution.strides[0] = 7;
            conv.configured = true;
            auto& grad = base::object(fdx, Kind::kTensor);
            grad.tensor = base::object(dx, Kind::kTensor).tensor;
            grad.tensor.sizes[0] = 103;
            grad.configured = true;
            transform::object(dyt, Kind::kTensorTransform).transform.dimensions = 105;
            transform::object(dxt, Kind::kTensorTransform).transform.dimensions = 106;
            last.final_source_stride =
                foundation::descriptor(convolution, Kind::kConvolution).convolution.strides[0];
            if (unfold)
                transform::object(unfold, Kind::kTensorTransform).transform.dimensions =
                    static_cast<unsigned int>(107 + last.final_source_stride);
            return status;
        };
    if (missing == 1) sdk.folded_backward_data = nullptr;
    if (missing == 2) sdk.convolution_get_nd = nullptr;
    return sdk;
}
inline CudaSessionDispatch session_dispatch(unsigned int missing = 0) {
    auto sdk = transform::session_dispatch();
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
    return sdk;
}
}  // namespace glimmer::tests::dnn_folded_convolution
