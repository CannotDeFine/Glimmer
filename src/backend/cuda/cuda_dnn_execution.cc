#include "internal/cuda_dnn_execution.h"
#include "../../internal/cudnn_sdk_profile.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace glimmer::backend::cuda {
namespace {
namespace dnn_profile = ::glimmer::internal::cudnn_profile;
class AttnCall final : public CudaExecutionLibraryCall {
   public:
    AttnCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
             CudaSessionDnnAttn problem, std::array<CudaDeviceMemoryView, 15> buffers,
             CudaDnnFindContext context, std::shared_ptr<CudaDnnAttnCompletion> result,
             std::array<CudaDeviceMemoryView, 4> dropout_states)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          buffers_(std::move(buffers)),
          context_(context),
          result_(std::move(result)),
          dropout_states_(std::move(dropout_states)) {}
    std::int32_t issue(CUstream expected, std::span<const std::byte>) noexcept override {
        if (!problem_.attn.lease->begin_call()) {
            fail(CUDA_ERROR_UNKNOWN);
            result_->is_ready = true;
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        struct Guard {
            CudaDnnRnnLease& lease;
            ~Guard() {
                lease.end_call();
            }
        } guard{*problem_.attn.lease};
        if (const auto proof = check_context(); proof != CUDA_SUCCESS) {
            fail(proof);
            result_->is_ready = true;
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        completion_.was_native_issued = true;
        const auto native = invoke();
        result_->result.disposition = CudaDnnDisposition::kComplete;
        result_->result.status = CUDA_SUCCESS;
        result_->result.original_status = native;
        auto proof = check_context();
        if (proof == CUDA_SUCCESS && native != CUDNN_STATUS_SUCCESS) {
            // An original error can bypass joins of vendor-internal streams.
            const auto synchronized = context_.synchronize();
            proof = check_context();
            if (proof == CUDA_SUCCESS) proof = synchronized;
        }
        if (proof == CUDA_SUCCESS) {
            cudaStream_t observed{};
            const auto queried = dispatch_->get_stream(handle_, &observed);
            proof = check_context();
            if (proof == CUDA_SUCCESS && (queried != CUDNN_STATUS_SUCCESS ||
                                          reinterpret_cast<CUstream>(observed) != expected))
                proof = CUDA_ERROR_UNKNOWN;
        }
        if (proof != CUDA_SUCCESS) fail(proof);
        result_->is_ready = true;
        return native;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }

   private:
    CUresult check_context() const noexcept {
        CUcontext current{};
        const auto status = context_.current(&current);
        return status != CUDA_SUCCESS       ? status
               : current == context_.native ? CUDA_SUCCESS
                                            : CUDA_ERROR_INVALID_CONTEXT;
    }
    void fail(CUresult status) noexcept {
        result_->result.disposition = CudaDnnDisposition::kRejected;
        result_->result.status = status;
        result_->has_context_loss = result_->is_cleanup_uncertain = true;
        completion_.status = status;
        completion_.is_cleanup_uncertain = true;
    }
    cudnnStatus_t invoke() const noexcept {
        const auto attn = static_cast<CudaDnnAttn>(problem_.attn.object.native);
        const auto q = static_cast<CudaDnnSeqData>(problem_.descriptors[0].native);
        const auto k = static_cast<CudaDnnSeqData>(problem_.descriptors[1].native);
        const auto v = static_cast<CudaDnnSeqData>(problem_.descriptors[2].native);
        const auto o = static_cast<CudaDnnSeqData>(problem_.descriptors[3].native);
        // NOLINTBEGIN(performance-no-int-to-ptr): validated service-owned GPU ABI.
        const auto pointer = [&](std::size_t index) {
            return reinterpret_cast<void*>(buffers_[index].address);
        };
        const auto* qo = reinterpret_cast<const int*>(buffers_[0].address);
        const auto* kv = reinterpret_cast<const int*>(buffers_[1].address);
        // NOLINTEND(performance-no-int-to-ptr)
        const auto* lo = problem_.window_presence & 1U ? problem_.windows[0].data() : nullptr;
        const auto* hi = problem_.window_presence & 2U ? problem_.windows[1].data() : nullptr;
        if (problem_.kind == CudaDnnAttnComputeKind::kForward)
            return dispatch_->attn_forward(
                handle_, attn, problem_.current, lo, hi, qo, kv, q, pointer(2), pointer(5), k,
                pointer(3), v, pointer(4), o, pointer(6), problem_.weight_bytes, pointer(11),
                problem_.workspace_bytes, pointer(13), problem_.reserve_bytes, pointer(14));
        if (problem_.kind == CudaDnnAttnComputeKind::kBackwardData)
            return dispatch_->attn_backward_data(handle_, attn, lo, hi, qo, kv, o, pointer(7), q,
                                                 pointer(8), pointer(2), k, pointer(9), pointer(3),
                                                 v, pointer(10), pointer(4), problem_.weight_bytes,
                                                 pointer(11), problem_.workspace_bytes, pointer(13),
                                                 problem_.reserve_bytes, pointer(14));
        return dispatch_->attn_backward_weights(
            handle_, attn, static_cast<cudnnWgradMode_t>(problem_.mode), q, pointer(2), k,
            pointer(3), v, pointer(4), o, pointer(7), problem_.weight_bytes, pointer(11),
            pointer(12), problem_.workspace_bytes, pointer(13), problem_.reserve_bytes,
            pointer(14));
    }
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnAttn problem_;
    const std::array<CudaDeviceMemoryView, 15> buffers_;
    const CudaDnnFindContext context_;
    const std::shared_ptr<CudaDnnAttnCompletion> result_;
    const std::array<CudaDeviceMemoryView, 4> dropout_states_;
    CudaExecutionLibraryCompletion completion_{CUDA_SUCCESS, false, false};
};
class RnnCall final : public CudaExecutionLibraryCall {
   public:
    RnnCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
            CudaSessionDnnRnn problem, std::array<CudaDeviceMemoryView, 16> buffers,
            CudaDnnFindContext context, std::shared_ptr<CudaDnnRnnCompletion> result,
            std::array<CudaDeviceMemoryView, 4> dropout_states)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          buffers_(std::move(buffers)),
          context_(context),
          result_(std::move(result)),
          dropout_states_(std::move(dropout_states)) {}
    std::int32_t issue(CUstream expected, std::span<const std::byte>) noexcept override {
        if (!problem_.rnn.lease->begin_call()) {
            fail(CUDA_ERROR_UNKNOWN);
            result_->is_ready = true;
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        struct Guard {
            CudaDnnRnnLease& lease;
            ~Guard() {
                lease.end_call();
            }
        } guard{*problem_.rnn.lease};
        if (const auto proof = check_context(); proof != CUDA_SUCCESS) {
            fail(proof);
            result_->is_ready = true;
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        completion_.was_native_issued = true;
        const auto native = invoke();
        result_->result.disposition = CudaDnnDisposition::kComplete;
        result_->result.status = CUDA_SUCCESS;
        result_->result.original_status = native;
        auto proof = check_context();
        if (proof == CUDA_SUCCESS && native != CUDNN_STATUS_SUCCESS) {
            // Native error paths can bypass joins of their internal streams.
            const auto synchronized = context_.synchronize();
            proof = check_context();
            if (proof == CUDA_SUCCESS) proof = synchronized;
        }
        if (proof == CUDA_SUCCESS) {
            cudaStream_t observed{};
            const auto queried = dispatch_->get_stream(handle_, &observed);
            proof = check_context();
            if (proof == CUDA_SUCCESS && (queried != CUDNN_STATUS_SUCCESS ||
                                          reinterpret_cast<CUstream>(observed) != expected))
                proof = CUDA_ERROR_UNKNOWN;
        }
        if (proof != CUDA_SUCCESS) fail(proof);
        result_->is_ready = true;
        return native;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }

   private:
    CUresult check_context() const noexcept {
        CUcontext current{};
        const auto status = context_.current(&current);
        return status != CUDA_SUCCESS       ? status
               : current == context_.native ? CUDA_SUCCESS
                                            : CUDA_ERROR_INVALID_CONTEXT;
    }
    void fail(CUresult status) noexcept {
        result_->result.disposition = CudaDnnDisposition::kRejected;
        result_->result.status = status;
        result_->has_context_loss = result_->is_cleanup_uncertain = true;
        completion_.status = status;
        completion_.is_cleanup_uncertain = true;
    }
    cudnnStatus_t invoke() const noexcept {
        const auto rnn = static_cast<CudaDnnRnn>(problem_.rnn.object.native);
        const auto x = static_cast<CudaDnnRnnData>(problem_.descriptors[0].native);
        const auto y = static_cast<CudaDnnRnnData>(problem_.descriptors[1].native);
        const auto h = static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[2].native);
        const auto c = static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[3].native);
        // NOLINTBEGIN(performance-no-int-to-ptr): validated service-owned GPU ABI.
        const auto pointer = [&](std::size_t index) {
            return reinterpret_cast<void*>(buffers_[index].address);
        };
        const auto* lengths = reinterpret_cast<const std::int32_t*>(buffers_[0].address);
        // NOLINTEND(performance-no-int-to-ptr)
        if (problem_.kind == CudaDnnRnnComputeKind::kForward)
            return dispatch_->rnn_forward(
                handle_, rnn, static_cast<cudnnForwardMode_t>(problem_.mode), lengths, x,
                pointer(1), y, pointer(2), h, pointer(5), pointer(6), c, pointer(9), pointer(10),
                problem_.weight_bytes, pointer(13), problem_.workspace_bytes, pointer(14),
                problem_.reserve_bytes, pointer(15));
        if (problem_.kind == CudaDnnRnnComputeKind::kBackwardData)
            return dispatch_->rnn_backward_data(
                handle_, rnn, lengths, y, pointer(2), pointer(3), x, pointer(4), h, pointer(5),
                pointer(7), pointer(8), c, pointer(9), pointer(11), pointer(12),
                problem_.weight_bytes, pointer(13), problem_.workspace_bytes, pointer(14),
                problem_.reserve_bytes, pointer(15));
        return dispatch_->rnn_backward_weights(
            handle_, rnn, static_cast<cudnnWgradMode_t>(problem_.mode), lengths, x, pointer(1), h,
            pointer(5), y, pointer(2), problem_.weight_bytes, pointer(13), problem_.workspace_bytes,
            pointer(14), problem_.reserve_bytes, pointer(15));
    }
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnRnn problem_;
    const std::array<CudaDeviceMemoryView, 16> buffers_;
    const CudaDnnFindContext context_;
    const std::shared_ptr<CudaDnnRnnCompletion> result_;
    const std::array<CudaDeviceMemoryView, 4> dropout_states_;
    CudaExecutionLibraryCompletion completion_{CUDA_SUCCESS, false, false};
};
class DnnCall final : public CudaExecutionLibraryCall {
   public:
    DnnCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
            CudaSessionDnnCompute problem, std::array<CudaDeviceMemoryView, 2> buffers)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          buffers_(std::move(buffers)) {}
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        const auto x_desc = static_cast<cudnnTensorDescriptor_t>(problem_.tensors[0].native);
        const auto y_desc = static_cast<cudnnTensorDescriptor_t>(problem_.tensors[1].native);
        // NOLINTBEGIN(performance-no-int-to-ptr): validated original GPU address ABI.
        const auto* x = reinterpret_cast<const void*>(buffers_[0].address);
        auto* y = reinterpret_cast<void*>(buffers_[1].address);
        // NOLINTEND(performance-no-int-to-ptr)
        if (problem_.kind == CudaDnnComputeKind::kActivationForward)
            return dispatch_->activation_forward(
                handle_, static_cast<CudaDnnActivation>(problem_.activation.native),
                problem_.alpha.data(), x_desc, x, problem_.beta.data(), y_desc, y);
        if (problem_.kind == CudaDnnComputeKind::kPoolingForward)
            return dispatch_->pooling_forward(
                handle_, static_cast<CudaDnnPooling>(problem_.pooling.native),
                problem_.alpha.data(), x_desc, x, problem_.beta.data(), y_desc, y);
        if (problem_.kind == CudaDnnComputeKind::kSoftmaxForward)
            return dispatch_->softmax_forward(
                handle_, static_cast<cudnnSoftmaxAlgorithm_t>(problem_.algorithm),
                static_cast<cudnnSoftmaxMode_t>(problem_.mode), problem_.alpha.data(), x_desc, x,
                problem_.beta.data(), y_desc, y);
        if (problem_.kind == CudaDnnComputeKind::kLrnForward)
            return dispatch_->lrn_forward(handle_, static_cast<CudaDnnLrn>(problem_.lrn.native),
                                          static_cast<cudnnLRNMode_t>(problem_.mode),
                                          problem_.alpha.data(), x_desc, x, problem_.beta.data(),
                                          y_desc, y);
        if (problem_.kind == CudaDnnComputeKind::kTransformTensor)
            return dispatch_->transform_tensor(handle_, problem_.alpha.data(), x_desc, x,
                                               problem_.beta.data(), y_desc, y);
        return dispatch_->add_tensor(handle_, problem_.alpha.data(), x_desc, x,
                                     problem_.beta.data(), y_desc, y);
    }

   private:
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnCompute problem_;
    // CPU leases keep the resource owner alive; they never free native buffers.
    const std::array<CudaDeviceMemoryView, 2> buffers_;
};
class ConvolutionCall final : public CudaExecutionLibraryCall {
   public:
    ConvolutionCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
                    CudaSessionDnnConvolution problem, std::array<CudaDeviceMemoryView, 4> buffers)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          buffers_(std::move(buffers)) {}
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        // NOLINTBEGIN(performance-no-int-to-ptr): validated service-owned GPU ABI.
        return dispatch_->convolution_forward(
            handle_, problem_.alpha.data(),
            static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[0].native),
            reinterpret_cast<const void*>(buffers_[0].address),
            static_cast<CudaDnnFilterHandle>(problem_.descriptors[1].native),
            reinterpret_cast<const void*>(buffers_[1].address),
            static_cast<CudaDnnConvolution>(problem_.descriptors[2].native),
            static_cast<cudnnConvolutionFwdAlgo_t>(problem_.algorithm),
            reinterpret_cast<void*>(buffers_[3].address), problem_.workspace_bytes,
            problem_.beta.data(),
            static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[3].native),
            reinterpret_cast<void*>(buffers_[2].address));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnConvolution problem_;
    const std::array<CudaDeviceMemoryView, 4> buffers_;
};
class ReductionCall final : public CudaExecutionLibraryCall {
   public:
    ReductionCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
                  CudaSessionDnnReduction problem, std::array<CudaDeviceMemoryView, 4> buffers)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          buffers_(std::move(buffers)) {}
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        // NOLINTBEGIN(performance-no-int-to-ptr): validated service-owned GPU ABI.
        return dispatch_->reduce_tensor(
            handle_, static_cast<CudaDnnReduction>(problem_.descriptors[1].native),
            reinterpret_cast<void*>(buffers_[2].address), problem_.indices_bytes,
            reinterpret_cast<void*>(buffers_[3].address), problem_.workspace_bytes,
            problem_.alpha.data(),
            static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[0].native),
            reinterpret_cast<const void*>(buffers_[0].address), problem_.beta.data(),
            static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[2].native),
            reinterpret_cast<void*>(buffers_[1].address));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnReduction problem_;
    const std::array<CudaDeviceMemoryView, 4> buffers_;
};
class ReorderCall final : public CudaExecutionLibraryCall {
   public:
    ReorderCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
                CudaSessionDnnReorder problem, std::array<CudaDeviceMemoryView, 4> buffers)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          buffers_(std::move(buffers)) {
        if (!dispatch_ || !dispatch_->reorder_filter_bias || !handle_ || problem_.reorder > 1 ||
            !valid_cuda_dnn_reorder_descriptors(problem_))
            throw std::invalid_argument("invalid owned cuDNN filter reorder call");
        for (std::size_t index = 0; index != buffers_.size(); ++index) {
            const auto& buffer = buffers_[index];
            const auto& range = problem_.buffers[index];
            if (index >= 2 && !problem_.reorder_bias) {
                if (range.allocation || range.offset || buffer.address || buffer.bytes ||
                    buffer.lifetime)
                    throw std::invalid_argument("noncanonical ignored cuDNN reorder bias");
            } else if (!range.allocation || !buffer.address || !buffer.bytes || !buffer.lifetime) {
                throw std::invalid_argument("missing owned cuDNN reorder buffer");
            }
        }
    }
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        // NOLINTBEGIN(performance-no-int-to-ptr): validated service-owned GPU ABI.
        return dispatch_->reorder_filter_bias(
            handle_, static_cast<CudaDnnFilterHandle>(problem_.filter.native),
            static_cast<decltype(CUDNN_DEFAULT_REORDER)>(problem_.reorder),
            reinterpret_cast<const void*>(buffers_[0].address),
            reinterpret_cast<void*>(buffers_[1].address), problem_.reorder_bias,
            reinterpret_cast<const void*>(buffers_[2].address),
            reinterpret_cast<void*>(buffers_[3].address));
        // NOLINTEND(performance-no-int-to-ptr)
    }

   private:
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnReorder problem_;
    // A bias failure can follow an issued filter operation. The dispatcher
    // retains every owner through its trailing event, including native errors.
    const std::array<CudaDeviceMemoryView, 4> buffers_;
};
class BatchNormCall final : public CudaExecutionLibraryCall {
   public:
    BatchNormCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
                  CudaSessionDnnBatchNorm problem, std::array<CudaDeviceMemoryView, 13> buffers)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          buffers_(std::move(buffers)) {
        const bool is_forward = problem_.kind == CudaDnnBatchNormKind::kForward;
        const bool is_backward = problem_.kind == CudaDnnBatchNormKind::kBackward;
        if (!dispatch_ || !handle_ || (!is_forward && !is_backward) ||
            !valid_cuda_dnn_batch_norm_descriptors(problem_) || problem_.mode > 3 ||
            problem_.operation > 3 ||
            (problem_.scalar_bytes != sizeof(float) && problem_.scalar_bytes != sizeof(double)) ||
            (is_forward && !dispatch_->batch_norm_forward) ||
            (is_backward && !dispatch_->batch_norm_backward))
            throw std::invalid_argument("invalid owned cuDNN BatchNormalization Ex call");
    }
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        const auto tensor = [&](std::size_t index) {
            return static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[index].native);
        };
        const auto memory = [&](std::size_t index) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr): validated service-owned GPU ABI.
            return reinterpret_cast<void*>(buffers_[index].address);
        };
        const auto mode = static_cast<decltype(CUDNN_BATCHNORM_SPATIAL)>(problem_.mode);
        const auto operation = static_cast<decltype(CUDNN_BATCHNORM_OPS_BN)>(problem_.operation);
        const auto activation = static_cast<CudaDnnActivation>(problem_.descriptors[6].native);
        const auto& coefficients = problem_.coefficients;
        if (problem_.kind == CudaDnnBatchNormKind::kForward)
            return dispatch_->batch_norm_forward(
                handle_, mode, operation, coefficients[0].data(), coefficients[1].data(), tensor(0),
                memory(0), tensor(3), memory(3), tensor(1), memory(1), tensor(5), memory(5),
                memory(6), problem_.average_factor, memory(7), memory(8), problem_.epsilon,
                memory(9), memory(10), activation, memory(11), problem_.workspace_bytes, memory(12),
                problem_.reserve_bytes);
        return dispatch_->batch_norm_backward(
            handle_, mode, operation, coefficients[0].data(), coefficients[1].data(),
            coefficients[2].data(), coefficients[3].data(), tensor(0), memory(0), tensor(1),
            memory(1), tensor(2), memory(2), tensor(3), memory(3), tensor(4), memory(4), tensor(5),
            memory(5), memory(6), memory(7), memory(8), problem_.epsilon, memory(9), memory(10),
            activation, memory(11), problem_.workspace_bytes, memory(12), problem_.reserve_bytes);
    }

   private:
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnBatchNorm problem_;
    const std::array<CudaDeviceMemoryView, 13> buffers_;
};
class DropoutCall final : public CudaExecutionLibraryCall {
   public:
    DropoutCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
                CudaSessionDnnDropout problem, CudaDeviceMemoryView states,
                CudaDnnDropoutBinding binding, CudaDnnFindContext context)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          states_(std::move(states)),
          binding_(std::move(binding)),
          context_(context) {
        const bool is_set = problem_.kind == CudaDnnDropoutKind::kSet;
        const bool is_restore = problem_.kind == CudaDnnDropoutKind::kRestore;
        if (!dispatch_ || !handle_ || (!is_set && !is_restore) ||
            (is_set && !dispatch_->dropout_set) || (is_restore && !dispatch_->dropout_restore) ||
            !context_.native || !context_.current || !context_.synchronize || !problem_.mutation ||
            !problem_.mutation->native() || !problem_.mutation->claim())
            throw std::invalid_argument("invalid owned cuDNN dropout mutation");
    }
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        const auto descriptor = problem_.mutation->native();
        if (!descriptor || !problem_.mutation->mark_issued()) {
            completion_ = {CUDA_ERROR_UNKNOWN, true};
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        const auto operation = problem_.kind == CudaDnnDropoutKind::kSet
                                   ? dispatch_->dropout_set
                                   : dispatch_->dropout_restore;
        // NOLINTNEXTLINE(performance-no-int-to-ptr): validated service-owned GPU ABI.
        const auto status = operation(
            descriptor, handle_, std::bit_cast<float>(problem_.probability_bits),
            reinterpret_cast<void*>(states_.address), problem_.state_bytes, problem_.seed);
        // Non-null Set can initialize on SDK-internal streams. Apply the same
        // conservative stop proof to both operations, including native errors.
        auto proof = check_context();
        if (proof == CUDA_SUCCESS) {
            const auto synchronized = context_.synchronize();
            const auto restored = check_context();
            proof = synchronized != CUDA_SUCCESS ? synchronized : restored;
        }
        if (proof != CUDA_SUCCESS) completion_ = {proof, true};
        problem_.mutation->finish(status, binding_, proof);
        return status;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }

   private:
    [[nodiscard]] CUresult check_context() const noexcept {
        CUcontext current{};
        const auto status = context_.current(&current);
        if (status != CUDA_SUCCESS) return status;
        return current == context_.native ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
    }
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnDropout problem_;
    const CudaDeviceMemoryView states_;
    const CudaDnnDropoutBinding binding_;
    const CudaDnnFindContext context_;
    CudaExecutionLibraryCompletion completion_;
};
class ConvolutionFindCall final : public CudaExecutionLibraryCall {
   public:
    ConvolutionFindCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
                        CudaSessionDnnConvolution problem, int capacity, CudaDnnFindContext context,
                        std::shared_ptr<CudaDnnConvolutionFindResult> result)
        : dispatch_(std::move(dispatch)),
          handle_(handle),
          problem_(std::move(problem)),
          capacity_(capacity),
          context_(context),
          result_(std::move(result)) {
        if (!dispatch_ || !dispatch_->convolution_forward_find || !handle_ || !result_ ||
            !context_.native || !context_.current || !context_.synchronize ||
            !valid_cuda_dnn_convolution_descriptors(problem_) ||
            capacity_ > static_cast<int>(kCudaDnnMaxConvolutionAlgorithms))
            throw std::invalid_argument("invalid owned cuDNN Find call");
    }
    std::int32_t issue(CUstream, std::span<const std::byte>) noexcept override {
        std::array<cudnnConvolutionFwdAlgoPerfStruct, kCudaDnnMaxConvolutionAlgorithms> native{};
        int count{};
        const auto status = dispatch_->convolution_forward_find(
            handle_, static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[0].native),
            static_cast<CudaDnnFilterHandle>(problem_.descriptors[1].native),
            static_cast<CudaDnnConvolution>(problem_.descriptors[2].native),
            static_cast<cudnnTensorDescriptor_t>(problem_.descriptors[3].native), capacity_, &count,
            native.data());
        CudaDnnConvolutionQueryResult output;
        output.result.disposition = CudaDnnDisposition::kComplete;
        output.result.status = CUDA_SUCCESS;
        output.result.original_status = status;

        // The SDK may benchmark on private streams, even when it returns an
        // error. A stream-local trailing event is not sufficient stop proof.
        auto proof = check_context();
        if (proof == CUDA_SUCCESS) {
            const auto synchronized = context_.synchronize();
            const auto restored = check_context();
            proof = synchronized != CUDA_SUCCESS ? synchronized : restored;
        }
        if (proof != CUDA_SUCCESS) {
            completion_ = {proof, true};
        } else if (status == CUDNN_STATUS_SUCCESS) {
            if (count < 0 || count > capacity_) {
                completion_.status = CUDA_ERROR_UNKNOWN;
            } else {
                decltype(output.performance) candidate{};
                for (int index = 0; index < count; ++index) {
                    const auto& value = native[static_cast<std::size_t>(index)];
                    auto& converted = candidate[static_cast<std::size_t>(index)];
                    // Do not load unvalidated native enum values as enums.
                    std::memcpy(&converted.algorithm, &value.algo, sizeof(converted.algorithm));
                    std::memcpy(&converted.status, &value.status, sizeof(converted.status));
                    std::memcpy(&converted.determinism, &value.determinism,
                                sizeof(converted.determinism));
                    std::memcpy(&converted.math, &value.mathType, sizeof(converted.math));
                    converted.time_bits = std::bit_cast<std::uint32_t>(value.time);
                    converted.memory = value.memory;
                    std::copy(std::begin(value.reserved), std::end(value.reserved),
                              converted.reserved.begin());
                    if (converted.algorithm > 15 ||
                        converted.status > dnn_profile::status_enum_max() ||
                        converted.determinism > 1 || converted.math > 3) {
                        completion_.status = CUDA_ERROR_UNKNOWN;
                        break;
                    }
                }
                if (completion_.status == CUDA_SUCCESS) {
                    output.count = static_cast<std::uint32_t>(count);
                    output.performance = candidate;
                }
            }
        }
        if (completion_.status != CUDA_SUCCESS) {
            output.result.disposition = CudaDnnDisposition::kRejected;
            output.result.status = completion_.status;
        }
        result_->query = output;
        result_->is_ready = true;
        return status;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }

   private:
    [[nodiscard]] CUresult check_context() const noexcept {
        CUcontext current{};
        const auto status = context_.current(&current);
        if (status != CUDA_SUCCESS) return status;
        return current == context_.native ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
    }
    const std::shared_ptr<const CudaDnnDispatch> dispatch_;
    const cudnnHandle_t handle_;
    const CudaSessionDnnConvolution problem_;
    const int capacity_;
    const CudaDnnFindContext context_;
    const std::shared_ptr<CudaDnnConvolutionFindResult> result_;
    CudaExecutionLibraryCompletion completion_;
};
}  // namespace
bool valid_cuda_dnn_attn_descriptors(const CudaSessionDnnAttn& problem) noexcept {
    const bool forward = problem.kind == CudaDnnAttnComputeKind::kForward;
    const bool data = problem.kind == CudaDnnAttnComputeKind::kBackwardData;
    const bool weights = problem.kind == CudaDnnAttnComputeKind::kBackwardWeights;
    if ((!forward && !data && !weights) || problem.mode > 1 || (!weights && problem.mode) ||
        (!forward && problem.current) || problem.window_presence > 3 ||
        (weights && problem.window_presence) ||
        problem.attn.object.kind != CudaDnnObjectKind::kAttn || !problem.attn.object.native ||
        !problem.attn.object.lifetime || !problem.attn.lease)
        return false;
    for (const auto& descriptor : problem.descriptors) {
        if (!descriptor.native && !descriptor.lifetime && descriptor.kind == CudaDnnObjectKind{})
            continue;
        if (descriptor.kind != CudaDnnObjectKind::kSeqData || !descriptor.native ||
            !descriptor.lifetime)
            return false;
    }
    for (std::size_t index = 0; index < problem.buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        const bool unused =
            (forward && (index == 7 || index == 8 || index == 9 || index == 10 || index == 12)) ||
            (data && (index == 5 || index == 6 || index == 12)) ||
            (weights && (index == 0 || index == 1 || index == 5 || index == 6 || index == 8 ||
                         index == 9 || index == 10));
        if ((!range.allocation && range.offset) || (unused && (range.allocation || range.offset)))
            return false;
    }
    for (std::size_t index = 0; index < problem.windows.size(); ++index)
        if (problem.windows[index].size() > kCudaDnnMaxSequenceLengths ||
            (!(problem.window_presence & (1U << index)) && !problem.windows[index].empty()))
            return false;
    return problem.window_presence != 3 || problem.windows[0].size() == problem.windows[1].size();
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_attn_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnAttn problem, std::array<CudaDeviceMemoryView, 15> buffers,
    CudaDnnFindContext context, std::shared_ptr<CudaDnnAttnCompletion> result,
    std::array<CudaDeviceMemoryView, 4> dropout_states, std::uint32_t first) {
    for (std::size_t index = 0; index < problem.windows.size(); ++index) {
        auto& window = problem.windows[index];
        if (first > kCudaDnnMaxSequenceLengths ||
            window.size() > kCudaDnnMaxSequenceLengths - first)
            throw std::invalid_argument("attention window prefix exceeds its owned bound");
        if (!(problem.window_presence & (1U << index))) continue;
        // Empty capture is permitted only after the session proves no read.
        // Preserve pointer presence without forming a pointer before storage.
        if (window.empty())
            window.resize(1);
        else if (first)
            window.insert(window.begin(), first, 0);
    }
    return std::make_shared<AttnCall>(std::move(dispatch), handle, std::move(problem),
                                      std::move(buffers), context, std::move(result),
                                      std::move(dropout_states));
}
bool valid_cuda_dnn_rnn_descriptors(const CudaSessionDnnRnn& problem) noexcept {
    const bool forward = problem.kind == CudaDnnRnnComputeKind::kForward;
    const bool data = problem.kind == CudaDnnRnnComputeKind::kBackwardData;
    const bool weights = problem.kind == CudaDnnRnnComputeKind::kBackwardWeights;
    if ((!forward && !data && !weights) || problem.mode > 1 || (data && problem.mode) ||
        problem.rnn.object.kind != CudaDnnObjectKind::kRnn || !problem.rnn.object.native ||
        !problem.rnn.object.lifetime || !problem.rnn.lease)
        return false;
    for (std::size_t index = 0; index < problem.descriptors.size(); ++index) {
        const auto& descriptor = problem.descriptors[index];
        const bool absent =
            !descriptor.native && !descriptor.lifetime && descriptor.kind == CudaDnnObjectKind{};
        if (weights && index == 3) {
            if (!absent) return false;
            continue;
        }
        if (absent && index >= 2) continue;
        if (descriptor.kind !=
                (index < 2 ? CudaDnnObjectKind::kRnnData : CudaDnnObjectKind::kTensor) ||
            !descriptor.native || !descriptor.lifetime)
            return false;
    }
    for (std::size_t index = 0; index < problem.buffers.size(); ++index) {
        const auto& range = problem.buffers[index];
        const bool unused = (forward && (index == 3 || index == 4 || index == 7 || index == 8 ||
                                         index == 11 || index == 12)) ||
                            (data && (index == 1 || index == 6 || index == 10)) ||
                            (weights && (index == 3 || index == 4 || index == 6 || index == 7 ||
                                         index == 8 || (index >= 9 && index <= 12)));
        if ((!range.allocation && range.offset) || (unused && (range.allocation || range.offset)))
            return false;
        if (range.allocation && index >= 5 && index <= 12 &&
            !problem.descriptors[index <= 8 ? 2 : 3].native)
            return false;
    }
    return true;
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_rnn_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnRnn problem, std::array<CudaDeviceMemoryView, 16> buffers,
    CudaDnnFindContext context, std::shared_ptr<CudaDnnRnnCompletion> result,
    std::array<CudaDeviceMemoryView, 4> dropout_states) {
    return std::make_shared<RnnCall>(std::move(dispatch), handle, std::move(problem),
                                     std::move(buffers), context, std::move(result),
                                     std::move(dropout_states));
}
bool valid_cuda_dnn_reorder_descriptors(const CudaSessionDnnReorder& problem) noexcept {
    return problem.filter.kind == CudaDnnObjectKind::kFilter && problem.filter.native &&
           problem.filter.lifetime;
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_reorder_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnReorder problem, std::array<CudaDeviceMemoryView, 4> buffers) {
    return std::make_shared<ReorderCall>(std::move(dispatch), handle, std::move(problem),
                                         std::move(buffers));
}
bool valid_cuda_dnn_batch_norm_descriptors(const CudaSessionDnnBatchNorm& problem) noexcept {
    const bool is_forward = problem.kind == CudaDnnBatchNormKind::kForward;
    const bool is_backward = problem.kind == CudaDnnBatchNormKind::kBackward;
    const bool is_reserve = problem.kind == CudaDnnBatchNormKind::kReserve;
    if (!is_forward && !is_backward && !is_reserve) return false;
    for (std::size_t index = 0; index != problem.descriptors.size(); ++index) {
        const auto& descriptor = problem.descriptors[index];
        const bool is_absent =
            descriptor.kind == CudaDnnObjectKind{} && !descriptor.native && !descriptor.lifetime;
        const bool is_unused =
            (is_forward && (index == 2 || index == 4)) || (is_reserve && index != 0 && index != 6);
        if (is_unused) {
            if (!is_absent) return false;
            continue;
        }
        const bool is_required = index == 0 || (!is_reserve && index == 5) ||
                                 (is_forward && index == 1) ||
                                 (is_backward && (index == 2 || index == 4));
        if (is_absent) {
            if (is_required) return false;
            continue;
        }
        const auto kind = index == 6 ? CudaDnnObjectKind::kActivation : CudaDnnObjectKind::kTensor;
        if (descriptor.kind != kind || !descriptor.native || !descriptor.lifetime) return false;
    }
    return true;
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_batch_norm_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnBatchNorm problem, std::array<CudaDeviceMemoryView, 13> buffers) {
    return std::make_shared<BatchNormCall>(std::move(dispatch), handle, std::move(problem),
                                           std::move(buffers));
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_dropout_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnDropout problem, CudaDeviceMemoryView states, CudaDnnDropoutBinding binding,
    CudaDnnFindContext context) {
    return std::make_shared<DropoutCall>(std::move(dispatch), handle, std::move(problem),
                                         std::move(states), std::move(binding), context);
}
bool valid_cuda_dnn_reduction_descriptors(const CudaSessionDnnReduction& problem) noexcept {
    constexpr std::array kinds{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kReduction,
                               CudaDnnObjectKind::kTensor};
    for (std::size_t index = 0; index != kinds.size(); ++index) {
        const auto& descriptor = problem.descriptors[index];
        if (descriptor.kind != kinds[index] || !descriptor.native || !descriptor.lifetime)
            return false;
    }
    return true;
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_reduction_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnReduction problem, std::array<CudaDeviceMemoryView, 4> buffers) {
    return std::make_shared<ReductionCall>(std::move(dispatch), handle, std::move(problem),
                                           std::move(buffers));
}
bool valid_cuda_dnn_convolution_descriptors(const CudaSessionDnnConvolution& problem) noexcept {
    constexpr std::array kinds{CudaDnnObjectKind::kTensor, CudaDnnObjectKind::kFilter,
                               CudaDnnObjectKind::kConvolution, CudaDnnObjectKind::kTensor};
    for (std::size_t index = 0; index != kinds.size(); ++index) {
        const auto& descriptor = problem.descriptors[index];
        if (descriptor.kind != kinds[index] || !descriptor.native || !descriptor.lifetime)
            return false;
    }
    return true;
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_convolution_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnConvolution problem, std::array<CudaDeviceMemoryView, 4> buffers) {
    return std::make_shared<ConvolutionCall>(std::move(dispatch), handle, std::move(problem),
                                             std::move(buffers));
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_convolution_find_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnConvolution problem, int capacity, CudaDnnFindContext context,
    std::shared_ptr<CudaDnnConvolutionFindResult> result) {
    return std::make_shared<ConvolutionFindCall>(std::move(dispatch), handle, std::move(problem),
                                                 capacity, context, std::move(result));
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaSessionDnnCompute problem, std::array<CudaDeviceMemoryView, 2> buffers) {
    return std::make_shared<DnnCall>(std::move(dispatch), handle, std::move(problem),
                                     std::move(buffers));
}
}  // namespace glimmer::backend::cuda
