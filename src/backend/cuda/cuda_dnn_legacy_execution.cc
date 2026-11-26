#include "internal/cuda_dnn_legacy.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace glimmer::backend::cuda {
std::optional<std::size_t> cuda_dnn_legacy_storage_bytes(
    const CudaSessionDnnLegacyRnn& input) noexcept {
    if (input.sequence_length > static_cast<int>(kCudaDnnMaxSequenceLengths)) return {};
    // At most three arrays of each of the three element types. Each vector is
    // resized once from empty. Padding covers alignment of every allocation;
    // null upstream still enforces the bound if an STL requests extra capacity.
    std::size_t bytes = 9 * (alignof(std::max_align_t) - 1);
    const auto add = [&](std::size_t count, std::size_t width) {
        if (count > (std::numeric_limits<std::size_t>::max() - bytes) / width) return false;
        bytes += count * width;
        return true;
    };
    for (const auto& array : input.sequences) {
        const auto count = array.descriptors.size();
        if (array.has_descriptors > 1 || count > kCudaDnnMaxSequenceLengths ||
            count != (array.has_descriptors && input.sequence_length > 0
                          ? static_cast<std::size_t>(input.sequence_length)
                          : 0))
            return {};
        if (!add(count, sizeof(CudaDnnObjectView)) || !add(count, sizeof(CudaDnnLegacyExtent)) ||
            !add(array.has_descriptors ? std::max<std::size_t>(1, count) : 0,
                 sizeof(cudnnTensorDescriptor_t)))
            return {};
    }
    return bytes;
}
namespace {
#if CUDNN_MAJOR == 8
struct LegacyRnnParameters {
    CudaDnnLegacyComputeKind kind;
    std::int32_t sequence_length;
    std::uint64_t workspace_bytes, reserve_bytes;
    std::array<bool, 3> has_descriptors;
};
class LegacyRnnCall final : public CudaExecutionLibraryCall {
   public:
    LegacyRnnCall(std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t native_handle,
                  const CudaSessionDnnLegacyRnn& request, CudaDnnLegacyView views,
                  std::array<CudaDeviceMemoryView, 13> memory, CudaDnnFindContext proof_context,
                  std::shared_ptr<CudaDnnRnnCompletion> completion,
                  std::array<CudaDeviceMemoryView, 4> rng_memory)
        : sdk(std::move(dispatch)),
          handle(native_handle),
          problem{
              request.kind,
              request.sequence_length,
              request.workspace_bytes,
              request.reserve_bytes,
              {request.sequences[0].has_descriptors != 0, request.sequences[1].has_descriptors != 0,
               request.sequences[2].has_descriptors != 0}},
          descriptors(std::move(views)),
          buffers(std::move(memory)),
          context(proof_context),
          result(std::move(completion)),
          dropout(std::move(rng_memory)),
          arrays_{std::pmr::vector<cudnnTensorDescriptor_t>{descriptors.resource()},
                  std::pmr::vector<cudnnTensorDescriptor_t>{descriptors.resource()},
                  std::pmr::vector<cudnnTensorDescriptor_t>{descriptors.resource()}} {
        for (std::size_t i = 0; i < arrays_.size(); ++i) {
            if (!problem.has_descriptors[i]) continue;
            arrays_[i].resize(std::max<std::size_t>(1, descriptors.sequences[i].size()));
            for (std::size_t j = 0; j < descriptors.sequences[i].size(); ++j)
                arrays_[i][j] =
                    static_cast<cudnnTensorDescriptor_t>(descriptors.sequences[i][j].native);
        }
    }
    std::int32_t issue(CUstream expected, std::span<const std::byte>) noexcept override {
        if (!descriptors.rnn.lease->begin_call()) {
            fail(CUDA_ERROR_UNKNOWN);
            result->is_ready = true;
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        struct Guard {
            CudaDnnRnnLease& lease;
            ~Guard() {
                lease.end_call();
            }
        } guard{*descriptors.rnn.lease};
        if (const auto proof = check(); proof != CUDA_SUCCESS) {
            fail(proof);
            result->is_ready = true;
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        completion_.was_native_issued = true;
        const auto native = invoke();
        result->result.disposition = CudaDnnDisposition::kComplete;
        result->result.status = CUDA_SUCCESS;
        result->result.original_status = native;
        auto proof = check();
        if (proof == CUDA_SUCCESS && native != CUDNN_STATUS_SUCCESS) {
            // Error return is not evidence that vendor-internal streams joined.
            // Keep every immutable resource until synchronization and the real
            // dispatcher tail prove physical retirement; never replay.
            const auto synchronized = context.synchronize();
            proof = check();
            if (proof == CUDA_SUCCESS) proof = synchronized;
        }
        if (proof == CUDA_SUCCESS) {
            cudaStream_t observed{};
            const auto queried = sdk->get_stream(handle, &observed);
            proof = check();
            if (proof == CUDA_SUCCESS && (queried != CUDNN_STATUS_SUCCESS ||
                                          reinterpret_cast<CUstream>(observed) != expected))
                proof = CUDA_ERROR_UNKNOWN;
        }
        if (proof != CUDA_SUCCESS) fail(proof);
        result->is_ready = true;
        return native;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }

   private:
    CUresult check() const noexcept {
        CUcontext current{};
        const auto proof = context.current(&current);
        return proof != CUDA_SUCCESS       ? proof
               : current == context.native ? CUDA_SUCCESS
                                           : CUDA_ERROR_INVALID_CONTEXT;
    }
    void fail(CUresult proof) noexcept {
        result->result.disposition = CudaDnnDisposition::kRejected;
        result->result.status = proof;
        result->has_context_loss = result->is_cleanup_uncertain = true;
        completion_.status = proof;
        completion_.is_cleanup_uncertain = true;
    }
    cudnnStatus_t invoke() const noexcept {
        const auto rnn = static_cast<CudaDnnRnn>(descriptors.rnn.object.native);
        const auto filter = static_cast<CudaDnnFilterHandle>(descriptors.filter.native);
        const auto array = [&](std::size_t index) {
            return problem.has_descriptors[index] ? arrays_[index].data() : nullptr;
        };
        const auto state = [&](std::size_t index) {
            return static_cast<cudnnTensorDescriptor_t>(descriptors.states[index].native);
        };
        const auto pointer = [&](std::size_t index) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr): exact owned allocation lease.
            return reinterpret_cast<void*>(buffers[index].address);
        };
        const auto work = static_cast<std::size_t>(problem.workspace_bytes);
        const auto reserve = static_cast<std::size_t>(problem.reserve_bytes);
        if (problem.kind == CudaDnnLegacyComputeKind::kForwardTraining)
            return sdk->legacy.forward(
                handle, rnn, problem.sequence_length, array(0), pointer(0), state(0), pointer(4),
                state(1), pointer(5), filter, pointer(10), array(1), pointer(1), state(2),
                pointer(6), state(3), pointer(7), pointer(11), work, pointer(12), reserve);
        if (problem.kind == CudaDnnLegacyComputeKind::kBackwardData)
            return sdk->legacy.backward_data(
                handle, rnn, problem.sequence_length, array(0), pointer(1), array(1), pointer(2),
                state(2), pointer(6), state(3), pointer(7), filter, pointer(10), state(0),
                pointer(4), state(1), pointer(5), array(2), pointer(3), state(4), pointer(8),
                state(5), pointer(9), pointer(11), work, pointer(12), reserve);
        return sdk->legacy.backward_weights(
            handle, rnn, problem.sequence_length, array(0), pointer(0), state(0), pointer(4),
            array(1), pointer(1), pointer(11), work, filter, pointer(10), pointer(12), reserve);
    }
    const std::shared_ptr<const CudaDnnDispatch> sdk;
    const cudnnHandle_t handle;
    const LegacyRnnParameters problem;
    const CudaDnnLegacyView descriptors;
    const std::array<CudaDeviceMemoryView, 13> buffers;
    const CudaDnnFindContext context;
    const std::shared_ptr<CudaDnnRnnCompletion> result;
    const std::array<CudaDeviceMemoryView, 4> dropout;
    std::array<std::pmr::vector<cudnnTensorDescriptor_t>, 3> arrays_;
    CudaExecutionLibraryCompletion completion_{CUDA_SUCCESS, false, false};
};
#endif
}  // namespace
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_legacy_call(
    std::shared_ptr<const CudaDnnDispatch> sdk, cudnnHandle_t handle,
    const CudaSessionDnnLegacyRnn& problem, CudaDnnLegacyView descriptors,
    std::array<CudaDeviceMemoryView, 13> buffers, CudaDnnFindContext context,
    std::shared_ptr<CudaDnnRnnCompletion> result, std::array<CudaDeviceMemoryView, 4> dropout) {
#if CUDNN_MAJOR == 8
    return std::make_shared<LegacyRnnCall>(std::move(sdk), handle, problem, std::move(descriptors),
                                           std::move(buffers), context, std::move(result),
                                           std::move(dropout));
#else
    (void)sdk;
    (void)handle;
    (void)problem;
    (void)descriptors;
    (void)buffers;
    (void)context;
    (void)result;
    (void)dropout;
    return {};
#endif
}
}  // namespace glimmer::backend::cuda
