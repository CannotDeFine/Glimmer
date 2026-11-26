#include "internal/cuda_dnn_backend_execution.h"

#include <cstring>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace glimmer::backend::cuda {
namespace {
constexpr std::size_t kMaxEntries = 100;
constexpr std::size_t kMaxHostStorage = std::size_t{2} << 20;
class BackendCall;
struct PackRegistration {
    cudnnBackendDescriptor_t native{};
    BackendCall* owner{};
};
// Native descriptor creation must not alias another live owner, even across
// execution contexts on this thread. Slots are acquired before any SDK issue.
thread_local std::vector<PackRegistration> pack_registrations;
std::size_t tensor_storage_bits(std::uint32_t type) noexcept {
    switch (type) {
        case 0:
        case 4:
        case 14:
            return 32;
        case 1:
        case 10:
            return 64;
        case 2:
        case 9:
            return 16;
        case 3:
        case 6:
        case 12:
        case 13:
        case 15:
            return 8;
        default:
            return 0;
    }
}
bool multiply(std::size_t& value, std::size_t factor) noexcept {
    if (factor && value > std::numeric_limits<std::size_t>::max() / factor) return false;
    value *= factor;
    return true;
}
CUresult tensor_requirement(const CudaDnnBackendTensorObservation& tensor,
                            CudaDnnBackendTensorRequirement& output) noexcept {
    if (tensor.dimension_count < 2 || tensor.dimension_count > 8 ||
        tensor.dimension_count != tensor.stride_count || tensor.alignment.bytes <= 0 ||
        static_cast<std::uint64_t>(tensor.alignment.bytes) > SIZE_MAX || tensor.vector_count <= 0)
        return CUDA_ERROR_INVALID_VALUE;
    const auto bits = tensor_storage_bits(tensor.type);
    // Special swizzled storage needs an independent footprint contract; never
    // treat an unqualified layout as permission to read a smaller allocation.
    if (!bits || tensor.reordering) return CUDA_ERROR_NOT_SUPPORTED;
    if (tensor.vector_count != 1 &&
        (tensor.type != 3 || (tensor.vector_count != 4 && tensor.vector_count != 32) ||
         tensor.vectorized_dimension < 0 || tensor.vectorized_dimension >= tensor.dimension_count))
        return CUDA_ERROR_NOT_SUPPORTED;
    bool is_empty{};
    for (std::size_t i = 0; i != tensor.dimension_count; ++i) {
        const auto dimension = tensor.dimensions[i], stride = tensor.strides[i];
        if (dimension < 0 || stride < 0 || static_cast<std::uint64_t>(dimension) > SIZE_MAX ||
            static_cast<std::uint64_t>(stride) > SIZE_MAX)
            return CUDA_ERROR_INVALID_VALUE;
        is_empty |= dimension == 0;
    }
    std::size_t elements{is_empty ? 0U : 1U};
    if (!is_empty)
        for (std::size_t i = 0; i != tensor.dimension_count; ++i) {
            auto term = static_cast<std::size_t>(tensor.dimensions[i] - 1);
            if (!multiply(term, static_cast<std::size_t>(tensor.strides[i])) ||
                term > SIZE_MAX - elements)
                return CUDA_ERROR_INVALID_VALUE;
            elements += term;
        }
    if (!multiply(elements, static_cast<std::size_t>(tensor.vector_count)) ||
        !multiply(elements, bits))
        return CUDA_ERROR_INVALID_VALUE;
    output = {tensor.uid, elements / 8 + (elements % 8 != 0),
              static_cast<std::size_t>(tensor.alignment.bytes), tensor.is_by_value};
    return CUDA_SUCCESS;
}
class CallScope final {
   public:
    explicit CallScope(CudaDnnRnnLease& lease) noexcept
        : lease_(lease), has_call(lease_.begin_call()) {}
    ~CallScope() {
        if (has_call) lease_.end_call();
    }
    [[nodiscard]] bool is_active() const noexcept {
        return has_call;
    }

   private:
    CudaDnnRnnLease& lease_;
    const bool has_call;
};
class BackendCall final : public CudaExecutionLibraryCall {
   public:
    BackendCall(std::shared_ptr<const CudaDnnDispatch> dispatch_value, cudnnHandle_t handle_value,
                CudaDnnBackendExecutionArguments arguments_value, CudaDnnFindContext context_value,
                std::shared_ptr<CudaDnnBackendExecutionCompletion> result_value)
        : dispatch(std::move(dispatch_value)),
          handle(handle_value),
          arguments(std::move(arguments_value)),
          context(context_value),
          result(std::move(result_value)) {
        if (!dispatch || !handle || !context.native || !context.current || !context.synchronize ||
            !result || !arguments.plan.call || !arguments.plan.descriptor.object.native ||
            arguments.plan.descriptor.object.kind != CudaDnnObjectKind::kBackend ||
            !arguments.plan.descriptor.object.lifetime ||
            arguments.uids.size() != arguments.pointers.size() ||
            arguments.uids.size() > kMaxEntries || !dispatch->backend_create ||
            !dispatch->backend_set || !dispatch->backend_finalize || !dispatch->backend_destroy ||
            !dispatch->backend_execute || !dispatch->get_stream)
            throw std::invalid_argument("invalid trusted backend execution capsule");
        pointers_.resize(arguments.pointers.size());
        host_storage_.resize(arguments.pointers.size());
        std::size_t retained{};
        for (std::size_t i = 0; i != arguments.pointers.size(); ++i) {
            const auto& pointer = arguments.pointers[i];
            if (pointer.kind == CudaDnnBackendPointerKind::kNull) {
                if (pointer.device.address || pointer.device.bytes || pointer.device.lifetime ||
                    !pointer.value.empty())
                    throw std::invalid_argument("noncanonical backend NULL binding");
            } else if (pointer.kind == CudaDnnBackendPointerKind::kDevice) {
                if (!pointer.device.address || !pointer.device.lifetime || !pointer.value.empty())
                    throw std::invalid_argument("unowned backend device binding");
                // NOLINTNEXTLINE(performance-no-int-to-ptr): validated server-owned UVA.
                pointers_[i] = reinterpret_cast<void*>(pointer.device.address);
            } else if (pointer.kind == CudaDnnBackendPointerKind::kHostValue) {
                if (pointer.device.address || pointer.device.bytes || pointer.device.lifetime ||
                    pointer.value.empty() || !pointer.alignment ||
                    pointer.alignment > kMaxHostStorage || pointer.value.size() > kMaxHostStorage ||
                    pointer.value.size() > kMaxHostStorage - (pointer.alignment - 1))
                    throw std::invalid_argument("invalid backend host-value storage");
                const auto bytes = pointer.value.size() + pointer.alignment - 1;
                if (bytes > kMaxHostStorage - retained)
                    throw std::length_error("backend host-value storage quota exceeded");
                retained += bytes;
                auto& storage = host_storage_[i];
                storage.resize(bytes);
                const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
                const auto remainder = address % pointer.alignment;
                const auto offset = remainder ? pointer.alignment - remainder : 0;
                pointers_[i] = storage.data() + offset;
                std::memcpy(pointers_[i], pointer.value.data(), pointer.value.size());
            } else {
                throw std::invalid_argument("unknown backend pointer binding");
            }
        }
        if ((arguments.workspace.address && !arguments.workspace.lifetime) ||
            (!arguments.workspace.address &&
             (arguments.workspace.bytes || arguments.workspace.lifetime)))
            throw std::invalid_argument("unowned backend workspace");
        pack_registrations.push_back({nullptr, this});
    }
    ~BackendCall() override {
        for (auto item = pack_registrations.begin(); item != pack_registrations.end(); ++item)
            if (item->owner == this) {
                if (item->native)
                    item->owner = nullptr;
                else
                    pack_registrations.erase(item);
                break;
            }
    }

    std::int32_t issue(CUstream expected, std::span<const std::byte>) noexcept override {
        if (has_issue_attempt_) return CUDNN_STATUS_INTERNAL_ERROR;
        has_issue_attempt_ = true;
        const CallScope scope{*arguments.plan.call};
        if (!scope.is_active()) {
            fail_preparation(CUDA_ERROR_NOT_READY);
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        if (!prove_context()) return CUDNN_STATUS_INTERNAL_ERROR;
        const auto created =
            dispatch->backend_create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &pack_);
        bool is_duplicate = pack_ && arguments.plan.call->owns_backend_pointer(pack_);
        for (auto& entry : pack_registrations)
            if (pack_ && entry.owner != this && entry.native == pack_) {
                is_duplicate = true;
                if (entry.owner) entry.owner->fail_uncertain(CUDA_ERROR_UNKNOWN);
            }
        for (auto& entry : pack_registrations)
            if (entry.owner == this) entry.native = pack_;
        // A contradictory/partial Create output is not a known descriptor that
        // can safely be passed to Destroy, even after the stream has stopped.
        if (is_duplicate || (created != CUDNN_STATUS_SUCCESS && pack_) ||
            (created == CUDNN_STATUS_SUCCESS && !pack_)) {
            if (is_duplicate) arguments.plan.call->quarantine();
            result->preparation_status = created;
            if (prove_context()) fail_uncertain(CUDA_ERROR_UNKNOWN);
            return CUDNN_STATUS_INTERNAL_ERROR;
        }
        if (!private_result(created)) return CUDNN_STATUS_INTERNAL_ERROR;
        const auto count = static_cast<std::int64_t>(arguments.uids.size());
        // Empty-but-present arrays remain distinct from NULL when count is zero.
        std::int64_t empty_uid{};
        void* empty_pointer{};
        const auto* uids = count ? arguments.uids.data() : &empty_uid;
        auto* pointers = count ? pointers_.data() : &empty_pointer;
        if (!private_result(dispatch->backend_set(pack_, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                                  CUDNN_TYPE_INT64, count, uids)) ||
            !private_result(dispatch->backend_set(pack_, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                                  CUDNN_TYPE_VOID_PTR, count, pointers)))
            return CUDNN_STATUS_INTERNAL_ERROR;
        // NOLINTNEXTLINE(performance-no-int-to-ptr): validated server-owned UVA.
        auto* workspace = reinterpret_cast<void*>(arguments.workspace.address);
        if (!private_result(dispatch->backend_set(pack_, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                                  CUDNN_TYPE_VOID_PTR, 1, &workspace)) ||
            !private_result(dispatch->backend_finalize(pack_)))
            return CUDNN_STATUS_INTERNAL_ERROR;
        completion_.was_native_issued = true;
        const auto native = dispatch->backend_execute(
            handle, static_cast<cudnnBackendDescriptor_t>(arguments.plan.descriptor.object.native),
            pack_);
        result->result.disposition = CudaDnnDisposition::kComplete;
        result->result.status = CUDA_SUCCESS;
        result->result.original_status = native;
        if (!prove_context()) return native;
        if (native != CUDNN_STATUS_SUCCESS) {
            const auto synchronized = context.synchronize();
            if (!prove_context()) return native;
            if (synchronized != CUDA_SUCCESS) {
                fail_uncertain(synchronized);
                return native;
            }
        }
        cudaStream_t observed{};
        const auto stream_status = dispatch->get_stream(handle, &observed);
        if (!prove_context()) return native;
        if (stream_status != CUDNN_STATUS_SUCCESS ||
            reinterpret_cast<CUstream>(observed) != expected)
            fail_uncertain(CUDA_ERROR_UNKNOWN);
        result->is_ready = true;
        return native;
    }
    CudaExecutionLibraryCompletion completion() const noexcept override {
        return completion_;
    }
    CUresult cleanup_after_physical_stop() noexcept override {
        if (completion_.is_cleanup_uncertain) return completion_.status;
        if (!pack_) return CUDA_SUCCESS;
        const CallScope scope{*arguments.plan.call};
        if (!scope.is_active() || has_destroy_attempt_) return CUDA_ERROR_UNKNOWN;
        if (!prove_context()) return completion_.status;
        has_destroy_attempt_ = true;
        const auto status = dispatch->backend_destroy(pack_);
        const auto is_matching = prove_context();
        if (status != CUDNN_STATUS_SUCCESS || !is_matching) {
            fail_uncertain(is_matching ? CUDA_ERROR_UNKNOWN : completion_.status);
            return completion_.status;
        }
        pack_ = nullptr;
        for (auto& entry : pack_registrations)
            if (entry.owner == this) entry.native = nullptr;
        return CUDA_SUCCESS;
    }

   private:
    void fail_preparation(CUresult status) noexcept {
        completion_.status = status;
        result->result.disposition = CudaDnnDisposition::kRejected;
        result->result.status = status;
        result->is_ready = true;
    }
    void fail_uncertain(CUresult status, bool has_context_loss = false) noexcept {
        completion_.status = status;
        completion_.is_cleanup_uncertain = true;
        result->result.disposition = CudaDnnDisposition::kRejected;
        result->result.status = status;
        result->has_context_loss |= has_context_loss;
        result->is_cleanup_uncertain = true;
        result->is_ready = true;
    }
    bool prove_context() noexcept {
        CUcontext current{};
        const auto status = context.current(&current);
        if (status != CUDA_SUCCESS || current != context.native) {
            fail_uncertain(status != CUDA_SUCCESS ? status : CUDA_ERROR_INVALID_CONTEXT, true);
            return false;
        }
        return true;
    }
    bool private_result(cudnnStatus_t status) noexcept {
        result->preparation_status = status;
        if (!prove_context()) return false;
        if (status == CUDNN_STATUS_SUCCESS) return true;
        fail_preparation(CUDA_ERROR_UNKNOWN);
        return false;
    }
    const std::shared_ptr<const CudaDnnDispatch> dispatch;
    const cudnnHandle_t handle;
    const CudaDnnBackendExecutionArguments arguments;
    const CudaDnnFindContext context;
    const std::shared_ptr<CudaDnnBackendExecutionCompletion> result;
    std::vector<void*> pointers_;
    std::vector<std::vector<std::byte>> host_storage_;
    cudnnBackendDescriptor_t pack_{};
    CudaExecutionLibraryCompletion completion_{CUDA_SUCCESS, false, false};
    bool has_issue_attempt_{}, has_destroy_attempt_{};
};
}  // namespace
CudaDnnBackendTensorRequirements cuda_dnn_backend_tensor_requirements(
    const CudaDnnBackendSnapshot& snapshot) noexcept {
    CudaDnnBackendTensorRequirements output;
    try {
        std::vector<const CudaDnnBackendSnapshot*> pending{&snapshot}, seen;
        while (!pending.empty()) {
            const auto* current = pending.back();
            pending.pop_back();
            if (std::ranges::find(seen, current) != seen.end()) continue;
            if (seen.size() == 4096) {
                output.status = CUDA_ERROR_OUT_OF_MEMORY;
                return output;
            }
            seen.push_back(current);
            if (!current->is_valid || !current->is_finalized) {
                output.status = CUDA_ERROR_INVALID_VALUE;
                return output;
            }
            if (current->kind == 17) {
                if (!current->tensor) {
                    output.status = CUDA_ERROR_NOT_SUPPORTED;
                    return output;
                }
                if (current->tensor->is_virtual) continue;
                CudaDnnBackendTensorRequirement requirement;
                output.status = tensor_requirement(*current->tensor, requirement);
                if (output.status != CUDA_SUCCESS) return output;
                const auto prior = std::ranges::find(output.tensors, requirement.uid,
                                                     &CudaDnnBackendTensorRequirement::uid);
                if (prior == output.tensors.end()) {
                    output.tensors.push_back(requirement);
                } else {
                    if (prior->is_by_value != requirement.is_by_value) {
                        output.status = CUDA_ERROR_INVALID_VALUE;
                        return output;
                    }
                    prior->bytes = std::max(prior->bytes, requirement.bytes);
                    auto alignment =
                        prior->alignment / std::gcd(prior->alignment, requirement.alignment);
                    if (!multiply(alignment, requirement.alignment)) {
                        output.status = CUDA_ERROR_INVALID_VALUE;
                        return output;
                    }
                    prior->alignment = alignment;
                }
            }
            if (current->generated_graph) {
                if (pending.size() == 4096) {
                    output.status = CUDA_ERROR_OUT_OF_MEMORY;
                    return output;
                }
                pending.push_back(current->generated_graph.get());
            }
            for (const auto& attribute : current->attributes)
                for (const auto& child : attribute.descriptors)
                    if (child) {
                        // These rows copy CPU device descriptions, not graph
                        // tensors. Their header cannot grant or withhold ranges.
                        if (child->kind == 38 && attribute.type == 15 &&
                            ((current->kind == 4 && attribute.name == 204) ||
                             (current->kind == 2 && attribute.name == 1307) ||
                             (current->kind == 5 && attribute.name == 407)))
                            continue;
                        if (pending.size() == 4096) {
                            output.status = CUDA_ERROR_OUT_OF_MEMORY;
                            return output;
                        }
                        pending.push_back(child.get());
                    }
        }
        output.status = CUDA_SUCCESS;
    } catch (...) {
        output.status = CUDA_ERROR_OUT_OF_MEMORY;
    }
    return output;
}
std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_backend_call(
    std::shared_ptr<const CudaDnnDispatch> dispatch, cudnnHandle_t handle,
    CudaDnnBackendExecutionArguments arguments, CudaDnnFindContext context,
    std::shared_ptr<CudaDnnBackendExecutionCompletion> result) {
    return std::make_shared<BackendCall>(std::move(dispatch), handle, std::move(arguments), context,
                                         std::move(result));
}
}  // namespace glimmer::backend::cuda
