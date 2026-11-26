#include "nccl_output_effects.h"

#include <algorithm>

namespace glimmer::app {
namespace {
namespace wire = control;
using Kind = backend::cuda::CudaNcclNativeOperation;
bool has_output(const wire::ExecutionNcclCall& call) noexcept {
    if (const auto* value = std::get_if<wire::ExecutionNcclGetUniqueId>(&call))
        return value->has_output;
    if (const auto* value = std::get_if<wire::ExecutionNcclInitRank>(&call))
        return value->has_output;
    if (const auto* value = std::get_if<wire::ExecutionNcclInitRankConfig>(&call))
        return value->initialization.has_output;
    if (const auto* value = std::get_if<wire::ExecutionNcclGetAsyncError>(&call))
        return value->has_output;
    return false;
}
CUresult produce(void*, const NcclActorEffectInput& input,
                 wire::ExecutionNcclObservation& destination) noexcept {
    if (input.request.call.valueless_by_exception()) return CUDA_ERROR_INVALID_VALUE;
    const auto expected = static_cast<Kind>(input.request.call.index() + 1);
    if (input.native && input.native->operation != expected) return CUDA_ERROR_INVALID_VALUE;
    wire::ExecutionNcclObservation result;
    if (!has_output(input.request.call)) {
        if (input.native && input.native->has_output) return CUDA_ERROR_UNKNOWN;
        destination = result;
        return CUDA_SUCCESS;
    }
    // In particular a missing grouped-inner snapshot is not a no-write proof.
    if (!input.native) return CUDA_ERROR_NOT_READY;
    const auto& native = *input.native;
    if (!native.has_output || !native.output_written) return CUDA_ERROR_UNKNOWN;
    if (!*native.output_written) {
        if (native.communicator_changed || native.async_error_changed ||
            std::ranges::any_of(native.unique_id_changed,
                                [](std::byte value) { return value != std::byte{}; }))
            return CUDA_ERROR_UNKNOWN;
        destination = result;
        return CUDA_SUCCESS;
    }
    switch (expected) {
        case Kind::kGetUniqueId:
            result.effects = 1;
            result.unique_id_mask.fill(std::byte{255});
            result.unique_id = native.unique_id;
            break;
        case Kind::kInitRank:
        case Kind::kInitRankConfig:
            result.effects = 2;
            if (native.communicator) {
                if (!input.reserved_communicator.domain || !input.reserved_communicator.object)
                    return CUDA_ERROR_INVALID_HANDLE;
                result.communicator = input.reserved_communicator;
            }
            break;
        case Kind::kGetAsyncError:
            result.effects = 4;
            result.async_error_mask = 15;
            result.async_error = native.async_error;
            break;
        default:
            return CUDA_ERROR_UNKNOWN;
    }
    destination = result;
    return CUDA_SUCCESS;
}
}  // namespace

NcclActorEffectProducer make_pinned_nccl_effect_producer() {
    NcclActorEffectProducer producer;
    producer.produce = produce;
    producer.lifetime = std::make_shared<const int>(0);
    producer.output_contract = backend::cuda::CudaNcclNativeOutputContract::kPinnedHostStable;
    return producer;
}
}  // namespace glimmer::app
