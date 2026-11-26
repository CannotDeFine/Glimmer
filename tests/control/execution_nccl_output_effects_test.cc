#include "nccl_output_effects.h"

#include <array>
#include <bit>
#include <iostream>
#include <stdexcept>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void independent_effects() {
    const auto producer = make_pinned_nccl_effect_producer();
    expect(producer.produce && producer.lifetime &&
               producer.output_contract == CudaNcclNativeOutputContract::kPinnedHostStable,
           "trusted producer selects the distinct pinned contract");
    ExecutionNcclRequest request;
    request.call = ExecutionNcclInitRank{true, 1, 0, {}};
    std::optional<CudaNcclNativeObservation> native{CudaNcclNativeObservation{}};
    native->operation = CudaNcclNativeOperation::kInitRank;
    native->has_output = true;
    native->output_written = true;
    for (const auto status :
         std::array<std::int32_t, 10>{0, 1, 2, 3, 4, 5, 6, 7, INT32_MIN, INT32_MAX}) {
        ExecutionNcclObservation result;
        const NcclActorEffectInput input{request, status, native, {9, 17}, false};
        expect(producer.produce(nullptr, input, result) == CUDA_SUCCESS && result.effects == 2 &&
                   result.communicator == ExecutionNcclCommunicator{},
               "equal NULL write is an effect independently of every original status");
    }
    native->output_written = false;
    ExecutionNcclObservation result;
    expect(
        producer.produce(nullptr, {request, 0, native, {9, 17}, false}, result) == CUDA_SUCCESS &&
            result.effects == 0,
        "success alone never fabricates a missing write");
    native->output_written.reset();
    result.effects = 4;
    expect(producer.produce(nullptr, {request, 4, native, {9, 17}, false}, result) ==
                   CUDA_ERROR_UNKNOWN &&
               result.effects == 4,
           "byte-difference evidence alone cannot authorize public projection");
    native.reset();
    expect(producer.produce(nullptr, {request, 0, native, {9, 17}, true}, result) ==
                   CUDA_ERROR_NOT_READY &&
               result.effects == 4,
           "missing grouped observation never means no output");
}
void whole_fields() {
    const auto producer = make_pinned_nccl_effect_producer();
    ExecutionNcclRequest request;
    std::optional<CudaNcclNativeObservation> native{CudaNcclNativeObservation{}};
    native->has_output = true;
    native->output_written = true;
    native->operation = CudaNcclNativeOperation::kGetUniqueId;
    native->unique_id.fill(std::byte{0xa5});
    request.call = ExecutionNcclGetUniqueId{true};
    ExecutionNcclObservation result;
    expect(producer.produce(nullptr, {request, 3, native, {}, false}, result) == CUDA_SUCCESS &&
               result.effects == 1 && result.unique_id == native->unique_id &&
               result.unique_id_mask ==
                   std::array<std::byte, 16>{
                       std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255},
                       std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255},
                       std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255},
                       std::byte{255}, std::byte{255}, std::byte{255}, std::byte{255}},
           "all equal poison UID bytes project under independent actual-write evidence");
    request.call = ExecutionNcclGetAsyncError{{1, 2}, true};
    native->operation = CudaNcclNativeOperation::kGetAsyncError;
    native->async_error = std::bit_cast<std::int32_t>(UINT32_C(0xa5a5a5a5));
    expect(producer.produce(nullptr, {request, 2, native, {}, false}, result) == CUDA_SUCCESS &&
               result.effects == 4 && result.async_error_mask == 15 &&
               result.async_error == native->async_error,
           "full raw async bits survive an equal-value write and requested error");
    request.call = ExecutionNcclInitRank{true, 1, 0, {}};
    native->operation = CudaNcclNativeOperation::kInitRank;
    char object{};
    native->communicator = reinterpret_cast<ncclComm_t>(&object);
    expect(
        producer.produce(nullptr, {request, 4, native, {8, 11}, false}, result) == CUDA_SUCCESS &&
            result.effects == 2 && result.communicator == ExecutionNcclCommunicator{8, 11},
        "nonnull native output maps only to the exact reserved symbolic token");
    const auto before = result;
    expect(producer.produce(nullptr, {request, 0, native, {}, false}, result) ==
                   CUDA_ERROR_INVALID_HANDLE &&
               result == before,
           "missing reserved token rejects transactionally without raw pointer publication");
}
}  // namespace
int main() {
    try {
        independent_effects();
        whole_fields();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
