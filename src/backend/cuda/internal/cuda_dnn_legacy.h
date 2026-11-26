#pragma once

#include "cuda_dnn_execution.h"
#include "../../internal/execution_submission_queue.h"

namespace glimmer::backend::cuda {
struct CudaDnnLegacyDescriptorArray {
    std::uint32_t has_descriptors{};
    std::vector<std::uint64_t> descriptors;
};
enum class CudaDnnLegacyMetadataKind : std::uint32_t {
    kVersionCheck = 1,
    kSetAlgorithm,
    kSetMath,
    kGetMath,
    kSetBias,
    kGetBias
};
enum class CudaDnnLegacyVersionCheck : std::uint32_t {
    kOpsInfer = 1,
    kOpsTrain,
    kCnnInfer,
    kCnnTrain,
    kAdvInfer,
    kAdvTrain
};
struct CudaDnnLegacyMetadata {
    CudaDnnLegacyMetadataKind kind{};
    std::uint64_t object{};
    std::uint32_t value{}, has_output{}, seed{};
};
struct CudaDnnLegacySetup {
    std::int32_t hidden_size{}, layers{};
    std::uint64_t dropout{};
    std::uint32_t input_mode{}, direction{}, cell_mode{}, algorithm{}, math_precision{};
};
struct CudaDnnLegacyParameterQuery {
    std::uint64_t input{}, filter{}, destination{};
    std::int32_t pseudo_layer{}, linear_layer{};
    CudaAllocationRange weights{};
};
enum class CudaDnnLegacyControlKind : std::uint32_t {
    kSetup = 1,
    kClip,
    kWorkspaceSize,
    kReserveSize,
    kParameterSize,
    kMatrixParameter,
    kBiasParameter,
    kSetAlgorithm,
    kInferenceMaxCount
};
struct CudaDnnLegacyControl {
    CudaDnnLegacyControlKind kind{};
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::uint32_t has_output{};
    std::uint64_t seed{};
    CudaDnnLegacySetup setup{};
    CudaDnnRnnClipState clip{};
    std::int32_t sequence_length{};
    CudaDnnLegacyDescriptorArray sequence{};
    std::uint64_t input{};
    std::uint32_t data_type{};
    CudaDnnLegacyParameterQuery parameters{};
    std::uint64_t algorithm{};
};
struct CudaDnnLegacyOutput {
    std::uint32_t written{};
    std::uint64_t scalar{};
    std::uint32_t non_null{};
    std::uint64_t offset{}, extent{};
};
struct CudaDnnLegacyResult {
    CudaDnnResult result{};
    CudaDnnLegacyOutput output{};
    bool has_metadata_quarantine{}, has_context_loss{};
};
enum class CudaDnnLegacyComputeKind : std::uint32_t {
    kForwardTraining = 1,
    kBackwardData,
    kBackwardWeights
};
struct CudaSessionDnnLegacyRnn {
    CudaDnnLegacyComputeKind kind{};
    std::uint64_t handle{}, metadata{}, descriptor{};
    std::int32_t sequence_length{};
    std::array<CudaDnnLegacyDescriptorArray, 3> sequences;
    std::array<std::uint64_t, 6> states{};
    std::uint64_t filter{};
    std::array<CudaAllocationRange, 13> buffers{};
    std::uint64_t workspace_bytes{}, reserve_bytes{};
};
// One transaction retains positional aliases and all dependency owners. The
// native arrays never contain frontend pointers or unborrowed descriptor IDs.
struct CudaDnnLegacyView {
    // Declared first so all PMR vectors are destroyed before their resource.
    std::shared_ptr<ExecutionHostStorage> storage;
    CudaDnnRnnView rnn;
    std::array<CudaDnnObjectView, 2> algorithms;
    std::array<std::pmr::vector<CudaDnnObjectView>, 3> sequences;
    std::array<CudaDnnObjectView, 6> states;
    CudaDnnObjectView filter;
    explicit CudaDnnLegacyView(std::shared_ptr<ExecutionHostStorage> owner = {})
        : storage(std::move(owner)),
          sequences{std::pmr::vector<CudaDnnObjectView>{resource()},
                    std::pmr::vector<CudaDnnObjectView>{resource()},
                    std::pmr::vector<CudaDnnObjectView>{resource()}} {}
    CudaDnnLegacyView(CudaDnnLegacyView&&) noexcept = default;
    CudaDnnLegacyView(const CudaDnnLegacyView&) = delete;
    CudaDnnLegacyView& operator=(const CudaDnnLegacyView&) = delete;
    CudaDnnLegacyView& operator=(CudaDnnLegacyView&&) = delete;
    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept {
        return storage ? storage->resource() : std::pmr::null_memory_resource();
    }
};
struct CudaDnnLegacyExtent {
    std::size_t bytes{}, elements{}, width{};
};
// Validates counts and computes all variable arena allocations without copying
// IDs or borrowing descriptors. Absent arrays and nonpositive lengths are kept.
[[nodiscard]] std::optional<std::size_t> cuda_dnn_legacy_storage_bytes(
    const CudaSessionDnnLegacyRnn&) noexcept;
struct CudaDnnLegacyBorrowResult {
    CudaDnnResult result{};
    CudaDnnLegacyView view;
};
struct CudaDnnLegacyContext {
    CUresult (*check)(const void*) noexcept {};
    const void* opaque{};
    const CudaExecutionSession* session{};
};
[[nodiscard]] std::shared_ptr<CudaExecutionLibraryCall> make_cuda_dnn_legacy_call(
    std::shared_ptr<const CudaDnnDispatch>, cudnnHandle_t, const CudaSessionDnnLegacyRnn&,
    CudaDnnLegacyView, std::array<CudaDeviceMemoryView, 13>, CudaDnnFindContext,
    std::shared_ptr<CudaDnnRnnCompletion>, std::array<CudaDeviceMemoryView, 4>);
}  // namespace glimmer::backend::cuda
