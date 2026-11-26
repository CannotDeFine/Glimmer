#pragma once

#include <cuda.h>
#include <cudnn.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "../../../internal/cudnn_sdk_profile.h"
#include "cuda_dnn_legacy_dispatch.h"

namespace glimmer::backend {
class ExecutionHostStorage;
}
namespace glimmer::backend::cuda {
// The installed SDK deprecates the legacy typedef and entry points, not the
// underlying ABI. Keep compatibility warnings local to the dispatch definition.
using CudaDnnActivation = cudnnActivationStruct*;
using CudaDnnConvolution = cudnnConvolutionStruct*;
using CudaDnnFilterHandle = cudnnFilterStruct*;
using CudaDnnPooling = cudnnPoolingStruct*;
using CudaDnnLrn = cudnnLRNStruct*;
using CudaDnnReduction = cudnnReduceTensorStruct*;
using CudaDnnDropout = cudnnDropoutStruct*;
using CudaDnnOpTensor = cudnnOpTensorStruct*;
using CudaDnnTensorTransform = cudnnTensorTransformStruct*;
using CudaDnnRnnData = cudnnRNNDataStruct*;
using CudaDnnSeqData = cudnnSeqDataStruct*;
using CudaDnnRnn = cudnnRNNStruct*;
using CudaDnnAttn = cudnnAttnStruct*;
inline constexpr std::size_t kCudaDnnMaxSequenceLengths = std::size_t{1} << 20;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
struct CudaDnnDispatch {
    decltype(&cudnnCreate) create{};
    decltype(&cudnnDestroy) destroy{};
    decltype(&cudnnSetStream) set_stream{};
    decltype(&cudnnGetStream) get_stream{};
    decltype(&cudnnCreateTensorDescriptor) tensor_create{};
    decltype(&cudnnDestroyTensorDescriptor) tensor_destroy{};
    decltype(&cudnnSetTensor4dDescriptor) tensor_set_4d{};
    decltype(&cudnnSetTensor4dDescriptorEx) tensor_set_4d_ex{};
    decltype(&cudnnGetTensor4dDescriptor) tensor_get_4d{};
    decltype(&cudnnSetTensorNdDescriptor) tensor_set_nd{};
    decltype(&cudnnSetTensorNdDescriptorEx) tensor_set_nd_ex{};
    decltype(&cudnnGetTensorNdDescriptor) tensor_get_nd{};
    decltype(&cudnnGetTensorSizeInBytes) tensor_size{};
    decltype(&cudnnCreateActivationDescriptor) activation_create{};
    decltype(&cudnnDestroyActivationDescriptor) activation_destroy{};
    decltype(&cudnnSetActivationDescriptor) activation_set{};
    decltype(&cudnnGetActivationDescriptor) activation_get{};
    decltype(&cudnnActivationForward) activation_forward{};
    decltype(&cudnnAddTensor) add_tensor{};
    decltype(&cudnnCreateFilterDescriptor) filter_create{};
    decltype(&cudnnDestroyFilterDescriptor) filter_destroy{};
    decltype(&cudnnSetFilter4dDescriptor) filter_set_4d{};
    decltype(&cudnnGetFilter4dDescriptor) filter_get_4d{};
    decltype(&cudnnSetFilterNdDescriptor) filter_set_nd{};
    decltype(&cudnnGetFilterNdDescriptor) filter_get_nd{};
    decltype(&cudnnGetFilterSizeInBytes) filter_size{};
    decltype(&cudnnCreateConvolutionDescriptor) convolution_create{};
    decltype(&cudnnDestroyConvolutionDescriptor) convolution_destroy{};
    decltype(&cudnnSetConvolution2dDescriptor) convolution_set_2d{};
    decltype(&cudnnGetConvolution2dDescriptor) convolution_get_2d{};
    decltype(&cudnnSetConvolutionNdDescriptor) convolution_set_nd{};
    decltype(&cudnnGetConvolutionNdDescriptor) convolution_get_nd{};
    decltype(&cudnnSetConvolutionGroupCount) convolution_group_set{};
    decltype(&cudnnGetConvolutionGroupCount) convolution_group_get{};
    decltype(&cudnnSetConvolutionMathType) convolution_math_set{};
    decltype(&cudnnGetConvolutionMathType) convolution_math_get{};
    decltype(&cudnnSetConvolutionReorderType) convolution_reorder_set{};
    decltype(&cudnnGetConvolutionReorderType) convolution_reorder_get{};
    decltype(&cudnnGetConvolution2dForwardOutputDim) convolution_output_2d{};
    decltype(&cudnnGetConvolutionNdForwardOutputDim) convolution_output_nd{};
    decltype(&cudnnConvolutionForward) convolution_forward{};
    decltype(&cudnnGetConvolutionForwardWorkspaceSize) convolution_forward_workspace{};
    decltype(&cudnnGetConvolutionForwardAlgorithm_v7) convolution_forward_heuristic{};
    decltype(&cudnnGetConvolutionForwardAlgorithmMaxCount) convolution_forward_max_count{};
    decltype(&cudnnFindConvolutionForwardAlgorithm) convolution_forward_find{};
    decltype(&cudnnCreatePoolingDescriptor) pooling_create{};
    decltype(&cudnnDestroyPoolingDescriptor) pooling_destroy{};
    decltype(&cudnnSetPoolingNdDescriptor) pooling_set_nd{};
    decltype(&cudnnGetPoolingNdDescriptor) pooling_get_nd{};
    decltype(&cudnnGetPoolingNdForwardOutputDim) pooling_output_nd{};
    decltype(&cudnnPoolingForward) pooling_forward{};
    decltype(&cudnnSoftmaxForward) softmax_forward{};
    decltype(&cudnnCreateLRNDescriptor) lrn_create{};
    decltype(&cudnnDestroyLRNDescriptor) lrn_destroy{};
    decltype(&cudnnSetLRNDescriptor) lrn_set{};
    decltype(&cudnnLRNCrossChannelForward) lrn_forward{};
    decltype(&cudnnTransformTensor) transform_tensor{};
    decltype(&cudnnCreateReduceTensorDescriptor) reduction_create{};
    decltype(&cudnnDestroyReduceTensorDescriptor) reduction_destroy{};
    decltype(&cudnnSetReduceTensorDescriptor) reduction_set{};
    decltype(&cudnnGetReductionIndicesSize) reduction_indices_size{};
    decltype(&cudnnGetReductionWorkspaceSize) reduction_workspace_size{};
    decltype(&cudnnReduceTensor) reduce_tensor{};
    decltype(&cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize)
        batch_norm_forward_workspace{};
    decltype(&cudnnGetBatchNormalizationBackwardExWorkspaceSize) batch_norm_backward_workspace{};
    decltype(&cudnnGetBatchNormalizationTrainingExReserveSpaceSize) batch_norm_reserve_size{};
    decltype(&cudnnBatchNormalizationForwardTrainingEx) batch_norm_forward{};
    decltype(&cudnnBatchNormalizationBackwardEx) batch_norm_backward{};
    decltype(&cudnnCreateDropoutDescriptor) dropout_create{};
    decltype(&cudnnDestroyDropoutDescriptor) dropout_destroy{};
    decltype(&cudnnDropoutGetStatesSize) dropout_states_size{};
    decltype(&cudnnSetDropoutDescriptor) dropout_set{};
    decltype(&cudnnRestoreDropoutDescriptor) dropout_restore{};
    decltype(&cudnnCreateOpTensorDescriptor) op_tensor_create{};
    decltype(&cudnnDestroyOpTensorDescriptor) op_tensor_destroy{};
    decltype(&cudnnSetOpTensorDescriptor) op_tensor_set{};
    decltype(&cudnnGetOpTensorDescriptor) op_tensor_get{};
    decltype(&cudnnCreateTensorTransformDescriptor) tensor_transform_create{};
    decltype(&cudnnDestroyTensorTransformDescriptor) tensor_transform_destroy{};
    decltype(&cudnnSetTensorTransformDescriptor) tensor_transform_set{};
    decltype(&cudnnInitTransformDest) init_transform_dest{};
    decltype(&cudnnGetVersion) version{};
    decltype(&cudnnReorderFilterAndBias) reorder_filter_bias{};
    decltype(&cudnnGetCudartVersion) cudart_version{};
    decltype(&cudnnGetMaxDeviceVersion) max_device_version{};
    decltype(&cudnnGetProperty) property{};
    decltype(&cudnnGetConvolutionBackwardDataAlgorithmMaxCount)
        convolution_backward_data_max_count{};
    decltype(&cudnnGetConvolutionBackwardFilterAlgorithmMaxCount)
        convolution_backward_filter_max_count{};
    decltype(&cudnnGetConvolutionBackwardDataWorkspaceSize) convolution_backward_data_workspace{};
    decltype(&cudnnGetFoldedConvBackwardDataDescriptors) folded_backward_data{};
    decltype(&cudnnCreateRNNDataDescriptor) rnn_data_create{};
    decltype(&cudnnDestroyRNNDataDescriptor) rnn_data_destroy{};
    decltype(&cudnnSetRNNDataDescriptor) rnn_data_set{};
    decltype(&cudnnCreateSeqDataDescriptor) seq_data_create{};
    decltype(&cudnnDestroySeqDataDescriptor) seq_data_destroy{};
    decltype(&cudnnSetSeqDataDescriptor) seq_data_set{};
    decltype(&cudnnGetSeqDataDescriptor) seq_data_get{};
    decltype(&cudnnCreateRNNDescriptor) rnn_create{};
    decltype(&cudnnDestroyRNNDescriptor) rnn_destroy{};
    decltype(&cudnnSetRNNDescriptor_v8) rnn_set_v8{};
    decltype(&cudnnCreateAttnDescriptor) attn_create{};
    decltype(&cudnnDestroyAttnDescriptor) attn_destroy{};
    decltype(&cudnnSetAttnDescriptor) attn_set{};
    decltype(&cudnnGetRNNWeightSpaceSize) rnn_weight_space_size{};
    decltype(&cudnnGetRNNTempSpaceSizes) rnn_temp_space_sizes{};
    decltype(&cudnnGetMultiHeadAttnBuffers) attn_buffers{};
    decltype(&cudnnRNNSetClip_v8) rnn_set_clip_v8{};
    // Private safety preflight, not a public GetRNNData implementation.
    decltype(&cudnnGetRNNDataDescriptor) rnn_data_get{};
    decltype(&cudnnRNNForward) rnn_forward{};
    decltype(&cudnnRNNBackwardData_v8) rnn_backward_data{};
    decltype(&cudnnRNNBackwardWeights_v8) rnn_backward_weights{};
    decltype(&cudnnBuildRNNDynamic) rnn_build_dynamic{};
    decltype(&cudnnGetRNNWeightParams) rnn_weight_params{};
    decltype(&cudnnGetMultiHeadAttnWeights) attn_weights{};
    decltype(&cudnnMultiHeadAttnForward) attn_forward{};
    decltype(&cudnnMultiHeadAttnBackwardData) attn_backward_data{};
    decltype(&cudnnMultiHeadAttnBackwardWeights) attn_backward_weights{};
    // Private extent observation, not a public GetAttnDescriptor route.
    decltype(&cudnnGetAttnDescriptor) attn_get{};
    decltype(&cudnnBackendCreateDescriptor) backend_create{};
    decltype(&cudnnBackendDestroyDescriptor) backend_destroy{};
    decltype(&cudnnBackendInitialize) backend_initialize{};
    decltype(&cudnnBackendSetAttribute) backend_set{};
    decltype(&cudnnBackendGetAttribute) backend_get{};
    decltype(&cudnnBackendFinalize) backend_finalize{};
    decltype(&cudnnBackendExecute) backend_execute{};
    CudaDnnLegacyDispatch legacy{};
};
#pragma GCC diagnostic pop

enum class CudaDnnObjectKind : std::uint32_t {
    kTensor = 1,
    kActivation,
    kFilter,
    kConvolution,
    kPooling,
    kLrn,
    kReduction,
    kDropout,
    kOpTensor,
    kTensorTransform,
    kRnnData,
    kSeqData,
    kRnn,
    kAttn,
    kBackend,
    kAlgorithm
};
enum class CudaDnnDisposition { kComplete, kRejected, kDeferred };
struct CudaDnnLegacyMetadata;
struct CudaDnnLegacyControl;
struct CudaDnnLegacyResult;
struct CudaDnnLegacyBorrowResult;
struct CudaSessionDnnLegacyRnn;
struct CudaDnnLegacyContext;
enum class CudaDnnLibraryInfoKind : std::uint32_t {
    kVersion = 1,
    kCudartVersion,
    kMaxDeviceVersion,
    kProperty
};
struct CudaDnnTensor {
    // Untrusted integer fields are range checked before forming SDK enums.
    std::uint32_t type{}, format{};
    int dimensions{};
    std::array<int, CUDNN_DIM_MAX> sizes{}, strides{};
};
struct CudaDnnActivationState {
    std::uint32_t mode{}, nan{};
    double coefficient{};
};
struct CudaDnnFilter {
    std::uint32_t type{}, format{};
    int dimensions{};
    std::array<int, CUDNN_DIM_MAX> sizes{};
};
struct CudaDnnConvolutionState {
    std::uint32_t mode{}, type{};
    int dimensions{};
    std::array<int, CUDNN_DIM_MAX> padding{}, strides{}, dilation{};
};
struct CudaDnnPoolingState {
    std::uint32_t mode{}, nan{};
    int dimensions{};
    std::array<int, CUDNN_DIM_MAX> window{}, padding{}, strides{};
};
struct CudaDnnLrnState {
    std::uint32_t n{};
    double alpha{}, beta{}, k{};
};
struct CudaDnnReductionState {
    std::uint32_t operation{}, type{}, nan{}, indices{}, indices_type{};
};
struct CudaDnnOpTensorState {
    std::uint32_t operation{}, type{}, nan{};
};
struct CudaDnnTensorTransformState {
    std::uint32_t dimensions{}, format{}, direction{}, presence{};
    // Presence bits 1/2/4 preserve independently nullable native arrays.
    // Padding arrays use dimensions entries; fold uses max(dimensions-2, 0).
    std::array<std::int32_t, CUDNN_DIM_MAX> before{}, after{};
    std::array<std::uint32_t, CUDNN_DIM_MAX> fold{};
};
struct CudaDnnFoldedDescriptors {
    // Same-registry identities in the original ABI order. Only outputs[7]
    // may be absent; preserve positional aliases across both arrays.
    std::array<std::uint64_t, 4> inputs{};
    std::array<std::uint64_t, 8> outputs{};
    std::uint32_t format{};
};
struct CudaDnnRnnDataState {
    // Presence bits: lengths=1, padding=2. Early native rejection paths need
    // no snapshots, but retain independently nonnull pointer presence.
    std::uint32_t type{}, layout{}, presence{}, padding_bytes{};
    int max_sequence_length{}, batch_size{}, vector_size{};
    std::vector<int> lengths;
    alignas(double) std::array<std::byte, 8> padding{};
};
struct CudaDnnSeqDataState {
    // Presence bits: sizes=1, axes=2, lengths=4, padding=8.
    std::uint32_t type{}, presence{}, padding_bytes{};
    int dimensions{};
    std::array<int, 4> sizes{};
    std::array<std::uint32_t, 4> axes{};
    std::uint64_t sequence_count{};
    std::vector<int> lengths;
    alignas(double) std::array<std::byte, 8> padding{};
};
struct CudaDnnSeqDataQuery {
    int dimensions_requested{};
    std::uint64_t lengths_requested{};
    // Bits: type=1, rank=2, sizes=4, axes=8, count=16, lengths=32, padding=64.
    std::uint32_t outputs{};
};
struct CudaDnnRnnState {
    std::uint32_t algorithm{}, cell_mode{}, bias_mode{}, direction{}, input_mode{}, type{},
        math_precision{}, math_type{};
    std::int32_t input_size{}, hidden_size{}, projection_size{}, layers{};
    std::uint64_t dropout{};
    std::uint32_t flags{};
};
struct CudaDnnAttnState {
    std::uint32_t mode{};
    std::int32_t heads{};
    double scale{};
    std::uint32_t type{}, precision{}, math_type{};
    // Same-registry dropout identities; zero remains native null, and aliases
    // preserve both original positions without duplicating the liveness edge.
    std::array<std::uint64_t, 2> dropouts{};
    std::int32_t query_size{}, key_size{}, value_size{}, query_projection{}, key_projection{},
        value_projection{}, output_projection{}, max_query_sequence{}, max_key_sequence{},
        max_batch{}, max_beam{};
};
enum class CudaDnnRnnQueryKind : std::uint32_t { kWeightSpace = 1, kTempSpace, kAttentionBuffers };
struct CudaDnnRnnQuery {
    CudaDnnRnnQueryKind kind{};
    std::uint64_t descriptor{}, input{};
    // Global output bits: weight=1, workspace=2, reserve=4.
    std::uint32_t mode{}, outputs{};
};
struct CudaDnnRnnWeightParams {
    std::uint64_t descriptor{};
    std::int32_t pseudo_layer{}, linear_layer{};
    // Matrix/bias destinations and independently present address outputs.
    std::array<std::uint64_t, 2> destinations{};
    std::uint32_t outputs{};
};
struct CudaDnnAttnWeights {
    std::uint64_t descriptor{};
    std::uint32_t kind{}, has_base{};
    // Address-only layout capacity; no caller pointer or allocation authority.
    std::size_t weight_bytes{};
    std::uint64_t destination{};
    std::uint32_t has_output{};
};
struct CudaDnnRnnClipState {
    std::uint32_t mode{}, nan{};
    double lower{}, upper{};
};
enum class CudaDnnConvolutionAttribute : std::uint32_t { kGroup = 1, kMath, kReorder };
struct CudaDnnResult {
    CudaDnnDisposition disposition{CudaDnnDisposition::kRejected};
    CUresult status{CUDA_ERROR_UNKNOWN};
    std::optional<cudnnStatus_t> original_status{};
    std::uint64_t id{}, value{};
    CudaDnnTensor tensor{};
    CudaDnnActivationState activation{};
    CudaDnnFilter filter{};
    CudaDnnConvolutionState convolution{};
    CudaDnnPoolingState pooling{};
    CudaDnnOpTensorState op_tensor{};
    CudaDnnSeqDataState seq_data{};
};
struct CudaDnnRnnQueryResult {
    CudaDnnResult result;
    // Full-width native sizes in weight/workspace/reserve order.
    std::array<std::uint64_t, 3> sizes{};
};
struct CudaDnnRnnWeightParamsResult {
    CudaDnnResult result;
    std::uint32_t non_null{};
    // Relative to the passed owned weight pointer, or numerical offsets for
    // native null-base queries. Neither interpretation creates an allocation.
    std::array<std::uint64_t, 2> offsets{};
    bool has_context_loss{};
};
struct CudaDnnAttnWeightsResult {
    CudaDnnResult result;
    std::uint32_t non_null{};
    std::uint64_t offset{};
    bool has_context_loss{};
};
// Native pointers and SDK code are backend-only. A view is a trusted owner
// product, never a serialized pointer or application-supplied callback.
struct CudaDnnObjectView {
    CudaDnnObjectKind kind{};
    void* native{};
    std::shared_ptr<const void> lifetime;
};
enum class CudaDnnBackendAttributeOperation : std::uint32_t { kSet = 1, kGet };
enum class CudaDnnBackendAttributeDirection : std::uint32_t {
    kNone = 0,
    kInput,
    kScalarOutput,
    kDescriptorOutput,
    kHandleOutput,
    kPointerOutput
};
struct CudaDnnBackendAttributeHeader {
    std::uint64_t descriptor{};
    std::uint32_t name{}, type{};
    std::int64_t count{};
    // Array=1, count output=2. Set never has a count output.
    std::uint32_t presence{};
};
struct CudaDnnBackendHandleReference {
    std::uint64_t owner{}, id{};
};
struct CudaDnnBackendContextProvenance {
    std::uint64_t owner{};
    std::weak_ptr<const void> lifetime;
    int device_ordinal{-1};
};
// Trusted, synchronous session authority. Native pointers/callbacks are never
// retained in semantic snapshots or accepted from the protocol.
struct CudaDnnBackendContext {
    cudnnHandle_t native{};
    std::uint64_t owner{}, handle{};
    std::weak_ptr<const void> lifetime;
    CUresult (*check_context)(void*) noexcept {};
    void* opaque{};
    int device_ordinal{-1};
    // Synchronous CPU lookup under session/registry exclusion. A missing
    // callback permits only the single native/handle pair above.
    cudnnHandle_t (*resolve_handle)(std::uint64_t id, void* opaque) noexcept {};
};
struct CudaDnnBackendAttributeValues {
    // Native-host scalar representation; protocol conversion is a service concern.
    std::vector<std::byte> scalars{};
    std::vector<std::uint64_t> descriptors{};
    std::vector<CudaDnnBackendHandleReference> handles{};
    std::vector<std::uint64_t> pointer_slots{};
};
struct CudaDnnBackendAttributePreparation {
    CudaDnnResult result;
    std::uint64_t generation{};
    CudaDnnBackendAttributeDirection direction{};
    std::uint32_t input_elements{}, output_elements{}, count_bytes{};
    // Exact caller write span; typed storage may have a zero-filled suffix.
    std::uint32_t output_bytes{};
};
struct CudaDnnBackendAttributeResult {
    CudaDnnResult result;
    std::uint64_t generation{};
    bool is_configuration_valid{};
    std::int64_t count{};
    std::uint32_t count_bytes{};
    CudaDnnBackendAttributeValues values{};
    bool has_context_loss{};
};
struct CudaDnnBackendSnapshot;
struct CudaDnnBackendSnapshotAttribute {
    std::uint32_t name{}, type{};
    std::int64_t count{};
    std::vector<std::byte> scalars{};
    std::vector<std::shared_ptr<const CudaDnnBackendSnapshot>> descriptors{};
    std::vector<CudaDnnBackendHandleReference> handles{};
    std::vector<std::uint64_t> pointer_slots{};
};
enum class CudaDnnBackendTensorAlignmentSource { kNativeDefault = 1, kNativeSet };
struct CudaDnnBackendTensorAlignment {
    std::int64_t bytes{};
    CudaDnnBackendTensorAlignmentSource source{};
};
// Fixed native observations are not allocation grants. The configured SDK has
// no alignment getter; its separately audited constructor/setter provenance is
// explicit rather than masquerading as a successful private query.
struct CudaDnnBackendTensorObservation {
    std::uint32_t type{}, reordering{};
    std::int64_t uid{}, vector_count{}, vectorized_dimension{};
    CudaDnnBackendTensorAlignment alignment;
    std::uint32_t dimension_count{}, stride_count{};
    std::array<std::int64_t, 9> dimensions{}, strides{};
    bool is_virtual{}, is_by_value{};
};
// Device descriptions never confer execution or device-allocation authority.
enum class CudaDnnBackendPropertyMode { kUnknown, kDiscovery, kHandle, kJson, kPopulated };
struct CudaDnnBackendSnapshot {
    std::uint32_t kind{};
    std::uint64_t generation{};
    bool is_valid{}, is_finalized{};
    std::vector<CudaDnnBackendSnapshotAttribute> attributes;
    std::optional<CudaDnnBackendTensorAlignment> tensor_alignment;
    std::optional<CudaDnnBackendTensorObservation> tensor;
    std::optional<CudaDnnBackendContextProvenance> context;
    // Native-generated Config/Engine copies carry graph semantics without
    // inventing a caller Set attribute or retaining its original native owner.
    std::shared_ptr<const CudaDnnBackendSnapshot> generated_graph;
    // Copied device description lineage is independent of tensor execution
    // requirements and never retains the originating descriptor/handle owner.
    std::shared_ptr<const CudaDnnBackendSnapshot> generated_property;
    // Private native observation: heuristic raw capacity may exceed the
    // actual emitted prefix. This is a transport bound, not execution proof.
    std::optional<std::uint32_t> generated_array_capacity;
    // Independent native observations; Engine's knob count is not a bound for
    // its notes or generated layouts. JSON counts exclude a terminating NUL.
    std::optional<std::uint32_t> numerical_note_count, behavior_note_count, layout_type_count,
        json_bytes;
    // Generated kind-9/38 copies preserve native header state. An unavailable
    // private observation is neither a finalized default nor an ownership fault.
    std::optional<bool> metadata_finalized;
    CudaDnnBackendPropertyMode property_mode{CudaDnnBackendPropertyMode::kUnknown};
    std::optional<std::int32_t> property_device_id;
    // VariantPack order is semantic: duplicate UIDs must not be deduplicated.
    std::vector<std::int64_t> variant_uids;
    std::vector<std::uint64_t> pointer_slots;
    std::uint64_t workspace_slot{};
};
struct CudaDnnBackendView {
    CudaDnnObjectView object;
    std::shared_ptr<const CudaDnnBackendSnapshot> snapshot;
};
class CudaDnnRnnLease;
struct CudaDnnBackendCallView {
    CudaDnnBackendView descriptor;
    std::shared_ptr<CudaDnnRnnLease> call;
};
// Provenance does not own application allocations. Every consumer must resolve
// the exact live session incarnation and range before acquiring command leases.
struct CudaDnnDropoutBinding {
    std::weak_ptr<const void> context;
    std::uint64_t allocation{};
    std::size_t offset{}, bytes{};
};
struct CudaDnnDropoutView {
    CudaDnnObjectView object;
    CudaDnnDropoutBinding binding;
};
// Scoped native-call exclusion is separate from the physical immutable lease.
// The shared control survives the registry facade, without native cleanup on
// the thread releasing the final command owner.
class CudaDnnRnnLease final {
   public:
    ~CudaDnnRnnLease();
    CudaDnnRnnLease(const CudaDnnRnnLease&) = delete;
    CudaDnnRnnLease& operator=(const CudaDnnRnnLease&) = delete;
    [[nodiscard]] bool begin_call() noexcept;
    void end_call() noexcept;
    void quarantine() noexcept;
    // Trusted native-Create collision check, only while this lease owns call
    // exclusion. An unauthorized probe returns true conservatively. The weak
    // index retains neither native objects nor a registry facade pointer.
    [[nodiscard]] bool owns_backend_pointer(void* native) const noexcept;

   private:
    friend class CudaDnnRegistry;
    struct State;
    struct Implementation;
    CudaDnnRnnLease(std::shared_ptr<State> state, std::shared_ptr<const void> owner);
    std::unique_ptr<Implementation> implementation_;
};
struct CudaDnnRnnView {
    CudaDnnObjectView object;
    std::array<CudaDnnDropoutView, 4> dropouts;
    std::shared_ptr<CudaDnnRnnLease> lease;
    std::array<CudaDnnObjectView, 2> algorithms{};
};
struct CudaDnnRnnBorrowResult {
    CudaDnnResult result;
    CudaDnnRnnView view;
};
struct CudaDnnRnnBuildResult {
    CudaDnnResult result;
    bool has_context_loss{};
};
class CudaDnnDropoutMutation final {
   public:
    ~CudaDnnDropoutMutation();
    CudaDnnDropoutMutation(const CudaDnnDropoutMutation&) = delete;
    CudaDnnDropoutMutation& operator=(const CudaDnnDropoutMutation&) = delete;
    [[nodiscard]] CudaDnnDropout native() const noexcept;
    [[nodiscard]] const CudaDnnDropoutBinding& old_binding() const noexcept;
    // Trusted queue admission and issue are one-shot. Cancellation before issue
    // leaves the old binding unchanged; final lease destruction unlocks it.
    [[nodiscard]] bool claim() noexcept;
    [[nodiscard]] bool mark_issued() noexcept;
    void finish(cudnnStatus_t status, const CudaDnnDropoutBinding& binding,
                CUresult proof) noexcept;
    [[nodiscard]] std::optional<CudaDnnResult> completion_result() const noexcept;

   private:
    friend class CudaDnnRegistry;
    struct State;
    struct Implementation;
    explicit CudaDnnDropoutMutation(std::shared_ptr<State> state,
                                    std::shared_ptr<const void> lifetime, CudaDnnDropout native);
    std::unique_ptr<Implementation> implementation_;
};
struct CudaDnnDropoutMutationResult {
    CudaDnnResult result;
    std::shared_ptr<CudaDnnDropoutMutation> mutation;
};
enum class CudaDnnTensorSetter { kFour, kFourStrided, kNd, kNdPacked };
struct CudaDnnRegistrySnapshot {
    std::size_t objects{}, borrowed{};
    bool is_closing{}, is_closed{}, is_quarantined{};
};

// One process/nonrecycled thread owns every SDK call. Public metadata operations
// require no context or GPU work; the private Folded operation is session-owned.
// Immutable leases defer native mutation/destruction until physical consumers
// release them; a lease's destructor itself performs no SDK calls.
class CudaDnnRegistry final {
   public:
    explicit CudaDnnRegistry(CudaDnnDispatch dispatch, std::size_t max_objects = 256);
    ~CudaDnnRegistry();
    CudaDnnRegistry(const CudaDnnRegistry&) = delete;
    CudaDnnRegistry& operator=(const CudaDnnRegistry&) = delete;
    // Context-independent queries still require the live registry owner.
    // Scalar success has no original_status; value retains the entire native
    // size_t, including zero. Property publishes requested successful int bits.
    [[nodiscard]] CudaDnnResult library_info(CudaDnnLibraryInfoKind kind,
                                             std::uint32_t property = 0,
                                             bool has_output = false) noexcept;
    [[nodiscard]] CudaDnnResult create(CudaDnnObjectKind kind) noexcept;
    [[nodiscard]] CudaDnnLegacyResult legacy_metadata(const CudaDnnLegacyMetadata&) noexcept;
    [[nodiscard]] CudaDnnLegacyResult legacy_control(cudnnHandle_t, const CudaDnnLegacyControl&,
                                                     const CudaDnnLegacyContext&) noexcept;
    [[nodiscard]] CudaDnnLegacyBorrowResult borrow_legacy_rnn(
        const CudaSessionDnnLegacyRnn&, std::shared_ptr<ExecutionHostStorage>) noexcept;
    [[nodiscard]] CudaDnnResult destroy(std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnResult create_backend(std::uint32_t kind) noexcept;
    [[nodiscard]] CudaDnnResult initialize_backend(std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnResult finalize_backend(
        std::uint64_t id, const CudaDnnBackendContext* context = nullptr) noexcept;
    [[nodiscard]] CudaDnnBackendAttributePreparation prepare_backend_attribute(
        CudaDnnBackendAttributeOperation operation, const CudaDnnBackendAttributeHeader& header,
        const CudaDnnBackendContext* context = nullptr) noexcept;
    [[nodiscard]] CudaDnnBackendAttributeResult set_backend_attribute(
        const CudaDnnBackendAttributeHeader& header, std::uint64_t generation,
        const CudaDnnBackendAttributeValues& values,
        const CudaDnnBackendContext* context = nullptr) noexcept;
    [[nodiscard]] CudaDnnBackendAttributeResult get_backend_attribute(
        const CudaDnnBackendAttributeHeader& header, std::uint64_t generation,
        const CudaDnnBackendAttributeValues& values,
        const CudaDnnBackendContext* context = nullptr) noexcept;
    [[nodiscard]] std::optional<CudaDnnBackendView> borrow_backend(std::uint64_t id) noexcept;
    // Native execution/proof/cleanup use the same registry-wide exclusion as
    // queued RNN calls; the immutable object lease survives physical retirement.
    [[nodiscard]] std::optional<CudaDnnBackendCallView> borrow_backend_call(
        std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnResult set_tensor(std::uint64_t id, CudaDnnTensorSetter setter,
                                           const CudaDnnTensor& tensor) noexcept;
    [[nodiscard]] CudaDnnResult get_tensor(std::uint64_t id, int dimensions,
                                           bool is_four = false) noexcept;
    [[nodiscard]] CudaDnnResult tensor_size(std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnResult set_activation(std::uint64_t id,
                                               CudaDnnActivationState activation) noexcept;
    [[nodiscard]] CudaDnnResult get_activation(std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnResult set_filter(std::uint64_t id, const CudaDnnFilter& filter,
                                           bool is_four = false) noexcept;
    [[nodiscard]] CudaDnnResult get_filter(std::uint64_t id, int requested,
                                           bool is_four = false) noexcept;
    [[nodiscard]] CudaDnnResult filter_size(std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnResult set_convolution(std::uint64_t id,
                                                const CudaDnnConvolutionState& convolution,
                                                bool is_two = false) noexcept;
    [[nodiscard]] CudaDnnResult get_convolution(std::uint64_t id, int requested,
                                                bool is_two = false) noexcept;
    [[nodiscard]] CudaDnnResult set_convolution_attribute(std::uint64_t id,
                                                          CudaDnnConvolutionAttribute attribute,
                                                          std::uint32_t value) noexcept;
    [[nodiscard]] CudaDnnResult get_convolution_attribute(
        std::uint64_t id, CudaDnnConvolutionAttribute attribute) noexcept;
    [[nodiscard]] CudaDnnResult convolution_output(std::uint64_t convolution, std::uint64_t tensor,
                                                   std::uint64_t filter, int dimensions,
                                                   bool is_two = false) noexcept;
    [[nodiscard]] CudaDnnResult set_pooling(std::uint64_t id,
                                            const CudaDnnPoolingState& pooling) noexcept;
    [[nodiscard]] CudaDnnResult get_pooling(std::uint64_t id, int requested) noexcept;
    [[nodiscard]] CudaDnnResult pooling_output(std::uint64_t pooling, std::uint64_t tensor,
                                               int dimensions) noexcept;
    [[nodiscard]] CudaDnnResult set_lrn(std::uint64_t id, CudaDnnLrnState lrn) noexcept;
    [[nodiscard]] CudaDnnResult set_reduction(std::uint64_t id,
                                              const CudaDnnReductionState& reduction) noexcept;
    [[nodiscard]] CudaDnnResult set_op_tensor(std::uint64_t id,
                                              CudaDnnOpTensorState operation) noexcept;
    // Bits 1/2/4 request operation/type/NaN outputs; zero forwards three nulls.
    [[nodiscard]] CudaDnnResult get_op_tensor(std::uint64_t id,
                                              std::uint32_t output_mask = 7) noexcept;
    [[nodiscard]] CudaDnnResult set_tensor_transform(
        std::uint64_t id, const CudaDnnTensorTransformState& transform) noexcept;
    [[nodiscard]] CudaDnnResult init_transform_dest(std::uint64_t transform, std::uint64_t source,
                                                    std::uint64_t destination,
                                                    bool has_size = true) noexcept;
    [[nodiscard]] CudaDnnResult set_rnn_data(std::uint64_t id,
                                             const CudaDnnRnnDataState& data) noexcept;
    [[nodiscard]] CudaDnnResult set_seq_data(std::uint64_t id,
                                             const CudaDnnSeqDataState& data) noexcept;
    [[nodiscard]] CudaDnnResult get_seq_data(std::uint64_t id, CudaDnnSeqDataQuery query) noexcept;
    // Durable dropout dependencies protect descriptor liveness, not immutable
    // configuration or RNG storage. Future compute must acquire fresh leases.
    [[nodiscard]] CudaDnnResult set_rnn(std::uint64_t id, const CudaDnnRnnState& rnn) noexcept;
    [[nodiscard]] CudaDnnResult set_attn(std::uint64_t id, const CudaDnnAttnState& attn) noexcept;
    [[nodiscard]] CudaDnnResult set_rnn_clip(std::uint64_t id, CudaDnnRnnClipState clip) noexcept;
    [[nodiscard]] CudaDnnDropoutMutationResult begin_dropout_mutation(std::uint64_t id) noexcept;
    [[nodiscard]] std::optional<CudaDnnDropoutView> borrow_dropout(std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnRnnBorrowResult borrow_rnn(std::uint64_t id) noexcept;
    [[nodiscard]] CudaDnnRnnBorrowResult borrow_attn(std::uint64_t id) noexcept;
    [[nodiscard]] std::optional<CudaDnnObjectView> borrow(std::uint64_t id,
                                                          CudaDnnObjectKind kind) noexcept;
    void close() noexcept;
    [[nodiscard]] CudaDnnResult shutdown() noexcept;
    [[nodiscard]] CudaDnnRegistrySnapshot snapshot() const noexcept;

   private:
    friend class CudaExecutionSession;
    [[nodiscard]] CudaDnnRnnBorrowResult borrow_composite(std::uint64_t id,
                                                          CudaDnnObjectKind kind) noexcept;
    struct FoldedResult {
        CudaDnnResult result;
        bool has_context_loss{};
        // This helper's status is diagnostic, never original Folded provenance.
        std::optional<cudnnStatus_t> preflight_status{};
    };
    [[nodiscard]] FoldedResult folded_backward_data(cudnnHandle_t handle,
                                                    const CudaDnnFoldedDescriptors& descriptors,
                                                    CUresult (*check_context)(const void*) noexcept,
                                                    const void* session_state) noexcept;
    struct RnnQueryResult {
        CudaDnnRnnQueryResult result;
        bool has_context_loss{};
        // Diagnostic only: never original status of the requested size query.
        std::optional<cudnnStatus_t> preflight_status{};
    };
    [[nodiscard]] RnnQueryResult query_rnn_sizes(cudnnHandle_t handle, const CudaDnnRnnQuery& query,
                                                 CUresult (*check_context)(const void*) noexcept,
                                                 const void* session_state) noexcept;
    [[nodiscard]] CudaDnnRnnBuildResult build_rnn_dynamic(
        cudnnHandle_t handle, std::uint64_t id, int mini_batch,
        CUresult (*check_context)(const void*) noexcept, const void* session_state) noexcept;
    [[nodiscard]] CudaDnnRnnWeightParamsResult rnn_weight_params(
        cudnnHandle_t handle, const CudaDnnRnnWeightParams& parameters, CUdeviceptr weights,
        std::size_t weight_bytes, CUresult (*check_context)(const void*) noexcept,
        const void* session_state) noexcept;
    [[nodiscard]] CudaDnnAttnWeightsResult attn_weights(
        cudnnHandle_t handle, const CudaDnnAttnWeights& parameters,
        CUresult (*check_context)(const void*) noexcept, const void* session_state) noexcept;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};
}  // namespace glimmer::backend::cuda
