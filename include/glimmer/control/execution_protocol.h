#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

// Protocol enums intentionally retain their declared fixed-width wire ABI.
// NOLINTBEGIN(performance-enum-size)
namespace glimmer::control {

inline constexpr std::size_t kExecutionHeaderBytes = 48;
inline constexpr std::size_t kExecutionMaxPayloadBytes = std::size_t{16} * 1024 * 1024;
inline constexpr std::size_t kExecutionSourceContinuationBytes = 64;
inline constexpr std::uint32_t kExecutionSourceContinuationMagic = 0x314e4353;  // SCN1
inline constexpr std::uint32_t kExecutionSourceContinuationCleanup = 1;
// Optional, exact four-byte tail on an ordinary launch. Empty tails retain the
// actual SDK host-result contract; source continuations cannot request admission.
inline constexpr std::uint32_t kExecutionAsyncLaunchMagic = 0x31595341;  // ASY1
// An eight-byte Hello carries this role followed by context flags. Legacy
// application Hello remains empty/flags-only or uses CTX1 below; host owners
// cannot run kernels.
inline constexpr std::uint32_t kExecutionHostOwnerRole = 0x31545348;  // HST1
// Private, never application-visible backing. Kernel argument admission must
// not retain these ranges: only exact staged transfer commands can use them.
inline constexpr std::uint32_t kExecutionStagingOwnerRole = 0x31475453;  // STG1
// Private unregistered shared backing for exact unified-address SDK copies.
// Never public pinned memory, a device alias or a kernel-argument capability.
inline constexpr std::uint32_t kExecutionPageableOwnerRole = 0x31474150;  // PAG1
// CPU-only library metadata. This role cannot initialize a CUDA context, own
// device allocations, mint priority, or issue any application GPU command.
inline constexpr std::uint32_t kExecutionLtMetadataRole = 0x314d544c;   // LTM1
inline constexpr std::uint32_t kExecutionDnnMetadataRole = 0x314d4e44;  // DNM1
// Address-only metadata carrier; no CUDA context or application GPU commands.
inline constexpr std::uint32_t kExecutionPointerMetadataRole = 0x31444d50;  // PMD1
// Context-free process/device allocation metadata carrier. Allocations issue
// through execution Sessions; this carrier cannot submit application GPU work.
inline constexpr std::uint32_t kExecutionProcessMemoryRole = 0x314d4d50;  // PMM1
// Process-owned native library calls. Unlike metadata roles, this carrier can
// retain group/async/device execution ownership and requires scheduler grants.
inline constexpr std::uint32_t kExecutionNcclRole = 0x314c434e;  // NCL1
// Cold context descriptor only. A client cannot select a shared execution
// cohort or acquire scheduling authority by declaring a purpose.
inline constexpr std::uint32_t kExecutionContextRole = 0x31585443;  // CTX1
inline constexpr std::size_t kExecutionContextBootstrapBytes = 24;
enum class ExecutionContextPurpose : std::uint32_t {
    kLegacyIsolated = 0,
    kExplicitUser = 1,
    kProcessPrimary = 2,
    kRuntimePrimary = 3,
};
struct ExecutionContextBootstrap {
    ExecutionContextPurpose purpose{ExecutionContextPurpose::kLegacyIsolated};
    std::uint32_t flags{};
    std::uint64_t source_generation{};
    bool operator==(const ExecutionContextBootstrap&) const = default;
};
// Pure, allocation-free codecs. Decode also accepts legacy empty/flags-only
// ordinary context requests; helper roles are deliberately not ordinary contexts.
[[nodiscard]] std::optional<std::array<std::byte, kExecutionContextBootstrapBytes>>
encode_execution_context_bootstrap(ExecutionContextBootstrap descriptor) noexcept;
[[nodiscard]] std::optional<ExecutionContextBootstrap> decode_execution_context_bootstrap(
    std::span<const std::byte> bytes) noexcept;
enum class ExecutionMessageKind : std::uint16_t { kRequest = 1, kReply = 2 };
// Wire identities are stable: append new values or introduce a new protocol version.
enum class ExecutionOpcode : std::uint32_t {
    kHello = 1,
    kLoadPtx = 2,
    kGetFunction = 3,
    kUnloadModule = 4,
    kAllocate = 5,
    kRelease = 6,
    kUpload = 7,
    kDownload = 8,
    kCreateStream = 9,
    kDestroyStream = 10,
    kLaunch = 11,
    kDrain = 12,
    kSnapshot = 13,
    kClose = 14,
    kCaptureFrontier = 15,
    kQueryFrontier = 16,
    kReleaseFrontier = 17,
    kLoadBinary = 18,
    kGetContextFlags = 19,
    kSetContextFlags = 20,
    kMemory = 21,
    kImportHostMemory = 22,
    kHostTransfer = 23,
    kReleaseHostMemory = 24,
    kStagedHostTransfer = 25,
    kCreateBlas = 26,
    kDestroyBlas = 27,
    kSetBlasAttribute = 28,
    kGetBlasAttribute = 29,
    kSgemm = 30,
    kGemmEx = 31,
    kStridedGemmEx = 32,
    kCreateLtMetadata = 33,
    kDestroyLtMetadata = 34,
    kSetLtAttribute = 35,
    kGetLtAttribute = 36,
    kLtVersion = 37,
    kCreateLt = 38,
    kDestroyLt = 39,
    kLtHeuristic = 40,
    kLtAlgorithmCheck = 41,
    kLtMatmul = 42,
    kRuntimeMetadata = 43,
    kCreatePriorityStream = 44,
    kGetGlobal = 45,
    kLoadModuleFile = 46,
    kCreateEvent = 47,
    kRecordEvent = 48,
    kQueryEvent = 49,
    kSynchronizeEvent = 50,
    kDestroyEvent = 51,
    kStreamWaitEvent = 52,
    kEventElapsedTime = 53,
    kWaitForeignEvent = 54,
    kSetBlasWorkspace = 55,
    kStridedSgemm = 56,
    kCreateRand = 57,
    kSetRandSeed = 58,
    kDestroyRand = 59,
    kDriverContextControl = 60,
    kRuntimeContextControl = 61,
    kCreateDnnMetadata = 62,
    kDestroyDnnMetadata = 63,
    kSetDnnTensor = 64,
    kGetDnnTensor = 65,
    kDnnTensorSize = 66,
    kSetDnnActivation = 67,
    kGetDnnActivation = 68,
    kCreateDnn = 69,
    kDestroyDnn = 70,
    kSetDnnStream = 71,
    kGetDnnStream = 72,
    kDnnCompute = 73,
    kSetDnnFilter = 74,
    kGetDnnFilter = 75,
    kDnnFilterSize = 76,
    kSetDnnConvolution = 77,
    kGetDnnConvolution = 78,
    kSetDnnConvolutionAttribute = 79,
    kGetDnnConvolutionAttribute = 80,
    kDnnConvolutionOutput = 81,
    kDnnConvolutionQuery = 82,
    kDnnConvolutionForward = 83,
    kDnnConvolutionFind = 84,
    kSetDnnPooling = 85,
    kGetDnnPooling = 86,
    kDnnPoolingOutput = 87,
    kDnnTensorForward = 88,
    kSetDnnLrn = 89,
    kSetDnnReduction = 90,
    kDnnReductionQuery = 91,
    kDnnReduction = 92,
    kDnnBatchNormQuery = 93,
    kDnnBatchNorm = 94,
    kDnnDropoutStatesSize = 95,
    kDnnDropout = 96,
    kSetDnnOpTensor = 97,
    kGetDnnOpTensor = 98,
    kSetDnnTensorTransform = 99,
    kDnnInitTransformDest = 100,
    kDnnReorderFilterAndBias = 101,
    kDnnLibraryInfo = 102,
    kDnnBackwardConvolutionQuery = 103,
    kDnnFoldedConvBackwardDataDescriptors = 104,
    kSetDnnRnnData = 105,
    kSetDnnSeqData = 106,
    kGetDnnSeqData = 107,
    kSetDnnRnn = 108,
    kSetDnnAttn = 109,
    kDnnRnnQuery = 110,
    kSetDnnRnnClip = 111,
    kDnnRnnCompute = 112,
    kDnnBuildRnnDynamic = 113,
    kDnnRnnWeightParams = 114,
    kDnnAttnWeights = 115,
    kPrepareDnnAttn = 116,
    kDnnAttnCompute = 117,
    kDnnBackendCreate = 118,
    kDnnBackendDestroy = 119,
    kDnnBackendInitialize = 120,
    kDnnBackendFinalize = 121,
    kDnnBackendPrepare = 122,
    kDnnBackendSet = 123,
    kDnnBackendGet = 124,
    kDnnBackendContext = 125,
    kPrepareDnnBackendExecute = 126,
    kDnnBackendExecute = 127,
    kFunctionControl = 128,
    kGeam = 129,
    kPointerMetadata = 130,
    kNcclCall = 131,
    kSourceContinuation = 132,
    kQueryStagingIdle = 133,
    // CPU-only source-stream boundaries. No application function/user pointer
    // crosses the wire; successful completion is not a GPU-stop receipt.
    kEnqueueHostFunction = 134,
    kQueryHostFunction = 135,
    kFinishHostFunction = 136,
    // Ordinary ASY1 observation metadata only; never source continuations.
    kQueryAsyncFrontiers = 137,
    kReleaseAsyncFrontiers = 138,
    kSgemv = 139,
    kSgemmEx = 140,
    kUnifiedCopy = 141,
    kLoadModuleJit = 142,
    // CPU algorithm capability metadata, never a GPU command.
    kLtAlgorithmCapability = 143,
    // Process-owned allocation/pool SDK calls; never admission-only success.
    kProcessMemory = 144,
    // Native graph construction is not execution admission. Replay has its own
    // ordinary queue command and physical tail; neither accepts a source scope.
    kGraphControl = 145,
    kGraphLaunch = 146,
    // CPU-only immutable SDK-profile binding; no native status or resource.
    kBindDnnProfile = 147,
    kDnnLegacyMetadata = 148,
    kDnnLegacyControl = 149,
    kDnnLegacyCompute = 150,
};
inline constexpr std::size_t kExecutionMaxAsyncFrontiers = 256;
struct ExecutionAsyncFrontierKey {
    std::uint64_t id{}, sequence{};
    bool operator==(const ExecutionAsyncFrontierKey&) const = default;
};
struct ExecutionAsyncFrontierObservation {
    ExecutionAsyncFrontierKey key;
    std::int32_t status{};
    bool is_retired{};
    bool operator==(const ExecutionAsyncFrontierObservation&) const = default;
};
// Versioned, bounded, CUDA-free codecs. Identity authentication and exact reply
// correspondence belong to the service/frontend; duplicates and malformed
// status/retirement combinations are rejected here without partial output.
[[nodiscard]] std::optional<std::vector<std::byte>> encode_execution_async_frontier_batch(
    std::span<const ExecutionAsyncFrontierKey> keys) noexcept;
[[nodiscard]] std::optional<std::vector<ExecutionAsyncFrontierKey>>
decode_execution_async_frontier_batch(std::span<const std::byte> bytes) noexcept;
[[nodiscard]] std::optional<std::vector<std::byte>> encode_execution_async_frontier_observations(
    std::span<const ExecutionAsyncFrontierObservation> observations) noexcept;
[[nodiscard]] std::optional<std::vector<ExecutionAsyncFrontierObservation>>
decode_execution_async_frontier_observations(std::span<const std::byte> bytes) noexcept;
// Source provenance is authenticated by the service, never by this wire codec.
// Cleanup has its own per-parent sequence and cannot borrow the live caller's.
struct ExecutionSourceContinuation {
    std::uint32_t flags{};
    std::array<std::uint64_t, 2> caller{};
    std::uint64_t group{}, ordinary_sequence{}, origin_source{}, cleanup_parent{};
};
struct ExecutionRequestView {
    ExecutionOpcode operation{ExecutionOpcode::kHello};
    std::span<const std::byte> payload;
    std::optional<ExecutionSourceContinuation> continuation;
};
// A view borrows the unchanged request body. The outer header remains the reply
// correlation identity; only this effective operation controls request semantics.
[[nodiscard]] std::optional<ExecutionRequestView> normalize_execution_request(
    ExecutionOpcode outer, std::span<const std::byte> payload,
    std::size_t ordinary_limit = kExecutionMaxPayloadBytes) noexcept;
[[nodiscard]] std::optional<std::vector<std::byte>> encode_execution_source_continuation(
    const ExecutionSourceContinuation& continuation, ExecutionOpcode inner,
    std::span<const std::byte> payload,
    std::size_t ordinary_limit = kExecutionMaxPayloadBytes) noexcept;
enum class ExecutionFunctionControl : std::uint32_t {
    kGetAttribute = 1,
    kSetAttribute = 2,
    kSetCache = 3,
    kGetAttributes = 4,
};
enum class ExecutionContextControl : std::uint32_t {
    kGetLimit = 1,
    kSetLimit = 2,
    kGetCache = 3,
    kSetCache = 4,
    kGetSharedMemory = 5,
    kSetSharedMemory = 6,
    kResetPersistingCache = 7,
    kGetApiVersion = 8,
    kGetAffinity = 9,
    kGetPriorityRange = 10,
};
[[nodiscard]] constexpr bool execution_context_control_is_query(
    ExecutionContextControl operation) noexcept {
    return operation == ExecutionContextControl::kGetLimit ||
           operation == ExecutionContextControl::kGetCache ||
           operation == ExecutionContextControl::kGetSharedMemory ||
           operation == ExecutionContextControl::kGetApiVersion ||
           operation == ExecutionContextControl::kGetAffinity ||
           operation == ExecutionContextControl::kGetPriorityRange;
}
enum class ExecutionModuleImageKind : std::uint32_t { kBinary = 1, kPtx = 2 };
// Read-only DNN descriptor observations. These may invoke the vendor SDK and
// retain active-call and descriptor pins. Service metadata is broader: it also
// includes setters and other queries with different mutation effects.
[[nodiscard]] constexpr bool execution_is_dnn_descriptor_observation(
    ExecutionOpcode operation) noexcept {
    switch (operation) {
        case ExecutionOpcode::kGetDnnTensor:
        case ExecutionOpcode::kDnnTensorSize:
        case ExecutionOpcode::kGetDnnActivation:
        case ExecutionOpcode::kGetDnnFilter:
        case ExecutionOpcode::kDnnFilterSize:
        case ExecutionOpcode::kGetDnnConvolution:
        case ExecutionOpcode::kGetDnnConvolutionAttribute:
        case ExecutionOpcode::kDnnConvolutionOutput:
        case ExecutionOpcode::kGetDnnPooling:
        case ExecutionOpcode::kDnnPoolingOutput:
        case ExecutionOpcode::kGetDnnOpTensor:
        case ExecutionOpcode::kGetDnnSeqData:
            return true;
        default:
            return false;
    }
}
[[nodiscard]] constexpr bool execution_requires_descriptor(ExecutionOpcode operation) noexcept {
    return operation == ExecutionOpcode::kImportHostMemory ||
           operation == ExecutionOpcode::kLoadModuleFile ||
           operation == ExecutionOpcode::kLoadModuleJit;
}
enum class ExecutionLtObjectKind : std::uint32_t { kLayout = 1, kOperation = 2, kPreference = 3 };
// Library SDK replies and validated private Driver-compute replies have
// different status domains. Keep the permitted operation set in one place.
[[nodiscard]] constexpr bool execution_allows_compute_replacement(
    ExecutionOpcode operation) noexcept {
    return operation == ExecutionOpcode::kSgemm || operation == ExecutionOpcode::kGemmEx ||
           operation == ExecutionOpcode::kStridedGemmEx ||
           operation == ExecutionOpcode::kStridedSgemm || operation == ExecutionOpcode::kLtMatmul;
}
enum class ExecutionBlasAttribute : std::uint32_t {
    kStream = 0,
    kPointerMode = 1,
    kMathMode = 2,
    kVersion = 3,
    kAtomicsMode = 4,
    kSmCountTarget = 5
};
enum class ExecutionMemoryKind : std::uint32_t { kCopy = 0, kFill8 = 1, kFill16 = 2, kFill32 = 3 };
struct ExecutionWireHeader {
    ExecutionMessageKind kind{ExecutionMessageKind::kRequest};
    ExecutionOpcode operation{ExecutionOpcode::kHello};
    std::uint32_t payload_bytes{};
    std::uint64_t session{}, sequence{};
    std::array<std::uint64_t, 2> incarnation{};
};
struct ExecutionWireMessage {
    ExecutionWireHeader header;
    std::vector<std::byte> payload;
};
// Explicit little-endian wire fields, never native structs or process pointers.
[[nodiscard]] std::optional<ExecutionWireHeader> decode_execution_header(
    std::span<const std::byte> bytes, std::size_t max_payload = kExecutionMaxPayloadBytes) noexcept;
[[nodiscard]] std::optional<std::vector<std::byte>> encode_execution_message(
    const ExecutionWireMessage& message,
    std::size_t max_payload = kExecutionMaxPayloadBytes) noexcept;

class ExecutionPayloadWriter final {
   public:
    explicit ExecutionPayloadWriter(std::size_t limit = kExecutionMaxPayloadBytes) noexcept;
    // Reserve total payload capacity without changing encoded bytes. Failure
    // leaves contents unchanged; writes within that capacity need no allocation.
    [[nodiscard]] bool reserve_capacity(std::size_t bytes) noexcept;
    // Reserve a composable field's additional bytes without publishing a prefix.
    [[nodiscard]] bool reserve_additional_capacity(std::size_t bytes) noexcept;
    [[nodiscard]] bool write_u32(std::uint32_t value) noexcept;
    [[nodiscard]] bool write_u64(std::uint64_t value) noexcept;
    [[nodiscard]] bool write_blob(std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] bool write_string(std::string_view value) noexcept;
    [[nodiscard]] std::vector<std::byte> take() noexcept;

   private:
    [[nodiscard]] bool reserve_extra(std::size_t bytes) noexcept;
    std::size_t limit_;
    std::vector<std::byte> bytes_;
};
class ExecutionPayloadReader final {
   public:
    explicit ExecutionPayloadReader(std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] bool read_u32(std::uint32_t& value) noexcept;
    [[nodiscard]] bool read_u64(std::uint64_t& value) noexcept;
    [[nodiscard]] bool read_blob(std::span<const std::byte>& value) noexcept;
    [[nodiscard]] bool read_string(std::string_view& value) noexcept;
    [[nodiscard]] bool empty() const noexcept;

   private:
    std::span<const std::byte> bytes_;
    std::size_t position_{};
};

enum class ExecutionCopyEndpointKind : std::uint32_t {
    kNull = 0,
    kDevice = 1,
    kHost = 2,
    kPageable = 3,
};
struct ExecutionCopyEndpoint {
    ExecutionCopyEndpointKind kind{ExecutionCopyEndpointKind::kNull};
    std::uint64_t connection{}, allocation{}, offset{};
    bool operator==(const ExecutionCopyEndpoint&) const = default;
};
struct ExecutionUnifiedCopy {
    ExecutionCopyEndpoint destination, source;
    std::uint64_t bytes{}, stream{};
    bool is_asynchronous{};
    bool operator==(const ExecutionUnifiedCopy&) const = default;
};
inline constexpr std::size_t kExecutionUnifiedCopyBytes = 76;
// Canonical capabilities only. Authentication, live ranges and CUDA memory
// class are checked by the owner, not supplied by a public pointer value.
[[nodiscard]] bool write_execution_unified_copy(ExecutionPayloadWriter& output,
                                                const ExecutionUnifiedCopy& value) noexcept;
[[nodiscard]] bool read_execution_unified_copy(ExecutionPayloadReader& input,
                                               ExecutionUnifiedCopy& value) noexcept;

enum class ExecutionWireDisposition : std::uint32_t {
    kComplete = 1,
    kRejected = 2,
    // Actual framework compute SDK issue, not an original vendor-library call.
    // status remains in the Driver domain; original_status must be absent.
    kComputeReplacement = 3,
    // Reliable copied admission only: id is the exact command sequence and
    // address its retained public frontier. Neither field proves native issue.
    kAdmissionAccepted = 4,
    // Ordinary ASY1 has NOT been accepted. Empty terminal transport response
    // returns credit for independent progress; never SDK success or GPU proof.
    kAdmissionBusy = 5,
};
enum class ExecutionBootstrapOwnership : std::uint32_t {
    kUnproven = 0,
    kNone = 1,
    kRetained = 2,
    // Metadata carriers: retained metadata, never a CUDA context, device
    // execution reservation, or physical-stop proof. PMM1 resources are owned
    // independently of its connection and of this bootstrap observation.
    kMetadataRetained = 3,
};
// Optional Hello data only. Empty legacy data is not a no-resource proof.
// Pure allocation-free codecs; independent calls may run concurrently.
[[nodiscard]] std::optional<std::array<std::byte, 8>> encode_execution_bootstrap_ownership(
    ExecutionBootstrapOwnership ownership) noexcept;
[[nodiscard]] std::optional<ExecutionBootstrapOwnership> decode_execution_bootstrap_ownership(
    std::span<const std::byte> bytes) noexcept;
struct ExecutionWireArgument {
    std::uint64_t offset{}, bytes{};
};
struct ExecutionWireReply {
    // A completed host-import job carries exactly one u32 quarantine flag in
    // data (0 = known state, 1 = retained uncertainty). Empty/malformed error
    // data is not proof that registration left no resource. Hello instead uses
    // the versioned bootstrap ownership record above.
    ExecutionWireDisposition disposition{ExecutionWireDisposition::kRejected};
    std::int32_t status{};
    std::optional<std::int32_t> original_status;
    // LTM1/DNM1 and Lt/DNN handle lifecycle replies reserve parameter_bytes as a 0/1
    // native-owner quarantine flag with no kernel arguments, not a GPU extent.
    // Process-memory replies use the same quarantine flag, an empty argument
    // list and a versioned output-written observation. Their address field is
    // observed SDK output bits, never evidence of an authenticated capability.
    std::uint64_t id{}, address{}, parameter_bytes{};
    std::vector<ExecutionWireArgument> arguments;
    std::vector<std::byte> data;
};
[[nodiscard]] std::optional<std::vector<std::byte>> encode_execution_reply(
    const ExecutionWireReply& reply, std::size_t limit = kExecutionMaxPayloadBytes) noexcept;
[[nodiscard]] std::optional<ExecutionWireReply> decode_execution_reply(
    std::span<const std::byte> bytes) noexcept;

}  // namespace glimmer::control
// NOLINTEND(performance-enum-size)
