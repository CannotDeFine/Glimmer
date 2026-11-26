#pragma once

#include <cuda.h>
#include <nccl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace glimmer::backend::cuda {

inline constexpr std::size_t kCudaNcclNativeConfigStringLimit = 65535;
inline constexpr std::uint32_t kCudaNcclNativeVersion = 22903;

// This optional private adapter is bound to the audited installed public ABI.
// It does not inspect any vendor-private communicator or group structures.
static_assert(NCCL_VERSION_CODE == kCudaNcclNativeVersion);
static_assert(sizeof(void*) == 8 && sizeof(std::size_t) == 8);
static_assert(sizeof(ncclUniqueId) == 128 && sizeof(ncclResult_t) == 4);
static_assert(sizeof(ncclConfig_t) == 88 && offsetof(ncclConfig_t, size) == 0 &&
              offsetof(ncclConfig_t, magic) == 8 && offsetof(ncclConfig_t, version) == 12 &&
              offsetof(ncclConfig_t, blocking) == 16 && offsetof(ncclConfig_t, netName) == 32 &&
              offsetof(ncclConfig_t, commName) == 48 && offsetof(ncclConfig_t, numRmaCtx) == 84);

enum class CudaNcclNativeOperation : std::uint32_t {
    kGetUniqueId = 1,
    kInitRank,
    kInitRankConfig,
    kGetAsyncError,
    kDestroy,
    kAbort,
    kGroupStart,
    kGroupEnd,
    kAllGather,
    kAllReduce,
    kBcast,
};
struct CudaNcclNativeDispatch {
    decltype(&ncclGetUniqueId) get_unique_id{};
    decltype(&ncclCommInitRank) init_rank{};
    decltype(&ncclCommInitRankConfig) init_rank_config{};
    decltype(&ncclCommGetAsyncError) get_async_error{};
    decltype(&ncclCommDestroy) destroy{};
    decltype(&ncclCommAbort) abort{};
    decltype(&ncclGroupStart) group_start{};
    decltype(&ncclGroupEnd) group_end{};
    decltype(&ncclAllGather) all_gather{};
    decltype(&ncclAllReduce) all_reduce{};
    decltype(&ncclBcast) bcast{};
};

// Scalar bytes outside min(declared_bytes,88) must be zero. The full declared
// size survives even when less than eight. Partial scalar prefixes are retained;
// pointer cuts33..39/49..55 reject. Optional strings distinguish NULL from empty.
struct CudaNcclNativeConfig {
    std::uint64_t declared_bytes{};
    std::uint32_t magic{}, version{};
    std::array<std::int32_t, 14> fields{};
    std::optional<std::string> net_name, comm_name;
};
struct CudaNcclNativeCommunicator {
    ncclComm_t native{};
    // Non-NULL native input requires an immutable, authenticated owner lease.
    // A raw pointer supplied by a wire/frontend is never sufficient authority.
    std::shared_ptr<const void> lifetime;
};
enum class CudaNcclNativeOutputContract : std::uint8_t {
    // Injected/unknown providers may retain cells until a later group boundary.
    kDeferred = 0,
    // Trusted composer pins the installed 2.29.3 implementation: the eleven
    // covered APIs never retain their output-cell pointers after host return.
    // UID/comm/async stores are whole typed fields. No wire value grants this.
    kPinnedHostStable = 1,
};
struct CudaNcclNativeArguments {
    CudaNcclNativeOperation operation{};
    bool has_output{};
    std::array<std::byte, 128> unique_id{};
    std::int32_t ranks{}, rank{}, root{};
    std::uint64_t count{};
    // Private typed inputs only. A future wire decoder must check enum
    // representability BEFORE constructing/copying these arguments: the pinned
    // SDK datatype range is 0..15 and reduction range is 0..INT_MAX. This layer
    // preserves typed native values, not arbitrary out-of-range integer casts.
    ncclDataType_t datatype{ncclInt8};
    ncclRedOp_t reduction{ncclSum};
    CudaNcclNativeCommunicator communicator;
    const void* send{};
    void* receive{};
    cudaStream_t stream{};
    // Required for every collective. Trusted composition must independently
    // validate all extents, aliases, native argument safety, stream/context and
    // progress admission. This packet neither grants nor derives that authority.
    std::shared_ptr<const void> resources_lifetime;
    std::optional<CudaNcclNativeConfig> config;
    CudaNcclNativeOutputContract output_contract{CudaNcclNativeOutputContract::kDeferred};
};

// Full private native output snapshots plus BYTE-DIFFERENCE evidence relative to
// deterministic initial storage (UID/async A5, communicator NULL). Masks do not
// detect equal-value writes and are NOT an arbitrary-error public write oracle.
// Native communicator values are not public tokens or ownership/terminal proof.
struct CudaNcclNativeObservation {
    CudaNcclNativeOperation operation{};
    bool has_output{};
    std::array<std::byte, 128> unique_id{};
    std::array<std::byte, 16> unique_id_changed{};
    ncclComm_t communicator{};
    std::uint32_t communicator_changed{};
    std::int32_t async_error{};
    std::uint32_t async_error_changed{};
    // Independent isolated-cell COW evidence, including equal-value stores.
    // This proves ANY write, not a byte mask, status or resource terminal state.
    // Whole-field projection additionally requires the pinned SDK store contract.
    std::optional<bool> output_written{};
};
struct CudaNcclNativeCallCreation;

// Host-call/storage layer ONLY: no communicator registry, caller authentication,
// group/cohort admission, async polling, context proof, event or SDK cleanup.
// The generic native lane supplies exclusion and before/after context proofs.
// Keep this packet alive through ALL possible grouped/async writes and config
// reads, including deferred outer End and communicator retirement. Host return
// or lane-ticket release alone is not permission to release that storage.
class CudaNcclNativeCall final {
   public:
    [[nodiscard]] static CudaNcclNativeCallCreation create(
        CudaNcclNativeDispatch dispatch, CudaNcclNativeArguments arguments) noexcept;
    ~CudaNcclNativeCall();
    CudaNcclNativeCall(const CudaNcclNativeCall&) = delete;
    CudaNcclNativeCall& operator=(const CudaNcclNativeCall&) = delete;

    // Exactly once on the first bound, nonrecycled worker incarnation. Return
    // carrier status, never cast a requested ncclResult_t into CUresult.
    [[nodiscard]] CUresult invoke() noexcept;
    // CPU-safe after callback return; survives a later generic-lane proof error.
    [[nodiscard]] std::optional<std::int32_t> requested_status() const noexcept;
    [[nodiscard]] bool was_invoked() const noexcept;
    [[nodiscard]] bool has_stable_output() const noexcept;
    // Original worker only, after actual host return. Available ONLY under the
    // trusted pinned host-stable contract; keeps all cell/config owners retained.
    [[nodiscard]] std::optional<CudaNcclNativeObservation> snapshot_after_host_return() noexcept;
    // Original worker only. Trusted caller MUST first prove native/group/async
    // output-cell quiescence. May reread after a later real outer GroupEnd.
    // Never read a live cell from the service owner or infer quiescence here.
    [[nodiscard]] std::optional<CudaNcclNativeObservation> snapshot_after_quiescence() noexcept;

   private:
    struct Implementation;
    explicit CudaNcclNativeCall(std::unique_ptr<Implementation> implementation) noexcept;
    std::unique_ptr<Implementation> implementation_;
};
struct CudaNcclNativeCallCreation {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::shared_ptr<CudaNcclNativeCall> call{};
};

}  // namespace glimmer::backend::cuda
