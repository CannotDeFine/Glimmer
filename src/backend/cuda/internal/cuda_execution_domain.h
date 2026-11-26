#pragma once

#include "cuda_execution_dispatcher.h"

#include <cstddef>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace glimmer::backend::cuda {

struct CudaExecutionDomainConfigurationDispatch {
    decltype(&cuCtxGetFlags) get_flags{};
    decltype(&cuCtxSetFlags) set_flags{};
    decltype(&cuCtxGetLimit) get_limit{};
    decltype(&cuCtxSetLimit) set_limit{};
    decltype(&cuCtxGetCacheConfig) get_cache{};
    decltype(&cuCtxSetCacheConfig) set_cache{};
    // Exact legacy ABI without taking a deprecated public declaration's address.
    CUresult(CUDAAPI* get_shared)(CUsharedconfig*) {};
    CUresult(CUDAAPI* set_shared)(CUsharedconfig) {};
};
struct CudaExecutionDomainDispatch {
    decltype(&cuInit) initialize{};
    decltype(&cuDeviceGet) device_get{};
    decltype(&cuCtxCreate) context_create{};
    decltype(&cuCtxSetCurrent) context_set_current{};
    decltype(&cuCtxGetCurrent) context_get_current{};
    decltype(&cuCtxDestroy) context_destroy{};
    // Optional all-or-none cold configuration virtualization. Empty preserves
    // the immutable private-domain contract; partial capability is invalid.
    CudaExecutionDomainConfigurationDispatch configuration{};
};
struct CudaExecutionDomainConfig {
    std::uint64_t identity{};
    int device_ordinal{-1};
    unsigned int flags{CU_CTX_MAP_HOST};
    std::size_t maximum_members{64};
    std::size_t maximum_device_bytes{std::size_t{1024} * 1024 * 1024};
};
struct CudaExecutionDomainLogicalIdentity {
    ExecutionSessionIdentity session;
    std::uint64_t context{};
};
struct CudaExecutionDomainMemberView {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionContext context{};
    CudaExecutionDomainLogicalIdentity logical{};
    std::uint64_t generation{};
    int device_ordinal{-1};
    CUdevice native_device{};
    unsigned int flags{};
    bool is_attached{};
    std::uint64_t configuration_epoch{}, physical_configuration_epoch{};
    bool is_configuration_materialized{}, can_virtualize_configuration{};
    bool has_unobserved_configuration_effects{};
    // Cold logical initialization flags; configuration() tracks later mutations.
    unsigned int logical_flags{};
};
inline constexpr std::size_t kCudaExecutionDomainConfigurationFields = 19;
struct CudaExecutionDomainConfigurationView {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    // Flags/cache/shared-bank mode, followed by the16 exact CUlimit selectors.
    // Unsupported fields are absent and zero. This is metadata, not authority.
    std::uint32_t available{};
    std::array<std::uint64_t, kCudaExecutionDomainConfigurationFields> values{};
};
enum class CudaExecutionDomainConfigurationMutation {
    kFlags,
    kLimit,
    kCache,
    kSharedMemory,
    kResetPersistingCache
};
struct CudaExecutionDomainConfigurationCall {
    // Synchronous borrowed callback storage only; never retained or used for
    // asynchronous SDK work. The original owner supplies both nonnull fields.
    void* state{};
    std::int32_t (*invoke)(void*){};
};
struct CudaExecutionDomainConfigurationResult {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    // Requested Driver/Runtime status, distinct from preparation/proof failure.
    std::optional<std::int32_t> original_status;
};
struct CudaExecutionDomainMemberState;

// Opaque trusted backend capability, never a wire/application handle. Copies
// pin the physical context; dropping one never retires work or refunds bytes.
// Operations are original-owner/PID/thread only. view/owner_activity are CPU
// only; owns_native_scope is exclusively the Session's existing callback scope.
class CudaExecutionDomainMember final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept {
        return state_ != nullptr;
    }
    [[nodiscard]] CudaExecutionDomainMemberView view(bool owns_native_scope = false) const noexcept;
    // Exact opaque owner equality plus healthy attached distinct members and
    // equal materialized snapshots. No SDK, numeric-ID or native-pointer guess.
    [[nodiscard]] bool can_share_configuration(
        const CudaExecutionDomainMember& other) const noexcept;
    [[nodiscard]] std::shared_ptr<CudaExecutionOwnerActivity> owner_activity() const noexcept;
    // Once-only exact logical attachment; no SDK work or resource migration.
    [[nodiscard]] CUresult attach(CudaExecutionDomainLogicalIdentity logical) noexcept;
    // Trusted resource factory accounting. Reserve before SDK allocation;
    // refund only after a clean no-allocation result or proven native cleanup.
    // These metadata operations may run inside an already-owned native scope.
    [[nodiscard]] CUresult reserve_device_bytes(std::size_t bytes) noexcept;
    [[nodiscard]] CUresult release_device_bytes(std::size_t bytes) noexcept;
    // Trusted Session publishes issuing/awaiting-stop/started-native counts,
    // never retreat intent or a host ACK. False requires actual backend proof.
    [[nodiscard]] CUresult set_physical_reservation(bool has_physical_work) noexcept;
    // CPU-only, excluding this member. Used before potentially context-wide
    // cold SDK calls/cleanup; NOT_READY defers without waiting in a native call.
    [[nodiscard]] CUresult other_members_quiescent() const noexcept;
    // Owner-only configuration metadata/native transitions. A different profile
    // requires every real physical reservation (including this member) clear.
    // Equal profiles enter no SDK and can share. Numeric epochs confer no grant.
    [[nodiscard]] CUresult configuration_status(bool owns_native_scope = false) const noexcept;
    // CPU-only canonical values after materialization and native verification.
    // Partial capability, unobserved effects or stale materialization cannot
    // qualify a configuration-specific measurement/cache identity.
    [[nodiscard]] CudaExecutionDomainConfigurationView configuration(
        bool owns_native_scope = false) const noexcept;
    // Read-only profiling checkpoint, including while another member is still
    // physical. Exact final-retirement callbacks may verify their own sample;
    // this never clears whole-cohort unobserved state or supplies a stop proof.
    [[nodiscard]] CUresult verify_profile_configuration(
        const CudaExecutionDomainConfigurationView& expected,
        bool owns_native_scope = false) noexcept;
    [[nodiscard]] CUresult materialize_configuration(bool owns_native_scope = false) noexcept;
    // Read-only native verification after actual whole-cohort quiescence.
    // Never materializes a different member or assigns implicit effects to it.
    [[nodiscard]] CUresult verify_materialized_configuration(
        bool owns_native_scope = false) noexcept;
    [[nodiscard]] CudaExecutionDomainConfigurationResult mutate_configuration(
        CudaExecutionDomainConfigurationMutation mutation, std::uint32_t selector,
        CudaExecutionDomainConfigurationCall call, bool owns_native_scope = false) noexcept;
    // Account already discovered native storage even if it exceeds the budget;
    // saturate on overflow, quarantine, and never invent a successful refund.
    void retain_uncertain_device_bytes(std::size_t bytes) noexcept;
    void quarantine() noexcept;
    // Caller proves all member resources/physical tails retired and bytes zero.
    // Does not destroy context or invalidate other members. Exported copies
    // still pin the domain until dropped, and cannot be reattached or reused.
    [[nodiscard]] CUresult retire() noexcept;

   private:
    friend class CudaExecutionDomainOwner;
    std::shared_ptr<CudaExecutionDomainMemberState> state_;
};
struct CudaExecutionDomainMemberResult {
    CUresult status{CUDA_ERROR_INVALID_HANDLE};
    CudaExecutionDomainMember member;
};
struct CudaExecutionDomainSnapshot {
    CUresult status{CUDA_ERROR_INVALID_CONTEXT};
    std::size_t retained_members{}, attached_members{}, reserved_device_bytes{};
    bool has_context{}, is_closed{}, is_shutdown{}, is_quarantined{};
};

// Original constructing thread/process owns lifecycle. Native callbacks execute
// without metadata locks and under shared nonreentry exclusion. initialize
// requires and restores NULL current context. shutdown requires close and NULL;
// close stops admission but does not mint completion. Destructors make no SDK
// calls and retain an unsettled physical owner rather than releasing it early.
// Bound identities/generations/bytes never wrap; full-context faults are shared.
class CudaExecutionDomainOwner final {
   public:
    CudaExecutionDomainOwner(CudaExecutionDomainDispatch dispatch,
                             CudaExecutionDomainConfig config);
    ~CudaExecutionDomainOwner();
    CudaExecutionDomainOwner(const CudaExecutionDomainOwner&) = delete;
    CudaExecutionDomainOwner& operator=(const CudaExecutionDomainOwner&) = delete;
    [[nodiscard]] CUresult initialize() noexcept;
    [[nodiscard]] CudaExecutionDomainMemberResult create_member(
        CudaExecutionDomainLogicalIdentity logical,
        std::optional<unsigned int> requested_flags = std::nullopt) noexcept;
    [[nodiscard]] CudaExecutionDomainSnapshot snapshot() const noexcept;
    // Process allocations outlive a logical context member. These charges join
    // the same physical budget without pinning that member's retirement.
    [[nodiscard]] CUresult reserve_process_bytes(std::size_t bytes) noexcept;
    [[nodiscard]] CUresult release_process_bytes(std::size_t bytes) noexcept;
    [[nodiscard]] CUresult close() noexcept;
    [[nodiscard]] CUresult shutdown() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend::cuda
