#pragma once

#include "glimmer/core/quota_ledger.h"

#include <cuda.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace glimmer::interceptor {

struct AllocationIdentity {
    CUdeviceptr device_pointer = 0;
    CUcontext context = nullptr;
    CUdevice device = 0;

    friend bool operator==(const AllocationIdentity&, const AllocationIdentity&) = default;
};

struct AllocationIdentityHash {
    [[nodiscard]] std::size_t operator()(const AllocationIdentity& identity) const noexcept;
};

enum class AllocationScope : std::uint8_t {
    kContextBound,
    kContextIndependent,
};

class AllocationRegistry {
   public:
    enum class ReleaseStatus : std::uint8_t {
        kUnknown,
        kInProgress,
        kStarted,
    };

    struct ReleaseTicket {
        AllocationIdentity identity;
        core::MemoryBytes memory_bytes;
        CUstream stream = nullptr;
    };

    AllocationRegistry() = default;

    AllocationRegistry(const AllocationRegistry&) = delete;
    AllocationRegistry& operator=(const AllocationRegistry&) = delete;

    [[nodiscard]] bool record(AllocationIdentity identity, core::MemoryBytes memory_bytes,
                              AllocationScope scope = AllocationScope::kContextBound);

    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_release(
        AllocationIdentity identity);
    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_release_by_pointer(
        CUdeviceptr device_pointer, CUdevice device);
    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_async_release(
        AllocationIdentity identity, CUstream stream);
    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>>
    begin_async_release_by_pointer(CUdeviceptr device_pointer, CUdevice device, CUstream stream);
    [[nodiscard]] bool complete_release(const ReleaseTicket& ticket);
    [[nodiscard]] bool commit_async_release(const ReleaseTicket& ticket) noexcept;
    void cancel_release(const ReleaseTicket& ticket) noexcept;

    [[nodiscard]] core::MemoryBytes complete_async_releases_for_stream(CUstream stream) noexcept;
    [[nodiscard]] core::MemoryBytes complete_async_releases_for_context(CUcontext context,
                                                                        CUdevice device) noexcept;
    [[nodiscard]] core::MemoryBytes erase_context(CUcontext context) noexcept;

    [[nodiscard]] bool is_accounting_degraded() const noexcept;
    void mark_accounting_degraded() noexcept;

   private:
    struct AllocationRecord {
        core::MemoryBytes memory_bytes;
        AllocationScope scope = AllocationScope::kContextBound;
        bool is_releasing = false;
        bool is_async_release = false;
        bool is_async_release_submitted = false;
        CUstream pending_stream = nullptr;
    };

    mutable std::mutex mutex_;
    std::unordered_map<AllocationIdentity, AllocationRecord, AllocationIdentityHash> records_;
    bool is_accounting_degraded_ = false;
};

}  // namespace glimmer::interceptor
