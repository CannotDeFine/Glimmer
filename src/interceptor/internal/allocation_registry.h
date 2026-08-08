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
    };

    AllocationRegistry() = default;

    AllocationRegistry(const AllocationRegistry&) = delete;
    AllocationRegistry& operator=(const AllocationRegistry&) = delete;

    [[nodiscard]] bool record(AllocationIdentity identity, core::MemoryBytes memory_bytes);

    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_release(
        AllocationIdentity identity);
    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_release_by_pointer(
        CUdeviceptr device_pointer, CUdevice device);
    [[nodiscard]] bool complete_release(const ReleaseTicket& ticket);
    void cancel_release(const ReleaseTicket& ticket) noexcept;

    [[nodiscard]] core::MemoryBytes erase_context(CUcontext context) noexcept;

    [[nodiscard]] bool is_accounting_degraded() const noexcept;
    void mark_accounting_degraded() noexcept;

   private:
    struct AllocationRecord {
        core::MemoryBytes memory_bytes;
        bool is_releasing = false;
    };

    mutable std::mutex mutex_;
    std::unordered_map<AllocationIdentity, AllocationRecord, AllocationIdentityHash> records_;
    bool is_accounting_degraded_ = false;
};

}  // namespace glimmer::interceptor
