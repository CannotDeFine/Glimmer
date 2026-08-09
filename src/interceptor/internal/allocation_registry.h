#pragma once

#include "glimmer/core/quota_ledger.h"

#include <cuda.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <sys/types.h>

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

struct AsyncStreamIdentity {
    CUstream stream = nullptr;
    CUcontext context = nullptr;
    bool per_thread_default_stream = false;
    std::thread::id thread_id{};
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
        AsyncStreamIdentity stream;
    };

    struct DeviceReleaseSummary {
        CUdevice device = 0;
        core::MemoryBytes memory_bytes = 0;
    };

    AllocationRegistry();
    ~AllocationRegistry();

    AllocationRegistry(const AllocationRegistry&) = delete;
    AllocationRegistry& operator=(const AllocationRegistry&) = delete;

    [[nodiscard]] bool record(AllocationIdentity identity, core::MemoryBytes memory_bytes,
                              AllocationScope scope = AllocationScope::kContextBound);

    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_release(
        AllocationIdentity identity);
    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_release_by_pointer(
        CUdeviceptr device_pointer, CUdevice device);
    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>> begin_async_release(
        AllocationIdentity identity, AsyncStreamIdentity stream);
    [[nodiscard]] std::pair<ReleaseStatus, std::optional<ReleaseTicket>>
    begin_async_release_by_pointer(CUdeviceptr device_pointer, CUdevice device,
                                   AsyncStreamIdentity stream);
    [[nodiscard]] bool complete_release(const ReleaseTicket& ticket);
    [[nodiscard]] bool commit_async_release(const ReleaseTicket& ticket) noexcept;
    void cancel_release(const ReleaseTicket& ticket) noexcept;
    [[nodiscard]] bool detach_async_releases_for_stream(const AsyncStreamIdentity& stream) noexcept;

    [[nodiscard]] std::optional<std::vector<DeviceReleaseSummary>>
    complete_async_releases_for_stream_by_device(const AsyncStreamIdentity& stream) noexcept;
    [[nodiscard]] core::MemoryBytes complete_async_releases_for_context(CUcontext context,
                                                                        CUdevice device) noexcept;
    [[nodiscard]] std::optional<std::vector<DeviceReleaseSummary>>
    complete_async_releases_for_context_by_device(CUcontext context) noexcept;
    [[nodiscard]] std::optional<CUdevice> device_for_context(CUcontext context) const noexcept;
    [[nodiscard]] std::optional<std::vector<DeviceReleaseSummary>> erase_context_by_device(
        CUcontext context) noexcept;
    [[nodiscard]] core::MemoryBytes erase_context(CUcontext context) noexcept;
    [[nodiscard]] core::MemoryBytes erase_context(CUcontext context, CUdevice device) noexcept;

    [[nodiscard]] bool is_accounting_degraded() const noexcept;
    void mark_accounting_degraded() const noexcept;

   private:
    void reset_after_fork_if_needed() const noexcept;
    [[nodiscard]] std::mutex& mutex() const noexcept;

    struct AllocationRecord {
        core::MemoryBytes memory_bytes;
        AllocationScope scope = AllocationScope::kContextBound;
        bool is_releasing = false;
        bool is_async_release = false;
        bool is_async_release_submitted = false;
        bool is_async_release_stream_detached = false;
        AsyncStreamIdentity pending_stream;
    };

    mutable std::unique_ptr<std::mutex> mutex_;
    mutable std::unordered_map<AllocationIdentity, AllocationRecord, AllocationIdentityHash>
        records_;
    // Health must remain observable even when acquiring the registry mutex fails.
    mutable std::atomic<bool> is_accounting_degraded_{false};
    mutable pid_t process_id_ = 0;
};

}  // namespace glimmer::interceptor
