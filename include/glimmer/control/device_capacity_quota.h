#pragma once

#include "glimmer/control/process_memory_quota.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace glimmer::control {

struct DeviceMemoryCapacity {
    core::MemoryBytes total_bytes;
    core::MemoryBytes free_bytes;
};

using DeviceMemoryCapacityResolver = bool (*)(void* context, DeviceId device,
                                              DeviceMemoryCapacity* capacity) noexcept;

// Adds a physical-device capacity boundary to an existing quota composition.
// Capacity is resolved lazily per device so the control module remains
// independent of CUDA and dynamic-loader details. The resolver must not call
// back into this quota store or hold a caller-owned lock.
class DeviceCapacityQuota final : public QuotaStore {
   public:
    DeviceCapacityQuota(DeviceMemoryCapacityResolver resolver, void* context) noexcept;

    using QuotaStore::get_memory_info;
    using QuotaStore::release;
    using QuotaStore::try_reserve;
    using QuotaStore::usage;

    [[nodiscard]] std::optional<MemoryReservation> try_reserve(
        DeviceId device, core::MemoryBytes memory_bytes) override;
    [[nodiscard]] bool release(DeviceId device, core::MemoryBytes memory_bytes) override;
    [[nodiscard]] MemoryInfo get_memory_info(DeviceId device,
                                             core::MemoryBytes physical_total_bytes,
                                             core::MemoryBytes physical_free_bytes) const override;
    [[nodiscard]] core::QuotaUsage usage(DeviceId device) const override;
    [[nodiscard]] bool is_healthy() const noexcept override;

   private:
    [[nodiscard]] bool resolve(DeviceId device, DeviceMemoryCapacity* capacity) const noexcept;
    [[nodiscard]] ProcessMemoryQuota* ensure_device_quota_locked(
        DeviceId device, core::MemoryBytes total_bytes) const;

    DeviceMemoryCapacityResolver resolver_ = nullptr;
    void* context_ = nullptr;
    mutable std::mutex mutex_;
    mutable std::unordered_map<DeviceId, std::unique_ptr<ProcessMemoryQuota>> device_quotas_;
    mutable std::unordered_map<DeviceId, core::MemoryBytes> effective_limits_;
    mutable std::atomic<bool> healthy_ = true;
};

}  // namespace glimmer::control
