#pragma once

#include "glimmer/control/quota_store.h"

#include <atomic>
#include <memory>

namespace glimmer::control {

// Combines an aggregate quota with an optional narrower task quota. A
// reservation is admitted, committed, cancelled, and released in both stores
// so callers can keep one accounting path for process-local and shared modes.
class CompositeQuota final : public QuotaStore {
   public:
    CompositeQuota(const CompositeQuota&) = delete;
    CompositeQuota& operator=(const CompositeQuota&) = delete;

    CompositeQuota(std::unique_ptr<QuotaStore> aggregate, std::unique_ptr<QuotaStore> task);

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
    std::shared_ptr<QuotaStore> aggregate_;
    std::shared_ptr<QuotaStore> task_;
    std::shared_ptr<std::atomic<bool>> healthy_;
};

}  // namespace glimmer::control
