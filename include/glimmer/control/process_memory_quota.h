#pragma once

#include "glimmer/control/quota_store.h"

namespace glimmer::control {

class ProcessMemoryQuota final : public QuotaStore {
   public:
    // Thread-safe. The process-local ledger intentionally shares one quota across
    // all non-negative devices. Outstanding reservations must be destroyed before
    // this object.
    explicit ProcessMemoryQuota(core::MemoryBytes limit_bytes);

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
    core::QuotaLedger ledger_;
};

}  // namespace glimmer::control
