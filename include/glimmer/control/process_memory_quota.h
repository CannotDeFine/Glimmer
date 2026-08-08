#pragma once

#include "glimmer/core/quota_ledger.h"

#include <optional>

namespace glimmer::control {

struct MemoryInfo {
    core::MemoryBytes total_bytes;
    core::MemoryBytes free_bytes;
};

class MemoryReservation {
   public:
    MemoryReservation(const MemoryReservation&) = delete;
    MemoryReservation& operator=(const MemoryReservation&) = delete;

    MemoryReservation(MemoryReservation&&) noexcept = default;
    MemoryReservation& operator=(MemoryReservation&&) noexcept = default;

    ~MemoryReservation() = default;

    [[nodiscard]] bool commit();
    void cancel();

    [[nodiscard]] bool is_active() const;
    [[nodiscard]] core::MemoryBytes memory_bytes() const;

   private:
    friend class ProcessMemoryQuota;

    explicit MemoryReservation(core::QuotaReservation reservation);

    core::QuotaReservation reservation_;
};

class ProcessMemoryQuota {
   public:
    // Thread-safe. Outstanding reservations must be destroyed before this object.
    explicit ProcessMemoryQuota(core::MemoryBytes limit_bytes);

    [[nodiscard]] std::optional<MemoryReservation> try_reserve(core::MemoryBytes memory_bytes);
    [[nodiscard]] bool release(core::MemoryBytes memory_bytes);
    [[nodiscard]] MemoryInfo get_memory_info(core::MemoryBytes physical_total_bytes,
                                             core::MemoryBytes physical_free_bytes) const;
    [[nodiscard]] core::QuotaUsage usage() const;

   private:
    core::QuotaLedger ledger_;
};

}  // namespace glimmer::control
