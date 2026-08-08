#include "glimmer/control/process_memory_quota.h"

#include <algorithm>
#include <utility>

namespace glimmer::control {

MemoryReservation::MemoryReservation(core::QuotaReservation reservation)
    : reservation_(std::move(reservation)) {}

bool MemoryReservation::commit() {
    return reservation_.commit();
}

void MemoryReservation::cancel() {
    reservation_.cancel();
}

bool MemoryReservation::is_active() const {
    return reservation_.is_active();
}

core::MemoryBytes MemoryReservation::memory_bytes() const {
    return reservation_.memory_bytes();
}

ProcessMemoryQuota::ProcessMemoryQuota(core::MemoryBytes limit_bytes) : ledger_(limit_bytes) {}

std::optional<MemoryReservation> ProcessMemoryQuota::try_reserve(core::MemoryBytes memory_bytes) {
    auto reservation = ledger_.try_reserve(memory_bytes);
    if (!reservation.has_value()) {
        return std::nullopt;
    }

    return MemoryReservation(std::move(*reservation));
}

bool ProcessMemoryQuota::release(core::MemoryBytes memory_bytes) {
    return ledger_.release(memory_bytes);
}

MemoryInfo ProcessMemoryQuota::get_memory_info(core::MemoryBytes physical_total_bytes,
                                               core::MemoryBytes physical_free_bytes) const {
    const core::QuotaUsage current_usage = ledger_.usage();
    const core::MemoryBytes visible_total_bytes =
        std::min(current_usage.limit_bytes, physical_total_bytes);
    const core::MemoryBytes visible_free_bytes =
        std::min(current_usage.available_bytes(), physical_free_bytes);
    return MemoryInfo{
        .total_bytes = visible_total_bytes,
        .free_bytes = std::min(visible_total_bytes, visible_free_bytes),
    };
}

core::QuotaUsage ProcessMemoryQuota::usage() const {
    return ledger_.usage();
}

}  // namespace glimmer::control
