#include "glimmer/control/process_memory_quota.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace glimmer::control {

namespace {

class ProcessMemoryReservationState final : public detail::MemoryReservationState {
   public:
    explicit ProcessMemoryReservationState(core::QuotaReservation reservation)
        : reservation_(std::move(reservation)) {}

    [[nodiscard]] bool commit() noexcept override {
        try {
            return reservation_.commit();
        } catch (...) {
            return false;
        }
    }

    void cancel() noexcept override {
        reservation_.cancel();
    }

    void abandon() noexcept override {
        reservation_.abandon();
    }

    [[nodiscard]] bool is_active() const noexcept override {
        return reservation_.is_active();
    }

    [[nodiscard]] core::MemoryBytes memory_bytes() const noexcept override {
        return reservation_.memory_bytes();
    }

   private:
    core::QuotaReservation reservation_;
};

}  // namespace

MemoryReservation::MemoryReservation(std::unique_ptr<detail::MemoryReservationState> state) noexcept
    : state_(std::move(state)) {}

MemoryReservation::MemoryReservation(MemoryReservation&& other) noexcept
    : state_(std::move(other.state_)) {}

MemoryReservation& MemoryReservation::operator=(MemoryReservation&& other) noexcept {
    if (this != &other) {
        cancel();
        state_ = std::move(other.state_);
    }
    return *this;
}

MemoryReservation::~MemoryReservation() {
    cancel();
}

bool MemoryReservation::commit() noexcept {
    return state_ != nullptr && state_->commit();
}

void MemoryReservation::cancel() noexcept {
    if (state_ != nullptr) {
        state_->cancel();
    }
}

void MemoryReservation::abandon() noexcept {
    if (state_ != nullptr) {
        state_->abandon();
        state_.reset();
    }
}

bool MemoryReservation::is_active() const noexcept {
    return state_ != nullptr && state_->is_active();
}

core::MemoryBytes MemoryReservation::memory_bytes() const noexcept {
    return state_ == nullptr ? 0 : state_->memory_bytes();
}

ProcessMemoryQuota::ProcessMemoryQuota(core::MemoryBytes limit_bytes) : ledger_(limit_bytes) {}

// The quota API deliberately pairs a device identifier with a byte count.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): device and byte count
// are distinct domain values in the stable quota contract.
std::optional<MemoryReservation> ProcessMemoryQuota::try_reserve(DeviceId device,
                                                                 core::MemoryBytes memory_bytes) {
    if (device < 0) {
        return std::nullopt;
    }

    auto reservation = ledger_.try_reserve(memory_bytes);
    if (!reservation.has_value()) {
        return std::nullopt;
    }

    try {
        return MemoryReservation(
            std::make_unique<ProcessMemoryReservationState>(std::move(*reservation)));
    } catch (...) {
        return std::nullopt;
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): device and byte count
// are distinct domain values in the stable quota contract.
bool ProcessMemoryQuota::release(DeviceId device, core::MemoryBytes memory_bytes) {
    if (device < 0) {
        return false;
    }
    return ledger_.release(memory_bytes);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): device and byte counts
// are distinct domain values in the stable quota contract.
MemoryInfo ProcessMemoryQuota::get_memory_info(DeviceId device,
                                               core::MemoryBytes physical_total_bytes,
                                               core::MemoryBytes physical_free_bytes) const {
    if (device < 0) {
        return MemoryInfo{.total_bytes = 0, .free_bytes = 0};
    }

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

core::QuotaUsage ProcessMemoryQuota::usage(DeviceId device) const {
    if (device < 0) {
        return core::QuotaUsage{};
    }
    return ledger_.usage();
}

bool ProcessMemoryQuota::is_healthy() const noexcept {
    return true;
}

// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace glimmer::control
