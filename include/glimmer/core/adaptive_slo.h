#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>

namespace glimmer::core {

struct AdaptiveSloOptions {
    std::size_t initial_reserved_slots = 0;
    std::size_t min_reserved_slots = 0;
    std::size_t max_reserved_slots = 0;
    std::uint32_t priority_threshold = 1;
    std::uint64_t target_queue_wait_microseconds = 0;
    std::size_t observation_window = 32;
    // The controller normalizes this value to the inclusive range [1, 100].
    std::uint32_t violation_ratio_percent = 25;
    std::size_t increase_step = 1;
    std::size_t decrease_step = 1;
};

struct AdaptiveSloObservation {
    std::uint32_t priority = 0;
    std::uint64_t queue_wait_microseconds = 0;
};

// Thread-safe feedback controller for priority-capacity reservations. It
// considers only tasks at or above priority_threshold. After each complete
// observation window, a violation ratio at or above violation_ratio_percent
// increases the reservation, while a clean window decreases it. A null
// recommendation means that the current reservation should remain unchanged.
class AdaptiveSloController final {
   public:
    explicit AdaptiveSloController(AdaptiveSloOptions options = {});

    AdaptiveSloController(const AdaptiveSloController&) = delete;
    AdaptiveSloController& operator=(const AdaptiveSloController&) = delete;
    AdaptiveSloController(AdaptiveSloController&&) = delete;
    AdaptiveSloController& operator=(AdaptiveSloController&&) = delete;

    [[nodiscard]] std::optional<std::size_t> observe(
        const AdaptiveSloObservation& observation) noexcept;
    [[nodiscard]] std::size_t reserved_slots() const noexcept;
    [[nodiscard]] std::size_t observed_window_samples() const noexcept;

   private:
    static constexpr std::size_t k_maximum_observation_window = 1'000'000;

    AdaptiveSloOptions options_;
    mutable std::mutex mutex_;
    std::size_t reserved_slots_ = 0;
    std::size_t window_samples_ = 0;
    std::size_t window_violations_ = 0;
};

}  // namespace glimmer::core
