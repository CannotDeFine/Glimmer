#include "glimmer/core/adaptive_slo.h"

#include <algorithm>

namespace glimmer::core {

AdaptiveSloController::AdaptiveSloController(AdaptiveSloOptions options) : options_(options) {
    options_.max_reserved_slots =
        std::max(options_.max_reserved_slots, options_.min_reserved_slots);
    options_.initial_reserved_slots = std::clamp(
        options_.initial_reserved_slots, options_.min_reserved_slots, options_.max_reserved_slots);
    options_.observation_window =
        std::clamp(options_.observation_window, std::size_t{1}, k_maximum_observation_window);
    options_.violation_ratio_percent = std::clamp(options_.violation_ratio_percent, 1U, 100U);
    options_.increase_step = std::max(options_.increase_step, std::size_t{1});
    options_.decrease_step = std::max(options_.decrease_step, std::size_t{1});
    reserved_slots_ = options_.initial_reserved_slots;
}

std::optional<std::size_t> AdaptiveSloController::observe(
    const AdaptiveSloObservation& observation) noexcept {
    std::scoped_lock lock(mutex_);
    if (options_.target_queue_wait_microseconds == 0 ||
        observation.priority < options_.priority_threshold) {
        return std::nullopt;
    }

    ++window_samples_;
    if (observation.queue_wait_microseconds > options_.target_queue_wait_microseconds) {
        ++window_violations_;
    }
    if (window_samples_ < options_.observation_window) {
        return std::nullopt;
    }

    const std::size_t required_violations =
        (options_.observation_window / 100) * options_.violation_ratio_percent +
        (((options_.observation_window % 100) * options_.violation_ratio_percent) + 99) / 100;
    const bool violated = window_violations_ >= required_violations;
    window_samples_ = 0;
    window_violations_ = 0;

    const std::size_t previous_reserved_slots = reserved_slots_;
    if (violated) {
        const std::size_t available = options_.max_reserved_slots - reserved_slots_;
        reserved_slots_ += std::min(options_.increase_step, available);
    } else if (reserved_slots_ > options_.min_reserved_slots) {
        const std::size_t reduction =
            std::min(options_.decrease_step, reserved_slots_ - options_.min_reserved_slots);
        reserved_slots_ -= reduction;
    }
    if (reserved_slots_ == previous_reserved_slots) {
        return std::nullopt;
    }
    return reserved_slots_;
}

std::size_t AdaptiveSloController::reserved_slots() const noexcept {
    std::scoped_lock lock(mutex_);
    return reserved_slots_;
}

std::size_t AdaptiveSloController::observed_window_samples() const noexcept {
    std::scoped_lock lock(mutex_);
    return window_samples_;
}

}  // namespace glimmer::core
