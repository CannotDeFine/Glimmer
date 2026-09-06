#include "glimmer/core/adaptive_slo.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

using glimmer::core::AdaptiveSloController;
using glimmer::core::AdaptiveSloObservation;
using glimmer::core::AdaptiveSloOptions;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_disabled_and_priority_filtering() {
    AdaptiveSloController disabled(
        AdaptiveSloOptions{.initial_reserved_slots = 2, .max_reserved_slots = 4});
    expect(!disabled.observe(AdaptiveSloObservation{.priority = 10, .queue_wait_microseconds = 1})
                .has_value(),
           "zero target should disable feedback");
    expect(disabled.reserved_slots() == 2 && disabled.observed_window_samples() == 0,
           "disabled feedback should not change state");

    AdaptiveSloController filtered(AdaptiveSloOptions{.initial_reserved_slots = 1,
                                                      .max_reserved_slots = 3,
                                                      .priority_threshold = 5,
                                                      .target_queue_wait_microseconds = 10,
                                                      .observation_window = 2});
    expect(!filtered.observe(AdaptiveSloObservation{.priority = 4, .queue_wait_microseconds = 100})
                .has_value(),
           "below-threshold work should not be observed");
    expect(filtered.observed_window_samples() == 0,
           "below-threshold work should not consume a window sample");
}

void test_violation_increases_and_clean_window_decreases() {
    AdaptiveSloController controller(AdaptiveSloOptions{.initial_reserved_slots = 1,
                                                        .max_reserved_slots = 3,
                                                        .target_queue_wait_microseconds = 100,
                                                        .observation_window = 4,
                                                        .violation_ratio_percent = 50});
    for (int index = 0; index < 3; ++index) {
        expect(
            !controller
                 .observe(AdaptiveSloObservation{
                     .priority = 1,
                     .queue_wait_microseconds = index < 2 ? std::uint64_t{101} : std::uint64_t{1}})
                 .has_value(),
            "a partial observation window should not update capacity");
    }
    const auto increase =
        controller.observe(AdaptiveSloObservation{.priority = 1, .queue_wait_microseconds = 1});
    expect(increase.has_value() && increase.value() == 2 && controller.reserved_slots() == 2,
           "a violating window should increase the reservation by one step");

    for (int index = 0; index < 3; ++index) {
        expect(
            !controller.observe(AdaptiveSloObservation{.priority = 1, .queue_wait_microseconds = 1})
                 .has_value(),
            "a clean partial window should not update capacity");
    }
    const auto decrease =
        controller.observe(AdaptiveSloObservation{.priority = 1, .queue_wait_microseconds = 1});
    expect(decrease.has_value() && decrease.value() == 1 && controller.reserved_slots() == 1,
           "a clean window should decrease the reservation by one step");
}

void test_bounds_and_normalization() {
    AdaptiveSloController controller(
        AdaptiveSloOptions{.initial_reserved_slots = 9,
                           .min_reserved_slots = 3,
                           .max_reserved_slots = 1,
                           .target_queue_wait_microseconds = 1,
                           .observation_window = 0,
                           .violation_ratio_percent = 0,
                           .increase_step = std::numeric_limits<std::size_t>::max(),
                           .decrease_step = std::numeric_limits<std::size_t>::max()});
    expect(controller.reserved_slots() == 3,
           "constructor should normalize an inverted range and clamp the initial value");
    expect(controller.observe(AdaptiveSloObservation{.priority = 1, .queue_wait_microseconds = 2})
                   .has_value() == false,
           "a range whose bounds are equal should not report a capacity change");
    expect(controller.reserved_slots() == 3,
           "feedback must remain inside normalized reservation bounds");
}

}  // namespace

int main() {
    test_disabled_and_priority_filtering();
    test_violation_increases_and_clean_window_decreases();
    test_bounds_and_normalization();
    return EXIT_SUCCESS;
}
