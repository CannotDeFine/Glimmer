#include "glimmer/control/device_capacity_quota.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

struct CapacityState {
    glimmer::core::MemoryBytes total_bytes = 100;
    glimmer::core::MemoryBytes free_bytes = 100;
    bool available = true;
};

bool resolve_capacity(void* context, glimmer::control::DeviceId device,
                      glimmer::control::DeviceMemoryCapacity* capacity) noexcept {
    auto* state = static_cast<CapacityState*>(context);
    if (state == nullptr || capacity == nullptr || device != 0 || !state->available) {
        return false;
    }
    *capacity = glimmer::control::DeviceMemoryCapacity{.total_bytes = state->total_bytes,
                                                       .free_bytes = state->free_bytes};
    return true;
}

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_physical_capacity_rejects_when_free_memory_is_low() {
    CapacityState state;
    glimmer::control::DeviceCapacityQuota quota(&resolve_capacity, &state);

    auto reservation = quota.try_reserve(0, 80);
    expect(reservation.has_value(), "reservation below physical capacity should succeed");
    if (!reservation.has_value()) {
        return;
    }
    expect(reservation->commit(), "physical-capacity reservation should commit");
    expect(!quota.try_reserve(0, 21).has_value(),
           "reservation above the physical ledger limit should be rejected");

    state.free_bytes = 10;
    expect(!quota.try_reserve(0, 11).has_value(),
           "reservation above current physical free memory should be rejected");

    const glimmer::control::MemoryInfo info = quota.get_memory_info(0, 100, 10);
    expect(info.total_bytes == 100, "physical capacity should define visible total memory");
    expect(info.free_bytes == 10, "physical free memory should bound visible free memory");

    state.total_bytes = 80;
    state.free_bytes = 70;
    expect(!quota.try_reserve(0, 1).has_value(),
           "a changed physical total should not permit an over-capacity reservation");
    const glimmer::control::MemoryInfo changed_info = quota.get_memory_info(0, 80, 70);
    expect(changed_info.total_bytes == 80,
           "a changed physical total should clamp the visible capacity");
    expect(changed_info.free_bytes == 0,
           "visible free memory should include existing quota usage after a capacity change");
    expect(quota.is_healthy(), "a changed physical total should remain a healthy observation");
    expect(quota.release(0, 80), "physical-capacity release should succeed");
    expect(quota.usage(0).allocated_bytes == 0, "release should restore physical accounting");
}

void test_resolver_failure_fails_closed() {
    CapacityState state{.available = false};
    glimmer::control::DeviceCapacityQuota quota(&resolve_capacity, &state);

    expect(!quota.try_reserve(0, 1).has_value(), "resolver failure should reject allocation");
    expect(!quota.is_healthy(), "resolver failure should mark the quota unhealthy");
    const glimmer::control::MemoryInfo info = quota.get_memory_info(0, 100, 100);
    expect(info.total_bytes == 0 && info.free_bytes == 0,
           "unhealthy physical quota should expose no capacity");
}

}  // namespace

int main() {
    test_physical_capacity_rejects_when_free_memory_is_low();
    test_resolver_failure_fails_closed();
    return EXIT_SUCCESS;
}
