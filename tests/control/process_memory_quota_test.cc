#include "glimmer/control/process_memory_quota.h"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <utility>

namespace {

using glimmer::control::ProcessMemoryQuota;
using glimmer::core::MemoryBytes;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void test_memory_info_is_bounded_by_quota_and_physical_free_memory() {
    ProcessMemoryQuota quota(100);
    auto reservation = quota.try_reserve(40);
    expect(reservation.has_value(), "reservation should be admitted");
    if (!reservation.has_value()) {
        return;
    }
    expect(reservation.value().commit(), "reservation commit should succeed");

    const auto memory_info = quota.get_memory_info(150, 90);
    expect(memory_info.total_bytes == 100, "quota should determine visible total memory");
    expect(memory_info.free_bytes == 60, "quota usage should determine visible free memory");
}

void test_memory_info_does_not_claim_more_than_physical_memory() {
    ProcessMemoryQuota quota(200);

    const auto memory_info = quota.get_memory_info(120, 70);
    expect(memory_info.total_bytes == 120,
           "physical total memory should bound visible total memory");
    expect(memory_info.free_bytes == 70, "physical free memory should bound visible free memory");
}

void test_rejected_reservation_leaves_memory_info_unchanged() {
    ProcessMemoryQuota quota(100);

    const auto reservation = quota.try_reserve(101);
    expect(!reservation.has_value(), "reservation beyond quota should be rejected");

    const auto memory_info = quota.get_memory_info(100, 100);
    expect(memory_info.total_bytes == 100, "visible total memory should remain unchanged");
    expect(memory_info.free_bytes == 100, "visible free memory should remain unchanged");
}

void test_move_assignment_cancels_the_previous_reservation() {
    ProcessMemoryQuota quota(100);
    auto first = quota.try_reserve(40);
    auto second = quota.try_reserve(20);
    expect(first.has_value() && second.has_value(), "move-assignment reservations should exist");
    if (!first.has_value() || !second.has_value()) {
        return;
    }

    second = std::move(first);
    expect(quota.usage().reserved_bytes == 40,
           "move assignment should cancel the overwritten reservation");
}

void test_abandon_retains_reserved_capacity() {
    ProcessMemoryQuota quota(100);
    auto reservation = quota.try_reserve(60);
    expect(reservation.has_value(), "abandon test reservation should be admitted");
    if (!reservation.has_value()) {
        return;
    }

    reservation->abandon();
    expect(quota.usage().reserved_bytes == 60,
           "abandoned reservation should remain conservatively charged");
    expect(!quota.try_reserve(41).has_value(),
           "abandoned reservation should continue to block over-limit admission");
}

void test_process_quota_is_shared_across_devices() {
    ProcessMemoryQuota quota(100);
    auto reservation = quota.try_reserve(1, 60);
    expect(reservation.has_value(), "process quota should accept a second device identity");
    if (!reservation.has_value()) {
        return;
    }
    expect(reservation->commit(), "second-device process reservation should commit");
    expect(quota.usage(0).allocated_bytes == 60 && quota.usage(1).allocated_bytes == 60,
           "process-local quota should expose one aggregate ledger across devices");
    expect(quota.release(1, 60), "second-device process release should succeed");
}

}  // namespace

int main() {
    test_memory_info_is_bounded_by_quota_and_physical_free_memory();
    test_memory_info_does_not_claim_more_than_physical_memory();
    test_rejected_reservation_leaves_memory_info_unchanged();
    test_move_assignment_cancels_the_previous_reservation();
    test_abandon_retains_reserved_capacity();
    test_process_quota_is_shared_across_devices();
    return EXIT_SUCCESS;
}
