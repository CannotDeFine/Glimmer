#include "glimmer/control/process_memory_quota.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

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

}  // namespace

int main() {
    test_memory_info_is_bounded_by_quota_and_physical_free_memory();
    test_memory_info_does_not_claim_more_than_physical_memory();
    test_rejected_reservation_leaves_memory_info_unchanged();
    return EXIT_SUCCESS;
}
