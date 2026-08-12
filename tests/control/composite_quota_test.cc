#include "glimmer/control/composite_quota.h"
#include "glimmer/control/process_memory_quota.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>

namespace {

using glimmer::control::CompositeQuota;
using glimmer::control::ProcessMemoryQuota;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

std::unique_ptr<CompositeQuota> make_quota() {
    return std::make_unique<CompositeQuota>(std::make_unique<ProcessMemoryQuota>(100),
                                            std::make_unique<ProcessMemoryQuota>(40));
}

void test_narrower_task_limit_rejects_without_changing_aggregate_usage() {
    auto quota = make_quota();
    auto reservation = quota->try_reserve(41);
    expect(!reservation.has_value(), "task limit should reject an oversized reservation");
    expect(quota->usage().reserved_bytes == 0,
           "rejected task reservation should not charge the aggregate quota");
}

void test_commit_and_release_update_both_limits() {
    auto quota = make_quota();
    auto reservation = quota->try_reserve(40);
    expect(reservation.has_value(), "reservation at the task limit should succeed");
    if (!reservation.has_value()) {
        return;
    }
    expect(reservation->commit(), "composite reservation should commit");
    expect(quota->usage().allocated_bytes == 40,
           "task-visible usage should include the committed allocation");
    expect(quota->get_memory_info(0, 200, 200).total_bytes == 40,
           "visible total should use the narrower task limit");
    expect(quota->get_memory_info(0, 200, 200).free_bytes == 0,
           "visible free memory should reach zero at the task limit");
    expect(quota->release(40), "composite release should update both stores");
    expect(quota->usage().allocated_bytes == 0,
           "task-visible usage should be restored after release");
}

void test_cancellation_updates_both_limits() {
    auto quota = make_quota();
    auto reservation = quota->try_reserve(20);
    expect(reservation.has_value(), "cancellation reservation should succeed");
    if (!reservation.has_value()) {
        return;
    }
    reservation->cancel();
    expect(quota->usage().reserved_bytes == 0, "cancellation should clear the task reservation");
    expect(quota->try_reserve(40).has_value(),
           "cancellation should restore capacity in both stores");
}

void test_reservation_keeps_underlying_quotas_alive() {
    std::optional<glimmer::control::MemoryReservation> reservation;
    {
        auto quota = make_quota();
        reservation = quota->try_reserve(20);
        expect(reservation.has_value(), "lifetime reservation should succeed");
    }
    expect(reservation.has_value() && reservation->commit(),
           "reservation should remain valid after composite destruction");
    if (reservation.has_value()) {
        reservation->cancel();
    }
}

}  // namespace

int main() {
    test_narrower_task_limit_rejects_without_changing_aggregate_usage();
    test_commit_and_release_update_both_limits();
    test_cancellation_updates_both_limits();
    test_reservation_keeps_underlying_quotas_alive();
    return EXIT_SUCCESS;
}
