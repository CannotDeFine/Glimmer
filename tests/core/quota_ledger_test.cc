#include "glimmer/core/quota_ledger.h"

#include <atomic>
#include <barrier>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using glimmer::core::MemoryBytes;
using glimmer::core::QuotaLedger;
using glimmer::core::QuotaReservation;
using glimmer::core::QuotaUsage;

void expect(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

struct ExpectedUsage {
    MemoryBytes reserved_bytes;
    MemoryBytes allocated_bytes;
    MemoryBytes available_bytes;
};

void expect_usage(const QuotaUsage& usage, const ExpectedUsage& expected) {
    expect(usage.reserved_bytes == expected.reserved_bytes, "unexpected reserved bytes");
    expect(usage.allocated_bytes == expected.allocated_bytes, "unexpected allocated bytes");
    expect(usage.available_bytes() == expected.available_bytes, "unexpected available bytes");
}

void test_committed_reservation_becomes_allocated_usage() {
    QuotaLedger ledger(100);

    auto reservation = ledger.try_reserve(60);
    expect(reservation.has_value(), "reservation should be admitted");
    expect_usage(ledger.usage(),
                 {.reserved_bytes = 60, .allocated_bytes = 0, .available_bytes = 40});

    if (!reservation.has_value()) {
        return;
    }
    expect(reservation.value().commit(), "reservation commit should succeed");
    expect(!reservation.value().is_active(), "committed reservation should be inactive");
    expect_usage(ledger.usage(),
                 {.reserved_bytes = 0, .allocated_bytes = 60, .available_bytes = 40});

    expect(ledger.release(60), "allocated usage should be releasable");
    expect_usage(ledger.usage(),
                 {.reserved_bytes = 0, .allocated_bytes = 0, .available_bytes = 100});
}

void test_rejected_reservation_does_not_change_usage() {
    QuotaLedger ledger(100);
    auto accepted = ledger.try_reserve(75);
    expect(accepted.has_value(), "first reservation should be admitted");

    const auto rejected = ledger.try_reserve(26);
    expect(!rejected.has_value(), "reservation beyond limit should be rejected");
    expect_usage(ledger.usage(),
                 {.reserved_bytes = 75, .allocated_bytes = 0, .available_bytes = 25});
}

void test_cancelled_reservation_restores_capacity() {
    QuotaLedger ledger(100);
    {
        auto reservation = ledger.try_reserve(80);
        expect(reservation.has_value(), "reservation should be admitted");
        expect_usage(ledger.usage(),
                     {.reserved_bytes = 80, .allocated_bytes = 0, .available_bytes = 20});
    }

    expect_usage(ledger.usage(),
                 {.reserved_bytes = 0, .allocated_bytes = 0, .available_bytes = 100});
}

void test_invalid_release_does_not_change_allocated_usage() {
    QuotaLedger ledger(100);
    auto reservation = ledger.try_reserve(40);
    expect(reservation.has_value(), "reservation should be admitted");
    if (!reservation.has_value()) {
        return;
    }
    expect(reservation.value().commit(), "reservation commit should succeed");

    expect(!ledger.release(41), "release larger than usage should fail");
    expect_usage(ledger.usage(),
                 {.reserved_bytes = 0, .allocated_bytes = 40, .available_bytes = 60});
}

void test_concurrent_reservations_never_exceed_limit() {
    constexpr int k_thread_count = 8;
    QuotaLedger ledger(100);
    std::barrier start(k_thread_count);
    std::barrier finish(k_thread_count);
    std::atomic<int> accepted_count = 0;
    std::vector<std::thread> threads;
    threads.reserve(k_thread_count);

    for (int index = 0; index < k_thread_count; ++index) {
        threads.emplace_back([&] {
            start.arrive_and_wait();
            auto reservation = ledger.try_reserve(30);
            if (reservation.has_value()) {
                accepted_count.fetch_add(1, std::memory_order_relaxed);
            }
            finish.arrive_and_wait();
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }

    expect(accepted_count.load(std::memory_order_relaxed) == 3,
           "only three concurrent reservations should fit");
    expect_usage(ledger.usage(),
                 {.reserved_bytes = 0, .allocated_bytes = 0, .available_bytes = 100});
}

void test_reservation_remains_safe_after_ledger_destruction() {
    std::optional<QuotaReservation> reservation;
    {
        QuotaLedger ledger(100);
        reservation = ledger.try_reserve(50);
        expect(reservation.has_value(), "reservation should be admitted");
    }

    if (!reservation.has_value()) {
        return;
    }
    expect(reservation.value().is_active(), "reservation should retain safe internal state");
    reservation.value().cancel();
    expect(!reservation.value().is_active(), "cancelled reservation should be inactive");
}

void test_quota_usage_clamps_invalid_or_overflowing_values() {
    const QuotaUsage over_limit{
        .limit_bytes = 10,
        .reserved_bytes = 9,
        .allocated_bytes = 2,
    };
    expect(over_limit.available_bytes() == 0, "over-limit usage should not underflow");

    const QuotaUsage overflowing{
        .limit_bytes = std::numeric_limits<MemoryBytes>::max(),
        .reserved_bytes = std::numeric_limits<MemoryBytes>::max(),
        .allocated_bytes = 1,
    };
    expect(overflowing.used_bytes() == std::numeric_limits<MemoryBytes>::max(),
           "usage should saturate on overflow");
}

}  // namespace

int main() {
    test_committed_reservation_becomes_allocated_usage();
    test_rejected_reservation_does_not_change_usage();
    test_cancelled_reservation_restores_capacity();
    test_invalid_release_does_not_change_allocated_usage();
    test_concurrent_reservations_never_exceed_limit();
    test_reservation_remains_safe_after_ledger_destruction();
    test_quota_usage_clamps_invalid_or_overflowing_values();
    return EXIT_SUCCESS;
}
