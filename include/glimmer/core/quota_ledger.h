#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>

namespace glimmer::core {

using MemoryBytes = std::uint64_t;

namespace detail {
struct QuotaLedgerState;
}

struct QuotaUsage {
    MemoryBytes limit_bytes;
    MemoryBytes reserved_bytes;
    MemoryBytes allocated_bytes;

    [[nodiscard]] MemoryBytes used_bytes() const;
    [[nodiscard]] MemoryBytes available_bytes() const;
};

class QuotaLedger;

class QuotaReservation {
   public:
    QuotaReservation(const QuotaReservation&) = delete;
    QuotaReservation& operator=(const QuotaReservation&) = delete;

    QuotaReservation(QuotaReservation&& other) noexcept;
    QuotaReservation& operator=(QuotaReservation&& other) noexcept;

    ~QuotaReservation();

    [[nodiscard]] bool commit();
    void cancel();

    [[nodiscard]] bool is_active() const;
    [[nodiscard]] MemoryBytes memory_bytes() const;

   private:
    friend class QuotaLedger;

    QuotaReservation(std::shared_ptr<detail::QuotaLedgerState> state, MemoryBytes memory_bytes);

    std::shared_ptr<detail::QuotaLedgerState> state_;
    MemoryBytes memory_bytes_;
};

class QuotaLedger {
   public:
    // Thread-safe. Outstanding reservations retain the internal state safely.
    explicit QuotaLedger(MemoryBytes limit_bytes);

    QuotaLedger(const QuotaLedger&) = delete;
    QuotaLedger& operator=(const QuotaLedger&) = delete;
    QuotaLedger(QuotaLedger&&) = delete;
    QuotaLedger& operator=(QuotaLedger&&) = delete;

    [[nodiscard]] std::optional<QuotaReservation> try_reserve(MemoryBytes memory_bytes);
    [[nodiscard]] bool release(MemoryBytes memory_bytes);
    [[nodiscard]] QuotaUsage usage() const;

   private:
    friend class QuotaReservation;

    [[nodiscard]] static bool commit_reservation(
        const std::shared_ptr<detail::QuotaLedgerState>& state, MemoryBytes memory_bytes);
    static void cancel_reservation(const std::shared_ptr<detail::QuotaLedgerState>& state,
                                   MemoryBytes memory_bytes);

    std::shared_ptr<detail::QuotaLedgerState> state_;
};

}  // namespace glimmer::core
