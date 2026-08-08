#include "glimmer/core/quota_ledger.h"

#include <limits>
#include <mutex>
#include <utility>

namespace glimmer::core {

namespace detail {

struct QuotaLedgerState {
    explicit QuotaLedgerState(MemoryBytes limit) : limit_bytes(limit) {}

    const MemoryBytes limit_bytes;
    std::mutex mutex;
    MemoryBytes reserved_bytes = 0;
    MemoryBytes allocated_bytes = 0;
};

}  // namespace detail

MemoryBytes QuotaUsage::used_bytes() const {
    if (reserved_bytes > std::numeric_limits<MemoryBytes>::max() - allocated_bytes) {
        return std::numeric_limits<MemoryBytes>::max();
    }
    return reserved_bytes + allocated_bytes;
}

MemoryBytes QuotaUsage::available_bytes() const {
    const MemoryBytes used = used_bytes();
    return used >= limit_bytes ? 0 : limit_bytes - used;
}

QuotaReservation::QuotaReservation(std::shared_ptr<detail::QuotaLedgerState> state,
                                   MemoryBytes memory_bytes)
    : state_(std::move(state)), memory_bytes_(memory_bytes) {}

QuotaReservation::QuotaReservation(QuotaReservation&& other) noexcept
    : state_(std::move(other.state_)), memory_bytes_(std::exchange(other.memory_bytes_, 0)) {}

QuotaReservation& QuotaReservation::operator=(QuotaReservation&& other) noexcept {
    if (this != &other) {
        cancel();
        state_ = std::move(other.state_);
        memory_bytes_ = std::exchange(other.memory_bytes_, 0);
    }
    return *this;
}

QuotaReservation::~QuotaReservation() {
    cancel();
}

bool QuotaReservation::commit() {
    if (state_ == nullptr) {
        return false;
    }

    if (!QuotaLedger::commit_reservation(state_, memory_bytes_)) {
        return false;
    }
    state_.reset();
    memory_bytes_ = 0;
    return true;
}

void QuotaReservation::cancel() {
    if (state_ == nullptr) {
        return;
    }

    QuotaLedger::cancel_reservation(state_, memory_bytes_);
    state_.reset();
    memory_bytes_ = 0;
}

bool QuotaReservation::is_active() const {
    return state_ != nullptr;
}

MemoryBytes QuotaReservation::memory_bytes() const {
    return memory_bytes_;
}

QuotaLedger::QuotaLedger(MemoryBytes limit_bytes)
    : state_(std::make_shared<detail::QuotaLedgerState>(limit_bytes)) {}

std::optional<QuotaReservation> QuotaLedger::try_reserve(MemoryBytes memory_bytes) {
    std::scoped_lock lock(state_->mutex);
    const QuotaUsage current_usage{
        .limit_bytes = state_->limit_bytes,
        .reserved_bytes = state_->reserved_bytes,
        .allocated_bytes = state_->allocated_bytes,
    };
    if (memory_bytes > current_usage.available_bytes()) {
        return std::nullopt;
    }

    state_->reserved_bytes += memory_bytes;
    return QuotaReservation(state_, memory_bytes);
}

bool QuotaLedger::release(MemoryBytes memory_bytes) {
    std::scoped_lock lock(state_->mutex);
    if (memory_bytes > state_->allocated_bytes) {
        return false;
    }

    state_->allocated_bytes -= memory_bytes;
    return true;
}

QuotaUsage QuotaLedger::usage() const {
    std::scoped_lock lock(state_->mutex);
    return QuotaUsage{
        .limit_bytes = state_->limit_bytes,
        .reserved_bytes = state_->reserved_bytes,
        .allocated_bytes = state_->allocated_bytes,
    };
}

bool QuotaLedger::commit_reservation(const std::shared_ptr<detail::QuotaLedgerState>& state,
                                     MemoryBytes memory_bytes) {
    std::scoped_lock lock(state->mutex);
    if (memory_bytes > state->reserved_bytes) {
        return false;
    }
    state->reserved_bytes -= memory_bytes;
    state->allocated_bytes += memory_bytes;
    return true;
}

void QuotaLedger::cancel_reservation(const std::shared_ptr<detail::QuotaLedgerState>& state,
                                     MemoryBytes memory_bytes) {
    std::scoped_lock lock(state->mutex);
    if (memory_bytes > state->reserved_bytes) {
        return;
    }
    state->reserved_bytes -= memory_bytes;
}

}  // namespace glimmer::core
