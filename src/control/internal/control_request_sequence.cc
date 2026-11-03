#include "control_request_sequence.h"

#include <atomic>
#include <limits>

namespace glimmer::control::internal {

std::uint64_t next_epoch_request_sequence() noexcept {
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    static std::atomic<std::uint64_t> sequence{0};
    auto current = sequence.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<std::uint64_t>::max()) {
        if (sequence.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
            return current + 1;
    }
    return 0;
}

}  // namespace glimmer::control::internal
