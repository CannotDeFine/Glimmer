#include "local_control_wake.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <poll.h>
#include <sys/socket.h>
#include <system_error>
#include <unistd.h>

namespace glimmer::control::internal {

LocalControlWake::LocalControlWake() : process_id(::getpid()) {
    static_assert(std::atomic<bool>::is_always_lock_free);
    if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0, descriptors_.data()) !=
        0)
        throw std::system_error(errno, std::generic_category(), "local control wake socketpair");
}
LocalControlWake::~LocalControlWake() {
    static_cast<void>(close());
}

bool LocalControlWake::notify() noexcept {
    if (process_id != ::getpid() || descriptors_[1] < 0 ||
        has_error_.load(std::memory_order_acquire))
        return false;
    // Each producer publishes its preceding state through this RMW, even if
    // another producer already owns the pending wake byte.
    if (has_pending_wake_.exchange(true, std::memory_order_acq_rel))
        return !has_error_.load(std::memory_order_acquire);
    constexpr char hint = 'W';
    for (std::size_t attempt = 0; attempt < 2; ++attempt) {
        const auto sent = ::send(descriptors_[1], &hint, 1, MSG_DONTWAIT | MSG_NOSIGNAL);
        if (sent == 1 || (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))) return true;
        if (sent < 0 && errno == EINTR) continue;
        break;
    }
    has_error_.store(true, std::memory_order_release);
    return false;
}

ControlWaitOutcome LocalControlWake::consume() noexcept {
    std::array<char, 64> bytes{};
    const auto count = ::recv(descriptors_[0], bytes.data(), bytes.size(), MSG_DONTWAIT);
    if (count > 0 &&
        std::all_of(bytes.begin(), bytes.begin() + count, [](char byte) { return byte == 'W'; })) {
        // Rearm after the read. The acquire synchronizes with every producer
        // coalesced into this wake before the consumer reinspects its state.
        static_cast<void>(has_pending_wake_.exchange(false, std::memory_order_acq_rel));
        return {ControlWaitStatus::kReady, true};
    }
    if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return {ControlWaitStatus::kReady};
    if (count < 0 && errno == EINTR) return {ControlWaitStatus::kInterrupted};
    has_error_.store(true, std::memory_order_release);
    return {ControlWaitStatus::kError};
}

ControlWaitOutcome LocalControlWake::wait_until(ControlWaitPlan plan) noexcept {
    using Clock = OwnedControlExchange::Clock;
    if (process_id != ::getpid()) return {ControlWaitStatus::kRejected};
    if (descriptors_[0] < 0 || has_error_.load(std::memory_order_acquire))
        return {ControlWaitStatus::kError};
    const auto now = Clock::now();
    if (plan.deadline == Clock::time_point::max() ||
        (plan.deadline > now && plan.deadline - now > std::chrono::seconds(60)) ||
        (plan.reply && plan.reply->descriptor < 0))
        return {ControlWaitStatus::kRejected};
    const auto activity = consume();
    if (activity.has_activity || activity.status != ControlWaitStatus::kReady) return activity;
    const auto deadline =
        plan.reply ? std::min(plan.deadline, plan.reply->deadline) : plan.deadline;
    const auto observed = Clock::now();
    const auto remaining = deadline > observed ? deadline - observed : Clock::duration::zero();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(remaining);
    const auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(remaining - seconds);
    const timespec timeout{static_cast<time_t>(seconds.count()),
                           static_cast<long>(nanoseconds.count())};
    std::array<pollfd, 2> items{
        {{descriptors_[0], POLLIN, 0}, {plan.reply ? plan.reply->descriptor : -1, POLLIN, 0}}};
    const int ready = ::ppoll(items.data(), items.size(), &timeout, nullptr);
    if (ready == 0) return {ControlWaitStatus::kDue};
    if (ready < 0 && errno == EINTR) return {ControlWaitStatus::kInterrupted};
    if (ready < 0 || (items[0].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
        has_error_.store(true, std::memory_order_release);
        return {ControlWaitStatus::kError};
    }
    auto outcome = (items[0].revents & POLLIN) != 0 ? consume()
                                                    : ControlWaitOutcome{ControlWaitStatus::kReady};
    outcome.has_reply = items[1].revents != 0;
    outcome.has_peer_closure = (items[1].revents & (POLLERR | POLLHUP | POLLNVAL)) != 0;
    return outcome;
}

bool LocalControlWake::close() noexcept {
    bool is_ok = true;
    for (auto& descriptor : descriptors_) {
        if (descriptor >= 0 && ::close(descriptor) != 0) is_ok = false;
        descriptor = -1;
    }
    if (!is_ok) has_error_.store(true, std::memory_order_release);
    return is_ok;
}

}  // namespace glimmer::control::internal
