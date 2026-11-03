#include "owned_control_exchange.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace glimmer::control::internal {
namespace {

using Clock = OwnedControlExchange::Clock;
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
enum class SocketInterest : std::uint8_t { kRead, kWrite };

bool valid_line(std::string_view line) noexcept {
    return !line.empty() && line.size() <= 256 && line.back() == '\n' &&
           line.find('\n') == line.size() - 1 && line.find('\0') == std::string_view::npos;
}

std::chrono::milliseconds remaining(Clock::time_point deadline) noexcept {
    const auto now = Clock::now();
    if (now >= deadline) return std::chrono::milliseconds::zero();
    return std::chrono::ceil<std::chrono::milliseconds>(deadline - now);
}

bool wait_socket(int descriptor, SocketInterest interest, Clock::time_point deadline) noexcept {
    for (;;) {
        const auto budget = remaining(deadline);
        if (budget <= std::chrono::milliseconds::zero()) return false;
        const short events = interest == SocketInterest::kRead ? POLLIN : POLLOUT;
        pollfd item{.fd = descriptor, .events = events, .revents = 0};
        const int result = ::poll(&item, 1, static_cast<int>(budget.count()));
        if (result < 0 && errno == EINTR) continue;
        return result > 0 && (item.revents & events) != 0 &&
               (item.revents & (POLLERR | POLLHUP | POLLNVAL)) == 0 && Clock::now() < deadline;
    }
}

bool connect_socket(int descriptor, std::string_view path, Clock::time_point deadline) noexcept {
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.data(), path.size());
    const auto size = static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) + path.size() + 1);
    if (::connect(descriptor, reinterpret_cast<const sockaddr*>(&address), size) == 0)
        return Clock::now() < deadline;
    // AF_UNIX EAGAIN is backlog exhaustion, not a pending connection.
    if (errno != EINPROGRESS || !wait_socket(descriptor, SocketInterest::kWrite, deadline))
        return false;
    int error = 0;
    socklen_t length = sizeof(error);
    return ::getsockopt(descriptor, SOL_SOCKET, SO_ERROR, &error, &length) == 0 &&
           length == sizeof(error) && error == 0;
}

bool send_open(int descriptor, Clock::time_point deadline) noexcept {
    std::string_view bytes = kOwnerControlOpen;
    while (!bytes.empty()) {
        if (Clock::now() >= deadline) return false;
        const auto sent =
            ::send(descriptor, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
        if (sent > 0) {
            bytes.remove_prefix(static_cast<std::size_t>(sent));
        } else if (sent < 0 && errno == EINTR) {
            continue;
        } else if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!wait_socket(descriptor, SocketInterest::kWrite, deadline)) return false;
        } else {
            return false;
        }
    }
    return Clock::now() < deadline;
}

// Read one byte at a time only during cold startup. Never read ahead into the
// SCM_RIGHTS offer, and never renew the absolute budget after fragments/EINTR.
bool read_negotiation(int descriptor, Clock::time_point deadline, std::string& line) {
    while (line.size() < 256) {
        if (Clock::now() >= deadline) return false;
        char byte = 0;
        const auto received = ::recv(descriptor, &byte, 1, MSG_DONTWAIT);
        if (received == 1) {
            line.push_back(byte);
            if (byte == '\n') return Clock::now() < deadline;
        } else if (received < 0 && errno == EINTR) {
            continue;
        } else if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            if (!wait_socket(descriptor, SocketInterest::kRead, deadline)) return false;
        } else {
            return false;
        }
    }
    return false;
}

}  // namespace

bool ServiceIncarnationBinding::bind(ServiceIncarnation incarnation) noexcept {
    if (incarnation[0] == 0 || incarnation[1] == 0) return false;
    for (std::size_t index = 0; index < words_.size(); ++index) {
        std::uint64_t expected = 0;
        if (!words_[index].compare_exchange_strong(expected, incarnation[index]) &&
            expected != incarnation[index])
            return false;
    }
    return matches(incarnation);
}

bool ServiceIncarnationBinding::matches(ServiceIncarnation incarnation) const noexcept {
    return incarnation[0] != 0 && incarnation[1] != 0 && words_[0].load() == incarnation[0] &&
           words_[1].load() == incarnation[1];
}

bool ServiceIncarnationBinding::is_bound() const noexcept {
    return words_[0].load() != 0 || words_[1].load() != 0;
}

OwnedExchangeTicket::OwnedExchangeTicket(std::shared_ptr<const void> identity,
                                         std::uint64_t generation) noexcept
    : identity_(std::move(identity)), generation_(generation) {}

OwnedExchangeTicket::operator bool() const noexcept {
    return identity_ != nullptr && generation_ != 0;
}

OwnedControlExchange::OwnedControlExchange(ServiceIncarnationBinding& binding)
    : binding_(binding), identity(std::make_shared<const int>(0)), process_id(::getpid()) {}

OwnedControlExchange::~OwnedControlExchange() {
    static_cast<void>(close_transport());
}

bool OwnedControlExchange::close_transport() noexcept {
    channel_.reset();
    if (socket_fd_ >= 0) {
        // Never shutdown: after fork the same open-file description belongs
        // to the parent too. Do not retry close on a possibly reused fd.
        const int descriptor = std::exchange(socket_fd_, -1);
        if (::close(descriptor) != 0) has_cleanup_error_ = true;
    }
    return !has_cleanup_error_;
}

void OwnedControlExchange::quarantine() noexcept {
    state_ = State::kUncertain;
    static_cast<void>(close_transport());
}

OwnedExchangeColdStatus OwnedControlExchange::initialize(OwnedExchangeConfig config) noexcept {
    if (state_ != State::kDisconnected || process_id != ::getpid() || has_cleanup_error_ ||
        config.socket_path.empty() || config.socket_path.size() >= sizeof(sockaddr_un::sun_path) ||
        config.socket_path.find('\0') != std::string_view::npos ||
        config.io_timeout <= std::chrono::milliseconds::zero() ||
        config.io_timeout > std::chrono::milliseconds{60000})
        return OwnedExchangeColdStatus::kRejected;
    const auto startup_time = Clock::now();
    if (startup_time >
        Clock::time_point::max() - std::chrono::duration_cast<Clock::duration>(config.io_timeout))
        return OwnedExchangeColdStatus::kRejected;
    const auto startup_deadline = startup_time + config.io_timeout;
    socket_fd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (socket_fd_ < 0) return OwnedExchangeColdStatus::kRejected;
    auto result = OwnedExchangeColdStatus::kRejected;
    try {
        std::string line;
        if (connect_socket(socket_fd_, config.socket_path, startup_deadline) &&
            send_open(socket_fd_, startup_deadline) &&
            read_negotiation(socket_fd_, startup_deadline, line)) {
            if (line == kOwnerControlUnsupported && !binding_.is_bound()) {
                result = OwnedExchangeColdStatus::kUnsupported;
            } else if (line == kOwnerControlReady &&
                       channel_.receive_offer(socket_fd_, remaining(startup_deadline),
                                              SharedControlMode::kReplyHintV3) &&
                       Clock::now() < startup_deadline && process_id == ::getpid() &&
                       binding_.bind(channel_.service_incarnation())) {
                io_timeout_ = config.io_timeout;
                last_observation_ = Clock::time_point{};
                state_ = State::kReady;
                return OwnedExchangeColdStatus::kReady;
            }
        }
    } catch (...) {
        // Allocation failure is cold rejection, never an implicit request.
        result = OwnedExchangeColdStatus::kRejected;
    }
    if (!close_transport()) return OwnedExchangeColdStatus::kRejected;
    return result;
}

OwnedExchangePublication OwnedControlExchange::begin(std::string_view line,
                                                     Clock::time_point now) noexcept {
    if (state_ != State::kReady || process_id != ::getpid() || has_cleanup_error_ ||
        !binding_.matches(channel_.service_incarnation()) || !valid_line(line) ||
        now < last_observation_ ||
        now > Clock::time_point::max() - std::chrono::duration_cast<Clock::duration>(io_timeout_) ||
        generation_ == std::numeric_limits<std::uint64_t>::max())
        return {};
    const auto publication = channel_.begin_request(socket_fd_, line);
    if (publication == SharedControlPublication::kRejected) {
        state_ = State::kDisconnected;
        static_cast<void>(close_transport());
        return {};
    }
    ++generation_;
    last_observation_ = now;
    deadline_ = now + io_timeout_;
    state_ = State::kPending;
    if (publication == SharedControlPublication::kAmbiguous) quarantine();
    return {publication, OwnedExchangeTicket{identity, generation_}};
}

bool OwnedControlExchange::owns(const OwnedExchangeTicket& ticket) const noexcept {
    return ticket.identity_ == identity && ticket.generation_ != 0 &&
           ticket.generation_ == generation_;
}

OwnedExchangeProgress OwnedControlExchange::poll(const OwnedExchangeTicket& ticket,
                                                 std::string& line,
                                                 Clock::time_point now) noexcept {
    if (!owns(ticket) || process_id != ::getpid()) return OwnedExchangeProgress::kRejected;
    if (state_ == State::kUncertain) return OwnedExchangeProgress::kUncertain;
    if (state_ != State::kPending) return OwnedExchangeProgress::kRejected;
    if (now < last_observation_) {
        quarantine();
        return OwnedExchangeProgress::kUncertain;
    }
    last_observation_ = now;
    try {
        const auto result = channel_.poll_response(socket_fd_, line);
        if (result == SharedControlReadStatus::kLine) {
            state_ = State::kReady;
            return OwnedExchangeProgress::kResponse;
        }
        if (result == SharedControlReadStatus::kIdle && now < deadline_)
            return OwnedExchangeProgress::kPending;
    } catch (...) {
        // A possibly published response was not delivered to the domain layer.
        // Contain uncertainty; never replay after allocation failure.
        quarantine();
        return OwnedExchangeProgress::kUncertain;
    }
    quarantine();
    return OwnedExchangeProgress::kUncertain;
}

std::optional<OwnedExchangeWait> OwnedControlExchange::wait_state(
    const OwnedExchangeTicket& ticket) const noexcept {
    if (!owns(ticket) || state_ != State::kPending || process_id != ::getpid()) return std::nullopt;
    return OwnedExchangeWait{socket_fd_, deadline_};
}

bool OwnedControlExchange::abandon(const OwnedExchangeTicket& ticket) noexcept {
    if (!owns(ticket) || state_ != State::kPending || process_id != ::getpid()) return false;
    quarantine();
    return true;
}

bool OwnedControlExchange::close_idle() noexcept {
    if (state_ != State::kReady && state_ != State::kDisconnected) return false;
    state_ = State::kDisconnected;
    return close_transport();
}

bool OwnedControlExchange::has_cleanup_error() const noexcept {
    return has_cleanup_error_;
}

}  // namespace glimmer::control::internal
