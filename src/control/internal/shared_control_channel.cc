#include "shared_control_channel.h"

#include "glimmer/control/task_protocol.h"

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <linux/futex.h>
#include <memory>
#include <poll.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace glimmer::control::internal {
namespace {

// The Linux ABI is private and versioned by negotiation. Atomic payload access
// avoids a C++ data race even if a peer violates the single-writer publication
// contract. Only trusted local counters determine expected message sequence.
struct alignas(64) MessageSlot {
    std::atomic<std::uint32_t> sequence{0};
    std::atomic<std::uint32_t> length{0};
    std::array<std::atomic<unsigned char>, kTaskProtocolMaxLineBytes> bytes{};
    ServiceIncarnation incarnation{};
};
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t));
static_assert(alignof(std::atomic<std::uint32_t>) >= alignof(std::uint32_t));
static_assert(std::atomic<unsigned char>::is_always_lock_free);
using Clock = std::chrono::steady_clock;
constexpr int kSizeSeals = F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_SEAL;

[[nodiscard]] bool valid_line(std::string_view line) noexcept {
    return !line.empty() && line.size() <= kTaskProtocolMaxLineBytes && line.back() == '\n' &&
           line.find('\n') == line.size() - 1 && line.find('\0') == std::string_view::npos;
}

enum class HintStatus : std::uint8_t { kHealthy, kClosed, kMalformed };

struct ReplyHintBudget {
    std::uint64_t published_sequence;
    std::uint64_t& received_count;
};

[[nodiscard]] HintStatus drain_reply_hints(int socket_fd, ReplyHintBudget budget) noexcept {
    // At most the previous late hint and this request's hint can be unread:
    // the server sends each hint before it accepts the next request.
    std::array<char, 3> bytes{};
    const auto count = ::recv(socket_fd, bytes.data(), bytes.size(), MSG_DONTWAIT);
    if (count < 0) {
        return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR ? HintStatus::kHealthy
                                                                         : HintStatus::kClosed;
    }
    if (count == 0) return HintStatus::kClosed;
    if (count > 2 || budget.received_count > budget.published_sequence ||
        static_cast<std::uint64_t>(count) > budget.published_sequence - budget.received_count)
        return HintStatus::kMalformed;
    for (ssize_t index = 0; index < count; ++index) {
        if (bytes[static_cast<std::size_t>(index)] != 'R') return HintStatus::kMalformed;
    }
    budget.received_count += static_cast<std::uint64_t>(count);
    return HintStatus::kHealthy;
}

[[nodiscard]] bool socket_is_healthy(int socket_fd) noexcept {
    pollfd descriptor{.fd = socket_fd, .events = POLLIN, .revents = 0};
    const auto result = ::poll(&descriptor, 1, 0);
    return (result >= 0 || errno == EINTR) &&
           (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) == 0;
}

[[nodiscard]] std::optional<std::chrono::microseconds> configured_spin_budget() noexcept {
    const char* value = std::getenv("GLIMMER_CONTROL_SPIN_US");
    if (value == nullptr) return std::chrono::microseconds{0};
    const auto size = ::strnlen(value, 8);
    return parse_shared_control_spin_budget(std::string_view(value, size));
}

void relax_cpu() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

[[nodiscard]] int remaining_ms(Clock::time_point deadline) noexcept {
    const auto remaining = deadline - Clock::now();
    if (remaining <= Clock::duration::zero()) {
        return 0;
    }
    return static_cast<int>(std::chrono::ceil<std::chrono::milliseconds>(remaining).count());
}

[[nodiscard]] bool wait_socket(int socket_fd, short events, Clock::time_point deadline) noexcept {
    pollfd descriptor{.fd = socket_fd, .events = events, .revents = 0};
    while (true) {
        const int timeout_ms = remaining_ms(deadline);
        if (timeout_ms == 0) {
            return false;
        }
        const int result = ::poll(&descriptor, 1, timeout_ms);
        if (result > 0) {
            return (descriptor.revents & events) != 0 &&
                   (descriptor.revents & (POLLERR | POLLHUP | POLLNVAL)) == 0;
        }
        if (result == 0 || errno != EINTR) {
            return false;
        }
    }
}

[[nodiscard]] bool notify(std::atomic<std::uint32_t>& sequence) noexcept {
    return ::syscall(SYS_futex, &sequence, FUTEX_WAKE, 1, nullptr, nullptr, 0) >= 0;
}

[[nodiscard]] SharedControlReadStatus wait_message(const MessageSlot& slot, std::uint64_t expected,
                                                   Clock::time_point deadline, int socket_fd,
                                                   std::chrono::microseconds spin_budget) noexcept {
    // One budget per wait. Neither spurious wakeups nor EINTR replenishes it.
    const auto spin_deadline = spin_budget.count() == 0
                                   ? Clock::time_point{}
                                   : std::min(deadline, Clock::now() + spin_budget);
    auto next_liveness_check = Clock::time_point{};
    const auto after_liveness_failure = [&] {
        // Publication can commit after the loop's first read, immediately
        // followed by peer closure. Exact committed data is still consumable
        // once; closure alone, a missing reply or a future sequence is not.
        return slot.sequence.load(std::memory_order_acquire) == expected
                   ? SharedControlReadStatus::kLine
                   : SharedControlReadStatus::kError;
    };
    while (true) {
        const auto observed = slot.sequence.load(std::memory_order_acquire);
        if (observed == expected) {
            return SharedControlReadStatus::kLine;
        }
        if (observed != expected - 1) {
            return SharedControlReadStatus::kError;
        }
        if (spin_budget.count() != 0) {
            const auto now = Clock::now();
            if (now < spin_deadline) {
                if (now >= next_liveness_check) {
                    pollfd descriptor{.fd = socket_fd, .events = POLLIN, .revents = 0};
                    const auto alive = ::poll(&descriptor, 1, 0);
                    if (alive > 0 || (alive < 0 && errno != EINTR)) {
                        return after_liveness_failure();
                    }
                    next_liveness_check = now + std::chrono::microseconds{50};
                }
                relax_cpu();
                continue;
            }
        }
        const int remaining = remaining_ms(deadline);
        if (remaining == 0) {
            return SharedControlReadStatus::kIdle;
        }
        pollfd descriptor{.fd = socket_fd, .events = POLLIN, .revents = 0};
        const int alive = ::poll(&descriptor, 1, 0);
        if (alive > 0 || (alive < 0 && errno != EINTR)) {
            return after_liveness_failure();
        }
        // Non-private FUTEX_WAIT compares and sleeps atomically, including on
        // the client's read-only response mapping. No notification fd flags
        // can be changed by a peer to turn a nonblocking read into a hang.
        // Recheck socket liveness at most 10 ms after disconnect/shutdown.
        const auto duration =
            std::min(deadline - Clock::now(), Clock::duration(std::chrono::milliseconds{10}));
        if (duration <= Clock::duration::zero()) {
            return SharedControlReadStatus::kIdle;
        }
        const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
        const timespec timeout{.tv_sec = 0, .tv_nsec = static_cast<long>(nanoseconds.count())};
        const auto result =
            ::syscall(SYS_futex, &slot.sequence, FUTEX_WAIT,
                      static_cast<std::uint32_t>(expected - 1), &timeout, nullptr, 0);
        if (result < 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT) {
            return SharedControlReadStatus::kError;
        }
    }
}

[[nodiscard]] bool read_message(const MessageSlot& slot, std::uint64_t sequence,
                                std::string& line) noexcept {
    const auto length = slot.length.load(std::memory_order_relaxed);
    if (length == 0 || length > kTaskProtocolMaxLineBytes) {
        return false;
    }
    try {
        line.resize(length);
        for (std::size_t i = 0; i < length; ++i) {
            line[i] = static_cast<char>(slot.bytes[i].load(std::memory_order_relaxed));
        }
    } catch (...) {
        return false;
    }
    return slot.sequence.load(std::memory_order_acquire) == sequence && line.back() == '\n' &&
           line.find('\n') == line.size() - 1 && line.find('\0') == std::string::npos;
}

[[nodiscard]] bool write_message(MessageSlot& slot, std::uint64_t sequence,
                                 std::string_view line) noexcept {
    if (!valid_line(line)) {
        return false;
    }
    for (std::size_t i = 0; i < line.size(); ++i) {
        slot.bytes[i].store(static_cast<unsigned char>(line[i]), std::memory_order_relaxed);
    }
    slot.length.store(static_cast<std::uint32_t>(line.size()), std::memory_order_relaxed);
    slot.sequence.store(static_cast<std::uint32_t>(sequence), std::memory_order_release);
    return notify(slot.sequence);
}

[[nodiscard]] bool valid_backing(int descriptor, bool is_response) noexcept {
    struct stat metadata {};
    const int seals = ::fcntl(descriptor, F_GET_SEALS);
    const int required = kSizeSeals | (is_response ? F_SEAL_FUTURE_WRITE : 0);
    return seals >= 0 && (seals & required) == required && ::fstat(descriptor, &metadata) == 0 &&
           S_ISREG(metadata.st_mode) && metadata.st_size == static_cast<off_t>(sizeof(MessageSlot));
}

}  // namespace

std::optional<std::chrono::microseconds> parse_shared_control_spin_budget(
    std::string_view value) noexcept {
    if (value.empty() || value.size() > 4) return std::nullopt;
    std::uint32_t microseconds{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), microseconds);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() ||
        microseconds > 1000)
        return std::nullopt;
    return std::chrono::microseconds{microseconds};
}

ServiceIncarnation new_service_incarnation() noexcept {
    ServiceIncarnation incarnation{};
    // Fail closed even on short reads, EINTR or uninitialized entropy. This
    // cold-path operation must not make service startup block indefinitely.
    if (::getrandom(incarnation.data(), sizeof(incarnation), GRND_NONBLOCK) !=
            static_cast<ssize_t>(sizeof(incarnation)) ||
        incarnation[0] == 0 || incarnation[1] == 0) {
        return {};
    }
    return incarnation;
}

SharedControlChannel::~SharedControlChannel() {
    reset();
}

void SharedControlChannel::reset() noexcept {
    for (void** mapping : {&request_mapping_, &response_mapping_}) {
        if (*mapping != nullptr) {
            static_cast<void>(::munmap(*mapping, sizeof(MessageSlot)));
            *mapping = nullptr;
        }
    }
    for (int& descriptor : descriptors_) {
        if (descriptor >= 0) {
            static_cast<void>(::close(descriptor));
            descriptor = -1;
        }
    }
    sequence_ = 0;
    is_server_ = false;
    is_pending_ = false;
    incarnation_ = {};
    spin_budget_ = std::chrono::microseconds{0};
    mode_ = SharedControlMode::kV2;
    negotiated_socket_ = -1;
    process_id_ = 0;
    received_hints_ = 0;
    has_transport_error_ = false;
}

bool SharedControlChannel::active() const noexcept {
    return process_id_ == ::getpid() && request_mapping_ != nullptr && response_mapping_ != nullptr;
}

bool SharedControlChannel::create_server() noexcept {
    return create_server(new_service_incarnation());
}

ServiceIncarnation SharedControlChannel::service_incarnation() const noexcept {
    return incarnation_;
}

bool SharedControlChannel::create_server(ServiceIncarnation incarnation,
                                         SharedControlMode mode) noexcept {
    reset();
    if (incarnation[0] == 0 || incarnation[1] == 0 ||
        (mode != SharedControlMode::kV2 && mode != SharedControlMode::kReplyHintV3)) {
        return false;
    }
    const auto spin_budget = configured_spin_budget();
    if (!spin_budget) return false;
    for (std::size_t i = 0; i < 2; ++i) {
        descriptors_[i] = ::memfd_create("glimmer-control-v2", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if (descriptors_[i] < 0 ||
            ::ftruncate(descriptors_[i], static_cast<off_t>(sizeof(MessageSlot))) != 0) {
            reset();
            return false;
        }
        void* mapping = ::mmap(nullptr, sizeof(MessageSlot), PROT_READ | PROT_WRITE, MAP_SHARED,
                               descriptors_[i], 0);
        if (mapping == MAP_FAILED) {
            reset();
            return false;
        }
        (i == 0 ? request_mapping_ : response_mapping_) = mapping;
        std::construct_at(static_cast<MessageSlot*>(mapping));
        if (i == 1) {
            static_cast<MessageSlot*>(mapping)->incarnation = incarnation;
        }
        const int seals = kSizeSeals | (i == 1 ? F_SEAL_FUTURE_WRITE : 0);
        if (::fcntl(descriptors_[i], F_ADD_SEALS, seals) != 0) {
            reset();
            return false;
        }
    }
    is_server_ = true;
    incarnation_ = incarnation;
    spin_budget_ = *spin_budget;
    mode_ = mode;
    process_id_ = ::getpid();
    return true;
}

bool SharedControlChannel::send_offer(int socket_fd, std::chrono::milliseconds timeout) noexcept {
    if (!active() || !is_server_ || negotiated_socket_ != -1) {
        return false;
    }
    const auto deadline = Clock::now() + timeout;
    char marker = 0;
    if (!wait_socket(socket_fd, POLLIN, deadline) ||
        ::recv(socket_fd, &marker, 1, MSG_DONTWAIT) != 1 || marker != 'O' ||
        !wait_socket(socket_fd, POLLOUT, deadline)) {
        return false;
    }
    marker = sends_reply_hints() ? 'T' : 'S';
    iovec payload{.iov_base = &marker, .iov_len = 1};
    alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(sizeof(descriptors_))> ancillary{};
    msghdr message{};
    message.msg_iov = &payload;
    message.msg_iovlen = 1;
    message.msg_control = ancillary.data();
    message.msg_controllen = ancillary.size();
    cmsghdr* header = CMSG_FIRSTHDR(&message);
    header->cmsg_level = SOL_SOCKET;
    header->cmsg_type = SCM_RIGHTS;
    header->cmsg_len = CMSG_LEN(sizeof(descriptors_));
    std::memcpy(CMSG_DATA(header), descriptors_.data(), sizeof(descriptors_));
    if (::sendmsg(socket_fd, &message, MSG_NOSIGNAL | MSG_DONTWAIT) != 1) return false;
    negotiated_socket_ = socket_fd;
    return true;
}

bool SharedControlChannel::receive_offer(int socket_fd, std::chrono::milliseconds timeout,
                                         SharedControlMode mode) noexcept {
    reset();
    if (mode != SharedControlMode::kV2 && mode != SharedControlMode::kReplyHintV3) return false;
    const auto spin_budget = configured_spin_budget();
    if (!spin_budget) return false;
    struct ucred credentials {};
    socklen_t length = sizeof(credentials);
    if (::getsockopt(socket_fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) != 0 ||
        length != sizeof(credentials) || credentials.uid != ::geteuid() || credentials.pid <= 0) {
        return false;
    }
    const auto deadline = Clock::now() + timeout;
    char marker = 'O';
    if (!wait_socket(socket_fd, POLLOUT, deadline) ||
        ::send(socket_fd, &marker, 1, MSG_NOSIGNAL | MSG_DONTWAIT) != 1 ||
        !wait_socket(socket_fd, POLLIN, deadline)) {
        return false;
    }
    iovec payload{.iov_base = &marker, .iov_len = 1};
    // Extra received descriptors must be closed even on malformed/truncated offers.
    std::array<int, 8> received{};
    alignas(cmsghdr) std::array<unsigned char, CMSG_SPACE(sizeof(received))> ancillary{};
    msghdr message{};
    message.msg_iov = &payload;
    message.msg_iovlen = 1;
    message.msg_control = ancillary.data();
    message.msg_controllen = ancillary.size();
    const ssize_t result = ::recvmsg(socket_fd, &message, MSG_CMSG_CLOEXEC | MSG_DONTWAIT);
    std::size_t count = 0;
    bool is_valid = result == 1 && marker == (mode == SharedControlMode::kV2 ? 'S' : 'T') &&
                    (message.msg_flags & MSG_CTRUNC) == 0;
    for (cmsghdr* header = CMSG_FIRSTHDR(&message); header != nullptr;
         header = CMSG_NXTHDR(&message, header)) {
        if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS ||
            header->cmsg_len < CMSG_LEN(0)) {
            is_valid = false;
            continue;
        }
        const std::size_t bytes = header->cmsg_len - CMSG_LEN(0);
        if (bytes % sizeof(int) != 0 || count + bytes / sizeof(int) > received.size()) {
            is_valid = false;
            continue;
        }
        std::memcpy(received.data() + count, CMSG_DATA(header), bytes);
        count += bytes / sizeof(int);
    }
    if (!is_valid || count != descriptors_.size()) {
        for (std::size_t i = 0; i < count; ++i) {
            static_cast<void>(::close(received[i]));
        }
        return false;
    }
    std::copy_n(received.begin(), descriptors_.size(), descriptors_.begin());
    if (!valid_backing(descriptors_[0], false) || !valid_backing(descriptors_[1], true)) {
        reset();
        return false;
    }
    for (std::size_t i = 0; i < 2; ++i) {
        void* mapping = ::mmap(nullptr, sizeof(MessageSlot), PROT_READ | (i == 0 ? PROT_WRITE : 0),
                               MAP_SHARED, descriptors_[i], 0);
        if (mapping == MAP_FAILED) {
            reset();
            return false;
        }
        (i == 0 ? request_mapping_ : response_mapping_) = mapping;
        if (static_cast<MessageSlot*>(mapping)->sequence.load(std::memory_order_acquire) != 0) {
            reset();
            return false;
        }
    }
    incarnation_ = static_cast<const MessageSlot*>(response_mapping_)->incarnation;
    if (incarnation_[0] == 0 || incarnation_[1] == 0) {
        reset();
        return false;
    }
    spin_budget_ = *spin_budget;
    mode_ = mode;
    process_id_ = ::getpid();
    negotiated_socket_ = socket_fd;
    return true;
}

bool SharedControlChannel::publish_request(std::string_view line) noexcept {
    if (sends_reply_hints() || !active() || is_server_ || is_pending_ || negotiated_socket_ < 0 ||
        sequence_ == std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    // Publication makes the result ambiguous even if its notification fails.
    is_pending_ = true;
    ++sequence_;
    return write_message(*static_cast<MessageSlot*>(request_mapping_), sequence_, line);
}

bool SharedControlChannel::receive_response(int socket_fd, std::uint32_t timeout_ms,
                                            std::string& line) noexcept {
    if (sends_reply_hints() || !active() || is_server_ || !is_pending_ ||
        socket_fd != negotiated_socket_) {
        return false;
    }
    const auto& slot = *static_cast<MessageSlot*>(response_mapping_);
    if (wait_message(slot, sequence_, Clock::now() + std::chrono::milliseconds(timeout_ms),
                     socket_fd, spin_budget_) != SharedControlReadStatus::kLine ||
        !read_message(slot, sequence_, line)) {
        return false;
    }
    is_pending_ = false;
    return true;
}

SharedControlReadStatus SharedControlChannel::receive_request(int socket_fd,
                                                              std::uint32_t timeout_ms,
                                                              std::string& line) noexcept {
    if (!active() || !is_server_ || is_pending_ || socket_fd != negotiated_socket_ ||
        (sends_reply_hints() && has_transport_error_) ||
        sequence_ == std::numeric_limits<std::uint32_t>::max()) {
        return SharedControlReadStatus::kError;
    }
    const auto& slot = *static_cast<MessageSlot*>(request_mapping_);
    const auto status =
        wait_message(slot, sequence_ + 1, Clock::now() + std::chrono::milliseconds(timeout_ms),
                     socket_fd, spin_budget_);
    if (status != SharedControlReadStatus::kLine) {
        return status;
    }
    if (!read_message(slot, sequence_ + 1, line)) {
        return SharedControlReadStatus::kError;
    }
    ++sequence_;
    is_pending_ = true;
    return SharedControlReadStatus::kLine;
}

bool SharedControlChannel::publish_response(std::string_view line) noexcept {
    if (sends_reply_hints() || !active() || !is_server_ || !is_pending_ || negotiated_socket_ < 0) {
        return false;
    }
    if (!write_message(*static_cast<MessageSlot*>(response_mapping_), sequence_, line)) {
        return false;
    }
    is_pending_ = false;
    return true;
}

bool SharedControlChannel::sends_reply_hints() const noexcept {
    return mode_ == SharedControlMode::kReplyHintV3;
}

SharedControlPublication SharedControlChannel::begin_request(int socket_fd,
                                                             std::string_view line) noexcept {
    if (!active() || !sends_reply_hints() || is_server_ || is_pending_ || has_transport_error_ ||
        process_id_ != ::getpid() || socket_fd != negotiated_socket_ || !valid_line(line) ||
        sequence_ == std::numeric_limits<std::uint32_t>::max())
        return SharedControlPublication::kRejected;
    if (drain_reply_hints(socket_fd, {sequence_, received_hints_}) != HintStatus::kHealthy ||
        !socket_is_healthy(socket_fd)) {
        has_transport_error_ = true;
        return SharedControlPublication::kRejected;
    }
    is_pending_ = true;
    ++sequence_;
    if (!write_message(*static_cast<MessageSlot*>(request_mapping_), sequence_, line)) {
        has_transport_error_ = true;
        return SharedControlPublication::kAmbiguous;
    }
    return SharedControlPublication::kPublished;
}

SharedControlReadStatus SharedControlChannel::poll_response(int socket_fd,
                                                            std::string& line) noexcept {
    if (!active() || !sends_reply_hints() || is_server_ || !is_pending_ || has_transport_error_ ||
        process_id_ != ::getpid() || socket_fd != negotiated_socket_)
        return SharedControlReadStatus::kError;
    const auto hint = drain_reply_hints(socket_fd, {sequence_, received_hints_});
    if (hint == HintStatus::kMalformed) {
        has_transport_error_ = true;
        return SharedControlReadStatus::kError;
    }
    // Observe the slot after the notification/liveness checks. A validated
    // reply published before closure remains consumable, but cannot reconnect
    // this endpoint or authorize any later operation on an unhealthy channel.
    const bool is_healthy = hint == HintStatus::kHealthy && socket_is_healthy(socket_fd);
    const auto& slot = *static_cast<const MessageSlot*>(response_mapping_);
    const auto observed = slot.sequence.load(std::memory_order_acquire);
    if (slot.incarnation != incarnation_ || (observed != sequence_ && observed != sequence_ - 1)) {
        has_transport_error_ = true;
        return SharedControlReadStatus::kError;
    }
    if (observed == sequence_) {
        if (!read_message(slot, sequence_, line)) {
            has_transport_error_ = true;
            return SharedControlReadStatus::kError;
        }
        is_pending_ = false;
        has_transport_error_ = !is_healthy;
        return SharedControlReadStatus::kLine;
    }
    if (!is_healthy) {
        has_transport_error_ = true;
        return SharedControlReadStatus::kError;
    }
    return SharedControlReadStatus::kIdle;
}

bool SharedControlChannel::publish_response_with_hint(int socket_fd,
                                                      std::string_view line) noexcept {
    if (!active() || !sends_reply_hints() || !is_server_ || !is_pending_ || has_transport_error_ ||
        process_id_ != ::getpid() || socket_fd != negotiated_socket_)
        return false;
    // The actual response precedes the hint. Even a failed send can leave an
    // observable response, so never retry the domain operation or its report.
    const char hint = 'R';
    if (!write_message(*static_cast<MessageSlot*>(response_mapping_), sequence_, line) ||
        ::send(socket_fd, &hint, 1, MSG_DONTWAIT | MSG_NOSIGNAL) != 1) {
        has_transport_error_ = true;
        return false;
    }
    is_pending_ = false;
    return true;
}

}  // namespace glimmer::control::internal
