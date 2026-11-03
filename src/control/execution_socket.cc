#include "glimmer/control/execution_socket.h"

#include "internal/process_identity.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <sys/eventfd.h>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::control {
struct ExecutionSocketPollJournal::Implementation {
    const pid_t process{::getpid()}, thread{::gettid()};
    Clock clock;
    Failure failure{Failure::kNone};
    std::vector<Cycle> cycles;
    std::vector<Poll> polls;
    std::vector<Peer> peers;
    std::vector<Read> reads;
    std::vector<Frame> frames;
    std::vector<unsigned char> completed;
    std::size_t cycle_count{}, poll_count{}, peer_count{}, read_count{}, frame_count{}, tokens{};
    std::size_t active{}, active_poll{}, active_read{}, active_peer{}, bytes{};
    std::uint64_t last{};
    explicit Implementation(Limits limits, Clock clock_value) : clock(clock_value) {
        const auto add = [&](std::size_t count, std::size_t maximum, std::size_t width) {
            constexpr std::size_t maximum_bytes = std::size_t{512} * 1024 * 1024;
            if (!count || count > maximum || count > (maximum_bytes - bytes) / width)
                throw std::invalid_argument("poll journal exceeds bounded storage");
            bytes += count * width;
        };
        if (!clock) throw std::invalid_argument("poll journal requires a clock");
        add(limits.cycles, 524288, sizeof(Cycle));
        add(limits.peers, 4194304, sizeof(Peer));
        add(limits.polls, 1048576, sizeof(Poll));
        add(limits.reads, 4194304, sizeof(Read) + sizeof(unsigned char));
        add(limits.frames, 1048576, sizeof(Frame));
        cycles.resize(limits.cycles);
        peers.resize(limits.peers);
        polls.resize(limits.polls);
        reads.resize(limits.reads);
        completed.resize(limits.reads);
        frames.resize(limits.frames);
    }
    bool owns() const noexcept {
        return ::getpid() == process && ::gettid() == thread;
    }
    void fail(Failure value) noexcept {
        if (failure == Failure::kNone) failure = value;
    }
    bool healthy(std::size_t cycle) noexcept {
        if (failure != Failure::kNone || !cycle) return false;
        if (active != cycle) fail(Failure::kOrder);
        return failure == Failure::kNone;
    }
    std::uint64_t now() noexcept {
        if (failure != Failure::kNone) return 0;
        const auto saved = errno;
        std::uint64_t stamp{};
        const bool ok = clock(stamp);
        errno = saved;
        if (!ok || !stamp || stamp < last) {
            fail(Failure::kClock);
            return 0;
        }
        last = stamp;
        return stamp;
    }
};

ExecutionSocketPollJournal::ExecutionSocketPollJournal(Limits limits, Clock clock)
    : implementation_(std::make_unique<Implementation>(limits, clock)) {}
ExecutionSocketPollJournal::~ExecutionSocketPollJournal() = default;
bool ExecutionSocketPollJournal::system_clock(std::uint64_t& now) noexcept {
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0 || value.tv_sec < 0 ||
        static_cast<std::uint64_t>(value.tv_sec) > UINT64_MAX / 1000000000 - 1)
        return false;
    now = static_cast<std::uint64_t>(value.tv_sec) * 1000000000 +
          static_cast<std::uint64_t>(value.tv_nsec);
    return true;
}
ExecutionSocketPollJournal::Failure ExecutionSocketPollJournal::failure() const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return Failure::kOwner;
    if (state.failure != Failure::kNone) return state.failure;
    if (state.active || state.active_poll || state.active_read) return Failure::kOrder;
    return state.tokens == state.frame_count ? Failure::kNone : Failure::kCorrelation;
}
bool ExecutionSocketPollJournal::valid() const noexcept {
    return failure() == Failure::kNone;
}
std::size_t ExecutionSocketPollJournal::storage_bytes() const noexcept {
    return implementation_->owns() ? implementation_->bytes : 0;
}
std::span<const ExecutionSocketPollJournal::Cycle> ExecutionSocketPollJournal::cycles()
    const noexcept {
    const auto& state = *implementation_;
    return state.owns() && !state.active ? std::span{state.cycles}.first(state.cycle_count)
                                         : std::span<const Cycle>{};
}
std::span<const ExecutionSocketPollJournal::Poll> ExecutionSocketPollJournal::polls()
    const noexcept {
    const auto& state = *implementation_;
    return state.owns() && !state.active ? std::span{state.polls}.first(state.poll_count)
                                         : std::span<const Poll>{};
}
std::span<const ExecutionSocketPollJournal::Peer> ExecutionSocketPollJournal::peers()
    const noexcept {
    const auto& state = *implementation_;
    return state.owns() && !state.active ? std::span{state.peers}.first(state.peer_count)
                                         : std::span<const Peer>{};
}
std::span<const ExecutionSocketPollJournal::Read> ExecutionSocketPollJournal::reads()
    const noexcept {
    const auto& state = *implementation_;
    return state.owns() && !state.active ? std::span{state.reads}.first(state.read_count)
                                         : std::span<const Read>{};
}
std::span<const ExecutionSocketPollJournal::Frame> ExecutionSocketPollJournal::frames()
    const noexcept {
    const auto& state = *implementation_;
    return state.owns() && !state.active ? std::span{state.frames}.first(state.frame_count)
                                         : std::span<const Frame>{};
}

bool ExecutionSocketPollJournal::publish(const std::string& output) const noexcept {
    if (!valid() || cycles().empty() || frames().empty()) return false;
    try {
        constexpr std::size_t header_bytes = 128;
        constexpr std::size_t maximum_bytes = std::size_t{512} * 1024 * 1024;
        const auto& state = *implementation_;
        std::size_t total = header_bytes;
        for (const auto& [count, width] : {std::pair{state.cycle_count, sizeof(Cycle)},
                                           {state.poll_count, sizeof(Poll)},
                                           {state.peer_count, sizeof(Peer)},
                                           {state.read_count, sizeof(Read)},
                                           {state.frame_count, sizeof(Frame)}}) {
            if (count > (maximum_bytes - total) / width) return false;
            total += count * width;
        }
        const std::filesystem::path path{output};
        if (output.find('\0') != std::string::npos || !path.is_absolute() ||
            path.filename().empty() || path.filename() == "." || path.filename() == ".." ||
            path.lexically_normal() != path ||
            std::filesystem::canonical(path.parent_path()) != path.parent_path())
            return false;
        const auto identity = current_process_identity();
        if (!identity || identity->pid != static_cast<std::uint32_t>(state.process)) return false;
        ExecutionOwnedDescriptor parent(
            ::open(path.parent_path().c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
        struct stat parent_status {};
        if (parent.get() < 0 || ::fstat(parent.get(), &parent_status) != 0 ||
            !S_ISDIR(parent_status.st_mode) || parent_status.st_uid != ::geteuid())
            return false;
        ExecutionOwnedDescriptor file(::openat(parent.get(), path.filename().c_str(),
                                               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW,
                                               0600));
        if (file.get() < 0) return false;
        struct stat created {};
        const auto owned_file = [](const struct stat& info) {
            return S_ISREG(info.st_mode) && info.st_uid == ::geteuid() &&
                   (info.st_mode & 07777) == 0600 && info.st_nlink == 1;
        };
        if (::fstat(file.get(), &created) != 0 || !owned_file(created) || created.st_size != 0)
            return false;
        std::array<unsigned char, 65536> buffer{};
        std::size_t used{}, published{};
        const auto flush = [&] {
            std::size_t written{};
            while (written < used) {
                const auto count = ::write(file.get(), buffer.data() + written, used - written);
                if (count < 0 && errno == EINTR) continue;
                if (count <= 0) return false;
                written += static_cast<std::size_t>(count);
            }
            published += written;
            used = 0;
            return true;
        };
        const auto words = [&](std::initializer_list<std::uint64_t> values) {
            for (auto value : values) {
                if (used == buffer.size() && !flush()) return false;
                for (unsigned int byte = 0; byte < 8; ++byte) {
                    buffer[used++] = static_cast<unsigned char>(value & 0xff);
                    value >>= 8;
                }
            }
            return true;
        };
        if (!words({0x314c4c4f504d4c47ULL, 1, header_bytes, identity->pid,
                    identity->start_time_ticks, state.cycle_count, state.poll_count,
                    state.peer_count, state.read_count, state.frame_count, total - header_bytes, 0,
                    0, 0, 0, 0}))
            return false;
        for (const auto& row : cycles())
            if (!words({row.id, row.entry, row.exit, row.timeout, row.peer_begin, row.peer_count,
                        row.poll_begin, row.poll_count, row.success}))
                return false;
        for (const auto& row : polls())
            if (!words({row.cycle, row.entry, row.returned, static_cast<std::uint64_t>(row.result),
                        row.error}))
                return false;
        for (const auto& row : peers())
            if (!words({row.cycle, row.connection, row.pid, row.start, row.events, row.revents,
                        row.pending, row.output, row.flags, row.visit, row.alive}))
                return false;
        for (const auto& row : reads())
            if (!words({row.cycle, row.connection, row.token, row.entry, row.returned,
                        row.first_read, static_cast<std::uint64_t>(row.result), row.error}))
                return false;
        for (const auto& row : frames())
            if (!words({row.cycle, row.connection, row.token, row.wire, row.opcode, row.first_read,
                        row.received, row.pid, row.start}))
                return false;
        if (!flush() || published != total || ::fsync(file.get()) != 0) return false;
        struct stat finished {
        }, named{};
        if (::fstat(file.get(), &finished) != 0 || !owned_file(finished) || finished.st_size < 0 ||
            static_cast<std::size_t>(finished.st_size) != total ||
            finished.st_dev != created.st_dev || finished.st_ino != created.st_ino ||
            ::fstatat(parent.get(), path.filename().c_str(), &named, AT_SYMLINK_NOFOLLOW) != 0 ||
            named.st_dev != finished.st_dev || named.st_ino != finished.st_ino)
            return false;
        if (::close(file.release()) != 0) return false;
        return ::fsync(parent.get()) == 0;
    } catch (...) {
        return false;
    }
}

// Only the existing serialized transport call can write a journal. Owner checks
// occur at entry; private hooks cannot migrate to another thread during the call.
struct ExecutionSocketPollRecorder {
    using Journal = ExecutionSocketPollJournal;
    using Failure = Journal::Failure;
    struct Position {
        std::size_t value{};
    };
    struct PeerEvent {
        std::uint64_t connection{};
        short revents{};
    };
    static std::size_t begin(Journal& journal, std::uint64_t timeout) noexcept {
        auto& state = *journal.implementation_;
        if (!state.owns()) state.fail(Failure::kOwner);
        if (state.active) state.fail(Failure::kOrder);
        if (state.cycle_count == state.cycles.size()) state.fail(Failure::kCapacity);
        const auto stamp = state.now();
        if (!stamp) return 0;
        state.active = ++state.cycle_count;
        state.cycles[state.active - 1] = {state.active,     stamp, 0, timeout, state.peer_count, 0,
                                          state.poll_count, 0,     0};
        return state.active;
    }
    struct Scope {
        Journal* journal{};
        std::size_t cycle{};
        bool success{};
        ~Scope() {
            if (!journal || !cycle) return;
            auto& state = *journal->implementation_;
            if (state.healthy(cycle)) {
                if (state.active_poll || state.active_read) state.fail(Failure::kOrder);
                state.cycles[cycle - 1].exit = state.now();
                state.cycles[cycle - 1].success = success;
            }
            state.active = state.active_poll = state.active_read = state.active_peer = 0;
        }
    };
    static void peer(Journal& journal, std::size_t cycle, std::uint64_t connection,
                     TaskPeerIdentity identity, short events, std::size_t pending,
                     std::size_t output, std::uint64_t flags) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return;
        if (state.peer_count == state.peers.size()) return state.fail(Failure::kCapacity);
        state.peers[state.peer_count++] = {cycle,
                                           connection,
                                           identity.pid,
                                           identity.start_time_ticks,
                                           static_cast<std::uint16_t>(events),
                                           0,
                                           pending,
                                           output,
                                           flags,
                                           0,
                                           0};
        ++state.cycles[cycle - 1].peer_count;
    }
    static Position poll_enter(Journal& journal, std::size_t cycle) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return {};
        if (state.active_poll) state.fail(Failure::kOrder);
        if (state.poll_count == state.polls.size()) state.fail(Failure::kCapacity);
        const auto stamp = state.now();
        if (!stamp) return {};
        state.active_poll = ++state.poll_count;
        state.polls[state.active_poll - 1] = {cycle, stamp, 0, 0, 0};
        ++state.cycles[cycle - 1].poll_count;
        return {state.active_poll};
    }
    static void poll_return(Journal& journal, std::size_t cycle, Position token, int result,
                            int error) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return;
        if (!token.value || token.value != state.active_poll) return state.fail(Failure::kOrder);
        auto& row = state.polls[token.value - 1];
        row.returned = state.now();
        row.result = result;
        row.error = result < 0 ? static_cast<std::uint64_t>(error) : 0;
        state.active_poll = 0;
    }
    static void visit(Journal& journal, std::size_t cycle, Position index,
                      PeerEvent event) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return;
        const auto& turn = state.cycles[cycle - 1];
        if (index.value >= turn.peer_count) return state.fail(Failure::kCorrelation);
        const auto selected = turn.peer_begin + index.value;
        auto& row = state.peers[selected];
        if (row.connection != event.connection || row.visit || state.active_poll)
            return state.fail(Failure::kCorrelation);
        row.visit = state.now();
        row.revents = static_cast<std::uint16_t>(event.revents);
        state.active_peer = selected + 1;
    }
    static void alive(Journal& journal, std::size_t cycle, bool value) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return;
        if (!state.active_peer) return state.fail(Failure::kCorrelation);
        state.peers[state.active_peer - 1].alive = value;
    }
    static Position read_enter(Journal& journal, std::size_t cycle,
                               std::uint64_t connection) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return {};
        if (!state.active_peer || state.peers[state.active_peer - 1].connection != connection ||
            state.active_read)
            state.fail(Failure::kCorrelation);
        if (state.read_count == state.reads.size()) state.fail(Failure::kCapacity);
        const auto stamp = state.now();
        if (!stamp) return {};
        state.active_read = ++state.read_count;
        state.reads[state.active_read - 1] = {cycle, connection, 0, stamp, 0, 0, 0, 0};
        return {state.active_read};
    }
    static void read_return(Journal& journal, std::size_t cycle, Position token, ssize_t result,
                            int error) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return;
        if (!token.value || token.value != state.active_read) return state.fail(Failure::kOrder);
        auto& row = state.reads[token.value - 1];
        row.returned = state.now();
        row.result = result;
        row.error = result < 0 ? static_cast<std::uint64_t>(error) : 0;
    }
    static void finish_read(Journal& journal, std::size_t cycle, Position token,
                            std::uint64_t first, std::uint64_t& frame_token) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return;
        if (!token.value || token.value != state.active_read) return state.fail(Failure::kOrder);
        auto& row = state.reads[token.value - 1];
        if (row.result > 0 && !frame_token) {
            if (!first || first < row.returned) return state.fail(Failure::kClock);
            frame_token = token.value;
            row.first_read = first;
            state.last = first;
            ++state.tokens;
        } else if (first) {
            return state.fail(Failure::kCorrelation);
        }
        if (frame_token && (frame_token > token.value || !state.reads[frame_token - 1].first_read ||
                            state.reads[frame_token - 1].connection != row.connection ||
                            state.completed[frame_token - 1]))
            return state.fail(Failure::kCorrelation);
        row.token = frame_token;
        state.active_read = 0;
    }
    static void frame(Journal& journal, std::size_t cycle, std::uint64_t connection,
                      TaskPeerIdentity identity, std::uint64_t token,
                      const ExecutionWireHeader& header, std::uint64_t first,
                      std::uint64_t received) noexcept {
        auto& state = *journal.implementation_;
        if (!state.healthy(cycle)) return;
        if (!token || token > state.read_count || state.completed[token - 1] ||
            state.reads[token - 1].connection != connection ||
            state.reads[token - 1].first_read != first || !first || received < state.last ||
            state.active_read || header.kind != ExecutionMessageKind::kRequest)
            return state.fail(Failure::kCorrelation);
        if (state.frame_count == state.frames.size()) return state.fail(Failure::kCapacity);
        state.frames[state.frame_count++] = {cycle,
                                             connection,
                                             token,
                                             header.sequence,
                                             static_cast<std::uint64_t>(header.operation),
                                             first,
                                             received,
                                             identity.pid,
                                             identity.start_time_ticks};
        state.completed[token - 1] = 1;
        state.last = received;
    }
};

namespace {
using Clock = std::chrono::steady_clock;
void close_fd(int& descriptor) noexcept {
    if (descriptor < 0) return;
    const auto value = std::exchange(descriptor, -1);
    if (::close(value) != 0 && errno != EINTR) {
        try {
            spdlog::error("Execution socket descriptor cleanup failed: errno={}", errno);
        } catch (...) {
            // Cleanup is noexcept even when logging cannot allocate.
            return;
        }
    }
}
bool make_address(std::string_view path, sockaddr_un& address) noexcept {
    if (path.empty() || path.front() != '/' || path.size() >= sizeof(address.sun_path) ||
        path.find('\0') != std::string_view::npos)
        return false;
    address.sun_family = AF_UNIX;
    std::memcpy(address.sun_path, path.data(), path.size());
    return true;
}
bool same_process(const TaskPeerIdentity& identity) noexcept {
    std::uint64_t ticks{};
    return read_process_start_time(identity.pid, &ticks) && ticks == identity.start_time_ticks;
}
struct Channel {
    struct Frame {
        std::vector<std::byte> bytes;
        ExecutionOwnedDescriptor descriptor;
    };
    int descriptor{-1}, process_descriptor{-1};
    TaskPeerIdentity identity;
    std::vector<std::byte> input;
    std::deque<Frame> output;
    ExecutionOwnedDescriptor input_descriptor;
    std::optional<ExecutionWireHeader> header;
    std::size_t output_offset{};
    std::uint64_t first_read_ns{};
    std::uint64_t poll_frame_token{};
    bool is_ready{}, permits_descriptor{};
    Clock::time_point progress{Clock::now()};
    ~Channel() {
        close_fd(descriptor);
        close_fd(process_descriptor);
    }
    bool authenticate(std::optional<std::uint32_t> uid,
                      const TaskPeerIdentity* expected = nullptr) noexcept {
        ucred credentials{};
        socklen_t length = sizeof(credentials);
        if (::getsockopt(descriptor, SOL_SOCKET, SO_PEERCRED, &credentials, &length) != 0 ||
            length != sizeof(credentials) || credentials.pid <= 0 ||
            credentials.uid != uid.value_or(static_cast<std::uint32_t>(::geteuid())))
            return false;
        identity = {static_cast<std::uint32_t>(credentials.pid),
                    static_cast<std::uint32_t>(credentials.uid), 0};
        if (!read_process_start_time(identity.pid, &identity.start_time_ticks) ||
            (expected && identity != *expected))
            return false;
#ifdef SYS_pidfd_open
        process_descriptor = static_cast<int>(::syscall(SYS_pidfd_open, credentials.pid, 0));
        if (process_descriptor < 0 && errno != ENOSYS && errno != EINVAL && errno != EPERM)
            return false;
#endif
        return same_process(identity);
    }
    bool is_alive() const noexcept {
        if (process_descriptor < 0) return same_process(identity);
        pollfd process{process_descriptor, POLLIN, 0};
        int result{};
        do {
            result = ::poll(&process, 1, 0);
        } while (result < 0 && errno == EINTR);
        return result == 0;
    }
    bool credentials_valid(const msghdr& message) noexcept {
        bool has_credentials = false,
             has_invalid = (message.msg_flags & (MSG_CTRUNC | MSG_TRUNC)) != 0;
        // CMSG traversal does not mutate the message; libc's API lacks constness.
        auto* mutable_message = const_cast<msghdr*>(&message);
        for (auto* item = CMSG_FIRSTHDR(mutable_message); item;
             item = CMSG_NXTHDR(mutable_message, item)) {
            if (item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_RIGHTS) {
                if (item->cmsg_len < CMSG_LEN(0)) {
                    has_invalid = true;
                } else {
                    const auto count = (item->cmsg_len - CMSG_LEN(0)) / sizeof(int);
                    if (!permits_descriptor || !input.empty() || input_descriptor.get() >= 0 ||
                        count != 1 || (item->cmsg_len - CMSG_LEN(0)) % sizeof(int))
                        has_invalid = true;
                    for (std::size_t index = 0; index < count; ++index) {
                        int received{};
                        std::memcpy(&received, CMSG_DATA(item) + index * sizeof(int), sizeof(int));
                        ExecutionOwnedDescriptor owned(received);
                        if (!has_invalid && input_descriptor.get() < 0)
                            input_descriptor = std::move(owned);
                        else
                            has_invalid = true;
                    }
                }
            } else if (item->cmsg_level == SOL_SOCKET && item->cmsg_type == SCM_CREDENTIALS &&
                       item->cmsg_len == CMSG_LEN(sizeof(ucred))) {
                ucred credentials{};
                std::memcpy(&credentials, CMSG_DATA(item), sizeof(credentials));
                if (has_credentials || credentials.pid != static_cast<pid_t>(identity.pid) ||
                    credentials.uid != identity.uid)
                    has_invalid = true;
                has_credentials = true;
            } else
                has_invalid = true;
        }
        return has_credentials && !has_invalid && is_alive();
    }
    bool receive(std::size_t limit, bool records_receive_time = false,
                 ExecutionSocketPollJournal* journal = nullptr, std::size_t cycle = 0,
                 std::uint64_t connection = 0) noexcept {
        if (is_ready) return true;
        try {
            // Complete already-readable small frames in one owner turn, without
            // letting a fragmented/large peer monopolize nonblocking progress.
            // EINTR also consumes the attempt budget; never read past one frame.
            constexpr std::size_t maximum_receive_attempts = 8;
            for (std::size_t attempt = 0; attempt < maximum_receive_attempts; ++attempt) {
                const auto target =
                    header ? kExecutionHeaderBytes + header->payload_bytes : kExecutionHeaderBytes;
                std::array<std::byte, 16384> bytes;
                alignas(cmsghdr)
                    std::array<std::byte, CMSG_SPACE(sizeof(ucred)) + CMSG_SPACE(sizeof(int) * 8)>
                        ancillary{};
                iovec data{bytes.data(), std::min(bytes.size(), target - input.size())};
                msghdr message{};
                message.msg_iov = &data;
                message.msg_iovlen = 1;
                message.msg_control = ancillary.data();
                message.msg_controllen = ancillary.size();
                const auto read =
                    journal ? ExecutionSocketPollRecorder::read_enter(*journal, cycle, connection)
                            : ExecutionSocketPollRecorder::Position{};
                const auto count = ::recvmsg(descriptor, &message, MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
                if (journal)
                    ExecutionSocketPollRecorder::read_return(*journal, cycle, read, count, errno);
                const bool is_first =
                    count > 0 && records_receive_time && input.empty() && !first_read_ns;
                if (is_first) {
                    const auto observed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                              Clock::now().time_since_epoch())
                                              .count();
                    first_read_ns = observed > 0 ? static_cast<std::uint64_t>(observed) : 0;
                }
                if (journal)
                    ExecutionSocketPollRecorder::finish_read(
                        *journal, cycle, read, is_first ? first_read_ns : 0, poll_frame_token);
                if (count < 0 && errno == EINTR) continue;
                if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return true;
                if (count <= 0) return false;
                if (!credentials_valid(message)) return false;
                input.insert(input.end(), bytes.begin(), bytes.begin() + count);
                progress = Clock::now();
                if (!header && input.size() == kExecutionHeaderBytes) {
                    header = decode_execution_header(input, limit);
                    if (!header) return false;
                    input.reserve(kExecutionHeaderBytes + header->payload_bytes);
                }
                is_ready = header && input.size() == kExecutionHeaderBytes + header->payload_bytes;
                if (is_ready) {
                    bool requires_descriptor = false;
                    if (permits_descriptor && header->kind == ExecutionMessageKind::kRequest) {
                        const auto request = normalize_execution_request(
                            header->operation, std::span{input}.subspan(kExecutionHeaderBytes),
                            limit);
                        if (!request) return false;
                        requires_descriptor = execution_requires_descriptor(request->operation);
                    }
                    return requires_descriptor == (input_descriptor.get() >= 0);
                }
            }
            return true;
        } catch (...) {
            return false;
        }
    }
    bool send_pending() noexcept {
        if (output.empty()) return true;
        ssize_t count{};
        auto& frame = output.front();
        const auto& bytes = frame.bytes;
        constexpr std::size_t maximum_send_attempts = 8;
        for (std::size_t attempt = 0; attempt < maximum_send_attempts; ++attempt) {
            if (frame.descriptor.get() >= 0) {
                alignas(cmsghdr) std::array<std::byte, CMSG_SPACE(sizeof(int))> ancillary{};
                iovec data{frame.bytes.data() + output_offset, bytes.size() - output_offset};
                msghdr message{};
                message.msg_iov = &data;
                message.msg_iovlen = 1;
                message.msg_control = ancillary.data();
                message.msg_controllen = ancillary.size();
                auto* header = CMSG_FIRSTHDR(&message);
                header->cmsg_level = SOL_SOCKET;
                header->cmsg_type = SCM_RIGHTS;
                header->cmsg_len = CMSG_LEN(sizeof(int));
                const auto attached = frame.descriptor.get();
                std::memcpy(CMSG_DATA(header), &attached, sizeof(attached));
                count = ::sendmsg(descriptor, &message, MSG_DONTWAIT | MSG_NOSIGNAL);
            } else {
                count = ::send(descriptor, bytes.data() + output_offset,
                               bytes.size() - output_offset, MSG_DONTWAIT | MSG_NOSIGNAL);
            }
            if (count >= 0 || errno != EINTR) break;
        }
        // Exhausted interruptions retain the exact unsent bytes and descriptor;
        // the next normal poll uses the existing progress/deadline contract.
        if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) return true;
        if (count <= 0) return false;
        // A positive partial write has already transferred SCM_RIGHTS. Never
        // attach it again when continuing the remaining frame bytes.
        frame.descriptor = ExecutionOwnedDescriptor{};
        output_offset += static_cast<std::size_t>(count);
        progress = Clock::now();
        if (output_offset == bytes.size()) {
            output.pop_front();
            output_offset = 0;
        }
        return true;
    }
    ExecutionWireMessage take() {
        if (!header) throw std::logic_error("ready execution frame has no header");
        const auto& ready_header = *header;
        ExecutionWireMessage message{ready_header, {}};
        if (ready_header.payload_bytes)
            message.payload.assign(input.begin() + kExecutionHeaderBytes, input.end());
        input.clear();
        first_read_ns = 0;
        poll_frame_token = 0;
        header.reset();
        is_ready = false;
        return message;
    }
};
bool pass_credentials(int descriptor) noexcept {
    const int enabled = 1;
    return ::setsockopt(descriptor, SOL_SOCKET, SO_PASSCRED, &enabled, sizeof(enabled)) == 0;
}
}  // namespace

bool execution_uses_preparation_deadline(ExecutionOpcode operation) noexcept {
    return operation == ExecutionOpcode::kLoadPtx || operation == ExecutionOpcode::kLoadBinary ||
           operation == ExecutionOpcode::kLoadModuleFile ||
           operation == ExecutionOpcode::kLoadModuleJit ||
           operation == ExecutionOpcode::kGetFunction;
}

ExecutionOwnedDescriptor::ExecutionOwnedDescriptor(int descriptor) noexcept
    : descriptor_(descriptor) {}
ExecutionOwnedDescriptor::~ExecutionOwnedDescriptor() {
    close_fd(descriptor_);
}
ExecutionOwnedDescriptor::ExecutionOwnedDescriptor(ExecutionOwnedDescriptor&& other) noexcept
    : descriptor_(other.release()) {}
ExecutionOwnedDescriptor& ExecutionOwnedDescriptor::operator=(
    ExecutionOwnedDescriptor&& other) noexcept {
    if (this != &other) {
        close_fd(descriptor_);
        descriptor_ = other.release();
    }
    return *this;
}
int ExecutionOwnedDescriptor::get() const noexcept {
    return descriptor_;
}
int ExecutionOwnedDescriptor::release() noexcept {
    return std::exchange(descriptor_, -1);
}

ExecutionSocketReadiness::ExecutionSocketReadiness(ExecutionOwnedDescriptor descriptor,
                                                   short events) noexcept
    : process_(static_cast<std::uint32_t>(::getpid())),
      descriptor_(std::move(descriptor)),
      events_(events) {}
bool ExecutionSocketReadiness::wait_for_ns(std::uint64_t timeout_ns) const noexcept {
    if (static_cast<std::uint32_t>(::getpid()) != process_ || descriptor_.get() < 0 ||
        !timeout_ns || timeout_ns > 50000)
        return false;
    pollfd descriptor{descriptor_.get(), events_, 0};
    const timespec timeout{0, static_cast<long>(timeout_ns)};
    // Readiness/timeout/error only changes when the caller polls again. It
    // cannot manufacture a reply, reset a deadline or alter channel ownership.
    return ::ppoll(&descriptor, 1, &timeout, nullptr) >= 0 && !(descriptor.revents & POLLNVAL);
}

ExecutionSocketWake::ExecutionSocketWake()
    : process(static_cast<std::uint32_t>(::getpid())),
      descriptor_(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
    if (descriptor_.get() < 0) throw std::runtime_error("execution wake creation failed");
}
int ExecutionSocketWake::descriptor() const noexcept {
    return static_cast<std::uint32_t>(::getpid()) == process ? descriptor_.get() : -1;
}
bool ExecutionSocketWake::notify() const noexcept {
    const auto descriptor = this->descriptor();
    if (descriptor < 0) return false;
    const auto saved = errno;
    const std::uint64_t increment = 1;
    bool is_notified{};
    for (unsigned int attempt = 0; attempt < 8; ++attempt) {
        const auto count = ::write(descriptor, &increment, sizeof(increment));
        if (count < 0 && errno == EINTR) continue;
        // A saturated eventfd is already readable; no additional wake is needed.
        is_notified =
            count == static_cast<ssize_t>(sizeof(increment)) || (count < 0 && errno == EAGAIN);
        break;
    }
    errno = saved;
    return is_notified;
}
bool ExecutionSocketWake::consume() const noexcept {
    const auto descriptor = this->descriptor();
    if (descriptor < 0) return false;
    const auto saved = errno;
    std::uint64_t notifications{};
    bool is_consumed{};
    for (unsigned int attempt = 0; attempt < 8; ++attempt) {
        const auto count = ::read(descriptor, &notifications, sizeof(notifications));
        if (count < 0 && errno == EINTR) continue;
        is_consumed =
            count == static_cast<ssize_t>(sizeof(notifications)) || (count < 0 && errno == EAGAIN);
        break;
    }
    errno = saved;
    return is_consumed;
}

struct ExecutionSocketServer::Implementation {
    struct Peer {
        std::uint64_t id{};
        std::unique_ptr<Channel> channel;
        std::vector<ExecutionWireHeader> pending;
        std::uint64_t frame_received_ns{};
    };
    const ExecutionSocketConfig config;
    std::shared_ptr<ExecutionSocketPollJournal> journal;
    const pid_t process{::getpid()};
    int listener{-1};
    std::uint64_t device{}, inode{}, next_id{1};
    bool has_id_space{true};
    std::vector<Peer> peers;
    explicit Implementation(ExecutionSocketConfig value) : config(std::move(value)) {
        if (!config.max_peers || config.max_peers > 1024 || !config.io_timeout_ms ||
            !config.max_pending_requests || config.max_pending_requests > 64 ||
            !config.max_payload_bytes ||
            config.max_payload_bytes >
                std::numeric_limits<std::uint32_t>::max() - kExecutionSourceContinuationBytes ||
            config.max_payload_bytes > std::numeric_limits<std::size_t>::max() -
                                           kExecutionHeaderBytes -
                                           kExecutionSourceContinuationBytes)
            throw std::invalid_argument("invalid execution socket bounds");
        peers.reserve(config.max_peers);
    }
    bool owns() const noexcept {
        return process == ::getpid();
    }
    void remove_path() noexcept {
        struct stat current {};
        if (inode && ::lstat(config.socket_path.c_str(), &current) == 0 &&
            S_ISSOCK(current.st_mode) && static_cast<std::uint64_t>(current.st_dev) == device &&
            static_cast<std::uint64_t>(current.st_ino) == inode) {
            if (::unlink(config.socket_path.c_str()) != 0 && errno != ENOENT) {
                try {
                    spdlog::error("Execution socket path cleanup failed: errno={}", errno);
                } catch (...) {
                    // Cleanup is noexcept even when logging cannot allocate.
                    inode = 0;
                    return;
                }
            }
        }
        inode = 0;
    }
};
ExecutionSocketServer::ExecutionSocketServer(ExecutionSocketConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}
ExecutionSocketServer::ExecutionSocketServer(ExecutionSocketConfig config,
                                             std::shared_ptr<ExecutionSocketPollJournal> journal)
    : ExecutionSocketServer(std::move(config)) {
    if (!journal || !journal->valid() || !journal->cycles().empty() ||
        !implementation_->config.should_record_receive_time)
        throw std::invalid_argument(
            "diagnostic transport requires a fresh owner journal and receive stamps");
    implementation_->journal = std::move(journal);
}
ExecutionSocketServer::~ExecutionSocketServer() {
    if (!implementation_->owns()) {
        [[maybe_unused]] const auto* child_inherited_state = implementation_.release();
        return;
    }
    stop();
}
bool ExecutionSocketServer::start() noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.listener >= 0) return false;
    sockaddr_un address{};
    if (!make_address(state.config.socket_path, address)) return false;
    state.listener = ::socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (state.listener < 0) return false;
    if (::bind(state.listener, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        close_fd(state.listener);
        return false;
    }
    struct stat created {};
    if (::lstat(state.config.socket_path.c_str(), &created) != 0 || !S_ISSOCK(created.st_mode)) {
        close_fd(state.listener);
        return false;
    }
    state.device = static_cast<std::uint64_t>(created.st_dev);
    state.inode = static_cast<std::uint64_t>(created.st_ino);
    if (::chmod(state.config.socket_path.c_str(), 0600) != 0 || !pass_credentials(state.listener) ||
        ::listen(state.listener, 16) != 0) {
        stop();
        return false;
    }
    return true;
}
bool ExecutionSocketServer::poll_for(std::uint32_t timeout_ms) noexcept {
    return poll_for_ns(static_cast<std::uint64_t>(timeout_ms) * 1000000);
}
bool ExecutionSocketServer::poll_for_ns(std::uint64_t timeout_ns) noexcept {
    auto& state = *implementation_;
    if (!state.owns() || state.listener < 0 ||
        timeout_ns > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) * 1000000)
        return false;
    auto* journal = state.journal.get();
    ExecutionSocketPollRecorder::Scope observation{
        journal, journal ? ExecutionSocketPollRecorder::begin(*journal, timeout_ns) : 0, false};
    const auto cycle = observation.cycle;
    try {
        std::vector<pollfd> descriptors;
        descriptors.reserve(state.peers.size() + 2);
        descriptors.push_back({state.listener, POLLIN, 0});
        if (state.config.wake) {
            const auto descriptor = state.config.wake->descriptor();
            if (descriptor < 0) return false;
            descriptors.push_back({descriptor, POLLIN, 0});
        }
        for (const auto& peer : state.peers) {
            descriptors.push_back(
                {peer.channel->descriptor,
                 static_cast<short>(
                     (!peer.channel->is_ready && peer.pending.size() + peer.channel->output.size() <
                                                     state.config.max_pending_requests
                          ? POLLIN
                          : 0) |
                     (!peer.channel->output.empty() ? POLLOUT : 0)),
                 0});
            if (journal)
                ExecutionSocketPollRecorder::peer(
                    *journal, cycle, peer.id, peer.channel->identity, descriptors.back().events,
                    peer.pending.size(), peer.channel->output.size(),
                    (peer.channel->is_ready ? 1ULL : 0ULL) |
                        (peer.pending.size() + peer.channel->output.size() >=
                                 state.config.max_pending_requests
                             ? 2ULL
                             : 0ULL));
        }
        const auto deadline = Clock::now() + std::chrono::nanoseconds{timeout_ns};
        auto remaining = std::chrono::nanoseconds{timeout_ns};
        int result{};
        do {
            const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(remaining);
            const timespec timeout{static_cast<time_t>(seconds.count()),
                                   static_cast<long>((remaining - seconds).count())};
            const auto poll = journal ? ExecutionSocketPollRecorder::poll_enter(*journal, cycle)
                                      : ExecutionSocketPollRecorder::Position{};
            result = ::ppoll(descriptors.data(), descriptors.size(), &timeout, nullptr);
            if (journal)
                ExecutionSocketPollRecorder::poll_return(*journal, cycle, poll, result, errno);
            if (result >= 0 || errno != EINTR) break;
            remaining = std::max(
                std::chrono::nanoseconds::zero(),
                std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - Clock::now()));
        } while (remaining.count() > 0);
        if (result < 0 && errno == EINTR) {
            result = 0;
            for (auto& descriptor : descriptors) descriptor.revents = 0;
        }
        if (result < 0) return false;
        if (state.config.wake) {
            const auto events = descriptors[1].revents;
            if ((events & (POLLERR | POLLHUP | POLLNVAL)) ||
                ((events & POLLIN) && !state.config.wake->consume()))
                return false;
        }
        for (std::size_t index = 0; index < state.peers.size();) {
            auto& peer = state.peers[index];
            const auto polled = std::ranges::find_if(descriptors, [&](const auto& value) {
                return value.fd == peer.channel->descriptor;
            });
            if (journal) {
                const auto offset = state.config.wake ? 2U : 1U;
                ExecutionSocketPollRecorder::visit(
                    *journal, cycle,
                    {static_cast<std::size_t>(polled - descriptors.begin()) - offset},
                    {peer.id, polled != descriptors.end() ? polled->revents : short{0}});
            }
            bool valid = peer.channel->is_alive();
            if (journal) ExecutionSocketPollRecorder::alive(*journal, cycle, valid);
            if (valid && polled != descriptors.end()) {
                if (polled->revents & (POLLERR | POLLNVAL)) valid = false;
                if (valid && (polled->revents & POLLIN)) {
                    valid = peer.channel->receive(state.config.max_payload_bytes,
                                                  state.config.should_record_receive_time, journal,
                                                  cycle, peer.id);
                    if (valid && peer.channel->is_ready &&
                        state.config.should_record_receive_time) {
                        const auto observed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                  Clock::now().time_since_epoch())
                                                  .count();
                        peer.frame_received_ns =
                            observed > 0 ? static_cast<std::uint64_t>(observed) : 0;
                        if (journal) {
                            const auto ready_header = peer.channel->header;
                            if (!ready_header) {
                                valid = false;
                            } else {
                                ExecutionSocketPollRecorder::frame(
                                    *journal, cycle, peer.id, peer.channel->identity,
                                    peer.channel->poll_frame_token, *ready_header,
                                    peer.channel->first_read_ns, peer.frame_received_ns);
                            }
                        }
                    }
                }
                if (valid && (polled->revents & POLLOUT)) valid = peer.channel->send_pending();
                if (polled->revents & POLLHUP) valid = false;
            }
            const bool has_partial_io = (!peer.channel->input.empty() && !peer.channel->is_ready) ||
                                        !peer.channel->output.empty();
            if (has_partial_io && Clock::now() - peer.channel->progress >
                                      std::chrono::milliseconds{state.config.io_timeout_ms})
                valid = false;
            if (!valid)
                state.peers.erase(state.peers.begin() + static_cast<std::ptrdiff_t>(index));
            else
                ++index;
        }
        if (descriptors.front().revents & POLLIN) {
            for (std::size_t count = 0; count < state.config.max_peers; ++count) {
                auto peer = std::make_unique<Channel>();
                const int accepted =
                    ::accept4(state.listener, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
                if (accepted < 0) {
                    if (errno == EINTR) continue;
                    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
                    return false;
                }
                peer->descriptor = accepted;
                peer->permits_descriptor = true;
                if (state.peers.size() == state.config.max_peers || !state.has_id_space ||
                    !pass_credentials(accepted) || !peer->authenticate(state.config.allowed_uid))
                    continue;
                const auto id = state.next_id;
                if (id == std::numeric_limits<std::uint64_t>::max())
                    state.has_id_space = false;
                else
                    ++state.next_id;
                std::vector<ExecutionWireHeader> pending;
                pending.reserve(state.config.max_pending_requests);
                state.peers.push_back({id, std::move(peer), std::move(pending)});
            }
        }
        observation.success = true;
        return true;
    } catch (...) {
        return false;
    }
}
std::optional<ExecutionReceivedRequest> ExecutionSocketServer::take_request() noexcept {
    return take_request_impl(nullptr);
}
std::optional<ExecutionSocketRequestView> ExecutionSocketServer::peek_request(
    std::uint64_t connection) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return std::nullopt;
    for (const auto& peer : state.peers) {
        const auto ready_header = peer.channel->header;
        if (peer.id == connection && peer.channel->is_ready && ready_header)
            return ExecutionSocketRequestView{
                peer.id, peer.channel->identity, *ready_header,
                std::span<const std::byte>{peer.channel->input}.subspan(kExecutionHeaderBytes),
                peer.channel->input_descriptor.get() >= 0};
    }
    return std::nullopt;
}
std::optional<ExecutionReceivedRequest> ExecutionSocketServer::take_request(
    const ExecutionSocketRequestView& expected) noexcept {
    return take_request_impl(&expected);
}
std::optional<ExecutionReceivedRequest> ExecutionSocketServer::take_request_impl(
    const ExecutionSocketRequestView* expected) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return std::nullopt;
    try {
        for (auto& peer : state.peers)
            if (peer.channel->is_ready && (!expected || peer.id == expected->connection)) {
                const auto ready_header = peer.channel->header;
                if (!ready_header) return std::nullopt;
                if (expected) {
                    const auto& header = *ready_header;
                    const auto& wanted = expected->header;
                    if (peer.channel->identity != expected->peer || header.kind != wanted.kind ||
                        header.operation != wanted.operation ||
                        header.payload_bytes != wanted.payload_bytes ||
                        header.session != wanted.session || header.sequence != wanted.sequence ||
                        header.incarnation != wanted.incarnation ||
                        (peer.channel->input_descriptor.get() >= 0) != expected->has_descriptor)
                        return std::nullopt;
                }
                const auto first_read_ns = peer.channel->first_read_ns;
                auto descriptor = std::move(peer.channel->input_descriptor);
                auto message = peer.channel->take();
                if (message.header.kind != ExecutionMessageKind::kRequest) {
                    close_peer(peer.id);
                    return std::nullopt;
                }
                if (std::ranges::any_of(peer.pending, [&](const auto& header) {
                        return header.sequence == message.header.sequence;
                    })) {
                    close_peer(peer.id);
                    return std::nullopt;
                }
                peer.pending.push_back(message.header);
                return ExecutionReceivedRequest{peer.id,
                                                peer.channel->identity,
                                                std::move(message),
                                                std::move(descriptor),
                                                std::exchange(peer.frame_received_ns, 0),
                                                first_read_ns};
            }
    } catch (...) {
        // Allocation failure does not publish a partial request.
        return std::nullopt;
    }
    return std::nullopt;
}
bool ExecutionSocketServer::reply(std::uint64_t connection, const ExecutionWireMessage& message,
                                  std::uint64_t* queued_ns) noexcept {
    if (queued_ns) *queued_ns = 0;
    auto& state = *implementation_;
    if (!state.owns() || message.header.kind != ExecutionMessageKind::kReply) return false;
    try {
        for (auto& peer : state.peers)
            if (peer.id == connection) {
                const auto pending = std::ranges::find_if(peer.pending, [&](const auto& header) {
                    return header.sequence == message.header.sequence &&
                           header.operation == message.header.operation;
                });
                if (pending == peer.pending.end()) return false;
                auto encoded = encode_execution_message(message, state.config.max_payload_bytes);
                if (!encoded) return false;
                const bool can_send = peer.channel->output.empty();
                if (can_send) peer.channel->progress = Clock::now();
                peer.channel->output.push_back({std::move(*encoded), ExecutionOwnedDescriptor{}});
                peer.pending.erase(pending);
                if (queued_ns) {
                    const auto observed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                 Clock::now().time_since_epoch())
                                                 .count();
                    if (observed_ns > 0) *queued_ns = static_cast<std::uint64_t>(observed_ns);
                }
                if (can_send && !peer.channel->send_pending()) break;
                return true;
            }
        // No peer references survive removal. The request was already accepted;
        // losing its result delivery is not permission to replay native work.
        close_peer(connection);
    } catch (...) {
        // Allocation failure does not report successful delivery.
        return false;
    }
    return false;
}
bool ExecutionSocketServer::has_peer(std::uint64_t connection) const noexcept {
    const auto& state = *implementation_;
    return state.owns() && std::ranges::any_of(state.peers, [connection](const auto& peer) {
               return peer.id == connection;
           });
}
bool ExecutionSocketServer::response_pending(std::uint64_t connection) const noexcept {
    const auto& state = *implementation_;
    if (!state.owns()) return false;
    for (const auto& peer : state.peers)
        if (peer.id == connection) return !peer.channel->output.empty();
    return false;
}
void ExecutionSocketServer::close_peer(std::uint64_t connection) noexcept {
    auto& state = *implementation_;
    if (state.owns())
        std::erase_if(state.peers,
                      [connection](const auto& peer) { return peer.id == connection; });
}
void ExecutionSocketServer::stop() noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return;
    state.peers.clear();
    close_fd(state.listener);
    state.remove_path();
}

struct ExecutionSocketClient::Implementation {
    struct Pending {
        ExecutionOpcode operation;
        ExecutionOpcode effective_operation;
        std::uint64_t sequence{};
        Clock::time_point deadline;
        std::optional<ExecutionWireReply> reply;
        bool is_metadata_bootstrap{};
    };
    const ExecutionSocketClientConfig config;
    const pid_t process{::getpid()};
    Channel channel;
    std::uint64_t session{}, next_sequence{1};
    std::array<std::uint64_t, 2> incarnation{};
    std::vector<Pending> pending;
    std::optional<ExecutionSocketExpiry> expiry;
    bool has_attempted{}, is_poisoned{}, is_closing{}, has_sequence_space{true};
    explicit Implementation(ExecutionSocketClientConfig value) : config(std::move(value)) {
        if (!config.expected_server.pid || !config.expected_server.start_time_ticks ||
            !config.timeout_ms || !config.max_payload_bytes || !config.max_pending_requests ||
            (config.preparation_timeout_ms &&
             (!*config.preparation_timeout_ms ||
              *config.preparation_timeout_ms > kExecutionMaxPreparationTimeoutMs)) ||
            config.max_pending_requests > 64 ||
            config.max_payload_bytes >
                std::numeric_limits<std::uint32_t>::max() - kExecutionSourceContinuationBytes ||
            config.max_payload_bytes > std::numeric_limits<std::size_t>::max() -
                                           kExecutionHeaderBytes -
                                           kExecutionSourceContinuationBytes)
            throw std::invalid_argument(
                "execution client requires trusted server identity and bounds");
        pending.reserve(config.max_pending_requests);
    }
    void poison() noexcept {
        is_poisoned = true;
        close_fd(channel.descriptor);
        channel.output.clear();
        channel.input_descriptor = ExecutionOwnedDescriptor{};
    }
    bool observe_expiry(Clock::time_point now) noexcept {
        const Pending* earliest{};
        for (const auto& request : pending)
            if (!request.reply && now >= request.deadline &&
                (!earliest || request.deadline < earliest->deadline))
                earliest = &request;
        if (!earliest) return false;
        if (!expiry) {
            const auto timeout_ms =
                config.preparation_timeout_ms &&
                        execution_uses_preparation_deadline(earliest->effective_operation)
                    ? *config.preparation_timeout_ms
                    : config.timeout_ms;
            expiry = ExecutionSocketExpiry{earliest->operation, earliest->effective_operation,
                                           earliest->sequence, session, timeout_ms};
        }
        return true;
    }
    ExecutionSocketSubmission submit(ExecutionOpcode operation, std::span<const std::byte> payload,
                                     int borrowed_descriptor = -1,
                                     std::uint64_t* queued_ns = nullptr) noexcept {
        if (queued_ns) *queued_ns = 0;
        const bool is_host_function_observation =
            operation == ExecutionOpcode::kQueryHostFunction ||
            operation == ExecutionOpcode::kFinishHostFunction;
        if (::getpid() != process || is_poisoned || (is_closing && !is_host_function_observation) ||
            !has_sequence_space || channel.descriptor < 0)
            return {};
        const auto request =
            normalize_execution_request(operation, payload, config.max_payload_bytes);
        if (!request || borrowed_descriptor < -1 ||
            (execution_requires_descriptor(request->operation) != (borrowed_descriptor >= 0)))
            return {ExecutionSocketSubmitStatus::kInvalid, 0};
        if (pending.size() == config.max_pending_requests ||
            (request->operation == ExecutionOpcode::kClose && !pending.empty()))
            return {ExecutionSocketSubmitStatus::kFull, 0};
        try {
            ExecutionWireMessage frame{
                {ExecutionMessageKind::kRequest, operation, 0, session, next_sequence, incarnation},
                {}};
            if (!payload.empty()) frame.payload.assign(payload.begin(), payload.end());
            auto encoded = encode_execution_message(frame, config.max_payload_bytes);
            if (!encoded) return {ExecutionSocketSubmitStatus::kInvalid, 0};
            ExecutionOwnedDescriptor descriptor;
            if (borrowed_descriptor >= 0) {
                descriptor =
                    ExecutionOwnedDescriptor{::fcntl(borrowed_descriptor, F_DUPFD_CLOEXEC, 0)};
                if (descriptor.get() < 0) return {ExecutionSocketSubmitStatus::kInvalid, 0};
            }
            const auto effective_timeout_ms =
                config.preparation_timeout_ms &&
                        execution_uses_preparation_deadline(request->operation)
                    ? *config.preparation_timeout_ms
                    : config.timeout_ms;
            ExecutionPayloadReader bootstrap(request->payload);
            std::uint32_t role{}, flags{};
            const bool is_metadata_bootstrap =
                request->operation == ExecutionOpcode::kHello && bootstrap.read_u32(role) &&
                (role == kExecutionLtMetadataRole || role == kExecutionDnnMetadataRole ||
                 role == kExecutionPointerMetadataRole || role == kExecutionProcessMemoryRole) &&
                bootstrap.read_u32(flags) && !flags && bootstrap.empty();
            const auto now = Clock::now();
            pending.push_back({operation, request->operation, next_sequence,
                               now + std::chrono::milliseconds{effective_timeout_ms}, std::nullopt,
                               is_metadata_bootstrap});
            try {
                channel.output.push_back({std::move(*encoded), std::move(descriptor)});
            } catch (...) {
                pending.pop_back();
                return {};
            }
            if (channel.output.size() == 1) channel.progress = now;
            const auto sequence = next_sequence;
            if (next_sequence == std::numeric_limits<std::uint64_t>::max())
                has_sequence_space = false;
            else
                ++next_sequence;
            if (request->operation == ExecutionOpcode::kClose) is_closing = true;
            if (queued_ns) {
                const auto observed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          Clock::now().time_since_epoch())
                                          .count();
                if (observed > 0) *queued_ns = static_cast<std::uint64_t>(observed);
            }
            // Admission is committed before delivery. Preserve its exact
            // sequence on failure; later observation reports uncertainty rather
            // than manufacturing a rejection or permitting native replay.
            if (channel.output.size() == 1 && !channel.send_pending()) poison();
            return {ExecutionSocketSubmitStatus::kAccepted, sequence};
        } catch (...) {
            return {};
        }
    }
    bool receive_reply() {
        if (!channel.is_ready) return true;
        const auto message = channel.take();
        const auto found = std::ranges::find_if(
            pending, [&](const auto& value) { return value.sequence == message.header.sequence; });
        if (found == pending.end() || found->reply) return false;
        const auto now = Clock::now();
        if (now >= found->deadline) {
            (void)observe_expiry(now);
            return false;
        }
        if (message.header.kind != ExecutionMessageKind::kReply ||
            message.header.operation != found->operation || !message.header.session ||
            message.header.incarnation == std::array<std::uint64_t, 2>{} ||
            (session &&
             (message.header.session != session || message.header.incarnation != incarnation)))
            return false;
        auto reply = decode_execution_reply(message.payload);
        if (!reply) return false;
        const bool is_bootstrap = found->effective_operation == ExecutionOpcode::kHello;
        if (is_bootstrap && (!reply->data.empty() || found->is_metadata_bootstrap)) {
            const auto ownership = decode_execution_bootstrap_ownership(reply->data);
            const auto expected = found->is_metadata_bootstrap
                                      ? ExecutionBootstrapOwnership::kMetadataRetained
                                      : ExecutionBootstrapOwnership::kRetained;
            if (!ownership ||
                (reply->status == 0 && reply->disposition == ExecutionWireDisposition::kComplete &&
                 *ownership != expected))
                return false;
        }
        session = message.header.session;
        incarnation = message.header.incarnation;
        const bool is_failed_bootstrap =
            is_bootstrap &&
            (reply->disposition != ExecutionWireDisposition::kComplete || reply->status != 0);
        found->reply = std::move(*reply);
        if (found->effective_operation == ExecutionOpcode::kClose || is_failed_bootstrap) poison();
        return true;
    }
    bool poll_for(std::uint32_t timeout_ms) noexcept {
        if (::getpid() != process || is_poisoned || channel.descriptor < 0) return false;
        try {
            const auto now = Clock::now();
            auto deadline = now + std::chrono::milliseconds{timeout_ms};
            for (const auto& value : pending)
                if (!value.reply) deadline = std::min(deadline, value.deadline);
            if (observe_expiry(now) || !channel.is_alive()) {
                poison();
                return false;
            }
            pollfd descriptor{channel.descriptor,
                              static_cast<short>(POLLIN | (!channel.output.empty() ? POLLOUT : 0)),
                              0};
            int status{};
            do {
                const auto remaining =
                    std::chrono::ceil<std::chrono::milliseconds>(deadline - Clock::now()).count();
                status = ::poll(&descriptor, 1,
                                static_cast<int>(std::clamp<std::int64_t>(
                                    remaining, 0, std::numeric_limits<int>::max())));
            } while (status < 0 && errno == EINTR && Clock::now() < deadline);
            if (status < 0 || (descriptor.revents & (POLLERR | POLLNVAL)) ||
                ((descriptor.revents & POLLOUT) && !channel.send_pending()) ||
                ((descriptor.revents & POLLIN) && !channel.receive(config.max_payload_bytes)) ||
                !receive_reply()) {
                poison();
                return false;
            }
            if ((descriptor.revents & POLLHUP) || observe_expiry(Clock::now())) {
                poison();
                return false;
            }
            return true;
        } catch (...) {
            poison();
            return false;
        }
    }
    std::optional<ExecutionWireReply> take_reply(std::uint64_t sequence) noexcept {
        if (::getpid() != process) return std::nullopt;
        const auto found = std::ranges::find_if(
            pending, [&](const auto& value) { return value.sequence == sequence; });
        if (found == pending.end() || !found->reply) return std::nullopt;
        auto reply = std::move(found->reply);
        pending.erase(found);
        return reply;
    }
    std::optional<ExecutionWireReply> exchange(ExecutionOpcode operation,
                                               std::span<const std::byte> payload,
                                               int borrowed_descriptor = -1) noexcept {
        if (::getpid() != process || !pending.empty()) return std::nullopt;
        const auto accepted = submit(operation, payload, borrowed_descriptor);
        if (accepted.status != ExecutionSocketSubmitStatus::kAccepted) return std::nullopt;
        for (;;) {
            if (auto reply = take_reply(accepted.sequence)) return reply;
            if (!poll_for(config.timeout_ms)) return take_reply(accepted.sequence);
        }
    }
};
ExecutionSocketClient::ExecutionSocketClient(ExecutionSocketClientConfig config)
    : implementation_(std::make_unique<Implementation>(std::move(config))) {}
ExecutionSocketClient::~ExecutionSocketClient() {
    if (implementation_->process != ::getpid()) {
        [[maybe_unused]] const auto* child_inherited_state = implementation_.release();
    }
}
std::optional<ExecutionWireReply> ExecutionSocketClient::open(
    std::span<const std::byte> bootstrap) noexcept {
    auto& state = *implementation_;
    const auto submitted = submit_open(bootstrap);
    if (submitted.status != ExecutionSocketSubmitStatus::kAccepted) return std::nullopt;
    for (;;) {
        if (auto reply = state.take_reply(submitted.sequence)) return reply;
        if (!state.poll_for(state.config.timeout_ms)) return state.take_reply(submitted.sequence);
    }
}
ExecutionSocketSubmission ExecutionSocketClient::submit_open(
    std::span<const std::byte> bootstrap,
    const std::optional<ExecutionSourceContinuation>& continuation) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || state.has_attempted) return {};
    state.has_attempted = true;
    std::optional<std::vector<std::byte>> wrapped;
    if (continuation) {
        wrapped = encode_execution_source_continuation(*continuation, ExecutionOpcode::kHello,
                                                       bootstrap, state.config.max_payload_bytes);
        if (!wrapped) return {ExecutionSocketSubmitStatus::kInvalid, 0};
    }
    sockaddr_un address{};
    if (!make_address(state.config.socket_path, address)) return {};
    struct stat path {};
    if (::lstat(state.config.socket_path.c_str(), &path) != 0 || !S_ISSOCK(path.st_mode) ||
        path.st_uid != state.config.expected_server.uid || (path.st_mode & 0077))
        return {};
    state.channel.descriptor = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (state.channel.descriptor < 0 || !pass_credentials(state.channel.descriptor)) {
        state.poison();
        return {};
    }
    const auto connected = ::connect(state.channel.descriptor,
                                     reinterpret_cast<const sockaddr*>(&address), sizeof(address));
    if (connected != 0) {
        state.poison();
        return {};
    }
    if (!state.channel.authenticate(state.config.expected_server.uid,
                                    &state.config.expected_server)) {
        state.poison();
        return {};
    }
    return state.submit(wrapped ? ExecutionOpcode::kSourceContinuation : ExecutionOpcode::kHello,
                        wrapped ? std::span<const std::byte>{*wrapped} : bootstrap);
}
std::optional<ExecutionWireReply> ExecutionSocketClient::exchange(
    ExecutionOpcode operation, std::span<const std::byte> payload,
    int borrowed_descriptor) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process || !state.session) return std::nullopt;
    const auto request =
        normalize_execution_request(operation, payload, state.config.max_payload_bytes);
    if (!request || request->operation == ExecutionOpcode::kHello) return std::nullopt;
    return state.exchange(operation, payload, borrowed_descriptor);
}
std::uint64_t ExecutionSocketClient::session() const noexcept {
    return implementation_->process == ::getpid() ? implementation_->session : 0;
}
std::optional<ExecutionSocketExpiry> ExecutionSocketClient::expired_request() const noexcept {
    const auto& state = *implementation_;
    return state.process == ::getpid() && state.is_poisoned ? state.expiry : std::nullopt;
}
ExecutionSocketSubmission ExecutionSocketClient::submit(ExecutionOpcode operation,
                                                        std::span<const std::byte> payload,
                                                        int borrowed_descriptor,
                                                        std::uint64_t* queued_ns) noexcept {
    if (queued_ns) *queued_ns = 0;
    auto& state = *implementation_;
    if (::getpid() != state.process || !state.session) return {};
    const auto request =
        normalize_execution_request(operation, payload, state.config.max_payload_bytes);
    if (!request) return {ExecutionSocketSubmitStatus::kInvalid, 0};
    if (request->operation == ExecutionOpcode::kHello) return {};
    return state.submit(operation, payload, borrowed_descriptor, queued_ns);
}
ExecutionSocketSubmission ExecutionSocketClient::submit_observed(ExecutionOpcode operation,
                                                                 std::span<const std::byte> payload,
                                                                 int borrowed_descriptor,
                                                                 std::uint64_t* queued_ns,
                                                                 bool* fully_flushed) noexcept {
    if (fully_flushed) *fully_flushed = false;
    const auto result = submit(operation, payload, borrowed_descriptor, queued_ns);
    // The accepted frame is the FIFO tail and no other producer may interleave
    // under external serialization. Poisoning may discard output: never treat
    // that empty queue as successful delivery.
    if (fully_flushed && result.status == ExecutionSocketSubmitStatus::kAccepted &&
        !implementation_->is_poisoned && implementation_->channel.output.empty())
        *fully_flushed = true;
    return result;
}
bool ExecutionSocketClient::poll_for(std::uint32_t timeout_ms) noexcept {
    return implementation_->poll_for(timeout_ms);
}
std::optional<ExecutionSocketReadiness> ExecutionSocketClient::readiness() const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process || state.is_poisoned || state.channel.descriptor < 0)
        return std::nullopt;
    ExecutionOwnedDescriptor duplicate{::fcntl(state.channel.descriptor, F_DUPFD_CLOEXEC, 0)};
    if (duplicate.get() < 0) return std::nullopt;
    return ExecutionSocketReadiness{
        std::move(duplicate),
        static_cast<short>(POLLIN | (!state.channel.output.empty() ? POLLOUT : 0))};
}
std::optional<ExecutionWireReply> ExecutionSocketClient::take_reply(
    std::uint64_t sequence) noexcept {
    return implementation_->take_reply(sequence);
}
}  // namespace glimmer::control
