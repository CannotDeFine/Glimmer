#pragma once

#include "glimmer/control/execution_protocol.h"
#include "glimmer/control/task_admission.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace glimmer::control {

// Owns one process-local descriptor reference, not its backing contents or CUDA
// lifetime. Moving transfers ownership; destruction closes once without retry.
class ExecutionOwnedDescriptor final {
   public:
    explicit ExecutionOwnedDescriptor(int descriptor = -1) noexcept;
    ~ExecutionOwnedDescriptor();
    ExecutionOwnedDescriptor(const ExecutionOwnedDescriptor&) = delete;
    ExecutionOwnedDescriptor& operator=(const ExecutionOwnedDescriptor&) = delete;
    ExecutionOwnedDescriptor(ExecutionOwnedDescriptor&& other) noexcept;
    ExecutionOwnedDescriptor& operator=(ExecutionOwnedDescriptor&& other) noexcept;
    [[nodiscard]] int get() const noexcept;
    [[nodiscard]] int release() noexcept;

   private:
    int descriptor_;
};

// Process-local readiness hint, not a transport/SDK/GPU result. Create under the
// client's external serialization, then wait outside it. Owns a CLOEXEC duplicate
// so concurrent poisoning/descriptor reuse cannot retarget the snapshot. Never
// consumes bytes or accesses client state; other consumers may drain readiness.
class ExecutionSocketReadiness final {
   public:
    ExecutionSocketReadiness(const ExecutionSocketReadiness&) = delete;
    ExecutionSocketReadiness& operator=(const ExecutionSocketReadiness&) = delete;
    ExecutionSocketReadiness(ExecutionSocketReadiness&&) noexcept = default;
    ExecutionSocketReadiness& operator=(ExecutionSocketReadiness&&) noexcept = default;
    // One syscall, no EINTR retry. True includes timeout, never receipt success.
    // Zero or more than50us rejects. Failure permits the caller's normal sleep.
    [[nodiscard]] bool wait_for_ns(std::uint64_t timeout_ns) const noexcept;

   private:
    friend class ExecutionSocketClient;
    ExecutionSocketReadiness(ExecutionOwnedDescriptor descriptor, short events) noexcept;
    std::uint32_t process_;
    ExecutionOwnedDescriptor descriptor_;
    short events_;
};

// Process-local CPU hint only: many producer threads, one externally serialized
// consumer. Shared ownership keeps the descriptor alive through worker teardown.
// Readiness confers no request, scheduling, SDK or completion authority.
class ExecutionSocketWake final {
   public:
    ExecutionSocketWake();
    ExecutionSocketWake(const ExecutionSocketWake&) = delete;
    ExecutionSocketWake& operator=(const ExecutionSocketWake&) = delete;
    [[nodiscard]] int descriptor() const noexcept;
    [[nodiscard]] bool notify() const noexcept;
    [[nodiscard]] bool consume() const noexcept;

   private:
    const std::uint32_t process;
    ExecutionOwnedDescriptor descriptor_;
};

struct ExecutionSocketConfig {
    std::string socket_path;
    std::size_t max_peers{8};
    std::size_t max_payload_bytes{kExecutionMaxPayloadBytes};
    std::uint32_t io_timeout_ms{30000};
    std::optional<std::uint32_t> allowed_uid;
    // Includes requests awaiting a reply and replies awaiting socket delivery.
    std::size_t max_pending_requests{1};
    // Trusted diagnostic policy only. Record the first positive read observation
    // and complete validated frame time; no wire claim or execution authority.
    bool should_record_receive_time{};
    std::shared_ptr<ExecutionSocketWake> wake{};
};
struct ExecutionReceivedRequest {
    std::uint64_t connection{};
    TaskPeerIdentity peer;
    ExecutionWireMessage message;
    ExecutionOwnedDescriptor descriptor;
    std::uint64_t frame_received_ns{};
    std::uint64_t first_read_ns{};
};
// Borrowed complete frame, not an admission or execution capability. Payload
// remains valid only until the next non-const server call. No descriptor ownership
// is exposed; a failed exact take leaves the original channel frame untouched.
struct ExecutionSocketRequestView {
    std::uint64_t connection{};
    TaskPeerIdentity peer;
    ExecutionWireHeader header;
    std::span<const std::byte> payload;
    bool has_descriptor{};
};

// Trusted, constructing-process/thread-owned wall-time diagnostics. Recording
// never grants execution or changes I/O. Storage is preallocated and bounded;
// exhausted or incomplete evidence is invalid, never silently truncated.
// Read views require external serialization and no active poll call. Frame
// identities are separate from syscall rows because headers may span cycles.
class ExecutionSocketPollJournal final {
   public:
    struct Limits {
        std::size_t cycles{}, peers{}, polls{}, reads{}, frames{};
    };
    struct Cycle {
        std::uint64_t id{}, entry{}, exit{}, timeout{}, peer_begin{}, peer_count{}, poll_begin{},
            poll_count{}, success{};
    };
    struct Poll {
        std::uint64_t cycle{}, entry{}, returned{};
        std::int64_t result{};
        std::uint64_t error{};
    };
    struct Peer {
        std::uint64_t cycle{}, connection{}, pid{}, start{}, events{}, revents{}, pending{},
            output{}, flags{}, visit{}, alive{};
    };
    struct Read {
        std::uint64_t cycle{}, connection{}, token{}, entry{}, returned{}, first_read{};
        std::int64_t result{};
        std::uint64_t error{};
    };
    struct Frame {
        std::uint64_t cycle{}, connection{}, token{}, wire{}, opcode{}, first_read{}, received{},
            pid{}, start{};
    };
    static_assert(sizeof(Cycle) == 72 && sizeof(Poll) == 40 && sizeof(Peer) == 88 &&
                  sizeof(Read) == 64 && sizeof(Frame) == 72);
    // Preserve the public enum's existing underlying type.
    // NOLINTNEXTLINE(performance-enum-size)
    enum class Failure { kNone, kOwner, kClock, kCapacity, kOrder, kCorrelation };
    using Clock = bool (*)(std::uint64_t&) noexcept;
    explicit ExecutionSocketPollJournal(Limits limits, Clock clock = system_clock);
    ~ExecutionSocketPollJournal();
    ExecutionSocketPollJournal(const ExecutionSocketPollJournal&) = delete;
    ExecutionSocketPollJournal& operator=(const ExecutionSocketPollJournal&) = delete;
    [[nodiscard]] static bool system_clock(std::uint64_t& now) noexcept;
    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] Failure failure() const noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;
    [[nodiscard]] std::span<const Cycle> cycles() const noexcept;
    [[nodiscard]] std::span<const Poll> polls() const noexcept;
    [[nodiscard]] std::span<const Peer> peers() const noexcept;
    [[nodiscard]] std::span<const Read> reads() const noexcept;
    [[nodiscard]] std::span<const Frame> frames() const noexcept;
    // Cold-path, owner-only publication after recording has ended. Creates one
    // new 0600 file without overwrite; checked little-endian schema, at most
    // 512 MiB. A failed partial file is retained and never valid evidence.
    // The caller must separately prove normal application shutdown.
    [[nodiscard]] bool publish(const std::string& path) const noexcept;

   private:
    friend struct ExecutionSocketPollRecorder;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

// CUDA-free, nonblocking transport. Externally serialized in one process.
// Only the constructing process owns the endpoint. After fork, the child must
// not use it and must promptly exec or _exit; its inherited state is not freed
// by the child destructor to avoid touching the parent's endpoint.
// No callbacks, SDK work, priority selection, reconnection or resource release.
// Every data chunk authenticates the actual sender, including inherited sockets.
// One poll consumes at most eight nonblocking receives of at most 16 KiB each
// per readable channel, including EINTR attempts, and never crosses a frame.
// Each send invocation is bounded to eight attempts including EINTR and stops
// after one positive write or would-block; partial frames retain FIFO ownership.
class ExecutionSocketServer final {
   public:
    explicit ExecutionSocketServer(ExecutionSocketConfig config);
    // Distinct diagnostic ABI: the existing config layout and constructor stay
    // unchanged. Requires receive timestamps and a fresh owner-thread journal.
    ExecutionSocketServer(ExecutionSocketConfig config,
                          std::shared_ptr<ExecutionSocketPollJournal> journal);
    ~ExecutionSocketServer();
    ExecutionSocketServer(const ExecutionSocketServer&) = delete;
    ExecutionSocketServer& operator=(const ExecutionSocketServer&) = delete;
    [[nodiscard]] bool start() noexcept;
    [[nodiscard]] bool poll_for(std::uint32_t timeout_ms) noexcept;
    // Same authenticated receive/cleanup contract, with submillisecond waits.
    [[nodiscard]] bool poll_for_ns(std::uint64_t timeout_ns) noexcept;
    [[nodiscard]] std::optional<ExecutionReceivedRequest> take_request() noexcept;
    // Owner-process/external-serialization contract is identical to take_request.
    // Peek performs no I/O, clock query, allocation or ownership transfer.
    [[nodiscard]] std::optional<ExecutionSocketRequestView> peek_request(
        std::uint64_t connection) const noexcept;
    // Compare the complete header, peer and descriptor presence, never a stale
    // payload pointer. Mismatches do not consume or close the current frame.
    [[nodiscard]] std::optional<ExecutionReceivedRequest> take_request(
        const ExecutionSocketRequestView& expected) noexcept;
    // Empty-to-nonempty output attempts one bounded nonblocking send immediately.
    // Backlog stays queued. A hard send error closes the peer and returns false;
    // failure never proves that the accepted request was not executed.
    // Optional trusted diagnostics capture enqueue time before sending, or zero
    // if no queue observation exists. The timestamp grants no completion proof.
    [[nodiscard]] bool reply(std::uint64_t connection, const ExecutionWireMessage& message,
                             std::uint64_t* queued_ns = nullptr) noexcept;
    [[nodiscard]] bool has_peer(std::uint64_t connection) const noexcept;
    [[nodiscard]] bool response_pending(std::uint64_t connection) const noexcept;
    void close_peer(std::uint64_t connection) noexcept;
    void stop() noexcept;

   private:
    [[nodiscard]] std::optional<ExecutionReceivedRequest> take_request_impl(
        const ExecutionSocketRequestView* expected) noexcept;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

inline constexpr std::uint32_t kExecutionMaxPreparationTimeoutMs = 600000;
// Trusted local policy only: no payload or peer can select a longer deadline.
[[nodiscard]] bool execution_uses_preparation_deadline(ExecutionOpcode operation) noexcept;

struct ExecutionSocketClientConfig {
    std::string socket_path;
    TaskPeerIdentity expected_server;
    std::uint32_t timeout_ms{30000};
    std::size_t max_payload_bytes{kExecutionMaxPayloadBytes};
    // Includes verified replies until their owner explicitly takes them.
    std::size_t max_pending_requests{8};
    // Unset preserves timeout_ms. A positive bounded override applies only to
    // cold module/function preparation, once when the request is submitted.
    std::optional<std::uint32_t> preparation_timeout_ms{};
};
// Preserve the public enum's existing underlying type.
// NOLINTNEXTLINE(performance-enum-size)
enum class ExecutionSocketSubmitStatus { kAccepted, kFull, kUnavailable, kInvalid };
struct ExecutionSocketSubmission {
    ExecutionSocketSubmitStatus status{ExecutionSocketSubmitStatus::kUnavailable};
    std::uint64_t sequence{};
};
// CPU diagnostic only. A transport timeout is not an SDK result, cancellation
// receipt, physical-stop proof or permission to replay accepted work.
struct ExecutionSocketExpiry {
    ExecutionOpcode operation{}, effective_operation{};
    std::uint64_t sequence{}, session{};
    std::uint32_t timeout_ms{};
    bool operator==(const ExecutionSocketExpiry&) const = default;
};
// Externally serialized in one process, including polling and receipt access.
// Bootstrap once; submit copies inputs but does not imply SDK or GPU success.
// Poll advances bounded I/O; replies are matched by sequence, not arrival order.
// Verified replies survive later poisoning; unresolved receipts are uncertain.
// Blocking exchange is exclusive and rejects while any receipt is outstanding.
// Close requires no retained receipts and seals new work. Until its reply, only
// direct host-function Query/Finish submissions may drain accepted callbacks;
// all normal validation and pending-window limits still apply. Close poisons.
// An uncertain exchange permanently poisons this instance; never reconnects.
// A fork child rejects before touching the parent's connection or mutexes.
class ExecutionSocketClient final {
   public:
    explicit ExecutionSocketClient(ExecutionSocketClientConfig config);
    ~ExecutionSocketClient();
    ExecutionSocketClient(const ExecutionSocketClient&) = delete;
    ExecutionSocketClient& operator=(const ExecutionSocketClient&) = delete;
    [[nodiscard]] std::optional<ExecutionWireReply> open(
        std::span<const std::byte> bootstrap = {}) noexcept;
    // Connect/authenticate and enqueue one bootstrap without waiting for its
    // reply. The ordinary receipt path validates the same ownership as open().
    [[nodiscard]] ExecutionSocketSubmission submit_open(
        std::span<const std::byte> bootstrap = {},
        const std::optional<ExecutionSourceContinuation>& continuation = std::nullopt) noexcept;
    [[nodiscard]] std::optional<ExecutionWireReply> exchange(
        ExecutionOpcode operation, std::span<const std::byte> payload = {},
        int borrowed_descriptor = -1) noexcept;
    // Empty-to-nonempty admission attempts one bounded nonblocking send. A hard
    // delivery failure keeps its consumed accepted sequence and poisons transport;
    // poll_for reports false, never an invented SDK result or replay permission.
    // Optional trusted CPU timestamp precedes eager delivery; zero if not admitted.
    [[nodiscard]] ExecutionSocketSubmission submit(ExecutionOpcode operation,
                                                   std::span<const std::byte> payload = {},
                                                   int borrowed_descriptor = -1,
                                                   std::uint64_t* queued_ns = nullptr) noexcept;
    // Same admission path and ABI-stable result. Under the caller's existing
    // serialization, true observes that the accepted tail was fully sent by
    // return. False includes unaccepted, partial, queued and uncertain delivery;
    // it is never a delivery time, SDK result or permission to replay work.
    // The ordinary submit path performs no additional diagnostic work.
    [[nodiscard]] ExecutionSocketSubmission submit_observed(ExecutionOpcode operation,
                                                            std::span<const std::byte> payload,
                                                            int borrowed_descriptor,
                                                            std::uint64_t* queued_ns,
                                                            bool* fully_flushed) noexcept;
    // False permanently closes transport; it does not cancel GPU ownership.
    [[nodiscard]] bool poll_for(std::uint32_t timeout_ms) noexcept;
    [[nodiscard]] std::optional<ExecutionWireReply> take_reply(std::uint64_t sequence) noexcept;
    [[nodiscard]] std::uint64_t session() const noexcept;
    // First observed expired unresolved request, retained after poisoning.
    // Non-timeout failure and fork-child access return null. Never consumes a
    // receipt, queries a clock, revives transport or changes a deadline.
    [[nodiscard]] std::optional<ExecutionSocketExpiry> expired_request() const noexcept;
    // No connection/bootstrap or transport mutation. Null on unavailable/dup
    // failure; the ordinary authenticated poll/receipt path remains authoritative.
    [[nodiscard]] std::optional<ExecutionSocketReadiness> readiness() const noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::control
