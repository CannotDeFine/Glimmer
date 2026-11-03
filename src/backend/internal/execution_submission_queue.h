#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <optional>
#include <span>

namespace glimmer::backend {

class ExecutionReceipt final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t sequence() const noexcept;

   private:
    friend class ExecutionSubmissionQueue;
    std::weak_ptr<const void> owner_;
    std::uint64_t sequence_{};
};

class ExecutionIssueToken final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] std::uint64_t sequence() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;

   private:
    friend class ExecutionSubmissionQueue;
    std::weak_ptr<const void> owner_;
    std::uint64_t sequence_{};
    std::uint64_t generation_{};
};

struct ExecutionSessionIdentity {
    std::uint64_t client_id{};
    std::uint64_t server_incarnation{};
    std::uint64_t first_sequence{1};
};

struct ExecutionSubmissionLimits {
    std::size_t max_commands{64};
    std::size_t max_parameter_bytes{std::size_t{4} * 1024 * 1024};
    std::size_t max_retained_host_bytes{std::size_t{64} * 1024 * 1024};
};

struct ExecutionRequest {
    std::uint32_t operation{};
    std::uint64_t context{};
    std::uint64_t stream{};
    std::span<const std::byte> parameters;
    // Trusted backend lease, not an application pointer or a wire field. Its
    // noexcept destruction must defer required GPU cleanup to its context owner.
    std::shared_ptr<const void> lifetime;
    // Trusted deferred preparation reserves its largest possible packed ABI
    // before acceptance. Zero preserves the ordinary exact-size allocation.
    std::size_t parameter_capacity_bytes{};
};

enum class ExecutionAcceptStatus : std::uint8_t {
    kAccepted,
    kInvalid,
    kWrongSequence,
    kWrongProcess,
    kFull,  // Recoverable occupied queue/parameter capacity; no acceptance.
    kClosed,
    kQuarantined,
    kOutOfMemory,  // Allocation failure or a reservation larger than the whole budget.
};

// Real backing storage for variable command data, not packed-parameter padding.
// Keep this owner alive until every arena-backed object has been destroyed.
// Only the executor uses the unsynchronized resource; final destruction may run
// on any thread. Null upstream forbids uncharged allocations on arena exhaustion.
class ExecutionHostStorage final {
   public:
    ~ExecutionHostStorage();
    ExecutionHostStorage(const ExecutionHostStorage&) = delete;
    ExecutionHostStorage& operator=(const ExecutionHostStorage&) = delete;
    [[nodiscard]] std::pmr::memory_resource* resource() noexcept;
    [[nodiscard]] std::size_t bytes() const noexcept;

   private:
    friend class ExecutionSubmissionQueue;
    struct Counter;
    struct Implementation;
    ExecutionHostStorage(std::shared_ptr<Counter> counter, std::size_t bytes);
    std::unique_ptr<Implementation> implementation_;
};
struct ExecutionHostAllocation {
    ExecutionAcceptStatus status{ExecutionAcceptStatus::kInvalid};
    std::shared_ptr<ExecutionHostStorage> storage;
};

struct ExecutionAcceptance {
    ExecutionAcceptStatus status{ExecutionAcceptStatus::kInvalid};
    ExecutionReceipt receipt;
    [[nodiscard]] bool accepted() const noexcept;
};

enum class ExecutionReplyOrigin : std::uint8_t {
    kNativeSdk,
    kRejectedBeforeIssue,
    kUncertainIssue,
};

struct ExecutionHostReply {
    std::int32_t status{};
    ExecutionReplyOrigin origin{ExecutionReplyOrigin::kNativeSdk};
};

struct ExecutionIssue {
    ExecutionReceipt receipt;
    ExecutionIssueToken token;
    std::uint32_t operation{};
    std::uint64_t context{};
    std::uint64_t stream{};
    // Borrowed only until final physical retirement; never transport this span.
    std::span<const std::byte> parameters;
};

struct ExecutionSubmissionSnapshot {
    std::size_t outstanding{};
    std::size_t queued{};
    std::size_t issuing{};
    std::size_t awaiting_physical_stop{};
    std::size_t paused{};
    std::size_t unacknowledged_replies{};
    std::size_t retained_parameter_bytes{};
    std::size_t retained_host_bytes{};
    bool has_executor{};
    bool is_closing{};
    bool is_quarantined{};
};

// Private server-backend primitive. All methods are thread-safe; the executor
// transitions additionally require the same nonrecycled thread lifetime bound
// by bind_executor(). The trusted adapter validates ABI/session/resource data
// before accept() and independently proves real physical stop before retirement.
// This class never performs CUDA, transport, policy or original application work.
// Keep the queue alive through every call and externally serialize destruction.
// An inherited post-fork queue rejects before touching any inherited mutex.
class ExecutionSubmissionQueue final {
   public:
    explicit ExecutionSubmissionQueue(ExecutionSessionIdentity identity,
                                      ExecutionSubmissionLimits limits = {});
    ~ExecutionSubmissionQueue();
    ExecutionSubmissionQueue(const ExecutionSubmissionQueue&) = delete;
    ExecutionSubmissionQueue& operator=(const ExecutionSubmissionQueue&) = delete;

    [[nodiscard]] bool bind_executor() noexcept;
    [[nodiscard]] bool owns_executor() const noexcept;
    // Owner-only health observation, with no traversal of queued records.
    [[nodiscard]] bool executor_is_quarantined() const noexcept;
    // Owner-only preflight allocation; no sequence, receipt or SDK authority.
    [[nodiscard]] ExecutionHostAllocation allocate_host_storage(std::size_t bytes) noexcept;
    [[nodiscard]] ExecutionAcceptance accept(std::uint64_t sequence,
                                             const ExecutionRequest& request) noexcept;
    [[nodiscard]] std::optional<ExecutionIssue> take_next() noexcept;
    // Trusted executor selection after independently checking logical readiness.
    // Exact receipt authority, not a caller-supplied numeric sequence or policy.
    [[nodiscard]] std::optional<ExecutionIssue> take_next(const ExecutionReceipt& receipt) noexcept;
    // Trusted executor-only exact ABI-prefix selection BEFORE first issue.
    // The full owned payload/charge remains retained; this neither resizes nor
    // refunds storage, grows a prefix, consumes a sequence or grants execution.
    [[nodiscard]] bool limit_unissued_parameters(const ExecutionReceipt& receipt,
                                                 std::size_t bytes) noexcept;
    // Owner-only CPU preparation of the same queued receipt. Capacity remains
    // charged until physical retirement; these neither issue nor consume IDs.
    [[nodiscard]] bool copy_unissued_parameters(const ExecutionReceipt& receipt,
                                                std::span<std::byte> output) const noexcept;
    [[nodiscard]] bool replace_unissued_parameters(const ExecutionReceipt& receipt,
                                                   std::span<const std::byte> input) noexcept;
    // Trusted executor-only private ABI update before native issue. Requires the
    // current issuing/resuming generation; no resize, new receipt or quota change.
    // Borrowed parameter spans must only be read on that executor during issue.
    [[nodiscard]] bool patch_issue_parameters(const ExecutionIssueToken& token, std::size_t offset,
                                              std::span<const std::byte> bytes) noexcept;
    [[nodiscard]] bool publish_host_result(const ExecutionIssueToken& token,
                                           std::int32_t native_status) noexcept;
    // The targeted operation was not called, but its private asynchronous
    // preparation may be live. Publish rejection WITHOUT releasing its payload.
    [[nodiscard]] bool publish_preparation_failure(const ExecutionIssueToken& token,
                                                   std::int32_t failure_status) noexcept;
    [[nodiscard]] bool fail_uncertain_issue(const ExecutionIssueToken& token,
                                            std::int32_t failure_status) noexcept;
    [[nodiscard]] bool reject_unissued(const ExecutionReceipt& receipt,
                                       std::int32_t failure_status) noexcept;
    // TRUSTED: the taken command has not called its original native operation.
    [[nodiscard]] bool reject_before_native_issue(const ExecutionIssueToken& token,
                                                  std::int32_t failure_status) noexcept;
    [[nodiscard]] std::optional<ExecutionHostReply> host_result(
        const ExecutionReceipt& receipt) const noexcept;
    [[nodiscard]] bool acknowledge_result(const ExecutionReceipt& receipt) noexcept;
    // TRUSTED: real final logical completion and physical stop are independently
    // proven. SDK return, timeout, retreat and admission are never this proof.
    [[nodiscard]] bool complete_after_physical_stop(const ExecutionIssueToken& token) noexcept;
    // TRUSTED: this exact physical epoch stopped, but logical work is incomplete.
    // Retains parameters/resources and the original reply; does not complete it.
    [[nodiscard]] bool pause_after_physical_stop(const ExecutionIssueToken& token) noexcept;
    // Mints a new nonrecycled physical-attempt generation, never a new receipt.
    // Close permits draining accepted work; quarantine forbids resumption.
    [[nodiscard]] std::optional<ExecutionIssue> resume_after_physical_stop(
        const ExecutionIssueToken& token) noexcept;
    // Actual continuation SDK return, not another application-visible reply.
    [[nodiscard]] bool finish_continuation_issue(const ExecutionIssueToken& token) noexcept;
    void close() noexcept;
    void quarantine() noexcept;
    [[nodiscard]] bool shutdown() noexcept;
    [[nodiscard]] ExecutionSubmissionSnapshot snapshot() const noexcept;
    [[nodiscard]] ExecutionSessionIdentity identity() const noexcept;

   private:
    [[nodiscard]] std::optional<ExecutionIssue> take_next_impl(
        const ExecutionReceipt* receipt) noexcept;
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
};

}  // namespace glimmer::backend
