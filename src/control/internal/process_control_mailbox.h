#pragma once

#include "remote_control_operation.h"
#include "local_control_wake.h"
#include "prepared_control_future.h"

#include <memory>
#include <optional>
#include <thread>

namespace glimmer::control::internal {

struct ProcessControlState;
struct ControlCompletionState;

struct ControlRequestOutcome {
    RemoteControlStep step = RemoteControlStep::kRejected;
    RemoteControlResult result;
    bool was_published = false;
};

// Retains CPU result storage only, not an endpoint or CUDA ownership. Copying a
// ticket cannot duplicate consumption or allow inherited fork authorization.
class ControlCompletionTicket final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] bool is_ready() const noexcept;
    [[nodiscard]] std::optional<ControlRequestOutcome> take_result() noexcept;

   private:
    friend struct ProcessControlState;
    std::shared_ptr<ControlCompletionState> state_;
};

class ControlOwnerToken final {
   public:
    [[nodiscard]] explicit operator bool() const noexcept;

   private:
    friend struct ProcessControlState;
    std::shared_ptr<const void> identity_;
    std::size_t slot_ = 0;
    std::uint64_t generation_ = 0;
};

enum class ControlMailboxProgress : std::uint8_t {
    kRejected,
    kIdle,
    kPending,
    kCompleted,
    kQuarantined
};

struct ControlMailboxSnapshot {
    std::size_t unsettled_requests = 0;
    std::size_t registered_owners = 0;
    bool has_executor = false;
    bool is_advancing = false;
    bool is_closed = false;
    bool is_quarantined = false;
    std::size_t executor_owners = 0;
    std::size_t accepting_executor_owners = 0;
};

struct ProcessControlMailboxBounds {
    std::size_t max_outstanding_requests = 1024;
    std::size_t max_owners = 64;
    // Prepared typed completion/submission only; default behavior stays exact.
    bool should_combine_adjacent_requests = false;
    // Optional legacy queued-only bound, allowing up to two active originals.
    std::size_t max_queued_requests = 0;
};

// Move-only, thread-bound CPU execution eligibility, not CUDA permission.
// Externally serialize progress, observation, move and destruction. Releasing
// between steps leaves any exact pending operation/endpoint owned by the gate.
class ControlExecutorLease final {
   public:
    ~ControlExecutorLease();
    ControlExecutorLease(ControlExecutorLease&& other) noexcept;
    ControlExecutorLease(const ControlExecutorLease&) = delete;
    ControlExecutorLease& operator=(const ControlExecutorLease&) = delete;
    ControlExecutorLease& operator=(ControlExecutorLease&&) = delete;
    [[nodiscard]] ControlMailboxProgress progress(
        RemoteControlOperation::Clock::time_point now =
            RemoteControlOperation::Clock::now()) noexcept;
    [[nodiscard]] std::optional<OwnedExchangeWait> wait_state() const noexcept;
    // Explicit sleeping boundary; hold this lease and keep its endpoint stable.
    // The caller supplies the earliest physical/control/heartbeat deadline.
    // Every outcome is a hint: recheck actual control/domain/physical state.
    [[nodiscard]] ControlWaitOutcome wait_until(
        RemoteControlOperation::Clock::time_point deadline) noexcept;

   private:
    friend struct ProcessControlState;
    ControlExecutorLease(std::shared_ptr<ProcessControlState> state, ControlOwnerToken owner,
                         std::uint64_t generation) noexcept;
    std::shared_ptr<ProcessControlState> state_;
    ControlOwnerToken owner_;
    std::uint64_t generation_ = 0;
    std::thread::id thread_;
};

// Thread-safe CPU FIFO/registration; exactly one executor advances it. The
// router must own one mailbox for its entire logical control session. No thread,
// callback, CUDA or automatic sleeping progress is created here. Explicit wait
// observes the private activity/reply wake; it is not a broadcast signal. Destruction
// is externally serialized with public calls; leases retain CPU state. Domain
// quarantine MUST remain router-owned until physical owners are contained.
class ProcessControlMailbox final {
   public:
    // Includes both original records of a pair and retained uncertainty.
    explicit ProcessControlMailbox(ProcessControlMailboxBounds bounds = {});
    ~ProcessControlMailbox();
    ProcessControlMailbox(const ProcessControlMailbox&) = delete;
    ProcessControlMailbox& operator=(const ProcessControlMailbox&) = delete;
    ProcessControlMailbox(ProcessControlMailbox&&) = delete;
    ProcessControlMailbox& operator=(ProcessControlMailbox&&) = delete;

    // Once-only finite cold startup, before registration and acceptance.
    [[nodiscard]] OwnedExchangeColdStatus initialize(OwnedExchangeConfig config) noexcept;
    [[nodiscard]] ControlOwnerToken register_owner() noexcept;
    // Metadata-only identity cannot lease or keep the last CPU executor alive.
    [[nodiscard]] ControlOwnerToken register_producer() noexcept;
    // Withdraw new acceptance eligibility atomically; retain lease eligibility
    // for accepted CPU drain. A later distinct registration may accept again.
    [[nodiscard]] bool retire_owner(const ControlOwnerToken& owner) noexcept;
    // Last detach is refused until every accepted outcome is settled. Permanent
    // quarantine remains retained even after all owners detach; it is NOT drain.
    [[nodiscard]] bool detach_owner(const ControlOwnerToken& owner) noexcept;
    [[nodiscard]] std::optional<ControlExecutorLease> try_acquire(
        const ControlOwnerToken& owner) noexcept;
    // Prepare operation before calling. Rejected acceptance returns no ticket
    // and performs no domain operation. Producer identity does not own execution.
    [[nodiscard]] ControlCompletionTicket submit(
        const ControlOwnerToken& owner, std::unique_ptr<RemoteControlOperation> operation) noexcept;
    // The original typed future is published before ticket/notification/head
    // retirement. Obtain it before submit, but return it publicly only after
    // valid acceptance. Failed terminal ACKs permanently contain successors.
    [[nodiscard]] ControlCompletionTicket submit(
        const ControlOwnerToken& owner, std::unique_ptr<PreparedControlFuture> request) noexcept;
    void close() noexcept;
    [[nodiscard]] bool notify_activity() noexcept;
    // CPU containment only; retains accepted records and cannot acknowledge
    // domain ownership or claim successful drain.
    void quarantine() noexcept;
    [[nodiscard]] ControlMailboxSnapshot snapshot() const noexcept;

   private:
    std::shared_ptr<ProcessControlState> state_;
};

}  // namespace glimmer::control::internal
