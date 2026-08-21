#pragma once

#include "glimmer/core/scheduler.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace glimmer::control {

struct TaskAdmissionRequest {
    std::string tenant_id;
    core::MemoryBytes memory_bytes = 0;
    std::uint32_t weight = 1;
    std::uint32_t work_units = 1;
    std::uint32_t priority = 0;
};

// Identity obtained from a trusted local transport such as SO_PEERCRED. The
// process start-time value prevents a recycled PID from inheriting a lease.
struct TaskPeerIdentity {
    std::uint32_t pid = 0;
    std::uint32_t uid = 0;
    std::uint64_t start_time_ticks = 0;

    friend bool operator==(const TaskPeerIdentity&, const TaskPeerIdentity&) = default;
};

using TaskResourceRegistrar = bool (*)(void* context, core::TaskId task_id) noexcept;

// Thread-safe for scheduler operations. The registrar is invoked synchronously
// after admission and must not throw or call back into this service. It owns
// backend-specific resource binding and must keep those resources alive until
// the task reaches a terminal state.
class TaskAdmissionService final {
   public:
    explicit TaskAdmissionService(
        core::Scheduler& scheduler,
        std::chrono::milliseconds lease_timeout = std::chrono::milliseconds::zero(),
        bool bind_leases_to_process = false) noexcept;

    [[nodiscard]] core::SubmitResult submit(const TaskAdmissionRequest& request,
                                            TaskResourceRegistrar registrar,
                                            void* registrar_context,
                                            std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool cancel(core::TaskId task_id);
    [[nodiscard]] bool cancel_queued(core::TaskId task_id);
    // Claims the next queued task and transitions it to running. The returned
    // snapshot is the worker lease metadata; no backend resource is owned by
    // this service. When process binding is enabled, peer must be a trusted
    // local identity with a non-zero start-time value.
    [[nodiscard]] std::optional<core::TaskSnapshot> claim_next(
        std::optional<TaskPeerIdentity> peer = std::nullopt);
    // Claims a specific queued task when it is next under the scheduler
    // policy. This is used by process-bound transparent launch clients so a
    // process cannot accidentally execute another tenant's lease.
    [[nodiscard]] std::optional<core::TaskSnapshot> claim(
        core::TaskId task_id, std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool heartbeat(core::TaskId task_id,
                                 std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool complete(core::TaskId task_id,
                                std::optional<TaskPeerIdentity> peer = std::nullopt);
    [[nodiscard]] bool fail(core::TaskId task_id,
                            std::optional<TaskPeerIdentity> peer = std::nullopt);
    // Expires the active lease when a timeout is configured. Returns true when
    // a running task was transitioned to FAILED and its quota was released.
    [[nodiscard]] bool reap_expired();
    [[nodiscard]] std::optional<core::TaskSnapshot> find(core::TaskId task_id) const;
    [[nodiscard]] core::SchedulerStats stats() const;

   private:
    core::Scheduler& scheduler_;
    using Clock = std::chrono::steady_clock;
    struct LeaseRecord {
        Clock::time_point deadline{};
        bool reaping = false;
        std::optional<TaskPeerIdentity> owner;
    };

    [[nodiscard]] bool owner_matches(const LeaseRecord& lease,
                                     const std::optional<TaskPeerIdentity>& peer) const noexcept;
    [[nodiscard]] std::optional<core::TaskSnapshot> record_lease(
        std::optional<core::TaskSnapshot> snapshot, std::optional<TaskPeerIdentity> peer);

    std::chrono::milliseconds lease_timeout_;
    bool bind_leases_to_process_ = false;
    std::mutex lease_mutex_;
    std::unordered_map<core::TaskId, LeaseRecord> active_leases_;
    std::unordered_map<core::TaskId, LeaseRecord> pending_leases_;
};

}  // namespace glimmer::control
