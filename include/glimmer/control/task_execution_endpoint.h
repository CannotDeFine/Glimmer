#pragma once

#include "glimmer/control/task_execution_admission.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

// Releases a successfully registered resource when its task is retired before
// any response granting or accepting it is returned. Must not throw or call
// back into the endpoint, and must release that task's resource synchronously.
using TaskResourceRollback = void (*)(void* context, core::TaskId task_id) noexcept;

// Owns the admission service and routes complete lines by exact wire envelope.
// The transferred service must have no pending or active work, including if
// endpoint construction fails before ownership is fully established.
// The registrar and observer contexts must outlive the endpoint. The registrar
// may bind resources but must not publish or issue work; observers are telemetry
// only and must not issue work. Legacy queued cancellation is owner-checked
// when process binding is enabled. QUERY/STATS/METRICS are not owner-filtered;
// the transport must restrict who can reach them. Before
// destruction the owner must stop admission and all callers, stop and join
// backend work, retire every lease, and release registered external resources.
// Endpoint destruction does not prove GPU completion or fence a CUDA context.
class TaskExecutionEndpoint final {
   public:
    explicit TaskExecutionEndpoint(std::unique_ptr<TaskExecutionAdmissionService> service,
                                   TaskResourceRegistrar registrar, TaskResourceRollback rollback,
                                   void* registrar_context);
    ~TaskExecutionEndpoint();

    TaskExecutionEndpoint(const TaskExecutionEndpoint&) = delete;
    TaskExecutionEndpoint& operator=(const TaskExecutionEndpoint&) = delete;
    TaskExecutionEndpoint(TaskExecutionEndpoint&&) = delete;
    TaskExecutionEndpoint& operator=(TaskExecutionEndpoint&&) = delete;

    [[nodiscard]] std::optional<std::string> handle(
        std::string_view line, std::optional<TaskPeerIdentity> peer = std::nullopt) noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace glimmer::control
