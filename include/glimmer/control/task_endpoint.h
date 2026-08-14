#pragma once

#include "glimmer/control/task_admission.h"

#include <optional>
#include <string>
#include <string_view>

namespace glimmer::control {

// Maps one complete protocol line to the process-local admission service.
// This class does not own a transport or CUDA resource. The registrar is
// called synchronously for accepted submissions and must obey the same
// lifetime contract as TaskAdmissionService.
class TaskControlEndpoint final {
   public:
    TaskControlEndpoint(TaskAdmissionService& admission_service, TaskResourceRegistrar registrar,
                        void* registrar_context) noexcept;

    // Returns one canonical response line. A null result means the response
    // could not be constructed, for example because of an allocation failure.
    [[nodiscard]] std::optional<std::string> handle(std::string_view line) noexcept;
    [[nodiscard]] std::optional<std::string> handle(std::string_view line,
                                                    std::optional<TaskPeerIdentity> peer) noexcept;

   private:
    TaskAdmissionService& admission_service_;
    TaskResourceRegistrar registrar_;
    void* registrar_context_;
};

}  // namespace glimmer::control
