#pragma once

#include "glimmer/control/task_admission.h"

namespace glimmer::control {
[[nodiscard]] bool read_process_start_time(std::uint32_t pid,
                                           std::uint64_t* start_time_ticks) noexcept;
[[nodiscard]] std::optional<TaskPeerIdentity> current_process_identity() noexcept;
}  // namespace glimmer::control
