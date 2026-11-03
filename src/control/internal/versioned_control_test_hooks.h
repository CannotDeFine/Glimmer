#pragma once
#include <atomic>

namespace glimmer::control::internal::testing {

inline constexpr unsigned kSocketBeforeWrite = 1U;
inline constexpr unsigned kSocketAfterPartialWrite = 2U;
inline constexpr unsigned kSharedBeforePublish = 4U;
inline constexpr unsigned kSharedAfterPublication = 8U;
inline thread_local unsigned versioned_control_faults = 0;
inline std::atomic<bool> drop_next_task_response{false};

}  // namespace glimmer::control::internal::testing
