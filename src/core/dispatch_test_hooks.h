#pragma once

#include <cstdint>

// Available only to host test builds. Production builds do not compile hooks.
namespace glimmer::core::internal {
inline constexpr std::uint32_t kDispatchFailSnapshot = 1U;
inline constexpr std::uint32_t kDispatchFailCancel = 2U;
inline constexpr std::uint32_t kDispatchFailRelease = 4U;
inline constexpr std::uint32_t kDispatchFailActiveRecord = 8U;
extern thread_local std::uint32_t dispatch_test_faults;
}  // namespace glimmer::core::internal
