#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace glimmer::core {

enum class SchedulerMode : std::uint8_t {
    kOff,
    kObserve,
    kEnforce,
};

[[nodiscard]] std::optional<SchedulerMode> parse_scheduler_mode(std::string_view value) noexcept;
[[nodiscard]] std::string_view scheduler_mode_name(SchedulerMode mode) noexcept;

}  // namespace glimmer::core
