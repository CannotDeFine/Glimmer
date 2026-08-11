#include "glimmer/core/scheduler_mode.h"

namespace glimmer::core {

std::optional<SchedulerMode> parse_scheduler_mode(std::string_view value) noexcept {
    if (value == "off") {
        return SchedulerMode::kOff;
    }
    if (value == "observe") {
        return SchedulerMode::kObserve;
    }
    if (value == "enforce") {
        return SchedulerMode::kEnforce;
    }
    return std::nullopt;
}

std::string_view scheduler_mode_name(SchedulerMode mode) noexcept {
    switch (mode) {
        case SchedulerMode::kOff:
            return "off";
        case SchedulerMode::kObserve:
            return "observe";
        case SchedulerMode::kEnforce:
            return "enforce";
    }
    return "invalid";
}

}  // namespace glimmer::core
