#pragma once

#include <cstdint>
#include <limits>

namespace glimmer::app {

// Injected only by explicit diagnostics. The ordinary service never loads a
// profiler. Callbacks run synchronously on the owning issue thread.
struct ExecutionSgemmRangeHooks {
    int (*push)(const char*){};
    int (*pop)(){};
};

struct SgemmIssueRangeSummary {
    std::uint64_t attempts{}, resumes{};
    int push{-1}, pop{-1}, native_status{-1}, completion_status{-1}, library_status{-1};
    bool first_issued{}, has_library_status{}, saturated{};
};

struct SgemmIssueStatuses {
    int native{}, completion{};
};

class SgemmIssueRangeTracker final {
   public:
    explicit SgemmIssueRangeTracker(ExecutionSgemmRangeHooks hooks = {}) noexcept : hooks_(hooks) {}

    // Returns true only for the first target attempt when it is not a resume.
    // Even an unsuccessful push counts as an attempt and cannot be retried as
    // though it were a unique launch. Unrelated issues have no effects.
    [[nodiscard]] bool begin(bool is_target, bool is_resume) noexcept {
        if (!is_target) return false;
        increment(summary_.attempts);
        if (is_resume) increment(summary_.resumes);
        if (summary_.attempts != 1 || is_resume) return false;
        if (hooks_.push) {
            try {
                summary_.push = hooks_.push("glimmer.sgemm.target.issue.v1");
            } catch (...) {
                summary_.push = -1;
            }
        }
        return true;
    }

    void finish(bool first, bool issued, SgemmIssueStatuses statuses, bool has_library,
                int library) noexcept {
        if (!first) return;
        if (summary_.push >= 0 && hooks_.pop) {
            try {
                summary_.pop = hooks_.pop();
            } catch (...) {
                summary_.pop = -1;
            }
        }
        summary_.first_issued = issued;
        summary_.native_status = statuses.native;
        summary_.completion_status = statuses.completion;
        summary_.has_library_status = has_library;
        summary_.library_status = has_library ? library : -1;
    }

    [[nodiscard]] const SgemmIssueRangeSummary& summary() const noexcept {
        return summary_;
    }

   private:
    void increment(std::uint64_t& value) noexcept {
        if (value == std::numeric_limits<std::uint64_t>::max())
            summary_.saturated = true;
        else
            ++value;
    }

    ExecutionSgemmRangeHooks hooks_;
    SgemmIssueRangeSummary summary_;
};

}  // namespace glimmer::app
