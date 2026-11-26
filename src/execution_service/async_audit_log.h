#pragma once

#include <memory>
#include <string>
#include <utility>

#include <spdlog/fmt/fmt.h>

namespace glimmer::app {

// A bounded diagnostics-only queue. A full queue or logging failure drops
// evidence and marks it lost; it never changes scheduling or SDK results.
class AsyncAuditLog final {
   public:
    AsyncAuditLog();
    ~AsyncAuditLog();
    AsyncAuditLog(const AsyncAuditLog&) = delete;
    AsyncAuditLog& operator=(const AsyncAuditLog&) = delete;

    template <class... Args>
    bool info(fmt::format_string<Args...> format, Args&&... args) noexcept {
        try {
            return enqueue(fmt::format(format, std::forward<Args>(args)...));
        } catch (...) {
            mark_lost();
            return false;
        }
    }
    [[nodiscard]] bool lost() const noexcept;
    // Only call after the service has no more audit producers.
    void flush() noexcept;

   private:
    struct Implementation;
    std::unique_ptr<Implementation> implementation_;
    bool enqueue(std::string message) noexcept;
    void mark_lost() noexcept;
};

}  // namespace glimmer::app
