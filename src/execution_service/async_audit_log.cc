#include "async_audit_log.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unistd.h>

#include <spdlog/spdlog.h>

namespace glimmer::app {
struct AsyncAuditLog::Implementation {
    static constexpr std::size_t k_max_entries = 4096;
    static constexpr std::size_t k_max_bytes = std::size_t{4} * 1024 * 1024;
    const pid_t process{::getpid()};
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::string> records;
    std::size_t bytes{};
    bool stopping{}, writing{};
    std::atomic<bool> lost{};
    std::shared_ptr<spdlog::logger> logger;
    std::thread worker;

    Implementation() : logger(spdlog::default_logger()->clone("glimmer-audit")) {
        // spdlog normally catches sink exceptions internally. The private
        // handler makes those failures visible without mutating the process's
        // global logger or its handler.
        logger->set_error_handler([this](const std::string&) { lost.store(true); });
    }

    void run() noexcept {
        for (;;) {
            std::string record;
            {
                std::unique_lock lock(mutex);
                changed.wait(lock, [&] { return stopping || !records.empty(); });
                if (records.empty() && stopping) return;
                record = std::move(records.front());
                records.pop_front();
                bytes -= record.size();
                writing = true;
            }
            try {
                logger->info("{}", record);
            } catch (...) {
                lost.store(true);
            }
            {
                const std::lock_guard lock(mutex);
                writing = false;
            }
            changed.notify_all();
        }
    }
};

AsyncAuditLog::AsyncAuditLog() : implementation_(std::make_unique<Implementation>()) {
    implementation_->worker = std::thread([state = implementation_.get()] { state->run(); });
}
AsyncAuditLog::~AsyncAuditLog() {
    auto& state = *implementation_;
    if (::getpid() != state.process) {
        (void)implementation_.release();
        return;
    }
    {
        const std::lock_guard lock(state.mutex);
        state.stopping = true;
    }
    state.changed.notify_all();
    if (state.worker.joinable()) state.worker.join();
}
bool AsyncAuditLog::enqueue(std::string message) noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process) return false;
    try {
        {
            const std::lock_guard lock(state.mutex);
            if (state.stopping || state.records.size() == Implementation::k_max_entries ||
                message.size() > Implementation::k_max_bytes - state.bytes) {
                state.lost.store(true);
                return false;
            }
            const auto bytes = message.size();
            state.records.push_back(std::move(message));
            state.bytes += bytes;
        }
        state.changed.notify_one();
        return true;
    } catch (...) {
        state.lost.store(true);
        return false;
    }
}
void AsyncAuditLog::mark_lost() noexcept {
    if (::getpid() == implementation_->process) implementation_->lost.store(true);
}
bool AsyncAuditLog::lost() const noexcept {
    return ::getpid() != implementation_->process || implementation_->lost.load();
}
void AsyncAuditLog::flush() noexcept {
    auto& state = *implementation_;
    if (::getpid() != state.process) return;
    try {
        {
            std::unique_lock lock(state.mutex);
            state.changed.wait(lock, [&] { return state.records.empty() && !state.writing; });
        }
        state.logger->flush();
    } catch (...) {
        state.lost.store(true);
    }
}
}  // namespace glimmer::app
