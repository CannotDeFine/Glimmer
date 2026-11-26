#pragma once

#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <time.h>
#include <unistd.h>

namespace glimmer::interceptor::execution {
class CopyAudit;
namespace copy_audit_detail {
inline const pid_t kProcess{::getpid()};
inline std::atomic<bool> is_enabled{};
inline std::atomic<std::uint64_t> sequence{};
inline thread_local CopyAudit* current{};
inline thread_local bool is_reporting{};
static_assert(std::atomic<bool>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);

inline std::uint64_t now(clockid_t clock = CLOCK_MONOTONIC) noexcept {
    const auto saved = errno;
    timespec value{};
    const auto status = ::clock_gettime(clock, &value);
    errno = saved;
    constexpr auto billion = std::uint64_t{1000000000};
    if (status || value.tv_sec < 0 || value.tv_nsec < 0 || value.tv_nsec >= 1000000000 ||
        static_cast<std::uint64_t>(value.tv_sec) >
            (UINT64_MAX - static_cast<std::uint64_t>(value.tv_nsec)) / billion)
        return 0;
    return static_cast<std::uint64_t>(value.tv_sec) * billion +
           static_cast<std::uint64_t>(value.tv_nsec);
}
class Buffer {
   public:
    void text(std::string_view value) noexcept {
        if (value.size() > data_.size() - size_) {
            is_valid_ = false;
            return;
        }
        for (const auto character : value) data_[size_++] = character;
    }
    template <class Number>
    void number(Number value) noexcept {
        const auto result = std::to_chars(data_.data() + size_, data_.data() + data_.size(), value);
        if (result.ec != std::errc{})
            is_valid_ = false;
        else
            size_ = static_cast<std::size_t>(result.ptr - data_.data());
    }
    template <class Number>
    void field(std::string_view key, Number value) noexcept {
        text(key);
        number(value);
    }
    void emit() noexcept {
        if (!is_valid_) return;
        const auto written = ::write(STDERR_FILENO, data_.data(), size_);
        if (written != static_cast<ssize_t>(size_))
            return;  // Incomplete diagnostics, never SDK failure.
    }

   private:
    std::array<char, 4096> data_;
    std::size_t size_{};
    bool is_valid_{true};
};
}  // namespace copy_audit_detail

// Cold frontend configuration only. No environment lookup on a copy/RPC path.
inline void configure_copy_audit(bool enabled) noexcept {
    if (::getpid() == copy_audit_detail::kProcess)
        copy_audit_detail::is_enabled.store(enabled, std::memory_order_relaxed);
}

// One actual caller owns its fixed-capacity records. A different TLS caller or
// fork cannot borrow them. Emit only after frontend exclusion has been released;
// timestamps/status/IDs are diagnostics, never execution or completion authority.
class CopyAudit {
   public:
    enum class Api : std::uint32_t {
        kDriverUpload = 1,
        kDriverAsyncUpload,
        kRuntime,
        kRuntimeAsync
    };
    CopyAudit(Api api, std::uint64_t bytes) noexcept : api_(api), bytes_(bytes) {
        using namespace copy_audit_detail;
        if (!is_enabled.load(std::memory_order_relaxed) || current || is_reporting ||
            ::getpid() != kProcess)
            return;
        entry_ = now();
        if (!entry_) return;
        auto previous = sequence.load(std::memory_order_relaxed);
        for (unsigned int attempt = 0; attempt < 8 && previous != UINT64_MAX; ++attempt)
            if (sequence.compare_exchange_weak(previous, previous + 1, std::memory_order_relaxed)) {
                serial_ = previous + 1;
                cpu_entry_ = now(CLOCK_THREAD_CPUTIME_ID);
                current = this;
                return;
            }
    }
    CopyAudit(const CopyAudit&) = delete;
    CopyAudit& operator=(const CopyAudit&) = delete;
    ~CopyAudit() {
        using namespace copy_audit_detail;
        if (!serial_ || ::getpid() != kProcess) return;
        const auto saved = errno;
        is_reporting = true;
        Buffer output;
        output.text("Glimmer execution copy audit: v=1");
        output.field(", pid=", kProcess);
        output.field(", serial=", serial_);
        output.field(", api=", static_cast<std::uint32_t>(api_));
        output.field(", bytes=", bytes_);
        output.field(", entry_ns=", entry_);
        output.field(", locked_ns=", locked_);
        output.field(", finish_ns=", finished_);
        output.field(", cpu_entry_ns=", cpu_entry_);
        output.field(", cpu_finish_ns=", cpu_finished_);
        output.field(", has_result=", static_cast<unsigned int>(has_result_));
        output.field(", status=", status_);
        output.field(", calls=", count_);
        output.field(", dropped=", dropped_);
        output.text("\n");
        for (std::size_t index = 0; index < count_; ++index) {
            const auto& call = calls_[index];
            output.text("Glimmer execution copy rpc: v=1");
            output.field(", pid=", kProcess);
            output.field(", serial=", serial_);
            output.field(", call=", index + 1);
            output.field(", operation=", call.operation);
            output.field(", session=", call.session);
            output.field(", request=", call.request);
            output.field(", begin_ns=", call.begin);
            output.field(", submitted_ns=", call.submitted);
            output.field(", done_ns=", call.done);
            output.field(", has_reply=", static_cast<unsigned int>(call.has_reply));
            output.field(", status=", call.status);
            output.text("\n");
        }
        output.emit();
        current = nullptr;
        is_reporting = false;
        errno = saved;
    }
    void finish(std::int32_t status) noexcept {
        if (!serial_ || ::getpid() != copy_audit_detail::kProcess) return;
        status_ = status;
        has_result_ = true;
        cpu_finished_ = cpu_entry_ ? copy_audit_detail::now(CLOCK_THREAD_CPUTIME_ID) : 0;
        finished_ = copy_audit_detail::now();
    }
    static void acquired() noexcept {
        auto* audit = copy_audit_detail::current;
        if (audit && !audit->locked_ && !copy_audit_detail::is_reporting &&
            ::getpid() == copy_audit_detail::kProcess)
            audit->locked_ = copy_audit_detail::now();
    }

   private:
    friend class CopyRpcAudit;
    struct Call {
        std::uint64_t session, request, begin, submitted, done;
        std::uint32_t operation;
        std::int32_t status;
        bool has_reply;
    };
    std::array<Call, 8> calls_;
    Api api_;
    std::uint64_t bytes_, serial_{}, entry_{}, locked_{}, finished_{}, dropped_{};
    std::uint64_t cpu_entry_{}, cpu_finished_{};
    std::size_t count_{};
    std::int32_t status_{};
    bool has_result_{};
};

class CopyRpcAudit {
   public:
    CopyRpcAudit(std::uint32_t operation, std::uint64_t session) noexcept {
        auto* candidate = copy_audit_detail::current;
        if (!candidate || copy_audit_detail::is_reporting ||
            ::getpid() != copy_audit_detail::kProcess)
            return;
        if (candidate->count_ == candidate->calls_.size()) {
            if (candidate->dropped_ != UINT64_MAX) ++candidate->dropped_;
            return;
        }
        owner_ = candidate;
        index_ = owner_->count_++;
        owner_->calls_[index_] = {session, 0, copy_audit_detail::now(), 0, 0, operation, 0, false};
    }
    CopyRpcAudit(const CopyRpcAudit&) = delete;
    CopyRpcAudit& operator=(const CopyRpcAudit&) = delete;
    ~CopyRpcAudit() {
        if (owner_ && ::getpid() == copy_audit_detail::kProcess)
            owner_->calls_[index_].done = copy_audit_detail::now();
    }
    std::uint64_t* submission_timestamp() noexcept {
        if (!owner_ || ::getpid() != copy_audit_detail::kProcess) return nullptr;
        has_transport_timestamp_ = true;
        return &owner_->calls_[index_].submitted;
    }
    void submitted(std::uint64_t request) noexcept {
        if (!owner_ || ::getpid() != copy_audit_detail::kProcess) return;
        owner_->calls_[index_].request = request;
        if (!has_transport_timestamp_) owner_->calls_[index_].submitted = copy_audit_detail::now();
    }
    void replied(std::int32_t status) noexcept {
        if (!owner_ || ::getpid() != copy_audit_detail::kProcess) return;
        owner_->calls_[index_].status = status;
        owner_->calls_[index_].has_reply = true;
    }

   private:
    CopyAudit* owner_{};
    std::size_t index_{};
    bool has_transport_timestamp_{};
};
}  // namespace glimmer::interceptor::execution
