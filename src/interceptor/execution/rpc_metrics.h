#pragma once

#include "copy_audit.h"
#include "glimmer/control/execution_protocol.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fcntl.h>
#include <locale>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace glimmer::interceptor::execution {

// Default-off CPU diagnostics. Record mutation requires the existing frontend
// exclusion; no additional mutex or transport/CUDA operation is introduced.
// Each scope owns its timestamps until completion, including while exclusion
// is temporarily released. Array order is completion order, not timestamp order.
class RpcMetrics final {
   public:
    struct Stamp {
        std::uint64_t wall{}, cpu{};
    };
    struct Request {
        std::uint64_t session, opcode;
    };
    struct Interval {
        std::uint64_t begin, end;
    };
    struct Record {
        std::uint64_t kind{}, serial{}, thread{}, boundary{}, session{}, wire{}, opcode{};
        std::uint64_t enter{}, locked{}, send_begin{}, send_end{}, reply{}, finish{};
        std::uint64_t cpu_enter{}, cpu_finish{}, flags{};
    };
    enum class Phase : std::uint64_t {
        kQuickYield,
        kWaitCall,
        kFallbackSleep,
        kMutexReacquire,
        kPollTake,
        kBlasValueRangeSuccess,
        kBlasValueRangeFailure,
        kBlasStreamRangeSuccess,
        kBlasStreamRangeFailure
    };
    enum class HostRangeCaller { kBlasValue, kBlasStream };
    struct PhaseRecord {
        std::uint64_t serial{}, phase{}, begin{}, end{};
    };
    enum class HostStage : std::size_t { kOpen, kRead, kParse, kCloseOther };
    struct HostRecord {
        std::uint64_t serial{}, begin{}, end{}, cpu_begin{}, cpu_end{};
        std::array<std::uint64_t, 4> wall{}, cpu{};
        std::uint64_t reads{}, bytes{}, closing{}, hint_attempted{}, hint_hit{};
    };
    // Stack-owned by the outer callback scope, so descriptor destruction is
    // included. Each transition shares its timestamp with both adjacent stages.
    class HostTrace final {
       public:
        explicit HostTrace(bool enabled) noexcept : enabled_(enabled) {
            if (!enabled_) return;
            last_ = stamp();
            row_.begin = last_.wall;
            row_.cpu_begin = last_.cpu;
            valid_ = last_.wall && last_.cpu;
        }
        void transition(HostStage next) noexcept {
            if (!enabled_) return;
            advance();
            const auto index = static_cast<std::size_t>(next);
            if (index >= row_.wall.size()) {
                valid_ = false;
                return;
            }
            stage_ = index;
            if (next == HostStage::kCloseOther) ++row_.closing;
        }
        void read_result(ssize_t bytes) noexcept {
            if (!enabled_) return;
            if (row_.reads == UINT64_MAX ||
                (bytes > 0 && static_cast<std::uint64_t>(bytes) > UINT64_MAX - row_.bytes)) {
                valid_ = false;
                return;
            }
            ++row_.reads;
            if (bytes > 0) row_.bytes += static_cast<std::uint64_t>(bytes);
        }
        void hint_result(bool hit) noexcept {
            if (!enabled_) return;
            if (row_.hint_attempted) valid_ = false;
            row_.hint_attempted = 1;
            row_.hint_hit = hit ? 1 : 0;
        }
        [[nodiscard]] bool finish(std::uint64_t serial) noexcept {
            advance();
            row_.serial = serial;
            row_.end = last_.wall;
            row_.cpu_end = last_.cpu;
            return enabled_ && valid_ && row_.closing <= 1;
        }
        [[nodiscard]] const HostRecord& record() const noexcept {
            return row_;
        }

       private:
        static Stamp stamp() noexcept {
            return {copy_audit_detail::now(), copy_audit_detail::now(CLOCK_THREAD_CPUTIME_ID)};
        }
        void advance() noexcept {
            if (!enabled_) return;
            const auto next = stamp();
            if (!next.wall || !next.cpu || next.wall < last_.wall || next.cpu < last_.cpu) {
                valid_ = false;
                return;
            }
            row_.wall[stage_] += next.wall - last_.wall;
            row_.cpu[stage_] += next.cpu - last_.cpu;
            last_ = next;
        }
        HostRecord row_{};
        Stamp last_{};
        std::size_t stage_{};
        bool enabled_{}, valid_{};
    };
    static_assert(sizeof(Record) == 128);
    static_assert(sizeof(PhaseRecord) == 32);
    static_assert(sizeof(HostRecord) == 144);
    static constexpr std::size_t capacity_limit = 2097152;
    static constexpr std::size_t phase_capacity_limit = 8388608;
    static constexpr std::size_t host_capacity_limit = 131072;
    static constexpr std::size_t output_limit = std::size_t{1023} * 1024 * 1024;
    // Include delimiters and the worst 20 decimal digits for every uint64.
    static_assert(capacity_limit * (338 + 72) + 8192 < output_limit);
    static_assert(phase_capacity_limit * 86 + host_capacity_limit * 380 + 8192 < output_limit);
    explicit RpcMetrics(std::string directory, std::size_t capacity = capacity_limit,
                        bool host_detail = false)
        : process_(::getpid()),
          directory_(std::move(directory)),
          records_(capacity <= capacity_limit ? capacity : 0),
          phases_(capacity <= capacity_limit ? capacity * 4 : 0),
          hosts_(host_detail && capacity <= capacity_limit ? std::min(capacity, host_capacity_limit)
                                                           : 0),
          host_detail_(host_detail) {
        valid_ = capacity && capacity <= capacity_limit && valid_directory(directory_);
        started_ = copy_audit_detail::now();
        valid_ = valid_ && started_ != 0;
    }
    [[nodiscard]] static bool valid_directory(const std::string& directory) noexcept {
        try {
            const std::filesystem::path path{directory};
            struct stat info {};
            return path.is_absolute() && path.lexically_normal() == path &&
                   std::filesystem::canonical(path) == path && ::lstat(path.c_str(), &info) == 0 &&
                   S_ISDIR(info.st_mode) && info.st_uid == ::geteuid();
        } catch (...) {
            return false;
        }
    }
    [[nodiscard]] Stamp entry() const noexcept {
        if (!recording_.load(std::memory_order_relaxed) || ::getpid() != process_) return {};
        return {copy_audit_detail::now(), copy_audit_detail::now(CLOCK_THREAD_CPUTIME_ID)};
    }
    // Called under the frontend exclusion after its background workers join.
    // No records may be appended after this explicit observation cutoff.
    void seal() noexcept {
        recording_.store(false, std::memory_order_relaxed);
        ended_ = copy_audit_detail::now();
        if (active_ || !ended_ || ended_ < started_) valid_ = false;
    }
    [[nodiscard]] bool valid() const noexcept {
        return valid_;
    }
    [[nodiscard]] std::size_t size() const noexcept {
        return used_;
    }
    [[nodiscard]] const Record& record(std::size_t index) const {
        return records_.at(index);
    }
    [[nodiscard]] std::size_t phase_size() const noexcept {
        return phases_used_;
    }
    [[nodiscard]] const PhaseRecord& phase_record(std::size_t index) const {
        return phases_.at(index);
    }
    [[nodiscard]] std::size_t host_size() const noexcept {
        return hosts_used_;
    }
    [[nodiscard]] const HostRecord& host_record(std::size_t index) const {
        return hosts_.at(index);
    }

    class Scope final {
       public:
        Scope(RpcMetrics* metrics, std::unique_lock<std::mutex>& exclusion, Stamp entry,
              bool boundary, Request request = {}) noexcept
            : metrics_(metrics), exclusion_(exclusion), is_boundary_(boundary) {
            if (!metrics_ || ::getpid() != metrics_->process_ ||
                !metrics_->recording_.load(std::memory_order_relaxed)) {
                metrics_ = nullptr;
                return;
            }
            if (!exclusion_.owns_lock()) {
                metrics_->valid_ = false;
                metrics_ = nullptr;
                return;
            }
            if (!boundary) entry = metrics_->entry();
            row_.kind = boundary ? 0 : 1;
            row_.serial = ++metrics_->serial_;
            row_.thread = static_cast<std::uint64_t>(::gettid());
            row_.boundary = current_boundary;
            row_.session = request.session;
            row_.opcode = request.opcode;
            row_.enter = entry.wall;
            row_.cpu_enter = entry.cpu;
            row_.locked = copy_audit_detail::now();
            ++metrics_->active_;
            if (is_boundary_) {
                current_boundary = row_.serial;
                previous_boundary_scope_ = current_boundary_scope;
                current_boundary_scope = this;
            }
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() {
            if (!metrics_ || ::getpid() != metrics_->process_) return;
            if (is_boundary_) {
                current_boundary = row_.boundary;
                current_boundary_scope = previous_boundary_scope_;
            }
            if (!exclusion_.owns_lock()) {
                metrics_->valid_ = false;
                return;
            }
            const auto end = metrics_->entry();
            row_.finish = end.wall;
            row_.cpu_finish = end.cpu;
            --metrics_->active_;
            if (!row_.serial || !row_.enter || !row_.cpu_enter || row_.enter > row_.locked ||
                row_.locked > row_.finish || row_.cpu_enter > row_.cpu_finish ||
                last_phase_end_ > row_.finish ||
                (row_.wire && (!row_.send_begin || row_.send_begin < row_.locked ||
                               row_.send_begin > row_.send_end || row_.send_end > row_.reply ||
                               row_.reply > row_.finish)) ||
                metrics_->used_ == metrics_->records_.size()) {
                metrics_->valid_ = false;
                return;
            }
            metrics_->records_[metrics_->used_++] = row_;
        }
        void sending() noexcept {
            if (metrics_ && ::getpid() == metrics_->process_)
                row_.send_begin = copy_audit_detail::now();
        }
        [[nodiscard]] bool observes_delivery() const noexcept {
            return metrics_ != nullptr;
        }
        void submitted(std::uint64_t wire,
                       std::optional<bool> fully_flushed = std::nullopt) noexcept {
            if (!metrics_ || ::getpid() != metrics_->process_) return;
            row_.wire = wire;
            row_.send_end = copy_audit_detail::now();
            row_.flags |= 1;
            if (fully_flushed)
                row_.flags |= delivery_known | (*fully_flushed ? fully_flushed_bit : 0);
        }
        void received() noexcept {
            if (!metrics_ || ::getpid() != metrics_->process_) return;
            row_.reply = copy_audit_detail::now();
            row_.flags |= 2;
        }
        // Only accepted asynchronous-frontier RPCs collect wait phases. The
        // clock may be read while unlocked; completed pairs are appended only
        // after the existing exclusion has been reacquired by this same thread.
        [[nodiscard]] std::uint64_t phase_clock() const noexcept {
            if (!watches_waits() || ::getpid() != metrics_->process_) return 0;
            return copy_audit_detail::now();
        }
        void phase(Phase kind, Interval interval) noexcept {
            if (!watches_waits() || ::getpid() != metrics_->process_) return;
            const auto value = static_cast<std::uint64_t>(kind);
            if (!exclusion_.owns_lock() || static_cast<std::uint64_t>(::gettid()) != row_.thread ||
                value > static_cast<std::uint64_t>(Phase::kPollTake) || !interval.begin ||
                interval.begin < row_.enter || interval.begin < last_phase_end_ ||
                interval.begin > interval.end ||
                metrics_->phases_used_ == metrics_->phases_.size()) {
                metrics_->valid_ = false;
                return;
            }
            metrics_->phases_[metrics_->phases_used_++] = {row_.serial, value, interval.begin,
                                                           interval.end};
            last_phase_end_ = interval.end;
        }
        // Observe only the unchanged local validation callback. Unlike wait
        // phases, these spans belong to the active API boundary, not an RPC.
        // Both timestamps and the append occur under the existing exclusion.
        template <typename Function>
        [[nodiscard]] static bool host_range(RpcMetrics* metrics,
                                             std::unique_lock<std::mutex>& exclusion,
                                             HostRangeCaller caller, Function&& function) noexcept {
            if (!metrics || ::getpid() != metrics->process_ ||
                !metrics->recording_.load(std::memory_order_relaxed))
                return function(nullptr);
            auto* boundary = current_boundary_scope;
            if (!boundary || boundary->metrics_ != metrics || &boundary->exclusion_ != &exclusion ||
                !exclusion.owns_lock() ||
                boundary->row_.thread != static_cast<std::uint64_t>(::gettid()) ||
                !current_boundary || boundary->row_.serial != current_boundary ||
                (caller != HostRangeCaller::kBlasValue && caller != HostRangeCaller::kBlasStream)) {
                metrics->valid_ = false;
                return function(nullptr);
            }
            const auto begin = copy_audit_detail::now();
            HostTrace trace(metrics->host_detail_);
            const bool result = function(metrics->host_detail_ ? &trace : nullptr);
            if (metrics->host_detail_) {
                if (!trace.finish(current_boundary) ||
                    metrics->hosts_used_ == metrics->hosts_.size())
                    metrics->valid_ = false;
                else
                    metrics->hosts_[metrics->hosts_used_++] = trace.record();
            }
            const auto end = copy_audit_detail::now();
            if (!begin || begin < boundary->row_.locked || begin < boundary->last_phase_end_ ||
                begin > end || !exclusion.owns_lock() ||
                metrics->phases_used_ == metrics->phases_.size()) {
                metrics->valid_ = false;
                return result;
            }
            const auto kind = caller == HostRangeCaller::kBlasValue
                                  ? Phase::kBlasValueRangeSuccess
                                  : Phase::kBlasStreamRangeSuccess;
            metrics->phases_[metrics->phases_used_++] = {
                current_boundary, static_cast<std::uint64_t>(kind) + (result ? 0U : 1U), begin,
                end};
            boundary->last_phase_end_ = end;
            return result;
        }

       private:
        [[nodiscard]] bool watches_waits() const noexcept {
            using control::ExecutionOpcode;
            return metrics_ && row_.wire &&
                   (row_.opcode ==
                        static_cast<std::uint64_t>(ExecutionOpcode::kQueryAsyncFrontiers) ||
                    row_.opcode ==
                        static_cast<std::uint64_t>(ExecutionOpcode::kReleaseAsyncFrontiers));
        }
        RpcMetrics* metrics_{};
        std::unique_lock<std::mutex>& exclusion_;
        Record row_{};
        bool is_boundary_{};
        std::uint64_t last_phase_end_{};
        Scope* previous_boundary_scope_{};
        inline static thread_local std::uint64_t current_boundary{};
        inline static thread_local Scope* current_boundary_scope{};
    };

    // Cold, after seal and outside frontend exclusion. Failures affect evidence
    // only; no runtime status is changed and an existing file is never replaced.
    [[nodiscard]] bool publish() const noexcept {
        const auto saved_errno = errno;
        struct Restore {
            int value;
            ~Restore() {
                errno = value;
            }
        } restore{saved_errno};
        try {
            if (::getpid() != process_ || recording_.load(std::memory_order_relaxed) ||
                !valid_directory(directory_))
                return false;
            // Publish independently bounded ledgers. Consumers must require
            // both files and matching lifetime/count metadata; partial output
            // never qualifies an observation. Publish the base commit marker last.
            for (const bool is_phase_file : {true, false}) {
                std::ostringstream json;
                json.imbue(std::locale::classic());
                json << "{\"schema\":6,\"diagnostic_only\":true,\"valid\":"
                     << (valid_ ? "true" : "false") << ",\"pid\":" << process_
                     << ",\"begin_ns\":" << started_ << ",\"end_ns\":" << ended_
                     << ",\"base_count\":" << used_ << ",\"phase_count\":" << phases_used_
                     << ",\"host_detail\":" << (host_detail_ ? "true" : "false")
                     << ",\"host_count\":" << hosts_used_;
                if (!is_phase_file) {
                    json << ",\"capacity\":" << records_.size()
                         << ",\"fields\":[\"kind\",\"serial\",\"thread\",\"boundary\",\"session\","
                            "\"wire\","
                            "\"opcode\",\"enter\",\"locked\",\"send_begin\",\"send_end\",\"reply\","
                            "\"finish\","
                            "\"cpu_enter\",\"cpu_finish\",\"flags\"],\"records\":[";
                    for (std::size_t i = 0; i < used_; ++i) {
                        const auto& r = records_[i];
                        json << (i ? ",[" : "[") << r.kind << ',' << r.serial << ',' << r.thread
                             << ',' << r.boundary << ',' << r.session << ',' << r.wire << ','
                             << r.opcode << ',' << r.enter << ',' << r.locked << ',' << r.send_begin
                             << ',' << r.send_end << ',' << r.reply << ',' << r.finish << ','
                             << r.cpu_enter << ',' << r.cpu_finish << ',' << (r.flags & 3) << ']';
                        if (!json || json.tellp() < 0 ||
                            static_cast<std::size_t>(json.tellp()) > output_limit)
                            return false;
                    }
                    json << "],\"submit_delivery\":{\"schema\":1,\"fields\":[\"serial\","
                            "\"session\",\"wire\",\"fully_flushed\"],\"records\":[";
                    bool first = true;
                    for (std::size_t i = 0; i < used_; ++i) {
                        const auto& r = records_[i];
                        if (!(r.flags & delivery_known)) continue;
                        json << (first ? "[" : ",[") << r.serial << ',' << r.session << ','
                             << r.wire << ',' << ((r.flags & fully_flushed_bit) ? "true" : "false")
                             << ']';
                        first = false;
                        if (!json || json.tellp() < 0 ||
                            static_cast<std::size_t>(json.tellp()) > output_limit)
                            return false;
                    }
                    json << ']';
                } else {
                    json << ",\"phase_fields\":[\"serial\",\"phase\",\"begin\",\"end\"],"
                            "\"phase_names\":[\"quick_yield\",\"wait_call\",\"fallback_sleep\","
                            "\"mutex_reacquire\",\"poll_take\",\"blas_value_range_success\","
                            "\"blas_value_range_failure\",\"blas_stream_range_success\","
                            "\"blas_stream_range_failure\"],\"phase_capacity\":"
                         << phases_.size() << ",\"phase_events\":[";
                    for (std::size_t i = 0; i < phases_used_; ++i) {
                        const auto& p = phases_[i];
                        json << (i ? ",[" : "[") << p.serial << ',' << p.phase << ',' << p.begin
                             << ',' << p.end << ']';
                        if (!json || json.tellp() < 0 ||
                            static_cast<std::size_t>(json.tellp()) > output_limit)
                            return false;
                    }
                    json << "],\"host_capacity\":" << hosts_.size()
                         << ",\"host_fields\":[\"serial\",\"begin\",\"end\",\"cpu_begin\",\"cpu_"
                            "end\","
                            "\"open_wall\",\"read_wall\",\"parse_wall\",\"close_other_wall\","
                            "\"open_cpu\",\"read_cpu\",\"parse_cpu\",\"close_other_cpu\","
                            "\"reads\",\"bytes\",\"closing\",\"hint_attempted\",\"hint_hit\"],"
                            "\"host_details\":[";
                    for (std::size_t i = 0; i < hosts_used_; ++i) {
                        const auto& h = hosts_[i];
                        json << (i ? ",[" : "[") << h.serial << ',' << h.begin << ',' << h.end
                             << ',' << h.cpu_begin << ',' << h.cpu_end;
                        for (const auto value : h.wall) json << ',' << value;
                        for (const auto value : h.cpu) json << ',' << value;
                        json << ',' << h.reads << ',' << h.bytes << ',' << h.closing << ','
                             << h.hint_attempted << ',' << h.hint_hit << ']';
                        if (!json || json.tellp() < 0 ||
                            static_cast<std::size_t>(json.tellp()) > output_limit)
                            return false;
                    }
                }
                json << (is_phase_file ? "]}\n" : "}}\n");
                const auto bytes = json.str();
                if (bytes.size() > output_limit) return false;
                if (!publish_file(is_phase_file, bytes)) return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

   private:
    // Internal bits do not change the existing serialized lifecycle flags.
    static constexpr std::uint64_t delivery_known = std::uint64_t{1} << 62;
    static constexpr std::uint64_t fully_flushed_bit = std::uint64_t{1} << 63;
    [[nodiscard]] bool publish_file(bool is_phase_file, const std::string& bytes) const {
        const auto name = std::string{is_phase_file ? "frontend-phases-" : "frontend-rpc-"} +
                          std::to_string(process_) + ".json";
        struct Descriptor {
            int value;
            ~Descriptor() {
                if (value >= 0) (void)::close(value);
            }
        } directory{::open(directory_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)};
        struct stat info {};
        if (directory.value < 0 || ::fstat(directory.value, &info) != 0 ||
            info.st_uid != ::geteuid())
            return false;
        const auto temporary = name + ".pending";
        Descriptor file{::openat(directory.value, temporary.c_str(),
                                 O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600)};
        if (file.value < 0) return false;
        bool ok = true;
        std::size_t offset{};
        while (offset < bytes.size()) {
            const auto written = ::write(file.value, bytes.data() + offset, bytes.size() - offset);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                ok = false;
                break;
            }
            offset += static_cast<std::size_t>(written);
        }
        ok = (::fsync(file.value) == 0) && ok;
        const auto descriptor = file.value;
        file.value = -1;
        ok = (::close(descriptor) == 0) && ok;
        if (ok)
            ok =
                ::linkat(directory.value, temporary.c_str(), directory.value, name.c_str(), 0) == 0;
        const bool removed = ::unlinkat(directory.value, temporary.c_str(), 0) == 0;
        return ::fsync(directory.value) == 0 && ok && removed;
    }
    pid_t process_{};
    std::string directory_;
    std::vector<Record> records_;
    std::vector<PhaseRecord> phases_;
    std::vector<HostRecord> hosts_;
    bool host_detail_{};
    std::atomic<bool> recording_{true}, valid_{true};
    std::uint64_t serial_{}, active_{}, started_{}, ended_{};
    std::size_t used_{}, phases_used_{}, hosts_used_{};
};

}  // namespace glimmer::interceptor::execution
