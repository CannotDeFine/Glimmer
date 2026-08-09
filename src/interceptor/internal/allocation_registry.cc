#include "internal/allocation_registry.h"

#include "internal/diagnostics.h"

#include <functional>
#include <limits>
#include <new>
#include <unistd.h>

namespace glimmer::interceptor {

namespace {

[[nodiscard]] bool same_async_stream(const AsyncStreamIdentity& left,
                                     const AsyncStreamIdentity& right) noexcept {
    if (left.stream != right.stream || left.context != right.context) {
        return false;
    }
    if (left.stream != nullptr) {
        return true;
    }
    if (left.per_thread_default_stream != right.per_thread_default_stream) {
        return false;
    }
    return !left.per_thread_default_stream || left.thread_id == right.thread_id;
}

}  // namespace

AllocationRegistry::AllocationRegistry()
    : mutex_(std::make_unique<std::mutex>()), process_id_(::getpid()) {}

AllocationRegistry::~AllocationRegistry() {
    if (process_id_ != ::getpid()) {
        // Avoid destroying a mutex inherited while it may have been held by a
        // parent thread that no longer exists in this child.
        const bool inherited_mutex_present = mutex_.release() != nullptr;
        if (!inherited_mutex_present) {
            is_accounting_degraded_.store(true, std::memory_order_relaxed);
        }
    }
}

void AllocationRegistry::reset_after_fork_if_needed() const noexcept {
    const pid_t current_process_id = ::getpid();
    if (process_id_ == current_process_id) {
        return;
    }

    // The child may inherit a mutex held by a thread that no longer exists.
    // Do not destroy that inherited object; install a fresh process-local lock
    // and discard metadata that belongs to the parent process.
    const bool inherited_mutex_present = mutex_.release() != nullptr;
    if (!inherited_mutex_present) {
        is_accounting_degraded_.store(true, std::memory_order_relaxed);
    }
    mutex_.reset(new (std::nothrow) std::mutex());
    if (mutex_ == nullptr) {
        is_accounting_degraded_.store(true, std::memory_order_relaxed);
        return;
    }
    records_.clear();
    is_accounting_degraded_.store(false, std::memory_order_relaxed);
    process_id_ = current_process_id;
}

std::mutex& AllocationRegistry::mutex() const noexcept {
    reset_after_fork_if_needed();
    if (mutex_ == nullptr) {
        static std::mutex fallback_mutex;
        return fallback_mutex;
    }
    return *mutex_;
}

std::size_t AllocationIdentityHash::operator()(const AllocationIdentity& identity) const noexcept {
    const std::size_t pointer_hash = std::hash<CUdeviceptr>{}(identity.device_pointer);
    const std::size_t context_hash = std::hash<const void*>{}(identity.context);
    const std::size_t device_hash = std::hash<CUdevice>{}(identity.device);
    return pointer_hash ^
           (context_hash + static_cast<std::size_t>(0x9e3779b9U) + (pointer_hash << 6U) +
            (pointer_hash >> 2U)) ^
           (device_hash + static_cast<std::size_t>(0x9e3779b9U) + (context_hash << 6U) +
            (context_hash >> 2U));
}

bool AllocationRegistry::record(AllocationIdentity identity, core::MemoryBytes memory_bytes,
                                AllocationScope scope) {
    try {
        std::scoped_lock lock(mutex());
        if (is_accounting_degraded_.load(std::memory_order_relaxed) ||
            records_.contains(identity)) {
            return false;
        }
        return records_
            .emplace(identity,
                     AllocationRecord{
                         .memory_bytes = memory_bytes, .scope = scope, .pending_stream = {}})
            .second;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_release(AllocationIdentity identity) {
    try {
        std::scoped_lock lock(mutex());
        const auto record = records_.find(identity);
        if (record == records_.end()) {
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        if (record->second.is_releasing) {
            return {ReleaseStatus::kInProgress, std::nullopt};
        }

        record->second.is_releasing = true;
        return {ReleaseStatus::kStarted, ReleaseTicket{.identity = identity,
                                                       .memory_bytes = record->second.memory_bytes,
                                                       .stream = {}}};
    } catch (...) {
        mark_accounting_degraded();
        return {ReleaseStatus::kUnknown, std::nullopt};
    }
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_release_by_pointer(CUdeviceptr device_pointer, CUdevice device) {
    try {
        std::scoped_lock lock(mutex());
        auto candidate = records_.end();
        for (auto record = records_.begin(); record != records_.end(); ++record) {
            if (record->first.device_pointer != device_pointer || record->first.device != device) {
                continue;
            }
            if (candidate != records_.end()) {
                return {ReleaseStatus::kUnknown, std::nullopt};
            }
            candidate = record;
        }

        if (candidate == records_.end()) {
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        if (candidate->second.is_releasing) {
            return {ReleaseStatus::kInProgress, std::nullopt};
        }

        candidate->second.is_releasing = true;
        return {ReleaseStatus::kStarted,
                ReleaseTicket{.identity = candidate->first,
                              .memory_bytes = candidate->second.memory_bytes,
                              .stream = {}}};
    } catch (...) {
        mark_accounting_degraded();
        return {ReleaseStatus::kUnknown, std::nullopt};
    }
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_async_release(AllocationIdentity identity, AsyncStreamIdentity stream) {
    try {
        std::scoped_lock lock(mutex());
        const auto record = records_.find(identity);
        if (record == records_.end()) {
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        if (record->second.is_releasing) {
            return {ReleaseStatus::kInProgress, std::nullopt};
        }

        record->second.is_releasing = true;
        record->second.is_async_release = true;
        record->second.is_async_release_submitted = false;
        record->second.is_async_release_stream_detached = false;
        record->second.pending_stream = stream;
        return {ReleaseStatus::kStarted, ReleaseTicket{.identity = identity,
                                                       .memory_bytes = record->second.memory_bytes,
                                                       .stream = stream}};
    } catch (...) {
        mark_accounting_degraded();
        return {ReleaseStatus::kUnknown, std::nullopt};
    }
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_async_release_by_pointer(CUdeviceptr device_pointer, CUdevice device,
                                                   AsyncStreamIdentity stream) {
    try {
        std::scoped_lock lock(mutex());
        auto candidate = records_.end();
        for (auto record = records_.begin(); record != records_.end(); ++record) {
            if (record->first.device_pointer != device_pointer || record->first.device != device) {
                continue;
            }
            if (candidate != records_.end()) {
                return {ReleaseStatus::kUnknown, std::nullopt};
            }
            candidate = record;
        }

        if (candidate == records_.end()) {
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        if (candidate->second.is_releasing) {
            return {ReleaseStatus::kInProgress, std::nullopt};
        }

        candidate->second.is_releasing = true;
        candidate->second.is_async_release = true;
        candidate->second.is_async_release_submitted = false;
        candidate->second.is_async_release_stream_detached = false;
        candidate->second.pending_stream = stream;
        return {ReleaseStatus::kStarted,
                ReleaseTicket{.identity = candidate->first,
                              .memory_bytes = candidate->second.memory_bytes,
                              .stream = stream}};
    } catch (...) {
        mark_accounting_degraded();
        return {ReleaseStatus::kUnknown, std::nullopt};
    }
}

bool AllocationRegistry::complete_release(const ReleaseTicket& ticket) {
    try {
        std::scoped_lock lock(mutex());
        const auto record = records_.find(ticket.identity);
        if (record == records_.end() || !record->second.is_releasing ||
            record->second.is_async_release || record->second.memory_bytes != ticket.memory_bytes) {
            return false;
        }
        records_.erase(record);
        return true;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

bool AllocationRegistry::commit_async_release(const ReleaseTicket& ticket) noexcept {
    try {
        std::scoped_lock lock(mutex());
        const auto record = records_.find(ticket.identity);
        if (record == records_.end() || !record->second.is_releasing ||
            !record->second.is_async_release ||
            record->second.memory_bytes != ticket.memory_bytes ||
            !same_async_stream(record->second.pending_stream, ticket.stream)) {
            return false;
        }
        record->second.is_async_release_submitted = true;
        return true;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

void AllocationRegistry::cancel_release(const ReleaseTicket& ticket) noexcept {
    try {
        std::scoped_lock lock(mutex());
        const auto record = records_.find(ticket.identity);
        if (record == records_.end() || !record->second.is_releasing ||
            record->second.memory_bytes != ticket.memory_bytes) {
            return;
        }
        record->second.is_releasing = false;
        record->second.is_async_release = false;
        record->second.is_async_release_submitted = false;
        record->second.is_async_release_stream_detached = false;
        record->second.pending_stream = {};
    } catch (...) {
        mark_accounting_degraded();
    }
}

bool AllocationRegistry::detach_async_releases_for_stream(
    const AsyncStreamIdentity& stream) noexcept {
    try {
        std::scoped_lock lock(mutex());
        for (auto& [identity, record] : records_) {
            static_cast<void>(identity);
            if (!record.is_releasing || !record.is_async_release ||
                !record.is_async_release_submitted || record.is_async_release_stream_detached ||
                !same_async_stream(record.pending_stream, stream)) {
                continue;
            }
            record.is_async_release_stream_detached = true;
            record.pending_stream = {};
        }
        return true;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

namespace {

[[nodiscard]] bool append_device_release_summary(
    std::vector<AllocationRegistry::DeviceReleaseSummary>& summaries, CUdevice device,
    core::MemoryBytes memory_bytes) noexcept {
    for (auto& summary : summaries) {
        if (summary.device != device) {
            continue;
        }
        if (summary.memory_bytes > std::numeric_limits<core::MemoryBytes>::max() - memory_bytes) {
            return false;
        }
        summary.memory_bytes += memory_bytes;
        return true;
    }

    try {
        summaries.push_back(AllocationRegistry::DeviceReleaseSummary{.device = device,
                                                                     .memory_bytes = memory_bytes});
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

std::optional<std::vector<AllocationRegistry::DeviceReleaseSummary>>
AllocationRegistry::complete_async_releases_for_stream_by_device(
    const AsyncStreamIdentity& stream) noexcept {
    try {
        std::scoped_lock lock(mutex());
        std::vector<DeviceReleaseSummary> summaries;
        for (const auto& [identity, record] : records_) {
            if (!record.is_releasing || !record.is_async_release ||
                !record.is_async_release_submitted || record.is_async_release_stream_detached ||
                !same_async_stream(record.pending_stream, stream)) {
                continue;
            }
            if (!append_device_release_summary(summaries, identity.device, record.memory_bytes)) {
                return std::nullopt;
            }
        }

        if (summaries.empty()) {
            return summaries;
        }

        for (auto record = records_.begin(); record != records_.end();) {
            if (record->second.is_releasing && record->second.is_async_release &&
                record->second.is_async_release_submitted &&
                !record->second.is_async_release_stream_detached &&
                same_async_stream(record->second.pending_stream, stream)) {
                record = records_.erase(record);
            } else {
                ++record;
            }
        }
        return summaries;
    } catch (...) {
        mark_accounting_degraded();
        return std::nullopt;
    }
}

core::MemoryBytes AllocationRegistry::complete_async_releases_for_context(
    CUcontext context, CUdevice device) noexcept {
    try {
        std::scoped_lock lock(mutex());
        core::MemoryBytes released_bytes = 0;
        for (auto record = records_.begin(); record != records_.end();) {
            if (!record->second.is_releasing || !record->second.is_async_release ||
                !record->second.is_async_release_submitted || record->first.context != context ||
                record->first.device != device) {
                ++record;
                continue;
            }

            const core::MemoryBytes allocation_bytes = record->second.memory_bytes;
            if (released_bytes > std::numeric_limits<core::MemoryBytes>::max() - allocation_bytes) {
                released_bytes = std::numeric_limits<core::MemoryBytes>::max();
            } else {
                released_bytes += allocation_bytes;
            }
            record = records_.erase(record);
        }
        return released_bytes;
    } catch (...) {
        mark_accounting_degraded();
        return 0;
    }
}

std::optional<std::vector<AllocationRegistry::DeviceReleaseSummary>>
AllocationRegistry::complete_async_releases_for_context_by_device(CUcontext context) noexcept {
    try {
        std::scoped_lock lock(mutex());
        std::vector<DeviceReleaseSummary> summaries;
        for (const auto& [identity, record] : records_) {
            if (!record.is_releasing || !record.is_async_release ||
                !record.is_async_release_submitted || identity.context != context) {
                continue;
            }
            if (!append_device_release_summary(summaries, identity.device, record.memory_bytes)) {
                return std::nullopt;
            }
        }

        if (summaries.empty()) {
            return summaries;
        }

        for (auto record = records_.begin(); record != records_.end();) {
            if (record->second.is_releasing && record->second.is_async_release &&
                record->second.is_async_release_submitted && record->first.context == context) {
                record = records_.erase(record);
            } else {
                ++record;
            }
        }
        return summaries;
    } catch (...) {
        mark_accounting_degraded();
        return std::nullopt;
    }
}

std::optional<CUdevice> AllocationRegistry::device_for_context(CUcontext context) const noexcept {
    try {
        std::scoped_lock lock(mutex());
        std::optional<CUdevice> device;
        for (const auto& [identity, record] : records_) {
            if (identity.context != context) {
                continue;
            }
            if (device.has_value() && *device != identity.device) {
                return std::nullopt;
            }
            device = identity.device;
        }
        return device;
    } catch (...) {
        mark_accounting_degraded();
        return std::nullopt;
    }
}

std::optional<std::vector<AllocationRegistry::DeviceReleaseSummary>>
AllocationRegistry::erase_context_by_device(CUcontext context) noexcept {
    try {
        std::scoped_lock lock(mutex());
        std::vector<DeviceReleaseSummary> summaries;
        for (const auto& [identity, record] : records_) {
            if (identity.context != context || record.scope != AllocationScope::kContextBound) {
                continue;
            }
            if (!append_device_release_summary(summaries, identity.device, record.memory_bytes)) {
                return std::nullopt;
            }
        }

        for (auto record = records_.begin(); record != records_.end();) {
            if (record->first.context == context &&
                record->second.scope == AllocationScope::kContextBound) {
                record = records_.erase(record);
            } else {
                ++record;
            }
        }
        return summaries;
    } catch (...) {
        mark_accounting_degraded();
        return std::nullopt;
    }
}

core::MemoryBytes AllocationRegistry::erase_context(CUcontext context) noexcept {
    try {
        std::scoped_lock lock(mutex());
        core::MemoryBytes released_bytes = 0;
        for (auto record = records_.begin(); record != records_.end();) {
            if (record->first.context != context ||
                record->second.scope != AllocationScope::kContextBound) {
                ++record;
                continue;
            }

            const core::MemoryBytes allocation_bytes = record->second.memory_bytes;
            if (released_bytes > std::numeric_limits<core::MemoryBytes>::max() - allocation_bytes) {
                released_bytes = std::numeric_limits<core::MemoryBytes>::max();
            } else {
                released_bytes += allocation_bytes;
            }
            record = records_.erase(record);
        }
        return released_bytes;
    } catch (...) {
        mark_accounting_degraded();
        return 0;
    }
}

core::MemoryBytes AllocationRegistry::erase_context(CUcontext context, CUdevice device) noexcept {
    try {
        std::scoped_lock lock(mutex());
        core::MemoryBytes released_bytes = 0;
        for (auto record = records_.begin(); record != records_.end();) {
            if (record->first.context != context || record->first.device != device ||
                record->second.scope != AllocationScope::kContextBound) {
                ++record;
                continue;
            }

            const core::MemoryBytes allocation_bytes = record->second.memory_bytes;
            if (released_bytes > std::numeric_limits<core::MemoryBytes>::max() - allocation_bytes) {
                released_bytes = std::numeric_limits<core::MemoryBytes>::max();
            } else {
                released_bytes += allocation_bytes;
            }
            record = records_.erase(record);
        }
        return released_bytes;
    } catch (...) {
        mark_accounting_degraded();
        return 0;
    }
}

bool AllocationRegistry::is_accounting_degraded() const noexcept {
    return is_accounting_degraded_.load(std::memory_order_relaxed);
}

void AllocationRegistry::mark_accounting_degraded() const noexcept {
    bool expected = false;
    const bool should_report = is_accounting_degraded_.compare_exchange_strong(
        expected, true, std::memory_order_relaxed, std::memory_order_relaxed);
    if (should_report) {
        report_diagnostic("[glimmer] CUDA accounting entered degraded mode\n");
    }
}

}  // namespace glimmer::interceptor
