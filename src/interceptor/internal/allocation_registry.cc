#include "internal/allocation_registry.h"

#include "internal/diagnostics.h"

#include <functional>
#include <limits>

namespace glimmer::interceptor {

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
    std::scoped_lock lock(mutex_);
    if (is_accounting_degraded_ || records_.contains(identity)) {
        return false;
    }

    try {
        return records_
            .emplace(identity, AllocationRecord{.memory_bytes = memory_bytes, .scope = scope})
            .second;
    } catch (...) {
        return false;
    }
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_release(AllocationIdentity identity) {
    std::scoped_lock lock(mutex_);
    const auto record = records_.find(identity);
    if (record == records_.end()) {
        return {ReleaseStatus::kUnknown, std::nullopt};
    }
    if (record->second.is_releasing) {
        return {ReleaseStatus::kInProgress, std::nullopt};
    }

    record->second.is_releasing = true;
    return {ReleaseStatus::kStarted,
            ReleaseTicket{.identity = identity, .memory_bytes = record->second.memory_bytes}};
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_release_by_pointer(CUdeviceptr device_pointer, CUdevice device) {
    std::scoped_lock lock(mutex_);
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
    return {ReleaseStatus::kStarted, ReleaseTicket{.identity = candidate->first,
                                                   .memory_bytes = candidate->second.memory_bytes}};
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_async_release(AllocationIdentity identity, CUstream stream) {
    std::scoped_lock lock(mutex_);
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
    record->second.pending_stream = stream;
    return {ReleaseStatus::kStarted, ReleaseTicket{.identity = identity,
                                                   .memory_bytes = record->second.memory_bytes,
                                                   .stream = stream}};
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::begin_async_release_by_pointer(CUdeviceptr device_pointer, CUdevice device,
                                                   CUstream stream) {
    std::scoped_lock lock(mutex_);
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
    candidate->second.pending_stream = stream;
    return {ReleaseStatus::kStarted, ReleaseTicket{.identity = candidate->first,
                                                   .memory_bytes = candidate->second.memory_bytes,
                                                   .stream = stream}};
}

bool AllocationRegistry::complete_release(const ReleaseTicket& ticket) {
    {
        std::scoped_lock lock(mutex_);
        const auto record = records_.find(ticket.identity);
        if (record == records_.end() || !record->second.is_releasing ||
            record->second.is_async_release || record->second.memory_bytes != ticket.memory_bytes) {
            return false;
        }
        records_.erase(record);
    }
    return true;
}

bool AllocationRegistry::commit_async_release(const ReleaseTicket& ticket) noexcept {
    std::scoped_lock lock(mutex_);
    const auto record = records_.find(ticket.identity);
    if (record == records_.end() || !record->second.is_releasing ||
        !record->second.is_async_release || record->second.memory_bytes != ticket.memory_bytes ||
        record->second.pending_stream != ticket.stream) {
        return false;
    }
    record->second.is_async_release_submitted = true;
    return true;
}

void AllocationRegistry::cancel_release(const ReleaseTicket& ticket) noexcept {
    {
        std::scoped_lock lock(mutex_);
        const auto record = records_.find(ticket.identity);
        if (record == records_.end() || !record->second.is_releasing ||
            record->second.memory_bytes != ticket.memory_bytes) {
            return;
        }
        record->second.is_releasing = false;
        record->second.is_async_release = false;
        record->second.is_async_release_submitted = false;
        record->second.pending_stream = nullptr;
    }
}

core::MemoryBytes AllocationRegistry::complete_async_releases_for_stream(CUstream stream) noexcept {
    std::scoped_lock lock(mutex_);
    core::MemoryBytes released_bytes = 0;
    for (auto record = records_.begin(); record != records_.end();) {
        if (!record->second.is_releasing || !record->second.is_async_release ||
            !record->second.is_async_release_submitted || record->second.pending_stream != stream) {
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
}

core::MemoryBytes AllocationRegistry::complete_async_releases_for_context(
    CUcontext context, CUdevice device) noexcept {
    std::scoped_lock lock(mutex_);
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
}

core::MemoryBytes AllocationRegistry::erase_context(CUcontext context) noexcept {
    std::scoped_lock lock(mutex_);
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
}

bool AllocationRegistry::is_accounting_degraded() const noexcept {
    std::scoped_lock lock(mutex_);
    return is_accounting_degraded_;
}

void AllocationRegistry::mark_accounting_degraded() noexcept {
    bool should_report = false;
    {
        std::scoped_lock lock(mutex_);
        should_report = !is_accounting_degraded_;
        is_accounting_degraded_ = true;
    }
    if (should_report) {
        report_diagnostic("[glimmer] CUDA accounting entered degraded mode\n");
    }
}

}  // namespace glimmer::interceptor
