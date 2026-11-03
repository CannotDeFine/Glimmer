#include "internal/allocation_registry.h"

#include "internal/diagnostics.h"

#include <functional>
#include <limits>
#include <new>
#include <unordered_set>
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
    vmm_records_.clear();
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
                                AllocationScope scope, AllocationProvenance provenance) {
    try {
        std::scoped_lock lock(mutex());
        if (is_accounting_degraded_.load(std::memory_order_relaxed) ||
            records_.contains(identity)) {
            return false;
        }
        return records_
            .emplace(identity, AllocationRecord{.memory_bytes = memory_bytes,
                                                .scope = scope,
                                                .provenance = provenance,
                                                .pending_stream = {}})
            .second;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

bool AllocationRegistry::has_live_independent_allocation(
    AllocationIdentity identity, core::MemoryBytes required_bytes) const noexcept {
    if (!identity.device_pointer || !identity.context || identity.device < 0 || !required_bytes ||
        identity.device_pointer > std::numeric_limits<CUdeviceptr>::max() - required_bytes)
        return false;
    try {
        std::scoped_lock lock(mutex());
        const auto found = records_.find(identity);
        return !is_accounting_degraded_.load(std::memory_order_relaxed) &&
               found != records_.end() && !found->second.is_releasing &&
               found->second.scope == AllocationScope::kContextBound &&
               found->second.provenance == AllocationProvenance::kIndependentDevice &&
               found->second.memory_bytes <=
                   std::numeric_limits<CUdeviceptr>::max() - identity.device_pointer &&
               required_bytes <= found->second.memory_bytes;
    } catch (...) {
        return false;
    }
}

bool AllocationRegistry::record_vmm(VmmAllocationIdentity identity,
                                    core::MemoryBytes memory_bytes) {
    try {
        std::scoped_lock lock(mutex());
        if (is_accounting_degraded_.load(std::memory_order_relaxed) || identity.handle == 0 ||
            identity.device < 0 || vmm_records_.contains(identity.handle)) {
            return false;
        }
        return vmm_records_
            .emplace(identity.handle, VmmAllocationRecord{.memory_bytes = memory_bytes,
                                                          .device = identity.device,
                                                          .reference_count = 1})
            .second;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

bool AllocationRegistry::retain_vmm_handle(CUmemGenericAllocationHandle handle) noexcept {
    try {
        std::scoped_lock lock(mutex());
        const auto record = vmm_records_.find(handle);
        if (record == vmm_records_.end() || record->second.is_releasing ||
            record->second.reference_count == std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        ++record->second.reference_count;
        return true;
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

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::VmmReleaseTicket>>
AllocationRegistry::begin_vmm_release(VmmAllocationIdentity identity) {
    try {
        std::scoped_lock lock(mutex());
        const auto record = vmm_records_.find(identity.handle);
        if (record == vmm_records_.end() || record->second.device != identity.device) {
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        if (record->second.is_releasing) {
            return {ReleaseStatus::kInProgress, std::nullopt};
        }

        record->second.is_releasing = true;
        return {ReleaseStatus::kStarted,
                VmmReleaseTicket{.identity = identity,
                                 .memory_bytes = record->second.memory_bytes,
                                 .is_last_reference = record->second.reference_count == 1}};
    } catch (...) {
        mark_accounting_degraded();
        return {ReleaseStatus::kUnknown, std::nullopt};
    }
}

std::pair<AllocationRegistry::ReleaseStatus, std::optional<AllocationRegistry::VmmReleaseTicket>>
AllocationRegistry::begin_vmm_release_by_handle(CUmemGenericAllocationHandle handle) {
    try {
        std::scoped_lock lock(mutex());
        const auto record = vmm_records_.find(handle);
        if (record == vmm_records_.end()) {
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        if (record->second.is_releasing) {
            return {ReleaseStatus::kInProgress, std::nullopt};
        }

        record->second.is_releasing = true;
        return {ReleaseStatus::kStarted,
                VmmReleaseTicket{.identity = VmmAllocationIdentity{.handle = handle,
                                                                   .device = record->second.device},
                                 .memory_bytes = record->second.memory_bytes,
                                 .is_last_reference = record->second.reference_count == 1}};
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

        if (!next_async_release_generation_) {
            mark_accounting_degraded();
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        record->second.async_release_generation = next_async_release_generation_++;

        record->second.is_releasing = true;
        record->second.is_async_release = true;
        record->second.is_async_release_submitted = false;
        record->second.is_async_release_stream_detached = false;
        record->second.pending_stream = stream;
        return {ReleaseStatus::kStarted,
                ReleaseTicket{.identity = identity,
                              .memory_bytes = record->second.memory_bytes,
                              .stream = stream,
                              .async_release_generation = record->second.async_release_generation}};
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

        if (!next_async_release_generation_) {
            mark_accounting_degraded();
            return {ReleaseStatus::kUnknown, std::nullopt};
        }
        candidate->second.async_release_generation = next_async_release_generation_++;

        candidate->second.is_releasing = true;
        candidate->second.is_async_release = true;
        candidate->second.is_async_release_submitted = false;
        candidate->second.is_async_release_stream_detached = false;
        candidate->second.pending_stream = stream;
        return {
            ReleaseStatus::kStarted,
            ReleaseTicket{.identity = candidate->first,
                          .memory_bytes = candidate->second.memory_bytes,
                          .stream = stream,
                          .async_release_generation = candidate->second.async_release_generation}};
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

bool AllocationRegistry::complete_vmm_release(const VmmReleaseTicket& ticket) noexcept {
    try {
        std::scoped_lock lock(mutex());
        const auto record = vmm_records_.find(ticket.identity.handle);
        if (record == vmm_records_.end() || record->second.device != ticket.identity.device ||
            !record->second.is_releasing || record->second.memory_bytes != ticket.memory_bytes) {
            return false;
        }
        if (ticket.is_last_reference) {
            if (record->second.reference_count != 1) {
                return false;
            }
            vmm_records_.erase(record);
        } else {
            if (record->second.reference_count <= 1) {
                return false;
            }
            --record->second.reference_count;
            record->second.is_releasing = false;
        }
        return true;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

bool AllocationRegistry::matches_async_release(const AllocationRecord& record,
                                               const ReleaseTicket& ticket) noexcept {
    return record.is_releasing && record.is_async_release && ticket.async_release_generation != 0 &&
           record.async_release_generation == ticket.async_release_generation &&
           record.memory_bytes == ticket.memory_bytes &&
           same_async_stream(record.pending_stream, ticket.stream);
}

bool AllocationRegistry::can_complete_async_release(const AllocationRecord& record) noexcept {
    return record.is_releasing && record.is_async_release && record.is_async_release_submitted &&
           (!record.is_cooperative_release || record.is_cooperative_release_complete);
}

bool AllocationRegistry::prepare_cooperative_release(const ReleaseTicket& ticket) noexcept {
    try {
        std::scoped_lock lock(mutex());
        const auto record = records_.find(ticket.identity);
        if (record == records_.end() || !matches_async_release(record->second, ticket) ||
            record->second.is_async_release_submitted || record->second.is_cooperative_release) {
            return false;
        }
        record->second.is_cooperative_release = true;
        return true;
    } catch (...) {
        mark_accounting_degraded();
        return false;
    }
}

bool AllocationRegistry::confirm_cooperative_release(const ReleaseTicket& ticket) noexcept {
    try {
        std::scoped_lock lock(mutex());
        const auto record = records_.find(ticket.identity);
        if (record == records_.end() || !matches_async_release(record->second, ticket) ||
            !record->second.is_cooperative_release ||
            record->second.is_cooperative_release_complete) {
            return false;
        }
        record->second.is_cooperative_release_complete = true;
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
        if (record == records_.end() || !matches_async_release(record->second, ticket)) {
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
            record->second.memory_bytes != ticket.memory_bytes ||
            record->second.async_release_generation != ticket.async_release_generation) {
            return;
        }
        record->second.is_releasing = false;
        record->second.is_async_release = false;
        record->second.is_async_release_submitted = false;
        record->second.is_async_release_stream_detached = false;
        record->second.is_cooperative_release = false;
        record->second.is_cooperative_release_complete = false;
        record->second.async_release_generation = 0;
        record->second.pending_stream = {};
    } catch (...) {
        mark_accounting_degraded();
    }
}

void AllocationRegistry::cancel_vmm_release(const VmmReleaseTicket& ticket) noexcept {
    try {
        std::scoped_lock lock(mutex());
        const auto record = vmm_records_.find(ticket.identity.handle);
        if (record == vmm_records_.end() || record->second.device != ticket.identity.device ||
            !record->second.is_releasing || record->second.memory_bytes != ticket.memory_bytes) {
            return;
        }
        record->second.is_releasing = false;
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
            // Retain inert CPU release identity for generation-bound physical
            // confirmation and context fences; never query this retired stream.
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
            if (!can_complete_async_release(record) || record.is_async_release_stream_detached ||
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
            if (can_complete_async_release(record->second) &&
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

std::optional<std::vector<AllocationRegistry::ReleaseTicket>>
AllocationRegistry::submitted_async_releases(bool should_include_detached_streams) const noexcept {
    try {
        std::scoped_lock lock(mutex());
        std::vector<ReleaseTicket> submitted;
        for (const auto& [identity, record] : records_) {
            if (record.is_releasing && record.is_async_release &&
                record.is_async_release_submitted &&
                (should_include_detached_streams || !record.is_async_release_stream_detached)) {
                submitted.push_back({identity, record.memory_bytes, record.pending_stream,
                                     record.async_release_generation});
            }
        }
        return submitted;
    } catch (...) {
        return std::nullopt;  // No original completion has issued; nothing was retired.
    }
}

std::optional<std::vector<AllocationRegistry::DeviceReleaseSummary>>
AllocationRegistry::complete_submitted_async_releases(std::span<const ReleaseTicket> submitted,
                                                      const AsyncStreamIdentity& stream) noexcept {
    return complete_submitted_releases(submitted, &stream, nullptr);
}
std::optional<std::vector<AllocationRegistry::DeviceReleaseSummary>>
AllocationRegistry::complete_submitted_context_releases(std::span<const ReleaseTicket> submitted,
                                                        CUcontext context) noexcept {
    return context ? complete_submitted_releases(submitted, nullptr, context) : std::nullopt;
}
std::optional<std::vector<AllocationRegistry::DeviceReleaseSummary>>
AllocationRegistry::complete_submitted_releases(std::span<const ReleaseTicket> submitted,
                                                const AsyncStreamIdentity* stream,
                                                CUcontext context) noexcept {
    try {
        std::scoped_lock lock(mutex());
        std::vector<DeviceReleaseSummary> summaries;
        std::unordered_set<AllocationIdentity, AllocationIdentityHash> retired;
        for (const auto& ticket : submitted) {
            const auto found = records_.find(ticket.identity);
            if (found == records_.end() || !can_complete_async_release(found->second) ||
                (stream ? found->second.is_async_release_stream_detached ||
                              !same_async_stream(found->second.pending_stream, *stream)
                        : found->second.pending_stream.context != context) ||
                !matches_async_release(found->second, ticket)) {
                continue;
            }
            if (!retired.insert(ticket.identity).second) {
                mark_accounting_degraded();
                return std::nullopt;  // Duplicate proof must not count the allocation twice.
            }
            if (!append_device_release_summary(summaries, ticket.identity.device,
                                               found->second.memory_bytes)) {
                return std::nullopt;
            }
        }
        std::erase_if(records_, [&](const auto& entry) { return retired.contains(entry.first); });
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
            if (!can_complete_async_release(record->second) || record->first.context != context ||
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
            if (!can_complete_async_release(record) || identity.context != context) {
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
            if (can_complete_async_release(record->second) && record->first.context == context) {
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
