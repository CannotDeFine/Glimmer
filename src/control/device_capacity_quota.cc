#include "glimmer/control/device_capacity_quota.h"

#include <algorithm>
#include <memory>

namespace glimmer::control {

// NOLINTBEGIN(bugprone-easily-swappable-parameters)

DeviceCapacityQuota::DeviceCapacityQuota(DeviceMemoryCapacityResolver resolver,
                                         void* context) noexcept
    : resolver_(resolver), context_(context), healthy_(resolver != nullptr) {}

bool DeviceCapacityQuota::resolve(DeviceId device, DeviceMemoryCapacity* capacity) const noexcept {
    if (device < 0 || capacity == nullptr || resolver_ == nullptr) {
        healthy_.store(false, std::memory_order_release);
        return false;
    }

    DeviceMemoryCapacity resolved{};
    if (!resolver_(context_, device, &resolved) || resolved.total_bytes == 0) {
        healthy_.store(false, std::memory_order_release);
        return false;
    }
    resolved.free_bytes = std::min(resolved.free_bytes, resolved.total_bytes);
    *capacity = resolved;
    return true;
}

ProcessMemoryQuota* DeviceCapacityQuota::ensure_device_quota_locked(
    DeviceId device, core::MemoryBytes total_bytes) const {
    const auto existing = device_quotas_.find(device);
    if (existing != device_quotas_.end()) {
        return existing->second.get();
    }

    auto quota = std::make_unique<ProcessMemoryQuota>(total_bytes);
    ProcessMemoryQuota* quota_pointer = quota.get();
    device_quotas_.emplace(device, std::move(quota));
    effective_limits_.emplace(device, total_bytes);
    return quota_pointer;
}

std::optional<MemoryReservation> DeviceCapacityQuota::try_reserve(DeviceId device,
                                                                  core::MemoryBytes memory_bytes) {
    DeviceMemoryCapacity capacity{};
    if (!resolve(device, &capacity)) {
        return std::nullopt;
    }

    try {
        std::scoped_lock lock(mutex_);
        ProcessMemoryQuota* quota = ensure_device_quota_locked(device, capacity.total_bytes);
        if (quota == nullptr) {
            healthy_.store(false, std::memory_order_release);
            return std::nullopt;
        }

        auto limit = effective_limits_.find(device);
        if (limit == effective_limits_.end()) {
            healthy_.store(false, std::memory_order_release);
            return std::nullopt;
        }
        limit->second = std::min(limit->second, capacity.total_bytes);
        const core::QuotaUsage usage = quota->usage(device);
        const core::MemoryBytes ledger_available = usage.available_bytes();
        const core::MemoryBytes physical_available =
            limit->second > usage.used_bytes() ? limit->second - usage.used_bytes() : 0;
        if (memory_bytes > capacity.free_bytes || memory_bytes > physical_available ||
            memory_bytes > ledger_available) {
            return std::nullopt;
        }
        return quota->try_reserve(device, memory_bytes);
    } catch (...) {
        healthy_.store(false, std::memory_order_release);
        return std::nullopt;
    }
}

bool DeviceCapacityQuota::release(DeviceId device, core::MemoryBytes memory_bytes) {
    if (device < 0) {
        return false;
    }

    ProcessMemoryQuota* quota = nullptr;
    try {
        std::scoped_lock lock(mutex_);
        const auto existing = device_quotas_.find(device);
        if (existing == device_quotas_.end()) {
            return false;
        }
        quota = existing->second.get();
    } catch (...) {
        healthy_.store(false, std::memory_order_release);
        return false;
    }
    return quota->release(device, memory_bytes);
}

MemoryInfo DeviceCapacityQuota::get_memory_info(DeviceId device,
                                                core::MemoryBytes physical_total_bytes,
                                                core::MemoryBytes physical_free_bytes) const {
    if (device < 0 || !healthy_.load(std::memory_order_acquire)) {
        return MemoryInfo{.total_bytes = 0, .free_bytes = 0};
    }

    // The caller already performed the physical query. This is important for
    // NVML-only processes, which may not have initialized the CUDA Driver.
    // Allocation admission and device-total queries still resolve capacity
    // through the Driver callback.
    if (physical_total_bytes == 0) {
        DeviceMemoryCapacity capacity{};
        if (!resolve(device, &capacity)) {
            return MemoryInfo{.total_bytes = 0, .free_bytes = 0};
        }
        physical_total_bytes = capacity.total_bytes;
        physical_free_bytes = capacity.free_bytes;
    }

    try {
        std::scoped_lock lock(mutex_);
        ProcessMemoryQuota* quota = ensure_device_quota_locked(device, physical_total_bytes);
        if (quota == nullptr) {
            healthy_.store(false, std::memory_order_release);
            return MemoryInfo{.total_bytes = 0, .free_bytes = 0};
        }
        auto limit = effective_limits_.find(device);
        if (limit == effective_limits_.end()) {
            healthy_.store(false, std::memory_order_release);
            return MemoryInfo{.total_bytes = 0, .free_bytes = 0};
        }
        limit->second = std::min(limit->second, physical_total_bytes);
        const core::MemoryBytes visible_total = limit->second;
        return quota->get_memory_info(device, visible_total,
                                      std::min(physical_free_bytes, visible_total));
    } catch (...) {
        healthy_.store(false, std::memory_order_release);
        return MemoryInfo{.total_bytes = 0, .free_bytes = 0};
    }
}

core::QuotaUsage DeviceCapacityQuota::usage(DeviceId device) const {
    DeviceMemoryCapacity capacity{};
    if (!resolve(device, &capacity)) {
        return core::QuotaUsage{};
    }
    try {
        std::scoped_lock lock(mutex_);
        ProcessMemoryQuota* quota = ensure_device_quota_locked(device, capacity.total_bytes);
        auto limit = effective_limits_.find(device);
        if (quota == nullptr || limit == effective_limits_.end()) {
            healthy_.store(false, std::memory_order_release);
            return core::QuotaUsage{};
        }
        limit->second = std::min(limit->second, capacity.total_bytes);
        core::QuotaUsage usage = quota->usage(device);
        usage.limit_bytes = limit->second;
        return usage;
    } catch (...) {
        healthy_.store(false, std::memory_order_release);
        return core::QuotaUsage{};
    }
}

bool DeviceCapacityQuota::is_healthy() const noexcept {
    return healthy_.load(std::memory_order_acquire);
}

// NOLINTEND(bugprone-easily-swappable-parameters)

}  // namespace glimmer::control
