#pragma once

#include "glimmer/core/quota_ledger.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace glimmer::control {

using DeviceId = std::int32_t;

struct MemoryInfo {
    core::MemoryBytes total_bytes;
    core::MemoryBytes free_bytes;
};

namespace detail {

class MemoryReservationState {
   public:
    virtual ~MemoryReservationState() = default;

    [[nodiscard]] virtual bool commit() noexcept = 0;
    virtual void cancel() noexcept = 0;
    // Retain the accounting when cleanup cannot prove that an external
    // allocation has been released. This is deliberately conservative: the
    // reservation remains charged until the owning process is recovered.
    virtual void abandon() noexcept = 0;
    [[nodiscard]] virtual bool is_active() const noexcept = 0;
    [[nodiscard]] virtual core::MemoryBytes memory_bytes() const noexcept = 0;
};

}  // namespace detail

class ProcessMemoryQuota;
class SharedMemoryQuota;
class CompositeQuota;
class DeviceCapacityQuota;

class MemoryReservation {
   public:
    MemoryReservation(const MemoryReservation&) = delete;
    MemoryReservation& operator=(const MemoryReservation&) = delete;

    MemoryReservation(MemoryReservation&& other) noexcept;
    MemoryReservation& operator=(MemoryReservation&& other) noexcept;

    ~MemoryReservation();

    [[nodiscard]] bool commit() noexcept;
    void cancel() noexcept;
    // Retain the reservation when external cleanup cannot be proven complete.
    // The quota remains charged until the owning process is recovered.
    void abandon() noexcept;

    [[nodiscard]] bool is_active() const noexcept;
    [[nodiscard]] core::MemoryBytes memory_bytes() const noexcept;

   private:
    friend class ProcessMemoryQuota;
    friend class SharedMemoryQuota;
    friend class CompositeQuota;
    friend class DeviceCapacityQuota;

    explicit MemoryReservation(std::unique_ptr<detail::MemoryReservationState> state) noexcept;

    std::unique_ptr<detail::MemoryReservationState> state_;
};

class QuotaStore {
   public:
    static constexpr DeviceId default_device = 0;

    virtual ~QuotaStore() = default;

    [[nodiscard]] virtual std::optional<MemoryReservation> try_reserve(
        DeviceId device, core::MemoryBytes memory_bytes) = 0;
    [[nodiscard]] virtual bool release(DeviceId device, core::MemoryBytes memory_bytes) = 0;
    [[nodiscard]] virtual MemoryInfo get_memory_info(
        DeviceId device, core::MemoryBytes physical_total_bytes,
        core::MemoryBytes physical_free_bytes) const = 0;
    [[nodiscard]] virtual core::QuotaUsage usage(DeviceId device) const = 0;
    // Interceptor queries fail closed when the control plane is unhealthy.
    [[nodiscard]] virtual bool is_healthy() const noexcept = 0;

    [[nodiscard]] std::optional<MemoryReservation> try_reserve(core::MemoryBytes memory_bytes) {
        return try_reserve(default_device, memory_bytes);
    }

    [[nodiscard]] bool release(core::MemoryBytes memory_bytes) {
        return release(default_device, memory_bytes);
    }

    [[nodiscard]] MemoryInfo get_memory_info(core::MemoryBytes physical_total_bytes,
                                             core::MemoryBytes physical_free_bytes) const {
        return get_memory_info(default_device, physical_total_bytes, physical_free_bytes);
    }

    [[nodiscard]] core::QuotaUsage usage() const {
        return usage(default_device);
    }
};

}  // namespace glimmer::control
