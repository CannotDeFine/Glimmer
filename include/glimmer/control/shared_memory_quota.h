#pragma once

#include "glimmer/control/quota_store.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace glimmer::control {

namespace detail {
struct SharedQuotaState;
}

struct SharedMemoryQuotaConfig {
    std::string tenant_id;
    DeviceId device = QuotaStore::default_device;
    core::MemoryBytes limit_bytes = 0;
};

enum class SharedMemoryQuotaError : std::uint8_t {
    kNone,
    kInvalidConfiguration,
    kNameTooLong,
    kOpenFailed,
    kResizeFailed,
    kMapFailed,
    kLockFailed,
    kInitializationFailed,
    kIncompatibleRegion,
    kDeviceLimitMismatch,
    kDeviceTableFull,
    kProcessTableFull,
    kProcessIdentityFailed,
    kUnknown,
};

class SharedMemoryQuota final : public QuotaStore {
   public:
    static constexpr std::size_t max_tenant_id_bytes = 96;

    SharedMemoryQuota(const SharedMemoryQuota&) = delete;
    SharedMemoryQuota& operator=(const SharedMemoryQuota&) = delete;

    SharedMemoryQuota(SharedMemoryQuota&& other) noexcept;
    SharedMemoryQuota& operator=(SharedMemoryQuota&& other) noexcept;

    ~SharedMemoryQuota() override;

    [[nodiscard]] static std::unique_ptr<SharedMemoryQuota> open(
        const SharedMemoryQuotaConfig& config, SharedMemoryQuotaError* error = nullptr) noexcept;

    // Administrative cleanup. It must only be used when no process is using the
    // tenant region.
    [[nodiscard]] static bool remove_region(std::string_view tenant_id) noexcept;

    using QuotaStore::get_memory_info;
    using QuotaStore::release;
    using QuotaStore::try_reserve;
    using QuotaStore::usage;

    [[nodiscard]] std::optional<MemoryReservation> try_reserve(
        DeviceId device, core::MemoryBytes memory_bytes) override;
    [[nodiscard]] bool release(DeviceId device, core::MemoryBytes memory_bytes) override;
    [[nodiscard]] MemoryInfo get_memory_info(DeviceId device,
                                             core::MemoryBytes physical_total_bytes,
                                             core::MemoryBytes physical_free_bytes) const override;
    [[nodiscard]] core::QuotaUsage usage(DeviceId device) const override;
    [[nodiscard]] bool is_healthy() const noexcept override;

   private:
    explicit SharedMemoryQuota(std::shared_ptr<detail::SharedQuotaState> state) noexcept;

    std::shared_ptr<detail::SharedQuotaState> state_;
};

}  // namespace glimmer::control
