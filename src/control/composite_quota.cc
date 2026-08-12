#include "glimmer/control/composite_quota.h"

#include <algorithm>
#include <utility>

namespace glimmer::control {

namespace {

struct CompositeReservationArguments {
    DeviceId device;
    core::MemoryBytes memory_bytes;
    std::unique_ptr<MemoryReservation> aggregate;
    std::unique_ptr<MemoryReservation> task;
    std::shared_ptr<QuotaStore> aggregate_store;
    std::shared_ptr<std::atomic<bool>> healthy;
};

class CompositeReservationState final : public detail::MemoryReservationState {
   public:
    explicit CompositeReservationState(CompositeReservationArguments arguments)
        : device_(arguments.device),
          memory_bytes_(arguments.memory_bytes),
          aggregate_(std::move(arguments.aggregate)),
          task_(std::move(arguments.task)),
          aggregate_store_(std::move(arguments.aggregate_store)),
          healthy_(std::move(arguments.healthy)) {}

    [[nodiscard]] bool commit() noexcept override {
        try {
            if (aggregate_ != nullptr && !aggregate_->commit()) {
                return false;
            }
            if (task_ != nullptr && !task_->commit()) {
                if (aggregate_ != nullptr && aggregate_store_ != nullptr) {
                    if (!aggregate_store_->release(device_, memory_bytes_) && healthy_ != nullptr) {
                        healthy_->store(false, std::memory_order_release);
                    }
                } else if (healthy_ != nullptr) {
                    healthy_->store(false, std::memory_order_release);
                }
                return false;
            }
            return true;
        } catch (...) {
            return false;
        }
    }

    void cancel() noexcept override {
        if (aggregate_ != nullptr) {
            aggregate_->cancel();
        }
        if (task_ != nullptr) {
            task_->cancel();
        }
    }

    void abandon() noexcept override {
        if (aggregate_ != nullptr) {
            aggregate_->abandon();
        }
        if (task_ != nullptr) {
            task_->abandon();
        }
    }

    [[nodiscard]] bool is_active() const noexcept override {
        return (aggregate_ != nullptr && aggregate_->is_active()) ||
               (task_ != nullptr && task_->is_active());
    }

    [[nodiscard]] core::MemoryBytes memory_bytes() const noexcept override {
        return memory_bytes_;
    }

   private:
    DeviceId device_;
    core::MemoryBytes memory_bytes_;
    std::unique_ptr<MemoryReservation> aggregate_;
    std::unique_ptr<MemoryReservation> task_;
    std::shared_ptr<QuotaStore> aggregate_store_;
    std::shared_ptr<std::atomic<bool>> healthy_;
};

[[nodiscard]] core::MemoryBytes minimum_limit(const core::QuotaUsage& aggregate,
                                              const core::QuotaUsage& task) noexcept {
    return std::min(aggregate.limit_bytes, task.limit_bytes);
}

}  // namespace

CompositeQuota::CompositeQuota(std::unique_ptr<QuotaStore> aggregate,
                               std::unique_ptr<QuotaStore> task)
    : aggregate_(std::move(aggregate)),
      task_(std::move(task)),
      healthy_(std::make_shared<std::atomic<bool>>(true)) {}

std::optional<MemoryReservation> CompositeQuota::try_reserve(DeviceId device,
                                                             core::MemoryBytes memory_bytes) {
    try {
        if (healthy_ == nullptr || !healthy_->load(std::memory_order_acquire) ||
            aggregate_ == nullptr || !aggregate_->is_healthy() ||
            (task_ != nullptr && !task_->is_healthy())) {
            return std::nullopt;
        }

        auto aggregate_reservation = aggregate_->try_reserve(device, memory_bytes);
        if (!aggregate_reservation.has_value()) {
            return std::nullopt;
        }

        std::unique_ptr<MemoryReservation> task_reservation;
        if (task_ != nullptr) {
            auto reservation = task_->try_reserve(device, memory_bytes);
            if (!reservation.has_value()) {
                return std::nullopt;
            }
            task_reservation = std::make_unique<MemoryReservation>(std::move(*reservation));
        }

        auto aggregate_holder =
            std::make_unique<MemoryReservation>(std::move(*aggregate_reservation));
        return MemoryReservation(std::make_unique<CompositeReservationState>(
            CompositeReservationArguments{.device = device,
                                          .memory_bytes = memory_bytes,
                                          .aggregate = std::move(aggregate_holder),
                                          .task = std::move(task_reservation),
                                          .aggregate_store = aggregate_,
                                          .healthy = healthy_}));
    } catch (...) {
        return std::nullopt;
    }
}

bool CompositeQuota::release(DeviceId device, core::MemoryBytes memory_bytes) {
    if (healthy_ == nullptr || !healthy_->load(std::memory_order_acquire) ||
        aggregate_ == nullptr || !aggregate_->release(device, memory_bytes)) {
        if (healthy_ != nullptr) {
            healthy_->store(false, std::memory_order_release);
        }
        return false;
    }
    if (task_ != nullptr && !task_->release(device, memory_bytes)) {
        healthy_->store(false, std::memory_order_release);
        return false;
    }
    return true;
}

MemoryInfo CompositeQuota::get_memory_info(DeviceId device, core::MemoryBytes physical_total_bytes,
                                           core::MemoryBytes physical_free_bytes) const {
    if (healthy_ == nullptr || !healthy_->load(std::memory_order_acquire) ||
        aggregate_ == nullptr || !aggregate_->is_healthy() ||
        (task_ != nullptr && !task_->is_healthy())) {
        return MemoryInfo{.total_bytes = 0, .free_bytes = 0};
    }

    const MemoryInfo aggregate_info =
        aggregate_->get_memory_info(device, physical_total_bytes, physical_free_bytes);
    if (task_ == nullptr) {
        return aggregate_info;
    }

    const MemoryInfo task_info =
        task_->get_memory_info(device, physical_total_bytes, physical_free_bytes);
    return MemoryInfo{
        .total_bytes = std::min(aggregate_info.total_bytes, task_info.total_bytes),
        .free_bytes = std::min(aggregate_info.free_bytes, task_info.free_bytes),
    };
}

core::QuotaUsage CompositeQuota::usage(DeviceId device) const {
    if (healthy_ == nullptr || !healthy_->load(std::memory_order_acquire) ||
        aggregate_ == nullptr || !aggregate_->is_healthy() ||
        (task_ != nullptr && !task_->is_healthy())) {
        return core::QuotaUsage{};
    }
    const core::QuotaUsage aggregate_usage = aggregate_->usage(device);
    if (task_ == nullptr) {
        return aggregate_usage;
    }
    const core::QuotaUsage task_usage = task_->usage(device);
    return core::QuotaUsage{
        .limit_bytes = minimum_limit(aggregate_usage, task_usage),
        .reserved_bytes = std::min(aggregate_usage.reserved_bytes, task_usage.reserved_bytes),
        .allocated_bytes = std::min(aggregate_usage.allocated_bytes, task_usage.allocated_bytes),
    };
}

bool CompositeQuota::is_healthy() const noexcept {
    return healthy_ != nullptr && healthy_->load(std::memory_order_acquire) &&
           aggregate_ != nullptr && aggregate_->is_healthy() &&
           (task_ == nullptr || task_->is_healthy());
}

}  // namespace glimmer::control
