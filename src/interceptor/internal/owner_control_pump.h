#pragma once

#include "glimmer/control/async_cooperative_launch_gate.h"

#include <cuda.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace glimmer::interceptor {

// Existing actor-thread CPU role only: no thread, CUDA or domain callback.
// Each method releases its CPU lease before returning to physical work.
// Construction precedes actor publication. Retire on the actor before join;
// no other method may race retirement or destruction.
class OwnerControlPump final {
   public:
    using Gate = control::AsyncCooperativeLaunchGate;
    using Clock = std::chrono::steady_clock;
    explicit OwnerControlPump(std::shared_ptr<Gate> selected_gate)
        : gate(std::move(selected_gate)),
          is_selected(gate && gate->owner_pump_snapshot().is_enabled) {
        if (attach() != CUDA_SUCCESS)
            throw std::runtime_error("CPU control owner registration rejected");
    }
    OwnerControlPump(const OwnerControlPump&) = delete;
    OwnerControlPump& operator=(const OwnerControlPump&) = delete;
    [[nodiscard]] bool enabled() const noexcept {
        return is_selected;
    }
    [[nodiscard]] CUresult attach() noexcept {
        if (!enabled() || (owner_ && !is_retired_)) return CUDA_SUCCESS;
        owner_.reset();
        owner_.emplace(gate->register_cpu_owner());
        is_retired_ = !static_cast<bool>(*owner_);
        return is_retired_ ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult status() const noexcept {
        return enabled() && gate->owner_pump_snapshot().is_quarantined ? CUDA_ERROR_UNKNOWN
                                                                       : CUDA_SUCCESS;
    }
    [[nodiscard]] CUresult progress() noexcept {
        if (!enabled()) return CUDA_SUCCESS;
        if (is_retired_) return CUDA_ERROR_DEINITIALIZED;
        if (!owner_) return contain();
        if (const auto failure = status(); failure != CUDA_SUCCESS) return failure;
        auto executor = gate->try_acquire_cpu(*owner_);
        if (!executor) return CUDA_SUCCESS;  // another real CPU actor owns the lease
        return classify(executor->progress());
    }
    [[nodiscard]] CUresult wait_until(
        Clock::time_point deadline,
        const std::shared_ptr<Gate::ProgressSignal>& fallback) noexcept {
        if (enabled() && is_retired_) return CUDA_ERROR_DEINITIALIZED;
        if (const auto failure = status(); failure != CUDA_SUCCESS) return failure;
        if (enabled()) {
            if (!owner_) return contain();
            auto executor = gate->try_acquire_cpu(*owner_);
            if (executor) {
                const auto step = executor->progress();
                if (const auto failure = classify(step); failure != CUDA_SUCCESS) return failure;
                if (step == Gate::CpuProgress::kCompleted) return CUDA_SUCCESS;
                const auto hint = executor->wait_until(deadline);
                if (hint.status == Gate::CpuWaitStatus::kError ||
                    hint.status == Gate::CpuWaitStatus::kRejected) {
                    gate->quarantine_owner_pump();
                    return CUDA_ERROR_UNKNOWN;
                }
                return status();  // every hint requires normal actor/timer reinspection
            }
        }
        const auto now = Clock::now();
        if (deadline > now) {
            const auto delay =
                std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
            try {
                if (fallback)
                    static_cast<void>(fallback->wait_for(delay));
                else
                    std::this_thread::sleep_until(deadline);
            } catch (...) {
                return enabled() ? contain() : CUDA_ERROR_UNKNOWN;
            }
        }
        return status();
    }
    [[nodiscard]] CUresult retire() noexcept {
        if (!enabled() || is_retired_) return status();
        if (!owner_) return contain();
        if (!owner_->begin_retirement()) return contain();
        const auto deadline = Clock::now() + std::chrono::seconds{2};
        for (;;) {
            if (owner_->detach()) {
                is_retired_ = true;
                return status();
            }
            if (status() != CUDA_SUCCESS || Clock::now() >= deadline) return contain();
            if (progress() != CUDA_SUCCESS || wait_until(deadline, {}) != CUDA_SUCCESS)
                return contain();
        }
    }

   private:
    [[nodiscard]] CUresult classify(Gate::CpuProgress step) noexcept {
        if (step == Gate::CpuProgress::kRejected || step == Gate::CpuProgress::kQuarantined)
            return contain();
        return status();
    }
    [[nodiscard]] CUresult contain() noexcept {
        gate->quarantine_owner_pump();
        return CUDA_ERROR_UNKNOWN;
    }
    const std::shared_ptr<Gate> gate;
    const bool is_selected;
    std::optional<Gate::CpuOwner> owner_;
    bool is_retired_ = false;
};

}  // namespace glimmer::interceptor
