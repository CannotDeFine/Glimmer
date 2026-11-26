#include "cuda_cooperative_pool.h"

#include "diagnostics.h"

#include <atomic>
#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace glimmer::interceptor {
namespace {
struct alignas(16) HostSlot {
    std::uint32_t retreat{};
    std::uint64_t completed{};
    std::uint64_t zero{};
    std::uint32_t first_tile{};
    std::uint32_t first_compute{};
};
static_assert(sizeof(HostSlot) == 32);
static_assert(offsetof(HostSlot, first_compute) == offsetof(HostSlot, first_tile) + 4);
}  // namespace

struct CudaCooperativePool::Implementation {
    CUcontext context;
    CudaCooperativePoolDispatch dispatch;
    const std::size_t audit_block_capacity;
    const std::size_t host_stride;
    std::vector<bool> occupied;
    mutable std::mutex mutex;
    CUdeviceptr counters{};
    CUdeviceptr mapped_host{};
    void* host{};
    bool is_initialized{};
    bool is_closed{};
    bool has_device_uncertainty{};
    bool has_host_uncertainty{};

    Implementation(CUcontext native_context, CudaCooperativePoolDispatch native_dispatch,
                   std::size_t capacity, std::size_t audit_capacity)
        : context(native_context),
          dispatch(std::move(native_dispatch)),
          audit_block_capacity(audit_capacity),
          host_stride(audit_capacity <= 16384
                          ? (sizeof(HostSlot) + audit_capacity * sizeof(std::uint32_t) + 15) / 16 *
                                16
                          : 0),
          occupied(capacity, false) {
        if (!context || capacity == 0 || capacity > 65536 || !dispatch.context_get_current ||
            audit_capacity > 16384 || (audit_capacity && capacity > 1024) ||
            !dispatch.allocate_device || !dispatch.free_device || !dispatch.allocate_host ||
            !dispatch.free_host || !dispatch.map_host || !dispatch.copy_device_to_host ||
            !dispatch.accounting_owner) {
            throw std::invalid_argument("invalid cooperative pool configuration");
        }
    }
    CUresult validate_context() const noexcept {
        CUcontext current{};
        const auto result = dispatch.context_get_current(&current);
        return result != CUDA_SUCCESS
                   ? result
                   : (current == context ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT);
    }
    HostSlot& slot(std::size_t index) noexcept {
        return *std::launder(
            reinterpret_cast<HostSlot*>(static_cast<std::byte*>(host) + index * host_stride));
    }
    std::uint32_t* counts(std::size_t index) noexcept {
        return reinterpret_cast<std::uint32_t*>(static_cast<std::byte*>(host) +
                                                index * host_stride + sizeof(HostSlot));
    }
};

class CudaCooperativePool::Continuation final : public CudaCooperativeContinuation {
   public:
    Continuation(std::shared_ptr<CudaCooperativePool> pool, std::size_t index,
                 const CooperativeKernelVariant& variant, std::shared_ptr<void> resources)
        : pool_(std::move(pool)),
          resources_(std::move(resources)),
          index_(index),
          logical_blocks_(variant.shape.logical_blocks),
          worker_blocks_(variant.worker_blocks),
          is_audited_(variant.audit.has_value()),
          has_compute_witness_(variant.audit && variant.audit->has_compute_witness) {}
    ~Continuation() override {
        auto& owner = *pool_->implementation_;
        const std::lock_guard lock(owner.mutex);
        owner.occupied[index_] = false;
    }
    CUresult prepare_launch(std::array<unsigned int, 3>& grid,
                            OwnedKernelArguments&) noexcept override {
        auto& owner = *pool_->implementation_;
        std::atomic_ref(owner.slot(index_).retreat).store(0, std::memory_order_release);
        if (is_audited_) {
            std::atomic_ref(owner.slot(index_).first_tile).store(0, std::memory_order_release);
            std::atomic_ref(owner.slot(index_).first_compute).store(0, std::memory_order_release);
            has_observed_first_tile_ = false;
            has_observed_first_compute_ = false;
        }
        grid = {worker_blocks_, 1, 1};
        ++progress_.physical_launches;
        trace("prepare", false);
        return CUDA_SUCCESS;
    }
    CUresult request_retreat() noexcept override {
        auto& owner = *pool_->implementation_;
        std::atomic_ref(owner.slot(index_).retreat).store(1, std::memory_order_release);
        trace("retreat", false);
        return CUDA_SUCCESS;
    }
    void execution_granted(CudaQueueExecutionIdentity identity) noexcept override {
        task_ = identity.task;
        generation_ = identity.generation;
        trace("grant", false);
    }
    void observe_pending() noexcept override {
        if (has_compute_witness_ && !has_observed_first_compute_ &&
            std::atomic_ref(pool_->implementation_->slot(index_).first_compute)
                .load(std::memory_order_acquire)) {
            has_observed_first_compute_ = true;
            // Arithmetic has run, but no additional output tile is claimed.
            trace("first_compute", false);
        }
        if (is_audited_ && !has_observed_first_tile_ &&
            std::atomic_ref(pool_->implementation_->slot(index_).first_tile)
                .load(std::memory_order_acquire)) {
            has_observed_first_tile_ = true;
            trace("first_tile", false, progress_.observed_blocks + 1);
        }
    }
    void observe_priority_retreat() noexcept override {
        trace("priority_retreat", false);
    }
    CUresult stage_completion(CUstream stream) noexcept override {
        auto& owner = *pool_->implementation_;
        return owner.dispatch.copy_device_to_host(&owner.slot(index_).completed,
                                                  owner.counters + index_ * sizeof(std::uint64_t),
                                                  sizeof(std::uint64_t), stream);
    }
    CUresult inspect_stopped(bool& is_complete) noexcept override {
        progress_.observed_blocks = pool_->implementation_->slot(index_).completed;
        is_complete = progress_.observed_blocks >= logical_blocks_;
        if (is_audited_) {
            const auto* counts = pool_->implementation_->counts(index_);
            std::uint64_t finished{};
            bool is_exact = true;
            for (std::size_t tile = 0; tile < logical_blocks_; ++tile) {
                is_exact &= counts[tile] <= 1;
                finished += counts[tile];
            }
            is_exact &= finished == std::min(progress_.observed_blocks, logical_blocks_) &&
                        progress_.observed_blocks <= logical_blocks_ + worker_blocks_;
            trace(is_complete ? "complete" : "suspended", is_exact, finished);
            if (!is_exact) return CUDA_ERROR_UNKNOWN;
        }
        if (!is_complete) {
            progress_.partial_blocks = progress_.observed_blocks;
        }
        return CUDA_SUCCESS;
    }
    CudaCooperativeProgress progress() const noexcept override {
        return progress_;
    }

   private:
    void trace(const char* phase, bool is_exact,
               std::optional<std::uint64_t> finished = {}) const noexcept {
        if (is_audited_)
            report_persistent_audit({phase, task_, generation_, progress_.physical_launches,
                                     logical_blocks_, finished.value_or(progress_.observed_blocks),
                                     is_exact});
    }
    std::shared_ptr<CudaCooperativePool> pool_;
    std::shared_ptr<void> resources_;
    std::size_t index_;
    std::uint64_t logical_blocks_;
    unsigned int worker_blocks_;
    CudaCooperativeProgress progress_;
    bool is_audited_{};
    bool has_observed_first_tile_{};
    bool has_compute_witness_{};
    bool has_observed_first_compute_{};
    std::uint64_t task_{}, generation_{};
};

CudaCooperativePool::CudaCooperativePool(CUcontext context, CudaCooperativePoolDispatch dispatch,
                                         std::size_t capacity, std::size_t audit_block_capacity)
    : implementation_(std::make_unique<Implementation>(context, std::move(dispatch), capacity,
                                                       audit_block_capacity)) {}
CudaCooperativePool::~CudaCooperativePool() {
    if (implementation_->counters || implementation_->host) {
        // Never infer physical stop from the last shared owner disappearing.
        (void)implementation_.release();
        report_diagnostic("cooperative progress pool not released; resources quarantined");
    }
}
CUresult CudaCooperativePool::initialize() noexcept {
    auto& owner = *implementation_;
    const auto context_result = owner.validate_context();
    if (context_result != CUDA_SUCCESS) {
        return context_result;
    }
    if (owner.is_closed) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (owner.is_initialized) {
        return CUDA_SUCCESS;
    }
    // A failed partial initialization must be explicitly released before retry.
    if (owner.counters || owner.host) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    try {
        const auto result = owner.dispatch.allocate_device(
            &owner.counters, owner.occupied.size() * sizeof(std::uint64_t));
        if (result != CUDA_SUCCESS) {
            owner.has_device_uncertainty = owner.counters != 0;
            return result;
        }
        if (!owner.counters) {
            return CUDA_ERROR_UNKNOWN;
        }
        const auto host_result = owner.dispatch.allocate_host(
            &owner.host, owner.occupied.size() * owner.host_stride, CU_MEMHOSTALLOC_DEVICEMAP);
        if (host_result != CUDA_SUCCESS) {
            owner.has_host_uncertainty = owner.host != nullptr;
            return host_result;
        }
        if (!owner.host || reinterpret_cast<std::uintptr_t>(owner.host) % alignof(HostSlot) != 0) {
            return CUDA_ERROR_UNKNOWN;
        }
        const auto map_result = owner.dispatch.map_host(&owner.mapped_host, owner.host, 0);
        if (map_result != CUDA_SUCCESS) {
            return map_result;
        }
        if (!owner.mapped_host ||
            owner.mapped_host > std::numeric_limits<CUdeviceptr>::max() -
                                    owner.occupied.size() * owner.host_stride ||
            owner.counters > std::numeric_limits<CUdeviceptr>::max() -
                                 owner.occupied.size() * sizeof(std::uint64_t) ||
            owner.counters % alignof(std::uint64_t) != 0 ||
            owner.mapped_host % alignof(std::uint32_t) != 0) {
            return CUDA_ERROR_UNKNOWN;
        }
        for (std::size_t index = 0; index < owner.occupied.size(); ++index) {
            std::construct_at(reinterpret_cast<HostSlot*>(static_cast<std::byte*>(owner.host) +
                                                          index * owner.host_stride));
            for (std::size_t tile = 0; tile < owner.audit_block_capacity; ++tile)
                std::construct_at(owner.counts(index) + tile, 0U);
        }
        owner.is_initialized = true;
        return CUDA_SUCCESS;
    } catch (...) {
        owner.has_device_uncertainty |= owner.counters != 0;
        owner.has_host_uncertainty |= owner.host != nullptr;
        return CUDA_ERROR_UNKNOWN;
    }
}
CUresult CudaCooperativePool::release() noexcept {
    auto& owner = *implementation_;
    const auto context_result = owner.validate_context();
    if (context_result != CUDA_SUCCESS) {
        return context_result;
    }
    {
        const std::lock_guard lock(owner.mutex);
        for (const bool is_occupied : owner.occupied) {
            if (is_occupied) {
                return CUDA_ERROR_NOT_READY;
            }
        }
        owner.is_closed = true;
    }
    // No metadata lock may cover native CUDA or accounting callbacks.
    if (owner.has_device_uncertainty || owner.has_host_uncertainty) {
        return CUDA_ERROR_UNKNOWN;
    }
    try {
        if (owner.counters) {
            const auto result = owner.dispatch.free_device(owner.counters);
            if (result.ownership == CudaNativeReleaseState::kReleased) {
                owner.counters = 0;
            } else if (result.ownership != CudaNativeReleaseState::kRetained) {
                owner.has_device_uncertainty = true;
            }
            owner.has_device_uncertainty |= result.status == CUDA_SUCCESS &&
                                            result.ownership != CudaNativeReleaseState::kReleased;
            if (result.status != CUDA_SUCCESS || owner.counters) {
                return result.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result.status;
            }
        }
        if (owner.host) {
            const auto result = owner.dispatch.free_host(owner.host);
            if (result.ownership == CudaNativeReleaseState::kReleased) {
                owner.host = nullptr;
                owner.mapped_host = 0;
            } else if (result.ownership != CudaNativeReleaseState::kRetained) {
                owner.has_host_uncertainty = true;
            }
            owner.has_host_uncertainty |= result.status == CUDA_SUCCESS &&
                                          result.ownership != CudaNativeReleaseState::kReleased;
            if (result.status != CUDA_SUCCESS || owner.host) {
                return result.status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : result.status;
            }
        }
        return CUDA_SUCCESS;
    } catch (...) {
        owner.has_device_uncertainty |= owner.counters != 0;
        owner.has_host_uncertainty |= owner.host != nullptr;
        return CUDA_ERROR_UNKNOWN;
    }
}
std::size_t CudaCooperativePool::occupied_slots() const noexcept {
    auto& owner = *implementation_;
    const std::lock_guard lock(owner.mutex);
    std::size_t count = 0;
    for (const bool is_occupied : owner.occupied) {
        count += static_cast<std::size_t>(is_occupied);
    }
    return count;
}
std::optional<PreparedCooperativeKernel> CudaCooperativePool::prepare(
    const CooperativeKernelVariant& variant, std::array<unsigned int, 3> grid,
    std::array<unsigned int, 3> block, void* const* arguments,
    std::shared_ptr<void> kernel_resources) noexcept {
    auto& owner = *implementation_;
    if (!variant.function || variant.context != owner.context || !kernel_resources ||
        !variant.shape.logical_blocks || !variant.shape.block_threads || !variant.worker_blocks ||
        variant.worker_blocks > variant.shape.logical_blocks ||
        variant.parameter_layout.size() > 251 ||
        (!variant.parameter_layout.empty() && !arguments)) {
        return std::nullopt;
    }
    if (variant.audit) {
        const auto valid_pointer = [&](std::size_t parameter) {
            return parameter < variant.parameter_layout.size() &&
                   variant.parameter_layout[parameter].size_bytes == 8 &&
                   variant.parameter_layout[parameter].alignment_bytes == 8;
        };
        if (variant.shape.logical_blocks > owner.audit_block_capacity ||
            variant.audit->first_tile == variant.audit->tile_counts ||
            !valid_pointer(variant.audit->first_tile) || !valid_pointer(variant.audit->tile_counts))
            return std::nullopt;
    }
    std::uint64_t grid_product = 1;
    std::uint64_t block_product = 1;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (!grid[axis] || !block[axis] ||
            grid_product > std::numeric_limits<std::uint32_t>::max() / grid[axis] ||
            block_product > std::numeric_limits<std::uint32_t>::max() / block[axis]) {
            return std::nullopt;
        }
        grid_product *= grid[axis];
        block_product *= block[axis];
    }
    if (grid_product != variant.shape.logical_blocks ||
        block_product != variant.shape.block_threads) {
        return std::nullopt;
    }
    auto pool = weak_from_this().lock();
    if (!pool) {
        return std::nullopt;
    }
    std::size_t index = owner.occupied.size();
    {
        const std::lock_guard lock(owner.mutex);
        if (!owner.is_initialized || owner.is_closed) {
            return std::nullopt;
        }
        for (std::size_t candidate = 0; candidate < owner.occupied.size(); ++candidate) {
            if (!owner.occupied[candidate]) {
                index = candidate;
                owner.occupied[index] = true;
                break;
            }
        }
    }
    if (index == owner.occupied.size()) {
        return std::nullopt;
    }
    std::shared_ptr<Continuation> continuation;
    try {
        continuation =
            std::make_shared<Continuation>(pool, index, variant, std::move(kernel_resources));
        auto layout = variant.parameter_layout;
        layout.insert(layout.end(), {{4, 4}, {4, 4}, {4, 4}, {8, 8}, {8, 8}});
        std::vector<void*> addresses;
        addresses.reserve(layout.size());
        for (std::size_t parameter = 0; parameter < variant.parameter_layout.size(); ++parameter) {
            addresses.push_back(arguments[parameter]);
        }
        auto counter = owner.counters + index * sizeof(std::uint64_t);
        auto retreat = owner.mapped_host + index * owner.host_stride;
        auto first_tile = retreat + offsetof(HostSlot, first_tile);
        auto counts = retreat + sizeof(HostSlot);
        if (variant.audit) {
            addresses[variant.audit->first_tile] = &first_tile;
            addresses[variant.audit->tile_counts] = &counts;
        }
        addresses.insert(addresses.end(), {&grid[0], &grid[1], &grid[2], &counter, &retreat});
        auto owned_arguments = OwnedKernelArguments::copy(layout, addresses.data());
        if (!owned_arguments) {
            return std::nullopt;
        }
        auto& host_slot = owner.slot(index);
        host_slot = {};
        if (variant.audit) std::fill_n(owner.counts(index), variant.shape.logical_blocks, 0U);
        return PreparedCooperativeKernel{
            CudaQueuedCopy{CudaQueuedCopyKind::kHostToDevice, counter, 0, &host_slot.zero,
                           sizeof(std::uint64_t), continuation},
            CudaQueuedKernel{variant.function, grid, block, 0, std::move(*owned_arguments),
                             continuation, continuation},
            continuation};
    } catch (...) {
        if (!continuation) {
            const std::lock_guard lock(owner.mutex);
            owner.occupied[index] = false;
        }
        return std::nullopt;
    }
}

}  // namespace glimmer::interceptor
