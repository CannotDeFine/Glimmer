#include "internal/cuda_process_memory_owner.h"

#include <algorithm>
#include <atomic>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> memory_thread_identity;
enum class Phase : std::uint8_t {
    kReserved,
    kNativeAllocation,
    kLive,
    kNativeFree,
    kFreePending,
    kUnknown
};
struct Allocation {
    std::uint64_t id{}, pool{};
    std::size_t bytes{};
    CUdeviceptr address{};
    Phase phase{Phase::kReserved};
    CudaProcessMemoryFreeKind free_kind{CudaProcessMemoryFreeKind::kSynchronous};
    std::optional<core::QuotaReservation> reservation;
    std::shared_ptr<CudaProcessMemoryCompletion> tail;
    std::atomic<std::size_t> borrows{};
};
struct BorrowPin {
    std::shared_ptr<Allocation> allocation;
    explicit BorrowPin(std::shared_ptr<Allocation> value) : allocation(std::move(value)) {}
    ~BorrowPin() {
        allocation->borrows.fetch_sub(1, std::memory_order_release);
    }
};
struct Pool {
    std::uint64_t id{};
    CUmemoryPool native{};
};
std::uint64_t take_id(std::uint64_t& next) noexcept {
    return std::exchange(next, next == UINT64_MAX ? 0 : next + 1);
}
bool clean_allocation_error(CUresult status) noexcept {
    return status == CUDA_ERROR_INVALID_VALUE || status == CUDA_ERROR_OUT_OF_MEMORY ||
           status == CUDA_ERROR_NOT_SUPPORTED || status == CUDA_ERROR_INVALID_HANDLE ||
           status == CUDA_ERROR_INVALID_CONTEXT || status == CUDA_ERROR_NOT_INITIALIZED;
}
CudaProcessMemoryResult rejected(CUresult status) noexcept {
    return {status, {}, 0, 0};
}
}  // namespace

struct CudaProcessMemoryOwner::Implementation {
    const pid_t process{::getpid()};
    const std::weak_ptr<const void> owner{memory_thread_identity};
    const CudaProcessMemoryConfig config;
    const std::shared_ptr<core::QuotaLedger> quota;
    std::vector<std::shared_ptr<Allocation>> allocations;
    std::vector<Pool> pools;
    std::uint64_t next_allocation{1}, next_pool{1}, default_pool{}, current_pool{};
    std::uint64_t native_call{};
    std::size_t context_bytes{};
    CUresult failure{CUDA_SUCCESS};
    bool active{}, is_closed{}, is_shutdown{};

    Implementation(CudaProcessMemoryConfig settings, std::shared_ptr<core::QuotaLedger> budget)
        : config(settings), quota(std::move(budget)) {
        allocations.reserve(config.maximum_allocations);
        pools.reserve(config.maximum_pools);
    }
    bool owns() const noexcept {
        return ::getpid() == process && memory_thread_identity &&
               owner.lock() == memory_thread_identity;
    }
    CUresult gate(bool publication = false) const noexcept {
        if (!owns()) return CUDA_ERROR_INVALID_CONTEXT;
        if (active || (native_call && !publication)) return CUDA_ERROR_NOT_READY;
        if (failure != CUDA_SUCCESS) return CUDA_ERROR_UNKNOWN;
        return is_shutdown ? CUDA_ERROR_DEINITIALIZED : CUDA_SUCCESS;
    }
    struct Scope {
        Implementation& state;
        explicit Scope(Implementation& value) : state(value) {
            state.active = true;
        }
        ~Scope() {
            state.active = false;
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
    };
    CUresult fault(CUresult status) noexcept {
        if (failure == CUDA_SUCCESS) failure = status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
        return status == CUDA_SUCCESS ? CUDA_ERROR_UNKNOWN : status;
    }
    std::shared_ptr<Allocation> find(std::uint64_t id) const noexcept {
        const auto entry =
            std::ranges::find_if(allocations, [id](const auto& value) { return value->id == id; });
        return entry == allocations.end() ? nullptr : *entry;
    }
    const Pool* find_pool(std::uint64_t id) const noexcept {
        const auto entry = std::ranges::find(pools, id, &Pool::id);
        return entry == pools.end() ? nullptr : &*entry;
    }
    void erase(const std::shared_ptr<Allocation>& allocation) noexcept {
        std::erase(allocations, allocation);
    }
    CUresult refund(const std::shared_ptr<Allocation>& allocation) noexcept {
        try {
            if (!quota->release(allocation->bytes)) return fault(CUDA_ERROR_UNKNOWN);
            erase(allocation);
            return CUDA_SUCCESS;
        } catch (...) {
            return fault(CUDA_ERROR_UNKNOWN);
        }
    }
    CUresult ready(const std::shared_ptr<Allocation>& allocation) noexcept {
        if (!allocation->tail) return CUDA_SUCCESS;
        const auto status = allocation->tail->observe();
        if (status == CUDA_SUCCESS)
            allocation->tail.reset();
        else if (status != CUDA_ERROR_NOT_READY)
            return fault(status);
        return status;
    }
};

CudaProcessMemoryOwner::CudaProcessMemoryOwner(CudaProcessMemoryConfig config,
                                               std::shared_ptr<core::QuotaLedger> quota) {
    if (!config.identity || config.device_ordinal < 0 || !config.maximum_allocations ||
        !config.maximum_pools || !config.maximum_borrows || !quota ||
        config.maximum_allocations > 65536 || config.maximum_pools > 65536 ||
        config.maximum_borrows > 65536)
        throw std::invalid_argument("invalid process memory owner configuration");
    if (!memory_thread_identity) memory_thread_identity = std::make_shared<int>(0);
    implementation_ = std::make_unique<Implementation>(config, std::move(quota));
}
CudaProcessMemoryOwner::~CudaProcessMemoryOwner() {
    // An unsettled owner retains its ledger, records and tails. Wrapper loss is
    // not native free or proof of server-side GPU stop, even after peer exit.
    if (!implementation_->owns() || !implementation_->allocations.empty() ||
        implementation_->context_bytes)
        (void)implementation_.release();
}

CudaProcessMemoryResult CudaProcessMemoryOwner::publish_pool(
    CudaProcessPoolQuery query, const CudaProcessPoolObservation& observation) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    if (state.is_closed) return rejected(CUDA_ERROR_DEINITIALIZED);
    if (query != CudaProcessPoolQuery::kDefault && query != CudaProcessPoolQuery::kCurrent)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    const Implementation::Scope scope(state);
    auto result = CudaProcessMemoryResult{observation.original_status, observation.original_status};
    if (observation.owner_status != CUDA_SUCCESS) {
        result.status = state.fault(observation.owner_status);
        return result;
    }
    if (observation.original_status != CUDA_SUCCESS) return result;
    if (observation.output.status != CUDA_SUCCESS || !observation.output.was_written ||
        !observation.native) {
        result.status = state.fault(CUDA_ERROR_UNKNOWN);
        return result;
    }
    auto found = std::ranges::find(state.pools, observation.native, &Pool::native);
    if (query == CudaProcessPoolQuery::kDefault && state.default_pool &&
        (found == state.pools.end() || found->id != state.default_pool)) {
        result.status = state.fault(CUDA_ERROR_UNKNOWN);
        return result;
    }
    if (found == state.pools.end()) {
        if (state.pools.size() == state.config.maximum_pools || !state.next_pool) {
            result.status = CUDA_ERROR_OUT_OF_MEMORY;
            return result;
        }
        state.pools.push_back({take_id(state.next_pool), observation.native});
        found = std::prev(state.pools.end());
    }
    result.id = found->id;
    if (query == CudaProcessPoolQuery::kDefault)
        state.default_pool = result.id;
    else
        state.current_pool = result.id;
    return result;
}
CudaProcessPoolView CudaProcessMemoryOwner::pool(std::uint64_t id) const noexcept {
    const auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return {status, nullptr};
    const auto* found = state.find_pool(id);
    return found ? CudaProcessPoolView{CUDA_SUCCESS, found->native} : CudaProcessPoolView{};
}
CudaProcessMemoryResult CudaProcessMemoryOwner::publish_selected_pool(
    std::uint64_t id, CUresult original_status, CUresult owner_status) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    if (state.is_closed) return rejected(CUDA_ERROR_DEINITIALIZED);
    if (!state.find_pool(id)) return rejected(CUDA_ERROR_INVALID_HANDLE);
    const Implementation::Scope scope(state);
    CudaProcessMemoryResult result{original_status, original_status};
    if (owner_status != CUDA_SUCCESS || original_status != CUDA_SUCCESS) {
        result.status = state.fault(owner_status != CUDA_SUCCESS ? owner_status : original_status);
        return result;
    }
    state.current_pool = id;
    result.id = id;
    return result;
}

// Quota bytes and an independently validated pool capability.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CudaProcessMemoryResult CudaProcessMemoryOwner::reserve(std::size_t bytes,
                                                        std::uint64_t pool_id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    if (state.is_closed) return rejected(CUDA_ERROR_DEINITIALIZED);
    if (pool_id && !state.find_pool(pool_id)) return rejected(CUDA_ERROR_INVALID_HANDLE);
    if (!state.next_allocation || state.allocations.size() == state.config.maximum_allocations)
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    const Implementation::Scope scope(state);
    try {
        auto reservation = state.quota->try_reserve(bytes);
        if (!reservation) return rejected(CUDA_ERROR_OUT_OF_MEMORY);
        auto allocation = std::make_shared<Allocation>();
        allocation->bytes = bytes;
        allocation->pool = pool_id;
        allocation->reservation.emplace(std::move(*reservation));
        allocation->id = take_id(state.next_allocation);
        state.allocations.push_back(allocation);
        return {CUDA_SUCCESS, {}, allocation->id, 0};
    } catch (const std::bad_alloc&) {
        return rejected(CUDA_ERROR_OUT_OF_MEMORY);
    } catch (...) {
        return rejected(state.fault(CUDA_ERROR_UNKNOWN));
    }
}
CUresult CudaProcessMemoryOwner::cancel(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kReserved) return CUDA_ERROR_INVALID_HANDLE;
    const Implementation::Scope scope(state);
    allocation->reservation.reset();
    state.erase(allocation);
    return CUDA_SUCCESS;
}
CUresult CudaProcessMemoryOwner::begin_allocation(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kReserved) return CUDA_ERROR_INVALID_HANDLE;
    const Implementation::Scope scope(state);
    try {
        if (!allocation->reservation->commit()) return state.fault(CUDA_ERROR_UNKNOWN);
        allocation->reservation.reset();
        allocation->phase = Phase::kNativeAllocation;
        state.native_call = id;
        return CUDA_SUCCESS;
    } catch (...) {
        allocation->phase = Phase::kUnknown;
        return state.fault(CUDA_ERROR_UNKNOWN);
    }
}
CudaProcessMemoryResult CudaProcessMemoryOwner::publish_allocation(
    std::uint64_t id, CudaProcessMemoryObservation observation) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kNativeAllocation || state.native_call != id)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    const Implementation::Scope scope(state);
    state.native_call = 0;
    CudaProcessMemoryResult result{observation.original_status, observation.original_status};
    allocation->phase = Phase::kUnknown;
    allocation->tail = std::move(observation.tail);
    // Retain even unpublishable native output as private evidence. An unwritten
    // seed is not an allocation address and must never become free authority.
    allocation->address =
        observation.output.status == CUDA_SUCCESS && observation.output.was_written
            ? observation.address
            : 0;
    if (observation.owner_status != CUDA_SUCCESS || observation.output.status != CUDA_SUCCESS) {
        result.status =
            state.fault(observation.owner_status != CUDA_SUCCESS ? observation.owner_status
                                                                 : observation.output.status);
        return result;
    }
    if (observation.original_status != CUDA_SUCCESS) {
        if (clean_allocation_error(observation.original_status) &&
            (!observation.output.was_written || !observation.address)) {
            const auto status = state.refund(allocation);
            if (status != CUDA_SUCCESS) result.status = status;
        } else
            state.fault(observation.original_status);
        return result;
    }
    if (!observation.output.was_written || (!allocation->bytes && observation.address) ||
        (allocation->bytes &&
         (!observation.address || !allocation->tail ||
          allocation->bytes - 1 > std::numeric_limits<CUdeviceptr>::max() - observation.address))) {
        result.status = state.fault(CUDA_ERROR_UNKNOWN);
        return result;
    }
    if (!allocation->bytes) {
        result.status = state.refund(allocation);
        return result;
    }
    const auto last = observation.address + allocation->bytes - 1;
    if (std::ranges::any_of(state.allocations, [&](const auto& other) {
            return other != allocation && other->phase == Phase::kLive && other->address <= last &&
                   observation.address <= other->address + other->bytes - 1;
        })) {
        result.status = state.fault(CUDA_ERROR_UNKNOWN);
        return result;
    }
    allocation->address = observation.address;
    allocation->phase = Phase::kLive;
    result.id = id;
    result.address = allocation->address;
    return result;
}
CUresult CudaProcessMemoryOwner::allocation_ready(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kLive) return CUDA_ERROR_INVALID_HANDLE;
    const Implementation::Scope scope(state);
    return state.ready(allocation);
}
// Stable allocation ID, byte offset, then byte extent.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
CudaProcessMemoryBorrow CudaProcessMemoryOwner::borrow(std::uint64_t id, std::size_t offset,
                                                       std::size_t bytes) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return {status, 0, 0, {}};
    if (state.is_closed) return {CUDA_ERROR_DEINITIALIZED, 0, 0, {}};
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kLive || offset > allocation->bytes ||
        bytes > allocation->bytes - offset ||
        offset > std::numeric_limits<CUdeviceptr>::max() - allocation->address)
        return {};
    const Implementation::Scope scope(state);
    if (allocation->borrows.load(std::memory_order_acquire) >= state.config.maximum_borrows)
        return {CUDA_ERROR_OUT_OF_MEMORY, 0, 0, {}};
    allocation->borrows.fetch_add(1, std::memory_order_relaxed);
    try {
        return {CUDA_SUCCESS, allocation->address + offset, bytes,
                std::make_shared<BorrowPin>(allocation)};
    } catch (...) {
        allocation->borrows.fetch_sub(1, std::memory_order_release);
        return {CUDA_ERROR_OUT_OF_MEMORY, 0, 0, {}};
    }
}
std::optional<CudaProcessMemoryView> CudaProcessMemoryOwner::allocation(
    std::uint64_t id) const noexcept {
    const auto& state = *implementation_;
    if (state.gate() != CUDA_SUCCESS || state.is_closed) return std::nullopt;
    const auto found = state.find(id);
    if (!found || found->phase != Phase::kLive) return std::nullopt;
    return CudaProcessMemoryView{found->id, found->address, found->bytes};
}
std::vector<CudaProcessMemoryView> CudaProcessMemoryOwner::live_allocations() const {
    const auto& state = *implementation_;
    std::vector<CudaProcessMemoryView> result;
    if (state.gate() != CUDA_SUCCESS) return result;
    result.reserve(state.allocations.size());
    for (const auto& allocation : state.allocations)
        if (allocation->phase == Phase::kLive)
            result.push_back({allocation->id, allocation->address, allocation->bytes});
    return result;
}
CudaProcessMemoryLeases CudaProcessMemoryOwner::borrow_live() noexcept {
    auto& state = *implementation_;
    const auto gate = state.gate();
    if (gate != CUDA_SUCCESS) return {gate, {}};
    if (state.is_closed) return {CUDA_ERROR_DEINITIALIZED, {}};
    if (std::ranges::none_of(state.allocations, [](const auto& allocation) {
            return allocation->phase == Phase::kLive;
        }))
        return {CUDA_SUCCESS, {}};
    try {
        auto leases = std::make_shared<std::vector<std::shared_ptr<const void>>>();
        leases->reserve(state.allocations.size());
        for (const auto& allocation : state.allocations) {
            if (allocation->phase != Phase::kLive) continue;
            auto captured = borrow(allocation->id, 0, allocation->bytes);
            if (captured.status != CUDA_SUCCESS) return {captured.status, {}};
            leases->push_back(std::move(captured.lifetime));
        }
        return {CUDA_SUCCESS, leases->empty() ? nullptr : std::move(leases)};
    } catch (...) {
        return {CUDA_ERROR_OUT_OF_MEMORY, {}};
    }
}
CUresult CudaProcessMemoryOwner::reserve_context_bytes(std::size_t bytes) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    if (state.is_closed) return CUDA_ERROR_DEINITIALIZED;
    if (bytes > SIZE_MAX - state.context_bytes) return CUDA_ERROR_OUT_OF_MEMORY;
    try {
        auto reserved = state.quota->try_reserve(bytes);
        if (!reserved) return CUDA_ERROR_OUT_OF_MEMORY;
        if (!reserved->commit()) return state.fault(CUDA_ERROR_UNKNOWN);
        state.context_bytes += bytes;
        return CUDA_SUCCESS;
    } catch (...) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
}
CUresult CudaProcessMemoryOwner::release_context_bytes(std::size_t bytes) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    if (bytes > state.context_bytes) return state.fault(CUDA_ERROR_INVALID_VALUE);
    try {
        if (!state.quota->release(bytes)) return state.fault(CUDA_ERROR_UNKNOWN);
        state.context_bytes -= bytes;
        return CUDA_SUCCESS;
    } catch (...) {
        return state.fault(CUDA_ERROR_UNKNOWN);
    }
}
CUresult CudaProcessMemoryOwner::quarantine(CUresult failure) noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return CUDA_ERROR_INVALID_CONTEXT;
    return state.fault(failure);
}
std::optional<std::uint64_t> CudaProcessMemoryOwner::pool_identity(
    CUmemoryPool native) const noexcept {
    const auto& state = *implementation_;
    if (state.gate() != CUDA_SUCCESS) return std::nullopt;
    const auto found = std::ranges::find(state.pools, native, &Pool::native);
    return found == state.pools.end() ? std::nullopt : std::optional{found->id};
}
CudaProcessMemoryResult CudaProcessMemoryOwner::begin_release(
    std::uint64_t id, CudaProcessMemoryFreeKind kind) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return rejected(status);
    if (kind != CudaProcessMemoryFreeKind::kSynchronous &&
        kind != CudaProcessMemoryFreeKind::kAsynchronous)
        return rejected(CUDA_ERROR_INVALID_VALUE);
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kLive)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    const Implementation::Scope scope(state);
    const auto ready = state.ready(allocation);
    if (ready != CUDA_SUCCESS) return rejected(ready);
    if (allocation->borrows.load(std::memory_order_acquire)) return rejected(CUDA_ERROR_NOT_READY);
    allocation->phase = Phase::kNativeFree;
    allocation->free_kind = kind;
    state.native_call = id;
    return {CUDA_SUCCESS, {}, id, allocation->address};
}
CudaProcessMemoryResult CudaProcessMemoryOwner::publish_release(
    std::uint64_t id, CUresult original_status, CUresult owner_status,
    std::shared_ptr<CudaProcessMemoryCompletion> tail) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(true); status != CUDA_SUCCESS) return rejected(status);
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kNativeFree || state.native_call != id)
        return rejected(CUDA_ERROR_INVALID_HANDLE);
    const Implementation::Scope scope(state);
    state.native_call = 0;
    allocation->phase = Phase::kUnknown;
    allocation->tail = std::move(tail);
    CudaProcessMemoryResult result{original_status, original_status};
    if (owner_status != CUDA_SUCCESS || original_status != CUDA_SUCCESS) {
        result.status = state.fault(owner_status != CUDA_SUCCESS ? owner_status : original_status);
        return result;
    }
    if (allocation->free_kind == CudaProcessMemoryFreeKind::kAsynchronous) {
        if (!allocation->tail) {
            result.status = state.fault(CUDA_ERROR_UNKNOWN);
            return result;
        }
        allocation->phase = Phase::kFreePending;
        result.id = id;
    } else if (allocation->tail)
        result.status = state.fault(CUDA_ERROR_UNKNOWN);
    else
        result.status = state.refund(allocation);
    return result;
}
CUresult CudaProcessMemoryOwner::release_ready(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    const auto allocation = state.find(id);
    if (!allocation || allocation->phase != Phase::kFreePending) return CUDA_ERROR_INVALID_HANDLE;
    const Implementation::Scope scope(state);
    const auto ready = state.ready(allocation);
    if (ready != CUDA_SUCCESS) return ready;
    if (allocation->borrows.load(std::memory_order_acquire)) return CUDA_ERROR_NOT_READY;
    return state.refund(allocation);
}
CUresult CudaProcessMemoryOwner::progress() noexcept {
    auto& state = *implementation_;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    const Implementation::Scope scope(state);
    bool pending{};
    for (std::size_t index = 0; index < state.allocations.size();) {
        const auto allocation = state.allocations[index];
        if (allocation->phase == Phase::kLive || allocation->phase == Phase::kFreePending) {
            const auto status = state.ready(allocation);
            if (status == CUDA_ERROR_NOT_READY)
                pending = true;
            else if (status != CUDA_SUCCESS)
                return status;
            else if (allocation->phase == Phase::kFreePending) {
                if (allocation->borrows.load(std::memory_order_acquire))
                    pending = true;
                else {
                    const auto freed = state.refund(allocation);
                    if (freed != CUDA_SUCCESS) return freed;
                    continue;
                }
            }
        }
        ++index;
    }
    return pending ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
}
CudaProcessMemorySnapshot CudaProcessMemoryOwner::snapshot() const noexcept {
    const auto& state = *implementation_;
    CudaProcessMemorySnapshot result;
    if (!state.owns()) return result;
    if (state.active) {
        result.status = CUDA_ERROR_NOT_READY;
        return result;
    }
    result.status = state.failure == CUDA_SUCCESS ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
    result.allocations = state.allocations.size();
    result.pools = state.pools.size();
    for (const auto& allocation : state.allocations) {
        result.retained_bytes += allocation->bytes;
        const auto borrowed = allocation->borrows.load(std::memory_order_acquire);
        result.borrowed = borrowed > std::numeric_limits<std::size_t>::max() - result.borrowed
                              ? std::numeric_limits<std::size_t>::max()
                              : result.borrowed + borrowed;
    }
    result.default_pool = state.default_pool;
    result.current_pool = state.current_pool;
    try {
        result.quota = state.quota->usage();
    } catch (...) {
        result.status = CUDA_ERROR_UNKNOWN;
    }
    result.is_closed = state.is_closed;
    result.is_shutdown = state.is_shutdown;
    result.is_quarantined = state.failure != CUDA_SUCCESS;
    result.has_native_call = state.native_call != 0;
    return result;
}
CUresult CudaProcessMemoryOwner::close() noexcept {
    auto& state = *implementation_;
    if (!state.owns()) return CUDA_ERROR_INVALID_CONTEXT;
    if (state.active || state.native_call) return CUDA_ERROR_NOT_READY;
    state.is_closed = true;
    return state.failure == CUDA_SUCCESS ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}
CUresult CudaProcessMemoryOwner::shutdown() noexcept {
    auto& state = *implementation_;
    if (state.owns() && !state.active && !state.native_call && state.is_shutdown)
        return CUDA_SUCCESS;
    if (const auto status = state.gate(); status != CUDA_SUCCESS) return status;
    if (!state.is_closed) return CUDA_ERROR_INVALID_VALUE;
    if (!state.allocations.empty() || state.context_bytes) return CUDA_ERROR_NOT_READY;
    state.pools.clear();
    state.default_pool = state.current_pool = 0;
    state.is_shutdown = true;
    return CUDA_SUCCESS;
}

}  // namespace glimmer::backend::cuda
