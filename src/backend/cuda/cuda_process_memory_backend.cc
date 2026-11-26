#include "internal/cuda_process_memory_backend.h"
#include "internal/cuda_execution_domain.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unistd.h>

namespace glimmer::backend::cuda {
namespace {
thread_local std::shared_ptr<const void> process_memory_native_thread;
constexpr auto kProcessTag = UINT64_C(1) << 63;
std::uint64_t local_id(std::uint64_t id) noexcept {
    return id & ~kProcessTag;
}
}  // namespace

struct CudaProcessMemoryDevice::Implementation {
    const pid_t process{::getpid()};
    const std::weak_ptr<const void> thread{process_memory_native_thread};
    const CudaProcessMemoryDispatch dispatch;
    const std::function<CUresult()> native_entry;
    const std::shared_ptr<CudaExecutionDomainOwner> domain;
    bool active{}, uncertain{};
    Implementation(CudaProcessMemoryDispatch functions, std::function<CUresult()> entry,
                   std::shared_ptr<CudaExecutionDomainOwner> physical)
        : dispatch(functions), native_entry(std::move(entry)), domain(std::move(physical)) {}
    CUresult gate() const noexcept {
        if (::getpid() != process || !process_memory_native_thread ||
            thread.lock() != process_memory_native_thread)
            return CUDA_ERROR_INVALID_CONTEXT;
        if (active) return CUDA_ERROR_NOT_READY;
        if (uncertain) return CUDA_ERROR_UNKNOWN;
        return native_entry ? native_entry() : CUDA_SUCCESS;
    }
    CUresult prove(CUcontext expected) noexcept {
        CUcontext observed{};
        const auto status = dispatch.context_get_current(&observed);
        if (status != CUDA_SUCCESS || observed != expected) {
            uncertain = true;
            return status == CUDA_SUCCESS ? CUDA_ERROR_INVALID_CONTEXT : status;
        }
        return CUDA_SUCCESS;
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
};

struct CudaProcessMemoryBackend::Implementation {
    struct Charge {
        std::uint64_t id{};
        std::size_t bytes{};
    };
    const CudaProcessMemoryConfig config;
    const std::shared_ptr<CudaProcessMemoryOwner> owner;
    const std::shared_ptr<CudaProcessMemoryDevice> device;
    std::vector<Charge> charges;
    Implementation(CudaProcessMemoryConfig settings, std::shared_ptr<core::QuotaLedger> quota,
                   std::shared_ptr<CudaProcessMemoryDevice> physical)
        : config(settings),
          owner(std::make_shared<CudaProcessMemoryOwner>(settings, std::move(quota))),
          device(std::move(physical)) {
        charges.reserve(config.maximum_allocations);
    }
    CudaProcessMemoryReply failed(CUresult status) const noexcept {
        return {.status = status,
                .is_quarantined =
                    owner->snapshot().is_quarantined || device->implementation_->uncertain};
    }
    CUresult refund_domain(std::uint64_t id) noexcept {
        const auto found = std::ranges::find(charges, id, &Charge::id);
        if (found == charges.end()) return CUDA_ERROR_INVALID_HANDLE;
        auto& native = *device->implementation_;
        if (found->bytes && native.domain) {
            const auto result = native.domain->release_process_bytes(found->bytes);
            if (result != CUDA_SUCCESS) return owner->quarantine(result);
        }
        charges.erase(found);
        return CUDA_SUCCESS;
    }
    CUresult initial_context(CUcontext& context) noexcept {
        auto& native = *device->implementation_;
        auto status = native.dispatch.context_get_current(&context);
        if (status == CUDA_ERROR_NOT_INITIALIZED && !context) {
            status = native.dispatch.initialize(0);
            if (status != CUDA_SUCCESS) return status;
            return native.prove(nullptr);
        }
        return status;
    }
    CUresult ensure_current(CUcontext context) noexcept {
        if (owner->snapshot().current_pool) return CUDA_SUCCESS;
        auto& native = *device->implementation_;
        CUmemoryPool pool{};
        const auto original = native.dispatch.default_pool(&pool, config.device_ordinal);
        const auto proof = native.prove(context);
        auto published =
            owner->publish_pool(CudaProcessPoolQuery::kDefault,
                                {original, proof, {CUDA_SUCCESS, original == CUDA_SUCCESS}, pool});
        if (published.status != CUDA_SUCCESS) return published.status;
        published =
            owner->publish_pool(CudaProcessPoolQuery::kCurrent,
                                {original, proof, {CUDA_SUCCESS, original == CUDA_SUCCESS}, pool});
        return published.status;
    }
    CUresult select_logical(CUcontext context, CUmemoryPool& previous) noexcept {
        auto& native = *device->implementation_;
        auto status = ensure_current(context);
        if (status != CUDA_SUCCESS) return status;
        status = native.dispatch.current_pool(&previous, config.device_ordinal);
        const auto proof = native.prove(context);
        if (proof != CUDA_SUCCESS) return owner->quarantine(proof);
        if (status != CUDA_SUCCESS || !previous) return owner->quarantine(status);
        const auto logical = owner->pool(owner->snapshot().current_pool);
        if (logical.status != CUDA_SUCCESS) return logical.status;
        if (previous == logical.native) return CUDA_SUCCESS;
        status = native.dispatch.select_pool(config.device_ordinal, logical.native);
        const auto selected = native.prove(context);
        if (status != CUDA_SUCCESS || selected != CUDA_SUCCESS) {
            native.uncertain = true;
            return owner->quarantine(selected != CUDA_SUCCESS ? selected : status);
        }
        return prove_pool(context, logical.native);
    }
    CUresult prove_pool(CUcontext context, CUmemoryPool expected) noexcept {
        auto& native = *device->implementation_;
        CUmemoryPool actual{};
        const auto queried = native.dispatch.current_pool(&actual, config.device_ordinal);
        const auto proof = native.prove(context);
        if (queried != CUDA_SUCCESS || proof != CUDA_SUCCESS || actual != expected) {
            native.uncertain = true;
            return owner->quarantine(proof != CUDA_SUCCESS     ? proof
                                     : queried != CUDA_SUCCESS ? queried
                                                               : CUDA_ERROR_UNKNOWN);
        }
        return CUDA_SUCCESS;
    }
    CUresult restore(CUcontext context, CUmemoryPool previous) noexcept {
        auto& native = *device->implementation_;
        const auto before = native.prove(context);
        if (before != CUDA_SUCCESS) return owner->quarantine(before);
        const auto result = native.dispatch.select_pool(config.device_ordinal, previous);
        const auto proof = native.prove(context);
        CUmemoryPool observed{};
        const auto queried = proof == CUDA_SUCCESS && result == CUDA_SUCCESS
                                 ? native.dispatch.current_pool(&observed, config.device_ordinal)
                                 : CUDA_ERROR_UNKNOWN;
        const auto final = native.prove(context);
        if (result != CUDA_SUCCESS || proof != CUDA_SUCCESS || queried != CUDA_SUCCESS ||
            final != CUDA_SUCCESS || observed != previous) {
            native.uncertain = true;
            return owner->quarantine(CUDA_ERROR_UNKNOWN);
        }
        return CUDA_SUCCESS;
    }
};

CudaProcessMemoryDevice::CudaProcessMemoryDevice(CudaProcessMemoryDispatch dispatch,
                                                 std::function<CUresult()> native_entry,
                                                 std::shared_ptr<CudaExecutionDomainOwner> domain) {
    if (!dispatch.initialize || !dispatch.context_get_current || !dispatch.default_pool ||
        !dispatch.current_pool || !dispatch.select_pool || !dispatch.allocate ||
        !dispatch.allocate_from_pool || !dispatch.trim || !dispatch.release)
        throw std::invalid_argument("incomplete process memory dispatch");
    if (!process_memory_native_thread) process_memory_native_thread = std::make_shared<int>(0);
    implementation_ =
        std::make_unique<Implementation>(dispatch, std::move(native_entry), std::move(domain));
}
CudaProcessMemoryDevice::~CudaProcessMemoryDevice() = default;
CUresult CudaProcessMemoryDevice::native_health() const noexcept {
    const auto& state = *implementation_;
    if (::getpid() != state.process || !process_memory_native_thread ||
        state.thread.lock() != process_memory_native_thread)
        return CUDA_ERROR_INVALID_CONTEXT;
    if (state.uncertain) return CUDA_ERROR_UNKNOWN;
    return state.active ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
}
CudaProcessMemoryBackend::CudaProcessMemoryBackend(
    CudaProcessMemoryConfig config, std::shared_ptr<core::QuotaLedger> quota,
    std::shared_ptr<CudaProcessMemoryDevice> device) {
    if (!device) throw std::invalid_argument("missing process memory device");
    implementation_ = std::make_unique<Implementation>(config, std::move(quota), std::move(device));
}
CudaProcessMemoryBackend::~CudaProcessMemoryBackend() = default;
std::shared_ptr<CudaProcessMemoryOwner> CudaProcessMemoryBackend::owner() const noexcept {
    return implementation_->owner;
}
CUresult CudaProcessMemoryBackend::native_health() const noexcept {
    return implementation_->device->native_health();
}
CudaProcessMemoryReply CudaProcessMemoryBackend::pool_control(
    const CudaProcessMemoryRequest& input) noexcept {
    auto& state = *implementation_;
    auto& native = *state.device->implementation_;
    if (const auto status = native.gate(); status != CUDA_SUCCESS) return state.failed(status);
    if (state.owner->snapshot().status != CUDA_SUCCESS) return state.failed(CUDA_ERROR_UNKNOWN);
    const bool query = input.operation == CudaProcessMemoryOperation::kGetDefaultPool ||
                       input.operation == CudaProcessMemoryOperation::kGetCurrentPool;
    if (!query && input.operation != CudaProcessMemoryOperation::kSetCurrentPool &&
        input.operation != CudaProcessMemoryOperation::kTrimPool)
        return state.failed(CUDA_ERROR_INVALID_VALUE);
    if (input.operation != CudaProcessMemoryOperation::kTrimPool &&
        input.device != state.config.device_ordinal)
        return state.failed(CUDA_ERROR_INVALID_DEVICE);
    CUmemoryPool selected{};
    if (input.pool) {
        const auto pool = state.owner->pool(input.pool);
        if (pool.status != CUDA_SUCCESS) return state.failed(pool.status);
        selected = pool.native;
    }
    std::shared_ptr<CudaNativeOutputCell> cell;
    if (input.has_output) {
        auto created =
            CudaNativeOutputCell::create(std::as_bytes(std::span{&input.output_seed, 1}));
        if (created.status != CUDA_SUCCESS) return state.failed(created.status);
        cell = std::move(created.cell);
    }
    const CudaProcessMemoryDevice::Implementation::Scope scope(native);
    CUcontext context{};
    if (const auto status = state.initial_context(context); status != CUDA_SUCCESS)
        return state.failed(status);
    CUmemoryPool previous{};
    if (const auto status = state.select_logical(context, previous); status != CUDA_SUCCESS)
        return state.failed(status);
    CudaProcessMemoryReply result;
    auto* output = cell ? static_cast<CUmemoryPool*>(cell->data()) : nullptr;
    if (query) {
        const auto status = input.operation == CudaProcessMemoryOperation::kGetDefaultPool
                                ? native.dispatch.default_pool(output, input.device)
                                : native.dispatch.current_pool(output, input.device);
        result.original_status = static_cast<std::int32_t>(status);
        result.status = status;
    } else if (input.operation == CudaProcessMemoryOperation::kSetCurrentPool) {
        const auto status = native.dispatch.select_pool(input.device, selected);
        result.original_status = static_cast<std::int32_t>(status);
        result.status = status;
    } else {
        result.original_status = static_cast<std::int32_t>(native.dispatch.trim(
            reinterpret_cast<cudaMemPool_t>(selected), static_cast<std::size_t>(input.bytes)));
        result.status = CUDA_SUCCESS;
    }
    auto proof = native.prove(context);
    CudaNativeOutputObservation observed{CUDA_SUCCESS, false};
    if (cell) observed = cell->snapshot(std::as_writable_bytes(std::span{&result.output, 1}));
    result.output_written = observed.status == CUDA_SUCCESS && observed.was_written;
    if (proof == CUDA_SUCCESS) proof = observed.status;
    if (proof == CUDA_SUCCESS && query) {
        // These bits were captured from this SDK output cell, not a client
        // capability or fabricated address; restore the original opaque type.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto pool = reinterpret_cast<CUmemoryPool>(result.output);
        if (*result.original_status == CUDA_SUCCESS) {
            auto published = state.owner->publish_pool(
                input.operation == CudaProcessMemoryOperation::kGetDefaultPool
                    ? CudaProcessPoolQuery::kDefault
                    : CudaProcessPoolQuery::kCurrent,
                {CUDA_SUCCESS, proof, observed, pool});
            result.id = published.id;
            proof = published.status;
        } else if (result.output_written && pool) {
            const auto known = state.owner->pool_identity(pool);
            if (known)
                result.id = *known;
            else
                proof = state.owner->quarantine(CUDA_ERROR_UNKNOWN);
        }
    } else if (proof == CUDA_SUCCESS &&
               input.operation == CudaProcessMemoryOperation::kSetCurrentPool) {
        CUmemoryPool actual{};
        const auto queried = native.dispatch.current_pool(&actual, input.device);
        proof = native.prove(context);
        if (proof == CUDA_SUCCESS &&
            (queried != CUDA_SUCCESS || !state.owner->pool_identity(actual) ||
             (*result.original_status == CUDA_SUCCESS && (!selected || actual != selected))))
            proof = state.owner->quarantine(CUDA_ERROR_UNKNOWN);
        if (proof == CUDA_SUCCESS)
            // Publish the independent getter's actual observation, including a
            // setter effect followed by an earlier asynchronous error. Never
            // replace that setter's original error with successful provenance.
            proof = state.owner
                        ->publish_pool(CudaProcessPoolQuery::kCurrent,
                                       {queried, proof, {CUDA_SUCCESS, true}, actual})
                        .status;
    }
    const auto restored = state.restore(context, previous);
    if (restored != CUDA_SUCCESS) proof = restored;
    if (proof != CUDA_SUCCESS) {
        result.status = proof;
        (void)state.owner->quarantine(proof);
    }
    result.is_quarantined = state.owner->snapshot().is_quarantined || native.uncertain;
    return result;
}
CudaProcessMemoryResult CudaProcessMemoryBackend::reserve(
    const CudaProcessMemoryRequest& input) noexcept {
    auto& state = *implementation_;
    if (input.operation != CudaProcessMemoryOperation::kAllocateAsync &&
        input.operation != CudaProcessMemoryOperation::kAllocateFromPoolAsync)
        return {};
    auto reserved = state.owner->reserve(static_cast<std::size_t>(input.bytes), input.pool);
    if (reserved.status != CUDA_SUCCESS) return reserved;
    if (reserved.id >= kProcessTag) {
        (void)state.owner->cancel(reserved.id);
        return {CUDA_ERROR_OUT_OF_MEMORY};
    }
    auto& native = *state.device->implementation_;
    if (input.bytes && native.domain) {
        const auto status =
            native.domain->reserve_process_bytes(static_cast<std::size_t>(input.bytes));
        if (status != CUDA_SUCCESS) {
            (void)state.owner->cancel(reserved.id);
            return {status};
        }
    }
    state.charges.push_back({reserved.id, static_cast<std::size_t>(input.bytes)});
    reserved.id |= kProcessTag;
    return reserved;
}
CUresult CudaProcessMemoryBackend::cancel(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    const auto status = state.owner->cancel(local_id(id));
    return status == CUDA_SUCCESS ? state.refund_domain(local_id(id)) : status;
}
CudaProcessMemoryReply CudaProcessMemoryBackend::allocate(
    std::uint64_t id, const CudaProcessMemoryRequest& input, CUstream stream,
    const std::shared_ptr<CudaNativeOutputCell>& output,
    std::shared_ptr<CudaProcessMemoryCompletion> tail) noexcept {
    auto& state = *implementation_;
    auto& native = *state.device->implementation_;
    if (const auto status = native.gate(); status != CUDA_SUCCESS) return state.failed(status);
    const CudaProcessMemoryDevice::Implementation::Scope scope(native);
    CUcontext context{};
    if (const auto status = state.initial_context(context); status != CUDA_SUCCESS)
        return state.failed(status);
    CUmemoryPool pool{};
    if (input.pool) {
        const auto mapped = state.owner->pool(input.pool);
        if (mapped.status != CUDA_SUCCESS) return state.failed(mapped.status);
        pool = mapped.native;
    }
    CUmemoryPool previous{};
    if (const auto status = state.select_logical(context, previous); status != CUDA_SUCCESS)
        return state.failed(status);
    const auto begun = state.owner->begin_allocation(local_id(id));
    if (begun != CUDA_SUCCESS) {
        (void)state.restore(context, previous);
        return state.failed(begun);
    }
    auto* address = output ? static_cast<CUdeviceptr*>(output->data()) : nullptr;
    const auto original =
        input.operation == CudaProcessMemoryOperation::kAllocateAsync
            ? native.dispatch.allocate(address, static_cast<std::size_t>(input.bytes), stream)
            : native.dispatch.allocate_from_pool(address, static_cast<std::size_t>(input.bytes),
                                                 pool, stream);
    CudaProcessMemoryReply result{.status = original,
                                  .original_status = static_cast<std::int32_t>(original)};
    auto proof = native.prove(context);
    CudaNativeOutputObservation observed{CUDA_SUCCESS, false};
    if (output) observed = output->snapshot(std::as_writable_bytes(std::span{&result.output, 1}));
    result.output_written = observed.status == CUDA_SUCCESS && observed.was_written;
    const auto restored = state.restore(context, previous);
    if (restored != CUDA_SUCCESS) proof = restored;
    const auto published = state.owner->publish_allocation(
        local_id(id), {original, proof, observed, result.output, std::move(tail)});
    result.status = published.status;
    if (published.id) result.id = published.id | kProcessTag;
    if (state.owner->snapshot().status == CUDA_SUCCESS && !state.owner->allocation(local_id(id))) {
        const auto refunded = state.refund_domain(local_id(id));
        if (refunded != CUDA_SUCCESS) result.status = refunded;
    }
    result.is_quarantined = state.owner->snapshot().is_quarantined || native.uncertain;
    return result;
}
CudaProcessMemoryReply CudaProcessMemoryBackend::release(std::uint64_t id) noexcept {
    auto& state = *implementation_;
    auto& native = *state.device->implementation_;
    if (const auto status = native.gate(); status != CUDA_SUCCESS) return state.failed(status);
    const CudaProcessMemoryDevice::Implementation::Scope scope(native);
    CUcontext context{};
    if (const auto status = state.initial_context(context); status != CUDA_SUCCESS)
        return state.failed(status);
    const auto begun =
        state.owner->begin_release(local_id(id), CudaProcessMemoryFreeKind::kSynchronous);
    if (begun.status != CUDA_SUCCESS) return state.failed(begun.status);
    const auto original = native.dispatch.release(begun.address);
    const auto proof = native.prove(context);
    const auto published = state.owner->publish_release(local_id(id), original, proof);
    CudaProcessMemoryReply result{.status = published.status,
                                  .original_status = static_cast<std::int32_t>(original)};
    if (published.status == CUDA_SUCCESS) result.status = state.refund_domain(local_id(id));
    result.is_quarantined = state.owner->snapshot().is_quarantined || native.uncertain;
    return result;
}
CUresult CudaProcessMemoryBackend::progress() noexcept {
    return implementation_->owner->progress();
}
CUresult CudaProcessMemoryBackend::close() noexcept {
    return implementation_->owner->close();
}
CUresult CudaProcessMemoryBackend::shutdown() noexcept {
    return implementation_->owner->shutdown();
}
bool CudaProcessMemoryBackend::has_allocations() const noexcept {
    return implementation_->owner->snapshot().allocations != 0;
}

}  // namespace glimmer::backend::cuda
