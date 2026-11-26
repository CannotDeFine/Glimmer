#include "internal/cuda_process_memory_backend.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend::cuda;
constexpr auto kTag = UINT64_C(1) << 63;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
enum class SetterErrorEffect { kUnchanged, kRequested, kUnknown, kUnobserved };
struct Native {
    std::array<char, 4> handles{};
    CUcontext context{};
    CUmemoryPool current{}, allocated_pool{}, trimmed_pool{};
    std::map<CUdeviceptr, std::vector<std::byte>> allocations;
    CUresult allocation_status{CUDA_SUCCESS};
    cudaError_t trim_status{cudaSuccess};
    unsigned int calls{}, allocations_called{}, frees{}, selects{};
    bool blocked{}, error_writes_zero{}, ignores_selection{};
    bool inject_setter_error{}, fail_next_current_query{};
    CUmemoryPool setter_error_target{};
    SetterErrorEffect setter_error_effect{};
} native;
CUmemoryPool pool(unsigned int index) {
    return reinterpret_cast<CUmemoryPool>(&native.handles.at(index));
}
CUresult CUDAAPI initialize(unsigned int) {
    ++native.calls;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI context(CUcontext* output) {
    ++native.calls;
    *output = native.context;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI default_pool(CUmemoryPool* output, CUdevice device) {
    ++native.calls;
    if (!output || device) return CUDA_ERROR_INVALID_VALUE;
    *output = pool(0);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI current_pool(CUmemoryPool* output, CUdevice device) {
    ++native.calls;
    if (!output || device) return CUDA_ERROR_INVALID_VALUE;
    if (native.fail_next_current_query) {
        native.fail_next_current_query = false;
        return CUDA_ERROR_UNKNOWN;
    }
    *output = native.current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI select_pool(CUdevice device, CUmemoryPool selected) {
    ++native.calls;
    ++native.selects;
    if (device || !selected) return CUDA_ERROR_INVALID_VALUE;
    if (native.inject_setter_error && selected == native.setter_error_target) {
        native.inject_setter_error = false;
        if (native.setter_error_effect == SetterErrorEffect::kRequested)
            native.current = selected;
        else if (native.setter_error_effect == SetterErrorEffect::kUnknown)
            native.current = pool(3);
        else if (native.setter_error_effect == SetterErrorEffect::kUnobserved) {
            native.current = selected;
            native.fail_next_current_query = true;
        }
        return CUDA_ERROR_LAUNCH_FAILED;
    }
    if (!native.ignores_selection) native.current = selected;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI allocate_from(CUdeviceptr* output, std::size_t bytes, CUmemoryPool selected,
                               CUstream) {
    ++native.calls;
    ++native.allocations_called;
    native.allocated_pool = selected;
    if (!output || !selected) return CUDA_ERROR_INVALID_VALUE;
    if (native.allocation_status != CUDA_SUCCESS) {
        if (native.error_writes_zero) *output = 0;
        return native.allocation_status;
    }
    if (!bytes) {
        *output = 0;
        return CUDA_SUCCESS;
    }
    std::vector<std::byte> storage(bytes, std::byte{0x7b});
    *output = reinterpret_cast<CUdeviceptr>(storage.data());
    native.allocations.emplace(*output, std::move(storage));
    return CUDA_SUCCESS;
}
CUresult CUDAAPI allocate(CUdeviceptr* output, std::size_t bytes, CUstream stream) {
    return allocate_from(output, bytes, native.current, stream);
}
cudaError_t CUDARTAPI trim(cudaMemPool_t selected, std::size_t) {
    ++native.calls;
    native.trimmed_pool = reinterpret_cast<CUmemoryPool>(selected);
    return native.trim_status;
}
CUresult CUDAAPI release(CUdeviceptr address) {
    ++native.calls;
    if (!native.context || native.allocations.erase(address) != 1) return CUDA_ERROR_INVALID_VALUE;
    ++native.frees;
    return CUDA_SUCCESS;
}
struct Tail final : CudaProcessMemoryCompletion {
    bool ready{};
    unsigned int observations{};
    CUresult observe() noexcept override {
        ++observations;
        return ready ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    }
};
CudaProcessMemoryReply allocated(CudaProcessMemoryBackend& backend,
                                 CudaProcessMemoryRequest request,
                                 const std::shared_ptr<Tail>& tail) {
    auto cell = CudaNativeOutputCell::create(std::as_bytes(std::span{&request.output_seed, 1}));
    expect(cell.status == CUDA_SUCCESS, "real Linux COW output cell");
    const auto reserved = backend.reserve(request);
    expect(reserved.status == CUDA_SUCCESS && (reserved.id & kTag), "tagged reservation");
    return backend.allocate(reserved.id, request, nullptr, request.has_output ? cell.cell : nullptr,
                            tail);
}
void complete_path() {
    native = {};
    native.current = pool(2);  // Independent native actor's actual selection.
    auto device = std::make_shared<CudaProcessMemoryDevice>(
        CudaProcessMemoryDispatch{initialize, context, default_pool, current_pool, select_pool,
                                  allocate, allocate_from, trim, release},
        [] { return native.blocked ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS; });
    auto quota = std::make_shared<glimmer::core::QuotaLedger>(128);
    CudaProcessMemoryBackend first({1, 0}, quota, device);
    CudaProcessMemoryBackend second({2, 0}, std::make_shared<glimmer::core::QuotaLedger>(128),
                                    device);
    const CudaProcessMemoryRequest query{CudaProcessMemoryOperation::kGetDefaultPool, 0, true};
    const auto a = first.pool_control(query);
    expect(a.status == CUDA_SUCCESS && a.original_status == CUDA_SUCCESS && a.id &&
               a.output_written && a.output == reinterpret_cast<std::uintptr_t>(pool(0)) &&
               native.current == pool(2),
           "default pool has authority and restores actual selection");
    // The seam supplies a second independently trusted native observation. This
    // is not a caller-provided pool pointer or a new public pool-creation API.
    const auto b =
        first.owner()->publish_pool(CudaProcessPoolQuery::kCurrent,
                                    {CUDA_SUCCESS, CUDA_SUCCESS, {CUDA_SUCCESS, true}, pool(1)});
    expect(b.status == CUDA_SUCCESS && b.id && b.id != a.id, "distinct observed pool capabilities");
    auto current = query;
    current.operation = CudaProcessMemoryOperation::kGetCurrentPool;
    auto observed = first.pool_control(current);
    expect(observed.id == b.id && observed.output == reinterpret_cast<std::uintptr_t>(pool(1)) &&
               native.current == pool(2),
           "logical selected pool materializes and restores");
    auto selected =
        first.pool_control({CudaProcessMemoryOperation::kSetCurrentPool, 0, false, a.id});
    expect(selected.original_status == CUDA_SUCCESS && native.current == pool(2),
           "actual setter restores external actor");
    expect(first.pool_control(current).id == a.id, "setter changes subsequent logical getter");
    expect(
        first.pool_control({CudaProcessMemoryOperation::kSetCurrentPool, 0, false, b.id}).status ==
            CUDA_SUCCESS,
        "select independently observed second pool");
    expect(second.pool_control(current).output == reinterpret_cast<std::uintptr_t>(pool(0)) &&
               first.pool_control(current).id == b.id && native.current == pool(2),
           "two logical processes cannot inherit each other's pool selection");
    const auto calls = native.calls;
    native.blocked = true;
    expect(first.pool_control(current).status == CUDA_ERROR_NOT_READY && native.calls == calls,
           "actual native actor exclusion rejects before any SDK call");
    native.blocked = false;
    native.trim_status = cudaErrorInvalidResourceHandle;
    const auto trimmed =
        first.pool_control({CudaProcessMemoryOperation::kTrimPool, 0, false, b.id, 4096});
    expect(trimmed.status == CUDA_SUCCESS &&
               trimmed.original_status == cudaErrorInvalidResourceHandle &&
               native.trimmed_pool == pool(1) && native.current == pool(2),
           "Runtime trim status is not Driver mapped");
    expect(first.owner()->reserve_context_bytes(96) == CUDA_SUCCESS,
           "classic bytes share process quota");
    expect(first.reserve({CudaProcessMemoryOperation::kAllocateAsync, 0, true, 0, 33}).status ==
               CUDA_ERROR_OUT_OF_MEMORY,
           "pooled bytes cannot obtain a second independent budget");
    expect(first.owner()->release_context_bytes(96) == CUDA_SUCCESS,
           "classic release refunds exact shared charge");
    auto tail = std::make_shared<Tail>();
    auto result =
        allocated(first, {CudaProcessMemoryOperation::kAllocateAsync, 0, true, 0, 64}, tail);
    expect(result.original_status == CUDA_SUCCESS && result.status == CUDA_SUCCESS && result.id &&
               result.output && native.allocated_pool == pool(1) && native.current == pool(2),
           "allocation uses logical current pool and captured original pointer");
    const auto raw = result.id & ~kTag;
    auto borrower = first.owner()->borrow(raw, 7, 32);
    expect(borrower.status == CUDA_SUCCESS && tail->observations == 0,
           "consumer enqueue borrow never waits for allocator physical completion");
    native.context = reinterpret_cast<CUcontext>(&native.handles[0]);
    expect(first.release(result.id).status == CUDA_ERROR_NOT_READY && native.frees == 0,
           "synchronous free waits actual allocator tail");
    tail->ready = true;
    expect(first.progress() == CUDA_SUCCESS, "allocator source lifetime retires independently");
    native.context = reinterpret_cast<CUcontext>(&native.handles[1]);
    expect(first.release(result.id).status == CUDA_ERROR_NOT_READY && native.frees == 0,
           "cross-context release waits live consumer pin");
    borrower.lifetime.reset();
    expect(first.release(result.id).original_status == CUDA_SUCCESS && native.frees == 1 &&
               quota->usage().used_bytes() == 0,
           "second context original synchronous free exactly once");
    result = allocated(
        first, {CudaProcessMemoryOperation::kAllocateFromPoolAsync, 0, true, a.id, 16}, tail);
    expect(result.status == CUDA_SUCCESS && native.allocated_pool == pool(0),
           "explicit pool allocation does not use current pool");
    expect(first.release(result.id).status == CUDA_SUCCESS && native.frees == 2,
           "explicit pool release");
    result = allocated(first, {CudaProcessMemoryOperation::kAllocateAsync, 0, true}, tail);
    expect(result.status == CUDA_SUCCESS && result.output_written && !result.output && !result.id &&
               quota->usage().used_bytes() == 0,
           "zero allocation preserves actual null output without range");
    native.allocation_status = CUDA_ERROR_INVALID_VALUE;
    native.error_writes_zero = true;
    result =
        allocated(first, {CudaProcessMemoryOperation::kAllocateAsync, 0, true, 0, 16, 0, 99}, tail);
    expect(result.original_status == CUDA_ERROR_INVALID_VALUE && result.output_written &&
               !result.output && !result.id && !result.is_quarantined &&
               quota->usage().used_bytes() == 0,
           "clean native error retains written output independently of success");
    bool rejected{};
    std::thread foreign(
        [&] { rejected = first.pool_control(current).status == CUDA_ERROR_INVALID_CONTEXT; });
    foreign.join();
    expect(rejected, "foreign thread cannot enter process device SDK actor");
    expect(first.close() == CUDA_SUCCESS && first.shutdown() == CUDA_SUCCESS &&
               second.close() == CUDA_SUCCESS && second.shutdown() == CUDA_SUCCESS &&
               native.allocations.empty(),
           "ordinary teardown retains no resource or quota");
}
void setter_error_observation(SetterErrorEffect effect) {
    native = {};
    native.current = pool(2);
    auto device = std::make_shared<CudaProcessMemoryDevice>(
        CudaProcessMemoryDispatch{initialize, context, default_pool, current_pool, select_pool,
                                  allocate, allocate_from, trim, release},
        [] { return CUDA_SUCCESS; });
    auto quota = std::make_shared<glimmer::core::QuotaLedger>(128);
    CudaProcessMemoryBackend owner({8, 0}, quota, device);
    const auto first = owner.pool_control({CudaProcessMemoryOperation::kGetDefaultPool, 0, true});
    const auto second =
        owner.owner()->publish_pool(CudaProcessPoolQuery::kCurrent,
                                    {CUDA_SUCCESS, CUDA_SUCCESS, {CUDA_SUCCESS, true}, pool(1)});
    expect(first.status == CUDA_SUCCESS && second.status == CUDA_SUCCESS && first.id != second.id &&
               owner.pool_control({CudaProcessMemoryOperation::kSetCurrentPool, 0, false, first.id})
                       .status == CUDA_SUCCESS,
           "two trusted pool identities precede setter error controls");
    native.setter_error_target = pool(1);
    native.setter_error_effect = effect;
    native.inject_setter_error = true;
    const auto result =
        owner.pool_control({CudaProcessMemoryOperation::kSetCurrentPool, 0, false, second.id});
    expect(result.original_status == CUDA_ERROR_LAUNCH_FAILED && native.current == pool(2),
           "setter original error survives independent proof and external pool restoration");
    if (effect == SetterErrorEffect::kUnknown || effect == SetterErrorEffect::kUnobserved) {
        expect(result.is_quarantined && result.status == CUDA_ERROR_UNKNOWN && !result.id &&
                   !result.output_written,
               "unknown identity or failed effect getter cannot invent logical selection");
        const auto calls = native.calls;
        const auto rejected =
            owner.pool_control({CudaProcessMemoryOperation::kGetCurrentPool, 0, true});
        expect(rejected.is_quarantined && !rejected.original_status && native.calls == calls,
               "unproven logical selection never retries or enters another requested SDK");
        // Deliberate retained uncertainty belongs only to this isolated child.
        std::_Exit(0);
    }
    const bool changed = effect == SetterErrorEffect::kRequested;
    const auto expected_id = changed ? second.id : first.id;
    const auto expected_pool = changed ? pool(1) : pool(0);
    expect(result.status == CUDA_ERROR_LAUNCH_FAILED && !result.is_quarantined &&
               owner.pool_control({CudaProcessMemoryOperation::kGetCurrentPool, 0, true}).id ==
                   expected_id &&
               native.current == pool(2),
           "independent getter reconciles changed or unchanged logical selection on error");
    const auto invalid =
        owner.pool_control({CudaProcessMemoryOperation::kSetCurrentPool, 0, false, 0});
    expect(invalid.status == CUDA_ERROR_INVALID_VALUE &&
               invalid.original_status == CUDA_ERROR_INVALID_VALUE && !invalid.is_quarantined &&
               owner.pool_control({CudaProcessMemoryOperation::kGetCurrentPool, 0, true}).id ==
                   expected_id &&
               native.current == pool(2),
           "NULL setter retains original error with proven unchanged logical selection");
    auto tail = std::make_shared<Tail>();
    tail->ready = true;
    const auto allocation =
        allocated(owner, {CudaProcessMemoryOperation::kAllocateAsync, 0, true, 0, 16}, tail);
    expect(allocation.status == CUDA_SUCCESS && allocation.id &&
               native.allocated_pool == expected_pool && native.current == pool(2),
           "next actual allocation uses independently observed error-side selection");
    native.context = reinterpret_cast<CUcontext>(&native.handles[0]);
    expect(owner.release(allocation.id).status == CUDA_SUCCESS && native.frees == 1 &&
               quota->usage().used_bytes() == 0 && owner.close() == CUDA_SUCCESS &&
               owner.shutdown() == CUDA_SUCCESS && native.allocations.empty(),
           "known setter-error effects preserve normal exactly-once resource cleanup");
}
void setter_error_controls() {
    setter_error_observation(SetterErrorEffect::kUnchanged);
    setter_error_observation(SetterErrorEffect::kRequested);
    for (const auto effect : {SetterErrorEffect::kUnknown, SetterErrorEffect::kUnobserved}) {
        const auto child = ::fork();
        expect(child >= 0, "isolate uncertain setter effect control");
        if (!child) {
            try {
                setter_error_observation(effect);
                std::_Exit(1);
            } catch (const std::exception& error) {
                std::cerr << error.what() << '\n';
                std::_Exit(1);
            }
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "uncertain error-side logical selection remains quarantined");
    }
}
void ignored_materialization_is_rejected() {
    const auto child = ::fork();
    expect(child >= 0, "isolate deliberately unprovable native selection");
    if (!child) {
        try {
            native = {};
            native.current = pool(2);
            native.ignores_selection = true;
            auto device = std::make_shared<CudaProcessMemoryDevice>(
                CudaProcessMemoryDispatch{initialize, context, default_pool, current_pool,
                                          select_pool, allocate, allocate_from, trim, release},
                [] { return CUDA_SUCCESS; });
            CudaProcessMemoryBackend owner(
                {7, 0}, std::make_shared<glimmer::core::QuotaLedger>(128), device);
            const auto logical = owner.owner()->publish_pool(
                CudaProcessPoolQuery::kCurrent,
                {CUDA_SUCCESS, CUDA_SUCCESS, {CUDA_SUCCESS, true}, pool(1)});
            expect(logical.status == CUDA_SUCCESS, "independent trusted logical pool observation");
            const auto observed =
                owner.pool_control({CudaProcessMemoryOperation::kGetCurrentPool, 0, true});
            expect(observed.status == CUDA_ERROR_UNKNOWN && !observed.original_status &&
                       observed.is_quarantined && !observed.id && native.current == pool(2) &&
                       native.allocations_called == 0,
                   "setter success alone cannot grant wrong-owner pool use or requested SDK "
                   "provenance");
            const auto calls = native.calls;
            expect(owner.pool_control({CudaProcessMemoryOperation::kGetCurrentPool, 0, true})
                           .is_quarantined &&
                       native.calls == calls,
                   "unknown pool selection cannot replay or continue");
            std::_Exit(0);
        } catch (...) {
            std::_Exit(1);
        }
    }
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
           "ignored materialization retains uncertainty in bounded child control");
}
}  // namespace
int main() {
    try {
        complete_path();
        setter_error_controls();
        ignored_materialization_is_rejected();
        std::cout << "Process memory SDK ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
