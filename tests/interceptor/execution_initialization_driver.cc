#include <cuda.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace {
std::atomic<bool> is_initialized{}, should_hold{}, has_entered{}, can_return{};
std::atomic<unsigned int> queries{};
}  // namespace
extern "C" {
CUresult CUDAAPI cuInit(unsigned int flags) {
    if (flags) return CUDA_ERROR_INVALID_VALUE;
    is_initialized.store(true, std::memory_order_release);
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxGetCurrent(CUcontext* output) {
    ++queries;
    if (should_hold.load(std::memory_order_acquire)) {
        has_entered.store(true, std::memory_order_release);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
        while (!can_return.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) return CUDA_ERROR_UNKNOWN;
            std::this_thread::yield();
        }
    }
    if (!is_initialized.load(std::memory_order_acquire)) return CUDA_ERROR_NOT_INITIALIZED;
    if (!output) return CUDA_ERROR_INVALID_VALUE;
    // Deliberate native handle witness: the public projection must never
    // publish this unrelated caller SDK value or import it as an owned context.
    *output = reinterpret_cast<CUcontext>(1);
    return CUDA_SUCCESS;
}
unsigned int glimmer_initialization_queries() {
    return queries.load();
}
void glimmer_initialization_hold() {
    should_hold.store(true, std::memory_order_release);
}
bool glimmer_initialization_entered() {
    return has_entered.load(std::memory_order_acquire);
}
void glimmer_initialization_release() {
    can_return.store(true, std::memory_order_release);
}
}
