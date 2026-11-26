#include "internal/cuda_cooperative_pool.h"
#include "internal/driver_api_interceptor.h"
#include "internal/runtime_api_bridge.h"

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {
using namespace glimmer::interceptor;
enum class Action { kAllocate, kRelease };
Action action{Action::kAllocate};
CUdeviceptr pointer{};
std::size_t memory_bytes{1024};
CudaCooperativeReleaseResult released;
bool has_runtime_scope{};

CUresult invoke_private_hook() {
    void* address = nullptr;
    return runtime_driver_get_proc_address("glimmer_test_progress_accounting_callback", &address,
                                           CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT);
}
bool memory_is(std::size_t expected_free) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    return intercept_mem_get_info(&free_bytes, &total_bytes) == CUDA_SUCCESS &&
           total_bytes == 4096 && free_bytes == expected_free;
}
bool run(std::string_view mode) {
    if (intercept_init(0) != CUDA_SUCCESS || !memory_is(4096) ||
        invoke_private_hook() != CUDA_SUCCESS || !has_runtime_scope || !pointer ||
        !memory_is(3072)) {
        return false;
    }
    const auto original = pointer;
    if (mode == "--uncertain-allocation") {
        if (setenv("GLIMMER_FAKE_PROGRESS_ALLOCATION_ERROR", "1", 1) != 0) {
            return false;
        }
        return invoke_private_hook() == CUDA_ERROR_INVALID_VALUE && pointer != 0 &&
               allocate_cooperative_progress(&pointer, 1) == CUDA_ERROR_UNKNOWN && pointer == 0;
    }
    if (mode == "--rollback") {
        if (setenv("GLIMMER_FAKE_DUPLICATE_POINTER", "1", 1) != 0) {
            return false;
        }
        return invoke_private_hook() == CUDA_ERROR_OUT_OF_MEMORY && pointer == 0 &&
               allocate_cooperative_progress(&pointer, 1) == CUDA_ERROR_UNKNOWN;
    }
    if (mode == "--released-accounting-error") {
        if (setenv("GLIMMER_FAKE_PROGRESS_CONTEXT_ERROR", "1", 1) != 0) {
            return false;
        }
        action = Action::kRelease;
        return invoke_private_hook() == CUDA_ERROR_UNKNOWN &&
               released.ownership == CudaNativeReleaseState::kReleased;
    }
    memory_bytes = 4096;
    if (invoke_private_hook() != CUDA_ERROR_OUT_OF_MEMORY || pointer != 0 || !memory_is(3072)) {
        return false;
    }
    memory_bytes = 1536;
    if (invoke_private_hook() != CUDA_ERROR_INVALID_VALUE || pointer != 0 || !memory_is(3072)) {
        return false;
    }
    pointer = original;
    action = Action::kRelease;
    if (invoke_private_hook() != CUDA_SUCCESS ||
        released.ownership != CudaNativeReleaseState::kReleased || !memory_is(4096)) {
        return false;
    }
    // Unknown native error ownership cannot be relabeled as retained just
    // because the Driver status is INVALID_VALUE. Do not retry this release.
    pointer = 0xdeadbeef;
    return invoke_private_hook() == CUDA_ERROR_INVALID_VALUE &&
           released.ownership == CudaNativeReleaseState::kUncertain;
}
}  // namespace

extern "C" CUresult glimmer_test_progress_accounting_callback() {
    has_runtime_scope = is_inside_runtime_call();
    if (!has_runtime_scope) {
        return CUDA_ERROR_UNKNOWN;
    }
    if (action == Action::kAllocate) {
        return allocate_cooperative_progress(&pointer, memory_bytes);
    }
    released = release_cooperative_progress(pointer);
    return released.status;
}

extern "C" int glimmer_run_progress_accounting_fixture(const char* mode) {
    const bool passed = run(mode ? std::string_view{mode} : std::string_view{});
    if (!passed) {
        std::cerr << "private cooperative progress accounting failed\n";
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
