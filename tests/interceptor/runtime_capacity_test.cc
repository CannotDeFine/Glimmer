#include "internal/driver_dispatch.h"
#include "internal/runtime_api_bridge.h"
#include "internal/driver_api_interceptor.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

extern "C" unsigned int glimmer_test_capacity_allocation_calls();
extern "C" unsigned int glimmer_test_capacity_memory_queries();
extern "C" bool glimmer_fake_runtime_capacity_available();

namespace {
using namespace glimmer::interceptor;
constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

unsigned int release_calls{};
cudaError_t release_result{cudaSuccess};
void* expected_pointer{};

cudaError_t release_original(void* pointer) {
    require(pointer == expected_pointer, "release original receives the unchanged pointer");
    ++release_calls;
    return release_result;
}

void check_unbound_release(bool is_null) {
    require(setenv("GLIMMER_FAKE_UNBOUND_CAPACITY", "1", 1) == 0,
            "select a genuinely empty fake caller context");
    unsigned int token{};
    expected_pointer = is_null ? nullptr : &token;
    for (const auto error : {cudaErrorInvalidValue, cudaErrorNotSupported}) {
        release_result = error;
        require(intercept_runtime_free(expected_pointer, release_original) == error,
                "original release rejection remains unchanged without an accounting effect");
    }
    release_result = cudaSuccess;
    require(intercept_runtime_free(expected_pointer, release_original) ==
                    (is_null ? cudaSuccess : cudaErrorUnknown) &&
                release_calls == 3,
            "null success is non-accounting; uncertain nonnull success remains fail-closed");
    require(unsetenv("GLIMMER_FAKE_UNBOUND_CAPACITY") == 0,
            "restore fake original allocation context");
    void* allocation{};
    const auto status = cudaMalloc(&allocation, 4096);
    require(status == (is_null ? cudaSuccess : cudaErrorUnknown),
            "only an uncertain nonnull release poisons later allocation accounting");
    if (is_null) {
        std::size_t free_bytes{}, total_bytes{};
        require(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
                    total_bytes == kQuotaBytes && free_bytes == kQuotaBytes - 4096 &&
                    intercept_runtime_free(nullptr, release_original) == cudaSuccess &&
                    release_calls == 4 &&
                    cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
                    free_bytes == kQuotaBytes - 4096 && cudaFree(allocation) == cudaSuccess,
                "warm null free cannot retire a live charge and tracked release still works");
    }
    require(!is_inside_driver_call() && !is_inside_runtime_call(),
            "release success/error leaves caller scopes balanced");
}

void run() {
    require(glimmer_fake_runtime_capacity_available(), "original fake SDK is independently linked");
    void* output{};
    require(cudaMalloc(&output, kQuotaBytes + 1) == cudaErrorMemoryAllocation && !output &&
                glimmer_test_capacity_allocation_calls() == 0 &&
                glimmer_test_capacity_memory_queries() == 0,
            "cold quota rejection performs neither SDK allocation nor unsafe free query");
    require(cudaMalloc(&output, 1536) == cudaErrorMemoryAllocation && !output &&
                glimmer_test_capacity_allocation_calls() == 1 &&
                glimmer_test_capacity_memory_queries() == 0,
            "cold original allocation error is unchanged and reserves no committed charge");
    DriverDispatch driver;
    require(driver.initialize(), "resolve independent fake Driver metadata");
    CUcontext before{};
    require(driver.context_get_current(&before) == CUDA_SUCCESS && before,
            "cold nonnull context alone is not free-query readiness proof");
    require(cudaMalloc(&output, 4096) == cudaSuccess && output &&
                glimmer_test_capacity_allocation_calls() == 2 &&
                glimmer_test_capacity_memory_queries() == 0,
            "original allocation alone initializes cold state, without an extra prequery");
    CUcontext after{};
    require(driver.context_get_current(&after) == CUDA_SUCCESS && before == after,
            "capacity resolution never replaces the caller context");
    std::size_t free_bytes{};
    std::size_t total_bytes{};
    require(cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
                total_bytes == kQuotaBytes && free_bytes == kQuotaBytes - 4096,
            "native failure rolls back reservation and success charges exactly once");
    void* vendor = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    const auto resolve = resolve_real_dlsym();
    require(vendor && resolve, "observe already linked fake Driver originals");
    const auto allocate = reinterpret_cast<MemAllocFunction>(resolve(vendor, "cuMemAlloc_v2"));
    const auto release = reinterpret_cast<MemFreeFunction>(resolve(vendor, "cuMemFree_v2"));
    require(allocate && release, "resolve external physical occupancy, not a quota wrapper");
    CUdeviceptr external{};
    {
        const NativeDriverCallScope native;
        require(allocate(&external, std::size_t{57} * 1024 * 1024) == CUDA_SUCCESS && external,
                "external native allocation occupies physical memory only");
    }
    const auto allocations = glimmer_test_capacity_allocation_calls();
    const auto queries = glimmer_test_capacity_memory_queries();
    void* rejected{};
    require(cudaMalloc(&rejected, std::size_t{7} * 1024 * 1024) == cudaErrorMemoryAllocation &&
                !rejected && glimmer_test_capacity_allocation_calls() == allocations &&
                glimmer_test_capacity_memory_queries() == queries + 1,
            "ready-context physical-free rejection still precedes original allocation");
    {
        const NativeDriverCallScope native;
        require(release(external) == CUDA_SUCCESS, "release external physical occupancy");
    }
    require(cudaFree(output) == cudaSuccess &&
                cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess &&
                total_bytes == kQuotaBytes && free_bytes == kQuotaBytes &&
                !is_inside_driver_call() && !is_inside_runtime_call(),
            "successful release restores all quota and balanced caller TLS");
    require(dlclose(vendor) == 0, "balance fake Driver observation reference");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2) {
            const auto mode = std::string_view(argv[1]);
            require(mode == "--null-free" || mode == "--unknown-free", "known release mode");
            check_unbound_release(mode == "--null-free");
            return 0;
        }
        run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
