#include "internal/persistent_sgemm.h"
#include "internal/cuda_cooperative_pool.h"

#include <cuda.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

namespace {
using glimmer::interceptor::PersistentSgemmProblem;
namespace interceptor = glimmer::interceptor;

bool check(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return true;
    std::cerr << operation << " failed with Driver status " << status << '\n';
    return false;
}
bool check_blas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::cerr << operation << " failed with cuBLAS status " << status << '\n';
    return false;
}

bool load_gemm(CUmodule& module) {
    std::array<char, 4096> log{};
    std::array<CUjit_option, 2> options{CU_JIT_ERROR_LOG_BUFFER,
                                        CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    std::array<void*, 2> values{log.data(), std::bit_cast<void*>(std::uintptr_t{log.size()})};
    const auto status = cuModuleLoadDataEx(&module, glimmer::interceptor::persistent_sgemm_ptx(),
                                           options.size(), options.data(), values.data());
    if (status != CUDA_SUCCESS) std::cerr << log.data() << '\n';
    return check(status, "JIT owned GEMM");
}

struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUstream stream{};
    cublasHandle_t blas{};
    void* retreat_host{};
    CUdeviceptr retreat_device{};
    void* first_tile_host{};
    CUdeviceptr first_tile_device{};
    std::vector<CUdeviceptr> allocations;
    std::shared_ptr<interceptor::CudaCooperativePool> pool;
    std::optional<interceptor::PreparedCooperativeKernel> prepared;

    ~Resources() {
        (void)cleanup();
    }
    bool cleanup() {
        bool passed = true;
        if (!context) return true;
        passed &= check(cuCtxSynchronize(), "drain test context");
        prepared.reset();
        if (pool) {
            passed &= pool->occupied_slots() == 0;
            passed &= check(pool->release(), "release cooperative progress pool");
            pool.reset();
        }
        if (blas) {
            passed &= check_blas(cublasDestroy(blas), "destroy reference handle");
            blas = nullptr;
        }
        if (module) {
            passed &= check(cuModuleUnload(module), "unload GEMM module");
            module = nullptr;
        }
        for (const auto address : allocations)
            passed &= check(cuMemFree(address), "free test allocation");
        allocations.clear();
        if (retreat_host) {
            passed &= check(cuMemFreeHost(retreat_host), "free retreat flag");
            retreat_host = nullptr;
        }
        if (first_tile_host) {
            passed &= check(cuMemFreeHost(first_tile_host), "free physical progress observation");
            first_tile_host = nullptr;
        }
        if (stream) {
            passed &= check(cuStreamDestroy(stream), "destroy test stream");
            stream = nullptr;
        }
        passed &= check(cuCtxDestroy(context), "destroy test context");
        context = nullptr;
        return passed;
    }
    bool allocate(CUdeviceptr& pointer, std::size_t bytes) {
        allocations.reserve(allocations.size() + 1);
        if (!check(cuMemAlloc(&pointer, bytes), "allocate test storage")) return false;
        allocations.push_back(pointer);
        return true;
    }
};

bool await_first_tile(void* first_tile_host) {
    auto& observed = *static_cast<std::uint32_t*>(first_tile_host);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (std::atomic_ref(observed).load(std::memory_order_acquire) == 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            std::cerr << "no physical tile observed before retreat deadline\n";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::microseconds{50});
    }
    return true;
}

struct PhysicalProgressWitness {
    std::uint64_t stopped_count{};
    std::uint32_t compute_witness{};
};

bool run_pooled(Resources& resources, CUfunction function,
                const interceptor::PersistentSgemmLayout& layout, void* const* arguments,
                PhysicalProgressWitness& observation) {
    const auto release_result = [](CUresult status) {
        return interceptor::CudaCooperativeReleaseResult{
            status, status == CUDA_SUCCESS ? interceptor::CudaNativeReleaseState::kReleased
                                           : interceptor::CudaNativeReleaseState::kUncertain};
    };
    interceptor::CudaCooperativePoolDispatch dispatch{
        &cuCtxGetCurrent,
        &cuMemAlloc,
        [release_result](CUdeviceptr pointer) { return release_result(cuMemFree(pointer)); },
        &cuMemHostAlloc,
        [release_result](void* pointer) { return release_result(cuMemFreeHost(pointer)); },
        &cuMemHostGetDevicePointer,
        &cuMemcpyDtoHAsync,
        std::make_shared<int>(0)};
    const auto total = layout.grid[0] * layout.grid[1];
    resources.pool = std::make_shared<interceptor::CudaCooperativePool>(
        resources.context, std::move(dispatch), 1, total);
    if (!check(resources.pool->initialize(), "initialize cooperative progress pool")) return false;
    const interceptor::CooperativeKernelVariant variant{
        resources.context,
        function,
        {interceptor::kPersistentSgemmParameters.begin(),
         interceptor::kPersistentSgemmParameters.end()},
        {total, 256},
        3,
        interceptor::CooperativeKernelVariant::AuditParameters{14, 15, true}};
    // The fixture retains the module until physical drain. This separate token
    // verifies that the production pool retains the supplied resource lease.
    auto code_lease = std::make_shared<int>(1);
    const std::weak_ptr<int> weak_lease = code_lease;
    resources.prepared =
        resources.pool->prepare(variant, layout.grid, layout.block, arguments, code_lease);
    code_lease.reset();
    if (!resources.prepared || weak_lease.expired() || resources.pool->occupied_slots() != 1)
        return false;
    auto& prepared = *resources.prepared;
    CUdeviceptr first_tile_device{};
    std::memcpy(&first_tile_device, prepared.kernel.arguments.addresses()[14],
                sizeof(first_tile_device));
    void* first_tile_host{};
    if (!check(cuPointerGetAttribute(&first_tile_host, CU_POINTER_ATTRIBUTE_HOST_POINTER,
                                     first_tile_device),
               "resolve pooled physical observation"))
        return false;
    auto& compute = static_cast<std::uint32_t*>(first_tile_host)[1];
    const auto& init = prepared.initialization;
    if (init.kind != interceptor::CudaQueuedCopyKind::kHostToDevice ||
        !check(cuMemcpyHtoDAsync(init.destination, init.host_pointer, init.memory_bytes,
                                 resources.stream),
               "initialize pooled counter once"))
        return false;
    const auto launch = [&](bool should_stop_before_launch) {
        auto& kernel = prepared.kernel;
        if (!check(prepared.continuation->prepare_launch(kernel.grid, kernel.arguments),
                   "prepare pooled physical launch"))
            return false;
        if (should_stop_before_launch &&
            !check(prepared.continuation->request_retreat(), "request initial pooled retreat"))
            return false;
        const auto owned = kernel.arguments.addresses();
        // The native ABI does not modify the pointer array; copy only its addresses.
        std::vector<void*> pointers(owned.begin(), owned.end());
        return check(
            cuLaunchKernel(kernel.function, kernel.grid[0], kernel.grid[1], kernel.grid[2],
                           kernel.block[0], kernel.block[1], kernel.block[2],
                           kernel.shared_memory_bytes, resources.stream, pointers.data(), nullptr),
            "launch pooled GEMM");
    };
    const auto inspect = [&](bool expected_complete) {
        bool is_complete = !expected_complete;
        return check(prepared.continuation->stage_completion(resources.stream),
                     "stage pooled progress readback") &&
               check(cuStreamSynchronize(resources.stream), "prove pooled physical stop") &&
               check(prepared.continuation->inspect_stopped(is_complete),
                     "inspect pooled stopped progress") &&
               is_complete == expected_complete;
    };
    if (!launch(true) || !inspect(false) ||
        prepared.continuation->progress().observed_blocks != 0 ||
        std::atomic_ref(compute).load(std::memory_order_acquire) != 0 ||
        std::atomic_ref(*static_cast<std::uint32_t*>(first_tile_host))
                .load(std::memory_order_acquire) != 0)
        return false;
    if (!launch(false) || !await_first_tile(first_tile_host)) return false;
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
    if (!check(prepared.continuation->request_retreat(), "request pooled retreat") ||
        !inspect(false))
        return false;
    observation.stopped_count = prepared.continuation->progress().observed_blocks;
    observation.compute_witness = std::atomic_ref(compute).load(std::memory_order_acquire);
    const auto stopped_count = observation.stopped_count;
    if (stopped_count == 0 || stopped_count >= total) return false;
    // No initialization copy on resume: all logical tile ownership survives.
    if (!launch(false) || !inspect(true)) return false;
    const auto progress = prepared.continuation->progress();
    if (progress.physical_launches != 3 || progress.partial_blocks != stopped_count ||
        progress.observed_blocks < total || progress.observed_blocks > total + 3)
        return false;
    resources.prepared.reset();
    return weak_lease.expired() && resources.pool->occupied_slots() == 0;
}

bool run(PersistentSgemmProblem problem, bool should_retreat, bool uses_pool = false,
         float witness_a = 0.0F) {
    constexpr std::size_t guard = 32;
    constexpr float sentinel = 12345.0F;
    if (should_retreat) {
        problem.leading_a += 16;
        problem.leading_b += 16;
        problem.leading_c += 16;
    }
    problem.alpha = 0.75F;
    problem.beta = should_retreat && !problem.transpose_a ? 1.0F : 0.0F;
    const auto layout = glimmer::interceptor::persistent_sgemm_layout(problem);
    if (!layout) return false;
    Resources resources;
    CUdevice device{};
    if (!check(cuInit(0), "initialize Driver") || !check(cuDeviceGet(&device, 0), "select GPU") ||
        !check(cuCtxCreate(&resources.context, CU_CTX_MAP_HOST, device), "create test context") ||
        !check(cuStreamCreate(&resources.stream, CU_STREAM_NON_BLOCKING), "create test stream") ||
        !check_blas(cublasCreate(&resources.blas), "create reference handle") ||
        !check_blas(cublasSetStream(resources.blas, resources.stream), "select reference stream") ||
        !check_blas(cublasSetPointerMode(resources.blas, CUBLAS_POINTER_MODE_HOST),
                    "host reference scalars") ||
        !check_blas(cublasSetMathMode(resources.blas, CUBLAS_DEFAULT_MATH),
                    "reference math mode") ||
        !load_gemm(resources.module) ||
        !check(cuMemHostAlloc(&resources.retreat_host, sizeof(std::uint32_t),
                              CU_MEMHOSTALLOC_DEVICEMAP),
               "allocate retreat flag") ||
        !check(cuMemHostGetDevicePointer(&resources.retreat_device, resources.retreat_host, 0),
               "map retreat flag") ||
        !check(cuMemHostAlloc(&resources.first_tile_host, 2 * sizeof(std::uint32_t),
                              CU_MEMHOSTALLOC_DEVICEMAP),
               "allocate first tile observation") ||
        !check(
            cuMemHostGetDevicePointer(&resources.first_tile_device, resources.first_tile_host, 0),
            "map first tile observation"))
        return false;
    std::atomic_ref(*static_cast<std::uint32_t*>(resources.first_tile_host))
        .store(0, std::memory_order_release);
    auto& compute = static_cast<std::uint32_t*>(resources.first_tile_host)[1];
    std::atomic_ref(compute).store(0, std::memory_order_release);
    CUfunction function{};
    if (!check(cuModuleGetFunction(&function, resources.module,
                                   glimmer::interceptor::kPersistentSgemmName),
               "find owned GEMM"))
        return false;
    std::vector<float> a(layout->a_bytes / sizeof(float) + 2 * guard, sentinel);
    std::vector<float> b(layout->b_bytes / sizeof(float) + 2 * guard, sentinel);
    std::vector<float> initial(layout->c_bytes / sizeof(float) + 2 * guard, sentinel);
    for (std::size_t index = guard; index < a.size() - guard; ++index)
        a[index] = static_cast<float>(static_cast<int>(index % 29) - 14) / 32.0F;
    for (std::size_t index = guard; index < b.size() - guard; ++index)
        b[index] = static_cast<float>(static_cast<int>(index % 31) - 15) / 32.0F;
    if (witness_a != 0.0F) {
        std::fill(a.begin() + guard, a.end() - guard, witness_a);
        std::fill(b.begin() + guard, b.end() - guard, 2.0F);
    }
    for (std::uint32_t col = 0; col < problem.columns; ++col)
        for (std::uint32_t row = 0; row < problem.rows; ++row)
            initial[guard + std::size_t{col} * problem.leading_c + row] =
                problem.beta != 0.0F ? 3.25F : std::numeric_limits<float>::quiet_NaN();
    std::vector<float> bias(problem.rows + 2 * guard, sentinel);
    for (std::uint32_t row = 0; row < problem.rows; ++row)
        bias[guard + row] = static_cast<float>(static_cast<int>(row % 17) - 8) / 16.0F;
    CUdeviceptr bias_all{}, bias_pointer{};
    if (problem.transpose_a) {
        if (!resources.allocate(bias_all, bias.size() * sizeof(float)) ||
            !check(cuMemcpyHtoD(bias_all, bias.data(), bias.size() * sizeof(float)), "upload bias"))
            return false;
        bias_pointer = bias_all + guard * sizeof(float);
    }
    CUdeviceptr a_all{}, b_all{}, c_all{}, reference_all{}, counter{};
    if (!resources.allocate(a_all, a.size() * sizeof(float)) ||
        !resources.allocate(b_all, b.size() * sizeof(float)) ||
        !resources.allocate(c_all, initial.size() * sizeof(float)) ||
        !resources.allocate(reference_all, initial.size() * sizeof(float)) ||
        !resources.allocate(counter, sizeof(std::uint64_t)) ||
        !check(cuMemcpyHtoD(a_all, a.data(), a.size() * sizeof(float)), "upload A") ||
        !check(cuMemcpyHtoD(b_all, b.data(), b.size() * sizeof(float)), "upload B") ||
        !check(cuMemcpyHtoD(c_all, initial.data(), initial.size() * sizeof(float)), "upload C") ||
        !check(cuMemcpyHtoD(reference_all, initial.data(), initial.size() * sizeof(float)),
               "upload reference C") ||
        !check(cuMemsetD32(counter, 0, 2), "initialize progress once"))
        return false;
    auto a_pointer = a_all + guard * sizeof(float);
    auto b_pointer = b_all + guard * sizeof(float);
    auto c_pointer = c_all + guard * sizeof(float);
    const auto reference_pointer = reference_all + guard * sizeof(float);
    if (!check_blas(
            cublasSgemm(resources.blas, problem.transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                        problem.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                        static_cast<int>(problem.rows), static_cast<int>(problem.columns),
                        static_cast<int>(problem.depth), &problem.alpha,
                        std::bit_cast<const float*>(a_pointer), static_cast<int>(problem.leading_a),
                        std::bit_cast<const float*>(b_pointer), static_cast<int>(problem.leading_b),
                        &problem.beta, std::bit_cast<float*>(reference_pointer),
                        static_cast<int>(problem.leading_c)),
            "native reference SGEMM") ||
        !check(cuStreamSynchronize(resources.stream), "finish reference"))
        return false;
    auto grid = layout->grid;
    CUdeviceptr tile_counts{};
    const auto total = std::uint64_t{grid[0]} * grid[1];
    if (!uses_pool && (!resources.allocate(tile_counts, total * sizeof(std::uint32_t)) ||
                       !check(cuMemsetD32(tile_counts, 0, total), "initialize tile counts once")))
        return false;
    auto reduction_depth = layout->reduction_depth;
    std::array<void*, 22> arguments{&a_pointer,
                                    &b_pointer,
                                    &c_pointer,
                                    &problem.rows,
                                    &problem.columns,
                                    &problem.depth,
                                    &problem.leading_a,
                                    &problem.leading_b,
                                    &problem.leading_c,
                                    &problem.transpose_b,
                                    &problem.alpha,
                                    &problem.beta,
                                    &problem.transpose_a,
                                    &bias_pointer,
                                    &resources.first_tile_device,
                                    &tile_counts,
                                    &reduction_depth,
                                    &grid[0],
                                    &grid[1],
                                    &grid[2],
                                    &counter,
                                    &resources.retreat_device};
    PhysicalProgressWitness observation;
    auto& stopped_count = observation.stopped_count;
    auto& compute_witness = observation.compute_witness;
    if (uses_pool) {
        if (!run_pooled(resources, function, *layout, arguments.data(), observation)) return false;
    } else {
        auto& flag = *static_cast<std::uint32_t*>(resources.retreat_host);
        const auto launch = [&](unsigned int workers) {
            return check(cuLaunchKernel(function, workers, 1, 1, 256, 1, 1, 0, resources.stream,
                                        arguments.data(), nullptr),
                         "launch owned GEMM");
        };
        // First prove that an already requested retreat claims no work.
        std::atomic_ref(flag).store(1, std::memory_order_release);
        if (!launch(3) || !check(cuStreamSynchronize(resources.stream), "finish initial retreat"))
            return false;
        if (!check(cuMemcpyDtoH(&stopped_count, counter, sizeof(stopped_count)),
                   "read initial progress") ||
            stopped_count != 0 || std::atomic_ref(compute).load(std::memory_order_acquire) != 0 ||
            std::atomic_ref(*static_cast<std::uint32_t*>(resources.first_tile_host))
                    .load(std::memory_order_acquire) != 0)
            return false;
        std::atomic_ref(flag).store(0, std::memory_order_release);
        if (!launch(3)) return false;
        if (should_retreat) {
            if (!await_first_tile(resources.first_tile_host)) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
            std::atomic_ref(flag).store(1, std::memory_order_release);
            if (!check(cuStreamSynchronize(resources.stream), "prove physical retreat") ||
                !check(cuMemcpyDtoH(&stopped_count, counter, sizeof(stopped_count)),
                       "read partial progress"))
                return false;
            if (stopped_count == 0 || stopped_count >= total) {
                std::cerr << "retreat did not preserve incomplete physical work: " << stopped_count
                          << '/' << total << '\n';
                return false;
            }
            std::atomic_ref(flag).store(0, std::memory_order_release);
            if (!launch(5)) return false;
        }
        if (!check(cuStreamSynchronize(resources.stream), "finish resumed GEMM")) return false;
        compute_witness = std::atomic_ref(compute).load(std::memory_order_acquire);
        std::uint64_t finished_count{};
        if (!check(cuMemcpyDtoH(&finished_count, counter, sizeof(finished_count)),
                   "read final progress") ||
            finished_count < total || finished_count > total + (should_retreat ? 5 : 3))
            return false;
    }
    if (compute_witness == 0 ||
        (witness_a != 0.0F &&
         compute_witness != (std::bit_cast<std::uint32_t>(32.0F * witness_a) | 1U))) {
        std::cerr << "missing or operand-independent compute witness\n";
        return false;
    }
    if (!uses_pool) {
        std::vector<std::uint32_t> counts(total);
        if (!check(cuMemcpyDtoH(counts.data(), tile_counts, counts.size() * sizeof(std::uint32_t)),
                   "read exact tile counts") ||
            !std::all_of(counts.begin(), counts.end(), [](auto count) { return count == 1; }))
            return false;
    }
    std::vector<float> actual(initial.size()), expected(initial.size()), read_a(a.size()),
        read_b(b.size());
    if (!check(cuMemcpyDtoH(actual.data(), c_all, actual.size() * sizeof(float)),
               "read output and guards") ||
        !check(cuMemcpyDtoH(expected.data(), reference_all, expected.size() * sizeof(float)),
               "read reference") ||
        !check(cuMemcpyDtoH(read_a.data(), a_all, read_a.size() * sizeof(float)), "read input A") ||
        !check(cuMemcpyDtoH(read_b.data(), b_all, read_b.size() * sizeof(float)), "read input B"))
        return false;
    if (a != read_a || b != read_b) return false;
    if (problem.transpose_a) {
        std::vector<float> read_bias(bias.size());
        if (!check(cuMemcpyDtoH(read_bias.data(), bias_all, bias.size() * sizeof(float)),
                   "read bias") ||
            bias != read_bias)
            return false;
        for (std::uint32_t col = 0; col < problem.columns; ++col)
            for (std::uint32_t row = 0; row < problem.rows; ++row)
                expected[guard + std::size_t{col} * problem.leading_c + row] += bias[guard + row];
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
            std::abs(actual[index] - expected[index]) >
                0.00001F + 0.00001F * std::abs(expected[index])) {
            std::cerr << "output/padding/guard mismatch at " << index << ": " << actual[index]
                      << " vs " << expected[index] << '\n';
            return false;
        }
    }
    if (!resources.cleanup()) return false;
    std::cout << "persistent_sgemm m=" << problem.rows << " n=" << problem.columns
              << " k=" << problem.depth << " transpose_b=" << problem.transpose_b
              << " transpose_a=" << problem.transpose_a << " retreat=" << should_retreat
              << " pool=" << uses_pool << " stopped_tiles=" << stopped_count
              << " compute_witness=" << compute_witness << " total_tiles=" << total
              << " native_match=1 inputs=1 guards=1 cleanup=1\n";
    return true;
}
}  // namespace

int main() {
    for (const auto& problem : std::array<PersistentSgemmProblem, 14>{
             {{4096, 32, 1024, 4096, 1024, 4096, 0},
              {4096, 1024, 32, 4096, 1024, 4096, 1},
              {1024, 4096, 32, 1024, 4096, 1024, 1},
              {4096, 32, 1024, 1024, 1024, 4096, 0, 1.0F, 0.0F, 1},
              {1024, 32, 4096, 4096, 4096, 1024, 0, 1.0F, 0.0F, 1},
              {768, 768, 768, 768, 768, 768, 0},
              {3072, 768, 768, 3072, 768, 3072, 0},
              {768, 768, 3072, 768, 3072, 768, 0},
              {768, 768, 768, 768, 768, 768, 1},
              {3072, 768, 768, 3072, 768, 3072, 1},
              {768, 3072, 768, 768, 3072, 768, 1},
              {768, 768, 768, 768, 768, 768, 0, 1.0F, 0.0F, 1},
              {3072, 768, 768, 768, 768, 3072, 0, 1.0F, 0.0F, 1},
              {768, 768, 3072, 3072, 3072, 768, 0, 1.0F, 0.0F, 1}}}) {
        if (!run(problem, false) || !run(problem, true) || !run(problem, true, true))
            return EXIT_FAILURE;
    }
    const PersistentSgemmProblem witness_problem{768, 768, 768, 768, 768, 768, 0};
    if (!run(witness_problem, false, false, 1.0F) || !run(witness_problem, false, false, -0.5F))
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
