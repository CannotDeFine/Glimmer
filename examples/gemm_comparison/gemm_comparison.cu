#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
constexpr int kGuardElements = 256;
constexpr float kGuardValue = 12345.0F;
constexpr int kWorkers = 320;
constexpr int kSamples = 21;

void require(bool passed, const char* operation) {
    if (!passed) throw std::runtime_error(operation);
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

void check_blas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
}

void check_cutlass(cutlass::Status status, const char* operation) {
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cutlassGetStatusString(status));
}

// One shared coordinate per physical CTA, read by both CUTLASS swizzle calls.
__device__ int* logical_coordinate() {
    __shared__ int coordinate[2];
    return coordinate;
}

struct PersistentSwizzle : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1> {
    __device__ static cutlass::gemm::GemmCoord get_tile_offset(int) {
        auto* coordinate = logical_coordinate();
        return {coordinate[0], coordinate[1], 0};
    }
};

__global__ void read_sm_id_extent(unsigned int* extent) {
    asm volatile("mov.u32 %0, %%nsmid;" : "=r"(*extent));
}

struct TileProgress {
    unsigned int* claimed;
    unsigned int* tile_done;
    volatile unsigned int* retreat;
    unsigned int* counts;
    unsigned int* sm_counts;
    unsigned int* completed;
    unsigned int stop_after;
};

template <class Kernel, bool EnableAudit>
__global__ void persistent_gemm(typename Kernel::Params params, TileProgress progress) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<typename Kernel::SharedStorage*>(storage);
    __shared__ bool should_run;
    __shared__ bool should_retreat;
    const unsigned int total = params.grid_tiled_shape.m() * params.grid_tiled_shape.n();
    for (unsigned int tile = blockIdx.x; tile < total; tile += gridDim.x) {
        if (threadIdx.x == 0) {
            should_retreat = *progress.retreat != 0;
            should_run = !should_retreat && !progress.tile_done[tile];
            if constexpr (EnableAudit) {
                if (should_run) atomicAdd(progress.claimed, 1U);
            }
            auto* coordinate = logical_coordinate();
            coordinate[0] = tile % params.grid_tiled_shape.m();
            coordinate[1] = tile / params.grid_tiled_shape.m();
        }
        __syncthreads();
        if (should_retreat) return;
        if (!should_run) {
            // A completed tile can be skipped after a physically drained resume.
            // All warps must consume this decision before the leader publishes
            // the next iteration's shared flags and logical coordinate.
            __syncthreads();
            continue;
        }
        Kernel{}(params, shared);
        // The epilogue must finish before another tile reuses shared storage.
        __syncthreads();
        if (threadIdx.x == 0) {
            progress.tile_done[tile] = 1;
            if constexpr (EnableAudit) {
                atomicAdd(progress.counts + tile, 1U);
                unsigned int sm_id;
                asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
                atomicAdd(progress.sm_counts + sm_id, 1U);
                if (atomicAdd(progress.completed, 1U) + 1U == progress.stop_after)
                    *progress.retreat = 1;
            }
        }
        __syncthreads();
    }
}

template <class Kernel>
__global__ void audited_ordinary_gemm(typename Kernel::Params params, unsigned int* sm_counts) {
    extern __shared__ __align__(16) unsigned char storage[];
    auto& shared = *reinterpret_cast<typename Kernel::SharedStorage*>(storage);
    Kernel{}(params, shared);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int sm_id;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
        atomicAdd(sm_counts + sm_id, 1U);
    }
}

template <class T>
class DeviceBuffer {
   public:
    explicit DeviceBuffer(std::size_t size) : size_(size) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_), size * sizeof(T)), "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() {
        if (pointer_ && cudaFree(pointer_) != cudaSuccess) std::cerr << "device cleanup failed\n";
    }
    T* get() const {
        return pointer_;
    }
    void zero(cudaStream_t stream) {
        check_cuda(cudaMemsetAsync(pointer_, 0, size_ * sizeof(T), stream), "clear buffer");
    }
    void upload(const std::vector<T>& source) {
        require(source.size() == size_, "upload extent mismatch");
        check_cuda(cudaMemcpy(pointer_, source.data(), size_ * sizeof(T), cudaMemcpyHostToDevice),
                   "upload buffer");
        // Pageable H2D can return after staging but before DMA completes. The
        // nonblocking test stream does not implicitly wait for the default stream.
        check_cuda(cudaStreamSynchronize(nullptr), "prove upload completion");
    }
    std::vector<T> download() const {
        std::vector<T> result(size_);
        check_cuda(cudaMemcpy(result.data(), pointer_, size_ * sizeof(T), cudaMemcpyDeviceToHost),
                   "download buffer");
        return result;
    }
    void close() {
        const auto status = cudaFree(pointer_);
        pointer_ = nullptr;
        check_cuda(status, "free buffer");
    }

   private:
    T* pointer_{};
    std::size_t size_{};
};

struct Resources {
    cudaStream_t stream{};
    cublasHandle_t blas{};
    cudaEvent_t begin{};
    cudaEvent_t end{};
    Resources() {
        try {
            check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
            check_blas(cublasCreate(&blas), "create cuBLAS");
            check_blas(cublasSetStream(blas, stream), "set cuBLAS stream");
            check_blas(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "set FP32 math");
            check_blas(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST), "set scalar mode");
            check_cuda(cudaEventCreate(&begin), "create begin event");
            check_cuda(cudaEventCreate(&end), "create end event");
        } catch (...) {
            (void)cleanup();
            throw;
        }
    }
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() {
        if (!cleanup()) std::cerr << "resource cleanup failed\n";
    }
    bool cleanup() noexcept {
        bool passed = true;
        if (stream) passed &= cudaStreamSynchronize(stream) == cudaSuccess;
        if (end) passed &= cudaEventDestroy(end) == cudaSuccess;
        if (begin) passed &= cudaEventDestroy(begin) == cudaSuccess;
        if (blas) passed &= cublasDestroy(blas) == CUBLAS_STATUS_SUCCESS;
        if (stream) passed &= cudaStreamDestroy(stream) == cudaSuccess;
        stream = nullptr;
        blas = nullptr;
        begin = end = nullptr;
        return passed;
    }
};

struct Problem {
    const char* name;
    int m;
    int n;
    int k;
};

struct OperandSpec {
    std::size_t elements;
    std::uint32_t seed;
};

std::vector<float> operand(OperandSpec specification) {
    auto seed = specification.seed;
    std::vector<float> result(
        specification.elements + 2U * static_cast<std::size_t>(kGuardElements), kGuardValue);
    for (std::size_t i = 0; i < specification.elements; ++i) {
        seed = seed * 1664525U + 1013904223U;
        result[i + kGuardElements] =
            (static_cast<float>((seed >> 8) & 65535U) / 65536.0F - 0.5F) * 0.25F;
    }
    return result;
}

void check_guards(const std::vector<float>& values) {
    for (int i = 0; i < kGuardElements; ++i)
        require(values[i] == kGuardValue && values[values.size() - 1 - i] == kGuardValue,
                "matrix guard changed");
}

double compare(const std::vector<float>& expected, const std::vector<float>& actual) {
    require(expected.size() == actual.size(), "output extent mismatch");
    check_guards(actual);
    double maximum_error = 0;
    for (std::size_t i = kGuardElements; i < actual.size() - kGuardElements; ++i) {
        const double error = std::abs(static_cast<double>(expected[i]) - actual[i]);
        require(std::isfinite(actual[i]) && error <= 2e-5 + 2e-4 * std::abs(expected[i]),
                "native output tolerance exceeded");
        maximum_error = std::max(maximum_error, error);
    }
    return maximum_error;
}

struct DeviceGeometry {
    int sm_count;
    unsigned int sm_id_extent;
};

enum class ExecutionMode { kNative, kOrdinary, kPersistent };

template <bool TransposeA, bool TransposeB>
void run_problem(Resources& resources, const Problem& problem, DeviceGeometry geometry,
                 bool check_only) {
    const int sm_count = geometry.sm_count;
    const auto sm_id_extent = geometry.sm_id_extent;
    using Row = cutlass::layout::RowMajor;
    using Column = cutlass::layout::ColumnMajor;
    // C^T = op(B)^T * op(A)^T: row-major CUTLASS output aliases column-major BLAS C.
    using LayoutA = std::conditional_t<TransposeB, Column, Row>;
    using LayoutB = std::conditional_t<TransposeA, Column, Row>;
    using Output = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
    using Identity = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
    using Ordinary =
        cutlass::gemm::device::Gemm<float, LayoutA, float, LayoutB, float, Row, float,
                                    cutlass::arch::OpClassSimt, cutlass::arch::Sm70,
                                    cutlass::gemm::GemmShape<64, 64, 8>,
                                    cutlass::gemm::GemmShape<32, 32, 8>,
                                    cutlass::gemm::GemmShape<1, 1, 1>, Output, Identity, 2>;
    using Persistent = cutlass::gemm::device::Gemm<
        float, LayoutA, float, LayoutB, float, Row, float, cutlass::arch::OpClassSimt,
        cutlass::arch::Sm70, cutlass::gemm::GemmShape<64, 64, 8>,
        cutlass::gemm::GemmShape<32, 32, 8>, cutlass::gemm::GemmShape<1, 1, 1>, Output,
        PersistentSwizzle, 2>;
    using OrdinaryKernel = typename Ordinary::GemmKernel;
    using Kernel = typename Persistent::GemmKernel;
    static_assert(std::is_same_v<typename Kernel::Mma, typename OrdinaryKernel::Mma>);
    static_assert(std::is_same_v<typename Kernel::Epilogue, typename OrdinaryKernel::Epilogue>);
    constexpr auto shared_bytes = sizeof(typename Kernel::SharedStorage);
    constexpr int threads = Kernel::kThreadCount;
    const int leading_a = TransposeA ? problem.k : problem.m;
    const int leading_b = TransposeB ? problem.n : problem.k;
    const auto a_host = operand({static_cast<std::size_t>(problem.m) * problem.k, 817});
    const auto b_host = operand({static_cast<std::size_t>(problem.k) * problem.n, 918});
    auto c_host = operand({static_cast<std::size_t>(problem.m) * problem.n, 1019});
    DeviceBuffer<float> a(a_host.size()), b(b_host.size()), c(c_host.size()), d(c_host.size());
    a.upload(a_host);
    b.upload(b_host);
    auto* a_data = a.get() + kGuardElements;
    auto* b_data = b.get() + kGuardElements;
    auto* c_data = c.get() + kGuardElements;
    auto* d_data = d.get() + kGuardElements;
    const cutlass::gemm::GemmCoord shape{problem.n, problem.m, problem.k};
    const auto grid = Identity::get_tiled_shape(shape, {64, 64, 8}, 1);
    const unsigned int total_tiles = grid.m() * grid.n();
    DeviceBuffer<unsigned int> next_tile(1), tile_done(total_tiles), retreat(1),
        counts(total_tiles), sms(sm_id_extent), completed(1);
    int resident_blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &resident_blocks, persistent_gemm<Kernel, false>, threads, shared_bytes),
               "persistent residency");
    require(resident_blocks > 0 && kWorkers <= resident_blocks * sm_count,
            "320 workers exceed full-device resident capacity");
    int ordinary_sms = sm_count;
    int persistent_sms = sm_count;
    double maximum_error = 0;
    std::array<std::vector<double>, 3> timings;
    for (const float beta : {0.0F, 1.0F}) {
        if (beta == 0.0F)
            std::fill(c_host.begin() + kGuardElements, c_host.end() - kGuardElements,
                      std::numeric_limits<float>::quiet_NaN());
        else
            c_host = operand({static_cast<std::size_t>(problem.m) * problem.n, 1019});
        c.upload(c_host);
        d.upload(c_host);
        const float alpha = 1.0F;
        const typename Ordinary::Arguments args{shape,
                                                {b_data, leading_b},
                                                {a_data, leading_a},
                                                {c_data, problem.m},
                                                {d_data, problem.m},
                                                {alpha, beta}};
        Ordinary ordinary;
        check_cutlass(ordinary.can_implement(args), "CUTLASS contract");
        check_cutlass(ordinary.initialize(args), "initialize ordinary GEMM");
        const typename Kernel::Params params{shape,
                                             grid,
                                             {b_data, leading_b},
                                             {a_data, leading_a},
                                             {c_data, problem.m},
                                             {d_data, problem.m},
                                             {alpha, beta}};
        const typename OrdinaryKernel::Params ordinary_params{shape,
                                                              grid,
                                                              {b_data, leading_b},
                                                              {a_data, leading_a},
                                                              {c_data, problem.m},
                                                              {d_data, problem.m},
                                                              {alpha, beta}};
        const auto launch = [&](ExecutionMode mode, unsigned int workers, bool audit,
                                unsigned int stop_after = 0) {
            if (mode == ExecutionMode::kNative)
                check_blas(cublasSgemm(resources.blas, TransposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
                                       TransposeB ? CUBLAS_OP_T : CUBLAS_OP_N, problem.m, problem.n,
                                       problem.k, &alpha, a_data, leading_a, b_data, leading_b,
                                       &beta, d_data, problem.m),
                           "native SGEMM");
            else if (mode == ExecutionMode::kOrdinary && !audit)
                check_cutlass(ordinary(resources.stream), "ordinary CUTLASS GEMM");
            else if (mode == ExecutionMode::kOrdinary)
                audited_ordinary_gemm<OrdinaryKernel>
                    <<<Identity::get_grid_shape(grid), threads, shared_bytes, resources.stream>>>(
                        ordinary_params, sms.get());
            else if (audit)
                persistent_gemm<Kernel, true><<<workers, threads, shared_bytes, resources.stream>>>(
                    params, {next_tile.get(), tile_done.get(), retreat.get(), counts.get(),
                             sms.get(), completed.get(), stop_after});
            else
                persistent_gemm<Kernel, false>
                    <<<workers, threads, shared_bytes, resources.stream>>>(
                        params,
                        {nullptr, tile_done.get(), retreat.get(), nullptr, nullptr, nullptr, 0});
            check_cuda(cudaGetLastError(), "GEMM launch");
        };
        const auto reset = [&] {
            next_tile.zero(resources.stream);
            tile_done.zero(resources.stream);
            retreat.zero(resources.stream);
            counts.zero(resources.stream);
            sms.zero(resources.stream);
            completed.zero(resources.stream);
        };
        launch(ExecutionMode::kNative, kWorkers, false);
        check_cuda(cudaStreamSynchronize(resources.stream), "native completion");
        const auto reference = d.download();
        compare(reference, reference);
        // Independent sampled double oracle, including nonzero beta.
        for (int sample = 0; sample < 257; ++sample) {
            const int row = (sample * 113) % problem.m;
            const int column = (sample * 127) % problem.n;
            double expected = beta == 0 ? 0 : c_host[kGuardElements + column * problem.m + row];
            for (int depth = 0; depth < problem.k; ++depth) {
                const int ai = TransposeA ? row * leading_a + depth : depth * leading_a + row;
                const int bi = TransposeB ? depth * leading_b + column : column * leading_b + depth;
                expected +=
                    static_cast<double>(a_host[kGuardElements + ai]) * b_host[kGuardElements + bi];
            }
            const double actual = reference[kGuardElements + column * problem.m + row];
            if (!std::isfinite(actual) ||
                std::abs(expected - actual) > 2e-5 + 2e-4 * std::abs(expected))
                std::cerr << "double oracle mismatch case=" << problem.name << " beta=" << beta
                          << " row=" << row << " column=" << column << " expected=" << expected
                          << " actual=" << actual << '\n';
            require(std::isfinite(actual) &&
                        std::abs(expected - actual) <= 2e-5 + 2e-4 * std::abs(expected),
                    "sampled double oracle mismatch");
        }
        d.upload(c_host);
        launch(ExecutionMode::kOrdinary, kWorkers, false);
        check_cuda(cudaStreamSynchronize(resources.stream), "ordinary completion");
        const auto ordinary_output = d.download();
        maximum_error = std::max(maximum_error, compare(reference, ordinary_output));
        d.upload(c_host);
        reset();
        launch(ExecutionMode::kPersistent, kWorkers, true);
        check_cuda(cudaStreamSynchronize(resources.stream), "persistent completion");
        const auto persistent_output = d.download();
        maximum_error = std::max(maximum_error, compare(reference, persistent_output));
        require(std::memcmp(ordinary_output.data(), persistent_output.data(),
                            ordinary_output.size() * sizeof(float)) == 0,
                "ordinary and persistent arithmetic differ");
        for (const auto count : counts.download())
            require(count == 1, "tile execution not exactly once");
        const auto observed_sms = sms.download();
        persistent_sms =
            std::min(persistent_sms,
                     static_cast<int>(std::count_if(observed_sms.begin(), observed_sms.end(),
                                                    [](unsigned int count) { return count > 0; })));
        sms.zero(resources.stream);
        launch(ExecutionMode::kOrdinary, kWorkers, true);
        check_cuda(cudaStreamSynchronize(resources.stream), "ordinary SM audit completion");
        const auto ordinary_observed_sms = sms.download();
        ordinary_sms =
            std::min(ordinary_sms, static_cast<int>(std::count_if(
                                       ordinary_observed_sms.begin(), ordinary_observed_sms.end(),
                                       [](unsigned int count) { return count > 0; })));
        // A flag set before launch must prevent even the first claim.
        d.upload(c_host);
        reset();
        check_cuda(cudaMemsetAsync(retreat.get(), 1, sizeof(unsigned int), resources.stream),
                   "request initial retreat");
        launch(ExecutionMode::kPersistent, kWorkers, true);
        check_cuda(cudaStreamSynchronize(resources.stream), "initial retreat stop");
        const auto untouched_output = d.download();
        require(next_tile.download()[0] == 0 && std::memcmp(untouched_output.data(), c_host.data(),
                                                            c_host.size() * sizeof(float)) == 0,
                "pre-requested retreat claimed or changed work");
        reset();
        launch(ExecutionMode::kPersistent, 1, true, 5);
        check_cuda(cudaStreamSynchronize(resources.stream), "partial physical stop");
        const auto partial = completed.download()[0];
        require(partial == 5 && partial < total_tiles && next_tile.download()[0] == 5,
                "partial retreat lacks retained progress");
        retreat.zero(resources.stream);
        launch(ExecutionMode::kPersistent, kWorkers, true);
        check_cuda(cudaStreamSynchronize(resources.stream), "resumed physical completion");
        for (const auto count : counts.download()) require(count == 1, "resume duplicated a tile");
        compare(reference, d.download());
        // Validate the exact non-audited specialization used by normal timing.
        // Its complete-work launch is also separately qualified by NCU.
        d.upload(c_host);
        reset();
        launch(ExecutionMode::kPersistent, kWorkers, false);
        check_cuda(cudaStreamSynchronize(resources.stream), "normal persistent completion");
        const auto normal_output = d.download();
        maximum_error = std::max(maximum_error, compare(reference, normal_output));
        require(std::memcmp(ordinary_output.data(), normal_output.data(),
                            ordinary_output.size() * sizeof(float)) == 0,
                "normal and ordinary arithmetic differ");
        for (const auto done : tile_done.download()) require(done == 1, "normal tile incomplete");
        d.upload(c_host);
        reset();
        check_cuda(cudaMemsetAsync(retreat.get(), 1, sizeof(unsigned int), resources.stream),
                   "request normal initial retreat");
        launch(ExecutionMode::kPersistent, kWorkers, false);
        check_cuda(cudaStreamSynchronize(resources.stream), "normal initial retreat stop");
        const auto normal_untouched = d.download();
        require(
            std::memcmp(normal_untouched.data(), c_host.data(), c_host.size() * sizeof(float)) == 0,
            "normal pre-requested retreat changed output");
        for (const auto done : tile_done.download())
            require(done == 0, "normal initial retreat ran a tile");
        reset();
        launch(ExecutionMode::kPersistent, 1, true, 5);
        check_cuda(cudaStreamSynchronize(resources.stream), "normal resume partial physical stop");
        require(completed.download()[0] == 5 && next_tile.download()[0] == 5,
                "normal resume lacks retained partial progress");
        retreat.zero(resources.stream);
        launch(ExecutionMode::kPersistent, kWorkers, false);
        check_cuda(cudaStreamSynchronize(resources.stream), "normal resumed physical completion");
        compare(reference, d.download());
        for (const auto done : tile_done.download())
            require(done == 1, "normal resume tile incomplete");
        const auto retained_counts = counts.download();
        for (unsigned int tile = 0; tile < total_tiles; ++tile)
            require(retained_counts[tile] == (tile < 5 ? 1U : 0U),
                    "normal kernel executed audit instrumentation");
        require(a.download() == a_host && b.download() == b_host,
                "GEMM mutated operand or input guards");
        const auto unchanged_c = c.download();
        require(std::memcmp(unchanged_c.data(), c_host.data(), c_host.size() * sizeof(float)) == 0,
                "GEMM mutated old C or its guards");
        if (beta != 0.0F || check_only) continue;
        // Untimed setup/reset precedes the event; audits are disabled in all timings.
        for (int sample = -3; sample < kSamples; ++sample) {
            for (int order = 0; order < 3; ++order) {
                const int mode = ((sample + 3) + order) % 3;
                next_tile.zero(resources.stream);
                tile_done.zero(resources.stream);
                retreat.zero(resources.stream);
                check_cuda(cudaEventRecord(resources.begin, resources.stream), "begin timing");
                launch(static_cast<ExecutionMode>(mode), kWorkers, false);
                check_cuda(cudaEventRecord(resources.end, resources.stream), "end timing");
                check_cuda(cudaEventSynchronize(resources.end), "timed completion");
                float elapsed_ms = 0;
                check_cuda(cudaEventElapsedTime(&elapsed_ms, resources.begin, resources.end),
                           "read GPU time");
                if (sample >= 0) timings[mode].push_back(elapsed_ms * 1000.0);
            }
        }
    }
    std::array<double, 3> medians{};
    for (int mode = 0; mode < 3; ++mode) {
        auto samples = timings[mode];
        std::sort(samples.begin(), samples.end());
        if (!samples.empty()) medians[mode] = samples[samples.size() / 2];
    }
    std::cout << std::setprecision(9) << "{\"case\":\"" << problem.name << "\",\"m\":" << problem.m
              << ",\"n\":" << problem.n << ",\"k\":" << problem.k << ",\"threads\":" << threads
              << ",\"workers\":" << kWorkers << ",\"logical_tiles\":" << total_tiles
              << ",\"resident_blocks_per_sm\":" << resident_blocks << ",\"sm_count\":" << sm_count
              << ",\"ordinary_compute_sms\":" << ordinary_sms
              << ",\"persistent_compute_sms\":" << persistent_sms
              << ",\"persistent_audit_specialized\":true,\"normal_kernel_checks\":true"
              << ",\"max_native_abs_error\":" << maximum_error
              << ",\"ordinary_persistent_bitwise_equal\":true,\"retreat_resume\":true"
              << ",\"samples_per_mode\":" << timings[0].size()
              << ",\"native_median_us\":" << medians[0] << ",\"ordinary_median_us\":" << medians[1]
              << ",\"persistent_median_us\":" << medians[2] << ",\"gpu_samples_us\":[";
    for (int mode = 0; mode < 3; ++mode) {
        if (mode) std::cout << ',';
        std::cout << '[';
        for (std::size_t sample = 0; sample < timings[mode].size(); ++sample) {
            if (sample) std::cout << ',';
            std::cout << timings[mode][sample];
        }
        std::cout << ']';
    }
    std::cout << "]}\n";
    completed.close();
    sms.close();
    counts.close();
    retreat.close();
    tile_done.close();
    next_tile.close();
    d.close();
    c.close();
    b.close();
    a.close();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool check_only = argc == 2 && std::string(argv[1]) == "--check-only";
        require(argc == 1 || check_only, "usage: glimmer_gemm_comparison [--check-only]");
        require(std::getenv("LD_PRELOAD") == nullptr, "the standalone comparison forbids preload");
        cudaDeviceProp device{};
        check_cuda(cudaGetDeviceProperties(&device, 0), "device properties");
        require(device.major == 7 && device.minor == 0,
                "this experiment is qualified for Sm70 only");
        Resources resources;
        DeviceBuffer<unsigned int> sm_extent_buffer(1);
        read_sm_id_extent<<<1, 1, 0, resources.stream>>>(sm_extent_buffer.get());
        check_cuda(cudaGetLastError(), "SM identifier extent launch");
        check_cuda(cudaStreamSynchronize(resources.stream), "SM identifier extent completion");
        const auto sm_id_extent = sm_extent_buffer.download()[0];
        require(sm_id_extent >= static_cast<unsigned int>(device.multiProcessorCount) &&
                    sm_id_extent <= 1024,
                "invalid SM identifier extent");
        sm_extent_buffer.close();
        for (const auto& problem : std::array<Problem, 3>{{{"forward_qkv", 768, 768, 768},
                                                           {"forward_expand", 3072, 768, 768},
                                                           {"forward_contract", 768, 768, 3072}}})
            run_problem<true, false>(resources, problem, {device.multiProcessorCount, sm_id_extent},
                                     check_only);
        for (const auto& problem : std::array<Problem, 3>{{{"input_qkv", 768, 768, 768},
                                                           {"input_expand", 768, 768, 3072},
                                                           {"input_contract", 3072, 768, 768}}})
            run_problem<false, false>(resources, problem,
                                      {device.multiProcessorCount, sm_id_extent}, check_only);
        for (const auto& problem : std::array<Problem, 3>{{{"weight_qkv", 768, 768, 768},
                                                           {"weight_expand", 3072, 768, 768},
                                                           {"weight_contract", 768, 3072, 768}}})
            run_problem<false, true>(resources, problem, {device.multiProcessorCount, sm_id_extent},
                                     check_only);
        require(resources.cleanup(), "resource cleanup failed");
        std::cout << "{\"compiled_source_sha256\":\"" << GLIMMER_EXPERIMENT_SOURCE_SHA256
                  << "\",\"status\":\"ok\",\"cases\":9,\"diagnostic_only\":true,"
                     "\"application_acceptance\":false,\"priority_acceptance\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "GEMM experiment failed: " << error.what() << '\n';
        return 1;
    }
}
