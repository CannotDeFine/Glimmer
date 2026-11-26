#include <cutlass/gemm/device/gemm.h>

#include "internal/cuda_gemm_block_completion.cuh"
#include "internal/cuda_gemm_geometry.h"
#include "internal/cuda_gemm_signal_layout.h"

#ifndef GLIMMER_GEMM_TILE_SIDE
#define GLIMMER_GEMM_TILE_SIDE 64
#endif
#ifndef GLIMMER_GEMM_SHARDED_SIGNAL
#define GLIMMER_GEMM_SHARDED_SIGNAL 0
#endif

#include <type_traits>

namespace {
constexpr int kGemmTileSide = GLIMMER_GEMM_TILE_SIDE;
constexpr int kGemmWarpSide = kGemmTileSide / 2;
constexpr int kGemmTileElements = kGemmTileSide * kGemmTileSide;
static_assert(glimmer::backend::cuda::cuda_gemm_tile_side_is_valid(kGemmTileSide));
using glimmer::backend::cuda::GemmBlockCompletion;

__device__ int* tile_coordinate() {
    __shared__ int coordinate[2];
    return coordinate;
}
__device__ bool is_retreat_requested(const volatile unsigned int* flag) {
#if GLIMMER_GEMM_SHARDED_SIGNAL
    using glimmer::backend::cuda::kGemmSignalShardBytes;
    using glimmer::backend::cuda::kGemmSignalShards;
    flag += (blockIdx.x % kGemmSignalShards) * (kGemmSignalShardBytes / sizeof(*flag));
#endif
    unsigned int value;
    // The default path publishes the aligned mapped flag with a system-visible
    // release store. Volatile alone does not establish inter-thread ordering.
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(flag) : "memory");
    return value != 0;
}
struct PersistentSwizzle : cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1> {
    __device__ static cutlass::gemm::GemmCoord get_tile_offset(int) {
        const auto* coordinate = tile_coordinate();
        return {coordinate[0], coordinate[1], 0};
    }
};
template <typename Element, typename Result>
struct Matrices {
    const Element* a;
    const Element* b;
    Result* c;
};
struct BatchGeometry {
    long long stride_a{}, stride_b{}, stride_c{};
    int count{1};
};
// NOLINTBEGIN(bugprone-easily-swappable-parameters): mirror the flat private GEMM ABI.
template <typename Element, typename Result, bool TransposeA, bool TransposeB, bool Batched = false,
          bool Observes = false>
__device__ void compute(Matrices<Element, Result> matrices, int rows, int columns, int depth,
                        int leading_a, int leading_b, int leading_c, float alpha, float beta,
                        const float* device_alpha, const float* device_beta,
                        unsigned int device_scalars, unsigned long long* claimed,
                        const volatile unsigned int* retreat, BatchGeometry geometry = {},
                        unsigned int* completed = nullptr) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    const auto [a, b, c] = matrices;
    using Row = cutlass::layout::RowMajor;
    using Column = cutlass::layout::ColumnMajor;
    using LayoutA = std::conditional_t<TransposeB, Column, Row>;
    using LayoutB = std::conditional_t<TransposeA, Column, Row>;
    using Output = cutlass::epilogue::thread::LinearCombination<Result, 1, float, float>;
    using Gemm = cutlass::gemm::device::Gemm<
        Element, LayoutA, Element, LayoutB, Result, Row, float, cutlass::arch::OpClassSimt,
        cutlass::arch::Sm70, cutlass::gemm::GemmShape<kGemmTileSide, kGemmTileSide, 8>,
        cutlass::gemm::GemmShape<kGemmWarpSide, kGemmWarpSide, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>, Output, PersistentSwizzle, 2>;
    using Kernel = typename Gemm::GemmKernel;
    static_assert(Kernel::kThreadCount == 128);
    __shared__ typename Kernel::SharedStorage shared;
    __shared__ unsigned long long tile;
    const int tile_rows = (columns + kGemmTileSide - 1) / kGemmTileSide;
    const int tile_columns = (rows + kGemmTileSide - 1) / kGemmTileSide;
    const auto matrix_tiles = static_cast<unsigned long long>(tile_rows) * tile_columns;
    const auto total = matrix_tiles * static_cast<unsigned int>(geometry.count);
    GemmBlockCompletion<Observes> completion(completed, claimed);
    if (device_scalars) {
        alpha = *device_alpha;
        beta = *device_beta;
    }
    for (;;) {
        if (threadIdx.x == 0) {
            tile = is_retreat_requested(retreat) ? total : atomicAdd(claimed, 1ULL);
            auto* coordinate = tile_coordinate();
            const auto local = tile % matrix_tiles;
            coordinate[0] = static_cast<int>(local % tile_rows);
            coordinate[1] = static_cast<int>(local / tile_rows);
        }
        __syncthreads();
        if (tile >= total) return;
        const auto batch = Batched ? static_cast<long long>(tile / matrix_tiles) : 0;
        const auto* batch_a = a + batch * geometry.stride_a;
        const auto* batch_b = b + batch * geometry.stride_b;
        auto* batch_c = c + batch * geometry.stride_c;
        const typename Kernel::Params params{{columns, rows, depth},
                                             {tile_rows, tile_columns, 1},
                                             {const_cast<Element*>(batch_b), leading_b},
                                             {const_cast<Element*>(batch_a), leading_a},
                                             {batch_c, leading_c},
                                             {batch_c, leading_c},
                                             {alpha, beta}};
        if (alpha == 0.0F && !device_scalars) {
            // Match the installed cuBLAS host-scalar zero fast path. Device
            // alpha-zero retains its measured NaN propagation through A/B.
            const auto local = tile % matrix_tiles;
            const int column = static_cast<int>(local % tile_rows) * kGemmTileSide;
            const int row = static_cast<int>(local / tile_rows) * kGemmTileSide;
            for (unsigned int index = threadIdx.x; index < kGemmTileElements; index += blockDim.x) {
                const int x = row + static_cast<int>(index % kGemmTileSide);
                const int y = column + static_cast<int>(index / kGemmTileSide);
                if (x < rows && y < columns) {
                    const auto offset = static_cast<long long>(y) * leading_c + x;
                    batch_c[offset] = static_cast<Result>(
                        beta == 0.0F ? 0.0F : beta * static_cast<float>(batch_c[offset]));
                }
            }
        } else {
            Kernel{}(params, shared);
        }
        // Claimed work is complete before another claim or physical retreat.
        __syncthreads();
        completion.publish();
    }
}
}  // namespace

// Flat private ABI; no CUTLASS layout/type leaks into host or wire contracts.
// RESULT is a type in this pointer declaration, never a macro expression.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define GLIMMER_SGEMM_ENTRY(NAME, ELEMENT, RESULT, TA, TB)                                      \
    extern "C" __global__ void NAME(                                                            \
        const ELEMENT* a, const ELEMENT* b, RESULT* c, int rows, int columns, int depth,        \
        int leading_a, int leading_b, int leading_c, float alpha, float beta,                   \
        const float* device_alpha, const float* device_beta, unsigned int device_scalars,       \
        unsigned long long* claimed, const volatile unsigned int* retreat) {                    \
        compute<ELEMENT, RESULT, TA, TB>({a, b, c}, rows, columns, depth, leading_a, leading_b, \
                                         leading_c, alpha, beta, device_alpha, device_beta,     \
                                         device_scalars, claimed, retreat);                     \
    }                                                                                           \
    extern "C" __global__ void NAME##_observed(                                                 \
        const ELEMENT* a, const ELEMENT* b, RESULT* c, int rows, int columns, int depth,        \
        int leading_a, int leading_b, int leading_c, float alpha, float beta,                   \
        const float* device_alpha, const float* device_beta, unsigned int device_scalars,       \
        unsigned long long* claimed, const volatile unsigned int* retreat,                      \
        unsigned int* completed) {                                                              \
        compute<ELEMENT, RESULT, TA, TB, false, true>(                                          \
            {a, b, c}, rows, columns, depth, leading_a, leading_b, leading_c, alpha, beta,      \
            device_alpha, device_beta, device_scalars, claimed, retreat, {}, completed);        \
    }
GLIMMER_SGEMM_ENTRY(glimmer_owned_sgemm_nn, float, float, false, false)
GLIMMER_SGEMM_ENTRY(glimmer_owned_sgemm_nt, float, float, false, true)
GLIMMER_SGEMM_ENTRY(glimmer_owned_sgemm_tn, float, float, true, false)
GLIMMER_SGEMM_ENTRY(glimmer_owned_sgemm_tt, float, float, true, true)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hgemm_nn, cutlass::half_t, cutlass::half_t, false, false)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hgemm_nt, cutlass::half_t, cutlass::half_t, false, true)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hgemm_tn, cutlass::half_t, cutlass::half_t, true, false)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hgemm_tt, cutlass::half_t, cutlass::half_t, true, true)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hsgemm_nn, cutlass::half_t, float, false, false)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hsgemm_nt, cutlass::half_t, float, false, true)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hsgemm_tn, cutlass::half_t, float, true, false)
GLIMMER_SGEMM_ENTRY(glimmer_owned_hsgemm_tt, cutlass::half_t, float, true, true)
#undef GLIMMER_SGEMM_ENTRY
#define GLIMMER_BATCH_GEMM_ENTRY(NAME, ELEMENT, RESULT, TA, TB)                                 \
    extern "C" __global__ void NAME(                                                            \
        const ELEMENT* a, const ELEMENT* b, RESULT* c, int rows, int columns, int depth,        \
        int leading_a, int leading_b, int leading_c, float alpha, float beta,                   \
        const float* device_alpha, const float* device_beta, unsigned int device_scalars,       \
        unsigned long long* claimed, const volatile unsigned int* retreat, long long stride_a,  \
        long long stride_b, long long stride_c, int batches) {                                  \
        compute<ELEMENT, RESULT, TA, TB, true>({a, b, c}, rows, columns, depth, leading_a,      \
                                               leading_b, leading_c, alpha, beta, device_alpha, \
                                               device_beta, device_scalars, claimed, retreat,   \
                                               {stride_a, stride_b, stride_c, batches});        \
    }                                                                                           \
    extern "C" __global__ void NAME##_observed(                                                 \
        const ELEMENT* a, const ELEMENT* b, RESULT* c, int rows, int columns, int depth,        \
        int leading_a, int leading_b, int leading_c, float alpha, float beta,                   \
        const float* device_alpha, const float* device_beta, unsigned int device_scalars,       \
        unsigned long long* claimed, const volatile unsigned int* retreat, long long stride_a,  \
        long long stride_b, long long stride_c, int batches, unsigned int* completed) {         \
        compute<ELEMENT, RESULT, TA, TB, true, true>(                                           \
            {a, b, c}, rows, columns, depth, leading_a, leading_b, leading_c, alpha, beta,      \
            device_alpha, device_beta, device_scalars, claimed, retreat,                        \
            {stride_a, stride_b, stride_c, batches}, completed);                                \
    }
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bsgemm_nn, float, float, false, false)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bsgemm_nt, float, float, false, true)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bsgemm_tn, float, float, true, false)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bsgemm_tt, float, float, true, true)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhgemm_nn, cutlass::half_t, cutlass::half_t, false, false)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhgemm_nt, cutlass::half_t, cutlass::half_t, false, true)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhgemm_tn, cutlass::half_t, cutlass::half_t, true, false)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhgemm_tt, cutlass::half_t, cutlass::half_t, true, true)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhsgemm_nn, cutlass::half_t, float, false, false)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhsgemm_nt, cutlass::half_t, float, false, true)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhsgemm_tn, cutlass::half_t, float, true, false)
GLIMMER_BATCH_GEMM_ENTRY(glimmer_owned_bhsgemm_tt, cutlass::half_t, float, true, true)
#undef GLIMMER_BATCH_GEMM_ENTRY
// NOLINTEND(bugprone-macro-parentheses)

namespace {
template <typename Element, typename Result>
struct LtMatrices {
    const Element* a;
    const Element* b;
    const Result* c;
    Result* d;
    const void* bias;
};
struct LtGeometry {
    int rows, columns, depth, leading_a, leading_b, leading_c, leading_d;
    unsigned int flags;
    long long stride_a, stride_b, stride_c, stride_d, stride_bias;
    int batches;
    unsigned int reduction_depth;
};
template <typename Element, bool TransposeA, bool TransposeB>
struct LtMma {
    using Row = cutlass::layout::RowMajor;
    using Column = cutlass::layout::ColumnMajor;
    using LayoutA = std::conditional_t<TransposeA, Row, Column>;
    using LayoutB = std::conditional_t<TransposeB, Row, Column>;
    using Output = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
    using Gemm = cutlass::gemm::device::Gemm<
        Element, LayoutA, Element, LayoutB, float, Row, float, cutlass::arch::OpClassSimt,
        cutlass::arch::Sm70, cutlass::gemm::GemmShape<kGemmTileSide, kGemmTileSide, 8>,
        cutlass::gemm::GemmShape<kGemmWarpSide, kGemmWarpSide, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>, Output,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2>;
    using Kernel = typename Gemm::GemmKernel;
    using Mma = typename Kernel::Mma;
};
template <typename Element>
struct LtShared {
    union MmaStorage {
        typename LtMma<Element, false, false>::Mma::SharedStorage nn;
        typename LtMma<Element, false, true>::Mma::SharedStorage nt;
        typename LtMma<Element, true, false>::Mma::SharedStorage tn;
        typename LtMma<Element, true, true>::Mma::SharedStorage tt;
    } storage;
    float accumulated[kGemmTileElements];
    unsigned long long tile;
    template <bool TransposeA, bool TransposeB>
    __device__ auto& mma() {
        if constexpr (TransposeA && TransposeB)
            return storage.tt;
        else if constexpr (TransposeA)
            return storage.tn;
        else if constexpr (TransposeB)
            return storage.nt;
        else
            return storage.nn;
    }
};
static_assert(sizeof(LtShared<float>) <= std::size_t{48} * 1024);
static_assert(sizeof(LtShared<cutlass::half_t>) <= std::size_t{48} * 1024);
// NOLINTBEGIN(bugprone-easily-swappable-parameters): mirror the flat private Lt ABI.
template <typename Element, typename Result, bool TransposeA, bool TransposeB,
          bool Observes = false>
__device__ void compute_lt(LtMatrices<Element, Result> matrices, LtGeometry geometry, float alpha,
                           float beta, const float* device_alpha, const float* device_beta,
                           unsigned int device_scalars, unsigned int bias_float,
                           unsigned long long* claimed, const volatile unsigned int* retreat,
                           LtShared<Element>& arena, unsigned int* completed = nullptr) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    using Traits = LtMma<Element, TransposeA, TransposeB>;
    using LayoutA = typename Traits::LayoutA;
    using LayoutB = typename Traits::LayoutB;
    using Mma = typename Traits::Mma;
    using WarpMma = typename Mma::Operator;
    static_assert(Traits::Kernel::kThreadCount == 128 && Mma::WarpCount::kK == 1);
    auto& shared = arena.template mma<TransposeA, TransposeB>();
    auto* accumulated = arena.accumulated;
    auto& tile = arena.tile;
    const int tile_rows = (geometry.rows + kGemmTileSide - 1) / kGemmTileSide;
    const int tile_columns = (geometry.columns + kGemmTileSide - 1) / kGemmTileSide;
    const auto matrix_tiles = static_cast<unsigned long long>(tile_rows) * tile_columns;
    const auto total = matrix_tiles * static_cast<unsigned int>(geometry.batches);
    GemmBlockCompletion<Observes> completion(completed, claimed);
    const int warp = static_cast<int>(threadIdx.x / 32);
    const int lane = static_cast<int>(threadIdx.x % 32);
    if (device_scalars) {
        alpha = *device_alpha;
        beta = *device_beta;
    }
    for (;;) {
        if (threadIdx.x == 0)
            tile = is_retreat_requested(retreat) ? total : atomicAdd(claimed, 1ULL);
        __syncthreads();
        if (tile >= total) return;
        const auto batch = static_cast<long long>(tile / matrix_tiles);
        const auto local = tile % matrix_tiles;
        const int row = static_cast<int>(local % tile_rows) * kGemmTileSide;
        const int column = static_cast<int>(local / tile_rows) * kGemmTileSide;
        const auto* a = matrices.a + batch * geometry.stride_a;
        const auto* b = matrices.b + batch * geometry.stride_b;
        const auto* c = matrices.c + batch * geometry.stride_c;
        auto* d = matrices.d + batch * geometry.stride_d;
        const typename Mma::IteratorA::Params parameters_a{LayoutA(geometry.leading_a)};
        const typename Mma::IteratorB::Params parameters_b{LayoutB(geometry.leading_b)};
        constexpr int values_per_thread = kGemmTileElements / Traits::Kernel::kThreadCount;
        float combined[values_per_thread];
        if (geometry.reduction_depth) {
            // Qualified split-K profiles seed the ordered reduction with bias,
            // rather than adding it after the last scaled partial.
            for (int item = 0; item < values_per_thread; ++item) {
                const auto index = threadIdx.x + item * Traits::Kernel::kThreadCount;
                const int x = row + static_cast<int>(index / kGemmTileSide);
                const int y = column + static_cast<int>(index % kGemmTileSide);
                combined[item] = 0.0F;
                if (matrices.bias && x < geometry.rows && y < geometry.columns) {
                    const auto offset = batch * geometry.stride_bias + x;
                    combined[item] =
                        bias_float
                            ? static_cast<const float*>(matrices.bias)[offset]
                            : static_cast<float>(static_cast<const Result*>(matrices.bias)[offset]);
                }
            }
        }
        int begin = 0;
        do {
            const int remaining = geometry.depth - begin;
            const int partition =
                geometry.reduction_depth &&
                        geometry.reduction_depth < static_cast<unsigned int>(remaining)
                    ? static_cast<int>(geometry.reduction_depth)
                    : remaining;
            const int end = begin + partition;
            // Consecutive original-K partitions use fresh iterators and MMA
            // fragments. The extent bounds predication at this partition's end.
            typename Mma::IteratorA iterator_a(parameters_a, const_cast<Element*>(a),
                                               {geometry.rows, end}, static_cast<int>(threadIdx.x),
                                               {row, begin});
            typename Mma::IteratorB iterator_b(parameters_b, const_cast<Element*>(b),
                                               {end, geometry.columns},
                                               static_cast<int>(threadIdx.x), {begin, column});
            typename Mma::FragmentC fragment;
            fragment.clear();
            Mma mma(shared, static_cast<int>(threadIdx.x), warp, lane);
            mma((partition + 7) / 8, fragment, iterator_a, iterator_b, fragment);
            typename WarpMma::IteratorC accumulator({accumulated, kGemmTileSide}, lane);
            accumulator.add_tile_offset({warp % Mma::WarpCount::kM, warp / Mma::WarpCount::kM});
            accumulator.store(fragment);
            __syncthreads();
            if (geometry.reduction_depth) {
                for (int item = 0; item < values_per_thread; ++item) {
                    const auto index = threadIdx.x + item * Traits::Kernel::kThreadCount;
                    combined[item] =
                        __fadd_rn(combined[item], __fmul_rn(alpha, accumulated[index]));
                }
                // Every thread consumes the current partial before any thread
                // can reuse the shared accumulator for the next partition.
                __syncthreads();
            }
            begin = end;
        } while (begin < geometry.depth);
        for (unsigned int index = threadIdx.x; index < kGemmTileElements; index += blockDim.x) {
            const int x = row + static_cast<int>(index / kGemmTileSide);
            const int y = column + static_cast<int>(index % kGemmTileSide);
            if (x >= geometry.rows || y >= geometry.columns) continue;
            const auto source = (geometry.flags & 4U)
                                    ? static_cast<long long>(x) * geometry.leading_c + y
                                    : static_cast<long long>(y) * geometry.leading_c + x;
            const auto destination = (geometry.flags & 8U)
                                         ? static_cast<long long>(x) * geometry.leading_d + y
                                         : static_cast<long long>(y) * geometry.leading_d + x;
            float value = geometry.reduction_depth ? combined[index / Traits::Kernel::kThreadCount]
                          : alpha == 0.0F && !device_scalars ? 0.0F
                                                             : alpha * accumulated[index];
            if (beta != 0.0F) value += beta * static_cast<float>(c[source]);
            if (matrices.bias && !geometry.reduction_depth) {
                const auto offset = batch * geometry.stride_bias + x;
                value +=
                    bias_float
                        ? static_cast<const float*>(matrices.bias)[offset]
                        : static_cast<float>(static_cast<const Result*>(matrices.bias)[offset]);
            }
            d[destination] = static_cast<Result>(value);
        }
        // No next claim or retreat may overtake any part of the tile epilogue.
        __syncthreads();
        completion.publish();
    }
}
template <typename Element, typename Result, bool Observes = false>
__device__ void dispatch_lt(LtMatrices<Element, Result> matrices, LtGeometry geometry, float alpha,
                            float beta, const float* device_alpha, const float* device_beta,
                            unsigned int device_scalars, unsigned int bias_float,
                            unsigned long long* claimed, const volatile unsigned int* retreat,
                            LtShared<Element>& arena, unsigned int* completed = nullptr) {
    if (geometry.flags & 1U) {
        if (geometry.flags & 2U)
            compute_lt<Element, Result, true, true, Observes>(
                matrices, geometry, alpha, beta, device_alpha, device_beta, device_scalars,
                bias_float, claimed, retreat, arena, completed);
        else
            compute_lt<Element, Result, true, false, Observes>(
                matrices, geometry, alpha, beta, device_alpha, device_beta, device_scalars,
                bias_float, claimed, retreat, arena, completed);
    } else {
        if (geometry.flags & 2U)
            compute_lt<Element, Result, false, true, Observes>(
                matrices, geometry, alpha, beta, device_alpha, device_beta, device_scalars,
                bias_float, claimed, retreat, arena, completed);
        else
            compute_lt<Element, Result, false, false, Observes>(
                matrices, geometry, alpha, beta, device_alpha, device_beta, device_scalars,
                bias_float, claimed, retreat, arena, completed);
    }
}
}  // namespace
// NOLINTBEGIN(bugprone-macro-parentheses): flat private type/field declarations.
#define GLIMMER_LT_ENTRY(NAME, ELEMENT, RESULT)                                                   \
    extern "C" __global__ void NAME(                                                              \
        const ELEMENT* a, const ELEMENT* b, const RESULT* c, RESULT* d, const void* bias,         \
        int rows, int columns, int depth, int leading_a, int leading_b, int leading_c,            \
        int leading_d, unsigned int flags, float alpha, float beta, const float* device_alpha,    \
        const float* device_beta, unsigned int device_scalars, unsigned int bias_float,           \
        unsigned long long* claimed, const volatile unsigned int* retreat, long long stride_a,    \
        long long stride_b, long long stride_c, long long stride_d, long long stride_bias,        \
        int batches, unsigned int reduction_depth) {                                              \
        __shared__ LtShared<ELEMENT> arena;                                                       \
        dispatch_lt<ELEMENT, RESULT>(                                                             \
            {a, b, c, d, bias},                                                                   \
            {rows, columns, depth, leading_a, leading_b, leading_c, leading_d, flags, stride_a,   \
             stride_b, stride_c, stride_d, stride_bias, batches, reduction_depth},                \
            alpha, beta, device_alpha, device_beta, device_scalars, bias_float, claimed, retreat, \
            arena);                                                                               \
    }                                                                                             \
    extern "C" __global__ void NAME##_observed(                                                   \
        const ELEMENT* a, const ELEMENT* b, const RESULT* c, RESULT* d, const void* bias,         \
        int rows, int columns, int depth, int leading_a, int leading_b, int leading_c,            \
        int leading_d, unsigned int flags, float alpha, float beta, const float* device_alpha,    \
        const float* device_beta, unsigned int device_scalars, unsigned int bias_float,           \
        unsigned long long* claimed, const volatile unsigned int* retreat, long long stride_a,    \
        long long stride_b, long long stride_c, long long stride_d, long long stride_bias,        \
        int batches, unsigned int reduction_depth, unsigned int* completed) {                     \
        __shared__ LtShared<ELEMENT> arena;                                                       \
        dispatch_lt<ELEMENT, RESULT, true>(                                                       \
            {a, b, c, d, bias},                                                                   \
            {rows, columns, depth, leading_a, leading_b, leading_c, leading_d, flags, stride_a,   \
             stride_b, stride_c, stride_d, stride_bias, batches, reduction_depth},                \
            alpha, beta, device_alpha, device_beta, device_scalars, bias_float, claimed, retreat, \
            arena, completed);                                                                    \
    }
GLIMMER_LT_ENTRY(glimmer_owned_lt_sgemm, float, float)
GLIMMER_LT_ENTRY(glimmer_owned_lt_hgemm, cutlass::half_t, cutlass::half_t)
GLIMMER_LT_ENTRY(glimmer_owned_lt_hsgemm, cutlass::half_t, float)
#undef GLIMMER_LT_ENTRY
// NOLINTEND(bugprone-macro-parentheses)

// Keep the standard SGEMM ABI intact; only this qualified entry appends the
// partition depth and reuses the ordered, complete-tile reduction.
extern "C" __global__ void glimmer_owned_sgemm_nn_split_k(
    const float* a, const float* b, float* c, int rows, int columns, int depth, int leading_a,
    int leading_b, int leading_c, float alpha, float beta, const float* device_alpha,
    const float* device_beta, unsigned int device_scalars, unsigned long long* claimed,
    const volatile unsigned int* retreat, unsigned int reduction_depth) {
    __shared__ LtShared<float> arena;
    compute_lt<float, float, false, false>({a, b, c, c, nullptr},
                                           {rows, columns, depth, leading_a, leading_b, leading_c,
                                            leading_c, 0, 0, 0, 0, 0, 0, 1, reduction_depth},
                                           alpha, beta, device_alpha, device_beta, device_scalars,
                                           0, claimed, retreat, arena);
}

extern "C" __global__ void glimmer_owned_sgemm_nn_split_k_observed(
    const float* a, const float* b, float* c, int rows, int columns, int depth, int leading_a,
    int leading_b, int leading_c, float alpha, float beta, const float* device_alpha,
    const float* device_beta, unsigned int device_scalars, unsigned long long* claimed,
    const volatile unsigned int* retreat, unsigned int reduction_depth, unsigned int* completed) {
    __shared__ LtShared<float> arena;
    compute_lt<float, float, false, false, true>(
        {a, b, c, c, nullptr},
        {rows, columns, depth, leading_a, leading_b, leading_c, leading_c, 0, 0, 0, 0, 0, 0, 1,
         reduction_depth},
        alpha, beta, device_alpha, device_beta, device_scalars, 0, claimed, retreat, arena,
        completed);
}
