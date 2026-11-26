#include "persistent_sgemm.h"

#include <cmath>
#include <limits>

namespace glimmer::interceptor {
namespace {

std::optional<std::size_t> matrix_bytes(std::uint32_t rows, std::uint32_t columns,
                                        std::uint32_t leading) noexcept {
    if (leading < rows) return std::nullopt;
    const auto elements = std::uint64_t{columns - 1} * leading + rows;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) return std::nullopt;
    return static_cast<std::size_t>(elements) * sizeof(float);
}

constexpr char kKernel[] = R"ptx(
.version 6.0
.target sm_70
.address_size 64

.visible .entry glimmer_persistent_sgemm(
    .param .u64 a, .param .u64 b, .param .u64 c,
    .param .u32 m, .param .u32 n, .param .u32 k,
    .param .u32 lda, .param .u32 ldb, .param .u32 ldc,
    .param .u32 transpose_b, .param .f32 alpha, .param .f32 beta,
    .param .u32 transpose_a, .param .u64 bias,
    .param .u64 first_tile_observer,
    .param .u64 tile_counts,
    .param .u32 reduction_depth,
    .param .u32 grid_x, .param .u32 grid_y, .param .u32 grid_z,
    .param .u64 counter, .param .u64 retreat)
.reqntid 256, 1, 1
{
    .reg .pred %leader, %stop, %transposed, %transposed_a, %more, %zero_beta, %has_bias, %has_observer, %has_counts, %compute_reporter;
    .reg .u32 %thread_id, %tx, %ty, %gx, %gy, %depth, %la, %lb, %lc, %tr;
    .reg .u32 %flat, %tile_row, %tile_col, %row, %col, %base_k, %inner;
    .reg .u32 %index_a, %index_b, %index_c, %tmp, %flag, %physical_block, %compute_bits;
    .reg .u32 %shared_offset, %shared_a, %shared_b, %shared_index;
    .reg .u64 %pa, %pb, %pc, %counter_ptr, %retreat_ptr, %claim, %total;
    .reg .u64 %address_a, %address_b, %address_c, %offset, %bias_ptr, %observer_ptr, %counts_ptr, %compute_ptr;
    .reg .f32 %av, %bv, %sum, %scale_a, %scale_c, %old_c, %result;
    .reg .f32 %aggregate, %partial;
    .reg .u32 %partition_depth, %partition_remainder;
    .reg .pred %partitioned, %partition_end;
    .shared .align 8 .u64 claimed_tile;
    .shared .align 16 .b8 tile_a[1024];
    .shared .align 16 .b8 tile_b[1024];

    ld.param.u64 %pa, [a];
    ld.param.u64 %pb, [b];
    ld.param.u64 %pc, [c];
    ld.param.u64 %counter_ptr, [counter];
    ld.param.u64 %retreat_ptr, [retreat];
    ld.param.u32 %gx, [grid_x];
    ld.param.u32 %gy, [grid_y];
    mul.wide.u32 %total, %gx, %gy;
    ld.param.u32 %depth, [k];
    ld.param.u32 %partition_depth, [reduction_depth];
    setp.lt.u32 %partitioned, %partition_depth, %depth;
    ld.param.u32 %la, [lda];
    ld.param.u32 %lb, [ldb];
    ld.param.u32 %lc, [ldc];
    ld.param.u32 %tr, [transpose_b];
    ld.param.f32 %scale_a, [alpha];
    ld.param.f32 %scale_c, [beta];
    setp.ne.u32 %transposed, %tr, 0;
    ld.param.u32 %tr, [transpose_a];
    setp.ne.u32 %transposed_a, %tr, 0;
    ld.param.u64 %bias_ptr, [bias];
    setp.ne.u64 %has_bias, %bias_ptr, 0;
    ld.param.u64 %observer_ptr, [first_tile_observer];
    setp.ne.u64 %has_observer, %observer_ptr, 0;
    ld.param.u64 %counts_ptr, [tile_counts];
    setp.ne.u64 %has_counts, %counts_ptr, 0;
    setp.eq.f32 %zero_beta, %scale_c, 0f00000000;
    mov.u32 %thread_id, %tid.x;
    and.b32 %tx, %thread_id, 15;
    shr.u32 %ty, %thread_id, 4;
    setp.eq.u32 %leader, %thread_id, 0;
    mov.u32 %physical_block, %ctaid.x;
    setp.eq.u32 %compute_reporter, %physical_block, 0;
    and.pred %compute_reporter, %compute_reporter, %leader;
    and.pred %compute_reporter, %compute_reporter, %has_observer;
    add.u64 %compute_ptr, %observer_ptr, 4;
    shl.b32 %shared_offset, %thread_id, 2;
    mov.u32 %shared_a, tile_a;
    mov.u32 %shared_b, tile_b;

claim_next:
    @!%leader bra broadcast;
    ld.volatile.global.u32 %flag, [%retreat_ptr];
    setp.ne.u32 %stop, %flag, 0;
    mov.u64 %claim, %total;
    @%stop bra publish;
    atom.global.add.u64 %claim, [%counter_ptr], 1;
publish:
    st.shared.u64 [claimed_tile], %claim;
broadcast:
    bar.sync 0;
    ld.shared.u64 %claim, [claimed_tile];
    setp.ge.u64 %stop, %claim, %total;
    @%stop ret;
    cvt.u32.u64 %flat, %claim;
    rem.u32 %tile_row, %flat, %gx;
    div.u32 %tile_col, %flat, %gx;
    mad.lo.u32 %row, %tile_row, 16, %tx;
    mad.lo.u32 %col, %tile_col, 16, %ty;
    mov.f32 %sum, 0f00000000;
    mov.f32 %aggregate, 0f00000000;
    @!%partitioned bra begin_depth;
    @!%has_bias bra begin_depth;
    mul.wide.u32 %offset, %row, 4;
    add.u64 %address_a, %bias_ptr, %offset;
    ld.global.f32 %aggregate, [%address_a];
begin_depth:
    mov.u32 %base_k, 0;

load_tiles:
    add.u32 %tmp, %base_k, %ty;
    @!%transposed_a mad.lo.u32 %index_a, %tmp, %la, %row;
    @%transposed_a mad.lo.u32 %index_a, %row, %la, %tmp;
    mul.wide.u32 %offset, %index_a, 4;
    add.u64 %address_a, %pa, %offset;
    ld.global.f32 %av, [%address_a];
    add.u32 %tmp, %base_k, %tx;
    @%transposed mad.lo.u32 %index_b, %tmp, %lb, %col;
    @!%transposed mad.lo.u32 %index_b, %col, %lb, %tmp;
    mul.wide.u32 %offset, %index_b, 4;
    add.u64 %address_b, %pb, %offset;
    ld.global.f32 %bv, [%address_b];
    add.u32 %shared_index, %shared_a, %shared_offset;
    st.shared.f32 [%shared_index], %av;
    add.u32 %shared_index, %shared_b, %shared_offset;
    st.shared.f32 [%shared_index], %bv;
    bar.sync 1;
    mov.u32 %inner, 0;
multiply_tile:
    mad.lo.u32 %tmp, %inner, 16, %tx;
    shl.b32 %tmp, %tmp, 2;
    add.u32 %shared_index, %shared_a, %tmp;
    ld.shared.f32 %av, [%shared_index];
    mad.lo.u32 %tmp, %ty, 16, %inner;
    shl.b32 %tmp, %tmp, 2;
    add.u32 %shared_index, %shared_b, %tmp;
    ld.shared.f32 %bv, [%shared_index];
    fma.rn.f32 %sum, %av, %bv, %sum;
    add.u32 %inner, %inner, 1;
    setp.lt.u32 %more, %inner, 16;
    @%more bra.uni multiply_tile;
    @!%compute_reporter bra compute_observed;
    // Preserve a data dependency on actual sixteen-FMA input accumulation.
    // This separate word is not an output-tile completion counter.
    mov.b32 %compute_bits, %sum;
    or.b32 %compute_bits, %compute_bits, 1;
    membar.sys;
    st.volatile.global.u32 [%compute_ptr], %compute_bits;
    setp.eq.u32 %compute_reporter, 0, 1;
compute_observed:
    bar.sync 1;
    add.u32 %base_k, %base_k, 16;
    @!%partitioned bra continue_depth;
    rem.u32 %partition_remainder, %base_k, %partition_depth;
    setp.eq.u32 %partition_end, %partition_remainder, 0;
    @!%partition_end bra continue_depth;
    mul.rn.f32 %partial, %scale_a, %sum;
    add.rn.f32 %aggregate, %aggregate, %partial;
    mov.f32 %sum, 0f00000000;
continue_depth:
    setp.lt.u32 %more, %base_k, %depth;
    @%more bra.uni load_tiles;

    mad.lo.u32 %index_c, %col, %lc, %row;
    mul.wide.u32 %offset, %index_c, 4;
    add.u64 %address_c, %pc, %offset;
    @%partitioned mov.f32 %result, %aggregate;
    @%partitioned bra apply_beta;
    mul.rn.f32 %result, %scale_a, %sum;
apply_beta:
    @%zero_beta bra write_output;
    ld.global.f32 %old_c, [%address_c];
    fma.rn.f32 %result, %scale_c, %old_c, %result;
write_output:
    @%partitioned bra store_output;
    @!%has_bias bra store_output;
    mul.wide.u32 %offset, %row, 4;
    add.u64 %address_a, %bias_ptr, %offset;
    ld.global.f32 %av, [%address_a];
    add.rn.f32 %result, %result, %av;
store_output:
    st.global.f32 [%address_c], %result;
    bar.sync 0;
    @!%leader bra next_tile;
    @!%has_counts bra observe_tile;
    mul.wide.u32 %offset, %flat, 4;
    add.u64 %address_a, %counts_ptr, %offset;
    atom.global.add.u32 %flag, [%address_a], 1;
observe_tile:
    @!%has_observer bra next_tile;
    mov.u32 %flag, 1;
    membar.sys;
    st.volatile.global.u32 [%observer_ptr], %flag;
next_tile:
    bra.uni claim_next;
}
)ptx";

}  // namespace

std::optional<PersistentSgemmLayout> persistent_sgemm_layout(
    const PersistentSgemmProblem& problem) noexcept {
    const bool is_input_gradient = problem.transpose_a == 0 && problem.rows == 4096 &&
                                   problem.columns == 32 && problem.depth == 1024 &&
                                   problem.transpose_b == 0;
    const bool is_weight_gradient = problem.transpose_a == 0 && problem.depth == 32 &&
                                    problem.transpose_b == 1 &&
                                    ((problem.rows == 4096 && problem.columns == 1024) ||
                                     (problem.rows == 1024 && problem.columns == 4096));
    const bool is_bert_input_gradient =
        problem.transpose_a == 0 && problem.transpose_b == 0 && problem.columns == 768 &&
        ((problem.rows == 768 && (problem.depth == 768 || problem.depth == 3072)) ||
         (problem.rows == 3072 && problem.depth == 768));
    const bool is_bert_weight_gradient =
        problem.transpose_a == 0 && problem.transpose_b == 1 && problem.depth == 768 &&
        ((problem.rows == 768 && (problem.columns == 768 || problem.columns == 3072)) ||
         (problem.rows == 3072 && problem.columns == 768));
    const bool is_bert_forward_shape =
        problem.columns == 768 &&
        ((problem.rows == 768 && (problem.depth == 768 || problem.depth == 3072)) ||
         (problem.rows == 3072 && problem.depth == 768));
    const bool is_forward =
        problem.transpose_a == 1 && problem.transpose_b == 0 && problem.beta == 0.0F &&
        (is_bert_forward_shape ||
         (problem.columns == 32 && ((problem.rows == 4096 && problem.depth == 1024) ||
                                    (problem.rows == 1024 && problem.depth == 4096))));
    // Zero alpha has separate BLAS no-read semantics. Keep it native until the
    // wrapper can qualify that contract; no kernel is accepted by this helper.
    if ((!is_input_gradient && !is_weight_gradient && !is_bert_input_gradient &&
         !is_bert_weight_gradient && !is_forward) ||
        !std::isfinite(problem.alpha) || problem.alpha == 0.0F || !std::isfinite(problem.beta)) {
        return std::nullopt;
    }
    const auto a =
        matrix_bytes(problem.transpose_a ? problem.depth : problem.rows,
                     problem.transpose_a ? problem.rows : problem.depth, problem.leading_a);
    const auto b =
        matrix_bytes(problem.transpose_b ? problem.columns : problem.depth,
                     problem.transpose_b ? problem.depth : problem.columns, problem.leading_b);
    const auto c = matrix_bytes(problem.rows, problem.columns, problem.leading_c);
    // Device indexing is u32 elements. Bound the complete addressed span before
    // deriving any launch geometry, including padded leading dimensions.
    if (!a || !b || !c || *a / sizeof(float) > std::numeric_limits<std::uint32_t>::max() ||
        *b / sizeof(float) > std::numeric_limits<std::uint32_t>::max() ||
        *c / sizeof(float) > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    // Match the qualified profile's split-K arithmetic without exposing
    // partial tiles to retreat or allocating vendor-style reduction scratch.
    auto reduction_depth = problem.depth;
    if (problem.rows == 768 && problem.columns == 768) {
        if (is_forward && problem.depth == 768) reduction_depth = 128;
        if (is_forward && problem.depth == 3072) reduction_depth = 1024;
        if (is_bert_input_gradient && problem.depth == 3072) reduction_depth = 768;
    }
    return PersistentSgemmLayout{{problem.rows / 16, problem.columns / 16, 1},
                                 {256, 1, 1},
                                 *a,
                                 *b,
                                 *c,
                                 is_forward ? std::size_t{problem.rows} * sizeof(float) : 0,
                                 reduction_depth};
}

const char* persistent_sgemm_ptx() noexcept {
    return kKernel;
}

}  // namespace glimmer::interceptor
