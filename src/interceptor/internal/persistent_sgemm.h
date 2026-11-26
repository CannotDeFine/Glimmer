#pragma once

#include "ptx_transform.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace glimmer::interceptor {

// Column-major FP32 GEMM. The forward subset transposes A and adds row bias;
// the backward subset leaves A unchanged and optionally transposes B.
// This CPU contract supplies no pointer, stream or execution authorization.
struct PersistentSgemmProblem {
    std::uint32_t rows{};
    std::uint32_t columns{};
    std::uint32_t depth{};
    std::uint32_t leading_a{};
    std::uint32_t leading_b{};
    std::uint32_t leading_c{};
    std::uint32_t transpose_b{};
    float alpha{1.0F};
    float beta{};
    std::uint32_t transpose_a{};
};

struct PersistentSgemmLayout {
    std::array<unsigned int, 3> grid;
    std::array<unsigned int, 3> block;
    std::size_t a_bytes{};
    std::size_t b_bytes{};
    std::size_t c_bytes{};
    std::size_t bias_bytes{};
    std::uint32_t reduction_depth{};
};

inline constexpr char kPersistentSgemmName[] = "glimmer_persistent_sgemm";
inline constexpr std::array<PtxParameterLayout, 17> kPersistentSgemmParameters{{{8, 8},
                                                                                {8, 8},
                                                                                {8, 8},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {4, 4},
                                                                                {8, 8},
                                                                                {8, 8},
                                                                                {8, 8},
                                                                                {4, 4}}};

[[nodiscard]] std::optional<PersistentSgemmLayout> persistent_sgemm_layout(
    const PersistentSgemmProblem& problem) noexcept;

// NUL-terminated, source-owned sm_70 PTX. The cooperative pool appends its
// grid x/y/z, device counter and mapped retreat arguments after the 17 above.
// Parameter 14 names an optional pair of u32 first-tile/first-compute flags;
// parameter 15 names per-tile u32 counts. The compute word depends on the
// first sixteen FMAs in physical CTA zero/thread zero, never on preparation.
// Parameter 16 is the validated, tile-aligned contiguous reduction depth.
// Keep them live until physical stop; clear counts only before initial issue,
// never on resume. Pool-owned audit storage replaces these private arguments.
// Launch only with the validated layout and 256x1x1 physical threads.
[[nodiscard]] const char* persistent_sgemm_ptx() noexcept;

}  // namespace glimmer::interceptor
