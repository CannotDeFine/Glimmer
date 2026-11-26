#pragma once

#include "qualified_cubin.h"

namespace glimmer::interceptor {

inline constexpr char kQualifiedAddEntry[] =
    "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_"
    "addIfEESt5arrayIPcLm3EEEEviT0_T1_";
inline constexpr char kQualifiedGemmEntry[] = "volta_sgemm_128x32_nt";

// Optional Linux self-memory snapshot, preserving errno and never directly
// dereferencing image. Only complete pinned raw cubin or fatbinary
// identities qualify, optionally through the supported version-1 SDK wrapper.
// Called after successful original source loading; returned bytes are private.
// Mapping selects the pinned single-CTA or range output, not a new input scope.
// The actual device must still be checked before native private preparation.
[[nodiscard]] std::optional<std::vector<std::byte>> snapshot_qualified_add_cubin(
    const void* image, QualifiedAddMapping mapping = QualifiedAddMapping::kSingleCta) noexcept;

// Same safe optional CPU snapshot contract, for the pinned GEMM cubin,
// compressed envelope or version-one SDK wrapper with no auxiliary source.
// This does not register/load/select a variant; actual context/device and
// source/queue lifetimes remain the caller's responsibility.
[[nodiscard]] std::optional<std::vector<std::byte>> snapshot_qualified_gemm_cubin(
    const void* image, QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) noexcept;

// CUDA 12.6's single BINARY_IS_PRESERVED tag affects source ownership only.
// Read the tag safely without adding a CUDA dependency to the binary parser.
[[nodiscard]] bool qualified_library_source_options(unsigned int jit_count,
                                                    const void* library_options,
                                                    unsigned int library_count) noexcept;

}  // namespace glimmer::interceptor
