#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace glimmer::transform {
inline constexpr std::size_t kMaxCudaModuleBytes = std::size_t{64} * 1024 * 1024;
inline constexpr std::size_t kMaxCudaModuleWorkBytes = std::size_t{1024} * 1024 * 1024;
struct CudaImageLimits {
    std::size_t image_bytes{kMaxCudaModuleBytes};
    std::size_t sections{8192}, kernels{4096}, parameters{1024};
    std::size_t parameter_bytes{std::size_t{4} * 1024 * 1024};
    std::size_t variables{4096};
    // Independent logical limits: input, one decoded member, retained metadata,
    // and cumulative materialization/scan work are not interchangeable.
    std::size_t decoded_bytes{kMaxCudaModuleBytes};
    std::size_t retained_bytes{kMaxCudaModuleBytes};
    std::size_t work_bytes{kMaxCudaModuleBytes};
    bool compact_ptx_arities{};
};
struct CudaImageArgument {
    std::size_t offset{}, bytes{};
    bool operator==(const CudaImageArgument&) const = default;
};
struct CudaImageKernel {
    std::string name;
    std::vector<CudaImageArgument> arguments;
};
struct CudaImageVariable {
    std::string name;
    std::size_t bytes{};
    bool operator==(const CudaImageVariable&) const = default;
};
struct CudaImagePtxArity {
    std::string name;
    std::size_t parameter_count{};
    bool has_conflicting_counts{};
};
struct CudaModuleMetadata {
    std::vector<CudaImageKernel> kernels;
    std::vector<CudaImageVariable> variables;
    // Matching-architecture PTX companion, or the selected PTX-only image.
    std::optional<std::string> ptx;
    // Candidate fallback sources must agree on any exposed entry's arity.
    std::vector<std::string> validation_ptx;
    // Opt-in replacement for validation_ptx. A missing fact is not zero arity;
    // conflicts are sticky across sources and apply only to the queried name.
    std::vector<CudaImagePtxArity> ptx_arities;
    bool is_ptx_selected{};
    bool has_complete_kernel_metadata{true};
    bool has_complete_variable_metadata{true};
    std::size_t retained_bytes{};
    // Logical global/constant section reservation, including alignment slack.
    // Excludes kernel parameter banks and driver/JIT/code allocations.
    std::size_t static_storage_bytes{};
};

// Pure bounded inspection of owned bytes, never host-pointer dereferencing,
// SDK, loader, filesystem or scheduling. Extent helpers permit a checked caller
// to snapshot tables before deriving the complete referenced image extent.
[[nodiscard]] std::optional<std::size_t> cuda_module_table_extent(
    std::span<const std::byte> prefix, CudaImageLimits limits = {}) noexcept;
[[nodiscard]] std::optional<std::size_t> cuda_module_image_extent(
    std::span<const std::byte> tables, CudaImageLimits limits = {}) noexcept;
[[nodiscard]] std::optional<CudaModuleMetadata> inspect_cuda_module(
    std::span<const std::byte> image, unsigned int compute_capability,
    CudaImageLimits limits = {}) noexcept;
// Streaming, CPU-only declaration inventory, independent of executable
// transformation support. Unknown/overflowing owned extents reject; external
// declarations own no storage. The result includes conservative alignment slack.
[[nodiscard]] std::optional<std::size_t> inspect_ptx_static_storage(
    std::string_view source) noexcept;
[[nodiscard]] std::optional<std::size_t> inspect_ptx_static_storage(
    std::string_view source, std::size_t source_bytes) noexcept;
}  // namespace glimmer::transform
