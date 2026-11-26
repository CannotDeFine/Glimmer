#pragma once

#include <cuda.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::backend::cuda {

inline constexpr char kCudaModuleJitPrivateReferencePattern[] = "*__glimmer*";

struct CudaModuleJitOption {
    std::uint32_t option{};
    std::uint64_t value{};
    std::vector<std::byte> data;
    bool operator==(const CudaModuleJitOption&) const = default;
};

// Cold, caller-serialized SDK argument storage, not execution authority. Owns
// no CUDA resource or callback. An authorized Session supplies context/lifetime
// gates and handles original status/partial native handles independently.
class CudaModuleJitArguments final {
   public:
    [[nodiscard]] static bool accepts(std::span<const CudaModuleJitOption> options) noexcept;
    [[nodiscard]] static std::unique_ptr<CudaModuleJitArguments> create(
        std::span<const CudaModuleJitOption> options) noexcept;
    CudaModuleJitArguments(const CudaModuleJitArguments&) = delete;
    CudaModuleJitArguments& operator=(const CudaModuleJitArguments&) = delete;
    // No allocation, SDK call or execution permission. Invalidates captured
    // output views and restores the exact original immutable inputs.
    void reset() noexcept;
    // Separate cold recipe retains generated entries; original metadata stays exact.
    void reset_for_private() noexcept;
    [[nodiscard]] bool has_private_references() const noexcept;
    [[nodiscard]] std::span<CUjit_option> native_options() noexcept;
    [[nodiscard]] std::span<void*> native_values() noexcept;
    [[nodiscard]] std::span<const CudaModuleJitOption> inputs() const noexcept;
    // No allocation. Captures every output byte including untouched tails.
    // Returns no view for malformed pointer/input-only mutations; the compiler
    // owner must retain uncertain native resources, never retry or infer cleanup.
    // Validates against the selected original/private recipe. A returned view
    // survives only until either reset method or destruction.
    [[nodiscard]] std::optional<std::span<const CudaModuleJitOption>> capture() noexcept;
    [[nodiscard]] std::size_t storage_bytes() const noexcept;

   private:
    CudaModuleJitArguments() = default;
    void reset_active() noexcept;
    [[nodiscard]] std::span<const CudaModuleJitOption> active_inputs() const noexcept;
    std::vector<CudaModuleJitOption> inputs_, private_inputs_, outputs_;
    std::array<CUjit_option, 34> options_{};
    std::array<void*, 34> values_{};
    // Independent input-only kernel/variable pointer arrays into owned data.
    std::array<std::array<const char*, 33>, 2> names_{};
    bool is_private_{};
};

}  // namespace glimmer::backend::cuda
