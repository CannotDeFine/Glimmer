#pragma once

#include "ptx_transform.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace glimmer::interceptor {

// Move-only launch storage. CUDA argument addresses point into these owned
// bytes, never into the application's stack. No GPU pointer is dereferenced.
// The owning submission must retain this object through every continuation.
class OwnedKernelArguments {
   public:
    OwnedKernelArguments(const OwnedKernelArguments&) = delete;
    OwnedKernelArguments& operator=(const OwnedKernelArguments&) = delete;
    OwnedKernelArguments(OwnedKernelArguments&&) noexcept = default;
    OwnedKernelArguments& operator=(OwnedKernelArguments&&) noexcept = default;

    [[nodiscard]] static std::optional<OwnedKernelArguments> copy(
        std::span<const PtxParameterLayout> layout, void* const* arguments) noexcept;
    [[nodiscard]] std::span<void* const> addresses() const noexcept {
        return addresses_;
    }
    // Serialized owner-only update; retains addresses and rejects every
    // out-of-range extent before touching bytes. Source may overlap storage.
    [[nodiscard]] bool overwrite(std::size_t parameter, std::size_t offset,
                                 std::span<const std::byte> bytes) noexcept;

   private:
    OwnedKernelArguments() = default;
    std::vector<std::vector<std::byte>> values_;
    std::vector<void*> addresses_;
};

}  // namespace glimmer::interceptor
