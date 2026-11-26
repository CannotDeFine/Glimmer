#include "owned_kernel_arguments.h"

#include <cstring>

namespace glimmer::interceptor {

bool OwnedKernelArguments::overwrite(std::size_t parameter, std::size_t offset,
                                     std::span<const std::byte> bytes) noexcept {
    if (parameter >= values_.size() || offset > values_[parameter].size() ||
        bytes.size() > values_[parameter].size() - offset) {
        return false;
    }
    if (!bytes.empty()) {
        std::memmove(values_[parameter].data() + offset, bytes.data(), bytes.size());
    }
    return true;
}

std::optional<OwnedKernelArguments> OwnedKernelArguments::copy(
    std::span<const PtxParameterLayout> layout, void* const* arguments) noexcept {
    if (layout.size() > 256 || (!layout.empty() && arguments == nullptr)) {
        return std::nullopt;
    }
    std::size_t total_bytes = 0;
    for (std::size_t index = 0; index < layout.size(); ++index) {
        const auto& parameter = layout[index];
        if (arguments[index] == nullptr || parameter.size_bytes == 0 ||
            parameter.size_bytes > 65536 - total_bytes || parameter.alignment_bytes == 0 ||
            parameter.alignment_bytes > 128 ||
            (parameter.alignment_bytes & (parameter.alignment_bytes - 1)) != 0) {
            return std::nullopt;
        }
        total_bytes += parameter.size_bytes;
    }
    try {
        OwnedKernelArguments owned;
        owned.values_.reserve(layout.size());
        owned.addresses_.reserve(layout.size());
        for (std::size_t index = 0; index < layout.size(); ++index) {
            auto& value = owned.values_.emplace_back(layout[index].size_bytes);
            std::memcpy(value.data(), arguments[index], value.size());
            owned.addresses_.push_back(value.data());
        }
        return owned;
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace glimmer::interceptor
