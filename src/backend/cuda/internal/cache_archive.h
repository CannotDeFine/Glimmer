#pragma once

#include "glimmer/transform/ptx_transform.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace glimmer::backend::cuda::internal {

// Corruption detection only. Trusted file ownership supplies archive authority.
inline std::uint64_t checksum(std::span<const std::byte> bytes) noexcept {
    std::uint64_t value = 14695981039346656037ULL;
    for (const auto byte : bytes)
        value = (value ^ std::to_integer<unsigned int>(byte)) * 1099511628211ULL;
    return value;
}

struct ArchiveWriter {
    const std::size_t maximum;
    std::vector<std::byte> bytes;
    explicit ArchiveWriter(std::size_t bound) : maximum(bound) {}
    void number(std::uint64_t value, std::size_t width = 4) {
        if (!width || width > 8 || (width < 8 && (value >> (width * 8))) || width > maximum ||
            bytes.size() > maximum - width)
            throw std::length_error("cache archive number bound");
        for (std::size_t index = 0; index < width; ++index) {
            bytes.push_back(static_cast<std::byte>(value & 255U));
            value >>= 8;
        }
    }
    void text(std::string_view value) {
        if (value.size() > UINT32_MAX) throw std::length_error("cache archive string bound");
        number(value.size());
        if (value.size() > maximum - bytes.size()) throw std::length_error("cache archive bound");
        const auto content = std::as_bytes(std::span{value.data(), value.size()});
        bytes.insert(bytes.end(), content.begin(), content.end());
    }
};

struct ArchiveReader {
    std::span<const std::byte> bytes;
    std::uint64_t number(std::size_t width = 4) {
        if (!width || width > 8 || bytes.size() < width)
            throw std::invalid_argument("truncated cache archive number");
        std::uint64_t value{};
        for (std::size_t index = 0; index < width; ++index)
            value |= std::uint64_t{std::to_integer<unsigned int>(bytes[index])} << (8 * index);
        bytes = bytes.subspan(width);
        return value;
    }
    std::string_view text(std::size_t maximum = std::numeric_limits<std::size_t>::max()) {
        const auto length = static_cast<std::size_t>(number());
        if (length > bytes.size() || length > maximum)
            throw std::invalid_argument("truncated or oversized cache archive string");
        const std::string_view value{reinterpret_cast<const char*>(bytes.data()), length};
        bytes = bytes.subspan(length);
        return value;
    }
    bool boolean() {
        const auto value = number();
        if (value > 1) throw std::invalid_argument("invalid cache archive boolean");
        return value != 0;
    }
    transform::PtxExecutionKind kind() {
        const auto value = number();
        if (value > static_cast<unsigned int>(transform::PtxExecutionKind::kSliced))
            throw std::invalid_argument("invalid cache archive family");
        return static_cast<transform::PtxExecutionKind>(value);
    }
};

}  // namespace glimmer::backend::cuda::internal
