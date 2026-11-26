#include "qualified_cubin_source.h"

#include <cerrno>
#include <bit>
#include <sys/uio.h>
#include <unistd.h>

namespace glimmer::interceptor {
namespace {
bool snapshot(const void* source, std::span<std::byte> destination) noexcept {
    iovec local{destination.data(), destination.size()};
    iovec remote{const_cast<void*>(source), destination.size()};
    const auto saved_errno = errno;
    const auto copied = ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0);
    errno = saved_errno;
    return copied >= 0 && static_cast<std::size_t>(copied) == destination.size();
}
}  // namespace

bool qualified_library_source_options(unsigned int jit_count, const void* library_options,
                                      unsigned int library_count) noexcept {
    if (jit_count != 0) return false;
    if (library_count == 0) return true;
    if (library_count != 1 || !library_options) return false;
    std::uint32_t option{};
    return snapshot(library_options, std::as_writable_bytes(std::span{&option, 1})) && option == 1;
}

std::optional<std::vector<std::byte>> snapshot_qualified_add_cubin(
    const void* image, QualifiedAddMapping mapping) noexcept {
    struct RestoreErrno {
        int value{errno};
        ~RestoreErrno() {
            errno = value;
        }
    } restore_errno;
    if (!image || sizeof(void*) != 8 || std::endian::native != std::endian::little)
        return std::nullopt;
    std::array<std::uint64_t, 2> prefix{};
    if (!snapshot(image, std::as_writable_bytes(std::span{prefix}))) return std::nullopt;
    constexpr std::uint64_t wrapper_version_one = 0x00000001466243b1;
    constexpr std::array<std::uint64_t, 2> fatbin_header{0x00100001ba55ed50, 0x4d64e8};
    if (prefix[0] == wrapper_version_one) {
        // CUDA 12.6 fatbinary_section.h: magic/version, data, auxiliary source.
        // Never inspect nested/link wrappers or a nonnull additional source.
        std::array<std::uint64_t, 3> wrapper{};
        if (!snapshot(image, std::as_writable_bytes(std::span{wrapper})) ||
            wrapper[0] != wrapper_version_one || !wrapper[1] || wrapper[2])
            return std::nullopt;
        image = std::bit_cast<const void*>(static_cast<std::uintptr_t>(wrapper[1]));
        if (!snapshot(image, std::as_writable_bytes(std::span{prefix})) || prefix != fatbin_header)
            return std::nullopt;
    }
    if (prefix == fatbin_header) {
        try {
            std::vector<std::byte> original(5072120);
            return snapshot(image, original) ? rewrite_qualified_add_fatbin(original, 70, mapping)
                                             : std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
    if (static_cast<std::uint32_t>(prefix[0]) != 0x464c457f) return std::nullopt;
    // The input header only bounds inspection; the full owned SHA-256 remains
    // authoritative. No filename, symbol name or header alone can qualify code.
    constexpr BinaryImageDigest header_digest{0xe141051b, 0x0cff2d02, 0x8277990c, 0x9fc4d7f7,
                                              0x444834d1, 0x2755146e, 0x8da8cede, 0x26bf9d34};
    std::array<std::byte, 64> header{};
    if (!image || !snapshot(image, header) || binary_image_digest(header) != header_digest)
        return std::nullopt;
    std::optional<std::vector<std::byte>> result;
    try {
        std::vector<std::byte> original(5336008);
        if (snapshot(image, original)) result = rewrite_qualified_add_cubin(original, 70, mapping);
    } catch (...) {
        // Optional private preparation must not alter original module success.
    }
    return result;
}

std::optional<std::vector<std::byte>> snapshot_qualified_gemm_cubin(
    const void* image, QualifiedGemmMapping mapping) noexcept {
    struct RestoreErrno {
        int value{errno};
        ~RestoreErrno() {
            errno = value;
        }
    } restore_errno;
    if (!image || sizeof(void*) != 8 || std::endian::native != std::endian::little)
        return std::nullopt;
    std::array<std::uint64_t, 2> prefix{};
    if (!snapshot(image, std::as_writable_bytes(std::span{prefix}))) return std::nullopt;
    constexpr std::uint64_t wrapper_version_one = 0x00000001466243b1;
    constexpr std::array<std::uint64_t, 2> fatbin_header{0x00100001ba55ed50, 0x58f50};
    if (prefix[0] == wrapper_version_one) {
        std::array<std::uint64_t, 3> wrapper{};
        if (!snapshot(image, std::as_writable_bytes(std::span{wrapper})) ||
            wrapper[0] != wrapper_version_one || !wrapper[1] || wrapper[2])
            return std::nullopt;
        image = std::bit_cast<const void*>(static_cast<std::uintptr_t>(wrapper[1]));
        if (!snapshot(image, std::as_writable_bytes(std::span{prefix})) || prefix != fatbin_header)
            return std::nullopt;
    }
    if (prefix == fatbin_header) {
        try {
            std::vector<std::byte> original(364384);
            return snapshot(image, original) ? rewrite_qualified_gemm_fatbin(original, 70, mapping)
                                             : std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }
    if (static_cast<std::uint32_t>(prefix[0]) != 0x464c457f) return std::nullopt;
    constexpr BinaryImageDigest header_digest{0x58b38cec, 0x44d9f2a1, 0xd6f6842c, 0xe8eb8e2c,
                                              0xdb75365c, 0xdbf5f576, 0x4cf74ff7, 0x83e4a1f6};
    std::array<std::byte, 64> header{};
    if (!snapshot(image, header) || binary_image_digest(header) != header_digest)
        return std::nullopt;
    try {
        std::vector<std::byte> original(3055056);
        return snapshot(image, original) ? rewrite_qualified_gemm_cubin(original, 70, mapping)
                                         : std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace glimmer::interceptor
