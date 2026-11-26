#include "qualified_cubin.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <elf.h>

namespace glimmer::interceptor {
namespace {

// FIPS 180-4 sections 4.2.2, 5.1.1, 5.3.3 and 6.2.2.
constexpr std::array<std::uint32_t, 64> kRoundConstants{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

void digest_block(BinaryImageDigest& state, std::span<const std::byte, 64> bytes) noexcept {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index != 16; ++index) {
        for (std::size_t octet = 0; octet != 4; ++octet)
            words[index] =
                (words[index] << 8) | std::to_integer<std::uint32_t>(bytes[index * 4 + octet]);
    }
    for (std::size_t index = 16; index != words.size(); ++index) {
        const auto x = words[index - 15];
        const auto y = words[index - 2];
        words[index] = words[index - 16] + (std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3)) +
                       words[index - 7] + (std::rotr(y, 17) ^ std::rotr(y, 19) ^ (y >> 10));
    }
    auto work = state;
    for (std::size_t index = 0; index != words.size(); ++index) {
        const auto first =
            work[7] + (std::rotr(work[4], 6) ^ std::rotr(work[4], 11) ^ std::rotr(work[4], 25)) +
            ((work[4] & work[5]) ^ (~work[4] & work[6])) + kRoundConstants[index] + words[index];
        const auto second =
            (std::rotr(work[0], 2) ^ std::rotr(work[0], 13) ^ std::rotr(work[0], 22)) +
            ((work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]));
        work = {first + second,  work[0], work[1], work[2],
                work[3] + first, work[4], work[5], work[6]};
    }
    for (std::size_t index = 0; index != state.size(); ++index) state[index] += work[index];
}

constexpr BinaryImageDigest kOriginalDigest{0xabce1108, 0xcbfc22a1, 0x5238324b, 0x7ec510c3,
                                            0xd48803ec, 0xfb393ea4, 0x3ffcdfa6, 0x66c98a1e};
constexpr BinaryImageDigest kMappedDigest{0x566dbdba, 0xd924718c, 0x43de2f93, 0x222a6110,
                                          0xb2962245, 0xb6eac3be, 0xa5a92c4a, 0x07749bfc};
constexpr BinaryImageDigest kRangeDigest{0x800ed88d, 0x402224d7, 0x292c2202, 0x09fc1557,
                                         0xa9d7c23a, 0xb0b99cdc, 0x5b3d5532, 0x5b22a5aa};
constexpr std::size_t kOriginalBytes = 5336008;
constexpr std::size_t kInsertion = 0xb06a4;
constexpr std::size_t kPadding = 128;
constexpr std::size_t kSectionHeaders = 5282536;
constexpr std::size_t kProgramHeaders = 5335784;
constexpr std::size_t kSectionCount = 832;
constexpr std::size_t kProgramCount = 4;
constexpr std::size_t kConstantSection = 415;
constexpr std::size_t kMetadata = 0x966e8;
constexpr std::size_t kInstruction = 0x2eef00 + 0x20;
static_assert(kSectionHeaders + kSectionCount * sizeof(Elf64_Shdr) == kProgramHeaders);
static_assert(kMetadata + 47 < kInsertion && kInsertion < kInstruction);
static_assert(kInstruction + 16 < kSectionHeaders && kConstantSection < kSectionCount);

// Only called on the already owned, exact-hash-qualified layout, never on
// arbitrary ELF offsets. All table positions and loop bounds are fixed.
template <typename T>
T read_record(const std::vector<std::byte>& bytes, std::size_t offset) noexcept {
    T value{};
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}
template <typename T>
void write_record(std::vector<std::byte>& bytes, std::size_t offset, const T& value) noexcept {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}
void map_owned_image(std::vector<std::byte>& image) {
    image.insert(image.begin() + kInsertion, kPadding, std::byte{});
    auto header = read_record<Elf64_Ehdr>(image, 0);
    header.e_phoff += kPadding;
    header.e_shoff += kPadding;
    write_record(image, 0, header);
    for (std::size_t index = 0; index != kSectionCount; ++index) {
        const auto position = kSectionHeaders + kPadding + index * sizeof(Elf64_Shdr);
        auto section = read_record<Elf64_Shdr>(image, position);
        if (section.sh_offset >= kInsertion) section.sh_offset += kPadding;
        if (index == kConstantSection) section.sh_size += 8;
        write_record(image, position, section);
    }
    for (std::size_t index = 0; index != kProgramCount; ++index) {
        const auto position = kProgramHeaders + kPadding + index * sizeof(Elf64_Phdr);
        auto program = read_record<Elf64_Phdr>(image, position);
        if (program.p_offset < kInsertion && kInsertion < program.p_offset + program.p_filesz) {
            program.p_filesz += kPadding;
            program.p_memsz += kPadding;
        } else if (program.p_offset >= kInsertion) {
            program.p_offset += kPadding;
        }
        write_record(image, position, program);
    }
    image[kMetadata + 26] = std::byte{40};
    image[kMetadata + 30] = std::byte{40};
    image[kMetadata + 46] = std::byte{129};
    constexpr std::array<std::uint64_t, 2> instruction{0x00006000ff037b82, 0x000e240000000800};
    write_record(image, kInstruction + kPadding, instruction);
}

// Apply the second insertion only after the whole intermediate image matches.
// Preserve every original instruction offset and extend only the target symbol.
void extend_owned_mapping(std::vector<std::byte>& image) {
    constexpr std::size_t code_section = 688;
    constexpr std::size_t code_offset = 0x2eef00 + kPadding;
    constexpr std::size_t code_bytes = 0xb00;
    constexpr std::size_t insertion = code_offset + code_bytes;
    constexpr std::size_t symbol_offset = 0x816c8 + 764 * sizeof(Elf64_Sym);
    static_assert(sizeof(Elf64_Sym) == 24);
    static_assert(symbol_offset + sizeof(Elf64_Sym) < kInsertion);
    static_assert(insertion < kSectionHeaders && code_section < kSectionCount);
    image.insert(image.begin() + insertion, kPadding, std::byte{});
    auto header = read_record<Elf64_Ehdr>(image, 0);
    header.e_phoff += kPadding;
    header.e_shoff += kPadding;
    write_record(image, 0, header);
    for (std::size_t index = 0; index != kSectionCount; ++index) {
        const auto position = kSectionHeaders + 2 * kPadding + index * sizeof(Elf64_Shdr);
        auto section = read_record<Elf64_Shdr>(image, position);
        if (section.sh_offset >= insertion) section.sh_offset += kPadding;
        if (index == code_section) section.sh_size += kPadding;
        write_record(image, position, section);
    }
    for (std::size_t index = 0; index != kProgramCount; ++index) {
        const auto position = kProgramHeaders + 2 * kPadding + index * sizeof(Elf64_Phdr);
        auto program = read_record<Elf64_Phdr>(image, position);
        if (program.p_offset < insertion && insertion < program.p_offset + program.p_filesz) {
            program.p_filesz += kPadding;
            program.p_memsz += kPadding;
        } else if (program.p_offset >= insertion) {
            program.p_offset += kPadding;
        }
        write_record(image, position, program);
    }
    auto symbol = read_record<Elf64_Sym>(image, symbol_offset);
    symbol.st_size += kPadding;
    write_record(image, symbol_offset, symbol);
    constexpr std::array<std::uint64_t, 2> entry{0x00000ad000007947, 0x000fea0003800000};
    constexpr std::array<std::uint64_t, 6> stub{0x0000000000037919, 0x000e240000002500,
                                                0x0000600003037a10, 0x001fc60007ffe0ff,
                                                0xfffff50000007947, 0x000fea000383ffff};
    constexpr std::array<std::uint64_t, 2> nop{0x0000000000007918, 0x000fc00000000000};
    write_record(image, code_offset + 0x20, entry);
    write_record(image, insertion, stub);
    for (std::size_t index = 3; index != 8; ++index)
        write_record(image, insertion + index * sizeof(nop), nop);
}

constexpr std::size_t kGemmOriginalBytes = 3055056;
constexpr BinaryImageDigest kGemmOriginalDigest{0x3ccb33a8, 0xb1b96337, 0x2ed285b2, 0x837cc95f,
                                                0xe959954a, 0x4cd05d5b, 0xf2f0c48b, 0x6d7b626c};
constexpr BinaryImageDigest kGemmMappedDigest{0x2bd0a00e, 0x52d867bd, 0xf208a1fd, 0xf4001c0e,
                                              0x74b1a649, 0x2dcb6830, 0x3f31b34b, 0x5d45da65};
constexpr BinaryImageDigest kGemmRangeDigest{0x2bdedfb3, 0x16f02b1b, 0x8e389f1f, 0x5c63453e,
                                             0xf78885fe, 0xa453e578, 0x8231d934, 0xde99fc6e};

// The pinned relocatable ELF has no program headers; its section table is
// before this insertion and remains at offset 64. Call only AFTER whole-image
// identity on owned bytes. No untrusted offset controls these reads or writes.
void map_owned_gemm_image(std::vector<std::byte>& image) {
    constexpr std::size_t insertion = 0xfc510;
    constexpr std::size_t section_headers = 64;
    constexpr std::size_t section_count = 366;
    constexpr std::size_t constant_section = 63;
    constexpr std::size_t metadata = 0x2e8590;
    constexpr std::size_t code = 0xf5c00;
    static_assert(section_headers + section_count * sizeof(Elf64_Shdr) < code);
    static_assert(code + 0x1690 < insertion && constant_section < section_count);
    static_assert(insertion < metadata && metadata + 32 <= kGemmOriginalBytes);
    image.insert(image.begin() + insertion, kPadding, std::byte{});
    for (std::size_t index = 0; index != section_count; ++index) {
        const auto position = section_headers + index * sizeof(Elf64_Shdr);
        auto section = read_record<Elf64_Shdr>(image, position);
        if (section.sh_offset >= insertion) section.sh_offset += kPadding;
        if (index == constant_section) section.sh_size += 8;
        write_record(image, position, section);
    }
    image[metadata + kPadding + 10] = std::byte{184};
    image[metadata + kPadding + 14] = std::byte{184};
    image[metadata + kPadding + 30] = std::byte{225};
    constexpr std::array<std::uint64_t, 2> first{0x00008400ff017b82, 0x000f240000000800};
    constexpr std::array<std::uint64_t, 2> second{0x00008400ff177b82, 0x000f240000000800};
    write_record(image, code + 0x10, first);
    write_record(image, code + 0x1680, second);
}

// Apply only after verifying the complete plane identity. Two original CTAID.Z
// sites branch into verified unreachable NOP padding and return immediately.
// Compiler-produced S2R/IADD3 controls preserve the original write barrier 4
// and wait only for it, without disturbing outstanding memory dependencies.
// No extra register, code extent, symbol or relocation changes are needed.
void map_owned_gemm_ranges(std::vector<std::byte>& image) {
    constexpr std::size_t code = 0xf5c00;
    constexpr std::array<std::uint64_t, 2> first{0x0000267000007947, 0x000fea0003800000};
    constexpr std::array<std::uint64_t, 2> second{0x0000103000007947, 0x000fea0003800000};
    constexpr std::array<std::uint64_t, 6> first_stub{0x0000000000017919, 0x000f240000002700,
                                                      0x0000840001017a10, 0x010fe40007ffe0ff,
                                                      0xffffd96000007947, 0x000fea000383ffff};
    constexpr std::array<std::uint64_t, 6> second_stub{0x0000000000177919, 0x000f240000002700,
                                                       0x0000840017177a10, 0x010fe40007ffe0ff,
                                                       0xffffefa000007947, 0x000fea000383ffff};
    write_record(image, code + 0x10, first);
    write_record(image, code + 0x1680, second);
    write_record(image, code + 0x2690, first_stub);
    write_record(image, code + 0x26c0, second_stub);
}

}  // namespace

std::optional<BinaryImageDigest> binary_image_digest(std::span<const std::byte> image) noexcept {
    if (image.size() > kMaxBinaryImageBytes) return std::nullopt;
    BinaryImageDigest state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto size = image.size();
    while (image.size() >= 64) {
        digest_block(state, image.first<64>());
        image = image.subspan(64);
    }
    std::array<std::byte, 128> tail{};
    std::ranges::copy(image, tail.begin());
    tail[image.size()] = std::byte{0x80};
    const std::size_t padded_bytes = image.size() < 56 ? 64 : 128;
    const auto bits = static_cast<std::uint64_t>(size) * 8;
    for (std::size_t index = 0; index != 8; ++index)
        tail[padded_bytes - 1 - index] = static_cast<std::byte>((bits >> (index * 8)) & 0xff);
    digest_block(state, std::span{tail}.first<64>());
    if (padded_bytes == 128) digest_block(state, std::span{tail}.last<64>());
    return state;
}

std::optional<std::vector<std::byte>> rewrite_qualified_add_cubin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedAddMapping mapping) noexcept {
    static_assert(sizeof(Elf64_Ehdr) == 64 && sizeof(Elf64_Shdr) == 64 && sizeof(Elf64_Phdr) == 56);
    static_assert(kProgramHeaders + kProgramCount * sizeof(Elf64_Phdr) == kOriginalBytes);
    if (compute_capability != 70 || original.size() != kOriginalBytes ||
        std::endian::native != std::endian::little ||
        (mapping != QualifiedAddMapping::kSingleCta && mapping != QualifiedAddMapping::kRange))
        return std::nullopt;
    try {
        std::vector<std::byte> image(original.begin(), original.end());
        if (binary_image_digest(image) != kOriginalDigest) return std::nullopt;
        map_owned_image(image);
        if (binary_image_digest(image) != kMappedDigest) return std::nullopt;
        if (mapping == QualifiedAddMapping::kRange) {
            extend_owned_mapping(image);
            if (binary_image_digest(image) != kRangeDigest) return std::nullopt;
        }
        return image;
    } catch (...) {
        return std::nullopt;
    }
}

bool decode_binary_lz4_block(std::span<const std::byte> input,
                             std::span<std::byte> output) noexcept {
    if (input.empty() || input.size() > kMaxBinaryImageBytes ||
        output.size() > kMaxBinaryImageBytes)
        return false;
    std::size_t source{}, destination{};
    auto length = [&](std::size_t base) -> std::optional<std::size_t> {
        if (base == 15) {
            unsigned int extension{};
            do {
                if (source == input.size()) return std::nullopt;
                extension = std::to_integer<unsigned int>(input[source++]);
                if (extension > output.size() || base > output.size() - extension)
                    return std::nullopt;
                base += extension;
            } while (extension == 255);
        }
        return base;
    };
    while (source < input.size()) {
        const auto token = std::to_integer<unsigned int>(input[source++]);
        const auto literals = length(token >> 4);
        if (!literals || *literals > input.size() - source ||
            *literals > output.size() - destination)
            return false;
        if (*literals) std::memcpy(output.data() + destination, input.data() + source, *literals);
        source += *literals;
        destination += *literals;
        if (source == input.size()) return destination == output.size();
        if (input.size() - source < 2) return false;
        const auto distance = std::to_integer<unsigned int>(input[source]) |
                              (std::to_integer<unsigned int>(input[source + 1]) << 8);
        source += 2;
        if (!distance || distance > destination) return false;
        const auto match = length(token & 15);
        const auto remaining = output.size() - destination;
        if (!match || remaining < 4 || *match > remaining - 4) return false;
        // memmove is wrong for matches whose later bytes are produced by the
        // same match (for example, a distance-one run).
        for (std::size_t index = 0; index != *match + 4; ++index) {
            output[destination] = output[destination - distance];
            ++destination;
        }
    }
    return false;
}

std::optional<std::vector<std::byte>> rewrite_qualified_add_fatbin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedAddMapping mapping) noexcept {
    constexpr std::size_t fatbin_bytes = 5072120;
    constexpr std::size_t member_offset = 0x20fe28;
    constexpr std::size_t member_bytes = 642986;
    constexpr BinaryImageDigest digest{0x97881de5, 0x6d81ee99, 0x671bfe20, 0xb6e1ccca,
                                       0xb8654576, 0xabb7e5f1, 0xdc441088, 0x940b2b62};
    static_assert(member_offset + member_bytes < fatbin_bytes);
    if (compute_capability != 70 || original.size() != fatbin_bytes) return std::nullopt;
    try {
        const std::vector<std::byte> owned(original.begin(), original.end());
        if (binary_image_digest(owned) != digest) return std::nullopt;
        std::vector<std::byte> cubin(kOriginalBytes);
        if (!decode_binary_lz4_block(std::span{owned}.subspan(member_offset, member_bytes), cubin))
            return std::nullopt;
        return rewrite_qualified_add_cubin(cubin, compute_capability, mapping);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::vector<std::byte>> rewrite_qualified_gemm_cubin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedGemmMapping mapping) noexcept {
    static_assert(sizeof(Elf64_Shdr) == 64);
    if (compute_capability != 70 || original.size() != kGemmOriginalBytes ||
        std::endian::native != std::endian::little ||
        (mapping != QualifiedGemmMapping::kPlane && mapping != QualifiedGemmMapping::kRange))
        return std::nullopt;
    try {
        std::vector<std::byte> image(original.begin(), original.end());
        if (binary_image_digest(image) != kGemmOriginalDigest) return std::nullopt;
        map_owned_gemm_image(image);
        if (binary_image_digest(image) != kGemmMappedDigest) return std::nullopt;
        if (mapping == QualifiedGemmMapping::kRange) {
            map_owned_gemm_ranges(image);
            if (binary_image_digest(image) != kGemmRangeDigest) return std::nullopt;
        }
        return image;
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<std::vector<std::byte>> rewrite_qualified_gemm_fatbin(
    std::span<const std::byte> original, unsigned int compute_capability,
    QualifiedGemmMapping mapping) noexcept {
    constexpr std::size_t fatbin_bytes = 364384;
    constexpr std::size_t member_offset = 0x50;
    constexpr std::size_t member_bytes = 0x2c6df;
    constexpr BinaryImageDigest digest{0xc7712d58, 0x7387839b, 0x7678a8a5, 0xff4faf57,
                                       0x45a31c37, 0x4a961290, 0x0a4ba7cb, 0x8a17060d};
    static_assert(member_offset + member_bytes < fatbin_bytes);
    if (compute_capability != 70 || original.size() != fatbin_bytes) return std::nullopt;
    try {
        const std::vector<std::byte> owned(original.begin(), original.end());
        if (binary_image_digest(owned) != digest) return std::nullopt;
        std::vector<std::byte> cubin(kGemmOriginalBytes);
        if (!decode_binary_lz4_block(std::span{owned}.subspan(member_offset, member_bytes), cubin))
            return std::nullopt;
        return rewrite_qualified_gemm_cubin(cubin, compute_capability, mapping);
    } catch (...) {
        return std::nullopt;
    }
}

}  // namespace glimmer::interceptor
