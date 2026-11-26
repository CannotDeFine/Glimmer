#include "internal/qualified_cubin.h"
#include "internal/qualified_cubin_source.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {
using namespace glimmer::interceptor;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::string hex_digest(std::span<const std::byte> image) {
    const auto digest = binary_image_digest(image);
    require(digest.has_value(), "digest size rejected");
    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (auto word : *digest) text << std::setw(8) << word;
    return text.str();
}
std::vector<std::byte> read_image(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "image open failed");
    const auto size = input.tellg();
    require(size > 0 && size <= static_cast<std::streamoff>(kMaxBinaryImageBytes), "image size");
    std::vector<std::byte> image(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size()));
    require(input.good(), "image read failed");
    return image;
}
void digest_cases() {
    struct Case {
        std::size_t bytes;
        const char* expected;
    };
    // Independent Python hashlib oracle over byte[i] = i mod 256.
    constexpr std::array cases{
        Case{0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        Case{55, "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59"},
        Case{56, "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562"},
        Case{63, "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488"},
        Case{64, "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108"},
        Case{65, "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781"},
        Case{127, "92ca0fa6651ee2f97b884b7246a562fa71250fedefe5ebf270d31c546bfea976"},
        Case{128, "471fb943aa23c511f6f72f8d1652d9c880cfa392ad80503120547703e56a2be5"},
        Case{129, "5099c6a56203f9687f7d33f4bfdf576d31dc91f6b695ecea38b2770c87631135"},
        Case{kMaxBinaryImageBytes,
             "7d212b9c884f5c77896de960ae17cc341cda43b14d6a971f34ca29ebd4badf7f"}};
    for (const auto& item : cases) {
        std::vector<std::byte> image(item.bytes);
        for (std::size_t index = 0; index != image.size(); ++index)
            image[index] = static_cast<std::byte>(index & 0xff);
        require(hex_digest(image) == item.expected, "binary padding/length digest mismatch");
    }
    const std::string abc = "abc";
    require(hex_digest(std::as_bytes(std::span{abc.data(), abc.size()})) ==
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "abc known answer");
    std::vector<std::byte> million(1000000, std::byte{'a'});
    require(
        hex_digest(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "million-a known answer");
    std::vector<std::byte> oversized(kMaxBinaryImageBytes + 1);
    require(!binary_image_digest(oversized) && !rewrite_qualified_add_cubin(oversized, 70) &&
                !rewrite_qualified_add_fatbin(oversized, 70) &&
                !decode_binary_lz4_block(oversized, {}) &&
                !decode_binary_lz4_block(std::as_bytes(std::span{abc}), oversized),
            "bounded input");
    std::vector<std::byte> wrong(5336008);
    require(!rewrite_qualified_add_cubin(wrong, 70) && !rewrite_qualified_add_cubin({}, 70),
            "unqualified input");
}
void block_decoder_cases() {
    auto check_decode = [](std::vector<unsigned char> encoded, const std::string& expected) {
        const auto input = std::as_bytes(std::span{encoded});
        std::vector<std::byte> output(expected.size());
        require(decode_binary_lz4_block(input, output) &&
                    std::memcmp(output.data(), expected.data(), output.size()) == 0,
                "literal and overlapping match decoder result");
        for (std::size_t length = 0; length < input.size(); ++length)
            require(!decode_binary_lz4_block(input.first(length), output), "truncated sequence");
        if (!output.empty()) {
            output.pop_back();
            require(!decode_binary_lz4_block(input, output), "short output");
            output.resize(expected.size() + 1);
            require(!decode_binary_lz4_block(input, output), "oversized output");
        }
    };
    check_decode({0x50, 'h', 'e', 'l', 'l', 'o'}, "hello");
    check_decode({0x1f, 'a', 1, 0, 0, 0x50, 'b', 'c', 'd', 'e', 'f'},
                 std::string(20, 'a') + "bcdef");
    check_decode({0x22, 'a', 'b', 2, 0, 0x50, 'c', 'd', 'e', 'f', 'g'}, "ababababcdefg");
    check_decode({0x1f, 'a', 1, 0, 255, 26, 0x50, 'b', 'c', 'd', 'e', 'f'},
                 std::string(301, 'a') + "bcdef");
    std::vector<unsigned char> literal{0xf0, 255, 30};
    literal.insert(literal.end(), 300, 'x');
    check_decode(literal, std::string(300, 'x'));
    const std::array zero{std::byte{0}};
    require(decode_binary_lz4_block(zero, {}) && !decode_binary_lz4_block({}, {}),
            "empty block still needs a terminal token");
    for (const auto& malformed : std::vector<std::vector<unsigned char>>{{0xf0},
                                                                         {0xf0, 255},
                                                                         {0x10},
                                                                         {0x10, 'a', 0},
                                                                         {0x10, 'a', 0, 0},
                                                                         {0x10, 'a', 2, 0},
                                                                         {0x1f, 'a', 1, 0},
                                                                         {0x1f, 'a', 1, 0, 255},
                                                                         {0x10, 'a', 1, 0}}) {
        std::array<std::byte, 128> output{};
        require(!decode_binary_lz4_block(std::as_bytes(std::span{malformed}), output),
                "malformed literal, offset, match length or missing final sequence");
    }
    std::uint32_t random = 0x148;
    auto next = [&] {
        random = random * 1664525U + 1013904223U;
        return random;
    };
    for (unsigned int iteration = 0; iteration != 10000; ++iteration) {
        std::vector<std::byte> input(next() % 65);
        for (auto& byte : input) byte = static_cast<std::byte>(next() >> 24);
        const auto bytes = next() % 129;
        std::vector<std::byte> guarded(bytes + 32, std::byte{0x5a});
        (void)decode_binary_lz4_block(input, std::span{guarded}.subspan(16, bytes));
        for (std::size_t index = 0; index != 16; ++index)
            require(
                guarded[index] == std::byte{0x5a} && guarded[16 + bytes + index] == std::byte{0x5a},
                "bounded mutation cannot write outside output");
    }
}
class GuardedSource {
   public:
    GuardedSource() {
        const auto page = ::sysconf(_SC_PAGESIZE);
        require(page >= 64, "page size");
        page_bytes = static_cast<std::size_t>(page);
        address = ::mmap(nullptr, 2 * page_bytes, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(address != MAP_FAILED, "source mapping");
    }
    ~GuardedSource() {
        if (::munmap(address, 2 * page_bytes) != 0) std::terminate();
    }
    void protect() {
        require(
            ::mprotect(static_cast<std::byte*>(address) + page_bytes, page_bytes, PROT_NONE) == 0,
            "source protection");
    }
    void* address{};
    std::size_t page_bytes{};
};
void snapshot_rejections() {
    GuardedSource source;
    source.protect();
    errno = EDOM;
    require(!qualified_library_source_options(
                0, static_cast<std::byte*>(source.address) + source.page_bytes, 1) &&
                !qualified_library_source_options(
                    0, static_cast<std::byte*>(source.address) + source.page_bytes - 2, 1) &&
                errno == EDOM,
            "unreadable and partial library tags decline without dereference or errno changes");
    require(!snapshot_qualified_add_cubin(nullptr) &&
                !snapshot_qualified_add_cubin(reinterpret_cast<const void*>(1)) &&
                !snapshot_qualified_add_cubin(static_cast<std::byte*>(source.address) +
                                              source.page_bytes) &&
                !snapshot_qualified_add_cubin(static_cast<std::byte*>(source.address) +
                                              source.page_bytes - 32) &&
                !snapshot_qualified_add_cubin(source.address) && errno == EDOM,
            "unreadable/partial/unknown snapshot must decline without a fault or errno change");
}
void original_cases(const char* original_path, const char* expected_path,
                    QualifiedAddMapping mapping = QualifiedAddMapping::kSingleCta) {
    auto original = read_image(original_path);
    const auto before = original;
    const auto expected = read_image(expected_path);
    auto mapped = rewrite_qualified_add_cubin(original, 70, mapping);
    require(mapped && *mapped == expected && original == before,
            "independent exact mapped-image oracle");
    errno = EDOM;
    const auto snapshot = snapshot_qualified_add_cubin(original.data(), mapping);
    require(snapshot && *snapshot == expected && errno == EDOM, "owned source snapshot and errno");
    original[0x2eef20] ^= std::byte{1};
    require(!snapshot_qualified_add_cubin(original.data(), mapping) && errno == EDOM,
            "unchanged header cannot qualify altered code");
    original[0x2eef20] ^= std::byte{1};
    GuardedSource source;
    auto* short_image = static_cast<std::byte*>(source.address) + source.page_bytes - 64;
    std::memcpy(short_image, original.data(), 64);
    source.protect();
    errno = EDOM;
    require(!snapshot_qualified_add_cubin(short_image, mapping) && errno == EDOM,
            "qualified header cannot authorize reading an inaccessible body");
    require(!rewrite_qualified_add_cubin(original, 80, mapping) &&
                !rewrite_qualified_add_cubin(original, 0, mapping) &&
                !rewrite_qualified_add_cubin(original, 70, static_cast<QualifiedAddMapping>(255)),
            "wrong device");
    require(
        !rewrite_qualified_add_cubin(std::span{original}.first(original.size() - 1), 70, mapping),
        "truncated original");
    original.push_back(std::byte{});
    require(!rewrite_qualified_add_cubin(original, 70, mapping), "appended original");
    original.pop_back();
    for (std::size_t offset :
         {std::size_t{0}, std::size_t{0x966e8}, std::size_t{0x2eef20}, original.size() - 1}) {
        original[offset] ^= std::byte{1};
        require(!rewrite_qualified_add_cubin(original, 70, mapping), "changed original accepted");
        original[offset] ^= std::byte{1};
    }
    require(!rewrite_qualified_add_cubin(*mapped, 70, mapping),
            "private image cannot be rewritten twice");
    original.clear();
    require(*mapped == expected, "output retains no source reference");
    std::cout << "exact_original_to_mapped=1 changed_inputs_rejected=1 owned_output=1\n";
}
void fatbinary_cases(const char* fatbin_path, const char* expected_path,
                     QualifiedAddMapping mapping = QualifiedAddMapping::kSingleCta) {
    auto original = read_image(fatbin_path);
    const auto before = original;
    const auto expected = read_image(expected_path);
    auto mapped = rewrite_qualified_add_fatbin(original, 70, mapping);
    require(mapped && *mapped == expected && original == before,
            "exact compressed source produces independently qualified private image");
    require(!rewrite_qualified_add_fatbin(original, 80, mapping) &&
                !rewrite_qualified_add_fatbin(std::span{original}.first(original.size() - 1), 70,
                                              mapping),
            "fatbinary wrong device or truncation");
    errno = EDOM;
    auto snapshot = snapshot_qualified_add_cubin(original.data(), mapping);
    require(snapshot && *snapshot == expected && errno == EDOM, "raw fatbinary snapshot");
    std::array<std::uint64_t, 3> wrapper{0x00000001466243b1,
                                         reinterpret_cast<std::uintptr_t>(original.data()), 0};
    snapshot = snapshot_qualified_add_cubin(wrapper.data(), mapping);
    require(snapshot && *snapshot == expected && errno == EDOM, "SDK wrapper source snapshot");
    wrapper[2] = 1;
    require(!snapshot_qualified_add_cubin(wrapper.data(), mapping) && errno == EDOM,
            "auxiliary source must reject");
    wrapper[2] = 0;
    wrapper[0] = 0x00000002466243b1;
    require(!snapshot_qualified_add_cubin(wrapper.data(), mapping), "linked wrapper must reject");
    wrapper[0] = 0x00000001466243b1;
    for (std::uint64_t pointer :
         {std::uint64_t{0}, std::uint64_t{1}, reinterpret_cast<std::uintptr_t>(wrapper.data())}) {
        wrapper[1] = pointer;
        require(!snapshot_qualified_add_cubin(wrapper.data(), mapping),
                "missing, unreadable or nested source");
    }
    wrapper[1] = reinterpret_cast<std::uintptr_t>(original.data());
    GuardedSource source;
    auto* end = static_cast<std::byte*>(source.address) + source.page_bytes;
    std::memcpy(end - 16, wrapper.data(), 16);
    source.protect();
    require(!snapshot_qualified_add_cubin(end - 16, mapping),
            "partial wrapper cannot hide auxiliary source");
    std::memcpy(end - 16, original.data(), 16);
    require(!snapshot_qualified_add_cubin(end - 16, mapping),
            "complete envelope cannot prove readable body");
    wrapper[1] = reinterpret_cast<std::uintptr_t>(end - 16);
    require(!snapshot_qualified_add_cubin(wrapper.data(), mapping),
            "wrapped unreadable envelope body");
    original.push_back(std::byte{});
    require(!rewrite_qualified_add_fatbin(original, 70, mapping), "appended envelope");
    original.pop_back();
    for (const auto offset : {std::size_t{0}, std::size_t{0x50}, std::size_t{0x20fde8},
                              std::size_t{0x20fe28}, std::size_t{0x2acdd7}, original.size() - 1}) {
        original[offset] ^= std::byte{1};
        require(!rewrite_qualified_add_fatbin(original, 70, mapping) &&
                    !snapshot_qualified_add_cubin(original.data(), mapping),
                "changed header, other architecture, compressed member or padding rejects");
        original[offset] ^= std::byte{1};
    }
    original.clear();
    require(*mapped == expected && *snapshot == expected,
            "owned output survives source destruction");
    std::cout << "exact_fatbinary_to_mapped=1 original_preserved=1 wrapper_checked=1\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc != 1 && argc != 3 && argc != 4 && argc != 5) return 2;
    try {
        std::uint32_t preserved = 1;
        std::uint32_t unknown = 0;
        errno = EBUSY;
        require(qualified_library_source_options(0, nullptr, 0) &&
                    !qualified_library_source_options(1, nullptr, 0) &&
                    !qualified_library_source_options(0, &preserved, 2) &&
                    !qualified_library_source_options(0, nullptr, 1) && errno == EBUSY,
                "library source option bounds preserve errno");
        if (argc > 1) {
            require(qualified_library_source_options(0, &preserved, 1) &&
                        !qualified_library_source_options(0, &unknown, 1) && errno == EBUSY,
                    "only the memory ownership hint qualifies");
        }
        digest_cases();
        block_decoder_cases();
        snapshot_rejections();
        if (argc >= 3) original_cases(argv[1], argv[2]);
        if (argc >= 4) fatbinary_cases(argv[3], argv[2]);
        if (argc == 5) {
            original_cases(argv[1], argv[4], QualifiedAddMapping::kRange);
            fatbinary_cases(argv[3], argv[4], QualifiedAddMapping::kRange);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
