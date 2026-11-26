#include "internal/qualified_cubin.h"
#include "internal/qualified_cubin_source.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace {
using namespace glimmer::interceptor;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::vector<std::byte> read_image(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "image open");
    const auto size = input.tellg();
    require(size > 0 && size <= static_cast<std::streamoff>(kMaxBinaryImageBytes), "image size");
    std::vector<std::byte> image(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size()));
    require(input.good(), "image read");
    return image;
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
        require(::mprotect(end(), page_bytes, PROT_NONE) == 0, "source protection");
    }
    std::byte* end() {
        return static_cast<std::byte*>(address) + page_bytes;
    }
    void* address{};
    std::size_t page_bytes{};
};
void default_rejections() {
    for (const std::size_t bytes : {0U, 64U, 364384U, 3055056U}) {
        std::vector<std::byte> input(bytes);
        const auto before = input;
        for (const auto mapping : {QualifiedGemmMapping::kPlane, QualifiedGemmMapping::kRange,
                                   static_cast<QualifiedGemmMapping>(99)})
            require(!rewrite_qualified_gemm_cubin(input, 70, mapping) &&
                        !rewrite_qualified_gemm_fatbin(input, 70, mapping) && input == before,
                    "unknown size/content/mapping cannot qualify or mutate input");
    }
    GuardedSource source;
    source.protect();
    errno = EDOM;
    for (const void* pointer :
         std::array<const void*, 6>{nullptr, reinterpret_cast<void*>(1), source.address,
                                    source.end(), source.end() - 8, source.end() - 32}) {
        require(!snapshot_qualified_gemm_cubin(pointer) && errno == EDOM,
                "unreadable/partial/unknown source preserves errno and declines");
    }
    std::array<std::uint64_t, 3> wrapper{0x00000001466243b1, 0, 0};
    require(!snapshot_qualified_gemm_cubin(wrapper.data()) && errno == EDOM,
            "null wrapped data declines");
}
void qualified_images(const char* original_path, const char* expected_path, const char* fatbin_path,
                      QualifiedGemmMapping mapping = QualifiedGemmMapping::kPlane) {
    auto original = read_image(original_path);
    const auto before = original;
    const auto expected = read_image(expected_path);
    auto mapped = rewrite_qualified_gemm_cubin(original, 70, mapping);
    require(!rewrite_qualified_gemm_cubin(original, 70, static_cast<QualifiedGemmMapping>(99)),
            "qualified bytes do not authorize unknown mapping");
    require(mapped && *mapped == expected && original == before,
            "exact independent raw-image oracle");
    errno = EDOM;
    auto snapshot = snapshot_qualified_gemm_cubin(original.data(), mapping);
    require(snapshot && *snapshot == expected && errno == EDOM, "raw snapshot and errno");
    for (unsigned device : {0U, 60U, 75U, 80U, 90U})
        require(!rewrite_qualified_gemm_cubin(original, device, mapping),
                "unqualified device rejects");
    require(
        !rewrite_qualified_gemm_cubin(std::span{original}.first(original.size() - 1), 70, mapping),
        "truncated raw image rejects");
    original.push_back(std::byte{});
    require(!rewrite_qualified_gemm_cubin(original, 70, mapping), "appended raw image rejects");
    original.pop_back();
    for (const std::size_t offset : {0U, 40U, 64U, 0xf5c10U, 0xf7280U, 0xfc510U, 0x2e859aU,
                                     0x2e859eU, 0x2e85aeU, 0x2e98d8U, 3055055U}) {
        original[offset] ^= std::byte{1};
        require(!rewrite_qualified_gemm_cubin(original, 70, mapping) &&
                    !snapshot_qualified_gemm_cubin(original.data(), mapping) && errno == EDOM,
                "changed raw header/code/metadata/relocations/body reject");
        original[offset] ^= std::byte{1};
    }
    GuardedSource guarded;
    std::memcpy(guarded.end() - 64, original.data(), 64);
    guarded.protect();
    require(!snapshot_qualified_gemm_cubin(guarded.end() - 64, mapping) && errno == EDOM,
            "valid raw header does not authorize inaccessible body");
    require(!rewrite_qualified_gemm_cubin(*mapped, 70, mapping) &&
                !rewrite_qualified_add_cubin(original, 70),
            "mapping cannot repeat or select add");

    auto fatbin = read_image(fatbin_path);
    const auto fatbin_before = fatbin;
    auto decoded = rewrite_qualified_gemm_fatbin(fatbin, 70, mapping);
    require(decoded && *decoded == expected && fatbin == fatbin_before,
            "full-envelope and decompressed cubin independently qualify");
    snapshot = snapshot_qualified_gemm_cubin(fatbin.data(), mapping);
    require(snapshot && *snapshot == expected && errno == EDOM, "raw envelope snapshot");
    std::array<std::uint64_t, 3> wrapper{0x00000001466243b1,
                                         reinterpret_cast<std::uintptr_t>(fatbin.data()), 0};
    auto wrapped = snapshot_qualified_gemm_cubin(wrapper.data(), mapping);
    require(wrapped && *wrapped == expected && errno == EDOM, "actual complete SDK wrapper");
    wrapper[2] = 1;
    require(!snapshot_qualified_gemm_cubin(wrapper.data(), mapping) && errno == EDOM,
            "auxiliary source cannot be ignored");
    wrapper[2] = 0;
    wrapper[0] = 0x00000002466243b1;
    require(!snapshot_qualified_gemm_cubin(wrapper.data(), mapping),
            "unknown wrapper version rejects");
    wrapper[0] = 0x00000001466243b1;
    for (const auto pointer :
         {reinterpret_cast<std::uintptr_t>(wrapper.data()),
          reinterpret_cast<std::uintptr_t>(original.data()),
          reinterpret_cast<std::uintptr_t>(guarded.end()), std::uintptr_t{1}}) {
        wrapper[1] = pointer;
        require(!snapshot_qualified_gemm_cubin(wrapper.data(), mapping) && errno == EDOM,
                "nested wrapper, wrapped raw cubin or unreadable payload rejects");
    }
    wrapper[1] = reinterpret_cast<std::uintptr_t>(fatbin.data());
    std::memcpy(guarded.end() - 16, wrapper.data(), 16);
    require(!snapshot_qualified_gemm_cubin(guarded.end() - 16, mapping), "partial wrapper rejects");
    std::memcpy(guarded.end() - 16, fatbin.data(), 16);
    require(!snapshot_qualified_gemm_cubin(guarded.end() - 16, mapping),
            "inaccessible envelope body rejects");
    require(!rewrite_qualified_gemm_fatbin(std::span{fatbin}.first(fatbin.size() - 1), 70, mapping),
            "truncated envelope rejects");
    fatbin.push_back(std::byte{});
    require(!rewrite_qualified_gemm_fatbin(fatbin, 70, mapping), "appended envelope rejects");
    fatbin.pop_back();
    wrapper[1] = reinterpret_cast<std::uintptr_t>(fatbin.data());
    for (const std::size_t offset : {0U, 8U, 16U, 0x50U, 0x2c72fU, 0x2c730U, 0x2c780U, 364383U}) {
        fatbin[offset] ^= std::byte{1};
        require(!rewrite_qualified_gemm_fatbin(fatbin, 70, mapping) &&
                    !snapshot_qualified_gemm_cubin(fatbin.data(), mapping) &&
                    !snapshot_qualified_gemm_cubin(wrapper.data(), mapping) && errno == EDOM,
                "selected/unselected member or padding changes reject");
        fatbin[offset] ^= std::byte{1};
    }
    require(!rewrite_qualified_gemm_fatbin(fatbin, 75, mapping) &&
                !rewrite_qualified_add_fatbin(fatbin, 70),
            "wrong device/image family rejects");
    std::ranges::fill(original, std::byte{});
    std::ranges::fill(fatbin, std::byte{});
    wrapper.fill(0);
    require(*mapped == expected && *decoded == expected && *snapshot == expected &&
                *wrapped == expected,
            "all outputs survive caller storage overwrite");
    std::cout << "qualified_gemm raw=1 fatbin=1 wrapper=1 exact_output=1 original_preserved=1 "
                 "changed_sources_rejected=1 protected_sources_rejected=1 owned_output=1\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc != 1 && argc != 4 && argc != 5) return 2;
    try {
        default_rejections();
        if (argc >= 4) qualified_images(argv[1], argv[2], argv[3]);
        if (argc == 5) qualified_images(argv[1], argv[4], argv[3], QualifiedGemmMapping::kRange);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
