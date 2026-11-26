#pragma once

#include "cudnn_profile_providers.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <link.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

// Private process-local loaded-provider proof shared by composition and lookup.
// It uses the generated exact ELF manifest, not caller SDK entry points.
#pragma GCC visibility push(hidden)
namespace glimmer::internal::cudnn_profile {
namespace detail {
inline const pid_t kProviderProcess{::getpid()};
inline bool is_dnn_name(const char* filename) noexcept {
    if (!filename) return false;
    const auto* separator = std::strrchr(filename, '/');
    const std::string_view name{separator ? separator + 1 : filename};
    return name.starts_with("libcudnn.so") ||
           (name.starts_with("libcudnn_") && name.find(".so") != std::string_view::npos);
}
// Loader callbacks hold loader exclusion, never State's resource mutex. Read
// potentially mutable mapped ELF bytes through the kernel, not raw dereferences.
constexpr std::size_t kDnnMaxProgramHeaders = 128, kDnnMaxDynamicEntries = 512;
constexpr std::size_t kDnnMaxNoteBytes = 144, kDnnMaxSonameBytes = 256;
struct DnnProfileCache {
    unsigned long long additions{}, subtractions{};
    bool is_valid{}, is_matched{};
};
inline thread_local DnnProfileCache dnn_profile_cache;
inline bool dnn_read_self(std::uintptr_t source, void* output, std::size_t bytes) noexcept {
    if (!source || bytes > UINTPTR_MAX - source) return false;
    iovec local{output, bytes};
    // NOLINTNEXTLINE(performance-no-int-to-ptr): kernel-checked source address.
    iovec remote{reinterpret_cast<void*>(source), bytes};
    for (unsigned int attempt = 0; attempt != 8; ++attempt) {
        const auto result =
            ::syscall(SYS_process_vm_readv, kProviderProcess, &local, 1, &remote, 1, 0);
        if (result >= 0) return static_cast<std::size_t>(result) == bytes;
        if (errno != EINTR) return false;
    }
    return false;
}
inline bool dnn_readable(std::uintptr_t base, std::span<const ElfW(Phdr)> headers,
                         std::uintptr_t address, std::size_t bytes) noexcept {
    if (bytes > UINTPTR_MAX - address) return false;
    for (const auto& header : headers) {
        if (header.p_type != PT_LOAD || !(header.p_flags & PF_R) ||
            header.p_vaddr > UINTPTR_MAX - base)
            continue;
        const auto first = base + static_cast<std::uintptr_t>(header.p_vaddr);
        if (address >= first && address - first <= header.p_memsz &&
            bytes <= header.p_memsz - (address - first))
            return true;
    }
    return false;
}
struct DnnElfIdentity {
    std::array<char, kDnnMaxSonameBytes> soname{};
    std::array<char, 129> build_id{};
    bool has_soname{}, has_build_id{}, is_valid{};
};
inline bool dnn_configured_build_id(const dl_phdr_info& info, std::span<const ElfW(Phdr)> headers,
                                    DnnElfIdentity& identity) noexcept {
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(ElfW(Nhdr)) == 12);
    if (!info.dlpi_name || !identity.has_soname) return false;
    const ProviderIdentity* selected{};
    for (const auto& expected : kCudnnProfileProviders) {
        if (expected.path != info.dlpi_name || expected.soname != identity.soname.data()) continue;
        if (selected) return false;
        selected = &expected;
    }
    if (!selected) return false;
    // The trusted build manifest records the section's virtual address, not
    // its file offset. Some vendor ELFs have stale PT_NOTE metadata; never
    // substitute another note, scan runtime section tables or read a file.
    const auto base = static_cast<std::uintptr_t>(info.dlpi_addr);
    const auto bytes = selected->note_bytes;
    constexpr auto descriptor_offset = sizeof(ElfW(Nhdr)) + 4;
    if (selected->note_address > UINTPTR_MAX - base || bytes < descriptor_offset ||
        bytes > kDnnMaxNoteBytes)
        return false;
    const auto address = base + selected->note_address;
    std::array<std::byte, kDnnMaxNoteBytes> storage{};
    if (!dnn_readable(base, headers, address, bytes) ||
        !dnn_read_self(address, storage.data(), bytes))
        return false;
    ElfW(Nhdr) note{};
    std::memcpy(&note, storage.data(), sizeof(note));
    if (note.n_type != NT_GNU_BUILD_ID || note.n_namesz != 4 || !note.n_descsz ||
        note.n_descsz > (identity.build_id.size() - 1) / 2 ||
        std::memcmp(storage.data() + sizeof(note), "GNU\0", 4) != 0)
        return false;
    const auto descriptions = static_cast<std::size_t>(note.n_descsz);
    const auto aligned_descriptions = (descriptions + 3U) & ~std::size_t{3};
    if (bytes != descriptor_offset + aligned_descriptions) return false;
    for (auto offset = descriptor_offset + descriptions; offset < bytes; ++offset)
        if (storage[offset] != std::byte{}) return false;
    constexpr std::string_view hexadecimal{"0123456789abcdef"};
    for (std::size_t index = 0; index < descriptions; ++index) {
        const auto value = std::to_integer<unsigned char>(storage[descriptor_offset + index]);
        identity.build_id[index * 2] = hexadecimal[value >> 4];
        identity.build_id[index * 2 + 1] = hexadecimal[value & 15U];
    }
    return true;
}
inline DnnElfIdentity dnn_elf_identity(const dl_phdr_info& info) noexcept {
    DnnElfIdentity result;
    std::array<ElfW(Phdr), kDnnMaxProgramHeaders> storage{};
    if (!info.dlpi_phnum || info.dlpi_phnum > storage.size() ||
        !dnn_read_self(reinterpret_cast<std::uintptr_t>(info.dlpi_phdr), storage.data(),
                       static_cast<std::size_t>(info.dlpi_phnum) * sizeof(storage[0])))
        return result;
    const std::span headers{storage.data(), static_cast<std::size_t>(info.dlpi_phnum)};
    const auto base = static_cast<std::uintptr_t>(info.dlpi_addr);
    std::uintptr_t string_table{};
    std::size_t string_bytes{}, soname_offset{};
    bool has_dynamic{}, has_soname_offset{};
    for (const auto& header : headers) {
        if (header.p_type != PT_DYNAMIC) continue;
        if (has_dynamic || header.p_vaddr > UINTPTR_MAX - base ||
            header.p_memsz % sizeof(ElfW(Dyn)) ||
            header.p_memsz > kDnnMaxDynamicEntries * sizeof(ElfW(Dyn)))
            return result;
        has_dynamic = true;
        std::array<ElfW(Dyn), kDnnMaxDynamicEntries> dynamic{};
        const auto address = base + static_cast<std::uintptr_t>(header.p_vaddr);
        const auto bytes = static_cast<std::size_t>(header.p_memsz);
        if (!dnn_readable(base, headers, address, bytes) ||
            !dnn_read_self(address, dynamic.data(), bytes))
            return result;
        bool has_end{};
        for (std::size_t index = 0; index < bytes / sizeof(dynamic[0]); ++index) {
            const auto& item = dynamic[index];
            if (item.d_tag == DT_NULL) {
                has_end = true;
                break;
            }
            if (item.d_tag == DT_STRTAB) {
                if (string_table) return result;
                string_table = static_cast<std::uintptr_t>(item.d_un.d_ptr);
            } else if (item.d_tag == DT_STRSZ) {
                if (string_bytes || item.d_un.d_val > SIZE_MAX) return result;
                string_bytes = static_cast<std::size_t>(item.d_un.d_val);
            } else if (item.d_tag == DT_SONAME) {
                if (has_soname_offset || item.d_un.d_val > SIZE_MAX) return result;
                soname_offset = static_cast<std::size_t>(item.d_un.d_val);
                has_soname_offset = true;
            }
        }
        if (!has_end) return result;
    }
    if (!has_soname_offset) {
        result.is_valid = true;  // Executables and unrelated objects may have no SONAME.
        return result;
    }
    if (!string_table || !string_bytes || soname_offset >= string_bytes) return result;
    if (!dnn_readable(base, headers, string_table, string_bytes)) {
        // glibc relocates most DT_STRTAB addresses. The VDSO and some images
        // retain relative values; either form must prove its own PT_LOAD range.
        if (string_table > UINTPTR_MAX - base) return result;
        string_table += base;
        if (!dnn_readable(base, headers, string_table, string_bytes)) return result;
    }
    const auto available = std::min(string_bytes - soname_offset, result.soname.size());
    if (!dnn_read_self(string_table + soname_offset, result.soname.data(), available) ||
        !std::memchr(result.soname.data(), '\0', available))
        return result;
    result.has_soname = true;
    result.has_build_id = dnn_configured_build_id(info, headers, result);
    result.is_valid = true;
    return result;
}
inline bool dnn_identity_matches(const dl_phdr_info& info,
                                 const DnnElfIdentity& identity) noexcept {
    if (!identity.is_valid || !identity.has_soname || !identity.has_build_id || !info.dlpi_name)
        return false;
    for (const auto& expected : glimmer::internal::cudnn_profile::kCudnnProfileProviders)
        if (expected.path == info.dlpi_name && expected.soname == identity.soname.data() &&
            expected.build_id == identity.build_id.data())
            return true;
    return false;
}
struct DnnProfileScan {
    bool is_first{true}, has_counters{}, has_provider{}, is_matched{true}, has_target{},
        has_target_image{};
    std::uintptr_t target{};
    unsigned long long additions{}, subtractions{};
};
inline int scan_dnn_profile(dl_phdr_info* info, std::size_t bytes, void* opaque) noexcept {
    auto& scan = *static_cast<DnnProfileScan*>(opaque);
    if (scan.is_first && !scan.has_target) {
        scan.is_first = false;
        scan.has_counters = bytes >= offsetof(dl_phdr_info, dlpi_subs) + sizeof(info->dlpi_subs);
        if (scan.has_counters) {
            scan.additions = info->dlpi_adds;
            scan.subtractions = info->dlpi_subs;
            if (dnn_profile_cache.is_valid && dnn_profile_cache.additions == scan.additions &&
                dnn_profile_cache.subtractions == scan.subtractions) {
                scan.has_provider = true;
                scan.is_matched = dnn_profile_cache.is_matched;
                return 1;
            }
        }
    }
    if (scan.has_target && scan.target != info->dlpi_addr) return 0;
    const auto identity = dnn_elf_identity(*info);
    const bool is_candidate = is_dnn_name(info->dlpi_name) ||
                              (identity.has_soname && is_dnn_name(identity.soname.data()));
    if (scan.has_target) {
        scan.has_target_image = true;
        scan.is_matched = is_candidate && dnn_identity_matches(*info, identity);
        return 1;
    }
    if (!is_candidate) return 0;
    scan.has_provider = true;
    scan.is_matched = scan.is_matched && dnn_identity_matches(*info, identity);
    return scan.is_matched ? 0 : 1;
}
}  // namespace detail

// Caller must not hold a frontend resource lock. dl_iterate_phdr provides the
// loader snapshot; additions/removals invalidate the thread-local parsed cache.
inline bool process_providers_match() noexcept {
    if (::getpid() != detail::kProviderProcess) return false;
    detail::DnnProfileScan scan;
    (void)::dl_iterate_phdr(detail::scan_dnn_profile, &scan);
    const bool is_matched = scan.has_provider && scan.is_matched;
    if (scan.has_counters)
        detail::dnn_profile_cache = {scan.additions, scan.subtractions, true, is_matched};
    return is_matched;
}

// A public cached-lookup caller additionally proves LM_ID_BASE before using
// this address lookup; an unenumerated namespace is never native fallback.
inline bool provider_base_matches(std::uintptr_t base) noexcept {
    if (::getpid() != detail::kProviderProcess) return false;
    detail::DnnProfileScan scan;
    scan.has_target = true;
    scan.target = base;
    (void)::dl_iterate_phdr(detail::scan_dnn_profile, &scan);
    return scan.has_target_image && scan.is_matched;
}
}  // namespace glimmer::internal::cudnn_profile
#pragma GCC visibility pop
