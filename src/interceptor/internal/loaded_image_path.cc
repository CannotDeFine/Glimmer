#include "internal/loaded_image_path.h"

#include <dlfcn.h>
#include <link.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>

namespace glimmer::interceptor {
namespace {

struct PreserveErrno {
    const int original{errno};
    ~PreserveErrno() {
        errno = original;
    }
};

struct ImageQuery {
    std::uintptr_t address{};
    std::array<char, 4096> name{};
    bool has_name{};
    bool is_main{};
};

int find_image(dl_phdr_info* image, std::size_t, void* opaque) noexcept {
    auto& query = *static_cast<ImageQuery*>(opaque);
    for (std::size_t index = 0; index < image->dlpi_phnum; ++index) {
        const auto& segment = image->dlpi_phdr[index];
        if (segment.p_type != PT_LOAD ||
            segment.p_vaddr > std::numeric_limits<std::uintptr_t>::max() - image->dlpi_addr) {
            continue;
        }
        const auto begin = static_cast<std::uintptr_t>(image->dlpi_addr + segment.p_vaddr);
        if (query.address < begin || query.address - begin >= segment.p_memsz) {
            continue;
        }
        if (image->dlpi_name != nullptr) {
            const auto length = ::strnlen(image->dlpi_name, query.name.size());
            if (length < query.name.size()) {
                std::memcpy(query.name.data(), image->dlpi_name, length + 1);
                query.has_name = true;
                query.is_main = length == 0;
            }
        }
        return 1;
    }
    return 0;
}

}  // namespace

std::optional<std::string> loaded_image_path(const void* address) {
    const PreserveErrno preserve;
    if (address == nullptr) {
        return std::nullopt;
    }
    ImageQuery query;
    query.address = reinterpret_cast<std::uintptr_t>(address);
    (void)::dl_iterate_phdr(find_image, &query);
    if (query.has_name && !query.is_main) {
        return std::string(query.name.data());
    }
    if (query.is_main) {
        const auto length = ::readlink("/proc/self/exe", query.name.data(), query.name.size());
        if (length > 0 && static_cast<std::size_t>(length) < query.name.size()) {
            return std::string(query.name.data(), static_cast<std::size_t>(length));
        }
    }
    Dl_info image{};
    if (::dladdr(address, &image) == 0 || image.dli_fname == nullptr) {
        return std::nullopt;
    }
    return std::string(image.dli_fname);
}

}  // namespace glimmer::interceptor
