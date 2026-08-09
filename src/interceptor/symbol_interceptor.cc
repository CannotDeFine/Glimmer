#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_dispatch.h"
#include "internal/runtime_api_bridge.h"
#include "internal/symbol_registry.h"

#include <dlfcn.h>
#include <link.h>

#include <cstdint>
#include <string_view>

namespace glimmer::interceptor {

namespace {

[[nodiscard]] bool is_cuda_driver_handle(void* handle) noexcept {
    if (handle == nullptr || handle == RTLD_NEXT) {
        return false;
    }

    void* link_map_storage = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, reinterpret_cast<void*>(&link_map_storage)) != 0 ||
        link_map_storage == nullptr) {
        return false;
    }

    const auto* link_map = static_cast<const struct link_map*>(link_map_storage);
    return link_map->l_name != nullptr &&
           std::string_view(link_map->l_name).find("libcuda.so") != std::string_view::npos;
}

[[nodiscard]] bool is_cuda_runtime_handle(void* handle) noexcept {
    if (handle == nullptr || handle == RTLD_NEXT) {
        return false;
    }

    void* link_map_storage = nullptr;
    if (dlinfo(handle, RTLD_DI_LINKMAP, reinterpret_cast<void*>(&link_map_storage)) != 0 ||
        link_map_storage == nullptr) {
        return false;
    }

    const auto* link_map = static_cast<const struct link_map*>(link_map_storage);
    return link_map->l_name != nullptr &&
           std::string_view(link_map->l_name).find("libcudart.so") != std::string_view::npos;
}

[[nodiscard]] bool is_called_from_cuda_driver() noexcept {
    Dl_info caller_info{};
    if (dladdr(__builtin_return_address(0), &caller_info) == 0 ||
        caller_info.dli_fname == nullptr) {
        return false;
    }
    return std::string_view(caller_info.dli_fname).find("libcuda.so") != std::string_view::npos;
}

[[nodiscard]] bool is_called_from_cuda_runtime() noexcept {
    Dl_info caller_info{};
    if (dladdr(__builtin_return_address(0), &caller_info) == 0 ||
        caller_info.dli_fname == nullptr) {
        return false;
    }
    return std::string_view(caller_info.dli_fname).find("libcudart.so") != std::string_view::npos;
}

[[nodiscard]] bool is_driver_symbol_name(const char* name) noexcept {
    if (name == nullptr) {
        return false;
    }
    const std::string_view symbol{name};
    return symbol.size() >= 3 && symbol[0] == 'c' && symbol[1] == 'u' && symbol[2] >= 'A' &&
           symbol[2] <= 'Z';
}

[[nodiscard]] bool is_runtime_symbol_name(const char* name) noexcept {
    if (name == nullptr) {
        return false;
    }
    const std::string_view symbol{name};
    return symbol.starts_with("cuda");
}

}  // namespace

}  // namespace glimmer::interceptor

extern "C" void* dlsym(void* handle, const char* name) {
    const std::uintptr_t name_address = reinterpret_cast<std::uintptr_t>(name);
    if (name_address == 0U) {
        return nullptr;
    }
    try {
        const bool called_from_driver = glimmer::interceptor::is_called_from_cuda_driver();
        const bool inside_driver_call = glimmer::interceptor::is_inside_driver_call();
        const bool called_from_runtime = glimmer::interceptor::is_called_from_cuda_runtime();
        if (!called_from_driver && !called_from_runtime && !inside_driver_call &&
            !glimmer::interceptor::is_inside_runtime_call() &&
            (handle == RTLD_DEFAULT ||
             (glimmer::interceptor::is_cuda_driver_handle(handle) &&
              glimmer::interceptor::is_driver_symbol_name(name)) ||
             (glimmer::interceptor::is_cuda_runtime_handle(handle) &&
              glimmer::interceptor::is_runtime_symbol_name(name)))) {
            if (void* intercepted_symbol = glimmer::interceptor::find_interceptor_symbol(name);
                intercepted_symbol != nullptr) {
                return intercepted_symbol;
            }
        }

        const glimmer::interceptor::DlsymFunction real_dlsym =
            glimmer::interceptor::resolve_real_dlsym();
        return real_dlsym == nullptr ? nullptr : real_dlsym(handle, name);
    } catch (...) {
        return nullptr;
    }
}
