#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "internal/driver_dispatch.h"
#include "internal/nvml_dispatch.h"
#include "internal/runtime_api_bridge.h"
#include "internal/symbol_registry.h"

#include <dlfcn.h>
#include <limits.h>
#include <link.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <string_view>

namespace glimmer::interceptor {

namespace {

// ASan resolves mmap through dlsym before its shadow memory is initialized.
// This leaf and the ABI entry must not execute instrumented code on RTLD_NEXT.
[[nodiscard]] __attribute__((no_sanitize("address", "undefined"))) DlsymFunction
resolve_bootstrap_dlsym() noexcept {
    static const auto native =
        reinterpret_cast<DlsymFunction>(dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    return native;
}

template <std::size_t Size>
[[nodiscard]] __attribute__((no_sanitize("address", "undefined"))) bool bootstrap_name_has_prefix(
    const char* name, const char (&prefix)[Size]) noexcept {
    for (std::size_t index = 0; index + 1 < Size; ++index) {
        if (name[index] != prefix[index]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] __attribute__((no_sanitize("address", "undefined"))) bool
bootstrap_name_needs_routing(const char* name) noexcept {
    const bool is_driver = name[0] == 'c' && name[1] == 'u' && name[2] >= 'A' && name[2] <= 'Z';
    return is_driver || bootstrap_name_has_prefix(name, "cuda") ||
           bootstrap_name_has_prefix(name, "nvml") ||
           bootstrap_name_has_prefix(name, "__cudaLaunchKernel");
}

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

struct NormalizedProxyPath {
    std::array<char, PATH_MAX> characters{};
    std::size_t length{};
    bool is_valid{};
};

[[nodiscard]] NormalizedProxyPath normalize_proxy_path(const char* path) noexcept {
    if (path == nullptr || path[0] != '/') return {};
    const auto length = strnlen(path, PATH_MAX);
    if (length == PATH_MAX) return {};
    const std::string_view input{path, length};
    NormalizedProxyPath output;
    output.characters[0] = '/';
    output.length = 1;
    for (std::size_t begin = 1; begin < input.size();) {
        const auto separator = input.find('/', begin);
        const auto end = separator == std::string_view::npos ? input.size() : separator;
        const auto component = input.substr(begin, end - begin);
        begin = end + 1;
        if (component.empty() || component == ".") continue;
        if (component == "..") return {};
        if (output.length > 1) output.characters[output.length++] = '/';
        if (component.size() >= output.characters.size() - output.length) return {};
        std::memcpy(output.characters.data() + output.length, component.data(), component.size());
        output.length += component.size();
    }
    output.is_valid = true;
    return output;
}

[[nodiscard]] bool is_profile_driver_proxy_handle(void* handle) noexcept {
    static const auto configured =
        normalize_proxy_path(std::getenv("GLIMMER_PROFILE_DRIVER_PROXY_PATH"));
    if (!configured.is_valid || !cooperative_runtime_preview_requested() || handle == nullptr ||
        handle == RTLD_NEXT) {
        return false;
    }
    void* storage{};
    if (dlinfo(handle, RTLD_DI_LINKMAP, &storage) != 0 || storage == nullptr) {
        return false;
    }
    const auto* module = static_cast<const struct link_map*>(storage);
    const auto loaded = normalize_proxy_path(module->l_name);
    return loaded.is_valid && std::string_view{loaded.characters.data(), loaded.length} ==
                                  std::string_view{configured.characters.data(), configured.length};
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

[[nodiscard]] bool is_nvml_handle(void* handle) noexcept {
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
           std::string_view(link_map->l_name).find("libnvidia-ml.so") != std::string_view::npos;
}

[[nodiscard]] bool is_called_from_cuda_driver(void* caller) noexcept {
    Dl_info caller_info{};
    if (dladdr(caller, &caller_info) == 0 || caller_info.dli_fname == nullptr) {
        return false;
    }
    return std::string_view(caller_info.dli_fname).find("libcuda.so") != std::string_view::npos;
}

[[nodiscard]] bool is_called_from_cuda_runtime(void* caller) noexcept {
    Dl_info caller_info{};
    if (dladdr(caller, &caller_info) == 0 || caller_info.dli_fname == nullptr) {
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
    return symbol.starts_with("cuda") || symbol.starts_with("__cudaLaunchKernel");
}

[[nodiscard]] bool is_nvml_symbol_name(const char* name) noexcept {
    return name != nullptr && std::string_view(name).starts_with("nvml");
}

}  // namespace

}  // namespace glimmer::interceptor

// The entry supplies the original caller and a private stack output. Native
// lookup happens only after the entry restores that caller's return address.
// Keep this bootstrap free of sanitizer instrumentation and libc's nonnull
// declaration for dlsym; covered CUDA classification remains instrumented.
extern "C" __attribute__((visibility("hidden"), no_sanitize("address", "undefined"))) void*
glimmer_dispatch_dlsym(void* handle, const char* name, void* caller,
                       glimmer::interceptor::DlsymFunction* forward) noexcept {
    *forward = nullptr;
    if (name == nullptr) {
        return nullptr;
    }
    if (handle == RTLD_NEXT || !glimmer::interceptor::bootstrap_name_needs_routing(name)) {
        *forward = glimmer::interceptor::resolve_bootstrap_dlsym();
        return nullptr;
    }
    try {
        const bool called_from_driver = glimmer::interceptor::is_called_from_cuda_driver(caller);
        const bool inside_driver_call = glimmer::interceptor::is_inside_driver_call();
        const bool called_from_runtime = glimmer::interceptor::is_called_from_cuda_runtime(caller);
        const bool inside_nvml_call = glimmer::interceptor::is_inside_nvml_call();
        if (!called_from_driver &&
            (called_from_runtime ||
             (!inside_driver_call && glimmer::interceptor::is_inside_runtime_call())) &&
            glimmer::interceptor::runtime_persistent_lookup_is_enabled() &&
            (handle == RTLD_DEFAULT || glimmer::interceptor::is_cuda_driver_handle(handle) ||
             (glimmer::interceptor::is_driver_symbol_name(name) &&
              glimmer::interceptor::is_profile_driver_proxy_handle(handle))) &&
            (std::string_view(name) == "cuLaunchKernel" ||
             std::string_view(name) == "cuLaunchKernel_ptsz" ||
             std::string_view(name) == "cuGetProcAddress" ||
             std::string_view(name) == "cuGetProcAddress_v2" ||
             glimmer::interceptor::cooperative_runtime_preview_requested())) {
            if (std::string_view(name) == "cuGetProcAddress") {
                return reinterpret_cast<void*>(
                    &glimmer::interceptor::runtime_driver_get_proc_address);
            }
            if (std::string_view(name) == "cuGetProcAddress_v2") {
                return reinterpret_cast<void*>(
                    &glimmer::interceptor::runtime_driver_get_proc_address_v2);
            }
            if (auto* symbol = glimmer::interceptor::find_interceptor_symbol(name);
                symbol != nullptr) {
                return symbol;
            }
        }
        if (!called_from_driver && !called_from_runtime && !inside_driver_call &&
            !inside_nvml_call && !glimmer::interceptor::is_inside_runtime_call() &&
            (handle == RTLD_DEFAULT ||
             (glimmer::interceptor::is_cuda_driver_handle(handle) &&
              glimmer::interceptor::is_driver_symbol_name(name)) ||
             (glimmer::interceptor::is_driver_symbol_name(name) &&
              glimmer::interceptor::is_profile_driver_proxy_handle(handle)) ||
             (glimmer::interceptor::is_cuda_runtime_handle(handle) &&
              glimmer::interceptor::is_runtime_symbol_name(name)) ||
             (glimmer::interceptor::is_nvml_handle(handle) &&
              glimmer::interceptor::is_nvml_symbol_name(name)))) {
            if (std::string_view(name) == "cudaGraphInstantiate") {
                return glimmer::interceptor::runtime_graph_constructor_lookup(handle);
            }
            if (std::string_view(name) == "cudaGraphExecUpdate") {
                return glimmer::interceptor::runtime_graph_update_lookup(handle);
            }
            if (void* intercepted_symbol = glimmer::interceptor::find_interceptor_symbol(name);
                intercepted_symbol != nullptr) {
                return intercepted_symbol;
            }
        }

        *forward = glimmer::interceptor::resolve_real_dlsym();
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

#ifndef GLIMMER_DLSYM_NATIVE_TRAMPOLINE
// Other architectures retain the previous C++ forwarding behavior. Their
// caller-relative native lookup is not qualified by the x86-64 trampoline.
extern "C" void* glimmer_dlsym(void* handle, const char* name) noexcept __asm__("dlsym");
extern "C" __attribute__((no_sanitize("address", "undefined"))) void* glimmer_dlsym(
    void* handle, const char* name) noexcept {
    glimmer::interceptor::DlsymFunction forward{};
    void* result = glimmer_dispatch_dlsym(handle, name, __builtin_return_address(0), &forward);
    return forward == nullptr ? result : forward(handle, name);
}
#endif
