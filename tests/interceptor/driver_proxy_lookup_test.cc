#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {
struct CloseLibrary {
    void operator()(void* handle) const noexcept {
        if (handle != nullptr && dlclose(handle) != 0) std::abort();
    }
};
using Library = std::unique_ptr<void, CloseLibrary>;
using Lookup = void* (*)(void*, const char*);

Library open_library(const char* path) {
    Library library{dlopen(path, RTLD_NOW | RTLD_LOCAL)};
    if (!library) throw std::runtime_error("proxy fixture dlopen failed");
    return library;
}

bool is_interceptor(void* address) {
    Dl_info information{};
    return address != nullptr && dladdr(address, &information) != 0 &&
           information.dli_fname != nullptr &&
           std::string_view(information.dli_fname).find("libglimmer_cuda_interceptor.so") !=
               std::string_view::npos;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 6 || (std::string_view(argv[4]) != "0" && std::string_view(argv[4]) != "1") ||
        (std::string_view(argv[5]) != "0" && std::string_view(argv[5]) != "1")) {
        return EXIT_FAILURE;
    }
    try {
        auto first = open_library(argv[1]);
        auto second = open_library(argv[2]);
        auto caller = open_library(argv[3]);
        auto libc = open_library("libc.so.6");
        const auto native = reinterpret_cast<Lookup>(dlvsym(libc.get(), "dlsym", "GLIBC_2.2.5"));
        const auto runtime =
            reinterpret_cast<Lookup>(dlsym(caller.get(), "glimmer_proxy_runtime_lookup"));
        if (!native || !runtime) return EXIT_FAILURE;
        const std::array handles{first.get(), second.get()};
        const std::array expected{std::string_view(argv[4]) == "1",
                                  std::string_view(argv[5]) == "1"};
        std::atomic<bool> passed{true};
        const auto check = [&] {
            for (unsigned int repetition = 0; repetition < 100; ++repetition) {
                for (std::size_t index = 0; index < handles.size(); ++index) {
                    for (const auto lookup : std::array<Lookup, 2>{&dlsym, runtime}) {
                        for (const auto* name : {"cuLaunchKernel", "cuGetProcAddress_v2"}) {
                            auto* original = native(handles[index], name);
                            auto* actual = lookup(handles[index], name);
                            if (!original || !actual || is_interceptor(actual) != expected[index] ||
                                (!expected[index] && original != actual)) {
                                passed = false;
                            }
                        }
                        auto* original_runtime = native(handles[index], "cudaGetLastError");
                        if (!original_runtime ||
                            lookup(handles[index], "cudaGetLastError") != original_runtime) {
                            passed = false;
                        }
                        const auto value = reinterpret_cast<int (*)()>(
                            lookup(handles[index], "cuGlimmerProxyValue"));
                        if (!value || value() != 42) passed = false;
                        (void)dlerror();
                        auto* missing = lookup(handles[index], "cuGlimmerProxyMissing");
                        const auto* error = dlerror();
                        if (missing != nullptr || error == nullptr) passed = false;
                    }
                }
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(8);
        for (unsigned int index = 0; index < 8; ++index) workers.emplace_back(check);
        for (auto& worker : workers) worker.join();
        if (!passed) {
            std::cerr << "Driver proxy qualification, domain or native-error check failed\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
