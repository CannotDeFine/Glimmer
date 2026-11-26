#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
using Lookup = void* (*)(void*, const char*);
using Marker = int (*)();

void require(bool value, const char* reason) {
    if (!value) {
        throw std::runtime_error(reason);
    }
}

class Library final {
   public:
    explicit Library(const char* path, int flags = RTLD_NOW | RTLD_NOLOAD)
        : handle_(dlopen(path, flags)) {
        require(handle_ != nullptr, "fixture could not be loaded");
    }
    ~Library() {
        if (dlclose(handle_) != 0) {
            std::terminate();
        }
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    void* get() const {
        return handle_;
    }

   private:
    void* handle_;
};

void check(Lookup first, Lookup second, void* middle, void* last) {
    for (const char* name :
         {"glimmer_lookup_marker", "cuCallerLookupMarker", "nvml_caller_lookup_marker"}) {
        const auto next_first = reinterpret_cast<Marker>(first(RTLD_NEXT, name));
        const auto next_second = reinterpret_cast<Marker>(second(RTLD_NEXT, name));
        require(next_first && next_first() == 2, "first caller must find the middle provider");
        require(next_second && next_second() == 3, "second caller must skip its own provider");
        const auto initial = reinterpret_cast<Marker>(first(RTLD_DEFAULT, name));
        const auto explicit_last = reinterpret_cast<Marker>(first(last, name));
        require(initial && initial() == 1 && explicit_last && explicit_last() == 3,
                "default or explicit handle lookup changed");
    }
    auto* first_next_tls = static_cast<int*>(first(RTLD_NEXT, "glimmer_lookup_tls"));
    auto* second_next_tls = static_cast<int*>(second(RTLD_NEXT, "glimmer_lookup_tls"));
    require(first_next_tls && second_next_tls && first_next_tls != second_next_tls &&
                first_next_tls == dlsym(middle, "glimmer_lookup_tls") &&
                second_next_tls == dlsym(last, "glimmer_lookup_tls"),
            "next lookup must preserve provider and caller-thread TLS identity");
    ++*first_next_tls;
    *second_next_tls += 2;
    require(*second_next_tls == 2 * *first_next_tls, "thread-local providers alias or lose state");
    static_cast<void>(dlerror());
    require(first(RTLD_NEXT, "glimmer_missing_lookup_symbol") == nullptr && dlerror() != nullptr,
            "missing next symbol must preserve native dlerror");
    static_cast<void>(dlerror());
    require(first(last, "glimmer_lookup_absolute_zero") == nullptr && dlerror() == nullptr,
            "successful null-valued symbol must remain distinct from lookup failure");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 4, "expected middle, last and local fixture paths");
        const Library middle(argv[1]);
        const Library last(argv[2]);
        const Library local(argv[3], RTLD_NOW | RTLD_LOCAL);
        const auto lookup = reinterpret_cast<Lookup>(dlsym(RTLD_DEFAULT, "dlsym"));
        require(lookup != nullptr, "cached loader entry missing");
        const auto first = reinterpret_cast<Lookup>(lookup(RTLD_DEFAULT, "glimmer_lookup_first"));
        const auto second = reinterpret_cast<Lookup>(lookup(RTLD_DEFAULT, "glimmer_lookup_second"));
        require(first && second, "preload callers missing");
        const auto local_lookup =
            reinterpret_cast<Lookup>(lookup(local.get(), "glimmer_lookup_local"));
        require(local_lookup != nullptr, "local-scope caller missing");
        for (const char* name : {"glimmer_local_lookup_marker", "cuLocalLookupMarker"}) {
            const auto selected = reinterpret_cast<Marker>(local_lookup(RTLD_DEFAULT, name));
            require(selected && selected() == 4, "default lookup lost the caller's local scope");
            static_cast<void>(dlerror());
            require(lookup(RTLD_DEFAULT, name) == nullptr && dlerror() != nullptr,
                    "local fixture was unexpectedly promoted to global scope");
        }
        const auto from_main = reinterpret_cast<Marker>(lookup(RTLD_NEXT, "glimmer_lookup_marker"));
        require(from_main && from_main() == 1,
                "main next lookup must retain the application's origin");
        check(first, second, middle.get(), last.get());
        std::atomic<bool> passed{true};
        std::vector<std::jthread> threads;
        threads.reserve(8);
        for (int index = 0; index < 8; ++index) {
            threads.emplace_back([&] {
                try {
                    for (int iteration = 0; iteration < 100; ++iteration) {
                        check(first, second, middle.get(), last.get());
                    }
                } catch (...) {
                    passed = false;
                }
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        require(passed, "concurrent caller-relative lookup failed");
        std::cout << "dlsym_chain status=ok callers=4 threads=8\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
