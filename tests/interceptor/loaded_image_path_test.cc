#include "internal/loaded_image_path.h"

#include <dlfcn.h>
#include <link.h>

#include <atomic>
#include <barrier>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using glimmer::interceptor::loaded_image_path;

void require(bool passed, std::string_view message) {
    if (!passed) {
        std::cerr << "Test failed: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int main_image_data = 17;

class LoadedFixture {
   public:
    explicit LoadedFixture(const char* path, bool is_new_namespace = false)
        : handle_(is_new_namespace ? ::dlmopen(LM_ID_NEWLM, path, RTLD_NOW | RTLD_LOCAL)
                                   : ::dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        require(handle_ != nullptr, "fixture must load");
        address_ = ::dlsym(handle_, "glimmer_loaded_image_fixture");
        require(address_ != nullptr, "fixture function must resolve");
    }
    ~LoadedFixture() {
        close();
    }
    LoadedFixture(const LoadedFixture&) = delete;
    LoadedFixture& operator=(const LoadedFixture&) = delete;

    const void* address() const {
        return address_;
    }

    void close() {
        if (handle_ != nullptr) {
            require(::dlclose(handle_) == 0, "owned fixture must close exactly once");
            handle_ = nullptr;
        }
    }

   private:
    void* handle_{};
    void* address_{};
};

void expect_path(const void* address, std::string_view expected) {
    errno = EDOM;
    const auto actual = loaded_image_path(address);
    require(errno == EDOM, "successful lookup must preserve caller errno");
    require(actual && *actual == expected, "lookup must return the exact image pathname");
}

void test_main_and_unknown(const std::string& main_path) {
    expect_path(reinterpret_cast<const void*>(&test_main_and_unknown), main_path);
    expect_path(&main_image_data, main_path);
    expect_path(
        // Synthetic metadata-only address; never dereferenced.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(&test_main_and_unknown) + 1),
        main_path);
    int stack_data = 9;
    const auto heap_data = std::make_unique<int>(9);
    for (const void* address :
         {static_cast<const void*>(nullptr), reinterpret_cast<const void*>(1),
          // Exercise the overflow boundary, not a hot-path pointer conversion.
          // NOLINTNEXTLINE(performance-no-int-to-ptr)
          reinterpret_cast<const void*>(std::numeric_limits<std::uintptr_t>::max()),
          static_cast<const void*>(&stack_data), static_cast<const void*>(heap_data.get())}) {
        errno = ERANGE;
        require(!loaded_image_path(address), "unknown addresses must not invent ELF ownership");
        require(errno == ERANGE, "failed lookup must preserve caller errno");
    }
}

void test_fixture_lifetime(const char* path) {
    for (unsigned int index = 0; index < 8; ++index) {
        LoadedFixture fixture(path);
        const auto address = fixture.address();
        expect_path(address, path);
        // Synthetic metadata-only address; never dereferenced.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        expect_path(reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(address) + 1),
                    path);
        const auto owned_path = loaded_image_path(address);
        fixture.close();
        errno = EINVAL;
        require(!loaded_image_path(address), "unloaded images must not leave stale cached owners");
        require(errno == EINVAL && owned_path && *owned_path == path,
                "returned paths must remain owned after unload");
    }
#ifndef GLIMMER_SANITIZED_IMAGE_TEST
    // dl_iterate_phdr in the caller namespace cannot prove this image. The
    // original dladdr fallback must retain the existing namespace behavior.
    LoadedFixture other_namespace(path, true);
    expect_path(other_namespace.address(), path);
#else
    // A second namespace cannot host this fixture's independent ASan runtime.
    // Ordinary Debug/lint/optimized runs cover this original dladdr fallback.
    std::cout << "Separate loader namespace skipped for the instrumented fixture\n";
#endif
}

void test_concurrent(const char* path, const std::string& main_path) {
    LoadedFixture fixture(path);
    std::barrier start(5);
    std::atomic<bool> has_failed{};
    std::vector<std::thread> callers;
    for (unsigned int index = 0; index < 4; ++index) {
        callers.emplace_back([&] {
            start.arrive_and_wait();
            for (unsigned int count = 0; count < 1000; ++count) {
                errno = EBUSY;
                const auto shared = loaded_image_path(fixture.address());
                if (errno != EBUSY || !shared || *shared != path) {
                    has_failed = true;
                }
                errno = ENOTTY;
                const auto main = loaded_image_path(&main_image_data);
                if (errno != ENOTTY || !main || *main != main_path) {
                    has_failed = true;
                }
            }
        });
    }
    start.arrive_and_wait();
    for (auto& caller : callers) {
        caller.join();
    }
    require(!has_failed.load(), "concurrent lookup must preserve owned paths and thread errno");
}

}  // namespace

int main(int argc, char** argv) {
    require(argc == 2, "fixture path must be supplied");
    const auto main_path = std::filesystem::read_symlink("/proc/self/exe").string();
    test_main_and_unknown(main_path);
    test_fixture_lifetime(argv[1]);
    test_concurrent(argv[1], main_path);
    return EXIT_SUCCESS;
}
