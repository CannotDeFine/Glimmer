#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        return EXIT_FAILURE;
    }
    // Load the fixture and its interceptor dependency only after sanitizer
    // startup. Keep process-owned interceptor state mapped through exit.
    void* fixture = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
    if (!fixture) {
        std::cerr << "cannot load private accounting fixture: " << dlerror() << '\n';
        return EXIT_FAILURE;
    }
    const auto run = reinterpret_cast<int (*)(const char*)>(
        dlsym(fixture, "glimmer_run_progress_accounting_fixture"));
    if (!run) {
        std::cerr << "private accounting fixture entry unavailable\n";
        return EXIT_FAILURE;
    }
    return run(argv[2]);
}
