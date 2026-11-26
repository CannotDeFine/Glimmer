#include "persistent_ptx_fixture.h"

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 3 || std::string_view(argv[1]) != "--dump-ptx" ||
        std::getenv("LD_PRELOAD") != nullptr ||
        std::getenv("GLIMMER_PERSISTENT_RUNTIME") != nullptr) {
        return EXIT_FAILURE;
    }
    const std::string_view mode(argv[2]);
    if (mode == "/failure") {
        return EXIT_FAILURE;
    }
    if (mode == "/timeout") {
        for (;;) {
            (void)::pause();
        }
    }
    if (mode == "/overflow") {
        for (int index = 0; index < 17000; ++index) {
            std::cout << std::string(1024, 'x');
        }
        return EXIT_SUCCESS;
    }
    if (mode == "/no-ptx") {
        std::cout << "Fatbin elf code:\n================\narch = sm_70\n";
        return EXIT_SUCCESS;
    }
    std::cout << "Fatbin ptx code:\n================\ncompressed\n" << kPersistentTestPtx;
    if (mode == "/duplicate") {
        std::cout << "\nFatbin ptx code:\n================\n" << kPersistentTestPtx;
    }
    std::cout << "\nFatbin elf code:\n================\narch = sm_70\n";
    return EXIT_SUCCESS;
}
