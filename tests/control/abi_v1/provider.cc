#include "layout.h"

#include <cstdlib>

extern "C" bool glimmer_abi_v1_consumer(const std::size_t* layout, std::size_t count);

int main() {
    const auto layout = control_layout();
    return glimmer_abi_v1_consumer(layout.data(), layout.size()) ? EXIT_SUCCESS : EXIT_FAILURE;
}
