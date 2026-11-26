#include <cuda.h>
#include <curand.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <dlfcn.h>

namespace {
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
Function select(Function original, void* library, const char* name, bool cached) {
    if (!cached) return original;
    const auto pointer = ::dlsym(library, name);
    expect(pointer != nullptr, "resolve original/owned cuRAND lifecycle entry");
    return reinterpret_cast<Function>(pointer);
}
void run(bool cached) {
    expect(cuInit(0) == CUDA_SUCCESS, "initialize Driver");
    CUdevice device{};
    expect(cuDeviceGet(&device, 0) == CUDA_SUCCESS, "select device");
    CUcontext context{};
    expect(cuCtxCreate(&context, 0, device) == CUDA_SUCCESS, "create application context");
    void* library = cached ? ::dlopen("libcurand.so.10", RTLD_NOW | RTLD_LOCAL) : nullptr;
    expect(!cached || library, "open installed cuRAND provider");
    const auto create = select(&curandCreateGenerator, library, "curandCreateGenerator", cached);
    const auto seed = select(&curandSetPseudoRandomGeneratorSeed, library,
                             "curandSetPseudoRandomGeneratorSeed", cached);
    const auto destroy = select(&curandDestroyGenerator, library, "curandDestroyGenerator", cached);
    for (const auto type :
         {CURAND_RNG_PSEUDO_XORWOW, CURAND_RNG_PSEUDO_PHILOX4_32_10, CURAND_RNG_QUASI_SOBOL32}) {
        curandGenerator_t generator{};
        expect(create(&generator, type) == CURAND_STATUS_SUCCESS && generator,
               "original SDK generator creation");
        for (const auto value : std::array<unsigned long long, 3>{0, 17, UINT64_MAX - 17})
            expect(seed(generator, value) == (type == CURAND_RNG_QUASI_SOBOL32
                                                  ? CURAND_STATUS_TYPE_ERROR
                                                  : CURAND_STATUS_SUCCESS),
                   "unchanged generator-type and full-width seed statuses");
        expect(destroy(generator) == CURAND_STATUS_SUCCESS, "destroy original generator once");
    }
    expect(!cached || ::dlclose(library) == 0, "balance original provider reference");
    expect(cuCtxDestroy(context) == CUDA_SUCCESS, "destroy application context");
    std::cout << "cuRAND lifecycle verified: generators=3; seed_calls=9; cached=" << cached
              << "; numerical_qualified=false; full_device_qualified=false; "
                 "priority_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--cached"), "arguments");
        run(argc == 2);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
