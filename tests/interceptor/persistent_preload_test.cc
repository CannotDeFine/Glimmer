#include "persistent_ptx_fixture.h"

#include <cuda.h>
#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

extern "C" unsigned int glimmer_fake_persistent_worker_count();
extern "C" unsigned int glimmer_fake_persistent_logical_count();
extern "C" unsigned int glimmer_fake_library_calls();

namespace {

using LoadModule = CUresult (*)(CUmodule*, const void*);

bool library_entries() {
    for (int mode = 0; mode != 3; ++mode) {
        const auto resolve = [mode](const char* name, void* direct) {
            if (mode == 0) return direct;
            if (mode == 1) return dlsym(RTLD_DEFAULT, name);
            void* address{};
            CUdriverProcAddressQueryResult status{};
            if (cuGetProcAddress(name, &address, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &status) !=
                    CUDA_SUCCESS ||
                status != CU_GET_PROC_ADDRESS_SUCCESS)
                return static_cast<void*>(nullptr);
            return address;
        };
        auto load = reinterpret_cast<decltype(&cuLibraryLoadData)>(
            resolve("cuLibraryLoadData", reinterpret_cast<void*>(&cuLibraryLoadData)));
        auto file = reinterpret_cast<decltype(&cuLibraryLoadFromFile)>(
            resolve("cuLibraryLoadFromFile", reinterpret_cast<void*>(&cuLibraryLoadFromFile)));
        auto module = reinterpret_cast<decltype(&cuLibraryGetModule)>(
            resolve("cuLibraryGetModule", reinterpret_cast<void*>(&cuLibraryGetModule)));
        auto unload = reinterpret_cast<decltype(&cuLibraryUnload)>(
            resolve("cuLibraryUnload", reinterpret_cast<void*>(&cuLibraryUnload)));
        if (!load || !file || !module || !unload) return false;
        CUlibrary library{};
        CUmodule borrowed{};
        CUjit_option option = CU_JIT_MAX_REGISTERS;
        CUlibraryOption library_option = CU_LIBRARY_BINARY_IS_PRESERVED;
        void* value = &library;
        const auto before = glimmer_fake_library_calls();
        if (load(&library, "opaque", &option, &value, 1, &library_option, &value, 1) !=
                CUDA_ERROR_INVALID_IMAGE ||
            library ||
            file(&library, "opaque", &option, &value, 1, &library_option, &value, 1) !=
                CUDA_ERROR_INVALID_IMAGE ||
            library ||
            load(&library, "opaque", nullptr, nullptr, 0, nullptr, nullptr, 0) != CUDA_SUCCESS ||
            module(&borrowed, library) != CUDA_SUCCESS || !borrowed ||
            unload(library) != CUDA_SUCCESS ||
            file(&library, "opaque", nullptr, nullptr, 0, nullptr, nullptr, 0) != CUDA_SUCCESS ||
            unload(library) != CUDA_SUCCESS || unload(nullptr) != CUDA_ERROR_INVALID_HANDLE ||
            glimmer_fake_library_calls() != before + 8)
            return false;
    }
    return true;
}

bool run() {
    if (cuInit(0) != CUDA_SUCCESS) {
        return false;
    }
    if (!library_entries()) return false;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    unsigned int output = 0;
    unsigned int count = 999;
    void* arguments[] = {&output, &count};
    // No Glimmer API is used by this application fixture.
    for (const bool use_proc_address : {false, true}) {
        void* address = nullptr;
        if (use_proc_address) {
            if (cuGetProcAddress("cuModuleLoadData", &address, CUDA_VERSION,
                                 CU_GET_PROC_ADDRESS_DEFAULT, nullptr) != CUDA_SUCCESS) {
                return false;
            }
        } else {
            address = dlsym(RTLD_DEFAULT, "cuModuleLoadData");
        }
        if (address == nullptr ||
            reinterpret_cast<LoadModule>(address)(&module, kPersistentTestPtx) != CUDA_SUCCESS ||
            cuModuleGetFunction(&function, module, "write_indices") != CUDA_SUCCESS ||
            cuLaunchKernel(function, 17, 1, 1, 64, 1, 1, 0, nullptr, arguments, nullptr) !=
                CUDA_SUCCESS ||
            glimmer_fake_persistent_worker_count() != 3 ||
            glimmer_fake_persistent_logical_count() != 17 ||
            cuModuleUnload(module) != CUDA_SUCCESS) {
            return false;
        }
        const char binary[] = {'\x7f', 'E', 'L', 'F', '\0'};
        if (cuModuleLoadDataEx(&module, binary, 0, nullptr, nullptr) != CUDA_SUCCESS ||
            cuModuleGetFunction(&function, module, "write_indices") != CUDA_SUCCESS ||
            cuLaunchKernel(function, 17, 1, 1, 64, 1, 1, 0, nullptr, arguments, nullptr) !=
                CUDA_SUCCESS ||
            glimmer_fake_persistent_worker_count() != 17 ||
            cuModuleUnload(module) != CUDA_SUCCESS) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const bool passed = run();
    if (!passed) {
        std::cerr << "transparent persistent preload routing failed\n";
    }
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
