#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using DlsymFunction = void* (*)(void* handle, const char* name);

bool is_from_interceptor(void* symbol) {
    if (symbol == nullptr) {
        return false;
    }

    Dl_info info{};
    if (dladdr(symbol, &info) == 0 || info.dli_fname == nullptr) {
        return false;
    }

    return std::string_view(info.dli_fname).find("libglimmer_cuda_interceptor.so") !=
           std::string_view::npos;
}

bool is_not_from_interceptor(void* symbol) {
    return symbol != nullptr && !is_from_interceptor(symbol);
}

}  // namespace

int main() {
    const char* const intercepted_names[] = {
        "cuInit",
        "cuMemAlloc",
        "cuMemAlloc_v2",
        "cuMemAllocManaged",
        "cuMemAllocPitch",
        "cuMemAllocPitch_v2",
        "cuMemFree",
        "cuMemFree_v2",
        "cuMemGetInfo",
        "cuMemGetInfo_v2",
        "cuDeviceTotalMem",
        "cuDeviceTotalMem_v2",
        "cuCtxGetCurrent",
        "cuCtxGetDevice",
        "cuCtxDestroy",
        "cuCtxDestroy_v2",
        "cuMemAllocAsync",
        "cuMemAllocAsync_ptsz",
        "cuMemAllocFromPoolAsync",
        "cuMemAllocFromPoolAsync_ptsz",
        "cuMemFreeAsync",
        "cuMemFreeAsync_ptsz",
        "cuStreamGetDevice",
        "cuStreamGetDevice_ptsz",
        "cuStreamGetCtx",
        "cuStreamGetCtx_ptsz",
        "cuStreamQuery",
        "cuStreamQuery_ptsz",
        "cuStreamSynchronize",
        "cuStreamSynchronize_ptsz",
        "cuStreamDestroy",
        "cuStreamDestroy_v2",
        "cuCtxSynchronize",
        "cuGetProcAddress",
        "cuGetProcAddress_v2",
    };

    for (const char* name : intercepted_names) {
        if (!is_from_interceptor(dlsym(RTLD_DEFAULT, name))) {
            std::cerr << "dlsym did not return the interceptor for " << name << '\n';
            return EXIT_FAILURE;
        }
    }

    const char* const runtime_names[] = {"cudaMalloc",
                                         "cudaMallocAsync",
                                         "cudaMallocAsync_ptsz",
                                         "cudaFree",
                                         "cudaFreeAsync",
                                         "cudaFreeAsync_ptsz",
                                         "cudaMemGetInfo",
                                         "cudaDeviceSynchronize",
                                         "cudaStreamSynchronize",
                                         "cudaStreamSynchronize_ptsz",
                                         "cudaStreamQuery",
                                         "cudaStreamQuery_ptsz",
                                         "cudaStreamDestroy"};
    for (const char* name : runtime_names) {
        if (!is_from_interceptor(dlsym(RTLD_DEFAULT, name))) {
            std::cerr << "dlsym did not return the Runtime interceptor for " << name << '\n';
            return EXIT_FAILURE;
        }
    }

    if (!is_not_from_interceptor(dlsym(RTLD_DEFAULT, "dlopen"))) {
        std::cerr << "dlsym did not delegate an unsupported symbol\n";
        return EXIT_FAILURE;
    }

    const DlsymFunction dlsym_without_nonnull_attribute = reinterpret_cast<DlsymFunction>(&dlsym);
    if (dlsym_without_nonnull_attribute(RTLD_DEFAULT, nullptr) != nullptr) {
        std::cerr << "dlsym did not reject a null symbol name\n";
        return EXIT_FAILURE;
    }

    void* cuda_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (cuda_handle != nullptr) {
        const bool explicit_handle_is_intercepted =
            is_from_interceptor(dlsym(cuda_handle, "cuMemAlloc_v2"));
        const bool explicit_handle_keeps_runtime_scope =
            dlsym(cuda_handle, "cudaMalloc") == nullptr;
        dlclose(cuda_handle);
        if (!explicit_handle_is_intercepted || !explicit_handle_keeps_runtime_scope) {
            std::cerr << "dlsym did not route an explicit CUDA handle\n";
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
