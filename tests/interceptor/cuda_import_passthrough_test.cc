#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cuda.h>

#include <dlfcn.h>

#include <cstdlib>
#include <iostream>

namespace {

using InitFunction = CUresult (*)(unsigned int flags);
using ImportHandleFunction = CUresult (*)(CUmemGenericAllocationHandle* handle, void* os_handle,
                                          CUmemAllocationHandleType handle_type);
using ReleaseFunction = CUresult (*)(CUmemGenericAllocationHandle handle);
using PoolImportHandleFunction = CUresult (*)(CUmemoryPool* pool_out, void* handle,
                                              CUmemAllocationHandleType handle_type,
                                              unsigned long long flags);
using PoolImportPointerFunction = CUresult (*)(CUdeviceptr* pointer_out, CUmemoryPool pool,
                                               CUmemPoolPtrExportData* share_data);

template <typename Function>
Function resolve(const char* name) {
    return reinterpret_cast<Function>(dlsym(RTLD_DEFAULT, name));
}

}  // namespace

int main() {
    unsetenv("GLIMMER_MEMORY_LIMIT_BYTES");
    unsetenv("GLIMMER_QUOTA_MODE");
    unsetenv("GLIMMER_QUOTA_TENANT_ID");

    const InitFunction init = resolve<InitFunction>("cuInit");
    const ImportHandleFunction import_handle =
        resolve<ImportHandleFunction>("cuMemImportFromShareableHandle");
    const ReleaseFunction release = resolve<ReleaseFunction>("cuMemRelease");
    const PoolImportHandleFunction import_pool =
        resolve<PoolImportHandleFunction>("cuMemPoolImportFromShareableHandle");
    const PoolImportPointerFunction import_pointer =
        resolve<PoolImportPointerFunction>("cuMemPoolImportPointer");
    if (init == nullptr || import_handle == nullptr || release == nullptr ||
        import_pool == nullptr || import_pointer == nullptr) {
        std::cerr << "CUDA import passthrough symbols were not exported\n";
        return EXIT_FAILURE;
    }
    if (init(0) != CUDA_SUCCESS) {
        std::cerr << "CUDA initialization failed\n";
        return EXIT_FAILURE;
    }

    CUmemGenericAllocationHandle imported_handle = 0;
    const bool imported = import_handle(&imported_handle, nullptr,
                                        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS &&
                          imported_handle != 0;
    const bool released = imported && release(imported_handle) == CUDA_SUCCESS;
    CUmemoryPool imported_pool = nullptr;
    const bool pool_imported =
        import_pool(&imported_pool, nullptr, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
            CUDA_SUCCESS &&
        imported_pool != nullptr;
    CUmemPoolPtrExportData share_data{};
    CUdeviceptr imported_pointer = 0;
    const bool pointer_imported =
        import_pointer(&imported_pointer, imported_pool, &share_data) == CUDA_SUCCESS &&
        imported_pointer != 0;
    if (!imported || !released || !pool_imported || !pointer_imported) {
        std::cerr << "CUDA imports were not delegated without a quota\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
