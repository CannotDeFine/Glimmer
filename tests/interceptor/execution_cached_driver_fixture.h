#pragma once

#include <cuda.h>
#include <cudaTypedefs.h>

#include <stdexcept>
#include <string>
#include <dlfcn.h>

// An ordinary public Driver cache, not a framework interface or private ABI.
struct ExecutionCachedDriver {
    decltype(&cuInit) initialize{&cuInit};
    decltype(&cuDeviceGet) get_device{&cuDeviceGet};
    decltype(&cuDeviceGetAttribute) device_attribute{&cuDeviceGetAttribute};
    decltype(&cuCtxCreate_v4) create_context_v4{&cuCtxCreate_v4};
    decltype(&cuModuleLoadData) load_module{&cuModuleLoadData};
    decltype(&cuModuleGetFunction) get_function{&cuModuleGetFunction};
    decltype(&cuMemAlloc) allocate{&cuMemAlloc};
    decltype(&cuMemAlloc) explicit_allocate{&cuMemAlloc};
    decltype(&cuMemcpyHtoD) upload{&cuMemcpyHtoD};
    decltype(&cuMemcpyDtoH) download{&cuMemcpyDtoH};
    decltype(&cuStreamCreate) create_stream{&cuStreamCreate};
    decltype(&cuLaunchKernel) launch{&cuLaunchKernel};
    decltype(&cuStreamSynchronize) synchronize_stream{&cuStreamSynchronize};
    decltype(&cuCtxSynchronize) synchronize_context{&cuCtxSynchronize};
    decltype(&cuMemFree) free_memory{&cuMemFree};
    decltype(&cuStreamDestroy) destroy_stream{&cuStreamDestroy};
    decltype(&cuModuleUnload) unload_module{&cuModuleUnload};
    decltype(&cuCtxDestroy) destroy_context{&cuCtxDestroy};
    decltype(&cuDevicePrimaryCtxRetain) retain_primary{&cuDevicePrimaryCtxRetain};
    decltype(&cuDevicePrimaryCtxRelease) release_primary{&cuDevicePrimaryCtxRelease};
    decltype(&cuDevicePrimaryCtxReset) reset_primary{&cuDevicePrimaryCtxReset};
    decltype(&cuDevicePrimaryCtxGetState) primary_state{&cuDevicePrimaryCtxGetState};
    decltype(&cuDevicePrimaryCtxSetFlags) primary_flags{&cuDevicePrimaryCtxSetFlags};
    decltype(&cuCtxGetCurrent) get_current{&cuCtxGetCurrent};
    decltype(&cuCtxSetCurrent) set_current{&cuCtxSetCurrent};
    decltype(&cuCtxGetFlags) get_context_flags{&cuCtxGetFlags};
    decltype(&cuCtxSetFlags) set_context_flags{&cuCtxSetFlags};
    decltype(&cuCtxGetDevice) context_device{&cuCtxGetDevice};
    bool is_cached{};

    void bind(bool should_cache, bool should_primary = false) {
        if (!should_cache) return;
        struct Library {
            void* value{::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL)};
            ~Library() {
                if (value && ::dlclose(value) != 0) std::terminate();
            }
        } library;
        if (!library.value) throw std::runtime_error("load installed Driver scope");
        auto modern = reinterpret_cast<PFN_cuGetProcAddress_v12000>(
            ::dlsym(library.value, "cuGetProcAddress_v2"));
        if (!modern) throw std::runtime_error("lookup typed public resolver");
        void* pointer{};
        CUdriverProcAddressQueryResult status{};
        if (modern("cuGetProcAddress", &pointer, 11030, 0, &status) != CUDA_SUCCESS || !pointer ||
            status != CU_GET_PROC_ADDRESS_SUCCESS)
            throw std::runtime_error("cache legacy resolver through modern ABI");
        const auto legacy = reinterpret_cast<PFN_cuGetProcAddress_v11030>(pointer);
        pointer = nullptr;
        if (legacy("cuGetProcAddress", &pointer, 12000, 0) != CUDA_SUCCESS || !pointer)
            throw std::runtime_error("cache modern resolver through legacy ABI");
        modern = reinterpret_cast<PFN_cuGetProcAddress_v12000>(pointer);
        const auto resolve = [&](const char* name, int version) {
            void* result{};
            CUdriverProcAddressQueryResult observed{};
            if (modern(name, &result, version, 0, &observed) != CUDA_SUCCESS || !result ||
                observed != CU_GET_PROC_ADDRESS_SUCCESS)
                throw std::runtime_error(std::string("cache public Driver entry: ") + name);
            return result;
        };
#define CACHED(FIELD, NAME, VERSION) \
    (FIELD) = reinterpret_cast<decltype(FIELD)>(resolve((NAME), (VERSION)))
        CACHED(initialize, "cuInit", 11030);
        CACHED(get_device, "cuDeviceGet", 11030);
        CACHED(device_attribute, "cuDeviceGetAttribute", 11030);
        CACHED(create_context_v4, "cuCtxCreate", 12050);
        CACHED(load_module, "cuModuleLoadData", 11030);
        CACHED(get_function, "cuModuleGetFunction", 11030);
        CACHED(allocate, "cuMemAlloc", 11030);
        CACHED(upload, "cuMemcpyHtoD", 11030);
        CACHED(download, "cuMemcpyDtoH", 11030);
        CACHED(create_stream, "cuStreamCreate", 11030);
        CACHED(launch, "cuLaunchKernel", 11030);
        CACHED(synchronize_stream, "cuStreamSynchronize", 11030);
        CACHED(synchronize_context, "cuCtxSynchronize", 11030);
        CACHED(free_memory, "cuMemFree", 11030);
        CACHED(destroy_stream, "cuStreamDestroy", 11030);
        CACHED(unload_module, "cuModuleUnload", 11030);
        CACHED(destroy_context, "cuCtxDestroy", 11030);
        if (should_primary) {
            CACHED(retain_primary, "cuDevicePrimaryCtxRetain", 11030);
            CACHED(release_primary, "cuDevicePrimaryCtxRelease", 11030);
            CACHED(reset_primary, "cuDevicePrimaryCtxReset", 11030);
            CACHED(primary_state, "cuDevicePrimaryCtxGetState", 11030);
            CACHED(primary_flags, "cuDevicePrimaryCtxSetFlags", 11030);
            CACHED(get_current, "cuCtxGetCurrent", 11030);
            CACHED(set_current, "cuCtxSetCurrent", 11030);
            CACHED(get_context_flags, "cuCtxGetFlags", 11030);
            CACHED(set_context_flags, "cuCtxSetFlags", 12010);
            CACHED(context_device, "cuCtxGetDevice", 11030);
        }
#undef CACHED
        explicit_allocate =
            reinterpret_cast<decltype(explicit_allocate)>(::dlsym(library.value, "cuMemAlloc_v2"));
        if (!explicit_allocate) throw std::runtime_error("cache explicit Driver allocation entry");
        is_cached = true;
    }
    CUresult create_context(CUcontext* output, unsigned int flags, CUdevice device) const {
        if (!is_cached) return cuCtxCreate(output, flags, device);
        CUexecAffinityParam ignored{};
        CUctxCreateParams parameters{&ignored, 0, nullptr};
        return create_context_v4(output, &parameters, flags, device);
    }
};
