#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <dlfcn.h>

// SDK-only application cache, independent of Glimmer and server configuration.
struct ExecutionCachedRuntime {
    using Allocate = cudaError_t(CUDARTAPI*)(void**, std::size_t);
    using Launch = cudaError_t(CUDARTAPI*)(const void*, dim3, dim3, void**, std::size_t,
                                           cudaStream_t);
    using DefaultQuery = cudaError_t(CUDARTAPI*)(const char*, void**, unsigned long long,
                                                 cudaDriverEntryPointQueryResult*);
    using VersionQuery = cudaError_t(CUDARTAPI*)(const char*, void**, unsigned int,
                                                 unsigned long long,
                                                 cudaDriverEntryPointQueryResult*);
    Allocate allocate{static_cast<Allocate>(&cudaMalloc)};
    decltype(&cudaFree) release{&cudaFree};
    decltype(&cudaSetDevice) select{&cudaSetDevice};
    decltype(&cudaMemcpy) copy{&cudaMemcpy};
    decltype(&cudaStreamCreateWithFlags) create_stream{&cudaStreamCreateWithFlags};
    decltype(&cudaStreamSynchronize) synchronize_stream{&cudaStreamSynchronize};
    decltype(&cudaStreamDestroy) destroy_stream{&cudaStreamDestroy};
    decltype(&cudaDeviceSynchronize) synchronize_device{&cudaDeviceSynchronize};
    decltype(&cudaDeviceReset) reset{&cudaDeviceReset};
    decltype(&cudaGetLastError) last_error{&cudaGetLastError};
    Launch launch{static_cast<Launch>(&cudaLaunchKernel)};
    decltype(&cuMemAlloc) driver_allocate{};
    decltype(&cuCtxGetCurrent) current{};
    void* library{};
    ExecutionCachedRuntime() = default;
    ExecutionCachedRuntime(const ExecutionCachedRuntime&) = delete;
    ExecutionCachedRuntime& operator=(const ExecutionCachedRuntime&) = delete;
    ~ExecutionCachedRuntime() {
        (void)close();
    }  // Failure-path fallback only.
    bool close() {
        return !library || ::dlclose(std::exchange(library, nullptr)) == 0;
    }
    template <class Function>
    Function lookup(const char* name) {
        auto* value = ::dlsym(library, name);
        if (!value) throw std::runtime_error(std::string("missing SDK cache entry: ") + name);
        return reinterpret_cast<Function>(value);
    }
    void enable() {
        library = ::dlopen("libcudart.so.12", RTLD_NOW | RTLD_NOLOAD);
        if (!library) throw std::runtime_error("shared Runtime is not loaded");
        allocate = lookup<Allocate>("cudaMalloc");
        release = lookup<decltype(release)>("cudaFree");
        select = lookup<decltype(select)>("cudaSetDevice");
        copy = lookup<decltype(copy)>("cudaMemcpy");
        create_stream = lookup<decltype(create_stream)>("cudaStreamCreateWithFlags");
        synchronize_stream = lookup<decltype(synchronize_stream)>("cudaStreamSynchronize");
        destroy_stream = lookup<decltype(destroy_stream)>("cudaStreamDestroy");
        synchronize_device = lookup<decltype(synchronize_device)>("cudaDeviceSynchronize");
        reset = lookup<decltype(reset)>("cudaDeviceReset");
        last_error = lookup<decltype(last_error)>("cudaGetLastError");
        launch = lookup<Launch>("cudaLaunchKernel");
        const auto default_query = lookup<DefaultQuery>("cudaGetDriverEntryPoint");
        const auto version_query = lookup<VersionQuery>("cudaGetDriverEntryPointByVersion");
        cudaDriverEntryPointQueryResult status{};
        void* entry{};
        const auto default_result =
            default_query("cuCtxGetCurrent", &entry, cudaEnableLegacyStream, &status);
        if (default_result != cudaSuccess || !entry || status != cudaDriverEntryPointSuccess)
            throw std::runtime_error(
                "default Runtime Driver query: result=" + std::to_string(default_result) +
                "; status=" + std::to_string(status) +
                "; pointer_present=" + std::to_string(entry != nullptr));
        current = reinterpret_cast<decltype(current)>(entry);
        if (version_query("cuMemAlloc", &entry, 3020, cudaEnableLegacyStream, &status) !=
                cudaSuccess ||
            !entry || status != cudaDriverEntryPointSuccess)
            throw std::runtime_error("versioned Runtime Driver query");
        driver_allocate = reinterpret_cast<decltype(driver_allocate)>(entry);
    }
};
