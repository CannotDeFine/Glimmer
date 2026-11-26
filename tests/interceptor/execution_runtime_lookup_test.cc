#include "internal/process_identity.h"
#include "runtime_lookup.h"

#include <cuda.h>

#include <array>
#include <barrier>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_execution_runtime_lookup_count(unsigned int);
extern "C" unsigned int glimmer_execution_lookup_count(unsigned int);
extern "C" void* glimmer_execution_runtime_internal_lookup(void*, const char*);
namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool owned(void* address) {
    Dl_info info{};
    return address && ::dladdr(address, &info) && info.dli_fname &&
           std::string_view(info.dli_fname).ends_with("libglimmer_execution_interceptor.so");
}
void typed() {
    void* address{};
    cudaDriverEntryPointQueryResult status{};
    expect(cudaGetDriverEntryPoint("cuCtxGetCurrent", &address, 0, &status) == cudaSuccess &&
               owned(address) && status == cudaDriverEntryPointSuccess,
           "Runtime query selects owned exact Driver ABI");
    CUcontext context = reinterpret_cast<CUcontext>(1);
    expect(
        reinterpret_cast<decltype(&cuCtxGetCurrent)>(address)(&context) == CUDA_SUCCESS && !context,
        "Runtime-cached Driver pointer reads virtual state");
    expect(
        glimmer_execution_lookup_count(3) >= 1 && glimmer_execution_lookup_count(3) <= 8,
        "bounded concurrent read-only initialization observations never publish the decoy context");
    expect(
        cudaGetDriverEntryPointByVersion("cuMemAlloc", &address, 3020, 0, &status) == cudaSuccess &&
            owned(address),
        "Runtime by-version query chooses the modern allocation ABI");
    expect(reinterpret_cast<decltype(&cuMemAlloc)>(address)(reinterpret_cast<CUdeviceptr*>(1), 4) ==
               CUDA_ERROR_INVALID_VALUE,
           "Runtime-cached Driver resource call remains guarded");
    expect(cudaGetDriverEntryPointByVersion("cuMemAlloc", &address, 2000, 0, &status) ==
                   cudaErrorNotSupported &&
               !address,
           "unsupported historical allocation ABI cannot escape");
    expect(cudaGetDriverEntryPoint_ptsz("cuLaunchKernel", &address, 0, &status) ==
                   cudaErrorNotSupported &&
               !address,
           "PTDS query never substitutes legacy launch semantics");
    expect(cudaGetDriverEntryPointByVersion_ptsz("cuCtxGetCurrent", &address, 12000,
                                                 cudaEnableLegacyStream, &status) == cudaSuccess &&
               owned(address),
           "explicit PTDS query retains an exact native legacy selection");
}
void failures() {
    void* address = reinterpret_cast<void*>(1);
    cudaDriverEntryPointQueryResult status{};
    int raw = 0x71717171;
    std::memcpy(&status, &raw, sizeof(status));
    expect(cudaGetDriverEntryPoint("runtime_sdk_error", &address, 0, &status) ==
                   cudaErrorSystemDriverMismatch &&
               !address,
           "original Driver metadata error maps to Runtime without native entry escape");
    std::memcpy(&raw, &status, sizeof(raw));
    expect(raw == 0x71717171 && cudaGetLastError() == cudaErrorSystemDriverMismatch,
           "untouched arbitrary SDK output bytes are preserved without enum UB");
    expect(cudaGetDriverEntryPoint("cuCtxGetCurrent", &address, 0, &status) == cudaSuccess &&
               owned(address) && status == cudaDriverEntryPointSuccess,
           "uninitialized status output is not an enum input");
    expect(cudaGetDriverEntryPoint("missing", &address, 0, &status) == cudaSuccess && !address &&
               status == cudaDriverEntryPointSymbolNotFound,
           "original successful missing result stays missing");
    expect(cudaGetDriverEntryPoint("old_version", &address, 0, &status) == cudaSuccess &&
               !address && status == cudaDriverEntryPointVersionNotSufficent,
           "SDK version result remains authoritative");
    expect(cudaGetDriverEntryPoint("runtime_malformed", &address, 0, &status) == cudaErrorUnknown &&
               !address,
           "contradictory SDK success never publishes a pointer");
    expect(cudaGetDriverEntryPoint("cuLookupUnsupported", &address, 0, nullptr) ==
                   cudaErrorNotSupported &&
               !address,
           "unknown SDK ABI cannot return a native resource pointer");
    const auto before = glimmer_execution_runtime_lookup_count(0);
    expect(cudaGetDriverEntryPoint(reinterpret_cast<const char*>(1), &address, 0, nullptr) ==
                   cudaErrorInvalidValue &&
               cudaGetDriverEntryPoint("cuMemAlloc", reinterpret_cast<void**>(1), 0, nullptr) ==
                   cudaErrorInvalidValue &&
               cudaGetDriverEntryPoint("cuMemAlloc", &address, 0,
                                       reinterpret_cast<cudaDriverEntryPointQueryResult*>(1)) ==
                   cudaErrorInvalidValue &&
               glimmer_execution_runtime_lookup_count(0) == before,
           "bad caller storage rejects before native metadata");
    cudaUUID_t id{};
    const void* table = reinterpret_cast<const void*>(1);
    expect(cudaGetExportTable(&table, &id) == cudaErrorNotSupported && !table &&
               glimmer_execution_runtime_lookup_count(5) == 0,
           "private export table cannot escape virtual ownership");
}
void scopes(void* runtime, const char* other_path) {
    const auto malloc = reinterpret_cast<decltype(&cudaMalloc)>(::dlsym(runtime, "cudaMalloc"));
    expect(owned(reinterpret_cast<void*>(malloc)),
           "explicit Runtime cache must select owned allocation");
    expect(malloc(reinterpret_cast<void**>(1), 4) == cudaErrorInvalidValue,
           "explicit cached Runtime resource call reaches controlled validation");
    for (const auto* name :
         {"cudaFree", "cudaGetDriverEntryPoint", "cudaGetDriverEntryPointByVersion"})
        expect(owned(::dlsym(runtime, name)), "covered Runtime SDK-handle cache stays owned");
    for (const auto* name :
         {"cudaMalloc", "cudaLaunchKernel", "__cudaRegisterFatBinary", "__cudaUnregisterFatBinary"})
        expect(owned(::dlsym(RTLD_DEFAULT, name)), "global Runtime/compiler lookup stays owned");
    expect(!owned(glimmer_execution_runtime_internal_lookup(runtime, "cudaMalloc")),
           "native Runtime internal SDK scope remains native");
    ::dlerror();
    expect(!::dlsym(runtime, "cudaLookupUnsupported") && ::dlerror(),
           "unknown Runtime SDK lookup gives real loader failure");
    void* other = ::dlopen(other_path, RTLD_NOW | RTLD_LOCAL);
    expect(other, "open unrelated same-name Runtime scope");
    const auto separate = reinterpret_cast<decltype(&cudaMalloc)>(::dlsym(other, "cudaMalloc"));
    expect(separate && !owned(reinterpret_cast<void*>(separate)) &&
               separate(nullptr, 4) == cudaErrorInvalidDevice,
           "unrelated explicit same-name function is not hijacked");
    const auto local =
        reinterpret_cast<void* (*)(const char*)>(::dlsym(other, "glimmer_execution_local_lookup"));
    const auto next =
        reinterpret_cast<void* (*)(const char*)>(::dlsym(other, "glimmer_execution_next_lookup"));
    expect(local && next && local("cudaExecutionLocalMarker"),
           "caller-local default scope survives");
    ::dlerror();
    expect(!next("cudaExecutionLocalMarker") && ::dlerror(),
           "true Runtime NEXT starts after actual caller");
    expect(::dlclose(other) == 0, "close unrelated owned loader reference");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 3 || (argc == 4 && std::string_view(argv[3]) == "--concurrent-first"),
               "expected actual fake Runtime/other DSO paths");
        const auto peer = glimmer::control::current_process_identity();
        expect(peer.has_value(), "process identity");
        expect(::setenv("GLIMMER_EXECUTION_SOCKET", "/tmp/glimmer-runtime-lookup-unused.sock", 1) ==
                       0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(peer->pid).c_str(), 1) ==
                       0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_START",
                            std::to_string(peer->start_time_ticks).c_str(), 1) == 0,
               "CPU-only metadata configuration");
        void* runtime = ::dlopen(argv[1], RTLD_NOW | RTLD_NOLOAD);
        expect(runtime, "Runtime fixture remains loaded for cached pointer lifetime");
        if (argc == 4) {
            std::barrier start(8);
            std::vector<std::thread> first;
            for (unsigned int index = 0; index < 8; ++index)
                first.emplace_back([&] {
                    start.arrive_and_wait();
                    expect(owned(::dlsym(runtime, "cudaMalloc")),
                           "concurrent first SDK snapshot is complete");
                    typed();
                });
            for (auto& thread : first) thread.join();
        }
        scopes(runtime, argv[2]);
        typed();
        failures();
        std::vector<std::thread> threads;
        for (unsigned int index = 0; index < 8; ++index)
            threads.emplace_back([] {
                for (unsigned int call = 0; call < 16; ++call) typed();
            });
        for (auto& thread : threads) thread.join();
        const auto child = ::fork();
        expect(child >= 0, "fork cached Runtime fixture");
        if (child == 0) {
            void* address{};
            _exit(cudaGetDriverEntryPoint("cuCtxGetCurrent", &address, 0, nullptr) ==
                              cudaErrorIncompatibleDriverContext &&
                          !::dlsym(RTLD_DEFAULT, "cudaMalloc")
                      ? 0
                      : 1);
        }
        int status{};
        expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                   WEXITSTATUS(status) == 0 && glimmer_execution_runtime_lookup_count(2) == 0 &&
                   glimmer_execution_lookup_count(2) == 0 &&
                   glimmer_execution_runtime_lookup_count(0) == 0 &&
                   glimmer_execution_runtime_lookup_count(1) == 0,
               "all cached calls/fork paths keep original resource dispatch at zero");
        expect(::dlclose(runtime) == 0, "close exact owned Runtime loader reference");
        std::cout << "Public cached Runtime ownership verified: native_resources=0; "
                     "typed_metadata=true\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
