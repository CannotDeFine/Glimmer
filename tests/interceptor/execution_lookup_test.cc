#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "internal/process_identity.h"

#include <cuda.h>

#include <array>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);
extern "C" unsigned int glimmer_execution_lookup_count(unsigned int);
extern "C" void* glimmer_execution_driver_internal_lookup(void*, const char*);
namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool belongs(void* address, const char* suffix) {
    Dl_info info{};
    return address && ::dladdr(address, &info) && info.dli_fname &&
           std::string_view(info.dli_fname).ends_with(suffix);
}
void* queried(const char* name, int version) {
    void* output{};
    CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    expect(cuGetProcAddress_v2(name, &output, version, 0, &status) == CUDA_SUCCESS && output &&
               status == CU_GET_PROC_ADDRESS_SUCCESS,
           "typed SDK lookup selects owned wrapper");
    expect(belongs(output, "libglimmer_execution_interceptor.so"), "native pointer cannot escape");
    return output;
}
void typed_lookup() {
    auto old = reinterpret_cast<decltype(&cuGetProcAddress)>(queried("cuGetProcAddress", 11030));
    auto current =
        reinterpret_cast<decltype(&cuGetProcAddress_v2)>(queried("cuGetProcAddress", 12000));
    void* output{};
    expect(old("cuCtxGetCurrent", &output, 4000, 0) == CUDA_SUCCESS && output,
           "cached four-argument resolver remains controlled");
    CUcontext context = reinterpret_cast<CUcontext>(1);
    expect(
        reinterpret_cast<decltype(&cuCtxGetCurrent)>(output)(&context) == CUDA_SUCCESS && !context,
        "cached context query reads virtual thread state, not native SDK");
    CUdriverProcAddressQueryResult status{};
    expect(
        current("cuMemAlloc", &output, 3020, 0, &status) == CUDA_SUCCESS && output && status == 0,
        "cached five-argument resolver preserves exact ABI");
    CUdeviceptr memory{};
    expect(
        reinterpret_cast<decltype(&cuMemAlloc)>(output)(&memory, 4) == CUDA_ERROR_INVALID_CONTEXT,
        "cached allocation cannot escape to native resource creation");
    auto create2 = reinterpret_cast<decltype(&cuCtxCreate_v2)>(queried("cuCtxCreate", 3020));
    auto create3 = reinterpret_cast<decltype(&cuCtxCreate_v3)>(queried("cuCtxCreate", 11040));
    auto create4 = reinterpret_cast<decltype(&cuCtxCreate_v4)>(queried("cuCtxCreate", 12050));
    CUexecAffinityParam ignored{};
    CUctxCreateParams parameters{&ignored, 0, nullptr};
    CUctxCreateParams empty_parameters{};
    // Scheduling flags are now supported. A coredump-policy request remains
    // unsupported and still proves each typed ABI reaches owned validation.
    expect(
        create2(&context, CU_CTX_COREDUMP_ENABLE, 0) == CUDA_ERROR_NOT_SUPPORTED &&
            create3(&context, nullptr, 0, CU_CTX_COREDUMP_ENABLE, 0) == CUDA_ERROR_NOT_SUPPORTED &&
            create4(&context, &parameters, CU_CTX_COREDUMP_ENABLE, 0) == CUDA_ERROR_NOT_SUPPORTED &&
            create4(&context, &empty_parameters, CU_CTX_COREDUMP_ENABLE, 0) ==
                CUDA_ERROR_NOT_SUPPORTED,
        "context ABIs invoke owned validation, never guessed casts");
    expect(create3(&context, nullptr, 1, 0, 0) == CUDA_ERROR_NOT_SUPPORTED &&
               create4(&context, reinterpret_cast<CUctxCreateParams*>(1), 0, 0) ==
                   CUDA_ERROR_INVALID_VALUE &&
               create4(&context, nullptr, 0, 0) == CUDA_ERROR_INVALID_VALUE,
           "nondefault partition requests reject before resource mutation");
    parameters.numExecAffinityParams = 1;
    expect(create4(&context, &parameters, 0, 0) == CUDA_ERROR_NOT_SUPPORTED,
           "nonempty affinity cannot partition the compared device");
    parameters.numExecAffinityParams = -1;
    expect(create4(&context, &parameters, 0, 0) == CUDA_ERROR_INVALID_VALUE &&
               create3(&context, nullptr, -1, 0, 0) == CUDA_ERROR_INVALID_VALUE,
           "negative affinity counts are invalid, not native fallback");
    CUctxCigParam graphics{};
    parameters = {nullptr, 0, &graphics};
    expect(create4(&context, &parameters, 0, 0) == CUDA_ERROR_NOT_SUPPORTED,
           "graphics context family never escapes the execution domain");
    parameters.execAffinityParams = &ignored;
    expect(create4(&context, &parameters, 0, 0) == CUDA_ERROR_INVALID_VALUE,
           "mutually exclusive native context families reject safely");
    auto launch = reinterpret_cast<decltype(&cuLaunchKernel)>(queried("cuLaunchKernel", 11030));
    expect(launch(nullptr, 320, 1, 1, 32, 1, 1, 0, nullptr, nullptr, nullptr) ==
               CUDA_ERROR_INVALID_HANDLE,
           "cached launch retains virtual-handle admission");
}
void rejected_lookup() {
    CUdriverProcAddressQueryResult uninitialized{};
    const int unknown_bytes = 0x71717171;
    std::memcpy(&uninitialized, &unknown_bytes, sizeof(uninitialized));
    void* selected{};
    expect(cuGetProcAddress_v2("cuCtxGetCurrent", &selected, 12000, 0, &uninitialized) ==
                   CUDA_SUCCESS &&
               selected && uninitialized == CU_GET_PROC_ADDRESS_SUCCESS,
           "SDK status output need not contain an initialized C++ enum");
    for (const auto* name : {"missing", "old_version"}) {
        void* output = reinterpret_cast<void*>(1);
        CUdriverProcAddressQueryResult status{};
        expect(cuGetProcAddress_v2(name, &output, 12000, 0, &status) == CUDA_SUCCESS && !output &&
                   status == (std::string_view(name) == "missing"
                                  ? CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND
                                  : CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT),
               "native missing/version results remain successful NULL");
    }
    for (const auto& item :
         std::array{std::pair{"cuMemAlloc", 2000}, std::pair{"cuLookupUnsupported", 12000}}) {
        void* output = reinterpret_cast<void*>(1);
        expect(cuGetProcAddress(item.first, &output, item.second, 0) == CUDA_ERROR_NOT_SUPPORTED &&
                   !output,
               "unsupported SDK ABI never leaks its original entry");
    }
    void* output = reinterpret_cast<void*>(1);
    CUdriverProcAddressQueryResult status{};
    expect(
        cuGetProcAddress_v2("sdk_error", &output, 12000, 0, &status) == CUDA_ERROR_DEINITIALIZED &&
            !output,
        "original SDK error is unchanged without a usable native pointer");
    expect(cuGetProcAddress_v2("malformed", &output, 12000, 0, &status) == CUDA_ERROR_UNKNOWN &&
               !output && status == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT,
           "malformed pointer/status success cannot authorize execution");
    expect(cuGetProcAddress_v2("cuLaunchKernel", &output, 12000,
                               CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
                               nullptr) == CUDA_ERROR_NOT_SUPPORTED &&
               !output,
           "PTDS cannot be replaced by legacy semantics");
    const auto before = glimmer_execution_lookup_count(0) + glimmer_execution_lookup_count(1);
    expect(cuGetProcAddress("cuMemAlloc", reinterpret_cast<void**>(1), 3020, 0) ==
                   CUDA_ERROR_INVALID_VALUE &&
               cuGetProcAddress(reinterpret_cast<const char*>(1), &output, 3020, 0) ==
                   CUDA_ERROR_INVALID_VALUE &&
               cuGetProcAddress_v2("cuMemAlloc", &output, 3020, 0,
                                   reinterpret_cast<CUdriverProcAddressQueryResult*>(1)) ==
                   CUDA_ERROR_INVALID_VALUE,
           "unreadable name/output/status rejects before SDK lookup");
    expect(before == glimmer_execution_lookup_count(0) + glimmer_execution_lookup_count(1),
           "invalid storage issues no native metadata operation");
}
void scopes(const char* driver_path, const char* other_path) {
    void* driver = ::dlopen(driver_path, RTLD_NOW | RTLD_NOLOAD);
    void* other = ::dlopen(other_path, RTLD_NOW | RTLD_LOCAL);
    expect(driver && other, "load exact fixture scopes");
    expect(belongs(::dlsym(driver, "cuCtxGetCurrent"), "libglimmer_execution_interceptor.so"),
           "explicit SDK handle stays controlled");
    auto* separate = ::dlsym(other, "cuCtxGetCurrent");
    expect(belongs(separate, "libglimmer_execution_lookup_other.so") &&
               reinterpret_cast<decltype(&cuCtxGetCurrent)>(separate)(nullptr) ==
                   CUDA_ERROR_INVALID_DEVICE,
           "unrelated explicit same-name scope is not hijacked");
    expect(belongs(glimmer_execution_driver_internal_lookup(driver, "cuCtxGetCurrent"),
                   "libcuda.so.execution-lookup-fixture.so"),
           "native internal explicit lookup retains the original SDK leaf");
    auto local =
        reinterpret_cast<void* (*)(const char*)>(::dlsym(other, "glimmer_execution_local_lookup"));
    auto next =
        reinterpret_cast<void* (*)(const char*)>(::dlsym(other, "glimmer_execution_next_lookup"));
    expect(local && next &&
               belongs(local("cuExecutionLocalMarker"), "libglimmer_execution_lookup_other.so"),
           "caller-relative RTLD_DEFAULT retains caller-local exports");
    ::dlerror();
    expect(local("cuExecutionAbsoluteZero") == nullptr && ::dlerror() == nullptr,
           "valid zero-address symbol is not a fabricated loader failure");
    ::dlerror();
    expect(next("cuExecutionLocalMarker") == nullptr && ::dlerror() != nullptr,
           "RTLD_NEXT begins after actual caller, not interceptor");
    ::dlerror();
    expect(::dlsym(driver, "cuLookupUnsupported") == nullptr && ::dlerror() != nullptr,
           "unsupported explicit SDK lookup returns a real loader error");
    expect(::dlclose(other) == 0 && ::dlclose(driver) == 0, "close owned loader references");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 3, "expected fake Driver and unrelated scope paths");
        const auto identity = glimmer::control::current_process_identity();
        expect(identity.has_value(), "actual frontend identity");
        expect(::setenv("GLIMMER_EXECUTION_SOCKET", "/tmp/glimmer-lookup-unused.sock", 1) == 0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(identity->pid).c_str(),
                            1) == 0 &&
                   ::setenv("GLIMMER_EXECUTION_SERVER_START",
                            std::to_string(identity->start_time_ticks).c_str(), 1) == 0,
               "metadata configuration creates no connection or native context");
        typed_lookup();
        rejected_lookup();
        scopes(argv[1], argv[2]);
        std::vector<std::thread> threads;
        for (unsigned int index = 0; index < 8; ++index)
            threads.emplace_back([] {
                for (unsigned int call = 0; call < 32; ++call) typed_lookup();
            });
        for (auto& thread : threads) thread.join();
        const auto child = ::fork();
        expect(child >= 0, "fork inherited cached-pointer fixture");
        if (child == 0) {
            void* output{};
            _exit(cuGetProcAddress("cuCtxGetCurrent", &output, 4000, 0) ==
                              CUDA_ERROR_INVALID_CONTEXT &&
                          ::dlsym(RTLD_DEFAULT, "cuCtxGetCurrent") == nullptr
                      ? EXIT_SUCCESS
                      : EXIT_FAILURE);
        }
        int status{};
        expect(
            ::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "fork rejects before inherited SDK lookup state");
        expect(glimmer_execution_lookup_count(2) == 0,
               "cached resources never escape to native creation/launch");
        std::cout << "Execution cached Driver ABI and caller-relative lookup checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
