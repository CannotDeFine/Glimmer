#include "internal/process_identity.h"

#include <cuda.h>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename Function>
Function symbol(void* library, const char* name) {
    auto* value = ::dlsym(library, name);
    expect(value != nullptr, "original initialization helper exists");
    return reinterpret_cast<Function>(value);
}
void run(const char* provider) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual process identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    const auto socket = "/tmp/glimmer-no-initialization-endpoint-" + pid + ".sock";
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", socket.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "bind CPU-only configuration without starting a server");
    auto missing = reinterpret_cast<CUcontext>(1);
    expect(cuCtxGetCurrent(&missing) == CUDA_ERROR_NOT_INITIALIZED &&
               missing == reinterpret_cast<CUcontext>(1),
           "missing SDK is not successful Driver initialization");
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library != nullptr, "load exact initialization SDK fixture");
    const auto queries = symbol<unsigned int (*)()>(library, "glimmer_initialization_queries");
    const auto hold = symbol<void (*)()>(library, "glimmer_initialization_hold");
    const auto entered = symbol<bool (*)()>(library, "glimmer_initialization_entered");
    const auto finish = symbol<void (*)()>(library, "glimmer_initialization_release");
    auto* selected = ::dlsym(library, "cuCtxGetCurrent");
    expect(selected == ::dlsym(RTLD_DEFAULT, "cuCtxGetCurrent") &&
               selected == reinterpret_cast<void*>(&cuCtxGetCurrent),
           "direct/global/explicit current-context identity");
    const auto cached = reinterpret_cast<decltype(&cuCtxGetCurrent)>(selected);
    for (const auto get : {&cuCtxGetCurrent, cached}) {
        auto context = reinterpret_cast<CUcontext>(1);
        expect(get(&context) == CUDA_ERROR_NOT_INITIALIZED &&
                   context == reinterpret_cast<CUcontext>(1) &&
                   get(nullptr) == CUDA_ERROR_NOT_INITIALIZED,
               "pre-init original error wins without publishing caller output");
    }
    expect(cuInit(1) == CUDA_ERROR_INVALID_VALUE, "original rejected initialization is unchanged");
    CUcontext context{};
    expect(cached(&context) == CUDA_ERROR_NOT_INITIALIZED && !context,
           "failed initialization does not create logical success");
    hold();
    auto blocked = std::async(std::launch::async, [&] {
        auto value = reinterpret_cast<CUcontext>(1);
        expect(cached(&value) == CUDA_SUCCESS && !value,
               "completed native observation publishes only owned virtual state");
    });
    struct Release {
        void (*finish)();
        ~Release() {
            finish();
        }
    } release{finish};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (!entered()) {
        expect(std::chrono::steady_clock::now() < deadline, "bounded original SDK query entry");
        std::this_thread::yield();
    }
    expect(cuInit(0) == CUDA_SUCCESS,
           "explicit original initialization succeeds without a GPU context");
    auto independent = std::async(std::launch::async, [] { return cuCtxSetCurrent(nullptr); });
    expect(independent.wait_for(std::chrono::seconds{1}) == std::future_status::ready &&
               independent.get() == CUDA_SUCCESS,
           "withheld SDK metadata query never holds frontend exclusion");
    finish();
    blocked.get();
    const auto observed = queries();
    expect(
        cached(&context) == CUDA_SUCCESS && !context && cached(nullptr) == CUDA_ERROR_INVALID_VALUE,
        "initialized projection preserves validation without exposing the native handle witness");
    auto other = std::async(std::launch::async, [&] {
        auto value = reinterpret_cast<CUcontext>(1);
        return cached(&value) == CUDA_SUCCESS && !value;
    });
    expect(other.get() && queries() == observed,
           "process initialization observation caches across threads");
    const auto child = ::fork();
    expect(child >= 0, "fork exclusion child");
    if (!child) ::_exit(cached(&context) == CUDA_ERROR_INVALID_CONTEXT ? 0 : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0 &&
               queries() == observed,
           "fork rejects before SDK, locks or context state");
    expect(::dlclose(library) == 0, "release caller SDK reference");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2, "initialization fixture provider argument");
        run(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
