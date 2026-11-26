#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <dlfcn.h>

namespace {
bool owned(void* address) {
    Dl_info info{};
    return address && ::dladdr(address, &info) && info.dli_fname &&
           std::strstr(info.dli_fname, "libglimmer_execution_interceptor.so");
}
}  // namespace
// Unsanitized ordinary application: the injected sanitizer must bootstrap cold.
int main() {
    for (const auto* name : {"mmap", "malloc", "free", "snprintf", "dlopen", "pthread_create"}) {
        void* result = ::dlsym(RTLD_DEFAULT, name);
        if (!result || owned(result)) return EXIT_FAILURE;
    }
    for (const auto* name : {"", "c", "cu", "__", "__cuda"})
        if (::dlsym(RTLD_DEFAULT, name) || ::dlsym(RTLD_NEXT, name)) return EXIT_FAILURE;
    for (const auto* name : {"cuInit", "cuCtxGetCurrent", "cuLaunchKernel", "cuGetProcAddress",
                             "cuGetProcAddress_v2", "cudaMalloc", "cudaGetDriverEntryPoint",
                             "__cudaRegisterFatBinary", "__cudaUnregisterFatBinary"})
        if (!owned(::dlsym(RTLD_DEFAULT, name))) return EXIT_FAILURE;
    void* libc = ::dlopen("libc.so.6", RTLD_NOW | RTLD_NOLOAD);
    if (!libc) return EXIT_FAILURE;
    void* native = ::dlsym(libc, "mmap");
    const bool is_native = native && !owned(native);
    return ::dlclose(libc) == 0 && is_native ? EXIT_SUCCESS : EXIT_FAILURE;
}
