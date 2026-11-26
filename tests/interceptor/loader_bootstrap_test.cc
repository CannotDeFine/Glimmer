#include <dlfcn.h>

#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {
bool is_interceptor(void* address) {
    Dl_info info{};
    return address && dladdr(address, &info) != 0 && info.dli_fname &&
           std::strstr(info.dli_fname, "libglimmer_cuda_interceptor.so");
}
}  // namespace

// Deliberately an unsanitized ordinary application. The instrumented preload
// must survive ASan initialization before this application's main is entered.
int main() {
    for (const auto* name : {"mmap", "malloc", "free", "snprintf", "dlopen", "pthread_create"}) {
        void* global = dlsym(RTLD_DEFAULT, name);
        if (!global || is_interceptor(global)) {
            return EXIT_FAILURE;
        }
    }
    for (const auto* name : {"", "c", "cu", "nv", "nvm", "__", "__cuda"}) {
        if (dlsym(RTLD_DEFAULT, name) || dlsym(RTLD_NEXT, name)) {
            return EXIT_FAILURE;
        }
    }
    for (const auto* name : {"cuInit", "cudaMalloc", "__cudaLaunchKernel", "nvmlInit"}) {
        if (!is_interceptor(dlsym(RTLD_DEFAULT, name))) {
            return EXIT_FAILURE;
        }
    }
    void* libc = dlopen("libc.so.6", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!libc) {
        return EXIT_FAILURE;
    }
    void* native_mmap = dlsym(libc, "mmap");
    const bool is_native = native_mmap && !is_interceptor(native_mmap);
    if (dlclose(libc) != 0 || !is_native) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
