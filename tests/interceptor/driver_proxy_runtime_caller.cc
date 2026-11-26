#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

// Build this caller in a Runtime-named DSO with sibling-call optimization off:
// the native caller identity, not a test-only dispatcher hook, selects routing.
extern "C" void* glimmer_proxy_runtime_lookup(void* handle, const char* name) {
    return dlsym(handle, name);
}
