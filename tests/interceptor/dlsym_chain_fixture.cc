#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>

extern "C" {
thread_local int glimmer_lookup_tls = 0;
int glimmer_lookup_marker() {
    return GLIMMER_LOOKUP_ID;
}
int cuCallerLookupMarker() {
    return GLIMMER_LOOKUP_ID;
}
int nvml_caller_lookup_marker() {
    return GLIMMER_LOOKUP_ID;
}
#if GLIMMER_LOOKUP_ID == 4
int glimmer_local_lookup_marker() {
    return 4;
}
int cuLocalLookupMarker() {
    return 4;
}
#endif
// These fixtures are compiled without sibling-call optimization: the loader
// must observe this DSO, not the application's call to the fixture helper.
void* GLIMMER_LOOKUP_ENTRY(void* handle, const char* name) {
    return dlsym(handle, name);
}

#if GLIMMER_LOOKUP_ID == 3
asm(".globl glimmer_lookup_absolute_zero\n"
    ".set glimmer_lookup_absolute_zero, 0\n");
#endif
}
