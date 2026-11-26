#include <curand.h>
#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
curandStatus_t unexpected() {
    ++calls;
    return CURAND_STATUS_INTERNAL_ERROR;
}
}  // namespace
extern "C" unsigned int glimmer_random_native_calls() {
    return calls.load();
}
extern "C" {
curandStatus_t CURANDAPI curandCreateGenerator(curandGenerator_t*, curandRngType_t) {
    return unexpected();
}
curandStatus_t CURANDAPI curandDestroyGenerator(curandGenerator_t) {
    return unexpected();
}
curandStatus_t CURANDAPI curandSetPseudoRandomGeneratorSeed(curandGenerator_t, unsigned long long) {
    return unexpected();
}
curandStatus_t CURANDAPI curandGenerateUniform(curandGenerator_t, float*, std::size_t) {
    return unexpected();
}
}
