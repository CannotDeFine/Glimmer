#include <cuda_runtime.h>

namespace {
__global__ void original_runtime_capture_kernel(unsigned int* counter) {
    atomicAdd(counter, 1U);
}
__global__ void original_runtime_ready_first(unsigned int* counter) {
    *counter += 1U;
}
__global__ void original_runtime_ready_second(unsigned int* counter) {
    *counter += 2U;
}
__global__ void original_runtime_pending_kernel(unsigned int* counter,
                                                unsigned long long duration_nanoseconds) {
    unsigned long long start{}, now{};
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < duration_nanoseconds);
    atomicAdd(counter, 1U);
}
__global__ void original_runtime_graph_dependency_kernel(unsigned int* counter,
                                                         unsigned int* premature_reads,
                                                         const unsigned int* producer,
                                                         unsigned int elements) {
    for (unsigned int index = 0; index < elements; ++index) {
        if (producer[index] != 1U) atomicAdd(premature_reads, 1U);
    }
    atomicAdd(counter, 1U);
}
}  // namespace

extern "C" const void* glimmer_runtime_capture_kernel() {
    return reinterpret_cast<const void*>(original_runtime_capture_kernel);
}

extern "C" const void* glimmer_runtime_ready_kernel(bool is_second) {
    return is_second ? reinterpret_cast<const void*>(original_runtime_ready_second)
                     : reinterpret_cast<const void*>(original_runtime_ready_first);
}
extern "C" const void* glimmer_runtime_pending_kernel() {
    return reinterpret_cast<const void*>(original_runtime_pending_kernel);
}
extern "C" const void* glimmer_runtime_graph_dependency_kernel() {
    return reinterpret_cast<const void*>(original_runtime_graph_dependency_kernel);
}
