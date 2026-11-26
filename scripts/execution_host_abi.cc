#include "runtime.h"
#include "internal/cuda_execution_session.h"
#include "glimmer/control/execution_protocol.h"
#include "glimmer/core/execution_priority.h"

#include <iostream>

#ifdef NDEBUG
#error "This diagnostic requires assertions to remain enabled."
#endif

template <typename T>
void layout(const char* name) {
    std::cout << name << ' ' << sizeof(T) << ' ' << alignof(T) << '\n';
}

int main() {
    layout<glimmer::app::ExecutionServiceConfig>("ExecutionServiceConfig");
    layout<glimmer::backend::cuda::CudaSessionDispatch>("CudaSessionDispatch");
    layout<glimmer::backend::cuda::CudaExecutionKernel>("CudaExecutionKernel");
    layout<glimmer::control::ExecutionWireReply>("ExecutionWireReply");
    layout<glimmer::core::ExecutionPriorityPlan>("ExecutionPriorityPlan");
}
