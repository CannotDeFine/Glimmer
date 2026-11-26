#pragma once

#include "internal/cuda_execution_session.h"
#include "glimmer/control/execution_process_memory_protocol.h"

#include <cstdint>

namespace glimmer::app {
// A missing native result is not proof that the requested cleanup ran.
inline bool completed_idle_cleanup(std::uint64_t allocation,
                                   const backend::cuda::CudaSessionResult& result) noexcept {
    return !(allocation & control::kExecutionProcessAllocationTag) &&
           result.disposition == backend::cuda::CudaSessionDisposition::kComplete &&
           result.status == CUDA_SUCCESS && result.original_status == CUDA_SUCCESS;
}

inline bool completed_idle_cleanup(const backend::cuda::CudaSessionBlasResult& result) noexcept {
    return result.disposition == backend::cuda::CudaSessionDisposition::kComplete &&
           result.status == CUDA_SUCCESS && result.original_status == CUBLAS_STATUS_SUCCESS;
}
}  // namespace glimmer::app
