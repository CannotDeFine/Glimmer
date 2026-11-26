#pragma once

#include <cuda.h>

namespace glimmer::interceptor {

// Shared typed source intent; no Runtime/Driver handle conversion or ownership.
enum class CudaSourceCaptureKind { kBegin, kEnd };
enum class CudaSourceMemoryOrigin { kDriver, kRuntime };

// Borrowed typed source intent, never deferred or inferred from SDK handles.
struct CudaSourceEventOperation {
    enum class Kind { kRecord, kWait };
    Kind kind{};
    CUevent event{};
    unsigned int flags{};
};

}  // namespace glimmer::interceptor
