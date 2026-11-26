#pragma once

namespace glimmer::backend::cuda {

// Trusted CPU-only measurement lease. The physical owner calls begin once and
// finish only after final retirement; destruction cancels an unfinished sample.
// No SDK, application callbacks or external writes. Calls run outside exclusion.
class CudaExecutionObserver {
   public:
    virtual ~CudaExecutionObserver() = default;
    virtual void begin() noexcept = 0;
    virtual void finish(bool is_successful) noexcept = 0;
};

}  // namespace glimmer::backend::cuda
