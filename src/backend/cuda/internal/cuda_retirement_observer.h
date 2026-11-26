#pragma once

#include <cstdint>

namespace glimmer::backend::cuda {

// CPU-only passive observation, invoked under the existing dispatcher callback
// exclusion after actual native retirement. It must not reenter the owner,
// invoke the SDK, or turn an observation into execution authority.
class CudaRetirementObserver {
   public:
    virtual ~CudaRetirementObserver() = default;
    // Physical retirement can succeed after an execution/progress error.
    // The status preserves that failure; retirement alone is not success.
    virtual void retired(std::uint64_t sequence, int status) noexcept = 0;
};

}  // namespace glimmer::backend::cuda
