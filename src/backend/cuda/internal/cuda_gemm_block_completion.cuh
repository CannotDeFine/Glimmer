#pragma once

namespace glimmer::backend::cuda {

// Private observed-provider witness, never a stop/completion authority. Call
// only after the whole-CTA barrier following every final tile output store.
template <bool Observes>
class GemmBlockCompletion {
   public:
    __device__ explicit GemmBlockCompletion(unsigned int* completed,
                                            unsigned long long* counter = nullptr)
        : completed_(completed),
          election_(Observes && counter ? counter + 1 : nullptr),
          should_publish_(Observes && completed && election_ && threadIdx.x == 0) {}

    __device__ bool publish() {
        if constexpr (Observes) {
            if (should_publish_) {
                should_publish_ = false;
                // Election is on the private device allocation, never on the
                // mapped host witness. Retained resumes keep the winning word.
                if (atomicCAS(election_, 0ULL, 1ULL) != 0) return false;
                asm volatile("st.release.sys.global.u32 [%0], %1;"
                             :
                             : "l"(completed_), "r"(1U)
                             : "memory");
                return true;
            }
        }
        return false;
    }

   private:
    unsigned int* completed_;
    unsigned long long* election_;
    bool should_publish_;
};

}  // namespace glimmer::backend::cuda
