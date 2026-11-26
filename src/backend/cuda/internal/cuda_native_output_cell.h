#pragma once

#include <cuda.h>

#include <cstddef>
#include <memory>
#include <span>

namespace glimmer::backend::cuda {

struct CudaNativeOutputObservation {
    CUresult status{CUDA_ERROR_UNKNOWN};
    bool was_written{};
};
struct CudaNativeOutputCellCreation;

// Linux-only isolated, sealed-file-backed private output page. No SDK calls,
// signal handlers, pagemap writes, process-global dirty-bit resets or privileged
// interfaces are used. A native store, including an equal-value store, makes
// this mapping anonymous through copy-on-write. Evidence is never reset.
//
// This is ANY-WRITE evidence for one isolated page, not a byte mask or proof of
// a full typed write, requested status, native completion or context ownership.
// Trusted users must limit native access to the supplied cell extent, retain
// the cell through every possible deferred write, and prove writer quiescence
// before snapshot or destruction. They must not remap, discard, or change the
// protections/advice of this private mapping.
class CudaNativeOutputCell final {
   public:
    // The seed is a nonempty, readable owned span no larger than one actual
    // system base page. It is copied before native access becomes possible.
    [[nodiscard]] static CudaNativeOutputCellCreation create(
        std::span<const std::byte> initial) noexcept;
    ~CudaNativeOutputCell();
    CudaNativeOutputCell(const CudaNativeOutputCell&) = delete;
    CudaNativeOutputCell& operator=(const CudaNativeOutputCell&) = delete;

    // May be used by a distinct native worker in the creating process. An
    // inherited post-fork cell returns NULL before touching its absent mapping.
    [[nodiscard]] void* data() const noexcept;
    // Caller MUST independently prove quiescence. Destination must have exactly
    // the original seed size and must not overlap this page. Bytes are copied
    // read-only from the cell before checking page provenance; they are not a
    // qualified observation unless the returned status is CUDA_SUCCESS.
    [[nodiscard]] CudaNativeOutputObservation snapshot(std::span<std::byte> output) const noexcept;

   private:
    struct Implementation;
    explicit CudaNativeOutputCell(std::unique_ptr<Implementation> implementation) noexcept;
    std::unique_ptr<Implementation> implementation_;
};
struct CudaNativeOutputCellCreation {
    CUresult status{CUDA_ERROR_INVALID_VALUE};
    std::shared_ptr<CudaNativeOutputCell> cell{};
};

}  // namespace glimmer::backend::cuda
