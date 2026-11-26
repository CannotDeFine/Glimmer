#include <cudnn.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {
std::atomic<unsigned int> calls{};
template <typename Handle>
cudnnStatus_t unexpected_create(Handle* output) {
    ++calls;
    // Any native fallback visibly changes the output, without a GPU call.
    // NOLINTNEXTLINE(performance-no-int-to-ptr): synthetic, never dereferenced.
    if (output) *output = reinterpret_cast<Handle>(std::uintptr_t{0x9bad});
    return CUDNN_STATUS_SUCCESS;
}
}  // namespace

extern "C" unsigned int glimmer_dnn_profile_native_calls() {
    return calls.load();
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" {
cudnnStatus_t CUDNNWINAPI cudnnCreate(cudnnHandle_t* output) {
    return unexpected_create(output);
}
cudnnStatus_t CUDNNWINAPI cudnnDestroy(cudnnHandle_t) {
    ++calls;
    return CUDNN_STATUS_SUCCESS;
}
cudnnStatus_t CUDNNWINAPI cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* output) {
    return unexpected_create(output);
}
cudnnStatus_t CUDNNWINAPI cudnnSetTensor(cudnnHandle_t, cudnnTensorDescriptor_t, void*,
                                         const void*) {
    ++calls;
    return CUDNN_STATUS_SUCCESS;
}
std::size_t CUDNNWINAPI cudnnGetVersion() {
    ++calls;
    return CUDNN_VERSION;
}
}
#pragma GCC diagnostic pop
