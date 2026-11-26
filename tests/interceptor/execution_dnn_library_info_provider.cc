#include <cudnn.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
}
extern "C" {
unsigned int glimmer_dnn_library_info_native_calls() {
    return calls.load();
}
std::size_t CUDNNWINAPI cudnnGetVersion() {
    ++calls;
    return 0;
}
std::size_t CUDNNWINAPI cudnnGetCudartVersion() {
    ++calls;
    return 0;
}
std::size_t CUDNNWINAPI cudnnGetMaxDeviceVersion() {
    ++calls;
    return 0;
}
cudnnStatus_t CUDNNWINAPI cudnnGetProperty(libraryPropertyType, int*) {
    ++calls;
    return CUDNN_STATUS_NOT_SUPPORTED;
}
}
