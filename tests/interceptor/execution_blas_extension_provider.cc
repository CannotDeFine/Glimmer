#include <cublas_v2.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cublasStatus_t unexpected() {
    ++calls;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
}  // namespace

extern "C" unsigned int glimmer_blas_extension_native_calls() {
    return calls.load();
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed cuBLAS C ABIs.
extern "C" cublasStatus_t CUBLASWINAPI cublasSgemv_v2(cublasHandle_t, cublasOperation_t, int, int,
                                                      const float*, const float*, int, const float*,
                                                      int, const float*, float*, int) {
    return unexpected();
}
extern "C" cublasStatus_t CUBLASWINAPI cublasSgemmEx(cublasHandle_t, cublasOperation_t,
                                                     cublasOperation_t, int, int, int, const float*,
                                                     const void*, cudaDataType, int, const void*,
                                                     cudaDataType, int, const float*, void*,
                                                     cudaDataType, int) {
    return unexpected();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
