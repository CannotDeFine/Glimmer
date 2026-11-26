#include <cublas_v2.h>
#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cublasStatus_t unexpected() {
    ++calls;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_blas_native_calls() {
    return calls.load();
}
extern "C" {
cublasStatus_t CUBLASWINAPI cublasCreate_v2(cublasHandle_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasDestroy_v2(cublasHandle_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasSetStream_v2(cublasHandle_t, cudaStream_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGetStream_v2(cublasHandle_t, cudaStream_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasSetPointerMode_v2(cublasHandle_t, cublasPointerMode_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGetPointerMode_v2(cublasHandle_t, cublasPointerMode_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasSetMathMode(cublasHandle_t, cublasMath_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGetMathMode(cublasHandle_t, cublasMath_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGetVersion_v2(cublasHandle_t, int*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasSetWorkspace_v2(cublasHandle_t, void*, std::size_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasSetAtomicsMode(cublasHandle_t, cublasAtomicsMode_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGetAtomicsMode(cublasHandle_t, cublasAtomicsMode_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasSetSmCountTarget(cublasHandle_t, int) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGetSmCountTarget(cublasHandle_t, int*) {
    return unexpected();
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact native cuBLAS ABI.
cublasStatus_t CUBLASWINAPI cublasSgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                           int, int, int, const float*, const float*, int,
                                           const float*, int, const float*, float*, int) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int,
                                         int, int, const void*, const void*, cudaDataType, int,
                                         const void*, cudaDataType, int, const void*, void*,
                                         cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasGemmStridedBatchedEx(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*,
    cudaDataType, int, long long, const void*, cudaDataType, int, long long, const void*, void*,
    cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasSgemmStridedBatched(cublasHandle_t, cublasOperation_t,
                                                      cublasOperation_t, int, int, int,
                                                      const float*, const float*, int, long long,
                                                      const float*, int, long long, const float*,
                                                      float*, int, long long, int) {
    return unexpected();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
}
