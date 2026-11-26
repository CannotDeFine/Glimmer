#include <cublasLt.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace {
std::atomic<unsigned int> calls{};
cublasStatus_t unexpected() {
    ++calls;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
}  // namespace
extern "C" unsigned int glimmer_lt_native_calls() {
    return calls.load();
}
extern "C" {
cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t*, cudaDataType,
                                                       std::uint64_t, std::uint64_t, std::int64_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmulDescCreate(cublasLtMatmulDesc_t*, cublasComputeType_t,
                                                     cudaDataType) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t*) {
    return unexpected();
}
#define LT_FIXTURE(PREFIX, HANDLE, ATTRIBUTE)                                               \
    cublasStatus_t CUBLASWINAPI PREFIX##Destroy(HANDLE) {                                   \
        return unexpected();                                                                \
    }                                                                                       \
    cublasStatus_t CUBLASWINAPI PREFIX##SetAttribute(HANDLE, ATTRIBUTE, const void*,        \
                                                     std::size_t) {                         \
        return unexpected();                                                                \
    }                                                                                       \
    cublasStatus_t CUBLASWINAPI PREFIX##GetAttribute(HANDLE, ATTRIBUTE, void*, std::size_t, \
                                                     std::size_t*) {                        \
        return unexpected();                                                                \
    }
LT_FIXTURE(cublasLtMatrixLayout, cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t)
LT_FIXTURE(cublasLtMatmulDesc, cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t)
LT_FIXTURE(cublasLtMatmulPreference, cublasLtMatmulPreference_t,
           cublasLtMatmulPreferenceAttributes_t)
#undef LT_FIXTURE
std::size_t CUBLASWINAPI cublasLtGetVersion() {
    ++calls;
    return 1;
}
std::size_t CUBLASWINAPI cublasLtGetCudartVersion() {
    ++calls;
    return 1;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed SDK ABI.
cublasStatus_t CUBLASWINAPI cublasLtCreate(cublasLtHandle_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtDestroy(cublasLtHandle_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int,
    cublasLtMatmulHeuristicResult_t*, int*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCheck(cublasLtHandle_t, cublasLtMatmulDesc_t,
                                                    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
                                                    const cublasLtMatmulAlgo_t*,
                                                    cublasLtMatmulHeuristicResult_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmulAlgoCapGetAttribute(const cublasLtMatmulAlgo_t*,
                                                              cublasLtMatmulAlgoCapAttributes_t,
                                                              void*, std::size_t, std::size_t*) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatrixTransform(cublasLtHandle_t, cublasLtMatrixTransformDesc_t,
                                                    const void*, const void*,
                                                    cublasLtMatrixLayout_t, const void*,
                                                    const void*, cublasLtMatrixLayout_t, void*,
                                                    cublasLtMatrixLayout_t, cudaStream_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatrixLayoutInit_internal(cublasLtMatrixLayout_t, std::size_t,
                                                              cudaDataType, std::uint64_t,
                                                              std::uint64_t, std::int64_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmulDescInit_internal(cublasLtMatmulDesc_t, std::size_t,
                                                            cublasComputeType_t, cudaDataType) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmulPreferenceInit_internal(cublasLtMatmulPreference_t,
                                                                  std::size_t) {
    return unexpected();
}
cublasStatus_t CUBLASWINAPI cublasLtMatmul(cublasLtHandle_t, cublasLtMatmulDesc_t, const void*,
                                           const void*, cublasLtMatrixLayout_t, const void*,
                                           cublasLtMatrixLayout_t, const void*, const void*,
                                           cublasLtMatrixLayout_t, void*, cublasLtMatrixLayout_t,
                                           const cublasLtMatmulAlgo_t*, void*, std::size_t,
                                           cudaStream_t) {
    return unexpected();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
}
