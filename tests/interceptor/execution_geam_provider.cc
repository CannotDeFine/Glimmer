#include <cublas_v2.h>

#include <atomic>

namespace {
std::atomic<unsigned int> calls{};
cublasStatus_t unexpected() {
    ++calls;
    return CUBLAS_STATUS_NOT_SUPPORTED;
}
}  // namespace

extern "C" unsigned int glimmer_geam_native_calls() {
    return calls.load();
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact installed S/D GEAM ABIs.
extern "C" cublasStatus_t CUBLASWINAPI cublasSgeam(cublasHandle_t, cublasOperation_t,
                                                   cublasOperation_t, int, int, const float*,
                                                   const float*, int, const float*, const float*,
                                                   int, float*, int) {
    return unexpected();
}
extern "C" cublasStatus_t CUBLASWINAPI cublasDgeam(cublasHandle_t, cublasOperation_t,
                                                   cublasOperation_t, int, int, const double*,
                                                   const double*, int, const double*, const double*,
                                                   int, double*, int) {
    return unexpected();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
