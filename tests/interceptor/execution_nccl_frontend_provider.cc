#include <nccl.h>

#include <atomic>
#include <cstddef>

namespace {
std::atomic<unsigned int> native_calls{};
ncclResult_t unexpected() noexcept {
    ++native_calls;
    return ncclInternalError;
}
}  // namespace
extern "C" unsigned int glimmer_nccl_frontend_native_calls() {
    return native_calls.load();
}
// Exact caller-provider ABIs. Controlled wrappers must never invoke any of them.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
extern "C" ncclResult_t ncclGetUniqueId(ncclUniqueId*) {
    return unexpected();
}
extern "C" ncclResult_t ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int) {
    return unexpected();
}
extern "C" ncclResult_t ncclCommInitRankConfig(ncclComm_t*, int, ncclUniqueId, int, ncclConfig_t*) {
    return unexpected();
}
extern "C" ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t*) {
    return unexpected();
}
extern "C" ncclResult_t ncclCommDestroy(ncclComm_t) {
    return unexpected();
}
extern "C" ncclResult_t ncclCommAbort(ncclComm_t) {
    return unexpected();
}
extern "C" ncclResult_t ncclGroupStart() {
    return unexpected();
}
extern "C" ncclResult_t ncclGroupEnd() {
    return unexpected();
}
extern "C" ncclResult_t ncclAllGather(const void*, void*, std::size_t, ncclDataType_t, ncclComm_t,
                                      cudaStream_t) {
    return unexpected();
}
extern "C" ncclResult_t ncclAllReduce(const void*, void*, std::size_t, ncclDataType_t, ncclRedOp_t,
                                      ncclComm_t, cudaStream_t) {
    return unexpected();
}
extern "C" ncclResult_t ncclBcast(void*, std::size_t, ncclDataType_t, int, ncclComm_t,
                                  cudaStream_t) {
    return unexpected();
}
extern "C" ncclResult_t ncclCommCount(ncclComm_t, int*) {
    return unexpected();
}
extern "C" ncclResult_t pncclAllGather(const void*, void*, std::size_t, ncclDataType_t, ncclComm_t,
                                       cudaStream_t) {
    return unexpected();
}
// NOLINTEND(bugprone-easily-swappable-parameters)
