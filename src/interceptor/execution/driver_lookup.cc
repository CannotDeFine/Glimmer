#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "driver_lookup.h"
#include "runtime_lookup.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>
#ifdef GLIMMER_HAS_NCCL
#include <nccl.h>
#endif
#ifdef GLIMMER_HAS_CUDNN
#include "../../internal/cudnn_provider_identity.h"
#include <cudnn.h>
#endif

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string_view>
#include <type_traits>
#include <dlfcn.h>
#include <link.h>
#include <unistd.h>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
#ifdef cuLaunchKernel
#undef cuLaunchKernel
#endif
#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
#ifdef cuDevicePrimaryCtxSetFlags
#undef cuDevicePrimaryCtxSetFlags
#endif
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxRelease(CUdevice);
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxReset(CUdevice);
extern "C" CUresult CUDAAPI cuDevicePrimaryCtxSetFlags(CUdevice, unsigned int);
extern "C" CUresult CUDAAPI cuMemcpy_ptds(CUdeviceptr, CUdeviceptr, std::size_t);
extern "C" CUresult CUDAAPI cuMemcpyAsync_ptsz(CUdeviceptr, CUdeviceptr, std::size_t, CUstream);
extern "C" cudaError_t CUDARTAPI cudaLaunchHostFunc_ptsz(cudaStream_t, cudaHostFn_t, void*);

namespace glimmer::interceptor::execution {
namespace {
const pid_t kLookupProcess{::getpid()};
using NativeLookup = void* (*)(void*, const char*);
// ASan resolves its own primitives before shadow initialization. This leaf and
// the native forwarding branch must remain uninstrumented and allocation-free.
__attribute__((no_sanitize("address", "undefined"))) NativeLookup native_lookup() noexcept {
    static const auto value =
        reinterpret_cast<NativeLookup>(::dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"));
    return value;
}
struct Symbol {
    std::string_view exported, queried;
    void* wrapper{};
};
// Symbol tables deliberately retain deprecated shared-bank and thread ABIs.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
const auto& symbols() noexcept {
#define SYMBOL(EXPORTED, QUERIED)                                  \
    Symbol {                                                       \
        #EXPORTED, (QUERIED), reinterpret_cast<void*>(&(EXPORTED)) \
    }
    static const std::array values{
        SYMBOL(cuInit, "cuInit"),
        SYMBOL(cuDeviceGet, "cuDeviceGet"),
        SYMBOL(cuDeviceGetCount, "cuDeviceGetCount"),
        SYMBOL(cuDeviceGetAttribute, "cuDeviceGetAttribute"),
        SYMBOL(cuDriverGetVersion, "cuDriverGetVersion"),
        SYMBOL(cuGetErrorName, "cuGetErrorName"),
        SYMBOL(cuGetErrorString, "cuGetErrorString"),
        SYMBOL(cuCtxCreate_v2, "cuCtxCreate"),
        SYMBOL(cuCtxCreate_v3, "cuCtxCreate"),
        SYMBOL(cuCtxCreate_v4, "cuCtxCreate"),
        SYMBOL(cuCtxDestroy_v2, "cuCtxDestroy"),
        SYMBOL(cuCtxGetCurrent, "cuCtxGetCurrent"),
        SYMBOL(cuCtxSetCurrent, "cuCtxSetCurrent"),
        SYMBOL(cuCtxPushCurrent_v2, "cuCtxPushCurrent"),
        SYMBOL(cuCtxPopCurrent_v2, "cuCtxPopCurrent"),
        SYMBOL(cuCtxSynchronize, "cuCtxSynchronize"),
        SYMBOL(cuCtxGetDevice, "cuCtxGetDevice"),
        SYMBOL(cuCtxGetFlags, "cuCtxGetFlags"),
        SYMBOL(cuCtxSetFlags, "cuCtxSetFlags"),
        SYMBOL(cuCtxGetLimit, "cuCtxGetLimit"),
        SYMBOL(cuCtxSetLimit, "cuCtxSetLimit"),
        SYMBOL(cuCtxGetCacheConfig, "cuCtxGetCacheConfig"),
        SYMBOL(cuCtxSetCacheConfig, "cuCtxSetCacheConfig"),
        SYMBOL(cuCtxGetSharedMemConfig, "cuCtxGetSharedMemConfig"),
        SYMBOL(cuCtxSetSharedMemConfig, "cuCtxSetSharedMemConfig"),
        SYMBOL(cuCtxResetPersistingL2Cache, "cuCtxResetPersistingL2Cache"),
        SYMBOL(cuCtxGetApiVersion, "cuCtxGetApiVersion"),
        SYMBOL(cuCtxGetExecAffinity, "cuCtxGetExecAffinity"),
        SYMBOL(cuCtxGetStreamPriorityRange, "cuCtxGetStreamPriorityRange"),
        SYMBOL(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain"),
        SYMBOL(cuDevicePrimaryCtxRelease_v2, "cuDevicePrimaryCtxRelease"),
        SYMBOL(cuDevicePrimaryCtxReset_v2, "cuDevicePrimaryCtxReset"),
        SYMBOL(cuDevicePrimaryCtxGetState, "cuDevicePrimaryCtxGetState"),
        SYMBOL(cuDevicePrimaryCtxSetFlags_v2, "cuDevicePrimaryCtxSetFlags"),
        SYMBOL(cuDevicePrimaryCtxRelease, "cuDevicePrimaryCtxRelease"),
        SYMBOL(cuDevicePrimaryCtxReset, "cuDevicePrimaryCtxReset"),
        SYMBOL(cuDevicePrimaryCtxSetFlags, "cuDevicePrimaryCtxSetFlags"),
        SYMBOL(cuModuleLoadData, "cuModuleLoadData"),
        SYMBOL(cuModuleLoadFatBinary, "cuModuleLoadFatBinary"),
        SYMBOL(cuModuleLoadDataEx, "cuModuleLoadDataEx"),
        SYMBOL(cuModuleGetFunction, "cuModuleGetFunction"),
        SYMBOL(cuFuncGetAttribute, "cuFuncGetAttribute"),
        SYMBOL(cuPointerGetAttribute, "cuPointerGetAttribute"),
        SYMBOL(cuFuncSetAttribute, "cuFuncSetAttribute"),
        SYMBOL(cuFuncSetCacheConfig, "cuFuncSetCacheConfig"),
        SYMBOL(cuLinkCreate_v2, "cuLinkCreate"),
        SYMBOL(cuLinkAddData_v2, "cuLinkAddData"),
        SYMBOL(cuLinkAddFile_v2, "cuLinkAddFile"),
        SYMBOL(cuLinkComplete, "cuLinkComplete"),
        SYMBOL(cuLinkDestroy, "cuLinkDestroy"),
        SYMBOL(cuModuleGetGlobal_v2, "cuModuleGetGlobal"),
        SYMBOL(cuModuleUnload, "cuModuleUnload"),
        SYMBOL(cuMemAlloc_v2, "cuMemAlloc"),
        SYMBOL(cuMemAllocAsync, "cuMemAllocAsync"),
        SYMBOL(cuMemAllocFromPoolAsync, "cuMemAllocFromPoolAsync"),
        SYMBOL(cuDeviceGetDefaultMemPool, "cuDeviceGetDefaultMemPool"),
        SYMBOL(cuDeviceGetMemPool, "cuDeviceGetMemPool"),
        SYMBOL(cuDeviceSetMemPool, "cuDeviceSetMemPool"),
        SYMBOL(cuMemFree_v2, "cuMemFree"),
        SYMBOL(cuMemHostAlloc, "cuMemHostAlloc"),
        SYMBOL(cuMemAllocHost_v2, "cuMemAllocHost"),
        SYMBOL(cuMemFreeHost, "cuMemFreeHost"),
        SYMBOL(cuMemHostGetFlags, "cuMemHostGetFlags"),
        SYMBOL(cuMemHostGetDevicePointer_v2, "cuMemHostGetDevicePointer"),
        SYMBOL(cuMemHostRegister_v2, "cuMemHostRegister"),
        SYMBOL(cuMemHostUnregister, "cuMemHostUnregister"),
        SYMBOL(cuMemcpy, "cuMemcpy"),
        SYMBOL(cuMemcpyAsync, "cuMemcpyAsync"),
        SYMBOL(cuMemcpy_ptds, "cuMemcpy"),
        SYMBOL(cuMemcpyAsync_ptsz, "cuMemcpyAsync"),
        SYMBOL(cuMemcpyHtoD_v2, "cuMemcpyHtoD"),
        SYMBOL(cuMemcpyDtoH_v2, "cuMemcpyDtoH"),
        SYMBOL(cuMemcpyHtoDAsync_v2, "cuMemcpyHtoDAsync"),
        SYMBOL(cuMemcpyDtoHAsync_v2, "cuMemcpyDtoHAsync"),
        SYMBOL(cuMemcpyDtoD_v2, "cuMemcpyDtoD"),
        SYMBOL(cuMemcpyDtoDAsync_v2, "cuMemcpyDtoDAsync"),
        SYMBOL(cuMemsetD8_v2, "cuMemsetD8"),
        SYMBOL(cuMemsetD8Async, "cuMemsetD8Async"),
        SYMBOL(cuMemsetD16_v2, "cuMemsetD16"),
        SYMBOL(cuMemsetD16Async, "cuMemsetD16Async"),
        SYMBOL(cuMemsetD32_v2, "cuMemsetD32"),
        SYMBOL(cuMemsetD32Async, "cuMemsetD32Async"),
        SYMBOL(cuStreamCreate, "cuStreamCreate"),
        SYMBOL(cuStreamCreateWithPriority, "cuStreamCreateWithPriority"),
        SYMBOL(cuStreamGetPriority, "cuStreamGetPriority"),
        SYMBOL(cuStreamGetFlags, "cuStreamGetFlags"),
        SYMBOL(cuStreamDestroy_v2, "cuStreamDestroy"),
        SYMBOL(cuStreamQuery, "cuStreamQuery"),
        SYMBOL(cuStreamSynchronize, "cuStreamSynchronize"),
        SYMBOL(cuEventCreate, "cuEventCreate"),
        SYMBOL(cuEventRecord, "cuEventRecord"),
        SYMBOL(cuEventRecordWithFlags, "cuEventRecordWithFlags"),
        SYMBOL(cuEventQuery, "cuEventQuery"),
        SYMBOL(cuEventSynchronize, "cuEventSynchronize"),
        SYMBOL(cuEventDestroy_v2, "cuEventDestroy"),
        SYMBOL(cuEventElapsedTime, "cuEventElapsedTime"),
        SYMBOL(cuStreamWaitEvent, "cuStreamWaitEvent"),
        SYMBOL(cuLaunchKernel, "cuLaunchKernel"),
        SYMBOL(cuLaunchCooperativeKernel, "cuLaunchCooperativeKernel"),
        SYMBOL(cuGraphLaunch, "cuGraphLaunch"),
        SYMBOL(cuGetProcAddress, "cuGetProcAddress"),
        SYMBOL(cuGetProcAddress_v2, "cuGetProcAddress")};
#undef SYMBOL
    return values;
}
const auto& runtime_symbols() noexcept {
    // cuRAND's SDK header includes Runtime C++ overloads. Bind the exported C ABI.
    using Allocate = cudaError_t(CUDARTAPI*)(void**, std::size_t);
    using HostAllocate = cudaError_t(CUDARTAPI*)(void**, std::size_t, unsigned int);
    using HostPointer = cudaError_t(CUDARTAPI*)(void**, void*, unsigned int);
    using CreateEvent = cudaError_t(CUDARTAPI*)(cudaEvent_t*);
    using Launch =
        cudaError_t(CUDARTAPI*)(const void*, dim3, dim3, void**, std::size_t, cudaStream_t);
    using FunctionAttributes = cudaError_t(CUDARTAPI*)(cudaFuncAttributes*, const void*);
    using FunctionAttribute = cudaError_t(CUDARTAPI*)(const void*, cudaFuncAttribute, int);
    using PointerAttributes = cudaError_t(CUDARTAPI*)(cudaPointerAttributes*, const void*);
#define RUNTIME_SYMBOL(NAME)                        \
    Symbol {                                        \
        #NAME, {}, reinterpret_cast<void*>(&(NAME)) \
    }
    static const std::array values{
        RUNTIME_SYMBOL(cudaGetDevice),
        RUNTIME_SYMBOL(cudaGetDeviceCount),
        RUNTIME_SYMBOL(cudaGetDeviceProperties_v2),
        RUNTIME_SYMBOL(cudaDeviceGetAttribute),
        RUNTIME_SYMBOL(cudaDeviceGetP2PAttribute),
        RUNTIME_SYMBOL(cudaDriverGetVersion),
        RUNTIME_SYMBOL(cudaDeviceCanAccessPeer),
        RUNTIME_SYMBOL(cudaDeviceGetStreamPriorityRange),
        RUNTIME_SYMBOL(cudaMemGetInfo),
        Symbol{"cudaPointerGetAttributes",
               {},
               reinterpret_cast<void*>(static_cast<PointerAttributes>(&cudaPointerGetAttributes))},
        RUNTIME_SYMBOL(cudaSetDevice),
        Symbol{"cudaMalloc", {}, reinterpret_cast<void*>(static_cast<Allocate>(&cudaMalloc))},
        RUNTIME_SYMBOL(cudaFree),
        RUNTIME_SYMBOL(cudaMemPoolTrimTo),
        Symbol{"cudaHostAlloc",
               {},
               reinterpret_cast<void*>(static_cast<HostAllocate>(&cudaHostAlloc))},
        Symbol{
            "cudaMallocHost", {}, reinterpret_cast<void*>(static_cast<Allocate>(&cudaMallocHost))},
        RUNTIME_SYMBOL(cudaFreeHost),
        RUNTIME_SYMBOL(cudaHostGetFlags),
        Symbol{"cudaHostGetDevicePointer",
               {},
               reinterpret_cast<void*>(static_cast<HostPointer>(&cudaHostGetDevicePointer))},
        RUNTIME_SYMBOL(cudaHostRegister),
        RUNTIME_SYMBOL(cudaHostUnregister),
        RUNTIME_SYMBOL(cudaMemcpy),
        RUNTIME_SYMBOL(cudaMemcpyAsync),
        RUNTIME_SYMBOL(cudaGetSymbolAddress),
        RUNTIME_SYMBOL(cudaGetSymbolSize),
        RUNTIME_SYMBOL(cudaMemcpyToSymbol),
        RUNTIME_SYMBOL(cudaMemcpyFromSymbol),
        RUNTIME_SYMBOL(cudaMemcpyToSymbolAsync),
        RUNTIME_SYMBOL(cudaMemcpyFromSymbolAsync),
        RUNTIME_SYMBOL(cudaMemset),
        RUNTIME_SYMBOL(cudaMemsetAsync),
        RUNTIME_SYMBOL(cudaStreamCreate),
        RUNTIME_SYMBOL(cudaStreamCreateWithFlags),
        RUNTIME_SYMBOL(cudaStreamCreateWithPriority),
        RUNTIME_SYMBOL(cudaStreamGetPriority),
        RUNTIME_SYMBOL(cudaStreamGetFlags),
        RUNTIME_SYMBOL(cudaStreamIsCapturing),
        RUNTIME_SYMBOL(cudaStreamBeginCapture),
        RUNTIME_SYMBOL(cudaStreamEndCapture),
        RUNTIME_SYMBOL(cudaGraphInstantiateWithFlags),
        RUNTIME_SYMBOL(cudaGraphDestroy),
        RUNTIME_SYMBOL(cudaGraphExecDestroy),
        RUNTIME_SYMBOL(cudaGraphLaunch),
        RUNTIME_SYMBOL(cudaStreamGetCaptureInfo_v2),
        RUNTIME_SYMBOL(cudaThreadExchangeStreamCaptureMode),
        RUNTIME_SYMBOL(cudaStreamDestroy),
        RUNTIME_SYMBOL(cudaStreamQuery),
        RUNTIME_SYMBOL(cudaStreamSynchronize),
        RUNTIME_SYMBOL(cudaLaunchHostFunc),
        RUNTIME_SYMBOL(cudaLaunchHostFunc_ptsz),
        Symbol{"cudaEventCreate",
               {},
               reinterpret_cast<void*>(static_cast<CreateEvent>(&cudaEventCreate))},
        RUNTIME_SYMBOL(cudaEventCreateWithFlags),
        RUNTIME_SYMBOL(cudaEventRecord),
        RUNTIME_SYMBOL(cudaEventRecordWithFlags),
        RUNTIME_SYMBOL(cudaEventQuery),
        RUNTIME_SYMBOL(cudaEventSynchronize),
        RUNTIME_SYMBOL(cudaEventDestroy),
        RUNTIME_SYMBOL(cudaEventElapsedTime),
        RUNTIME_SYMBOL(cudaStreamWaitEvent),
        RUNTIME_SYMBOL(cudaDeviceSynchronize),
        RUNTIME_SYMBOL(cudaDeviceReset),
        RUNTIME_SYMBOL(cudaThreadSynchronize),
        RUNTIME_SYMBOL(cudaThreadExit),
        Symbol{"cudaFuncGetAttributes",
               {},
               reinterpret_cast<void*>(static_cast<FunctionAttributes>(&cudaFuncGetAttributes))},
        Symbol{"cudaFuncSetAttribute",
               {},
               reinterpret_cast<void*>(static_cast<FunctionAttribute>(&cudaFuncSetAttribute))},
        RUNTIME_SYMBOL(cudaDeviceGetLimit),
        RUNTIME_SYMBOL(cudaDeviceSetLimit),
        RUNTIME_SYMBOL(cudaDeviceGetCacheConfig),
        RUNTIME_SYMBOL(cudaDeviceSetCacheConfig),
        RUNTIME_SYMBOL(cudaDeviceGetSharedMemConfig),
        RUNTIME_SYMBOL(cudaDeviceSetSharedMemConfig),
        RUNTIME_SYMBOL(cudaCtxResetPersistingL2Cache),
        RUNTIME_SYMBOL(cudaThreadGetLimit),
        RUNTIME_SYMBOL(cudaThreadSetLimit),
        RUNTIME_SYMBOL(cudaThreadGetCacheConfig),
        RUNTIME_SYMBOL(cudaThreadSetCacheConfig),
        Symbol{"cudaLaunchKernel",
               {},
               reinterpret_cast<void*>(static_cast<Launch>(&cudaLaunchKernel))},
        RUNTIME_SYMBOL(cudaGetLastError),
        RUNTIME_SYMBOL(cudaPeekAtLastError),
        RUNTIME_SYMBOL(cudaGetErrorName),
        RUNTIME_SYMBOL(cudaGetErrorString),
        RUNTIME_SYMBOL(cudaGetDriverEntryPoint),
        RUNTIME_SYMBOL(cudaGetDriverEntryPointByVersion),
        RUNTIME_SYMBOL(cudaGetDriverEntryPoint_ptsz),
        RUNTIME_SYMBOL(cudaGetDriverEntryPointByVersion_ptsz),
        RUNTIME_SYMBOL(cudaGetExportTable),
        RUNTIME_SYMBOL(__cudaRegisterFatBinary),
        RUNTIME_SYMBOL(__cudaRegisterFatBinaryEnd),
        RUNTIME_SYMBOL(__cudaUnregisterFatBinary),
        RUNTIME_SYMBOL(__cudaRegisterFunction),
        RUNTIME_SYMBOL(__cudaRegisterVar),
        RUNTIME_SYMBOL(__cudaRegisterManagedVar),
        RUNTIME_SYMBOL(__cudaRegisterTexture),
        RUNTIME_SYMBOL(__cudaRegisterSurface),
        RUNTIME_SYMBOL(__cudaInitModule)};
#undef RUNTIME_SYMBOL
    return values;
}
#pragma GCC diagnostic pop
const auto& blas_symbols() noexcept {
    using GemmEx = cublasStatus_t(CUBLASWINAPI*)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*,
        const void*, cudaDataType, int, const void*, cudaDataType, int, const void*, void*,
        cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
    using StridedGemmEx = cublasStatus_t(CUBLASWINAPI*)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*,
        const void*, cudaDataType, int, long long, const void*, cudaDataType, int, long long,
        const void*, void*, cudaDataType, int, long long, int, cublasComputeType_t,
        cublasGemmAlgo_t);
#define BLAS_SYMBOL(NAME)                              \
    Symbol {                                           \
        #NAME, #NAME, reinterpret_cast<void*>(&(NAME)) \
    }
    static const std::array values{
        BLAS_SYMBOL(cublasCreate_v2),
        BLAS_SYMBOL(cublasDestroy_v2),
        BLAS_SYMBOL(cublasSetStream_v2),
        BLAS_SYMBOL(cublasGetStream_v2),
        BLAS_SYMBOL(cublasSetPointerMode_v2),
        BLAS_SYMBOL(cublasGetPointerMode_v2),
        BLAS_SYMBOL(cublasSetMathMode),
        BLAS_SYMBOL(cublasGetMathMode),
        BLAS_SYMBOL(cublasGetVersion_v2),
        BLAS_SYMBOL(cublasSetWorkspace_v2),
        BLAS_SYMBOL(cublasSetAtomicsMode),
        BLAS_SYMBOL(cublasGetAtomicsMode),
        BLAS_SYMBOL(cublasSetSmCountTarget),
        BLAS_SYMBOL(cublasGetSmCountTarget),
        BLAS_SYMBOL(cublasSgemm_v2),
        BLAS_SYMBOL(cublasSgemv_v2),
        BLAS_SYMBOL(cublasSgemmEx),
        BLAS_SYMBOL(cublasSgeam),
        BLAS_SYMBOL(cublasDgeam),
        Symbol{"cublasGemmEx", "cublasGemmEx",
               reinterpret_cast<void*>(static_cast<GemmEx>(&cublasGemmEx))},
        Symbol{"cublasSgemmStridedBatched", "cublasSgemmStridedBatched",
               reinterpret_cast<void*>(&cublasSgemmStridedBatched)},
        Symbol{"cublasGemmStridedBatchedEx", "cublasGemmStridedBatchedEx",
               reinterpret_cast<void*>(static_cast<StridedGemmEx>(&cublasGemmStridedBatchedEx))}};
#undef BLAS_SYMBOL
    return values;
}
const auto& lt_symbols() noexcept {
#define LT_SYMBOL(NAME)                                \
    Symbol {                                           \
        #NAME, #NAME, reinterpret_cast<void*>(&(NAME)) \
    }
    static const std::array values{LT_SYMBOL(cublasLtMatrixLayoutCreate),
                                   LT_SYMBOL(cublasLtMatrixLayoutDestroy),
                                   LT_SYMBOL(cublasLtMatrixLayoutSetAttribute),
                                   LT_SYMBOL(cublasLtMatrixLayoutGetAttribute),
                                   LT_SYMBOL(cublasLtMatmulDescCreate),
                                   LT_SYMBOL(cublasLtMatmulDescDestroy),
                                   LT_SYMBOL(cublasLtMatmulDescSetAttribute),
                                   LT_SYMBOL(cublasLtMatmulDescGetAttribute),
                                   LT_SYMBOL(cublasLtMatmulPreferenceCreate),
                                   LT_SYMBOL(cublasLtMatmulPreferenceDestroy),
                                   LT_SYMBOL(cublasLtMatmulPreferenceSetAttribute),
                                   LT_SYMBOL(cublasLtMatmulPreferenceGetAttribute),
                                   LT_SYMBOL(cublasLtGetVersion),
                                   LT_SYMBOL(cublasLtGetCudartVersion),
                                   LT_SYMBOL(cublasLtMatrixLayoutInit_internal),
                                   LT_SYMBOL(cublasLtMatmulDescInit_internal),
                                   LT_SYMBOL(cublasLtMatmulPreferenceInit_internal),
                                   LT_SYMBOL(cublasLtMatmul),
                                   LT_SYMBOL(cublasLtCreate),
                                   LT_SYMBOL(cublasLtDestroy),
                                   LT_SYMBOL(cublasLtMatmulAlgoGetHeuristic),
                                   LT_SYMBOL(cublasLtMatmulAlgoCheck),
                                   LT_SYMBOL(cublasLtMatmulAlgoCapGetAttribute),
                                   LT_SYMBOL(cublasLtMatrixTransform)};
#undef LT_SYMBOL
    return values;
}
const auto& random_symbols() noexcept {
#define RANDOM_SYMBOL(NAME)                         \
    Symbol {                                        \
        #NAME, {}, reinterpret_cast<void*>(&(NAME)) \
    }
    static const std::array values{RANDOM_SYMBOL(curandCreateGenerator),
                                   RANDOM_SYMBOL(curandSetPseudoRandomGeneratorSeed),
                                   RANDOM_SYMBOL(curandDestroyGenerator),
                                   RANDOM_SYMBOL(curandCreateGeneratorHost),
                                   RANDOM_SYMBOL(curandSetGeneratorOffset),
                                   RANDOM_SYMBOL(curandSetGeneratorOrdering),
                                   RANDOM_SYMBOL(curandSetQuasiRandomGeneratorDimensions),
                                   RANDOM_SYMBOL(curandSetStream),
                                   RANDOM_SYMBOL(curandGenerateSeeds),
                                   RANDOM_SYMBOL(curandGenerate),
                                   RANDOM_SYMBOL(curandGenerateLongLong),
                                   RANDOM_SYMBOL(curandGenerateUniform),
                                   RANDOM_SYMBOL(curandGenerateUniformDouble),
                                   RANDOM_SYMBOL(curandGenerateNormal),
                                   RANDOM_SYMBOL(curandGenerateNormalDouble),
                                   RANDOM_SYMBOL(curandGenerateLogNormal),
                                   RANDOM_SYMBOL(curandGenerateLogNormalDouble),
                                   RANDOM_SYMBOL(curandGeneratePoisson),
                                   RANDOM_SYMBOL(curandGeneratePoissonMethod),
                                   RANDOM_SYMBOL(curandGenerateBinomial),
                                   RANDOM_SYMBOL(curandGenerateBinomialMethod)};
#undef RANDOM_SYMBOL
    return values;
}
#ifdef GLIMMER_HAS_CUDNN
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
const auto& dnn_symbols() noexcept {
#define DNN_SYMBOL(NAME)                            \
    Symbol {                                        \
        #NAME, {}, reinterpret_cast<void*>(&(NAME)) \
    }
    static const std::array values{
        DNN_SYMBOL(cudnnCreate),
        DNN_SYMBOL(cudnnBackendCreateDescriptor),
        DNN_SYMBOL(cudnnBackendDestroyDescriptor),
        DNN_SYMBOL(cudnnBackendInitialize),
        DNN_SYMBOL(cudnnBackendFinalize),
        DNN_SYMBOL(cudnnBackendSetAttribute),
        DNN_SYMBOL(cudnnBackendGetAttribute),
        DNN_SYMBOL(cudnnBackendExecute),
#if CUDNN_MAJOR == 9
        DNN_SYMBOL(cudnnBackendPopulateCudaGraph),
        DNN_SYMBOL(cudnnBackendUpdateCudaGraph),
#endif
#if CUDNN_MAJOR == 8
        DNN_SYMBOL(cudnnOpsInferVersionCheck),
        DNN_SYMBOL(cudnnOpsTrainVersionCheck),
        DNN_SYMBOL(cudnnCnnInferVersionCheck),
        DNN_SYMBOL(cudnnCnnTrainVersionCheck),
        DNN_SYMBOL(cudnnAdvInferVersionCheck),
        DNN_SYMBOL(cudnnAdvTrainVersionCheck),
        DNN_SYMBOL(cudnnCreateAlgorithmDescriptor),
        DNN_SYMBOL(cudnnSetAlgorithmDescriptor),
        DNN_SYMBOL(cudnnDestroyAlgorithmDescriptor),
        DNN_SYMBOL(cudnnSetRNNDescriptor_v6),
        DNN_SYMBOL(cudnnSetRNNMatrixMathType),
        DNN_SYMBOL(cudnnGetRNNMatrixMathType),
        DNN_SYMBOL(cudnnSetRNNBiasMode),
        DNN_SYMBOL(cudnnGetRNNBiasMode),
        DNN_SYMBOL(cudnnRNNSetClip),
        DNN_SYMBOL(cudnnGetRNNWorkspaceSize),
        DNN_SYMBOL(cudnnGetRNNTrainingReserveSize),
        DNN_SYMBOL(cudnnGetRNNParamsSize),
        DNN_SYMBOL(cudnnGetRNNLinLayerMatrixParams),
        DNN_SYMBOL(cudnnGetRNNLinLayerBiasParams),
        DNN_SYMBOL(cudnnSetRNNAlgorithmDescriptor),
        DNN_SYMBOL(cudnnGetRNNForwardInferenceAlgorithmMaxCount),
        DNN_SYMBOL(cudnnRNNForwardTraining),
        DNN_SYMBOL(cudnnRNNBackwardData),
        DNN_SYMBOL(cudnnRNNBackwardWeights),
#endif
        DNN_SYMBOL(cudnnCreateRNNDataDescriptor),
        DNN_SYMBOL(cudnnDestroyRNNDataDescriptor),
        DNN_SYMBOL(cudnnSetRNNDataDescriptor),
        DNN_SYMBOL(cudnnGetRNNDataDescriptor),
        DNN_SYMBOL(cudnnCreateSeqDataDescriptor),
        DNN_SYMBOL(cudnnDestroySeqDataDescriptor),
        DNN_SYMBOL(cudnnSetSeqDataDescriptor),
        DNN_SYMBOL(cudnnGetSeqDataDescriptor),
        DNN_SYMBOL(cudnnCreateRNNDescriptor),
        DNN_SYMBOL(cudnnDestroyRNNDescriptor),
        DNN_SYMBOL(cudnnSetRNNDescriptor_v8),
        DNN_SYMBOL(cudnnGetRNNDescriptor_v8),
        DNN_SYMBOL(cudnnCreateAttnDescriptor),
        DNN_SYMBOL(cudnnDestroyAttnDescriptor),
        DNN_SYMBOL(cudnnSetAttnDescriptor),
        DNN_SYMBOL(cudnnGetAttnDescriptor),
        DNN_SYMBOL(cudnnGetRNNWeightSpaceSize),
        DNN_SYMBOL(cudnnGetRNNTempSpaceSizes),
        DNN_SYMBOL(cudnnGetMultiHeadAttnBuffers),
        DNN_SYMBOL(cudnnRNNSetClip_v8),
        DNN_SYMBOL(cudnnRNNGetClip_v8),
        DNN_SYMBOL(cudnnRNNForward),
        DNN_SYMBOL(cudnnRNNBackwardData_v8),
        DNN_SYMBOL(cudnnRNNBackwardWeights_v8),
        DNN_SYMBOL(cudnnBuildRNNDynamic),
        DNN_SYMBOL(cudnnGetRNNWeightParams),
        DNN_SYMBOL(cudnnGetMultiHeadAttnWeights),
        DNN_SYMBOL(cudnnMultiHeadAttnForward),
        DNN_SYMBOL(cudnnMultiHeadAttnBackwardData),
        DNN_SYMBOL(cudnnMultiHeadAttnBackwardWeights),
        DNN_SYMBOL(cudnnDestroy),
        DNN_SYMBOL(cudnnSetStream),
        DNN_SYMBOL(cudnnGetStream),
        DNN_SYMBOL(cudnnCreateTensorDescriptor),
        DNN_SYMBOL(cudnnDestroyTensorDescriptor),
        DNN_SYMBOL(cudnnSetTensor4dDescriptor),
        DNN_SYMBOL(cudnnSetTensor4dDescriptorEx),
        DNN_SYMBOL(cudnnGetTensor4dDescriptor),
        DNN_SYMBOL(cudnnSetTensorNdDescriptor),
        DNN_SYMBOL(cudnnSetTensorNdDescriptorEx),
        DNN_SYMBOL(cudnnGetTensorNdDescriptor),
        DNN_SYMBOL(cudnnGetTensorSizeInBytes),
        DNN_SYMBOL(cudnnCreateActivationDescriptor),
        DNN_SYMBOL(cudnnDestroyActivationDescriptor),
        DNN_SYMBOL(cudnnSetActivationDescriptor),
        DNN_SYMBOL(cudnnGetActivationDescriptor),
        DNN_SYMBOL(cudnnActivationForward),
        DNN_SYMBOL(cudnnAddTensor),
        DNN_SYMBOL(cudnnCreateFilterDescriptor),
        DNN_SYMBOL(cudnnDestroyFilterDescriptor),
        DNN_SYMBOL(cudnnSetFilter4dDescriptor),
        DNN_SYMBOL(cudnnGetFilter4dDescriptor),
        DNN_SYMBOL(cudnnSetFilterNdDescriptor),
        DNN_SYMBOL(cudnnGetFilterNdDescriptor),
        DNN_SYMBOL(cudnnGetFilterSizeInBytes),
        DNN_SYMBOL(cudnnCreateConvolutionDescriptor),
        DNN_SYMBOL(cudnnDestroyConvolutionDescriptor),
        DNN_SYMBOL(cudnnSetConvolution2dDescriptor),
        DNN_SYMBOL(cudnnGetConvolution2dDescriptor),
        DNN_SYMBOL(cudnnSetConvolutionNdDescriptor),
        DNN_SYMBOL(cudnnGetConvolutionNdDescriptor),
        DNN_SYMBOL(cudnnSetConvolutionGroupCount),
        DNN_SYMBOL(cudnnGetConvolutionGroupCount),
        DNN_SYMBOL(cudnnSetConvolutionMathType),
        DNN_SYMBOL(cudnnGetConvolutionMathType),
        DNN_SYMBOL(cudnnSetConvolutionReorderType),
        DNN_SYMBOL(cudnnGetConvolutionReorderType),
        DNN_SYMBOL(cudnnGetConvolution2dForwardOutputDim),
        DNN_SYMBOL(cudnnGetConvolutionNdForwardOutputDim),
        DNN_SYMBOL(cudnnGetConvolutionForwardWorkspaceSize),
        DNN_SYMBOL(cudnnGetConvolutionBackwardDataAlgorithmMaxCount),
        DNN_SYMBOL(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount),
        DNN_SYMBOL(cudnnGetConvolutionBackwardDataWorkspaceSize),
        DNN_SYMBOL(cudnnGetFoldedConvBackwardDataDescriptors),
        DNN_SYMBOL(cudnnGetConvolutionForwardAlgorithmMaxCount),
        DNN_SYMBOL(cudnnGetConvolutionForwardAlgorithm_v7),
        DNN_SYMBOL(cudnnConvolutionForward),
        DNN_SYMBOL(cudnnFindConvolutionForwardAlgorithm),
        DNN_SYMBOL(cudnnFindConvolutionForwardAlgorithmEx),
        DNN_SYMBOL(cudnnCreatePoolingDescriptor),
        DNN_SYMBOL(cudnnDestroyPoolingDescriptor),
        DNN_SYMBOL(cudnnSetPoolingNdDescriptor),
        DNN_SYMBOL(cudnnGetPoolingNdDescriptor),
        DNN_SYMBOL(cudnnGetPoolingNdForwardOutputDim),
        DNN_SYMBOL(cudnnPoolingForward),
        DNN_SYMBOL(cudnnSoftmaxForward),
        DNN_SYMBOL(cudnnSetPooling2dDescriptor),
        DNN_SYMBOL(cudnnGetPooling2dDescriptor),
        DNN_SYMBOL(cudnnGetPooling2dForwardOutputDim),
        DNN_SYMBOL(cudnnPoolingBackward),
        DNN_SYMBOL(cudnnSoftmaxBackward),
        DNN_SYMBOL(cudnnCreateLRNDescriptor),
        DNN_SYMBOL(cudnnSetLRNDescriptor),
        DNN_SYMBOL(cudnnDestroyLRNDescriptor),
        DNN_SYMBOL(cudnnLRNCrossChannelForward),
        DNN_SYMBOL(cudnnGetLRNDescriptor),
        DNN_SYMBOL(cudnnLRNCrossChannelBackward),
        DNN_SYMBOL(cudnnDivisiveNormalizationForward),
        DNN_SYMBOL(cudnnDivisiveNormalizationBackward),
        DNN_SYMBOL(cudnnCreateReduceTensorDescriptor),
        DNN_SYMBOL(cudnnSetReduceTensorDescriptor),
        DNN_SYMBOL(cudnnDestroyReduceTensorDescriptor),
        DNN_SYMBOL(cudnnGetReductionIndicesSize),
        DNN_SYMBOL(cudnnGetReductionWorkspaceSize),
        DNN_SYMBOL(cudnnReduceTensor),
        DNN_SYMBOL(cudnnGetReduceTensorDescriptor),
        DNN_SYMBOL(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize),
        DNN_SYMBOL(cudnnGetBatchNormalizationBackwardExWorkspaceSize),
        DNN_SYMBOL(cudnnGetBatchNormalizationTrainingExReserveSpaceSize),
        DNN_SYMBOL(cudnnBatchNormalizationForwardTrainingEx),
        DNN_SYMBOL(cudnnBatchNormalizationBackwardEx),
        DNN_SYMBOL(cudnnBatchNormalizationForwardTraining),
        DNN_SYMBOL(cudnnBatchNormalizationForwardInference),
        DNN_SYMBOL(cudnnBatchNormalizationBackward),
        DNN_SYMBOL(cudnnDeriveBNTensorDescriptor),
        DNN_SYMBOL(cudnnCreateDropoutDescriptor),
        DNN_SYMBOL(cudnnDestroyDropoutDescriptor),
        DNN_SYMBOL(cudnnDropoutGetStatesSize),
        DNN_SYMBOL(cudnnSetDropoutDescriptor),
        DNN_SYMBOL(cudnnRestoreDropoutDescriptor),
        DNN_SYMBOL(cudnnDropoutGetReserveSpaceSize),
        DNN_SYMBOL(cudnnGetDropoutDescriptor),
        DNN_SYMBOL(cudnnDropoutForward),
        DNN_SYMBOL(cudnnDropoutBackward),
        DNN_SYMBOL(cudnnGetVersion),
        DNN_SYMBOL(cudnnGetCudartVersion),
        DNN_SYMBOL(cudnnGetMaxDeviceVersion),
        DNN_SYMBOL(cudnnGetProperty),
        DNN_SYMBOL(cudnnCreateOpTensorDescriptor),
        DNN_SYMBOL(cudnnSetOpTensorDescriptor),
        DNN_SYMBOL(cudnnGetOpTensorDescriptor),
        DNN_SYMBOL(cudnnDestroyOpTensorDescriptor),
        DNN_SYMBOL(cudnnCreateTensorTransformDescriptor),
        DNN_SYMBOL(cudnnSetTensorTransformDescriptor),
        DNN_SYMBOL(cudnnDestroyTensorTransformDescriptor),
        DNN_SYMBOL(cudnnInitTransformDest),
        DNN_SYMBOL(cudnnReorderFilterAndBias),
        DNN_SYMBOL(cudnnGetTensorTransformDescriptor),
        DNN_SYMBOL(cudnnOpTensor),
        DNN_SYMBOL(cudnnSetActivationDescriptorSwishBeta),
        DNN_SYMBOL(cudnnGetActivationDescriptorSwishBeta),
        DNN_SYMBOL(cudnnSetTensor),
        DNN_SYMBOL(cudnnScaleTensor),
        DNN_SYMBOL(cudnnTransformTensor),
        DNN_SYMBOL(cudnnTransformTensorEx),
        DNN_SYMBOL(cudnnActivationBackward)};
#undef DNN_SYMBOL
    return values;
}
#pragma GCC diagnostic pop
#endif
#ifdef GLIMMER_HAS_NCCL
const auto& nccl_symbols() noexcept {
#define NCCL_SYMBOL(NAME)                           \
    Symbol {                                        \
        #NAME, {}, reinterpret_cast<void*>(&(NAME)) \
    }
    static const std::array values{NCCL_SYMBOL(ncclGetUniqueId),
                                   NCCL_SYMBOL(ncclCommInitRank),
                                   NCCL_SYMBOL(ncclCommInitRankConfig),
                                   NCCL_SYMBOL(ncclCommGetAsyncError),
                                   NCCL_SYMBOL(ncclCommDestroy),
                                   NCCL_SYMBOL(ncclCommAbort),
                                   NCCL_SYMBOL(ncclGroupStart),
                                   NCCL_SYMBOL(ncclGroupEnd),
                                   NCCL_SYMBOL(ncclAllGather),
                                   NCCL_SYMBOL(ncclAllReduce),
                                   NCCL_SYMBOL(ncclBcast)};
#undef NCCL_SYMBOL
    return values;
}
#endif
enum class SdkLibrary { kOther, kDriver, kRuntime, kBlas, kLt, kRandom, kDnn, kNccl };
SdkLibrary library_kind(const char* filename) noexcept {
    if (!filename) return SdkLibrary::kOther;
    const auto* separator = std::strrchr(filename, '/');
    const std::string_view name{separator ? separator + 1 : filename};
    if (name.starts_with("libcuda.so")) return SdkLibrary::kDriver;
    if (name.starts_with("libcudart.so")) return SdkLibrary::kRuntime;
    if (name.starts_with("libcublas.so")) return SdkLibrary::kBlas;
    if (name.starts_with("libcublasLt.so")) return SdkLibrary::kLt;
    if (name.starts_with("libcurand.so")) return SdkLibrary::kRandom;
#ifdef GLIMMER_HAS_NCCL
    if (name.starts_with("libnccl.so")) return SdkLibrary::kNccl;
#endif
#ifdef GLIMMER_HAS_CUDNN
    if (name.starts_with("libcudnn.so") ||
        (name.starts_with("libcudnn_") && name.find(".so") != std::string_view::npos))
        return SdkLibrary::kDnn;
#endif
    return SdkLibrary::kOther;
}
#ifdef GLIMMER_HAS_CUDNN
std::atomic<bool> dnn_lookup_is_uncertain{};
#endif
bool driver_image(const void* address) noexcept {
    Dl_info info{};
    if (!address || !::dladdr(address, &info) || !info.dli_fname) return false;
    return library_kind(info.dli_fname) == SdkLibrary::kDriver;
}
bool runtime_image(const void* address) noexcept {
    Dl_info info{};
    return address && ::dladdr(address, &info) &&
           library_kind(info.dli_fname) == SdkLibrary::kRuntime;
}
SdkLibrary sdk_handle(void* handle, const char* requested) noexcept {
    if (handle == RTLD_DEFAULT || handle == RTLD_NEXT) return SdkLibrary::kOther;
    void* storage{};
    if (::dlinfo(handle, RTLD_DI_LINKMAP, &storage) != 0 || !storage) {
#ifdef GLIMMER_HAS_CUDNN
        if (std::string_view{requested}.starts_with("cudnn")) {
            dnn_lookup_is_uncertain.store(true, std::memory_order_release);
            return SdkLibrary::kDnn;
        }
#endif
        return SdkLibrary::kOther;
    }
    const auto* map = static_cast<const struct link_map*>(storage);
#ifdef GLIMMER_HAS_CUDNN
    if (std::string_view{requested}.starts_with("cudnn")) {
        // Equal spelling cannot authorize a native consumer of remote handles.
        // A renamed or other-namespace provider stays intercepted and poisons
        // the profile unless its loaded identity is independently proved.
        Lmid_t namespace_id{};
        if (::dlinfo(handle, RTLD_DI_LMID, &namespace_id) != 0 || namespace_id != LM_ID_BASE ||
            !glimmer::internal::cudnn_profile::provider_base_matches(
                static_cast<std::uintptr_t>(map->l_addr)))
            dnn_lookup_is_uncertain.store(true, std::memory_order_release);
        return SdkLibrary::kDnn;
    }
#else
    (void)requested;
#endif
    return library_kind(map->l_name);
}
constexpr std::size_t kMaxCachedSdkSymbols = 128;
struct CachedSdkProvider {
    void* library{};
    std::array<void*, kMaxCachedSdkSymbols> originals{};
    CachedSdkProvider* retained_next{};
    int runtime_version{};
};
static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(symbols())>> <= kMaxCachedSdkSymbols);
static_assert(std::tuple_size_v<std::remove_cvref_t<decltype(runtime_symbols())>> <=
              kMaxCachedSdkSymbols);
struct ProcessSdkAnchor {
    // Cached native function pointers require the matching SDK to remain
    // loaded. These two bounded loader references intentionally belong to the
    // process lifetime, including RTLD_LOCAL providers; no GPU resource lives
    // here. A failed duplicate-reference release stays marked uncertain.
    std::atomic<CachedSdkProvider*> provider{};
    std::atomic<CachedSdkProvider*> uncertain_references{};
    std::atomic<bool> is_uncertain{};
};
ProcessSdkAnchor& sdk_anchor(SdkLibrary kind) noexcept {
    static ProcessSdkAnchor driver, runtime;
    return kind == SdkLibrary::kDriver ? driver : runtime;
}
bool anchor_sdk(void* handle, SdkLibrary kind, NativeLookup lookup) noexcept {
    auto& anchor = sdk_anchor(kind);
    if (anchor.is_uncertain.load(std::memory_order_acquire)) return false;
    if (auto* established = anchor.provider.load(std::memory_order_acquire))
        return established->library == handle;
    void* storage{};
    if (::dlinfo(handle, RTLD_DI_LINKMAP, &storage) != 0 || !storage) return false;
    const auto* map = static_cast<const struct link_map*>(storage);
    if (!map->l_name) return false;
    int runtime_version{};
    if (kind == SdkLibrary::kRuntime) {
        const auto version_query = reinterpret_cast<decltype(&cudaRuntimeGetVersion)>(
            lookup(handle, "cudaRuntimeGetVersion"));
        if (!version_query || version_query(&runtime_version) != cudaSuccess ||
            runtime_version != CUDART_VERSION)
            return false;
        auto* selected = lookup(handle, "cudaGetDriverEntryPoint");
        auto* global = lookup(RTLD_NEXT, "cudaGetDriverEntryPoint");
        if (global && selected != global) return false;
    }
    auto candidate = std::unique_ptr<CachedSdkProvider>(new (std::nothrow) CachedSdkProvider);
    if (!candidate) return false;
    candidate->runtime_version = runtime_version;
    // Build the complete immutable snapshot before publication. No lock spans
    // loader work and another thread can never consume a half-filled table.
    std::size_t index{};
    if (kind == SdkLibrary::kDriver) {
        for (const auto& entry : symbols())
            candidate->originals[index++] = lookup(handle, entry.exported.data());
    } else {
        for (const auto& entry : runtime_symbols())
            candidate->originals[index++] = lookup(handle, entry.exported.data());
    }
    candidate->library = ::dlopen(map->l_name, RTLD_NOW | RTLD_NOLOAD);
    if (!candidate->library) return false;
    CachedSdkProvider* expected{};
    if (!anchor.provider.compare_exchange_strong(expected, candidate.get(),
                                                 std::memory_order_acq_rel)) {
        if (::dlclose(candidate->library) != 0) {
            anchor.is_uncertain.store(true, std::memory_order_release);
            auto* retained = candidate.release();
            auto* head = anchor.uncertain_references.load(std::memory_order_acquire);
            do {
                retained->retained_next = head;
            } while (!anchor.uncertain_references.compare_exchange_weak(head, retained,
                                                                        std::memory_order_acq_rel));
            return false;
        }
        return expected->library == handle;
    }
    (void)candidate.release();  // Owned by the bounded process-lifetime anchor.
    return true;
}
void* controlled_driver_entry(std::string_view name, void* address) noexcept {
    for (const auto& symbol : symbols())
        if (symbol.queried == name && address == native_driver_symbol(symbol.exported.data()))
            return symbol.wrapper;
    return nullptr;
}
}  // namespace

bool dnn_application_profile_matches() noexcept {
#ifdef GLIMMER_HAS_CUDNN
    if (::getpid() != kLookupProcess || dnn_lookup_is_uncertain.load(std::memory_order_acquire))
        return false;
    const bool is_matched = glimmer::internal::cudnn_profile::process_providers_match();
    return is_matched && !dnn_lookup_is_uncertain.load(std::memory_order_acquire);
#else
    return false;
#endif
}

void* native_driver_symbol(const char* name) noexcept {
    if (const auto* provider =
            sdk_anchor(SdkLibrary::kDriver).provider.load(std::memory_order_acquire)) {
        std::size_t index{};
        for (const auto& entry : symbols()) {
            if (entry.exported == name) return provider->originals[index];
            ++index;
        }
    }
    if (const auto* provider =
            sdk_anchor(SdkLibrary::kRuntime).provider.load(std::memory_order_acquire)) {
        std::size_t index{};
        for (const auto& entry : runtime_symbols()) {
            if (entry.exported == name) return provider->originals[index];
            ++index;
        }
    }
    const auto lookup = native_lookup();
    return lookup ? lookup(RTLD_NEXT, name) : nullptr;
}
CUresult initialize_driver_metadata() noexcept {
    if (::getpid() != kLookupProcess) return CUDA_ERROR_INVALID_CONTEXT;
    if (!native_driver_symbol("cuInit")) {
        const auto lookup = native_lookup();
        auto* driver = lookup ? ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL) : nullptr;
        if (!driver) return CUDA_ERROR_NOT_INITIALIZED;
        const bool anchored = anchor_sdk(driver, SdkLibrary::kDriver, lookup);
        const bool closed = ::dlclose(driver) == 0;
        if (!anchored || !closed) {
            sdk_anchor(SdkLibrary::kDriver).is_uncertain.store(true, std::memory_order_release);
            return CUDA_ERROR_UNKNOWN;
        }
    }
    const auto initialize = reinterpret_cast<decltype(&cuInit)>(native_driver_symbol("cuInit"));
    return initialize ? initialize(0) : CUDA_ERROR_NOT_INITIALIZED;
}
DriverEntryResult driver_entry(const DriverEntryQuery& query) noexcept {
    DriverEntryResult result;
    result.symbol_status = query.symbol_status;
    // Optional query status is an output, not an initialized enum input. Keep
    // unknown caller bytes without evaluating an out-of-range enum in C++.
    CUdriverProcAddressQueryResult native_status{};
    static_assert(sizeof(native_status) == sizeof(result.symbol_status));
    std::memcpy(&native_status, &result.symbol_status, sizeof(native_status));
    if (query.is_v2) {
        const auto native = reinterpret_cast<decltype(&cuGetProcAddress_v2)>(
            native_driver_symbol("cuGetProcAddress_v2"));
        if (!native) return result;
        result.status = native(query.symbol.data(), &result.address, query.version, query.flags,
                               query.has_symbol_status ? &native_status : nullptr);
    } else {
        const auto native =
            reinterpret_cast<decltype(&cuGetProcAddress)>(native_driver_symbol("cuGetProcAddress"));
        if (!native) return result;
        result.status = native(query.symbol.data(), &result.address, query.version, query.flags);
    }
    std::memcpy(&result.symbol_status, &native_status, sizeof(native_status));
    if (result.status != CUDA_SUCCESS) {
        result.address = nullptr;  // Never expose a native resource entry, even on SDK failure.
        return result;
    }
    if (!result.address) return result;  // SDK missing/version status is authoritative.
    if (query.is_v2 && query.has_symbol_status &&
        result.symbol_status != CU_GET_PROC_ADDRESS_SUCCESS) {
        result.status = CUDA_ERROR_UNKNOWN;
        result.address = nullptr;
        return result;
    }
    if (auto* address = controlled_driver_entry(query.symbol, result.address)) {
        result.address = address;
        return result;
    }
    result.status = CUDA_ERROR_NOT_SUPPORTED;
    result.address = nullptr;
    result.symbol_status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return result;
}

RuntimeEntryResult runtime_entry(const RuntimeEntryQuery& query) noexcept {
    RuntimeEntryResult result;
    result.symbol_status = query.symbol_status;
    if (::getpid() != kLookupProcess) return result;
    int runtime_version{};
    if (const auto* provider =
            sdk_anchor(SdkLibrary::kRuntime).provider.load(std::memory_order_acquire))
        runtime_version = provider->runtime_version;
    else {
        const auto version_query = reinterpret_cast<decltype(&cudaRuntimeGetVersion)>(
            native_driver_symbol("cudaRuntimeGetVersion"));
        if (!version_query) return result;
        result.status = version_query(&runtime_version);
        if (result.status != cudaSuccess) return result;
    }
    if (runtime_version != CUDART_VERSION) {
        result.status = cudaErrorNotSupported;
        return result;
    }
    if (!native_driver_symbol("cuGetProcAddress_v2")) {
        const auto lookup = native_lookup();
        void* driver = lookup ? ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL) : nullptr;
        if (!driver) {
            result.status = cudaErrorInsufficientDriver;
            return result;
        }
        const auto anchored = anchor_sdk(driver, SdkLibrary::kDriver, lookup);
        const auto closed = ::dlclose(driver) == 0;
        if (!anchored || !closed) {
            sdk_anchor(SdkLibrary::kDriver).is_uncertain.store(true, std::memory_order_release);
            result.status = cudaErrorUnknown;
            return result;
        }
    }
    const auto initialize = reinterpret_cast<decltype(&cuInit)>(native_driver_symbol("cuInit"));
    if (!initialize) {
        result.status = cudaErrorInsufficientDriver;
        return result;
    }
    const auto initialized = initialize(0);
    DriverEntryResult selected;
    if (initialized == CUDA_SUCCESS) {
        DriverEntryQuery request{
            query.symbol,
            runtime_version,
            query.is_ptds && !query.flags
                ? static_cast<cuuint64_t>(CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM)
                : query.flags,
            true,
            query.has_symbol_status,
            query.symbol_status};
        // Preserve unsigned Runtime version bits across the signed Driver ABI.
        static_assert(sizeof(request.version) == sizeof(query.version));
        if (query.is_versioned)
            std::memcpy(&request.version, &query.version, sizeof(request.version));
        selected = driver_entry(request);
    } else {
        selected.status = initialized;
        selected.symbol_status = query.symbol_status;
    }
    switch (selected.status) {
        case CUDA_SUCCESS:
            result.status = cudaSuccess;
            break;
        case CUDA_ERROR_INVALID_VALUE:
            result.status = cudaErrorInvalidValue;
            break;
        case CUDA_ERROR_NOT_INITIALIZED:
            result.status = cudaErrorInitializationError;
            break;
        case CUDA_ERROR_DEINITIALIZED:
            result.status = cudaErrorCudartUnloading;
            break;
        case CUDA_ERROR_NO_DEVICE:
            result.status = cudaErrorNoDevice;
            break;
        case CUDA_ERROR_NOT_SUPPORTED:
            result.status = cudaErrorNotSupported;
            break;
        case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH:
            result.status = cudaErrorSystemDriverMismatch;
            break;
        default:
            result.status = cudaErrorUnknown;
            break;
    }
    result.address = selected.address;
    result.symbol_status = selected.symbol_status;
    return result;
}

// Selected CUDA names take the instrumented cold path only after the loader
// bootstrap is usable. Native SDK-internal lookups keep their original scope.
void* routed_lookup(void* handle, const char* name, void* caller, NativeLookup& forward) noexcept {
    const auto lookup = native_lookup();
    if (!lookup) return nullptr;
    if (driver_image(caller)) {
        forward = lookup;
        return nullptr;
    }
    const auto library = sdk_handle(handle, name);
    if ((handle != RTLD_DEFAULT && library == SdkLibrary::kOther) ||
        (library == SdkLibrary::kRuntime && runtime_image(caller))) {
        forward = lookup;
        return nullptr;
    }
    // cuBLAS wrappers retain no code pointers into the caller's provider: all
    // native vendor calls belong to the service's independently loaded SDK.
    if ((library == SdkLibrary::kDriver || library == SdkLibrary::kRuntime) &&
        !anchor_sdk(handle, library, lookup)) {
        (void)lookup(handle, "__glimmer_execution_sdk_provider_not_supported__");
        return nullptr;
    }
    if (library == SdkLibrary::kOther || library == SdkLibrary::kDriver) {
        for (const auto& symbol : symbols()) {
            if (symbol.exported != name) continue;
            if (library == SdkLibrary::kDriver && !lookup(handle, name)) return nullptr;
            return symbol.wrapper;
        }
    }
    if (library == SdkLibrary::kOther || library == SdkLibrary::kRuntime) {
        for (const auto& symbol : runtime_symbols()) {
            if (symbol.exported != name) continue;
            if (library == SdkLibrary::kRuntime) {
                auto* address = lookup(handle, name);
                if (!address) return nullptr;
                if (address != native_driver_symbol(name)) break;
            }
            return symbol.wrapper;
        }
    }
    if (library == SdkLibrary::kOther || library == SdkLibrary::kBlas) {
        for (const auto& symbol : blas_symbols()) {
            if (symbol.exported != name) continue;
            if (library == SdkLibrary::kBlas && !lookup(handle, name)) return nullptr;
            return symbol.wrapper;
        }
    }
    if (library == SdkLibrary::kOther || library == SdkLibrary::kLt) {
        for (const auto& symbol : lt_symbols()) {
            if (symbol.exported != name) continue;
            if (library == SdkLibrary::kLt && !lookup(handle, name)) return nullptr;
            return symbol.wrapper;
        }
    }
    if (library == SdkLibrary::kOther || library == SdkLibrary::kRandom) {
        for (const auto& symbol : random_symbols()) {
            if (symbol.exported != name) continue;
            if (library == SdkLibrary::kRandom && !lookup(handle, name)) return nullptr;
            return symbol.wrapper;
        }
    }
#ifdef GLIMMER_HAS_CUDNN
    if (library == SdkLibrary::kOther || library == SdkLibrary::kDnn) {
        for (const auto& symbol : dnn_symbols()) {
            if (symbol.exported != name) continue;
            if (library == SdkLibrary::kDnn && !lookup(handle, name)) return nullptr;
            return symbol.wrapper;
        }
        if (std::string_view{name}.starts_with("cudnn")) {
            // An unsupported caller-native consumer must not receive opaque
            // remote objects through global or explicit dynamic lookup.
            (void)lookup(handle, "__glimmer_execution_cudnn_lookup_not_supported__");
            return nullptr;
        }
    }
#endif
#ifdef GLIMMER_HAS_NCCL
    if (library == SdkLibrary::kOther || library == SdkLibrary::kNccl) {
        for (const auto& symbol : nccl_symbols()) {
            if (symbol.exported != name) continue;
            if (library == SdkLibrary::kNccl && !lookup(handle, name)) return nullptr;
            return symbol.wrapper;
        }
        if (std::string_view{name}.starts_with("nccl") ||
            std::string_view{name}.starts_with("pnccl")) {
            // Native opaque consumers cannot accept remote communicator tokens.
            (void)lookup(handle, "__glimmer_execution_nccl_lookup_not_supported__");
            return nullptr;
        }
    }
#endif
    if (library != SdkLibrary::kOther) {
        // Leave a real loader error rather than a false successful NULL lookup.
        (void)lookup(handle, "__glimmer_execution_driver_lookup_not_supported__");
        return nullptr;
    }
    forward = lookup;
    return nullptr;
}
}  // namespace glimmer::interceptor::execution

// Exact hidden assembly callback ABI shared by the independent DSOs.
// NOLINTNEXTLINE(bugprone-reserved-identifier)
extern "C" __attribute__((visibility("hidden"), no_sanitize("address", "undefined"))) void*
glimmer_dispatch_dlsym(void* handle, const char* name, void* caller,
                       void* (**forward)(void*, const char*)) noexcept {
    *forward = nullptr;
    if (!name) return nullptr;
    const bool is_driver = name[0] == 'c' && name[1] == 'u' && name[2] >= 'A' && name[2] <= 'Z';
    const bool is_runtime = name[0] == 'c' && name[1] == 'u' && name[2] == 'd' && name[3] == 'a' &&
                            name[4] >= 'A' && name[4] <= 'Z';
    const bool is_compiler = name[0] == '_' && name[1] == '_' && name[2] == 'c' && name[3] == 'u' &&
                             name[4] == 'd' && name[5] == 'a' &&
                             (name[6] == 'R' || name[6] == 'U' || name[6] == 'I');
    const bool is_blas = name[0] == 'c' && name[1] == 'u' && name[2] == 'b' && name[3] == 'l' &&
                         name[4] == 'a' && name[5] == 's' && name[6] >= 'A' && name[6] <= 'Z';
    const bool is_random = name[0] == 'c' && name[1] == 'u' && name[2] == 'r' && name[3] == 'a' &&
                           name[4] == 'n' && name[5] == 'd' && name[6] >= 'A' && name[6] <= 'Z';
#ifdef GLIMMER_HAS_CUDNN
    const bool is_dnn = name[0] == 'c' && name[1] == 'u' && name[2] == 'd' && name[3] == 'n' &&
                        name[4] == 'n' && name[5] >= 'A' && name[5] <= 'Z';
#else
    constexpr bool is_dnn = false;
#endif
#ifdef GLIMMER_HAS_NCCL
    // This leaf also runs before ASan shadow initialization. Do not call even
    // inline/instrumented STL helpers for unrelated loader-bootstrap symbols.
    const bool is_nccl =
        (name[0] == 'n' && name[1] == 'c' && name[2] == 'c' && name[3] == 'l') ||
        (name[0] == 'p' && name[1] == 'n' && name[2] == 'c' && name[3] == 'c' && name[4] == 'l');
#else
    constexpr bool is_nccl = false;
#endif
    if (handle == RTLD_NEXT || (!is_driver && !is_runtime && !is_compiler && !is_blas &&
                                !is_random && !is_dnn && !is_nccl)) {
        *forward = glimmer::interceptor::execution::native_lookup();
        return nullptr;
    }
    if (::getpid() != glimmer::interceptor::execution::kLookupProcess) return nullptr;
    return glimmer::interceptor::execution::routed_lookup(handle, name, caller, *forward);
}
