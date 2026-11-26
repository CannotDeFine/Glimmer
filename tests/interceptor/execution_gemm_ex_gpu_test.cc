#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>

namespace {
using GemmEx = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                             int, int, int, const void*, const void*, cudaDataType,
                                             int, const void*, cudaDataType, int, const void*,
                                             void*, cudaDataType, int, cublasComputeType_t,
                                             cublasGemmAlgo_t);
using StridedGemmEx = cublasStatus_t(CUBLASWINAPI*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*,
    cudaDataType, int, long long, const void*, cudaDataType, int, long long, const void*, void*,
    cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);
using Number = std::complex<double>;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status) {
    if (status != CUDA_SUCCESS) throw std::runtime_error("Driver " + std::to_string(status));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("cuBLAS " + std::to_string(status));
}
constexpr std::size_t kElements = 1056;
constexpr int kRows = 17, kColumns = 19, kDepth = 7, kLeading = 32;
struct Profile {
    cudaDataType input, output;
    cublasComputeType_t compute;
};
constexpr std::array kProfiles{Profile{CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
                               Profile{CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F},
                               Profile{CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F},
                               Profile{CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F},
                               Profile{CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC},
                               Profile{CUDA_R_64F, CUDA_R_64F, CUBLAS_COMPUTE_64F},
                               Profile{CUDA_C_32F, CUDA_C_32F, CUBLAS_COMPUTE_32F},
                               Profile{CUDA_C_64F, CUDA_C_64F, CUBLAS_COMPUTE_64F}};
bool complex_type(cudaDataType type) {
    return type == CUDA_C_32F || type == CUDA_C_64F;
}
std::size_t width(cudaDataType type) {
    if (type == CUDA_R_16F) return 2;
    if (type == CUDA_R_32F) return 4;
    if (type == CUDA_C_64F) return 16;
    return 8;
}
void encode(std::byte* output, cudaDataType type, Number value) {
    if (type == CUDA_R_16F) {
        const auto half = __float2half(static_cast<float>(value.real()));
        std::memcpy(output, &half, sizeof(half));
    } else if (type == CUDA_R_32F || type == CUDA_C_32F) {
        const std::array parts{static_cast<float>(value.real()), static_cast<float>(value.imag())};
        std::memcpy(output, parts.data(), width(type));
    } else {
        const std::array parts{value.real(), value.imag()};
        std::memcpy(output, parts.data(), width(type));
    }
}
Number decode(const std::byte* input, cudaDataType type) {
    if (type == CUDA_R_16F) {
        __half half;
        std::memcpy(&half, input, sizeof(half));
        return __half2float(half);
    }
    if (type == CUDA_R_32F || type == CUDA_C_32F) {
        std::array<float, 2> parts{};
        std::memcpy(parts.data(), input, width(type));
        return {parts[0], parts[1]};
    }
    std::array<double, 2> parts{};
    std::memcpy(parts.data(), input, width(type));
    return {parts[0], parts[1]};
}
constexpr const char* kProducer = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry coefficients(.param .u64 pointer, .param .u32 kind) {
 .reg .pred %p<4>; .reg .b32 %r<7>; .reg .b64 %d<4>;
 mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
 mad.lo.u32 %r4,%r1,%r2,%r3; setp.ne.u32 %p1,%r4,0; @%p1 bra done;
 ld.param.u64 %d1,[pointer]; ld.param.u32 %r5,[kind];
 add.u64 %d2,%d1,16;
 setp.eq.u32 %p2,%r5,1; @%p2 bra half;
 setp.eq.u32 %p3,%r5,2; @%p3 bra double;
 st.global.u32 [%d1],0x40000000; st.global.u32 [%d2],0x3f000000;
 add.u64 %d3,%d1,4; st.global.u32 [%d3],0;
 add.u64 %d3,%d2,4; st.global.u32 [%d3],0; bra done;
half:
 st.global.u16 [%d1],0x4000; st.global.u16 [%d2],0x3800; bra done;
double:
 st.global.u64 [%d1],0x4000000000000000; st.global.u64 [%d2],0x3fe0000000000000;
 add.u64 %d3,%d1,8; st.global.u64 [%d3],0;
 add.u64 %d3,%d2,8; st.global.u64 [%d3],0;
done: ret;
}
)ptx";
void run(std::string_view lookup, bool strided, bool uses_classic_batch, bool broadcasts) {
    void* provider =
        lookup == "explicit" ? ::dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL) : RTLD_DEFAULT;
    if (lookup == "explicit") expect(provider, "open installed provider");
    GemmEx gemm = static_cast<GemmEx>(&cublasGemmEx);
    StridedGemmEx batch_gemm = static_cast<StridedGemmEx>(&cublasGemmStridedBatchedEx);
    auto classic_batch = &cublasSgemmStridedBatched;
    if (lookup != "direct") {
        ::dlerror();
        gemm = reinterpret_cast<GemmEx>(::dlsym(provider, "cublasGemmEx"));
        expect(gemm && !::dlerror(), "original direct/cached GemmEx ABI");
        if (strided) {
            batch_gemm =
                reinterpret_cast<StridedGemmEx>(::dlsym(provider, "cublasGemmStridedBatchedEx"));
            expect(batch_gemm && !::dlerror(), "original direct/cached strided GemmEx ABI");
        }
        if (uses_classic_batch) {
            classic_batch = reinterpret_cast<decltype(classic_batch)>(
                ::dlsym(provider, "cublasSgemmStridedBatched"));
            expect(classic_batch && !::dlerror(), "original direct/cached classic FP32 batch ABI");
        }
    }
    check(cuInit(0));
    CUdevice device{};
    CUcontext context{};
    check(cuDeviceGet(&device, 0));
    check(cuCtxCreate(&context, 0, device));
    cublasHandle_t handle{};
    check(cublasCreate_v2(&handle));
    int version{};
    check(cublasGetVersion_v2(handle, &version));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cublasSetStream_v2(handle, reinterpret_cast<cudaStream_t>(stream)));
    CUmodule module{};
    CUfunction producer{};
    check(cuModuleLoadData(&module, kProducer));
    check(cuModuleGetFunction(&producer, module, "coefficients"));
    CUdeviceptr scalars{};
    check(cuMemAlloc(&scalars, 32));
    std::size_t checked{};
    unsigned int calls{};
    const int batches = strided ? 3 : 1;
    // Keep gaps between positive-stride batches in the complete guard oracle.
    const auto elements = kElements * (strided ? 11U : 1U);
    for (const auto profile : kProfiles) {
        if (uses_classic_batch && (profile.input != CUDA_R_32F || profile.output != CUDA_R_32F ||
                                   profile.compute != CUBLAS_COMPUTE_32F))
            continue;
        const auto input_width = width(profile.input), output_width = width(profile.output);
        std::vector<std::byte> a(elements * input_width), b(a.size()), c(elements * output_width),
            output(c.size());
        std::vector<Number> oracle_a(elements), oracle_b(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            oracle_a[index] = {static_cast<double>(static_cast<int>(index % 5) - 2),
                               complex_type(profile.input)
                                   ? static_cast<double>(static_cast<int>(index % 3) - 1)
                                   : 0.0};
            oracle_b[index] = {static_cast<double>(static_cast<int>(index % 7) - 3),
                               complex_type(profile.input)
                                   ? static_cast<double>(static_cast<int>(index % 3) - 1)
                                   : 0.0};
            encode(a.data() + index * input_width, profile.input, oracle_a[index]);
            encode(b.data() + index * input_width, profile.input, oracle_b[index]);
            encode(c.data() + index * output_width, profile.output, 6.0);
        }
        CUdeviceptr da{}, db{}, dc{};
        check(cuMemAlloc(&da, a.size()));
        check(cuMemAlloc(&db, b.size()));
        check(cuMemAlloc(&dc, c.size()));
        check(cuMemcpyHtoD(da, a.data(), a.size()));
        check(cuMemcpyHtoD(db, b.data(), b.size()));
        for (const auto mode : {CUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_DEVICE}) {
            check(cublasSetPointerMode_v2(handle, mode));
            for (int geometry = 0; geometry < (strided ? 3 : 1); ++geometry) {
                const int factor_a = broadcasts      ? (geometry == 1 ? 2 : 0)
                                     : geometry == 0 ? 1
                                     : geometry == 1 ? 2
                                                     : 3;
                const int factor_b = broadcasts      ? (geometry == 0 ? 2 : 0)
                                     : geometry == 0 ? 1
                                     : geometry == 1 ? 3
                                                     : 2;
                const int factor_c = geometry == 0 ? 1 : geometry == 1 ? 4 : 5;
                for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
                    for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
                        check(cuMemcpyHtoD(dc, c.data(), c.size()));
                        alignas(16) std::array<std::byte, 16> alpha{}, beta{};
                        const auto scalar_type = profile.compute == CUBLAS_COMPUTE_16F ? CUDA_R_16F
                                                 : profile.compute == CUBLAS_COMPUTE_64F
                                                     ? profile.output
                                                 : complex_type(profile.output) ? CUDA_C_32F
                                                                                : CUDA_R_32F;
                        encode(alpha.data(), scalar_type, 2.0);
                        encode(beta.data(), scalar_type, 0.5);
                        if (mode == CUBLAS_POINTER_MODE_DEVICE) {
                            std::array<std::byte, 32> stale{};
                            check(cuMemcpyHtoD(scalars, stale.data(), stale.size()));
                            unsigned int kind = scalar_type == CUDA_R_16F               ? 1U
                                                : profile.compute == CUBLAS_COMPUTE_64F ? 2U
                                                                                        : 0U;
                            void* arguments[]{&scalars, &kind};
                            check(cuLaunchKernel(producer, 320, 1, 1, 32, 1, 1, 0, stream,
                                                 arguments, nullptr));
                        }
                        // NOLINTBEGIN(performance-no-int-to-ptr): original SDK GPU address ABI.
                        const auto* av = mode == CUBLAS_POINTER_MODE_HOST
                                             ? alpha.data()
                                             : reinterpret_cast<const void*>(scalars);
                        const auto* bv = mode == CUBLAS_POINTER_MODE_HOST
                                             ? beta.data()
                                             : reinterpret_cast<const void*>(scalars + 16);
                        for (int repeat = 0; repeat < 2; ++repeat) {
                            const auto status =
                                uses_classic_batch
                                    ? classic_batch(handle, ta, tb, kRows, kColumns, kDepth,
                                                    static_cast<const float*>(av),
                                                    reinterpret_cast<const float*>(da), kLeading,
                                                    static_cast<long long>(kElements) * factor_a,
                                                    reinterpret_cast<const float*>(db), kLeading,
                                                    static_cast<long long>(kElements) * factor_b,
                                                    static_cast<const float*>(bv),
                                                    reinterpret_cast<float*>(dc), kLeading,
                                                    static_cast<long long>(kElements) * factor_c,
                                                    batches)
                                : strided
                                    ? batch_gemm(
                                          handle, ta, tb, kRows, kColumns, kDepth, av,
                                          reinterpret_cast<const void*>(da), profile.input,
                                          kLeading, static_cast<long long>(kElements) * factor_a,
                                          reinterpret_cast<const void*>(db), profile.input,
                                          kLeading, static_cast<long long>(kElements) * factor_b,
                                          bv, reinterpret_cast<void*>(dc), profile.output, kLeading,
                                          static_cast<long long>(kElements) * factor_c, batches,
                                          profile.compute, CUBLAS_GEMM_DEFAULT)
                                    : gemm(handle, ta, tb, kRows, kColumns, kDepth, av,
                                           reinterpret_cast<const void*>(da), profile.input,
                                           kLeading, reinterpret_cast<const void*>(db),
                                           profile.input, kLeading, bv, reinterpret_cast<void*>(dc),
                                           profile.output, kLeading, profile.compute,
                                           CUBLAS_GEMM_DEFAULT);
                            if (status != CUBLAS_STATUS_SUCCESS)
                                throw std::runtime_error(
                                    "profile input=" + std::to_string(profile.input) +
                                    "; output=" + std::to_string(profile.output) +
                                    "; compute=" + std::to_string(profile.compute) +
                                    "; status=" + std::to_string(status));
                            ++calls;
                        }
                        // NOLINTEND(performance-no-int-to-ptr)
                        alpha.fill(std::byte{99});
                        beta.fill(std::byte{99});
                        const auto synchronized = cuStreamSynchronize(stream);
                        if (synchronized != CUDA_SUCCESS)
                            throw std::runtime_error(
                                "typed GPU stop: input=" + std::to_string(profile.input) +
                                "; output=" + std::to_string(profile.output) +
                                "; compute=" + std::to_string(profile.compute) + "; geometry=" +
                                std::to_string(geometry) + "; mode=" + std::to_string(mode) +
                                "; ta=" + std::to_string(ta) + "; tb=" + std::to_string(tb) +
                                "; Driver=" + std::to_string(synchronized));
                        check(cuMemcpyDtoH(output.data(), dc, output.size()));
                        for (std::size_t index = 0; index < elements; ++index) {
                            const auto local = index % kElements;
                            const auto physical = index / kElements;
                            const auto batch = physical / static_cast<unsigned int>(factor_c);
                            const bool active =
                                physical % static_cast<unsigned int>(factor_c) == 0 &&
                                batch < static_cast<unsigned int>(batches);
                            const auto a_start =
                                batch * static_cast<unsigned int>(factor_a) * kElements;
                            const auto b_start =
                                batch * static_cast<unsigned int>(factor_b) * kElements;
                            const int row = static_cast<int>(local % kLeading),
                                      column = static_cast<int>(local / kLeading);
                            Number expected = 6.0;
                            if (active && row < kRows && column < kColumns) {
                                Number sum{};
                                for (int k = 0; k < kDepth; ++k) {
                                    auto x = oracle_a[a_start + static_cast<std::size_t>(
                                                                    ta == CUBLAS_OP_N
                                                                        ? row + k * kLeading
                                                                        : k + row * kLeading)];
                                    auto y = oracle_b[b_start + static_cast<std::size_t>(
                                                                    tb == CUBLAS_OP_N
                                                                        ? k + column * kLeading
                                                                        : column + k * kLeading)];
                                    if (ta == CUBLAS_OP_C) x = std::conj(x);
                                    if (tb == CUBLAS_OP_C) y = std::conj(y);
                                    sum += x * y;
                                }
                                expected = 3.0 * sum + 1.5;
                            }
                            if (decode(output.data() + index * output_width, profile.output) !=
                                expected)
                                throw std::runtime_error(
                                    "typed GemmEx oracle mismatch: profile=" +
                                    std::to_string(profile.input) + "/" +
                                    std::to_string(profile.output) +
                                    "; compute=" + std::to_string(profile.compute) +
                                    "; mode=" + std::to_string(mode) +
                                    "; ta=" + std::to_string(ta) + "; tb=" + std::to_string(tb) +
                                    "; index=" + std::to_string(index));
                            ++checked;
                        }
                    }
                }
            }
        }
        expect(output.size() >= a.size(), "typed input check has a complete output buffer");
        check(cuMemcpyDtoH(output.data(), da, a.size()));
        expect(output.size() >= a.size() && std::memcmp(output.data(), a.data(), a.size()) == 0,
               "all original A bytes unchanged");
        check(cuMemcpyDtoH(output.data(), db, b.size()));
        expect(std::memcmp(output.data(), b.data(), b.size()) == 0,
               "all original B bytes unchanged");
        check(cuMemFree(da));
        check(cuMemFree(db));
        check(cuMemFree(dc));
    }
    check(cublasDestroy_v2(handle));
    check(cuMemFree(scalars));
    check(cuModuleUnload(module));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    if (lookup == "explicit") expect(::dlclose(provider) == 0, "close original provider");
    std::cout << (uses_classic_batch ? "Ordinary SgemmStridedBatched verified: profiles=1"
                                     : "Ordinary GemmEx verified: profiles=8")
              << "; transpose_pairs=9; scalar_modes=2; calls=" << calls << "; values=" << checked
              << "; gpu_scalars=true; guards=true; lookup=" << lookup << "; version=" << version
              << "; full_device_qualified=false; performance_qualified=false";
    if (strided) std::cout << "; strided=true; batches=3; geometries=3";
    if (broadcasts) std::cout << "; broadcast=true";
    std::cout << '\n';
}
}  // namespace
int main(int argc, char** argv) {
    try {
        const std::string_view lookup = argc >= 2 ? argv[1] : "direct";
        const std::string_view batch_option = argc == 3 ? argv[2] : "";
        const bool broadcasts =
            batch_option == "--broadcast" || batch_option == "--broadcast-sgemm";
        const bool uses_classic_batch =
            batch_option == "--strided-sgemm" || batch_option == "--broadcast-sgemm";
        const bool strided = uses_classic_batch || broadcasts || batch_option == "--strided";
        expect((argc <= 2 || strided) &&
                   (lookup == "direct" || lookup == "global" || lookup == "explicit"),
               "known lookup option");
        run(lookup, strided, uses_classic_batch, broadcasts);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
