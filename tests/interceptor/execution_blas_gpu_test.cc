#include <cuda.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>

namespace {
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " + std::to_string(result));
}
void check(cublasStatus_t result, const char* operation) {
    if (result != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS " + std::to_string(result));
}
void expect(bool condition, const char* operation) {
    if (!condition) throw std::runtime_error(operation);
}
constexpr int kRows = 97, kColumns = 113, kDepth = 71, kLeading = 128;
constexpr std::size_t kElements = kLeading * kLeading + 32;
constexpr unsigned int kWorkers = 320, kThreads = 32;
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry produce(.param .u64 matrix, .param .u64 scalars,
                       .param .u32 count, .param .u64 duration) {
 .reg .pred %p<4>; .reg .b32 %r<12>; .reg .b64 %d<10>;
 ld.param.u64 %d1, [matrix]; ld.param.u64 %d2, [scalars];
 ld.param.u32 %r1, [count]; ld.param.u64 %d3, [duration];
 mov.u64 %d4, %globaltimer;
wait:
 mov.u64 %d5, %globaltimer; sub.u64 %d6, %d5, %d4;
 setp.lt.u64 %p1, %d6, %d3; @%p1 bra wait;
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4;
 mov.u32 %r6, %nctaid.x; mul.lo.u32 %r6, %r6, %r3;
 setp.ne.u32 %p2, %r5, 0; @%p2 bra fill;
 st.global.u32 [%d2], 0x40000000; add.u64 %d7, %d2, 4;
 st.global.u32 [%d7], 0x3f000000;
fill:
 setp.ge.u32 %p3, %r5, %r1; @%p3 bra done;
 rem.u32 %r7, %r5, 5; sub.s32 %r8, %r7, 2;
 setp.eq.u32 %p2, %r5, 0; @%p2 mov.u32 %r8, 3;
 cvt.rn.f32.s32 %r9, %r8;
 mul.wide.u32 %d8, %r5, 4; add.u64 %d9, %d1, %d8;
 st.global.u32 [%d9], %r9;
 add.u32 %r5, %r5, %r6; bra fill;
done: ret;
}
)ptx";
template <typename Function>
Function select(Function original, const char* name, void* provider, bool cached) {
    if (!cached) return original;
    ::dlerror();
    auto* value = ::dlsym(provider, name);
    const auto* error = ::dlerror();
    expect(value && !error, "covered ordinary cached cuBLAS symbol");
    return reinterpret_cast<Function>(value);
}
void run(bool cached, bool global, bool scalar_edges, bool handle_policy) {
    void* provider =
        cached && !global ? ::dlopen("libcublas.so.12", RTLD_NOW | RTLD_LOCAL) : RTLD_DEFAULT;
    if (cached && !global) expect(provider, "open original installed cuBLAS provider");
#define SELECT(NAME) const auto NAME##_call = select(&(NAME), #NAME, provider, cached)
    SELECT(cublasCreate_v2);
    SELECT(cublasDestroy_v2);
    SELECT(cublasSetStream_v2);
    SELECT(cublasGetStream_v2);
    SELECT(cublasSetPointerMode_v2);
    SELECT(cublasGetPointerMode_v2);
    SELECT(cublasSetMathMode);
    SELECT(cublasGetMathMode);
    SELECT(cublasGetVersion_v2);
    SELECT(cublasSgemm_v2);
    SELECT(cublasSetWorkspace_v2);
    SELECT(cublasSetAtomicsMode);
    SELECT(cublasGetAtomicsMode);
    SELECT(cublasSetSmCountTarget);
    SELECT(cublasGetSmCountTarget);
#undef SELECT
    check(cuInit(0), "initialize");
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "device");
    int sms{};
    check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
          "available SMs");
    expect(sms == 80, "identified current V100 has 80 available SMs, not participation proof");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, device), "context");
    cublasHandle_t handle{};
    check(cublasCreate_v2_call(&handle), "create original cuBLAS");
    int version{};
    check(cublasGetVersion_v2_call(handle, &version), "original cuBLAS version");
    expect(version > 0, "actual SDK version");
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "stream");
    check(cublasSetStream_v2_call(handle, reinterpret_cast<cudaStream_t>(stream)),
          "bind owned stream");
    cudaStream_t queried{};
    check(cublasGetStream_v2_call(handle, &queried), "query actual stream");
    expect(queried == reinterpret_cast<cudaStream_t>(stream), "stream identity is unchanged");
    check(cublasSetMathMode_call(handle, CUBLAS_DEFAULT_MATH), "set math mode");
    cublasMath_t math{};
    check(cublasGetMathMode_call(handle, &math), "get math mode");
    expect(math == CUBLAS_DEFAULT_MATH, "actual original math mode");
    expect(
        cublasSetMathMode_call(handle, static_cast<cublasMath_t>(5)) == CUBLAS_STATUS_INVALID_VALUE,
        "original native invalid-math status survives transparent transport");
    check(cublasGetMathMode_call(handle, &math), "math state after original rejection");
    expect(math == CUBLAS_DEFAULT_MATH,
           "rejected original configuration does not poison later SGEMM");
    CUmodule module{};
    CUfunction producer{};
    check(cuModuleLoadData(&module, kSource), "load original producer");
    check(cuModuleGetFunction(&producer, module, "produce"), "original producer function");
    CUdeviceptr a{}, b{}, c{}, scalar{};
    check(cuMemAlloc(&a, kElements * sizeof(float)), "allocate A");
    check(cuMemAlloc(&b, kElements * sizeof(float)), "allocate B");
    check(cuMemAlloc(&c, kElements * sizeof(float)), "allocate C");
    check(cuMemAlloc(&scalar, 2 * sizeof(float)), "allocate device scalars");
    constexpr std::size_t kWorkspaceBytes = 4 * 1024 * 1024;
    constexpr std::size_t kWorkspaceGuardBytes = 256;
    constexpr unsigned int kWorkspaceCanary = 0x713a4b2c;
    CUdeviceptr workspace{};
    if (handle_policy) {
        check(cuMemAlloc(&workspace, kWorkspaceBytes + 2 * kWorkspaceGuardBytes),
              "allocate guarded application-owned workspace");
        expect(workspace % 256 == 0, "actual SDK workspace base is 256-byte aligned");
        check(cuMemsetD32(workspace, kWorkspaceCanary,
                          (kWorkspaceBytes + 2 * kWorkspaceGuardBytes) / sizeof(unsigned int)),
              "initialize workspace and both guards");
        cublasAtomicsMode_t atomics{};
        int sm_count{-1};
        check(cublasGetAtomicsMode_call(handle, &atomics), "original default atomics mode");
        check(cublasGetSmCountTarget_call(handle, &sm_count), "original default SM hint");
        expect(atomics == CUBLAS_ATOMICS_NOT_ALLOWED && sm_count == 0,
               "default per-handle policies do not partition the device");
        check(cublasSetAtomicsMode_call(handle, CUBLAS_ATOMICS_ALLOWED), "allow original atomics");
        check(cublasGetAtomicsMode_call(handle, &atomics), "query original atomics mutation");
        expect(atomics == CUBLAS_ATOMICS_ALLOWED,
               "atomics mutation survives transparent transport");
        check(cublasSetAtomicsMode_call(handle, CUBLAS_ATOMICS_NOT_ALLOWED),
              "restore atomics policy");
        // The SDK, not a private substitute, determines alignment errors.
        expect(cublasSetWorkspace_v2_call(
                   handle, reinterpret_cast<void*>(workspace + kWorkspaceGuardBytes + 1),
                   kWorkspaceBytes - 1) == CUBLAS_STATUS_INVALID_VALUE,
               "original workspace alignment rejection is preserved");
    }
    std::vector<float> expected_a(kElements), input_b(kElements), initial(kElements, 6.0F),
        output(kElements);
    for (std::size_t i = 0; i < kElements; ++i) {
        expected_a[i] = i == 0 ? 3.0F : static_cast<float>(static_cast<int>(i % 5) - 2);
        input_b[i] = static_cast<float>(static_cast<int>(i % 7) - 3);
    }
    check(cuMemcpyHtoD(b, input_b.data(), kElements * sizeof(float)), "upload B");
    std::size_t checked{};
    for (const auto mode : {CUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_DEVICE}) {
        check(cublasSetPointerMode_v2_call(handle, mode), "set original scalar mode");
        cublasPointerMode_t queried_mode{};
        check(cublasGetPointerMode_v2_call(handle, &queried_mode), "get original scalar mode");
        expect(mode == queried_mode, "actual pointer mode is preserved");
        for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
            for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
                if (handle_policy) {
                    // Explicit zero disables the pool; even identical SetStream
                    // restores it. Rebind the actual application buffer after it.
                    check(cublasSetWorkspace_v2_call(handle, nullptr, 0), "disable default pool");
                    check(cublasSetStream_v2_call(handle, reinterpret_cast<cudaStream_t>(stream)),
                          "same-stream workspace reset");
                    check(cublasSetWorkspace_v2_call(
                              handle, reinterpret_cast<void*>(workspace + kWorkspaceGuardBytes),
                              kWorkspaceBytes),
                          "bind guarded explicit workspace");
                    // Both unrestricted replacement eligibility and a genuine
                    // native per-handle hint consume the unchanged numerical oracle.
                    const int target = ta == CUBLAS_OP_N ? 0 : 7;
                    check(cublasSetSmCountTarget_call(handle, target), "set per-handle SM hint");
                    int actual{-1};
                    check(cublasGetSmCountTarget_call(handle, &actual), "query per-handle SM hint");
                    expect(actual == target,
                           "original SM hint is neither ignored nor global policy");
                }
                const std::array<float, 2> stale{99.0F, 99.0F};
                check(cuMemsetD32(a, 0x42c60000, kElements), "stale A before GPU producer");
                check(cuMemcpyHtoD(scalar, stale.data(), sizeof(stale)), "stale device scalars");
                check(cuMemcpyHtoD(c, initial.data(), kElements * sizeof(float)),
                      "reset guarded C");
                auto count = static_cast<unsigned int>(kElements);
                std::uint64_t duration_ns = 70'000'000;
                void* parameters[]{&a, &scalar, &count, &duration_ns};
                check(cuLaunchKernel(producer, kWorkers, 1, 1, kThreads, 1, 1, 0, stream,
                                     parameters, nullptr),
                      "produce real matrix and device scalars");
                float alpha{2.0F}, beta{0.5F};
                // Exact SDK pointer ABI: device scalars are never host-read.
                // NOLINTBEGIN(performance-no-int-to-ptr)
                const auto* av = mode == CUBLAS_POINTER_MODE_HOST
                                     ? &alpha
                                     : reinterpret_cast<const float*>(scalar);
                const auto* bv = mode == CUBLAS_POINTER_MODE_HOST
                                     ? &beta
                                     : reinterpret_cast<const float*>(scalar + sizeof(float));
                for (int repeat = 0; repeat < 2; ++repeat)
                    check(cublasSgemm_v2_call(handle, ta, tb, kRows, kColumns, kDepth, av,
                                              reinterpret_cast<const float*>(a), kLeading,
                                              reinterpret_cast<const float*>(b), kLeading, bv,
                                              reinterpret_cast<float*>(c), kLeading),
                          "original queued SGEMM");
                // NOLINTEND(performance-no-int-to-ptr)
                alpha = beta = 99.0F;
                // A synchronous legacy-stream copy does not wait for an
                // explicit nonblocking stream in the original Driver contract.
                check(cuStreamSynchronize(stream), "complete original nonblocking stream");
                check(cuMemcpyDtoH(output.data(), c, kElements * sizeof(float)),
                      "completed ordinary download");
                for (std::size_t index = 0; index < output.size(); ++index) {
                    const int row = static_cast<int>(index % kLeading);
                    const int column = static_cast<int>(index / kLeading);
                    float expected{6.0F};
                    if (row < kRows && column < kColumns) {
                        float sum{};
                        for (int k = 0; k < kDepth; ++k) {
                            const auto ai =
                                ta == CUBLAS_OP_N ? row + k * kLeading : k + row * kLeading;
                            const auto bi =
                                tb == CUBLAS_OP_N ? k + column * kLeading : column + k * kLeading;
                            sum += expected_a[static_cast<std::size_t>(ai)] *
                                   input_b[static_cast<std::size_t>(bi)];
                        }
                        expected = 3.0F * sum + 1.5F;
                    }
                    if (output[index] != expected)
                        throw std::runtime_error(
                            "GEMM oracle mismatch: mode=" + std::to_string(mode) +
                            "; ta=" + std::to_string(ta) + "; tb=" + std::to_string(tb) +
                            "; index=" + std::to_string(index) + "; expected=" +
                            std::to_string(expected) + "; actual=" + std::to_string(output[index]));
                    ++checked;
                }
            }
        }
    }
    check(cuMemcpyDtoH(output.data(), a, kElements * sizeof(float)),
          "verify producer source unchanged by GEMM");
    expect(output == expected_a, "all GPU-produced inputs retain their original values");
    if (handle_policy) {
        check(cuMemcpyDtoH(output.data(), b, kElements * sizeof(float)), "verify complete B input");
        expect(output == input_b, "workspace configuration cannot mutate a matrix input");
        std::array<float, 2> actual_scalars{};
        check(cuMemcpyDtoH(actual_scalars.data(), scalar, sizeof(actual_scalars)),
              "verify complete device scalar inputs");
        expect(actual_scalars == std::array<float, 2>{2.0F, 0.5F},
               "workspace and per-handle policy retain device scalar values");
        check(cublasSetSmCountTarget_call(handle, 0), "restore unrestricted handle policy");
    }
    std::size_t edge_values{};
    if (scalar_edges) {
        for (const auto mode : {CUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_DEVICE}) {
            check(cublasSetPointerMode_v2_call(handle, mode), "edge scalar mode");
            for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
                for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
                    for (int edge = 0; edge < 3; ++edge) {
                        std::vector<float> edge_a = expected_a, edge_b = input_b, edge_c = initial;
                        const float nan = std::numeric_limits<float>::quiet_NaN();
                        if (edge < 2) {
                            std::fill(edge_a.begin(), edge_a.end(), nan);
                            std::fill(edge_b.begin(), edge_b.end(), nan);
                        }
                        for (int column = 0; column < kColumns; ++column)
                            for (int row = 0; row < kRows; ++row)
                                edge_c[static_cast<std::size_t>(column) * kLeading +
                                       static_cast<std::size_t>(row)] = edge == 1 ? 6.0F : nan;
                        check(cuMemcpyHtoD(a, edge_a.data(), edge_a.size() * sizeof(float)),
                              "edge A");
                        check(cuMemcpyHtoD(b, edge_b.data(), edge_b.size() * sizeof(float)),
                              "edge B");
                        check(cuMemcpyHtoD(c, edge_c.data(), edge_c.size() * sizeof(float)),
                              "edge C");
                        const std::array<float, 2> values{edge < 2 ? 0.0F : 2.0F,
                                                          edge == 1 ? 0.5F : 0.0F};
                        check(cuMemcpyHtoD(scalar, values.data(), sizeof(values)),
                              "edge device scalars");
                        // NOLINTBEGIN(performance-no-int-to-ptr): exact native ABI.
                        const auto* av = mode == CUBLAS_POINTER_MODE_HOST
                                             ? &values[0]
                                             : reinterpret_cast<const float*>(scalar);
                        const auto* bv =
                            mode == CUBLAS_POINTER_MODE_HOST
                                ? &values[1]
                                : reinterpret_cast<const float*>(scalar + sizeof(float));
                        check(cublasSgemm_v2_call(handle, ta, tb, kRows, kColumns, kDepth, av,
                                                  reinterpret_cast<const float*>(a), kLeading,
                                                  reinterpret_cast<const float*>(b), kLeading, bv,
                                                  reinterpret_cast<float*>(c), kLeading),
                              "edge original SGEMM");
                        // NOLINTEND(performance-no-int-to-ptr)
                        check(cuStreamSynchronize(stream), "edge actual stream completion");
                        check(cuMemcpyDtoH(output.data(), c, output.size() * sizeof(float)),
                              "edge output");
                        for (std::size_t index = 0; index < output.size(); ++index) {
                            const int row = static_cast<int>(index % kLeading),
                                      column = static_cast<int>(index / kLeading);
                            float expected = 6.0F;
                            if (row < kRows && column < kColumns) {
                                expected = edge == 1 ? 3.0F : 0.0F;
                                if (edge == 2) {
                                    for (int k = 0; k < kDepth; ++k) {
                                        const auto ai = ta == CUBLAS_OP_N ? row + k * kLeading
                                                                          : k + row * kLeading;
                                        const auto bi = tb == CUBLAS_OP_N ? k + column * kLeading
                                                                          : column + k * kLeading;
                                        expected += 2.0F *
                                                    expected_a[static_cast<std::size_t>(ai)] *
                                                    input_b[static_cast<std::size_t>(bi)];
                                    }
                                }
                            }
                            const bool is_native_nan = mode == CUBLAS_POINTER_MODE_DEVICE &&
                                                       edge < 2 && row < kRows && column < kColumns;
                            if (is_native_nan ? !std::isnan(output[index])
                                              : output[index] != expected)
                                throw std::runtime_error(
                                    "scalar edge mismatch: mode=" + std::to_string(mode) +
                                    "; ta=" + std::to_string(ta) + "; tb=" + std::to_string(tb) +
                                    "; edge=" + std::to_string(edge) +
                                    "; index=" + std::to_string(index) +
                                    "; expected=" + std::to_string(expected) +
                                    "; actual=" + std::to_string(output[index]));
                            ++edge_values;
                        }
                    }
                }
            }
        }
    }
    check(cublasDestroy_v2_call(handle), "destroy original cuBLAS handle");
    if (handle_policy) {
        std::array<unsigned int, kWorkspaceGuardBytes / sizeof(unsigned int)> guard{};
        for (const auto offset : {std::size_t{0}, kWorkspaceGuardBytes + kWorkspaceBytes}) {
            check(cuMemcpyDtoH(guard.data(), workspace + offset, sizeof(guard)),
                  "workspace remains application-owned after handle destruction");
            expect(std::all_of(guard.begin(), guard.end(),
                               [](auto value) { return value == kWorkspaceCanary; }),
                   "original and replacement work preserve both explicit workspace guards");
        }
        check(cuMemFree(workspace), "caller releases its own workspace exactly once");
    }
    check(cuMemFree(a), "release A");
    check(cuMemFree(b), "release B");
    check(cuMemFree(c), "release C");
    check(cuMemFree(scalar), "release scalars");
    check(cuModuleUnload(module), "release original module");
    check(cuStreamDestroy(stream), "release stream");
    check(cuCtxDestroy(context), "release context");
    if (cached && !global) expect(::dlclose(provider) == 0, "close caller provider reference");
    std::cout << "Ordinary cuBLAS application verified: values=" << checked
              << "; transpose_pairs=9; scalar_modes=2; producer_dataflow=true; guards=true; "
                 "workers=320; cached="
              << std::boolalpha << cached << "; global=" << global << "; version=" << version
              << "; full_device_qualified=false; performance_qualified=false\n";
    if (scalar_edges)
        std::cout << "Original SGEMM scalar edges verified: values=" << edge_values
                  << "; alpha_zero_scalar_modes_match_native=true; beta_zero_nan_output=true\n";
    if (handle_policy)
        std::cout << "Original cuBLAS handle policies verified: explicit_workspace_bytes="
                  << kWorkspaceBytes
                  << "; workspace_guards=128; atomics=true; sm_targets=0,7; "
                     "same_stream_set=true; numerical_oracle_unchanged=true; "
                     "full_device_qualified=false; priority_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        bool cached{}, global{}, scalar_edges{}, handle_policy{};
        for (int i = 1; i < argc; ++i) {
            const std::string_view value(argv[i]);
            if (value == "--cached-lookups")
                cached = true;
            else if (value == "--global-lookups")
                cached = global = true;
            else if (value == "--scalar-edges")
                scalar_edges = true;
            else if (value == "--handle-policy")
                handle_policy = true;
            else
                throw std::runtime_error("unknown ordinary SDK fixture option");
        }
        run(cached, global, scalar_edges, handle_policy);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
