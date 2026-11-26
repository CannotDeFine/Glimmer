#include <cuda.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {
bool check(CUresult result) {
    if (result == CUDA_SUCCESS) return true;
    std::cerr << "Driver status=" << result << '\n';
    return false;
}
bool check_blas(cublasStatus_t result) {
    if (result == CUBLAS_STATUS_SUCCESS) return true;
    std::cerr << "cuBLASLt status=" << result << '\n';
    return false;
}
struct Descriptors {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{};
    cublasLtMatmulPreference_t preference{};
    ~Descriptors() {
        (void)cleanup();
    }
    bool cleanup() {
        bool passed = true;
        if (operation) passed &= check_blas(cublasLtMatmulDescDestroy(operation));
        operation = nullptr;
        for (auto* layout : {&a, &b, &c}) {
            if (*layout) passed &= check_blas(cublasLtMatrixLayoutDestroy(*layout));
            *layout = nullptr;
        }
        if (preference) passed &= check_blas(cublasLtMatmulPreferenceDestroy(preference));
        preference = nullptr;
        return passed;
    }
};
struct Resources {
    CUcontext context{};
    CUstream stream{};
    cublasLtHandle_t handle{}, unknown{};
    cublasHandle_t blas{};
    void* library{};
    std::vector<CUdeviceptr> allocations;
    ~Resources() {
        (void)cleanup();
    }
    bool cleanup() {
        bool passed = true;
        if (context) passed &= check(cuCtxSynchronize());
        if (handle) passed &= check_blas(cublasLtDestroy(handle));
        handle = nullptr;
        if (unknown) passed &= check_blas(cublasLtDestroy(unknown));
        unknown = nullptr;
        if (blas) passed &= check_blas(cublasDestroy(blas));
        blas = nullptr;
        for (const auto pointer : allocations) passed &= check(cuMemFree(pointer));
        allocations.clear();
        if (stream) passed &= check(cuStreamDestroy(stream));
        stream = nullptr;
        if (context) passed &= check(cuCtxDestroy(context));
        context = nullptr;
        if (library) passed &= dlclose(library) == 0;
        library = nullptr;
        return passed;
    }
    bool allocate(CUdeviceptr& pointer, std::size_t bytes) {
        allocations.reserve(allocations.size() + 1);
        if (!check(cuMemAlloc(&pointer, bytes))) return false;
        allocations.push_back(pointer);
        return true;
    }
};

// Modes: direct, global lookup, device scalars, explicit stream, nonzero beta,
// no bias, unseen Lt handle, and SDK-documented cuBLAS-as-Lt handle.
bool run(Resources& resources, int m, int k, int mode, int n = 32) {
    constexpr std::size_t guard = 64;
    constexpr float sentinel = -12345.0F;
    constexpr std::size_t workspace_bytes = std::size_t{8} * 1024 * 1024;
    const auto rows = static_cast<unsigned int>(m);
    const auto depth = static_cast<unsigned int>(k);
    const auto columns = static_cast<unsigned int>(n);
    const auto lda = depth + 64;
    const auto ldb = depth + 64;
    const auto ldc = rows + 64;
    std::vector<float> left(std::size_t{lda} * rows + 2 * guard, sentinel);
    std::vector<float> right(std::size_t{ldb} * columns + 2 * guard, sentinel);
    std::vector<float> output(std::size_t{ldc} * columns + 2 * guard, sentinel);
    std::vector<float> bias(rows + 2 * guard, sentinel);
    for (unsigned int row = 0; row < rows; ++row) {
        bias[guard + row] = static_cast<float>(static_cast<int>(row % 17) - 8) / 16.0F;
        for (unsigned int inner = 0; inner < depth; ++inner)
            left[guard + std::size_t{row} * lda + inner] =
                static_cast<float>(static_cast<int>((row + inner) % 29) - 14) / 32.0F;
    }
    for (unsigned int col = 0; col < columns; ++col) {
        for (unsigned int inner = 0; inner < depth; ++inner)
            right[guard + std::size_t{col} * ldb + inner] =
                static_cast<float>(static_cast<int>((inner + col) % 31) - 15) / 32.0F;
        for (unsigned int row = 0; row < rows; ++row)
            output[guard + std::size_t{col} * ldc + row] =
                mode == 4 ? 3.0F : std::numeric_limits<float>::quiet_NaN();
    }
    CUdeviceptr a{}, b{}, c{}, bias_all{}, reference{}, workspace{}, scalars{};
    const std::array<float, 2> values{0.75F, mode == 4 ? 0.5F : 0.0F};
    if (!resources.allocate(a, left.size() * sizeof(float)) ||
        !resources.allocate(b, right.size() * sizeof(float)) ||
        !resources.allocate(c, output.size() * sizeof(float)) ||
        !resources.allocate(reference, output.size() * sizeof(float)) ||
        !resources.allocate(bias_all, bias.size() * sizeof(float)) ||
        !resources.allocate(workspace, workspace_bytes) ||
        !resources.allocate(scalars, sizeof(values)) ||
        !check(cuMemcpyHtoD(a, left.data(), left.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(b, right.data(), right.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(c, output.data(), output.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(reference, output.data(), output.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(bias_all, bias.data(), bias.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(scalars, values.data(), sizeof(values))))
        return false;
    Descriptors desc;
    const cublasOperation_t transpose = CUBLAS_OP_T;
    const cublasLtEpilogue_t epilogue =
        mode == 5 ? CUBLASLT_EPILOGUE_DEFAULT : CUBLASLT_EPILOGUE_BIAS;
    const cublasLtPointerMode_t pointers =
        mode == 2 ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
    const auto* bias_pointer =
        mode == 5 ? nullptr : std::bit_cast<const float*>(bias_all + guard * sizeof(float));
    if (!check_blas(cublasLtMatmulDescCreate(&desc.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F)) ||
        !check_blas(cublasLtMatrixLayoutCreate(&desc.a, CUDA_R_32F, depth, rows, lda)) ||
        !check_blas(cublasLtMatrixLayoutCreate(&desc.b, CUDA_R_32F, depth, columns, ldb)) ||
        !check_blas(cublasLtMatrixLayoutCreate(&desc.c, CUDA_R_32F, rows, columns, ldc)) ||
        !check_blas(cublasLtMatmulDescSetAttribute(desc.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                   &transpose, sizeof(transpose))) ||
        !check_blas(cublasLtMatmulDescSetAttribute(desc.operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                   &epilogue, sizeof(epilogue))) ||
        !check_blas(cublasLtMatmulDescSetAttribute(desc.operation,
                                                   CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
                                                   sizeof(bias_pointer))) ||
        !check_blas(cublasLtMatmulDescSetAttribute(
            desc.operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointers, sizeof(pointers))) ||
        !check_blas(cublasLtMatmulPreferenceCreate(&desc.preference)) ||
        !check_blas(cublasLtMatmulPreferenceSetAttribute(
            desc.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
            sizeof(workspace_bytes))))
        return false;
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned{};
    const auto reference_handle = resources.handle ? resources.handle : resources.unknown;
    if (!check_blas(cublasLtMatmulAlgoGetHeuristic(reference_handle, desc.operation, desc.a, desc.b,
                                                   desc.c, desc.c, desc.preference, 1, &heuristic,
                                                   &returned)) ||
        returned != 1 || !check_blas(heuristic.state))
        return false;
    const auto native =
        reinterpret_cast<decltype(&cublasLtMatmul)>(dlsym(resources.library, "cublasLtMatmul"));
    auto function = &cublasLtMatmul;
    if (mode == 1)
        function = reinterpret_cast<decltype(function)>(dlsym(RTLD_DEFAULT, "cublasLtMatmul"));
    if (!native || !function) return false;
    const auto* alpha = mode == 2 ? std::bit_cast<const float*>(scalars) : &values[0];
    const auto* beta =
        mode == 2 ? std::bit_cast<const float*>(scalars + sizeof(float)) : &values[1];
    const auto invoke = [&](auto selected, cublasLtHandle_t handle, CUdeviceptr dest) {
        auto* destination = std::bit_cast<float*>(dest + guard * sizeof(float));
        return selected(handle, desc.operation, alpha,
                        std::bit_cast<const float*>(a + guard * sizeof(float)), desc.a,
                        std::bit_cast<const float*>(b + guard * sizeof(float)), desc.b, beta,
                        destination, desc.c, destination, desc.c, &heuristic.algo,
                        std::bit_cast<void*>(workspace), workspace_bytes,
                        mode == 3 ? resources.stream : nullptr);
    };
    if (!check_blas(invoke(native, reference_handle, reference)) || !check(cuCtxSynchronize()))
        return false;
    const auto handle = mode == 6   ? resources.unknown
                        : mode == 7 ? reinterpret_cast<cublasLtHandle_t>(resources.blas)
                                    : resources.handle;
    std::cerr << "public_lt_begin expected="
              << ((n == 32 || n == 768) && (mode == 0 || mode == 1 || mode == 7)) << '\n';
    if (!check_blas(invoke(function, handle, c))) return false;
    std::cerr << "public_lt_end\n";
    // Accepted work cannot retain borrowed descriptors/algorithm bytes. Destroy
    // them before any explicit completion operation, just as the framework does.
    if (!desc.cleanup() || !check(cuCtxSynchronize())) return false;
    std::vector<float> expected(output.size()), actual(output.size()), read_left(left.size()),
        read_right(right.size()), read_bias(bias.size());
    if (!check(cuMemcpyDtoH(expected.data(), reference, expected.size() * sizeof(float))) ||
        !check(cuMemcpyDtoH(actual.data(), c, actual.size() * sizeof(float))) ||
        !check(cuMemcpyDtoH(read_left.data(), a, left.size() * sizeof(float))) ||
        !check(cuMemcpyDtoH(read_right.data(), b, right.size() * sizeof(float))) ||
        !check(cuMemcpyDtoH(read_bias.data(), bias_all, bias.size() * sizeof(float))) ||
        read_left != left || read_right != right || read_bias != bias)
        return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const bool is_output =
            index >= guard && index < actual.size() - guard && (index - guard) % ldc < rows;
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
            std::abs(actual[index] - expected[index]) > 2e-5F + 2e-4F * std::abs(expected[index]) ||
            (!is_output && (actual[index] != sentinel || expected[index] != sentinel))) {
            std::cerr << "Lt mismatch index=" << index << " mode=" << mode << '\n';
            return false;
        }
    }
    std::cout << "public_lt m=" << m << " n=" << n << " k=" << k << " mode=" << mode
              << " native_match=1 inputs=1 guards=1 descriptors_retired=1\n";
    return true;
}
}  // namespace

int main() {
    Resources resources;
    CUdevice device{};
    if (!check(cuInit(0)) || !check(cuDeviceGet(&device, 0)) ||
        !check(cuCtxCreate(&resources.context, CU_CTX_MAP_HOST, device)) ||
        !check(cuStreamCreate(&resources.stream, CU_STREAM_NON_BLOCKING)))
        return EXIT_FAILURE;
    resources.library = dlopen("libcublasLt.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!resources.library) return EXIT_FAILURE;
    const auto create =
        reinterpret_cast<decltype(&cublasLtCreate)>(dlsym(resources.library, "cublasLtCreate"));
    if (!create || !check_blas(create(&resources.unknown))) return EXIT_FAILURE;
    // CUDA 12.6 may return the same Lt handle for repeated Create calls. Prove
    // decline before *any* public creation, not by assuming a distinct address.
    if (!run(resources, 4096, 1024, 6) || !check_blas(cublasLtCreate(&resources.handle)) ||
        !check_blas(cublasCreate(&resources.blas)))
        return EXIT_FAILURE;
    for (int mode = 0; mode < 2; ++mode)
        if (!run(resources, 4096, 1024, mode) || !run(resources, 1024, 4096, mode))
            return EXIT_FAILURE;
    for (int mode = 2; mode < 8; ++mode)
        if (mode != 6 && !run(resources, 4096, 1024, mode)) return EXIT_FAILURE;
    for (int mode = 0; mode < 2; ++mode) {
        if (!run(resources, 768, 768, mode, 768) || !run(resources, 3072, 768, mode, 768) ||
            !run(resources, 768, 3072, mode, 768))
            return EXIT_FAILURE;
    }
    for (int mode = 2; mode < 5; ++mode)
        if (!run(resources, 768, 768, mode, 768)) return EXIT_FAILURE;
    if (!run(resources, 768, 768, 0, 128)) return EXIT_FAILURE;
    if (!resources.cleanup()) return EXIT_FAILURE;
    std::cout << "public_lt cleanup=1\n";
    return EXIT_SUCCESS;
}
