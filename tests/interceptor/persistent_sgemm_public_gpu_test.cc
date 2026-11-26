#include <cuda.h>
#include <cublas_v2.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
bool check(CUresult result) {
    if (result == CUDA_SUCCESS) return true;
    std::cerr << "Driver status=" << result << '\n';
    return false;
}
bool check_blas(cublasStatus_t result) {
    if (result == CUBLAS_STATUS_SUCCESS) return true;
    std::cerr << "cuBLAS status=" << result << '\n';
    return false;
}
struct Resources {
    CUcontext context{};
    CUstream stream{};
    cublasHandle_t handle{};
    std::vector<CUdeviceptr> allocations;
    ~Resources() {
        (void)cleanup();
    }
    bool cleanup() {
        if (!context) return true;
        bool passed = check(cuCtxSynchronize());
        if (handle) passed &= check_blas(cublasDestroy(handle));
        handle = nullptr;
        for (const auto pointer : allocations) passed &= check(cuMemFree(pointer));
        allocations.clear();
        if (stream) passed &= check(cuStreamDestroy(stream));
        stream = nullptr;
        passed &= check(cuCtxDestroy(context));
        context = nullptr;
        return passed;
    }
    bool allocate(CUdeviceptr& pointer, std::size_t bytes) {
        allocations.reserve(allocations.size() + 1);
        if (!check(cuMemAlloc(&pointer, bytes))) return false;
        allocations.push_back(pointer);
        return true;
    }
};

bool run(Resources& resources, int m, int n, int k, bool transpose_b, int mode,
         bool should_destroy = false, bool is_qualified = true) {
    constexpr std::size_t guard = 32;
    constexpr float sentinel = -12345.0F;
    std::vector<float> left(std::size_t{static_cast<unsigned int>(m)} * k, 1.0F);
    std::vector<float> right(std::size_t{static_cast<unsigned int>(n)} * k, 2.0F);
    std::vector<float> output(std::size_t{static_cast<unsigned int>(m)} * n + 2 * guard, sentinel);
    std::fill(output.begin() + guard, output.end() - guard, 3.0F);
    CUdeviceptr a{}, b{}, c{}, scalars{};
    std::array<float, 2> values{0.75F, 0.5F};
    if (!resources.allocate(a, left.size() * sizeof(float)) ||
        !resources.allocate(b, right.size() * sizeof(float)) ||
        !resources.allocate(c, output.size() * sizeof(float)) ||
        !resources.allocate(scalars, sizeof(values)) ||
        !check(cuMemcpyHtoD(a, left.data(), left.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(b, right.data(), right.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(c, output.data(), output.size() * sizeof(float))) ||
        !check(cuMemcpyHtoD(scalars, values.data(), sizeof(values))) ||
        !check_blas(cublasSetPointerMode(
            resources.handle, mode == 2 ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST)) ||
        !check_blas(cublasSetStream(resources.handle, mode == 3 ? resources.stream : nullptr)))
        return false;
    auto function = &cublasSgemm;
    if (mode == 1)
        function = reinterpret_cast<decltype(function)>(dlsym(RTLD_DEFAULT, "cublasSgemm_v2"));
    if (!function) return false;
    const float* alpha = mode == 2 ? std::bit_cast<const float*>(scalars) : &values[0];
    const float* beta =
        mode == 2 ? std::bit_cast<const float*>(scalars + sizeof(float)) : &values[1];
    std::cerr << "public_sgemm_begin expected=" << (is_qualified && mode < 2) << '\n';
    if (!check_blas(function(resources.handle, CUBLAS_OP_N, transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                             m, n, k, alpha, std::bit_cast<const float*>(a), m,
                             std::bit_cast<const float*>(b), transpose_b ? n : k, beta,
                             std::bit_cast<float*>(c + guard * sizeof(float)), m)))
        return false;
    std::cerr << "public_sgemm_end\n";
    if (should_destroy) {
        if (!check_blas(cublasDestroy(resources.handle))) return false;
        resources.handle = nullptr;
    } else if (!check(cuCtxSynchronize())) {
        return false;
    }
    if (!check(cuMemcpyDtoH(output.data(), c, output.size() * sizeof(float)))) return false;
    const auto expected = values[0] * static_cast<float>(k) * 2.0F + values[1] * 3.0F;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const bool is_guard = index < guard || index >= output.size() - guard;
        if (!std::isfinite(output[index]) || output[index] != (is_guard ? sentinel : expected)) {
            std::cerr << "public GEMM output mismatch at " << index << '\n';
            return false;
        }
    }
    std::cout << "public_sgemm m=" << m << " n=" << n << " k=" << k << " mode=" << mode
              << " numerical=1 guards=1\n";
    return true;
}
}  // namespace

int main() {
    Resources resources;
    CUdevice device{};
    if (!check(cuInit(0)) || !check(cuDeviceGet(&device, 0)) ||
        !check(cuCtxCreate(&resources.context, CU_CTX_MAP_HOST, device)) ||
        !check(cuStreamCreate(&resources.stream, CU_STREAM_NON_BLOCKING)) ||
        !check_blas(cublasCreate(&resources.handle)))
        return EXIT_FAILURE;
    for (int mode = 0; mode != 2; ++mode)
        if (!run(resources, 4096, 32, 1024, false, mode) ||
            !run(resources, 4096, 1024, 32, true, mode) ||
            !run(resources, 1024, 4096, 32, true, mode, mode == 1))
            return EXIT_FAILURE;
    if (!check_blas(cublasCreate(&resources.handle))) return EXIT_FAILURE;
    for (int mode = 0; mode != 2; ++mode) {
        for (const auto& shape : std::array<std::array<int, 4>, 6>{{{768, 768, 768, 0},
                                                                    {3072, 768, 768, 0},
                                                                    {768, 768, 3072, 0},
                                                                    {768, 768, 768, 1},
                                                                    {3072, 768, 768, 1},
                                                                    {768, 3072, 768, 1}}}) {
            if (!run(resources, shape[0], shape[1], shape[2], shape[3] != 0, mode))
                return EXIT_FAILURE;
        }
    }
    if (!run(resources, 768, 768, 768, false, 2) || !run(resources, 768, 768, 768, false, 3) ||
        !run(resources, 768, 3072, 768, false, 0, false, false) ||
        !run(resources, 768, 768, 2, false, 0, false, false))
        return EXIT_FAILURE;
    if (!run(resources, 4096, 32, 1024, false, 2) || !run(resources, 4096, 32, 1024, false, 3))
        return EXIT_FAILURE;
    // A handle created through the explicit SDK handle is deliberately unseen
    // by the preload wrapper. Missing provenance must stay on the original path.
    if (!check_blas(cublasDestroy(resources.handle))) return EXIT_FAILURE;
    resources.handle = nullptr;
    void* library = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_NOLOAD);
    if (!library) return EXIT_FAILURE;
    const auto native_create =
        reinterpret_cast<decltype(&cublasCreate)>(dlsym(library, "cublasCreate_v2"));
    const bool created = native_create && check_blas(native_create(&resources.handle));
    const bool released = dlclose(library) == 0;
    if (!created || !released || !run(resources, 4096, 32, 1024, false, 0, false, false) ||
        !resources.cleanup())
        return EXIT_FAILURE;
    std::cout << "public_sgemm cleanup=1\n";
    return EXIT_SUCCESS;
}
