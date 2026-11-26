#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}
void copy_chunks(float* destination, const float* source, std::size_t count, cudaMemcpyKind kind) {
    constexpr std::size_t chunk = std::size_t{1024} * 1024;
    for (std::size_t offset = 0; offset < count; offset += chunk)
        check(cudaMemcpy(destination + offset, source + offset,
                         std::min(chunk, count - offset) * sizeof(float), kind),
              "copy chunk");
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 2 ||
        (std::string_view{argv[1]} != "--low" && std::string_view{argv[1]} != "--high"))
        return 2;
    const bool low = std::string_view{argv[1]} == "--low";
    const int side = low ? 4096 : 256;
    const std::size_t elements = static_cast<std::size_t>(side) * side;
    std::vector<float> input(elements, 1.0F / 64.0F), output(elements, 1.0F);
    check(cudaSetDevice(0), "select device");
    cublasHandle_t handle{};
    check(cublasCreate(&handle), "create cuBLAS handle");
    check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH), "set cuBLAS math mode");
    float *a{}, *b{}, *c{};
    check(cudaMalloc(&a, elements * sizeof(float)), "allocate A");
    check(cudaMalloc(&b, elements * sizeof(float)), "allocate B");
    check(cudaMalloc(&c, elements * sizeof(float)), "allocate C");
    copy_chunks(a, input.data(), elements, cudaMemcpyHostToDevice);
    copy_chunks(b, input.data(), elements, cudaMemcpyHostToDevice);
    copy_chunks(c, output.data(), elements, cudaMemcpyHostToDevice);
    std::printf("READY %s\n", low ? "low" : "high");
    std::fflush(stdout);
    char line[16]{};
    if (!std::fgets(line, sizeof(line), stdin) || std::string_view{line} != "run\n")
        throw std::runtime_error("missing explicit execution trigger");
    const float alpha = 1.0F, beta = 0.5F;
    check(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, side, side, side, &alpha, a, side, b, side,
                      &beta, c, side),
          "submit GEMM");
    check(cudaDeviceSynchronize(), "synchronize GEMM");
    copy_chunks(output.data(), c, elements, cudaMemcpyDeviceToHost);
    const float expected = static_cast<float>(side) / 4096.0F + 0.5F;
    if (!std::ranges::all_of(output, [expected](float value) {
            return std::isfinite(value) && std::abs(value - expected) <= 1.0e-5F;
        }))
        throw std::runtime_error("complete output mismatch");
    std::printf("VERIFIED %s %zu\n", low ? "low" : "high", elements);
    std::fflush(stdout);
    check(cudaFree(c), "release C");
    check(cudaFree(b), "release B");
    check(cudaFree(a), "release A");
    check(cublasDestroy(handle), "release cuBLAS handle");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "GEMM client failed: %s\n", error.what());
    return 1;
}
