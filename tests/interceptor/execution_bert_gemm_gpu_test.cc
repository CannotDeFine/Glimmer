#include <cuda.h>
#include <cublas_v2.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
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
constexpr int kRows = 768, kColumns = 768, kDepth = 3072;
constexpr std::size_t kGuard = 64;
constexpr float kSentinel = 777.0F;

// This is the same ordinary SDK application in native and intercepted modes.
// No framework API or command-line switch selects the private provider.
void run() {
    check(cuInit(0));
    CUdevice device{};
    CUcontext context{};
    CUstream stream{};
    cublasHandle_t handle{};
    check(cuDeviceGet(&device, 0));
    check(cuCtxCreate(&context, 0, device));
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cublasCreate_v2(&handle));
    check(cublasSetStream_v2(handle, reinterpret_cast<cudaStream_t>(stream)));
    check(cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST));
    check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    int version{};
    check(cublasGetVersion_v2(handle, &version));
    expect(version == 120604, "this reduction-order diagnostic requires the pinned SDK");

    const auto a_elements = static_cast<std::size_t>(kRows) * kDepth;
    const auto b_elements = static_cast<std::size_t>(kDepth) * kColumns;
    const auto c_elements = static_cast<std::size_t>(kRows) * kColumns;
    std::vector<float> a(a_elements + 2 * kGuard, kSentinel);
    std::vector<float> b(b_elements + 2 * kGuard, kSentinel);
    std::vector<float> initial(c_elements + 2 * kGuard, kSentinel);
    for (std::size_t index = 0; index < a_elements; ++index) a[kGuard + index] = 0.0F;
    for (int row = 0; row < kRows; ++row) {
        const auto x = static_cast<std::size_t>(row);
        a[kGuard + x] = 100000000.0F;
        a[kGuard + static_cast<std::size_t>(768) * kRows + x] = -100000000.0F;
        a[kGuard + static_cast<std::size_t>(769) * kRows + x] = static_cast<float>(1 + row % 2);
        a[kGuard + static_cast<std::size_t>(1536) * kRows + x] = static_cast<float>(2 + row % 5);
        a[kGuard + static_cast<std::size_t>(2304) * kRows + x] = -0.5F;
    }
    for (int column = 0; column < kColumns; ++column) {
        const auto start = kGuard + static_cast<std::size_t>(column) * kDepth;
        for (int k = 0; k < kDepth; ++k)
            b[start + static_cast<std::size_t>(k)] = column % 2 ? -1.0F : 1.0F;
    }
    // A single continuous accumulator retains the 1/2 term after cancellation;
    // resetting the second partition loses it before partials are combined.
    // The last two partitions leave a nonzero exact, row-dependent result.
    for (std::size_t index = 0; index < c_elements; ++index)
        initial[kGuard + index] = std::numeric_limits<float>::quiet_NaN();
    std::array<CUdeviceptr, 3> memory{};
    const std::array sizes{a.size(), b.size(), initial.size()};
    for (std::size_t index = 0; index < memory.size(); ++index)
        check(cuMemAlloc(&memory[index], sizes[index] * sizeof(float)));
    check(cuMemcpyHtoD(memory[0], a.data(), a.size() * sizeof(float)));
    check(cuMemcpyHtoD(memory[1], b.data(), b.size() * sizeof(float)));
    check(cuCtxSynchronize());
    // NOLINTBEGIN(performance-no-int-to-ptr): Driver and cuBLAS use distinct device pointer ABIs.
    const auto* device_a = reinterpret_cast<const float*>(memory[0] + kGuard * sizeof(float));
    const auto* device_b = reinterpret_cast<const float*>(memory[1] + kGuard * sizeof(float));
    auto* device_c = reinterpret_cast<float*>(memory[2] + kGuard * sizeof(float));
    // NOLINTEND(performance-no-int-to-ptr)
    for (const bool extended : {false, true}) {
        check(cuMemcpyHtoD(memory[2], initial.data(), initial.size() * sizeof(float)));
        check(cuCtxSynchronize());
        const float alpha = 1.0F, beta = 0.0F;
        if (extended)
            check(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, kRows, kColumns, kDepth, &alpha,
                               device_a, CUDA_R_32F, kRows, device_b, CUDA_R_32F, kDepth, &beta,
                               device_c, CUDA_R_32F, kRows, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT));
        else
            check(cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, kRows, kColumns, kDepth, &alpha,
                                 device_a, kRows, device_b, kDepth, &beta, device_c, kRows));
        check(cuStreamSynchronize(stream));
        std::vector<float> output(initial.size());
        check(cuMemcpyDtoH(output.data(), memory[2], output.size() * sizeof(float)));
        for (int column = 0; column < kColumns; ++column) {
            for (int row = 0; row < kRows; ++row) {
                const float oracle = static_cast<float>(2 + row % 5) - 0.5F;
                const float expected = column % 2 ? -oracle : oracle;
                const auto offset = kGuard + static_cast<std::size_t>(column) * kRows +
                                    static_cast<std::size_t>(row);
                if (output[offset] != expected)
                    throw std::runtime_error(std::string(extended ? "GemmEx" : "SGEMM") +
                                             " reduction mismatch at " + std::to_string(row) + "," +
                                             std::to_string(column) +
                                             ": actual=" + std::to_string(output[offset]) +
                                             ", expected=" + std::to_string(expected));
            }
        }
        for (std::size_t guard = 0; guard < kGuard; ++guard)
            expect(output[guard] == kSentinel && output[kGuard + c_elements + guard] == kSentinel,
                   "complete output allocation guards remain unchanged");
        std::vector<float> actual_a(a.size()), actual_b(b.size());
        check(cuMemcpyDtoH(actual_a.data(), memory[0], actual_a.size() * sizeof(float)));
        check(cuMemcpyDtoH(actual_b.data(), memory[1], actual_b.size() * sizeof(float)));
        expect(actual_a == a && actual_b == b, "all input elements and guards remain unchanged");
        std::cout << "BERT input-gradient diagnostic verified: api="
                  << (extended ? "GemmEx" : "SGEMM") << "; values=" << c_elements
                  << "; partition_depth=768; partitions=4; guards=true; beta_zero_nan=true; "
                     "full_device_qualified=false; priority_qualified=false\n";
    }
    check(cublasDestroy_v2(handle));
    for (const auto allocation : memory) check(cuMemFree(allocation));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
}
}  // namespace

int main() {
    try {
        run();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
