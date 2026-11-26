#include <cuda.h>
#include <cublasLt.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <poll.h>
#include <unistd.h>

namespace {
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status, unsigned int line = __builtin_LINE()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("Driver " + std::to_string(status) +
                                 "; line=" + std::to_string(line));
}
void check(cublasStatus_t status, unsigned int line = __builtin_LINE()) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Lt " + std::to_string(status) + "; line=" + std::to_string(line));
}
constexpr std::size_t kGuard = 32;
constexpr float kSentinel = -8193.25F;
constexpr int kDepth = 8192;

struct Buffer {
    std::vector<float> expected;
    CUdeviceptr device{};
    explicit Buffer(std::size_t elements) : expected(elements + 2 * kGuard, kSentinel) {
        check(cuMemAlloc(&device, expected.size() * sizeof(float)));
    }
    void upload() const {
        check(cuMemcpyHtoD(device, expected.data(), expected.size() * sizeof(float)));
    }
    CUdeviceptr address() const {
        return device + kGuard * sizeof(float);
    }
    std::vector<float> download() const {
        std::vector<float> actual(expected.size());
        check(cuMemcpyDtoH(actual.data(), device, actual.size() * sizeof(float)));
        return actual;
    }
    void verify_unchanged(const char* name) const {
        const auto actual = download();
        if (actual != expected)
            throw std::runtime_error(std::string(name) + " input or allocation guard changed");
    }
    void release() {
        check(cuMemFree(device));
        device = 0;
    }
};

void await_run() {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::seconds(30);
    std::array<char, 4> command{};
    std::size_t received{};
    while (received != command.size()) {
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now());
        if (remaining.count() <= 0) throw std::runtime_error("bounded Lt workload start barrier");
        pollfd input{STDIN_FILENO, POLLIN, 0};
        const auto ready = ::poll(&input, 1, static_cast<int>(remaining.count()));
        if (ready < 0 && errno == EINTR) continue;
        if (ready != 1 || !(input.revents & (POLLIN | POLLHUP)))
            throw std::runtime_error("bounded Lt workload start barrier");
        const auto bytes =
            ::read(STDIN_FILENO, command.data() + received, command.size() - received);
        if (bytes < 0 && errno == EINTR) continue;
        if (bytes <= 0) throw std::runtime_error("incomplete Lt workload start command");
        received += static_cast<std::size_t>(bytes);
    }
    expect(command == std::array{'r', 'u', 'n', '\n'}, "invalid Lt workload start command");
}

cublasLtMatmulAlgo_t select_algorithm(cublasLtHandle_t handle, cublasLtMatmulDesc_t operation,
                                      const std::array<cublasLtMatrixLayout_t, 3>& layouts,
                                      cublasLtMatmulPreference_t preference,
                                      std::size_t workspace_bytes) {
    std::array<cublasLtMatmulHeuristicResult_t, 64> algorithms{};
    int count{};
    check(cublasLtMatmulAlgoGetHeuristic(
        handle, operation, layouts[0], layouts[1], layouts[2], layouts[2], preference,
        static_cast<int>(algorithms.size()), algorithms.data(), &count));
    expect(count > 0 && static_cast<std::size_t>(count) <= algorithms.size(),
           "bounded original Lt heuristic inventory");
    for (int index = 0; index < count; ++index) {
        const auto& candidate = algorithms[static_cast<std::size_t>(index)];
        if (candidate.state != CUBLAS_STATUS_SUCCESS || candidate.workspaceSize > workspace_bytes)
            continue;
        std::uint32_t split{}, reduction{};
        std::size_t written{};
        check(cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                   &split, sizeof(split), &written));
        expect(written == sizeof(split), "complete original Lt split count");
        check(cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,
                                                   CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                   &reduction, sizeof(reduction), &written));
        expect(written == sizeof(reduction), "complete original Lt reduction scheme");
        if (split > 1 || reduction != 0) continue;
        cublasLtMatmulHeuristicResult_t validated{};
        const auto status =
            cublasLtMatmulAlgoCheck(handle, operation, layouts[0], layouts[1], layouts[2],
                                    layouts[2], &candidate.algo, &validated);
        if (status != CUBLAS_STATUS_SUCCESS || validated.state != CUBLAS_STATUS_SUCCESS ||
            validated.workspaceSize > workspace_bytes)
            continue;
        std::cout << "Ordinary Lt algorithm selected: index=" << index << "; split=" << split
                  << "; reduction=" << reduction << "; workspace_bytes=" << validated.workspaceSize
                  << '\n';
        return candidate.algo;
    }
    throw std::runtime_error("no checked original direct-reduction Lt algorithm");
}

// NOLINTBEGIN(performance-no-int-to-ptr): Driver and Lt use distinct device pointer ABIs.
void run(bool is_device_tool, bool uses_device_scalars) {
    const int dimension = is_device_tool ? 2048 : 4096;
    const auto input_elements = static_cast<std::size_t>(dimension) * kDepth;
    const auto output_elements = static_cast<std::size_t>(dimension) * dimension;
    constexpr std::size_t workspace_bytes = std::size_t{8} * 1024 * 1024;
    constexpr std::size_t workspace_guard = 256;
    check(cuInit(0));
    CUdevice device{};
    CUcontext context{};
    CUstream stream{};
    cublasLtHandle_t handle{};
    check(cuDeviceGet(&device, 0));
    check(cuCtxCreate(&context, 0, device));
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cublasLtCreate(&handle));
    Buffer a(input_elements), b(input_elements), output(output_elements),
        bias(static_cast<std::size_t>(dimension)), scalars(2);
    for (std::size_t index = 0; index < input_elements; ++index) {
        a.expected[kGuard + index] = static_cast<float>(index % dimension % 2 + 1);
        b.expected[kGuard + index] = static_cast<float>(index / kDepth % 3 + 1);
    }
    std::fill_n(output.expected.data() + kGuard, output_elements, 6.0F);
    for (int row = 0; row < dimension; ++row)
        bias.expected[kGuard + static_cast<std::size_t>(row)] = static_cast<float>(row % 4) * 0.25F;
    const std::array<float, 2> coefficients{0.125F, 0.5F};
    std::array<float, 2 * kGuard + 2> host_coefficients{};
    host_coefficients.fill(kSentinel);
    std::copy(coefficients.begin(), coefficients.end(), host_coefficients.data() + kGuard);
    const auto expected_host_coefficients = host_coefficients;
    std::copy(coefficients.begin(), coefficients.end(), scalars.expected.data() + kGuard);
    for (const auto* buffer : {&a, &b, &output, &bias, &scalars}) buffer->upload();
    CUdeviceptr workspace{};
    std::vector<std::byte> workspace_image(workspace_bytes + 2 * workspace_guard, std::byte{0x5a});
    check(cuMemAlloc(&workspace, workspace_image.size()));
    check(cuMemcpyHtoD(workspace, workspace_image.data(), workspace_image.size()));

    std::array<cublasLtMatrixLayout_t, 3> layouts{};
    check(cublasLtMatrixLayoutCreate(&layouts[0], CUDA_R_32F, dimension, kDepth, dimension));
    check(cublasLtMatrixLayoutCreate(&layouts[1], CUDA_R_32F, kDepth, dimension, kDepth));
    check(cublasLtMatrixLayoutCreate(&layouts[2], CUDA_R_32F, dimension, dimension, dimension));
    constexpr auto order = CUBLASLT_ORDER_COL;
    for (const auto layout : layouts)
        check(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                               sizeof(order)));
    cublasLtMatmulDesc_t operation{};
    cublasLtMatmulPreference_t preference{};
    check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    constexpr auto transpose = CUBLAS_OP_N;
    constexpr auto epilogue = CUBLASLT_EPILOGUE_BIAS;
    constexpr auto bias_type = CUDA_R_32F;
    const auto pointer_mode =
        uses_device_scalars ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
    const auto* bias_pointer = reinterpret_cast<const void*>(bias.address());
    for (const auto attribute :
         {CUBLASLT_MATMUL_DESC_TRANSA, CUBLASLT_MATMUL_DESC_TRANSB, CUBLASLT_MATMUL_DESC_TRANSC})
        check(cublasLtMatmulDescSetAttribute(operation, attribute, &transpose, sizeof(transpose)));
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                         sizeof(epilogue)));
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                         sizeof(bias_type)));
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                         &bias_pointer, sizeof(bias_pointer)));
    check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                         &pointer_mode, sizeof(pointer_mode)));
    check(cublasLtMatmulPreferenceCreate(&preference));
    check(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)));
    // Head guards preserve 128-byte matrix alignment, not the allocation's
    // stronger alignment. Tell the original heuristic the actual guarantee.
    constexpr std::uint32_t alignment_bytes = kGuard * sizeof(float);
    for (const auto attribute :
         {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES})
        check(cublasLtMatmulPreferenceSetAttribute(preference, attribute, &alignment_bytes,
                                                   sizeof(alignment_bytes)));
    const auto algorithm =
        select_algorithm(handle, operation, layouts, preference, workspace_bytes);
    // Pageable uploads are not ordered by this nonblocking computation stream.
    check(cuCtxSynchronize());
    std::cout << "Ordinary Lt priority ready: role=low; pid=" << ::getpid()
              << "; tiles=" << (dimension / 64) * (dimension / 64) << "; dimension=" << dimension
              << '\n'
              << std::flush;
    await_run();
    const auto* alpha = uses_device_scalars ? reinterpret_cast<const void*>(scalars.address())
                                            : host_coefficients.data() + kGuard;
    const auto* beta = uses_device_scalars
                           ? reinterpret_cast<const void*>(scalars.address() + sizeof(float))
                           : host_coefficients.data() + kGuard + 1;
    auto* destination = reinterpret_cast<void*>(output.address());
    check(cublasLtMatmul(handle, operation, alpha, reinterpret_cast<const void*>(a.address()),
                         layouts[0], reinterpret_cast<const void*>(b.address()), layouts[1], beta,
                         destination, layouts[2], destination, layouts[2], &algorithm,
                         reinterpret_cast<void*>(workspace + workspace_guard), workspace_bytes,
                         reinterpret_cast<cudaStream_t>(stream)));
    check(cuStreamSynchronize(stream));
    const auto actual = output.download();
    for (std::size_t index = 0; index < output_elements; ++index) {
        const auto row = index % static_cast<std::size_t>(dimension);
        const auto column = index / static_cast<std::size_t>(dimension);
        const float expected =
            0.125F * kDepth * static_cast<float>(row % 2 + 1) * static_cast<float>(column % 3 + 1) +
            3.0F + static_cast<float>(row % 4) * 0.25F;
        if (actual[kGuard + index] != expected)
            throw std::runtime_error("Lt once-only beta/bias mismatch at " + std::to_string(index) +
                                     ": actual=" + std::to_string(actual[kGuard + index]) +
                                     "; expected=" + std::to_string(expected));
    }
    for (std::size_t index = 0; index < kGuard; ++index)
        expect(actual[index] == kSentinel && actual[kGuard + output_elements + index] == kSentinel,
               "Lt output allocation guards remain unchanged");
    a.verify_unchanged("A");
    b.verify_unchanged("B");
    bias.verify_unchanged("bias");
    scalars.verify_unchanged("scalars");
    expect(host_coefficients == expected_host_coefficients,
           "actual host scalar values and allocation guards remain unchanged");
    std::array<std::byte, workspace_guard> guard{};
    for (const auto offset : {std::size_t{0}, workspace_guard + workspace_bytes}) {
        check(cuMemcpyDtoH(guard.data(), workspace + offset, guard.size()));
        expect(std::ranges::all_of(guard, [](auto byte) { return byte == std::byte{0x5a}; }),
               "Lt workspace allocation guards remain unchanged");
    }
    check(cublasLtMatmulPreferenceDestroy(preference));
    check(cublasLtMatmulDescDestroy(operation));
    for (const auto layout : layouts) check(cublasLtMatrixLayoutDestroy(layout));
    check(cublasLtDestroy(handle));
    for (auto* buffer : {&a, &b, &output, &bias, &scalars}) buffer->release();
    check(cuMemFree(workspace));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    std::cout << "Ordinary Lt priority verified: role=low; values=" << actual.size()
              << "; input_values=" << a.expected.size() + b.expected.size()
              << "; bias_values=" << bias.expected.size()
              << "; beta_once=true; guards=true; in_place=true; device_scalars=" << std::boolalpha
              << uses_device_scalars
              << "; full_device_qualified=false; performance_qualified=false\n";
}
// NOLINTEND(performance-no-int-to-ptr)
}  // namespace

int main(int argc, char** argv) {
    try {
        bool is_device_tool{}, uses_device_scalars{};
        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--device-tool")
                is_device_tool = true;
            else if (option == "--device-scalars")
                uses_device_scalars = true;
            else
                throw std::runtime_error("unknown Lt priority option");
        }
        run(is_device_tool, uses_device_scalars);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
