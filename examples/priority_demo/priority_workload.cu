#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

namespace {

constexpr std::size_t kDefaultBytes = 256 * 1024;
constexpr std::uint32_t kDefaultIterations = 20;
constexpr std::uint32_t kDefaultWorkUnits = 1;
constexpr std::uint32_t kThreadsPerBlock = 256;
constexpr unsigned int kGemmTileSize = 16;
constexpr std::size_t kDefaultInferenceMatrixSize = 1024;
constexpr std::size_t kDefaultTrainingMatrixSize = 2048;
constexpr auto kStartBarrierTimeout = std::chrono::seconds{30};

enum class Role : std::uint8_t {
    kInference,
    kTraining,
};

enum class KernelKind : std::uint8_t {
    kModel,
    kGemm,
    kPointwise,
};

struct Options {
    Role role = Role::kInference;
    KernelKind kernel = KernelKind::kModel;
    std::size_t bytes = kDefaultBytes;
    std::size_t matrix_size = 0;
    std::uint32_t iterations = kDefaultIterations;
    std::uint32_t work_units = kDefaultWorkUnits;
    std::uint32_t interval_us = 0;
    int device = 0;
    std::string output_path;
    std::string ready_path;
    std::string start_path;
};

using Clock = std::chrono::steady_clock;

__global__ void priority_kernel(float* buffer, std::size_t elements, std::uint32_t work_units) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }

    float value = buffer[index];
    for (std::uint32_t unit = 0; unit < work_units; ++unit) {
        value = value * 1.000001F + 0.000001F;
    }
    buffer[index] = value;
}

__global__ void initialize_matrix_kernel(float* matrix, std::size_t elements, float base_value) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        matrix[index] = base_value + static_cast<float>(index % 1024U) * 0.0001F;
    }
}

__global__ void parameter_update_kernel(float* matrix, std::size_t elements) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        matrix[index] *= 0.9999F;
    }
}

__global__ void gemm_kernel(const float* matrix_a, const float* matrix_b, float* matrix_c,
                            std::size_t matrix_size, std::uint32_t repeats) {
    __shared__ float tile_a[kGemmTileSize][kGemmTileSize];
    __shared__ float tile_b[kGemmTileSize][kGemmTileSize];

    const unsigned int local_row = threadIdx.y;
    const unsigned int local_column = threadIdx.x;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y) * kGemmTileSize + local_row;
    const std::size_t column = static_cast<std::size_t>(blockIdx.x) * kGemmTileSize + local_column;
    float value = 0.0F;

    for (std::uint32_t repeat = 0; repeat < repeats; ++repeat) {
        value = 0.0F;
        for (std::size_t tile_start = 0; tile_start < matrix_size; tile_start += kGemmTileSize) {
            const std::size_t a_column = tile_start + local_column;
            const std::size_t b_row = tile_start + local_row;
            tile_a[local_row][local_column] = row < matrix_size && a_column < matrix_size
                                                  ? matrix_a[row * matrix_size + a_column]
                                                  : 0.0F;
            tile_b[local_row][local_column] = b_row < matrix_size && column < matrix_size
                                                  ? matrix_b[b_row * matrix_size + column]
                                                  : 0.0F;
            __syncthreads();

            for (unsigned int index = 0; index < kGemmTileSize; ++index) {
                value += tile_a[local_row][index] * tile_b[index][local_column];
            }
            __syncthreads();
        }
    }

    if (row < matrix_size && column < matrix_size) {
        matrix_c[row * matrix_size + column] = value;
    }
}

[[nodiscard]] bool check_cuda(cudaError_t status, std::string_view operation);

[[nodiscard]] bool check_cublas(cublasStatus_t status, std::string_view operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    std::cerr << operation << " failed with cuBLAS status " << static_cast<int>(status) << '\n';
    return false;
}

[[nodiscard]] bool run_model_step(cublasHandle_t handle, Role role, int matrix_size,
                                  std::uint32_t steps, float* matrix_a, float* matrix_b,
                                  float* matrix_c, float* matrix_scratch, std::size_t element_count,
                                  unsigned int activation_blocks) {
    if (handle == nullptr || matrix_a == nullptr || matrix_b == nullptr || matrix_c == nullptr ||
        matrix_scratch == nullptr || steps == 0) {
        return false;
    }
    constexpr float kAlpha = 1.0F;
    constexpr float kBeta = 0.0F;
    for (std::uint32_t step = 0; step < steps; ++step) {
        if (role == Role::kInference) {
            // A compact MLP-like inference path: projection, activation, and
            // output projection. All operations remain on the default stream.
            if (!check_cublas(
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size, matrix_size,
                                matrix_size, &kAlpha, matrix_a, matrix_size, matrix_b, matrix_size,
                                &kBeta, matrix_scratch, matrix_size),
                    "inference projection")) {
                return false;
            }
            priority_kernel<<<activation_blocks, kThreadsPerBlock>>>(matrix_scratch, element_count,
                                                                     8);
            if (!check_cuda(cudaGetLastError(), "inference activation")) {
                return false;
            }
            if (!check_cublas(
                    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size, matrix_size,
                                matrix_size, &kAlpha, matrix_scratch, matrix_size, matrix_b,
                                matrix_size, &kBeta, matrix_c, matrix_size),
                    "inference output projection")) {
                return false;
            }
        } else {
            // A compact training step: forward projection, activation,
            // backward weight/input projections, and a parameter update.
            if (!check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size,
                                          matrix_size, matrix_size, &kAlpha, matrix_a, matrix_size,
                                          matrix_b, matrix_size, &kBeta, matrix_c, matrix_size),
                              "training forward")) {
                return false;
            }
            priority_kernel<<<activation_blocks, kThreadsPerBlock>>>(matrix_c, element_count, 8);
            if (!check_cuda(cudaGetLastError(), "training activation")) {
                return false;
            }
            if (!check_cublas(
                    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, matrix_size, matrix_size,
                                matrix_size, &kAlpha, matrix_a, matrix_size, matrix_c, matrix_size,
                                &kBeta, matrix_scratch, matrix_size),
                    "training backward weights")) {
                return false;
            }
            if (!check_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, matrix_size,
                                          matrix_size, matrix_size, &kAlpha, matrix_c, matrix_size,
                                          matrix_b, matrix_size, &kBeta, matrix_a, matrix_size),
                              "training backward inputs")) {
                return false;
            }
            parameter_update_kernel<<<activation_blocks, kThreadsPerBlock>>>(matrix_b,
                                                                             element_count);
            if (!check_cuda(cudaGetLastError(), "training parameter update")) {
                return false;
            }
        }
    }
    return true;
}

[[nodiscard]] std::string_view role_name(Role role) noexcept {
    return role == Role::kInference ? "inference" : "training";
}

[[nodiscard]] std::string_view kernel_name(KernelKind kernel) noexcept {
    switch (kernel) {
        case KernelKind::kModel:
            return "model";
        case KernelKind::kGemm:
            return "gemm";
        case KernelKind::kPointwise:
            return "pointwise";
    }
    return "unknown";
}

[[nodiscard]] std::uint64_t timestamp_us() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now().time_since_epoch())
            .count());
}

void print_usage(std::ostream& output) {
    output << "Usage: glimmer_cuda_priority_workload [options]\n"
           << "  --role inference|training  workload class (default: inference)\n"
           << "  --kernel model|gemm|pointwise CUDA workload profile (default: model)\n"
           << "  --bytes BYTES              pointwise device buffer size (default: 262144)\n"
           << "  --matrix-size ELEMENTS     square matrix dimension (default: role profile)\n"
           << "  --iterations COUNT         number of kernel launches (default: 20)\n"
           << "  --work-units COUNT         model steps, GEMM repetitions, or pointwise work\n"
           << "  --interval-us MICROSECONDS delay between launches (default: 0)\n"
           << "  --device INDEX             CUDA device index (default: 0)\n"
           << "  --output PATH              write trace CSV to PATH\n"
           << "  --ready-file PATH          signal CUDA initialization completion\n"
           << "  --start-file PATH          wait for a shared launch barrier\n"
           << "  --help                     show this message\n";
}

template <typename Integer>
[[nodiscard]] bool parse_unsigned(std::string_view value, Integer* result) {
    if (result == nullptr || value.empty()) {
        return false;
    }
    Integer parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return false;
    }
    *result = parsed;
    return true;
}

[[nodiscard]] bool parse_options(int argc, char** argv, Options* options, bool* help_requested) {
    if (options == nullptr || help_requested == nullptr) {
        return false;
    }
    *help_requested = false;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (argument == "--help") {
            print_usage(std::cout);
            *help_requested = true;
            return false;
        }
        if (index + 1 >= argc) {
            std::cerr << "Missing value for " << argument << '\n';
            return false;
        }
        const std::string_view value = argv[++index];
        if (argument == "--role") {
            if (value == "inference") {
                options->role = Role::kInference;
            } else if (value == "training") {
                options->role = Role::kTraining;
            } else {
                std::cerr << "Invalid role: " << value << '\n';
                return false;
            }
            continue;
        }
        if (argument == "--kernel") {
            if (value == "model") {
                options->kernel = KernelKind::kModel;
            } else if (value == "gemm") {
                options->kernel = KernelKind::kGemm;
            } else if (value == "pointwise") {
                options->kernel = KernelKind::kPointwise;
            } else {
                std::cerr << "Invalid kernel: " << value << '\n';
                return false;
            }
            continue;
        }
        if (argument == "--output") {
            options->output_path = std::string(value);
            continue;
        }
        if (argument == "--ready-file") {
            options->ready_path = std::string(value);
            continue;
        }
        if (argument == "--start-file") {
            options->start_path = std::string(value);
            continue;
        }

        std::uint64_t parsed = 0;
        if (!parse_unsigned(value, &parsed)) {
            std::cerr << "Invalid numeric value for " << argument << ": " << value << '\n';
            return false;
        }
        if (argument == "--bytes") {
            if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
                std::cerr << "--bytes is outside the supported range\n";
                return false;
            }
            options->bytes = static_cast<std::size_t>(parsed);
        } else if (argument == "--matrix-size") {
            if (parsed == 0 || parsed > std::numeric_limits<std::size_t>::max()) {
                std::cerr << "--matrix-size is outside the supported range\n";
                return false;
            }
            options->matrix_size = static_cast<std::size_t>(parsed);
        } else if (argument == "--iterations") {
            if (parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max()) {
                std::cerr << "--iterations is outside the supported range\n";
                return false;
            }
            options->iterations = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--work-units") {
            if (parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max()) {
                std::cerr << "--work-units is outside the supported range\n";
                return false;
            }
            options->work_units = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--interval-us") {
            if (parsed > std::numeric_limits<std::uint32_t>::max()) {
                std::cerr << "--interval-us is outside the supported range\n";
                return false;
            }
            options->interval_us = static_cast<std::uint32_t>(parsed);
        } else if (argument == "--device") {
            if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                std::cerr << "--device is outside the supported range\n";
                return false;
            }
            options->device = static_cast<int>(parsed);
        } else {
            std::cerr << "Unknown option: " << argument << '\n';
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool check_cuda(cudaError_t status, std::string_view operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed with cudaError " << static_cast<int>(status) << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
}

[[nodiscard]] bool write_header(std::ostream& output) {
    output << "role,iteration,launch_requested_us,launch_returned_us,completed_us,"
              "admission_wait_ms,kernel_ms,total_ms\n";
    return output.good();
}

[[nodiscard]] bool write_row(std::ostream& output, Role role, std::uint32_t iteration,
                             std::uint64_t launch_requested, std::uint64_t launch_returned,
                             std::uint64_t completed, float kernel_ms) {
    const double admission_wait_ms =
        static_cast<double>(launch_returned - launch_requested) / 1000.0;
    const double total_ms = static_cast<double>(completed - launch_requested) / 1000.0;
    output << role_name(role) << ',' << iteration << ',' << launch_requested << ','
           << launch_returned << ',' << completed << ',' << std::fixed << std::setprecision(3)
           << admission_wait_ms << ',' << kernel_ms << ',' << total_ms << '\n';
    return output.good();
}

int run_workload(const Options& options, std::ostream& output) {
    if (!check_cuda(cudaSetDevice(options.device), "cudaSetDevice")) {
        return EXIT_FAILURE;
    }

    float* device_buffer = nullptr;
    float* matrix_a = nullptr;
    float* matrix_b = nullptr;
    float* matrix_c = nullptr;
    float* matrix_scratch = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    const bool use_model = options.kernel == KernelKind::kModel;
    const bool use_gemm = options.kernel == KernelKind::kGemm;
    const bool use_matrix_workload = use_model || use_gemm;
    const std::size_t matrix_size =
        options.matrix_size != 0 ? options.matrix_size
                                 : (options.role == Role::kTraining ? kDefaultTrainingMatrixSize
                                                                    : kDefaultInferenceMatrixSize);
    std::size_t elements = 0;
    std::size_t allocation_bytes = options.bytes;
    if (use_matrix_workload) {
        if (matrix_size > std::numeric_limits<std::size_t>::max() / matrix_size) {
            std::cerr << "--matrix-size is too large\n";
            return EXIT_FAILURE;
        }
        elements = matrix_size * matrix_size;
        if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            std::cerr << "--matrix-size allocation is too large\n";
            return EXIT_FAILURE;
        }
        allocation_bytes = elements * sizeof(float);
        if (matrix_size > std::numeric_limits<int>::max() ||
            matrix_size > std::numeric_limits<unsigned int>::max()) {
            std::cerr << "--matrix-size exceeds the CUDA grid limit\n";
            return EXIT_FAILURE;
        }
        if (!check_cuda(cudaMalloc(&matrix_a, allocation_bytes), "cudaMalloc(matrix_a)")) {
            return EXIT_FAILURE;
        }
        if (!check_cuda(cudaMalloc(&matrix_b, allocation_bytes), "cudaMalloc(matrix_b)")) {
            static_cast<void>(cudaFree(matrix_a));
            return EXIT_FAILURE;
        }
        if (!check_cuda(cudaMalloc(&matrix_c, allocation_bytes), "cudaMalloc(matrix_c)")) {
            static_cast<void>(cudaFree(matrix_b));
            static_cast<void>(cudaFree(matrix_a));
            return EXIT_FAILURE;
        }
        if (use_model && !check_cuda(cudaMalloc(&matrix_scratch, allocation_bytes),
                                     "cudaMalloc(matrix_scratch)")) {
            static_cast<void>(cudaFree(matrix_c));
            static_cast<void>(cudaFree(matrix_b));
            static_cast<void>(cudaFree(matrix_a));
            return EXIT_FAILURE;
        }
    } else {
        elements = options.bytes / sizeof(float);
        if (elements == 0) {
            std::cerr << "--bytes must provide at least one float element\n";
            return EXIT_FAILURE;
        }
        if (!check_cuda(cudaMalloc(&device_buffer, options.bytes), "cudaMalloc")) {
            return EXIT_FAILURE;
        }
    }

    const auto cleanup_device = [&]() {
        cudaError_t cleanup_status = cudaSuccess;
        if (cublas_handle != nullptr) {
            const cublasStatus_t status = cublasDestroy(cublas_handle);
            if (cleanup_status == cudaSuccess && status != CUBLAS_STATUS_SUCCESS) {
                cleanup_status = cudaErrorUnknown;
            }
            cublas_handle = nullptr;
        }
        const auto free_pointer = [&cleanup_status](float*& pointer) {
            if (pointer == nullptr) {
                return;
            }
            const cudaError_t status = cudaFree(pointer);
            if (cleanup_status == cudaSuccess) {
                cleanup_status = status;
            }
            pointer = nullptr;
        };
        free_pointer(matrix_c);
        free_pointer(matrix_scratch);
        free_pointer(matrix_b);
        free_pointer(matrix_a);
        free_pointer(device_buffer);
        return cleanup_status;
    };

    if (use_matrix_workload) {
        if (!check_cuda(cudaMemset(matrix_a, 0, allocation_bytes), "cudaMemset(matrix_a)")) {
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
        if (!check_cuda(cudaMemset(matrix_b, 0, allocation_bytes), "cudaMemset(matrix_b)")) {
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
        if (!check_cuda(cudaMemset(matrix_c, 0, allocation_bytes), "cudaMemset(matrix_c)")) {
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
        if (use_model && !check_cuda(cudaMemset(matrix_scratch, 0, allocation_bytes),
                                     "cudaMemset(matrix_scratch)")) {
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
    } else if (!check_cuda(cudaMemset(device_buffer, 0, allocation_bytes), "cudaMemset")) {
        static_cast<void>(cleanup_device());
        return EXIT_FAILURE;
    }

    if (use_model && !check_cublas(cublasCreate(&cublas_handle), "cublasCreate")) {
        static_cast<void>(cleanup_device());
        return EXIT_FAILURE;
    }

    if (!write_header(output)) {
        std::cerr << "Unable to write workload trace header\n";
        static_cast<void>(cleanup_device());
        return EXIT_FAILURE;
    }
    if (!options.ready_path.empty()) {
        std::ofstream ready_file(options.ready_path);
        if (!ready_file.is_open()) {
            std::cerr << "Unable to open ready file: " << options.ready_path << '\n';
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
        ready_file << "ready\n";
        if (!ready_file.good()) {
            std::cerr << "Unable to write ready file: " << options.ready_path << '\n';
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
    }
    if (!options.start_path.empty()) {
        const auto start_wait_begin = Clock::now();
        for (;;) {
            std::ifstream start_file(options.start_path);
            if (start_file.good()) {
                break;
            }
            if (Clock::now() - start_wait_begin >= kStartBarrierTimeout) {
                std::cerr << "Timed out waiting for start file: " << options.start_path << '\n';
                static_cast<void>(cleanup_device());
                return EXIT_FAILURE;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    dim3 grid;
    dim3 block;
    const std::size_t activation_block_count = (elements - 1) / kThreadsPerBlock + 1;
    if (activation_block_count > std::numeric_limits<unsigned int>::max()) {
        std::cerr << "workload requests too many CUDA blocks\n";
        static_cast<void>(cleanup_device());
        return EXIT_FAILURE;
    }
    const unsigned int activation_blocks = static_cast<unsigned int>(activation_block_count);
    if (use_gemm) {
        const std::size_t block_count =
            (matrix_size - 1) / static_cast<std::size_t>(kGemmTileSize) + 1;
        if (block_count > std::numeric_limits<unsigned int>::max()) {
            std::cerr << "--matrix-size requests too many CUDA blocks\n";
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
        const unsigned int blocks = static_cast<unsigned int>(block_count);
        grid = dim3(blocks, blocks, 1);
        block = dim3(kGemmTileSize, kGemmTileSize, 1);
    } else {
        grid = dim3(activation_blocks, 1, 1);
        block = dim3(kThreadsPerBlock, 1, 1);
    }

    if (use_model) {
        initialize_matrix_kernel<<<activation_blocks, kThreadsPerBlock>>>(matrix_a, elements,
                                                                          0.001F);
        initialize_matrix_kernel<<<activation_blocks, kThreadsPerBlock>>>(matrix_b, elements,
                                                                          0.002F);
        initialize_matrix_kernel<<<activation_blocks, kThreadsPerBlock>>>(matrix_c, elements,
                                                                          0.003F);
        initialize_matrix_kernel<<<activation_blocks, kThreadsPerBlock>>>(matrix_scratch, elements,
                                                                          0.004F);
        if (!check_cuda(cudaGetLastError(), "model input initialization") ||
            !check_cuda(cudaDeviceSynchronize(), "model input synchronization")) {
            static_cast<void>(cleanup_device());
            return EXIT_FAILURE;
        }
    }

    bool success = true;
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        const std::uint64_t launch_requested = timestamp_us();
        cudaError_t status = cudaEventCreate(&start_event);
        if (status == cudaSuccess) {
            status = cudaEventCreate(&stop_event);
        }
        if (status == cudaSuccess) {
            status = cudaEventRecord(start_event);
        }
        if (status == cudaSuccess) {
            if (use_model) {
                if (!run_model_step(cublas_handle, options.role, static_cast<int>(matrix_size),
                                    options.work_units, matrix_a, matrix_b, matrix_c,
                                    matrix_scratch, elements, activation_blocks)) {
                    status = cudaErrorUnknown;
                }
            } else if (use_gemm) {
                gemm_kernel<<<grid, block>>>(matrix_a, matrix_b, matrix_c, matrix_size,
                                             options.work_units);
                status = cudaGetLastError();
            } else {
                priority_kernel<<<grid, block>>>(device_buffer, elements, options.work_units);
                status = cudaGetLastError();
            }
        }
        const std::uint64_t launch_returned = timestamp_us();
        float kernel_ms = 0.0F;
        if (status == cudaSuccess) {
            status = cudaEventRecord(stop_event);
        }
        if (status == cudaSuccess) {
            status = cudaEventSynchronize(stop_event);
        }
        if (status == cudaSuccess) {
            status = cudaEventElapsedTime(&kernel_ms, start_event, stop_event);
        }
        const std::uint64_t completed = timestamp_us();
        if (start_event != nullptr) {
            const cudaError_t cleanup_status = cudaEventDestroy(start_event);
            if (status == cudaSuccess) {
                status = cleanup_status;
            }
        }
        if (stop_event != nullptr) {
            const cudaError_t cleanup_status = cudaEventDestroy(stop_event);
            if (status == cudaSuccess) {
                status = cleanup_status;
            }
        }
        if (!check_cuda(status, "priority workload iteration")) {
            success = false;
            break;
        }
        if (!write_row(output, options.role, iteration, launch_requested, launch_returned,
                       completed, kernel_ms)) {
            std::cerr << "Unable to write workload trace row\n";
            success = false;
            break;
        }
        output.flush();
        if (options.interval_us != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(options.interval_us));
        }
    }

    const cudaError_t cleanup_status = cleanup_device();
    if (!check_cuda(cleanup_status, "cudaFree")) {
        success = false;
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool help_requested = false;
    if (!parse_options(argc, argv, &options, &help_requested)) {
        return help_requested ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    std::ofstream file;
    std::ostream* output = &std::cout;
    if (!options.output_path.empty()) {
        file.open(options.output_path);
        if (!file.is_open()) {
            std::cerr << "Unable to open output file: " << options.output_path << '\n';
            return EXIT_FAILURE;
        }
        output = &file;
    }

    const int status = run_workload(options, *output);
    if (status == EXIT_SUCCESS) {
        std::cerr << "priority_workload role=" << role_name(options.role)
                  << " kernel=" << kernel_name(options.kernel)
                  << " status=ok iterations=" << options.iterations << '\n';
    }
    return status;
}
