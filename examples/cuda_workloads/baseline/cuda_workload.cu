#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <system_error>

namespace {

constexpr std::size_t kDefaultAllocationBytes = std::size_t{512} * 1024 * 1024;
constexpr unsigned int kDefaultIterations = 10;
constexpr unsigned int kThreadsPerBlock = 256;
constexpr unsigned int kMaximumBlocks = 65535;

struct Options {
    std::size_t allocation_bytes = kDefaultAllocationBytes;
    unsigned int iterations = kDefaultIterations;
    int device = 0;
    std::string output_path;
};

struct WorkloadResult {
    int device = -1;
    std::size_t requested_bytes = 0;
    std::size_t visible_total_before_bytes = 0;
    std::size_t visible_free_before_bytes = 0;
    std::size_t visible_free_after_alloc_bytes = 0;
    std::size_t visible_free_after_release_bytes = 0;
    float kernel_ms = 0.0F;
    cudaError_t device_status = cudaSuccess;
    cudaError_t initial_memory_info_status = cudaSuccess;
    cudaError_t allocation_status = cudaSuccess;
    cudaError_t allocation_memory_info_status = cudaSuccess;
    cudaError_t kernel_launch_status = cudaSuccess;
    cudaError_t synchronize_status = cudaSuccess;
    cudaError_t release_status = cudaSuccess;
    cudaError_t event_cleanup_status = cudaSuccess;
    cudaError_t final_memory_info_status = cudaSuccess;
};

__global__ void touch_buffer(unsigned char* buffer, std::size_t bytes, unsigned int iterations) {
    const std::size_t thread_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
        for (std::size_t index = thread_index; index < bytes; index += stride) {
            buffer[index] = static_cast<unsigned char>(iteration + index);
        }
    }
}

void print_usage(std::ostream& output) {
    output << "Usage: glimmer_cuda_workload [options]\n"
           << "  --bytes BYTES       device allocation size (default: 536870912)\n"
           << "  --iterations COUNT  kernel write repetitions (default: 10)\n"
           << "  --device INDEX      CUDA device index (default: 0)\n"
           << "  --output PATH       write CSV to PATH instead of stdout\n"
           << "  --help              show this message\n";
}

bool parse_unsigned(std::string_view value, std::uint64_t* result) {
    if (result == nullptr || value.empty()) {
        return false;
    }
    std::uint64_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return false;
    }
    *result = parsed;
    return true;
}

bool parse_options(int argc, char** argv, Options* options, bool* help_requested) {
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
        if (argument == "--output") {
            options->output_path = value;
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
            options->allocation_bytes = static_cast<std::size_t>(parsed);
        } else if (argument == "--iterations") {
            if (parsed == 0 || parsed > std::numeric_limits<unsigned int>::max()) {
                std::cerr << "--iterations is outside the supported range\n";
                return false;
            }
            options->iterations = static_cast<unsigned int>(parsed);
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

bool check_cuda(cudaError_t status, std::string_view operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << operation << " failed with cudaError " << static_cast<int>(status) << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
}

void write_csv_row(std::ostream& output, const WorkloadResult& result) {
    output << "workload,device,requested_bytes,visible_total_before_bytes,"
              "visible_free_before_bytes,visible_free_after_alloc_bytes,"
              "visible_free_after_release_bytes,kernel_ms,device_status,"
              "initial_memory_info_status,allocation_status,allocation_memory_info_status,"
              "kernel_launch_status,synchronize_status,release_status,event_cleanup_status,"
              "final_memory_info_status\n";
    output << "cuda_memory_kernel," << result.device << ',' << result.requested_bytes << ','
           << result.visible_total_before_bytes << ',' << result.visible_free_before_bytes << ','
           << result.visible_free_after_alloc_bytes << ','
           << result.visible_free_after_release_bytes << ',' << std::fixed << std::setprecision(3)
           << result.kernel_ms << ',' << static_cast<int>(result.device_status) << ','
           << static_cast<int>(result.initial_memory_info_status) << ','
           << static_cast<int>(result.allocation_status) << ','
           << static_cast<int>(result.allocation_memory_info_status) << ','
           << static_cast<int>(result.kernel_launch_status) << ','
           << static_cast<int>(result.synchronize_status) << ','
           << static_cast<int>(result.release_status) << ','
           << static_cast<int>(result.event_cleanup_status) << ','
           << static_cast<int>(result.final_memory_info_status) << '\n';
}

int run_workload(const Options& options, WorkloadResult* result) {
    if (result == nullptr) {
        return EXIT_FAILURE;
    }
    result->device = options.device;
    result->requested_bytes = options.allocation_bytes;

    result->device_status = cudaSetDevice(options.device);
    if (!check_cuda(result->device_status, "cudaSetDevice")) {
        return EXIT_FAILURE;
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    result->initial_memory_info_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    result->visible_total_before_bytes = total_bytes;
    result->visible_free_before_bytes = free_bytes;
    if (!check_cuda(result->initial_memory_info_status, "cudaMemGetInfo initial")) {
        return EXIT_FAILURE;
    }

    unsigned char* device_buffer = nullptr;
    result->allocation_status = cudaMalloc(&device_buffer, options.allocation_bytes);
    if (!check_cuda(result->allocation_status, "cudaMalloc")) {
        return EXIT_FAILURE;
    }

    result->allocation_memory_info_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    result->visible_free_after_alloc_bytes = free_bytes;
    check_cuda(result->allocation_memory_info_status, "cudaMemGetInfo allocated");

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    result->kernel_launch_status = cudaEventCreate(&start_event);
    check_cuda(result->kernel_launch_status, "cudaEventCreate start");
    if (result->kernel_launch_status == cudaSuccess) {
        result->kernel_launch_status = cudaEventCreate(&stop_event);
        check_cuda(result->kernel_launch_status, "cudaEventCreate stop");
    }
    if (result->kernel_launch_status == cudaSuccess) {
        const std::size_t block_count = options.allocation_bytes / kThreadsPerBlock +
                                        (options.allocation_bytes % kThreadsPerBlock == 0 ? 0 : 1);
        const unsigned int blocks = static_cast<unsigned int>(
            std::min<std::size_t>(block_count, static_cast<std::size_t>(kMaximumBlocks)));
        result->kernel_launch_status = cudaEventRecord(start_event);
        if (result->kernel_launch_status == cudaSuccess) {
            touch_buffer<<<blocks, kThreadsPerBlock>>>(device_buffer, options.allocation_bytes,
                                                       options.iterations);
            result->kernel_launch_status = cudaGetLastError();
        }
        if (result->kernel_launch_status == cudaSuccess) {
            result->kernel_launch_status = cudaEventRecord(stop_event);
        }
    }
    if (!check_cuda(result->kernel_launch_status, "kernel launch")) {
        result->synchronize_status = result->kernel_launch_status;
    } else {
        result->synchronize_status = cudaEventSynchronize(stop_event);
        if (result->synchronize_status == cudaSuccess) {
            result->synchronize_status =
                cudaEventElapsedTime(&result->kernel_ms, start_event, stop_event);
        }
        check_cuda(result->synchronize_status, "kernel synchronization");
    }

    if (stop_event != nullptr) {
        result->event_cleanup_status = cudaEventDestroy(stop_event);
        check_cuda(result->event_cleanup_status, "cudaEventDestroy stop");
    }
    if (start_event != nullptr) {
        const cudaError_t start_cleanup_status = cudaEventDestroy(start_event);
        if (result->event_cleanup_status == cudaSuccess) {
            result->event_cleanup_status = start_cleanup_status;
        }
        check_cuda(start_cleanup_status, "cudaEventDestroy start");
    }

    result->release_status = cudaFree(device_buffer);
    check_cuda(result->release_status, "cudaFree");
    result->final_memory_info_status = cudaMemGetInfo(&free_bytes, &total_bytes);
    result->visible_free_after_release_bytes = free_bytes;
    check_cuda(result->final_memory_info_status, "cudaMemGetInfo restored");

    return result->device_status == cudaSuccess &&
                   result->initial_memory_info_status == cudaSuccess &&
                   result->allocation_status == cudaSuccess &&
                   result->allocation_memory_info_status == cudaSuccess &&
                   result->kernel_launch_status == cudaSuccess &&
                   result->synchronize_status == cudaSuccess &&
                   result->release_status == cudaSuccess &&
                   result->event_cleanup_status == cudaSuccess &&
                   result->final_memory_info_status == cudaSuccess
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool help_requested = false;
    if (!parse_options(argc, argv, &options, &help_requested)) {
        return help_requested ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    WorkloadResult result;
    const int workload_status = run_workload(options, &result);
    if (options.output_path.empty()) {
        write_csv_row(std::cout, result);
        return workload_status;
    }

    std::ofstream output(options.output_path);
    if (!output) {
        std::cerr << "Could not open CSV output: " << options.output_path << '\n';
        return EXIT_FAILURE;
    }
    write_csv_row(output, result);
    return output.good() ? workload_status : EXIT_FAILURE;
}
