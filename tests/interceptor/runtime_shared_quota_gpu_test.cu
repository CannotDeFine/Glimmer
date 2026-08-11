#include <cuda_runtime.h>

#include "glimmer/control/shared_memory_quota.h"

#include <charconv>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr std::size_t kQuotaBytes = std::size_t{8} * 1024 * 1024;
constexpr std::size_t kHolderAllocationBytes = std::size_t{6} * 1024 * 1024;
constexpr std::size_t kProbeAllocationBytes = std::size_t{4} * 1024 * 1024;

bool check(cudaError_t result, std::string_view operation) {
    if (result == cudaSuccess) {
        return true;
    }

    std::cerr << operation << " failed with cudaError " << static_cast<int>(result) << ": "
              << cudaGetErrorString(result) << '\n';
    return false;
}

bool write_byte(int descriptor) {
    const char value = '1';
    while (true) {
        const ssize_t result = ::write(descriptor, &value, sizeof(value));
        if (result == static_cast<ssize_t>(sizeof(value))) {
            return true;
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
}

bool read_byte(int descriptor) {
    char value = 0;
    while (true) {
        const ssize_t result = ::read(descriptor, &value, sizeof(value));
        if (result == static_cast<ssize_t>(sizeof(value))) {
            return value == '1';
        }
        if (result < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
}

bool wait_for_child(pid_t child, std::string_view name) {
    int status = 0;
    while (::waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        std::cerr << "waitpid failed for " << name << "\n";
        return false;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
        std::cerr << name << " exited unsuccessfully\n";
        return false;
    }
    return true;
}

int run_holder(int ready_descriptor, int release_descriptor) {
    void* device_pointer = nullptr;
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    bool succeeded =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "holder cudaMemGetInfo initial") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes;

    succeeded = succeeded &&
                check(cudaMalloc(&device_pointer, kHolderAllocationBytes), "holder cudaMalloc");
    succeeded =
        succeeded &&
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "holder cudaMemGetInfo allocated") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes - kHolderAllocationBytes;
    if (!succeeded || !write_byte(ready_descriptor)) {
        if (device_pointer != nullptr) {
            static_cast<void>(cudaFree(device_pointer));
        }
        (void)::close(ready_descriptor);
        (void)::close(release_descriptor);
        return EXIT_FAILURE;
    }

    succeeded = read_byte(release_descriptor);
    succeeded = check(cudaFree(device_pointer), "holder cudaFree") && succeeded;
    succeeded =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "holder cudaMemGetInfo restored") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes && succeeded;

    (void)::close(ready_descriptor);
    (void)::close(release_descriptor);
    return succeeded ? EXIT_SUCCESS : EXIT_FAILURE;
}

int run_probe() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    bool succeeded =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "probe cudaMemGetInfo initial") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes - kHolderAllocationBytes;
    if (!succeeded) {
        return EXIT_FAILURE;
    }

    void* device_pointer = nullptr;
    const cudaError_t allocation_result = cudaMalloc(&device_pointer, kProbeAllocationBytes);
    const bool rejected =
        allocation_result == cudaErrorMemoryAllocation && device_pointer == nullptr;
    if (!rejected) {
        std::cerr << "probe allocation was not rejected; result="
                  << static_cast<int>(allocation_result) << "\n";
        if (device_pointer != nullptr) {
            static_cast<void>(cudaFree(device_pointer));
        }
    }
    succeeded = succeeded && rejected;
    succeeded =
        check(cudaMemGetInfo(&free_bytes, &total_bytes), "probe cudaMemGetInfo unchanged") &&
        total_bytes == kQuotaBytes && free_bytes == kQuotaBytes - kHolderAllocationBytes &&
        succeeded;
    return succeeded ? EXIT_SUCCESS : EXIT_FAILURE;
}

bool set_unique_tenant(std::string* tenant_id) {
    if (tenant_id == nullptr) {
        return false;
    }
    *tenant_id = "glimmer-real-gpu-shared-" + std::to_string(static_cast<long long>(::getpid()));
    return ::setenv("GLIMMER_QUOTA_TENANT_ID", tenant_id->c_str(), 1) == 0;
}

bool parse_descriptor(const char* value, int* descriptor) noexcept {
    if (value == nullptr || descriptor == nullptr) {
        return false;
    }
    const char* end = value + std::strlen(value);
    int parsed_descriptor = -1;
    const auto [parsed_end, error] = std::from_chars(value, end, parsed_descriptor);
    if (error != std::errc{} || parsed_end != end || parsed_descriptor < 0) {
        return false;
    }
    *descriptor = parsed_descriptor;
    return true;
}

[[noreturn]] void exec_holder(const char* executable, int ready_descriptor,
                              int release_descriptor) {
    const std::string ready_argument = std::to_string(ready_descriptor);
    const std::string release_argument = std::to_string(release_descriptor);
    char* const arguments[] = {const_cast<char*>(executable), const_cast<char*>("--holder"),
                               const_cast<char*>(ready_argument.c_str()),
                               const_cast<char*>(release_argument.c_str()), nullptr};
    ::execv(executable, arguments);
    std::_Exit(127);
}

[[noreturn]] void exec_probe(const char* executable) {
    char* const arguments[] = {const_cast<char*>(executable), const_cast<char*>("--probe"),
                               nullptr};
    ::execv(executable, arguments);
    std::_Exit(127);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc > 1) {
        if (argc == 4 && std::string_view(argv[1]) == "--holder") {
            int ready_descriptor = -1;
            int release_descriptor = -1;
            if (!parse_descriptor(argv[2], &ready_descriptor) ||
                !parse_descriptor(argv[3], &release_descriptor)) {
                return EXIT_FAILURE;
            }
            return run_holder(ready_descriptor, release_descriptor);
        }
        if (argc == 2 && std::string_view(argv[1]) == "--probe") {
            return run_probe();
        }
        std::cerr << "unknown test role\n";
        return EXIT_FAILURE;
    }

    std::string tenant_id;
    if (!set_unique_tenant(&tenant_id)) {
        std::cerr << "failed to configure a unique shared quota tenant\n";
        return EXIT_FAILURE;
    }

    int ready_pipe[2] = {-1, -1};
    int release_pipe[2] = {-1, -1};
    if (::pipe(ready_pipe) != 0 || ::pipe(release_pipe) != 0) {
        std::cerr << "pipe creation failed\n";
        if (ready_pipe[0] >= 0) {
            (void)::close(ready_pipe[0]);
            (void)::close(ready_pipe[1]);
        }
        return EXIT_FAILURE;
    }

    // CUDA Runtime state is not generally fork-safe. Fork only transfers the
    // synchronization descriptors; each child immediately execs a fresh image.
    const pid_t holder = ::fork();
    if (holder < 0) {
        std::cerr << "failed to fork holder process\n";
        (void)::close(ready_pipe[0]);
        (void)::close(ready_pipe[1]);
        (void)::close(release_pipe[0]);
        (void)::close(release_pipe[1]);
        return EXIT_FAILURE;
    }
    if (holder == 0) {
        (void)::close(ready_pipe[0]);
        (void)::close(release_pipe[1]);
        exec_holder(argv[0], ready_pipe[1], release_pipe[0]);
    }

    (void)::close(ready_pipe[1]);
    (void)::close(release_pipe[0]);
    const bool holder_ready = read_byte(ready_pipe[0]);
    (void)::close(ready_pipe[0]);
    if (!holder_ready) {
        (void)::close(release_pipe[1]);
        static_cast<void>(wait_for_child(holder, "holder"));
        static_cast<void>(glimmer::control::SharedMemoryQuota::remove_region(tenant_id));
        return EXIT_FAILURE;
    }

    const pid_t probe = ::fork();
    if (probe < 0) {
        std::cerr << "failed to fork probe process\n";
        static_cast<void>(write_byte(release_pipe[1]));
        (void)::close(release_pipe[1]);
        static_cast<void>(wait_for_child(holder, "holder"));
        static_cast<void>(glimmer::control::SharedMemoryQuota::remove_region(tenant_id));
        return EXIT_FAILURE;
    }
    if (probe == 0) {
        (void)::close(release_pipe[1]);
        exec_probe(argv[0]);
    }

    const bool probe_finished = wait_for_child(probe, "probe");
    const bool release_sent = write_byte(release_pipe[1]);
    (void)::close(release_pipe[1]);
    const bool holder_finished = wait_for_child(holder, "holder");
    const bool region_removed = glimmer::control::SharedMemoryQuota::remove_region(tenant_id);
    if (!region_removed) {
        std::cerr << "failed to remove the temporary shared quota region\n";
    }
    return probe_finished && release_sent && holder_finished && region_removed ? EXIT_SUCCESS
                                                                               : EXIT_FAILURE;
}
