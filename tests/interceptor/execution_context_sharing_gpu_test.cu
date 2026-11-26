#include <cuda_runtime.h>

#include "execution_binary_kernel_fixture.cu"
#include "execution_cached_driver_fixture.h"
#include "execution_cached_runtime_fixture.h"

#include <array>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
constexpr unsigned int kWorkers = 320, kThreads = 32, kLogicalBlocks = 65536, kGuardCount = 64;
constexpr std::uint32_t kGuard = 0x194c7b35;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudaError_t status, const char* phase) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(phase) + ": Runtime " + std::to_string(status));
}
void check(CUresult status, const char* phase) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(phase) + ": Driver " + std::to_string(status));
}
struct Options {
    std::string directory;
    bool is_low{}, is_cached{}, reverses_contexts{}, uses_equal_flags{};
};
Options options(int argc, char** argv) {
    expect(argc >= 4, "role and absolute --output-dir required");
    Options result;
    const std::string_view role = argv[1];
    expect(role == "--low" || role == "--high", "one low/high role required");
    result.is_low = role == "--low";
    for (int index = 2; index < argc; ++index) {
        const std::string_view key = argv[index];
        if (key == "--cached-lookups") {
            expect(!result.is_cached, "duplicate cache option");
            result.is_cached = true;
        } else if (key == "--reverse-contexts") {
            expect(!result.reverses_contexts, "duplicate context option");
            result.reverses_contexts = true;
        } else if (key == "--equal-flags") {
            expect(!result.uses_equal_flags, "duplicate flags option");
            result.uses_equal_flags = true;
        } else {
            expect(key == "--output-dir" && result.directory.empty() && index + 1 < argc,
                   "unknown, duplicate or incomplete path option");
            result.directory = argv[++index];
            expect(!result.directory.empty() && result.directory.front() == '/',
                   "absolute output directory required");
        }
    }
    expect(!result.directory.empty(), "output directory required");
    return result;
}
struct Resources {
    ExecutionCachedDriver& driver;
    ExecutionCachedRuntime& runtime;
    CUcontext context{};
    CUdevice device{};
    std::uint32_t* allocation{};
    cudaStream_t stream{};
    bool is_primary{}, attempted{}, healthy{true};
    bool close() noexcept {
        if (attempted) return healthy;
        attempted = true;
        const auto observed = [&](auto status) {
            if (static_cast<int>(status) != 0) {
                healthy = false;
                std::cerr << "typed context cleanup: status=" << static_cast<int>(status) << '\n';
            }
        };
        if (stream) observed(runtime.synchronize_stream(stream));
        if (allocation) observed(runtime.release(allocation));
        if (stream) observed(runtime.destroy_stream(stream));
        if (context) {
            observed(driver.set_current(nullptr));
            observed(is_primary ? driver.release_primary(device) : driver.destroy_context(context));
        }
        return healthy;
    }
    ~Resources() {
        (void)close();
    }
};
void save(const std::string& path, const std::vector<std::uint32_t>& values) {
    const auto descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    expect(descriptor >= 0, "create exclusive full-array evidence");
    const auto* data = reinterpret_cast<const char*>(values.data());
    auto remaining = values.size() * sizeof(values[0]);
    bool healthy = true;
    while (remaining) {
        const auto count = ::write(descriptor, data, remaining);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            healthy = false;
            break;
        }
        remaining -= static_cast<std::size_t>(count);
        data += count;
    }
    const auto closed = ::close(descriptor);
    expect(healthy && closed == 0, "write and close every full-array byte");
}
void run(const Options& input) {
    ExecutionCachedDriver driver;
    driver.bind(input.is_cached, true);
    check(driver.initialize(0), "initialize Driver before Runtime");
    ExecutionCachedRuntime runtime;
    Resources owned{driver, runtime};
    owned.is_primary = input.is_low != input.reverses_contexts;
    const auto flags = static_cast<unsigned int>(
        input.is_low || input.uses_equal_flags ? CU_CTX_SCHED_YIELD : CU_CTX_SCHED_SPIN);
    check(driver.get_device(&owned.device, 0), "get ordinary Driver device");
    if (owned.is_primary) {
        check(driver.primary_flags(owned.device, flags), "set inactive primary flags");
        check(driver.retain_primary(&owned.context, owned.device), "retain process primary");
        check(driver.set_current(owned.context), "bind process primary");
    } else
        check(driver.create_context(&owned.context, flags, owned.device), "create user context");
    unsigned int observed{};
    check(driver.get_context_flags(&observed), "read original public context flags");
    const auto context_flags =
        flags | (owned.is_primary ? static_cast<unsigned int>(CU_CTX_MAP_HOST) : 0U);
    expect(observed == context_flags,
           "public context flags differ from original SDK creation flags");
    if (owned.is_primary) {
        int active{};
        check(driver.primary_state(owned.device, &observed, &active), "read active primary state");
        expect(observed == flags && active == 1, "primary/context flags disagree");
        expect(
            driver.primary_flags(owned.device, flags | CU_CTX_MAP_HOST) == CUDA_ERROR_INVALID_VALUE,
            "primary setter must reject its implicit mapped-host capability");
        check(driver.primary_state(owned.device, &observed, &active),
              "read primary after rejected flags");
        expect(observed == flags && active == 1, "rejected primary flags changed logical state");
    }
    if (input.is_cached) runtime.enable();
    check(runtime.release(nullptr), "Runtime uses the existing Driver context");
    CUcontext current{};
    check(driver.get_current(&current), "read current after Runtime initialization");
    expect(current == owned.context, "Runtime replaced the caller's explicit Driver context");
    const auto count = (input.is_low ? kLogicalBlocks : kWorkers) * kThreads;
    std::vector<std::uint32_t> values(count + 2 * kGuardCount, kGuard);
    for (unsigned int index = 0; index < count; ++index) values[kGuardCount + index] = 0;
    check(runtime.allocate(reinterpret_cast<void**>(&owned.allocation),
                           values.size() * sizeof(values[0])),
          "allocate guarded output");
    auto* output = owned.allocation + kGuardCount;
    check(runtime.create_stream(&owned.stream, cudaStreamNonBlocking), "create ordinary stream");
    check(runtime.copy(owned.allocation, values.data(), values.size() * sizeof(values[0]),
                       cudaMemcpyHostToDevice),
          "upload complete guarded input");
    const auto launch_low = [&](unsigned int elements, ExecutionBinaryPair parameters) {
        if (input.is_cached) {
            void* arguments[]{&output, &elements, &parameters};
            check(runtime.launch(reinterpret_cast<const void*>(&binary_add),
                                 dim3{elements ? kLogicalBlocks : 1U}, dim3{kThreads}, arguments, 0,
                                 owned.stream),
                  "cached low launch");
        } else
            binary_add<<<elements ? kLogicalBlocks : 1U, kThreads, 0, owned.stream>>>(
                output, elements, parameters);
        parameters.add = 99;  // Both native and owned launch already own the by-value bytes.
        check(runtime.last_error(), "low launch error");
    };
    const auto launch_high = [&](unsigned long long duration) {
        if (input.is_cached) {
            void* arguments[]{&output, &duration};
            check(runtime.launch(reinterpret_cast<const void*>(&binary_high), dim3{kWorkers},
                                 dim3{kThreads}, arguments, 0, owned.stream),
                  "cached high launch");
        } else
            binary_high<<<kWorkers, kThreads, 0, owned.stream>>>(output, duration);
        check(runtime.last_error(), "high launch error");
    };
    if (input.is_low)
        launch_low(0, {});
    else
        launch_high(0);
    check(runtime.synchronize_stream(owned.stream), "warmup completion");
    check(runtime.copy(owned.allocation, values.data(), values.size() * sizeof(values[0]),
                       cudaMemcpyHostToDevice),
          "restore complete guarded input");
    const auto* role = input.is_low ? "low" : "high";
    std::cout << "Ordinary Runtime application ready: role=" << role << '\n' << std::flush;
    std::string signal;
    expect(static_cast<bool>(std::getline(std::cin, signal)) && signal == "run",
           "ordinary run input");
    if (input.is_low)
        launch_low(count, {1, 8192, {0, 0}});
    else
        for (unsigned int index = 0; index < 2; ++index) launch_high(100000000);
    check(runtime.synchronize_stream(owned.stream), "complete original work");
    check(runtime.copy(values.data(), owned.allocation, values.size() * sizeof(values[0]),
                       cudaMemcpyDeviceToHost),
          "download full output and guards");
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto expected = index < kGuardCount || index >= count + kGuardCount ? kGuard
                              : input.is_low                                      ? 1U
                                                                                  : 2U;
        expect(values[index] == expected,
               "complete non-idempotent output or allocation guard mismatch");
    }
    check(driver.get_context_flags(&observed), "observe flags after shared execution");
    expect(observed == context_flags, "another member corrupted public logical flags");
    expect(owned.close(), "complete SDK resource/context retirement");
    expect(runtime.close(), "close exact Runtime cache reference");
    save(input.directory + "/" + role + ".bin", values);
    std::cout << "typed_context_receipt={\"status\":\"passed\",\"role\":\"" << role
              << "\",\"context\":\"" << (owned.is_primary ? "primary" : "user")
              << "\",\"flags\":" << flags << ",\"context_flags\":" << context_flags
              << ",\"cached\":" << (input.is_cached ? "true" : "false") << ",\"elements\":" << count
              << ",\"guards\":128,\"workers\":320,\"cleanup\":true}\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "--low|--high --output-dir ABS [--cached-lookups] [--reverse-contexts] "
                         "[--equal-flags]\n";
            return EXIT_SUCCESS;
        }
        run(options(argc, argv));
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
