#include <cuda.h>
#include "execution_cached_driver_fixture.h"

#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <poll.h>
#include <unistd.h>

namespace {
ExecutionCachedDriver driver;
constexpr unsigned int kWorkers = 320, kThreads = 32, kLogicalBlocks = 65536;
constexpr const char* kSource = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry low_add(.param .u64 output, .param .u32 iterations) {
 .reg .pred %p<3>;
 .reg .b32 %r<8>;
 .reg .b64 %rd<4>;
 ld.param.u64 %rd1, [output]; ld.param.u32 %r1, [iterations];
 mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x;
 mad.lo.u32 %r5, %r2, %r3, %r4;
 mul.wide.u32 %rd2, %r5, 4; add.u64 %rd3, %rd1, %rd2;
 ld.global.u32 %r6, [%rd3]; add.u32 %r6, %r6, 1; st.global.u32 [%rd3], %r6;
 setp.ne.u32 %p1, %r4, 0; @%p1 bra EXIT;
WAIT:
 ld.volatile.global.u32 %r7, [%rd3]; sub.u32 %r1, %r1, 1;
 setp.ne.u32 %p2, %r1, 0; @%p2 bra WAIT;
EXIT: ret;
}
.visible .entry high_add(.param .u64 output, .param .u64 duration) {
 .reg .pred %p;
 .reg .b32 %r<6>;
 .reg .b64 %rd<8>;
 ld.param.u64 %rd1, [output]; ld.param.u64 %rd2, [duration];
 mov.u64 %rd3, %globaltimer;
WAIT:
 mov.u64 %rd4, %globaltimer; sub.u64 %rd5, %rd4, %rd3;
 setp.lt.u64 %p, %rd5, %rd2; @%p bra WAIT;
 mov.u32 %r1, %ctaid.x; mov.u32 %r2, %ntid.x; mov.u32 %r3, %tid.x;
 mad.lo.u32 %r4, %r1, %r2, %r3;
 mul.wide.u32 %rd6, %r4, 4; add.u64 %rd7, %rd1, %rd6;
 ld.global.u32 %r5, [%rd7]; add.u32 %r5, %r5, 1; st.global.u32 [%rd7], %r5;
 ret;
}
)ptx";
void check(CUresult status, const char* operation) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": CUDA " + std::to_string(status));
}
struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUdeviceptr output{};
    CUstream stream{};
    CUstream independent_stream{};
    unsigned int primary_references{};
    bool is_primary{};
    bool is_cleanup_ok{true};
    void cleanup() noexcept {
        const auto checked = [&](CUresult status) { is_cleanup_ok &= status == CUDA_SUCCESS; };
        if (context) checked(driver.synchronize_context());
        if (output) {
            checked(driver.free_memory(output));
            output = 0;
        }
        if (stream) {
            checked(driver.destroy_stream(stream));
            stream = nullptr;
        }
        if (independent_stream) {
            checked(driver.destroy_stream(independent_stream));
            independent_stream = nullptr;
        }
        if (module) {
            checked(driver.unload_module(module));
            module = nullptr;
        }
        if (context) {
            if (is_primary) {
                while (primary_references) {
                    const auto status = driver.release_primary(0);
                    checked(status);
                    --primary_references;
                    if (status != CUDA_SUCCESS) break;
                }
                unsigned int flags{};
                int active{};
                CUcontext current{};
                checked(driver.primary_state(0, &flags, &active));
                checked(driver.get_current(&current));
                is_cleanup_ok &= flags == 0 && active == 0 && current == context;
                checked(driver.set_current(nullptr));
            } else
                checked(driver.destroy_context(context));
            context = nullptr;
        }
    }
    ~Resources() {
        cleanup();
    }
};
void run(bool is_low, std::string_view binary_path, bool primary, bool independent_low) {
    std::vector<std::byte> image;
    if (!binary_path.empty()) {
        std::ifstream stream(std::string(binary_path), std::ios::binary | std::ios::ate);
        if (!stream) throw std::runtime_error("open ordinary application module image");
        const auto size = stream.tellg();
        if (size <= 0 || size > std::streamoff{16} * 1024 * 1024)
            throw std::runtime_error("application module image exceeds its input bound");
        image.resize(static_cast<std::size_t>(size));
        stream.seekg(0);
        if (!stream.read(reinterpret_cast<char*>(image.data()), size))
            throw std::runtime_error("read complete ordinary application module image");
    }
    const bool is_binary = !image.empty();
    check(driver.initialize(0), "initialize");
    CUdevice device{};
    check(driver.get_device(&device, 0), "device");
    int sms{};
    check(driver.device_attribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
          "SM count");
    if (sms != 80) throw std::runtime_error("explicit V100 geometry requires 80 available SMs");
    Resources resources;
    resources.is_primary = primary;
    if (primary) {
        unsigned int flags{};
        int active{};
        check(driver.primary_state(device, &flags, &active), "initial primary state");
        if (flags || active) throw std::runtime_error("primary must start inactive");
        check(driver.primary_flags(device, CU_CTX_SCHED_YIELD), "inactive primary flags");
        check(driver.retain_primary(&resources.context, device), "primary retain");
        ++resources.primary_references;
        CUcontext again{}, current{};
        check(driver.retain_primary(&again, device), "shared primary retain");
        ++resources.primary_references;
        check(driver.get_current(&current), "retain leaves current unchanged");
        if (again != resources.context || current)
            throw std::runtime_error("primary identity or current stack mismatch");
        check(driver.set_current(resources.context), "select primary context");
        CUdevice selected{-1};
        check(driver.context_device(&selected), "primary context device");
        check(driver.get_context_flags(&flags), "primary context flags");
        if (selected != device || flags != (CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST))
            throw std::runtime_error("primary native flag/device mismatch");
        check(driver.primary_flags(device, CU_CTX_SCHED_SPIN), "active primary flags");
        check(driver.reset_primary(device), "reset owned primary generation");
        check(driver.primary_state(device, &flags, &active), "reset primary state");
        check(driver.get_current(&current), "reset retains current handle");
        if (flags || active || current != resources.context)
            throw std::runtime_error("reset changed primary references or current identity");
        check(driver.retain_primary(&again, device), "reactivate primary generation");
        ++resources.primary_references;
        if (again != resources.context)
            throw std::runtime_error("primary handle recycled on reset");
        check(driver.release_primary(device), "release third primary reference");
        --resources.primary_references;
        check(driver.release_primary(device), "release second primary reference");
        --resources.primary_references;
        check(driver.primary_state(device, &flags, &active), "retained primary remains active");
        if (flags || active != 1) throw std::runtime_error("reset discarded retained references");
        check(driver.set_context_flags(0), "set native context wait flags");
    } else
        check(driver.create_context(&resources.context, 0, device), "context");
    CUdeviceptr scope_check{};
    check(driver.explicit_allocate(&scope_check, 4), "explicit Driver scope allocation");
    check(driver.free_memory(scope_check), "cached free owns explicit-scope allocation");
    check(driver.load_module(&resources.module,
                             is_binary ? static_cast<const void*>(image.data()) : kSource),
          "original application module");
    std::fill(image.begin(), image.end(), std::byte{});
    CUfunction function{};
    const auto* name =
        is_binary ? (is_low ? "binary_add" : "binary_high") : (is_low ? "low_add" : "high_add");
    check(driver.get_function(&function, resources.module, name), "original function");
    const unsigned int blocks = is_low ? kLogicalBlocks : kWorkers;
    const auto command_elements = std::size_t{blocks} * kThreads;
    const auto low_commands = is_low && independent_low ? 2U : 1U;
    std::vector<std::uint32_t> output(command_elements * low_commands, 0);
    const auto bytes = output.size() * sizeof(output[0]);
    const auto command_bytes = command_elements * sizeof(output[0]);
    check(driver.allocate(&resources.output, bytes), "allocate original output");
    for (unsigned int index = 0; index < low_commands; ++index)
        check(driver.upload(resources.output + index * command_bytes,
                            output.data() + index * command_elements, command_bytes),
              "initialize all outputs with bounded ordinary copies");
    check(driver.synchronize_context(), "complete uploads before nonblocking stream work");
    check(driver.create_stream(&resources.stream, CU_STREAM_NON_BLOCKING), "original stream");
    if (low_commands == 2)
        check(driver.create_stream(&resources.independent_stream, CU_STREAM_NON_BLOCKING),
              "independent original stream");
    std::cout << "Ordinary priority application ready: role=" << (is_low ? "low" : "high")
              << "; pid=" << ::getpid() << '\n'
              << std::flush;
    pollfd input{STDIN_FILENO, POLLIN, 0};
    if (::poll(&input, 1, 30000) != 1) throw std::runtime_error("bounded workload start barrier");
    std::string command;
    if (!std::getline(std::cin, command) || command != "run")
        throw std::runtime_error("invalid workload start barrier");
    std::uint32_t iterations = 8192;
    std::uint64_t duration = 100000000;
    void* arguments[]{&resources.output,
                      is_low ? static_cast<void*>(&iterations) : static_cast<void*>(&duration)};
    struct alignas(16) BinaryValues {
        std::uint32_t add{1}, iterations{}, unused[2]{};
    } values{1, iterations, {}};
    auto count = static_cast<std::uint32_t>(output.size());
    void* binary_arguments[]{&resources.output, &count, &values};
    for (unsigned int index = 0; index < (is_low ? low_commands : 2U); ++index) {
        auto destination = resources.output + (is_low ? index * command_bytes : 0);
        arguments[0] = &destination;
        binary_arguments[0] = &destination;
        check(driver.launch(function, blocks, 1, 1, kThreads, 1, 1, 0,
                            is_low && index == 1 ? resources.independent_stream : resources.stream,
                            is_binary && is_low ? binary_arguments : arguments, nullptr),
              "original pointer-array launch");
    }
    check(driver.synchronize_stream(resources.stream), "logical original stream completion");
    if (low_commands == 2)
        check(driver.synchronize_stream(resources.independent_stream),
              "logical independent stream completion");
    for (unsigned int index = 0; index < low_commands; ++index)
        check(driver.download(output.data() + index * command_elements,
                              resources.output + index * command_bytes, command_bytes),
              "download every original output with bounded ordinary copies");
    for (const auto value : output)
        if (value != (is_low ? 1U : 2U)) throw std::runtime_error("once-only output mismatch");
    resources.cleanup();
    if (!resources.is_cleanup_ok) throw std::runtime_error("incomplete original resource teardown");
    std::cout << "Ordinary priority application verified: role=" << (is_low ? "low" : "high")
              << "; elements=" << output.size()
              << "; cached_lookup=" << (driver.is_cached ? "true" : "false")
              << "; binary_image=" << (is_binary ? "true" : "false")
              << "; primary_context=" << (primary ? "true" : "false")
              << "; low_commands=" << low_commands
              << "; framework_api=false; full_device_qualified=false; timing_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc < 2 ||
            (std::string_view(argv[1]) != "--low" && std::string_view(argv[1]) != "--high"))
            throw std::runtime_error("expected --low or --high");
        bool cached{}, primary{}, independent_low{};
        std::string_view binary_path;
        for (int index = 2; index < argc; ++index) {
            const std::string_view option{argv[index]};
            if (option == "--cached-lookups" && !cached)
                cached = true;
            else if (option == "--primary-context" && !primary)
                primary = true;
            else if (option == "--independent-low" && !independent_low)
                independent_low = true;
            else if (option == "--binary-image" && binary_path.empty() && index + 1 < argc)
                binary_path = argv[++index];
            else
                throw std::runtime_error("invalid ordinary application option");
        }
        driver.bind(cached, primary);
        run(std::string_view(argv[1]) == "--low", binary_path, primary, independent_low);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
