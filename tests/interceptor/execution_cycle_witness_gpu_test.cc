#include "execution_cycle_witness_fixture.h"
#include "execution_cached_driver_fixture.h"
#include "execution_cached_runtime_fixture.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

namespace {
using namespace cycle_witness;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value, const char* operation) {
    if (value != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": Driver " + std::to_string(value));
}
void check(cudaError_t value, const char* operation) {
    if (value != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": Runtime " + std::to_string(value));
}
struct Options {
    std::string mode, image, directory;
    bool cached{};
};
Options options(int argc, char** argv) {
    expect(argc >= 6, "mode, --image and --output-dir required");
    Options result;
    result.mode = argv[1];
    expect(result.mode == "--low" || result.mode == "--high" || result.mode == "--calibrate" ||
               result.mode == "--serialized" || result.mode == "--concurrent",
           "one fixed explicit witness mode required");
    for (int index = 2; index < argc; ++index) {
        const std::string_view key = argv[index];
        if (key == "--cached-lookups") {
            expect(!result.cached, "duplicate cached option");
            result.cached = true;
        } else {
            expect(index + 1 < argc, "missing option value");
            auto* field = key == "--image"        ? &result.image
                          : key == "--output-dir" ? &result.directory
                                                  : nullptr;
            expect(field && field->empty(), "unknown or duplicate path option");
            *field = argv[++index];
            expect(
                !field->empty() && field->front() == '/' && field->find('\0') == std::string::npos,
                "absolute original image/output path required");
        }
    }
    expect(!result.image.empty() && !result.directory.empty(), "both original paths required");
    return result;
}
std::vector<char> image(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    expect(stream.good(), "open fixed compiler image");
    const auto count = stream.tellg();
    expect(count > 0 && count < std::streamoff{1024} * 1024, "bounded original sm70 image");
    std::vector<char> result(static_cast<std::size_t>(count) + 1);
    stream.seekg(0);
    stream.read(result.data(), static_cast<std::streamsize>(count));
    expect(stream.good(), "read complete original image");
    return result;
}
struct Buffer {
    std::uint32_t kind{};
    CUdeviceptr trace{}, values{};
    CUstream stream{};
    CUfunction function{};
    std::vector<Sample> samples{std::size_t{kRecords} + std::size_t{2} * kGuardRecords};
    std::vector<std::uint32_t> result =
        std::vector<std::uint32_t>(kValues + 2 * kGuardValues, kGuard);
    explicit Buffer(std::uint32_t group) : kind(group) {
        std::memset(samples.data(), 0xA5, kTraceBytes);
    }
};
struct Resources {
    ExecutionCachedDriver& driver;
    CUmodule module{};
    std::vector<std::unique_ptr<Buffer>> buffers;
    bool runtime_ready{}, attempted{}, healthy{true};
    explicit Resources(ExecutionCachedDriver& functions) : driver(functions) {}
    bool close() noexcept {
        if (attempted) return healthy;
        attempted = true;
        const auto observed = [&](CUresult value) {
            if (value != CUDA_SUCCESS) {
                healthy = false;
                std::cerr << "cycle witness cleanup: Driver " << value << '\n';
            }
            return healthy;
        };
        if (runtime_ready && !observed(driver.synchronize_context())) return false;
        for (auto& buffer : buffers) {
            if (buffer->trace && !observed(driver.free_memory(buffer->trace))) return false;
            buffer->trace = 0;
            if (buffer->values && !observed(driver.free_memory(buffer->values))) return false;
            buffer->values = 0;
            if (buffer->stream && !observed(driver.destroy_stream(buffer->stream))) return false;
            buffer->stream = nullptr;
        }
        if (module && !observed(driver.unload_module(module))) return false;
        module = nullptr;
        return healthy;
    }
    ~Resources() {
        if (!close()) {
            // Unknown native DMA/physical state cannot free its host data.
            for (auto& buffer : buffers) (void)buffer.release();
        }
    }
};
void launch(ExecutionCachedDriver& driver, Buffer& buffer, unsigned int active) {
    auto trace = buffer.trace + kGuardBytes;
    auto values = buffer.values + kGuardBytes;
    void* parameters[]{&trace, &values, &active};
    check(driver.launch(buffer.function, active ? kBlocks : 1, 1, 1, kThreads, 1, 1, 0,
                        buffer.stream, parameters, nullptr),
          "submit original kernel");
    // Make delayed use of the caller's argument storage observable even in an
    // optimized build: zero would skip every record/value store in the kernel.
    volatile unsigned int* snapshot_boundary = &active;
    *snapshot_boundary = 0;
}
struct Affine {
    std::uint32_t multiplier{1}, increment{};
    void advance(std::uint32_t a, std::uint32_t b) {
        multiplier = a * multiplier;
        increment = a * increment + b;
    }
    std::uint32_t apply(std::uint32_t value) const {
        return multiplier * value + increment;
    }
};
std::uint32_t rounds(std::uint32_t kind) {
    return kind == 2 ? 64U : kind == 0 ? 4096U : 2048U;
}
std::uint32_t group(std::uint32_t kind, std::uint32_t identity) {
    return kind == 2 ? identity % 2 : kind;
}
bool verify(const Buffer& buffer) {
    const auto guard = [](std::span<const std::byte> bytes) {
        return std::ranges::all_of(bytes, [](auto value) { return value == std::byte{0xA5}; });
    };
    bool valid = guard(std::as_bytes(std::span(buffer.samples)).first(kGuardBytes)) &&
                 guard(std::as_bytes(std::span(buffer.samples)).last(kGuardBytes)) &&
                 guard(std::as_bytes(std::span(buffer.result)).first(kGuardBytes)) &&
                 guard(std::as_bytes(std::span(buffer.result)).last(kGuardBytes));
    Affine step;
    for (std::uint32_t index = 0; index < rounds(buffer.kind); ++index)
        step.advance(kMultiplier, kIncrement);
    for (std::uint32_t identity = 0; identity < kWarps; ++identity) {
        const auto expected_group = group(buffer.kind, identity);
        auto state = kSeed + expected_group * kGroupSeed + (identity * kWarp + 1) * 17U;
        std::uint64_t previous_cycle{}, previous_time{};
        const auto first = std::size_t{kGuardRecords} + std::size_t{identity} * kSamples;
        const auto sm = buffer.samples[first].sm_before;
        for (std::uint32_t index = 0; index < kSamples; ++index) {
            const auto& sample = buffer.samples[first + index];
            state = step.apply(state);
            valid = valid && sample.identity == identity && sample.group == expected_group &&
                    sample.value == state && !sample.reserved && sample.sm_before == sm &&
                    sample.sm_after == sm && sample.before_ns && sample.cycle > previous_cycle &&
                    sample.before_ns >= previous_time && sample.after_ns >= sample.before_ns;
            previous_cycle = sample.cycle;
            previous_time = sample.after_ns;
        }
    }
    Affine complete;
    for (std::uint32_t index = 0; index < kSamples; ++index)
        complete.advance(step.multiplier, step.increment);
    for (std::uint32_t index = 0; index < kValues; ++index) {
        const auto seed =
            kSeed + group(buffer.kind, index / kWarp) * kGroupSeed + (index + 1) * 17U;
        valid = valid && buffer.result[kGuardValues + index] == complete.apply(seed);
    }
    return valid;
}
class Output {
   public:
    explicit Output(const std::string& path)
        : descriptor_(
              ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600)) {
        expect(descriptor_ >= 0,
               "create one new diagnostic image without replacing existing files");
    }
    Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;
    ~Output() {
        if (descriptor_ >= 0) (void)::close(descriptor_);
    }
    void write(std::span<const std::byte> bytes) {
        unsigned int interrupted{};
        while (!bytes.empty()) {
            const auto count = ::write(descriptor_, bytes.data(), bytes.size());
            if (count < 0 && errno == EINTR && ++interrupted < 8) continue;
            expect(count > 0 && static_cast<std::size_t>(count) <= bytes.size(),
                   "complete bounded diagnostic image write");
            bytes = bytes.subspan(static_cast<std::size_t>(count));
        }
    }
    void close() {
        const auto owned = std::exchange(descriptor_, -1);
        expect(::close(owned) == 0, "close complete diagnostic image");
    }

   private:
    int descriptor_;
};
void save(const Buffer& buffer, const std::string& directory) {
    const std::array<std::uint32_t, 16> header{0x31574347U,
                                               1,
                                               64,
                                               sizeof(Sample),
                                               kBlocks,
                                               kThreads,
                                               kSamples,
                                               rounds(buffer.kind),
                                               buffer.kind,
                                               kGuardBytes,
                                               kRecords,
                                               kValues,
                                               kRecords * sizeof(Sample),
                                               kValues * 4,
                                               kGuard,
                                               0};
    const auto* name = buffer.kind == 2 ? "calibration" : buffer.kind == 0 ? "low" : "high";
    Output file(directory + "/" + name + ".gcw");
    file.write(std::as_bytes(std::span(header)));
    file.write(std::as_bytes(std::span(buffer.samples)));
    file.write(std::as_bytes(std::span(buffer.result)));
    file.close();
}
bool run(const Options& input) {
    ExecutionCachedRuntime runtime;
    if (input.cached) runtime.enable();
    ExecutionCachedDriver driver;
    driver.bind(input.cached, true);
    Resources owned(driver);
    check(runtime.select(0), "select original device");
    check(runtime.release(nullptr), "ordinary lazy Runtime primary initialization");
    owned.runtime_ready = true;
    CUcontext context{};
    check(driver.get_current(&context), "observe original current primary");
    expect(context, "Runtime supplied the ordinary primary context");
    CUdevice device{};
    check(driver.get_device(&device, 0), "get original device");
    int sms{}, threads{};
    check(driver.device_attribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
          "SM metadata");
    check(driver.device_attribute(&threads, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                                  device),
          "warp metadata");
    expect(sms == 80 && threads == 2048, "fixed V100 diagnostic platform contract");
    const auto creation = [&](CUresult status, bool has_output, const char* phase) {
        if (status == CUDA_SUCCESS && has_output) return;
        // Do not explicitly retire an unproven partial factory handle.
        owned.attempted = true;
        owned.healthy = false;
        check(status, phase);
        throw std::runtime_error(std::string(phase) + ": successful NULL output");
    };
    const auto bytes = image(input.image);
    auto status = driver.load_module(&owned.module, bytes.data());
    creation(status, owned.module != nullptr, "load unchanged original sm70 image");
    const bool pair = input.mode == "--serialized" || input.mode == "--concurrent";
    const auto first = input.mode == "--calibrate" ? 2U : input.mode == "--high" ? 1U : 0U;
    for (unsigned int leg = 0; leg < (pair ? 2U : 1U); ++leg) {
        auto buffer = std::make_unique<Buffer>(pair ? leg : first);
        const auto* entry = buffer->kind == 2   ? "cycle_calibration"
                            : buffer->kind == 0 ? "cycle_low"
                                                : "cycle_high";
        owned.buffers.push_back(std::move(buffer));
        auto& current = *owned.buffers.back();
        check(driver.get_function(&current.function, owned.module, entry),
              "bind original function");
        expect(current.function, "original lookup returned a non-NULL function");
        status = driver.allocate(&current.trace, kTraceBytes);
        creation(status, current.trace != 0, "allocate complete original sample extent");
        status = driver.allocate(&current.values, kValueBytes);
        creation(status, current.values != 0, "allocate complete original value extent");
        status = driver.create_stream(&current.stream, CU_STREAM_NON_BLOCKING);
        creation(status, current.stream != nullptr, "create original nonblocking stream");
        check(driver.upload(current.trace, current.samples.data(), kTraceBytes),
              "upload all sample guards");
        check(driver.upload(current.values, current.result.data(), kValueBytes),
              "upload all value guards");
        launch(driver, current, 0);
        check(driver.synchronize_stream(current.stream),
              "materialize original binding before readiness");
    }
    const auto* role = input.mode == "--low" ? "low" : input.mode == "--high" ? "high" : "control";
    std::cout << "Ordinary Runtime application ready: role=" << role << '\n' << std::flush;
    std::string signal;
    expect(static_cast<bool>(std::getline(std::cin, signal)) && signal == "run",
           "ordinary run input required");
    for (auto& buffer : owned.buffers) {
        launch(driver, *buffer, 1);
        if (input.mode == "--serialized")
            check(driver.synchronize_stream(buffer->stream),
                  "explicit serialized negative-control boundary");
    }
    check(driver.synchronize_context(), "actual complete logical physical join");
    bool valid = true;
    for (auto& buffer : owned.buffers) {
        check(driver.download(buffer->samples.data(), buffer->trace, kTraceBytes),
              "read every sample and guard");
        check(driver.download(buffer->result.data(), buffer->values, kValueBytes),
              "read every value and guard");
        valid = verify(*buffer) && valid;
    }
    expect(owned.close(), "all original modules/allocations/streams retire normally");
    check(runtime.reset(), "reset this completed Runtime primary generation");
    expect(runtime.close(), "close original public Runtime cache");
    for (const auto& buffer : owned.buffers) save(*buffer, input.directory);
    std::cout << "{\"status\":\"" << (valid ? "passed" : "failed") << "\",\"mode\":\""
              << input.mode.substr(2) << "\",\"cached\":" << (input.cached ? "true" : "false")
              << ",\"datasets\":" << owned.buffers.size() << ",\"records_per_dataset\":" << kRecords
              << ",\"values_per_dataset\":" << kValues
              << ",\"cleanup\":true,\"full_device_qualified\":false,"
                 "\"co_residency_qualified\":false,\"performance_qualified\":false}\n";
    return valid;
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout
                << "Cycle witness SDK input: --low|--high|--calibrate|--serialized|--concurrent "
                   "--image PATH --output-dir DIR [--cached-lookups]\n";
            return EXIT_SUCCESS;
        }
        return run(options(argc, argv)) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Cycle witness input: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
