#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

// Ordinary, unchanged SDK input for native and preload execution. This fixed
// device-containing matrix does not establish H2H, timing or SM coverage. The
// separate --per-thread mode selects only the legacy API's special handle.
namespace {
constexpr std::size_t kStorageBytes = 256, kGuardBytes = 16;
constexpr std::size_t kSourceOffset = 17, kDestinationOffset = 43;
constexpr std::size_t kImageCount = 6, kCaseCount = 40;
using Image = std::array<unsigned char, kStorageBytes>;
using Images = std::array<Image, kImageCount>;
constexpr std::array<const char*, kImageCount> kImageNames{
    "device_source",        "device_destination", "pageable_source",
    "pageable_destination", "pinned_source",      "pinned_destination"};
constexpr std::array<const char*, 5> kRoutes{"pageable_h2d", "pinned_h2d", "pageable_d2h",
                                             "pinned_d2h", "d2d"};
constexpr std::array<const char*, 4> kModes{"sync_legacy", "async_legacy", "async_blocking",
                                            "async_nonblocking"};
constexpr std::array<std::size_t, 5> kSources{2, 4, 0, 0, 0};
constexpr std::array<std::size_t, 5> kDestinations{1, 1, 3, 5, 1};
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::string quote(std::string_view input) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char value : input) {
        if (value == '"' || value == '\\') {
            result += '\\';
            result += static_cast<char>(value);
        } else if (value < 32) {
            result += "\\u00";
            result += digits[value >> 4];
            result += digits[value & 15];
        } else {
            result += static_cast<char>(value);
        }
    }
    result += '"';
    return result;
}
std::string hex(const Image& image) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(image.size() * 2);
    for (const auto value : image) {
        result += digits[value >> 4];
        result += digits[value & 15];
    }
    return result;
}
std::string uuid_text(const CUuuid& uuid) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "GPU-";
    for (std::size_t index = 0; index != sizeof(uuid.bytes); ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) result += '-';
        const auto value = static_cast<unsigned char>(uuid.bytes[index]);
        result += digits[value >> 4];
        result += digits[value & 15];
    }
    return result;
}
struct Sdk {
    decltype(&cuMemcpy) copy{&cuMemcpy};
    decltype(&cuMemcpyAsync) asynchronous{&cuMemcpyAsync};
    bool is_cached{};
    void bind(bool cached) {
        if (!cached) return;
        const auto resolve = [](const char* name) {
            void* pointer{};
            CUdriverProcAddressQueryResult query{};
            const auto status = cuGetProcAddress_v2(name, &pointer, 4000,
                                                    CU_GET_PROC_ADDRESS_LEGACY_STREAM, &query);
            expect(status == CUDA_SUCCESS && pointer && query == CU_GET_PROC_ADDRESS_SUCCESS,
                   "resolve the original CUDA 4.0 generic-copy ABI");
            return pointer;
        };
        copy = reinterpret_cast<decltype(copy)>(resolve("cuMemcpy"));
        asynchronous = reinterpret_cast<decltype(asynchronous)>(resolve("cuMemcpyAsync"));
        is_cached = true;
    }
};
struct Experiment {
    Sdk sdk;
    CUcontext context{};
    std::array<CUstream, 2> streams{};
    std::array<CUdeviceptr, 2> devices{};
    std::array<void*, 2> pinned{};
    Images before{}, expected{}, actual{};
    std::string gpu_uuid;
    int driver_version{};
    std::size_t sync_calls{}, async_calls{}, completed_cases{};
    bool is_uncertain{}, is_closed{}, uses_per_thread{};
    std::ostringstream cases;

    void check(CUresult status, const char* operation) {
        if (status != CUDA_SUCCESS) {
            is_uncertain = true;
            throw std::runtime_error(std::string(operation) + " returned " +
                                     std::to_string(status));
        }
    }
    void proof() {
        CUcontext current{};
        check(cuCtxGetCurrent(&current), "observe current context before completion");
        if (!context || current != context) {
            is_uncertain = true;
            throw std::runtime_error("current context ownership lost");
        }
        check(cuCtxSynchronize(), "complete owned context");
        check(cuCtxGetCurrent(&current), "observe current context after completion");
        if (current != context) {
            is_uncertain = true;
            throw std::runtime_error("current context ownership changed during completion");
        }
    }
    void start(bool cached) {
        check(cuInit(0), "initialize Driver");
        sdk.bind(cached);
        check(cuDriverGetVersion(&driver_version), "query Driver version");
        CUdevice device{};
        check(cuDeviceGet(&device, 0), "select device zero");
        check(cuCtxCreate(&context, 0, device), "create owned context");
        expect(context != nullptr, "successful context creation has an identity");
        CUuuid uuid{};
        CUdevice actual_device{};
        check(cuCtxGetDevice(&actual_device), "query actual context device");
        check(cuDeviceGetUuid(&uuid, actual_device), "query actual GPU UUID");
        gpu_uuid = uuid_text(uuid);
        check(cuStreamCreate(&streams[0], CU_STREAM_DEFAULT), "create blocking stream");
        check(cuStreamCreate(&streams[1], CU_STREAM_NON_BLOCKING), "create nonblocking stream");
        expect(streams[0] && streams[1], "successful streams have identities");
        for (auto& pointer : devices) {
            check(cuMemAlloc(&pointer, kStorageBytes), "allocate device image");
            expect(pointer != 0, "successful device allocation has an identity");
        }
        for (auto& pointer : pinned) {
            check(cuMemHostAlloc(&pointer, kStorageBytes, CU_MEMHOSTALLOC_PORTABLE),
                  "allocate public portable host image");
            expect(pointer != nullptr, "successful host allocation has an identity");
        }
    }
    static CUdeviceptr host_address(const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer);
    }
    CUdeviceptr endpoint(std::size_t image) const {
        if (image < 2) return devices[image];
        if (image < 4) return host_address(actual[image].data());
        return host_address(pinned[image - 4]);
    }
    void ordinary_copy(CUdeviceptr destination, CUdeviceptr source, std::size_t bytes) {
        ++sync_calls;
        check(sdk.copy(destination, source, bytes), "generic setup/readback cuMemcpy");
    }
    void initialize(std::size_t ordinal) {
        for (std::size_t image = 0; image != kImageCount; ++image)
            for (std::size_t index = 0; index != kStorageBytes; ++index)
                before[image][index] = static_cast<unsigned char>(
                    index < kGuardBytes || index >= kStorageBytes - kGuardBytes
                        ? 0x80 + image
                        : (index * 17 + image * 31 + ordinal * 7) & 255);
        expected = actual = before;
        for (std::size_t image = 0; image != pinned.size(); ++image)
            std::memcpy(pinned[image], before[image + 4].data(), kStorageBytes);
        for (std::size_t image = 0; image != devices.size(); ++image)
            ordinary_copy(devices[image], host_address(before[image].data()), kStorageBytes);
        // Setup is not the operation under test. Finish it before selecting any
        // blocking/nonblocking route so no incidental dependency is assumed.
        proof();
    }
    void run_case(std::size_t bytes, std::size_t mode, std::size_t route) {
        const auto ordinal = completed_cases;
        initialize(ordinal);
        const auto source = kSources[route];
        const auto destination = kDestinations[route];
        std::copy_n(before[source].begin() + kSourceOffset, bytes,
                    expected[destination].begin() + kDestinationOffset);
        const auto stream = uses_per_thread && mode == 3 ? CU_STREAM_PER_THREAD
                            : mode >= 2                  ? streams[mode - 2]
                                                         : nullptr;
        CUresult status{};
        if (mode) {
            ++async_calls;
            status = sdk.asynchronous(endpoint(destination) + kDestinationOffset,
                                      endpoint(source) + kSourceOffset, bytes, stream);
        } else {
            ++sync_calls;
            status = sdk.copy(endpoint(destination) + kDestinationOffset,
                              endpoint(source) + kSourceOffset, bytes);
        }
        check(status, "original generic copy case");
        check(cuStreamSynchronize(stream), "complete the selected copy stream");
        for (std::size_t image = 0; image != devices.size(); ++image)
            ordinary_copy(host_address(actual[image].data()), devices[image], kStorageBytes);
        proof();
        for (std::size_t image = 0; image != pinned.size(); ++image)
            std::memcpy(actual[image + 4].data(), pinned[image], kStorageBytes);
        expect(actual == expected, "complete source/destination and all guard byte oracle");

        if (ordinal) cases << ',';
        cases << "{\"ordinal\":" << ordinal << ",\"route\":" << quote(kRoutes[route])
              << ",\"mode\":"
              << quote(uses_per_thread && mode == 3 ? "async_per_thread" : kModes[mode])
              << ",\"api\":" << quote(mode ? "cuMemcpyAsync" : "cuMemcpy") << ",\"bytes\":" << bytes
              << ",\"source_image\":" << quote(kImageNames[source])
              << ",\"destination_image\":" << quote(kImageNames[destination])
              << ",\"source_offset\":" << kSourceOffset
              << ",\"destination_offset\":" << kDestinationOffset
              << ",\"status\":" << static_cast<int>(status)
              << ",\"stream_complete\":true,\"full_bytes_verified\":true,\"checked_bytes\":"
              << kImageCount * kStorageBytes << ",\"images\":{";
        for (std::size_t image = 0; image != kImageCount; ++image) {
            if (image) cases << ',';
            cases << quote(kImageNames[image]) << ':' << quote(hex(actual[image]));
        }
        cases << "}}";
        ++completed_cases;
    }
    void close() {
        if (is_closed) return;
        expect(!is_uncertain, "uncertain SDK ownership cannot manufacture normal cleanup");
        if (context) {
            proof();
            for (auto& pointer : pinned)
                if (pointer) {
                    check(cuMemFreeHost(pointer), "release public host image");
                    pointer = nullptr;
                }
            for (auto& pointer : devices)
                if (pointer) {
                    check(cuMemFree(pointer), "release device image");
                    pointer = 0;
                }
            for (auto& stream : streams)
                if (stream) {
                    check(cuStreamDestroy(stream), "release owned stream");
                    stream = nullptr;
                }
            check(cuCtxDestroy(context), "release owned context");
            context = nullptr;
        }
        is_closed = true;
    }
};
}  // namespace

int main(int argc, char** argv) {
    std::unique_ptr<Experiment> experiment;
    try {
        bool cached{}, per_thread{};
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument{argv[index]};
            if (argument == "--cached" && !cached)
                cached = true;
            else if (argument == "--per-thread" && !per_thread)
                per_thread = true;
            else
                throw std::runtime_error("usage: unified-copy-gpu-test [--cached] [--per-thread]");
        }
        experiment = std::make_unique<Experiment>();
        experiment->uses_per_thread = per_thread;
        experiment->start(cached);
        for (const auto bytes : {std::size_t{32}, std::size_t{65}})
            for (std::size_t mode = 0; mode != kModes.size(); ++mode)
                for (std::size_t route = 0; route != kRoutes.size(); ++route)
                    experiment->run_case(bytes, mode, route);
        experiment->close();
        expect(experiment->completed_cases == kCaseCount && experiment->sync_calls == 170 &&
                   experiment->async_calls == 30,
               "complete fixed matrix and every original generic setup/case/readback call");
        std::cout << "{\"record\":"
                  << quote(per_thread ? "unified_copy_per_thread_result" : "unified_copy_result")
                  << ",\"schema\":1,\"cached\":" << (cached ? "true" : "false")
                  << ",\"gpu_uuid\":" << quote(experiment->gpu_uuid)
                  << ",\"driver_version\":" << experiment->driver_version
                  << ",\"case_count\":" << kCaseCount << ",\"storage_bytes\":" << kStorageBytes
                  << ",\"guard_bytes\":" << kGuardBytes << ",\"image_count\":" << kImageCount
                  << ",\"checked_bytes\":" << kCaseCount * kImageCount * kStorageBytes
                  << ",\"original_copy_calls\":200,\"sync_copy_calls\":170,"
                     "\"async_copy_calls\":30,\"normal_cleanup\":true,"
                     "\"full_device_qualified\":false,\"priority_qualified\":false,"
                     "\"formal_comparison_qualified\":false,\"cases\":["
                  << experiment->cases.str() << "]}\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        bool is_clean{};
        std::string cleanup_error;
        if (experiment && !experiment->is_uncertain) {
            try {
                experiment->close();
                is_clean = experiment->is_closed;
            } catch (const std::exception& failure) {
                cleanup_error = failure.what();
            }
        }
        std::cout << "{\"record\":\"unified_copy_failure\",\"schema\":1,\"error\":"
                  << quote(error.what()) << ",\"cleanup_error\":" << quote(cleanup_error)
                  << ",\"completed_cases\":" << (experiment ? experiment->completed_cases : 0)
                  << ",\"normal_cleanup\":" << (is_clean ? "true" : "false") << "}\n"
                  << std::flush;
        if (experiment && !is_clean) {
            // Preserve all host images and handles after an uncertain SDK
            // result. Process exit is containment, not successful retirement.
            (void)experiment.release();
            std::_Exit(EXIT_FAILURE);
        }
        return EXIT_FAILURE;
    }
}
