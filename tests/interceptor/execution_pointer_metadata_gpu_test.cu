#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <sys/mman.h>

__device__ unsigned int metadata_global[8];

// Ordinary registered Runtime input: no supplied PTX, Glimmer API or scheduling
// interface. The original logical launch must survive private transformation.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): public kernel parameter ABI.
__global__ void pointer_metadata_kernel(std::uint32_t* output, unsigned int count,
                                        unsigned int seed) {
    if (gridDim.x != 1024 || gridDim.y != 1 || gridDim.z != 1 || blockDim.x != 128 ||
        blockDim.y != 1 || blockDim.z != 1)
        return;
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) output[index] = output[index] * 1664525U + 1013904223U + seed;
}

namespace {
constexpr unsigned int kBlocks = 1024, kThreads = 128, kGuard = 64;
constexpr std::size_t kValues = std::size_t{kBlocks} * kThreads;
constexpr std::uint32_t kMarker = 0xa9c5e137U;
constexpr CUdeviceptr kPointerPoison = 0x1234abcddcba4321ULL;
constexpr int kOrdinalPoison = 0x12345678;
bool is_uncertain{}, has_cleanup_failure{};

void expect(bool condition, const char* message) {
    if (!condition) {
        is_uncertain = true;
        throw std::runtime_error(message);
    }
}
void checked(int status, const char* operation) {
    if (status) {
        is_uncertain = true;
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(status));
    }
}
std::uintptr_t bits(const void* pointer) {
    return reinterpret_cast<std::uintptr_t>(pointer);
}
struct Resources {
    void* device{};
    void* host{};
    std::uint32_t* compute{};
    cudaStream_t stream{};
    void* protected_page{MAP_FAILED};
    CUcontext original{}, secondary{};

    Resources() = default;
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() {
        // A failed call/proof never licenses more native cleanup. The process
        // owner retains those resources until exit, with no success receipt.
        const auto clean = [](int status) {
            if (!status) return true;
            has_cleanup_failure = true;
            is_uncertain = true;
            return false;
        };
        if (!is_uncertain && secondary) {
            if (clean(cuCtxSetCurrent(secondary)) && clean(cuCtxDestroy(secondary)))
                secondary = nullptr;
        }
        if (!is_uncertain && original && clean(cuCtxSetCurrent(original))) {
            if (stream && clean(cudaStreamDestroy(stream))) stream = nullptr;
            if (!is_uncertain && compute && clean(cudaFree(compute))) compute = nullptr;
            if (!is_uncertain && host && clean(cudaFreeHost(host))) host = nullptr;
            if (!is_uncertain && device && clean(cudaFree(device))) device = nullptr;
            if (!is_uncertain && clean(cudaDeviceReset())) original = nullptr;
        }
        if (protected_page != MAP_FAILED && munmap(protected_page, 4096) != 0)
            has_cleanup_failure = true;
    }
};
using ContextSample = std::array<int, 2>;
ContextSample current_context(const Resources& resources) {
    CUcontext current{};
    const int status = cuCtxGetCurrent(&current);
    int identity{};
    if (current) {
        if (current == resources.original)
            identity = 1;
        else if (current == resources.secondary)
            identity = 2;
        else
            expect(false, "metadata query exposed an unexpected current context");
    }
    return {status, identity};
}
int alias(std::uintptr_t value, std::uintptr_t input, bool allows_preserve = false) {
    if (allows_preserve && value == kPointerPoison) return -1;
    if (!value) return 0;
    expect(input && value == input, "pointer metadata exposed an unrelated raw alias");
    return 1;
}
enum class Kind { kUnknown, kDevice, kGlobal, kHost };
struct Row {
    const char* name{};
    std::array<int, 2> driver3{}, driver9{};
    std::array<int, 5> runtime{};
    std::array<ContextSample, 3> contexts{}, contexts_after{};
};
Row observe(const Resources& resources, const char* name, const void* pointer, Kind kind,
            int context, bool is_uninitialized = false) {
    const bool is_unknown = kind == Kind::kUnknown;
    const bool is_accessible = context != 0 && !(kind == Kind::kGlobal && context == 2);
    const int device_status = is_uninitialized ? 3
                              : is_unknown     ? 1
                              : context == 0   ? 201
                              : is_accessible  ? 0
                                               : 1;
    const int pointer_effect = is_unknown ? -1 : is_accessible ? 1 : 0;
    const int ordinal_status = is_uninitialized ? 3 : is_unknown ? 1 : 0;
    const int type = is_unknown ? 0 : kind == Kind::kHost ? 1 : 2;
    const std::array<int, 5> expected_runtime{
        0, type, is_unknown ? -2 : 0,
        is_unknown      ? 0
        : is_accessible ? 1
                        : 0,
        (is_unknown || kind == Kind::kHost) && pointer ? 1 : 0};
    const auto require = [name](bool condition, const char* operation) {
        if (!condition) {
            is_uncertain = true;
            throw std::runtime_error(std::string(name) + ": " + operation);
        }
    };
    const auto sample = [&](bool after_runtime = false) {
        const auto value = current_context(resources);
        require(value == ContextSample{is_uninitialized && !after_runtime ? 3 : 0, context},
                "unexpected current-context or initialization observation");
        return value;
    };
    struct DriverOutput {
        std::uint64_t before{0xc1a258d0793e46bfULL};
        CUdeviceptr device{kPointerPoison};
        int ordinal{kOrdinalPoison};
        std::uint64_t after{0xba457f36281390deULL};
    } driver;
    struct RuntimeOutput {
        std::uint64_t before{0x7a245bd3ce6f1809ULL};
        cudaPointerAttributes value;
        std::uint64_t after{0x6b7e98a1053dc24fULL};
    } runtime;
    std::memset(&runtime.value, 0x5a, sizeof(runtime.value));
    expect(bits(pointer) != kPointerPoison, "query input and untouched-output poison are distinct");
    Row row;
    row.name = name;
    row.contexts[0] = sample();
    row.driver3[0] = cuPointerGetAttribute(&driver.device, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                           static_cast<CUdeviceptr>(bits(pointer)));
    row.contexts_after[0] = sample();
    row.driver3[1] = alias(driver.device, bits(pointer), true);
    require(row.driver3 == std::array<int, 2>{device_status, pointer_effect},
            "Driver3 exact native status or output effect mismatch");
    row.contexts[1] = sample();
    row.driver9[0] = cuPointerGetAttribute(&driver.ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                           static_cast<CUdeviceptr>(bits(pointer)));
    row.contexts_after[1] = sample();
    row.driver9[1] = driver.ordinal;
    require(row.driver9 == std::array<int, 2>{ordinal_status, is_unknown ? kOrdinalPoison : 0},
            "Driver9 exact native status or output effect mismatch");
    row.contexts[2] = sample();
    row.runtime[0] = cudaPointerGetAttributes(&runtime.value, pointer);
    row.contexts_after[2] = sample(true);
    require(row.runtime[0] == 0, "Runtime original pointer query failed");
    static_assert(sizeof(row.runtime[1]) == sizeof(runtime.value.type));
    std::memcpy(&row.runtime[1], &runtime.value.type, sizeof(row.runtime[1]));
    row.runtime[2] = runtime.value.device;
    row.runtime[3] = alias(bits(runtime.value.devicePointer), bits(pointer));
    row.runtime[4] = alias(bits(runtime.value.hostPointer), bits(pointer));

    require(row.runtime == expected_runtime,
            "Runtime exact native fields or normalized aliases mismatch");
    // Reserved bytes and struct padding are deliberately not compared.
    expect(driver.before == 0xc1a258d0793e46bfULL && driver.after == 0xba457f36281390deULL &&
               runtime.before == 0x7a245bd3ce6f1809ULL && runtime.after == 0x6b7e98a1053dc24fULL,
           "every pointer metadata output preserves outer guards");
    return row;
}
std::uint64_t checksum(const std::vector<std::uint32_t>& values) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : values)
        for (unsigned int shift = 0; shift != 32; shift += 8) {
            hash ^= (value >> shift) & 255U;
            hash *= 1099511628211ULL;
        }
    return hash;
}
std::uint64_t compute(Resources& resources, std::vector<std::uint32_t>& expected,
                      unsigned int seed) {
    auto* output = resources.compute + kGuard;
    auto count = static_cast<unsigned int>(kValues);
    const auto original_seed = seed;
    std::array<void*, 3> arguments{&output, &count, &seed};
    checked(cudaLaunchKernel(reinterpret_cast<const void*>(&pointer_metadata_kernel), dim3{kBlocks},
                             dim3{kThreads}, arguments.data(), 0, resources.stream),
            "launch one non-idempotent original logical kernel");
    checked(cudaStreamSynchronize(resources.stream), "prove logical kernel completion");
    expect(arguments == std::array<void*, 3>{&output, &count, &seed} &&
               output == resources.compute + kGuard && count == kValues && seed == original_seed,
           "public Runtime parameter slots and values remain unchanged");
    for (std::size_t index = 0; index != kValues; ++index)
        expected[kGuard + index] =
            expected[kGuard + index] * 1664525U + 1013904223U + original_seed;
    std::vector<std::uint32_t> actual(expected.size());
    checked(cudaMemcpy(actual.data(), resources.compute, actual.size() * sizeof(actual[0]),
                       cudaMemcpyDeviceToHost),
            "read every output and guard");
    expect(actual == expected, "complete non-idempotent output and guard oracle matches exactly");
    return checksum(actual);
}
struct Receipt {
    std::vector<Row> rows;
    std::array<int, 3> null_output_statuses{};
    std::array<std::uint64_t, 2> checksums{};
};
Receipt run() {
    Resources resources;
    Receipt receipt;
    receipt.rows.reserve(30);
    auto& rows = receipt.rows;
    std::uint32_t marker = 0x275af084U;
    rows.push_back(observe(resources, "uninitialized-null", nullptr, Kind::kUnknown, 0, true));
    checked(cuInit(0), "initialize Driver");
    checked(cuCtxSetCurrent(nullptr), "detach before no-current unknown observations");
    rows.push_back(observe(resources, "detached-null", nullptr, Kind::kUnknown, 0));
    rows.push_back(observe(resources, "detached-stack", &marker, Kind::kUnknown, 0));
    checked(cudaSetDevice(0), "select original Runtime primary device");
    checked(cuCtxGetCurrent(&resources.original), "capture original primary context");
    expect(resources.original != nullptr, "Runtime device selection supplies primary context A");

    std::vector<std::uint32_t> expected(kValues + std::size_t{2} * kGuard, kMarker);
    for (std::size_t index = 0; index != kValues; ++index)
        expected[kGuard + index] = 0x12345678U + static_cast<std::uint32_t>(index) * 17U;
    checked(cudaMalloc(reinterpret_cast<void**>(&resources.compute),
                       expected.size() * sizeof(expected[0])),
            "allocate entire computation and guards");
    checked(cudaMemcpy(resources.compute, expected.data(), expected.size() * sizeof(expected[0]),
                       cudaMemcpyHostToDevice),
            "initialize complete computation input");
    checked(cudaDeviceSynchronize(), "order pageable upload before nonblocking stream");
    checked(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking),
            "create computation stream");
    receipt.checksums[0] = compute(resources, expected, 17);

    std::vector<std::uint32_t> heap(16, marker);
    resources.protected_page = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(resources.protected_page != MAP_FAILED,
           "allocate protected unknown pointee without reading it");
    rows.push_back(observe(resources, "null", nullptr, Kind::kUnknown, 1));
    rows.push_back(observe(resources, "stack", &marker, Kind::kUnknown, 1));
    rows.push_back(observe(resources, "heap", heap.data(), Kind::kUnknown, 1));
    rows.push_back(
        observe(resources, "protected-page", resources.protected_page, Kind::kUnknown, 1));
    rows.push_back(observe(resources, "low-invalid", reinterpret_cast<void*>(std::uintptr_t{1}),
                           Kind::kUnknown, 1));
    rows.push_back(observe(resources, "high-inaccessible",
                           reinterpret_cast<void*>(std::uintptr_t{0xffff800000001234ULL}),
                           Kind::kUnknown, 1));
    rows.push_back(observe(resources, "noncanonical-bits",
                           reinterpret_cast<void*>(std::uintptr_t{0x0200000000001234ULL}),
                           Kind::kUnknown, 1));

    checked(cudaMalloc(&resources.device, 256), "allocate metadata device input");
    const auto device_bits = bits(resources.device);
    constexpr std::array<std::pair<const char*, std::uintptr_t>, 5> offsets{
        {{"device-base", 0},
         {"device-interior", 17},
         {"device-last-byte", 255},
         {"device-endpoint", 256},
         {"device-beyond-end", 257}}};
    for (const auto& [name, offset] : offsets)
        rows.push_back(observe(resources, name, reinterpret_cast<void*>(device_bits + offset),
                               offset < 256 ? Kind::kDevice : Kind::kUnknown, 1));
    void* global{};
    checked(cudaGetSymbolAddress(&global, metadata_global),
            "resolve original registered module global");
    rows.push_back(observe(resources, "global-base", global, Kind::kGlobal, 1));
    rows.push_back(observe(resources, "global-interior", reinterpret_cast<void*>(bits(global) + 4),
                           Kind::kGlobal, 1));
    rows.push_back(observe(resources, "global-endpoint", reinterpret_cast<void*>(bits(global) + 32),
                           Kind::kUnknown, 1));
    checked(cudaHostAlloc(&resources.host, 256, cudaHostAllocMapped), "allocate mapped host input");
    std::memset(resources.host, 0x6d, 256);
    void* host_device{};
    checked(cudaHostGetDevicePointer(&host_device, resources.host, 0), "resolve mapped host alias");
    expect(host_device == resources.host,
           "configured UVA host allocation has the public input alias");
    rows.push_back(observe(resources, "host-base", resources.host, Kind::kHost, 1));
    rows.push_back(observe(resources, "host-interior",
                           reinterpret_cast<void*>(bits(resources.host) + 17), Kind::kHost, 1));
    rows.push_back(observe(resources, "host-endpoint",
                           reinterpret_cast<void*>(bits(resources.host) + 256), Kind::kUnknown, 1));
    rows.push_back(observe(resources, "host-device-alias", host_device, Kind::kHost, 1));
    receipt.null_output_statuses = {
        static_cast<int>(cuPointerGetAttribute(nullptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                               static_cast<CUdeviceptr>(device_bits))),
        static_cast<int>(cuPointerGetAttribute(nullptr, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                               static_cast<CUdeviceptr>(device_bits))),
        static_cast<int>(cudaPointerGetAttributes(nullptr, resources.device))};
    expect(receipt.null_output_statuses == std::array<int, 3>{1, 1, 1},
           "all three NULL outputs report exact native invalid value");

    checked(cuCtxSetCurrent(nullptr), "detach before live device metadata");
    rows.push_back(observe(resources, "detached-live-device", resources.device, Kind::kDevice, 0));
    checked(cuCtxSetCurrent(nullptr), "detach before live global metadata");
    rows.push_back(observe(resources, "detached-live-global", global, Kind::kGlobal, 0));
    checked(cuCtxSetCurrent(nullptr), "detach before live host metadata");
    rows.push_back(observe(resources, "detached-live-host", resources.host, Kind::kHost, 0));
    checked(cuCtxSetCurrent(resources.original), "restore A before creating B");
    CUdevice device{};
    checked(cuDeviceGet(&device, 0), "select secondary same-GPU device");
    checked(cuCtxCreate(&resources.secondary, 0, device), "create distinct current context B");
    expect(resources.secondary && resources.secondary != resources.original,
           "contexts A and B are distinct");
    rows.push_back(observe(resources, "other-context-device", resources.device, Kind::kDevice, 2));
    rows.push_back(observe(resources, "other-context-global", global, Kind::kGlobal, 2));
    rows.push_back(observe(resources, "other-context-host", resources.host, Kind::kHost, 2));
    checked(cuCtxSetCurrent(resources.original), "restore A before second original computation");
    receipt.checksums[1] = compute(resources, expected, 40);
    checked(cudaFree(resources.device), "release metadata device input exactly once");
    resources.device = nullptr;
    rows.push_back(observe(resources, "freed-device", reinterpret_cast<void*>(device_bits),
                           Kind::kUnknown, 1));
    const auto host_bits = bits(resources.host);
    for (std::size_t index = 0; index != 256; ++index)
        expect(static_cast<const unsigned char*>(resources.host)[index] == 0x6d,
               "metadata preserves every mapped-host input byte");
    checked(cudaFreeHost(resources.host), "release mapped host input exactly once");
    resources.host = nullptr;
    rows.push_back(
        observe(resources, "freed-host", reinterpret_cast<void*>(host_bits), Kind::kUnknown, 1));
    expect(marker == 0x275af084U && heap == std::vector<std::uint32_t>(16, marker),
           "metadata preserves stack and heap inputs");
    expect(rows.size() == 30, "complete fixed native pointer metadata matrix");
    return receipt;
}
template <std::size_t Size>
void write_array(const std::array<int, Size>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index != Size; ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}
void write_receipt(const Receipt& receipt) {
    std::cout << "{\"status\":\"passed\",\"rows\":[";
    for (std::size_t index = 0; index != receipt.rows.size(); ++index) {
        if (index) std::cout << ',';
        const auto& row = receipt.rows[index];
        std::cout << "{\"name\":\"" << row.name << "\",\"driver3\":";
        write_array(row.driver3);
        std::cout << ",\"driver9\":";
        write_array(row.driver9);
        std::cout << ",\"runtime\":";
        write_array(row.runtime);
        for (bool after : {false, true}) {
            std::cout << (after ? ",\"contexts_after\":[" : ",\"contexts\":[");
            const auto& samples = after ? row.contexts_after : row.contexts;
            for (std::size_t call = 0; call != samples.size(); ++call) {
                if (call) std::cout << ',';
                write_array(samples[call]);
            }
            std::cout << ']';
        }
        std::cout << '}';
    }
    std::cout << "],\"null_output_statuses\":";
    write_array(receipt.null_output_statuses);
    std::cout
        << ",\"checksums\":[" << receipt.checksums[0] << ',' << receipt.checksums[1]
        << "],\"checked_values\":262400,\"logical_calls\":2,\"replays\":0,"
           "\"input_unchanged\":true,\"guards\":true,\"parameters_unchanged\":true,"
           "\"cleanup\":true,\"full_device_qualified\":false,\"performance_qualified\":false}\n";
}
}  // namespace
int main() {
    try {
        const auto receipt = run();
        expect(!is_uncertain && !has_cleanup_failure,
               "all native resources cleaned up successfully");
        write_receipt(receipt);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "pointer metadata SDK input: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
