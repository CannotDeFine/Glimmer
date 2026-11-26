#include <cuda.h>
#include <cublas_v2.h>

#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>

// One isolated original SDK call. Inaccessible inputs distinguish observed
// no-read behavior from a call that merely leaves accessible output unchanged.
// These diagnostic observations do not qualify compatibility or performance.
namespace {
constexpr std::size_t kGuard = 16, kPayload = 128, kElements = 160;
constexpr std::size_t kStorageBytes = kElements * sizeof(float), kScalarElements = 34;
constexpr std::uint64_t kHashSeed = UINT64_C(14695981039346656037);
using Storage = std::array<std::byte, kStorageBytes>;
using Images = std::array<Storage, 3>;
using Scalars = std::array<float, kScalarElements>;
using ScalarImages = std::array<Scalars, 2>;

void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void checked(CUresult status) {
    if (status != CUDA_SUCCESS) throw std::runtime_error("Driver status " + std::to_string(status));
}
void checked(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("BLAS status " + std::to_string(status));
}
std::string quoted(std::string_view text) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char value : text) {
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
    return result + '"';
}
void flush_record() {
    std::cout << '\n' << std::flush;
    expect(std::cout.good(), "stdout record publication failed");
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

struct Probe {
    std::string_view id;
    bool is_vector{};
    int transpose_a{}, transpose_b{}, type_a{}, type_b{}, type_c{};
    int rows{}, columns{}, depth{}, leading_a{}, leading_b{}, leading_c{};
    int increment_x{}, increment_y{};
    bool is_zero_depth{};
};
std::vector<Probe> probes() {
    const Probe vector{"", true, 0, 0, 0, 0, 0, 5, 7, 0, 8, 0, 0, 1, 1, false};
    const Probe matrix{"", false, 0, 0, 0, 0, 0, 3, 4, 5, 8, 8, 8, 0, 0, false};
    std::vector<Probe> result;
    auto add = [&](std::string_view id, Probe probe, int Probe::*field, int value) {
        probe.id = id;
        probe.*field = value;
        result.push_back(probe);
    };
    add("sgemv_zero_rows", vector, &Probe::rows, 0);
    add("sgemv_zero_columns", vector, &Probe::columns, 0);
    add("sgemv_negative_rows", vector, &Probe::rows, -1);
    add("sgemv_negative_columns", vector, &Probe::columns, -1);
    add("sgemv_invalid_transpose", vector, &Probe::transpose_a, 3);
    add("sgemv_bad_lda", vector, &Probe::leading_a, 1);
    add("sgemv_zero_increment_x", vector, &Probe::increment_x, 0);
    add("sgemv_zero_increment_y", vector, &Probe::increment_y, 0);
    add("sgemm_ex_zero_rows", matrix, &Probe::rows, 0);
    add("sgemm_ex_zero_columns", matrix, &Probe::columns, 0);
    add("sgemm_ex_negative_rows", matrix, &Probe::rows, -1);
    add("sgemm_ex_negative_columns", matrix, &Probe::columns, -1);
    add("sgemm_ex_negative_depth", matrix, &Probe::depth, -1);
    add("sgemm_ex_invalid_transpose_a", matrix, &Probe::transpose_a, 3);
    add("sgemm_ex_invalid_transpose_b", matrix, &Probe::transpose_b, 3);
    add("sgemm_ex_bad_lda", matrix, &Probe::leading_a, 1);
    add("sgemm_ex_bad_ldb", matrix, &Probe::leading_b, 1);
    add("sgemm_ex_bad_ldc", matrix, &Probe::leading_c, 1);
    auto unsupported = matrix;
    unsupported.type_a = unsupported.type_b = unsupported.type_c = 1;
    unsupported.id = "sgemm_ex_unsupported_f64";
    result.push_back(unsupported);
    auto zero_depth = matrix;
    zero_depth.id = "sgemm_ex_zero_depth";
    zero_depth.depth = 0;
    zero_depth.is_zero_depth = true;
    result.push_back(zero_depth);
    return result;
}
struct Case {
    Probe probe;
    std::string_view mode, pointers;
    [[nodiscard]] bool is_device() const {
        return mode == "device";
    }
    [[nodiscard]] bool is_protected() const {
        return pointers.ends_with("protected");
    }
    [[nodiscard]] bool has_scalars() const {
        return !pointers.ends_with("null");
    }
    [[nodiscard]] bool has_data() const {
        return pointers.starts_with("owned");
    }
    [[nodiscard]] unsigned presence() const {
        return (has_scalars() ? 3U : 0U) | (has_data() ? 28U : 0U) |
               (probe.is_zero_depth ? 16U : 0U);
    }
};
std::vector<Case> inventory() {
    std::vector<Case> result;
    for (const auto& probe : probes()) {
        for (const auto mode : {"host", "device"}) {
            if (probe.is_zero_depth) {
                result.push_back({probe, mode, "zero-depth-real"});
                continue;
            }
            for (const auto variant : {"owned-real", "null-real", "owned-null", "null-null",
                                       "owned-protected", "null-protected"}) {
                const Case candidate{probe, mode, variant};
                if (!candidate.is_device() || !candidate.is_protected())
                    result.push_back(candidate);
            }
        }
    }
    return result;
}
void identity(const Case& test) {
    std::cout << "\"id\":" << quoted(test.probe.id) << ",\"mode\":" << quoted(test.mode)
              << ",\"pointers\":" << quoted(test.pointers);
}
void descriptor(const Case& test) {
    identity(test);
    const auto& p = test.probe;
    std::cout << ",\"api\":" << quoted(p.is_vector ? "cublasSgemv_v2" : "cublasSgemmEx")
              << ",\"arguments\":[\"--probe\"," << quoted(p.id) << ",\"--mode\","
              << quoted(test.mode) << ",\"--pointers\"," << quoted(test.pointers)
              << ",\"--start-gate\"]"
              << ",\"metadata\":{\"transpose_a\":" << p.transpose_a
              << ",\"transpose_b\":" << p.transpose_b << ",\"type_a\":" << p.type_a
              << ",\"type_b\":" << p.type_b << ",\"type_c\":" << p.type_c << ",\"rows\":" << p.rows
              << ",\"columns\":" << p.columns << ",\"depth\":" << p.depth
              << ",\"leading_a\":" << p.leading_a << ",\"leading_b\":" << p.leading_b
              << ",\"leading_c\":" << p.leading_c << ",\"increment_x\":" << p.increment_x
              << ",\"increment_y\":" << p.increment_y << "},\"presence\":" << test.presence()
              << ",\"host_scalar_protected\":" << (test.is_protected() ? "true" : "false")
              << ",\"zero_depth_beta_c\":" << (p.is_zero_depth ? "true" : "false")
              << ",\"storage_bytes\":640,\"payload_elements\":128,\"guard_elements\":16"
                 ",\"scalar_elements\":34,\"alpha\":2,\"beta\":0.5,\"checked_bytes\":2192";
}
void list_inventory() {
    std::cout << "{\"record\":\"blas_boundary_inventory\",\"schema\":1,\"cases\":[";
    bool first = true;
    for (const auto& test : inventory()) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << '{';
        descriptor(test);
        std::cout << '}';
    }
    std::cout << "]}";
    flush_record();
}

struct Library {
    void* handle{};
    bool is_retained{};
    decltype(&cublasSgemv) gemv{};
    decltype(&cublasSgemmEx) gemm{};
    void open(const char* path) {
        handle = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
        expect(handle != nullptr, "original cuBLAS provider load failed");
        gemv = reinterpret_cast<decltype(gemv)>(::dlsym(handle, "cublasSgemv_v2"));
        gemm = reinterpret_cast<decltype(gemm)>(::dlsym(handle, "cublasSgemmEx"));
        expect(gemv && gemm && gemv == &cublasSgemv && gemm == &cublasSgemmEx,
               "exact direct/cached SDK export identity");
    }
    void close() {
        if (!handle || is_retained) return;
        auto* original = handle;
        handle = nullptr;
        expect(::dlclose(original) == 0, "original provider close failed");
    }
    ~Library() {
        try {
            close();
        } catch (...) {
        }
    }
};
struct ProtectedPage {
    void* address{MAP_FAILED};
    std::size_t bytes{};
    bool is_retained{};
    void create() {
        const long page_bytes = ::sysconf(_SC_PAGESIZE);
        expect(page_bytes > 0, "page size unavailable");
        bytes = static_cast<std::size_t>(page_bytes);
        address = ::mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(address != MAP_FAILED, "inaccessible scalar mapping failed");
    }
    void close() {
        if (address == MAP_FAILED || is_retained) return;
        expect(::munmap(address, bytes) == 0, "inaccessible scalar unmap failed");
        address = MAP_FAILED;
    }
    ~ProtectedPage() {
        try {
            close();
        } catch (...) {
        }
    }
};
struct Native {
    CUcontext context{};
    CUstream stream{};
    cublasHandle_t handle{};
    std::array<CUdeviceptr, 3> matrices{};
    CUdeviceptr scalars{};
    std::string gpu_uuid;
    int driver_version{}, cublas_version{};
    bool is_uncertain{};
    void owned(CUresult status) {
        if (status != CUDA_SUCCESS) {
            is_uncertain = true;
            checked(status);
        }
    }
    void owned(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            is_uncertain = true;
            checked(status);
        }
    }
    void require_owner(bool value, const char* message) {
        if (!value) {
            is_uncertain = true;
            throw std::runtime_error(message);
        }
    }
    void proof() {
        CUcontext current{};
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != context) {
            is_uncertain = true;
            throw std::runtime_error("current context unproven before completion");
        }
        owned(cuCtxSynchronize());
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != context) {
            is_uncertain = true;
            throw std::runtime_error("current context unproven after completion");
        }
    }
    void start() {
        checked(cuInit(0));
        checked(cuDriverGetVersion(&driver_version));
        CUdevice device{};
        checked(cuDeviceGet(&device, 0));
        owned(cuCtxCreate(&context, 0, device));
        require_owner(context != nullptr, "successful context is NULL");
        CUdevice actual{};
        CUuuid uuid{};
        owned(cuCtxGetDevice(&actual));
        owned(cuDeviceGetUuid(&uuid, actual));
        gpu_uuid = uuid_text(uuid);
        owned(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        owned(cublasCreate(&handle));
        require_owner(stream && handle, "successful stream/handle is NULL");
        owned(cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
        owned(cublasGetVersion(handle, &cublas_version));
        expect(cublas_version == 120604, "access observations require cuBLAS 12.6.4");
        for (auto& pointer : matrices) {
            owned(cuMemAlloc(&pointer, kStorageBytes));
            require_owner(pointer != 0, "successful data allocation is NULL");
        }
        owned(cuMemAlloc(&scalars, sizeof(Scalars)));
        require_owner(scalars != 0, "successful scalar allocation is NULL");
    }
    void close() {
        expect(!is_uncertain, "physical ownership is uncertain; cleanup withheld");
        if (!context) return;
        proof();
        if (handle) {
            owned(cublasDestroy(handle));
            handle = nullptr;
        }
        for (auto& pointer : matrices) {
            if (pointer) {
                owned(cuMemFree(pointer));
                pointer = 0;
            }
        }
        if (scalars) {
            owned(cuMemFree(scalars));
            scalars = 0;
        }
        if (stream) {
            owned(cuStreamDestroy(stream));
            stream = nullptr;
        }
        owned(cuCtxDestroy(context));
        context = nullptr;
    }
    ~Native() {
        if (!is_uncertain && context) {
            try {
                close();
            } catch (...) {
                is_uncertain = true;
            }
        }
    }
};
void initial_images(Images& images, Scalars& scalars) {
    for (std::size_t matrix = 0; matrix != images.size(); ++matrix) {
        auto& image = images[matrix];
        image.fill(std::byte{0xa5});
        for (std::size_t index = 0; index != kPayload; ++index) {
            const float value =
                matrix == 2 ? 6.0F
                            : static_cast<float>(
                                  static_cast<int>((index * (matrix + 1) + 3 * matrix) % 7) - 3);
            std::memcpy(image.data() + (kGuard + index) * sizeof(float), &value, sizeof(value));
        }
    }
    scalars.fill(-4096.0F);
    scalars[kGuard] = 2.0F;
    scalars[kGuard + 1] = 0.5F;
}
std::uint64_t checksum(const Images& images, const ScalarImages& scalars) {
    auto hash = kHashSeed;
    auto append = [&](const void* data, std::size_t bytes) {
        const auto* input = static_cast<const unsigned char*>(data);
        for (std::size_t index = 0; index != bytes; ++index) {
            hash ^= input[index];
            hash *= UINT64_C(1099511628211);
        }
    };
    for (const auto& image : images) append(image.data(), image.size());
    for (const auto& image : scalars) append(image.data(), sizeof(image));
    return hash;
}
void await_run() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    std::string input;
    while (input.size() < 4) {
        const auto remaining = deadline - std::chrono::steady_clock::now();
        expect(remaining > std::chrono::steady_clock::duration::zero(), "start gate timed out");
        const auto timeout_ms = std::chrono::ceil<std::chrono::milliseconds>(remaining).count();
        pollfd descriptor{STDIN_FILENO, POLLIN, 0};
        const int ready = ::poll(&descriptor, 1, static_cast<int>(timeout_ms));
        if (ready < 0 && errno == EINTR) continue;
        expect(ready > 0 && (descriptor.revents & (POLLIN | POLLHUP)) != 0 &&
                   (descriptor.revents & (POLLERR | POLLNVAL)) == 0,
               "start gate unavailable or timed out");
        std::array<char, 5> bytes{};
        const auto count = ::read(STDIN_FILENO, bytes.data(), bytes.size());
        if (count < 0 && errno == EINTR) continue;
        expect(count > 0, "start gate closed before run");
        input.append(bytes.data(), static_cast<std::size_t>(count));
        expect(input.size() <= 4 && std::string_view("run\n").starts_with(input),
               "start gate requires exactly run newline");
    }
    expect(input == "run\n", "start gate command mismatch");
}
struct Runner {
    Library library;
    ProtectedPage protected_page;
    Native native;
    std::string_view step{"provider"};
    bool is_cleanup_failed{};
    void close() {
        expect(!is_cleanup_failed, "prior cleanup failed; repeated cleanup withheld");
        try {
            native.close();
            protected_page.close();
            library.close();
        } catch (...) {
            is_cleanup_failed = true;
            native.is_uncertain = true;
            protected_page.is_retained = true;
            library.is_retained = true;
            throw;
        }
    }
    bool failure_cleanup() noexcept {
        try {
            close();
            return true;
        } catch (...) {
            native.is_uncertain = true;
            protected_page.is_retained = true;
            library.is_retained = true;
            return false;
        }
    }
    void hardware_identity() const {
        std::cout << ",\"pid\":" << ::getpid() << ",\"gpu_uuid\":" << quoted(native.gpu_uuid)
                  << ",\"driver_version\":" << native.driver_version
                  << ",\"cublas_version\":" << native.cublas_version;
    }
    cublasStatus_t invoke(const Case& test, Scalars& host_scalars) {
        const auto& p = test.probe;
        const float* alpha{};
        const float* beta{};
        if (test.has_scalars()) {
            if (test.is_protected()) {
                alpha = beta = static_cast<const float*>(protected_page.address);
            } else if (test.is_device()) {
                alpha = std::bit_cast<const float*>(native.scalars + kGuard * sizeof(float));
                beta = std::bit_cast<const float*>(native.scalars + (kGuard + 1) * sizeof(float));
            } else {
                alpha = &host_scalars[kGuard];
                beta = &host_scalars[kGuard + 1];
            }
        }
        const float* a{};
        const float* b{};
        float* c{};
        if (test.has_data()) {
            a = std::bit_cast<const float*>(native.matrices[0] + kGuard * sizeof(float));
            b = std::bit_cast<const float*>(native.matrices[1] + kGuard * sizeof(float));
        }
        if (test.has_data() || p.is_zero_depth)
            c = std::bit_cast<float*>(native.matrices[2] + kGuard * sizeof(float));
        if (p.is_vector)
            return library.gemv(native.handle, static_cast<cublasOperation_t>(p.transpose_a),
                                p.rows, p.columns, alpha, a, p.leading_a, b, p.increment_x, beta, c,
                                p.increment_y);
        return library.gemm(native.handle, static_cast<cublasOperation_t>(p.transpose_a),
                            static_cast<cublasOperation_t>(p.transpose_b), p.rows, p.columns,
                            p.depth, alpha, a, static_cast<cudaDataType>(p.type_a), p.leading_a, b,
                            static_cast<cudaDataType>(p.type_b), p.leading_b, beta, c,
                            static_cast<cudaDataType>(p.type_c), p.leading_c);
    }
    void run(const Case& test, const char* provider, bool has_start_gate) {
        library.open(provider);
        step = "native_setup";
        native.start();
        const auto mode = test.is_device() ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST;
        native.owned(cublasSetPointerMode(native.handle, mode));
        if (test.is_protected()) protected_page.create();
        Images before{}, observed{};
        Scalars scalars{}, device_scalars{};
        initial_images(before, scalars);
        const auto initial_scalars = scalars;
        for (std::size_t index = 0; index != before.size(); ++index)
            native.owned(cuMemcpyHtoD(native.matrices[index], before[index].data(), kStorageBytes));
        native.owned(cuMemcpyHtoD(native.scalars, scalars.data(), sizeof(scalars)));
        native.proof();
        if (has_start_gate) {
            step = "awaiting_run";
            std::cout << "{\"record\":\"blas_boundary_ready\",\"schema\":1,";
            identity(test);
            hardware_identity();
            std::cout << ",\"step\":\"awaiting_run\"}";
            flush_record();
            await_run();
        }
        step = "original_call";
        const auto status = invoke(test, scalars);
        std::cout << "{\"record\":\"blas_boundary_return\",\"schema\":1,";
        identity(test);
        std::cout << ",\"pid\":" << ::getpid() << ",\"status\":" << static_cast<int>(status)
                  << ",\"step\":\"before_physical_proof\",\"complete\":false}";
        flush_record();
        step = "physical_proof";
        native.proof();
        expect(static_cast<int>(status) >= 0 && static_cast<int>(status) <= 31,
               "original status outside SDK enum bounds");
        cublasPointerMode_t actual_mode{};
        cudaStream_t actual_stream{};
        native.owned(cublasGetPointerMode(native.handle, &actual_mode));
        native.owned(cublasGetStream(native.handle, &actual_stream));
        expect(actual_mode ==
                       (test.is_device() ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST) &&
                   actual_stream == reinterpret_cast<cudaStream_t>(native.stream),
               "original handle policy changed");
        step = "full_byte_oracle";
        for (std::size_t index = 0; index != observed.size(); ++index)
            native.owned(
                cuMemcpyDtoH(observed[index].data(), native.matrices[index], kStorageBytes));
        native.owned(cuMemcpyDtoH(device_scalars.data(), native.scalars, sizeof(device_scalars)));
        native.proof();
        auto expected = before;
        if (test.probe.is_zero_depth && status == CUBLAS_STATUS_SUCCESS) {
            for (int column = 0; column != test.probe.columns; ++column) {
                for (int row = 0; row != test.probe.rows; ++row) {
                    const auto element = static_cast<std::size_t>(column) *
                                             static_cast<std::size_t>(test.probe.leading_c) +
                                         static_cast<std::size_t>(row);
                    const float value = 3.0F;
                    std::memcpy(expected[2].data() + (kGuard + element) * sizeof(float), &value,
                                sizeof(value));
                }
            }
        }
        expect(observed == expected, "complete matrix/padding/guard image mismatch");
        using ScalarBytes = std::array<std::byte, sizeof(Scalars)>;
        expect(std::bit_cast<ScalarBytes>(scalars) == std::bit_cast<ScalarBytes>(initial_scalars) &&
                   std::bit_cast<ScalarBytes>(device_scalars) ==
                       std::bit_cast<ScalarBytes>(initial_scalars),
               "complete host/device scalar image mismatch");
        const auto before_hash = checksum(before, {initial_scalars, initial_scalars});
        const auto after_hash = checksum(observed, {scalars, device_scalars});
        const auto expected_hash = checksum(expected, {initial_scalars, initial_scalars});
        step = "normal_cleanup";
        close();
        std::cout << "{\"record\":\"blas_boundary_result\",\"schema\":1,";
        descriptor(test);
        hardware_identity();
        std::cout
            << ",\"status\":" << static_cast<int>(status)
            << ",\"step\":\"normal_cleanup\",\"normal_cleanup\":true,\"full_byte_guards\":true"
               ",\"input_unchanged\":true,\"scalar_images_unchanged\":true"
               ",\"all_values_verified\":true,\"before_checksum\":"
            << before_hash << ",\"after_checksum\":" << after_hash
            << ",\"expected_checksum\":" << expected_hash
            << ",\"full_device_qualified\":false,\"priority_qualified\":false"
               ",\"formal_comparison_qualified\":false}";
        flush_record();
    }
};
}  // namespace

int main(int argc, char** argv) {
    Runner runner;
    Case selected{};
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--list") {
            list_inventory();
            return 0;
        }
        expect(argc == 8 || argc == 9,
               "expected provider --probe ID --mode MODE --pointers VARIANT");
        expect(std::string_view(argv[2]) == "--probe" && std::string_view(argv[4]) == "--mode" &&
                   std::string_view(argv[6]) == "--pointers" &&
                   (argc == 8 || std::string_view(argv[8]) == "--start-gate"),
               "unknown or out-of-order argument");
        bool found = false;
        for (const auto& candidate : inventory()) {
            if (candidate.probe.id == argv[3] && candidate.mode == argv[5] &&
                candidate.pointers == argv[7]) {
                selected = candidate;
                found = true;
                break;
            }
        }
        expect(found, "probe/mode/pointer combination is outside fixed inventory");
        runner.run(selected, argv[1], argc == 9);
        return 0;
    } catch (const std::exception& error) {
        const bool is_clean = runner.failure_cleanup();
        std::cout << "{\"record\":\"blas_boundary_failure\",\"schema\":1,";
        identity(selected);
        std::cout << ",\"pid\":" << ::getpid() << ",\"step\":" << quoted(runner.step)
                  << ",\"error\":" << quoted(error.what())
                  << ",\"normal_cleanup\":" << (is_clean ? "true" : "false") << "}\n"
                  << std::flush;
        return 1;
    }
}
