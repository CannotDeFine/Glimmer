#include <cuda.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr int kRows = 64, kColumns = 96, kDepth = 32;
constexpr std::array<int, 4> kLeading{72, 40, 72, 80};
constexpr std::array<std::size_t, 6> kElements{2336, 3872, 6944, 7712, 96, 8};
constexpr std::array<std::string_view, 6> kNames{"a", "b", "c", "d", "bias", "scalars"};
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult result) {
    if (result != CUDA_SUCCESS) throw std::runtime_error("Driver status=" + std::to_string(result));
}
void check(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Lt status=" + std::to_string(result));
}
struct Cleanup {
    bool is_clean{true};
    void record(int status, const char* operation) noexcept {
        if (!status) return;
        is_clean = false;
        std::cerr << operation << " cleanup status=" << status << '\n';
    }
};
struct NativeOwner {
    NativeOwner() = default;
    NativeOwner(const NativeOwner&) = delete;
    NativeOwner& operator=(const NativeOwner&) = delete;
};
struct Context : private NativeOwner {
    Cleanup& cleanup;
    CUcontext value{};
    explicit Context(Cleanup& result) : cleanup(result) {
        check(cuInit(0));
        check(cuCtxCreate(&value, 0, 0));
    }
    ~Context() {
        if (value) cleanup.record(cuCtxDestroy(value), "context");
    }
};
struct Handle : private NativeOwner {
    Cleanup& cleanup;
    cublasLtHandle_t value{};
    explicit Handle(Cleanup& result) : cleanup(result) {
        check(cublasLtCreate(&value));
    }
    ~Handle() {
        if (value) cleanup.record(cublasLtDestroy(value), "Lt handle");
    }
};
struct Stream : private NativeOwner {
    Cleanup& cleanup;
    CUstream value{};
    explicit Stream(Cleanup& result) : cleanup(result) {
        check(cuStreamCreate(&value, CU_STREAM_NON_BLOCKING));
    }
    ~Stream() {
        if (value) cleanup.record(cuStreamDestroy(value), "stream");
    }
};
struct Descriptors : private NativeOwner {
    Cleanup& cleanup;
    cublasLtMatmulDesc_t operation{};
    cublasLtMatmulPreference_t preference{};
    std::array<cublasLtMatrixLayout_t, 4> layouts{};
    explicit Descriptors(Cleanup& result) : cleanup(result) {}
    ~Descriptors() {
        if (preference) cleanup.record(cublasLtMatmulPreferenceDestroy(preference), "preference");
        if (operation) cleanup.record(cublasLtMatmulDescDestroy(operation), "operation");
        for (auto layout : layouts)
            if (layout) cleanup.record(cublasLtMatrixLayoutDestroy(layout), "layout");
    }
};
std::size_t width(cudaDataType type) {
    return type == CUDA_R_16F ? 2U : 4U;
}
struct Element {
    std::size_t index{};
    float value{};
};
void put(std::span<std::byte> bytes, cudaDataType type, Element element) {
    if (type == CUDA_R_16F) {
        const auto bits = std::bit_cast<std::array<std::byte, 2>>(__float2half(element.value));
        std::ranges::copy(bits, bytes.begin() + static_cast<std::ptrdiff_t>(2 * element.index));
    } else {
        const auto bits = std::bit_cast<std::array<std::byte, 4>>(element.value);
        std::ranges::copy(bits, bytes.begin() + static_cast<std::ptrdiff_t>(4 * element.index));
    }
}
int pattern(std::size_t index) {
    return static_cast<int>(index % 9) - 4;
}
struct Buffer : private NativeOwner {
    Cleanup& cleanup;
    cudaDataType type;
    std::vector<std::byte> original;
    CUdeviceptr device{};
    Buffer(Cleanup& result, cudaDataType format, std::size_t elements)
        : cleanup(result), type(format), original(elements * width(format)) {
        check(cuMemAlloc(&device, original.size()));
    }
    ~Buffer() {
        if (device) cleanup.record(cuMemFree(device), "buffer");
    }
    void initialize() {
        for (std::size_t index = 0; index < original.size() / width(type); ++index)
            put(original, type, {index, static_cast<float>(pattern(index)) / 8.0F});
    }
    void upload() const {
        check(cuMemcpyHtoD(device, original.data(), original.size()));
    }
    std::vector<std::byte> download() const {
        std::vector<std::byte> bytes(original.size());
        check(cuMemcpyDtoH(bytes.data(), device, bytes.size()));
        return bytes;
    }
};
void save(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    const int descriptor = ::open(
        path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, S_IRUSR | S_IWUSR);
    if (descriptor < 0) throw std::runtime_error("exclusive diagnostic image open failed");
    std::size_t offset{};
    while (offset < bytes.size()) {
        const auto written = ::write(descriptor, bytes.data() + offset, bytes.size() - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            const auto closed = ::close(descriptor);
            throw std::runtime_error(closed ? "diagnostic image write and close failed"
                                            : "complete diagnostic image write failed");
        }
        offset += static_cast<std::size_t>(written);
    }
    if (::close(descriptor) != 0) throw std::runtime_error("diagnostic image close failed");
}
// All inputs are integers divided by8. Alpha2, beta0/0.5 and bias produce an
// integer divided by64; no reduction-order tolerance or SDK result is the oracle.
struct Coordinates {
    int row{}, column{};
};
float reference(Coordinates cell, bool bias, bool zero_beta) {
    int product{};
    const auto ri = static_cast<std::size_t>(cell.row), ci = static_cast<std::size_t>(cell.column);
    for (int inner = 0; inner < kDepth; ++inner)
        product += pattern(ri + static_cast<std::size_t>(inner) * kLeading[0]) *
                   pattern(static_cast<std::size_t>(inner) + ci * kLeading[1]);
    const int c = zero_beta ? 0 : 4 * pattern(ri + ci * kLeading[2]);
    const int b = bias ? 8 * pattern(static_cast<std::size_t>(cell.row)) : 0;
    return static_cast<float>(2 * product + c + b) / 64.0F;
}
struct Observation {
    unsigned int id{}, format{};
    bool device_scalars{}, bias{}, in_place{}, zero_beta{}, inputs_unchanged{};
    std::int32_t algorithm{};
    std::uint32_t pointer_mask{}, epilogue_mask{};
    std::size_t mismatches{}, guard_mismatches{}, output_bytes{};
};
std::vector<Observation> planned_cases() {
    // The historical failing operation goes first in a fresh handle/context,
    // without preceding negative-status or other numerical SDK calls.
    std::vector<Observation> result{{.format = 1, .device_scalars = true, .bias = true}};
    for (unsigned int format = 0; format < 3; ++format)
        for (bool device : {false, true})
            for (bool bias : {false, true})
                for (bool in_place : {false, true})
                    for (bool zero_beta : {false, true}) {
                        if (format == 1 && device && bias && !in_place && !zero_beta) continue;
                        result.push_back({.id = static_cast<unsigned int>(result.size()),
                                          .format = format,
                                          .device_scalars = device,
                                          .bias = bias,
                                          .in_place = in_place,
                                          .zero_beta = zero_beta});
                    }
    return result;
}
Observation execute(cublasLtHandle_t handle, CUstream stream, Cleanup& cleanup,
                    const std::filesystem::path& images, Observation result) {
    const auto input_type = result.format == 0 ? CUDA_R_32F : CUDA_R_16F;
    const auto output_type = result.format == 1 ? CUDA_R_16F : CUDA_R_32F;
    Buffer a(cleanup, input_type, kElements[0]), b(cleanup, input_type, kElements[1]),
        c(cleanup, output_type, kElements[2]), d(cleanup, output_type, kElements[3]),
        bias(cleanup, output_type, kElements[4]), scalars(cleanup, CUDA_R_32F, kElements[5]),
        workspace(cleanup, CUDA_R_32F, 262144);
    const std::array buffers{&a, &b, &c, &d, &bias, &scalars};
    for (auto* buffer : buffers) buffer->initialize();
    std::ranges::fill(scalars.original, std::byte{});
    constexpr float alpha = 2.0F;
    const float beta = result.zero_beta ? 0.0F : 0.5F;
    put(scalars.original, scalars.type, {0, alpha});
    put(scalars.original, scalars.type, {4, beta});
    for (const auto* buffer : buffers) buffer->upload();
    Descriptors descriptors(cleanup);
    check(cublasLtMatmulDescCreate(&descriptors.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    constexpr auto transpose = CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &transpose, sizeof(transpose)));
    check(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &transpose, sizeof(transpose)));
    const auto pointer_mode =
        result.device_scalars ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
    const auto epilogue = result.bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): exact original SDK GPU-address ABI.
    const auto* bias_pointer = result.bias ? reinterpret_cast<const void*>(bias.device) : nullptr;
    check(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                         &pointer_mode, sizeof(pointer_mode)));
    check(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                         &epilogue, sizeof(epilogue)));
    check(cublasLtMatmulDescSetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                         &bias_pointer, sizeof(bias_pointer)));
    check(cublasLtMatrixLayoutCreate(&descriptors.layouts[0], input_type, kRows, kDepth,
                                     kLeading[0]));
    check(cublasLtMatrixLayoutCreate(&descriptors.layouts[1], input_type, kDepth, kColumns,
                                     kLeading[1]));
    for (std::size_t index = 2; index < 4; ++index)
        check(cublasLtMatrixLayoutCreate(&descriptors.layouts[index], output_type, kRows, kColumns,
                                         kLeading[index]));
    check(cublasLtMatmulPreferenceCreate(&descriptors.preference));
    const auto workspace_bytes = workspace.original.size();
    check(cublasLtMatmulPreferenceSetAttribute(descriptors.preference,
                                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &workspace_bytes, sizeof(workspace_bytes)));
    const auto output_layout = descriptors.layouts[result.in_place ? 2 : 3];
    auto& output = result.in_place ? c : d;
    std::array<cublasLtMatmulHeuristicResult_t, 4> candidates{};
    int count{};
    check(cublasLtMatmulAlgoGetHeuristic(handle, descriptors.operation, descriptors.layouts[0],
                                         descriptors.layouts[1], descriptors.layouts[2],
                                         output_layout, descriptors.preference, 4,
                                         candidates.data(), &count));
    expect(count > 0 && count <= 4 && candidates[0].state == CUBLAS_STATUS_SUCCESS,
           "first actual heuristic must be valid");
    const auto& algorithm = candidates[0].algo;
    cublasLtMatmulHeuristicResult_t checked{};
    check(cublasLtMatmulAlgoCheck(handle, descriptors.operation, descriptors.layouts[0],
                                  descriptors.layouts[1], descriptors.layouts[2], output_layout,
                                  &algorithm, &checked));
    expect(checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize <= workspace_bytes,
           "original algorithm check must accept complete workspace");
    std::size_t written{};
    check(cublasLtMatmulAlgoConfigGetAttribute(&algorithm, CUBLASLT_ALGO_CONFIG_ID,
                                               &result.algorithm, sizeof(result.algorithm),
                                               &written));
    expect(written == sizeof(result.algorithm), "complete native algorithm ID");
    check(cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_POINTER_MODE_MASK,
                                            &result.pointer_mask, sizeof(result.pointer_mask),
                                            &written));
    expect(written == sizeof(result.pointer_mask), "complete native pointer capability mask");
    check(cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_EPILOGUE_MASK,
                                            &result.epilogue_mask, sizeof(result.epilogue_mask),
                                            &written));
    expect(written == sizeof(result.epilogue_mask), "complete native epilogue capability mask");
    cublasLtPointerMode_t observed{};
    check(cublasLtMatmulDescGetAttribute(descriptors.operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                         &observed, sizeof(observed), &written));
    expect(written == sizeof(observed) && observed == pointer_mode, "actual scalar mode preserved");
    check(cuCtxSynchronize());
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const auto& buffer = *buffers[index];
        expect(buffer.download() == buffer.original, "complete inputs initialized before matmul");
        save(
            images / (std::to_string(result.id) + "-" + std::string(kNames[index]) + "-before.bin"),
            buffer.original);
    }
    // NOLINTBEGIN(performance-no-int-to-ptr): original SDK consumes Driver GPU addresses.
    check(cublasLtMatmul(
        handle, descriptors.operation,
        result.device_scalars ? reinterpret_cast<const void*>(scalars.device) : &alpha,
        reinterpret_cast<const void*>(a.device), descriptors.layouts[0],
        reinterpret_cast<const void*>(b.device), descriptors.layouts[1],
        result.device_scalars ? reinterpret_cast<const void*>(scalars.device + 16) : &beta,
        reinterpret_cast<const void*>(c.device), descriptors.layouts[2],
        reinterpret_cast<void*>(output.device), output_layout, &algorithm,
        reinterpret_cast<void*>(workspace.device), workspace_bytes,
        reinterpret_cast<cudaStream_t>(stream)));
    // NOLINTEND(performance-no-int-to-ptr)
    check(cuStreamSynchronize(stream));
    auto expected = output.original;
    const auto leading = kLeading[result.in_place ? 2 : 3];
    for (int column = 0; column < kColumns; ++column)
        for (int row = 0; row < kRows; ++row)
            put(expected, output.type,
                {static_cast<std::size_t>(row) +
                     static_cast<std::size_t>(column) * static_cast<std::size_t>(leading),
                 reference({row, column}, result.bias, result.zero_beta)});
    save(images / (std::to_string(result.id) + "-expected.bin"), expected);
    result.inputs_unchanged = true;
    result.output_bytes = expected.size();
    for (std::size_t index = 0; index < buffers.size(); ++index) {
        const auto& buffer = *buffers[index];
        const auto actual = buffer.download();
        save(images / (std::to_string(result.id) + "-" + std::string(kNames[index]) + "-after.bin"),
             actual);
        if (&buffer != &output) {
            result.inputs_unchanged = result.inputs_unchanged && actual == buffer.original;
            continue;
        }
        for (std::size_t element = 0; element < actual.size() / width(output.type); ++element) {
            const auto offset = element * width(output.type);
            if (std::equal(
                    actual.begin() + static_cast<std::ptrdiff_t>(offset),
                    actual.begin() + static_cast<std::ptrdiff_t>(offset + width(output.type)),
                    expected.begin() + static_cast<std::ptrdiff_t>(offset)))
                continue;
            const bool is_value = element / static_cast<std::size_t>(leading) < kColumns &&
                                  element % static_cast<std::size_t>(leading) < kRows;
            if (is_value)
                ++result.mismatches;
            else
                ++result.guard_mismatches;
        }
    }
    return result;
}
void self_test() {
    const auto cases = planned_cases();
    expect(cases.size() == 48 && cases[0].format == 1 && cases[0].device_scalars && cases[0].bias &&
               !cases[0].in_place && !cases[0].zero_beta,
           "fixed complete matrix starts with the independently isolated historical case");
    expect(kLeading[0] * kDepth + 32 == kElements[0] && kLeading[1] * kColumns + 32 == kElements[1],
           "exact historical padded hulls");
    expect(reference({0, 0}, true, false) == 0.5F && reference({0, 0}, false, false) == 1.0F,
           "original first mismatch has the unchanged full mathematical oracle");
    for (int column = 0; column < kColumns; ++column)
        for (int row = 0; row < kRows; ++row)
            for (bool bias : {false, true})
                for (bool zero : {false, true}) {
                    const auto value = reference({row, column}, bias, zero);
                    expect(__half2float(__float2half(value)) == value,
                           "every fixed-point result is exactly FP16 representable");
                }
    std::cout << "Exact fixed-point Lt oracle verified without CUDA initialization\n";
}
void run(const std::filesystem::path& images) {
    expect(std::filesystem::create_directory(images), "fresh exclusive diagnostic image directory");
    Cleanup cleanup;
    std::vector<Observation> observations;
    std::array<int, 3> blas_version{};
    {
        Context context(cleanup);
        Handle handle(cleanup);
        Stream stream(cleanup);
        for (const auto& planned : planned_cases()) {
            observations.push_back(execute(handle.value, stream.value, cleanup, images, planned));
            expect(cleanup.is_clean, "each diagnostic case releases all exact owners");
        }
    }
    expect(cleanup.is_clean,
           "all descriptors, memory, streams, handles and context retire normally");
    check(cublasGetProperty(MAJOR_VERSION, &blas_version[0]));
    check(cublasGetProperty(MINOR_VERSION, &blas_version[1]));
    check(cublasGetProperty(PATCH_LEVEL, &blas_version[2]));
    const auto lt_version = cublasLtGetVersion();
    const bool passed = std::ranges::all_of(observations, [](const auto& value) {
        return !value.mismatches && !value.guard_mismatches && value.inputs_unchanged;
    });
    std::cout << "{\"status\":\"" << (passed ? "passed" : "numerical_failure")
              << "\",\"cleanup\":true,\"diagnostic_only\":true,\"lt_version\":" << lt_version
              << ",\"blas_version\":[" << blas_version[0] << ',' << blas_version[1] << ','
              << blas_version[2] << "],\"cases\":[";
    for (const auto& value : observations) {
        if (value.id) std::cout << ',';
        std::cout << "{\"id\":" << value.id << ",\"format\":" << value.format
                  << ",\"device_scalars\":" << value.device_scalars << ",\"bias\":" << value.bias
                  << ",\"in_place\":" << value.in_place << ",\"zero_beta\":" << value.zero_beta
                  << ",\"algorithm\":" << value.algorithm
                  << ",\"pointer_mask\":" << value.pointer_mask
                  << ",\"epilogue_mask\":" << value.epilogue_mask
                  << ",\"mismatches\":" << value.mismatches
                  << ",\"guard_mismatches\":" << value.guard_mismatches
                  << ",\"inputs_unchanged\":" << value.inputs_unchanged
                  << ",\"output_bytes\":" << value.output_bytes << '}';
    }
    std::cout << "],\"full_goal_complete\":false,\"full_device_qualified\":false,"
                 "\"physical_priority_qualified\":false,\"performance_qualified\":false}\n";
    if (!passed)
        throw std::runtime_error("native Lt full-value oracle failed after complete cleanup");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        std::cout << std::boolalpha;
        if (argc == 2 && std::string_view(argv[1]) == "--self-test")
            self_test();
        else if (argc == 2 && std::string_view(argv[1]) == "--help")
            std::cout
                << "--self-test | --output-dir PATH (native diagnostic, no parity qualification)\n";
        else {
            expect(argc == 3 && std::string_view(argv[1]) == "--output-dir",
                   "explicit output directory required");
            run(argv[2]);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
