#include <cuda.h>
#include <cublas_v2.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <dlfcn.h>

// Ordinary SDK input: the same executable runs natively and with ownership
// interception. This bounded numerical diagnostic makes no SM/timing claim.
namespace {
constexpr std::size_t kGuard = 16, kPayload = 64, kMatrixElements = 96, kScalarElements = 34;
constexpr std::size_t kCases = 84, kValuesPerCase = 356;
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
struct Native {
    CUcontext context{};
    CUstream stream{};
    cublasHandle_t handle{};
    std::vector<CUdeviceptr> allocations;
    std::string gpu_uuid;
    bool uncertain{};
    void owned(CUresult status) {
        if (status != CUDA_SUCCESS) {
            uncertain = true;
            checked(status);
        }
    }
    void owned(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            uncertain = true;
            checked(status);
        }
    }
    void proof() {
        CUcontext current{};
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != context) {
            uncertain = true;
            throw std::runtime_error("current owner unproven before synchronization");
        }
        const auto stopped = cuCtxSynchronize();
        if (stopped != CUDA_SUCCESS) {
            uncertain = true;
            throw std::runtime_error("physical completion unproven; no later SDK work or cleanup");
        }
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != context) {
            uncertain = true;
            throw std::runtime_error(
                "physical completion/current owner unproven; no later work or cleanup");
        }
    }
    void start() {
        allocations.reserve(16);  // All bookkeeping allocation precedes native resource creation.
        checked(cuInit(0));
        CUdevice device{};
        checked(cuDeviceGet(&device, 0));
        const auto created = cuCtxCreate(&context, 0, device);
        owned(created);
        if (!context) {
            uncertain = true;
            throw std::runtime_error("successful creation returned NULL context");
        }
        CUdevice actual_device{};
        CUuuid uuid{};
        owned(cuCtxGetDevice(&actual_device));
        owned(cuDeviceGetUuid(&uuid, actual_device));
        gpu_uuid = uuid_text(uuid);
        owned(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        if (!stream) {
            uncertain = true;
            throw std::runtime_error("successful creation returned NULL stream");
        }
        owned(cublasCreate(&handle));
        if (!handle) {
            uncertain = true;
            throw std::runtime_error("successful creation returned NULL handle");
        }
        owned(cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
        int version{};
        owned(cublasGetVersion(handle, &version));
        expect(version == 120604,
               "this exact scalar/no-read diagnostic requires original SDK 12.6.4");
    }
    CUdeviceptr allocate(std::size_t bytes) {
        expect(allocations.size() < 16, "fixed native allocation inventory");
        CUdeviceptr pointer{};
        const auto status = cuMemAlloc(&pointer, bytes);
        if (status != CUDA_SUCCESS) {
            uncertain = true;
            checked(status);
        }
        if (!pointer) {
            uncertain = true;
            throw std::runtime_error("successful allocation returned NULL");
        }
        allocations.push_back(pointer);
        return pointer;
    }
    void upload(CUdeviceptr pointer, const void* values, std::size_t bytes) {
        const auto status = cuMemcpyHtoD(pointer, values, bytes);
        if (status != CUDA_SUCCESS) {
            uncertain = true;
            checked(status);
        }
    }
    void download(void* values, CUdeviceptr pointer, std::size_t bytes) {
        const auto status = cuMemcpyDtoH(values, pointer, bytes);
        if (status != CUDA_SUCCESS) {
            uncertain = true;
            checked(status);
        }
    }
    void close() {
        if (uncertain || !context) return;
        proof();
        if (handle) {
            const auto status = cublasDestroy(handle);
            if (status != CUBLAS_STATUS_SUCCESS) {
                uncertain = true;
                checked(status);
            }
            handle = nullptr;
        }
        for (const auto allocation : allocations) {
            const auto status = cuMemFree(allocation);
            if (status != CUDA_SUCCESS) {
                uncertain = true;
                checked(status);
            }
        }
        allocations.clear();
        if (stream) {
            const auto status = cuStreamDestroy(stream);
            if (status != CUDA_SUCCESS) {
                uncertain = true;
                checked(status);
            }
            stream = nullptr;
        }
        const auto status = cuCtxDestroy(context);
        if (status != CUDA_SUCCESS) {
            uncertain = true;
            checked(status);
        }
        context = nullptr;
    }
};
struct Functions {
    decltype(&cublasSgeam) single{};
    decltype(&cublasDgeam) wide{};
    explicit Functions(void* provider) {
        single = reinterpret_cast<decltype(single)>(::dlsym(provider, "cublasSgeam"));
        wide = reinterpret_cast<decltype(wide)>(::dlsym(provider, "cublasDgeam"));
        expect(single && wide && single == &cublasSgeam && wide == &cublasDgeam,
               "qualified and direct original/owned GEAM lookups agree");
    }
};
struct Case {
    const char* name{};
    cublasOperation_t ta{CUBLAS_OP_N}, tb{CUBLAS_OP_N};
    int rows{3}, columns{4}, lda{6}, ldb{7}, ldc{5};
    std::size_t a{}, b{1}, c{2};
    double alpha{0.5}, beta{-0.25};
    bool null_a{}, null_b{}, null_alpha{}, null_beta{}, null_c{};
    cublasStatus_t expected{CUBLAS_STATUS_SUCCESS};
};
std::vector<Case> inventory(bool device) {
    std::vector<Case> cases;
    constexpr std::array names{"NN", "NT", "NC", "TN", "TT", "TC", "CN", "CT", "CC"};
    for (unsigned int ta = 0; ta != 3; ++ta)
        for (unsigned int tb = 0; tb != 3; ++tb)
            cases.push_back({.name = names[ta * 3 + tb],
                             .ta = static_cast<cublasOperation_t>(ta),
                             .tb = static_cast<cublasOperation_t>(tb)});
    cases.push_back({.name = "ort_transpose_a_equals_b",
                     .ta = CUBLAS_OP_T,
                     .tb = CUBLAS_OP_T,
                     .lda = 4,
                     .ldb = 4,
                     .ldc = 3,
                     .b = 0,
                     .alpha = 1,
                     .beta = 0});
    cases.push_back({.name = "output_alias_a", .ldc = 6, .c = 0});
    cases.push_back({.name = "output_alias_b", .ldc = 7, .c = 1});
    cases.push_back({.name = "input_alias", .ldb = 6, .b = 0});
    cases.push_back({.name = "zero_alpha_null_a", .alpha = -0.0, .null_a = true});
    cases.push_back({.name = "zero_beta_null_b", .beta = 0, .null_b = true});
    cases.push_back({.name = "invalid_a_transpose",
                     .ta = CUBLAS_OP_T,
                     .lda = 5,
                     .c = 0,
                     .expected = CUBLAS_STATUS_INVALID_VALUE});
    cases.push_back({.name = "invalid_a_leading", .c = 0, .expected = CUBLAS_STATUS_INVALID_VALUE});
    cases.push_back({.name = "invalid_b_transpose",
                     .tb = CUBLAS_OP_T,
                     .ldb = 5,
                     .c = 1,
                     .expected = CUBLAS_STATUS_INVALID_VALUE});
    cases.push_back({.name = "invalid_b_leading", .c = 1, .expected = CUBLAS_STATUS_INVALID_VALUE});
    cases.push_back({.name = "null_alpha",
                     .null_alpha = true,
                     .expected = device ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS});
    cases.push_back({.name = "zero_rows_null_all",
                     .rows = 0,
                     .null_a = true,
                     .null_b = true,
                     .null_alpha = true,
                     .null_beta = true,
                     .null_c = true});
    expect(cases.size() == 21, "fixed independently enumerated GEAM inventory");
    return cases;
}

void checksum(std::uint64_t& hash, const void* data, std::size_t bytes) {
    const auto* input = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index != bytes; ++index) {
        hash ^= input[index];
        hash *= UINT64_C(1099511628211);
    }
}
struct Totals {
    std::size_t sequence{};
    std::uint64_t checksum{UINT64_C(14695981039346656037)};
};

// NOLINTBEGIN(performance-no-int-to-ptr): Driver/BLAS distinct device-address ABIs.
template <typename T>
void group(Native& native, const Functions& functions, bool device, Totals& totals) {
    constexpr bool wide = std::is_same_v<T, double>;
    const auto type = wide ? "double" : "float";
    const auto mode = device ? "device" : "host";
    std::array<CUdeviceptr, 3> storage{};
    for (auto& pointer : storage) pointer = native.allocate(kMatrixElements * sizeof(T));
    const auto scalar_storage = native.allocate(kScalarElements * sizeof(T));
    native.owned(cublasSetPointerMode(
        native.handle, device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST));
    for (const auto& test : inventory(device)) {
        std::array<std::array<T, kMatrixElements>, 3> before{};
        for (std::size_t matrix = 0; matrix != before.size(); ++matrix) {
            before[matrix].fill(T{-8192});
            for (std::size_t element = 0; element != kPayload; ++element)
                before[matrix][kGuard + element] =
                    static_cast<T>(static_cast<int>(matrix * 101 + element) - 31);
            native.upload(storage[matrix], before[matrix].data(), sizeof(before[matrix]));
        }
        std::array<T, kScalarElements> host_scalars{};
        host_scalars.fill(T{-4096});
        host_scalars[kGuard] = static_cast<T>(test.alpha);
        host_scalars[kGuard + 1] = static_cast<T>(test.beta);
        const auto saved_host = host_scalars;
        native.upload(scalar_storage, host_scalars.data(), sizeof(host_scalars));
        native.proof();  // Pageable uploads must finish before the nonblocking BLAS stream.
        const auto* alpha = test.null_alpha ? nullptr
                            : device
                                ? reinterpret_cast<const T*>(scalar_storage + kGuard * sizeof(T))
                                : host_scalars.data() + kGuard;
        const auto* beta =
            test.null_beta ? nullptr
            : device       ? reinterpret_cast<const T*>(scalar_storage + (kGuard + 1) * sizeof(T))
                           : host_scalars.data() + kGuard + 1;
        const auto* a = test.null_a
                            ? nullptr
                            : reinterpret_cast<const T*>(storage[test.a] + kGuard * sizeof(T));
        const auto* b = test.null_b
                            ? nullptr
                            : reinterpret_cast<const T*>(storage[test.b] + kGuard * sizeof(T));
        auto* c =
            test.null_c ? nullptr : reinterpret_cast<T*>(storage[test.c] + kGuard * sizeof(T));
        const auto sequence = ++totals.sequence;
        std::cout << "{\"record\":\"begin\",\"sequence\":" << sequence << ",\"type\":\"" << type
                  << "\",\"mode\":\"" << mode << "\",\"name\":\"" << test.name << "\"}\n"
                  << std::flush;
        cublasStatus_t status{};
        if constexpr (wide) {
            status = sequence % 3 == 0
                         ? cublasDgeam(native.handle, test.ta, test.tb, test.rows, test.columns,
                                       alpha, a, test.lda, beta, b, test.ldb, c, test.ldc)
                         : functions.wide(native.handle, test.ta, test.tb, test.rows, test.columns,
                                          alpha, a, test.lda, beta, b, test.ldb, c, test.ldc);
        } else {
            status =
                sequence % 3 == 0
                    ? cublasSgeam(native.handle, test.ta, test.tb, test.rows, test.columns, alpha,
                                  a, test.lda, beta, b, test.ldb, c, test.ldc)
                    : functions.single(native.handle, test.ta, test.tb, test.rows, test.columns,
                                       alpha, a, test.lda, beta, b, test.ldb, c, test.ldc);
        }
        native.proof();  // No reads, later cases or native cleanup after a failed proof.
        expect(status == test.expected,
               "actual original status differs from fixed native boundary");
        auto expected = before;
        if (status == CUBLAS_STATUS_SUCCESS && test.rows > 0) {
            const T av = test.null_alpha ? T{} : static_cast<T>(test.alpha);
            const T bv = test.null_beta ? T{} : static_cast<T>(test.beta);
            for (int row = 0; row != test.rows; ++row)
                for (int column = 0; column != test.columns; ++column) {
                    const auto ai =
                        test.ta == CUBLAS_OP_N ? column * test.lda + row : row * test.lda + column;
                    const auto bi =
                        test.tb == CUBLAS_OP_N ? column * test.ldb + row : row * test.ldb + column;
                    const T x = av == T{} ? T{} : before[test.a][kGuard + ai];
                    const T y = bv == T{} ? T{} : before[test.b][kGuard + bi];
                    expected[test.c][kGuard + static_cast<std::size_t>(column * test.ldc + row)] =
                        av * x + bv * y;
                }
        }
        std::uint64_t hash = UINT64_C(14695981039346656037);
        for (std::size_t matrix = 0; matrix != before.size(); ++matrix) {
            std::array<T, kMatrixElements> actual{};
            native.download(actual.data(), storage[matrix], sizeof(actual));
            using Bytes = std::array<std::byte, sizeof(actual)>;
            expect(std::bit_cast<Bytes>(actual) == std::bit_cast<Bytes>(expected[matrix]),
                   "complete exact matrix/input/padding/guard oracle mismatch");
            checksum(hash, actual.data(), sizeof(actual));
        }
        std::array<T, kScalarElements> actual_scalars{};
        native.download(actual_scalars.data(), scalar_storage, sizeof(actual_scalars));
        using ScalarBytes = std::array<std::byte, sizeof(saved_host)>;
        expect(
            std::bit_cast<ScalarBytes>(actual_scalars) == std::bit_cast<ScalarBytes>(saved_host) &&
                std::bit_cast<ScalarBytes>(host_scalars) == std::bit_cast<ScalarBytes>(saved_host),
            "all host/device scalar values and both guards remain bitwise unchanged");
        checksum(hash, actual_scalars.data(), sizeof(actual_scalars));
        checksum(hash, host_scalars.data(), sizeof(host_scalars));
        cublasPointerMode_t observed{};
        native.owned(cublasGetPointerMode(native.handle, &observed));
        expect(observed == (device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST),
               "temporary ownership staging never leaks a changed public pointer mode");
        cudaStream_t observed_stream{};
        native.owned(cublasGetStream(native.handle, &observed_stream));
        expect(observed_stream == reinterpret_cast<cudaStream_t>(native.stream),
               "the public handle retains its exact original stream binding");
        checksum(totals.checksum, &hash, sizeof(hash));
        std::cout << "{\"record\":\"case\",\"sequence\":" << sequence << ",\"type\":\"" << type
                  << "\",\"mode\":\"" << mode << "\",\"name\":\"" << test.name
                  << "\",\"status\":" << static_cast<int>(status)
                  << ",\"expected_status\":" << static_cast<int>(test.expected)
                  << ",\"checked_values\":" << kValuesPerCase << ",\"checksum\":" << hash
                  << ",\"all_values_verified\":true,"
                  << "\"guards_verified\":true,\"scalars_unchanged\":true}\n"
                  << std::flush;
    }
}
// NOLINTEND(performance-no-int-to-ptr)
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: execution_geam_gpu_test ORIGINAL_LIBCUBLAS\n";
        return 1;
    }
    Native native;
    void* provider{};
    try {
        provider = ::dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        expect(provider != nullptr, "open the exact caller SDK export provider");
        const Functions functions(provider);
        native.start();
        Totals totals;
        group<float>(native, functions, false, totals);
        group<float>(native, functions, true, totals);
        group<double>(native, functions, false, totals);
        group<double>(native, functions, true, totals);
        expect(totals.sequence == kCases, "complete S/D × HOST/DEVICE numerical inventory");
        native.close();
        expect(::dlclose(provider) == 0, "release explicit SDK lookup provider");
        provider = nullptr;
        std::cout << "{\"record\":\"summary\",\"api_calls\":" << totals.sequence
                  << ",\"checked_values\":" << totals.sequence * kValuesPerCase
                  << ",\"checksum\":" << totals.checksum << ",\"gpu_uuid\":\"" << native.gpu_uuid
                  << "\""
                  << ",\"numerical_diagnostic_passed\":true,\"full_device_qualified\":false,"
                  << "\"priority_qualified\":false,\"formal_comparison_qualified\":false}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        if (!native.uncertain) {
            try {
                native.close();
            } catch (const std::exception& cleanup) {
                std::cerr << cleanup.what() << '\n';
            }
        }
        if (provider && !native.uncertain) static_cast<void>(::dlclose(provider));
        return 1;
    }
}
