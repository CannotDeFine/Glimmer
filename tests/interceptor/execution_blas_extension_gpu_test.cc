#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>

// Unchanged ordinary SDK input for native and owned execution. These small,
// exact numerical diagnostics do not qualify SM participation or performance.
namespace {
constexpr std::size_t kGuard = 16, kPayload = 256, kElements = 288;
constexpr std::size_t kStorageBytes = kElements * sizeof(double), kScalarElements = 34;
constexpr std::uint64_t kHashSeed = UINT64_C(14695981039346656037);
using Storage = std::array<std::byte, kStorageBytes>;
using Scalars = std::array<float, kScalarElements>;
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
void checksum(std::uint64_t& hash, const void* data, std::size_t bytes) {
    const auto* input = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index != bytes; ++index) {
        hash ^= input[index];
        hash *= UINT64_C(1099511628211);
    }
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
constexpr const char* kProducer = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry blas_extension_coefficients(
 .param .u64 pointer, .param .f32 alpha, .param .f32 beta) {
 .reg .pred %p; .reg .b32 %r<5>; .reg .b64 %d<3>; .reg .f32 %f<3>;
 mov.u32 %r1,%ctaid.x; mov.u32 %r2,%ntid.x; mov.u32 %r3,%tid.x;
 mad.lo.u32 %r4,%r1,%r2,%r3; setp.ne.u32 %p,%r4,0; @%p bra done;
 ld.param.u64 %d1,[pointer]; ld.param.f32 %f1,[alpha]; ld.param.f32 %f2,[beta];
 add.u64 %d2,%d1,4; st.global.f32 [%d1],%f1; st.global.f32 [%d2],%f2;
done: ret;
}
)ptx";
struct Native {
    CUcontext context{};
    CUstream stream{};
    CUmodule module{};
    CUfunction producer{};
    cublasHandle_t handle{};
    std::array<CUdeviceptr, 3> matrices{};
    CUdeviceptr scalars{};
    std::string gpu_uuid;
    int driver_version{}, cublas_version{};
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
            throw std::runtime_error("current ownership unproven before physical completion");
        }
        owned(cuCtxSynchronize());
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != context) {
            uncertain = true;
            throw std::runtime_error("current ownership unproven after physical completion");
        }
    }
    void start() {
        checked(cuInit(0));
        checked(cuDriverGetVersion(&driver_version));
        CUdevice device{};
        checked(cuDeviceGet(&device, 0));
        owned(cuCtxCreate(&context, 0, device));
        if (!context) {
            uncertain = true;
            throw std::runtime_error("NULL successful context");
        }
        CUdevice actual{};
        CUuuid uuid{};
        owned(cuCtxGetDevice(&actual));
        owned(cuDeviceGetUuid(&uuid, actual));
        gpu_uuid = uuid_text(uuid);
        owned(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        owned(cublasCreate(&handle));
        if (!stream || !handle) {
            uncertain = true;
            throw std::runtime_error("NULL successful owner");
        }
        owned(cublasSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
        owned(cublasGetVersion(handle, &cublas_version));
        expect(cublas_version == 120604, "native boundary inventory requires cuBLAS 12.6.4");
        owned(cuModuleLoadData(&module, kProducer));
        owned(cuModuleGetFunction(&producer, module, "blas_extension_coefficients"));
        if (!module || !producer) {
            uncertain = true;
            throw std::runtime_error("NULL successful module/function");
        }
        for (auto& pointer : matrices) {
            owned(cuMemAlloc(&pointer, kStorageBytes));
            if (!pointer) {
                uncertain = true;
                throw std::runtime_error("NULL successful matrix allocation");
            }
        }
        owned(cuMemAlloc(&scalars, sizeof(Scalars)));
        if (!scalars) {
            uncertain = true;
            throw std::runtime_error("NULL successful scalar allocation");
        }
    }
    void close() {
        if (uncertain || !context) return;
        proof();
        if (handle) {
            owned(cublasDestroy(handle));
            handle = nullptr;
        }
        for (auto& pointer : matrices)
            if (pointer) {
                owned(cuMemFree(pointer));
                pointer = 0;
            }
        if (scalars) {
            owned(cuMemFree(scalars));
            scalars = 0;
        }
        if (module) {
            owned(cuModuleUnload(module));
            module = nullptr;
        }
        if (stream) {
            owned(cuStreamDestroy(stream));
            stream = nullptr;
        }
        owned(cuCtxDestroy(context));
        context = nullptr;
    }
    ~Native() {
        if (!uncertain && context) {
            try {
                close();
            } catch (...) {
                uncertain = true;
            }
        }
    }
};
struct Functions {
    decltype(&cublasSgemv) gemv{};
    decltype(&cublasSgemmEx) gemm{};
    explicit Functions(void* provider) {
        gemv = reinterpret_cast<decltype(gemv)>(::dlsym(provider, "cublasSgemv_v2"));
        gemm = reinterpret_cast<decltype(gemm)>(::dlsym(provider, "cublasSgemmEx"));
        expect(gemv && gemm && gemv == &cublasSgemv && gemm == &cublasSgemmEx,
               "exact direct/cached SDK export identity");
    }
};
std::size_t width(cudaDataType type) {
    switch (type) {
        case CUDA_R_8I:
            return 1;
        case CUDA_R_16F:
        case CUDA_R_16BF:
            return 2;
        case CUDA_R_32F:
            return 4;
        case CUDA_R_64F:
            return 8;
        default:
            throw std::runtime_error("unsupported fixture storage type");
    }
}
void encode(std::byte* output, cudaDataType type, float value) {
    if (type == CUDA_R_8I) {
        const auto integer = static_cast<std::int8_t>(value);
        std::memcpy(output, &integer, 1);
    } else if (type == CUDA_R_16F) {
        const auto half = __float2half(value);
        std::memcpy(output, &half, 2);
    } else if (type == CUDA_R_16BF) {
        // Fixture operands/results are exactly representable dyadic numbers.
        const auto bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
        std::memcpy(output, &bits, 2);
    } else if (type == CUDA_R_32F)
        std::memcpy(output, &value, 4);
    else if (type == CUDA_R_64F) {
        const auto wide = static_cast<double>(value);
        std::memcpy(output, &wide, 8);
    } else
        throw std::runtime_error("unexpected fixture encoding type");
}
float decode(const std::byte* input, cudaDataType type) {
    if (type == CUDA_R_8I) {
        std::int8_t value{};
        std::memcpy(&value, input, 1);
        return value;
    }
    if (type == CUDA_R_16F) {
        __half value;
        std::memcpy(&value, input, 2);
        return __half2float(value);
    }
    if (type == CUDA_R_16BF) {
        std::uint16_t value{};
        std::memcpy(&value, input, 2);
        return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
    }
    if (type == CUDA_R_32F) {
        float value{};
        std::memcpy(&value, input, 4);
        return value;
    }
    double value{};
    std::memcpy(&value, input, 8);
    return static_cast<float>(value);
}
enum class Expected {
    kSuccess,
    kInvalid,
    kUnsupported,
    kArchitectureDependent,
    kBoundary,
    kNegativeVectorBoundary
};
struct Case {
    std::string name;
    bool vector{};
    cublasOperation_t ta{CUBLAS_OP_N}, tb{CUBLAS_OP_N};
    cudaDataType a{CUDA_R_32F}, b{CUDA_R_32F}, c{CUDA_R_32F};
    int rows{8}, columns{12}, depth{4}, lda{16}, ldb{16}, ldc{16}, incx{1}, incy{1};
    float alpha{2}, beta{0.5};
    bool null_data{}, poison_inputs{}, poison_output{}, null_inputs{};
    Expected expected{Expected::kSuccess};
};
std::vector<Case> core_inventory() {
    std::vector<Case> cases;
    constexpr std::array<std::array<int, 2>, 5> strides{
        {{1, 1}, {2, 3}, {-1, 2}, {2, -2}, {-2, -3}}};
    for (const auto op : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
        for (std::size_t stride = 0; stride != strides.size(); ++stride) {
            Case value{.name = "sgemv_" + std::to_string(op) + "_" + std::to_string(stride),
                       .vector = true};
            value.ta = op;
            value.rows = 5;
            value.columns = 7;
            value.lda = 8;
            value.incx = strides[stride][0];
            value.incy = strides[stride][1];
            cases.push_back(value);
        }
    constexpr std::array<std::array<cudaDataType, 2>, 6> profiles{{{CUDA_R_32F, CUDA_R_32F},
                                                                   {CUDA_R_16F, CUDA_R_16F},
                                                                   {CUDA_R_16F, CUDA_R_32F},
                                                                   {CUDA_R_16BF, CUDA_R_16BF},
                                                                   {CUDA_R_16BF, CUDA_R_32F},
                                                                   {CUDA_R_8I, CUDA_R_32F}}};
    for (std::size_t profile = 0; profile != profiles.size(); ++profile)
        for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
            for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
                Case value{.name = "sgemm_ex_" + std::to_string(profile) + "_" +
                                   std::to_string(ta) + std::to_string(tb)};
                value.ta = ta;
                value.tb = tb;
                value.a = value.b = profiles[profile][0];
                value.c = profiles[profile][1];
                if (profile >= 3) value.expected = Expected::kArchitectureDependent;
                cases.push_back(value);
            }
    expect(cases.size() == 69, "fixed 15 vector and 54 typed matrix descriptors");
    return cases;
}
std::vector<Case> audit_inventory() {
    std::vector<Case> cases;
    for (const bool vector : {true, false}) {
        Case base{.name = "", .vector = vector};
        const auto add = [&](std::string name, auto change) {
            auto value = base;
            value.name = std::move(name);
            change(value);
            cases.push_back(value);
        };
        add("zero_rows", [](Case& c) { c.rows = 0; });
        add("zero_columns", [](Case& c) { c.columns = 0; });
        add("zero_rows_null_data", [](Case& c) {
            c.rows = 0;
            c.null_data = true;
            c.expected = Expected::kBoundary;
        });
        add("zero_columns_null_data", [](Case& c) {
            c.columns = 0;
            c.null_data = true;
            c.expected = Expected::kBoundary;
        });
        add("negative_rows", [](Case& c) {
            c.rows = -1;
            c.expected = c.vector ? Expected::kNegativeVectorBoundary : Expected::kInvalid;
        });
        add("negative_columns", [](Case& c) {
            c.columns = -1;
            c.expected = c.vector ? Expected::kNegativeVectorBoundary : Expected::kInvalid;
        });
        add("bad_lda", [](Case& c) {
            c.lda = 1;
            c.expected = Expected::kInvalid;
        });
        add("invalid_transpose_a", [](Case& c) {
            c.ta = static_cast<cublasOperation_t>(3);
            c.expected = Expected::kInvalid;
        });
        add("alpha_zero_beta_one", [](Case& c) {
            c.alpha = 0;
            c.beta = 1;
            c.poison_inputs = true;
        });
        add("beta_zero", [](Case& c) {
            c.beta = 0;
            c.poison_output = true;
        });
        if (vector) {
            add("zero_increment_x", [](Case& c) {
                c.incx = 0;
                c.expected = Expected::kInvalid;
            });
            add("zero_increment_y", [](Case& c) {
                c.incy = 0;
                c.expected = Expected::kInvalid;
            });
        } else {
            add("zero_depth", [](Case& c) { c.depth = 0; });
            add("negative_depth", [](Case& c) {
                c.depth = -1;
                c.expected = Expected::kInvalid;
            });
            add("bad_ldb", [](Case& c) {
                c.ldb = 1;
                c.expected = Expected::kInvalid;
            });
            add("bad_ldc", [](Case& c) {
                c.ldc = 1;
                c.expected = Expected::kInvalid;
            });
            add("invalid_transpose_b", [](Case& c) {
                c.tb = static_cast<cublasOperation_t>(3);
                c.expected = Expected::kInvalid;
            });
            add("unsupported_f64", [](Case& c) {
                c.a = c.b = c.c = CUDA_R_64F;
                c.expected = Expected::kUnsupported;
            });
            add("mismatched_types", [](Case& c) {
                c.a = CUDA_R_16F;
                c.expected = Expected::kUnsupported;
            });
        }
    }
    expect(cases.size() == 29, "fixed 12 vector and 17 matrix audit descriptors");
    return cases;
}
std::vector<Case> boundary_inventory() {
    // This independently fixed subset follows the completed isolated native
    // access probes. Dangerous SGEMV metadata and conditional-value behavior
    // remain in the separate audit inventory, never this matched input.
    std::vector<Case> cases;
    for (const bool vector : {true, false}) {
        Case base{.name = "", .vector = vector};
        base.rows = vector ? 5 : 3;
        base.columns = vector ? 7 : 4;
        base.depth = vector ? 0 : 5;
        base.lda = 8;
        base.ldb = base.ldc = vector ? 0 : 8;
        base.incx = base.incy = vector ? 1 : 0;
        const auto add = [&](std::string name, auto change) {
            auto value = base;
            value.name = std::move(name);
            change(value);
            cases.push_back(value);
            value.name += "_null_data";
            value.null_data = true;
            cases.push_back(value);
        };
        add("zero_rows", [](Case& value) { value.rows = 0; });
        add("zero_columns", [](Case& value) { value.columns = 0; });
        add("negative_rows", [](Case& value) {
            value.rows = -1;
            if (!value.vector) value.expected = Expected::kInvalid;
        });
        add("negative_columns", [](Case& value) {
            value.columns = -1;
            if (!value.vector) value.expected = Expected::kInvalid;
        });
        if (vector) continue;
        add("negative_depth", [](Case& value) {
            value.depth = -1;
            value.expected = Expected::kInvalid;
        });
        add("invalid_transpose_a", [](Case& value) {
            value.ta = static_cast<cublasOperation_t>(3);
            value.expected = Expected::kInvalid;
        });
        add("invalid_transpose_b", [](Case& value) {
            value.tb = static_cast<cublasOperation_t>(3);
            value.expected = Expected::kInvalid;
        });
        add("bad_lda", [](Case& value) {
            value.lda = 1;
            value.expected = Expected::kInvalid;
        });
        add("bad_ldb", [](Case& value) {
            value.ldb = 1;
            value.expected = Expected::kInvalid;
        });
        add("bad_ldc", [](Case& value) {
            value.ldc = 1;
            value.expected = Expected::kInvalid;
        });
        auto zero_depth = base;
        zero_depth.name = "zero_depth";
        zero_depth.depth = 0;
        cases.push_back(zero_depth);
        zero_depth.name = "zero_depth_null_inputs";
        zero_depth.null_inputs = true;
        cases.push_back(zero_depth);
    }
    expect(cases.size() == 30, "fixed eight vector and twenty-two matrix boundary descriptors");
    return cases;
}
void verify_status(const Case& test, cublasStatus_t status) {
    bool is_expected{};
    switch (test.expected) {
        case Expected::kSuccess:
            is_expected = status == CUBLAS_STATUS_SUCCESS;
            break;
        case Expected::kInvalid:
            is_expected = status == CUBLAS_STATUS_INVALID_VALUE;
            break;
        case Expected::kNegativeVectorBoundary:
            is_expected =
                test.vector && (test.rows < 0 || test.columns < 0) &&
                (status == CUBLAS_STATUS_SUCCESS || status == CUBLAS_STATUS_INVALID_VALUE);
            break;
        case Expected::kUnsupported:
            is_expected =
                status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_INVALID_VALUE;
            break;
        case Expected::kArchitectureDependent:
        case Expected::kBoundary:
            is_expected =
                status == CUBLAS_STATUS_SUCCESS || status == CUBLAS_STATUS_INVALID_VALUE ||
                status == CUBLAS_STATUS_ARCH_MISMATCH || status == CUBLAS_STATUS_NOT_SUPPORTED;
            break;
    }
    if (!is_expected)
        throw std::runtime_error(std::string("Unexpected original BLAS status: api=") +
                                 (test.vector ? "sgemv" : "sgemm_ex") + " case=" + test.name +
                                 " status=" + std::to_string(static_cast<int>(status)) +
                                 " expectation=" + std::to_string(static_cast<int>(test.expected)));
}
float get(const Storage& data, cudaDataType type, std::size_t index) {
    expect(index < kPayload, "bounded independent mathematical input index");
    return decode(data.data() + (kGuard + index) * width(type), type);
}
void put(Storage& data, cudaDataType type, std::size_t index, float value) {
    expect(index < kPayload, "bounded independent mathematical output index");
    encode(data.data() + (kGuard + index) * width(type), type, value);
}
std::array<Storage, 3> initial(const Case& test) {
    std::array<Storage, 3> result;
    const std::array types{test.a, test.b, test.c};
    for (std::size_t matrix = 0; matrix != result.size(); ++matrix) {
        result[matrix].fill(std::byte{0xa5});
        for (std::size_t index = 0; index != kPayload; ++index) {
            const float value =
                matrix == 2 ? 6.0F
                            : static_cast<float>(
                                  static_cast<int>((index * (matrix + 1) + 3 * matrix) % 7) - 3);
            put(result[matrix], types[matrix], index, value);
        }
        if ((matrix < 2 && test.poison_inputs) || (matrix == 2 && test.poison_output))
            for (std::size_t index = 0; index != kPayload; ++index)
                put(result[matrix], types[matrix], index,
                    std::bit_cast<float>(UINT32_C(0x7fc00001)));
    }
    return result;
}
std::array<Storage, 3> oracle(const Case& test, const std::array<Storage, 3>& before,
                              cublasStatus_t status) {
    auto expected = before;
    if (status != CUBLAS_STATUS_SUCCESS || test.rows == 0 || test.columns == 0) return expected;
    // Observe the native negative-dimension status without granting no-read
    // permission. Both permitted statuses must preserve every backing byte.
    if (test.vector && (test.rows < 0 || test.columns < 0)) return expected;
    if (test.vector) {
        const int x_length = test.ta == CUBLAS_OP_N ? test.columns : test.rows;
        const int y_length = test.ta == CUBLAS_OP_N ? test.rows : test.columns;
        const auto x_start = test.incx < 0 ? (x_length - 1) * -test.incx : 0;
        const auto y_start = test.incy < 0 ? (y_length - 1) * -test.incy : 0;
        for (int y = 0; y != y_length; ++y) {
            float sum{};
            if (test.alpha != 0)
                for (int x = 0; x != x_length; ++x) {
                    const auto matrix_index =
                        test.ta == CUBLAS_OP_N ? x * test.lda + y : y * test.lda + x;
                    sum += get(before[0], test.a, static_cast<std::size_t>(matrix_index)) *
                           get(before[1], test.b,
                               static_cast<std::size_t>(x_start +
                                                        static_cast<std::int64_t>(x) * test.incx));
                }
            const auto output =
                static_cast<std::size_t>(y_start + static_cast<std::int64_t>(y) * test.incy);
            const auto old = test.beta == 0 ? 0.0F : get(before[2], test.c, output);
            put(expected[2], test.c, output, test.alpha * sum + test.beta * old);
        }
    } else {
        for (int column = 0; column != test.columns; ++column)
            for (int row = 0; row != test.rows; ++row) {
                float sum{};
                if (test.alpha != 0)
                    for (int depth = 0; depth != test.depth; ++depth) {
                        const auto ai = test.ta == CUBLAS_OP_N ? depth * test.lda + row
                                                               : row * test.lda + depth;
                        const auto bi = test.tb == CUBLAS_OP_N ? column * test.ldb + depth
                                                               : depth * test.ldb + column;
                        sum += get(before[0], test.a, static_cast<std::size_t>(ai)) *
                               get(before[1], test.b, static_cast<std::size_t>(bi));
                    }
                const auto output = static_cast<std::size_t>(column) * test.ldc + row;
                const auto old = test.beta == 0 ? 0.0F : get(before[2], test.c, output);
                put(expected[2], test.c, output, test.alpha * sum + test.beta * old);
            }
    }
    return expected;
}
struct Totals {
    std::vector<std::string> cases;
    std::uint64_t checksum{kHashSeed};
    std::size_t checked_values{}, checked_bytes{}, producer_calls{};
};
struct OutputObservation {
    std::vector<std::uint32_t> bits;
    std::vector<const char*> categories;
};
OutputObservation observe_conditional_output(const Case& test, const Storage& actual,
                                             Storage& expected) {
    expect(test.c == CUDA_R_32F && (test.poison_inputs || test.poison_output),
           "conditional-input audit has an exact FP32 output representation");
    OutputObservation result;
    const auto observe = [&](std::size_t index) {
        const auto offset = (kGuard + index) * sizeof(float);
        expect(index < kPayload, "bounded conditional-input audit output");
        std::uint32_t bits{}, wanted{};
        std::memcpy(&bits, actual.data() + offset, sizeof(bits));
        std::memcpy(&wanted, expected.data() + offset, sizeof(wanted));
        const bool is_nan = (bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
                            (bits & UINT32_C(0x007fffff)) != 0;
        // Alpha=0 behavior is being observed, not assumed: exact original
        // DEVICE implementations can propagate poisoned A/B through arithmetic.
        // No other finite result is allowed. beta=0 retains its strict oracle.
        expect(bits == wanted || (test.poison_inputs && is_nan),
               "conditional audit permits only exact mathematical output or alpha-zero NaN");
        result.bits.push_back(bits);
        result.categories.push_back(bits == wanted ? "mathematical" : "nan");
        std::memcpy(expected.data() + offset, &bits, sizeof(bits));
    };
    if (test.vector) {
        const int length = test.ta == CUBLAS_OP_N ? test.rows : test.columns;
        const auto start = test.incy < 0 ? (length - 1) * -test.incy : 0;
        for (int index = 0; index != length; ++index)
            observe(static_cast<std::size_t>(start + static_cast<std::int64_t>(index) * test.incy));
    } else {
        for (int column = 0; column != test.columns; ++column)
            for (int row = 0; row != test.rows; ++row)
                observe(static_cast<std::size_t>(column) * test.ldc + row);
    }
    return result;
}
// NOLINTBEGIN(performance-no-int-to-ptr): original Driver/BLAS device-address ABIs.
void run_case(Native& native, const Functions& functions, const Case& test, bool device,
              int repeats, Totals& totals, bool is_boundary) {
    auto before = initial(test);
    for (std::size_t matrix = 0; matrix != before.size(); ++matrix)
        native.owned(cuMemcpyHtoD(native.matrices[matrix], before[matrix].data(), kStorageBytes));
    native.owned(cublasSetPointerMode(
        native.handle, device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST));
    for (int repeat = 0; repeat != repeats; ++repeat) {
        Scalars host;
        host.fill(-4096);
        host[kGuard] = test.alpha;
        host[kGuard + 1] = test.beta;
        const auto saved_host = host;
        auto uploaded = host;
        if (device) {
            uploaded[kGuard] = 7;
            uploaded[kGuard + 1] = -3;
        }
        native.owned(cuMemcpyHtoD(native.scalars, uploaded.data(), sizeof(uploaded)));
        native.proof();  // Pageable uploads complete before the nonblocking producer stream.
        if (device) {
            auto pointer = native.scalars + kGuard * sizeof(float);
            auto alpha = test.alpha, beta = test.beta;
            void* arguments[]{&pointer, &alpha, &beta};
            native.owned(cuLaunchKernel(native.producer, 320, 1, 1, 32, 1, 1, 0, native.stream,
                                        arguments, nullptr));
            ++totals.producer_calls;
        }
        const auto* alpha =
            device ? reinterpret_cast<const float*>(native.scalars + kGuard * sizeof(float))
                   : host.data() + kGuard;
        const auto* beta =
            device ? reinterpret_cast<const float*>(native.scalars + (kGuard + 1) * sizeof(float))
                   : host.data() + kGuard + 1;
        const auto* a =
            test.null_data || test.null_inputs
                ? nullptr
                : reinterpret_cast<const void*>(native.matrices[0] + kGuard * width(test.a));
        const auto* b =
            test.null_data || test.null_inputs
                ? nullptr
                : reinterpret_cast<const void*>(native.matrices[1] + kGuard * width(test.b));
        auto* c = test.null_data
                      ? nullptr
                      : reinterpret_cast<void*>(native.matrices[2] + kGuard * width(test.c));
        const auto sequence = totals.cases.size() + 1;
        const bool direct = sequence % 2 == 1;
        std::cerr << "BLAS case " << sequence << ' ' << (test.vector ? "sgemv " : "sgemm_ex ")
                  << test.name << ' ' << (device ? "device" : "host") << " repeat=" << repeat
                  << '\n';
        cublasStatus_t status{};
        if (test.vector) {
            const auto call = direct ? &cublasSgemv : functions.gemv;
            status = call(native.handle, test.ta, test.rows, test.columns, alpha,
                          static_cast<const float*>(a), test.lda, static_cast<const float*>(b),
                          test.incx, beta, static_cast<float*>(c), test.incy);
        } else {
            const auto call = direct ? &cublasSgemmEx : functions.gemm;
            status =
                call(native.handle, test.ta, test.tb, test.rows, test.columns, test.depth, alpha, a,
                     test.a, test.lda, b, test.b, test.ldb, beta, c, test.c, test.ldc);
        }
        native.proof();  // Never read/reuse/free backing storage after an unproven physical stop.
        verify_status(test, status);
        const auto expected = oracle(test, before, status);
        std::uint64_t hash = kHashSeed;
        OutputObservation observation;
        for (std::size_t matrix = 0; matrix != before.size(); ++matrix) {
            Storage actual;
            native.owned(cuMemcpyDtoH(actual.data(), native.matrices[matrix], actual.size()));
            auto comparison = expected[matrix];
            if (matrix == 2 && status == CUBLAS_STATUS_SUCCESS &&
                (test.poison_inputs || test.poison_output)) {
                expect(repeats == 1, "conditional observation belongs only to the audit matrix");
                observation = observe_conditional_output(test, actual, comparison);
            }
            expect(actual == comparison,
                   "complete exact matrix/vector/input/stride/padding/guard oracle");
            checksum(hash, actual.data(), actual.size());
        }
        Scalars actual_scalars;
        native.owned(cuMemcpyDtoH(actual_scalars.data(), native.scalars, sizeof(actual_scalars)));
        using ScalarBytes = std::array<std::byte, sizeof(Scalars)>;
        expect(
            std::bit_cast<ScalarBytes>(actual_scalars) == std::bit_cast<ScalarBytes>(saved_host) &&
                std::bit_cast<ScalarBytes>(host) == std::bit_cast<ScalarBytes>(saved_host),
            "same-stream producer scalars and complete scalar guards");
        checksum(hash, actual_scalars.data(), sizeof(actual_scalars));
        checksum(hash, host.data(), sizeof(host));
        cublasPointerMode_t observed_mode{};
        cudaStream_t observed_stream{};
        native.owned(cublasGetPointerMode(native.handle, &observed_mode));
        native.owned(cublasGetStream(native.handle, &observed_stream));
        expect(observed_mode == (device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST) &&
                   observed_stream == reinterpret_cast<cudaStream_t>(native.stream),
               "public stream and pointer mode remain unchanged");
        const auto values = kStorageBytes / width(test.a) + kStorageBytes / width(test.b) +
                            kStorageBytes / width(test.c) + 2 * kScalarElements;
        constexpr auto bytes = 3 * kStorageBytes + 2 * sizeof(Scalars);
        std::ostringstream record;
        record
            << "{\"sequence\":" << sequence << ",\"api\":\"" << (test.vector ? "sgemv" : "sgemm_ex")
            << "\",\"name\":\"" << test.name << "\",\"mode\":\"" << (device ? "device" : "host")
            << "\",\"lookup\":\"" << (direct ? "direct" : "cached")
            << "\",\"transpose_a\":" << test.ta << ",\"transpose_b\":" << test.tb
            << ",\"type_a\":" << test.a << ",\"type_b\":" << test.b << ",\"type_c\":" << test.c
            << ",\"rows\":" << test.rows << ",\"columns\":" << test.columns
            << ",\"depth\":" << test.depth << ",\"leading_a\":" << test.lda
            << ",\"leading_b\":" << test.ldb << ",\"leading_c\":" << test.ldc
            << ",\"increment_x\":" << test.incx << ",\"increment_y\":" << test.incy
            << ",\"alpha\":" << test.alpha << ",\"beta\":" << test.beta << ",\"repeat\":" << repeat
            << ",\"null_data\":" << (test.null_data ? "true" : "false")
            << ",\"poison_inputs\":" << (test.poison_inputs ? "true" : "false")
            << ",\"poison_output\":" << (test.poison_output ? "true" : "false")
            << ",\"status\":" << static_cast<int>(status) << ",\"checked_values\":" << values
            << ",\"checked_bytes\":" << bytes << ",\"checksum\":" << hash
            << ",\"all_values_verified\":true,\"guards_verified\":true,\"scalars_unchanged\":true";
        if (is_boundary) record << ",\"null_inputs\":" << (test.null_inputs ? "true" : "false");
        if (test.poison_inputs || test.poison_output) {
            record << ",\"logical_output_bits\":[";
            for (std::size_t index = 0; index != observation.bits.size(); ++index) {
                if (index) record << ',';
                record << observation.bits[index];
            }
            record << "],\"logical_output_categories\":[";
            for (std::size_t index = 0; index != observation.categories.size(); ++index) {
                if (index) record << ',';
                record << '"' << observation.categories[index] << '"';
            }
            record << ']';
        }
        record << '}';
        totals.cases.push_back(record.str());
        totals.checked_values += values;
        totals.checked_bytes += bytes;
        checksum(totals.checksum, &hash, sizeof(hash));
        before = expected;  // Second invocation is non-idempotent and never resets the result.
    }
}
// NOLINTEND(performance-no-int-to-ptr)
}  // namespace

int main(int argc, char** argv) {
    Native native;
    void* provider{};
    try {
        expect(argc >= 2,
               "usage: execution_blas_extension_gpu_test ORIGINAL_LIBCUBLAS [--audit | "
               "--boundary] [--output FILE]");
        bool audit{}, boundary{};
        std::string output_path;
        for (int index = 2; index < argc; ++index) {
            if (std::string_view(argv[index]) == "--audit" && !audit && !boundary)
                audit = true;
            else if (std::string_view(argv[index]) == "--boundary" && !audit && !boundary)
                boundary = true;
            else if (std::string_view(argv[index]) == "--output" && output_path.empty() &&
                     index + 1 < argc)
                output_path = argv[++index];
            else
                throw std::runtime_error("unknown or duplicate SDK fixture option");
        }
        std::ofstream output;
        if (!output_path.empty()) {
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output.open(output_path);
        }
        provider = ::dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        expect(provider != nullptr, "open exact original SDK export provider");
        const Functions functions(provider);
        native.start();
        const auto inventory = audit      ? audit_inventory()
                               : boundary ? boundary_inventory()
                                          : core_inventory();
        const auto expected_calls = audit ? 58U : boundary ? 120U : 276U;
        Totals totals;
        totals.cases.reserve(expected_calls);
        for (const auto& test : inventory)
            for (const bool device : {false, true})
                run_case(native, functions, test, device, audit ? 1 : 2, totals, boundary);
        expect(totals.cases.size() == expected_calls, "complete fixed selected inventory");
        native.close();
        expect(!native.context && !native.uncertain, "normal physical and owner cleanup completed");
        expect(::dlclose(provider) == 0, "release explicit SDK lookup provider");
        provider = nullptr;
        std::ostringstream receipt;
        receipt
            << "{\"record\":\"summary\",\"audit\":" << (audit ? "true" : "false")
            << ",\"api_calls\":" << totals.cases.size()
            << ",\"checked_values\":" << totals.checked_values
            << ",\"checked_bytes\":" << totals.checked_bytes << ",\"checksum\":" << totals.checksum
            << ",\"gpu_uuid\":\"" << native.gpu_uuid
            << "\",\"driver_version\":" << native.driver_version
            << ",\"cublas_version\":" << native.cublas_version
            << ",\"producer_symbol\":\"blas_extension_coefficients\",\"producer_blocks\":320,"
               "\"producer_threads\":32"
            << ",\"producer_calls\":" << totals.producer_calls
            << ",\"normal_cleanup\":true,\"numerical_diagnostic_passed\":true,\"full_device_"
               "qualified\":false"
            << ",\"priority_qualified\":false,\"formal_comparison_qualified\":false"
            << ",\"unqualified_boundaries\":[\"null_scalars\",\"unknown_or_protected_host_"
               "pointers\","
               "\"nonzero_extent_null_output\",\"output_input_overlap\",\"extreme_stride_extent\"]";
        if (boundary) receipt << ",\"boundary\":true";
        receipt << ",\"cases\":[";
        for (std::size_t index = 0; index != totals.cases.size(); ++index) {
            if (index) receipt << ',';
            receipt << totals.cases[index];
        }
        receipt << "]}\n";
        if (output.is_open()) {
            output << receipt.str();
            output.close();
        }
        std::cout << receipt.str() << std::flush;
        expect(std::cout.good(), "complete stdout receipt");
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
        if (provider && !native.uncertain) {
            if (::dlclose(provider) != 0)
                std::cerr << "SDK lookup provider cleanup failed\n";
            else
                provider = nullptr;
        }
        std::cerr << "Failed SDK input cleanup proved="
                  << (!native.context && !native.uncertain && !provider ? "true" : "false") << '\n';
        return 1;
    }
}
