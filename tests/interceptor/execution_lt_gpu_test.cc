#include <cuda.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <span>
#include <vector>

namespace {
void check(CUresult result) {
    if (result != CUDA_SUCCESS) throw std::runtime_error("Driver " + std::to_string(result));
}
void check(cublasStatus_t result, unsigned int line = __builtin_LINE()) {
    if (result != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Lt " + std::to_string(result) + "; line=" + std::to_string(line));
}
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Buffer {
    cudaDataType type;
    std::vector<std::byte> host;
    CUdeviceptr device{};
    Buffer(cudaDataType format, std::size_t count) : type(format), host(count * width()) {
        check(cuMemAlloc(&device, host.size()));
    }
    std::size_t width() const {
        return type == CUDA_R_16F ? 2 : 4;
    }
    void put(std::size_t index, float value) {
        if (type == CUDA_R_16F) {
            const auto half = __float2half(value);
            std::memcpy(host.data() + index * width(), &half, sizeof(half));
        } else
            std::memcpy(host.data() + index * width(), &value, sizeof(value));
    }
    float get(std::size_t index) const {
        return get_from(host, index);
    }
    float get_from(std::span<const std::byte> values, std::size_t index) const {
        if (type == CUDA_R_16F) {
            __half half;
            std::memcpy(&half, values.data() + index * width(), sizeof(half));
            return __half2float(half);
        }
        float value{};
        std::memcpy(&value, values.data() + index * width(), sizeof(value));
        return value;
    }
    void upload() const {
        check(cuMemcpyHtoD(device, host.data(), host.size()));
    }
    std::vector<std::byte> download() const {
        std::vector<std::byte> result(host.size());
        check(cuMemcpyDtoH(result.data(), device, result.size()));
        return result;
    }
    void release() {
        check(cuMemFree(device));
        device = 0;
    }
};
void run(bool use_blas_handle, std::string_view profile, int algorithm_index,
         bool use_fresh_operation, bool use_scalar_canary) {
    std::cout << "Lt diagnostic controls: algorithm_index=" << algorithm_index
              << "; fresh_operation=" << use_fresh_operation
              << "; scalar_canary=" << use_scalar_canary << std::endl;
    // This exactly FP16-representable FP32 alpha has low 16 bits encoding
    // FP16 2.0, a different value from the actual FP32 scalar.
    // It distinguishes an incorrect half load from a missing product term.
    const float alpha = use_scalar_canary ? std::bit_cast<float>(std::uint32_t{0x40004000}) : 2.0F;
    const float beta = 0.5F;
    const auto alpha_bits = static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(alpha));
    const float misread_alpha = __half2float(std::bit_cast<__half>(alpha_bits));
    check(cuInit(0));
    CUdevice device{};
    CUcontext context{};
    check(cuDeviceGet(&device, 0));
    check(cuCtxCreate(&context, 0, device));
    cublasLtHandle_t handle{};
    cublasHandle_t blas_handle{};
    if (use_blas_handle) {
        check(cublasCreate_v2(&blas_handle));
        handle = reinterpret_cast<cublasLtHandle_t>(blas_handle);
    } else
        check(cublasLtCreate(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    constexpr int rows = 64, columns = 96, depth = 32;
    std::size_t calls{}, checked_values{};
    const std::array formats{std::array{CUDA_R_32F, CUDA_R_32F}, std::array{CUDA_R_16F, CUDA_R_16F},
                             std::array{CUDA_R_16F, CUDA_R_32F}};
    for (const auto& format : formats) {
        if ((profile == "fp32" && format[0] != CUDA_R_32F) ||
            (profile == "fp16" && format[1] != CUDA_R_16F) ||
            (profile == "fp16-fp32" && (format[0] != CUDA_R_16F || format[1] != CUDA_R_32F)))
            continue;
        for (const auto ta : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
            for (const auto tb : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
                std::cout << "Lt profile: input=" << static_cast<int>(format[0])
                          << "; output=" << static_cast<int>(format[1])
                          << "; ta=" << static_cast<int>(ta) << "; tb=" << static_cast<int>(tb)
                          << std::endl;
                const int a_rows = ta == CUBLAS_OP_N ? rows : depth;
                const int a_columns = ta == CUBLAS_OP_N ? depth : rows;
                const int b_rows = tb == CUBLAS_OP_N ? depth : columns;
                const int b_columns = tb == CUBLAS_OP_N ? columns : depth;
                const std::array leading{a_rows + 8, b_rows + 8, rows + 8, rows + 16};
                const std::array<std::size_t, 4> extents{
                    static_cast<std::size_t>(leading[0] * a_columns + 32),
                    static_cast<std::size_t>(leading[1] * b_columns + 32),
                    static_cast<std::size_t>(leading[2] * columns + 32),
                    static_cast<std::size_t>(leading[3] * columns + 32)};
                Buffer a(format[0], extents[0]), b(format[0], extents[1]), c(format[1], extents[2]),
                    d(format[1], extents[3]), bias(format[1], rows + 32),
                    workspace(CUDA_R_32F, 262144), scalars(CUDA_R_32F, 8);
                for (auto* buffer : {&a, &b, &c, &d, &bias})
                    for (std::size_t index = 0; index < buffer->host.size() / buffer->width();
                         ++index)
                        buffer->put(index,
                                    static_cast<float>(static_cast<int>(index % 9) - 4) / 8.0F);
                for (auto* buffer : {&a, &b, &c, &bias}) buffer->upload();
                scalars.put(0, alpha);
                scalars.put(4, beta);
                scalars.upload();
                std::array<cublasLtMatrixLayout_t, 4> layouts{};
                check(cublasLtMatrixLayoutCreate(&layouts[0], format[0], a_rows, a_columns,
                                                 leading[0]));
                check(cublasLtMatrixLayoutCreate(&layouts[1], format[0], b_rows, b_columns,
                                                 leading[1]));
                check(
                    cublasLtMatrixLayoutCreate(&layouts[2], format[1], rows, columns, leading[2]));
                check(
                    cublasLtMatrixLayoutCreate(&layouts[3], format[1], rows, columns, leading[3]));
                cublasLtMatmulDesc_t operation{};
                cublasLtMatmulPreference_t preference{};
                check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
                check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                                     sizeof(ta)));
                check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                                     sizeof(tb)));
                check(cublasLtMatmulPreferenceCreate(&preference));
                const std::size_t maximum_workspace = workspace.host.size();
                check(cublasLtMatmulPreferenceSetAttribute(
                    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maximum_workspace,
                    sizeof(maximum_workspace)));
                for (const bool with_bias : {false, true})
                    for (const bool device_mode : {false, true}) {
                        if (use_fresh_operation) {
                            check(cublasLtMatmulDescDestroy(operation));
                            check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F,
                                                           CUDA_R_32F));
                            check(cublasLtMatmulDescSetAttribute(
                                operation, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
                            check(cublasLtMatmulDescSetAttribute(
                                operation, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
                        }
                        const auto epilogue =
                            with_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
                        const auto pointer_mode =
                            device_mode ? CUBLASLT_POINTER_MODE_DEVICE : CUBLASLT_POINTER_MODE_HOST;
                        const auto* bias_pointer =
                            with_bias ? reinterpret_cast<const void*>(bias.device) : nullptr;
                        check(cublasLtMatmulDescSetAttribute(
                            operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
                        check(cublasLtMatmulDescSetAttribute(operation,
                                                             CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                             &pointer_mode, sizeof(pointer_mode)));
                        check(cublasLtMatmulDescSetAttribute(operation,
                                                             CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                             &bias_pointer, sizeof(bias_pointer)));
                        std::array<cublasLtMatmulHeuristicResult_t, 4> algorithms{};
                        int count{};
                        check(cublasLtMatmulAlgoGetHeuristic(
                            handle, operation, layouts[0], layouts[1], layouts[2], layouts[3],
                            preference, 4, algorithms.data(), &count));
                        expect(count > algorithm_index && count <= 4 &&
                                   algorithms[static_cast<std::size_t>(algorithm_index)].state ==
                                       CUBLAS_STATUS_SUCCESS,
                               "complete original heuristic");
                        const auto& algorithm =
                            algorithms[static_cast<std::size_t>(algorithm_index)].algo;
                        cublasLtMatmulHeuristicResult_t validation{};
                        check(cublasLtMatmulAlgoCheck(handle, operation, layouts[0], layouts[1],
                                                      layouts[2], layouts[3], &algorithm,
                                                      &validation));
                        expect(validation.state == CUBLAS_STATUS_SUCCESS &&
                                   validation.workspaceSize <= maximum_workspace,
                               "original algorithm check accepts the complete workspace");
                        d.upload();
                        check(cuCtxSynchronize());
                        expect(scalars.download() == scalars.host,
                               "device scalar storage is unchanged before actual matmul");
                        cublasLtPointerMode_t observed_mode{};
                        std::size_t written{};
                        check(cublasLtMatmulDescGetAttribute(
                            operation, CUBLASLT_MATMUL_DESC_POINTER_MODE, &observed_mode,
                            sizeof(observed_mode), &written));
                        expect(written == sizeof(observed_mode) && observed_mode == pointer_mode,
                               "original SDK descriptor retains the selected scalar mode");
                        if (!calls) {
                            const auto* invalid_bias = reinterpret_cast<const void*>(bias.device);
                            check(cublasLtMatmulDescSetAttribute(
                                operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &invalid_bias,
                                sizeof(invalid_bias)));
                            const auto rejected = cublasLtMatmul(
                                handle, operation, &alpha, reinterpret_cast<const void*>(a.device),
                                layouts[0], reinterpret_cast<const void*>(b.device), layouts[1],
                                &beta, reinterpret_cast<const void*>(c.device), layouts[2],
                                reinterpret_cast<void*>(d.device), layouts[3], &algorithm,
                                reinterpret_cast<void*>(workspace.device), maximum_workspace,
                                reinterpret_cast<cudaStream_t>(stream));
                            expect(
                                rejected == CUBLAS_STATUS_INVALID_VALUE && d.download() == d.host,
                                "default epilogue with nonnull bias retains native invalid-value "
                                "status and unchanged output");
                            check(cublasLtMatmulDescSetAttribute(
                                operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
                                sizeof(bias_pointer)));
                        }
                        check(cublasLtMatmul(
                            handle, operation,
                            device_mode ? reinterpret_cast<const void*>(scalars.device) : &alpha,
                            reinterpret_cast<const void*>(a.device), layouts[0],
                            reinterpret_cast<const void*>(b.device), layouts[1],
                            device_mode ? reinterpret_cast<const void*>(scalars.device + 16)
                                        : &beta,
                            reinterpret_cast<const void*>(c.device), layouts[2],
                            reinterpret_cast<void*>(d.device), layouts[3], &algorithm,
                            reinterpret_cast<void*>(workspace.device), maximum_workspace,
                            reinterpret_cast<cudaStream_t>(stream)));
                        check(cuStreamSynchronize(stream));
                        const auto actual = d.download();
                        // Preserve independent input evidence even when the
                        // output oracle fails; never diagnose only the first value.
                        for (const auto* buffer : {&a, &b, &c, &bias, &scalars})
                            expect(buffer->download() == buffer->host,
                                   "all inputs and scalars remain unchanged after matmul");
                        auto expected = d.host;
                        std::size_t missing_product_matches{}, nonzero_product_outputs{},
                            misread_alpha_matches{};
                        for (int column = 0; column < columns; ++column)
                            for (int row = 0; row < rows; ++row) {
                                float sum{};
                                for (int inner = 0; inner < depth; ++inner) {
                                    const auto ai = ta == CUBLAS_OP_N ? row + inner * leading[0]
                                                                      : inner + row * leading[0];
                                    const auto bi = tb == CUBLAS_OP_N ? inner + column * leading[1]
                                                                      : column + inner * leading[1];
                                    sum += a.get(static_cast<std::size_t>(ai)) *
                                           b.get(static_cast<std::size_t>(bi));
                                }
                                const auto index =
                                    static_cast<std::size_t>(row + column * leading[3]);
                                const auto epilogue_only =
                                    0.5F *
                                        c.get(static_cast<std::size_t>(row + column * leading[2])) +
                                    (with_bias ? bias.get(static_cast<std::size_t>(row)) : 0.0F);
                                const auto value =
                                    alpha * sum +
                                    beta *
                                        c.get(static_cast<std::size_t>(row + column * leading[2])) +
                                    (with_bias ? bias.get(static_cast<std::size_t>(row)) : 0.0F);
                                const auto round_output = [&](float number) {
                                    return format[1] == CUDA_R_16F
                                               ? __half2float(__float2half(number))
                                               : number;
                                };
                                if (round_output(value) != round_output(epilogue_only)) {
                                    ++nonzero_product_outputs;
                                    if (d.get_from(actual, index) == round_output(epilogue_only))
                                        ++missing_product_matches;
                                    const auto misread_value =
                                        misread_alpha * sum +
                                        beta * c.get(static_cast<std::size_t>(
                                                   row + column * leading[2])) +
                                        (with_bias ? bias.get(static_cast<std::size_t>(row))
                                                   : 0.0F);
                                    if (d.get_from(actual, index) == round_output(misread_value))
                                        ++misread_alpha_matches;
                                }
                                d.put(index, value);
                                std::memcpy(expected.data() + index * d.width(),
                                            d.host.data() + index * d.width(), d.width());
                            }
                        if (actual != expected) {
                            std::cerr
                                << "Lt mismatch diagnosis: algorithm_index=" << algorithm_index
                                << "; fresh_operation=" << use_fresh_operation
                                << "; inputs_and_scalars_unchanged=1"
                                << "; nonzero_product_outputs=" << nonzero_product_outputs
                                << "; missing_product_matches=" << missing_product_matches
                                << "; misread_alpha_matches=" << misread_alpha_matches << '\n';
                            for (std::size_t index = 0; index < expected.size() / d.width();
                                 ++index)
                                if (std::memcmp(actual.data() + index * d.width(),
                                                expected.data() + index * d.width(),
                                                d.width()) != 0)
                                    throw std::runtime_error(
                                        "Lt numerical mismatch: index=" + std::to_string(index) +
                                        "; actual=" + std::to_string(d.get_from(actual, index)) +
                                        "; expected=" +
                                        std::to_string(d.get_from(expected, index)) +
                                        "; bias=" + std::to_string(with_bias) +
                                        "; device_scalars=" + std::to_string(device_mode));
                        }
                        for (const auto* buffer : {&a, &b, &c, &bias}) {
                            checked_values += buffer->host.size() / buffer->width();
                        }
                        checked_values += expected.size() / d.width();
                        ++calls;
                    }
                check(cublasLtMatmulPreferenceDestroy(preference));
                check(cublasLtMatmulDescDestroy(operation));
                for (auto layout : layouts) check(cublasLtMatrixLayoutDestroy(layout));
                for (auto* buffer : {&a, &b, &c, &d, &bias, &workspace, &scalars})
                    buffer->release();
            }
    }
    if (use_blas_handle)
        check(cublasDestroy_v2(blas_handle));
    else
        check(cublasLtDestroy(handle));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    std::cout << "Public Lt numerical chain verified: calls=" << calls
              << "; values=" << checked_values
              << "; invalid_status_controls=1; full_device_qualified=false; "
                 "priority_qualified=false; performance_qualified=false\n";
}
// NOLINTBEGIN(performance-no-int-to-ptr): exact original SDK GPU address ABI.
void run_bert_split() {
    check(cuInit(0));
    CUdevice device{};
    CUcontext context{};
    check(cuDeviceGet(&device, 0));
    check(cuCtxCreate(&context, 0, device));
    cublasLtHandle_t handle{};
    check(cublasLtCreate(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    constexpr int rows = 768, columns = 768;
    constexpr float alpha = 1.0F, beta = 0.0F;
    constexpr std::size_t guard = 32, workspace_bytes = 8519680;
    for (const int depth : {768, 3072}) {
        const int splits = depth == 768 ? 6 : 3;
        const int partition = depth / splits;
        Buffer a(CUDA_R_32F, static_cast<std::size_t>(rows) * depth + guard),
            b(CUDA_R_32F, static_cast<std::size_t>(columns) * depth + guard),
            output(CUDA_R_32F, static_cast<std::size_t>(rows) * columns + guard),
            bias(CUDA_R_32F, rows + guard), workspace(CUDA_R_32F, workspace_bytes / 4);
        for (int row = 0; row != rows; ++row) {
            const auto base = static_cast<std::size_t>(row) * depth;
            // Partition partials cancel at large magnitude; adding the small
            // bias after reduction gives a detectably different answer.
            a.put(base, 1e8F);
            a.put(base + partition, -1e8F);
            a.put(base + 2 * static_cast<std::size_t>(partition), static_cast<float>(1 + row % 7));
            bias.put(static_cast<std::size_t>(row), 0.125F);
        }
        for (int column = 0; column != columns; ++column)
            for (int k = 0; k != depth; ++k)
                b.put(static_cast<std::size_t>(column) * depth + k, column % 2 ? -1.0F : 1.0F);
        for (std::size_t i = 0; i != output.host.size() / 4; ++i) output.put(i, 0.375F);
        for (auto* buffer : {&a, &b, &output, &bias}) buffer->upload();
        std::array<cublasLtMatrixLayout_t, 3> layouts{};
        check(cublasLtMatrixLayoutCreate(&layouts[0], CUDA_R_32F, depth, rows, depth));
        check(cublasLtMatrixLayoutCreate(&layouts[1], CUDA_R_32F, depth, columns, depth));
        check(cublasLtMatrixLayoutCreate(&layouts[2], CUDA_R_32F, rows, columns, rows));
        cublasLtMatmulDesc_t operation{};
        cublasLtMatmulPreference_t preference{};
        check(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        constexpr auto transpose = CUBLAS_OP_T;
        constexpr auto epilogue = CUBLASLT_EPILOGUE_BIAS;
        const auto* bias_pointer = reinterpret_cast<const void*>(bias.device);
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                             sizeof(transpose)));
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                                             sizeof(epilogue)));
        check(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                             &bias_pointer, sizeof(bias_pointer)));
        check(cublasLtMatmulPreferenceCreate(&preference));
        check(cublasLtMatmulPreferenceSetAttribute(preference,
                                                   CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &workspace_bytes, sizeof(workspace_bytes)));
        std::array<cublasLtMatmulHeuristicResult_t, 64> algorithms{};
        int count{};
        check(cublasLtMatmulAlgoGetHeuristic(handle, operation, layouts[0], layouts[1], layouts[2],
                                             layouts[2], preference, 64, algorithms.data(),
                                             &count));
        expect(count > 0 && count <= 64, "bounded original BERT algorithm inventory");
        const cublasLtMatmulAlgo_t* algorithm{};
        for (int i = 0; i != count; ++i) {
            std::uint32_t split{}, reduction{};
            std::size_t written{};
            check(cublasLtMatmulAlgoConfigGetAttribute(&algorithms[i].algo,
                                                       CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split,
                                                       sizeof(split), &written));
            expect(written == sizeof(split), "complete original split count");
            check(cublasLtMatmulAlgoConfigGetAttribute(&algorithms[i].algo,
                                                       CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                                       &reduction, sizeof(reduction), &written));
            expect(written == sizeof(reduction), "complete original reduction scheme");
            if (split == static_cast<std::uint32_t>(splits) &&
                reduction == CUBLASLT_REDUCTION_SCHEME_INPLACE &&
                algorithms[i].state == CUBLAS_STATUS_SUCCESS) {
                algorithm = &algorithms[i].algo;
                break;
            }
        }
        expect(algorithm, "original heuristic supplies the pinned BERT split profile");
        cublasLtMatmulHeuristicResult_t validated{};
        check(cublasLtMatmulAlgoCheck(handle, operation, layouts[0], layouts[1], layouts[2],
                                      layouts[2], algorithm, &validated));
        expect(
            validated.state == CUBLAS_STATUS_SUCCESS && validated.workspaceSize <= workspace_bytes,
            "original BERT algorithm and workspace validation");
        check(cuCtxSynchronize());
        check(cublasLtMatmul(handle, operation, &alpha, reinterpret_cast<const void*>(a.device),
                             layouts[0], reinterpret_cast<const void*>(b.device), layouts[1], &beta,
                             reinterpret_cast<const void*>(output.device), layouts[2],
                             reinterpret_cast<void*>(output.device), layouts[2], algorithm,
                             reinterpret_cast<void*>(workspace.device), workspace_bytes,
                             reinterpret_cast<cudaStream_t>(stream)));
        check(cuStreamSynchronize(stream));
        const auto observed = output.download();
        std::array<std::array<float, rows>, 2> oracle{};
        // Every column has one of two exact input vectors, so compute both
        // full dot products per row once without weakening output coverage.
        for (std::size_t sign = 0; sign != oracle.size(); ++sign)
            for (int row = 0; row != rows; ++row) {
                float expected = bias.get(static_cast<std::size_t>(row));
                for (int part = 0; part != splits; ++part) {
                    float partial{};
                    for (int k = part * partition; k != (part + 1) * partition; ++k)
                        partial = std::fma(a.get(static_cast<std::size_t>(row) * depth + k),
                                           b.get(sign * depth + k), partial);
                    const volatile float scaled = alpha * partial;
                    expected += scaled;
                }
                oracle[sign][row] = expected;
            }
        for (int column = 0; column != columns; ++column)
            for (int row = 0; row != rows; ++row) {
                const auto index = static_cast<std::size_t>(column) * rows + row;
                expect(output.get_from(observed, index) == oracle[column % 2][row],
                       "every output preserves independent partition and bias-seed oracle");
            }
        const auto payload = static_cast<std::size_t>(rows) * columns * sizeof(float);
        expect(std::equal(observed.begin() + static_cast<std::ptrdiff_t>(payload), observed.end(),
                          output.host.begin() + static_cast<std::ptrdiff_t>(payload)),
               "full trailing output guard unchanged");
        for (const auto* buffer : {&a, &b, &bias})
            expect(buffer->download() == buffer->host, "all split inputs and guards unchanged");
        for (auto layout : layouts) check(cublasLtMatrixLayoutDestroy(layout));
        check(cublasLtMatmulPreferenceDestroy(preference));
        check(cublasLtMatmulDescDestroy(operation));
        for (auto* buffer : {&a, &b, &output, &bias, &workspace}) buffer->release();
        std::cout << "Lt split diagnostic verified: depth=" << depth << "; splits=" << splits
                  << "; values=" << rows * columns
                  << "; bias_seed_once=true; guards=true; "
                     "full_device_qualified=false; priority_qualified=false\n";
    }
    check(cublasLtDestroy(handle));
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
}
// NOLINTEND(performance-no-int-to-ptr)
}  // namespace
int main(int argc, char** argv) {
    try {
        bool use_blas_handle{};
        bool use_fresh_operation{};
        bool use_scalar_canary{};
        bool uses_bert_split{};
        int algorithm_index{};
        std::string_view profile = "all";
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            if (argument == "--blas-handle")
                use_blas_handle = true;
            else if (argument == "--fresh-operation")
                use_fresh_operation = true;
            else if (argument == "--scalar-canary")
                use_scalar_canary = true;
            else if (argument == "--bert-split")
                uses_bert_split = true;
            else if (argument == "--algorithm-index" && index + 1 < argc) {
                const std::string_view value = argv[++index];
                const auto parsed =
                    std::from_chars(value.data(), value.data() + value.size(), algorithm_index);
                expect(parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size() &&
                           algorithm_index >= 0 && algorithm_index < 4,
                       "bounded Lt diagnostic algorithm index");
            } else if (argument == "--profile" && index + 1 < argc)
                profile = argv[++index];
            else
                throw std::runtime_error("unknown Lt diagnostic argument");
        }
        expect(profile == "all" || profile == "fp32" || profile == "fp16" || profile == "fp16-fp32",
               "known Lt diagnostic profile");
        if (uses_bert_split) {
            expect(!use_blas_handle && !use_fresh_operation && !use_scalar_canary &&
                       algorithm_index == 0 && profile == "all",
                   "BERT split diagnostic does not combine alternate controls");
            run_bert_split();
        } else
            run(use_blas_handle, profile, algorithm_index, use_fresh_operation, use_scalar_canary);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
