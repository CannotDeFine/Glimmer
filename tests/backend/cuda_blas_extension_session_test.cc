#include "internal/cuda_execution_session.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
constexpr std::size_t kBytes = 512, kOffset = 32;
void expect(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned Driver operation succeeds");
}
void success(const CudaSessionBlasResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete &&
               result.status == CUDA_SUCCESS && result.original_status == CUBLAS_STATUS_SUCCESS,
           "actual owned BLAS metadata succeeds");
}
struct State {
    char context{}, handle{}, stream{};
    CUcontext current{};
    cudaStream_t stream_value{};
    cublasPointerMode_t mode{CUBLAS_POINTER_MODE_HOST};
    std::map<CUdeviceptr, std::vector<std::byte>> allocations;
    std::map<CUevent, std::unique_ptr<int>> events;
    std::array<CUdeviceptr, 5> last_addresses{};
    std::array<int, 8> last_shape{};
    std::array<cudaDataType, 3> last_types{};
    std::array<std::byte, sizeof(float)> last_alpha{}, last_beta{};
    bool context_live{}, handle_live{}, stream_live{}, can_stop{};
    unsigned int gemv_calls{}, gemm_calls{}, frees{}, destroys{}, context_destroys{}, copies{};
    unsigned int synchronizes{}, downloads{}, mode_sets{}, stream_queries{}, mode_queries{};
    unsigned int event_records{};
    unsigned int version_queries{}, last_scalar_presence{};
    int version{120604};
    bool is_metadata_case{};
    cublasStatus_t native_status{CUBLAS_STATUS_SUCCESS};
    cublasStatus_t stream_status{CUBLAS_STATUS_SUCCESS}, mode_status{CUBLAS_STATUS_SUCCESS};
    cublasStatus_t version_status{CUBLAS_STATUS_SUCCESS};
    std::function<void()> in_native, in_version;
} state;
CUcontext context() {
    return reinterpret_cast<CUcontext>(&state.context);
}
void native_owner(cublasHandle_t handle) {
    expect(state.context_live && state.current == context() && state.handle_live &&
               handle == reinterpret_cast<cublasHandle_t>(&state.handle),
           "SDK receives the exact live context-owned handle");
}
std::byte* memory(CUdeviceptr pointer, std::size_t bytes) {
    for (auto& [base, allocation] : state.allocations)
        if (pointer >= base && pointer - base <= allocation.size() &&
            bytes <= allocation.size() - static_cast<std::size_t>(pointer - base))
            return allocation.data() + (pointer - base);
    throw std::runtime_error("native fixture cannot access outside an owned allocation");
}
float read_float(CUdeviceptr pointer) {
    float value{};
    std::memcpy(&value, memory(pointer, sizeof(value)), sizeof(value));
    return value;
}
void write_float(CUdeviceptr pointer, float value) {
    std::memcpy(memory(pointer, sizeof(value)), &value, sizeof(value));
}
float scalar(const float* pointer) {
    expect(pointer, "fully owned computation preserves nonnull scalar presence");
    float value{};
    const auto* bytes = state.mode == CUBLAS_POINTER_MODE_DEVICE
                            ? memory(reinterpret_cast<CUdeviceptr>(pointer), sizeof(float))
                            : reinterpret_cast<const std::byte*>(pointer);
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}
void coefficients(const float* alpha, const float* beta) {
    const auto a = scalar(alpha), b = scalar(beta);
    std::memcpy(state.last_alpha.data(), &a, sizeof(a));
    std::memcpy(state.last_beta.data(), &b, sizeof(b));
}
struct VectorLayout {
    int length{}, increment{};
};
std::int64_t vector_element(int index, VectorLayout layout) {
    const auto stride = static_cast<std::int64_t>(layout.increment);
    return (stride < 0 ? (1 - static_cast<std::int64_t>(layout.length)) * stride : 0) +
           index * stride;
}
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact SDK callback signatures.
cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t transpose, int rows, int columns,
                    const float* alpha, const float* a, int leading, const float* x,
                    int increment_x, const float* beta, float* y, int increment_y) {
    native_owner(handle);
    ++state.gemv_calls;
    state.last_addresses = {reinterpret_cast<CUdeviceptr>(a), reinterpret_cast<CUdeviceptr>(x),
                            reinterpret_cast<CUdeviceptr>(y), reinterpret_cast<CUdeviceptr>(alpha),
                            reinterpret_cast<CUdeviceptr>(beta)};
    state.last_shape = {
        static_cast<int>(transpose), rows, columns, leading, increment_x, increment_y};
    state.last_scalar_presence = (alpha ? 1U : 0U) | (beta ? 2U : 0U);
    if (state.is_metadata_case) {
        if (state.in_native) state.in_native();
        return state.native_status;
    }
    coefficients(alpha, beta);
    if (state.in_native) state.in_native();
    if (state.native_status != CUBLAS_STATUS_SUCCESS) return state.native_status;
    const float av = scalar(alpha), bv = scalar(beta);
    const int outputs = transpose == CUBLAS_OP_N ? rows : columns;
    const int inputs = transpose == CUBLAS_OP_N ? columns : rows;
    for (int output = 0; output < outputs; ++output) {
        float dot{};
        for (int input = 0; input < inputs; ++input) {
            const int element =
                transpose == CUBLAS_OP_N ? input * leading + output : output * leading + input;
            dot += read_float(state.last_addresses[0] + sizeof(float) * element) *
                   read_float(state.last_addresses[1] +
                              sizeof(float) * vector_element(input, {.length = inputs,
                                                                     .increment = increment_x}));
        }
        const auto destination =
            state.last_addresses[2] +
            sizeof(float) * vector_element(output, {.length = outputs, .increment = increment_y});
        write_float(destination, av * dot + bv * read_float(destination));
    }
    return state.native_status;
}
cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows,
                    int columns, int depth, const float* alpha, const void* a, cudaDataType type_a,
                    int lda, const void* b, cudaDataType type_b, int ldb, const float* beta,
                    void* c, cudaDataType type_c, int ldc) {
    native_owner(handle);
    ++state.gemm_calls;
    state.last_addresses = {reinterpret_cast<CUdeviceptr>(a), reinterpret_cast<CUdeviceptr>(b),
                            reinterpret_cast<CUdeviceptr>(c), reinterpret_cast<CUdeviceptr>(alpha),
                            reinterpret_cast<CUdeviceptr>(beta)};
    state.last_shape = {
        static_cast<int>(ta), static_cast<int>(tb), rows, columns, depth, lda, ldb, ldc};
    state.last_types = {type_a, type_b, type_c};
    state.last_scalar_presence = (alpha ? 1U : 0U) | (beta ? 2U : 0U);
    if (state.is_metadata_case) {
        if (state.in_native) state.in_native();
        return state.native_status;
    }
    coefficients(alpha, beta);
    if (state.in_native) state.in_native();
    if (state.native_status != CUBLAS_STATUS_SUCCESS) return state.native_status;
    expect(type_a == CUDA_R_32F && type_b == CUDA_R_32F && type_c == CUDA_R_32F,
           "CPU arithmetic oracle only models real FP32; other types still reach exact SDK status");
    const float av = scalar(alpha), bv = scalar(beta);
    for (int column = 0; column < columns; ++column)
        for (int row = 0; row < rows; ++row) {
            float dot{};
            for (int inner = 0; inner < depth; ++inner) {
                const auto ai = ta == CUBLAS_OP_N ? inner * lda + row : row * lda + inner;
                const auto bi = tb == CUBLAS_OP_N ? column * ldb + inner : inner * ldb + column;
                dot += read_float(state.last_addresses[0] + sizeof(float) * ai) *
                       read_float(state.last_addresses[1] + sizeof(float) * bi);
            }
            const auto destination = state.last_addresses[2] + sizeof(float) * (column * ldc + row);
            write_float(destination, av * dot + bv * read_float(destination));
        }
    return state.native_status;
}
CudaSessionDispatch dispatch(unsigned int missing) {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* out, int ordinal) {
        *out = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* out, unsigned int, CUdevice) {
        expect(!state.context_live, "one native context per independent fixture");
        state.context_live = true;
        state.current = *out = context();
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext value) {
        state.current = value;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* out) {
        *out = state.current;
        return CUDA_SUCCESS;
    };
    sdk.context_destroy = [](CUcontext value) {
        expect(
            value == context() && state.context_live && state.events.empty() && !state.handle_live,
            "normal context destruction follows physical event and handle cleanup");
        state.allocations.clear();
        state.context_live = false;
        // Destroying this owned context also destroys its remaining streams.
        state.stream_live = false;
        state.current = nullptr;
        ++state.context_destroys;
        return CUDA_SUCCESS;
    };
    // General sessions require this dispatch surface, but the exact original
    // BLAS routes below must never prepare or substitute a private kernel.
    sdk.module_load = [](CUmodule*, const void*) -> CUresult {
        throw std::runtime_error("original BLAS extension cannot load a private module");
    };
    sdk.module_unload = [](CUmodule) -> CUresult {
        throw std::runtime_error("original BLAS extension owns no private module");
    };
    sdk.function_get = [](CUfunction*, CUmodule, const char*) -> CUresult {
        throw std::runtime_error("original BLAS extension cannot resolve a private kernel");
    };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) -> CUresult {
        throw std::runtime_error(
            "original BLAS extension cannot inspect private kernel parameters");
    };
    sdk.allocate = [](CUdeviceptr* out, std::size_t bytes) {
        std::vector<std::byte> allocation(bytes);
        *out = reinterpret_cast<CUdeviceptr>(allocation.data());
        expect(state.allocations.emplace(*out, std::move(allocation)).second,
               "distinct allocation");
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr address) {
        expect(state.allocations.erase(address) == 1, "release exact native allocation once");
        ++state.frees;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr to, const void* from, std::size_t bytes) {
        std::memcpy(memory(to, bytes), from, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download = [](void* to, CUdeviceptr from, std::size_t bytes) {
        ++state.downloads;
        std::memcpy(to, memory(from, bytes), bytes);
        return CUDA_SUCCESS;
    };
    sdk.execution.copy_device_async = [](CUdeviceptr to, CUdeviceptr from, std::size_t bytes,
                                         CUstream) {
        ++state.copies;
        std::memmove(memory(to, bytes), memory(from, bytes), bytes);
        return CUDA_SUCCESS;
    };
    sdk.stream_create = [](CUstream* out, unsigned int) {
        state.stream_live = true;
        *out = reinterpret_cast<CUstream>(&state.stream);
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream value) {
        expect(state.stream_live && value == reinterpret_cast<CUstream>(&state.stream),
               "exact stream");
        state.stream_live = false;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_create = [](CUevent* out, unsigned int) {
        auto token = std::make_unique<int>(1);
        *out = reinterpret_cast<CUevent>(token.get());
        state.events.emplace(*out, std::move(token));
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent event, CUstream) {
        expect(state.events.contains(event) && state.current == context(),
               "record actual owned event");
        ++state.event_records;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_query = [](CUevent event) {
        expect(state.events.contains(event), "query owned event");
        return state.can_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent event) {
        expect(state.events.erase(event) == 1, "retire one proven event");
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream) {
        ++state.synchronizes;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_wait_event = [](CUstream, CUevent event, unsigned int) {
        expect(state.events.contains(event), "only captured native event dependencies");
        return CUDA_SUCCESS;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) {
        throw std::runtime_error("new BLAS routes cannot substitute a Driver kernel");
        return CUDA_ERROR_UNKNOWN;
    };
    sdk.blas.create = [](cublasHandle_t* out) {
        expect(!state.handle_live, "one native BLAS handle");
        state.handle_live = true;
        *out = reinterpret_cast<cublasHandle_t>(&state.handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.destroy = [](cublasHandle_t handle) {
        native_owner(handle);
        state.handle_live = false;
        ++state.destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.set_stream = [](cublasHandle_t handle, cudaStream_t stream) {
        native_owner(handle);
        state.stream_value = stream;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.get_stream = [](cublasHandle_t handle, cudaStream_t* stream) {
        native_owner(handle);
        ++state.stream_queries;
        *stream = state.stream_value;
        return state.stream_status;
    };
    sdk.blas.set_pointer_mode = [](cublasHandle_t handle, cublasPointerMode_t mode) {
        native_owner(handle);
        ++state.mode_sets;
        state.mode = mode;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.get_pointer_mode = [](cublasHandle_t handle, cublasPointerMode_t* mode) {
        native_owner(handle);
        ++state.mode_queries;
        *mode = state.mode;
        return state.mode_status;
    };
    if (missing != 3)
        sdk.blas.get_version = [](cublasHandle_t handle, int* version) {
            native_owner(handle);
            ++state.version_queries;
            *version = state.version;
            if (state.in_version) state.in_version();
            return state.version_status;
        };
    sdk.blas.set_workspace = [](cublasHandle_t handle, void*, std::size_t) {
        native_owner(handle);
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                        const float*, const float*, int, const float*, int, const float*, float*,
                        int) {
        throw std::runtime_error("exact new API cannot be replaced by SGEMM");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    };
    sdk.blas.gemm_ex = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                          const void*, const void*, cudaDataType, int, const void*, cudaDataType,
                          int, const void*, void*, cudaDataType, int, cublasComputeType_t,
                          cublasGemmAlgo_t) {
        throw std::runtime_error("SgemmEx is not a GemmEx alias");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    };
    if (missing != 1) sdk.blas.sgemv = gemv;
    if (missing != 2) sdk.blas.sgemm_ex = gemm;
    return sdk;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

struct OwnerOptions {
    unsigned int missing{};
    std::size_t commands{64};
};
struct Owner {
    std::unique_ptr<CudaExecutionSession> session;
    std::array<CudaSessionResult, 5> allocations{};
    std::uint64_t handle{}, stream{}, next{1};
    explicit Owner(OwnerOptions options = {}) {
        state = {};
        CudaSessionLimits limits;
        limits.submissions.max_commands = options.commands;
        session = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{411, 412}, 413,
                                                         dispatch(options.missing), limits);
        success(session->initialize(0));
        const auto native = session->create_blas();
        success(native);
        handle = native.id;
        const auto created_stream = session->create_stream(CU_STREAM_NON_BLOCKING);
        success(created_stream);
        stream = created_stream.id;
        success(session->set_blas_stream(handle, stream));
        for (auto& allocation : allocations) {
            allocation = session->allocate(kBytes);
            success(allocation);
        }
    }
    template <typename Problem>
    Problem problem(bool device = false) {
        Problem result;
        result.handle = handle;
        result.presence = 31;
        result.rows = 3;
        result.columns = 2;
        result.leading_a = 8;
        if constexpr (std::is_same_v<Problem, CudaSessionSgemv>) {
            result.increment_x = result.increment_y = 1;
        } else {
            result.depth = 4;
            result.leading_b = result.leading_c = 8;
        }
        for (std::size_t index = 0; index < allocations.size(); ++index) {
            std::array<float, kBytes / sizeof(float)> values;
            for (std::size_t element = 0; element < values.size(); ++element)
                values[element] =
                    static_cast<float>(static_cast<int>((element + index * 3) % 11) - 5);
            success(session->upload({allocations[index].id, 0}, std::as_bytes(std::span{values})));
            if (index < 3 || device) result.buffers[index] = {allocations[index].id, kOffset};
        }
        const float alpha = 1.25F, beta = -0.5F;
        write_float(allocations[3].address + kOffset, alpha);
        write_float(allocations[4].address + kOffset, beta);
        if (device) {
            success(session->set_blas_attribute(handle, CudaBlasAttribute::kPointerMode,
                                                CUBLAS_POINTER_MODE_DEVICE));
            result.pointer_mode = CUBLAS_POINTER_MODE_DEVICE;
        } else {
            std::memcpy(result.alpha.data(), &alpha, sizeof(alpha));
            std::memcpy(result.beta.data(), &beta, sizeof(beta));
        }
        return result;
    }
    template <typename Problem>
    ExecutionAcceptance accept(const Problem& problem) {
        auto accepted = [&] {
            if constexpr (std::is_same_v<Problem, CudaSessionSgemv>)
                return session->accept_sgemv(next, problem);
            else
                return session->accept_sgemm_ex(next, problem);
        }();
        if (accepted.accepted()) ++next;
        return accepted;
    }
    void retire(const ExecutionAcceptance& accepted) {
        expect(session->acknowledge_result(accepted.receipt), "ACK the exact actual result");
        state.can_stop = true;
        expect(session->poll().retired == 1,
               "independent physical event retires exactly one record");
        state.can_stop = false;
    }
    void close() {
        success(session->destroy_blas(handle));
        success(session->destroy_stream(stream));
        success(session->shutdown());
        expect(!state.context_live && !state.handle_live && !state.stream_live &&
                   state.allocations.empty() && state.events.empty() &&
                   state.context_destroys == 1 && state.destroys == 1,
               "normal cleanup releases all native owners once");
    }
};
void issue(Owner& owner, const ExecutionAcceptance& accepted,
           cublasStatus_t expected = CUBLAS_STATUS_SUCCESS) {
    expect(accepted.accepted() && !owner.session->host_result(accepted.receipt),
           "acceptance does not invent an SDK result");
    const auto issued = owner.session->issue_next();
    const auto result = owner.session->host_result(accepted.receipt);
    expect(issued.state == CudaExecutionIssueState::kIssued &&
               issued.native_status == CUDA_SUCCESS && issued.library_status == expected &&
               result && result->origin == ExecutionReplyOrigin::kNativeSdk &&
               result->status == expected,
           "requested API result has exact native provenance, not replacement or converted status");
}
using Memory = std::map<CUdeviceptr, std::vector<std::byte>>;
template <typename Problem>
Memory oracle(const Owner& owner, const Problem& problem) {
    auto expected = state.allocations;
    const auto read = [&](std::size_t buffer, std::size_t element) {
        float value{};
        const auto& bytes = expected.at(owner.allocations[buffer].address);
        std::memcpy(&value, bytes.data() + kOffset + sizeof(float) * element, sizeof(value));
        return value;
    };
    const auto put = [&](std::size_t element, float value) {
        auto& bytes = expected.at(owner.allocations[2].address);
        std::memcpy(bytes.data() + kOffset + sizeof(float) * element, &value, sizeof(value));
    };
    float alpha{}, beta{};
    if (problem.pointer_mode) {
        alpha = read(3, 0);
        beta = read(4, 0);
    } else {
        std::memcpy(&alpha, problem.alpha.data(), sizeof(alpha));
        std::memcpy(&beta, problem.beta.data(), sizeof(beta));
    }
    if constexpr (std::is_same_v<Problem, CudaSessionSgemv>) {
        const auto index = [](int position, int length, int increment) {
            return static_cast<std::size_t>(increment > 0 ? position : length - position - 1) *
                   static_cast<std::size_t>(increment > 0 ? increment
                                                          : -static_cast<std::int64_t>(increment));
        };
        const bool transposed = problem.transpose != CUBLAS_OP_N;
        const int inputs = transposed ? problem.rows : problem.columns;
        const int outputs = transposed ? problem.columns : problem.rows;
        std::vector<float> values(static_cast<std::size_t>(outputs));
        for (int column = 0; column < problem.columns; ++column)
            for (int row = 0; row < problem.rows; ++row)
                values[transposed ? column : row] +=
                    read(0, static_cast<std::size_t>(column) * problem.leading_a + row) *
                    read(1, index(transposed ? row : column, inputs, problem.increment_x));
        for (int output = 0; output < outputs; ++output) {
            const auto at = index(output, outputs, problem.increment_y);
            put(at, alpha * values[output] + beta * read(2, at));
        }
    } else {
        for (int row = 0; row < problem.rows; ++row)
            for (int column = 0; column < problem.columns; ++column) {
                float sum{};
                for (int inner = 0; inner < problem.depth; ++inner) {
                    const auto ai = problem.transpose_a ? row * problem.leading_a + inner
                                                        : inner * problem.leading_a + row;
                    const auto bi = problem.transpose_b ? inner * problem.leading_b + column
                                                        : column * problem.leading_b + inner;
                    sum += read(0, static_cast<std::size_t>(ai)) *
                           read(1, static_cast<std::size_t>(bi));
                }
                const auto at = static_cast<std::size_t>(column) * problem.leading_c + row;
                put(at, alpha * sum + beta * read(2, at));
            }
    }
    return expected;
}
template <typename Problem>
void checked_numerical(Owner& owner, Problem problem) {
    const auto expected = oracle(owner, problem);
    const auto before_download = state.downloads, before_sync = state.synchronizes,
               before_set = state.mode_sets;
    const auto accepted = owner.accept(problem);
    problem.alpha.fill(std::byte{0xA5});
    problem.beta.fill(std::byte{0x5A});
    issue(owner, accepted);
    expect(state.allocations == expected,
           "all output, input, scalar, padding and guard bytes match independent oracle");
    expect(
        state.downloads == before_download && state.synchronizes == before_sync &&
            state.mode_sets == before_set,
        "fully owned HOST/DEVICE calls do not stage scalars, synchronize or change pointer mode");
    for (std::size_t index = 0; index < 3; ++index)
        expect(state.last_addresses[index] == owner.allocations[index].address + kOffset,
               "native physical base remains unchanged, including negative vector increments");
    owner.retire(accepted);
}
void numerical_routes() {
    for (bool device : {false, true}) {
        Owner owner;
        for (std::uint32_t transpose = 0; transpose <= 2; ++transpose)
            for (int increment_x : {-3, -1, 1, 2})
                for (int increment_y : {-2, -1, 1, 3}) {
                    auto problem = owner.problem<CudaSessionSgemv>(device);
                    problem.transpose = transpose;
                    problem.increment_x = increment_x;
                    problem.increment_y = increment_y;
                    checked_numerical(owner, problem);
                }
        for (std::uint32_t ta = 0; ta <= 2; ++ta)
            for (std::uint32_t tb = 0; tb <= 2; ++tb) {
                auto problem = owner.problem<CudaSessionSgemmEx>(device);
                problem.transpose_a = ta;
                problem.transpose_b = tb;
                checked_numerical(owner, problem);
            }
        expect(state.gemv_calls == 48 && state.gemm_calls == 9 && !state.version_queries,
               "every exact API executes once without substitutes");
        auto singleton = owner.problem<CudaSessionSgemv>(device);
        singleton.rows = singleton.columns = 1;
        singleton.increment_x = singleton.increment_y = std::numeric_limits<int>::min();
        checked_numerical(owner, singleton);
        expect(state.last_shape[4] == std::numeric_limits<int>::min() &&
                   state.last_shape[5] == std::numeric_limits<int>::min(),
               "INT_MIN singleton increments preserve exact signed ABI without overflowing hulls");
        owner.close();
    }
}
template <typename Problem>
void ownership_and_capacity() {
    Owner owner({.commands = 1});
    auto problem = owner.problem<Problem>(true);
    const auto accepted = owner.accept(problem);
    expect(accepted.accepted() && owner.accept(problem).status == ExecutionAcceptStatus::kFull &&
               owner.next == 2,
           "one bounded record, full rejection does not consume sequence or replay");
    issue(owner, accepted);
    expect(
        owner.session->acknowledge_result(accepted.receipt) && owner.session->poll().retained == 1,
        "host ACK cannot release independently held physical work");
    for (const auto& allocation : owner.allocations)
        expect(
            owner.session->release(allocation.id).disposition == CudaSessionDisposition::kDeferred,
            "all five physical owners remain retained after SDK success");
    expect(owner.session->destroy_blas(owner.handle).disposition ==
                   CudaSessionDisposition::kDeferred &&
               owner.session->set_blas_stream(owner.handle, 0).disposition ==
                   CudaSessionDisposition::kDeferred &&
               owner.session->set_blas_attribute(owner.handle, CudaBlasAttribute::kPointerMode, 0)
                       .disposition == CudaSessionDisposition::kDeferred,
           "pending execution excludes handle, stream and pointer-mode mutation");
    state.can_stop = true;
    expect(owner.session->poll().retired == 1, "exact physical proof recycles record capacity");
    state.can_stop = false;
    const auto cancelled = owner.accept(problem);
    expect(cancelled.accepted() && cancelled.receipt.sequence() == 2 &&
               owner.session->reject_queued(CUDA_ERROR_DEINITIALIZED) == 1,
           "known queued cancellation settles exactly the next accepted record");
    const auto result = owner.session->host_result(cancelled.receipt);
    expect(result && result->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               owner.session->acknowledge_result(cancelled.receipt) &&
               state.gemv_calls + state.gemm_calls == 1,
           "queued cancellation invokes no SDK and retains non-native provenance");
    owner.close();
}
template <typename Problem>
void invalid_inputs() {
    Owner owner;
    const auto valid = owner.problem<Problem>();
    const auto reject = [&](const auto& mutate) {
        auto problem = valid;
        mutate(problem);
        expect(!owner.accept(problem).accepted() && owner.next == 1 && !state.gemv_calls &&
                   !state.gemm_calls,
               "invalid ownership/canonical/shape cannot call SDK, return native status or consume "
               "sequence");
    };
    reject([](auto& p) { p.handle = UINT64_MAX; });
    reject([](auto& p) { p.presence = 30; });
    reject([](auto& p) { p.pointer_mode = 2; });
    reject([](auto& p) { p.alpha[4] = std::byte{1}; });
    reject([](auto& p) { p.buffers[0].allocation = UINT64_MAX; });
    reject([](auto& p) { p.buffers[1].offset = kBytes - 1; });
    reject([](auto& p) { p.buffers[2].offset = std::numeric_limits<std::size_t>::max(); });
    reject([](auto& p) { p.buffers[2] = p.buffers[0]; });
    reject([](auto& p) { p.buffers[2] = p.buffers[1]; });
    reject([](auto& p) { p.buffers[3] = p.buffers[0]; });
    if constexpr (std::is_same_v<Problem, CudaSessionSgemv>) {
        reject([](auto& p) { p.transpose = 4; });
        reject([](auto& p) { p.increment_y = std::numeric_limits<int>::min(); });
    } else {
        reject([](auto& p) { p.transpose_b = UINT32_MAX; });
        reject([](auto& p) { p.type_a = UINT32_MAX; });
        reject([](auto& p) { p.type_c = CUDA_R_4I; });
    }
    bool rejected_thread{};
    std::thread foreign([&] { rejected_thread = !owner.accept(valid).accepted(); });
    foreign.join();
    expect(rejected_thread && owner.next == 1,
           "foreign thread cannot acquire owned admission authority");
    const auto accepted = owner.accept(valid);
    issue(owner, accepted);
    // This one accepted record still requires normal host ACK and physical proof.
    const auto frontier = owner.session->capture_command_frontier(1);
    expect(frontier.status == CUDA_SUCCESS, "capture accepted exact prefix");
    state.can_stop = true;
    expect(owner.session->poll().retired == 1, "physical retirement independent of host receipt");
    expect(owner.session->acknowledge_result(accepted.receipt),
           "ACK the independently retired exact receipt");
    owner.close();
}
template <typename Problem>
void scalar_producer() {
    Owner owner;
    auto problem = owner.problem<Problem>(true);
    const auto producer =
        owner.session->accept_memory(owner.next, {.kind = CudaMemoryKind::kCopy,
                                                  .destination = problem.buffers[3],
                                                  .source = problem.buffers[4],
                                                  .count = sizeof(float),
                                                  .stream = owner.stream,
                                                  .is_asynchronous = true});
    expect(producer.accepted(), "device scalar producer accepted first");
    ++owner.next;
    const auto accepted = owner.accept(problem);
    expect(accepted.accepted() && state.copies == 0 && state.downloads == 0,
           "no early scalar snapshot");
    expect(
        owner.session->issue_next().state == CudaExecutionIssueState::kIssued && state.copies == 1,
        "actual same-stream producer executes before scalar-consuming library");
    const auto expected = oracle(owner, problem);
    issue(owner, accepted);
    expect(state.allocations == expected && state.last_alpha == state.last_beta &&
               !state.downloads && !state.synchronizes,
           "original DEVICE route reads ordered updated scalar without staging");
    expect(owner.session->acknowledge_result(producer.receipt) &&
               owner.session->acknowledge_result(accepted.receipt),
           "ACK both exact results");
    state.can_stop = true;
    expect(owner.session->poll().retired == 2, "both physical records retire");
    owner.close();
}
template <typename Problem>
void overlap_and_native_error() {
    Owner owner;
    const auto problem = owner.problem<Problem>(true);
    for (std::size_t index : {std::size_t{3}, std::size_t{4}}) {
        auto overlapping = problem;
        overlapping.buffers[index] = problem.buffers[2];
        expect(!owner.accept(overlapping).accepted(),
               "device scalars cannot overlap native output writes");
    }
    auto overlapping = problem;
    overlapping.buffers[2] = problem.buffers[0];
    overlapping.buffers[2].offset += sizeof(float);
    expect(!owner.accept(overlapping).accepted(),
           "partial input/output overlap is not an in-place permission");
    for (const auto& range : problem.buffers) {
        success(owner.session->set_blas_workspace(owner.handle, range, sizeof(float)));
        expect(!owner.accept(problem).accepted(),
               "native scratch workspace cannot alias any active matrix/vector/scalar");
    }
    success(owner.session->set_blas_workspace(owner.handle, {}, 0));
    expect(owner.next == 1 && !state.gemv_calls && !state.gemm_calls,
           "all overlap controls reject before native compute or sequence consumption");
    state.native_status = CUBLAS_STATUS_EXECUTION_FAILED;
    auto expected = state.allocations;
    const float marker = 123.0F;
    std::memcpy(expected.at(owner.allocations[2].address).data() + kOffset, &marker,
                sizeof(marker));
    state.in_native = [&] {
        write_float(owner.allocations[2].address + kOffset, marker);
        expect(
            !owner.accept(problem).accepted() &&
                owner.session->set_blas_attribute(owner.handle, CudaBlasAttribute::kPointerMode, 0)
                        .disposition == CudaSessionDisposition::kDeferred,
            "native callback cannot reenter admission or mutate the active handle");
    };
    const auto accepted = owner.accept(problem);
    issue(owner, accepted, CUBLAS_STATUS_EXECUTION_FAILED);
    expect(state.allocations == expected && owner.session->poll().retained == 1 &&
               !owner.session->snapshot().is_quarantined,
           "native error may have effects but is not ownership uncertainty or physical retirement");
    owner.retire(accepted);
    owner.close();
}
void typed_status() {
    for (const auto type :
         {CUDA_R_16F, CUDA_R_16BF, CUDA_R_8I, CUDA_R_64F, CUDA_C_32F, CUDA_C_64F}) {
        Owner owner;
        auto problem = owner.problem<CudaSessionSgemmEx>();
        problem.type_a = problem.type_b = problem.type_c = static_cast<std::uint32_t>(type);
        problem.rows = problem.columns = problem.depth = 1;
        state.native_status = CUBLAS_STATUS_NOT_SUPPORTED;
        const auto accepted = owner.accept(problem);
        issue(owner, accepted, state.native_status);
        expect(state.last_types == std::array{type, type, type} &&
                   state.last_alpha == std::bit_cast<std::array<std::byte, 4>>(1.25F),
               "exact raw types reach SgemmEx while alpha remains exactly FP32, even for rejected "
               "types");
        owner.retire(accepted);
        owner.close();
    }
    for (unsigned int missing = 1; missing <= 2; ++missing) {
        Owner owner({.missing = missing});
        if (missing == 1)
            expect(!owner.accept(owner.problem<CudaSessionSgemv>()).accepted(),
                   "missing SGEMV cannot use SGEMM");
        else
            expect(!owner.accept(owner.problem<CudaSessionSgemmEx>()).accepted(),
                   "missing SgemmEx cannot use GemmEx");
        owner.close();
    }
}
template <typename Problem>
void candidate_metadata_routes() {
    // CPU contract tests only: native SDK access qualification is independent.
    for (const bool device : {false, true}) {
        Owner owner;
        const auto base = owner.problem<Problem>(device);
        std::vector<Problem> cases;
        const auto add = [&](const auto& mutate) {
            auto value = base;
            mutate(value);
            cases.push_back(value);
        };
        add([](auto& value) { value.rows = 0; });
        add([](auto& value) { value.columns = 0; });
        add([](auto& value) { value.rows = -1; });
        add([](auto& value) { value.columns = -1; });
        if constexpr (!std::is_same_v<Problem, CudaSessionSgemv>) {
            add([](auto& value) { value.leading_a = 1; });
            add([](auto& value) { value.depth = -1; });
            add([](auto& value) { value.transpose_a = CUBLAS_OP_CONJG; });
            add([](auto& value) { value.transpose_b = CUBLAS_OP_CONJG; });
            add([](auto& value) { value.leading_b = 1; });
            add([](auto& value) { value.leading_c = 1; });
        }
        state.is_metadata_case = true;
        constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
        const auto before = state.allocations;
        for (auto problem : cases)
            for (std::uint32_t presence = 0; presence != 32; ++presence) {
                problem.presence = presence;
                problem.alpha = problem.beta = {};
                problem.buffers = {};
                std::array<CUdeviceptr, 3> expected_data{};
                for (std::size_t index = 0; index != bits.size(); ++index)
                    if ((presence & bits[index]) && (index < 3 || device)) {
                        problem.buffers[index] = base.buffers[index];
                        if (index < 3)
                            expected_data[index] = owner.allocations[index].address + kOffset;
                    }
                state.native_status = problem.rows == 0 || problem.columns == 0 ||
                                              (std::is_same_v<Problem, CudaSessionSgemv> &&
                                               (problem.rows < 0 || problem.columns < 0))
                                          ? CUBLAS_STATUS_SUCCESS
                                          : CUBLAS_STATUS_INVALID_VALUE;
                const auto versions = state.version_queries;
                const auto next = owner.next;
                const auto gemv_calls = state.gemv_calls, gemm_calls = state.gemm_calls;
                const auto accepted = owner.accept(problem);
                if (presence != 0 && presence != 3 && presence != 28 && presence != 31) {
                    expect(!accepted.accepted() && owner.next == next &&
                               state.version_queries == versions &&
                               state.gemv_calls == gemv_calls && state.gemm_calls == gemm_calls &&
                               state.allocations == before,
                           "unqualified mixed presence rejects before admission or original SDK "
                           "provenance");
                    continue;
                }
                issue(owner, accepted, state.native_status);
                expect(state.version_queries == versions + 1 &&
                           state.last_scalar_presence == (presence & 3U) &&
                           std::equal(expected_data.begin(), expected_data.end(),
                                      state.last_addresses.begin()) &&
                           state.allocations == before && !state.downloads && !state.synchronizes &&
                           state.mode ==
                               (device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST),
                       "exact boundary callback preserves scalar presence, owned/null bases "
                       "and mode");
                expect(owner.session->poll().retained == 1 &&
                           owner.session->release(owner.allocations[2].id).disposition ==
                               CudaSessionDisposition::kDeferred &&
                           owner.session->destroy_blas(owner.handle).disposition ==
                               CudaSessionDisposition::kDeferred,
                       "even an original empty/error result retains its actual physical tail");
                owner.retire(accepted);
            }
        owner.close();
    }
}
void rejected_positive_sgemv_geometry() {
    for (const bool device : {false, true}) {
        Owner owner;
        const auto base = owner.problem<CudaSessionSgemv>(device);
        for (unsigned int invalid = 0; invalid != 4; ++invalid)
            for (const bool has_pointers : {false, true}) {
                auto problem = base;
                if (invalid == 0) problem.transpose = CUBLAS_OP_CONJG;
                if (invalid == 1) problem.leading_a = 1;
                if (invalid == 2) problem.increment_x = 0;
                if (invalid == 3) problem.increment_y = 0;
                if (!has_pointers) {
                    problem.presence = 0;
                    problem.alpha = problem.beta = {};
                    problem.buffers = {};
                }
                expect(!owner.accept(problem).accepted(),
                       "unsafe positive SGEMV geometry has no metadata-only permission");
            }
        expect(owner.next == 1 && !state.version_queries && !state.gemv_calls && !state.gemm_calls,
               "unsafe SGEMV rejects without sequence consumption or native provenance");
        owner.close();
    }
}
template <typename Problem>
void candidate_optional_ownership() {
    Owner owner;
    auto base = owner.problem<Problem>(true);
    base.rows = 0;
    state.is_metadata_case = true;
    constexpr std::array<std::uint32_t, 5> bits{4, 8, 16, 1, 2};
    for (std::size_t index = 0; index != base.buffers.size(); ++index) {
        auto problem = base;
        problem.buffers[index].allocation = UINT64_MAX;
        expect(!owner.accept(problem).accepted(),
               "optional stale/foreign allocation is never NULL");
        problem = base;
        problem.buffers[index].offset = kBytes + 1;
        expect(!owner.accept(problem).accepted(), "optional owned offset remains bounded");
        problem = base;
        problem.buffers[index] = {};
        expect(!owner.accept(problem).accepted(),
               "present optional pointer cannot normalize unresolved ownership into NULL");
        problem = base;
        problem.buffers[index] = {0, 1};
        expect(!owner.accept(problem).accepted(), "unresolved address has no offset authority");
        problem = base;
        problem.presence &= ~bits[index];
        expect(!owner.accept(problem).accepted(), "absent pointer cannot carry an allocation");
    }
    expect(owner.next == 1 && !state.version_queries && !state.gemv_calls && !state.gemm_calls,
           "invalid optional ownership rejects before sequence or original callback");
    const auto accepted = owner.accept(base);
    issue(owner, accepted);
    owner.retire(accepted);
    for (auto& range : base.buffers) range.offset = kBytes;
    const auto endpoint = owner.accept(base);
    issue(owner, endpoint);
    for (std::size_t index = 0; index != 3; ++index)
        expect(state.last_addresses[index] == owner.allocations[index].address + kBytes,
               "known optional endpoint remains the exact owned non-NULL address");
    owner.retire(endpoint);
    owner.close();
}
void candidate_zero_depth() {
    for (const bool device : {false, true}) {
        Owner owner;
        for (std::uint32_t ta = 0; ta != 3; ++ta)
            for (std::uint32_t tb = 0; tb != 3; ++tb)
                for (const auto presence : {19U, 31U}) {
                    auto problem = owner.problem<CudaSessionSgemmEx>(device);
                    const auto owned_buffers = problem.buffers;
                    problem.transpose_a = ta;
                    problem.transpose_b = tb;
                    problem.depth = 0;
                    problem.presence = presence;
                    if (presence == 19) problem.buffers[0] = problem.buffers[1] = {};
                    const auto next = owner.next;
                    const auto gemm_calls = state.gemm_calls, versions = state.version_queries;
                    auto invalid = problem;
                    invalid.buffers[2] = {};
                    expect(!owner.accept(invalid).accepted(),
                           "zero depth still requires full owned C");
                    invalid = problem;
                    invalid.buffers[0] = {UINT64_MAX, 0};
                    invalid.presence |= 4;
                    expect(!owner.accept(invalid).accepted(),
                           "zero depth never ignores a stale A identity");
                    for (const auto bit : {4U, 8U}) {
                        invalid = problem;
                        invalid.presence = 31;
                        invalid.buffers = owned_buffers;
                        invalid.buffers[bit == 4 ? 0 : 1] = {};
                        expect(!owner.accept(invalid).accepted(),
                               "zero depth cannot normalize a present unresolved A or B to NULL");
                        invalid = problem;
                        invalid.presence = 19U | bit;
                        invalid.buffers = owned_buffers;
                        invalid.buffers[bit == 4 ? 1 : 0] = {};
                        expect(!owner.accept(invalid).accepted(),
                               "zero-depth mixed A/B presence 23/27 has no access qualification");
                    }
                    invalid = problem;
                    invalid.presence &= ~1U;
                    invalid.alpha = {};
                    invalid.buffers[3] = {};
                    expect(!owner.accept(invalid).accepted(),
                           "zero depth still requires a real alpha scalar");
                    expect(owner.next == next && state.gemm_calls == gemm_calls &&
                               state.version_queries == versions,
                           "zero-depth ownership and presence rejection has no native provenance");
                    const auto expected = oracle(owner, problem);
                    const auto accepted = owner.accept(problem);
                    issue(owner, accepted);
                    expect(
                        state.allocations == expected &&
                            state.last_addresses[0] ==
                                (presence == 19 ? 0 : owner.allocations[0].address + kOffset) &&
                            state.last_addresses[1] ==
                                (presence == 19 ? 0 : owner.allocations[1].address + kOffset) &&
                            state.version_queries == versions + 1 && !state.downloads &&
                            !state.synchronizes &&
                            owner.session->release(owner.allocations[2].id).disposition ==
                                CudaSessionDisposition::kDeferred,
                        "zero-depth exact original scales all C with beta and retains C until real "
                        "stop");
                    owner.retire(accepted);
                }
        owner.close();
    }
}
template <typename Problem>
void candidate_version_controls() {
    {
        Owner owner({.missing = 3});
        auto problem = owner.problem<Problem>();
        problem.rows = 0;
        expect(!owner.accept(problem).accepted() && owner.next == 1,
               "missing version callback cannot grant boundary permission");
        owner.close();
    }
    Owner owner;
    auto problem = owner.problem<Problem>();
    problem.rows = 0;
    state.version = 120603;
    const auto accepted = owner.accept(problem);
    expect(accepted.accepted(), "version is checked at ordered issue, not CPU acceptance");
    const auto issued = owner.session->issue_next();
    const auto result = owner.session->host_result(accepted.receipt);
    expect(issued.state == CudaExecutionIssueState::kRejected && !issued.library_status && result &&
               result->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
               result->status == CUDA_ERROR_NOT_SUPPORTED && !state.gemv_calls &&
               !state.gemm_calls && state.version_queries == 1 &&
               !owner.session->snapshot().is_quarantined,
           "wrong version cleanly rejects before original compute, without invented native status");
    owner.retire(accepted);
    problem = owner.problem<Problem>();
    const auto healthy = owner.accept(problem);
    issue(owner, healthy);
    expect(state.version_queries == 1, "full owned path needs no boundary version permission");
    owner.retire(healthy);
    owner.close();
}
template <typename Problem>
void known_preissue_query_failures() {
    for (unsigned int query = 0; query != 3; ++query) {
        Owner owner;
        auto problem = owner.problem<Problem>();
        if (query == 2) problem.rows = 0;
        const auto before = state.allocations;
        const auto accepted = owner.accept(problem);
        expect(accepted.accepted(), "admit before read-only metadata failure");
        const auto captured = owner.session->capture_command_frontier(accepted.receipt.sequence());
        expect(captured.status == CUDA_SUCCESS, "capture exact rejected command lifetime");
        (query == 2   ? state.version_status
         : query == 1 ? state.mode_status
                      : state.stream_status) = CUBLAS_STATUS_INTERNAL_ERROR;
        const auto issued = owner.session->issue_next();
        const auto result = owner.session->host_result(accepted.receipt);
        expect(issued.state == CudaExecutionIssueState::kRejected &&
                   issued.native_status == CUDA_ERROR_UNKNOWN &&
                   issued.completion_status == CUDA_ERROR_UNKNOWN && !issued.library_status &&
                   result && result->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                   result->status == CUDA_ERROR_UNKNOWN && state.event_records == 1 &&
                   state.events.size() == 1 && !state.gemv_calls && !state.gemm_calls &&
                   state.allocations == before && owner.session->snapshot().is_quarantined,
               "proven-owner metadata error invents no original compute result or physical stop");
        expect(owner.session->acknowledge_result(accepted.receipt),
               "ACK the exact pre-compute failure, not its physical tail");
        state.stream_status = state.mode_status = state.version_status = CUBLAS_STATUS_SUCCESS;
        const auto held = owner.session->poll();
        expect(!held.retired && held.retained == 1 && !captured.frontier.retirement().is_retired &&
                   !owner.session->can_cleanup_quarantined() && !owner.accept(problem).accepted() &&
                   owner.session->snapshot().is_quarantined,
               "healthy later metadata neither clears quarantine nor retires the held event");
        expect(owner.session->shutdown().disposition == CudaSessionDisposition::kDeferred &&
                   state.allocations == before && !state.frees && !state.destroys &&
                   !state.context_destroys && state.context_live && state.handle_live &&
                   state.stream_live && state.events.size() == 1,
               "all resources survive attempted shutdown before independent event completion");
        state.can_stop = true;
        const auto retired = owner.session->poll();
        expect(retired.retired == 1 && !retired.retained &&
                   captured.frontier.retirement().is_retired &&
                   owner.session->query_frontier(captured.frontier) == CUDA_ERROR_UNKNOWN &&
                   owner.session->snapshot().is_quarantined &&
                   owner.session->can_cleanup_quarantined() && state.allocations == before &&
                   !state.frees && !state.destroys && !state.context_destroys,
               "real tail retirement enables cleanup without erasing the failure or quarantine");
        success(owner.session->shutdown());
        expect(
            !state.context_live && !state.handle_live && !state.stream_live &&
                state.allocations.empty() && state.events.empty() && state.context_destroys == 1 &&
                state.destroys == 1 && !state.gemv_calls && !state.gemm_calls &&
                !state.synchronizes,
            "known owner closes exactly once after proof without replay or fake synchronization");
    }
}
template <typename Problem>
void retained_failure(unsigned int which) {
    Owner owner;
    auto problem = owner.problem<Problem>();
    const bool is_version_failure = which >= 8;
    if (is_version_failure) problem.rows = 0;
    const auto accepted = owner.accept(problem);
    expect(accepted.accepted(), "admit before injected uncertainty");
    const bool after = !is_version_failure && (which == 2 || which == 3 || which >= 6);
    if (is_version_failure) {
        state.in_version = [which] {
            if (which == 8)
                state.current = nullptr;
            else
                state.mode = CUBLAS_POINTER_MODE_DEVICE;
        };
    } else if (after) {
        state.native_status = CUBLAS_STATUS_EXECUTION_FAILED;
        state.in_native = [which] {
            if (which == 2)
                state.current = nullptr;
            else if (which == 3)
                state.mode = CUBLAS_POINTER_MODE_DEVICE;
            else if (which == 6)
                state.stream_status = CUBLAS_STATUS_INTERNAL_ERROR;
            else
                state.mode_status = CUBLAS_STATUS_INTERNAL_ERROR;
        };
    } else if (which == 0)
        state.mode = CUBLAS_POINTER_MODE_DEVICE;
    else if (which == 1)
        state.stream_value = nullptr;
    else if (which == 4)
        state.stream_status = CUBLAS_STATUS_INTERNAL_ERROR;
    else
        state.mode_status = CUBLAS_STATUS_INTERNAL_ERROR;
    const auto issued = owner.session->issue_next();
    const auto result = owner.session->host_result(accepted.receipt);
    expect(result && (after ? issued.library_status == CUBLAS_STATUS_EXECUTION_FAILED &&
                                  result->origin == ExecutionReplyOrigin::kNativeSdk &&
                                  result->status == CUBLAS_STATUS_EXECUTION_FAILED
                            : !issued.library_status &&
                                  result->origin == ExecutionReplyOrigin::kRejectedBeforeIssue),
           "ownership proof failure cannot invent or overwrite the actual SDK result");
    expect(owner.session->acknowledge_result(accepted.receipt),
           "ACK exact failure without releasing ownership");
    state.current = context();
    state.can_stop = true;
    static_cast<void>(owner.session->poll());
    const auto stopped = owner.session->shutdown();
    expect(
        state.allocations.size() == 5 && state.context_destroys == 0 && state.destroys == 0 &&
            !state.frees && stopped.status != CUDA_SUCCESS &&
            owner.session->snapshot().is_quarantined && !owner.session->can_cleanup_quarantined() &&
            state.gemv_calls + state.gemm_calls == (after ? 1U : 0U),
        "uncertain owner is retained, not cleaned or replayed after a later healthy-looking event");
}
void isolated_failures() {
    for (unsigned int which = 0; which < 20; ++which) {
        // Read-only query errors before computation have known ownership and
        // are covered with normal cleanup above, not intentional child leaks.
        if (which % 10 == 4 || which % 10 == 5) continue;
        const auto child = ::fork();
        expect(child >= 0, "fork isolated intentional retention test");
        if (!child) {
            try {
                if (which < 10)
                    retained_failure<CudaSessionSgemv>(which);
                else
                    retained_failure<CudaSessionSgemmEx>(which - 10);
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "BLAS isolated case " << which << ": " << error.what() << '\n';
                ::_exit(1);
            }
        }
        int result{};
        expect(
            ::waitpid(child, &result, 0) == child && WIFEXITED(result) && WEXITSTATUS(result) == 0,
            "exact retained-uncertainty child succeeds independently");
    }
}
}  // namespace

int main() {
    try {
        numerical_routes();
        ownership_and_capacity<CudaSessionSgemv>();
        ownership_and_capacity<CudaSessionSgemmEx>();
        invalid_inputs<CudaSessionSgemv>();
        invalid_inputs<CudaSessionSgemmEx>();
        scalar_producer<CudaSessionSgemv>();
        scalar_producer<CudaSessionSgemmEx>();
        overlap_and_native_error<CudaSessionSgemv>();
        overlap_and_native_error<CudaSessionSgemmEx>();
        typed_status();
        candidate_metadata_routes<CudaSessionSgemv>();
        candidate_metadata_routes<CudaSessionSgemmEx>();
        rejected_positive_sgemv_geometry();
        candidate_optional_ownership<CudaSessionSgemv>();
        candidate_optional_ownership<CudaSessionSgemmEx>();
        candidate_zero_depth();
        candidate_version_controls<CudaSessionSgemv>();
        candidate_version_controls<CudaSessionSgemmEx>();
        known_preissue_query_failures<CudaSessionSgemv>();
        known_preissue_query_failures<CudaSessionSgemmEx>();
        isolated_failures();
        std::cout << "Exact owned BLAS extension checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
