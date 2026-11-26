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
#include <type_traits>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::backend;
using namespace glimmer::backend::cuda;
void expect(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void success(const CudaSessionResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete && result.status == CUDA_SUCCESS,
           "owned Driver operation succeeds");
}
void success(const CudaSessionBlasResult& result) {
    expect(result.disposition == CudaSessionDisposition::kComplete &&
               result.status == CUDA_SUCCESS && result.original_status == CUBLAS_STATUS_SUCCESS,
           "owned original BLAS operation succeeds");
}
enum class Phase { kNone, kContext, kVersion, kStream, kMode, kSync, kDownload, kSetMode, kGeam };
struct Invocation {
    cudaDataType type{};
    std::array<int, 7> shape{};
    std::array<const void*, 5> pointers{};
    cublasPointerMode_t mode{};
    std::array<std::byte, 8> alpha{}, beta{};
};
struct State {
    char context{}, handle{}, stream{}, module{};
    CUcontext current{};
    cudaStream_t blas_stream{};
    cublasPointerMode_t pointer_mode{CUBLAS_POINTER_MODE_HOST};
    std::map<CUdeviceptr, std::vector<std::byte>> memory;
    std::map<CUevent, std::unique_ptr<int>> events;
    std::array<std::byte, 768> global{};
    bool context_live{}, handle_live{}, stream_live{}, module_live{}, can_stop{};
    bool arithmetic{true}, partial_write{}, fail_restore{};
    unsigned int contexts{}, context_destroys{}, creates{}, destroys{}, frees{}, unloads{};
    unsigned int calls{}, event_creates{}, records{}, queries{}, event_destroys{}, synchronizes{};
    unsigned int downloads{}, copies{}, versions{}, mode_gets{}, stream_gets{}, mode_sets{},
        probes{};
    int version{120604};
    CUresult sync_status{CUDA_SUCCESS}, download_status{CUDA_SUCCESS}, record_status{CUDA_SUCCESS};
    cublasStatus_t native_status{CUBLAS_STATUS_SUCCESS}, query_status{CUBLAS_STATUS_SUCCESS};
    Invocation last;
    Phase armed{Phase::kNone};
    std::function<void()> probe;
} state;
void touch(Phase phase) {
    if (state.armed != phase || phase == Phase::kNone) return;
    state.armed = Phase::kNone;
    ++state.probes;
    state.probe();
}
CUcontext context() {
    return reinterpret_cast<CUcontext>(&state.context);
}
void native_owner(cublasHandle_t handle) {
    expect(state.context_live && state.current == context() && state.handle_live &&
               handle == reinterpret_cast<cublasHandle_t>(&state.handle),
           "only the actual context-owned fake SDK handle is consumed");
}
std::byte* memory(CUdeviceptr pointer, std::size_t bytes) {
    for (auto& [base, data] : state.memory)
        if (pointer >= base && pointer - base <= data.size() &&
            bytes <= data.size() - static_cast<std::size_t>(pointer - base))
            return data.data() + (pointer - base);
    const auto base = reinterpret_cast<CUdeviceptr>(state.global.data());
    if (state.module_live && pointer >= base && pointer - base <= state.global.size() &&
        bytes <= state.global.size() - static_cast<std::size_t>(pointer - base))
        return state.global.data() + (pointer - base);
    throw std::runtime_error("fake SDK never reads outside independently owned storage");
}
template <typename T>
T read_scalar(const T* pointer, cublasPointerMode_t invocation_mode) {
    T value{};
    if (pointer) {
        const auto* source = invocation_mode == CUBLAS_POINTER_MODE_HOST
                                 ? reinterpret_cast<const std::byte*>(pointer)
                                 : memory(reinterpret_cast<CUdeviceptr>(pointer), sizeof(T));
        std::memcpy(&value, source, sizeof(value));
    }
    return value;
}
// CPU mathematical fake only; no vendor implementation or GPU qualification.
// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact original GEAM ABI.
template <typename T>
cublasStatus_t invoke(cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int rows,
                      int columns, const T* alpha, const T* a, int lda, const T* beta, const T* b,
                      int ldb, T* c, int ldc) {
    native_owner(handle);
    ++state.calls;
    state.last = {std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F,
                  {static_cast<int>(ta), static_cast<int>(tb), rows, columns, lda, ldb, ldc},
                  {a, b, c, alpha, beta},
                  state.pointer_mode};
    touch(Phase::kGeam);
    if (state.current != context()) return state.native_status;
    if (state.native_status != CUBLAS_STATUS_SUCCESS) {
        if (state.partial_write && c) {
            const T marker = 123;
            std::memcpy(memory(reinterpret_cast<CUdeviceptr>(c), sizeof(T)), &marker, sizeof(T));
        }
        return state.native_status;
    }
    if (static_cast<unsigned int>(ta) > 2 || static_cast<unsigned int>(tb) > 2 || rows < 0 ||
        columns < 0 || lda < std::max(1, ta == CUBLAS_OP_N ? rows : columns) ||
        ldb < std::max(1, tb == CUBLAS_OP_N ? rows : columns) || ldc < std::max(1, rows))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (!rows || !columns) return CUBLAS_STATUS_SUCCESS;
    if (state.last.mode == CUBLAS_POINTER_MODE_DEVICE && (!alpha || !beta))
        return CUBLAS_STATUS_INVALID_VALUE;
    if ((c == a && (ta != CUBLAS_OP_N || lda != ldc)) ||
        (c == b && (tb != CUBLAS_OP_N || ldb != ldc)))
        return CUBLAS_STATUS_INVALID_VALUE;
    // The injected post-call state mutation must not reinterpret arguments of
    // the operation already entered with its original pointer mode.
    const auto av = read_scalar(alpha, state.last.mode), bv = read_scalar(beta, state.last.mode);
    std::memcpy(state.last.alpha.data(), &av, sizeof(T));
    std::memcpy(state.last.beta.data(), &bv, sizeof(T));
    if (!state.arithmetic) return CUBLAS_STATUS_SUCCESS;
    expect(c && (av == T{} || a) && (bv == T{} || b), "active native inputs have safe storage");
    std::vector<T> results;
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            T x{}, y{};
            if (av != T{}) {
                const auto offset = ta == CUBLAS_OP_N ? column * lda + row : row * lda + column;
                std::memcpy(
                    &x, memory(reinterpret_cast<CUdeviceptr>(a) + sizeof(T) * offset, sizeof(T)),
                    sizeof(T));
            }
            if (bv != T{}) {
                const auto offset = tb == CUBLAS_OP_N ? column * ldb + row : row * ldb + column;
                std::memcpy(
                    &y, memory(reinterpret_cast<CUdeviceptr>(b) + sizeof(T) * offset, sizeof(T)),
                    sizeof(T));
            }
            results.push_back(av * x + bv * y);
        }
    }
    std::size_t index{};
    for (int column = 0; column < columns; ++column)
        for (int row = 0; row < rows; ++row)
            std::memcpy(memory(reinterpret_cast<CUdeviceptr>(c) + sizeof(T) * (column * ldc + row),
                               sizeof(T)),
                        &results[index++], sizeof(T));
    return CUBLAS_STATUS_SUCCESS;
}

CudaSessionDispatch dispatch(unsigned int missing = 0) {
    CudaSessionDispatch sdk;
    sdk.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    sdk.device_get = [](CUdevice* out, int ordinal) {
        *out = ordinal;
        return CUDA_SUCCESS;
    };
    sdk.context_create = [](CUcontext* out, unsigned int, CUdevice) {
        expect(!state.context_live, "one independent fake owner");
        state.context_live = true;
        state.current = *out = context();
        ++state.contexts;
        return CUDA_SUCCESS;
    };
    sdk.context_set_current = [](CUcontext value) {
        state.current = value;
        return CUDA_SUCCESS;
    };
    sdk.context_destroy = [](CUcontext value) {
        expect(value == context() && state.current == value && state.context_live,
               "known current context destruction once");
        state.context_live = state.module_live = state.stream_live = false;
        state.memory.clear();
        state.current = nullptr;
        ++state.context_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.context_get_current = [](CUcontext* out) {
        touch(Phase::kContext);
        *out = state.current;
        return CUDA_SUCCESS;
    };
    sdk.module_load = [](CUmodule* out, const void*) {
        state.module_live = true;
        *out = reinterpret_cast<CUmodule>(&state.module);
        return CUDA_SUCCESS;
    };
    sdk.module_unload = [](CUmodule value) {
        expect(state.module_live && value == reinterpret_cast<CUmodule>(&state.module),
               "module parent released once");
        state.module_live = false;
        ++state.unloads;
        return CUDA_SUCCESS;
    };
    sdk.module_global = [](CUdeviceptr* out, std::size_t* bytes, CUmodule, const char*) {
        *out = reinterpret_cast<CUdeviceptr>(state.global.data());
        *bytes = state.global.size();
        return CUDA_SUCCESS;
    };
    sdk.function_get = [](CUfunction* out, CUmodule, const char*) {
        *out = reinterpret_cast<CUfunction>(&state.module);
        return CUDA_SUCCESS;
    };
    sdk.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        return CUDA_ERROR_INVALID_VALUE;
    };
    sdk.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                     unsigned int, unsigned int, unsigned int, unsigned int,
                                     CUstream, void**, void**) {
        throw std::runtime_error("GEAM must never substitute a Driver kernel");
        return CUDA_ERROR_UNKNOWN;
    };
    sdk.allocate = [](CUdeviceptr* out, std::size_t bytes) {
        std::vector<std::byte> data(bytes);
        *out = reinterpret_cast<CUdeviceptr>(data.data());
        expect(state.memory.emplace(*out, std::move(data)).second, "distinct native allocation");
        return CUDA_SUCCESS;
    };
    sdk.release = [](CUdeviceptr value) {
        expect(state.memory.erase(value) == 1, "only owned native memory can be freed");
        ++state.frees;
        return CUDA_SUCCESS;
    };
    sdk.upload = [](CUdeviceptr to, const void* from, std::size_t bytes) {
        std::memcpy(memory(to, bytes), from, bytes);
        return CUDA_SUCCESS;
    };
    sdk.download = [](void* to, CUdeviceptr from, std::size_t bytes) {
        ++state.downloads;
        touch(Phase::kDownload);
        if (state.download_status == CUDA_SUCCESS && state.current == context())
            std::memcpy(to, memory(from, bytes), bytes);
        return state.download_status;
    };
    sdk.execution.copy_device_async = [](CUdeviceptr to, CUdeviceptr from, std::size_t bytes,
                                         CUstream) {
        std::memmove(memory(to, bytes), memory(from, bytes), bytes);
        ++state.copies;
        return CUDA_SUCCESS;
    };
    sdk.stream_create = [](CUstream* out, unsigned int) {
        state.stream_live = true;
        *out = reinterpret_cast<CUstream>(&state.stream);
        return CUDA_SUCCESS;
    };
    sdk.stream_destroy = [](CUstream value) {
        expect(state.stream_live && value == reinterpret_cast<CUstream>(&state.stream),
               "known original stream cleanup");
        state.stream_live = false;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_create = [](CUevent* out, unsigned int) {
        auto token = std::make_unique<int>(1);
        *out = reinterpret_cast<CUevent>(token.get());
        state.events.emplace(*out, std::move(token));
        ++state.event_creates;
        return CUDA_SUCCESS;
    };
    sdk.execution.event_record = [](CUevent event, CUstream) {
        expect(state.events.contains(event) && state.current == context(), "record owned event");
        ++state.records;
        return state.record_status;
    };
    sdk.execution.event_query = [](CUevent event) {
        expect(state.events.contains(event) && state.current == context(), "query owned event");
        ++state.queries;
        return state.can_stop ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY;
    };
    sdk.execution.event_destroy = [](CUevent event) {
        expect(state.current == context() && state.events.erase(event) == 1,
               "destroy proven native event once");
        ++state.event_destroys;
        return CUDA_SUCCESS;
    };
    sdk.execution.stream_synchronize = [](CUstream stream) {
        expect(stream == reinterpret_cast<CUstream>(state.blas_stream), "sync actual BLAS stream");
        ++state.synchronizes;
        touch(Phase::kSync);
        return state.sync_status;
    };
    sdk.blas.create = [](cublasHandle_t* out) {
        expect(!state.handle_live, "one original BLAS owner");
        state.handle_live = true;
        *out = reinterpret_cast<cublasHandle_t>(&state.handle);
        ++state.creates;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.destroy = [](cublasHandle_t value) {
        native_owner(value);
        state.handle_live = false;
        ++state.destroys;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.set_stream = [](cublasHandle_t value, cudaStream_t stream) {
        native_owner(value);
        state.blas_stream = stream;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.get_stream = [](cublasHandle_t value, cudaStream_t* stream) {
        native_owner(value);
        ++state.stream_gets;
        touch(Phase::kStream);
        *stream = state.blas_stream;
        return state.query_status;
    };
    sdk.blas.set_pointer_mode = [](cublasHandle_t value, cublasPointerMode_t mode) {
        native_owner(value);
        ++state.mode_sets;
        touch(Phase::kSetMode);
        if (state.fail_restore && mode == CUBLAS_POINTER_MODE_DEVICE)
            return CUBLAS_STATUS_EXECUTION_FAILED;
        state.pointer_mode = mode;
        return CUBLAS_STATUS_SUCCESS;
    };
    sdk.blas.get_pointer_mode = [](cublasHandle_t value, cublasPointerMode_t* mode) {
        native_owner(value);
        ++state.mode_gets;
        touch(Phase::kMode);
        *mode = state.pointer_mode;
        return state.query_status;
    };
    sdk.blas.get_version = [](cublasHandle_t value, int* version) {
        native_owner(value);
        ++state.versions;
        touch(Phase::kVersion);
        *version = state.version;
        return state.query_status;
    };
    sdk.blas.sgemm = [](cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                        const float*, const float*, int, const float*, int, const float*, float*,
                        int) {
        throw std::runtime_error("GEAM cannot be implemented by calling GEMM");
        return CUBLAS_STATUS_NOT_SUPPORTED;
    };
    sdk.blas.sgeam = invoke<float>;
    sdk.blas.dgeam = invoke<double>;
    if (missing == 1) sdk.blas.sgeam = nullptr;
    if (missing == 2) sdk.blas.dgeam = nullptr;
    if (missing == 3) sdk.blas.get_version = nullptr;
    return sdk;
}
// NOLINTEND(bugprone-easily-swappable-parameters)

struct Owner {
    std::unique_ptr<CudaExecutionSession> session;
    std::array<CudaSessionResult, 5> allocations;
    std::uint64_t handle{}, stream{}, next{1};
    explicit Owner(unsigned int missing = 0) {
        state = {};
        session = std::make_unique<CudaExecutionSession>(ExecutionSessionIdentity{181, 182}, 183,
                                                         dispatch(missing));
        success(session->initialize(0));
        const auto created = session->create_blas();
        success(created);
        handle = created.id;
        const auto created_stream = session->create_stream(CU_STREAM_NON_BLOCKING);
        success(created_stream);
        stream = created_stream.id;
        success(session->set_blas_stream(handle, stream));
        for (auto& allocation : allocations) {
            allocation = session->allocate(768);
            success(allocation);
        }
    }
    template <typename T>
    void write(std::size_t index, std::span<const T> values, std::size_t offset = 0) {
        success(session->upload({allocations[index].id, offset}, std::as_bytes(values)));
    }
    template <typename T>
    CudaSessionGeam problem() {
        CudaSessionGeam result;
        result.type = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
        result.handle = handle;
        result.presence = 31;
        result.rows = 3;
        result.columns = 4;
        result.leading_a = 6;
        result.leading_b = 7;
        result.leading_c = 5;
        for (std::size_t index = 0; index < 3; ++index) {
            result.buffers[index] = {allocations[index].id, 16 * sizeof(T)};
            std::array<T, 96> values;
            values.fill(T{-8192});
            for (std::size_t element = 0; element < 64; ++element)
                values[16 + element] = static_cast<T>(static_cast<int>(index * 100 + element) - 32);
            write<T>(index, values);
        }
        const T alpha = 0.5, beta = -0.25;
        std::memcpy(result.alpha.data(), &alpha, sizeof(T));
        std::memcpy(result.beta.data(), &beta, sizeof(T));
        write<T>(3, std::span{&alpha, std::size_t{1}});
        write<T>(4, std::span{&beta, std::size_t{1}});
        return result;
    }
    void device(CudaSessionGeam& problem) {
        success(session->set_blas_attribute(handle, CudaBlasAttribute::kPointerMode,
                                            CUBLAS_POINTER_MODE_DEVICE));
        problem.pointer_mode = CUBLAS_POINTER_MODE_DEVICE;
        problem.alpha = problem.beta = {};
        problem.buffers[3] = {allocations[3].id, 0};
        problem.buffers[4] = {allocations[4].id, 0};
    }
    ExecutionAcceptance accept(const CudaSessionGeam& problem) {
        auto result = session->accept_geam(next, problem);
        if (result.accepted()) ++next;
        return result;
    }
    void retire(const ExecutionAcceptance& accepted) {
        expect(session->acknowledge_result(accepted.receipt), "ACK actual admitted reply");
        state.can_stop = true;
        expect(session->poll().retired == 1, "one physically stopped command retires");
        state.can_stop = false;
    }
    void close() {
        success(session->destroy_blas(handle));
        success(session->destroy_stream(stream));
        success(session->shutdown());
        expect(state.destroys == 1 && state.context_destroys == 1 && state.memory.empty() &&
                   state.events.empty(),
               "all proven native owners retire once");
    }
};

void original(Owner& owner, const ExecutionAcceptance& accepted,
              cublasStatus_t expected = CUBLAS_STATUS_SUCCESS) {
    expect(accepted.accepted(), "GEAM admitted without a requested SDK call");
    const auto issued = owner.session->issue_next();
    expect(issued.state == CudaExecutionIssueState::kIssued && issued.library_status == expected &&
               issued.native_status == CUDA_SUCCESS,
           "exact requested original status, never a replacement or Driver conversion");
    const auto reply = owner.session->host_result(accepted.receipt);
    expect(reply && reply->origin == ExecutionReplyOrigin::kNativeSdk && reply->status == expected,
           "host receipt records actual native provenance");
}

using Storage = std::map<CUdeviceptr, std::vector<std::byte>>;
template <typename T>
void oracle(const Owner& owner, const CudaSessionGeam& problem, Storage& expected) {
    const auto before = expected;
    auto address = [&](std::size_t slot) {
        for (const auto& allocation : owner.allocations)
            if (allocation.id == problem.buffers[slot].allocation)
                return std::pair{allocation.address, problem.buffers[slot].offset};
        throw std::runtime_error("oracle requires its own explicitly identified storage");
    };
    T alpha{}, beta{};
    if (problem.presence & 1U) {
        if (problem.pointer_mode) {
            const auto [base, offset] = address(3);
            std::memcpy(&alpha, before.at(base).data() + offset, sizeof(T));
        } else
            std::memcpy(&alpha, problem.alpha.data(), sizeof(T));
    }
    if (problem.presence & 2U) {
        if (problem.pointer_mode) {
            const auto [base, offset] = address(4);
            std::memcpy(&beta, before.at(base).data() + offset, sizeof(T));
        } else
            std::memcpy(&beta, problem.beta.data(), sizeof(T));
    }
    const auto [destination, destination_offset] = address(2);
    for (int row = 0; row < problem.rows; ++row) {
        for (int column = 0; column < problem.columns; ++column) {
            T result{};
            for (std::size_t input = 0; input != 2; ++input) {
                const auto coefficient = input == 0 ? alpha : beta;
                if (coefficient == T{}) continue;
                const auto [base, offset] = address(input);
                const bool transpose =
                    (input == 0 ? problem.transpose_a : problem.transpose_b) != 0;
                const int leading = input == 0 ? problem.leading_a : problem.leading_b;
                const auto element = transpose ? row * leading + column : column * leading + row;
                T value{};
                std::memcpy(&value, before.at(base).data() + offset + sizeof(T) * element,
                            sizeof(T));
                result += coefficient * value;
            }
            const auto element = column * problem.leading_c + row;
            std::memcpy(expected.at(destination).data() + destination_offset + sizeof(T) * element,
                        &result, sizeof(T));
        }
    }
}

template <typename T>
void numerical() {
    for (unsigned int device = 0; device != 2; ++device) {
        for (unsigned int ta = 0; ta != 3; ++ta) {
            for (unsigned int tb = 0; tb != 3; ++tb) {
                Owner owner;
                auto problem = owner.problem<T>();
                problem.transpose_a = ta;
                problem.transpose_b = tb;
                if (device) owner.device(problem);
                auto expected = state.memory;
                oracle<T>(owner, problem, expected);
                const auto setters = state.mode_sets;
                auto accepted = owner.accept(problem);
                expect(state.calls == 0 && state.downloads == 0 && state.versions == 0 &&
                           state.stream_gets == 0 && state.mode_gets == 0,
                       "admission never queries the SDK or snapshots DEVICE scalars");
                problem.alpha.fill(std::byte{0xFF});
                problem.beta.fill(std::byte{0xFF});
                original(owner, accepted);
                expect(state.memory == expected, "full independent padded-matrix and guard oracle");
                expect(
                    state.last.type == (std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F) &&
                        state.last.shape == std::array<int, 7>{static_cast<int>(ta),
                                                               static_cast<int>(tb), 3, 4, 6, 7, 5},
                    "all independent S/D and N/T/C arguments reach the exact original ABI");
                for (std::size_t index = 0; index != 3; ++index)
                    expect(reinterpret_cast<CUdeviceptr>(state.last.pointers[index]) ==
                               owner.allocations[index].address + 16 * sizeof(T),
                           "matrix identities and offsets do not swap");
                expect(state.synchronizes == 0 && state.downloads == 0 &&
                           state.mode_sets == setters &&
                           state.last.mode ==
                               (device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST),
                       "fully owned native path neither snapshots nor changes pointer mode");
                if (device)
                    for (std::size_t index = 3; index != 5; ++index)
                        expect(reinterpret_cast<CUdeviceptr>(state.last.pointers[index]) ==
                                   owner.allocations[index].address,
                               "actual device scalar identities");
                owner.retire(accepted);
                owner.close();
            }
        }
    }
}

void aliases_and_nulls() {
    for (unsigned int which = 0; which != 8; ++which) {
        Owner owner;
        auto problem = owner.problem<double>();
        if (which < 2) {
            problem.buffers[2] = problem.buffers[which];
            problem.leading_c = which == 0 ? problem.leading_a : problem.leading_b;
        } else if (which == 2) {
            problem.buffers[1] = problem.buffers[0];
            problem.leading_b = problem.leading_a;
        } else if (which == 3) {
            problem.presence &= ~1U;
            problem.alpha = {};
            problem.buffers[0] = {};  // Present unresolved A is ignored by absent HOST alpha.
        } else if (which == 4) {
            const double minus_zero = -0.0;
            std::memcpy(problem.alpha.data(), &minus_zero, sizeof(minus_zero));
            problem.buffers[0] = {};
        } else if (which == 5) {
            problem.presence &= ~4U;
            problem.buffers[0] = {};
            problem.alpha = {};
        } else if (which == 6) {
            problem.buffers[2] = problem.buffers[0];
            problem.transpose_a = CUBLAS_OP_T;
        } else {
            problem.rows = 0;
            problem.buffers = {};
            problem.presence = 0;
            problem.alpha = problem.beta = {};
        }
        auto expected = state.memory;
        if (which < 6) oracle<double>(owner, problem, expected);
        const auto accepted = owner.accept(problem);
        original(owner, accepted, which == 6 ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS);
        expect(state.memory == expected,
               "aliases, ignored NULLs and zero-sized calls preserve guards");
        if (which == 4) {
            std::uint64_t bits{};
            std::memcpy(&bits, state.last.alpha.data(), sizeof(bits));
            expect(bits == UINT64_C(0x8000000000000000), "negative-zero coefficient bits survive");
        }
        owner.retire(accepted);
        owner.close();
    }
    for (const auto bits : {UINT64_C(0x7ff0000000000000), UINT64_C(0x7ff8000000000029)}) {
        Owner owner;
        auto problem = owner.problem<double>();
        std::memcpy(problem.alpha.data(), &bits, sizeof(bits));
        state.arithmetic = false;  // ABI-only test; no pretend NaN numerical oracle.
        const auto accepted = owner.accept(problem);
        original(owner, accepted);
        expect(std::memcmp(state.last.alpha.data(), &bits, sizeof(bits)) == 0,
               "infinity and NaN payload are not normalized by transport");
        owner.retire(accepted);
        owner.close();
    }
}

void clean_preparation_and_cancellation() {
    for (unsigned int which = 0; which != 4; ++which) {
        Owner owner;
        auto problem = owner.problem<float>();
        if (which == 0) {
            owner.device(problem);
            problem.buffers[0] = {};  // Positive snapshot makes unresolved A genuinely active.
        } else if (which == 1) {
            problem.rows = 0;
            state.version = 120605;
        } else if (which == 2) {
            owner.device(problem);
            problem.buffers[0] = problem.buffers[2];
            problem.buffers[0].offset +=
                sizeof(float);  // Unsafe partial overlap needs actual alpha.
        }
        const auto accepted = owner.accept(problem);
        expect(accepted.accepted(), "classification and scalar observation occur at queued issue");
        if (which == 3) {
            expect(owner.session->reject_queued(CUDA_ERROR_NOT_READY) == 1,
                   "cancel queued GEAM once");
            const auto reply = owner.session->host_result(accepted.receipt);
            expect(reply && reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       reply->status == CUDA_ERROR_NOT_READY && state.calls == 0 &&
                       state.versions == 0 && state.stream_gets == 0 && state.downloads == 0 &&
                       state.event_creates == 0,
                   "queued cancellation does no original or private SDK preparation");
            expect(owner.session->acknowledge_result(accepted.receipt), "ACK cancelled result");
        } else {
            const auto issue = owner.session->issue_next();
            const auto reply = owner.session->host_result(accepted.receipt);
            const auto status = which == 1 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_ERROR_INVALID_VALUE;
            expect(issue.state == CudaExecutionIssueState::kRejected && !issue.library_status &&
                       reply && reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue &&
                       reply->status == status && state.calls == 0,
                   "clean policy failure is not an invented GEAM native status");
            expect(!owner.session->snapshot().is_quarantined &&
                       !owner.session->snapshot().submissions.is_quarantined,
                   "known preparation rejection leaves the owner usable");
            expect(state.records == 1, "private preparation retains a trailing physical fence");
            owner.retire(accepted);
            state.version = 120604;
            auto valid = owner.problem<float>();
            if (which == 0 || which == 2) owner.device(valid);
            const auto successor = owner.accept(valid);
            original(owner, successor);
            owner.retire(successor);
        }
        owner.close();
    }
}

void admission_controls() {
    Owner owner;
    const auto good = owner.problem<float>();
    for (unsigned int which = 0; which != 11; ++which) {
        auto invalid = good;
        switch (which) {
            case 0:
                invalid.handle = 999999;
                break;
            case 1:
                invalid.type = CUDA_R_16F;
                break;
            case 2:
                invalid.transpose_a = UINT32_MAX;
                break;
            case 3:
                invalid.pointer_mode = 2;
                break;
            case 4:
                invalid.presence = 32;
                break;
            case 5:
                invalid.buffers[0].allocation = 999999;
                break;
            case 6:
                invalid.buffers[0].offset = 769;
                break;
            case 7:
                invalid.alpha[15] = std::byte{1};
                break;
            case 8:
                invalid.buffers[3] = {owner.allocations[3].id, 0};
                break;
            case 9:
                invalid.presence &= ~4U;
                break;
            case 10:
                invalid.buffers[2].offset = 764;
                break;
        }
        const auto rejected = owner.accept(invalid);
        expect(!rejected.accepted() && !rejected.receipt && state.calls == 0 &&
                   state.versions == 0 && state.stream_gets == 0,
               "malformed authority rejects without requested provenance");
    }
    for (unsigned int which = 0; which != 3; ++which) {
        auto native_invalid = good;
        if (which == 0) native_invalid.transpose_a = 3;
        if (which == 1) native_invalid.rows = -1;
        if (which == 2) native_invalid.leading_c = 0;
        const auto accepted = owner.accept(native_invalid);
        original(owner, accepted, CUBLAS_STATUS_INVALID_VALUE);
        owner.retire(accepted);
    }
    owner.close();
    for (unsigned int missing = 1; missing != 4; ++missing) {
        Owner incomplete(missing);
        auto problem = missing == 2 ? incomplete.problem<double>() : incomplete.problem<float>();
        if (missing == 3) problem.rows = 0;
        expect(!incomplete.accept(problem).accepted() && state.calls == 0,
               "missing required callbacks reject before issuing or fabricating a native error");
        incomplete.close();
    }
}

void physical_lifetime() {
    Owner owner;
    auto problem = owner.problem<float>();
    owner.device(problem);
    const auto accepted = owner.accept(problem);
    original(owner, accepted);
    const auto frontier = owner.session->capture_command_frontier(accepted.receipt.sequence());
    expect(frontier.status == CUDA_SUCCESS && owner.session->acknowledge_result(accepted.receipt),
           "capture and ACK the actual GEAM command");
    expect(owner.session->poll().retained == 1 &&
               owner.session->query_frontier(frontier.frontier) == CUDA_ERROR_NOT_READY,
           "host success and ACK do not prove physical completion");
    for (const auto& allocation : owner.allocations)
        expect(
            owner.session->release(allocation.id).disposition == CudaSessionDisposition::kDeferred,
            "all referenced matrices and device scalars remain physically leased");
    expect(owner.session->destroy_blas(owner.handle).disposition ==
                   CudaSessionDisposition::kDeferred &&
               owner.session->set_blas_stream(owner.handle, 0).disposition ==
                   CudaSessionDisposition::kDeferred &&
               owner.session->set_blas_attribute(owner.handle, CudaBlasAttribute::kPointerMode, 0)
                       .disposition == CudaSessionDisposition::kDeferred &&
               state.frees == 0,
           "pending GEAM excludes native handle mutation and destruction");
    state.can_stop = true;
    expect(owner.session->poll().retired == 1 &&
               owner.session->query_frontier(frontier.frontier) == CUDA_SUCCESS,
           "only physical event proof releases the command's leases");
    for (const auto& allocation : owner.allocations) success(owner.session->release(allocation.id));
    owner.close();
}

void device_producer_and_null_scalar() {
    Owner owner;
    auto problem = owner.problem<float>();
    owner.device(problem);
    const float zero = 0, one = 1;
    owner.write<float>(3, std::span{&one, std::size_t{1}});
    owner.write<float>(4, std::span{&zero, std::size_t{1}});
    const auto producer =
        owner.session->accept_memory(owner.next, {.kind = CudaMemoryKind::kCopy,
                                                  .destination = problem.buffers[3],
                                                  .source = problem.buffers[4],
                                                  .count = sizeof(float),
                                                  .stream = owner.stream,
                                                  .is_asynchronous = true});
    expect(producer.accepted(), "owned scalar producer admitted first");
    ++owner.next;
    problem.buffers[0] = {};
    const auto accepted = owner.accept(problem);
    expect(
        accepted.accepted() && state.downloads == 0 && state.synchronizes == 0 && state.copies == 0,
        "queued scalar snapshot cannot overtake the earlier producer");
    expect(
        owner.session->issue_next().state == CudaExecutionIssueState::kIssued && state.copies == 1,
        "actual producer executes first");
    original(owner, accepted);
    expect(
        state.synchronizes == 1 && state.downloads == 2 && state.mode_sets == 3 &&
            state.last.mode == CUBLAS_POINTER_MODE_HOST &&
            state.pointer_mode == CUBLAS_POINTER_MODE_DEVICE,
        "conditional device proof synchronizes, snapshots both, invokes HOST and restores DEVICE");
    for (int column = 0; column != 4; ++column)
        for (int row = 0; row != 3; ++row) {
            float actual{};
            std::memcpy(
                &actual,
                memory(owner.allocations[2].address + sizeof(float) * (16 + column * 5 + row),
                       sizeof(float)),
                sizeof(float));
            expect(actual == 0, "producer's zero scalar is used instead of admission-time one");
        }
    expect(owner.session->acknowledge_result(producer.receipt) &&
               owner.session->acknowledge_result(accepted.receipt),
           "ACK both real results");
    state.can_stop = true;
    expect(owner.session->poll().retired == 2, "both producer and consumer retire after proof");
    state.can_stop = false;
    problem = owner.problem<float>();
    owner.device(problem);
    problem.presence &= ~1U;
    problem.buffers[3] = {};
    const auto syncs = state.synchronizes, downloads = state.downloads, setters = state.mode_sets;
    const auto absent = owner.accept(problem);
    original(owner, absent, CUBLAS_STATUS_INVALID_VALUE);
    expect(state.last.mode == CUBLAS_POINTER_MODE_DEVICE && state.last.pointers[3] == nullptr &&
               state.synchronizes == syncs && state.downloads == downloads &&
               state.mode_sets == setters,
           "NULL DEVICE scalar remains a real native DEVICE error, not HOST zero semantics");
    owner.retire(absent);
    owner.close();
}

void callback_exclusion() {
    constexpr std::array phases{Phase::kContext, Phase::kVersion,  Phase::kStream,  Phase::kMode,
                                Phase::kSync,    Phase::kDownload, Phase::kSetMode, Phase::kGeam};
    for (const auto phase : phases) {
        Owner owner;
        auto problem = owner.problem<float>();
        owner.device(problem);
        const float zero = 0;
        owner.write<float>(3, std::span{&zero, std::size_t{1}});
        problem.buffers[0] = {};
        state.probe = [&] {
            const auto calls = state.calls, sets = state.mode_sets, gets = state.mode_gets;
            expect(!owner.accept(problem).accepted() &&
                       owner.session->poll().status == CUDA_ERROR_NOT_READY &&
                       owner.session->drain_issued().status == CUDA_ERROR_NOT_READY &&
                       owner.session->destroy_blas(owner.handle).disposition ==
                           CudaSessionDisposition::kDeferred &&
                       owner.session->set_blas_stream(owner.handle, 0).disposition ==
                           CudaSessionDisposition::kDeferred &&
                       owner.session->shutdown().disposition == CudaSessionDisposition::kDeferred,
                   "native callback cannot recursively admit, drain, mutate or destroy its owner");
            expect(state.calls == calls && state.mode_sets == sets && state.mode_gets == gets,
                   "blocked callback reentry performs no nested native callback");
        };
        if (phase == Phase::kContext) state.armed = phase;
        const auto accepted = owner.accept(problem);
        if (phase != Phase::kContext) state.armed = phase;
        original(owner, accepted);
        expect(state.probes == 1, "the selected real callback boundary was exercised");
        owner.retire(accepted);
        owner.close();
    }
}

void poison(unsigned int which) {
    Owner owner;
    auto problem = owner.problem<float>();
    const auto allocations = state.memory.size();
    if (which <= 3) {
        owner.device(problem);
        const float zero = 0;
        owner.write<float>(3, std::span{&zero, std::size_t{1}});
        problem.buffers[0] = {};
        if (which == 0) state.sync_status = CUDA_ERROR_UNKNOWN;
        if (which == 1) state.download_status = CUDA_ERROR_UNKNOWN;
        if (which == 2) state.fail_restore = true;
        if (which == 3) {
            state.armed = Phase::kDownload;
            state.probe = [] { state.current = nullptr; };
        }
    } else if (which == 4 || which == 5) {
        state.armed = Phase::kGeam;
        state.probe = [which] {
            if (which == 4)
                state.current = nullptr;
            else
                state.pointer_mode = CUBLAS_POINTER_MODE_DEVICE;
        };
    } else if (which == 6) {
        state.record_status = CUDA_ERROR_UNKNOWN;
        state.sync_status = CUDA_ERROR_UNKNOWN;
    } else {
        state.native_status = CUBLAS_STATUS_EXECUTION_FAILED;
        state.partial_write = true;
    }
    const auto accepted = owner.accept(problem);
    expect(accepted.accepted(), "fault is injected only after admission");
    const auto issued = owner.session->issue_next();
    const bool was_called = which == 2 || which >= 4;
    const auto reply = owner.session->host_result(accepted.receipt);
    expect(reply && issued.library_status.has_value() == was_called &&
               (was_called ? reply->origin == ExecutionReplyOrigin::kNativeSdk
                           : reply->origin == ExecutionReplyOrigin::kRejectedBeforeIssue),
           "private failure and actual original GEAM status remain distinguishable");
    if (was_called)
        expect(*issued.library_status ==
                   (which == 7 ? CUBLAS_STATUS_EXECUTION_FAILED : CUBLAS_STATUS_SUCCESS),
               "post-call physical failure cannot replace actual native status");
    expect(owner.session->acknowledge_result(accepted.receipt), "ACK fault receipt");
    if (which == 7) {
        float marker{};
        std::memcpy(&marker,
                    memory(owner.allocations[2].address + 16 * sizeof(float), sizeof(float)),
                    sizeof(float));
        expect(
            marker == 123 && state.calls == 1 && state.records == 1 &&
                owner.session->poll().retained == 1 && !owner.session->snapshot().is_quarantined,
            "native error can partially write and still has exact status and physical retention");
        state.can_stop = true;
        expect(owner.session->poll().retired == 1, "healthy native-error work physically retires");
        owner.close();
    } else {
        state.current = context();
        state.can_stop = true;
        static_cast<void>(owner.session->poll());
        static_cast<void>(owner.session->drain_issued());
        static_cast<void>(owner.session->shutdown());
        expect(state.memory.size() == allocations && state.destroys == 0 &&
                   state.context_destroys == 0 && state.calls == (was_called ? 1U : 0U),
               "unknown completion never frees physical owners or retries requested native work");
    }
}

void isolated_faults() {
    for (unsigned int which = 0; which != 8; ++which) {
        const auto child = ::fork();
        expect(child >= 0, "isolated proof fixture fork");
        if (child == 0) {
            try {
                poison(which);
                ::_exit(0);
            } catch (const std::exception& error) {
                std::cerr << "GEAM poison " << which << ": " << error.what() << '\n';
                ::_exit(1);
            }
        }
        int result{};
        expect(
            ::waitpid(child, &result, 0) == child && WIFEXITED(result) && WEXITSTATUS(result) == 0,
            "isolated GEAM physical-proof case");
    }
}
}  // namespace

int main() {
    try {
        numerical<float>();
        numerical<double>();
        aliases_and_nulls();
        clean_preparation_and_cancellation();
        admission_controls();
        physical_lifetime();
        device_producer_and_null_scalar();
        callback_exclusion();
        isolated_faults();
        std::cout << "Owned original GEAM CPU fixture passed (no SDK/GPU execution).\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
