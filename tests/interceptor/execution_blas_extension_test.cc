#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <cuda.h>
#include <cublas_v2.h>

#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_memory_native_calls();
extern "C" unsigned int glimmer_blas_native_calls();
extern "C" unsigned int glimmer_blas_extension_native_calls();

namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kReceiptCases = 23;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned Driver operation");
}
void check(cublasStatus_t value) {
    expect(value == CUBLAS_STATUS_SUCCESS, "owned BLAS operation");
}
constexpr CUdeviceptr address(std::size_t index) {
    return 0x100000 + index * 0x1000;
}
// NOLINTBEGIN(performance-no-int-to-ptr): opaque fake GPU addresses, never host memory.
template <typename T>
T* pointer(CUdeviceptr value) {
    return reinterpret_cast<T*>(value);
}
// NOLINTEND(performance-no-int-to-ptr)
template <typename T>
T symbol(void* provider, const char* name) {
    const auto result = ::dlsym(provider, name);
    expect(result != nullptr, "qualified SDK lookup");
    return reinterpret_cast<T>(result);
}
struct Libraries {
    void* driver{};
    void* blas{};
    void* extension{};
    decltype(&cublasSgemv_v2) sgemv{};
    decltype(&cublasSgemmEx) sgemm_ex{};
    explicit Libraries(char** paths) {
        driver = ::dlopen(paths[1], RTLD_NOW | RTLD_LOCAL);
        blas = ::dlopen(paths[2], RTLD_NOW | RTLD_LOCAL);
        extension = ::dlopen(paths[3], RTLD_NOW | RTLD_LOCAL);
        expect(driver && blas && extension, "load independent native trap providers");
        sgemv = symbol<decltype(sgemv)>(extension, "cublasSgemv_v2");
        sgemm_ex = symbol<decltype(sgemm_ex)>(extension, "cublasSgemmEx");
        expect(sgemv == &cublasSgemv_v2 && sgemm_ex == &cublasSgemmEx &&
                   reinterpret_cast<void*>(sgemv) == ::dlsym(RTLD_DEFAULT, "cublasSgemv_v2") &&
                   reinterpret_cast<void*>(sgemm_ex) == ::dlsym(RTLD_DEFAULT, "cublasSgemmEx"),
               "PLT, cached, global and qualified lookup select the exact owned ABIs");
    }
    ~Libraries() {
        if (extension) static_cast<void>(::dlclose(extension));
        if (blas) static_cast<void>(::dlclose(blas));
        if (driver) static_cast<void>(::dlclose(driver));
    }
};
struct Case {
    bool is_gemm{}, is_device{}, is_direct{};
    std::uint32_t ta{}, tb{1}, type{};
    int rows{3}, columns{4}, depth{5}, lda{8}, ldb{9}, ldc{7}, incx{1}, incy{1};
    std::array<CUdeviceptr, 3> buffers{address(0) + 16, address(1) + 32, address(2) + 48};
    std::array<std::uint64_t, 3> buffer_ids{11, 12, 13};
    std::array<CUdeviceptr, 2> scalar_buffers{address(3) + 4, address(3) + 16};
    std::array<std::uint64_t, 2> scalar_ids{14, 14};
    std::uint32_t presence{31};
    bool has_host_scalar_copy{true}, has_null_host_scalars{}, has_protected_host_scalars{};
    cublasStatus_t status{CUBLAS_STATUS_SUCCESS};
};
std::array<Case, 8> cases() {
    std::array<Case, 8> result;
    result[0].is_direct = true;
    result[1].is_device = true;
    result[1].ta = 1;
    result[1].incx = -2;
    result[1].incy = -3;
    result[2].ta = 2;
    result[2].incx = 3;
    result[3].status = CUBLAS_STATUS_EXECUTION_FAILED;
    for (std::size_t index = 4; index != result.size(); ++index) result[index].is_gemm = true;
    result[4].is_direct = true;
    result[5].type = 2;  // CUDA_R_16F; scalar ABI remains FP32.
    result[5].ta = 2;
    result[5].tb = 0;
    result[6].is_device = true;
    result[7].status = CUBLAS_STATUS_EXECUTION_FAILED;
    return result;
}

std::vector<Case> boundary_cases() {
    std::vector<Case> result;
    for (const bool is_gemm : {false, true}) {
        for (const bool is_device : {false, true}) {
            Case zero;
            zero.is_gemm = is_gemm;
            zero.is_device = is_device;
            zero.is_direct = !is_device;
            zero.rows = 0;
            zero.buffers = {};
            zero.buffer_ids = {};
            zero.scalar_buffers = {};
            zero.scalar_ids = {};
            zero.presence = 0;
            zero.has_host_scalar_copy = false;
            zero.has_null_host_scalars = true;
            result.push_back(zero);

            Case negative;
            negative.is_gemm = is_gemm;
            negative.is_device = is_device;
            negative.rows = -1;
            negative.has_host_scalar_copy = false;
            negative.has_protected_host_scalars = true;
            negative.status = is_gemm ? CUBLAS_STATUS_INVALID_VALUE : CUBLAS_STATUS_SUCCESS;
            result.push_back(negative);

            Case endpoint = negative;
            if (is_gemm) {
                endpoint.rows = 3;
                endpoint.lda = 1;
            }
            endpoint.buffers = {address(0) + 512, address(1) + 512, address(2) + 512};
            endpoint.scalar_buffers = {address(3) + 512, address(3) + 512};
            result.push_back(endpoint);

            Case zero_columns = negative;
            zero_columns.rows = 3;
            zero_columns.columns = 0;
            zero_columns.has_protected_host_scalars = false;
            zero_columns.status = CUBLAS_STATUS_SUCCESS;
            result.push_back(zero_columns);
            if (is_gemm) {
                Case zero_depth;
                zero_depth.is_gemm = true;
                zero_depth.is_device = is_device;
                zero_depth.depth = 0;
                zero_depth.buffers[0] = zero_depth.buffers[1] = 0;
                zero_depth.buffer_ids[0] = zero_depth.buffer_ids[1] = 0;
                zero_depth.presence = 19;
                result.push_back(zero_depth);
            }
        }
    }
    // Keep a live ignored operand pinned across the final held receipt.
    Case retained;
    retained.rows = -1;
    retained.has_host_scalar_copy = false;
    retained.has_protected_host_scalars = true;
    result.push_back(retained);
    return result;
}

// Independent literal little-endian ledger, not the production writer or reader.
std::vector<std::byte> expected_bytes(const Case& test) {
    std::vector<std::byte> result;
    const auto integer = [&](std::uint64_t value, unsigned int bytes) {
        for (unsigned int index = 0; index != bytes; ++index)
            result.push_back(static_cast<std::byte>((value >> (index * 8)) & 255));
    };
    const auto signed32 = [&](int value) { integer(std::bit_cast<std::uint32_t>(value), 4); };
    integer(test.ta, 4);
    if (test.is_gemm) {
        integer(test.tb, 4);
        for (unsigned int index = 0; index != 3; ++index) integer(test.type, 4);
    }
    signed32(test.rows);
    signed32(test.columns);
    if (test.is_gemm) signed32(test.depth);
    signed32(test.lda);
    signed32(test.is_gemm ? test.ldb : test.incx);
    signed32(test.is_gemm ? test.ldc : test.incy);
    integer(70, 8);
    integer(test.is_device ? 1 : 0, 4);
    integer(test.presence, 4);
    for (std::size_t index = 0; index != 3; ++index) {
        integer(test.buffer_ids[index], 8);
        integer(test.buffer_ids[index] ? test.buffers[index] - address(index) : 0, 8);
    }
    for (unsigned int index = 0; index != 2; ++index) {
        integer(test.is_device ? test.scalar_ids[index] : 0, 8);
        integer(
            test.is_device && test.scalar_ids[index] ? test.scalar_buffers[index] - address(3) : 0,
            8);
    }
    for (const auto bits : {0x3f000000U, 0xbe800000U}) {
        integer(test.is_device || !test.has_host_scalar_copy ? 0 : bits, 4);
        integer(0, 8);
        integer(0, 4);
    }
    expect(result.size() == (test.is_gemm ? 172U : 152U), "literal ABI byte count");
    return result;
}

struct HostScalars {
    const float* alpha{};
    const float* beta{};
};
cublasStatus_t invoke(const Libraries& sdk, cublasHandle_t handle, const Case& test,
                      HostScalars host) {
    const auto* alpha = test.is_device ? pointer<float>(test.scalar_buffers[0]) : host.alpha;
    const auto* beta = test.is_device ? pointer<float>(test.scalar_buffers[1]) : host.beta;
    if (test.is_gemm) {
        const auto function = test.is_direct ? &cublasSgemmEx : sdk.sgemm_ex;
        return function(
            handle, static_cast<cublasOperation_t>(test.ta),
            static_cast<cublasOperation_t>(test.tb), test.rows, test.columns, test.depth, alpha,
            pointer<void>(test.buffers[0]), static_cast<cudaDataType>(test.type), test.lda,
            pointer<void>(test.buffers[1]), static_cast<cudaDataType>(test.type), test.ldb, beta,
            pointer<void>(test.buffers[2]), static_cast<cudaDataType>(test.type), test.ldc);
    }
    const auto function = test.is_direct ? &cublasSgemv_v2 : sdk.sgemv;
    return function(handle, static_cast<cublasOperation_t>(test.ta), test.rows, test.columns, alpha,
                    pointer<float>(test.buffers[0]), test.lda, pointer<float>(test.buffers[1]),
                    test.incx, beta, pointer<float>(test.buffers[2]), test.incy);
}
struct Hostile {
    ExecutionWireReply reply;
    cublasStatus_t expected{CUBLAS_STATUS_INTERNAL_ERROR};
    bool is_healthy{}, is_disconnect{};
};
Hostile hostile(unsigned int which) {
    Hostile test;
    auto& reply = test.reply;
    reply = {.disposition = ExecutionWireDisposition::kComplete,
             .status = 0,
             .original_status = 0,
             .id = 1,
             .arguments = {},
             .data = {}};
    switch (which) {
        case 0:
            reply.id = 0;
            break;
        case 1:
            reply.original_status.reset();
            break;
        case 2:
            reply.status = 13;
            break;
        case 3:
            reply.status = -1;
            reply.original_status = -1;
            break;
        case 4:
            reply.status = 32;
            reply.original_status = 32;
            break;
        case 5:
            reply.address = 1;
            break;
        case 6:
            reply.data.push_back(std::byte{1});
            break;
        case 7:
            reply.parameter_bytes = 2;
            break;
        case 8:
            reply.parameter_bytes = 1;
            reply.arguments.push_back({0, 1});
            break;
        case 9:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            break;
        case 10:
        case 11:
        case 12:
        case 13:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = which == 10 ? 0 : which == 11 ? -1 : which == 12 ? 1024 : 1;
            if (which != 13) reply.original_status.reset();
            break;
        case 14:
            reply.parameter_bytes = 1;
            test.expected = CUBLAS_STATUS_SUCCESS;
            break;
        case 15:
            reply.parameter_bytes = 1;
            reply.status = 13;
            reply.original_status = 13;
            test.expected = CUBLAS_STATUS_EXECUTION_FAILED;
            break;
        case 16:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.id = 0;
            reply.status = CUDA_ERROR_INVALID_VALUE;
            test.expected = CUBLAS_STATUS_INVALID_VALUE;
            test.is_healthy = true;
            break;
        case 17:
            reply.status = 13;
            reply.original_status = 13;
            test.expected = CUBLAS_STATUS_EXECUTION_FAILED;
            test.is_healthy = true;
            break;
        case 18:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            test.expected = CUBLAS_STATUS_NOT_SUPPORTED;
            test.is_healthy = true;
            break;
        case 19:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_UNKNOWN;
            reply.parameter_bytes = 1;
            break;
        case 20:
            reply.status = 31;
            reply.original_status = 31;
            test.expected = static_cast<cublasStatus_t>(31);
            test.is_healthy = true;
            break;
        case 21:
            reply.disposition = ExecutionWireDisposition::kAdmissionAccepted;
            reply.original_status.reset();
            reply.address = 1;
            break;
        case 22:
            test.is_disconnect = true;
            break;
        default:
            throw std::runtime_error("unknown isolated receipt");
    }
    return test;
}
struct Latches {
    std::atomic<bool> saw_command{}, release_command{}, setter_started{};
};
HostScalars boundary_host(const Case& test, HostScalars ordinary, const float* guard) {
    if (test.has_null_host_scalars) return {};
    if (test.has_protected_host_scalars) return {guard, guard};
    return ordinary;
}
void boundary_application(const Libraries& sdk, CUcontext context, cublasHandle_t handle,
                          std::array<CUdeviceptr, 4>& allocations, HostScalars host,
                          const float* guard, Latches& latches) {
    const auto boundary = boundary_cases();
    for (std::size_t index = 0; index + 1 != boundary.size(); ++index) {
        const auto& test = boundary[index];
        check(cublasSetPointerMode(
            handle, test.is_device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST));
        expect(invoke(sdk, handle, test, boundary_host(test, host, guard)) == test.status,
               "exact original boundary status, presence and no-read scalar serialization");
    }
    for (const bool is_gemm : {false, true}) {
        check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        Case active;
        active.is_gemm = is_gemm;
        expect(invoke(sdk, handle, active, {guard, host.beta}) == CUBLAS_STATUS_INVALID_VALUE,
               "active HOST scalar remains safely copied");
        expect(invoke(sdk, handle, active, {nullptr, host.beta}) == CUBLAS_STATUS_NOT_SUPPORTED,
               "active NULL scalar remains outside qualified boundary semantics");
        active.buffers[0] = address(0) + 512;
        expect(invoke(sdk, handle, active, host) == CUBLAS_STATUS_INVALID_VALUE,
               "active data endpoint cannot acquire positive-extent authority");
        active.buffers[0] = address(0);
        active.is_device = true;
        active.scalar_buffers[0] = address(3) + 511;
        check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
        expect(invoke(sdk, handle, active, host) == CUBLAS_STATUS_INVALID_VALUE,
               "active DEVICE coefficient retains its complete four-byte extent");
        for (const bool is_device : {false, true}) {
            check(cublasSetPointerMode(
                handle, is_device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST));
            Case unresolved;
            unresolved.is_gemm = is_gemm;
            unresolved.is_device = is_device;
            unresolved.rows = 0;
            unresolved.buffers[0] = 1;
            expect(invoke(sdk, handle, unresolved, {guard, guard}) == CUBLAS_STATUS_INVALID_VALUE,
                   "metadata-only nonnull data never normalizes to native NULL");
            if (is_device) {
                unresolved.buffers[0] = address(0);
                unresolved.scalar_buffers[0] = 1;
                expect(invoke(sdk, handle, unresolved, host) == CUBLAS_STATUS_INVALID_VALUE,
                       "metadata-only nonnull DEVICE scalar needs an owned identity");
            }
            Case mixed;
            mixed.is_gemm = is_gemm;
            mixed.is_device = is_device;
            mixed.rows = 0;
            mixed.buffers[0] = 0;
            expect(invoke(sdk, handle, mixed, {guard, guard}) == CUBLAS_STATUS_NOT_SUPPORTED,
                   "unqualified mixed data presence cannot reach the original SDK");
            mixed.buffers[0] = address(0);
            mixed.scalar_buffers[0] = 0;
            expect(invoke(sdk, handle, mixed, {nullptr, guard}) == CUBLAS_STATUS_NOT_SUPPORTED,
                   "unqualified mixed scalar presence cannot acquire no-read permission");
        }
    }
    check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    Case zero_depth;
    zero_depth.is_gemm = true;
    zero_depth.depth = 0;
    zero_depth.buffers = {};
    expect(invoke(sdk, handle, zero_depth, host) == CUBLAS_STATUS_NOT_SUPPORTED,
           "zero-depth product still requires its nonzero output matrix");
    zero_depth.buffers[2] = 1;
    expect(invoke(sdk, handle, zero_depth, host) == CUBLAS_STATUS_INVALID_VALUE,
           "zero-depth output cannot become unresolved native authority");
    zero_depth.buffers[2] = address(2);
    zero_depth.buffers[0] = 1;
    expect(invoke(sdk, handle, zero_depth, host) == CUBLAS_STATUS_INVALID_VALUE,
           "zero-depth ignored nonnull input cannot normalize to native NULL");
    zero_depth.buffers[0] = address(0);
    expect(invoke(sdk, handle, zero_depth, host) == CUBLAS_STATUS_NOT_SUPPORTED,
           "zero-depth mixed A-only presence remains outside qualified classes");
    zero_depth.buffers[0] = 0;
    zero_depth.buffers[1] = address(1);
    expect(invoke(sdk, handle, zero_depth, host) == CUBLAS_STATUS_NOT_SUPPORTED,
           "zero-depth mixed B-only presence remains outside qualified classes");

    CUcontext foreign_context{};
    CUdeviceptr foreign_allocation{};
    check(cuCtxCreate(&foreign_context, 0, 0));
    check(cuMemAlloc(&foreign_allocation, 512));
    expect(foreign_allocation == address(4), "independent foreign-context allocation ledger");
    check(cuCtxSetCurrent(context));
    for (const bool is_gemm : {false, true}) {
        Case foreign;
        foreign.is_gemm = is_gemm;
        foreign.rows = 0;
        foreign.buffers[0] = foreign_allocation;
        expect(invoke(sdk, handle, foreign, {guard, guard}) == CUBLAS_STATUS_INVALID_VALUE,
               "known foreign allocation is not unresolved ignored presence");
    }
    check(cuCtxSetCurrent(foreign_context));
    check(cuMemFree(foreign_allocation));
    check(cuCtxDestroy(foreign_context));
    check(cuCtxSetCurrent(context));

    latches.saw_command = false;
    latches.release_command = false;
    latches.setter_started = false;
    auto command = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return invoke(sdk, handle, boundary.back(), {guard, guard});
    });
    const auto deadline = Clock::now() + std::chrono::seconds{3};
    while (!latches.saw_command && Clock::now() < deadline) std::this_thread::yield();
    expect(latches.saw_command, "held ignored-operand request observed");
    auto release = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        latches.setter_started = true;
        return cuMemFree(allocations[0]);
    });
    while (!latches.setter_started && Clock::now() < deadline) std::this_thread::yield();
    const bool is_blocked =
        latches.setter_started &&
        release.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        command.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
    latches.release_command = true;
    check(command.get());
    check(release.get());
    expect(is_blocked, "ignored but owned operands remain pinned through exact SDK receipt");
    allocations[0] = 0;
    for (const bool is_gemm : {false, true}) {
        Case stale;
        stale.is_gemm = is_gemm;
        stale.rows = 0;
        expect(invoke(sdk, handle, stale, {guard, guard}) == CUBLAS_STATUS_INVALID_VALUE,
               "known retired allocation is not unresolved ignored presence");
    }
}
void application(char** paths, std::optional<unsigned int> malformed, bool is_boundary,
                 Latches& latches) {
    Libraries sdk(paths);
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cublasHandle_t handle{};
    check(cublasCreate(&handle));
    std::array<CUdeviceptr, 4> allocations{};
    for (std::size_t index = 0; index != allocations.size(); ++index) {
        check(cuMemAlloc(&allocations[index], 512));
        expect(allocations[index] == address(index), "independent allocation ledger");
    }
    const auto page_size = ::sysconf(_SC_PAGESIZE);
    expect(page_size > 0, "host scalar page size");
    const auto page_bytes = static_cast<std::size_t>(page_size);
    auto* pages =
        ::mmap(nullptr, page_bytes * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(pages != MAP_FAILED, "scalar guard mapping");
    auto* guard = static_cast<std::byte*>(pages) + page_bytes;
    expect(::mprotect(guard, page_bytes, PROT_NONE) == 0, "unreadable scalar guard");
    const float alpha = 0.5F, beta = -0.25F;
    auto* edge_alpha = reinterpret_cast<float*>(guard - sizeof(float));
    std::memcpy(edge_alpha, &alpha, sizeof(alpha));
    if (is_boundary) {
        boundary_application(sdk, context, handle, allocations, {edge_alpha, &beta},
                             reinterpret_cast<float*>(guard), latches);
        check(cublasDestroy(handle));
        for (auto allocation : allocations)
            if (allocation) check(cuMemFree(allocation));
        check(cuCtxDestroy(context));
    } else if (malformed) {
        const auto test = hostile(*malformed % kReceiptCases);
        Case call;
        call.is_gemm = *malformed >= kReceiptCases;
        expect(invoke(sdk, handle, call, {.alpha = edge_alpha, .beta = &beta}) == test.expected,
               "strict original receipt and quarantine semantics");
        expect(invoke(sdk, handle, call, {.alpha = edge_alpha, .beta = &beta}) ==
                   (test.is_healthy ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED),
               "native errors remain usable; malformed/lost receipts never replay");
        if (test.is_healthy) {
            check(cublasDestroy(handle));
            for (auto allocation : allocations) check(cuMemFree(allocation));
            check(cuCtxDestroy(context));
        }
    } else {
        const auto ordinary = cases();
        for (std::size_t index = 0; index != ordinary.size(); ++index) {
            const auto& test = ordinary[index];
            check(cublasSetPointerMode(
                handle, test.is_device ? CUBLAS_POINTER_MODE_DEVICE : CUBLAS_POINTER_MODE_HOST));
            if (index == 0 || index == 4) {
                latches.saw_command = false;
                latches.release_command = false;
                latches.setter_started = false;
                auto command = std::async(std::launch::async, [&] {
                    check(cuCtxSetCurrent(context));
                    return invoke(sdk, handle, test, {.alpha = edge_alpha, .beta = &beta});
                });
                const auto deadline = Clock::now() + std::chrono::seconds{3};
                while (!latches.saw_command && Clock::now() < deadline) std::this_thread::yield();
                expect(latches.saw_command, "held exact owned command observed");
                auto setter = std::async(std::launch::async, [&] {
                    check(cuCtxSetCurrent(context));
                    latches.setter_started = true;
                    return cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
                });
                while (!latches.setter_started && Clock::now() < deadline)
                    std::this_thread::yield();
                const bool is_blocked =
                    latches.setter_started &&
                    setter.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
                    command.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
                latches.release_command = true;
                check(command.get());
                check(setter.get());
                expect(is_blocked, "both exact command leases exclude handle setters");
            } else
                expect(
                    invoke(sdk, handle, test, {.alpha = edge_alpha, .beta = &beta}) == test.status,
                    "exact original status, scalar ABI and range serialization");
        }
        check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
        for (const bool is_gemm : {false, true}) {
            Case invalid;
            invalid.is_gemm = is_gemm;
            expect(invoke(sdk, handle, invalid,
                          {.alpha = reinterpret_cast<float*>(guard), .beta = &beta}) ==
                       CUBLAS_STATUS_INVALID_VALUE,
                   "active unreadable HOST scalar rejected without memory fault or RPC");
            expect(invoke(sdk, handle, invalid, {.alpha = nullptr, .beta = &beta}) ==
                       CUBLAS_STATUS_NOT_SUPPORTED,
                   "unqualified NULL semantics are explicit policy, not invented native status");
            invalid.buffers[0] = 1;
            expect(invoke(sdk, handle, invalid, {.alpha = &alpha, .beta = &beta}) ==
                       CUBLAS_STATUS_INVALID_VALUE,
                   "unowned device input never becomes native pointer authority");
            expect(invoke(sdk, nullptr, invalid, {.alpha = &alpha, .beta = &beta}) ==
                       CUBLAS_STATUS_NOT_INITIALIZED,
                   "invalid owned handle never falls through to caller library");
        }
        check(cublasDestroy(handle));
        for (auto allocation : allocations) check(cuMemFree(allocation));
        check(cuCtxDestroy(context));
    }
    expect(::munmap(pages, page_bytes * 2) == 0, "release scalar guards");
    expect(glimmer_blas_extension_native_calls() == 0 && glimmer_blas_native_calls() == 0 &&
               glimmer_memory_native_calls() == 0,
           "all direct and cached caller-native traps remain untouched");
}
void respond(ExecutionSocketServer& server, const ExecutionReceivedRequest& request,
             const ExecutionWireReply& reply) {
    auto header = request.message.header;
    header.kind = ExecutionMessageKind::kReply;
    header.session = request.connection;
    header.incarnation = {197, 199};
    auto payload = encode_execution_reply(reply);
    expect(payload && decode_execution_reply(*payload), "structurally valid receipt capsule");
    expect(server.reply(request.connection, {header, std::move(*payload)}), "authenticated reply");
}
void run(char** paths, const std::string& socket, std::optional<unsigned int> malformed,
         bool is_boundary) {
    ExecutionSocketConfig config;
    config.socket_path = socket;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated CPU-only extension service");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "real server process incarnation");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", socket.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "authenticated endpoint configured before frontend first use");
    Latches latches;
    auto client = std::async(std::launch::async,
                             [&] { application(paths, malformed, is_boundary, latches); });
    struct Guard {
        ExecutionSocketServer& server;
        Latches& latches;
        ~Guard() {
            latches.release_command = true;
            server.stop();
        }
    } guard{server, latches};
    std::optional<ExecutionReceivedRequest> pending;
    const auto original = cases();
    const auto ordinary =
        is_boundary ? boundary_cases() : std::vector<Case>(original.begin(), original.end());
    unsigned int hello{}, creates{}, allocations{}, calls{}, releases{}, destroys{}, closes{};
    bool is_disconnected{};
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline, "bounded fake service progress");
        if (is_disconnected) {
            std::this_thread::yield();
            continue;
        }
        expect(server.poll_for(1), "fake service progress");
        if (pending && latches.release_command) {
            expect(pending->message.payload == expected_bytes(ordinary.at(calls - 1)),
                   "held serialized request owns immutable scalar bytes and logical ranges");
            respond(server, *pending,
                    {.disposition = ExecutionWireDisposition::kComplete,
                     .status = 0,
                     .original_status = 0,
                     .id = calls,
                     .arguments = {},
                     .data = {}});
            pending.reset();
        }
        auto request = server.take_request();
        if (!request) continue;
        ExecutionPayloadReader input(request->message.payload);
        ExecutionWireReply reply{.disposition = ExecutionWireDisposition::kComplete,
                                 .status = 0,
                                 .original_status = 0,
                                 .arguments = {},
                                 .data = {}};
        const auto operation = request->message.header.operation;
        if (operation == ExecutionOpcode::kHello) {
            expect(decode_execution_context_bootstrap(request->message.payload) ==
                       ExecutionContextBootstrap{ExecutionContextPurpose::kExplicitUser, 0, 1},
                   "ordinary explicit owner bootstrap");
            ++hello;
        } else if (operation == ExecutionOpcode::kCreateBlas) {
            expect(input.empty(), "create contains no native handle");
            reply.id = 70;
            ++creates;
        } else if (operation == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(input.read_u64(bytes) && bytes == 512 && input.empty() &&
                       allocations < (is_boundary ? 5U : 4U),
                   "exact bounded fake allocation ledger");
            reply.id = 11 + allocations;
            reply.address = address(allocations++);
        } else if (operation == ExecutionOpcode::kSetBlasAttribute) {
            std::uint64_t handle{}, value{};
            std::uint32_t kind{};
            expect(input.read_u64(handle) && handle == 70 && input.read_u32(kind) &&
                       input.read_u64(value) && input.empty() &&
                       ((kind == 1 && value <= 1) || (kind == 2 && value == CUBLAS_DEFAULT_MATH)),
                   "exact owned mode mutation");
            expect(!pending, "no handle setter overtakes the command receipt");
        } else if (operation == ExecutionOpcode::kSgemv || operation == ExecutionOpcode::kSgemmEx) {
            Case expected;
            if (malformed)
                expected.is_gemm = *malformed >= kReceiptCases;
            else
                expected = ordinary.at(calls);
            expect(operation == (expected.is_gemm ? ExecutionOpcode::kSgemmEx
                                                  : ExecutionOpcode::kSgemv) &&
                       request->message.payload == expected_bytes(expected),
                   "independent full literal wire ledger, exact entry and signed base increments");
            reply.id = ++calls;
            if (malformed && calls == 1) {
                const auto test = hostile(*malformed % kReceiptCases);
                if (test.is_disconnect) {
                    server.stop();
                    is_disconnected = true;
                    continue;
                }
                reply = test.reply;
            } else if (!malformed) {
                reply.status = expected.status;
                reply.original_status = reply.status;
                if (is_boundary ? calls == ordinary.size() : calls == 1 || calls == 5) {
                    pending = std::move(*request);
                    latches.saw_command = true;
                    continue;
                }
            }
        } else if (operation == ExecutionOpcode::kRelease) {
            std::uint64_t id{};
            expect(input.read_u64(id) && id >= 11 && id <= (is_boundary ? 15U : 14U) &&
                       input.empty() && !pending,
                   "exact allocation release identity");
            ++releases;
        } else if (operation == ExecutionOpcode::kDestroyBlas) {
            std::uint64_t id{};
            expect(input.read_u64(id) && id == 70 && input.empty(), "exact handle destruction");
            ++destroys;
        } else if (operation == ExecutionOpcode::kClose) {
            expect(input.empty(), "normal owner close");
            ++closes;
        } else
            throw std::runtime_error("unexpected RPC, staging, replay or substituted BLAS entry");
        respond(server, *request, reply);
    }
    client.get();
    const bool is_healthy = !malformed || hostile(*malformed % kReceiptCases).is_healthy;
    expect(hello == (is_boundary ? 2U : 1U) && creates == 1 &&
               allocations == (is_boundary ? 5U : 4U) &&
               calls == (malformed ? is_healthy ? 2U : 1U : ordinary.size()) &&
               releases == (is_healthy ? is_boundary ? 5U : 4U : 0U) &&
               destroys == (is_healthy ? 1U : 0U) &&
               closes == (is_healthy ? is_boundary ? 2U : 1U : 0U),
           "exact once counts; healthy cases clean normally, uncertain owners remain isolated");
    server.stop();
}
int one(char** paths, std::optional<unsigned int> malformed, bool is_boundary = false) {
    char pattern[] = "/tmp/glimmer-blas-extension-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return 1;
    int status{};
    try {
        run(paths, std::string(directory) + "/service.sock", malformed, is_boundary);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    if (::rmdir(directory) != 0) status = 1;
    return status;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 6 && std::string_view(argv[4]) == "--receipt-case") {
        unsigned int index{};
        const std::string_view text(argv[5]);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), index);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
            index >= kReceiptCases * 2)
            return 1;
        return one(argv, index);
    }
    if (argc != 4 && argc != 5) return 1;
    if (argc == 4) return one(argv, std::nullopt);
    if (std::string_view(argv[4]) == "--boundary") return one(argv, std::nullopt, true);
    if (std::string_view(argv[4]) != "--malformed-replies") return 1;
    for (unsigned int index = 0; index != kReceiptCases * 2; ++index) {
        const auto child = ::fork();
        if (child < 0) return 1;
        if (!child) {
            auto number = std::to_string(index);
            std::array<char*, 7> arguments{
                argv[0],       argv[1], argv[2], argv[3], const_cast<char*>("--receipt-case"),
                number.data(), nullptr};
            ::execv(argv[0], arguments.data());
            ::_exit(127);
        }
        int status{};
        if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status)) {
            std::cerr << "isolated BLAS extension receipt case " << index << '\n';
            return 1;
        }
    }
    std::cout << "46 BLAS extension receipts; caller_native_calls=0; GPU_qualified=false\n";
    return 0;
}
