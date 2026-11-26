#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" unsigned int glimmer_memory_native_calls();
extern "C" unsigned int glimmer_blas_native_calls();
extern "C" unsigned int glimmer_lt_native_calls();
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
using GemmEx = cublasStatus_t(CUBLASWINAPI*)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                             int, int, int, const void*, const void*, cudaDataType,
                                             int, const void*, cudaDataType, int, const void*,
                                             void*, cudaDataType, int, cublasComputeType_t,
                                             cublasGemmAlgo_t);
using StridedGemmEx = cublasStatus_t(CUBLASWINAPI*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*,
    cudaDataType, int, long long, const void*, cudaDataType, int, long long, const void*, void*,
    cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned Driver control succeeds");
}
void check(cublasStatus_t value) {
    expect(value == CUBLAS_STATUS_SUCCESS, "owned original cuBLAS call succeeds");
}
template <typename Function>
Function symbol(void* handle, const char* name) {
    auto* value = ::dlsym(handle, name);
    if (!value) throw std::runtime_error(std::string("missing owned cuBLAS export: ") + name);
    return reinterpret_cast<Function>(value);
}
struct Latches {
    std::atomic<bool> saw_launch{}, release_launch{};
    std::atomic<bool> saw_teardown_query{}, release_teardown_query{};
    std::atomic<bool> saw_extended{}, release_extended{};
    std::atomic<unsigned int> attribute_queries{};
};
void query_output_ranges(cublasHandle_t handle, Latches& latches) {
    const auto page = ::sysconf(_SC_PAGESIZE);
    expect(page > 0, "query output page extent");
    const auto page_bytes = static_cast<std::size_t>(page);
    // A long retained pathname must not enter the fixed-size VMA header.
    const auto name = std::string("glimmer-query-output-") + std::string(220, 'x');
    const auto descriptor = static_cast<int>(::syscall(SYS_memfd_create, name.c_str(), 1U));
    expect(descriptor >= 0, "create long named mapping fixture");
    struct Mapping {
        int descriptor;
        std::size_t bytes;
        void* address{MAP_FAILED};
        ~Mapping() {
            if (address != MAP_FAILED && ::munmap(address, bytes) != 0) std::abort();
            if (::close(descriptor) != 0) std::abort();
        }
    } mapping{descriptor, page_bytes * 3};
    expect(::ftruncate(descriptor, static_cast<off_t>(mapping.bytes)) == 0,
           "size named mapping fixture");
    mapping.address =
        ::mmap(nullptr, mapping.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    expect(mapping.address != MAP_FAILED, "map adjacent query outputs");
    auto* first = static_cast<std::byte*>(mapping.address);
    auto* second = first + page_bytes;
    auto* output = reinterpret_cast<int*>(second - sizeof(int) / 2);
    expect(::mprotect(second, page_bytes, PROT_WRITE) == 0, "distinct adjacent writable VMAs");
    check(cublasGetVersion_v2(handle, output));
    expect(::mprotect(second, page_bytes, PROT_READ) == 0, "revoke suffix write permission");
    int version{};
    std::memcpy(&version, output, sizeof(version));
    expect(version == 120605, "complete publication crosses adjacent writable VMAs");
    const auto queries = latches.attribute_queries.load();
    expect(cublasGetVersion_v2(handle, output) == CUBLAS_STATUS_INVALID_VALUE &&
               cublasGetVersion_v2(handle, reinterpret_cast<int*>(second)) ==
                   CUBLAS_STATUS_INVALID_VALUE &&
               cublasGetVersion_v2(handle, nullptr) == CUBLAS_STATUS_INVALID_VALUE &&
               // Deliberately invalid output: the frontend must not dereference it.
               cublasGetVersion_v2(
                   handle,
                   reinterpret_cast<int*>(UINTPTR_MAX - 1)) ==  // NOLINT(performance-no-int-to-ptr)
                   CUBLAS_STATUS_INVALID_VALUE &&
               latches.attribute_queries.load() == queries,
           "fresh maps reject readonly, partial, null and overflowing outputs before RPC");
    expect(::munmap(second, page_bytes) == 0, "create an unmapped suffix");
    expect(cublasGetVersion_v2(handle, output) == CUBLAS_STATUS_INVALID_VALUE &&
               latches.attribute_queries.load() == queries,
           "a gap cannot authorize a partial output or submit a query");
    check(cublasGetVersion_v2(handle, reinterpret_cast<int*>(first)));
    expect(latches.attribute_queries.load() == queries + 1,
           "range rejection leaves subsequent valid queries healthy");
    // Train a single-VMA search position, then revoke its permission. A hint
    // must never carry authorization across a mapping-state change.
    check(cublasGetVersion_v2(handle, reinterpret_cast<int*>(first)));
    const auto trained_queries = latches.attribute_queries.load();
    expect(::mprotect(first, page_bytes, PROT_READ) == 0, "revoke trained VMA write permission");
    expect(
        cublasGetVersion_v2(handle, reinterpret_cast<int*>(first)) == CUBLAS_STATUS_INVALID_VALUE &&
            latches.attribute_queries.load() == trained_queries,
        "a trained hint cannot authorize readonly output or issue an RPC");
    expect(::mprotect(first, page_bytes, PROT_READ | PROT_WRITE) == 0, "restore trained VMA");
    check(cublasGetVersion_v2(handle, reinterpret_cast<int*>(first)));
}
void long_query_output(cublasHandle_t handle, Latches& latches) {
    // A real maps line longer than the bounded hint fragment must take the
    // complete scanner. Relative opens allow the fixture to exceed PATH_MAX.
    struct Directories {
        std::array<char, 40> root{};
        std::string component = "segment-" + std::string(200, 'x');
        std::vector<int> descriptors;
        ~Directories() {
            if (!descriptors.empty()) {
                ::unlinkat(descriptors.back(), "mapping", 0);
                for (std::size_t index = descriptors.size() - 1; index > 0; --index) {
                    ::close(descriptors[index]);
                    ::unlinkat(descriptors[index - 1], component.c_str(), AT_REMOVEDIR);
                }
                ::close(descriptors.front());
            }
            if (root[0]) ::rmdir(root.data());
        }
    } directories;
    std::strcpy(directories.root.data(), "/tmp/glimmer-query-long-XXXXXX");
    expect(::mkdtemp(directories.root.data()), "long query path root");
    const auto root = ::open(directories.root.data(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    expect(root >= 0, "open long query root");
    directories.descriptors.push_back(root);
    for (int index = 0; index < 20; ++index) {
        expect(::mkdirat(directories.descriptors.back(), directories.component.c_str(), 0700) == 0,
               "create long query component");
        const auto child = ::openat(directories.descriptors.back(), directories.component.c_str(),
                                    O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        expect(child >= 0, "open long query component");
        directories.descriptors.push_back(child);
    }
    ExecutionOwnedDescriptor file(::openat(directories.descriptors.back(), "mapping",
                                           O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
    const auto page = ::sysconf(_SC_PAGESIZE);
    expect(page > 0 && file.get() >= 0 && ::ftruncate(file.get(), page) == 0,
           "create long query mapped file");
    struct Mapping {
        void* address;
        std::size_t bytes;
        ~Mapping() {
            if (address != MAP_FAILED) ::munmap(address, bytes);
        }
    } mapping{::mmap(nullptr, static_cast<std::size_t>(page), PROT_READ | PROT_WRITE, MAP_SHARED,
                     file.get(), 0),
              static_cast<std::size_t>(page)};
    expect(mapping.address != MAP_FAILED, "map long query path");
    {
        std::ifstream maps("/proc/self/maps");
        std::string line;
        bool found{};
        while (std::getline(maps, line))
            found |= line.size() > 4096 && line.find(directories.root.data()) != std::string::npos;
        expect(found, "actual long mapping line exceeds hint fragment");
    }
    check(cublasGetVersion_v2(handle, static_cast<int*>(mapping.address)));
    check(cublasGetVersion_v2(handle, static_cast<int*>(mapping.address)));
    const auto queries = latches.attribute_queries.load();
    expect(::mprotect(mapping.address, mapping.bytes, PROT_READ) == 0, "revoke long query mapping");
    expect(cublasGetVersion_v2(handle, static_cast<int*>(mapping.address)) ==
                   CUBLAS_STATUS_INVALID_VALUE &&
               latches.attribute_queries.load() == queries,
           "long-line fallback preserves rejection before RPC");
    auto* retired = static_cast<int*>(mapping.address);
    expect(::munmap(mapping.address, mapping.bytes) == 0, "unmap long query output");
    mapping.address = MAP_FAILED;
    expect(cublasGetVersion_v2(handle, retired) == CUBLAS_STATUS_INVALID_VALUE &&
               latches.attribute_queries.load() == queries,
           "long-line stale position cannot authorize unmapped output");
}
void host_range_overhead(cublasHandle_t handle) {
    if (!std::getenv("GLIMMER_TEST_HOST_RANGE_OVERHEAD")) return;
    const auto page = ::sysconf(_SC_PAGESIZE);
    expect(page > 0, "overhead fixture page size");
    const auto page_bytes = static_cast<std::size_t>(page);
    struct Mapping {
        void* address;
        std::size_t bytes;
        ~Mapping() {
            if (address != MAP_FAILED && ::munmap(address, bytes) != 0) std::abort();
        }
    } mapping{::mmap(nullptr, page_bytes * 4098, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0),
              page_bytes * 4098};
    expect(mapping.address != MAP_FAILED, "bounded many-VMA fixture allocation");
    auto* pages = static_cast<std::byte*>(mapping.address) + page_bytes;
    expect(::mprotect(pages, page_bytes * 4096, PROT_READ | PROT_WRITE) == 0,
           "guarded writable fixture interior");
    for (std::size_t index = 0; index < 4096; index += 2)
        expect(::mprotect(pages + index * page_bytes, page_bytes, PROT_READ) == 0,
               "alternating VMA permissions");
    const auto verify_layout = [&]() {
        std::ifstream maps("/proc/self/maps");
        expect(maps.is_open(), "open independent fixture layout verification");
        const auto first = reinterpret_cast<std::uintptr_t>(mapping.address);
        std::size_t matched{};
        std::string line;
        while (std::getline(maps, line)) {
            std::istringstream fields(line);
            std::uintptr_t left{}, right{};
            char separator{};
            std::string permissions;
            expect(static_cast<bool>(fields >> std::hex >> left >> separator >> right >>
                                     permissions) &&
                       separator == '-' && left < right,
                   "parse independent VMA layout");
            if (right <= first || left >= first + mapping.bytes) continue;
            const auto expected = matched == 0 || matched == 4097 ? "---p"
                                  : matched % 2 == 1              ? "r--p"
                                                                  : "rw-p";
            expect(matched < 4098 && left == first + matched * page_bytes &&
                       right == left + page_bytes && permissions == expected,
                   "exact guarded one-page alternating VMA layout");
            ++matched;
        }
        expect(maps.eof() && matched == 4098, "complete independent VMA layout receipt");
    };
    verify_layout();
    int version{};
    expect(reinterpret_cast<std::uintptr_t>(&version) >=
               reinterpret_cast<std::uintptr_t>(mapping.address) + mapping.bytes,
           "stack output must follow all fixture VMAs");
    for (unsigned int index = 0; index < 20; ++index) {
        check(cublasGetVersion_v2(handle, &version));
        expect(version == 120605, "warmup getter output");
    }
    const auto begin = Clock::now();
    for (unsigned int index = 0; index < 200; ++index) {
        check(cublasGetVersion_v2(handle, &version));
        expect(version == 120605, "measured getter output");
    }
    const auto end = Clock::now();
    verify_layout();
    std::cout
        << "Host range overhead: begin_ns="
        << std::chrono::duration_cast<std::chrono::nanoseconds>(begin.time_since_epoch()).count()
        << "; end_ns="
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end.time_since_epoch()).count()
        << "; thread=" << ::gettid() << "; page_bytes=" << page_bytes
        << "; vma_pages=4096; verified_guards=2; layout_checks=2; warmup=20; measured=200; "
           "successes=200\n";
}
void application(const char* driver_path, const char* blas_path, bool quarantine,
                 Latches& latches) {
    auto* driver = ::dlopen(driver_path, RTLD_NOW | RTLD_LOCAL);
    auto* provider = ::dlopen(blas_path, RTLD_NOW | RTLD_LOCAL);
    expect(driver && provider, "load metadata-only original SDK fixtures");
    Dl_info lt_information{};
    expect(::dladdr(reinterpret_cast<void*>(&glimmer_lt_native_calls), &lt_information) != 0 &&
               lt_information.dli_fname,
           "locate the independently linked original Lt provider");
    auto* lt_provider = ::dlopen(lt_information.dli_fname, RTLD_NOW | RTLD_LOCAL);
    expect(lt_provider != nullptr, "retain original Lt lookup provider");
    const auto create_lt = symbol<decltype(&cublasLtCreate)>(lt_provider, "cublasLtCreate");
    const auto destroy_lt = symbol<decltype(&cublasLtDestroy)>(lt_provider, "cublasLtDestroy");
    expect(create_lt == &cublasLtCreate && destroy_lt == &cublasLtDestroy &&
               reinterpret_cast<void*>(create_lt) == ::dlsym(RTLD_DEFAULT, "cublasLtCreate") &&
               reinterpret_cast<void*>(destroy_lt) == ::dlsym(RTLD_DEFAULT, "cublasLtDestroy"),
           "direct/global/explicit Lt handle ABI identity");
#define SELECT(NAME)                                                             \
    const auto NAME##_call = symbol<decltype(&(NAME))>(provider, #NAME);         \
    expect(reinterpret_cast<void*>(NAME##_call) == ::dlsym(RTLD_DEFAULT, #NAME), \
           "explicit/global library lookup identity")
    SELECT(cublasCreate_v2);
    SELECT(cublasDestroy_v2);
    SELECT(cublasGetVersion_v2);
    SELECT(cublasSetStream_v2);
    SELECT(cublasGetStream_v2);
    SELECT(cublasSgemm_v2);
    SELECT(cublasSetPointerMode_v2);
    SELECT(cublasGetPointerMode_v2);
    SELECT(cublasSetMathMode);
    SELECT(cublasGetMathMode);
    SELECT(cublasSetWorkspace_v2);
    SELECT(cublasSetAtomicsMode);
    SELECT(cublasGetAtomicsMode);
    SELECT(cublasSetSmCountTarget);
    SELECT(cublasGetSmCountTarget);
    SELECT(cublasSgemmStridedBatched);
#undef SELECT
    const auto gemm_ex = symbol<GemmEx>(provider, "cublasGemmEx");
    expect(reinterpret_cast<void*>(gemm_ex) == ::dlsym(RTLD_DEFAULT, "cublasGemmEx"),
           "extended explicit/global symbol identity");
    const auto strided = symbol<StridedGemmEx>(provider, "cublasGemmStridedBatchedEx");
    expect(reinterpret_cast<void*>(strided) == ::dlsym(RTLD_DEFAULT, "cublasGemmStridedBatchedEx"),
           "strided explicit/global symbol identity");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cublasHandle_t handle{};
    if (quarantine) {
        expect(cublasCreate_v2_call(&handle) == CUBLAS_STATUS_INTERNAL_ERROR && !handle &&
                   cublasCreate_v2_call(&handle) == CUBLAS_STATUS_INTERNAL_ERROR &&
                   cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN,
               "uncertain create retains its origin without publishing a handle or retrying");
        return;
    }
    cublasLtHandle_t lt{};
    expect(create_lt(nullptr) == CUBLAS_STATUS_INVALID_VALUE &&
               destroy_lt(nullptr) == CUBLAS_STATUS_NOT_INITIALIZED &&
               create_lt(&lt) == CUBLAS_STATUS_ALLOC_FAILED && !lt,
           "Lt handle validation and actual known-empty SDK error preserve caller output");
    check(create_lt(&lt));
    expect(lt && glimmer_lt_native_calls() == 0,
           "native Lt handle is owned remotely, never created in the caller");
    expect(cublasDestroy_v2_call(reinterpret_cast<cublasHandle_t>(lt)) ==
               CUBLAS_STATUS_NOT_INITIALIZED,
           "Lt identity cannot be used as a cuBLAS handle");
    check(destroy_lt(lt));
    expect(destroy_lt(lt) == CUBLAS_STATUS_NOT_INITIALIZED,
           "retired Lt identity cannot replay destruction or fall back to the caller SDK");
    expect(cublasCreate_v2_call(nullptr) == CUBLAS_STATUS_INVALID_VALUE,
           "invalid output rejects before native library creation");
    expect(cublasCreate_v2_call(&handle) == CUBLAS_STATUS_ALLOC_FAILED && !handle,
           "known-empty native rejection retains the actual library error and unchanged output");
    check(cublasCreate_v2_call(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cublasSetStream_v2_call(handle, reinterpret_cast<cudaStream_t>(stream)));
    cudaStream_t queried{};
    check(cublasGetStream_v2_call(handle, &queried));
    expect(queried == reinterpret_cast<cudaStream_t>(stream),
           "local/remote stream identities agree");
    cublasPointerMode_t pointer_mode{};
    check(cublasGetPointerMode_v2_call(handle, &pointer_mode));
    expect(pointer_mode == CUBLAS_POINTER_MODE_HOST, "original default host pointer mode");
    check(cublasSetMathMode_call(handle, CUBLAS_PEDANTIC_MATH));
    expect(
        cublasSetMathMode_call(handle, static_cast<cublasMath_t>(5)) == CUBLAS_STATUS_INVALID_VALUE,
        "original invalid math status is not a Driver enum or fabricated success");
    cublasMath_t math{};
    check(cublasGetMathMode_call(handle, &math));
    expect(math == CUBLAS_PEDANTIC_MATH, "native rejected mutation keeps its previous state");
    const auto extent = ::sysconf(_SC_PAGESIZE);
    expect(extent > 0, "page extent");
    auto* write_only = ::mmap(nullptr, static_cast<std::size_t>(extent), PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(write_only != MAP_FAILED, "write-only query output");
    check(cublasGetVersion_v2_call(handle, static_cast<int*>(write_only)));
    expect(::mprotect(write_only, static_cast<std::size_t>(extent), PROT_READ) == 0 &&
               *static_cast<int*>(write_only) == 120605 &&
               ::munmap(write_only, static_cast<std::size_t>(extent)) == 0,
           "library outputs do not require reading their old bytes");
    query_output_ranges(handle, latches);
    long_query_output(handle, latches);
    host_range_overhead(handle);
    std::array<CUdeviceptr, 4> allocations{};
    for (auto& pointer : allocations) check(cuMemAlloc(&pointer, 256));
    check(cublasSetWorkspace_v2_call(handle, reinterpret_cast<void*>(allocations[3]), 256));
    expect(cublasSetWorkspace_v2_call(handle, reinterpret_cast<void*>(allocations[3] + 1), 32) ==
                   CUBLAS_STATUS_INVALID_VALUE &&
               cublasSetWorkspace_v2_call(handle, reinterpret_cast<void*>(allocations[3] + 256),
                                          0) == CUBLAS_STATUS_INVALID_VALUE &&
               cublasSetWorkspace_v2_call(handle, nullptr, 16) == CUBLAS_STATUS_INVALID_VALUE,
           "alignment and owned-extent workspace failures do not invoke caller SDK resources");
    check(cublasSetWorkspace_v2_call(handle, nullptr, 0));
    check(cublasSetWorkspace_v2_call(handle, reinterpret_cast<void*>(allocations[3]), 0));
    check(cublasSetStream_v2_call(handle, reinterpret_cast<cudaStream_t>(stream)));
    cublasAtomicsMode_t atomics{};
    check(cublasSetAtomicsMode_call(handle, CUBLAS_ATOMICS_ALLOWED));
    check(cublasGetAtomicsMode_call(handle, &atomics));
    expect(atomics == CUBLAS_ATOMICS_ALLOWED &&
               cublasGetAtomicsMode_call(handle, reinterpret_cast<cublasAtomicsMode_t*>(1)) ==
                   CUBLAS_STATUS_INVALID_VALUE,
           "original atomics mode returns through validated write-only outputs");
    int sm_count{};
    check(cublasSetSmCountTarget_call(handle, 7));
    expect(cublasSetSmCountTarget_call(handle, 81) == CUBLAS_STATUS_INVALID_VALUE &&
               cublasSetSmCountTarget_call(handle, -1) == CUBLAS_STATUS_INVALID_VALUE,
           "SDK and local SM-hint rejections retain their actual error domains");
    check(cublasGetSmCountTarget_call(handle, &sm_count));
    expect(sm_count == 7, "rejected SM mutation leaves the original hint unchanged");
    check(cublasSetSmCountTarget_call(handle, 0));
    float alpha{2.0F}, beta{0.5F};
    // All synthetic device values are opaque ABI addresses, never host-read.
    // NOLINTBEGIN(performance-no-int-to-ptr)
    const auto* a = reinterpret_cast<const float*>(allocations[0] + 4);
    const auto* b = reinterpret_cast<const float*>(allocations[1] + 8);
    auto* c = reinterpret_cast<float*>(allocations[2] + 12);
    auto pending = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return cublasSgemm_v2_call(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, &alpha, a, 2, b, 2,
                                   &beta, c, 2);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (!latches.saw_launch.load(std::memory_order_acquire) && Clock::now() < deadline)
        std::this_thread::yield();
    const bool was_withheld =
        latches.saw_launch &&
        pending.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
    int version{};
    const auto query = cublasGetVersion_v2_call(handle, &version);
    alpha = beta = 99.0F;
    latches.release_launch.store(true, std::memory_order_release);
    expect(pending.get() == CUBLAS_STATUS_EXECUTION_FAILED && was_withheld &&
               query == CUBLAS_STATUS_SUCCESS && version == 120605,
           "original SDK result is withheld while independent metadata and immutable wire inputs "
           "progress");
    check(cublasSetPointerMode_v2_call(handle, CUBLAS_POINTER_MODE_DEVICE));
    check(cublasGetPointerMode_v2_call(handle, &pointer_mode));
    expect(pointer_mode == CUBLAS_POINTER_MODE_DEVICE, "actual device pointer mode query");
    check(cublasSgemm_v2_call(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2,
                              reinterpret_cast<const float*>(allocations[3] + 4), a, 2, b, 2,
                              reinterpret_cast<const float*>(allocations[3] + 8), c, 2));
    expect(cublasSgemm_v2_call(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2,
                               reinterpret_cast<const float*>(allocations[3] + 4), a, 2, b, 2,
                               reinterpret_cast<const float*>(allocations[3] + 8), c,
                               2) == CUBLAS_STATUS_EXECUTION_FAILED,
           "actual replacement Driver launch failure maps to library execution failure without an "
           "original cuBLAS claim");
    expect(cublasSgemm_v2_call(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, &alpha, a, 2, b, 2, &beta,
                               c, 2) == CUBLAS_STATUS_INVALID_VALUE,
           "device pointer mode cannot host-read scalar values or accept unowned addresses");
    check(cublasSgemm_v2_call(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2,
                              reinterpret_cast<const float*>(allocations[3] + 4),
                              reinterpret_cast<const float*>(allocations[0]), 2,
                              reinterpret_cast<const float*>(allocations[1]), 2,
                              reinterpret_cast<const float*>(allocations[3] + 8),
                              reinterpret_cast<float*>(allocations[2]), 2));
    // NOLINTEND(performance-no-int-to-ptr)
    // NOLINTBEGIN(performance-no-int-to-ptr): opaque synthetic GPU ABI addresses.
    check(cublasSetPointerMode_v2_call(handle, CUBLAS_POINTER_MODE_HOST));
    check(cublasSetMathMode_call(handle, CUBLAS_DEFAULT_MATH));
    float ex_alpha = 2.0F, ex_beta = 0.5F;
    auto extended = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return gemm_ex(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, &ex_alpha,
                       reinterpret_cast<const void*>(allocations[0]), CUDA_R_16F, 2,
                       reinterpret_cast<const void*>(allocations[1]), CUDA_R_16F, 2, &ex_beta,
                       reinterpret_cast<void*>(allocations[2]), CUDA_R_16F, 2, CUBLAS_COMPUTE_32F,
                       CUBLAS_GEMM_DEFAULT);
    });
    const auto extended_deadline = Clock::now() + std::chrono::seconds{2};
    while (!latches.saw_extended.load(std::memory_order_acquire) &&
           Clock::now() < extended_deadline)
        std::this_thread::yield();
    const bool extended_pending =
        extended.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
    const auto metadata_status = cublasGetVersion_v2_call(handle, &version);
    ex_alpha = ex_beta = 99.0F;
    latches.release_extended.store(true, std::memory_order_release);
    expect(extended.get() == CUBLAS_STATUS_NOT_SUPPORTED && extended_pending &&
               metadata_status == CUBLAS_STATUS_SUCCESS,
           "extended original error waits for actual reply while immutable coefficients and "
           "independent metadata progress");
    const std::uint16_t half_alpha = 0x4000, half_beta = 0x3800;
    check(gemm_ex(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, &half_alpha,
                  reinterpret_cast<const void*>(allocations[0]), CUDA_R_16F, 2,
                  reinterpret_cast<const void*>(allocations[1]), CUDA_R_16F, 2, &half_beta,
                  reinterpret_cast<void*>(allocations[2]), CUDA_R_16F, 2, CUBLAS_COMPUTE_16F,
                  CUBLAS_GEMM_ALGO3));
    const std::array<double, 2> complex_alpha{2.0, 3.0}, complex_beta{0.5, -1.0};
    check(gemm_ex(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, complex_alpha.data(),
                  reinterpret_cast<const void*>(allocations[0]), CUDA_C_64F, 2,
                  reinterpret_cast<const void*>(allocations[1]), CUDA_C_64F, 2, complex_beta.data(),
                  reinterpret_cast<void*>(allocations[2]), CUDA_C_64F, 2, CUBLAS_COMPUTE_64F,
                  CUBLAS_GEMM_ALGO3));
    check(cublasSetPointerMode_v2_call(handle, CUBLAS_POINTER_MODE_DEVICE));
    check(gemm_ex(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2,
                  reinterpret_cast<const void*>(allocations[3] + 16),
                  reinterpret_cast<const void*>(allocations[0]), CUDA_C_64F, 2,
                  reinterpret_cast<const void*>(allocations[1]), CUDA_C_64F, 2,
                  reinterpret_cast<const void*>(allocations[3] + 64),
                  reinterpret_cast<void*>(allocations[2]), CUDA_C_64F, 2, CUBLAS_COMPUTE_64F,
                  CUBLAS_GEMM_ALGO3));
    check(gemm_ex(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, reinterpret_cast<const void*>(allocations[3]),
        reinterpret_cast<const void*>(allocations[0]), CUDA_R_16F, 2,
        reinterpret_cast<const void*>(allocations[1]), CUDA_R_16F, 2,
        reinterpret_cast<const void*>(allocations[3] + 4), reinterpret_cast<void*>(allocations[2]),
        CUDA_R_16F, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    check(cublasSetPointerMode_v2_call(handle, CUBLAS_POINTER_MODE_HOST));
    ex_alpha = 2.0F;
    ex_beta = 0.5F;
    expect(
        strided(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, &ex_alpha,
                reinterpret_cast<const void*>(allocations[0]), CUDA_R_16F, 2, 0,
                reinterpret_cast<const void*>(allocations[1]), CUDA_R_16F, 2, 16, &ex_beta,
                reinterpret_cast<void*>(allocations[2]), CUDA_R_16F, 2, 16, 3, CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_NOT_SUPPORTED,
        "strided call retains the actual original library error, not invented replacement success");
    check(cublasSetPointerMode_v2_call(handle, CUBLAS_POINTER_MODE_DEVICE));
    expect(strided(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2,
                   reinterpret_cast<const void*>(allocations[3] + 16),
                   reinterpret_cast<const void*>(allocations[0] + 128), CUDA_C_64F, 2, -4,
                   reinterpret_cast<const void*>(allocations[1] + 128), CUDA_C_64F, 2, -4,
                   reinterpret_cast<const void*>(allocations[3] + 64),
                   reinterpret_cast<void*>(allocations[2] + 128), CUDA_C_64F, 2, -4, 3,
                   CUBLAS_COMPUTE_64F, CUBLAS_GEMM_ALGO3) == CUBLAS_STATUS_NOT_SUPPORTED,
           "known-unsafe reverse strides reject before native owner submission");
    check(strided(handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2,
                  reinterpret_cast<const void*>(allocations[3] + 16),
                  reinterpret_cast<const void*>(allocations[0]), CUDA_C_64F, 2, 4,
                  reinterpret_cast<const void*>(allocations[1]), CUDA_C_64F, 2, 4,
                  reinterpret_cast<const void*>(allocations[3] + 64),
                  reinterpret_cast<void*>(allocations[2]), CUDA_C_64F, 2, 4, 3, CUBLAS_COMPUTE_64F,
                  CUBLAS_GEMM_ALGO3));
    check(strided(
        handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, 2, 2, reinterpret_cast<const void*>(allocations[3]),
        reinterpret_cast<const void*>(allocations[0]), CUDA_R_16F, 2, 8,
        reinterpret_cast<const void*>(allocations[1]), CUDA_R_16F, 2, 12,
        reinterpret_cast<const void*>(allocations[3] + 4), reinterpret_cast<void*>(allocations[2]),
        CUDA_R_16F, 2, 16, 3, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    for (const auto mode : {CUBLAS_POINTER_MODE_HOST, CUBLAS_POINTER_MODE_DEVICE}) {
        check(cublasSetPointerMode_v2_call(handle, mode));
        const auto* av = mode == CUBLAS_POINTER_MODE_HOST
                             ? &ex_alpha
                             : reinterpret_cast<const float*>(allocations[3]);
        const auto* bv = mode == CUBLAS_POINTER_MODE_HOST
                             ? &ex_beta
                             : reinterpret_cast<const float*>(allocations[3] + 4);
        const auto call = [&](long long stride, cublasOperation_t ta = CUBLAS_OP_N) {
            return cublasSgemmStridedBatched_call(
                handle, ta, CUBLAS_OP_T, 2, 2, 2, av,
                reinterpret_cast<const float*>(allocations[0]), 2, stride,
                reinterpret_cast<const float*>(allocations[1]), 2, 12, bv,
                reinterpret_cast<float*>(allocations[2]), 2, 16, 3);
        };
        expect(call(-8) == CUBLAS_STATUS_NOT_SUPPORTED,
               "unsafe classic reverse strides reject before submission");
        expect(call(8, static_cast<cublasOperation_t>(999)) == CUBLAS_STATUS_INVALID_VALUE,
               "invalid external transpose bits reject before wire submission or enum evaluation");
        const auto status = call(8);
        expect(status == (mode == CUBLAS_POINTER_MODE_HOST ? CUBLAS_STATUS_NOT_SUPPORTED
                                                           : CUBLAS_STATUS_SUCCESS),
               "classic batch preserves original error or distinct private Driver-compute success");
    }
    // NOLINTEND(performance-no-int-to-ptr)
    check(cublasDestroy_v2_call(handle));
    expect(cublasGetVersion_v2_call(handle, &version) == CUBLAS_STATUS_NOT_INITIALIZED,
           "retired handles do not fall back into caller-native cuBLAS");
    for (const auto pointer : allocations) check(cuMemFree(pointer));
    check(cuStreamDestroy(stream));
    check(cublasCreate_v2_call(&handle));
    auto old_query = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        int value{};
        const auto status = cublasGetVersion_v2_call(handle, &value);
        return status == CUBLAS_STATUS_SUCCESS && value == 120605;
    });
    const auto query_deadline = Clock::now() + std::chrono::seconds{2};
    while (!latches.saw_teardown_query.load(std::memory_order_acquire) &&
           Clock::now() < query_deadline)
        std::this_thread::yield();
    auto closing = std::async(std::launch::async, [&] { return cuCtxDestroy(context); });
    bool saw_seal{};
    const auto seal_deadline = Clock::now() + std::chrono::seconds{1};
    while (Clock::now() < seal_deadline) {
        int value{};
        if (cublasGetVersion_v2_call(handle, &value) == CUBLAS_STATUS_NOT_INITIALIZED) {
            saw_seal = true;
            break;
        }
        std::this_thread::yield();
    }
    // Always release the old reply, including the counterfactual old-frontend
    // failure, so a failed assertion cannot strand an owned async caller.
    const bool was_pending =
        closing.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready;
    latches.release_teardown_query.store(true, std::memory_order_release);
    const bool old_completed = old_query.get();
    const auto closed = closing.get();
    expect(saw_seal && was_pending && old_completed && closed == CUDA_SUCCESS,
           "context close must seal new library queries before waiting for the finite old query");
    expect(cublasGetVersion_v2_call(handle, &version) == CUBLAS_STATUS_NOT_INITIALIZED &&
               glimmer_blas_native_calls() == 0 && glimmer_memory_native_calls() == 0 &&
               glimmer_lt_native_calls() == 0,
           "origin invalidation retains caller transparency without native resource operations");
    expect(::dlclose(lt_provider) == 0 && ::dlclose(provider) == 0 && ::dlclose(driver) == 0,
           "balance SDK fixture references");
}
void respond(ExecutionSocketServer& server, const ExecutionReceivedRequest& request,
             const ExecutionWireReply& reply) {
    auto header = request.message.header;
    header.kind = ExecutionMessageKind::kReply;
    header.session = request.connection;
    header.incarnation = {131, 137};
    auto bytes = encode_execution_reply(reply);
    expect(bytes && server.reply(request.connection, {header, std::move(*bytes)}),
           "authenticated original library reply");
}
void run(const std::string& path, const char* driver, const char* blas, bool quarantine) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated library endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual endpoint incarnation");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "bind exact authenticated service before frontend initialization");
    Latches latches;
    auto client =
        std::async(std::launch::async, [&] { application(driver, blas, quarantine, latches); });
    struct EndpointGuard {
        ExecutionSocketServer& server;
        Latches& latches;
        ~EndpointGuard() {
            latches.release_launch.store(true, std::memory_order_release);
            latches.release_extended.store(true, std::memory_order_release);
            latches.release_teardown_query.store(true, std::memory_order_release);
            server.stop();
        }
    } guard{server, latches};
    unsigned int creates{}, launches{}, allocations{}, destroys{}, closes{}, extended_calls{},
        batch_calls{}, classic_batch_calls{}, lt_creates{}, lt_destroys{};
    std::uint64_t stream{}, pointer_mode{}, math{}, atomics{}, sm_count{};
    std::optional<ExecutionReceivedRequest> pending, pending_query, pending_extended;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < deadline, "bounded library endpoint progress");
        if (pending && latches.release_launch.load(std::memory_order_acquire)) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.status = CUBLAS_STATUS_EXECUTION_FAILED;
            reply.original_status = reply.status;
            reply.id = 1;
            respond(server, *pending, reply);
            pending.reset();
        }
        if (pending_query && latches.release_teardown_query.load(std::memory_order_acquire)) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = 0;
            reply.address = 120605;
            respond(server, *pending_query, reply);
            pending_query.reset();
        }
        if (pending_extended && latches.release_extended.load(std::memory_order_acquire)) {
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            reply.original_status = CUBLAS_STATUS_NOT_SUPPORTED;
            reply.status = CUBLAS_STATUS_NOT_SUPPORTED;
            reply.id = 4;
            respond(server, *pending_extended, reply);
            pending_extended.reset();
        }
        auto request = server.take_request();
        if (!request) continue;
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        reply.original_status = 0;
        ExecutionPayloadReader input(request->message.payload);
        const auto operation = request->message.header.operation;
        if (operation == ExecutionOpcode::kCreateLt) {
            expect(input.empty(), "Lt creation contains no caller SDK handle");
            ++lt_creates;
            if (lt_creates == 1)
                reply.status = *reply.original_status = CUBLAS_STATUS_ALLOC_FAILED;
            else
                reply.id = 90;
        } else if (operation == ExecutionOpcode::kDestroyLt) {
            std::uint64_t handle{};
            expect(input.read_u64(handle) && handle == 90 && input.empty(),
                   "Lt destruction contains only the authenticated owned identity");
            ++lt_destroys;
        } else if (operation == ExecutionOpcode::kCreateBlas) {
            expect(input.empty(), "no caller-native handle in create");
            ++creates;
            if (quarantine) {
                reply.disposition = ExecutionWireDisposition::kRejected;
                reply.status = CUDA_ERROR_UNKNOWN;
            } else if (creates == 1) {
                reply.status = CUBLAS_STATUS_ALLOC_FAILED;
                reply.original_status = reply.status;
            } else
                reply.id = 70 + creates - 2;
        } else if (operation == ExecutionOpcode::kAllocate) {
            std::uint64_t bytes{};
            expect(input.read_u64(bytes) && input.empty() && bytes == 256,
                   "actual owned allocation request");
            reply.id = 11 + allocations;
            reply.address = 0x100000 + std::uint64_t{allocations++} * 0x100;
        } else if (operation == ExecutionOpcode::kCreateStream) {
            reply.id = 61;
        } else if (operation == ExecutionOpcode::kSetBlasWorkspace) {
            std::uint64_t handle{}, allocation{}, offset{}, bytes{};
            expect(input.read_u64(handle) && handle == 70 && input.read_u64(allocation) &&
                       input.read_u64(offset) && input.read_u64(bytes) && input.empty() &&
                       ((allocation == 0 && offset == 0 && bytes == 0) ||
                        (allocation == 14 && offset < 256 && bytes <= 256 - offset)),
                   "workspace message contains an authenticated allocation interval, never a "
                   "caller pointer");
            if (allocation && offset % 256) {
                reply.status = CUBLAS_STATUS_INVALID_VALUE;
                reply.original_status = reply.status;
            }
        } else if (operation == ExecutionOpcode::kSetBlasAttribute ||
                   operation == ExecutionOpcode::kGetBlasAttribute) {
            if (operation == ExecutionOpcode::kGetBlasAttribute) ++latches.attribute_queries;
            std::uint64_t handle{}, value{};
            std::uint32_t kind{};
            expect(input.read_u64(handle) && (handle == 70 || handle == 71) &&
                       input.read_u32(kind) && kind <= 5,
                   "scoped library identity and explicit attribute kind");
            if (handle == 71 && !latches.saw_teardown_query.load(std::memory_order_acquire)) {
                expect(
                    operation == ExecutionOpcode::kGetBlasAttribute && kind == 3 && input.empty(),
                    "hold only the exact old teardown metadata query");
                pending_query = std::move(*request);
                latches.saw_teardown_query.store(true, std::memory_order_release);
                continue;
            }
            if (operation == ExecutionOpcode::kSetBlasAttribute) {
                expect(input.read_u64(value) && input.empty(), "owned setter value");
                if (kind == 0) {
                    expect(value == 61, "server-owned stream ID");
                    stream = value;
                } else if (kind == 1) {
                    expect(value <= 1, "valid scalar mode");
                    pointer_mode = value;
                } else if (kind == 4) {
                    expect(value <= 1, "valid original atomics mode");
                    atomics = value;
                } else if (kind == 5) {
                    if (value > static_cast<std::uint64_t>(INT_MAX)) {
                        reply.status = CUDA_ERROR_INVALID_VALUE;
                        reply.disposition = ExecutionWireDisposition::kRejected;
                        reply.original_status.reset();
                    } else if (value > 80) {
                        reply.status = CUBLAS_STATUS_INVALID_VALUE;
                        reply.original_status = reply.status;
                    } else
                        sm_count = value;
                } else if (value == 5) {
                    reply.status = CUBLAS_STATUS_INVALID_VALUE;
                    reply.original_status = reply.status;
                } else {
                    expect(value == CUBLAS_PEDANTIC_MATH || value == CUBLAS_DEFAULT_MATH,
                           "exact supported original math mode");
                    math = value;
                }
            } else {
                expect(input.empty(), "query contains no output pointer");
                reply.address = kind == 0   ? stream
                                : kind == 1 ? pointer_mode
                                : kind == 2 ? math
                                : kind == 4 ? atomics
                                : kind == 5 ? sm_count
                                            : 120605;
            }
        } else if (operation == ExecutionOpcode::kStridedGemmEx ||
                   operation == ExecutionOpcode::kStridedSgemm) {
            const bool classic = operation == ExecutionOpcode::kStridedSgemm;
            if (classic)
                ++classic_batch_calls;
            else
                ++batch_calls;
            const std::array<std::int64_t, 3> strides =
                classic            ? std::array<std::int64_t, 3>{8, 12, 16}
                : batch_calls == 1 ? std::array<std::int64_t, 3>{0, 16, 16}
                : batch_calls == 2 ? std::array<std::int64_t, 3>{4, 4, 4}
                                   : std::array<std::int64_t, 3>{8, 12, 16};
            for (const auto stride : strides) {
                std::uint64_t bits{};
                expect(input.read_u64(bits) && std::bit_cast<std::int64_t>(bits) == stride,
                       "original signed element batch strides");
            }
            std::uint32_t batches{}, a{}, b{}, c{}, compute{}, algorithm{}, ta{}, tb{}, mode{};
            std::uint64_t handle{};
            const auto type = classic ? CUDA_R_32F : batch_calls == 2 ? CUDA_C_64F : CUDA_R_16F;
            expect(input.read_u32(batches) && batches == 3 && input.read_u32(a) && a == type &&
                       input.read_u32(b) && b == a && input.read_u32(c) && c == a &&
                       input.read_u32(compute) &&
                       compute == (!classic && batch_calls == 2 ? CUBLAS_COMPUTE_64F
                                                                : CUBLAS_COMPUTE_32F) &&
                       input.read_u32(algorithm) &&
                       std::bit_cast<std::int32_t>(algorithm) ==
                           (!classic && batch_calls == 2 ? CUBLAS_GEMM_ALGO3 : -1) &&
                       input.read_u64(handle) && handle == 70 && input.read_u32(ta) && ta == 0 &&
                       input.read_u32(tb) && tb == 1,
                   "typed batched metadata and owned handle");
            for (unsigned int field = 0; field < 6; ++field) {
                std::uint32_t value{};
                expect(input.read_u32(value) && value == 2, "original batch matrix geometry");
            }
            for (std::uint64_t operand = 0; operand < 3; ++operand) {
                std::uint64_t allocation{}, offset{};
                expect(input.read_u64(allocation) && allocation == 11 + operand &&
                           input.read_u64(offset) && offset == 0,
                       "positive-size batch start identity, including adjacent bases");
            }
            expect(input.read_u32(mode) &&
                       mode == ((classic ? classic_batch_calls : batch_calls) == 1 ? 0U : 1U) &&
                       mode == pointer_mode,
                   "native batch scalar mode");
            if (!mode) {
                std::span<const std::byte> av, bv;
                expect(
                    input.read_blob(av) && av.size() == 4 && input.read_blob(bv) && bv.size() == 4,
                    "copied batch host coefficient width");
                float alpha{}, beta{};
                std::memcpy(&alpha, av.data(), 4);
                std::memcpy(&beta, bv.data(), 4);
                expect(alpha == 2.0F && beta == 0.5F, "exact host batch coefficients");
            } else {
                for (const auto expected : !classic && batch_calls == 2
                                               ? std::array<std::uint64_t, 2>{16, 64}
                                               : std::array<std::uint64_t, 2>{0, 4}) {
                    std::uint64_t allocation{}, offset{};
                    expect(input.read_u64(allocation) && allocation == 14 &&
                               input.read_u64(offset) && offset == expected,
                           "typed batch coefficients stay on the device");
                }
            }
            expect(input.empty(), "no borrowed or trailing batch payload");
            reply.id = 9 + batch_calls;
            if ((classic ? classic_batch_calls : batch_calls) == 1)
                reply.status = *reply.original_status = CUBLAS_STATUS_NOT_SUPPORTED;
            if (classic ? classic_batch_calls == 2 : batch_calls == 3) {
                reply.disposition = ExecutionWireDisposition::kComputeReplacement;
                reply.original_status.reset();
            }
        } else if (operation == ExecutionOpcode::kGemmEx) {
            ++extended_calls;
            std::uint32_t a{}, b{}, c{}, compute{}, algorithm{}, ta{}, tb{}, mode{};
            std::uint64_t handle{};
            const bool complex = extended_calls == 3 || extended_calls == 4;
            const auto expected_type = complex ? CUDA_C_64F : CUDA_R_16F;
            const auto expected_compute = extended_calls == 2 ? CUBLAS_COMPUTE_16F
                                          : complex           ? CUBLAS_COMPUTE_64F
                                                              : CUBLAS_COMPUTE_32F;
            expect(input.read_u32(a) && input.read_u32(b) && input.read_u32(c) &&
                       a == expected_type && b == a && c == a && input.read_u32(compute) &&
                       compute == expected_compute && input.read_u32(algorithm) &&
                       std::bit_cast<std::int32_t>(algorithm) ==
                           (extended_calls == 1 || extended_calls == 5 ? -1 : CUBLAS_GEMM_ALGO3) &&
                       input.read_u64(handle) && handle == 70 && input.read_u32(ta) && ta == 0 &&
                       input.read_u32(tb) && tb == 1,
                   "exact extended datatype, compute, algorithm and original owned handle");
            for (int field = 0; field < 6; ++field) {
                std::uint32_t value{};
                expect(input.read_u32(value) && value == 2, "original extended matrix geometry");
            }
            for (std::uint64_t operand = 0; operand < 3; ++operand) {
                std::uint64_t allocation{}, offset{};
                expect(input.read_u64(allocation) && allocation == 11 + operand &&
                           input.read_u64(offset) && offset == 0,
                       "owned typed matrix ranges");
            }
            expect(input.read_u32(mode) && mode == (extended_calls >= 4 ? 1U : 0U) &&
                       mode == pointer_mode,
                   "actual extended scalar mode");
            if (!mode) {
                std::span<const std::byte> av, bv;
                const std::size_t width = extended_calls == 1 ? 4U : extended_calls == 2 ? 2U : 16U;
                expect(input.read_blob(av) && input.read_blob(bv) && av.size() == width &&
                           bv.size() == width,
                       "compute/output-dependent raw scalar widths");
                if (extended_calls == 1) {
                    float alpha{}, beta{};
                    std::memcpy(&alpha, av.data(), 4);
                    std::memcpy(&beta, bv.data(), 4);
                    expect(alpha == 2.0F && beta == 0.5F,
                           "original float coefficient bytes before caller reuse");
                } else if (extended_calls == 2) {
                    std::uint16_t alpha{}, beta{};
                    std::memcpy(&alpha, av.data(), 2);
                    std::memcpy(&beta, bv.data(), 2);
                    expect(alpha == 0x4000 && beta == 0x3800,
                           "half coefficients are not converted to float bytes");
                } else {
                    std::array<double, 2> alpha{}, beta{};
                    std::memcpy(alpha.data(), av.data(), 16);
                    std::memcpy(beta.data(), bv.data(), 16);
                    expect(alpha == std::array<double, 2>{2.0, 3.0} &&
                               beta == std::array<double, 2>{0.5, -1.0},
                           "complete complex-double coefficient bytes");
                }
            } else {
                for (const auto expected : extended_calls == 4
                                               ? std::array<std::uint64_t, 2>{16, 64}
                                               : std::array<std::uint64_t, 2>{0, 4}) {
                    std::uint64_t allocation{}, offset{};
                    expect(input.read_u64(allocation) && allocation == 14 &&
                               input.read_u64(offset) && offset == expected,
                           "device coefficients are owned ranges, never CPU-read");
                }
            }
            expect(input.empty(), "no extra or borrowed extended data");
            if (extended_calls == 1) {
                pending_extended = std::move(*request);
                latches.saw_extended.store(true, std::memory_order_release);
                continue;
            }
            reply.id = extended_calls + 3;
            if (extended_calls == 5) {
                reply.disposition = ExecutionWireDisposition::kComputeReplacement;
                reply.original_status.reset();
            }
        } else if (operation == ExecutionOpcode::kSgemm) {
            ++launches;
            std::uint64_t handle{};
            std::uint32_t ta{}, tb{}, mode{};
            expect(input.read_u64(handle) && handle == 70 && input.read_u32(ta) && ta == 0 &&
                       input.read_u32(tb) && tb == 1,
                   "original handle and transpose flags");
            for (int i = 0; i < 6; ++i) {
                std::uint32_t value{};
                expect(input.read_u32(value) && value == 2, "exact signed matrix fields");
            }
            for (std::uint64_t i = 0; i < 3; ++i) {
                std::uint64_t allocation{}, offset{};
                expect(
                    input.read_u64(allocation) && allocation == 11 + i && input.read_u64(offset) &&
                        offset == (launches == 4 ? 0 : (i + 1) * 4),
                    "owned matrix extents cross IPC as IDs/offsets, never native caller handles");
            }
            expect(input.read_u32(mode) && mode == pointer_mode,
                   "host/device scalar ABI matches actual handle state");
            if (!mode) {
                std::uint32_t av{}, bv{};
                expect(input.read_u32(av) && input.read_u32(bv) && input.empty() &&
                           std::bit_cast<float>(av) == 2.0F && std::bit_cast<float>(bv) == 0.5F,
                       "host scalar bits are independently copied before native issue");
                pending = std::move(*request);
                latches.saw_launch.store(true, std::memory_order_release);
                continue;
            }
            for (std::uint64_t offset : {4U, 8U}) {
                std::uint64_t allocation{}, actual{};
                expect(input.read_u64(allocation) && allocation == 14 && input.read_u64(actual) &&
                           actual == offset,
                       "device scalar pointers retain their owned allocation and byte offsets");
            }
            expect(input.empty(), "no borrowed/native data follows scalar fields");
            reply.id = launches;
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            if (launches == 3) reply.status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
        } else if (operation == ExecutionOpcode::kDestroyBlas) {
            ++destroys;
        } else if (operation == ExecutionOpcode::kClose) {
            ++closes;
        } else {
            expect(operation == ExecutionOpcode::kHello || operation == ExecutionOpcode::kRelease ||
                       operation == ExecutionOpcode::kDestroyStream,
                   "no unexpected native fallback or extra API");
        }
        respond(server, *request, reply);
    }
    client.get();
    expect(
        creates == (quarantine ? 1U : 3U) && launches == (quarantine ? 0U : 4U) &&
            destroys == (quarantine ? 0U : 1U) && closes == (quarantine ? 0U : 1U) &&
            extended_calls == (quarantine ? 0U : 5U) && batch_calls == (quarantine ? 0U : 3U) &&
            classic_batch_calls == (quarantine ? 0U : 2U) && lt_creates == (quarantine ? 0U : 2U) &&
            lt_destroys == (quarantine ? 0U : 1U),
        "original API operations execute once, invalid/uncertain/retired operations execute none");
    server.stop();
    std::cout << "Owned public cuBLAS ABI verified: native_caller_calls=0; quarantine="
              << std::boolalpha << quarantine
              << "; scalar_modes=2; sdk_result=true; gpu_qualified=false\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc < 3) return EXIT_FAILURE;
    if (argc == 5 && std::string_view(argv[3]) == "--require-export-provider") {
        auto* provider = ::dlopen(argv[4], RTLD_NOW | RTLD_LOCAL);
        if (!provider || !::dlsym(provider, "cublasCreate_v2")) {
            std::cerr << "missing owned cuBLAS export: cublasCreate_v2\n";
            return EXIT_FAILURE;
        }
        if (::dlclose(provider) != 0) return EXIT_FAILURE;
    }
    const bool quarantine = argc == 4 && std::string_view(argv[3]) == "--quarantine";
    char pattern[] = "/tmp/glimmer-blas-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int result = EXIT_SUCCESS;
    try {
        run(std::string(directory) + "/service.sock", argv[1], argv[2], quarantine);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = EXIT_FAILURE;
    }
    if (::rmdir(directory) != 0) result = EXIT_FAILURE;
    return result;
}
