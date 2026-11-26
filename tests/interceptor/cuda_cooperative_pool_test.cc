#include "internal/cuda_cooperative_pool.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::interceptor;
const auto kContext = reinterpret_cast<CUcontext>(0x1000);
CUcontext current_context = kContext;
unsigned int native_calls{};
unsigned int device_frees{};
unsigned int host_frees{};
std::size_t last_copy_bytes{};
CUdeviceptr last_copy_source{};
CUstream last_copy_stream{};
CUresult device_allocate_error{CUDA_SUCCESS};
CUresult host_allocate_error{CUDA_SUCCESS};
CUresult map_error{CUDA_SUCCESS};
CUresult device_free_error{CUDA_SUCCESS};
CUresult host_free_error{CUDA_SUCCESS};
CUresult copy_error{CUDA_SUCCESS};

void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
CUresult context(CUcontext* output) {
    ++native_calls;
    *output = current_context;
    return CUDA_SUCCESS;
}
CUresult allocate(CUdeviceptr* output, std::size_t memory_bytes) {
    ++native_calls;
    if (device_allocate_error != CUDA_SUCCESS) {
        return device_allocate_error;
    }
    *output = reinterpret_cast<CUdeviceptr>(new std::byte[memory_bytes]);
    return CUDA_SUCCESS;
}
CudaCooperativeReleaseResult free_device(CUdeviceptr pointer) {
    ++native_calls;
    if (device_free_error != CUDA_SUCCESS) {
        return {device_free_error, CudaNativeReleaseState::kRetained};
    }
    delete[] std::bit_cast<std::byte*>(pointer);
    ++device_frees;
    return {CUDA_SUCCESS, CudaNativeReleaseState::kReleased};
}
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): fixed CUDA native ABI.
CUresult allocate_host(void** output, std::size_t memory_bytes, unsigned int flags) {
    ++native_calls;
    require(flags == CU_MEMHOSTALLOC_DEVICEMAP, "host state must be mapped and pinned");
    if (host_allocate_error != CUDA_SUCCESS) {
        return host_allocate_error;
    }
    *output = ::operator new(memory_bytes, std::align_val_t{16});
    return CUDA_SUCCESS;
}
CudaCooperativeReleaseResult free_host(void* pointer) {
    ++native_calls;
    if (host_free_error != CUDA_SUCCESS) {
        return {host_free_error, CudaNativeReleaseState::kRetained};
    }
    ::operator delete(pointer, std::align_val_t{16});
    ++host_frees;
    return {CUDA_SUCCESS, CudaNativeReleaseState::kReleased};
}
CUresult map(CUdeviceptr* output, void* pointer, unsigned int flags) {
    ++native_calls;
    require(flags == 0, "native map flags must be zero");
    *output = reinterpret_cast<CUdeviceptr>(pointer);
    return map_error;
}
CUresult copy(void* destination, CUdeviceptr source, std::size_t memory_bytes, CUstream stream) {
    ++native_calls;
    last_copy_source = source;
    last_copy_bytes = memory_bytes;
    last_copy_stream = stream;
    if (copy_error == CUDA_SUCCESS) {
        std::memcpy(destination, std::bit_cast<void*>(source), memory_bytes);
    }
    return copy_error;
}
CudaCooperativePoolDispatch dispatch() {
    return {context,   allocate, free_device, allocate_host,
            free_host, map,      copy,        std::make_shared<int>(1)};
}
CooperativeKernelVariant variant() {
    return {kContext, reinterpret_cast<CUfunction>(0x2000), {{8, 8}, {4, 4}}, {105, 64}, 3};
}
template <typename T>
T parameter(const PreparedCooperativeKernel& kernel, std::size_t index) {
    T value{};
    std::memcpy(&value, kernel.kernel.arguments.addresses()[index], sizeof(value));
    return value;
}
void ownership_and_progress() {
    auto pool = std::make_shared<CudaCooperativePool>(kContext, dispatch(), 1);
    auto metadata = variant();
    std::uint64_t pointer = 0x7777;
    std::uint32_t work = 123;
    std::array<void*, 2> addresses{&pointer, &work};
    const auto owner = std::make_shared<int>(2);
    require(!pool->prepare(metadata, {7, 5, 3}, {8, 4, 2}, addresses.data(), owner),
            "uninitialized pool must not accept work");
    require(pool->initialize() == CUDA_SUCCESS && pool->initialize() == CUDA_SUCCESS,
            "pool initialization must be idempotent");
    const auto calls_before = native_calls;
    auto prepared = pool->prepare(metadata, {7, 5, 3}, {8, 4, 2}, addresses.data(), owner);
    require(prepared && native_calls == calls_before && pool->occupied_slots() == 1,
            "prepare must own a bounded slot without native allocation or synchronization");
    require(!pool->prepare(metadata, {7, 5, 3}, {8, 4, 2}, addresses.data(), owner),
            "full pool must reject before physical issue");
    pointer = 0;
    work = 0;
    addresses.fill(nullptr);
    require(parameter<std::uint64_t>(*prepared, 0) == 0x7777 &&
                parameter<std::uint32_t>(*prepared, 1) == 123 &&
                parameter<std::uint32_t>(*prepared, 2) == 7 &&
                parameter<std::uint32_t>(*prepared, 3) == 5 &&
                parameter<std::uint32_t>(*prepared, 4) == 3,
            "caller and private ABI values must survive caller mutation");
    auto counter = parameter<CUdeviceptr>(*prepared, 5);
    auto retreat = parameter<CUdeviceptr>(*prepared, 6);
    require(counter == prepared->initialization.destination &&
                prepared->initialization.memory_bytes == sizeof(std::uint64_t) &&
                *static_cast<std::uint64_t*>(prepared->initialization.host_pointer) == 0,
            "counter initialization must be an owned pinned ordered copy");
    std::uint64_t partial = 4;
    std::memcpy(std::bit_cast<void*>(counter), &partial, sizeof(partial));
    const auto stream = reinterpret_cast<CUstream>(0x3000);
    std::array<unsigned int, 3> physical{};
    auto continuation = prepared->continuation;
    require(continuation->prepare_launch(physical, prepared->kernel.arguments) == CUDA_SUCCESS &&
                physical == std::array<unsigned int, 3>{3, 1, 1} &&
                continuation->request_retreat() == CUDA_SUCCESS &&
                std::atomic_ref(*std::bit_cast<std::uint32_t*>(retreat)).load() == 1,
            "continuation must request retreat without a GPU call");
    bool is_complete = true;
    require(continuation->stage_completion(stream) == CUDA_SUCCESS &&
                continuation->inspect_stopped(is_complete) == CUDA_SUCCESS && !is_complete &&
                last_copy_bytes == 8 && last_copy_source == counter && last_copy_stream == stream,
            "same-stream readback must distinguish partial stop from completion");
    require(continuation->prepare_launch(physical, prepared->kernel.arguments) == CUDA_SUCCESS &&
                std::atomic_ref(*std::bit_cast<std::uint32_t*>(retreat)).load() == 0,
            "resume must clear only retreat, preserving the claim counter");
    std::memcpy(&partial, std::bit_cast<void*>(counter), sizeof(partial));
    require(partial == 4, "resume must never reset partial claims");
    partial = 108;
    std::memcpy(std::bit_cast<void*>(counter), &partial, sizeof(partial));
    require(continuation->stage_completion(stream) == CUDA_SUCCESS &&
                continuation->inspect_stopped(is_complete) == CUDA_SUCCESS && is_complete &&
                continuation->progress().partial_blocks == 4 &&
                continuation->progress().physical_launches == 2,
            "logical completion must retain the observed physical retreat evidence");
    require(pool->release() == CUDA_ERROR_NOT_READY, "live progress may not be freed");
    prepared.reset();
    require(pool->occupied_slots() == 1, "continuation alone must retain its slot");
    const auto calls_before_return = native_calls;
    continuation.reset();
    require(pool->occupied_slots() == 0 && native_calls == calls_before_return,
            "resource destruction must return metadata only");
    require(pool->release() == CUDA_SUCCESS && pool->release() == CUDA_SUCCESS &&
                pool->initialize() == CUDA_ERROR_INVALID_VALUE,
            "explicit release must be checked and close the pool permanently");
}
void audited_progress() {
    auto pool = std::make_shared<CudaCooperativePool>(kContext, dispatch(), 1, 8);
    require(pool->initialize() == CUDA_SUCCESS, "initialize audit pool");
    CooperativeKernelVariant metadata{kContext,
                                      reinterpret_cast<CUfunction>(0x2000),
                                      {{8, 8}, {8, 8}},
                                      {8, 64},
                                      3,
                                      CooperativeKernelVariant::AuditParameters{0, 1, true}};
    std::uint64_t ignored = 0x7777;
    std::array<void*, 2> addresses{&ignored, &ignored};
    auto owner = std::make_shared<int>(1);
    const auto prepare = [&] {
        return pool->prepare(metadata, {8, 1, 1}, {64, 1, 1}, addresses.data(), owner);
    };
    metadata.audit->tile_counts = 2;
    require(!prepare(), "reject out-of-range private audit argument");
    metadata.audit->tile_counts = 0;
    require(!prepare(), "reject aliased private audit arguments");
    metadata.audit->tile_counts = 1;
    metadata.parameter_layout[0].size_bytes = 4;
    require(!prepare(), "reject wrong private audit ABI");
    metadata.parameter_layout[0].size_bytes = 8;
    const auto calls = native_calls;
    auto prepared = prepare();
    require(prepared && native_calls == calls, "audit preparation must remain metadata-only");
    const auto started = parameter<CUdeviceptr>(*prepared, 0);
    auto* compute = std::bit_cast<std::uint32_t*>(started) + 1;
    auto* counts = std::bit_cast<std::uint32_t*>(parameter<CUdeviceptr>(*prepared, 1));
    auto* counter = std::bit_cast<std::uint64_t*>(parameter<CUdeviceptr>(*prepared, 5));
    require(started != ignored && reinterpret_cast<CUdeviceptr>(counts) != ignored &&
                std::all_of(counts, counts + 8, [](auto value) { return value == 0; }),
            "pool must replace borrowed audit pointers with zeroed owned storage");
    std::array<unsigned int, 3> grid{};
    auto& continuation = *prepared->continuation;
    continuation.execution_granted({41, 1});
    require(continuation.prepare_launch(grid, prepared->kernel.arguments) == CUDA_SUCCESS,
            "prepare audit launch");
    require(std::atomic_ref(*compute).load() == 0, "new compute observation must start empty");
    std::atomic_ref(*compute).store(0x42000001U);
    continuation.observe_pending();
    continuation.observe_pending();
    std::fill_n(counts, 4, 1U);
    *counter = 4;
    std::atomic_ref(*std::bit_cast<std::uint32_t*>(started)).store(1);
    continuation.observe_pending();
    bool complete = true;
    const auto inspect = [&] {
        return continuation.stage_completion(nullptr) == CUDA_SUCCESS
                   ? continuation.inspect_stopped(complete)
                   : CUDA_ERROR_UNKNOWN;
    };
    require(inspect() == CUDA_SUCCESS && !complete, "validate exact partial tile coverage");
    continuation.execution_granted({41, 2});
    require(continuation.prepare_launch(grid, prepared->kernel.arguments) == CUDA_SUCCESS &&
                std::atomic_ref(*std::bit_cast<std::uint32_t*>(started)).load() == 0 &&
                std::atomic_ref(*compute).load() == 0 &&
                std::all_of(counts, counts + 4, [](auto value) { return value == 1; }) &&
                *counter == 4,
            "resume must reset observation but preserve tile counts and claims");
    std::fill_n(counts + 4, 4, 1U);
    *counter = 11;
    require(inspect() == CUDA_SUCCESS && complete,
            "validate all tiles with bounded claim overshoot");
    counts[0] = 2;
    counts[1] = 0;
    require(inspect() == CUDA_ERROR_UNKNOWN,
            "reject duplicate/missing tiles even with an equal sum");
    counts[0] = 1;
    require(inspect() == CUDA_ERROR_UNKNOWN, "reject missing final tile");
    counts[1] = 1;
    *counter = 4;
    require(inspect() == CUDA_ERROR_UNKNOWN, "reject count/claim disagreement");
    *counter = 12;
    require(inspect() == CUDA_ERROR_UNKNOWN, "reject excessive claim overshoot");
    prepared.reset();
    prepared = prepare();
    counts = std::bit_cast<std::uint32_t*>(parameter<CUdeviceptr>(*prepared, 1));
    require(std::all_of(counts, counts + 8, [](auto value) { return value == 0; }),
            "reused slot must not inherit old tile counts");
    prepared.reset();
    require(pool->release() == CUDA_SUCCESS, "release audited host/device storage");
}

void rejection_and_reuse() {
    auto pool = std::make_shared<CudaCooperativePool>(kContext, dispatch(), 2);
    require(pool->initialize() == CUDA_SUCCESS, "initialize rejection pool");
    auto metadata = variant();
    std::uint64_t pointer = 1;
    std::uint32_t work = 1;
    std::array<void*, 2> addresses{&pointer, &work};
    auto owner = std::make_shared<int>(1);
    auto prepare = [&] {
        return pool->prepare(metadata, {7, 5, 3}, {8, 4, 2}, addresses.data(), owner);
    };
    metadata.context = nullptr;
    require(!prepare(), "foreign context must reject");
    metadata = variant();
    metadata.worker_blocks = 106;
    require(!prepare(), "invalid worker bounds must reject");
    metadata = variant();
    metadata.shape.logical_blocks = 104;
    require(!prepare(), "shape mismatch must reject");
    metadata = variant();
    addresses[1] = nullptr;
    require(!prepare() && pool->occupied_slots() == 0,
            "argument-copy failure must roll back its slot");
    addresses[1] = &work;
    auto first = prepare();
    auto second = prepare();
    require(
        first && second && parameter<CUdeviceptr>(*first, 5) != parameter<CUdeviceptr>(*second, 5),
        "simultaneous logical kernels must have independent counters");
    const auto prior_counter = parameter<CUdeviceptr>(*first, 5);
    first.reset();
    auto reused = prepare();
    require(reused && parameter<CUdeviceptr>(*reused, 5) == prior_counter,
            "a fully released slot must be reusable");
    copy_error = CUDA_ERROR_INVALID_CONTEXT;
    require(reused->continuation->stage_completion(nullptr) == copy_error,
            "native asynchronous progress failure must propagate unchanged");
    copy_error = CUDA_SUCCESS;
    reused.reset();
    second.reset();
    current_context = nullptr;
    require(pool->release() == CUDA_ERROR_INVALID_CONTEXT, "foreign-context release must not free");
    current_context = kContext;
    require(pool->release() == CUDA_SUCCESS, "release reusable pool");
}
void partial_initialization_and_free() {
    for (unsigned int failure = 0; failure < 3; ++failure) {
        auto pool = std::make_shared<CudaCooperativePool>(kContext, dispatch(), 1);
        device_allocate_error = failure == 0 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
        host_allocate_error = failure == 1 ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
        map_error = failure == 2 ? CUDA_ERROR_NOT_SUPPORTED : CUDA_SUCCESS;
        require(pool->initialize() != CUDA_SUCCESS, "partial native failure must propagate");
        device_allocate_error = host_allocate_error = map_error = CUDA_SUCCESS;
        require(pool->release() == CUDA_SUCCESS, "partial owner must support explicit cleanup");
    }
    auto pool = std::make_shared<CudaCooperativePool>(kContext, dispatch(), 1);
    require(pool->initialize() == CUDA_SUCCESS, "initialize failed-free pool");
    auto prior_device_frees = device_frees;
    auto prior_host_frees = host_frees;
    device_free_error = CUDA_ERROR_INVALID_CONTEXT;
    require(pool->release() == device_free_error && device_frees == prior_device_frees &&
                host_frees == prior_host_frees,
            "failed device free must retain both allocations");
    device_free_error = CUDA_SUCCESS;
    host_free_error = CUDA_ERROR_INVALID_VALUE;
    require(pool->release() == host_free_error && device_frees == prior_device_frees + 1 &&
                host_frees == prior_host_frees,
            "partial release must retain only the still-owned host allocation");
    host_free_error = CUDA_SUCCESS;
    require(pool->release() == CUDA_SUCCESS && device_frees == prior_device_frees + 1 &&
                host_frees == prior_host_frees + 1,
            "retry must never free the device counter twice");
    auto native = dispatch();
    native.allocate_device = [](CUdeviceptr*, std::size_t) -> CUresult {
        throw std::runtime_error("injected accounting callback exception");
    };
    auto throwing = std::make_shared<CudaCooperativePool>(kContext, std::move(native), 1);
    require(throwing->initialize() == CUDA_ERROR_UNKNOWN && throwing->release() == CUDA_SUCCESS,
            "backend callback exceptions must remain contained");

    auto released_error_dispatch = dispatch();
    released_error_dispatch.free_device = [](CUdeviceptr pointer) {
        const auto result = free_device(pointer);
        require(result.ownership == CudaNativeReleaseState::kReleased,
                "physical release fixture must actually free");
        return CudaCooperativeReleaseResult{CUDA_ERROR_UNKNOWN, CudaNativeReleaseState::kReleased};
    };
    auto released_error =
        std::make_shared<CudaCooperativePool>(kContext, std::move(released_error_dispatch), 1);
    require(released_error->initialize() == CUDA_SUCCESS, "initialize post-release error fixture");
    prior_device_frees = device_frees;
    require(released_error->release() == CUDA_ERROR_UNKNOWN &&
                released_error->release() == CUDA_SUCCESS && device_frees == prior_device_frees + 1,
            "accounting error after proven release must not replay the native free");
}

void uncertain_release_never_replays() {
    const auto child = ::fork();
    require(child >= 0, "uncertain native ownership fixture must fork");
    if (child == 0) {
        auto native = dispatch();
        native.free_device = [](CUdeviceptr pointer) {
            const auto result = free_device(pointer);
            return CudaCooperativeReleaseResult{CUDA_ERROR_UNKNOWN,
                                                result.status == CUDA_SUCCESS
                                                    ? CudaNativeReleaseState::kUncertain
                                                    : CudaNativeReleaseState::kRetained};
        };
        auto pool = std::make_shared<CudaCooperativePool>(kContext, std::move(native), 1);
        if (pool->initialize() != CUDA_SUCCESS) {
            ::_exit(EXIT_FAILURE);
        }
        const auto device_count = device_frees;
        const auto host_count = host_frees;
        const auto first = pool->release();
        const auto second = pool->release();
        pool.reset();
        ::_exit(first == CUDA_ERROR_UNKNOWN && second == CUDA_ERROR_UNKNOWN &&
                        device_frees == device_count + 1 && host_frees == host_count
                    ? EXIT_SUCCESS
                    : EXIT_FAILURE);
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
            "an uncertain native free must remain quarantined and never be replayed");
}

void unproven_destruction_quarantines() {
    const auto child = ::fork();
    require(child >= 0, "quarantine fixture must fork");
    if (child == 0) {
        auto pool = std::make_shared<CudaCooperativePool>(kContext, dispatch(), 1);
        if (pool->initialize() != CUDA_SUCCESS) {
            ::_exit(EXIT_FAILURE);
        }
        const auto calls = native_calls;
        const auto device_count = device_frees;
        const auto host_count = host_frees;
        pool.reset();
        // No physical stop/release was supplied. Process exit, not an inferred
        // last-owner proof, retires this intentionally quarantined fake owner.
        ::_exit(native_calls == calls && device_frees == device_count && host_frees == host_count
                    ? EXIT_SUCCESS
                    : EXIT_FAILURE);
    }
    int status = 0;
    pid_t waited = -1;
    do {
        waited = ::waitpid(child, &status, 0);
    } while (waited < 0 && errno == EINTR);
    require(waited == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS,
            "unproven pool destruction must not call CUDA or free native resources");
}
}  // namespace

int main() {
    try {
        ownership_and_progress();
        audited_progress();
        rejection_and_reuse();
        partial_initialization_and_free();
        uncertain_release_never_replays();
        unproven_destruction_quarantines();
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
