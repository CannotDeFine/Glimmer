#include "internal/cuda_binary_slice.h"
#include "internal/allocation_registry.h"
#include "internal/cuda_copy_range.h"
#include "internal/qualified_cubin_source.h"
#include "internal/persistent_kernel.h"
#include "gemm_slice_ptx_fixture.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace {
using namespace glimmer::interceptor;
constexpr std::size_t kGuard = 32;
constexpr std::uint32_t kCanary = 0x4b123456;
bool cleanup_ok = true;
std::size_t mapped_cases{}, partial_observations{}, cancelled_cases{}, independent_copies{};
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
}
void cleanup(CUresult result, const char* operation) noexcept {
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "%s: %d\n", operation, static_cast<int>(result));
        cleanup_ok = false;
    }
}
struct Owner {
    CUcontext context{};
    CUmodule module{}, mapped_module{};
    CUlibrary library{}, mapped_library{};
    std::unique_ptr<DriverDispatch> driver;
    std::unique_ptr<PersistentKernelRuntime> runtime;
    std::shared_ptr<void> private_owner;
    CUfunction owned_function{};
    QualifiedGemmMapping mapping{QualifiedGemmMapping::kPlane};
    std::array<CUdeviceptr, 4> buffers{};
    CUstream low{}, high{};
    CUdeviceptr independent{};
    ~Owner() {
        for (auto buffer : buffers)
            if (buffer) cleanup(cuMemFree(buffer), "free");
        if (independent) cleanup(cuMemFree(independent), "independent free");
        if (low) cleanup(cuStreamDestroy(low), "low destroy");
        if (high) cleanup(cuStreamDestroy(high), "high destroy");
        if (mapped_library)
            cleanup(cuLibraryUnload(mapped_library), "mapped library unload");
        else if (mapped_module)
            cleanup(cuModuleUnload(mapped_module), "mapped unload");
        if (library) {
            cleanup(cuLibraryUnload(library), "original library unload");
            if (runtime) runtime->retire_library_source(library);
        } else if (module) {
            cleanup(runtime ? runtime->unload_module(module) : cuModuleUnload(module), "unload");
        }
        private_owner.reset();
        if (runtime && context)
            cleanup(runtime->forget_context_checked(context), "private context retirement");
        if (context) cleanup(cuCtxDestroy(context), "destroy");
    }
};
struct Shape {
    unsigned int m, n, k, batches, padding;
};
struct Parameters {
    alignas(8) std::array<std::byte, 184> bytes{};
    template <class T>
    void put(std::size_t offset, T value) {
        if (offset + sizeof(value) > bytes.size()) throw std::runtime_error("parameter overflow");
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    }
};

void reject_physical_aliases(const std::shared_ptr<Owner>& owner, CUfunction mapped,
                             CUdevice device) {
    struct Aliases {
        CUmemGenericAllocationHandle handle{};
        std::array<CUdeviceptr, 2> addresses{};
        std::array<bool, 2> mapped{};
        std::size_t bytes{};
        ~Aliases() {
            for (std::size_t index = 0; index != addresses.size(); ++index) {
                if (mapped[index]) cleanup(cuMemUnmap(addresses[index], bytes), "alias unmap");
                if (addresses[index])
                    cleanup(cuMemAddressFree(addresses[index], bytes), "alias address free");
            }
            if (handle) cleanup(cuMemRelease(handle), "alias backing release");
        }
    } aliases;
    CUmemAllocationProp properties{};
    properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    properties.location = {CU_MEM_LOCATION_TYPE_DEVICE, device};
    check(cuMemGetAllocationGranularity(&aliases.bytes, &properties,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM),
          "alias granularity");
    check(cuMemCreate(&aliases.handle, aliases.bytes, &properties, 0), "alias backing");
    const CUmemAccessDesc access{{CU_MEM_LOCATION_TYPE_DEVICE, device},
                                 CU_MEM_ACCESS_FLAGS_PROT_READWRITE};
    for (std::size_t index = 0; index != aliases.addresses.size(); ++index) {
        check(cuMemAddressReserve(&aliases.addresses[index], aliases.bytes, 0, 0, 0),
              "alias reservation");
        check(cuMemMap(aliases.addresses[index], aliases.bytes, 0, aliases.handle, 0), "alias map");
        aliases.mapped[index] = true;
        check(cuMemSetAccess(aliases.addresses[index], aliases.bytes, &access, 1), "alias access");
    }
    const unsigned int written = 0x12345678;
    unsigned int read{};
    check(cuMemcpyHtoD(aliases.addresses[0], &written, sizeof(written)), "alias write");
    check(cuMemcpyDtoH(&read, aliases.addresses[1], sizeof(read)), "alias read");
    if (read != written)
        throw std::runtime_error("VMM addresses must share actual physical backing");
    Parameters parameters;
    parameters.put(0, aliases.addresses[0]);
    parameters.put(8, aliases.addresses[0] + 64);
    parameters.put(16, aliases.addresses[1]);
    parameters.put(168, aliases.addresses[1]);
    for (const auto offset : {24U, 32U}) parameters.put(offset, std::uint64_t{4});
    for (const auto offset : {40U, 48U}) parameters.put(offset, std::uint64_t{32});
    for (const auto offset : {56U, 64U, 72U, 80U}) parameters.put(offset, std::uint64_t{1});
    for (const auto offset : {88U, 92U, 96U, 100U, 104U, 108U, 112U, 128U, 132U})
        parameters.put(offset, std::uint32_t{1});
    parameters.put(152, 1.0F);
    void* arguments[]{parameters.bytes.data()};
    AllocationRegistry allocations;
    if (!allocations.record_vmm({aliases.handle, device}, aliases.bytes))
        throw std::runtime_error("record actual VMM handle");
    const CudaCopyRangeDispatch range_dispatch{cuPointerGetAttribute, cuMemGetAddressRange,
                                               cuMemHostGetFlags};
    if (!prepare_mapped_gemm_slices(mapped, {1, 1, 2}, {256, 1, 1}, arguments, owner, false,
                                    owner->mapping) ||
        prepare_checked_mapped_gemm_slices(mapped, {1, 1, 2}, {256, 1, 1}, arguments, owner,
                                           range_dispatch, allocations, owner->context, device,
                                           false, owner->mapping))
        throw std::runtime_error("physical aliases must reject despite valid virtual ranges");
    std::printf("allocation_proof physical_alias_observed=1 vmm_rejected=1\n");
}
CudaQueueDispatch dispatch() {
    CudaQueueDispatch result{};
    result.context_get_current = cuCtxGetCurrent;
    result.launch_kernel = cuLaunchKernel;
    result.copy_device_to_device = cuMemcpyDtoDAsync;
    result.event_create = cuEventCreate;
    result.event_record = cuEventRecord;
    result.event_query = cuEventQuery;
    result.event_destroy = cuEventDestroy;
    result.stream_synchronize = cuStreamSynchronize;
    return result;
}
template <typename Predicate>
void progress_until(CudaOperationQueue& queue, Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!predicate()) {
        check(queue.pump_admitted({}), "completion pump");
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("queue timeout");
        std::this_thread::yield();
    }
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct PublicCase {
    decltype(&cuLaunchKernel) launch{cuLaunchKernel};
    CUstream stream{};
    unsigned int shared{};
    bool packed{}, managed{}, graph{};
    std::function<void()> insert_high;
};
void run_case(const std::shared_ptr<Owner>& resources, CUfunction function, Shape shape,
              CUfunction mapped, float alpha, float beta, bool is_separate_input, unsigned pattern,
              bool should_unload_original = false, const PublicCase* public_case = nullptr) {
    auto& owner = *resources;
    const unsigned int width = owner.mapping == QualifiedGemmMapping::kRange ? 8U : 1U;
    std::unique_ptr<AllocationRegistry> allocations;
    if (!public_case) allocations = std::make_unique<AllocationRegistry>();
    CUdevice device{};
    check(cuCtxGetDevice(&device), "allocation device");
    const CudaCopyRangeDispatch range_dispatch{cuPointerGetAttribute, cuMemGetAddressRange,
                                               cuMemHostGetFlags};
    const std::array<std::size_t, 4> leading{shape.m + shape.padding, shape.n + shape.padding,
                                             shape.m + shape.padding, shape.m + shape.padding};
    const std::array<std::size_t, 4> matrix_elements{leading[0] * shape.k, leading[1] * shape.k,
                                                     leading[2] * shape.n, leading[3] * shape.n};
    std::array<std::size_t, 4> stride{};
    std::array<std::vector<float>, 4> before;
    for (std::size_t index = 0; index != 4; ++index) {
        stride[index] = matrix_elements[index] + shape.padding;
        before[index].assign(stride[index] * shape.batches + 2 * kGuard,
                             std::bit_cast<float>(kCanary));
        for (std::size_t batch = 0; batch != shape.batches; ++batch) {
            const std::size_t columns = index < 2 ? shape.k : shape.n;
            const std::size_t rows = index == 1 ? shape.n : shape.m;
            for (std::size_t column = 0; column != columns; ++column) {
                for (std::size_t row = 0; row != rows; ++row) {
                    const auto value =
                        static_cast<int>((row * 7 + column * 11 + batch * 3 + index * 5) % 17) - 8;
                    before[index][kGuard + batch * stride[index] + column * leading[index] + row] =
                        static_cast<float>(value) * static_cast<float>(index == 0 ? batch + 1 : 1) /
                        16.0F;
                    auto& element = before[index][kGuard + batch * stride[index] +
                                                  column * leading[index] + row];
                    if (pattern == 1) element /= 1.3F;
                    if (pattern == 2) {
                        constexpr std::array<std::uint32_t, 12> specials{
                            0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x7f800000, 0xff800000,
                            0x7fc12345, 0xffc23456, 0x00800000, 0x80800000, 0x3f800000, 0xbf800000};
                        if ((row + column + batch + index) % 7 == 0)
                            element = std::bit_cast<float>(
                                specials[(row * 3 + column * 5 + batch + index) % specials.size()]);
                    }
                }
            }
        }
        const auto bytes = before[index].size() * sizeof(float);
        check(public_case && public_case->managed
                  ? cuMemAllocManaged(&owner.buffers[index], bytes, CU_MEM_ATTACH_GLOBAL)
                  : cuMemAlloc(&owner.buffers[index], bytes),
              "allocate");
        if (allocations)
            require(allocations->record({owner.buffers[index], owner.context, device},
                                        before[index].size() * sizeof(float),
                                        AllocationScope::kContextBound,
                                        AllocationProvenance::kIndependentDevice),
                    "record observed ordinary allocation");
        check(cuMemcpyHtoD(owner.buffers[index], before[index].data(),
                           before[index].size() * sizeof(float)),
              "upload");
    }
    // Original fixed ABI: addresses, byte leading increments, element batch
    // strides, bounds, reduction extent, mode, scalar storage and input C.
    Parameters parameters;
    for (std::size_t index = 0; index != 3; ++index)
        parameters.put(index * 8, owner.buffers[index] + kGuard * sizeof(float));
    parameters.put(24, static_cast<std::uint64_t>(leading[0] * 4));
    parameters.put(32, static_cast<std::uint64_t>(leading[1] * 4));
    parameters.put(40, static_cast<std::uint64_t>(leading[0] * 32));
    parameters.put(48, static_cast<std::uint64_t>(leading[1] * 32));
    parameters.put(56, static_cast<std::uint64_t>(stride[0]));
    parameters.put(64, static_cast<std::uint64_t>(stride[1]));
    parameters.put(72, static_cast<std::uint64_t>(stride[2]));
    const std::size_t input = is_separate_input ? 3 : 2;
    parameters.put(80, static_cast<std::uint64_t>(stride[input]));
    parameters.put(88, static_cast<std::uint32_t>(leading[0]));
    parameters.put(92, static_cast<std::uint32_t>(leading[1]));
    parameters.put(96, static_cast<std::uint32_t>(leading[2]));
    parameters.put(100, static_cast<std::uint32_t>(leading[input]));
    parameters.put(104, shape.m);
    parameters.put(108, shape.n);
    parameters.put(112, shape.k);
    // Preserve a live alias or noncanonical inactive semaphore value. Exact
    // mode one must never read or write it in either original or private code.
    parameters.put(120, pattern == 0   ? std::uint64_t{0}
                        : pattern == 1 ? static_cast<std::uint64_t>(owner.buffers[0])
                                       : ~std::uint64_t{0});
    parameters.put(128, shape.k);
    parameters.put(132, std::uint32_t{1});
    parameters.put(152, alpha);
    parameters.put(156, beta);
    parameters.put(168, owner.buffers[input] + kGuard * sizeof(float));
    void* arguments[]{parameters.bytes.data()};
    std::size_t packed_size = 176;
    std::array<void*, 5> extra{CU_LAUNCH_PARAM_BUFFER_POINTER, parameters.bytes.data(),
                               CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size, CU_LAUNCH_PARAM_END};
    if (public_case) check(cuStreamSynchronize(nullptr), "public upload completion");
    const bool packed = public_case && public_case->packed;
    const auto launch = public_case ? public_case->launch : cuLaunchKernel;
    struct PublicGraph {
        CUgraph graph{};
        CUgraphExec executable{};
        ~PublicGraph() {
            if (executable)
                cleanup(cuGraphExecDestroy(executable), "public graph executable destroy");
            if (graph) cleanup(cuGraphDestroy(graph), "public graph destroy");
        }
    } captured;
    if (public_case && public_case->graph)
        check(cuStreamBeginCapture(public_case->stream, CU_STREAM_CAPTURE_MODE_GLOBAL),
              "public original graph capture");
    check(launch(function, (shape.m + 127) / 128, (shape.n + 31) / 32, shape.batches, 256, 1, 1,
                 public_case ? public_case->shared : 0, public_case ? public_case->stream : nullptr,
                 packed ? nullptr : arguments, packed ? extra.data() : nullptr),
          "original launch");
    if (public_case) {
        parameters.bytes.fill(std::byte{0xff});
        arguments[0] = nullptr;
        if (public_case->insert_high) public_case->insert_high();
        if (public_case->graph) {
            check(cuStreamEndCapture(public_case->stream, &captured.graph),
                  "public original graph end");
            check(cuGraphInstantiateWithFlags(&captured.executable, captured.graph, 0),
                  "public original graph instantiate");
            check(cuGraphLaunch(captured.executable, public_case->stream),
                  "public original graph replay");
        }
    }
    check(cuCtxSynchronize(), "original completion");
    auto expected = before[2];
    for (std::size_t batch = 0; pattern == 0 && batch != shape.batches; ++batch) {
        for (std::size_t column = 0; column != shape.n; ++column) {
            for (std::size_t row = 0; row != shape.m; ++row) {
                double sum{};
                for (std::size_t reduction = 0; reduction != shape.k; ++reduction)
                    sum +=
                        static_cast<double>(
                            before[0][kGuard + batch * stride[0] + reduction * leading[0] + row]) *
                        before[1][kGuard + batch * stride[1] + reduction * leading[1] + column];
                const auto value = static_cast<float>(
                    alpha * sum + beta * before[input][kGuard + batch * stride[input] +
                                                       column * leading[input] + row]);
                expected[kGuard + batch * stride[2] + column * leading[2] + row] = value;
            }
        }
    }
    std::array<std::vector<float>, 4> baseline;
    for (std::size_t index = 0; index != 4; ++index) {
        auto& actual = baseline[index];
        actual.resize(before[index].size());
        check(cuMemcpyDtoH(actual.data(), owner.buffers[index], actual.size() * sizeof(float)),
              "download");
        const auto& wanted = index == 2 ? expected : before[index];
        for (std::size_t element = 0; element != actual.size(); ++element) {
            // Signed zero is arithmetic-equivalent. Every untouched byte must
            // still compare exactly, including padding and inactive input C.
            const auto left = std::bit_cast<std::uint32_t>(actual[element]);
            const auto right = std::bit_cast<std::uint32_t>(wanted[element]);
            const bool arithmetic_zero = index == 2 &&
                                         wanted[element] != std::bit_cast<float>(kCanary) &&
                                         actual[element] == 0 && wanted[element] == 0;
            const bool active_output = index == 2 && element >= kGuard &&
                                       element < kGuard + stride[2] * shape.batches &&
                                       (element - kGuard) % stride[2] / leading[2] < shape.n &&
                                       (element - kGuard) % stride[2] % leading[2] < shape.m;
            if ((pattern == 0 || !active_output) && left != right && !arithmetic_zero) {
                std::fprintf(stderr,
                             "m=%u n=%u k=%u batches=%u padding=%u alpha=%g beta=%g separate=%d "
                             "buffer=%zu element=%zu got=%g expected=%g\n",
                             shape.m, shape.n, shape.k, shape.batches, shape.padding, alpha, beta,
                             is_separate_input, index, element, actual[element], wanted[element]);
                throw std::runtime_error("original GEMM oracle or guard mismatch");
            }
        }
    }
    if (public_case) {
        for (const auto& buffer : baseline)
            require(
                std::fwrite(buffer.data(), sizeof(float), buffer.size(), stdout) == buffer.size(),
                "public binary result stream");
        std::fprintf(
            stderr,
            "public_case m=%u n=%u k=%u batches=%u padding=%u alpha=%g beta=%g separate=%d "
            "pattern=%u result=pass\n",
            shape.m, shape.n, shape.k, shape.batches, shape.padding, alpha, beta, is_separate_input,
            pattern);
        for (auto& buffer : owner.buffers) {
            check(cuMemFree(buffer), "public checked free");
            buffer = 0;
        }
        return;
    }
    if (should_unload_original) {
        require(owner.runtime && owner.private_owner, "retirement needs a retained code owner");
        // Original baseline and all earlier queue work are physically complete.
        // Never retry an uncertain native unload from the fixture destructor.
        const auto module = std::exchange(owner.module, nullptr);
        if (owner.library) {
            const auto library = std::exchange(owner.library, nullptr);
            check(cuLibraryUnload(library), "original native library unload");
            owner.runtime->retire_library_source(library);
        } else {
            check(owner.runtime->unload_source_module(module), "original native module unload");
        }
        check(owner.runtime->forget_context_checked(owner.context), "retire source while leased");
        require(!owner.runtime->mapped_gemm_kernel(function), "retired original lookup is absent");
    }
    const auto saved_parameters = parameters.bytes;
    const auto check_output = [&](unsigned completed_batches) {
        for (std::size_t index = 0; index != 4; ++index) {
            std::vector<float> actual(before[index].size());
            check(cuMemcpyDtoH(actual.data(), owner.buffers[index], actual.size() * sizeof(float)),
                  "stopped output");
            for (std::size_t element = 0; element != actual.size(); ++element) {
                const bool written = index == 2 && element >= kGuard &&
                                     element < kGuard + stride[2] * completed_batches;
                const float wanted = written ? baseline[index][element] : before[index][element];
                if (std::bit_cast<std::uint32_t>(actual[element]) !=
                    std::bit_cast<std::uint32_t>(wanted))
                    throw std::runtime_error("queue bitwise original/private differential failed");
            }
        }
    };
    for (const bool cancel : {false, true}) {
        if (cancel && shape.batches <= width) continue;
        for (std::size_t index = 0; index != 4; ++index)
            check(cuMemcpyHtoD(owner.buffers[index], before[index].data(),
                               before[index].size() * sizeof(float)),
                  "restore");
        // Pageable uploads may finish after their host call; nonblocking
        // backend streams require explicit fixture upload completion.
        check(cuStreamSynchronize(nullptr), "upload completion");
        parameters.bytes = saved_parameters;
        AllocationRegistry unknown_allocations;
        require(!prepare_checked_mapped_gemm_slices(
                    mapped, {(shape.m + 127) / 128, (shape.n + 31) / 32, shape.batches},
                    {256, 1, 1}, arguments, resources, range_dispatch, unknown_allocations,
                    owner.context, device, false, owner.mapping),
                "native pointer metadata alone cannot authorize preparation");
        auto prepared = prepare_checked_mapped_gemm_slices(
            mapped, {(shape.m + 127) / 128, (shape.n + 31) / 32, shape.batches}, {256, 1, 1},
            arguments, resources, range_dispatch, *allocations, owner.context, device, false,
            owner.mapping);
        require(prepared.has_value(), "qualified owned GEMM preparation");
        auto continuation =
            std::dynamic_pointer_cast<CudaBinarySliceContinuation>(prepared->continuation);
        parameters.bytes.fill(std::byte{0xff});
        const auto plane_blocks = std::uint64_t{(shape.m + 127) / 128} * ((shape.n + 31) / 32);
        CudaOperationQueue queue(owner.context, dispatch());
        auto low = queue.bind_stream(owner.low, OrderedStreamKind::kNonBlocking, resources);
        auto high = queue.bind_stream(owner.high, OrderedStreamKind::kNonBlocking, resources);
        auto work = queue.submit(low, std::move(*prepared));
        auto dependent = queue.submit(low, CudaQueuedMarker{});
        prepared.reset();
        require(work && dependent, "queue submission");
        check(queue.pump_admitted({}), "ungranted pump");
        require(continuation->progress().physical_launches == 0, "ungranted work cannot issue");
        unsigned int launches{};
        for (unsigned batch = 0; batch < shape.batches; batch += width) {
            const auto completed_planes = std::min(shape.batches, batch + width);
            if (batch) require(queue.resume(work), "explicit resume");
            check(queue.pump_admitted(std::array{work}), "admitted plane issue");
            check(queue.request_retreat(work), "retreat at plane boundary");
            progress_until(queue, [&] {
                return continuation->progress().completed_blocks == completed_planes * plane_blocks;
            });
            require(continuation->progress().physical_launches == ++launches,
                    "one physical launch per completed plane/range");
            const bool final = completed_planes == shape.batches;
            if (!final)
                require(queue.state(work) == OrderedOperationState::kSuspended &&
                            queue.query(dependent) == CUDA_ERROR_NOT_READY,
                        "pending planes retain dependent work");
            if (batch == 0 || batch == width || final) {
                check_output(completed_planes);
                if (!final) ++partial_observations;
            }
            if (cancel) {
                require(queue.cancel_stopped(work), "cancel only proven stop");
                check(queue.shutdown(), "cancel shutdown");
                // Cancellation/drain must not issue any still-pending plane.
                check_output(completed_planes);
                require(continuation->progress().physical_launches == 1 &&
                            continuation->progress().completed_blocks ==
                                completed_planes * plane_blocks,
                        "cancellation cannot advance or replay pending work");
                ++cancelled_cases;
                break;
            }
            if (batch == 0 && !final) {
                auto copy = queue.submit(
                    high, CudaQueuedCopy{CudaQueuedCopyKind::kDeviceToDevice, owner.independent,
                                         owner.buffers[0] + kGuard * sizeof(float), nullptr, 16,
                                         resources});
                require(static_cast<bool>(copy), "independent copy submission");
                progress_until(queue, [&] { return queue.query(copy) == CUDA_SUCCESS; });
                std::array<float, 4> copied{};
                check(cuMemcpyDtoH(copied.data(), owner.independent, 16),
                      "independent copy result");
                const bool identical_copy =
                    std::equal(copied.begin(), copied.end(), before[0].begin() + kGuard,
                               [](float actual, float expected_value) {
                                   return std::bit_cast<std::uint32_t>(actual) ==
                                          std::bit_cast<std::uint32_t>(expected_value);
                               });
                require(identical_copy && queue.query(work) == CUDA_ERROR_NOT_READY &&
                            continuation->progress().completed_blocks ==
                                completed_planes * plane_blocks,
                        "independent GPU work completes while pending GEMM remains stopped");
                ++independent_copies;
            }
        }
        if (!cancel) {
            progress_until(queue, [&] { return queue.query(dependent) == CUDA_SUCCESS; });
            require(continuation->progress().completed_blocks == plane_blocks * shape.batches,
                    "all logical CTAs completed");
            check(queue.shutdown(), "completed shutdown");
            ++mapped_cases;
        }
    }
    std::printf(
        "case m=%u n=%u k=%u batches=%u padding=%u alpha=%g beta=%g separate=%d "
        "pattern=%u result=pass\n",
        shape.m, shape.n, shape.k, shape.batches, shape.padding, alpha, beta, is_separate_input,
        pattern);
    for (auto& buffer : owner.buffers) {
        check(cuMemFree(buffer), "checked free");
        buffer = 0;
    }
}
enum class ImageMode {
    kFixture,
    kFiles,
    kRewrite,
    kFatbin,
    kWrapper,
    kOwnerRaw,
    kOwnerFatbin,
    kOwnerWrapper
};
struct ImageInput {
    ImageMode mode{ImageMode::kFixture};
    const char* original{};
    const char* mapped{};
    QualifiedGemmMapping mapping{QualifiedGemmMapping::kPlane};
};
std::vector<std::byte> read_image(const char* path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(input.good(), "image open");
    const auto size = input.tellg();
    require(size > 0 && size <= static_cast<std::streamoff>(kMaxBinaryImageBytes), "image size");
    std::vector<std::byte> image(static_cast<std::size_t>(size));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size()));
    require(input.good(), "image read");
    return image;
}

// CUDA-only application branch. The same code and binary result stream run
// natively and under preload; no private Glimmer owner/preparer/queue is used.
void run_public(const std::string& format, const char* image_path, const std::string& scenario) {
    require(format == "--public-raw" || format == "--public-fatbin" || format == "--public-wrapper",
            "public source format");
    require(scenario == "source" || scenario == "priority" || scenario == "fallback" ||
                scenario == "capture" || scenario == "graph" || scenario == "error",
            "public scenario");
    for (unsigned int round = 0; round != 2; ++round) {
        auto resources = std::make_shared<Owner>();
        auto& owner = *resources;
        check(cuInit(0), "public init");
        CUdevice device{};
        check(cuDeviceGet(&device, 0), "public device");
        check(cuCtxCreate(&owner.context, 0, device), "public context");
        check(cuStreamCreate(&owner.low, CU_STREAM_NON_BLOCKING), "public low stream");
        const auto resolve = [round](const char* name,
                                     cuuint64_t flags = CU_GET_PROC_ADDRESS_DEFAULT) {
            void* address{};
            if (round == 0)
                address = dlsym(RTLD_DEFAULT, name);
            else {
                CUdriverProcAddressQueryResult status{};
                check(cuGetProcAddress(name, &address, 12000, flags, &status),
                      "public cached resolve");
                require(status == CU_GET_PROC_ADDRESS_SUCCESS, "public cached status");
            }
            require(address != nullptr, "public entry");
            return address;
        };
        if (scenario == "capture")
            check(cuStreamBeginCapture(owner.low, CU_STREAM_CAPTURE_MODE_GLOBAL),
                  "public source capture");
        auto image = read_image(image_path);
        std::array<std::uint64_t, 3> wrapper{0x00000001466243b1ULL,
                                             reinterpret_cast<std::uintptr_t>(image.data()), 0};
        if (format == "--public-wrapper") {
            const auto load =
                reinterpret_cast<decltype(&cuLibraryLoadData)>(resolve("cuLibraryLoadData"));
            const auto module =
                reinterpret_cast<decltype(&cuLibraryGetModule)>(resolve("cuLibraryGetModule"));
            check(load(&owner.library, wrapper.data(), nullptr, nullptr, 0, nullptr, nullptr, 0),
                  "public library load");
            check(module(&owner.module, owner.library), "public library module");
        } else
            check(cuModuleLoadData(&owner.module, image.data()), "public original source");
        CUfunction function{};
        check(cuModuleGetFunction(&function, owner.module, "volta_sgemm_128x32_nt"),
              "public function");
        check(cuFuncLoad(function), "public original readiness");
        std::ranges::fill(image, std::byte{0xff});
        wrapper.fill(0);
        if (scenario == "capture") {
            CUgraph graph{};
            check(cuStreamEndCapture(owner.low, &graph), "public source capture end");
            check(cuGraphDestroy(graph), "public empty graph destroy");
        }
        std::size_t offset{}, size{};
        check(cuFuncGetParamInfo(function, 0, &offset, &size), "public original ABI");
        require(offset == 0 && size == 176, "public original ABI remains 176 bytes");
        PublicCase options;
        options.graph = scenario == "graph";
        options.stream = owner.low;
        options.launch =
            round == 0 ? cuLaunchKernel
                       : reinterpret_cast<decltype(&cuLaunchKernel)>(resolve(
                             "cuLaunchKernel", CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM));
        if (scenario == "error") {
            Parameters parameters;
            void* arguments[]{parameters.bytes.data()};
            const auto status =
                options.launch(function, 1, 1, 1, 0, 1, 1, 0, options.stream, arguments, nullptr);
            require(status == CUDA_ERROR_INVALID_VALUE, "public invalid block preserves error");
            check(cuCtxSynchronize(), "public rejected launch does not poison completion");
            std::fprintf(stderr, "public_error round=%u status=%d subsequent_sync=0\n", round,
                         status);
            continue;
        }
        std::vector<float> high_input;
        Parameters high_parameters;
        const auto high_launch = [&] {
            void* arguments[]{high_parameters.bytes.data()};
            check(cuLaunchKernel(function, 1, 1, 1, 256, 1, 1, 0, owner.high, arguments, nullptr),
                  "public high launch");
            check(cuStreamSynchronize(owner.high), "public high logical completion");
        };
        const auto check_high = [&] {
            std::vector<float> result(high_input.size());
            check(cuMemcpyDtoH(result.data(), owner.independent, result.size() * sizeof(float)),
                  "public high download");
            for (std::size_t index = 0; index != result.size(); ++index)
                require(result[index] == (index == 64 ? 7.25F : high_input[index]),
                        "public high guards/result");
        };
        if (scenario == "priority") {
            check(cuStreamCreateWithPriority(&owner.high, CU_STREAM_NON_BLOCKING, -1),
                  "public high stream");
            high_input.assign(1024, std::bit_cast<float>(kCanary));
            high_input[32] = 2;
            high_input[48] = 3;
            high_input[64] = 5;
            check(cuMemAlloc(&owner.independent, high_input.size() * sizeof(float)),
                  "public high allocation");
            high_parameters.put(0, owner.independent + 128);
            high_parameters.put(8, owner.independent + 192);
            high_parameters.put(16, owner.independent + 256);
            high_parameters.put(168, owner.independent + 256);
            for (auto position : {24U, 32U}) high_parameters.put(position, std::uint64_t{4});
            for (auto position : {40U, 48U}) high_parameters.put(position, std::uint64_t{32});
            for (auto position : {56U, 64U, 72U, 80U})
                high_parameters.put(position, std::uint64_t{1});
            for (auto position : {88U, 92U, 96U, 100U, 104U, 108U, 112U, 128U, 132U})
                high_parameters.put(position, std::uint32_t{1});
            high_parameters.put(152, 1.0F);
            high_parameters.put(156, 0.25F);
            check(cuMemcpyHtoD(owner.independent, high_input.data(),
                               high_input.size() * sizeof(float)),
                  "public high warm upload");
            check(cuStreamSynchronize(nullptr), "public high upload completion");
            high_launch();
            check_high();
            options.insert_high = [&] {
                // Arrival delay is not completion proof. The external verifier
                // must observe actual high stop between nonfinal low slices.
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                high_launch();
                const auto pending = cuStreamQuery(owner.low);
                require(pending == CUDA_SUCCESS || pending == CUDA_ERROR_NOT_READY,
                        "public low query");
                std::fprintf(stderr, "public_priority high_completed=1 low_pending=%d\n",
                             pending == CUDA_ERROR_NOT_READY);
            };
        }
        constexpr std::array<Shape, 12> shapes{{{3, 16, 2048, 36, 0},
                                                {16, 32, 2048, 36, 0},
                                                {1, 1, 1, 1, 0},
                                                {3, 7, 7, 2, 0},
                                                {7, 3, 8, 3, 0},
                                                {16, 32, 9, 3, 0},
                                                {127, 31, 33, 2, 0},
                                                {128, 32, 32, 1, 0},
                                                {129, 33, 16, 2, 0},
                                                {3, 16, 17, 36, 5},
                                                {16, 32, 33, 3, 3},
                                                {32, 10, 32, 1, 0}}};
        std::size_t cases{};
        if (scenario == "source") {
            for (const auto shape : shapes)
                for (const auto scalar :
                     {std::array{1.F, 0.F}, std::array{-.5F, .25F}, std::array{1.25F, -1.F}})
                    for (const bool separate : {false, true})
                        for (unsigned int pattern = 0; pattern != 3; ++pattern) {
                            if (round == 0)
                                options.launch = cases % 2
                                                     ? reinterpret_cast<decltype(&cuLaunchKernel)>(
                                                           resolve("cuLaunchKernel"))
                                                     : cuLaunchKernel;
                            run_case(resources, function, shape, nullptr, scalar[0], scalar[1],
                                     separate, pattern, false, &options);
                            ++cases;
                        }
        } else {
            const unsigned int count = scenario == "fallback" ? 4 : 3;
            for (unsigned int index = 0; index != count; ++index) {
                auto shape = shapes[index % 2];
                if (scenario == "fallback") {
                    if (index == 0) shape.batches = 37;
                    options.shared = index == 1 ? 16 : 0;
                    options.packed = index == 2;
                    options.managed = index == 3;
                }
                if (scenario == "priority")
                    check(cuMemcpyHtoD(owner.independent, high_input.data(),
                                       high_input.size() * sizeof(float)),
                          "public high reset before low");
                run_case(resources, function, shape, nullptr, -.5F, .25F, false, 0, false,
                         &options);
                if (scenario == "priority") check_high();
                ++cases;
            }
        }
        std::fprintf(stderr, "public_round=%u cases=%zu source=%s scenario=%s original_abi=176\n",
                     round, cases, format.c_str(), scenario.c_str());
    }
    require(std::fflush(stdout) == 0, "public result flush");
}
// Retain caller bytes until explicit function readiness, then overwrite them.
// Default library options require the SDK to own its code, not borrow our image.
std::array<std::vector<std::byte>, 2> load_images(Owner& owner, ImageInput input) {
    owner.mapping = input.mapping;
    if (input.mode == ImageMode::kFixture) {
        check(cuModuleLoadData(&owner.module, gemm_slice_ptx_fixture(false).c_str()),
              "original PTX fixture");
        check(cuModuleLoadData(&owner.mapped_module, gemm_slice_ptx_fixture(true).c_str()),
              "mapped PTX fixture");
        return {};
    }
    if (input.mode == ImageMode::kFiles) {
        check(cuModuleLoad(&owner.module, input.original), "original module");
        check(cuModuleLoad(&owner.mapped_module, input.mapped), "mapped module");
        return {};
    }
    auto original = read_image(input.original);
    const auto before = original;
    std::array<std::uint64_t, 3> wrapper{0x00000001466243b1,
                                         reinterpret_cast<std::uintptr_t>(original.data()), 0};
    const bool is_wrapper =
        input.mode == ImageMode::kWrapper || input.mode == ImageMode::kOwnerWrapper;
    const bool has_owner = input.mode == ImageMode::kOwnerRaw ||
                           input.mode == ImageMode::kOwnerFatbin ||
                           input.mode == ImageMode::kOwnerWrapper;
    if (has_owner) {
        owner.driver = std::make_unique<DriverDispatch>();
        require(owner.driver->initialize(), "native dispatch initialization");
        owner.runtime = std::make_unique<PersistentKernelRuntime>(*owner.driver, 3, false, true);
    }
    const void* source = is_wrapper ? static_cast<void*>(wrapper.data()) : original.data();
    if (is_wrapper) {
        check(cuLibraryLoadData(&owner.library, source, nullptr, nullptr, 0, nullptr, nullptr, 0),
              "original wrapper library");
        check(cuLibraryGetModule(&owner.module, owner.library), "borrow original library module");
        if (owner.runtime) {
            owner.runtime->observe_library_source(owner.library, source);
            owner.runtime->observe_library_module(owner.library, owner.module, owner.context);
        }
    } else if (owner.runtime) {
        check(owner.runtime->load_module(&owner.module, source), "owned original module");
    } else {
        check(cuModuleLoadData(&owner.module, source), "original owned source");
    }
    auto mapped = snapshot_qualified_gemm_cubin(source, owner.mapping);
    require(mapped && *mapped == read_image(input.mapped) && original == before,
            "production source rewrite agrees with independent mapping and preserves input");
    if (owner.runtime) {
        CUfunction function{}, direct{};
        check(owner.runtime->get_function(&function, owner.module, kQualifiedGemmEntry),
              "owned original lookup");
        check(cuModuleGetFunction(&direct, owner.module, kQualifiedGemmEntry), "original identity");
        require(function == direct && !owner.runtime->module_source_is_prepared(function),
                "original handle preserved and private CUDA deferred");
        check(cuFuncLoad(function), "original readiness before caller overwrite");
        std::ranges::fill(original, std::byte{0xff});
        wrapper.fill(0);
        owner.runtime->prepare_module_source(function);
        auto variant = owner.runtime->mapped_gemm_kernel(function);
        require(variant && variant->context == owner.context && variant->function != function &&
                    variant->mapping == owner.mapping &&
                    !owner.runtime->mapped_binary_kernel(function),
                "typed owned GEMM only");
        owner.owned_function = variant->function;
        owner.private_owner = std::move(variant->resources);
        owner.runtime->prepare_module_source(function);
        const auto repeated = owner.runtime->mapped_gemm_kernel(function);
        require(repeated && repeated->function == owner.owned_function,
                "stable one-time private preparation");
        return {std::move(original), std::move(*mapped)};
    }
    check(cuLibraryLoadData(&owner.mapped_library, mapped->data(), nullptr, nullptr, 0, nullptr,
                            nullptr, 0),
          "private owned library");
    check(cuLibraryGetModule(&owner.mapped_module, owner.mapped_library),
          "borrow private library module");
    return {std::move(original), std::move(*mapped)};
}
void run(ImageInput input) {
    auto resources = std::make_shared<Owner>();
    auto& owner = *resources;
    check(cuInit(0), "init");
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "device");
    int major{}, minor{};
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
          "major");
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
          "minor");
    if (major != 7 || minor != 0) throw std::runtime_error("requires sm_70");
    check(cuCtxCreate(&owner.context, 0, device), "context");
    auto caller_images = load_images(owner, input);
    const bool is_binary = input.mode != ImageMode::kFixture;
    check(cuStreamCreate(&owner.low, CU_STREAM_NON_BLOCKING), "low stream");
    check(cuStreamCreate(&owner.high, CU_STREAM_NON_BLOCKING), "independent stream");
    check(cuMemAlloc(&owner.independent, 16), "independent allocation");
    CUfunction function{};
    check(
        cuModuleGetFunction(&function, owner.module, is_binary ? kQualifiedGemmEntry : "gemm_test"),
        "original function");
    check(cuFuncLoad(function), "original code readiness");
    std::size_t offset{}, size{};
    check(cuFuncGetParamInfo(function, 0, &offset, &size), "parameter ABI");
    if (offset != 0 || size != 176 ||
        cuFuncGetParamInfo(function, 1, &offset, &size) != CUDA_ERROR_INVALID_VALUE)
        throw std::runtime_error("unqualified parameter ABI");
    CUfunction mapped = owner.owned_function;
    if (!mapped)
        check(cuModuleGetFunction(&mapped, owner.mapped_module,
                                  is_binary ? kQualifiedGemmEntry : "gemm_test"),
              "mapped function");
    check(cuFuncLoad(mapped), "private code readiness");
    reject_physical_aliases(resources, mapped, device);
    for (auto& image : caller_images) std::ranges::fill(image, std::byte{0xff});
    check(cuFuncGetParamInfo(mapped, 0, &offset, &size), "private parameter ABI");
    if (offset != 0 || size != 184 ||
        cuFuncGetParamInfo(mapped, 1, &offset, &size) != CUDA_ERROR_INVALID_VALUE)
        throw std::runtime_error("unexpected private parameter ABI");
    for (auto attribute :
         {CU_FUNC_ATTRIBUTE_NUM_REGS, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
          CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK}) {
        int original_value{}, mapped_value{};
        check(cuFuncGetAttribute(&original_value, attribute, function), "original resources");
        check(cuFuncGetAttribute(&mapped_value, attribute, mapped), "mapped resources");
        if (is_binary && original_value != mapped_value)
            throw std::runtime_error("resource attributes changed");
        std::printf("resource attribute=%d original=%d mapped=%d\n", static_cast<int>(attribute),
                    original_value, mapped_value);
    }
    constexpr std::array<Shape, 12> shapes{{{3, 16, 2048, 36, 0},
                                            {16, 32, 2048, 36, 0},
                                            {1, 1, 1, 1, 0},
                                            {3, 7, 7, 2, 0},
                                            {7, 3, 8, 3, 0},
                                            {16, 32, 9, 3, 0},
                                            {127, 31, 33, 2, 0},
                                            {128, 32, 32, 1, 0},
                                            {129, 33, 16, 2, 0},
                                            {3, 16, 17, 36, 5},
                                            {16, 32, 33, 3, 3},
                                            {32, 10, 32, 1, 0}}};
    std::size_t cases{};
    for (auto shape : shapes)
        for (const auto scalars :
             {std::array{1.0F, 0.0F}, std::array{-0.5F, 0.25F}, std::array{1.25F, -1.0F}})
            for (const bool separate : {false, true})
                for (unsigned pattern = 0; pattern != 3; ++pattern) {
                    run_case(resources, function, shape, mapped, scalars[0], scalars[1], separate,
                             pattern);
                    ++cases;
                }
    if (owner.runtime) {
        // Read the original oracle, then unload its code before private issue.
        // Only the still-live context and the private code lease remain usable.
        run_case(resources, function, {3, 16, 2048, 36, 0}, mapped, -0.5F, 0.25F, false, 0, true);
        ++cases;
        std::printf("retired_source retained_private_execution=1 original_native_unloaded=1\n");
    }
    std::printf(
        "queued_gemm original_cases=%zu mapped_cases=%zu partial_observations=%zu "
        "cancelled_cases=%zu independent_copies=%zu actual_binary=%d checked_cleanup=1 "
        "post_cancel_output=1 checked_allocations=1 unknown_allocations_rejected=1\n",
        cases, mapped_cases, partial_observations, cancelled_cases, independent_copies, is_binary);
    std::printf("source mode=%d owned_images_overwritten=%d\n", static_cast<int>(input.mode),
                !caller_images[0].empty());
}
}  // namespace
int main(int argc, char** argv) {
    if (argc != 1 && argc != 3 && argc != 4) return 2;
    try {
        if (argc == 4 && std::string{argv[1]}.starts_with("--public-")) {
            run_public(argv[1], argv[2], argv[3]);
            return cleanup_ok ? 0 : 1;
        }
        ImageInput input;
        if (argc == 3) input = {ImageMode::kFiles, argv[1], argv[2]};
        if (argc == 4) {
            const std::string mode{argv[1]};
            if (mode == "--range") {
                input.mode = ImageMode::kRewrite;
                input.mapping = QualifiedGemmMapping::kRange;
            } else if (mode == "--rewrite")
                input.mode = ImageMode::kRewrite;
            else if (mode == "--fatbin")
                input.mode = ImageMode::kFatbin;
            else if (mode == "--wrapper")
                input.mode = ImageMode::kWrapper;
            else if (mode == "--owner-raw")
                input.mode = ImageMode::kOwnerRaw;
            else if (mode == "--owner-fatbin")
                input.mode = ImageMode::kOwnerFatbin;
            else if (mode == "--owner-wrapper")
                input.mode = ImageMode::kOwnerWrapper;
            else
                return 2;
            input.original = argv[2];
            input.mapped = argv[3];
            if (input.mode == ImageMode::kOwnerRaw || input.mode == ImageMode::kOwnerFatbin ||
                input.mode == ImageMode::kOwnerWrapper)
                input.mapping = QualifiedGemmMapping::kRange;
        }
        run(input);
        return cleanup_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
