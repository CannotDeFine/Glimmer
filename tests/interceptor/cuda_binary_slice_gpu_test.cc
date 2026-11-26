#include "internal/cuda_binary_slice.h"
#include "internal/qualified_cubin.h"
#include "internal/persistent_kernel.h"
#include <fatbinary_section.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
using namespace glimmer::interceptor;
constexpr char kActualEntry[] =
    "_ZN2at6native29vectorized_elementwise_kernelILi4ENS0_15CUDAFunctor_"
    "addIfEESt5arrayIPcLm3EEEEviT0_T1_";
// Self-contained ownership fixture, NOT a cubin transformation. Passing a
// separately qualified original cubin tests the actual original-code variant.
constexpr char kFixture[] = R"ptx(
.version 8.0
.target sm_70
.address_size 64
.visible .entry mapped_add_test(.param .u32 count, .param .f32 alpha,
                                .param .align 8 .b8 aggregate[32]) {
 .reg .b32 %n, %thread_index, %logical, %local, %i;
 .reg .b64 %out, %a, %b, %index64, %offset, %pa, %pb, %po;
 .reg .f32 %alpha, %av, %bv, %result;
 .reg .pred %done;
 ld.param.u32 %n, [count];
 ld.param.f32 %alpha, [alpha];
 ld.param.u64 %out, [aggregate];
 ld.param.u64 %a, [aggregate+8];
 ld.param.u64 %b, [aggregate+16];
 ld.param.u64 %index64, [aggregate+24];
 cvt.u32.u64 %logical, %index64;
 shl.b32 %logical, %logical, 10;
 mov.u32 %thread_index, %tid.x;
 mov.u32 %local, %thread_index;
LOOP:
 setp.ge.u32 %done, %local, 1024;
 @%done bra END;
 add.u32 %i, %logical, %local;
 setp.ge.u32 %done, %i, %n;
 @%done bra END;
 mul.wide.u32 %offset, %i, 4;
 add.u64 %pa, %a, %offset;
 add.u64 %pb, %b, %offset;
 add.u64 %po, %out, %offset;
 ld.global.f32 %av, [%pa];
 ld.global.f32 %bv, [%pb];
 fma.rn.f32 %result, %bv, %alpha, %av;
 st.global.f32 [%po], %result;
 add.u32 %local, %local, 128;
 bra LOOP;
END:
 ret;
}
)ptx";
bool cleanup_ok = true;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(CUresult result, const char* operation) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + std::to_string(result));
    }
}
void cleanup(CUresult result) noexcept {
    cleanup_ok &= result == CUDA_SUCCESS;
}
struct Resources {
    CUcontext context{};
    CUmodule module{};
    CUlibrary library{};
    CUstream low{};
    CUstream high{};
    CUdeviceptr data{};
    CUdeviceptr high_data{};
    std::unique_ptr<DriverDispatch> driver;
    std::unique_ptr<PersistentKernelRuntime> runtime;
    std::shared_ptr<void> private_owner;
    ~Resources() {
        if (data) cleanup(cuMemFree(data));
        if (high_data) cleanup(cuMemFree(high_data));
        if (low) cleanup(cuStreamDestroy(low));
        if (high) cleanup(cuStreamDestroy(high));
        if (module && !library)
            cleanup(runtime ? runtime->unload_module(module) : cuModuleUnload(module));
        if (library) cleanup(cuLibraryUnload(library));
        private_owner.reset();
        if (runtime && context) cleanup(runtime->forget_context_checked(context));
        if (context) cleanup(cuCtxDestroy(context));
    }
};
CudaQueueDispatch dispatch() {
    CudaQueueDispatch result{};
    result.context_get_current = cuCtxGetCurrent;
    result.launch_kernel = cuLaunchKernel;
    result.event_create = cuEventCreate;
    result.event_record = cuEventRecord;
    result.event_query = cuEventQuery;
    result.event_destroy = cuEventDestroy;
    result.stream_synchronize = cuStreamSynchronize;
    return result;
}
template <typename Predicate>
void progress_until(CudaOperationQueue& queue, Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        check(queue.pump_admitted({}), "completion pump");
        require(std::chrono::steady_clock::now() < deadline, "queue progress timed out");
        std::this_thread::yield();
    }
}
CudaQueuedKernel prepared(CUfunction function, CUdeviceptr pointer,
                          const std::shared_ptr<Resources>& resources, int count) {
    float alpha = 1.0F;
    std::array<std::uint64_t, 3> pointers{pointer, pointer, pointer};
    std::array<void*, 3> arguments{&count, &alpha, pointers.data()};
    auto kernel = prepare_mapped_add_slices(
        function, {(static_cast<unsigned int>(count) + 1023) / 1024, 1, 1}, {128, 1, 1},
        arguments.data(), resources);
    require(kernel.has_value(), "owned mapped preparation");
    count = 0;
    alpha = 0;
    pointers.fill(0);
    return std::move(*kernel);
}
enum class BinaryInputMode { kMapped, kRewrite, kSource };
void run(const char* binary, BinaryInputMode mode) {
    auto resources = std::make_shared<Resources>();
    check(cuInit(0), "init");
    CUdevice device{};
    check(cuDeviceGet(&device, 0), "device");
    check(cuCtxCreate(&resources->context, 0, device), "context");
    CUfunction function{};
    CUfunction original_function{};
    if (binary) {
        int major{}, minor{};
        check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
              "major");
        check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
              "minor");
        require(major == 7 && minor == 0, "qualified cubin requires sm_70");
        if (mode != BinaryInputMode::kMapped) {
            std::ifstream input(binary, std::ios::binary | std::ios::ate);
            require(input.good(), "original cubin open");
            const auto size = input.tellg();
            require(size > 0 && size <= static_cast<std::streamoff>(kMaxBinaryImageBytes),
                    "original cubin size");
            std::vector<std::byte> original(static_cast<std::size_t>(size));
            input.seekg(0);
            input.read(reinterpret_cast<char*>(original.data()),
                       static_cast<std::streamsize>(original.size()));
            require(input.good(), "original cubin read");
            if (mode == BinaryInputMode::kSource) {
                resources->driver = std::make_unique<DriverDispatch>();
                require(resources->driver->initialize(), "native dispatch initialization");
                resources->runtime =
                    std::make_unique<PersistentKernelRuntime>(*resources->driver, 3, false, true);
                check(resources->runtime->load_module(&resources->module, original.data()),
                      "original source module");
                check(resources->runtime->get_function(&original_function, resources->module,
                                                       kActualEntry),
                      "original source function");
                CUfunction direct{};
                check(cuModuleGetFunction(&direct, resources->module, kActualEntry),
                      "original direct identity");
                std::size_t offset{}, size{};
                check(cuFuncGetParamInfo(original_function, 2, &offset, &size), "original ABI");
                require(direct == original_function && offset == 8 && size == 24,
                        "original handle and ABI preservation");
                std::fill(original.begin(), original.end(), std::byte{});
                resources->runtime->prepare_module_source(original_function);
                auto variant = resources->runtime->mapped_binary_kernel(original_function);
                require(variant && variant->context == resources->context &&
                            variant->function != original_function,
                        "owned source-qualified private function");
                function = variant->function;
                resources->private_owner = std::move(variant->resources);
            } else {
                auto mapped = rewrite_qualified_add_cubin(
                    original, static_cast<unsigned int>(major * 10 + minor));
                require(mapped.has_value(), "framework cubin qualification/rewrite");
                check(cuModuleLoadData(&resources->module, mapped->data()),
                      "owned rewritten cubin");
            }
        } else {
            check(cuModuleLoad(&resources->module, binary), "qualified mapped cubin");
        }
    } else {
        check(cuModuleLoadData(&resources->module, kFixture), "private ABI ownership fixture");
    }
    if (!function) {
        check(cuModuleGetFunction(&function, resources->module,
                                  binary ? kActualEntry : "mapped_add_test"),
              "entry");
    }
    for (std::size_t parameter = 0; parameter != 3; ++parameter) {
        std::size_t offset{}, size{};
        check(cuFuncGetParamInfo(function, parameter, &offset, &size), "private ABI");
        require(offset == parameter * 4 && size == (parameter == 2 ? 32 : 4),
                "unexpected private ABI");
    }
    check(cuStreamCreate(&resources->low, CU_STREAM_NON_BLOCKING), "low stream");
    check(cuStreamCreate(&resources->high, CU_STREAM_NON_BLOCKING), "high stream");
    constexpr int count = 32769;
    constexpr std::size_t guard = 16;
    std::vector<float> input(count + 2 * guard, -123.0F);
    for (int element = 0; element != count; ++element)
        input[guard + element] = static_cast<float>(element % 113 + 1);
    const auto bytes = input.size() * sizeof(float);
    check(cuMemAlloc(&resources->data, bytes), "low allocation");
    check(cuMemAlloc(&resources->high_data, bytes), "high allocation");
    check(cuMemcpyHtoD(resources->data, input.data(), bytes), "low upload");
    check(cuMemcpyHtoD(resources->high_data, input.data(), bytes), "high upload");
    // Pageable HtoD may return after staging, before DMA completes. These
    // backend streams are nonblocking and cannot inherit default-stream order.
    check(cuStreamSynchronize(nullptr), "initial upload completion");
    CudaOperationQueue queue(resources->context, dispatch());
    auto low_stream = queue.bind_stream(resources->low, OrderedStreamKind::kNonBlocking, resources);
    auto high_stream =
        queue.bind_stream(resources->high, OrderedStreamKind::kNonBlocking, resources);
    auto low = prepared(function, resources->data + guard * sizeof(float), resources, count);
    auto continuation = std::dynamic_pointer_cast<CudaBinarySliceContinuation>(low.continuation);
    auto work = queue.submit(low_stream, std::move(low));
    auto after = queue.submit(low_stream, CudaQueuedMarker{});
    const std::array admitted{work};
    check(queue.pump_admitted(admitted), "first admitted slice");
    check(queue.request_retreat(work), "retreat to slice boundary");
    progress_until(queue, [&] { return queue.state(work) == OrderedOperationState::kSuspended; });
    require(continuation->progress().completed_blocks == 1 &&
                queue.query(after) == CUDA_ERROR_NOT_READY,
            "partial stop keeps logical dependency pending");
    unsigned int readback_phase = 0;
    auto check_output = [&](CUdeviceptr pointer, std::size_t completed_elements) {
        ++readback_phase;
        std::vector<float> actual(input.size());
        check(cuMemcpyDtoH(actual.data(), pointer, bytes), "output readback");
        for (std::size_t index = 0; index != actual.size(); ++index) {
            const float expected = index >= guard && index < guard + completed_elements
                                       ? input[index] * 2
                                       : input[index];
            if (actual[index] != expected) {
                std::cerr << "readback_phase=" << readback_phase
                          << " buffer=" << (pointer == resources->data ? "low" : "high")
                          << " completed_elements=" << completed_elements << " index=" << index
                          << " actual=" << actual[index] << " expected=" << expected << '\n';
                throw std::runtime_error("lost, repeated, pending or out-of-range slice write");
            }
        }
    };
    check_output(resources->data, 1024);
    auto high = queue.submit(
        high_stream,
        prepared(function, resources->high_data + guard * sizeof(float), resources, 1024));
    const std::array high_admitted{high};
    check(queue.pump_admitted(high_admitted), "independent admitted insertion");
    progress_until(queue, [&] { return queue.query(high) == CUDA_SUCCESS; });
    check_output(resources->high_data, 1024);
    check_output(resources->data, 1024);
    require(continuation->progress().completed_blocks == 1,
            "independent work cannot advance low continuation");
    while (queue.query(work) == CUDA_ERROR_NOT_READY) {
        require(queue.resume(work), "explicit retained resume");
        check(queue.pump_admitted(admitted), "resumed admitted slice");
        progress_until(queue, [&] { return queue.state(work) != OrderedOperationState::kRunning; });
    }
    progress_until(queue, [&] { return queue.query(after) == CUDA_SUCCESS; });
    check(queue.query(work), "final logical completion");
    require(continuation->progress().completed_blocks == 33 &&
                continuation->progress().physical_launches == 33,
            "exactly one launch per retained logical block");
    check_output(resources->data, count);
    check(cuMemcpyHtoD(resources->data, input.data(), bytes), "cancel input reset");
    check(cuStreamSynchronize(nullptr), "cancel reset upload completion");
    auto cancelled = queue.submit(
        low_stream, prepared(function, resources->data + guard * sizeof(float), resources, 2049));
    const std::array cancel_admitted{cancelled};
    check(queue.pump_admitted(cancel_admitted), "cancel first slice");
    require(!queue.cancel_stopped(cancelled), "no cancellation before physical proof");
    progress_until(queue,
                   [&] { return queue.state(cancelled) == OrderedOperationState::kSuspended; });
    require(queue.cancel_stopped(cancelled) && !queue.resume(cancelled),
            "cancel stopped continuation without replay");
    check_output(resources->data, 1024);
    check(queue.shutdown(), "queue drain");
    if (mode == BinaryInputMode::kSource) {
        check(resources->runtime->unload_module(resources->module),
              "drained original source unload");
        resources->module = nullptr;
        require(!resources->runtime->mapped_binary_kernel(original_function),
                "retired source association");
        int threads{};
        check(cuFuncGetAttribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function),
              "retained private function after original unload");
        require(threads == 128, "retained qualified private function");
        resources->private_owner.reset();
        check(resources->runtime->forget_context_checked(resources->context),
              "release retired binary library");
    }
    std::cout << "source="
              << (mode == BinaryInputMode::kSource
                      ? "owned_source_module"
                      : (mode == BinaryInputMode::kRewrite
                             ? "rewritten_original"
                             : (binary ? "qualified_cubin" : "abi_fixture")))
              << " completed_blocks=33 independent_insertion=1 stopped_cancel=1\n";
}

// Ordinary CUDA-only path: run unchanged natively and with LD_PRELOAD. No
// private Glimmer preparation, queue, lookup or retreat API is called here.
void run_ordinary_source(const char* binary, bool should_test_priority, bool should_test_fallback,
                         bool is_fatbinary, bool has_wrapper, bool should_capture_source,
                         bool is_large) {
    for (int round = 0; round != 2; ++round) {
        const auto resolve_library = [round](const char* name) {
            void* address{};
            if (round == 0) {
                address = dlsym(RTLD_DEFAULT, name);
            } else {
                CUdriverProcAddressQueryResult status{};
                check(cuGetProcAddress(name, &address, 12000, CU_GET_PROC_ADDRESS_DEFAULT, &status),
                      "ordinary cached library entry");
                require(status == CU_GET_PROC_ADDRESS_SUCCESS, "library resolver status");
            }
            require(address != nullptr, "library entry address");
            return address;
        };
        auto unload_library = &cuLibraryUnload;
        auto resources = std::make_shared<Resources>();
        check(cuInit(0), "ordinary init");
        CUdevice device{};
        check(cuDeviceGet(&device, 0), "ordinary device");
        check(cuCtxCreate(&resources->context, 0, device), "ordinary context");
        if (should_capture_source) {
            check(cuStreamCreate(&resources->low, CU_STREAM_NON_BLOCKING), "source capture stream");
            check(cuStreamBeginCapture(resources->low, CU_STREAM_CAPTURE_MODE_GLOBAL),
                  "source capture begin");
        }
        std::ifstream file(binary, std::ios::binary | std::ios::ate);
        const std::size_t image_bytes = is_fatbinary ? 5072120 : 5336008;
        require(file.good() && file.tellg() == static_cast<std::streamoff>(image_bytes),
                "ordinary original image");
        std::vector<char> image(image_bytes);
        file.seekg(0);
        file.read(image.data(), static_cast<std::streamsize>(image.size()));
        require(file.good(), "ordinary image read");
        __fatBinC_Wrapper_t wrapper{FATBINC_MAGIC, FATBINC_VERSION,
                                    reinterpret_cast<const unsigned long long*>(image.data()),
                                    nullptr};
        static_assert(sizeof(wrapper) == 24);
        if (has_wrapper) {
            const auto load_library = reinterpret_cast<decltype(&cuLibraryLoadData)>(
                resolve_library("cuLibraryLoadData"));
            const auto get_module = reinterpret_cast<decltype(&cuLibraryGetModule)>(
                resolve_library("cuLibraryGetModule"));
            unload_library =
                reinterpret_cast<decltype(&cuLibraryUnload)>(resolve_library("cuLibraryUnload"));
            check(load_library(&resources->library, &wrapper, nullptr, nullptr, 0, nullptr, nullptr,
                               0),
                  "ordinary source library load");
            check(get_module(&resources->module, resources->library),
                  "ordinary borrowed library module");
        } else {
            check(cuModuleLoadData(&resources->module, image.data()), "ordinary source load");
        }
        wrapper.data = nullptr;
        std::fill(image.begin(), image.end(), 0);
        CUfunction function{};
        check(cuModuleGetFunction(&function, resources->module, kActualEntry),
              "ordinary original function");
        if (should_capture_source) {
            CUgraph graph{};
            check(cuStreamEndCapture(resources->low, &graph), "source capture end");
            const auto destroyed = cuGraphDestroy(graph);
            check(destroyed, "empty source graph destroy");
        }
        std::size_t offset{}, size{};
        check(cuFuncGetParamInfo(function, 2, &offset, &size), "ordinary original ABI");
        require(offset == 8 && size == 24, "application still sees original parameter ABI");
        if (!resources->low)
            check(cuStreamCreate(&resources->low, CU_STREAM_NON_BLOCKING), "ordinary stream");
        if (should_test_priority)
            check(cuStreamCreateWithPriority(&resources->high, CU_STREAM_NON_BLOCKING, -1),
                  "ordinary high priority stream");
        constexpr std::size_t guard = 16;
        const std::size_t maximum_elements =
            (is_large || should_test_fallback ? 4194306 : 32769) + 2 * guard;
        check(cuMemAlloc(&resources->data, maximum_elements * sizeof(float)),
              "ordinary allocation");
        if (should_test_priority)
            check(cuMemAlloc(&resources->high_data, maximum_elements * sizeof(float)),
                  "ordinary high allocation");
        for (int repetition = 0; repetition != 3; ++repetition) {
            const int count = should_test_fallback && repetition == 0 ? 4194306
                              : is_large                              ? 4194303 + repetition
                                                                      : 32769;
            std::vector<float> input(count + 2 * guard, -123.0F);
            for (int index = 0; index != count; ++index)
                input[guard + index] = static_cast<float>(index % 113 + 1);
            const auto bytes = input.size() * sizeof(float);
            check(cuMemcpyHtoD(resources->data, input.data(), bytes), "ordinary upload");
            if (should_test_priority)
                check(cuMemcpyHtoD(resources->high_data, input.data(), bytes),
                      "ordinary high upload");
            check(cuStreamSynchronize(nullptr), "ordinary upload completion");
            auto high_launch = [&] {
                int elements = 1024;
                float alpha = 1;
                const auto pointer = resources->high_data + guard * sizeof(float);
                std::array<CUdeviceptr, 3> pointers{pointer, pointer, pointer};
                std::array<void*, 3> arguments{&elements, &alpha, pointers.data()};
                check(cuLaunchKernel(function, 1, 1, 1, 128, 1, 1, 0, resources->high,
                                     arguments.data(), nullptr),
                      "ordinary high original launch");
                elements = 0;
                pointers.fill(0);
            };
            if (should_test_priority) {
                high_launch();
                check(cuStreamSynchronize(resources->high), "ordinary warm high completion");
            }
            int elements = count;
            float alpha = 1.0F;
            const auto pointer = resources->data + guard * sizeof(float);
            std::array<CUdeviceptr, 3> pointers{pointer, pointer, pointer};
            std::array<void*, 3> arguments{&elements, &alpha, pointers.data()};
            struct PackedArguments {
                int count;
                float alpha;
                std::array<CUdeviceptr, 3> pointers;
            } packed{elements, alpha, pointers};
            static_assert(sizeof(PackedArguments) == 32);
            std::size_t packed_size = sizeof(packed);
            std::array<void*, 5> extra{CU_LAUNCH_PARAM_BUFFER_POINTER, &packed,
                                       CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                                       CU_LAUNCH_PARAM_END};
            const bool use_extra = should_test_fallback && repetition == 2;
            const unsigned int shared = should_test_fallback && repetition == 1 ? 16 : 0;
            const auto blocks = (static_cast<unsigned int>(count) + 1023) / 1024;
            check(cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, shared, resources->low,
                                 use_extra ? nullptr : arguments.data(),
                                 use_extra ? extra.data() : nullptr),
                  "ordinary original launch");
            elements = 0;
            alpha = 0;
            pointers.fill(0);
            packed = {};
            if (should_test_priority) {
                if (is_large) std::this_thread::sleep_for(std::chrono::microseconds{100});
                high_launch();
                check(cuStreamSynchronize(resources->high), "ordinary inserted high completion");
                const auto low_status = cuStreamQuery(resources->low);
                require(low_status == CUDA_SUCCESS || low_status == CUDA_ERROR_NOT_READY,
                        "ordinary pending low query");
                std::cout << "ordinary_priority high_completed=1 low_pending="
                          << (low_status == CUDA_ERROR_NOT_READY) << '\n';
            }
            check(cuStreamSynchronize(resources->low), "ordinary logical completion");
            std::vector<float> output(input.size());
            check(cuMemcpyDtoH(output.data(), resources->data, bytes), "ordinary readback");
            for (std::size_t index = 0; index != output.size(); ++index)
                require(output[index] == (index >= guard && index < guard + count ? 2 * input[index]
                                                                                  : input[index]),
                        "ordinary guarded original result");
            if (should_test_priority) {
                check(cuMemcpyDtoH(output.data(), resources->high_data, bytes),
                      "ordinary high readback");
                for (std::size_t index = 0; index != output.size(); ++index)
                    require(
                        output[index] == (index >= guard && index < guard + 1024 ? 4 * input[index]
                                                                                 : input[index]),
                        "ordinary high guarded exact-once result");
            }
        }
        check(resources->library ? unload_library(resources->library)
                                 : cuModuleUnload(resources->module),
              "ordinary source unload");
        resources->library = nullptr;
        resources->module = nullptr;
    }
    std::cout << "ordinary_source=1 original_abi=1 guarded_launches="
              << (should_test_priority ? 18 : 6) << " contexts=2 fallback=" << should_test_fallback
              << " priority=" << should_test_priority << " fatbinary=" << is_fatbinary
              << " wrapper=" << has_wrapper << " captured_source=" << should_capture_source << '\n';
    if (is_large) std::cout << "ordinary_range=1 maximum_elements=4194305\n";
}
}  // namespace
int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]).starts_with("--ordinary-")) {
        auto ordinary_mode = std::string_view(argv[1]).substr(11);
        const bool has_wrapper = ordinary_mode.starts_with("wrapper-");
        const bool is_fatbinary = has_wrapper || ordinary_mode.starts_with("fatbin-");
        if (is_fatbinary) ordinary_mode.remove_prefix(has_wrapper ? 8 : 7);
        const bool is_large = ordinary_mode.starts_with("large-");
        if (is_large) ordinary_mode.remove_prefix(6);
        if (ordinary_mode != "source" && ordinary_mode != "priority" &&
            ordinary_mode != "fallback" && ordinary_mode != "capture")
            return 2;
        try {
            run_ordinary_source(argv[2], ordinary_mode == "priority", ordinary_mode == "fallback",
                                is_fatbinary, has_wrapper, ordinary_mode == "capture", is_large);
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
        return cleanup_ok ? 0 : 1;
    }
    auto mode = BinaryInputMode::kMapped;
    if (argc == 3 && std::string_view(argv[1]) == "--original-cubin")
        mode = BinaryInputMode::kRewrite;
    if (argc == 3 && std::string_view(argv[1]) == "--source-module")
        mode = BinaryInputMode::kSource;
    if (argc > 2 && mode == BinaryInputMode::kMapped) return 2;
    try {
        run(argc == 3 ? argv[2] : (argc == 2 ? argv[1] : nullptr), mode);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return cleanup_ok ? 0 : 1;
}
