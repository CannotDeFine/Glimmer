#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/dnn_reorder_fixture.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <source_location>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_reorder_native_calls();
namespace {
using namespace glimmer::app;
namespace fixture = glimmer::tests::dnn_reorder;
namespace foundation = fixture::driver;
namespace base = fixture::base;
using Clock = std::chrono::steady_clock;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public reorder SDK " + std::to_string(status) + " at " +
                                 std::to_string(where.line()));
}
void check(CUresult status, std::source_location where = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public reorder Driver " + std::to_string(status) + " at " +
                                 std::to_string(where.line()));
}
using Reorder = decltype(&cudnnReorderFilterAndBias);
Reorder lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent reorder caller provider");
    auto* explicit_symbol = ::dlsym(library, "cudnnReorderFilterAndBias");
    expect(explicit_symbol == reinterpret_cast<void*>(&cudnnReorderFilterAndBias) &&
               ::dlsym(RTLD_DEFAULT, "cudnnReorderFilterAndBias") == explicit_symbol,
           "direct/global/explicit lookup selects controlled reorder wrapper");
    expect(::dlclose(library) == 0, "close explicit provider handle");
    return reinterpret_cast<Reorder>(explicit_symbol);
}
struct Case {
    cudnnFilterDescriptor_t descriptor{};
    glimmer::backend::cuda::CudaDnnFilter filter;
    std::uint32_t mode{};
    int flag{};
    std::array<std::size_t, 4> extents{}, slots{}, offsets{};
    std::array<CUdeviceptr, 4> pointers{};
    std::vector<CUdeviceptr> allocations;
    std::vector<std::vector<std::byte>> images;
    Case(std::uint32_t type, std::uint32_t reorder, int bias, bool combined = false,
         bool read_alias = false)
        : filter(fixture::filter(type)), mode(reorder), flag(bias) {
        check(cudnnCreateFilterDescriptor(&descriptor));
        configure();
        extents = {fixture::filter_bytes(filter), fixture::filter_bytes(filter),
                   bias ? fixture::bias_bytes(filter, mode) : 0,
                   bias ? fixture::bias_bytes(filter, mode) : 0};
        std::size_t total{};
        for (std::size_t i = 0; i != 4; ++i) {
            if (!extents[i]) continue;
            if (read_alias && i == 2) {
                expect(extents[i] <= extents[0], "read-only bias aliases a complete input range");
                slots[i] = slots[0];
                offsets[i] = offsets[0];
                continue;
            }
            slots[i] = combined ? 0 : images.size();
            offsets[i] = combined ? total + 16 : 16;
            if (combined)
                total += extents[i] + 32;
            else
                images.emplace_back(extents[i] + 32, std::byte{0xcd});
        }
        if (combined) images.emplace_back(total, std::byte{0xcd});
        for (std::size_t i : {0U, 2U}) {
            if (!extents[i] || (read_alias && i == 2)) continue;
            for (std::size_t j = 0; j != extents[i]; ++j)
                images[slots[i]][offsets[i] + j] =
                    static_cast<std::byte>((j * 37 + i * 19 + 11) & 255);
        }
        for (const auto& image : images) {
            CUdeviceptr allocation{};
            check(cuMemAlloc(&allocation, image.size()));
            allocations.push_back(allocation);
            check(cuMemcpyHtoD(allocation, image.data(), image.size()));
        }
        for (std::size_t i = 0; i != 4; ++i)
            if (extents[i]) pointers[i] = allocations[slots[i]] + offsets[i];
    }
    void configure() {
        check(cudnnSetFilterNdDescriptor(descriptor, static_cast<cudnnDataType_t>(filter.type),
                                         static_cast<cudnnTensorFormat_t>(filter.format),
                                         filter.dimensions, filter.sizes.data()));
    }
    cudnnStatus_t call(Reorder function, cudnnHandle_t handle) const {
        const auto before = fixture::control.calls.load();
        const auto status = function(
            handle, descriptor, static_cast<decltype(CUDNN_DEFAULT_REORDER)>(mode),
            reinterpret_cast<const void*>(pointers[0]), reinterpret_cast<void*>(pointers[1]), flag,
            reinterpret_cast<const void*>(pointers[2]), reinterpret_cast<void*>(pointers[3]));
        if (fixture::control.calls.load() != before)
            expect(
                fixture::control.calls.load() == before + 1 &&
                    fixture::control.last_flag.load() == flag &&
                    fixture::control.last_reorder.load() == mode,
                "public IPC preserves the full signed flag and mode in exactly one original call");
        return status;
    }
    void verify(bool invoked = true, bool failed = false) const {
        auto expected = images;
        if (invoked)
            for (std::size_t input : {0U, 2U}) {
                if (!extents[input] || (failed && input == 2)) continue;
                std::vector<std::byte> source(
                    images[slots[input]].begin() + static_cast<std::ptrdiff_t>(offsets[input]),
                    images[slots[input]].begin() +
                        static_cast<std::ptrdiff_t>(offsets[input] + extents[input]));
                if (!mode) std::reverse(source.begin(), source.end());
                std::copy(source.begin(), source.end(),
                          expected[slots[input + 1]].begin() +
                              static_cast<std::ptrdiff_t>(offsets[input + 1]));
            }
        for (std::size_t i = 0; i != allocations.size(); ++i) {
            std::vector<std::byte> actual(images[i].size());
            check(cuMemcpyDtoH(actual.data(), allocations[i], actual.size()));
            expect(actual == expected[i],
                   "complete synthetic outputs, unchanged inputs and allocation guards match");
        }
    }
    void finish() {
        for (auto allocation : allocations) check(cuMemFree(allocation));
        allocations.clear();
        check(cudnnDestroyFilterDescriptor(descriptor));
        descriptor = nullptr;
    }
};
void ignored_bias(Reorder function, cudnnHandle_t handle) {
    Case value(8, 0, 0);
    auto* inaccessible = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(inaccessible != MAP_FAILED, "allocate inaccessible ignored-pointer guard");
    value.pointers[2] = reinterpret_cast<CUdeviceptr>(inaccessible);
    value.pointers[3] = std::numeric_limits<CUdeviceptr>::max() - 3;
    const auto status = value.call(function, handle);
    expect(::munmap(inaccessible, 4096) == 0, "release ignored-pointer guard");
    check(status);
    value.verify();
    value.finish();
}
void rejected(Reorder function, cudnnHandle_t handle) {
    Case value(8, 1, -1, true);
    const auto original = value.pointers;
    const auto calls = fixture::control.calls.load();
    for (std::size_t i = 0; i != 4; ++i) {
        value.pointers = original;
        value.pointers[i] = value.allocations[0] + value.images[0].size() - value.extents[i] + 1;
        expect(value.call(function, handle) == CUDNN_STATUS_BAD_PARAM,
               "one-byte-short complete native range rejects before original execution");
    }
    for (const auto& pair : {std::pair{1U, 0U}, std::pair{3U, 2U}, std::pair{3U, 0U},
                             std::pair{1U, 2U}, std::pair{3U, 1U}}) {
        value.pointers = original;
        value.pointers[pair.first] = value.pointers[pair.second];
        expect(value.call(function, handle) == CUDNN_STATUS_BAD_PARAM,
               "all active write aliases reject instead of dispatching unsafe permutation");
    }
    value.pointers = original;
    value.pointers[2] = 0;
    expect(value.call(function, handle) == CUDNN_STATUS_BAD_PARAM,
           "negative active flag cannot bypass bias ownership with null input");
    value.pointers = original;
    for (std::size_t version : {std::size_t{0}, std::size_t{91001}, std::size_t{92301}}) {
        fixture::control.version.store(version);
        expect(value.call(function, handle) == CUDNN_STATUS_NOT_SUPPORTED,
               "unqualified runtime version exposes honest NOT_SUPPORTED rather than BAD_PARAM");
    }
    fixture::control.version.store(91002);
    fixture::control.query_status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    expect(value.call(function, handle) == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
           "native FilterNd error is returned integer-exact");
    fixture::control.query_status.store(CUDNN_STATUS_SUCCESS);
    fixture::control.size_status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    expect(value.call(function, handle) == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE,
           "native FilterSize error is returned integer-exact");
    fixture::control.size_status.store(CUDNN_STATUS_SUCCESS);
    expect(fixture::control.calls.load() == calls, "all rejected admissions execute no reorder");
    value.verify(false);
    value.finish();
}
void pending(Reorder function, cudnnHandle_t handle, CUcontext context, bool failed) {
    Case value(8, failed ? 0 : 1, INT_MIN);
    foundation::control.can_stop.store(false);
    fixture::control.compute_status.store(failed ? CUDNN_STATUS_EXECUTION_FAILED
                                                 : CUDNN_STATUS_SUCCESS);
    expect(value.call(function, handle) ==
               (failed ? CUDNN_STATUS_EXECUTION_FAILED : CUDNN_STATUS_SUCCESS),
           "host result preserves partial native failure or original success");
    fixture::control.compute_status.store(CUDNN_STATUS_SUCCESS);
    const auto calls = fixture::control.calls.load();
    std::size_t bytes{};
    check(cudnnGetFilterSizeInBytes(value.descriptor, &bytes));
    expect(bytes == value.extents[0] && !foundation::control.can_stop.load() &&
               fixture::control.calls.load() == calls,
           "native metadata query progresses without draining pending original work");
    auto mutation = std::async(std::launch::async, [&] { value.configure(); });
    auto stream = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto result = cudnnSetStream(handle, nullptr);
        check(cuCtxSetCurrent(nullptr));
        return result;
    });
    const bool waits =
        mutation.wait_for(std::chrono::milliseconds{25}) != std::future_status::ready &&
        stream.wait_for(std::chrono::milliseconds{25}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    mutation.get();
    check(stream.get());
    expect(waits, "metadata mutation and handle stream mutation wait for physical completion");
    value.verify(true, failed);
    value.finish();
}
void pending_cleanup(Reorder function, cudnnHandle_t handle, CUcontext context) {
    Case value(8, 0, -1);
    foundation::control.can_stop.store(false);
    check(value.call(function, handle));
    auto descriptor = std::async(std::launch::async,
                                 [&] { return cudnnDestroyFilterDescriptor(value.descriptor); });
    auto native_handle = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto result = cudnnDestroy(handle);
        check(cuCtxSetCurrent(nullptr));
        return result;
    });
    std::array<std::future<CUresult>, 4> frees;
    for (std::size_t i = 0; i != frees.size(); ++i)
        frees[i] = std::async(std::launch::async, [&, i] {
            check(cuCtxSetCurrent(context));
            const auto result = cuMemFree(value.allocations[i]);
            check(cuCtxSetCurrent(nullptr));
            return result;
        });
    bool waits = descriptor.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
                 native_handle.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    for (auto& released : frees)
        waits &= released.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    check(descriptor.get());
    check(native_handle.get());
    for (auto& released : frees) check(released.get());
    expect(waits, "filter, handle and all four allocations remain leased past public host return");
}
void application(const char* provider) {
    const auto function = lookup(provider);
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cudnnHandle_t handle{};
    check(cudnnCreate(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
    for (std::uint32_t mode : {0U, 1U})
        for (int flag : {0, 1, -1, INT_MIN, INT_MAX}) {
            Case value(8, mode, flag, flag < 0);
            check(value.call(function, handle));
            value.verify();
            value.finish();
        }
    for (std::uint32_t type : {0U, 1U, 5U, 7U, 9U}) {
        Case value(type, 1, -1);
        check(value.call(function, handle));
        value.verify();
        value.finish();
    }
    {
        Case value(8, 0, 1, false, true);
        check(value.call(function, handle));
        value.verify();
        value.finish();
    }
    {
        Case value(0, 0, 1);
        expect(value.call(function, handle) == CUDNN_STATUS_NOT_SUPPORTED,
               "safe unsupported DEFAULT combination preserves native error");
        value.verify(false);
        value.finish();
    }
    ignored_bias(function, handle);
    rejected(function, handle);
    check(cudnnSetStream(handle, nullptr));
    pending(function, handle, context, false);
    pending(function, handle, context, true);
    const auto child = ::fork();
    expect(child >= 0, "fork inherited public reorder identity check");
    if (!child)
        ::_exit(function(handle, nullptr, CUDNN_DEFAULT_REORDER, nullptr, nullptr, -1, nullptr,
                         nullptr) == CUDNN_STATUS_NOT_INITIALIZED
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork child cannot access inherited execution owner");
    pending_cleanup(function, handle, context);
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));
    expect(glimmer_dnn_reorder_native_calls() == 0,
           "caller-native provider is unused across direct/global/explicit/cached route");
}
void run(const char* provider) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual reorder service process identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-reorder-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 64;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated owned reorder IPC");
    fixture::reset();
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start independent reorder service");
    auto client = std::async(std::launch::async, [&] { application(provider); });
    const auto deadline = Clock::now() + std::chrono::seconds{45};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public reorder progress");
    client.get();
    expect(service.request_stop() && service.finished(), "all proven reorder ownership settles");
    expect(fixture::state.creates == fixture::state.destroys && fixture::state.creates == 22 &&
               fixture::state.calls == 21 && !fixture::state.four_queries &&
               base::state.handle_creates == 1 && base::state.handle_destroys == 1 &&
               foundation::driver.memory.empty() && foundation::driver.contexts.empty() &&
               foundation::driver.events.empty() && foundation::driver.streams.empty() &&
               !foundation::control.context_synchronizes.load() &&
               foundation::control.host_registers.load() ==
                   foundation::control.host_unregisters.load() &&
               glimmer_dnn_reorder_native_calls() == 0,
           "twenty-one original calls and all resource owners retire without native fallback");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2, "independent reorder provider path");
        run(argv[1]);
        std::cout << "Reorder public synthetic CPU byte and IPC ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
