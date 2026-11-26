#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/dnn_dropout_fixture.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <source_location>
#include <dlfcn.h>
#include <unistd.h>

extern "C" unsigned int glimmer_dnn_dropout_native_calls();
namespace {
using namespace glimmer::app;
namespace fixture = glimmer::tests::dnn_dropout;
namespace foundation = fixture::driver;
namespace base = glimmer::tests::dnn;
using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kSeed = 0xfedcba9876543210ULL;
constexpr std::size_t kBytes = fixture::kStateBytes + 16;
constexpr std::size_t kReported = std::numeric_limits<std::size_t>::max() - 7;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public dropout SDK " + std::to_string(status) + " at " +
                                 std::to_string(where.line()));
}
void check(CUresult status, std::source_location where = std::source_location::current()) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error("public dropout Driver " + std::to_string(status) + " at " +
                                 std::to_string(where.line()));
}
template <typename Function>
void symbol(void* library, const char* name, Function direct) {
    expect(::dlsym(library, name) == reinterpret_cast<void*>(direct) &&
               ::dlsym(RTLD_DEFAULT, name) == reinterpret_cast<void*>(direct),
           "direct/global/explicit provider selects controlled dropout wrapper");
}
struct Cached {
    decltype(&cudnnCreateDropoutDescriptor) create{};
    decltype(&cudnnDestroyDropoutDescriptor) destroy{};
    decltype(&cudnnDropoutGetStatesSize) size{};
    decltype(&cudnnSetDropoutDescriptor) set{};
    decltype(&cudnnRestoreDropoutDescriptor) restore{};
};
Cached lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent dropout caller provider");
#define CHECK(NAME) symbol(library, #NAME, &(NAME))
    CHECK(cudnnCreateDropoutDescriptor);
    CHECK(cudnnDestroyDropoutDescriptor);
    CHECK(cudnnDropoutGetStatesSize);
    CHECK(cudnnSetDropoutDescriptor);
    CHECK(cudnnRestoreDropoutDescriptor);
    CHECK(cudnnGetDropoutDescriptor);
    CHECK(cudnnDropoutGetReserveSpaceSize);
    CHECK(cudnnDropoutForward);
    CHECK(cudnnDropoutBackward);
#undef CHECK
    const Cached cached{
        reinterpret_cast<decltype(cached.create)>(::dlsym(library, "cudnnCreateDropoutDescriptor")),
        reinterpret_cast<decltype(cached.destroy)>(
            ::dlsym(library, "cudnnDestroyDropoutDescriptor")),
        reinterpret_cast<decltype(cached.size)>(::dlsym(library, "cudnnDropoutGetStatesSize")),
        reinterpret_cast<decltype(cached.set)>(::dlsym(library, "cudnnSetDropoutDescriptor")),
        reinterpret_cast<decltype(cached.restore)>(
            ::dlsym(library, "cudnnRestoreDropoutDescriptor"))};
    expect(::dlclose(library) == 0, "cached dropout wrappers outlive explicit provider handle");
    return cached;
}
void guards(cudnnDropoutDescriptor_t descriptor) {
    float probability = 17;
    void* states = reinterpret_cast<void*>(0x1234);
    unsigned long long seed = 29;
    std::size_t bytes = 31;
    expect(cudnnGetDropoutDescriptor(descriptor, nullptr, &probability, &states, &seed) ==
                   CUDNN_STATUS_NOT_SUPPORTED &&
               probability == 17 && states == reinterpret_cast<void*>(0x1234) && seed == 29 &&
               cudnnDropoutGetReserveSpaceSize(nullptr, &bytes) == CUDNN_STATUS_NOT_SUPPORTED &&
               bytes == 31 &&
               cudnnDropoutForward(nullptr, descriptor, nullptr, nullptr, nullptr, nullptr, nullptr,
                                   0) == CUDNN_STATUS_NOT_SUPPORTED &&
               cudnnDropoutBackward(nullptr, descriptor, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, 0) == CUDNN_STATUS_NOT_SUPPORTED,
           "four reference stubs remain typed guards with untouched caller output");
}
std::vector<std::byte> image(std::uint64_t seed) {
    std::vector<std::byte> result(kBytes, std::byte{0xcd});
    const auto bytes = std::bit_cast<std::array<std::byte, 8>>(seed);
    for (std::size_t i = 0; i != fixture::kStateBytes; ++i)
        result[i + 8] = bytes[i % 8] ^ static_cast<std::byte>(0x5b + i);
    return result;
}
CUdeviceptr allocate() {
    CUdeviceptr pointer{};
    check(cuMemAlloc(&pointer, kBytes));
    const std::vector<std::byte> initial(kBytes, std::byte{0xcd});
    check(cuMemcpyHtoD(pointer, initial.data(), initial.size()));
    return pointer;
}
void verify(CUdeviceptr pointer, std::uint64_t seed) {
    std::vector<std::byte> actual(kBytes);
    check(cuMemcpyDtoH(actual.data(), pointer, actual.size()));
    expect(actual == image(seed), "complete state and guard bytes match independent CPU oracle");
}
void query(Cached cached, cudnnHandle_t handle) {
    std::size_t bytes = 17;
    check(cached.size(handle, &bytes));
    expect(bytes == kReported, "states-size query preserves full native size_t result");
    fixture::control.query_status.store(CUDNN_STATUS_NOT_SUPPORTED);
    bytes = 19;
    expect(cached.size(handle, &bytes) == CUDNN_STATUS_NOT_SUPPORTED && bytes == 19,
           "failed original size query preserves exact status and untouched output");
    fixture::control.query_status.store(CUDNN_STATUS_SUCCESS);
}
void pending_mutation(Cached cached, cudnnHandle_t handle, CUcontext context) {
    cudnnDropoutDescriptor_t descriptor{};
    check(cached.create(&descriptor));
    const auto memory = allocate();
    foundation::control.can_stop.store(false);
    check(cached.set(descriptor, handle, 0.25F, reinterpret_cast<void*>(memory + 8),
                     fixture::kStateBytes, kSeed));
    const auto calls = fixture::control.callbacks.load();
    query(cached, handle);
    expect(!foundation::control.can_stop.load() && fixture::control.callbacks.load() == calls,
           "public size query adds no compute and progresses while original event is NOT_READY");
    auto mutation = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto status =
            cached.restore(descriptor, handle, 0.5F, reinterpret_cast<void*>(memory + 8),
                           fixture::kStateBytes, kSeed);
        check(cuCtxSetCurrent(nullptr));
        return status;
    });
    const bool pending =
        mutation.wait_for(std::chrono::milliseconds{40}) != std::future_status::ready &&
        fixture::control.callbacks.load() == calls;
    foundation::control.can_stop.store(true);
    check(mutation.get());
    expect(pending,
           "configuration reservation survives host ACK and blocks a second native mutation");
    verify(memory, kSeed);
    check(cuMemFree(memory));
    check(cached.destroy(descriptor));
}
void pending_cleanup(Cached cached, cudnnHandle_t handle, CUcontext context) {
    cudnnDropoutDescriptor_t descriptor{};
    check(cached.create(&descriptor));
    const auto memory = allocate();
    foundation::control.can_stop.store(false);
    check(cached.set(descriptor, handle, 0.25F, reinterpret_cast<void*>(memory + 8),
                     fixture::kStateBytes, kSeed));
    auto metadata = std::async(std::launch::async, [&] { return cached.destroy(descriptor); });
    auto released = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto status = cuMemFree(memory);
        check(cuCtxSetCurrent(nullptr));
        return status;
    });
    auto destroyed = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        const auto status = cudnnDestroy(handle);
        check(cuCtxSetCurrent(nullptr));
        return status;
    });
    const bool pending =
        metadata.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        released.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        destroyed.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    foundation::control.can_stop.store(true);
    check(metadata.get());
    check(released.get());
    check(destroyed.get());
    expect(pending,
           "pending initialization holds metadata, caller state allocation and native handle");
}
void ordinary(const char* provider) {
    const auto cached = lookup(provider);
    cudnnDropoutDescriptor_t descriptor{};
    check(cached.create(&descriptor));
    guards(descriptor);
    expect(!foundation::control.context_creates.load(),
           "dropout create/guards need no CUDA context");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cudnnHandle_t handle{};
    check(cudnnCreate(&handle));
    CUstream stream{};
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
    query(cached, handle);
    check(cached.set(descriptor, handle, std::bit_cast<float>(0x80000000U), nullptr,
                     std::numeric_limits<std::size_t>::max(), kSeed));
    const auto first = allocate(), second = allocate();
    check(cached.set(descriptor, handle, 0.25F, reinterpret_cast<void*>(first + 8),
                     fixture::kStateBytes, kSeed));
    verify(first, kSeed);
    check(cached.restore(descriptor, handle, 0.5F, reinterpret_cast<void*>(first + 8),
                         fixture::kStateBytes, ~kSeed));
    verify(first, kSeed);
    check(cached.set(descriptor, handle, 0.75F, nullptr, 0x100000003ULL, ~kSeed));
    check(cached.restore(descriptor, handle, 0.5F, nullptr, 0x100000007ULL, kSeed));
    verify(first, kSeed);
    check(cached.set(descriptor, handle, 0.125F, reinterpret_cast<void*>(second + 8),
                     fixture::kStateBytes, ~kSeed));
    verify(second, ~kSeed);
    verify(first, kSeed);
    check(cuMemFree(first));
    check(cuMemFree(second));
    check(cached.destroy(descriptor));
    expect(cached.destroy(descriptor) == CUDNN_STATUS_NOT_INITIALIZED,
           "state may be freed before native descriptor destruction, and stale handle rejects");
    check(cudnnSetStream(handle, nullptr));
    pending_mutation(cached, handle, context);
    pending_cleanup(cached, handle, context);
    check(cuStreamDestroy(stream));
    check(cuCtxDestroy(context));

    check(cached.create(&descriptor));
    check(cuCtxCreate(&context, 0, 0));
    check(cudnnCreate(&handle));
    const auto bound = allocate();
    check(cached.set(descriptor, handle, 0.25F, reinterpret_cast<void*>(bound + 8),
                     fixture::kStateBytes, kSeed));
    check(cuCtxSynchronize());
    check(cudnnDestroy(handle));
    check(cuCtxDestroy(context));
    check(cached.destroy(descriptor));
    expect(glimmer_dnn_dropout_native_calls() == 0,
           "caller native provider remains unused across cached and direct execution");
}
void quarantined(const char* provider, bool proof_failure) {
    const auto cached = lookup(provider);
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    cudnnHandle_t handle{};
    check(cudnnCreate(&handle));
    cudnnDropoutDescriptor_t descriptor{};
    check(cached.create(&descriptor));
    const auto memory = allocate();
    if (proof_failure)
        foundation::control.synchronize_status.store(CUDA_ERROR_LAUNCH_FAILED);
    else {
        fixture::control.set_status.store(CUDNN_STATUS_BAD_PARAM);
        fixture::control.should_mutate_on_error.store(true);
    }
    const auto status = cached.set(descriptor, handle, 0.25F, reinterpret_cast<void*>(memory + 8),
                                   fixture::kStateBytes, kSeed);
    expect(status == (proof_failure ? CUDNN_STATUS_INTERNAL_ERROR : CUDNN_STATUS_BAD_PARAM),
           "public reply distinguishes original failure from failed proof after native SUCCESS");
    const auto calls = fixture::control.callbacks.load();
    expect(cached.restore(descriptor, handle, 0.5F, reinterpret_cast<void*>(memory + 8),
                          fixture::kStateBytes, kSeed) != CUDNN_STATUS_SUCCESS &&
               cached.destroy(descriptor) != CUDNN_STATUS_SUCCESS &&
               fixture::control.callbacks.load() == calls &&
               glimmer_dnn_dropout_native_calls() == 0,
           "quarantined binding cannot retry native configuration or destroy uncertain descriptor");
    // The service intentionally retains uncertain native metadata (and, after
    // failed physical proof, all GPU owners). The public frontend additionally
    // closes the execution owner; direct backend recovery is tested separately.
}
void run(const char* provider, bool quarantine, bool proof_failure) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual dropout service process identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-dropout-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 64;
    config.exit_when_idle = true;
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated original-owned dropout IPC");
    fixture::reset();
    fixture::state.reported_bytes = kReported;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start independent dropout service");
    auto client = std::async(std::launch::async, [=] {
        if (quarantine || proof_failure)
            quarantined(provider, proof_failure);
        else
            ordinary(provider);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public dropout progress");
    client.get();
    if (quarantine || proof_failure) {
        expect(fixture::state.sets == 1 && !fixture::state.restores && !fixture::state.destroys &&
                   fixture::state.initializations == 1 && foundation::driver.frees == 0 &&
                   base::state.handle_destroys == 0 && glimmer_dnn_dropout_native_calls() == 0,
               "isolated uncertain path retains exact owners without hidden replay/cleanup");
        std::cout << "Dropout public isolated quarantine checks passed\n" << std::flush;
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished(), "all proven dropout owners settle");
    expect(fixture::state.creates == 4 && fixture::state.destroys == 4 &&
               fixture::state.sets == 7 && fixture::state.restores == 3 &&
               fixture::state.initializations == 5 && foundation::driver.frees == 4 &&
               base::state.handle_creates == 2 && base::state.handle_destroys == 2 &&
               foundation::driver.memory.empty() && foundation::driver.contexts.empty() &&
               foundation::driver.events.empty() && foundation::driver.streams.empty() &&
               foundation::control.host_registers.load() &&
               foundation::control.host_registers.load() ==
                   foundation::control.host_unregisters.load() &&
               glimmer_dnn_dropout_native_calls() == 0,
           "ten original configurations, exact state writes and caller-owned cleanup settle once");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 3, "dropout provider path and optional isolated mode");
        const std::string_view mode = argc == 3 ? argv[2] : "";
        expect(mode.empty() || mode == "--quarantine" || mode == "--proof-failure",
               "known bounded dropout mode");
        run(argv[1], mode == "--quarantine", mode == "--proof-failure");
        std::cout << "Dropout public CPU state and ownership checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
