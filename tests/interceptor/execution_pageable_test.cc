#include "execution_staging_fixture.h"
#include "internal/process_identity.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <dlfcn.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
thread_local bool count_local_polls{};
thread_local unsigned int local_polls{};
}  // namespace
// Export only this test's poll shim so the real frontend socket path remains
// observable without altering its transport or affecting the worker thread.
extern "C" int poll(struct pollfd* descriptors, nfds_t count, int timeout) {
    if (count_local_polls) ++local_polls;
    return static_cast<int>(::syscall(SYS_poll, descriptors, count, timeout));
}

extern "C" unsigned int glimmer_memory_native_calls();
namespace {
using namespace glimmer::control;
using Clock = std::chrono::steady_clock;
void expect(bool valid, const char* message) {
    if (!valid) throw std::runtime_error(message);
}
void check(CUresult result) {
    expect(result == CUDA_SUCCESS, "public pageable operation succeeds");
}
struct Latches {
    std::atomic<bool> upload_returned{}, release_observed{}, download_withheld{}, may_reply{};
    std::atomic<bool> origin_closing{};
};
template <class Predicate>
void await(Predicate&& predicate) {
    const auto deadline = Clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
        expect(Clock::now() < deadline, "bounded public staging latch");
        std::this_thread::yield();
    }
}
void application(const char* sdk_path, Latches& latches, bool is_quarantine, bool has_idle_error) {
    void* sdk = ::dlopen(sdk_path, RTLD_NOW | RTLD_LOCAL);
    expect(sdk, "fake SDK metadata scope");
    auto query =
        reinterpret_cast<decltype(&cuGetProcAddress_v2)>(::dlsym(sdk, "cuGetProcAddress_v2"));
    expect(query, "cached Driver query entry");
    const auto resolve = [&](const char* name) {
        void* result{};
        CUdriverProcAddressQueryResult status{};
        check(query(name, &result, 12000, 0, &status));
        expect(result && status == CU_GET_PROC_ADDRESS_SUCCESS, "owned cached copy ABI");
        return result;
    };
    auto upload = reinterpret_cast<decltype(&cuMemcpyHtoDAsync)>(resolve("cuMemcpyHtoDAsync"));
    auto download = reinterpret_cast<decltype(&cuMemcpyDtoHAsync)>(resolve("cuMemcpyDtoHAsync"));
    auto runtime_copy =
        reinterpret_cast<decltype(&cudaMemcpyAsync)>(::dlsym(RTLD_DEFAULT, "cudaMemcpyAsync"));
    expect(runtime_copy, "public Runtime copy entry");
    CUcontext context{};
    check(cuCtxCreate(&context, 0, 0));
    CUdeviceptr device{};
    check(cuMemAlloc(&device, 256));
    std::array<unsigned char, 256> input{}, output{};
    input.fill(43);
    if (is_quarantine) {
        expect(upload(device, input.data(), input.size(), nullptr) == CUDA_ERROR_UNKNOWN &&
                   upload(device, input.data(), input.size(), nullptr) == CUDA_ERROR_UNKNOWN &&
                   cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN,
               "unproven staging registration retains ownership, blocks replay and seals context "
               "teardown");
        expect(glimmer_memory_native_calls() == 0 && ::dlclose(sdk) == 0,
               "unknown staging cannot fall back into caller-native resources");
        return;
    }
    expect(
        upload(device, reinterpret_cast<const void*>(1), 256, nullptr) == CUDA_ERROR_INVALID_VALUE,
        "unreadable input cannot import backing or issue native work");
    expect(upload(device, input.data(), input.size(), nullptr) == CUDA_ERROR_OUT_OF_MEMORY,
           "known empty native staging OOM preserves the original failure");
    check(cuMemcpyHtoDAsync(device, input.data(), input.size(), nullptr));
    input.fill(17);
    latches.upload_returned.store(true);
    // No CUDA API drives this wait: cleanup needs its own bounded transport
    // progress, and the upload must not wait for the deferred free result.
    await([&] { return latches.release_observed.load(); });
    if (!has_idle_error) {
        CUcontext observed{};
        check(cuCtxGetCurrent(&observed));  // Warm the original Driver initialization.
        local_polls = 0;
        count_local_polls = true;
        const auto result = cuCtxGetCurrent(&observed);
        count_local_polls = false;
        expect(result == CUDA_SUCCESS && observed == context && local_polls == 0,
               "unrelated context query does not poll the staging transport");
    }
    if (has_idle_error) {
        expect(cuCtxDestroy(context) == CUDA_ERROR_UNKNOWN,
               "mismatched idle receipt retains registration and rejects context teardown");
        expect(glimmer_memory_native_calls() == 0 && ::dlclose(sdk) == 0,
               "idle uncertainty cannot authorize native fallback or release");
        return;
    }
    auto pending = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return download(output.data(), device, output.size(), nullptr);
    });
    await([&] { return latches.download_withheld.load(); });
    unsigned int flags{99};
    check(cuCtxGetFlags(&flags));
    expect(!flags && pending.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
               output == std::array<unsigned char, 256>{},
           "unrelated metadata progresses while a download retains private, not public, results");
    latches.may_reply.store(true);
    expect(pending.get() == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES &&
               output == std::array<unsigned char, 256>{},
           "actual native failure cannot copy partial staging bytes into application output");
    check(download(output.data(), device, output.size(), nullptr));
    for (const auto value : output) expect(value == 43, "source snapshot survives caller mutation");
    const auto page = ::sysconf(_SC_PAGESIZE);
    expect(page > 256, "valid protection-test page geometry");
    auto* destination =
        static_cast<unsigned char*>(::mmap(nullptr, static_cast<std::size_t>(page) * 2, PROT_WRITE,
                                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    expect(destination != MAP_FAILED, "create write-only destination without readable permissions");
    check(download(destination, device, output.size(), nullptr));
    expect(::mprotect(destination, static_cast<std::size_t>(page), PROT_READ) == 0,
           "protect result against new writes");
    for (std::size_t index = 0; index < output.size(); ++index)
        expect(destination[index] == 43, "write-only destination receives all exact bytes");
    expect(download(destination, device, output.size(), nullptr) == CUDA_ERROR_INVALID_VALUE,
           "read-only download cannot issue GPU work or change output");
    expect(::mprotect(destination, static_cast<std::size_t>(page), PROT_WRITE) == 0 &&
               ::mprotect(destination + page, static_cast<std::size_t>(page), PROT_NONE) == 0,
           "create a writable prefix followed by an inaccessible VMA");
    expect(download(destination + page - 128, device, 256, nullptr) == CUDA_ERROR_INVALID_VALUE &&
               download(destination + page, device, 256, nullptr) == CUDA_ERROR_INVALID_VALUE,
           "partial writable coverage and inaccessible targets reject before import");
    expect(::munmap(destination, static_cast<std::size_t>(page) * 2) == 0,
           "release exact test mappings once");
    // Synthetic fake-SDK numeric UVA, never dereferenced by the application.
    // NOLINTBEGIN(performance-no-int-to-ptr)
    auto* target =
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(device));  // Synthetic SDK UVA.
    // NOLINTEND(performance-no-int-to-ptr)
    for (unsigned int iteration = 0; iteration < 32; ++iteration) {
        input.fill(static_cast<unsigned char>(iteration));
        expect(runtime_copy(target, input.data(), input.size(), cudaMemcpyDefault, nullptr) ==
                   cudaSuccess,
               "Runtime default pageable upload shares ownership");
        output.fill(255);
        expect(runtime_copy(output.data(), target, output.size(), cudaMemcpyDeviceToHost,
                            nullptr) == cudaSuccess &&
                   output == input,
               "Runtime pageable download returns complete exact bytes");
    }
    check(cuMemFree(device));
    check(cuCtxDestroy(context));
    expect(glimmer_memory_native_calls() == 0 && ::dlclose(sdk) == 0,
           "all pageable resources remain server-owned, not caller-native fallback");
}
void pressure_application(Latches& latches) {
    latches.upload_returned = true;
    for (unsigned int generation = 0; generation < 2; ++generation) {
        CUcontext context{};
        check(cuCtxCreate(&context, 0, 0));
        CUdeviceptr device{};
        check(cuMemAlloc(&device, 512));
        std::array<unsigned char, 512> input{}, output{};
        const unsigned int transfers = generation == 0 ? 300 : 1;
        for (unsigned int count = 1; count <= transfers; ++count) {
            input.fill(static_cast<unsigned char>(count));
            output.fill(0);
            check(cuMemcpyHtoDAsync(device, input.data(), count, nullptr));
            check(cuMemcpyDtoHAsync(output.data(), device, count, nullptr));
            for (unsigned int index = 0; index < count; ++index)
                expect(output[index] == input[index], "full pressure-transfer output matches");
        }
        latches.origin_closing = true;
        check(cuMemFree(device));
        check(cuCtxDestroy(context));
    }
    expect(glimmer_memory_native_calls() == 0, "cache pressure never uses caller-native fallback");
}
void run(const char* sdk, std::string_view mode) {
    const bool is_quarantine = mode == "--quarantine";
    const bool should_reuse = !mode.empty() && !is_quarantine;
    const bool has_pressure = mode == "--pressure";
    const bool has_idle_error = mode == "--idle-error";
    const bool has_invalid_setting = mode == "--invalid-cache";
    const auto path = "/tmp/glimmer-pageable-" + std::to_string(::getpid());
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(config);
    expect(server.start(), "start authenticated pageable endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted CPU service identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select owned frontend before first use");
    Latches latches;
    expect((should_reuse ? ::unsetenv("GLIMMER_EXECUTION_STAGING_CACHE")
                         : ::setenv("GLIMMER_EXECUTION_STAGING_CACHE", "0", 1)) == 0,
           "explicit legacy rollback or default-enabled reuse");
    if (has_invalid_setting)
        expect(::setenv("GLIMMER_EXECUTION_STAGING_CACHE", "2", 1) == 0,
               "invalid cache setting with otherwise valid authenticated configuration");
    auto client = std::async(std::launch::async, [&] {
        if (has_invalid_setting) {
            CUcontext context{};
            expect(cuCtxCreate(&context, 0, 0) == CUDA_ERROR_NOT_INITIALIZED && !context &&
                       glimmer_memory_native_calls() == 0,
                   "invalid cold cache configuration rejects without connection or native work");
        } else if (has_pressure)
            pressure_application(latches);
        else
            application(sdk, latches, is_quarantine, has_idle_error);
    });
    std::array<std::byte, 512> device{};
    ExecutionStagingFixture staging(device, 7);
    std::optional<ExecutionReceivedRequest> withheld;
    std::optional<ExecutionReceivedRequest> pending_release;
    ExecutionWireReply withheld_reply;
    ExecutionWireReply release_reply;
    unsigned int imports{}, downloads{}, releases{}, idle_queries{};
    unsigned int pressure_evictions{};
    const auto reply = [&](const ExecutionReceivedRequest& request,
                           const ExecutionWireReply& result) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {107, 109};
        auto payload = encode_execution_reply(result);
        expect(payload && server.reply(request.connection, {header, std::move(*payload)}),
               "exact original reply identity");
    };
    const auto deadline = Clock::now() + std::chrono::seconds(10);
    while (client.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        expect(server.poll_for(1) && Clock::now() < deadline, "bounded actual pageable IPC");
        if (withheld && latches.may_reply) {
            reply(*withheld, withheld_reply);
            withheld.reset();
        }
        if (pending_release && latches.upload_returned) {
            latches.release_observed.store(true);
            reply(*pending_release, release_reply);
            pending_release.reset();
        }
        auto request = server.take_request();
        if (!request) continue;
        ExecutionWireReply result;
        result.disposition = ExecutionWireDisposition::kComplete;
        result.original_status = 0;
        const auto operation = request->message.header.operation;
        if (operation == ExecutionOpcode::kImportHostMemory && ++imports == 1 && !has_pressure) {
            if (is_quarantine) {
                auto retained = staging.handle(*request);
                expect(retained.has_value(),
                       "fake native registration retains actual shared backing");
            }
            result.status = is_quarantine ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_OUT_OF_MEMORY;
            result.original_status = result.status;
            ExecutionPayloadWriter state;
            expect(state.write_u32(is_quarantine ? 1U : 0U),
                   "explicit native registration ownership state");
            result.data = state.take();
        } else if (auto handled = staging.handle(*request)) {
            result = std::move(*handled);
            if (has_pressure && operation == ExecutionOpcode::kReleaseHostMemory &&
                !latches.origin_closing)
                ++pressure_evictions;
            if (has_pressure)
                expect(imports - releases <= 256,
                       "cache includes idle entries in its record bound");
            if (operation == ExecutionOpcode::kQueryStagingIdle && ++idle_queries == 1) {
                if (has_idle_error) ++result.id;
                pending_release = std::move(*request);
                release_reply = std::move(result);
                continue;
            }
            if (operation == ExecutionOpcode::kReleaseHostMemory && ++releases == 1) {
                pending_release = std::move(*request);
                release_reply = std::move(result);
                continue;
            }
            if (operation == ExecutionOpcode::kStagedHostTransfer &&
                request->message.payload.back() == std::byte{0} &&
                request->message.payload[56] == std::byte{0} && ++downloads == 1 && !has_pressure) {
                result.status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
                result.original_status = result.status;
                withheld = std::move(*request);
                withheld_reply = std::move(result);
                latches.download_withheld.store(true);
                continue;
            }
        } else if (operation == ExecutionOpcode::kAllocate) {
            result.id = 7;
            result.address = 0x100000;
        } else if (operation == ExecutionOpcode::kGetContextFlags)
            result.id = 0;
        else if (operation == ExecutionOpcode::kCaptureFrontier) {
            result.id = 13;
            result.original_status.reset();
        } else if (operation == ExecutionOpcode::kQueryFrontier ||
                   operation == ExecutionOpcode::kReleaseFrontier)
            result.original_status.reset();
        else
            expect(operation == ExecutionOpcode::kHello || operation == ExecutionOpcode::kClose ||
                       operation == ExecutionOpcode::kRelease,
                   "only original covered APIs reach the endpoint");
        reply(*request, result);
    }
    client.get();
    if (has_idle_error || has_invalid_setting) {
        expect(!withheld && !pending_release && !downloads && !releases &&
                   (has_idle_error ? !staging.empty() && imports == 2 && idle_queries == 1
                                   : staging.empty() && !imports && !idle_queries),
               "invalid configuration cannot issue; uncertain idle receipt cannot unregister");
        server.stop();
        return;
    }
    expect(!withheld && !pending_release &&
               (is_quarantine ? !staging.empty() && imports == 1 && releases == 0 && downloads == 0
                              : staging.empty() && releases + (has_pressure ? 0 : 1) == imports &&
                                    downloads == (has_pressure ? 301 : 35)),
           "all successful staging imports retire once; failed work is never replayed");
    server.stop();
    std::cout << "Staging imports=" << imports << " releases=" << releases
              << " idle_queries=" << idle_queries << '\n';
    expect(has_pressure   ? imports >= 301 && pressure_evictions > 0
           : should_reuse ? imports < 32
                          : is_quarantine || imports == 69,
           "default staging reuse avoids repeated registration after complete normal cleanup");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc == 2 || (argc == 3 && (std::string_view(argv[2]) == "--quarantine" ||
                                           std::string_view(argv[2]) == "--reuse" ||
                                           std::string_view(argv[2]) == "--pressure" ||
                                           std::string_view(argv[2]) == "--idle-error" ||
                                           std::string_view(argv[2]) == "--invalid-cache")),
               "fake SDK fixture path required");
        run(argv[1], argc == 3 ? std::string_view(argv[2]) : std::string_view{});
        std::cout << "Public pageable staging lifetime and asynchronous cleanup checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
