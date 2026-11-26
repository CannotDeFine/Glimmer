#include "internal/driver_api_interceptor.h"
#include "internal/driver_cooperative_router.h"
#include "internal/runtime_api_bridge.h"
#include "cooperative_ptx_fixture.h"

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace {
using namespace glimmer::interceptor;
constexpr unsigned int kBlocks = 84, kThreads = 80;
constexpr std::size_t kElements = static_cast<std::size_t>(kBlocks) * kThreads;
using Release = CUresult(CUDAAPI*)(CUdevice);
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
void check(CUresult result, const char* message) {
    if (result != CUDA_SUCCESS) {
        throw std::runtime_error(std::string(message) + ": " + std::to_string(result));
    }
}
template <typename Operation>
CUresult application(bool native, Operation&& operation) {
    std::optional<NativeDriverCallScope> scope;
    if (native) {
        scope.emplace();
    }
    return operation();
}
void inactive() {
    unsigned int flags{};
    int active{};
    const NativeDriverCallScope native;
    check(cuDevicePrimaryCtxGetState(0, &flags, &active), "observe non-activating primary state");
    require(active == 0, "last original release leaves primary inactive");
}
void runtime_tls() {
    const NativeDriverCallScope native;
    require(cudaPeekAtLastError() == cudaSuccess && cudaGetLastError() == cudaSuccess &&
                cudaPeekAtLastError() == cudaSuccess,
            "Driver lifecycle and private work preserve actual caller Runtime error TLS");
}
void qualify(bool native, bool modern, bool cached, unsigned int caller) {
    check(cuInit(0), "publish original process dispatch without Runtime primary activation");
    DriverDispatch driver;
    require(driver.initialize(), "resolve independent original Driver table");
    const int version = modern ? 11000 : 7000;
    void* native_address{};
    check(driver.get_proc_address("cuDevicePrimaryCtxRelease", &native_address, version, 0),
          "select exact original ownership version");
    void* controlled =
        dlsym(RTLD_DEFAULT, modern ? "cuDevicePrimaryCtxRelease_v2" : "cuDevicePrimaryCtxRelease");
    require(controlled && controlled != native_address,
            "public dynamic release is controlled and never aliases original table");
    if (cached) {
        inactive();
        void* address{};
        cudaDriverEntryPointQueryResult query{};
        {
            // Actual SDK cold lookup caches this address while trusted native
            // initialization is lexical. Later application calls are ordinary.
            const NativeDriverCallScope scope;
            const auto result = cudaGetDriverEntryPointByVersion("cuDevicePrimaryCtxRelease",
                                                                 &address, version, 0, &query);
            require(result == cudaSuccess && query == cudaDriverEntryPointSuccess &&
                        address == controlled && is_inside_driver_call() &&
                        !is_inside_runtime_call(),
                    "actual cold SDK cache keeps exact controlled release ABI");
        }
        require(!is_inside_driver_call() && !is_inside_runtime_call(),
                "SDK cache lookup restores both lexical guards");
        inactive();
        controlled = address;
    }
    const auto release = reinterpret_cast<Release>(native ? native_address : controlled);
    CUcontext foreign{};
    if (caller == 2) {
        const NativeDriverCallScope scope;
        check(cuCtxCreate(&foreign, 0, 0), "create independently owned foreign caller");
        check(cuCtxSetCurrent(nullptr), "unbind independent caller during primary setup");
    }
    // Caller-owned, page-aligned storage survives native context destruction.
    // Each subsequent original registration proves the old registration retired.
    alignas(4096) std::array<std::uint32_t, 1024> markers{};
    alignas(4096) std::array<std::uint32_t, 7168> readback{};
    DriverCooperativeRouter* router{};
    for (unsigned int generation = 0; generation < 3; ++generation) {
        CUcontext context{};
        CUstream stream{}, high_stream{};
        CUevent event{};
        CUdeviceptr output{}, mapped{};
        {
            const NativeDriverCallScope scope;
            check(cuDevicePrimaryCtxRetain(&context, 0),
                  "retain sole application primary reference");
            check(cuCtxSetCurrent(context), "bind independently retained primary generation");
            check(cuMemHostRegister(markers.data(), sizeof(markers), CU_MEMHOSTREGISTER_DEVICEMAP),
                  "register original caller-owned marker pages exactly once per generation");
            check(cuMemHostRegister(readback.data(), sizeof(readback), 0),
                  "register caller-owned readback pages without stale previous registration");
            check(cuMemHostGetDevicePointer(&mapped, markers.data(), 0),
                  "map actual marker registration");
            check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                  "create original application stream");
            check(cuStreamCreateWithPriority(&high_stream, CU_STREAM_NON_BLOCKING, -1),
                  "create original high-priority stream");
        }
        if (!native) {
            std::size_t available{}, total{};
            check(cuMemGetInfo(&available, &total), "inspect fresh generation quota before import");
            require(available == 8388608 && total == available,
                    "last release recovered all exact preceding context charges");
        }
        check(application(native,
                          [&] { return cuMemAlloc(&output, kElements * sizeof(std::uint32_t)); }),
              "allocate original application output through unchanged API");
        {
            const NativeDriverCallScope scope;
            check(cuMemsetD32(output, 0, kElements), "initialize cumulative exactly-once output");
        }
        CUmodule module{};
        CUfunction low{}, high{};
        check(application(native, [&] { return cuModuleLoadData(&module, kCooperativePtx); }),
              "load application PTX without explicit framework preparation");
        check(application(native, [&] { return cuModuleGetFunction(&low, module, "cooperative"); }),
              "look up original cooperative function through unchanged API");
        check(application(native,
                          [&] { return cuModuleGetFunction(&high, module, "priority_probe"); }),
              "look up original priority function");
        check(application(native, [&] { return cuEventCreate(&event, CU_EVENT_DISABLE_TIMING); }),
              "create original event through public API");
        std::shared_ptr<CudaCooperativeFrontend> owner;
        CudaEventProjection registration;
        if (!native) {
            {
                const ProcessCallGate::Call observation(cooperative_process_call_gate());
                router = cooperative_source_router();
            }
            require(router, "ordinary resource calls configure the public process owner");
            check(router->existing_frontend(context, owner),
                  "observe already-created public owner");
            require(owner != nullptr, "public setup imports the exact existing context owner");
            check(router->event_projection(event, registration),
                  "snapshot original event registration");
            if (generation == 0) {
                auto& process_gate = cooperative_process_call_gate();
                const ProcessCallGate::Call observation(process_gate);
                void* begin_address{};
                void* end_address{};
                check(driver.get_proc_address("cuStreamBeginCapture", &begin_address, 12600,
                                              CU_GET_PROC_ADDRESS_LEGACY_STREAM),
                      "resolve original source fixture begin");
                check(driver.get_proc_address("cuStreamEndCapture", &end_address, 12600,
                                              CU_GET_PROC_ADDRESS_LEGACY_STREAM),
                      "resolve original source fixture end");
                const auto begin = reinterpret_cast<decltype(&cuStreamBeginCapture)>(begin_address);
                const auto end = reinterpret_cast<decltype(&cuStreamEndCapture)>(end_address);
                require(begin && end, "actual source fixture callbacks are available");
                // The existing private source assembly is only a phase guard
                // fixture, not activation of public Graph/capture support.
                check(router->source_capture_transition(
                          process_gate, context, stream, false, CudaSourceCaptureKind::kBegin,
                          [&] {
                              const NativeDriverCallScope scope;
                              return begin(stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
                          }),
                      "assemble actual retained source before public release rejection");
                require(release(0) == CUDA_ERROR_NOT_SUPPORTED,
                        "public and SDK-cached release reject retained source before native issue");
                CUgraph captured{};
                check(router->source_capture_transition(process_gate, context, stream, false,
                                                        CudaSourceCaptureKind::kEnd,
                                                        [&] {
                                                            const NativeDriverCallScope scope;
                                                            return end(stream, &captured);
                                                        }),
                      "end unchanged original source fixture after rejected release");
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
                while (process_gate.is_native_only_on_current_thread()) {
                    const auto result = router->try_finish_source_capture(process_gate);
                    require(result == CUDA_SUCCESS || result == CUDA_ERROR_NOT_READY,
                            "known ended source can reconcile without release quarantine");
                    require(std::chrono::steady_clock::now() < deadline,
                            "source reconciliation completes before ordinary lifecycle test");
                    std::this_thread::yield();
                }
                check(cuGraphDestroy(captured), "destroy original ended capture graph while live");
            }
        }
        CUgraph graph{};
        CUgraphExec executable{};
        check(application(native, [&] { return cuGraphCreate(&graph, 0); }),
              "create original empty graph");
        check(
            application(native, [&] { return cuGraphInstantiateWithFlags(&executable, graph, 0); }),
            "instantiate original native graph");
        CudaNativeGraphExecutable graph_registration;
        if (!native) {
            check(router->graphs().lookup(executable, graph_registration),
                  "snapshot exact graph metadata");
        }
        bool pending_pause{};
        for (unsigned int cycle = 0; cycle < 4; ++cycle) {
            if (cycle == 3) {
                // Finish graph lifetime before accepting the last pending work:
                // a graph-destroy boundary must not provide the release fence.
                check(application(native, [&] { return cuGraphExecDestroy(executable); }),
                      "destroy original graph executable while target still lives");
                check(application(native, [&] { return cuGraphDestroy(graph); }),
                      "destroy original source graph while target still lives");
            }
            markers[0] = markers[1] = 0;
            unsigned int work = cycle == 1 || cycle == 3 ? 150000 : 1000;
            std::array<void*, 3> arguments{&output, &mapped, &work};
            check(application(native,
                              [&] {
                                  return cuLaunchKernel(low, kBlocks, 1, 1, kThreads, 1, 1, 0,
                                                        stream, arguments.data(), nullptr);
                              }),
                  "public original launch automatically reaches framework transformation");
            if (cycle == 1 && !native) {
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
                while (std::atomic_ref<std::uint32_t>(markers[0]).load(std::memory_order_acquire) ==
                       0) {
                    require(std::chrono::steady_clock::now() < deadline,
                            "actual low kernel starts after nonlast release");
                    std::this_thread::sleep_for(std::chrono::milliseconds{1});
                }
                CUdeviceptr high_marker = mapped + sizeof(std::uint32_t);
                std::array<void*, 1> high_arguments{&high_marker};
                check(owner->native(
                          high_stream, false,
                          [&] {
                              const NativeDriverCallScope scope;
                              std::array<std::uint32_t, kElements> sampled{};
                              check(cuMemcpyDtoH(sampled.data(), output, sizeof(sampled)),
                                    "sample physically suspended low work");
                              pending_pause =
                                  std::any_of(sampled.begin(), sampled.end(),
                                              [&](auto value) { return value == cycle; });
                              return driver.launch_kernel(high, 1, 1, 1, 1, 1, 1, 0, high_stream,
                                                          high_arguments.data(), nullptr);
                          },
                          true),
                      "insert admitted high work after original nonlast release");
            }
            check(application(native,
                              [&] {
                                  return cuMemcpyDtoHAsync(readback.data(), output,
                                                           kElements * sizeof(std::uint32_t),
                                                           stream);
                              }),
                  "accept original asynchronous readback before release");
            check(application(native, [&] { return cuEventRecord(event, stream); }),
                  "accept exact event record before release");
            if (native) {
                const NativeDriverCallScope scope;
                // Native nonlast release does not promise an application fence.
                check(cuEventSynchronize(event), "complete native comparison outputs explicitly");
            }
            CUcontext expected = caller == 0 ? context : caller == 1 ? nullptr : foreign;
            std::size_t before{}, total{};
            if (!native) {
                check(cuMemGetInfo(&before, &total), "snapshot exact live context charge");
            }
            {
                const NativeDriverCallScope scope;
                if (cycle != 3) {
                    CUcontext extra{};
                    check(cuDevicePrimaryCtxRetain(&extra, 0),
                          "retain independent extra application reference");
                    require(extra == context, "extra retain belongs to original primary identity");
                }
                check(cuCtxSetCurrent(expected), "select exact bound/null/foreign release caller");
            }
            runtime_tls();
            if (cycle == 3 && !native) {
                require(owner->stats().acknowledged_completions < 5,
                        "last public release starts with actually pending accepted low work");
            }
            check(release(0), "ordinary public or SDK-cached release owns complete lifecycle");
            CUcontext after{};
            check(driver.context_get_current(&after),
                  "observe exact SDK post-release caller without rebinding");
            require(after == expected && !is_inside_driver_call() && !is_inside_runtime_call(),
                    "release preserves actual caller context and lexical guards");
            runtime_tls();
            require(std::all_of(readback.begin(), readback.begin() + kElements,
                                [&](auto value) { return value == cycle + 1; }),
                    "release completed every accepted output exactly once without explicit "
                    "framework checkpoint");
            if (cycle == 1 && !native) {
                require(pending_pause && markers[1] == 2,
                        "real priority handoff survives progress-pool turnover");
            }
            if (cycle != 3) {
                unsigned int flags{};
                int active{};
                {
                    const NativeDriverCallScope scope;
                    check(cuDevicePrimaryCtxGetState(0, &flags, &active),
                          "prove nonlast native lifetime survives");
                    check(cuCtxSetCurrent(context),
                          "restore application binding after observing SDK caller");
                    std::size_t nodes{};
                    check(cuGraphGetNodes(graph, nullptr, &nodes),
                          "original source graph survives nonlast release");
                    require(nodes == 0 && active == 1,
                            "nonlast release preserves actual graph and primary lifetime");
                }
                if (!native) {
                    std::shared_ptr<CudaCooperativeFrontend> same;
                    CudaEventProjection current;
                    check(router->existing_frontend(context, same),
                          "observe retained context actor CPU-only");
                    check(router->refresh_event_projection(registration, current),
                          "observe retained exact event registration");
                    require(same == owner && current.context() == context && current.is_logical(),
                            "nonlast release preserves exact actor and event identity");
                    check(router->graphs().is_current(graph_registration),
                          "nonlast release preserves graph generation");
                    std::size_t available{};
                    check(cuMemGetInfo(&available, &total),
                          "observe nonlast exact quota preservation");
                    require(available == before,
                            "nonlast release cannot release live application charges");
                }
                check(
                    application(native,
                                [&] { return cuStreamWaitEvent(CU_STREAM_PER_THREAD, event, 0); }),
                    "reuse old event dependency through same PTDS owner");
                check(
                    application(native, [&] { return cuStreamSynchronize(CU_STREAM_PER_THREAD); }),
                    "complete preserved original PTDS dependency");
            } else {
                inactive();
                if (!native) {
                    std::shared_ptr<CudaCooperativeFrontend> retired;
                    CudaEventProjection old;
                    check(router->existing_frontend(context, retired),
                          "inspect last CPU actor retirement without CUDA");
                    require(!retired &&
                                router->event_projection(event, old) == CUDA_ERROR_INVALID_HANDLE,
                            "last release removes old actor and exact event CPU registration");
                    const auto stats = owner->stats();
                    require(stats.granted_executions == 6 && stats.retreat_requests == 1 &&
                                stats.acknowledged_suspensions == 1 && stats.logical_resumes == 1 &&
                                stats.acknowledged_completions == 5 &&
                                stats.acknowledged_failures == 0,
                            "last CPU retirement preserves exact completed control evidence");
                }
                const NativeDriverCallScope scope;
                check(cuCtxSetCurrent(nullptr),
                      "clear original unchanged destroyed caller binding");
            }
        }
        owner.reset();
    }
    if (foreign) {
        const NativeDriverCallScope scope;
        check(cuCtxDestroy(foreign), "destroy still-owned independent foreign context");
    }
    inactive();
    std::cout << "public_primary_release native=" << native << " version=" << version
              << " cached=" << cached << " caller=" << caller
              << " generations=3 nonlast=9 last=3 outputs=6720 quota_checked=" << !native
              << " priority_checked=" << !native << " application_checkpoint=0\n";
}
void child(const char* executable, const std::string& flag) {
    const auto pid = fork();
    require(pid >= 0, "fork isolated exact lifecycle case");
    if (pid == 0) {
        execl(executable, executable, flag.c_str(), nullptr);
        _exit(127);
    }
    int status{};
    require(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
            "isolated original/public/cached lifecycle qualification succeeds");
}
}  // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]).starts_with("--case-")) {
            const std::string_view flag(argv[1]);
            require(flag.size() == 13, "select complete native/version/cache/caller case");
            qualify(flag[7] == '1', flag[9] == '1', flag[11] == '1', flag[12] - '0');
        } else {
            require(argc == 1, "use isolated lifecycle matrix");
            for (const bool native : {true, false}) {
                for (const bool modern : {false, true}) {
                    for (const bool cached : {false, true}) {
                        for (unsigned int caller = 0; caller < 3; ++caller) {
                            const auto flag = std::string("--case-") + (native ? "1" : "0") + "-" +
                                              (modern ? "1" : "0") + "-" + (cached ? "1" : "0") +
                                              std::to_string(caller);
                            child(argv[0], flag);
                        }
                    }
                }
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "public primary release qualification: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
