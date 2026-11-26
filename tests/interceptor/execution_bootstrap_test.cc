#include "runtime.h"
#include "internal/process_identity.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_execution_lookup_count(unsigned int);
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using Clock = std::chrono::steady_clock;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult status) {
    expect(status == CUDA_SUCCESS, "public bootstrap operation succeeds");
}
struct HostDelete {
    void operator()(std::byte* value) const noexcept {
        ::operator delete(value, std::align_val_t{64});
    }
};
struct Native {
    std::string_view mode;
    CUcontext current{};
    std::vector<std::unique_ptr<int>> contexts;
    std::unique_ptr<std::byte, HostDelete> host;
    std::vector<std::byte> counters;
    unsigned int creates{}, destroys{}, allocations{}, host_allocations{}, unexpected{}, flags{};
} native;
bool clean_mode(std::string_view mode) {
    return mode == "clean" || mode == "partial-clean" || mode == "private-clean";
}
bool private_mode(std::string_view mode) {
    return mode == "private-clean" || mode == "unknown-host";
}
CudaSessionDispatch dispatch() {
    CudaSessionDispatch value;
    value.initialize = [](unsigned int) { return CUDA_SUCCESS; };
    value.device_get = [](CUdevice* device, int ordinal) {
        *device = ordinal;
        return CUDA_SUCCESS;
    };
    value.execution.context_get_current = [](CUcontext* result) {
        if ((native.mode == "current-query-error" || native.mode == "partial-query-error") &&
            native.creates == 1)
            return CUDA_ERROR_UNKNOWN;
        *result = native.current;
        return CUDA_SUCCESS;
    };
    value.context_create = [](CUcontext* result, unsigned int flags, CUdevice) {
        ++native.creates;
        native.flags = flags;
        if (native.creates == 1) {
            if (native.mode == "clean" || native.mode == "current-query-error")
                return CUDA_ERROR_OUT_OF_MEMORY;
            if (native.mode == "null-success") return CUDA_SUCCESS;
            if (native.mode == "unknown") return CUDA_ERROR_UNKNOWN;
        }
        native.contexts.push_back(std::make_unique<int>(0));
        native.current = reinterpret_cast<CUcontext>(native.contexts.back().get());
        if (native.creates == 1 && native.mode == "unreturned-current")
            return CUDA_ERROR_OUT_OF_MEMORY;
        *result = native.current;
        return native.creates == 1 && native.mode.starts_with("partial-") ? CUDA_ERROR_OUT_OF_MEMORY
                                                                          : CUDA_SUCCESS;
    };
    value.context_set_current = [](CUcontext current) {
        native.current = current;
        return CUDA_SUCCESS;
    };
    value.context_destroy = [](CUcontext current) {
        ++native.destroys;
        expect(native.current == current, "cleanup runs in the actual owned context");
        const auto found = std::ranges::find_if(native.contexts, [current](const auto& item) {
            return reinterpret_cast<CUcontext>(item.get()) == current;
        });
        expect(found != native.contexts.end(), "never destroy an unreturned or foreign context");
        if (native.mode == "partial-uncertain") return CUDA_ERROR_UNKNOWN;
        native.contexts.erase(found);
        native.current = nullptr;
        return CUDA_SUCCESS;
    };
    value.context_get_flags = [](unsigned int* flags) {
        *flags = native.flags;
        return CUDA_SUCCESS;
    };
    value.allocate = [](CUdeviceptr* pointer, std::size_t bytes) {
        ++native.allocations;
        if (native.mode == "private-clean" && native.allocations == 1)
            return CUDA_ERROR_OUT_OF_MEMORY;
        native.counters.resize(bytes);
        *pointer = 0x1000;
        return CUDA_SUCCESS;
    };
    value.release = [](CUdeviceptr pointer) {
        expect(pointer == 0x1000 && !native.counters.empty(),
               "release only the owned private counter");
        native.counters.clear();
        return CUDA_SUCCESS;
    };
    value.host_allocate = [](void** pointer, std::size_t bytes, unsigned int) {
        ++native.host_allocations;
        if (native.mode == "unknown-host") return CUDA_SUCCESS;  // Deliberate NULL-success.
        native.host.reset(static_cast<std::byte*>(::operator new(bytes, std::align_val_t{64})));
        *pointer = native.host.get();
        return CUDA_SUCCESS;
    };
    value.host_release = [](void* pointer) {
        expect(pointer == native.host.get(), "release only a known successful host allocation");
        native.host.reset();
        return CUDA_SUCCESS;
    };
    value.host_map = [](CUdeviceptr* pointer, void*, unsigned int) {
        *pointer = 0x8000;
        return CUDA_SUCCESS;
    };
    value.occupancy = [](int* active, CUfunction, int, std::size_t) {
        *active = 4;
        return CUDA_SUCCESS;
    };
    value.function_set_attribute = [](CUfunction, CUfunction_attribute, int) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.module_load = [](CUmodule*, const void*) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.module_unload = [](CUmodule) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.function_get = [](CUfunction*, CUmodule, const char*) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.parameter_info = [](CUfunction, std::size_t, std::size_t*, std::size_t*) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.upload = [](CUdeviceptr, const void*, std::size_t) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.download = [](void*, CUdeviceptr, std::size_t) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.upload_async = [](CUdeviceptr, const void*, std::size_t, CUstream) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.download_async = [](void*, CUdeviceptr, std::size_t, CUstream) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.stream_create = [](CUstream*, unsigned int) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.stream_destroy = [](CUstream) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.execution.launch_kernel = [](CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int, unsigned int,
                                       CUstream, void**, void**) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.execution.event_create = [](CUevent*, unsigned int) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.execution.event_record = [](CUevent, CUstream) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.execution.event_query = [](CUevent) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.execution.event_destroy = [](CUevent) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    value.execution.stream_synchronize = [](CUstream) {
        ++native.unexpected;
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    return value;
}

void application(std::string_view mode) {
    void* address{};
    CUdriverProcAddressQueryResult queried{};
    check(cuGetProcAddress("cuDevicePrimaryCtxRetain", &address, 12000, 0, &queried));
    expect(address && queried == CU_GET_PROC_ADDRESS_SUCCESS,
           "cache the actual controlled primary ABI");
    const auto retain = reinterpret_cast<decltype(&cuDevicePrimaryCtxRetain)>(address);
    check(cuDevicePrimaryCtxSetFlags(0, CU_CTX_SCHED_YIELD));
    CUcontext context = reinterpret_cast<CUcontext>(1);
    const auto first = retain(&context, 0);
    const bool unknown = mode == "null-success" || mode == "unknown" || mode == "unknown-host" ||
                         mode == "lost-reply";
    expect(first == (unknown ? CUDA_ERROR_UNKNOWN : CUDA_ERROR_OUT_OF_MEMORY) &&
               context == reinterpret_cast<CUcontext>(1),
           "failed bootstrap preserves the actual error and never publishes a handle");
    if (!clean_mode(mode)) {
        expect(retain(&context, 0) == CUDA_ERROR_UNKNOWN,
               "unproven/retained/lost/legacy failure cannot authorize another factory");
        expect(glimmer_execution_lookup_count(2) == 0,
               "uncertainty never escapes into caller-native resources");
        return;
    }
    unsigned int flags{};
    int active{};
    expect(cuDevicePrimaryCtxGetState(0, &flags, &active) == CUDA_SUCCESS,
           "proven-clean bootstrap failure remains inspectable before an explicit retry");
    expect(!active && flags == CU_CTX_SCHED_YIELD,
           "clean failure preserves flags and reference accounting");
    expect(retain(&context, 0) == CUDA_SUCCESS,
           "clean bootstrap failure permits a fresh explicit primary attempt");
    CUcontext again{}, current{};
    check(retain(&again, 0));
    check(cuCtxGetCurrent(&current));
    expect(again == context && !current,
           "successful fresh factory shares identity without pushing current");
    check(cuCtxSetCurrent(context));
    check(cuCtxGetFlags(&flags));
    expect(flags == (CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST), "new owner retains requested flags");
    check(cuCtxSetCurrent(nullptr));
    check(cuDevicePrimaryCtxRelease(0));
    check(cuDevicePrimaryCtxRelease(0));
    check(cuDevicePrimaryCtxGetState(0, &flags, &active));
    expect(!active && !flags && cuDevicePrimaryCtxRelease(0) == CUDA_ERROR_INVALID_CONTEXT,
           "fresh successful reference count retires exactly once");
    expect(glimmer_execution_lookup_count(2) == 0, "no caller-native creation or cleanup");
}
void configure(const std::string& path) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "trusted bootstrap owner identity");
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "select the actual owner endpoint before the first public call");
}
void actual_owner(const std::string& path, std::string_view mode) {
    native.mode = mode;
    ExecutionServiceConfig config;
    config.transport.socket_path = path;
    config.session_limits.submissions.max_commands = 1;
    config.persistent.workers = 320;
    const auto peer = current_process_identity();
    expect(peer.has_value(), "trusted actual client credentials");
    config.client_priorities.push_back({*peer, private_mode(mode) ? 0U : 100U});
    if (private_mode(mode)) {
        config.client_priorities.push_back({{UINT32_MAX, peer->uid, 1}, 100});
    }
    ExecutionService service(std::move(config), dispatch());
    expect(service.start(), "start the real authenticated native owner");
    configure(path);
    auto client = std::async(std::launch::async, [mode] { application(mode); });
    const auto deadline = Clock::now() + std::chrono::seconds{4};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(service.step(1), "progress the real owner and bootstrap transport");
        expect(Clock::now() < deadline, "bounded explicit retry/uncertain retention");
    }
    client.get();
    expect(native.unexpected == 0, "bootstrap recovery never submits application GPU work");
    if (clean_mode(mode)) {
        expect(service.request_stop() && native.creates == 2 &&
                   native.destroys == (mode == "clean" ? 1U : 2U) && native.contexts.empty() &&
                   !native.host && native.counters.empty(),
               "two explicit native attempts settle once, without private/resource leaks");
    } else {
        expect(native.creates == 1 &&
                   native.destroys ==
                       ((mode == "partial-uncertain" || mode == "partial-query-error") ? 1U : 0U),
               "uncertain cleanup/NULL output never repeats a target SDK call");
        std::cout << "Bootstrap uncertain ownership retained: case=" << mode << '\n' << std::flush;
        // Deliberate unknown ownership is not ordinary leak-free recovery.
        const auto directory = path.substr(0, path.find_last_of('/'));
        expect(::unlink(path.c_str()) == 0 && ::rmdir(directory.c_str()) == 0,
               "remove only this child's private socket fixture before intentional retention exit");
        _exit(EXIT_SUCCESS);
    }
}
void legacy_or_lost(const std::string& path, std::string_view mode) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start actual transport with an older/lost bootstrap reply");
    configure(path);
    auto client = std::async(std::launch::async, [mode] { application(mode); });
    unsigned int requests{};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(server.poll_for(1), "progress legacy/lost reply transport");
        if (const auto request = server.take_request()) {
            ++requests;
            expect(request->message.header.operation == ExecutionOpcode::kHello,
                   "no target replay");
            if (mode == "lost-reply") {
                server.close_peer(request->connection);
            } else {
                ExecutionWireReply reply;
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.status = CUDA_ERROR_OUT_OF_MEMORY;
                reply.original_status = reply.status;
                const auto payload = encode_execution_reply(reply);
                auto header = request->message.header;
                header.kind = ExecutionMessageKind::kReply;
                header.session = request->connection;
                header.incarnation = {43, 47};
                expect(payload && server.reply(request->connection, {header, *payload}),
                       "legacy status has no ownership proof");
            }
        }
    }
    client.get();
    expect(requests == 1, "legacy/lost reply cannot cause a second bootstrap request");
    server.stop();
}
int run(std::string_view mode) {
    char pattern[] = "/tmp/glimmer-bootstrap-XXXXXX";
    const auto directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    int status = EXIT_SUCCESS;
    std::jthread watchdog([](const std::stop_token& stop) {
        const auto deadline = Clock::now() + std::chrono::seconds{8};
        while (!stop.stop_requested()) {
            if (Clock::now() >= deadline) _exit(98);
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
        }
    });
    try {
        const std::string path = std::string(directory) + "/service.sock";
        if (mode == "legacy-failure" || mode == "lost-reply")
            legacy_or_lost(path, mode);
        else
            actual_owner(path, mode);
        std::cout << "Bootstrap ownership recovery verified: case=" << mode << "; replay=false\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        status = EXIT_FAILURE;
    }
    watchdog.request_stop();
    if (::rmdir(directory) != 0) status = EXIT_FAILURE;
    return status;
}
}  // namespace
int main(int argc, char** argv) {
    if (argc == 2) return run(argv[1]);
    for (const auto* mode :
         {"clean", "partial-clean", "private-clean", "partial-uncertain", "null-success", "unknown",
          "current-query-error", "partial-query-error", "unreturned-current", "unknown-host",
          "legacy-failure", "lost-reply"}) {
        const auto child = ::fork();
        if (child < 0) return EXIT_FAILURE;
        if (child == 0) {
            ::execl("/proc/self/exe", argv[0], mode, static_cast<char*>(nullptr));
            _exit(99);
        }
        int status{};
        if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            return EXIT_FAILURE;
    }
    std::cout << "Public bootstrap ownership suite passed: cases=12; target_replay=false\n";
    return EXIT_SUCCESS;
}
