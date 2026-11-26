#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string_view>

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult CUDAAPI cuGetProcAddress(const char*, void**, int, cuuint64_t);

#ifdef GLIMMER_FUNCTION_CONTROL_PROVIDER
namespace {
std::atomic<unsigned int> native_consumers{};
}
extern "C" unsigned int glimmer_function_control_native_calls() {
    return native_consumers.load();
}
extern "C" CUresult CUDAAPI cuInit(unsigned int) {
    return CUDA_SUCCESS;
}
extern "C" cudaError_t CUDARTAPI cudaRuntimeGetVersion(int* version) {
    if (!version) return cudaErrorInvalidValue;
    *version = CUDART_VERSION;
    return cudaSuccess;
}
extern "C" CUresult CUDAAPI cuFuncGetAttribute(int*, CUfunction_attribute, CUfunction) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuFuncSetAttribute(CUfunction, CUfunction_attribute, int) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuFuncSetCacheConfig(CUfunction, CUfunc_cache) {
    ++native_consumers;
    return CUDA_ERROR_UNKNOWN;
}
extern "C" CUresult CUDAAPI cuGetProcAddress_v2(const char* name, void** output, int,
                                                cuuint64_t flags,
                                                CUdriverProcAddressQueryResult* status) {
    if (!name || !output || flags) return CUDA_ERROR_INVALID_VALUE;
    *output = nullptr;
    if (std::string_view(name) == "cuFuncGetAttribute")
        *output = reinterpret_cast<void*>(&cuFuncGetAttribute);
    if (std::string_view(name) == "cuFuncSetAttribute")
        *output = reinterpret_cast<void*>(&cuFuncSetAttribute);
    if (std::string_view(name) == "cuFuncSetCacheConfig")
        *output = reinterpret_cast<void*>(&cuFuncSetCacheConfig);
    if (status)
        *status = *output ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return CUDA_SUCCESS;
}
extern "C" CUresult CUDAAPI cuGetProcAddress(const char* name, void** output, int version,
                                             cuuint64_t flags) {
    return cuGetProcAddress_v2(name, output, version, flags, nullptr);
}
#else
#include "glimmer/control/execution_socket.h"
#include "internal/process_identity.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

extern "C" unsigned int glimmer_function_control_native_calls();
namespace {
using namespace glimmer::control;
using namespace std::chrono_literals;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned public Driver operation succeeds");
}
bool owned(void* address) {
    Dl_info information{};
    return address && ::dladdr(address, &information) && information.dli_fname &&
           std::string_view(information.dli_fname).ends_with("libglimmer_execution_interceptor.so");
}
struct Functions {
    decltype(&cuFuncGetAttribute) get{};
    decltype(&cuFuncSetAttribute) set{};
    decltype(&cuFuncSetCacheConfig) cache{};
};
struct Control {
    std::atomic<unsigned int> calls{}, unloads{}, closes{};
    std::atomic<bool> hold{}, entered{}, release{};
    std::atomic<void*> protected_output{};
};
constexpr std::uint32_t kQueryArgument = 4;
constexpr std::uint32_t kSetterArgument = 8;
constexpr int kQueryValue = std::numeric_limits<int>::min();
constexpr unsigned int kLastCase = 26;

std::vector<Functions> routes(const char* provider_path) {
    void* provider = ::dlopen(provider_path, RTLD_NOW | RTLD_LOCAL);
    expect(provider, "load the independent Driver metadata provider");
    std::vector<Functions> result{
        {&cuFuncGetAttribute, &cuFuncSetAttribute, &cuFuncSetCacheConfig}};
    for (unsigned int route = 0; route < 5; ++route) {
        std::array<void*, 3> values{};
        constexpr std::array names{"cuFuncGetAttribute", "cuFuncSetAttribute",
                                   "cuFuncSetCacheConfig"};
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (route < 2) {
                values[index] = ::dlsym(route ? provider : RTLD_DEFAULT, names[index]);
            } else if (route == 2) {
                check(cuGetProcAddress(names[index], &values[index], 12060, 0));
            } else if (route == 3) {
                CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
                check(cuGetProcAddress_v2(names[index], &values[index], 12060, 0, &status));
                expect(status == CU_GET_PROC_ADDRESS_SUCCESS, "Driver cached query status");
            } else {
                cudaDriverEntryPointQueryResult status = cudaDriverEntryPointSymbolNotFound;
                expect(cudaGetDriverEntryPointByVersion(names[index], &values[index], 12060, 0,
                                                        &status) == cudaSuccess &&
                           status == cudaDriverEntryPointSuccess,
                       "Runtime cached query keeps exact Driver entry");
            }
            expect(owned(values[index]), "cached native resource consumer cannot escape");
        }
        result.push_back({reinterpret_cast<decltype(&cuFuncGetAttribute)>(values[0]),
                          reinterpret_cast<decltype(&cuFuncSetAttribute)>(values[1]),
                          reinterpret_cast<decltype(&cuFuncSetCacheConfig)>(values[2])});
    }
    expect(::dlclose(provider) == 0, "cached entries survive explicit provider close");
    return result;
}

void exercise(const char* provider, unsigned int scenario, Control& control) {
    const auto entries = routes(provider);
    CUcontext context{};
    CUmodule module{};
    CUfunction function{};
    check(cuCtxCreate(&context, 0, 0));
    check(cuModuleLoadData(&module, "independent function-control PTX transport fixture"));
    check(cuModuleGetFunction(&function, module, "kernel"));
    auto attribute = static_cast<CUfunction_attribute>(kQueryArgument);
    int output = 17;
    if (scenario == 26) {
        const auto page_size = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
        void* memory =
            ::mmap(nullptr, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(memory != MAP_FAILED, "allocate guarded output page");
        auto* guarded = static_cast<int*>(memory);
        *guarded = 17;
        control.protected_output = memory;
        expect(cuFuncGetAttribute(guarded, attribute, function) == CUDA_ERROR_INVALID_VALUE,
               "late unwritable output rejects publication after proven native success");
        expect(::mprotect(memory, page_size, PROT_READ | PROT_WRITE) == 0 && *guarded == 17,
               "failed local publication preserves the complete original value");
        expect(::munmap(memory, page_size) == 0, "release guarded output page");
        check(cuFuncGetAttribute(&output, attribute, function));
        expect(output == kQueryValue && control.calls == 2,
               "local publication failure does not quarantine a proven healthy owner");
        check(cuModuleUnload(module));
        check(cuCtxDestroy(context));
        return;
    }
    if (scenario) {
        const bool setter = scenario == 19 || scenario == 20;
        const auto result = setter ? cuFuncSetAttribute(function, attribute, -7)
                                   : cuFuncGetAttribute(&output, attribute, function);
        const auto expected = scenario == 1   ? CUDA_ERROR_INVALID_VALUE
                              : scenario == 2 ? CUDA_ERROR_NOT_SUPPORTED
                              : scenario == 3 || scenario == 22 || scenario == 23
                                  ? CUDA_ERROR_INVALID_CONTEXT
                              : scenario == 24 || scenario == 25 ? CUDA_SUCCESS
                                                                 : CUDA_ERROR_UNKNOWN;
        expect(result == expected, "exact requested status or fail-closed malformed receipt");
        expect(output == (scenario == 24   ? 0
                          : scenario == 25 ? -1
                                           : 17),
               "all failure paths preserve original host output");
        const auto before = control.calls.load();
        output = 19;
        const auto again = cuFuncGetAttribute(&output, attribute, function);
        if (scenario == 1 || scenario == 2 || scenario == 24 || scenario == 25) {
            expect(again == CUDA_SUCCESS && output == kQueryValue &&
                       control.calls.load() == before + 1,
                   "clean native/policy failure keeps owner reusable");
            check(cuModuleUnload(module));
            check(cuCtxDestroy(context));
        } else {
            expect(again != CUDA_SUCCESS && output == 19 && control.calls.load() == before,
                   "uncertain receipt quarantines before any further remote request");
        }
        return;
    }
    for (const auto& api : entries) {
        output = 17;
        check(api.get(&output, attribute, function));
        expect(output == kQueryValue, "full signed 32-bit getter value survives all routes");
        check(api.set(function, static_cast<CUfunction_attribute>(kSetterArgument),
                      std::numeric_limits<int>::min()));
        check(api.cache(function, CU_FUNC_CACHE_PREFER_EQUAL));
    }
    const auto before = control.calls.load();
    expect(cuFuncGetAttribute(nullptr, attribute, function) == CUDA_ERROR_INVALID_VALUE &&
               cuFuncGetAttribute(reinterpret_cast<int*>(1), attribute, function) ==
                   CUDA_ERROR_INVALID_VALUE &&
               cuFuncGetAttribute(&output, attribute, reinterpret_cast<CUfunction>(73)) ==
                   CUDA_ERROR_INVALID_HANDLE &&
               cuFuncSetAttribute(reinterpret_cast<CUfunction>(module), attribute, 1) ==
                   CUDA_ERROR_INVALID_HANDLE,
           "invalid host spans, raw server IDs and wrong kinds reject locally");
    CUcontext other{};
    check(cuCtxCreate(&other, 0, 0));
    expect(cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_NONE) == CUDA_ERROR_INVALID_HANDLE,
           "another current context cannot configure this function");
    check(cuCtxDestroy(other));
    check(cuCtxSetCurrent(context));
    expect(control.calls.load() == before, "invalid requests issue no function-control RPC");

    // Hold the actual query reply, then prove module unload cannot enter the
    // transport while the query owns its parent. Physical SDK deferral is a
    // separate backend regression, not inferred from this CPU scheduling check.
    control.hold = true;
    auto query = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        int value{};
        check(cuFuncGetAttribute(&value, attribute, function));
        expect(value == kQueryValue, "held query publishes only its original value");
    });
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while (!control.entered) {
        expect(std::chrono::steady_clock::now() < deadline, "query reaches the fake owner");
        std::this_thread::yield();
    }
    auto unload = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return cuModuleUnload(module);
    });
    expect(unload.wait_for(10ms) == std::future_status::timeout && control.unloads == 0,
           "module remains pinned until the query reply is validated");
    control.release = true;
    query.get();
    check(unload.get());
    expect(
        cuFuncGetAttribute(&output, attribute, function) == CUDA_ERROR_INVALID_HANDLE &&
            cuFuncSetAttribute(function, attribute, 1) == CUDA_ERROR_INVALID_HANDLE &&
            cuFuncSetCacheConfig(function, CU_FUNC_CACHE_PREFER_NONE) == CUDA_ERROR_INVALID_HANDLE,
        "destroyed module invalidates all three function entry routes");
    check(cuModuleLoadData(&module, "independent function-control PTX transport fixture"));
    check(cuModuleGetFunction(&function, module, "kernel"));
    control.entered = false;
    control.release = false;
    control.hold = true;
    auto setter = std::async(std::launch::async, [&] {
        check(cuCtxSetCurrent(context));
        return cuFuncSetAttribute(function, static_cast<CUfunction_attribute>(kSetterArgument),
                                  std::numeric_limits<int>::min());
    });
    const auto setter_deadline = std::chrono::steady_clock::now() + 5s;
    while (!control.entered) {
        expect(std::chrono::steady_clock::now() < setter_deadline, "setter reaches the fake owner");
        std::this_thread::yield();
    }
    auto close = std::async(std::launch::async, [&] { return cuCtxDestroy(context); });
    expect(close.wait_for(10ms) == std::future_status::timeout && control.closes == 1,
           "context close cannot cross an outstanding configuration mutation");
    control.release = true;
    check(setter.get());
    check(close.get());
}

ExecutionWireReply response(unsigned int scenario) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = 0;
    reply.id = std::bit_cast<std::uint32_t>(kQueryValue);
    switch (scenario) {
        case 1:
            reply.status = CUDA_ERROR_INVALID_VALUE;
            reply.original_status = reply.status;
            reply.id = 0;
            break;
        case 2:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            reply.original_status.reset();
            reply.id = 0;
            break;
        case 3:
        case 22:
        case 23:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.id = 0;
            reply.parameter_bytes = 1;
            if (scenario == 22) reply.original_status = CUDA_ERROR_INVALID_VALUE;
            if (scenario == 23) reply.original_status.reset();
            break;
        case 4:
            reply.parameter_bytes = 1;
            break;
        case 5:
            reply.id = std::uint64_t{1} << 32;
            break;
        case 6:
            reply.address = 1;
            break;
        case 7:
            reply.data = {std::byte{1}};
            break;
        case 8:
            reply.parameter_bytes = 1;
            reply.arguments = {{0, 1}};
            break;
        case 9:
            reply.original_status.reset();
            break;
        case 10:
            reply.original_status = CUDA_ERROR_INVALID_VALUE;
            break;
        case 11:
            reply.status = -1;
            reply.original_status = -1;
            break;
        case 12:
            reply.status = 1000;
            reply.original_status = 1000;
            break;
        case 13:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_CONTEXT;
            reply.id = 0;
            break;
        case 14:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.id = 0;
            break;
        case 15:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            reply.id = 1;
            break;
        case 16:
            reply.parameter_bytes = 2;
            break;
        case 17:
            reply.status = CUDA_ERROR_INVALID_VALUE;
            reply.original_status = reply.status;
            break;
        case 18:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            reply.original_status.reset();
            break;
        case 19:
            reply.id = 1;
            break;
        case 20:
            reply.id = 0;
            reply.parameter_bytes = 1;
            break;
        case 24:
            reply.id = 0;
            break;
        case 25:
            reply.id = UINT32_MAX;
            break;
        default:
            break;
    }
    return reply;
}

void run(const char* provider, unsigned int scenario, const std::string& path) {
    ExecutionSocketConfig config;
    config.socket_path = path;
    config.max_pending_requests = 8;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated fake function owner");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual endpoint process identity");
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", std::to_string(identity->pid).c_str(), 1) ==
                   0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START",
                        std::to_string(identity->start_time_ticks).c_str(), 1) == 0,
           "configure endpoint before any resource operation");
    Control control;
    auto client = std::async(std::launch::async, [&] { exercise(provider, scenario, control); });
    std::optional<ExecutionReceivedRequest> pending;
    ExecutionWireReply pending_reply;
    auto publish = [&](ExecutionReceivedRequest& request, ExecutionWireReply reply) {
        auto header = request.message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = request.connection;
        header.incarnation = {31, 37};
        const auto bytes = encode_execution_reply(reply);
        expect(bytes.has_value() && decode_execution_reply(*bytes).has_value(),
               "even hostile envelopes reach the new route as valid wire packets");
        expect(server.reply(request.connection, {header, *bytes}), "publish exact fake receipt");
    };
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    try {
        while (client.wait_for(0ms) != std::future_status::ready) {
            expect(server.poll_for(1), "progress owned frontend transport");
            if (pending && control.release) {
                publish(*pending, std::move(pending_reply));
                pending.reset();
            }
            if (auto request = server.take_request()) {
                ExecutionPayloadReader input(request->message.payload);
                ExecutionWireReply reply;
                reply.disposition = ExecutionWireDisposition::kComplete;
                reply.original_status = 0;
                switch (request->message.header.operation) {
                    case ExecutionOpcode::kHello:
                        break;
                    case ExecutionOpcode::kLoadPtx:
                        reply.id = 41;
                        break;
                    case ExecutionOpcode::kGetFunction: {
                        std::uint64_t module{};
                        std::string_view name;
                        expect(input.read_u64(module) && module == 41 && input.read_string(name) &&
                                   name == "kernel" && input.empty(),
                               "public function derives from its exact owned module");
                        reply.id = 73;
                        break;
                    }
                    case ExecutionOpcode::kFunctionControl: {
                        std::uint32_t operation{}, argument{}, value{};
                        std::uint64_t function{};
                        expect(request->message.payload.size() == 20 && input.read_u32(operation) &&
                                   input.read_u64(function) && input.read_u32(argument) &&
                                   input.read_u32(value) && input.empty() && function == 73 &&
                                   operation >= 1 && operation <= 3,
                               "canonical20-byte operation/function/argument/value payload");
                        if (!scenario) {
                            expect((operation == 1 && argument == kQueryArgument && !value) ||
                                       (operation == 2 && argument == kSetterArgument &&
                                        value == std::bit_cast<std::uint32_t>(
                                                     std::numeric_limits<int>::min())) ||
                                       (operation == 3 && argument == 3 && !value),
                                   "all routes preserve signed setters and canonical zero fields");
                        } else {
                            expect(argument == kQueryArgument &&
                                       value ==
                                           (operation == 2 ? std::bit_cast<std::uint32_t>(-7) : 0),
                                   "error controls preserve requested scalar bytes");
                        }
                        const auto number = ++control.calls;
                        if (scenario == 21 && number == 1) {
                            server.close_peer(request->connection);
                            continue;
                        }
                        reply = response(number == 1 ? scenario : 0);
                        if (!scenario && operation != 1) reply.id = 0;
                        if (scenario == 26 && number == 1) {
                            const auto page_size =
                                static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
                            expect(::mprotect(control.protected_output.load(), page_size,
                                              PROT_NONE) == 0,
                                   "output becomes inaccessible only after the request arrived");
                        }
                        if (control.hold) {
                            control.hold = false;
                            control.entered = true;
                            pending_reply = std::move(reply);
                            pending = std::move(request);
                            continue;
                        }
                        break;
                    }
                    case ExecutionOpcode::kUnloadModule: {
                        std::uint64_t module{};
                        expect(input.read_u64(module) && module == 41 && input.empty(),
                               "cleanup retains exact module identity");
                        ++control.unloads;
                        break;
                    }
                    case ExecutionOpcode::kClose:
                        ++control.closes;
                        break;
                    default:
                        throw std::runtime_error("unexpected or duplicate frontend operation");
                }
                publish(*request, std::move(reply));
            }
            expect(std::chrono::steady_clock::now() < deadline, "bounded function fixture");
        }
        client.get();
    } catch (...) {
        control.release = true;
        server.stop();
        throw;
    }
    server.stop();
    expect(glimmer_function_control_native_calls() == 0,
           "no native caller consumer ever receives an owned function token");
    if (!scenario)
        expect(control.calls == 20 && control.unloads == 1 && control.closes == 2,
               "six lookup routes and pinned controls issue exactly once with full cleanup");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        expect(argc == 2 || argc == 4, "expected provider path and optional isolated case");
        if (argc == 2) {
            for (unsigned int scenario = 0; scenario <= kLastCase; ++scenario) {
                const auto text = std::to_string(scenario);
                const auto child = ::fork();
                expect(child >= 0, "create isolated receipt control");
                if (!child) {
                    ::execl(argv[0], argv[0], argv[1], "--case", text.c_str(), nullptr);
                    _exit(127);
                }
                int status{};
                expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                           WEXITSTATUS(status) == 0,
                       "isolated function-control case failed");
            }
            return EXIT_SUCCESS;
        }
        expect(std::string_view(argv[2]) == "--case", "explicit case selector");
        const auto scenario = static_cast<unsigned int>(std::stoul(argv[3]));
        expect(scenario <= kLastCase, "bounded receipt case number");
        char pattern[] = "/tmp/glimmer-function-control-XXXXXX";
        const auto* directory = ::mkdtemp(pattern);
        expect(directory, "create private socket directory");
        try {
            run(argv[1], scenario, std::string(directory) + "/service.sock");
        } catch (...) {
            (void)::rmdir(directory);
            throw;
        }
        expect(::rmdir(directory) == 0, "remove completed private socket directory");
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
#endif
