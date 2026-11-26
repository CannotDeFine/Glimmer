#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "glimmer/control/execution_socket.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "internal/process_identity.h"

#include <cudnn.h>

#include <chrono>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <dlfcn.h>
#include <link.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace glimmer::control;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
class Library final {
   public:
    explicit Library(const char* path, bool is_new_namespace = false)
        : handle_(is_new_namespace ? ::dlmopen(LM_ID_NEWLM, path, RTLD_NOW | RTLD_LOCAL)
                                   : ::dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        expect(handle_ != nullptr, "load declared CPU provider");
    }
    ~Library() {
        if (::dlclose(handle_) != 0) std::terminate();
    }
    Library(const Library&) = delete;
    Library& operator=(const Library&) = delete;
    template <typename Function>
    Function symbol(const char* name) const {
        auto* raw = ::dlsym(handle_, name);
        expect(raw != nullptr, "resolve explicit provider symbol");
        Function result{};
        static_assert(sizeof(result) == sizeof(raw));
        std::memcpy(&result, &raw, sizeof(result));
        return result;
    }

   private:
    void* handle_{};
};
template <typename Pointer>
Pointer token(std::uintptr_t bits) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): inaccessible output sentinel.
    return reinterpret_cast<Pointer>(bits);
}
void check_rejection(cudnnStatus_t result) {
    expect(static_cast<int>(result) == 1, "profile mismatch returns the shared safe policy value");
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
void application(const char* approved_path, const char* wrong_path, bool is_new_namespace) {
    Library approved(approved_path);
    using Count = unsigned int (*)();
    const auto approved_calls = approved.symbol<Count>("glimmer_dnn_profile_native_calls");
    expect(cudnnSetTensor(nullptr, nullptr, nullptr, nullptr) == CUDNN_STATUS_NOT_SUPPORTED,
           "exact selected provider is accepted by the direct guard before any RPC");
    using Create = cudnnStatus_t(CUDNNWINAPI*)(cudnnHandle_t*);
    using Destroy = cudnnStatus_t(CUDNNWINAPI*)(cudnnHandle_t);
    using Tensor = cudnnStatus_t(CUDNNWINAPI*)(cudnnTensorDescriptor_t*);
    using Guard =
        cudnnStatus_t(CUDNNWINAPI*)(cudnnHandle_t, cudnnTensorDescriptor_t, void*, const void*);
    using Version = std::size_t(CUDNNWINAPI*)();
    const auto approved_create = approved.symbol<Create>("cudnnCreate");
    expect(approved_create == &cudnnCreate, "approved cached lookup retains the owned wrapper");
    Library rejected(is_new_namespace ? approved_path : wrong_path, is_new_namespace);
    if (!is_new_namespace) check_rejection(cudnnSetTensor(nullptr, nullptr, nullptr, nullptr));
    const auto rejected_calls = rejected.symbol<Count>("glimmer_dnn_profile_native_calls");
    const auto create = rejected.symbol<Create>("cudnnCreate");
    const auto destroy = rejected.symbol<Destroy>("cudnnDestroy");
    const auto tensor = rejected.symbol<Tensor>("cudnnCreateTensorDescriptor");
    const auto guard = rejected.symbol<Guard>("cudnnSetTensor");
    const auto version = rejected.symbol<Version>("cudnnGetVersion");
    expect(create == &cudnnCreate && destroy == &cudnnDestroy &&
               tensor == &cudnnCreateTensorDescriptor && guard == &cudnnSetTensor &&
               version == &cudnnGetVersion,
           "unproved explicit provider never escapes to native SDK functions");
    auto handle = token<cudnnHandle_t>(0x1234);
    auto descriptor = token<cudnnTensorDescriptor_t>(0x5678);
    check_rejection(cudnnCreate(&handle));
    check_rejection(create(&handle));
    check_rejection(approved_create(&handle));
    check_rejection(cudnnCreateTensorDescriptor(&descriptor));
    check_rejection(tensor(&descriptor));
    expect(handle == token<cudnnHandle_t>(0x1234) &&
               descriptor == token<cudnnTensorDescriptor_t>(0x5678),
           "rejected direct and cached constructors leave all output bytes untouched");
    check_rejection(cudnnDestroy(handle));
    check_rejection(destroy(handle));
    check_rejection(cudnnSetTensor(handle, descriptor, nullptr, nullptr));
    check_rejection(guard(handle, descriptor, nullptr, nullptr));
    expect(cudnnGetVersion() == 0 && version() == 0,
           "size-returning observations fail explicitly without native fallback");
    expect(approved_calls() == 0 && rejected_calls() == 0,
           "neither provider receives a single native cuDNN call");
}
#pragma GCC diagnostic pop

void run(const char* approved_path, const char* wrong_path, bool is_new_namespace) {
    const auto path = "/tmp/glimmer-dnn-profile-" + std::to_string(::getpid()) + ".sock";
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start no-RPC observer endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "read exact authenticated endpoint identity");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure a real endpoint before frontend initialization");
    auto client = std::async(std::launch::async,
                             [=] { application(approved_path, wrong_path, is_new_namespace); });
    struct Stop {
        ExecutionSocketServer& server;
        ~Stop() {
            server.stop();
        }
    } stop{server};
    bool has_request{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (client.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready &&
           std::chrono::steady_clock::now() < deadline) {
        expect(server.poll_for(1), "poll observer endpoint");
        if (auto request = server.take_request()) {
            has_request = true;
            server.close_peer(request->connection);
        }
    }
    expect(client.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready,
           "profile rejection is bounded without a service response");
    client.get();
    expect(server.poll_for(0) && !server.take_request() && !has_request,
           "profile rejection emitted no bind, metadata or SDK RPC");
}

constexpr unsigned int kBindingReceiptCases = 8;
ExecutionWireReply malformed_binding(unsigned int scenario) {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    ExecutionPayloadWriter output;
    const auto version = scenario == 0 ? (CUDNN_VERSION == 8907 ? 91002U : 8907U)
                                       : static_cast<std::uint32_t>(CUDNN_VERSION);
    expect(write_execution_dnn_profile(output, version), "encode hostile binding echo");
    reply.data = output.take();
    switch (scenario) {
        case 0:
            break;
        case 1:
            reply.data.push_back(std::byte{1});
            break;
        case 2:
            reply.original_status = 0;
            break;
        case 3:
            reply.id = 1;
            break;
        case 4:
            reply.address = 1;
            break;
        case 5:
            reply.parameter_bytes = 1;
            break;
        case 6:
            reply.parameter_bytes = 4;
            reply.arguments = {{0, 4}};
            break;
        case 7:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = 1;  // Driver INVALID_VALUE is not canonical profile mismatch.
            reply.data.clear();
            break;
        default:
            throw std::runtime_error("unknown binding receipt scenario");
    }
    return reply;
}
void binding_application(const char* approved_path) {
    Library approved(approved_path);
    using Count = unsigned int (*)();
    using Tensor = cudnnStatus_t(CUDNNWINAPI*)(cudnnTensorDescriptor_t*);
    const auto calls = approved.symbol<Count>("glimmer_dnn_profile_native_calls");
    const auto cached = approved.symbol<Tensor>("cudnnCreateTensorDescriptor");
    expect(cached == &cudnnCreateTensorDescriptor,
           "receipt fixture cached constructor retains the owned frontend");
    auto first = token<cudnnTensorDescriptor_t>(0x1234);
    expect(cudnnCreateTensorDescriptor(&first) == CUDNN_STATUS_INTERNAL_ERROR &&
               first == token<cudnnTensorDescriptor_t>(0x1234),
           "malformed binding returns INTERNAL_ERROR without publishing a descriptor");
    auto later = token<cudnnTensorDescriptor_t>(0x5678);
    expect(cached(&later) == CUDNN_STATUS_INTERNAL_ERROR &&
               cudnnCreateTensorDescriptor(&later) == CUDNN_STATUS_INTERNAL_ERROR &&
               later == token<cudnnTensorDescriptor_t>(0x5678),
           "quarantined direct and cached constructors keep output unchanged");
    expect(calls() == 0, "malformed binding never enters the approved native SDK provider");
}
void binding_receipt_child(const char* approved_path, unsigned int scenario) {
    expect(scenario < kBindingReceiptCases, "bounded representative binding receipt scenario");
    const auto path = "/tmp/glimmer-dnn-binding-" + std::to_string(::getpid()) + ".sock";
    ExecutionSocketConfig config;
    config.socket_path = path;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start authenticated malformed-binding endpoint");
    const auto identity = current_process_identity();
    expect(identity.has_value(), "obtain exact malformed-binding endpoint incarnation");
    const auto pid = std::to_string(identity->pid);
    const auto start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure owned metadata endpoint before first constructor");
    auto client = std::async(std::launch::async, [=] { binding_application(approved_path); });
    struct Stop {
        ExecutionSocketServer& server;
        ~Stop() {
            server.stop();
        }
    } stop{server};
    unsigned int hellos{}, bindings{}, unexpected{};
    std::uint64_t connection{};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (client.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        expect(std::chrono::steady_clock::now() < deadline, "bounded malformed-binding exchange");
        expect(server.poll_for(1), "poll authenticated malformed-binding traffic");
        auto request = server.take_request();
        if (!request) continue;
        ExecutionPayloadReader input(request->message.payload);
        ExecutionWireReply reply;
        reply.disposition = ExecutionWireDisposition::kComplete;
        const auto operation = request->message.header.operation;
        if (operation == ExecutionOpcode::kHello && !hellos && !bindings) {
            std::uint32_t role{}, flags{};
            expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                       input.read_u32(flags) && !flags && input.empty() &&
                       request->descriptor.get() < 0,
                   "constructor opens only canonical DNM1 with no attached resource");
            connection = request->connection;
            ++hellos;
            const auto ownership = encode_execution_bootstrap_ownership(
                ExecutionBootstrapOwnership::kMetadataRetained);
            expect(ownership.has_value(), "encode real CPU metadata bootstrap ownership");
            reply.data.assign(ownership->begin(), ownership->end());
        } else if (operation == ExecutionOpcode::kBindDnnProfile && hellos == 1 && !bindings) {
            std::uint32_t version{};
            expect(request->connection == connection && request->descriptor.get() < 0 &&
                       read_execution_dnn_profile(input, version) && version == CUDNN_VERSION &&
                       input.empty(),
                   "first descriptor request binds its actual header profile before any DNN RPC");
            ++bindings;
            reply = malformed_binding(scenario);
        } else {
            ++unexpected;
            server.close_peer(request->connection);
            continue;
        }
        auto header = request->message.header;
        header.kind = ExecutionMessageKind::kReply;
        header.session = connection;
        header.incarnation = {211, 223};
        auto encoded = encode_execution_reply(reply);
        expect(encoded && server.reply(connection, {header, std::move(*encoded)}),
               "deliver exactly the selected hostile binding receipt");
    }
    client.get();
    expect(
        server.poll_for(0) && !server.take_request() && hellos == 1 && bindings == 1 && !unexpected,
        "quarantine prevents DNN, rebind, reconnect and fabricated cleanup RPCs");
}
void binding_receipts(const char* approved_path, const char* wrong_path) {
    for (unsigned int scenario = 0; scenario != kBindingReceiptCases; ++scenario) {
        const auto index = std::to_string(scenario);
        const auto child = ::fork();
        expect(child >= 0, "fork an independently initialized receipt process");
        if (!child) {
            ::execl("/proc/self/exe", "execution_dnn_profile_lookup_test", approved_path,
                    wrong_path, "--binding-receipt-child", index.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("binding receipt child " + index +
                                     " failed, wait status=" + std::to_string(status));
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string_view{argv[3]} == "--binding-receipt-child") {
            unsigned int scenario{};
            const std::string_view index{argv[4]};
            const auto parsed =
                std::from_chars(index.data(), index.data() + index.size(), scenario);
            expect(parsed.ec == std::errc{} && parsed.ptr == index.data() + index.size(),
                   "receipt child has one exact numeric scenario");
            binding_receipt_child(argv[1], scenario);
            return 0;
        }
        if (argc == 4 && std::string_view{argv[3]} == "--binding-receipts") {
            binding_receipts(argv[1], argv[2]);
            std::cout << "cuDNN profile malformed binding receipts passed\n";
            return 0;
        }
        expect(argc == 3 || (argc == 4 && std::string_view{argv[3]} == "--new-namespace"),
               "usage: test approved-provider wrong-provider [--new-namespace]");
        run(argv[1], argv[2], argc == 4);
        std::cout << "cuDNN profile direct/cached identity rejection passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
