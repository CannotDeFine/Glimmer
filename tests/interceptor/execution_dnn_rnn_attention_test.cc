#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_rnn_attention_fixture.h"

#include <cerrno>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_rnn_attention_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_rnn_attention;
using Clock = std::chrono::steady_clock;
constexpr unsigned int kHostileCases = 48;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t result) {
    expect(result == CUDNN_STATUS_SUCCESS, "original RNN/attention success");
}
struct Cached {
    decltype(&cudnnCreateRNNDescriptor) create_rnn{};
    decltype(&cudnnDestroyRNNDescriptor) destroy_rnn{};
    decltype(&cudnnSetRNNDescriptor_v8) set_rnn{};
    decltype(&cudnnGetRNNDescriptor_v8) get_rnn{};
    decltype(&cudnnCreateAttnDescriptor) create_attn{};
    decltype(&cudnnDestroyAttnDescriptor) destroy_attn{};
    decltype(&cudnnSetAttnDescriptor) set_attn{};
    decltype(&cudnnGetAttnDescriptor) get_attn{};
};
Cached lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "open independent RNN/attention provider");
    Cached api;
#define LOOKUP(member, symbol)                                                                    \
    do {                                                                                          \
        auto* value = ::dlsym(library, #symbol);                                                  \
        expect(                                                                                   \
            value == reinterpret_cast<void*>(&symbol) && value == ::dlsym(RTLD_DEFAULT, #symbol), \
            "all lookup paths stay controlled");                                                  \
        api.member = reinterpret_cast<decltype(api.member)>(value);                               \
    } while (false)
    LOOKUP(create_rnn, cudnnCreateRNNDescriptor);
    LOOKUP(destroy_rnn, cudnnDestroyRNNDescriptor);
    LOOKUP(set_rnn, cudnnSetRNNDescriptor_v8);
    LOOKUP(get_rnn, cudnnGetRNNDescriptor_v8);
    LOOKUP(create_attn, cudnnCreateAttnDescriptor);
    LOOKUP(destroy_attn, cudnnDestroyAttnDescriptor);
    LOOKUP(set_attn, cudnnSetAttnDescriptor);
    LOOKUP(get_attn, cudnnGetAttnDescriptor);
#undef LOOKUP
    expect(::dlclose(library) == 0, "cached public symbols survive provider close");
    return api;
}
cudnnStatus_t rnn(const Cached& api, cudnnRNNDescriptor_t parent,
                  cudnnDropoutDescriptor_t dependency) {
    return api.set_rnn(parent, static_cast<cudnnRNNAlgo_t>(7), CUDNN_GRU, CUDNN_RNN_DOUBLE_BIAS,
                       CUDNN_BIDIRECTIONAL, CUDNN_SKIP_INPUT, static_cast<cudnnDataType_t>(31),
                       static_cast<cudnnDataType_t>(30), CUDNN_FMA_MATH, INT32_MIN, INT32_MAX, -7,
                       0, dependency, UINT32_MAX);
}
cudnnStatus_t attn(const Cached& api, cudnnAttnDescriptor_t parent, cudnnDropoutDescriptor_t first,
                   cudnnDropoutDescriptor_t second) {
    return api.set_attn(parent, UINT32_MAX, INT32_MIN, -0.0, static_cast<cudnnDataType_t>(31),
                        static_cast<cudnnDataType_t>(30), CUDNN_FMA_MATH, first, second, INT32_MIN,
                        INT32_MAX, -1, 0, 1, 2, 3, 4, 5, 6, 7);
}
void guards(const Cached& api, cudnnRNNDescriptor_t recurrent, cudnnAttnDescriptor_t attention) {
    std::array<int, 11> sizes{};
    sizes.fill(91);
    unsigned int flags = 0xabcdef01;
    double scale = -13.5;
    cudnnDataType_t type = CUDNN_DATA_DOUBLE, precision = CUDNN_DATA_HALF;
    cudnnMathType_t math = CUDNN_FMA_MATH;
    cudnnRNNAlgo_t algorithm = CUDNN_RNN_ALGO_STANDARD;
    cudnnRNNMode_t cell = CUDNN_GRU;
    cudnnRNNBiasMode_t bias = CUDNN_RNN_DOUBLE_BIAS;
    cudnnDirectionMode_t direction = CUDNN_BIDIRECTIONAL;
    cudnnRNNInputMode_t input = CUDNN_SKIP_INPUT;
    cudnnDropoutDescriptor_t first{}, second{};
    expect(api.get_rnn(recurrent, &algorithm, &cell, &bias, &direction, &input, &type, &precision,
                       &math, &sizes[0], &sizes[1], &sizes[2], &sizes[3], &first,
                       &flags) == CUDNN_STATUS_NOT_SUPPORTED &&
               api.get_attn(attention, &flags, &sizes[0], &scale, &type, &precision, &math, &first,
                            &second, &sizes[0], &sizes[1], &sizes[2], &sizes[3], &sizes[4],
                            &sizes[5], &sizes[6], &sizes[7], &sizes[8], &sizes[9],
                            &sizes[10]) == CUDNN_STATUS_NOT_SUPPORTED &&
               std::ranges::all_of(sizes, [](int value) { return value == 91; }) &&
               flags == 0xabcdef01 && scale == -13.5 && type == CUDNN_DATA_DOUBLE &&
               precision == CUDNN_DATA_HALF && math == CUDNN_FMA_MATH && !first && !second &&
               algorithm == CUDNN_RNN_ALGO_STANDARD && cell == CUDNN_GRU &&
               bias == CUDNN_RNN_DOUBLE_BIAS && direction == CUDNN_BIDIRECTIONAL &&
               input == CUDNN_SKIP_INPUT,
           "two typed pinned getters are guards and never publish outputs");
}
void wait_flag(const std::atomic<bool>& flag) {
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    while (!flag.load()) {
        expect(Clock::now() < deadline, "bounded started-thread handshake");
        std::this_thread::yield();
    }
}
void ordinary(const char* provider) {
    const auto api = lookup(provider);
    cudnnRNNDescriptor_t recurrent{};
    cudnnAttnDescriptor_t attention{};
    cudnnDropoutDescriptor_t first{}, second{}, third{};
    check(cudnnCreateRNNDescriptor(&recurrent));
    check(cudnnCreateAttnDescriptor(&attention));
    check(cudnnCreateDropoutDescriptor(&first));
    check(cudnnCreateDropoutDescriptor(&second));
    check(cudnnCreateDropoutDescriptor(&third));
    check(rnn(api, recurrent, first));
    check(attn(api, attention, first, first));
    auto before = Clock::now();
    expect(cudnnDestroyDropoutDescriptor(first) == CUDNN_STATUS_BAD_PARAM &&
               Clock::now() - before < std::chrono::seconds{2},
           "referenced dropout destruction is a prompt clean policy error, not deferred forever");
    check(rnn(api, recurrent, second));
    expect(cudnnDestroyDropoutDescriptor(first) == CUDNN_STATUS_BAD_PARAM,
           "remaining attention dependency survives unrelated RNN rebind");
    check(attn(api, attention, second, third));
    check(cudnnDestroyDropoutDescriptor(first));
    check(attn(api, attention, second, second));
    check(cudnnDestroyDropoutDescriptor(third));
    // Pause an accepted mutation; caller destroy must wait for the transient pin
    // and then observe the new dependency. Service serialization also applies.
    fixture::control.pause.store(true);
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    auto mutation =
        std::async(std::launch::async, [&] { return attn(api, attention, second, second); });
    wait_flag(fixture::control.entered);
    std::atomic<bool> started{};
    auto destroy = std::async(std::launch::async, [&] {
        started.store(true);
        return cudnnDestroyDropoutDescriptor(second);
    });
    wait_flag(started);
    expect(destroy.wait_for(std::chrono::milliseconds{20}) == std::future_status::timeout,
           "in-flight dependency pointer cannot retire before the original setter returns");
    fixture::control.release.store(true);
    check(mutation.get());
    expect(destroy.get() == CUDNN_STATUS_BAD_PARAM,
           "completed logical dependency still rejects destruction");
    const auto calls = fixture::control.callbacks.load();
    expect(
        rnn(api, recurrent, first) == CUDNN_STATUS_NOT_INITIALIZED &&
            rnn(api, recurrent, reinterpret_cast<cudnnDropoutDescriptor_t>(attention)) ==
                CUDNN_STATUS_NOT_INITIALIZED &&
            attn(api, attention, second, reinterpret_cast<cudnnDropoutDescriptor_t>(recurrent)) ==
                CUDNN_STATUS_NOT_INITIALIZED &&
            calls == fixture::control.callbacks.load(),
        "stale and wrong-kind dependencies reject before SDK");
    guards(api, recurrent, attention);
    const auto child = ::fork();
    expect(child >= 0, "fork inherited RNN/attention frontend");
    if (!child)
        ::_exit(rnn(api, recurrent, second) == CUDNN_STATUS_NOT_INITIALIZED &&
                        !glimmer_dnn_rnn_attention_native_calls()
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "inherited process cannot use original metadata or native fallback");
    check(rnn(api, recurrent, nullptr));
    expect(cudnnDestroyDropoutDescriptor(second) == CUDNN_STATUS_BAD_PARAM,
           "null RNN detach does not erase aliased attention dependency");
    check(attn(api, attention, nullptr, nullptr));
    check(cudnnDestroyDropoutDescriptor(second));
    check(api.destroy_rnn(recurrent));
    check(api.destroy_attn(attention));
    expect(api.destroy_rnn(recurrent) == CUDNN_STATUS_NOT_INITIALIZED &&
               !glimmer_dnn_rnn_attention_native_calls(),
           "stale handles remain contained");
}
void poison(const char* provider, bool attention_case) {
    const auto api = lookup(provider);
    cudnnRNNDescriptor_t recurrent{};
    cudnnAttnDescriptor_t attention{};
    std::array<cudnnDropoutDescriptor_t, 4> dependencies{};
    if (attention_case)
        check(api.create_attn(&attention));
    else
        check(api.create_rnn(&recurrent));
    for (auto& dependency : dependencies) check(cudnnCreateDropoutDescriptor(&dependency));
    if (attention_case)
        check(attn(api, attention, dependencies[0], dependencies[1]));
    else
        check(rnn(api, recurrent, dependencies[0]));
    fixture::control.set_status.store(CUDNN_STATUS_BAD_PARAM);
    expect((attention_case ? attn(api, attention, dependencies[2], dependencies[3])
                           : rnn(api, recurrent, dependencies[2])) == CUDNN_STATUS_BAD_PARAM,
           "issued partially mutating rebind returns exact original SDK error");
    const auto calls = fixture::control.callbacks.load();
    for (auto* dependency : dependencies)
        expect(cudnnDestroyDropoutDescriptor(dependency) == CUDNN_STATUS_NOT_INITIALIZED,
               "uncertainty retains every old and candidate dependency");
    expect(
        (attention_case ? attn(api, attention, nullptr, nullptr) : rnn(api, recurrent, nullptr)) ==
                CUDNN_STATUS_NOT_INITIALIZED &&
            calls == fixture::control.callbacks.load() && !glimmer_dnn_rnn_attention_native_calls(),
        "quarantine cannot replay, detach or invoke caller-native SDK");
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated metadata endpoint");
}
void run(const char* provider, ExecutionSchedulingMode mode, int failure = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path = "/tmp/glimmer-rnn-attn-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start genuine original metadata service");
    client = std::async(std::launch::async, [=] {
        if (failure >= 0)
            poison(provider, failure != 0);
        else
            ordinary(provider);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{20};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded original metadata progress");
    client.get();
    expect(fixture::foundation::is_resource_free() && !glimmer_dnn_rnn_attention_native_calls(),
           "all host metadata runs with zero Driver/context/compute/caller-native calls");
    if (failure >= 0) {
        expect(!fixture::state.destroys && !fixture::dropout::state.destroys &&
                   fixture::control.callbacks.load() == 2,
               "uncertain original graph remains retained after exactly one failed rebind");
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished() && fixture::state.creates == 2 &&
               fixture::state.destroys == 2 && fixture::dropout::state.creates == 3 &&
               fixture::dropout::state.destroys == 3 && fixture::state.rnn_sets == 3 &&
               fixture::state.attn_sets == 5 && fixture::state.last_rnn.flags == UINT32_MAX &&
               fixture::state.last_attn.scale_bits == 0x8000000000000000ULL,
           "exact argument transport and once-only dependent/dropout cleanup");
}

struct Hostile {
    ExecutionWireReply reply;
    bool attention{}, clean{};
    unsigned int operation{};  // Setter=0, Create=1, Destroy=2.
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
};
Hostile hostile(unsigned int index) {
    Hostile result;
    result.attention = index >= 24;
    const unsigned int variant = index % 24;
    result.operation = variant < 12 ? 0 : variant < 18 ? 1 : 2;
    auto& reply = result.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    if (result.operation == 1) reply.id = 99;
    const auto local = variant < 12 ? variant : variant < 18 ? variant - 12 : variant - 18;
    if (local == 0) reply.parameter_bytes = 1;
    if (local == 1) {
        if (result.operation == 1)
            reply.address = 7;
        else
            reply.id = 7;
    }
    if (local == 2) {
        if (result.operation == 1)
            reply.id = 0;
        else
            reply.address = 7;
    }
    if (local == 3) reply.data = {std::byte{7}};
    if (local == 4) {
        reply.parameter_bytes = 1;
        reply.arguments = {{0, 1}};
    }
    if (result.operation == 0) {
        if (local == 5) reply.original_status.reset();
        if (local == 6) reply.original_status = CUDNN_STATUS_BAD_PARAM;
        if (local == 7) {
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = -1;
        }
        if (local == 8 || local == 9) {
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            if (local == 9) {
                reply.parameter_bytes = 1;
                result.status = CUDNN_STATUS_BAD_PARAM;
            }
        }
        if (local == 10) {
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            result.status = CUDNN_STATUS_NOT_SUPPORTED;
            result.clean = true;
        }
        if (local == 11) {
            result.clean = true;
            result.status = CUDNN_STATUS_SUCCESS;
        }
    } else if (local == 5) {
        reply.id = 0;
        reply.status = CUDNN_STATUS_ALLOC_FAILED;
        reply.original_status = reply.status;
        reply.parameter_bytes = result.operation == 2 ? 1 : 0;
        result.status = CUDNN_STATUS_ALLOC_FAILED;
        result.clean = result.operation == 1;
    }
    return result;
}
void hostile_case(const char* provider, unsigned int index) {
    const auto test = hostile(index);
    const auto identity = current_process_identity();
    expect(identity.has_value(), "hostile dependency endpoint identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-rnn-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start real hostile dependency transport");
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnRNNDescriptor_t recurrent{};
        cudnnAttnDescriptor_t attention{};
        cudnnDropoutDescriptor_t dependency{};
        if (test.attention)
            check(api.create_attn(&attention));
        else
            check(api.create_rnn(&recurrent));
        check(cudnnCreateDropoutDescriptor(&dependency));
        cudnnStatus_t status{};
        if (test.operation == 0)
            status = test.attention ? attn(api, attention, dependency, dependency)
                                    : rnn(api, recurrent, dependency);
        else if (test.operation == 2)
            status = test.attention ? api.destroy_attn(attention) : api.destroy_rnn(recurrent);
        else if (test.attention) {
            auto* output = reinterpret_cast<cudnnAttnDescriptor_t>(0xfee);
            status = api.create_attn(&output);
            expect(output == reinterpret_cast<cudnnAttnDescriptor_t>(0xfee),
                   "failed/malformed Attn create output untouched");
        } else {
            auto* output = reinterpret_cast<cudnnRNNDescriptor_t>(0xfee);
            status = api.create_rnn(&output);
            expect(output == reinterpret_cast<cudnnRNNDescriptor_t>(0xfee),
                   "failed/malformed RNN create output untouched");
        }
        expect(status == test.status, "hostile dependency receipt has exact contained status");
        const auto follow = test.attention ? attn(api, attention, dependency, dependency)
                                           : rnn(api, recurrent, dependency);
        expect(follow == (test.clean ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_INITIALIZED) &&
                   !glimmer_dnn_rnn_attention_native_calls(),
               "only healthy original/policy receipts allow another owned operation");
    });
    unsigned int hellos{}, creates{}, operations{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1),
               "bounded hostile dependency progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "authenticate actual hostile peer");
            if (profiles.respond(server, *request)) continue;
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            ExecutionPayloadReader input(request->message.payload);
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "metadata-only DNM1 bootstrap");
                ++hellos;
                const auto ownership = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(ownership.has_value(), "encode metadata ownership");
                reply.data.assign(ownership->begin(), ownership->end());
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata &&
                       creates < 2) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && input.empty() &&
                           kind == static_cast<unsigned int>(
                                       creates          ? ExecutionDnnObjectKind::kDropout
                                       : test.attention ? ExecutionDnnObjectKind::kAttn
                                                        : ExecutionDnnObjectKind::kRnn),
                       "exact parent/dropout creation kinds");
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = ++creates + 16;
            } else {
                ++operations;
                const unsigned int operation = operations == 1 ? test.operation : 0;
                if (operation == 1) {
                    std::uint32_t kind{};
                    expect(
                        request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata &&
                            input.read_u32(kind) && input.empty() &&
                            kind == static_cast<unsigned int>(test.attention
                                                                  ? ExecutionDnnObjectKind::kAttn
                                                                  : ExecutionDnnObjectKind::kRnn),
                        "tested creation request is exact");
                } else {
                    std::uint64_t id{};
                    expect(input.read_u64(id) && id == 17, "exact original parent id");
                    if (operation == 2)
                        expect(request->message.header.operation ==
                                       ExecutionOpcode::kDestroyDnnMetadata &&
                                   input.empty(),
                               "canonical destroy request");
                    else if (test.attention) {
                        ExecutionDnnAttnState value;
                        expect(request->message.header.operation == ExecutionOpcode::kSetDnnAttn &&
                                   read_execution_dnn_attn(input, value) && input.empty() &&
                                   value.dropouts == std::array<std::uint64_t, 2>{18, 18},
                               "attention preserves both positional aliased dependencies");
                    } else {
                        ExecutionDnnRnnState value;
                        expect(request->message.header.operation == ExecutionOpcode::kSetDnnRnn &&
                                   read_execution_dnn_rnn(input, value) && input.empty() &&
                                   value.dropout == 18,
                               "RNN carries exact dependency identity");
                    }
                }
                if (operations == 1)
                    reply = test.reply;
                else {
                    expect(test.clean && operations == 2, "no hidden recovery request");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded && decode_execution_reply(*encoded),
                   "hostile receipt is genuinely wire-valid");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {71, 93};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver controlled hostile receipt");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && creates == 2 && operations == (test.clean ? 2U : 1U),
           "malformed receipts never replay setters or cleanup");
    server.stop();
}
unsigned int number(const char* value) {
    unsigned int result{};
    const std::string_view text{value};
    expect(!text.empty() && text.size() <= 2, "bounded child index");
    for (const char digit : text) {
        expect(digit >= '0' && digit <= '9', "decimal child index");
        result = result * 10 + static_cast<unsigned int>(digit - '0');
    }
    return result;
}
void children(const char* executable, const char* provider, bool malformed) {
    for (unsigned int index = 0; index != (malformed ? kHostileCases : 2); ++index) {
        const auto value = std::to_string(index);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated RNN/attention fixture");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider,
                    malformed ? "--malformed-case" : "--poison-case", value.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("isolated RNN/attention case failed: " + value);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional RNN/attention mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--poison" || mode == "--malformed-replies") {
            expect(argc == 3, "aggregate mode");
            children(argv[0], argv[1], mode == "--malformed-replies");
        } else if (mode == "--poison-case" || mode == "--malformed-case") {
            expect(argc == 4, "isolated case index");
            const auto index = number(argv[3]);
            if (mode == "--malformed-case") {
                expect(index < kHostileCases, "known hostile case");
                hostile_case(argv[1], index);
                ::_exit(0);
            }
            expect(index < 2, "known poison case");
            run(argv[1], ExecutionSchedulingMode::kNaive, static_cast<int>(index));
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known scheduling mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "RNN/attention original metadata CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
