#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_attn_compute_fixture.h"

#include <cerrno>
#include <atomic>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
extern "C" unsigned int glimmer_dnn_attn_compute_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_attn_compute;
namespace foundation = fixture::foundation;
namespace oracle = fixture::oracle;
using Clock = std::chrono::steady_clock;
struct TailHold {
    std::atomic<bool> is_armed{}, has_entered{};
};
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudnnStatus_t value) {
    if (value != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public attention SDK status " + std::to_string(value));
}
void driver(CUresult value) {
    expect(value == CUDA_SUCCESS, "owned public Driver operation");
}
void near(double actual, double expected, double tolerance) {
    expect(std::isfinite(actual) && std::abs(actual - expected) < tolerance,
           "complete public projected attention numerical output");
}
struct Cached {
    decltype(&cudnnMultiHeadAttnForward) forward{};
    decltype(&cudnnMultiHeadAttnBackwardData) data{};
    decltype(&cudnnMultiHeadAttnBackwardWeights) weights{};
};
Cached lookup(const char* provider) {
    auto* library = dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent caller provider");
    Cached api;
#define LOOKUP(member, symbol)                                                                     \
    do {                                                                                           \
        auto* value = dlsym(library, #symbol);                                                     \
        expect(value == reinterpret_cast<void*>(&symbol) && value == dlsym(RTLD_DEFAULT, #symbol), \
               "direct/global/explicit provider symbol identity");                                 \
        api.member = reinterpret_cast<decltype(api.member)>(value);                                \
    } while (false)
    LOOKUP(forward, cudnnMultiHeadAttnForward);
    LOOKUP(data, cudnnMultiHeadAttnBackwardData);
    LOOKUP(weights, cudnnMultiHeadAttnBackwardWeights);
#undef LOOKUP
    expect(dlclose(library) == 0, "cached wrappers survive explicit provider close");
    return api;
}
struct Application {
    Cached api;
    cudnnAttnDescriptor_t parent{};
    std::array<cudnnSeqDataDescriptor_t, 4> descriptors{};
    std::array<glimmer::backend::cuda::CudaDnnSeqDataState, 4> data;
    std::array<CUdeviceptr, 15> memory{};
    std::array<std::size_t, 15> bytes{};
    std::array<std::vector<std::byte>, 15> initial;
    CUcontext context{};
    CUstream stream{};
    cudnnHandle_t handle{};
    std::size_t scalar{};
    explicit Application(const char* provider, bool is_double = false) : api(lookup(provider)) {
        const auto type = is_double ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT;
        scalar = is_double ? 8 : 4;
        check(cudnnCreateAttnDescriptor(&parent));
        check(cudnnSetAttnDescriptor(
            parent, CUDNN_ATTN_ENABLE_PROJ_BIASES | CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, 1, .75, type,
            type, CUDNN_DEFAULT_MATH, nullptr, nullptr, 2, 2, 2, 2, 2, 2, 2, 4, 5, 3, 3));
        for (std::size_t i = 0; i != 4; ++i) {
            check(cudnnCreateSeqDataDescriptor(&descriptors[i]));
            const int time = i == 1 || i == 2 ? 3 : 2;
            data[i].type = static_cast<unsigned int>(type);
            data[i].dimensions = 4;
            data[i].sizes = {time, 2, 2, 2};
            data[i].axes = fixture::kAxes[is_double ? 5 : 2];
            data[i].sequence_count = 4;
            data[i].lengths.assign(4, time);
            set_sequence(i);
        }
        driver(cuCtxCreate(&context, 0, 0));
        check(cudnnCreate(&handle));
        if (!is_double) {
            driver(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
            check(cudnnSetStream(handle, reinterpret_cast<cudaStream_t>(stream)));
        }
        bytes = {16,          16,          16 * scalar, 24 * scalar, 24 * scalar,
                 16 * scalar, 16 * scalar, 16 * scalar, 16 * scalar, 24 * scalar,
                 24 * scalar, 24 * scalar, 24 * scalar, 32,          fixture::kReserve * scalar};
        for (std::size_t i = 0; i != 15; ++i) {
            driver(cuMemAlloc(&memory[i], bytes[i] + 32));
            initial[i].assign(bytes[i] + 32, std::byte{0xa5});
            std::fill(initial[i].begin() + 16, initial[i].end() - 16, std::byte{});
        }
        if (is_double)
            initialize<double>();
        else
            initialize<float>();
        for (std::size_t i = 0; i != 15; ++i)
            driver(cuMemcpyHtoD(memory[i], initial[i].data(), initial[i].size()));
    }
    void set_sequence(std::size_t index) {
        std::array<cudnnSeqDataAxis_t, 4> axes{};
        for (std::size_t i = 0; i != 4; ++i)
            axes[i] = static_cast<cudnnSeqDataAxis_t>(data[index].axes[i]);
        check(cudnnSetSeqDataDescriptor(
            descriptors[index], static_cast<cudnnDataType_t>(data[index].type), 4,
            data[index].sizes.data(), axes.data(), 4, data[index].lengths.data(), nullptr));
    }
    template <typename T>
    void store(std::size_t slot, std::size_t offset, T value) {
        std::memcpy(initial[slot].data() + 16 + offset * sizeof(T), &value, sizeof(T));
    }
    template <typename T>
    void initialize() {
        for (std::size_t i = 0; i != 4; ++i) {
            store<int>(0, i, 2);
            store<int>(1, i, 3);
        }
        const auto weights = fixture::flatten(oracle::seeded_parameters());
        for (std::size_t i = 0; i != 24; ++i) {
            store<T>(11, i, static_cast<T>(weights[i]));
            store<T>(12, i, T{.125});
        }
        for (std::size_t b = 0; b != 2; ++b)
            for (std::size_t beam = 0; beam != 2; ++beam) {
                const auto input = fixture::inputs(b, beam);
                const auto upstream = fixture::upstream(b, beam);
                for (std::size_t t = 0; t != 2; ++t)
                    for (std::size_t c = 0; c != 2; ++c) {
                        const auto i = fixture::index(data[0], t, b, beam, c);
                        store<T>(2, i, static_cast<T>(input.queries[2 * t + c]));
                        store<T>(5, i, static_cast<T>(input.residual[2 * t + c]));
                        store<T>(7, fixture::index(data[3], t, b, beam, c),
                                 static_cast<T>(upstream.out[2 * t + c]));
                    }
                for (std::size_t t = 0; t != 3; ++t)
                    for (std::size_t c = 0; c != 2; ++c) {
                        store<T>(3, fixture::index(data[1], t, b, beam, c),
                                 static_cast<T>(input.keys[2 * t + c]));
                        store<T>(4, fixture::index(data[2], t, b, beam, c),
                                 static_cast<T>(input.values[2 * t + c]));
                    }
            }
    }
    void* pointer(std::size_t i) const {
        return reinterpret_cast<void*>(memory[i] + 16);
    }
    cudnnStatus_t forward(int current = -1, const int* lower = nullptr, const int* upper = nullptr,
                          bool training = true, bool missing_q = false) const {
        static const int lo[]{0, 0, 0, 0}, hi[]{3, 3, 3, 3};
        return api.forward(handle, parent, current, lower ? lower : lo, upper ? upper : hi,
                           static_cast<const int*>(pointer(0)), static_cast<const int*>(pointer(1)),
                           missing_q ? nullptr : descriptors[0], pointer(2), pointer(5),
                           descriptors[1], pointer(3), descriptors[2], pointer(4), descriptors[3],
                           pointer(6), bytes[11], pointer(11), bytes[13], pointer(13),
                           training ? bytes[14] : 0, training ? pointer(14) : nullptr);
    }
    cudnnStatus_t invoke(unsigned int kind, bool set = false) const {
        static const int lo[]{0, 0, 0, 0}, hi[]{3, 3, 3, 3};
        if (kind == 0) return forward();
        if (kind == 1)
            return api.data(handle, parent, lo, hi, static_cast<const int*>(pointer(0)),
                            static_cast<const int*>(pointer(1)), descriptors[3], pointer(7),
                            descriptors[0], pointer(8), pointer(2), descriptors[1], pointer(9),
                            pointer(3), descriptors[2], pointer(10), pointer(4), bytes[11],
                            pointer(11), bytes[13], pointer(13), bytes[14], pointer(14));
        return api.weights(handle, parent, set ? CUDNN_WGRAD_MODE_SET : CUDNN_WGRAD_MODE_ADD,
                           descriptors[0], pointer(2), descriptors[1], pointer(3), descriptors[2],
                           pointer(4), descriptors[3], pointer(7), bytes[11], pointer(11),
                           pointer(12), bytes[13], pointer(13), bytes[14], pointer(14));
    }
    template <typename T>
    std::vector<T> read(std::size_t slot) const {
        std::vector<T> result(bytes[slot] / sizeof(T));
        driver(cuMemcpyDtoH(result.data(), memory[slot] + 16, bytes[slot]));
        return result;
    }
    void verify_storage() const {
        for (std::size_t i = 0; i != 15; ++i) {
            std::vector<std::byte> actual(bytes[i] + 32);
            driver(cuMemcpyDtoH(actual.data(), memory[i], actual.size()));
            expect(std::equal(actual.begin(), actual.begin() + 16, initial[i].begin()) &&
                       std::equal(actual.end() - 16, actual.end(), initial[i].end() - 16),
                   "complete allocation guards preserved");
            if (i <= 5 || i == 7 || i == 11)
                expect(actual == initial[i], "all readonly payload bytes preserved");
        }
    }
    void close() {
        for (const auto address : memory) driver(cuMemFree(address));
        check(cudnnDestroy(handle));
        for (const auto descriptor : descriptors) check(cudnnDestroySeqDataDescriptor(descriptor));
        check(cudnnDestroyAttnDescriptor(parent));
        if (stream) driver(cuStreamDestroy(stream));
        driver(cuCtxDestroy(context));
    }
};
template <typename T>
void numerical(Application& app) {
    check(app.invoke(0));
    check(app.invoke(1));
    check(app.invoke(2));
    check(app.invoke(2));
    const auto y = app.read<T>(6), dq = app.read<T>(8), dk = app.read<T>(9), dv = app.read<T>(10);
    std::array<double, 24> gradient{};
    const auto parameters = oracle::seeded_parameters();
    const double epsilon = 1e-5, tolerance = sizeof(T) == 8 ? 2e-8 : 8e-5;
    for (std::size_t b = 0; b != 2; ++b)
        for (std::size_t beam = 0; beam != 2; ++beam) {
            auto input = fixture::inputs(b, beam);
            const auto upstream = fixture::upstream(b, beam);
            const auto expected = oracle::forward(parameters, input);
            for (std::size_t t = 0; t != 2; ++t)
                for (std::size_t c = 0; c != 2; ++c)
                    near(y[fixture::index(app.data[3], t, b, beam, c)], expected.out[2 * t + c],
                         tolerance);
            const auto difference = [&](double& variable) {
                const auto saved = variable;
                variable = saved + epsilon;
                const auto plus = oracle::loss(oracle::forward(parameters, input), upstream);
                variable = saved - epsilon;
                const auto minus = oracle::loss(oracle::forward(parameters, input), upstream);
                variable = saved;
                return (plus - minus) / (2 * epsilon);
            };
            for (std::size_t i = 0; i != 4; ++i)
                near(dq[fixture::index(app.data[0], i / 2, b, beam, i % 2)],
                     difference(input.queries[i]), tolerance);
            for (std::size_t i = 0; i != 6; ++i) {
                near(dk[fixture::index(app.data[1], i / 2, b, beam, i % 2)],
                     difference(input.keys[i]), tolerance);
                near(dv[fixture::index(app.data[2], i / 2, b, beam, i % 2)],
                     difference(input.values[i]), tolerance);
            }
            for (std::size_t i = 0; i != 24; ++i) {
                auto changed = parameters;
                auto& variable =
                    i % 6 < 4 ? changed.matrices[i / 6][i % 6] : changed.biases[i / 6][i % 6 - 4];
                variable += epsilon;
                const auto plus = oracle::loss(oracle::forward(changed, input), upstream);
                variable -= 2 * epsilon;
                const auto minus = oracle::loss(oracle::forward(changed, input), upstream);
                gradient[i] += (plus - minus) / (2 * epsilon);
            }
        }
    auto weights = app.read<T>(12);
    for (std::size_t i = 0; i != 24; ++i) near(weights[i], .125 + 2 * gradient[i], tolerance);
    check(app.invoke(2, true));
    weights = app.read<T>(12);
    for (std::size_t i = 0; i != 24; ++i) near(weights[i], gradient[i], tolerance);
    app.verify_storage();
}
void unread_windows(Application& app) {
    fixture::control.arithmetic.store(false);
    const auto page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    expect(page >= 4096, "positive page size");
    auto* region = static_cast<std::byte*>(
        mmap(nullptr, 2 * page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    expect(region != MAP_FAILED, "guarded host-window mapping");
    expect(mprotect(region + page, page, PROT_READ | PROT_WRITE) == 0,
           "selected window page writable");
    const int selected = 1;
    std::memcpy(region + page, &selected, sizeof(selected));
    const auto* source = reinterpret_cast<const int*>(region + page - sizeof(int));
    check(app.forward(1, source, source, false));
    expect(fixture::state.observed_windows[0] == std::vector<int>{1} &&
               fixture::state.observed_windows[1] == std::vector<int>{1},
           "incremental capture reads only selected element, never protected prefix");
    app.data[0].lengths.assign(4, 1);
    app.set_sequence(0);
    app.data[3].lengths.assign(4, 1);
    app.set_sequence(3);
    const auto* unread = reinterpret_cast<const int*>(region);
    check(app.forward(1, unread, unread, false));
    expect(app.forward(2, unread, unread, false) == CUDNN_STATUS_BAD_PARAM,
           "out-of-range original error does not read caller windows");
    expect(app.forward(-1, unread, unread, true, true) == CUDNN_STATUS_BAD_PARAM,
           "missing descriptor original error does not read caller windows");
    app.data[0].lengths.assign(4, 2);
    app.set_sequence(0);
    app.data[3].lengths.assign(4, 2);
    app.set_sequence(3);
    expect(munmap(region, 2 * page) == 0, "release exact guard mapping");
}
void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "authenticated endpoint");
}
void run(const char* provider, ExecutionSchedulingMode mode, unsigned int missing = 0,
         int poison = -1) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "peer identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-attn-compute-" + std::to_string(getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    const auto socket_path = config.transport.socket_path;
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    const auto tail_hold = std::make_shared<TailHold>();
    fixture::state.callback = [tail_hold](unsigned int phase) {
        // Profile mode must retire its prior tail before this original enters.
        if (phase == 3 && tail_hold->is_armed.exchange(false)) {
            foundation::control.can_stop.store(false);
            tail_hold->has_entered.store(true);
        }
    };
    ExecutionService service(std::move(config), fixture::session_dispatch(missing));
    expect(service.start(), "start actual owned attention IPC service");
    auto client = std::async(std::launch::async, [=] {
        Application app(provider);
        if (missing) {
            expect(app.invoke(missing <= 3 ? missing - 1 : 0) == CUDNN_STATUS_NOT_SUPPORTED &&
                       !fixture::calls(),
                   "missing original/preflight capability does not fall back");
            app.close();
            return;
        }
        if (poison >= 0) {
            fixture::control.arithmetic.store(false);
            if (poison == 0) fixture::control.context_loss.store(3);
            if (poison == 1) {
                fixture::control.native_status.store(CUDNN_STATUS_EXECUTION_FAILED);
                fixture::control.sync_status.store(CUDA_ERROR_LAUNCH_FAILED);
            }
            if (poison == 2) fixture::control.stream_change.store(true);
            if (poison == 3) fixture::control.corruption.store(1);
            if (poison == 4) fixture::control.context_loss.store(2);
            const auto expected = poison == 0 || poison == 4 ? CUDNN_STATUS_NOT_INITIALIZED
                                                             : CUDNN_STATUS_INTERNAL_ERROR;
            expect(app.invoke(0) == expected,
                   "poisoned proof/private metadata cannot publish success");
            const auto before = fixture::calls();
            expect(app.invoke(0) == CUDNN_STATUS_NOT_INITIALIZED && fixture::calls() == before,
                   "quarantined public owner rejects repeat without RPC/original issue");
            return;
        }
        numerical<float>(app);
        unread_windows(app);
        const auto child = fork();
        expect(child >= 0, "fork inherited attention handles");
        if (!child) _exit(app.invoke(0) == CUDNN_STATUS_NOT_INITIALIZED ? 0 : 1);
        int status{};
        expect(waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
               "fork inherited authority rejects locally");
        CUcontext other{};
        driver(cuCtxCreate(&other, 0, 0));
        const auto before = fixture::calls();
        expect(app.invoke(0) == CUDNN_STATUS_NOT_INITIALIZED && fixture::calls() == before,
               "cross-context handle rejection");
        driver(cuCtxDestroy(other));
        driver(cuCtxSetCurrent(app.context));
        const auto syncs = fixture::control.syncs.load();
        fixture::control.native_status.store(CUDNN_STATUS_EXECUTION_FAILED);
        expect(app.invoke(0) == CUDNN_STATUS_EXECUTION_FAILED &&
                   fixture::control.syncs.load() == syncs + 1,
               "original error preserved after exactly one same-context proof");
        fixture::control.native_status.store(CUDNN_STATUS_SUCCESS);
        tail_hold->is_armed.store(true);
        check(app.invoke(0));
        expect(tail_hold->has_entered.load() && !tail_hold->is_armed.load(),
               "selected original call arms its own physical tail exactly once");
        std::atomic<bool> started{};
        auto mutation = std::async(std::launch::async, [&] {
            started.store(true);
            app.set_sequence(0);
        });
        const auto deadline = Clock::now() + std::chrono::seconds{3};
        while (!started.load()) {
            expect(Clock::now() < deadline, "bounded mutation start");
            std::this_thread::yield();
        }
        const bool blocked =
            mutation.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
        foundation::control.can_stop.store(true);
        mutation.get();
        expect(blocked, "host ACK does not release physical descriptor leases");
        app.close();
        fixture::control.arithmetic.store(true);
        Application doubles(provider, true);
        numerical<double>(doubles);
        doubles.close();
    });
    const auto deadline = Clock::now() + std::chrono::seconds{25};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded real attention IPC progress");
    client.get();
    fixture::state.callback = {};
    expect(!glimmer_dnn_attn_compute_native_calls(), "zero caller-native callbacks");
    if (poison >= 0) {
        expect(fixture::calls() == (poison < 3 ? 1U : 0U),
               "poison case issues exactly once, or not at all for private observation failure");
        if (poison < 3) {
            expect(
                foundation::driver.memory.size() == 15 && !foundation::driver.frees &&
                    fixture::base::state.handle_creates == 1 &&
                    !fixture::base::state.handle_destroys &&
                    std::ranges::count_if(fixture::base::state.handles,
                                          [](const auto& handle) { return handle.alive; }) == 1 &&
                    !fixture::metadata::state.destroys,
                "unknown issued completion retains all device buffers, handle and parent");
        }
        // Private malformed metadata (3) and pre-issue context loss (4) do not
        // assert an issued GPU lifetime. In particular, metadata quarantine
        // alone is not evidence that a known CUDA owner cannot be cleaned.
        // These isolated children deliberately avoid uncertain native cleanup;
        // process exit is containment, not proof of native resource release.
        expect(unlink(socket_path.c_str()) == 0, "remove this isolated child's socket pathname");
        ::_exit(0);
    }
    expect(foundation::driver.memory.empty() && foundation::driver.contexts.empty(),
           "exact clean ownership teardown");
}

// A real authenticated fake peer supplies hostile receipts; no parser helper
// is called directly and no host pointer becomes a server allocation identity.
void malformed(const char* provider, unsigned int index) {
    const bool preparation = index < 18;
    const auto test = preparation ? index : index - 18;
    const bool clean =
        preparation ? (test == 15 || test == 16) : (test == 12 || test == 13 || test == 14);
    const auto expected = preparation ? (test == 15   ? CUDNN_STATUS_NOT_SUPPORTED
                                         : test == 16 ? CUDNN_STATUS_SUCCESS
                                         : test == 17 ? CUDNN_STATUS_BAD_PARAM
                                                      : CUDNN_STATUS_INTERNAL_ERROR)
                                      : (test == 12   ? CUDNN_STATUS_NOT_SUPPORTED
                                         : test == 13 ? CUDNN_STATUS_SUCCESS
                                         : test >= 14 ? CUDNN_STATUS_BAD_PARAM
                                                      : CUDNN_STATUS_INTERNAL_ERROR);
    const auto identity = current_process_identity();
    expect(bool(identity), "hostile peer identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-attn-hostile-" + std::to_string(getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "hostile peer starts");
    auto client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        cudnnAttnDescriptor_t parent{};
        check(cudnnCreateAttnDescriptor(&parent));
        CUcontext context{};
        driver(cuCtxCreate(&context, 0, 0));
        cudnnHandle_t handle{};
        check(cudnnCreate(&handle));
        const auto invoke = [&] {
            const auto* unread = reinterpret_cast<const int*>(1);
            return preparation
                       ? api.forward(handle, parent, -1, unread, unread, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                     nullptr, 0, nullptr, 0, nullptr, 0, nullptr)
                       : api.weights(handle, parent, CUDNN_WGRAD_MODE_ADD, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
                                     nullptr, nullptr, 0, nullptr, 0, nullptr);
        };
        expect(invoke() == expected, "hostile first response precise status");
        expect(invoke() == (clean ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_INITIALIZED),
               "only healthy envelope permits follow-up");
        expect(!glimmer_dnn_attn_compute_native_calls(), "hostile receipt no local fallback");
    });
    unsigned int hellos{}, creates{}, handles{}, prepares{}, computes{};
    std::uint64_t metadata{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{8};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded hostile peer progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "authenticated hostile control peer");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            const auto opcode = request->message.header.operation;
            if (opcode == ExecutionOpcode::kHello) {
                ++hellos;
                auto ownership = ExecutionBootstrapOwnership::kRetained;
                if (!decode_execution_context_bootstrap(request->message.payload)) {
                    std::uint32_t role{}, flags{};
                    expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                               input.read_u32(flags) && !flags && input.empty(),
                           "DNM1 bootstrap");
                    metadata = request->connection;
                    ownership = ExecutionBootstrapOwnership::kMetadataRetained;
                }
                const auto encoded = encode_execution_bootstrap_ownership(ownership);
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (opcode == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(input.read_u32(kind) && kind == 14 && input.empty(), "typed Attn creation");
                ++creates;
                reply.id = 17;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else if (opcode == ExecutionOpcode::kCreateDnn) {
                expect(input.empty(), "empty native handle creation");
                ++handles;
                reply.id = 23;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            } else {
                ExecutionDnnAttnCompute value;
                expect(read_execution_dnn_attn_compute(input, value) && input.empty() &&
                           value.handle == 23 && value.metadata == metadata &&
                           value.descriptors == std::array<std::uint64_t, 5>{17, 0, 0, 0, 0},
                       "canonical attention request identities");
                if (opcode == ExecutionOpcode::kPrepareDnnAttn) {
                    ++prepares;
                    expect(preparation && value.lower.empty() && value.upper.empty(),
                           "preparation never transfers unread host bytes");
                    ExecutionPayloadWriter output;
                    expect(write_execution_dnn_attn_window_range(output, {}),
                           "encode zero capture interval");
                    reply.data = std::move(output).take();
                    if (prepares == 1) switch (test) {
                            case 0:
                                reply.id = 1;
                                break;
                            case 1:
                                reply.address = 1;
                                break;
                            case 2:
                                reply.original_status = CUDNN_STATUS_SUCCESS;
                                break;
                            case 3:
                                reply.parameter_bytes = 4;
                                break;
                            case 4:
                                reply.status = 1;
                                break;
                            case 5:
                                reply.data = {std::byte{1}};
                                break;
                            case 6:
                                reply.data.push_back(std::byte{1});
                                break;
                            case 7:
                                reply.parameter_bytes = 1;
                                break;
                            case 8:
                                reply.parameter_bytes = 2;
                                break;
                            case 9:
                                reply.data[0] = std::byte{1};
                                break;
                            case 10:
                                reply.data[4] = std::byte{1};
                                reply.data[0] = std::byte{1};
                                break;
                            case 11:
                                reply.data[7] = std::byte{0x7f};
                                break;
                            case 12:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.data.clear();
                                reply.status = -1;
                                break;
                            case 13:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.data.clear();
                                reply.status = 1024;
                                break;
                            case 14:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.status = CUDA_ERROR_INVALID_VALUE;
                                break;
                            case 15:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.data.clear();
                                reply.status = CUDA_ERROR_NOT_SUPPORTED;
                                break;
                            case 16:
                                break;
                            case 17:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.data.clear();
                                reply.status = CUDA_ERROR_INVALID_VALUE;
                                reply.parameter_bytes = 1;
                                break;
                        }
                } else {
                    expect(opcode == ExecutionOpcode::kDnnAttnCompute,
                           "only actual computation opcode");
                    ++computes;
                    reply.id = 1;
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    if (!preparation && computes == 1) switch (test) {
                            case 0:
                                reply.parameter_bytes = 1;
                                break;
                            case 1:
                                reply.parameter_bytes = 2;
                                break;
                            case 2:
                                reply.parameter_bytes = 3;
                                break;
                            case 3:
                                reply.id = 0;
                                break;
                            case 4:
                                reply.address = 1;
                                break;
                            case 5:
                                reply.data = {std::byte{1}};
                                break;
                            case 6:
                                reply.original_status.reset();
                                break;
                            case 7:
                                reply.status = CUDNN_STATUS_BAD_PARAM;
                                break;
                            case 8:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.status = -1;
                                reply.original_status.reset();
                                break;
                            case 9:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.status = 1024;
                                reply.original_status.reset();
                                break;
                            case 10:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.status = CUDA_ERROR_UNKNOWN;
                                break;
                            case 11:
                                reply.parameter_bytes = 4;
                                break;
                            case 12:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.id = 0;
                                reply.status = CUDA_ERROR_NOT_SUPPORTED;
                                reply.original_status.reset();
                                break;
                            case 13:
                                break;
                            case 14:
                                reply.status = CUDNN_STATUS_BAD_PARAM;
                                reply.original_status = reply.status;
                                break;
                            case 15:
                                reply.disposition = ExecutionWireDisposition::kRejected;
                                reply.status = CUDA_ERROR_INVALID_VALUE;
                                reply.parameter_bytes = 2;
                                break;
                        }
                }
            }
            const auto payload = encode_execution_reply(reply);
            expect(payload && decode_execution_reply(*payload), "wire-valid hostile reply");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *payload}),
                   "deliver authenticated hostile receipt");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 2 && creates == 1 && handles == 1, "exact authority bootstrap counts");
    expect(preparation ? (prepares == (clean ? 2U : 1U) && computes == (test == 16   ? 2U
                                                                        : test == 15 ? 1U
                                                                                     : 0U))
                       : (prepares == 0 && computes == (clean ? 2U : 1U)),
           "no implicit retry or fallback RPC");
    server.stop();
}
void isolated(const char* executable, const char* provider, std::string_view mode) {
    const auto count = mode == "--malformed-replies" ? 34U : mode == "--poison" ? 5U : 6U;
    const auto selected = mode == "--malformed-replies" ? "--malformed-case"
                          : mode == "--poison"          ? "--poison-case"
                                                        : "--missing-case";
    for (unsigned int i = 0; i != count; ++i) {
        const auto number = std::to_string(i);
        const auto child = fork();
        expect(child >= 0, "isolated case fork");
        if (!child) {
            execl("/proc/self/exe", executable, provider, selected, number.c_str(),
                  static_cast<char*>(nullptr));
            _exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error(std::string(selected) + " failed " + number);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--poison" || mode == "--missing-callback")
            isolated(argv[0], argv[1], mode);
        else if (argc == 4) {
            const auto index = static_cast<unsigned int>(std::stoul(argv[3]));
            if (mode == "--malformed-case") {
                expect(index < 34, "known malformed case");
                malformed(argv[1], index);
                _exit(0);
            }
            expect(
                (mode == "--poison-case" && index < 5) || (mode == "--missing-case" && index < 6),
                "known native case");
            run(argv[1], ExecutionSchedulingMode::kNaive, mode == "--missing-case" ? index + 1 : 0,
                mode == "--poison-case" ? static_cast<int>(index) : -1);
        } else {
            expect(mode.empty() || mode == "--profile" || mode == "--priority",
                   "known scheduling mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "Attention public CPU arithmetic and authenticated IPC checks passed\n";
    } catch (const std::exception& error) {
        fixture::control.release.store(true);
        foundation::control.can_stop.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#pragma GCC diagnostic pop
