#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_op_tensor_fixture.h"

#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <future>
#include <iostream>
#include <source_location>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_op_tensor_native_calls();
namespace {
using namespace glimmer::app;
namespace fixture = glimmer::tests::dnn_op_tensor;
namespace foundation = fixture::driver;
namespace base = fixture::base;
using Clock = std::chrono::steady_clock;
void expect(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void check(cudnnStatus_t status, std::source_location where = std::source_location::current()) {
    if (status != CUDNN_STATUS_SUCCESS)
        throw std::runtime_error("public OpTensor SDK " + std::to_string(status) + " at " +
                                 std::to_string(where.line()));
}
template <typename Function>
void symbol(void* library, const char* name, Function direct) {
    expect(::dlsym(library, name) == reinterpret_cast<void*>(direct) &&
               ::dlsym(RTLD_DEFAULT, name) == reinterpret_cast<void*>(direct),
           "direct/global/explicit provider selects controlled metadata wrapper");
}
struct Cached {
    decltype(&cudnnCreateOpTensorDescriptor) create_op{};
    decltype(&cudnnDestroyOpTensorDescriptor) destroy_op{};
    decltype(&cudnnSetOpTensorDescriptor) set_op{};
    decltype(&cudnnGetOpTensorDescriptor) get_op{};
    decltype(&cudnnCreateTensorTransformDescriptor) create_transform{};
    decltype(&cudnnDestroyTensorTransformDescriptor) destroy_transform{};
    decltype(&cudnnSetTensorTransformDescriptor) set_transform{};
    decltype(&cudnnInitTransformDest) initialize{};
};
Cached lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "load independent OpTensor caller provider");
#define CHECK(NAME) symbol(library, #NAME, &(NAME))
    CHECK(cudnnCreateOpTensorDescriptor);
    CHECK(cudnnDestroyOpTensorDescriptor);
    CHECK(cudnnSetOpTensorDescriptor);
    CHECK(cudnnGetOpTensorDescriptor);
    CHECK(cudnnCreateTensorTransformDescriptor);
    CHECK(cudnnDestroyTensorTransformDescriptor);
    CHECK(cudnnSetTensorTransformDescriptor);
    CHECK(cudnnInitTransformDest);
    CHECK(cudnnGetTensorTransformDescriptor);
    CHECK(cudnnOpTensor);
    CHECK(cudnnTransformTensorEx);
#undef CHECK
    Cached result;
#define CACHE(FIELD, NAME) \
    result.FIELD = reinterpret_cast<decltype(result.FIELD)>(::dlsym(library, #NAME))
    CACHE(create_op, cudnnCreateOpTensorDescriptor);
    CACHE(destroy_op, cudnnDestroyOpTensorDescriptor);
    CACHE(set_op, cudnnSetOpTensorDescriptor);
    CACHE(get_op, cudnnGetOpTensorDescriptor);
    CACHE(create_transform, cudnnCreateTensorTransformDescriptor);
    CACHE(destroy_transform, cudnnDestroyTensorTransformDescriptor);
    CACHE(set_transform, cudnnSetTensorTransformDescriptor);
    CACHE(initialize, cudnnInitTransformDest);
#undef CACHE
    expect(::dlclose(library) == 0, "cached controlled symbols survive explicit provider close");
    return result;
}
void guards(cudnnTensorTransformDescriptor_t conversion) {
    auto format = CUDNN_TENSOR_NHWC;
    auto direction = CUDNN_TRANSFORM_UNFOLD;
    std::array<std::int32_t, 4> before{7, 7, 7, 7}, after{9, 9, 9, 9};
    std::array<std::uint32_t, 2> fold{11, 11};
    expect(
        cudnnGetTensorTransformDescriptor(conversion, 4, &format, before.data(), after.data(),
                                          fold.data(), &direction) == CUDNN_STATUS_NOT_SUPPORTED &&
            format == CUDNN_TENSOR_NHWC && direction == CUDNN_TRANSFORM_UNFOLD &&
            before == std::array<std::int32_t, 4>{7, 7, 7, 7} &&
            after == std::array<std::int32_t, 4>{9, 9, 9, 9} &&
            fold == std::array<std::uint32_t, 2>{11, 11} &&
            cudnnOpTensor(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                          nullptr, nullptr, nullptr) == CUDNN_STATUS_NOT_SUPPORTED &&
            cudnnTransformTensorEx(nullptr, conversion, nullptr, nullptr, nullptr, nullptr, nullptr,
                                   nullptr) == CUDNN_STATUS_NOT_SUPPORTED,
        "three adjacent typed guards neither publish output nor invoke caller-native code");
}
void verify(cudnnTensorDescriptor_t descriptor, const std::array<int, 4>& sizes,
            const std::array<int, 4>& strides) {
    auto type = CUDNN_DATA_FLOAT;
    int dimensions{};
    std::array<int, 8> actual_sizes{}, actual_strides{};
    actual_sizes.fill(997);
    actual_strides.fill(991);
    check(cudnnGetTensorNdDescriptor(descriptor, 8, &type, &dimensions, actual_sizes.data(),
                                     actual_strides.data()));
    expect(type == CUDNN_DATA_DOUBLE && dimensions == 4 &&
               std::equal(sizes.begin(), sizes.end(), actual_sizes.begin()) &&
               std::equal(strides.begin(), strides.end(), actual_strides.begin()) &&
               std::all_of(actual_sizes.begin() + 4, actual_sizes.end(),
                           [](int value) { return value == 997; }) &&
               std::all_of(actual_strides.begin() + 4, actual_strides.end(),
                           [](int value) { return value == 991; }),
           "complete packed descriptor matches independent oracle and leaves query tail untouched");
}
void reciprocal(Cached cached, cudnnTensorTransformDescriptor_t conversion,
                cudnnTensorDescriptor_t source, cudnnTensorDescriptor_t destination) {
    check(cached.set_transform(conversion, 4, CUDNN_TENSOR_NCHW, nullptr, nullptr, nullptr,
                               CUDNN_TRANSFORM_FOLD));
    check(cached.initialize(conversion, source, destination, nullptr));
    fixture::control.release_init.store(false);
    fixture::control.pause_init.store(true);
    auto first = std::async(std::launch::async, [&] {
        return cached.initialize(conversion, source, destination, nullptr);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{3};
    while (!fixture::control.init_entered.load() && Clock::now() < deadline)
        std::this_thread::yield();
    const bool entered = fixture::control.init_entered.load();
    auto second = std::async(std::launch::async, [&] {
        return cached.initialize(conversion, destination, source, nullptr);
    });
    auto query =
        std::async(std::launch::async, [&] { verify(destination, {2, 3, 4, 5}, {60, 20, 5, 1}); });
    const bool pending =
        first.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        second.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready &&
        query.wait_for(std::chrono::milliseconds{20}) != std::future_status::ready;
    fixture::control.release_init.store(true);
    check(first.get());
    check(second.get());
    query.get();
    expect(entered && pending,
           "reciprocal source/destination mutation and read wait without holding conflicting pins");
}
void arrays(Cached cached, cudnnTensorTransformDescriptor_t conversion) {
    const std::array<std::int32_t, 8> before{INT32_MIN, -1, 0, 1, 2, 3, 4, 5};
    const std::array<std::int32_t, 8> after{INT32_MAX, 1, 0, -1, -2, -3, -4, -5};
    const std::array<std::uint32_t, 6> fold{0xffffffffU, 0x80000000U, 0, 1, 2, 3};
    for (unsigned int mask = 0; mask != 8; ++mask)
        check(cached.set_transform(conversion, 8, CUDNN_TENSOR_NHWC,
                                   mask & 1U ? before.data() : nullptr,
                                   mask & 2U ? after.data() : nullptr,
                                   mask & 4U ? fold.data() : nullptr, CUDNN_TRANSFORM_UNFOLD));
    for (unsigned int rank : {0U, 1U, 2U})
        expect(cached.set_transform(conversion, rank, CUDNN_TENSOR_NCHW, nullptr, nullptr, nullptr,
                                    CUDNN_TRANSFORM_FOLD) == CUDNN_STATUS_BAD_PARAM,
               "small bounded ranks preserve native errors rather than frontend clipping");
    // PROT_NONE proves rank overflow rejects without reading any user array.
    auto* inaccessible = ::mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    expect(inaccessible != MAP_FAILED, "allocate unreadable argument guard page");
    const auto calls = fixture::control.callbacks.load();
    const auto status = cached.set_transform(
        conversion, 9, CUDNN_TENSOR_NCHW, static_cast<std::int32_t*>(inaccessible),
        static_cast<std::int32_t*>(inaccessible), static_cast<std::uint32_t*>(inaccessible),
        CUDNN_TRANSFORM_FOLD);
    expect(::munmap(inaccessible, 4096) == 0, "release argument guard page");
    expect(status == CUDNN_STATUS_BAD_PARAM && fixture::control.callbacks.load() == calls,
           "rank overflow does not read arrays or dispatch a native call");
}
void ordinary(const char* provider) {
    const auto cached = lookup(provider);
    cudnnOpTensorDescriptor_t operation{};
    cudnnTensorTransformDescriptor_t conversion{};
    check(cached.create_op(&operation));
    check(cached.create_transform(&conversion));
    guards(conversion);
    auto op = CUDNN_OP_TENSOR_NOT;
    auto type = CUDNN_DATA_DOUBLE;
    auto nan = CUDNN_PROPAGATE_NAN;
    check(cached.get_op(operation, &op, &type, &nan));
    expect(op == CUDNN_OP_TENSOR_ADD && type == CUDNN_DATA_FLOAT && nan == CUDNN_NOT_PROPAGATE_NAN,
           "public metadata creation exposes original defaults without hidden setters");
    check(cached.set_op(operation, CUDNN_OP_TENSOR_MAX, CUDNN_DATA_DOUBLE, CUDNN_PROPAGATE_NAN));
    for (unsigned int mask = 0; mask != 8; ++mask) {
        op = CUDNN_OP_TENSOR_MUL;
        type = CUDNN_DATA_FLOAT;
        nan = CUDNN_NOT_PROPAGATE_NAN;
        check(cached.get_op(operation, mask & 1U ? &op : nullptr, mask & 2U ? &type : nullptr,
                            mask & 4U ? &nan : nullptr));
        expect(op == (mask & 1U ? CUDNN_OP_TENSOR_MAX : CUDNN_OP_TENSOR_MUL) &&
                   type == (mask & 2U ? CUDNN_DATA_DOUBLE : CUDNN_DATA_FLOAT) &&
                   nan == (mask & 4U ? CUDNN_PROPAGATE_NAN : CUDNN_NOT_PROPAGATE_NAN),
               "all optional getter masks preserve exact outputs and absent sentinels");
    }
    fixture::control.get_status.store(CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE);
    op = CUDNN_OP_TENSOR_MUL;
    type = CUDNN_DATA_FLOAT;
    nan = CUDNN_NOT_PROPAGATE_NAN;
    expect(cached.get_op(operation, &op, &type, &nan) == CUDNN_STATUS_NOT_SUPPORTED_DATA_TYPE &&
               op == CUDNN_OP_TENSOR_MUL && type == CUDNN_DATA_FLOAT &&
               nan == CUDNN_NOT_PROPAGATE_NAN,
           "failed native getter does not publish any already-written field");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    fixture::control.invalid_output_mask.store(7);
    check(cached.get_op(operation, nullptr, nullptr, nullptr));
    fixture::control.invalid_output_mask.store(0);
    expect(cached.set_op(operation, static_cast<cudnnOpTensorOp_t>(7), CUDNN_DATA_FLOAT,
                         CUDNN_NOT_PROPAGATE_NAN) == CUDNN_STATUS_BAD_PARAM,
           "representable unsupported operation retains original status");
    check(cached.get_op(operation, &op, &type, &nan));
    arrays(cached, conversion);
    cudnnTensorDescriptor_t source{}, destination{};
    check(cudnnCreateTensorDescriptor(&source));
    check(cudnnCreateTensorDescriptor(&destination));
    check(cudnnSetTensor4dDescriptor(source, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 2, 3, 4, 5));
    const std::array<std::int32_t, 4> before{0, 0, 1, 2}, after{0, 0, 2, 1};
    const std::array<std::uint32_t, 2> fold{1, 1};
    check(cached.set_transform(conversion, 4, CUDNN_TENSOR_NHWC, before.data(), after.data(),
                               fold.data(), CUDNN_TRANSFORM_FOLD));
    std::size_t bytes{};
    check(cached.initialize(conversion, source, destination, &bytes));
    expect(bytes == 2U * 3U * 7U * 8U * 8U, "original complete destination size is returned");
    verify(destination, {2, 3, 7, 8}, {168, 1, 24, 3});
    verify(source, {2, 3, 4, 5}, {60, 20, 5, 1});
    fixture::control.reported_bytes.store(std::numeric_limits<std::size_t>::max() - 7);
    check(cached.initialize(conversion, source, destination, &bytes));
    expect(bytes == std::numeric_limits<std::size_t>::max() - 7, "native size_t retains high bits");
    fixture::control.reported_bytes.store(0);
    check(cached.initialize(conversion, source, destination, nullptr));
    reciprocal(cached, conversion, source, destination);
    check(cached.initialize(conversion, source, source, &bytes));
    expect(bytes == 960, "same-source destination identity initializes without deadlock");
    verify(source, {2, 3, 4, 5}, {60, 20, 5, 1});
    expect(cached.get_op(reinterpret_cast<cudnnOpTensorDescriptor_t>(conversion), nullptr, nullptr,
                         nullptr) == CUDNN_STATUS_NOT_INITIALIZED,
           "opaque wrong-kind handle does not escape owner validation");
    const auto child = ::fork();
    expect(child >= 0, "fork inherited public metadata authority check");
    if (!child)
        ::_exit(cached.get_op(operation, nullptr, nullptr, nullptr) ==
                            CUDNN_STATUS_NOT_INITIALIZED &&
                        cached.initialize(conversion, source, destination, nullptr) ==
                            CUDNN_STATUS_NOT_INITIALIZED
                    ? 0
                    : 1);
    int status{};
    expect(::waitpid(child, &status, 0) == child && WIFEXITED(status) && !WEXITSTATUS(status),
           "fork child cannot reuse inherited metadata connection");
    check(cudnnDestroyTensorDescriptor(destination));
    check(cudnnDestroyTensorDescriptor(source));
    check(cached.destroy_transform(conversion));
    check(cached.destroy_op(operation));
    expect(cached.destroy_op(operation) == CUDNN_STATUS_NOT_INITIALIZED &&
               cached.initialize(conversion, source, destination, nullptr) ==
                   CUDNN_STATUS_NOT_INITIALIZED &&
               !foundation::control.context_creates.load() &&
               glimmer_dnn_op_tensor_native_calls() == 0,
           "stale metadata rejects; all public routes remain context-free with no native fallback");
}
void quarantine(const char* provider) {
    const auto cached = lookup(provider);
    cudnnTensorTransformDescriptor_t conversion{};
    cudnnTensorDescriptor_t source{}, destination{};
    check(cached.create_transform(&conversion));
    check(cudnnCreateTensorDescriptor(&source));
    check(cudnnCreateTensorDescriptor(&destination));
    check(cudnnSetTensor4dDescriptor(source, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 2, 3, 4, 5));
    check(cached.set_transform(conversion, 4, CUDNN_TENSOR_NCHW, nullptr, nullptr, nullptr,
                               CUDNN_TRANSFORM_FOLD));
    fixture::control.init_status.store(CUDNN_STATUS_BAD_PARAM);
    std::size_t bytes = 0xabcdef0123456789ULL;
    expect(cached.initialize(conversion, source, destination, &bytes) == CUDNN_STATUS_BAD_PARAM &&
               bytes == 0xabcdef0123456789ULL,
           "partially mutating Init preserves exact original error without publishing size");
    const auto callbacks = fixture::control.callbacks.load();
    expect(cached.initialize(conversion, source, destination, &bytes) ==
                   CUDNN_STATUS_NOT_INITIALIZED &&
               cudnnDestroyTensorDescriptor(destination) == CUDNN_STATUS_NOT_INITIALIZED &&
               fixture::control.callbacks.load() == callbacks &&
               glimmer_dnn_op_tensor_native_calls() == 0,
           "quarantine rejects further mutation/destruction locally without replay or fallback");
}
void run(const char* provider, bool is_quarantine, bool profile) {
    const auto identity = glimmer::control::current_process_identity();
    expect(identity.has_value(), "actual metadata service process identity");
    ExecutionServiceConfig config;
    config.transport.socket_path =
        "/tmp/glimmer-execution-op-tensor-" + std::to_string(::getpid()) + ".sock";
    config.session_limits.max_objects = 64;
    config.exit_when_idle = true;
    if (profile) {
        config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kProfile;
        config.uses_original_kernels = true;
    }
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.transport.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated original-owned metadata IPC");
    fixture::reset();
    ExecutionService service(std::move(config), fixture::session_dispatch());
    expect(service.start(), "start independent metadata service");
    auto client = std::async(std::launch::async, [=] {
        if (is_quarantine)
            quarantine(provider);
        else
            ordinary(provider);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{30};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline, "bounded public metadata progress");
    client.get();
    if (is_quarantine) {
        expect(fixture::state.initializes == 1 && !fixture::state.destroys &&
                   !base::state.destroys &&
                   base::object(fixture::state.last_destination,
                                glimmer::backend::cuda::CudaDnnObjectKind::kTensor)
                           .tensor.sizes[0] == 777,
               "uncertain destination retains native partial mutation without unsafe cleanup");
        std::cout << "Public OpTensor isolated quarantine checks passed\n" << std::flush;
        ::_exit(0);
    }
    expect(service.request_stop() && service.finished(), "all metadata owners settle");
    expect(fixture::state.creates == 2 && fixture::state.destroys == 2 &&
               base::state.creates == 2 && base::state.destroys == 2 &&
               fixture::state.query_masks == 255 && fixture::state.array_masks == 255 &&
               fixture::state.wide_transform.before ==
                   std::array<std::int32_t, 8>{INT32_MIN, -1, 0, 1, 2, 3, 4, 5} &&
               fixture::state.wide_transform.after ==
                   std::array<std::int32_t, 8>{INT32_MAX, 1, 0, -1, -2, -3, -4, -5} &&
               fixture::state.wide_transform.fold ==
                   std::array<std::uint32_t, 8>{0xffffffffU, 0x80000000U, 0, 1, 2, 3} &&
               fixture::state.wide_transform.direction == CUDNN_TRANSFORM_UNFOLD &&
               fixture::state.initializes == 7 && !foundation::control.context_creates.load() &&
               foundation::driver.contexts.empty() && foundation::driver.memory.empty() &&
               !base::state.handle_creates && !fixture::control.device_uuid_calls.load() &&
               !fixture::control.driver_version_calls.load() &&
               glimmer_dnn_op_tensor_native_calls() == 0,
           "complete optional masks and seven native initializations retire metadata exactly once");
}
// This route uses a one-bit metadata receipt, unlike Folded's two-owner mask.
// Keep a dedicated fake IPC peer so native failure controls do not accidentally
// excuse malformed envelopes, uncertain success or caller-output publication.
struct InitReplyCase {
    glimmer::control::ExecutionWireReply reply;
    cudnnStatus_t status{CUDNN_STATUS_INTERNAL_ERROR};
    std::size_t bytes{0xabcdef0123456789ULL};
    bool has_size{true}, clean{}, disconnect{};
};
constexpr unsigned int kInitReplyCases = 22;
InitReplyCase init_reply_case(unsigned int index) {
    using namespace glimmer::control;
    InitReplyCase test;
    auto& reply = test.reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    switch (index) {
        case 0:
            reply.parameter_bytes = 1;
            reply.address = 123;
            break;
        case 1:
            reply.parameter_bytes = 1;
            test.has_size = false;
            break;
        case 2:
            reply.id = 17;
            reply.address = 123;
            break;
        case 3:
        case 4:
        case 5:
        case 13:
        case 19:
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.parameter_bytes = index == 5 ? 0 : 1;
            if (index == 3) reply.address = 99;
            if (index == 4) reply.data = {std::byte{7}};
            if (index == 19) reply.id = 99;
            if (index == 13) test.status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 6:
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            // A replacement is a valid transport shape only with a nonzero
            // identity and no original status; it is invalid for this route.
            reply.original_status.reset();
            reply.id = 17;
            break;
        case 7:
            reply.original_status.reset();
            break;
        case 8:
            reply.original_status = CUDNN_STATUS_BAD_PARAM;
            break;
        case 9:
        case 10:
        case 11:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = index == 9 ? -1 : index == 10 ? 1024 : CUDA_ERROR_INVALID_CONTEXT;
            if (index == 11)
                reply.parameter_bytes = 1;
            else
                reply.original_status.reset();
            break;
        case 12:
            reply.parameter_bytes = 2;
            break;
        case 14:
            reply.address = SIZE_MAX;
            test.bytes = SIZE_MAX;
            test.status = CUDNN_STATUS_SUCCESS;
            test.clean = true;
            break;
        case 15:
            test.has_size = false;
            test.status = CUDNN_STATUS_SUCCESS;
            test.clean = true;
            break;
        case 16:
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            reply.original_status.reset();
            test.status = CUDNN_STATUS_NOT_SUPPORTED;
            test.clean = true;
            break;
        case 17:
            test.disconnect = true;
            break;
        case 18:
            test.has_size = false;
            reply.address = 7;
            break;
        case 20:
            reply.data = {std::byte{7}};
            break;
        case 21:
            reply.status = 8192;
            reply.original_status = 8192;
            break;
        default:
            throw std::runtime_error("unknown InitTransformDest hostile receipt");
    }
    return test;
}
void malformed_init(const char* provider, unsigned int index) {
    using namespace glimmer::control;
    const auto identity = current_process_identity();
    expect(identity.has_value(), "actual hostile Init endpoint identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-init-hostile-" + std::to_string(::getpid()) + ".sock";
    const auto pid = std::to_string(identity->pid),
               start = std::to_string(identity->start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", config.socket_path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated hostile Init peer");
    const auto test = init_reply_case(index);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start hostile Init metadata endpoint");
    client = std::async(std::launch::async, [=] {
        const auto cached = lookup(provider);
        cudnnTensorTransformDescriptor_t transform{};
        cudnnTensorDescriptor_t source{}, destination{};
        check(cached.create_transform(&transform));
        check(cudnnCreateTensorDescriptor(&source));
        check(cudnnCreateTensorDescriptor(&destination));
        struct Guarded {
            std::uint64_t before{0x123456789abcdef0ULL};
            std::size_t bytes{0xabcdef0123456789ULL};
            std::uint64_t after{0xfedcba9876543210ULL};
        } output;
        expect(cached.initialize(transform, source, destination,
                                 test.has_size ? &output.bytes : nullptr) == test.status,
               "Init native/policy/malformed envelope has exact public status");
        expect(output.bytes == test.bytes && output.before == 0x123456789abcdef0ULL &&
                   output.after == 0xfedcba9876543210ULL,
               "failed or uncertain Init never writes size or its guards");
        const auto follow = cached.initialize(transform, source, destination, &output.bytes);
        expect(follow == (test.clean ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_INITIALIZED) &&
                   output.bytes == (test.clean ? 29 : test.bytes),
               "only a canonical clean receipt permits another remote initialization");
        expect(!glimmer_dnn_op_tensor_native_calls(),
               "hostile Init never falls back to caller SDK");
    });
    unsigned int hellos{}, creates{}, calls{};
    bool disconnected{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline, "bounded hostile Init progress");
        if (disconnected) {
            std::this_thread::yield();
            continue;
        }
        expect(server.poll_for(1), "poll hostile Init metadata peer");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "hostile Init still uses real peer authentication");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            if (request->message.header.operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty() && !hellos++,
                       "Init opens exactly one DNM1 role, never an ordinary CUDA context");
                const auto encoded = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(encoded.has_value(), "encode exact DNM1 retention bootstrap");
                reply.data.assign(encoded->begin(), encoded->end());
            } else if (request->message.header.operation == ExecutionOpcode::kCreateDnnMetadata) {
                std::uint32_t kind{};
                expect(creates < 3 && input.read_u32(kind) && input.empty() &&
                           kind == (creates ? 1U : 10U),
                       "create exact transform/source/destination metadata");
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = ++creates;
            } else {
                std::uint64_t transform{}, source{}, destination{};
                std::uint32_t has_size{};
                expect(
                    request->message.header.operation == ExecutionOpcode::kDnnInitTransformDest &&
                        input.read_u64(transform) && input.read_u64(source) &&
                        input.read_u64(destination) && input.read_u32(has_size) && input.empty() &&
                        transform == 1 && source == 2 && destination == 3 &&
                        has_size == (calls           ? 1U
                                     : test.has_size ? 1U
                                                     : 0U),
                    "Init wire preserves exact three IDs and original nullable output");
                if (++calls == 1) {
                    if (test.disconnect) {
                        server.stop();
                        disconnected = true;
                        continue;
                    }
                    reply = test.reply;
                } else {
                    expect(test.clean && calls == 2, "quarantined Init never retries remotely");
                    reply.original_status = CUDNN_STATUS_SUCCESS;
                    reply.address = 29;
                }
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded.has_value(), "encode hostile Init receipt");
            const auto decoded = decode_execution_reply(*encoded);
            expect(decoded && decoded->disposition == reply.disposition,
                   "hostile envelope reaches route validation through the actual wire codec");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {31, 37};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver hostile Init receipt");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && creates == 3 && calls == (test.clean ? 2U : 1U),
           "no extra connection, hidden setter, retry or native dispatch after quarantine");
    server.stop();
}
void isolated_init(const char* executable, const char* provider) {
    for (unsigned int index = 0; index != kInitReplyCases; ++index) {
        const auto text = std::to_string(index);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated Init receipt");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider, "--malformed-case", text.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("hostile Init case failed: " + text);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider path and optional isolated/profile mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies") {
            expect(argc == 3, "isolated Init parent has no index");
            isolated_init(argv[0], argv[1]);
            return 0;
        }
        if (mode == "--malformed-case") {
            expect(argc == 4, "hostile Init case requires index");
            const std::string_view text{argv[3]};
            expect(!text.empty() && text.size() <= 2, "bounded hostile Init index");
            unsigned int index{};
            for (const char digit : text) {
                expect(digit >= '0' && digit <= '9', "decimal hostile Init index");
                index = 10 * index + static_cast<unsigned int>(digit - '0');
            }
            expect(index < kInitReplyCases, "known hostile Init index");
            malformed_init(argv[1], index);
            ::_exit(0);
        }
        expect(argc <= 3 && (mode.empty() || mode == "--quarantine" || mode == "--profile"),
               "known bounded metadata test mode");
        run(argv[1], mode == "--quarantine", mode == "--profile");
        std::cout << "Public OpTensor/Transform CPU IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        fixture::control.release_init.store(true);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
