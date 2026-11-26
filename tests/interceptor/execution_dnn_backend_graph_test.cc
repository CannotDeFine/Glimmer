#include "execution_dnn_profile_fixture.h"
#include "runtime.h"
#include "internal/process_identity.h"
#include "glimmer/control/execution_dnn_backend_protocol.h"
#include "glimmer/control/execution_dnn_protocol.h"
#include "../backend/dnn_backend_graph_fixture.h"

#include <cerrno>
#include <future>
#include <iostream>
#include <string_view>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/wait.h>

extern "C" unsigned int glimmer_dnn_backend_graph_native_calls();
namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using glimmer::core::ExecutionSchedulingMode;
namespace fixture = glimmer::tests::dnn_backend_graph;
using Clock = std::chrono::steady_clock;
using Native = cudnnBackendDescriptor_t;
constexpr unsigned int kHostileCases = 51;
void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(cudnnStatus_t result) {
    expect(result == CUDNN_STATUS_SUCCESS, "original graph metadata success");
}
struct Api {
    decltype(&cudnnBackendCreateDescriptor) create{};
    decltype(&cudnnBackendDestroyDescriptor) destroy{};
    decltype(&cudnnBackendInitialize) initialize{};
    decltype(&cudnnBackendFinalize) finalize{};
    decltype(&cudnnBackendSetAttribute) set{};
    decltype(&cudnnBackendGetAttribute) get{};
    decltype(&cudnnBackendExecute) execute{};
    decltype(&cudnnBackendPopulateCudaGraph) populate{};
    decltype(&cudnnBackendUpdateCudaGraph) update{};
};
Api lookup(const char* provider) {
    auto* library = ::dlopen(provider, RTLD_NOW | RTLD_LOCAL);
    expect(library, "open independent graph provider");
    Api api;
#define LOOKUP(member, symbol)                                             \
    do {                                                                   \
        auto* value = ::dlsym(library, #symbol);                           \
        expect(value == reinterpret_cast<void*>(&(symbol)) &&              \
                   value == ::dlsym(RTLD_DEFAULT, #symbol),                \
               "direct/global/explicit graph lookup remains intercepted"); \
        api.member = reinterpret_cast<decltype(api.member)>(value);        \
    } while (false)
    LOOKUP(create, cudnnBackendCreateDescriptor);
    LOOKUP(destroy, cudnnBackendDestroyDescriptor);
    LOOKUP(initialize, cudnnBackendInitialize);
    LOOKUP(finalize, cudnnBackendFinalize);
    LOOKUP(set, cudnnBackendSetAttribute);
    LOOKUP(get, cudnnBackendGetAttribute);
    LOOKUP(execute, cudnnBackendExecute);
    LOOKUP(populate, cudnnBackendPopulateCudaGraph);
    LOOKUP(update, cudnnBackendUpdateCudaGraph);
#undef LOOKUP
    expect(::dlclose(library) == 0, "cached graph entry points survive provider close");
    return api;
}
Native create(const Api& api, unsigned int kind) {
    Native value{};
    check(api.create(static_cast<cudnnBackendDescriptorType_t>(kind), &value));
    expect(value, "nonzero owned metadata token");
    return value;
}
cudnnStatus_t set(const Api& api, Native descriptor, unsigned int name, unsigned int type,
                  std::int64_t count, const void* values) {
    return api.set(descriptor, static_cast<cudnnBackendAttributeName_t>(name),
                   static_cast<cudnnBackendAttributeType_t>(type), count, values);
}
cudnnStatus_t get(const Api& api, Native descriptor, unsigned int name, unsigned int type,
                  std::int64_t requested, std::int64_t* count, void* values) {
    return api.get(descriptor, static_cast<cudnnBackendAttributeName_t>(name),
                   static_cast<cudnnBackendAttributeType_t>(type), requested, count, values);
}
void tensor(const Api& api, Native descriptor, std::int64_t uid) {
    const std::array<std::int64_t, 4> dimensions{1, 2, 3, 4}, strides{24, 12, 4, 1};
    const std::int64_t alignment = 16;
    const cudnnDataType_t type = CUDNN_DATA_FLOAT;
    const bool is_virtual = false;
    check(set(api, descriptor, 900, 3, 1, &alignment));
    check(set(api, descriptor, 901, 1, 1, &type));
    check(set(api, descriptor, 902, 3, 4, dimensions.data()));
    check(set(api, descriptor, 903, 3, 4, strides.data()));
    check(set(api, descriptor, 906, 3, 1, &uid));
    check(set(api, descriptor, 907, 2, 1, &is_virtual));
    check(api.finalize(descriptor));
}
class Pages {
   public:
    Pages() {
        const auto size = ::sysconf(_SC_PAGESIZE);
        expect(size > 0, "host page size");
        page = static_cast<std::size_t>(size);
        memory =
            ::mmap(nullptr, page * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        expect(memory != MAP_FAILED, "allocate guarded host storage");
        expect(::mprotect(static_cast<std::byte*>(memory) + page, page, PROT_NONE) == 0,
               "protect host tail");
    }
    Pages(const Pages&) = delete;
    Pages& operator=(const Pages&) = delete;
    ~Pages() {
        if (::munmap(memory, page * 2) != 0) std::terminate();
    }
    void* tail(std::size_t bytes) const {
        return static_cast<std::byte*>(memory) + page - bytes;
    }
    void* memory{};
    std::size_t page{};
};
void wait_flag(const std::atomic<bool>& flag) {
    const auto deadline = Clock::now() + std::chrono::seconds{5};
    while (!flag.load()) {
        expect(Clock::now() < deadline, "bounded metadata handshake");
        std::this_thread::yield();
    }
}
void ordinary(const char* provider) {
    const auto api = lookup(provider);
    expect(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, nullptr) ==
               CUDNN_STATUS_BAD_PARAM,
           "null create output is contained before original allocation");
    for (const unsigned int kind :
         {0U, 1U, 10U, 11U, 12U, 13U, 17U, 18U, 19U, 21U, 22U, 24U, 32U}) {
        const auto object = create(api, kind);
        check(api.initialize(object));
        check(api.destroy(object));
    }
    const auto source = create(api, 17);
    tensor(api, source, 73);
    std::array<std::int64_t, 6> dimensions;
    dimensions.fill(-9);
    std::int64_t count = -7;
    check(get(api, source, 902, 3, 2, &count, dimensions.data()));
    expect(count == 2 && dimensions == std::array<std::int64_t, 6>{1, 2, -9, -9, -9, -9},
           "short array publishes exact prefix, not rank or requested capacity");
    check(get(api, source, 902, 3, INT64_MAX, &count, nullptr));
    expect(count == 4, "null array reports full native rank without capacity allocation");
    Pages pages;
    std::uint64_t reordering = 0xabcdef01ffffffffULL;
    check(get(api, source, 909, 3, 1, &count, &reordering));
    expect(count == 1 && reordering == 0xabcdef0100000000ULL,
           "INT64 reordering query publishes only the native four-byte prefix");
    check(get(api, source, 909, 3, 1, nullptr, pages.tail(4)));
    std::uint32_t reordering_prefix{99};
    std::memcpy(&reordering_prefix, pages.tail(4), 4);
    expect(!reordering_prefix,
           "reordering query never requires inaccessible nominal INT64 padding");
    count = -7;
    reordering = 0xabcdef01ffffffffULL;
    expect(get(api, source, 909, 20, 1, &count, &reordering) ==
                   CUDNN_STATUS_BAD_PARAM_ATTRIBUTE_TYPE &&
               count == -7 && reordering == 0xabcdef01ffffffffULL,
           "declared reordering enum getter retains original rejection without caller writes");
    auto* exact = static_cast<std::int64_t*>(pages.tail(32));
    check(get(api, source, 902, 3, INT64_MAX, nullptr, exact));
    expect(std::equal(exact, exact + 4, std::array<std::int64_t, 4>{1, 2, 3, 4}.begin()),
           "huge capacity writes only audited actual prefix at protected page end");
    std::int64_t alias = -1;
    check(get(api, source, 906, 3, 1, &alias, &alias));
    expect(alias == 1, "native scalar-before-count order preserved for aliased outputs");
    count = -7;
    std::int64_t scalar = -11;
    fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM);
    expect(get(api, source, 906, 3, 1, &count, &scalar) == CUDNN_STATUS_BAD_PARAM && count == -7 &&
               scalar == -11,
           "ordinary native failure leaves every caller output unchanged");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    check(get(api, source, 906, 3, 1, &count, &scalar));
    expect(count == 1 && scalar == 73, "ordinary getter error does not poison owner");

    const auto matmul = create(api, 18);
    const std::uint64_t padding = 0x7ff8123480000000ULL;
    std::memcpy(pages.tail(8), &padding, 8);
    check(set(api, matmul, 1503, 4, 1, pages.tail(8)));
    expect(set(api, matmul, 1503, 4, 1, pages.tail(4)) == CUDNN_STATUS_BAD_PARAM,
           "FLOAT padding setter requires native eight bytes, not nominal four");
    check(api.finalize(matmul));
    std::uint64_t observed{};
    count = -1;
    check(get(api, matmul, 1503, 4, 1, &count, &observed));
    expect(observed == padding && count == 1,
           "FLOAT padding carries all eight bits-preserving bytes with native count one");
    std::uint32_t untouched = 0xa5a5a5a5;
    std::memcpy(pages.tail(4), &untouched, 4);
    count = -7;
    expect(get(api, matmul, 1503, 4, 1, &count, pages.tail(4)) == CUDNN_STATUS_BAD_PARAM &&
               count == -7 && std::memcmp(pages.tail(4), &untouched, 4) == 0,
           "all actual scalar/count spans validate before first publication");
    check(api.destroy(matmul));

    const auto matmul_operation = create(api, 19);
    const std::int64_t offset = 0x1234567811223344LL;
    check(set(api, matmul_operation, 1524, 3, 1, &offset));
    check(api.finalize(matmul_operation));
    std::uint64_t partial = 0xabcdef0100000000ULL;
    count = -7;
    check(get(api, matmul_operation, 1524, 3, 1, &count, &partial));
    expect(partial == 0xabcdef0111223344ULL && count == 1,
           "INT64 Matmul operation output preserves native untouched upper four bytes");
    check(get(api, matmul_operation, 1524, 3, 1, nullptr, pages.tail(4)));
    std::uint32_t prefix{};
    std::memcpy(&prefix, pages.tail(4), 4);
    expect(prefix == 0x11223344, "partial scalar output does not require inaccessible padding");
    check(api.destroy(matmul_operation));

    const auto pointwise = create(api, 0);
    for (const std::int64_t invalid_count : {0, -1}) {
        expect(set(api, pointwise, 0, 14, invalid_count, pages.tail(0)) == CUDNN_STATUS_BAD_PARAM,
               "Pointwise count rejection precedes any inaccessible caller input read");
        check(api.initialize(pointwise));
    }
    check(api.destroy(pointwise));

    const auto reduction = create(api, 21), operation = create(api, 22),
               destination = create(api, 17);
    const cudnnReduceTensorOp_t sum = CUDNN_REDUCE_TENSOR_ADD;
    const cudnnDataType_t type = CUDNN_DATA_FLOAT;
    check(set(api, reduction, 1600, 18, 1, &sum));
    check(set(api, reduction, 1601, 1, 1, &type));
    check(api.finalize(reduction));
    check(set(api, operation, 1610, 15, 1, &source));
    check(set(api, operation, 1611, 15, 1, &source));
    check(set(api, operation, 1612, 15, 1, &reduction));
    check(api.finalize(operation));
    check(api.initialize(source));
    tensor(api, source, 99);
    check(api.destroy(source));
    check(api.destroy(reduction));
    Native output = destination;
    count = -1;
    check(get(api, operation, 1610, 15, 1, &count, &output));
    expect(output == destination && count == 1,
           "descriptor Get mutates precreated token, never replaces it");
    scalar = -1;
    check(get(api, destination, 906, 3, 1, &count, &scalar));
    expect(scalar == 73, "copied child survives source reset, reconfiguration and destruction");
    fixture::control.get_status.store(CUDNN_STATUS_BAD_PARAM);
    count = -7;
    expect(get(api, operation, 1610, 15, 1, &count, &output) == CUDNN_STATUS_BAD_PARAM &&
               output == destination && count == -7,
           "partially mutating native destination error preserves public handles and count");
    fixture::control.get_status.store(CUDNN_STATUS_SUCCESS);
    check(api.initialize(destination));
    check(get(api, operation, 1610, 15, 1, &count, &output));
    check(api.destroy(operation));
    check(get(api, destination, 906, 3, 1, &count, &scalar));
    expect(scalar == 73, "copied destination survives parent destruction");
    check(api.initialize(destination));
    fixture::control.set_status.store(CUDNN_STATUS_BAD_PARAM);
    expect(set(api, destination, 906, 3, 1, &scalar) == CUDNN_STATUS_BAD_PARAM,
           "partially committed setter preserves original error");
    fixture::control.set_status.store(CUDNN_STATUS_SUCCESS);
    check(api.initialize(destination));
    tensor(api, destination, 88);
    fixture::control.finalize_status.store(CUDNN_STATUS_BAD_PARAM);
    expect(api.finalize(destination) == CUDNN_STATUS_BAD_PARAM,
           "native Finalize failure remains recoverable");
    fixture::control.finalize_status.store(CUDNN_STATUS_SUCCESS);
    check(api.initialize(destination));
    tensor(api, destination, 77);

    fixture::control.pause.store(true);
    fixture::control.entered.store(false);
    fixture::control.release.store(false);
    auto pending = std::async(std::launch::async, [&] {
        std::int64_t value{}, length{};
        return get(api, destination, 906, 3, 1, &length, &value);
    });
    wait_flag(fixture::control.entered);
    std::atomic<bool> started{};
    auto resetting = std::async(std::launch::async, [&] {
        started.store(true);
        return api.initialize(destination);
    });
    wait_flag(started);
    expect(resetting.wait_for(std::chrono::milliseconds{20}) == std::future_status::timeout,
           "in-flight source observation excludes a concurrent reset");
    fixture::control.release.store(true);
    // A reset may win after preparation releases its immutable pin; final
    // generation validation must either complete the Get or reject it cleanly.
    const auto pending_status = pending.get();
    expect(pending_status == CUDNN_STATUS_SUCCESS || pending_status == CUDNN_STATUS_BAD_PARAM,
           "generation race cannot issue using a stale semantic observation");
    check(resetting.get());
    check(api.destroy(destination));
    const auto callbacks = fixture::control.callbacks.load();
    expect(api.destroy(destination) == CUDNN_STATUS_NOT_INITIALIZED &&
               api.initialize(source) == CUDNN_STATUS_NOT_INITIALIZED &&
               callbacks == fixture::control.callbacks.load(),
           "stale tokens do not reach original callbacks");
    expect(api.execute(nullptr, destination, source) == CUDNN_STATUS_NOT_INITIALIZED &&
               api.populate(nullptr, destination, source, nullptr) == CUDNN_STATUS_NOT_SUPPORTED &&
               api.update(nullptr, destination, source, nullptr) == CUDNN_STATUS_NOT_SUPPORTED,
           "Execute rejects invalid authority; the two CUDA graph routes remain guards");
    expect(!glimmer_dnn_backend_graph_native_calls(), "no caller-native graph fallback");
}

void endpoint(const std::string& path, TaskPeerIdentity identity) {
    const auto pid = std::to_string(identity.pid),
               start = std::to_string(identity.start_time_ticks);
    expect(::setenv("GLIMMER_EXECUTION_SOCKET", path.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_PID", pid.c_str(), 1) == 0 &&
               ::setenv("GLIMMER_EXECUTION_SERVER_START", start.c_str(), 1) == 0,
           "configure authenticated DNM1 endpoint");
}
void missing(const char* provider, unsigned int index) {
    const auto api = lookup(provider);
    if (index <= 3) {
        auto* output = reinterpret_cast<Native>(0xfee);
        expect(api.create(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &output) == CUDNN_STATUS_NOT_SUPPORTED &&
                   output == reinterpret_cast<Native>(0xfee),
               "missing creation or rollback capability never publishes or falls back");
        return;
    }
    const auto object = create(api, 17);
    std::int64_t value = 7, count = -1;
    cudnnStatus_t status{};
    if (index == 4) status = api.initialize(object);
    if (index == 5) status = api.finalize(object);
    if (index == 6) status = set(api, object, 906, 3, 1, &value);
    if (index == 7) {
        tensor(api, object, 7);
        status = get(api, object, 906, 3, 1, &count, &value);
        expect(count == -1 && value == 7, "missing getter leaves outputs untouched");
    }
    expect(status == CUDNN_STATUS_NOT_SUPPORTED,
           "missing original callback is a clean policy rejection");
    check(api.destroy(object));
}
void poisoned(const char* provider, unsigned int index) {
    const auto api = lookup(provider);
    const auto object = create(api, 17);
    tensor(api, object, 7);
    if (index == 0)
        fixture::control.corrupt_count.store(true);
    else
        fixture::control.corrupt_enum.store(true);
    std::int64_t count = -9;
    std::uint32_t type = 0x12345678;
    expect(get(api, object, 901, 1, 1, &count, &type) == CUDNN_STATUS_INTERNAL_ERROR &&
               count == -9 && type == 0x12345678,
           "malformed original getter success cannot publish private count or enum");
    const auto callbacks = fixture::control.callbacks.load();
    expect(api.initialize(object) == CUDNN_STATUS_NOT_INITIALIZED &&
               callbacks == fixture::control.callbacks.load(),
           "uncertain original success cannot be retried through the public owner");
}
struct RunOptions {
    unsigned int missing_index{};
    int poison_index{-1};
};
void run(const char* provider, ExecutionSchedulingMode mode, RunOptions options = {}) {
    const auto identity = current_process_identity();
    expect(identity.has_value(), "real service identity");
    ExecutionServiceConfig config;
    config.transport.socket_path = "/tmp/glimmer-backend-" + std::to_string(::getpid()) + ".sock";
    config.scheduling_mode = mode;
    config.uses_original_kernels = true;
    config.exit_when_idle = true;
    if (mode == ExecutionSchedulingMode::kPriority) config.client_priorities = {{*identity, 7}};
    endpoint(config.transport.socket_path, *identity);
    fixture::reset();
    std::future<void> client;
    ExecutionService service(std::move(config), fixture::session_dispatch(options.missing_index));
    expect(service.start(), "start genuine host-only backend metadata service");
    client = std::async(std::launch::async, [=] {
        if (options.missing_index)
            missing(provider, options.missing_index);
        else if (options.poison_index >= 0)
            poisoned(provider, static_cast<unsigned int>(options.poison_index));
        else
            ordinary(provider);
    });
    const auto deadline = Clock::now() + std::chrono::seconds{20};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready)
        expect(service.step(1) && Clock::now() < deadline,
               "bounded metadata-only service progress");
    client.get();
    expect(fixture::foundation::is_resource_free() && !glimmer_dnn_backend_graph_native_calls(),
           "no Driver/context/compute or caller-native graph calls");
    if (options.poison_index >= 0) ::_exit(0);
    expect(service.request_stop() && service.finished() && !fixture::alive() &&
               fixture::state.creates == fixture::state.destroys,
           "every proven host descriptor destroys once");
}

enum class Stage { kPreparation, kGet, kSet, kCreate, kLifecycle };
struct Hostile {
    Stage stage{Stage::kGet};
    ExecutionWireReply reply;
    cudnnStatus_t expected{CUDNN_STATUS_INTERNAL_ERROR};
    bool healthy{};
};
ExecutionWireReply good_get() {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    reply.original_status = CUDNN_STATUS_SUCCESS;
    ExecutionPayloadWriter output;
    expect(write_execution_dnn_backend_get_result(
               output, {8, 1, {3, fixture::scalar(std::int64_t{73}), {}}}),
           "encode independent scalar response");
    reply.data = output.take();
    return reply;
}
ExecutionWireReply good_preparation() {
    ExecutionWireReply reply;
    reply.disposition = ExecutionWireDisposition::kComplete;
    ExecutionPayloadWriter output;
    expect(write_execution_dnn_backend_preparation(
               output, {9, ExecutionDnnBackendAttributeDirection::kScalarOutput, 0, 1, 8, 8}),
           "encode canonical private preparation");
    reply.data = output.take();
    return reply;
}
Hostile hostile(unsigned int index) {
    Hostile result;
    if (index >= 48) {
        if (index == 50) {
            result.stage = Stage::kPreparation;
            result.reply = good_preparation();
            result.reply.data[16] = std::byte{2};
        } else {
            result.reply = good_get();
            if (index == 48)
                result.reply.data[24] = std::byte{1};
            else {
                result.expected = CUDNN_STATUS_SUCCESS;
                result.healthy = true;
            }
        }
        return result;
    }
    result.stage = index < 12   ? Stage::kPreparation
                   : index < 32 ? Stage::kGet
                   : index < 38 ? Stage::kSet
                   : index < 43 ? Stage::kCreate
                                : Stage::kLifecycle;
    auto& reply = result.reply;
    reply = result.stage == Stage::kPreparation ? good_preparation() : good_get();
    if (result.stage != Stage::kPreparation && result.stage != Stage::kGet) reply.data.clear();
    if (result.stage == Stage::kCreate) reply.id = 77;
    const auto variant = index < 12   ? index
                         : index < 32 ? index - 12
                         : index < 38 ? index - 32
                         : index < 43 ? index - 38
                                      : index - 43;
    if (variant == 0) reply.parameter_bytes = 1;
    if (variant == 1) reply.address = 7;
    if (variant == 2) reply.id = result.stage == Stage::kCreate ? 0 : 7;
    if (variant == 3) {
        reply.parameter_bytes = 1;
        reply.arguments = {{0, 1}};
    }
    if (variant == 4) reply.parameter_bytes = 2;
    if (result.stage == Stage::kPreparation) {
        if (variant == 5) reply.original_status = CUDNN_STATUS_SUCCESS;
        if (variant == 6) reply.data.clear();
        if (variant == 7) reply.data.push_back(std::byte{1});
        if (variant == 8)
            reply.data[0] = std::byte{0};  // Zero generation, with remaining bytes zero.
        if (variant == 9)
            reply.data[20] = std::byte{1};  // Different count-byte semantics are allowed;
        // For this case use an impossible input prefix for scalar output instead.
        if (variant == 9) reply.data[12] = std::byte{1};
        if (variant == 10) {
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_NOT_SUPPORTED;
            reply.data.clear();
            result.expected = CUDNN_STATUS_NOT_SUPPORTED;
            result.healthy = true;
        }
        if (variant == 11) {
            result.expected = CUDNN_STATUS_SUCCESS;
            result.healthy = true;
        }
    } else if (result.stage == Stage::kGet) {
        if (variant == 5) reply.original_status.reset();
        if (variant == 6) reply.status = CUDNN_STATUS_BAD_PARAM;
        if (variant == 7) reply.data.clear();
        if (variant == 8) reply.data.push_back(std::byte{1});
        if (variant == 9) reply.data[0] = std::byte{1};
        if (variant == 10)
            reply.data[12] = std::byte{5};  // Wrong returned type (still wire-valid).
        if (variant == 11) reply.data.resize(reply.data.size() - 1);
        if (variant == 12) {
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.status = CUDA_ERROR_INVALID_VALUE;
            reply.data.clear();  // Original status on a rejection remains contradictory.
        }
        if (variant == 13 || variant == 14) {
            reply.status = CUDNN_STATUS_BAD_PARAM;
            reply.original_status = reply.status;
            reply.data.clear();
            result.expected = CUDNN_STATUS_BAD_PARAM;
            result.healthy = variant == 13;
            if (variant == 14) reply.parameter_bytes = 1;
        }
        if (variant == 15 || variant == 16) {
            reply.disposition = ExecutionWireDisposition::kRejected;
            reply.original_status.reset();
            reply.data.clear();
            reply.status = variant == 15 ? CUDA_ERROR_NOT_SUPPORTED : 0;
            if (variant == 15) {
                result.healthy = true;
                result.expected = CUDNN_STATUS_NOT_SUPPORTED;
            }
        }
        if (variant == 17) {
            reply.disposition = ExecutionWireDisposition::kComputeReplacement;
            reply.original_status.reset();
            reply.id = 1;
        }
        if (variant == 18) reply.original_status = 8192;
        if (variant == 19) {
            result.healthy = true;
            result.expected = CUDNN_STATUS_SUCCESS;
        }
    } else if (result.stage == Stage::kSet && variant == 5) {
        reply.data = {std::byte{1}};
    }
    return result;
}
void hostile_case(const char* provider, unsigned int index) {
    const auto test = hostile(index);
    const auto identity = current_process_identity();
    expect(identity.has_value(), "hostile authenticated identity");
    ExecutionSocketConfig config;
    config.socket_path = "/tmp/glimmer-backend-hostile-" + std::to_string(::getpid()) + ".sock";
    endpoint(config.socket_path, *identity);
    std::future<void> client;
    ExecutionSocketServer server(std::move(config));
    expect(server.start(), "start real hostile graph transport");
    client = std::async(std::launch::async, [=] {
        const auto api = lookup(provider);
        const auto object = create(api, 17);
        std::int64_t count = -9, value = -11;
        Native created = reinterpret_cast<Native>(0xfee);
        cudnnStatus_t status{};
        if (test.stage == Stage::kCreate)
            status = api.create(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &created);
        else if (test.stage == Stage::kLifecycle)
            status = api.initialize(object);
        else if (test.stage == Stage::kSet)
            status = set(api, object, 906, 3, 1, &value);
        else
            status = get(api, object, 906, 3, 1, &count, &value);
        expect(status == test.expected, "hostile receipt exact containment status");
        if (status == CUDNN_STATUS_SUCCESS)
            expect(std::bit_cast<std::uint64_t>(value) ==
                           (index == 49 ? UINT64_C(0xffffffff00000049) : UINT64_C(73)) &&
                       count == 1,
                   "healthy control publishes only exact scalar byte prefix");
        else
            expect(value == -11 && count == -9 && created == reinterpret_cast<Native>(0xfee),
                   "failed or contradictory receipts publish no caller outputs");
        expect(api.initialize(object) ==
                   (test.healthy ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_NOT_INITIALIZED),
               "only clean native/policy receipts allow continued owner use");
        expect(!glimmer_dnn_backend_graph_native_calls(),
               "hostile path never calls provider native");
    });
    unsigned int hellos{}, creates{}, prepares{}, finals{}, followups{};
    bool injected{};
    glimmer::tests::ExecutionDnnProfileFixture profiles;
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
        expect(Clock::now() < deadline && server.poll_for(1), "bounded hostile graph progress");
        if (auto request = server.take_request()) {
            expect(request->peer == *identity, "authenticated hostile graph peer");
            if (profiles.respond(server, *request)) continue;
            ExecutionPayloadReader input(request->message.payload);
            ExecutionWireReply reply;
            reply.disposition = ExecutionWireDisposition::kComplete;
            const auto operation = request->message.header.operation;
            bool target{};
            if (operation == ExecutionOpcode::kHello) {
                std::uint32_t role{}, flags{};
                expect(input.read_u32(role) && role == kExecutionDnnMetadataRole &&
                           input.read_u32(flags) && !flags && input.empty(),
                       "exact DNM1 Hello");
                ++hellos;
                const auto ownership = encode_execution_bootstrap_ownership(
                    ExecutionBootstrapOwnership::kMetadataRetained);
                expect(ownership.has_value(), "metadata-only bootstrap receipt");
                reply.data.assign(ownership->begin(), ownership->end());
            } else if (operation == ExecutionOpcode::kDnnBackendCreate) {
                ExecutionDnnBackendCreate wire;
                expect(read_execution_dnn_backend_create(input, wire) && input.empty() &&
                           wire.kind == 17 && wire.has_output == 1,
                       "canonical Create wire");
                target = creates++ != 0;
                reply.original_status = CUDNN_STATUS_SUCCESS;
                reply.id = 17;
            } else if (operation == ExecutionOpcode::kDnnBackendPrepare) {
                ExecutionDnnBackendPrepare wire;
                expect(read_execution_dnn_backend_prepare(input, wire) && input.empty() &&
                           wire.header.descriptor == 17 && wire.header.name == 906 &&
                           wire.header.type == 3 && wire.header.count == 1 &&
                           wire.header.presence == (test.stage == Stage::kSet ? 1U : 3U) &&
                           wire.operation == (test.stage == Stage::kSet
                                                  ? ExecutionDnnBackendAttributeOperation::kSet
                                                  : ExecutionDnnBackendAttributeOperation::kGet),
                       "exact private Prepare wire");
                ++prepares;
                reply = good_preparation();
                if (index == 48 || index == 49) reply.data[24] = std::byte{4};
                if (test.stage == Stage::kSet) {
                    ExecutionPayloadWriter output;
                    expect(
                        write_execution_dnn_backend_preparation(
                            output, {9, ExecutionDnnBackendAttributeDirection::kInput, 1, 0, 0, 0}),
                        "Set preparation");
                    reply.data = output.take();
                }
                target = test.stage == Stage::kPreparation;
            } else if (operation == ExecutionOpcode::kDnnBackendGet ||
                       operation == ExecutionOpcode::kDnnBackendSet) {
                ExecutionDnnBackendAttribute wire;
                expect(read_execution_dnn_backend_attribute(input, wire) && input.empty() &&
                           wire.header.descriptor == 17 && wire.header.name == 906 &&
                           wire.header.type == 3 && wire.generation == 9 &&
                           wire.values.references.empty(),
                       "exact final generation and value transport");
                expect(operation == ExecutionOpcode::kDnnBackendSet
                           ? wire.values.scalars == fixture::scalar(std::int64_t{-11})
                           : wire.values.scalars.empty(),
                       "final scalar input bytes are exact");
                ++finals;
                reply = good_get();
                target = test.stage == Stage::kGet || test.stage == Stage::kSet;
            } else {
                std::uint64_t id{};
                expect(operation == ExecutionOpcode::kDnnBackendInitialize && input.read_u64(id) &&
                           id == 17 && input.empty(),
                       "exact lifecycle ID");
                target = test.stage == Stage::kLifecycle && !injected;
                if (!target) ++followups;
                reply.original_status = CUDNN_STATUS_SUCCESS;
            }
            if (target && !injected) {
                reply = test.reply;
                injected = true;
            }
            const auto encoded = encode_execution_reply(reply);
            expect(encoded && decode_execution_reply(*encoded),
                   "hostile envelope is wire-valid and reaches route checks");
            auto header = request->message.header;
            header.kind = ExecutionMessageKind::kReply;
            header.session = request->connection;
            header.incarnation = {71, 93};
            expect(server.reply(request->connection, {header, *encoded}),
                   "deliver hostile graph reply");
        }
    }
    client.get();
    profiles.verify();
    expect(hellos == 1 && creates == (test.stage == Stage::kCreate ? 2U : 1U) && injected &&
               followups == (test.healthy ? 1U : 0U),
           "no hidden recovery or cleanup after poisoned receipt");
    expect(
        prepares == (test.stage == Stage::kCreate || test.stage == Stage::kLifecycle ? 0U : 1U) &&
            finals ==
                (test.stage == Stage::kGet || test.stage == Stage::kSet || index == 11 ? 1U : 0U),
        "preparation failures never issue original metadata call");
    server.stop();
}
unsigned int number(const char* value) {
    unsigned int result{};
    const std::string_view text{value};
    expect(!text.empty() && text.size() <= 2, "bounded child number");
    for (const char digit : text) {
        expect(digit >= '0' && digit <= '9', "decimal child number");
        result = result * 10 + static_cast<unsigned int>(digit - '0');
    }
    return result;
}
void children(const char* executable, const char* provider, std::string_view mode) {
    const unsigned int count = mode == "--malformed-replies"  ? kHostileCases
                               : mode == "--missing-callback" ? 7
                                                              : 2;
    const char* option = mode == "--malformed-replies"  ? "--malformed-case"
                         : mode == "--missing-callback" ? "--missing-case"
                                                        : "--poison-case";
    for (unsigned int index = 0; index != count; ++index) {
        const auto value = std::to_string(index);
        const auto child = ::fork();
        expect(child >= 0, "fork isolated backend metadata case");
        if (!child) {
            ::execl("/proc/self/exe", executable, provider, option, value.c_str(),
                    static_cast<char*>(nullptr));
            ::_exit(127);
        }
        int status{};
        pid_t waited{};
        do {
            waited = ::waitpid(child, &status, 0);
        } while (waited < 0 && errno == EINTR);
        if (waited != child || !WIFEXITED(status) || WEXITSTATUS(status))
            throw std::runtime_error("isolated backend metadata case failed: " + std::string{mode} +
                                     " " + value);
    }
}
}  // namespace
int main(int argc, char** argv) {
    try {
        expect(argc >= 2 && argc <= 4, "provider and optional backend graph mode");
        const std::string_view mode = argc >= 3 ? argv[2] : "";
        if (mode == "--malformed-replies" || mode == "--missing-callback" || mode == "--poison") {
            expect(argc == 3, "aggregate graph mode");
            children(argv[0], argv[1], mode);
        } else if (mode == "--malformed-case" || mode == "--missing-case" ||
                   mode == "--poison-case") {
            expect(argc == 4, "isolated graph case");
            const auto index = number(argv[3]);
            if (mode == "--malformed-case") {
                expect(index < kHostileCases, "known hostile case");
                hostile_case(argv[1], index);
                ::_exit(0);
            }
            expect(index < (mode == "--missing-case" ? 7U : 2U), "known fixture failure case");
            run(argv[1], ExecutionSchedulingMode::kNaive,
                {.missing_index = mode == "--missing-case" ? index + 1 : 0,
                 .poison_index = mode == "--poison-case" ? static_cast<int>(index) : -1});
        } else {
            expect(argc <= 3 && (mode.empty() || mode == "--profile" || mode == "--priority"),
                   "known scheduling mode");
            run(argv[1], mode == "--profile"    ? ExecutionSchedulingMode::kProfile
                         : mode == "--priority" ? ExecutionSchedulingMode::kPriority
                                                : ExecutionSchedulingMode::kNaive);
        }
        std::cout << "Backend graph original metadata CPU/local-IPC checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
