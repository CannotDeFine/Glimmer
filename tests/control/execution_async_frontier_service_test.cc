#include "runtime.h"
#include "internal/process_identity.h"
#include "../backend/cuda_execution_domain_fixture.h"

#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <string>
#include <thread>

namespace {
using namespace glimmer::app;
using namespace glimmer::control;
using namespace glimmer::backend::cuda;
using namespace glimmer::test::cuda_domain;
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kGuard = 0xC0FFEE00;
std::atomic<bool> error_tail_ready{};
std::atomic<unsigned> native_launches{}, native_metadata{}, native_resources{};
std::atomic<unsigned> checkpoint_requested{}, checkpoint_completed{};

template <class Predicate>
void await(Predicate predicate) {
    const auto deadline = Clock::now() + std::chrono::seconds{10};
    while (!predicate()) {
        expect(Clock::now() < deadline, "bounded independent service progress");
        std::this_thread::yield();
    }
}

std::array<unsigned, 2> observation_checkpoint() {
    const auto generation = checkpoint_requested.fetch_add(1) + 1;
    await([&] { return checkpoint_completed.load() == generation; });
    return {native_metadata.load(), native_resources.load()};
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters): exact cuLaunchKernel callback ABI.
CUresult CUDAAPI observed_launch(CUfunction function, unsigned gx, unsigned gy, unsigned gz,
                                 unsigned bx, unsigned by, unsigned bz, unsigned shared,
                                 CUstream stream, void** arguments, void** extra) {
    // NOLINTEND(bugprone-easily-swappable-parameters)
    expect(extra && *static_cast<const std::size_t*>(extra[3]) == 12,
           "original command retains its exact public ABI");
    std::uint32_t value{};
    std::memcpy(&value, static_cast<const std::byte*>(extra[1]) + 8, sizeof(value));
    if (value == 13) driver.next_launch_status = CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    const auto result = launch(function, gx, gy, gz, bx, by, bz, shared, stream, arguments, extra);
    native_launches = driver.launches;
    return result;
}
CUresult CUDAAPI observed_event_query(CUevent native) {
    const auto* event = reinterpret_cast<FakeEvent*>(native);
    expect(event && event->is_live && event->is_recorded && event->context == driver.current,
           "only actually recorded exact-context physical events may be observed");
    return event->launch == 2 && !error_tail_ready ? CUDA_ERROR_NOT_READY : CUDA_SUCCESS;
}

bool empty_metadata(const ExecutionWireReply& reply) {
    return !reply.original_status && !reply.id && !reply.address && !reply.parameter_bytes &&
           reply.arguments.empty();
}
ExecutionWireReply native_success(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
               reply->status == CUDA_SUCCESS && reply->original_status == CUDA_SUCCESS,
           "requested native operation preserves its exact successful SDK result");
    return std::move(*reply);
}
ExecutionWireReply metadata_success(std::optional<ExecutionWireReply> reply) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
               reply->status == CUDA_SUCCESS && empty_metadata(*reply),
           "batch success contains only the canonical metadata envelope and data");
    return std::move(*reply);
}
void rejected(std::optional<ExecutionWireReply> reply, CUresult status) {
    expect(reply && reply->disposition == ExecutionWireDisposition::kRejected &&
               reply->status == status && empty_metadata(*reply) && reply->data.empty(),
           "batch rejection carries no partial receipt or invented native status");
}

struct LaunchInput {
    std::uint64_t stream{};
    std::uint32_t value{};
};

struct Client {
    ExecutionSocketClient socket;
    std::uint64_t module{}, function{}, allocation{};
    CUdeviceptr address{};
    std::vector<std::uint64_t> streams;
    Client(const std::string& path, TaskPeerIdentity peer) : socket({path, peer, 10000}) {
        native_success(socket.open());
        ExecutionPayloadWriter image;
        expect(image.write_string(kSource), "encode exact original numerical kernel");
        module = native_success(socket.exchange(ExecutionOpcode::kLoadPtx, image.take())).id;
        ExecutionPayloadWriter name;
        expect(name.write_u64(module) && name.write_string("kernel"),
               "encode exact function lookup");
        function = native_success(socket.exchange(ExecutionOpcode::kGetFunction, name.take())).id;
        ExecutionPayloadWriter bytes;
        expect(bytes.write_u64(64), "allocate all numerical values and guards");
        const auto memory =
            native_success(socket.exchange(ExecutionOpcode::kAllocate, bytes.take()));
        allocation = memory.id;
        address = memory.address;
        std::array<std::uint32_t, 16> values{};
        for (std::size_t index = 0; index != values.size(); ++index)
            if (index < 4 || index >= 12) values[index] = kGuard;
        ExecutionPayloadWriter upload;
        expect(upload.write_u64(allocation) && upload.write_u64(0) &&
                   upload.write_blob(std::as_bytes(std::span{values})),
               "initialize complete original tensor and guard bytes");
        native_success(socket.exchange(ExecutionOpcode::kUpload, upload.take()));
    }
    std::uint64_t stream() {
        ExecutionPayloadWriter flags;
        expect(flags.write_u32(CU_STREAM_NON_BLOCKING), "independent logical stream flags");
        const auto id =
            native_success(socket.exchange(ExecutionOpcode::kCreateStream, flags.take())).id;
        streams.push_back(id);
        return id;
    }
    ExecutionWireReply by_id(ExecutionOpcode operation, std::uint64_t id) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(id), "encode exact owned resource identity");
        const auto reply = socket.exchange(operation, input.take());
        expect(reply.has_value(), "exact resource request receives a complete wire reply");
        return *reply;
    }
    ExecutionAsyncFrontierKey admit(LaunchInput launch) {
        std::array<std::byte, 12> parameters{};
        const auto output = address + 16;
        std::memcpy(parameters.data(), &output, 8);
        std::memcpy(parameters.data() + 8, &launch.value, 4);
        ExecutionPayloadWriter input;
        expect(input.write_u64(function) && input.write_u64(launch.stream) && input.write_u32(8) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(32) &&
                   input.write_u32(1) && input.write_u32(1) && input.write_u32(0) &&
                   input.write_blob(parameters) && input.write_u32(kExecutionAsyncLaunchMagic),
               "submit an ordinary copied ASY1 kernel command");
        const auto reply = socket.exchange(ExecutionOpcode::kLaunch, input.take());
        parameters.fill(std::byte{0xA5});
        expect(reply && reply->disposition == ExecutionWireDisposition::kAdmissionAccepted &&
                   !reply->status && reply->id && reply->address && !reply->original_status &&
                   !reply->parameter_bytes && reply->arguments.empty() && reply->data.empty(),
               "reliable admission is distinct from SDK return and physical retirement");
        return {reply->address, reply->id};
    }
    std::vector<ExecutionAsyncFrontierObservation> query(
        std::span<const ExecutionAsyncFrontierKey> keys) {
        const auto payload = encode_execution_async_frontier_batch(keys);
        expect(payload.has_value(), "encode bounded exact asynchronous identities");
        const auto reply =
            metadata_success(socket.exchange(ExecutionOpcode::kQueryAsyncFrontiers, *payload));
        const auto observed = decode_execution_async_frontier_observations(reply.data);
        expect(observed && observed->size() == keys.size(), "complete ordered batch observation");
        for (std::size_t index = 0; index != keys.size(); ++index)
            expect((*observed)[index].key == keys[index],
                   "every observation echoes its exact command identity");
        return *observed;
    }
    void release(std::span<const ExecutionAsyncFrontierKey> keys) {
        const auto payload = encode_execution_async_frontier_batch(keys);
        expect(payload.has_value(), "encode exact bounded retired batch");
        const auto reply =
            metadata_success(socket.exchange(ExecutionOpcode::kReleaseAsyncFrontiers, *payload));
        expect(reply.data == *payload,
               "atomic release echoes every exact retired identity in order");
    }
    ExecutionAsyncFrontierKey capture() {
        ExecutionPayloadWriter input;
        expect(input.write_u32(0) && input.write_u64(0), "encode ordinary non-ASY context capture");
        const auto reply = socket.exchange(ExecutionOpcode::kCaptureFrontier, input.take());
        expect(reply && reply->disposition == ExecutionWireDisposition::kComplete &&
                   !reply->status && reply->id,
               "ordinary capture has its own valid but non-ASY identity");
        return {reply->id, 1};
    }
    void verify(std::uint32_t expected) {
        ExecutionPayloadWriter input;
        expect(input.write_u64(allocation) && input.write_u64(0) && input.write_u64(64),
               "read complete numerical output and guards");
        const auto reply =
            native_success(socket.exchange(ExecutionOpcode::kDownload, input.take()));
        expect(reply.data.size() == 64, "complete readback length");
        std::array<std::uint32_t, 16> values{};
        std::memcpy(values.data(), reply.data.data(), reply.data.size());
        for (std::size_t index = 0; index != values.size(); ++index)
            expect(values[index] == (index >= 4 && index < 12 ? expected : kGuard),
                   "every additive result and surrounding guard preserves exact admitted bytes");
    }
    void free_resources() {
        native_success(by_id(ExecutionOpcode::kRelease, allocation));
        for (const auto id : streams) native_success(by_id(ExecutionOpcode::kDestroyStream, id));
        native_success(by_id(ExecutionOpcode::kUnloadModule, module));
    }
    void close() {
        native_success(socket.exchange(ExecutionOpcode::kClose));
    }
};

void exercise(const std::string& path, TaskPeerIdentity identity) {
    Client client(path, identity);
    const auto first_stream = client.stream(), error_stream = client.stream(),
               held_stream = client.stream();
    const auto complete = client.admit({first_stream, 5});
    await([&] { return client.query(std::span{&complete, 1})[0].is_retired; });
    const auto failed = client.admit({error_stream, 13});
    await([&] { return native_launches.load() == 2; });
    const auto callback = client.by_id(ExecutionOpcode::kEnqueueHostFunction, held_stream);
    expect(callback.disposition == ExecutionWireDisposition::kComplete && !callback.status &&
               callback.id,
           "real CPU boundary independently prevents later physical issue");
    const auto held = client.admit({held_stream, 7});
    const std::array keys{complete, failed, held};
    const std::vector<ExecutionAsyncFrontierObservation> pending{
        {complete, CUDA_SUCCESS, true},
        {failed, CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, false},
        {held, CUDA_ERROR_NOT_READY, false}};
    expect(client.query(keys) == pending,
           "batch keeps pending error distinct from physical completion");
    const auto counters_before = observation_checkpoint();
    for (unsigned iteration = 0; iteration != 16; ++iteration)
        expect(client.query(keys) == pending, "query is repeatable and never erases an error");
    expect(native_launches == 2 && observation_checkpoint() == counters_before,
           "metadata observation neither issues/prepares a kernel nor mutates native resource "
           "ownership");
    const auto rejects = [&](ExecutionOpcode operation,
                             std::span<const ExecutionAsyncFrontierKey> values, CUresult status) {
        const auto payload = encode_execution_async_frontier_batch(values);
        expect(payload.has_value(), "encode semantically invalid but well-formed batch");
        rejected(client.socket.exchange(operation, *payload), status);
    };
    rejects(ExecutionOpcode::kReleaseAsyncFrontiers, keys, CUDA_ERROR_NOT_READY);
    expect(client.query(keys) == pending, "pending final member prevents every earlier release");
    const std::array mismatch{complete,
                              ExecutionAsyncFrontierKey{failed.id, failed.sequence + 100}};
    for (const auto operation :
         {ExecutionOpcode::kQueryAsyncFrontiers, ExecutionOpcode::kReleaseAsyncFrontiers})
        rejects(operation, mismatch, CUDA_ERROR_INVALID_HANDLE);
    expect(client.query(keys) == pending,
           "wrong final command identity cannot delete an earlier valid token");
    const auto ordinary = client.capture();
    for (const auto operation :
         {ExecutionOpcode::kQueryAsyncFrontiers, ExecutionOpcode::kReleaseAsyncFrontiers})
        rejects(operation, std::span{&ordinary, 1}, CUDA_ERROR_INVALID_HANDLE);
    const auto ordinary_release = client.by_id(ExecutionOpcode::kReleaseFrontier, ordinary.id);
    expect(ordinary_release.disposition == ExecutionWireDisposition::kComplete &&
               !ordinary_release.status,
           "batch rejection preserves the original ordinary-capture release contract");
    auto malformed = *encode_execution_async_frontier_batch(std::span{&complete, 1});
    malformed.push_back(std::byte{0});
    rejected(client.socket.exchange(ExecutionOpcode::kQueryAsyncFrontiers, malformed),
             CUDA_ERROR_INVALID_VALUE);
    rejected(client.socket.exchange(ExecutionOpcode::kReleaseAsyncFrontiers, malformed),
             CUDA_ERROR_INVALID_VALUE);
    const std::array duplicates{
        std::array{complete, complete},
        std::array{complete, ExecutionAsyncFrontierKey{failed.id, complete.sequence}},
        std::array{complete, ExecutionAsyncFrontierKey{complete.id, failed.sequence}}};
    for (const auto& duplicated : duplicates) {
        ExecutionPayloadWriter invalid;
        expect(invalid.write_u32(1) && invalid.write_u32(2),
               "encode intentionally hostile duplicate batch framing");
        for (const auto& key : duplicated)
            expect(invalid.write_u64(key.id) && invalid.write_u64(key.sequence),
                   "encode duplicate identity without the validating client codec");
        const auto payload = invalid.take();
        for (const auto operation :
             {ExecutionOpcode::kQueryAsyncFrontiers, ExecutionOpcode::kReleaseAsyncFrontiers})
            rejected(client.socket.exchange(operation, payload), CUDA_ERROR_INVALID_VALUE);
    }
    expect(client.query(keys) == pending, "malformed batches have no partial mutation");
    error_tail_ready = true;
    auto settled = pending;
    settled[1].is_retired = true;
    await([&] { return client.query(keys) == settled; });
    rejects(ExecutionOpcode::kReleaseAsyncFrontiers, keys, CUDA_ERROR_NOT_READY);
    expect(client.query(keys) == settled,
           "a later pending member retains both earlier success and error observations");

    Client foreign(path, identity);
    std::vector<std::uint64_t> foreign_captures;
    for (unsigned index = 0; index != 6; ++index) foreign_captures.push_back(foreign.capture().id);
    const auto other = foreign.admit({foreign.stream(), 3});
    await([&] { return foreign.query(std::span{&other, 1})[0].is_retired; });
    const std::array foreign_final{failed, other};
    for (const auto operation :
         {ExecutionOpcode::kQueryAsyncFrontiers, ExecutionOpcode::kReleaseAsyncFrontiers})
        rejects(operation, foreign_final, CUDA_ERROR_INVALID_HANDLE);
    expect(client.query(keys) == settled,
           "foreign final identity does not release any local token");
    foreign.release(std::span{&other, 1});
    for (const auto id : foreign_captures) {
        const auto reply = foreign.by_id(ExecutionOpcode::kReleaseFrontier, id);
        expect(reply.disposition == ExecutionWireDisposition::kComplete && !reply.status,
               "all foreign ordinary captures remain explicitly owned");
    }
    const std::array retired{failed, complete};
    client.release(retired);
    rejects(ExecutionOpcode::kQueryAsyncFrontiers, retired, CUDA_ERROR_INVALID_HANDLE);
    rejects(ExecutionOpcode::kReleaseAsyncFrontiers, retired, CUDA_ERROR_INVALID_HANDLE);
    expect(client.query(std::span{&held, 1})[0] == settled[2],
           "retired subset release leaves pending ownership untouched");
    const auto permitted = client.by_id(ExecutionOpcode::kQueryHostFunction, callback.id);
    expect(permitted.disposition == ExecutionWireDisposition::kComplete && !permitted.status &&
               permitted.id == callback.id,
           "exact callback permission observes only its independent predecessor");
    ExecutionPayloadWriter returned;
    expect(returned.write_u64(callback.id) && returned.write_u32(1),
           "settle the known returned CPU callback");
    const auto finished =
        client.socket.exchange(ExecutionOpcode::kFinishHostFunction, returned.take());
    expect(finished && finished->disposition == ExecutionWireDisposition::kComplete &&
               !finished->status && finished->id == callback.id,
           "only explicit callback return releases its later kernel");
    await([&] { return client.query(std::span{&held, 1})[0].is_retired; });
    expect(client.query(std::span{&held, 1})[0].status == CUDA_SUCCESS,
           "later exact command succeeds independently of retired error history");
    client.release(std::span{&held, 1});
    client.verify(12);
    foreign.verify(3);
    client.free_resources();
    foreign.free_resources();
    client.close();
    foreign.close();
}
}  // namespace

int main() {
    char pattern[] = "/tmp/glimmer-async-frontier-service-XXXXXX";
    const auto* directory = ::mkdtemp(pattern);
    if (!directory) return EXIT_FAILURE;
    try {
        driver = {};
        driver.requires_explicit_cleanup = true;
        auto dispatch = session_dispatch();
        dispatch.execution.launch_kernel = observed_launch;
        dispatch.execution.event_query = observed_event_query;
        ExecutionServiceConfig config;
        config.transport.socket_path = std::string(directory) + "/service.sock";
        config.transport.max_peers = 2;
        config.session_limits.max_objects = 32;
        config.session_limits.submissions.max_commands = 8;
        config.uses_original_kernels = true;
        config.scheduling_mode = glimmer::core::ExecutionSchedulingMode::kNaive;
        config.exit_when_idle = true;
        const auto path = config.transport.socket_path;
        const auto identity = current_process_identity();
        expect(identity.has_value(), "authenticated exact service process incarnation");
        ExecutionService service(std::move(config), dispatch);
        expect(service.start(),
               "actual service starts without any GPU or caller-native dependency");
        auto client = std::async(std::launch::async, [&] { exercise(path, *identity); });
        const auto deadline = Clock::now() + std::chrono::seconds{25};
        while (client.wait_for(std::chrono::milliseconds{0}) != std::future_status::ready) {
            expect(Clock::now() < deadline && service.step(1),
                   "bounded actual IPC/Core/backend progress");
            const auto checkpoint = checkpoint_requested.load();
            if (checkpoint != checkpoint_completed.load()) {
                native_metadata = driver.parameter_calls + driver.configuration_calls;
                native_resources =
                    driver.allocation_calls + driver.module_load_calls + driver.cleanup_calls();
                checkpoint_completed = checkpoint;
            }
        }
        client.get();
        while (!service.finished())
            expect(Clock::now() < deadline && service.step(0),
                   "normal complete service ownership retirement");
        expect(service.request_stop() && !driver.live_resources() && driver.context_creates == 2 &&
                   driver.context_destroys == 2 && driver.launches == 4 && !driver.current,
               "all exact commands, events, modules, streams, contexts and copied arguments retire "
               "normally");
        expect(::rmdir(directory) == 0, "remove empty owned fixture socket directory");
        std::cout << "Actual asynchronous frontier batch service checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
